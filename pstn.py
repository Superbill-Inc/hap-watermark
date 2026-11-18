"""PSTN watermark demo over a real Telnyx call.

    mic -> watermark -> 8 kHz PCMU --(Telnyx call: FROM -> TO)--> PCMU -> 16 kHz -> extract -> playback

1. Runs a local webhook + WebSocket server on --port (exposed by ngrok as --public-url).
2. Places a call FROM -> TO with a bidirectional media stream (the TX leg). Watermarked
   PCMU packets are sent into the call over that WebSocket, one 20 ms packet at a time.
3. When the incoming call reaches TO, the webhook answers it with a media stream
   (the RX leg). The received PCMU is decoded, the payload is extracted, and the audio
   is played on the local output device.

Both numbers must be on the Call Control app (--connection-id) whose webhook is
--public-url + --webhook-path.

Needs TELNYX_API_KEY (and TELNYX_PUBLIC_KEY for webhook signature checks) in the environment.

    uv run python pstn.py --payload 42351
    uv run python pstn.py --file original.wav --duration 30
"""

import argparse
import asyncio
import logging
import queue
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

warnings.filterwarnings("ignore")
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)  # recompile-limit noise

from audioseal import AudioSeal  # noqa: E402

from main import NBITS, SR, bits_to_int, int_to_bits, load_audio, pcmu_decode, pcmu_encode  # noqa: E402
from stream import FRAME, StreamingWatermarker, device_arg, tensor  # noqa: E402
from telnyx_call import (  # noqa: E402
    PACKET, PCMU_SR, TelnyxCall, add_call_args, check_call_env, log,
)



class Demo(TelnyxCall):
    """AudioSeal over the call: mic/file -> watermark -> PCMU -> call -> extract -> playback."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.block = int(SR * 0.04)
        print("Loading models...", flush=True)
        self.detector = AudioSeal.load_detector("audioseal_detector_16bits")
        self.wm = StreamingWatermarker(
            AudioSeal.load_generator("audioseal_wm_16bits"), args.payload, self.block,
            lookahead=int(SR * 0.04), context=int(SR * 0.5),
        )
        with torch.no_grad():  # warm up
            z = tensor(np.random.default_rng(0).normal(0, 0.01, SR).astype(np.float32))
            self.wm.g.get_watermark(z, sample_rate=SR, message=self.wm.msg)
            self.detector.detect_watermark(z, sample_rate=SR)
        # TX: source -> watermark -> 8 kHz -> PCMU packets
        self.down = soxr.ResampleStream(SR, PCMU_SR, 1, dtype="float32")
        self.src_q: queue.Queue = queue.Queue()
        self.tx_pending = b""
        # RX: PCMU packets -> 16 kHz -> detector ring + playback
        self.up = soxr.ResampleStream(PCMU_SR, SR, 1, dtype="float32")
        self.window = int(SR * args.window_s)
        self.rx_ring = np.zeros(self.window + 2 * FRAME, np.float32)
        self.rx_total = 0
        self.rx_lock = threading.Lock()
        self.play_q: queue.Queue = queue.Queue()
        self.play_pending = np.zeros(0, np.float32)
        self.playing = False
        self.src_log: list[np.ndarray] = []
        self.expected = int_to_bits(args.payload).flatten()
        self.stats.update(windows=0, detected=0, correct=0, underruns=0)

    def on_rx_audio(self, data: bytes):
        pcm16 = self.up.resample_chunk(pcmu_decode(np.frombuffer(data, np.uint8)))
        with self.rx_lock:
            self.rx_ring = np.concatenate([self.rx_ring, pcm16])[-len(self.rx_ring):]
            self.rx_total += len(pcm16)
        self.play_pending = np.concatenate([self.play_pending, pcm16])
        while len(self.play_pending) >= self.block:
            self.play_q.put(self.play_pending[: self.block])
            self.play_pending = self.play_pending[self.block:]

    # --- threads: source processing, detector, status -----------------------------------
    def tx_process_loop(self):
        while not self.stop.is_set():
            try:
                x = self.src_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if not self.tx_active.is_set():
                continue  # nobody to send to yet
            self.src_log.append(x)
            marked = self.wm.process(x)
            self.tx_pending += pcmu_encode(self.down.resample_chunk(marked)).tobytes()
            while len(self.tx_pending) >= PACKET:
                self.tx_q.put(self.tx_pending[:PACKET])
                self.tx_pending = self.tx_pending[PACKET:]

    @torch.no_grad()
    def detect_loop(self):
        while not self.stop.wait(0.5):
            with self.rx_lock:
                total, ring = self.rx_total, self.rx_ring.copy()
            end = (total - self.args.rx_offset) // FRAME * FRAME + self.args.rx_offset
            if end - self.window < 2 * FRAME:
                continue
            stop = len(ring) - (total - end)
            window = ring[stop - self.window:stop]
            level = 20 * np.log10(np.sqrt(np.mean(window**2)) + 1e-12)
            prob, msg = self.detector.detect_watermark(tensor(window), sample_rate=SR)
            prob, bits = prob.item(), msg.flatten()
            payload = bits_to_int(bits)
            s = self.stats
            s["windows"] += 1
            if prob >= self.args.threshold:
                s["detected"] += 1
                s["correct"] += payload == self.args.payload
                ok = "OK " if payload == self.args.payload else "BAD"
                self.status = f"DETECTED p={prob:.2f} payload={payload:>5} {ok} ({(bits == self.expected).float().mean():.0%} bits)"
            else:
                self.status = f"no watermark p={prob:.2f}" + (" (silence)" if level < -55 else "")

    def status_loop(self):
        while not self.stop.wait(0.25):
            s = self.stats
            sys.stdout.write(f"\rTX {s['tx_pkts']:5d} pkts (fill {s['tx_fill']}, drop {s['tx_drop']}) | "
                             f"RX {s['rx_pkts']:5d} pkts (gap {s['rx_gap']}) | {self.status}".ljust(120))
            sys.stdout.flush()

    # --- audio devices ------------------------------------------------------------------
    def source_file_loop(self, audio: np.ndarray):
        i, t0, n = 0, time.monotonic(), 0
        while not self.stop.is_set():
            if i + self.block > len(audio):
                i = 0
            self.src_q.put(audio[i:i + self.block])
            i += self.block
            n += 1
            time.sleep(max(0.0, t0 + n * self.block / SR - time.monotonic()))

    def playback_cb(self, outdata, frames, t, status):
        if not self.playing and self.play_q.qsize() >= self.args.play_prebuffer:
            self.playing = True
        try:
            outdata[:, 0] = self.play_q.get_nowait() if self.playing else 0
        except queue.Empty:
            outdata[:, 0] = 0
            if self.playing:
                self.stats["underruns"] += 1

    # --- main ---------------------------------------------------------------------------
    async def run(self):
        import sounddevice as sd

        a = self.args
        threads = [threading.Thread(target=f, daemon=True) for f in (self.tx_process_loop, self.detect_loop, self.status_loop)]
        streams = []

        def on_ready():
            print(f"Payload {a.payload} -> {self.expected.tolist()}; call limit {a.duration:.0f}s. Ctrl+C to hang up.\n")
            if a.file:
                threads.append(threading.Thread(target=self.source_file_loop, args=(load_audio(a.file),), daemon=True))
                log(f"[src] looping file {a.file}")
            else:
                streams.append(sd.InputStream(samplerate=SR, blocksize=self.block, channels=1, dtype="float32",
                                              device=device_arg(a.input_device),
                                              callback=lambda i, f, t, s: self.src_q.put(i[:, 0].copy())))
                log(f"[src] mic: {sd.query_devices(device_arg(a.input_device), 'input')['name']}")
            if not a.no_playback:
                streams.append(sd.OutputStream(samplerate=SR, blocksize=self.block, channels=1, dtype="float32",
                                               device=device_arg(a.output_device), callback=self.playback_cb))
                log(f"[out] playback: {sd.query_devices(device_arg(a.output_device), 'output')['name']}")
            for t in threads:
                t.start()
            for st in streams:
                st.start()

        def on_teardown():
            for st in streams:
                st.stop()
                st.close()

        await self.serve_call(on_ready, on_teardown)
        self.summary()

    def summary(self):
        s = self.stats
        print("\n\n=== Summary ===")
        print(f"TX: {s['tx_pkts']} packets sent ({s['tx_fill']} silence fills, {s['tx_drop']} dropped)")
        print(f"RX: {s['rx_pkts']} packets received ({s['rx_gap']} gaps filled), playback underruns {s['underruns']}")
        print(f"Detector windows: {s['windows']}, watermark detected in {s['detected']}, "
              f"correct payload in {s['correct']}/{s['detected']}")
        tx = pcmu_decode(np.frombuffer(b"".join(self.sent_log), np.uint8)) if self.sent_log else None
        rx = pcmu_decode(np.frombuffer(b"".join(self.rx_log), np.uint8)) if self.rx_log else None
        if tx is not None and rx is not None and len(rx) > PCMU_SR and len(tx) > PCMU_SR:
            lag, corr = channel_lag(tx, rx)
            # the jitter buffer shifts the delay by whole packets during a call, so only the phase is meaningful
            needed = 2 * (lag % PACKET)  # 16 kHz grid offset matching the measured 8 kHz phase
            used = self.args.rx_offset % FRAME
            verdict = ("detector grid matched" if needed == used else
                       f"detector grid OFF (used --rx-offset {used}); rerun with --rx-offset {needed}")
            print(f"Channel delay (TX->RX, measured after the call): ~{lag / 8:.0f} ms, correlation {corr:.2f}; "
                  f"grid phase {lag % PACKET}/{PACKET} -> {verdict}")
        if self.args.save and tx is not None:
            out = Path(self.args.save)
            out.mkdir(parents=True, exist_ok=True)
            if self.src_log:
                sf.write(str(out / "source_16k.wav"), np.concatenate(self.src_log), SR)
            sf.write(str(out / "tx_sent_8k.wav"), tx, PCMU_SR)
            if rx is not None:
                sf.write(str(out / "rx_received_8k.wav"), rx, PCMU_SR)
            print(f"Saved source/tx/rx audio to {out}/")


def channel_lag(tx: np.ndarray, rx: np.ndarray) -> tuple[int, float]:
    """Delay of rx relative to tx (samples) via FFT cross-correlation, and normalized peak."""
    n = 1 << int(np.ceil(np.log2(len(tx) + len(rx))))
    c = np.fft.irfft(np.fft.rfft(rx, n) * np.conj(np.fft.rfft(tx, n)), n)
    lag = int(np.argmax(np.abs(c[: len(rx)])))
    seg_rx, seg_tx = rx[lag:], tx[: len(rx) - lag]
    m = min(len(seg_rx), len(seg_tx))
    corr = float(np.dot(seg_rx[:m], seg_tx[:m]) / (np.linalg.norm(seg_rx[:m]) * np.linalg.norm(seg_tx[:m]) + 1e-12))
    return lag, corr


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--payload", type=int, default=42351, help=f"{NBITS}-bit integer (0-{2**NBITS - 1})")
    add_call_args(p)
    p.add_argument("--file", type=Path, help="loop an audio file instead of the mic")
    p.add_argument("--input-device")
    p.add_argument("--output-device")
    p.add_argument("--no-playback", action="store_true")
    p.add_argument("--window-s", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--rx-offset", type=int, default=0, help="shift the detector grid by N samples @ 16 kHz")
    p.add_argument("--play-prebuffer", type=int, default=4, help="40 ms blocks buffered before playback starts")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if not 0 <= args.payload < 2**NBITS:
        sys.exit(f"error: payload must be in [0, {2**NBITS - 1}]")
    check_call_env(args)
    demo = Demo(args)
    try:
        asyncio.run(demo.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
