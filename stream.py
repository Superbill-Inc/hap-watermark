"""Live streaming demo: mic -> watermark -> 8 kHz G.711 PCMU -> 16 kHz -> extract -> playback.

Watermarking uses the base AudioSeal model block by block, with past context and a short
lookahead. The "streaming" AudioSeal checkpoint is not used: its watermark does not survive
the 4 kHz telephony band-limit, while the base model's watermark does.

The detector only decodes reliably when its window starts on the watermark's 320-sample
frame grid. The pipeline is sample-aligned end to end, so windows are cut on that grid.
Pass --misalign N to shift the window off the grid and watch decoding break.

Use headphones. With speakers, the playback feeds back into the mic.

Examples:
  uv run python stream.py --payload 42351
  uv run python stream.py --no-playback --duration 10
  uv run python stream.py --file original.wav            # simulate the mic with a file
"""

import argparse
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

PCMU_SR = 8000
FRAME = 320  # AudioSeal frame size (8*5*4*2); the detector must be aligned to this grid


def tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(1, 1, -1)


class StreamingWatermarker:
    """Base AudioSeal generator applied block-wise with context + lookahead.

    Each call takes a new block and returns the watermarked block from `lookahead` ago,
    which adds `lookahead` of latency.
    """

    def __init__(self, generator, payload: int, block: int, lookahead: int, context: int):
        self.g, self.msg = generator, int_to_bits(payload)
        self.block, self.lookahead, self.context = block, lookahead, context
        self.buf = np.zeros(context + lookahead, np.float32)

    @torch.no_grad()
    def process(self, x: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, x])
        wm = self.g.get_watermark(tensor(self.buf), sample_rate=SR, message=self.msg).flatten().numpy()
        end = len(self.buf) - self.lookahead
        out = np.clip(self.buf[end - len(x):end] + wm[end - len(x):end], -1, 1)
        self.buf = self.buf[-(self.context + self.lookahead):]
        return out


class PcmuChannel:
    """16 kHz -> 8 kHz -> PCMU bytes -> decode -> 16 kHz, with streaming resamplers."""

    def __init__(self):
        self.down = soxr.ResampleStream(SR, PCMU_SR, 1, dtype="float32")
        self.up = soxr.ResampleStream(PCMU_SR, SR, 1, dtype="float32")

    def process(self, x: np.ndarray) -> tuple[bytes, np.ndarray]:
        pcmu = pcmu_encode(self.down.resample_chunk(x))
        return pcmu.tobytes(), self.up.resample_chunk(pcmu_decode(pcmu))


class Demo:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.block = int(SR * args.block_ms / 1000)
        print("Loading models...", flush=True)
        gen = AudioSeal.load_generator("audioseal_wm_16bits")
        self.detector = AudioSeal.load_detector("audioseal_detector_16bits")
        self.wm = StreamingWatermarker(
            gen, args.payload, self.block,
            lookahead=int(SR * args.lookahead_ms / 1000), context=int(SR * args.context_ms / 1000),
        )
        self.channel = PcmuChannel()
        self.expected = int_to_bits(args.payload).flatten()

        self.in_q: queue.Queue = queue.Queue()
        self.out_q: queue.Queue = queue.Queue()
        self.window = int(SR * args.window_s)
        self.rx_ring = np.zeros(self.window + 2 * FRAME, np.float32)
        self.rx_total = 0  # received samples so far; rx sample i is aligned with sent sample i
        # soxr's streaming resampler emits in bursts (~1660 samples), so buffer that much + 2 blocks
        self.prebuffer = -(-1700 // self.block) + 2
        self.rx_lock = threading.Lock()
        self.stop = threading.Event()

        self.recorded: dict[str, list] = {"input": [], "watermarked": [], "pcmu": [], "received": []}
        self.stats = dict(blocks=0, pcmu_bytes=0, underruns=0, gen_ms=0.0, windows=0, detected=0, correct=0)
        self.status = "waiting for audio..."
        self.in_level = -120.0
        self.playing = False

    # --- audio thread ----------------------------------------------------------------
    def callback(self, indata, outdata, frames, time_info, status):
        if indata is not None:
            self.in_q.put(indata[:, 0].copy())
        if outdata is not None:
            # jitter buffer: absorbs resampler bursts and processing jitter
            if not self.playing and self.out_q.qsize() >= self.prebuffer:
                self.playing = True
            try:
                outdata[:, 0] = self.out_q.get_nowait() if self.playing else 0
            except queue.Empty:
                outdata[:, 0] = 0
                self.stats["underruns"] += 1

    # --- processing thread -----------------------------------------------------------
    def process_loop(self):
        while not self.stop.is_set():
            try:
                x = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            marked = x if self.args.no_watermark else self.wm.process(x)
            pcmu, rx = self.channel.process(marked)
            dt = (time.perf_counter() - t0) * 1000

            # resampler output size varies per chunk; keep exactly one block per playback slot
            self._rx_pending = np.concatenate([getattr(self, "_rx_pending", np.zeros(0, np.float32)), rx])
            while len(self._rx_pending) >= self.block:
                self.out_q.put(self._rx_pending[: self.block])
                self._rx_pending = self._rx_pending[self.block:]

            with self.rx_lock:
                self.rx_ring = np.concatenate([self.rx_ring, rx])[-len(self.rx_ring):]
                self.rx_total += len(rx)
            self.in_level = 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)
            s = self.stats
            s["blocks"] += 1
            s["pcmu_bytes"] += len(pcmu)
            s["gen_ms"] = 0.9 * s["gen_ms"] + 0.1 * dt if s["blocks"] > 1 else dt
            if self.args.save:
                for k, v in (("input", x), ("watermarked", marked), ("pcmu", pcmu), ("received", rx)):
                    self.recorded[k].append(v)

    # --- detector thread -------------------------------------------------------------
    @torch.no_grad()
    def detect_loop(self):
        while not self.stop.wait(self.args.detect_every_s):
            with self.rx_lock:
                total, ring = self.rx_total, self.rx_ring.copy()
            # last window that ends on the frame grid (shifted by --misalign)
            end = (total - self.args.misalign) // FRAME * FRAME + self.args.misalign
            if end - self.window < 2 * FRAME:  # not enough audio yet (skip the zero-padded start)
                continue
            stop = len(ring) - (total - end)
            window = ring[stop - self.window:stop]
            level = 20 * np.log10(np.sqrt(np.mean(window**2)) + 1e-12)
            prob, msg = self.detector.detect_watermark(tensor(window), sample_rate=SR)
            prob, bits = prob.item(), msg.flatten()
            payload, bit_acc = bits_to_int(bits), (bits == self.expected).float().mean().item()
            s = self.stats
            s["windows"] += 1
            if prob >= self.args.threshold:
                s["detected"] += 1
                s["correct"] += payload == self.args.payload
                verdict = f"DETECTED p={prob:.2f} payload={payload:>5} {'OK ' if payload == self.args.payload else 'BAD'} ({bit_acc:.0%} bits)"
            else:
                verdict = f"no watermark p={prob:.2f}" + (" (silence)" if level < -55 else "")
            self.status = verdict

    def status_loop(self):
        t0 = time.time()
        while not self.stop.wait(0.2):
            s = self.stats
            line = (f"\r{time.time() - t0:6.1f}s | mic {self.in_level:5.0f} dBFS | "
                    f"PCMU {s['pcmu_bytes'] // 160:5d} pkts | wm {s['gen_ms']:4.1f}ms/blk | {self.status}")
            sys.stdout.write(line.ljust(110))
            sys.stdout.flush()

    @torch.no_grad()
    def warmup(self):
        """Run each model once so the first live blocks aren't slow."""
        z = tensor(np.random.default_rng(0).normal(0, 0.01, SR).astype(np.float32))
        self.wm.g.get_watermark(z, sample_rate=SR, message=self.wm.msg)
        self.detector.detect_watermark(z, sample_rate=SR)

    # --- sources ---------------------------------------------------------------------
    def run(self):
        a = self.args
        self.warmup()
        latency = a.block_ms + a.lookahead_ms + 1000 * self.prebuffer * self.block // SR
        print(f"Payload {a.payload} -> {self.expected.tolist()}")
        print(f"Block {a.block_ms} ms, lookahead {a.lookahead_ms} ms, jitter buffer {self.prebuffer} blocks "
              f"(~{latency} ms mic-to-ear + device buffers); detector window {a.window_s}s every "
              f"{a.detect_every_s}s, grid offset {a.misalign}")
        threads = [threading.Thread(target=f, daemon=True) for f in (self.process_loop, self.detect_loop, self.status_loop)]
        for t in threads:
            t.start()
        try:
            if a.file:
                self.run_file()
            else:
                self.run_mic()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            for t in threads:
                t.join(timeout=1)
        self.summary()

    def run_mic(self):
        import sounddevice as sd

        kind = "input-only" if self.args.no_playback else "duplex"
        print(f"Streaming ({kind}). Ctrl+C to stop.\n")
        stream_cls = sd.InputStream if self.args.no_playback else sd.Stream
        cb = (lambda i, f, t, s: self.callback(i, None, f, t, s)) if self.args.no_playback else self.callback
        dev_in, dev_out = device_arg(self.args.input_device), device_arg(self.args.output_device)
        print(f"Input:  {sd.query_devices(dev_in, 'input')['name']}")
        if not self.args.no_playback:
            print(f"Output: {sd.query_devices(dev_out, 'output')['name']}")
        device = dev_in if self.args.no_playback else (dev_in, dev_out)
        with stream_cls(samplerate=SR, blocksize=self.block, channels=1, dtype="float32", callback=cb, device=device):
            self._wait()

    def run_file(self):
        """Feed a file as if it were the mic, in real time (or as fast as possible)."""
        audio = load_audio(self.args.file)
        print(f"Streaming file {self.args.file} ({len(audio) / SR:.1f}s). Ctrl+C to stop.\n")
        n = len(audio) // self.block
        if self.args.no_playback:
            for i in range(n):
                self.callback(audio[i * self.block:(i + 1) * self.block, None], None, self.block, None, None)
                if not self.args.fast:
                    time.sleep(self.block / SR)
            while not self.in_q.empty():
                time.sleep(0.05)
            time.sleep(self.args.detect_every_s * 2)
            return
        import sounddevice as sd

        pos = [0]

        def cb(outdata, frames, t, status):
            chunk = audio[pos[0]:pos[0] + frames]
            pos[0] += frames
            if len(chunk) == frames:
                self.callback(chunk[:, None], outdata, frames, t, status)
            else:
                outdata[:] = 0
                raise sd.CallbackStop

        with sd.OutputStream(samplerate=SR, blocksize=self.block, channels=1, dtype="float32", callback=cb,
                             device=device_arg(self.args.output_device)) as st:
            while st.active:
                time.sleep(0.1)

    def _wait(self):
        end = time.time() + self.args.duration if self.args.duration else None
        while end is None or time.time() < end:
            time.sleep(0.1)

    def summary(self):
        s = self.stats
        print("\n\n=== Summary ===")
        print(f"Blocks processed: {s['blocks']} ({s['blocks'] * self.block / SR:.1f}s), "
              f"PCMU {s['pcmu_bytes']} bytes, playback underruns: {s['underruns']}")
        print(f"Detector windows: {s['windows']}, watermark detected in {s['detected']}, "
              f"correct payload in {s['correct']}/{s['detected']}")
        if self.args.save and s["blocks"]:
            out = Path(self.args.save)
            out.mkdir(parents=True, exist_ok=True)
            for k in ("input", "watermarked", "received"):
                sf.write(str(out / f"{k}.wav"), np.concatenate(self.recorded[k]), SR)
            (out / "channel_8k.pcmu").write_bytes(b"".join(self.recorded["pcmu"]))
            print(f"Saved input/watermarked/received .wav and channel_8k.pcmu to {out}/")


def device_arg(value):
    """Device index or name substring; None = system default."""
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--payload", type=int, default=42351, help=f"{NBITS}-bit integer (0-{2**NBITS - 1})")
    p.add_argument("--block-ms", type=int, default=40, help="processing/audio block size")
    p.add_argument("--lookahead-ms", type=int, default=40, help="future audio the watermarker sees (adds latency)")
    p.add_argument("--context-ms", type=int, default=500, help="past audio the watermarker sees")
    p.add_argument("--window-s", type=float, default=1.0, help="detector window length")
    p.add_argument("--detect-every-s", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--misalign", type=int, default=0, help="shift detector window off the 320-sample grid")
    p.add_argument("--input-device", help="mic device index or name substring (see --list-devices)")
    p.add_argument("--output-device", help="playback device index or name substring")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--no-watermark", action="store_true", help="control run: skip embedding (expect no detection)")
    p.add_argument("--no-playback", action="store_true", help="don't play the received audio")
    p.add_argument("--duration", type=float, help="stop after N seconds (mic mode)")
    p.add_argument("--file", type=Path, help="use an audio file instead of the mic")
    p.add_argument("--fast", action="store_true", help="with --file --no-playback: don't pace in real time")
    p.add_argument("--save", metavar="DIR", help="write input/watermarked/pcmu/received at the end")
    args = p.parse_args()
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if not 0 <= args.payload < 2**NBITS:
        sys.exit(f"error: payload must be in [0, {2**NBITS - 1}]")
    Demo(args).run()


if __name__ == "__main__":
    main()
