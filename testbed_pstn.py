"""Watermark testbed over a real Telnyx call, for any scheme behind the watermark SPI.

    our "bot" (TX leg): bot voice (mic / file / silence) -> embedder -> PCMU -> Telnyx call
    "responder" (RX leg, answered by the webhook): received PCMU -> detector -> payload

The detector re-arms after every detection, so one call yields many independent detections
(each timed from when that detector started listening). Call setup: see telnyx_call.py.

    (set TELNYX_API_KEY, TELNYX_PUBLIC_KEY, TELNYX_FROM, TELNYX_TO, TELNYX_CONNECTION_ID, PUBLIC_URL; see README)
    uv run python testbed_pstn.py --scheme dsss --duration 45
    uv run python testbed_pstn.py --scheme dsss --speech say --speech-duty 0.6 --opt speech_margin_db=20
    uv run python testbed_pstn.py --scheme dsss --mic --playback --opt speech_margin_db=20
    uv run python testbed_pstn.py --scheme dsss --mic --playback --wm toggle --toggle-s 10
    uv run python testbed_pstn.py --scheme audioseal --mic --playback --opt grid_offset=86

Use headphones with --mic --playback, or the speaker feeds back into the mic.
"""

import argparse
import asyncio
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import watermark
from watermark.factory import parse_opts
from watermark.util import payload_from_hex, payload_hex
from audio_io import pcmu_decode, pcmu_encode, speech_source
from telnyx_call import PACKET, PCMU_SR, TelnyxCall, add_call_args, check_call_env, log
from testbed_local import speech_track

FS = PCMU_SR
MIC_MAX = FS * 3 // 10  # cap mic backlog at 300 ms so latency can't build up
PLAY_PREBUFFER = 6  # 20 ms packets buffered before playback starts


def device_arg(value):
    """Audio device index or name substring; None = system default."""
    if value is None:
        return None
    return int(value) if value.isdigit() else value


class Delay:
    """Fixed sample delay (keeps the unwatermarked path in step with the embedder's latency)."""

    def __init__(self, n: int):
        self.buf = np.zeros(n, np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not len(self.buf):
            return x
        self.buf = np.concatenate([self.buf, x])
        out, self.buf = self.buf[: len(x)], self.buf[len(x):]
        return out


class WatermarkCall(TelnyxCall):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.scheme = watermark.default_factory.create(args.scheme, scheme_options(args))
        self.payload = payload_from_hex(self.scheme, args.payload)
        self.embedder = watermark.at_rate(self.scheme.embedder(self.payload), FS, self.scheme)
        self.dry_delay = Delay(self.embedder.latency)
        n = int((args.duration + 10) * FS)
        if args.speech:
            sp = speech_source(None if args.speech == "say" else args.speech, FS)
            self.speech = speech_track(n, args.speech_duty, np.random.default_rng(7), sp)
        else:
            self.speech = np.zeros(n, np.float32)
        self.tx_pos = 0
        self.wm_gain = 1.0  # current watermark mix (ramped on toggles)
        self.wm_state: bool | None = None
        self.mic_q: queue.Queue = queue.Queue()
        self.mic_buf = np.zeros(0, np.float32)
        self.mic_level = -120.0
        self.play_q: queue.Queue = queue.Queue()
        self.playing = False
        self.rx_q: queue.Queue = queue.Queue()  # received audio for the detector thread
        self.detector = watermark.at_rate(self.scheme.detector(), FS, self.scheme)
        self.rx_samples = 0  # received so far (RX timeline)
        self.det_samples = 0  # fed to the detector so far
        self.armed_at = 0  # det_samples when the current detector started
        self.detections: list[dict] = []
        self.stats.update(decodes=0, wrong=0, mic_drop=0, underruns=0)

    # --- TX: one packet at a time, paced by the sender's 20 ms clock ------------------------
    def tx_loop(self):
        self.tx_active.wait()
        while not self.stop.is_set():
            while self.tx_q.qsize() < self.args.tx_prebuffer + 3:
                sp = self.next_mic_packet() if self.args.mic else self.speech[self.tx_pos:self.tx_pos + PACKET]
                if len(sp) < PACKET:
                    sp = np.pad(sp, (0, PACKET - len(sp)))
                x = self.mix(sp.astype(np.float32))
                self.tx_q.put(pcmu_encode(np.clip(x, -1, 1)).tobytes())
                self.tx_pos += PACKET
            time.sleep(0.005)

    def wm_on_at(self, t: float) -> bool:
        if self.args.wm == "toggle":
            return int(t // self.args.toggle_s) % 2 == 0  # ON first, then OFF, ...
        return self.args.wm == "on"

    def mix(self, sp: np.ndarray) -> np.ndarray:
        """Watermarked or raw (delay-matched) audio. The embedder always runs, so its state
        continues through OFF periods; switches crossfade over one packet to avoid clicks."""
        t = self.tx_pos / FS  # sender timeline: packet k goes out k * 20 ms after the TX stream starts
        on = self.wm_on_at(t)
        if on != self.wm_state:
            if self.args.wm == "toggle" or self.wm_state is None:
                log(f"[wm] {'ON ' if on else 'OFF'} at tx t={t:5.1f}s")
            self.wm_state = on
        wet, dry = self.embedder.process(sp), self.dry_delay(sp)
        target = 1.0 if on else 0.0
        g = np.linspace(self.wm_gain, target, PACKET, dtype=np.float32)
        self.wm_gain = target
        return dry + g * (wet - dry)

    def next_mic_packet(self) -> np.ndarray:
        """Latest mic audio for one packet (shorter if the mic is behind)."""
        while True:
            try:
                self.mic_buf = np.concatenate([self.mic_buf, self.mic_q.get_nowait()])
            except queue.Empty:
                break
        if len(self.mic_buf) > MIC_MAX:  # mic ahead of the call clock (or pre-answer backlog): drop the oldest
            self.stats["mic_drop"] += len(self.mic_buf) - MIC_MAX // 2
            self.mic_buf = self.mic_buf[-(MIC_MAX // 2):]
        pkt, self.mic_buf = self.mic_buf[:PACKET], self.mic_buf[PACKET:]
        if len(pkt):
            self.mic_level = 20 * np.log10(np.sqrt(np.mean(pkt**2)) + 1e-12)
        return pkt

    def playback_cb(self, outdata, frames, t, status):
        if not self.playing and self.play_q.qsize() >= PLAY_PREBUFFER:
            self.playing = True
        try:
            outdata[:, 0] = self.play_q.get_nowait() if self.playing else 0
        except queue.Empty:
            outdata[:, 0] = 0
            if self.playing:
                self.stats["underruns"] += 1

    # --- RX: playback + detector thread (re-arms after each detection) -----------------------
    def on_rx_audio(self, data: bytes):
        x = pcmu_decode(np.frombuffer(data, np.uint8))
        if self.args.playback and len(x) == PACKET:
            self.play_q.put(x)
        self.rx_samples += len(x)
        self.rx_q.put(x)

    def detect_loop(self):
        while not self.stop.is_set():
            try:
                x = self.rx_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.det_samples += len(x)
            r = self.detector.push(x)
            if r is not None:
                self.report(r)
                self.detector = watermark.at_rate(self.scheme.detector(), FS, self.scheme)  # re-arm
                self.armed_at = self.det_samples

    def report(self, r: watermark.Detection):
        ok = r.payload == self.payload
        took = r.at_sample / FS
        self.stats["decodes"] += ok
        self.stats["wrong"] += not ok
        self.detections.append(dict(ok=ok, took=took, at=self.det_samples / FS, info=r.info))
        log(f"[detect] {payload_hex(r.payload)} {'OK' if ok else 'WRONG'} at rx t={self.det_samples / FS:5.1f}s, "
            f"after {took:.1f}s of listening (confidence {r.confidence:.2f}, "
            f"{'verified' if r.verified else 'unverified'}, {r.info})")

    def status_loop(self):
        while not self.stop.wait(0.25):
            s = self.stats
            listening = (self.det_samples - self.armed_at) / FS
            mic = f"mic {self.mic_level:4.0f} dBFS | " if self.args.mic else ""
            sys.stdout.write(f"\r{mic}TX {s['tx_pkts']:5d} | RX {s['rx_pkts']:5d} (gap {s['rx_gap']}) | "
                             f"detections {s['decodes']} ok / {s['wrong']} wrong | listening {listening:4.1f}s".ljust(110))
            sys.stdout.flush()

    async def run(self):
        a = self.args
        threads = [threading.Thread(target=f, daemon=True) for f in (self.tx_loop, self.detect_loop, self.status_loop)]
        streams = []

        def on_ready():
            if a.mic or a.playback:
                import sounddevice as sd
            if a.mic:
                streams.append(sd.InputStream(samplerate=FS, blocksize=PACKET, channels=1, dtype="float32",
                                              device=device_arg(a.input_device),
                                              callback=lambda i, f, t, st: self.mic_q.put(i[:, 0].copy())))
                log(f"[src] mic: {sd.query_devices(device_arg(a.input_device), 'input')['name']}")
            if a.playback:
                streams.append(sd.OutputStream(samplerate=FS, blocksize=PACKET, channels=1, dtype="float32",
                                               device=device_arg(a.output_device), callback=self.playback_cb))
                log(f"[out] playback of the answered leg: {sd.query_devices(device_arg(a.output_device), 'output')['name']}")
            for st in streams:
                st.start()
            voice = (", bot voice = live mic" if a.mic else
                     f", bot speech {a.speech_duty:.0%} of the time" if a.speech else ", bot silent")
            mode = {"on": "watermark ON", "off": "watermark OFF (raw audio; detector still listens: false-detection check)",
                    "toggle": f"watermark TOGGLE {a.toggle_s:g}s on / {a.toggle_s:g}s off"}[a.wm]
            print(f"{self.scheme.describe()}\nPayload {payload_hex(self.payload)}; {mode}{voice}; "
                  f"embed latency {self.embedder.latency / FS * 1000:.0f} ms; call limit {a.duration:.0f}s\n")
            for t in threads:
                t.start()

        def on_teardown():
            for st in streams:
                st.stop()
                st.close()

        await self.serve_call(on_ready, on_teardown)
        self.summary()

    def summary(self):
        s, d = self.stats, self.detections
        print("\n\n=== Summary ===")
        print(f"TX {s['tx_pkts']} packets ({s['tx_fill']} silence fills, {s['tx_drop']} dropped); "
              f"RX {s['rx_pkts']} packets ({s['rx_gap']} gaps filled), {self.rx_samples / FS:.1f}s received")
        print(f"Detections: {s['decodes']} correct, {s['wrong']} wrong")
        if self.args.mic or self.args.playback:
            print(f"Mic samples dropped (latency cap): {s['mic_drop']}; playback underruns: {s['underruns']}")
        took = [x["took"] for x in d if x["ok"]]
        if took:
            print(f"First correct detection {next(x['at'] for x in d if x['ok']):.1f}s into the call; "
                  f"time-to-decode per detection: median {np.median(took):.1f}s, max {np.max(took):.1f}s")
        if self.args.save and self.sent_log:
            out = Path(self.args.save)
            out.mkdir(parents=True, exist_ok=True)
            sf.write(str(out / "tx_sent_8k.wav"), pcmu_decode(np.frombuffer(b"".join(self.sent_log), np.uint8)), FS)
            if self.rx_log:
                sf.write(str(out / "rx_received_8k.wav"), pcmu_decode(np.frombuffer(b"".join(self.rx_log), np.uint8)), FS)
            print(f"Saved sent/received audio to {out}/")


def scheme_options(args: argparse.Namespace) -> dict[str, str]:
    opts = parse_opts(args.opt)
    # shortcuts kept from the DSSS-only testbed
    if args.level is not None:
        opts.setdefault("level_dbfs", str(args.level))
    if args.speech_margin is not None:
        opts.setdefault("speech_margin_db", str(args.speech_margin))
    return opts


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheme", default="dsss", help=f"watermark scheme: {', '.join(watermark.default_factory.names())}")
    p.add_argument("--opt", action="append", metavar="KEY=VALUE", help="scheme option (repeatable)")
    p.add_argument("--payload", help="hex payload (default: a fixed pattern of the scheme's size)")
    p.add_argument("--wm", "--bed", dest="wm", choices=("on", "off", "toggle"), default="on",
                   help="off = raw audio only (expect 0 detections); toggle = on/off every --toggle-s")
    p.add_argument("--toggle-s", type=float, default=10.0, help="toggle period half: X s on, X s off")
    p.add_argument("--level", type=float, help="DSSS shortcut for --opt level_dbfs=")
    p.add_argument("--speech-margin", type=float, help="DSSS shortcut for --opt speech_margin_db=")
    p.add_argument("--speech", help='bot speech file, or "say" to synthesize one (default: bot silent)')
    p.add_argument("--speech-duty", type=float, default=0.6, help="fraction of time the bot talks")
    p.add_argument("--mic", action="store_true", help="live mic as the bot's voice")
    p.add_argument("--playback", action="store_true", help="play the answered leg's received audio")
    p.add_argument("--input-device", help="mic device index or name substring")
    p.add_argument("--output-device", help="speaker device index or name substring")
    add_call_args(p)
    return p.parse_args(argv)


def main():
    args = parse_args()
    check_call_env(args)
    try:
        asyncio.run(WatermarkCall(args).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
