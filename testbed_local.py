"""Local round-trip testbed for any watermark scheme (watermark SPI), through a simulated phone channel.

    uv run python testbed_local.py --scheme dsss
    uv run python testbed_local.py --scheme dsss --opt level_dbfs=-57 --opt speech_margin_db=20
    uv run python testbed_local.py --scheme audioseal --trials 5
    uv run python testbed_local.py --list-schemes

Each trial: TX = embedder(bot speech, or zeros while the bot is silent) -> channel -> detector,
in 20 ms packets at the 8 kHz telephony rate (schemes with another native rate run through
watermark.at_rate). The channel models what we measured on Telnyx calls (unknown sub-packet offset,
whole-packet jitter-buffer insertions/drops) plus the risks still open.
"""

import argparse
import time

import numpy as np
import soxr

import watermark
from watermark.factory import parse_opts
from watermark.util import payload_from_hex, payload_hex
from audio_io import pcmu_roundtrip, speech_source

FS = 8000
PACKET = 160


def _phrases(speech: np.ndarray) -> list[np.ndarray]:
    """Split speech into phrases at its natural pauses (>= 120 ms more than 40 dB below the peak)."""
    frames = speech[: len(speech) // 160 * 160].reshape(-1, 160)
    rms = np.sqrt((frames**2).mean(axis=1))
    quiet = rms < rms.max() * 10 ** (-40 / 20)
    cuts, run = [0], 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run == 6:  # 120 ms of pause: cut in the middle of it
            cuts.append((i - 3) * 160)
    cuts.append(len(speech))
    out = []
    for a, b in zip(cuts, cuts[1:]):
        seg = speech[a:b]
        loud = np.nonzero(np.abs(seg) > np.abs(speech).max() * 10 ** (-40 / 20))[0]
        if len(loud):
            out.append(seg[loud[0]:loud[-1] + 1])
    return out or [speech]


def _faded(x: np.ndarray, ms: float = 5.0) -> np.ndarray:
    n = min(int(FS * ms / 1000), len(x) // 2)
    if n:
        x = x.copy()
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def speech_track(n: int, duty: float, rng: np.random.Generator, speech: np.ndarray) -> np.ndarray:
    """Bot speech covering about `duty` of the time: talk spurts of whole phrases (1-3 s), cut at the
    sample's natural pauses with short fades, separated by silence. duty >= 1: phrases back to back."""
    out = np.zeros(n, np.float32)
    if duty <= 0:
        return out
    phrases = _phrases(speech)
    k = int(rng.integers(0, len(phrases)))  # start at a random phrase

    def next_spurt(min_s: float) -> np.ndarray:
        nonlocal k
        parts = []
        while sum(len(p) for p in parts) < min_s * FS:
            parts += [_faded(phrases[k % len(phrases)]), np.zeros(int(0.15 * FS), np.float32)]
            k += 1
        return np.concatenate(parts[:-1])

    t = 0 if duty >= 1 else int(rng.uniform(0, 1.5) * FS)
    while t < n:
        seg = next_spurt(rng.uniform(1, 3))
        out[t:t + len(seg)] = seg[: max(0, min(len(seg), n - t))]
        gap = 0.15 if duty >= 1 else len(seg) / FS * (1 - duty) / duty * rng.uniform(0.5, 1.5)
        t += len(seg) + int(gap * FS)
    return out


def channel(x: np.ndarray, sc: dict, rng: np.random.Generator) -> np.ndarray:
    # unknown start: whole packets of pre-roll + a sub-packet phase (Telnyx measured 43)
    phase = 0 if sc.get("aligned") else int(rng.integers(0, PACKET))
    pre = np.zeros(int(rng.integers(0, 20)) * PACKET + phase, np.float32)
    y = np.concatenate([pre, x])
    if sc.get("drift_ppm"):
        y = soxr.resample(y, FS, FS * (1 + sc["drift_ppm"] * 1e-6)).astype(np.float32)
    if sc.get("invert"):
        y = -y
    y = pcmu_roundtrip(y)
    # jitter buffer: insert/drop whole packets at random times (more of them early in the call)
    for _ in range(sc.get("jumps", 0)):
        t = int(min(rng.exponential(3.0), len(y) / FS - 1) * FS) // PACKET * PACKET
        if rng.random() < 0.8:
            y = np.concatenate([y[:t], y[t - PACKET:t], y[t:]])  # PLC-style repeat
        else:
            y = np.concatenate([y[:t], y[t + PACKET:]])  # drop
    if sc.get("agc"):
        y = y * rng.uniform(0.4, 2.0)
    if sc.get("noise_dbfs") is not None:
        y = y + rng.normal(0, 10 ** (sc["noise_dbfs"] / 20), len(y)).astype(np.float32)
    if sc.get("dtx_dbfs") is not None:  # silence suppression + comfort noise per 20 ms packet
        y = y.copy()
        thr = 10 ** (sc["dtx_dbfs"] / 20)
        for i in range(0, len(y) - PACKET, PACKET):
            rms = np.sqrt(np.mean(y[i:i + PACKET] ** 2))
            if rms < thr:
                y[i:i + PACKET] = rng.normal(0, rms, PACKET)
    return pcmu_roundtrip(y)


SCENARIOS = [
    ("packet-aligned, bot speaking", {"aligned": True, "speech_duty": 1.0}),
    ("clean (PCMU + unknown phase)", {}),
    ("+ jitter jumps (6 / call)", {"jumps": 6}),
    ("+ jumps + AGC + line noise -60", {"jumps": 6, "agc": True, "noise_dbfs": -60}),
    ("+ inverted polarity", {"jumps": 6, "invert": True}),
    ("+ bot speech 30% of time", {"jumps": 6, "speech_duty": 0.3}),
    ("+ bot speech 60% of time", {"jumps": 6, "speech_duty": 0.6}),
    ("+ bot speech continuous", {"jumps": 6, "speech_duty": 1.0}),
    ("+ clock drift 50 ppm", {"jumps": 6, "drift_ppm": 50}),
    ("silence suppression @ -45 dBFS", {"dtx_dbfs": -45}),
]


def run_trial(scheme: watermark.Scheme, payload: bytes, sc: dict, n: int, speech: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    sp = speech_track(n, sc.get("speech_duty", 0), rng, speech)
    emb = watermark.at_rate(scheme.embedder(payload), FS, scheme)
    t0 = time.perf_counter()
    tx = np.concatenate([emb.process(sp[i:i + PACKET]) for i in range(0, n - PACKET + 1, PACKET)])
    t_emb = time.perf_counter() - t0
    rx = channel(tx, sc, rng)
    det = watermark.at_rate(scheme.detector(), FS, scheme)
    t0, r = time.perf_counter(), None
    for i in range(0, len(rx) - PACKET + 1, PACKET):
        r = det.push(rx[i:i + PACKET])
        if r is not None:
            break
    return r, t_emb, time.perf_counter() - t0


def run(args):
    scheme = watermark.default_factory.create(args.scheme, parse_opts(args.opt))
    payload = payload_from_hex(scheme, args.payload)
    speech = speech_source(args.speech, FS)
    n = int(args.seconds * FS)
    print(f"{scheme.describe()}\npayload {payload_hex(payload)}; {args.trials} trials x {args.seconds:.0f}s "
          f"at {FS} Hz (PCMU channel)\n")
    print(f"{'scenario':<32} {'decoded':>8} {'wrong':>6} {'t50':>6} {'t90':>6}  {'embed RTF':>9} {'detect RTF':>10}")
    print("-" * 86)
    for name, sc in SCENARIOS:
        if args.only and args.only.lower() not in name.lower():
            continue
        ok = wrong = 0
        times, t_emb, t_det = [], 0.0, 0.0
        for t in range(args.trials):
            r, te, td = run_trial(scheme, payload, sc, n, speech, 1000 + t)
            t_emb += te
            t_det += td
            if r is None:
                continue
            if r.payload == payload:
                ok += 1
                times.append(r.at_sample / FS)
            else:
                wrong += 1
        t50 = f"{np.percentile(times, 50):.1f}s" if times else "-"
        t90 = f"{np.percentile(times, 90):.1f}s" if times else "-"
        audio_s = args.trials * args.seconds
        print(f"{name:<32} {ok:>4}/{args.trials:<3} {wrong:>6} {t50:>6} {t90:>6}  "
              f"{t_emb / audio_s:>9.3f} {t_det / audio_s:>10.3f}")
    print("\ndecoded = correct payload within the window; wrong = a different payload was reported;"
          "\nt50/t90 = audio seconds until decode; RTF = CPU seconds per second of audio (embedder / detector)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheme", default="dsss", help="watermark scheme (see --list-schemes)")
    p.add_argument("--opt", action="append", metavar="KEY=VALUE", help="scheme option (repeatable)")
    p.add_argument("--payload", help="hex payload (default: a fixed pattern of the scheme's size)")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seconds", type=float, default=10.0, help="detection window per trial")
    p.add_argument("--speech", help="bot speech file (default: synthesized with macOS say)")
    p.add_argument("--only", help="run scenarios whose name contains this text")
    p.add_argument("--list-schemes", action="store_true")
    args = p.parse_args()
    if args.list_schemes:
        for name in watermark.default_factory.names():
            print(watermark.default_factory.create(name).describe())
        return
    run(args)


if __name__ == "__main__":
    main()
