"""AudioSeal watermarking test app.

Commands:
  embed  IN OUT --payload N   embed a 16-bit integer payload into an audio file
  detect IN                   detect a watermark and decode its payload
  bench  [IN] --payload N     embed, then run detection after common distortions
  mic    --payload N          mic -> watermark -> 8 kHz G.711 PCMU -> 16 kHz -> extract
"""

import argparse
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from audioseal import AudioSeal  # noqa: E402

import audio_io  # noqa: E402
from audio_io import pcmu_decode, pcmu_encode  # noqa: E402,F401  (re-exported for older scripts)

SR = 16000  # AudioSeal models operate at 16 kHz
NBITS = 16


def int_to_bits(value: int) -> torch.Tensor:
    if not 0 <= value < 2**NBITS:
        raise ValueError(f"payload must be in [0, {2**NBITS - 1}]")
    bits = [(value >> (NBITS - 1 - i)) & 1 for i in range(NBITS)]  # MSB first
    return torch.tensor([bits], dtype=torch.int32)


def bits_to_int(bits: torch.Tensor) -> int:
    return int("".join(str(int(b)) for b in bits.flatten().tolist()), 2)


def load_audio(path: Path) -> np.ndarray:
    """Load any soundfile-readable file as mono float32 at 16 kHz."""
    return audio_io.load_audio(path, SR)


def to_tensor(audio: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).reshape(1, 1, -1)


class Watermarker:
    def __init__(self) -> None:
        self.generator = AudioSeal.load_generator("audioseal_wm_16bits")
        self.detector = AudioSeal.load_detector("audioseal_detector_16bits")

    @torch.no_grad()
    def embed(self, audio: np.ndarray, payload: int) -> np.ndarray:
        x = to_tensor(audio)
        wm = self.generator.get_watermark(x, sample_rate=SR, message=int_to_bits(payload))
        return (x + wm).clamp(-1.0, 1.0).flatten().numpy()

    @torch.no_grad()
    def detect(self, audio: np.ndarray) -> tuple[float, int, torch.Tensor]:
        prob, msg = self.detector.detect_watermark(to_tensor(audio), sample_rate=SR, message_threshold=0.5)
        return float(prob.item()), bits_to_int(msg), msg.flatten()


# --- distortions used by `bench` ---------------------------------------------

def add_noise(a: np.ndarray, snr_db: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    p_sig = np.mean(a**2) + 1e-12
    noise = rng.normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), a.shape)
    return (a + noise).astype(np.float32)


def resample_roundtrip(a: np.ndarray, mid_sr: int) -> np.ndarray:
    return soxr.resample(soxr.resample(a, SR, mid_sr), mid_sr, SR).astype(np.float32)


def telephony_channel(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """16 kHz -> 8 kHz -> PCMU bytes -> decode -> 16 kHz.

    Returns (pcmu_bytes_at_8k, received_audio_at_16k).
    """
    narrow = soxr.resample(a, SR, 8000)
    pcmu = pcmu_encode(narrow)
    received = soxr.resample(pcmu_decode(pcmu), 8000, SR).astype(np.float32)
    return pcmu, received


def mulaw_8k(a: np.ndarray) -> np.ndarray:
    return telephony_channel(a)[1]


def crop(a: np.ndarray, seconds: float) -> np.ndarray:
    return a[int(seconds * SR):]


DISTORTIONS = {
    "clean": lambda a: a,
    "volume x0.3": lambda a: a * 0.3,
    "noise 30dB SNR": lambda a: add_noise(a, 30),
    "noise 20dB SNR": lambda a: add_noise(a, 20),
    "noise 10dB SNR": lambda a: add_noise(a, 10),
    "resample 8k roundtrip": lambda a: resample_roundtrip(a, 8000),
    "G.711 mu-law 8k": mulaw_8k,
    "crop first 1s": lambda a: crop(a, 1.0),
}


def tts_sample() -> np.ndarray:
    """Generate a speech sample with macOS `say` when no input file is given."""
    text = (
        "Hello, this is a test call to verify a patient's insurance eligibility. "
        "Could you please confirm the member ID and the effective date of coverage?"
    )
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sample.wav"
        subprocess.run(
            ["say", "-o", str(out), "--file-format=WAVE", f"--data-format=LEF32@{SR}", text],
            check=True,
        )
        return load_audio(out)


# --- commands ------------------------------------------------------------------

def cmd_embed(args: argparse.Namespace) -> None:
    wm = Watermarker()
    audio = load_audio(args.input)
    marked = wm.embed(audio, args.payload)
    sf.write(str(args.output), marked, SR, subtype="PCM_16")
    snr = 10 * np.log10(np.sum(audio**2) / (np.sum((marked - audio) ** 2) + 1e-12))
    print(f"Embedded payload {args.payload} ({int_to_bits(args.payload).flatten().tolist()})")
    print(f"Wrote {args.output} ({len(marked) / SR:.2f}s @ {SR} Hz, watermark SNR {snr:.1f} dB)")


def cmd_detect(args: argparse.Namespace) -> None:
    wm = Watermarker()
    prob, payload, bits = wm.detect(load_audio(args.input))
    print(f"Watermark probability: {prob:.3f} ({'DETECTED' if prob >= args.threshold else 'not detected'})")
    print(f"Decoded payload: {payload} (bits {bits.tolist()})")
    if prob < args.threshold:
        print("Note: with no watermark present the decoded payload is random noise.")


def cmd_bench(args: argparse.Namespace) -> None:
    wm = Watermarker()
    audio = load_audio(args.input) if args.input else tts_sample()
    print(f"Audio: {len(audio) / SR:.2f}s, payload {args.payload}\n")
    marked = wm.embed(audio, args.payload)
    if args.save:
        sf.write("original.wav", audio, SR)
        sf.write("watermarked.wav", marked, SR)
        print("Saved original.wav and watermarked.wav\n")

    expected = int_to_bits(args.payload).flatten()
    rows = [("UNWATERMARKED baseline", audio)] + [(n, f(marked)) for n, f in DISTORTIONS.items()]

    print(f"{'condition':<24} {'det prob':>8} {'decoded':>8} {'bit acc':>8}  result")
    print("-" * 64)
    for name, a in rows:
        prob, payload, bits = wm.detect(a)
        acc = (bits == expected).float().mean().item()
        if name.startswith("UNWATERMARKED"):
            ok = "PASS" if prob < args.threshold else "FALSE POSITIVE"
        else:
            ok = "PASS" if prob >= args.threshold and payload == args.payload else "FAIL"
        print(f"{name:<24} {prob:>8.3f} {payload:>8} {acc:>8.0%}  {ok}")


def record_mic(seconds: float) -> np.ndarray:
    import sounddevice as sd

    print(f"Recording {seconds:.0f}s from the default microphone... speak now.")
    rec = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    print("Recording done.")
    return rec.flatten()


def cmd_mic(args: argparse.Namespace) -> None:
    wm = Watermarker()
    audio = load_audio(args.file) if args.file else record_mic(args.seconds)

    rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-12)
    if rms_db < -60:
        print(f"WARNING: input is near-silent ({rms_db:.0f} dBFS). Check the terminal's "
              "microphone permission (System Settings > Privacy & Security > Microphone).")

    print(f"\n[1] Input:        {len(audio) / SR:.2f}s @ 16 kHz, level {rms_db:.1f} dBFS")
    marked = wm.embed(audio, args.payload)
    print(f"[2] Watermarked:  payload {args.payload} -> bits {int_to_bits(args.payload).flatten().tolist()}")

    pcmu, received = telephony_channel(marked)
    print(f"[3] Telephony:    downsampled to 8 kHz, G.711 PCMU encoded ({len(pcmu)} bytes, "
          f"{len(pcmu) * 8 / (len(pcmu) / 8000) / 1000:.0f} kbps)")
    print("[4] Received:     PCMU decoded, upsampled back to 16 kHz")

    prob, payload, bits = wm.detect(received)
    print("\n=== Extraction ===")
    print(f"Watermark probability: {prob:.3f} ({'DETECTED' if prob >= args.threshold else 'NOT detected'})")
    print(f"Extracted payload:     {payload}")
    print(f"Extracted bits:        {bits.tolist()}")
    match = payload == args.payload
    print(f"Matches embedded:      {'YES' if match else 'NO'}")

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        sf.write(str(out / "1_input.wav"), audio, SR)
        sf.write(str(out / "2_watermarked.wav"), marked, SR)
        (out / "3_channel_8k.pcmu").write_bytes(pcmu.tobytes())
        sf.write(str(out / "4_received_16k.wav"), received, SR)
        print(f"\nSaved stages to {out}/ (3_channel_8k.pcmu is raw headerless mu-law @ 8 kHz)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="embed a payload into an audio file")
    e.add_argument("input", type=Path)
    e.add_argument("output", type=Path)
    e.add_argument("--payload", type=int, default=42351, help="16-bit integer (0-65535)")
    e.set_defaults(func=cmd_embed)

    d = sub.add_parser("detect", help="detect watermark and decode payload")
    d.add_argument("input", type=Path)
    d.add_argument("--threshold", type=float, default=0.5)
    d.set_defaults(func=cmd_detect)

    b = sub.add_parser("bench", help="robustness benchmark (uses macOS `say` if no input)")
    b.add_argument("input", type=Path, nargs="?")
    b.add_argument("--payload", type=int, default=42351)
    b.add_argument("--threshold", type=float, default=0.5)
    b.add_argument("--save", action="store_true", help="write original.wav / watermarked.wav")
    b.set_defaults(func=cmd_bench)

    m = sub.add_parser("mic", help="mic -> watermark -> 8k PCMU -> 16k -> extract payload")
    m.add_argument("--payload", type=int, default=42351, help="16-bit integer (0-65535)")
    m.add_argument("--seconds", type=float, default=5.0, help="recording length")
    m.add_argument("--file", type=Path, help="use an audio file instead of the microphone")
    m.add_argument("--threshold", type=float, default=0.5)
    m.add_argument("--save", metavar="DIR", help="write each pipeline stage to DIR")
    m.set_defaults(func=cmd_mic)

    args = p.parse_args()
    try:
        args.func(args)
    except ValueError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
