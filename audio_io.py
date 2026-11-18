"""Torch-free audio helpers: file loading and a bit-exact G.711 mu-law (PCMU) codec."""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_audio(path: Path | str, rate: int) -> np.ndarray:
    """Load any soundfile-readable file as mono float32 at `rate` Hz."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != rate:
        mono = soxr.resample(mono, sr, rate).astype(np.float32)
    return mono


_ULAW_BIAS = 0x84


def pcmu_encode(a: np.ndarray) -> np.ndarray:
    """Float [-1, 1] -> G.711 mu-law bytes (bit-exact with the CCITT/Sun g711.c encoder)."""
    pcm14 = np.round(np.clip(a, -1, 1) * 32767).astype(np.int32) >> 2  # 14-bit, floor shift
    sign = np.where(pcm14 < 0, 0x80, 0)
    mag = np.minimum(np.where(pcm14 < 0, -pcm14, pcm14) + 0x21, 0x1FFF)  # bias, saturate
    exponent = np.clip(np.floor(np.log2(mag >> 5)), 0, 7).astype(np.int32)
    mantissa = (mag >> (exponent + 1)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def pcmu_decode(b: np.ndarray) -> np.ndarray:
    """G.711 mu-law bytes -> float [-1, 1]."""
    u = ~b.astype(np.int32) & 0xFF
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + _ULAW_BIAS) << exponent) - _ULAW_BIAS
    return (np.where(u & 0x80, -mag, mag) / 32768.0).astype(np.float32)


def pcmu_roundtrip(a: np.ndarray) -> np.ndarray:
    """Encode + decode: what a G.711 mu-law hop does to the signal."""
    return pcmu_decode(pcmu_encode(np.clip(a, -1, 1)))


SPEECH_TEXT = (
    "Hello, this is a test call to verify a patient's insurance eligibility. "
    "Could you please confirm the member ID and the effective date of coverage?"
)


def tts_sample(rate: int, text: str = SPEECH_TEXT) -> np.ndarray:
    """Speech sample synthesized with macOS `say` (no files left behind)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "speech.wav"
        subprocess.run(["say", "-o", str(out), "--file-format=WAVE", "--data-format=LEF32@16000", text], check=True)
        return load_audio(out, rate)


def speech_source(path: Path | str | None, rate: int) -> np.ndarray:
    """Bot speech for tests: the given file, or a synthesized sample."""
    return load_audio(path, rate) if path else tts_sample(rate)
