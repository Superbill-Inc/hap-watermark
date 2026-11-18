"""Run a scheme's embedder/detector inside a pipeline with a different sample rate."""

from __future__ import annotations

import dataclasses

import numpy as np
import soxr

from .spi import Detection, Detector, Embedder, Scheme

PREFILL_S = 0.3  # output buffer that absorbs soxr's bursty streaming output (bursts up to ~0.2 s)


def at_rate(obj: Embedder | Detector, rate: int, scheme: Scheme | None = None) -> Embedder | Detector:
    """Wrap an embedder or detector so it accepts/returns audio at `rate`.

    soxr's streaming resamplers keep sample i of the output aligned with sample i of the input
    (no fractional delay), and the embedder adapter's buffering is a whole number of the scheme's
    frames, so frame-aligned schemes (AudioSeal) keep their grid.
    """
    if obj.sample_rate == rate:
        return obj
    frame = scheme.frame_samples if scheme is not None else 1
    if isinstance(obj, Embedder):
        return ResampledEmbedder(obj, rate, frame)
    return ResampledDetector(obj, rate)


class ResampledEmbedder(Embedder):
    def __init__(self, inner: Embedder, rate: int, frame_samples: int = 1):
        self.inner, self.sample_rate = inner, rate
        self.up = soxr.ResampleStream(rate, inner.sample_rate, 1, dtype="float32")
        self.down = soxr.ResampleStream(inner.sample_rate, rate, 1, dtype="float32")
        # keep the prefill a whole number of scheme frames (in pipeline samples)
        align = max(1, round(frame_samples * rate / inner.sample_rate))
        prefill = -(-int(PREFILL_S * rate) // align) * align
        self.fifo = np.zeros(prefill, np.float32)
        self.latency = prefill + round(inner.latency * rate / inner.sample_rate)
        self.underruns = 0

    def process(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        up = self.up.resample_chunk(audio.astype(np.float32))
        if len(up):
            self.fifo = np.concatenate([self.fifo, self.down.resample_chunk(self.inner.process(up))])
        out, self.fifo = self.fifo[:n], self.fifo[n:]
        if len(out) < n:  # resampler held back more than the prefill covers
            self.underruns += 1
            out = np.pad(out, (0, n - len(out)))
        return out


class ResampledDetector(Detector):
    def __init__(self, inner: Detector, rate: int):
        self.inner, self.sample_rate = inner, rate
        self.up = soxr.ResampleStream(rate, inner.sample_rate, 1, dtype="float32")

    def push(self, audio: np.ndarray) -> Detection | None:
        up = self.up.resample_chunk(audio.astype(np.float32))
        if not len(up):
            return None
        r = self.inner.push(up)
        if r is None:
            return None
        return dataclasses.replace(r, at_sample=round(r.at_sample * self.sample_rate / self.inner.sample_rate))
