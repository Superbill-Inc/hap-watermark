"""AudioSeal scheme (Meta, base 16-bit model) behind the watermark SPI.

Embedder: the base generator applied block by block with past context and a short lookahead
(the "streaming" checkpoint is not used: its watermark doesn't survive 8 kHz telephony).
It modifies the outgoing signal, so silence carries no watermark.

Detector: runs the base detector on sliding windows cut on the watermark's 320-sample frame
grid. The payload only decodes if the received stream is on that grid; `grid_offset` shifts
the windows (86 on the Telnyx route we measured). AudioSeal has no checksum, so a payload is
reported after `confirm` consecutive windows agree, with verified=False.
"""

from __future__ import annotations

import functools
import logging
import warnings
from dataclasses import dataclass

import numpy as np

from ..base import BaseScheme
from ..spi import Detection, Detector, Embedder

SR = 16000
FRAME = 320  # 8*5*4*2: AudioSeal encoder stride


@dataclass
class AudioSealOptions:
    block_ms: int = 40  # embedder block (multiple of 20 ms)
    lookahead_ms: int = 40  # future audio the generator sees (multiple of 20 ms)
    context_ms: int = 500  # past audio the generator sees
    window_s: float = 1.0  # detector window
    hop_s: float = 0.5  # detector runs every hop
    threshold: float = 0.5  # detection probability needed for a window to count
    grid_offset: int = 0  # detector window offset on the 320-sample grid (16 kHz samples)
    confirm: int = 2  # consecutive agreeing windows before reporting a payload


@functools.lru_cache(maxsize=1)
def _models():
    warnings.filterwarnings("ignore")
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    import torch
    from audioseal import AudioSeal

    gen = AudioSeal.load_generator("audioseal_wm_16bits")
    det = AudioSeal.load_detector("audioseal_detector_16bits")
    with torch.no_grad():  # warm up so the first live blocks aren't slow
        z = torch.from_numpy(np.random.default_rng(0).normal(0, 0.01, SR).astype(np.float32)).reshape(1, 1, -1)
        gen.get_watermark(z, sample_rate=SR)
        det.detect_watermark(z, sample_rate=SR)
    return gen, det


def _tensor(x: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(1, 1, -1)


class AudioSealScheme(BaseScheme):
    name = "audioseal"
    Options = AudioSealOptions
    payload_bits = 16
    sample_rate = SR
    frame_samples = FRAME

    def __init__(self, options=None):
        super().__init__(options)
        for f in ("block_ms", "lookahead_ms"):
            if getattr(self.options, f) % 20:
                raise ValueError(f"audioseal {f} must be a multiple of 20 ms (keeps the 320-sample grid)")

    def embedder(self, payload: bytes) -> Embedder:
        return _AudioSealEmbedder(int.from_bytes(self.check_payload(payload), "big"), self.options)

    def detector(self) -> Detector:
        return _AudioSealDetector(self.options)


class _AudioSealEmbedder(Embedder):
    sample_rate = SR

    def __init__(self, payload: int, o: AudioSealOptions):
        from stream import StreamingWatermarker

        gen, _ = _models()
        self.block = SR * o.block_ms // 1000
        self.wm = StreamingWatermarker(gen, payload, self.block, lookahead=SR * o.lookahead_ms // 1000,
                                       context=SR * o.context_ms // 1000)
        self.pending = np.zeros(0, np.float32)
        self.fifo = np.zeros(self.block, np.float32)  # one block of output buffering
        self.latency = self.block + self.wm.lookahead

    def process(self, audio: np.ndarray) -> np.ndarray:
        import torch

        self.pending = np.concatenate([self.pending, audio.astype(np.float32)])
        with torch.no_grad():
            while len(self.pending) >= self.block:
                self.fifo = np.concatenate([self.fifo, self.wm.process(self.pending[: self.block])])
                self.pending = self.pending[self.block:]
        n = len(audio)
        out, self.fifo = self.fifo[:n], self.fifo[n:]
        return np.pad(out, (0, n - len(out)))


class _AudioSealDetector(Detector):
    sample_rate = SR

    def __init__(self, o: AudioSealOptions):
        _, self.det = _models()
        self.o = o
        self.window, self.hop = int(SR * o.window_s), int(SR * o.hop_s)
        self.ring = np.zeros(self.window + 2 * FRAME, np.float32)
        self.total = 0
        self.next_run = self.window + 2 * FRAME
        self.streak: tuple[int, int] | None = None  # (payload, count)
        self.done = False

    def push(self, audio: np.ndarray) -> Detection | None:
        if self.done:
            return None
        self.ring = np.concatenate([self.ring, audio.astype(np.float32)])[-len(self.ring):]
        self.total += len(audio)
        result = None
        while self.total >= self.next_run and result is None:
            result = self._run()
            self.next_run += self.hop
        return result

    def _run(self) -> Detection | None:
        import torch

        off = self.o.grid_offset
        end = (self.total - off) // FRAME * FRAME + off  # last window end on the (shifted) grid
        stop = len(self.ring) - (self.total - end)
        window = self.ring[stop - self.window:stop]
        with torch.no_grad():
            prob, msg = self.det.detect_watermark(_tensor(window), sample_rate=SR)
        p, bits = float(prob.item()), msg.flatten().tolist()
        if p < self.o.threshold:
            self.streak = None
            return None
        payload = int("".join(map(str, bits)), 2)
        count = self.streak[1] + 1 if self.streak and self.streak[0] == payload else 1
        self.streak = (payload, count)
        if count < self.o.confirm:
            return None
        self.done = True
        return Detection(payload=payload.to_bytes(2, "big"), confidence=p, at_sample=self.total, verified=False,
                         info={"windows_agreeing": count, "grid_offset": off})
