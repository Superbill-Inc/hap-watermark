"""Watermark SPI: interface definitions for pluggable streaming watermark schemes.

A **Scheme** is one watermarking method with fixed properties (payload size, native sample
rate) and tunable `Options`. It creates:

  * **Embedder** (sender side): `process(block) -> block`. Takes the next block of the outgoing
    audio (the bot's voice; zeros while it's silent) and returns the same number of samples
    with the watermark applied. Schemes that modify the signal and schemes that add their own
    carrier look the same to the caller.
  * **Detector** (receiver side): `push(block) -> Detection | None`. Returns a Detection
    once, when it has decided on a payload; create a new detector to listen again.

All audio is mono float32 in [-1, 1], in blocks of any length, at the object's `sample_rate`.

Implementations live in `watermark.impls`, usually extend `watermark.base.BaseScheme`, and are registered
with a `watermark.factory.SchemeFactory`. Utilities live in `watermark.util` and `watermark.adapters`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


@dataclass(frozen=True)
class Detection:
    payload: bytes  # payload_bits / 8 bytes
    confidence: float  # scheme-defined score in [0, 1]
    at_sample: int  # samples this detector had consumed when it decided (its own sample rate)
    verified: bool  # True if the scheme checked payload integrity (e.g. CRC); False = best guess
    info: dict[str, Any] = field(default_factory=dict)  # scheme-specific diagnostics


class Embedder(ABC):
    sample_rate: int
    latency: int = 0  # samples between an input sample and its watermarked output

    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Watermark the next block of the outgoing stream; returns len(audio) samples."""


class Detector(ABC):
    sample_rate: int

    @abstractmethod
    def push(self, audio: np.ndarray) -> Detection | None:
        """Feed the next block of received audio; returns a Detection once, when decided."""


@dataclass
class NoOptions:
    """Options type for schemes without tunables."""


class Scheme(ABC):
    """A watermarking method. Constructed as `SchemeClass(options)`, where `options` is an
    instance of the class's `Options` dataclass (the factory builds it from strings)."""

    name: ClassVar[str]
    Options: ClassVar[type] = NoOptions  # dataclass of tunables; every field must have a default
    payload_bits: ClassVar[int]  # multiple of 8
    sample_rate: ClassVar[int]  # native rate of embedder and detector
    # Embedder output is structured in frames of this many native samples, and the detector
    # needs the received stream on the same grid (e.g. AudioSeal = 320). Rate adapters keep
    # every delay they add a multiple of it. 1 = no alignment requirement.
    frame_samples: ClassVar[int] = 1

    options: Any  # instance of Options

    @abstractmethod
    def embedder(self, payload: bytes) -> Embedder:
        """New embedder carrying `payload` (payload_bits / 8 bytes)."""

    @abstractmethod
    def detector(self) -> Detector:
        """New detector."""

    @abstractmethod
    def describe(self) -> str:
        """One-line human-readable description including the active options."""
