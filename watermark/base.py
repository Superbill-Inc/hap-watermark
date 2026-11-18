"""Convenience base class for Scheme implementations."""

from __future__ import annotations

import dataclasses
from typing import Any

from .spi import Scheme


class BaseScheme(Scheme):
    """Stores options, validates payload size, and describes itself from its options."""

    def __init__(self, options: Any = None):
        self.options = options if options is not None else self.Options()

    @property
    def payload_bytes(self) -> int:
        return self.payload_bits // 8

    def check_payload(self, payload: bytes) -> bytes:
        if len(payload) != self.payload_bytes:
            raise ValueError(f"{self.name} carries {self.payload_bits}-bit payloads, got {len(payload) * 8} bits")
        return payload

    def describe(self) -> str:
        opts = ", ".join(f"{f.name}={getattr(self.options, f.name)}" for f in dataclasses.fields(self.options))
        return f"{self.name}: {self.payload_bits}-bit payload @ {self.sample_rate} Hz" + (f" ({opts})" if opts else "")
