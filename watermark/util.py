"""Payload helpers for callers of the watermark SPI."""

from __future__ import annotations

from .spi import Scheme

_DEFAULT_PATTERN = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF")


def payload_from_hex(scheme: Scheme, text: str | None) -> bytes:
    """Parse a hex payload sized for `scheme`; None -> a fixed default pattern of that size."""
    size = scheme.payload_bits // 8
    if text is None:
        return _DEFAULT_PATTERN[:size]
    h = text.lower().removeprefix("0x")
    if len(h) > 2 * size:
        raise ValueError(f"{scheme.name} carries {scheme.payload_bits}-bit payloads; {text!r} is longer")
    return bytes.fromhex(h.zfill(2 * size))


def payload_hex(payload: bytes) -> str:
    return "0x" + payload.hex().upper()
