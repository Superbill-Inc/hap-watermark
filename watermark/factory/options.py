"""Scheme options from strings: `--opt key=value` pairs -> the scheme's Options dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any


def parse_opts(pairs: list[str] | None) -> dict[str, str]:
    """["a=1", "b=x"] -> {"a": "1", "b": "x"} (for SchemeFactory.create)."""
    out = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--opt expects key=value, got {pair!r}")
        out[key.strip()] = value.strip()
    return out


def options_from_strings(options_cls: type, values: dict[str, str]) -> Any:
    """Build a scheme's Options dataclass from strings, coerced to each field's default type."""
    fields = {f.name: f for f in dataclasses.fields(options_cls)}
    kwargs = {}
    for key, raw in values.items():
        if key not in fields:
            raise ValueError(f"unknown option {key!r}; valid: {', '.join(fields)}")
        kwargs[key] = _coerce(raw, fields[key].default)
    return options_cls(**kwargs)


def _coerce(raw: str, default: Any) -> Any:
    if raw.lower() in ("none", "null", ""):
        return None
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float) or default is None:
        try:
            return float(raw)
        except ValueError:
            return raw
    if isinstance(default, Path):
        return Path(raw)
    return raw
