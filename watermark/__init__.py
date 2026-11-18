"""Pluggable audio watermarking.

    watermark.spi       interfaces: Scheme, Embedder, Detector, Detection
    watermark.factory   SchemeFactory (register / create), options parsing, `default_factory` with the built-ins
    watermark.base      BaseScheme: convenience base for implementations
    watermark.impls     implementations: dsss, audioseal
    watermark.adapters  at_rate(): run a scheme at another pipeline sample rate
    watermark.util      payload helpers

    import watermark
    scheme = watermark.default_factory.create("dsss", {"level_dbfs": "-57"})
    embedder = watermark.at_rate(scheme.embedder(payload), 8000, scheme)
    detector = watermark.at_rate(scheme.detector(), 8000, scheme)
"""

from .adapters import at_rate
from .base import BaseScheme
from .factory import SchemeFactory, default_factory
from .spi import Detection, Detector, Embedder, Scheme

__all__ = ["BaseScheme", "Detection", "Detector", "Embedder", "Scheme", "SchemeFactory", "at_rate", "default_factory"]
