"""SchemeFactory: register watermark scheme implementations by name and instantiate them."""

from __future__ import annotations

import importlib
from typing import Any

from ..spi import Scheme
from .options import options_from_strings


class SchemeFactory:
    """Registry of Scheme implementations.

        factory = SchemeFactory.with_builtins()
        factory.register("mine", MyScheme)                 # a class
        factory.register("other", "pkg.module:OtherScheme")  # imported on first use
        scheme = factory.create("dsss", {"level_dbfs": "-57"})
    """

    def __init__(self):
        self._schemes: dict[str, type[Scheme] | str] = {}

    @classmethod
    def with_builtins(cls) -> SchemeFactory:
        from .builtins import BUILTIN_IMPLS

        factory = cls()
        for name, target in BUILTIN_IMPLS.items():
            factory.register(name, target)
        return factory

    def register(self, name: str, scheme: type[Scheme] | str) -> None:
        """Register a Scheme subclass, or a "module:Class" path imported on first use."""
        if not isinstance(scheme, str) and not (isinstance(scheme, type) and issubclass(scheme, Scheme)):
            raise TypeError(f"{scheme!r} is not a Scheme subclass or a 'module:Class' path")
        self._schemes[name] = scheme

    def names(self) -> list[str]:
        return sorted(self._schemes)

    def scheme_class(self, name: str) -> type[Scheme]:
        try:
            target = self._schemes[name]
        except KeyError:
            raise ValueError(f"unknown scheme {name!r}; available: {', '.join(self.names())}") from None
        if isinstance(target, str):
            module, _, attr = target.partition(":")
            target = getattr(importlib.import_module(module), attr)
            if not (isinstance(target, type) and issubclass(target, Scheme)):
                raise TypeError(f"{self._schemes[name]} is not a Scheme subclass")
            self._schemes[name] = target
        return target

    def create(self, name: str, options: dict[str, str] | Any = None) -> Scheme:
        """Instantiate a scheme. `options`: an Options instance, or strings by field name
        (e.g. from `--opt key=value`), coerced to each field's type."""
        cls = self.scheme_class(name)
        if options is None or isinstance(options, dict):
            options = options_from_strings(cls.Options, options or {})
        return cls(options)
