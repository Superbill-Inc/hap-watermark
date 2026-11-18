"""Factory: registry of scheme implementations, options parsing, and the default factory.

    from watermark.factory import default_factory, parse_opts
    scheme = default_factory.create("dsss", parse_opts(["level_dbfs=-57"]))
"""

from .builtins import BUILTIN_IMPLS
from .options import options_from_strings, parse_opts
from .scheme_factory import SchemeFactory

default_factory = SchemeFactory.with_builtins()

__all__ = ["BUILTIN_IMPLS", "SchemeFactory", "default_factory", "options_from_strings", "parse_opts"]
