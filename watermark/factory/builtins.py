"""Implementations shipped in watermark.impls, as lazy "module:Class" paths (AudioSeal pulls in torch)."""

BUILTIN_IMPLS = {
    "dsss": "watermark.impls.dsss:DsssScheme",
    "audioseal": "watermark.impls.audioseal:AudioSealScheme",
}
