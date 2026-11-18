"""Watermark scheme implementations (one module per scheme, registered in watermark.factory.builtins).

Naming: module watermark/impls/<name>.py defines <Name>Scheme (registered as "<name>") and
<Name>Options; its Embedder/Detector implementations are private (_<Name>Embedder,
_<Name>Detector) and are created only through the scheme.

Modules are imported lazily by the factory, so e.g. `dsss` never loads torch.
"""
