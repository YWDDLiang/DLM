"""Isolated crystal-generation runtime package.

This file intentionally mirrors the package marker in the project source so
the frozen runtime wins import resolution even when an older shared checkout
also exposes a ``crystal_dlm`` package.
"""

__all__: list[str] = []
