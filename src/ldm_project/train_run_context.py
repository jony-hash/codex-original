"""Compatibility shim aliasing application run metadata helpers."""
import sys

from .v1.application import run_context as _implementation

sys.modules[__name__] = _implementation
