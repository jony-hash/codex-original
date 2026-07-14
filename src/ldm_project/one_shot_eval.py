"""Compatibility shim aliasing the formal student evaluation module."""
import sys

from .student import evaluation as _implementation

sys.modules[__name__] = _implementation
