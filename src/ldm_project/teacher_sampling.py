"""Compatibility shim aliasing teacher sampling stages."""
import sys

from .teacher import sampling as _implementation

sys.modules[__name__] = _implementation
