"""Compatibility shim aliasing teacher sampling stages."""
import sys

from .v1.teacher import sampling as _implementation

sys.modules[__name__] = _implementation
