"""Compatibility shim aliasing teacher training stages."""
import sys

from .v1.teacher import training as _implementation

sys.modules[__name__] = _implementation
