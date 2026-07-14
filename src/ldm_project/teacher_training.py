"""Compatibility shim aliasing teacher training stages."""
import sys

from .teacher import training as _implementation

sys.modules[__name__] = _implementation
