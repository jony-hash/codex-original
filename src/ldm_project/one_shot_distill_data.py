"""Compatibility shim aliasing the neutral distillation payload module."""
import sys

from .distillation import dataset as _implementation

sys.modules[__name__] = _implementation
