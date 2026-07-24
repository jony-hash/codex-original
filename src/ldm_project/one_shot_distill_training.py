"""Compatibility shim aliasing the formal student distillation loop."""
import sys

from .v1.student import training as _implementation

sys.modules[__name__] = _implementation
