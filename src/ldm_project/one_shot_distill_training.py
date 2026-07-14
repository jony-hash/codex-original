"""Compatibility shim aliasing the formal student distillation loop."""
import sys

from .student import training as _implementation

sys.modules[__name__] = _implementation
