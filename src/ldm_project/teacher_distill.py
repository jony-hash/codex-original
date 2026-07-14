"""Compatibility shim aliasing teacher distillation stages."""
import sys

from .teacher import distill as _implementation

sys.modules[__name__] = _implementation
