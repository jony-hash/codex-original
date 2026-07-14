"""Compatibility shim aliasing teacher training utilities."""
import sys

from .teacher import training_utils as _implementation

sys.modules[__name__] = _implementation
