"""Compatibility shim aliasing teacher training utilities."""
import sys

from .v1.teacher import training_utils as _implementation

sys.modules[__name__] = _implementation
