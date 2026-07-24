"""Compatibility shim aliasing teacher distillation stages."""
import sys

from .v1.teacher import distill_legacy as _implementation

sys.modules[__name__] = _implementation
