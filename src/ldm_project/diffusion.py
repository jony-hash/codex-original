"""Compatibility shim aliasing the teacher diffusion implementation."""
import sys

from .v1.teacher import diffusion as _implementation

sys.modules[__name__] = _implementation
