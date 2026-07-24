"""Compatibility shim aliasing teacher diffusion objectives."""
import sys

from .v1.teacher import diffusion_losses as _implementation

sys.modules[__name__] = _implementation
