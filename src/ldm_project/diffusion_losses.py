"""Compatibility shim aliasing teacher diffusion objectives."""
import sys

from .teacher import diffusion_losses as _implementation

sys.modules[__name__] = _implementation
