"""Compatibility shim aliasing teacher diffusion sampling."""
import sys

from .teacher import diffusion_sampling as _implementation

sys.modules[__name__] = _implementation
