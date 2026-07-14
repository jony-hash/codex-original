"""Compatibility shim aliasing teacher first-stage objectives."""
import sys

from .teacher import first_stage_losses as _implementation

sys.modules[__name__] = _implementation
