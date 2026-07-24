"""Compatibility shim aliasing teacher first-stage objectives."""
import sys

from .v1.teacher import first_stage_losses as _implementation

sys.modules[__name__] = _implementation
