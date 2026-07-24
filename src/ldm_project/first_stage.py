"""Compatibility shim aliasing the teacher first-stage implementation."""
import sys

from .v1.teacher import first_stage as _implementation

sys.modules[__name__] = _implementation
