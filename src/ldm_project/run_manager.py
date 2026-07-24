"""Compatibility shim aliasing application run management."""
import sys

from .v1.application import run_manager as _implementation

sys.modules[__name__] = _implementation
