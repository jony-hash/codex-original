"""Compatibility shim aliasing application stage dispatch."""
import sys

from .v1.application import stages as _implementation

sys.modules[__name__] = _implementation
