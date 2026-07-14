"""Compatibility shim aliasing the application CLI."""
import sys

from .application import cli as _implementation

sys.modules[__name__] = _implementation
