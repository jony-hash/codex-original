"""Compatibility shim for diagnostic D2NN reports."""
from .diagnostics import d2nn as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
