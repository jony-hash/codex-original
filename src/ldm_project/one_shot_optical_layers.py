"""Compatibility shim for student optical layers."""
from .v1.student import layers as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
