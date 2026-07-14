"""Compatibility shim for student optical layers."""
from .student import layers as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
