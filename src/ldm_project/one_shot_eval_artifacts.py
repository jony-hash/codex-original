"""Compatibility shim for student artifact writers."""
from .student import artifacts as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
