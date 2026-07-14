"""Compatibility shim for neutral code sampling and display keys."""
from .generation import codes as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
