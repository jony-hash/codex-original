"""Compatibility shim for student visualizations."""
from .student import visualization as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
