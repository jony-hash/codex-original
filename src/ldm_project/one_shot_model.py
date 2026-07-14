"""Compatibility shim for the formal student model."""
from .student import model as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
