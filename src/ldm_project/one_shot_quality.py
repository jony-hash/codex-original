"""Compatibility shim for student quality gates."""
from .student import quality as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
