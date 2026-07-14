"""Compatibility shim for formal and experimental student objectives."""
from .student import objectives as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
