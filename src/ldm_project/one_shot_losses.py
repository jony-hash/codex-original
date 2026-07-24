"""Compatibility shim for formal and experimental student objectives."""
from .v1.student import objectives as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
