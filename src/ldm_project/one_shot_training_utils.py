"""Compatibility shim for student training utilities."""
from .v1.student import training_utils as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
