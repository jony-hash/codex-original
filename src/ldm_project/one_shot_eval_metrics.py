"""Compatibility shim for student evaluation metrics."""
from .student import metrics as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
