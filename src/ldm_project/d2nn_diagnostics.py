"""Compatibility shim for neutral D2NN reports."""
from .evaluation import d2nn as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
