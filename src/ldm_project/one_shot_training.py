"""Compatibility shim for baseline direct-MNIST student training."""
from .v1.student import baseline_training as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
