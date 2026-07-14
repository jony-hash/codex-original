"""Compatibility shim for diagnostic D2NN visualizations."""
from .diagnostics import d2nn_visualization as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
