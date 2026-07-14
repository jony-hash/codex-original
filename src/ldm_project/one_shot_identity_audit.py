"""Compatibility shim for diagnostic identity-audit probes."""
from .diagnostics import identity_audit as _implementation

globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
