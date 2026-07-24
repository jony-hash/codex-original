"""Compatibility shim aliasing application output-layout helpers."""
import sys

from .v1.application import output_layout as _implementation

sys.modules[__name__] = _implementation
