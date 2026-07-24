"""Compatibility shim aliasing teacher fault-localization probes."""
import sys

from .v1.teacher import fault_localization as _implementation

sys.modules[__name__] = _implementation
