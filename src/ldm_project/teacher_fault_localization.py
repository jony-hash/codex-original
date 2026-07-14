"""Compatibility shim aliasing teacher fault-localization probes."""
import sys

from .teacher import fault_localization as _implementation

sys.modules[__name__] = _implementation
