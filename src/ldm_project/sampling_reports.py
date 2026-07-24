"""Legacy compatibility facade; canonical v1 reports live in :mod:`ldm_project.v1`."""
import sys
from .v1 import sampling_reports as _implementation
sys.modules[__name__] = _implementation
