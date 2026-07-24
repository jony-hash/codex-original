"""Legacy compatibility facade; canonical v1 data lives in :mod:`ldm_project.v1`."""
import sys
from .v1 import data as _implementation
sys.modules[__name__] = _implementation
