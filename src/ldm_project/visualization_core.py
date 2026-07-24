"""Legacy compatibility facade; canonical v1 visualization lives in :mod:`ldm_project.v1`."""
import sys
from .v1 import visualization_core as _implementation
sys.modules[__name__] = _implementation
