"""Legacy compatibility facade; canonical v1 config lives in :mod:`ldm_project.v1`."""
import sys
from .v1 import config as _implementation
sys.modules[__name__] = _implementation
