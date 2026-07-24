"""Legacy compatibility facade; canonical v1 one-shot API lives in :mod:`ldm_project.v1`."""
import sys
from .v1 import one_shot as _implementation
sys.modules[__name__] = _implementation
