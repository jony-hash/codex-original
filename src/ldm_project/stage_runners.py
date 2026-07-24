"""Legacy compatibility facade; canonical v1 runners live in :mod:`ldm_project.v1`."""
import sys
from .v1 import stage_runners as _implementation
sys.modules[__name__] = _implementation
