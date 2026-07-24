"""Legacy compatibility facade; canonical v1 stages live in :mod:`ldm_project.v1`."""
import sys
from .v1 import one_shot_stages as _implementation
sys.modules[__name__] = _implementation
