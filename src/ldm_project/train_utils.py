"""Legacy compatibility facade; canonical v1 utilities live in :mod:`ldm_project.v1`."""
import sys
from .v1 import train_utils as _implementation
sys.modules[__name__] = _implementation
