"""Legacy compatibility facade; canonical v1 teacher stages live in :mod:`ldm_project.v1`."""
import sys
from .v1 import teacher_stages as _implementation
sys.modules[__name__] = _implementation
