"""Legacy compatibility facade; canonical v1 loss helpers live in :mod:`ldm_project.v1`."""
import sys
from .v1 import training_loss_common as _implementation
sys.modules[__name__] = _implementation
