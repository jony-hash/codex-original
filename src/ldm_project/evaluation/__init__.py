"""Legacy package facade for :mod:`ldm_project.v1.evaluation`."""
import sys
from ..v1 import evaluation as _implementation
sys.modules[__name__] = _implementation
