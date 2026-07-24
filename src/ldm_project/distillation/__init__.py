"""Legacy package facade for :mod:`ldm_project.v1.distillation`."""
import sys
from ..v1 import distillation as _implementation
sys.modules[__name__] = _implementation
