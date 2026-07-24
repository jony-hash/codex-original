"""Legacy package facade for :mod:`ldm_project.v1.diagnostics`."""
import sys
from ..v1 import diagnostics as _implementation
sys.modules[__name__] = _implementation
