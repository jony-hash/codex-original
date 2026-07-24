"""Legacy package facade for :mod:`ldm_project.v1.optics`."""
import sys
from ..v1 import optics as _implementation
sys.modules[__name__] = _implementation
