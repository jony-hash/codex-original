"""Legacy package facade for :mod:`ldm_project.v1.configuration`."""
import sys
from ..v1 import configuration as _implementation
sys.modules[__name__] = _implementation
