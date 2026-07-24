"""Legacy package facade for :mod:`ldm_project.v1.teacher`."""
import sys
from ..v1 import teacher as _implementation
sys.modules[__name__] = _implementation
