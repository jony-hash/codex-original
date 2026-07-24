"""Legacy package facade for :mod:`ldm_project.v1.generation`."""
import sys
from ..v1 import generation as _implementation
sys.modules[__name__] = _implementation
