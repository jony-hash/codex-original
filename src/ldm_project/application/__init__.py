"""Legacy package facade for :mod:`ldm_project.v1.application`."""
import sys
from ..v1 import application as _implementation
sys.modules[__name__] = _implementation
