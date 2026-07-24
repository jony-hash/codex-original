"""Legacy package facade for :mod:`ldm_project.v1.student`."""
import sys
from ..v1 import student as _implementation
sys.modules[__name__] = _implementation
