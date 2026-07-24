"""Stable generator import facade.

Concrete v2 conditional generators live in ``conditional_generator``. The
legacy basis generator remains re-exported here for checkpoint compatibility.
"""

from .conditional_generator import *  # noqa: F401,F403
