"""Compatibility facade for the v2 conditional application stages.

The implementation is kept in ``conditional_runtime`` so the public CLI and
older internal imports remain stable while domain facades expose narrower
responsibility groups.
"""

from .conditional_runtime import *  # noqa: F401,F403
