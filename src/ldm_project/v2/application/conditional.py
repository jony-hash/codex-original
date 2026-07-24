"""Compatibility facade for the v2 conditional application stages.

The implementation is kept in ``conditional_runtime`` so the public CLI and
older internal imports remain stable while domain facades expose narrower
responsibility groups.
"""

from .conditional_runtime import *  # noqa: F401,F403
from .conditional_runtime import _first_hundred_schedule_hash, _identity_isolation_classification, _require_current_acceptance, _semantic_localization_classification  # noqa: F401

__all__ = [
    name for name in globals()
    if not name.startswith("__")
]
