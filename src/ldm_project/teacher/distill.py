"""Compatibility facade for the teacher distillation workflows.

Canonical scoped imports are available from ``export``, ``filter``,
``readable`` and ``label_audit``.  The legacy namespace is preserved because
older tools intentionally import private audit helpers.
"""
from __future__ import annotations

from . import distill_legacy as _legacy

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

del _name
del _legacy
