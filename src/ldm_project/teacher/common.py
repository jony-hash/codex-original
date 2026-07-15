"""Compatibility facade for teacher distillation helpers.

Workflow implementations live in the scoped export, label_audit, filter and
readable modules. This module only preserves historical imports, including
private helper names used by old diagnostics.
"""
from __future__ import annotations

from . import distill_support, export, filter, label_audit, readable

for _module in (distill_support, export, label_audit, filter, readable):
    for _name in dir(_module):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_module, _name)

del _module
del _name
