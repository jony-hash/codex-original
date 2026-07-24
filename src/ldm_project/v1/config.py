"""Public configuration facade.

The import path remains ``ldm_project.config`` for CLI/checkpoint
compatibility.  Implementation ownership is under ``configuration`` so schema,
loading and validation can be split without changing YAML semantics.
"""
from __future__ import annotations

from .configuration import legacy as _legacy

# Preserve both public dataclasses/loaders and historical private constants used
# by compatibility callers.  New code should import the scoped modules below.
for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

del _name
del _legacy
