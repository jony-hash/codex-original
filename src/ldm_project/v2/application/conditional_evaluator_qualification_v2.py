"""Deprecated import shim for v2 diagnostic evaluator qualification."""
from __future__ import annotations
from importlib import import_module

def __getattr__(name: str):
    value = getattr(import_module(".diagnostic.evaluator_qualification_v2", __package__), name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(import_module(".diagnostic.evaluator_qualification_v2", __package__))))
