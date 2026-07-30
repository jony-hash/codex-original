"""Deprecated import shim for the v7.1 transport-manifold diagnostic."""
from __future__ import annotations
from importlib import import_module

def __getattr__(name: str):
    value = getattr(import_module(".diagnostic.transport_aligned_manifold", __package__), name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(import_module(".diagnostic.transport_aligned_manifold", __package__))))
