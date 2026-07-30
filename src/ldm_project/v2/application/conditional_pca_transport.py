"""Deprecated import shim for the v6 PCA transport diagnostic."""
from __future__ import annotations
from importlib import import_module

def __getattr__(name: str):
    value = getattr(import_module(".diagnostic.pca_transport", __package__), name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(import_module(".diagnostic.pca_transport", __package__))))
