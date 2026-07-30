"""Deprecated import shim for the diagnostic optical-feasibility audit."""

from __future__ import annotations

from importlib import import_module


def __getattr__(name: str):
    module = import_module(".diagnostic.optical_feasibility", __package__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(import_module(".diagnostic.optical_feasibility", __package__))))
