"""Lazy compatibility facade for historical conditional runtime imports."""

from __future__ import annotations

from importlib import import_module


def _implementation():
    return import_module(".legacy.runtime", __package__)


def __getattr__(name: str):
    value = getattr(_implementation(), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation())))
