"""Lazy compatibility facade for historical conditional admission imports."""

from __future__ import annotations

from importlib import import_module

_ALIASES = {"conditional_semantic_localization": "_identity_semantic_localization"}


def __getattr__(name: str):
    module = import_module(".legacy.runtime", __package__)
    value = getattr(module, _ALIASES.get(name, name))
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(import_module(".legacy.runtime", __package__))) | set(_ALIASES))
