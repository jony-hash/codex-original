"""Compatibility facade for the historical v2 CLI module.

The public v2 dispatcher owns parsing and routing.  This module retains the
old import path without maintaining a second 1,400-line parser and handler
registry.  Historical helper symbols resolve lazily from their owning legacy
implementation module.
"""

from __future__ import annotations

from importlib import import_module


_LEGACY_MODULES = (
    ".teacher",
    ".prototypes",
    ".geometry",
    ".templates",
)


def _parser():
    """Return the canonical side-effect-free v2 parser."""
    return import_module("..cli_schema", __package__)._parser()


def main(argv: list[str] | None = None) -> int:
    """Run the canonical dispatcher while preserving the legacy entrypoint."""
    return int(import_module("..cli", __package__).main(argv))


def __getattr__(name: str):
    for module_name in _LEGACY_MODULES:
        module = import_module(module_name, __package__)
        try:
            value = getattr(module, name)
        except AttributeError:
            continue
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"main", "_parser"})
