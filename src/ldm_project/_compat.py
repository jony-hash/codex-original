"""Import-only compatibility helpers for pre-v1 namespace module paths."""
from __future__ import annotations

from importlib import import_module
from pkgutil import walk_packages
import sys


def alias_package(legacy_name: str, canonical_name: str) -> None:
    """Make every legacy package/submodule name resolve to its canonical module object."""
    package = import_module(canonical_name)
    aliases = [(legacy_name, canonical_name)]
    if hasattr(package, "__path__"):
        aliases.extend(
            (f"{legacy_name}{item.name[len(canonical_name):]}", item.name)
            for item in walk_packages(package.__path__, f"{canonical_name}.")
        )
    for old_name, new_name in aliases:
        sys.modules[old_name] = import_module(new_name)
