"""Compatibility facade for teacher distillation workflows.

Algorithms live in scoped canonical modules (currently common.py while the workflow
functions are being split).  This module preserves the historical namespace and the
legacy monkeypatch point used by old runners.
"""
from __future__ import annotations

from . import common as _common

# Keep every historical public and private helper available to old tools.
for _name in dir(_common):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_common, _name)


def _with_legacy_prototype_builder(function, *args, **kwargs):
    """Route legacy prototype monkeypatches into the canonical implementation."""
    builder = globals().get("_build_mnist_label_prototype_bank")
    original = getattr(_common, "_build_mnist_label_prototype_bank", None)
    if builder is not None and original is not None and builder is not original:
        _common._build_mnist_label_prototype_bank = builder
    try:
        return function(*args, **kwargs)
    finally:
        if original is not None:
            _common._build_mnist_label_prototype_bank = original


def run_teacher_distill_label_strict_filter(*args, **kwargs):
    return _with_legacy_prototype_builder(_common.run_teacher_distill_label_strict_filter, *args, **kwargs)


def run_teacher_distill_readable_split(*args, **kwargs):
    return _with_legacy_prototype_builder(_common.run_teacher_distill_readable_split, *args, **kwargs)


del _name

