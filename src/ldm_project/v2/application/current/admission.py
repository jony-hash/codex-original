# ruff: noqa: F401
"""Formal lifecycle bindings for neutral conditional admission workflows."""
from __future__ import annotations

from .._conditional_admission_shared import ConditionalRuntimeBindings, bind_admission
from . import runtime as _runtime

globals().update(bind_admission(ConditionalRuntimeBindings(vars(_runtime))))
