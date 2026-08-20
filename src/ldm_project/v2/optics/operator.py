"""Replaceable complex-field optical operator boundary.

The formal v2 system uses :func:`identity_ols`; research operators implement
the same small callable contract without adding a fiber option to formal
configuration.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts import ComplexField


@runtime_checkable
class OpticalLatentOperator(Protocol):
    """A linear optical latent operator with explicit physical planes."""

    def __call__(self, field: ComplexField) -> ComplexField: ...
