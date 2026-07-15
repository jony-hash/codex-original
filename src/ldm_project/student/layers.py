"""Legacy import shim for student layer symbols.

New formal code imports generator and optics directly.  Diagnostic helpers
remain under ``student.experimental``; this module intentionally contains no
implementation of its own.
"""
from __future__ import annotations

from .generator import ElectronicGeneratorE
from .experimental.layers import (
    PhaseAmplitudeDiffractiveStack,
    _angular_spectrum_propagate_center_padded,
    _angular_spectrum_propagate_center_padded_with_diagnostics,
    _apply_complex_ols,
    _center_crop_complex_field,
    _center_embed_complex_field,
    _complex_field_energy,
    _flatten_complex_field,
    _resize_complex_field,
    _resize_real_intensity,
    _vector_to_complex_field,
)
from ..optics.d2nn import PhaseOnlyDiffractiveStack
from ..optics.devices import IdealAmplitudeSLMModel

__all__ = [
    "ElectronicGeneratorE",
    "IdealAmplitudeSLMModel",
    "PhaseAmplitudeDiffractiveStack",
    "PhaseOnlyDiffractiveStack",
    "_angular_spectrum_propagate_center_padded",
    "_angular_spectrum_propagate_center_padded_with_diagnostics",
    "_apply_complex_ols",
    "_center_crop_complex_field",
    "_center_embed_complex_field",
    "_complex_field_energy",
    "_flatten_complex_field",
    "_resize_complex_field",
    "_resize_real_intensity",
    "_vector_to_complex_field",
]
