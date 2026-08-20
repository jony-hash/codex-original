"""Formal optics exports with lazy diagnostic compatibility exports."""

from __future__ import annotations

from importlib import import_module

from .d2nn import FiniteCropPhaseStack, NearConjugateFiniteCropPhaseStack
from .devices import IdealAmplitudeSLM, IdentityOLS, identity_ols
from .fiber_operator import FiberArrayOLS, FiberArraySpec, FiberTransportEnsembleOLS, lp01_radial_profile
from .fiber_pool import FiberResearchSpec, FiberTransportPool
from .operator import OpticalLatentOperator
from .propagation import angular_spectrum_propagate, reference_angular_spectrum_propagate
from .relays import ABCDMatrix, CoherentImagingRelay, NearConjugateRelay, reference_b0_imaging, reference_near_conjugate_lens_relay
from .system import FormalAmplitudeFiniteCropIdentitySystem, FormalForwardResult

__all__ = [
    "ABCDMatrix",
    "CoherentImagingRelay",
    "NearConjugateRelay",
    "FiniteCropPhaseStack",
    "NearConjugateFiniteCropPhaseStack",
    "FormalAmplitudeFiniteCropIdentitySystem",
    "FormalForwardResult",
    "FiberModeBankV2",
    "FiberOLSStateV2",
    "IdealAmplitudeSLM",
    "LightGenAmplitudeFiberOLSSystemV2",
    "LightGenForwardResultV2",
    "angular_spectrum_propagate",
    "identity_ols",
    "IdentityOLS",
    "OpticalLatentOperator",
    "FiberArrayOLS",
    "FiberArraySpec",
    "FiberTransportEnsembleOLS",
    "FiberResearchSpec",
    "FiberTransportPool",
    "lp01_radial_profile",
    "gaussian_single_mode_bank",
    "project_single_mode_fibers",
    "reference_project_single_mode_fibers",
    "reemit_single_mode_fibers",
    "reference_angular_spectrum_propagate",
    "reference_b0_imaging",
    "reference_near_conjugate_lens_relay",
]

_DIAGNOSTIC_EXPORTS = {
    "LightGenAmplitudeFiberOLSSystemV2": (".lightgen", "LightGenAmplitudeFiberOLSSystemV2"),
    "LightGenForwardResultV2": (".lightgen", "LightGenForwardResultV2"),
    "FiberModeBankV2": (".fiber_ols", "FiberModeBankV2"),
    "FiberOLSStateV2": (".fiber_ols", "FiberOLSStateV2"),
    "gaussian_single_mode_bank": (".fiber_ols", "gaussian_single_mode_bank"),
    "project_single_mode_fibers": (".fiber_ols", "project_single_mode_fibers"),
    "reference_project_single_mode_fibers": (".fiber_ols", "reference_project_single_mode_fibers"),
    "reemit_single_mode_fibers": (".fiber_ols", "reemit_single_mode_fibers"),
}


def __getattr__(name: str):
    """Resolve diagnostic-only optics only for legacy callers."""
    target = _DIAGNOSTIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value
