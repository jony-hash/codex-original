"""Stable, lazy generator import facade.

The formal identity generator is independent of legacy capacity artifacts.
Historical generator names remain available only when a legacy caller asks for
them explicitly.
"""

from __future__ import annotations

from importlib import import_module

_FORMAL = {
    "IdentityAnchoredConditionalAmplitudeGenerator",
    "ConditionalIdentityLogitAmplitudeGenerator",
}
_LEGACY = {
    "GeneratorDiagnostics",
    "LabelPrototypeBasisGenerator",
    "ConditionalGlobalAmplitudeGenerator",
    "ConditionalClassConditionedModesGenerator",
    "ConditionalParityStabilizedModesGenerator",
    "ConditionalMultiAnchorLocalModesGenerator",
    "ConditionalPCATransportCodesignGenerator",
    "ConditionalPCASelfImagingTransportGenerator",
    "ConditionalNonlinearExpandedManifoldGenerator",
    "ConditionalTransportAlignedManifoldGenerator",
    "ConditionalTransportAlignedPhaseCapacityGenerator",
}


def __getattr__(name: str):
    if name in _FORMAL:
        module = import_module(".identity_generator", __package__)
    elif name in _LEGACY:
        module = import_module(".conditional_generator", __package__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = sorted(_FORMAL | _LEGACY)
