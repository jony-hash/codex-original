"""Deprecated static facade for diagnostic shared helpers.

Candidate modules import explicit canonical owners.  This file remains only for
historical candidate modules that have not yet moved their import spelling.
"""

from __future__ import annotations

from importlib import import_module


_ADMISSION = {"_fixed_control_probe"}
_SYMBOL_MODULES = {
    name: (".admission" if name in _ADMISSION else ".runtime")
    for name in (
        "ConditionalBundle", "IDENTITY_ARCHITECTURE",
        "BALANCED_TRAINING_SCHEDULE_CONTRACT", "CONDITIONAL_CONTROL_METRIC_CONTRACT",
        "PCA_TRANSPORT_ARCHITECTURE", "PCA_TRANSPORT_GENERATOR_CONTRACT",
        "SHORT_SPACING_ARCHITECTURE", "SHORT_SPACING_GENERATOR_CONTRACT",
        "FORMAL_SPACING_ARCHITECTURE", "FORMAL_SPACING_GENERATOR_CONTRACT",
        "NEAR_CONJUGATE_ARCHITECTURE", "NEAR_CONJUGATE_GENERATOR_CONTRACT",
        "EXPANDED_MANIFOLD_ARCHITECTURE",
        "TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE",
        "TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE",
        "_context", "_new_run", "_random_codes", "_split", "_write_evidence",
        "_system_signature", "_load_bundle", "_fixed_control_probe",
    )
}


def __getattr__(name: str):
    try:
        module_name = _SYMBOL_MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __package__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SYMBOL_MODULES))
