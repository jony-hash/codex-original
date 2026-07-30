"""Lazy public exports for the v2 student namespace."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CheckpointManifestV2": (".checkpoints", "CheckpointManifestV2"),
    "WarmStartManifestV2": (".checkpoints", "WarmStartManifestV2"),
    "load_exact_checkpoint": (".checkpoints", "load_exact_checkpoint"),
    "save_exact_checkpoint": (".checkpoints", "save_exact_checkpoint"),
    "ConditionalGlobalAmplitudeGenerator": (".generator", "ConditionalGlobalAmplitudeGenerator"),
    "ConditionalIdentityLogitAmplitudeGenerator": (".generator", "ConditionalIdentityLogitAmplitudeGenerator"),
    "IdentityAnchoredConditionalAmplitudeGenerator": (".generator", "IdentityAnchoredConditionalAmplitudeGenerator"),
    "LabelPrototypeBasisGenerator": (".generator", "LabelPrototypeBasisGenerator"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value
