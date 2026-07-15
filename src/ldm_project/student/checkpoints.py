"""Student checkpoint contract and compatibility loading entry points."""
from __future__ import annotations

from typing import Any, Mapping

import torch


def validate_one_shot_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    """Reject unknown versions or incomplete payloads before touching model state."""
    from .contracts import ONE_SHOT_MODEL_VERSION

    if payload.get("model_version") != ONE_SHOT_MODEL_VERSION:
        raise ValueError("Incompatible one-shot checkpoint: unexpected model_version.")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("Incompatible one-shot checkpoint: missing model_state_dict.")


def load_one_shot_checkpoint(model: Any, payload: Mapping[str, Any]) -> None:
    """Load a complete checkpoint using the model's registered state-key contract."""
    validate_one_shot_checkpoint_payload(payload)
    _load_one_shot_checkpoint_state(
        model, payload["model_state_dict"], description="one-shot checkpoint"
    )


def load_one_shot_checkpoint_subset(
    model: Any,
    payload: Mapping[str, Any],
    *,
    include_prefixes: tuple[str, ...],
    description: str,
    expected_missing_prefixes: tuple[str, ...] = (),
    expected_skipped_prefixes: tuple[str, ...] = (),
) -> dict[str, int]:
    """Warm-start only explicitly named modules; all omissions remain auditable."""
    validate_one_shot_checkpoint_payload(payload)
    return _load_one_shot_checkpoint_state(
        model,
        payload["model_state_dict"],
        include_prefixes=include_prefixes,
        description=description,
        allow_partial=True,
        expected_missing_prefixes=expected_missing_prefixes,
        expected_skipped_prefixes=expected_skipped_prefixes,
    )


def _phase_slice_compatible_state(
    target: torch.Tensor, value: torch.Tensor
) -> torch.Tensor | None:
    if target.ndim < 1 or value.ndim != target.ndim:
        return None
    if tuple(target.shape[1:]) != tuple(value.shape[1:]):
        return None
    shared_layers = min(int(target.shape[0]), int(value.shape[0]))
    if shared_layers <= 0:
        return None
    sliced = target.detach().clone()
    sliced[:shared_layers].copy_(
        value[:shared_layers].to(device=target.device, dtype=target.dtype)
    )
    return sliced


def _collect_compatible_checkpoint_state(
    model_state: Mapping[str, torch.Tensor],
    checkpoint_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    compatible_state = {}
    skipped_keys = []
    sliced_keys = []
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is not None and tuple(target.shape) == tuple(value.shape):
            compatible_state[key] = value
        elif target is not None and key.endswith(".phase"):
            sliced_value = _phase_slice_compatible_state(target, value)
            if sliced_value is not None:
                compatible_state[key] = sliced_value
                sliced_keys.append(key)
            else:
                skipped_keys.append(key)
        else:
            skipped_keys.append(key)
    return compatible_state, skipped_keys, sliced_keys


def _load_compatible_checkpoint_state(
    model: Any,
    compatible_state: Mapping[str, torch.Tensor],
    *,
    description: str,
) -> tuple[list[str], list[str]]:
    try:
        missing_keys, unexpected_keys = model.load_state_dict(
            compatible_state, strict=False
        )
    except RuntimeError as exc:
        raise ValueError(
            f"Incompatible {description}: state dict does not match model config."
        ) from exc
    return missing_keys, unexpected_keys


def _checkpoint_load_summary(
    compatible_state: Mapping[str, torch.Tensor],
    sliced_keys: list[str],
    skipped_keys: list[str],
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> dict[str, int]:
    return {
        "loaded": len(compatible_state),
        "sliced": len(sliced_keys),
        "skipped": len(skipped_keys),
        "missing": len(missing_keys),
        "unexpected": len(unexpected_keys),
    }


def _print_partial_checkpoint_load(
    description: str, summary: Mapping[str, int]
) -> None:
    print(
        f"partially loaded {description}: "
        f"loaded={summary['loaded']} sliced={summary['sliced']} skipped={summary['skipped']} "
        f"missing={summary['missing']} unexpected={summary['unexpected']}"
    )


def _validate_optical_tolerant_full_checkpoint_load(
    description: str,
    skipped_keys: list[str],
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    optical_prefixes = ("encoder.", "decoder.", "ols_phase_screen", "ols_amplitude_log")
    unexpected_missing = [
        key for key in missing_keys if not key.startswith(optical_prefixes)
    ]
    unexpected_skipped = [
        key for key in skipped_keys if not key.startswith(optical_prefixes)
    ]
    unexpected_loaded = [
        key for key in unexpected_keys if not key.startswith(optical_prefixes)
    ]
    if unexpected_missing or unexpected_skipped or unexpected_loaded:
        raise ValueError(
            f"Incompatible {description}: full checkpoint load requires an exact model match "
            f"(missing={len(unexpected_missing)} unexpected={len(unexpected_loaded)} "
            f"skipped={len(unexpected_skipped)})."
        )


def _validate_exact_checkpoint_load(
    description: str, summary: Mapping[str, int]
) -> None:
    if summary["skipped"] or summary["missing"] or summary["unexpected"]:
        raise ValueError(
            f"Incompatible {description}: full checkpoint load requires an exact model match "
            f"(loaded={summary['loaded']} skipped={summary['skipped']} "
            f"missing={summary['missing']} unexpected={summary['unexpected']})."
        )


def _validate_partial_checkpoint_load(
    description: str,
    include_prefixes: tuple[str, ...] | None,
    expected_missing_prefixes: tuple[str, ...],
    expected_skipped_prefixes: tuple[str, ...],
    skipped_keys: list[str],
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    unexpected_missing = [
        key
        for key in missing_keys
        if include_prefixes is not None
        and key.startswith(include_prefixes)
        and not key.startswith(expected_missing_prefixes)
    ]
    unexpected_skipped = [
        key for key in skipped_keys if not key.startswith(expected_skipped_prefixes)
    ]
    if unexpected_missing or unexpected_skipped or unexpected_keys:
        raise ValueError(
            f"Incompatible {description}: partial checkpoint load found unexpected keys "
            f"(missing={unexpected_missing[:5]} skipped={unexpected_skipped[:5]} "
            f"unexpected={unexpected_keys[:5]})."
        )


def _load_one_shot_checkpoint_state(
    model: Any,
    checkpoint_state: Mapping[str, torch.Tensor],
    *,
    include_prefixes: tuple[str, ...] | None = None,
    description: str = "one-shot checkpoint",
    allow_partial: bool = False,
    expected_missing_prefixes: tuple[str, ...] = (),
    expected_skipped_prefixes: tuple[str, ...] = (),
) -> dict[str, int]:
    model_state = model.state_dict()
    if include_prefixes is None and not allow_partial:
        compatible_state, skipped_keys, sliced_keys = (
            _collect_compatible_checkpoint_state(model_state, checkpoint_state)
        )
        missing_keys, unexpected_keys = _load_compatible_checkpoint_state(
            model, compatible_state, description=description
        )
        _validate_optical_tolerant_full_checkpoint_load(
            description, skipped_keys, missing_keys, unexpected_keys
        )
        summary = _checkpoint_load_summary(
            compatible_state, sliced_keys, skipped_keys, missing_keys, unexpected_keys
        )
        if skipped_keys or missing_keys or unexpected_keys:
            _print_partial_checkpoint_load(description, summary)
        return summary

    filtered_state = {
        key: value
        for key, value in checkpoint_state.items()
        if include_prefixes is None or key.startswith(include_prefixes)
    }
    if not filtered_state:
        raise ValueError(
            f"Incompatible {description}: no checkpoint state entries match prefixes {include_prefixes}."
        )
    compatible_state, skipped_keys, sliced_keys = _collect_compatible_checkpoint_state(
        model_state, filtered_state
    )
    if not compatible_state:
        raise ValueError(
            f"Incompatible {description}: no state dict entries match the current model config."
        )
    missing_keys, unexpected_keys = _load_compatible_checkpoint_state(
        model, compatible_state, description=description
    )
    summary = _checkpoint_load_summary(
        compatible_state, sliced_keys, skipped_keys, missing_keys, unexpected_keys
    )
    if not allow_partial:
        _validate_exact_checkpoint_load(description, summary)
    else:
        _validate_partial_checkpoint_load(
            description,
            include_prefixes,
            expected_missing_prefixes,
            expected_skipped_prefixes,
            skipped_keys,
            missing_keys,
            unexpected_keys,
        )
    if skipped_keys or missing_keys or unexpected_keys:
        _print_partial_checkpoint_load(description, summary)
    return summary


def checkpoint_state_hash(model: Any) -> str:
    """Return a deterministic digest useful for frozen-model diagnostic audits."""
    import hashlib

    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "checkpoint_state_hash",
    "load_one_shot_checkpoint",
    "load_one_shot_checkpoint_subset",
    "validate_one_shot_checkpoint_payload",
]
