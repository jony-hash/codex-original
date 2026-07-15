"""Student checkpoint contract and compatibility loading entry points."""
from __future__ import annotations

from typing import Any, Mapping


def validate_one_shot_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    """Reject unknown versions or incomplete payloads before touching model state."""
    from .model import ONE_SHOT_MODEL_VERSION

    if payload.get("model_version") != ONE_SHOT_MODEL_VERSION:
        raise ValueError("Incompatible one-shot checkpoint: unexpected model_version.")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("Incompatible one-shot checkpoint: missing model_state_dict.")


def load_one_shot_checkpoint(model: Any, payload: Mapping[str, Any]) -> None:
    """Load a complete checkpoint using the model's registered state-key contract."""
    validate_one_shot_checkpoint_payload(payload)
    from . import model as model_module

    model_module._load_one_shot_checkpoint_state(
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
    from . import model as model_module

    return model_module._load_one_shot_checkpoint_state(
        model,
        payload["model_state_dict"],
        include_prefixes=include_prefixes,
        description=description,
        allow_partial=True,
        expected_missing_prefixes=expected_missing_prefixes,
        expected_skipped_prefixes=expected_skipped_prefixes,
    )


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
