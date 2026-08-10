"""Strict v2 checkpoints; migrations are explicit and never implicit full loads."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..evidence import (
    ArtifactRefV2,
    atomic_torch_save,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)


def _load_weights_only(
    path: str | Path, *, expected_sha256: str | None = None
) -> Mapping[str, Any]:
    """Load a v2 checkpoint without executing arbitrary pickle globals."""
    try:
        payload = safe_torch_load(path, expected_sha256=expected_sha256)
    except ValueError as exc:
        raise ValueError("v2 checkpoint is not a safe weights-only payload") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("invalid v2 checkpoint payload")
    return payload


def _validate_state_dict_payload(state: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise ValueError("v2 checkpoint state_dict must contain only named tensors")
    return state


def state_signature(state: Mapping[str, torch.Tensor]) -> str:
    descriptor = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(state.items())
    }
    return canonical_sha256(descriptor)


def state_content_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor_sha256(value).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointManifestV2:
    schema_version: int
    system_kind: str
    boundary_contract: str
    system_signature: str
    config_hash: str
    dataset_hash: str
    teacher_manifest_hash: str | None
    calibration_hash: str
    basis_hash: str
    state_signature: str
    state_content_hash: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def save_exact_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    manifest: CheckpointManifestV2,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    state = model.state_dict()
    if (
        manifest.schema_version != 2
        or manifest.state_signature != state_signature(state)
        or manifest.state_content_hash != state_content_hash(state)
    ):
        raise ValueError("checkpoint manifest does not exactly describe model state")
    payload = {
        "manifest": manifest.to_dict(),
        "state_dict": state,
        "extra": dict(extra or {}),
    }
    atomic_torch_save(path, payload)


def load_exact_checkpoint(
    path: str | Path, model: torch.nn.Module, expected: CheckpointManifestV2
) -> Mapping[str, Any]:
    payload = _load_weights_only(path)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("manifest"), dict)
        or not isinstance(payload.get("state_dict"), dict)
    ):
        raise ValueError("invalid v2 checkpoint payload")
    actual = CheckpointManifestV2(**payload["manifest"])
    if actual != expected:
        raise ValueError(
            "v2 checkpoint manifest does not match the requested formal system"
        )
    state = _validate_state_dict_payload(payload["state_dict"])
    if (
        state_signature(state) != actual.state_signature
        or state_content_hash(state) != actual.state_content_hash
        or state_signature(model.state_dict()) != actual.state_signature
    ):
        raise ValueError("v2 checkpoint state signature is not exact")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError("strict v2 checkpoint load found state mismatch")
    return payload.get("extra", {})


def checkpoint_file_hash(path: str | Path) -> str:
    return file_sha256(path)


@dataclass(frozen=True)
class WarmStartManifestV2:
    schema_version: int
    source_checkpoint: ArtifactRefV2
    source_system_signature: str
    target_system_signature: str
    loaded_key_signatures: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_checkpoint"] = self.source_checkpoint.to_dict()
        return payload


def load_prototype_warm_start(
    path: str | Path, model: torch.nn.Module, *, expected_system_signature: str
) -> WarmStartManifestV2:
    """Load only the auditable prototype/template and phase state boundary."""
    payload = _load_weights_only(path)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    manifest_payload = payload.get("manifest") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not isinstance(manifest_payload, dict):
        raise ValueError("prototype checkpoint is invalid")
    source_manifest = CheckpointManifestV2(**manifest_payload)
    if source_manifest.system_signature != expected_system_signature:
        raise ValueError(
            "prototype checkpoint system signature does not match warm-start target"
        )
    allowed = (
        "generator.template_logits",
        "system.encoder.phase",
        "system.decoder.phase",
    )
    target = model.state_dict()
    loaded: dict[str, str] = {}
    for key in allowed:
        if (
            key not in state
            or key not in target
            or state[key].shape != target[key].shape
        ):
            raise ValueError(f"prototype warm-start lacks compatible {key}")
        target[key].copy_(state[key])
        loaded[key] = state_signature({key: state[key]})
    model.load_state_dict(target, strict=True)
    return WarmStartManifestV2(
        schema_version=2,
        source_checkpoint=ArtifactRefV2(
            str(Path(path).resolve()), file_sha256(path), "prototype_checkpoint"
        ),
        source_system_signature=source_manifest.system_signature,
        target_system_signature=expected_system_signature,
        loaded_key_signatures=loaded,
    )


@dataclass(frozen=True)
class ConditionalCheckpointManifestV2:
    """Exact checkpoint contract for the teacher-free conditional architecture."""

    schema_version: int
    architecture_kind: str
    system_signature: str
    config_hash: str
    dataset_split_manifest_hash: str
    evaluator_checkpoint_hash: str
    exposure_scale: float
    state_signature: str
    state_content_hash: str
    # These fields are optional only for the historical v1 contract.  A v2
    # checkpoint must carry all of them; keeping defaults here lets the legacy
    # reader continue to inspect old manifests without a silent migration.
    generator_contract_hash: str | None = None
    metric_contract_hash: str | None = None
    training_schedule_hash: str | None = None
    initialization_seed: int | None = None
    initialization_state_hash: str | None = None
    # v6 is the first conditional architecture whose electronic generator is
    # derived from a content-addressed physical basis.  These optional fields
    # keep every older manifest readable while making v6 loads exact.
    config_contract_hash: str | None = None
    pca_basis_hash: str | None = None
    lowpass_contract_hash: str | None = None
    scale_contract_hash: str | None = None
    route_contract_hash: str | None = None
    semantic_source_checkpoint_hash: str | None = None
    safe_scale_evidence_hash: str | None = None
    first_400_schedule_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_conditional_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    manifest: ConditionalCheckpointManifestV2,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    state = model.state_dict()
    _validate_conditional_manifest(manifest, model=model)
    if manifest.state_signature != state_signature(
        state
    ) or manifest.state_content_hash != state_content_hash(state):
        raise ValueError(
            "conditional checkpoint manifest does not exactly describe model state"
        )
    atomic_torch_save(
        path,
        {
            "manifest": manifest.to_dict(),
            "state_dict": state,
            "extra": dict(extra or {}),
        },
    )


def load_conditional_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    expected: ConditionalCheckpointManifestV2,
    *,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    payload = _load_weights_only(path, expected_sha256=expected_sha256)
    return load_conditional_checkpoint_payload(payload, model, expected)


def load_conditional_checkpoint_payload(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    expected: ConditionalCheckpointManifestV2,
) -> Mapping[str, Any]:
    """Validate and restore an already safely-loaded conditional payload."""
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("manifest"), dict)
        or not isinstance(payload.get("state_dict"), dict)
    ):
        raise ValueError("invalid conditional checkpoint")
    try:
        actual = ConditionalCheckpointManifestV2(**payload["manifest"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid conditional checkpoint manifest") from exc
    _validate_conditional_manifest(actual, model=model)
    if actual != expected:
        raise ValueError(
            "conditional checkpoint does not match the requested architecture/system/config"
        )
    state = _validate_state_dict_payload(payload["state_dict"])
    if (
        state_signature(state) != actual.state_signature
        or state_content_hash(state) != actual.state_content_hash
    ):
        raise ValueError("conditional checkpoint state content changed")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError("conditional checkpoint state mismatch")
    return payload.get("extra", {})


def _validate_conditional_manifest(
    manifest: ConditionalCheckpointManifestV2, *, model: torch.nn.Module | None = None
) -> None:
    """Validate the architecture-specific manifest contract.

    v1 is retained as an explicit historical format.  v2 is intentionally
    content-addressed by its generator, metric and schedule contracts plus the
    immutable initialization state.  No partial/defaulted v2 manifest can be
    saved or loaded.
    """
    if manifest.schema_version != 2:
        raise ValueError("invalid conditional checkpoint manifest schema_version")
    allowed = {
        "conditional_global_amplitude_v1",
        "conditional_identity_logit_amplitude_v2",
        "conditional_class_conditioned_modes_v3",
        "conditional_parity_stabilized_modes_v4",
        "conditional_multi_anchor_local_modes_v5",
        "conditional_pca_transport_codesign_v6",
        "conditional_pca_self_imaging_transport_v6_2",
        "conditional_pca_formal_spacing_self_imaging_v6_3",
        "conditional_pca_near_conjugate_relay_v6_4",
        "conditional_nonlinear_expanded_manifold_v7",
        "conditional_transport_aligned_manifold_v7_1",
        "conditional_transport_aligned_phase_capacity_v7_2",
        "conditional_transport_aligned_evaluation_qualified_v1",
        "conditional_transport_aligned_evaluation_qualified_v1_1",
    }
    if manifest.architecture_kind not in allowed:
        raise ValueError("unsupported conditional checkpoint architecture")
    model_kind = (
        getattr(model, "architecture_kind", None) if model is not None else None
    )
    if model_kind is not None and model_kind != manifest.architecture_kind:
        raise ValueError("conditional checkpoint architecture does not match model")
    if manifest.architecture_kind == "conditional_global_amplitude_v1":
        # Legacy v1 manifests predate the v2 context fields.  Reject mixed
        # manifests so a v1 state cannot masquerade as a v2 initialization.
        if any(
            value is not None
            for value in (
                manifest.generator_contract_hash,
                manifest.metric_contract_hash,
                manifest.training_schedule_hash,
                manifest.initialization_seed,
                manifest.initialization_state_hash,
            )
        ):
            raise ValueError(
                "v1 conditional checkpoint cannot carry v2 contract fields"
            )
        return
    required = {
        "generator_contract_hash": manifest.generator_contract_hash,
        "metric_contract_hash": manifest.metric_contract_hash,
        "training_schedule_hash": manifest.training_schedule_hash,
        "initialization_state_hash": manifest.initialization_state_hash,
    }
    if manifest.initialization_seed != 1234 or any(
        not isinstance(value, str) or not value for value in required.values()
    ):
        raise ValueError(
            "v2 conditional checkpoint requires initialization seed 1234 and complete contract hashes"
        )
    if manifest.architecture_kind in {
        "conditional_pca_transport_codesign_v6",
        "conditional_pca_self_imaging_transport_v6_2",
        "conditional_pca_formal_spacing_self_imaging_v6_3",
        "conditional_pca_near_conjugate_relay_v6_4",
        "conditional_nonlinear_expanded_manifold_v7",
        "conditional_transport_aligned_manifold_v7_1",
        "conditional_transport_aligned_phase_capacity_v7_2",
        "conditional_transport_aligned_evaluation_qualified_v1",
        "conditional_transport_aligned_evaluation_qualified_v1_1",
    }:
        v6_required = {
            "config_contract_hash": manifest.config_contract_hash,
            "pca_basis_hash": manifest.pca_basis_hash,
            "lowpass_contract_hash": manifest.lowpass_contract_hash,
            "scale_contract_hash": manifest.scale_contract_hash,
            "route_contract_hash": manifest.route_contract_hash,
            "semantic_source_checkpoint_hash": manifest.semantic_source_checkpoint_hash,
            "safe_scale_evidence_hash": manifest.safe_scale_evidence_hash,
            "first_400_schedule_hash": manifest.first_400_schedule_hash,
        }
        if any(
            not isinstance(value, str) or not value for value in v6_required.values()
        ):
            raise ValueError(
                "PCA transport checkpoint requires complete basis/route provenance"
            )
