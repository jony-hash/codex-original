"""Observation-only Identity pre-OLS manifold diagnostics for H-LOMA-I0."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.nn import functional as F

from ...config import FormalSystemConfig
from ...contracts import ComplexField
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
)
from . import optical_manifold as base
from . import optical_manifold_full_epoch as full_epoch


BRANCH = "research/v2-lightgen-optical-manifold-latent-observer-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_identity_latent_observer_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_identity_latent_observer_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_identity_latent_observer_validation.v1"
OBSERVER_SCHEMA = "lightgen_optical_manifold_identity_latent_observer.v1"
FIELD_SCHEMA = "lightgen_optical_manifold_identity_pre_ols_field.v1"
INTERPOLATION_SCHEMA = "lightgen_optical_manifold_identity_latent_interpolation_panel.v1"
SOURCE_AUDIT_SCHEMA = "lightgen_optical_manifold_identity_reconstruction_audit.v1"
SOURCE_VALIDATION_SCHEMA = "lightgen_optical_manifold_identity_reconstruction_audit_validation.v1"
SOURCE_CHECKPOINT_SCHEMA = "lightgen_optical_manifold_identity_full_epoch_resume.v1"
SOURCE_AUDIT_HASH = "31ee69fd050902d0191fbc1b6847ebde040d8f91e0f01d9a6773a7bd8d284905"
SOURCE_AUDIT_VALIDATION_HASH = "fe86d7b1990f65d7f2cf51332bc1acca917302160afc1530471106ecdb442bd6"
SOURCE_TRAINING_HASH = "c01dae121210a4dff0534a9979661f92d1e0fc749087c8451ad51be24a458948"
SOURCE_CHECKPOINT_HASH = "56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f"
SOURCE_MANIFEST_HASH = "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
SOURCE_PANEL_HASH = "6842b3e49da0cda84cac3499240a6d276ae018eb545860330c457ec457d6af25"
SOURCE_HELDOUT_HASH = "9ffe5fd2df41a1a7627efd926e04159506d8aabb228799d9c384d9cb34565f03"
SOURCE_STEP = 5500
SOURCE_PAIRED_MSE = 0.03437697887420654
HELDOUT_COUNT = 1000
HELDOUT_PER_LABEL = 100
PCA_COMPONENTS = 50
PERMUTATION_SEED = 1234
ENDPOINT_MSE_TOLERANCE = 1.0e-9
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_WALL_SECONDS = 600.0
MAX_STORAGE_BYTES = 512 * 1024**2
NEXT_PASS_ACTION = "manual_review_identity_manifold_regularization_contract"
NEXT_VALID_FAILED_ACTION = "manual_review_identity_representation_capacity"
EXPECTED_FORMAL_KIND = base.EXPECTED_FORMAL_KIND
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass(frozen=True)
class ObserverSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ObserverData:
    full_data: full_epoch.FullEpochData
    images: torch.Tensor
    labels: torch.Tensor
    heldout_indices: torch.Tensor
    endpoint_indices: torch.Tensor
    amplitudes: torch.Tensor
    target: torch.Tensor
    manifest: dict[str, Any]


@dataclass(frozen=True)
class TraceCapture:
    pre_ols: torch.Tensor
    amplitude: torch.Tensor
    wrapped_phase: torch.Tensor
    sample_power: torch.Tensor
    endpoint_raw: torch.Tensor
    endpoint_positions: torch.Tensor
    offline_mse: float
    pre_post_exact: bool
    pre_plane: dict[str, Any]
    pre_provenance: dict[str, Any]


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"]:
        raise ValueError("Identity latent observer requires a clean worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"Identity latent observer requires branch {BRANCH}")


def _require_deterministic_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "latent observer requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before the Python process starts"
        )


def _expected_config_keys() -> set[str]:
    return {
        "schema",
        "hypothesis",
        "formal_system_config",
        "source",
        "dataset",
        "optical_contract",
        "trace_contract",
        "observer",
        "structure",
        "interpolation",
        "gates",
        "authorization",
    }


def _load_spec(path: str | Path) -> ObserverSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _expected_config_keys():
        raise ValueError("latent observer config has unexpected or missing fields")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported latent observer config")
    return ObserverSpec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: ObserverSpec) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != REVIEW_SCHEMA:
        raise ValueError("unsupported latent observer review contract")
    if payload.get("hypothesis") != HYPOTHESIS or payload.get("reviewer_role") != "project_owner":
        raise ValueError("latent observer review ownership is invalid")
    source = spec.payload["source"]
    expected = {
        "source_attribution_evidence": source["attribution_evidence"],
        "source_attribution_evidence_sha256": source["attribution_evidence_sha256"],
        "source_attribution_validation": source["attribution_validation"],
        "source_attribution_validation_sha256": source["attribution_validation_sha256"],
        "source_checkpoint": source["checkpoint"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_dataset_manifest_hash": source["mnist_manifest_sha256"],
    }
    if payload.get("decision") != "approve_observation_only_identity_latent_observer":
        raise ValueError("latent observer review decision is invalid")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("latent observer review source binding mismatch")
    approved = payload.get("approved_contract")
    if approved != {
        "heldout_count": HELDOUT_COUNT,
        "heldout_per_label": HELDOUT_PER_LABEL,
        "pre_ols_plane": "encoder_layer_2",
        "post_ols_plane": "ols_output",
        "exact_pre_post_identity_field": True,
        "no_model_update": True,
        "pca_components": PCA_COMPONENTS,
        "permutation_seed": PERMUTATION_SEED,
        "endpoint_mse_tolerance": ENDPOINT_MSE_TOLERANCE,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "post_pass_next_action": NEXT_PASS_ACTION,
        "post_valid_failed_next_action": NEXT_VALID_FAILED_ACTION,
    }:
        raise ValueError("latent observer review scope mismatch")
    if (
        payload.get("no_update_observer_authorized") is not True
        or payload.get("training_authorized") is not False
        or payload.get("formal_training_authorized") is not False
        or payload.get("formal_promotion") is not False
    ):
        raise ValueError("latent observer authorization boundary is invalid")
    return payload, target


def _load_formal(spec: ObserverSpec) -> tuple[FormalSystemConfig, Path]:
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    source_spec = full_epoch._load_spec("configs/research/lightgen_optical_manifold_identity_full_epoch_v1.yaml")
    formal, source_formal_path = full_epoch._load_formal(source_spec)
    if source_formal_path != formal_path or formal.kind != EXPECTED_FORMAL_KIND:
        raise ValueError("latent observer formal config mismatch")
    return formal, formal_path


def _load_data(dataset_root: str | Path) -> ObserverData:
    full_data = full_epoch._load_data(dataset_root)
    pilot_data = full_data.pilot_data
    if pilot_data.manifest["manifest_hash"] != SOURCE_MANIFEST_HASH:
        raise ValueError("MNIST source manifest changed")
    heldout = pilot_data.heldout_indices
    labels = pilot_data.labels.index_select(0, heldout)
    expected = torch.arange(10).repeat_interleave(HELDOUT_PER_LABEL)
    if heldout.numel() != HELDOUT_COUNT or not torch.equal(labels, expected):
        raise ValueError("held-out panel is not the fixed balanced 1000-image panel")
    if base.tensor_sha256(heldout) != SOURCE_HELDOUT_HASH:
        raise ValueError("held-out indices changed")
    endpoint = heldout.reshape(10, HELDOUT_PER_LABEL)[:, [0, HELDOUT_PER_LABEL - 1]].reshape(-1)
    images = pilot_data.images.index_select(0, heldout)
    rendered = base.render_intensity_batch(images)
    return ObserverData(
        full_data=full_data,
        images=images,
        labels=labels,
        heldout_indices=heldout,
        endpoint_indices=endpoint,
        amplitudes=rendered.amplitude,
        target=rendered.target,
        manifest={
            "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
            "full_epoch_manifest_hash": full_data.manifest["manifest_hash"],
            "heldout_indices_sha256": base.tensor_sha256(heldout),
            "heldout_count": HELDOUT_COUNT,
            "heldout_per_label": HELDOUT_PER_LABEL,
            "endpoint_indices_sha256": base.tensor_sha256(endpoint),
            "train_count": int(pilot_data.train_indices.numel()),
            "validation_count": int(pilot_data.validation_indices.numel()),
        },
    )


def _assert_ancestor(commit: str, current: str) -> None:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, current], cwd=base._repo_root(), check=False)
    if result.returncode != 0:
        raise ValueError("source evidence commit is not an ancestor of observer branch")


def _load_source(
    spec: ObserverSpec,
    review_path: Path,
    data: ObserverData,
    current_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    source = spec.payload["source"]
    audit_path = _resolve(source["attribution_evidence"])
    validation_path = _resolve(source["attribution_validation"])
    checkpoint_path = _resolve(source["checkpoint"])
    panel_path = _resolve(source["attribution_panel"])
    for path, expected, role in (
        (audit_path, SOURCE_AUDIT_HASH, "attribution evidence"),
        (validation_path, SOURCE_AUDIT_VALIDATION_HASH, "attribution validation"),
        (checkpoint_path, SOURCE_CHECKPOINT_HASH, "full-epoch checkpoint"),
        (panel_path, SOURCE_PANEL_HASH, "attribution panel"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"{role} hash mismatch")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    audit_record = EvidenceRecordV2.from_dict(audit_payload["evidence"])
    validation_record = EvidenceRecordV2.from_dict(validation_payload["evidence"])
    if audit_record.evidence_kind != SOURCE_AUDIT_SCHEMA or audit_record.status != "passed":
        raise ValueError("attribution evidence is not passed")
    if validation_record.evidence_kind != SOURCE_VALIDATION_SCHEMA or validation_record.status != "passed":
        raise ValueError("attribution validation is not passed")
    audit_record.validate()
    validation_record.validate()
    _assert_ancestor(audit_record.git_commit, current_commit)
    _assert_ancestor(validation_record.git_commit, current_commit)
    if (
        audit_payload.get("training_authorized") is not False
        or audit_payload.get("formal_promotion") is not False
        or validation_payload.get("training_authorized") is not False
        or validation_payload.get("formal_promotion") is not False
    ):
        raise ValueError("source attribution boundary is invalid")
    if audit_record.input_hashes.get("source_checkpoint") != SOURCE_CHECKPOINT_HASH:
        raise ValueError("attribution checkpoint binding mismatch")
    if audit_record.input_hashes.get("source_mnist_manifest") != SOURCE_MANIFEST_HASH:
        raise ValueError("attribution MNIST binding mismatch")
    checkpoint = safe_torch_load(checkpoint_path, expected_sha256=SOURCE_CHECKPOINT_HASH, map_location="cpu")
    if (
        checkpoint.get("schema") != SOURCE_CHECKPOINT_SCHEMA
        or checkpoint.get("global_step") != SOURCE_STEP
        or len(checkpoint.get("loss_history", [])) != SOURCE_STEP
        or not isinstance(checkpoint.get("model_state"), dict)
        or not isinstance(checkpoint.get("optimizer_state"), dict)
        or not isinstance(checkpoint.get("rng_state"), dict)
    ):
        raise ValueError("source checkpoint state is incomplete or at the wrong step")
    evidence_hashes = checkpoint.get("evidence_hashes")
    if (
        not isinstance(evidence_hashes, dict)
        or evidence_hashes.get("full_epoch_validation_evidence") is None
        or evidence_hashes.get("full_epoch_preflight_evidence") is None
    ):
        raise ValueError("source checkpoint evidence binding is incomplete")
    if data.manifest["heldout_indices_sha256"] != SOURCE_HELDOUT_HASH:
        raise ValueError("observer data binding mismatch")
    hashes = {
        "observer_config": spec.config_hash,
        "review_contract": file_sha256(review_path),
        "attribution_validation": SOURCE_AUDIT_VALIDATION_HASH,
        "attribution_evidence": SOURCE_AUDIT_HASH,
        "full_epoch_training_evidence": SOURCE_TRAINING_HASH,
        "source_checkpoint": SOURCE_CHECKPOINT_HASH,
        "source_mnist_manifest": SOURCE_MANIFEST_HASH,
        "attribution_panel": SOURCE_PANEL_HASH,
        "heldout_indices": SOURCE_HELDOUT_HASH,
    }
    return audit_payload, validation_payload, checkpoint, hashes


def _load_observer_validation(
    path: str | Path,
    spec: ObserverSpec,
    formal: FormalSystemConfig,
    data: ObserverData,
    state: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    validation_hash = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != VALIDATION_SCHEMA
        or payload.get("hypothesis") != HYPOTHESIS
        or payload.get("training_authorized") is not False
        or payload.get("formal_promotion") is not False
    ):
        raise ValueError("latent observer validation evidence has an invalid schema or authorization")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed":
        raise ValueError("latent observer validation evidence must be passed")
    if record.git_commit != state["commit"] or record.config_hash != spec.config_hash:
        raise ValueError("latent observer validation evidence is stale")
    if record.context != _context(spec, formal, data):
        raise ValueError("latent observer validation context does not match the current run")
    if record.input_hashes != dict(source_hashes):
        raise ValueError("latent observer validation source hashes do not match the current inputs")
    if record.permitted_next_actions != ("run_identity_latent_observer",):
        raise ValueError("latent observer validation next action is invalid")
    if payload.get("contract", {}).get("no_update_observer_authorized") is not True:
        raise ValueError("latent observer validation is missing no-update authorization")
    return payload, validation_hash


def _context(spec: ObserverSpec, formal: FormalSystemConfig, data: ObserverData) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=data.manifest["full_epoch_manifest_hash"],
        calibration_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _record_payload(schema: str, record: EvidenceRecordV2, contract: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": schema,
        "evidence": record.to_dict(),
        "hypothesis": HYPOTHESIS,
        "training_authorized": False,
        "formal_promotion": False,
        "contract": dict(contract),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_new_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite latent observer evidence: {target}")
    atomic_write_json(target, payload)


def _validation_record(state: Mapping[str, Any], spec: ObserverSpec, formal: FormalSystemConfig, data: ObserverData, hashes: Mapping[str, str]) -> EvidenceRecordV2:
    gates = tuple(
        GateResult(name, True, metrics)
        for name, metrics in (
            ("clean_lineage", {"clean": True}),
            ("independent_latent_observer_branch", {"branch_matches": True}),
            ("passed_attribution_source", {"checkpoint_step": SOURCE_STEP}),
            ("fixed_mnist_panel", {"count": HELDOUT_COUNT, "per_label": HELDOUT_PER_LABEL}),
            ("fixed_formal_geometry_identity_ols", {"grid": 2048, "identity": True}),
            ("trace_field_contract", {"pre_ols": "encoder_layer_2", "post_ols": "ols_output"}),
            ("zero_update_scope", {"optimizer": False, "checkpoint": False}),
        )
    )
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=VALIDATION_SCHEMA,
        tier="diagnostic",
        status="passed",
        git_commit=str(state["commit"]),
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, data),
        input_hashes=dict(hashes),
        gates=gates,
        permitted_next_actions=("run_identity_latent_observer",),
        invalidation_reason=None,
    )
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, _ = _load_formal(spec)
    data = _load_data(args.dataset_root)
    _, _, _, hashes = _load_source(spec, review_path, data, state["commit"])
    record = _validation_record(state, spec, formal, data, hashes)
    payload = _record_payload(
        VALIDATION_SCHEMA,
        record,
        {
            "source_attribution_evidence_hash": SOURCE_AUDIT_HASH,
            "source_checkpoint_hash": SOURCE_CHECKPOINT_HASH,
            "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
            "heldout_indices_sha256": SOURCE_HELDOUT_HASH,
            "no_update_observer_authorized": review["no_update_observer_authorized"],
            "trace_lookup": "plane_and_operation_provenance",
        },
    )
    _write_new_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_equal(model: base.IdentityOpticalManifoldAutoencoder, expected: Mapping[str, torch.Tensor]) -> bool:
    actual = model.state_dict()
    return set(actual) == set(expected) and all(torch.equal(actual[name].detach().cpu(), value) for name, value in expected.items())


def _find_trace_field(trace: tuple[ComplexField, ...] | None, plane_id: str, operation: str) -> ComplexField:
    if trace is None:
        raise ValueError("formal optical trace is missing")
    matches = [field for field in trace if field.plane.plane_id == plane_id and field.provenance.operation == operation]
    if len(matches) != 1:
        raise ValueError(f"expected one trace field for {plane_id}/{operation}, found {len(matches)}")
    return matches[0]


def _capture_trace(model: base.IdentityOpticalManifoldAutoencoder, data: ObserverData, device: torch.device) -> TraceCapture:
    pre_values: list[torch.Tensor] = []
    amplitudes: list[torch.Tensor] = []
    phases: list[torch.Tensor] = []
    powers: list[torch.Tensor] = []
    offline_losses: list[torch.Tensor] = []
    endpoint_positions = torch.cat(
        [
            torch.arange(label * HELDOUT_PER_LABEL, label * HELDOUT_PER_LABEL + 1)
            for label in range(10)
        ]
        + [
            torch.arange(
                label * HELDOUT_PER_LABEL + HELDOUT_PER_LABEL - 1,
                label * HELDOUT_PER_LABEL + HELDOUT_PER_LABEL,
            )
            for label in range(10)
        ]
    )
    endpoint_positions = torch.sort(endpoint_positions).values
    endpoint_set = set(endpoint_positions.tolist())
    endpoint_raw: dict[int, torch.Tensor] = {}
    pre_plane: dict[str, Any] | None = None
    pre_provenance: dict[str, Any] | None = None
    pre_post_exact = True
    model.eval()
    with torch.no_grad():
        for position in range(HELDOUT_COUNT):
            amplitude = data.amplitudes[position : position + 1].to(device)
            result = model.system(
                model.system.command(amplitude),
                exposure=model.exposure,
                return_optical_trace=True,
            )
            pre = _find_trace_field(result.optical_trace, "encoder_layer_2", "finite_crop_hard_aperture")
            post = _find_trace_field(result.optical_trace, "ols_output", "identity_ols")
            if not torch.equal(pre.tensor, post.tensor):
                raise ValueError("pre-OLS and post-Identity-OLS tensors are not exactly equal")
            raw = result.raw_camera.tensor
            target = data.target[position : position + 1].to(device)
            offline_losses.append(F.mse_loss(raw, target).detach().cpu())
            pre_cpu = pre.tensor.detach().cpu().to(torch.complex64)
            pre_values.append(pre_cpu)
            amplitudes.append(pre_cpu.abs())
            phases.append(torch.angle(pre_cpu))
            powers.append(pre_cpu.abs().square().sum(dim=(-3, -2, -1)))
            if position in endpoint_set:
                endpoint_raw[position] = raw.detach().cpu().float()
            current_plane = pre.plane.to_dict()
            current_provenance = pre.provenance.to_dict()
            if pre_plane is None:
                pre_plane, pre_provenance = current_plane, current_provenance
            elif pre_plane != current_plane or pre_provenance != current_provenance:
                raise ValueError("pre-OLS trace provenance changed across samples")
    if pre_plane is None or pre_provenance is None:
        raise ValueError("no pre-OLS trace was captured")
    endpoint_tensor = torch.cat([endpoint_raw[index] for index in endpoint_positions.tolist()], dim=0)
    return TraceCapture(
        pre_ols=torch.cat(pre_values, dim=0),
        amplitude=torch.cat(amplitudes, dim=0),
        wrapped_phase=torch.cat(phases, dim=0),
        sample_power=torch.cat(powers, dim=0),
        endpoint_raw=endpoint_tensor,
        endpoint_positions=endpoint_positions,
        offline_mse=float(torch.stack(offline_losses).mean().item()),
        pre_post_exact=pre_post_exact,
        pre_plane=pre_plane,
        pre_provenance=pre_provenance,
    )


def _distance_ratio(gram: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    diagonal = gram.diag()
    distances = (diagonal[:, None] + diagonal[None, :] - 2.0 * gram).clamp_min(0.0).sqrt()
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    same = upper & labels[:, None].eq(labels[None, :])
    different = upper & ~labels[:, None].eq(labels[None, :])
    intra = distances[same].mean()
    inter = distances[different].mean()
    return {
        "intra": float(intra.item()),
        "inter": float(inter.item()),
        "inter_intra_ratio": float(inter.div(intra.clamp_min(1.0e-12)).item()),
    }


def _structure_metrics(values: torch.Tensor, labels: torch.Tensor, permuted_labels: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
    flat = values.flatten(1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    gram = (centered @ centered.conj().T).real if torch.is_complex(centered) else centered @ centered.T
    gram = gram.to(torch.float32).div(float(flat.shape[0]))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    eigenvectors = eigenvectors.index_select(1, order)
    total = eigenvalues.sum().clamp_min(1.0e-20)
    probabilities = eigenvalues / total
    entropy_rank = torch.exp(-(probabilities * probabilities.clamp_min(1.0e-20).log()).sum())
    participation_rank = 1.0 / probabilities.square().sum().clamp_min(1.0e-20)
    count = min(PCA_COMPONENTS, eigenvalues.numel())
    scores = eigenvectors[:, :count] * eigenvalues[:count].sqrt()
    true_distance = _distance_ratio(gram, labels)
    permuted_distance = _distance_ratio(gram, permuted_labels)
    metrics = {
        "top_50_eigenvalues": eigenvalues[:count].tolist(),
        "top_50_explained_variance": eigenvalues[:count].div(total).tolist(),
        "top_50_explained_variance_sum": float(eigenvalues[:count].sum().div(total).item()),
        "entropy_effective_rank": float(entropy_rank.item()),
        "participation_rank": float(participation_rank.item()),
        "true_label_distance": true_distance,
        "permuted_label_distance": permuted_distance,
        "finite": bool(torch.isfinite(eigenvalues).all() and torch.isfinite(scores).all()),
    }
    return metrics, scores


def _endpoint_mse(raw: torch.Tensor, endpoint_raw: torch.Tensor, endpoint_index: int) -> float:
    target = endpoint_raw[endpoint_index : endpoint_index + 1]
    if raw.shape != target.shape:
        raise ValueError(f"endpoint camera shape mismatch: {tuple(raw.shape)} != {tuple(target.shape)}")
    return float(F.mse_loss(raw, target).item())


def _interpolate(
    model: base.IdentityOpticalManifoldAutoencoder,
    capture: TraceCapture,
    data: ObserverData,
    device: torch.device,
    source_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = []
    fixed_values = []
    labels = []
    start_indices = []
    end_indices = []
    t_values = []
    endpoint_mse: list[float] = []
    endpoint_lookup = {int(position): index for index, position in enumerate(capture.endpoint_positions.tolist())}
    fields = capture.pre_ols.to(device)
    provenance = capture.pre_provenance
    from ...contracts import OperationProvenance, PlaneSpec

    plane = PlaneSpec(
        plane_id=capture.pre_plane["plane_id"],
        shape=tuple(capture.pre_plane["shape"]),
        pixel_pitch_m=tuple(capture.pre_plane["pixel_pitch_m"]),
        wavelength_m=float(capture.pre_plane["wavelength_m"]),
        z_m=float(capture.pre_plane["z_m"]),
        coordinate_frame=str(capture.pre_plane["coordinate_frame"]),
        origin_m=tuple(capture.pre_plane["origin_m"]),
    )
    provenance_object = OperationProvenance(
        operation=str(provenance["operation"]),
        upstream_plane=provenance["upstream_plane"],
        propagation_distance_m=provenance["propagation_distance_m"],
        parameters=provenance["parameters"],
    )
    model.eval()
    with torch.no_grad():
        for label in range(10):
            start = label * HELDOUT_PER_LABEL
            end = start + HELDOUT_PER_LABEL - 1
            for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                field_tensor = (1.0 - t) * fields[start : start + 1] + t * fields[end : end + 1]
                pre = ComplexField(field_tensor, plane, provenance_object)
                post = model.system.ols(pre)
                decoded, _, _ = model.system.decoder(post, "decoder")
                chip = ComplexField(decoded.tensor, model.system.chip_plane, decoded.provenance)
                camera = model.system.back(chip)
                raw = camera.tensor.abs().square()
                fixed = (raw / model.exposure.scale).clamp(0.0, 1.0)
                if t in {0.0, 1.0}:
                    endpoint_position = start if t == 0.0 else end
                    endpoint_mse.append(_endpoint_mse(raw.cpu(), capture.endpoint_raw, endpoint_lookup[endpoint_position]))
                values.append(raw.detach().cpu().float())
                fixed_values.append(fixed.detach().cpu().float())
                labels.append(label)
                start_indices.append(int(data.heldout_indices[start]))
                end_indices.append(int(data.heldout_indices[end]))
                t_values.append(t)
    panel = {
        "schema": INTERPOLATION_SCHEMA,
        "raw_cmos": torch.cat(values, dim=0),
        "fixed_exposure_cmos": torch.cat(fixed_values, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
        "start_indices": torch.tensor(start_indices, dtype=torch.long),
        "end_indices": torch.tensor(end_indices, dtype=torch.long),
        "endpoint_indices": data.endpoint_indices.cpu(),
        "t_values": torch.tensor(t_values, dtype=torch.float32),
        "source_hashes": dict(source_hashes),
        "pre_ols_plane": capture.pre_plane,
        "pre_ols_provenance": capture.pre_provenance,
        "affects_gate_or_selection": False,
    }
    metadata = {"endpoint_mse_max": max(endpoint_mse), "endpoint_mse_values": endpoint_mse, "endpoint_mse_tolerance": ENDPOINT_MSE_TOLERANCE, "finite": bool(torch.isfinite(panel["raw_cmos"]).all() and torch.isfinite(panel["fixed_exposure_cmos"]).all())}
    return panel, metadata


def _observer_record(state: Mapping[str, Any], spec: ObserverSpec, formal: FormalSystemConfig, data: ObserverData, hashes: Mapping[str, str], gates: tuple[GateResult, ...], status: str, actions: tuple[str, ...], reason: str | None) -> EvidenceRecordV2:
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=OBSERVER_SCHEMA,
        tier="diagnostic",
        status=status,
        git_commit=str(state["commit"]),
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, data),
        input_hashes=dict(hashes),
        gates=gates,
        permitted_next_actions=actions,
        invalidation_reason=reason,
    )
    record.validate()
    return record


@contextmanager
def _deterministic_algorithms() -> Any:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _run_cuda_observer(
    *,
    spec: ObserverSpec,
    state: Mapping[str, Any],
    formal: FormalSystemConfig,
    data: ObserverData,
    checkpoint: Mapping[str, Any],
    hashes: Mapping[str, str],
    output: Path,
    device: torch.device,
    free_bytes: int,
    total_bytes: int,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    before = _state_cpu(model)
    capture = _capture_trace(model, data, device)
    source_replay_mse = full_epoch.pilot._evaluate(model, data.full_data.pilot_data, data.heldout_indices, device)
    model.eval()
    labels = data.labels
    generator = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    permuted_labels = labels.index_select(0, torch.randperm(labels.numel(), generator=generator))
    normalized = capture.pre_ols / capture.sample_power.clamp_min(1.0e-20).sqrt().reshape(-1, 1, 1, 1)
    raw_structure, raw_scores = _structure_metrics(capture.pre_ols, labels, permuted_labels)
    normalized_structure, normalized_scores = _structure_metrics(normalized, labels, permuted_labels)
    input_structure, input_scores = _structure_metrics(data.target, labels, permuted_labels)
    field_payload = {
        "schema": FIELD_SCHEMA,
        "indices": data.heldout_indices.cpu(),
        "labels": labels.cpu(),
        "pre_ols_field": capture.pre_ols,
        "amplitude": capture.amplitude,
        "wrapped_phase": capture.wrapped_phase,
        "sample_power": capture.sample_power,
        "raw_pca_scores": raw_scores,
        "normalized_pca_scores": normalized_scores,
        "input_pca_scores": input_scores,
        "source_hashes": dict(hashes),
        "plane": capture.pre_plane,
        "provenance": capture.pre_provenance,
        "normalization": "unit_sample_power",
        "affects_training_or_selection": False,
    }
    field_path = output.parent / "pre_ols_field.pt"
    atomic_torch_save(field_path, field_payload)
    panel_payload, interpolation = _interpolate(model, capture, data, device, hashes)
    panel_path = output.parent / "latent_interpolation_panel.pt"
    atomic_torch_save(panel_path, panel_payload)
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
    normalized_ratio = normalized_structure["true_label_distance"]["inter_intra_ratio"]
    permuted_ratio = normalized_structure["permuted_label_distance"]["inter_intra_ratio"]
    after = _state_cpu(model)
    state_unchanged = _state_equal(model, before) and _state_equal(model, after)
    endpoint_gate = interpolation["endpoint_mse_max"] <= ENDPOINT_MSE_TOLERANCE
    gates = (
        GateResult("source_reconstruction_reproduced", abs(source_replay_mse - SOURCE_PAIRED_MSE) <= 1.0e-9, {"source": SOURCE_PAIRED_MSE, "authoritative_source_replay_mse": source_replay_mse, "offline_attribution_mse": capture.offline_mse}),
        GateResult("normalized_inter_intra_gt_one", normalized_ratio > 1.0, {"ratio": normalized_ratio}),
        GateResult("true_ratio_gt_permuted_ratio", normalized_ratio > permuted_ratio, {"true": normalized_ratio, "permuted": permuted_ratio}),
        GateResult("finite_structure_metrics", all(item["finite"] for item in (raw_structure, normalized_structure, input_structure)), {}),
        GateResult("exact_pre_post_identity_field", capture.pre_post_exact, {"samples": HELDOUT_COUNT}),
        GateResult("endpoint_camera_reproduction", endpoint_gate, {"max_mse": interpolation["endpoint_mse_max"], "tolerance": ENDPOINT_MSE_TOLERANCE}),
        GateResult("model_state_unchanged", state_unchanged, {}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("optical_provenance", capture.pre_plane["plane_id"] == "encoder_layer_2" and capture.pre_provenance["operation"] == "finite_crop_hard_aperture", {}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
        GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
    )
    harness = gates[0:1] + gates[3:]
    science = gates[1:3]
    if not all(gate.passed for gate in harness):
        status, actions, reason = "invalid_harness", (), "Identity latent observer harness or resource gate failed"
    elif not all(gate.passed for gate in science):
        status, actions, reason = "valid_failed", (NEXT_VALID_FAILED_ACTION,), "pre-registered Identity normalized representation gate failed"
    else:
        status, actions, reason = "passed", (NEXT_PASS_ACTION,), None
    hashes = dict(hashes)
    hashes["pre_ols_field"] = file_sha256(field_path)
    hashes["interpolation_panel"] = file_sha256(panel_path)
    record = _observer_record(state, spec, formal, data, hashes, gates, status, actions, reason)
    payload = _record_payload(
        OBSERVER_SCHEMA,
        record,
        {
            "source_attribution_evidence_hash": SOURCE_AUDIT_HASH,
            "source_checkpoint_hash": SOURCE_CHECKPOINT_HASH,
            "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
            "observer_validation_hash": hashes["observer_validation"],
            "source_reconstruction": {"authoritative_source_replay_mse": source_replay_mse, "offline_attribution_mse": capture.offline_mse, "source_tolerance": 1.0e-9},
            "structure": {"raw": raw_structure, "normalized": normalized_structure, "rendered_input_reference": input_structure, "permuted_seed": PERMUTATION_SEED},
            "pre_ols_field": {"path": str(field_path.resolve()), "sha256": hashes["pre_ols_field"], "schema": FIELD_SCHEMA, "shape": list(capture.pre_ols.shape), "dtype": str(capture.pre_ols.dtype)},
            "interpolation_panel": {"path": str(panel_path.resolve()), "sha256": hashes["interpolation_panel"], "schema": INTERPOLATION_SCHEMA, "count": 50, "endpoint": interpolation},
            "model_state_unchanged": state_unchanged,
            "evaluator_calls": 0,
            "free_cuda_bytes_before": int(free_bytes),
            "total_cuda_bytes": int(total_bytes),
            "peak_cuda_bytes": peak,
            "wall_seconds": elapsed,
            "run_storage_bytes": storage,
        },
    )
    _write_new_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "passed" else 2


def run(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("latent observer requires CUDA")
    _require_deterministic_cuda_runtime()
    spec = _load_spec(args.config)
    _review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, _ = _load_formal(spec)
    data = _load_data(args.dataset_root)
    _, _, checkpoint, hashes = _load_source(spec, review_path, data, state["commit"])
    _validation_payload, validation_hash = _load_observer_validation(
        args.validation_evidence,
        spec,
        formal,
        data,
        state,
        hashes,
    )
    hashes = dict(hashes)
    hashes["observer_validation"] = validation_hash
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite latent observer evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for latent observer")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    with _deterministic_algorithms():
        return _run_cuda_observer(
            spec=spec,
            state=state,
            formal=formal,
            data=data,
            checkpoint=checkpoint,
            hashes=hashes,
            output=output,
            device=device,
            free_bytes=free_bytes,
            total_bytes=total_bytes,
        )


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "latent-observer-validate":
        return validate(args)
    if args.research_action == "latent-observer-run":
        return run(args)
    raise ValueError(f"unsupported latent observer action: {args.research_action}")
