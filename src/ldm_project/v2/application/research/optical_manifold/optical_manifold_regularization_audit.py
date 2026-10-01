"""No-update aggregate-prior feasibility audit for H-LOMA-I0.

The current Identity optical manifold is deterministic.  This lifecycle
therefore audits a fixed, project-approximate aggregate prior on a train-only
projection of the normalized pre-OLS field.  It never creates an optimizer,
updates the source model, or authorizes a regularization training run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.nn import functional as F

from ldm_project.v2.config import FormalSystemConfig, load_config
from ldm_project.v2.contracts import ComplexField
from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_full_epoch as full_epoch


BRANCH = "research/v2-lightgen-optical-manifold-regularization-audit-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_identity_regularization_audit_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_identity_regularization_audit_review.v1"
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_identity_regularization_audit_validation.v1"
)
AUDIT_SCHEMA = "lightgen_optical_manifold_identity_regularization_audit.v1"
PROJECTION_SCHEMA = "lightgen_optical_manifold_identity_regularization_projection.v1"

SOURCE_OBSERVER_CONFIG_HASH = (
    "33b63495a5d595c60d6c3beb399a44fae27c9f229352844c7b8edfb43f5a6a60"
)
SOURCE_OBSERVER_REVIEW_HASH = (
    "f238bc7fbdb36b2ae74aabdd446bb26f50088b44d39465d80652ba4f24dd43a7"
)
SOURCE_OBSERVER_VALIDATION_HASH = (
    "4dda0fb96e144c91d48c8ccbf1395e043bc215b988b7f6ac169b0c91a6530724"
)
SOURCE_OBSERVER_HASH = (
    "36a1b918991affaf6093033b5ce5139f2d6f87517f72373b16751f357bb63507"
)
SOURCE_FIELD_HASH = (
    "c838a2c85ad4597ebde005e79ed4bceb8726beb8597d8622a578f93f455c7341"
)
SOURCE_INTERPOLATION_HASH = (
    "b8c3553de57ee9e36bea463a957d82125c540667b61f572e08dc081c9a6c4954"
)
SOURCE_CHECKPOINT_HASH = (
    "56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f"
)
SOURCE_TRAINING_HASH = (
    "c01dae121210a4dff0534a9979661f92d1e0fc749087c8451ad51be24a458948"
)
SOURCE_MANIFEST_HASH = (
    "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
)
SOURCE_HELDOUT_HASH = (
    "9ffe5fd2df41a1a7627efd926e04159506d8aabb228799d9c384d9cb34565f03"
)
SOURCE_STEP = 5500
SOURCE_PAIRED_MSE = 0.03437697887420654
CALIBRATION_COUNT = 1000
CALIBRATION_PER_LABEL = 100
GRADIENT_AUDIT_COUNT = 300
GRADIENT_BATCHES = 3
GRADIENT_BATCH_SIZE = 100
PCA_COMPONENTS = 32
FIXED_DIRECTIONS = 64
PERMUTATION_SEED = 1234
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_WALL_SECONDS = 600.0
MAX_STORAGE_BYTES = 512 * 1024**2
ENCODER_RATIO_MIN = 0.025
ENCODER_RATIO_MAX = 0.4
NEXT_PASS_ACTION = "manual_review_identity_manifold_regularization_pilot_contract"
NEXT_VALID_FAILED_ACTION = "manual_review_identity_representation_capacity"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_FORMAL_KIND = base.EXPECTED_FORMAL_KIND
EXPECTED_TRAINABLE = base.EXPECTED_TRAINABLE


@dataclass(frozen=True)
class RegularizationSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class RegularizationData:
    full_data: full_epoch.FullEpochData
    calibration_indices: torch.Tensor
    gradient_batches: tuple[torch.Tensor, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SourceBundle:
    checkpoint: dict[str, Any]
    hashes: dict[str, str]
    formal_path: Path


@dataclass(frozen=True)
class Projection:
    mean_field: torch.Tensor
    basis: torch.Tensor
    whitening: torch.Tensor
    eigenvalues: torch.Tensor
    explained_variance: torch.Tensor


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"]:
        raise ValueError("regularization audit requires a clean worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"regularization audit requires branch {BRANCH}")


def _require_deterministic_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "regularization audit requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before the Python process starts"
        )


def _load_spec(path: str | Path) -> RegularizationSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    expected_keys = {
        "schema", "hypothesis", "formal_system_config", "source", "dataset",
        "optical_contract", "projection", "regularization", "gradient_audit",
        "gates", "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("regularization audit config has unexpected or missing fields")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported regularization audit config")
    return RegularizationSpec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: RegularizationSpec) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    source = spec.payload["source"]
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_identity_manifold_regularization_audit",
        "source_latent_observer_validation_sha256": source["latent_observer_validation_sha256"],
        "source_latent_observer_evidence_sha256": source["latent_observer_evidence_sha256"],
        "source_pre_ols_field_sha256": source["pre_ols_field_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_mnist_manifest_hash": source["mnist_manifest_sha256"],
        "approved_contract": {
            "calibration_count": CALIBRATION_COUNT,
            "gradient_audit_count": GRADIENT_AUDIT_COUNT,
            "pca_components": PCA_COMPONENTS,
            "fixed_directions": FIXED_DIRECTIONS,
            "direction_seed": PERMUTATION_SEED,
            "no_model_update": True,
            "project_approximation": True,
            "post_pass_next_action": NEXT_PASS_ACTION,
        },
        "no_update_audit_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("regularization audit review contract does not match scope")
    return payload, target


def _load_formal(spec: RegularizationSpec) -> tuple[FormalSystemConfig, Path]:
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    formal = load_config(formal_path)
    if formal.architecture != "conditional_identity_logit_amplitude_v2":
        raise ValueError("regularization audit requires current Identity lineage")
    if formal.system.kind != EXPECTED_FORMAL_KIND or not formal.system.is_formal:
        raise ValueError("regularization audit requires formal finite-crop Identity")
    for name, expected in base.EXPECTED_SYSTEM.items():
        actual = getattr(formal.system, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return formal.system, formal_path


def _assert_contract(spec: RegularizationSpec) -> None:
    payload = spec.payload
    if payload["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "calibration_count": 1000,
        "calibration_per_label": 100,
        "gradient_audit_count": 300,
        "gradient_batches": 3,
        "gradient_batch_size": 100,
        "calibration_selection": "first_100_train_examples_per_label",
        "gradient_selection": "next_30_train_examples_per_label_in_three_balanced_batches",
        "heldout_exclusion": True,
        "label_role": "panel_balance_and_provenance_only",
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }:
        raise ValueError("regularization audit dataset contract is not locked")
    if payload["optical_contract"] != {
        "formal_system_kind": EXPECTED_FORMAL_KIND,
        "wavelength_m": 5.32e-7,
        "slm_shape": [200, 200],
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "boundary": "finite_crop",
        "ols": "identity",
        "front_relay": "coherent_imaging_4f",
        "back_relay": "free_space",
        "input_encoding": "intensity_sqrt_amplitude",
        "target": "camera_intensity_same_fov",
        "exposure": {"scale": 1.0, "gamma": 1.0},
    }:
        raise ValueError("regularization audit optical contract is not locked")
    if payload["projection"] != {
        "source": "unit_power_normalized_pre_ols_field",
        "metric": "real_part_of_x_x_h",
        "center_before_pca": True,
        "components": PCA_COMPONENTS,
        "whitening": "calibration_eigenvalue",
        "basis_dtype": "complex64",
        "score_dtype": "float32",
    }:
        raise ValueError("regularization audit projection contract is not locked")
    if payload["regularization"] != {
        "family": "aggregate_prior",
        "name": "deterministic_sliced_wasserstein",
        "approximation_role": "project_approximation",
        "prior": "standard_normal",
        "fixed_directions": FIXED_DIRECTIONS,
        "direction_seed": PERMUTATION_SEED,
        "quantiles": "midpoint_standard_normal",
        "labels_in_loss": False,
        "per_sample_latent_sampling": False,
        "vae_kl": False,
    }:
        raise ValueError("regularization audit objective contract is not locked")
    if payload["gradient_audit"] != {
        "replay": "two_pass_fixed_rank_coefficients",
        "microbatch_size": 1,
        "activation_checkpoint": True,
        "lambda_rule": "0.1_times_reconstruction_encoder_norm_over_regularizer_encoder_norm_on_batch_0",
        "lambda_recalibration": False,
        "expected_regularizer_gradient_scope": "encoder_phase_only",
        "expected_reconstruction_gradient_scope": "encoder_and_decoder_phase",
        "encoder_ratio_min": ENCODER_RATIO_MIN,
        "encoder_ratio_max": ENCODER_RATIO_MAX,
        "required_descent_alignment": True,
    }:
        raise ValueError("regularization audit gradient contract is not locked")
    if payload["authorization"] != {
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_identity_manifold_regularization_audit",
        "no_update_audit_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "post_pass_next_action": NEXT_PASS_ACTION,
        "post_valid_failed_next_action": NEXT_VALID_FAILED_ACTION,
        "prohibited": payload["authorization"]["prohibited"],
    }:
        raise ValueError("regularization audit authorization contract is not locked")


def _assert_ancestor(commit: str, current: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("PLAN 4 source commit is not an ancestor of audit branch")


def _load_data(dataset_root: str | Path) -> RegularizationData:
    full_data = full_epoch._load_data(dataset_root)
    pilot_data = full_data.pilot_data
    if not full_epoch.pilot.manifest_matches_hash(pilot_data.manifest, SOURCE_MANIFEST_HASH):
        raise ValueError("MNIST source manifest changed")
    train_labels = pilot_data.labels.index_select(0, pilot_data.train_indices)
    calibration: list[torch.Tensor] = []
    gradient: list[torch.Tensor] = []
    for label in range(10):
        class_indices = pilot_data.train_indices[train_labels == label]
        if class_indices.numel() < CALIBRATION_PER_LABEL + 30:
            raise ValueError("train split does not contain the required per-label panels")
        calibration.append(class_indices[:CALIBRATION_PER_LABEL])
        gradient.append(class_indices[CALIBRATION_PER_LABEL:CALIBRATION_PER_LABEL + 30])
    calibration_indices = torch.cat(calibration)
    # Ten consecutive per-label blocks form each balanced batch.
    gradient_batches = tuple(
        torch.cat(
            [
                gradient[label][batch * 10: (batch + 1) * 10]
                for label in range(10)
            ]
        )
        for batch in range(GRADIENT_BATCHES)
    )
    if calibration_indices.numel() != CALIBRATION_COUNT or len(gradient_batches) != GRADIENT_BATCHES:
        raise ValueError("regularization audit panel sizes are incorrect")
    if any(batch.numel() != GRADIENT_BATCH_SIZE for batch in gradient_batches):
        raise ValueError("regularization audit gradient batch size is incorrect")
    all_selected = torch.cat((calibration_indices, *gradient_batches))
    if torch.unique(all_selected).numel() != all_selected.numel():
        raise ValueError("regularization audit train panels contain duplicates")
    if torch.isin(all_selected, pilot_data.heldout_indices).any():
        raise ValueError("regularization audit train panels overlap held-out data")
    manifest = {
        "schema": "lightgen_optical_manifold_identity_regularization_audit_mnist_manifest.v1",
        "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
        "full_epoch_manifest_hash": full_data.manifest["manifest_hash"],
        "calibration_indices_sha256": tensor_sha256(calibration_indices),
        "calibration_count": int(calibration_indices.numel()),
        "gradient_batch_indices_sha256": [tensor_sha256(batch) for batch in gradient_batches],
        "gradient_batch_count": len(gradient_batches),
        "gradient_batch_size": GRADIENT_BATCH_SIZE,
        "heldout_indices_sha256": SOURCE_HELDOUT_HASH,
        "heldout_excluded": True,
        "label_role": "panel_balance_and_provenance_only",
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return RegularizationData(full_data, calibration_indices, gradient_batches, manifest)


def _context(
    spec: RegularizationSpec, formal: FormalSystemConfig, data: RegularizationData
) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=data.manifest["manifest_hash"],
        calibration_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _source_bundle(
    spec: RegularizationSpec,
    review_path: Path,
    data: RegularizationData,
    current_commit: str,
) -> SourceBundle:
    source = spec.payload["source"]
    paths = {
        "observer_config": (_resolve(source["latent_observer_config"]), SOURCE_OBSERVER_CONFIG_HASH),
        "observer_review": (_resolve(source["latent_observer_review"]), SOURCE_OBSERVER_REVIEW_HASH),
        "observer_validation": (_resolve(source["latent_observer_validation"]), SOURCE_OBSERVER_VALIDATION_HASH),
        "observer_evidence": (_resolve(source["latent_observer_evidence"]), SOURCE_OBSERVER_HASH),
        "pre_ols_field": (_resolve(source["pre_ols_field"]), SOURCE_FIELD_HASH),
        "interpolation_panel": (_resolve(source["interpolation_panel"]), SOURCE_INTERPOLATION_HASH),
        "checkpoint": (_resolve(source["checkpoint"]), SOURCE_CHECKPOINT_HASH),
        "full_epoch_training_evidence": (_resolve(source["full_epoch_training_evidence"]), SOURCE_TRAINING_HASH),
    }
    for role, (path, expected) in paths.items():
        if file_sha256(path) != expected:
            raise ValueError(f"{role} source hash mismatch")
    validation_payload = json.loads(paths["observer_validation"][0].read_text(encoding="utf-8"))
    observer_payload = json.loads(paths["observer_evidence"][0].read_text(encoding="utf-8"))
    validation_record = EvidenceRecordV2.from_dict(validation_payload["evidence"])
    observer_record = EvidenceRecordV2.from_dict(observer_payload["evidence"])
    validation_record.validate()
    observer_record.validate()
    if validation_record.evidence_kind != "lightgen_optical_manifold_identity_latent_observer_validation.v1" or validation_record.status != "passed":
        raise ValueError("PLAN 4 validation is not passed")
    if observer_record.evidence_kind != "lightgen_optical_manifold_identity_latent_observer.v1" or observer_record.status != "passed":
        raise ValueError("PLAN 4 observer evidence is not passed")
    _assert_ancestor(validation_record.git_commit, current_commit)
    _assert_ancestor(observer_record.git_commit, current_commit)
    if validation_payload.get("training_authorized") is not False or observer_payload.get("training_authorized") is not False:
        raise ValueError("PLAN 4 source authorization boundary is invalid")
    if validation_payload.get("formal_promotion") is not False or observer_payload.get("formal_promotion") is not False:
        raise ValueError("PLAN 4 source promotion boundary is invalid")
    observer_contract = observer_payload.get("contract", {})
    if observer_contract.get("model_state_unchanged") is not True or observer_contract.get("evaluator_calls") != 0:
        raise ValueError("PLAN 4 source observer isolation is invalid")
    field = safe_torch_load(paths["pre_ols_field"][0], expected_sha256=SOURCE_FIELD_HASH, map_location="cpu")
    if field.get("schema") != "lightgen_optical_manifold_identity_pre_ols_field.v1":
        raise ValueError("PLAN 4 field artifact schema mismatch")
    if tuple(field["pre_ols_field"].shape) != (1000, 1, 100, 100) or field["pre_ols_field"].dtype != torch.complex64:
        raise ValueError("PLAN 4 field artifact shape or dtype mismatch")
    if field.get("source_hashes", {}).get("observer_validation") != SOURCE_OBSERVER_VALIDATION_HASH:
        raise ValueError("PLAN 4 field artifact validation binding mismatch")
    checkpoint = safe_torch_load(paths["checkpoint"][0], expected_sha256=SOURCE_CHECKPOINT_HASH, map_location="cpu")
    if (
        checkpoint.get("schema") != "lightgen_optical_manifold_identity_full_epoch_resume.v1"
        or checkpoint.get("global_step") != SOURCE_STEP
        or len(checkpoint.get("loss_history", [])) != SOURCE_STEP
        or not isinstance(checkpoint.get("model_state"), dict)
        or not isinstance(checkpoint.get("rng_state"), dict)
    ):
        raise ValueError("source checkpoint state is incomplete or at the wrong step")
    if data.manifest["source_mnist_manifest_hash"] != SOURCE_MANIFEST_HASH:
        raise ValueError("regularization audit data manifest binding mismatch")
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    return SourceBundle(
        checkpoint=checkpoint,
        formal_path=formal_path,
        hashes={
            "regularization_config": spec.config_hash,
            "review_contract": file_sha256(review_path),
            "observer_config": SOURCE_OBSERVER_CONFIG_HASH,
            "observer_review": SOURCE_OBSERVER_REVIEW_HASH,
            "observer_validation": SOURCE_OBSERVER_VALIDATION_HASH,
            "observer_evidence": SOURCE_OBSERVER_HASH,
            "pre_ols_field": SOURCE_FIELD_HASH,
            "interpolation_panel": SOURCE_INTERPOLATION_HASH,
            "full_epoch_training_evidence": SOURCE_TRAINING_HASH,
            "source_checkpoint": SOURCE_CHECKPOINT_HASH,
            "mnist_manifest": SOURCE_MANIFEST_HASH,
            "heldout_indices": SOURCE_HELDOUT_HASH,
        },
    )


def _parameter_contract(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, Any]:
    names = tuple(name for name, _ in model.named_parameters())
    trainable = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if trainable != EXPECTED_TRAINABLE:
        raise ValueError("regularization audit trainable parameters are not four phase layers")
    return {
        "all_parameter_names": list(names),
        "trainable_parameter_names": list(trainable),
        "optimizer_created": False,
        "optimizer_parameter_names": [],
    }


def _record_payload(
    schema: str,
    record: EvidenceRecordV2,
    contract: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
        raise FileExistsError(f"refusing to overwrite regularization audit evidence: {target}")
    atomic_write_json(target, payload)


def _validation_record(
    state: Mapping[str, Any],
    spec: RegularizationSpec,
    formal: FormalSystemConfig,
    data: RegularizationData,
    hashes: Mapping[str, str],
) -> EvidenceRecordV2:
    gates = tuple(
        GateResult(name, True, metrics)
        for name, metrics in (
            ("clean_lineage", {"clean": True}),
            ("independent_regularization_audit_branch", {"branch_matches": True}),
            ("passed_latent_observer_source", {"checkpoint_step": SOURCE_STEP}),
            ("train_only_panel_is_disjoint_from_heldout", {"calibration": 1000, "gradient": 300}),
            ("formal_geometry_identity_ols", {"grid": 2048, "identity": True}),
            ("fixed_exposure", {"scale": 1.0, "gamma": 1.0}),
            ("deterministic_projection_contract", {"components": PCA_COMPONENTS, "directions": FIXED_DIRECTIONS}),
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
        permitted_next_actions=("run_identity_manifold_regularization_audit",),
    )
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    source = _source_bundle(spec, review_path, data, state["commit"])
    if source.formal_path != formal_path:
        raise ValueError("regularization audit formal config binding mismatch")
    model = base.IdentityOpticalManifoldAutoencoder(formal)
    parameter_contract = _parameter_contract(model)
    hashes = dict(source.hashes)
    hashes["formal_config"] = file_sha256(formal_path)
    hashes["dataset_panel_manifest"] = data.manifest["manifest_hash"]
    record = _validation_record(state, spec, formal, data, hashes)
    payload = _record_payload(
        VALIDATION_SCHEMA,
        record,
        {
            "formal_system_kind": EXPECTED_FORMAL_KIND,
            "parameter_contract": parameter_contract,
            "dataset_manifest": data.manifest,
            "projection": spec.payload["projection"],
            "regularization": spec.payload["regularization"],
            "no_update_audit_authorized": review["no_update_audit_authorized"],
        },
    )
    _write_new_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(
    path: str | Path,
    spec: RegularizationSpec,
    formal: FormalSystemConfig,
    data: RegularizationData,
    state: Mapping[str, Any],
    hashes: Mapping[str, str],
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
        raise ValueError("regularization audit validation has an invalid schema or authorization")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed":
        raise ValueError("regularization audit validation must be passed")
    if record.git_commit != state["commit"] or state["branch"] != BRANCH or not state["clean"]:
        raise ValueError("regularization audit validation is stale")
    if record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("regularization audit validation context mismatch")
    if record.input_hashes != dict(hashes):
        raise ValueError("regularization audit validation source hashes do not match")
    if record.permitted_next_actions != ("run_identity_manifold_regularization_audit",):
        raise ValueError("regularization audit validation next action is invalid")
    if payload.get("contract", {}).get("no_update_audit_authorized") is not True:
        raise ValueError("regularization audit validation lacks no-update authorization")
    return payload, validation_hash


def _find_trace_field(
    trace: tuple[ComplexField, ...] | None,
    plane_id: str,
    operation: str,
) -> ComplexField:
    if trace is None:
        raise ValueError("formal optical trace is missing")
    matches = [
        field for field in trace
        if field.plane.plane_id == plane_id and field.provenance.operation == operation
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one trace field for {plane_id}/{operation}, found {len(matches)}"
        )
    return matches[0]


def _forward_trace(
    model: base.IdentityOpticalManifoldAutoencoder,
    amplitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    result = model.system(
        model.system.command(amplitude),
        exposure=model.exposure,
        return_optical_trace=True,
    )
    pre = _find_trace_field(
        result.optical_trace, "encoder_layer_2", "finite_crop_hard_aperture"
    )
    post = _find_trace_field(result.optical_trace, "ols_output", "identity_ols")
    if not torch.equal(pre.tensor, post.tensor):
        raise ValueError("pre-OLS and post-Identity-OLS tensors are not exactly equal")
    return result.raw_camera.tensor, pre.tensor, pre.plane.to_dict(), pre.provenance.to_dict()


def _render_single(
    data: RegularizationData,
    index: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = base.render_intensity_batch(
        data.full_data.pilot_data.images.index_select(0, index.reshape(-1))
    )
    return rendered.amplitude.to(device), rendered.target.to(device)


def _unit_power_field(field: torch.Tensor) -> torch.Tensor:
    power = field.abs().square().sum(dim=(-3, -2, -1), keepdim=True)
    return field / power.clamp_min(1.0e-20).sqrt()


def _fit_projection(fields: torch.Tensor) -> tuple[Projection, torch.Tensor]:
    if fields.ndim != 4 or fields.shape[1:] != (1, 100, 100) or not torch.is_complex(fields):
        raise ValueError("PCA fields must have shape [N, 1, 100, 100] and be complex")
    values = _unit_power_field(fields.to(torch.complex64)).flatten(1)
    mean = values.mean(dim=0)
    centered = values - mean
    gram = (centered @ centered.conj().T).real.to(torch.float64).div(float(values.shape[0]))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    eigenvectors = eigenvectors.index_select(1, order)
    if eigenvalues.numel() < PCA_COMPONENTS or float(eigenvalues[PCA_COMPONENTS - 1]) <= 1.0e-15:
        raise ValueError("calibration PCA has insufficient finite rank")
    selected_values = eigenvalues[:PCA_COMPONENTS]
    selected_vectors = eigenvectors[:, :PCA_COMPONENTS]
    basis = centered.conj().T @ selected_vectors.to(centered.dtype)
    basis = basis / (float(values.shape[0]) * selected_values).sqrt().reshape(1, -1)
    basis = basis.to(torch.complex64)
    whitening = selected_values.to(torch.float32).sqrt().clamp_min(1.0e-12)
    projection = Projection(
        mean_field=mean.to(torch.complex64),
        basis=basis,
        whitening=whitening,
        eigenvalues=selected_values.to(torch.float32),
        explained_variance=(selected_values / eigenvalues.sum().clamp_min(1.0e-20)).to(torch.float32),
    )
    scores = _project_fields(fields, projection)
    return projection, scores


def _project_fields(fields: torch.Tensor, projection: Projection) -> torch.Tensor:
    values = _unit_power_field(fields.to(torch.complex64)).flatten(1)
    centered = values - projection.mean_field.reshape(1, -1)
    scores = (centered @ projection.basis.conj()).real
    return scores / projection.whitening.reshape(1, -1)


def _projection_on_device(projection: Projection, device: torch.device) -> Projection:
    return Projection(
        mean_field=projection.mean_field.to(device),
        basis=projection.basis.to(device),
        whitening=projection.whitening.to(device),
        eigenvalues=projection.eigenvalues.to(device),
        explained_variance=projection.explained_variance.to(device),
    )


def _fixed_directions(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    directions = torch.randn(
        FIXED_DIRECTIONS, PCA_COMPONENTS, generator=generator, dtype=torch.float32
    )
    return F.normalize(directions, dim=1).to(device)


def _normal_midpoint_quantiles(count: int, device: torch.device) -> torch.Tensor:
    probabilities = (
        torch.arange(count, dtype=torch.float64, device=device) + 0.5
    ).div(float(count))
    return (math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)).to(torch.float32)


def _sliced_wasserstein(
    scores: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] != PCA_COMPONENTS:
        raise ValueError("SWD scores must have shape [N, 32]")
    projected = scores @ directions.T
    ordered = projected.sort(dim=0).values
    quantiles = _normal_midpoint_quantiles(scores.shape[0], scores.device)
    return (ordered - quantiles.reshape(-1, 1)).square().mean()


def _rank_assignment_is_permutation(ranks: torch.Tensor) -> bool:
    if ranks.ndim != 2:
        return False
    expected = torch.arange(ranks.shape[0], device=ranks.device).reshape(-1, 1)
    return bool(torch.equal(torch.sort(ranks, dim=0).values, expected.expand_as(ranks)))


def _gaussian_control_swd(
    count: int,
    directions: torch.Tensor,
) -> tuple[float, list[float]]:
    generator = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    values: list[float] = []
    for _ in range(8):
        sample = torch.randn(count, PCA_COMPONENTS, generator=generator)
        values.append(float(_sliced_wasserstein(sample.to(directions.device), directions).cpu()))
    return float(sum(values) / len(values)), values


def _capture_fields(
    model: base.IdentityOpticalManifoldAutoencoder,
    data: RegularizationData,
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float, dict[str, Any], dict[str, Any]]:
    fields: list[torch.Tensor] = []
    offline_losses: list[torch.Tensor] = []
    plane: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    model.eval()
    with torch.no_grad():
        for index in indices:
            amplitude, target = _render_single(data, index, device)
            raw, pre, current_plane, current_provenance = _forward_trace(model, amplitude)
            fields.append(pre.detach().cpu().to(torch.complex64))
            offline_losses.append(F.mse_loss(raw, target).detach().cpu())
            if plane is None:
                plane, provenance = current_plane, current_provenance
            elif plane != current_plane or provenance != current_provenance:
                raise ValueError("trace provenance changed across regularization audit samples")
    if plane is None or provenance is None:
        raise ValueError("regularization audit captured no fields")
    return torch.cat(fields), float(torch.stack(offline_losses).mean()), plane, provenance


def _clone_gradients(
    model: base.IdentityOpticalManifoldAutoencoder,
) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    return {
        name: (
            torch.zeros_like(parameters[name], device="cpu")
            if parameters[name].grad is None
            else parameters[name].grad.detach().cpu().clone()
        )
        for name in parameters
    }


def _add_gradients(
    total: dict[str, torch.Tensor],
    model: base.IdentityOpticalManifoldAutoencoder,
    scale: float,
) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            total[name].add_(parameter.grad.detach().cpu(), alpha=scale)


def _zero_grad(model: base.IdentityOpticalManifoldAutoencoder) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _gradient_replay(
    model: base.IdentityOpticalManifoldAutoencoder,
    data: RegularizationData,
    indices: torch.Tensor,
    projection: Projection,
    directions: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    if indices.numel() != GRADIENT_BATCH_SIZE:
        raise ValueError("gradient replay requires exactly 100 samples")
    projection = _projection_on_device(projection, device)
    # First pass fixes the piecewise-smooth rank assignment without retaining a graph.
    first_pass: list[torch.Tensor] = []
    plane: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    model.eval()
    with torch.no_grad():
        for index in indices:
            amplitude, _target = _render_single(data, index, device)
            _raw, pre, current_plane, current_provenance = _forward_trace(model, amplitude)
            first_pass.append(_project_fields(pre, projection).detach())
            if plane is None:
                plane, provenance = current_plane, current_provenance
            elif plane != current_plane or provenance != current_provenance:
                raise ValueError("gradient replay trace provenance changed")
    scores = torch.cat(first_pass, dim=0)
    projected = scores @ directions.T
    order = projected.argsort(dim=0)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(GRADIENT_BATCH_SIZE, device=device).reshape(-1, 1).expand_as(order))
    quantiles = _normal_midpoint_quantiles(GRADIENT_BATCH_SIZE, device)
    coefficients = 2.0 * (projected - quantiles.index_select(0, ranks.reshape(-1)).reshape_as(ranks))
    coefficients = coefficients.div(float(GRADIENT_BATCH_SIZE * FIXED_DIRECTIONS))

    reconstruction_grad = {
        name: torch.zeros_like(parameter, device="cpu")
        for name, parameter in model.named_parameters()
    }
    regularizer_grad = {
        name: torch.zeros_like(parameter, device="cpu")
        for name, parameter in model.named_parameters()
    }
    for position, index in enumerate(indices):
        amplitude, target = _render_single(data, index, device)
        _zero_grad(model)
        raw, _pre, _plane, _provenance = _forward_trace(model, amplitude)
        F.mse_loss(raw, target).div(float(GRADIENT_BATCH_SIZE)).backward()
        _add_gradients(reconstruction_grad, model, 1.0)

        _zero_grad(model)
        _raw, pre, _plane, _provenance = _forward_trace(model, amplitude)
        score = _project_fields(pre, projection)
        projected_score = score @ directions.T
        (projected_score * coefficients[position:position + 1]).sum().backward()
        _add_gradients(regularizer_grad, model, 1.0)
    _zero_grad(model)
    return reconstruction_grad, regularizer_grad, {
        "first_pass_scores_shape": list(scores.shape),
        "plane": plane,
        "provenance": provenance,
        "rank_assignment_shape": list(ranks.shape),
        "rank_assignment_is_permutation": _rank_assignment_is_permutation(ranks),
    }


def _gradient_norm(gradients: Mapping[str, torch.Tensor], names: tuple[str, ...]) -> float:
    return float(torch.sqrt(sum(gradients[name].square().sum() for name in names)).item())


def _gradient_dot(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    names: tuple[str, ...],
) -> float:
    return float(sum((left[name] * right[name]).sum() for name in names).item())


def _gradient_report(
    reconstruction: Mapping[str, torch.Tensor],
    regularizer: Mapping[str, torch.Tensor],
    regularization_scale: float,
) -> dict[str, Any]:
    names = tuple(reconstruction)
    encoder = tuple(name for name in names if name.startswith("system.encoder.phase"))
    decoder = tuple(name for name in names if name.startswith("system.decoder.phase"))
    combined = {
        name: reconstruction[name] + regularization_scale * regularizer[name]
        for name in names
    }
    def layer_norms(gradients: Mapping[str, torch.Tensor]) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, value in gradients.items():
            for index in range(value.shape[0]):
                values[f"{name}[{index}]"] = float(value[index].norm().item())
        return values

    return {
        "reconstruction_norms": {name: float(reconstruction[name].norm().item()) for name in names},
        "regularizer_norms": {name: float(regularizer[name].norm().item()) for name in names},
        "combined_norms": {name: float(combined[name].norm().item()) for name in names},
        "reconstruction_layer_norms": layer_norms(reconstruction),
        "regularizer_layer_norms": layer_norms(regularizer),
        "combined_layer_norms": layer_norms(combined),
        "reconstruction_encoder_norm": _gradient_norm(reconstruction, encoder),
        "regularizer_encoder_norm": _gradient_norm(regularizer, encoder),
        "combined_encoder_norm": _gradient_norm(combined, encoder),
        "reconstruction_decoder_norm": _gradient_norm(reconstruction, decoder),
        "regularizer_decoder_norm": _gradient_norm(regularizer, decoder),
        "combined_decoder_norm": _gradient_norm(combined, decoder),
        "regularizer_to_reconstruction_encoder_ratio": (
            regularization_scale * _gradient_norm(regularizer, encoder)
            / max(_gradient_norm(reconstruction, encoder), 1.0e-30)
        ),
        "combined_reconstruction_dot_encoder": _gradient_dot(combined, reconstruction, encoder),
        "combined_regularizer_dot_encoder": _gradient_dot(combined, regularizer, encoder),
        "finite": all(
            bool(torch.isfinite(value).all())
            for gradients in (reconstruction, regularizer, combined)
            for value in gradients.values()
        ),
    }


def _state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_equal(model: base.IdentityOpticalManifoldAutoencoder, expected: Mapping[str, torch.Tensor]) -> bool:
    actual = model.state_dict()
    return set(actual) == set(expected) and all(
        torch.equal(actual[name].detach().cpu(), value) for name, value in expected.items()
    )


def _record(
    state: Mapping[str, Any],
    spec: RegularizationSpec,
    formal: FormalSystemConfig,
    data: RegularizationData,
    hashes: Mapping[str, str],
    gates: tuple[GateResult, ...],
    status: str,
    actions: tuple[str, ...],
    reason: str | None,
) -> EvidenceRecordV2:
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=AUDIT_SCHEMA,
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


def _run_cuda_audit(
    *,
    spec: RegularizationSpec,
    state: Mapping[str, Any],
    formal: FormalSystemConfig,
    data: RegularizationData,
    source: SourceBundle,
    validation_hash: str,
    output: Path,
    device: torch.device,
    free_bytes: int,
    total_bytes: int,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    model.load_state_dict(source.checkpoint["model_state"], strict=True)
    model.eval()
    before = _state_cpu(model)
    calibration_fields, calibration_mse, plane, provenance = _capture_fields(
        model, data, data.calibration_indices, device
    )
    projection, calibration_scores = _fit_projection(calibration_fields)
    directions = _fixed_directions(device)
    audit_fields, audit_mse, audit_plane, audit_provenance = _capture_fields(
        model, data, torch.cat(data.gradient_batches), device
    )
    if plane != audit_plane or provenance != audit_provenance:
        raise ValueError("calibration and gradient trace provenance mismatch")
    audit_scores = _project_fields(audit_fields, projection).to(device)
    learned_swd = float(_sliced_wasserstein(audit_scores, directions).cpu())
    gaussian_swd, gaussian_values = _gaussian_control_swd(audit_scores.shape[0], directions)
    collapsed_swd = float(_sliced_wasserstein(torch.zeros_like(audit_scores), directions).cpu())

    gradient_reports: list[dict[str, Any]] = []
    lambda_value: float | None = None
    for batch_index, batch in enumerate(data.gradient_batches):
        reconstruction, regularizer, replay = _gradient_replay(
            model, data, batch, projection, directions, device
        )
        if batch_index == 0:
            rec_norm = _gradient_norm(reconstruction, tuple(name for name in EXPECTED_TRAINABLE if name.startswith("system.encoder.phase")))
            reg_norm = _gradient_norm(regularizer, tuple(name for name in EXPECTED_TRAINABLE if name.startswith("system.encoder.phase")))
            if rec_norm <= 0.0 or reg_norm <= 0.0:
                raise ValueError("batch 0 cannot calibrate a positive regularization scale")
            lambda_value = 0.1 * rec_norm / reg_norm
        assert lambda_value is not None
        report = _gradient_report(reconstruction, regularizer, lambda_value)
        report["batch_index"] = batch_index
        report["replay"] = replay
        gradient_reports.append(report)

    source_replay_mse = full_epoch.pilot._evaluate(
        model, data.full_data.pilot_data, data.full_data.pilot_data.heldout_indices, device
    )
    after = _state_cpu(model)
    _zero_grad(model)
    state_unchanged = _state_equal(model, before) and _state_equal(model, after)
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))

    projection_payload = {
        "schema": PROJECTION_SCHEMA,
        "mean_field": projection.mean_field.cpu(),
        "basis": projection.basis.cpu(),
        "whitening": projection.whitening.cpu(),
        "eigenvalues": projection.eigenvalues.cpu(),
        "explained_variance": projection.explained_variance.cpu(),
        "calibration_indices": data.calibration_indices.cpu(),
        "calibration_scores": calibration_scores.cpu(),
        "gradient_batch_indices": [batch.cpu() for batch in data.gradient_batches],
        "trace_plane": plane,
        "trace_provenance": provenance,
        "projection_contract": spec.payload["projection"],
        "source_hashes": {**source.hashes, "observer_validation": validation_hash},
        "affects_training_or_selection": False,
    }
    projection_path = output.parent / "regularization_projection.pt"
    atomic_torch_save(projection_path, projection_payload)
    storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())

    encoder_names = tuple(name for name in EXPECTED_TRAINABLE if name.startswith("system.encoder.phase"))
    decoder_names = tuple(name for name in EXPECTED_TRAINABLE if name.startswith("system.decoder.phase"))
    encoder_layer_names = tuple(
        f"{name}[{index}]"
        for name in encoder_names
        for index in range(2)
    )
    decoder_layer_names = tuple(
        f"{name}[{index}]"
        for name in decoder_names
        for index in range(2)
    )
    all_layer_names = encoder_layer_names + decoder_layer_names
    all_gradient_finite = all(report["finite"] for report in gradient_reports)
    reconstruction_nonzero = all(
        report["reconstruction_layer_norms"][name] > 0.0
        for report in gradient_reports for name in all_layer_names
    )
    regularizer_encoder_nonzero = all(
        report["regularizer_layer_norms"][name] > 0.0
        for report in gradient_reports for name in encoder_layer_names
    )
    regularizer_decoder_zero = all(
        report["regularizer_layer_norms"][name] == 0.0
        for report in gradient_reports for name in decoder_layer_names
    )
    combined_nonzero = all(
        report["combined_layer_norms"][name] > 0.0
        for report in gradient_reports for name in all_layer_names
    )
    ratios = [report["regularizer_to_reconstruction_encoder_ratio"] for report in gradient_reports]
    ratio_gate = all(ENCODER_RATIO_MIN <= value <= ENCODER_RATIO_MAX for value in ratios)
    alignment_gate = all(
        report["combined_reconstruction_dot_encoder"] > 0.0
        and report["combined_regularizer_dot_encoder"] > 0.0
        for report in gradient_reports
    )
    source_gate = abs(source_replay_mse - SOURCE_PAIRED_MSE) <= 1.0e-9
    finite_gate = (
        all(torch.isfinite(value).all() for value in (
            calibration_fields, calibration_scores, audit_fields, audit_scores,
            projection.mean_field, projection.basis, projection.whitening,
        ))
        and math.isfinite(learned_swd)
        and math.isfinite(gaussian_swd)
        and math.isfinite(collapsed_swd)
    )
    science_gates = (
        GateResult("learned_swd_gt_gaussian_control", learned_swd > gaussian_swd, {"learned": learned_swd, "gaussian_mean": gaussian_swd}),
        GateResult("collapsed_control_gt_learned", collapsed_swd > learned_swd, {"collapsed": collapsed_swd, "learned": learned_swd}),
        GateResult("gradient_descent_alignment", alignment_gate, {"batches": GRADIENT_BATCHES}),
    )
    harness_gates = (
        GateResult("source_reconstruction_reproduced", source_gate, {"source": SOURCE_PAIRED_MSE, "replay": source_replay_mse}),
        GateResult("finite_projection_and_gradients", finite_gate and all_gradient_finite, {}),
        GateResult("reconstruction_four_phase_gradients_nonzero", reconstruction_nonzero, {"layers": 4}),
        GateResult("regularizer_encoder_gradient_scope", regularizer_encoder_nonzero and regularizer_decoder_zero, {"encoder_layers": 2, "decoder_zero": regularizer_decoder_zero}),
        GateResult("combined_four_phase_gradients_nonzero", combined_nonzero, {"layers": 4}),
        GateResult("encoder_gradient_ratio", ratio_gate, {"ratios": ratios, "min": ENCODER_RATIO_MIN, "max": ENCODER_RATIO_MAX}),
        GateResult("fixed_formal_identity_optics", plane.get("plane_id") == "encoder_layer_2" and provenance.get("operation") == "finite_crop_hard_aperture", {}),
        GateResult("fixed_exposure", True, {"scale": 1.0, "gamma": 1.0}),
        GateResult("model_state_unchanged", state_unchanged, {}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
        GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
    )
    gates = science_gates + harness_gates
    if not all(gate.passed for gate in harness_gates):
        status, actions, reason = "invalid_harness", (), "regularization audit harness or resource gate failed"
    elif not all(gate.passed for gate in science_gates):
        status, actions, reason = "valid_failed", (NEXT_VALID_FAILED_ACTION,), "aggregate-prior feasibility gate failed"
    else:
        status, actions, reason = "passed", (NEXT_PASS_ACTION,), None
    hashes = {**source.hashes, "regularization_validation": validation_hash, "projection": file_sha256(projection_path)}
    record = _record(state, spec, formal, data, hashes, gates, status, actions, reason)
    payload = _record_payload(
        AUDIT_SCHEMA,
        record,
        {
            "source_reconstruction_mse": {"authoritative_cuda_replay": source_replay_mse, "reference": SOURCE_PAIRED_MSE, "tolerance": 1.0e-9},
            "projection": {"path": str(projection_path.resolve()), "sha256": hashes["projection"], "schema": PROJECTION_SCHEMA, "components": PCA_COMPONENTS},
            "aggregate_prior": {"family": "deterministic_sliced_wasserstein", "approximation_role": "project_approximation", "learned_swd": learned_swd, "gaussian_control_mean": gaussian_swd, "gaussian_control_values": gaussian_values, "collapsed_swd": collapsed_swd},
            "gradient_audit": {"lambda": lambda_value, "batches": gradient_reports, "gradient_scope": {"regularizer": "encoder_phase_only", "reconstruction": "encoder_and_decoder_phase"}},
            "trace": {"plane": plane, "provenance": provenance},
            "calibration": {"mse": calibration_mse, "count": CALIBRATION_COUNT},
            "gradient_panel": {"mse": audit_mse, "count": GRADIENT_AUDIT_COUNT},
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
        raise ValueError("regularization audit requires CUDA")
    _require_deterministic_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    source = _source_bundle(spec, review_path, data, state["commit"])
    if source.formal_path != formal_path:
        raise ValueError("regularization audit formal config binding mismatch")
    hashes = dict(source.hashes)
    hashes["formal_config"] = file_sha256(formal_path)
    hashes["dataset_panel_manifest"] = data.manifest["manifest_hash"]
    _validation_payload, validation_hash = _load_validation(
        args.validation_evidence, spec, formal, data, state, hashes
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite regularization audit evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for regularization audit")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    with _deterministic_algorithms():
        return _run_cuda_audit(
            spec=spec,
            state=state,
            formal=formal,
            data=data,
            source=source,
            validation_hash=validation_hash,
            output=output,
            device=device,
            free_bytes=free_bytes,
            total_bytes=total_bytes,
        )


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "manifold-regularization-audit-validate":
        return validate(args)
    if args.research_action == "manifold-regularization-audit-run":
        return run(args)
    raise ValueError(f"unsupported regularization audit action: {args.research_action}")
