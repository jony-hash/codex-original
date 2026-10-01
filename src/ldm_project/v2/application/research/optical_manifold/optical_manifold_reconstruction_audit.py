"""No-update Identity reconstruction attribution audit.

This lifecycle closes the reconstruction stage of H-LOMA-I0.  It consumes the
authoritative full-epoch checkpoint, evaluates fixed phase and template
counterfactuals, and never creates an optimizer or checkpoint.  A passing
record only permits manual review of a later latent-observer contract.
"""

from __future__ import annotations

import argparse
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

from ldm_project.v2.config import FormalSystemConfig
from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, safe_torch_load
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_full_epoch as full_epoch


BRANCH = "research/v2-lightgen-optical-manifold-identity-attribution-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_identity_reconstruction_audit_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_identity_reconstruction_audit_review.v1"
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_identity_reconstruction_audit_validation.v1"
)
AUDIT_SCHEMA = "lightgen_optical_manifold_identity_reconstruction_audit.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_identity_reconstruction_audit_panel.v1"
SOURCE_TRAINING_SCHEMA = "lightgen_optical_manifold_identity_full_epoch_training.v1"
SOURCE_CHECKPOINT_SCHEMA = (
    "lightgen_optical_manifold_identity_full_epoch_resume.v1"
)
SOURCE_TRAINING_HASH = (
    "c01dae121210a4dff0534a9979661f92d1e0fc749087c8451ad51be24a458948"
)
SOURCE_CHECKPOINT_HASH = (
    "56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f"
)
SOURCE_MANIFEST_HASH = (
    "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
)
SOURCE_FULL_EPOCH_CONFIG_HASH = (
    "f59e79090704319704e9e9161587003d369ed18ad4b502facef202d3b2a954a1"
)
SOURCE_STEP = 5500
SOURCE_PAIRED_MSE = 0.03437697887420654
SOURCE_PERMUTED_MSE = 0.06315752118825912
EXPECTED_HELDOUT = 1000
EXPECTED_HELDOUT_PER_LABEL = 100
EXPECTED_VISUAL = 100
EXPECTED_VISUAL_PER_LABEL = 10
RANDOM_PHASE_SEED = 1234
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_WALL_SECONDS = 1800.0
MAX_STORAGE_BYTES = 512 * 1024**2
NEXT_PASS_ACTION = "manual_review_identity_latent_observer_contract"
NEXT_VALID_FAILED_ACTION = "manual_review_identity_reconstruction_objective_contract"
EXPECTED_FORMAL_KIND = base.EXPECTED_FORMAL_KIND


@dataclass(frozen=True)
class AuditSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class AuditData:
    pilot_data: full_epoch.pilot.PilotData
    images: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    heldout_indices: torch.Tensor
    visual_indices: torch.Tensor
    manifest: dict[str, Any]


@dataclass(frozen=True)
class Evaluation:
    raw_cmos: torch.Tensor
    reconstruction: torch.Tensor
    metrics: dict[str, Any]


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"]:
        raise ValueError("Identity reconstruction audit requires a clean worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"Identity reconstruction audit requires branch {BRANCH}")


def _expected_config_payload() -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "formal_system_config": "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml",
        "source_training_evidence": "outputs/20260827_v2/runs/lightgen_optical_manifold_identity_full_epoch_v1_attempt_04/epoch_training.json",
        "source_training_evidence_sha256": SOURCE_TRAINING_HASH,
        "source_training_config": "configs/research/optical_manifold/lightgen_optical_manifold_identity_full_epoch_v1.yaml",
        "source_training_config_sha256": SOURCE_FULL_EPOCH_CONFIG_HASH,
        "source_checkpoint": "outputs/20260827_v2/runs/lightgen_optical_manifold_identity_full_epoch_v1_attempt_04/full_epoch_final_step_005500.pt",
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "source_dataset_manifest_hash": SOURCE_MANIFEST_HASH,
        "source_checkpoint_step": SOURCE_STEP,
        "dataset": {
            "train_count": 55000,
            "validation_count": 5000,
            "heldout_count": EXPECTED_HELDOUT,
            "heldout_per_label": EXPECTED_HELDOUT_PER_LABEL,
            "visual_panel_count": EXPECTED_VISUAL,
            "visual_panel_per_label": EXPECTED_VISUAL_PER_LABEL,
            "visual_panel_selection": "first_occurrence_in_heldout_panel",
            "train_template_source": "fixed_55k_train_split",
            "input_shape": [1, 28, 28],
            "rendered_shape": [1, 200, 200],
            "no_training": True,
            "label_role": "offline_grouping_and_template_baseline_only",
        },
        "optical_contract": {
            "formal_system_kind": EXPECTED_FORMAL_KIND,
            "wavelength_m": 5.32e-7,
            "slm_shape": [200, 200],
            "slm_pixel_pitch_m": 2.5e-6,
            "metasurface_shape": [100, 100],
            "metasurface_pixel_pitch_m": 2.0e-6,
            "propagation_grid_size": 2048,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "ols": "identity",
            "boundary": "finite_crop",
            "front_relay": "coherent_imaging_4f",
            "back_relay": "free_space",
            "input_encoding": "intensity_sqrt_amplitude",
            "target": "camera_intensity_same_fov",
            "exposure": {"scale": 1.0, "gamma": 1.0},
        },
        "controls": {
            "trained": "learned_full_epoch_checkpoint",
            "zero_phase": "encoder_and_decoder_phase_zero",
            "encoder_zero": "encoder_phase_zero_decoder_learned",
            "decoder_zero": "encoder_learned_decoder_phase_zero",
            "random_phase": "uniform_phase_both_stacks_seed_1234",
            "cyclic_target": "target_cyclic_shift_by_one",
            "global_template": "train_split_global_mean",
            "class_templates": "train_split_per_label_mean",
            "ideal_reference": "paired_target_identity_only",
        },
        "metrics": {
            "mse": "paired_and_cyclic_target_mse",
            "sample_correlation": "centered_pearson",
            "output_variance": "global_and_within_label_variance_retention",
            "instance_geometry": "within_label_pairwise_distance_correlation",
            "retrieval": "exact_target_and_same_label_top1",
        },
        "gates": {
            "source_checkpoint_step": SOURCE_STEP,
            "source_metric_tolerance": 1.0e-9,
            "trained_below_cyclic": True,
            "trained_below_global_template": True,
            "trained_below_class_templates": True,
            "trained_below_all_phase_controls": True,
            "sample_correlation_positive_and_control_superior": True,
            "within_label_geometry_positive_and_control_superior": True,
            "exact_retrieval_above_templates_and_controls": True,
            "model_state_unchanged": True,
            "evaluator_calls": 0,
            "min_free_cuda_bytes": MIN_FREE_BYTES,
            "max_peak_cuda_bytes": MAX_PEAK_BYTES,
            "max_wall_seconds": MAX_WALL_SECONDS,
            "max_storage_bytes": MAX_STORAGE_BYTES,
        },
        "authorization": {
            "reviewer_role": "project_owner",
            "decision": "approve_bounded_identity_reconstruction_audit",
            "no_update_audit_authorized": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal_promotion": False,
            "post_pass_next_action": NEXT_PASS_ACTION,
            "post_valid_failed_next_action": NEXT_VALID_FAILED_ACTION,
            "prohibited": [
                "optimizer",
                "checkpoint",
                "latent_observer",
                "kl",
                "smf",
                "fmf",
                "evaluator",
                "generation",
                "promotion",
            ],
        },
    }


def _load_spec(path: str | Path) -> AuditSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload != _expected_config_payload():
        raise ValueError("reconstruction audit config does not exactly match contract")
    source_config = _resolve(payload["source_training_config"], base_path=target.parent)
    source_spec = full_epoch._load_spec(source_config)
    _, source_formal_path = full_epoch._load_formal(source_spec)
    formal_path = _resolve(payload["formal_system_config"], base_path=target.parent)
    if formal_path != source_formal_path:
        raise ValueError("audit formal config does not match source formal config")
    return AuditSpec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: AuditSpec) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_bounded_identity_reconstruction_audit",
        "source_training_evidence": spec.payload["source_training_evidence"],
        "source_training_evidence_sha256": SOURCE_TRAINING_HASH,
        "source_training_config": spec.payload["source_training_config"],
        "source_training_config_sha256": SOURCE_FULL_EPOCH_CONFIG_HASH,
        "source_checkpoint": spec.payload["source_checkpoint"],
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "source_dataset_manifest_hash": SOURCE_MANIFEST_HASH,
        "approved_contract": {
            "heldout_count": EXPECTED_HELDOUT,
            "controls": [
                "zero_phase",
                "encoder_zero",
                "decoder_zero",
                "random_phase_seed_1234",
                "cyclic_target",
                "global_template",
                "class_templates",
            ],
            "no_model_update": True,
            "source_metric_tolerance": 1.0e-9,
            "max_wall_seconds": MAX_WALL_SECONDS,
            "post_pass_next_action": NEXT_PASS_ACTION,
            "post_valid_failed_next_action": NEXT_VALID_FAILED_ACTION,
        },
        "no_update_audit_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("project-owner review contract does not match audit scope")
    return payload, target


def _load_formal(spec: AuditSpec) -> tuple[FormalSystemConfig, Path]:
    source_config = _resolve(spec.payload["source_training_config"], base_path=spec.path.parent)
    source_spec = full_epoch._load_spec(source_config)
    formal, formal_path = full_epoch._load_formal(source_spec)
    expected_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    if formal_path != expected_path:
        raise ValueError("audit formal config path mismatch")
    return formal, formal_path


def _assert_ancestor(commit: str, current: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("source evidence commit is not an ancestor of audit branch")


def _load_source(
    spec: AuditSpec,
    review_path: Path,
    training_path: str | Path,
    checkpoint_path: str | Path,
    current_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    training_target = _resolve(training_path)
    checkpoint_target = _resolve(checkpoint_path)
    expected_training = _resolve(spec.payload["source_training_evidence"])
    expected_checkpoint = _resolve(spec.payload["source_checkpoint"])
    if training_target != expected_training or checkpoint_target != expected_checkpoint:
        raise ValueError("audit must consume the registered source evidence paths")
    if file_sha256(training_target) != SOURCE_TRAINING_HASH:
        raise ValueError("full-epoch training evidence hash mismatch")
    if file_sha256(checkpoint_target) != SOURCE_CHECKPOINT_HASH:
        raise ValueError("full-epoch checkpoint hash mismatch")
    payload = json.loads(training_target.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SOURCE_TRAINING_SCHEMA
        or payload.get("hypothesis") != HYPOTHESIS
        or payload.get("training_authorized") is not False
        or payload.get("formal_promotion") is not False
    ):
        raise ValueError("source training evidence has invalid boundary")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != SOURCE_TRAINING_SCHEMA or record.status != "passed":
        raise ValueError("source training evidence must be passed")
    _assert_ancestor(record.git_commit, current_commit)
    if payload.get("contract", {}).get("final_checkpoint") != str(checkpoint_target.resolve()):
        raise ValueError("source training evidence checkpoint binding mismatch")
    contract = payload.get("contract", {})
    heldout = contract.get("heldout", {})
    if (
        contract.get("global_step_final") != SOURCE_STEP
        or heldout.get("final_paired_mse") != SOURCE_PAIRED_MSE
        or heldout.get("final_permuted_target_mse") != SOURCE_PERMUTED_MSE
    ):
        raise ValueError("source training metrics or step changed")
    checkpoint = safe_torch_load(
        checkpoint_target,
        expected_sha256=SOURCE_CHECKPOINT_HASH,
        map_location="cpu",
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("source checkpoint must be a mapping")
    if (
        checkpoint.get("schema") != SOURCE_CHECKPOINT_SCHEMA
        or checkpoint.get("global_step") != SOURCE_STEP
        or checkpoint.get("continuation_cursor") != 4500
        or checkpoint.get("config_hash") != SOURCE_FULL_EPOCH_CONFIG_HASH
        or checkpoint.get("continuation_schedule_sha256")
        != full_epoch.CONTINUATION_SCHEDULE_HASH
        or checkpoint.get("source_checkpoint_sha256")
        != full_epoch.SOURCE_CHECKPOINT_HASH
        or len(checkpoint.get("loss_history", [])) != SOURCE_STEP
        or checkpoint.get("data_cursor")
        != {
            "initial_global_step": 1000,
            "next_global_step": SOURCE_STEP + 1,
            "continuation_cursor": 4500,
        }
    ):
        raise ValueError("source checkpoint state binding mismatch")
    if not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError("source checkpoint model state is missing")
    if not isinstance(checkpoint.get("optimizer_state"), dict):
        raise ValueError("source checkpoint Adam state is missing")
    if not isinstance(checkpoint.get("rng_state"), dict):
        raise ValueError("source checkpoint RNG state is missing")
    if (
        len(checkpoint["optimizer_state"].get("param_groups", [])) != 1
        or len(checkpoint["optimizer_state"].get("state", {})) != len(base.EXPECTED_TRAINABLE)
        or not torch.is_tensor(checkpoint["rng_state"].get("cpu"))
        or not isinstance(checkpoint["rng_state"].get("cuda"), list)
    ):
        raise ValueError("source checkpoint optimizer or RNG state is incomplete")
    evidence_hashes = checkpoint.get("evidence_hashes")
    expected_evidence_hashes = {
        "pilot_training_evidence": full_epoch.SOURCE_TRAINING_HASH,
        "pilot_checkpoint": full_epoch.SOURCE_CHECKPOINT_HASH,
        "full_epoch_validation_evidence": contract.get("epoch_validation_hash"),
        "full_epoch_preflight_evidence": contract.get("epoch_preflight_hash"),
    }
    if evidence_hashes != expected_evidence_hashes:
        raise ValueError("source checkpoint evidence binding mismatch")
    return payload, checkpoint, training_target, checkpoint_target


def _load_data(dataset_root: str | Path) -> AuditData:
    full_data = full_epoch._load_data(dataset_root)
    if not full_epoch.pilot.manifest_matches_hash(
        full_data.pilot_data.manifest, SOURCE_MANIFEST_HASH
    ):
        raise ValueError("MNIST source manifest changed")
    if full_data.visual_indices.numel() != EXPECTED_VISUAL:
        raise ValueError("visual panel count mismatch")
    if full_data.pilot_data.heldout_indices.numel() != EXPECTED_HELDOUT:
        raise ValueError("held-out panel count mismatch")
    if full_data.manifest["train_count"] != 55000 or full_data.manifest["validation_count"] != 5000:
        raise ValueError("full-epoch dataset split counts changed")
    labels = full_data.pilot_data.labels
    heldout_labels = labels.index_select(0, full_data.pilot_data.heldout_indices)
    expected = torch.arange(10).repeat_interleave(EXPECTED_HELDOUT_PER_LABEL)
    if not torch.equal(heldout_labels, expected):
        raise ValueError("held-out panel is not ordered by label")
    return AuditData(
        pilot_data=full_data.pilot_data,
        images=full_data.pilot_data.images,
        labels=labels,
        train_indices=full_data.pilot_data.train_indices,
        heldout_indices=full_data.pilot_data.heldout_indices,
        visual_indices=full_data.visual_indices,
        manifest={
            "schema": PANEL_SCHEMA,
            "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
            "full_epoch_manifest_hash": full_data.manifest["manifest_hash"],
            "train_count": int(full_data.pilot_data.train_indices.numel()),
            "validation_count": int(full_data.pilot_data.validation_indices.numel()),
            "heldout_count": int(full_data.pilot_data.heldout_indices.numel()),
            "heldout_per_label": EXPECTED_HELDOUT_PER_LABEL,
            "heldout_indices_sha256": base.tensor_sha256(full_data.pilot_data.heldout_indices),
            "visual_indices_sha256": base.tensor_sha256(full_data.visual_indices),
            "visual_count": EXPECTED_VISUAL,
            "visual_per_label": EXPECTED_VISUAL_PER_LABEL,
            "visual_selection": "first_occurrence_in_heldout_panel",
        },
    )


def _parameter_contract(model: base.IdentityOpticalManifoldAutoencoder) -> None:
    actual = tuple(name for name, _ in model.named_parameters())
    if tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad) != base.EXPECTED_TRAINABLE:
        raise ValueError("audit model trainable parameter contract failed")
    if not actual or any(
        parameter.requires_grad != (name in base.EXPECTED_TRAINABLE)
        for name, parameter in model.named_parameters()
    ):
        raise ValueError("audit model parameter freeze contract failed")


def _state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_equal(
    model: base.IdentityOpticalManifoldAutoencoder,
    expected: Mapping[str, torch.Tensor],
) -> bool:
    actual = model.state_dict()
    return set(actual) == set(expected) and all(
        torch.equal(actual[name].detach().cpu(), value) for name, value in expected.items()
    )


def _context(
    spec: AuditSpec,
    formal: FormalSystemConfig,
    data: AuditData,
) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=data.manifest["full_epoch_manifest_hash"],
        calibration_hash=canonical_sha256(
            {"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}
        ),
        exposure_scale=1.0,
    )


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
        "no_update_audit_authorized": True,
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
        raise FileExistsError(f"refusing to overwrite audit evidence: {target}")
    atomic_write_json(target, payload)


def _source_hashes(
    spec: AuditSpec,
    review_path: Path,
    formal_path: Path,
    training_path: Path,
    checkpoint_path: Path,
    data: AuditData,
) -> dict[str, str]:
    source_config = _resolve(spec.payload["source_training_config"], base_path=spec.path.parent)
    return {
        "audit_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "formal_config": file_sha256(formal_path),
        "source_training_config": file_sha256(source_config),
        "source_training_evidence": file_sha256(training_path),
        "source_checkpoint": file_sha256(checkpoint_path),
        "source_mnist_manifest": SOURCE_MANIFEST_HASH,
        "full_epoch_manifest": data.manifest["full_epoch_manifest_hash"],
    }


def _validation_contract(
    spec: AuditSpec,
    review_path: Path,
    formal_path: Path,
    training_path: Path,
    checkpoint_path: Path,
    data: AuditData,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    model = base.IdentityOpticalManifoldAutoencoder(_load_formal(spec)[0])
    _parameter_contract(model)
    checkpoint = safe_torch_load(checkpoint_path, expected_sha256=SOURCE_CHECKPOINT_HASH)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return {
        "branch": state["branch"],
        "commit": state["commit"],
        "formal_system_kind": EXPECTED_FORMAL_KIND,
        "system_geometry": dict(base.EXPECTED_SYSTEM),
        "input_encoding": "intensity_sqrt_amplitude",
        "target": "camera_intensity_same_fov",
        "exposure": {"scale": 1.0, "gamma": 1.0},
        "no_model_update": True,
        "source": {
            "training_evidence": str(training_path.resolve()),
            "training_evidence_sha256": SOURCE_TRAINING_HASH,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
            "checkpoint_step": SOURCE_STEP,
            "dataset_manifest_sha256": SOURCE_MANIFEST_HASH,
        },
        "trainable_parameter_names": list(base.EXPECTED_TRAINABLE),
        "data_manifest": data.manifest,
        "controls": spec.payload["controls"],
        "metrics": spec.payload["metrics"],
        "review_contract_sha256": file_sha256(review_path),
        "formal_config_sha256": file_sha256(formal_path),
        "authorization": {
            "no_update_audit_authorized": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal_promotion": False,
            "post_pass_next_action": NEXT_PASS_ACTION,
            "post_valid_failed_next_action": NEXT_VALID_FAILED_ACTION,
        },
    }


def _validation_record(
    state: Mapping[str, Any],
    spec: AuditSpec,
    formal: FormalSystemConfig,
    data: AuditData,
    input_hashes: Mapping[str, str],
) -> EvidenceRecordV2:
    gates = (
        GateResult("clean_lineage", True, {"clean": True}),
        GateResult("independent_attribution_branch", True, {"branch_matches": True}),
        GateResult("source_training_evidence_and_checkpoint", True, {"step": SOURCE_STEP}),
        GateResult("formal_geometry_and_identity_ols", True, {"grid": 2048, "identity": True}),
        GateResult("fixed_heldout_panel", True, {"count": EXPECTED_HELDOUT, "per_label": EXPECTED_HELDOUT_PER_LABEL}),
        GateResult("fixed_train_template_source", True, {"count": 55000}),
        GateResult("zero_update_scope", True, {"optimizer": False, "checkpoint": False}),
        GateResult("trainable_parameter_contract", True, {"count": len(base.EXPECTED_TRAINABLE)}),
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
        input_hashes=dict(input_hashes),
        gates=gates,
        permitted_next_actions=("run_identity_reconstruction_attribution_audit",),
    )
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    training_payload, checkpoint, training_path, checkpoint_path = _load_source(
        spec,
        review_path,
        args.source_training_evidence,
        args.source_checkpoint,
        state["commit"],
    )
    del review, training_payload
    data = _load_data(args.dataset_root)
    if checkpoint.get("dataset_manifest_hash") != data.manifest["full_epoch_manifest_hash"]:
        raise ValueError("source checkpoint full-epoch manifest mismatch")
    contract = _validation_contract(
        spec, review_path, formal_path, training_path, checkpoint_path, data, state
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit validation: {output}")
    record = _validation_record(
        state,
        spec,
        formal,
        data,
        _source_hashes(spec, review_path, formal_path, training_path, checkpoint_path, data),
    )
    payload = _record_payload(VALIDATION_SCHEMA, record, contract)
    _write_new_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(
    path: str | Path,
    spec: AuditSpec,
    review_path: Path,
    formal_path: Path,
    training_path: Path,
    checkpoint_path: Path,
    data: AuditData,
    state: Mapping[str, Any],
) -> tuple[EvidenceRecordV2, str]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != VALIDATION_SCHEMA
        or payload.get("hypothesis") != HYPOTHESIS
        or payload.get("no_update_audit_authorized") is not True
        or payload.get("training_authorized") is not False
        or payload.get("formal_promotion") is not False
    ):
        raise ValueError("invalid reconstruction audit validation boundary")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed":
        raise ValueError("reconstruction audit validation must be passed")
    if record.git_commit != state["commit"] or not state["clean"] or state["branch"] != BRANCH:
        raise ValueError("reconstruction audit validation is stale")
    if record.config_hash != spec.config_hash or record.context.dataset_hash != data.manifest["full_epoch_manifest_hash"]:
        raise ValueError("reconstruction audit validation context mismatch")
    expected = _source_hashes(spec, review_path, formal_path, training_path, checkpoint_path, data)
    for key, value in expected.items():
        if record.input_hashes.get(key) != value:
            raise ValueError(f"reconstruction audit validation binding mismatch for {key}")
    return record, file_sha256(target)


def _mse_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Match the authoritative full-epoch evaluator's per-image reduction and
    # stack order so source-metric reproduction is an actual lineage check.
    return torch.stack(
        [F.mse_loss(prediction[index : index + 1], target[index : index + 1]) for index in range(prediction.shape[0])]
    )


def _centered_pearson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    left = prediction.flatten(1)
    right = target.flatten(1)
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    denominator = left.square().sum(dim=1).sqrt() * right.square().sum(dim=1).sqrt()
    return (left * right).sum(dim=1) / denominator.clamp_min(1.0e-12)


def _pairwise_distance_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float | None, list[float | None]]:
    prediction_flat = prediction.flatten(1)
    target_flat = target.flatten(1)
    values: list[float | None] = []
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        left = prediction_flat.index_select(0, indices)
        right = target_flat.index_select(0, indices)
        left_gram = left @ left.T
        right_gram = right @ right.T
        left_distance = (left.square().sum(1, keepdim=True) + left.square().sum(1, keepdim=True).T - 2.0 * left_gram).clamp_min(0.0).sqrt()
        right_distance = (right.square().sum(1, keepdim=True) + right.square().sum(1, keepdim=True).T - 2.0 * right_gram).clamp_min(0.0).sqrt()
        upper = torch.triu(torch.ones_like(left_distance, dtype=torch.bool), diagonal=1)
        x = left_distance[upper]
        y = right_distance[upper]
        x = x - x.mean()
        y = y - y.mean()
        denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
        values.append(float((x * y).sum().div(denominator).item()) if denominator > 1.0e-12 else None)
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return (sum(finite) / len(finite) if finite else None), values


def _retrieval_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    left = prediction.flatten(1).float()
    right = target.flatten(1).float()
    left_norm = left.square().sum(1, keepdim=True)
    right_norm = right.square().sum(1, keepdim=True).T
    distances = (left_norm + right_norm - 2.0 * (left @ right.T)).clamp_min(0.0)
    nearest = distances.argmin(dim=1)
    expected = torch.arange(target.shape[0])
    return {
        "exact_target_top1": float((nearest == expected).float().mean().item()),
        "same_label_top1": float((labels.index_select(0, nearest) == labels).float().mean().item()),
    }


def _variance_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    target_global = target.flatten(1).var(dim=0, unbiased=False).mean()
    prediction_global = prediction.flatten(1).var(dim=0, unbiased=False).mean()
    target_within: list[torch.Tensor] = []
    prediction_within: list[torch.Tensor] = []
    for label in range(10):
        mask = labels == label
        target_within.append(target[mask].flatten(1).var(dim=0, unbiased=False).mean())
        prediction_within.append(prediction[mask].flatten(1).var(dim=0, unbiased=False).mean())
    target_within_mean = torch.stack(target_within).mean()
    prediction_within_mean = torch.stack(prediction_within).mean()
    return {
        "global_target_pixel_variance": float(target_global.item()),
        "global_prediction_pixel_variance": float(prediction_global.item()),
        "global_variance_retention": float(prediction_global.div(target_global.clamp_min(1.0e-12)).item()),
        "within_label_target_pixel_variance": float(target_within_mean.item()),
        "within_label_prediction_pixel_variance": float(prediction_within_mean.item()),
        "within_label_variance_retention": float(prediction_within_mean.div(target_within_mean.clamp_min(1.0e-12)).item()),
    }


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    paired = _mse_per_sample(prediction, target)
    pearson = _centered_pearson(prediction, target)
    geometry, geometry_by_label = _pairwise_distance_correlation(prediction, target, labels)
    result: dict[str, Any] = {
        "paired_mse": float(paired.mean().item()),
        "paired_mse_by_label": [float(paired[labels == label].mean().item()) for label in range(10)],
        "centered_pearson_mean": float(pearson.mean().item()),
        "centered_pearson_min": float(pearson.min().item()),
        "within_label_distance_correlation": geometry,
        "within_label_distance_correlation_by_label": geometry_by_label,
        **_variance_metrics(prediction, target, labels),
        **_retrieval_metrics(prediction, target, labels),
    }
    return result


def _authoritative_source_replay_mse(
    model: base.IdentityOpticalManifoldAutoencoder,
    data: AuditData,
    device: torch.device,
) -> float:
    """Replay the exact source evaluator reduction on the trained phase state."""
    value = full_epoch.pilot._evaluate(
        model,
        data.pilot_data,
        data.heldout_indices,
        device,
    )
    model.eval()
    return value


def _source_metric_reproduction_gate(
    authoritative_mse: float,
    offline_mse: float,
) -> GateResult:
    return _gate_bool(
        "source_metrics_reproduced",
        abs(authoritative_mse - SOURCE_PAIRED_MSE) <= 1.0e-9,
        {
            "source": SOURCE_PAIRED_MSE,
            "authoritative_source_replay_mse": authoritative_mse,
            "offline_attribution_mse": offline_mse,
        },
    )


def _render_targets(data: AuditData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = data.images.index_select(0, data.heldout_indices)
    rendered = base.render_intensity_batch(images)
    return images, rendered.target, rendered.amplitude


def _template_predictions(data: AuditData, target: torch.Tensor) -> dict[str, torch.Tensor]:
    train_images = data.images.index_select(0, data.train_indices)
    global_template = base.render_intensity_batch(train_images.mean(dim=0, keepdim=True)).target
    class_templates = []
    train_labels = data.labels.index_select(0, data.train_indices)
    for label in range(10):
        class_images = train_images[train_labels == label]
        class_templates.append(
            base.render_intensity_batch(class_images.mean(dim=0, keepdim=True)).target
        )
    global_prediction = global_template.expand_as(target).clone()
    class_prediction = torch.cat(
        [class_templates[label].expand((int((data.labels.index_select(0, data.heldout_indices) == label).sum()), -1, -1, -1)) for label in range(10)],
        dim=0,
    )
    return {"global_template": global_prediction, "class_templates": class_prediction}


def _phase_variant(
    model: base.IdentityOpticalManifoldAutoencoder,
    variant: str,
) -> None:
    if variant not in {"trained", "zero_phase", "encoder_zero", "decoder_zero", "random_phase"}:
        raise ValueError(f"unsupported phase variant {variant}")
    if variant == "trained":
        return
    with torch.no_grad():
        if variant in {"zero_phase", "encoder_zero"}:
            model.system.encoder.phase.zero_()
        if variant in {"zero_phase", "decoder_zero"}:
            model.system.decoder.phase.zero_()
        if variant == "random_phase":
            generator = torch.Generator(device="cpu").manual_seed(RANDOM_PHASE_SEED)
            for parameter in (model.system.encoder.phase, model.system.decoder.phase):
                values = torch.rand(parameter.shape, generator=generator, dtype=torch.float32)
                parameter.copy_((values * (2.0 * math.pi) - math.pi).to(parameter.device))


def _restore_phases(
    model: base.IdentityOpticalManifoldAutoencoder,
    encoder_phase: torch.Tensor,
    decoder_phase: torch.Tensor,
) -> None:
    with torch.no_grad():
        model.system.encoder.phase.copy_(encoder_phase)
        model.system.decoder.phase.copy_(decoder_phase)


def _evaluate_model(
    model: base.IdentityOpticalManifoldAutoencoder,
    amplitudes: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Evaluation:
    raw_values: list[torch.Tensor] = []
    reconstruction_values: list[torch.Tensor] = []
    with torch.no_grad():
        for index in range(amplitudes.shape[0]):
            output = model(amplitudes[index : index + 1].to(device), activation_checkpoint=False)
            raw_values.append(output.raw_camera.detach().cpu().float())
            reconstruction_values.append(output.reconstruction.detach().cpu().float())
    raw = torch.cat(raw_values, dim=0)
    reconstruction = torch.cat(reconstruction_values, dim=0)
    if not torch.isfinite(raw).all() or not torch.isfinite(reconstruction).all():
        raise ValueError("audit model produced non-finite camera values")
    return Evaluation(raw, reconstruction, _metrics(reconstruction, target, labels))


def _panel_payload(
    data: AuditData,
    target: torch.Tensor,
    panel_values: Mapping[str, Evaluation],
) -> dict[str, Any]:
    labels = data.labels.index_select(0, data.visual_indices)
    visual_positions = []
    heldout_lookup = {int(value): index for index, value in enumerate(data.heldout_indices.tolist())}
    for value in data.visual_indices.tolist():
        visual_positions.append(heldout_lookup[int(value)])
    positions = torch.as_tensor(visual_positions, dtype=torch.long)
    input_images = data.images.index_select(0, data.visual_indices)
    payload: dict[str, Any] = {
        "schema": PANEL_SCHEMA,
        "indices": data.visual_indices.cpu(),
        "labels": labels.cpu(),
        "input_intensity": input_images.cpu(),
        "target": target.index_select(0, positions).cpu(),
        "selection": "first_occurrence_in_heldout_panel",
        "affects_training_or_selection": False,
        "variants": {},
    }
    for name, evaluation in panel_values.items():
        payload["variants"][name] = {
            "raw_cmos": evaluation.raw_cmos.index_select(0, positions).cpu(),
            "reconstruction": evaluation.reconstruction.index_select(0, positions).cpu(),
        }
    return payload


def _save_panel(
    run_dir: Path,
    data: AuditData,
    target: torch.Tensor,
    panel_values: Mapping[str, Evaluation],
) -> dict[str, Any]:
    panel_path = run_dir / "reconstruction_audit_panel.pt"
    target_png = run_dir / "reconstruction_audit_target_panel.png"
    trained_png = run_dir / "reconstruction_audit_trained_panel.png"
    if panel_path.exists() or target_png.exists() or trained_png.exists():
        raise FileExistsError("refusing to overwrite reconstruction audit panel")
    payload = _panel_payload(data, target, panel_values)
    atomic_torch_save(panel_path, payload)
    from torchvision.utils import save_image

    save_image(payload["target"], target_png, nrow=EXPECTED_VISUAL_PER_LABEL, value_range=(0.0, 1.0))
    save_image(payload["variants"]["trained"]["reconstruction"], trained_png, nrow=EXPECTED_VISUAL_PER_LABEL, value_range=(0.0, 1.0))
    return {
        "schema": PANEL_SCHEMA,
        "path": str(panel_path.resolve()),
        "sha256": file_sha256(panel_path),
        "target_png": str(target_png.resolve()),
        "target_png_sha256": file_sha256(target_png),
        "trained_png": str(trained_png.resolve()),
        "trained_png_sha256": file_sha256(trained_png),
        "count": EXPECTED_VISUAL,
        "per_label": EXPECTED_VISUAL_PER_LABEL,
        "variants": sorted(panel_values),
        "affects_training_or_selection": False,
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _finite_metrics(metrics: Mapping[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, list):
            if any(item is not None and (not isinstance(item, (int, float)) or not math.isfinite(item)) for item in value):
                return False
        elif value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            return False
    return True


def _gate_bool(name: str, passed: bool, metrics: Mapping[str, Any]) -> GateResult:
    return GateResult(name, bool(passed), dict(metrics))


def _audit_record(
    state: Mapping[str, Any],
    spec: AuditSpec,
    formal: FormalSystemConfig,
    data: AuditData,
    input_hashes: Mapping[str, str],
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
        input_hashes=dict(input_hashes),
        gates=gates,
        permitted_next_actions=actions,
        invalidation_reason=reason,
    )
    record.validate()
    return record


def _resource_record(
    state: Mapping[str, Any],
    spec: AuditSpec,
    formal: FormalSystemConfig,
    data: AuditData,
    input_hashes: Mapping[str, str],
    reason: str,
) -> EvidenceRecordV2:
    gates = (
        GateResult("no_update_scope", True, {"optimizer": False, "checkpoint": False}),
        GateResult("resource_precondition", False, {"reason": reason}),
    )
    return _audit_record(state, spec, formal, data, input_hashes, gates, "resource_aborted", (), reason)


def run(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("Identity reconstruction attribution audit requires CUDA")
    spec = _load_spec(args.config)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source_training, checkpoint, training_path, checkpoint_path = _load_source(
        spec,
        review_path,
        args.source_training_evidence,
        args.source_checkpoint,
        state["commit"],
    )
    data = _load_data(args.dataset_root)
    if checkpoint.get("dataset_manifest_hash") != data.manifest["full_epoch_manifest_hash"]:
        raise ValueError("source checkpoint full-epoch manifest mismatch")
    _, validation_hash = _load_validation(
        args.validation_evidence,
        spec,
        review_path,
        formal_path,
        training_path,
        checkpoint_path,
        data,
        state,
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reconstruction audit evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    input_hashes = _source_hashes(spec, review_path, formal_path, training_path, checkpoint_path, data)
    input_hashes["validation_evidence"] = validation_hash
    if not torch.cuda.is_available():
        record = _resource_record(state, spec, formal, data, input_hashes, "CUDA is unavailable")
        payload = _record_payload(AUDIT_SCHEMA, record, {"source_training_evidence_hash": SOURCE_TRAINING_HASH, "source_checkpoint_hash": SOURCE_CHECKPOINT_HASH, "validation_evidence_hash": validation_hash, "status_reason": "CUDA is unavailable"})
        _write_new_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        reason = f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}"
        record = _resource_record(state, spec, formal, data, input_hashes, reason)
        payload = _record_payload(AUDIT_SCHEMA, record, {"source_training_evidence_hash": SOURCE_TRAINING_HASH, "source_checkpoint_hash": SOURCE_CHECKPOINT_HASH, "validation_evidence_hash": validation_hash, "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "status_reason": reason})
        _write_new_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    _parameter_contract(model)
    model.eval()
    state_before = _state_cpu(model)
    encoder_phase_before = model.system.encoder.phase.detach().clone()
    decoder_phase_before = model.system.decoder.phase.detach().clone()
    _, target, amplitudes = _render_targets(data)
    heldout_labels = data.labels.index_select(0, data.heldout_indices)
    evaluations: dict[str, Evaluation] = {}
    phase_variants = ("trained", "zero_phase", "encoder_zero", "decoder_zero", "random_phase")
    authoritative_source_mse: float | None = None
    try:
        for variant in phase_variants:
            _restore_phases(model, encoder_phase_before, decoder_phase_before)
            _phase_variant(model, variant)
            evaluations[variant] = _evaluate_model(model, amplitudes, target, heldout_labels, device)
            if variant == "trained":
                authoritative_source_mse = _authoritative_source_replay_mse(model, data, device)
            if time.monotonic() - started > MAX_WALL_SECONDS:
                raise TimeoutError("Identity reconstruction attribution audit exceeded its resource stop")
        _restore_phases(model, encoder_phase_before, decoder_phase_before)
        templates = _template_predictions(data, target)
        evaluations.update({
            name: Evaluation(value.clone(), value.clone(), _metrics(value, target, heldout_labels))
            for name, value in templates.items()
        })
        control_target = target.roll(shifts=1, dims=0)
        cyclic_metrics = _metrics(evaluations["trained"].reconstruction, control_target, heldout_labels)
        evaluations["cyclic_target"] = Evaluation(
            evaluations["trained"].raw_cmos.clone(),
            evaluations["trained"].reconstruction.clone(),
            {"paired_mse": cyclic_metrics["paired_mse"], "centered_pearson_mean": cyclic_metrics["centered_pearson_mean"]},
        )
    finally:
        _restore_phases(model, encoder_phase_before, decoder_phase_before)
        state_unchanged = _state_equal(model, state_before)
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    storage_before_panel = _directory_size(output.parent)
    trained = evaluations["trained"]
    phase_controls = [evaluations[name] for name in ("zero_phase", "encoder_zero", "decoder_zero", "random_phase")]
    template_global = evaluations["global_template"]
    template_class = evaluations["class_templates"]
    trained_geometry = trained.metrics["within_label_distance_correlation"]
    control_geometries = [item.metrics["within_label_distance_correlation"] for item in phase_controls]
    finite = all(_finite_metrics(item.metrics) for item in evaluations.values())
    if authoritative_source_mse is None:
        raise RuntimeError("trained source replay did not run")
    trained_mse = trained.metrics["paired_mse"]
    controls_mse = [item.metrics["paired_mse"] for item in phase_controls]
    trained_pearson = trained.metrics["centered_pearson_mean"]
    control_pearsons = [item.metrics["centered_pearson_mean"] for item in phase_controls]
    trained_retrieval = trained.metrics["exact_target_top1"]
    control_retrieval = [item.metrics["exact_target_top1"] for item in phase_controls]
    template_retrieval = [template_global.metrics["exact_target_top1"], template_class.metrics["exact_target_top1"]]
    geometry_gate = trained_geometry is not None and trained_geometry > 0.0 and all(
        value is None or trained_geometry > value for value in control_geometries
    )
    gates = (
        _source_metric_reproduction_gate(authoritative_source_mse, trained_mse),
        _gate_bool("trained_below_cyclic", trained_mse < evaluations["cyclic_target"].metrics["paired_mse"], {"trained": trained_mse, "cyclic": evaluations["cyclic_target"].metrics["paired_mse"]}),
        _gate_bool("trained_below_global_template", trained_mse < template_global.metrics["paired_mse"], {"trained": trained_mse, "global_template": template_global.metrics["paired_mse"]}),
        _gate_bool("trained_below_class_templates", trained_mse < template_class.metrics["paired_mse"], {"trained": trained_mse, "class_templates": template_class.metrics["paired_mse"]}),
        _gate_bool("trained_below_all_phase_controls", all(trained_mse < value for value in controls_mse), {"trained": trained_mse, "phase_controls": controls_mse}),
        _gate_bool("sample_correlation_positive_and_control_superior", trained_pearson > 0.0 and all(trained_pearson > value for value in control_pearsons), {"trained": trained_pearson, "phase_controls": control_pearsons}),
        _gate_bool("within_label_geometry_positive_and_control_superior", geometry_gate, {"trained": trained_geometry, "phase_controls": control_geometries}),
        _gate_bool("exact_retrieval_above_templates_and_controls", trained_retrieval > max(*template_retrieval, *control_retrieval), {"trained": trained_retrieval, "templates": template_retrieval, "phase_controls": control_retrieval}),
        _gate_bool("finite_values", finite, {"variants": len(evaluations)}),
        _gate_bool("model_state_unchanged", state_unchanged, {}),
        _gate_bool("evaluator_isolation", True, {"evaluator_calls": 0}),
        _gate_bool("fixed_formal_relay_identity_ols", True, {"grid": 2048, "identity": True}),
        _gate_bool("fixed_exposure", True, {"scale": 1.0, "gamma": 1.0}),
        _gate_bool("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
        _gate_bool("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
    )
    panel_values = {name: evaluations[name] for name in phase_variants}
    panel = _save_panel(output.parent, data, target, panel_values)
    storage = _directory_size(output.parent)
    gates = gates + (_gate_bool("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),)
    harness_gates = gates[0:1] + gates[8:]
    scientific_gates = gates[1:8]
    if not all(gate.passed for gate in harness_gates):
        status = "invalid_harness"
        actions: tuple[str, ...] = ()
        reason = "Identity reconstruction attribution harness or resource gate failed"
    elif not all(gate.passed for gate in scientific_gates):
        status = "valid_failed"
        actions = ()
        reason = "pre-registered Identity reconstruction instance/phase gate failed"
    else:
        status = "passed"
        actions = (NEXT_PASS_ACTION,)
        reason = None
    input_hashes["audit_panel"] = panel["sha256"]
    input_hashes["target_panel_png"] = panel["target_png_sha256"]
    input_hashes["trained_panel_png"] = panel["trained_png_sha256"]
    metrics = {name: evaluation.metrics for name, evaluation in evaluations.items()}
    record = _audit_record(state, spec, formal, data, input_hashes, gates, status, actions, reason)
    payload = _record_payload(
        AUDIT_SCHEMA,
        record,
        {
            "source_training_evidence_hash": SOURCE_TRAINING_HASH,
            "source_checkpoint_hash": SOURCE_CHECKPOINT_HASH,
            "validation_evidence_hash": validation_hash,
            "source_checkpoint_step": SOURCE_STEP,
            "mse_reduction": {
                "authoritative_source_replay_mse": authoritative_source_mse,
                "offline_attribution_mse": trained_mse,
                "absolute_difference": abs(authoritative_source_mse - trained_mse),
                "authoritative_provenance": "optical_manifold_training._evaluate: singleton CUDA F.mse_loss -> CUDA stack.mean -> CPU scalar",
                "offline_provenance": "audit._evaluate_model: CPU reconstruction tensors -> CPU singleton F.mse_loss -> CPU mean",
            },
            "metrics": metrics,
            "panel": panel,
            "variants": list(evaluations),
            "evaluator_calls": 0,
            "no_model_update": True,
            "free_cuda_bytes_before": int(free_bytes),
            "total_cuda_bytes": int(total_bytes),
            "peak_cuda_bytes": peak,
            "wall_seconds": elapsed,
            "run_storage_bytes_before_panel": storage_before_panel,
            "run_storage_bytes": storage,
            "status_reason": reason,
        },
    )
    _write_new_json(output, payload)
    torch.use_deterministic_algorithms(deterministic_before)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "passed" else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "reconstruction-audit-validate":
        return validate(args)
    if args.research_action == "reconstruction-audit-run":
        return run(args)
    raise ValueError(f"unsupported optical-manifold reconstruction audit action: {args.research_action}")
