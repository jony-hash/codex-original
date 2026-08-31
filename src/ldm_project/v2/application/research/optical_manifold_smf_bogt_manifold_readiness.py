"""Frozen-weight manifold-readiness audit for the scaled-Chen route.

This module is deliberately a research-only consumer.  It compares the
registered Identity reference with the three frozen scaled-Chen checkpoints,
measures information retention at the optical boundaries, and checks the
same-route BOGT prior without sampling or rendering from that prior.  The
module never creates an optimizer, calls an evaluator, performs a backward
pass, writes a checkpoint, or changes a formal v2 runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.nn import functional as F

from ...config import FormalSystemConfig, load_config
from ...contracts import ComplexField, OperationProvenance
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from . import optical_manifold as base
from . import optical_manifold_reconstruction_audit as reconstruction
from . import optical_manifold_smf_bogt as bogt
from . import optical_manifold_smf_bogt_horizon as horizon


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_manifold_readiness_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_manifold_readiness_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_manifold_readiness_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_manifold_readiness.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_bogt_manifold_readiness_panel.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S0-MRA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
SELECTION_SEED = 1235
PERMUTATION_SEED = 1234
BOOTSTRAP_SEED = 1236
SELECTION_COUNT = 1000
SELECTION_PER_LABEL = 100
CONFIRMATION_COUNT = 5000
PORT_COUNT = 100
RANK_RETENTION_THRESHOLD = 0.90
MATERIAL_RELATIVE_IMPROVEMENT = 0.05
MATERIAL_ABSOLUTE_IMPROVEMENT = 0.05
BOOTSTRAP_REPLICATES = 2000
CONFIDENCE_LEVEL = 0.95
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2
MIN_DISK_FREE_BYTES = 10 * 1024**3
NEXT_ACTION_CANARY = "manual_review_bounded_scaled_chen_prior_to_cmos_generation_canary_contract"
NEXT_ACTION_ENCODER = "manual_review_scaled_chen_encoder_bottleneck_closeout_contract"
NEXT_ACTION_OLS = "manual_review_scaled_chen_ols_transport_bottleneck_closeout_contract"
NEXT_ACTION_DECODER = "manual_review_scaled_chen_decoder_bottleneck_closeout_contract"
NEXT_ACTION_DISTRIBUTED = "manual_review_scaled_chen_distributed_bottleneck_closeout_contract"
NEXT_ACTION_UNDERDETERMINED = "manual_review_scaled_chen_manifold_readiness_attribution_underdetermined_contract"
EXPECTED_PRIOR = {
    "latent": "z_j=abs(c_j)^2/s_j",
    "kind": "truncated_normal",
    "mean": 1.0,
    "variance": 0.25,
    "lower": 0.0,
    "phase": "fixed_calibrated_zero_residual",
}
CHECKPOINT_STEPS = (0, 1000, 5000)
MODEL_NAMES = ("identity_step_5500", "scaled_chen_step_0", "scaled_chen_step_1000", "scaled_chen_step_5000")
STAGE_NAMES = (
    "slm_input",
    "encoder_layer_1",
    "encoder_layer_2",
    "pre_ols",
    "modal_complex",
    "modal_power",
    "post_ols",
    "decoder_input",
    "decoder_layer_1",
    "decoder_layer_2",
    "cmos",
)
SEGMENTS = (
    ("slm_to_pre_ols", "slm_input", "pre_ols"),
    ("pre_ols_to_modal", "pre_ols", "modal_complex"),
    ("modal_to_post_ols", "modal_complex", "post_ols"),
    ("post_ols_to_decoder_input", "post_ols", "decoder_input"),
    ("decoder_input_to_decoder_layer_2", "decoder_input", "decoder_layer_2"),
    ("decoder_layer_2_to_cmos", "decoder_layer_2", "cmos"),
)

# Immutable producer bindings.  The Identity and step-1000 files are restored
# from the named archive before the real run; their hashes are intentionally
# explicit in the checked-in YAML rather than discovered from "latest".
SOURCE_HASHES = {
    "scaled_chen_step_0": "460f9745cbce4c231f75cf3443d23a861c07a5159bb57fa2a97543d153bc57ba",
    "scaled_chen_step_1000": "d18b359e2e54e7f027c47cbc87c78fc9b81714db67cc1c9529a1a0ec3415b850",
    "scaled_chen_step_5000": "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de",
    "horizon_training": "bf707a836dabbea6917aa6c8016e6c00764d9219cd67bb41f551f911f28c2c51",
    "horizon_validation": "ddb882b26aa1463a74d41896f0f8ff8b406614a29bf25a72b918f58094d3e945",
    "horizon_preflight": "e988485f18e1f45964bf44e6338308ec882048038a1ba910f09758521fdfea83",
    "scaled_chen_calibration": "a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959",
    "horizon_v3": "98c55f8c6a2c2063a879e5dbe1eeeec26adc6c49a3f37990d88726097c107e26",
    "horizon_core": "4bacfb93bde8df6bf7831a41feaad50738e530dc518cab6f7922f14dcfe8973d",
    "horizon_variants": "e44fc2dd75e3ed7da4ca9d70074227cf2208e457a10e791e99c11de24b792041",
    "horizon_jacobian": "38d7fae07a85371f98c072426e822d53bf5ba9fda8fbec80e69494e020deaab1",
    "horizon_counterfactual": "c31ab1119835284f293d378e8d13325cb01a9e5ac3f3e6c4fe5038446897a582",
    "identity_checkpoint": "56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f",
    "identity_training": "c01dae121210a4dff0534a9979661f92d1e0fc749087c8451ad51be24a458948",
    "identity_validation": "258af9978653a3552f8786b0fc7d76e8434347ed8c7bd5833f41f0f21f9351cd",
    "identity_preflight": "e14f733bd8a83738892404a71452972eff11a7c6014d174aff1a33703e721a75",
    "plan3_validation": "fe86d7b1990f65d7f2cf51332bc1acca917302160afc1530471106ecdb442bd6",
    "plan3_audit": "31ee69fd050902d0191fbc1b6847ebde040d8f91e0f01d9a6773a7bd8d284905",
    "plan4_observer": "2109ccdbe9711e3d31799486b28dfb370418dc0580583c613182177aa8584281",
    "plan4_pre_ols": "ebf6af4a9efb7a9e8e91127c85566f27caa5f4f26e361dc9a3660e916ae91580",
    "plan4_interpolation": "268ee1b89d5984ec5c2924b1c6a6d6b119f8d02350b6d99cc267da9fa120e50f",
    "smf_background_validation": "4185875318a0a301ab966f99c8ab4eedde517b14c5354864e0520f12fedacb81",
    "smf_background_preflight": "0000000000000000000000000000000000000000000000000000000000000000",
    "smf_background_training": "0000000000000000000000000000000000000000000000000000000000000000",
    "smf_background_audit": "6f9b87adff81069b0b08d2d86e623d5c49565f30b2ca903640aa59b2c9e7b766",
    "fmf_background_validation": "f4af9a87a2f953a4873a11a1f09814e1ab9826dc336638940bb05148a4abacce",
    "fmf_background_preflight": "14abce541b0961ca848678f44d600e48c70e13e22b6cbdbf2ef8c32ba38100ca",
    "fmf_background_training": "f699fccce7da7fb5f1560c5823ee76e2c713ba9e726fb9f6213fe967da36afb6",
    "fmf_background_audit": "f699fccce7da7fb5f1560c5823ee76e2c713ba9e726fb9f6213fe967da36afb6",
}


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class AuditData:
    images: torch.Tensor
    labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor
    selection_indices: torch.Tensor
    confirmation_indices: torch.Tensor
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SourceBundle:
    paths: dict[str, Path]
    hashes: dict[str, str]
    json_payloads: dict[str, dict[str, Any]]
    checkpoints: dict[str, dict[str, Any]]
    calibration: bogt.Calibration | None
    prior: dict[str, Any]


@dataclass(frozen=True)
class ModelEvaluation:
    metrics: dict[str, Any]
    stage_values: dict[str, torch.Tensor]
    sample_mse: torch.Tensor
    labels: torch.Tensor
    state_unchanged: bool
    ols_state_unchanged: bool


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"manifold readiness audit requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {"schema", "hypothesis", "formal_system_config", "parent_prior", "source", "dataset", "optical_contract", "measurement", "gates", "authorization"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("manifold readiness config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("manifold readiness config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    payload = spec.payload
    if payload["parent_prior"] != {
        "config": "configs/research/lightgen_optical_manifold_smf_bogt_v1.yaml",
        "config_sha256": "e3d1b536aa081a8298b21b0b608f232463444a2863e138f941765adf96d8a701",
        **EXPECTED_PRIOR,
        "binding": "scaled_chen_parent_contract_only",
    }:
        raise ValueError("manifold readiness parent prior contract is not locked")
    source = payload["source"]
    required_source_keys = {"paths", "hashes", "producer_commits", "horizon_config_sha256", "identity_config_sha256", "plan3_config_sha256", "plan4_config_sha256"}
    if set(source) != required_source_keys or source["hashes"] != SOURCE_HASHES:
        raise ValueError("manifold readiness source bindings are not locked")
    if source["horizon_config_sha256"] != "304ff57307a7b4b301adf37766d3948c25cab9a83d90da5c85ed6a081003dd18":
        raise ValueError("manifold readiness horizon config binding is stale")
    if source["identity_config_sha256"] != "f59e79090704319704e9e9161587003d369ed18ad4b502facef202d3b2a954a1":
        raise ValueError("manifold readiness Identity config binding is stale")
    if source["plan3_config_sha256"] != "3d7544963c41eb7fa349d8da093760980ed2533b3b0369497f848edc4cb71e07":
        raise ValueError("manifold readiness PLAN-3 config binding is stale")
    if source["plan4_config_sha256"] != "1dc998ed7fcf6493b931cfc5f3927cd6a0e88c6c2a38ad9d178d299e7d62f54b":
        raise ValueError("manifold readiness PLAN-4 config binding is stale")
    if set(source["paths"]) != set(SOURCE_HASHES):
        raise ValueError("manifold readiness source paths are incomplete")
    expected_dataset = {
        "train_count": 55000,
        "validation_count": 5000,
        "selection_count": SELECTION_COUNT,
        "selection_per_label": SELECTION_PER_LABEL,
        "selection_source": "mnist_test_indices_5000_9999",
        "selection_seed": SELECTION_SEED,
        "confirmation_count": CONFIRMATION_COUNT,
        "confirmation_source": "horizon_heldout_train_indices_55000_59999",
        "calibration_excluded": True,
        "schedule_excluded": True,
        "plan3_plan4_panels_excluded": True,
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }
    if payload["dataset"] != expected_dataset:
        raise ValueError("manifold readiness dataset contract is not locked")
    expected_optical = {
        "formal_system_kind": "formal_amplitude_finite_crop_identity",
        "architecture": "2plus2_phase_only",
        "wavelength_m": 5.32e-7,
        "slm_shape": [200, 200],
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "ols_ports": 100,
        "mode_model": "step_index_lp01_ports",
        "mode_waist_m": 7.111e-6,
        "mode_v_number": 2.2,
        "mode_grid_shape": [10, 10],
        "fixed_transport": "identity",
        "fixed_exposure": {"scale": 1.0, "gamma": 1.0},
        "no_complex_resize": True,
        "formal_v2_unchanged": True,
    }
    if payload["optical_contract"] != expected_optical:
        raise ValueError("manifold readiness optical contract is not locked")
    expected_measurement = {
        "objective": "frozen_forward_only",
        "models": list(MODEL_NAMES),
        "scaled_chen_steps": list(CHECKPOINT_STEPS),
        "stages": list(STAGE_NAMES),
        "segments": [list(item) for item in SEGMENTS],
        "selection_metric": "paired_mse",
        "negative_controls": ["permuted_label", "cyclic_target", "class_template", "identity_reference"],
        "jacobian": "reuse_matching_horizon_v3_else_remeasure_all",
        "prior_metrics": ["bogt_kl_surrogate", "truncated_normal_numerical_kl", "complex_effective_rank", "power_effective_rank"],
        "no_update": True,
    }
    if payload["measurement"] != expected_measurement:
        raise ValueError("manifold readiness measurement contract is not locked")
    expected_gates = {
        "rank_retention": RANK_RETENTION_THRESHOLD,
        "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT,
        "material_absolute_improvement": MATERIAL_ABSOLUTE_IMPROVEMENT,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "min_free_cuda_bytes": MIN_FREE_BYTES,
        "max_peak_cuda_bytes": MAX_PEAK_BYTES,
        "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
        "max_storage_bytes": MAX_STORAGE_BYTES,
        "wall_clock": "observational_only",
        "optimizer_created": False,
        "evaluator_calls": 0,
        "formal_promotion": False,
    }
    if payload["gates"] != expected_gates:
        raise ValueError("manifold readiness gate contract is not locked")
    if payload["authorization"] != {
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_scaled_chen_manifold_readiness_audit",
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": ["optimizer", "backward", "checkpoint_write", "generation", "evaluator", "fmf", "prior_sampling", "formal_promotion"],
    }:
        raise ValueError("manifold readiness authorization boundary is not locked")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_scaled_chen_manifold_readiness_audit",
        "approved_contract": {
            "selection": {"count": SELECTION_COUNT, "per_label": SELECTION_PER_LABEL, "source": "mnist_test_indices_5000_9999", "seed": SELECTION_SEED},
            "confirmation": {"count": CONFIRMATION_COUNT, "source": "horizon_heldout_train_indices_55000_59999"},
            "models": list(MODEL_NAMES),
            "stages": list(STAGE_NAMES),
            "rank_retention": RANK_RETENTION_THRESHOLD,
            "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT,
            "no_update": True,
            "generation_canary_not_automatic": True,
            "next_action_if_ready": NEXT_ACTION_CANARY,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("manifold readiness review contract does not match scope")
    return target


def _load_formal(spec: Spec) -> tuple[FormalSystemConfig, Path]:
    path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    config = load_config(path)
    if config.architecture != "conditional_identity_logit_amplitude_v2" or config.system.kind != bogt.EXPECTED_FORMAL_KIND or not config.system.is_formal:
        raise ValueError("manifold readiness requires the current formal Identity geometry")
    formal = config.system
    for name, expected in (("wavelength_m", 532e-9), ("slm_shape", (200, 200)), ("slm_pixel_pitch_m", 2.5e-6), ("metasurface_shape", (100, 100)), ("metasurface_pixel_pitch_m", 2e-6), ("propagation_grid_size", 2048), ("encoder_layers", 2), ("decoder_layers", 2), ("layer_spacing_m", 1.5e-3)):
        actual = getattr(formal, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1e-12, 1e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return formal, path


def _select_indices(test_labels: torch.Tensor) -> torch.Tensor:
    if test_labels.ndim != 1 or test_labels.numel() != 10000:
        raise ValueError("manifold readiness selection requires the complete MNIST test split")
    candidates = torch.arange(5000, 10000, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(SELECTION_SEED)
    selected: list[torch.Tensor] = []
    for label in range(10):
        matches = candidates[test_labels.index_select(0, candidates) == label]
        if matches.numel() < SELECTION_PER_LABEL:
            raise ValueError(f"selection has too few samples for label {label}")
        shuffled = matches.index_select(0, torch.randperm(matches.numel(), generator=generator))
        selected.append(shuffled[:SELECTION_PER_LABEL])
    result = torch.cat(selected).sort().values
    counts = torch.bincount(test_labels.index_select(0, result), minlength=10)
    if result.numel() != SELECTION_COUNT or torch.unique(result).numel() != result.numel() or not torch.equal(counts, torch.full((10,), SELECTION_PER_LABEL, dtype=torch.long)):
        raise ValueError("selection is not deterministic, unique, and class balanced")
    return result


def _load_data(dataset_root: str | Path) -> AuditData:
    train_images, train_labels = base.load_local_mnist(dataset_root, train=True)
    test_images, test_labels = base.load_local_mnist(dataset_root, train=False)
    selection = _select_indices(test_labels)
    confirmation = torch.arange(55000, 60000, dtype=torch.long)
    if train_images.shape != (60000, 1, 28, 28) or train_labels.shape != (60000,) or test_images.shape != (10000, 1, 28, 28) or test_labels.shape != (10000,):
        raise ValueError("MNIST shapes do not match the registered split")
    horizon_data = horizon._load_data(dataset_root)
    if not torch.equal(confirmation, horizon_data.heldout_indices):
        raise ValueError("confirmation panel is not the registered horizon held-out panel")
    # Selection indices belong to the independent MNIST *test* split.  Their
    # integer values must not be compared with train-split indices (the same
    # number can legitimately occur in both datasets).  Isolation from the
    # horizon calibration/schedule is therefore established by split identity;
    # only the confirmation panel shares the train index namespace.
    if torch.isin(selection, confirmation).any():
        raise ValueError("selection overlaps the confirmation panel")
    manifest: dict[str, Any] = {
        "schema": "scaled_chen_manifold_readiness_mnist_manifest.v1",
        "train_images_sha256": tensor_sha256(train_images),
        "train_labels_sha256": tensor_sha256(train_labels),
        "test_images_sha256": tensor_sha256(test_images),
        "test_labels_sha256": tensor_sha256(test_labels),
        "selection_indices_sha256": tensor_sha256(selection),
        "confirmation_indices_sha256": tensor_sha256(confirmation),
        "horizon_manifest_hash": horizon_data.manifest["manifest_hash"],
        "selection_count": SELECTION_COUNT,
        "confirmation_count": CONFIRMATION_COUNT,
        "selected_and_confirmation_disjoint": True,
        "calibration_and_schedule_excluded": True,
        "selection_test_split_disjoint_from_train_reserved": True,
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return AuditData(train_images, train_labels, test_images, test_labels, selection, confirmation, manifest)


def _source_paths(spec: Spec) -> dict[str, Path]:
    # Source paths are repository-relative, including optional zero-hash
    # members that are intentionally absent.  Resolve both existing and absent
    # members against the repository root so evidence never records a
    # misleading configs/research/outputs fallback path.
    root = base._repo_root()
    return {
        name: (Path(value) if Path(value).is_absolute() else root / Path(value))
        for name, value in spec.payload["source"]["paths"].items()
    }


def _validate_source_hashes(spec: Spec) -> tuple[dict[str, Path], dict[str, str]]:
    paths = _source_paths(spec)
    hashes = dict(spec.payload["source"]["hashes"])
    if set(paths) != set(hashes):
        raise ValueError("source path/hash keys do not match")
    for name, path in paths.items():
        expected = hashes[name]
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected.lower()):
            raise ValueError(f"source hash for {name} is malformed")
        if not path.is_file():
            if expected == "0" * 64 and name.endswith(("_preflight", "_training", "_audit")):
                continue
            raise FileNotFoundError(f"registered source artifact is missing: {path}")
        if file_sha256(path) != expected:
            raise ValueError(f"registered source artifact hash mismatch: {name}")
    return paths, hashes


def _json_payload(path: Path, expected_hash: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError(f"historical evidence does not match {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"historical evidence is not a mapping: {path}")
    return payload


def _validate_producer_lineage(name: str, payload: Mapping[str, Any], spec: Spec) -> None:
    """Reject stale producer evidence before any model is constructed."""
    expected_commits = spec.payload["source"]["producer_commits"]
    producer_group = (
        "scaled_chen_horizon" if name.startswith("horizon_") else
        "identity_full_epoch" if name.startswith("identity_") else
        "plan3_reconstruction" if name.startswith("plan3_") else
        "plan4_latent_observer" if name.startswith("plan4_") else None
    )
    evidence = payload.get("evidence")
    if producer_group is None or not isinstance(evidence, Mapping):
        return
    try:
        record = EvidenceRecordV2.from_dict(evidence)
        record.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"producer evidence lineage is invalid: {name}") from exc
    if record.git_commit != expected_commits[producer_group]:
        raise ValueError(f"producer evidence commit is stale: {name}")
    expected_config = {
        "scaled_chen_horizon": spec.payload["source"]["horizon_config_sha256"],
        "identity_full_epoch": spec.payload["source"]["identity_config_sha256"],
        "plan3_reconstruction": spec.payload["source"]["plan3_config_sha256"],
        "plan4_latent_observer": spec.payload["source"]["plan4_config_sha256"],
    }[producer_group]
    if record.config_hash != expected_config:
        raise ValueError(f"producer evidence config is stale: {name}")
    if payload.get("formal_promotion") is not False or payload.get("training_authorized") is not False:
        raise ValueError(f"producer evidence authorization boundary is invalid: {name}")


def _validate_parent_prior(spec: Spec) -> dict[str, Any]:
    parent = _resolve(spec.payload["parent_prior"]["config"], base_path=spec.path.parent)
    if not parent.is_file() or file_sha256(parent) != spec.payload["parent_prior"]["config_sha256"]:
        raise ValueError("scaled-Chen parent prior config is missing, stale, or tampered")
    payload = yaml.safe_load(parent.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != bogt.CONFIG_SCHEMA:
        raise ValueError("scaled-Chen parent prior schema is invalid")
    bogt_contract = payload.get("bogt")
    prior = bogt_contract.get("prior") if isinstance(bogt_contract, Mapping) else None
    if not isinstance(bogt_contract, Mapping) or bogt_contract.get("latent") != EXPECTED_PRIOR["latent"]:
        raise ValueError("scaled-Chen parent latent definition is not registered")
    if not isinstance(prior, Mapping) or {
        "kind": prior.get("kind"), "mean": prior.get("mean"), "variance": prior.get("variance"), "lower": prior.get("lower"),
    } != {key: EXPECTED_PRIOR[key] for key in ("kind", "mean", "variance", "lower")}:
        raise ValueError("scaled-Chen sampling prior is not registered")
    if bogt_contract.get("prior_phase") != EXPECTED_PRIOR["phase"]:
        raise ValueError("scaled-Chen prior phase contract is not registered")
    return dict(spec.payload["parent_prior"])


def _load_source(spec: Spec, *, load_checkpoints: bool) -> SourceBundle:
    paths, hashes = _validate_source_hashes(spec)
    prior = _validate_parent_prior(spec)
    json_payloads: dict[str, dict[str, Any]] = {}
    for name in ("horizon_training", "horizon_validation", "horizon_preflight", "horizon_v3", "horizon_core", "horizon_variants", "horizon_jacobian", "horizon_counterfactual", "identity_training", "identity_validation", "identity_preflight", "plan3_validation", "plan3_audit", "plan4_observer", "smf_background_validation", "smf_background_audit", "fmf_background_validation", "fmf_background_audit"):
        if name in paths and paths[name].is_file():
            json_payloads[name] = _json_payload(paths[name], hashes[name])
            _validate_producer_lineage(name, json_payloads[name], spec)
    required_schemas = {
        "horizon_v3": "scaled_chen_bogt_horizon_no_update_attribution_audit_v3",
        "horizon_core": "scaled_chen_bogt_horizon_no_update_attribution_v1",
        "horizon_variants": "scaled_chen_bogt_horizon_no_update_variants_v1",
        "horizon_jacobian": "scaled_chen_bogt_horizon_no_update_decoder_jacobian_v1",
        "horizon_counterfactual": "scaled_chen_bogt_horizon_no_update_counterfactual_residual_v1",
        "identity_training": "lightgen_optical_manifold_identity_full_epoch_training.v1",
        "identity_validation": "lightgen_optical_manifold_identity_full_epoch_validation.v1",
        "identity_preflight": "lightgen_optical_manifold_identity_full_epoch_preflight.v1",
        "plan3_validation": "lightgen_optical_manifold_identity_reconstruction_audit_validation.v1",
        "plan3_audit": "lightgen_optical_manifold_identity_reconstruction_audit.v1",
        "plan4_observer": "lightgen_optical_manifold_identity_latent_observer.v1",
    }
    for name, schema in required_schemas.items():
        if json_payloads[name].get("schema") != schema:
            raise ValueError(f"historical producer schema mismatch: {name}")
    checkpoints: dict[str, dict[str, Any]] = {}
    if load_checkpoints:
        for name, step in (("scaled_chen_step_0", 0), ("scaled_chen_step_1000", 1000), ("scaled_chen_step_5000", 5000)):
            payload = safe_torch_load(paths[name], expected_sha256=hashes[name], map_location="cpu")
            if not isinstance(payload, Mapping) or payload.get("schema") != "scaled_chen_bogt_horizon_checkpoint.v1" or int(payload.get("global_step", -1)) != step or not isinstance(payload.get("model_state"), Mapping):
                raise ValueError(f"scaled-Chen checkpoint schema or step mismatch: {name}")
            checkpoints[name] = dict(payload)
        identity = safe_torch_load(paths["identity_checkpoint"], expected_sha256=hashes["identity_checkpoint"], map_location="cpu")
        if not isinstance(identity, Mapping) or not isinstance(identity.get("model_state"), Mapping):
            raise ValueError("Identity checkpoint schema is invalid")
        checkpoints["identity_step_5500"] = dict(identity)
    calibration: bogt.Calibration | None = None
    if load_checkpoints:
        calibration_payload = safe_torch_load(paths["scaled_chen_calibration"], expected_sha256=hashes["scaled_chen_calibration"], map_location="cpu")
        if not isinstance(calibration_payload, Mapping) or calibration_payload.get("schema") != "scaled_chen_bogt_calibration.v1":
            raise ValueError("scaled-Chen calibration schema is invalid")
        calibration = bogt.Calibration(calibration_payload["port_scale"], calibration_payload["plane_reference"], int(calibration_payload["sample_count"]), calibration_payload.get("plane_lower"), calibration_payload.get("plane_upper"))
        calibration.validate()
    return SourceBundle(paths, hashes, json_payloads, checkpoints, calibration, prior)


def _input_hashes(spec: Spec, review_path: Path, formal_path: Path, data: AuditData, source: SourceBundle) -> dict[str, str]:
    return {
        "research_config": file_sha256(spec.path),
        "formal_config": file_sha256(formal_path),
        "review_contract": file_sha256(review_path),
        "dataset_manifest": data.manifest["manifest_hash"],
        "selection_indices": data.manifest["selection_indices_sha256"],
        "confirmation_indices": data.manifest["confirmation_indices_sha256"],
        "parent_prior_config": spec.payload["parent_prior"]["config_sha256"],
        **{f"source_{name}": value for name, value in source.hashes.items()},
    }


def _context(spec: Spec, data: AuditData) -> EvidenceContextV2:
    signature = canonical_sha256({"optical_contract": spec.payload["optical_contract"], "measurement": spec.payload["measurement"], "hypothesis": HYPOTHESIS})
    return EvidenceContextV2(signature, spec.config_hash, dataset_hash=data.manifest["manifest_hash"], dataset_split_manifest_hash=data.manifest["manifest_hash"])


def _mode_rank(operator: FiberArrayOLS) -> int:
    area = operator.input_plane.pixel_pitch_m[0] * operator.input_plane.pixel_pitch_m[1]
    gram = operator.input_modes.flatten(1).conj() @ operator.input_modes.flatten(1).T * area
    return int(torch.linalg.matrix_rank(gram).item())


def _build_research_ols(formal: FormalSystemConfig) -> FiberArrayOLS:
    spec = FiberArraySpec(grid_shape=(10, 10), pitch_m=2.0e-6, wavelength_m=532e-9, waist_m=7.111e-6, fill=0.8, v_number=2.2, mode_model="step_index_lp01_ports", expected_fov_m=(200e-6, 200e-6))
    return FiberArrayOLS(base.IdentityOpticalManifoldAutoencoder(formal).system.encoder_plane, spec)


def _state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_equal(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> bool:
    return set(before) == set(after) and all(torch.equal(before[name], after[name]) for name in before)


def _find_trace_field(trace: Sequence[ComplexField] | None, plane_id: str, operation: str) -> ComplexField:
    if trace is None:
        raise ValueError("formal optical trace is missing")
    matches = [field for field in trace if field.plane.plane_id == plane_id and field.provenance.operation == operation]
    if len(matches) != 1:
        raise ValueError(f"expected one trace field for {plane_id}/{operation}, found {len(matches)}")
    return matches[0]


def _feature_projection(value: torch.Tensor, *, complex_value: bool | None = None) -> torch.Tensor:
    """Make a compact deterministic feature matrix without writing full fields."""
    if value.ndim == 2:
        if torch.is_complex(value):
            return torch.cat((value.real, value.imag), dim=1).float()
        return value.float()
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("stage value must be [N,1,H,W] or [N,P]")
    if complex_value is None:
        complex_value = torch.is_complex(value)
    if complex_value:
        real = F.adaptive_avg_pool2d(value.real.float(), (16, 16)).flatten(1)
        imag = F.adaptive_avg_pool2d(value.imag.float(), (16, 16)).flatten(1)
        return torch.cat((real, imag), dim=1)
    return F.adaptive_avg_pool2d(value.float(), (16, 16)).flatten(1)


def _distance_ratio(values: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    flat = values.float()
    gram = flat @ flat.T
    diagonal = gram.diag()
    distances = (diagonal[:, None] + diagonal[None, :] - 2.0 * gram).clamp_min(0.0).sqrt()
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    same = upper & labels[:, None].eq(labels[None, :])
    different = upper & ~labels[:, None].eq(labels[None, :])
    intra = distances[same].mean()
    inter = distances[different].mean()
    return {"intra": float(intra), "inter": float(inter), "inter_intra_ratio": float(inter / intra.clamp_min(1e-12))}


def _permuted_labels(labels: torch.Tensor, *, seed: int = PERMUTATION_SEED) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return labels.index_select(0, torch.randperm(labels.numel(), generator=generator))


def _rank_metrics(values: torch.Tensor, labels: torch.Tensor, *, permuted_labels: torch.Tensor | None = None) -> dict[str, Any]:
    features = _feature_projection(values)
    if features.shape[0] < 2:
        raise ValueError("rank metrics require at least two samples")
    centered = features - features.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum().clamp_min(1e-20)
    probability = energy / total
    entropy_rank = torch.exp(-(probability * probability.clamp_min(1e-20).log()).sum())
    participation_rank = 1.0 / probability.square().sum().clamp_min(1e-20)
    permuted = _permuted_labels(labels) if permuted_labels is None else permuted_labels
    true = _distance_ratio(features, labels)
    perm = _distance_ratio(features, permuted)
    nearest_distance = (features.square().sum(dim=1, keepdim=True) + features.square().sum(dim=1, keepdim=True).T - 2.0 * (features @ features.T)).clamp_min(0.0)
    nearest_distance.fill_diagonal_(float("inf"))
    nearest = nearest_distance.argmin(dim=1)
    retrieval = float((labels.index_select(0, nearest) == labels).float().mean())
    return {
        "entropy_effective_rank": float(entropy_rank),
        "participation_rank": float(participation_rank),
        "true_label_distance": true,
        "permuted_label_distance": perm,
        "instance_retrieval_same_label_top1": retrieval,
        "feature_dim": int(features.shape[1]),
        "finite": bool(torch.isfinite(features).all() and math.isfinite(float(entropy_rank)) and math.isfinite(float(participation_rank))),
    }


def _stage_rank_report(stage_values: Mapping[str, torch.Tensor], labels: torch.Tensor) -> dict[str, Any]:
    permuted = _permuted_labels(labels)
    return {name: _rank_metrics(value, labels, permuted_labels=permuted) for name, value in stage_values.items()}


def _retention_report(stage_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, left, right in SEGMENTS:
        left_metric = stage_metrics[left]
        right_metric = stage_metrics[right]
        entropy = float(right_metric["entropy_effective_rank"]) / max(float(left_metric["entropy_effective_rank"]), 1e-12)
        participation = float(right_metric["participation_rank"]) / max(float(left_metric["participation_rank"]), 1e-12)
        result[name] = {
            "entropy_retention": entropy,
            "participation_retention": participation,
            "geometric_mean_retention": math.sqrt(max(entropy, 0.0) * max(participation, 0.0)),
        }
    return result


def _mse_metrics(camera: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
    paired = (camera - target).square().mean(dim=(1, 2, 3))
    cyclic_target = torch.roll(target, shifts=1, dims=0)
    cyclic = (camera - cyclic_target).square().mean(dim=(1, 2, 3))
    templates = []
    for label in range(10):
        mask = labels == label
        if not mask.any():
            raise ValueError(f"panel is missing label {label}")
        templates.append(target[mask].mean(dim=0, keepdim=True))
    template = torch.cat([templates[int(label)] for label in labels], dim=0)
    class_template = (camera - template).square().mean(dim=(1, 2, 3))
    # Retrieval and within-class geometry use the same compact representation
    # as the rank audit.  Keeping the full 200x200 fields only for the paired
    # MSE avoids a quadratic CPU allocation over the 5,000-sample confirmation
    # panel while preserving the established metric definitions.
    compact_camera = _feature_projection(camera)
    compact_target = _feature_projection(target)
    retrieval = reconstruction._retrieval_metrics(compact_camera, compact_target, labels)
    geometry, geometry_by_label = reconstruction._pairwise_distance_correlation(compact_camera, compact_target, labels)
    compact_templates = torch.stack([compact_target[labels == label].mean(dim=0) for label in range(10)])
    template_distances = (
        compact_camera.square().sum(dim=1, keepdim=True)
        + compact_templates.square().sum(dim=1).unsqueeze(0)
        - 2.0 * (compact_camera @ compact_templates.T)
    ).clamp_min(0.0)
    template_nearest = template_distances.argmin(dim=1)
    class_template_retrieval = float((template_nearest == labels).float().mean())
    cyclic_compact = compact_target.roll(shifts=1, dims=0)
    cyclic_distances = (
        compact_camera.square().sum(dim=1, keepdim=True)
        + cyclic_compact.square().sum(dim=1).unsqueeze(0)
        - 2.0 * (compact_camera @ cyclic_compact.T)
    ).clamp_min(0.0)
    cyclic_nearest = cyclic_distances.argmin(dim=1)
    cyclic_retrieval = float((labels.roll(shifts=1).index_select(0, cyclic_nearest) == labels).float().mean())
    result = {
        "paired_mse": float(paired.mean()),
        "cyclic_mse": float(cyclic.mean()),
        "class_template_mse": float(class_template.mean()),
        "paired_mse_by_label": [float(paired[labels == label].mean()) for label in range(10)],
        "within_label_distance_correlation": geometry,
        "within_label_distance_correlation_by_label": geometry_by_label,
        "class_template_same_label_top1": class_template_retrieval,
        "cyclic_same_label_top1": cyclic_retrieval,
        **retrieval,
        "finite": bool(torch.isfinite(paired).all() and torch.isfinite(cyclic).all() and torch.isfinite(class_template).all()),
    }
    return result, paired


def _load_identity_model(formal: FormalSystemConfig, payload: Mapping[str, Any], device: torch.device) -> base.IdentityOpticalManifoldAutoencoder:
    model = base.IdentityOpticalManifoldAutoencoder(formal)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_scaled_model(formal: FormalSystemConfig, payload: Mapping[str, Any], device: torch.device) -> bogt.ScaledChenBOGTModel:
    model = bogt.ScaledChenBOGTModel(formal)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()) or any(parameter.requires_grad for parameter in model.ols.parameters()):
        raise ValueError("scaled-Chen audit model is not frozen")
    return model


def _identity_trace_stages(model: base.IdentityOpticalManifoldAutoencoder, amplitude: torch.Tensor, research_ols: FiberArrayOLS) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    result = model.system(model.system.command(amplitude), exposure=model.exposure, return_optical_trace=True)
    trace = result.optical_trace
    if trace is None:
        raise ValueError("Identity trace is missing")
    encoder_1 = _find_trace_field(trace, "encoder_layer_1", "finite_crop_hard_aperture").tensor
    encoder_2 = _find_trace_field(trace, "encoder_layer_2", "finite_crop_hard_aperture").tensor
    pre = encoder_2
    ols = research_ols(_find_trace_field(trace, "encoder_layer_2", "finite_crop_hard_aperture"))
    post = ols.tensor
    decoder_1 = _find_trace_field(trace, "decoder_layer_1", "finite_crop_hard_aperture").tensor
    decoder_2 = _find_trace_field(trace, "decoder_layer_2", "finite_crop_hard_aperture").tensor
    camera = result.raw_camera.tensor
    coefficients = torch.einsum("mhw,bhw->bm", research_ols.input_modes.conj().to(pre), pre[:, 0]) * (2.0e-6**2)
    stages = {
        "slm_input": amplitude,
        "encoder_layer_1": encoder_1,
        "encoder_layer_2": encoder_2,
        "pre_ols": pre,
        "modal_complex": coefficients,
        "modal_power": coefficients.abs().square(),
        "post_ols": post,
        "decoder_input": post,
        "decoder_layer_1": decoder_1,
        "decoder_layer_2": decoder_2,
        "cmos": camera,
    }
    return stages, camera


def _scaled_trace_stages(model: bogt.ScaledChenBOGTModel, amplitude: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    output = model(amplitude, return_trace=True)
    if len(output.encoder_records) != 2 or len(output.decoder_records) != 2:
        raise ValueError("scaled-Chen trace does not contain the registered 2+2 records")
    pre = output.encoder_records[-1].tensor
    post = output.ols_field.tensor
    stages = {
        "slm_input": amplitude,
        "encoder_layer_1": output.encoder_records[0].tensor,
        "encoder_layer_2": output.encoder_records[1].tensor,
        "pre_ols": pre,
        "modal_complex": output.coefficients,
        "modal_power": output.coefficients.abs().square(),
        "post_ols": post,
        "decoder_input": post,
        "decoder_layer_1": output.decoder_records[0].tensor,
        "decoder_layer_2": output.decoder_records[1].tensor,
        "cmos": output.camera,
    }
    return stages, output.camera


def _decoder_camera_from_coefficients(model_name: str, model: torch.nn.Module, research_ols: FiberArrayOLS, coefficients: torch.Tensor) -> torch.Tensor:
    """Re-emit fixed modal coefficients through the decoder/CMOS path only."""
    if model_name != "identity_step_5500":
        return model.reemit_decoder(coefficients)
    modes = research_ols.output_modes.to(coefficients)
    values = torch.einsum("bm,mhw->bhw", coefficients, modes).unsqueeze(1)
    field = ComplexField(
        values,
        model.system.encoder_plane.with_plane("ols_output"),
        OperationProvenance("fixed_ols_jacobian_probe", "ols_output", 0.0, {"transport": "identity"}),
    )
    decoded, _, _ = model.system.decoder(field, "decoder")
    chip = ComplexField(decoded.tensor, model.system.chip_plane, decoded.provenance)
    return model.system.back(chip).tensor.abs().square()


def _decoder_jacobian_report(model_name: str, model: torch.nn.Module, research_ols: FiberArrayOLS, coefficients: torch.Tensor) -> dict[str, Any]:
    """Finite-difference decoder controllability on one fixed panel probe.

    The old horizon Jacobian is not reused when its panel lineage differs.  A
    common 100-amplitude + 100-phase probe is therefore measured for every
    checkpoint, including Identity and step 0, under no-grad and without any
    optimizer/backward path.
    """
    if coefficients.ndim != 2 or coefficients.shape[0] != 1 or coefficients.shape[1] != PORT_COUNT or not torch.is_complex(coefficients):
        raise ValueError("decoder Jacobian probe requires one finite complex 100-port coefficient vector")
    base_output = _feature_projection(_decoder_camera_from_coefficients(model_name, model, research_ols, coefficients))
    amplitude_step = max(float(coefficients.abs().mean().detach().cpu()) * 1.0e-3, 1.0e-5)
    phase_step = 1.0e-3
    columns: list[torch.Tensor] = []
    with torch.no_grad():
        for kind, step in (("amplitude", amplitude_step), ("phase", phase_step)):
            for start in range(0, PORT_COUNT, 10):
                count = min(10, PORT_COUNT - start)
                perturbed = coefficients.expand(count, -1).clone()
                for offset in range(count):
                    port = start + offset
                    if kind == "amplitude":
                        perturbed[offset, port] = perturbed[offset, port] + step
                    else:
                        perturbed[offset, port] = perturbed[offset, port] * torch.exp(torch.tensor(1j * step, device=perturbed.device, dtype=perturbed.dtype))
                output = _feature_projection(_decoder_camera_from_coefficients(model_name, model, research_ols, perturbed))
                columns.append(((output - base_output) / step).T.detach().cpu())
    matrix = torch.cat(columns, dim=1)
    rank = _jacobian_effective_rank(matrix)
    return {
        "effective_rank": rank,
        "input_dimension": int(matrix.shape[1]),
        "output_dimension": int(matrix.shape[0]),
        "finite": bool(torch.isfinite(matrix).all()),
        "threshold": 1.0e-3,
        "probe_count": 1,
        "coordinate_definition": "100 amplitude + 100 phase; global phase not separately projected",
        "source": "fresh_panel_probe_due_to_horizon_lineage_mismatch",
    }


def _evaluate_model(model_name: str, model: torch.nn.Module, research_ols: FiberArrayOLS, images: torch.Tensor, labels: torch.Tensor, device: torch.device, calibration: bogt.Calibration | None) -> ModelEvaluation:
    stage_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in STAGE_NAMES}
    camera_chunks: list[torch.Tensor] = []
    before = _state_cpu(model)
    before_ols = _state_cpu(research_ols)
    model.eval()
    with torch.no_grad():
        for indices in torch.arange(images.shape[0], dtype=torch.long).split(10):
            rendered = base.render_intensity_batch(images.index_select(0, indices))
            amplitude = rendered.amplitude.to(device)
            if model_name == "identity_step_5500":
                stages, camera = _identity_trace_stages(model, amplitude, research_ols)
            else:
                stages, camera = _scaled_trace_stages(model, amplitude)
            camera_cpu = camera.detach().cpu().float()
            camera_chunks.append(camera_cpu)
            for name, value in stages.items():
                # Stage rank/geometry never needs the full field tensor.  Pool
                # each batch before moving it to CPU so the confirmation panel
                # remains bounded and the audit cannot accidentally retain
                # gigabytes of optical fields.  Modal coefficients/powers are
                # already compact port vectors and stay raw for the OLS/prior
                # calculations below.
                detached = value.detach().cpu()
                stage_chunks[name].append(
                    detached if name in {"modal_complex", "modal_power"} else _feature_projection(detached)
                )
    after = _state_cpu(model)
    stage_values = {name: torch.cat(values, dim=0) for name, values in stage_chunks.items()}
    camera = torch.cat(camera_chunks, dim=0)
    target = base.render_intensity_batch(images).target.float()
    mse, sample_mse = _mse_metrics(camera, target, labels)
    stage_metrics = _stage_rank_report(stage_values, labels)
    metrics: dict[str, Any] = {
        "model": model_name,
        "reconstruction": mse,
        "stages": stage_metrics,
        "retention": _retention_report(stage_metrics),
        "state_unchanged": _state_equal(before, after),
        "evaluation_count": int(images.shape[0]),
    }
    coefficients = stage_values["modal_complex"]
    powers = stage_values["modal_power"]
    if torch.is_complex(coefficients) and calibration is not None:
        z = bogt.normalize_port_power(powers, calibration.port_scale)
        mean, variance = bogt.aggregate_moments(z)
        metrics["ols"] = {
            "modal_power_retention": float((powers.sum(dim=1) / calibration.port_scale.sum()).mean()),
            "projection_residual": float((stage_values["pre_ols"] - stage_values["post_ols"]).square().mean()),
            "active_port_count": int((powers.mean(dim=0) > 1.0e-3 * powers.mean()).sum()),
            "modal_entropy_effective_rank": float(stage_metrics["modal_power"]["entropy_effective_rank"]),
            "modal_participation_rank": float(stage_metrics["modal_power"]["participation_rank"]),
            "bogt_kl_surrogate": float(bogt.bogt_kl_surrogate(mean, variance)),
            "truncated_normal_numerical_kl": float(bogt.truncated_normal_kl_reference(mean, variance)),
            "complex_effective_rank": float(stage_metrics["modal_complex"]["participation_rank"]),
            "power_effective_rank": float(stage_metrics["modal_power"]["participation_rank"]),
            "power_moments": {"mean": float(mean.mean()), "variance": float(variance.mean())},
            "power_quantiles": torch.quantile(z, torch.tensor([0.01, 0.5, 0.99])).tolist(),
        }
        pre_rank = stage_metrics["pre_ols"]
        post_rank = stage_metrics["post_ols"]
        metrics["ols"]["pre_to_post_rank_retention"] = {
            "entropy": float(post_rank["entropy_effective_rank"] / max(pre_rank["entropy_effective_rank"], 1.0e-12)),
            "participation": float(post_rank["participation_rank"] / max(pre_rank["participation_rank"], 1.0e-12)),
        }
    metrics["decoder_jacobian"] = _decoder_jacobian_report(model_name, model, research_ols, coefficients[:1].to(device))
    after_ols = _state_cpu(research_ols)
    metrics["ols_state_unchanged"] = _state_equal(before_ols, after_ols)
    return ModelEvaluation(metrics, stage_values, sample_mse, labels.clone(), bool(metrics["state_unchanged"]), bool(metrics["ols_state_unchanged"]))


def _stratified_bootstrap_ci(delta: torch.Tensor, labels: torch.Tensor, *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED, confidence_level: float = CONFIDENCE_LEVEL) -> tuple[float, float]:
    if delta.ndim != 1 or labels.shape != delta.shape or delta.numel() == 0 or not torch.isfinite(delta).all():
        raise ValueError("bootstrap inputs must be finite vectors with equal shape")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = []
    for _ in range(replicates):
        class_means = []
        for label in range(10):
            class_delta = delta[labels == label]
            if class_delta.numel() == 0:
                raise ValueError(f"bootstrap class {label} is empty")
            draws = torch.randint(class_delta.numel(), (class_delta.numel(),), generator=generator)
            class_means.append(class_delta.index_select(0, draws).mean())
        values.append(torch.stack(class_means).mean())
    samples = torch.stack(values)
    alpha = (1.0 - confidence_level) / 2.0
    return float(torch.quantile(samples, alpha)), float(torch.quantile(samples, 1.0 - alpha))


def _jacobian_effective_rank(matrix: torch.Tensor, threshold: float = 1.0e-3) -> int:
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError("Jacobian matrix must be finite and two-dimensional")
    singular = torch.linalg.svdvals(matrix)
    if singular.numel() == 0 or float(singular.max()) <= 0.0:
        return 0
    return int((singular >= singular.max() * threshold).sum())


def _jacobian_lineage_compatible(payload: Mapping[str, Any], *, expected_hash: str, expected_steps: Sequence[int] = CHECKPOINT_STEPS) -> bool:
    return payload.get("schema") == "scaled_chen_bogt_horizon_no_update_decoder_jacobian_v1" and payload.get("source_hash") in {expected_hash, None} and all(str(step) in payload for step in expected_steps if step != 0)


def _classify_attribution(selection: Mapping[str, Any], confirmation: Mapping[str, Any], *, controls: Mapping[str, Any] | None = None) -> str:
    controls = controls or {}
    stable = True
    stable_segments: dict[str, bool] = {}
    for name, *_ in SEGMENTS:
        left = selection.get("retention", {}).get(name, {}).get("geometric_mean_retention")
        right = confirmation.get("retention", {}).get(name, {}).get("geometric_mean_retention")
        stable_segments[name] = isinstance(left, (int, float)) and isinstance(right, (int, float)) and float(left) < RANK_RETENTION_THRESHOLD and float(right) < RANK_RETENTION_THRESHOLD
    stable = all(isinstance(selection.get("retention", {}).get(name), Mapping) and isinstance(confirmation.get("retention", {}).get(name), Mapping) for name, *_ in SEGMENTS)
    if not stable:
        return "attribution_underdetermined"
    failed = [name for name, value in stable_segments.items() if value]
    if len(failed) >= 2:
        return "distributed_bottleneck_supported"
    if failed == ["slm_to_pre_ols"] and controls.get("encoder_causal_improvement", 0.0) >= MATERIAL_RELATIVE_IMPROVEMENT:
        return "encoder_bottleneck_supported"
    if failed and failed[0] in {"pre_ols_to_modal", "modal_to_post_ols"} and controls.get("ols_causal_improvement", 0.0) >= MATERIAL_RELATIVE_IMPROVEMENT:
        return "ols_transport_bottleneck_supported"
    if failed and failed[0] in {"decoder_input_to_decoder_layer_2", "decoder_layer_2_to_cmos"} and controls.get("decoder_jacobian_rank", 100) < 90:
        return "decoder_bottleneck_supported"
    if failed:
        return "distributed_bottleneck_supported"
    return "attribution_underdetermined"


def _classify_prior_accessibility(step0: Mapping[str, Any], step1000: Mapping[str, Any], step5000: Mapping[str, Any], reconstruction_supported: bool) -> tuple[str, dict[str, Any]]:
    def metric(item: Mapping[str, Any], key: str) -> float:
        value = item.get("ols", {}).get(key)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else float("nan")
    checks = {
        "gaussian_improved": metric(step5000, "bogt_kl_surrogate") <= 0.95 * metric(step0, "bogt_kl_surrogate"),
        "numerical_improved": metric(step5000, "truncated_normal_numerical_kl") <= 0.95 * metric(step0, "truncated_normal_numerical_kl"),
        "not_worse_than_step1000_gaussian": metric(step5000, "bogt_kl_surrogate") <= metric(step1000, "bogt_kl_surrogate"),
        "not_worse_than_step1000_numerical": metric(step5000, "truncated_normal_numerical_kl") <= metric(step1000, "truncated_normal_numerical_kl"),
        "complex_rank_retained": metric(step5000, "complex_effective_rank") >= RANK_RETENTION_THRESHOLD * max(metric(step0, "complex_effective_rank"), 1e-12),
        "power_rank_retained": metric(step5000, "power_effective_rank") >= RANK_RETENTION_THRESHOLD * max(metric(step0, "power_effective_rank"), 1e-12),
        "reconstruction_supported": reconstruction_supported,
    }
    if any(not isinstance(value, bool) or not value for value in checks.values()):
        return "generation_prior_not_accessible", checks
    return "prior_accessible", checks


def _reconstruction_supported(evaluation: Mapping[str, Any]) -> bool:
    reconstruction_metrics = evaluation.get("reconstruction", {})
    return bool(
        reconstruction_metrics.get("paired_mse", float("inf")) < reconstruction_metrics.get("cyclic_mse", float("-inf"))
        and reconstruction_metrics.get("paired_mse", float("inf")) < reconstruction_metrics.get("class_template_mse", float("-inf"))
        and reconstruction_metrics.get("finite") is True
        and reconstruction_metrics.get("same_label_top1", 0.0) >= reconstruction_metrics.get("class_template_same_label_top1", -1.0)
    )


def _validation_gates(formal: FormalSystemConfig, data: AuditData) -> tuple[GateResult, ...]:
    operator = _build_research_ols(formal)
    rank = _mode_rank(operator)
    finite = bool(torch.isfinite(operator.input_modes).all() and torch.isfinite(operator.output_modes).all())
    return (
        GateResult("selection_balanced", bool(torch.equal(torch.bincount(data.test_labels.index_select(0, data.selection_indices), minlength=10), torch.full((10,), SELECTION_PER_LABEL, dtype=torch.long))), {}),
        GateResult("selection_confirmation_disjoint", not bool(torch.isin(data.selection_indices, data.confirmation_indices).any()), {}),
        GateResult("mode_bank_rank_100", rank == PORT_COUNT, {"rank": rank}),
        GateResult("finite_mode_bank", finite, {}),
        GateResult("formal_identity_boundary", formal.kind == bogt.EXPECTED_FORMAL_KIND and formal.is_formal, {}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "evaluator_calls": 0}),
    )


def _record(schema: str, status: str, classification: str | None, spec: Spec, state: Mapping[str, Any], data: AuditData, hashes: Mapping[str, str], gates: Sequence[GateResult], actions: Sequence[str], contract: Mapping[str, Any], reason: str | None = None) -> dict[str, Any]:
    record_status = status
    record = EvidenceRecordV2(2, schema, "diagnostic", record_status, str(state["commit"]), None, spec.config_hash, _context(spec, data), dict(hashes), tuple(gates), tuple(actions), reason)
    record.validate()
    return {
        "schema": schema,
        "hypothesis": HYPOTHESIS,
        "status": status,
        "classification": classification,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_write_new_json(path, payload, artifact_kind="manifold readiness artifact")


def _load_validation(path: str | Path, spec: Spec, review_path: Path, formal_path: Path, data: AuditData, source: SourceBundle, state: Mapping[str, Any]) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, formal_path, data, source)
    if payload.get("schema") != VALIDATION_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or payload.get("status") != "passed" or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.input_hashes != expected_hashes or record.permitted_next_actions != ("run_scaled_chen_manifold_readiness",):
        raise ValueError("manifold readiness validation evidence is stale or mismatched")


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    # Hash and schema validation happen before any checkpoint load or CUDA
    # context.  ``load_checkpoints=False`` is intentional and tested.
    source = _load_source(spec, load_checkpoints=False)
    gates = _validation_gates(formal, data)
    hashes = _input_hashes(spec, review_path, formal_path, data, source)
    valid = all(gate.passed for gate in gates)
    payload = _record(VALIDATION_SCHEMA, "passed" if valid else "invalid_harness", None, spec, state, data, hashes, gates, ("run_scaled_chen_manifold_readiness",) if valid else (), {"formal_config": str(formal_path), "review_contract": str(review_path), "dataset_manifest": data.manifest, "source_paths": {name: str(path) for name, path in source.paths.items()}, "prior": source.prior})
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("manifold readiness audit requires CUDA; CPU is not a formal fallback")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError("manifold readiness audit requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("manifold readiness audit requires 10 GiB free storage")
    return int(free), int(total)


def _run_one_panel(model_name: str, formal: FormalSystemConfig, checkpoint: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, source: SourceBundle, device: torch.device) -> ModelEvaluation:
    research_ols = _build_research_ols(formal).to(device)
    if model_name == "identity_step_5500":
        model = _load_identity_model(formal, checkpoint, device)
    else:
        model = _load_scaled_model(formal, checkpoint, device)
    return _evaluate_model(model_name, model, research_ols, images, labels, device, source.calibration)


def _cross_swap_panel(formal: FormalSystemConfig, source: SourceBundle, evaluations: Mapping[str, ModelEvaluation], images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Measure current-lineage encoder/decoder cross-swaps on one fixed panel."""
    target = base.render_intensity_batch(images).target.float()
    records: dict[str, Any] = {}
    for decoder_step, decoder_name in ((1000, "scaled_chen_step_1000"), (5000, "scaled_chen_step_5000")):
        research_ols = _build_research_ols(formal).to(device)
        model = _load_scaled_model(formal, source.checkpoints[decoder_name], device)
        for encoder_step, encoder_name in ((1000, "scaled_chen_step_1000"), (5000, "scaled_chen_step_5000")):
            if encoder_step == decoder_step:
                continue
            coefficients = evaluations[encoder_name].stage_values["modal_complex"].to(device)
            camera_chunks: list[torch.Tensor] = []
            with torch.no_grad():
                for indices in torch.arange(coefficients.shape[0], dtype=torch.long).split(10):
                    camera_chunks.append(model.reemit_decoder(coefficients.index_select(0, indices.to(coefficients.device))).detach().cpu().float())
            camera = torch.cat(camera_chunks, dim=0)
            metrics, _ = _mse_metrics(camera, target, labels)
            records[f"encoder_{encoder_step}_decoder_{decoder_step}"] = {
                "paired_mse": metrics["paired_mse"],
                "cyclic_mse": metrics["cyclic_mse"],
                "class_template_mse": metrics["class_template_mse"],
                "same_label_top1": metrics["same_label_top1"],
                "finite": metrics["finite"],
                "evaluation_count": int(images.shape[0]),
            }
        del model, research_ols
        torch.cuda.empty_cache()
    return records


def _historical_control_lineage(source: SourceBundle, data: AuditData) -> dict[str, Any]:
    payload = source.json_payloads.get("horizon_v3", {})
    lineage = payload.get("lineage", {}) if isinstance(payload, Mapping) else {}
    source_manifest = lineage.get("dataset_manifest_hash") if isinstance(lineage, Mapping) else None
    current_manifest = data.manifest["horizon_manifest_hash"]
    compatible = source_manifest == current_manifest
    return {
        "historical_horizon_v3_dataset_manifest": source_manifest,
        "current_horizon_dataset_manifest": current_manifest,
        "reuse_compatible": compatible,
        "cross_swap_reuse": False,
        "bypass_reuse": False,
        "reason": None if compatible else "historical_probe_lineage_mismatch; current cross-swap was remeasured",
    }


def _compact_summary(evaluations: Mapping[str, Mapping[str, Any]], *, attribution: str, readiness: str, prior_status: str) -> dict[str, Any]:
    return {
        "schema": "scaled_chen_manifold_readiness_compact_report.v1",
        "attribution": attribution,
        "generation_readiness": readiness,
        "prior_status": prior_status,
        "models": {name: {"paired_mse": value.get("reconstruction", {}).get("paired_mse"), "retention": value.get("retention", {})} for name, value in evaluations.items()},
        "generative_manifold_supported": False,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "evaluator_calls": 0,
    }


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("manifold readiness audit requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    # Validate source hashes and all producer JSON before creating CUDA models.
    source = _load_source(spec, load_checkpoints=False)
    _load_validation(args.validation_evidence, spec, review_path, formal_path, data, source, state)
    output_path = Path(args.output)
    compact_path = output_path.with_name("compact_report.json")
    panel_path = output_path.with_name("panel.pt")
    for target in (output_path, compact_path, panel_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite manifold readiness artifact: {target}")
    device = torch.device(args.device)
    free_bytes, total_bytes = _resource_preflight(output_path, device)
    # Checkpoint loading happens only after the fail-closed validation and CUDA
    # resource gates above.
    source = _load_source(spec, load_checkpoints=True)
    started = time.monotonic()
    torch.manual_seed(PERMUTATION_SEED)
    torch.cuda.manual_seed_all(PERMUTATION_SEED)
    torch.cuda.reset_peak_memory_stats(device)
    selection_images = data.test_images.index_select(0, data.selection_indices)
    selection_labels = data.test_labels.index_select(0, data.selection_indices)
    confirmation_images = data.images.index_select(0, data.confirmation_indices)
    confirmation_labels = data.labels.index_select(0, data.confirmation_indices)
    # The Identity reference is measured on the same panels but never enters
    # Scaled-Chen checkpoint selection.  Only compact metrics and per-sample
    # errors are retained; full optical fields are released after each model.
    selection_records: dict[str, Any] = {}
    confirmation_records: dict[str, Any] = {}
    selection_eval: dict[str, ModelEvaluation] = {}
    confirmation_eval: dict[str, ModelEvaluation] = {}
    for name in MODEL_NAMES:
        selection_eval[name] = _run_one_panel(name, formal, source.checkpoints[name], selection_images, selection_labels, source, device)
        selection_records[name] = selection_eval[name].metrics
        confirmation_eval[name] = _run_one_panel(name, formal, source.checkpoints[name], confirmation_images, confirmation_labels, source, device)
        confirmation_records[name] = confirmation_eval[name].metrics
        torch.cuda.empty_cache()
    cross_swaps = {
        "selection": _cross_swap_panel(formal, source, selection_eval, selection_images, selection_labels, device),
        "confirmation": _cross_swap_panel(formal, source, confirmation_eval, confirmation_images, confirmation_labels, device),
    }
    control_lineage = _historical_control_lineage(source, data)
    # The first loop is intentionally guarded above only for the resource
    # preflight; delete its empty dictionaries and retain the one compact pass.
    best_step5000 = confirmation_records["scaled_chen_step_5000"]["reconstruction"]["paired_mse"]
    identity_ref = confirmation_records["identity_step_5500"]["reconstruction"]["paired_mse"]
    baseline = confirmation_records["scaled_chen_step_0"]["reconstruction"]["paired_mse"]
    selection_delta = selection_eval["scaled_chen_step_0"].sample_mse - selection_eval["scaled_chen_step_5000"].sample_mse
    confirmation_delta = confirmation_eval["scaled_chen_step_0"].sample_mse - confirmation_eval["scaled_chen_step_5000"].sample_mse
    selection_ci = _stratified_bootstrap_ci(selection_delta, selection_labels)
    confirmation_ci = _stratified_bootstrap_ci(confirmation_delta, confirmation_labels)
    bootstrap = {
        "selection": {
            "delta_definition": "step_0_paired_mse - step_5000_paired_mse",
            "mean_delta": float(selection_delta.mean()),
            "relative_improvement": float(selection_delta.mean() / max(float(selection_eval["scaled_chen_step_0"].metrics["reconstruction"]["paired_mse"]), 1.0e-12)),
            "confidence_level": CONFIDENCE_LEVEL,
            "ci": list(selection_ci),
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "confirmation": {
            "delta_definition": "step_0_paired_mse - step_5000_paired_mse",
            "mean_delta": float(confirmation_delta.mean()),
            "relative_improvement": float(confirmation_delta.mean() / max(baseline, 1.0e-12)),
            "confidence_level": CONFIDENCE_LEVEL,
            "ci": list(confirmation_ci),
            "replicates": BOOTSTRAP_REPLICATES,
        },
    }
    reconstruction_supported = all(_reconstruction_supported(confirmation_records[name]) for name in ("scaled_chen_step_0", "scaled_chen_step_1000", "scaled_chen_step_5000"))
    attribution = _classify_attribution(selection_records["scaled_chen_step_5000"], confirmation_records["scaled_chen_step_5000"], controls={"decoder_jacobian_rank": 100})
    prior_status, prior_checks = _classify_prior_accessibility(confirmation_records["scaled_chen_step_0"], confirmation_records["scaled_chen_step_1000"], confirmation_records["scaled_chen_step_5000"], reconstruction_supported)
    if attribution == "attribution_underdetermined":
        readiness = "underdetermined"
        classification = "attribution_underdetermined"
        action = NEXT_ACTION_UNDERDETERMINED
    elif prior_status == "prior_accessible" and reconstruction_supported:
        readiness = "canary_ready"
        classification = attribution
        action = NEXT_ACTION_CANARY
    else:
        readiness = "not_ready"
        classification = attribution
        action = {"encoder_bottleneck_supported": NEXT_ACTION_ENCODER, "ols_transport_bottleneck_supported": NEXT_ACTION_OLS, "decoder_bottleneck_supported": NEXT_ACTION_DECODER}.get(attribution, NEXT_ACTION_DISTRIBUTED)
    state_unchanged = all(item.state_unchanged and item.ols_state_unchanged for item in (*selection_eval.values(), *confirmation_eval.values()))
    finite = all(item.metrics.get("reconstruction", {}).get("finite") is True and item.metrics.get("decoder_jacobian", {}).get("finite") is True for item in (*selection_eval.values(), *confirmation_eval.values()))
    peak = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.monotonic() - started
    harness_gates = (
        GateResult("finite_measurements", finite, {}),
        GateResult("frozen_state_unchanged", state_unchanged, {}),
        GateResult("negative_controls_present", all("cyclic_mse" in item["reconstruction"] and "class_template_mse" in item["reconstruction"] for item in confirmation_records.values()), {}),
        GateResult("optimizer_isolation", True, {"optimizer_created": False}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak, "maximum": MAX_PEAK_BYTES}),
    )
    science_gates = (
        GateResult("reconstruction_manifold_supported", reconstruction_supported, {}),
        GateResult(
            "heldout_material_improvement",
            all(item["relative_improvement"] >= MATERIAL_RELATIVE_IMPROVEMENT and item["ci"][0] > 0.0 for item in bootstrap.values()),
            bootstrap,
        ),
        GateResult("prior_accessibility", prior_status == "prior_accessible", prior_checks),
    )
    status = "valid_failed" if all(gate.passed for gate in harness_gates) else "invalid_harness"
    all_gates = harness_gates + science_gates
    if status == "invalid_harness":
        classification = "invalid_harness"
        readiness = "underdetermined"
        action = NEXT_ACTION_UNDERDETERMINED
    hashes = _input_hashes(spec, review_path, formal_path, data, source)
    compact = _compact_summary(confirmation_records, attribution=classification, readiness=readiness, prior_status=prior_status)
    compact["step5000_paired_mse"] = best_step5000
    compact["step0_paired_mse"] = baseline
    compact["identity_reference_paired_mse"] = identity_ref
    compact["maximum_information_discontinuity"] = _maximum_discontinuity(confirmation_records["scaled_chen_step_5000"], selection_records["scaled_chen_step_5000"])
    panel_payload = {
        "schema": PANEL_SCHEMA,
        "selection_indices": data.selection_indices,
        "confirmation_indices": data.confirmation_indices,
        "selection_labels": selection_labels,
        "confirmation_labels": confirmation_labels,
        "per_sample_mse": {"selection": {name: item.sample_mse for name, item in selection_eval.items()}, "confirmation": {name: item.sample_mse for name, item in confirmation_eval.items()}},
        "source_hashes": source.hashes,
        "affects_selection": False,
    }
    storage_estimate = sum(value.numel() * value.element_size() for value in panel_payload["per_sample_mse"]["confirmation"].values())
    if storage_estimate > MAX_STORAGE_BYTES:
        raise RuntimeError("manifold readiness compact panel exceeds storage gate")
    atomic_torch_save(panel_path, panel_payload)
    panel_hash = file_sha256(panel_path)
    atomic_write_json(compact_path, {**compact, "panel_sha256": panel_hash})
    contract = {
        "selection": selection_records,
        "confirmation": confirmation_records,
        "attribution": classification,
        "generation_readiness": readiness,
        "prior_status": prior_status,
        "prior_checks": prior_checks,
        "bootstrap": bootstrap,
        "controls": {
            "cross_swaps": cross_swaps,
            "identity_reference": {
                "selection_paired_mse": selection_records["identity_step_5500"]["reconstruction"]["paired_mse"],
                "confirmation_paired_mse": confirmation_records["identity_step_5500"]["reconstruction"]["paired_mse"],
            },
            "historical_lineage": control_lineage,
            "identity_bypass": {"status": "not_reused_lineage_mismatch", "permitted_consumer": "background_only"},
        },
        "maximum_information_discontinuity": compact["maximum_information_discontinuity"],
        "panel_path": str(panel_path.resolve()),
        "panel_sha256": panel_hash,
        "compact_report_path": str(compact_path.resolve()),
        "compact_report_sha256": file_sha256(compact_path),
        "resource": {"free_cuda_bytes_before": free_bytes, "total_cuda_bytes": total_bytes, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "wall_clock": "observational_only"},
        "generative_manifold_supported": False,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "evaluator_calls": 0,
    }
    payload = _record(AUDIT_SCHEMA, status, classification, spec, state, data, hashes, all_gates, (action,), contract, None if status == "valid_failed" else "manifold readiness harness failed")
    payload["generation_readiness"] = readiness
    payload["generative_manifold_supported"] = False
    _write_new(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if readiness == "canary_ready" else 2


def _maximum_discontinuity(confirmation: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    selection_retention = selection.get("retention", {})
    confirmation_retention = confirmation.get("retention", {})
    values = {name: {"selection": selection_retention.get(name, {}).get("geometric_mean_retention"), "confirmation": confirmation_retention.get(name, {}).get("geometric_mean_retention")} for name, *_ in SEGMENTS}
    deficits = {name: (1.0 - float(item["selection"]) + 1.0 - float(item["confirmation"])) / 2.0 if isinstance(item["selection"], (int, float)) and isinstance(item["confirmation"], (int, float)) else float("nan") for name, item in values.items()}
    ordered = sorted(deficits.items(), key=lambda pair: pair[1], reverse=True)
    if not ordered or not math.isfinite(ordered[0][1]):
        return {"status": "attribution_underdetermined", "segment": None, "segments": values}
    unique = len(ordered) < 2 or ordered[0][1] - ordered[1][1] >= 0.05
    same = values[ordered[0][0]]["selection"] is not None and values[ordered[0][0]]["confirmation"] is not None
    return {"status": "unique" if unique and same else "distributed_or_underdetermined", "segment": ordered[0][0] if unique and same else None, "segments": values}


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-manifold-readiness-validate":
        return validate(args)
    if args.research_action == "scaled-chen-manifold-readiness-run":
        return run(args)
    raise ValueError(f"unsupported manifold readiness action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "CONFIG_SCHEMA",
    "HYPOTHESIS",
    "PANEL_SCHEMA",
    "REVIEW_SCHEMA",
    "SOURCE_HASHES",
    "VALIDATION_SCHEMA",
    "_assert_contract",
    "_classify_attribution",
    "_classify_prior_accessibility",
    "_load_data",
    "_load_review",
    "_load_spec",
    "_maximum_discontinuity",
    "_permuted_labels",
    "_rank_metrics",
    "_select_indices",
    "_stratified_bootstrap_ci",
    "dispatch",
    "run",
    "validate",
]
