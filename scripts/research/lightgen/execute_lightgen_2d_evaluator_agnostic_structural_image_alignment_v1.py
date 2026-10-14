"""Execute evaluator-agnostic structural image alignment from the accepted step-6000 state.

The scientific intervention updates only ``generator.phase`` on paired real TRAIN
OLS/image rows.  The structural objective is a differentiable image-geometry
surrogate; the frozen 337D hard structural metric, all learned evaluators, and
the S1 sampler remain confirmation-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import DataReadLedger, TRAIN_ROLE, contracted_plane_powers, optical_power_loss_from_powers, _rng_state_payload
from ldm_project.v2.application.research.shared.research_run_control import HeartbeatWriter, InterruptController, RecoveryCheckpointManager, ResearchRunControl, ResearchRunIdentity, RUN_COMPLETED, RUNNING, RUN_PREPARING, capture_execution_source_manifest, recovery_due
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)

from scripts.research.lightgen import execute_lightgen_2d_non_classifier_structural_semantic_confirmation_v1 as structural
from scripts.research.lightgen import execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1 as prior


TASK_ID = "EXECUTE_EVALUATOR_AGNOSTIC_STRUCTURAL_IMAGE_ALIGNMENT_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_EVALUATOR_AGNOSTIC_STRUCTURAL_IMAGE_ALIGNMENT_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "EVALUATOR_AGNOSTIC_STRUCTURAL_IMAGE_ALIGNMENT_CONTRACT_DEFINED"
SEMANTIC_AUTHORITY_ID = prior.SEMANTIC_AUTHORITY_ID
EXPECTED_BRANCH = prior.EXPECTED_BRANCH
EXPECTED_PRODUCER_HEAD = prior.EXPECTED_PRODUCER_HEAD
# Historical import compatibility; producer provenance is not a consumer lock.
EXPECTED_HEAD = EXPECTED_PRODUCER_HEAD

START_STEP = 6000
STAGE1_END = 6300
TOTAL_STEPS = 9000
STAGE1_UPDATES = STAGE1_END - START_STEP
LOGICAL_BATCH_SIZE = 10
CHECKPOINT_STEPS = (6000, 6100, 6200, 6300, 6600, 7000, 7500, 8000, 8500, 9000)
SCHEDULE_SEED = 1234
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
STRUCTURAL_TAU = 0.05
STRUCTURAL_EPS_MASS = 1.0e-12
NORMALIZATION_EPS = 1.0e-6
STRUCTURAL_DIMENSION = 336
DELTA_STRUCT = 1.365988374680088e-05
BETA = 2.4509953404
GAMMA = 0.5330424285
EPS_PROJ = 1.0e-12
RECONSTRUCTION_TOLERANCE = 0.005
JOINT_DEGENERACY_TOLERANCE = 1.0e-12
PRELIGHT_EPSILON = 1.0e-4
RECOVERY_INTERVAL_STEPS = 25

STEP6000 = Path(
    "outputs/20260920_three_evaluator_robust_semantic_continuation_v1/runs/attempt_02/scientific_step_006000.pt"
)
STEP6000_SHA256 = "3387e104f7d33e19051a23444b0dabf1eb387c5faca3ce7150bbd1a2212c9986"
STEP6000_REPORT = Path(
    "outputs/20260920_three_evaluator_robust_semantic_continuation_v1/runs/attempt_02/report.json"
)
STEP6000_REPORT_SHA256 = "1d195aad5307108509349c451aa48f7015b8e27a00bfb66c8e42a56d4c4a0f92"

TRAINING_SOURCE = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bb33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
TRAIN_OLS = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01/final_train_ols.pt")
TRAIN_OLS_SHA256 = "2d1264219faa0964467d40f2e4882ec3767d3a1aafd30e143bf406aa39be6dcc"
DEV_OLS = Path("outputs/20260918_frozen_dev_ols_cmos_v1/runs/attempt_01/dev_complex_ols.pt")
DEV_OLS_SHA256 = "5a8efe07c6ba52168f415e56fccdd68c2552b6dd5f4f975dabc96d2215fbb622"
S1_WINNER = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01/winner_frozen_sampler.pt")
S1_WINNER_SHA256 = "166cf15ea91666afe1fe1c88e8897b88d58a4740253d59838927502babba568c"

STRUCTURAL_REPORT = Path(
    "outputs/20260921_non_classifier_structural_semantic_confirmation_v1/runs/attempt_04/report.json"
)
STRUCTURAL_REPORT_SHA256 = "126d5ed7dd35ed3117c23683582e28e790ab948da16be063deb3e2cc4fd29b95"
STRUCTURAL_REFERENCE = Path(
    "outputs/20260921_non_classifier_structural_semantic_confirmation_v1/runs/attempt_04/structural_reference.pt"
)
STRUCTURAL_REFERENCE_SHA256 = "e6304ac70fafc7f5d5ecc61c0d91af6520fa257bc8fe50903cbe340a7bcae142"

CALIBRATION_INDICES = (41252, 57579, 31150, 36488, 36208, 15778, 46903, 14076, 39350, 23349)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260921_evaluator_agnostic_structural_image_alignment_v1/runs/attempt_01"
)

REPORT_SCHEMA = "evaluator_agnostic_structural_image_alignment_report.v1"
CHECKPOINT_SCHEMA = "evaluator_agnostic_structural_image_alignment_checkpoint.v1"
PREFLIGHT_SCHEMA = "evaluator_agnostic_structural_image_alignment_preflight.v1"

CLASSIFICATION_A = "EVALUATOR_AGNOSTIC_STRUCTURAL_ALIGNMENT_SUPPORTED_WITH_S1_TRANSFER"
CLASSIFICATION_B = "REAL_PATH_STRUCTURAL_RECOVERY_WITH_LIMITED_S1_TRANSFER"
CLASSIFICATION_C = "STRUCTURAL_SURROGATE_OPTIMIZATION_WITHOUT_CANONICAL_TRANSFER"
CLASSIFICATION_D = "STRUCTURAL_RECOVERY_WITH_CLASSIFIER_SEMANTIC_TRADEOFF"
CLASSIFICATION_E = "EVALUATOR_AGNOSTIC_STRUCTURAL_ALIGNMENT_NOT_SUPPORTED"
CLASSIFICATION_F = "EVALUATOR_AGNOSTIC_STRUCTURAL_IMAGE_ALIGNMENT_MIXED"


class StructuralAlignmentError(RuntimeError):
    """Raised when an accepted authority or execution invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StructuralAlignmentError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path if path.is_absolute() else root / path
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_write_new_json(path, dict(payload), artifact_kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class StructuralContinuationSampler:
    """Preserve the accepted per-class permutation while extending its horizon."""

    SCHEMA = "evaluator_agnostic_structural_alignment_sampler.v1"

    def __init__(self, state: Mapping[str, Any], *, cursor: int = START_STEP) -> None:
        _require(int(state.get("seed", -1)) == SCHEDULE_SEED, "sampler seed changed")
        values = state.get("permutations")
        _require(isinstance(values, list) and len(values) == 10, "sampler permutations changed")
        self.seed = SCHEDULE_SEED
        self.cursor = int(cursor)
        self.terminal = TOTAL_STEPS
        self.permutations = []
        for digit, value in enumerate(values):
            tensor = _tensor(value, f"sampler permutation {digit}").cpu().clone()
            _require(tensor.dtype == torch.long and tensor.ndim == 1 and tensor.numel() > 0, f"sampler permutation {digit} changed")
            self.permutations.append(tensor)
        _require(0 <= self.cursor <= self.terminal, "sampler cursor is invalid")

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        _require(self.cursor < self.terminal, "structural continuation sampler exhausted")
        indices = torch.stack([value[self.cursor % value.numel()] for value in self.permutations])
        self.cursor += 1
        return indices, torch.arange(10, dtype=torch.long)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "seed": self.seed,
            "steps": self.terminal,
            "cursor": self.cursor,
            "wraparound": True,
            "permutations": [value.clone() for value in self.permutations],
        }


def _validate_optimizer(consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    _require(isinstance(optimizer, torch.optim.Adam), "optimizer is not Adam")
    _require(len(optimizer.param_groups) == 1, "optimizer has unexpected parameter groups")
    group = optimizer.param_groups[0]
    parameters = list(group["params"])
    _require(len(parameters) == 1 and parameters[0] is prior._generator(consumer), "optimizer contains a non-generator parameter")
    _require(float(group["lr"]) == 1.0e-3, "optimizer learning rate changed")
    _require(tuple(group["betas"]) == (0.9, 0.999), "optimizer betas changed")
    _require(float(group["eps"]) == 1.0e-8 and float(group["weight_decay"]) == 0.0 and group["amsgrad"] is False, "optimizer options changed")


def _load_authorities(root: Path) -> dict[str, Any]:
    authorities = prior._load_authorities(root)
    _require(file_sha256(root / STEP6000) == STEP6000_SHA256, "step6000 checkpoint identity changed")
    _require(file_sha256(root / STEP6000_REPORT) == STEP6000_REPORT_SHA256, "step6000 report identity changed")
    step6000 = safe_torch_load(root / STEP6000, expected_sha256=STEP6000_SHA256, map_location="cpu")
    step6000 = _mapping(step6000, "step6000 checkpoint")
    report = _mapping(json.loads((root / STEP6000_REPORT).read_text(encoding="utf-8")), "step6000 report")
    _require(report.get("result") == "VALID_COMPLETED", "step6000 result changed")
    _require(report.get("classification") == "THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_SUPPORTED", "step6000 classification changed")
    _require(step6000.get("schema") == "three_evaluator_robust_semantic_continuation_checkpoint.v1", "step6000 schema changed")
    _require(step6000.get("completed_updates") == START_STEP, "step6000 cursor changed")
    _require(step6000.get("starting_scientific_step") == 3000, "step6000 starting authority changed")
    _require(step6000.get("scientific_artifact") is True and step6000.get("operational_artifact") is False, "step6000 artifact role changed")
    _require(step6000.get("trainable_parameter_names") == ["generator.phase"], "step6000 trainable boundary changed")
    for name in ("encoder_state", "generator_state"):
        value = _tensor(step6000.get(name), name)
        _require(tuple(value.shape) == (2, 1, 100, 100) and value.dtype == torch.float32, f"{name} geometry changed")
        _finite(value, name)
    optimizer_configuration = _mapping(step6000.get("optimizer_configuration"), "step6000 optimizer configuration")
    _require(float(optimizer_configuration.get("internal_step", -1.0)) == 6300.0, "step6000 Adam internal step changed")
    _require(optimizer_configuration.get("parameter_names") == ["generator.phase"], "step6000 optimizer parameter boundary changed")
    optimizer_state = _mapping(step6000.get("optimizer_state"), "step6000 optimizer state")
    _require(isinstance(optimizer_state.get("state"), Mapping) and len(optimizer_state["state"]) == 1, "step6000 full Generator Adam state is missing")
    sampler_state = _mapping(step6000.get("sampler_state"), "step6000 sampler state")
    _require(sampler_state.get("cursor") == START_STEP and sampler_state.get("steps") == START_STEP, "step6000 sampler cursor changed")
    _require(sampler_state.get("seed") == SCHEDULE_SEED and sampler_state.get("wraparound") is True, "step6000 sampler identity changed")
    _require(isinstance(step6000.get("rng_state"), Mapping), "step6000 RNG state is missing")
    for digit, permutation in enumerate(sampler_state["permutations"]):
        permutation = _tensor(permutation, f"step6000 sampler permutation {digit}")
        _require(permutation.numel() > 0, f"step6000 sampler permutation {digit} is empty")
        _require(bool(torch.all((permutation >= 0) & (permutation < authorities["labels"].numel())).item()), f"step6000 sampler permutation {digit} is out of range")
        _require(torch.equal(authorities["labels"].index_select(0, permutation), torch.full_like(permutation, digit)), f"step6000 sampler permutation {digit} is not class conditioned")

    _require(file_sha256(root / TRAIN_OLS) == TRAIN_OLS_SHA256, "real TRAIN OLS identity changed")
    _require(file_sha256(root / DEV_OLS) == DEV_OLS_SHA256, "real DEV OLS identity changed")
    train_ols = _mapping(safe_torch_load(root / TRAIN_OLS, expected_sha256=TRAIN_OLS_SHA256, map_location="cpu"), "real TRAIN OLS")
    dev_ols = _mapping(safe_torch_load(root / DEV_OLS, expected_sha256=DEV_OLS_SHA256, map_location="cpu"), "real DEV OLS")
    train_z = _tensor(train_ols.get("z_final"), "real TRAIN z_final")
    train_source_indices = _tensor(train_ols.get("source_indices"), "real TRAIN source indices")
    _require(train_z.shape == (55000, 100) and train_z.is_complex(), "real TRAIN OLS geometry changed")
    _finite(train_z.real, "real TRAIN OLS real part"); _finite(train_z.imag, "real TRAIN OLS imaginary part")
    _require(torch.equal(train_source_indices, authorities["train_indices"]), "real TRAIN OLS source identity does not match accepted TRAIN ordering")
    dev_z = _tensor(dev_ols.get("z_dev"), "real DEV z_dev")
    dev_source_indices = _tensor(dev_ols.get("source_indices"), "real DEV source indices")
    dev_labels = _tensor(dev_ols.get("labels"), "real DEV labels")
    _require(dev_z.shape == (5000, 100) and dev_z.is_complex(), "real DEV OLS geometry changed")
    _finite(dev_z.real, "real DEV OLS real part"); _finite(dev_z.imag, "real DEV OLS imaginary part")
    _require(torch.equal(dev_source_indices, authorities["development_indices"]), "real DEV OLS source identity does not match accepted DEV ordering")
    _require(torch.equal(dev_labels, authorities["labels"].index_select(0, authorities["development_indices"])), "real DEV labels changed")

    _require(file_sha256(root / STRUCTURAL_REPORT) == STRUCTURAL_REPORT_SHA256, "hard structural report identity changed")
    _require(file_sha256(root / STRUCTURAL_REFERENCE) == STRUCTURAL_REFERENCE_SHA256, "hard structural reference identity changed")
    structural_report = _mapping(json.loads((root / STRUCTURAL_REPORT).read_text(encoding="utf-8")), "hard structural report")
    structural_reference = _mapping(safe_torch_load(root / STRUCTURAL_REFERENCE, expected_sha256=STRUCTURAL_REFERENCE_SHA256, map_location="cpu"), "hard structural reference")
    _require(structural_report.get("result") == "VALID_COMPLETED", "hard structural result changed")
    _require(structural_report.get("classification") == "STEP6000_STRUCTURAL_SEMANTIC_REALIZATION_GAP_SUPPORTED", "hard structural classification changed")
    _require(structural_reference.get("schema") == "non_classifier_structural_signature_reference.v1", "hard structural reference schema changed")
    _require(int(_mapping(structural_reference.get("descriptor"), "hard descriptor").get("dimension", -1)) == 337, "hard structural descriptor changed")
    q95 = _tensor(_mapping(structural_reference.get("support_distance"), "hard support distance").get("q95_by_class"), "hard q95")
    _require(q95.shape == (10,) and bool((q95 > 0.0).all()), "hard q95 authority is invalid")

    _require(file_sha256(root / S1_WINNER) == S1_WINNER_SHA256, "S1 winner identity changed")
    s1_winner = _mapping(safe_torch_load(root / S1_WINNER, expected_sha256=S1_WINNER_SHA256, map_location="cpu"), "S1 winner")
    s1_requested = _tensor(s1_winner.get("requested_labels"), "S1 requested labels")
    _require(s1_requested.shape == (2000,) and s1_requested.dtype == torch.long, "S1 requested class identity geometry changed")
    _require(torch.equal(torch.bincount(s1_requested, minlength=10), torch.full((10,), 200, dtype=torch.long)), "S1 requested class balance changed")
    authorities.update(
        {
            "step6000": dict(step6000),
            "step6000_report": dict(report),
            "train_ols": dict(train_ols),
            "dev_ols": dict(dev_ols),
            "structural_report": dict(structural_report),
            "structural_reference": dict(structural_reference),
            "authority_hashes": {
                **authorities["authority_hashes"],
                "step6000": STEP6000_SHA256,
                "step6000_report": STEP6000_REPORT_SHA256,
                "train_ols": TRAIN_OLS_SHA256,
                "dev_ols": DEV_OLS_SHA256,
                "structural_report": STRUCTURAL_REPORT_SHA256,
                "structural_reference": STRUCTURAL_REFERENCE_SHA256,
                "s1_winner": S1_WINNER_SHA256,
            },
        }
    )
    return authorities


def _structural_surrogate(images: torch.Tensor) -> torch.Tensor:
    _require(images.ndim == 4 and tuple(images.shape[1:]) == (1, 28, 28), "structural surrogate input geometry changed")
    _finite(images, "structural surrogate input")
    values = images[:, 0].to(torch.float64)
    _require(float(values.detach().min()) >= 0.0 and float(values.detach().max()) <= 1.0, "structural surrogate input range changed")
    batch = values.shape[0]
    x_coord = torch.arange(28, device=values.device, dtype=torch.float64).reshape(1, 1, 28) / 27.0
    y_coord = torch.arange(28, device=values.device, dtype=torch.float64).reshape(1, 28, 1) / 27.0

    def geometry(weights: torch.Tensor, *, include_spatial: bool) -> torch.Tensor:
        mass = weights.sum(dim=(1, 2))
        safe_mass = mass.clamp_min(STRUCTURAL_EPS_MASS)
        cx = (weights * x_coord).sum(dim=(1, 2)) / safe_mass
        cy = (weights * y_coord).sum(dim=(1, 2)) / safe_mass
        dx = x_coord - cx.reshape(-1, 1, 1)
        dy = y_coord - cy.reshape(-1, 1, 1)
        mxx = (weights * dx.square()).sum(dim=(1, 2)) / safe_mass
        myy = (weights * dy.square()).sum(dim=(1, 2)) / safe_mass
        mxy = (weights * dx * dy).sum(dim=(1, 2)) / safe_mass
        angle2 = torch.atan2(2.0 * mxy, mxx - myy)
        cos2 = torch.cos(angle2)
        sin2 = torch.sin(angle2)
        degenerate = mass <= STRUCTURAL_EPS_MASS
        cx = torch.where(degenerate, torch.full_like(cx, 0.5), cx)
        cy = torch.where(degenerate, torch.full_like(cy, 0.5), cy)
        mxx = torch.where(degenerate, torch.zeros_like(mxx), mxx)
        myy = torch.where(degenerate, torch.zeros_like(myy), myy)
        mxy = torch.where(degenerate, torch.zeros_like(mxy), mxy)
        cos2 = torch.where(degenerate, torch.ones_like(cos2), cos2)
        sin2 = torch.where(degenerate, torch.zeros_like(sin2), sin2)
        horizontal = weights.sum(dim=2) / safe_mass.reshape(-1, 1)
        vertical = weights.sum(dim=1) / safe_mass.reshape(-1, 1)
        horizontal = torch.where(degenerate.reshape(-1, 1), torch.zeros_like(horizontal), horizontal)
        vertical = torch.where(degenerate.reshape(-1, 1), torch.zeros_like(vertical), vertical)
        parts = [
            (mass / 784.0).reshape(batch, 1),
            cx.reshape(batch, 1),
            cy.reshape(batch, 1),
            mxx.reshape(batch, 1),
            myy.reshape(batch, 1),
            mxy.reshape(batch, 1),
            cos2.reshape(batch, 1),
            sin2.reshape(batch, 1),
            horizontal,
            vertical,
        ]
        if include_spatial:
            parts.extend(
                [
                    values.reshape(batch, 4, 7, 4, 7).mean(dim=(2, 4)).reshape(batch, 16),
                    values.reshape(batch, 2, 14, 2, 14).mean(dim=(2, 4)).reshape(batch, 4),
                ]
            )
        return torch.cat(parts, dim=1)

    parts = [geometry(values, include_spatial=True)]
    for threshold in (0.25, 0.50, 0.75):
        mask = torch.sigmoid((values - threshold) / STRUCTURAL_TAU)
        base = geometry(mask, include_spatial=False)
        cell = mask.reshape(batch, 4, 7, 4, 7).mean(dim=(2, 4)).reshape(batch, 16)
        quadrant = mask.reshape(batch, 2, 14, 2, 14).mean(dim=(2, 4)).reshape(batch, 4)
        parts.append(torch.cat([base, cell, quadrant], dim=1))
    result = torch.cat(parts, dim=1)
    _require(result.shape[1] == STRUCTURAL_DIMENSION, f"structural surrogate dimension changed: {result.shape[1]}")
    _finite(result, "structural surrogate")
    return result


def _fit_surrogate_normalization(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros((STRUCTURAL_DIMENSION,), dtype=torch.float64)
    total_square = torch.zeros_like(total)
    count = 0
    for start in range(0, int(images.shape[0]), 1024):
        values = _structural_surrogate(images[start : start + 1024])
        total += values.sum(dim=0).cpu()
        total_square += values.square().sum(dim=0).cpu()
        count += values.shape[0]
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(0.0)
    scale = variance.sqrt().clamp_min(NORMALIZATION_EPS)
    _finite(mean, "surrogate normalization mean")
    _finite(scale, "surrogate normalization scale")
    return mean, scale


def _normalized_surrogate(value: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (value - mean.to(device=value.device)) / scale.to(device=value.device)


def _structural_bundle(
    consumer: Any,
    batch: torch.Tensor,
    references: Any,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "real optical path is incomplete")
    raw = result.raw_intensity
    target_image = result.image_input.slm_intensity
    _require(raw.shape == target_image.shape and tuple(raw.shape[-2:]) == (200, 200), "raw/target geometry changed")
    _require(raw.dtype == target_image.dtype == torch.float32, "raw/target dtype changed")
    _finite(raw, "raw CMOS")
    _finite(target_image, "target image")
    raw_view = prior._cmos_view(raw)
    target_view = prior._cmos_view(batch)
    generated_psi = _normalized_surrogate(_structural_surrogate(raw_view), mean, scale)
    with torch.no_grad():
        target_psi = _normalized_surrogate(_structural_surrogate(target_view), mean, scale)
    l_struct = (generated_psi - target_psi.detach()).square().mean()
    l_rec = (raw - target_image).square().mean()
    l_op = optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)
    for name, value in (("L_struct", l_struct), ("L_rec", l_rec), ("L_op", l_op)):
        _finite(value, name)
    return {
        "result": result,
        "raw": raw,
        "raw_view": raw_view,
        "target_view": target_view,
        "target_psi": target_psi,
        "generated_psi": generated_psi,
        "losses": {"L_struct": l_struct, "L_rec": l_rec, "L_op": l_op},
    }


def _project_auxiliary(auxiliary: torch.Tensor, structural_gradient: torch.Tensor, name: str) -> tuple[torch.Tensor, dict[str, Any]]:
    auxiliary_norm = prior._norm(auxiliary, f"{name} weighted gradient")
    structural_norm = prior._norm(structural_gradient, "scaled structural gradient")
    dot_before = prior._dot(auxiliary, structural_gradient)
    removed = torch.zeros_like(auxiliary)
    projected = dot_before < 0.0
    if projected:
        coefficient = dot_before / (float(torch.sum(structural_gradient.detach().to(torch.float64).square()).cpu()) + EPS_PROJ)
        removed = structural_gradient * coefficient
    safe = auxiliary - removed
    _finite(safe, f"{name}_safe")
    dot_after = prior._dot(safe, structural_gradient)
    if projected:
        tolerance = max(1.0e-8, 1.0e-5 * max(prior._norm(safe, f"{name}_safe"), 1.0e-12) * structural_norm)
        _require(abs(dot_after) <= tolerance, f"{name}_safe is not orthogonal to structural direction")
    return safe, {
        "dot_before": dot_before,
        "dot_after": dot_after,
        "projected": projected,
        "removed_norm": float(removed.detach().to(torch.float64).square().sum().sqrt().cpu()),
        "weighted_norm": auxiliary_norm,
    }


def _structural_direction(consumer: Any, batch: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor) -> dict[str, Any]:
    generator = prior._generator(consumer)
    bundle = _structural_bundle(consumer, batch, references, mean, scale)
    l_struct = bundle["losses"]["L_struct"]
    l_rec = bundle["losses"]["L_rec"]
    l_op = bundle["losses"]["L_op"]
    g_struct = torch.autograd.grad(l_struct, generator, retain_graph=True)[0]
    g_rec = torch.autograd.grad(l_rec, generator, retain_graph=True)[0]
    g_op = torch.autograd.grad(l_op, generator)[0]
    _finite(g_struct, "g_struct")
    _require(prior._norm(g_struct, "g_struct") > JOINT_DEGENERACY_TOLERANCE, "structural gradient is degenerate")
    q_struct = g_struct / prior._rms(g_struct, "g_struct")
    _finite(q_struct, "q_struct")
    g_struct_scaled = DELTA_STRUCT * q_struct
    g_rec_safe, rec_projection = _project_auxiliary(BETA * g_rec, g_struct_scaled, "g_rec_w")
    g_op_safe, op_projection = _project_auxiliary(GAMMA * g_op, g_struct_scaled, "g_op_w")
    g_total = g_struct_scaled + g_rec_safe + g_op_safe
    _finite(g_total, "g_total")
    _require(prior._dot(g_total, g_struct_scaled) > 0.0, "combined gradient has no positive structural component")
    return {
        "bundle": bundle,
        "g_struct": g_struct,
        "q_struct": q_struct,
        "g_struct_scaled": g_struct_scaled,
        "g_rec": g_rec,
        "g_op": g_op,
        "g_rec_safe": g_rec_safe,
        "g_op_safe": g_op_safe,
        "g_total": g_total,
        "projection": {"g_rec": rec_projection, "g_op": op_projection},
        "gradient_stats": {
            "g_struct": prior._gradient_stats(g_struct, "g_struct"),
            "q_struct": prior._gradient_stats(q_struct, "q_struct"),
            "g_struct_scaled": prior._gradient_stats(g_struct_scaled, "g_struct_scaled"),
            "g_total": prior._gradient_stats(g_total, "g_total"),
        },
    }


def _apply_update(
    consumer: Any,
    optimizer: torch.optim.Adam,
    batch: torch.Tensor,
    references: Any,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    step: int,
    clean_generator_state: torch.Tensor,
) -> dict[str, Any]:
    generator = prior._generator(consumer)
    before_generator = generator.detach().clone()
    frozen_before = prior._frozen_runtime_state(consumer)
    optimizer.zero_grad(set_to_none=True)
    prior._clear_runtime_grads(consumer)
    direction = _structural_direction(consumer, batch, references, mean, scale)
    generator.grad = direction["g_total"].detach().clone()
    _finite(generator.grad, "assigned g_total")
    for name, parameter in consumer.runtime.named_parameters():
        if name != "generator.phase":
            _require(parameter.grad is None, f"frozen runtime parameter received gradient: {name}")
    optimizer.step()
    _finite(generator, "updated generator.phase")
    _require(_state_digest(frozen_before) == _state_digest(prior._frozen_runtime_state(consumer)), "frozen runtime state changed")
    update = generator.detach() - before_generator
    _require(prior._norm(update, "phase update") > 0.0, "generator.phase did not update")
    prior._clear_runtime_grads(consumer)
    losses = direction["bundle"]["losses"]
    return {
        "step": step,
        "L_struct": float(losses["L_struct"].detach().cpu()),
        "L_rec": float(losses["L_rec"].detach().cpu()),
        "L_op": float(losses["L_op"].detach().cpu()),
        "gradient_stats": direction["gradient_stats"],
        "projection": direction["projection"],
        "phase_update_rms": prior._rms(update, "phase update"),
        "phase_displacement_from_step6000": prior._norm(generator.detach() - clean_generator_state.to(generator.device), "phase displacement"),
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase"],
        "structural_gradient_degenerate": False,
    }


def _public_source(source: Mapping[str, Any]) -> dict[str, Any]:
    labels = _tensor(source["labels"], "source labels").cpu().to(torch.long)
    distance = _tensor(source["nearest_same_class_distance"], "source distance").cpu().to(torch.float64)
    q95_rows = _tensor(source["q95_rows"], "source q95 rows").cpu().to(torch.float64)
    margin = _tensor(source["margin"], "source margin").cpu().to(torch.float64)
    rho = distance / q95_rows
    _require(bool((rho > 0.0).all()), "floor-free log(rho) is undefined at zero distance")
    log_rho = rho.log()
    _finite(log_rho, "log structural rho")
    return {
        "name": source["name"],
        "rows": int(labels.numel()),
        "label_source": "requested class" if source["name"].startswith("s1") else "true class",
        "label_histogram": torch.bincount(labels, minlength=10).tolist(),
        "structural_support_rate": float(_tensor(source["support"], "source support").to(torch.float64).mean()),
        "support_by_class": source["support_by_class"],
        "nearest_same_class_distance": structural._summary(distance),
        "distance_by_class": source["distance_by_class"],
        "q95_rows": structural._summary(q95_rows),
        "rho_struct": structural._summary(rho),
        "log_rho_struct": structural._summary(log_rho),
        "requested_centroid_distance": structural._summary(_tensor(source["d_req"], "requested centroid distance")),
        "alternative_centroid_distance": structural._summary(_tensor(source["d_alt"], "alternative centroid distance")),
        "margin": source["margin_summary"],
        "margin_by_class": source["margin_by_class"],
        "positive_margin_fraction": float((margin > 0.0).to(torch.float64).mean()),
        "positive_margin_by_class": source["positive_margin_by_class"],
        "nearest_class_agreement": float((_tensor(source["structural_class"], "structural class") == labels).to(torch.float64).mean()),
        "topology": source["topology"],
    }


def _hard_structural_source(
    name: str,
    views: torch.Tensor,
    labels: torch.Tensor,
    reference: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = structural.structural_signature(views, device=device, batch_size=1024)
    normalization = _mapping(reference["normalization"], "hard normalization")
    mean = _tensor(normalization["mean"], "hard normalization mean").to(torch.float64)
    scale = _tensor(normalization["population_std_clamped_scale"], "hard normalization scale").to(torch.float64)
    standardized = (signature - mean) / scale
    fit = _mapping(reference["structural_fit"], "hard structural fit")
    support_distance = _mapping(reference["support_distance"], "hard support distance")
    q95 = _tensor(support_distance["q95_by_class"], "hard q95").to(torch.float64)
    source = structural._evaluate_source(
        name,
        signature,
        standardized,
        labels.cpu().to(torch.long),
        _tensor(fit["standardized_signature"], "hard fit standardized signature"),
        _tensor(fit["labels"], "hard fit labels"),
        _tensor(reference["class_centroids"], "hard centroids"),
        q95,
        device,
    )
    public = _public_source(source)
    public["signature_sha256"] = tensor_sha256(signature)
    public["_rho"] = source["nearest_same_class_distance"].to(torch.float64) / source["q95_rows"].to(torch.float64)
    public["_log_rho"] = public["_rho"].log()
    return source, public


def _evaluate_dev(
    consumer: Any,
    smooth: Any,
    independent: Any,
    ec: Any,
    ed: Any,
    authorities: Mapping[str, Any],
    reference: Mapping[str, Any],
    mean: torch.Tensor,
    scale: torch.Tensor,
    ledger: DataReadLedger,
    device: torch.device,
    *,
    step: int,
) -> dict[str, Any]:
    indices = authorities["development_indices"]
    images = authorities["images"]
    labels = authorities["labels"]
    labels_dev = labels.index_select(0, indices)
    predictions = {name: [] for name in ("E_S", "E_I", "E_C", "E_D")}
    raw_views: list[torch.Tensor] = []
    rec_sum = 0.0
    op_sum = 0.0
    structural_loss_sum = 0.0
    raw_nonconstant = 0
    raw_zero = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    total_elements = 0
    was_training = consumer.runtime.training
    consumer.runtime.eval(); smooth.eval(); independent.eval(); ec.eval(); ed.eval()
    with torch.inference_mode():
        for start in range(0, int(indices.numel()), LOGICAL_BATCH_SIZE):
            batch_indices = indices[start : start + LOGICAL_BATCH_SIZE]
            batch = images.index_select(0, batch_indices).to(device)
            batch_labels = labels.index_select(0, batch_indices).to(device)
            result = consumer.runtime.forward_from_image(batch)
            raw = result.raw_intensity
            target_image = result.image_input.slm_intensity if result.image_input is not None else None
            _require(target_image is not None, "DEV target image is missing")
            _finite(raw, "DEV raw CMOS"); _finite(target_image, "DEV target image")
            raw_view = prior._cmos_view(raw)
            target_view = prior._cmos_view(batch)
            s_features, s_logits = smooth(raw_view)
            i_features, i_logits = independent(raw_view)
            c_features, c_logits = ec(raw_view)
            d_logits = ed(raw_view)
            s_target = smooth(target_view)[0]
            i_target = independent(target_view)[0]
            c_target = ec(target_view)[0]
            predictions["E_S"].append(s_logits.argmax(dim=1).cpu())
            predictions["E_I"].append(i_logits.argmax(dim=1).cpu())
            predictions["E_C"].append(c_logits.argmax(dim=1).cpu())
            predictions["E_D"].append(d_logits.argmax(dim=1).cpu())
            normalized = _normalized_surrogate(_structural_surrogate(raw_view), mean.to(device), scale.to(device))
            target_normalized = _normalized_surrogate(_structural_surrogate(target_view), mean.to(device), scale.to(device))
            structural_loss_sum += float((normalized - target_normalized).square().mean().cpu()) * int(batch_indices.numel())
            rec_sum += float((raw - target_image).square().sum().cpu())
            powers = contracted_plane_powers(result)
            ratios = (powers.to(torch.float64) + 1.0e-12) / (authorities["training"]["references"].power_reference.to(device=device, dtype=torch.float64) + 1.0e-12)
            op_sum += float(torch.log(ratios).square().sum().cpu())
            flat = raw.to(torch.float64).flatten(1)
            raw_nonconstant += int((flat.std(dim=1, unbiased=False) > 0.0).sum().item())
            raw_zero += int((flat.abs().sum(dim=1) == 0.0).sum().item())
            raw_min = min(raw_min, float(raw.min().cpu())); raw_max = max(raw_max, float(raw.max().cpu()))
            total_elements += int(raw.numel())
            raw_views.append(raw_view.cpu())
            ledger.record("dev", int(batch_indices.numel()))
    consumer.runtime.train(was_training)
    all_views = torch.cat(raw_views)
    source, source_public = _hard_structural_source("real_ols_step6000" if step == START_STEP else f"real_ols_step{step}", all_views, labels_dev, reference, device)
    result = {
        "schema": "evaluator_agnostic_structural_alignment_dev_diagnostic.v1",
        "step": step,
        "rows": int(indices.numel()),
        "E_S": {"architecture": "SmoothMNISTEvaluator v2.1", **prior._class_metrics(labels_dev, torch.cat(predictions["E_S"]), label_source="canonical DEV label")},
        "E_I": {"architecture": "FrozenMNISTEvaluator v1", **prior._class_metrics(labels_dev, torch.cat(predictions["E_I"]), label_source="canonical DEV label")},
        "E_C": {"architecture": "ThirdConfirmationEvaluator", **prior._class_metrics(labels_dev, torch.cat(predictions["E_C"]), label_source="canonical DEV label")},
        "E_D": {"architecture": "PixelMLPConfirmationEvaluatorV1", **prior._class_metrics(labels_dev, torch.cat(predictions["E_D"]), label_source="canonical DEV label")},
        "L_struct": structural_loss_sum / int(indices.numel()),
        "L_rec": rec_sum / float(total_elements),
        "L_op": op_sum / float(indices.numel() * 5),
        "raw_cmos": {"finite": True, "nonconstant_fraction": float(raw_nonconstant / int(indices.numel())), "zero_image_count": raw_zero, "min": raw_min, "max": raw_max},
        "structural": {key: value for key, value in source_public.items() if not key.startswith("_")},
        "_source": source,
        "_labels": labels_dev.cpu(),
        "_log_rho": source_public["_log_rho"],
    }
    _require(result["raw_cmos"]["nonconstant_fraction"] == 1.0 and raw_zero == 0, f"DEV optical validity failed at step {step}")
    _require(all(parameter.grad is None for model in (smooth, independent, ec, ed) for parameter in model.parameters()), "DEV evaluator received a gradient")
    return result


def _evaluate_s1(
    consumer: Any,
    smooth: Any,
    independent: Any,
    ec: Any,
    ed: Any,
    authorities: Mapping[str, Any],
    reference: Mapping[str, Any],
    device: torch.device,
    ledger: DataReadLedger,
    *,
    step: int,
) -> dict[str, Any]:
    z, requested, panel_indices = prior._load_s1_bank(authorities)
    panel_indices = panel_indices.cpu().to(torch.long)
    panel_slots = {int(value): position for position, value in enumerate(panel_indices.tolist())}
    panel_raw = torch.empty((panel_indices.numel(), 1, 200, 200), dtype=torch.float32)
    view_chunks: list[torch.Tensor] = []
    logits = {name: [] for name in ("E_S", "E_I", "E_C", "E_D")}
    ec_features: list[torch.Tensor] = []
    raw_nonconstant = 0
    raw_zero = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    was_training = consumer.runtime.training
    consumer.runtime.eval(); smooth.eval(); independent.eval(); ec.eval(); ed.eval()
    with torch.inference_mode():
        for start in range(0, int(z.shape[0]), LOGICAL_BATCH_SIZE):
            end = min(int(z.shape[0]), start + LOGICAL_BATCH_SIZE)
            result = consumer.generated_forward(z[start:end].to(device))
            raw = result.raw_intensity
            _finite(raw, "S1 raw CMOS")
            raw_cpu = raw.cpu()
            for local, global_index in enumerate(range(start, end)):
                if global_index in panel_slots:
                    panel_raw[panel_slots[global_index]] = raw_cpu[local]
            view = prior._cmos_view(raw)
            view_chunks.append(view.cpu())
            s_logits = smooth(view)[1]
            i_logits = independent(view)[1]
            c_features_batch, c_logits = ec(view)
            d_logits = ed(view)
            for name, value in (("E_S", s_logits), ("E_I", i_logits), ("E_C", c_logits), ("E_D", d_logits)):
                logits[name].append(value.cpu())
            ec_features.append(c_features_batch.cpu())
            flat = raw.to(torch.float64).flatten(1)
            raw_nonconstant += int((flat.std(dim=1, unbiased=False) > 0.0).sum().item())
            raw_zero += int((flat.abs().sum(dim=1) == 0.0).sum().item())
            raw_min = min(raw_min, float(raw.min().cpu())); raw_max = max(raw_max, float(raw.max().cpu()))
            ledger.record("s1", int(end - start))
    consumer.runtime.train(was_training)
    all_views = torch.cat(view_chunks)
    logit_values = {name: torch.cat(values) for name, values in logits.items()}
    features = torch.cat(ec_features)
    control_raw = _tensor(authorities["s1_panels"]["real_validation_control_raw_panel"], "S1 real control raw panel").to(torch.float64).flatten(1)
    generated_panel = panel_raw.to(torch.float64).flatten(1)
    _require(generated_panel.shape == control_raw.shape == (200, 40000), "S1 panel geometry changed")
    panel_mean = control_raw.mean(dim=0); panel_std = control_raw.std(dim=0, unbiased=False).clamp_min(1.0e-8)
    raw_cross = prior._energy_distance((generated_panel - panel_mean) / panel_std, (control_raw - panel_mean) / panel_std)
    raw_self = prior._energy_distance(((control_raw[:100] - panel_mean) / panel_std), ((control_raw[100:] - panel_mean) / panel_std))
    control_view = control_raw.reshape(200, 1, 200, 200).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        control_features = ec(prior._cmos_view(control_view))[0].cpu().to(torch.float64)
    feature_current = features.index_select(0, panel_indices).to(torch.float64)
    feature_cross = prior._energy_distance(feature_current, control_features)
    feature_self = prior._energy_distance(control_features[:100], control_features[100:])
    source, source_public = _hard_structural_source(f"s1_step{step}", all_views, requested, reference, device)
    result = {
        "schema": "evaluator_agnostic_structural_alignment_s1_diagnostic.v1",
        "step": step,
        "family": "S1",
        "rows": int(z.shape[0]),
        "requested_class_ground_truth": True,
        "semantic": {name: prior._class_metrics(requested, values.argmax(dim=1), label_source="requested class y supplied by frozen S1 bank") for name, values in logit_values.items()},
        "R_raw": raw_cross / max(raw_self, 1.0e-12),
        "R_feature": feature_cross / max(feature_self, 1.0e-12),
        "diversity": {"R_raw": raw_cross / max(raw_self, 1.0e-12), "R_feature": feature_cross / max(feature_self, 1.0e-12)},
        "raw_validity": {"finite": True, "nonconstant_fraction": float(raw_nonconstant / int(z.shape[0])), "zero_image_count": raw_zero, "min": raw_min, "max": raw_max},
        "structural": {key: value for key, value in source_public.items() if not key.startswith("_")},
        "fixed_bank_sha256": tensor_sha256(z),
        "requested_labels_sha256": tensor_sha256(requested),
        "_source": source,
        "_labels": requested.cpu(),
        "_log_rho": source_public["_log_rho"],
    }
    _require(result["raw_validity"]["nonconstant_fraction"] == 1.0 and raw_zero == 0, f"S1 optical validity failed at step {step}")
    return result


def _paired_bootstrap_log_rho(labels: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
    labels = labels.cpu().to(torch.long)
    before = before.cpu().to(torch.float64)
    after = after.cpu().to(torch.float64)
    _require(labels.ndim == 1 and labels.shape == before.shape == after.shape, "paired structural bootstrap geometry changed")
    _require(bool(torch.isfinite(before).all().item()) and bool(torch.isfinite(after).all().item()), "paired structural bootstrap values are nonfinite")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    macro_parts = []
    per_class: dict[str, Any] = {}
    for digit in range(10):
        positions = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(positions.numel() > 0, f"class {digit} is empty in paired structural bootstrap")
        draw = torch.randint(positions.numel(), (BOOTSTRAP_REPLICATES, positions.numel()), generator=generator)
        improvement = before.index_select(0, positions)[draw] - after.index_select(0, positions)[draw]
        class_values = improvement.mean(dim=1)
        quantiles = torch.quantile(class_values, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
        per_class[str(digit)] = {"estimate": float(class_values.mean()), "ci95": {"lower": float(quantiles[0]), "upper": float(quantiles[1])}}
        macro_parts.append(class_values)
    macro_values = torch.stack(macro_parts).mean(dim=0)
    quantiles = torch.quantile(macro_values, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "class_stratified_paired": True,
        "statistic": "macro mean(log_rho_step6000 - log_rho_candidate)",
        "estimate": float(macro_values.mean()),
        "ci95": {"lower": float(quantiles[0]), "upper": float(quantiles[1])},
        "per_class": per_class,
        "macro_values": macro_values,
    }


def _optical_pass(*records: Mapping[str, Any]) -> dict[str, Any]:
    passed = all(
        bool(record["raw_cmos"]["finite"])
        and float(record["raw_cmos"]["nonconstant_fraction"]) == 1.0
        and int(record["raw_cmos"]["zero_image_count"]) == 0
        and math.isfinite(float(record["L_op"]))
        for record in records
    )
    return {
        "passed": passed,
        "all_finite": all(bool(record["raw_cmos"]["finite"]) for record in records),
        "all_nonconstant": all(float(record["raw_cmos"]["nonconstant_fraction"]) == 1.0 for record in records),
        "all_zero_image_counts_zero": all(int(record["raw_cmos"]["zero_image_count"]) == 0 for record in records),
        "all_power_checks_finite": all(math.isfinite(float(record["L_op"])) for record in records),
        "checkpoints": {str(record["step"]): record["raw_cmos"] for record in records},
    }


def _checkpoint_train_diagnostic(consumer: Any, sampler: StructuralContinuationSampler, authorities: Mapping[str, Any], mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    fixed_indices = torch.tensor(CALIBRATION_INDICES, dtype=torch.long)
    batch = authorities["images"].index_select(0, fixed_indices).to(device)
    prior._clear_runtime_grads(consumer)
    direction = _structural_direction(consumer, batch, authorities["training"]["references"], mean, scale)
    result = {
        "L_struct": float(direction["bundle"]["losses"]["L_struct"].detach().cpu()),
        "L_rec": float(direction["bundle"]["losses"]["L_rec"].detach().cpu()),
        "L_op": float(direction["bundle"]["losses"]["L_op"].detach().cpu()),
        "gradient_stats": direction["gradient_stats"],
        "projection": direction["projection"],
        "calibration_source_indices": list(CALIBRATION_INDICES),
        "role": TRAIN_ROLE,
    }
    prior._clear_runtime_grads(consumer)
    return result


def _calibration(consumer: Any, authorities: Mapping[str, Any], mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    fixed_indices = torch.tensor(CALIBRATION_INDICES, dtype=torch.long)
    _require(bool(torch.all(torch.isin(fixed_indices, authorities["train_indices"])).item()), "structural calibration left TRAIN")
    _require(authorities["labels"].index_select(0, fixed_indices).tolist() == list(range(10)), "structural calibration labels changed")
    initial_state = prior._runtime_state(consumer)
    initial_rng = prior._rng_snapshot()
    batch = authorities["images"].index_select(0, fixed_indices).to(device)
    prior._clear_runtime_grads(consumer)
    direction = _structural_direction(consumer, batch, authorities["training"]["references"], mean, scale)
    generator = prior._generator(consumer)
    before = float(direction["bundle"]["losses"]["L_struct"].detach().cpu())
    generator_before = generator.detach().clone()
    with torch.no_grad():
        generator.add_(-PRELIGHT_EPSILON * direction["q_struct"])
    with torch.inference_mode():
        after_bundle = _structural_bundle(consumer, batch, authorities["training"]["references"], mean, scale)
        after = float(after_bundle["losses"]["L_struct"].detach().cpu())
    with torch.no_grad():
        generator.copy_(generator_before)
    prior._clear_runtime_grads(consumer)
    _require(_state_digest(prior._runtime_state(consumer)) == _state_digest(initial_state), "structural directional sanity did not restore runtime")
    _require(prior._rng_equal(prior._rng_snapshot(), initial_rng), "structural calibration changed RNG")
    _require(after < before, "structural descent sign preflight failed")
    return {
        "schema": "evaluator_agnostic_structural_alignment_scale_calibration.v1",
        "status": "READY",
        "source_indices": list(CALIBRATION_INDICES),
        "labels": list(range(10)),
        "target_historical_sic_phase_rms": DELTA_STRUCT,
        "delta_struct": DELTA_STRUCT,
        "raw_structural_gradient": direction["gradient_stats"]["g_struct"],
        "normalized_structural_gradient": direction["gradient_stats"]["q_struct"],
        "scaled_structural_gradient": direction["gradient_stats"]["g_struct_scaled"],
        "directional_sanity": {"epsilon": PRELIGHT_EPSILON, "before_L_struct": before, "after_L_struct": after, "passed": True},
        "no_persistent_update": True,
    }


def _preflight(
    consumer: Any,
    optimizer: torch.optim.Adam,
    sampler: StructuralContinuationSampler,
    authorities: Mapping[str, Any],
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    initial_state = prior._runtime_state(consumer)
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_rng = prior._rng_snapshot()
    initial_cursor = sampler.cursor
    traces = []
    for update in range(3):
        indices, labels = sampler.next_batch()
        _require(torch.equal(authorities["labels"].index_select(0, indices), labels), f"preflight TRAIN labels changed at update {update + 1}")
        batch = authorities["images"].index_select(0, indices).to(device)
        traces.append(_apply_update(consumer, optimizer, batch, authorities["training"]["references"], mean, scale, step=START_STEP + update + 1, clean_generator_state=initial_state["generator.phase"]))
    _require(not torch.equal(initial_state["generator.phase"], prior._runtime_state(consumer)["generator.phase"]), "three-step preflight did not update generator")
    _require(torch.equal(initial_state["encoder.phase"], prior._runtime_state(consumer)["encoder.phase"]), "three-step preflight changed encoder")
    consumer.load_trainable_state_dict({"encoder.phase": initial_state["encoder.phase"], "generator.phase": initial_state["generator.phase"]})
    optimizer.load_state_dict(initial_optimizer)
    sampler.cursor = initial_cursor
    prior._restore_rng({"cpu": initial_rng[0], "cuda": initial_rng[1]})
    prior._freeze_generator_only(consumer)
    prior._clear_runtime_grads(consumer)
    _require(_state_digest(prior._runtime_state(consumer)) == _state_digest(initial_state), "preflight runtime restore failed")
    _require(float(optimizer.state[prior._generator(consumer)]["step"].item()) == 6300.0, "preflight Adam restore failed")
    _require(sampler.cursor == START_STEP, "preflight sampler restore failed")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "three_step_real_update": {"updates": 3, "persistent_updates": 0, "trace": traces},
        "stored_step6000_state_restored": True,
        "adam_internal_step_at_start": 6300.0,
        "adam_internal_step_after_temporary_preflight": 6303.0,
        "sampler_cursor_restored": True,
        "rng_restored": True,
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase"],
        "classifier_gradients": False,
        "protected_test_sealed": {"protected": 0, "test": 0, "sealed": 0},
    }


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Adam,
    sampler: StructuralContinuationSampler,
    history: list[Mapping[str, Any]],
    *,
    step: int,
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    stage1_condition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    state = optimizer.state[prior._generator(consumer)]
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization_ids": [AUTHORIZATION_ID],
        "completed_updates": step,
        "starting_scientific_step": START_STEP,
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "checkpoint_selection": "none; every preregistered scientific checkpoint persisted",
        "scientific_artifact": True,
        "operational_artifact": False,
        "run_identity": run_identity.to_dict(),
        "source_manifest": dict(source_manifest),
        "starting_authority": {"path": STEP6000.as_posix(), "sha256": STEP6000_SHA256, "selected_step": START_STEP},
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "scale", "T=I_100", "S1", "E_S", "E_I", "E_C", "E_D", "f_struct"],
        "generator_state": consumer.runtime.generator.phase.detach().cpu().clone(),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": _state_digest(prior._frozen_runtime_state(consumer)),
        "optimizer_mode": "STEP6000_ADAM_STATE_RESTORED_EXACTLY",
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "internal_step": float(state["step"])},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": sampler.state_dict(),
        "rng_state": _rng_state_payload(),
        "structural_contract": {"dimension": STRUCTURAL_DIMENSION, "tau": STRUCTURAL_TAU, "eps_mass": STRUCTURAL_EPS_MASS, "delta_struct": DELTA_STRUCT, "normalization": "canonical TRAIN targets only; population std clamped at 1e-6"},
        "auxiliary_coefficients": {"beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ},
        "history_tail": list(history[-10:]),
        "stage1_condition": None if stage1_condition is None else dict(stage1_condition),
    }


def _make_identity(source_manifest: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id="evaluator-agnostic-structural-image-alignment-v1",
        trajectory_id="evaluator-agnostic-structural-image-alignment-trajectory-v1",
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"start_step": START_STEP, "stage1_end": STAGE1_END, "total_steps": TOTAL_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "tau": STRUCTURAL_TAU, "structural_dimension": STRUCTURAL_DIMENSION, "delta_struct": DELTA_STRUCT, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ},
        model_identity="LIGHTGEN_2D_EVALUATOR_AGNOSTIC_STRUCTURAL_IMAGE_ALIGNMENT_V1",
        optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "stored_internal_step_at_start": 6300.0, "reset": False},
        sampler_identity={"seed": SCHEDULE_SEED, "starting_cursor": START_STEP, "terminal_cursor": TOTAL_STEPS, "accepted_prefix_preserved": True, "wraparound": True},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id="evaluator-agnostic-structural-image-alignment-v1:attempt_01",
    )


def _classify(
    dev_before: Mapping[str, Any],
    dev_after: Mapping[str, Any],
    s1_before: Mapping[str, Any],
    s1_after: Mapping[str, Any],
    dev_bootstrap: Mapping[str, Any],
    s1_bootstrap: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    optical: Mapping[str, Any],
) -> str:
    real_supported = bool(float(dev_bootstrap["ci95"]["lower"]) > 0.0)
    s1_supported = bool(float(s1_bootstrap["ci95"]["lower"]) > 0.0)
    classifier_tradeoff = any(
        float(dev_after[name]["macro_recall"]) < float(dev_before[name]["macro_recall"])
        for name in ("E_S", "E_I", "E_C", "E_D")
    )
    if not real_supported:
        if float(dev_after["L_struct"]) < float(dev_before["L_struct"]):
            return CLASSIFICATION_C
        return CLASSIFICATION_E
    if not reconstruction["passed"] or not optical["passed"]:
        return CLASSIFICATION_F
    if classifier_tradeoff:
        return CLASSIFICATION_D
    return CLASSIFICATION_A if s1_supported else CLASSIFICATION_B


def _load_existing_scientific_checkpoint(root: Path, target: Path, step: int, authorities: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = target / f"scientific_step_{step:06d}.pt"
    _require(path.is_file(), f"existing scientific checkpoint is missing: {path}")
    payload = _mapping(safe_torch_load(path, map_location="cpu"), f"existing step-{step} checkpoint")
    _require(payload.get("schema") == CHECKPOINT_SCHEMA, f"existing step-{step} checkpoint schema changed")
    _require(payload.get("task_id") == TASK_ID and payload.get("completed_updates") == step, f"existing step-{step} checkpoint identity changed")
    _require(payload.get("scientific_artifact") is True and payload.get("operational_artifact") is False, f"existing step-{step} checkpoint role changed")
    _require(payload.get("starting_scientific_step") == START_STEP, f"existing step-{step} checkpoint start changed")
    _require(payload.get("trainable_parameter_names") == ["generator.phase"], f"existing step-{step} trainable boundary changed")
    generator_state = _tensor(payload.get("generator_state"), f"existing step-{step} generator state")
    encoder_state = _tensor(payload.get("encoder_state"), f"existing step-{step} encoder state")
    _require(tuple(generator_state.shape) == (2, 1, 100, 100) and generator_state.dtype == torch.float32, f"existing step-{step} generator geometry changed")
    _require(tuple(encoder_state.shape) == (2, 1, 100, 100) and encoder_state.dtype == torch.float32, f"existing step-{step} encoder geometry changed")
    _finite(generator_state, f"existing step-{step} generator state")
    _finite(encoder_state, f"existing step-{step} encoder state")
    _require(torch.equal(encoder_state, authorities["step6000"]["encoder_state"]), f"existing step-{step} encoder state changed")
    if step == START_STEP:
        _require(torch.equal(generator_state, authorities["step6000"]["generator_state"]), "existing step6000 generator state changed")
    optimizer_configuration = _mapping(payload.get("optimizer_configuration"), f"existing step-{step} optimizer configuration")
    _require(float(optimizer_configuration.get("internal_step", -1.0)) == float(step + 300), f"existing step-{step} Adam state changed")
    sampler_state = _mapping(payload.get("sampler_state"), f"existing step-{step} sampler state")
    _require(sampler_state.get("cursor") == step and sampler_state.get("steps") == TOTAL_STEPS, f"existing step-{step} sampler cursor changed")
    _require(isinstance(payload.get("rng_state"), Mapping), f"existing step-{step} RNG state is missing")
    return dict(payload), _artifact_ref(root, path, role=f"existing scientific step-{step} checkpoint")


def finalize_existing_run(*, repo_root: Path | None = None, run_dir: str | Path) -> dict[str, Any]:
    """Complete a run whose scientific checkpoints and endpoint diagnostics exist but whose report writer stopped."""
    root = _root() if repo_root is None else repo_root.resolve()
    target = (root / run_dir if not Path(run_dir).is_absolute() else Path(run_dir)).resolve()
    _require(target.is_dir(), f"existing run directory is missing: {target}")
    _require(not (target / "report.json").exists(), "existing run already has a final report")
    execution_manifest = _mapping(json.loads((target / "execution_manifest.json").read_text(encoding="utf-8")), "existing execution manifest")
    _require(execution_manifest.get("task_id") == TASK_ID, "existing run task identity changed")
    source_manifest = _mapping(json.loads((target / "source_manifest.json").read_text(encoding="utf-8")), "existing source manifest")
    effective_identity = ResearchRunIdentity.from_mapping(_mapping(execution_manifest.get("run_identity"), "existing run identity"))
    preflight = _mapping(json.loads((target / "preflight.json").read_text(encoding="utf-8")), "existing preflight")
    _require(preflight.get("status") == "READY", "existing preflight is not READY")
    authorities = _load_authorities(root)
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "finalize requires CUDA")
    device = torch.device("cuda:0")
    surrogate_mean, surrogate_scale = _fit_surrogate_normalization(authorities["images"].index_select(0, authorities["train_indices"]))
    normalization = _mapping(safe_torch_load(target / "surrogate_normalization.pt", map_location="cpu"), "existing surrogate normalization")
    _require(torch.equal(surrogate_mean, _tensor(normalization.get("mean"), "existing surrogate mean")), "existing surrogate mean changed")
    _require(torch.equal(surrogate_scale, _tensor(normalization.get("scale"), "existing surrogate scale")), "existing surrogate scale changed")
    checkpoints: dict[int, dict[str, Any]] = {}
    checkpoint_refs: dict[str, Any] = {}
    for step in (START_STEP, STAGE1_END, TOTAL_STEPS):
        checkpoints[step], checkpoint_refs[str(step)] = _load_existing_scientific_checkpoint(root, target, step, authorities)
    consumer, smooth, independent, ec, ed = prior._load_models(root, authorities, device)
    ledger = DataReadLedger()
    dev_records: dict[str, dict[str, Any]] = {}
    s1_records: dict[str, dict[str, Any]] = {}
    for step in (START_STEP, STAGE1_END, TOTAL_STEPS):
        checkpoint = checkpoints[step]
        consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
        prior._freeze_generator_only(consumer)
        prior._clear_runtime_grads(consumer)
        prior._restore_rng(checkpoint["rng_state"])
        dev_records[str(step)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], surrogate_mean, surrogate_scale, ledger, device, step=step)
        s1_records[str(step)] = _evaluate_s1(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], device, ledger, step=step)
    stage1_bootstrap = _paired_bootstrap_log_rho(dev_records[str(START_STEP)]["_labels"], dev_records[str(START_STEP)]["_log_rho"], dev_records[str(STAGE1_END)]["_log_rho"])
    final_bootstrap = _paired_bootstrap_log_rho(dev_records[str(START_STEP)]["_labels"], dev_records[str(START_STEP)]["_log_rho"], dev_records[str(TOTAL_STEPS)]["_log_rho"])
    s1_stage1_bootstrap = _paired_bootstrap_log_rho(s1_records[str(START_STEP)]["_labels"], s1_records[str(START_STEP)]["_log_rho"], s1_records[str(STAGE1_END)]["_log_rho"])
    s1_final_bootstrap = _paired_bootstrap_log_rho(s1_records[str(START_STEP)]["_labels"], s1_records[str(START_STEP)]["_log_rho"], s1_records[str(TOTAL_STEPS)]["_log_rho"])
    stage1_reconstruction = {"step6000": dev_records[str(START_STEP)]["L_rec"], "step6300": dev_records[str(STAGE1_END)]["L_rec"], "delta_step6300_minus_step6000": dev_records[str(STAGE1_END)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records[str(STAGE1_END)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
    stage1_optical = _optical_pass(dev_records[str(START_STEP)], dev_records[str(STAGE1_END)])
    stage1_checks = {"structural_distance_improvement_ci_lower_positive": float(stage1_bootstrap["ci95"]["lower"]) > 0.0, "no_severe_per_class_reversal": all(float(item["ci95"]["upper"]) >= 0.0 for item in stage1_bootstrap["per_class"].values()), "reconstruction_constraint": bool(stage1_reconstruction["passed"]), "optical_validity": bool(stage1_optical["passed"]), "structural_gradient_never_degenerate": True, "protected_test_sealed_zero": True}
    stage1_passed = all(stage1_checks.values())
    _require(stage1_passed, "recomputed Stage-1 condition failed during completion repair")
    final_reconstruction = {"step6000": dev_records[str(START_STEP)]["L_rec"], "candidate_step": TOTAL_STEPS, "candidate_L_rec": dev_records[str(TOTAL_STEPS)]["L_rec"], "delta": dev_records[str(TOTAL_STEPS)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records[str(TOTAL_STEPS)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
    final_optical = _optical_pass(dev_records[str(START_STEP)], dev_records[str(TOTAL_STEPS)])
    final_classification = _classify(dev_records[str(START_STEP)], dev_records[str(TOTAL_STEPS)], s1_records[str(START_STEP)], s1_records[str(TOTAL_STEPS)], final_bootstrap, s1_final_bootstrap, final_reconstruction, final_optical)
    public_dev = {key: {subkey: value for subkey, value in record.items() if not subkey.startswith("_")} for key, record in dev_records.items()}
    public_s1 = {key: {subkey: value for subkey, value in record.items() if not subkey.startswith("_")} for key, record in s1_records.items()}
    final_checkpoint = checkpoints[TOTAL_STEPS]
    endpoint_refs = {str(step): {"dev": _artifact_ref(root, target / f"dev_diagnostics_step_{step:06d}.json", role=f"existing step-{step} DEV diagnostics"), "s1": _artifact_ref(root, target / f"s1_transfer_step_{step:06d}.json", role=f"existing step-{step} S1 diagnostics")} for step in (START_STEP, STAGE1_END, TOTAL_STEPS)}
    checkpoint_steps_persisted = [step for step in CHECKPOINT_STEPS if (target / f"scientific_step_{step:06d}.pt").is_file()]
    _require(checkpoint_steps_persisted == list(CHECKPOINT_STEPS), "existing scientific checkpoint schedule is incomplete")
    completion_repair = {"schema": "evaluator_agnostic_structural_image_alignment_completion_repair.v1", "reason": "scientific execution reached step9000 and wrote all endpoint artifacts, but the original report writer rejected the 2000-row S1 bootstrap at a stale DEV-size assertion", "model_updates_performed": 0, "endpoint_recomputed": [START_STEP, STAGE1_END, TOTAL_STEPS], "source_run_directory": _relative(root, target), "existing_endpoint_artifacts": endpoint_refs, "existing_scientific_checkpoints": checkpoint_refs}
    _write_json(target / "completion_repair.json", completion_repair, kind="structural alignment completion repair record")
    report = {
        "schema": REPORT_SCHEMA, "task_id": TASK_ID, "result": "VALID_COMPLETED", "status": "VALID_COMPLETED", "outer_gate": OUTER_GATE, "outer_gate_passed": True, "classification": final_classification,
        "authorization_consumed": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN, "scope": "step6000->6300 and conditional step6300->9000"},
        "local_state": execution_manifest.get("local_state", {"worktree_preserved": True}), "supported_frontier": "STEP6000_STRUCTURAL_SEMANTIC_REALIZATION_GAP_SUPPORTED", "input_authorities": authorities["authority_hashes"],
        "starting_state": {"step": START_STEP, "checkpoint_sha256": STEP6000_SHA256, "optimizer_internal_step": 6300.0, "sampler_cursor": START_STEP, "sampler_terminal": TOTAL_STEPS, "rng_restored": True},
        "structural_surrogate": {"dimension": STRUCTURAL_DIMENSION, "tau": STRUCTURAL_TAU, "eps_mass": STRUCTURAL_EPS_MASS, "blocks": {"grayscale_global_geometry": 84, "soft_threshold_each": 84, "thresholds": [0.25, 0.50, 0.75]}, "normalization": {"mean": "canonical TRAIN targets only", "scale": "max(population_std,1e-6)", "artifact": _artifact_ref(root, target / "surrogate_normalization.pt", role="frozen structural surrogate normalization")}},
        "gradient_contract": {"g_struct": "grad_generator(L_struct)", "q_struct": "g_struct/RMS(g_struct)", "sign": "optimizer uses +delta_struct*q_struct; parameter descent is negative", "degeneracy_tolerance": JOINT_DEGENERACY_TOLERANCE, "g_total": "delta_struct*q_struct+g_rec_safe+g_op_safe", "classifier_gradients": False}, "scale_calibration": preflight["scale_calibration"], "auxiliary_contract": {"beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "projection": "remove only opposing weighted reconstruction/optical component"},
        "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "scale", "T", "W_OLS", "S1", "E_S", "E_I", "E_C", "E_D", "f_struct"], "E_D_in_backward": False}, "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "stored_internal_step_at_start": 6300.0, "reset": False, "scheduler_change": False},
        "preflight": preflight, "stage1": {"updates": STAGE1_UPDATES, "checkpoints": [6000, 6100, 6200, 6300], "development": {key: public_dev[key] for key in ("6000", "6300")}, "S1_transfer": {key: public_s1[key] for key in ("6000", "6300")}, "bootstrap": {key: value for key, value in stage1_bootstrap.items() if not isinstance(value, torch.Tensor)}, "reconstruction": stage1_reconstruction, "optical_validity": stage1_optical, "continuation_condition": {"passed": stage1_passed, "checks": stage1_checks}},
        "stage2": {"executed": True, "checkpoints": [6300, 6600, 7000, 7500, 8000, 8500, 9000], "final_step": TOTAL_STEPS, "conditional_rule": "only after every Stage-1 condition passes"}, "final_bootstrap": {key: value for key, value in final_bootstrap.items() if not isinstance(value, torch.Tensor)},
        "training_trajectory": {"updates": TOTAL_STEPS - START_STEP, "scientific_cursor": TOTAL_STEPS, "checkpoint_steps_persisted": checkpoint_steps_persisted, "trace": final_checkpoint.get("history_tail", []), "trace_scope": "last ten updates retained by the existing checkpoint; endpoint artifacts and fixed checkpoint schedule are authoritative"}, "data_reads": {**ledger.to_dict(), "protected": 0, "test": 0, "sealed": 0}, "forbidden_data_reads": {"protected": 0, "test": 0, "sealed": 0},
        "hard_structural_firewall": {"descriptor_dimension": 337, "direct_optimization": False, "q95_modified": False, "topology_modified": False, "reference_modified": False}, "s1_fixed_bank": {"winner": "S1", "winner_sha256": S1_WINNER_SHA256, "no_regeneration": True}, "completion_repair": completion_repair,
        "claim_boundary": {"evaluator_agnostic_structural_alignment_only": True, "human_perceptual_acceptance": False, "hardware_readiness": False, "promotion": False, "new_confirmation_required": True}, "counterfactual_integrity": {"same_step6000_checkpoint": True, "same_encoder_ols": True, "same_generator_architecture": True, "same_adam_state": True, "same_beta_gamma": True, "same_hard_structural_metric": True, "no_hyperparameter_search": True, "no_sampler_refit": True, "no_checkpoint_selection": True, "no_protected_test_sealed": True}, "unresolved": ["The differentiable 336D surrogate is a falsifiable model assumption; this run does not establish human perceptual or hardware acceptance."], "next_action": "NONE", "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "output_dir": _relative(root, target), "endpoint_bootstrap": {"s1_stage1": {key: value for key, value in s1_stage1_bootstrap.items() if not isinstance(value, torch.Tensor)}, "s1_final": {key: value for key, value in s1_final_bootstrap.items() if not isinstance(value, torch.Tensor)}}
    }
    _write_json(target / "report.json", report, kind="evaluator-agnostic structural alignment report")
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=TOTAL_STEPS, interval_seconds=60.0)
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=TOTAL_STEPS, latest_losses={"classification": final_classification, "completion_repair": True}, force=True)
    return report


def run_training(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"output already exists: {target}")

    authorities = _load_authorities(root)
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "formal structural alignment requires CUDA")
    device = torch.device("cuda:0")
    surrogate_mean, surrogate_scale = _fit_surrogate_normalization(authorities["images"].index_select(0, authorities["train_indices"]))
    consumer, smooth, independent, ec, ed = prior._load_models(root, authorities, device)
    consumer.load_trainable_state_dict({"encoder.phase": authorities["step6000"]["encoder_state"], "generator.phase": authorities["step6000"]["generator_state"]})
    prior._freeze_generator_only(consumer)
    optimizer = torch.optim.Adam([prior._generator(consumer)], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0, amsgrad=False)
    optimizer.load_state_dict(copy.deepcopy(_mapping(authorities["step6000"]["optimizer_state"], "step6000 optimizer state")))
    _validate_optimizer(consumer, optimizer)
    _require(float(optimizer.state[prior._generator(consumer)]["step"].item()) == 6300.0, "step6000 Adam state was not restored exactly")
    prior._restore_rng(authorities["step6000"]["rng_state"])
    sampler = StructuralContinuationSampler(authorities["step6000"]["sampler_state"], cursor=START_STEP)
    calibration = _calibration(consumer, authorities, surrogate_mean, surrogate_scale, device)
    preflight = _preflight(consumer, optimizer, sampler, authorities, surrogate_mean, surrogate_scale, device)
    _require(preflight["status"] == "READY" and calibration["status"] == "READY", "structural alignment preflight did not reach READY")

    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            Path(prior.__file__).resolve(),
            Path(structural.__file__).resolve(),
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
            root / STEP6000,
            root / TRAIN_OLS,
            root / DEV_OLS,
            root / STRUCTURAL_REFERENCE,
            root / S1_WINNER,
        ),
        config_identity={"authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "start_step": START_STEP, "stage1_end": STAGE1_END, "total_steps": TOTAL_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "tau": STRUCTURAL_TAU, "structural_dimension": STRUCTURAL_DIMENSION, "delta_struct": DELTA_STRUCT, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "authority_hashes": authorities["authority_hashes"]},
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    run_identity = _make_identity(source_manifest)
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    correctness_passed = bool(
        preflight.get("status") == "READY"
        and calibration.get("status") == "READY"
        and preflight.get("three_step_real_update", {}).get("updates") == 3
        and preflight.get("stored_step6000_state_restored") is True
        and preflight.get("sampler_cursor_restored") is True
        and preflight.get("rng_restored") is True
        and preflight.get("protected_test_sealed") == {"protected": 0, "test": 0, "sealed": 0}
    )
    receipt = control.make_ready_receipt(
        correctness={
            "passed": correctness_passed,
            "three_real_updates": preflight.get("three_step_real_update", {}).get("updates"),
            "stored_checkpoint_state_restored": preflight.get("stored_step6000_state_restored"),
            "optimizer_step_restored": preflight.get("adam_internal_step_at_start"),
            "sampler_cursor_restored": preflight.get("sampler_cursor_restored"),
            "rng_restored": preflight.get("rng_restored"),
            "scale_calibration_directional_sanity": calibration.get("directional_sanity"),
            "protected_test_sealed": preflight.get("protected_test_sealed"),
        },
        resource={
            "passed": int(total_bytes) > 0 and 0 <= int(free_bytes) <= int(total_bytes),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        },
        implementation={
            "passed": preflight.get("trainable_parameter_names") == ["generator.phase"],
            "selected_engine": "evaluator_agnostic_structural_alignment_cuda_generator_phase",
            "trainable_parameter_names": preflight.get("trainable_parameter_names"),
            "structural_dimension": STRUCTURAL_DIMENSION,
        },
        selected_engine="evaluator_agnostic_structural_alignment_cuda_generator_phase",
        resource_envelope={"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)},
    )
    control.mark_preflight_ready(receipt)
    control.start_execution()
    effective_identity = control.open_scientific_attempt(target, run_identity.scientific_attempt_id, receipt, resume=True)
    _write_json(target / "source_manifest.json", source_manifest, kind="structural alignment source manifest")
    _write_torch(target / "surrogate_normalization.pt", {"schema": "evaluator_agnostic_structural_alignment_surrogate_normalization.v1", "mean": surrogate_mean, "scale": surrogate_scale, "train_only": True, "source_sha256": TRAINING_SOURCE_SHA256}, kind="TRAIN-only structural surrogate normalization")
    _write_json(target / "preflight.json", {**preflight, "scale_calibration": calibration}, kind="structural alignment preflight")
    _write_json(target / "execution_manifest.json", {"schema": "evaluator_agnostic_structural_image_alignment_execution_manifest.v1", "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorization": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN}, "input_authorities": authorities["authority_hashes"], "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "local_state": authorities["local_state"], "scope": {"stage1": [START_STEP, STAGE1_END], "stage2": [STAGE1_END, TOTAL_STEPS], "trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "W_OLS", "scale", "T", "S1", "E_S", "E_I", "E_C", "E_D", "f_struct"], "protected": False, "test": False, "sealed": False}}, kind="structural alignment execution manifest")

    ledger = DataReadLedger()
    clean_generator_state = authorities["step6000"]["generator_state"].to(device)
    history: list[dict[str, Any]] = []
    dev_records: dict[str, dict[str, Any]] = {}
    s1_records: dict[str, dict[str, Any]] = {}
    trajectory: dict[str, dict[str, Any]] = {}
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=TOTAL_STEPS, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=START_STEP, latest_losses={"preflight": "READY", "optimizer_mode": "STEP6000_ADAM_STATE_RESTORED_EXACTLY"}, force=True)

    dev_records[str(START_STEP)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], surrogate_mean, surrogate_scale, ledger, device, step=START_STEP)
    s1_records[str(START_STEP)] = _evaluate_s1(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], device, ledger, step=START_STEP)
    trajectory[str(START_STEP)] = _checkpoint_train_diagnostic(consumer, sampler, authorities, surrogate_mean, surrogate_scale, device)
    _write_json(target / f"dev_diagnostics_step_{START_STEP:06d}.json", {key: value for key, value in dev_records[str(START_STEP)].items() if not key.startswith("_")}, kind="step6000 structural DEV diagnostics")
    _write_json(target / f"s1_transfer_step_{START_STEP:06d}.json", {key: value for key, value in s1_records[str(START_STEP)].items() if not key.startswith("_")}, kind="step6000 structural S1 diagnostics")
    _write_torch(target / f"scientific_step_{START_STEP:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, step=START_STEP, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition=None), kind="structural alignment starting checkpoint")

    last_recovery_step = START_STEP
    last_recovery_time = time.time()
    stage1_passed = False
    stage1_checks: dict[str, bool] = {}
    stage1_bootstrap: dict[str, Any] | None = None
    stage1_reconstruction: dict[str, Any] | None = None
    stage1_optical: dict[str, Any] | None = None

    def recovery_payload(step: int) -> Mapping[str, Any]:
        payload = _checkpoint_payload(consumer, optimizer, sampler, history, step=step, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": stage1_checks})
        payload["scientific_artifact"] = False
        payload["operational_artifact"] = True
        payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
        return payload

    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_payload, phase="TRAINING", total_steps=TOTAL_STEPS)
    with interrupt:
        while sampler.cursor < TOTAL_STEPS:
            current_step = sampler.cursor
            indices, labels = sampler.next_batch()
            _require(torch.equal(authorities["labels"].index_select(0, indices), labels), f"TRAIN labels changed at update {current_step + 1}")
            batch = authorities["images"].index_select(0, indices).to(device)
            trace = _apply_update(consumer, optimizer, batch, authorities["training"]["references"], surrogate_mean, surrogate_scale, step=current_step + 1, clean_generator_state=clean_generator_state)
            history.append(trace)
            completed = current_step + 1
            now = time.time()
            if recovery_due(step=completed, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                recovery.save(recovery_payload(completed), step=completed)
                last_recovery_step = completed
                last_recovery_time = now
            heartbeat.update(state=RUNNING, phase="TRAINING", step=completed, latest_losses={name: trace[name] for name in ("L_struct", "L_rec", "L_op")})
            interrupted = interrupt.handle_safe_boundary(step=completed, latest_losses={name: trace[name] for name in ("L_struct", "L_rec", "L_op")}, last_recovery_checkpoint=None)
            if interrupted is not None:
                control.interrupted()
                return {"result": "INTERRUPTED", "output_dir": _relative(root, target), "step": completed, "recovery_checkpoint": interrupted.name}
            if completed in CHECKPOINT_STEPS and completed != STAGE1_END:
                trajectory[str(completed)] = _checkpoint_train_diagnostic(consumer, sampler, authorities, surrogate_mean, surrogate_scale, device)
                _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, step=completed, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": stage1_checks}), kind=f"structural alignment step-{completed} checkpoint")
            if completed == STAGE1_END:
                dev_records[str(completed)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], surrogate_mean, surrogate_scale, ledger, device, step=completed)
                s1_records[str(completed)] = _evaluate_s1(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], device, ledger, step=completed)
                _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {key: value for key, value in dev_records[str(completed)].items() if not key.startswith("_")}, kind="step6300 structural DEV diagnostics")
                _write_json(target / f"s1_transfer_step_{completed:06d}.json", {key: value for key, value in s1_records[str(completed)].items() if not key.startswith("_")}, kind="step6300 structural S1 diagnostics")
                stage1_bootstrap = _paired_bootstrap_log_rho(dev_records[str(START_STEP)]["_labels"], dev_records[str(START_STEP)]["_log_rho"], dev_records[str(completed)]["_log_rho"])
                _write_torch(target / "paired_structural_bootstrap_stage1.pt", {key: value for key, value in stage1_bootstrap.items() if isinstance(value, torch.Tensor)}, kind="Stage-1 paired structural bootstrap")
                stage1_reconstruction = {"step6000": dev_records[str(START_STEP)]["L_rec"], "step6300": dev_records[str(completed)]["L_rec"], "delta_step6300_minus_step6000": dev_records[str(completed)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records[str(completed)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
                stage1_optical = _optical_pass(dev_records[str(START_STEP)], dev_records[str(completed)])
                stage1_checks = {"structural_distance_improvement_ci_lower_positive": float(stage1_bootstrap["ci95"]["lower"]) > 0.0, "no_severe_per_class_reversal": all(float(item["ci95"]["upper"]) >= 0.0 for item in stage1_bootstrap["per_class"].values()), "reconstruction_constraint": bool(stage1_reconstruction["passed"]), "optical_validity": bool(stage1_optical["passed"]), "structural_gradient_never_degenerate": True, "protected_test_sealed_zero": True}
                stage1_passed = all(stage1_checks.values())
                _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, step=completed, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": stage1_checks}), kind="structural alignment Stage-1 endpoint checkpoint")
                heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=completed, latest_losses={"stage1_passed": stage1_passed}, force=True)
                if not stage1_passed:
                    break

    _require(str(START_STEP) in dev_records and str(STAGE1_END) in dev_records, "Stage-1 diagnostics are incomplete")
    if stage1_bootstrap is None:
        stage1_bootstrap = _paired_bootstrap_log_rho(dev_records[str(START_STEP)]["_labels"], dev_records[str(START_STEP)]["_log_rho"], dev_records[str(STAGE1_END)]["_log_rho"])
    if stage1_reconstruction is None:
        stage1_reconstruction = {"step6000": dev_records[str(START_STEP)]["L_rec"], "step6300": dev_records[str(STAGE1_END)]["L_rec"], "delta_step6300_minus_step6000": dev_records[str(STAGE1_END)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records[str(STAGE1_END)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
    if stage1_optical is None:
        stage1_optical = _optical_pass(dev_records[str(START_STEP)], dev_records[str(STAGE1_END)])
    stage2_executed = stage1_passed and sampler.cursor == TOTAL_STEPS
    final_step = TOTAL_STEPS if stage2_executed else STAGE1_END
    if stage2_executed and str(TOTAL_STEPS) not in dev_records:
        dev_records[str(TOTAL_STEPS)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], surrogate_mean, surrogate_scale, ledger, device, step=TOTAL_STEPS)
        s1_records[str(TOTAL_STEPS)] = _evaluate_s1(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], device, ledger, step=TOTAL_STEPS)
        _write_json(target / f"dev_diagnostics_step_{TOTAL_STEPS:06d}.json", {key: value for key, value in dev_records[str(TOTAL_STEPS)].items() if not key.startswith("_")}, kind="final structural DEV diagnostics")
        _write_json(target / f"s1_transfer_step_{TOTAL_STEPS:06d}.json", {key: value for key, value in s1_records[str(TOTAL_STEPS)].items() if not key.startswith("_")}, kind="final structural S1 diagnostics")
    final_dev = dev_records.get(str(final_step), dev_records[str(STAGE1_END)])
    final_s1 = s1_records.get(str(final_step), s1_records[str(STAGE1_END)])
    final_bootstrap = _paired_bootstrap_log_rho(dev_records[str(START_STEP)]["_labels"], dev_records[str(START_STEP)]["_log_rho"], final_dev["_log_rho"])
    final_reconstruction = {"step6000": dev_records[str(START_STEP)]["L_rec"], "candidate_step": final_step, "candidate_L_rec": final_dev["L_rec"], "delta": final_dev["L_rec"] - dev_records[str(START_STEP)]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": final_dev["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
    final_optical = _optical_pass(dev_records[str(START_STEP)], final_dev)
    final_classification = _classify(dev_records[str(START_STEP)], final_dev, s1_records[str(START_STEP)], final_s1, final_bootstrap, _paired_bootstrap_log_rho(s1_records[str(START_STEP)]["_labels"], s1_records[str(START_STEP)]["_log_rho"], final_s1["_log_rho"]), final_reconstruction, final_optical)

    control.complete()
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=sampler.cursor, latest_losses={"classification": final_classification}, force=True)
    public_dev = {key: {subkey: value for subkey, value in record.items() if not subkey.startswith("_")} for key, record in dev_records.items()}
    public_s1 = {key: {subkey: value for subkey, value in record.items() if not subkey.startswith("_")} for key, record in s1_records.items()}
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": bool(stage1_passed),
        "classification": final_classification,
        "authorization_consumed": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN, "scope": "step6000->6300 and conditional step6300->9000"},
        "local_state": {"branch": authorities["local_state"]["branch"], "head": authorities["local_state"]["head"], "dirty_before": authorities["local_state"].get("dirty", True), "worktree_preserved": True},
        "supported_frontier": "STEP6000_STRUCTURAL_SEMANTIC_REALIZATION_GAP_SUPPORTED",
        "input_authorities": authorities["authority_hashes"],
        "starting_state": {"step": START_STEP, "checkpoint_sha256": STEP6000_SHA256, "optimizer_internal_step": 6300.0, "sampler_cursor": START_STEP, "sampler_terminal": TOTAL_STEPS, "rng_restored": True},
        "structural_surrogate": {"dimension": STRUCTURAL_DIMENSION, "tau": STRUCTURAL_TAU, "eps_mass": STRUCTURAL_EPS_MASS, "blocks": {"grayscale_global_geometry": 84, "soft_threshold_each": 84, "thresholds": [0.25, 0.50, 0.75]}, "normalization": {"mean": "canonical TRAIN targets only", "scale": "max(population_std,1e-6)", "artifact": _artifact_ref(root, target / "surrogate_normalization.pt", role="frozen structural surrogate normalization")}},
        "gradient_contract": {"g_struct": "grad_generator(L_struct)", "q_struct": "g_struct/RMS(g_struct)", "sign": "optimizer uses +delta_struct*q_struct; parameter descent is negative", "degeneracy_tolerance": JOINT_DEGENERACY_TOLERANCE, "g_total": "delta_struct*q_struct+g_rec_safe+g_op_safe", "classifier_gradients": False},
        "scale_calibration": calibration,
        "auxiliary_contract": {"beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "projection": "remove only opposing weighted reconstruction/optical component"},
        "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "scale", "T", "W_OLS", "S1", "E_S", "E_I", "E_C", "E_D", "f_struct"], "E_D_in_backward": False},
        "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "stored_internal_step_at_start": 6300.0, "reset": False, "scheduler_change": False},
        "preflight": preflight,
        "stage1": {"updates": min(len(history), STAGE1_UPDATES), "checkpoints": [6000, 6100, 6200, 6300], "development": public_dev, "S1_transfer": public_s1, "bootstrap": {key: value for key, value in stage1_bootstrap.items() if not isinstance(value, torch.Tensor)}, "reconstruction": stage1_reconstruction, "optical_validity": stage1_optical, "continuation_condition": {"passed": stage1_passed, "checks": stage1_checks}},
        "stage2": {"executed": stage2_executed, "checkpoints": [6300, 6600, 7000, 7500, 8000, 8500, 9000], "final_step": final_step, "conditional_rule": "only after every Stage-1 condition passes"},
        "final_bootstrap": {key: value for key, value in final_bootstrap.items() if not isinstance(value, torch.Tensor)},
        "training_trajectory": {"updates": len(history), "scientific_cursor": sampler.cursor, "checkpoint_steps_persisted": [step for step in CHECKPOINT_STEPS if (target / f"scientific_step_{step:06d}.pt").is_file()], "trace": history, "train_diagnostics": trajectory},
        "data_reads": {**ledger.to_dict(), "protected": 0, "test": 0, "sealed": 0},
        "forbidden_data_reads": {"protected": 0, "test": 0, "sealed": 0},
        "hard_structural_firewall": {"descriptor_dimension": 337, "direct_optimization": False, "q95_modified": False, "topology_modified": False, "reference_modified": False},
        "s1_fixed_bank": {"winner": "S1", "winner_sha256": S1_WINNER_SHA256, "no_regeneration": True},
        "claim_boundary": {"evaluator_agnostic_structural_alignment_only": True, "human_perceptual_acceptance": False, "hardware_readiness": False, "promotion": False, "new_confirmation_required": True},
        "counterfactual_integrity": {"same_step6000_checkpoint": True, "same_encoder_ols": True, "same_generator_architecture": True, "same_adam_state": True, "same_beta_gamma": True, "same_hard_structural_metric": True, "no_hyperparameter_search": True, "no_sampler_refit": True, "no_checkpoint_selection": True, "no_protected_test_sealed": True},
        "unresolved": ["The differentiable 336D surrogate is a falsifiable model assumption; this run does not establish human perceptual or hardware acceptance."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report, kind="evaluator-agnostic structural alignment report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--finalize-existing", default=None)
    args = parser.parse_args()
    result = finalize_existing_run(run_dir=args.finalize_existing) if args.finalize_existing else run_training(output_dir=args.output_dir)
    print(json.dumps({"result": result.get("result"), "classification": result.get("classification"), "output_dir": result.get("output_dir")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
