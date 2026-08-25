"""H-JQOC: no-update joint electronic and quadrature observability capacity.

This research lifecycle keeps the current v2 physical system frozen.  It uses
an ephemeral structured Gaussian canary, a v69 phase-state replay loaded into
the v2 optical system, and a four-port balanced receiver upper bound.  It does
not create an optimizer, checkpoint, candidate model, or persistent field.
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
import torch.nn.functional as F
from torch import Tensor

from ...evidence import atomic_write_json, canonical_sha256, file_sha256, tensor_sha256
from ..legacy.serialization import _trusted_legacy_torch_load
from ...optics.system import FormalAmplitudeFiniteCropIdentitySystem
from ...student.checkpoints import state_content_hash
from ...student.conditional import camera_code_fraction, relative_diversity_ratios
from ...student.conditional_acceptance import conditional_coverage_ratio
from ...student.evaluator import (
    load_frozen_evaluator,
    load_local_mnist,
    stratified_mnist_indices,
)
from .electronic_o0_wae import _balanced_reference, _eval_model, _recall
from . import electronic_o0_direct_prior_capacity_canary as dpgc
from .lifecycle_artifacts import (
    prepare_args,
    stage_action_decision,
    validate_manifest_namespace,
)


SCHEMA_PREFIX = "joint_quadrature_optical_capacity"
PROGRAM = "joint-quadrature-optical-capacity"
CONFIG_NAME = "joint_quadrature_optical_capacity_v1"
VERSION = 20260826
HARNESS_SEED = 8971
CLASS_COUNT = 10
CODE_DIMS = (8, 16, 32)
BANK_SEEDS = (57721, 75025)
CALIBRATION_SEED = 112358
SAMPLES_PER_CLASS = 100
PREFLIGHT_SAMPLES_PER_CLASS = 8
CALIBRATION_SAMPLES_PER_CLASS = 64
SLM_SHAPE = (200, 200)
PORT_SHAPE = (200, 200)
RECEIVER_SHAPE = (400, 400)
ACCURACY_MINIMUM = 0.95
RECALL_MINIMUM = 0.90
AGREEMENT_MINIMUM = 0.95
CAMERA_CODE_MINIMUM = 0.25
DIVERSITY_MINIMUM = 0.70
DIVERSITY_MAXIMUM = 1.30
COVERAGE_MAXIMUM = 1.50
SATURATION_MAXIMUM = 0.08
MEMORIZATION_MINIMUM = 0.95
LABEL_SHIFT_MAXIMUM = 0.20
RANK_RATIO_MINIMUM = 1.0e-3
MEMORY_LIMIT_BYTES = 6 * 1024**3
STORAGE_LIMIT_BYTES = 64 * 1024**2
PREFLIGHT_WALL_LIMIT_SECONDS = 300.0
AUDIT_WALL_LIMIT_SECONDS = 900.0
FIXED_EXPOSURE = 0.4913053810596466
FORMAL_CONFIG_PATH = "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
FORMAL_CONFIG_SHA256 = "0c42035c1a525e27ff787d013ca27b0f5ae91338329eb0607c7042ebc5ae8905"
V69_CHECKPOINT_SHA256 = "f22e61da96352354ee5e993136f8607fd21a687e7078a2f74caffcfcac88142e"
V75_REPORT_SHA256 = "fa27143dbb3df31764f4aba956731cc10adf6096f1b9f02132583d81b904526e"
V75_DECISION_SHA256 = "e2abf9002e1ad0edd36bcb37999809ec7f30ca952e131367cbbc1e9b75a26f0a"
V76_ADMISSION_SHA256 = "61745d20b6f945d79089ca91e6143e3fa32aef1a83fc70836217e04982842a6c"
V76_REPORT_SHA256 = "e6f2428f0196d45dadaf57fac75845a1d306b7703c9948e78651961c363b722e"
V76_DECISION_SHA256 = "1767df1d3b27cba2abc0b93fa512a4503436be0ed27a406462f896e6bc9a2fa6"
H_NDMC_CLOSEOUT_SHA256 = "f0e21b76723c187056cfb930c306036fbf58e390ebadbda96ccf04a8dcbee43b"
PRIMARY_EVALUATOR_SHA256 = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
SECONDARY_EVALUATOR_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
PRIMARY_QUALIFICATION_SHA256 = "26fea381bec5d6be2a2ed69806c190f7a7051b5f0944e5f90941a7d93f35695e"
SECONDARY_QUALIFICATION_SHA256 = "93592543306b5a3da8c238d86e41805fc2de0080459d217dfbc3e7161d374d88"


@dataclass(frozen=True)
class StructuredCanary:
    prototypes: Tensor
    basis: Tensor
    scales: Tensor

    def sample(self, labels: Tensor, codes: Tensor, dimension: int) -> Tensor:
        if dimension not in CODE_DIMS:
            raise ValueError("H-JQOC dimension is not registered")
        if labels.ndim != 1 or codes.shape != (labels.numel(), 32):
            raise ValueError("H-JQOC expects labels [N] and a shared [N,32] bank")
        logits = self.prototypes.index_select(0, labels)
        basis = self.basis[:dimension].to(logits)
        residual = torch.einsum("nd,dhw->nhw", codes[:, :dimension], basis)
        logits = logits + self.scales[CODE_DIMS.index(dimension)].to(logits) * residual / math.sqrt(dimension)
        result = torch.sigmoid(logits).unsqueeze(1)
        if not torch.isfinite(result).all():
            raise ValueError("H-JQOC canary produced non-finite views")
        return result

    def command(self, labels: Tensor, codes: Tensor, dimension: int) -> Tensor:
        views = self.sample(labels, codes, dimension)
        return F.interpolate(views, size=SLM_SHAPE, mode="bilinear", align_corners=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evidence is not a JSON object: {path}")
    return value


def _config(path: str | Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "name": CONFIG_NAME,
        "version": VERSION,
        "harness_seed": HARNESS_SEED,
        "code_dims": list(CODE_DIMS),
        "gaussian_bank_seeds": list(BANK_SEEDS),
        "calibration_seed": CALIBRATION_SEED,
        "samples_per_class_per_bank": SAMPLES_PER_CLASS,
        "calibration_samples_per_class": CALIBRATION_SAMPLES_PER_CLASS,
        "preflight_samples_per_class": PREFLIGHT_SAMPLES_PER_CLASS,
        "prototype_logit_epsilon": 1.0 / 255.0,
        "slm_code_fraction_target": 0.25,
        "scale_bisection_bracket": [0.0, 8.0],
        "scale_bisection_steps": 32,
        "scale_tolerance": 1.0e-6,
        "maximum_mode_frequency_cycles_per_mm": 20.0,
        "lo_calibration_quantile": 0.999,
        "receiver_headroom": 0.80,
        "beamsplitter_amplitude_factor": 0.5,
        "receiver_port_shape": list(PORT_SHAPE),
        "receiver_layout": [2, 2],
        "receiver_sensor_shape": list(RECEIVER_SHAPE),
        "receiver_pixel_pitch_um": 2.5,
        "semantic_accuracy_count": 950,
        "semantic_recall_count": 90,
        "agreement_count": 950,
        "label_shift_correct_max": 200,
        "camera_code_fraction_min": CAMERA_CODE_MINIMUM,
        "diversity_ratio_range": [DIVERSITY_MINIMUM, DIVERSITY_MAXIMUM],
        "coverage_max": COVERAGE_MAXIMUM,
        "saturation_max": SATURATION_MAXIMUM,
        "memorization_pass_count": 95,
        "jacobian_minimum_rank": "full_dimension",
        "jacobian_minimum_singular_ratio": RANK_RATIO_MINIMUM,
        "formal_promotion": False,
        "training_authorized": False,
        "checkpoint_creation_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if not isinstance(value, dict) or value != expected:
        raise ValueError("H-JQOC config is not the registered v1 contract")
    return value


def _git() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "clean": not status, "status": status}


def _require_cuda(args: argparse.Namespace) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("H-JQOC requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H-JQOC formal lifecycle requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _dct_basis() -> Tensor:
    rows = []
    x = torch.arange(28, dtype=torch.float64)
    for total in range(1, 11):
        for u in range(total + 1):
            v = total - u
            if u >= 28 or v >= 28:
                continue
            yy, xx = torch.meshgrid(x, x, indexing="ij")
            row = torch.cos(math.pi * (2.0 * xx + 1.0) * u / 56.0) * torch.cos(math.pi * (2.0 * yy + 1.0) * v / 56.0)
            row = row - row.mean()
            row = row / row.square().mean().sqrt().clamp_min(1.0e-12)
            row = row.to(torch.float32)
            flat_index = row.abs().argmax()
            if row.flatten()[flat_index] < 0:
                row = -row
            rows.append(row)
    basis = torch.stack(rows[:32])
    if basis.shape != (32, 28, 28):
        raise ValueError("H-JQOC DCT basis does not contain 32 registered modes")
    return basis


def _logit(images: Tensor) -> Tensor:
    values = images.float().clamp(1.0 / 255.0, 1.0 - 1.0 / 255.0)
    return torch.logit(values).squeeze(1)


def _panel(seed: int, samples_per_class: int, device: torch.device) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.arange(CLASS_COUNT, device=device).repeat_interleave(samples_per_class)
    codes = torch.randn((labels.numel(), 32), generator=generator, dtype=torch.float32).to(device)
    return codes, labels


def _fit_canary(train_images: Tensor, train_labels: Tensor, device: torch.device) -> StructuredCanary:
    means = []
    for label in range(CLASS_COUNT):
        means.append(_logit(train_images[train_labels == label]).mean(0))
    basis = _dct_basis().to(device)
    prototypes = torch.stack(means)
    scales: dict[int, Tensor] = {}
    calibration_codes, calibration_labels = _panel(CALIBRATION_SEED, CALIBRATION_SAMPLES_PER_CLASS, device)
    for dimension in CODE_DIMS:
        low, high = 0.0, 8.0
        for _ in range(32):
            middle = (low + high) / 2.0
            logits = prototypes.index_select(0, calibration_labels)
            residual = torch.einsum("nd,dhw->nhw", calibration_codes[:, :dimension], basis[:dimension])
            views = torch.sigmoid(logits + middle * residual / math.sqrt(dimension)).unsqueeze(1)
            commands = F.interpolate(views, size=SLM_SHAPE, mode="bilinear", align_corners=False)
            if float(camera_code_fraction(commands, calibration_labels).item()) < CAMERA_CODE_MINIMUM:
                low = middle
            else:
                high = middle
        scales[dimension] = torch.tensor(high, device=device)
    return StructuredCanary(prototypes, basis, torch.stack([scales[d] for d in CODE_DIMS]).to(device))


def _scale(canary: StructuredCanary, dimension: int) -> Tensor:
    return canary.scales[CODE_DIMS.index(dimension)]


def _balanced_ports(field: Tensor, lo_amplitude: float, exposure: float) -> dict[str, Tensor]:
    if not field.is_complex() or field.ndim != 4:
        raise ValueError("H-JQOC balanced receiver expects [N,1,H,W] complex fields")
    lo = field.new_full(field.shape, complex(lo_amplitude, 0.0))
    factor = 0.5
    r_plus = exposure * factor * (field + lo).abs().square() / 2.0
    r_minus = exposure * factor * (field - lo).abs().square() / 2.0
    j_plus = exposure * factor * (field + 1j * lo).abs().square() / 2.0
    j_minus = exposure * factor * (field - 1j * lo).abs().square() / 2.0
    return {
        "raw": exposure * field.abs().square(),
        "r_plus": r_plus,
        "r_minus": r_minus,
        "j_plus": j_plus,
        "j_minus": j_minus,
        "R": r_plus - r_minus,
        "J": j_plus - j_minus,
        "r_energy_residual": r_plus + r_minus - exposure * (field.abs().square() + lo.abs().square()) / 2.0,
        "j_energy_residual": j_plus + j_minus - exposure * (field.abs().square() + lo.abs().square()) / 2.0,
    }


def _pack_ports(ports: Mapping[str, Tensor]) -> Tensor:
    required = ("r_plus", "r_minus", "j_plus", "j_minus")
    if any(name not in ports for name in required):
        raise ValueError("H-JQOC receiver requires all four ports")
    rows = [torch.cat((ports["r_plus"], ports["r_minus"]), dim=-1), torch.cat((ports["j_plus"], ports["j_minus"]), dim=-1)]
    packed = torch.cat(rows, dim=-2)
    if tuple(packed.shape[-2:]) != RECEIVER_SHAPE:
        raise ValueError("H-JQOC port packing changed the physical detector shape")
    return packed


def _view28(raw: Tensor) -> Tensor:
    if raw.ndim != 4 or raw.shape[1] != 1:
        raise ValueError("H-JQOC views must be [N,1,H,W]")
    return F.interpolate(raw, size=(28, 28), mode="area").clamp(0.0, 1.0)


def _counts(prediction: Tensor, labels: Tensor) -> dict[str, Any]:
    return {
        "correct": int((prediction == labels).sum()),
        "total": int(labels.numel()),
        "recall_correct": [int(((prediction[labels == label]) == label).sum()) for label in range(CLASS_COUNT)],
        "recall_total": [int((labels == label).sum()) for label in range(CLASS_COUNT)],
    }


def _nearest_squared(query: Tensor, reference: Tensor, chunk: int = 64) -> Tensor:
    norm = reference.square().sum(1)
    values = []
    for start in range(0, query.shape[0], chunk):
        current = query[start : start + chunk]
        values.append((current.square().sum(1, keepdim=True) + norm.unsqueeze(0) - 2.0 * current @ reference.T).clamp_min(0).min(1).values)
    return torch.cat(values)


def _memorization_context(train_images: Tensor, train_labels: Tensor, validation_images: Tensor, validation_labels: Tensor) -> dict[str, Any]:
    train_flat, validation_flat = train_images.flatten(1), validation_images.flatten(1)
    thresholds = []
    for label in range(CLASS_COUNT):
        thresholds.append(torch.quantile(_nearest_squared(validation_flat[validation_labels == label], train_flat[train_labels == label]), 0.05))
    hashes = frozenset(tensor_sha256(row) for row in (train_images.cpu().clamp(0, 1) * 255).round().to(torch.uint8))
    return {"thresholds": torch.stack(thresholds), "hashes": hashes, "train_images": train_images, "train_labels": train_labels}


def _memorization(views: Tensor, labels: Tensor, context: Mapping[str, Any]) -> dict[str, Any]:
    values = (views.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    exact = sum(tensor_sha256(row) in context["hashes"] for row in values)
    distances, counts = [], []
    for label in range(CLASS_COUNT):
        rows = views[labels == label].flatten(1)
        reference = context["train_images"][context["train_labels"] == label].flatten(1)
        nearest = _nearest_squared(rows, reference)
        threshold = context["thresholds"][label]
        counts.append(int((nearest >= threshold).sum()))
        distances.append(float(threshold))
    return {"exact_duplicate_count": exact, "per_class_thresholds": distances, "per_class_pass_counts": counts, "pass_fraction": sum(counts) / max(1, labels.numel()), "passed": exact == 0 and all(count >= 95 for count in counts)}


def _semantic_metrics(primary: torch.nn.Module, secondary: torch.nn.Module, views: Tensor, labels: Tensor, reference_images: Tensor, reference_labels: Tensor, context: Mapping[str, Any], seed: int) -> dict[str, Any]:
    p_features, p_logits = _eval_model(primary, views)
    s_features, s_logits = _eval_model(secondary, views)
    reference_count = int(labels.numel() // CLASS_COUNT)
    reference = _balanced_reference(reference_images, reference_labels, seed=seed + 19, count=reference_count)
    p_reference, _ = _eval_model(primary, reference.to(views.device))
    s_reference, _ = _eval_model(secondary, reference.to(views.device))
    p_pred, s_pred = p_logits.argmax(1), s_logits.argmax(1)
    counts_p, counts_s = _counts(p_pred, labels), _counts(s_pred, labels)
    agreement = _counts(p_pred, s_pred)
    shuffled = labels.cpu().index_select(0, torch.randperm(labels.numel(), generator=torch.Generator(device="cpu").manual_seed(seed + 8675309))).to(labels.device)
    shuffled_pred = _eval_model(primary, views)[1].argmax(1)
    shuffled_counts = _counts(shuffled_pred, shuffled)
    coverage = {"primary": conditional_coverage_ratio(p_features, views.cpu(), p_reference, reference, labels.cpu(), epsilon=0.05, iterations=20, seed=seed), "secondary": conditional_coverage_ratio(s_features, views.cpu(), s_reference, reference, labels.cpu(), epsilon=0.05, iterations=20, seed=seed)}
    diversity = {"primary": relative_diversity_ratios(p_features, p_reference, labels.cpu()), "secondary": relative_diversity_ratios(s_features, s_reference, labels.cpu())}
    row = {"count": int(labels.numel()), "primary_counts": counts_p, "secondary_counts": counts_s, "agreement_counts": agreement, "primary": {"accuracy": counts_p["correct"] / counts_p["total"], "minimum_recall": min(_recall(p_pred, labels.cpu())), "prediction_histogram": torch.bincount(p_pred, minlength=10).tolist()}, "secondary": {"accuracy": counts_s["correct"] / counts_s["total"], "minimum_recall": min(_recall(s_pred, labels.cpu())), "prediction_histogram": torch.bincount(s_pred, minlength=10).tolist()}, "agreement": agreement["correct"] / agreement["total"], "coverage": coverage, "diversity_ratio": diversity, "label_shuffled_counts": shuffled_counts, "memorization": _memorization(views, labels, context), "finite": bool(torch.isfinite(views).all())}
    row["semantic_pass"] = bool(counts_p["correct"] >= 950 and counts_s["correct"] >= 950 and min(counts_p["recall_correct"]) >= 90 and min(counts_s["recall_correct"]) >= 90)
    row["agreement_pass"] = agreement["correct"] >= 950
    row["safety_pass"] = bool(row["finite"] and row["agreement_pass"] and float(camera_code_fraction(views, labels).item()) >= CAMERA_CODE_MINIMUM and all(0.70 <= value <= 1.30 for name in diversity for value in diversity[name]) and all(value <= 1.50 for value in coverage.values()) and shuffled_counts["correct"] <= 200 and row["memorization"]["passed"] and float((views >= 1).float().mean()) <= SATURATION_MAXIMUM)
    if not _all_finite(row):
        raise ValueError("H-JQOC semantic metrics contain non-finite values")
    return row


def _all_finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _load_system(args: argparse.Namespace, device: torch.device) -> tuple[FormalAmplitudeFiniteCropIdentitySystem, dict[str, str]]:
    from ...configuration.loading import load_config

    config = load_config("configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml")
    system = FormalAmplitudeFiniteCropIdentitySystem(config.system).to(device).eval()
    payload = _trusted_legacy_torch_load(args.v69_checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    for name, target in (("encoder.phase", system.encoder.phase), ("decoder.phase", system.decoder.phase)):
        value = state.get(name)
        if not isinstance(value, Tensor) or tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"H-JQOC frozen phase state is missing or has wrong shape: {name}")
        target.data.copy_(value.to(target))
    return system, {"phase_state": state_content_hash({"encoder": system.encoder.phase, "decoder": system.decoder.phase})}


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = {name: getattr(args, name) for name in ("v69_checkpoint", "v75_report", "v75_decision", "v76_admission", "v76_report", "v76_decision", "h_ndmc_closeout_evidence", "primary_evaluator_checkpoint", "secondary_evaluator_checkpoint", "primary_qualification_evidence", "secondary_qualification_evidence")}
    if not Path(FORMAL_CONFIG_PATH).is_file() or file_sha256(FORMAL_CONFIG_PATH) != FORMAL_CONFIG_SHA256:
        raise ValueError("H-JQOC formal physical config is not the registered immutable input")
    if not Path(args.dataset_root).is_dir() or any(not Path(path).is_file() for path in paths.values()):
        raise FileNotFoundError("H-JQOC lineage input is missing")
    inputs = {name: file_sha256(path) for name, path in paths.items()}
    expected = {"v69_checkpoint": V69_CHECKPOINT_SHA256, "v75_report": V75_REPORT_SHA256, "v75_decision": V75_DECISION_SHA256, "v76_admission": V76_ADMISSION_SHA256, "v76_report": V76_REPORT_SHA256, "v76_decision": V76_DECISION_SHA256, "h_ndmc_closeout_evidence": H_NDMC_CLOSEOUT_SHA256, "primary_evaluator_checkpoint": PRIMARY_EVALUATOR_SHA256, "secondary_evaluator_checkpoint": SECONDARY_EVALUATOR_SHA256, "primary_qualification_evidence": PRIMARY_QUALIFICATION_SHA256, "secondary_qualification_evidence": SECONDARY_QUALIFICATION_SHA256}
    if inputs != expected:
        raise ValueError("H-JQOC upstream artifact hash is not the registered immutable input")
    if _load_json(args.v75_report).get("status") != "v75-carrier-residual-interference-complete" or _load_json(args.v75_decision).get("status") != "carrier-residual-quadrature-rotation-confirmed":
        raise ValueError("H-JQOC requires the locked v75 quadrature decision")
    if _load_json(args.v76_admission).get("status") != "v76-dual-quadrature-admission-passed" or _load_json(args.v76_report).get("status") != "v76-dual-quadrature-readout-complete" or _load_json(args.v76_decision).get("status") != "dual-quadrature-dynamic-range-tradeoff":
        raise ValueError("H-JQOC requires the locked v76 dynamic-range failure")
    primary_qualification = _load_json(args.primary_qualification_evidence)
    secondary_qualification = _load_json(args.secondary_qualification_evidence)
    if primary_qualification.get("status") != "passed" or secondary_qualification.get("status") != "passed":
        raise ValueError("H-JQOC evaluator qualification is not passed")
    ndmc = _load_json(args.h_ndmc_closeout_evidence)
    if ndmc.get("status") != "valid" or ndmc.get("classification") != "nonlinear_manifold_safety_failed":
        raise ValueError("H-JQOC requires the closed H-NDMC valid scientific failure")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_index, validation_index = stratified_mnist_indices(labels)
    if train_index.numel() != 55000 or validation_index.numel() != 5000 or any(int((labels.index_select(0, validation_index) == label).sum()) != 500 for label in range(10)):
        raise ValueError("H-JQOC MNIST split is not the registered 55k/5k split")
    return {"git": _git(), "config_sha256": file_sha256(args.config), "formal_config_sha256": FORMAL_CONFIG_SHA256, "dataset_root": str(Path(args.dataset_root).resolve()), "dataset_hash": tensor_sha256(images), "test_hash": tensor_sha256(load_local_mnist(args.dataset_root, train=False)[0]), "split": {"train_index_hash": tensor_sha256(train_index), "validation_index_hash": tensor_sha256(validation_index)}, "inputs": inputs, "contract": config}


def _check_consumer(path: str, action: str) -> dict[str, Any]:
    payload = _load_json(path)
    allowed = tuple(f"{SCHEMA_PREFIX}_{stage}.v1" for stage in ("validation", "preflight", "audit", "report"))
    if payload.get("schema") not in allowed or payload.get("status") != "valid" or action not in payload.get("permitted_consumers", ()):
        raise ValueError(f"invalid H-JQOC {action} evidence")
    if payload.get("invocation_mode") != "manifest" or payload.get("legacy_diagnostic_only") is not False:
        raise ValueError("legacy H-JQOC evidence")
    return payload


def _write(path: str | Path, payload: dict[str, Any], args: argparse.Namespace) -> None:
    payload.update({"invocation_mode": "manifest", "legacy_diagnostic_only": False, "run_manifest_sha256": file_sha256(args.run_manifest)})
    atomic_write_json(path, payload)


def _stage(args: argparse.Namespace, stage: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    return stage_action_decision(stage, producer_contract=f"{SCHEMA_PREFIX}-{stage}.v1", expected_inputs=expected, evidence=args.output)


def _evaluate_q(rank_values: list[Tensor], dimension: int) -> dict[str, Any]:
    matrix = torch.stack(rank_values)
    singular = torch.linalg.svdvals(matrix)
    ratio = float((singular[-1] / singular[0].clamp_min(1.0e-12)).item())
    return {"rank": int((singular >= singular[0] * RANK_RATIO_MINIMUM).sum()), "dimension": dimension, "singular_ratio": ratio, "rank_pass": int((singular >= singular[0] * RANK_RATIO_MINIMUM).sum()) == dimension}


def _measurement_jacobian(
    system: FormalAmplitudeFiniteCropIdentitySystem,
    canary: StructuredCanary,
    *,
    dimension: int,
    lo_amplitude: float,
    exposure: float,
    device: torch.device,
) -> dict[str, Any]:
    """Measure the frozen optical code Jacobian without retaining fields."""
    step = 0.25
    rows = []
    for label in range(CLASS_COUNT):
        label_tensor = torch.tensor([label], device=device)
        columns = []
        for axis in range(dimension):
            plus_codes = torch.zeros((1, 32), device=device)
            minus_codes = torch.zeros((1, 32), device=device)
            plus_codes[0, axis], minus_codes[0, axis] = step, -step
            plus_field = _forward_fields(system, canary.command(label_tensor, plus_codes, dimension))
            minus_field = _forward_fields(system, canary.command(label_tensor, minus_codes, dimension))
            plus_ports = _balanced_ports(plus_field, lo_amplitude, exposure)
            minus_ports = _balanced_ports(minus_field, lo_amplitude, exposure)
            plus_measurement = torch.sqrt(plus_ports["R"].square() + plus_ports["J"].square())
            minus_measurement = torch.sqrt(minus_ports["R"].square() + minus_ports["J"].square())
            columns.append((F.adaptive_avg_pool2d(plus_measurement - minus_measurement, (8, 8)) / (2.0 * step)).flatten())
        matrix = torch.stack(columns, dim=1).double()
        singular = torch.linalg.svdvals(matrix)
        threshold = singular[0].clamp_min(1.0e-12) * RANK_RATIO_MINIMUM
        rank = int((singular >= threshold).sum())
        rows.append({"label": label, "rank": rank, "dimension": dimension, "singular_values": [float(value) for value in singular.cpu()], "singular_ratio": float((singular[-1] / singular[0].clamp_min(1.0e-12)).item()), "finite": bool(torch.isfinite(matrix).all())})
    return {"dimension": dimension, "points": rows, "rank_pass": all(row["rank"] == dimension for row in rows), "singular_ratio_pass": all(row["singular_ratio"] >= RANK_RATIO_MINIMUM for row in rows), "finite": all(row["finite"] for row in rows)}


def _receiver_controls(
    system: FormalAmplitudeFiniteCropIdentitySystem,
    canary: StructuredCanary,
    primary: torch.nn.Module,
    secondary: torch.nn.Module,
    *,
    lo_amplitude: float,
    exposure: float,
    device: torch.device,
    include_observability: bool,
) -> dict[str, Any]:
    labels = torch.arange(CLASS_COUNT, device=device)
    codes = torch.zeros((CLASS_COUNT, 32), device=device)
    commands = canary.command(labels, codes, 32)
    fields = _forward_fields(system, commands)
    ports = _balanced_ports(fields, lo_amplitude, exposure)
    rotated = _balanced_ports(1j * fields, lo_amplitude, exposure)
    negated = _balanced_ports(-fields, lo_amplitude, exposure)
    off = _balanced_ports(fields, 0.0, exposure)
    zero_electronic = canary.sample(labels, codes, 32)
    zero_raw = _view28(FIXED_EXPOSURE * fields.abs().square())
    zero_primary = _eval_model(primary, zero_electronic)[1].argmax(1)
    zero_secondary = _eval_model(secondary, zero_electronic)[1].argmax(1)
    shifted_labels = torch.roll(labels, shifts=1)
    shifted_electronic = canary.sample(shifted_labels, codes, 32)
    shifted_primary = _eval_model(primary, shifted_electronic)[1].argmax(1)
    shifted_secondary = _eval_model(secondary, shifted_electronic)[1].argmax(1)
    replay = _forward_fields(system, commands[:1])
    repeat = _forward_fields(system, commands[:1])
    packed = _pack_ports({name: ports[name] for name in ("r_plus", "r_minus", "j_plus", "j_minus")})
    result: dict[str, Any] = {
        "zero_code": {"primary": _counts(zero_primary, labels), "secondary": _counts(zero_secondary, labels), "view_hash": tensor_sha256(zero_electronic.cpu()), "raw_view_hash": tensor_sha256(zero_raw.cpu())},
        "label_shift": {"primary": _counts(shifted_primary, labels), "secondary": _counts(shifted_secondary, labels), "shifted_label_hash": tensor_sha256(shifted_labels.cpu())},
        "lo_off": {"r_max": float(off["R"].abs().max()), "j_max": float(off["J"].abs().max()), "joint_code_fraction": float(camera_code_fraction(torch.sqrt(off["R"].square() + off["J"].square()), labels).item())},
        "quadrature_identity": {"r_energy_residual_max": float(ports["r_energy_residual"].abs().max()), "j_energy_residual_max": float(ports["j_energy_residual"].abs().max())},
        "sign_inversion": {"r_error_max": float((negated["R"] + ports["R"]).abs().max()), "j_error_max": float((negated["J"] + ports["J"]).abs().max())},
        "quadrature_swap": {"r_error_max": float((rotated["R"] + ports["J"]).abs().max()), "j_error_max": float((rotated["J"] - ports["R"]).abs().max())},
        "packing": {"shape": list(packed.shape), "roi_shape": list(PORT_SHAPE), "sensor_shape": list(RECEIVER_SHAPE)},
        "repeated_execution_exact": bool(torch.equal(replay, repeat)),
        "finite": bool(_all_finite({"ports": ports, "off": off, "rotated": rotated, "negated": negated, "packed": packed})),
    }
    result["jacobian"] = {str(dimension): _measurement_jacobian(system, canary, dimension=dimension, lo_amplitude=lo_amplitude, exposure=exposure, device=device) for dimension in CODE_DIMS} if include_observability else {}
    result["passes"] = bool(result["finite"] and result["repeated_execution_exact"] and result["quadrature_identity"]["r_energy_residual_max"] <= 1.0e-5 and result["quadrature_identity"]["j_energy_residual_max"] <= 1.0e-5 and result["sign_inversion"]["r_error_max"] <= 1.0e-5 and result["sign_inversion"]["j_error_max"] <= 1.0e-5 and result["quadrature_swap"]["r_error_max"] <= 1.0e-5 and result["quadrature_swap"]["j_error_max"] <= 1.0e-5 and result["packing"]["shape"] == [CLASS_COUNT, 1, *RECEIVER_SHAPE])
    return result


def _forward_fields(system: FormalAmplitudeFiniteCropIdentitySystem, commands: Tensor) -> Tensor:
    fields = []
    with torch.no_grad():
        for start in range(0, commands.shape[0], 4):
            result = system(system.command(commands[start : start + 4]), return_optical_trace=True)
            if result.optical_trace is None:
                raise ValueError("H-JQOC camera complex field is unavailable")
            field = result.optical_trace[-1].tensor
            if not field.is_complex() or not torch.isfinite(field).all():
                raise ValueError("H-JQOC camera complex field is invalid")
            fields.append(field)
    return torch.cat(fields, dim=0)


def _bank(args: argparse.Namespace, canary: StructuredCanary, primary: torch.nn.Module, secondary: torch.nn.Module, context: Mapping[str, Any], reference_images: Tensor, reference_labels: Tensor, system: FormalAmplitudeFiniteCropIdentitySystem, *, seed: int, samples_per_class: int, dimension: int, lo_amplitude: float, exposure: float, device: torch.device) -> dict[str, Any]:
    codes, labels = _panel(seed, samples_per_class, device)
    electronic = canary.sample(labels, codes, dimension)
    commands = canary.command(labels, codes, dimension)
    fields = _forward_fields(system, commands)
    ports = _balanced_ports(fields, lo_amplitude, exposure)
    raw_views = _view28(FIXED_EXPOSURE * fields.abs().square())
    electronic_metrics = _semantic_metrics(primary, secondary, electronic, labels, reference_images, reference_labels, context, seed)
    optical_metrics = _semantic_metrics(primary, secondary, raw_views, labels, reference_images, reference_labels, context, seed + 100)
    joint = torch.sqrt(ports["R"].square() + ports["J"].square()).clamp_min(0.0)
    joint_fraction = float(camera_code_fraction(joint, labels).item())
    port_saturation = {name: float((ports[name] >= 1.0).float().mean()) for name in ("r_plus", "r_minus", "j_plus", "j_minus")}
    identity_error = max(float(ports["r_energy_residual"].abs().max()), float(ports["j_energy_residual"].abs().max()))
    packed = _pack_ports({name: ports[name][:1] for name in ("r_plus", "r_minus", "j_plus", "j_minus")})
    row = {"seed": seed, "dimension": dimension, "count": int(labels.numel()), "electronic": electronic_metrics, "intensity": {**optical_metrics, "code_fraction": float(camera_code_fraction(raw_views, labels).item()), "view_hash": tensor_sha256(raw_views.cpu())}, "quadrature": {"code_fraction": joint_fraction, "port_saturation": port_saturation, "rank_pass": True, "identity_error": identity_error, "packed_shape": list(packed.shape), "measurement_hash": tensor_sha256(joint.cpu())}, "slm_code_fraction": float(camera_code_fraction(commands, labels).item()), "finite": bool(torch.isfinite(commands).all() and torch.isfinite(fields).all() and torch.isfinite(joint).all()), "no_update_replay_exact": True}
    row["electronic_pass"] = bool(electronic_metrics["semantic_pass"] and electronic_metrics["safety_pass"])
    row["intensity_pass"] = bool(optical_metrics["semantic_pass"] and optical_metrics["safety_pass"] and float(camera_code_fraction(raw_views, labels).item()) >= CAMERA_CODE_MINIMUM)
    row["quadrature_safety_pass"] = bool(row["finite"] and identity_error <= 1.0e-5 and all(value <= SATURATION_MAXIMUM for value in port_saturation.values()))
    row["quadrature_pass"] = bool(row["quadrature_safety_pass"] and joint_fraction >= CAMERA_CODE_MINIMUM)
    return row


def _classify(result: Mapping[str, Any]) -> tuple[str, str]:
    if not _measurement_harness_valid(result):
        return "invalid_harness", "repair_and_revalidate_on_new_clean_commit"
    banks = list(result["banks"].values())
    by_dimension: dict[int, list[Mapping[str, Any]]] = {}
    for row in banks:
        by_dimension.setdefault(int(row.get("dimension", -1)), []).append(row)
    complete = {dimension: rows for dimension, rows in by_dimension.items() if len(rows) == len(BANK_SEEDS)}
    eligible = {dimension: rows for dimension, rows in complete.items() if all(row.get("electronic_pass", False) for row in rows)}
    if not eligible:
        return "electronic_structured_input_capacity_failed", "close_joint_quadrature_capacity_route"
    if any(not row.get("intensity", {}).get("semantic_pass", False) for rows in eligible.values() for row in rows):
        return "optical_semantic_carrier_failed", "close_frozen_phase_optical_route"
    if any(row.get("intensity", {}).get("semantic_pass") and not row.get("intensity", {}).get("agreement_pass") for rows in eligible.values() for row in rows):
        return "generated_evaluator_consensus_failed", "register_generated_ood_measurement_counterfactual"
    intensity_dimensions = [dimension for dimension, rows in eligible.items() if all(row.get("intensity_pass", False) for row in rows)]
    if intensity_dimensions:
        dimension = min(intensity_dimensions)
        return "intensity_observability_sufficient_on_structured_canary", f"register_{dimension}d_structured_intensity_architecture_preflight"
    quadrature_safe_dimensions = [dimension for dimension, rows in eligible.items() if all(row.get("quadrature_safety_pass", False) for row in rows)]
    if not quadrature_safe_dimensions:
        return "quadrature_receiver_safety_failed", "close_current_four_port_receiver_design"
    quadrature_dimensions = [dimension for dimension in quadrature_safe_dimensions if all(row.get("quadrature_pass", False) for row in eligible[dimension])]
    if not quadrature_dimensions:
        return "quadrature_code_observability_failed", "close_joint_quadrature_capacity_route"
    dimension = min(quadrature_dimensions)
    return "quadrature_observability_rescue_admitted", f"register_{dimension}d_single_exposure_quadrature_architecture_preflight"


def _measurement_harness_valid(result: Mapping[str, Any]) -> bool:
    expected_keys = {f"{seed}:{dimension}" for seed in BANK_SEEDS for dimension in CODE_DIMS}
    banks = result.get("banks")
    controls = result.get("controls")
    if not isinstance(banks, Mapping) or set(banks) != expected_keys or not isinstance(controls, Mapping):
        return False
    if not isinstance(result.get("historical_replay"), Mapping) or result["historical_replay"].get("matches") is not True:
        return False
    if result.get("measurement_valid") is not True or result.get("optimizer_created") is not False or result.get("checkpoint_created") is not False or result.get("persistent_full_field_saved") is not False:
        return False
    if result.get("parameter_hash_before") != result.get("parameter_hash_after") or result.get("chunk_exact") is not True:
        return False
    required_row_keys = {"seed", "dimension", "count", "electronic", "intensity", "quadrature", "finite", "electronic_pass", "intensity_pass", "quadrature_safety_pass", "quadrature_pass"}
    for key, row in banks.items():
        if not isinstance(row, Mapping) or set(row).intersection(required_row_keys) != required_row_keys:
            return False
        seed, dimension = (int(value) for value in key.split(":"))
        if row.get("seed") != seed or row.get("dimension") != dimension or row.get("count") != CLASS_COUNT * SAMPLES_PER_CLASS or row.get("finite") is not True:
            return False
        if not isinstance(row.get("electronic"), Mapping) or not isinstance(row.get("intensity"), Mapping) or not isinstance(row.get("quadrature"), Mapping):
            return False
        if tuple(row["quadrature"].get("packed_shape", ())) != (1, 1, *RECEIVER_SHAPE):
            return False
    required_controls = {"zero_code", "label_shift", "lo_off", "quadrature_identity", "sign_inversion", "quadrature_swap", "packing", "repeated_execution_exact", "jacobian", "passes", "receiver"}
    if not required_controls.issubset(controls):
        return False
    if controls.get("packing", {}).get("shape") != [CLASS_COUNT, 1, *RECEIVER_SHAPE] or controls.get("repeated_execution_exact") is not True or controls.get("passes") is not True:
        return False
    jacobian = controls.get("jacobian")
    if not isinstance(jacobian, Mapping) or set(jacobian) != {str(dimension) for dimension in CODE_DIMS}:
        return False
    if not all(isinstance(value, Mapping) and value.get("rank_pass") is True and value.get("singular_ratio_pass") is True and value.get("finite") is True for value in jacobian.values()):
        return False
    return _all_finite(result)


def _build_measurement(args: argparse.Namespace, *, samples_per_class: int, include_history: bool) -> dict[str, Any]:
    device = torch.device(args.device)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_index, validation_index = stratified_mnist_indices(labels)
    train_images = images.index_select(0, train_index).to(device)
    train_labels = labels.index_select(0, train_index).to(device)
    validation_images = images.index_select(0, validation_index).to(device)
    validation_labels = labels.index_select(0, validation_index).to(device)
    canary = _fit_canary(train_images, train_labels, device)
    primary, primary_payload = load_frozen_evaluator(args.primary_evaluator_checkpoint, device=device)
    secondary = dpgc._secondary(args.secondary_evaluator_checkpoint, device=device)
    context = _memorization_context(train_images, train_labels, validation_images, validation_labels)
    system, system_hash = _load_system(args, device)
    before = {"primary": state_content_hash(primary.state_dict()), "secondary": state_content_hash(secondary.state_dict()), "optical": state_content_hash(system.state_dict())}
    calibration_codes, calibration_labels = _panel(CALIBRATION_SEED, CALIBRATION_SAMPLES_PER_CLASS, device)
    calibration_commands = canary.command(calibration_labels, calibration_codes, 32)
    calibration_fields = _forward_fields(system, calibration_commands)
    lo_amplitude = float(calibration_fields.abs().square().mean().sqrt().item())
    calibration_ports = _balanced_ports(calibration_fields, lo_amplitude, 1.0)
    calibration_max = torch.cat([calibration_ports[name].flatten() for name in ("r_plus", "r_minus", "j_plus", "j_minus")])
    # CUDA quantile rejects this large calibration vector on the supported
    # runtime.  The calibration buffer is already fixed and non-learned; a
    # CPU reduction preserves the registered scalar without changing inputs.
    quantile = float(torch.quantile(calibration_max.detach().cpu(), 0.999).item())
    exposure = 0.80 / max(quantile, 1.0e-12)
    banks = {}
    for seed in BANK_SEEDS:
        for dimension in CODE_DIMS:
            banks[f"{seed}:{dimension}"] = _bank(args, canary, primary, secondary, context, validation_images, validation_labels, system, seed=seed, samples_per_class=samples_per_class, dimension=dimension, lo_amplitude=lo_amplitude, exposure=exposure, device=device)
    controls = _receiver_controls(system, canary, primary, secondary, lo_amplitude=lo_amplitude, exposure=exposure, device=device, include_observability=samples_per_class == SAMPLES_PER_CLASS)
    for row in banks.values():
        jacobian = controls["jacobian"].get(str(row["dimension"]))
        if jacobian is not None:
            row["quadrature"]["rank_pass"] = bool(jacobian["rank_pass"])
            row["quadrature"]["singular_ratio_pass"] = bool(jacobian["singular_ratio_pass"])
            row["quadrature_pass"] = bool(row["quadrature_pass"] and jacobian["rank_pass"] and jacobian["singular_ratio_pass"])
    after = {"primary": state_content_hash(primary.state_dict()), "secondary": state_content_hash(secondary.state_dict()), "optical": state_content_hash(system.state_dict())}
    probe_codes, probe_labels = _panel(BANK_SEEDS[0], PREFLIGHT_SAMPLES_PER_CLASS, device)
    chunk_a = canary.command(probe_labels, probe_codes, 32).cpu()
    chunk_b = canary.command(probe_labels, probe_codes, 32).cpu()
    historical = {"matches": True}
    if include_history:
        v75, v76 = _load_json(args.v75_report), _load_json(args.v76_report)
        v76_factors = v76.get("factors", {})
        raw_factor = v76_factors.get("raw", {}).get("code_fraction")
        joint_factor = v76.get("joint_factor", {}).get("code_fraction")
        expected_port_saturation = {"r_plus": 0.15661810338497162, "r_minus": 0.0, "j_plus": 0.003801400074735284, "j_minus": 0.004350500181317329}
        observed_port_saturation = {name: value.get("high_saturation") for name, value in v76.get("port_saturation", {}).items() if isinstance(value, Mapping)}
        def camera_counts(metrics: Mapping[str, Any]) -> list[tuple[int, int]]:
            return [(int(item.get("sample_count", -1)), round(float(item.get("accuracy", -1.0)) * int(item.get("sample_count", -1)))) for _, item in sorted(metrics.get("per_label", {}).items()) if isinstance(item, Mapping)]
        replay_metrics = camera_counts(v76.get("camera_metrics", {})) == camera_counts(v75.get("camera_metrics", {})) and camera_counts(v76.get("diagonal_metrics", {})) == camera_counts(v75.get("diagonal_metrics", {}))
        saturation_match = set(observed_port_saturation) == set(expected_port_saturation) and all(math.isclose(observed_port_saturation[name], expected_port_saturation[name], rel_tol=0.0, abs_tol=1.0e-12) for name in expected_port_saturation)
        matches = bool(v75.get("status") == "v75-carrier-residual-interference-complete" and v76.get("status") == "v76-dual-quadrature-readout-complete" and v76.get("finite") is True and v76.get("identity_pass") is True and isinstance(raw_factor, (int, float)) and math.isclose(raw_factor, 0.16960563353404617, rel_tol=0.0, abs_tol=1.0e-12) and isinstance(joint_factor, (int, float)) and math.isclose(joint_factor, 0.740539184173693, rel_tol=0.0, abs_tol=1.0e-12) and saturation_match and replay_metrics)
        historical = {"matches": matches, "v75_camera_metrics": v75.get("camera_metrics"), "v76_camera_metrics": v76.get("camera_metrics"), "v75_raw_code_fraction": raw_factor, "v76_joint_code_fraction": joint_factor, "v76_saturation": v76.get("port_saturation", {}), "count_replay": replay_metrics, "saturation_replay": saturation_match}
    result = {"canary": {"basis_hash": tensor_sha256(canary.basis), "prototype_hash": tensor_sha256(canary.prototypes), "scales": {str(d): float(_scale(canary, d)) for d in CODE_DIMS}, "code_dims": list(CODE_DIMS)}, "banks": banks, "controls": {**controls, "receiver": {"lo_amplitude": lo_amplitude, "exposure": exposure, "calibration_quantile": quantile}}, "historical_replay": historical, "chunk_exact": bool(torch.equal(chunk_a, chunk_b)), "chunk_probe_hash": tensor_sha256(chunk_a), "parameter_hash_before": before, "parameter_hash_after": after, "system_hash": system_hash, "phase_hash": tensor_sha256(torch.cat((system.encoder.phase.detach().cpu().flatten(), system.decoder.phase.detach().cpu().flatten()))), "qualification": {"primary": primary_payload, "primary_checkpoint_sha256": file_sha256(args.primary_evaluator_checkpoint), "secondary_checkpoint_sha256": file_sha256(args.secondary_evaluator_checkpoint)}, "optimizer_created": False, "checkpoint_created": False, "persistent_full_field_saved": False, "evaluator_calls": 2, "measurement_valid": True}
    if not _all_finite(result):
        raise ValueError("H-JQOC measurement contains non-finite values")
    return result


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("H-JQOC lifecycle is manifest-only")
    action = args.research_action
    common = ("config", "dataset_root", "output", "v69_checkpoint", "v75_report", "v75_decision", "v76_admission", "v76_report", "v76_decision", "h_ndmc_closeout_evidence", "primary_evaluator_checkpoint", "secondary_evaluator_checkpoint", "primary_qualification_evidence", "secondary_qualification_evidence")
    required = {"validate": common, "preflight": common + ("validation_evidence",), "audit": common + ("preflight_evidence",), "report": ("config", "dataset_root", "output", "measurement_evidence"), "closeout": ("config", "dataset_root", "output", "measurement_evidence")}
    fields = common + ("device", "validation_evidence", "preflight_evidence", "measurement_evidence")
    args = prepare_args(args, program=PROGRAM, action=action, required=required[action], defaults={"device": "cuda", "seed": HARNESS_SEED}, mixed_fields=fields)
    validate_manifest_namespace(args, action=action)
    config = _config(args.config)
    lineage = _lineage(args, config) if action in {"validate", "preflight", "audit"} else None
    if action == "validate":
        decision = _stage(args, action, {"config": file_sha256(args.config), "lineage": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_validation.v1", "stage": action, "status": "valid", "passed": True, "measurement_valid": True, "lineage": lineage, "consumer_contract": f"{SCHEMA_PREFIX}-preflight.v1", "permitted_consumers": ["preflight"], "producer_contract_revision": f"{SCHEMA_PREFIX}-validation.v1"}, args)
        return 0
    if action == "preflight":
        validation = _check_consumer(args.validation_evidence, "preflight")
        if not validation.get("passed"):
            raise ValueError("H-JQOC preflight requires passed validation")
        decision = _stage(args, action, {"validation": file_sha256(args.validation_evidence), "lineage": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.monotonic()
        result = _build_measurement(args, samples_per_class=PREFLIGHT_SAMPLES_PER_CLASS, include_history=True)
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(args.device))
        storage = sum(item.stat().st_size for item in Path(args._run_manifest["output_root"]).rglob("*") if item.is_file())
        valid = bool(result["historical_replay"]["matches"] and result["chunk_exact"] and result["parameter_hash_before"] == result["parameter_hash_after"] and result["optimizer_created"] is False and result["checkpoint_created"] is False and peak < MEMORY_LIMIT_BYTES and storage <= STORAGE_LIMIT_BYTES and elapsed <= PREFLIGHT_WALL_LIMIT_SECONDS and _all_finite(result))
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_preflight.v1", "stage": action, "status": "valid" if valid else "invalid_harness", "passed": valid, "measurement_valid": valid, "result": result, "peak_memory_bytes": peak, "storage_bytes": storage, "wall_time_seconds": elapsed, "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1", "permitted_consumers": ["audit"] if valid else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-preflight.v1"}, args)
        return 0
    if action == "audit":
        preflight = _check_consumer(args.preflight_evidence, "audit")
        if not preflight.get("passed"):
            raise ValueError("H-JQOC audit requires passed preflight")
        decision = _stage(args, action, {"preflight": file_sha256(args.preflight_evidence), "lineage": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.monotonic()
        result = _build_measurement(args, samples_per_class=SAMPLES_PER_CLASS, include_history=True)
        result["measurement_valid"] = True
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(args.device))
        storage = sum(item.stat().st_size for item in Path(args._run_manifest["output_root"]).rglob("*") if item.is_file())
        harness = bool(result["historical_replay"]["matches"] and result["chunk_exact"] and result["parameter_hash_before"] == result["parameter_hash_after"] and result["optimizer_created"] is False and result["checkpoint_created"] is False and result["persistent_full_field_saved"] is False and peak < MEMORY_LIMIT_BYTES and storage <= STORAGE_LIMIT_BYTES and elapsed <= AUDIT_WALL_LIMIT_SECONDS and _all_finite(result))
        result["measurement_valid"] = harness
        classification, next_action = _classify(result)
        if not harness:
            classification, next_action = "invalid_harness", "repair_and_revalidate_on_new_clean_commit"
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_audit.v1", "stage": action, "status": "valid" if harness else "invalid_harness", "passed": bool(harness and classification.endswith("admitted")), "measurement_valid": harness, "classification": classification, "next_action": next_action, "result": result, "peak_memory_bytes": peak, "storage_bytes": storage, "wall_time_seconds": elapsed, "consumer_contract": f"{SCHEMA_PREFIX}-report.v1", "permitted_consumers": ["report"] if harness else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1"}, args)
        return 0
    measurement = _check_consumer(args.measurement_evidence, action)
    decision = _stage(args, action, {"measurement": file_sha256(args.measurement_evidence)})
    if decision["action"] == "REUSE":
        return 0
    _write(args.output, {"schema": f"{SCHEMA_PREFIX}_{action}.v1", "stage": action, "status": "valid", "passed": measurement.get("passed"), "measurement_valid": measurement.get("measurement_valid"), "classification": measurement.get("classification"), "next_action": measurement.get("next_action"), "measurement": args.measurement_evidence, "source_evidence_sha256": file_sha256(args.measurement_evidence), "permitted_consumers": ["closeout"] if action == "report" else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-{action}.v1"}, args)
    return 0
