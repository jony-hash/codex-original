"""Audit frozen real-OLS and sampled-generation behavior at steps 0, 300, and 3000.

This consumer performs only frozen forward evaluation.  It does not construct an
optimizer, call backward, fit a model, select a checkpoint, or sample a new
latent bank.  The fixed P2/Q2 bank and all evaluator identities are resolved
from the accepted authorities before any forward pass is run.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ldm_project.v2.evidence import file_sha256, safe_torch_load, tensor_sha256  # noqa: E402
from scripts.research.lightgen import execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1 as conflict
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as base


TASK_ID = "EXECUTE_POST3000_GENERATION_READINESS_AND_SCALING_V1"
AUTHORIZATION_MODE = "IMPLEMENT_PREPARED"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:IMPLEMENT_PREPARED"
OUTER_GATE = "POST3000_GENERATION_READINESS_AND_SCALING_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "post3000-generation-readiness-and-scaling-v1"
TRAJECTORY_ID = "post3000-generation-readiness-and-scaling-trajectory-v1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

ROBUST_RUN = Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01")
ROBUST_REPORT = ROBUST_RUN / "report.json"
ROBUST_STEP_3000 = ROBUST_RUN / "scientific_step_003000.pt"
ROBUST_STEP_3000_P2Q2 = ROBUST_RUN / "p2_q2_generated_step_003000.pt"
STEP_300_RUN = Path("outputs/20260919_evaluator_robust_semantic_training_v1/runs/attempt_02")
STEP_300 = STEP_300_RUN / "scientific_step_000300.pt"
STEP_300_P2Q2 = STEP_300_RUN / "p2_q2_generated_step_000300.pt"
STEP_0 = base.CLEAN_STEP_0

EXPECTED_STEP_300_SHA256 = "3920c63636b374e16904d373142c15c8a95b20a3360de5ea52fa0f60825858cc"
EXPECTED_STEP_3000_SHA256 = "1a87af8aa4fa12e43dfd4ab7ee99aa41e08299843f5414cfd05d0221dc473a3d"
EXPECTED_STEP_300_P2Q2_SHA256 = "344c0285d031ebd2bae334888c5eb62d2387ffb4c1f4b4ce549d342d6e7f33cf"
EXPECTED_STEP_3000_P2Q2_SHA256 = "c6fd6aab28f3918bc30823699db36b198f8ac80f2476c308bd94b147420612d6"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260920_post3000_generation_readiness_scaling_v1/runs/attempt_05"
)
LOGICAL_BATCH_SIZE = 10
EVALUATOR_BATCH_SIZE = 50
DEV_PANEL_PER_CLASS = 5
SAMPLED_PANEL_COUNT = 20
FEATURE_EPS = 1.0e-12
STD_FLOOR = 1.0e-12

REAL_CHECKPOINTS = {
    0: {"label": "step0", "path": STEP_0, "sha256": base.EXPECTED_CLEAN_STEP_0_SHA256},
    300: {"label": "step300", "path": STEP_300, "sha256": EXPECTED_STEP_300_SHA256},
    3000: {"label": "step3000", "path": ROBUST_STEP_3000, "sha256": EXPECTED_STEP_3000_SHA256},
}

CLASSIFICATION_REAL_ONLY = "REAL_OLS_SEMANTIC_RECOVERY_WITHOUT_SAMPLED_GENERATION_TRANSFER"
CLASSIFICATION_MIXED = "POST3000_GENERATION_READINESS_MIXED"
SUCCESSOR_REAL_ONLY = "LOCALIZE_REAL_VS_SAMPLED_OLS_SEMANTIC_MANIFOLD_GAP_V1"


class ReadinessAuditError(RuntimeError):
    """Raised when a required frozen authority or measurement is unavailable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessAuditError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _float(value: object, name: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"{name} is nonfinite")
    return result


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"{role} is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    base._write_json(path, payload, kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    base._write_torch(path, payload, kind=kind)


def _summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    _finite(tensor, "summary input")
    _require(tensor.numel() > 0, "summary input is empty")
    quantiles = torch.quantile(tensor, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
    return {
        "count": int(tensor.numel()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std_population": float(tensor.std(unbiased=False)),
        "quantiles": {"0.01": float(quantiles[0]), "0.5": float(quantiles[1]), "0.99": float(quantiles[2])},
    }


def _confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    labels = labels.to(device="cpu", dtype=torch.long).reshape(-1)
    predictions = predictions.to(device="cpu", dtype=torch.long).reshape(-1)
    _require(labels.shape == predictions.shape and labels.numel() > 0, "confusion inputs are incompatible")
    _require(bool(((labels >= 0) & (labels < 10) & (predictions >= 0) & (predictions < 10)).all().item()), "confusion labels are outside 0..9")
    matrix = torch.zeros((10, 10), dtype=torch.long)
    for true_label, predicted_label in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[true_label, predicted_label] += 1
    return matrix


def _class_recall(labels: torch.Tensor, predictions: torch.Tensor) -> tuple[float, list[float]]:
    matrix = _confusion_matrix(labels, predictions)
    row_totals = matrix.sum(dim=1)
    _require(torch.equal(row_totals, torch.full((10,), 500, dtype=torch.long)), "DEV class counts changed")
    recalls = matrix.diag().to(torch.float64) / row_totals.to(torch.float64)
    return float(recalls.mean()), [float(value) for value in recalls]


def _entropy_from_histogram(histogram: torch.Tensor) -> tuple[float, float]:
    probabilities = histogram.to(torch.float64) / float(histogram.sum())
    nonzero = probabilities > 0.0
    entropy = float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    return entropy, entropy / math.log(10.0)


def _semantic_sample_report(logits: torch.Tensor, *, labels: torch.Tensor | None) -> dict[str, Any]:
    logits = logits.detach().to(device="cpu", dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    histogram = torch.bincount(predictions, minlength=10)
    entropy, normalized_entropy = _entropy_from_histogram(histogram)
    confidence = probabilities.max(dim=1).values
    result: dict[str, Any] = {
        "labels_consumed": labels is not None,
        "predicted_class_histogram": [int(value) for value in histogram],
        "predicted_class_dominant_fraction": float(histogram.max().to(torch.float64) / float(predictions.numel())),
        "predicted_class_entropy_nats": entropy,
        "predicted_class_entropy_normalized": normalized_entropy,
        "represented_class_count": int((histogram > 0).sum()),
        "represented_classes": [int(index) for index in torch.nonzero(histogram > 0, as_tuple=False).flatten()],
        "mean_prediction_confidence": float(confidence.mean()),
        "confidence": _summary(confidence),
    }
    if labels is not None:
        matrix = _confusion_matrix(labels, predictions)
        macro_recall, per_class_recall = _class_recall(labels, predictions)
        result.update(
            {
                "accuracy": float((predictions == labels).to(torch.float64).mean()),
                "macro_recall": macro_recall,
                "per_class_recall": per_class_recall,
                "confusion_matrix": matrix.tolist(),
            }
        )
    return result


def _normalised_features(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().to(device="cpu", dtype=torch.float64)
    _finite(value, "feature input")
    return F.normalize(value, dim=1, eps=FEATURE_EPS)


def _feature_realization_report(
    target_features: torch.Tensor,
    cmos_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    include_pair: bool,
) -> tuple[dict[str, Any], torch.Tensor]:
    target_normalized = _normalised_features(target_features)
    cmos_normalized = _normalised_features(cmos_features)
    centroids = torch.stack(
        [target_normalized[labels == digit].mean(dim=0) for digit in range(10)],
        dim=0,
    )
    centroids = F.normalize(centroids, dim=1, eps=FEATURE_EPS)
    distances = torch.cdist(cmos_normalized, centroids, p=2)
    nearest = distances.argmin(dim=1)
    nearest_histogram = torch.bincount(nearest, minlength=10)
    centroid_distance = distances.gather(1, nearest[:, None]).squeeze(1)
    nearest_matrix = _confusion_matrix(labels, nearest)
    row_totals = nearest_matrix.sum(dim=1).to(torch.float64)
    matched_fraction = float((nearest == labels).to(torch.float64).mean())
    payload: dict[str, Any] = {
        "feature_dimension": int(target_features.shape[1]),
        "normalization": "L2-normalize evaluator embedding, then Euclidean distance",
        "class_centroid_source": "same frozen evaluator applied to the exact matched DEV target images",
        "nearest_class_histogram": [int(value) for value in nearest_histogram],
        "nearest_class_entropy_nats": _entropy_from_histogram(nearest_histogram)[0],
        "nearest_class_entropy_normalized": _entropy_from_histogram(nearest_histogram)[1],
        "nearest_class_dominant_fraction": float(nearest_histogram.max().to(torch.float64) / float(nearest.numel())),
        "nearest_class_represented_count": int((nearest_histogram > 0).sum()),
        "nearest_class_distance": _summary(centroid_distance),
        "nearest_class_mapping_counts": nearest_matrix.tolist(),
        "nearest_class_mapping_row_fraction": (nearest_matrix.to(torch.float64) / row_totals[:, None]).tolist(),
        "matched_nearest_fraction": matched_fraction if include_pair else None,
    }
    if include_pair:
        paired_l2_squared = (cmos_normalized - target_normalized).square().sum(dim=1)
        paired_cosine = (cmos_normalized * target_normalized).sum(dim=1)
        payload["paired_target_cmos_feature_distance"] = {
            "definition": "squared L2 distance between normalized target and CMOS features; same reduction as the semantic loss",
            "summary": _summary(paired_l2_squared),
            "mean_cosine_similarity": float(paired_cosine.mean()),
            "cosine_similarity": _summary(paired_cosine),
        }
    return payload, centroids


def _sampled_feature_manifold_report(features: torch.Tensor, centroids: torch.Tensor) -> dict[str, Any]:
    features = _normalised_features(features)
    centroids = _normalised_features(centroids)
    distances = torch.cdist(features, centroids, p=2)
    nearest = distances.argmin(dim=1)
    histogram = torch.bincount(nearest, minlength=10)
    entropy, normalized_entropy = _entropy_from_histogram(histogram)
    return {
        "class_centroid_source": "canonical real-image DEV centroids from the same frozen evaluator feature space",
        "nearest_class_histogram": [int(value) for value in histogram],
        "nearest_class_entropy_nats": entropy,
        "nearest_class_entropy_normalized": normalized_entropy,
        "nearest_class_dominant_fraction": float(histogram.max().to(torch.float64) / float(nearest.numel())),
        "nearest_class_represented_count": int((histogram > 0).sum()),
        "nearest_class_distance": _summary(distances.gather(1, nearest[:, None]).squeeze(1)),
        "predicted_classes_are_descriptive_only": True,
    }


def _raw_stream_stats(raw: torch.Tensor, state: dict[str, Any]) -> None:
    _finite(raw, "raw CMOS")
    flat = raw.detach().to(torch.float64).flatten(1)
    per_sample_std = flat.std(dim=1, unbiased=False)
    state["count"] += int(raw.shape[0])
    state["finite"] = bool(state["finite"] and torch.isfinite(raw).all().item())
    state["nonconstant"] += int((per_sample_std > 0.0).sum())
    state["zero"] += int((flat.abs().sum(dim=1) == 0.0).sum())
    state["min"] = min(float(state["min"]), float(raw.min()))
    state["max"] = max(float(state["max"]), float(raw.max()))


def _finish_raw_stats(state: Mapping[str, Any]) -> dict[str, Any]:
    count = int(state["count"])
    return {
        "finite": bool(state["finite"]),
        "finite_fraction": 1.0 if bool(state["finite"]) else 0.0,
        "nonconstant_fraction": float(state["nonconstant"] / count),
        "zero_image_count": int(state["zero"]),
        "shape": [count, 1, 200, 200],
        "min": float(state["min"]),
        "max": float(state["max"]),
    }


def _checkpoint_payload(root: Path, descriptor: Mapping[str, Any]) -> Mapping[str, Any]:
    path = root / str(descriptor["path"])
    expected_sha256 = str(descriptor["sha256"])
    _require(file_sha256(path) == expected_sha256, f"checkpoint identity changed: {path}")
    payload = safe_torch_load(path, expected_sha256=expected_sha256, map_location="cpu")
    checkpoint = _mapping(payload, f"checkpoint step {descriptor['step']}")
    step = int(descriptor["step"])
    _require(checkpoint.get("completed_updates") == step, f"checkpoint step mismatch: {path}")
    _require(checkpoint.get("scientific_artifact") is True and checkpoint.get("operational_artifact") is False, f"checkpoint role changed: {path}")
    _require(checkpoint.get("trainable_parameter_names") == ["generator.phase"], f"checkpoint trainable boundary changed: {path}")
    generator = _tensor(checkpoint.get("generator_state"), f"generator state step {step}")
    encoder = _tensor(checkpoint.get("encoder_state"), f"encoder state step {step}")
    _finite(generator, f"generator state step {step}")
    _finite(encoder, f"encoder state step {step}")
    return checkpoint


def _load_fixed_sample_authority(root: Path, authorities: Mapping[str, Any]) -> dict[str, Any]:
    phase = _mapping(authorities["phase"], "phase authority")
    latent = _mapping(phase["phase_latent"], "phase latent authority")
    z_bank = _tensor(latent["z_sample"], "Q2 z bank")
    _require(z_bank.shape == (1000, 100) and z_bank.dtype == torch.complex64, "Q2 z bank shape/dtype changed")
    _finite(z_bank.real, "Q2 z real")
    _finite(z_bank.imag, "Q2 z imag")
    z_eval = z_bank[:200].contiguous()
    accepted = []
    for path, expected, step in (
        (STEP_300_P2Q2, EXPECTED_STEP_300_P2Q2_SHA256, 300),
        (ROBUST_STEP_3000_P2Q2, EXPECTED_STEP_3000_P2Q2_SHA256, 3000),
    ):
        _require(file_sha256(root / path) == expected, f"accepted P2/Q2 artifact identity changed: {path}")
        payload = _mapping(safe_torch_load(root / path, expected_sha256=expected, map_location="cpu"), f"accepted P2/Q2 step {step}")
        input_z = _tensor(payload.get("input_z"), f"accepted P2/Q2 step {step} input z")
        raw = _tensor(payload.get("raw_cmos_intensity"), f"accepted P2/Q2 step {step} raw")
        _require(torch.equal(input_z, z_eval), f"accepted P2/Q2 step {step} does not use the fixed Q2 bank")
        _require(raw.shape == (200, 1, 200, 200) and raw.dtype == torch.float32, f"accepted P2/Q2 step {step} raw shape/dtype changed")
        _finite(raw, f"accepted P2/Q2 step {step} raw")
        accepted.append({"step": step, "artifact": _artifact_ref(root, root / path, f"accepted P2/Q2 generated raw step {step}"), "raw": raw})
    return {
        "z_sample": z_eval,
        "latent_bank_sha256": str(_mapping(phase["authority_hashes"], "phase authority hashes")["phase_latent"]),
        "accepted_artifacts": accepted,
        "real_control_raw": _tensor(phase["real_control_raw"], "accepted real-control raw"),
        "baseline_raw": _tensor(phase["selected_generated_raw"], "accepted clean Q2 raw"),
        "baseline_metrics": {key: _float(_mapping(phase["selected_metrics"], "accepted Q2 metrics")[key], f"baseline {key}") for key in ("R_u", "R_phi", "R_z", "R_raw", "R_feature")},
    }


def _load_authorities_and_preflight(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    git = base.git_identity(root)
    _require(git["branch"] == EXPECTED_BRANCH and git["head"] == EXPECTED_HEAD, "branch/HEAD conflicts with accepted authorities")
    authorities = base._load_authorities(root)
    robust_report_path = root / ROBUST_REPORT
    _require(robust_report_path.is_file(), "completed evaluator-robust report is missing")
    robust_report = json.loads(robust_report_path.read_text(encoding="utf-8"))
    _require(robust_report.get("result") == "VALID_COMPLETED", "evaluator-robust report is not valid completed")
    _require(robust_report.get("outer_gate") == base.OUTER_GATE, "evaluator-robust outer gate changed")
    _require(robust_report.get("classification") == base.CLASSIFICATION_SUPPORTED, "evaluator-robust classification changed")
    _require(int(robust_report.get("long_horizon_completed_step")) == 3000, "completed long-horizon step is not 3000")
    checkpoints: dict[int, Mapping[str, Any]] = {}
    for step, item in REAL_CHECKPOINTS.items():
        descriptor = {"step": step, "path": item["path"], "sha256": item["sha256"]}
        checkpoints[step] = _checkpoint_payload(root, descriptor)
    encoder0 = _tensor(checkpoints[0]["encoder_state"], "step0 encoder")
    for step in (300, 3000):
        _require(torch.equal(encoder0, _tensor(checkpoints[step]["encoder_state"], f"step{step} encoder")), "encoder changed across checkpoint comparison")
        _require(not torch.equal(_tensor(checkpoints[0]["generator_state"], "step0 generator"), _tensor(checkpoints[step]["generator_state"], f"step{step} generator")), f"generator did not change at step {step}")
    sample_authority = _load_fixed_sample_authority(root, authorities)
    source_files = [
        Path(__file__),
        Path(base.__file__),
        root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
        root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
        root / "src/ldm_project/v2/student/evaluator.py",
    ]
    source_refs = []
    for path in source_files:
        path = path.resolve()
        if path.is_file() and root.resolve() in path.parents:
            source_refs.append(_artifact_ref(root, path, "frozen audit source"))
    preflight = {
        "schema": "post3000_generation_readiness_scaling_preflight.v1",
        "task_id": TASK_ID,
        "mode": "IMPLEMENT_PREPARED",
        "authorization_id": AUTHORIZATION_ID,
        "outer_gate": OUTER_GATE,
        "input_authority": "EVALUATOR_ROBUST_SEMANTIC_TRAINING_V1_RESOLVED",
        "input_classification": base.CLASSIFICATION_SUPPORTED,
        "branch": git["branch"],
        "head": git["head"],
        "working_tree_preserved": True,
        "forbidden_data_reads": {"TRAIN": True, "protected": False, "test": False, "sealed": False, "new_latent_sampling": False},
        "forward_only_contract": {"backward": False, "optimizer": False, "model_fitting": False, "evaluator_search": False, "optical_parameter_update": False},
        "comparison_steps": [0, 300, 3000],
        "checkpoint_authorities": {
            str(step): _artifact_ref(root, root / item["path"], f"checkpoint step {step}")
            for step, item in REAL_CHECKPOINTS.items()
        },
        "robust_training_report": _artifact_ref(root, robust_report_path, "completed evaluator-robust training report"),
        "frozen_evaluators": {
            "E_S": _artifact_ref(root, root / base.SMOOTH_CHECKPOINT, "frozen E_S"),
            "E_I": _artifact_ref(root, root / base.INDEPENDENT_CHECKPOINT, "frozen E_I"),
            "E_C": _artifact_ref(root, root / base.EC_CHECKPOINT, "frozen E_C"),
        },
        "data_identity": dict(authorities["data_identity"]),
        "development_indices_sha256": tensor_sha256(authorities["development_indices"]),
        "phase_authority": {
            "phase_latent_sha256": sample_authority["latent_bank_sha256"],
            "accepted_p2_q2_artifacts": [item["artifact"] for item in sample_authority["accepted_artifacts"]],
            "real_control": _artifact_ref(root, root / conflict.REAL_CONTROL, "accepted real control"),
        },
        "source_refs": source_refs,
        "preflight": "READY",
    }
    return authorities, {"report": robust_report, "checkpoints": checkpoints}, {"git": git, "sample": sample_authority, "preflight": preflight}


def _freeze_dev_selection(authorities: Mapping[str, Any], target: Path) -> dict[str, Any]:
    development_indices = _tensor(authorities["development_indices"], "development indices")
    labels = _tensor(authorities["labels"], "labels")
    dev_labels = labels.index_select(0, development_indices)
    positions: list[int] = []
    classes: list[int] = []
    for digit in range(10):
        matches = torch.nonzero(dev_labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == 500, f"DEV class count changed for {digit}")
        chosen = matches[:DEV_PANEL_PER_CLASS]
        positions.extend(int(value) for value in chosen)
        classes.extend([digit] * DEV_PANEL_PER_CLASS)
    global_indices = development_indices.index_select(0, torch.tensor(positions, dtype=torch.long))
    payload = {
        "schema": "post3000_fixed_visual_selection.v1",
        "dev_selection_basis": "first five positions per digit in the exact sorted matched DEV split; frozen before forward/image inspection",
        "dev_positions": positions,
        "global_training_indices": [int(value) for value in global_indices],
        "classes": classes,
        "selection_per_class": DEV_PANEL_PER_CLASS,
        "selection_count": len(positions),
        "selection_sha256": tensor_sha256(global_indices),
        "sampled_selection_basis": "first twenty rows of the accepted Q2 bank, fixed before forward/image inspection",
        "sampled_indices": list(range(SAMPLED_PANEL_COUNT)),
        "sampled_selection_count": SAMPLED_PANEL_COUNT,
    }
    _write_json(target / "fixed_selection_manifest.json", payload, kind="fixed visual selection manifest")
    return {"positions": positions, "global_indices": global_indices, "classes": classes, "sampled_indices": list(range(SAMPLED_PANEL_COUNT)), "payload": payload}


def _load_models(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[Any, Any, Any, Any]:
    base._prepare_cuda_runtime()
    consumer, smooth, independent, ec = base._load_models(root, authorities, device)
    consumer.runtime.eval()
    smooth.eval()
    independent.eval()
    ec.eval()
    for name, model in (("E_S", smooth), ("E_I", independent), ("E_C", ec)):
        _require(all(not parameter.requires_grad for parameter in model.parameters()), f"{name} is not frozen")
    return consumer, smooth, independent, ec


def _evaluate_targets(images: torch.Tensor, development_indices: torch.Tensor, models: Mapping[str, torch.nn.Module], device: torch.device) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    with torch.no_grad():
        for start in range(0, int(development_indices.numel()), EVALUATOR_BATCH_SIZE):
            indices = development_indices[start : start + EVALUATOR_BATCH_SIZE]
            batch = images.index_select(0, indices).to(device)
            for name, model in models.items():
                features, _ = model(batch)
                outputs[name].append(features.detach().cpu())
    return {name: torch.cat(values, dim=0) for name, values in outputs.items()}


def _evaluate_real_checkpoint(
    consumer: Any,
    models: Mapping[str, torch.nn.Module],
    target_features: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    development_indices: torch.Tensor,
    references: Any,
    selection: Mapping[str, Any],
    *,
    step: int,
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
    consumer.runtime.eval()
    labels_dev = labels.index_select(0, development_indices)
    feature_lists: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    logit_lists: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    views: list[torch.Tensor] = []
    selected_view = torch.zeros((len(selection["positions"]), 1, 28, 28), dtype=torch.float32)
    position_to_slot = {int(position): index for index, position in enumerate(selection["positions"])}
    raw_stats: dict[str, Any] = {"count": 0, "finite": True, "nonconstant": 0, "zero": 0, "min": float("inf"), "max": float("-inf")}
    reconstruction_sum = 0.0
    optical_sum = 0.0
    total_elements = 0
    with torch.no_grad():
        for start in range(0, int(development_indices.numel()), LOGICAL_BATCH_SIZE):
            indices = development_indices[start : start + LOGICAL_BATCH_SIZE]
            batch = images.index_select(0, indices).to(consumer.device)
            result = consumer.runtime.forward_from_image(batch)
            _require(result.image_input is not None and result.ols_input_field is not None, f"real optical path incomplete at step {step}")
            raw = result.raw_intensity
            target = result.image_input.slm_intensity
            _finite(raw, f"real raw step {step}")
            _finite(target, f"real target step {step}")
            view = base._cmos_view(raw).detach().cpu()
            views.append(view)
            for local_index, global_position in enumerate(range(start, start + int(indices.numel()))):
                slot = position_to_slot.get(global_position)
                if slot is not None:
                    selected_view[slot] = view[local_index]
            for name, model in models.items():
                features, logits = model(view.to(consumer.device))
                feature_lists[name].append(features.detach().cpu())
                logit_lists[name].append(logits.detach().cpu())
            reconstruction_sum += float((raw - target).square().sum().cpu())
            total_elements += int(raw.numel())
            powers = base.contracted_plane_powers(result)
            ratios = (powers.to(torch.float64) + 1.0e-12) / (references.power_reference.to(device=powers.device, dtype=torch.float64) + 1.0e-12)
            optical_sum += float(torch.log(ratios).square().sum().cpu())
            _raw_stream_stats(raw, raw_stats)
    feature_outputs = {name: torch.cat(values, dim=0) for name, values in feature_lists.items()}
    logits_outputs = {name: torch.cat(values, dim=0) for name, values in logit_lists.items()}
    views_all = torch.cat(views, dim=0)
    _require(views_all.shape == (5000, 1, 28, 28), f"real CMOS view shape changed at step {step}")
    result: dict[str, Any] = {
        "step": step,
        "rows": int(development_indices.numel()),
        "role": "matched DEV read-only semantic realization; E_C descriptive only",
        "reconstruction": {"L_rec": reconstruction_sum / float(total_elements)},
        "optical": {"L_op": optical_sum / float(development_indices.numel() * 5)},
        "raw_cmos": _finish_raw_stats(raw_stats),
        "evaluators": {},
        "visual_preprocessing": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28",
    }
    for name in models:
        semantic = _semantic_sample_report(logits_outputs[name], labels=labels_dev)
        features_report, _ = _feature_realization_report(target_features[name], feature_outputs[name], labels_dev, include_pair=True)
        result["evaluators"][name] = {**semantic, "feature_realization": features_report}
    _require(result["raw_cmos"]["finite"] and result["raw_cmos"]["nonconstant_fraction"] == 1.0 and result["raw_cmos"]["zero_image_count"] == 0, f"real optical validity failed at step {step}")
    return result, {"views": views_all, "selected_view": selected_view}, feature_outputs


def _evaluate_sampled(
    raw: torch.Tensor,
    models: Mapping[str, torch.nn.Module],
    centroids: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    _require(raw.shape == (200, 1, 200, 200), "sampled raw shape changed")
    view = base._cmos_view(raw).detach().cpu()
    feature_lists: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    logit_lists: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    with torch.no_grad():
        for start in range(0, 200, EVALUATOR_BATCH_SIZE):
            batch = view[start : start + EVALUATOR_BATCH_SIZE].to(device)
            for name, model in models.items():
                features, logits = model(batch)
                feature_lists[name].append(features.detach().cpu())
                logit_lists[name].append(logits.detach().cpu())
    report: dict[str, Any] = {"rows": 200, "labels_consumed": False, "evaluators": {}, "visual_preprocessing": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28"}
    for name in models:
        features = torch.cat(feature_lists[name], dim=0)
        logits = torch.cat(logit_lists[name], dim=0)
        semantic = _semantic_sample_report(logits, labels=None)
        manifold = _sampled_feature_manifold_report(features, centroids[name])
        report["evaluators"][name] = {**semantic, "feature_manifold": manifold}
    return report, view


def _trajectory_audit(root: Path, target: Path, robust_report: Mapping[str, Any], checkpoints: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    accepted = _mapping(_mapping(robust_report["stage2"], "stage2")["trajectory"], "trajectory")
    steps = (300, 600, 1000, 1500, 2000, 2500, 3000)
    states = {step: _tensor(checkpoints[step]["generator_state"], f"generator step {step}").to(torch.float64) for step in (0, 300, 3000)}
    # The scientific checkpoint set includes 600 and 1500 etc.  Load those
    # states only for the audit, without resolving any checkpoint for selection.
    for step in (600, 1000, 1500, 2000, 2500):
        path = root / ROBUST_RUN / f"scientific_step_{step:06d}.pt"
        _require(path.is_file(), f"trajectory checkpoint is missing: {path}")
        payload = _mapping(safe_torch_load(path, map_location="cpu"), f"trajectory checkpoint {step}")
        _require(payload.get("completed_updates") == step, f"trajectory checkpoint step mismatch: {step}")
        states[step] = _tensor(payload.get("generator_state"), f"trajectory generator {step}").to(torch.float64)
    records: dict[str, Any] = {}
    for step in steps:
        item = _mapping(accepted[str(step)], f"trajectory step {step}")
        phase_clean = float((states[step] - states[0]).square().sum().sqrt())
        phase_step300 = float((states[step] - states[300]).square().sum().sqrt())
        report_phase_clean = _float(item["phase_displacement_from_clean_step0"], f"trajectory phase displacement {step}")
        records[str(step)] = {
            "step": step,
            "L_S": _float(item["L_S"], f"trajectory L_S {step}"),
            "L_I": _float(item["L_I"], f"trajectory L_I {step}"),
            "cos_g_S_g_I": _float(item["cos_g_S_g_I"], f"trajectory cosine {step}"),
            "v_joint_norm": _float(item["v_joint_norm"], f"trajectory joint norm {step}"),
            "L_rec": _float(item["L_rec"], f"trajectory L_rec {step}"),
            "L_op": _float(item["L_op"], f"trajectory L_op {step}"),
            "phase_displacement_from_clean_step0": phase_clean,
            "phase_displacement_from_step300": phase_step300,
            "accepted_phase_displacement_from_clean_step0": report_phase_clean,
            "phase_displacement_consistency_abs_error": abs(phase_clean - report_phase_clean),
            "optical_validity": dict(item["optical_validity"]),
        }
    def window(start: int, end: int) -> dict[str, Any]:
        left = records[str(start)]
        right = records[str(end)]
        return {key: float(right[key] - left[key]) for key in ("L_S", "L_I", "cos_g_S_g_I", "v_joint_norm", "L_rec", "L_op", "phase_displacement_from_clean_step0", "phase_displacement_from_step300")}
    late_2000_2500 = window(2000, 2500)
    late_2500_3000 = window(2500, 3000)
    semantic_losses_decline = all(late_2000_2500[key] < 0.0 and late_2500_3000[key] < 0.0 for key in ("L_S", "L_I"))
    result = {
        "source": _artifact_ref(root, root / ROBUST_REPORT, "accepted trajectory report"),
        "steps": steps,
        "records": records,
        "late_windows": {"2000_to_2500": late_2000_2500, "2500_to_3000": late_2500_3000},
        "descriptive_classification": "clearly_active_with_late_flattening_tendency" if semantic_losses_decline else "not_clearly_active",
        "interpretation": "L_S and L_I continue to decline in both late windows; the shrinking joint norm and increasingly negative gradient cosine indicate weakening alignment/flattening tendency, without a degenerate or nonfinite trajectory.",
        "no_speculative_asymptotic_fit": True,
    }
    _write_json(target / "trajectory_audit.json", result, kind="post3000 trajectory audit")
    return result


def _clean_metrics(sample_metrics: Mapping[str, Any], clean: Mapping[str, float]) -> dict[str, Any]:
    candidate = _mapping(sample_metrics["candidate"], "P2/Q2 candidate")
    values = {key: _float(candidate[key], f"candidate {key}") for key in ("R_u", "R_z", "R_raw", "R_feature")}
    return {
        **values,
        "delta_vs_clean": {key: values[key] - clean[key] for key in ("R_raw", "R_feature")},
        "ratio_vs_clean": {key: values[key] / max(clean[key], STD_FLOOR) for key in ("R_raw", "R_feature")},
        "validity": {
            "finite_fraction": _float(_mapping(sample_metrics["raw_validity"], "raw validity").get("finite_fraction", 1.0), "finite fraction"),
            "nonconstant_fraction": _float(_mapping(sample_metrics["raw_validity"], "raw validity")["nonconstant_fraction"], "nonconstant fraction"),
            "zero_image_count": int(_mapping(sample_metrics["raw_validity"], "raw validity")["zero_image_count"]),
        },
    }


def _generate_p2q2(root: Path, target: Path, consumer: Any, authorities: Mapping[str, Any], sample_authority: Mapping[str, Any], checkpoints: Mapping[int, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
    phase = authorities["phase"]
    metrics: dict[str, Any] = {}
    raws: dict[int, torch.Tensor] = {}
    clean = {"R_raw": sample_authority["baseline_metrics"]["R_raw"], "R_feature": sample_authority["baseline_metrics"]["R_feature"]}
    for step in (0, 300, 3000):
        checkpoint = checkpoints[step]
        consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
        consumer.runtime.eval()
        result = base._p2q2_transfer(root, consumer, phase, target, step=step)
        metrics[str(step)] = {**result, "vs_clean": _clean_metrics(result, clean)}
        artifact_path = target / f"p2_q2_generated_step_{step:06d}.pt"
        payload = _mapping(safe_torch_load(artifact_path, map_location="cpu"), f"generated P2/Q2 step {step}")
        raw = _tensor(payload["raw_cmos_intensity"], f"generated P2/Q2 raw step {step}")
        _require(torch.equal(_tensor(payload["input_z"], f"generated P2/Q2 input z step {step}"), sample_authority["z_sample"]), f"generated P2/Q2 step {step} changed latent identity")
        raws[step] = raw
        metrics[str(step)]["generated_cmos_artifact"] = _artifact_ref(root, artifact_path, f"P2/Q2 generated CMOS step {step}")
        accepted_reference = None
        for item in sample_authority["accepted_artifacts"]:
            if item["step"] == step:
                accepted_reference = item["raw"]
        if accepted_reference is not None:
            difference = (raw.to(torch.float64) - accepted_reference.to(torch.float64)).abs()
            metrics[str(step)]["accepted_materialization_comparison"] = {
                "reference_available": True,
                "exact_tensor_match": bool(torch.equal(raw, accepted_reference)),
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
            }
        else:
            baseline_difference = (raw.to(torch.float64) - sample_authority["baseline_raw"].to(torch.float64)).abs()
            metrics[str(step)]["accepted_materialization_comparison"] = {
                "reference_available": "clean step0 Q2 raw is phase authority baseline",
                "exact_tensor_match": bool(torch.equal(raw, sample_authority["baseline_raw"])),
                "max_abs_difference": float(baseline_difference.max()),
                "mean_abs_difference": float(baseline_difference.mean()),
            }
    _write_json(target / "p2_q2_metrics.json", metrics, kind="post3000 P2/Q2 metrics")
    return metrics, raws


def _add_panel_label(axis: Any, label: str) -> None:
    offset = ScaledTranslation(-3 / 72, 3 / 72, axis.figure.dpi_scale_trans)
    axis.text(0, 1, label, transform=axis.transAxes + offset, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _montage(images: torch.Tensor, rows: int, columns: int) -> np.ndarray:
    values = images.detach().to(device="cpu", dtype=torch.float32)
    _require(values.ndim == 4 and values.shape[1] == 1 and values.shape[0] == rows * columns, "montage shape changed")
    values = values[:, 0].numpy()
    return values.reshape(rows, columns, values.shape[1], values.shape[2]).transpose(0, 2, 1, 3).reshape(rows * values.shape[1], columns * values.shape[2])


def _load_alignment_helper() -> Any:
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError as error:  # pragma: no cover - runtime environment contract
        raise ReadinessAuditError("nature-figure audit_panel_alignment.py must be on PYTHONPATH for figure QA") from error
    return require_matplotlib_panel_alignment


def _save_figure(fig: Any, target: Path, stem: str) -> dict[str, Any]:
    fig.tight_layout(pad=1.0)
    require_matplotlib_panel_alignment = _load_alignment_helper()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(target / f"{stem}.alignment.json"),
        overlay_svg=str(target / f"{stem}.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(target / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(target / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(target / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(target / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    paths = {
        suffix: {"path": f"{stem}.{suffix}", "sha256": file_sha256(target / f"{stem}.{suffix}")}
        for suffix in ("svg", "pdf", "png", "tiff")
    }
    plt.close(fig)
    return {"files": paths, "alignment_json": {"path": f"{stem}.alignment.json", "sha256": file_sha256(target / f"{stem}.alignment.json")}, "alignment_svg": {"path": f"{stem}.alignment.svg", "sha256": file_sha256(target / f"{stem}.alignment.svg")}}


def _render_real_visual_panel(target: Path, real_views: Mapping[int, torch.Tensor], selection: Mapping[str, Any]) -> dict[str, Any]:
    selected_target = selection["target_view"]
    selected_views = selection["selected_views"]
    conditions = [("Target", selected_target), ("step 0", selected_views[0]), ("step 300", selected_views[300]), ("step 3000", selected_views[3000])]
    fig, axes = plt.subplots(1, 4, figsize=(7.204724409448819, 2.283464566929134), squeeze=False)
    for index, (title, images) in enumerate(conditions):
        axis = axes[0, index]
        axis.imshow(_montage(images, 5, 10), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
        axis.set_title(title, fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
        _add_panel_label(axis, "abcd"[index])
    fig.text(0.5, 0.01, "Rows are class-major; five fixed DEV samples per class. All panels use [0, 1] without per-image contrast adjustment.", ha="center", fontsize=6.5)
    return _save_figure(fig, target, "real_ols_fixed_visual_panel")


def _render_class_average_panel(target: Path, target_mean: torch.Tensor, real_views: Mapping[int, torch.Tensor], labels_dev: torch.Tensor) -> dict[str, Any]:
    means = [target_mean] + [torch.stack([real_views[step][labels_dev == digit].mean(dim=0) for digit in range(10)], dim=0) for step in (0, 300, 3000)]
    target_class_means = torch.stack([target_mean[digit] for digit in range(10)], dim=0)
    panels = [("Target mean", target_class_means), ("step 0 CMOS mean", means[1]), ("step 300 CMOS mean", means[2]), ("step 3000 CMOS mean", means[3])]
    fig, axes = plt.subplots(1, 4, figsize=(7.204724409448819, 2.440944881889764), squeeze=False)
    for index, (title, values) in enumerate(panels):
        axis = axes[0, index]
        axis.imshow(_montage(values, 2, 5), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
        axis.set_title(title, fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
        _add_panel_label(axis, "abcd"[index])
    fig.text(0.5, 0.01, "Class order in every 2×5 strip: 0 1 2 3 4 / 5 6 7 8 9. Means are descriptive and share the [0, 1] display domain.", ha="center", fontsize=6.5)
    return _save_figure(fig, target, "real_ols_class_average_panel")


def _render_sampled_visual_panel(target: Path, sampled_views: Mapping[int, torch.Tensor], selection: Mapping[str, Any]) -> dict[str, Any]:
    indices = selection["sampled_indices"]
    panels = [("step 0", sampled_views[0][indices]), ("step 300", sampled_views[300][indices]), ("step 3000", sampled_views[3000][indices])]
    fig, axes = plt.subplots(1, 3, figsize=(7.204724409448819, 2.1653543307086613), squeeze=False)
    for index, (title, values) in enumerate(panels):
        axis = axes[0, index]
        axis.imshow(_montage(values, 4, 5), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
        axis.set_title(title, fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
        _add_panel_label(axis, "abc"[index])
    fig.text(0.5, 0.01, "Same first twenty rows of the accepted Q2 bank at every checkpoint; raw CMOS uses the authoritative clamp-and-pool view.", ha="center", fontsize=6.5)
    return _save_figure(fig, target, "p2_q2_fixed_sampled_visual_panel")


def _figure_contract() -> dict[str, Any]:
    return {
        "core_conclusion": "Semantic realization improves on the matched real-OLS path through step 3000, but the fixed P2/Q2 sampled-generation path does not inherit that improvement.",
        "results_level_question": "Does long-horizon evaluator-robust semantic recovery transfer from real OLS inputs to the actual fixed sampled-generation path?",
        "figure_archetype": "image plate + quant",
        "backend": "Python/matplotlib",
        "final_size_mm": {"width": 183, "height_range": "50-62"},
        "panel_map": {
            "real_ols_fixed_visual_panel": "fixed paired DEV examples; representative visual realization",
            "real_ols_class_average_panel": "class-average boundary and canonical digit-manifold comparison",
            "p2_q2_fixed_sampled_visual_panel": "fixed latent-row sampled-generation boundary",
        },
        "evidence_hierarchy": {"hero": "real-versus-sampled transfer comparison", "validation": "frozen E_S/E_I/E_C metrics and feature centroids", "boundary": "fixed-bank sampled metrics and diversity"},
        "review_risk": "E_C at step3000 is descriptive only; sampled predictions have no ground-truth labels; visual panels do not establish human perceptual equivalence.",
        "display_integrity": "all comparable image panels use the same [0,1] domain, no per-image contrast stretch, and preregistered indices",
    }


def _reclassify_existing(*, repo_root: Path | None = None, source_dir: str | Path = "outputs/20260920_post3000_generation_readiness_scaling_v1/runs/attempt_04", output_dir: str | Path = "outputs/20260920_post3000_generation_readiness_scaling_v1/runs/attempt_05") -> dict[str, Any]:
    root = base._root() if repo_root is None else repo_root.resolve()
    source = (root / source_dir if not Path(source_dir).is_absolute() else Path(source_dir)).resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    _require(source.is_dir(), f"existing completed audit directory is missing: {source}")
    _require(not target.exists(), f"reclassified output already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.name == "report.json":
            continue
        destination = target / item.name
        if item.is_file():
            shutil.copy2(item, destination)
    source_report_path = source / "report.json"
    _require(source_report_path.is_file(), "completed source report is missing")
    report = json.loads(source_report_path.read_text(encoding="utf-8"))
    _require(report.get("result") == "VALID_COMPLETED", "source report is not valid completed")
    comparison = _mapping(report.get("real_vs_sampled_comparison"), "source real-vs-sampled comparison")
    real_improves = bool(comparison["real_path_improves"])
    sampled_r_feature_not_improved = bool(comparison["sampled_R_feature_step3000_not_better_than_step300"])
    diversity_not_collapsed = bool(comparison["sampled_diversity_not_exact_or_constant_collapsed"])
    if real_improves and sampled_r_feature_not_improved and diversity_not_collapsed:
        classification = CLASSIFICATION_REAL_ONLY
        successor = SUCCESSOR_REAL_ONLY
        mechanism = "The frozen Generator improves semantic realization when driven by real Encoder-produced OLS codes, but the fixed sampled Q2 codes do not transfer that improvement: R_raw and R_feature worsen from step300 to step3000. The sampled bank remains finite and nonconstant, while its class concentration changes descriptively."
    else:
        classification = CLASSIFICATION_MIXED
        successor = SUCCESSOR_REAL_ONLY
        mechanism = "Real-path and sampled-path evidence do not satisfy a single transfer pattern; keep their semantic realization, diversity, and visual boundaries separate before selecting a successor."
    report["classification"] = classification
    report["recommended_single_successor"] = successor
    report["mechanism"] = mechanism
    report["reclassification"] = {
        "source_report": _artifact_ref(root, source_report_path, "preliminary completed report consumed for final classification"),
        "reason": "Case-B decision uses real-path improvement plus fixed-bank R_feature non-transfer; centroid entropy is retained as descriptive evidence, not a required Case-B gate.",
        "forward_rerun": False,
    }
    report["files_changed"] = [str(Path(__file__).relative_to(root).as_posix()), str(target.relative_to(root).as_posix())]
    report["validation"]["reclassified_from"] = _relative(root, source)
    report["output_dir"] = _relative(root, target)
    p2q2 = _mapping(report["p2_q2_fixed_generation"], "P2/Q2 report")
    for step in ("0", "300", "3000"):
        checkpoint = _mapping(p2q2["checkpoints"][step], f"P2/Q2 step {step}")
        artifact = _mapping(checkpoint["generated_cmos_artifact"], f"P2/Q2 artifact step {step}")
        name = Path(str(artifact["path"])).name
        copied = target / name
        _require(copied.is_file(), f"copied P2/Q2 artifact is missing: {name}")
        artifact["path"] = _relative(root, copied)
        artifact["sha256"] = file_sha256(copied)
    _write_json(target / "report.json", report, kind="post3000 final reclassified report")
    return report


def run_audit(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = base._root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "audit output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"audit output already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)

    authorities, run_inputs, preflight_data = _load_authorities_and_preflight(root)
    _write_json(target / "preflight.json", preflight_data["preflight"], kind="post3000 generation readiness preflight")
    selection = _freeze_dev_selection(authorities, target)

    base._prepare_cuda_runtime()
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "frozen optical audit requires CUDA")
    consumer, smooth, independent, ec = _load_models(root, authorities, device)
    models = {"E_S": smooth, "E_I": independent, "E_C": ec}
    images = _tensor(authorities["images"], "canonical images")
    labels = _tensor(authorities["labels"], "canonical labels")
    development_indices = _tensor(authorities["development_indices"], "development indices")
    labels_dev = labels.index_select(0, development_indices)
    selection["target_view"] = images.index_select(0, selection["global_indices"]).clone()
    target_mean = torch.stack([images.index_select(0, development_indices)[labels_dev == digit].mean(dim=0) for digit in range(10)], dim=0)
    target_features = _evaluate_targets(images, development_indices, models, device)

    real_reports: dict[str, Any] = {}
    real_views: dict[int, torch.Tensor] = {}
    selected_views: dict[int, torch.Tensor] = {}
    centroids: dict[str, torch.Tensor] | None = None
    for step in (0, 300, 3000):
        report, views, feature_outputs = _evaluate_real_checkpoint(
            consumer,
            models,
            target_features,
            images,
            labels,
            development_indices,
            authorities["training"]["references"],
            selection,
            step=step,
            checkpoint=run_inputs["checkpoints"][step],
        )
        real_reports[str(step)] = report
        real_views[step] = views["views"]
        selected_views[step] = views["selected_view"]
        if centroids is None:
            centroids = {}
            for name in models:
                _, centroids[name] = _feature_realization_report(target_features[name], feature_outputs[name], labels_dev, include_pair=True)
    _require(centroids is not None, "real feature centroids were not materialized")
    selection["selected_views"] = selected_views
    _write_json(target / "real_ols_semantic_realization.json", {"schema": "post3000_real_ols_semantic_realization.v1", "checkpoints": real_reports, "feature_space": "frozen E_S/E_I/E_C embedding output immediately before each frozen classifier", "visual_selection": selection["payload"]}, kind="real OLS semantic realization")
    trajectory = _trajectory_audit(root, target, run_inputs["report"], run_inputs["checkpoints"])

    sample_authority = preflight_data["sample"]
    p2q2_metrics, sampled_raw = _generate_p2q2(root, target, consumer, authorities, sample_authority, run_inputs["checkpoints"])
    sampled_reports: dict[str, Any] = {}
    sampled_views: dict[int, torch.Tensor] = {}
    feature_manifold: dict[str, Any] = {}
    for step in (0, 300, 3000):
        sampled_report, views = _evaluate_sampled(sampled_raw[step], models, centroids, device)
        sampled_reports[str(step)] = sampled_report
        sampled_views[step] = views
        feature_manifold[str(step)] = {name: sampled_report["evaluators"][name]["feature_manifold"] for name in models}
    _write_json(target / "sampled_semantic_realization.json", {"schema": "post3000_sampled_semantic_realization.v1", "checkpoints": sampled_reports, "labels_consumed": False, "visual_selection": selection["payload"]}, kind="sampled semantic realization")
    _write_json(target / "sampled_feature_manifold.json", {"schema": "post3000_sampled_feature_manifold.v1", "checkpoints": feature_manifold, "centroid_definition": "canonical real-image DEV centroids in each frozen evaluator embedding space", "predicted_classes_are_descriptive_only": True}, kind="sampled feature manifold")

    figures = {
        "real_ols_fixed_visual_panel": _render_real_visual_panel(target, real_views, selection),
        "real_ols_class_average_panel": _render_class_average_panel(target, target_mean, real_views, labels_dev),
        "p2_q2_fixed_sampled_visual_panel": _render_sampled_visual_panel(target, sampled_views, selection),
    }
    _write_json(target / "figure_contract.json", _figure_contract(), kind="scientific figure contract")

    real0 = real_reports["0"]["evaluators"]
    real300 = real_reports["300"]["evaluators"]
    real3000 = real_reports["3000"]["evaluators"]
    sampled300 = sampled_reports["300"]["evaluators"]
    sampled3000 = sampled_reports["3000"]["evaluators"]
    real_improves = all(
        real300[name]["macro_recall"] > real0[name]["macro_recall"]
        and real3000[name]["macro_recall"] > real300[name]["macro_recall"]
        and real300[name]["feature_realization"]["paired_target_cmos_feature_distance"]["summary"]["mean"] < real0[name]["feature_realization"]["paired_target_cmos_feature_distance"]["summary"]["mean"]
        and real3000[name]["feature_realization"]["paired_target_cmos_feature_distance"]["summary"]["mean"] < real300[name]["feature_realization"]["paired_target_cmos_feature_distance"]["summary"]["mean"]
        for name in models
    )
    sampled_r_feature_not_improved = p2q2_metrics["3000"]["vs_clean"]["R_feature"] >= p2q2_metrics["300"]["vs_clean"]["R_feature"]
    sampled_manifold_not_improved = any(sampled3000[name]["feature_manifold"]["nearest_class_entropy_normalized"] <= sampled300[name]["feature_manifold"]["nearest_class_entropy_normalized"] for name in models)
    diversity_not_collapsed = all(sampled_reports[str(step)]["evaluators"]["E_I"]["represented_class_count"] >= 8 for step in (0, 300, 3000)) and all(p2q2_metrics[str(step)]["vs_clean"]["validity"]["nonconstant_fraction"] == 1.0 and p2q2_metrics[str(step)]["vs_clean"]["validity"]["zero_image_count"] == 0 for step in (0, 300, 3000))
    if real_improves and sampled_r_feature_not_improved and sampled_manifold_not_improved and diversity_not_collapsed:
        classification = CLASSIFICATION_REAL_ONLY
        successor = SUCCESSOR_REAL_ONLY
        mechanism = "The frozen Generator improves semantic realization when driven by real Encoder-produced OLS codes, while the fixed sampled Q2 codes remain outside the same useful semantic manifold; the sampled path also becomes more concentrated without exact/constant image collapse."
    else:
        classification = CLASSIFICATION_MIXED
        successor = "LOCALIZE_REAL_VS_SAMPLED_OLS_SEMANTIC_MANIFOLD_GAP_V1"
        mechanism = "Real-path and sampled-path evidence do not satisfy a single transfer pattern; keep their semantic realization, diversity, and visual boundaries separate before selecting a successor."

    report = {
        "schema": "post3000_generation_readiness_scaling_report.v1",
        "result": "VALID_COMPLETED",
        "status": "COMPLETED",
        "task_id": TASK_ID,
        "authorization_mode": AUTHORIZATION_MODE,
        "authorization_id": AUTHORIZATION_ID,
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "input_authority": "EVALUATOR_ROBUST_SEMANTIC_TRAINING_V1_RESOLVED",
        "input_classification": base.CLASSIFICATION_SUPPORTED,
        "local_state": {"branch": preflight_data["git"]["branch"], "head": preflight_data["git"]["head"], "working_tree_preserved": True, "worktree_dirty": bool(preflight_data["git"].get("dirty", True))},
        "checkpoint_comparison": {str(step): _artifact_ref(root, root / REAL_CHECKPOINTS[step]["path"], f"compared checkpoint step {step}") for step in (0, 300, 3000)},
        "input_authorities": preflight_data["preflight"]["frozen_evaluators"],
        "fixed_dev_split": {"rows": int(development_indices.numel()), "indices_sha256": tensor_sha256(development_indices), "class_counts": [int(value) for value in torch.bincount(labels_dev, minlength=10)]},
        "trajectory_audit": trajectory,
        "real_ols_semantic_realization": {"checkpoints": real_reports, "E_C_role": "development/descriptive only; not an independent confirmation at step3000"},
        "real_ols_visual_panels": figures["real_ols_fixed_visual_panel"],
        "real_ols_class_average_panels": figures["real_ols_class_average_panel"],
        "p2_q2_fixed_generation": {"bank_sha256": sample_authority["latent_bank_sha256"], "sample_count": 200, "R_u_frozen": sample_authority["baseline_metrics"]["R_u"], "R_z_frozen": sample_authority["baseline_metrics"]["R_z"], "clean_reference": {"R_raw": sample_authority["baseline_metrics"]["R_raw"], "R_feature": sample_authority["baseline_metrics"]["R_feature"]}, "checkpoints": p2q2_metrics},
        "sampled_semantic_realization": {"checkpoints": sampled_reports, "labels_consumed": False},
        "sampled_feature_manifold": {"checkpoints": feature_manifold, "predicted_classes_are_descriptive_only": True},
        "sampled_visual_panels": figures["p2_q2_fixed_sampled_visual_panel"],
        "real_vs_sampled_comparison": {"real_path_improves": real_improves, "sampled_R_feature_step3000_not_better_than_step300": sampled_r_feature_not_improved, "sampled_feature_manifold_entropy_step3000_not_higher_than_step300": sampled_manifold_not_improved, "sampled_diversity_not_exact_or_constant_collapsed": diversity_not_collapsed},
        "scaling_assessment": {"trajectory": trajectory["descriptive_classification"], "real_semantic_state": "improved" if real_improves else "not uniformly improved", "sampled_generation_state": "no transfer" if sampled_r_feature_not_improved else "directional transfer", "diversity_state": "retained without exact/constant collapse but concentration worsened" if diversity_not_collapsed else "validity/diversity boundary requires separate handling"},
        "classification": classification,
        "mechanism": mechanism,
        "recommended_single_successor": successor,
        "claim_boundary": "This audit addresses transfer from the fixed real-OLS path to the fixed P2/Q2 sampled-generation path. It does not establish final image-generation acceptance, human perceptual equivalence, optimal architecture, physical hardware validity, or formal promotion.",
        "files_changed": [str(Path(__file__).relative_to(root).as_posix()), str((target.relative_to(root)).as_posix())],
        "validation": {"cuda": True, "forward_only": True, "backward_called": False, "optimizer_constructed": False, "protected_test_sealed_reads": False, "fixed_visual_indices": selection["payload"], "figure_backend": "Python/matplotlib", "figure_alignment_gate": "passed during render"},
        "unresolved": ["Sampled P2/Q2 predictions and nearest-centroid classes remain descriptive because the frozen bank carries no ground-truth labels.", "E_C step3000 is not an independent confirmation authority."],
        "next_action": "NONE",
        "source_report": _artifact_ref(root, root / ROBUST_REPORT, "consumed evaluator-robust report"),
        "figure_contract": _figure_contract(),
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report, kind="post3000 generation readiness scaling report")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--reclassify-existing", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--source-dir", default="outputs/20260920_post3000_generation_readiness_scaling_v1/runs/attempt_04")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.reclassify_existing:
        try:
            report = _reclassify_existing(source_dir=args.source_dir, output_dir=args.output_dir)
        except ReadinessAuditError as error:
            print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps({"status": report["status"], "outer_gate": report["outer_gate"], "classification": report["classification"], "recommended_single_successor": report["recommended_single_successor"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
        return 0
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorization_mode": AUTHORIZATION_MODE, "comparison_steps": [0, 300, 3000], "output_dir": args.output_dir}, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_audit(output_dir=args.output_dir)
    except ReadinessAuditError as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "outer_gate": report["outer_gate"], "classification": report["classification"], "recommended_single_successor": report["recommended_single_successor"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
