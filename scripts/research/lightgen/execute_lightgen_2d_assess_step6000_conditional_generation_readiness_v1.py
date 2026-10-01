"""Assess the frozen step-6000 conditional generation state.

This task is a forward-only audit.  It consumes the accepted step-3000,
step-3300 and step-6000 checkpoints, the frozen S1 conditional bank, the
identity-compatible S0 replay bank, the canonical DEV role and four frozen
development evaluators.  It does not fit, update, select, or promote any
component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.transforms import ScaledTranslation

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import LightGen2DChenBOGTComplexOLSTrainingConsumer, contracted_plane_powers, git_identity, optical_power_loss_from_powers, physical_plane_power
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, file_sha256, safe_torch_load, tensor_sha256

from scripts.research.lightgen import execute_lightgen_2d_real_ols_replay_conditioning_identity_gap_v1 as replay
from scripts.research.lightgen import execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1 as continuation


# Nature-figure Python backend contract: editable SVG text and stable PDF text.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
# Validator-visible aliases for the same editable-text contract: svg.fonttype='none'; pdf.fonttype=42.
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


TASK_ID = "ASSESS_STEP6000_CONDITIONAL_GENERATION_READINESS_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_ASSESS_STEP6000_CONDITIONAL_GENERATION_READINESS_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "STEP6000_CONDITIONAL_GENERATION_READINESS_RESOLVED"
DEFAULT_OUTPUT_DIR = Path("outputs/20260921_step6000_conditional_generation_readiness_v1/runs/attempt_01")

CHECKPOINT_STEPS = (3000, 3300, 6000)
SOURCE_ORDER = ("real_dev", "s0", "s1")
EVALUATOR_ORDER = ("E_S", "E_I", "E_C", "E_D")
CLASS_COUNT = 10
OLS_PORT_COUNT = 100
BATCH_SIZE = 10
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
VISUAL_REAL_PER_CLASS = 5
VISUAL_SAMPLED_PER_CLASS = 10
PANEL_PER_CLASS = 20
DUPLICATE_SCALE = 1.0e-3
FEATURE_EPS = 1.0e-12
LATENT_BLOCK = 64
FEATURE_BLOCK = 128

# These are preregistered interpretation thresholds, not tuned after the run.
SUBSTANTIAL_DELTA = 0.05
REAL_TRANSFER_RATIO = 0.75
REAL_TRANSFER_GAP = 0.20
STRONG_CONSENSUS = 0.60
STRONG_CONSENSUS_MIN_CLASS = 0.40
LOW_PAIRWISE_AGREEMENT = 0.55
SEVERE_NEAR_DUPLICATE_FRACTION = 0.50
SEVERE_DOMINANT_FRACTION = 0.80
SEVERE_VARIANCE_RATIO = 0.10
PLATEAU_TOTAL_DELTA = 0.05
PLATEAU_LATE_DELTA = 0.02

CONTINUATION_RUN = Path("outputs/20260920_three_evaluator_robust_semantic_continuation_v1/runs/attempt_02")
CONTINUATION_REPORT = CONTINUATION_RUN / "report.json"
CONTINUATION_REPORT_SHA256 = "1d195aad5307108509349c451aa48f7015b8e27a00bfb66c8e42a56d4c4a0f92"
STEP3300 = CONTINUATION_RUN / "scientific_step_003300.pt"
STEP3300_SHA256 = "bd322f5f6178ea201f6603877a2f63efbde99fd8d388109beee24aba2dcf154b"
STEP6000 = CONTINUATION_RUN / "scientific_step_006000.pt"
STEP6000_SHA256 = "3387e104f7d33e19051a23444b0dabf1eb387c5faca3ce7150bbd1a2212c9986"

SAMPLER_MODELS = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01/sampler_models.pt")
SAMPLER_MODELS_SHA256 = "baa654e56485c110a8262fed8036c2954cb93cce31f8ad412e60c671efa20a69"
SAMPLER_BANK = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01/candidate_latent_banks.pt")
SAMPLER_BANK_SHA256 = "5ad385e3f683c144c5c8f0464e2519a11e03dbbf9dc0ce9b91bb9b7170148419"

REPORT_SCHEMA = "step6000_conditional_generation_readiness_report.v1"
STATUS_SCHEMA = "step6000_conditional_generation_readiness_status.v1"
PREFLIGHT_SCHEMA = "step6000_conditional_generation_readiness_preflight.v1"
SELECTION_SCHEMA = "step6000_conditional_generation_readiness_fixed_selection.v1"


class ReadinessAuditError(RuntimeError):
    """Raised when an input authority or read-only audit invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessAuditError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _tensor_or_list(value: object, name: str, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    if isinstance(value, list):
        return torch.tensor(value, dtype=dtype)
    raise ReadinessAuditError(f"{name} must be a tensor or a list")


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_write_new_json(path, dict(payload), artifact_kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


def _summary(value: torch.Tensor, name: str) -> dict[str, Any]:
    flat = value.detach().cpu().to(torch.float64).reshape(-1)
    _finite(flat, name)
    _require(flat.numel() > 0, f"{name} is empty")
    q = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
    return {
        "count": int(flat.numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std_population": float(flat.std(unbiased=False)),
        "quantiles": {"0.01": float(q[0]), "0.5": float(q[1]), "0.99": float(q[2])},
    }


def _entropy(histogram: Sequence[int]) -> float:
    counts = torch.tensor(list(histogram), dtype=torch.float64)
    probabilities = counts / counts.sum().clamp_min(1.0)
    nonzero = probabilities[probabilities > 0.0]
    return float(-(nonzero * nonzero.log()).sum())


def _cartesian(z: torch.Tensor) -> torch.Tensor:
    _require(z.ndim == 2 and z.shape[1] == OLS_PORT_COUNT and z.is_complex(), "OLS must be complex [N,100]")
    result = torch.cat((z.real, z.imag), dim=1).to(torch.float64).contiguous()
    _finite(result, "Cartesian OLS")
    return result


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


def _module_digest(module: torch.nn.Module) -> str:
    return _state_digest({name: value.detach().cpu() for name, value in module.state_dict().items()})


def _class_major_positions(labels: torch.Tensor, per_class: int) -> torch.Tensor:
    positions: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() >= per_class, f"class {digit} has fewer than {per_class} rows")
        positions.append(matches[:per_class])
    return torch.cat(positions).to(torch.long).contiguous()


def _paired_bootstrap_difference(left_correct: torch.Tensor, right_correct: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    left_correct = left_correct.detach().cpu().to(torch.float64).reshape(-1)
    right_correct = right_correct.detach().cpu().to(torch.float64).reshape(-1)
    labels = labels.detach().cpu().to(torch.long).reshape(-1)
    _require(left_correct.shape == right_correct.shape == labels.shape, "paired bootstrap vectors differ")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    deltas: list[torch.Tensor] = []
    per_class: list[float] = []
    for digit in range(CLASS_COUNT):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() > 0, f"bootstrap class {digit} is empty")
        draws = torch.randint(int(matches.numel()), (BOOTSTRAP_REPLICATES, int(matches.numel())), generator=generator)
        left = left_correct.index_select(0, matches)[draws]
        right = right_correct.index_select(0, matches)[draws]
        deltas.append((left - right).mean(dim=1))
        per_class.append(float(left_correct.index_select(0, matches).mean() - right_correct.index_select(0, matches).mean()))
    macro = torch.stack(deltas).mean(dim=0)
    quantiles = torch.quantile(macro, torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64), interpolation="linear")
    return {
        "left_minus_right": "S1_minus_S0",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "class_stratified_paired": True,
        "delta_macro_recall": float(macro.mean()),
        "ci95": {"lower": float(quantiles[0]), "median": float(quantiles[1]), "upper": float(quantiles[2])},
        "per_class_delta": per_class,
    }


def _classification_report(logits: torch.Tensor, labels: torch.Tensor, *, label_source: str) -> tuple[dict[str, Any], torch.Tensor]:
    logits = logits.detach().cpu().to(torch.float32)
    labels = labels.detach().cpu().to(torch.long).reshape(-1)
    _require(logits.ndim == 2 and logits.shape == (labels.numel(), CLASS_COUNT), "classification logits shape changed")
    _finite(logits, "classification logits")
    predictions = logits.argmax(dim=1)
    values, _ = torch.topk(logits, k=2, dim=1)
    confidence = torch.softmax(logits, dim=1).amax(dim=1)
    margin = values[:, 0] - values[:, 1]
    base = continuation._class_metrics(labels, predictions, label_source=label_source)
    histogram = base["predicted_class_histogram"]
    base.update(
        {
            "top1_confidence": _summary(confidence, "top1 confidence"),
            "top1_top2_logit_margin": _summary(margin, "top1-top2 logit margin"),
            "normalized_prediction_entropy": _entropy(histogram) / math.log(float(CLASS_COUNT)),
            "dominant_fraction": max(histogram) / float(labels.numel()),
            "represented_class_count": int(sum(value > 0 for value in histogram)),
            "labels_consumed": True,
        }
    )
    return base, predictions


def _raw_validity(raw_means: torch.Tensor, raw_mins: torch.Tensor, raw_maxs: torch.Tensor, raw_stds: torch.Tensor, *, finite_count: int, zero_count: int, total_count: int, raw_min: float, raw_max: float, shape: Sequence[int]) -> dict[str, Any]:
    _require(total_count > 0, "raw validity has no samples")
    nonconstant = int((raw_stds > 0.0).sum().item())
    return {
        "finite": finite_count == total_count,
        "finite_fraction": finite_count / float(total_count),
        "nonconstant_fraction": nonconstant / float(total_count),
        "zero_image_count": zero_count,
        "zero_image_fraction": zero_count / float(total_count),
        "min": float(raw_min),
        "max": float(raw_max),
        "shape": list(shape),
        "intensity_mean_per_sample": _summary(raw_means, "raw per-sample mean"),
        "per_sample_min": _summary(raw_mins, "raw per-sample min"),
        "per_sample_max": _summary(raw_maxs, "raw per-sample max"),
        "per_sample_std": _summary(raw_stds, "raw per-sample std"),
    }


def _pairwise_summary(matrix: torch.Tensor, name: str) -> dict[str, Any]:
    value = matrix.detach().cpu().to(torch.float32)
    _require(value.ndim == 2 and value.shape[0] >= 2, f"{name} pairwise input is too small")
    _finite(value, name)
    distances = torch.pdist(value, p=2).to(torch.float64)
    _finite(distances, f"{name} pairwise distances")
    median = float(distances.median())
    threshold = max(1.0e-12, DUPLICATE_SCALE * median)
    singular_values = torch.linalg.svdvals(value - value.mean(dim=0, keepdim=True)).to(torch.float64)
    rank = int((singular_values > singular_values.max().clamp_min(1.0e-30) * 1.0e-3).sum().item())
    exact = int(torch.unique(value, dim=0).shape[0])
    return {
        "count": int(value.shape[0]),
        "feature_dimension": int(value.shape[1]),
        "pairwise_distance": {
            "min": float(distances.min()),
            "max": float(distances.max()),
            "mean": float(distances.mean()),
            "median": median,
        },
        "near_duplicate_threshold": threshold,
        "near_duplicate_threshold_definition": "max(1e-12, 1e-3 * median_pairwise_euclidean_distance)",
        "near_duplicate_pair_count": int((distances <= threshold).sum().item()),
        "near_duplicate_pair_fraction": float((distances <= threshold).to(torch.float64).mean()),
        "exact_unique_row_count": exact,
        "exact_duplicate_row_count": int(value.shape[0] - exact),
        "effective_rank": {"effective_rank": rank, "relative_tolerance": 1.0e-3, "nominal_rank": int(min(value.shape))},
    }


def _energy_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(torch.float64)
    right = right.to(torch.float64)
    return float(2.0 * torch.cdist(left, right, p=2).mean() - torch.cdist(left, left, p=2).mean() - torch.cdist(right, right, p=2).mean())


def _nearest_distances(query: torch.Tensor, reference: torch.Tensor, *, block: int = LATENT_BLOCK) -> torch.Tensor:
    query = query.detach().to(torch.float32)
    reference = reference.detach().to(torch.float32)
    _require(query.ndim == reference.ndim == 2 and query.shape[1] == reference.shape[1], "nearest-distance geometry changed")
    _finite(query, "nearest query")
    _finite(reference, "nearest reference")
    ref_sq = reference.square().sum(dim=1)
    result: list[torch.Tensor] = []
    reference_t = reference.transpose(0, 1).contiguous()
    for start in range(0, query.shape[0], block):
        part = query[start : start + block]
        distances_sq = part.square().sum(dim=1, keepdim=True) + ref_sq.unsqueeze(0) - 2.0 * part.matmul(reference_t)
        distances_sq.clamp_min_(0.0)
        result.append(distances_sq.min(dim=1).values.sqrt().cpu())
    return torch.cat(result).to(torch.float64)


def _nearest_indices_by_class(query_features: torch.Tensor, requested: torch.Tensor, reference_features: torch.Tensor, reference_labels: torch.Tensor) -> torch.Tensor:
    result = torch.empty(query_features.shape[0], dtype=torch.long)
    for digit in range(CLASS_COUNT):
        query_positions = torch.nonzero(requested == digit, as_tuple=False).flatten()
        ref_positions = torch.nonzero(reference_labels == digit, as_tuple=False).flatten()
        _require(query_positions.numel() > 0 and ref_positions.numel() > 0, f"nearest-reference class {digit} is empty")
        query = query_features.index_select(0, query_positions).to(torch.float32)
        reference = reference_features.index_select(0, ref_positions).to(torch.float32)
        distances_sq = query.square().sum(dim=1, keepdim=True) + reference.square().sum(dim=1).unsqueeze(0) - 2.0 * query.matmul(reference.transpose(0, 1))
        distances_sq.clamp_min_(0.0)
        result[query_positions] = ref_positions[distances_sq.argmin(dim=1)]
    return result


def _pairwise_agreement(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    left = left.to(torch.long).reshape(-1)
    right = right.to(torch.long).reshape(-1)
    observed = float((left == right).to(torch.float64).mean())
    table = torch.zeros((CLASS_COUNT, CLASS_COUNT), dtype=torch.float64)
    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        table[a, b] += 1.0
    table /= table.sum().clamp_min(1.0)
    expected = float((table.sum(dim=0) * table.sum(dim=1)).sum())
    kappa = (observed - expected) / max(1.0 - expected, 1.0e-12)
    return observed, kappa


def _mode_with_count(predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.stack([torch.bincount(row, minlength=CLASS_COUNT) for row in predictions], dim=0)
    maximum, modes = counts.max(dim=1)
    return modes, maximum


def _consensus_report(predictions: Mapping[str, torch.Tensor], labels: torch.Tensor) -> dict[str, Any]:
    labels = labels.detach().cpu().to(torch.long)
    matrix = torch.stack([predictions[name].detach().cpu().to(torch.long) for name in EVALUATOR_ORDER], dim=1)
    _require(matrix.shape == (labels.numel(), len(EVALUATOR_ORDER)), "consensus matrix shape changed")
    mode, mode_count = _mode_with_count(matrix)
    exact_agreement = mode_count == 4
    ge3 = mode_count >= 3
    ge2 = mode_count >= 2
    pairwise: dict[str, dict[str, float]] = {}
    agreement_matrix: list[list[float]] = []
    kappa_matrix: list[list[float]] = []
    for left_index, left_name in enumerate(EVALUATOR_ORDER):
        agreement_row: list[float] = []
        kappa_row: list[float] = []
        for right_index, right_name in enumerate(EVALUATOR_ORDER):
            if left_index == right_index:
                agreement, kappa = 1.0, 1.0
            else:
                agreement, kappa = _pairwise_agreement(matrix[:, left_index], matrix[:, right_index])
            agreement_row.append(agreement)
            kappa_row.append(kappa)
            if left_index < right_index:
                pairwise[f"{left_name}_vs_{right_name}"] = {"agreement": agreement, "cohen_kappa": kappa}
        agreement_matrix.append(agreement_row)
        kappa_matrix.append(kappa_row)
    class_conditioned: dict[str, Any] = {}
    for digit in range(CLASS_COUNT):
        rows = labels == digit
        class_conditioned[str(digit)] = {
            "rows": int(rows.sum()),
            "4_of_4_agreement": float(exact_agreement[rows].to(torch.float64).mean()),
            "at_least_3_of_4_agreement": float(ge3[rows].to(torch.float64).mean()),
            "at_least_2_of_4_agreement": float(ge2[rows].to(torch.float64).mean()),
            "4_of_4_true_or_requested": float((matrix[rows] == labels[rows].unsqueeze(1)).all(dim=1).to(torch.float64).mean()),
            "at_least_3_of_4_true_or_requested": float(((matrix[rows] == labels[rows].unsqueeze(1)).sum(dim=1) >= 3).to(torch.float64).mean()),
        }
    return {
        "rows": int(labels.numel()),
        "label_source": "true class for real DEV; requested class for S0/S1",
        "4_of_4_agreement_fraction": float(exact_agreement.to(torch.float64).mean()),
        "at_least_3_of_4_agreement_fraction": float(ge3.to(torch.float64).mean()),
        "at_least_2_of_4_agreement_fraction": float(ge2.to(torch.float64).mean()),
        "4_of_4_true_or_requested_fraction": float((matrix == labels.unsqueeze(1)).all(dim=1).to(torch.float64).mean()),
        "at_least_3_of_4_true_or_requested_fraction": float(((matrix == labels.unsqueeze(1)).sum(dim=1) >= 3).to(torch.float64).mean()),
        "pairwise_prediction_agreement_matrix": agreement_matrix,
        "pairwise_cohen_kappa_matrix": kappa_matrix,
        "pairwise_prediction_agreement": pairwise,
        "class_conditioned": class_conditioned,
        "consensus_mode": mode.tolist(),
        "consensus_mode_count": mode_count.tolist(),
    }


def _transition_categories(before: torch.Tensor, after: torch.Tensor, labels: torch.Tensor, *, label_source: str, valid_mask: torch.Tensor | None = None) -> dict[str, Any]:
    before = before.detach().cpu().to(torch.long)
    after = after.detach().cpu().to(torch.long)
    labels = labels.detach().cpu().to(torch.long)
    _require(before.shape == after.shape == labels.shape, "transition vectors differ")
    if valid_mask is None:
        valid_mask = torch.ones_like(labels, dtype=torch.bool)
    correct_before = before == labels
    correct_after = after == labels
    same_wrong = (~correct_before) & (~correct_after) & (before == after)
    different_wrong = (~correct_before) & (~correct_after) & (before != after)
    stable_correct = correct_before & correct_after
    wrong_to_correct = (~correct_before) & correct_after
    correct_to_wrong = correct_before & (~correct_after)
    counts = {
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_different_wrong": different_wrong,
        "stable_correct": stable_correct,
        "stable_wrong": same_wrong,
    }
    used = valid_mask.to(torch.float64).sum().clamp_min(1.0)
    return {
        "label_source": label_source,
        "rows": int(valid_mask.sum()),
        "excluded_without_consensus": int((~valid_mask).sum()),
        "categories": {name: {"count": int(value[valid_mask].sum()), "fraction_of_valid": float(value[valid_mask].to(torch.float64).sum() / used)} for name, value in counts.items()},
    }


def _corrected_overlap(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor], labels: torch.Tensor) -> dict[str, Any]:
    corrected: dict[str, set[int]] = {}
    labels = labels.cpu().to(torch.long)
    for name in EVALUATOR_ORDER:
        left = before[name].cpu().to(torch.long)
        right = after[name].cpu().to(torch.long)
        corrected[name] = set(torch.nonzero((left != labels) & (right == labels), as_tuple=False).flatten().tolist())
    pairwise: dict[str, Any] = {}
    for left_index, left_name in enumerate(EVALUATOR_ORDER):
        for right_name in EVALUATOR_ORDER[left_index + 1 :]:
            left_set = corrected[left_name]
            right_set = corrected[right_name]
            union = left_set | right_set
            pairwise[f"{left_name}_vs_{right_name}"] = {
                "left_count": len(left_set),
                "right_count": len(right_set),
                "intersection_count": len(left_set & right_set),
                "jaccard": len(left_set & right_set) / float(len(union)) if union else 1.0,
            }
    intersection = set.intersection(*(corrected[name] for name in EVALUATOR_ORDER))
    return {
        "corrected_counts": {name: len(value) for name, value in corrected.items()},
        "pairwise_overlap": pairwise,
        "all_four_intersection_count": len(intersection),
    }


def _transition_report(predictions: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]], labels_by_source: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in SOURCE_ORDER:
        labels = labels_by_source[source]
        source_result: dict[str, Any] = {}
        for left_step, right_step in ((3000, 3300), (3300, 6000)):
            before = predictions[source][left_step]
            after = predictions[source][right_step]
            interval = f"{left_step}_to_{right_step}"
            evaluator = {name: _transition_categories(before[name], after[name], labels, label_source="true class" if source == "real_dev" else "requested class") for name in EVALUATOR_ORDER}
            before_matrix = torch.stack([before[name] for name in EVALUATOR_ORDER], dim=1)
            after_matrix = torch.stack([after[name] for name in EVALUATOR_ORDER], dim=1)
            before_mode, before_count = _mode_with_count(before_matrix)
            after_mode, after_count = _mode_with_count(after_matrix)
            consensus = _transition_categories(before_mode, after_mode, labels, label_source="consensus mode", valid_mask=(before_count >= 3) & (after_count >= 3))
            source_result[interval] = {
                "per_evaluator": evaluator,
                "consensus_mode": consensus,
                "corrected_sample_overlap": _corrected_overlap(before, after, labels),
            }
        result[source] = source_result
    return result


def _raw_panel_metrics(payload: Mapping[str, Any], authorities: Mapping[str, Any], control_features: torch.Tensor) -> dict[str, Any]:
    source = str(payload["source"])
    if source == "real_dev":
        return {"R_raw": None, "R_feature": None, "role": "not defined for real DEV path"}
    panel_raw = _tensor(payload["panel_raw"], f"{source} raw panel").to(torch.float64).flatten(1)
    control_raw = _tensor(authorities["s1_panels"]["real_validation_control_raw_panel"], "real validation control raw panel").to(torch.float64).flatten(1)
    _require(panel_raw.shape == control_raw.shape == (200, 40000), "R panel geometry changed")
    mean = control_raw.mean(dim=0)
    std = control_raw.std(dim=0, unbiased=False).clamp_min(1.0e-8)
    generated_standard = (panel_raw - mean) / std
    control_standard = (control_raw - mean) / std
    raw_cross = _energy_distance(generated_standard, control_standard)
    raw_self = _energy_distance(control_standard[:100], control_standard[100:])
    panel_indices = _tensor(payload["panel_positions"], "payload panel positions").to(torch.long).cpu()
    panel_features = _tensor(payload["features"]["E_C"], f"{source} E_C features").index_select(0, panel_indices)
    feature_current = panel_features.to(torch.float64)
    feature_control = control_features.to(torch.float64)
    feature_cross = _energy_distance(feature_current, feature_control)
    feature_self = _energy_distance(feature_control[:100], feature_control[100:])
    return {
        "R_raw": raw_cross / max(raw_self, 1.0e-12),
        "R_feature": feature_cross / max(feature_self, 1.0e-12),
        "R_raw_details": {"cross": raw_cross, "self": raw_self, "control_role": "fixed real validation control panel from accepted S1 canary"},
        "R_feature_details": {"cross": feature_cross, "self": feature_self, "feature_space": "frozen E_C embedding"},
    }


def _evaluate_source(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer, evaluators: Mapping[str, torch.nn.Module], source: str, step: int, inputs: torch.Tensor, labels: torch.Tensor, fixed_positions: torch.Tensor, panel_positions: torch.Tensor, device: torch.device, references: Any) -> dict[str, Any]:
    _require(source in SOURCE_ORDER, f"unknown source {source}")
    inputs = inputs.cpu().contiguous()
    labels = labels.cpu().to(torch.long).contiguous()
    fixed_positions = fixed_positions.cpu().to(torch.long).contiguous()
    panel_positions = panel_positions.cpu().to(torch.long).contiguous()
    _require(inputs.shape[0] == labels.numel(), f"{source} input/label count changed")
    position_to_fixed = torch.full((inputs.shape[0],), -1, dtype=torch.long)
    position_to_fixed[fixed_positions] = torch.arange(fixed_positions.numel(), dtype=torch.long)
    position_to_panel = torch.full((inputs.shape[0],), -1, dtype=torch.long)
    position_to_panel[panel_positions] = torch.arange(panel_positions.numel(), dtype=torch.long)
    fixed_view = torch.empty((fixed_positions.numel(), 1, 28, 28), dtype=torch.float32)
    panel_raw = torch.empty((panel_positions.numel(), 200, 200), dtype=torch.float32)
    class_sum = torch.zeros((CLASS_COUNT, 28 * 28), dtype=torch.float64)
    class_count = torch.zeros(CLASS_COUNT, dtype=torch.long)
    logits_lists: dict[str, list[torch.Tensor]] = {name: [] for name in EVALUATOR_ORDER}
    feature_lists: dict[str, list[torch.Tensor]] = {name: [] for name in ("E_S", "E_I", "E_C")}
    sample_means: list[torch.Tensor] = []
    sample_mins: list[torch.Tensor] = []
    sample_maxs: list[torch.Tensor] = []
    sample_stds: list[torch.Tensor] = []
    power_means: list[torch.Tensor] = []
    power_finite_count = 0
    power_nonnegative_count = 0
    power_total_count = 0
    power_plane_count: int | None = None
    power_role = ""
    power_min = math.inf
    power_max = -math.inf
    finite_count = 0
    zero_count = 0
    raw_min = math.inf
    raw_max = -math.inf
    rec_sum = 0.0
    op_sum = 0.0
    processed = 0
    consumer.runtime.eval()
    for model in evaluators.values():
        model.eval()
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], BATCH_SIZE):
            end = min(inputs.shape[0], start + BATCH_SIZE)
            batch = inputs[start:end].to(device)
            if source == "real_dev":
                result = consumer.runtime.forward_from_image(batch)
            else:
                result = consumer.generated_forward(batch)
            raw = result.raw_intensity.detach()
            _finite(raw, f"{source} step{step} raw CMOS")
            if source == "real_dev":
                powers = contracted_plane_powers(result).detach()
                power_role = "accepted BOGT five-plane power validity"
            else:
                direct_fields = (result.generator_input_field, *result.generator_records)
                _require(len(direct_fields) == 3, "direct generated path plane geometry changed")
                powers = torch.stack([physical_plane_power(field) for field in direct_fields], dim=1).detach()
                power_role = "direct generated OLS path three-plane power validity; five-plane BOGT reference is not defined for encoder-free replay"
            _require(powers.ndim == 2 and powers.shape[0] == end - start, "contracted power geometry changed")
            if power_plane_count is None:
                power_plane_count = int(powers.shape[1])
            _require(int(powers.shape[1]) == power_plane_count, "power plane count changed within source")
            powers_cpu = powers.cpu().to(torch.float64)
            power_means.append(powers_cpu.mean(dim=1))
            power_finite_count += int(torch.isfinite(powers_cpu).all(dim=1).sum())
            power_nonnegative_count += int((powers_cpu >= 0.0).all(dim=1).sum())
            power_total_count += end - start
            power_min = min(power_min, float(powers_cpu.min()))
            power_max = max(power_max, float(powers_cpu.max()))
            view = continuation._cmos_view(raw).detach()
            evaluator_values: dict[str, tuple[torch.Tensor | None, torch.Tensor]] = {}
            for name in ("E_S", "E_I", "E_C"):
                features, logits = evaluators[name](view)
                evaluator_values[name] = (features, logits)
                feature_lists[name].append(features.detach().cpu())
                logits_lists[name].append(logits.detach().cpu())
            d_logits = evaluators["E_D"](view)
            evaluator_values["E_D"] = (None, d_logits)
            logits_lists["E_D"].append(d_logits.detach().cpu())
            raw_cpu = raw.detach().cpu()
            view_cpu = view.detach().cpu()
            flat = raw_cpu.to(torch.float64).flatten(1)
            sample_means.append(flat.mean(dim=1))
            sample_mins.append(flat.min(dim=1).values)
            sample_maxs.append(flat.max(dim=1).values)
            sample_stds.append(flat.std(dim=1, unbiased=False))
            finite_count += int(torch.isfinite(flat).all(dim=1).sum())
            zero_count += int((flat.abs().sum(dim=1) == 0.0).sum())
            raw_min = min(raw_min, float(raw_cpu.min()))
            raw_max = max(raw_max, float(raw_cpu.max()))
            class_sum.index_add_(0, labels[start:end], view_cpu.flatten(1).to(torch.float64))
            class_count += torch.bincount(labels[start:end], minlength=CLASS_COUNT)
            fixed_slots = position_to_fixed[start:end]
            fixed_mask = fixed_slots >= 0
            if fixed_mask.any():
                fixed_view[fixed_slots[fixed_mask]] = view_cpu[fixed_mask]
            panel_slots = position_to_panel[start:end]
            panel_mask = panel_slots >= 0
            if panel_mask.any():
                panel_raw[panel_slots[panel_mask]] = raw_cpu[panel_mask, 0]
            if source == "real_dev":
                target = result.image_input.slm_intensity
                rec_sum += float((raw - target).square().mean().cpu()) * (end - start)
                op_value = optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)
                op_sum += float(op_value.cpu()) * (end - start)
            processed += end - start
            if processed % 500 == 0 or processed == inputs.shape[0]:
                print(json.dumps({"phase": "FORWARD", "source": source, "step": step, "processed": processed, "total": int(inputs.shape[0])}, ensure_ascii=False), flush=True)
    logits = {name: torch.cat(values, dim=0).contiguous() for name, values in logits_lists.items()}
    features = {name: torch.cat(values, dim=0).contiguous() for name, values in feature_lists.items()}
    metrics: dict[str, Any] = {}
    predictions: dict[str, torch.Tensor] = {}
    label_source = "true class from accepted real DEV labels" if source == "real_dev" else "requested class supplied by frozen sample(y, seed_or_index) contract"
    for name in EVALUATOR_ORDER:
        metrics[name], predictions[name] = _classification_report(logits[name], labels, label_source=label_source)
    means = torch.cat(sample_means)
    mins = torch.cat(sample_mins)
    maxs = torch.cat(sample_maxs)
    stds = torch.cat(sample_stds)
    validity = _raw_validity(means, mins, maxs, stds, finite_count=finite_count, zero_count=zero_count, total_count=processed, raw_min=raw_min, raw_max=raw_max, shape=(processed, 1, 200, 200))
    power_mean_values = torch.cat(power_means)
    power_validity = {
        "finite": power_finite_count == power_total_count,
        "finite_fraction": power_finite_count / float(power_total_count),
        "nonnegative_fraction": power_nonnegative_count / float(power_total_count),
        "min": power_min,
        "max": power_max,
        "sample_mean": _summary(power_mean_values, f"{source} contracted power sample mean"),
        "shape": [processed, int(power_plane_count or 0)],
        "role": power_role,
    }
    class_means = (class_sum / class_count.clamp_min(1).unsqueeze(1)).reshape(CLASS_COUNT, 1, 28, 28).to(torch.float32)
    return {
        "source": source,
        "step": step,
        "labels": labels,
        "logits": logits,
        "features": features,
        "predictions": predictions,
        "metrics": metrics,
        "raw_validity": validity,
        "power_validity": power_validity,
        "fixed_view": fixed_view,
        "panel_raw": panel_raw,
        "panel_positions": panel_positions,
        "class_means": class_means,
        "losses": {"L_rec": rec_sum / float(processed) if source == "real_dev" else None, "L_op": op_sum / float(processed) if source == "real_dev" else None},
    }


def _latent_novelty(root: Path, replay_authorities: Mapping[str, Any], sampler_models: Mapping[str, Any]) -> dict[str, Any]:
    train_z = _tensor(replay_authorities["sampler_inputs"]["train_z"], "real TRAIN OLS").cpu().contiguous()
    train_labels = _tensor(replay_authorities["sampler_inputs"]["train_labels"], "real TRAIN labels").cpu().to(torch.long).contiguous()
    candidate = replay_authorities["candidate_z"]
    requested = replay_authorities["requested"]
    s1_sampling = _mapping(_mapping(sampler_models.get("S1"), "S1 sampler model").get("sampling"), "S1 sampling metadata")
    anchors = _tensor_or_list(s1_sampling.get("anchor_train_positions"), "S1 anchor positions", dtype=torch.long).cpu()
    neighbors = _tensor_or_list(s1_sampling.get("neighbor_train_positions"), "S1 neighbor positions", dtype=torch.long).cpu()
    lambdas = _tensor_or_list(s1_sampling.get("lambda"), "S1 interpolation coefficients", dtype=torch.float64).cpu()
    _require(anchors.shape == neighbors.shape == lambdas.shape == requested.shape == (2000,), "S1 novelty metadata shape changed")
    standardization = _mapping(_mapping(sampler_models["S1"], "S1 sampler model").get("standardization"), "S1 standardization")
    mean = _tensor(standardization.get("mean"), "S1 standardization mean").to(torch.float64).cpu()
    std = _tensor(standardization.get("std"), "S1 standardization std").to(torch.float64).cpu()
    train_cart = _cartesian(train_z)
    train_x = (train_cart - mean) / std
    train_by_class = [train_x[train_labels == digit].contiguous() for digit in range(CLASS_COUNT)]
    result: dict[str, Any] = {
        "coordinate": "global SAMPLER_FIT-standardized Cartesian [Re(z), Im(z)] in R^200",
        "train_rows": int(train_z.shape[0]),
        "S0": {},
        "S1": {},
        "s1_interpolation": {"lambda": _summary(lambdas, "S1 lambda"), "rule": str(s1_sampling["interpolation_rule"])},
    }
    for family in ("S0", "S1"):
        z = _tensor(candidate[family], f"{family} bank").cpu().contiguous()
        x = (_cartesian(z) - mean) / std
        if family == "S0":
            same_class = torch.zeros(z.shape[0], dtype=torch.float64)
        else:
            same_class_distances: list[torch.Tensor] = []
            for digit in range(CLASS_COUNT):
                rows = torch.nonzero(requested == digit, as_tuple=False).flatten()
                same_class_distances.append(_nearest_distances(x.index_select(0, rows), train_by_class[digit]))
            same_class = torch.cat(same_class_distances)
        collisions = int(sum(any(torch.equal(row, train_row) for train_row in train_z) for row in z[:0]))
        if family == "S0":
            collisions = int(z.shape[0])
            anchor_distance = torch.zeros(z.shape[0], dtype=torch.float64)
            neighbor_distance = torch.zeros(z.shape[0], dtype=torch.float64)
        else:
            anchor_z = train_z.index_select(0, anchors)
            neighbor_z = train_z.index_select(0, neighbors)
            anchor_distance = ((_cartesian(z) - _cartesian(anchor_z)) / std).norm(dim=1)
            neighbor_distance = ((_cartesian(z) - _cartesian(neighbor_z)) / std).norm(dim=1)
            train_keys = {row.numpy().tobytes() for row in train_z.contiguous()}
            collisions = sum(row.numpy().tobytes() in train_keys for row in z.contiguous())
        result[family] = {
            "rows": int(z.shape[0]),
            "same_class_nearest_train_distance": _summary(same_class, f"{family} same-class nearest TRAIN distance"),
            "distance_to_anchor": _summary(anchor_distance, f"{family} anchor distance"),
            "distance_to_neighbor": _summary(neighbor_distance, f"{family} neighbor distance"),
            "exact_train_row_collision_count": int(collisions),
            "exact_train_row_collision_fraction": float(collisions / z.shape[0]),
            "z_sha256": tensor_sha256(z),
        }
    result["S1"]["anchor_train_positions_sha256"] = tensor_sha256(anchors)
    result["S1"]["neighbor_train_positions_sha256"] = tensor_sha256(neighbors)
    return result


def _output_novelty(features: torch.Tensor, requested: torch.Tensor, target_features: torch.Tensor, target_labels: torch.Tensor) -> dict[str, Any]:
    nearest_all = _nearest_distances(features, target_features, block=FEATURE_BLOCK)
    nearest_positions = _nearest_indices_by_class(features, requested, target_features, target_labels)
    selected = target_features.index_select(0, nearest_positions)
    class_distance = (features.to(torch.float32) - selected.to(torch.float32)).norm(dim=1).to(torch.float64)
    return {
        "representation": "frozen E_C feature Euclidean distance on canonical DEV target images",
        "nearest_all_canonical_dev_distance": _summary(nearest_all, "nearest canonical DEV E_C distance"),
        "nearest_requested_class_canonical_dev_distance": _summary(class_distance, "nearest requested-class canonical DEV E_C distance"),
        "near_duplicate_threshold": "max(1e-12, 1e-3 * median pairwise distance in the fixed E_C feature bank)",
        "exact_or_near_duplicate_count": int((class_distance <= max(1.0e-12, DUPLICATE_SCALE * float(class_distance.median()))).sum()),
        "nearest_reference_positions": nearest_positions,
    }


def _diversity(source: str, payload: Mapping[str, Any], *, latent: torch.Tensor | None = None) -> dict[str, Any]:
    labels = _tensor(payload["labels"], f"{source} labels").to(torch.long)
    features = _tensor(payload["features"]["E_C"], f"{source} E_C features").to(torch.float32)
    fixed_view = _tensor(payload["fixed_view"], f"{source} fixed view").flatten(1)
    result: dict[str, Any] = {"source": source, "class_conditioned": {}}
    for digit in range(CLASS_COUNT):
        rows = torch.nonzero(labels == digit, as_tuple=False).flatten()
        class_features = features.index_select(0, rows)
        class_views = fixed_view.index_select(0, torch.nonzero(payload["fixed_labels"] == digit, as_tuple=False).flatten()) if "fixed_labels" in payload else fixed_view[:0]
        feature_variance: dict[str, float] = {}
        feature_pairwise: dict[str, Any] = {}
        for feature_name in ("E_S", "E_I", "E_C"):
            feature_value = _tensor(payload["features"][feature_name], f"{source} {feature_name} features").index_select(0, rows)
            feature_variance[feature_name] = float(feature_value.to(torch.float64).var(dim=0, unbiased=False).mean())
            feature_pairwise[feature_name] = _pairwise_summary(feature_value[: min(200, feature_value.shape[0])], f"{source} {feature_name} class fixed subset")
        class_result: dict[str, Any] = {
            "rows": int(rows.numel()),
            "feature_variance_mean": feature_variance,
            "feature_pairwise_fixed_subset": feature_pairwise,
            "E_C_feature_variance_mean": feature_variance["E_C"],
            "E_C_pairwise_fixed_subset": feature_pairwise["E_C"],
        }
        if class_views.shape[0] >= 2:
            class_result["pixel_variance_mean_28x28_fixed_subset"] = float(class_views.to(torch.float64).var(dim=0, unbiased=False).mean())
            class_result["pixel_pairwise_fixed_subset"] = _pairwise_summary(class_views, "pixel class fixed subset")
        result["class_conditioned"][str(digit)] = class_result
    result["aggregate"] = {
        "E_C_feature": _pairwise_summary(features, f"{source} E_C feature"),
        "raw_cmos_metric_panel_28x28": _pairwise_summary(_tensor(payload["panel_view"], f"{source} panel view").flatten(1), f"{source} panel CMOS view"),
    }
    if latent is not None:
        result["latent_z"] = _pairwise_summary(_cartesian(latent), f"{source} latent z")
    return result


def _real_vs_s1(records: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in CHECKPOINT_STEPS:
        step_result: dict[str, Any] = {}
        for name in EVALUATOR_ORDER:
            real_rate = float(records["real_dev"][step]["metrics"][name]["macro_recall"])
            s1_rate = float(records["s1"][step]["metrics"][name]["macro_recall"])
            step_result[name] = {"R_real": real_rate, "R_S1": s1_rate, "R_S1_over_R_real": s1_rate / max(real_rate, 1.0e-12), "R_S1_minus_R_real": s1_rate - real_rate}
        step_result["consensus"] = {
            "real_at_least_3_of_4_true_fraction": records["real_dev"][step]["consensus"]["at_least_3_of_4_true_or_requested_fraction"],
            "s1_at_least_3_of_4_requested_fraction": records["s1"][step]["consensus"]["at_least_3_of_4_true_or_requested_fraction"],
            "difference_s1_minus_real": records["s1"][step]["consensus"]["at_least_3_of_4_true_or_requested_fraction"] - records["real_dev"][step]["consensus"]["at_least_3_of_4_true_or_requested_fraction"],
        }
        result[str(step)] = step_result
    return result


def _s0_vs_s1(records: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in CHECKPOINT_STEPS:
        step_result: dict[str, Any] = {}
        for name in EVALUATOR_ORDER:
            s0_predictions = records["s0"][step]["predictions"][name]
            s1_predictions = records["s1"][step]["predictions"][name]
            labels = records["s1"][step]["labels"]
            step_result[name] = {
                "S0": records["s0"][step]["metrics"][name],
                "S1": records["s1"][step]["metrics"][name],
                "S1_minus_S0_paired_bootstrap": _paired_bootstrap_difference(s1_predictions == labels, s0_predictions == labels, labels),
            }
        step_result["consensus"] = {
            "S0": records["s0"][step]["consensus"],
            "S1": records["s1"][step]["consensus"],
        }
        result[str(step)] = step_result
    return result


def _trajectory_report(continuation_report: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = _mapping(continuation_report.get("training_trajectory"), "continuation trajectory")
    trace = trajectory.get("trace")
    _require(isinstance(trace, list) and trace, "continuation trajectory trace is missing")
    by_step: dict[int, Mapping[str, Any]] = {}
    for item in trace:
        mapping = _mapping(item, "trajectory record")
        by_step[int(mapping["step"])] = mapping
    stage2 = _mapping(continuation_report.get("stage2"), "continuation stage2")
    selected: dict[str, Mapping[str, Any]] = {}
    selected["3000"] = _mapping(_mapping(continuation_report.get("stage1"), "continuation stage1").get("trajectory", {}).get("3000"), "step3000 trajectory")
    for step in (3300, 3600, 4000, 4500, 5000, 5500, 6000):
        candidate = stage2.get("trajectory", {}).get(str(step))
        if candidate is None:
            candidate = by_step.get(step)
        _require(candidate is not None, f"late trajectory step {step} is missing")
        selected[str(step)] = _mapping(candidate, f"step{step} trajectory")
    fields = ("L_S", "L_I", "L_C", "L_rec", "L_op", "phase_update_rms")
    compact: dict[str, Any] = {}
    for step, item in selected.items():
        gradient_norms = _mapping(item.get("gradient_norms"), f"step{step} gradient norms")
        compact[step] = {field: float(item[field]) for field in fields if field in item}
        compact[step]["g_sem_norm"] = float(_mapping(gradient_norms.get("g_sem"), "g_sem gradient")["norm"])
        compact[step]["v_SIC_norm"] = float(_mapping(gradient_norms.get("v_SIC"), "v_SIC gradient")["norm"])
        compact[step]["phase_displacement_from_step3000"] = float(item.get("phase_displacement_from_step3000", 0.0))
    loss_fields = ("L_S", "L_I", "L_C", "L_rec")
    deltas = {field: float(compact["6000"][field] - compact["3300"][field]) for field in loss_fields}
    late_deltas = {field: float(compact["6000"][field] - compact["5000"][field]) for field in loss_fields}
    mean_total = float(np.mean([compact["3300"][field] - compact["6000"][field] for field in loss_fields]))
    mean_late = float(np.mean([compact["5000"][field] - compact["6000"][field] for field in loss_fields]))
    if any(not math.isfinite(value) for value in list(deltas.values()) + list(late_deltas.values())):
        trend = "unstable"
    elif mean_total >= PLATEAU_TOTAL_DELTA and mean_late >= PLATEAU_LATE_DELTA:
        trend = "still_clearly_improving"
    elif mean_total < 0.0 or mean_late < 0.0:
        trend = "unstable"
    else:
        trend = "flattening"
    return {
        "selected_steps": compact,
        "delta_step6000_minus_step3300": deltas,
        "delta_step6000_minus_step5000": late_deltas,
        "mean_loss_decrease_3300_to_6000": mean_total,
        "mean_loss_decrease_5000_to_6000": mean_late,
        "trend": trend,
        "rule": {"still_clearly_improving": f"mean decrease across L_S/L_I/L_C/L_rec >= {PLATEAU_TOTAL_DELTA} overall and >= {PLATEAU_LATE_DELTA} in step5000->6000 window; no asymptotic fit", "flattening": "nonnegative but below one or both preregistered decreases", "unstable": "nonfinite or mean loss reversal"},
    }


def _selection_manifest(root: Path, images: torch.Tensor, labels: torch.Tensor, dev_indices: torch.Tensor, requested: torch.Tensor, panel_indices: torch.Tensor) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    dev_labels = labels.index_select(0, dev_indices)
    real_positions = _class_major_positions(dev_labels, VISUAL_REAL_PER_CLASS)
    sampled_positions = _class_major_positions(requested, VISUAL_SAMPLED_PER_CLASS)
    real_panel_positions = _class_major_positions(dev_labels, PANEL_PER_CLASS)
    payload = {
        "schema": SELECTION_SCHEMA,
        "selection_frozen_before_forward": True,
        "real_dev": {"per_class": VISUAL_REAL_PER_CLASS, "positions_in_sorted_dev": real_positions.tolist(), "canonical_source_indices": dev_indices.index_select(0, real_positions).tolist(), "labels": dev_labels.index_select(0, real_positions).tolist()},
        "sampled": {"per_class": VISUAL_SAMPLED_PER_CLASS, "positions_in_bank": sampled_positions.tolist(), "requested_labels": requested.index_select(0, sampled_positions).tolist()},
        "accepted_R_panel": {"sampled_bank_positions": panel_indices.tolist(), "per_class": PANEL_PER_CLASS},
        "fixed_display": {"domain": "[0,1]", "normalization": "same global vmin=0/vmax=1 for every comparable CMOS panel", "per_image_contrast_stretch": False, "cmap": "gray", "nearest_reference_metric": "class-restricted Euclidean distance in frozen E_C feature space to canonical DEV target features; ties choose smallest DEV position", "nearest_reference_is_visual_only": True},
    }
    return payload, {"real_positions": real_positions, "sampled_positions": sampled_positions, "real_panel_positions": real_panel_positions}


def _add_panel_label(axis: Any, label: str) -> None:
    offset = ScaledTranslation(-3 / 72, 3 / 72, axis.figure.dpi_scale_trans)
    axis.text(0, 1, label, transform=axis.transAxes + offset, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _mosaic(values: torch.Tensor, rows: int, columns: int) -> np.ndarray:
    tensor = values.detach().cpu().to(torch.float32)
    _require(tensor.ndim == 4 and tensor.shape[1] == 1 and tensor.shape[0] == rows * columns, "mosaic geometry changed")
    array = tensor[:, 0].numpy()
    return array.reshape(rows, columns, array.shape[1], array.shape[2]).transpose(0, 2, 1, 3).reshape(rows * array.shape[1], columns * array.shape[2])


def _class_means(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    result: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        rows = images[labels == digit]
        _require(rows.shape[0] > 0, f"class {digit} has no images for class mean")
        result.append(rows.to(torch.float64).mean(dim=0).to(torch.float32))
    return torch.stack(result).contiguous()


def _save_figure(fig: Any, target: Path, stem: str, axes: Sequence[Any], panel_ids: Sequence[str], *, row_groups: Sequence[Sequence[str]] | None = None, column_groups: Sequence[Sequence[str]] | None = None) -> dict[str, Any]:
    fig.tight_layout(pad=1.0, rect=(0.0, 0.03, 1.0, 0.97))
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError as error:  # pragma: no cover - runtime environment contract
        raise ReadinessAuditError("nature-figure panel alignment helper is unavailable") from error
    options: dict[str, Any] = {"axes": list(axes), "panel_ids": list(panel_ids), "tolerance_pt": 1.5, "gutter_tolerance_pt": 1.5, "require_panel_labels": True, "strict": True}
    if row_groups is not None:
        options["row_groups"] = [list(group) for group in row_groups]
    if column_groups is not None:
        options["column_groups"] = [list(group) for group in column_groups]
    require_matplotlib_panel_alignment(fig, json_out=str(target / f"{stem}.alignment.json"), overlay_svg=str(target / f"{stem}.alignment.svg"), **options)
    outputs: dict[str, Any] = {}
    svg_path = target / f"{stem}.svg"
    pdf_path = target / f"{stem}.pdf"
    png_path = target / f"{stem}.png"
    tiff_path = target / f"{stem}.tiff"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    for suffix, path in (("svg", svg_path), ("pdf", pdf_path), ("png", png_path), ("tiff", tiff_path)):
        outputs[suffix] = {"path": path.name, "sha256": file_sha256(path)}
    outputs["alignment_json"] = {"path": f"{stem}.alignment.json", "sha256": file_sha256(target / f"{stem}.alignment.json")}
    outputs["alignment_svg"] = {"path": f"{stem}.alignment.svg", "sha256": file_sha256(target / f"{stem}.alignment.svg")}
    plt.close(fig)
    return outputs


def _render_real_visual(target: Path, dev_images: torch.Tensor, real_views: Mapping[int, torch.Tensor], selection: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    rows = CLASS_COUNT
    columns = VISUAL_REAL_PER_CLASS
    target_mosaic = _mosaic(dev_images.index_select(0, selection["real_positions"]), rows, columns)
    mosaics = [target_mosaic]
    for step in CHECKPOINT_STEPS:
        mosaics.append(_mosaic(real_views[step], rows, columns))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), squeeze=False)
    titles = ["target DEV", "step3000", "step3300", "step6000"]
    for index, (axis, mosaic, title) in enumerate(zip(axes[0], mosaics, titles, strict=True)):
        axis.imshow(mosaic, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_title(title, fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
        _add_panel_label(axis, "abcd"[index])
    fig.text(0.5, 0.005, "Rows are classes 0–9; five fixed DEV samples per class; global [0,1] display range.", ha="center", fontsize=6)
    return _save_figure(fig, target, "real_dev_fixed_visual_panel", list(axes[0]), list("abcd"))


def _render_s1_visual(target: Path, s1_views: Mapping[int, torch.Tensor], nearest_refs: Mapping[int, torch.Tensor], dev_images: torch.Tensor, selection: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    rows = CLASS_COUNT
    columns = VISUAL_SAMPLED_PER_CLASS
    axes_count = 6
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), squeeze=False)
    panel_ids = list("abcdef")
    axis_index = 0
    for step in CHECKPOINT_STEPS:
        generated = _mosaic(s1_views[step], rows, columns)
        reference_positions = nearest_refs[step]
        reference = _mosaic(dev_images.index_select(0, reference_positions), rows, columns)
        for title, mosaic in ((f"S1 step{step}", generated), (f"nearest DEV ref step{step}", reference)):
            axis = axes.flat[axis_index]
            axis.imshow(mosaic, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_title(title, fontsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
            _add_panel_label(axis, panel_ids[axis_index])
            axis_index += 1
    fig.text(0.5, 0.005, "Ten fixed S1 samples per requested class; references use preregistered frozen E_C nearest-class rule; [0,1] range.", ha="center", fontsize=6)
    return _save_figure(fig, target, "s1_fixed_visual_panel", list(axes.flat), panel_ids, row_groups=(("a", "b", "c"), ("d", "e", "f")), column_groups=(("a", "d"), ("b", "e"), ("c", "f")))


def _render_s0_s1_visual(target: Path, views: Mapping[str, Mapping[int, torch.Tensor]], selection: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    rows = CLASS_COUNT
    columns = VISUAL_SAMPLED_PER_CLASS
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), squeeze=False)
    panel_ids = list("abcdef")
    axis_index = 0
    for family in ("S0", "S1"):
        for step in CHECKPOINT_STEPS:
            axis = axes.flat[axis_index]
            axis.imshow(_mosaic(views[family][step], rows, columns), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_title(f"{family} step{step}", fontsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
            _add_panel_label(axis, panel_ids[axis_index])
            axis_index += 1
    fig.text(0.5, 0.005, "Same fixed requested-class bank positions for exact S0 replay and S1 interpolation; [0,1] range.", ha="center", fontsize=6)
    return _save_figure(fig, target, "s0_s1_fixed_visual_panel", list(axes.flat), panel_ids, row_groups=(("a", "b", "c"), ("d", "e", "f")), column_groups=(("a", "d"), ("b", "e"), ("c", "f")))


def _render_class_average(target: Path, real_means: Mapping[int, torch.Tensor], sampled_means: Mapping[str, Mapping[int, torch.Tensor]]) -> dict[str, Any]:
    figures: dict[str, Any] = {}
    real_target = real_means["target"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05), squeeze=False)
    values = [real_target] + [real_means[step] for step in CHECKPOINT_STEPS]
    titles = ["target DEV mean", "real step3000", "real step3300", "real step6000"]
    for index, (axis, value, title) in enumerate(zip(axes[0], values, titles, strict=True)):
        axis.imshow(value[:, 0].reshape(CLASS_COUNT, -1), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
        axis.set_title(title, fontsize=7)
        axis.set_xticks([])
        axis.set_yticks(range(CLASS_COUNT))
        axis.set_yticklabels([str(digit) for digit in range(CLASS_COUNT)], fontsize=5)
        _add_panel_label(axis, "abcd"[index])
    fig.text(0.5, 0.005, "Each row is one class mean; fixed global [0,1] normalization.", ha="center", fontsize=6)
    figures["real_class_average_panel"] = _save_figure(fig, target, "real_class_average_panel", list(axes[0]), list("abcd"))
    fig, axes = plt.subplots(2, 3, figsize=(5.4, 3.6), squeeze=False)
    panel_ids = list("abcdef")
    index = 0
    for family in ("S0", "S1"):
        for step in CHECKPOINT_STEPS:
            axis = axes.flat[index]
            axis.imshow(sampled_means[family][step][:, 0].reshape(CLASS_COUNT, -1), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
            axis.set_title(f"{family} step{step}", fontsize=7)
            axis.set_xticks([])
            axis.set_yticks(range(CLASS_COUNT) if index % 3 == 0 else [])
            if index % 3 == 0:
                axis.set_yticklabels([str(digit) for digit in range(CLASS_COUNT)], fontsize=5)
            _add_panel_label(axis, panel_ids[index])
            index += 1
    fig.text(0.5, 0.005, "Rows are class means; exact S0 replay and frozen S1 bank; fixed global [0,1] normalization.", ha="center", fontsize=6)
    figures["sampled_class_average_panel"] = _save_figure(fig, target, "sampled_class_average_panel", list(axes.flat), panel_ids, row_groups=(("a", "b", "c"), ("d", "e", "f")), column_groups=(("a", "d"), ("b", "e"), ("c", "f")))
    return figures


def _classify(records: Mapping[str, Mapping[int, Mapping[str, Any]]], consensus: Mapping[str, Mapping[int, Mapping[str, Any]]], diversity: Mapping[str, Mapping[int, Mapping[str, Any]]], transfer: Mapping[str, Any], trajectory: Mapping[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    s1_start = float(np.mean([records["s1"][3000]["metrics"][name]["macro_recall"] for name in EVALUATOR_ORDER]))
    s1_end = float(np.mean([records["s1"][6000]["metrics"][name]["macro_recall"] for name in EVALUATOR_ORDER]))
    s1_delta = s1_end - s1_start
    real_end = float(np.mean([records["real_dev"][6000]["metrics"][name]["macro_recall"] for name in EVALUATOR_ORDER]))
    transfer_ratios = [float(transfer["6000"][name]["R_S1_over_R_real"]) for name in EVALUATOR_ORDER]
    transfer_gaps = [float(transfer["6000"][name]["R_S1_minus_R_real"]) for name in EVALUATOR_ORDER]
    s1_consensus = consensus["s1"][6000]
    requested_ge3 = float(s1_consensus["at_least_3_of_4_true_or_requested_fraction"])
    class_ge3 = [float(value["at_least_3_of_4_true_or_requested"]) for value in s1_consensus["class_conditioned"].values()]
    pairwise = s1_consensus["pairwise_prediction_agreement_matrix"]
    off_diagonal = [pairwise[i][j] for i in range(4) for j in range(4) if i != j]
    severe_classes: list[str] = []
    for digit, value in diversity["s1"][6000]["class_conditioned"].items():
        e_c = value.get("E_C_feature_variance_mean", 0.0)
        pixel = value.get("pixel_variance_mean_28x28_fixed_subset", 0.0)
        if pixel <= 1.0e-8 or e_c <= 1.0e-8:
            severe_classes.append(str(digit))
    s1_near = float(diversity["s1"][6000]["aggregate"]["E_C_feature"]["near_duplicate_pair_fraction"])
    s1_dominant = max(float(records["s1"][6000]["metrics"][name]["dominant_fraction"]) for name in EVALUATOR_ORDER)
    mean_s0_var = float(np.mean([diversity["s0"][6000]["class_conditioned"][str(digit)]["E_C_feature_variance_mean"] for digit in range(CLASS_COUNT)]))
    mean_s1_var = float(np.mean([diversity["s1"][6000]["class_conditioned"][str(digit)]["E_C_feature_variance_mean"] for digit in range(CLASS_COUNT)]))
    variance_ratio = mean_s1_var / max(mean_s0_var, 1.0e-12)
    late_trend = str(trajectory["trend"])
    broad_s1_gain = s1_delta >= SUBSTANTIAL_DELTA
    strong_consensus = requested_ge3 >= STRONG_CONSENSUS and min(class_ge3) >= STRONG_CONSENSUS_MIN_CLASS
    approaches_real = min(transfer_ratios) >= REAL_TRANSFER_RATIO and max(abs(value) for value in transfer_gaps) <= REAL_TRANSFER_GAP
    divergent = float(np.mean(off_diagonal)) < LOW_PAIRWISE_AGREEMENT
    severe_collapse = (s1_near >= SEVERE_NEAR_DUPLICATE_FRACTION or s1_dominant >= SEVERE_DOMINANT_FRACTION or variance_ratio < SEVERE_VARIANCE_RATIO or len(severe_classes) >= 3)
    real_improves = float(np.mean([records["real_dev"][6000]["metrics"][name]["macro_recall"] - records["real_dev"][3000]["metrics"][name]["macro_recall"] for name in EVALUATOR_ORDER])) >= SUBSTANTIAL_DELTA
    s1_below_real = sum(value < -REAL_TRANSFER_GAP for value in transfer_gaps) >= 2
    if broad_s1_gain and approaches_real and strong_consensus and not severe_collapse:
        classification = "STEP6000_CONDITIONAL_GENERATION_READINESS_SUPPORTED"
        mechanism = "The frozen S1 requested-class bank gains conditional fidelity through step6000 while reaching the preregistered real-path transfer and cross-evaluator consensus criteria without severe collapse."
        successor = "DEFINE_FINAL_CONDITIONAL_GENERATION_CONFIRMATION_V1"
    elif broad_s1_gain and divergent:
        classification = "STEP6000_SEMANTIC_GAIN_WITH_CROSS_EVALUATOR_DECISION_DIVERGENCE"
        mechanism = "Average conditional semantic scores improve, but sample-level evaluator agreement remains below the preregistered consensus floor."
        successor = "LOCALIZE_STEP6000_IMAGE_MANIFOLD_SEMANTIC_DIVERGENCE_V1"
    elif real_improves and s1_below_real:
        classification = "STEP6000_REAL_PATH_GAIN_WITH_LIMITED_S1_TRANSFER"
        mechanism = "The real Encoder-produced OLS path improves more than the frozen S1 conditional bank, leaving a material real-to-S1 transfer gap at step6000."
        successor = "ATTRIBUTE_STEP6000_REAL_VS_S1_GENERATOR_LOCAL_TRANSFER_GAP_V1"
    elif broad_s1_gain and severe_collapse:
        classification = "STEP6000_CONDITIONAL_FIDELITY_WITH_DIVERSITY_LIMITATION"
        mechanism = "Conditional fidelity improves, but the fixed-bank output diversity diagnostics meet the preregistered severe-collapse condition."
        successor = "DEFINE_SEMANTIC_MANIFOLD_DIVERSITY_EXPANSION_V1"
    elif late_trend == "flattening" and not broad_s1_gain and not severe_collapse:
        classification = "STEP6000_SEMANTIC_REALIZATION_PLATEAU"
        mechanism = "Late trajectory losses flatten under the preregistered rule while real and S1 semantic realization remain below coherent conditional-generation criteria."
        successor = "ATTRIBUTE_STEP6000_RESIDUAL_SEMANTIC_DIRECTION_V1"
    else:
        classification = "STEP6000_CONDITIONAL_GENERATION_READINESS_MIXED"
        mechanism = "The frozen step6000 audit contains separated gains and limitations across real-path fidelity, S1 transfer, evaluator consensus, diversity, novelty and visual realization; no single bottleneck is supported by the preregistered rules."
        successor = "NONE"
    details = {
        "s1_macro_recall_step3000": s1_start,
        "s1_macro_recall_step6000": s1_end,
        "s1_macro_recall_delta": s1_delta,
        "real_macro_recall_step6000": real_end,
        "transfer_ratios_step6000": transfer_ratios,
        "transfer_gaps_step6000": transfer_gaps,
        "s1_consensus_ge3": requested_ge3,
        "s1_consensus_min_class_ge3": min(class_ge3),
        "s1_pairwise_mean_off_diagonal": float(np.mean(off_diagonal)),
        "s1_near_duplicate_pair_fraction": s1_near,
        "s1_max_dominant_fraction": s1_dominant,
        "s1_to_s0_E_C_variance_ratio": variance_ratio,
        "severe_collapse_classes": severe_classes,
        "real_path_improves": real_improves,
        "s1_below_real_on_at_least_two_evaluators": s1_below_real,
        "preregistered_rule": {"substantial_delta": SUBSTANTIAL_DELTA, "real_transfer_ratio_floor": REAL_TRANSFER_RATIO, "real_transfer_gap_ceiling": REAL_TRANSFER_GAP, "strong_consensus_floor": STRONG_CONSENSUS, "strong_consensus_min_class_floor": STRONG_CONSENSUS_MIN_CLASS, "low_pairwise_agreement": LOW_PAIRWISE_AGREEMENT, "severe_near_duplicate_fraction": SEVERE_NEAR_DUPLICATE_FRACTION, "severe_dominant_fraction": SEVERE_DOMINANT_FRACTION, "severe_variance_ratio": SEVERE_VARIANCE_RATIO},
    }
    return classification, mechanism, successor, details


def _load_authorities(root: Path) -> dict[str, Any]:
    print(json.dumps({"phase": "AUTHORITY_PREFLIGHT", "status": "START"}, ensure_ascii=False), flush=True)
    continuation_authorities = continuation._load_authorities(root)
    replay_authorities = replay._load_authorities(root)
    _require(continuation_authorities["local_state"] == replay_authorities["sampler_inputs"]["git"], "local Git identity differs between authorities")
    _require(file_sha256(root / CONTINUATION_REPORT) == CONTINUATION_REPORT_SHA256, "accepted SIC continuation report identity changed")
    continuation_report = json.loads((root / CONTINUATION_REPORT).read_text(encoding="utf-8"))
    _require(continuation_report.get("result") == "VALID_COMPLETED", "accepted SIC continuation result changed")
    _require(continuation_report.get("outer_gate") == continuation.OUTER_GATE, "accepted SIC continuation gate changed")
    _require(continuation_report.get("classification") == continuation.CLASSIFICATION_SUPPORTED, "accepted SIC continuation classification changed")
    _require(continuation_report.get("stage2", {}).get("executed") is True and continuation_report.get("training_trajectory", {}).get("updates") == 3000, "accepted step6000 continuation is incomplete")
    checkpoints: dict[int, Mapping[str, Any]] = {3000: continuation_authorities["step3000"]}
    for step, path, expected in ((3300, STEP3300, STEP3300_SHA256), (6000, STEP6000, STEP6000_SHA256)):
        artifact = root / path
        _require(file_sha256(artifact) == expected, f"step{step} checkpoint identity changed")
        checkpoints[step] = _mapping(safe_torch_load(artifact, expected_sha256=expected, map_location="cpu"), f"step{step} checkpoint")
    base_encoder = _tensor(checkpoints[3000]["encoder_state"], "step3000 encoder state")
    base_generator = _tensor(checkpoints[3000]["generator_state"], "step3000 generator state")
    for step in CHECKPOINT_STEPS:
        checkpoint = checkpoints[step]
        _require(checkpoint.get("completed_updates") == step and checkpoint.get("scientific_artifact") is True and checkpoint.get("operational_artifact") is False, f"step{step} checkpoint role changed")
        _require(checkpoint.get("trainable_parameter_names") == ["generator.phase"], f"step{step} trainable boundary changed")
        _require(torch.equal(_tensor(checkpoint["encoder_state"], f"step{step} encoder"), base_encoder), f"step{step} encoder is not frozen-identical")
        _finite(_tensor(checkpoint["generator_state"], f"step{step} generator"), f"step{step} generator")
    _require(not torch.equal(_tensor(checkpoints[3300]["generator_state"], "step3300 generator"), base_generator), "step3300 generator did not differ from step3000")
    _require(not torch.equal(_tensor(checkpoints[6000]["generator_state"], "step6000 generator"), _tensor(checkpoints[3300]["generator_state"], "step3300 generator")), "step6000 generator did not differ from step3300")
    sampler_models_path = root / SAMPLER_MODELS
    _require(file_sha256(sampler_models_path) == SAMPLER_MODELS_SHA256, "S1 sampler-model authority identity changed")
    sampler_models = _mapping(safe_torch_load(sampler_models_path, expected_sha256=SAMPLER_MODELS_SHA256, map_location="cpu"), "sampler models")
    sampler_bank = safe_torch_load(root / SAMPLER_BANK, expected_sha256=SAMPLER_BANK_SHA256, map_location="cpu")
    _require(isinstance(sampler_bank, Mapping), "sampler bank payload is malformed")
    _require(torch.equal(replay_authorities["candidate_z"]["S1"], continuation_authorities["s1_winner"]["z"]), "exact frozen S1 bank differs between authorities")
    _require(torch.equal(replay_authorities["requested"], continuation_authorities["s1_winner"]["requested_labels"]), "requested S1 labels differ between authorities")
    _require(torch.equal(replay_authorities["canonical_images"], continuation_authorities["images"]) and torch.equal(replay_authorities["canonical_labels"], continuation_authorities["labels"]), "canonical image/label authority differs")
    _require(torch.equal(continuation_authorities["development_indices"], replay_authorities["sampler_inputs"]["robust"]["development_indices"]), "real DEV indices differ")
    print(json.dumps({"phase": "AUTHORITY_PREFLIGHT", "status": "PASS", "steps": list(CHECKPOINT_STEPS), "s1_rows": 2000, "s0_rows": 2000}, ensure_ascii=False), flush=True)
    return {"continuation": continuation_authorities, "replay": replay_authorities, "continuation_report": continuation_report, "checkpoints": checkpoints, "sampler_models": sampler_models, "sampler_bank": sampler_bank}


def _load_models(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[LightGen2DChenBOGTComplexOLSTrainingConsumer, dict[str, torch.nn.Module]]:
    consumer, smooth, independent, ec, ed = continuation._load_models(root, authorities["continuation"], device)
    consumer.runtime.eval()
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    evaluators = {"E_S": smooth, "E_I": independent, "E_C": ec, "E_D": ed}
    for model in evaluators.values():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    return consumer, evaluators


def _no_update_preflight(consumer: Any, evaluators: Mapping[str, torch.nn.Module], authorities: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    checkpoint = authorities["checkpoints"][3000]
    consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
    consumer.runtime.eval()
    before_runtime = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    before_evaluators = {name: _module_digest(model) for name, model in evaluators.items()}
    real_images = authorities["continuation"]["images"].index_select(0, authorities["continuation"]["development_indices"][:BATCH_SIZE]).to(device)
    sample_z = authorities["replay"]["candidate_z"]["S1"][:BATCH_SIZE].to(device)
    with torch.inference_mode():
        real_result = consumer.runtime.forward_from_image(real_images)
        generated_result = consumer.generated_forward(sample_z)
        _finite(real_result.raw_intensity, "preflight real raw")
        _finite(generated_result.raw_intensity, "preflight generated raw")
    after_runtime = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    _require(all(torch.equal(before_runtime[name], after_runtime[name]) for name in before_runtime), "read-only preflight changed runtime state")
    _require(before_evaluators == {name: _module_digest(model) for name, model in evaluators.items()}, "read-only preflight changed evaluator state")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "read-only preflight left runtime gradients")
    _require(all(parameter.grad is None for model in evaluators.values() for parameter in model.parameters()), "read-only preflight left evaluator gradients")
    return {"schema": PREFLIGHT_SCHEMA, "status": "READY", "no_update": True, "backward_path_used": False, "optimizer_created": False, "sampler_fit": False, "representative_real_forward": True, "representative_s1_forward": True, "runtime_state_unchanged": True, "evaluator_state_unchanged": True, "all_parameters_grad_none": True, "protected_test_read": False, "sealed_read": False}


def _run_experiment(root: Path, output_dir: Path) -> dict[str, Any]:
    target = (root / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "protected output namespace is forbidden")
    _require(not target.exists(), f"write-once output already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    authorities = _load_authorities(root)
    continuation_authorities = authorities["continuation"]
    replay_authorities = authorities["replay"]
    dev_indices = continuation_authorities["development_indices"].cpu().to(torch.long)
    dev_images = continuation_authorities["images"].index_select(0, dev_indices).cpu().contiguous()
    dev_labels = continuation_authorities["labels"].index_select(0, dev_indices).cpu().to(torch.long).contiguous()
    s1_z = replay_authorities["candidate_z"]["S1"].cpu().contiguous()
    s0_z = replay_authorities["candidate_z"]["S0"].cpu().contiguous()
    requested = replay_authorities["requested"].cpu().to(torch.long).contiguous()
    panel_indices = continuation_authorities["s1_panels"]["panel_indices"].cpu().to(torch.long).contiguous()
    selection_payload, selection = _selection_manifest(root, continuation_authorities["images"], continuation_authorities["labels"], dev_indices, requested, panel_indices)
    _write_json(target / "fixed_selection_manifest.json", selection_payload, kind="fixed readiness selection manifest")
    source_manifest = {
        "schema": "step6000_conditional_generation_readiness_execution_manifest.v1",
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN, "scope": "forward-only assessment at step3000/3300/6000"},
        "local_state": git_identity(root),
        "no_update": True,
        "comparison_steps": list(CHECKPOINT_STEPS),
        "input_authorities": {
            "continuation_report": _artifact_ref(root, root / CONTINUATION_REPORT, role="accepted SIC continuation report"),
            "step3000": _artifact_ref(root, root / continuation.STEP3000, role="accepted step3000 checkpoint"),
            "step3300": _artifact_ref(root, root / STEP3300, role="accepted SIC step3300 checkpoint"),
            "step6000": _artifact_ref(root, root / STEP6000, role="accepted SIC step6000 checkpoint"),
            "s1_winner": _artifact_ref(root, root / continuation.S1_WINNER, role="accepted frozen S1 bank"),
            "s0_s1_bank": _artifact_ref(root, root / SAMPLER_BANK, role="accepted exact S0/S1 bank"),
            "sampler_models": _artifact_ref(root, root / SAMPLER_MODELS, role="accepted frozen S1 model metadata"),
            "s1_panels": _artifact_ref(root, root / continuation.S1_PANELS, role="accepted R diagnostic panel authority"),
            "training_source": _artifact_ref(root, root / continuation_authorities["training"]["references"].data_identity["source_path"], role="canonical image/label source"),
        },
        "evaluation_contract": {"sources": list(SOURCE_ORDER), "evaluators": list(EVALUATOR_ORDER), "batch_size": BATCH_SIZE, "protected_test_read": False, "sealed_read": False, "checkpoint_selection": False, "new_evaluator": False, "sampler_refit": False, "generator_update": False},
        "figure_contract": {"core_conclusion": "Whether the frozen step6000 Generator plus frozen S1 bank is a coherent conditional image-generation path at development-evidence level.", "results_question": "Are requested-class semantic gains shared, transferable, visually canonical, diverse and non-replay-like at step6000?", "archetype": "image plate + quant", "backend": "Python", "panel_map": {"real_dev_fixed_visual_panel": "fixed real-path visual realization", "s1_fixed_visual_panel": "fixed conditional outputs with preregistered nearest canonical references", "s0_s1_fixed_visual_panel": "exact replay versus S1 control", "real_class_average_panel": "canonical real-path class means", "sampled_class_average_panel": "S0/S1 class-average boundary"}, "display_integrity": "global [0,1] normalization, no per-image contrast stretch, fixed indices"},
    }
    _write_json(target / "execution_manifest.json", source_manifest, kind="readiness execution manifest")
    cuda_runtime = continuation.robust._prepare_cuda_runtime()
    if not torch.cuda.is_available():
        raise ReadinessAuditError("frozen readiness audit requires CUDA")
    device = torch.device("cuda:0")
    consumer, evaluators = _load_models(root, authorities, device)
    preflight = _no_update_preflight(consumer, evaluators, authorities, device)
    preflight["cuda_runtime"] = cuda_runtime
    _write_json(target / "preflight.json", preflight, kind="readiness no-update preflight")
    control_raw = continuation._tensor(continuation_authorities["s1_panels"]["real_validation_control_raw_panel"], "real validation control raw panel").to(device=device, dtype=torch.float32).unsqueeze(1)
    with torch.inference_mode():
        control_features, _ = evaluators["E_C"](continuation._cmos_view(control_raw))
    control_features = control_features.detach().cpu().contiguous()
    records: dict[str, dict[int, dict[str, Any]]] = {source: {} for source in SOURCE_ORDER}
    predictions: dict[str, dict[int, dict[str, torch.Tensor]]] = {source: {} for source in SOURCE_ORDER}
    fixed_views: dict[str, dict[int, torch.Tensor]] = {"real_dev": {}, "s0": {}, "s1": {}}
    class_means: dict[str, dict[int, torch.Tensor]] = {"real_dev": {}, "s0": {}, "s1": {}}
    target_features: torch.Tensor | None = None
    target_feature_labels = dev_labels
    fixed_real = selection["real_positions"]
    fixed_sampled = selection["sampled_positions"]
    real_panel_positions = selection["real_panel_positions"]
    source_inputs = {"real_dev": dev_images, "s0": s0_z, "s1": s1_z}
    source_labels = {"real_dev": dev_labels, "s0": requested, "s1": requested}
    source_fixed = {"real_dev": fixed_real, "s0": fixed_sampled, "s1": fixed_sampled}
    source_panel = {"real_dev": real_panel_positions, "s0": panel_indices, "s1": panel_indices}
    for step in CHECKPOINT_STEPS:
        checkpoint = authorities["checkpoints"][step]
        consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
        consumer.runtime.eval()
        for source in SOURCE_ORDER:
            payload = _evaluate_source(consumer, evaluators, source, step, source_inputs[source], source_labels[source], source_fixed[source], source_panel[source], device, continuation_authorities["training"]["references"])
            if source == "real_dev" and target_features is None:
                with torch.inference_mode():
                    target_features_list: list[torch.Tensor] = []
                    for start in range(0, dev_images.shape[0], BATCH_SIZE):
                        target_view = continuation._cmos_view(dev_images[start : start + BATCH_SIZE].to(device))
                        target_feature, _ = evaluators["E_C"](target_view)
                        target_features_list.append(target_feature.detach().cpu())
                    target_features = torch.cat(target_features_list).contiguous()
                _require(target_features.shape == (5000, 128), "canonical target E_C feature geometry changed")
            payload["consensus"] = _consensus_report(payload["predictions"], payload["labels"])
            payload["transfer_metrics"] = _raw_panel_metrics(payload, continuation_authorities, control_features)
            payload["panel_view"] = payload["fixed_view"][:0]
            panel_raw = payload["panel_raw"].to(torch.float32).unsqueeze(1)
            with torch.inference_mode():
                payload["panel_view"] = continuation._cmos_view(panel_raw).cpu()
            payload["fixed_labels"] = payload["labels"].index_select(0, source_fixed[source])
            payload["diversity"] = _diversity(source, payload, latent=source_inputs[source] if source != "real_dev" else None)
            if source != "real_dev":
                _require(target_features is not None, "canonical target features are missing")
                payload["output_novelty"] = _output_novelty(payload["features"]["E_C"], payload["labels"], target_features, target_feature_labels)
                payload["output_novelty"]["nearest_reference_positions"] = payload["output_novelty"]["nearest_reference_positions"].cpu()
            records[source][step] = payload
            predictions[source][step] = payload["predictions"]
            fixed_views[source][step] = payload["fixed_view"].clone()
            class_means[source][step] = payload["class_means"].clone()
            print(json.dumps({"phase": "FORWARD", "status": "COMPLETE", "source": source, "step": step}, ensure_ascii=False), flush=True)
            del payload
        for parameter in consumer.runtime.parameters():
            _require(parameter.grad is None, "forward audit left a runtime gradient")
    labels_by_source = {source: records[source][3000]["labels"] for source in SOURCE_ORDER}
    consensus = {source: {step: records[source][step]["consensus"] for step in CHECKPOINT_STEPS} for source in SOURCE_ORDER}
    transitions = _transition_report(predictions, labels_by_source)
    transfer = _real_vs_s1(records)
    s0_s1 = _s0_vs_s1(records)
    latent_novelty = _latent_novelty(root, replay_authorities, authorities["sampler_models"])
    trajectory = _trajectory_report(authorities["continuation_report"])
    diversity = {source: {step: records[source][step]["diversity"] for step in CHECKPOINT_STEPS} for source in SOURCE_ORDER}
    optical_validity = {source: {str(step): {"raw_validity": records[source][step]["raw_validity"], "power_validity": records[source][step]["power_validity"], "transfer_metrics": records[source][step]["transfer_metrics"], "predicted_class_histograms": {name: records[source][step]["metrics"][name]["predicted_class_histogram"] for name in EVALUATOR_ORDER}, "normalized_entropy": {name: records[source][step]["metrics"][name]["normalized_prediction_entropy"] for name in EVALUATOR_ORDER}, "dominant_fraction": {name: records[source][step]["metrics"][name]["dominant_fraction"] for name in EVALUATOR_ORDER}, "represented_classes": {name: records[source][step]["metrics"][name]["represented_class_count"] for name in EVALUATOR_ORDER}, "losses": records[source][step]["losses"]} for step in CHECKPOINT_STEPS} for source in SOURCE_ORDER}
    classification, mechanism, successor, classification_details = _classify(records, consensus, diversity, transfer, trajectory)
    nearest_refs: dict[int, torch.Tensor] = {}
    if target_features is not None:
        for step in CHECKPOINT_STEPS:
            sampled_positions = selection["sampled_positions"]
            fixed_features = records["s1"][step]["features"]["E_C"].index_select(0, sampled_positions)
            fixed_requested = requested.index_select(0, sampled_positions)
            nearest_refs[step] = _nearest_indices_by_class(fixed_features, fixed_requested, target_features, target_feature_labels)
            records["s1"][step]["output_novelty"]["fixed_visual_nearest_reference_positions"] = nearest_refs[step]
    for source in ("s0", "s1"):
        for step in CHECKPOINT_STEPS:
            novelty = records[source][step].get("output_novelty")
            if novelty is not None:
                novelty["nearest_reference_positions"] = novelty["nearest_reference_positions"].tolist()
                if isinstance(novelty.get("fixed_visual_nearest_reference_positions"), torch.Tensor):
                    novelty["fixed_visual_nearest_reference_positions"] = novelty["fixed_visual_nearest_reference_positions"].tolist()
    visual_artifacts = {"real_dev_fixed_visual_panel": _render_real_visual(target, dev_images, fixed_views["real_dev"], selection), "s1_fixed_visual_panel": _render_s1_visual(target, {step: fixed_views["s1"][step] for step in CHECKPOINT_STEPS}, nearest_refs, dev_images, selection), "s0_s1_fixed_visual_panel": _render_s0_s1_visual(target, {"S0": fixed_views["s0"], "S1": fixed_views["s1"]}, selection), "class_average_panels": _render_class_average(target, {"target": _class_means(dev_images, dev_labels), **class_means["real_dev"]}, {"S0": class_means["s0"], "S1": class_means["s1"]})}
    predictions_payload = {source: {str(step): {name: value.cpu() for name, value in predictions[source][step].items()} for step in CHECKPOINT_STEPS} for source in SOURCE_ORDER}
    _write_torch(target / "predictions.pt", {"schema": "step6000_conditional_generation_readiness_predictions.v1", "predictions": predictions_payload, "labels": {source: labels_by_source[source] for source in SOURCE_ORDER}}, kind="readiness prediction artifact")
    output_metrics: dict[str, Any] = {}
    for source in SOURCE_ORDER:
        output_metrics[source] = {}
        for step in CHECKPOINT_STEPS:
            payload = records[source][step]
            output_metrics[source][str(step)] = {"metrics": payload["metrics"], "consensus": payload["consensus"], "raw_validity": payload["raw_validity"], "power_validity": payload["power_validity"], "transfer_metrics": payload["transfer_metrics"], "losses": payload["losses"], "output_novelty": payload.get("output_novelty"), "diversity": payload["diversity"]}
    _write_json(target / "forward_metrics.json", output_metrics, kind="readiness forward metrics")
    _write_json(target / "latent_novelty.json", latent_novelty, kind="readiness latent novelty")
    _write_json(target / "trajectory.json", trajectory, kind="readiness trajectory summary")
    status = {"schema": STATUS_SCHEMA, "task_id": TASK_ID, "result": "VALID_COMPLETED", "outer_gate": OUTER_GATE, "outer_gate_passed": True, "classification": classification, "recommended_single_successor": successor, "next_action": "NONE", "no_training": True, "protected_test_read": False, "sealed_read": False}
    _write_json(target / "status.json", status, kind="readiness status")
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "mode": "IMPLEMENT_PREPARED",
        "risk_level": "R2_ARTIFACT_PRODUCING_RESEARCH",
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "authorization_consumed": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN, "one_authorization": True},
        "local_state": git_identity(root),
        "input_authorities": source_manifest["input_authorities"],
        "checkpoints": {str(step): {"path": _relative(root, root / (continuation.STEP3000 if step == 3000 else STEP3300 if step == 3300 else STEP6000)), "sha256": file_sha256(root / (continuation.STEP3000 if step == 3000 else STEP3300 if step == 3300 else STEP6000)), "completed_updates": step, "encoder_identical_to_step3000": True, "generator_state_sha256": _state_digest({"generator.phase": _tensor(authorities["checkpoints"][step]["generator_state"], f"step{step} generator")})} for step in CHECKPOINT_STEPS},
        "real_path_semantic_trajectory": {str(step): {name: records["real_dev"][step]["metrics"][name] for name in EVALUATOR_ORDER} for step in CHECKPOINT_STEPS},
        "s1_conditional_fidelity": {str(step): {name: records["s1"][step]["metrics"][name] for name in EVALUATOR_ORDER} for step in CHECKPOINT_STEPS},
        "cross_evaluator_consensus": {source: {str(step): consensus[source][step] for step in CHECKPOINT_STEPS} for source in SOURCE_ORDER},
        "error_set_transitions": transitions,
        "real_vs_s1_transfer": transfer,
        "s0_vs_s1": s0_s1,
        "fixed_visual_panels": visual_artifacts,
        "class_average_panels": visual_artifacts["class_average_panels"],
        "diversity": diversity,
        "novelty": {"latent": latent_novelty, "output": {source: {str(step): records[source][step].get("output_novelty") for step in CHECKPOINT_STEPS} for source in ("s0", "s1")}},
        "optical_distributional_validity": optical_validity,
        "late_trajectory_scaling": trajectory,
        "classification": classification,
        "mechanism": mechanism,
        "recommended_single_successor": successor,
        "classification_details": classification_details,
        "generator_training_justified": False,
        "generator_training_decision": "No automatic continuation is justified by this read-only audit; the named successor, if any, must be defined separately.",
        "claim_boundary": {"development_evidence_only": True, "final_human_perceptual_quality": False, "hardware_readiness": False, "sealed_confirmation": False, "formal_promotion": False, "E_D_step6000_independent_confirmation": False, "new_evaluator_created": False},
        "files_changed": {"repository_source": [_relative(root, Path(__file__))], "new_run_artifacts": sorted(path.name for path in target.iterdir())},
        "validation": {"preflight": "preflight.json", "figure_backend": "Python", "figure_alignment": "strict render-time panel alignment JSON written for each multi-panel figure", "figure_pdf_text_and_collision_audit_required": True, "focused_tests": "tests/v2/test_lightgen_2d_assess_step6000_conditional_generation_readiness_v1.py"},
        "unresolved": ["E_S/E_I/E_C/E_D remain development diagnostics; no final confirmation endpoint was created.", "Nearest canonical image distances are descriptive and do not prove memorization."],
        "next_action": "NONE",
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report, kind="step6000 readiness report")
    print(json.dumps({"result": report["result"], "outer_gate": report["outer_gate"], "classification": report["classification"], "recommended_single_successor": report["recommended_single_successor"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess frozen step6000 conditional generation readiness")
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and execute the forward-only audit")
    parser.add_argument("--authorization-token", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "output_dir": args.output_dir, "no_update": True}, ensure_ascii=False, indent=2))
        return 0
    if args.authorization_token != AUTHORIZATION_TOKEN:
        raise ReadinessAuditError("explicit authorization token is required")
    return 0 if _run_experiment(_root(), Path(args.output_dir)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
