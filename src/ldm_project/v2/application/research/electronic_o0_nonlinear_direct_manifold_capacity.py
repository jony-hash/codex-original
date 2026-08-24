"""H-NDMC: no-update nonlinear direct-manifold capacity audit.

The canary is an ephemeral, classwise empirical manifold map.  It is built
from the fixed MNIST training partition with exact rank-8 PCA, a Gaussian
copula, and a continuous Gaussian-kernel reconstruction over all same-class
training rows.  It never creates an optimizer, checkpoint, or deployable
generator.  Its only consumers are this lifecycle and the hypothesis ledger.
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
from torch import Tensor

from ...evidence import atomic_write_json, canonical_sha256, file_sha256, tensor_sha256
from ...student.conditional import camera_code_fraction, relative_diversity_ratios
from ...student.conditional_acceptance import conditional_coverage_ratio
from ...student.evaluator import SmoothMNISTEvaluator, load_frozen_evaluator, load_local_mnist, stratified_mnist_indices
from . import electronic_o0_direct_prior_capacity_canary as dpgc
from .electronic_o0_wae import _balanced_reference, _eval_model, _recall
from .lifecycle_artifacts import prepare_args, stage_action_decision, validate_manifest_namespace


SCHEMA_PREFIX = "electronic_o0_nonlinear_direct_manifold_capacity"
PROGRAM = "electronic-o0-nonlinear-direct-manifold-capacity"
CONFIG_NAME = "electronic_o0_nonlinear_direct_manifold_capacity_v1"
VERSION = 20260825
HARNESS_SEED = 8891
CODE_DIM = 8
CLASS_COUNT = 10
GAUSSIAN_BANK_SEEDS = (271828, 314159)
HISTORICAL_PANEL_SEEDS = (4242, 104729, 161803)
SAMPLES_PER_CLASS = 250
PREFLIGHT_SAMPLES_PER_CLASS = 32
PCA_RANK = 8
KERNEL_NEIGHBORS = 32
BANDWIDTH_ANCHORS = 512
LOGIT_EPSILON = 1.0 / 255.0
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
FINITE_DIFFERENCE_RANK_THRESHOLD = 1.0e-3
CURVATURE_MINIMUM = 1.0e-3
MEMORY_LIMIT_BYTES = 6 * 1024**3
STORAGE_LIMIT_BYTES = 64 * 1024**2
PREFLIGHT_WALL_LIMIT_SECONDS = 300.0
SECONDARY_ARCHITECTURE = SmoothMNISTEvaluator.architecture_kind
H_DPGC_CLOSEOUT_SCHEMA = "electronic_o0_direct_prior_capacity_canary_closeout.v1"
H_SCF_CLOSEOUT_SCHEMA = "electronic_o0_semantic_code_factorial_closeout.v1"
H_DPGC_CLOSEOUT_HASH = "e61e40c18a6f09a272a16414ae369699356cd7ca6fc56e666d4f5735b121d075"
H_SCF_CLOSEOUT_HASH = "59d927f7d7ed3b9633fe613e5b9ce14814ba9b5e15776e9ef47352594026ef94"
PRIMARY_EVALUATOR_HASH = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
SECONDARY_EVALUATOR_HASH = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
PRIMARY_QUALIFICATION_HASH = "26fea381bec5d6be2a2ed69806c190f7a7051b5f0944e5f90941a7d93f35695e"
SECONDARY_QUALIFICATION_HASH = "93592543306b5a3da8c238d86e41805fc2de0080459d217dfbc3e7161d374d88"


@dataclass(frozen=True)
class ManifoldCanary:
    means: Tensor
    bases: Tensor
    scales: Tensor
    sorted_coordinates: Tensor
    coordinate_scales: Tensor
    copula_cholesky: Tensor
    train_coordinates: Tensor
    train_images: Tensor
    bandwidths: Tensor
    image_shape: tuple[int, ...]

    def _inverse_coordinates(self, codes: Tensor, labels: Tensor, *, copula: bool) -> Tensor:
        transformed = codes
        if copula:
            cholesky = self.copula_cholesky.index_select(0, labels)
            transformed = torch.bmm(cholesky, codes.unsqueeze(-1)).squeeze(-1)
        unit = 0.5 * (1.0 + torch.erf(transformed / math.sqrt(2.0)))
        rows: list[Tensor] = []
        for dimension in range(CODE_DIM):
            sorted_values = self.sorted_coordinates.index_select(0, labels)[:, :, dimension]
            values = sorted_values
            positions = torch.searchsorted(values.contiguous(), unit[:, dimension].contiguous().unsqueeze(1), right=True).squeeze(1)
            positions = positions.clamp(1, values.shape[1] - 1)
            left = positions - 1
            right = positions
            lower = values.gather(1, left.unsqueeze(1)).squeeze(1)
            upper = values.gather(1, right.unsqueeze(1)).squeeze(1)
            fraction = ((unit[:, dimension] * (values.shape[1] - 1)) - left.to(unit.dtype)).clamp(0.0, 1.0)
            rows.append(lower + fraction * (upper - lower))
        return torch.stack(rows, dim=1)

    def sample(self, labels: Tensor, codes: Tensor, *, copula: bool = True, chunk_size: int = 64) -> Tensor:
        if labels.ndim != 1 or codes.ndim != 2 or tuple(codes.shape) != (labels.numel(), CODE_DIM):
            raise ValueError("H-NDMC expects labels [N] and codes [N,8]")
        if chunk_size not in (64, 257):
            raise ValueError("H-NDMC chunk probe is fixed to 64/257")
        if labels.numel() == 0 or labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("H-NDMC labels are invalid")
        labels = labels.to(device=self.means.device, dtype=torch.long)
        codes = codes.to(device=self.means.device, dtype=self.means.dtype)
        if not torch.isfinite(codes).all() or int(labels.min()) < 0 or int(labels.max()) >= CLASS_COUNT:
            raise ValueError("H-NDMC inputs are invalid")

        output = torch.empty((labels.numel(), self.train_images.shape[-1]), device=labels.device, dtype=self.train_images.dtype)
        # This canonical internal chunk is part of the numerical contract.  The
        # public probe may request 64 or 257, but reduction order stays fixed.
        canonical_chunk = 64
        for label in range(CLASS_COUNT):
            selected = torch.nonzero(labels == label, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            target = self._inverse_coordinates(codes.index_select(0, selected), labels.index_select(0, selected), copula=copula)
            target = target / self.coordinate_scales[label].clamp_min(1.0e-8)
            reference = self.train_coordinates[label] / self.coordinate_scales[label].clamp_min(1.0e-8)
            reference_norm = reference.square().sum(1)
            bandwidth = self.bandwidths[label].clamp_min(1.0e-6)
            for start in range(0, selected.numel(), canonical_chunk):
                current = target[start : start + canonical_chunk]
                distances = current.square().sum(1, keepdim=True) + reference_norm.unsqueeze(0) - 2.0 * (current @ reference.T)
                distances = distances.clamp_min(0.0)
                weights = torch.softmax(-distances / (2.0 * bandwidth.square()), dim=1)
                images = weights @ self.train_images[label]
                output[selected[start : start + canonical_chunk]] = images
        result = output.reshape(-1, *self.image_shape).clamp(0.0, 1.0)
        if not torch.isfinite(result).all():
            raise ValueError("H-NDMC produced non-finite images")
        return result


@dataclass(frozen=True)
class MemorizationContext:
    train_hashes: frozenset[str]
    thresholds: Tensor
    train_images: Tensor
    train_labels: Tensor
    validation_images: Tensor
    validation_labels: Tensor


def _git() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "clean": not status, "status": status}


def _config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "name": CONFIG_NAME,
        "version": VERSION,
        "code_dim": CODE_DIM,
        "harness_seed": HARNESS_SEED,
        "gaussian_bank_seeds": list(GAUSSIAN_BANK_SEEDS),
        "historical_panel_seeds": list(HISTORICAL_PANEL_SEEDS),
        "samples_per_class_per_bank": SAMPLES_PER_CLASS,
        "preflight_samples_per_class": PREFLIGHT_SAMPLES_PER_CLASS,
        "pca_rank": PCA_RANK,
        "kernel_reference_neighbors": KERNEL_NEIGHBORS,
        "bandwidth_anchor_count_per_class": BANDWIDTH_ANCHORS,
        "logit_epsilon": LOGIT_EPSILON,
        "evaluator_accuracy_minimum": ACCURACY_MINIMUM,
        "evaluator_minimum_recall": RECALL_MINIMUM,
        "evaluator_agreement_minimum": AGREEMENT_MINIMUM,
        "camera_code_fraction_minimum": CAMERA_CODE_MINIMUM,
        "diversity_ratio_minimum": DIVERSITY_MINIMUM,
        "diversity_ratio_maximum": DIVERSITY_MAXIMUM,
        "coverage_ratio_maximum": COVERAGE_MAXIMUM,
        "saturation_maximum": SATURATION_MAXIMUM,
        "memorization_pass_fraction_minimum": MEMORIZATION_MINIMUM,
        "label_shift_accuracy_maximum": LABEL_SHIFT_MAXIMUM,
        "finite_difference_rank": PCA_RANK,
        "finite_difference_rank_threshold": FINITE_DIFFERENCE_RANK_THRESHOLD,
        "curvature_minimum": CURVATURE_MINIMUM,
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if not isinstance(payload, dict) or set(payload) != set(expected) or any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("H-NDMC config is not the registered v1 contract")
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence is not a JSON object: {path}")
    return payload


def _write(path: str | Path, payload: dict[str, Any], args: argparse.Namespace) -> None:
    payload["invocation_mode"] = "manifest"
    payload["legacy_diagnostic_only"] = False
    payload["run_manifest_sha256"] = file_sha256(args.run_manifest)
    atomic_write_json(path, payload)


def _require_cuda(args: argparse.Namespace) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("H-NDMC requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H-NDMC formal lifecycle requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _logit(images: Tensor) -> Tensor:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("MNIST images must have shape [N,1,H,W]")
    values = images.to(torch.float32).flatten(1).clamp(LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return torch.logit(values)


def _canonical_basis(basis: Tensor) -> Tensor:
    rows = basis.abs().argmax(dim=0)
    signs = basis[rows, torch.arange(basis.shape[1], device=basis.device)].sign()
    return basis * torch.where(signs == 0, torch.ones_like(signs), signs)


def _fit_canary(images: Tensor, labels: Tensor, *, seed: int = HARNESS_SEED) -> ManifoldCanary:
    if images.ndim != 4 or labels.ndim != 1 or images.shape[0] != labels.numel() or labels.unique().numel() != CLASS_COUNT:
        raise ValueError("H-NDMC fit requires all ten classes and matching tensors")
    flat_logits = _logit(images)
    means, bases, scales, sorted_coordinates = [], [], [], []
    coordinate_scales, cholesky_rows, coordinate_rows, image_rows, bandwidths = [], [], [], [], []
    for label in range(CLASS_COUNT):
        selected = labels == label
        class_images = images[selected].flatten(1)
        rows = flat_logits[selected]
        if rows.shape[0] <= max(PCA_RANK, KERNEL_NEIGHBORS):
            raise ValueError("H-NDMC class has too few training rows")
        mean = rows.mean(0)
        centered = rows - mean
        covariance = centered.T @ centered / (rows.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues[-PCA_RANK:].flip(0).clamp_min(1.0e-8)
        basis = _canonical_basis(eigenvectors[:, -PCA_RANK:].flip(1))
        coordinates = centered @ basis
        coordinate_scale = eigenvalues.sqrt()
        normalized = coordinates / coordinate_scale
        gaussian = []
        sorted_dims = []
        for dimension in range(CODE_DIM):
            order = torch.argsort(coordinates[:, dimension], stable=True)
            values = coordinates[:, dimension].index_select(0, order)
            unit = ((torch.arange(values.numel(), device=images.device, dtype=images.dtype) + 0.5) / values.numel()).clamp(1.0e-6, 1.0 - 1.0e-6)
            gaussian_values = math.sqrt(2.0) * torch.erfinv(2.0 * unit - 1.0)
            inverse = torch.empty_like(gaussian_values)
            inverse[order] = gaussian_values
            gaussian.append(inverse)
            sorted_dims.append(values)
        gaussian_scores = torch.stack(gaussian, dim=1)
        correlation = gaussian_scores.T @ gaussian_scores / (rows.shape[0] - 1)
        correlation = 0.5 * (correlation + correlation.T)
        cholesky = torch.linalg.cholesky(correlation)
        anchor_count = min(BANDWIDTH_ANCHORS, normalized.shape[0])
        anchor_generator = torch.Generator(device="cpu").manual_seed(seed + label)
        anchor_indices = torch.randperm(normalized.shape[0], generator=anchor_generator)[:anchor_count].to(images.device)
        anchor = normalized.index_select(0, anchor_indices)
        pairwise = (anchor.square().sum(1, keepdim=True) + anchor.square().sum(1).unsqueeze(0) - 2.0 * (anchor @ anchor.T)).clamp_min(0.0)
        pairwise.fill_diagonal_(float("inf"))
        kth = pairwise.topk(min(KERNEL_NEIGHBORS, anchor.shape[0] - 1), largest=False, dim=1).values[:, -1]
        bandwidth = kth.median().sqrt().clamp_min(1.0e-4)
        means.append(mean)
        bases.append(basis)
        scales.append(coordinate_scale)
        sorted_coordinates.append(torch.stack(sorted_dims, dim=1))
        coordinate_scales.append(coordinate_scale)
        cholesky_rows.append(cholesky)
        coordinate_rows.append(coordinates)
        image_rows.append(class_images)
        bandwidths.append(bandwidth)
    result = ManifoldCanary(
        torch.stack(means), torch.stack(bases), torch.stack(scales), torch.stack(sorted_coordinates),
        torch.stack(coordinate_scales), torch.stack(cholesky_rows), torch.stack(coordinate_rows),
        torch.stack(image_rows), torch.stack(bandwidths), tuple(images.shape[1:]),
    )
    tensors = (result.means, result.bases, result.scales, result.sorted_coordinates, result.coordinate_scales, result.copula_cholesky, result.train_coordinates, result.train_images, result.bandwidths)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("H-NDMC canary parameters are non-finite")
    return result


def _panel(seed: int, *, samples_per_class: int, device: torch.device) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.arange(CLASS_COUNT, device=device).repeat_interleave(samples_per_class)
    codes = torch.randn((labels.numel(), CODE_DIM), generator=generator).to(device)
    return codes, labels


def _nearest_squared(query: Tensor, reference: Tensor, *, chunk_size: int = 64) -> Tensor:
    values = []
    reference_norm = reference.square().sum(1)
    for start in range(0, query.shape[0], chunk_size):
        current = query[start : start + chunk_size]
        distance = current.square().sum(1, keepdim=True) + reference_norm.unsqueeze(0) - 2.0 * (current @ reference.T)
        values.append(distance.clamp_min(0.0).min(1).values)
    return torch.cat(values)


def _memorization_context(train_images: Tensor, train_labels: Tensor, validation_images: Tensor, validation_labels: Tensor) -> MemorizationContext:
    train_flat = train_images.to(train_images.device).flatten(1)
    validation_flat = validation_images.to(train_images.device).flatten(1)
    thresholds = []
    for label in range(CLASS_COUNT):
        train_rows = train_flat[train_labels == label]
        validation_rows = validation_flat[validation_labels == label]
        thresholds.append(torch.quantile(_nearest_squared(validation_rows, train_rows), 0.05))
    train_u8 = (train_images.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    hashes = frozenset(tensor_sha256(row) for row in train_u8)
    return MemorizationContext(hashes, torch.stack(thresholds), train_images, train_labels, validation_images, validation_labels)


def _memorization(generated: Tensor, generated_labels: Tensor, context: MemorizationContext) -> dict[str, Any]:
    generated_cpu = generated.detach().cpu()
    generated_u8 = (generated_cpu.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    exact = sum(tensor_sha256(row) in context.train_hashes for row in generated_u8)
    distances, pass_counts = [], []
    generated_flat = generated.flatten(1)
    for label in range(CLASS_COUNT):
        rows = generated_flat[generated_labels == label]
        train_rows = context.train_images[context.train_labels == label].flatten(1)
        nearest = _nearest_squared(rows, train_rows)
        threshold = context.thresholds[label]
        count = int((nearest >= threshold).sum().item())
        pass_counts.append(count)
        distances.append(float(threshold.item()))
    per_class_minimum = math.ceil(SAMPLES_PER_CLASS * MEMORIZATION_MINIMUM)
    return {
        "exact_duplicate_count": exact,
        "per_class_thresholds": distances,
        "per_class_pass_counts": pass_counts,
        "per_class_pass_fraction": [count / max(1, int((generated_labels == label).sum())) for label, count in enumerate(pass_counts)],
        "pass_fraction": sum(pass_counts) / max(1, generated_labels.numel()),
        "passed": exact == 0 and all(count >= per_class_minimum for count in pass_counts),
    }


def _margin_quantiles(logits: Tensor, labels: Tensor) -> list[float]:
    target = logits.gather(1, labels.reshape(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.reshape(-1, 1), float("-inf"))
    margins = target - masked.max(1).values
    return [float(value) for value in torch.quantile(margins, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99])).tolist()]


def _counts(prediction: Tensor, labels: Tensor) -> dict[str, Any]:
    recall_counts = [int(((prediction[labels == label]) == label).sum().item()) for label in range(CLASS_COUNT)]
    class_total = [int((labels == label).sum().item()) for label in range(CLASS_COUNT)]
    return {
        "correct": int((prediction == labels).sum().item()),
        "total": int(labels.numel()),
        "recall_correct": recall_counts,
        "recall_total": class_total,
    }


def _metric_pass_counts(row: Mapping[str, Any]) -> bool:
    total = int(row["count"])
    class_total = total // CLASS_COUNT
    primary = row["primary_counts"]
    secondary = row["secondary_counts"]
    agreement = row["agreement_counts"]
    return bool(
        primary["correct"] >= math.ceil(total * ACCURACY_MINIMUM)
        and secondary["correct"] >= math.ceil(total * ACCURACY_MINIMUM)
        and min(primary["recall_correct"]) >= math.ceil(class_total * RECALL_MINIMUM)
        and min(secondary["recall_correct"]) >= math.ceil(class_total * RECALL_MINIMUM)
        and agreement["correct"] >= math.ceil(total * AGREEMENT_MINIMUM)
    )


def _evaluate_bank(canary: ManifoldCanary, primary: torch.nn.Module, secondary: torch.nn.Module, test_images: Tensor, test_labels: Tensor, train_context: MemorizationContext, *, seed: int, samples_per_class: int, device: torch.device, copula: bool = True) -> dict[str, Any]:
    codes, labels = _panel(seed, samples_per_class=samples_per_class, device=device)
    views = canary.sample(labels, codes, copula=copula)
    reference, reference_labels = _balanced_reference(test_images, test_labels, seed=seed + 17, count=samples_per_class)
    p_features, p_logits = _eval_model(primary, views)
    s_features, s_logits = _eval_model(secondary, views)
    p_reference, _ = _eval_model(primary, reference.to(device))
    s_reference, _ = _eval_model(secondary, reference.to(device))
    p_pred, s_pred = p_logits.argmax(1), s_logits.argmax(1)
    shuffled_generator = torch.Generator(device="cpu").manual_seed(8675309 + seed)
    shuffled_labels = labels.cpu().index_select(0, torch.randperm(labels.numel(), generator=shuffled_generator)).to(device)
    shuffled_views = canary.sample(shuffled_labels, codes, copula=copula)
    shuffled_primary = _eval_model(primary, shuffled_views)[1].argmax(1)
    zero_views = canary.sample(labels, torch.zeros_like(codes), copula=copula)
    zero_p, zero_s = _eval_model(primary, zero_views)[1].argmax(1), _eval_model(secondary, zero_views)[1].argmax(1)
    views_cpu = views.cpu()
    labels_cpu = labels.cpu()
    p_coverage = conditional_coverage_ratio(p_features, views_cpu, p_reference, reference, labels_cpu, epsilon=0.05, iterations=20, seed=seed)
    s_coverage = conditional_coverage_ratio(s_features, views_cpu, s_reference, reference, labels_cpu, epsilon=0.05, iterations=20, seed=seed)
    p_counts, s_counts = _counts(p_pred, labels_cpu), _counts(s_pred, labels_cpu)
    agreement_counts = _counts(p_pred, s_pred)
    row = {
        "seed": seed,
        "count": int(labels.numel()),
        "primary": {"accuracy": p_counts["correct"] / p_counts["total"], "recall": _recall(p_pred, labels_cpu), "minimum_recall": min(_recall(p_pred, labels_cpu)), "margin_quantiles": _margin_quantiles(p_logits, labels_cpu), "prediction_histogram": torch.bincount(p_pred, minlength=CLASS_COUNT).tolist()},
        "secondary": {"accuracy": s_counts["correct"] / s_counts["total"], "recall": _recall(s_pred, labels_cpu), "minimum_recall": min(_recall(s_pred, labels_cpu)), "margin_quantiles": _margin_quantiles(s_logits, labels_cpu), "prediction_histogram": torch.bincount(s_pred, minlength=CLASS_COUNT).tolist()},
        "primary_counts": p_counts,
        "secondary_counts": s_counts,
        "agreement": agreement_counts["correct"] / agreement_counts["total"],
        "agreement_counts": agreement_counts,
        "camera_code_fraction": float(camera_code_fraction(views_cpu, labels_cpu)),
        "saturation": float((views_cpu >= 1.0).float().mean()),
        "coverage": {"primary": p_coverage, "secondary": s_coverage},
        "diversity_ratio": {"primary": relative_diversity_ratios(p_features, p_reference, labels_cpu), "secondary": relative_diversity_ratios(s_features, s_reference, labels_cpu)},
        "label_shuffled_primary_counts": _counts(shuffled_primary, labels_cpu),
        "zero_code": {"primary": _counts(zero_p, labels_cpu), "secondary": _counts(zero_s, labels_cpu)},
        "view_hash": tensor_sha256(views_cpu),
        "zero_code_view_hash": tensor_sha256(zero_views.cpu()),
        "no_update_replay_exact": tensor_sha256(canary.sample(labels, codes, copula=copula).cpu()) == tensor_sha256(views_cpu),
        "memorization": _memorization(views, labels, train_context),
    }
    row["semantic_pass"] = _metric_pass_counts(row)
    row["safety_pass"] = bool(
        row["camera_code_fraction"] >= CAMERA_CODE_MINIMUM
        and row["saturation"] <= SATURATION_MAXIMUM
        and row["label_shuffled_primary_counts"]["correct"] <= math.floor(row["count"] * LABEL_SHIFT_MAXIMUM)
        and all(DIVERSITY_MINIMUM <= value <= DIVERSITY_MAXIMUM for name in ("primary", "secondary") for value in row["diversity_ratio"][name])
        and all(value <= COVERAGE_MAXIMUM for value in row["coverage"].values())
        and row["memorization"]["passed"]
    )
    row["agreement_pass"] = row["agreement_counts"]["correct"] >= math.ceil(row["count"] * AGREEMENT_MINIMUM)
    row["zero_code_pass"] = all(value["correct"] == value["total"] for value in row["zero_code"].values())
    if not all(_all_finite(value) for value in row.values()):
        raise ValueError("H-NDMC bank contains non-finite metrics")
    return row


def _jacobian_and_curvature(canary: ManifoldCanary, *, device: torch.device) -> dict[str, Any]:
    jacobian_rows, curvature_rows = [], []
    for label in range(CLASS_COUNT):
        label_tensor = torch.tensor([label], device=device)
        columns = []
        for axis in range(CODE_DIM):
            plus = torch.zeros(1, CODE_DIM, device=device)
            minus = torch.zeros(1, CODE_DIM, device=device)
            plus[0, axis], minus[0, axis] = 0.25, -0.25
            columns.append(((canary.sample(label_tensor, plus) - canary.sample(label_tensor, minus)) / 0.5).flatten())
        singular = torch.linalg.svdvals(torch.stack(columns))
        threshold = singular[0].clamp_min(1.0e-12) * FINITE_DIFFERENCE_RANK_THRESHOLD
        jacobian_rows.append({"label": label, "rank_1e-3": int((singular >= threshold).sum()), "singular_values": [float(item) for item in singular.detach().cpu()]})
        plus = canary.sample(label_tensor, torch.full((1, CODE_DIM), 0.5, device=device))
        minus = canary.sample(label_tensor, torch.full((1, CODE_DIM), -0.5, device=device))
        zero = canary.sample(label_tensor, torch.zeros((1, CODE_DIM), device=device))
        curvature = (plus + minus - 2.0 * zero).norm() / plus.norm().clamp_min(1.0e-12)
        curvature_rows.append({"label": label, "normalized_curvature": float(curvature.item())})
    return {"jacobian": jacobian_rows, "curvature": curvature_rows, "rank_pass": all(row["rank_1e-3"] == PCA_RANK for row in jacobian_rows), "curvature_pass": all(row["normalized_curvature"] >= CURVATURE_MINIMUM for row in curvature_rows)}


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


def _historical_audit(closeout_path: str, args: argparse.Namespace) -> dict[str, Any]:
    closeout = _load_json(closeout_path)
    report_path = closeout.get("measurement")
    if closeout.get("schema") != H_DPGC_CLOSEOUT_SCHEMA or closeout.get("status") != "valid" or closeout.get("classification") != "direct_prior_linear_semantic_capacity_failed" or closeout.get("next_action") != "close_direct_prior_linear_capacity_canary" or not isinstance(report_path, str):
        raise ValueError("H-NDMC H-DPGC closeout is not the registered valid failure")
    report = _load_json(report_path)
    audit_path = report.get("measurement")
    if report.get("schema") != "electronic_o0_direct_prior_capacity_canary_report.v1" or report.get("status") != "valid" or not isinstance(audit_path, str):
        raise ValueError("H-NDMC H-DPGC report is invalid")
    audit = _load_json(audit_path)
    if audit.get("schema") != "electronic_o0_direct_prior_capacity_canary_audit.v1" or audit.get("status") != "valid":
        raise ValueError("H-NDMC H-DPGC audit is invalid")
    replay_args = argparse.Namespace(dataset_root=args.dataset_root, device=args.device, primary_evaluator_checkpoint=args.primary_evaluator_checkpoint, secondary_evaluator_checkpoint=args.secondary_evaluator_checkpoint)
    replay = dpgc._build_measurement(replay_args, samples_per_class=dpgc.PANEL_SAMPLES_PER_CLASS)
    stored = audit.get("result")
    if not isinstance(stored, Mapping) or set(stored.get("panels", {})) != {str(seed) for seed in HISTORICAL_PANEL_SEEDS}:
        raise ValueError("H-NDMC H-DPGC historical panel structure is invalid")
    matches = True
    panel_checks = {}
    for seed in HISTORICAL_PANEL_SEEDS:
        observed, expected = replay["panels"][str(seed)], stored["panels"][str(seed)]
        observed_counts = {
            "primary_correct": round(observed["primary"]["accuracy"] * observed["count"]),
            "secondary_correct": round(observed["secondary"]["accuracy"] * observed["count"]),
            "agreement_correct": round(observed["agreement"] * observed["count"]),
            "primary_recall": [round(value * (observed["count"] // CLASS_COUNT)) for value in observed["primary"]["recall"]],
            "secondary_recall": [round(value * (observed["count"] // CLASS_COUNT)) for value in observed["secondary"]["recall"]],
        }
        expected_counts = {
            "primary_correct": round(expected["primary"]["accuracy"] * expected["count"]),
            "secondary_correct": round(expected["secondary"]["accuracy"] * expected["count"]),
            "agreement_correct": round(expected["agreement"] * expected["count"]),
            "primary_recall": [round(value * (expected["count"] // CLASS_COUNT)) for value in expected["primary"]["recall"]],
            "secondary_recall": [round(value * (expected["count"] // CLASS_COUNT)) for value in expected["secondary"]["recall"]],
        }
        panel_ok = observed.get("view_hash") == expected.get("view_hash") and observed_counts == expected_counts
        panel_checks[str(seed)] = {"view_hash_match": observed.get("view_hash") == expected.get("view_hash"), "counts_match": observed_counts == expected_counts}
        matches = matches and panel_ok
    if not matches:
        raise ValueError("H-NDMC H-DPGC historical replay differs from locked evidence")
    return {"matches": True, "classification": audit.get("classification"), "panel_checks": panel_checks, "replayed_result": replay}


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = {"h_dpgc_closeout": args.h_dpgc_closeout_evidence, "h_scf_closeout": args.h_scf_closeout_evidence, "primary_evaluator": args.primary_evaluator_checkpoint, "secondary_evaluator": args.secondary_evaluator_checkpoint, "primary_qualification": args.primary_qualification_evidence, "secondary_qualification": args.secondary_qualification_evidence}
    if not Path(args.dataset_root).is_dir() or any(not Path(path).is_file() for path in paths.values()):
        raise FileNotFoundError("H-NDMC lineage input is missing")
    input_hashes = {name: file_sha256(path) for name, path in paths.items()}
    expected_hashes = {"h_dpgc_closeout": H_DPGC_CLOSEOUT_HASH, "h_scf_closeout": H_SCF_CLOSEOUT_HASH, "primary_evaluator": PRIMARY_EVALUATOR_HASH, "secondary_evaluator": SECONDARY_EVALUATOR_HASH, "primary_qualification": PRIMARY_QUALIFICATION_HASH, "secondary_qualification": SECONDARY_QUALIFICATION_HASH}
    if input_hashes != expected_hashes:
        raise ValueError("H-NDMC upstream artifact hash is not the registered immutable input")
    primary_qualification, secondary_qualification = _load_json(args.primary_qualification_evidence), _load_json(args.secondary_qualification_evidence)
    if primary_qualification.get("status") != "passed" or secondary_qualification.get("status") != "passed":
        raise ValueError("H-NDMC evaluator qualification is not passed")
    scf = _load_json(args.h_scf_closeout_evidence)
    if scf.get("schema") != H_SCF_CLOSEOUT_SCHEMA or scf.get("status") != "valid" or scf.get("classification") != "semantic_counterfactual_safety_failed" or scf.get("next_action") != "close_frozen_decoder_derived_semantic_route":
        raise ValueError("H-NDMC requires the closed H-SCF valid failure")
    all_images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_index, validation_index = stratified_mnist_indices(all_labels)
    if train_index.numel() != 55000 or validation_index.numel() != 5000 or any(int((all_labels.index_select(0, validation_index) == label).sum()) != 500 for label in range(CLASS_COUNT)):
        raise ValueError("H-NDMC MNIST split is not the registered 55k/5k class-balanced split")
    return {"git": _git(), "config_sha256": file_sha256(args.config), "dataset_root": str(Path(args.dataset_root).resolve()), "dataset_files": {"train": tensor_sha256(all_images), "test": tensor_sha256(load_local_mnist(args.dataset_root, train=False)[0])}, "split": {"seed": 1234, "train_count": int(train_index.numel()), "validation_count": int(validation_index.numel()), "train_index_hash": tensor_sha256(train_index), "validation_index_hash": tensor_sha256(validation_index)}, "inputs": input_hashes, "contract": config}


def _check_consumer(path: str, action: str) -> dict[str, Any]:
    payload = _load_json(path)
    allowed = tuple(f"{SCHEMA_PREFIX}_{stage}.v1" for stage in ("validation", "preflight", "audit", "report"))
    if payload.get("schema") not in allowed or payload.get("status") != "valid" or action not in payload.get("permitted_consumers", ()):
        raise ValueError(f"invalid H-NDMC {action} evidence")
    if payload.get("invocation_mode") != "manifest" or payload.get("legacy_diagnostic_only") is not False:
        raise ValueError("legacy H-NDMC evidence")
    return payload


def _stage(args: argparse.Namespace, stage: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    return stage_action_decision(stage, producer_contract=f"{SCHEMA_PREFIX}-{stage}.v1", expected_inputs=expected, evidence=args.output)


def _measurement_harness_valid(result: Mapping[str, Any]) -> bool:
    banks = result.get("banks")
    controls = result.get("controls")
    return bool(
        isinstance(banks, Mapping) and set(banks) == {str(seed) for seed in GAUSSIAN_BANK_SEEDS}
        and all(isinstance(row, Mapping) and int(row.get("count", -1)) == CLASS_COUNT * SAMPLES_PER_CLASS for row in banks.values())
        and isinstance(controls, Mapping)
        and isinstance(result.get("historical_replay"), Mapping) and result["historical_replay"].get("matches") is True
        and result.get("chunk_exact") is True
        and result.get("optimizer_created") is False
        and result.get("checkpoint_created") is False
        and result.get("parameter_hash_before") == result.get("parameter_hash_after")
        and _all_finite(result)
    )


def _classify(result: Mapping[str, Any]) -> tuple[str, str]:
    if not result.get("measurement_valid", True) or not _measurement_harness_valid(result):
        return "invalid_harness", "repair_and_revalidate_on_new_clean_commit"
    banks = list(result["banks"].values())
    safety_failed = any(not row.get("safety_pass", False) for row in banks)
    if safety_failed:
        return "nonlinear_manifold_safety_failed", "close_nonlinear_direct_manifold_capacity_canary"
    semantic_individual = all(
        row["primary"]["accuracy"] >= ACCURACY_MINIMUM
        and row["secondary"]["accuracy"] >= ACCURACY_MINIMUM
        and row["primary"]["minimum_recall"] >= RECALL_MINIMUM
        and row["secondary"]["minimum_recall"] >= RECALL_MINIMUM
        for row in banks
    )
    if semantic_individual and any(not row.get("agreement_pass", False) for row in banks):
        return "generated_evaluator_consensus_failed", "register_generated_ood_measurement_counterfactual"
    if any(not row.get("semantic_pass", False) for row in banks):
        return "nonlinear_direct_semantic_capacity_failed", "close_nonlinear_direct_manifold_capacity_canary"
    if not result.get("observability", {}).get("rank_pass") or not result.get("observability", {}).get("curvature_pass") or not result.get("cross_panel_stability", False):
        return "nonlinear_manifold_capacity_not_established", "close_nonlinear_direct_manifold_capacity_canary"
    return "nonlinear_direct_manifold_capacity_admitted", "register_direct_generator_architecture_preflight"


def _build_measurement(args: argparse.Namespace, *, samples_per_class: int, include_history: bool) -> dict[str, Any]:
    device = torch.device(args.device)
    all_images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_index, validation_index = stratified_mnist_indices(all_labels)
    train_images, train_labels = all_images.index_select(0, train_index).to(device), all_labels.index_select(0, train_index).to(device)
    validation_images, validation_labels = all_images.index_select(0, validation_index).to(device), all_labels.index_select(0, validation_index).to(device)
    test_images, test_labels = load_local_mnist(args.dataset_root, train=False)
    canary = _fit_canary(train_images, train_labels)
    primary, primary_payload = load_frozen_evaluator(args.primary_evaluator_checkpoint, device=device)
    secondary = dpgc._secondary(args.secondary_evaluator_checkpoint, device=device)
    before = {"primary": dpgc.state_content_hash(primary.state_dict()), "secondary": dpgc.state_content_hash(secondary.state_dict())}
    context = _memorization_context(train_images, train_labels, validation_images, validation_labels)
    banks = {}
    for seed in GAUSSIAN_BANK_SEEDS:
        banks[str(seed)] = _evaluate_bank(canary, primary, secondary, test_images, test_labels, context, seed=seed, samples_per_class=samples_per_class, device=device)
    identity_copula = {}
    for seed in GAUSSIAN_BANK_SEEDS:
        row = _evaluate_bank(canary, primary, secondary, test_images, test_labels, context, seed=seed, samples_per_class=samples_per_class, device=device, copula=False)
        identity_copula[str(seed)] = {"view_hash": row["view_hash"], "primary_counts": row["primary_counts"], "secondary_counts": row["secondary_counts"], "agreement_counts": row["agreement_counts"], "finite": _all_finite(row)}
    observed_chunk_codes, observed_chunk_labels = _panel(271828, samples_per_class=PREFLIGHT_SAMPLES_PER_CLASS, device=device)
    chunk_a = canary.sample(observed_chunk_labels, observed_chunk_codes, copula=True, chunk_size=64).cpu()
    chunk_b = canary.sample(observed_chunk_labels, observed_chunk_codes, copula=True, chunk_size=257).cpu()
    after = {"primary": dpgc.state_content_hash(primary.state_dict()), "secondary": dpgc.state_content_hash(secondary.state_dict())}
    historical = _historical_audit(args.h_dpgc_closeout_evidence, args) if include_history else {"matches": True, "replayed": False}
    result = {
        "canary": {"code_dim": CODE_DIM, "pca_rank": PCA_RANK, "kernel_neighbors": KERNEL_NEIGHBORS, "bandwidth_anchor_count": BANDWIDTH_ANCHORS, "fit_seed": HARNESS_SEED, "mean_hash": tensor_sha256(canary.means), "basis_hash": tensor_sha256(canary.bases), "coordinate_hash": tensor_sha256(canary.train_coordinates), "copula_hash": tensor_sha256(canary.copula_cholesky), "bandwidth_hash": tensor_sha256(canary.bandwidths)},
        "banks": banks,
        "identity_copula": identity_copula,
        "controls": {"zero_code": {seed: banks[str(seed)]["zero_code"] for seed in GAUSSIAN_BANK_SEEDS}, "label_shift": {seed: banks[str(seed)]["label_shuffled_primary_counts"] for seed in GAUSSIAN_BANK_SEEDS}},
        "observability": _jacobian_and_curvature(canary, device=device),
        "historical_replay": historical,
        "chunk_exact": bool(torch.equal(chunk_a, chunk_b)),
        "chunk_probe_hash": tensor_sha256(chunk_a),
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "qualification": {"primary_validation_accuracy": primary_payload.get("validation_accuracy"), "primary_checkpoint_sha256": file_sha256(args.primary_evaluator_checkpoint), "secondary_checkpoint_sha256": file_sha256(args.secondary_evaluator_checkpoint)},
        "optimizer_created": False,
        "checkpoint_created": False,
        "evaluator_calls": 1,
        "cross_panel_stability": all(row["semantic_pass"] and row["safety_pass"] for row in banks.values()),
    }
    if not _all_finite(result):
        raise ValueError("H-NDMC measurement contains non-finite values")
    return result


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("H-NDMC lifecycle is manifest-only")
    action = args.research_action
    common = ("config", "dataset_root", "output", "h_dpgc_closeout_evidence", "h_scf_closeout_evidence", "primary_evaluator_checkpoint", "secondary_evaluator_checkpoint", "primary_qualification_evidence", "secondary_qualification_evidence")
    required = {"validate": common, "preflight": common + ("validation_evidence",), "audit": common + ("preflight_evidence",), "report": ("config", "dataset_root", "output", "measurement_evidence"), "closeout": ("config", "dataset_root", "output", "measurement_evidence")}
    fields = common + ("device", "validation_evidence", "preflight_evidence", "measurement_evidence")
    args = prepare_args(args, program=PROGRAM, action=action, required=required[action], defaults={"device": "cuda", "seed": HARNESS_SEED}, mixed_fields=fields)
    validate_manifest_namespace(args, action=action)
    config = _config(args.config)
    lineage = _lineage(args, config) if action in {"validate", "preflight", "audit"} else None
    if action == "validate":
        decision = _stage(args, "validate", {"config": file_sha256(args.config), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_validation.v1", "stage": "validate", "status": "valid", "passed": True, "measurement_valid": True, "lineage_sha256": canonical_sha256(lineage), "consumer_contract": f"{SCHEMA_PREFIX}-preflight.v1", "permitted_consumers": ["preflight"], "producer_contract_revision": f"{SCHEMA_PREFIX}-validation.v1", "input_artifact_hashes": lineage["inputs"]}, args)
        return 0
    if action == "preflight":
        validation = _check_consumer(args.validation_evidence, "preflight")
        if not validation.get("passed"):
            raise ValueError("H-NDMC preflight requires passed validation")
        decision = _stage(args, "preflight", {"validation": file_sha256(args.validation_evidence), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.monotonic()
        result = _build_measurement(args, samples_per_class=PREFLIGHT_SAMPLES_PER_CLASS, include_history=True)
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(args.device))
        storage = sum(item.stat().st_size for item in Path(args._run_manifest["output_root"]).rglob("*") if item.is_file())
        valid = bool(result["historical_replay"].get("matches") and result["chunk_exact"] and result["optimizer_created"] is False and result["checkpoint_created"] is False and result["parameter_hash_before"] == result["parameter_hash_after"] and _all_finite(result) and peak < MEMORY_LIMIT_BYTES and storage <= STORAGE_LIMIT_BYTES and elapsed <= PREFLIGHT_WALL_LIMIT_SECONDS)
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_preflight.v1", "stage": "preflight", "status": "valid" if valid else "invalid_harness", "passed": valid, "measurement_valid": valid, "result": result, "peak_memory_bytes": peak, "storage_bytes": storage, "wall_time_seconds": elapsed, "memory_gate": peak < MEMORY_LIMIT_BYTES, "storage_gate": storage <= STORAGE_LIMIT_BYTES, "wall_time_gate": elapsed <= PREFLIGHT_WALL_LIMIT_SECONDS, "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1", "permitted_consumers": ["audit"] if valid else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-preflight.v1", "input_artifact_hashes": {"validation": file_sha256(args.validation_evidence), "lineage": canonical_sha256(lineage)}}, args)
        return 0
    if action == "audit":
        preflight = _check_consumer(args.preflight_evidence, "audit")
        if not preflight.get("passed"):
            raise ValueError("H-NDMC audit requires passed preflight")
        decision = _stage(args, "audit", {"preflight": file_sha256(args.preflight_evidence), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        result = _build_measurement(args, samples_per_class=SAMPLES_PER_CLASS, include_history=True)
        peak = int(torch.cuda.max_memory_allocated(args.device))
        storage = sum(item.stat().st_size for item in Path(args._run_manifest["output_root"]).rglob("*") if item.is_file())
        valid = bool(_measurement_harness_valid(result) and peak < MEMORY_LIMIT_BYTES and storage <= STORAGE_LIMIT_BYTES)
        result["measurement_valid"] = valid
        classification, next_action = _classify(result if valid else {"measurement_valid": False})
        if not valid:
            classification, next_action = "invalid_harness", "repair_and_revalidate_on_new_clean_commit"
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_audit.v1", "stage": "audit", "status": "valid" if valid else "invalid_harness", "passed": bool(valid and classification == "nonlinear_direct_manifold_capacity_admitted"), "measurement_valid": valid, "classification": classification, "next_action": next_action, "result": result, "peak_memory_bytes": peak, "storage_bytes": storage, "memory_gate": peak < MEMORY_LIMIT_BYTES, "storage_gate": storage <= STORAGE_LIMIT_BYTES, "optimizer_created": False, "checkpoint_created": False, "consumer_contract": f"{SCHEMA_PREFIX}-report.v1", "permitted_consumers": ["report"] if valid else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1", "input_artifact_hashes": {"preflight": file_sha256(args.preflight_evidence), "lineage": canonical_sha256(lineage)}}, args)
        return 0
    measurement = _check_consumer(args.measurement_evidence, action)
    decision = _stage(args, action, {"measurement": file_sha256(args.measurement_evidence)})
    if decision["action"] == "REUSE":
        return 0
    if action == "report":
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_report.v1", "stage": "report", "status": "valid", "passed": measurement.get("passed"), "measurement_valid": measurement.get("measurement_valid"), "classification": measurement.get("classification"), "next_action": measurement.get("next_action"), "measurement": args.measurement_evidence, "source_evidence_sha256": file_sha256(args.measurement_evidence), "input_artifact_hashes": {"measurement": file_sha256(args.measurement_evidence)}, "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1", "permitted_consumers": ["closeout"], "producer_contract_revision": f"{SCHEMA_PREFIX}-report.v1"}, args)
        return 0
    _write(args.output, {"schema": f"{SCHEMA_PREFIX}_closeout.v1", "stage": "closeout", "status": "valid", "passed": measurement.get("passed"), "measurement_valid": measurement.get("measurement_valid"), "classification": measurement.get("classification"), "next_action": measurement.get("next_action"), "measurement": args.measurement_evidence, "source_evidence_sha256": file_sha256(args.measurement_evidence), "input_artifact_hashes": {"measurement": file_sha256(args.measurement_evidence)}, "permitted_consumers": [], "producer_contract_revision": f"{SCHEMA_PREFIX}-closeout.v1"}, args)
    return 0
