"""Package A: fail-closed electronic distribution range recovery.

This route is a research consumer of the frozen v3 electronic generator.  It
does not construct the optical system, create a model checkpoint, or call an
evaluator while optimizing latent codes.  The evaluator calls in H-GFA and
post-inversion H-LIR are deliberately separated from the image-only objective.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from ...evidence import (
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...student.checkpoints import state_content_hash, state_signature
from ...student.conditional import relative_diversity_ratios
from ...student.conditional_acceptance import conditional_coverage_ratio, memorization_statistics
from ...student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
    load_local_mnist,
)
from ...student.spatial_quadrature_generator import HierarchicalSpatialAmplitudeGenerator
from .._runtime_shared import split_mnist


PROGRAM = "electronic-distribution-recovery"
SCHEMA_PREFIX = "electronic_distribution_recovery"
CONFIG_NAME = "electronic_distribution_recovery_v1"
CODE_DIM = 32
CLASS_COUNT = 10
PRIMARY_HASH = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
SECONDARY_HASH = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
GENERATOR_HASH = "5f1e36bfa99b9520c41dc1424abb4a629a695d449c52b783ef4cf9cffb40c75c"
PRIMARY_QUALIFICATION_HASH = "26fea381bec5d6be2a2ed69806c190f7a7051b5f0944e5f90941a7d93f35695e"
SECONDARY_QUALIFICATION_HASH = "93592543306b5a3da8c238d86e41805fc2de0080459d217dfbc3e7161d374d88"
FORMAL_V3_CONFIG = Path("configs/v2/current/formal_spatial_quadrature_v3.yaml")
FORMAL_V3_ARCHITECTURE = "conditional_spatial_quadrature_v3"
FORBIDDEN_ACTIONS = (
    "ddpm_training",
    "student_training",
    "optical_training",
    "phase_training",
    "quadrature_training",
    "ols_training",
    "fiber_training",
    "candidate_export",
    "promotion",
)
_POOL_WEIGHTS: dict[tuple[int, int, str, str], Tensor] = {}


def _release_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"head": commit, "dirty_worktree": bool(status), "status": status}


def _require_clean() -> dict[str, Any]:
    git = _git()
    if git["dirty_worktree"]:
        raise RuntimeError("Package A scientific stages require a clean worktree")
    return git


def _require_cuda(device: str) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Package A scientific stages require CUDA")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("name") != CONFIG_NAME:
        raise ValueError("Package A config name is invalid")
    expected = {
        "schema_version": 1,
        "name": CONFIG_NAME,
        "version": 20260825,
        "program": PROGRAM,
        "code_dim": CODE_DIM,
        "generator_architecture": FORMAL_V3_ARCHITECTURE,
        "generator_checkpoint_sha256": GENERATOR_HASH,
        "primary_evaluator_sha256": PRIMARY_HASH,
        "secondary_evaluator_sha256": SECONDARY_HASH,
        "primary_qualification_sha256": PRIMARY_QUALIFICATION_HASH,
        "secondary_qualification_sha256": SECONDARY_QUALIFICATION_HASH,
        "split_seed": 1234,
        "bank_partition_seed": 32452843,
        "bank_samples_per_class": 250,
        "chi_square_probability": 0.99,
        "inversion_steps": 128,
        "inversion_learning_rate": 0.05,
        "latent_microbatch_size": 16,
        "geometry_samples_per_class": 8,
        "geometry_rank_threshold": 0.001,
        "primary_accuracy_minimum": 0.95,
        "minimum_recall_minimum": 0.90,
        "evaluator_agreement_minimum": 0.95,
        "diversity_ratio_minimum": 0.70,
        "diversity_ratio_maximum": 1.30,
        "coverage_ratio_maximum": 1.50,
        "memorization_anomaly_maximum": 0.05,
        "preflight_wall_limit_seconds": 600.0,
        "memory_limit_bytes": 6442450944,
        "formal_promotion": False,
        "training_authorized": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "candidate_export_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if value != expected:
        raise ValueError("Package A config does not match the registered contract")
    return value


def _dataset(dataset_root: str) -> dict[str, Any]:
    images, labels = load_local_mnist(dataset_root, train=True)
    train_images, train_labels, validation_images, validation_labels, split = split_mnist(
        images, labels, dataset_root
    )
    return {
        "all_images": images,
        "all_labels": labels,
        "train_images": train_images,
        "train_labels": train_labels,
        "validation_images": validation_images,
        "validation_labels": validation_labels,
        "split_hash": split.manifest_hash,
        "dataset_hash": canonical_sha256(
            {"images": tensor_sha256(images), "labels": tensor_sha256(labels)}
        ),
    }


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "generator": args.generator_checkpoint,
        "primary_evaluator": args.primary_evaluator_checkpoint,
        "secondary_evaluator": args.secondary_evaluator_checkpoint,
        "primary_qualification": args.primary_qualification_evidence,
        "secondary_qualification": args.secondary_qualification_evidence,
        "formal_v3_config": str(FORMAL_V3_CONFIG),
    }
    for role, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Package A lineage input is missing: {role}: {path}")
    actual = {role: file_sha256(path) for role, path in paths.items()}
    expected = {
        "generator": config["generator_checkpoint_sha256"],
        "primary_evaluator": config["primary_evaluator_sha256"],
        "secondary_evaluator": config["secondary_evaluator_sha256"],
        "primary_qualification": config["primary_qualification_sha256"],
        "secondary_qualification": config["secondary_qualification_sha256"],
    }
    for role, digest in expected.items():
        if actual[role] != digest:
            raise ValueError(f"Package A {role} hash does not match registered config")
    data = _dataset(args.dataset_root)
    if data["split_hash"] != "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912":
        raise ValueError("Package A dataset split is not the qualified split")
    primary_qualification = _json(args.primary_qualification_evidence)
    secondary_qualification = _json(args.secondary_qualification_evidence)
    if primary_qualification.get("status") != "passed" or secondary_qualification.get("status") != "passed":
        raise ValueError("Package A evaluator qualification is not passed")
    if primary_qualification.get("context", {}).get("dataset_split_manifest_hash") != data["split_hash"]:
        raise ValueError("primary qualification split mismatch")
    if secondary_qualification.get("context", {}).get("dataset_split_manifest_hash") != data["split_hash"]:
        raise ValueError("secondary qualification split mismatch")
    if primary_qualification.get("input_hashes", {}).get("evaluator_checkpoint") != actual["primary_evaluator"]:
        raise ValueError("primary qualification evaluator mismatch")
    secondary_inputs = secondary_qualification.get("input_hashes", {})
    secondary_bound_hashes = {
        secondary_inputs.get("evaluator_checkpoint"),
        secondary_inputs.get("secondary_evaluator"),
        secondary_inputs.get("smooth_evaluator"),
    }
    if actual["secondary_evaluator"] not in secondary_bound_hashes:
        raise ValueError("secondary qualification evaluator mismatch")
    formal = yaml.safe_load(FORMAL_V3_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(formal, dict) or formal.get("architecture") != FORMAL_V3_ARCHITECTURE:
        raise ValueError("formal v3 config architecture mismatch")
    generator_payload = safe_torch_load(args.generator_checkpoint)
    manifest = generator_payload.get("manifest") if isinstance(generator_payload, dict) else None
    state = generator_payload.get("state_dict") if isinstance(generator_payload, dict) else None
    if not isinstance(manifest, dict) or not isinstance(state, dict):
        raise ValueError("v3 generator checkpoint payload is invalid")
    if manifest.get("architecture_kind") != FORMAL_V3_ARCHITECTURE or manifest.get("schema_version") != 3:
        raise ValueError("v3 generator checkpoint architecture mismatch")
    if state_signature(state) != manifest.get("state_signature") or state_content_hash(state) != manifest.get("state_content_hash"):
        raise ValueError("v3 generator checkpoint state hash mismatch")
    if manifest.get("dataset_split_manifest_hash") != data["split_hash"]:
        raise ValueError("v3 generator checkpoint split mismatch")
    if manifest.get("evaluator_checkpoint_hash") != actual["primary_evaluator"]:
        raise ValueError("v3 generator checkpoint evaluator mismatch")
    return {
        "paths": paths,
        "hashes": actual,
        "dataset_hash": data["dataset_hash"],
        "split_hash": data["split_hash"],
        "generator_manifest": manifest,
        "formal_config_sha256": actual["formal_v3_config"],
    }


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, payload)


def _base(stage: str, lineage: dict[str, Any], git: dict[str, Any], *, status: str, passed: bool, classification: str, first_failed_gate: str | None, permitted: tuple[str, ...], evaluator_calls: int = 0, optimizer_created: bool = False, checkpoint_created: bool = False) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_{stage}.v1",
        "stage": stage,
        "status": status,
        "passed": passed,
        "classification": classification,
        "first_failed_gate": first_failed_gate,
        "permitted_next_actions": list(permitted),
        "forbidden_next_actions": list(FORBIDDEN_ACTIONS),
        "input_artifact_hashes": dict(lineage["hashes"]),
        "dataset_hash": lineage["dataset_hash"],
        "dataset_split_hash": lineage["split_hash"],
        "git_head": git["head"],
        "dirty_worktree": git["dirty_worktree"],
        "evaluator_calls": evaluator_calls,
        "optimizer_created": optimizer_created,
        "checkpoint_created": checkpoint_created,
        "synthetic_diagnostic_only": True,
    }


def _partition(images: Tensor, labels: Tensor, *, seed: int, samples_per_class: int) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first_images, first_labels, second_images, second_labels = [], [], [], []
    for label in range(CLASS_COUNT):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() != 2 * samples_per_class:
            raise ValueError("Package A requires exactly two complete validation banks per class")
        order = indices.index_select(0, torch.randperm(indices.numel(), generator=generator))
        for target, selected in ((first_images, order[:samples_per_class]), (second_images, order[samples_per_class:])):
            target.append(images.index_select(0, selected))
        first_labels.append(torch.full((samples_per_class,), label, dtype=torch.long))
        second_labels.append(torch.full((samples_per_class,), label, dtype=torch.long))
    first = (torch.cat(first_images), torch.cat(first_labels))
    second = (torch.cat(second_images), torch.cat(second_labels))
    return first, second


def _class_balanced_indices(labels: Tensor, samples_per_class: int) -> Tensor:
    rows = []
    for label in range(CLASS_COUNT):
        matches = torch.nonzero(labels == label, as_tuple=False).flatten()
        if matches.numel() < samples_per_class:
            raise ValueError("class-balanced selection has too few samples")
        rows.append(matches[:samples_per_class])
    return torch.cat(rows)


def _load_generator(path: str | Path, device: torch.device) -> tuple[HierarchicalSpatialAmplitudeGenerator, str]:
    payload = safe_torch_load(path, map_location=device)
    state = payload["state_dict"]
    generator = HierarchicalSpatialAmplitudeGenerator().to(device)
    generator.load_state_dict(state, strict=True)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator, state_content_hash(state)


def _eval_chunks(model: torch.nn.Module, views: Tensor, counter: Counter[str], *, chunk_size: int = 50) -> tuple[Tensor, Tensor]:
    features, logits = [], []
    with torch.no_grad():
        for start in range(0, views.shape[0], chunk_size):
            feature, logit = model(views[start : start + chunk_size])
            counter["evaluator_forward"] += 1
            features.append(feature)
            logits.append(logit)
    return torch.cat(features), torch.cat(logits)


def _confusion(prediction: Tensor, labels: Tensor) -> list[list[int]]:
    result = torch.zeros((CLASS_COUNT, CLASS_COUNT), dtype=torch.long)
    for truth, predicted in zip(labels.cpu(), prediction.cpu(), strict=True):
        result[int(truth), int(predicted)] += 1
    return result.tolist()


def _classification_metrics(prediction: Tensor, labels: Tensor) -> dict[str, Any]:
    recalls, precisions, accuracies = [], [], []
    for label in range(CLASS_COUNT):
        truth = labels == label
        predicted = prediction == label
        recalls.append(float((prediction[truth] == label).float().mean().item()))
        precisions.append(float((labels[predicted] == label).float().mean().item()) if predicted.any() else 0.0)
        accuracies.append(float((prediction[truth] == label).float().mean().item()))
    return {
        "accuracy": float((prediction == labels).float().mean().item()),
        "recall": recalls,
        "precision": precisions,
        "support_accuracy": accuracies,
        "minimum_recall": min(recalls),
        "confusion_matrix": _confusion(prediction, labels),
    }


def _memorization_summary(value: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in value["per_class"]:
        distances = torch.as_tensor(row["nearest_distances"], dtype=torch.float64)
        rows.append({
            "label": int(row["label"]),
            "threshold": float(row["threshold"]),
            "anomalous_rate": float(row["anomalous_rate"]),
            "nearest_distance_p05": float(torch.quantile(distances, 0.05)),
            "nearest_distance_median": float(torch.quantile(distances, 0.50)),
            "nearest_distance_p95": float(torch.quantile(distances, 0.95)),
        })
    return {"overall_anomalous_rate": value["overall_anomalous_rate"], "per_class": rows}


def _pair_metrics(primary: FrozenMNISTEvaluator, secondary: SmoothMNISTEvaluator, candidate: Tensor, reference: Tensor, labels: Tensor, train: Tensor, train_labels: Tensor, train_primary_features: Tensor | None = None, train_secondary_features: Tensor | None = None) -> tuple[dict[str, Any], int]:
    counter: Counter[str] = Counter()
    p_features, p_logits = _eval_chunks(primary, candidate, counter)
    p_reference, _ = _eval_chunks(primary, reference, counter)
    s_features, s_logits = _eval_chunks(secondary, candidate, counter)
    s_reference, _ = _eval_chunks(secondary, reference, counter)
    p_prediction = p_logits.argmax(1)
    s_prediction = s_logits.argmax(1)
    p = _classification_metrics(p_prediction, labels)
    s = _classification_metrics(s_prediction, labels)
    train_p_features = train_primary_features
    train_s_features = train_secondary_features
    if train_p_features is None or train_s_features is None:
        train_p_features, _ = _eval_chunks(primary, train, counter)
        train_s_features, _ = _eval_chunks(secondary, train, counter)
    p_memo = memorization_statistics(p_features.cpu(), labels.cpu(), train_p_features.cpu(), train_labels.cpu())
    s_memo = memorization_statistics(s_features.cpu(), labels.cpu(), train_s_features.cpu(), train_labels.cpu())
    p_coverage = conditional_coverage_ratio(p_features, candidate, p_reference, reference, labels, epsilon=0.05, iterations=20)
    s_coverage = conditional_coverage_ratio(s_features, candidate, s_reference, reference, labels, epsilon=0.05, iterations=20)
    result = {
        "primary": {**p, "diversity": relative_diversity_ratios(p_features, p_reference, labels), "coverage": p_coverage, "memorization": _memorization_summary(p_memo)},
        "secondary": {**s, "diversity": relative_diversity_ratios(s_features, s_reference, labels), "coverage": s_coverage, "memorization": _memorization_summary(s_memo)},
        "agreement": float((p_prediction == s_prediction).float().mean().item()),
        "candidate_hash": tensor_sha256(candidate),
        "reference_hash": tensor_sha256(reference),
    }
    return result, int(counter["evaluator_forward"])


def _semantic_gate(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    def one(value: dict[str, Any]) -> bool:
        return bool(
            value["accuracy"] >= config["primary_accuracy_minimum"]
            and value["minimum_recall"] >= config["minimum_recall_minimum"]
            and all(config["diversity_ratio_minimum"] <= item <= config["diversity_ratio_maximum"] for item in value["diversity"])
            and value["coverage"] <= config["coverage_ratio_maximum"]
        )
    return bool(one(metrics["primary"]) and one(metrics["secondary"]) and metrics["agreement"] >= config["evaluator_agreement_minimum"])


def _gfa_gate(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    if not _semantic_gate(metrics, config):
        return False
    return all(
        row["anomalous_rate"] <= config["memorization_anomaly_maximum"]
        for evaluator in ("primary", "secondary")
        for row in metrics[evaluator]["memorization"]["per_class"]
    )


def _chi_square_radius(probability: float, *, degrees: int = CODE_DIM) -> float:
    distribution = torch.distributions.Chi2(torch.tensor(float(degrees)))
    low, high = 0.0, float(degrees * 4)
    for _ in range(80):
        middle = (low + high) / 2.0
        if float(distribution.cdf(torch.tensor(middle))) < probability:
            low = middle
        else:
            high = middle
    return math.sqrt(high)


def _project_codes(codes: Tensor, radius: float) -> Tensor:
    norm = codes.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1.0e-12)
    return codes * (radius / norm).clamp_max(1.0)


def _sobel(value: Tensor) -> tuple[Tensor, Tensor]:
    kernel_x = value.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    return F.conv2d(value, kernel_x, padding=1), F.conv2d(value, kernel_y, padding=1)


def _block_pool(value: Tensor, factor: int) -> Tensor:
    if value.shape[-2] % factor or value.shape[-1] % factor:
        raise ValueError("deterministic block pooling requires divisible spatial dimensions")
    height, width = value.shape[-2:]
    return value.reshape(value.shape[0], value.shape[1], height // factor, factor, width // factor, factor).mean((3, 5))


def _adaptive_pool_weights(input_size: int, output_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (input_size, output_size, str(device), str(dtype))
    cached = _POOL_WEIGHTS.get(key)
    if cached is not None:
        return cached
    weights = torch.zeros((output_size, input_size), device=device, dtype=dtype)
    for output_index in range(output_size):
        start = (output_index * input_size) // output_size
        end = ((output_index + 1) * input_size + output_size - 1) // output_size
        weights[output_index, start:end] = 1.0 / float(end - start)
    _POOL_WEIGHTS[key] = weights
    return weights


def _deterministic_adaptive_pool(value: Tensor, output_size: int = 28) -> Tensor:
    height, width = value.shape[-2:]
    height_weights = _adaptive_pool_weights(height, output_size, device=value.device, dtype=value.dtype)
    width_weights = _adaptive_pool_weights(width, output_size, device=value.device, dtype=value.dtype)
    horizontal = torch.einsum("oi,ncij->ncoj", width_weights, value)
    return torch.einsum("pj,ncoj->ncop", height_weights, horizontal)


def _distance_terms(value: Tensor, target: Tensor) -> Tensor:
    value_x, value_y = _sobel(value)
    target_x, target_y = _sobel(target)
    return torch.stack((
        F.smooth_l1_loss(value, target),
        F.smooth_l1_loss(_block_pool(value, 2), _block_pool(target, 2)),
        F.smooth_l1_loss(_block_pool(value, 4), _block_pool(target, 4)),
        F.smooth_l1_loss(value_x, target_x),
        F.smooth_l1_loss(value_y, target_y),
    ))


def _distance_scales(first: Tensor, second: Tensor) -> Tensor:
    return _distance_terms(first, second).detach().clamp_min(1.0e-6)


def _image_loss(value: Tensor, target: Tensor, scales: Tensor) -> Tensor:
    return (_distance_terms(value, target) / scales).mean()


def _generator_views(generator: HierarchicalSpatialAmplitudeGenerator, codes: Tensor, labels: Tensor) -> Tensor:
    return _deterministic_adaptive_pool(generator(codes, labels), 28)


def _generator_views_chunked(generator: HierarchicalSpatialAmplitudeGenerator, codes: Tensor, labels: Tensor, microbatch: int) -> Tensor:
    views = []
    for start in range(0, codes.shape[0], microbatch):
        with torch.inference_mode():
            views.append(_generator_views(generator, codes[start : start + microbatch], labels[start : start + microbatch]))
        _release_cuda()
    return torch.cat(views, dim=0)


def _optimize_codes(generator: HierarchicalSpatialAmplitudeGenerator, initial: Tensor, labels: Tensor, targets: Tensor, *, steps: int, lr: float, microbatch: int, radius: float | None, scales: Tensor, state_path: str | Path | None = None) -> tuple[Tensor, dict[str, Any]]:
    output = torch.empty_like(initial)
    losses = []
    for start in range(0, initial.shape[0], microbatch):
        stop = min(start + microbatch, initial.shape[0])
        codes = initial[start:stop].detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam((codes,), lr=lr)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            value = _generator_views(generator, codes, labels[start:stop])
            loss = _image_loss(value, targets[start:stop], scales)
            if not torch.isfinite(loss):
                raise ValueError("H-LIR produced a non-finite image loss")
            loss.backward()
            if codes.grad is None or not torch.isfinite(codes.grad).all():
                raise ValueError("H-LIR produced a non-finite latent gradient")
            optimizer.step()
            with torch.no_grad():
                if radius is not None:
                    codes.copy_(_project_codes(codes, radius))
        output[start:stop] = codes.detach()
        losses.append(float(loss.detach().item()))
        del value, loss, codes, optimizer
        _release_cuda()
    if state_path is not None:
        atomic_torch_save(state_path, {"codes": output.detach().cpu(), "labels": labels.detach().cpu(), "losses": losses, "latent_only": True})
    return output, {"final_loss_per_chunk": losses, "state_hash": tensor_sha256(output)}


def _arm_metrics(primary: FrozenMNISTEvaluator, secondary: SmoothMNISTEvaluator, generated: Tensor, target: Tensor, reference: Tensor, labels: Tensor, train: Tensor, train_labels: Tensor, config: dict[str, Any], train_primary_features: Tensor | None = None, train_secondary_features: Tensor | None = None) -> tuple[dict[str, Any], int]:
    metrics, calls = _pair_metrics(primary, secondary, generated, reference, labels, train, train_labels, train_primary_features, train_secondary_features)
    metrics["range_gate"] = _semantic_gate(metrics, config)
    metrics["target_pixel_l1"] = float(F.l1_loss(generated, target).item())
    return metrics, calls


def _feature_blocks(value: Tensor) -> tuple[Tensor, ...]:
    sobel_x, sobel_y = _sobel(value)
    dct = []
    yy, xx = torch.meshgrid(torch.arange(28, device=value.device, dtype=value.dtype), torch.arange(28, device=value.device, dtype=value.dtype), indexing="ij")
    for total in range(8):
        for u in range(total + 1):
            v = total - u
            basis = torch.cos(math.pi * (2.0 * xx + 1.0) * u / 56.0) * torch.cos(math.pi * (2.0 * yy + 1.0) * v / 56.0)
            dct.append((value[:, 0] * basis).mean(dim=(-2, -1), keepdim=False))
    return (value.flatten(1), _block_pool(value, 2).flatten(1), _block_pool(value, 4).flatten(1), sobel_x.flatten(1), sobel_y.flatten(1), torch.stack(dct, dim=1))


def _feature_scales(value: Tensor, reference: Tensor) -> Tensor:
    blocks, references = _feature_blocks(value), _feature_blocks(reference)
    return torch.stack([torch.cat((left, right), dim=0).square().mean().sqrt().clamp_min(1.0e-6) for left, right in zip(blocks, references, strict=True)])


def _feature_vector(value: Tensor, scales: Tensor) -> Tensor:
    return torch.cat([block.flatten(1) / scale for block, scale in zip(_feature_blocks(value), scales, strict=True)], dim=1)


def _effective_rank(singular: Tensor, threshold: float) -> dict[str, Any]:
    singular = singular.detach().cpu()
    maximum = float(singular[0].item()) if singular.numel() else 0.0
    nominal = singular[singular >= maximum * threshold] if maximum > 0 else singular[:0]
    effective = float(nominal.sum().square().div(nominal.square().sum().clamp_min(1.0e-12)).item()) if nominal.numel() else 0.0
    condition = float((nominal[0] / nominal[-1]).item()) if nominal.numel() else float("inf")
    return {"nominal_rank": int(nominal.numel()), "effective_rank": effective, "condition_number": condition, "singular_values": [float(item) for item in singular]}


def _jacobian(generator: HierarchicalSpatialAmplitudeGenerator, codes: Tensor, labels: Tensor, scales: Tensor, threshold: float) -> list[dict[str, Any]]:
    rows = []
    def function(current: Tensor, current_label: Tensor) -> Tensor:
        return _feature_vector(
            _generator_views(generator, current.unsqueeze(0), current_label.unsqueeze(0)), scales
        ).squeeze(0)

    for code, label in zip(codes.detach(), labels, strict=True):
        matrix = torch.autograd.functional.jacobian(
            lambda current: function(current, label),
            code,
            create_graph=False,
            vectorize=True,
            strategy="forward-mode",
        )
        if matrix.ndim != 2 or not torch.isfinite(matrix).all():
            raise ValueError("H-LGS Jacobian is invalid")
        direction = torch.ones_like(code)
        direction = direction / direction.norm()
        epsilon = 1.0e-3
        with torch.no_grad():
            finite_difference = (function(code + epsilon * direction, label) - function(code - epsilon * direction, label)) / (2.0 * epsilon)
        projected = matrix @ direction
        fd_error = float((finite_difference - projected).abs().max().div(projected.abs().max().clamp_min(1.0e-8)).item())
        row = _effective_rank(torch.linalg.svdvals(matrix), threshold)
        row["finite_difference_relative_max_error"] = fd_error
        rows.append(row)
        del matrix, finite_difference, projected
    return rows


def _real_pca(value: Tensor, labels: Tensor, scales: Tensor, threshold: float) -> dict[str, Any]:
    features = _feature_vector(value, scales).detach().cpu()
    rows = []
    for label in range(CLASS_COUNT):
        centered = features[labels.cpu() == label] - features[labels.cpu() == label].mean(0, keepdim=True)
        rows.append(_effective_rank(torch.linalg.svdvals(centered), threshold))
    return {"per_class": rows}


def _geometry_fixture() -> dict[str, Any]:
    matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    spectrum = torch.linalg.svdvals(matrix)
    zero = torch.zeros((4, 4))
    return {"known_spectrum": [float(value) for value in spectrum], "known_effective_rank": _effective_rank(spectrum, 1.0e-3), "constant_output_nominal_rank": int((torch.linalg.svdvals(zero) > 0).sum())}


def _load_evaluators(args: argparse.Namespace, device: torch.device) -> tuple[FrozenMNISTEvaluator, SmoothMNISTEvaluator]:
    primary, _ = load_frozen_evaluator(args.primary_evaluator_checkpoint, device=device)
    payload = safe_torch_load(args.secondary_evaluator_checkpoint, map_location=device)
    if payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind or payload.get("split_manifest_hash") != "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912":
        raise ValueError("secondary evaluator lineage is invalid")
    secondary = SmoothMNISTEvaluator().to(device)
    secondary.load_state_dict(payload["state_dict"], strict=True)
    secondary.freeze()
    return primary, secondary


def _gate_audit(args: argparse.Namespace, config: dict[str, Any], lineage: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(args.device)
    data = _dataset(args.dataset_root)
    first, second = _partition(data["validation_images"], data["validation_labels"], seed=config["bank_partition_seed"], samples_per_class=config["bank_samples_per_class"])
    primary, secondary = _load_evaluators(args, device)
    train_images, train_labels = data["train_images"].to(device), data["train_labels"].to(device)
    train_counter = Counter()
    p_train, _ = _eval_chunks(primary, train_images, train_counter)
    s_train, _ = _eval_chunks(secondary, train_images, train_counter)
    banks = []
    for bank_index, (candidate, reference) in enumerate(((first, second), (second, first))):
        candidate_images, labels = candidate[0].to(device), candidate[1].to(device)
        reference_images = reference[0].to(device)
        p_features, p_logits = _eval_chunks(primary, candidate_images, train_counter)
        p_reference, _ = _eval_chunks(primary, reference_images, train_counter)
        s_features, s_logits = _eval_chunks(secondary, candidate_images, train_counter)
        s_reference, _ = _eval_chunks(secondary, reference_images, train_counter)
        p_prediction, s_prediction = p_logits.argmax(1), s_logits.argmax(1)
        p = _classification_metrics(p_prediction, labels)
        s = _classification_metrics(s_prediction, labels)
        p_memo = memorization_statistics(p_features.cpu(), labels.cpu(), p_train.cpu(), train_labels.cpu())
        s_memo = memorization_statistics(s_features.cpu(), labels.cpu(), s_train.cpu(), train_labels.cpu())
        metrics = {
            "bank": bank_index,
            "candidate_hash": tensor_sha256(candidate_images.cpu()),
            "reference_hash": tensor_sha256(reference_images.cpu()),
            "primary": {**p, "diversity": relative_diversity_ratios(p_features, p_reference, labels), "coverage": conditional_coverage_ratio(p_features, candidate_images, p_reference, reference_images, labels, epsilon=0.05, iterations=20), "memorization": _memorization_summary(p_memo)},
            "secondary": {**s, "diversity": relative_diversity_ratios(s_features, s_reference, labels), "coverage": conditional_coverage_ratio(s_features, candidate_images, s_reference, reference_images, labels, epsilon=0.05, iterations=20), "memorization": _memorization_summary(s_memo)},
            "agreement": float((p_prediction == s_prediction).float().mean().item()),
        }
        metrics["passed"] = _gfa_gate(metrics, config)
        banks.append(metrics)
    shifted_labels = ((first[1] + 1) % CLASS_COUNT).to(device)
    shifted_prediction = _eval_chunks(primary, first[0].to(device), train_counter)[1].argmax(1)
    negative = _classification_metrics(shifted_prediction, shifted_labels)
    negative["rejected"] = not (negative["accuracy"] >= config["primary_accuracy_minimum"] and negative["minimum_recall"] >= config["minimum_recall_minimum"])
    passed = all(row["passed"] for row in banks) and negative["rejected"]
    classification = "gate_feasibility_supported" if passed else "gate_feasibility_failed"
    first_failed = None if passed else ("label_shift_negative_control" if not negative["rejected"] else "real_real_admission_gate")
    return {"banks": banks, "negative_control": negative, "passed": passed, "classification": classification, "first_failed_gate": first_failed, "evaluator_calls": int(train_counter["evaluator_forward"]), "partition": {"disjoint": True, "bank_a_hash": tensor_sha256(first[0]), "bank_b_hash": tensor_sha256(second[0])}}


def _preflight(args: argparse.Namespace, config: dict[str, Any], lineage: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    if gate.get("classification") != "gate_feasibility_supported":
        raise ValueError("H-LIR preflight requires passed H-GFA")
    _require_cuda(args.device)
    device = torch.device(args.device)
    data = _dataset(args.dataset_root)
    first, _ = _partition(data["validation_images"], data["validation_labels"], seed=config["bank_partition_seed"], samples_per_class=config["bank_samples_per_class"])
    generator, before_state = _load_generator(args.generator_checkpoint, device)
    selected = _class_balanced_indices(first[1], 8)
    labels = first[1].index_select(0, selected).to(device)
    targets = first[0].index_select(0, selected).to(device)
    initial = torch.randn((labels.numel(), CODE_DIM), generator=torch.Generator(device="cpu").manual_seed(7103)).to(device)
    scales = _distance_scales(targets, targets.roll(1, 0))
    radius = _chi_square_radius(config["chi_square_probability"])
    codes = initial[:16].detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam((codes,), lr=config["inversion_learning_rate"])
    losses, latent_updates = [], []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = _image_loss(_generator_views(generator, codes, labels[:16]), targets[:16], scales)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            codes.copy_(_project_codes(codes, radius))
        losses.append(float(loss.detach().item()))
        latent_updates.append(float(codes.detach().sub(initial[:16]).abs().max().item()))
    resumed = initial[:16].detach().clone().requires_grad_(True)
    resumed_optimizer = torch.optim.Adam((resumed,), lr=config["inversion_learning_rate"])
    for _ in range(2):
        resumed_optimizer.zero_grad(set_to_none=True)
        value = _image_loss(_generator_views(generator, resumed, labels[:16]), targets[:16], scales)
        value.backward()
        resumed_optimizer.step()
        with torch.no_grad():
            resumed.copy_(_project_codes(resumed, radius))
    checkpoint = {"codes": resumed.detach().cpu(), "optimizer": copy.deepcopy(resumed_optimizer.state_dict()), "step": 2}
    restored = checkpoint["codes"].to(device).detach().clone().requires_grad_(True)
    restored_optimizer = torch.optim.Adam((restored,), lr=config["inversion_learning_rate"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    for _ in range(1):
        restored_optimizer.zero_grad(set_to_none=True)
        value = _image_loss(_generator_views(generator, restored, labels[:16]), targets[:16], scales)
        value.backward()
        restored_optimizer.step()
        with torch.no_grad():
            restored.copy_(_project_codes(restored, radius))
    resumed_optimizer.zero_grad(set_to_none=True)
    value = _image_loss(_generator_views(generator, resumed, labels[:16]), targets[:16], scales)
    value.backward()
    resumed_optimizer.step()
    with torch.no_grad():
        resumed.copy_(_project_codes(resumed, radius))
    resume_max_abs_diff = float(restored.detach().sub(resumed.detach()).abs().max().item())
    resume_exact = torch.equal(restored.detach(), resumed.detach())
    negative_labels = (labels + 1) % CLASS_COUNT
    negative_codes = initial.detach().clone().requires_grad_(True)
    negative_optimizer = torch.optim.Adam((negative_codes,), lr=config["inversion_learning_rate"])
    for _ in range(3):
        negative_optimizer.zero_grad(set_to_none=True)
        loss = _image_loss(_generator_views(generator, negative_codes, negative_labels), targets, scales)
        loss.backward()
        negative_optimizer.step()
        with torch.no_grad():
            negative_codes.copy_(_project_codes(negative_codes, radius))
    primary, secondary = _load_evaluators(args, device)
    counter = Counter()
    negative_predictions = _eval_chunks(primary, _generator_views(generator, negative_codes.detach(), negative_labels), counter)[1].argmax(1)
    negative_metrics = _classification_metrics(negative_predictions, labels)
    after_state = state_content_hash({name: value.detach().cpu() for name, value in generator.state_dict().items()})
    memory = int(torch.cuda.max_memory_allocated(device))
    passed = bool(all(torch.isfinite(torch.tensor(losses))) and max(latent_updates) > 0.0 and before_state == after_state and resume_exact and memory < config["memory_limit_bytes"] and not (negative_metrics["accuracy"] >= config["primary_accuracy_minimum"] and negative_metrics["minimum_recall"] >= config["minimum_recall_minimum"]))
    return {"passed": passed, "classification": "preflight_passed" if passed else "invalid_harness", "first_failed_gate": None if passed else "preflight_update_resume_or_negative_control", "losses": losses, "latent_updates": latent_updates, "resume_exact": resume_exact, "resume_max_abs_diff": resume_max_abs_diff, "negative_control": negative_metrics, "generator_state_before": before_state, "generator_state_after": after_state, "evaluator_calls": int(counter["evaluator_forward"]), "optimizer_created": True, "peak_memory_bytes": memory, "memory_gate": memory < config["memory_limit_bytes"], "radius": radius}


def _audit(args: argparse.Namespace, config: dict[str, Any], lineage: dict[str, Any], gate: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    if gate.get("classification") != "gate_feasibility_supported" or not preflight.get("passed"):
        raise ValueError("H-LIR audit requires passed H-GFA and preflight")
    _require_cuda(args.device)
    device = torch.device(args.device)
    data = _dataset(args.dataset_root)
    first, second = _partition(data["validation_images"], data["validation_labels"], seed=config["bank_partition_seed"], samples_per_class=config["bank_samples_per_class"])
    generator, before_state = _load_generator(args.generator_checkpoint, device)
    primary, secondary = _load_evaluators(args, device)
    train_images = data["train_images"].to(device)
    train_labels = data["train_labels"].to(device)
    train_counter = Counter()
    train_primary_features, _ = _eval_chunks(primary, train_images, train_counter)
    train_secondary_features, _ = _eval_chunks(secondary, train_images, train_counter)
    radius = _chi_square_radius(config["chi_square_probability"])
    all_arms, evaluator_calls = {}, int(train_counter["evaluator_forward"])
    latent_payload: dict[str, Tensor] = {}
    for bank_index, (target_bank, reference_bank, code_seed) in enumerate(((first, second, 32452843), (second, first, 49979687))):
        targets, labels, reference = target_bank[0].to(device), target_bank[1].to(device), reference_bank[0].to(device)
        initial = torch.randn((labels.numel(), CODE_DIM), generator=torch.Generator(device="cpu").manual_seed(code_seed)).to(device)
        scales = _distance_scales(targets, reference)
        constrained, c_info = _optimize_codes(generator, initial, labels, targets, steps=config["inversion_steps"], lr=config["inversion_learning_rate"], microbatch=config["latent_microbatch_size"], radius=radius, scales=scales)
        unconstrained, u_info = _optimize_codes(generator, initial, labels, targets, steps=config["inversion_steps"], lr=config["inversion_learning_rate"], microbatch=config["latent_microbatch_size"], radius=None, scales=scales)
        arms = {}
        for name, codes, info in (("random", initial, {"state_hash": tensor_sha256(initial)}), ("constrained", constrained, c_info), ("unconstrained", unconstrained, u_info)):
            generated = _generator_views_chunked(generator, codes, labels, config["latent_microbatch_size"])
            metrics, calls = _arm_metrics(primary, secondary, generated, targets, reference, labels, train_images, train_labels, config, train_primary_features, train_secondary_features)
            evaluator_calls += calls
            metrics["inversion"] = info
            metrics["code_norm_squared_mean"] = float(codes.square().sum(1).mean().item())
            arms[name] = metrics
            latent_payload[f"bank_{bank_index}_{name}"] = codes.detach().cpu()
            del generated, metrics
            _release_cuda()
        all_arms[f"bank_{bank_index}"] = arms
    state_after = state_content_hash({name: value.detach().cpu() for name, value in generator.state_dict().items()})
    arm_pass = {name: all(all_arms[f"bank_{index}"][name]["range_gate"] for index in (0, 1)) for name in ("random", "constrained", "unconstrained")}
    if arm_pass["random"]:
        classification = "generator_distribution_supported_under_gaussian_prior"
        findings = ["gaussian_prior_supported"]
    elif arm_pass["constrained"]:
        classification = "generator_range_supported"
        findings = ["prior_occupancy_suspected"]
    elif arm_pass["unconstrained"]:
        classification = "generator_range_supported_outside_prior_mass"
        findings = ["prior_support_mismatch_supported"]
    else:
        classification = "generator_range_incomplete"
        findings = ["generator_inductive_bias_suspected"]
    del primary, secondary, train_images, train_labels, train_primary_features, train_secondary_features
    _release_cuda()
    geometry = _geometry(generator, latent_payload, first, second, config)
    payload_path = Path(args.output).with_suffix(".latent.pt")
    atomic_torch_save(payload_path, {"schema": f"{SCHEMA_PREFIX}_latent.v1", "latents": latent_payload, "generator_hash": lineage["hashes"]["generator"], "latent_only": True})
    passed = state_after == before_state and geometry["passed"]
    if not passed:
        classification = "invalid_harness"
    return {"passed": passed, "classification": classification, "findings": findings, "first_failed_gate": None if passed else "generator_hash_or_geometry_harness", "arms": all_arms, "arm_pass": arm_pass, "geometry": geometry, "latent_artifact": str(payload_path), "latent_artifact_sha256": file_sha256(payload_path), "generator_state_before": before_state, "generator_state_after": state_after, "evaluator_calls": evaluator_calls, "optimizer_created": True, "checkpoint_created": False}


def _geometry(generator: HierarchicalSpatialAmplitudeGenerator, latent_payload: dict[str, Tensor], first: tuple[Tensor, Tensor], second: tuple[Tensor, Tensor], config: dict[str, Any]) -> dict[str, Any]:
    device = next(generator.parameters()).device
    reference = torch.cat((first[0], second[0]), dim=0).to(device)
    reference_labels = torch.cat((first[1], second[1]), dim=0).to(device)
    scales = _feature_scales(reference, reference.roll(1, 0))
    rows = {}
    geometry_indices = (_class_balanced_indices(first[1], config["geometry_samples_per_class"]), _class_balanced_indices(second[1], config["geometry_samples_per_class"]))
    labels_by_bank = (first[1].index_select(0, geometry_indices[0]).to(device), second[1].index_select(0, geometry_indices[1]).to(device))
    for name in ("random", "constrained", "unconstrained"):
        per_bank = []
        for bank_index in range(2):
            count = config["geometry_samples_per_class"] * CLASS_COUNT
            codes = latent_payload[f"bank_{bank_index}_{name}"].index_select(0, geometry_indices[bank_index]).to(device)
            bank_labels = labels_by_bank[bank_index][:count]
            per_bank.append({"samples": count, "per_sample": _jacobian(generator, codes, bank_labels, scales, config["geometry_rank_threshold"])})
        rows[name] = {"banks": per_bank}
    fixture = _geometry_fixture()
    finite_difference_ok = all(
        row["finite_difference_relative_max_error"] <= 1.0e-2
        for arm in rows.values()
        for bank in arm["banks"]
        for row in bank["per_sample"]
    )
    return {"passed": fixture["constant_output_nominal_rank"] == 0 and fixture["known_spectrum"] == [4.0, 3.0, 2.0, 1.0] and finite_difference_ok, "fixture": fixture, "real_pca": _real_pca(reference, reference_labels, scales, config["geometry_rank_threshold"]), "arms": rows, "evaluator_calls": 0, "feature_scales": [float(value) for value in scales], "finite_difference_gate": finite_difference_ok}


def _dispatch_validate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    git = _require_clean()
    lineage = _lineage(args, config)
    payload = _base("validation", lineage, git, status="valid", passed=True, classification="lineage_validated", first_failed_gate=None, permitted=("gate-audit",))
    payload["config_sha256"] = file_sha256(args.config)
    payload["config"] = config
    payload["checkpoint_hashes"] = {"generator": lineage["hashes"]["generator"]}
    payload["evaluator_hashes"] = {"primary": lineage["hashes"]["primary_evaluator"], "secondary": lineage["hashes"]["secondary_evaluator"]}
    _write(args.output, payload)
    return 0


def dispatch(args: argparse.Namespace) -> int:
    config = _config(args.config)
    action = args.research_action
    if action == "validate":
        return _dispatch_validate(args, config)
    if action == "gate-audit":
        git = _require_clean()
        lineage = _lineage(args, config)
        validation = _json(args.validation_evidence)
        if validation.get("schema") != f"{SCHEMA_PREFIX}_validation.v1" or not validation.get("passed"):
            raise ValueError("gate-audit requires passed validation evidence")
        _require_cuda(args.device)
        start = time.monotonic()
        result = _gate_audit(args, config, lineage)
        payload = _base("gate_audit", lineage, git, status="valid", passed=bool(result["passed"]), classification=result["classification"], first_failed_gate=result["first_failed_gate"], permitted=("preflight", "audit") if result["passed"] else ("evaluator_contract_audit_only",), evaluator_calls=result["evaluator_calls"])
        payload.update(result)
        payload["elapsed_seconds"] = time.monotonic() - start
        payload["validation_evidence_sha256"] = file_sha256(args.validation_evidence)
        _write(args.output, payload)
        return 0
    if action == "preflight":
        git = _require_clean()
        lineage = _lineage(args, config)
        gate = _json(args.gate_audit_evidence)
        if gate.get("schema") != f"{SCHEMA_PREFIX}_gate_audit.v1" or not gate.get("passed"):
            raise ValueError("preflight requires passed H-GFA evidence")
        result = _preflight(args, config, lineage, gate)
        payload = _base("preflight", lineage, git, status="valid" if result["passed"] else "invalid_harness", passed=result["passed"], classification=result["classification"], first_failed_gate=result["first_failed_gate"], permitted=("audit",) if result["passed"] else (), evaluator_calls=result["evaluator_calls"], optimizer_created=True)
        payload.update(result)
        payload["gate_audit_evidence_sha256"] = file_sha256(args.gate_audit_evidence)
        _write(args.output, payload)
        return 0
    if action == "audit":
        git = _require_clean()
        lineage = _lineage(args, config)
        gate, preflight = _json(args.gate_audit_evidence), _json(args.preflight_evidence)
        if gate.get("schema") != f"{SCHEMA_PREFIX}_gate_audit.v1" or not gate.get("passed") or preflight.get("schema") != f"{SCHEMA_PREFIX}_preflight.v1" or not preflight.get("passed"):
            raise ValueError("audit requires passed H-GFA and preflight evidence")
        result = _audit(args, config, lineage, gate, preflight)
        payload = _base("audit", lineage, git, status="valid" if result["passed"] else "invalid_harness", passed=result["passed"], classification=result["classification"], first_failed_gate=result["first_failed_gate"], permitted=("report",) if result["passed"] else (), evaluator_calls=result["evaluator_calls"], optimizer_created=True)
        payload.update(result)
        payload["gate_audit_evidence_sha256"] = file_sha256(args.gate_audit_evidence)
        payload["preflight_evidence_sha256"] = file_sha256(args.preflight_evidence)
        _write(args.output, payload)
        return 0
    if action == "report":
        audit = _json(args.audit_evidence)
        if audit.get("schema") != f"{SCHEMA_PREFIX}_audit.v1" or audit.get("status") != "valid":
            raise ValueError("report requires valid audit evidence")
        git = _git()
        report = {"schema": f"{SCHEMA_PREFIX}_report.v1", "status": "valid", "classification": audit["classification"], "findings": audit.get("findings", []), "first_failed_gate": audit.get("first_failed_gate"), "permitted_next_actions": ["closeout"], "forbidden_next_actions": list(FORBIDDEN_ACTIONS), "audit_evidence_sha256": file_sha256(args.audit_evidence), "git_head": git["head"], "dirty_worktree": git["dirty_worktree"], "measurement_summary": {"arm_pass": audit.get("arm_pass"), "geometry": audit.get("geometry"), "evaluator_calls": audit.get("evaluator_calls")}}
        _write(args.output, report)
        return 0
    if action == "closeout":
        report = _json(args.report_evidence)
        if report.get("schema") != f"{SCHEMA_PREFIX}_report.v1" or report.get("status") != "valid":
            raise ValueError("closeout requires valid report evidence")
        git = _git()
        closeout = {"schema": f"{SCHEMA_PREFIX}_closeout.v1", "status": "valid", "classification": report["classification"], "first_failed_gate": report.get("first_failed_gate"), "permitted_next_actions": report.get("permitted_next_actions", []), "forbidden_next_actions": list(FORBIDDEN_ACTIONS), "report_evidence_sha256": file_sha256(args.report_evidence), "git_head": git["head"], "dirty_worktree": git["dirty_worktree"], "promotion": "forbidden", "optical_training": "forbidden", "fiber_training": "forbidden"}
        _write(args.output, closeout)
        synthesis = Path(args.output).with_name("electronic_distribution_range_synthesis.md")
        synthesis.write_text(f"# Electronic distribution range synthesis\n\n- Classification: `{report['classification']}`\n- Report: `{file_sha256(args.report_evidence)}`\n- Package B teacher audit: permitted only as a separately reviewed next action.\n- Optical, phase, quadrature, OLS and fiber training: forbidden.\n", encoding="utf-8")
        return 0
    raise ValueError(f"unknown Package A action: {action}")
