"""H-DPGC: deterministic direct-prior capacity canary.

The canary is deliberately not a trainable model.  It fits a fixed, ephemeral
classwise rank-8 randomized SVD in logit pixel space and maps standard Gaussian
codes directly to images.  It is a diagnostic admission test for the
``label + 8D N(0,I)`` interface; it never creates an optimizer or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ...evidence import atomic_write_json, canonical_sha256, file_sha256, safe_torch_load, tensor_sha256
from ...student.conditional import camera_code_fraction, relative_diversity_ratios
from ...student.conditional_acceptance import conditional_coverage_ratio
from ...student.evaluator import SmoothMNISTEvaluator, load_frozen_evaluator, load_local_mnist, stratified_mnist_indices
from .electronic_o0_wae import LOCKED_SEEDS, _balanced_reference, _eval_model, _recall
from .lifecycle_artifacts import prepare_args, stage_action_decision, validate_manifest_namespace


SCHEMA_PREFIX = "electronic_o0_direct_prior_capacity_canary"
PROGRAM = "electronic-o0-direct-prior-capacity-canary"
HARNESS_SEED = 8771
CODE_DIM = 8
RANK = 8
OVERSAMPLE = 8
POWER_ITERATIONS = 4
LOGIT_EPSILON = 1.0 / 255.0
LOCKED_PANEL_SEEDS = LOCKED_SEEDS
HISTORICAL_SEED = 90210
PANEL_SAMPLES_PER_CLASS = 50
PREFLIGHT_SAMPLES_PER_CLASS = 32
OFFICIAL_COUNT = 10_000
SECONDARY_ARCHITECTURE = SmoothMNISTEvaluator.architecture_kind
MEMORIZATION_PASS_MINIMUM = 0.95
JACOBIAN_PASS_MINIMUM = 0.80


@dataclass(frozen=True)
class DirectPriorCanary:
    means: Tensor
    bases: Tensor
    scales: Tensor
    image_shape: tuple[int, ...]

    def sample(self, labels: Tensor, codes: Tensor, *, chunk_size: int = 512) -> Tensor:
        if labels.ndim != 1 or codes.ndim != 2 or codes.shape != (labels.numel(), CODE_DIM):
            raise ValueError("direct-prior canary expects labels [N] and codes [N,8]")
        if labels.numel() == 0 or labels.dtype not in (torch.int64, torch.int32):
            raise ValueError("direct-prior labels are invalid")
        labels = labels.to(device=self.means.device, dtype=torch.long)
        codes = codes.to(device=self.means.device, dtype=self.means.dtype)
        if not torch.isfinite(codes).all():
            raise ValueError("direct-prior codes must be finite")
        if int(labels.min()) < 0 or int(labels.max()) >= self.means.shape[0]:
            raise ValueError("direct-prior labels are out of range")
        rows: list[Tensor] = []
        for start in range(0, labels.numel(), chunk_size):
            current_labels = labels[start : start + chunk_size]
            current_codes = codes[start : start + chunk_size]
            logits = self.means.index_select(0, current_labels)
            basis = self.bases.index_select(0, current_labels)
            scales = self.scales.index_select(0, current_labels)
            logits = logits + torch.bmm(
                basis,
                (current_codes * scales).unsqueeze(-1),
            ).squeeze(-1)
            rows.append(torch.sigmoid(logits).reshape(-1, *self.image_shape))
        result = torch.cat(rows, dim=0)
        if not torch.isfinite(result).all():
            raise ValueError("direct-prior canary produced non-finite images")
        return result


def _git() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "clean": not status, "status": status}


def _config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "name": "electronic_o0_direct_prior_capacity_canary_v1",
        "version": 20260824,
        "code_dim": CODE_DIM,
        "direct_prior_capacity_canary_seed": HARNESS_SEED,
        "rank": RANK,
        "randomized_svd_oversample": OVERSAMPLE,
        "randomized_svd_iterations": POWER_ITERATIONS,
        "logit_epsilon": LOGIT_EPSILON,
        "historical_seed": HISTORICAL_SEED,
        "locked_panel_seeds": list(LOCKED_PANEL_SEEDS),
        "panel_samples_per_class": PANEL_SAMPLES_PER_CLASS,
        "preflight_samples_per_class": PREFLIGHT_SAMPLES_PER_CLASS,
        "official_count": OFFICIAL_COUNT,
        "camera_code_fraction_minimum": 0.25,
        "saturation_maximum": 0.08,
        "evaluator_accuracy_minimum": 0.95,
        "evaluator_minimum_recall": 0.90,
        "evaluator_agreement_minimum": 0.95,
        "diversity_ratio_minimum": 0.70,
        "diversity_ratio_maximum": 1.30,
        "coverage_ratio_maximum": 1.50,
        "memorization_pass_fraction_minimum": MEMORIZATION_PASS_MINIMUM,
        "jacobian_rank_pass_fraction_minimum": JACOBIAN_PASS_MINIMUM,
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if not isinstance(payload, dict) or any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("H-DPGC config is not the registered v1 contract")
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
        raise RuntimeError("H-DPGC requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H-DPGC formal lifecycle requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _logit(images: Tensor) -> Tensor:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("MNIST images must have shape [N,1,H,W]")
    values = images.to(torch.float32).flatten(1).clamp(LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return torch.logit(values)


def _canonical_basis(basis: Tensor) -> Tensor:
    indices = basis.abs().argmax(dim=0)
    signs = basis[indices, torch.arange(basis.shape[1], device=basis.device)].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return basis * signs


def _fit_rank8(images: Tensor, labels: Tensor, *, seed: int = HARNESS_SEED) -> DirectPriorCanary:
    if images.ndim != 4 or labels.ndim != 1 or images.shape[0] != labels.numel():
        raise ValueError("canary fit shape mismatch")
    if images.shape[0] < 2 or labels.unique().numel() != 10:
        raise ValueError("canary fit requires all ten classes")
    device = images.device
    flat = _logit(images)
    means, bases, scales = [], [], []
    for label in range(10):
        rows = flat[labels.to(device=device) == label]
        if rows.shape[0] <= RANK:
            raise ValueError("canary class has too few samples")
        mean = rows.mean(dim=0)
        centered = rows - mean
        generator = torch.Generator(device="cpu").manual_seed(seed + label)
        q = torch.randn((centered.shape[1], RANK + OVERSAMPLE), generator=generator, dtype=centered.dtype).to(device)
        q, _ = torch.linalg.qr(q, mode="reduced")
        for _ in range(POWER_ITERATIONS):
            q, _ = torch.linalg.qr(centered.T @ (centered @ q), mode="reduced")
        projected = centered @ q
        _, singular, vh = torch.linalg.svd(projected, full_matrices=False)
        basis = q @ vh[:RANK].T
        basis = _canonical_basis(basis)
        scale = singular[:RANK].clamp_min(1.0e-8) / (rows.shape[0] - 1) ** 0.5
        means.append(mean)
        bases.append(basis)
        scales.append(scale)
    result = DirectPriorCanary(torch.stack(means), torch.stack(bases), torch.stack(scales), tuple(images.shape[1:]))
    if not all(torch.isfinite(value).all() for value in (result.means, result.bases, result.scales)):
        raise ValueError("canary parameters are non-finite")
    return result


def _codes(seed: int, labels: Tensor, *, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((labels.numel(), CODE_DIM), generator=generator).to(device)


def _panel(seed: int, *, samples_per_class: int, device: torch.device) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.arange(10, device=device).repeat_interleave(samples_per_class)
    codes = torch.randn((labels.numel(), CODE_DIM), generator=generator).to(device)
    return codes, labels


def _secondary(path: str | Path, *, device: torch.device) -> SmoothMNISTEvaluator:
    payload = safe_torch_load(path)
    if not isinstance(payload, dict) or payload.get("architecture_kind") != SECONDARY_ARCHITECTURE:
        raise ValueError("H-DPGC secondary evaluator qualification is invalid")
    model = SmoothMNISTEvaluator().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.freeze()
    return model


def _memorization(generated: Tensor, generated_labels: Tensor, train: Tensor, train_labels: Tensor, validation: Tensor, validation_labels: Tensor) -> dict[str, Any]:
    generated_u8 = (generated.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    train_u8 = (train.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    hashes = {tensor_sha256(row) for row in train_u8}
    exact = sum(tensor_sha256(row) in hashes for row in generated_u8)
    thresholds, passes = [], []
    train_flat, validation_flat, generated_flat = train.flatten(1), validation.flatten(1), generated.flatten(1)
    for label in range(10):
        train_rows = train_flat[train_labels.cpu() == label]
        heldout_rows = validation_flat[validation_labels.cpu() == label]
        generated_rows = generated_flat[generated_labels.cpu() == label]
        heldout_dist = torch.cdist(heldout_rows, train_rows).square().min(1).values
        threshold = float(torch.quantile(heldout_dist, 0.05))
        distances = torch.cdist(generated_rows, train_rows).square().min(1).values
        thresholds.append(threshold)
        passes.append(float((distances >= threshold).float().mean()))
    return {"exact_duplicate_count": exact, "per_class_thresholds": thresholds, "per_class_pass_fraction": passes, "pass_fraction": sum(passes) / 10.0, "passed": exact == 0 and min(passes) >= MEMORIZATION_PASS_MINIMUM}


def _jacobian(canary: DirectPriorCanary, *, device: torch.device) -> dict[str, Any]:
    rows = []
    for label in range(10):
        codes = torch.zeros((1, CODE_DIM), device=device)
        images = canary.sample(torch.tensor([label], device=device), codes).flatten()
        derivative = torch.diag(images * (1.0 - images)) @ (canary.bases[label] * canary.scales[label])
        singular = torch.linalg.svdvals(derivative)
        threshold = singular[0].clamp_min(1.0e-12) * 1.0e-3
        rows.append({"label": label, "rank_1e-3": int((singular >= threshold).sum()), "singular_values": [float(value) for value in singular.detach().cpu()]})
    return {"points": rows, "rank_1e-3_pass_fraction": sum(row["rank_1e-3"] >= 7 for row in rows) / 10.0}


def _eval_pair(primary: torch.nn.Module, secondary: torch.nn.Module, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    p_features, p_logits = _eval_model(primary, images)
    s_features, s_logits = _eval_model(secondary, images)
    return p_features, p_logits, s_features, s_logits


def _panel_measurement(canary: DirectPriorCanary, primary: torch.nn.Module, secondary: torch.nn.Module, test_images: Tensor, test_labels: Tensor, train_images: Tensor, train_labels: Tensor, validation_images: Tensor, validation_labels: Tensor, *, seed: int, samples_per_class: int, device: torch.device) -> dict[str, Any]:
    codes, labels = _panel(seed, samples_per_class=samples_per_class, device=device)
    views = canary.sample(labels, codes)
    reference, reference_labels = _balanced_reference(test_images, test_labels, seed=seed + 17, count=samples_per_class)
    p_features, p_logits, s_features, s_logits = _eval_pair(primary, secondary, views)
    p_ref, _ = _eval_model(primary, reference.to(device))
    s_ref, _ = _eval_model(secondary, reference.to(device))
    p_pred, s_pred = p_logits.argmax(1), s_logits.argmax(1)
    shuffle_generator = torch.Generator(device="cpu").manual_seed(8675309 + seed)
    shuffled_labels = labels.cpu().index_select(0, torch.randperm(labels.numel(), generator=shuffle_generator)).to(device)
    shuffled = canary.sample(shuffled_labels, codes)
    shuffled_primary = _eval_model(primary, shuffled)[1].argmax(1)
    zero = canary.sample(labels, torch.zeros_like(codes))
    repeated = tensor_sha256(canary.sample(labels, codes)) == tensor_sha256(views)
    # Evaluator features are returned on CPU by the shared evaluation helper;
    # keep the paired image tensors on that same device for index_select.
    coverage_views = views.cpu()
    coverage_reference = reference.cpu()
    p_coverage = conditional_coverage_ratio(p_features, coverage_views, p_ref, coverage_reference, labels.cpu(), epsilon=0.05, iterations=20, seed=seed)
    s_coverage = conditional_coverage_ratio(s_features, coverage_views, s_ref, coverage_reference, labels.cpu(), epsilon=0.05, iterations=20, seed=seed)
    return {
        "seed": seed,
        "count": int(labels.numel()),
        "primary": {"accuracy": float((p_pred == labels.cpu()).float().mean()), "recall": _recall(p_pred, labels.cpu()), "minimum_recall": min(_recall(p_pred, labels.cpu()))},
        "secondary": {"accuracy": float((s_pred == labels.cpu()).float().mean()), "recall": _recall(s_pred, labels.cpu()), "minimum_recall": min(_recall(s_pred, labels.cpu()))},
        "agreement": float((p_pred == s_pred).float().mean()),
        "camera_code_fraction": float(camera_code_fraction(views, labels)),
        "saturation": float((views >= 1.0).float().mean()),
        "coverage": {"primary": p_coverage, "secondary": s_coverage},
        "diversity_ratio": {"primary": relative_diversity_ratios(p_features, p_ref, labels.cpu()), "secondary": relative_diversity_ratios(s_features, s_ref, labels.cpu())},
        "label_shuffled_primary_accuracy": float((shuffled_primary == labels.cpu()).float().mean()),
        "view_hash": tensor_sha256(views),
        "zero_code_view_hash": tensor_sha256(zero),
        "no_update_replay_exact": repeated,
        "memorization": _memorization(views, labels, train_images, train_labels, validation_images, validation_labels),
    }


def _panel_pass(row: dict[str, Any]) -> bool:
    return bool(
        row["primary"]["accuracy"] >= 0.95
        and row["secondary"]["accuracy"] >= 0.95
        and row["primary"]["minimum_recall"] >= 0.90
        and row["secondary"]["minimum_recall"] >= 0.90
        and row["agreement"] >= 0.95
        and row["camera_code_fraction"] >= 0.25
        and row["saturation"] <= 0.08
        and row["label_shuffled_primary_accuracy"] <= 0.20
        and all(0.70 <= value <= 1.30 for name in ("primary", "secondary") for value in row["diversity_ratio"][name])
        and all(value <= 1.50 for value in row["coverage"].values())
        and row["memorization"]["exact_duplicate_count"] == 0
        and row["memorization"]["pass_fraction"] >= MEMORIZATION_PASS_MINIMUM
    )


def _official_measurement(canary: DirectPriorCanary, primary: torch.nn.Module, secondary: torch.nn.Module, test_images: Tensor, test_labels: Tensor, device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(HISTORICAL_SEED)
    labels = test_labels.to(device)
    codes = torch.randn((labels.numel(), CODE_DIM), generator=generator).to(device)
    views = canary.sample(labels, codes)
    p_features, p_logits, s_features, s_logits = _eval_pair(primary, secondary, views)
    p_pred, s_pred = p_logits.argmax(1), s_logits.argmax(1)
    return {"count": int(labels.numel()), "primary_accuracy": float((p_pred == test_labels).float().mean()), "secondary_accuracy": float((s_pred == test_labels).float().mean()), "primary_minimum_recall": min(_recall(p_pred, test_labels)), "secondary_minimum_recall": min(_recall(s_pred, test_labels)), "agreement": float((p_pred == s_pred).float().mean()), "camera_code_fraction": float(camera_code_fraction(views, labels)), "saturation": float((views >= 1.0).float().mean()), "view_hash": tensor_sha256(views)}


def _build_measurement(args: argparse.Namespace, *, samples_per_class: int) -> dict[str, Any]:
    device = torch.device(args.device)
    all_images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_index, validation_index = stratified_mnist_indices(all_labels)
    train_images, train_labels = all_images.index_select(0, train_index), all_labels.index_select(0, train_index)
    validation_images, validation_labels = all_images.index_select(0, validation_index), all_labels.index_select(0, validation_index)
    test_images, test_labels = load_local_mnist(args.dataset_root, train=False)
    canary = _fit_rank8(train_images.to(device), train_labels.to(device))
    primary, primary_payload = load_frozen_evaluator(args.primary_evaluator_checkpoint, device=device)
    secondary = _secondary(args.secondary_evaluator_checkpoint, device=device)
    panels = [_panel_measurement(canary, primary, secondary, test_images, test_labels, train_images, train_labels, validation_images, validation_labels, seed=seed, samples_per_class=samples_per_class, device=device) for seed in LOCKED_PANEL_SEEDS]
    official = _official_measurement(canary, primary, secondary, test_images, test_labels, device)
    jacobian = _jacobian(canary, device=device)
    return {
        "canary": {"rank": RANK, "code_dim": CODE_DIM, "logit_epsilon": LOGIT_EPSILON, "fit_seed": HARNESS_SEED, "mean_hash": tensor_sha256(canary.means), "basis_hash": tensor_sha256(canary.bases), "scale_hash": tensor_sha256(canary.scales)},
        "panels": {str(row["seed"]): row for row in panels},
        "official_test": official,
        "jacobian": jacobian,
        "qualification": {"primary_validation_accuracy": primary_payload.get("validation_accuracy"), "primary_checkpoint_sha256": file_sha256(args.primary_evaluator_checkpoint), "secondary_checkpoint_sha256": file_sha256(args.secondary_evaluator_checkpoint)},
        "optimizer_created": False,
        "evaluator_calls": 1,
        "passed": all(_panel_pass(row) for row in panels) and jacobian["rank_1e-3_pass_fraction"] >= JACOBIAN_PASS_MINIMUM,
    }


def _classify(measurement: dict[str, Any]) -> tuple[str, str]:
    if measurement.get("status") == "invalid_harness" or not measurement.get("measurement_valid", True):
        return "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit"
    panels = list(measurement.get("panels", {}).values())
    semantic = all(row["primary"]["accuracy"] >= 0.95 and row["secondary"]["accuracy"] >= 0.95 and row["primary"]["minimum_recall"] >= 0.90 and row["secondary"]["minimum_recall"] >= 0.90 and row["agreement"] >= 0.95 for row in panels)
    diversity = all(_panel_pass(row) for row in panels) and measurement.get("jacobian", {}).get("rank_1e-3_pass_fraction", 0.0) >= JACOBIAN_PASS_MINIMUM
    if semantic and diversity:
        return "direct_prior_linear_capacity_admitted", "register_direct_generator_architecture_preflight"
    if not semantic:
        return "direct_prior_linear_semantic_capacity_failed", "close_direct_prior_linear_capacity_canary"
    if any(row["memorization"]["exact_duplicate_count"] != 0 or row["memorization"]["pass_fraction"] < MEMORIZATION_PASS_MINIMUM for row in panels):
        return "direct_prior_linear_diversity_capacity_failed", "close_direct_prior_linear_capacity_canary"
    return "direct_prior_linear_capacity_not_supported", "close_direct_prior_linear_capacity_canary"


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = [args.primary_evaluator_checkpoint, args.secondary_evaluator_checkpoint, args.primary_qualification_evidence, args.secondary_qualification_evidence, args.h_ccf_closeout_evidence]
    if not Path(args.dataset_root).is_dir() or any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError("H-DPGC lineage input is missing")
    primary_qualification = _load_json(args.primary_qualification_evidence)
    secondary_qualification = _load_json(args.secondary_qualification_evidence)
    if primary_qualification.get("status") != "passed" or secondary_qualification.get("status") != "passed":
        raise ValueError("H-DPGC evaluator qualification is not passed")
    closeout = _load_json(args.h_ccf_closeout_evidence)
    if closeout.get("schema") != "electronic_o0_prior_cycle_composition_closeout.v1" or closeout.get("status") != "valid" or closeout.get("classification") != "cycle_composition_attribution_underdetermined" or closeout.get("next_action") != "close_prior_native_attribution_family":
        raise ValueError("H-DPGC requires the passed H-CCF family-closeout decision")
    return {"git": _git(), "config_sha256": file_sha256(args.config), "dataset_root": str(Path(args.dataset_root).resolve()), "dataset_files": {"train": tensor_sha256(load_local_mnist(args.dataset_root, train=True)[0]), "test": tensor_sha256(load_local_mnist(args.dataset_root, train=False)[0])}, "inputs": {"primary_evaluator": file_sha256(args.primary_evaluator_checkpoint), "secondary_evaluator": file_sha256(args.secondary_evaluator_checkpoint), "primary_qualification": file_sha256(args.primary_qualification_evidence), "secondary_qualification": file_sha256(args.secondary_qualification_evidence), "h_ccf_closeout": file_sha256(args.h_ccf_closeout_evidence)}, "contract": config}


def _check_consumer(path: str, action: str) -> dict[str, Any]:
    payload = _load_json(path)
    allowed = tuple(f"{SCHEMA_PREFIX}_{stage}.v1" for stage in ("validation", "preflight", "audit", "report"))
    if payload.get("schema") not in allowed or payload.get("status") != "valid" or action not in payload.get("permitted_consumers", ()):
        raise ValueError(f"invalid H-DPGC {action} evidence")
    if payload.get("invocation_mode") != "manifest" or payload.get("legacy_diagnostic_only") is not False:
        raise ValueError("legacy H-DPGC evidence")
    return payload


def _stage(args: argparse.Namespace, stage: str, expected: dict[str, Any]) -> dict[str, Any]:
    return stage_action_decision(stage, producer_contract=f"{SCHEMA_PREFIX}-{stage}.v1", expected_inputs=expected, evidence=args.output)


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("H-DPGC lifecycle is manifest-only")
    action = args.research_action
    common = ("config", "dataset_root", "output", "primary_evaluator_checkpoint", "secondary_evaluator_checkpoint", "primary_qualification_evidence", "secondary_qualification_evidence", "h_ccf_closeout_evidence")
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
            raise ValueError("H-DPGC preflight requires passed validation")
        decision = _stage(args, "preflight", {"validation": file_sha256(args.validation_evidence), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        result = _build_measurement(args, samples_per_class=PREFLIGHT_SAMPLES_PER_CLASS)
        peak = int(torch.cuda.max_memory_allocated(args.device))
        root = Path(args._run_manifest["output_root"])
        storage = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        valid = bool(result["optimizer_created"] is False and result["evaluator_calls"] > 0 and all(row["no_update_replay_exact"] for row in result["panels"].values()) and peak < 6 * 1024**3 and storage <= 2 * 1024**3)
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_preflight.v1", "stage": "preflight", "status": "valid" if valid else "invalid_harness", "passed": valid, "measurement_valid": valid, "result": result, "peak_memory_bytes": peak, "storage_bytes": storage, "memory_gate": peak < 6 * 1024**3, "storage_gate": storage <= 2 * 1024**3, "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1", "permitted_consumers": ["audit"] if valid else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-preflight.v1", "input_artifact_hashes": {"validation": file_sha256(args.validation_evidence), "lineage": canonical_sha256(lineage)}}, args)
        return 0
    if action == "audit":
        preflight = _check_consumer(args.preflight_evidence, "audit")
        if not preflight.get("passed"):
            raise ValueError("H-DPGC audit requires passed preflight")
        decision = _stage(args, "audit", {"preflight": file_sha256(args.preflight_evidence), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        result = _build_measurement(args, samples_per_class=PANEL_SAMPLES_PER_CLASS)
        peak = int(torch.cuda.max_memory_allocated(args.device))
        root = Path(args._run_manifest["output_root"])
        storage = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        classification, next_action = _classify(result)
        valid = bool(result["optimizer_created"] is False and result["evaluator_calls"] > 0 and all(row["no_update_replay_exact"] for row in result["panels"].values()) and peak < 6 * 1024**3 and storage <= 2 * 1024**3)
        if not valid:
            classification, next_action = "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit"
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_audit.v1", "stage": "audit", "status": "valid" if valid else "invalid_harness", "passed": bool(valid and result["passed"]), "measurement_valid": valid, "classification": classification, "next_action": next_action, "result": result, "peak_memory_bytes": peak, "storage_bytes": storage, "memory_gate": peak < 6 * 1024**3, "storage_gate": storage <= 2 * 1024**3, "consumer_contract": f"{SCHEMA_PREFIX}-report.v1", "permitted_consumers": ["report"] if valid else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1", "input_artifact_hashes": {"preflight": file_sha256(args.preflight_evidence), "lineage": canonical_sha256(lineage)}}, args)
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
