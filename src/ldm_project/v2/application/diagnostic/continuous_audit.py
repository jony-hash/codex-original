"""Read-only feasibility audit for continuous, data-aligned class modes.

This module deliberately stops at an evidence report.  It does not register a
new formal generator, create an optimizer, or write a checkpoint.  The audit is
the bridge between the rejected hard four-anchor candidate and any future
continuous-mode architecture.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ...evidence import EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, file_sha256, canonical_sha256, tensor_sha256
from ...student.checkpoints import state_content_hash
from ...student.checkpoints import ConditionalCheckpointManifestV2, load_conditional_checkpoint
from ...student.conditional import camera_code_fraction, fixed_exposure_view, raw_from_generator_batched, relative_diversity_ratios
from ...student.evaluator import MNISTSplitManifest, evaluator_checkpoint_hash, load_frozen_evaluator, load_local_mnist
from ...student.generator import ConditionalClassConditionedModesGenerator
from ...student.review_artifacts import write_review_artifacts
from .support import (
    IDENTITY_ARCHITECTURE,
    _context,
    ConditionalBundle,
    _fixed_control_probe,
    _new_run,
    _random_codes,
    _split,
    _write_evidence,
)


CONTINUOUS_MODE_CAPACITY_CONTRACT = canonical_sha256({
    "kind": "conditional_continuous_mode_capacity_v1",
    "feature_pca_components": 8,
    "amplitude_pca_components": 8,
    "kmeans_baseline": {"k": 4, "iterations": 20, "seed": 1234},
    "bandwidth_cycles_per_mm": 20.0,
    "optimizer": False,
    "checkpoint": False,
})


def _stable_kmeans(features: torch.Tensor, *, k: int = 4, iterations: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic farthest-point k-means used only as the v5 baseline."""
    if features.ndim != 2 or features.shape[0] < k:
        raise ValueError("k-means requires [N,D] features with N >= k")
    data = features.detach().to(dtype=torch.float64, device="cpu")
    centers = [data[0]]
    distances = torch.cdist(data[:1], data).square().squeeze(0)
    for _ in range(1, k):
        index = int(torch.argmax(distances).item())
        centers.append(data[index])
        distances = torch.minimum(distances, torch.cdist(data[index : index + 1], data).square().squeeze(0))
    centers_tensor = torch.stack(centers)
    assignments = torch.zeros(data.shape[0], dtype=torch.long)
    for _ in range(iterations):
        assignments = torch.cdist(data, centers_tensor).square().argmin(dim=1)
        updated = []
        for cluster in range(k):
            members = data[assignments == cluster]
            updated.append(members.mean(dim=0) if members.numel() else centers_tensor[cluster])
        updated_tensor = torch.stack(updated)
        if torch.equal(updated_tensor, centers_tensor):
            break
        centers_tensor = updated_tensor
    return assignments, centers_tensor


def _pca(data: torch.Tensor, components: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean, descending eigenvalues and deterministic principal vectors."""
    values = data.detach().to(dtype=torch.float64, device="cpu")
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / max(1, values.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    eigenvectors = eigenvectors.index_select(1, order)
    eigenvectors = eigenvectors[:, :components]
    # Eigenvector signs are arbitrary; make them content-deterministic.
    for column in range(eigenvectors.shape[1]):
        pivot = int(torch.argmax(eigenvectors[:, column].abs()).item())
        if eigenvectors[pivot, column] < 0:
            eigenvectors[:, column].neg_()
    return mean, eigenvalues[:components], eigenvectors


def _lowpass_modes(modes: torch.Tensor, *, shape: tuple[int, int], pixel_pitch_m: float, max_cycles_per_mm: float) -> torch.Tensor:
    enlarged = F.interpolate(modes.unsqueeze(1).to(torch.float32), size=shape, mode="bilinear", align_corners=False).squeeze(1)
    height, width = shape
    fy = torch.fft.fftfreq(height, d=pixel_pitch_m) / 1000.0
    fx = torch.fft.fftfreq(width, d=pixel_pitch_m) / 1000.0
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    mask = (radius <= max_cycles_per_mm).to(enlarged.dtype)
    spectrum = torch.fft.fft2(enlarged)
    filtered = torch.fft.ifft2(spectrum * mask).real
    filtered = filtered - filtered.mean(dim=(-2, -1), keepdim=True)
    # Deterministic Gram--Schmidt after the physical bandwidth projection.
    rows: list[torch.Tensor] = []
    for row in filtered:
        vector = row.flatten()
        for basis in rows:
            vector = vector - torch.dot(vector, basis) * basis
        norm = vector.norm().clamp_min(1.0e-12)
        rows.append(vector / norm)
    return torch.stack(rows).reshape(filtered.shape)


def _panel_codes(device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    axis, labels = _fixed_control_probe(device=device)
    return torch.cat((axis, _random_codes(170, device=device, seed=seed))), torch.cat((labels, labels))


def _panel_summary(raw: torch.Tensor, labels: torch.Tensor, evaluator: nn.Module, exposure: float, real_views: torch.Tensor) -> dict[str, object]:
    view = fixed_exposure_view(raw, exposure)
    features, logits = evaluator(view)
    predictions = logits.argmax(dim=1)
    ratios = relative_diversity_ratios(features, evaluator(real_views)[0].detach(), labels)
    confusion = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    recall = []
    for label in range(10):
        selected = labels == label
        recall.append(float((predictions[selected] == label).to(torch.float32).mean().item()))
    return {
        "accuracy": float((predictions == labels).to(torch.float32).mean().item()),
        "recall": recall,
        "confusion": confusion.cpu().tolist(),
        "camera_code_fraction": float(camera_code_fraction(raw, labels).detach().item()),
        "feature_distance_ratios": ratios,
        "saturation": float(torch.relu(raw / exposure - 1.0).square().mean().item()),
        "raw_rms": float(raw.square().mean().sqrt().item()),
        "raw_finite": bool(torch.isfinite(raw).all()),
    }


def _real_probe_views(images: torch.Tensor, labels: torch.Tensor, *, count: int = 17) -> torch.Tensor:
    rows = []
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()[:count]
        if indices.numel() < count:
            raise ValueError("the fixed train split does not contain enough samples for the probe")
        rows.append(images.index_select(0, indices))
    return torch.cat(rows, dim=0)


def conditional_continuous_mode_capacity(args, config) -> int:
    """Run the fixed data-capacity audit and, if it passes, one read-only optical probe."""
    if config.architecture != IDENTITY_ARCHITECTURE or config.conditional_generation is None:
        raise ValueError("continuous-mode audit requires the identity v2 configuration")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    checkpoint_hash = file_sha256(args.semantic_checkpoint)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_hash)
    evaluator, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device("cpu"))
    payload_split = MNISTSplitManifest(**payload["split_manifest"])
    if payload_split.manifest_hash != split.manifest_hash:
        raise ValueError("evaluator checkpoint uses another MNIST split")
    payload = torch.load(args.semantic_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
        raise ValueError("semantic checkpoint is invalid")
    source_manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    if source_manifest.architecture_kind != IDENTITY_ARCHITECTURE or source_manifest.system_signature != _context(config).system_signature or source_manifest.evaluator_checkpoint_hash != evaluator_hash or source_manifest.dataset_split_manifest_hash != split.manifest_hash:
        raise ValueError("continuous-mode audit requires the semantic identity checkpoint, not a smoke final")
    bundle = ConditionalBundle(config).to(torch.device("cpu"))
    load_conditional_checkpoint(args.semantic_checkpoint, bundle, source_manifest)
    before = state_content_hash(bundle.state_dict())
    with torch.no_grad():
        features, _ = evaluator(train_images)
    features = F.normalize(features.to(torch.float64), dim=1)
    amplitude = train_images.sqrt().clamp(1.0e-4, 1.0)
    feature_metrics: dict[str, object] = {}
    amplitude_modes = []
    for label in range(10):
        selected = train_labels == label
        class_features = features[selected]
        class_amplitude = amplitude[selected]
        _, kmeans_centers = _stable_kmeans(class_features)
        kmeans_assignments, _ = _stable_kmeans(class_features)
        class_mean = class_features.mean(dim=0)
        total = (class_features - class_mean).square().sum().clamp_min(1.0e-12)
        cluster_residual = sum((class_features[kmeans_assignments == cluster] - kmeans_centers[cluster]).square().sum() for cluster in range(4))
        _, eigenvalues, _ = _pca(class_features)
        captured = eigenvalues.sum() * max(1, class_features.shape[0] - 1)
        pca_reduction = float((captured / total).clamp(0.0, 1.0).item())
        kmeans_reduction = float((1.0 - cluster_residual / total).item())
        amp_flat = class_amplitude.flatten(1).to(torch.float64)
        amp_mean, amp_eigenvalues, amp_vectors = _pca(amp_flat)
        amp_total = (amp_flat - amp_mean).square().sum().clamp_min(1.0e-12)
        amp_reduction = float((amp_eigenvalues.sum() * max(1, amp_flat.shape[0] - 1) / amp_total).item())
        amplitude_modes.append(amp_vectors.T.reshape(8, 28, 28))
        feature_metrics[str(label)] = {
            "sample_count": int(selected.sum().item()),
            "kmeans_counts": [int((kmeans_assignments == cluster).sum().item()) for cluster in range(4)],
            "kmeans_reduction": kmeans_reduction,
            "pca8_reduction": pca_reduction,
            "pca8_advantage": pca_reduction - kmeans_reduction,
            "amplitude_pca8_reduction": amp_reduction,
            "pca_eigenvalues": [float(value) for value in eigenvalues],
        }
    shape = config.system.slm_shape
    modes = torch.stack([_lowpass_modes(item, shape=shape, pixel_pitch_m=config.system.slm_pixel_pitch_m, max_cycles_per_mm=20.0) for item in amplitude_modes])
    basis_hash = tensor_sha256(modes)
    mean_reduction = sum(float(item["pca8_reduction"]) for item in feature_metrics.values()) / 10.0
    mean_advantage = sum(float(item["pca8_advantage"]) for item in feature_metrics.values()) / 10.0
    min_advantage = min(float(item["pca8_advantage"]) for item in feature_metrics.values())
    min_amplitude_reduction = min(float(item["amplitude_pca8_reduction"]) for item in feature_metrics.values())
    data_checks = (
        GateResult("feature_pca8_dominates_kmeans", min_advantage >= 0.0, {"minimum_advantage": min_advantage, "mean_advantage": mean_advantage}),
        GateResult("feature_pca8_mean_advantage", mean_advantage >= 0.10, {"mean_advantage": mean_advantage}),
        GateResult("amplitude_pca8_reduction", min_amplitude_reduction >= 0.30, {"minimum_reduction": min_amplitude_reduction}),
        GateResult("basis_finite_rank_eight", bool(torch.isfinite(modes).all() and all(torch.linalg.matrix_rank(modes[label].flatten(1)).item() == 8 for label in range(10))), {"basis_hash": basis_hash}),
    )
    data_passed = all(item.passed for item in data_checks)
    report_payload: dict[str, object] = {"contract": CONTINUOUS_MODE_CAPACITY_CONTRACT, "source_checkpoint_config_hash": source_manifest.config_hash, "feature_metrics": feature_metrics, "mean_pca8_reduction": mean_reduction, "mean_pca_advantage_over_kmeans": mean_advantage, "minimum_pca_advantage": min_advantage, "minimum_amplitude_reduction": min_amplitude_reduction, "basis_hash": basis_hash, "data_checks": [{"name": item.name, "passed": item.passed, "metrics": dict(item.metrics)} for item in data_checks], "optimizer_created": False}
    probe_checks: list[GateResult] = []
    panel_metrics: dict[str, object] = {}
    panel_artifacts: dict[str, object] = {}
    probe_harness_invalid = False
    if data_passed and str(config.run.device) == "cuda" and not torch.cuda.is_available():
        probe_harness_invalid = True
        probe_checks.append(GateResult("cuda_available_for_optical_probe", False, {"cuda_available": False}))
    elif data_passed:
        device = torch.device(config.run.device)
        evaluator = evaluator.to(device).eval()
        bundle = bundle.to(device)
        modes = modes.to(device=device, dtype=torch.float32)
        generator = ConditionalClassConditionedModesGenerator(shape=shape, code_dim=8, hidden_dim=64, command_gain=0.5, class_carrier_logits=bundle.generator.class_carrier_logits.detach(), class_modes=modes)
        real_views = _real_probe_views(train_images, train_labels).to(device)
        panels = {"axis": _panel_codes(device, 1234)[0][:170], "uniform_1234": _random_codes(170, device=device, seed=1234), "uniform_5678": _random_codes(170, device=device, seed=5678)}
        axis_labels = _fixed_control_probe(device=device)[1]
        for name, codes in panels.items():
            panel_labels = axis_labels if name == "axis" else axis_labels
            with torch.no_grad():
                raw = raw_from_generator_batched(bundle.system, generator, codes, panel_labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
                panel_metrics[name] = _panel_summary(raw, panel_labels, evaluator, config.conditional_generation.fixed_exposure_scale, real_views)
                panel_artifacts[name] = {"raw": raw.detach().cpu(), "labels": panel_labels.detach().cpu()}
        trace = bundle.system.forward_from_generator(generator, panels["axis"][:1], axis_labels[:1], return_optical_trace=True).optical_trace or ()
        operations = tuple(field.provenance.operation for field in trace)
        probe_checks.extend((
            GateResult("panel_code_fraction", all(float(item["camera_code_fraction"]) >= 0.10 for item in panel_metrics.values()), {}),
            GateResult("panel_saturation", all(float(item["saturation"]) <= 0.08 for item in panel_metrics.values()), {}),
            GateResult("complete_optical_trace", any("identity_ols" in item for item in operations), {"operations": operations}),
        ))
    else:
        report_payload["probe_skipped"] = True
    after = state_content_hash(bundle.state_dict())
    probe_checks.append(GateResult("parameters_unchanged", before == after, {}))
    checks = data_checks + tuple(probe_checks)
    passed = data_passed and all(item.passed for item in checks)
    classification = "continuous_class_modes_supported" if passed else ("continuous_manifold_capacity_unconfirmed" if not data_passed else ("harness_invalid" if probe_harness_invalid else "optical_continuous_mode_unreachable"))
    report_payload.update({"panel_metrics": panel_metrics, "classification": classification, "state_hash_before": before, "state_hash_after": after, "provenance_complete": any(item.name == "complete_optical_trace" and item.passed for item in probe_checks)})
    run = _new_run(config)
    report = run.root / "conditional_continuous_mode_capacity_report.json"
    basis = run.root / "continuous_mode_basis_manifest.json"
    atomic_write_json(report, report_payload)
    basis_payload = {"contract": CONTINUOUS_MODE_CAPACITY_CONTRACT, "basis_hash": basis_hash, "shape": list(modes.shape), "architecture_status": "diagnostic_only", "checkpoint_created": False}
    atomic_write_json(basis, basis_payload)
    for name, artifact in panel_artifacts.items():
        raw_path = run.root / f"continuous_{name}_raw_cmos.pt"
        atomic_torch_save(raw_path, artifact["raw"])
        preview = write_review_artifacts(root=run.root, stem=f"continuous_{name}", kind="conditional_continuous_mode_capacity", raw_cmos=artifact["raw"], labels=artifact["labels"], exposure_scale=config.conditional_generation.fixed_exposure_scale)
        panel_metrics[name]["raw_hash"] = file_sha256(raw_path)
        panel_metrics[name]["preview_hash"] = preview.preview_sha256
    atomic_write_json(report, report_payload)
    record = EvidenceRecordV2(2, "conditional_continuous_mode_capacity", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"report": file_sha256(report), "basis_manifest": file_sha256(basis), "semantic_checkpoint": checkpoint_hash, "source_checkpoint_config_hash": source_manifest.config_hash, "basis_hash": basis_hash}, checks, ("conditional-continuous-mode-admission-design",) if passed else (), None if passed else classification)
    _write_evidence(run, "conditional_continuous_mode_capacity_evidence.json", record)
    print(run.root)
    return 0
