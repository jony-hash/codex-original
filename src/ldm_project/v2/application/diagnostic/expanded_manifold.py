"""Read-only v7 expanded nonlinear electronic-manifold admission.

The candidate deliberately preserves the v6.4 near-conjugate optical system.
Only the pre-SLM electronic manifold changes, and neither audit creates an
optimizer, a training checkpoint, or a warm start from the v6.4 smoke final.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ...evidence import safe_torch_load, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, tensor_sha256
from ...student.checkpoints import state_content_hash
from ...student.conditional import fixed_exposure_view, raw_from_generator_batched
from ...student.evaluator import MNISTSplitManifest, evaluator_checkpoint_hash, load_frozen_evaluator, load_local_mnist
from ...student.generator import ConditionalNonlinearExpandedManifoldGenerator
from .continuous_audit import _lowpass_modes, _pca
from .formal_spacing import _direction_response
from .pca_transport import _metrics
from .support import ConditionalBundle, EXPANDED_MANIFOLD_ARCHITECTURE, _context, _fixed_control_probe, _new_run, _random_codes, _split, _write_evidence


EXPANDED_MANIFOLD_CAPACITY_CONTRACT = canonical_sha256({
    "kind": "conditional_nonlinear_expanded_manifold_capacity_v1",
    "components": 32,
    "source_first_modes": 8,
    "amplitude": "sqrt(mnist_intensity)",
    "lowpass_cycles_per_mm": 20.0,
    "optimizer": False,
    "checkpoint": False,
})
EXPANDED_MANIFOLD_ADMISSION_CONTRACT = canonical_sha256({
    "kind": "conditional_nonlinear_expanded_manifold_admission_v1",
    "panels": ["axis", "uniform_1234", "uniform_5678", "holdout_9012"],
    "basis_split": "train_only",
    "feature_reference_split": "validation_only",
    "optimizer": False,
    "checkpoint": False,
})


def _require(config) -> None:
    if config.architecture != EXPANDED_MANIFOLD_ARCHITECTURE or config.conditional_generation is None:
        raise ValueError("expanded manifold audit requires conditional_nonlinear_expanded_manifold_v7")


def _source(path: str) -> dict[str, torch.Tensor | str]:
    payload = safe_torch_load(path)
    required = {"carrier", "basis", "scales", "basis_hash", "source_checkpoint_hash"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("expanded manifold requires a v6.2 electronic initialization artifact")
    if tuple(payload["carrier"].shape) != (10, 200, 200) or tuple(payload["basis"].shape) != (10, 8, 200, 200) or tuple(payload["scales"].shape) != (10, 8):
        raise ValueError("v6.2 electronic initialization has incompatible shapes")
    return payload


def _orthogonal_extra(source_modes: torch.Tensor, candidate_modes: torch.Tensor) -> torch.Tensor:
    """Gram--Schmidt PCA9--32 against the immutable source PCA1--8 frame."""
    rows: list[torch.Tensor] = []
    base = ConditionalNonlinearExpandedManifoldGenerator.normalized_modes(source_modes.unsqueeze(0)).squeeze(0)
    for item in base:
        rows.append(item.flatten() / item.flatten().norm().clamp_min(1.0e-12))
    result: list[torch.Tensor] = []
    for item in candidate_modes:
        vector = item.flatten()
        for row in rows:
            vector = vector - torch.dot(vector, row) * row
        norm = vector.norm()
        if norm <= 1.0e-10:
            raise ValueError("expanded PCA mode lost rank after source-frame orthogonalization")
        row = vector / norm
        rows.append(row)
        result.append(row.reshape_as(item))
    return torch.stack(result)


def _build_artifact(train_images: torch.Tensor, train_labels: torch.Tensor, source: dict[str, torch.Tensor | str], config) -> tuple[dict[str, object], dict[str, float]]:
    source_basis, source_scales = source["basis"], source["scales"]
    modes, values, totals, reports = [], [], [], {}
    amplitude = train_images.sqrt().clamp(1.0e-4, 1.0)
    for label in range(10):
        _, eigenvalues, vectors = _pca(amplitude[train_labels == label].flatten(1), components=32)
        # _pca returns selected values only; total variance is recovered from a
        # small deterministic covariance pass so explained fractions are exact.
        centered = amplitude[train_labels == label].flatten(1).double()
        centered -= centered.mean(0)
        total = float(centered.square().sum().item() / max(1, centered.shape[0] - 1))
        lowpass = _lowpass_modes(vectors.T.reshape(32, 28, 28), shape=config.system.slm_shape, pixel_pitch_m=config.system.slm_pixel_pitch_m, max_cycles_per_mm=20.0)
        extra = _orthogonal_extra(source_basis[label], lowpass[8:])
        modes.append(torch.cat((source_basis[label], extra), dim=0))
        values.append(eigenvalues)
        totals.append(total)
        e8, e32 = float(eigenvalues[:8].sum() / total), float(eigenvalues.sum() / total)
        reports[str(label)] = {
            "rank": int(torch.linalg.matrix_rank(lowpass.flatten(1)).item()),
            "explained_8": e8,
            "explained_32": e32,
            "incremental": e32 - e8,
            "remaining_recovery": (e32 - e8) / max(1.0e-12, 1.0 - e8),
        }
    basis, eigenvalues = torch.stack(modes).float(), torch.stack(values).float()
    extra_scale = torch.stack([
        source_scales[label].median() * torch.sqrt(eigenvalues[label, 8:] / eigenvalues[label, 7].clamp_min(1.0e-12))
        for label in range(10)
    ]).clamp(0.05, 0.50)
    scales = torch.cat((source_scales.float(), extra_scale.float()), dim=1)
    artifact = {
        "schema_version": 1,
        "contract": EXPANDED_MANIFOLD_CAPACITY_CONTRACT,
        "source_basis_hash": source["basis_hash"],
        "source_electronic_hash": source["source_checkpoint_hash"],
        "carrier": source["carrier"].float(),
        "basis": basis,
        "eigenvalues": eigenvalues,
        "scales": scales,
        "basis_hash": tensor_sha256(basis),
        "scale_hash": tensor_sha256(scales),
    }
    return artifact, reports


def conditional_nonlinear_expanded_manifold_capacity(args, config) -> int:
    _require(config)
    source = _source(args.electronic_initialization)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    artifact, reports = _build_artifact(train_images, train_labels, source, config)
    basis_path_hash = tensor_sha256(artifact["basis"])
    gates = (
        GateResult("rank_32", all(row["rank"] == 32 for row in reports.values()), reports),
        GateResult("explained_variance", all(row["explained_32"] >= 0.77 for row in reports.values()), reports),
        GateResult("incremental_variance", all(row["incremental"] >= 0.15 for row in reports.values()), reports),
        GateResult("remaining_variance_recovery", all(row["remaining_recovery"] >= 0.55 for row in reports.values()), reports),
        GateResult("source_integrity", str(source["basis_hash"]) == tensor_sha256(source["basis"]), {"source_basis_hash": source["basis_hash"]}),
    )
    passed = all(gate.passed for gate in gates)
    run = _new_run(config)
    artifact_path = run.root / "conditional_nonlinear_expanded_manifold_basis.pt"
    atomic_torch_save(artifact_path, artifact)
    report = run.root / "conditional_nonlinear_expanded_manifold_capacity_report.json"
    atomic_write_json(report, {"contract": EXPANDED_MANIFOLD_CAPACITY_CONTRACT, "per_label": reports, "basis_hash": basis_path_hash, "artifact": str(artifact_path), "optimizer_created": False, "checkpoint_created": False})
    context = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.electronic_initialization))
    record = EvidenceRecordV2(2, "conditional_nonlinear_expanded_manifold_capacity", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, context, {"electronic_initialization": file_sha256(args.electronic_initialization), "basis_artifact": file_sha256(artifact_path), "report": file_sha256(report)}, gates, ("conditional-nonlinear-expanded-manifold-admission",) if passed else (), None if passed else "expanded_manifold_capacity_insufficient")
    _write_evidence(run, "conditional_nonlinear_expanded_manifold_capacity_evidence.json", record)
    print(run.root)
    return 0


def _panels(device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    axis, labels = _fixed_control_probe(device=device)
    return {"axis": (axis, labels), "uniform_1234": (_random_codes(170, device=device, seed=1234), labels), "uniform_5678": (_random_codes(170, device=device, seed=5678), labels), "holdout_9012": (_random_codes(170, device=device, seed=9012), labels)}


def _validation_reference(images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    rows = []
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()[:17]
        if selected.numel() != 17:
            raise ValueError("validation split lacks the fixed 17-image panel")
        rows.append(images.index_select(0, selected))
    return torch.cat(rows).to(device)


def _rank_summary(rows: torch.Tensor) -> dict[str, float | int]:
    singular = torch.linalg.svdvals(rows.float())
    # The panel has 17 samples, hence centering necessarily introduces a zero
    # final singular value.  The anti-template gate is explicitly an 8-D
    # observability gate, so it must inspect s[7], not the null-space tail.
    if singular.numel() < 8:
        raise ValueError("rank-8 summary requires at least eight observations")
    ratio = float((singular[7] / singular[0].clamp_min(1.0e-12)).item())
    return {"rank": int((singular / singular[0].clamp_min(1.0e-12) >= 1.0e-3).sum().item()), "sigma8_over_sigma1": ratio}


def _command_jacobian(generator, device: torch.device) -> dict[str, dict[str, float | int]]:
    rows = {}
    with torch.no_grad():
        for label in range(10):
            directions = []
            for axis in range(8):
                plus, minus = torch.zeros(1, 8, device=device), torch.zeros(1, 8, device=device)
                plus[0, axis], minus[0, axis] = 1.0, -1.0
                command = generator(plus, torch.tensor([label], device=device)) - generator(minus, torch.tensor([label], device=device))
                directions.append(command.flatten())
            rows[str(label)] = _rank_summary(torch.stack(directions))
    return rows


def _curvature(generator, device: torch.device) -> dict[str, float]:
    output = {}
    with torch.no_grad():
        for label in range(10):
            plus = torch.full((1, 8), 0.5, device=device)
            minus = -plus
            zero = torch.zeros_like(plus)
            labels = torch.tensor([label], device=device)
            c_plus, c_minus, c_zero = (generator.coefficients(item, labels)[0, 8:] for item in (plus, minus, zero))
            output[str(label)] = float((c_plus + c_minus - 2.0 * c_zero).norm() / c_plus.norm().clamp_min(1.0e-12))
    return output


def _crop_minimum(bundle, codes: torch.Tensor, labels: torch.Tensor) -> float:
    values = []
    with torch.no_grad():
        for index in range(codes.shape[0]):
            command = bundle.generator(codes[index:index + 1], labels[index:index + 1])
            result = bundle.system.forward(bundle.system.command(command))
            for key, value in result.crop_statistics.items():
                if key.endswith("crop_retention"):
                    values.append(float(value.min().item()))
    return min(values)


def conditional_nonlinear_expanded_manifold_admission(args, config) -> int:
    _require(config)
    capacity = EvidenceRecordV2.from_dict(json.loads(Path(args.capacity_evidence).read_text(encoding="utf-8")))
    capacity.validate()
    if capacity.evidence_kind != "conditional_nonlinear_expanded_manifold_capacity" or capacity.status != "passed" or "conditional-nonlinear-expanded-manifold-admission" not in capacity.permitted_next_actions:
        raise ValueError("expanded manifold capacity evidence does not authorize admission")
    artifact = safe_torch_load(args.basis_artifact)
    if not isinstance(artifact, dict) or artifact.get("contract") != EXPANDED_MANIFOLD_CAPACITY_CONTRACT:
        raise ValueError("invalid expanded manifold basis artifact")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    _, _, validation_images, validation_labels, split = _split(images, labels, args.dataset_root)
    expected = _context(config, evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint), split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.electronic_initialization))
    capacity.context.assert_matches(expected, fields=("system_signature", "evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash", "exposure_scale"))
    if capacity.input_hashes.get("basis_artifact") != file_sha256(args.basis_artifact):
        raise ValueError("basis artifact does not match capacity evidence")
    device = torch.device(config.run.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("expanded manifold admission requires configured CUDA")
    evaluator, evaluator_payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if MNISTSplitManifest(**evaluator_payload["split_manifest"]).manifest_hash != split.manifest_hash:
        raise ValueError("evaluator split mismatch")
    bundle = ConditionalBundle(config, carrier_logits=artifact["carrier"], class_modes=artifact["basis"], manifold_scales=artifact["scales"]).to(device).eval()
    before = state_content_hash(bundle.state_dict())
    panels, reference = _panels(device), _validation_reference(validation_images, validation_labels, device)
    torch.cuda.reset_peak_memory_stats(device)
    metrics, raw_hashes, raw_rank, feature_rank, crops = {}, {}, {}, {}, {}
    axis_raw = None
    with torch.no_grad():
        for name, (codes, panel_labels) in panels.items():
            values = _metrics(bundle, evaluator, codes, panel_labels, reference, config.conditional_generation.fixed_exposure_scale)
            raw = values["raw"]
            metrics[name] = {key: value for key, value in values.items() if key not in {"raw", "command", "features", "logits"}}
            raw_hashes[name] = [tensor_sha256(row) for row in raw.detach().cpu()]
            crops[name] = _crop_minimum(bundle, codes, panel_labels)
            features = values["features"].detach()
            rows = {}
            for label in range(10):
                selected = features[panel_labels == label] - features[panel_labels == label].mean(0, keepdim=True)
                rows[str(label)] = _rank_summary(selected)
            feature_rank[name] = rows
            if name == "axis":
                axis_raw = raw.detach()
                response = {}
                for label in range(10):
                    selected = raw[panel_labels == label, 0]
                    response[str(label)] = _rank_summary((selected[1:9] - selected[9:17]).flatten(1))
                raw_rank = response
    command_rank, curvature = _command_jacobian(bundle.generator, device), _curvature(bundle.generator, device)
    zero_labels = torch.arange(10, device=device)
    with torch.no_grad():
        zero_raw = raw_from_generator_batched(bundle.system, bundle.generator, torch.zeros(10, 8, device=device), zero_labels, microbatch_size=1, activation_checkpoint=False)
        _, zero_logits = evaluator(fixed_exposure_view(zero_raw, config.conditional_generation.fixed_exposure_scale))
    zero_accuracy = float((zero_logits.argmax(1) == zero_labels).float().mean().item())
    after = state_content_hash(bundle.state_dict())
    peak, total = torch.cuda.max_memory_reserved(device), torch.cuda.get_device_properties(device).total_memory
    panel_semantics = all(value["accuracy"] >= .95 and min(value["recall"]) >= .90 for value in metrics.values())
    fraction = all(value["camera_fraction"] >= .25 for value in metrics.values()) and all(metrics[name]["camera_fraction"] >= .40 for name in ("uniform_1234", "uniform_5678", "holdout_9012"))
    ratios = min(metrics["axis"]["ratios"]) >= .50 and all(min(metrics[name]["ratios"]) >= .60 for name in ("uniform_1234", "uniform_5678", "holdout_9012"))
    ranks = all(value["rank"] == 8 and value["sigma8_over_sigma1"] >= 1e-3 for value in command_rank.values())
    # Direction response is recomputed from the in-memory axis raw tensor.
    assert axis_raw is not None
    raw_ranks = all(value["rank"] == 8 and value["sigma8_over_sigma1"] >= 1e-3 for value in raw_rank.values())
    direction = _direction_response(axis_raw)
    raw_ranks = raw_ranks and min(direction) >= .01
    noncollapsed = all(value["rank"] >= 8 and value["sigma8_over_sigma1"] >= 1e-3 for rows in feature_rank.values() for value in rows.values()) and all(len(set(values)) == 170 for values in raw_hashes.values())
    saturation = all(value["saturation"] <= .08 for value in metrics.values())
    gates = (
        GateResult("semantics", panel_semantics, metrics), GateResult("zero_semantics", zero_accuracy >= .95, {"accuracy": zero_accuracy}),
        GateResult("camera_fraction", fraction, {name: value["camera_fraction"] for name, value in metrics.items()}), GateResult("feature_ratio", ratios, {name: min(value["ratios"]) for name, value in metrics.items()}),
        GateResult("command_jacobian", ranks, command_rank), GateResult("raw_direction_response", raw_ranks, {"rank": raw_rank, "response": direction}),
        GateResult("nonlinear_curvature", all(value >= 1e-3 for value in curvature.values()), curvature), GateResult("anti_template_collapse", noncollapsed, {"feature_rank": feature_rank, "unique_raw": {name: len(set(values)) for name, values in raw_hashes.items()}}),
        GateResult("saturation", saturation, {name: value["saturation"] for name, value in metrics.items()}), GateResult("relay_crop_retention", all(value >= .90 for value in crops.values()), crops),
        GateResult("execution_integrity", before == after and peak <= .92 * total, {"state_unchanged": before == after, "reserved": peak, "total": total}), GateResult("no_optimizer", True, {"optimizer_created": False}), GateResult("no_checkpoint", True, {"checkpoint_created": False}),
    )
    passed = all(gate.passed for gate in gates)
    run = _new_run(config)
    report = run.root / "conditional_nonlinear_expanded_manifold_admission_report.json"
    atomic_write_json(report, {"contract": EXPANDED_MANIFOLD_ADMISSION_CONTRACT, "classification": "nonlinear_expanded_manifold_admitted_requires_contract" if passed else "nonlinear_expanded_manifold_not_admitted", "metrics": metrics, "command_jacobian": command_rank, "raw_direction": {"rank": raw_rank, "response": direction}, "curvature": curvature, "crops": crops, "state_hash_before": before, "state_hash_after": after, "optimizer_created": False, "checkpoint_created": False})
    record = EvidenceRecordV2(2, "conditional_nonlinear_expanded_manifold_admission", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"capacity_evidence": file_sha256(args.capacity_evidence), "basis_artifact": file_sha256(args.basis_artifact), "electronic_initialization": file_sha256(args.electronic_initialization), "report": file_sha256(report)}, gates, ("conditional-nonlinear-expanded-manifold-training-contract-design",) if passed else (), None if passed else "nonlinear_expanded_manifold_not_admitted")
    _write_evidence(run, "conditional_nonlinear_expanded_manifold_admission_evidence.json", record)
    print(run.root)
    return 0
