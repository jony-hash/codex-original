"""Shared, content-bound runtime for evaluator-qualified transport stages.

This module deliberately owns the measurements that must mean the same thing
in admission, tangent, and smoke.  Stage modules only decide *when* to run a
measurement; they do not reimplement its physical or evaluator semantics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from ...evidence import safe_torch_load, EvidenceRecordV2, canonical_sha256, file_sha256, tensor_sha256
from ...student.checkpoints import ConditionalCheckpointManifestV2, state_content_hash, state_signature
from ...student.conditional import fixed_exposure_view, raw_from_generator_batched, relative_diversity_ratios
from ...student.evaluator import MNISTSplitManifest, SmoothMNISTEvaluator, load_frozen_evaluator
from .evaluator_qualification_v2 import V21_CONTRACT
from .expanded_manifold import _crop_minimum, _panels, _rank_summary, _validation_reference
from .support import EVALUATION_QUALIFIED_TRANSPORT_V11_ARCHITECTURE
from .transport_aligned_manifold import _delta_correlation, _device, _direction_ranks


ADMISSION_CONTRACT = canonical_sha256({"kind": "conditional_evaluation_qualified_transport_admission_v1", "panels": ["axis", "uniform_1234", "uniform_5678", "holdout_9012"], "representations": ["primary", "smooth", "low_frequency"], "anti_template": "merged_per_class"})
TANGENT_CONTRACT = canonical_sha256({"kind": "conditional_evaluation_qualified_transport_tangent_v1", "aggregate_composite": True, "optimizer": False})
SMOKE_CONTRACT = canonical_sha256({"kind": "evaluation_qualified_transport_smoke_v1", "seeds": [1234, 4321], "steps": 400, "probes": [0, 50, 100, 200, 400]})
EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE = EVALUATION_QUALIFIED_TRANSPORT_V11_ARCHITECTURE
GENERATOR_CONTRACT = canonical_sha256({"architecture": EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE, "formula": "sigmoid(By+sum(s*tanh(z)*normalize(WP)))"})
METRIC_CONTRACT = canonical_sha256({"delta": "cosine(raw(z)-raw(0),.80*A2(z)-.80*A2(0))", "representations": ["primary", "smooth", "low_frequency"]})


def load_record(path: str, kind: str, *, status: str = "passed", action: str | None = None) -> EvidenceRecordV2:
    value = EvidenceRecordV2.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    value.validate()
    if value.evidence_kind != kind or value.status != status:
        raise ValueError(f"expected {kind} evidence with status {status}")
    if action is not None and action not in value.permitted_next_actions:
        raise ValueError(f"{kind} evidence does not authorize {action}")
    return value


def load_evaluators(args, config, split_hash: str):
    qualification = load_record(args.evaluator_qualification_evidence, "conditional_evaluator_qualification_v21")
    bundle = json.loads(Path(args.evaluator_bundle).read_text(encoding="utf-8"))
    if bundle.get("contract") != V21_CONTRACT or bundle.get("split_manifest_hash") != split_hash:
        raise ValueError("evaluator v2.1 bundle contract/split mismatch")
    if bundle.get("primary_checkpoint") != file_sha256(args.primary_evaluator_checkpoint) or bundle.get("smooth_checkpoint") != file_sha256(args.smooth_evaluator_checkpoint):
        raise ValueError("evaluator v2.1 bundle checkpoint hash mismatch")
    device = _device(config)
    primary, payload = load_frozen_evaluator(args.primary_evaluator_checkpoint, device=device)
    if MNISTSplitManifest(**payload["split_manifest"]).manifest_hash != split_hash:
        raise ValueError("primary evaluator split mismatch")
    smooth_payload = safe_torch_load(args.smooth_evaluator_checkpoint)
    required = {"architecture_kind": SmoothMNISTEvaluator.architecture_kind, "contract": V21_CONTRACT, "split_manifest_hash": split_hash, "training_seed": 2234}
    if not isinstance(smooth_payload, dict) or any(smooth_payload.get(key) != value for key, value in required.items()):
        raise ValueError("smooth evaluator is not qualified evaluator v2.1")
    smooth = SmoothMNISTEvaluator().to(device)
    smooth.load_state_dict(smooth_payload["state_dict"], strict=True)
    return device, primary, smooth.freeze(), qualification, bundle


@dataclass
class PanelSnapshot:
    codes: torch.Tensor
    labels: torch.Tensor
    raw: torch.Tensor
    view: torch.Tensor
    primary_features: torch.Tensor
    primary_logits: torch.Tensor
    smooth_features: torch.Tensor
    smooth_logits: torch.Tensor
    low_frequency: torch.Tensor


def snapshots(bundle, primary, smooth, reference, exposure: float, device: torch.device) -> tuple[dict[str, PanelSnapshot], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    panels = _panels(device)
    result = {}
    with torch.no_grad():
        for name, (codes, labels) in panels.items():
            raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, labels, microbatch_size=1, activation_checkpoint=False)
            view = fixed_exposure_view(raw, exposure)
            pf, pl = primary(view)
            sf, sl = smooth(view)
            result[name] = PanelSnapshot(codes, labels, raw, view, pf, pl, sf, sl, F.avg_pool2d(view, 2))
    return result, panels


def _ratio(features: torch.Tensor, reference: torch.Tensor, labels: torch.Tensor) -> list[float]:
    return [float(item) for item in relative_diversity_ratios(features, reference, labels)]


def _pearson(first: torch.Tensor, second: torch.Tensor) -> float:
    first, second = first.flatten().float(), second.flatten().float()
    first, second = first - first.mean(), second - second.mean()
    return float((first @ second / (first.norm() * second.norm()).clamp_min(1e-12)).item())


def admission_values(bundle, primary, smooth, validation_images, validation_labels, exposure: float, device: torch.device) -> dict[str, object]:
    reference = _validation_reference(validation_images, validation_labels, device)
    with torch.no_grad():
        primary_reference, _ = primary(reference)
        smooth_reference, _ = smooth(reference)
    values, panels = snapshots(bundle, primary, smooth, reference, exposure, device)
    metrics, panel_detail, crops = {}, {}, {}
    for name, item in values.items():
        ratios = {"primary": _ratio(item.primary_features, primary_reference, item.labels), "smooth": _ratio(item.smooth_features, smooth_reference, item.labels), "low_frequency": _ratio(item.low_frequency.flatten(1), F.avg_pool2d(reference, 2).flatten(1), item.labels)}
        floor = .50 if name == "axis" else .60
        catastrophe = .35 if name == "axis" else .45
        composite = [sorted((ratios[key][label] for key in ratios), reverse=True)[1] for label in range(10)]
        recall = [float(((item.primary_logits.argmax(1)[item.labels == label] == label).float().mean()).item()) for label in range(10)]
        metrics[name] = {"accuracy": float((item.primary_logits.argmax(1) == item.labels).float().mean().item()), "recall": recall, "camera_fraction": float(__import__("ldm_project.v2.student.conditional", fromlist=["camera_code_fraction"]).camera_code_fraction(item.raw, item.labels).item()), "saturation": float((item.raw / exposure > 1).float().mean().item()), "delta_correlation": _delta_correlation(bundle, item.codes, item.labels, item.raw), "ratios": ratios, "composite": composite, "composite_floor": floor, "catastrophe_floor": catastrophe}
        panel_detail[name] = {"raw_hash_unique": len({tensor_sha256(row.cpu()) for row in item.raw}), "ratio_second_high_min": min(composite)}
        crops[name] = _crop_minimum(bundle, item.codes, item.labels)
    all_labels = torch.cat([item.labels for item in values.values()])
    all_primary = torch.cat([item.primary_logits for item in values.values()])
    merged = {}
    for label in range(10):
        rows = {"primary": torch.cat([item.primary_features[item.labels == label] for item in values.values()]), "smooth": torch.cat([item.smooth_features[item.labels == label] for item in values.values()]), "view": torch.cat([item.view[item.labels == label] for item in values.values()]), "codes": torch.cat([item.codes[item.labels == label] for item in values.values()])}
        centered_view = rows["view"].flatten(1) - rows["view"].flatten(1).mean(0, keepdim=True)
        quantized = (rows["view"] * 255).round() / 255
        distances = torch.pdist(quantized.flatten(1), p=2) / quantized[0].numel() ** .5
        code_distance, output_distance = torch.pdist(rows["codes"]), torch.pdist(quantized.flatten(1))
        merged[str(label)] = {"primary_rank": _rank_summary(rows["primary"] - rows["primary"].mean(0, keepdim=True)), "smooth_rank": _rank_summary(rows["smooth"] - rows["smooth"].mean(0, keepdim=True)), "effective_rank": int(torch.linalg.matrix_rank(centered_view.float()).item()), "quantized_distance_median": float(distances.median().item()), "quantized_nonzero_fraction": float((distances > 0).float().mean().item()), "code_output_correlation": _pearson(code_distance, output_distance)}
    zero_labels = torch.arange(10, device=device)
    with torch.no_grad():
        zero_raw = raw_from_generator_batched(bundle.system, bundle.generator, torch.zeros(10, 8, device=device), zero_labels, microbatch_size=1, activation_checkpoint=False)
        _, zero_logits = primary(fixed_exposure_view(zero_raw, exposure))
    command_rank, raw_rank = _direction_ranks(bundle, device)
    merged_recall = [float(((all_primary.argmax(1)[all_labels == label] == label).float().mean()).item()) for label in range(10)]
    return {"metrics": metrics, "panel_detail": panel_detail, "crops": crops, "merged": merged, "command_rank": command_rank, "raw_rank": raw_rank, "zero_accuracy": float((zero_logits.argmax(1) == zero_labels).float().mean().item()), "merged_recall": merged_recall}


def admission_gate_dict(values: dict[str, object], *, final: bool = True) -> dict[str, bool]:
    metrics = values["metrics"]
    semantics = all(row["accuracy"] >= (.95 if final else .90) and min(row["recall"]) >= (.85 if final else .80) for row in metrics.values()) and (not final or min(values["merged_recall"]) >= .90)
    distribution = all(min(row["composite"]) >= row["composite_floor"] and all(min(row["ratios"][kind]) >= row["catastrophe_floor"] for kind in row["ratios"]) for row in metrics.values())
    anti = all(row["primary_rank"]["rank"] >= 8 and row["primary_rank"]["sigma8_over_sigma1"] >= 1e-3 and row["smooth_rank"]["rank"] >= 8 and row["smooth_rank"]["sigma8_over_sigma1"] >= 1e-3 and row["effective_rank"] >= 8 and row["quantized_distance_median"] > 1 / 255 and row["quantized_nonzero_fraction"] >= .95 and row["code_output_correlation"] >= .20 for row in values["merged"].values())
    directions = all(row["rank"] == 8 and row["sigma8_over_sigma1"] >= 1e-3 for table in (values["command_rank"], values["raw_rank"]) for row in table.values())
    return {"semantics": semantics, "zero_code": values["zero_accuracy"] >= .95, "camera": metrics["axis"]["camera_fraction"] >= .25 and all(metrics[name]["camera_fraction"] >= .40 for name in metrics if name != "axis"), "delta": all(row["delta_correlation"] >= .80 for row in metrics.values()), "distribution": distribution, "directions": directions, "anti_template": anti, "saturation": all(row["saturation"] <= .08 for row in metrics.values()), "crop": all(value >= .90 for value in values["crops"].values())}


def manifest(config, split_hash: str, primary_hash: str, bundle, design_hash: str) -> ConditionalCheckpointManifestV2:
    state = bundle.state_dict()
    return ConditionalCheckpointManifestV2(2, EVALUATION_QUALIFIED_TRANSPORT_ARCHITECTURE, canonical_sha256(config.system.model_dump(mode="json")), canonical_sha256(config.model_dump(mode="json")), split_hash, primary_hash, config.conditional_generation.fixed_exposure_scale, state_signature(state), state_content_hash(state), GENERATOR_CONTRACT, METRIC_CONTRACT, SMOKE_CONTRACT, 1234, state_content_hash(state), config_contract_hash=canonical_sha256(config.model_dump(mode="json")), pca_basis_hash=tensor_sha256(bundle.generator.candidate_modes), lowpass_contract_hash=canonical_sha256({"cycles_per_mm": 20.0}), scale_contract_hash=tensor_sha256(bundle.generator.manifold_scales), route_contract_hash=canonical_sha256({"code": "tanh", "tau": .80}), semantic_source_checkpoint_hash=design_hash, safe_scale_evidence_hash=design_hash, first_400_schedule_hash=SMOKE_CONTRACT)
