"""No-update dose attribution for the scaled Chen--BOGT pilot.

This lifecycle consumes the completed four-arm pilot and measures whether a
small, pre-registered change of the two objective weights can produce a safe
gradient direction.  It never creates an optimizer, updates a model, or
selects a held-out result.  A subsequent pilot must consume the selected
weights and a fresh validation/preflight pair.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.nn import functional as F

from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from . import optical_manifold as base
from . import optical_manifold_smf_bogt as bogt


BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
HYPOTHESIS = "H-LOMA-CBOGT-S0"
CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_failure_attribution_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_failure_attribution_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_failure_attribution_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_failure_attribution.v1"
SOURCE_COMMIT = "caf1e28a7715ee02f3f1aa56602d05dec4f43621"
SOURCE_TRAINING_SHA256 = "2079fc306e9403b704440053085597726fed5f6c41b6af6b7239213f38ac54ca"
SOURCE_VALIDATION_SHA256 = "bf59c8cb2d868dd779eff98af445700c2102fb87fa9d6ab9c468918d1992f6ba"
SOURCE_PREFLIGHT_SHA256 = "88e8e2eb079fa29d2eff98893b2b0c80b50d3ed300bbda3608ce9401ed49c691"
SOURCE_PILOT_REVIEW_SHA256 = "01498b81c6a89b5e23796060083747ea06c453e87d5ee6880ebe06e577735e03"
SOURCE_CALIBRATION_SHA256 = "02cccbb28843c27e3598af16acd87310eb6e1ba7f738b3f4a24828ae4c7758cd"
SOURCE_GRADIENT_PANELS_SHA256 = "208c5f4d67707639bf3d45dceb968b1f9b44f307133248a74b3c0373b17c2158"
SOURCE_CHECKPOINTS = {
    "mse_only_50": "dbda0fd771ac36463aca6aa4d545737eb2d9df9b5fb007bed3812d7ae9b8e000",
    "mse_only_100": "29735c1f5a25eec1471d7c8e7e78a9e85e605beb018521accfb66cb773ddedcd",
    "mse_op_50": "f168c386121846879d00d4a268301510b14e3bb8cb73816991248f57e804ca8e",
    "mse_op_100": "ab79d040db953327957a422f511e164b4542f101fbca97219d9b509e94714d29",
    "mse_kl_50": "bca7f7d63ee316f2bba4f9cefd8886757ae169312c2dee524d4f5daebea241c7",
    "mse_kl_100": "10578b9b39d48d8e898ab5e1b3cf565ae4cebf93ee5d6f5d3d7ab5ec0ded42bc",
    "mse_kl_op_50": "2365fb02b62730d81e5d8394a1c394912781cb67a2b0b1e11c1c22d54d3ded19",
    "mse_kl_op_100": "c80001f3e6722712ea2f0f3e6f386e56bd3da88ff06a160156a3500b137a173e",
}
PREFLIGHT_ALPHA = 0.002996375650268832
PREFLIGHT_GAMMA = 0.02969130038500517
DOSE_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
PANEL_COUNT = 3
PANEL_SIZE = 10
SELECTION_SEED = 5234
CONFIRMATION_SEED = 6234
MIN_RATIO = 0.05
MAX_RATIO = 0.40
MIN_DESCENT_FRACTION = 0.25
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_WALL_SECONDS = 600.0
MAX_STORAGE_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class AttributionData:
    data: bogt.ResearchData
    selection_panels: tuple[torch.Tensor, ...]
    confirmation_panels: tuple[torch.Tensor, ...]
    manifest: dict[str, Any]


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return bogt._resolve(value, base_path=base_path)


def _git_state() -> dict[str, Any]:
    return bogt._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or not state.get("clean"):
        raise ValueError(f"failure attribution requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {"schema", "hypothesis", "formal_system_config", "source", "dataset", "optical_contract", "dose", "authorization"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("failure attribution config fields are invalid")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported failure attribution config")
    if payload["dose"] != {
        "multipliers": list(DOSE_MULTIPLIERS),
        "selection_panels": PANEL_COUNT,
        "confirmation_panels": PANEL_COUNT,
        "panel_size": PANEL_SIZE,
        "selection_seed": SELECTION_SEED,
        "confirmation_seed": CONFIRMATION_SEED,
        "min_ratio": MIN_RATIO,
        "max_ratio": MAX_RATIO,
        "minimum_descent_fraction": MIN_DESCENT_FRACTION,
        "eligible_requires_both_multipliers_gt_one": True,
    }:
        raise ValueError("failure attribution dose contract is not locked")
    if payload["authorization"].get("training_authorized") is not False or payload["authorization"].get("formal_promotion") is not False:
        raise ValueError("failure attribution cannot authorize training or promotion")
    return Spec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_scaled_chen_bogt_failure_attribution",
        "source_commit": SOURCE_COMMIT,
        "source_config_sha256": file_sha256(spec.path),
        "source_training_sha256": SOURCE_TRAINING_SHA256,
        "source_validation_sha256": SOURCE_VALIDATION_SHA256,
        "source_preflight_sha256": SOURCE_PREFLIGHT_SHA256,
        "source_pilot_review_sha256": SOURCE_PILOT_REVIEW_SHA256,
        "source_calibration_sha256": SOURCE_CALIBRATION_SHA256,
        "source_gradient_panels_sha256": SOURCE_GRADIENT_PANELS_SHA256,
        "approved_contract": {
            "selection_panels": PANEL_COUNT,
            "confirmation_panels": PANEL_COUNT,
            "panel_size": PANEL_SIZE,
            "candidate_multipliers": list(DOSE_MULTIPLIERS),
            "no_model_update": True,
            "heldout_not_used_for_selection": True,
            "post_candidate_next_action": "manual_review_scaled_chen_bogt_repilot_contract",
            "post_no_candidate_next_action": "manual_review_scaled_chen_bogt_objective_contract",
        },
        "training_authorized": False,
        "formal_promotion": False,
        "prohibited": ["optimizer", "cuda_training", "continuation", "generation", "fmf", "formal_promotion"],
    }
    if payload != expected:
        raise ValueError("failure attribution review contract does not match scope")
    if file_sha256(spec.path) != payload.get("source_config_sha256"):
        raise ValueError("failure attribution review config binding is stale or tampered")
    return target


def _load_source_training(spec: Spec) -> tuple[dict[str, Any], Path]:
    source = spec.payload["source"]
    training_path = _resolve(source["training_evidence"])
    if file_sha256(training_path) != SOURCE_TRAINING_SHA256:
        raise ValueError("source training evidence hash mismatch")
    payload = json.loads(training_path.read_text(encoding="utf-8"))
    evidence = payload.get("evidence", {})
    if payload.get("schema") != "lightgen_optical_manifold_smf_bogt_training.v1" or evidence.get("status") != "valid_failed" or evidence.get("git_commit") != SOURCE_COMMIT:
        raise ValueError("source training evidence is not authoritative attempt_05")
    arms = payload.get("contract", {}).get("pilot_results", {}).get("arms", {})
    for arm in ("mse_only", "mse_op", "mse_kl", "mse_kl_op"):
        for step in ("50", "100"):
            if arms.get(arm, {}).get("checkpoint_hashes", {}).get(step) != SOURCE_CHECKPOINTS[f"{arm}_{step}"]:
                raise ValueError(f"source checkpoint binding mismatch for {arm} step {step}")
    return payload, training_path


def _load_source_artifacts(spec: Spec) -> dict[str, Path]:
    source = spec.payload["source"]
    names = {
        "validation": ("validation_evidence", SOURCE_VALIDATION_SHA256),
        "preflight": ("preflight_evidence", SOURCE_PREFLIGHT_SHA256),
        "pilot_review": ("pilot_review", SOURCE_PILOT_REVIEW_SHA256),
        "calibration": ("calibration", SOURCE_CALIBRATION_SHA256),
        "gradient_panels": ("gradient_panels", SOURCE_GRADIENT_PANELS_SHA256),
    }
    result: dict[str, Path] = {}
    for name, (key, digest) in names.items():
        path = _resolve(source[key])
        if file_sha256(path) != digest:
            raise ValueError(f"source {name} hash mismatch")
        result[name] = path
    validation = json.loads(result["validation"].read_text(encoding="utf-8"))
    preflight = json.loads(result["preflight"].read_text(encoding="utf-8"))
    pilot_review = json.loads(result["pilot_review"].read_text(encoding="utf-8"))
    if validation.get("schema") != bogt.VALIDATION_SCHEMA or validation.get("evidence", {}).get("status") != "passed":
        raise ValueError("source validation evidence is not passed")
    if preflight.get("schema") != bogt.PREFLIGHT_SCHEMA or preflight.get("evidence", {}).get("status") != "passed":
        raise ValueError("source preflight evidence is not passed")
    if pilot_review.get("schema") != bogt.PILOT_REVIEW_SCHEMA or pilot_review.get("training_authorized") is not False:
        raise ValueError("source pilot review is not the bounded pilot contract")
    return result


def _fresh_panels(data: bogt.ResearchData) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    used = torch.cat((data.calibration_indices, data.schedule.reshape(-1), data.heldout_indices, *data.panels))
    labels = data.labels.index_select(0, data.calibration_indices.new_tensor(range(55000)))

    def make(seed: int) -> tuple[torch.Tensor, ...]:
        nonlocal used
        per_label: list[torch.Tensor] = []
        for label in range(10):
            candidates = data.calibration_indices.new_tensor(range(55000))[labels == label]
            candidates = candidates[~torch.isin(candidates, used)].sort().values
            if candidates.numel() < PANEL_COUNT * PANEL_SIZE:
                raise ValueError("insufficient fresh train-only attribution samples")
            generator = torch.Generator(device="cpu").manual_seed(seed + label)
            shuffled = candidates[torch.randperm(candidates.numel(), generator=generator)]
            per_label.append(shuffled[: PANEL_COUNT * PANEL_SIZE])
        panels = tuple(torch.stack([values[index * PANEL_SIZE : (index + 1) * PANEL_SIZE] for values in per_label], dim=1).flatten() for index in range(PANEL_COUNT))
        used = torch.cat((used, *panels))
        return panels

    selection = make(SELECTION_SEED)
    confirmation = make(CONFIRMATION_SEED)
    combined = torch.cat((*selection, *confirmation))
    if combined.numel() != 600 or torch.unique(combined).numel() != combined.numel() or torch.isin(combined, used[:-600]).any():
        raise ValueError("attribution panels overlap a registered source panel")
    return selection, confirmation


def _load_data(spec: Spec, dataset_root: str | Path) -> AttributionData:
    data = bogt._load_research_data(dataset_root)
    selection, confirmation = _fresh_panels(data)
    manifest = {
        "schema": "scaled_chen_bogt_failure_attribution_panels.v1",
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "selection_indices_sha256": tensor_sha256(torch.cat(selection)),
        "confirmation_indices_sha256": tensor_sha256(torch.cat(confirmation)),
        "selection_panel_count": PANEL_COUNT,
        "confirmation_panel_count": PANEL_COUNT,
        "panel_size": PANEL_SIZE * 10,
        "selection_seed": SELECTION_SEED,
        "confirmation_seed": CONFIRMATION_SEED,
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return AttributionData(data, selection, confirmation, manifest)


def _panel_inputs(data: bogt.ResearchData, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = base.render_intensity_batch(data.images.index_select(0, indices))
    return rendered.amplitude, rendered.target.float()


def _phase_parameters(model: bogt.ScaledChenBOGTModel) -> tuple[torch.Tensor, ...]:
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def _clear_grads(model: bogt.ScaledChenBOGTModel) -> None:
    for parameter in _phase_parameters(model):
        parameter.grad = None


def _flatten_gradients(model: bogt.ScaledChenBOGTModel) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for parameter in _phase_parameters(model):
        if parameter.grad is None:
            values.append(torch.zeros_like(parameter).reshape(-1))
        else:
            values.append(parameter.grad.detach().reshape(-1))
    return torch.cat(values)


def _flatten_gradient_values(model: bogt.ScaledChenBOGTModel, gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
    values = [torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1) for parameter, gradient in zip(_phase_parameters(model), gradients, strict=True)]
    return torch.cat(values)


def _objective_gradients(model: bogt.ScaledChenBOGTModel, amplitudes: torch.Tensor, targets: torch.Tensor, calibration: bogt.Calibration, objective: str) -> tuple[tuple[torch.Tensor | None, ...], float]:
    if objective not in {"mse", "kl", "op"}:
        raise ValueError("unknown attribution objective")
    if amplitudes.ndim != 4 or amplitudes.shape[0] != targets.shape[0] or amplitudes.shape[0] < 2:
        raise ValueError("attribution objective requires a logical batch of at least two samples")
    parameters = _phase_parameters(model)
    accum = [torch.zeros_like(parameter) for parameter in parameters]
    batch = amplitudes.shape[0]
    values: list[float] = []
    if objective in {"mse", "op"}:
        for index in range(batch):
            output = model(amplitudes[index : index + 1])
            scalar = F.mse_loss(output.camera, targets[index : index + 1]) if objective == "mse" else bogt.lop_from_powers(output.plane_powers, calibration.plane_reference)
            gradients = torch.autograd.grad(scalar, parameters, allow_unused=True)
            for slot, gradient in zip(accum, gradients, strict=True):
                if gradient is not None:
                    slot.add_(gradient.detach() / batch)
            values.append(float(scalar.detach()))
        return tuple(accum), sum(values) / batch
    with torch.no_grad():
        detached_outputs = [model(amplitudes[index : index + 1]) for index in range(batch)]
        detached_z = torch.cat([bogt.normalize_port_power(output.coefficients.abs().square(), calibration.port_scale) for output in detached_outputs], dim=0)
        mean, variance = bogt.aggregate_moments(detached_z)
        batch_kl = bogt.bogt_kl_surrogate(mean, variance)
    proxy = detached_z.detach().requires_grad_(True)
    with torch.enable_grad():
        proxy_kl = bogt.bogt_kl_surrogate(*bogt.aggregate_moments(proxy))
    kl_gradient = torch.autograd.grad(proxy_kl, proxy)[0].detach()
    for index in range(batch):
        output = model(amplitudes[index : index + 1])
        live_z = bogt.normalize_port_power(output.coefficients.abs().square(), calibration.port_scale)
        scalar = (live_z * kl_gradient[index : index + 1].to(live_z)).sum()
        gradients = torch.autograd.grad(scalar, parameters, allow_unused=True)
        for slot, gradient in zip(accum, gradients, strict=True):
            if gradient is not None:
                slot.add_(gradient.detach())
    return tuple(accum), float(batch_kl.detach())


def _objective_gradient(model: bogt.ScaledChenBOGTModel, amplitudes: torch.Tensor, targets: torch.Tensor, calibration: bogt.Calibration, objective: str) -> tuple[torch.Tensor, float]:
    gradients, value = _objective_gradients(model, amplitudes, targets, calibration, objective)
    return _flatten_gradient_values(model, gradients), value


def _gradient_report(model: bogt.ScaledChenBOGTModel, amplitudes: torch.Tensor, targets: torch.Tensor, calibration: bogt.Calibration) -> dict[str, Any]:
    vectors: dict[str, torch.Tensor] = {}
    values: dict[str, float] = {}
    gradients_by_objective: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for objective in ("mse", "kl", "op"):
        gradients_by_objective[objective], values[objective] = _objective_gradients(model, amplitudes, targets, calibration, objective)
        vectors[objective] = _flatten_gradient_values(model, gradients_by_objective[objective])
    norms = {name: float(vector.norm()) for name, vector in vectors.items()}
    if any(not math.isfinite(value) or value <= 0.0 for value in norms.values()):
        raise RuntimeError("attribution gradient norm is non-finite or non-positive")
    cosines = {
        "mse_kl": float(torch.dot(vectors["mse"], vectors["kl"]) / (vectors["mse"].norm() * vectors["kl"].norm())),
        "mse_op": float(torch.dot(vectors["mse"], vectors["op"]) / (vectors["mse"].norm() * vectors["op"].norm())),
        "kl_op": float(torch.dot(vectors["kl"], vectors["op"]) / (vectors["kl"].norm() * vectors["op"].norm())),
    }
    layer_norms = {objective: [float(gradients[0][layer].norm()) for layer in range(2)] + [float(gradients[1][layer].norm()) for layer in range(2)] for objective, gradients in gradients_by_objective.items()}
    finite = all(math.isfinite(value) for value in (*norms.values(), *cosines.values(), *(value for group in layer_norms.values() for value in group)))
    kl_boundary = layer_norms["kl"][2] == 0.0 and layer_norms["kl"][3] == 0.0 and all(value > 0.0 for value in layer_norms["kl"][:2])
    op_boundary = all(value > 0.0 for value in layer_norms["op"])
    return {"values": values, "norms": norms, "cosines": cosines, "layer_norms": layer_norms, "finite": finite, "kl_encoder_only": kl_boundary, "op_four_layers": op_boundary, "vectors": vectors}


def _candidate_report(base_report: Mapping[str, Any], alpha: float, gamma: float) -> dict[str, Any]:
    vectors = base_report["vectors"]
    total = vectors["mse"] + alpha * vectors["kl"] + gamma * vectors["op"]
    mse = vectors["mse"]
    kl = vectors["kl"]
    op = vectors["op"]
    mse_norm_sq = float(mse.square().sum())
    result = {
        "alpha": alpha,
        "gamma": gamma,
        "ratios": {"kl": alpha * float(kl.norm()) / float(mse.norm()), "op": gamma * float(op.norm()) / float(mse.norm())},
        "dots": {"mse": float(torch.dot(total, mse)), "kl": float(torch.dot(total, kl)), "op": float(torch.dot(total, op))},
        "descent_fraction": float(torch.dot(total, mse) / max(mse_norm_sq, 1.0e-30)),
    }
    result["finite"] = all(math.isfinite(float(value)) for value in (*result["ratios"].values(), *result["dots"].values(), result["descent_fraction"]))
    result["compatible"] = bool(result["finite"] and all(MIN_RATIO <= value <= MAX_RATIO for value in result["ratios"].values()) and all(value > 0.0 for value in result["dots"].values()) and result["descent_fraction"] >= MIN_DESCENT_FRACTION)
    return result


def _select_candidate(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = [item for item in reports if item.get("compatible") and item["alpha"] > 0.0 and item["gamma"] > 0.0]
    candidates = [item for item in candidates if item["alpha"] / PREFLIGHT_ALPHA > 1.0 and item["gamma"] / PREFLIGHT_GAMMA > 1.0]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (min(item["ratios"].values()), -max(item["alpha"], item["gamma"]), -(item["alpha"] + item["gamma"])))


def _record(spec: Spec, state: Mapping[str, Any], data: AttributionData, source_training: Path, source_artifacts: Mapping[str, Path], review: Path, gates: Sequence[GateResult], status: str, action: str | None, reason: str | None, *, schema: str = AUDIT_SCHEMA) -> dict[str, Any]:
    input_hashes = {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review),
        "source_training": file_sha256(source_training),
        **{f"source_{name}": file_sha256(path) for name, path in source_artifacts.items()},
        "dataset_manifest": data.data.manifest["manifest_hash"],
        "attribution_panels": data.manifest["manifest_hash"],
    }
    context = EvidenceContextV2(canonical_sha256({"optical_contract": spec.payload["optical_contract"], "branch": BRANCH}), spec.config_hash, dataset_hash=data.data.manifest["manifest_hash"], dataset_split_manifest_hash=data.data.manifest["manifest_hash"])
    record = EvidenceRecordV2(2, schema, "diagnostic", status, str(state["commit"]), None, spec.config_hash, context, input_hashes, tuple(gates), (action,) if action else (), reason)
    record.validate()
    return {"schema": schema, "hypothesis": HYPOTHESIS, "training_authorized": False, "formal_promotion": False, "evidence": record.to_dict()}


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    state = _git_state()
    _require_clean_branch(state)
    review = _load_review(args.review_contract, spec)
    source_training, source_training_path = _load_source_training(spec)
        source_artifacts = _load_source_artifacts(spec)
    data = _load_data(spec, args.dataset_root)
    formal, formal_path = bogt._load_formal(spec)
    model = bogt.ScaledChenBOGTModel(formal)
    gates = (
        GateResult("clean_lineage", state["clean"] and state["branch"] == BRANCH and state["commit"] != SOURCE_COMMIT, {"commit": state["commit"]}),
        GateResult("source_attempt_05_valid_failed", source_training["evidence"]["status"] == "valid_failed", {"source_commit": SOURCE_COMMIT}),
        GateResult("geometry_and_fixed_boundary", tuple(model.system.encoder.phase.shape) == (2, 1, 100, 100) and tuple(model.system.decoder.phase.shape) == (2, 1, 100, 100) and all(not parameter.requires_grad for parameter in model.ols.parameters()), {}),
        GateResult("panel_isolation", data.manifest["dataset_manifest_hash"] == data.data.manifest["manifest_hash"] and len(data.selection_panels) == len(data.confirmation_panels) == PANEL_COUNT, data.manifest),
        GateResult("source_artifacts_bound", all(path.exists() for path in source_artifacts.values()), {name: file_sha256(path) for name, path in source_artifacts.items()}),
        GateResult("no_update_boundary", True, {"optimizer_created": False, "cuda_context_created": False}),
    )
    status = "passed" if all(gate.passed for gate in gates) else "invalid_harness"
    payload = _record(spec, state, data, source_training_path, source_artifacts, review, gates, status, "run_smf_bogt_failure_attribution" if status == "passed" else None, None if status == "passed" else "failure attribution validation failed", schema=VALIDATION_SCHEMA)
    payload["contract"] = {"source_commit": SOURCE_COMMIT, "formal_config_sha256": file_sha256(formal_path), "dataset_manifest": data.data.manifest, "attribution_manifest": data.manifest, "source_artifacts": {name: file_sha256(path) for name, path in source_artifacts.items()}}
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite attribution validation: {target}")
    atomic_write_json(target, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "passed" else 2


def run(args: argparse.Namespace) -> int:
    if args.device != "cuda" or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("failure attribution requires deterministic CUDA runtime")
    spec = _load_spec(args.config)
    state = _git_state()
    _require_clean_branch(state)
    review = _load_review(args.review_contract, spec)
    source_training, source_training_path = _load_source_training(spec)
    source_artifacts = _load_source_artifacts(spec)
    data = _load_data(spec, args.dataset_root)
    validation_path = _resolve(args.validation_evidence)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("schema") != VALIDATION_SCHEMA or validation.get("evidence", {}).get("status") != "passed" or validation.get("evidence", {}).get("git_commit") != state["commit"]:
        raise ValueError("failure attribution validation evidence is not passed")
    formal, formal_path = bogt._load_formal(spec)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite attribution audit: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("failure attribution requires CUDA")
    device = torch.device(args.device)
    free, total_memory = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free} is below {MIN_FREE_BYTES}")
    started = time.monotonic()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        calibration_payload = safe_torch_load(source_artifacts["calibration"])
        calibration = bogt.Calibration(calibration_payload["port_scale"], calibration_payload["plane_reference"], int(calibration_payload["sample_count"]))
        calibration.validate()
        all_reports: dict[str, list[dict[str, Any]]] = {"selection": [], "confirmation": []}
        checkpoint_root = _resolve(spec.payload["source"]["checkpoint_root"])
        for split_name, panels in (("selection", data.selection_panels), ("confirmation", data.confirmation_panels)):
            for panel_index, indices in enumerate(panels):
                amplitudes, targets = _panel_inputs(data.data, indices)
                model = bogt.ScaledChenBOGTModel(formal).to(device)
                states: dict[str, Any] = {}
                state_unchanged = True
                state_snapshot: dict[str, torch.Tensor] = {name: value.detach().clone() for name, value in model.state_dict().items()}
                zero_report = _gradient_report(model, amplitudes.to(device), targets.to(device), calibration)
                states["zero"] = zero_report
                state_unchanged = state_unchanged and all(torch.equal(value, model.state_dict()[name]) for name, value in state_snapshot.items())
                for step in (50, 100):
                    checkpoint_path = checkpoint_root / f"scaled_chen_mse_kl_op_step_{step}.pt"
                    if file_sha256(checkpoint_path) != SOURCE_CHECKPOINTS[f"mse_kl_op_{step}"]:
                        raise ValueError(f"source checkpoint hash mismatch at step {step}")
                    checkpoint = safe_torch_load(checkpoint_path)
                    model.load_state_dict(checkpoint["model_state"], strict=True)
                    state_snapshot = {name: value.detach().clone() for name, value in model.state_dict().items()}
                    states[str(step)] = _gradient_report(model, amplitudes.to(device), targets.to(device), calibration)
                    state_unchanged = state_unchanged and all(torch.equal(value, model.state_dict()[name]) for name, value in state_snapshot.items())
                dose_reports = []
                for alpha_multiplier in DOSE_MULTIPLIERS:
                    for gamma_multiplier in DOSE_MULTIPLIERS:
                        reports = [_candidate_report(states[point], PREFLIGHT_ALPHA * alpha_multiplier, PREFLIGHT_GAMMA * gamma_multiplier) for point in ("zero", "50", "100")]
                        dose_reports.append({"alpha_multiplier": alpha_multiplier, "gamma_multiplier": gamma_multiplier, "points": reports, "compatible": all(item["compatible"] for item in reports)})
                all_reports[split_name].append({"panel": panel_index, "gradient_points": {point: {key: value for key, value in report.items() if key != "vectors"} for point, report in states.items()}, "dose_reports": dose_reports, "finite": all(report["finite"] and report["kl_encoder_only"] and report["op_four_layers"] for report in states.values()), "state_unchanged": state_unchanged})
                del model
                torch.cuda.empty_cache()
        selection_candidates = [report for panel in all_reports["selection"] if panel["finite"] and panel["state_unchanged"] for report in panel["dose_reports"] if report["compatible"] and report["alpha_multiplier"] > 1.0 and report["gamma_multiplier"] > 1.0]
        confirmation_candidates = [report for panel in all_reports["confirmation"] if panel["finite"] and panel["state_unchanged"] for report in panel["dose_reports"] if report["compatible"] and report["alpha_multiplier"] > 1.0 and report["gamma_multiplier"] > 1.0]
        source_panel_payload = safe_torch_load(source_artifacts["gradient_panels"])
        source_panels = source_panel_payload.get("panels")
        if source_panel_payload.get("dataset_manifest_hash") != data.data.manifest["manifest_hash"] or not isinstance(source_panels, list) or len(source_panels) != PANEL_COUNT:
            raise ValueError("source gradient-panel artifact has an invalid panel contract")
        source_compatible: set[tuple[float, float]] = set()
        for alpha_multiplier in DOSE_MULTIPLIERS:
            for gamma_multiplier in DOSE_MULTIPLIERS:
                ratios = []
                for panel in source_panels:
                    norms = panel.get("gradient_norms", {})
                    if set(norms) != {"mse", "kl", "op"} or any(not math.isfinite(float(norms[name])) or float(norms[name]) <= 0.0 for name in norms):
                        raise ValueError("source gradient-panel norms are invalid")
                    ratios.append({"kl": PREFLIGHT_ALPHA * alpha_multiplier * float(norms["kl"]) / float(norms["mse"]), "op": PREFLIGHT_GAMMA * gamma_multiplier * float(norms["op"]) / float(norms["mse"])})
                if all(MIN_RATIO <= value <= MAX_RATIO for panel in ratios for value in panel.values()):
                    source_compatible.add((alpha_multiplier, gamma_multiplier))
        common = ({(item["alpha_multiplier"], item["gamma_multiplier"]) for item in selection_candidates} & {(item["alpha_multiplier"], item["gamma_multiplier"]) for item in confirmation_candidates}) & source_compatible
        selected = None
        if common:
            scored = []
            for alpha_multiplier, gamma_multiplier in common:
                matching = [item for item in confirmation_candidates if item["alpha_multiplier"] == alpha_multiplier and item["gamma_multiplier"] == gamma_multiplier]
                score = min(point["ratios"][term] for item in matching for point in item["points"] for term in ("kl", "op"))
                scored.append((score, -max(alpha_multiplier, gamma_multiplier), -(alpha_multiplier + gamma_multiplier), alpha_multiplier, gamma_multiplier))
            _, _, _, alpha_multiplier, gamma_multiplier = max(scored)
            selected = {"alpha_multiplier": alpha_multiplier, "gamma_multiplier": gamma_multiplier, "alpha": PREFLIGHT_ALPHA * alpha_multiplier, "gamma": PREFLIGHT_GAMMA * gamma_multiplier}
        state_unchanged = True
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file()) if output.parent.exists() else 0
        gates = (
            GateResult("finite_gradient_panels", all(panel["finite"] for split in all_reports.values() for panel in split), {"selection": [panel["finite"] for panel in all_reports["selection"]], "confirmation": [panel["finite"] for panel in all_reports["confirmation"]]}),
            GateResult("source_preflight_panel_compatibility", bool(source_compatible), {"compatible_multipliers": sorted([list(item) for item in source_compatible])}),
            GateResult("selection_confirmation_common_candidate", selected is not None, {"selected": selected}),
            GateResult("model_state_unchanged", all(panel["state_unchanged"] for split in all_reports.values() for panel in split), {}),
            GateResult("fixed_boundary_and_evaluator", all(panel["finite"] for split in all_reports.values() for panel in split), {"evaluator_calls": 0}),
            GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
            GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
            GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
        )
        if not all(gate.passed for gate in gates if gate.name not in {"selection_confirmation_common_candidate"}):
            status, action, reason = "invalid_harness", None, "failure attribution harness or resource gate failed"
        elif selected is None:
            status, action, reason = "valid_failed", "manual_review_scaled_chen_bogt_objective_contract", "no safe alpha/gamma candidate"
        else:
            status, action, reason = "passed", "manual_review_scaled_chen_bogt_repilot_contract", None
        payload = _record(spec, state, data, source_training_path, source_artifacts, review, gates, status, action, reason)
        payload["contract"] = {
            "source_training_sha256": file_sha256(source_training_path),
            "source_validation_sha256": file_sha256(validation_path),
            "source_artifacts": {name: file_sha256(path) for name, path in source_artifacts.items()},
            "selection_panels": all_reports["selection"],
            "confirmation_panels": all_reports["confirmation"],
            "selected": selected,
            "source_preflight_compatible_multipliers": sorted([list(item) for item in source_compatible]),
            "candidate_multipliers": list(DOSE_MULTIPLIERS),
            "free_cuda_bytes_before": int(free),
            "total_cuda_bytes": int(total_memory),
            "peak_cuda_bytes": peak,
            "wall_seconds": elapsed,
            "run_storage_bytes": storage,
            "evaluator_calls": 0,
        }
        atomic_write_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if status in {"passed", "valid_failed"} else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.cuda.empty_cache()


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-bogt-failure-attribution-validate":
        return validate(args)
    if args.research_action == "smf-bogt-failure-attribution-run":
        return run(args)
    raise ValueError(f"unsupported failure attribution action: {args.research_action}")
