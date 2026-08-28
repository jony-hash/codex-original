"""Corrected no-update beta-dose attribution audit for the SMF VAE chain."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ...config import FormalSystemConfig
from ...evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_write_json, canonical_sha256, file_sha256, tensor_sha256
from . import optical_manifold as base
from . import optical_manifold_smf_vae_kl_attribution_audit as old_audit
from . import optical_manifold_smf_vae_pilot as vae


BRANCH = "research/v2-lightgen-optical-manifold-kl-ols-loss-recovery-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_dose_reaudit_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_dose_reaudit_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_dose_reaudit_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_dose_reaudit.v1"
SOURCE_AUDIT_HASH = "40d612e20ffc7044c0a8e21bce86c7d07f3076a31ea0d6aadee5fa6f53ea22ce"
TRAINING_BETA = 0.0018256452619527068
CALIBRATION_SEED = 3234
CONFIRMATION_SEED = 4234
PANEL_COUNT = 3
PANEL_SIZE = 100
SOURCE_MSE_TOLERANCE = 1.0e-9
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_WALL_SECONDS = 1200.0
MAX_STORAGE_BYTES = 512 * 1024**2
PORT_COUNT = 100
DOSES = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
MIN_DESCENT_FRACTION = 0.25


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"] or state["branch"] != BRANCH:
        raise ValueError(f"KL dose reaudit requires clean branch {BRANCH}")


def _require_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("KL dose reaudit requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {"schema", "hypothesis", "formal_system_config", "source", "context", "audit", "optical_contract", "authorization", "gates"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("KL dose reaudit config is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    source = spec.payload["source"]
    expected_source = {
        "vae_validation_sha256": old_audit.SOURCE_VAE_VALIDATION_HASH,
        "vae_preflight_sha256": old_audit.SOURCE_VAE_PREFLIGHT_HASH,
        "vae_training_sha256": old_audit.SOURCE_VAE_TRAINING_HASH,
        "vae_panel_sha256": old_audit.SOURCE_VAE_PANEL_HASH,
        "vae_normalization_sha256": old_audit.SOURCE_VAE_NORMALIZATION_HASH,
        "source_checkpoint_sha256": old_audit.SOURCE_CHECKPOINT_HASH,
        "beta_checkpoint_sha256": old_audit.BETA_CHECKPOINT_HASH,
        "stochastic_checkpoint_sha256": old_audit.STOCHASTIC_CHECKPOINT_HASH,
        "deterministic_checkpoint_sha256": old_audit.DETERMINISTIC_CHECKPOINT_HASH,
        "mnist_manifest_sha256": old_audit.MNIST_MANIFEST_HASH,
        "kl_attribution_audit_sha256": SOURCE_AUDIT_HASH,
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise ValueError("KL dose reaudit source binding mismatch")
    audit = spec.payload["audit"]
    if audit != {
        "training_beta": TRAINING_BETA,
        "calibration_panel_count": PANEL_COUNT,
        "confirmation_panel_count": PANEL_COUNT,
        "panel_images": PANEL_SIZE,
        "calibration_seed": CALIBRATION_SEED,
        "confirmation_seed": CONFIRMATION_SEED,
        "candidate_multipliers": list(DOSES),
        "minimum_reconstruction_descent_fraction": MIN_DESCENT_FRACTION,
        "source_mse_tolerance": SOURCE_MSE_TOLERANCE,
    }:
        raise ValueError("KL dose reaudit measurement contract mismatch")
    optical = spec.payload["optical_contract"]
    expected_optical = {
        "formal_system_kind": base.EXPECTED_FORMAL_KIND,
        "formal_ols": "identity",
        "research_ols": "100_port_lp01_smf",
        "posterior_domain": "100_complex_coefficients_as_200_real",
        "posterior_status": "project_approximation",
        "input_encoding": "intensity_sqrt_amplitude",
        "target": "camera_intensity_same_fov",
        "fixed_exposure": {"scale": 1.0, "gamma": 1.0},
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "microbatch_size": 1,
        "evaluator_calls": 0,
    }
    if optical != expected_optical:
        raise ValueError("KL dose reaudit optical contract mismatch")
    authorization = spec.payload["authorization"]
    if authorization.get("reviewer_role") != "project_owner" or authorization.get("decision") != "approve_no_update_smf_vae_kl_dose_reaudit" or authorization.get("no_update_diagnostic_authorized") is not True or authorization.get("training_authorized") is not False or authorization.get("formal_training_authorized") is not False or authorization.get("formal_promotion") is not False:
        raise ValueError("KL dose reaudit authorization boundary is invalid")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_smf_vae_kl_dose_reaudit",
        "approved_contract": {
            "source_step": 1000,
            "final_step": 100,
            "calibration_panels": PANEL_COUNT,
            "confirmation_panels": PANEL_COUNT,
            "panel_images": PANEL_SIZE,
            "candidate_multipliers": list(DOSES),
            "minimum_reconstruction_descent_fraction": MIN_DESCENT_FRACTION,
            "post_candidate_next_action": "manual_review_smf_vae_kl_effect_pilot_contract",
            "post_no_candidate_next_action": "manual_review_smf_vae_kl_objective_contract",
            "post_invalid_next_action": "manual_review_smf_vae_kl_objective_contract",
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("KL dose reaudit review contract does not match scope")
    return target


def _load_formal(spec: Spec) -> tuple[FormalSystemConfig, Path]:
    proxy = old_audit.Spec(spec.path, spec.payload, spec.config_hash)
    return old_audit._load_formal(proxy)


def _load_sources(spec: Spec, formal_path: Path, current: str) -> old_audit.SourceBundle:
    proxy = old_audit.Spec(spec.path, spec.payload, spec.config_hash)
    source = old_audit._load_sources(proxy, formal_path, current)
    audit_path = _resolve(spec.payload["source"]["kl_attribution_audit"])
    if file_sha256(audit_path) != SOURCE_AUDIT_HASH:
        raise ValueError("KL attribution audit source hash mismatch")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_payload.get("schema") != old_audit.AUDIT_SCHEMA or audit_payload.get("contract", {}).get("outcome") != "diagnostic_completed_parameterization_limited":
        raise ValueError("unexpected attempt-06 KL attribution source")
    return source


def _context(spec: Spec, formal: FormalSystemConfig, data: vae.Data) -> EvidenceContextV2:
    return old_audit._context(old_audit.Spec(spec.path, spec.payload, spec.config_hash), formal, data)


def _panels(data: vae.Data, prior_panels: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    excluded = torch.cat((data.calibration_indices.flatten(), data.schedule.flatten(), data.heldout_indices.flatten(), *(panel.flatten() for panel in prior_panels)))
    used = excluded.clone()

    def make(seed: int) -> list[torch.Tensor]:
        nonlocal used
        labels = data.labels.index_select(0, data.train_indices)
        by_label: list[torch.Tensor] = []
        for label in range(10):
            candidates = data.train_indices[labels == label]
            candidates = candidates[~torch.isin(candidates, used)].sort().values
            generator = torch.Generator(device="cpu").manual_seed(seed + label)
            shuffled = candidates[torch.randperm(candidates.numel(), generator=generator)]
            by_label.append(shuffled[: PANEL_COUNT * 10])
        if any(values.numel() != PANEL_COUNT * 10 for values in by_label):
            raise ValueError("insufficient fresh balanced MNIST samples for KL dose reaudit")
        result = [torch.stack([values[index * 10 : (index + 1) * 10] for values in by_label], dim=1).flatten() for index in range(PANEL_COUNT)]
        used = torch.cat((used, *result))
        return result

    calibration = make(CALIBRATION_SEED)
    confirmation = make(CONFIRMATION_SEED)
    all_indices = torch.cat((*calibration, *confirmation))
    if all_indices.numel() != 600 or torch.unique(all_indices).numel() != 600 or torch.isin(all_indices, excluded).any():
        raise ValueError("KL dose reaudit panels overlap prior data")
    return calibration, confirmation


def _input_hashes(spec: Spec, review: Path, data: vae.Data, source: old_audit.SourceBundle, panels: tuple[list[torch.Tensor], list[torch.Tensor]]) -> dict[str, str]:
    calibration, confirmation = panels
    return {
        **source.hashes,
        "kl_attribution_audit": SOURCE_AUDIT_HASH,
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review),
        "dataset_manifest": data.manifest["manifest_hash"],
        "calibration_panels": tensor_sha256(torch.cat(calibration)),
        "confirmation_panels": tensor_sha256(torch.cat(confirmation)),
    }


def _validation_payload(state: Mapping[str, Any], spec: Spec, formal: FormalSystemConfig, data: vae.Data, source: old_audit.SourceBundle, review: Path, panels: tuple[list[torch.Tensor], list[torch.Tensor]]) -> dict[str, Any]:
    hashes = _input_hashes(spec, review, data, source, panels)
    gates = tuple(GateResult(name, True, metrics) for name, metrics in (
        ("clean_lineage", {"branch": BRANCH, "clean": True}),
        ("authoritative_vae_source", {"training": "valid_failed", "source_step": 1000}),
        ("attempt_06_preserved_as_invalid_dose_consumer", {"audit_sha256": SOURCE_AUDIT_HASH}),
        ("mnist_panel_isolation", {"calibration": 300, "confirmation": 300, "heldout": 1000}),
        ("formal_identity_smf_optics", {"grid": 2048, "formal_ols": "identity", "research_ols": "100_port_lp01_smf"}),
        ("zero_update_boundary", {"optimizer_created": False, "training_authorized": False}),
        ("source_provenance", {"input_hashes": hashes}),
    ))
    record = EvidenceRecordV2(2, VALIDATION_SCHEMA, "diagnostic", "passed", str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, gates, ("run_smf_vae_kl_dose_reaudit",))
    record.validate()
    return {
        "schema": VALIDATION_SCHEMA,
        "evidence": record.to_dict(),
        "hypothesis": HYPOTHESIS,
        "training_authorized": False,
        "formal_promotion": False,
        "contract": {
            "formal_config": str(_resolve(spec.payload["formal_system_config"], base_path=spec.path.parent).resolve()),
            "source_hashes": hashes,
            "calibration_panel_indices": [panel.tolist() for panel in panels[0]],
            "confirmation_panel_indices": [panel.tolist() for panel in panels[1]],
            "optimizer_created": False,
            "cuda_context_created": False,
        },
    }


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite KL dose reaudit artifact: {target}")
    atomic_write_json(target, payload)


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_sources(spec, formal_path, state["commit"])
    data, prior_panels = old_audit._load_data(args.dataset_root)
    panels = _panels(data, prior_panels)
    if tuple(data.images.shape) != (60000, 1, 28, 28) or tuple(data.labels.shape) != (60000,):
        raise ValueError("KL dose reaudit requires complete MNIST tensors")
    _write_new(args.output, _validation_payload(state, spec, formal, data, source, review, panels))
    print(Path(args.output).read_text(encoding="utf-8"))
    return 0


def _load_validation(path: str | Path, spec: Spec, formal: FormalSystemConfig, data: vae.Data, source: old_audit.SourceBundle, review: Path, panels: tuple[list[torch.Tensor], list[torch.Tensor]], state: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if payload.get("schema") != VALIDATION_SCHEMA or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.context != _context(spec, formal, data) or record.input_hashes != _input_hashes(spec, review, data, source, panels) or record.permitted_next_actions != ("run_smf_vae_kl_dose_reaudit",):
        raise ValueError("KL dose reaudit validation is stale or mismatched")
    return payload, digest


def _dose_report(rec: Mapping[str, torch.Tensor], kl: Mapping[str, torch.Tensor], multiplier: float) -> dict[str, Any]:
    effective_beta = TRAINING_BETA * multiplier
    rec_squared = sum(float(value.square().sum()) for value in rec.values())
    kl_squared = sum(float(value.square().sum()) for value in kl.values())
    rec_dot_kl = sum(float((rec[name] * kl[name]).sum()) for name in rec)
    combined = {name: rec[name] + effective_beta * kl[name] for name in rec}
    combined_rec = sum(float((combined[name] * rec[name]).sum()) for name in rec)
    combined_kl = sum(float((combined[name] * kl[name]).sum()) for name in rec)
    return {
        "multiplier": multiplier,
        "registered_beta": TRAINING_BETA,
        "effective_beta": effective_beta,
        "reconstruction_gradient_norm_squared": rec_squared,
        "kl_gradient_norm_squared": kl_squared,
        "reconstruction_dot_kl": rec_dot_kl,
        "combined_dot_reconstruction": combined_rec,
        "combined_dot_kl": combined_kl,
        "reconstruction_descent_fraction": combined_rec / max(rec_squared, 1.0e-30),
        "compatible_with_both": combined_rec > 0.0 and combined_kl > 0.0 and combined_rec / max(rec_squared, 1.0e-30) >= MIN_DESCENT_FRACTION,
        "finite": all(torch.isfinite(value).all() for value in combined.values()),
    }


def _phase_layer_norms(values: Mapping[str, torch.Tensor], side: str) -> list[float]:
    """Return the per-layer norms from the stacked phase parameter tensor.

    The gradient helper intentionally groups each two-layer phase stack under
    one parameter name.  The audit gate must therefore inspect the leading
    layer dimension rather than looking for synthetic ``phase[0]`` keys.
    """
    phase = values[f"system.{side}.phase"]
    if phase.ndim < 1 or phase.shape[0] != 2:
        raise ValueError(f"expected exactly two {side} phase layers")
    return [float(phase[layer].norm()) for layer in range(phase.shape[0])]


def _gradient_boundary_checks(
    rec: Mapping[str, torch.Tensor], kl: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    rec_encoder_layers = _phase_layer_norms(rec, "encoder")
    rec_decoder_layers = _phase_layer_norms(rec, "decoder")
    kl_encoder_layers = _phase_layer_norms(kl, "encoder")
    kl_decoder_norm = float(kl["system.decoder.phase"].norm())
    posterior_norms = {
        name: float(value.norm()) for name, value in kl.items() if name.startswith("posterior.")
    }
    return {
        "reconstruction_encoder_layer_norms": rec_encoder_layers,
        "reconstruction_decoder_layer_norms": rec_decoder_layers,
        "kl_encoder_layer_norms": kl_encoder_layers,
        "kl_decoder_norm": kl_decoder_norm,
        "kl_posterior_gradient_norms": posterior_norms,
        "four_phase_reconstruction": all(norm > 0.0 for norm in (*rec_encoder_layers, *rec_decoder_layers)),
        "kl_encoder_posterior_only": (
            kl_decoder_norm == 0.0
            and all(norm > 0.0 for norm in kl_encoder_layers)
            and any(norm > 0.0 for norm in posterior_norms.values())
        ),
    }


def _run_panel(model: torch.nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior, data: vae.Data, panel: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(CONFIRMATION_SEED + int(panel[0]))
    epsilon = torch.randn((panel.numel(), PORT_COUNT, 2), generator=generator, dtype=torch.float32)
    rec, rec_value = old_audit._gradient_vectors(model, operator, posterior, data, panel, mean, scale, epsilon, device, "reconstruction")
    kl, kl_value = old_audit._gradient_vectors(model, operator, posterior, data, panel, mean, scale, epsilon, device, "kl")
    rec_norms = old_audit._vector_norms(rec)
    kl_norms = old_audit._vector_norms(kl)
    boundary = _gradient_boundary_checks(rec, kl)
    doses = [_dose_report(rec, kl, multiplier) for multiplier in DOSES]
    return {
        "reconstruction_value": rec_value,
        "kl_value": kl_value,
        "reconstruction_gradient_norms": rec_norms,
        "kl_gradient_norms": kl_norms,
        "kl_decoder_norm": boundary["kl_decoder_norm"],
        "reconstruction_encoder_layer_norms": boundary["reconstruction_encoder_layer_norms"],
        "reconstruction_decoder_layer_norms": boundary["reconstruction_decoder_layer_norms"],
        "kl_encoder_layer_norms": boundary["kl_encoder_layer_norms"],
        "kl_posterior_gradient_norms": boundary["kl_posterior_gradient_norms"],
        "doses": doses,
        "finite": all(torch.isfinite(value).all() for value in (*rec.values(), *kl.values())),
        "four_phase_reconstruction": boundary["four_phase_reconstruction"],
        "kl_encoder_posterior_only": boundary["kl_encoder_posterior_only"],
    }


def _record_payload(record: EvidenceRecordV2, contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": AUDIT_SCHEMA,
        "evidence": record.to_dict(),
        "hypothesis": HYPOTHESIS,
        "training_authorized": False,
        "formal_promotion": False,
        "contract": dict(contract),
    }


def run(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("KL dose reaudit requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_sources(spec, formal_path, state["commit"])
    data, prior_panels = old_audit._load_data(args.dataset_root)
    panels = _panels(data, prior_panels)
    validation, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, source, review, panels, state)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite KL dose reaudit evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free} is below {MIN_FREE_BYTES}")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        beta_payload = source.checkpoints["smf_beta_vae"]
        model, operator, posterior = old_audit._runtime(formal, source, "smf_beta_vae", beta_payload, device)
        snapshot = old_audit._load_runtime_snapshot(model, operator, posterior)
        mean = source.normalization["mean"].to(device)
        scale = source.normalization["scale"].to(device)
        source_model, source_operator, source_posterior = old_audit._runtime(formal, source, "smf_deterministic_mse", source.source.checkpoint, device)
        source_metrics, _ = old_audit._metric(source_model, source_operator, None, data, data.heldout_indices, mean, scale, device, None, False)
        authoritative = source.vae_training["contract"]["initial_metrics"]["smf_beta_vae"]["paired_mse"]
        source_reproduction = {"measured": source_metrics["paired_mse"], "authoritative": authoritative, "difference": abs(source_metrics["paired_mse"] - authoritative), "passed": abs(source_metrics["paired_mse"] - authoritative) <= SOURCE_MSE_TOLERANCE}
        panel_reports = [_run_panel(model, operator, posterior, data, panel, mean, scale, device) for panel in (*panels[0], *panels[1])]
        candidates = [multiplier for multiplier in DOSES if all(report["finite"] and report["kl_encoder_posterior_only"] and report["four_phase_reconstruction"] and next(item for item in report["doses"] if item["multiplier"] == multiplier)["compatible_with_both"] for report in panel_reports)]
        state_unchanged = old_audit._snapshot_equal(snapshot, model, operator, posterior)
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
        gates = (
            GateResult("source_metrics_reproduced", source_reproduction["passed"], source_reproduction),
            GateResult("finite_gradients", all(report["finite"] for report in panel_reports), {}),
            GateResult("four_phase_reconstruction_gradients", all(report["four_phase_reconstruction"] for report in panel_reports), {}),
            GateResult("kl_encoder_posterior_boundary", all(report["kl_encoder_posterior_only"] for report in panel_reports), {}),
            GateResult("corrected_beta_dose_formula", all(all(item["effective_beta"] == TRAINING_BETA * item["multiplier"] for item in report["doses"]) for report in panel_reports), {"registered_beta": TRAINING_BETA}),
            GateResult("compatible_candidate", bool(candidates), {"candidates": candidates}),
            GateResult("model_state_unchanged", state_unchanged, {}),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
            GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
            GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
        )
        harness_ok = all(gate.passed for gate in gates if gate.name not in {"compatible_candidate"})
        resources_ok = all(gate.passed for gate in gates if gate.name in {"cuda_peak_memory", "wall_time", "run_storage"})
        if not harness_ok:
            status, outcome, actions = "invalid_harness", "invalid_harness", ()
        elif not resources_ok:
            status, outcome, actions = "resource_aborted", "resource_aborted", ()
        elif not candidates:
            status, outcome, actions = "valid_failed", "valid_failed_kl_dose_no_compatible_candidate", ("manual_review_smf_vae_kl_objective_contract",)
        else:
            status, outcome, actions = "diagnostic", "diagnostic_completed_relative_kl_constraint", ("manual_review_smf_vae_kl_effect_pilot_contract",)
        hashes = _input_hashes(spec, review, data, source, panels)
        hashes["validation"] = validation_hash
        record = EvidenceRecordV2(2, AUDIT_SCHEMA, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, gates, actions, None if status == "diagnostic" else "KL dose reaudit did not produce a valid candidate")
        record.validate()
        payload = _record_payload(record, {
            "outcome": outcome,
            "registered_beta": TRAINING_BETA,
            "candidate_multipliers": list(DOSES),
            "selected_multiplier": min(candidates) if candidates else None,
            "source_reproduction": source_reproduction,
            "calibration_panels": panel_reports[:PANEL_COUNT],
            "confirmation_panels": panel_reports[PANEL_COUNT:],
            "free_cuda_bytes_before": int(free),
            "total_cuda_bytes": int(total),
            "peak_cuda_bytes": peak,
            "wall_seconds": elapsed,
            "run_storage_bytes": storage,
            "evaluator_calls": 0,
            "formal_candidate_checkpoint": False,
            "attempt_06_dose_result_invalid_harness": True,
        })
        _write_new(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if status in {"diagnostic", "valid_failed"} else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.cuda.empty_cache()


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-kl-dose-reaudit-validate":
        return validate(args)
    if args.research_action == "smf-kl-dose-reaudit-run":
        return run(args)
    raise ValueError(f"unsupported KL dose reaudit action: {args.research_action}")
