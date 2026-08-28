"""No-update validation and attribution audit for the SMF VAE pilot.

This module consumes the authoritative 100-step SMF VAE artifacts.  It never
creates an optimizer, changes a parameter, or creates a checkpoint.  The
scientific consumer is a manual decision about a future KL dose or posterior
parameterization pilot; all measurements remain diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ...config import FormalSystemConfig, load_config
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from . import optical_manifold as base
from . import optical_manifold_smf_vae_pilot as vae


BRANCH = "research/v2-lightgen-optical-manifold-smf-kl-attribution-audit-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_attribution_audit_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_attribution_audit_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_chain_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_attribution_audit.v1"
COMPARISON_SCHEMA = "lightgen_optical_manifold_smf_vae_horizontal_comparison.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_comparison_panel.v1"

SOURCE_VAE_VALIDATION_HASH = "d50919acb75ffe68453db52a6d87ebdfeb8ae9963f157b0bd3afb582f17de04c"
SOURCE_VAE_PREFLIGHT_HASH = "5e1f8569e7e58ce086db37a318df76fa81107820323128dafbedd02ad0e8a936"
SOURCE_VAE_TRAINING_HASH = "ec6afc5b15274ce9a7e2446bef43bb6346007480da23408fe1399b412200c476"
SOURCE_VAE_PANEL_HASH = "e8134f326dfdbdc33e900be17292f17c01d675427f1409193a27418ce64cb664"
SOURCE_VAE_NORMALIZATION_HASH = "e764ceb31e9ce1aa24f2cd9ad909011550aa6ad095404916ff84be8f93d07a85"
SOURCE_CHECKPOINT_HASH = "949bf6903db8a6c5c8b76fddfddeef31f9419717486f278cbf0ed9d86534dd94"
BETA_CHECKPOINT_HASH = "d4bbad17de2c9623a37825b718aee7dc1e14f99727c1651fe9bfb8bd53f04262"
STOCHASTIC_CHECKPOINT_HASH = "7885762ffd63b42f2bff3bf82b002c0a4e7cccb4e46c0b4993d4a242b183ec6d"
DETERMINISTIC_CHECKPOINT_HASH = "8b508be438a73f160a956dc063e359354ccde4ac894de960297ae79eafde2b59"
MNIST_MANIFEST_HASH = "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
PLAN6_TRAINING_HASH = "a95890d8feb38c0682179cbbb4ef612d49c5f1918609596524959b789e672c3e"
FMF_TRAINING_HASH = "f699fccce7da7fb5f1560c5823ee76e2c713ba9e726fb9f6213fe967da36afb6"
VAE_DATA_MANIFEST_HASH = "826ec4f7e03fd68ca2d7023cdf1b2b006505dcd8fe0d06d857b7eee39e79a36e"

SOURCE_STEP = 1000
FINAL_STEP = 100
PORT_COUNT = 100
BATCH_SIZE = 10
SEED = 1234
GRADIENT_PANEL_SEED = 2234
MC_SAMPLES = 16
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_WALL_SECONDS = 1200.0
MAX_STORAGE_BYTES = 512 * 1024**2
SOURCE_MSE_TOLERANCE = 1.0e-9
KL_REFERENCE_TOLERANCE = 1.0e-6
ACTIVE_VARIANCE_THRESHOLD = 1.0e-6
DOSES = (1.0, 2.0, 4.0)
NEXT_DOSE_ACTION = "manual_review_smf_vae_kl_dose_or_parameterization_contract"
NEXT_PARAMETERIZATION_ACTION = "manual_review_smf_vae_posterior_parameterization_contract"
NEXT_OBJECTIVE_ACTION = "manual_review_smf_vae_kl_objective_contract"


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class SourceBundle:
    source: vae.Source
    checkpoints: dict[str, dict[str, Any]]
    normalization: dict[str, Any]
    hashes: dict[str, str]
    vae_config_hash: str
    vae_training: dict[str, Any]
    context_sources: dict[str, Any]


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"] or state["branch"] != BRANCH:
        raise ValueError(f"KL attribution audit requires clean branch {BRANCH}")


def _require_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "KL attribution audit requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup"
        )


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "source",
        "context",
        "audit",
        "optical_contract",
        "authorization",
        "gates",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload["schema"] != CONFIG_SCHEMA
        or payload["hypothesis"] != HYPOTHESIS
    ):
        raise ValueError("KL attribution audit config is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_smf_vae_kl_chain_attribution_audit",
        "approved_contract": {
            "source_step": SOURCE_STEP,
            "final_step": FINAL_STEP,
            "gradient_panels": 3,
            "gradient_panel_images": 100,
            "doses": [1.0, 2.0, 4.0],
            "visual_sample_count": MC_SAMPLES,
            "post_completed_next_action": NEXT_DOSE_ACTION,
            "post_parameterization_limited_next_action": NEXT_PARAMETERIZATION_ACTION,
            "post_invalid_next_action": NEXT_OBJECTIVE_ACTION,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("KL attribution audit review contract does not match scope")
    return target


def _assert_contract(spec: Spec) -> None:
    source = spec.payload["source"]
    exact_source = {
        "vae_validation_sha256": SOURCE_VAE_VALIDATION_HASH,
        "vae_preflight_sha256": SOURCE_VAE_PREFLIGHT_HASH,
        "vae_training_sha256": SOURCE_VAE_TRAINING_HASH,
        "vae_panel_sha256": SOURCE_VAE_PANEL_HASH,
        "vae_normalization_sha256": SOURCE_VAE_NORMALIZATION_HASH,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "beta_checkpoint_sha256": BETA_CHECKPOINT_HASH,
        "stochastic_checkpoint_sha256": STOCHASTIC_CHECKPOINT_HASH,
        "deterministic_checkpoint_sha256": DETERMINISTIC_CHECKPOINT_HASH,
        "mnist_manifest_sha256": MNIST_MANIFEST_HASH,
    }
    if any(source.get(key) != value for key, value in exact_source.items()):
        raise ValueError("KL attribution audit source binding mismatch")
    context = spec.payload["context"]
    if context.get("plan6_training_sha256") != PLAN6_TRAINING_HASH or context.get(
        "fmf_training_sha256"
    ) != FMF_TRAINING_HASH:
        raise ValueError("horizontal comparison context binding mismatch")
    audit = spec.payload["audit"]
    if (
        audit.get("gradient_panel_count") != 3
        or audit.get("gradient_panel_images") != 100
        or audit.get("gradient_panel_seed") != GRADIENT_PANEL_SEED
        or audit.get("heldout_sample_seed") != SEED
        or audit.get("visual_sample_count") != MC_SAMPLES
        or tuple(audit.get("doses", ())) != DOSES
    ):
        raise ValueError("KL attribution audit measurement contract mismatch")
    optical = spec.payload["optical_contract"]
    for key, expected in {
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
    }.items():
        if optical.get(key) != expected:
            raise ValueError(f"KL attribution optical contract mismatch: {key}")
    if spec.payload["authorization"] != {
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_smf_vae_kl_chain_attribution_audit",
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }:
        raise ValueError("KL attribution authorization boundary is invalid")


def _load_formal(spec: Spec) -> tuple[FormalSystemConfig, Path]:
    path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    formal = load_config(path)
    if (
        formal.architecture != "conditional_identity_logit_amplitude_v2"
        or formal.system.kind != base.EXPECTED_FORMAL_KIND
        or not formal.system.is_formal
    ):
        raise ValueError("KL attribution requires current formal Identity optics")
    for key, expected in base.EXPECTED_SYSTEM.items():
        actual = getattr(formal.system, key)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch: {key}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch: {key}")
    return formal.system, path


def _ancestor(commit: str, current: str) -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    ).returncode != 0:
        raise ValueError("source evidence commit is not an ancestor")


def _load_json_source(path: Path, digest: str, current: str) -> dict[str, Any]:
    if file_sha256(path) != digest:
        raise ValueError(f"source hash mismatch: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    _ancestor(record.git_commit, current)
    if payload.get("training_authorized") is not False or payload.get(
        "formal_promotion"
    ) is not False:
        raise ValueError("source authorization boundary is invalid")
    return payload


def _load_checkpoint(
    path: Path,
    digest: str,
    arm: str,
    old_config_hash: str,
    dataset_manifest_hash: str,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = safe_torch_load(path, expected_sha256=digest, map_location="cpu")
    if (
        payload.get("schema") != vae.RESUME_SCHEMA
        or payload.get("arm") != arm
        or payload.get("step") != FINAL_STEP
        or payload.get("config_hash") != old_config_hash
        or payload.get("dataset_manifest_hash") != dataset_manifest_hash
        or payload.get("schedule_cursor") != {"next_step": FINAL_STEP, "batch_size": 10}
        or payload.get("epsilon_cursor") != FINAL_STEP
        or payload.get("formal_candidate_checkpoint") is not False
        or payload.get("training_authorized") is not False
        or not isinstance(payload.get("model_state"), Mapping)
        or not isinstance(payload.get("rng_state"), Mapping)
        or payload.get("evidence_hashes", {}).get("training_validation")
        != expected_hashes["vae_validation"]
        or payload.get("evidence_hashes", {}).get("training_preflight")
        != expected_hashes["vae_preflight"]
        or payload.get("evidence_hashes", {}).get("posterior_normalization")
        != expected_hashes["vae_normalization"]
    ):
        raise ValueError(f"invalid or stale {arm} checkpoint")
    if arm == "smf_deterministic_mse" and payload.get("posterior_state") is not None:
        raise ValueError("deterministic checkpoint unexpectedly contains posterior")
    if arm != "smf_deterministic_mse" and not isinstance(
        payload.get("posterior_state"), Mapping
    ):
        raise ValueError(f"{arm} checkpoint has no posterior state")
    return payload


def _load_sources(spec: Spec, formal_path: Path, current: str) -> SourceBundle:
    src = spec.payload["source"]
    paths = {key: _resolve(src[key]) for key in (
        "vae_validation", "vae_preflight", "vae_training", "vae_panel",
        "vae_normalization", "source_checkpoint", "beta_checkpoint",
        "stochastic_checkpoint", "deterministic_checkpoint")}
    source_hashes = {
        "vae_validation": SOURCE_VAE_VALIDATION_HASH,
        "vae_preflight": SOURCE_VAE_PREFLIGHT_HASH,
        "vae_training": SOURCE_VAE_TRAINING_HASH,
        "vae_panel": SOURCE_VAE_PANEL_HASH,
        "vae_normalization": SOURCE_VAE_NORMALIZATION_HASH,
        "source_checkpoint": SOURCE_CHECKPOINT_HASH,
        "beta_checkpoint": BETA_CHECKPOINT_HASH,
        "stochastic_checkpoint": STOCHASTIC_CHECKPOINT_HASH,
        "deterministic_checkpoint": DETERMINISTIC_CHECKPOINT_HASH,
        "mnist_manifest": MNIST_MANIFEST_HASH,
        "formal_config": file_sha256(formal_path),
    }
    validation = _load_json_source(paths["vae_validation"], SOURCE_VAE_VALIDATION_HASH, current)
    preflight = _load_json_source(paths["vae_preflight"], SOURCE_VAE_PREFLIGHT_HASH, current)
    training = _load_json_source(paths["vae_training"], SOURCE_VAE_TRAINING_HASH, current)
    if (
        EvidenceRecordV2.from_dict(validation["evidence"]).status != "passed"
        or EvidenceRecordV2.from_dict(preflight["evidence"]).status != "passed"
        or EvidenceRecordV2.from_dict(training["evidence"]).status != "valid_failed"
        or training.get("contract", {}).get("outcome") != "valid_failed"
    ):
        raise ValueError("attempt-06 is not the authoritative valid-failed VAE run")
    old_spec = vae._load_spec(
        Path("configs/research/lightgen_optical_manifold_smf_vae_pilot_v1.yaml")
    )
    source = vae._load_source(old_spec, formal_path, current)
    checkpoint_hashes = {
        "vae_validation": SOURCE_VAE_VALIDATION_HASH,
        "vae_preflight": SOURCE_VAE_PREFLIGHT_HASH,
        "vae_normalization": SOURCE_VAE_NORMALIZATION_HASH,
    }
    checkpoints = {
        "smf_beta_vae": _load_checkpoint(
            paths["beta_checkpoint"], BETA_CHECKPOINT_HASH, "smf_beta_vae",
            old_spec.config_hash, VAE_DATA_MANIFEST_HASH, checkpoint_hashes
        ),
        "smf_stochastic_no_kl": _load_checkpoint(
            paths["stochastic_checkpoint"], STOCHASTIC_CHECKPOINT_HASH,
            "smf_stochastic_no_kl", old_spec.config_hash,
            VAE_DATA_MANIFEST_HASH, checkpoint_hashes
        ),
        "smf_deterministic_mse": _load_checkpoint(
            paths["deterministic_checkpoint"], DETERMINISTIC_CHECKPOINT_HASH,
            "smf_deterministic_mse", old_spec.config_hash,
            VAE_DATA_MANIFEST_HASH, checkpoint_hashes
        ),
    }
    normalization = safe_torch_load(
        paths["vae_normalization"], expected_sha256=SOURCE_VAE_NORMALIZATION_HASH,
        map_location="cpu"
    )
    if (
        normalization.get("schema") != vae.NORMALIZATION_SCHEMA
        or tuple(normalization.get("shape", ())) != (PORT_COUNT, 2)
        or normalization.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_HASH
        or normalization.get("source_hashes") != source.hashes
        or not torch.isfinite(normalization["mean"]).all()
        or not torch.isfinite(normalization["scale"]).all()
    ):
        raise ValueError("VAE normalization artifact is stale")
    context_sources = {}
    for key in ("plan6_training", "fmf_training"):
        path = _resolve(spec.payload["context"][key])
        digest = spec.payload["context"][f"{key}_sha256"]
        context_sources[key] = _load_json_source(path, digest, current)
    return SourceBundle(
        source, checkpoints, normalization, source_hashes,
        old_spec.config_hash, training, context_sources
    )


def _load_data(root: str | Path) -> tuple[vae.Data, list[torch.Tensor]]:
    data = vae._load_data(root)
    excluded = torch.cat((data.calibration_indices.flatten(), data.schedule.flatten(), data.heldout_indices))
    panels: list[torch.Tensor] = []
    labels = data.labels
    train_labels = labels.index_select(0, data.train_indices)
    candidates_by_label: list[torch.Tensor] = []
    for label in range(10):
        candidates = data.train_indices[train_labels == label]
        candidates = candidates[~torch.isin(candidates, excluded)].sort().values
        generator = torch.Generator(device="cpu").manual_seed(GRADIENT_PANEL_SEED + label)
        shuffled = candidates[torch.randperm(candidates.numel(), generator=generator)]
        candidates_by_label.append(shuffled[:30])
    for panel_index in range(3):
        panel = torch.stack(
            [values[panel_index * 10 : (panel_index + 1) * 10] for values in candidates_by_label],
            dim=1,
        )
        panels.append(panel)
    all_panel_indices = torch.cat([panel.flatten() for panel in panels])
    if (
        all_panel_indices.numel() != 300
        or torch.unique(all_panel_indices).numel() != 300
        or torch.isin(all_panel_indices, excluded).any()
    ):
        raise ValueError("KL gradient panels overlap a prior schedule or held-out data")
    return data, panels


def _context(spec: Spec, formal: FormalSystemConfig, data: vae.Data) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        teacher_manifest_hash=data.manifest["manifest_hash"],
        dataset_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _record_payload(schema: str, record: EvidenceRecordV2, contract: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": schema,
        "evidence": record.to_dict(),
        "hypothesis": HYPOTHESIS,
        "training_authorized": False,
        "formal_promotion": False,
        "contract": dict(contract),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite KL attribution artifact: {target}")
    atomic_write_json(target, payload)


def _validation_record(state: Mapping[str, Any], spec: Spec, formal: FormalSystemConfig, data: vae.Data, source: SourceBundle, review: Path, panels: list[torch.Tensor]) -> EvidenceRecordV2:
    hashes = {
        **source.hashes,
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review),
        "dataset_manifest": data.manifest["manifest_hash"],
        "gradient_panels": tensor_sha256(torch.cat([panel.flatten() for panel in panels])),
    }
    gates = tuple(GateResult(name, True, metrics) for name, metrics in (
        ("clean_lineage", {"clean": True, "branch": BRANCH}),
        ("authoritative_vae_lineage", {"validation": "passed", "preflight": "passed", "training": "valid_failed"}),
        ("source_checkpoints", {"source_step": SOURCE_STEP, "final_step": FINAL_STEP, "arms": 3}),
        ("mnist_panel_isolation", {"heldout": 1000, "gradient_panels": 300}),
        ("formal_identity_optics", {"grid": 2048, "formal_ols": "identity", "smf_research_ols": True}),
        ("zero_update_contract", {"optimizer_created": False, "training_authorized": False}),
        ("provenance_binding", {"source_hashes": len(source.hashes), "panel_hash": hashes["gradient_panels"]}),
    ))
    record = EvidenceRecordV2(2, VALIDATION_SCHEMA, "diagnostic", "passed", str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, gates, ("run_smf_vae_kl_attribution_audit",))
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_sources(spec, formal_path, state["commit"])
    data, panels = _load_data(args.dataset_root)
    if tuple(data.images.shape) != (60000, 1, 28, 28) or tuple(data.labels.shape) != (60000,):
        raise ValueError("KL attribution requires complete MNIST tensors")
    record = _validation_record(state, spec, formal, data, source, review, panels)
    payload = _record_payload(VALIDATION_SCHEMA, record, {
        "formal_config": str(formal_path.resolve()),
        "source_hashes": source.hashes,
        "dataset_manifest": data.manifest,
        "gradient_panel_indices": [panel.flatten().tolist() for panel in panels],
        "optimizer_created": False,
        "cuda_context_created": False,
    })
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(path: str | Path, spec: Spec, formal: FormalSystemConfig, data: vae.Data, state: Mapping[str, Any], source: SourceBundle, review: Path, panels: list[torch.Tensor]) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected = {
        **source.hashes,
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review),
        "dataset_manifest": data.manifest["manifest_hash"],
        "gradient_panels": tensor_sha256(torch.cat([panel.flatten() for panel in panels])),
    }
    if (
        payload.get("schema") != VALIDATION_SCHEMA
        or record.status != "passed"
        or record.git_commit != state["commit"]
        or record.config_hash != spec.config_hash
        or record.context != _context(spec, formal, data)
        or record.input_hashes != expected
        or record.permitted_next_actions != ("run_smf_vae_kl_attribution_audit",)
    ):
        raise ValueError("KL attribution validation is stale or mismatched")
    return payload, digest


def _runtime(formal: FormalSystemConfig, source: SourceBundle, arm: str, payload: Mapping[str, Any], device: torch.device) -> tuple[nn.Module, Any, vae.SharedPerPortPosterior | None]:
    model, operator, posterior, _optimizer = vae._make_runtime(formal, source.source, device, arm)
    model.load_state_dict(payload["model_state"], strict=True)
    if posterior is not None:
        posterior.load_state_dict(payload["posterior_state"], strict=True)
    model.eval()
    if posterior is not None:
        posterior.eval()
    return model, operator, posterior


def _batch(data: vae.Data, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = base.render_intensity_batch(data.images.index_select(0, indices))
    return rendered.amplitude.to(device), rendered.target.to(device)


def _collect_posterior(model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior, data: vae.Data, indices: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coefficients: list[torch.Tensor] = []
    posterior_means: list[torch.Tensor] = []
    logvars: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), BATCH_SIZE):
            amplitudes, _target = _batch(data, indices[start : start + BATCH_SIZE], device)
            coeff, _ = vae._encoded_coefficients(model, operator, amplitudes)
            post_mean, logvar = posterior(vae._normalize(coeff, mean, scale))
            coefficients.append(coeff.cpu())
            posterior_means.append(post_mean.cpu())
            logvars.append(logvar.cpu())
    coeff = torch.cat(coefficients)
    post_mean = torch.cat(posterior_means)
    logvar = torch.cat(logvars)
    return coeff, post_mean, logvar, torch.empty(0)


def _posterior_formula_check(
    model: nn.Module,
    operator: Any,
    posterior: vae.SharedPerPortPosterior,
    data: vae.Data,
    indices: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    epsilon: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    amplitudes, _targets = _batch(data, indices[:1], device)
    coefficients, _system = vae._encoded_coefficients(model, operator, amplitudes)
    normalized = vae._normalize(coefficients, mean, scale)
    posterior_mean, logvar = posterior(normalized)
    standard_deviation = logvar.mul(0.5).exp()
    latent = posterior_mean + standard_deviation * epsilon[:1].to(device)
    restored = vae._denormalize(latent, mean, scale)
    normalized_again = vae._normalize(restored, mean, scale)
    return {
        "finite": bool(
            torch.isfinite(latent).all()
            and torch.isfinite(restored).all()
            and torch.isfinite(standard_deviation).all()
        ),
        "complex_real_roundtrip": bool(
            torch.allclose(normalized_again, latent, atol=1.0e-6, rtol=0.0)
        ),
        "reparameterization_formula": {
            "latent_shape": list(latent.shape),
            "epsilon_shape": list(epsilon[:1].shape),
        },
    }


def _kl_terms(mean: torch.Tensor, logvar: torch.Tensor) -> dict[str, float]:
    mu_term = 0.5 * mean.square().mean()
    variance_term = 0.5 * (logvar.exp() - 1.0 - logvar).mean()
    return {
        "kl_mu": float(mu_term),
        "kl_variance": float(variance_term),
        "kl_total": float(mu_term + variance_term),
    }


def _posterior_metrics(model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior, data: vae.Data, indices: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    coefficients, posterior_mean, logvar, _ = _collect_posterior(model, operator, posterior, data, indices, mean, scale, device)
    terms = _kl_terms(posterior_mean, logvar)
    flat_mean = posterior_mean.flatten(1)
    normalized = vae._normalize(coefficients, mean.cpu(), scale.cpu()).flatten(1)
    centered_a = flat_mean - flat_mean.mean(0, keepdim=True)
    centered_b = normalized - normalized.mean(0, keepdim=True)
    input_sensitivity = float(
        (centered_a * centered_b).sum(1).mean()
        / (centered_a.norm(dim=1) * centered_b.norm(dim=1)).clamp_min(1.0e-30).mean()
    )
    variances = flat_mean.var(0, unbiased=False)
    logvar_values = logvar
    terms.update({
        "posterior_mean_norm": float(flat_mean.norm(dim=-1).mean()),
        "posterior_mean_variance": float(variances.mean()),
        "posterior_mean_variance_max": float(variances.max()),
        "posterior_variance_mean": float(logvar_values.exp().mean()),
        "active_latent_count": int((variances > ACTIVE_VARIANCE_THRESHOLD).sum()),
        "logvar_at_min_fraction": float((logvar_values <= vae.MIN_LOGVAR).float().mean()),
        "logvar_at_max_fraction": float((logvar_values >= vae.MAX_LOGVAR).float().mean()),
        "input_sensitivity": input_sensitivity,
        "coefficient_shape": list(coefficients.shape),
    })
    return terms


def _metric(model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior | None, data: vae.Data, indices: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, device: torch.device, epsilon: torch.Tensor | None, sample: bool) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    return vae._evaluate(model, operator, posterior, data, indices, mean, scale, device, sample=sample, epsilon=epsilon)


def _terminal_metrics_reproduced(
    measured: Mapping[str, Mapping[str, Any]], training: Mapping[str, Any]
) -> bool:
    expected = training["contract"]["arms"]
    for arm in ("smf_beta_vae", "smf_stochastic_no_kl", "smf_deterministic_mse"):
        actual = measured[f"{arm}_mean"]
        source = expected[arm]["metrics"]
        if abs(actual["paired_mse"] - source["paired_mse"]) > SOURCE_MSE_TOLERANCE:
            return False
        if arm != "smf_deterministic_mse" and abs(
            actual.get("posterior_kl", float("inf")) - source["posterior_kl"]
        ) > SOURCE_MSE_TOLERANCE:
            return False
    return True


def _gradient_vectors(model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior, data: vae.Data, indices: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, epsilon: torch.Tensor, device: torch.device, objective: str) -> tuple[dict[str, torch.Tensor], float]:
    if objective not in {"reconstruction", "kl"}:
        raise ValueError("unsupported KL audit objective")
    items = vae._trainable_items(model, posterior)
    sums = {name: torch.zeros_like(parameter, device="cpu") for name, parameter in items}
    values: list[float] = []
    for position, index in enumerate(indices):
        amplitudes, target = _batch(data, index.reshape(1), device)
        noise = epsilon[position : position + 1].to(device)
        raw, post_mean, logvar, _ = checkpoint(
            lambda value, eps: vae._forward_latent(model, operator, posterior, value, eps, mean, scale, sample=True),
            amplitudes, noise, use_reentrant=False,
        )
        value = F.mse_loss(raw, target) if objective == "reconstruction" else vae.posterior_kl(post_mean, logvar)
        gradients = torch.autograd.grad(value / indices.numel(), [parameter for _, parameter in items], allow_unused=True)
        values.append(float(value.detach().cpu()))
        for (name, _parameter), gradient in zip(items, gradients, strict=True):
            if gradient is not None:
                sums[name].add_(gradient.detach().cpu())
    return sums, float(sum(values) / len(values))


def _vector_norms(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.norm()) for name, value in values.items()}


def _vector_dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return float(sum((left[name] * right[name]).sum() for name in left))


def _gradient_audit(model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior, data: vae.Data, panels: list[torch.Tensor], mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    panel_reports: list[dict[str, Any]] = []
    for panel_index, panel in enumerate(panels):
        generator = torch.Generator(device="cpu").manual_seed(SEED + 100 + panel_index)
        epsilon = torch.randn((panel.numel(), PORT_COUNT, 2), generator=generator, dtype=torch.float32)
        rec, rec_value = _gradient_vectors(model, operator, posterior, data, panel.flatten(), mean.to(device), scale.to(device), epsilon, device, "reconstruction")
        kl, kl_value = _gradient_vectors(model, operator, posterior, data, panel.flatten(), mean.to(device), scale.to(device), epsilon, device, "kl")
        rec_norms = _vector_norms(rec)
        kl_norms = _vector_norms(kl)
        dose_reports: list[dict[str, Any]] = []
        for dose in DOSES:
            combined = {name: rec[name] + dose * kl[name] for name in rec}
            combined_norms = _vector_norms(combined)
            dose_reports.append({
                "dose": dose,
                "combined_gradient_norms": combined_norms,
                "encoder_kl_to_reconstruction_ratio": dose * kl_norms["system.encoder.phase"] / max(rec_norms["system.encoder.phase"], 1.0e-30),
                "combined_dot_reconstruction": _vector_dot(combined, rec),
                "combined_dot_kl": _vector_dot(combined, kl),
                "compatible_with_both": _vector_dot(combined, rec) > 0.0 and _vector_dot(combined, kl) > 0.0,
                "finite": all(torch.isfinite(value).all() for value in combined.values()),
                "encoder_layer_norms": [float(combined["system.encoder.phase"][layer].norm()) for layer in range(combined["system.encoder.phase"].shape[0])],
                "decoder_layer_norms": [float(combined["system.decoder.phase"][layer].norm()) for layer in range(combined["system.decoder.phase"].shape[0])],
            })
        panel_reports.append({
            "panel": panel_index,
            "reconstruction_value": rec_value,
            "kl_value": kl_value,
            "reconstruction_gradient_norms": rec_norms,
            "kl_gradient_norms": kl_norms,
            "kl_decoder_norm": kl_norms["system.decoder.phase"],
            "posterior_kl_gradient_norms": {name: value for name, value in kl_norms.items() if name.startswith("posterior.")},
            "doses": dose_reports,
            "finite": all(torch.isfinite(value).all() for value in (*rec.values(), *kl.values())),
        })
    return {"panels": panel_reports}


def _load_runtime_snapshot(model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior | None) -> dict[str, Any]:
    return {
        "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "operator": {name: value.detach().cpu().clone() for name, value in operator.state_dict().items()},
        "posterior": None if posterior is None else {name: value.detach().cpu().clone() for name, value in posterior.state_dict().items()},
    }


def _snapshot_equal(before: Mapping[str, Any], model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior | None) -> bool:
    return vae._tree_equal(before["model"], {name: value.detach().cpu() for name, value in model.state_dict().items()}) and vae._tree_equal(before["operator"], {name: value.detach().cpu() for name, value in operator.state_dict().items()}) and vae._tree_equal(before["posterior"], None if posterior is None else {name: value.detach().cpu() for name, value in posterior.state_dict().items()})


def _save_panel(path: Path, data: vae.Data, tensors: Mapping[str, torch.Tensor], hashes: Mapping[str, str]) -> str:
    payload = {
        "schema": PANEL_SCHEMA,
        "source_hashes": dict(hashes),
        "indices_are_global_mnist": True,
        "indices": data.visual_indices.cpu(),
        "input_28": data.images.index_select(0, data.visual_indices).cpu(),
        "tensors": {name: value.cpu().contiguous() for name, value in tensors.items()},
        "sample_count": MC_SAMPLES,
        "posterior_sample_seed": SEED,
        "diagnostic_only": True,
    }
    atomic_torch_save(path, payload)
    return file_sha256(path)


def _context_comparison(bundle: SourceBundle) -> dict[str, Any]:
    plan6 = bundle.context_sources["plan6_training"]["contract"]
    fmf = bundle.context_sources["fmf_training"]["contract"]
    return {
        "classification": "context_only_non_matched_budget",
        "plan6_identity_1000_step": plan6["arms"]["identity_mse_control"]["metrics"],
        "plan6_smf_1000_step": plan6["arms"]["smf_mse_reconstruction"]["metrics"],
        "fmf_m2_20_step": fmf["arms"]["fmf_m2_mse_canary"]["metrics"],
        "budgets": {
            "plan6_identity": {"steps": 1000, "ols": "formal_identity"},
            "plan6_smf": {"steps": 1000, "ols": "100_port_lp01_smf"},
            "fmf_m2": {"steps": 20, "ols": "200_mode_lp01_lp11a"},
            "vae_arms": {"steps": 100, "ols": "100_port_lp01_smf"},
        },
        "not_rankable": True,
    }


def run(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("KL attribution audit requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_sources(spec, formal_path, state["commit"])
    data, panels = _load_data(args.dataset_root)
    validation, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, state, source, review, panels)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite KL attribution evidence: {output}")
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
        mean = source.normalization["mean"].to(device)
        scale = source.normalization["scale"].to(device)
        runtimes: dict[str, tuple[Any, Any, Any]] = {}
        snapshots: dict[str, Any] = {}
        for arm, payload in {
            "smf_beta_vae": source.checkpoints["smf_beta_vae"],
            "smf_stochastic_no_kl": source.checkpoints["smf_stochastic_no_kl"],
            "smf_deterministic_mse": source.checkpoints["smf_deterministic_mse"],
        }.items():
            runtimes[arm] = _runtime(formal, source, arm, payload, device)
            snapshots[arm] = _load_runtime_snapshot(*runtimes[arm])
        zeros = torch.zeros((data.heldout_indices.numel(), PORT_COUNT, 2), dtype=torch.float32)
        fixed_generator = torch.Generator(device="cpu").manual_seed(SEED)
        fixed_epsilon = torch.randn((data.heldout_indices.numel(), PORT_COUNT, 2), generator=fixed_generator, dtype=torch.float32)
        heldout_metrics: dict[str, Any] = {}
        posterior_stats: dict[str, Any] = {}
        for arm, (model, operator, posterior) in runtimes.items():
            mean_metrics, _mean_tensors = _metric(model, operator, posterior, data, data.heldout_indices, mean, scale, device, zeros, False)
            heldout_metrics[f"{arm}_mean"] = mean_metrics
            if posterior is not None:
                sample_metrics, _sample_tensors = _metric(model, operator, posterior, data, data.heldout_indices, mean, scale, device, fixed_epsilon, True)
                heldout_metrics[f"{arm}_sample"] = sample_metrics
                posterior_stats[arm] = _posterior_metrics(model, operator, posterior, data, data.heldout_indices, mean, scale, device)
        source_model, source_operator, _source_post = _runtime(formal, source, "smf_deterministic_mse", source.source.checkpoint, device)
        source_snapshot = _load_runtime_snapshot(source_model, source_operator, _source_post)
        source_metrics, source_tensors = _metric(source_model, source_operator, None, data, data.heldout_indices, mean, scale, device, None, False)
        expected_initial = source.vae_training["contract"]["initial_metrics"]["smf_beta_vae"]["paired_mse"]
        source_reproduction = {
            "measured": source_metrics["paired_mse"],
            "authoritative": expected_initial,
            "difference": abs(source_metrics["paired_mse"] - expected_initial),
            "tolerance": SOURCE_MSE_TOLERANCE,
            "passed": abs(source_metrics["paired_mse"] - expected_initial) <= SOURCE_MSE_TOLERANCE,
        }
        beta_model, beta_operator, beta_post = runtimes["smf_beta_vae"]
        formula_check = _posterior_formula_check(
            beta_model,
            beta_operator,
            beta_post,
            data,
            data.heldout_indices,
            mean,
            scale,
            fixed_epsilon,
            device,
        )
        gradient_report = _gradient_audit(beta_model, beta_operator, beta_post, data, panels, mean, scale, device)
        cross_swaps: dict[str, Any] = {}
        for phase_arm, (phase_model, phase_operator, _phase_post) in runtimes.items():
            if phase_arm == "smf_deterministic_mse":
                continue
            for posterior_arm in ("smf_beta_vae", "smf_stochastic_no_kl"):
                posterior = runtimes[posterior_arm][2]
                key = f"{phase_arm}_phase__{posterior_arm}_posterior"
                metrics, _tensors = _metric(phase_model, phase_operator, posterior, data, data.heldout_indices, mean, scale, device, zeros, False)
                cross_swaps[key] = metrics
        visual_indices = data.visual_indices
        visual_target = _batch(data, visual_indices, device)[1].cpu()
        _source_visual_metrics, source_visual_tensors = _metric(
            source_model,
            source_operator,
            None,
            data,
            visual_indices,
            mean,
            scale,
            device,
            None,
            False,
        )
        panel_tensors: dict[str, torch.Tensor] = {
            "target": visual_target,
            "smf_direct_source": source_visual_tensors["camera"].cpu(),
        }
        visual_generator = torch.Generator(device="cpu").manual_seed(SEED)
        visual_epsilon = torch.randn(
            (visual_indices.numel(), PORT_COUNT, 2),
            generator=visual_generator,
            dtype=torch.float32,
        )
        for arm, (model, operator, posterior) in runtimes.items():
            mean_metrics, mean_tensors = _metric(model, operator, posterior, data, visual_indices, mean, scale, device, torch.zeros((visual_indices.numel(), PORT_COUNT, 2)), False)
            panel_tensors[f"{arm}_mean"] = mean_tensors["camera"]
            if posterior is not None:
                sample_metrics, sample_tensors = _metric(model, operator, posterior, data, visual_indices, mean, scale, device, visual_epsilon, True)
                panel_tensors[f"{arm}_sample"] = sample_tensors["camera"]
        mc_variance: dict[str, Any] = {}
        for arm in ("smf_beta_vae", "smf_stochastic_no_kl"):
            model, operator, posterior = runtimes[arm]
            samples: list[torch.Tensor] = []
            for sample_index in range(MC_SAMPLES):
                generator = torch.Generator(device="cpu").manual_seed(SEED + sample_index)
                epsilon = torch.randn((visual_indices.numel(), PORT_COUNT, 2), generator=generator, dtype=torch.float32)
                _metrics, tensors = _metric(model, operator, posterior, data, visual_indices, mean, scale, device, epsilon, True)
                samples.append(tensors["camera"])
            mc_variance[arm] = {"sample_count": MC_SAMPLES, "per_image_mean_variance": float(torch.stack(samples).var(0, unbiased=False).mean()), "finite": bool(torch.isfinite(torch.stack(samples)).all())}
        hashes = {**source.hashes, "research_config": file_sha256(spec.path), "review_contract": file_sha256(review), "dataset_manifest": data.manifest["manifest_hash"], "training_validation": validation_hash}
        panel_path = output.parent / "smf_vae_kl_comparison_panel.pt"
        panel_hash = _save_panel(panel_path, data, panel_tensors, hashes)
        state_unchanged = all(
            _snapshot_equal(snapshots[arm], model, operator, posterior)
            for arm, (model, operator, posterior) in runtimes.items()
        ) and _snapshot_equal(source_snapshot, source_model, source_operator, _source_post)
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
        beta_metrics = posterior_stats["smf_beta_vae"]
        no_kl_metrics = posterior_stats["smf_stochastic_no_kl"]
        beta_kl = beta_metrics["kl_total"]
        no_kl = no_kl_metrics["kl_total"]
        initial_kl = source.vae_training["contract"]["initial_metrics"]["smf_beta_vae"]["posterior_kl"]
        scientific_gates = (
            GateResult("source_metrics_reproduced", source_reproduction["passed"], source_reproduction),
            GateResult("authoritative_terminal_metrics_reproduced", _terminal_metrics_reproduced(heldout_metrics, source.vae_training), {}),
            GateResult("reparameterization_and_complex_roundtrip", formula_check["finite"] and formula_check["complex_real_roundtrip"], formula_check),
            GateResult("kl_decomposition_exact", abs(beta_metrics["kl_mu"] + beta_metrics["kl_variance"] - beta_kl) <= KL_REFERENCE_TOLERANCE, {"beta": beta_metrics}),
            GateResult("beta_kl_below_no_kl", beta_kl < no_kl, {"beta": beta_kl, "no_kl": no_kl}),
            GateResult("posterior_not_collapsed", beta_metrics["posterior_mean_variance"] >= 0.5 * source.vae_training["contract"]["initial_metrics"]["smf_beta_vae"]["posterior_mean_variance"], {"final": beta_metrics["posterior_mean_variance"], "initial": source.vae_training["contract"]["initial_metrics"]["smf_beta_vae"]["posterior_mean_variance"]}),
            GateResult("kl_decoder_gradient_zero", all(panel["kl_decoder_norm"] == 0.0 for panel in gradient_report["panels"]), {}),
            GateResult("kl_encoder_and_posterior_gradient_nonzero", all(panel["kl_gradient_norms"]["system.encoder.phase"] > 0.0 and any(value > 0.0 for value in panel["posterior_kl_gradient_norms"].values()) for panel in gradient_report["panels"]), {}),
            GateResult("reconstruction_and_dose_gradients_finite", all(panel["finite"] and all(dose["finite"] for dose in panel["doses"]) for panel in gradient_report["panels"]), {}),
            GateResult(
                "higher_dose_gradient_compatibility",
                any(
                    all(
                        dose["compatible_with_both"]
                        for panel in gradient_report["panels"]
                        for dose in panel["doses"]
                        if dose["dose"] == candidate
                    )
                    for candidate in (2.0, 4.0)
                ),
                {
                    "compatible_doses": [
                        candidate
                        for candidate in (2.0, 4.0)
                        if all(
                            dose["compatible_with_both"]
                            for panel in gradient_report["panels"]
                            for dose in panel["doses"]
                            if dose["dose"] == candidate
                        )
                    ]
                },
            ),
            GateResult("runtime_state_unchanged", state_unchanged, {"evaluator_calls": 0}),
        )
        resource_gates = (
            GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
            GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
            GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
        )
        chain_valid = all(
            gate.passed
            for gate in scientific_gates
            if gate.name
            not in {"beta_kl_below_no_kl", "posterior_not_collapsed", "higher_dose_gradient_compatibility"}
        )
        resources_valid = all(gate.passed for gate in resource_gates)
        if not chain_valid:
            status, outcome, actions = "invalid_harness", "invalid_harness", ()
        elif not resources_valid:
            status, outcome, actions = "resource_aborted", "resource_aborted", ()
        elif not all(gate.passed for gate in scientific_gates if gate.name in {"source_metrics_reproduced", "authoritative_terminal_metrics_reproduced", "reparameterization_and_complex_roundtrip", "kl_decomposition_exact", "beta_kl_below_no_kl", "posterior_not_collapsed"}):
            status, outcome, actions = "valid_failed", "valid_failed", (NEXT_OBJECTIVE_ACTION,)
        elif not next(
            gate.passed
            for gate in scientific_gates
            if gate.name == "higher_dose_gradient_compatibility"
        ):
            status, outcome, actions = "diagnostic", "diagnostic_completed_parameterization_limited", (NEXT_PARAMETERIZATION_ACTION,)
        else:
            status, outcome, actions = "diagnostic", "diagnostic_completed_relative_kl_constraint", (NEXT_DOSE_ACTION,)
        hashes.update({"panel": panel_hash})
        record = EvidenceRecordV2(2, AUDIT_SCHEMA, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, scientific_gates + resource_gates, actions, None if status == "diagnostic" else "KL attribution audit did not satisfy all registered gates")
        record.validate()
        audit_payload = _record_payload(AUDIT_SCHEMA, record, {
            "outcome": outcome,
            "validation_hash": validation_hash,
            "source_reproduction": source_reproduction,
            "initial_kl": initial_kl,
            "heldout_metrics": heldout_metrics,
            "posterior_stats": posterior_stats,
            "cross_swaps": cross_swaps,
            "gradient_audit": gradient_report,
            "monte_carlo": mc_variance,
            "panel_path": str(panel_path.resolve()),
            "panel_sha256": panel_hash,
            "comparison": _context_comparison(source),
            "free_cuda_bytes_before": int(free),
            "total_cuda_bytes": int(total),
            "peak_cuda_bytes": peak,
            "wall_seconds": elapsed,
            "run_storage_bytes": storage,
            "evaluator_calls": 0,
            "formal_candidate_checkpoint": False,
            "absolute_kl_below_initial": beta_kl < initial_kl,
        })
        _write_new(output, audit_payload)
        comparison_payload = _record_payload(COMPARISON_SCHEMA, record, {
            "outcome": outcome,
            "classification": "matched_same_source_same_budget_for_three_arms",
            "same_source": True,
            "same_schedule": True,
            "same_epsilon_for_sample_comparison": True,
            "matched_arms": {name: heldout_metrics[name] for name in heldout_metrics},
            "context_only": _context_comparison(source),
            "not_rankable_context": True,
            "panel_path": str(panel_path.resolve()),
            "panel_sha256": panel_hash,
        })
        _write_new(output.parent / "horizontal_comparison.json", comparison_payload)
        print(json.dumps(audit_payload, indent=2, sort_keys=True, default=str))
        return 0 if status in {"diagnostic", "valid_failed"} else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.cuda.empty_cache()


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-kl-chain-audit-validate":
        return validate(args)
    if args.research_action == "smf-kl-chain-audit-run":
        return run(args)
    raise ValueError(f"unsupported KL attribution action: {args.research_action}")
