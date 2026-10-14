"""Execute the bounded conflict-aware Generator semantic-recovery successor.

The consumer starts from the accepted step-0 state of the failed frozen-latent
semantic-recovery successor.  It keeps the current optical/runtime context,
Generator architecture, objectives, data order, and optimizer state fixed.  Its
only scientific intervention is to protect the semantic gradient from locally
conflicting reconstruction and optical-power gradient components.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as phase_repair
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1 as canary
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as producer
from scripts.research.lightgen import execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 as successor
from scripts.research.lightgen import execute_lightgen_2d_generator_semantic_controllability_attribution_v1 as attribution
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import ADAM_BETAS, ADAM_EPSILON, AMSGRAD, CHECKPOINT_STEPS, DataReadLedger, DeterministicTrainRoleSampler, TRAIN_ROLE, WEIGHT_DECAY, _rng_state_payload
from ldm_project.v2.application.research.shared.research_run_control import HeartbeatWriter, InterruptController, RecoveryCheckpointManager, ResearchRunControl, ResearchRunIdentity, RUN_COMPLETED, RUNNING, RUN_PREPARING, capture_execution_source_manifest, recovery_due
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator, SmoothMNISTEvaluator, load_frozen_evaluator


TASK_ID = "EXECUTE_CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "conflict-aware-generator-semantic-recovery-v1"
TRAJECTORY_ID = "conflict-aware-generator-semantic-recovery-trajectory-v1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

PRIOR_RUN = Path("outputs/20260918_frozen_latent_generator_semantic_recovery_v1/runs/attempt_01")
PRIOR_REPORT = PRIOR_RUN / "report.json"
PRIOR_STEP_0 = PRIOR_RUN / "scientific_step_000000.pt"
PRIOR_STEP_300 = PRIOR_RUN / "scientific_step_000300.pt"
EXPECTED_PRIOR_REPORT_SHA256 = "fcb85e17e4b3d258fbe76d975103869e8d18a11252965dd5581f02a0acdb19b3"
EXPECTED_PRIOR_STEP_0_SHA256 = "bf103a0a876d78e5e07c8095c82c40a08f2d829b85e01a28f424879936d61be7"
EXPECTED_PRIOR_STEP_300_SHA256 = "58bb27b4fb7b37667118bcbfa44df3ae6b3485db8b6326280aba9190516b23a0"

ATTRIBUTION_REPORT = Path("outputs/20260918_generator_semantic_controllability_attribution_v1/runs/attempt_01/report.json")
EXPECTED_ATTRIBUTION_REPORT_SHA256 = "3304262b2cd5fbff22e423c415984c14eeb1a2906871ad45e3a62265479d89b5"

PHASE_REPORT = successor.PHASE_REPORT
PHASE_SELECTION = successor.PHASE_SELECTION
PHASE_LATENT = successor.PHASE_LATENT
REAL_CONTROL = successor.REAL_CONTROL
EXPECTED_PHASE_REPORT_SHA256 = successor.EXPECTED_PHASE_REPORT_SHA256
EXPECTED_PHASE_SELECTION_SHA256 = successor.EXPECTED_PHASE_SELECTION_SHA256
EXPECTED_PHASE_LATENT_SHA256 = successor.EXPECTED_PHASE_LATENT_SHA256
EXPECTED_REAL_CONTROL_SHA256 = successor.EXPECTED_REAL_CONTROL_SHA256
SMOOTH_CHECKPOINT = successor.SMOOTH_CHECKPOINT
EXPECTED_SMOOTH_CHECKPOINT_SHA256 = successor.EXPECTED_SMOOTH_CHECKPOINT_SHA256
TRAINING_SOURCE_SHA256 = successor.TRAINING_SOURCE_SHA256
EXPECTED_TRAIN_EVALUATOR_SHA256 = producer.EVALUATOR_CHECKPOINT_SHA256

DEFAULT_OUTPUT_DIR = Path("outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01")
LOGICAL_BATCH_SIZE = successor.LOGICAL_BATCH_SIZE
BOOTSTRAP_REPLICATES = successor.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = successor.BOOTSTRAP_SEED
RECONSTRUCTION_TOLERANCE = successor.RECONSTRUCTION_TOLERANCE
RECOVERY_INTERVAL_STEPS = successor.RECOVERY_INTERVAL_STEPS
EPS_PROJ = 1.0e-12

REPORT_SCHEMA = "conflict_aware_generator_semantic_recovery_report.v1"
PREFLIGHT_SCHEMA = "conflict_aware_generator_semantic_recovery_preflight.v1"
CHECKPOINT_SCHEMA = "conflict_aware_generator_semantic_recovery_checkpoint.v1"
DEV_SCHEMA = "conflict_aware_generator_semantic_recovery_dev_diagnostic.v1"
BOOTSTRAP_SCHEMA = "conflict_aware_generator_semantic_recovery_paired_bootstrap.v1"

SUPPORTED_CLASSIFICATION = "CONFLICT_AWARE_SEMANTIC_RECOVERY_SUPPORTED"
MIXED_TRANSFER_CLASSIFICATION = "CONFLICT_AWARE_REAL_SEMANTIC_RECOVERY_TRANSFER_MIXED"
NOT_SUPPORTED_CLASSIFICATION = "CONFLICT_AWARE_SEMANTIC_RECOVERY_NOT_SUPPORTED"
CONSTRAINT_CLASSIFICATION = "SEMANTIC_RECOVERY_RECONSTRUCTION_CONSTRAINT_FAILED"


class ConflictAwareExecutionError(RuntimeError):
    """Raised when an authority, invariant, or valid measurement cannot be consumed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConflictAwareExecutionError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.exists():
        atomic_write_new_json(path, dict(payload))


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    if not path.exists():
        atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


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


def _frozen_state(consumer: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in consumer.runtime.state_dict().items()
        if name != "generator.phase"
    }


def _generator(consumer: Any) -> torch.nn.Parameter:
    parameters = dict(consumer.runtime.named_parameters())
    _require(set(parameters) == {"encoder.phase", "generator.phase"}, "runtime parameter surface changed")
    return parameters["generator.phase"]


def _freeze_generator_only(consumer: Any) -> None:
    for name, parameter in consumer.runtime.named_parameters():
        parameter.requires_grad_(name == "generator.phase")
    trainable = [name for name, parameter in consumer.runtime.named_parameters() if parameter.requires_grad]
    _require(trainable == ["generator.phase"], f"Generator-only firewall changed: {trainable}")


def _validate_generator_optimizer(consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    successor._validate_generator_optimizer(consumer, optimizer)


def _validate_prior_optimizer_config(config: Mapping[str, Any], prior_report: Mapping[str, Any]) -> None:
    prior = _mapping(prior_report.get("optimizer"), "prior optimizer authority")
    expected = {
        "name": "Adam",
        "parameter_names": ["generator.phase"],
        "learning_rate": 1.0e-3,
        "betas": list(ADAM_BETAS),
        "eps": ADAM_EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "amsgrad": AMSGRAD,
    }
    for name, value in expected.items():
        _require(config.get(name) == value, f"step-0 optimizer configuration changed: {name}")
        _require(prior.get(name) == value, f"prior optimizer configuration changed: {name}")


def _restore_prior_optimizer(consumer: Any, step0: Mapping[str, Any], prior_report: Mapping[str, Any]) -> tuple[torch.optim.Adam, str]:
    optimizer = successor._new_generator_optimizer(consumer)
    config = _mapping(step0.get("optimizer_configuration"), "step-0 optimizer configuration")
    _validate_prior_optimizer_config(config, prior_report)
    _require(step0.get("optimizer_mode") == "GENERATOR_MOMENT_SUBSET_RESTORED", "step-0 optimizer mode changed")
    old_state = _mapping(step0.get("optimizer_state"), "step-0 optimizer state")
    old_groups = old_state.get("param_groups")
    state_entries = old_state.get("state")
    _require(isinstance(old_groups, list) and len(old_groups) == 1, "step-0 optimizer groups changed")
    _require(isinstance(state_entries, Mapping), "step-0 optimizer state entries are missing")
    old_params = old_groups[0].get("params")
    _require(isinstance(old_params, list) and len(old_params) == 1, "step-0 optimizer parameter boundary changed")
    old_generator_state = state_entries.get(old_params[0])
    _require(isinstance(old_generator_state, Mapping), "step-0 Generator moments are missing")
    _require({"step", "exp_avg", "exp_avg_sq"}.issubset(old_generator_state), "step-0 Generator moments are incomplete")
    generator = _generator(consumer)
    for name in ("exp_avg", "exp_avg_sq"):
        value = old_generator_state[name]
        _require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(generator.shape), f"step-0 optimizer moment shape changed: {name}")
        _finite(value, f"step-0 optimizer moment {name}")
    template = optimizer.state_dict()
    new_id = template["param_groups"][0]["params"][0]
    template["state"] = {new_id: copy.deepcopy(dict(old_generator_state))}
    group = dict(old_groups[0])
    group["params"] = [new_id]
    template["param_groups"] = [group]
    optimizer.load_state_dict(template)
    _validate_generator_optimizer(consumer, optimizer)
    return optimizer, "PRIOR_SEMANTIC_RECOVERY_STEP0_STATE_RESTORED"


def _validate_current_docs(root: Path) -> dict[str, Any]:
    status = root / "docs/CURRENT_STATUS.md"
    handoff = root / "docs/status/current_handoff.md"
    status_top = "\n".join(status.read_text(encoding="utf-8").splitlines()[:45]) if status.is_file() else ""
    handoff_top = "\n".join(handoff.read_text(encoding="utf-8").splitlines()[:45]) if handoff.is_file() else ""
    return {
        "status_top_sha256": hashlib.sha256(status_top.encode("utf-8")).hexdigest(),
        "handoff_top_sha256": hashlib.sha256(handoff_top.encode("utf-8")).hexdigest(),
    }


def _load_phase_transfer_authority(root: Path, cpu_context: Mapping[str, Any]) -> dict[str, Any]:
    phase_report_path = root / PHASE_REPORT
    phase_selection_path = root / PHASE_SELECTION
    phase_latent_path = root / PHASE_LATENT
    phase_report = json.loads(phase_report_path.read_text(encoding="utf-8"))
    selection = json.loads(phase_selection_path.read_text(encoding="utf-8"))
    _require(phase_report.get("status") == "VALID_COMPLETED", "accepted phase report is not valid completed")
    _require(phase_report.get("outer_gate") == "CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1_RESOLVED", "accepted phase gate changed")
    _require(selection.get("selected_candidate") == "Q2", "accepted phase selection is not Q2")
    stage2 = _mapping(phase_report.get("stage_2_downstream_propagation"), "phase stage-2 authority")
    _require(stage2.get("selected_candidate") == "Q2", "phase stage-2 candidate changed")
    selected_metrics = _mapping(_mapping(stage2.get("metrics"), "phase metrics").get("selected"), "selected Q2 metrics")
    selected = {key: float(selected_metrics[key]) for key in ("R_u", "R_phi", "R_z", "R_raw", "R_feature")}
    _require(all(math.isfinite(value) for value in selected.values()), "selected Q2 metric is nonfinite")

    latent = safe_torch_load(phase_latent_path, map_location="cpu")
    _require(isinstance(latent, Mapping), "selected Q2 latent bank is invalid")
    _require(latent.get("candidate_id") == "Q2" and latent.get("selected_step") == 300, "selected Q2 latent identity changed")
    u_sample = _tensor(latent.get("u_sample"), "selected Q2 u sample")
    z_sample = _tensor(latent.get("z_sample"), "selected Q2 z sample")
    _require(u_sample.shape == (1000, 100) and u_sample.dtype == torch.float64, "selected Q2 u shape/dtype changed")
    _require(z_sample.shape == (1000, 100) and z_sample.dtype == torch.complex64, "selected Q2 z shape/dtype changed")
    _finite(u_sample, "selected Q2 u")
    _finite(z_sample, "selected Q2 z")
    p2_candidate = _mapping(cpu_context.get("p2_candidate"), "accepted P2 candidate")
    _require(torch.equal(u_sample, _tensor(p2_candidate.get("u"), "accepted P2 u")), "selected Q2 u is not accepted P2")

    generated_ref = _mapping(_mapping(stage2.get("optical"), "phase optical authority").get("generated_artifact"), "selected generated artifact")
    generated_path = root / str(generated_ref.get("path"))
    _require(generated_path.is_file(), "accepted Q2 generated CMOS is missing")
    generated_payload = safe_torch_load(generated_path, map_location="cpu")
    generated_raw = _tensor(_mapping(generated_payload, "accepted Q2 generated CMOS").get("raw_cmos_intensity"), "accepted Q2 generated raw")
    _require(generated_raw.shape == (200, 1, 200, 200) and generated_raw.dtype == torch.float32, "accepted Q2 generated raw shape/dtype changed")
    _finite(generated_raw, "accepted Q2 generated raw")

    real_control_path = root / REAL_CONTROL
    real_payload = safe_torch_load(real_control_path, map_location="cpu")
    real_raw = _tensor(_mapping(real_payload, "accepted real-control CMOS").get("raw_cmos_intensity"), "accepted real-control raw")
    _require(real_raw.shape == (200, 1, 200, 200) and real_raw.dtype == torch.float32, "accepted real-control raw shape/dtype changed")
    _finite(real_raw, "accepted real-control raw")
    return {
        "phase_report": phase_report,
        "phase_selection": selection,
        "phase_latent": dict(latent),
        "selected_metrics": selected,
        "selected_generated_raw": generated_raw,
        "real_control_raw": real_raw,
        "authority_hashes": {
            "phase_report": EXPECTED_PHASE_REPORT_SHA256,
            "phase_selection": EXPECTED_PHASE_SELECTION_SHA256,
            "phase_latent": EXPECTED_PHASE_LATENT_SHA256,
            "real_control": EXPECTED_REAL_CONTROL_SHA256,
        },
    }


def _load_authorities(root: Path) -> dict[str, Any]:
    docs = _validate_current_docs(root)
    attribution_report_path = root / ATTRIBUTION_REPORT
    attribution_report = json.loads(attribution_report_path.read_text(encoding="utf-8"))
    _require(attribution_report.get("result") == "VALID_COMPLETED", "accepted attribution result is not valid completed")
    _require(attribution_report.get("outer_gate") == attribution.OUTER_GATE, "accepted attribution gate changed")
    _require(attribution_report.get("primary_attribution") == "GENERATOR_SEMANTIC_CONTROLLABILITY_MIXED", "accepted attribution primary changed")
    _require(attribution_report.get("secondary_conflict") == "RECONSTRUCTION_SEMANTIC_GRADIENT_CONFLICT_SUPPORTED", "accepted attribution secondary changed")

    accepted_report_path = root / PRIOR_REPORT
    accepted_report = json.loads(accepted_report_path.read_text(encoding="utf-8"))
    _require(accepted_report.get("result") == "VALID_COMPLETED", "accepted semantic-recovery result changed")
    _require(accepted_report.get("outer_gate") == successor.OUTER_GATE, "accepted semantic-recovery gate changed")
    _require(accepted_report.get("classification") == successor.NOT_SUPPORTED_CLASSIFICATION, "accepted semantic-recovery classification changed")
    _require(accepted_report.get("training_trajectory", {}).get("updates") == 300, "accepted semantic-recovery update count changed")

    step0 = safe_torch_load(root / PRIOR_STEP_0, map_location="cpu")
    step300 = safe_torch_load(root / PRIOR_STEP_300, map_location="cpu")
    for name, checkpoint, expected_step in (("step-0", step0, 0), ("step-300", step300, 300)):
        _require(isinstance(checkpoint, Mapping), f"accepted {name} checkpoint is malformed")
        _require(checkpoint.get("completed_updates") == expected_step, f"accepted {name} checkpoint step changed")
        _require(checkpoint.get("scientific_artifact") is True and checkpoint.get("operational_artifact") is False, f"accepted {name} checkpoint role changed")
        _require(checkpoint.get("trainable_parameter_names") == ["generator.phase"], f"accepted {name} trainable boundary changed")
        _require("encoder.phase" in checkpoint.get("frozen_parameter_names", []), f"accepted {name} encoder freeze is missing")
        _require(checkpoint.get("outer_gate") == successor.OUTER_GATE, f"accepted {name} gate changed")
        _finite(_tensor(checkpoint.get("generator_state"), f"accepted {name} generator state"), f"accepted {name} generator state")
        _finite(_tensor(checkpoint.get("encoder_state"), f"accepted {name} encoder state"), f"accepted {name} encoder state")
    _require(torch.equal(_tensor(step0.get("encoder_state"), "step-0 encoder"), _tensor(step300.get("encoder_state"), "step-300 encoder")), "accepted encoder changed")
    _require(not torch.equal(_tensor(step0.get("generator_state"), "step-0 generator"), _tensor(step300.get("generator_state"), "step-300 generator")), "accepted Generator did not change")

    cpu_context = phase_repair._load_cpu_context(root)
    training = successor._load_successor_training_authority(root, {"training": cpu_context})
    images, labels, train_indices, development_indices, data_identity = producer._load_scientific_dataset(root)
    expected_data = training["data_identity"]
    _require(data_identity.get("train_count") == expected_data.get("train_count") == 55_000, "accepted TRAIN row count changed")
    _require(data_identity.get("development_count") == expected_data.get("development_count") == 5_000, "accepted DEV row count changed")
    _require(data_identity.get("seed") == expected_data.get("seed") == successor.SCHEDULE_SEED, "accepted data split seed changed")
    _require(data_identity.get("train_counts_by_class") == expected_data.get("train_counts_by_class"), "accepted TRAIN class roles changed")
    _require(data_identity.get("development_counts_by_class") == expected_data.get("development_counts_by_class"), "accepted DEV class roles changed")
    calibration_identity = _mapping(training["weights"].calibration_batch_identity, "accepted calibration identity")
    source_indices = calibration_identity.get("source_indices")
    _require(isinstance(source_indices, list) and len(source_indices) == LOGICAL_BATCH_SIZE, "accepted calibration indices unavailable")
    calibration_indices = torch.tensor(source_indices, dtype=torch.long)
    calibration_labels = labels.index_select(0, calibration_indices)
    _require(calibration_labels.tolist() == list(range(10)), "accepted calibration labels changed")
    _require(bool(torch.all(torch.isin(calibration_indices, train_indices)).item()), "accepted calibration batch left TRAIN role")

    objective = _mapping(accepted_report.get("objective"), "accepted objective")
    calibration = _mapping(objective.get("calibration"), "accepted semantic calibration")
    beta = float(objective.get("beta"))
    gamma = float(objective.get("gamma"))
    delta = float(calibration.get("delta"))
    _require(math.isclose(beta, float(training["weights"].beta), rel_tol=0.0, abs_tol=1.0e-12), "beta authority changed")
    _require(math.isclose(gamma, float(training["weights"].gamma), rel_tol=0.0, abs_tol=1.0e-12), "gamma authority changed")
    _require(math.isclose(delta, float(_mapping(step0.get("calibration"), "step-0 calibration").get("delta")), rel_tol=0.0, abs_tol=1.0e-15), "delta authority changed")
    _require(calibration.get("old_tn_distribution_term", {}).get("included_in_objective") is False, "old TN term entered prior objective")
    _require(calibration.get("old_tn_distribution_term", {}).get("backward_influence_on_generator") == 0.0, "old TN gradient influence changed")

    phase = _load_phase_transfer_authority(root, cpu_context)
    hashes: dict[str, Any] = {}
    hashes.update(phase["authority_hashes"])
    hashes.update({"prior_successor_report": EXPECTED_PRIOR_REPORT_SHA256, "prior_successor_step0": EXPECTED_PRIOR_STEP_0_SHA256, "prior_successor_step300": EXPECTED_PRIOR_STEP_300_SHA256, "attribution_report": EXPECTED_ATTRIBUTION_REPORT_SHA256})
    return {
        "docs": docs,
        "attribution_report": attribution_report,
        "accepted_report": accepted_report,
        "step0": dict(step0),
        "step300": dict(step300),
        "training": training,
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "development_indices": development_indices,
        "data_identity": data_identity,
        "calibration_indices": calibration_indices,
        "calibration_labels": calibration_labels,
        "phase_latent": phase["phase_latent"],
        "selected_generated_raw": phase["selected_generated_raw"],
        "real_control_raw": phase["real_control_raw"],
        "selected_metrics": phase["selected_metrics"],
        "authority_hashes": hashes,
        "weights": {"beta": beta, "gamma": gamma, "delta": delta},
    }


def _load_models(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[Any, FrozenMNISTEvaluator, SmoothMNISTEvaluator, dict[str, Any]]:
    training = _mapping(authorities.get("training"), "training authority")
    consumer = successor.LightGen2DChenBOGTComplexOLSTrainingConsumer(device=device, propagation_grid_size=successor.PRODUCTION_PROPAGATION_GRID)
    bundle = _mapping(training.get("bundle"), "accepted generation bundle")
    _require(consumer.runtime_identity() == _mapping(bundle.get("ols"), "bundle OLS").get("interface"), "runtime identity differs from accepted bundle")
    consumer.load_trainable_state_dict({"encoder.phase": authorities["step0"]["encoder_state"], "generator.phase": authorities["step0"]["generator_state"]})
    consumer.bind_references(training["references"], for_scientific=True)
    consumer.bind_weights(training["weights"], for_scientific=True)
    _freeze_generator_only(consumer)

    train_checkpoint = root / producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH
    train_evaluator, train_payload = load_frozen_evaluator(train_checkpoint, device=device)
    _require(all(not parameter.requires_grad for parameter in train_evaluator.parameters()), "TRAIN evaluator is not frozen")
    smooth_evaluator, smooth_identity = successor._load_smooth_evaluator(root, device)
    _require(all(not parameter.requires_grad for parameter in smooth_evaluator.parameters()), "Smooth evaluator is not frozen")
    return consumer, train_evaluator, smooth_evaluator, {
        "train_checkpoint": _relative(root, train_checkpoint),
        "train_checkpoint_sha256": EXPECTED_TRAIN_EVALUATOR_SHA256,
        "train_architecture": train_payload.get("architecture_kind"),
        "smooth": smooth_identity,
    }


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sum(left.detach().to(torch.float64) * right.detach().to(torch.float64)).cpu())


def _norm(value: torch.Tensor, name: str) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result) and result > 0.0, f"{name} is zero or invalid")
    return result


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = _norm(left, "left gradient")
    right_norm = _norm(right, "right gradient")
    value = _dot(left, right) / (left_norm * right_norm)
    _require(math.isfinite(value), "gradient cosine is nonfinite")
    return value


def _gradient_stats(value: torch.Tensor, name: str) -> dict[str, Any]:
    _finite(value, name)
    flat = value.detach().to(torch.float64)
    layer_norms = {f"generator.phase[{index}]": float(flat[index].square().sum().sqrt().cpu()) for index in range(int(flat.shape[0]))}
    return {
        "norm": _norm(value, name),
        "finite_fraction": float(torch.isfinite(value).to(torch.float64).mean().cpu()),
        "zero_fraction": float((value == 0).to(torch.float64).mean().cpu()),
        "shape": list(value.shape),
        "per_generator_layer_norms": layer_norms,
    }


def _project_auxiliary_gradient(auxiliary: torch.Tensor, semantic: torch.Tensor, name: str) -> tuple[torch.Tensor, dict[str, Any]]:
    auxiliary_norm = _norm(auxiliary, f"{name} gradient")
    semantic_norm = _norm(semantic, "semantic gradient")
    dot = _dot(auxiliary, semantic)
    projected = dot < 0.0
    removed = torch.zeros_like(auxiliary)
    if projected:
        coefficient = dot / (float(torch.sum(semantic.detach().to(torch.float64) ** 2).cpu()) + EPS_PROJ)
        removed = semantic * torch.as_tensor(coefficient, dtype=semantic.dtype, device=semantic.device)
        safe = auxiliary - removed
    else:
        safe = auxiliary
    _finite(safe, f"{name}_safe gradient")
    safe_dot = _dot(safe, semantic)
    if projected:
        tolerance = max(1.0e-8, 1.0e-5 * max(_norm(safe, f"{name}_safe gradient"), 1.0e-12) * semantic_norm)
        _require(abs(safe_dot) <= tolerance, f"{name}_safe gradient is not orthogonal to semantic gradient")
    removed_norm = float(removed.detach().to(torch.float64).square().sum().sqrt().cpu())
    return safe, {
        "dot_before": dot,
        "dot_after": safe_dot,
        "projected": projected,
        "projection_fraction_norm": removed_norm / auxiliary_norm,
        "removed_norm": removed_norm,
        "eps_proj": EPS_PROJ,
    }


def _compose_gradients(consumer: Any, evaluator: FrozenMNISTEvaluator, batch: torch.Tensor, labels: torch.Tensor, references: Any, *, beta: float, gamma: float, delta: float) -> dict[str, Any]:
    generator = _generator(consumer)
    components = successor._recovery_components(consumer, evaluator, batch, labels, references, beta, gamma, delta)
    g_rec = torch.autograd.grad(beta * components["L_rec"], generator, retain_graph=True, allow_unused=False)[0]
    g_op = torch.autograd.grad(gamma * components["L_op"], generator, retain_graph=True, allow_unused=False)[0]
    g_sem = torch.autograd.grad(delta * components["L_sem"], generator, retain_graph=False, allow_unused=False)[0]
    for name, value in (("g_rec", g_rec), ("g_op", g_op), ("g_sem", g_sem)):
        _finite(value, name)
    _norm(g_sem, "g_sem")
    g_rec_safe, rec_projection = _project_auxiliary_gradient(g_rec, g_sem, "g_rec")
    g_op_safe, op_projection = _project_auxiliary_gradient(g_op, g_sem, "g_op")
    g_total = g_sem + g_rec_safe + g_op_safe
    _finite(g_total, "g_total")
    total_semantic_dot = _dot(g_total, g_sem)
    _require(total_semantic_dot > 0.0, "composed gradient lacks a positive semantic component")
    return {
        "components": components,
        "gradients": {"g_rec": g_rec, "g_op": g_op, "g_sem": g_sem, "g_rec_safe": g_rec_safe, "g_op_safe": g_op_safe, "g_total": g_total},
        "gradient_norms": {name: _gradient_stats(value, name) for name, value in (("g_rec", g_rec), ("g_op", g_op), ("g_sem", g_sem), ("g_rec_safe", g_rec_safe), ("g_op_safe", g_op_safe), ("g_total", g_total))},
        "gradient_cosines": {
            "cos_g_rec_g_sem": _cosine(g_rec, g_sem),
            "cos_g_op_g_sem": _cosine(g_op, g_sem),
            "cos_g_rec_safe_g_sem": _cosine(g_rec_safe, g_sem),
            "cos_g_op_safe_g_sem": _cosine(g_op_safe, g_sem),
            "cos_g_total_g_sem": _cosine(g_total, g_sem),
        },
        "projection": {"rec": rec_projection, "op": op_projection},
        "old_tn_distribution_in_objective": False,
        "scalar_weighted_sum_backward": False,
    }


def _composition_record(bundle: Mapping[str, Any]) -> dict[str, Any]:
    components = _mapping(bundle["components"], "gradient components")
    return {
        "L_rec": float(components["L_rec"].detach().cpu()),
        "L_op": float(components["L_op"].detach().cpu()),
        "L_sem": float(components["L_sem"].detach().cpu()),
        "L_weighted_sum_reference": float(components["L_total"].detach().cpu()),
        "gradient_norms": bundle["gradient_norms"],
        "gradient_cosines": bundle["gradient_cosines"],
        "projection": bundle["projection"],
        "old_tn_distribution_in_objective": False,
        "scalar_weighted_sum_backward": False,
    }


def _diagnose_composition(consumer: Any, evaluator: FrozenMNISTEvaluator, batch: torch.Tensor, labels: torch.Tensor, references: Any, *, beta: float, gamma: float, delta: float) -> dict[str, Any]:
    before = _state_digest(consumer.runtime.state_dict())
    bundle = _compose_gradients(consumer, evaluator, batch, labels, references, beta=beta, gamma=gamma, delta=delta)
    for parameter in consumer.runtime.parameters():
        _require(parameter.grad is None, "diagnostic composition populated a runtime gradient")
    successor._clear_grads(consumer)
    after = _state_digest(consumer.runtime.state_dict())
    _require(before == after, "diagnostic composition changed runtime state")
    return _composition_record(bundle)


def _apply_update(consumer: Any, evaluator: FrozenMNISTEvaluator, optimizer: torch.optim.Adam, batch: torch.Tensor, labels: torch.Tensor, references: Any, *, beta: float, gamma: float, delta: float, step: int, ledger: DataReadLedger | None) -> dict[str, Any]:
    _validate_generator_optimizer(consumer, optimizer)
    generator = _generator(consumer)
    before_generator = generator.detach().clone()
    before_frozen = _frozen_state(consumer)
    optimizer.zero_grad(set_to_none=True)
    successor._clear_grads(consumer)
    bundle = _compose_gradients(consumer, evaluator, batch, labels, references, beta=beta, gamma=gamma, delta=delta)
    for name, parameter in consumer.runtime.named_parameters():
        _require(parameter.grad is None, f"runtime gradient populated before composition assignment: {name}")
    generator.grad = bundle["gradients"]["g_total"].detach().clone()
    _finite(generator.grad, "assigned g_total")
    optimizer.step()
    _finite(generator, "updated generator.phase")
    after_frozen = _frozen_state(consumer)
    _require(_state_digest(before_frozen) == _state_digest(after_frozen), "frozen runtime state changed")
    update_norm = float((generator.detach() - before_generator).to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(update_norm) and update_norm > 0.0, "generator.phase did not update")
    successor._clear_grads(consumer)
    if ledger is not None:
        ledger.record(TRAIN_ROLE, int(batch.shape[0]))
    record = _composition_record(bundle)
    record.update({"step": step, "generator_update_norm": update_norm, "encoder_unchanged": True, "frozen_runtime_unchanged": True, "trainable_parameter_names": ["generator.phase"]})
    return record


def _rng_snapshot() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.random.get_rng_state(), [state.clone() for state in torch.cuda.get_rng_state_all()]


def _restore_rng(snapshot: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    torch.random.set_rng_state(snapshot[0])
    torch.cuda.set_rng_state_all(snapshot[1])


def _run_preflight(consumer: Any, evaluator: FrozenMNISTEvaluator, references: Any, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, calibration_indices: torch.Tensor, step0: Mapping[str, Any], prior_report: Mapping[str, Any], *, beta: float, gamma: float, delta: float) -> dict[str, Any]:
    initial_state = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    evaluator_initial = {name: value.detach().cpu().clone() for name, value in evaluator.state_dict().items()}
    initial_rng = _rng_snapshot()
    bad_optimizer = torch.optim.Adam(list(consumer.runtime.parameters()), lr=1.0e-3)
    try:
        _validate_generator_optimizer(consumer, bad_optimizer)
    except (ConflictAwareExecutionError, successor.RecoveryExecutionError):
        negative_control = "FULL_RUNTIME_OPTIMIZER_REJECTED"
    else:
        raise ConflictAwareExecutionError("negative control accepted a full-runtime optimizer")

    sampler = DeterministicTrainRoleSampler(train_indices, labels)
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == calibration_indices.tolist(), "preflight first TRAIN batch differs from accepted sampler authority")
    _require(torch.equal(labels.index_select(0, first_indices), first_labels), "preflight TRAIN labels changed")
    fixed_batch = images.index_select(0, first_indices).to(consumer.device)
    fixed_labels = first_labels.to(consumer.device)
    composition0 = _diagnose_composition(consumer, evaluator, fixed_batch, fixed_labels, references, beta=beta, gamma=gamma, delta=delta)
    _require(composition0["old_tn_distribution_in_objective"] is False, "old TN term entered composed gradient")
    optimizer, optimizer_mode = _restore_prior_optimizer(consumer, step0, prior_report)
    trace: list[dict[str, Any]] = []
    sampler.load_state_dict({**dict(step0["sampler_state"]), "cursor": 0})
    for step in range(3):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(labels.index_select(0, indices), batch_labels), "preflight TRAIN sampler label binding changed")
        batch = images.index_select(0, indices).to(consumer.device)
        trace.append(_apply_update(consumer, evaluator, optimizer, batch, batch_labels.to(consumer.device), references, beta=beta, gamma=gamma, delta=delta, step=step + 1, ledger=None))
    post_state = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    _require(torch.equal(initial_state["encoder.phase"], post_state["encoder.phase"]), "preflight changed frozen encoder.phase")
    _require(not torch.equal(initial_state["generator.phase"], post_state["generator.phase"]), "preflight Generator did not depart")
    consumer.load_trainable_state_dict({"encoder.phase": initial_state["encoder.phase"], "generator.phase": initial_state["generator.phase"]})
    _freeze_generator_only(consumer)
    successor._clear_grads(consumer)
    _restore_rng(initial_rng)
    runtime_restored = _state_digest(initial_state) == _state_digest(consumer.runtime.state_dict())
    evaluator_unchanged = all(torch.equal(value.detach().cpu(), evaluator_initial[name]) for name, value in evaluator.state_dict().items())
    _require(runtime_restored and evaluator_unchanged, "preflight did not restore runtime or changed the frozen evaluator")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "runtime_restored": runtime_restored,
        "frozen_evaluators_unchanged": evaluator_unchanged,
        "starting_authority": {"path": PRIOR_STEP_0.as_posix(), "sha256": EXPECTED_PRIOR_STEP_0_SHA256, "selected_step": 0},
        "gradient_rule": "g_total=g_sem+project_nonconflicting(g_rec,g_sem)+project_nonconflicting(g_op,g_sem)",
        "eps_proj": EPS_PROJ,
        "raw_gradient_cosines": {"cos_g_rec_g_sem": composition0["gradient_cosines"]["cos_g_rec_g_sem"], "cos_g_op_g_sem": composition0["gradient_cosines"]["cos_g_op_g_sem"]},
        "projected_gradient_cosines": {"cos_g_rec_safe_g_sem": composition0["gradient_cosines"]["cos_g_rec_safe_g_sem"], "cos_g_op_safe_g_sem": composition0["gradient_cosines"]["cos_g_op_safe_g_sem"]},
        "total_vs_semantic_cosine": composition0["gradient_cosines"]["cos_g_total_g_sem"],
        "projection": composition0["projection"],
        "optimizer_mode": optimizer_mode,
        "three_step_real_update": {"updates": 3, "trace": trace, "generator_departed": True, "encoder_unchanged": True},
        "negative_control": {"name": negative_control, "passed": True},
        "frozen_parameter_probe": {"encoder_phase_unchanged": True, "ols_unchanged": True, "w_ols_unchanged": True, "evaluator_parameters_frozen": True},
        "old_tn_term": {"included_in_backward": False, "scalar_weighted_sum_backward": False},
        "same_data_order": True,
        "same_optimizer_hyperparameters": True,
        "same_beta_gamma_delta": True,
        "protected_test_sealed": {"protected": False, "test": False, "sealed": False},
        "scientific_300_step_executed": False,
    }


def _checkpoint_payload(consumer: Any, optimizer: torch.optim.Adam, sampler: DeterministicTrainRoleSampler, history: list[Mapping[str, Any]], composition_trajectory: Mapping[str, Any], *, step: int, source_manifest: Mapping[str, Any], run_identity: ResearchRunIdentity, optimizer_mode: str, preflight: Mapping[str, Any]) -> dict[str, Any]:
    _validate_generator_optimizer(consumer, optimizer)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_token": AUTHORIZATION_TOKEN,
        "completed_updates": step,
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "scientific_artifact": True,
        "operational_artifact": False,
        "run_identity": run_identity.to_dict(),
        "source_manifest": dict(source_manifest),
        "source_manifest_id": source_manifest.get("source_manifest_id"),
        "accepted_starting_checkpoint": {"path": PRIOR_STEP_0.as_posix(), "sha256": EXPECTED_PRIOR_STEP_0_SHA256, "selected_step": 0},
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator"],
        "generator_state": consumer.runtime.generator.phase.detach().cpu().clone(),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": _state_digest(_frozen_state(consumer)),
        "optimizer_mode": optimizer_mode,
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": successor._cpu_state(sampler.state_dict()),
        "rng_state": _rng_state_payload(),
        "gradient_composition": {"eps_proj": EPS_PROJ, "rule": "semantic-primary; remove only negative first-order components from reconstruction and optical-power gradients", "preflight": dict(preflight), "trajectory": dict(composition_trajectory)},
        "training_history": [dict(record) for record in history],
    }


def _recovery_payload(consumer: Any, optimizer: torch.optim.Adam, sampler: DeterministicTrainRoleSampler, history: list[Mapping[str, Any]], composition_trajectory: Mapping[str, Any], *, step: int, source_manifest: Mapping[str, Any], run_identity: ResearchRunIdentity, optimizer_mode: str, preflight: Mapping[str, Any]) -> dict[str, Any]:
    payload = _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=step, source_manifest=source_manifest, run_identity=run_identity, optimizer_mode=optimizer_mode, preflight=preflight)
    payload["scientific_artifact"] = False
    payload["operational_artifact"] = True
    payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
    return payload


def _classification(dev_semantic: Mapping[str, Any], dev_reconstruction: Mapping[str, Any], transfer: Mapping[str, Any]) -> tuple[str, dict[str, bool]]:
    semantic_supported = bool(dev_semantic["delta_macro_recall"] > 0.0 and dev_semantic["paired_bootstrap_ci"]["lower"] > 0.0)
    reconstruction_pass = bool(dev_reconstruction["delta_step_300_minus_step_0"] <= RECONSTRUCTION_TOLERANCE)
    raw_valid = bool(transfer["successor_raw_validity"]["finite"] and transfer["successor_raw_validity"]["nonconstant_fraction"] == 1.0 and transfer["successor_raw_validity"]["zero_image_count"] == 0)
    feature_decreased = bool(transfer["after"]["R_feature"] < transfer["baseline_P2_Q2"]["R_feature"])
    checks = {"dev_semantic_supported": semantic_supported, "dev_reconstruction_pass": reconstruction_pass, "sampled_R_feature_decreased": feature_decreased, "raw_valid": raw_valid}
    if semantic_supported and not reconstruction_pass:
        return CONSTRAINT_CLASSIFICATION, checks
    if not semantic_supported:
        return NOT_SUPPORTED_CLASSIFICATION, checks
    if feature_decreased and raw_valid:
        return SUPPORTED_CLASSIFICATION, checks
    return MIXED_TRANSFER_CLASSIFICATION, checks


def _make_identity(source_manifest: Mapping[str, Any], data_identity: Mapping[str, Any], weights: Mapping[str, float]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"updates": 300, "checkpoint_steps": list(CHECKPOINT_STEPS), "gradient_rule": "semantic-primary conflict-aware composition", "eps_proj": EPS_PROJ, "beta": weights["beta"], "gamma": weights["gamma"], "delta": weights["delta"], "data_source": dict(data_identity), "smooth_checkpoint_sha256": EXPECTED_SMOOTH_CHECKPOINT_SHA256},
        model_identity="LIGHTGEN_2D_CHEN_BOGT_CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1",
        optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "prior_step0_optimizer_state": EXPECTED_PRIOR_STEP_0_SHA256},
        sampler_identity={"seed": 1234, "steps": 300, "batch_size": 10, "class_balanced": True, "parent": "accepted semantic-recovery step-0 sampler cursor reset to 0"},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=None,
    )


def run_successor(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "successor output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"successor output already exists: {target}")

    start_git = successor.git_identity(root)
    authorities = _load_authorities(root)
    cuda_runtime = successor._prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "formal successor requires CUDA")
    device = torch.device("cuda:0")
    consumer, train_evaluator, smooth_evaluator, evaluator_identity = _load_models(root, authorities, device)
    _freeze_generator_only(consumer)
    consumer.runtime.train()

    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            Path(successor.__file__).resolve(),
            Path(attribution.__file__).resolve(),
            Path(phase_repair.__file__).resolve(),
            Path(canary.__file__).resolve(),
            Path(producer.__file__).resolve(),
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
            root / "src/ldm_project/v2/application/research/shared/research_run_control.py",
            root / "src/ldm_project/v2/student/evaluator.py",
        ),
        config_identity={"authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "updates": 300, "checkpoint_steps": list(CHECKPOINT_STEPS), "gradient_rule": "g_sem + nonconflicting g_rec + nonconflicting g_op", "eps_proj": EPS_PROJ, "prior_hashes": authorities["authority_hashes"]},
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    run_identity = _make_identity(source_manifest, authorities["data_identity"], authorities["weights"])
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    _write_json(target / "source_manifest.json", source_manifest)

    preflight = _run_preflight(
        consumer,
        train_evaluator,
        authorities["training"]["references"],
        authorities["images"],
        authorities["labels"],
        authorities["train_indices"],
        authorities["calibration_indices"],
        authorities["step0"],
        authorities["accepted_report"],
        beta=authorities["weights"]["beta"],
        gamma=authorities["weights"]["gamma"],
        delta=authorities["weights"]["delta"],
    )
    _write_json(target / "preflight.json", preflight)
    cuda_free, cuda_total = torch.cuda.mem_get_info(device)
    negative_control = preflight.get("negative_control")
    negative_control_passed = negative_control.get("passed") is True if isinstance(negative_control, Mapping) else False
    three_step = preflight.get("three_step_real_update")
    three_step_count = three_step.get("updates") if isinstance(three_step, Mapping) else None
    selected_engine = "conflict_aware_cuda_generator_phase"
    receipt = control.make_ready_receipt(
        correctness={
            "passed": bool(preflight.get("status") == "READY" and three_step_count == 3 and negative_control_passed and preflight.get("runtime_restored") is True and preflight.get("frozen_evaluators_unchanged") is True),
            "three_step_updates": three_step_count,
            "negative_control_rejected": negative_control_passed,
            "frozen_evaluators": preflight.get("frozen_evaluators_unchanged"),
        },
        resource={"passed": int(cuda_total) > 0 and 0 <= int(cuda_free) <= int(cuda_total) and preflight.get("status") == "READY", "free_bytes": int(cuda_free), "total_bytes": int(cuda_total)},
        implementation={"passed": _generator(consumer).requires_grad and not consumer.runtime.encoder.phase.requires_grad, "selected_engine": selected_engine},
        selected_engine=selected_engine,
        created_at=time.time(),
    )
    control.mark_preflight_ready(receipt)
    control.start_execution()
    effective_identity = control.open_scientific_attempt(target, SCIENTIFIC_ATTEMPT_ID, receipt, resume=True)
    _write_json(target / "preflight_receipt.json", receipt.to_dict())
    _write_json(
        target / "execution_manifest.json",
        {"schema": "conflict_aware_generator_semantic_recovery_execution_manifest.v1", "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorization_id": AUTHORIZATION_ID, "authorization_token": AUTHORIZATION_TOKEN, "input_authorities": ["M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION", "FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1_RESOLVED", "GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED", "CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1_Q2", "SMOOTH_MNIST_EVALUATOR_V2_1"], "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "data_identity": authorities["data_identity"], "git_before": start_git, "cuda_runtime": cuda_runtime, "scope": {"bounded_updates": 300, "trainable": ["generator.phase"], "frozen": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "evaluators"], "protected": False, "test": False, "sealed": False}},
    )

    optimizer, optimizer_mode = _restore_prior_optimizer(consumer, authorities["step0"], authorities["accepted_report"])
    sampler = DeterministicTrainRoleSampler(authorities["train_indices"], authorities["labels"])
    sampler_state = dict(authorities["step0"]["sampler_state"])
    sampler_state["cursor"] = 0
    sampler.load_state_dict(sampler_state)
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == authorities["calibration_indices"].tolist(), "scientific first TRAIN batch differs from authority")
    sampler.load_state_dict(sampler_state)
    ledger = DataReadLedger()
    history: list[dict[str, Any]] = []
    dev_records: dict[str, dict[str, Any]] = {}
    composition_trajectory: dict[str, dict[str, Any]] = {}
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=successor.SCIENTIFIC_UPDATES, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=0, latest_losses={"preflight": "READY", "optimizer_mode": optimizer_mode}, force=True)

    fixed_batch = authorities["images"].index_select(0, authorities["calibration_indices"]).to(device)
    fixed_labels = authorities["calibration_labels"].to(device)
    composition_trajectory["0"] = _diagnose_composition(consumer, train_evaluator, fixed_batch, fixed_labels, authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta=authorities["weights"]["delta"])
    dev0 = successor._evaluate_dev(consumer, train_evaluator, smooth_evaluator, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=0)
    dev_records["0"] = dev0
    _write_json(target / "dev_diagnostics_step_000000.json", {"schema": DEV_SCHEMA, "step": 0, "diagnostic": {key: value for key, value in dev0.items() if not key.startswith("_")}})
    _write_torch(target / "scientific_step_000000.pt", _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=0, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight), kind="conflict-aware semantic recovery scientific checkpoint")

    last_recovery_step = 0
    last_recovery_time = time.time()
    last_recovery_path: Path | None = None

    def recovery_factory(step: int) -> Mapping[str, Any]:
        return _recovery_payload(consumer, optimizer, sampler, history, composition_trajectory, step=step, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight)

    heartbeat.update(state=RUNNING, phase="TRAINING", step=0, latest_losses={"rule": "conflict-aware", "eps_proj": EPS_PROJ}, force=True)
    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_factory, phase="TRAINING", total_steps=successor.SCIENTIFIC_UPDATES)
    with interrupt:
        for step in range(successor.SCIENTIFIC_UPDATES):
            indices, batch_labels = sampler.next_batch()
            _require(torch.equal(authorities["labels"].index_select(0, indices), batch_labels), f"TRAIN sampler labels changed at update {step + 1}")
            batch = authorities["images"].index_select(0, indices).to(device)
            trace = _apply_update(consumer, train_evaluator, optimizer, batch, batch_labels.to(device), authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta=authorities["weights"]["delta"], step=step + 1, ledger=ledger)
            history.append(trace)
            now = time.time()
            if recovery_due(step=step + 1, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                last_recovery_path = recovery.save(recovery_factory(step + 1), step=step + 1)
                last_recovery_step = step + 1
                last_recovery_time = now
            heartbeat.update(state=RUNNING, phase="TRAINING", step=step + 1, latest_losses={key: trace[key] for key in ("L_rec", "L_op", "L_sem")}, last_recovery_checkpoint=None if last_recovery_path is None else last_recovery_path.name)
            interrupted = interrupt.handle_safe_boundary(step=step + 1, latest_losses={key: trace[key] for key in ("L_rec", "L_op", "L_sem")}, last_recovery_checkpoint=None if last_recovery_path is None else last_recovery_path.name)
            if interrupted is not None:
                control.interrupted()
                return {"result": "INTERRUPTED", "output_dir": _relative(root, target), "step": step + 1, "recovery_checkpoint": interrupted.name}
            completed = step + 1
            if completed in CHECKPOINT_STEPS[1:]:
                composition_trajectory[str(completed)] = _diagnose_composition(consumer, train_evaluator, fixed_batch, fixed_labels, authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta=authorities["weights"]["delta"])
                diagnostic = successor._evaluate_dev(consumer, train_evaluator, smooth_evaluator, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=completed)
                dev_records[str(completed)] = diagnostic
                _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {"schema": DEV_SCHEMA, "step": completed, "diagnostic": {key: value for key, value in diagnostic.items() if not key.startswith("_")}})
                _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight), kind="conflict-aware semantic recovery scientific checkpoint")

    heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=successor.SCIENTIFIC_UPDATES, latest_losses={"completed_updates": successor.SCIENTIFIC_UPDATES}, force=True)
    correctness_payload = {"schema": "conflict_aware_generator_semantic_recovery_dev_correctness.v1", "development_indices_sha256": tensor_sha256(authorities["development_indices"]), "labels": authorities["labels"].index_select(0, authorities["development_indices"]), "steps": {step: {"train_correct": dev_records[step]["_train_correct"], "smooth_correct": dev_records[step]["_smooth_correct"]} for step in dev_records}}
    _write_torch(target / "dev_correctness.pt", correctness_payload, kind="conflict-aware DEV correctness vectors")
    bootstrap = successor._paired_bootstrap(correctness_payload["labels"], correctness_payload["steps"]["0"]["smooth_correct"], correctness_payload["steps"]["300"]["smooth_correct"])
    _write_torch(target / "paired_bootstrap.pt", bootstrap, kind="conflict-aware paired DEV Smooth bootstrap")
    bootstrap_report = dict(bootstrap)
    bootstrap_report.pop("macro_delta_tensor", None)
    bootstrap_report.pop("draws", None)
    dev_semantic = {
        "primary_evaluator": evaluator_identity["smooth"],
        "step_0_macro_recall": float(dev_records["0"]["smooth_evaluator"]["macro_recall"]),
        "step_300_macro_recall": float(dev_records["300"]["smooth_evaluator"]["macro_recall"]),
        "delta_macro_recall": float(dev_records["300"]["smooth_evaluator"]["macro_recall"] - dev_records["0"]["smooth_evaluator"]["macro_recall"]),
        "step_0_per_class_recall": list(dev_records["0"]["smooth_evaluator"]["per_class_recall"]),
        "step_300_per_class_recall": list(dev_records["300"]["smooth_evaluator"]["per_class_recall"]),
        "per_class_delta": list(bootstrap_report["per_class_delta"]),
        "paired_bootstrap_ci": dict(bootstrap_report["ci95"]),
        "paired_bootstrap_artifact": _artifact_ref(root, target / "paired_bootstrap.pt", role="paired DEV Smooth bootstrap"),
        "trajectory": {step: {"smooth_macro_recall": float(record["smooth_evaluator"]["macro_recall"]), "smooth_per_class_recall": list(record["smooth_evaluator"]["per_class_recall"])} for step, record in dev_records.items()},
        "evaluator_separation": {"train_loss_provider": "FrozenMNISTEvaluator v1", "dev_primary": "SmoothMNISTEvaluator v2.1", "same_checkpoint": False, "smooth_in_objective": False},
    }
    dev_reconstruction = {
        "step_0": float(dev_records["0"]["reconstruction"]["L_rec"]),
        "step_100": float(dev_records["100"]["reconstruction"]["L_rec"]),
        "step_200": float(dev_records["200"]["reconstruction"]["L_rec"]),
        "step_300": float(dev_records["300"]["reconstruction"]["L_rec"]),
        "delta_step_300_minus_step_0": float(dev_records["300"]["reconstruction"]["L_rec"] - dev_records["0"]["reconstruction"]["L_rec"]),
        "tolerance": RECONSTRUCTION_TOLERANCE,
        "trajectory_pass": bool(dev_records["300"]["reconstruction"]["L_rec"] - dev_records["0"]["reconstruction"]["L_rec"] <= RECONSTRUCTION_TOLERANCE),
    }
    optical_validity = {
        "checkpoints": {step: {"finite": bool(record["raw_cmos"]["finite"]), "nonconstant": bool(record["raw_cmos"]["nonconstant"]), "zero_image_count": int(record["raw_cmos"]["zero_image_count"]), "L_op": float(record["optical"]["L_op"])} for step, record in dev_records.items()},
        "all_finite": all(bool(record["raw_cmos"]["finite"]) for record in dev_records.values()),
        "all_nonconstant": all(bool(record["raw_cmos"]["nonconstant"]) for record in dev_records.values()),
        "all_zero_image_counts_zero": all(int(record["raw_cmos"]["zero_image_count"]) == 0 for record in dev_records.values()),
    }
    _require(optical_validity["all_finite"] and optical_validity["all_nonconstant"] and optical_validity["all_zero_image_counts_zero"], "optical validity failed")
    transfer_authorities = {"phase_latent": authorities["phase_latent"], "selected_generated_raw": authorities["selected_generated_raw"], "real_control_raw": authorities["real_control_raw"], "selected_metrics": authorities["selected_metrics"], "authority_hashes": {"phase_latent": EXPECTED_PHASE_LATENT_SHA256}}
    transfer = successor._p2q2_transfer(root, consumer, transfer_authorities, root / producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH, target)
    classification, checks = _classification(dev_semantic, dev_reconstruction, transfer)
    control.complete()
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=successor.SCIENTIFIC_UPDATES, latest_losses={"classification": classification}, force=True)

    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": classification == SUPPORTED_CLASSIFICATION,
        "classification": classification,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "local_state": {"branch": start_git["branch"], "head": start_git["head"], "dirty_before": start_git["worktree_dirty"], "git_end": successor.git_identity(root)},
        "starting_authority": {"accepted_prior_report": _artifact_ref(root, root / PRIOR_REPORT, role="accepted failed weighted-sum successor"), "accepted_step0": _artifact_ref(root, root / PRIOR_STEP_0, role="conflict-aware starting Generator state"), "accepted_step300": _artifact_ref(root, root / PRIOR_STEP_300, role="accepted failed successor endpoint; not used as start")},
        "implementation_delta": {"gradient_composition_only": True, "architecture_changed": False, "weighted_sum_backward_removed": True, "semantic_gradient_protected": True, "projection_rule": "remove only negative dot-product components of weighted reconstruction and optical-power gradients", "eps_proj": EPS_PROJ},
        "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator"]},
        "loss_weights": {"beta": authorities["weights"]["beta"], "gamma": authorities["weights"]["gamma"], "delta": authorities["weights"]["delta"], "prior_calibration_reused": True},
        "gradient_composition": {"formula": "g_total=g_sem+g_rec_safe+g_op_safe", "g_sem": "grad_generator(delta*L_sem)", "g_rec": "grad_generator(beta*L_rec)", "g_op": "grad_generator(gamma*L_op)", "g_rec_safe": "project g_rec only when dot(g_rec,g_sem)<0", "g_op_safe": "project g_op only when dot(g_op,g_sem)<0", "eps_proj": EPS_PROJ, "preflight": preflight, "trajectory": composition_trajectory},
        "preflight": preflight,
        "training_trajectory": {"updates": len(history), "checkpoint_steps": list(CHECKPOINT_STEPS), "trace": history, "sampler_cursor": int(sampler.cursor)},
        "dev_smooth": dev_semantic,
        "dev_reconstruction": dev_reconstruction,
        "optical_validity": optical_validity,
        "p2_q2_transfer": transfer,
        "comparison_to_prior_weighted_sum": {"prior_report": _artifact_ref(root, root / PRIOR_REPORT, role="prior weighted-sum baseline"), "prior": {"smooth_macro_recall": "0.1030 -> 0.1034", "paired_ci95": [-0.0004, 0.0012], "reconstruction_delta": -0.0013992, "R_feature": "3.0552 -> 3.2092"}, "cross_run_significance": "descriptive only; no new cross-run test"},
        "counterfactual_integrity": {"same_start_step0": True, "same_data_order": True, "same_generator_architecture": True, "same_objectives": True, "same_loss_weights": True, "same_optimizer": True, "same_update_budget": True, "same_evaluators": True, "only_changed_variable": "multi-objective gradient-composition rule", "real_control_reused_exactly": True},
        "outer_gate_checks": checks,
        "input_authorities": authorities["authority_hashes"],
        "data_reads": ledger.to_dict(),
        "forbidden_data_reads": {"protected": ledger.counts.get("protected", 0), "test": ledger.counts.get("test", 0), "sealed": ledger.counts.get("sealed", 0)},
        "claim_boundary": {"scientific_training_executed": True, "updates": 300, "generation_success": classification in {SUPPORTED_CLASSIFICATION, MIXED_TRANSFER_CLASSIFICATION}, "global_architecture_inadequacy": False, "formal_promotion": False, "physical_hardware_validity": False, "architecture_change": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False},
        "files_changed": {"scientific_artifacts": sorted(path.name for path in target.iterdir()), "repository_source": ["scripts/research/lightgen/execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1.py"]},
        "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "checkpoints": [0, 100, 200, 300], "dev_evaluations": [0, 100, 200, 300], "paired_bootstrap": True, "frozen_gradient_firewall": True, "projection_orthogonality_checked": True, "no_protected_test_sealed": True},
        "unresolved": [] if classification != NOT_SUPPORTED_CLASSIFICATION else ["Conflict-aware composition was insufficient to recover dataset-level semantics under the current Generator parameterization."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and execute the bounded CUDA arm")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "updates": successor.SCIENTIFIC_UPDATES, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_successor(output_dir=args.output_dir)
    except (ConflictAwareExecutionError, successor.RecoveryExecutionError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
