"""Execute the bounded frozen-latent Generator semantic-recovery successor.

The consumer starts from the accepted Chen BOGT step-300 state, keeps the
encoder, OLS, scale, transport, latent normalization, phase law, and
evaluators fixed, and updates only ``generator.phase`` for exactly 300 TRAIN
updates.  The script owns one scientific gate and writes write-once evidence
under a new ``outputs/YYYYMMDD_<version>/runs/...`` child.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as producer
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as phase_repair
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1 as canary
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import ADAM_BETAS, ADAM_EPSILON, AMSGRAD, BogtLossWeights, CHECKPOINT_STEPS, DataReadLedger, DEV_ROLE, DeterministicTrainRoleSampler, FrozenBogtReferences, LightGen2DChenBOGTComplexOLSTrainingConsumer, PRODUCTION_PROPAGATION_GRID, SCIENTIFIC_UPDATES, TRAIN_ROLE, WEIGHT_DECAY, contracted_plane_powers, fixed_prior, git_identity, optical_power_loss_from_powers, q_from_batch, _rng_state_payload, statistical_variable, truncated_normal_positive_kl
from ldm_project.v2.application.research.shared.research_run_control import HeartbeatWriter, InterruptController, RecoveryCheckpointManager, ResearchRunControl, ResearchRunIdentity, RUN_COMPLETED, RUNNING, RUN_PREPARING, capture_execution_source_manifest, recovery_due
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
)


TASK_ID = "EXECUTE_FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "frozen-latent-generator-semantic-recovery-v1"
TRAJECTORY_ID = "frozen-latent-generator-semantic-recovery-trajectory-v1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

SOURCE_RUN = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01")
SOURCE_REPORT = SOURCE_RUN / "report.json"
SOURCE_CHECKPOINT = SOURCE_RUN / "scientific_step_000300.pt"
SOURCE_BUNDLE = SOURCE_RUN / "generation_ready_bundle.pt"
SOURCE_REFERENCES = SOURCE_RUN / "frozen_references.pt"
SOURCE_WEIGHTS = SOURCE_RUN / "frozen_weights.pt"
EXPECTED_SOURCE_REPORT_SHA256 = "e2e87a9aece563616f9e64b329411a4bcd9a0a3f6dcbcab7f220e690bee15dfc"
EXPECTED_SOURCE_CHECKPOINT_SHA256 = "574440ac9344b1c8fb407c45debf3514fea2c9ab6526873c3286852077f3a0c4"
EXPECTED_SOURCE_BUNDLE_SHA256 = "4c9a8d10ab1694d69aeaf9e58c5637c2695088a5804b15f4b976e15cc489439f"
EXPECTED_SOURCE_REFERENCES_SHA256 = "dd341390a7d02b12959321dc27363a8ef5d70b1ba0ac24cb6c29e7e01c06dd69"
EXPECTED_SOURCE_WEIGHTS_SHA256 = "6d482296e99a1520151e6a2349f57cfb13d20256fc923c6c9324593612c9de41"

PHASE_RUN = Path("outputs/20260918_chen_bogt_phase_completion_repair_v1/runs/attempt_01")
PHASE_REPORT = PHASE_RUN / "phase_completion_repair_report.json"
PHASE_SELECTION = PHASE_RUN / "phase_selection.json"
PHASE_LATENT = PHASE_RUN / "selected_phase_completion_latent_bank.pt"
EXPECTED_PHASE_REPORT_SHA256 = "3c265025cf5f813b978556edc5bb0bd31174700e4ddcfa43c624b6945b2f41a1"
EXPECTED_PHASE_SELECTION_SHA256 = "38c59c9f81162d68ed7eb00d46390fc58ff51ce1eabe3691a0230a3108065fa8"
EXPECTED_PHASE_LATENT_SHA256 = "2f374dfb16b8b95488ee5bf1b0ee776aaee6d2d37128ab6c4dc606d7cc308db2"

REAL_CONTROL = Path("outputs/20260918_chen_bogt_repaired_prior_generation_canary_v1/runs/attempt_01/real_control_cmos.pt")
EXPECTED_REAL_CONTROL_SHA256 = "6fe1796c7b8feac98b155c96851400431c1f99cee2d25a9752bff73d2a925bc2"

SMOOTH_CHECKPOINT = Path(
    "outputs/20260730_v2/runs/"
    "20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/"
    "smooth_mnist_evaluator_v21.pt"
)
EXPECTED_SMOOTH_CHECKPOINT_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260918_frozen_latent_generator_semantic_recovery_v1/runs/attempt_01"
)

TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
LOGICAL_BATCH_SIZE = 10
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
SEMANTIC_TARGET_GRADIENT_SCALE = 0.0019319
RECONSTRUCTION_TOLERANCE = 0.005
RECOVERY_INTERVAL_STEPS = 25

REPORT_SCHEMA = "frozen_latent_generator_semantic_recovery_report.v1"
CHECKPOINT_SCHEMA = "frozen_latent_generator_semantic_recovery_checkpoint.v1"
PREFLIGHT_SCHEMA = "frozen_latent_generator_semantic_recovery_preflight.v1"
DEV_SCHEMA = "frozen_latent_generator_semantic_recovery_dev_diagnostic.v1"
BOOTSTRAP_SCHEMA = "frozen_latent_generator_semantic_recovery_paired_bootstrap.v1"
CMOS_SCHEMA = "frozen_latent_generator_semantic_recovery_cmos.v1"

SUPPORTED_CLASSIFICATION = "GENERATOR_SEMANTIC_RECOVERY_SUPPORTED"
MIXED_TRANSFER_CLASSIFICATION = "REAL_LATENT_SEMANTIC_RECOVERY_SAMPLED_TRANSFER_MIXED"
NOT_SUPPORTED_CLASSIFICATION = "GENERATOR_SEMANTIC_RECOVERY_NOT_SUPPORTED"
TRADEOFF_CLASSIFICATION = "SEMANTIC_RECONSTRUCTION_TRADEOFF_MIXED"


class RecoveryExecutionError(RuntimeError):
    """Raised when an input, boundary, or valid measurement cannot be consumed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryExecutionError(message)


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


def _cpu_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, Mapping):
            result[key] = _cpu_state(value)
        elif isinstance(value, list):
            result[key] = [item.detach().cpu().clone() if isinstance(item, torch.Tensor) else item for item in value]
        else:
            result[key] = value
    return result


def _prepare_cuda_runtime_dll_search_path() -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "not_applicable", "scope": "current_python_process_only"}
    cuda_build = torch.version.cuda
    _require(isinstance(cuda_build, str) and len(cuda_build.split(".")) == 2, "CUDA build cannot identify the environment NVRTC DLL")
    major, minor = cuda_build.split(".")
    environment_bin = (Path(sys.prefix) / "bin").resolve()
    dll_path = environment_bin / f"nvrtc-builtins64_{major}{minor}.dll"
    _require(environment_bin.is_dir() and dll_path.is_file(), f"required CUDA runtime DLL is unavailable: {dll_path}")
    current_path = os.environ.get("PATH", "")
    normalized_bin = os.path.normcase(os.path.normpath(str(environment_bin)))
    entries = [entry for entry in current_path.split(os.pathsep) if entry]
    present = any(os.path.normcase(os.path.normpath(entry)) == normalized_bin for entry in entries)
    if not present:
        os.environ["PATH"] = str(environment_bin) + (os.pathsep + current_path if current_path else "")
    return {
        "status": "already_present" if present else "prepended",
        "scope": "current_python_process_only",
        "cuda_build": cuda_build,
        "environment_bin": str(environment_bin),
        "expected_dll": dll_path.name,
        "expected_dll_exists": True,
        "path_changed": not present,
    }


def _validate_current_docs(root: Path) -> dict[str, Any]:
    status = root / "docs/CURRENT_STATUS.md"
    handoff = root / "docs/status/current_handoff.md"
    status_top = "\n".join(status.read_text(encoding="utf-8").splitlines()[:40]) if status.is_file() else ""
    handoff_top = "\n".join(handoff.read_text(encoding="utf-8").splitlines()[:40]) if handoff.is_file() else ""
    return {
        "status_top_sha256": hashlib.sha256(status_top.encode("utf-8")).hexdigest(),
        "handoff_top_sha256": hashlib.sha256(handoff_top.encode("utf-8")).hexdigest(),
        "role": "provenance only; document edits do not change scientific input admission",
    }


def _load_authorities(root: Path) -> dict[str, Any]:
    docs = _validate_current_docs(root)
    cpu = phase_repair._load_cpu_context(root)

    phase_report_sha = file_sha256(root / PHASE_REPORT)
    phase_selection_sha = file_sha256(root / PHASE_SELECTION)
    phase_latent_sha = file_sha256(root / PHASE_LATENT)
    phase_report = json.loads((root / PHASE_REPORT).read_text(encoding="utf-8"))
    selection = json.loads((root / PHASE_SELECTION).read_text(encoding="utf-8"))
    _require(phase_report.get("status") == "VALID_COMPLETED", "accepted phase-repair report is not valid completed")
    _require(phase_report.get("outer_gate") == "CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1_RESOLVED", "accepted phase-repair gate changed")
    _require(selection.get("selected_candidate") == "Q2", "accepted phase selection is not Q2")
    stage2 = _mapping(phase_report.get("stage_2_downstream_propagation"), "phase stage-2 authority")
    _require(stage2.get("selected_candidate") == "Q2", "phase stage-2 selected candidate changed")
    selected_metrics = _mapping(_mapping(stage2.get("metrics"), "phase metrics").get("selected"), "selected Q2 metrics")
    for key in ("R_u", "R_phi", "R_z", "R_raw", "R_feature"):
        _require(math.isfinite(float(selected_metrics.get(key))), f"selected Q2 metric is nonfinite: {key}")

    latent = safe_torch_load(root / PHASE_LATENT, map_location="cpu")
    _require(isinstance(latent, Mapping), "selected Q2 latent bank is invalid")
    _require(latent.get("candidate_id") == "Q2" and latent.get("selected_step") == 300, "selected Q2 latent identity changed")
    u_sample = _tensor(latent.get("u_sample"), "selected Q2 u sample")
    z_sample = _tensor(latent.get("z_sample"), "selected Q2 z sample")
    _require(u_sample.shape == (1000, 100) and u_sample.dtype == torch.float64, "selected Q2 u shape/dtype changed")
    _require(z_sample.shape == (1000, 100) and z_sample.dtype == torch.complex64, "selected Q2 z shape/dtype changed")
    _finite(u_sample, "selected Q2 u")
    _finite(z_sample, "selected Q2 z")
    _require(torch.equal(u_sample, cpu["p2_candidate"]["u"]), "selected Q2 u is not the accepted P2 bank")

    generated_ref = _mapping(_mapping(stage2.get("optical"), "phase optical authority").get("generated_artifact"), "selected Q2 generated artifact")
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
        "docs": docs,
        "training": cpu,
        "phase_report": phase_report,
        "phase_selection": selection,
        "phase_latent": dict(latent),
        "selected_metrics": {key: float(selected_metrics[key]) for key in ("R_u", "R_phi", "R_z", "R_raw", "R_feature")},
        "selected_generated_path": generated_path,
        "selected_generated_raw": generated_raw,
        "real_control_path": real_control_path,
        "real_control_raw": real_raw,
        "authority_hashes": {
            "training_report": EXPECTED_SOURCE_REPORT_SHA256,
            "training_checkpoint_step_300": EXPECTED_SOURCE_CHECKPOINT_SHA256,
            "generation_bundle": EXPECTED_SOURCE_BUNDLE_SHA256,
            "frozen_references": EXPECTED_SOURCE_REFERENCES_SHA256,
            "frozen_weights": EXPECTED_SOURCE_WEIGHTS_SHA256,
            "phase_report": phase_report_sha,
            "phase_selection": phase_selection_sha,
            "phase_latent": phase_latent_sha,
            "real_control": EXPECTED_REAL_CONTROL_SHA256,
        },
    }


def _load_successor_training_authority(root: Path, authorities: Mapping[str, Any]) -> dict[str, Any]:
    training = _mapping(authorities.get("training"), "training authority")
    report = _mapping(training.get("report"), "accepted training report")
    _require(report.get("result") == "VALID_COMPLETED_GENERATION_BUNDLE_READY", "accepted Chen BOGT result changed")
    _require(report.get("outer_gate") == "CHEN_BOGT_ON_COMPLEX_OLS_BOUNDED_TRAINING_V1_RESOLVED", "accepted Chen BOGT gate changed")
    data_identity = _mapping(report.get("data_identity"), "accepted training data identity")
    checkpoint = safe_torch_load(root / SOURCE_CHECKPOINT, map_location="cpu")
    bundle = safe_torch_load(root / SOURCE_BUNDLE, map_location="cpu")
    _require(isinstance(checkpoint, Mapping) and isinstance(bundle, Mapping), "accepted Chen BOGT state is malformed")
    _require(checkpoint.get("completed_updates") == 300 and checkpoint.get("scientific_artifact") is True, "accepted step-300 checkpoint is not scientific step 300")
    _require(checkpoint.get("engineering_only") is False, "accepted checkpoint is engineering-only")
    state = _mapping(checkpoint.get("trainable_parameter_state"), "accepted trainable state")
    bundle_state = _mapping(_mapping(bundle.get("final_encoder_generator"), "accepted bundle final state").get("state"), "accepted bundle state")
    _require(set(state) == {"encoder.phase", "generator.phase"}, "accepted trainable allowlist changed")
    for name in state:
        _require(torch.equal(_tensor(state[name], name), _tensor(bundle_state.get(name), f"bundle {name}")), f"checkpoint/bundle state differs: {name}")
    _require(checkpoint.get("optimizer_configuration", {}).get("parameter_names") == ["encoder.phase", "generator.phase"], "accepted optimizer parameter order changed")
    references = FrozenBogtReferences.from_payload(_mapping(checkpoint.get("references"), "accepted checkpoint references"))
    weights = BogtLossWeights.from_payload(_mapping(checkpoint.get("weights"), "accepted checkpoint weights"))
    _require(math.isclose(weights.beta, 2.4509953404250195, rel_tol=0.0, abs_tol=1.0e-12), "accepted beta authority changed")
    _require(math.isclose(weights.gamma, 0.5330424284576345, rel_tol=0.0, abs_tol=1.0e-12), "accepted gamma authority changed")
    _require(weights.data_role == TRAIN_ROLE and not weights.engineering_only, "accepted weights are not scientific TRAIN weights")
    _require(references.data_role == TRAIN_ROLE and references.count == 55_000 and not references.engineering_only, "accepted references are not scientific TRAIN references")
    return {"checkpoint": dict(checkpoint), "bundle": dict(bundle), "state": dict(state), "references": references, "weights": weights, "data_identity": dict(data_identity)}


def _load_smooth_evaluator(root: Path, device: torch.device) -> tuple[SmoothMNISTEvaluator, dict[str, Any]]:
    path = root / SMOOTH_CHECKPOINT
    _require(path.is_file(), "independent Smooth evaluator authority is unavailable")
    payload = safe_torch_load(path, map_location="cpu")
    _require(isinstance(payload, Mapping) and payload.get("architecture_kind") == "smooth_mnist_evaluator_v2", "Smooth evaluator architecture changed")
    state = payload.get("state_dict")
    _require(isinstance(state, Mapping), "Smooth evaluator state is missing")
    evaluator = SmoothMNISTEvaluator().to(device)
    evaluator.load_state_dict(state, strict=True)
    evaluator.freeze()
    return evaluator, {
        "implementation": "src/ldm_project/v2/student/evaluator.py",
        "implementation_sha256": file_sha256(root / "src/ldm_project/v2/student/evaluator.py"),
        "checkpoint": SMOOTH_CHECKPOINT.as_posix(),
        "checkpoint_sha256": EXPECTED_SMOOTH_CHECKPOINT_SHA256,
        "architecture": "SmoothMNISTEvaluator v2.1",
        "role": "independent DEV primary; frozen evaluation only; no training gradient",
        "view": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28",
    }


def _freeze_generator_only(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> None:
    for name, parameter in consumer.runtime.named_parameters():
        parameter.requires_grad_(name == "generator.phase")
    names = [name for name, parameter in consumer.runtime.named_parameters() if parameter.requires_grad]
    _require(names == ["generator.phase"], f"successor trainable firewall changed: {names}")


def _generator_parameter(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> torch.nn.Parameter:
    parameters = dict(consumer.runtime.named_parameters())
    _require(set(parameters) == {"encoder.phase", "generator.phase"}, "runtime parameter surface changed")
    return parameters["generator.phase"]


def _validate_generator_optimizer(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer, optimizer: torch.optim.Optimizer) -> None:
    _require(isinstance(optimizer, torch.optim.Adam), "successor optimizer must be Adam")
    _require(len(optimizer.param_groups) == 1, "successor optimizer must have one parameter group")
    generator = _generator_parameter(consumer)
    _require(optimizer.param_groups[0]["params"] == [generator], "successor optimizer must contain only generator.phase")
    group = optimizer.param_groups[0]
    _require(group["lr"] == 1.0e-3 and tuple(group["betas"]) == ADAM_BETAS and group["eps"] == ADAM_EPSILON, "successor optimizer hyperparameters changed")
    _require(group["weight_decay"] == WEIGHT_DECAY and group["amsgrad"] is AMSGRAD, "successor optimizer options changed")


def _new_generator_optimizer(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(
        [_generator_parameter(consumer)],
        lr=1.0e-3,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    _validate_generator_optimizer(consumer, optimizer)
    return optimizer


def _restore_generator_optimizer(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    checkpoint: Mapping[str, Any],
) -> tuple[torch.optim.Adam, str]:
    optimizer = _new_generator_optimizer(consumer)
    old_state = _mapping(checkpoint.get("optimizer_state"), "accepted optimizer state")
    old_groups = old_state.get("param_groups")
    state_entries = old_state.get("state")
    if not isinstance(old_groups, list) or len(old_groups) != 1 or not isinstance(state_entries, Mapping):
        return optimizer, "OPTIMIZER_REINITIALIZED_GENERATOR_ONLY"
    old_params = old_groups[0].get("params")
    if not isinstance(old_params, list) or len(old_params) != 2:
        return optimizer, "OPTIMIZER_REINITIALIZED_GENERATOR_ONLY"
    generator_old_id = old_params[1]
    old_generator_state = state_entries.get(generator_old_id)
    if not isinstance(old_generator_state, Mapping):
        return optimizer, "OPTIMIZER_REINITIALIZED_GENERATOR_ONLY"
    required = {"step", "exp_avg", "exp_avg_sq"}
    if not required.issubset(old_generator_state):
        return optimizer, "OPTIMIZER_REINITIALIZED_GENERATOR_ONLY"
    generator = _generator_parameter(consumer)
    if any(not isinstance(old_generator_state[name], torch.Tensor) for name in ("exp_avg", "exp_avg_sq")):
        return optimizer, "OPTIMIZER_REINITIALIZED_GENERATOR_ONLY"
    if tuple(old_generator_state["exp_avg"].shape) != tuple(generator.shape) or tuple(old_generator_state["exp_avg_sq"].shape) != tuple(generator.shape):
        return optimizer, "OPTIMIZER_REINITIALIZED_GENERATOR_ONLY"
    template = optimizer.state_dict()
    new_id = template["param_groups"][0]["params"][0]
    template["state"] = {new_id: copy.deepcopy(dict(old_generator_state))}
    group = dict(old_groups[0])
    group["params"] = [new_id]
    template["param_groups"] = [group]
    try:
        optimizer.load_state_dict(template)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return _new_generator_optimizer(consumer), "OPTIMIZER_REINITIALIZED_GENERATOR_ONLY"
    _validate_generator_optimizer(consumer, optimizer)
    return optimizer, "GENERATOR_MOMENT_SUBSET_RESTORED"


def _clear_grads(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> None:
    for parameter in consumer.runtime.parameters():
        parameter.grad = None


def _recovery_components(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    evaluator: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: FrozenBogtReferences,
    beta: float,
    gamma: float,
    delta: float,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "successor real path is incomplete")
    raw = result.raw_intensity
    target = result.image_input.slm_intensity
    _require(raw.shape == target.shape and tuple(raw.shape[-2:]) == (200, 200) and raw.dtype == torch.float32, "successor raw/target shape or dtype changed")
    _finite(raw, "successor raw CMOS")
    _finite(target, "successor SLM target")
    l_rec = (raw - target).square().mean()
    l_op = optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)
    view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))
    _, logits = evaluator(view)
    l_sem = F.cross_entropy(logits, labels.to(device=batch.device, dtype=torch.long))
    l_total = beta * l_rec + gamma * l_op + delta * l_sem
    for name, value in (("L_rec", l_rec), ("L_op", l_op), ("L_sem", l_sem), ("L_total", l_total)):
        _finite(value, name)
    with torch.no_grad():
        u = statistical_variable(result.z_ols.coefficients, references)
        q = q_from_batch(u)
        l_dist = truncated_normal_positive_kl(q, fixed_prior(device=q.loc.device, dtype=q.loc.dtype)).mean()
    return {"result": result, "raw": raw, "target": target, "L_rec": l_rec, "L_op": l_op, "L_sem": l_sem, "L_total": l_total, "L_dist_diagnostic": l_dist}


def _grad_norm(gradient: torch.Tensor | None, name: str) -> float:
    _require(gradient is not None, f"{name} gradient is missing")
    _finite(gradient, name)
    value = float(gradient.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(value) and value > 0.0, f"{name} gradient is zero or invalid")
    return value


def _frozen_successor_state(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in consumer.runtime.state_dict().items()
        if name != "generator.phase"
    }


def _apply_update(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    evaluator: FrozenMNISTEvaluator,
    optimizer: torch.optim.Adam,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: FrozenBogtReferences,
    *,
    beta: float,
    gamma: float,
    delta: float,
    step: int,
    ledger: DataReadLedger | None,
) -> dict[str, Any]:
    _validate_generator_optimizer(consumer, optimizer)
    generator = _generator_parameter(consumer)
    before_generator = generator.detach().clone()
    before_frozen = _frozen_successor_state(consumer)
    optimizer.zero_grad(set_to_none=True)
    _clear_grads(consumer)
    components = _recovery_components(consumer, evaluator, batch, labels, references, beta, gamma, delta)
    components["L_total"].backward()
    generator_gradient_norm = _grad_norm(generator.grad, "generator.phase")
    for name, parameter in consumer.runtime.named_parameters():
        if name != "generator.phase":
            _require(parameter.grad is None, f"frozen parameter received a gradient: {name}")
    optimizer.step()
    _finite(generator, "updated generator.phase")
    after_frozen = _frozen_successor_state(consumer)
    _require(_state_digest(before_frozen) == _state_digest(after_frozen), "encoder/OLS/frozen runtime state changed")
    update_norm = float((generator.detach() - before_generator).to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(update_norm) and update_norm > 0.0, "generator.phase did not update")
    if ledger is not None:
        ledger.record(TRAIN_ROLE, int(batch.shape[0]))
    return {
        "step": step,
        "L_rec": float(components["L_rec"].detach().cpu()),
        "L_op": float(components["L_op"].detach().cpu()),
        "L_sem": float(components["L_sem"].detach().cpu()),
        "L_total": float(components["L_total"].detach().cpu()),
        "L_dist_diagnostic": float(components["L_dist_diagnostic"].detach().cpu()),
        "gradient_norm_generator_phase": generator_gradient_norm,
        "gradient_norm_encoder_phase": None,
        "generator_update_norm": update_norm,
        "encoder_unchanged": True,
        "frozen_runtime_unchanged": True,
        "old_tn_distribution_in_objective": False,
        "trainable_parameter_names": ["generator.phase"],
    }


def _calibrate_semantic_weight(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    evaluator: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: FrozenBogtReferences,
    weights: BogtLossWeights,
) -> dict[str, Any]:
    generator = _generator_parameter(consumer)
    _clear_grads(consumer)
    components = _recovery_components(consumer, evaluator, batch, labels, references, weights.beta, weights.gamma, 1.0)
    values = (components["L_rec"], components["L_op"], components["L_sem"])
    gradients: list[torch.Tensor] = []
    for value in values:
        gradients.append(torch.autograd.grad(value, generator, retain_graph=True, allow_unused=False)[0])
    rec_norm = _grad_norm(gradients[0], "generator gradient L_rec")
    op_norm = _grad_norm(gradients[1], "generator gradient L_op")
    sem_norm = _grad_norm(gradients[2], "generator gradient L_sem")
    delta = SEMANTIC_TARGET_GRADIENT_SCALE / sem_norm
    _require(math.isfinite(delta) and delta > 0.0, "semantic calibration delta is invalid")
    total = weights.beta * components["L_rec"] + weights.gamma * components["L_op"] + delta * components["L_sem"]
    weighted_gradient = torch.autograd.grad(total, generator, retain_graph=True, allow_unused=False)[0]
    weighted_norm = _grad_norm(weighted_gradient, "weighted initial generator gradient")

    wrong_labels = torch.roll(labels, shifts=1, dims=0)
    wrong_components = _recovery_components(consumer, evaluator, batch, wrong_labels, references, weights.beta, weights.gamma, delta)
    wrong_gradient = torch.autograd.grad(wrong_components["L_sem"], generator, retain_graph=True, allow_unused=False)[0]
    wrong_norm = _grad_norm(wrong_gradient, "wrong-label semantic gradient")

    # The old TN term is measured in a fresh graph only as a negative-control
    # diagnostic.  It is deliberately not part of ``total`` above.  Keep the
    # graph live here so ``allow_unused=True`` can prove that generator.phase
    # receives no TN gradient, instead of asking autograd about a detached
    # scalar.
    tn_result = consumer.runtime.forward_from_image(batch)
    _require(tn_result.ols_input_field is not None, "isolated old TN path is incomplete")
    tn_u = statistical_variable(tn_result.z_ols.coefficients, references)
    tn_q = q_from_batch(tn_u)
    tn_p = fixed_prior(device=tn_q.loc.device, dtype=tn_q.loc.dtype)
    l_dist = truncated_normal_positive_kl(tn_q, tn_p).mean()
    tn_gradient = None if not l_dist.requires_grad else torch.autograd.grad(l_dist, generator, retain_graph=False, allow_unused=True)[0]
    tn_norm = 0.0 if tn_gradient is None else float(tn_gradient.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(tn_norm), "isolated old TN gradient is nonfinite")
    _clear_grads(consumer)
    return {
        "beta": float(weights.beta),
        "gamma": float(weights.gamma),
        "g_rec": rec_norm,
        "g_op": op_norm,
        "g_sem": sem_norm,
        "g_target": SEMANTIC_TARGET_GRADIENT_SCALE,
        "delta": float(delta),
        "weighted_initial_gradient_norms": {
            "beta_L_rec": float(weights.beta * rec_norm),
            "gamma_L_op": float(weights.gamma * op_norm),
            "delta_L_sem": float(delta * sem_norm),
            "total": weighted_norm,
        },
        "wrong_label_negative_control": {
            "labels_shift": 1,
            "L_sem_wrong": float(wrong_components["L_sem"].detach().cpu()),
            "gradient_norm": wrong_norm,
            "label_tensor_different": bool(not torch.equal(labels, wrong_labels)),
        },
        "old_tn_distribution_term": {
            "formula": "KL(Q||P)",
            "included_in_objective": False,
            "backward_influence_on_generator": 0.0,
            "isolated_generator_gradient_norm_if_differentiated": tn_norm,
            "structurally_zero_if_isolated": bool(tn_norm == 0.0),
        },
        "freeze_after_step_zero": True,
    }


def _run_preflight(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    evaluator: FrozenMNISTEvaluator,
    references: FrozenBogtReferences,
    weights: BogtLossWeights,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    delta: float,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    initial_state = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    sampler = DeterministicTrainRoleSampler(train_indices, labels)
    accepted_sampler = dict(_mapping(checkpoint.get("sampler_state"), "accepted sampler"))
    accepted_sampler["cursor"] = 0
    sampler.load_state_dict(accepted_sampler)
    bad_optimizer = torch.optim.Adam(list(consumer.runtime.parameters()), lr=1.0e-3)
    try:
        _validate_generator_optimizer(consumer, bad_optimizer)
    except RecoveryExecutionError:
        negative_control = "FULL_RUNTIME_OPTIMIZER_REJECTED"
    else:
        raise RecoveryExecutionError("negative control accepted a full-runtime optimizer")
    optimizer = _new_generator_optimizer(consumer)
    trace: list[dict[str, Any]] = []
    for step in range(3):
        indices, batch_labels = sampler.next_batch()
        source_labels = labels.index_select(0, indices)
        _require(torch.equal(source_labels, batch_labels), "preflight TRAIN sampler label binding changed")
        batch = images.index_select(0, indices).to(consumer.device)
        trace.append(_apply_update(consumer, evaluator, optimizer, batch, batch_labels.to(consumer.device), references, beta=weights.beta, gamma=weights.gamma, delta=delta, step=step + 1, ledger=None))
    post_state = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    _require(torch.equal(initial_state["encoder.phase"], post_state["encoder.phase"]), "preflight changed frozen encoder.phase")
    _require(not torch.equal(initial_state["generator.phase"], post_state["generator.phase"]), "preflight generator did not depart")
    consumer.load_trainable_state_dict({"encoder.phase": initial_state["encoder.phase"], "generator.phase": initial_state["generator.phase"]})
    _freeze_generator_only(consumer)
    _clear_grads(consumer)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "static_lineage": "accepted step-300 state, frozen P2/Q2, TRAIN role, and independent Smooth evaluator validated",
        "three_step_real_update": {"updates": 3, "trace": trace, "generator_departed": True, "encoder_unchanged": True},
        "negative_control": {"name": negative_control, "passed": True},
        "frozen_parameter_probe": {"encoder_phase_unchanged": True, "ols_unchanged": True},
        "protected_test_sealed": {"protected": False, "test": False, "sealed": False},
        "scientific_300_step_executed": False,
    }


@torch.no_grad()
def _evaluate_dev(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    development_indices: torch.Tensor,
    references: FrozenBogtReferences,
    ledger: DataReadLedger,
    *,
    step: int,
) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    train_evaluator.eval()
    smooth_evaluator.eval()
    train_correct: list[torch.Tensor] = []
    smooth_correct: list[torch.Tensor] = []
    smooth_logits_parts: list[torch.Tensor] = []
    train_logits_parts: list[torch.Tensor] = []
    reconstruction_sum = 0.0
    raw_count = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    raw_square_sum = 0.0
    raw_sum = 0.0
    nonconstant = True
    zero_images = 0
    power_sum = torch.zeros(5, dtype=torch.float64)
    optical_loss_sum = 0.0
    try:
        for start in range(0, int(development_indices.numel()), LOGICAL_BATCH_SIZE):
            indices = development_indices[start : start + LOGICAL_BATCH_SIZE]
            batch = images.index_select(0, indices).to(consumer.device)
            target_labels = labels.index_select(0, indices)
            result = consumer.runtime.forward_from_image(batch)
            _require(result.image_input is not None and result.ols_input_field is not None, "DEV path is incomplete")
            raw = result.raw_intensity
            target = result.image_input.slm_intensity
            _finite(raw, "DEV raw CMOS")
            _finite(target, "DEV SLM target")
            reconstruction_sum += float((raw - target).square().sum().cpu())
            powers = contracted_plane_powers(result)
            power_sum += powers.to(torch.float64).sum(dim=0).cpu()
            ratios = (powers.to(torch.float64) + 1.0e-12) / (references.power_reference.to(device=powers.device, dtype=torch.float64) + 1.0e-12)
            optical_loss_sum += float(torch.log(ratios).square().sum().cpu())
            flat = raw.detach().cpu().to(torch.float64).flatten(1)
            sample_std = flat.std(dim=1, unbiased=False)
            nonconstant = nonconstant and bool((sample_std > 0.0).all().item())
            zero_images += int((flat.abs().sum(dim=1) == 0.0).sum().item())
            raw_count += int(flat.numel())
            raw_sum += float(flat.sum())
            raw_square_sum += float(flat.square().sum())
            raw_min = min(raw_min, float(flat.min()))
            raw_max = max(raw_max, float(flat.max()))
            view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))
            _, train_logits = train_evaluator(view)
            _, smooth_logits = smooth_evaluator(view)
            train_logits_cpu = train_logits.cpu()
            smooth_logits_cpu = smooth_logits.cpu()
            train_logits_parts.append(train_logits_cpu)
            smooth_logits_parts.append(smooth_logits_cpu)
            train_correct.append((train_logits_cpu.argmax(dim=1) == target_labels).contiguous())
            smooth_correct.append((smooth_logits_cpu.argmax(dim=1) == target_labels).contiguous())
            ledger.record(DEV_ROLE, int(indices.numel()))
    finally:
        consumer.runtime.train(was_training)
    train_correct_cpu = torch.cat(train_correct)
    smooth_correct_cpu = torch.cat(smooth_correct)
    train_logits_cpu = torch.cat(train_logits_parts)
    smooth_logits_cpu = torch.cat(smooth_logits_parts)

    def recalls(correct: torch.Tensor) -> tuple[list[float], float]:
        values: list[float] = []
        for digit in range(10):
            mask = labels.index_select(0, development_indices) == digit
            _require(int(mask.sum()) == 500, f"DEV class count changed for {digit}")
            values.append(float(correct[mask].to(torch.float64).mean()))
        return values, float(sum(values) / 10.0)

    train_per_class, train_macro = recalls(train_correct_cpu)
    smooth_per_class, smooth_macro = recalls(smooth_correct_cpu)
    variance = raw_square_sum / float(raw_count) - (raw_sum / float(raw_count)) ** 2
    diagnostic = {
        "step": step,
        "rows": int(development_indices.numel()),
        "role": DEV_ROLE,
        "train_compatible_evaluator": {
            "architecture": "FrozenMNISTEvaluator v1",
            "macro_recall": train_macro,
            "per_class_recall": train_per_class,
            "labels_consumed_for_diagnostic": True,
        },
        "smooth_evaluator": {
            "architecture": "SmoothMNISTEvaluator v2.1",
            "macro_recall": smooth_macro,
            "per_class_recall": smooth_per_class,
            "labels_consumed": True,
            "independent_primary": True,
        },
        "reconstruction": {"L_rec": reconstruction_sum / float(raw_count)},
        "optical": {
            "L_op": float(optical_loss_sum / float(development_indices.numel() * 5)),
            "power_mean": [float(value) for value in (power_sum / float(development_indices.numel())).tolist()],
        },
        "raw_cmos": {
            "finite": True,
            "nonconstant": nonconstant,
            "zero_image_count": zero_images,
            "zero_image_fraction": float(zero_images / float(development_indices.numel())),
            "min": raw_min,
            "max": raw_max,
            "std_population": max(0.0, variance) ** 0.5,
        },
        "evaluator_parameter_gradients_none": bool(
            all(parameter.grad is None for parameter in train_evaluator.parameters())
            and all(parameter.grad is None for parameter in smooth_evaluator.parameters())
        ),
        "_train_correct": train_correct_cpu,
        "_smooth_correct": smooth_correct_cpu,
        "_train_logits": train_logits_cpu,
        "_smooth_logits": smooth_logits_cpu,
    }
    _require(diagnostic["raw_cmos"]["nonconstant"] and diagnostic["raw_cmos"]["zero_image_count"] == 0, f"DEV raw validity failed at step {step}")
    _require(diagnostic["evaluator_parameter_gradients_none"], "DEV evaluator received a gradient")
    return diagnostic


def _paired_bootstrap(labels: torch.Tensor, step0: torch.Tensor, step300: torch.Tensor) -> dict[str, Any]:
    _require(step0.shape == step300.shape == labels.shape == (5000,), "paired bootstrap vectors have the wrong shape")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    class_deltas: list[torch.Tensor] = []
    per_class_delta: list[float] = []
    draws: list[torch.Tensor] = []
    for digit in range(10):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == 500, f"bootstrap DEV class count changed for {digit}")
        draw = torch.randint(matches.numel(), (BOOTSTRAP_REPLICATES, matches.numel()), generator=generator)
        draws.append(draw)
        base = step0.index_select(0, matches)[draw]
        candidate = step300.index_select(0, matches)[draw]
        class_deltas.append((candidate.to(torch.float64) - base.to(torch.float64)).mean(dim=1))
        per_class_delta.append(float(step300.index_select(0, matches).to(torch.float64).mean() - step0.index_select(0, matches).to(torch.float64).mean()))
    macro_delta = torch.stack(class_deltas).mean(dim=0)
    quantiles = torch.quantile(macro_delta, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "type": "paired, class-stratified bootstrap over the same 500 DEV examples per class",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "quantile_convention": "torch.quantile interpolation=linear",
        "delta_macro_recall": float(macro_delta.mean()),
        "ci95": {"lower": float(quantiles[0]), "upper": float(quantiles[1])},
        "per_class_delta": per_class_delta,
        "macro_delta_tensor": macro_delta,
        "draws": draws,
    }


def _p2q2_transfer(
    root: Path,
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    authorities: Mapping[str, Any],
    evaluator_checkpoint: Path,
    target: Path,
) -> dict[str, Any]:
    latent = _mapping(authorities.get("phase_latent"), "selected Q2 latent")
    z_sample = _tensor(latent.get("z_sample"), "selected Q2 z")[:200].contiguous()
    baseline_raw = _tensor(authorities.get("selected_generated_raw"), "accepted Q2 generated raw")
    real_raw = _tensor(authorities.get("real_control_raw"), "accepted real control raw")
    outputs: list[torch.Tensor] = []
    consumer.runtime.eval()
    for start in range(0, 200, LOGICAL_BATCH_SIZE):
        result = consumer.generated_forward(z_sample[start : start + LOGICAL_BATCH_SIZE].to(consumer.device))
        raw = result.raw_intensity.detach().cpu()
        _finite(raw, "successor P2+Q2 raw CMOS")
        outputs.append(raw)
    successor_raw = torch.cat(outputs)
    _require(successor_raw.shape == (200, 1, 200, 200), "successor P2+Q2 raw shape changed")
    _write_torch(
        target / "successor_p2q2_generated_cmos.pt",
        {"schema": CMOS_SCHEMA, "group": "successor_p2q2_generated", "input_z": z_sample, "raw_cmos_intensity": successor_raw, "candidate_id": "Q2", "selected_step": 300},
        kind="successor P2+Q2 generated CMOS",
    )
    successor_report, successor_outputs = canary._evaluate_frozen_semantics(checkpoint=evaluator_checkpoint, generated_raw=successor_raw, real_raw=real_raw)
    baseline_report, baseline_outputs = canary._evaluate_frozen_semantics(checkpoint=evaluator_checkpoint, generated_raw=baseline_raw, real_raw=real_raw)
    real_view = real_raw.to(torch.float64).flatten(1)
    raw_mean = real_view.mean(dim=0)
    raw_std = real_view.std(dim=0, unbiased=False)
    successor_raw_view = canary._standardize(successor_raw.to(torch.float64).flatten(1), raw_mean, raw_std)
    baseline_raw_view = canary._standardize(baseline_raw.to(torch.float64).flatten(1), raw_mean, raw_std)
    real_raw_view = canary._standardize(real_view, raw_mean, raw_std)
    raw_self = canary._energy_distance(real_raw_view[:100], real_raw_view[100:])
    successor_raw_ed = canary._energy_distance(successor_raw_view, real_raw_view)
    baseline_raw_ed = canary._energy_distance(baseline_raw_view, real_raw_view)
    successor_feature = float(successor_report["comparison"]["R_feature"])
    baseline_feature = float(baseline_report["comparison"]["R_feature"])
    expected = _mapping(authorities.get("selected_metrics"), "accepted Q2 metrics")
    _require(abs(baseline_feature - float(expected["R_feature"])) < 1.0e-9, "recomputed accepted Q2 R_feature disagrees with authority")
    baseline_r_raw = baseline_raw_ed / max(raw_self, 1.0e-12)
    _require(abs(baseline_r_raw - float(expected["R_raw"])) < 1.0e-9, "recomputed accepted Q2 R_raw disagrees with authority")
    successor_r_raw = successor_raw_ed / max(raw_self, 1.0e-12)
    after = {
        "R_u": float(expected["R_u"]),
        "R_phi": float(expected["R_phi"]),
        "R_z": float(expected["R_z"]),
        "R_raw": float(successor_r_raw),
        "R_feature": successor_feature,
    }
    baseline = {key: float(expected[key]) for key in ("R_u", "R_phi", "R_z", "R_raw", "R_feature")}
    return {
        "input": {"candidate_id": "Q2", "selected_step": 300, "sample_count": 200, "latent_bank_sha256": authorities["authority_hashes"]["phase_latent"], "real_control_reused_exactly": True},
        "frozen_context": {"R_u": baseline["R_u"], "phase_candidate": "Q2_ANCHOR_CONDITIONED_JOINT_UNCONDITIONAL", "R_phi": baseline["R_phi"], "R_z": baseline["R_z"]},
        "baseline_P2_Q2": baseline,
        "after": after,
        "absolute_delta_after_minus_baseline": {key: after[key] - baseline[key] for key in after},
        "ratio_after_over_baseline": {key: after[key] / max(baseline[key], 1.0e-12) for key in after},
        "raw_distribution": {
            "R_raw": successor_r_raw,
            "baseline_R_raw_recomputed": baseline_r_raw,
            "successor_energy_distance_to_real": successor_raw_ed,
            "baseline_energy_distance_to_real": baseline_raw_ed,
            "real_control_split_energy_distance": raw_self,
            "standardization": "per-pixel mean/std from the exact accepted 200-frame real-control CMOS artifact",
        },
        "diversity": {
            "successor_raw": canary._diversity_view(successor_raw, name="successor_p2q2_raw"),
            "baseline_raw": canary._diversity_view(baseline_raw, name="accepted_p2q2_raw"),
            "successor_feature": canary._diversity_view(successor_outputs["generated_embedding"], name="successor_p2q2_feature"),
            "baseline_feature": canary._diversity_view(baseline_outputs["generated_embedding"], name="accepted_p2q2_feature"),
        },
        "successor_raw_validity": canary._raw_cmos_statistics(successor_raw, name="successor_p2q2_generated"),
        "baseline_raw_validity": canary._raw_cmos_statistics(baseline_raw, name="accepted_p2q2_generated"),
        "evaluator": {"successor": successor_report, "baseline": baseline_report},
        "generated_cmos_artifact": _artifact_ref(root, target / "successor_p2q2_generated_cmos.pt", role="successor P2+Q2 generated CMOS"),
    }


def _checkpoint_payload(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    optimizer: torch.optim.Adam,
    sampler: DeterministicTrainRoleSampler,
    history: list[Mapping[str, Any]],
    *,
    step: int,
    calibration: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    optimizer_mode: str,
) -> dict[str, Any]:
    _validate_generator_optimizer(consumer, optimizer)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_token": AUTHORIZATION_TOKEN,
        "completed_updates": step,
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "selection_rule": "fixed successor endpoint step 300",
        "scientific_artifact": True,
        "operational_artifact": False,
        "run_identity": run_identity.to_dict(),
        "source_manifest": dict(source_manifest),
        "source_manifest_id": source_manifest.get("source_manifest_id"),
        "accepted_parent_checkpoint": {"path": SOURCE_CHECKPOINT.as_posix(), "sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256, "selected_step": 300},
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator"],
        "generator_state": consumer.runtime.generator.phase.detach().cpu().clone(),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": _state_digest(_frozen_successor_state(consumer)),
        "optimizer_mode": optimizer_mode,
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": _cpu_state(sampler.state_dict()),
        "rng_state": _rng_state_payload(),
        "calibration": dict(calibration),
        "training_history": [dict(record) for record in history],
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    _write_torch(path, payload, kind="frozen-latent semantic recovery scientific checkpoint")


def _recovery_payload(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    optimizer: torch.optim.Adam,
    sampler: DeterministicTrainRoleSampler,
    history: list[Mapping[str, Any]],
    *,
    step: int,
    calibration: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    optimizer_mode: str,
) -> dict[str, Any]:
    payload = _checkpoint_payload(consumer, optimizer, sampler, history, step=step, calibration=calibration, source_manifest=source_manifest, run_identity=run_identity, optimizer_mode=optimizer_mode)
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
    if not semantic_supported:
        return NOT_SUPPORTED_CLASSIFICATION, checks
    if not reconstruction_pass:
        return TRADEOFF_CLASSIFICATION, checks
    if feature_decreased and raw_valid:
        return SUPPORTED_CLASSIFICATION, checks
    return MIXED_TRANSFER_CLASSIFICATION, checks


def _make_identity(source_manifest: Mapping[str, Any], data_identity: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"updates": 300, "checkpoint_steps": list(CHECKPOINT_STEPS), "objective": "L_recovery=beta*L_rec+gamma*L_op+delta*L_sem", "data_source": dict(data_identity), "smooth_checkpoint_sha256": EXPECTED_SMOOTH_CHECKPOINT_SHA256},
        model_identity="LIGHTGEN_2D_CHEN_BOGT_FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1",
        optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD},
        sampler_identity={"seed": 1234, "steps": 300, "batch_size": 10, "class_balanced": True, "parent": "accepted Chen BOGT TRAIN sampler permutations; cursor reset to 0"},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=SCIENTIFIC_ATTEMPT_ID,
    )


def run_successor(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "successor output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"successor output already exists: {target}")

    start_git = git_identity(root)
    authorities = _load_authorities(root)
    training_authority = _load_successor_training_authority(root, authorities)
    cuda_runtime = _prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "formal successor requires CUDA")
    device = torch.device("cuda:0")

    images, labels, train_indices, development_indices, data_identity = producer._load_scientific_dataset(root)
    expected_data = training_authority["data_identity"]
    _require(data_identity["train_count"] == expected_data.get("train_count") == 55_000, "successor TRAIN row count differs from the accepted authority")
    _require(data_identity["development_count"] == expected_data.get("development_count") == 5_000, "successor DEV row count differs from the accepted authority")
    _require(data_identity["seed"] == expected_data.get("seed") == producer.SCHEDULE_SEED, "successor TRAIN/DEV split seed differs from the accepted authority")
    _require(data_identity["train_counts_by_class"] == expected_data.get("train_counts_by_class"), "successor TRAIN class roles differ from the accepted authority")
    _require(data_identity["development_counts_by_class"] == expected_data.get("development_counts_by_class"), "successor DEV class roles differ from the accepted authority")
    smooth_evaluator, smooth_identity = _load_smooth_evaluator(root, device)
    frozen_checkpoint_path = root / producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH
    train_evaluator, train_eval_payload = load_frozen_evaluator(frozen_checkpoint_path, device=device)
    _require(all(not parameter.requires_grad for parameter in train_evaluator.parameters()), "TRAIN evaluator is not frozen")

    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            Path(producer.__file__).resolve(),
            Path(phase_repair.__file__).resolve(),
            Path(canary.__file__).resolve(),
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
            root / "src/ldm_project/v2/application/research/shared/research_run_control.py",
            root / "src/ldm_project/v2/student/evaluator.py",
        ),
        config_identity={"authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "updates": 300, "checkpoint_steps": list(CHECKPOINT_STEPS), "objective": "L_recovery=beta*L_rec+gamma*L_op+delta*L_sem", "parent_hashes": authorities["authority_hashes"], "smooth_checkpoint_sha256": EXPECTED_SMOOTH_CHECKPOINT_SHA256},
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    run_identity = _make_identity(source_manifest, data_identity)
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    control.mark_preflight_ready()
    control.start_execution()
    effective_identity = control.begin_scientific_attempt(SCIENTIFIC_ATTEMPT_ID)
    _write_json(target / "source_manifest.json", source_manifest)
    _write_json(
        target / "execution_manifest.json",
        {"schema": "frozen_latent_generator_semantic_recovery_execution_manifest.v1", "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorization_id": AUTHORIZATION_ID, "authorization_token": AUTHORIZATION_TOKEN, "input_authorities": ["M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION", "CHEN_BOGT_ON_COMPLEX_OLS_TRAINING_V1_STEP_300", "CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1_Q2", "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA", "SMOOTH_MNIST_EVALUATOR_V2_1"], "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "data_identity": data_identity, "git_before": start_git, "cuda_runtime": cuda_runtime, "scope": {"bounded_updates": 300, "trainable": ["generator.phase"], "frozen": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "evaluators"], "protected": False, "test": False, "sealed": False}},
    )

    consumer = LightGen2DChenBOGTComplexOLSTrainingConsumer(device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    bundle = training_authority["bundle"]
    _require(consumer.runtime_identity() == _mapping(bundle.get("ols"), "bundle OLS").get("interface"), "current runtime identity differs from accepted bundle")
    consumer.load_trainable_state_dict(training_authority["state"])
    consumer.bind_references(training_authority["references"], for_scientific=True)
    consumer.bind_weights(training_authority["weights"], for_scientific=True)
    _freeze_generator_only(consumer)
    consumer.runtime.train()

    accepted_sampler = dict(_mapping(training_authority["checkpoint"].get("sampler_state"), "accepted sampler"))
    _require(accepted_sampler.get("cursor") == 300, "accepted parent sampler cursor is not step 300")
    sampler = DeterministicTrainRoleSampler(train_indices, labels)
    initial_sampler = dict(accepted_sampler)
    initial_sampler["cursor"] = 0
    sampler.load_state_dict(initial_sampler)
    first_indices, first_labels = sampler.next_batch()
    _require(torch.equal(labels.index_select(0, first_indices), first_labels), "accepted TRAIN sampler labels changed")
    calibration_identity = _mapping(training_authority["weights"].calibration_batch_identity, "accepted calibration identity")
    _require(first_indices.tolist() == calibration_identity.get("source_indices"), "successor first TRAIN batch differs from accepted sampler/calibration authority")
    sampler.load_state_dict(initial_sampler)

    ledger = DataReadLedger()
    calibration_batch = images.index_select(0, first_indices).to(device)
    calibration = _calibrate_semantic_weight(consumer, train_evaluator, calibration_batch, first_labels.to(device), training_authority["references"], training_authority["weights"])
    delta = float(calibration["delta"])
    preflight = _run_preflight(consumer, train_evaluator, training_authority["references"], training_authority["weights"], images, labels, train_indices, delta=delta, checkpoint=training_authority["checkpoint"])
    _write_json(target / "preflight.json", preflight)
    _write_json(target / "semantic_weight_calibration.json", calibration)

    optimizer, optimizer_mode = _restore_generator_optimizer(consumer, training_authority["checkpoint"])
    _freeze_generator_only(consumer)
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=SCIENTIFIC_UPDATES, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    history: list[dict[str, Any]] = []
    dev_records: dict[str, dict[str, Any]] = {}
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=0, latest_losses={"preflight": "READY", "optimizer_mode": optimizer_mode}, force=True)

    dev0 = _evaluate_dev(consumer, train_evaluator, smooth_evaluator, images, labels, development_indices, training_authority["references"], ledger, step=0)
    dev_records["0"] = dev0
    _write_json(target / "dev_diagnostics_step_000000.json", {"schema": DEV_SCHEMA, "step": 0, "diagnostic": {key: value for key, value in dev0.items() if not key.startswith("_")}})
    _save_checkpoint(target / "scientific_step_000000.pt", _checkpoint_payload(consumer, optimizer, sampler, history, step=0, calibration=calibration, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode))

    last_recovery_step = 0
    last_recovery_time = time.time()
    last_recovery_path: Path | None = None

    def recovery_factory(step: int) -> Mapping[str, Any]:
        return _recovery_payload(consumer, optimizer, sampler, history, step=step, calibration=calibration, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode)

    heartbeat.update(state=RUNNING, phase="TRAINING", step=0, latest_losses={"L_sem": calibration["g_sem"], "delta": delta}, force=True)
    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_factory, phase="TRAINING", total_steps=SCIENTIFIC_UPDATES)
    with interrupt:
        for step in range(SCIENTIFIC_UPDATES):
            indices, batch_labels = sampler.next_batch()
            _require(torch.equal(labels.index_select(0, indices), batch_labels), f"TRAIN sampler labels changed at update {step + 1}")
            batch = images.index_select(0, indices).to(device)
            trace = _apply_update(consumer, train_evaluator, optimizer, batch, batch_labels.to(device), training_authority["references"], beta=training_authority["weights"].beta, gamma=training_authority["weights"].gamma, delta=delta, step=step + 1, ledger=ledger)
            history.append(trace)
            now = time.time()
            if recovery_due(step=step + 1, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                last_recovery_path = recovery.save(recovery_factory(step + 1), step=step + 1)
                last_recovery_step = step + 1
                last_recovery_time = now
            heartbeat.update(state=RUNNING, phase="TRAINING", step=step + 1, latest_losses={key: trace[key] for key in ("L_rec", "L_op", "L_sem", "L_total")}, last_recovery_checkpoint=None if last_recovery_path is None else last_recovery_path.name)
            interrupted = interrupt.handle_safe_boundary(step=step + 1, latest_losses={key: trace[key] for key in ("L_rec", "L_op", "L_sem", "L_total")}, last_recovery_checkpoint=None if last_recovery_path is None else last_recovery_path.name)
            if interrupted is not None:
                control.interrupted()
                return {"result": "INTERRUPTED", "output_dir": _relative(root, target), "step": step + 1, "recovery_checkpoint": interrupted.name}
            completed = step + 1
            if completed in CHECKPOINT_STEPS[1:]:
                diagnostic = _evaluate_dev(consumer, train_evaluator, smooth_evaluator, images, labels, development_indices, training_authority["references"], ledger, step=completed)
                dev_records[str(completed)] = diagnostic
                _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {"schema": DEV_SCHEMA, "step": completed, "diagnostic": {key: value for key, value in diagnostic.items() if not key.startswith("_")}})
                _save_checkpoint(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, step=completed, calibration=calibration, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode))

    heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=SCIENTIFIC_UPDATES, latest_losses={"completed_updates": SCIENTIFIC_UPDATES}, force=True)
    correctness_payload = {"schema": "frozen_latent_generator_semantic_recovery_dev_correctness.v1", "development_indices_sha256": tensor_sha256(development_indices), "labels": labels.index_select(0, development_indices), "steps": {step: {"train_correct": dev_records[step]["_train_correct"], "smooth_correct": dev_records[step]["_smooth_correct"]} for step in dev_records}}
    _write_torch(target / "dev_correctness.pt", correctness_payload, kind="successor DEV correctness vectors")
    bootstrap = _paired_bootstrap(correctness_payload["labels"], correctness_payload["steps"]["0"]["smooth_correct"], correctness_payload["steps"]["300"]["smooth_correct"])
    _write_torch(target / "paired_bootstrap.pt", bootstrap, kind="successor paired bootstrap")
    for key in ("macro_delta_tensor", "draws"):
        bootstrap.pop(key, None)
    dev_semantic = {
        "primary_evaluator": smooth_identity,
        "step_0_macro_recall": float(dev_records["0"]["smooth_evaluator"]["macro_recall"]),
        "step_300_macro_recall": float(dev_records["300"]["smooth_evaluator"]["macro_recall"]),
        "delta_macro_recall": float(dev_records["300"]["smooth_evaluator"]["macro_recall"] - dev_records["0"]["smooth_evaluator"]["macro_recall"]),
        "step_0_per_class_recall": list(dev_records["0"]["smooth_evaluator"]["per_class_recall"]),
        "step_300_per_class_recall": list(dev_records["300"]["smooth_evaluator"]["per_class_recall"]),
        "per_class_delta": list(bootstrap["per_class_delta"]),
        "paired_bootstrap_ci": dict(bootstrap["ci95"]),
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
    transfer = _p2q2_transfer(root, consumer, authorities, frozen_checkpoint_path, target)
    classification, checks = _classification(dev_semantic, dev_reconstruction, transfer)
    _require(optical_validity["all_finite"] and optical_validity["all_nonconstant"] and optical_validity["all_zero_image_counts_zero"], "optical validity failed")
    control.complete()
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=SCIENTIFIC_UPDATES, latest_losses={"classification": classification}, force=True)

    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": classification == SUPPORTED_CLASSIFICATION,
        "classification": classification,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "local_state": {"branch": start_git["branch"], "head": start_git["head"], "dirty_before": start_git["worktree_dirty"], "git_end": git_identity(root)},
        "implementation_delta": {"new_executor": "scripts/research/lightgen/execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1.py", "architecture_changed": False, "old_tn_distribution_pressure_removed_from_backward": True, "semantic_loss_added_to_generator_objective": True},
        "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator"]},
        "objective": {"formula": "L_recovery=beta*L_rec+gamma*L_op+delta*L_sem", "L_rec": "exact existing raw CMOS versus SLM intensity MSE", "L_op": "exact existing five-plane relative log-power loss", "L_sem": "CE(FrozenMNISTEvaluator(clamp(raw_cmos,[0,1])->adaptive_avg_pool2d(28,28)), label)", "beta": float(training_authority["weights"].beta), "gamma": float(training_authority["weights"].gamma), "calibration": calibration, "old_tn_term": calibration["old_tn_distribution_term"]},
        "optimizer": {"mode": optimizer_mode, "name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD},
        "preflight": preflight,
        "training_trajectory": {"updates": len(history), "checkpoint_steps": list(CHECKPOINT_STEPS), "trace": history, "sampler_cursor": int(sampler.cursor)},
        "dev_semantic": dev_semantic,
        "dev_reconstruction": dev_reconstruction,
        "optical_validity": optical_validity,
        "p2_q2_transfer": transfer,
        "counterfactual_integrity": {"parent": "accepted Chen BOGT step-300 state", "phase": "accepted Q2_ANCHOR_CONDITIONED_JOINT_UNCONDITIONAL", "P2": "accepted P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA", "real_control": "exact accepted artifact reuse", "generator_only_update": True},
        "outer_gate_checks": checks,
        "input_authorities": authorities["authority_hashes"],
        "data_reads": ledger.to_dict(),
        "forbidden_data_reads": {"protected": ledger.counts.get("protected", 0), "test": ledger.counts.get("test", 0), "sealed": ledger.counts.get("sealed", 0)},
        "claim_boundary": {"scientific_training_executed": True, "updates": 300, "generation_success": False, "promotion": False, "architecture_change": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False},
        "files_changed": {"scientific_artifacts": sorted(path.name for path in target.iterdir()), "repository_source": ["scripts/research/lightgen/execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1.py"]},
        "validation": {"cuda": True, "preflight": "READY", "checkpoints": [0, 100, 200, 300], "dev_evaluations": [0, 100, 200, 300], "paired_bootstrap": True, "frozen_gradient_firewall": True},
        "unresolved": [] if classification == SUPPORTED_CLASSIFICATION else ["semantic-recovery gate did not satisfy all support conditions"],
        "next_action": "stop at result interpretation; no automatic continuation, promotion, architecture modification, or second arm",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit successor authorization and execute the bounded CUDA arm")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "updates": SCIENTIFIC_UPDATES, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_successor(output_dir=args.output_dir)
    except (RecoveryExecutionError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
