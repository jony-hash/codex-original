"""Execute the bounded Chen BOGT phase-completion repair task.

The task has one scientific gate and two ordered stages.  Stage 1 is CPU-only:
it constructs the preregistered Q0-Q4 phase-support candidates and applies the
within-pair permutation calibration.  Stage 2 is entered only after the
lowest-structure compatible candidate is frozen, and consumes the frozen P2
amplitude bank in one downstream direct-generator canary.

This executor deliberately imports the already accepted canary's metric and
optical helpers.  It does not modify the frozen training, OLS, Generator, or
evaluator implementations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1 as baseline
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import EPS_S, LightGen2DChenBOGTComplexOLSTrainingConsumer, OLS_PORT_COUNT, PRODUCTION_PROPAGATION_GRID, TRAIN_COUNT, WrappedConditionalPhaseModel, git_identity, max_q_gauge, statistical_variable, wrap_phase
from ldm_project.v2.application.research.shared.research_run_control import HeartbeatWriter, ResearchRunControl, ResearchRunIdentity, RUN_COMPLETED, RUN_PREPARING, RUNNING, capture_execution_source_manifest
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)


TASK_ID = "EXECUTE_CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1"
OUTER_GATE = "CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "chen-bogt-phase-completion-repair-v1"
TRAJECTORY_ID = "chen-bogt-phase-completion-repair-trajectory-v1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_PRODUCER_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"
# Historical import compatibility.  This is producer provenance, not a
# consumer exact-source lock under HASH_ON_TRUST_BOUNDARY_V2.
EXPECTED_HEAD = EXPECTED_PRODUCER_HEAD

SOURCE_RUN = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01")
BUNDLE_RELATIVE = SOURCE_RUN / "generation_ready_bundle.pt"
REPORT_RELATIVE = SOURCE_RUN / "report.json"
FINAL_COLLECTION_RELATIVE = SOURCE_RUN / "final_train_ols.pt"

ACCEPTED_REPAIRED_RUN = Path(
    "outputs/20260918_chen_bogt_repaired_prior_generation_canary_v1/runs/attempt_01"
)
ACCEPTED_REPAIRED_REPORT_RELATIVE = ACCEPTED_REPAIRED_RUN / "sampled_generation_canary_report.json"
ACCEPTED_REPAIRED_LATENT_RELATIVE = ACCEPTED_REPAIRED_RUN / "sampled_latent_bank.pt"
ACCEPTED_REPAIRED_REAL_CONTROL_RELATIVE = ACCEPTED_REPAIRED_RUN / "real_control_cmos.pt"
EXPECTED_ACCEPTED_REPAIRED_REPORT_SHA256 = "d013bcde80358ca93ad4b3cf7623b5a634de280281b2b9fab6309ec714a41e5c"
EXPECTED_ACCEPTED_REPAIRED_LATENT_SHA256 = "b22fe7162e653b2b9f816b79da36708b7e4d7ef6f3aa1b067ac3710a7d328e72"
EXPECTED_ACCEPTED_REPAIRED_REAL_CONTROL_SHA256 = "6fe1796c7b8feac98b155c96851400431c1f99cee2d25a9752bff73d2a925bc2"
ACCEPTED_REPAIRED_OUTER_GATE = "CHEN_BOGT_REPAIRED_PRIOR_GENERATION_CANARY_V1_RESOLVED"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260918_chen_bogt_phase_completion_repair_v1/runs/attempt_01"
)

N_REFERENCE = 1_000
N_OPTICAL = 200
OPTICAL_BATCH_SIZE = 10
CALIBRATION_PERMUTATIONS = 999
CALIBRATION_SEED = 20_260_928
DOWNSTREAM_PHASE_SEED = 20_260_919

# Q0 reproduces the accepted real-u phase diagnostic and Q3 uses the same
# random stream so the residual-mean intervention has a common-noise view.
CANDIDATE_SEEDS = {
    "Q0": 20_260_921,
    "Q1": 20_260_922,
    "Q2": 20_260_923,
    "Q3": 20_260_921,
}

PHASE_CANDIDATE_SCHEMA = "chen_bogt_phase_completion_candidate_bank.v1"
PHASE_SELECTION_SCHEMA = "chen_bogt_phase_completion_selection.v1"
REPORT_SCHEMA = "chen_bogt_phase_completion_repair_report.v1"
LATENT_SCHEMA = "chen_bogt_phase_completion_repair_latent_bank.v1"
PHASE_LOG_EPS = 1.0e-12
STANDARD_DEVIATION_FLOOR = 1.0e-12
CALIBRATION_BATCH_SIZE = 16
COMPLEX64_RELATIVE_ROUNDING_TOLERANCE = 16.0 * float(torch.finfo(torch.float32).eps)


class PhaseRepairError(RuntimeError):
    """Raised when a frozen boundary or valid-measurement invariant fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseRepairError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _json_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def _int_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(resolved)
    return {"path": relative, "sha256": file_sha256(resolved), "role": role}


def _bind_git_provenance(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Separate historical producer identity from the current consumer state."""

    return {
        "producer": {"branch": EXPECTED_BRANCH, "head": EXPECTED_PRODUCER_HEAD, "role": "historical source provenance"},
        "consumer": dict(identity),
        "consumer_head_exact_required": False,
        "required_consumer_head": None,
        "head_equal": identity.get("head") == EXPECTED_PRODUCER_HEAD,
        "producer_head_is_provenance_only": True,
        "semantics": "HASH_ON_TRUST_BOUNDARY_V2: historical producer/source HEAD is provenance unless an explicit consumer exact-source lock is consumed",
    }


def _phase_bank_check(
    phase: torch.Tensor,
    anchors: torch.Tensor,
    target_u: torch.Tensor,
    *,
    name: str,
) -> None:
    phase = phase.to(device="cpu", dtype=torch.float64)
    anchors = anchors.to(device="cpu", dtype=torch.long)
    target_u = target_u.to(device="cpu", dtype=torch.float64)
    _require(phase.shape == (N_REFERENCE, OLS_PORT_COUNT), f"{name} phase shape changed")
    _require(anchors.shape == (N_REFERENCE,), f"{name} anchor shape changed")
    _require(target_u.shape == (N_REFERENCE, OLS_PORT_COUNT), f"{name} target-u shape changed")
    _finite(phase, f"{name} phase")
    _finite(target_u, f"{name} target u")
    _require(bool((phase >= -math.pi).all().item()), f"{name} phase left wrapped lower range")
    _require(bool((phase < math.pi).all().item()), f"{name} phase left wrapped upper range")
    _require(bool(((anchors >= 0) & (anchors < OLS_PORT_COUNT)).all().item()), f"{name} anchors are invalid")
    _require(bool((target_u >= 0.0).all().item()), f"{name} target u left nonnegative support")


def _prepare_cuda_runtime_dll_search_path() -> dict[str, Any]:
    """Expose the supported environment's CUDA DLLs to this process only."""
    if os.name != "nt":
        return {"status": "not_applicable", "scope": "current_python_process_only"}
    cuda_build = torch.version.cuda
    _require(
        isinstance(cuda_build, str)
        and len(cuda_build.split(".")) == 2
        and all(part.isdigit() for part in cuda_build.split(".")),
        "CUDA build cannot identify the environment NVRTC DLL",
    )
    major, minor = cuda_build.split(".")
    environment_bin = (Path(sys.prefix) / "bin").resolve()
    dll_path = environment_bin / f"nvrtc-builtins64_{major}{minor}.dll"
    _require(environment_bin.is_dir() and dll_path.is_file(), f"required CUDA runtime DLL is unavailable: {dll_path}")
    current_path = os.environ.get("PATH", "")
    normalized_bin = os.path.normcase(os.path.normpath(str(environment_bin)))
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    already_present = any(os.path.normcase(os.path.normpath(entry)) == normalized_bin for entry in path_entries)
    if not already_present:
        os.environ["PATH"] = str(environment_bin) + (os.pathsep + current_path if current_path else "")
    return {
        "status": "already_present" if already_present else "prepended",
        "scope": "current_python_process_only",
        "cuda_build": cuda_build,
        "environment_bin": str(environment_bin),
        "expected_dll": dll_path.name,
        "expected_dll_exists": True,
        "path_changed": not already_present,
    }


def _target_anchors(u: torch.Tensor, *, name: str) -> torch.Tensor:
    _require(u.shape == (N_REFERENCE, OLS_PORT_COUNT), f"{name} u shape changed")
    _finite(u, f"{name} u")
    _require(bool((u >= 0.0).all().item()), f"{name} u left nonnegative support")
    maximum, anchors = u.max(dim=1)
    _require(bool((maximum > 0.0).all().item()), f"{name} has no positive max-q anchor")
    return anchors.to(torch.long)


def _phase_to_gauged(
    u: torch.Tensor,
    raw_phase: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require(raw_phase.shape == u.shape, "phase and u shapes differ")
    amplitude = ((scale + EPS_S) * u).clamp_min(0.0).sqrt()
    z = torch.complex(amplitude * raw_phase.cos(), amplitude * raw_phase.sin()).to(torch.complex64)
    gauged, _, anchors = max_q_gauge(z, scale)
    phase = wrap_phase(torch.angle(gauged).to(torch.float64)).cpu()
    return phase, anchors.to(torch.long).cpu()


def _current_phase_raw(
    u: torch.Tensor,
    phase_model: WrappedConditionalPhaseModel,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return phase_model.sample(u, generator=generator).to(device="cpu", dtype=torch.float64)


def _single_residual_mean_phase_raw(
    u: torch.Tensor,
    phase_model: WrappedConditionalPhaseModel,
    seed: int,
) -> torch.Tensor:
    """Q3: keep B/C/L fixed and add the fitted residual mean exactly once."""
    design = phase_model._design(u)
    predicted = design.to(torch.complex128) @ phase_model.coefficients
    _require(bool((predicted.abs() > 1.0e-12).all().item()), "Q3 conditional phase mean phasor is degenerate")
    mean = wrap_phase(torch.angle(predicted) + phase_model.residual_mean)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(u.shape, device="cpu", dtype=torch.float64, generator=generator)
    return wrap_phase(mean + noise @ phase_model.cholesky.T).to(torch.float64)


def _anchor_conditioned_marginal_phase(
    target_u: torch.Tensor,
    train_u: torch.Tensor,
    train_phase: torch.Tensor,
    train_anchors: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_anchors = _target_anchors(target_u, name="Q1 target")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    phase = torch.zeros_like(target_u, dtype=torch.float64)
    for anchor in torch.unique(target_anchors, sorted=True).tolist():
        target_rows = torch.nonzero(target_anchors == int(anchor), as_tuple=False).reshape(-1)
        donor_rows = torch.nonzero(train_anchors == int(anchor), as_tuple=False).reshape(-1)
        _require(donor_rows.numel() > 0, f"Q1 has no TRAIN donor rows for anchor {anchor}")
        for port in range(OLS_PORT_COUNT):
            if port == int(anchor):
                continue
            draws = torch.randint(
                donor_rows.numel(),
                (target_rows.numel(),),
                generator=generator,
                device="cpu",
            )
            phase[target_rows, port] = train_phase[donor_rows[draws], port]
        phase[target_rows, int(anchor)] = 0.0
    return wrap_phase(phase), target_anchors


def _anchor_conditioned_joint_phase(
    target_u: torch.Tensor,
    train_phase: torch.Tensor,
    train_anchors: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_anchors = _target_anchors(target_u, name="Q2 target")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    phase = torch.zeros_like(target_u, dtype=torch.float64)
    for anchor in torch.unique(target_anchors, sorted=True).tolist():
        target_rows = torch.nonzero(target_anchors == int(anchor), as_tuple=False).reshape(-1)
        donor_rows = torch.nonzero(train_anchors == int(anchor), as_tuple=False).reshape(-1)
        _require(donor_rows.numel() > 0, f"Q2 has no TRAIN donor rows for anchor {anchor}")
        draws = torch.randint(
            donor_rows.numel(),
            (target_rows.numel(),),
            generator=generator,
            device="cpu",
        )
        phase[target_rows] = train_phase[donor_rows[draws]]
        phase[target_rows, int(anchor)] = 0.0
    return wrap_phase(phase), target_anchors


def _construct_phase_candidate(
    candidate_id: str,
    target_u: torch.Tensor,
    train_u: torch.Tensor,
    train_phase: torch.Tensor,
    train_anchors: torch.Tensor,
    phase_model: WrappedConditionalPhaseModel,
    scale: torch.Tensor,
    *,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_seed = CANDIDATE_SEEDS[candidate_id] if seed is None else seed
    if candidate_id == "Q0":
        raw_phase = _current_phase_raw(target_u, phase_model, candidate_seed)
        target_anchors = _target_anchors(target_u, name="Q0 target")
    elif candidate_id == "Q1":
        raw_phase, target_anchors = _anchor_conditioned_marginal_phase(
            target_u,
            train_u,
            train_phase,
            train_anchors,
            seed=candidate_seed,
        )
    elif candidate_id == "Q2":
        raw_phase, target_anchors = _anchor_conditioned_joint_phase(
            target_u,
            train_phase,
            train_anchors,
            seed=candidate_seed,
        )
    elif candidate_id == "Q3":
        raw_phase = _single_residual_mean_phase_raw(target_u, phase_model, candidate_seed)
        target_anchors = _target_anchors(target_u, name="Q3 target")
    else:
        raise PhaseRepairError(f"unknown phase candidate: {candidate_id}")

    phase, gauged_anchors = _phase_to_gauged(target_u, raw_phase, scale)
    _require(torch.equal(gauged_anchors, target_anchors), f"{candidate_id} max-q anchor changed during construction")
    if candidate_id in {"Q1", "Q2"}:
        phase[torch.arange(N_REFERENCE), target_anchors] = 0.0
    phase = wrap_phase(phase)
    _phase_bank_check(phase, gauged_anchors, target_u, name=candidate_id)
    return phase, target_anchors, gauged_anchors


def _bilinear(mask_left: torch.Tensor, matrix: torch.Tensor, mask_right: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,ij,bj->b", mask_left, matrix, mask_right)


def _paired_swap_statistic(
    swaps: torch.Tensor,
    candidate_candidate: torch.Tensor,
    candidate_real: torch.Tensor,
    real_real: torch.Tensor,
) -> torch.Tensor:
    """Return the biased V-statistic ED for each paired label-swap replicate."""
    swaps = swaps.to(device="cpu", dtype=torch.bool)
    left_mask = (~swaps).to(torch.float64)
    right_mask = swaps.to(torch.float64)
    within_left = (
        _bilinear(left_mask, candidate_candidate, left_mask)
        + 2.0 * _bilinear(left_mask, candidate_real, right_mask)
        + _bilinear(right_mask, real_real, right_mask)
    )
    within_right = (
        _bilinear(left_mask, real_real, left_mask)
        + 2.0 * _bilinear(right_mask, candidate_real, left_mask)
        + _bilinear(right_mask, candidate_candidate, right_mask)
    )
    cross = (
        _bilinear(left_mask, candidate_real, left_mask)
        + _bilinear(left_mask, candidate_candidate, right_mask)
        + _bilinear(right_mask, real_real, left_mask)
        + _bilinear(right_mask, candidate_real.T, right_mask)
    )
    n_squared = float(swaps.shape[1] * swaps.shape[1])
    return (2.0 * cross - within_left - within_right).div(n_squared).clamp_min(0.0)


def _paired_swap_calibration(
    candidate: torch.Tensor,
    real_a: torch.Tensor,
    *,
    real_real_energy_distance: float,
) -> dict[str, Any]:
    candidate = candidate.to(device="cpu", dtype=torch.float64)
    real_a = real_a.to(device="cpu", dtype=torch.float64)
    _require(candidate.shape == real_a.shape == (N_REFERENCE, 2 * OLS_PORT_COUNT), "phase calibration views have incompatible shapes")
    candidate_candidate = torch.cdist(candidate, candidate, p=2)
    candidate_real = torch.cdist(candidate, real_a, p=2)
    real_real = torch.cdist(real_a, real_a, p=2)
    observed = _json_float(
        (2.0 * candidate_real.mean() - candidate_candidate.mean() - real_real.mean()).clamp_min(0.0)
    )

    generator = torch.Generator(device="cpu").manual_seed(CALIBRATION_SEED)
    permutation_values: list[torch.Tensor] = []
    for start in range(0, CALIBRATION_PERMUTATIONS, CALIBRATION_BATCH_SIZE):
        stop = min(start + CALIBRATION_BATCH_SIZE, CALIBRATION_PERMUTATIONS)
        swaps = torch.rand(
            (stop - start, N_REFERENCE),
            device="cpu",
            dtype=torch.float64,
            generator=generator,
        ) < 0.5
        permutation_values.append(
            _paired_swap_statistic(swaps, candidate_candidate, candidate_real, real_real)
        )
    permutation = torch.cat(permutation_values, dim=0)
    _finite(permutation, "phase permutation statistics")
    q95 = _json_float(torch.quantile(permutation, 0.95, interpolation="linear"))
    exceedance = int((permutation >= observed).sum().item())
    p_value = (1.0 + exceedance) / float(CALIBRATION_PERMUTATIONS + 1)
    percentile = 100.0 * float((permutation <= observed).to(torch.float64).mean().item())
    return {
        "raw_ed": observed,
        "real_a_vs_real_b_ed": real_real_energy_distance,
        "R_phi": observed / max(real_real_energy_distance, STANDARD_DEVIATION_FLOOR),
        "q95": q95,
        "permutation_percentile": percentile,
        "permutation_exceedance_count": exceedance,
        "permutation_count": CALIBRATION_PERMUTATIONS,
        "p_value": p_value,
        "compatible": bool(observed <= q95),
        "calibration": {
            "null": "within-pair independent candidate/real-A label swaps",
            "pair_count": N_REFERENCE,
            "seed": CALIBRATION_SEED,
            "q95_interpolation": "linear",
            "p_value_formula": "(1 + count(T_perm >= T_obs)) / (B + 1)",
            "embedding": "[cos(phi), sin(phi)] in R^200",
            "biased_v_statistic_diagonal_terms_retained": True,
        },
    }


def _load_accepted_repaired_authority(root: Path) -> dict[str, Any]:
    report_path = root / ACCEPTED_REPAIRED_REPORT_RELATIVE
    latent_path = root / ACCEPTED_REPAIRED_LATENT_RELATIVE
    control_path = root / ACCEPTED_REPAIRED_REAL_CONTROL_RELATIVE
    _require(report_path.is_file() and latent_path.is_file() and control_path.is_file(), "accepted repaired-prior authority is incomplete")
    report_sha = file_sha256(report_path)
    latent_sha = file_sha256(latent_path)
    control_sha = file_sha256(control_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema") == "chen_bogt_repaired_prior_generation_canary_report.v1", "accepted repaired-prior report schema changed")
    _require(report.get("status") == "VALID_COMPLETED", "accepted repaired-prior report is not valid completed")
    _require(report.get("outer_gate") == ACCEPTED_REPAIRED_OUTER_GATE, "accepted repaired-prior outer gate changed")
    _require(report.get("action_executed") == "EXECUTE_CHEN_BOGT_REPAIRED_PRIOR_GENERATION_CANARY_V1", "accepted repaired-prior task identity changed")
    artifacts = _mapping(report.get("artifacts"), "accepted repaired-prior artifacts")
    control_ref = _mapping(artifacts.get("real_control_cmos.pt"), "accepted repaired-prior real control ref")
    _require(control_ref.get("path") == ACCEPTED_REPAIRED_REAL_CONTROL_RELATIVE.as_posix(), "accepted repaired-prior control path changed")
    latent = safe_torch_load(latent_path, map_location="cpu")
    _require(isinstance(latent, Mapping), "accepted repaired-prior latent bank is not a mapping")
    _require(latent.get("schema") == "chen_bogt_repaired_prior_generation_latent_bank.v1", "accepted repaired-prior latent schema changed")
    _require(latent.get("selected_step") == 300, "accepted repaired-prior selected step changed")
    _require(_mapping(latent.get("seeds"), "accepted repaired-prior seeds").get("conditional_phase") == DOWNSTREAM_PHASE_SEED, "accepted repaired-prior phase seed changed")
    return {
        "report_path": report_path,
        "latent_path": latent_path,
        "control_path": control_path,
        "report_sha256": report_sha,
        "latent_sha256": latent_sha,
        "control_sha256": control_sha,
        "report": dict(report),
        "latent": dict(latent),
    }


def _load_cpu_context(root: Path) -> dict[str, Any]:
    identity = git_identity(root)
    git_provenance = _bind_git_provenance(identity)

    report_path = root / REPORT_RELATIVE
    bundle_path = root / BUNDLE_RELATIVE
    final_path = root / FINAL_COLLECTION_RELATIVE
    for path in (report_path, bundle_path, final_path):
        _require(path.is_file(), f"frozen input is missing: {path}")
    report_sha = file_sha256(report_path)
    bundle_sha = file_sha256(bundle_path)
    final_sha = file_sha256(final_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema") == "chen_bogt_complex_ols_bounded_training_report.v1", "training report schema changed")
    _require(report.get("result") == "VALID_COMPLETED_GENERATION_BUNDLE_READY", "training report is not the frozen bundle-ready result")
    bundle = safe_torch_load(bundle_path, map_location="cpu")
    final_payload = safe_torch_load(final_path, map_location="cpu")
    _require(isinstance(bundle, Mapping) and isinstance(final_payload, Mapping), "frozen bundle payload is invalid")
    _require(bundle.get("schema") == baseline.BUNDLE_SCHEMA, "generation bundle schema changed")
    _require(bundle.get("selected_step") == 300, "generation bundle selected step changed")
    _require(bundle.get("bundle_role") == "GENERATION_INPUTS_COMPLETE", "generation bundle role changed")
    _require(final_payload.get("schema") == baseline.FINAL_COLLECTION_SCHEMA, "final TRAIN OLS schema changed")
    _require(final_payload.get("selected_step") == 300, "final TRAIN OLS selected step changed")

    refs = _mapping(bundle.get("frozen_references"), "bundle frozen references")
    scale = _tensor(refs.get("scale"), "bundle scale").to(device="cpu", dtype=torch.float64)
    _require(scale.shape == (OLS_PORT_COUNT,), "bundle scale shape changed")
    _finite(scale, "bundle scale")
    _require(bool((scale > EPS_S).all().item()), "bundle scale is not strictly positive")
    phase_model = baseline._phase_model_from_bundle(_mapping(bundle.get("phase_model"), "bundle phase model"))
    _require(phase_model.fit_data_role == "train" and phase_model.fit_count == TRAIN_COUNT, "phase model TRAIN fit identity changed")
    _require(phase_model.engineering_only is False, "phase model engineering flag changed")

    z_final = _tensor(final_payload.get("z_final"), "final TRAIN z").to(device="cpu")
    u_final = _tensor(final_payload.get("u_final"), "final TRAIN u").to(device="cpu")
    _require(z_final.shape == (TRAIN_COUNT, OLS_PORT_COUNT) and z_final.dtype == torch.complex64, "final TRAIN z shape/dtype changed")
    _require(u_final.shape == (TRAIN_COUNT, OLS_PORT_COUNT) and u_final.dtype == torch.float64, "final TRAIN u shape/dtype changed")
    _finite(z_final, "final TRAIN z")
    _finite(u_final, "final TRAIN u")
    _require(bool((u_final >= 0.0).all().item()), "final TRAIN u left nonnegative support")
    _require(bool(torch.equal(statistical_variable(z_final, scale), u_final)), "final TRAIN u transform changed")

    historical_p0 = baseline._load_historical_p0_authority(root)
    p2_candidate = baseline._load_p2_candidate(root)
    accepted = _load_accepted_repaired_authority(root)
    accepted_u = _tensor(accepted["latent"].get("u_sample"), "accepted P2 u")
    _require(torch.equal(accepted_u, p2_candidate["u"]), "accepted P2 u differs from frozen P2 candidate")
    _require(accepted_u.shape == (N_REFERENCE, OLS_PORT_COUNT) and accepted_u.dtype == torch.float64, "accepted P2 u shape/dtype changed")

    refs_bank = baseline._reuse_reference_bank(z_final, u_final, scale, historical_p0)
    real_a_z, real_b_z = refs_bank["z"][:N_REFERENCE], refs_bank["z"][N_REFERENCE:]
    real_a_u, real_b_u = refs_bank["u"][:N_REFERENCE], refs_bank["u"][N_REFERENCE:]
    train_gauged, _, train_anchors = max_q_gauge(z_final, scale)
    train_phase = wrap_phase(torch.angle(train_gauged).to(torch.float64)).cpu()
    train_u_anchors = u_final.argmax(dim=1).to(torch.long)
    _require(torch.equal(train_anchors.cpu(), train_u_anchors), "TRAIN anchor identity changed")
    real_a_gauged, _, real_a_anchors = max_q_gauge(real_a_z, scale)
    real_b_gauged, _, real_b_anchors = max_q_gauge(real_b_z, scale)
    real_a_phase = wrap_phase(torch.angle(real_a_gauged).to(torch.float64)).cpu()
    real_b_phase = wrap_phase(torch.angle(real_b_gauged).to(torch.float64)).cpu()
    _finite(train_phase, "TRAIN phase support")

    accepted_metrics = {
        "R_u": float(baseline._nested(accepted["report"], "stage_a_prior_support", "joint_support", "R_u")),
        "R_phi_context": float(baseline._nested(accepted["report"], "stage_b_phase_completion", "comparison", "R_phi")),
        "R_z": float(baseline._nested(accepted["report"], "stage_c_full_complex_latent", "complex_comparison", "R_z")),
        "R_raw": float(baseline._nested(accepted["report"], "stage_d_e_raw_cmos", "raw_distribution_comparison", "R_raw")),
        "R_feature": float(baseline._nested(accepted["report"], "stage_f_frozen_evaluator", "comparison", "R_feature")),
    }
    _require(all(math.isfinite(value) for value in accepted_metrics.values()), "accepted P2 baseline metrics are nonfinite")
    return {
        "git_before": identity,
        "git_provenance": git_provenance,
        "report": report,
        "bundle": dict(bundle),
        "final_payload": dict(final_payload),
        "report_path": report_path,
        "bundle_path": bundle_path,
        "final_path": final_path,
        "scale": scale,
        "phase_model": phase_model,
        "z_final": z_final,
        "u_final": u_final,
        "train_phase": train_phase,
        "train_anchors": train_anchors.cpu(),
        "real_a_z": real_a_gauged.cpu(),
        "real_b_z": real_b_gauged.cpu(),
        "real_a_u": real_a_u.cpu(),
        "real_b_u": real_b_u.cpu(),
        "real_a_phase": real_a_phase,
        "real_b_phase": real_b_phase,
        "real_a_anchors": real_a_anchors.cpu(),
        "real_b_anchors": real_b_anchors.cpu(),
        "p2_candidate": p2_candidate,
        "historical_p0": historical_p0,
        "accepted": accepted,
        "accepted_metrics": accepted_metrics,
        "input_hashes": {
            "training_report": report_sha,
            "generation_bundle": bundle_sha,
            "final_train_ols": final_sha,
            "historical_p0_report": historical_p0["report_sha256"],
            "historical_p0_latent": historical_p0["latent_sha256"],
            "p2_candidate": p2_candidate["candidate_sha256"],
            "p2_manifest": p2_candidate["manifest_sha256"],
            "accepted_repaired_report": accepted["report_sha256"],
            "accepted_repaired_latent": accepted["latent_sha256"],
            "accepted_repaired_real_control": accepted["control_sha256"],
        },
    }


def _candidate_metadata(candidate_id: str, phase: torch.Tensor, target_anchors: torch.Tensor, gauged_anchors: torch.Tensor, target_u: torch.Tensor) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "schema": PHASE_CANDIDATE_SCHEMA,
        "source_authority": "final TRAIN gauged phase support and frozen phase-model payload",
        "seed": CANDIDATE_SEEDS[candidate_id],
        "target_u_shape": list(target_u.shape),
        "phase_shape": list(phase.shape),
        "phase_dtype": str(phase.dtype),
        "phase_range": "[-pi, pi)",
        "anchor_semantics": "max-q argmax over target u with lowest-index ties",
        "target_anchor_histogram": _int_list(torch.bincount(target_anchors, minlength=OLS_PORT_COUNT)),
        "gauged_anchor_histogram": _int_list(torch.bincount(gauged_anchors, minlength=OLS_PORT_COUNT)),
        "finite": True,
    }


def _write_candidate_artifact(
    target: Path,
    candidate_id: str,
    phase: torch.Tensor,
    target_anchors: torch.Tensor,
    gauged_anchors: torch.Tensor,
    target_u: torch.Tensor,
) -> Path:
    path = target / f"phase_candidate_{candidate_id}.pt"
    atomic_torch_save_new(
        path,
        {
            "schema": PHASE_CANDIDATE_SCHEMA,
            "candidate_id": candidate_id,
            "metadata": _candidate_metadata(candidate_id, phase, target_anchors, gauged_anchors, target_u),
            "phase": phase.cpu(),
            "target_anchors": target_anchors.cpu(),
            "gauged_anchors": gauged_anchors.cpu(),
        },
        artifact_kind=f"phase candidate {candidate_id}",
    )
    return path


def _make_identity(source_manifest: Mapping[str, Any]) -> ResearchRunIdentity:
    config = {
        "N_REFERENCE": N_REFERENCE,
        "N_OPTICAL": N_OPTICAL,
        "optical_batch_size": OPTICAL_BATCH_SIZE,
        "calibration_permutations": CALIBRATION_PERMUTATIONS,
        "calibration_seed": CALIBRATION_SEED,
        "candidate_seeds": dict(CANDIDATE_SEEDS),
        "downstream_phase_seed": DOWNSTREAM_PHASE_SEED,
        "selection_order": ["Q1", "Q2", "Q3"],
    }
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role="train_reference_ols_only",
        config_identity=config,
        model_identity="CHEN_BOGT_PHASE_COMPLETION_REPAIR_Q1_Q2_Q3_V1",
        optimizer_identity={"name": "none", "backward": False},
        sampler_identity={"candidate_domain": ["Q0", "Q1", "Q2", "Q3"], "control": "Q4"},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=SCIENTIFIC_ATTEMPT_ID,
    )


def _source_manifest(root: Path) -> dict[str, Any]:
    source_paths = (
        Path(__file__).resolve(),
        root / "scripts/research/lightgen/execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1.py",
        root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
        root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
        root / "src/ldm_project/v2/application/research/shared/research_run_control.py",
        root / "src/ldm_project/v2/student/evaluator.py",
    )
    manifest = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=source_paths,
        config_identity={
            "outer_gate": OUTER_GATE,
            "authorization_token": AUTHORIZATION_TOKEN,
            "candidate_seeds": CANDIDATE_SEEDS,
            "calibration_seed": CALIBRATION_SEED,
            "calibration_permutations": CALIBRATION_PERMUTATIONS,
        },
    )
    return manifest.to_dict()


def _paired_real_phase_controls(real_a_view: torch.Tensor, real_b_view: torch.Tensor) -> dict[str, Any]:
    real_real = baseline._energy_distance(real_a_view, real_b_view)
    identity = baseline._energy_distance(real_a_view, real_a_view)
    return {
        "identity_ceiling_ed_real_a_real_a": identity,
        "real_a_vs_real_b_ed": real_real,
        "estimator": "biased Euclidean energy-distance V-statistic with diagonal terms retained",
    }


def _load_optical_context(root: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """Load the frozen CUDA consumer without applying the prior canary's doc guard."""
    cuda_runtime = _prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available(), "CUDA is required for the selected downstream canary")
    _require(torch.cuda.device_count() >= 1, "no CUDA device is available")
    checkpoint, evaluator_identity = baseline._load_evaluator_identity(root, context["report"])
    device = torch.device("cuda:0")
    consumer = LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device=device,
        propagation_grid_size=PRODUCTION_PROPAGATION_GRID,
    )
    ols = _mapping(context["bundle"].get("ols"), "bundle OLS")
    _require(consumer.runtime_identity() == ols.get("interface"), "current runtime identity differs from frozen bundle runtime")
    state_container = _mapping(context["bundle"].get("final_encoder_generator"), "bundle final state")
    state = _mapping(state_container.get("state"), "bundle final trainable state")
    consumer.load_trainable_state_dict(state)
    consumer.runtime.eval()
    for parameter in consumer.parameters():
        parameter.requires_grad_(False)
    _require(all(not parameter.requires_grad for parameter in consumer.parameters()), "selected canary runtime parameter freeze failed")
    return {
        "consumer": consumer,
        "evaluator_checkpoint": checkpoint,
        "evaluator_identity": evaluator_identity,
        "cuda_runtime": cuda_runtime,
    }


def _run_stage_one(
    *,
    context: Mapping[str, Any],
    target: Path,
    heartbeat: HeartbeatWriter,
) -> dict[str, Any]:
    target_u = context["real_a_u"]
    target_view = baseline._phase_embedding(context["real_a_phase"])
    real_b_view = baseline._phase_embedding(context["real_b_phase"])
    real_real_ed = baseline._energy_distance(target_view, real_b_view)
    controls = _paired_real_phase_controls(target_view, real_b_view)

    candidate_results: dict[str, Any] = {}
    candidate_paths: dict[str, Path] = {}
    candidate_phases: dict[str, torch.Tensor] = {}
    candidate_anchors: dict[str, torch.Tensor] = {}
    for index, candidate_id in enumerate(("Q0", "Q1", "Q2", "Q3"), start=1):
        heartbeat.update(
            state=RUNNING,
            phase="CALIBRATION",
            step=index,
            latest_losses={"candidate": candidate_id, "stage": "CONSTRUCT"},
            force=True,
        )
        phase, target_anchors, gauged_anchors = _construct_phase_candidate(
            candidate_id,
            target_u,
            context["u_final"],
            context["train_phase"],
            context["train_anchors"],
            context["phase_model"],
            context["scale"],
        )
        candidate_phases[candidate_id] = phase
        candidate_anchors[candidate_id] = gauged_anchors
        candidate_paths[candidate_id] = _write_candidate_artifact(
            target,
            candidate_id,
            phase,
            target_anchors,
            gauged_anchors,
            target_u,
        )
        calibration = _paired_swap_calibration(
            baseline._phase_embedding(phase),
            target_view,
            real_real_energy_distance=real_real_ed,
        )
        candidate_results[candidate_id] = {
            "candidate_id": candidate_id,
            "mechanism": {
                "Q0": "current double residual-mean sampling semantics; baseline only",
                "Q1": "anchor-conditioned circular marginals; independent non-anchor ports",
                "Q2": "anchor-conditioned joint TRAIN phase-row sampling",
                "Q3": "same B/C/L with residual mean added exactly once",
            }[candidate_id],
            "artifact": _artifact_ref(root=target.parents[3], path=candidate_paths[candidate_id], role=f"frozen phase candidate {candidate_id}"),
            "phase_diagnostics": baseline._phase_diagnostics(phase, gauged_anchors, name=f"{candidate_id}_real_a_u"),
            "compatibility": calibration,
        }
        heartbeat.update(
            state=RUNNING,
            phase="CALIBRATION",
            step=index,
            latest_losses={"candidate": candidate_id, "stage": "CALIBRATED", "compatible": calibration["compatible"]},
            force=True,
        )

    q0 = candidate_results["Q0"]["compatibility"]
    q3 = candidate_results["Q3"]["compatibility"]
    candidate_results["Q0_vs_Q3"] = {
        "Q0": q0,
        "Q3": q3,
        "absolute_ed_delta_Q3_minus_Q0": q3["raw_ed"] - q0["raw_ed"],
        "R_phi_ratio_Q3_over_Q0": q3["R_phi"] / max(q0["R_phi"], STANDARD_DEVIATION_FLOOR),
        "common_noise_seed": CANDIDATE_SEEDS["Q0"],
        "interpretation": "Q0 to Q3 isolates the second residual-mean contribution while keeping B/C/L and target rows fixed; Q0/Q3 are also re-gauged by the unchanged max-q construction.",
    }
    compatible_candidates = [candidate_id for candidate_id in ("Q1", "Q2", "Q3") if candidate_results[candidate_id]["compatibility"]["compatible"]]
    selected = compatible_candidates[0] if compatible_candidates else None
    selection = {
        "schema": PHASE_SELECTION_SCHEMA,
        "selection_order": ["Q1", "Q2", "Q3"],
        "compatible_candidates_in_order": compatible_candidates,
        "selected_candidate": selected,
        "rule": "select the first compatible candidate in Q1 -> Q2 -> Q3; Q0 baseline only and Q4 control only",
        "claim_boundary": "lowest-structure candidate in the preregistered order that was not rejected by the frozen paired finite-sample phase-support test",
    }
    atomic_write_new_json(target / "phase_selection.json", selection, artifact_kind="phase candidate selection")
    stage_one = {
        "status": "VALID_COMPLETED",
        "phase_candidates": candidate_results,
        "Q4_control": controls,
        "selection": selection,
        "target": {
            "source": "accepted real-A u reference",
            "shape": list(target_u.shape),
            "dtype": str(target_u.dtype),
            "sha256": tensor_sha256(target_u),
            "paired_real_a_rows": True,
        },
        "real_phase_support": {
            "real_a": baseline._phase_diagnostics(context["real_a_phase"], context["real_a_anchors"], name="real_A"),
            "real_b": baseline._phase_diagnostics(context["real_b_phase"], context["real_b_anchors"], name="real_B"),
        },
        "candidate_seeds": dict(CANDIDATE_SEEDS),
        "calibration_seed": CALIBRATION_SEED,
        "protected_test_sealed_reads": 0,
    }
    atomic_write_new_json(target / "stage1_phase_support_report.json", stage_one, artifact_kind="stage 1 phase report")
    return {
        "report": stage_one,
        "candidate_paths": candidate_paths,
        "candidate_phases": candidate_phases,
        "candidate_anchors": candidate_anchors,
        "selected": selected,
    }


def _load_stage_one_from_disk(
    *,
    root: Path,
    target: Path,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = target / "stage1_phase_support_report.json"
    selection_path = target / "phase_selection.json"
    _require(report_path.is_file() and selection_path.is_file(), "resume stage-1 artifacts are incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    _require(report.get("status") == "VALID_COMPLETED", "resume stage-1 report is not complete")
    _require(report.get("selection") == selection, "resume stage-1 selection differs from report")
    target_u = context["real_a_u"]
    candidate_paths: dict[str, Path] = {}
    candidate_phases: dict[str, torch.Tensor] = {}
    candidate_anchors: dict[str, torch.Tensor] = {}
    for candidate_id in ("Q0", "Q1", "Q2", "Q3"):
        path = target / f"phase_candidate_{candidate_id}.pt"
        _require(path.is_file(), f"resume phase candidate is missing: {candidate_id}")
        payload = safe_torch_load(path, map_location="cpu")
        _require(isinstance(payload, Mapping), f"resume phase candidate is invalid: {candidate_id}")
        _require(payload.get("schema") == PHASE_CANDIDATE_SCHEMA, f"resume phase candidate schema changed: {candidate_id}")
        _require(payload.get("candidate_id") == candidate_id, f"resume phase candidate identity changed: {candidate_id}")
        phase = _tensor(payload.get("phase"), f"resume {candidate_id} phase").to(device="cpu", dtype=torch.float64)
        target_anchors = _tensor(payload.get("target_anchors"), f"resume {candidate_id} target anchors").to(device="cpu", dtype=torch.long)
        anchors = _tensor(payload.get("gauged_anchors"), f"resume {candidate_id} anchors").to(device="cpu", dtype=torch.long)
        expected_phase, expected_target_anchors, expected_anchors = _construct_phase_candidate(
            candidate_id,
            target_u,
            context["u_final"],
            context["train_phase"],
            context["train_anchors"],
            context["phase_model"],
            context["scale"],
        )
        _require(torch.equal(phase, expected_phase), f"resume phase candidate differs from its loaded TRAIN inputs: {candidate_id}")
        _require(torch.equal(target_anchors, expected_target_anchors), f"resume phase target rows changed: {candidate_id}")
        _require(torch.equal(anchors, expected_anchors), f"resume phase anchors changed: {candidate_id}")
        _phase_bank_check(phase, anchors, target_u, name=f"resume {candidate_id}")
        candidate_paths[candidate_id] = path
        candidate_phases[candidate_id] = phase
        candidate_anchors[candidate_id] = anchors
    return {
        "report": report,
        "candidate_paths": candidate_paths,
        "candidate_phases": candidate_phases,
        "candidate_anchors": candidate_anchors,
        "selected": selection.get("selected_candidate"),
    }


def _evaluate_selected_optical(
    *,
    consumer: Any,
    sampled_z: torch.Tensor,
    heartbeat: HeartbeatWriter,
) -> torch.Tensor:
    _require(sampled_z.shape[0] >= N_OPTICAL, "selected optical bank is too short")
    output: list[torch.Tensor] = []
    for start in range(0, N_OPTICAL, OPTICAL_BATCH_SIZE):
        stop = min(start + OPTICAL_BATCH_SIZE, N_OPTICAL)
        with torch.no_grad():
            result = consumer.generated_forward(sampled_z[start:stop].to(consumer.device))
            raw = result.raw_intensity.detach().cpu()
        baseline._finite(raw, "selected candidate raw CMOS")
        output.append(raw)
        heartbeat.update(
            state=RUNNING,
            phase="POST_TRAIN_COLLECTION",
            step=stop,
            latest_losses={"canary_phase": "SELECTED_CANDIDATE_GENERATED", "completed_optical": stop},
            force=True,
        )
    return torch.cat(output, dim=0)


def _selected_downstream(
    *,
    root: Path,
    target: Path,
    context: Mapping[str, Any],
    stage_one: Mapping[str, Any],
    selected: str,
    heartbeat: HeartbeatWriter,
) -> dict[str, Any]:
    # This is the only point where the existing CUDA consumer is loaded.  The
    # CPU candidate/compatibility stage has already frozen `selected`.
    optical_context = _load_optical_context(root, context)
    selected_phase_raw, _, _ = _construct_phase_candidate(
        selected,
        context["p2_candidate"]["u"],
        context["u_final"],
        context["train_phase"],
        context["train_anchors"],
        context["phase_model"],
        context["scale"],
        seed=DOWNSTREAM_PHASE_SEED,
    )
    selected_phase, selected_anchors = _phase_to_gauged(
        context["p2_candidate"]["u"],
        selected_phase_raw,
        context["scale"],
    )
    _phase_bank_check(selected_phase, selected_anchors, context["p2_candidate"]["u"], name=f"selected {selected} P2")
    selected_z_complex = torch.complex(
        ((context["scale"] + EPS_S) * context["p2_candidate"]["u"]).clamp_min(0.0).sqrt() * selected_phase.cos(),
        ((context["scale"] + EPS_S) * context["p2_candidate"]["u"]).clamp_min(0.0).sqrt() * selected_phase.sin(),
    ).to(torch.complex64)
    selected_z, _, selected_anchors_after = max_q_gauge(selected_z_complex, context["scale"])
    selected_z = selected_z.cpu()
    selected_phase = wrap_phase(torch.angle(selected_z).to(torch.float64)).cpu()
    _require(torch.equal(selected_anchors_after.cpu(), selected_anchors), "selected P2 anchor changed after complex construction")
    magnitude_error = (statistical_variable(selected_z, context["scale"]) - context["p2_candidate"]["u"]).abs()
    relative_magnitude_error = magnitude_error / context["p2_candidate"]["u"].clamp_min(STANDARD_DEVIATION_FLOOR)
    _require(
        _json_float(relative_magnitude_error.max()) <= COMPLEX64_RELATIVE_ROUNDING_TOLERANCE,
        "selected phase changed P2 amplitude support beyond complex64 roundoff",
    )

    accepted_control = baseline._load_existing_cmos(
        context["accepted"]["control_path"],
        group="real_control",
        expected_z=context["real_a_z"][:N_OPTICAL],
    )
    # The accepted repaired report's control is the exact same historical
    # real-A control; the path/hash check above and payload z check establish
    # exact reuse without a second real-control forward.
    generated_raw = _evaluate_selected_optical(
        consumer=optical_context["consumer"],
        sampled_z=selected_z,
        heartbeat=heartbeat,
    )
    generated_stats = baseline._raw_cmos_statistics(generated_raw, name="selected_generated")
    real_control_stats = baseline._raw_cmos_statistics(accepted_control, name="reused_real_control")

    selected_phase_comparison = baseline._phase_distribution_comparison(
        selected_phase,
        context["real_a_phase"],
        context["real_b_phase"],
        selected_anchors,
        context["real_a_anchors"],
        context["real_b_anchors"],
    )
    complex_stage, _, _ = baseline._complex_comparison(
        selected_z,
        context["real_a_z"],
        context["real_b_z"],
        context["z_final"],
    )

    real_raw_view = accepted_control.flatten(1).to(torch.float64)
    generated_raw_view = generated_raw.flatten(1).to(torch.float64)
    raw_mean = real_raw_view.mean(dim=0)
    raw_std = real_raw_view.std(dim=0, unbiased=False).clamp_min(STANDARD_DEVIATION_FLOOR)
    raw_cross = baseline._energy_distance(
        baseline._standardize(generated_raw_view, raw_mean, raw_std),
        baseline._standardize(real_raw_view, raw_mean, raw_std),
    )
    raw_self = baseline._energy_distance(
        baseline._standardize(real_raw_view[:100], raw_mean, raw_std),
        baseline._standardize(real_raw_view[100:], raw_mean, raw_std),
    )
    optical_report = {
        "N_OPTICAL": N_OPTICAL,
        "batch_size": OPTICAL_BATCH_SIZE,
        "input": "frozen P2 amplitude u -> selected phase -> same max-q gauge -> T=I_100/W_OLS -> frozen Generator -> raw CMOS",
        "generated": generated_stats,
        "real_control": real_control_stats,
        "real_control_reuse": {
            "exact_reuse": True,
            "path": _artifact_ref(root, context["accepted"]["control_path"], role="accepted exact real-control CMOS reuse"),
            "re_forwarded": False,
        },
        "raw_distribution_comparison": {
            "standardization": "per-pixel mean/std from the reused 200 real-control raw CMOS frames",
            "energy_distance_generated_vs_real_control": raw_cross,
            "energy_distance_real_control_first100_vs_second100": raw_self,
            "R_raw": raw_cross / max(raw_self, STANDARD_DEVIATION_FLOOR),
        },
        "forbidden_paths_used": {
            "encoder": False,
            "labels": False,
            "electronic_decoder": False,
            "adapter": False,
            "phase_repair_after_construction": False,
            "posthoc_optimization": False,
        },
    }
    evaluator_report, evaluator_outputs = baseline._evaluate_frozen_semantics(
        checkpoint=optical_context["evaluator_checkpoint"],
        generated_raw=generated_raw,
        real_raw=accepted_control,
    )
    diversity = {
        "generated_latent": baseline._diversity_view(selected_z, name="selected_generated_latent_z"),
        "real_control_latent": baseline._diversity_view(context["real_a_z"][:N_OPTICAL], name="reused_real_control_latent_z"),
        "generated_raw": baseline._diversity_view(generated_raw, name="selected_generated_raw_cmos"),
        "real_control_raw": baseline._diversity_view(accepted_control, name="reused_real_control_raw_cmos"),
        "generated_feature": baseline._diversity_view(evaluator_outputs["generated_embedding"], name="selected_generated_evaluator_feature"),
        "real_control_feature": baseline._diversity_view(evaluator_outputs["real_control_embedding"], name="reused_real_control_evaluator_feature"),
    }

    latent_path = target / "selected_phase_completion_latent_bank.pt"
    atomic_torch_save_new(
        latent_path,
        {
            "schema": LATENT_SCHEMA,
            "candidate_id": selected,
            "selected_step": 300,
            "phase_seed": DOWNSTREAM_PHASE_SEED,
            "candidate_artifact_sha256": file_sha256(stage_one["candidate_paths"][selected]),
            "p2_candidate_sha256": context["p2_candidate"]["candidate_sha256"],
            "p2_manifest_sha256": context["p2_candidate"]["manifest_sha256"],
            "u_sample": context["p2_candidate"]["u"].cpu(),
            "phase_sample": selected_phase,
            "z_sample": selected_z,
            "real_u_a": context["real_a_u"],
            "real_u_b": context["real_b_u"],
            "real_z_a_gauged": context["real_a_z"],
            "real_z_b_gauged": context["real_b_z"],
        },
        artifact_kind="selected phase completion latent bank",
    )
    generated_cmos_path = target / "selected_generated_cmos_canary.pt"
    baseline._write_cmos_artifact(generated_cmos_path, generated_raw, group="generated", z=selected_z[:N_OPTICAL])
    contact_sheet = baseline._make_contact_sheet(generated_raw, target / "selected_generated_contact_sheet.png")

    selected_metrics = {
        "R_u": context["accepted_metrics"]["R_u"],
        "R_phi": float(selected_phase_comparison["R_phi"]),
        "R_z": float(complex_stage["R_z"]),
        "R_raw": float(optical_report["raw_distribution_comparison"]["R_raw"]),
        "R_feature": float(evaluator_report["comparison"]["R_feature"]),
    }
    baseline_metrics = {
        "R_u": context["accepted_metrics"]["R_u"],
        "R_phi": context["accepted_metrics"]["R_phi_context"],
        "R_z": context["accepted_metrics"]["R_z"],
        "R_raw": context["accepted_metrics"]["R_raw"],
        "R_feature": context["accepted_metrics"]["R_feature"],
    }
    deltas = {
        name: selected_metrics[name] - baseline_metrics[name]
        for name in ("R_z", "R_raw", "R_feature")
    }
    ratios = {
        name: selected_metrics[name] / max(baseline_metrics[name], STANDARD_DEVIATION_FLOOR)
        for name in ("R_z", "R_raw", "R_feature")
    }
    downstream_branch = (
        "DOWNSTREAM_DIRECTIONALLY_IMPROVED"
        if all(deltas[name] < 0.0 for name in ("R_z", "R_raw", "R_feature"))
        else "DOWNSTREAM_MIXED"
    )
    return {
        "status": "VALID_COMPLETED",
        "selected_candidate": selected,
        "candidate_phase_on_P2": {
            "shape": list(selected_phase.shape),
            "dtype": str(selected_phase.dtype),
            "finite": True,
            "wrapped": True,
            "anchor_semantics": "same max-q gauge after complex-z construction",
            "phase_support": selected_phase_comparison,
        },
        "latent": {
            "artifact": _artifact_ref(root, latent_path, role="selected phase-completion latent bank"),
            "magnitude_preservation": {
                "max_abs_error_u": _json_float(magnitude_error.max()),
                "mean_abs_error_u": _json_float(magnitude_error.mean()),
                "max_relative_error_u": _json_float(relative_magnitude_error.max()),
            },
            "complex_comparison": complex_stage,
        },
        "optical": {
            **optical_report,
            "generated_artifact": _artifact_ref(root, generated_cmos_path, role="selected generated CMOS canary"),
        },
        "evaluator": evaluator_report,
        "diversity": diversity,
        "contact_sheet": contact_sheet,
        "metrics": {
            "selected": selected_metrics,
            "frozen_P2_Q0_baseline": baseline_metrics,
            "absolute_delta_selected_minus_P2_Q0": deltas,
            "ratio_selected_over_P2_Q0": ratios,
        },
        "classification": downstream_branch,
        "claim_boundary": "under frozen P2 amplitude support and the current frozen optical model, the selected phase-completion repair changes the specified complex-latent and downstream CMOS/feature diagnostics relative to the current Q0 producer; this is not a true phase-law, generation-acceptance, physical-validity, or promotion claim",
        "data_reads": {"raw_images": 0, "labels": 0, "protected": 0, "test": 0, "sealed": 0},
    }


def run_repair(output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    root = _repo_root()
    context = _load_cpu_context(root)
    target = output_dir if output_dir.is_absolute() else root / output_dir
    target = target.resolve()
    _require(root.resolve() in target.parents, "phase repair output must remain inside the repository")
    _require(target.relative_to(root).as_posix().startswith("outputs/"), "phase repair output must remain below outputs/")
    if resume:
        _require(target.is_dir(), f"cannot resume missing phase repair output: {target}")
        _require(not (target / "phase_completion_repair_report.json").exists(), "completed phase repair report already exists")
        stage_one = _load_stage_one_from_disk(root=root, target=target, context=context)
        repair_index = 1
        while (
            target / f"source_manifest_repaired_{repair_index:02d}.json"
        ).exists() or (target / f"execution_manifest_repaired_{repair_index:02d}.json").exists():
            repair_index += 1
        source_manifest_path = target / f"source_manifest_repaired_{repair_index:02d}.json"
        execution_manifest_path = target / f"execution_manifest_repaired_{repair_index:02d}.json"
    else:
        _require(not target.exists(), f"phase repair output already exists: {target}")
        target.mkdir(parents=True, exist_ok=False)
        source_manifest_path = target / "source_manifest.json"
        execution_manifest_path = target / "execution_manifest.json"

    source_manifest = _source_manifest(root)
    identity = _make_identity(source_manifest)
    atomic_write_new_json(source_manifest_path, source_manifest, artifact_kind="phase repair source manifest")
    atomic_write_new_json(
        execution_manifest_path,
        {
            "schema": "chen_bogt_phase_completion_repair_execution_manifest.v1",
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "authorization_token": AUTHORIZATION_TOKEN,
            "authorization_id": AUTHORIZATION_ID,
            "run_identity": identity.to_dict(),
            "source_manifest": source_manifest,
            "input_hashes": context["input_hashes"],
            "scope": {
                "stage_1": "CPU Q0/Q1/Q2/Q3/Q4 plus paired permutation calibration",
                "stage_2": "one selected candidate on frozen P2 with 200 generated direct-generator frames",
                "real_control": "exact reuse of accepted repaired-prior real-control CMOS",
                "training": False,
                "optimizer": False,
                "promotion": False,
            },
        },
        artifact_kind="phase repair execution manifest",
    )
    heartbeat = HeartbeatWriter(target, RUN_ID, total_steps=4 + N_OPTICAL, interval_seconds=60.0)
    heartbeat.update(state=RUN_PREPARING, phase="STATISTICS", step=0, latest_losses={"stage": "CPU_PREFLIGHT_READY"}, force=True)
    control = ResearchRunControl(identity, AUTHORIZATION_ID)
    cpu_inputs_valid = bool(
        context.get("report", {}).get("result") == "VALID_COMPLETED_GENERATION_BUNDLE_READY"
        and context.get("bundle", {}).get("selected_step") == 300
        and context.get("final_payload", {}).get("selected_step") == 300
        and tuple(context.get("z_final", torch.empty(0)).shape) == (TRAIN_COUNT, OLS_PORT_COUNT)
        and tuple(context.get("u_final", torch.empty(0)).shape) == (TRAIN_COUNT, OLS_PORT_COUNT)
        and context.get("phase_model") is not None
    )
    cpu_execution_valid = bool(cpu_inputs_valid and context["z_final"].device.type == "cpu" and context["u_final"].device.type == "cpu")
    receipt = control.make_ready_receipt(
        correctness={
            "passed": cpu_inputs_valid,
            "frozen_inputs_validated": cpu_inputs_valid,
            "data_reads": {"raw_images": 0, "labels": 0, "protected": 0, "test": 0, "sealed": 0},
        },
        resource={"passed": cpu_execution_valid, "phase": "CPU_PHASE_SUPPORT_ANALYSIS", "cuda_allocations_before_stage_two": 0},
        implementation={
            "passed": cpu_execution_valid,
            "selected_engine": "phase_completion_repair_cpu_support_and_frozen_cuda_diagnostics",
            "training": False,
            "optimizer": False,
        },
        selected_engine="phase_completion_repair_cpu_support_and_frozen_cuda_diagnostics",
    )
    control.mark_preflight_ready(receipt)
    control.start_execution()
    control.open_scientific_attempt(target, SCIENTIFIC_ATTEMPT_ID, receipt, resume=True)

    heartbeat.update(state=RUNNING, phase="STATISTICS", step=0, latest_losses={"stage": "STAGE_1_CPU_PHASE_SUPPORT"}, force=True)
    if not resume:
        stage_one = _run_stage_one(context=context, target=target, heartbeat=heartbeat)
    else:
        heartbeat.update(state=RUNNING, phase="CALIBRATION", step=4, latest_losses={"stage": "RESUMED_FROZEN_STAGE_1"}, force=True)
    selected = stage_one["selected"]
    downstream: dict[str, Any] | None = None
    if selected is not None:
        heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=0, latest_losses={"stage": "STAGE_2_SELECTED_CANDIDATE"}, force=True)
        downstream = _selected_downstream(
            root=root,
            target=target,
            context=context,
            stage_one=stage_one,
            selected=selected,
            heartbeat=heartbeat,
        )
    else:
        heartbeat.update(state=RUNNING, phase="FINALIZING", step=4, latest_losses={"stage": "NO_COMPATIBLE_PHASE_CANDIDATE"}, force=True)

    control.complete()
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=4 + (N_OPTICAL if downstream is not None else 0), latest_losses={"stage": "COMPLETED"}, force=True)
    status = "VALID_COMPLETED" if downstream is not None else "VALID_FAILED"
    classification = (
        downstream["classification"]
        if downstream is not None
        else "NO_MINIMAL_PHASE_SUPPORT_REPAIR_SUPPORTED"
    )
    artifacts = {
        "source_manifest": _artifact_ref(root, source_manifest_path, role="execution source manifest"),
        "execution_manifest": _artifact_ref(root, execution_manifest_path, role="execution manifest"),
        "stage1_phase_support_report": _artifact_ref(root, target / "stage1_phase_support_report.json", role="stage 1 phase support report"),
        "phase_selection": _artifact_ref(root, target / "phase_selection.json", role="selected phase candidate decision"),
        **{
            f"phase_candidate_{candidate_id}": _artifact_ref(root, path, role=f"frozen phase candidate {candidate_id}")
            for candidate_id, path in stage_one["candidate_paths"].items()
        },
    }
    if downstream is not None:
        artifacts.update(
            {
                "selected_phase_completion_latent_bank": downstream["latent"]["artifact"],
                "selected_generated_cmos_canary": downstream["optical"]["generated_artifact"],
                "selected_generated_contact_sheet": _artifact_ref(root, target / "selected_generated_contact_sheet.png", role="selected generated contact sheet"),
                "reused_real_control_cmos": _artifact_ref(root, context["accepted"]["control_path"], role="accepted exact real-control CMOS reuse"),
            }
        )

    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "status": status,
        "classification": classification,
        "project_mainline": SEMANTIC_AUTHORITY_ID,
        "authorization_consumed": AUTHORIZATION_TOKEN,
        "authorization_id": AUTHORIZATION_ID,
        "implementation_delta": "one bounded phase-completion executor; frozen training/OLS/Generator/evaluator sources unchanged",
        "engineering_repair": (
            {
                "resumed_same_attempt": True,
                "reason": "pre-optical harness corrections; Stage 1 artifacts were frozen and reused",
                "previous_source_manifest": _artifact_ref(root, target / "source_manifest.json", role="pre-repair source manifest"),
            }
            if resume
            else None
        ),
        "git": {
            "branch": context["git_before"]["branch"],
            "head": context["git_before"]["head"],
            "dirty_before": context["git_before"]["worktree_dirty"],
            "dirty_after": git_identity(root)["worktree_dirty"],
        },
        "input_authorities": {
            "training_report": _artifact_ref(root, context["report_path"], role="frozen upstream training report"),
            "generation_ready_bundle": _artifact_ref(root, context["bundle_path"], role="frozen generation-ready bundle"),
            "final_train_ols": _artifact_ref(root, context["final_path"], role="frozen final TRAIN OLS collection"),
            "historical_p0_report": _artifact_ref(root, context["historical_p0"]["report_path"], role="accepted real phase reference authority"),
            "historical_p0_latent": _artifact_ref(root, context["historical_p0"]["latent_path"], role="accepted real phase reference bank"),
            "p2_candidate": _artifact_ref(root, context["p2_candidate"]["candidate_path"], role="frozen P2 amplitude candidate"),
            "p2_manifest": _artifact_ref(root, context["p2_candidate"]["manifest_path"], role="frozen P2 amplitude manifest"),
            "accepted_repaired_report": _artifact_ref(root, context["accepted"]["report_path"], role="accepted P2+Q0 canary report"),
            "accepted_repaired_latent": _artifact_ref(root, context["accepted"]["latent_path"], role="accepted P2+Q0 latent bank"),
            "accepted_repaired_real_control": _artifact_ref(root, context["accepted"]["control_path"], role="accepted exact real-control CMOS"),
        },
        "run_identity": identity.to_dict(),
        "source_manifest": source_manifest,
        "preflight": {
            "status": "READY",
            "cpu_phase_stage_before_cuda": True,
            "phase_model_identity": True,
            "real_A_real_B_identity": True,
            "p2_identity": True,
            "p2_shape": [N_REFERENCE, OLS_PORT_COUNT],
            "p2_dtype": "torch.float64",
            "p2_refit": False,
            "p2_resampling": False,
            "phase_candidate_schema": PHASE_CANDIDATE_SCHEMA,
            "anchor_identity": True,
            "circular_embedding_unchanged": True,
            "paired_swap_permutation_deterministic": True,
            "no_protected_test_sealed_reads": True,
        },
        "stage_1_cpu_phase_support": stage_one["report"],
        "stage_2_downstream_propagation": downstream,
        "frozen_P2_Q0_baseline": {
            "metrics": context["accepted_metrics"],
            "report_sha256": context["accepted"]["report_sha256"],
            "latent_sha256": context["accepted"]["latent_sha256"],
            "real_control_sha256": context["accepted"]["control_sha256"],
        },
        "data_reads": {
            "saved_final_train_ols_rows_available": TRAIN_COUNT,
            "saved_final_train_ols_rows_selected": 2 * N_REFERENCE,
            "raw_images": 0,
            "labels": 0,
            "protected": 0,
            "test": 0,
            "sealed": 0,
        },
        "claim_boundary": (
            "Stage 1: the selected candidate is the lowest-structure candidate in the preregistered Q1->Q2->Q3 order not rejected by the frozen finite-sample phase-support test. "
            "Stage 2: under frozen P2 amplitude support and the frozen optical model, the selected candidate's specified downstream diagnostics are reported relative to P2+Q0."
            if selected is not None
            else "The complete Stage 1 measurement found no candidate compatible under the preregistered Q1->Q2->Q3 rule; no downstream claim is made."
        ),
        "no_automatic_follow_up": True,
        "artifacts": artifacts,
    }
    atomic_write_new_json(target / "phase_completion_repair_report.json", report, artifact_kind="phase completion repair report")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute the explicitly authorized phase repair task")
    parser.add_argument("--resume", action="store_true", help="resume the same attempt after an engineering pre-optical stop")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required")
    try:
        report = run_repair(args.output_dir, resume=args.resume)
    except (PhaseRepairError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "outer_gate": report["outer_gate"],
                "classification": report["classification"],
                "selected_candidate": report["stage_1_cpu_phase_support"]["selection"]["selected_candidate"],
                "report": str((_repo_root() / args.output_dir if not args.output_dir.is_absolute() else args.output_dir).resolve() / "phase_completion_repair_report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
