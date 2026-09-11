"""Execute the frozen N3 q-producer compatibility self-null calibration.

The consumer fits exactly the two already registered producers on each
fit-only V2 fold.  For every fixed trial it generates one 1000-row
pseudo-DATA bank, four independent 200-row MODEL banks, and eight independent
200-row MODEL reference banks.  It then calls the complete existing V2 raw-q
evaluator and its existing mean/q95 aggregation.  Held-out q values and
sealed confirmation values are never used for a metric.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.evidence import (
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import (
    define_n3_q_producer_compatibility_self_null_calibration_v1 as contract_definition,
    execute_n3_development_bank_v2_empirical_ceiling as v2_ceiling,
    execute_n3_empirical_marginal_fixed_r_student_t_copula_v1 as student_t,
    execute_n3_empirical_marginal_gaussian_copula_producer_v1 as gaussian,
)


TASK_ID = "N3_Q_PRODUCER_COMPATIBILITY_SELF_NULL_CALIBRATION_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_q_producer_compatibility_self_null_calibration_report.v1"
REPORT_NAME = "n3_q_producer_compatibility_self_null_calibration_report.v1.json"
EXPECTED_BRANCH = contract_definition.EXPECTED_BRANCH
EXPECTED_STARTING_HEAD = contract_definition.EXPECTED_STARTING_HEAD

CONTRACT_PATH = contract_definition.OUTPUT_DIR / contract_definition.CONTRACT_NAME
EXPECTED_CONTRACT_SHA256 = "32ba12a8f09a850f0126bf528741928a247d957cf0a2105bd4329d102a97662c"
V2_Q_TENSOR_SHA256 = "d660fd8dc449725edab081d00c8f553f151029cddef9210360a7a6fad4281209"
OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_q_producer_compatibility_self_null_calibration_v1_attempt_01"
)

CLASSIFICATION_STABLE = "N3_Q_COMPATIBILITY_SELF_NULL_STABLE"
CLASSIFICATION_FALSE_FAIL_LIMITATION = "N3_Q_COMPATIBILITY_SELF_NULL_FALSE_FAIL_LIMITATION"
CLASSIFICATION_MIXED_OR_UNRESOLVED = "N3_Q_COMPATIBILITY_SELF_NULL_MIXED_OR_UNRESOLVED"
ALLOWED_CLASSIFICATIONS = (
    CLASSIFICATION_STABLE,
    CLASSIFICATION_FALSE_FAIL_LIMITATION,
    CLASSIFICATION_MIXED_OR_UNRESOLVED,
)

PRODUCER_IDS = contract_definition.PRODUCER_IDS
FOLD_COUNT = contract_definition.FOLD_COUNT
POOL_COUNT = contract_definition.POOL_COUNT
DIMENSION = contract_definition.DIMENSION
FIT_COUNT_PER_FOLD = contract_definition.FIT_COUNT_PER_FOLD
HELDOUT_COUNT_PER_FOLD = contract_definition.HELDOUT_COUNT_PER_FOLD
PSEUDO_DATA_SAMPLE_COUNT = contract_definition.PSEUDO_DATA_SAMPLE_COUNT
MODEL_SAMPLE_COUNT = contract_definition.MODEL_SAMPLE_COUNT
DATA_VS_MODEL_REPLICATES = contract_definition.DATA_VS_MODEL_REPLICATES
MODEL_VS_MODEL_REPLICATES = contract_definition.MODEL_VS_MODEL_REPLICATES
MODEL_REFERENCE_PAIRS = contract_definition.MODEL_REFERENCE_PAIRS
TRIAL_COUNT_PER_PRODUCER_FOLD = contract_definition.TRIAL_COUNT_PER_PRODUCER_FOLD
COMPATIBILITY_Q = contract_definition.COMPATIBILITY_Q
SCALAR_METRICS = contract_definition.SCALAR_METRICS
JOINT_METRICS = contract_definition.JOINT_METRICS
MARGINAL_METRICS = contract_definition.MARGINAL_METRICS
RNG_NAMESPACE = contract_definition.RNG_NAMESPACE

METRIC_IMPLEMENTATION = contract_definition.METRIC_IMPLEMENTATION
AGGREGATION_IMPLEMENTATION = contract_definition.AGGREGATION_IMPLEMENTATION


class InvalidHarnessError(RuntimeError):
    """A frozen self-null boundary or evaluator invariant failed."""


def _invalid(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidHarnessError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _invalid(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _git_state(repo_root: Path) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "branch": branch,
        "head": head,
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
        "worktree_clean": not status,
    }


def _read_contract(repo_root: Path) -> dict[str, Any]:
    path = repo_root / CONTRACT_PATH
    _invalid(path.is_file(), f"self-null contract is missing: {path}")
    _invalid(file_sha256(path) == EXPECTED_CONTRACT_SHA256, "self-null contract hash changed")
    contract = json.loads(path.read_text(encoding="utf-8"))
    _invalid(isinstance(contract, Mapping), "self-null contract is not a mapping")
    _invalid(contract.get("schema") == contract_definition.CONTRACT_SCHEMA, "self-null contract schema changed")
    _invalid(contract.get("status") == "valid_completed" and contract.get("result") == "READY", "self-null contract is not ready")
    _invalid(contract.get("classification") == contract_definition.READY_CLASSIFICATION, "self-null contract classification changed")
    _invalid(contract.get("definition_only") is True, "self-null contract is not definition-only")
    simulation = _mapping(contract.get("simulation"), "self-null simulation contract")
    _invalid(simulation.get("trials_per_producer_fold") == TRIAL_COUNT_PER_PRODUCER_FOLD, "trial count changed after contract freeze")
    _invalid(simulation.get("no_early_stop") is True and simulation.get("no_retry") is True, "trial stopping or retry policy changed")
    _invalid(simulation.get("sample_size_asymmetry", {}).get("pseudo_data_vs_model") == [1000, 200], "DATA-vs-MODEL sample sizes changed")
    _invalid(simulation.get("sample_size_asymmetry", {}).get("model_vs_model_reference") == [200, 200], "MODEL-vs-MODEL sample sizes changed")
    producer_set = _mapping(contract.get("producer_set"), "self-null producer set")
    _invalid(producer_set.get("identities") == list(PRODUCER_IDS), "producer identity set changed")
    evaluator = _mapping(contract.get("evaluator"), "self-null evaluator contract")
    _invalid(evaluator.get("complete_existing_evaluator") is True, "complete evaluator is not frozen")
    _invalid(evaluator.get("new_evaluator_created") is False, "new evaluator was introduced")
    _invalid(evaluator.get("all_eight_scalar_guardrails") == list(SCALAR_METRICS), "guardrail set changed")
    _invalid(evaluator.get("threshold_quantile") == COMPATIBILITY_Q, "evaluator quantile changed")
    classification = _mapping(contract.get("aggregate_classification"), "self-null classification contract")
    _invalid(classification.get("allowed") == list(ALLOWED_CLASSIFICATIONS), "classification set changed")
    _invalid(classification.get("no_threshold_changes_after_observation") is True, "post-observation threshold changes are allowed")
    firewalls = _mapping(contract.get("firewalls"), "self-null firewalls")
    for key in (
        "heldout_q_metric_evaluation",
        "heldout_q_used_for_fit_or_sampling",
        "sealed_confirmation_values_loaded",
        "new_producer_family",
        "nu_changed",
        "marginals_changed",
        "R_changed",
        "optical_or_image_evaluation",
        "cuda_or_training",
    ):
        _invalid(firewalls.get(key) is False, f"self-null firewall changed: {key}")
    _invalid(firewalls.get("selected_producer") is None, "self-null contract selected a producer")
    _invalid(firewalls.get("training_authorized") is False, "self-null contract opened training")
    _invalid(firewalls.get("formal_promotion") is False, "self-null contract opened promotion")
    _invalid(firewalls.get("generation_readiness") == "not_ready", "self-null contract opened generation")
    return dict(contract)


def _validate_current_authority(repo_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    # The definition consumer performs the metadata/hash-only authority check;
    # repeating it here prevents execution from consuming a changed input.
    authorities = contract_definition._validate_authorities(repo_root)
    _invalid(file_sha256(repo_root / CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256, "self-null contract changed after load")
    authority_refs = _mapping(contract.get("authorities"), "contract authorities")
    v2_q = _mapping(authority_refs.get("v2_q"), "contract V2 q authority")
    _invalid(file_sha256(repo_root / contract_definition.V2_Q_ARTIFACT) == v2_q.get("artifact_sha256"), "V2 q artifact changed after contract freeze")
    return authorities


def _validate_fold_partition(fold_ids: torch.Tensor) -> None:
    values = fold_ids.to(dtype=torch.int64).reshape(-1)
    _invalid(tuple(values.shape) == (POOL_COUNT,), "fold assignment shape changed")
    _invalid(set(int(value) for value in values.tolist()) == set(range(FOLD_COUNT)), "fold assignment domain changed")
    for fold in range(FOLD_COUNT):
        heldout = (values == fold).nonzero(as_tuple=False).reshape(-1)
        fitting = (values != fold).nonzero(as_tuple=False).reshape(-1)
        _invalid(heldout.numel() == HELDOUT_COUNT_PER_FOLD, f"fold {fold} heldout count changed")
        _invalid(fitting.numel() == FIT_COUNT_PER_FOLD, f"fold {fold} fit count changed")
        _invalid(not bool(torch.isin(fitting, heldout).any()), f"fold {fold} fit/heldout overlap")
        _invalid(torch.equal(torch.sort(torch.cat((fitting, heldout))).values, torch.arange(POOL_COUNT, dtype=torch.int64)), f"fold {fold} partition does not cover V2 bank")


def _load_q_after_preflight(repo_root: Path, source: Mapping[str, Any], rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    payload = safe_torch_load(
        repo_root / v2_ceiling.Q_ARTIFACT,
        expected_sha256=contract_definition.AUTHORITY_SHA256[contract_definition.V2_Q_ARTIFACT.as_posix()],
        map_location="cpu",
    )
    payload_map = _mapping(payload, "V2 q artifact")
    _invalid(payload_map.get("schema") == "n3_development_bank_v2_q.v1", "V2 q artifact schema changed")
    _invalid(payload_map.get("status") == "VALID_COMPLETED", "V2 q artifact is incomplete")
    q = payload_map.get("q")
    source_indices = payload_map.get("source_indices")
    labels = payload_map.get("labels")
    _invalid(torch.is_tensor(q) and q.dtype == torch.float64 and tuple(q.shape) == (POOL_COUNT, DIMENSION), "V2 q shape or dtype changed")
    _invalid(torch.is_tensor(source_indices) and source_indices.dtype == torch.int64 and tuple(source_indices.shape) == (POOL_COUNT,), "V2 source-index shape changed")
    _invalid(torch.is_tensor(labels) and labels.dtype == torch.int64 and tuple(labels.shape) == (POOL_COUNT,), "V2 label shape changed")
    q = q.detach().cpu().contiguous()
    source_indices = source_indices.detach().cpu().contiguous()
    labels = labels.detach().cpu().contiguous()
    _invalid(bool(torch.isfinite(q).all()) and bool(q.ge(0.0).all()), "V2 q values are invalid")
    _invalid(tensor_sha256(q) == V2_Q_TENSOR_SHA256, "V2 q tensor identity changed")
    expected_indices = torch.tensor([int(row["source_index"]) for row in rows], dtype=torch.int64)
    expected_labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.int64)
    _invalid(torch.equal(source_indices, expected_indices), "V2 source indices changed")
    _invalid(torch.equal(labels, expected_labels), "V2 labels changed")
    identities_value = payload_map.get("source_row_identity")
    _invalid(isinstance(identities_value, list) and len(identities_value) == POOL_COUNT, "V2 stable source identities are incomplete")
    identities = [str(value) for value in identities_value]
    expected_identities = [str(row["stable_source_row_identity"]) for row in rows]
    _invalid(identities == expected_identities, "V2 stable source identities changed")
    return q, source_indices, labels, identities


def _fit_positions(fold_ids: torch.Tensor, fold: int) -> tuple[torch.Tensor, torch.Tensor]:
    heldout = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1)
    fitting = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
    _invalid(fitting.numel() == FIT_COUNT_PER_FOLD and heldout.numel() == HELDOUT_COUNT_PER_FOLD, f"fold {fold} partition counts changed")
    return fitting, heldout


def _fit_producer(producer_id: str, q_fit: torch.Tensor) -> dict[str, Any]:
    _invalid(tuple(q_fit.shape) == (FIT_COUNT_PER_FOLD, DIMENSION), f"{producer_id} fit q shape changed")
    _invalid(q_fit.dtype == torch.float64 and bool(torch.isfinite(q_fit).all()) and bool(q_fit.ge(0.0).all()), f"{producer_id} fit q is invalid")
    with torch.no_grad():
        if producer_id == PRODUCER_IDS[0]:
            previous_count = gaussian.FIT_COUNT_PER_FOLD
            gaussian.FIT_COUNT_PER_FOLD = FIT_COUNT_PER_FOLD
            try:
                fit = gaussian._fit_fold(q_fit)
            finally:
                gaussian.FIT_COUNT_PER_FOLD = previous_count
            return fit
        if producer_id == PRODUCER_IDS[1]:
            return student_t._fit_fold(q_fit)
    raise InvalidHarnessError(f"unregistered producer reached fit dispatch: {producer_id}")


def _fit_summary(producer_id: str, fold: int, q_fit: torch.Tensor, fit_positions: torch.Tensor, source_indices: torch.Tensor, labels: torch.Tensor, fit: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "producer_id": producer_id,
        "fold": fold,
        "fit_rows": FIT_COUNT_PER_FOLD,
        "heldout_rows": HELDOUT_COUNT_PER_FOLD,
        "dimension": DIMENSION,
        "fit_q_sha256": tensor_sha256(q_fit),
        "fit_row_positions_sha256": tensor_sha256(fit_positions),
        "fit_source_indices_sha256": tensor_sha256(source_indices),
        "fit_labels_sha256": tensor_sha256(labels),
        "heldout_values_used_for_fit": False,
        "parameter_state": {},
    }
    if producer_id == PRODUCER_IDS[0]:
        dependence = _mapping(fit.get("dependence"), "Gaussian dependence")
        summary["parameter_state"] = {
            "sorted_values_sha256": tensor_sha256(fit["sorted_values"]),
            "plotting_positions_sha256": tensor_sha256(fit["plotting_positions"]),
            "z_sha256": tensor_sha256(fit["z"]),
            "R_shrunk_sha256": tensor_sha256(dependence["R_shrunk"]),
            "cholesky_sha256": tensor_sha256(dependence["cholesky"]),
            "alpha": float(dependence["alpha"]),
            "fixed_R": True,
        }
    else:
        baseline_fit = _mapping(fit.get("baseline"), "Student-t baseline fit")
        dependence = _mapping(baseline_fit.get("dependence"), "Student-t dependence")
        summary["parameter_state"] = {
            "sorted_values_sha256": tensor_sha256(baseline_fit["sorted_values"]),
            "plotting_positions_sha256": tensor_sha256(baseline_fit["plotting_positions"]),
            "z_sha256": tensor_sha256(baseline_fit["z"]),
            "R_shrunk_sha256": tensor_sha256(dependence["R_shrunk"]),
            "cholesky_sha256": tensor_sha256(dependence["cholesky"]),
            "nu": float(fit["nu"]),
            "fixed_R": True,
            "marginals_fit_only": True,
        }
    return summary


def _seed(producer_id: str, fold: int, trial: int, role: str, bank: int) -> int:
    material = f"{RNG_NAMESPACE}|producer={producer_id}|fold={fold}|trial={trial}|role={role}|bank={bank}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def validate_seed_namespace(producer_id: str, fold: int, trial: int) -> list[int]:
    roles = [("pseudo-data", 0)]
    roles.extend(("data-vs-model", index) for index in range(DATA_VS_MODEL_REPLICATES))
    roles.extend(("model-vs-model-reference", index) for index in range(MODEL_VS_MODEL_REPLICATES))
    seeds = [_seed(producer_id, fold, trial, role, bank) for role, bank in roles]
    _invalid(len(seeds) == len(set(seeds)), "RNG namespace collision within a trial")
    return seeds


def _numeric_summary(value: torch.Tensor) -> dict[str, float]:
    view = value.detach().to(dtype=torch.float64).reshape(-1)
    _invalid(view.numel() > 0 and bool(torch.isfinite(view).all()), "sampling summary input is invalid")
    return {"min": float(view.min()), "mean": float(view.mean()), "max": float(view.max())}


def _sampling_record(producer_id: str, fold: int, trial: int, role: str, bank: int, count: int, seed: int, q_new: torch.Tensor) -> dict[str, Any]:
    _invalid(tuple(q_new.shape) == (count, DIMENSION), f"{producer_id} {role} sample shape changed")
    _invalid(q_new.dtype == torch.float64, f"{producer_id} {role} sample dtype changed")
    _invalid(bool(torch.isfinite(q_new).all()) and bool(q_new.ge(0.0).all()), f"{producer_id} {role} sample support changed")
    return {
        "producer_id": producer_id,
        "fold": fold,
        "trial": trial,
        "role": role,
        "bank": bank,
        "seed": seed,
        "rng_namespace": RNG_NAMESPACE,
        "sample_count": count,
        "dtype": "float64",
        "q_sha256": tensor_sha256(q_new),
        "q_summary": _numeric_summary(q_new),
        "best_of_n": False,
        "seed_retry": False,
        "heldout_seed_tuning": False,
        "post_evaluation_resampling": False,
    }


def _sample_gaussian(fit: Mapping[str, Any], producer_id: str, fold: int, trial: int, role: str, bank: int, count: int) -> tuple[torch.Tensor, dict[str, Any]]:
    seed = _seed(producer_id, fold, trial, role, bank)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((count, DIMENSION), generator=generator, dtype=torch.float64)
    dependence = _mapping(fit.get("dependence"), "Gaussian dependence")
    z_new = noise.mm(dependence["cholesky"].transpose(0, 1))
    probabilities = torch.special.ndtr(z_new)
    q_new = gaussian._empirical_quantile_batch(fit["sorted_values"], probabilities).contiguous()
    record = _sampling_record(producer_id, fold, trial, role, bank, count, seed, q_new)
    record.update({"sampler": "existing empirical-marginal Gaussian-copula equations", "latent_q_sha256": tensor_sha256(z_new), "probability_sha256": tensor_sha256(probabilities)})
    return q_new, record


def _sample_student_t(fit: Mapping[str, Any], producer_id: str, fold: int, trial: int, role: str, bank: int, count: int) -> tuple[torch.Tensor, dict[str, Any]]:
    seed = _seed(producer_id, fold, trial, role, bank)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    nu = float(fit["nu"])
    baseline_fit = _mapping(fit.get("baseline"), "Student-t baseline fit")
    dependence = _mapping(baseline_fit.get("dependence"), "Student-t dependence")
    epsilon = torch.randn((count, DIMENSION), generator=generator, dtype=torch.float64)
    gaussian_latent = epsilon.mm(dependence["cholesky"].transpose(0, 1))
    shape = torch.full((count,), nu / 2.0, dtype=torch.float64)
    gamma_unit = torch._standard_gamma(shape, generator=generator)
    radial = gamma_unit * (2.0 / nu)
    _invalid(bool(torch.isfinite(radial).all()) and bool(radial.gt(0.0).all()), "Student-t radial draw is invalid")
    latent = gaussian_latent / torch.sqrt(radial).unsqueeze(1)
    probabilities = student_t._student_t_cdf(latent, nu)
    q_new = gaussian._empirical_quantile_batch(baseline_fit["sorted_values"], probabilities).contiguous()
    record = _sampling_record(producer_id, fold, trial, role, bank, count, seed, q_new)
    record.update({"sampler": "existing fixed-R empirical-marginal Student-t equations", "nu": nu, "epsilon_sha256": tensor_sha256(epsilon), "latent_sha256": tensor_sha256(latent), "probability_sha256": tensor_sha256(probabilities)})
    return q_new, record


def _sample_bank(producer_id: str, fit: Mapping[str, Any], fold: int, trial: int, role: str, bank: int, count: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if producer_id == PRODUCER_IDS[0]:
        return _sample_gaussian(fit, producer_id, fold, trial, role, bank, count)
    if producer_id == PRODUCER_IDS[1]:
        return _sample_student_t(fit, producer_id, fold, trial, role, bank, count)
    raise InvalidHarnessError(f"unregistered producer reached sampling dispatch: {producer_id}")


def _validate_metric_mapping(metrics: Mapping[str, Any], label: str) -> None:
    v2_ceiling._validate_metrics(metrics, label)
    _invalid(set(metrics) >= set(SCALAR_METRICS), f"{label} is missing a scalar guardrail")


def _evaluate_trial(pseudo_data: torch.Tensor, model_banks: list[torch.Tensor], reference_banks: list[torch.Tensor]) -> dict[str, Any]:
    _invalid(tuple(pseudo_data.shape) == (PSEUDO_DATA_SAMPLE_COUNT, DIMENSION), "pseudo-DATA sample size changed")
    _invalid(len(model_banks) == DATA_VS_MODEL_REPLICATES and all(tuple(bank.shape) == (MODEL_SAMPLE_COUNT, DIMENSION) for bank in model_banks), "MODEL sample structure changed")
    _invalid(len(reference_banks) == MODEL_VS_MODEL_REPLICATES and all(tuple(bank.shape) == (MODEL_SAMPLE_COUNT, DIMENSION) for bank in reference_banks), "reference sample structure changed")
    data_metrics = [v2_ceiling._pair_metrics_raw_q(pseudo_data, bank) for bank in model_banks]
    reference_pairs = [v2_ceiling._pair_metrics_raw_q(reference_banks[left], reference_banks[right]) for left, right in MODEL_REFERENCE_PAIRS]
    for index, metrics in enumerate(data_metrics):
        _validate_metric_mapping(metrics, f"DATA-vs-MODEL replicate {index}")
    for index, metrics in enumerate(reference_pairs):
        _validate_metric_mapping(metrics, f"MODEL-vs-MODEL reference pair {index}")
    data_mean = gaussian._mean_metrics(data_metrics)
    reference_q95 = gaussian._q95_metrics(reference_pairs)
    _validate_metric_mapping(data_mean, "DATA-vs-MODEL mean")
    _validate_metric_mapping(reference_q95, "MODEL-vs-MODEL reference q95")
    components = {name: float(data_mean[name]) <= float(reference_q95[name]) for name in SCALAR_METRICS}
    per_coordinate_checks = [
        float(data_mean["per_coordinate_marginal_energy_distance"][index]) <= float(reference_q95["per_coordinate_marginal_energy_distance"][index])
        for index in range(DIMENSION)
    ]
    return {
        "data_vs_model_replicates": data_metrics,
        "data_vs_model_mean": data_mean,
        "model_vs_model_reference_pairs": reference_pairs,
        "model_vs_model_reference_q95": reference_q95,
        "compatibility_components": components,
        "primary_pass": bool(components["energy_distance"]),
        "fold_pass": all(components.values()),
        "failed_guardrails": [name for name in SCALAR_METRICS if not components[name]],
        "per_coordinate_marginal_q95_checks": per_coordinate_checks,
        "per_coordinate_all_pass": all(per_coordinate_checks),
        "per_coordinate_vector_role": "descriptive only; not a scalar compatibility gate",
        "compatibility_rule": "existing V2 mean of four DATA-vs-MODEL raw-q metrics <= q95 of four MODEL-vs-MODEL reference-pair metrics for all eight scalar guardrails",
    }


def _trial_summary(producer_id: str, fold: int, trial: int, pseudo_record: Mapping[str, Any], model_records: list[Mapping[str, Any]], reference_records: list[Mapping[str, Any]], evaluated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "producer_id": producer_id,
        "fold": fold,
        "trial": trial,
        "pseudo_data": dict(pseudo_record),
        "model_banks": [dict(record) for record in model_records],
        "reference_banks": [dict(record) for record in reference_records],
        **dict(evaluated),
    }


def _failure_key(trial: Mapping[str, Any]) -> str:
    failed = trial.get("failed_guardrails", [])
    return "PASS" if not failed else "+".join(str(value) for value in failed)


def _fold_summary(producer_id: str, fold: int, fit_summary: Mapping[str, Any], trials: list[Mapping[str, Any]]) -> dict[str, Any]:
    _invalid(len(trials) == TRIAL_COUNT_PER_PRODUCER_FOLD, f"{producer_id} fold {fold} trial count is incomplete")
    trial_count = len(trials)
    total_pass_count = sum(bool(item["fold_pass"]) for item in trials)
    primary_fail_count = sum(not bool(item["primary_pass"]) for item in trials)
    guardrails = {
        name: {
            "false_fail_count": sum(not bool(item["compatibility_components"][name]) for item in trials),
            "false_fail_rate": sum(not bool(item["compatibility_components"][name]) for item in trials) / float(trial_count),
        }
        for name in SCALAR_METRICS
    }
    combinations = Counter(_failure_key(item) for item in trials)
    return {
        "producer_id": producer_id,
        "fold": fold,
        "trials": trial_count,
        "total_pass_count": total_pass_count,
        "total_pass_rate": total_pass_count / float(trial_count),
        "primary_false_fail_count": primary_fail_count,
        "primary_false_fail_rate": primary_fail_count / float(trial_count),
        "guardrail_false_failures": guardrails,
        "dominant_failure_combinations": [
            {"combination": key, "count": count, "fraction": count / float(trial_count)}
            for key, count in sorted(combinations.items(), key=lambda item: (-item[1], item[0]))
        ],
        "fit": dict(fit_summary),
        "trial_records": [dict(trial) for trial in trials],
    }


def _producer_summary(producer_id: str, folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    fold_rates = [float(item["total_pass_rate"]) for item in folds]
    pooled_trials = sum(int(item["trials"]) for item in folds)
    pooled_pass_count = sum(int(item["total_pass_count"]) for item in folds)
    pooled_primary_fail_count = sum(int(item["primary_false_fail_count"]) for item in folds)
    return {
        "producer_id": producer_id,
        "fold_count": len(folds),
        "pooled_trials": pooled_trials,
        "pooled_pass_count": pooled_pass_count,
        "pooled_pass_rate": pooled_pass_count / float(pooled_trials),
        "pooled_primary_false_fail_rate": pooled_primary_fail_count / float(pooled_trials),
        "fold_pass_rates": fold_rates,
        "implied_five_fold_all_pass_probability": math.prod(fold_rates),
        "independence_assumption": "diagnostic multiplication of observed fold rates only; fold independence is not asserted as a population fact",
    }


def classify_self_null(producer_fold_summaries: list[Mapping[str, Any]], producer_summaries: list[Mapping[str, Any]], classification_contract: Mapping[str, Any]) -> str:
    stable = _mapping(classification_contract.get("stable_if"), "stable classification rule")
    limitation = _mapping(classification_contract.get("false_fail_limitation_if"), "false-fail limitation rule")
    stable_min_pass = float(stable["every_producer_fold_pass_rate_at_least"])
    stable_max_guardrail = float(stable["every_producer_fold_guardrail_false_fail_rate_at_most"])
    stable_min_product = float(stable["every_producer_implied_five_fold_probability_at_least"])
    limitation_max_pass = float(limitation["any_producer_fold_pass_rate_at_most"])
    limitation_min_guardrail = float(limitation["or_any_producer_fold_guardrail_false_fail_rate_at_least"])
    limitation_max_product = float(limitation["or_any_producer_implied_five_fold_probability_at_most"])
    if (
        all(float(item["total_pass_rate"]) >= stable_min_pass for item in producer_fold_summaries)
        and all(
            float(item["guardrail_false_failures"][name]["false_fail_rate"]) <= stable_max_guardrail
            for item in producer_fold_summaries
            for name in SCALAR_METRICS
        )
        and all(float(item["implied_five_fold_all_pass_probability"]) >= stable_min_product for item in producer_summaries)
    ):
        return CLASSIFICATION_STABLE
    if (
        any(float(item["total_pass_rate"]) <= limitation_max_pass for item in producer_fold_summaries)
        or any(
            float(item["guardrail_false_failures"][name]["false_fail_rate"]) >= limitation_min_guardrail
            for item in producer_fold_summaries
            for name in SCALAR_METRICS
        )
        or any(float(item["implied_five_fold_all_pass_probability"]) <= limitation_max_product for item in producer_summaries)
    ):
        return CLASSIFICATION_FALSE_FAIL_LIMITATION
    return CLASSIFICATION_MIXED_OR_UNRESOLVED


def _authority_snapshot(repo_root: Path) -> dict[str, str]:
    return {
        path_text: file_sha256(repo_root / Path(path_text))
        for path_text in contract_definition.AUTHORITY_SHA256
    }


def _validate_authority_snapshot(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    _invalid(dict(before) == dict(after), "frozen authority changed during self-null execution")
    for path_text, expected in contract_definition.AUTHORITY_SHA256.items():
        _invalid(after.get(path_text) == expected, f"frozen authority hash changed: {path_text}")


def _classification_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(contract.get("aggregate_classification"), "classification contract")


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once self-null output: {output_dir}")
    git = _git_state(repo_root)
    _invalid(git["branch"] == EXPECTED_BRANCH, "current branch is not the self-null authority branch")
    _invalid(git["head"] == EXPECTED_STARTING_HEAD, "current HEAD is not the frozen V2/N4 base")
    contract = _read_contract(repo_root)
    authorities = _validate_current_authority(repo_root, contract)
    source, fold_manifest, rows, fold_ids, q_meta = v2_ceiling._validate_source_and_fold(repo_root)
    _invalid(q_meta.get("values_loaded") is False, "q values were loaded before self-null preflight")
    _validate_fold_partition(fold_ids)
    q, source_indices, labels, identities = _load_q_after_preflight(repo_root, source, rows)
    _invalid(file_sha256(repo_root / contract_definition.V2_Q_ARTIFACT) == contract_definition.AUTHORITY_SHA256[contract_definition.V2_Q_ARTIFACT.as_posix()], "V2 q authority changed after load")
    _invalid(len(identities) == POOL_COUNT, "V2 stable identities are incomplete")
    _invalid(torch.equal(labels, torch.tensor([int(row["label"]) for row in rows], dtype=torch.int64)), "V2 labels differ from authority")
    classification_contract = _classification_contract(contract)
    authority_snapshot_before = _authority_snapshot(repo_root)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    fold_records: list[dict[str, Any]] = []
    producer_fold_summaries: list[dict[str, Any]] = []
    producer_summaries: list[dict[str, Any]] = []
    with torch.no_grad():
        for producer_id in PRODUCER_IDS:
            producer_folds: list[dict[str, Any]] = []
            for fold in range(FOLD_COUNT):
                fit_positions, _heldout_positions = _fit_positions(fold_ids, fold)
                fit_q = q.index_select(0, fit_positions).contiguous()
                fit_source_indices = source_indices.index_select(0, fit_positions).contiguous()
                fit_labels = labels.index_select(0, fit_positions).contiguous()
                _invalid(torch.equal(torch.bincount(fit_labels, minlength=10), torch.full((10,), 400, dtype=torch.int64)), f"fold {fold} fit class balance changed")
                print(json.dumps({"stage": TASK_ID, "producer": producer_id, "fold": fold, "status": "fit_started"}), flush=True)
                fit = _fit_producer(producer_id, fit_q)
                fit_summary = _fit_summary(producer_id, fold, fit_q, fit_positions, fit_source_indices, fit_labels, fit)
                trial_records: list[dict[str, Any]] = []
                for trial in range(TRIAL_COUNT_PER_PRODUCER_FOLD):
                    validate_seed_namespace(producer_id, fold, trial)
                    pseudo_data, pseudo_record = _sample_bank(producer_id, fit, fold, trial, "pseudo-data", 0, PSEUDO_DATA_SAMPLE_COUNT)
                    model_banks: list[torch.Tensor] = []
                    model_records: list[dict[str, Any]] = []
                    for bank in range(DATA_VS_MODEL_REPLICATES):
                        sampled, record = _sample_bank(producer_id, fit, fold, trial, "data-vs-model", bank, MODEL_SAMPLE_COUNT)
                        model_banks.append(sampled)
                        model_records.append(record)
                    reference_banks: list[torch.Tensor] = []
                    reference_records: list[dict[str, Any]] = []
                    for bank in range(MODEL_VS_MODEL_REPLICATES):
                        sampled, record = _sample_bank(producer_id, fit, fold, trial, "model-vs-model-reference", bank, MODEL_SAMPLE_COUNT)
                        reference_banks.append(sampled)
                        reference_records.append(record)
                    evaluated = _evaluate_trial(pseudo_data, model_banks, reference_banks)
                    trial_records.append(_trial_summary(producer_id, fold, trial, pseudo_record, model_records, reference_records, evaluated))
                    if (trial + 1) % 5 == 0 or trial + 1 == TRIAL_COUNT_PER_PRODUCER_FOLD:
                        print(json.dumps({"stage": TASK_ID, "producer": producer_id, "fold": fold, "trial": trial + 1, "trials": TRIAL_COUNT_PER_PRODUCER_FOLD, "fold_pass_so_far": sum(bool(item["fold_pass"]) for item in trial_records)}), flush=True)
                summary = _fold_summary(producer_id, fold, fit_summary, trial_records)
                fold_records.append(summary)
                producer_folds.append(summary)
                producer_fold_summaries.append(summary)
                print(json.dumps({"stage": TASK_ID, "producer": producer_id, "fold": fold, "status": "completed", "total_pass_rate": summary["total_pass_rate"], "primary_false_fail_rate": summary["primary_false_fail_rate"]}), flush=True)
                _validate_authority_snapshot(authority_snapshot_before, _authority_snapshot(repo_root))
            producer_summaries.append(_producer_summary(producer_id, producer_folds))
    _validate_authority_snapshot(authority_snapshot_before, _authority_snapshot(repo_root))
    _invalid(len(producer_fold_summaries) == len(PRODUCER_IDS) * FOLD_COUNT, "producer/fold coverage is incomplete")
    classification = classify_self_null(producer_fold_summaries, producer_summaries, classification_contract)
    _invalid(classification in ALLOWED_CLASSIFICATIONS, "self-null classification is outside the frozen set")
    next_action = {
        CLASSIFICATION_STABLE: "close evaluator-noise concern; do not repeat Gaussian/Student-t; plan exactly one higher-expressivity producer family",
        CLASSIFICATION_FALSE_FAIL_LIMITATION: "stop family escalation; define N3_Q_COMPATIBILITY_EVALUATOR_V3 and calibrate it before further producer evaluation",
        CLASSIFICATION_MIXED_OR_UNRESOLVED: "no family zoo; resolve the single remaining self-null uncertainty with one bounded decision",
    }[classification]
    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "classification": classification,
        "created_at_utc": created_at,
        "git": {
            **git,
            "consumer_script": _relative(repo_root, Path(__file__).resolve()),
            "consumer_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "contract": {
            "path": _relative(repo_root, repo_root / CONTRACT_PATH),
            "sha256": EXPECTED_CONTRACT_SHA256,
            "classification": contract["classification"],
        },
        "input_authorities": {
            "v2_source_manifest": {"path": _relative(repo_root, repo_root / contract_definition.V2_SOURCE_MANIFEST), "sha256": contract_definition.AUTHORITY_SHA256[contract_definition.V2_SOURCE_MANIFEST.as_posix()]},
            "v2_q_artifact": {"path": _relative(repo_root, repo_root / contract_definition.V2_Q_ARTIFACT), "sha256": contract_definition.AUTHORITY_SHA256[contract_definition.V2_Q_ARTIFACT.as_posix()], "tensor_sha256": V2_Q_TENSOR_SHA256},
            "v2_fold_manifest": {"path": _relative(repo_root, repo_root / contract_definition.V2_FOLD_MANIFEST), "sha256": contract_definition.AUTHORITY_SHA256[contract_definition.V2_FOLD_MANIFEST.as_posix()], "assignment_sha256": "43ee0ce28e143fffdf2bcac98394625ccd766166ea717f23db66ed8770ef18a9"},
            "v2_ceiling_report": {"path": _relative(repo_root, repo_root / contract_definition.V2_CEILING_REPORT), "sha256": contract_definition.AUTHORITY_SHA256[contract_definition.V2_CEILING_REPORT.as_posix()]},
            "n4_report": {"path": _relative(repo_root, repo_root / contract_definition.N4_REPORT), "sha256": contract_definition.AUTHORITY_SHA256[contract_definition.N4_REPORT.as_posix()]},
            "governance_snapshot": dict(authorities["governance"]),
        },
        "producer_set": {
            "identities": list(PRODUCER_IDS),
            "count": len(PRODUCER_IDS),
            "no_new_family": True,
            "fit_from_frozen_fitting_q_only": True,
            "nu_marginals_R_changed": False,
        },
        "execution_settings": {
            "device": "cpu",
            "dtype": "float64",
            "fold_count": FOLD_COUNT,
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
            "pseudo_data_sample_count": PSEUDO_DATA_SAMPLE_COUNT,
            "model_sample_count": MODEL_SAMPLE_COUNT,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "model_reference_pairs": [list(pair) for pair in MODEL_REFERENCE_PAIRS],
            "trials_per_producer_fold": TRIAL_COUNT_PER_PRODUCER_FOLD,
            "total_trials": len(PRODUCER_IDS) * FOLD_COUNT * TRIAL_COUNT_PER_PRODUCER_FOLD,
            "rng_namespace": RNG_NAMESPACE,
            "no_early_stop": True,
            "no_retry": True,
            "best_of_n": False,
            "sample_size_asymmetry": {"pseudo_data_vs_model": [1000, 200], "model_vs_model_reference": [200, 200]},
        },
        "evaluator": {
            "implementation_path": _relative(repo_root, repo_root / METRIC_IMPLEMENTATION),
            "implementation_sha256": contract_definition.AUTHORITY_SHA256[METRIC_IMPLEMENTATION.as_posix()],
            "pair_metric_function": "v2_ceiling._pair_metrics_raw_q",
            "aggregation_implementation_path": _relative(repo_root, repo_root / AGGREGATION_IMPLEMENTATION),
            "aggregation_implementation_sha256": contract_definition.AUTHORITY_SHA256[AGGREGATION_IMPLEMENTATION.as_posix()],
            "aggregation_functions": ["gaussian._mean_metrics", "gaussian._q95_metrics"],
            "complete_existing_evaluator": True,
            "guardrails": list(SCALAR_METRICS),
            "primary": "energy_distance",
            "per_coordinate_vector_recorded_not_gated": True,
        },
        "producer_fold_summaries": producer_fold_summaries,
        "producer_summaries": producer_summaries,
        "five_fold_implications": {
            "producer_products": {item["producer_id"]: item["implied_five_fold_all_pass_probability"] for item in producer_summaries},
            "diagnostic_only": True,
            "independence_assumption_not_population_fact": True,
            "interpretation": "product of observed per-fold self-null pass rates estimates how often a strict five-fold gate would pass under independent-fold approximation; it is not a population probability claim",
        },
        "aggregate_decision": {
            "classification": classification,
            "thresholds_frozen_before_observation": True,
            "next_action": next_action,
        },
        "firewalls": {
            "heldout_q_values_used_for_fit": False,
            "heldout_q_values_used_for_metric": False,
            "heldout_q_values_used_for_sampling": False,
            "sealed_confirmation_values_loaded": False,
            "new_producer_family_evaluated": False,
            "nu_changed": False,
            "marginals_changed": False,
            "R_changed": False,
            "optical_or_image_evaluation": False,
            "cuda_or_training": False,
            "selected_producer": None,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "claim_boundary": {
            "fact": "empirical model-self false-failure behavior under this finite self-null simulation contract",
            "model_assumption": "whether isolated 4/5 producer failures are materially attributable to evaluator noise",
            "not_established": ["population stationarity", "producer selection", "optical compatibility", "image generation", "training authorization", "formal promotion"],
        },
        "limitations": [
            "Only 20 trials per producer/fold were feasible under the complete existing evaluator; rate estimates are coarse and cannot resolve small false-fail differences.",
            "Five-fold products are independence-based diagnostics only.",
            "The calibration uses generated pseudo-DATA and generated MODEL banks from the same fitted producer; it does not evaluate held-out q values.",
            "The result calibrates the V2 mean-vs-model-reference-q95 evaluator and does not select a producer or authorize training.",
        ],
        "sealed_confirmation": {
            "status": "RESERVED_NOT_MATERIALIZED",
            "consumed": False,
            "materialized": False,
            "values_loaded": False,
        },
        "not_executed": [
            "held-out q metric evaluation",
            "sealed confirmation loading",
            "new producer family",
            "nu/marginal/R changes",
            "optical or image evaluation",
            "CUDA",
            "training",
            "producer selection",
            "formal promotion",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(temporary / REPORT_NAME, report, artifact_kind="N3 q self-null calibration report")
        if output_dir.exists():
            raise FileExistsError(f"self-null output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    report_path = output_dir / REPORT_NAME
    return {
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "classification": classification,
        "contract": _relative(repo_root, repo_root / CONTRACT_PATH),
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "report": _relative(repo_root, report_path),
        "report_sha256": file_sha256(report_path),
        "producer_ids": list(PRODUCER_IDS),
        "total_trials": len(PRODUCER_IDS) * FOLD_COUNT * TRIAL_COUNT_PER_PRODUCER_FOLD,
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_consumed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
