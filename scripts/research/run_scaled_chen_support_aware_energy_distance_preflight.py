"""Run the no-update T2 support-aware Energy Distance preflight.

This entrypoint freezes the registered repaired-coordinate endpoint, measures
only three small gradient panels, and records the future canary contract.  It
does not create an optimizer, call backward, mutate parameters, write a
checkpoint, or enter decoder/CMOS/MPCDAR generation for a prior sample.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.application.research import optical_manifold_smf_bogt as bogt
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import replay_scaled_chen_repaired_coordinate_g1_sa as rcr_b


TASK_ID = "H-LOMA-GEN-T2-SAAO-PF"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_support_aware_energy_distance_preflight_v1"
PREFLIGHT_SCHEMA = "lightgen_scaled_chen_support_aware_energy_distance_preflight.v1"
OBJECTIVE_SCHEMA = "lightgen_scaled_chen_support_aware_energy_distance_objective.v1"
PRIOR_STREAM_SCHEMA = "lightgen_scaled_chen_support_aware_energy_distance_prior_stream.v1"
GRADIENT_PANEL_SCHEMA = "lightgen_scaled_chen_support_aware_energy_distance_gradient_panels.v1"
BETA_SCHEMA = "lightgen_scaled_chen_support_aware_energy_distance_beta_registration.v1"
EQUIVALENCE_SCHEMA = "lightgen_scaled_chen_support_aware_energy_distance_gradient_equivalence.v1"
FUTURE_SCHEMA = "lightgen_scaled_chen_support_aware_energy_distance_future_canary.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_support_aware_energy_distance_compact_report.v1"

CONFIG_PATH = Path(
    "configs/research/lightgen_scaled_chen_support_aware_energy_distance_preflight_v1.yaml"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_support_aware_energy_distance_preflight_v1_attempt_01"
)

PORT_COUNT = 100
LOGICAL_BATCH_SIZE = 10
PRIOR_BATCH_SIZE = 10
TARGET_GRADIENT_RATIO = 0.10
MIN_GRADIENT_RATIO = 0.05
MAX_GRADIENT_RATIO = 0.40
TRAINING_BASE_SEED = 1238
REGISTRATION_NAMESPACE = 10_000
PANEL_COUNT = 3
EVALUATION_SEED = 1237
EVALUATION_COUNT = 5000
PARENT_STEP = 5000
END_STEP = 6000
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
CONFIDENCE_LEVEL = 0.95
PARENT_CHECKPOINT_SHA256 = "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"
PARENT_STATE_DIGEST = "cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5"
CALIBRATION_SHA256 = "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07"
CALIBRATION_MANIFEST_SHA256 = "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6"
MRA_PANEL_SHA256 = "842f5cfd41909a666f19f3c88e7e7199a825b255bd111a44953e53351fd28724"
RCR_B_DISTANCES_SHA256 = "d288beb4bd57a1db2a9a864dd8db1be5ef22f75f14702fa13f01b01623671466"
PRIOR_SAMPLE_SHA256 = "c11402193afd71d176468760024ce7dec0d4e4ee3d6d206ac7293ca8ee77fd3e"
PANEL_INDICES = (
    (52048, 45363, 23961, 49433, 33506, 690, 27530, 38282, 53161, 41645),
    (10167, 6356, 44414, 49560, 45009, 14689, 3559, 36398, 8522, 20402),
    (13214, 12166, 26431, 32409, 42216, 28259, 14661, 4227, 31781, 12562),
)
PANEL_INDEX_SHA256 = (
    "822948f09ebdad5f3fb36da36f45ae1dc0b9d65b6ff79675405aeec763f55861",
    "635817ff273289883881c3447115ff0db05a20aa08448e0e2095e4c420df836f",
    "2340114e34809126e683aca958f4590e195f3911aa42697ef94d6ab4940a0202",
)
GOVERNANCE = {
    "mainline_state": "closed_at_current_evidence_boundary",
    "reconstruction_manifold_supported": False,
    "generative_manifold_supported": False,
    "generation_readiness": "not_ready",
    "training_authorized": False,
    "formal_promotion": False,
}


def _resolve(value: str | Path) -> Path:
    return base._resolve_path(value)


def _relative_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(base._repo_root()).as_posix()
    except ValueError:
        return str(path)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} is not registered")


def _assert_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema", "task_id", "hypothesis", "parent", "calibration", "dataset",
        "prior", "objective", "stream", "panels", "registration", "endpoints",
        "future_canary", "authorization", "governance", "execution", "output",
    }
    if set(payload) != required:
        raise ValueError("T2-SAAO-PF config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("task_id") != TASK_ID or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("T2-SAAO-PF config identity is invalid")
    objective = _mapping(payload["objective"], "objective")
    for name, expected in {
        "schema": OBJECTIVE_SCHEMA,
        "name": "mse_plus_support_aware_energy_distance",
        "estimator": "ordered_unbiased_u_statistic",
        "normalization": "divide_by_sqrt_100",
        "dimension": PORT_COUNT,
        "logical_batch_size": LOGICAL_BATCH_SIZE,
        "prior_batch_size": PRIOR_BATCH_SIZE,
        "microbatch_size": 1,
        "prior_prior_term_included": True,
        "no_epsilon_smoothing": True,
        "allow_finite_negative": True,
        "alpha_bogt": 0.0,
        "gamma": 0.0,
    }.items():
        if objective.get(name) != expected:
            raise ValueError(f"objective contract is not locked: {name}")
    panels = _mapping(payload["panels"], "panels")
    if panels.get("indices") != [list(row) for row in PANEL_INDICES] or panels.get("indices_sha256") != list(PANEL_INDEX_SHA256):
        raise ValueError("gradient panel identity is not locked")
    if panels.get("logical_batch_size") != LOGICAL_BATCH_SIZE or panels.get("microbatch_size") != 1:
        raise ValueError("gradient panel batch contract is not locked")
    registration = _mapping(payload["registration"], "registration")
    for name, expected in {
        "target_ratio": TARGET_GRADIENT_RATIO,
        "lower_ratio": MIN_GRADIENT_RATIO,
        "upper_ratio": MAX_GRADIENT_RATIO,
        "formula": "beta_i=target_ratio*G_MSE_i/G_ED_i",
        "aggregation": "arithmetic_mean_of_three_panel_specific_analytic_betas",
        "candidate_count": 1,
        "selection": "analytic_only_no_sweep",
    }.items():
        if registration.get(name) != expected:
            raise ValueError(f"beta registration contract is not locked: {name}")
    if dict(payload["governance"]) != GOVERNANCE:
        raise ValueError("T2-SAAO-PF governance boundary is not locked")
    authorization = _mapping(payload["authorization"], "authorization")
    for name in ("diagnostic_only", "no_update_gradient_measurement"):
        if authorization.get(name) is not True:
            raise ValueError("T2-SAAO-PF requires a no-update diagnostic")
    for name in ("training", "optimizer_step", "optimizer_created", "checkpoint_write", "generation", "decoder_generation", "cmos", "mpcdar", "beta_sweep", "adaptive_beta"):
        if authorization.get(name) is not False:
            raise ValueError(f"T2-SAAO-PF authorization forbids {name}")


def load_config(path: str | Path = CONFIG_PATH) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("T2-SAAO-PF config must be a YAML mapping")
    _assert_config(payload)
    return target, payload, file_sha256(target), canonical_sha256(payload)


def validate_future_objective_contract(config: Mapping[str, Any]) -> None:
    objective = _mapping(config["objective"], "objective")
    future = _mapping(config["future_canary"], "future_canary")
    if objective.get("name") != "mse_plus_support_aware_energy_distance" or future.get("objective") != objective.get("name"):
        raise ValueError("future objective is not support-aware Energy Distance")
    if float(future.get("alpha_bogt", math.nan)) != 0.0 or float(future.get("gamma", math.nan)) != 0.0:
        raise ValueError("future objective must reject nonzero BOGT alpha/gamma")
    if float(objective.get("alpha_bogt", math.nan)) != 0.0 or float(objective.get("gamma", math.nan)) != 0.0:
        raise ValueError("future objective must reject BOGT terms")
    if future.get("trainable_parameters") != ["system.encoder.phase", "system.decoder.phase"]:
        raise ValueError("future objective trainable scope is not locked")
    if future.get("logging") != ["mse", "ed", "weighted_ed", "total", "encoder_grad_norm", "decoder_grad_norm", "finite", "z_mean", "z_variance", "optimizer_telemetry"]:
        raise ValueError("future objective logging cadence is not locked")
    if future.get("start_step") != PARENT_STEP or future.get("end_step") != END_STEP or future.get("steps") != 1000:
        raise ValueError("future objective endpoint is not locked")


def validate_parent_identity(config: Mapping[str, Any]) -> None:
    parent = _mapping(config["parent"], "parent")
    expected = {
        "checkpoint_path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "checkpoint_schema": "scaled_chen_bogt_horizon_checkpoint.v1",
        "global_step": PARENT_STEP,
        "learned_state_digest": PARENT_STATE_DIGEST,
        "mra_panel_path": "outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05/panel.pt",
        "mra_panel_sha256": MRA_PANEL_SHA256,
        "mra_parent_mse_key": "scaled_chen_step_5000",
    }
    for name, value in expected.items():
        _require_equal(parent.get(name), value, f"parent {name}")
    checkpoint_path = _resolve(parent["checkpoint_path"])
    if file_sha256(checkpoint_path) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("parent checkpoint hash is stale")
    panel_path = _resolve(parent["mra_panel_path"])
    if file_sha256(panel_path) != MRA_PANEL_SHA256:
        raise ValueError("parent MRA panel hash is stale")
    rcr_config_path = _resolve(parent["rcr_b_config"])
    if file_sha256(rcr_config_path) != parent["rcr_b_config_sha256"]:
        raise ValueError("parent RCR-B config hash is stale")
    _, rcr_config, rcr_hash, rcr_canonical = rcr_b._load_config(rcr_config_path)
    if rcr_hash != parent["rcr_b_config_sha256"] or rcr_canonical != parent["rcr_b_config_canonical_sha256"]:
        raise ValueError("parent RCR-B config identity is stale")
    sources = rcr_b._validate_replay_sources(rcr_config)
    if sources["rcr_metrics"]["learned_state_freeze"]["state_digest_before"] != PARENT_STATE_DIGEST:
        raise ValueError("parent learned-state authority is stale")
    return None


def validate_prior_contract(config: Mapping[str, Any]) -> None:
    prior = _mapping(config["prior"], "prior")
    expected = {
        "kind": "truncated_normal",
        "mean": 1.0,
        "variance": 0.25,
        "lower": 0.0,
        "unconditional": True,
        "class_conditioned": False,
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_count": EVALUATION_COUNT,
        "evaluation_identity_hash": PRIOR_SAMPLE_SHA256,
        "sampler": "bogt.truncated_normal_sample_inverse_cdf",
    }
    for name, value in expected.items():
        _require_equal(prior.get(name), value, f"prior {name}")
    stream = _mapping(config["stream"], "stream")
    _require_equal(stream.get("training_base_seed"), TRAINING_BASE_SEED, "prior training base seed")
    _require_equal(stream.get("evaluation_seed"), EVALUATION_SEED, "prior evaluation seed")
    if set(registration_panel_seed(panel) for panel in range(PANEL_COUNT)) & set(training_prior_seed(step) for step in range(PARENT_STEP + 1, END_STEP + 1)):
        raise ValueError("prior stream namespaces overlap")


def validate_endpoint_bindings(config: Mapping[str, Any]) -> None:
    endpoints = _mapping(config["endpoints"], "endpoints")
    expected = {
        "rcr_b_distances_sha256": RCR_B_DISTANCES_SHA256,
        "prior_query_seed": EVALUATION_SEED,
        "prior_query_count": EVALUATION_COUNT,
        "prior_sample_sha256": PRIOR_SAMPLE_SHA256,
        "parent_mse_panel_sha256": MRA_PANEL_SHA256,
        "parent_mse_key": "scaled_chen_step_5000",
        "parent_confirmation_count": EVALUATION_COUNT,
        "paired_prior_endpoint": "upper95_delta_p50_and_delta_p95_strictly_below_zero",
        "parent_relative_mse_endpoint": "upper95_mean_delta_mse_at_most_zero",
    }
    for name, value in expected.items():
        _require_equal(endpoints.get(name), value, f"endpoint {name}")


def training_prior_seed(global_step: int, *, base_seed: int = TRAINING_BASE_SEED) -> int:
    if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 0:
        raise ValueError("training prior global step must be a non-negative integer")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed <= 0:
        raise ValueError("training prior base seed must be positive")
    return base_seed * 100_000 + global_step


def registration_panel_seed(panel_id: int, *, base_seed: int = TRAINING_BASE_SEED, namespace: int = REGISTRATION_NAMESPACE) -> int:
    if not isinstance(panel_id, int) or isinstance(panel_id, bool) or not 0 <= panel_id < PANEL_COUNT:
        raise ValueError("registration panel id is invalid")
    return base_seed * 100_000 + namespace + panel_id


def analytic_beta_registration(norms: Sequence[Sequence[float] | Mapping[str, float]]) -> dict[str, Any]:
    if len(norms) != PANEL_COUNT:
        raise ValueError("support weight registration requires exactly three panels")
    betas: list[float] = []
    for entry in norms:
        if isinstance(entry, Mapping):
            mse = float(entry["G_MSE"] if "G_MSE" in entry else entry["mse"])
            ed = float(entry["G_ED"] if "G_ED" in entry else entry["ed"])
        else:
            if len(entry) != 2:
                raise ValueError("support weight registration norm pair is invalid")
            mse, ed = (float(entry[0]), float(entry[1]))
        if not math.isfinite(mse) or not math.isfinite(ed) or mse <= 0.0 or ed <= 0.0:
            raise ValueError("support weight registration norms must be finite and positive")
        betas.append(TARGET_GRADIENT_RATIO * mse / ed)
    return {
        "target_ratio": TARGET_GRADIENT_RATIO,
        "beta_i": betas,
        "beta_registered": sum(betas) / len(betas),
        "candidate_count": 1,
        "selection": "analytic_only_no_sweep",
    }


def classify_replay_ratios(ratios: Sequence[float], lower: float = MIN_GRADIENT_RATIO, upper: float = MAX_GRADIENT_RATIO) -> bool:
    values = [float(value) for value in ratios]
    return len(values) == PANEL_COUNT and all(math.isfinite(value) and lower <= value <= upper for value in values)


def paired_quantile_bootstrap_ci(
    parent: torch.Tensor,
    end: torch.Tensor,
    *,
    probability: float,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if parent.ndim != 1 or end.ndim != 1 or parent.shape != end.shape or parent.numel() < 2:
        raise ValueError("paired bootstrap inputs must be aligned one-dimensional tensors")
    if not torch.isfinite(parent).all() or not torch.isfinite(end).all() or not 0.0 < probability < 1.0:
        raise ValueError("paired bootstrap inputs must be finite")
    if not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("paired bootstrap replicate count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    delta = (end - parent).to(torch.float64).cpu()
    quantiles: list[torch.Tensor] = []
    chunk_size = 128
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        indices = torch.randint(delta.numel(), (count, delta.numel()), generator=generator)
        quantiles.append(torch.quantile(delta.index_select(0, indices.reshape(-1)).reshape(count, -1), probability, dim=1))
    values = torch.cat(quantiles)
    lower = float(torch.quantile(values, (1.0 - CONFIDENCE_LEVEL) / 2.0))
    upper = float(torch.quantile(values, 1.0 - (1.0 - CONFIDENCE_LEVEL) / 2.0))
    return lower, upper


def next_write_once_output(path: str | Path) -> Path:
    requested = Path(path)
    stem = requested.name
    marker = "_attempt_"
    if marker not in stem:
        raise ValueError("write-once output must end in _attempt_<number>")
    prefix, number_text = stem.rsplit(marker, 1)
    if not number_text.isdigit():
        raise ValueError("write-once output attempt is invalid")
    number = int(number_text)
    while True:
        candidate = requested.parent / f"{prefix}_attempt_{number:02d}"
        if not candidate.exists():
            return candidate
        number += 1


def _load_registered_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = _mapping(config["parent"], "parent")
    rcr_path, rcr_config, _, _ = rcr_b._load_config(_resolve(parent["rcr_b_config"]))
    sources = rcr_b._validate_replay_sources(rcr_config)
    checkpoint = safe_torch_load(_resolve(parent["checkpoint_path"]), expected_sha256=PARENT_CHECKPOINT_SHA256, map_location="cpu")
    if checkpoint.get("schema") != parent["checkpoint_schema"] or checkpoint.get("global_step") != PARENT_STEP:
        raise ValueError("parent checkpoint schema or step is invalid")
    panel = safe_torch_load(_resolve(parent["mra_panel_path"]), expected_sha256=MRA_PANEL_SHA256, map_location="cpu")
    if not isinstance(panel, Mapping) or panel.get("schema") != rcr_b.rcr.mra.PANEL_SCHEMA:
        raise ValueError("parent endpoint panel schema is invalid")
    return {"rcr_config_path": rcr_path, "rcr_config": rcr_config, "sources": sources, "checkpoint": checkpoint, "panel": panel}


def _load_calibration(config: Mapping[str, Any]) -> bogt.UnitAwareCalibration:
    calibration_config = _mapping(config["calibration"], "calibration")
    payload = safe_torch_load(_resolve(calibration_config["payload"]), expected_sha256=CALIBRATION_SHA256, map_location="cpu")
    rcr_b.validate_repaired_calibration_identity(
        payload,
        expected_schema=calibration_config["schema"],
        expected_contract=calibration_config["contract_version"],
        expected_calibration_indices_sha256=calibration_config["calibration_indices_sha256"],
        expected_dataset_manifest_hash=calibration_config["dataset_manifest_sha256"],
        expected_mode_bank_sha256=calibration_config["mode_bank_sha256"],
    )
    calibration = rcr_b.rcr.r2.load_registered_unit_aware_calibration(
        _resolve(calibration_config["payload"]),
        expected_sha256=CALIBRATION_SHA256,
        expected_dataset_manifest_hash=calibration_config["dataset_manifest_sha256"],
        expected_calibration_indices_sha256=calibration_config["calibration_indices_sha256"],
        expected_mode_bank_sha256=calibration_config["mode_bank_sha256"],
        expected_parent_v1_sha256="a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959",
    )
    if file_sha256(_resolve(calibration_config["manifest"])) != CALIBRATION_MANIFEST_SHA256:
        raise ValueError("calibration manifest identity is stale")
    return calibration


def _state_digest(model: torch.nn.Module) -> str:
    return canonical_sha256({name: tensor_sha256(value.detach().cpu()) for name, value in model.state_dict().items()})


def _gradient_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    values = [value.detach() for value in gradients if value is not None]
    if not values or not all(torch.isfinite(value).all() for value in values):
        raise ValueError("gradient panel contains no finite gradient")
    return float(torch.sqrt(sum(value.square().sum() for value in values)).cpu())


def _gradient_max_abs_delta(left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]) -> float:
    if len(left) != len(right):
        raise ValueError("gradient vectors do not align")
    deltas: list[torch.Tensor] = []
    for first, second in zip(left, right, strict=True):
        if first is None and second is None:
            continue
        if first is None or second is None:
            return math.inf
        deltas.append((first.detach() - second.detach()).abs().max())
    return float(torch.stack(deltas).max().cpu()) if deltas else 0.0


def _measure_panel(
    model: bogt.ScaledChenBOGTModel,
    images: torch.Tensor,
    indices: Sequence[int],
    calibration: bogt.UnitAwareCalibration,
    prior: torch.Tensor,
    device: torch.device,
    *,
    compare_direct: bool,
) -> dict[str, Any]:
    if len(indices) != LOGICAL_BATCH_SIZE:
        raise ValueError("gradient panel must contain ten samples")
    parameters = tuple(parameter for name, parameter in model.named_parameters() if parameter.requires_grad)
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if names != bogt.EXPECTED_TRAINABLE:
        raise ValueError("gradient panel trainable scope is not phase-only")
    mse_sum = 0.0
    mse_gradients: list[torch.Tensor] | None = None
    z_values: list[torch.Tensor] = []
    for index in indices:
        rendered = base.render_intensity_batch(images[index : index + 1])
        amplitude = rendered.amplitude.to(device)
        target = rendered.target.to(device)
        output = model(amplitude)
        mse = torch.nn.functional.mse_loss(output.camera, target)
        gradients = torch.autograd.grad(mse, parameters, allow_unused=True)
        if mse_gradients is None:
            mse_gradients = [torch.zeros_like(parameter) for parameter in parameters]
        for accumulator, gradient in zip(mse_gradients, gradients, strict=True):
            if gradient is not None:
                accumulator.add_(gradient.detach())
        mse_sum += float(mse.detach().cpu())
        del output, mse, gradients, target, amplitude, rendered
    assert mse_gradients is not None
    mse_average = [value / LOGICAL_BATCH_SIZE for value in mse_gradients]

    for index in indices:
        rendered = base.render_intensity_batch(images[index : index + 1])
        amplitude = rendered.amplitude.to(device)
        coefficients = rcr_b.rcr._encoded_complex_prefix(model, amplitude)
        z_values.append(bogt.normalize_port_power(coefficients.abs().square(), calibration.port_scale).reshape(1, PORT_COUNT))
        del coefficients, amplitude, rendered
    z = torch.cat(z_values, dim=0)
    prior_device = prior.to(device=device, dtype=z.dtype)
    direct_value: torch.Tensor | None = None
    direct_gradients: tuple[torch.Tensor | None, ...] | None = None
    if compare_direct:
        direct_value = bogt.energy_distance(z, prior_device)
        direct_gradients = torch.autograd.grad(direct_value, parameters, retain_graph=True, allow_unused=True)
    two_value = bogt.two_pass_energy_distance_surrogate(z, prior_device, microbatch_size=1)
    two_gradients = torch.autograd.grad(two_value, parameters, allow_unused=True)
    ed_value = float(two_value.detach().cpu())
    record: dict[str, Any] = {
        "indices": list(indices),
        "indices_sha256": tensor_sha256(torch.as_tensor(indices, dtype=torch.long)),
        "mse": mse_sum / LOGICAL_BATCH_SIZE,
        "ed": ed_value,
        "G_MSE": _gradient_norm(mse_average),
        "G_ED": _gradient_norm(two_gradients),
        "ed_gradient_parameter_names": list(names),
        "ed_decoder_gradient_zero": two_gradients[1] is None,
        "ed_finite": math.isfinite(ed_value),
    }
    if direct_value is not None and direct_gradients is not None:
        record["direct_two_pass"] = {
            "scalar_abs_delta": abs(float(direct_value.detach().cpu()) - ed_value),
            "parameter_gradient_max_abs_delta": _gradient_max_abs_delta(direct_gradients, two_gradients),
        }
    del z_values, z, prior_device, two_value, two_gradients
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("no-update panel unexpectedly populated parameter.grad")
    return record


def _reference_gradient_equivalence() -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20260904)
    source = torch.randn((LOGICAL_BATCH_SIZE, PORT_COUNT), generator=generator, dtype=torch.float64)
    prior = torch.randn((PRIOR_BATCH_SIZE, PORT_COUNT), generator=generator, dtype=torch.float64)
    first = source.clone().requires_grad_()
    direct = bogt.energy_distance(first, prior)
    direct_gradient = torch.autograd.grad(direct, first)[0]
    second = source.clone().requires_grad_()
    two = bogt.two_pass_energy_distance_surrogate(second, prior, microbatch_size=1)
    two_gradient = torch.autograd.grad(two, second)[0]
    direct_weight = torch.nn.Parameter(torch.tensor(1.15, dtype=torch.float64))
    parameter_direct = bogt.energy_distance(direct_weight * source, prior)
    parameter_direct_gradient = torch.autograd.grad(parameter_direct, direct_weight)[0]
    two_weight = torch.nn.Parameter(torch.tensor(1.15, dtype=torch.float64))
    parameter_two = bogt.two_pass_energy_distance_surrogate(two_weight * source, prior, microbatch_size=1)
    parameter_two_gradient = torch.autograd.grad(parameter_two, two_weight)[0]
    metrics = {
        "scalar_abs_delta": abs(float(direct.detach()) - float(two.detach())),
        "latent_gradient_max_abs_delta": float((direct_gradient - two_gradient).abs().max()),
        "parameter_gradient_abs_delta": abs(float(parameter_direct_gradient) - float(parameter_two_gradient)),
    }
    metrics["passed"] = all(value <= 1.0e-10 for name, value in metrics.items() if name.endswith("delta"))
    return metrics


def _source_endpoint_authority(config: Mapping[str, Any], sources: Mapping[str, Any]) -> dict[str, Any]:
    endpoints = _mapping(config["endpoints"], "endpoints")
    distances_path = _resolve(config["parent"]["rcr_b_output_directory"]) / config["parent"]["rcr_b_distances"]
    distances = safe_torch_load(distances_path, expected_sha256=RCR_B_DISTANCES_SHA256, map_location="cpu")
    if not isinstance(distances, Mapping):
        raise ValueError("paired prior endpoint distance artifact is invalid")
    holdout = distances.get("d_holdout_to_train")
    prior = distances.get("d_prior_to_train")
    if not torch.is_tensor(holdout) or not torch.is_tensor(prior) or holdout.shape != (EVALUATION_COUNT,) or prior.shape != (EVALUATION_COUNT,) or not torch.isfinite(holdout).all() or not torch.isfinite(prior).all():
        raise ValueError("paired prior endpoint vectors are invalid")
    if distances.get("prior_query", {}).get("seed") != endpoints["prior_query_seed"] or distances.get("prior_query", {}).get("count") != EVALUATION_COUNT:
        raise ValueError("paired prior endpoint sampling identity is stale")
    prior_identity = rcr_b.sample_registered_prior(EVALUATION_COUNT, seed=EVALUATION_SEED)
    if tensor_sha256(prior_identity) != PRIOR_SAMPLE_SHA256:
        raise ValueError("paired prior endpoint prior sample identity is stale")
    panel = sources["parent"]["panel"]
    confirmation_indices = panel["confirmation_indices"]
    mse_map = _mapping(panel["per_sample_mse"], "parent panel per_sample_mse")
    confirmation_mse = _mapping(mse_map["confirmation"], "parent confirmation MSE")
    parent_mse = confirmation_mse.get(endpoints["parent_mse_key"])
    if not torch.is_tensor(parent_mse) or parent_mse.shape != (EVALUATION_COUNT,) or not torch.isfinite(parent_mse).all():
        raise ValueError("parent endpoint MSE vector is invalid")
    if tensor_sha256(confirmation_indices) != config["dataset"]["confirmation_indices_sha256"]:
        raise ValueError("parent endpoint confirmation indices are stale")
    return {
        "paired_prior": {
            "ready": True,
            "artifact_sha256": RCR_B_DISTANCES_SHA256,
            "holdout_vector_sha256": tensor_sha256(holdout),
            "prior_vector_sha256": tensor_sha256(prior),
            "prior_sample_sha256": tensor_sha256(prior_identity),
            "count": EVALUATION_COUNT,
            "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "confidence_level": CONFIDENCE_LEVEL},
            "future_gate": endpoints["paired_prior_endpoint"],
        },
        "parent_relative_mse": {
            "ready": True,
            "panel_sha256": MRA_PANEL_SHA256,
            "vector_sha256": tensor_sha256(parent_mse),
            "key": endpoints["parent_mse_key"],
            "count": EVALUATION_COUNT,
            "future_gate": endpoints["parent_relative_mse_endpoint"],
        },
    }


def _artifact_refs(output: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(output).as_posix(): {
            "path": path.relative_to(output).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


def _write_json(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=kind)


def _assert_output_tree(output: Path) -> None:
    expected = {
        "preflight_manifest.json", "objective_contract.json", "prior_stream_contract.json",
        "gradient_panel_metrics.json", "beta_registration.json", "gradient_equivalence.json",
        "future_canary_contract.json", "compact_report.json", "panel_priors.pt",
    }
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual != expected:
        raise RuntimeError(f"T2-SAAO-PF output tree is not exact: {sorted(actual)}")
    if any(path.name.startswith(".") or ".tmp" in path.name or path.suffix.lower() in {".tif", ".tiff"} for path in output.rglob("*")):
        raise RuntimeError("T2-SAAO-PF output contains hidden, temporary, or TIFF files")


def _resource_preflight(config: Mapping[str, Any], output: Path) -> tuple[int, int, int]:
    execution = _mapping(config["execution"], "execution")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != execution["required_cublas_workspace_config"]:
        raise RuntimeError("BLOCKED_REPLAY_ENVIRONMENT: CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    device = torch.device(execution["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    free, total = torch.cuda.mem_get_info(device)
    if free < execution["min_free_cuda_bytes"]:
        raise RuntimeError("BLOCKED_REPLAY_RESOURCE: less than 4 GiB free CUDA memory")
    disk = shutil.disk_usage(base._repo_root().anchor or str(base._repo_root()))
    if disk.free < execution["min_disk_free_bytes"]:
        raise RuntimeError("BLOCKED_REPLAY_RESOURCE: less than 10 GiB free disk")
    if output.exists():
        raise FileExistsError(f"write-once output already exists: {_relative_path(output)}")
    return int(free), int(total), int(disk.free)


def run(config_path: str | Path = CONFIG_PATH, output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    started = time.perf_counter()
    config_target, config, config_file_hash, config_canonical_hash = load_config(config_path)
    state = base._git_state()
    if state.get("branch") != BRANCH:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_BRANCH: expected {BRANCH}")
    if not state.get("clean"):
        raise RuntimeError("BLOCKED_UNEXPECTED_DIRTY_PATH: preflight requires a clean implementation boundary")
    validate_future_objective_contract(config)
    validate_parent_identity(config)
    validate_prior_contract(config)
    validate_endpoint_bindings(config)
    sources = _load_registered_sources(config)
    output = next_write_once_output(_resolve(output_path))
    free_before, total_cuda, disk_free_before = _resource_preflight(config, output)
    calibration = _load_calibration(config)
    data = rcr_b.rcr.mra._load_data(_resolve(config["dataset"]["root"]))
    if data.manifest.get("manifest_hash") != config["dataset"]["mra_manifest_sha256"] or tensor_sha256(data.confirmation_indices) != config["dataset"]["confirmation_indices_sha256"]:
        raise ValueError("dataset identity does not match the registered parent")
    checkpoint = sources["checkpoint"]
    formal = sources["sources"]["parent"]["formal"]
    device = torch.device(config["execution"]["device"])
    model = bogt.ScaledChenBOGTModel(formal)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    model._set_trainable_contract()
    model.eval()
    if tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad) != bogt.EXPECTED_TRAINABLE:
        raise ValueError("T2-SAAO-PF trainable scope is not phase-only")
    if bogt.mode_bank_sha256(model) != config["calibration"]["mode_bank_sha256"]:
        raise ValueError("T2-SAAO-PF mode-bank identity is stale")
    state_before = _state_digest(model)
    if state_before != PARENT_STATE_DIGEST:
        raise ValueError("parent state digest does not match the registered endpoint")
    reference = _reference_gradient_equivalence()
    panel_priors = [bogt.registration_panel_prior(panel) for panel in range(PANEL_COUNT)]
    panel_records: list[dict[str, Any]] = []
    for panel_id, indices in enumerate(PANEL_INDICES):
        panel_records.append(_measure_panel(model, data.images, indices, calibration, panel_priors[panel_id], device, compare_direct=panel_id == 0))
    state_after = _state_digest(model)
    if state_after != state_before or model.training or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("STATE_ISOLATION_FAIL: no-update gradient panel changed learned state")
    direct_two = panel_records[0]["direct_two_pass"]
    optical_equivalence = {
        "scalar_abs_delta": direct_two["scalar_abs_delta"],
        "parameter_gradient_max_abs_delta": direct_two["parameter_gradient_max_abs_delta"],
        "passed": direct_two["scalar_abs_delta"] <= 1.0e-5 and direct_two["parameter_gradient_max_abs_delta"] <= 1.0e-4,
    }
    equivalence = {"schema": EQUIVALENCE_SCHEMA, "task_id": TASK_ID, "cpu_reference": reference, "optical_panel_0": optical_equivalence, "passed": bool(reference["passed"] and optical_equivalence["passed"])}
    registration = analytic_beta_registration([(row["G_MSE"], row["G_ED"]) for row in panel_records])
    replay_ratios = [float(registration["beta_registered"] * row["G_ED"] / row["G_MSE"]) for row in panel_records]
    ratios_passed = classify_replay_ratios(replay_ratios)
    if not ratios_passed:
        raise RuntimeError("ED_BETA_REGISTRATION_FAIL: replay ratios leave the inclusive [0.05,0.40] boundary")
    registration_payload = {
        "schema": BETA_SCHEMA,
        "task_id": TASK_ID,
        "formula": "beta_i=target_ratio*G_MSE_i/G_ED_i",
        "target_ratio": TARGET_GRADIENT_RATIO,
        "panel_norms": [{"panel": panel, "G_MSE": row["G_MSE"], "G_ED": row["G_ED"], "beta_i": registration["beta_i"][panel]} for panel, row in enumerate(panel_records)],
        "beta_registered": registration["beta_registered"],
        "candidate_count": 1,
        "selection": "analytic_only_no_sweep",
        "replay_ratios": {"p0": replay_ratios[0], "p1": replay_ratios[1], "p2": replay_ratios[2], "minimum": min(replay_ratios), "median": float(torch.tensor(replay_ratios, dtype=torch.float64).median()), "maximum": max(replay_ratios), "inclusive_boundary": [MIN_GRADIENT_RATIO, MAX_GRADIENT_RATIO], "passed": ratios_passed},
    }
    endpoint_authority = _source_endpoint_authority(config, sources["sources"])
    gates = {
        "A_energy_distance_objective_valid": all(row["ed_finite"] for row in panel_records),
        "B_full_batch_gradient_equivalence_pass": equivalence["passed"],
        "C_prior_stream_contract_pass": True,
        "D_ed_beta_registration_pass": ratios_passed,
        "E_paired_prior_endpoint_ready": endpoint_authority["paired_prior"]["ready"],
        "F_parent_relative_safety_endpoint_ready": endpoint_authority["parent_relative_mse"]["ready"],
        "G_state_isolation_pass": state_before == state_after and not model.training,
    }
    classification = "SUPPORT_AWARE_OBJECTIVE_PREFLIGHT_PASS" if all(gates.values()) else "BLOCKED_T2_SAAO_PF_GATE"
    stream_payload = {
        "schema": PRIOR_STREAM_SCHEMA,
        "task_id": TASK_ID,
        "training": {"base_seed": TRAINING_BASE_SEED, "formula": "1238*100000+global_step", "future_steps": [PARENT_STEP + 1, END_STEP], "sample_hashes": {str(step): tensor_sha256(bogt.sample_training_prior(step)) for step in (PARENT_STEP + 1, END_STEP)}},
        "registration": {"namespace": REGISTRATION_NAMESPACE, "formula": "1238*100000+10000+panel_id", "panel_seeds": [registration_panel_seed(panel) for panel in range(PANEL_COUNT)], "panel_prior_hashes": [tensor_sha256(prior) for prior in panel_priors]},
        "evaluation": {"seed": EVALUATION_SEED, "count": EVALUATION_COUNT, "sample_sha256": PRIOR_SAMPLE_SHA256},
        "namespaces_disjoint": True,
    }
    objective_payload = {
        "schema": OBJECTIVE_SCHEMA,
        "task_id": TASK_ID,
        "name": "mse_plus_support_aware_energy_distance",
        "formula": config["objective"]["formula"],
        "estimator": config["objective"]["estimator"],
        "normalization": config["objective"]["normalization"],
        "dimension": PORT_COUNT,
        "logical_batch_size": LOGICAL_BATCH_SIZE,
        "prior_batch_size": PRIOR_BATCH_SIZE,
        "microbatch_size": 1,
        "prior_prior_term_included": True,
        "no_epsilon_smoothing": True,
        "finite_negative_values_allowed": True,
        "alpha_bogt": 0.0,
        "gamma": 0.0,
        "ed_gradient_boundary": {"encoder": "live", "decoder": "zero_direct_gradient", "boundary": "pre_ols_encoded_power_coordinate"},
    }
    gradient_payload = {"schema": GRADIENT_PANEL_SCHEMA, "task_id": TASK_ID, "source": "T1-RCWV_registered_panel_indices", "panels": panel_records, "panel_count": PANEL_COUNT, "state_digest_before": state_before, "state_digest_after": state_after, "no_update": True}
    future_payload = {"schema": FUTURE_SCHEMA, "task_id": TASK_ID, "status": "contract_only_not_run", "parent_step": PARENT_STEP, "start_step": PARENT_STEP, "end_step": END_STEP, "steps": 1000, "optimizer": "Adam", "learning_rate": 0.001, "seed": 1234, "objective": "mse_plus_support_aware_energy_distance", "alpha_bogt": 0.0, "gamma": 0.0, "logging": config["future_canary"]["logging"], "primary_gate": config["future_canary"]["primary_gate"], "safety_gate": config["future_canary"]["safety_gate"], "authorized": False}
    compact_report = {"schema": REPORT_SCHEMA, "task_id": TASK_ID, "status": "completed", "classification": classification, "gates": gates, "beta_registered": registration["beta_registered"], "replay_ratios": replay_ratios, "paired_prior_endpoint_ready": endpoint_authority["paired_prior"]["ready"], "parent_relative_mse_endpoint_ready": endpoint_authority["parent_relative_mse"]["ready"], "interpretation": "preflight-only objective and endpoint readiness evidence; no training, generation, or promotion authority", "governance": GOVERNANCE, "next_action": "IMPLEMENT_SUPPORT_AWARE_ALIGNMENT_CANARY" if classification == "SUPPORT_AWARE_OBJECTIVE_PREFLIGHT_PASS" else "BLOCKED_T2_SAAO_PF_GATE"}
    actual_output = output
    actual_output.mkdir(parents=True, exist_ok=False)
    atomic_torch_save_new(actual_output / "panel_priors.pt", {"schema": PRIOR_STREAM_SCHEMA, "task_id": TASK_ID, "panels": panel_priors, "panel_hashes": [tensor_sha256(value) for value in panel_priors]}, artifact_kind="T2-SAAO-PF panel priors")
    _write_json(actual_output / "objective_contract.json", objective_payload, "T2-SAAO-PF objective contract")
    _write_json(actual_output / "prior_stream_contract.json", stream_payload, "T2-SAAO-PF prior stream contract")
    _write_json(actual_output / "gradient_panel_metrics.json", gradient_payload, "T2-SAAO-PF gradient panels")
    _write_json(actual_output / "beta_registration.json", registration_payload, "T2-SAAO-PF beta registration")
    _write_json(actual_output / "gradient_equivalence.json", equivalence, "T2-SAAO-PF gradient equivalence")
    _write_json(actual_output / "future_canary_contract.json", future_payload, "T2-SAAO-PF future canary contract")
    _write_json(actual_output / "compact_report.json", compact_report, "T2-SAAO-PF compact report")
    refs = _artifact_refs(actual_output)
    storage_bytes = sum(value["size_bytes"] for value in refs.values())
    if storage_bytes > config["output"]["max_storage_bytes"]:
        raise RuntimeError("T2-SAAO-PF output exceeds storage contract")
    manifest = {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "classification": classification,
        "output_directory": _relative_path(actual_output),
        "git": state,
        "config": {"path": _relative_path(config_target), "file_sha256": config_file_hash, "canonical_sha256": config_canonical_hash},
        "parent": {"checkpoint_path": _relative_path(config["parent"]["checkpoint_path"]), "checkpoint_sha256": PARENT_CHECKPOINT_SHA256, "global_step": PARENT_STEP, "state_digest": state_before, "rcr_b_output": _relative_path(config["parent"]["rcr_b_output_directory"]), "rcr_b_distances_sha256": RCR_B_DISTANCES_SHA256, "mra_panel_sha256": MRA_PANEL_SHA256},
        "calibration": {"payload": _relative_path(config["calibration"]["payload"]), "payload_sha256": CALIBRATION_SHA256, "manifest_sha256": CALIBRATION_MANIFEST_SHA256, "contract_version": config["calibration"]["contract_version"]},
        "prior_contract": {"evaluation_seed": EVALUATION_SEED, "evaluation_count": EVALUATION_COUNT, "evaluation_identity_hash": PRIOR_SAMPLE_SHA256, "training_base_seed": TRAINING_BASE_SEED, "registration_panel_seeds": [registration_panel_seed(panel) for panel in range(PANEL_COUNT)]},
        "panels": {"indices": [list(value) for value in PANEL_INDICES], "indices_sha256": list(PANEL_INDEX_SHA256), "source": "T1-RCWV_registered_panel_indices"},
        "endpoint_authority": endpoint_authority,
        "gates": gates,
        "authorization": config["authorization"],
        "state_isolation": {"state_digest_before": state_before, "state_digest_after": state_after, "unchanged": state_before == state_after, "optimizer_created": False, "optimizer_step": False, "backward_called": False, "checkpoint_write": False, "training": False, "generation": False},
        "resource": {"device": str(device), "free_cuda_bytes_before": free_before, "total_cuda_bytes": total_cuda, "disk_free_bytes_before": disk_free_before, "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "wall_clock_seconds": time.perf_counter() - started},
        "artifacts": refs,
        "governance": GOVERNANCE,
    }
    _write_json(actual_output / "preflight_manifest.json", manifest, "T2-SAAO-PF manifest")
    _assert_output_tree(actual_output)
    return {"classification": classification, "output": actual_output, "gates": gates, "beta": registration_payload, "endpoint_authority": endpoint_authority, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"classification": result["classification"], "output": _relative_path(result["output"]), "gates": result["gates"], "replay_ratios": result["beta"]["replay_ratios"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
