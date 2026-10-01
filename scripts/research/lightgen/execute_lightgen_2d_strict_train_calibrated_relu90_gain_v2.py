"""Execute the strict TRAIN-calibrated ReLU90 successor.

This consumer keeps the optical geometry and objective of the completed ideal
facet run, audits that run's Linear endpoint for exact reuse, calibrates one
global intensity threshold from disjoint TRAIN-only data, and trains only the
new ReLU arm.  It is intentionally an ideal algorithmic surrogate: it does not
claim a Te material model or physical validation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load research authority: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The completed nominal executor is used only as a code authority.  Its old
# reports are consumed below only through the explicit exact-reuse audit.
ENGINE = _load_module(
    Path(__file__).with_name("execute_lightgen_2d_ideal_facet_relu_long_v1.py"),
    "strict_relu90_nominal_code_authority",
)
N6 = ENGINE.N6
N5 = ENGINE.N5
N4 = ENGINE.N4
V10 = ENGINE.V10
V11 = ENGINE.V11


TASK_ID = "EXECUTE_STRICT_TRAIN_CALIBRATED_RELU90_GAIN_V2"
DESIGN_AUTHORITY = "DEFINE_STRICT_TRAIN_CALIBRATED_RELU90_GAIN_V2"
OUTER_GATE = "STRICT_TRAIN_CALIBRATED_RELU90_GAIN_V2"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

RUN_ROOT = Path("outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs")
CONTRACT_DIR = RUN_ROOT / "strict_train_calibrated_relu90_gain_v2_contract_attempt_03"
CONTRACT_PATH = CONTRACT_DIR / "strict_train_calibrated_relu90_gain_v2_contract.v2.json"
PREFLIGHT_DIR = RUN_ROOT / "strict_train_calibrated_relu90_gain_v2_preflight_attempt_03"
PREFLIGHT_REPORT = PREFLIGHT_DIR / "strict_train_calibrated_relu90_gain_v2_preflight_report.v2.json"
PREFLIGHT_LINEAGE = PREFLIGHT_DIR / "strict_train_calibrated_relu90_gain_v2_preflight_lineage.v2.json"
PREFLIGHT_INPUT = PREFLIGHT_DIR / "strict_train_calibrated_relu90_gain_v2_preflight_input.v2.json"
CALIBRATION_DIR = RUN_ROOT / "strict_train_calibrated_relu90_gain_v2_train_calibration_attempt_03"
CALIBRATION_PATH = CALIBRATION_DIR / "strict_train_calibrated_relu90_gain_v2_train_calibration.v2.json"
REUSE_DIR = RUN_ROOT / "strict_train_calibrated_relu90_gain_v2_a_reuse_audit_attempt_03"
REUSE_PATH = REUSE_DIR / "strict_train_calibrated_relu90_gain_v2_a_reuse_audit.v2.json"
B_DIR = RUN_ROOT / "strict_train_calibrated_relu90_gain_v2_b_relu90_attempt_03"
AGGREGATE_DIR = RUN_ROOT / "strict_train_calibrated_relu90_gain_v2_aggregate_attempt_03"

PREVIOUS_A_REPORT = Path(
    "outputs/20260915_ideal_facet_relu_long/runs/"
    "lightgen_2d_ideal_facet_relu_long_a_linear_attempt_05_minimal_ondemand/"
    "lightgen_2d_ideal_facet_relu_long_a_linear_scientific_report.v1.json"
)
PREVIOUS_A_CONTRACT = Path(
    "outputs/20260915_ideal_facet_relu_long/runs/"
    "lightgen_2d_ideal_facet_relu_long_contract_v1_attempt_06_minimal_ondemand/"
    "lightgen_2d_ideal_facet_relu_long_contract.v1.json"
)
PREVIOUS_A_REPORT_SHA256 = "a45767585d846f3e324901751355e149ff8019f1a10093bf2decfafc8b0c22f7"
PREVIOUS_A_CONTRACT_SHA256 = "a8e827845417038a106f2372fdbf9776711d92cc25d971ab01c1d7595d1b77f1"
PREVIOUS_A_FINAL_CHECKPOINT_SHA256 = "c1d41d66d319249eb07363d28480ba700f48abf1e384675db28f8eb95ceb6986"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
ADAPTER_RANK = 9
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = ENGINE.PRODUCTION_GRID
K = 2000
STAGE_SWITCH = 1600
BLOCK_SIZE = 200
SAMPLER_SEED = 1234
PARTITION_SEED = 1931
CALIBRATION_SAMPLES_PER_CLASS = 100
VALIDATION_SAMPLES_PER_CLASS = 100
EXPANDED_SAMPLES_PER_CLASS = 500
CALIBRATION_TARGET = 0.90
CALIBRATION_TOLERANCE = 0.002
VALIDATION_INTERVAL = (0.89, 0.91)
MATERIAL_EPSILON = ENGINE.MATERIAL_EPSILON
MATERIAL_GAIN = 1.0
MATERIAL_SLOPE = 1.0
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
STAGE_A_FIREWALL = 0.937
MATERIAL_DELTA = 0.05
RECONSTRUCTION_MAX_DELTA = 0.005
LAMBDA_TRANSFER = 28.48464149920509
LAMBDA_IMAGE = 13.353282014834436
INITIAL_LR = 1.0e-3
MID_LR = 2.0e-4
FINAL_LR = 5.0e-5
ACTIVE_GPU_BUDGET_SECONDS = 18.0 * 3600.0

STRICT_CHECKPOINT_SCHEMA = "strict_train_calibrated_relu90_gain_v2_checkpoint.v2"
STRICT_SUFFICIENT_SCHEMA = "strict_train_calibrated_relu90_gain_v2_sufficient_observations.v2"
STRICT_DIAGNOSTIC_SCHEMA = "strict_train_calibrated_relu90_gain_v2_diagnostics.v2"
STRICT_RAW_SCHEMA = "strict_train_calibrated_relu90_gain_v2_cyclic_raw_outputs.v2"


class InvalidStrictRun(RuntimeError):
    """Invalid input, implementation, or lineage boundary."""


class HardBlockedStrictRun(RuntimeError):
    """A valid boundary prevents the authorized execution."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidStrictRun(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(root: Path, value: Path) -> str:
    return value.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    _require(_sha256(path) == expected.lower(), f"{name} file-byte SHA-256 mismatch")


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return list(left) == list(right) and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _digest_tree(value: Any) -> str:
    return N4._digest_tree(_cpu_tree(value))


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _geometry() -> dict[str, Any]:
    return {
        "encoder_layers": 2,
        "decoder_layers": 2,
        "layer_spacing_m": 1.50e-3,
        "ols_grid": [10, 10],
        "ols_ports": 100,
        "ols_pixel_pitch_m": 2.0e-6,
        "wavelength_m": 532.0e-9,
        "propagation_grid": PRODUCTION_GRID,
        "new_diffraction_plane": False,
    }


def _objective() -> dict[str, Any]:
    return {
        "formula": "L=L_anchor+lambda_transfer*L_transfer+lambda_image*L_image",
        "lambda_transfer": LAMBDA_TRANSFER,
        "lambda_image": LAMBDA_IMAGE,
    }


def _source(root: Path) -> dict[str, Any]:
    return ENGINE._load_source(root)


def _authorities(root: Path) -> dict[str, Any]:
    return ENGINE._resolve_bundle(root)


def _load_endpoint_payload(path: Path, expected: str | None, name: str) -> Mapping[str, Any]:
    if expected is not None:
        _validate_sha(path, expected, name)
    else:
        _require(path.is_file(), f"{name} is unavailable: {path}")
    return _mapping(N4.safe_torch_load(path, expected_sha256=_sha256(path), map_location="cpu"), name)


def _previous_ref(root: Path, ref: Mapping[str, Any], name: str) -> tuple[Path, dict[str, str]]:
    path_value = ref.get("path")
    digest = ref.get("sha256")
    _require(isinstance(path_value, str) and isinstance(digest, str), f"{name} reference is incomplete")
    path = _path(root, path_value)
    _validate_sha(path, digest, name)
    return path, {"path": _relative(root, path), "sha256": digest}


def _audit_previous_a(root: Path, source: Mapping[str, Any], *, write: bool = True) -> dict[str, Any]:
    """Prove the old A_LINEAR endpoint is an exact matched-control authority."""
    report_path = _path(root, PREVIOUS_A_REPORT)
    contract_path = _path(root, PREVIOUS_A_CONTRACT)
    _validate_sha(report_path, PREVIOUS_A_REPORT_SHA256, "previous A report")
    _validate_sha(contract_path, PREVIOUS_A_CONTRACT_SHA256, "previous A contract")
    report = _read_json(report_path, "previous A report")
    contract = _read_json(contract_path, "previous A contract")
    _require(report.get("result") == "VALID_COMPLETED" and report.get("status") == "valid_completed", "previous A report is not valid")
    _require(report.get("candidate") == "A_LINEAR" and int(report.get("claim_boundary", {}).get("scientific_updates", -1)) == K, "previous A identity changed")
    _require(report.get("source_n5_a") == dict(source["checkpoint_ref"]), "previous A source checkpoint is not the current N5-A authority")
    _require(contract.get("geometry") == _geometry(), "previous A geometry changed")
    _require(contract.get("objective") == _objective(), "previous A objective changed")
    training = _mapping(contract.get("training"), "previous A training contract")
    for key, expected in (("K", K), ("batch", LOGICAL_BATCH_SIZE), ("stage_switch", STAGE_SWITCH), ("sampler_seed", SAMPLER_SEED), ("block_size", BLOCK_SIZE), ("weight_decay", 0.0)):
        _require(training.get(key) == expected, f"previous A training field changed: {key}")
    _require(training.get("optimizer") == "Adam" and training.get("betas") == [0.9, 0.999] and float(training.get("eps")) == 1.0e-8, "previous A optimizer contract changed")
    _require(float(training.get("lr_start")) == INITIAL_LR and float(training.get("lr_switch")) == MID_LR and float(training.get("lr_end")) == FINAL_LR, "previous A learning-rate schedule changed")
    cache = _mapping(contract.get("cache"), "previous A cache contract")
    _require(cache.get("used") is False and cache.get("mode") == "direct_full_path", "previous A did not use the authorized direct path")
    _require(contract.get("data_roles", {}).get("TRAIN") == 55_000 and contract.get("data_roles", {}).get("DEV") == 5_000, "previous A data roles changed")
    _require(report.get("raw_validity_pass") is True, "previous A raw validity is not accepted")

    checkpoint_refs = _mapping(report.get("checkpoints"), "previous A checkpoint references")
    checked_checkpoints: dict[str, dict[str, str]] = {}
    step0_checkpoint: Mapping[str, Any] | None = None
    final_checkpoint: Mapping[str, Any] | None = None
    for step_text in sorted(checkpoint_refs, key=lambda value: int(value)):
        ref = _mapping(checkpoint_refs[step_text], f"previous A checkpoint reference {step_text}")
        checkpoint_path, checkpoint_ref = _previous_ref(root, ref, f"previous A checkpoint {step_text}")
        checkpoint = _load_endpoint_payload(checkpoint_path, str(ref["sha256"]), f"previous A checkpoint {step_text}")
        step = int(step_text)
        _require(checkpoint.get("schema") == "lightgen_2d_ideal_facet_relu_long_checkpoint.v1", f"previous A checkpoint {step} schema changed")
        _require(checkpoint.get("candidate") == "A_LINEAR" and checkpoint.get("checkpoint_step") == step and checkpoint.get("global_step") == 600 + step, f"previous A checkpoint {step} identity changed")
        _require(int(_mapping(checkpoint.get("sampler_state"), f"previous A sampler {step}").get("cursor", -1)) == step, f"previous A sampler cursor changed at {step}")
        checked_checkpoints[str(step)] = {**checkpoint_ref, "step": step, "global_step": 600 + step}
        if step == 0:
            step0_checkpoint = checkpoint
        if step == K:
            final_checkpoint = checkpoint
    _require(step0_checkpoint is not None and final_checkpoint is not None, "previous A endpoint checkpoints are incomplete")
    _require(_tree_equal(step0_checkpoint["runtime_state"], source["checkpoint"]["runtime_state"]), "previous A step-0 runtime state is not the common source")
    _require(_tree_equal(step0_checkpoint["adapter_state"], source["checkpoint"]["adapter_state"]), "previous A step-0 adapter state is not the common source")
    _require(_tree_equal(step0_checkpoint["optimizer_state"], source["checkpoint"]["optimizer_state"]), "previous A step-0 optimizer state is not the common source")
    _require(_sha256(_path(root, checked_checkpoints[str(K)]["path"])) == PREVIOUS_A_FINAL_CHECKPOINT_SHA256, "previous A final checkpoint identity changed")

    sufficient_refs = _mapping(report.get("sufficient_observations"), "previous A sufficient observation references")
    diagnostic_refs = _mapping(report.get("diagnostics"), "previous A diagnostic references")
    endpoints: dict[str, dict[str, Any]] = {}
    for step in (0, K):
        key = str(step)
        sufficient_path, sufficient_ref = _previous_ref(root, _mapping(sufficient_refs.get(key), f"previous A sufficient {key}"), f"previous A sufficient {key}")
        sufficient = _load_endpoint_payload(sufficient_path, sufficient_ref["sha256"], f"previous A sufficient {key}")
        _require(sufficient.get("schema") == "lightgen_2d_ideal_facet_relu_long_sufficient_observations.v1" and sufficient.get("candidate") == "A_LINEAR" and sufficient.get("global_step") == 600 + step, f"previous A sufficient endpoint {key} identity changed")
        dev_correct = sufficient.get("dev_smooth_correct")
        dev_labels = sufficient.get("dev_labels")
        _require(torch.is_tensor(dev_correct) and torch.is_tensor(dev_labels) and dev_correct.shape == dev_labels.shape == (5_000,), f"previous A endpoint {key} DEV observations are incomplete")
        _finite(dev_correct, f"previous A DEV correctness {key}")
        diagnostic_path, diagnostic_ref = _previous_ref(root, _mapping(diagnostic_refs.get(key), f"previous A diagnostic {key}"), f"previous A diagnostic {key}")
        diagnostic = _read_json(diagnostic_path, f"previous A diagnostic {key}")
        _require(diagnostic.get("candidate") == "A_LINEAR" and int(diagnostic.get("step", -1)) == step, f"previous A diagnostic {key} identity changed")
        endpoints[key] = {"sufficient": sufficient, "sufficient_ref": sufficient_ref, "diagnostic": diagnostic, "diagnostic_ref": diagnostic_ref}

    raw_ref_value = report.get("endpoint_raw_outputs")
    raw_path, raw_ref = _previous_ref(root, _mapping(raw_ref_value, "previous A raw output reference"), "previous A final raw outputs")
    raw = _load_endpoint_payload(raw_path, raw_ref["sha256"], "previous A final raw outputs")
    _require(raw.get("candidate") == "A_LINEAR", "previous A raw output candidate changed")
    endpoints[str(K)]["raw_ref"] = raw_ref
    endpoints[str(K)]["raw"] = raw

    audit_payload = {
        "schema": "strict_train_calibrated_relu90_gain_v2_a_reuse_audit.v2",
        "version": 2,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "PASS",
        "status": "valid_completed",
        "decision": "REUSED_VALID_AUTHORITY",
        "consumer": "STRICT_TRAIN_CALIBRATED_RELU90_GAIN_V2",
        "previous_a": {
            "report": {"path": _relative(root, report_path), "sha256": PREVIOUS_A_REPORT_SHA256},
            "contract": {"path": _relative(root, contract_path), "sha256": PREVIOUS_A_CONTRACT_SHA256},
            "checkpoint_refs": checked_checkpoints,
            "endpoint_refs": {key: {"sufficient": value["sufficient_ref"], "diagnostic": value["diagnostic_ref"]} for key, value in endpoints.items()},
            "final_raw": endpoints[str(K)]["raw_ref"],
        },
        "checks": {
            "report_sha256": True,
            "contract_geometry": True,
            "contract_objective": True,
            "contract_training_schedule": True,
            "contract_direct_path": True,
            "source_checkpoint_identity": True,
            "step0_runtime_adapter_optimizer_exact": True,
            "all_checkpoint_sha256_and_reload": True,
            "endpoint_sufficient_observations": True,
            "endpoint_diagnostics": True,
            "final_raw_outputs": True,
            "raw_validity": True,
        },
        "authority_metrics": {
            "R0": float(report["training_delta"]["R0"]),
            "R2000": float(report["training_delta"]["RK"]),
            "DEV_L_image_R0": float(report["reconstruction"]["DEV_L_image_R0"]),
            "DEV_L_image_R2000": float(report["reconstruction"]["DEV_L_image_RK"]),
        },
        "claim_boundary": {"new_scientific_updates": 0, "protected": False, "official_test": False, "sealed": False},
    }
    if write:
        REUSE_DIR.mkdir(parents=True, exist_ok=False)
        N4.atomic_write_new_json(REUSE_PATH, audit_payload, artifact_kind="strict ReLU90 A reuse audit")
    else:
        _validate_sha(REUSE_PATH, _sha256(REUSE_PATH), "strict A reuse audit")
        stored_audit = _read_json(REUSE_PATH, "strict A reuse audit")
        expected_audit = dict(audit_payload)
        stored_without_time = dict(stored_audit)
        expected_audit.pop("created_at_utc", None)
        stored_without_time.pop("created_at_utc", None)
        _require(stored_without_time == expected_audit, "strict A reuse audit changed after preflight")
    return {
        "payload": audit_payload,
        "ref": {"path": _relative(root, REUSE_PATH), "sha256": _sha256(REUSE_PATH)},
        "report": report,
        "contract": contract,
        "endpoints": endpoints,
    }


class FullFacetProbe(ENGINE.LongFacetProbe):
    """Capture all facet intensities for one deterministic TRAIN partition."""

    def __init__(self) -> None:
        super().__init__(mode="relu", tau=0.0, capture=False)
        self.full_intensities: list[torch.Tensor] = []

    def clear(self) -> None:
        super().clear()
        self.full_intensities.clear()

    def apply(self, field: Any) -> Any:
        intensity = field.tensor.abs().square().detach().cpu().to(torch.float32)
        _finite(intensity, "strict calibration facet intensity")
        _require(intensity.ndim >= 3 and tuple(intensity.shape[-2:]) == (100, 100), "strict calibration facet physical shape changed")
        self.full_intensities.append(intensity)
        return super().apply(field)


def _install_full_probe(consumer: Any) -> FullFacetProbe:
    generator = consumer.runtime.generator
    original_forward = generator.forward
    probe = FullFacetProbe()

    def forward_with_facet(field: Any, prefix: str) -> Any:
        return original_forward(probe.apply(field), prefix)

    generator.forward = forward_with_facet
    return probe


def _new_calibration_consumer(bundle: Mapping[str, Any], source: Mapping[str, Any], device: torch.device) -> tuple[Any, FullFacetProbe]:
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V10.V8._load_runtime_state(consumer, dict(source["checkpoint"]["runtime_state"]))
    adapter = V11.LowRankAdapter(device=device)
    adapter.load_state_dict(dict(source["checkpoint"]["adapter_state"]), strict=True)
    N5._configure_candidate(consumer, adapter, "A_CONTINUED_C1", trainable=False)
    return consumer, _install_full_probe(consumer)


def _partition_train(train_indices: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    _require(train_indices.dtype == torch.long and labels.dtype == torch.long, "strict partition role dtype changed")
    generator = torch.Generator(device="cpu").manual_seed(PARTITION_SEED)
    calibration_parts: list[torch.Tensor] = []
    expanded_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    train_labels = labels.index_select(0, train_indices).cpu()
    required = VALIDATION_SAMPLES_PER_CLASS + EXPANDED_SAMPLES_PER_CLASS
    for digit in range(CLASS_COUNT):
        candidates = train_indices.cpu()[train_labels == digit]
        _require(candidates.numel() >= required, f"TRAIN class {digit} is too small for strict calibration")
        order = torch.randperm(candidates.numel(), generator=generator)
        validation_parts.append(candidates.index_select(0, order[:VALIDATION_SAMPLES_PER_CLASS]))
        expanded = candidates.index_select(0, order[VALIDATION_SAMPLES_PER_CLASS:required])
        expanded_parts.append(expanded)
        calibration_parts.append(expanded[:CALIBRATION_SAMPLES_PER_CLASS])
    calibration = torch.cat(calibration_parts)
    expanded = torch.cat(expanded_parts)
    validation = torch.cat(validation_parts)
    _require(torch.unique(torch.cat([calibration, validation])).numel() == calibration.numel() + validation.numel(), "strict calibration and validation sets overlap")
    _require(calibration.numel() == 1_000 and expanded.numel() == 5_000 and validation.numel() == 1_000, "strict calibration partition counts changed")
    return {"calibration": calibration, "expanded": expanded, "validation": validation}


def _capture_intensities(bundle: Mapping[str, Any], source: Mapping[str, Any], images: torch.Tensor, sample_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, probe = _new_calibration_consumer(bundle, source, device)
    with torch.no_grad():
        for start in range(0, sample_ids.numel(), LOGICAL_BATCH_SIZE):
            ids = sample_ids[start : start + LOGICAL_BATCH_SIZE]
            consumer.runtime.forward_from_image(images.index_select(0, ids).to(device))
    expected = math.ceil(sample_ids.numel() / LOGICAL_BATCH_SIZE)
    _require(len(probe.full_intensities) == expected, "strict calibration facet batch capture count changed")
    intensities = torch.cat(probe.full_intensities, dim=0).contiguous()
    _require(intensities.ndim >= 3 and tuple(intensities.shape[0:1]) == (sample_ids.numel(),) and tuple(intensities.shape[-2:]) == (100, 100), "strict calibration intensity sample shape changed")
    _finite(intensities, "strict calibration captured intensity")
    del consumer, probe
    gc.collect()
    torch.cuda.empty_cache()
    return intensities


def _solve_tau(intensities: torch.Tensor, target: float) -> tuple[float, float]:
    values = intensities.to(torch.float64).flatten()
    _finite(values, "strict calibration intensity values")
    total = float(values.sum())
    upper = float(values.max())
    _require(total > 0.0 and upper > 0.0, "strict calibration intensity power is zero")
    lower = 0.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        retention = float(torch.relu(values - middle).sum()) / total
        if retention > target:
            lower = middle
        else:
            upper = middle
    tau = (lower + upper) / 2.0
    retention = float(torch.relu(values - tau).sum()) / total
    return tau, retention


def _retention_stats(intensities: torch.Tensor, tau: float, labels: torch.Tensor) -> dict[str, Any]:
    values = intensities.to(torch.float64)
    flat = values.flatten(1)
    input_power = flat.sum(dim=1)
    output_intensity = torch.relu(values - tau)
    output_power = output_intensity.flatten(1).sum(dim=1)
    eta = output_power / input_power.clamp_min(MATERIAL_EPSILON)
    active = (values > tau).flatten(1).to(torch.float64).mean(dim=1)
    suppressed = (output_intensity <= MATERIAL_EPSILON).flatten(1).to(torch.float64).mean(dim=1)
    for name, value in (("input_power", input_power), ("output_power", output_power), ("eta", eta), ("active", active), ("suppressed", suppressed)):
        _finite(value, f"strict calibration {name}")
    class_eta = []
    class_active = []
    class_suppressed = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        _require(bool(mask.any()), f"strict calibration class {digit} is absent")
        class_eta.append(float(eta[mask].mean()))
        class_active.append(float(active[mask].mean()))
        class_suppressed.append(float(suppressed[mask].mean()))
    quantile_values = torch.quantile(flat.flatten(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
    output_quantiles = torch.quantile(output_intensity.flatten(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
    keys = ("min", "q25", "median", "q75", "max")
    return {
        "sample_count": int(intensities.shape[0]),
        "input_power_total": float(input_power.sum()),
        "output_power_total": float(output_power.sum()),
        "eta": float(output_power.sum() / input_power.sum()),
        "eta_per_sample_mean": float(eta.mean()),
        "eta_per_sample_std": float(eta.std(unbiased=False)),
        "active_fraction": float(active.mean()),
        "suppressed_fraction": float(suppressed.mean()),
        "input_intensity_quantiles": {key: float(value) for key, value in zip(keys, quantile_values.tolist(), strict=True)},
        "output_intensity_quantiles": {key: float(value) for key, value in zip(keys, output_quantiles.tolist(), strict=True)},
        "per_class_eta": [{"class": digit, "eta": class_eta[digit], "active_fraction": class_active[digit], "suppressed_fraction": class_suppressed[digit]} for digit in range(CLASS_COUNT)],
    }


def _calibrate(root: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    partition = _partition_train(train_indices, labels)
    validation_labels = labels.index_select(0, partition["validation"])
    validation_intensities = _capture_intensities(bundle, source, images, partition["validation"], device)
    attempts: list[dict[str, Any]] = []
    for attempt, calibration_ids in (("initial_1000", partition["calibration"]), ("expanded_5000", partition["expanded"])):
        calibration_intensities = _capture_intensities(bundle, source, images, calibration_ids, device)
        calibration_labels = labels.index_select(0, calibration_ids)
        tau, eta_cal = _solve_tau(calibration_intensities, CALIBRATION_TARGET)
        calibration_stats = _retention_stats(calibration_intensities, tau, calibration_labels)
        validation_stats = _retention_stats(validation_intensities, tau, validation_labels)
        attempt_record = {
            "attempt": attempt,
            "calibration_count": int(calibration_ids.numel()),
            "calibration_ids_sha256": V10.V6.tensor_sha256(calibration_ids),
            "tau": tau,
            "eta_cal": eta_cal,
            "eta_train_valid": validation_stats["eta"],
            "calibration_stats": calibration_stats,
            "validation_stats": validation_stats,
            "passed": bool(abs(eta_cal - CALIBRATION_TARGET) <= CALIBRATION_TOLERANCE and VALIDATION_INTERVAL[0] <= validation_stats["eta"] <= VALIDATION_INTERVAL[1]),
        }
        attempts.append(attempt_record)
        if attempt_record["passed"]:
            payload = {
                "schema": "strict_train_calibrated_relu90_gain_v2_train_calibration.v2",
                "version": 2,
                "task_id": TASK_ID,
                "design_authority": DESIGN_AUTHORITY,
                "result": "PASS",
                "status": "valid_completed",
                "calibration_set": {"role": "TRAIN", "count": int(calibration_ids.numel()), "samples_per_class": int(calibration_ids.numel() // CLASS_COUNT), "ids": [int(value) for value in calibration_ids.tolist()], "ids_sha256": V10.V6.tensor_sha256(calibration_ids)},
                "validation_set": {"role": "TRAIN", "disjoint": True, "count": int(partition["validation"].numel()), "samples_per_class": VALIDATION_SAMPLES_PER_CLASS, "ids": [int(value) for value in partition["validation"].tolist()], "ids_sha256": V10.V6.tensor_sha256(partition["validation"])},
                "partition": {"seed": PARTITION_SEED, "class_order": list(range(CLASS_COUNT)), "fixed_order": True, "expanded_candidate_count": int(partition["expanded"].numel()), "no_dev_threshold_selection": True},
                "target": CALIBRATION_TARGET,
                "tolerance": CALIBRATION_TOLERANCE,
                "validation_interval": list(VALIDATION_INTERVAL),
                "tau_90": tau,
                "eta_cal": eta_cal,
                "eta_train_valid": validation_stats["eta"],
                "calibration_stats": calibration_stats,
                "validation_stats": validation_stats,
                "attempts": attempts,
                "facet": {"formula": "E_out=E_in*sqrt((relu(|E_in|^2-tau)+1e-12)/(|E_in|^2+1e-12))", "gain": MATERIAL_GAIN, "slope": MATERIAL_SLOPE, "phase_shift": 0.0, "trainable": False, "placement": "after LightGen OLS coefficient transmission/modal re-emission before decoder"},
                "source_n5_a": source["checkpoint_ref"],
                "claim_boundary": {"train_only": True, "dev_used_for_threshold": False, "protected": False, "official_test": False, "sealed": False, "scientific_updates": 0},
            }
            CALIBRATION_DIR.mkdir(parents=True, exist_ok=False)
            N4.atomic_write_new_json(CALIBRATION_PATH, payload, artifact_kind="strict TRAIN ReLU90 calibration")
            return {"payload": payload, "ref": {"path": _relative(root, CALIBRATION_PATH), "sha256": _sha256(CALIBRATION_PATH)}}
    raise HardBlockedStrictRun("BLOCKED_RELU90_CALIBRATION_NOT_REPRESENTATIVE", "TRAIN validation retention is outside [0.89, 0.91] after one permitted calibration expansion", attempts=attempts)


def _placeholder_cache(train_indices: torch.Tensor, dev_indices: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    sample_ids = torch.cat([train_indices.cpu(), dev_indices.cpu()])
    return {"schema": "strict_train_calibrated_relu90_gain_v2_direct_path_placeholder.v2", "mode": "direct_full_path", "path": "none", "sha256": "none", "sample_count": 0, "shape": [0, OLS_PORT_COUNT], "sample_ids": sample_ids, "train_sample_ids": train_indices.cpu(), "dev_sample_ids": dev_indices.cpu(), "labels": labels.index_select(0, sample_ids).cpu(), "train_count": int(train_indices.numel()), "dev_count": int(dev_indices.numel()), "z_ols": torch.zeros((sample_ids.numel(), OLS_PORT_COUNT), dtype=torch.complex64)}


def _configure_engine() -> None:
    # ENGINE helpers read their schedule constants at call time.  Keep the
    # direct-path objective and replace only the nominal K upper bound.
    ENGINE.K_MAX = K
    ENGINE.K_MIN = K
    ENGINE.BLOCK_SIZE = BLOCK_SIZE
    ENGINE.INITIAL_LR = INITIAL_LR
    ENGINE.MID_LR = MID_LR
    ENGINE.FINAL_LR = FINAL_LR


def _strict_checkpoint_payload(root: Path, contract_ref: Mapping[str, Any], source: Mapping[str, Any], cache_ref: Mapping[str, Any], consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: Any, *, step: int, initial_phase: torch.Tensor, tau: float) -> dict[str, Any]:
    lr, stage = ENGINE._schedule_lr(step, K)
    named = N5._candidate_named(consumer, adapter, "A_CONTINUED_C1")
    return {
        "schema": STRICT_CHECKPOINT_SCHEMA,
        "version": 2,
        "task_id": TASK_ID,
        "candidate": "B_RELU90",
        "contract": dict(contract_ref),
        "source_checkpoint": source["checkpoint_ref"],
        "cache": dict(cache_ref),
        "facet_tau": float(tau),
        "checkpoint_step": step,
        "global_step": 600 + step,
        "total_steps": K,
        "runtime_state": _cpu_tree(consumer.runtime.state_dict()),
        "adapter_state": _cpu_tree(adapter.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "rng_state": _cpu_tree(V10.V6._capture_rng()),
        "schedule_state": {"last_update": step, "stage": stage, "learning_rate": lr, "switch_step": STAGE_SWITCH},
        "trainable_parameter_names": [name for name, parameter in named.items() if parameter.requires_grad],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "facet", "Smooth evaluator", "templates", "prototypes"],
        "initial_generator_phase": _cpu_tree(initial_phase),
        "use_cache": False,
        "block_size": BLOCK_SIZE,
        "terminal_sampler_rule": "checkpoint step s saves sampler cursor=s and does not request s+1",
    }


def _strict_save_checkpoint(root: Path, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    N4.atomic_torch_save_new(path, dict(payload), artifact_kind=f"strict B checkpoint step {payload['checkpoint_step']}")
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "strict checkpoint reload")
    for key in ("schema", "candidate", "checkpoint_step", "global_step", "contract", "facet_tau"):
        _require(loaded.get(key) == payload.get(key), f"strict checkpoint {key} changed on reload")
    for key in ("runtime_state", "adapter_state", "optimizer_state", "sampler_state", "rng_state", "schedule_state"):
        _require(_tree_equal(loaded[key], payload[key]), f"strict checkpoint {key} changed on reload")
    _require(int(_mapping(loaded["sampler_state"], "strict sampler state")["cursor"]) == int(payload["checkpoint_step"]), "strict checkpoint sampler cursor is not terminal for its update index")
    return {"path": _relative(root, path), "sha256": digest, "step": int(payload["checkpoint_step"]), "global_step": int(payload["global_step"]), "reloaded": True}


def _strict_load_checkpoint(root: Path, path: Path, contract_ref: Mapping[str, Any], step: int) -> dict[str, Any]:
    _require(path.is_file(), f"strict B checkpoint {step} is unavailable: {path}")
    digest = _sha256(path)
    checkpoint = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"strict B checkpoint {step}")
    _require(checkpoint.get("schema") == STRICT_CHECKPOINT_SCHEMA and checkpoint.get("candidate") == "B_RELU90" and checkpoint.get("checkpoint_step") == step and checkpoint.get("global_step") == 600 + step and checkpoint.get("total_steps") == K, f"strict B checkpoint {step} identity changed")
    _require(checkpoint.get("contract") == dict(contract_ref), f"strict B checkpoint {step} contract binding changed")
    _require(int(_mapping(checkpoint.get("sampler_state"), f"strict B sampler {step}").get("cursor", -1)) == step, f"strict B sampler cursor changed at {step}")
    return {**dict(checkpoint), "_ref": {"path": _relative(root, path), "sha256": digest, "step": step, "global_step": 600 + step}}


def _strict_context(root: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], cache_ref: Mapping[str, Any], z_train: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, *, step: int, tau: float, contract_ref: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_path = B_DIR / f"b_relu90_step_{step:05d}_checkpoint.v2.pt"
    if step == 0 and not checkpoint_path.is_file():
        checkpoint = source["checkpoint"]
        source_for_model = source
    else:
        checkpoint = _strict_load_checkpoint(root, checkpoint_path, contract_ref, step)
        source_for_model = {"checkpoint": checkpoint, "checkpoint_ref": source["checkpoint_ref"]}
    rng_state = _cpu_tree(checkpoint["rng_state"])
    V10.V6._restore_rng(rng_state)
    consumer, adapter, probe = ENGINE._new_candidate(bundle, source_for_model, device=device, mode="relu", tau=tau, trainable=True)
    optimizer = ENGINE._optimizer(consumer, adapter)
    ENGINE._restore_optimizer(optimizer, checkpoint["optimizer_state"])
    V10.V6._restore_rng(rng_state)
    sampler = ENGINE.BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=K)
    if step > 0:
        sampler.load_state_dict(_mapping(checkpoint["sampler_state"], f"strict B sampler {step}"))
    _require(sampler.cursor == step, f"strict B context sampler cursor changed at {step}")
    expected_lr, expected_stage = ENGINE._schedule_lr(step, K)
    if step > 0:
        schedule = _mapping(checkpoint.get("schedule_state"), f"strict B schedule {step}")
        _require(schedule.get("stage") == expected_stage and math.isclose(float(schedule.get("learning_rate")), expected_lr, rel_tol=0.0, abs_tol=1.0e-15), f"strict B schedule changed at {step}")
    ENGINE._set_stage(consumer, adapter, coadapt=expected_stage == "adapter_decoder_coadaptation")
    ENGINE._set_lr(optimizer, step, K)
    initial_phase = checkpoint.get("initial_generator_phase", source["checkpoint"]["runtime_state"]["generator.phase"])
    return {"consumer": consumer, "adapter": adapter, "probe": probe, "optimizer": optimizer, "sampler": sampler, "checkpoint": checkpoint, "initial_phase": initial_phase.detach().cpu(), "position_map": ENGINE._train_position_map(train_indices), "z_train": z_train, "images": images, "labels": labels, "train_indices": train_indices, "bundle": bundle, "source": source, "cache_ref": cache_ref, "contract_ref": contract_ref, "device": device, "tau": tau, "step": step}


def _strict_monitor(context: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sampler = ENGINE.BalancedMultiEpochSampler(context["train_indices"], context["labels"], seed=SAMPLER_SEED, steps=K)
    indices, batch_labels = sampler.peek_batch()
    positions = ENGINE._cache_positions(context["position_map"], indices)
    return positions, context["images"].index_select(0, indices).to(context["device"]), batch_labels.to(context["device"])


def _strict_save_tensor(root: Path, path: Path, payload: Mapping[str, Any], name: str) -> dict[str, str]:
    N4.atomic_torch_save_new(path, dict(payload), artifact_kind=name)
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"{name} reload")
    _require(loaded.get("schema") == payload.get("schema"), f"{name} schema changed on reload")
    return {"path": _relative(root, path), "sha256": digest}


def _strict_endpoint(root: Path, context: Mapping[str, Any], dev_images: torch.Tensor, dev_labels: torch.Tensor, selection_images: torch.Tensor, selection_labels: torch.Tensor, selection_indices: torch.Tensor, *, step: int, train_metric: Mapping[str, Any], counters: dict[str, int]) -> dict[str, Any]:
    consumer = context["consumer"]
    adapter = context["adapter"]
    probe = context["probe"]
    bundle = context["bundle"]
    mode = "B_RELU90"
    tau = float(context["tau"])
    contract_ref = context["contract_ref"]
    runtime_before = _digest_tree(consumer.runtime.state_dict())
    adapter_before = _digest_tree(adapter.state_dict())
    probe.clear()
    probe.capture = True
    with torch.no_grad():
        counters["scientific_forward_evaluations"] += 1
        evaluation = N4._evaluate(consumer, adapter, bundle["authorities"]["smooth"], dev_images, dev_labels, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"], step=step, train_trace=None)
        counters["scientific_forward_evaluations"] += 1
        cyclic = V10._cyclic_transfer_v10(consumer, adapter, bundle["authorities"]["smooth"], selection_images, selection_labels, selection_indices, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"])
    _require(runtime_before == _digest_tree(consumer.runtime.state_dict()) and adapter_before == _digest_tree(adapter.state_dict()), f"strict B evaluation changed model state at {step}")
    facet = ENGINE._facet_observations(probe, int(dev_images.shape[0]))
    sufficient = N5._canonical_sufficient_payload(contract_ref, mode, step, evaluation, cyclic, dev_labels, train_metric)
    sufficient.update({"schema": STRICT_SUFFICIENT_SCHEMA, "version": 2, "task_id": TASK_ID, "candidate": mode, "global_step": 600 + step, "facet_observations": facet, "facet_material": {"kind": "IDEAL_INTENSITY_RELU_SURROGATE", "mode": mode, "tau": tau, "gain": MATERIAL_GAIN, "slope": MATERIAL_SLOPE, "phase_shift": 0.0, "trainable": False}, "forward_mode": "strict successor scientific endpoint; TRAIN-calibrated tau frozen before DEV evaluation"})
    sufficient_path = B_DIR / f"strict_train_calibrated_relu90_gain_v2_b_relu90_step_{step:05d}_sufficient_observations.v2.pt"
    sufficient_ref = _strict_save_tensor(root, sufficient_path, sufficient, f"strict B sufficient observations step {step}")
    loaded = _mapping(N4.safe_torch_load(_path(root, sufficient_ref["path"]), expected_sha256=sufficient_ref["sha256"], map_location="cpu"), f"strict B sufficient reload {step}")
    diagnostics = {"facet": ENGINE._facet_diagnostics(loaded["facet_observations"], mode=mode, tau=tau), "adapter": N5._adapter_diagnostics(adapter), "generator": N5._generator_diagnostics(consumer, "A_CONTINUED_C1", context["initial_phase"]), "cyclic": N5._cyclic_diagnostics_from_observations(loaded), "direct_diversity": ENGINE._direct_diversity(cyclic["_observations"]["raw_base"], cyclic["_observations"]["labels"])}
    diagnostic_path = B_DIR / f"strict_train_calibrated_relu90_gain_v2_b_relu90_step_{step:05d}_diagnostics.v2.json"
    diagnostic_payload = {"schema": STRICT_DIAGNOSTIC_SCHEMA, "version": 2, "task_id": TASK_ID, "candidate": mode, "step": step, "sufficient_observations": sufficient_ref, "diagnostics": diagnostics}
    N4.atomic_write_new_json(diagnostic_path, diagnostic_payload, artifact_kind=f"strict B endpoint diagnostics step {step}")
    diagnostic_ref = {"path": _relative(root, diagnostic_path), "sha256": _sha256(diagnostic_path)}
    raw_ref = None
    if step == K:
        raw_payload = {"schema": STRICT_RAW_SCHEMA, "version": 2, "task_id": TASK_ID, "candidate": mode, "contract": dict(contract_ref), "selection_indices": cyclic["_observations"]["selection_indices"], "labels": cyclic["_observations"]["labels"], "targets": cyclic["_observations"]["targets"], "raw_base": cyclic["_observations"]["raw_base"], "raw_shifted": cyclic["_observations"]["raw_shifted"]}
        raw_path = B_DIR / f"strict_train_calibrated_relu90_gain_v2_b_relu90_step_{step:05d}_cyclic_raw_outputs.v2.pt"
        raw_ref = _strict_save_tensor(root, raw_path, raw_payload, f"strict B endpoint raw outputs step {step}")
    return {"step": step, "mode": mode, "tau": tau, "global_step": 600 + step, "dev_labels": loaded["dev_labels"], "dev_correct": loaded["dev_smooth_correct"], "evaluation_public": loaded["evaluation_public"], "cyclic_public": loaded["cyclic_public"], "dev_image_loss": float(loaded["dev_image_loss"]), "train_metric": dict(train_metric), "sufficient_ref": sufficient_ref, "diagnostic_ref": diagnostic_ref, "facet_diagnostics": diagnostics["facet"], "raw_ref": raw_ref}


def _strict_sufficient_path(step: int) -> Path:
    return B_DIR / f"strict_train_calibrated_relu90_gain_v2_b_relu90_step_{step:05d}_sufficient_observations.v2.pt"


def _strict_diagnostic_path(step: int) -> Path:
    return B_DIR / f"strict_train_calibrated_relu90_gain_v2_b_relu90_step_{step:05d}_diagnostics.v2.json"


def _strict_checkpoint_path(step: int) -> Path:
    return B_DIR / f"b_relu90_step_{step:05d}_checkpoint.v2.pt"


def _strict_raw_path(step: int) -> Path:
    return B_DIR / f"strict_train_calibrated_relu90_gain_v2_b_relu90_step_{step:05d}_cyclic_raw_outputs.v2.pt"


def _strict_load_endpoint(root: Path, step: int) -> dict[str, Any]:
    sufficient_path = _strict_sufficient_path(step)
    _require(sufficient_path.is_file(), f"strict B sufficient endpoint {step} is unavailable")
    sufficient_digest = _sha256(sufficient_path)
    sufficient = _mapping(N4.safe_torch_load(sufficient_path, expected_sha256=sufficient_digest, map_location="cpu"), f"strict B sufficient endpoint {step}")
    _require(sufficient.get("schema") == STRICT_SUFFICIENT_SCHEMA and sufficient.get("candidate") == "B_RELU90" and sufficient.get("global_step") == 600 + step, f"strict B sufficient endpoint {step} identity changed")
    diagnostic_path = _strict_diagnostic_path(step)
    diagnostic = _read_json(diagnostic_path, f"strict B diagnostic endpoint {step}")
    _require(diagnostic.get("schema") == STRICT_DIAGNOSTIC_SCHEMA and diagnostic.get("candidate") == "B_RELU90" and int(diagnostic.get("step", -1)) == step, f"strict B diagnostic endpoint {step} identity changed")
    raw_ref = None
    if step == K:
        raw_path = _strict_raw_path(step)
        _require(raw_path.is_file(), "strict B final raw outputs are unavailable")
        raw_ref = {"path": _relative(root, raw_path), "sha256": _sha256(raw_path)}
    checkpoint_path = _strict_checkpoint_path(step)
    _require(checkpoint_path.is_file(), f"strict B endpoint checkpoint {step} is unavailable")
    return {"step": step, "sufficient": sufficient, "sufficient_ref": {"path": _relative(root, sufficient_path), "sha256": sufficient_digest}, "diagnostic": diagnostic, "diagnostic_ref": {"path": _relative(root, diagnostic_path), "sha256": _sha256(diagnostic_path)}, "checkpoint_ref": {"path": _relative(root, checkpoint_path), "sha256": _sha256(checkpoint_path), "step": step, "global_step": 600 + step}, "raw_ref": raw_ref, "dev_labels": sufficient["dev_labels"], "dev_correct": sufficient["dev_smooth_correct"], "evaluation_public": sufficient["evaluation_public"], "cyclic_public": sufficient["cyclic_public"], "dev_image_loss": float(sufficient["dev_image_loss"]), "train_metric": sufficient.get("train_metric", {})}


def _initialize_b(root: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], cache_ref: Mapping[str, Any], z_train: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, dev_indices: torch.Tensor, device: torch.device, *, tau: float, contract_ref: Mapping[str, Any]) -> dict[str, Any]:
    _require(not B_DIR.exists(), f"strict B output already exists before initialization: {B_DIR}")
    B_DIR.mkdir(parents=True, exist_ok=False)
    context = _strict_context(root, bundle, source, cache_ref, z_train, images, labels, train_indices, device, step=0, tau=tau, contract_ref=contract_ref)
    dev_images = images.index_select(0, dev_indices).contiguous()
    dev_labels = labels.index_select(0, dev_indices).contiguous()
    selection_images, selection_labels, selection_indices = V10.V6._load_selection(root, images, labels, dev_indices)
    positions, monitor_batch, monitor_labels = _strict_monitor(context)
    train_metric = ENGINE._train_metric(context["consumer"], context["adapter"], context["z_train"].index_select(0, positions).to(device), monitor_batch, monitor_labels, bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), use_cache=False)
    counters = {"scientific_forward_evaluations": 0}
    record = _strict_endpoint(root, context, dev_images, dev_labels, selection_images, selection_labels, selection_indices, step=0, train_metric=train_metric, counters=counters)
    checkpoint_payload = _strict_checkpoint_payload(root, contract_ref, source, cache_ref, context["consumer"], context["adapter"], context["optimizer"], context["sampler"], step=0, initial_phase=context["initial_phase"], tau=tau)
    checkpoint_ref = _strict_save_checkpoint(root, _strict_checkpoint_path(0), checkpoint_payload)
    del context, dev_images, dev_labels, selection_images, selection_labels, selection_indices, monitor_batch
    gc.collect()
    torch.cuda.empty_cache()
    return {"record": record, "checkpoint": checkpoint_ref, "scientific_forward_evaluations": counters["scientific_forward_evaluations"]}


def _run_b_block(root: Path, preflight: Mapping[str, Any], source: Mapping[str, Any], bundle: Mapping[str, Any], cache_ref: Mapping[str, Any], z_train: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, dev_indices: torch.Tensor, device: torch.device, *, tau: float, current_step: int, target_step: int) -> dict[str, Any]:
    contract_ref = preflight["contract_ref"]
    context = _strict_context(root, bundle, source, cache_ref, z_train, images, labels, train_indices, device, step=current_step, tau=tau, contract_ref=contract_ref)
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    batch_z = torch.zeros((LOGICAL_BATCH_SIZE, OLS_PORT_COUNT), dtype=torch.complex64, device=device)
    started = time.perf_counter()
    for step in range(current_step + 1, target_step + 1):
        indices, batch_labels = context["sampler"].next_batch()
        batch = images.index_select(0, indices).to(device)
        ENGINE._train_step(context["consumer"], context["adapter"], context["optimizer"], batch_z, batch, batch_labels.to(device), prototypes, templates, step=step, total_steps=K, use_cache=False)
        del batch
    _require(context["sampler"].cursor == target_step, f"strict B block cursor is invalid at {target_step}")
    final_record = None
    if target_step == K:
        dev_images = images.index_select(0, dev_indices).contiguous()
        dev_labels = labels.index_select(0, dev_indices).contiguous()
        selection_images, selection_labels, selection_indices = V10.V6._load_selection(root, images, labels, dev_indices)
        positions, monitor_batch, monitor_labels = _strict_monitor(context)
        train_metric = ENGINE._train_metric(context["consumer"], context["adapter"], context["z_train"].index_select(0, positions).to(device), monitor_batch, monitor_labels, prototypes, templates, use_cache=False)
        counters = {"scientific_forward_evaluations": 0}
        final_record = _strict_endpoint(root, context, dev_images, dev_labels, selection_images, selection_labels, selection_indices, step=K, train_metric=train_metric, counters=counters)
        del dev_images, dev_labels, selection_images, selection_labels, selection_indices, monitor_batch
    payload = _strict_checkpoint_payload(root, contract_ref, source, cache_ref, context["consumer"], context["adapter"], context["optimizer"], context["sampler"], step=target_step, initial_phase=context["initial_phase"], tau=tau)
    checkpoint_ref = _strict_save_checkpoint(root, _strict_checkpoint_path(target_step), payload)
    elapsed = time.perf_counter() - started
    del batch_z, context
    gc.collect()
    torch.cuda.empty_cache()
    return {"target_step": target_step, "checkpoint": checkpoint_ref, "final_record": final_record, "training_seconds": elapsed}


def _preflight_stage2_smoke(bundle: Mapping[str, Any], source: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, *, tau: float) -> dict[str, Any]:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, adapter, _probe = ENGINE._new_candidate(bundle, source, device=device, mode="relu", tau=tau, trainable=True)
    optimizer = ENGINE._optimizer(consumer, adapter)
    ENGINE._restore_optimizer(optimizer, source["checkpoint"]["optimizer_state"])
    sampler = ENGINE.BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=K)
    indices, batch_labels = sampler.next_batch()
    batch = images.index_select(0, indices).to(device)
    z = torch.zeros((LOGICAL_BATCH_SIZE, OLS_PORT_COUNT), dtype=torch.complex64, device=device)
    before = {name: parameter.detach().cpu().clone() for name, parameter in N5._candidate_named(consumer, adapter, "A_CONTINUED_C1").items()}
    trace = ENGINE._train_step(consumer, adapter, optimizer, z, batch, batch_labels.to(device), bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), step=STAGE_SWITCH + 1, total_steps=K, use_cache=False)
    after = {name: parameter.detach().cpu().clone() for name, parameter in N5._candidate_named(consumer, adapter, "A_CONTINUED_C1").items()}
    adapter_updates = {name: float((after[name] - before[name]).to(torch.float64).square().sum().sqrt()) for name in after if name.startswith("adapter.")}
    _require(all(value == 0.0 for value in adapter_updates.values()), "strict stage-2 adapter was updated in preflight")
    _require(float(trace["update_norms"]["generator.phase"]) > 0.0, "strict stage-2 decoder phase did not update")
    return {"status": "PASS", "step": STAGE_SWITCH + 1, "stage": trace["stage"], "adapter_update_norms": adapter_updates, "generator_update_norm": float(trace["update_norms"]["generator.phase"]), "scientific_updates": 0}


def _make_contract(root: Path, started: Mapping[str, Any], source: Mapping[str, Any], reuse: Mapping[str, Any], calibration: Mapping[str, Any], benchmark: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": "strict_train_calibrated_relu90_gain_v2_contract.v2",
        "version": 2,
        "task_id": TASK_ID,
        "design_authority": DESIGN_AUTHORITY,
        "status": "READY_FOR_BOUNDED_LONG_SCIENTIFIC_EXECUTION",
        "outer_gate": OUTER_GATE,
        "git_boundary": {"branch": started["branch"], "head": started["head"]},
        "source_n5_a_checkpoint": source["checkpoint_ref"],
        "source_n5_a_report": source["report_ref"],
        "geometry": _geometry(),
        "path": "image -> amplitude SLM -> frozen optical encoder -> complex 100-port OLS -> rank-9 adapter T=I+UV^H -> LightGen OLS modal re-emission -> output-facet operator -> existing decoder -> raw CMOS",
        "facet_placement": "after LightGen OLS coefficient transmission and modal re-emission, before decoder",
        "objective": _objective(),
        "arms": {"A": {"candidate": "A_LINEAR", "source": "REUSED_VALID_AUTHORITY", "report": reuse["payload"]["previous_a"]["report"], "final_checkpoint": reuse["payload"]["previous_a"]["checkpoint_refs"][str(K)]}, "B": {"candidate": "B_RELU90", "source": "NEW_STRICT_TRAIN_CALIBRATED_TRAINING"}, "executed": ["A_LINEAR_REUSED", "B_RELU90_TRAINED"], "primary": "B_RELU90_vs_reused_A_LINEAR"},
        "facet": {"kind": "IDEAL_INTENSITY_RELU_SURROGATE", "formula": "E_out=E_in*sqrt((relu(|E_in|^2-tau)+eps)/(|E_in|^2+eps))", "gain": MATERIAL_GAIN, "slope": MATERIAL_SLOPE, "phase_shift": 0.0, "trainable": False, "physical_te_validation": False, "additional_propagation": False, "tau_90": calibration["payload"]["tau_90"], "calibration": calibration["ref"]},
        "calibration": calibration["payload"],
        "training": {"K": K, "batch": LOGICAL_BATCH_SIZE, "stage_switch": STAGE_SWITCH, "stage_schedule": "first 80% adapter+decoder; last 20% decoder only", "optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "lr_start": INITIAL_LR, "lr_switch": MID_LR, "lr_end": FINAL_LR, "sampler": ENGINE.BalancedMultiEpochSampler.SCHEMA, "sampler_seed": SAMPLER_SEED, "block_size": BLOCK_SIZE, "evaluation_fractions": [0.0, 1.0]},
        "budget": benchmark,
        "cache": {"used": False, "mode": "direct_full_path", "path": "none", "sha256": "none", "sample_count": 0, "shape": [0, OLS_PORT_COUNT]},
        "data_roles": {"TRAIN": 55_000, "DEV": 5_000, "calibration": "TRAIN subset only", "validation": "disjoint TRAIN subset only", "protected": 0, "official_test": 0, "sealed": 0},
        "reuse_audit": reuse["ref"],
        "claim_boundary": {"ideal_simulation_only": True, "physical_te_validation": False, "stage_b": False, "latent_prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False},
        "pause_resume": {"checkpoint_state": ["model", "optimizer", "sampler", "RNG", "schedule", "stage", "update_index"], "explicit_resume_required": True},
        "no_retry": True,
        "no_sweep": True,
    }
    CONTRACT_DIR.mkdir(parents=True, exist_ok=False)
    N4.atomic_write_new_json(CONTRACT_PATH, payload, artifact_kind="strict TRAIN-calibrated ReLU90 contract")
    return {"payload": payload, "ref": {"path": _relative(root, CONTRACT_PATH), "sha256": _sha256(CONTRACT_PATH)}}


def _preflight_initial(started: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "strict_train_calibrated_relu90_gain_v2_preflight_report.v2", "version": 2, "task_id": TASK_ID, "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_STRICT_TRAIN_CALIBRATED_RELU90_PREFLIGHT", "created_at_utc": datetime.now(timezone.utc).isoformat(), "outer_gate": OUTER_GATE, "provenance": {"git_start": dict(started)}, "claim_boundary": {"engineering_preflight_executed": False, "scientific_training_executed": False, "scientific_updates": 0, "scientific_forward_evaluations": 0, "stage_b": False, "latent_prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}}


def run_preflight(root: Path) -> dict[str, Any]:
    _configure_engine()
    if RUN_ROOT.exists():
        _require(RUN_ROOT.is_dir(), f"strict successor output root is not a directory: {RUN_ROOT}")
    else:
        RUN_ROOT.mkdir(parents=True, exist_ok=False)
    for path in (CONTRACT_DIR, PREFLIGHT_DIR, CALIBRATION_DIR, REUSE_DIR, B_DIR, AGGREGATE_DIR):
        _require(not path.exists(), f"strict successor attempt output already exists: {path}")
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = ENGINE._status_lines(root)
    payload = _preflight_initial(started)
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from strict successor authority")
        source = _source(root)
        authority = _authorities(root)
        bundle = authority["bundle"]
        V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        if device.type != "cuda":
            raise HardBlockedStrictRun("BLOCKED_CUDA_UNAVAILABLE", "CUDA is required for strict scientific training", device=device_facts)
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, dev_indices, dataset = V10.V6._load_roles(root)
        _require(int(train_indices.numel()) == 55_000 and int(dev_indices.numel()) == 5_000, "TRAIN/DEV role counts changed")
        _require(dataset.get("protected", 0) in (0, [], None) and dataset.get("sealed", 0) in (0, [], None), "protected or sealed role is nonempty")
        reuse = _audit_previous_a(root, source)
        calibration = _calibrate(root, bundle, source, images, labels, train_indices, device)
        tau = float(calibration["payload"]["tau_90"])
        placeholder = _placeholder_cache(train_indices, dev_indices, labels)
        z_train = placeholder["z_ols"][: train_indices.numel()]
        V10.V6._restore_rng(source["checkpoint"]["rng_state"])
        linear_consumer, linear_adapter, _linear_probe = ENGINE._new_candidate(bundle, source, device=device, mode="linear", tau=0.0, trainable=True)
        linear_optimizer = ENGINE._optimizer(linear_consumer, linear_adapter)
        ENGINE._restore_optimizer(linear_optimizer, source["checkpoint"]["optimizer_state"])
        V10.V6._restore_rng(source["checkpoint"]["rng_state"])
        relu_consumer, relu_adapter, _relu_probe = ENGINE._new_candidate(bundle, source, device=device, mode="relu", tau=tau, trainable=True)
        relu_optimizer = ENGINE._optimizer(relu_consumer, relu_adapter)
        ENGINE._restore_optimizer(relu_optimizer, source["checkpoint"]["optimizer_state"])
        common_start_digest = _digest_tree({"runtime": linear_consumer.runtime.state_dict(), "adapter": linear_adapter.state_dict(), "optimizer": linear_optimizer.state_dict()})
        relu_start_digest = _digest_tree({"runtime": relu_consumer.runtime.state_dict(), "adapter": relu_adapter.state_dict(), "optimizer": relu_optimizer.state_dict()})
        _require(common_start_digest == relu_start_digest, "strict A/B common starting model or optimizer state differs")
        del linear_consumer, linear_adapter, linear_optimizer, relu_consumer, relu_adapter, relu_optimizer
        gc.collect()
        torch.cuda.empty_cache()
        smoke = ENGINE._preflight_smoke(bundle, source, z_train, images, labels, train_indices, device, mode="relu", tau=tau, steps=3, use_cache=False)
        stage2 = _preflight_stage2_smoke(bundle, source, images, labels, train_indices, device, tau=tau)
        resume = ENGINE._resume_equivalence(bundle, source, z_train, images, labels, train_indices, device, mode="relu", tau=tau, total_steps=1, use_cache=False)
        sampler_probe = ENGINE._sampler_boundary_probe(train_indices, labels)
        seconds_per_update = max(float(smoke["seconds_per_update"]), 1.0e-6) * 1.25
        estimate = K * seconds_per_update + (K // BLOCK_SIZE + 1) * 1.0 + 30.0
        _require(estimate <= ACTIVE_GPU_BUDGET_SECONDS, "strict B 2000 updates exceed the bounded active-GPU budget")
        benchmark = {"device": device_facts, "cache": {"used": False, "mode": "direct_full_path", "reason": "full OLS cache disabled; direct frozen path is the declared successor entrypoint"}, "smoke": smoke, "stage2_smoke": stage2, "resume_equivalence": resume, "sampler_boundary": sampler_probe, "seconds_per_update_conservative": seconds_per_update, "estimated_active_seconds_B2000": estimate, "active_gpu_budget_seconds": ACTIVE_GPU_BUDGET_SECONDS, "selected_K": K}
        contract = _make_contract(root, started, source, reuse, calibration, benchmark)
        input_payload = {"schema": "strict_train_calibrated_relu90_gain_v2_preflight_input.v2", "version": 2, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "contract": contract["ref"], "source": source["checkpoint_ref"], "reuse_audit": reuse["ref"], "calibration": calibration["ref"], "arms": ["A_LINEAR_REUSED", "B_RELU90"], "K": K, "cache_used": False, "benchmark": benchmark, "data_roles": {"TRAIN": 55_000, "DEV": 5_000, "protected": 0, "official_test": 0, "sealed": 0}, "scientific_updates": 0}
        N4.atomic_write_new_json(PREFLIGHT_INPUT, input_payload, artifact_kind="strict ReLU90 preflight input")
        payload.update({"result": "AUTHORIZABLE", "status": "valid_completed", "classification": "STRICT_TRAIN_CALIBRATED_RELU90_PREFLIGHT_AUTHORIZABLE", "contract": contract["ref"], "authority": {"n5_a_report": source["report_ref"], "n5_a_checkpoint": source["checkpoint_ref"], "reuse_audit": reuse["ref"], "calibration": calibration["ref"]}, "runtime": {"device": device_facts, "production_grid": PRODUCTION_GRID}, "dataset": N6._public(dataset), "arms": ["A_LINEAR_REUSED", "B_RELU90"], "K": K, "calibration": calibration["payload"], "benchmark": benchmark, "preflight_checks": {"authorities": True, "exact_A_reuse_audit": True, "strict_train_calibration": True, "disjoint_train_validation": True, "exact_common_start": True, "three_step_real_updates": True, "stage2_frozen_adapter_update": True, "negative_controls": True, "sampler_epoch_boundary": True, "sampler_resume": True, "RNG_resume": True, "checkpoint_reload": True, "direct_path": True, "device": "cuda", "scientific_updates": 0}, "input": {"path": _relative(root, PREFLIGHT_INPUT), "sha256": _sha256(PREFLIGHT_INPUT)}, "claim_boundary": {"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_updates": 0, "scientific_forward_evaluations": 0, "stage_b": False, "latent_prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}})
    except HardBlockedStrictRun as error:
        payload.update({"result": "BLOCKED", "status": "blocked", "classification": error.code, "hard_blocker": {"message": str(error), **error.details}, "next": "stop before scientific update"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_STRICT_TRAIN_CALIBRATED_RELU90_PREFLIGHT_UNEXPECTED", "invalid_input": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "repair harness and create a fresh successor attempt; no scientific update"})
    finally:
        ending = V10.V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == ENGINE._status_lines(root)})
    N4.atomic_write_new_json(PREFLIGHT_REPORT, payload, artifact_kind="strict ReLU90 preflight report")
    report_ref = {"path": _relative(root, PREFLIGHT_REPORT), "sha256": _sha256(PREFLIGHT_REPORT)}
    lineage = {"schema": "strict_train_calibrated_relu90_gain_v2_preflight_lineage.v2", "version": 2, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "contract": payload.get("contract"), "provenance": payload["provenance"], "claim_boundary": payload["claim_boundary"]}
    N4.atomic_write_new_json(PREFLIGHT_LINEAGE, lineage, artifact_kind="strict ReLU90 preflight lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": _relative(root, PREFLIGHT_LINEAGE), "sha256": _sha256(PREFLIGHT_LINEAGE)}, "contract": payload.get("contract"), "K": payload.get("K"), "arms": payload.get("arms")}


def _load_preflight(root: Path) -> dict[str, Any]:
    report = _read_json(PREFLIGHT_REPORT, "strict ReLU90 preflight report")
    lineage = _read_json(PREFLIGHT_LINEAGE, "strict ReLU90 preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed", "strict ReLU90 preflight is not authorizable")
    report_digest = _sha256(PREFLIGHT_REPORT)
    _require(lineage.get("report", {}).get("sha256") == report_digest, "strict preflight lineage binding changed")
    contract_ref = _mapping(report.get("contract"), "strict successor contract reference")
    contract_path = _path(root, contract_ref["path"])
    _validate_sha(contract_path, str(contract_ref["sha256"]), "strict successor contract")
    contract = _read_json(contract_path, "strict successor contract")
    _require(contract.get("task_id") == TASK_ID and contract.get("outer_gate") == OUTER_GATE, "strict successor contract identity changed")
    _require(contract.get("arms", {}).get("executed") == ["A_LINEAR_REUSED", "B_RELU90_TRAINED"], "strict successor arm set changed")
    _require(int(contract.get("training", {}).get("K", -1)) == K and contract.get("cache", {}).get("used") is False, "strict successor training contract changed")
    reuse_ref = _mapping(contract.get("reuse_audit"), "strict A reuse audit reference")
    reuse_path = _path(root, reuse_ref["path"])
    _validate_sha(reuse_path, str(reuse_ref["sha256"]), "strict A reuse audit")
    calibration_ref = _mapping(contract.get("facet", {}).get("calibration"), "strict calibration reference")
    _validate_sha(_path(root, calibration_ref["path"]), str(calibration_ref["sha256"]), "strict TRAIN calibration")
    _require(report.get("claim_boundary", {}).get("scientific_updates") == 0, "strict preflight scientific boundary changed")
    return {"report": report, "contract": contract, "contract_ref": dict(contract_ref), "report_ref": {"path": _relative(root, PREFLIGHT_REPORT), "sha256": report_digest}, "lineage_ref": {"path": _relative(root, PREFLIGHT_LINEAGE), "sha256": _sha256(PREFLIGHT_LINEAGE)}}


def _strict_arm_report(root: Path, preflight: Mapping[str, Any], source: Mapping[str, Any], calibration: Mapping[str, Any], *, training_seconds: float, resume_sessions: int) -> dict[str, Any]:
    initial = _strict_load_endpoint(root, 0)
    final = _strict_load_endpoint(root, K)
    labels = final["dev_labels"].to(torch.long)
    delta = final["dev_correct"].to(torch.float64) - initial["dev_correct"].to(torch.float64)
    ci = V10.V6._paired_bootstrap(delta, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    raw_validity = bool(final["evaluation_public"]["raw_validity"]["finite"] and final["evaluation_public"]["raw_validity"]["structured"] and final["evaluation_public"]["raw_validity"]["nonconstant"] and final["cyclic_public"]["raw_validity"]["base"]["finite"] and final["cyclic_public"]["raw_validity"]["shifted"]["finite"])
    payload = {"schema": "strict_train_calibrated_relu90_gain_v2_scientific_report.v2", "version": 2, "task_id": TASK_ID, "candidate": "B_RELU90", "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "VALID_COMPLETED", "status": "valid_completed", "classification": "STRICT_B_RELU90_SCIENTIFIC_COMPLETED", "outer_gate": OUTER_GATE, "contract": preflight["contract_ref"], "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"]}, "source_n5_a": source["checkpoint_ref"], "reuse_audit": preflight["contract"].get("reuse_audit"), "calibration": {"ref": preflight["contract"]["facet"]["calibration"], "tau_90": float(calibration["payload"]["tau_90"]), "eta_cal": float(calibration["payload"]["eta_cal"]), "eta_train_valid": float(calibration["payload"]["eta_train_valid"])}, "tau": float(calibration["payload"]["tau_90"]), "evaluations": {"0": initial["evaluation_public"], str(K): final["evaluation_public"]}, "train_metrics": {"0": initial["train_metric"], str(K): final["train_metric"]}, "sufficient_observations": {"0": initial["sufficient_ref"], str(K): final["sufficient_ref"]}, "diagnostics": {"0": initial["diagnostic_ref"], str(K): final["diagnostic_ref"]}, "checkpoints": {"0": initial["checkpoint_ref"], str(K): final["checkpoint_ref"]}, "endpoint_raw_outputs": final["raw_ref"], "training_delta": {"R0": float(initial["evaluation_public"]["smooth"]["macro_recall"]), "RK": float(final["evaluation_public"]["smooth"]["macro_recall"]), "Delta_R": float(delta.mean()), "paired_bootstrap": {key: value for key, value in ci.items() if key != "delta_records"}}, "reconstruction": {"DEV_L_image_R0": float(initial["dev_image_loss"]), "DEV_L_image_RK": float(final["dev_image_loss"]), "Delta_L_image_within_arm": float(final["dev_image_loss"] - initial["dev_image_loss"])}, "cyclic_transfer": final["cyclic_public"], "facet_diagnostics": {"0": initial["diagnostic"].get("diagnostics", {}).get("facet", {}), str(K): final["diagnostic"].get("diagnostics", {}).get("facet", {})}, "raw_validity_pass": raw_validity, "timing": {"training_seconds_current_execution": training_seconds, "pause_resume_sessions": resume_sessions}, "claim_boundary": {"scientific_training_executed": True, "scientific_data_consumed": True, "scientific_updates": K, "scientific_forward_evaluations": 4, "stage_b": False, "latent_prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {"rank": ADAPTER_RANK, "encoder_update": False, "ols_update": False, "w_ols_update": False, "material_trainable": False, "cache_used": False, "sampler": ENGINE.BalancedMultiEpochSampler.SCHEMA, "block_size": BLOCK_SIZE, "no_retry": True, "no_tau_sweep": True}}
    report_path = B_DIR / "strict_train_calibrated_relu90_gain_v2_b_relu90_scientific_report.v2.json"
    N4.atomic_write_new_json(report_path, payload, artifact_kind="strict B ReLU90 scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = B_DIR / "strict_train_calibrated_relu90_gain_v2_b_relu90_scientific_lineage.v2.json"
    N4.atomic_write_new_json(lineage_path, {"schema": "strict_train_calibrated_relu90_gain_v2_scientific_lineage.v2", "version": 2, "task_id": TASK_ID, "candidate": "B_RELU90", "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "contract": payload["contract"], "preflight": payload["preflight"], "claim_boundary": payload["claim_boundary"]}, artifact_kind="strict B ReLU90 scientific lineage")
    return {"report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "payload": payload}


def _cyclic_pass(cyclic: Mapping[str, Any]) -> bool:
    return bool(cyclic.get("template", {}).get("pass") and cyclic.get("smooth", {}).get("pass") and cyclic.get("broad") and not cyclic.get("pair_dominated"))


def _aggregate(root: Path, preflight: Mapping[str, Any], source: Mapping[str, Any], reuse: Mapping[str, Any], b_report: Mapping[str, Any], *, training_seconds: float, resume_sessions: int) -> dict[str, Any]:
    _require(not AGGREGATE_DIR.exists(), f"strict aggregate output already exists: {AGGREGATE_DIR}")
    a_report = reuse["report"]
    a_final = reuse["endpoints"][str(K)]["sufficient"]
    a_initial = reuse["endpoints"]["0"]["sufficient"]
    b_final = _strict_load_endpoint(root, K)
    b_initial = _strict_load_endpoint(root, 0)
    labels = a_final["dev_labels"].to(torch.long)
    _require(torch.equal(labels, b_final["dev_labels"].to(torch.long)), "reused A and strict B endpoint DEV labels are not matched")
    delta = b_final["dev_correct"].to(torch.float64) - a_final["dev_smooth_correct"].to(torch.float64)
    ci = V10.V6._paired_bootstrap(delta, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    image_delta = float(b_final["dev_image_loss"] - float(a_final["dev_image_loss"]))
    transfer_a = _cyclic_pass(a_final["cyclic_public"])
    transfer_b = _cyclic_pass(b_final["cyclic_public"])
    raw = bool(a_report.get("raw_validity_pass") and b_report.get("raw_validity_pass"))
    supported = bool(float(delta.mean()) >= MATERIAL_DELTA and float(ci["ci_lower"]) > 0.0 and image_delta <= RECONSTRUCTION_MAX_DELTA and transfer_a and transfer_b and raw)
    stage_a_best = max(float(a_report["training_delta"]["RK"]), float(b_report["training_delta"]["RK"]))
    stage_a = {"best_R": stage_a_best, "threshold": STAGE_A_FIREWALL, "status": "STAGE_A_FIREWALL_PASSED" if stage_a_best >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False}
    per_class = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        per_class.append({"class": digit, "A_recall": float(a_final["dev_smooth_correct"][mask].to(torch.float64).mean()), "B_recall": float(b_final["dev_correct"][mask].to(torch.float64).mean()), "B_minus_A": float(delta[mask].mean())})
    b_facet0 = b_report["facet_diagnostics"]["0"]
    b_facetk = b_report["facet_diagnostics"][str(K)]
    b_diag0 = _read_json(_path(root, b_report["diagnostics"]["0"]["path"]), "strict B step-0 diagnostics")
    b_diagk = _read_json(_path(root, b_report["diagnostics"][str(K)]["path"]), "strict B step-2000 diagnostics")
    mechanism = {"eta0": b_facet0.get("power_retention_mean"), "eta2000": b_facetk.get("power_retention_mean"), "active_fraction0": b_facet0.get("activation_mean"), "active_fraction2000": b_facetk.get("activation_mean"), "suppressed_fraction0": b_facet0.get("suppression_mean"), "suppressed_fraction2000": b_facetk.get("suppression_mean"), "tau_drift": 0.0, "bypass_if_eta_drifts_to_one": bool(float(b_facetk.get("power_retention_mean", 0.0)) >= 0.98), "calibration": preflight["contract"]["calibration"], "adapter_residual": {"step0": b_diag0.get("diagnostics", {}).get("adapter", {}), "step2000": b_diagk.get("diagnostics", {}).get("adapter", {})}, "decoder_phase_change": {"step0": b_diag0.get("diagnostics", {}).get("generator", {}), "step2000": b_diagk.get("diagnostics", {}).get("generator", {})}, "raw_finite_nonconstant_structured": {"A": bool(a_report.get("raw_validity_pass")), "B": bool(b_report.get("raw_validity_pass"))}}
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=False)
    payload = {"schema": "strict_train_calibrated_relu90_gain_v2_aggregate.v2", "version": 2, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "VALID_COMPLETED", "status": "valid_completed", "classification": "STRICT_TRAIN_CALIBRATED_RELU90_GAIN_SUPPORTED" if supported else "STRICT_TRAIN_CALIBRATED_RELU90_NOT_SUPPORTED", "outer_gate": OUTER_GATE, "contract": preflight["contract_ref"], "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"]}, "reuse_audit": reuse["ref"], "candidates": {"A_LINEAR_REUSED": {"report": {"path": _relative(root, PREVIOUS_A_REPORT), "sha256": PREVIOUS_A_REPORT_SHA256}, "RK": float(a_report["training_delta"]["RK"]), "checkpoint": reuse["payload"]["previous_a"]["checkpoint_refs"][str(K)]}, "B_RELU90": {"report": {"path": _relative(root, B_DIR / "strict_train_calibrated_relu90_gain_v2_b_relu90_scientific_report.v2.json"), "sha256": _sha256(B_DIR / "strict_train_calibrated_relu90_gain_v2_b_relu90_scientific_report.v2.json")}, "RK": float(b_report["training_delta"]["RK"]), "checkpoint": b_report["checkpoints"][str(K)]}}, "primary": {"A": "A_LINEAR_REUSED", "B": "B_RELU90", "R_B0": float(b_initial["evaluation_public"]["smooth"]["macro_recall"]), "R_B2000": float(b_final["evaluation_public"]["smooth"]["macro_recall"]), "R_A2000": float(a_final["evaluation_public"]["smooth"]["macro_recall"]), "Delta_R_B_minus_A2000": float(delta.mean()), "paired_bootstrap": {key: value for key, value in ci.items() if key != "delta_records"}, "image_delta_B_minus_A2000": image_delta, "reconstruction_max_delta": RECONSTRUCTION_MAX_DELTA, "cyclic_transfer_A": transfer_a, "cyclic_transfer_B": transfer_b, "raw_validity": raw, "supported": supported}, "per_class": per_class, "trajectory": {"A_LINEAR_REUSED": {"R0": float(a_report["training_delta"]["R0"]), "RK": float(a_report["training_delta"]["RK"]), "Delta_R": float(a_report["training_delta"]["Delta_R"])}, "B_RELU90": {"R0": float(b_report["training_delta"]["R0"]), "RK": float(b_report["training_delta"]["RK"]), "Delta_R": float(b_report["training_delta"]["Delta_R"])}}, "facet_diagnostics": {"B_RELU90": b_report["facet_diagnostics"]}, "mechanism_diagnostics": mechanism, "stage_a": stage_a, "timing": {"training_seconds_current_execution": training_seconds, "pause_resume_sessions": resume_sessions}, "claim_boundary": {"new_training_updates": K, "reused_A_updates": 0, "per_arm_new_updates": {"A_LINEAR": 0, "B_RELU90": K}, "ideal_simulation_only": True, "physical_te_validation": False, "stage_b": False, "latent_prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {"cache_used": False, "sampler": ENGINE.BalancedMultiEpochSampler.SCHEMA, "block_size": BLOCK_SIZE, "no_retry": True, "no_tau_sweep": True, "geometry_unchanged": True, "a_reuse": "REUSED_VALID_AUTHORITY"}, "next": "visualization may consume this aggregate; Stage B, generation, and physical Te validation remain closed"}
    report_path = AGGREGATE_DIR / "strict_train_calibrated_relu90_gain_v2_aggregate_report.v2.json"
    N4.atomic_write_new_json(report_path, payload, artifact_kind="strict ReLU90 aggregate report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = AGGREGATE_DIR / "strict_train_calibrated_relu90_gain_v2_aggregate_lineage.v2.json"
    N4.atomic_write_new_json(lineage_path, {"schema": "strict_train_calibrated_relu90_gain_v2_aggregate_lineage.v2", "version": 2, "task_id": TASK_ID, "result": payload["result"], "classification": payload["classification"], "report": report_ref, "contract": payload["contract"], "preflight": payload["preflight"], "claim_boundary": payload["claim_boundary"]}, artifact_kind="strict ReLU90 aggregate lineage")
    return {"result": payload["result"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "primary": payload["primary"], "stage_a": stage_a}


def run_scientific(root: Path, *, resume_b_step: int | None = None) -> dict[str, Any]:
    _configure_engine()
    preflight = _load_preflight(root)
    source = _source(root)
    authority = _authorities(root)
    bundle = authority["bundle"]
    V10.V6._prepare_cuda_runtime_dll_search_path()
    device, device_facts = V10.V6._cuda_facts()
    _require(device.type == "cuda", "CUDA is required for strict scientific execution")
    bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
    images, labels, train_indices, dev_indices, dataset = V10.V6._load_roles(root)
    _require(dataset.get("protected", 0) in (0, [], None) and dataset.get("sealed", 0) in (0, [], None), "protected or sealed role is nonempty")
    reuse = _audit_previous_a(root, source, write=False)
    calibration = {"payload": preflight["contract"]["calibration"], "ref": preflight["contract"]["facet"]["calibration"]}
    tau = float(calibration["payload"]["tau_90"])
    cache_ref = preflight["contract"]["cache"]
    placeholder = _placeholder_cache(train_indices, dev_indices, labels)
    z_train = placeholder["z_ols"][: train_indices.numel()]
    started = time.perf_counter()
    resume_sessions = 1 if resume_b_step is not None else 0
    if resume_b_step is None:
        _initialize_b(root, bundle, source, cache_ref, z_train, images, labels, train_indices, dev_indices, device, tau=tau, contract_ref=preflight["contract_ref"])
        current_step = 0
    else:
        _require(isinstance(resume_b_step, int) and 0 <= resume_b_step <= K and resume_b_step % BLOCK_SIZE == 0, "explicit B resume step must be a checkpoint boundary")
        _require(B_DIR.is_dir(), "explicit B resume directory is unavailable")
        _require(_strict_checkpoint_path(resume_b_step).is_file(), f"explicit B resume checkpoint {resume_b_step} is unavailable")
        current_step = resume_b_step
    print(json.dumps({"event": "scientific_start", "arms": ["A_LINEAR_REUSED", "B_RELU90"], "K": K, "A_source": "REUSED_VALID_AUTHORITY", "resume": resume_b_step is not None, "device": device_facts}, ensure_ascii=False), flush=True)
    training_seconds = 0.0
    while current_step < K:
        target_step = min(K, current_step + BLOCK_SIZE)
        block = _run_b_block(root, preflight, source, bundle, cache_ref, z_train, images, labels, train_indices, dev_indices, device, tau=tau, current_step=current_step, target_step=target_step)
        current_step = target_step
        training_seconds += float(block["training_seconds"])
        print(json.dumps({"event": "strict_relu90_progress", "candidate": "B_RELU90", "step": current_step, "K": K, "A_source": "REUSED_VALID_AUTHORITY", "training_seconds_current_execution": training_seconds}, ensure_ascii=False), flush=True)
    b_result = _strict_arm_report(root, preflight, source, calibration, training_seconds=training_seconds, resume_sessions=resume_sessions)
    aggregate = _aggregate(root, preflight, source, reuse, b_result["payload"], training_seconds=training_seconds, resume_sessions=resume_sessions)
    aggregate["arm"] = b_result["report"]
    aggregate["runtime"] = {"device": device_facts, "scientific_wall_seconds_current_execution": time.perf_counter() - started}
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific", "all"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--resume-b-step", type=int, default=None, help="explicit last B checkpoint step")
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    if args.action == "preflight":
        result = run_preflight(root)
    elif args.action == "scientific":
        result = run_scientific(root, resume_b_step=args.resume_b_step)
    else:
        _require(args.resume_b_step is None, "--action all does not accept --resume-b-step")
        result = run_preflight(root)
        if result.get("result") != "AUTHORIZABLE":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2
        result = run_scientific(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("result") in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
