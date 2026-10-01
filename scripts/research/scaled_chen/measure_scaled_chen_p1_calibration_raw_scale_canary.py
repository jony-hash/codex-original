"""Frozen Scaled-Chen calibration raw-scale canary.

This diagnostic replays the registered 1000-sample calibration population and
captures the pre-floor per-port ``|c_j|^2`` values.  It never writes a repaired
calibration, creates an optimizer, calls backward, enters the decoder, or
consumes the CMOS/generation path.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_horizon as horizon
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)


TASK_ID = "H-LOMA-GEN-P1-C6-RSC"
HYPOTHESIS = TASK_ID
BRANCH = bogt.BRANCH
EXPECTED_HEAD = "2fcd16a0bef7de2a6483f5ec42070a895bbcbcac"
CALIBRATION_SCHEMA = "scaled_chen_bogt_calibration.v1"
MANIFEST_SCHEMA = "lightgen_scaled_chen_p1_calibration_raw_scale_canary_manifest.v1"
METRICS_SCHEMA = "lightgen_scaled_chen_p1_calibration_raw_scale_canary_metrics.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_p1_calibration_raw_scale_canary_report.v1"
RAW_PORT_MEANS_SCHEMA = "lightgen_scaled_chen_p1_calibration_raw_port_means.v1"

C6_CANARY_PASS = "C6_CANARY_PASS"
C6_CANARY_BLOCKED_REFERENCE_CONTRACT = "C6_CANARY_BLOCKED_REFERENCE_CONTRACT"
C6_CANARY_BLOCKED_NUMERICAL = "C6_CANARY_BLOCKED_NUMERICAL"
INVALID_HARNESS = "invalid_harness"

PORT_COUNT = bogt.PORT_COUNT
CALIBRATION_COUNT = 1000
EXPECTED_CALIBRATION_SCHEMA = CALIBRATION_SCHEMA
EXPECTED_CALIBRATION_SHA256 = "a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959"
EXPECTED_DATASET_MANIFEST_SHA256 = "741e6402da754a740a677dbb343d48c35983f62b59911bbc129dfcc44f9e9ae9"
EXPECTED_CALIBRATION_INDICES_SHA256 = "369c5bded5673f901ae5236eaa51a0c8da17934dc2fab139a205c9fce41d5543"
EXPECTED_MODE_BANK_SHA256 = "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9"
EXPECTED_HORIZON_CONFIG_SHA256 = "aa31584781c3a3f6c91190979fbdd4daed8212cd7aa038fd8b762707e7a37f8f"
EXPECTED_HORIZON_REVIEW_SHA256 = "ca260de2c2d6ade1711047a436ea3f19499edc0599b726005a38f6553d566630"
EXPECTED_FORMAL_CONFIG_SHA256 = "0c42035c1a525e27ff787d013ca27b0f5ae91338329eb0607c7042ebc5ae8905"
EXPECTED_PREFLIGHT_SHA256 = "e988485f18e1f45964bf44e6338308ec882048038a1ba910f09758521fdfea83"
EXPECTED_VALIDATION_SHA256 = "ddb882b26aa1463a74d41896f0f8ff8b406614a29bf25a72b918f58094d3e945"

MAX_FREE_CUDA_BYTES = 4 * 1024**3
MAX_PEAK_CUDA_BYTES = 5 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2
FLOAT32_CAST_RELATIVE_TOLERANCE = float(torch.finfo(torch.float32).eps * 8.0)

DEFAULT_HORIZON_CONFIG_PATH = Path(
    "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_horizon_v2.yaml"
)
DEFAULT_HORIZON_REVIEW_PATH = Path(
    "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_horizon_review_v2.json"
)
DEFAULT_PARENT_CONFIG_PATH = Path(
    "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_v1.yaml"
)
DEFAULT_CALIBRATION_PATH = Path(
    "outputs/20260829_v2/runs/"
    "lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_04/"
    "scaled_chen_bogt_horizon_calibration.pt"
)
DEFAULT_PREFLIGHT_PATH = DEFAULT_CALIBRATION_PATH.parent / "preflight.json"
DEFAULT_VALIDATION_PATH = DEFAULT_CALIBRATION_PATH.parent / "validation.json"
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260903_v2/runs/"
    "lightgen_scaled_chen_p1_calibration_raw_scale_canary_v1_attempt_01"
)


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def next_write_once_output(base_path: str | Path) -> Path:
    """Return the first unused numbered attempt without touching the filesystem."""
    target = Path(base_path)
    match = re.fullmatch(r"(.+)_attempt_(\d+)", target.name)
    if match is None:
        raise ValueError("write-once output must end with _attempt_<number>")
    prefix, number_text = match.groups()
    number = int(number_text)
    while True:
        candidate = target.parent / f"{prefix}_attempt_{number:02d}"
        if not candidate.exists():
            return candidate
        number += 1


def float32_constants() -> dict[str, float]:
    info = torch.finfo(torch.float32)
    return {"eps": float(info.eps), "tiny": float(info.tiny)}


def validate_raw_means(raw_means: torch.Tensor) -> None:
    if raw_means.shape != (PORT_COUNT,):
        raise ValueError("raw means must have shape [100]")
    if not torch.isfinite(raw_means).all():
        raise ValueError("raw means must be finite")
    if (raw_means <= 0).any():
        raise ValueError("raw means must be positive")


def compute_dual_precision_means(
    raw_powers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce the same float32 raw powers with float32 and float64 accumulation."""
    if raw_powers.shape != (CALIBRATION_COUNT, PORT_COUNT):
        raise ValueError("raw calibration powers must have shape [1000,100]")
    if raw_powers.dtype != torch.float32:
        raise TypeError("raw calibration powers must remain float32")
    if not torch.isfinite(raw_powers).all() or (raw_powers < 0).any():
        raise ValueError("raw calibration powers must be finite and non-negative")
    mean_fp32 = raw_powers.mean(dim=0)
    mean_fp64 = raw_powers.to(torch.float64).mean(dim=0)
    return mean_fp32, mean_fp64


def raw_scale_statistics(
    raw_means: torch.Tensor,
    *,
    existing_relative_floor: float | None = None,
) -> dict[str, Any]:
    validate_raw_means(raw_means)
    values = raw_means.detach().to(torch.float64).cpu()
    eps = float32_constants()["eps"]
    tiny = float32_constants()["tiny"]
    relative_floor = (
        float(values.median()) * bogt.POWER_FLOOR_FRACTION
        if existing_relative_floor is None
        else float(existing_relative_floor)
    )
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "p5": float(quantiles[0]),
        "median": float(quantiles[1]),
        "mean": float(values.mean()),
        "p95": float(quantiles[2]),
        "max": float(values.max()),
        "positive_count": int((values > 0).sum()),
        "finite_count": int(torch.isfinite(values).sum()),
        "zero_count": int((values == 0).sum()),
        "count_below_float32_eps": int((values < eps).sum()),
        "count_below_float32_tiny": int((values < tiny).sum()),
        "count_below_existing_relative_floor": int((values < relative_floor).sum()),
        "existing_relative_floor": relative_floor,
    }


def float32_cast_errors(raw_means_fp64: torch.Tensor) -> dict[str, Any]:
    validate_raw_means(raw_means_fp64)
    source = raw_means_fp64.detach().to(torch.float64).cpu()
    cast_back = source.to(torch.float32).to(torch.float64)
    absolute = (cast_back - source).abs()
    relative = absolute / source.abs()
    return {
        "cast_values": cast_back,
        "absolute_error": absolute,
        "relative_error": relative,
        "cast_positive": bool((cast_back > 0).all()),
        "cast_finite": bool(torch.isfinite(cast_back).all()),
        "maximum_absolute_error": float(absolute.max()),
        "maximum_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
        "relative_tolerance": FLOAT32_CAST_RELATIVE_TOLERANCE,
    }


def old_floor_counterfactual(
    raw_means: torch.Tensor,
    registered_scale: torch.Tensor | None = None,
) -> dict[str, Any]:
    validate_raw_means(raw_means)
    values = raw_means.detach().to(torch.float64).cpu()
    constants = float32_constants()
    existing_relative_floor = float(values.median()) * bogt.POWER_FLOOR_FRACTION
    old_floor = max(existing_relative_floor, constants["eps"])
    old_scale = values.clamp_min(old_floor)
    distortion = old_scale / values
    result: dict[str, Any] = {
        "existing_relative_floor": existing_relative_floor,
        "old_floor": old_floor,
        "old_scale": old_scale,
        "distortion": distortion,
        "median_distortion": float(distortion.median()),
        "p95_distortion": float(torch.quantile(distortion, torch.tensor(0.95, dtype=torch.float64))),
        "maximum_distortion": float(distortion.max()),
        "old_scale_unique_count": int(torch.unique(old_scale).numel()),
    }
    if registered_scale is not None:
        registered = registered_scale.detach().to(torch.float64).cpu()
        if registered.shape != values.shape:
            raise ValueError("registered scale shape does not match raw means")
        result["registered_scale_equal_old_scale"] = bool(torch.equal(registered, old_scale))
        result["registered_scale_unique_count"] = int(torch.unique(registered).numel())
    else:
        result["registered_scale_equal_old_scale"] = None
        result["registered_scale_unique_count"] = None
    return result


def raw_coordinate_sanity(
    raw_powers: torch.Tensor,
    raw_means: torch.Tensor,
) -> dict[str, Any]:
    if raw_powers.shape != (CALIBRATION_COUNT, PORT_COUNT):
        raise ValueError("raw calibration powers must have shape [1000,100]")
    validate_raw_means(raw_means)
    z_rawref = raw_powers.to(torch.float64).cpu() / raw_means.to(torch.float64).cpu()
    per_port_mean = z_rawref.mean(dim=0)
    return {
        "per_port_mean": per_port_mean,
        "global_mean": float(z_rawref.mean()),
        "finite": bool(torch.isfinite(z_rawref).all()),
        "minimum": float(z_rawref.min()),
        "maximum": float(z_rawref.max()),
    }


def zero_phase_reference_status(
    parent_contract: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    model_phase_is_zero: bool,
) -> str:
    approved = review.get("approved_contract")
    if not isinstance(approved, Mapping):
        return "UNDERDETERMINED"
    if parent_contract.get("calibrated_phase") != "zero_residual_phase":
        return "UNDERDETERMINED"
    if approved.get("zero_phase_seed") != bogt.SEED:
        return "UNDERDETERMINED"
    return "YES" if model_phase_is_zero else "UNDERDETERMINED"


def classify_c6_gate(
    *,
    all_finite: bool,
    all_positive: bool,
    representable: bool,
    accumulation_ok: bool,
    floor_dominates: bool,
    old_replay_ok: bool,
    reference_status: str,
) -> str:
    if not all_finite or not all_positive or not representable or not accumulation_ok:
        return C6_CANARY_BLOCKED_NUMERICAL
    if reference_status != "YES":
        return C6_CANARY_BLOCKED_REFERENCE_CONTRACT
    if not floor_dominates or not old_replay_ok:
        return C6_CANARY_BLOCKED_NUMERICAL
    return C6_CANARY_PASS


def _state_digest(module: torch.nn.Module) -> str:
    return canonical_sha256(
        {name: tensor_sha256(value) for name, value in module.state_dict().items()}
    )


def _json_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"registered JSON artifact is missing or stale: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"registered JSON artifact is not a mapping: {path}")
    return payload


def _assert_frozen_dataset(data: horizon.HorizonData, historical_manifest: Mapping[str, Any]) -> dict[str, Any]:
    if set(data.manifest) != set(historical_manifest):
        raise ValueError("invalid_harness: current dataset manifest fields differ")
    for key, expected in historical_manifest.items():
        if key in {"dataset_root", "manifest_hash"}:
            continue
        if data.manifest.get(key) != expected:
            raise ValueError(f"invalid_harness: frozen dataset manifest mismatch: {key}")
    reconstructed = dict(data.manifest)
    reconstructed["dataset_root"] = historical_manifest["dataset_root"]
    reconstructed.pop("manifest_hash", None)
    if canonical_sha256(reconstructed) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("invalid_harness: historical dataset manifest cannot be reconstructed")
    if data.manifest["calibration_indices_sha256"] != EXPECTED_CALIBRATION_INDICES_SHA256:
        raise ValueError("invalid_harness: calibration indices binding is stale")
    return {
        "registered_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "current_loader_manifest_sha256": data.manifest["manifest_hash"],
        "calibration_indices_sha256": data.manifest["calibration_indices_sha256"],
        "historical_dataset_root": historical_manifest["dataset_root"],
        "current_dataset_root": data.manifest["dataset_root"],
        "path_only_manifest_difference": data.manifest["manifest_hash"] != EXPECTED_DATASET_MANIFEST_SHA256,
    }


def _load_frozen_lineage(
    horizon_config_path: str | Path,
    review_path: str | Path,
    parent_config_path: str | Path,
    calibration_path: str | Path,
    preflight_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    spec = horizon._load_spec(horizon_config_path)
    horizon._assert_contract(spec)
    review = horizon._load_review(review_path, spec)
    if file_sha256(spec.path) != EXPECTED_HORIZON_CONFIG_SHA256 or file_sha256(_resolve(review_path)) != EXPECTED_HORIZON_REVIEW_SHA256:
        raise ValueError("invalid_harness: horizon config or review bytes are stale")
    parent_spec = bogt._load_spec(parent_config_path)
    bogt._assert_contract(parent_spec)
    formal, formal_path = horizon._load_formal(spec)
    if file_sha256(formal_path) != EXPECTED_FORMAL_CONFIG_SHA256:
        raise ValueError("invalid_harness: formal config bytes are stale")

    preflight = _json_file(_resolve(preflight_path), EXPECTED_PREFLIGHT_SHA256)
    validation = _json_file(_resolve(validation_path), EXPECTED_VALIDATION_SHA256)
    historical_manifest = preflight["contract"]["dataset_manifest"]
    if preflight.get("status") != "passed" or validation.get("status") != "passed":
        raise ValueError("invalid_harness: registered calibration lineage is not passed")
    if preflight["contract"]["calibration_sha256"] != EXPECTED_CALIBRATION_SHA256:
        raise ValueError("invalid_harness: preflight calibration binding is stale")
    if preflight["evidence"]["context"]["dataset_hash"] != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("invalid_harness: preflight dataset binding is stale")
    if preflight["contract"]["mode_bank_sha256"] != EXPECTED_MODE_BANK_SHA256:
        raise ValueError("invalid_harness: preflight mode bank binding is stale")

    data = horizon._load_data(_resolve(spec.payload["dataset_root"]) if "dataset_root" in spec.payload["dataset"] else _resolve("data"))
    dataset_identity = _assert_frozen_dataset(data, historical_manifest)

    calibration_target = _resolve(calibration_path)
    payload = safe_torch_load(calibration_target, expected_sha256=EXPECTED_CALIBRATION_SHA256, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != EXPECTED_CALIBRATION_SCHEMA:
        raise ValueError("invalid_harness: registered calibration schema is stale")
    if payload.get("dataset_manifest_hash") != EXPECTED_DATASET_MANIFEST_SHA256 or int(payload.get("sample_count", -1)) != CALIBRATION_COUNT:
        raise ValueError("invalid_harness: registered calibration dataset binding is stale")
    calibration = bogt.Calibration(
        payload["port_scale"],
        payload["plane_reference"],
        int(payload["sample_count"]),
        payload.get("plane_lower"),
        payload.get("plane_upper"),
    )
    calibration.validate()
    return {
        "horizon_spec": spec,
        "horizon_review": review,
        "parent_spec": parent_spec,
        "formal": formal,
        "formal_path": formal_path,
        "preflight": preflight,
        "validation": validation,
        "data": data,
        "dataset_identity": dataset_identity,
        "calibration_path": calibration_target,
        "calibration_payload": payload,
        "calibration": calibration,
        "preflight_path": _resolve(preflight_path),
        "validation_path": _resolve(validation_path),
    }


def _resource_preflight() -> tuple[int, int]:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_RESOURCE_CONTRACT: CUDA is unavailable")
    free, total = torch.cuda.mem_get_info(device)
    if free < MAX_FREE_CUDA_BYTES:
        raise RuntimeError("BLOCKED_RESOURCE_CONTRACT: insufficient free CUDA memory")
    return int(free), int(total)


def _phase_is_zero(model: bogt.ScaledChenBOGTModel) -> bool:
    named = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if name in bogt.EXPECTED_TRAINABLE
    }
    return tuple(named) == bogt.EXPECTED_TRAINABLE and all(
        torch.equal(value, torch.zeros_like(value)) for value in named.values()
    )


def _artifact_refs(output_path: Path) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for path in sorted(output_path.rglob("*")):
        if path.is_file() and path.name != "raw_scale_manifest.json":
            relative = path.relative_to(output_path).as_posix()
            refs[relative] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    return refs


def _assert_output_tree(output_path: Path) -> None:
    allowed_suffixes = {".json", ".pt", ".png"}
    for path in output_path.rglob("*"):
        if path.name.startswith(".") or path.suffix.lower() in {".tmp", ".tif", ".tiff"}:
            raise ValueError(f"invalid hidden, temporary, or TIFF output: {path}")
        if path.is_file() and path.suffix.lower() not in allowed_suffixes:
            raise ValueError(f"unexpected C6 canary output file: {path}")


def _plot_raw_scale_summary(
    output_path: Path,
    raw_means: torch.Tensor,
    *,
    eps: float,
    tiny: float,
    existing_relative_floor: float,
    old_scale: torch.Tensor,
) -> None:
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    ports = list(range(1, PORT_COUNT + 1))
    values = raw_means.detach().to(torch.float64).cpu().numpy()
    axis.plot(ports, values, color="#1f77b4", linewidth=1.2, marker=".", markersize=3.0, label="raw mean |c_j|²")
    axis.plot(ports, old_scale.detach().to(torch.float64).cpu().numpy(), color="#d62728", linewidth=1.0, label="old registered scale")
    axis.axhline(existing_relative_floor, color="#2ca02c", linestyle="--", linewidth=1.0, label="median × 0.001")
    axis.axhline(eps, color="#9467bd", linestyle=":", linewidth=1.2, label="float32 eps")
    axis.axhline(tiny, color="#8c564b", linestyle="-.", linewidth=1.0, label="float32 tiny")
    axis.set_yscale("log")
    axis.set_xlabel("SMF port index")
    axis.set_ylabel("coefficient power (simulation units)")
    axis.set_title("C6 raw calibration scale before floor")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(loc="lower left", fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(output_path / "raw_scale_summary.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_new_json(path, payload, artifact_kind="C6 canary artifact")


def run(
    horizon_config_path: str | Path = DEFAULT_HORIZON_CONFIG_PATH,
    review_path: str | Path = DEFAULT_HORIZON_REVIEW_PATH,
    parent_config_path: str | Path = DEFAULT_PARENT_CONFIG_PATH,
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    preflight_path: str | Path = DEFAULT_PREFLIGHT_PATH,
    validation_path: str | Path = DEFAULT_VALIDATION_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    state_before_run = base._git_state()
    if state_before_run["branch"] != BRANCH or state_before_run["commit"] != EXPECTED_HEAD:
        raise RuntimeError("invalid_harness: branch or HEAD does not match the registered C6 canary start")

    lineage = _load_frozen_lineage(
        horizon_config_path,
        review_path,
        parent_config_path,
        calibration_path,
        preflight_path,
        validation_path,
    )
    free_before, total_cuda = _resource_preflight()
    device = torch.device("cuda")
    formal = lineage["formal"]
    model = bogt.ScaledChenBOGTModel(formal).to(device)
    mode_hash = bogt.mode_bank_sha256(model)
    if mode_hash != EXPECTED_MODE_BANK_SHA256:
        raise RuntimeError("invalid_harness: current mode bank does not match frozen calibration")
    state_digest_before = _state_digest(model)
    if not _phase_is_zero(model):
        raise RuntimeError("invalid_harness: calibration model is not zero phase")
    torch.cuda.reset_peak_memory_stats(device)

    data: horizon.HorizonData = lineage["data"]
    calibration_batch = base.render_intensity_batch(
        data.images.index_select(0, data.calibration_indices)
    ).amplitude.to(device)
    if calibration_batch.dtype != torch.float32 or tuple(calibration_batch.shape) != (CALIBRATION_COUNT, 1, 200, 200):
        raise RuntimeError("invalid_harness: calibration amplitude contract changed")

    with torch.no_grad():
        raw_powers = bogt.capture_calibration_raw_port_powers(model, calibration_batch)
    torch.cuda.synchronize(device)
    del calibration_batch
    mean_fp32, mean_fp64 = compute_dual_precision_means(raw_powers)
    validate_raw_means(mean_fp64)
    validate_raw_means(mean_fp32)

    constants = float32_constants()
    existing_relative_floor = float(mean_fp64.median()) * bogt.POWER_FLOOR_FRACTION
    counterfactual = old_floor_counterfactual(mean_fp64, lineage["calibration"].port_scale)
    cast = float32_cast_errors(mean_fp64)
    relative_reduction_error = (mean_fp32.to(torch.float64) - mean_fp64).abs() / mean_fp64.abs()
    accumulation_ok = bool(
        torch.isfinite(mean_fp32).all()
        and torch.isfinite(mean_fp64).all()
        and float(relative_reduction_error.max()) <= FLOAT32_CAST_RELATIVE_TOLERANCE
    )
    representable = bool(
        cast["cast_positive"]
        and cast["cast_finite"]
        and cast["maximum_relative_error"] <= FLOAT32_CAST_RELATIVE_TOLERANCE
    )
    sanity = raw_coordinate_sanity(raw_powers, mean_fp64)
    model_phase_is_zero = _phase_is_zero(model)
    reference_status = zero_phase_reference_status(
        lineage["parent_spec"].payload["optical_contract"],
        lineage["horizon_review"],
        model_phase_is_zero=model_phase_is_zero,
    )
    floor_dominates = bool(
        counterfactual["old_floor"] > float(mean_fp64.max())
        and counterfactual["old_scale_unique_count"] == 1
        and counterfactual["old_floor"] / float(mean_fp64.max()) > 100.0
    )
    old_replay_ok = bool(counterfactual["registered_scale_equal_old_scale"])
    gate = classify_c6_gate(
        all_finite=bool(torch.isfinite(mean_fp64).all()),
        all_positive=bool((mean_fp64 > 0).all()),
        representable=representable,
        accumulation_ok=accumulation_ok,
        floor_dominates=floor_dominates,
        old_replay_ok=old_replay_ok,
        reference_status=reference_status,
    )
    run_status = "completed" if gate == C6_CANARY_PASS else "valid_failed"

    state_digest_after = _state_digest(model)
    if state_digest_before != state_digest_after:
        raise RuntimeError("invalid_harness: model state changed during C6 canary")
    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    learned_state_freeze = {
        "state_digest_before": state_digest_before,
        "state_digest_after": state_digest_after,
        "state_unchanged": True,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoint_created": False,
        "refit": False,
        "decoder_entered": False,
        "prior_decoded": False,
        "generation": False,
        "cmos_called": False,
        "mpcdar_called": False,
    }

    per_port_dual_precision = [
        {
            "port": index + 1,
            "mean_fp32": float(mean_fp32[index]),
            "mean_fp64": float(mean_fp64[index]),
            "relative_difference": float(relative_reduction_error[index]),
        }
        for index in range(PORT_COUNT)
    ]
    raw_stats_fp64 = raw_scale_statistics(mean_fp64, existing_relative_floor=existing_relative_floor)
    raw_stats_fp32 = raw_scale_statistics(mean_fp32, existing_relative_floor=existing_relative_floor)
    metrics: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": run_status,
        "gate": gate,
        "calibration": {
            "schema": lineage["calibration_payload"]["schema"],
            "path": str(lineage["calibration_path"]),
            "sha256": EXPECTED_CALIBRATION_SHA256,
            "sample_count": CALIBRATION_COUNT,
            "registered_port_scale_dtype": str(lineage["calibration"].port_scale.dtype),
            "registered_port_scale_unique_count": int(torch.unique(lineage["calibration"].port_scale).numel()),
            "registered_port_scale_all_float32_eps": bool(torch.equal(lineage["calibration"].port_scale, torch.full_like(lineage["calibration"].port_scale, torch.finfo(torch.float32).eps))),
        },
        "lineage": {
            "branch": BRANCH,
            "starting_head": EXPECTED_HEAD,
            "measurement_head": state_before_run["commit"],
            "horizon_config_sha256": file_sha256(lineage["horizon_spec"].path),
            "horizon_review_sha256": file_sha256(_resolve(review_path)),
            "parent_config_sha256": file_sha256(lineage["parent_spec"].path),
            "formal_config_sha256": file_sha256(lineage["formal_path"]),
            "preflight_sha256": file_sha256(lineage["preflight_path"]),
            "validation_sha256": file_sha256(lineage["validation_path"]),
            "mode_bank_sha256": mode_hash,
            "dataset": lineage["dataset_identity"],
            "zero_phase_policy": "independent_train_only_zero_phase_1000_mean_port_power",
            "preprocessing": "MNIST intensity -> bilinear 200x200 -> sqrt amplitude",
            "optical_prefix": "SLM -> front relay -> two encoder phase layers -> fixed attenuation -> pre-OLS modal overlap",
        },
        "zero_phase_reference": {
            "status": reference_status,
            "model_phase_is_zero": model_phase_is_zero,
            "parent_contract_value": lineage["parent_spec"].payload["optical_contract"]["calibrated_phase"],
            "review_zero_phase_seed": lineage["horizon_review"]["approved_contract"]["zero_phase_seed"],
            "learned_state_comparison": {"status": "skipped", "reason": "optional comparison would require an additional optical forward"},
        },
        "raw_means_fp32": raw_stats_fp32,
        "raw_means_fp64": raw_stats_fp64,
        "dual_precision": {
            "aggregation_only": True,
            "optical_forward_dtype": "torch.float32/torch.complex64",
            "per_port": per_port_dual_precision,
            "maximum_relative_difference": float(relative_reduction_error.max()),
            "accumulation_ok": accumulation_ok,
        },
        "float32_cast": {
            "source_dtype": "torch.float64",
            "target_dtype": "torch.float32",
            "cast_positive": cast["cast_positive"],
            "cast_finite": cast["cast_finite"],
            "maximum_absolute_error": cast["maximum_absolute_error"],
            "maximum_relative_error": cast["maximum_relative_error"],
            "mean_relative_error": cast["mean_relative_error"],
            "relative_tolerance": cast["relative_tolerance"],
            "representability": "PASS" if representable else "FAIL",
        },
        "float32_constants": constants,
        "existing_relative_floor": existing_relative_floor,
        "old_absolute_floor": counterfactual["old_floor"],
        "old_floor_counterfactual": {
            "old_scale": counterfactual["old_scale"].tolist(),
            "distortion_ratio_old_scale_over_raw_mean": counterfactual["distortion"].tolist(),
            "median_distortion": counterfactual["median_distortion"],
            "p95_distortion": counterfactual["p95_distortion"],
            "maximum_distortion": counterfactual["maximum_distortion"],
            "old_scale_unique_count": counterfactual["old_scale_unique_count"],
            "registered_scale_equal_old_scale": counterfactual["registered_scale_equal_old_scale"],
            "registered_scale_unique_count": counterfactual["registered_scale_unique_count"],
        },
        "raw_coordinate_sanity": {
            "definition": "z_rawref=p/raw_mean",
            "per_port_mean": sanity["per_port_mean"].tolist(),
            "global_mean": sanity["global_mean"],
            "finite": sanity["finite"],
            "minimum": sanity["minimum"],
            "maximum": sanity["maximum"],
            "implementation_sanity_only": True,
        },
        "learned_state_freeze": learned_state_freeze,
        "resource": {
            "device": str(device),
            "free_cuda_bytes_before": free_before,
            "total_cuda_bytes": total_cuda,
            "peak_cuda_allocated_bytes": peak_cuda_bytes,
            "runtime_seconds": runtime_seconds,
        },
        "optional_learned_state_comparison": {"status": "skipped", "reason": "not needed to decide C6 and would add another optical forward"},
        "governance": {
            "diagnostic_only": True,
            "mainline_state": "closed_at_current_evidence_boundary",
            "reconstruction_manifold_supported": False,
            "generative_manifold_supported": False,
            "generation_readiness": "not_ready",
            "training_authorized": False,
            "formal_promotion": False,
        },
        "next_action": {
            C6_CANARY_PASS: "IMPLEMENT_R2_UNIT_AWARE_CALIBRATION_REPAIR",
            C6_CANARY_BLOCKED_REFERENCE_CONTRACT: "PLAN_R1_CALIBRATION_REFERENCE_CONTRACT",
            C6_CANARY_BLOCKED_NUMERICAL: "PLAN_R3_CALIBRATION_PRECISION_REPAIR",
        }[gate],
    }

    requested_output = _resolve(output_path)
    actual_output = next_write_once_output(requested_output)
    actual_output.mkdir(parents=True, exist_ok=False)
    raw_payload = {
        "schema": RAW_PORT_MEANS_SCHEMA,
        "task_id": TASK_ID,
        "calibration_schema": EXPECTED_CALIBRATION_SCHEMA,
        "calibration_sha256": EXPECTED_CALIBRATION_SHA256,
        "sample_count": CALIBRATION_COUNT,
        "port_count": PORT_COUNT,
        "raw_port_sample_powers": raw_powers,
        "mean_fp32": mean_fp32,
        "mean_fp64": mean_fp64,
        "float32_cast_back": cast["cast_values"],
        "float32_cast_absolute_error": cast["absolute_error"],
        "float32_cast_relative_error": cast["relative_error"],
        "raw_coordinate_per_port_mean": sanity["per_port_mean"],
        "old_scale": counterfactual["old_scale"],
        "old_scale_over_raw_mean": counterfactual["distortion"],
    }
    atomic_torch_save_new(actual_output / "raw_port_means.pt", raw_payload, artifact_kind="C6 raw port means")
    _plot_raw_scale_summary(
        actual_output,
        mean_fp64,
        eps=constants["eps"],
        tiny=constants["tiny"],
        existing_relative_floor=existing_relative_floor,
        old_scale=counterfactual["old_scale"],
    )
    _write_json_new(actual_output / "raw_scale_metrics.json", metrics)
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "result": gate,
        "raw_mean_min": raw_stats_fp64["min"],
        "raw_mean_median": raw_stats_fp64["median"],
        "raw_mean_mean": raw_stats_fp64["mean"],
        "raw_mean_max": raw_stats_fp64["max"],
        "float32_eps": constants["eps"],
        "float32_tiny": constants["tiny"],
        "old_absolute_floor": counterfactual["old_floor"],
        "old_floor_replay": old_replay_ok,
        "interpretation": "diagnostic_only_current_coordinate_floor_audit",
        "governance": metrics["governance"],
        "next_action": metrics["next_action"],
    }
    _write_json_new(actual_output / "compact_report.json", report)
    _assert_output_tree(actual_output)
    artifact_refs = _artifact_refs(actual_output)
    storage_bytes = sum(item["size_bytes"] for item in artifact_refs.values())
    if storage_bytes > MAX_STORAGE_BYTES:
        raise RuntimeError("C6 canary output exceeds the diagnostic storage contract")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": run_status,
        "attempt": actual_output.name,
        "requested_output": str(requested_output),
        "output_directory": str(actual_output),
        "diagnostic_only": True,
        "cuda_replay": True,
        "optical_data_changed": False,
        "source_calibration": {
            "path": str(lineage["calibration_path"]),
            "sha256": EXPECTED_CALIBRATION_SHA256,
            "schema": EXPECTED_CALIBRATION_SCHEMA,
            "immutable": True,
        },
        "lineage": metrics["lineage"],
        "zero_phase_reference": metrics["zero_phase_reference"],
        "gate": gate,
        "artifacts": artifact_refs,
        "storage_bytes_excluding_manifest": storage_bytes,
        "learned_state_freeze": learned_state_freeze,
        "governance": metrics["governance"],
        "next_action": metrics["next_action"],
    }
    _write_json_new(actual_output / "raw_scale_manifest.json", manifest)
    return {"output_directory": str(actual_output), "report": report, "metrics": metrics, "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-config", default=str(DEFAULT_HORIZON_CONFIG_PATH))
    parser.add_argument("--review", default=str(DEFAULT_HORIZON_REVIEW_PATH))
    parser.add_argument("--parent-config", default=str(DEFAULT_PARENT_CONFIG_PATH))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--preflight", default=str(DEFAULT_PREFLIGHT_PATH))
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(
        args.horizon_config,
        args.review,
        args.parent_config,
        args.calibration,
        args.preflight,
        args.validation,
        args.output,
    )
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
