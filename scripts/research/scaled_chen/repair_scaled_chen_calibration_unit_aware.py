"""Create and validate the repaired Scaled-Chen calibration contract.

This is a calibration-only post-processing step.  It consumes the immutable C6
raw-power replay and the historical v1 calibration, registers each positive
raw modal-power mean directly, and does not run a new optical forward.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
)


TASK_ID = "H-LOMA-GEN-P1-R2-UACR"
HYPOTHESIS = TASK_ID
BRANCH = bogt.BRANCH
R2_CALIBRATION_SCHEMA = bogt.UNIT_AWARE_CALIBRATION_SCHEMA
R2_CONTRACT_VERSION = bogt.UNIT_AWARE_CALIBRATION_CONTRACT_VERSION
R2_GATE = "R2_CALIBRATION_REPAIR_PASS"
R2_INVALID = "R2_REPAIR_INVALID"
C6_GATE = "C6_CANARY_PASS"
C6_MANIFEST_SCHEMA = "lightgen_scaled_chen_p1_calibration_raw_scale_canary_manifest.v1"
C6_METRICS_SCHEMA = "lightgen_scaled_chen_p1_calibration_raw_scale_canary_metrics.v1"
C6_RAW_SCHEMA = "lightgen_scaled_chen_p1_calibration_raw_port_means.v1"
PARENT_V1_SCHEMA = "scaled_chen_bogt_calibration.v1"
CALIBRATION_COUNT = 1000
PORT_COUNT = bogt.PORT_COUNT
MEAN_TOLERANCE = 1.0e-5
FLOOR_POLICY = "none_for_positive_finite_raw_means"
ZERO_PHASE_REFERENCE = "independent_train_only_zero_phase"
PLANE_IDENTITY = "pre_ols_plane_power_reference_v1"
OPTICAL_FORWARD_DTYPE = "torch.float32/torch.complex64"
NEXT_ACTION = "PLAN_REPAIRED_COORDINATE_REPLAY"

DEFAULT_CONFIG_PATH = Path(
    "configs/research/scaled_chen/"
    "lightgen_scaled_chen_bogt_calibration_contract_repair_v2.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_01"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    return base._resolve_path(value)


def _relative_path(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return target.relative_to(_repo_root()).as_posix()
    except ValueError:
        return str(target)


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


def _load_json(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    actual_sha256 = file_sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"JSON artifact content no longer matches {_relative_path(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact {_relative_path(path)} must be an object")
    return payload


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = _load_json(target)
    required = {
        "schema",
        "task_id",
        "hypothesis",
        "repaired_schema",
        "contract_version",
        "source_c6",
        "parent_v1",
        "lineage",
        "floor_policy",
        "zero_phase_reference",
        "plane_identity",
        "optical_forward_dtype",
        "prior_contract",
        "governance",
        "verification",
    }
    if set(payload) != required:
        raise ValueError("R2 calibration repair config fields are invalid")
    if payload["schema"] != "lightgen_scaled_chen_bogt_calibration_contract_repair_v2":
        raise ValueError("R2 calibration repair config schema is invalid")
    if payload["task_id"] != TASK_ID or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("R2 calibration repair task identity is invalid")
    if payload["repaired_schema"] != R2_CALIBRATION_SCHEMA:
        raise ValueError("R2 repaired calibration schema is invalid")
    if payload["contract_version"] != R2_CONTRACT_VERSION:
        raise ValueError("R2 calibration contract version is invalid")
    if payload["floor_policy"] != FLOOR_POLICY:
        raise ValueError("R2 calibration floor policy is invalid")
    if payload["zero_phase_reference"] != ZERO_PHASE_REFERENCE:
        raise ValueError("R2 zero-phase reference is invalid")
    if payload["plane_identity"] != PLANE_IDENTITY:
        raise ValueError("R2 plane identity is invalid")
    if payload["optical_forward_dtype"] != OPTICAL_FORWARD_DTYPE:
        raise ValueError("R2 optical forward dtype contract is invalid")
    if payload["prior_contract"] != {
        "kind": "truncated_normal",
        "mean": bogt.PRIOR_MEAN,
        "variance": bogt.PRIOR_VARIANCE,
        "lower": 0.0,
        "unconditional": True,
    }:
        raise ValueError("R2 prior contract must remain unchanged")
    if payload["governance"] != {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }:
        raise ValueError("R2 governance boundary is invalid")
    if payload["verification"] != {
        "source_gate": C6_GATE,
        "cuda_replay": False,
        "downstream_replay": False,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoint_created": False,
        "refit": False,
        "training": False,
        "prior_generation": False,
    }:
        raise ValueError("R2 calibration-only verification boundary is invalid")
    for section_name in ("source_c6", "parent_v1", "lineage"):
        if not isinstance(payload[section_name], Mapping):
            raise ValueError(f"R2 config section {section_name} must be an object")
    return target, payload, file_sha256(target), canonical_sha256(payload)


def _require_artifact_ref(directory: Path, name: str, reference: Mapping[str, Any]) -> Path:
    if set(reference) != {"sha256", "size_bytes"}:
        raise ValueError(f"C6 artifact reference for {name} is invalid")
    target = directory / name
    if not target.is_file():
        raise ValueError(f"C6 artifact is missing: {_relative_path(target)}")
    if file_sha256(target) != reference["sha256"]:
        raise ValueError(f"C6 artifact hash is stale: {name}")
    if target.stat().st_size != int(reference["size_bytes"]):
        raise ValueError(f"C6 artifact size is stale: {name}")
    return target


def _validate_c6_freeze(freeze: Mapping[str, Any]) -> None:
    forbidden = (
        "optimizer_created",
        "backward_called",
        "checkpoint_created",
        "refit",
        "decoder_entered",
        "prior_decoded",
        "generation",
        "cmos_called",
        "mpcdar_called",
    )
    if any(freeze.get(name) is not False for name in forbidden):
        raise ValueError("C6 source freeze contains a forbidden operation")
    if freeze.get("state_unchanged") is not True:
        raise ValueError("C6 source learned state was not unchanged")
    if freeze.get("state_digest_before") != freeze.get("state_digest_after"):
        raise ValueError("C6 source state digests do not match")


def _validate_raw_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != C6_RAW_SCHEMA:
        raise ValueError("C6 raw payload schema is stale")
    if payload.get("task_id") != "H-LOMA-GEN-P1-C6-RSC":
        raise ValueError("C6 raw payload task identity is stale")
    if payload.get("calibration_schema") != PARENT_V1_SCHEMA:
        raise ValueError("C6 raw payload parent calibration schema is stale")
    if int(payload.get("sample_count", -1)) != CALIBRATION_COUNT or int(payload.get("port_count", -1)) != PORT_COUNT:
        raise ValueError("C6 raw payload dimensions are stale")
    raw_powers = payload.get("raw_port_sample_powers")
    mean_fp32 = payload.get("mean_fp32")
    mean_fp64 = payload.get("mean_fp64")
    if not torch.is_tensor(raw_powers) or raw_powers.dtype != torch.float32 or tuple(raw_powers.shape) != (CALIBRATION_COUNT, PORT_COUNT):
        raise ValueError("C6 raw powers must remain float32 [1000,100]")
    if not torch.is_tensor(mean_fp32) or mean_fp32.dtype != torch.float32 or tuple(mean_fp32.shape) != (PORT_COUNT,):
        raise ValueError("C6 FP32 raw means must remain float32 [100]")
    if not torch.is_tensor(mean_fp64) or mean_fp64.dtype != torch.float64 or tuple(mean_fp64.shape) != (PORT_COUNT,):
        raise ValueError("C6 FP64 raw means must remain float64 [100]")
    if not torch.isfinite(raw_powers).all() or (raw_powers < 0).any():
        raise ValueError("C6 raw powers must be finite and non-negative")
    if not torch.isfinite(mean_fp32).all() or (mean_fp32 <= 0).any() or not torch.isfinite(mean_fp64).all() or (mean_fp64 <= 0).any():
        raise ValueError("C6 raw means must be finite and positive")
    if not torch.equal(mean_fp32, raw_powers.mean(dim=0)):
        raise ValueError("C6 FP32 raw means do not replay the raw powers")
    if not torch.equal(mean_fp64, raw_powers.to(torch.float64).mean(dim=0)):
        raise ValueError("C6 FP64 raw means do not replay the raw powers")
    cast_back = mean_fp64.to(torch.float32).to(torch.float64)
    if not torch.allclose(cast_back, mean_fp32.to(torch.float64), rtol=1.0e-6, atol=1.0e-18):
        raise ValueError("C6 dual-precision raw means are inconsistent")
    if not bool((mean_fp32 < torch.finfo(torch.float32).eps).all()):
        raise ValueError("C6 source does not document the registered absolute-epsilon failure")
    return {
        "raw_powers": raw_powers.detach().cpu(),
        "mean_fp32": mean_fp32.detach().cpu(),
        "mean_fp64": mean_fp64.detach().cpu(),
    }


def _load_c6_source(config: Mapping[str, Any]) -> dict[str, Any]:
    source_config = config["source_c6"]
    directory = _resolve(source_config["directory"])
    manifest_path = directory / str(source_config["manifest"])
    manifest = _load_json(manifest_path, expected_sha256=str(source_config["manifest_sha256"]))
    if manifest.get("schema") != C6_MANIFEST_SCHEMA or manifest.get("task_id") != "H-LOMA-GEN-P1-C6-RSC":
        raise ValueError("C6 source manifest schema or task identity is stale")
    if manifest.get("status") != "completed" or manifest.get("gate") != C6_GATE:
        raise ValueError("C6 source gate is not C6_CANARY_PASS")
    if manifest.get("diagnostic_only") is not True or manifest.get("optical_data_changed") is not False:
        raise ValueError("C6 source authorization boundary is invalid")
    if manifest.get("next_action") != "IMPLEMENT_R2_UNIT_AWARE_CALIBRATION_REPAIR":
        raise ValueError("C6 source next-action binding is stale")
    source_calibration = manifest.get("source_calibration")
    if not isinstance(source_calibration, Mapping) or source_calibration.get("schema") != PARENT_V1_SCHEMA or source_calibration.get("immutable") is not True:
        raise ValueError("C6 source parent calibration binding is invalid")
    if source_calibration.get("sha256") != config["parent_v1"]["sha256"]:
        raise ValueError("C6 source parent calibration hash is stale")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("C6 source artifact manifest is invalid")
    raw_name = str(source_config["raw_port_means"])
    metrics_name = str(source_config["metrics"])
    raw_path = _require_artifact_ref(directory, raw_name, artifacts.get(raw_name, {}))
    metrics_path = _require_artifact_ref(directory, metrics_name, artifacts.get(metrics_name, {}))
    if file_sha256(raw_path) != source_config["raw_port_means_sha256"] or file_sha256(metrics_path) != source_config["metrics_sha256"]:
        raise ValueError("C6 source artifact hash binding is stale")
    metrics = _load_json(metrics_path, expected_sha256=str(source_config["metrics_sha256"]))
    if metrics.get("schema") != C6_METRICS_SCHEMA or metrics.get("task_id") != "H-LOMA-GEN-P1-C6-RSC" or metrics.get("gate") != C6_GATE:
        raise ValueError("C6 source metrics gate is stale")
    counterfactual = metrics.get("old_floor_counterfactual")
    if not isinstance(counterfactual, Mapping) or counterfactual.get("registered_scale_equal_old_scale") is not True:
        raise ValueError("C6 old-floor replay evidence is missing")
    if metrics.get("zero_phase_reference", {}).get("status") != "YES":
        raise ValueError("C6 zero-phase reference is not approved")
    freeze = manifest.get("learned_state_freeze")
    if not isinstance(freeze, Mapping):
        raise ValueError("C6 learned-state freeze is missing")
    _validate_c6_freeze(freeze)
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping) or not isinstance(lineage.get("dataset"), Mapping):
        raise ValueError("C6 lineage is invalid")
    expected_lineage = config["lineage"]
    if lineage["dataset"].get("registered_manifest_sha256") != expected_lineage["dataset_manifest_sha256"]:
        raise ValueError("C6 dataset manifest binding is stale")
    if lineage["dataset"].get("calibration_indices_sha256") != expected_lineage["calibration_indices_sha256"]:
        raise ValueError("C6 calibration indices binding is stale")
    for field in (
        "mode_bank_sha256",
        "measurement_head",
        "horizon_config_sha256",
        "horizon_review_sha256",
        "formal_config_sha256",
        "parent_config_sha256",
        "preflight_sha256",
        "validation_sha256",
    ):
        if lineage.get(field) != expected_lineage[field]:
            raise ValueError(f"C6 lineage binding is stale: {field}")
    if lineage.get("zero_phase_policy") != "independent_train_only_zero_phase_1000_mean_port_power":
        raise ValueError("C6 zero-phase population binding is stale")
    raw_payload = safe_torch_load(raw_path, expected_sha256=str(source_config["raw_port_means_sha256"]), map_location="cpu")
    if not isinstance(raw_payload, Mapping):
        raise ValueError("C6 raw payload must be a mapping")
    raw = _validate_raw_payload(raw_payload)
    return {
        "directory": directory,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": file_sha256(manifest_path),
        "raw_path": raw_path,
        "raw_sha256": file_sha256(raw_path),
        "metrics_path": metrics_path,
        "metrics": metrics,
        "raw_payload": raw,
        "freeze": dict(freeze),
    }


def _load_parent_v1(config: Mapping[str, Any]) -> dict[str, Any]:
    parent_config = config["parent_v1"]
    if parent_config["schema"] != PARENT_V1_SCHEMA:
        raise ValueError("historical v1 calibration config schema is stale")
    path = _resolve(parent_config["path"])
    before = file_sha256(path)
    if before != parent_config["sha256"]:
        raise ValueError("historical v1 calibration hash is stale")
    payload = safe_torch_load(path, expected_sha256=str(parent_config["sha256"]), map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != PARENT_V1_SCHEMA:
        raise ValueError("historical v1 calibration schema is stale")
    if payload.get("dataset_manifest_hash") != config["lineage"]["dataset_manifest_sha256"] or int(payload.get("sample_count", -1)) != CALIBRATION_COUNT:
        raise ValueError("historical v1 calibration dataset binding is stale")
    calibration = bogt.Calibration(
        port_scale=payload["port_scale"],
        plane_reference=payload["plane_reference"],
        sample_count=int(payload["sample_count"]),
        plane_lower=payload["plane_lower"],
        plane_upper=payload["plane_upper"],
    )
    calibration.validate()
    return {"path": path, "payload": payload, "calibration": calibration, "sha256_before": before}


def calibration_z_sanity(raw_powers: torch.Tensor, registered_scale: torch.Tensor) -> dict[str, Any]:
    """Check the descriptive z=p/raw-scale mean contract without a forward pass."""
    if raw_powers.dtype != torch.float32 or tuple(raw_powers.shape) != (CALIBRATION_COUNT, PORT_COUNT):
        raise ValueError("raw powers must be float32 [1000,100]")
    if registered_scale.shape != (PORT_COUNT,) or not torch.is_floating_point(registered_scale):
        raise ValueError("registered scale must be floating point [100]")
    if not torch.isfinite(raw_powers).all() or (raw_powers < 0).any():
        raise ValueError("raw powers must be finite and non-negative")
    if not torch.isfinite(registered_scale).all() or (registered_scale <= 0).any():
        raise ValueError("registered scale must be finite and positive")
    z = raw_powers.to(torch.float64).cpu() / registered_scale.to(torch.float64).cpu()
    per_port_mean = z.mean(dim=0)
    per_port_variance = z.var(dim=0, unbiased=False)
    return {
        "definition": "z_j=abs(c_j)^2/registered_scale_j",
        "per_port_mean": per_port_mean.tolist(),
        "global_mean": float(z.mean()),
        "per_port_variance": per_port_variance.tolist(),
        "global_variance": float(z.var(unbiased=False)),
        "finite": bool(torch.isfinite(z).all()),
        "minimum": float(z.min()),
        "maximum": float(z.max()),
        "mean_tolerance": MEAN_TOLERANCE,
    }


def _scale_statistics(values: torch.Tensor) -> dict[str, Any]:
    if values.shape != (PORT_COUNT,) or not torch.is_floating_point(values):
        raise ValueError("scale statistics require floating point [100] values")
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("scale statistics require finite positive values")
    view = values.detach().to(torch.float64).cpu()
    quantiles = torch.quantile(view, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    eps = float(torch.finfo(torch.float32).eps)
    tiny = float(torch.finfo(torch.float32).tiny)
    return {
        "count": int(view.numel()),
        "min": float(view.min()),
        "p5": float(quantiles[0]),
        "median": float(quantiles[1]),
        "mean": float(view.mean()),
        "p95": float(quantiles[2]),
        "max": float(view.max()),
        "finite": True,
        "positive": True,
        "zero_count": int((view == 0).sum()),
        "count_below_float32_eps": int((view < eps).sum()),
        "count_below_float32_tiny": int((view < tiny).sum()),
        "unique_count": int(torch.unique(view).numel()),
    }


def build_repaired_calibration_payload(
    calibration: bogt.UnitAwareCalibration,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize the v2 contract while keeping raw and registered scales explicit."""
    calibration.validate()
    metadata_copy = dict(metadata)
    required_metadata = {
        "task_id",
        "contract_version",
        "dataset_manifest_hash",
        "calibration_indices_sha256",
        "zero_phase_reference",
        "mode_bank_sha256",
        "plane_identity",
        "raw_scale_dtype",
        "registered_scale_dtype",
        "raw_scale_statistics",
        "floor_policy",
        "parent_v1_artifact_sha256",
        "parent_v1_schema",
        "source_c6_manifest_sha256",
        "source_c6_payload_sha256",
        "producer_git",
        "prior_contract",
        "optical_forward_dtype",
    }
    missing = sorted(required_metadata.difference(metadata_copy))
    if missing:
        raise ValueError(f"v2 calibration metadata is missing: {', '.join(missing)}")
    if metadata_copy["task_id"] != TASK_ID or metadata_copy["contract_version"] != R2_CONTRACT_VERSION:
        raise ValueError("v2 calibration metadata identity is invalid")
    if metadata_copy["raw_scale_dtype"] != str(calibration.raw_port_mean.dtype) or metadata_copy["registered_scale_dtype"] != str(calibration.port_scale.dtype):
        raise ValueError("v2 calibration dtype metadata is inconsistent")
    reserved = {
        "schema",
        "raw_port_mean",
        "port_scale",
        "plane_reference",
        "plane_lower",
        "plane_upper",
        "sample_count",
        "port_count",
    }
    if reserved.intersection(metadata_copy):
        raise ValueError("v2 calibration metadata attempts to override contract fields")
    payload: dict[str, Any] = {
        "schema": R2_CALIBRATION_SCHEMA,
        "contract_version": metadata_copy.pop("contract_version"),
        "raw_port_mean": calibration.raw_port_mean.detach().cpu().clone(),
        "port_scale": calibration.port_scale.detach().cpu().clone(),
        "plane_reference": calibration.plane_reference.detach().cpu().clone(),
        "plane_lower": None if calibration.plane_lower is None else calibration.plane_lower.detach().cpu().clone(),
        "plane_upper": None if calibration.plane_upper is None else calibration.plane_upper.detach().cpu().clone(),
        "sample_count": calibration.sample_count,
        "port_count": PORT_COUNT,
    }
    payload.update(metadata_copy)
    return payload


def _validate_registered_payload(
    payload: Mapping[str, Any],
    *,
    expected_dataset_manifest_hash: str,
    expected_calibration_indices_sha256: str,
    expected_mode_bank_sha256: str,
    expected_parent_v1_sha256: str,
) -> bogt.UnitAwareCalibration:
    if payload.get("schema") != R2_CALIBRATION_SCHEMA:
        raise ValueError(f"registered consumer requires {R2_CALIBRATION_SCHEMA}")
    if payload.get("contract_version") != R2_CONTRACT_VERSION:
        raise ValueError("registered v2 calibration contract version is invalid")
    required = {
        "schema",
        "contract_version",
        "task_id",
        "dataset_manifest_hash",
        "calibration_indices_sha256",
        "mode_bank_sha256",
        "parent_v1_artifact_sha256",
        "parent_v1_schema",
        "zero_phase_reference",
        "floor_policy",
        "plane_identity",
        "optical_forward_dtype",
        "prior_contract",
        "raw_scale_statistics",
        "raw_port_mean",
        "port_scale",
        "plane_reference",
        "plane_lower",
        "plane_upper",
        "sample_count",
        "port_count",
        "raw_scale_dtype",
        "registered_scale_dtype",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"registered v2 calibration fields are missing: {', '.join(missing)}")
    if payload.get("task_id") != TASK_ID:
        raise ValueError("registered v2 calibration task identity is invalid")
    if payload.get("dataset_manifest_hash") != expected_dataset_manifest_hash:
        raise ValueError("registered v2 dataset manifest binding is stale")
    if payload.get("calibration_indices_sha256") != expected_calibration_indices_sha256:
        raise ValueError("registered v2 calibration indices binding is stale")
    if payload.get("mode_bank_sha256") != expected_mode_bank_sha256:
        raise ValueError("registered v2 mode-bank binding is stale")
    if payload.get("parent_v1_artifact_sha256") != expected_parent_v1_sha256:
        raise ValueError("registered v2 parent v1 binding is stale")
    if payload.get("parent_v1_schema") != PARENT_V1_SCHEMA:
        raise ValueError("registered v2 parent schema is stale")
    if payload.get("zero_phase_reference") != ZERO_PHASE_REFERENCE or payload.get("floor_policy") != FLOOR_POLICY:
        raise ValueError("registered v2 calibration semantic policy is invalid")
    if payload.get("plane_identity") != PLANE_IDENTITY or payload.get("optical_forward_dtype") != OPTICAL_FORWARD_DTYPE:
        raise ValueError("registered v2 optical identity is invalid")
    if payload.get("prior_contract") != {
        "kind": "truncated_normal",
        "mean": bogt.PRIOR_MEAN,
        "variance": bogt.PRIOR_VARIANCE,
        "lower": 0.0,
        "unconditional": True,
    }:
        raise ValueError("registered v2 prior contract is not the unchanged contract")
    raw_statistics = payload.get("raw_scale_statistics")
    if not isinstance(raw_statistics, Mapping) or raw_statistics.get("count") != PORT_COUNT or raw_statistics.get("finite") is not True or raw_statistics.get("positive") is not True:
        raise ValueError("registered v2 raw-scale statistics are invalid")
    raw = payload.get("raw_port_mean")
    registered = payload.get("port_scale")
    if not torch.is_tensor(raw) or not torch.is_tensor(registered):
        raise ValueError("registered v2 raw and registered scales must be tensors")
    if payload.get("raw_scale_dtype") != str(raw.dtype) or payload.get("registered_scale_dtype") != str(registered.dtype):
        raise ValueError("registered v2 scale dtype binding is stale")
    calibration = bogt.build_unit_aware_calibration(
        raw,
        payload["plane_reference"],
        sample_count=int(payload["sample_count"]),
        plane_lower=payload["plane_lower"],
        plane_upper=payload["plane_upper"],
    )
    if not torch.equal(calibration.port_scale, registered):
        raise ValueError("registered v2 port scale is not the raw mean one-to-one")
    return calibration


def load_registered_unit_aware_calibration(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_dataset_manifest_hash: str,
    expected_calibration_indices_sha256: str,
    expected_mode_bank_sha256: str,
    expected_parent_v1_sha256: str,
) -> bogt.UnitAwareCalibration:
    """Load only an explicitly hash-bound v2 calibration; never fall back to v1."""
    payload = safe_torch_load(_resolve(path), expected_sha256=expected_sha256, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("registered v2 calibration must be a mapping")
    return _validate_registered_payload(
        payload,
        expected_dataset_manifest_hash=expected_dataset_manifest_hash,
        expected_calibration_indices_sha256=expected_calibration_indices_sha256,
        expected_mode_bank_sha256=expected_mode_bank_sha256,
        expected_parent_v1_sha256=expected_parent_v1_sha256,
    )


def _scale_comparison(old_scale: torch.Tensor, new_scale: torch.Tensor) -> dict[str, Any]:
    old = old_scale.detach().to(torch.float64).cpu()
    new = new_scale.detach().to(torch.float64).cpu()
    if old.shape != (PORT_COUNT,) or new.shape != (PORT_COUNT,):
        raise ValueError("scale comparison requires [100] values")
    ratio = old / new
    return {
        "old_schema": PARENT_V1_SCHEMA,
        "new_schema": R2_CALIBRATION_SCHEMA,
        "old_scale": old.tolist(),
        "new_scale": new.tolist(),
        "old_scale_over_new_scale": ratio.tolist(),
        "old_scale_unique_count": int(torch.unique(old).numel()),
        "new_scale_unique_count": int(torch.unique(new).numel()),
        "old_scale_median": float(old.median()),
        "new_scale_median": float(new.median()),
        "new_scale_equals_raw_mean": True,
    }


def _evaluate_gate(
    raw_means: torch.Tensor,
    calibration: bogt.UnitAwareCalibration,
    sanity: Mapping[str, Any],
    *,
    old_sha_before: str,
    old_sha_after: str,
    expected_old_sha: str,
    new_artifact_sha256: str | None,
    freeze: Mapping[str, Any],
) -> dict[str, bool]:
    eps = torch.finfo(torch.float32).eps
    raw = raw_means.detach().cpu()
    forbidden = (
        "optimizer_created",
        "backward_called",
        "checkpoint_created",
        "refit",
        "phase_updated",
        "training",
        "prior_generation",
        "decoder_entered",
        "cmos_called",
        "mpcdar_called",
    )
    gates = {
        "A_raw_means_finite": bool(torch.isfinite(raw).all()),
        "B_raw_means_positive": bool((raw > 0).all()),
        "C_registered_equals_raw": bool(torch.equal(calibration.raw_port_mean, calibration.port_scale)),
        "D_not_all_float32_eps": not bool(torch.all(raw == eps)),
        "E_variation_preserved": int(torch.unique(raw).numel()) > 1 and float(raw.max() - raw.min()) > 0.0,
        "F_global_z_mean_near_one": abs(float(sanity["global_mean"]) - 1.0) <= MEAN_TOLERANCE,
        "G_per_port_z_mean_near_one": max(abs(float(value) - 1.0) for value in sanity["per_port_mean"]) <= MEAN_TOLERANCE,
        "H_z_finite": bool(sanity["finite"]),
        "I_parent_v1_unchanged": old_sha_before == expected_old_sha and old_sha_after == expected_old_sha,
        "J_new_artifact_differs_from_v1": new_artifact_sha256 is not None and new_artifact_sha256 != expected_old_sha,
        "K_source_state_digest_unchanged": freeze.get("state_unchanged") is True and freeze.get("state_digest_before") == freeze.get("state_digest_after"),
        "L_no_optimizer": all(freeze.get(name) is False for name in forbidden if name in freeze),
        "M_no_backward_or_training": all(freeze.get(name) is False for name in ("backward_called", "training")),
        "N_no_checkpoint_or_refit": all(freeze.get(name) is False for name in ("checkpoint_created", "refit")),
        "O_no_downstream_consumer": all(freeze.get(name) is False for name in ("prior_generation", "decoder_entered", "cmos_called", "mpcdar_called")),
    }
    return gates


def _artifact_refs(output_path: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for target in sorted(output_path.iterdir(), key=lambda item: item.name):
        if target.is_file() and target.name != "calibration_manifest.json":
            refs[target.name] = {"path": target.name, "sha256": file_sha256(target), "size_bytes": target.stat().st_size}
    return refs


def _assert_output_tree(output_path: Path) -> None:
    expected = {
        "repaired_calibration.pt",
        "calibration_manifest.json",
        "calibration_metrics.json",
        "old_new_calibration_comparison.json",
        "calibration_scale_summary.png",
        "compact_report.json",
    }
    actual = {target.relative_to(output_path).as_posix() for target in output_path.rglob("*") if target.is_file()}
    if actual != expected:
        raise RuntimeError(f"R2 output tree is not exact: {sorted(actual)}")
    if any(".tmp" in name or Path(name).name.startswith(".") for name in actual):
        raise RuntimeError("R2 output contains a temporary or hidden file")


def _plot_scale_summary(output_path: Path, raw_scale: torch.Tensor, old_scale: torch.Tensor) -> None:
    raw = raw_scale.detach().to(torch.float64).cpu().numpy()
    old = old_scale.detach().to(torch.float64).cpu().numpy()
    eps = float(torch.finfo(torch.float32).eps)
    tiny = float(torch.finfo(torch.float32).tiny)
    figure, axis = plt.subplots(figsize=(6.0, 4.0), dpi=300)
    try:
        ports = list(range(1, PORT_COUNT + 1))
        axis.plot(ports, raw, color="#1f77b4", linewidth=1.2, label="raw mean = registered")
        axis.plot(ports, old, color="#666666", linestyle="--", linewidth=1.0, label="v1 registered scale")
        axis.axhline(eps, color="#d62728", linestyle=":", linewidth=1.0, label="float32 eps")
        axis.axhline(tiny, color="#9467bd", linestyle="-.", linewidth=1.0, label="float32 tiny")
        axis.set_yscale("log")
        axis.set_ylim(tiny / 10.0, max(float(old.max()) * 3.0, eps * 10.0))
        axis.set_xlabel("OLS port")
        axis.set_ylabel("calibration scale (modal power)")
        axis.set_title("Unit-aware Scaled-Chen calibration")
        axis.grid(True, which="both", alpha=0.22, linewidth=0.5)
        axis.legend(fontsize=7, loc="best", frameon=False)
        figure.tight_layout()
        figure.savefig(output_path / "calibration_scale_summary.png", format="png", dpi=300)
    finally:
        plt.close(figure)


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_new_json(path, payload, artifact_kind="R2 calibration artifact")


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    config_target, config, config_sha256, config_canonical_sha256 = _load_config(config_path)
    state = base._git_state()
    if state.get("branch") != BRANCH:
        raise ValueError(f"R2 repair requires branch {BRANCH}")
    source = _load_c6_source(config)
    parent = _load_parent_v1(config)
    old_calibration = parent["calibration"]
    raw = source["raw_payload"]
    calibration = bogt.build_unit_aware_calibration(
        raw["mean_fp32"],
        old_calibration.plane_reference,
        sample_count=CALIBRATION_COUNT,
        plane_lower=old_calibration.plane_lower,
        plane_upper=old_calibration.plane_upper,
    )
    sanity = calibration_z_sanity(raw["raw_powers"], calibration.port_scale)
    old_sha_before = parent["sha256_before"]
    old_sha_after_pre = file_sha256(parent["path"])
    freeze = {
        **source["freeze"],
        "phase_updated": False,
        "training": False,
        "prior_generation": False,
    }
    pre_gates = _evaluate_gate(
        raw["mean_fp32"],
        calibration,
        sanity,
        old_sha_before=old_sha_before,
        old_sha_after=old_sha_after_pre,
        expected_old_sha=str(config["parent_v1"]["sha256"]),
        new_artifact_sha256=None,
        freeze=freeze,
    )
    if not all(value for name, value in pre_gates.items() if name != "J_new_artifact_differs_from_v1"):
        failed = [name for name, value in pre_gates.items() if not value and name != "J_new_artifact_differs_from_v1"]
        raise RuntimeError(f"{R2_INVALID}: {', '.join(failed)}")

    producer_git = {
        "branch": BRANCH,
        "measurement_commit": source["manifest"]["lineage"]["measurement_head"],
        "measurement_starting_head": source["manifest"]["lineage"]["starting_head"],
        "repair_generation_commit": state["commit"],
        "repair_generation_clean": bool(state["clean"]),
    }
    metadata = {
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "dataset_manifest_hash": config["lineage"]["dataset_manifest_sha256"],
        "calibration_indices_sha256": config["lineage"]["calibration_indices_sha256"],
        "zero_phase_reference": ZERO_PHASE_REFERENCE,
        "mode_bank_sha256": config["lineage"]["mode_bank_sha256"],
        "plane_identity": PLANE_IDENTITY,
        "raw_scale_dtype": str(calibration.raw_port_mean.dtype),
        "registered_scale_dtype": str(calibration.port_scale.dtype),
        "raw_scale_statistics": _scale_statistics(raw["mean_fp32"]),
        "raw_scale_statistics_fp64": _scale_statistics(raw["mean_fp64"]),
        "floor_policy": FLOOR_POLICY,
        "registered_scale_source": "C6 mean_fp32 direct raw port-power mean; no absolute or relative floor",
        "parent_v1_artifact_path": _relative_path(parent["path"]),
        "parent_v1_artifact_sha256": old_sha_before,
        "parent_v1_schema": PARENT_V1_SCHEMA,
        "source_c6_manifest_path": _relative_path(source["manifest_path"]),
        "source_c6_manifest_sha256": source["manifest_sha256"],
        "source_c6_payload_path": _relative_path(source["raw_path"]),
        "source_c6_payload_sha256": source["raw_sha256"],
        "source_c6_metrics_path": _relative_path(source["metrics_path"]),
        "source_c6_metrics_sha256": file_sha256(source["metrics_path"]),
        "producer_git": producer_git,
        "prior_contract": config["prior_contract"],
        "optical_forward_dtype": OPTICAL_FORWARD_DTYPE,
        "learned_state_freeze_inherited_from_c6": freeze,
        "source_c6_gate": C6_GATE,
    }
    payload = build_repaired_calibration_payload(calibration, metadata=metadata | {"contract_version": R2_CONTRACT_VERSION})
    requested_output = _resolve(output_path)
    actual_output = next_write_once_output(requested_output)
    actual_output.mkdir(parents=True, exist_ok=False)
    atomic_torch_save_new(actual_output / "repaired_calibration.pt", payload, artifact_kind="R2 repaired calibration")
    new_calibration_sha256 = file_sha256(actual_output / "repaired_calibration.pt")
    old_sha_after = file_sha256(parent["path"])
    gates = _evaluate_gate(
        raw["mean_fp32"],
        calibration,
        sanity,
        old_sha_before=old_sha_before,
        old_sha_after=old_sha_after,
        expected_old_sha=str(config["parent_v1"]["sha256"]),
        new_artifact_sha256=new_calibration_sha256,
        freeze=freeze,
    )
    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"{R2_INVALID}: {', '.join(failed)}")

    _plot_scale_summary(actual_output, calibration.raw_port_mean, old_calibration.port_scale)
    comparison = {
        "schema": "lightgen_scaled_chen_p1_calibration_old_new_comparison.v1",
        "task_id": TASK_ID,
        "gate": R2_GATE,
        "old_artifact": {
            "path": _relative_path(parent["path"]),
            "schema": PARENT_V1_SCHEMA,
            "sha256": old_sha_before,
        },
        "new_artifact": {
            "path": "repaired_calibration.pt",
            "schema": R2_CALIBRATION_SCHEMA,
            "sha256": new_calibration_sha256,
        },
        "comparison": _scale_comparison(old_calibration.port_scale, calibration.port_scale),
    }
    _write_json_new(actual_output / "old_new_calibration_comparison.json", comparison)
    compact_report = {
        "schema": "lightgen_scaled_chen_p1_calibration_contract_repair_report.v1",
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "result": R2_GATE,
        "interpretation": "unit_aware_calibration_contract_repaired_without_downstream_replay",
        "raw_mean_min": metadata["raw_scale_statistics"]["min"],
        "raw_mean_median": metadata["raw_scale_statistics"]["median"],
        "raw_mean_mean": metadata["raw_scale_statistics"]["mean"],
        "raw_mean_max": metadata["raw_scale_statistics"]["max"],
        "registered_scale_equals_raw_mean": True,
        "old_v1_artifact_unchanged": old_sha_before == old_sha_after,
        "source_c6_gate": C6_GATE,
        "governance": config["governance"],
        "next_action": NEXT_ACTION,
    }
    _write_json_new(actual_output / "compact_report.json", compact_report)
    metrics = {
        "schema": "lightgen_scaled_chen_p1_calibration_contract_repair_metrics.v1",
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "gate": R2_GATE,
        "gate_checks": gates,
        "raw_scale_statistics_fp32": metadata["raw_scale_statistics"],
        "raw_scale_statistics_fp64": metadata["raw_scale_statistics_fp64"],
        "registered_scale": {
            "schema": R2_CALIBRATION_SCHEMA,
            "dtype": str(calibration.port_scale.dtype),
            "raw_registered_exact_equal": bool(torch.equal(calibration.raw_port_mean, calibration.port_scale)),
            "unique_count": int(torch.unique(calibration.port_scale).numel()),
            "minimum": float(calibration.port_scale.min()),
            "maximum": float(calibration.port_scale.max()),
        },
        "z_sanity": sanity,
        "old_new": {
            "old_sha256": old_sha_before,
            "new_sha256": new_calibration_sha256,
            "old_sha256_unchanged": old_sha_before == old_sha_after,
            "new_sha256_differs": new_calibration_sha256 != old_sha_before,
            "old_scale_unique_count": int(torch.unique(old_calibration.port_scale).numel()),
            "new_scale_unique_count": int(torch.unique(calibration.port_scale).numel()),
        },
        "lineage": {
            "dataset_manifest_sha256": config["lineage"]["dataset_manifest_sha256"],
            "calibration_indices_sha256": config["lineage"]["calibration_indices_sha256"],
            "mode_bank_sha256": config["lineage"]["mode_bank_sha256"],
            "zero_phase_reference": ZERO_PHASE_REFERENCE,
            "parent_v1_artifact_sha256": old_sha_before,
            "source_c6_manifest_sha256": source["manifest_sha256"],
            "source_c6_payload_sha256": source["raw_sha256"],
            "config_sha256": config_sha256,
            "config_canonical_sha256": config_canonical_sha256,
        },
        "learned_state_freeze": freeze,
        "calibration_only": {
            "cuda_replay": False,
            "downstream_replay": False,
            "optimizer_created": False,
            "backward_called": False,
            "checkpoint_created": False,
            "refit": False,
            "phase_updated": False,
            "training": False,
            "prior_generation": False,
            "decoder_entered": False,
            "cmos_called": False,
            "mpcdar_called": False,
        },
        "artifact_hashes_before_manifest": _artifact_refs(actual_output),
        "governance": config["governance"],
        "next_action": NEXT_ACTION,
    }
    _write_json_new(actual_output / "calibration_metrics.json", metrics)
    artifact_refs = _artifact_refs(actual_output)
    manifest = {
        "schema": "lightgen_scaled_chen_p1_calibration_contract_repair_manifest.v1",
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "gate": R2_GATE,
        "attempt": actual_output.name,
        "output_directory": _relative_path(actual_output),
        "artifact_layout": {
            "paths_are_relative_to_output": True,
            "manifest_self_hash_excluded": True,
            "files": sorted(artifact_refs),
        },
        "config": {
            "path": _relative_path(config_target),
            "sha256": config_sha256,
            "canonical_sha256": config_canonical_sha256,
        },
        "source_c6": {
            "directory": _relative_path(source["directory"]),
            "manifest": _relative_path(source["manifest_path"]),
            "manifest_sha256": source["manifest_sha256"],
            "raw_port_means": _relative_path(source["raw_path"]),
            "raw_port_means_sha256": source["raw_sha256"],
            "metrics": _relative_path(source["metrics_path"]),
            "metrics_sha256": file_sha256(source["metrics_path"]),
            "gate": C6_GATE,
            "immutable": True,
        },
        "parent_v1": {
            "path": _relative_path(parent["path"]),
            "schema": PARENT_V1_SCHEMA,
            "sha256": old_sha_before,
            "sha256_unchanged_after_repair": old_sha_before == old_sha_after,
        },
        "calibration_contract": {
            "schema": R2_CALIBRATION_SCHEMA,
            "contract_version": R2_CONTRACT_VERSION,
            "raw_scale_field": "raw_port_mean",
            "registered_scale_field": "port_scale",
            "registered_scale_equals_raw": True,
            "floor_policy": FLOOR_POLICY,
            "zero_phase_reference": ZERO_PHASE_REFERENCE,
            "plane_identity": PLANE_IDENTITY,
            "optical_forward_dtype": OPTICAL_FORWARD_DTYPE,
            "prior_contract": config["prior_contract"],
        },
        "lineage": {
            "dataset_manifest_sha256": config["lineage"]["dataset_manifest_sha256"],
            "calibration_indices_sha256": config["lineage"]["calibration_indices_sha256"],
            "mode_bank_sha256": config["lineage"]["mode_bank_sha256"],
            "measurement_head": source["manifest"]["lineage"]["measurement_head"],
            "repair_generation_head": state["commit"],
        },
        "producer_git": producer_git,
        "learned_state_freeze": freeze,
        "optical_data_changed": False,
        "cuda_replay": False,
        "downstream_replay": False,
        "artifacts": artifact_refs,
        "storage_bytes_excluding_manifest": sum(ref["size_bytes"] for ref in artifact_refs.values()),
        "governance": config["governance"],
        "next_action": NEXT_ACTION,
    }
    _write_json_new(actual_output / "calibration_manifest.json", manifest)
    _assert_output_tree(actual_output)
    return {
        "output_directory": str(actual_output),
        "gate": R2_GATE,
        "report": compact_report,
        "metrics": metrics,
        "manifest": manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args.config, args.output)
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
