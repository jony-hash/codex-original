"""Frozen G1 support-accessibility observer for the registered scaled-Chen route.

The observer measures the exact normalized nearest-support distances of the
registered step-5000 encoded support against an equal-sized unconditional
100D truncated-normal prior.  It owns no optimizer, backward path, decoder,
CMOS path, evaluator, checkpoint writer, or generation path.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.application.research import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research import optical_manifold_smf_bogt_manifold_readiness as mra
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)


TASK_ID = "H-LOMA-GEN-G1-SA"
CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_prior_support_accessibility_v1"
OBSERVER_SCHEMA = "lightgen_scaled_chen_g1_prior_support_accessibility_observer_manifest.v1"
METRICS_SCHEMA = "lightgen_scaled_chen_g1_prior_support_accessibility_metrics.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_g1_prior_support_accessibility_compact_report.v1"
DISTANCES_SCHEMA = "lightgen_scaled_chen_g1_prior_support_accessibility_distances.v1"
ALIGNMENT_SCHEMA = "lightgen_scaled_chen_g1_support_figure_alignment.v1"
HYPOTHESIS = "H-LOMA-GEN-G1-SA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
LATENT_DIM = 100
PRIOR_SEED = 1237
BOOTSTRAP_SEED = 1236
BOOTSTRAP_REPLICATES = 2000
CONFIDENCE_LEVEL = 0.95
QUERY_CHUNK_SIZE = 128
SUPPORT_CHUNK_SIZE = 4096
BATCH_SIZE = 10
EXPECTED_MRA_Z_MEAN = 0.0014779161429032683
MAX_STORAGE_BYTES = 512 * 1024**2

DEFAULT_CONFIG_PATH = Path(
    "configs/research/lightgen_optical_manifold_smf_bogt_prior_support_accessibility_v1.yaml"
)
DEFAULT_REVIEW_PATH = Path(
    "configs/research/lightgen_optical_manifold_smf_bogt_manifold_readiness_review_v1.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260903_v2/runs/"
    "lightgen_scaled_chen_g1_prior_support_accessibility_v1_attempt_01"
)


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _json_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"registered JSON artifact is missing or stale: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"registered JSON artifact is not a mapping: {path}")
    return payload


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load_config(path: str | Path) -> dict[str, Any]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "parent_mra",
        "lineage",
        "dataset",
        "prior_sampling",
        "distance",
        "bootstrap",
        "gate",
        "execution",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("G1 support-accessibility config fields are invalid")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("G1 support-accessibility config schema or hypothesis is invalid")
    _assert_config(payload)
    return payload


def _assert_config(payload: Mapping[str, Any]) -> None:
    parent = _require_mapping(payload["parent_mra"], "parent_mra")
    expected_parent = {
        "config_sha256": "95e4587f87fc49e79fe3bef5a930f34502ebc9c95f36a312995d77f993a72cad",
        "config_canonical_sha256": "1a69f77b3e9163b8a756dcaa8c6af039828e90f2b607182b45975f81387f61d9",
        "review_contract_sha256": "3d0c807b0fadf0bb7f6ef0596c5a3279913175b82b792fa6b1015a086b207ac8",
        "audit_sha256": "6291a2dc03f5ec8d6d3cd9ba88c0bf7f3a4b3f44f83dba7fc16c80375e9dc556",
        "compact_report_sha256": "401fe2159f4342a4740b2c40955cb65fc7bb3600f2416fdc4171e1553eebeee0",
        "panel_sha256": "842f5cfd41909a666f19f3c88e7e7199a825b255bd111a44953e53351fd28724",
    }
    for key, expected in expected_parent.items():
        if parent.get(key) != expected:
            raise ValueError(f"G1 parent MRA binding is not locked: {key}")

    lineage = _require_mapping(payload["lineage"], "lineage")
    checkpoint = _require_mapping(lineage["checkpoint"], "lineage.checkpoint")
    calibration = _require_mapping(lineage["calibration"], "lineage.calibration")
    if checkpoint.get("sha256") != mra.SOURCE_HASHES["scaled_chen_step_5000"] or checkpoint.get("global_step") != 5000 or checkpoint.get("schema") != "scaled_chen_bogt_horizon_checkpoint.v1":
        raise ValueError("G1 step-5000 checkpoint lineage is not locked")
    if calibration.get("sha256") != mra.SOURCE_HASHES["scaled_chen_calibration"] or calibration.get("sample_count") != 1000 or calibration.get("schema") != "scaled_chen_bogt_calibration.v1":
        raise ValueError("G1 calibration lineage is not locked")
    latent = _require_mapping(lineage["latent"], "lineage.latent")
    if latent != {
        "definition": "z_j=abs(c_j)^2/s_j",
        "dimension": LATENT_DIM,
        "normalization": "sqrt_100_only_for_distance",
        "scale_source": "registered_calibration_port_scale",
    }:
        raise ValueError("G1 latent contract is not locked")

    dataset = _require_mapping(payload["dataset"], "dataset")
    if dataset != {
        "dataset_root": "data",
        "train_support_source": "registered_train_split_0_55000_full",
        "train_support_indices": [0, 55000],
        "train_support_count": 55000,
        "heldout_source": "horizon_heldout_train_indices_55000_59999",
        "heldout_indices": [55000, 60000],
        "heldout_count": 5000,
        "expected_manifest_sha256": "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb",
        "expected_heldout_indices_sha256": "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e7e5e2218a",
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
        "calibration_and_schedule_context_only": True,
    }:
        raise ValueError("G1 dataset support contract is not locked")

    prior = _require_mapping(payload["prior_sampling"], "prior_sampling")
    if prior != {
        "seed": PRIOR_SEED,
        "kind": "truncated_normal",
        "mean": 1.0,
        "variance": 0.25,
        "lower": 0.0,
        "unconditional": True,
        "class_conditioned": False,
        "count_equals_heldout": True,
        "sampler": "bogt.truncated_normal_sample_inverse_cdf",
    }:
        raise ValueError("G1 prior sampling contract is not locked")

    if _require_mapping(payload["distance"], "distance") != {
        "metric": "exact_normalized_euclidean",
        "normalization": "divide_by_sqrt_100",
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "support_chunk_size": SUPPORT_CHUNK_SIZE,
        "approximate_search": False,
    }:
        raise ValueError("G1 distance contract is not locked")
    if _require_mapping(payload["bootstrap"], "bootstrap") != {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": CONFIDENCE_LEVEL,
        "resampling_unit": "one_dimensional_distance_vectors",
    }:
        raise ValueError("G1 bootstrap contract is not locked")
    if _require_mapping(payload["gate"], "gate") != {
        "name": "P4",
        "supported": "lower_ci_r50_gt_1_and_lower_ci_r95_gt_1",
        "not_supported": "upper_ci_r50_le_1_and_upper_ci_r95_le_1",
        "otherwise": "P4_UNDERDETERMINED",
        "prior_exceedance_h95": "descriptive_only",
    }:
        raise ValueError("G1 P4 gate contract is not locked")
    if _require_mapping(payload["execution"], "execution") != {
        "device": "cuda",
        "batch_size": BATCH_SIZE,
        "no_update": True,
        "decoder_entered": False,
        "prior_decoded": False,
        "evaluator_calls": 0,
    }:
        raise ValueError("G1 execution contract is not locked")
    if _require_mapping(payload["authorization"], "authorization") != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_promotion": False,
        "prohibited": [
            "optimizer",
            "backward",
            "refit",
            "checkpoint_write",
            "prior_decode",
            "decoder",
            "relay_after_pre_ols",
            "cmos",
            "mpcdar",
            "class_conditioning",
            "metric_tuning",
            "subset_tuning",
        ],
    }:
        raise ValueError("G1 authorization boundary is not locked")


def validate_latent_bank(value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[1] != LATENT_DIM:
        raise ValueError("latent bank must have exact 100 dimensions")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise ValueError("latent bank must be real floating-point data")
    if not torch.isfinite(value).all():
        raise ValueError("latent bank must be finite")
    if (value < 0).any():
        raise ValueError("latent bank must be nonnegative")


def sample_registered_prior(
    count: int,
    *,
    seed: int = PRIOR_SEED,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("prior count must be positive")
    if device != "cpu" and str(device) != "cpu":
        raise ValueError("registered prior sampling must use the CPU generator")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = bogt.truncated_normal_sample(
        (count, LATENT_DIM),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    validate_latent_bank(result)
    return result


def exact_normalized_nearest_distances(
    query: torch.Tensor,
    support: torch.Tensor,
    *,
    query_chunk_size: int = QUERY_CHUNK_SIZE,
    support_chunk_size: int = SUPPORT_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute exact nearest distances without materializing query-by-support."""
    validate_latent_bank(query)
    validate_latent_bank(support)
    if query.device != support.device:
        raise ValueError("query and support must be on the same device")
    if query.shape[1] != support.shape[1] or query.shape[0] == 0 or support.shape[0] == 0:
        raise ValueError("query and support must be nonempty compatible banks")
    if query_chunk_size <= 0 or support_chunk_size <= 0:
        raise ValueError("distance chunk sizes must be positive")

    query64 = query.detach().to(dtype=torch.float64)
    support64 = support.detach().to(dtype=torch.float64)
    support_norm = support64.square().sum(dim=1)
    result = torch.empty(query64.shape[0], dtype=torch.float64, device=query.device)
    for query_start in range(0, query64.shape[0], query_chunk_size):
        query_chunk = query64[query_start : query_start + query_chunk_size]
        query_norm = query_chunk.square().sum(dim=1)
        best = torch.full(
            (query_chunk.shape[0],),
            float("inf"),
            dtype=torch.float64,
            device=query.device,
        )
        for support_start in range(0, support64.shape[0], support_chunk_size):
            support_chunk = support64[support_start : support_start + support_chunk_size]
            squared = (
                query_norm[:, None]
                + support_norm[support_start : support_start + support_chunk_size][None, :]
                - 2.0 * (query_chunk @ support_chunk.transpose(0, 1))
            )
            best = torch.minimum(best, squared.min(dim=1).values)
        result[query_start : query_start + query_chunk.shape[0]] = (
            best.clamp_min(0.0).sqrt() / math.sqrt(LATENT_DIM)
        )
    return result


def _quantile(value: torch.Tensor, probability: float, *, dim: int | None = None) -> torch.Tensor:
    return torch.quantile(value, probability, dim=dim, interpolation="linear")


def bootstrap_ratio_ci(
    holdout_distances: torch.Tensor,
    prior_distances: torch.Tensor,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    if holdout_distances.ndim != 1 or prior_distances.ndim != 1:
        raise ValueError("bootstrap distances must be one-dimensional")
    if holdout_distances.numel() != prior_distances.numel() or holdout_distances.numel() == 0:
        raise ValueError("bootstrap distance vectors must have equal nonzero length")
    if not torch.is_floating_point(holdout_distances) or not torch.is_floating_point(prior_distances):
        raise ValueError("bootstrap distances must be floating-point")
    if not torch.isfinite(holdout_distances).all() or not torch.isfinite(prior_distances).all() or (holdout_distances <= 0).any() or (prior_distances <= 0).any():
        raise ValueError("bootstrap distances must be finite and positive")
    if replicates < 2000 or not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap contract requires at least 2000 replicates and a valid confidence level")

    holdout = holdout_distances.detach().cpu().to(torch.float64)
    prior = prior_distances.detach().cpu().to(torch.float64)
    n = holdout.numel()
    point_h50 = _quantile(holdout, 0.5)
    point_h95 = _quantile(holdout, 0.95)
    point_p50 = _quantile(prior, 0.5)
    point_p95 = _quantile(prior, 0.95)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ratio50_chunks: list[torch.Tensor] = []
    ratio95_chunks: list[torch.Tensor] = []
    alpha = (1.0 - confidence_level) / 2.0
    for start in range(0, replicates, 64):
        count = min(64, replicates - start)
        holdout_index = torch.randint(n, (count, n), generator=generator)
        prior_index = torch.randint(n, (count, n), generator=generator)
        holdout_sample = holdout.index_select(0, holdout_index.reshape(-1)).reshape(count, n)
        prior_sample = prior.index_select(0, prior_index.reshape(-1)).reshape(count, n)
        holdout50 = _quantile(holdout_sample, 0.5, dim=1)
        holdout95 = _quantile(holdout_sample, 0.95, dim=1)
        prior50 = _quantile(prior_sample, 0.5, dim=1)
        prior95 = _quantile(prior_sample, 0.95, dim=1)
        ratio50_chunks.append(prior50 / holdout50)
        ratio95_chunks.append(prior95 / holdout95)
    ratio50 = torch.cat(ratio50_chunks)
    ratio95 = torch.cat(ratio95_chunks)
    return {
        "r50": float(point_p50 / point_h50),
        "r95": float(point_p95 / point_h95),
        "r50_ci": [
            float(_quantile(ratio50, alpha)),
            float(_quantile(ratio50, 1.0 - alpha)),
        ],
        "r95_ci": [
            float(_quantile(ratio95, alpha)),
            float(_quantile(ratio95, 1.0 - alpha)),
        ],
        "seed": seed,
        "replicates": replicates,
        "confidence_level": confidence_level,
        "resampling_unit": "one_dimensional_distance_vectors",
    }


def classify_p4_gate(r50_ci: Sequence[float], r95_ci: Sequence[float]) -> str:
    if len(r50_ci) != 2 or len(r95_ci) != 2:
        raise ValueError("P4 confidence intervals must be [lower, upper]")
    r50_lower, r50_upper = float(r50_ci[0]), float(r50_ci[1])
    r95_lower, r95_upper = float(r95_ci[0]), float(r95_ci[1])
    if r50_lower > 1.0 and r95_lower > 1.0:
        return "P4_SUPPORTED"
    if r50_upper <= 1.0 and r95_upper <= 1.0:
        return "P4_NOT_SUPPORTED"
    return "P4_UNDERDETERMINED"


def next_write_once_output(base_path: str | Path) -> Path:
    base_target = Path(base_path)
    match = re.fullmatch(r"(.+)_attempt_(\d+)", base_target.name)
    if match is None:
        if not base_target.exists():
            return base_target
        prefix, number = base_target.name, 1
    else:
        prefix, number = match.group(1), int(match.group(2))
        if not base_target.exists():
            return base_target
    while True:
        number += 1
        candidate = base_target.parent / f"{prefix}_attempt_{number:02d}"
        if not candidate.exists():
            return candidate


def _state_digest(module: torch.nn.Module) -> str:
    return canonical_sha256(
        {name: tensor_sha256(value) for name, value in module.state_dict().items()}
    )


def _load_registered_checkpoint(
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(lineage["checkpoint_path"])
    payload = safe_torch_load(path, expected_sha256=lineage["checkpoint_sha256"], map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != "scaled_chen_bogt_horizon_checkpoint.v1" or int(payload.get("global_step", -1)) != 5000 or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("registered step-5000 checkpoint schema or step is invalid")
    return dict(payload)


def _load_registered_calibration(
    lineage: Mapping[str, Any],
) -> bogt.Calibration:
    path = Path(lineage["calibration_path"])
    payload = safe_torch_load(path, expected_sha256=lineage["calibration_sha256"], map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != "scaled_chen_bogt_calibration.v1":
        raise ValueError("registered calibration schema is invalid")
    calibration = bogt.Calibration(
        payload["port_scale"],
        payload["plane_reference"],
        int(payload["sample_count"]),
        payload.get("plane_lower"),
        payload.get("plane_upper"),
    )
    calibration.validate()
    return calibration


def _encoded_power_latent_prefix(
    model: bogt.ScaledChenBOGTModel,
    amplitude: torch.Tensor,
    calibration: bogt.Calibration,
) -> torch.Tensor:
    """Replay only SLM -> front -> encoder -> normalized pre-OLS power."""
    if amplitude.ndim != 4 or tuple(amplitude.shape[1:]) != (1, 200, 200) or torch.is_complex(amplitude) or not torch.isfinite(amplitude).all() or amplitude.min() < 0.0 or amplitude.max() > 1.0:
        raise ValueError("encoded prefix requires finite [N,1,200,200] amplitude in [0,1]")
    slm = model.system.slm(model.system.command(amplitude))
    front = model.system.front(slm)
    encoded, encoder_records_raw, _ = model.system.encoder(front, "encoder")
    if len(encoder_records_raw) != 2:
        raise ValueError("registered encoder prefix must contain two layers")
    encoded = bogt._attenuate(encoded, bogt.FIELD_TRANSMISSION**2, "fixed_encoder_group_transmission")
    coefficients = model._coefficients(encoded)
    scale = calibration.port_scale.to(device=coefficients.device, dtype=coefficients.real.dtype)
    z = bogt.normalize_port_power(coefficients.abs().square(), scale)
    validate_latent_bank(z)
    return z


def _collect_latent_bank(
    model: bogt.ScaledChenBOGTModel,
    images: torch.Tensor,
    indices: torch.Tensor,
    calibration: bogt.Calibration,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), batch_size):
            batch_indices = indices[start : start + batch_size]
            amplitude = base.render_intensity_batch(
                images.index_select(0, batch_indices)
            ).amplitude.to(device)
            z = _encoded_power_latent_prefix(model, amplitude, calibration)
            chunks.append(z.detach().cpu())
            del amplitude, z
    result = torch.cat(chunks, dim=0)
    validate_latent_bank(result)
    return result


def _summary_stats(value: torch.Tensor) -> dict[str, float]:
    return {
        "count": int(value.numel()),
        "mean": float(value.mean()),
        "median": float(_quantile(value, 0.5)),
        "p95": float(_quantile(value, 0.95)),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
    }


def _load_parent_panel(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    payload = safe_torch_load(path, expected_sha256=expected_sha256, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != mra.PANEL_SCHEMA:
        raise ValueError("parent MRA panel schema is invalid")
    required = {"schema", "selection_indices", "selection_labels", "confirmation_indices", "confirmation_labels", "per_sample_mse", "source_hashes", "affects_selection"}
    if set(payload) != required:
        raise ValueError("parent MRA panel fields are invalid")
    return payload


def validate_registered_lineage(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    review_path: str | Path = DEFAULT_REVIEW_PATH,
) -> dict[str, Any]:
    """Validate attempt_05 and return trusted internal lineage objects."""
    config_target = _resolve(config_path)
    config = _load_config(config_target)
    parent = config["parent_mra"]
    mra_config_path = _resolve(parent["config"])
    if file_sha256(mra_config_path) != parent["config_sha256"]:
        raise ValueError("parent MRA config hash does not match")
    mra_spec = mra._load_spec(mra_config_path)
    if mra_spec.config_hash != parent["config_canonical_sha256"]:
        raise ValueError("parent MRA canonical config hash does not match")
    mra._assert_contract(mra_spec)
    review_target = _resolve(review_path)
    if file_sha256(review_target) != parent["review_contract_sha256"]:
        raise ValueError("parent MRA review hash does not match")
    mra._load_review(review_target, mra_spec)
    formal, formal_path = mra._load_formal(mra_spec)

    source = mra._load_source(mra_spec, load_checkpoints=False)
    lineage = config["lineage"]
    checkpoint_path = _resolve(lineage["checkpoint"]["path"])
    calibration_path = _resolve(lineage["calibration"]["path"])
    if checkpoint_path != source.paths["scaled_chen_step_5000"] or calibration_path != source.paths["scaled_chen_calibration"]:
        raise ValueError("G1 lineage paths do not match MRA source paths")
    if source.hashes["scaled_chen_step_5000"] != lineage["checkpoint"]["sha256"] or source.hashes["scaled_chen_calibration"] != lineage["calibration"]["sha256"]:
        raise ValueError("G1 lineage hashes do not match MRA source hashes")

    audit_path = _resolve(parent["audit"])
    compact_path = _resolve(parent["compact_report"])
    panel_path = _resolve(parent["panel"])
    audit = _json_file(audit_path, parent["audit_sha256"])
    compact = _json_file(compact_path, parent["compact_report_sha256"])
    panel = _load_parent_panel(panel_path, parent["panel_sha256"])
    if audit.get("schema") != mra.AUDIT_SCHEMA or audit.get("hypothesis") != mra.HYPOTHESIS or audit.get("status") != "valid_failed":
        raise ValueError("parent MRA attempt_05 status or schema is invalid")
    if audit.get("generative_manifold_supported") is not False or audit.get("generation_readiness") != "not_ready" or audit.get("training_authorized") is not False or audit.get("formal_promotion") is not False:
        raise ValueError("parent MRA governance boundary is invalid")
    if compact.get("schema") != "scaled_chen_manifold_readiness_compact_report.v1" or compact.get("prior_status") != "generation_prior_not_accessible" or compact.get("panel_sha256") != parent["panel_sha256"]:
        raise ValueError("parent MRA compact report binding is invalid")
    if audit["evidence"]["context"]["dataset_hash"] != config["dataset"]["expected_manifest_sha256"]:
        raise ValueError("parent MRA dataset manifest binding is invalid")
    confirmation = audit["contract"]["confirmation"]["scaled_chen_step_5000"]
    mra_z_mean = float(confirmation["ols"]["power_moments"]["mean"])
    if not 1.0e-4 <= mra_z_mean <= 1.0e-2:
        raise ValueError("parent MRA normalized-z aggregate sanity anchor is invalid")
    if panel["source_hashes"] != source.hashes:
        raise ValueError("parent MRA panel source hashes do not match registered source")

    return {
        "task_id": TASK_ID,
        "config_path": config_target,
        "config": config,
        "config_sha256": canonical_sha256(config),
        "config_file_sha256": file_sha256(config_target),
        "mra_spec": mra_spec,
        "review_path": review_target,
        "formal": formal,
        "formal_path": formal_path,
        "source": source,
        "audit": audit,
        "compact": compact,
        "panel": panel,
        "mra_attempt": _resolve(parent["output_directory"]).name,
        "checkpoint_step": 5000,
        "parent_prior": source.prior,
        "mra_z_mean": mra_z_mean,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": source.hashes["scaled_chen_step_5000"],
        "calibration_path": calibration_path,
        "calibration_sha256": source.hashes["scaled_chen_calibration"],
    }


def _resource_preflight() -> tuple[int, int]:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_RESOURCE_CONTRACT: CUDA is unavailable")
    free, total = torch.cuda.mem_get_info(device)
    if free < 4 * 1024**3:
        raise RuntimeError("BLOCKED_RESOURCE_CONTRACT: less than 4 GiB free CUDA memory")
    disk = shutil.disk_usage(base._repo_root().anchor or str(base._repo_root()))
    if disk.free < 10 * 1024**3:
        raise RuntimeError("BLOCKED_RESOURCE_CONTRACT: less than 10 GiB free disk")
    return int(free), int(total)


def _figure_alignment(stem: str) -> dict[str, Any]:
    return {
        "schema": ALIGNMENT_SCHEMA,
        "figure": stem,
        "verdict": "NOT_APPLICABLE_SINGLE_PANEL",
        "panel_count": 1,
        "axes_count": 1,
        "text_editable": True,
        "reason": "single-panel scientific diagnostic figure",
    }


def _save_figures(output_path: Path, holdout: torch.Tensor, prior: torch.Tensor, metrics: Mapping[str, Any]) -> None:
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    sorted_holdout = torch.sort(holdout).values.numpy()
    sorted_prior = torch.sort(prior).values.numpy()
    y_holdout = (torch.arange(1, holdout.numel() + 1, dtype=torch.float64) / holdout.numel()).numpy()
    y_prior = (torch.arange(1, prior.numel() + 1, dtype=torch.float64) / prior.numel()).numpy()
    x_max = max(float(prior.max()), float(holdout.max())) * 1.05

    figure = plt.figure(figsize=(6.2, 4.0))
    axis = figure.add_axes((0.12, 0.24, 0.72, 0.62))
    axis.step(sorted_holdout, y_holdout, where="post", color="#1f77b4", linewidth=1.6, label="Encoded holdout")
    axis.step(sorted_prior, y_prior, where="post", color="#d62728", linewidth=1.6, label="Truncated-normal prior")
    axis.set_xlim(0.0, x_max)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Normalized nearest-support distance")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title("G1 support accessibility: exact nearest-support ECDF")
    axis.grid(True, alpha=0.25)
    figure.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.48, 0.04),
        ncol=2,
        handlelength=2.2,
        columnspacing=1.4,
    )
    figure.savefig(output_path / "support_distance_distribution.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_path / "support_distance_distribution.pdf", bbox_inches="tight")
    figure.savefig(output_path / "support_distance_distribution.svg", bbox_inches="tight")
    plt.close(figure)

    labels = ["H50", "P50", "H95", "P95"]
    values = [
        metrics["holdout"]["median"],
        metrics["prior"]["median"],
        metrics["holdout"]["p95"],
        metrics["prior"]["p95"],
    ]
    colors = ["#1f77b4", "#d62728", "#1f77b4", "#d62728"]
    figure = plt.figure(figsize=(6.5, 4.0))
    axis = figure.add_axes((0.11, 0.16, 0.55, 0.70))
    axis.bar(labels, values, color=colors, width=0.62)
    axis.set_yscale("log")
    axis.set_ylim(1.0e-4, max(values) * 1.5)
    axis.set_ylabel("Normalized nearest-support distance (log scale)")
    axis.set_title("G1 support-distance summary")
    axis.grid(True, axis="y", alpha=0.25)
    figure.text(
        0.98,
        0.73,
        "R50 = {:.3f} [{:.3f}, {:.3f}]\nR95 = {:.3f} [{:.3f}, {:.3f}]".format(
            metrics["ratios"]["r50"],
            metrics["bootstrap"]["r50_ci"][0],
            metrics["bootstrap"]["r50_ci"][1],
            metrics["ratios"]["r95"],
            metrics["bootstrap"]["r95_ci"][0],
            metrics["bootstrap"]["r95_ci"][1],
        ),
        ha="right",
        va="top",
        fontsize=7,
    )
    figure.savefig(output_path / "support_distance_summary.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_path / "support_distance_summary.pdf", bbox_inches="tight")
    figure.savefig(output_path / "support_distance_summary.svg", bbox_inches="tight")
    plt.close(figure)


def _artifact_refs(output_path: Path) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for path in sorted(output_path.rglob("*")):
        if path.is_file() and path.name != "observer_manifest.json":
            relative = path.relative_to(output_path).as_posix()
            refs[relative] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    return refs


def _assert_output_tree(output_path: Path) -> None:
    allowed_suffixes = {".json", ".pt", ".png", ".pdf", ".svg"}
    for path in output_path.rglob("*"):
        if path.name.startswith(".") or path.suffix.lower() in {".tmp", ".tif", ".tiff"}:
            raise ValueError(f"invalid hidden, temporary, or TIFF output: {path}")
        if path.is_file() and path.suffix.lower() not in allowed_suffixes:
            raise ValueError(f"unexpected G1 output file: {path}")


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_new_json(path, payload, artifact_kind="G1 support-accessibility artifact")


def render_from_distance_artifact(
    source_directory: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Re-render figures from a completed distance artifact without CUDA replay."""
    source = _resolve(source_directory)
    source_manifest_path = source / "observer_manifest.json"
    source_metrics_path = source / "support_metrics.json"
    source_report_path = source / "compact_report.json"
    source_distance_path = source / "support_distances.pt"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_distance = safe_torch_load(source_distance_path, map_location="cpu")
    if source_manifest.get("schema") != OBSERVER_SCHEMA or source_manifest.get("status") != "completed":
        raise ValueError("render source observer manifest is invalid")
    if source_metrics.get("schema") != METRICS_SCHEMA or source_report.get("schema") != REPORT_SCHEMA:
        raise ValueError("render source metrics or compact report schema is invalid")
    if not isinstance(source_distance, Mapping) or source_distance.get("schema") != DISTANCES_SCHEMA:
        raise ValueError("render source distance artifact schema is invalid")
    source_ref = source_manifest.get("artifacts", {}).get("support_distances.pt", {})
    if source_ref.get("sha256") != file_sha256(source_distance_path):
        raise ValueError("render source distance artifact binding is invalid")
    holdout_distances = source_distance["d_holdout_to_train"]
    prior_distances = source_distance["d_prior_to_train"]
    if not torch.is_tensor(holdout_distances) or not torch.is_tensor(prior_distances):
        raise ValueError("render source distance vectors are invalid")
    holdout_distances = holdout_distances.cpu()
    prior_distances = prior_distances.cpu()
    if holdout_distances.shape != (5000,) or prior_distances.shape != (5000,):
        raise ValueError("render source distance vectors do not match the registered counts")
    actual_output = next_write_once_output(_resolve(output_path))
    actual_output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_distance_path, actual_output / "support_distances.pt")
    _save_figures(actual_output, holdout_distances, prior_distances, source_metrics)
    _write_json_new(actual_output / "support_distance_distribution.alignment.json", _figure_alignment("support_distance_distribution"))
    _write_json_new(actual_output / "support_distance_summary.alignment.json", _figure_alignment("support_distance_summary"))
    shutil.copyfile(source_metrics_path, actual_output / "support_metrics.json")
    shutil.copyfile(source_report_path, actual_output / "compact_report.json")
    _assert_output_tree(actual_output)
    artifact_refs = _artifact_refs(actual_output)
    storage_bytes = sum(item["size_bytes"] for item in artifact_refs.values())
    if storage_bytes > MAX_STORAGE_BYTES:
        raise RuntimeError("G1 render-only output exceeds the diagnostic storage contract")
    manifest = {
        "schema": OBSERVER_SCHEMA,
        "task_id": source_manifest["task_id"],
        "hypothesis": source_manifest["hypothesis"],
        "status": "completed",
        "attempt": actual_output.name,
        "requested_output": str(_resolve(output_path)),
        "output_directory": str(actual_output),
        "supersedes": source.name,
        "render_only_from_distance_artifact": True,
        "source_observer_manifest": {
            "path": str(source_manifest_path),
            "sha256": file_sha256(source_manifest_path),
        },
        "source_distance_artifact": {
            "path": str(source_distance_path),
            "sha256": file_sha256(source_distance_path),
        },
        "git": base._git_state(),
        "config": source_manifest["config"],
        "parent_mra": source_manifest["parent_mra"],
        "lineage": source_manifest["lineage"],
        "dataset": source_manifest["dataset"],
        "prior_sampling": source_manifest["prior_sampling"],
        "distance": source_manifest["distance"],
        "bootstrap": source_manifest["bootstrap"],
        "gate": source_manifest["gate"],
        "orientation": source_manifest["orientation"],
        "artifacts": artifact_refs,
        "storage_bytes_excluding_manifest": storage_bytes,
        "no_update_audit": source_manifest["no_update_audit"],
        "governance": source_manifest["governance"],
        "next_action": source_manifest["next_action"],
    }
    _write_json_new(actual_output / "observer_manifest.json", manifest)
    return {"output_directory": str(actual_output), "report": source_report, "manifest": manifest}


def run(config_path: str | Path = DEFAULT_CONFIG_PATH, review_path: str | Path = DEFAULT_REVIEW_PATH, output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    started = time.perf_counter()
    lineage = validate_registered_lineage(config_path, review_path)
    config = lineage["config"]
    free_before, total_cuda = _resource_preflight()
    dataset_root = _resolve(config["dataset"]["dataset_root"])
    data = mra._load_data(dataset_root)
    if data.manifest["manifest_hash"] != config["dataset"]["expected_manifest_sha256"] or data.manifest["confirmation_indices_sha256"] != config["dataset"]["expected_heldout_indices_sha256"]:
        raise ValueError("registered dataset manifest or heldout index binding is invalid")
    if data.images.shape != (60000, 1, 28, 28) or data.labels.shape != (60000,):
        raise ValueError("registered MNIST train split shape is invalid")
    train_indices = torch.arange(0, 55000, dtype=torch.long)
    heldout_indices = data.confirmation_indices
    if train_indices.numel() != 55000 or heldout_indices.numel() != 5000:
        raise ValueError("registered support/query counts are invalid")

    checkpoint = _load_registered_checkpoint({"checkpoint_path": str(lineage["checkpoint_path"]), "checkpoint_sha256": lineage["checkpoint_sha256"]})
    calibration = _load_registered_calibration({"calibration_path": str(lineage["calibration_path"]), "calibration_sha256": lineage["calibration_sha256"]})
    device = torch.device("cuda")
    model = mra._load_scaled_model(lineage["formal"], checkpoint, device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()) or any(parameter.requires_grad for parameter in model.ols.parameters()):
        raise ValueError("G1 model is not frozen before replay")
    state_before = _state_digest(model)
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        anchor_indices = heldout_indices[:BATCH_SIZE]
        anchor_amplitude = base.render_intensity_batch(data.images.index_select(0, anchor_indices)).amplitude.to(device)
        anchor_z = _encoded_power_latent_prefix(model, anchor_amplitude, calibration)
        anchor_mean = float(anchor_z.mean().detach().cpu())
        del anchor_amplitude, anchor_z
    if not 1.0e-5 <= anchor_mean <= 1.0e-1:
        raise ValueError("invalid_harness: replay anchor normalized-z mean is out of range")

    train_z = _collect_latent_bank(model, data.images, train_indices, calibration, device=device, batch_size=BATCH_SIZE)
    heldout_z = _collect_latent_bank(model, data.images, heldout_indices, calibration, device=device, batch_size=BATCH_SIZE)
    full_mean = float(heldout_z.mean())
    if not 0.25 * EXPECTED_MRA_Z_MEAN <= full_mean <= 4.0 * EXPECTED_MRA_Z_MEAN:
        raise ValueError("invalid_harness: step-5000 normalized-z aggregate disagrees with MRA anchor")
    prior_z = sample_registered_prior(heldout_z.shape[0], seed=PRIOR_SEED)
    if prior_z.shape[0] != heldout_z.shape[0] or abs(float(prior_z.mean()) - 1.0) > 0.1:
        raise ValueError("invalid_harness: registered prior count or mean is invalid")

    torch.cuda.synchronize(device)
    distance_support = train_z.to(device)
    distance_holdout = heldout_z.to(device)
    distance_prior = prior_z.to(device)
    holdout_distances = exact_normalized_nearest_distances(distance_holdout, distance_support).cpu()
    prior_distances = exact_normalized_nearest_distances(distance_prior, distance_support).cpu()
    torch.cuda.synchronize(device)
    del distance_support, distance_holdout, distance_prior
    ratios = {
        "r50": float(_quantile(prior_distances, 0.5) / _quantile(holdout_distances, 0.5)),
        "r95": float(_quantile(prior_distances, 0.95) / _quantile(holdout_distances, 0.95)),
    }
    bootstrap = bootstrap_ratio_ci(
        holdout_distances,
        prior_distances,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        confidence_level=CONFIDENCE_LEVEL,
    )
    p4_gate = classify_p4_gate(bootstrap["r50_ci"], bootstrap["r95_ci"])
    state_after = _state_digest(model)
    if state_before != state_after:
        raise ValueError("invalid_harness: learned model state changed during G1 observation")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("invalid_harness: model was not frozen after G1 observation")

    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    descriptive = {
        "prior_exceedance_h95": float((prior_distances > _quantile(holdout_distances, 0.95)).to(torch.float64).mean()),
        "interpretation": "descriptive_only_not_a_primary_gate",
    }
    metrics: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "latent": {
            "dimension": LATENT_DIM,
            "definition": "z_j=abs(c_j)^2/s_j",
            "dtype": str(heldout_z.dtype),
            "distance_dtype": str(holdout_distances.dtype),
            "normalization": "sqrt_100_only",
            "scale_source": "registered_calibration_port_scale",
        },
        "dataset": {
            "train_support_count": int(train_z.shape[0]),
            "heldout_query_count": int(heldout_z.shape[0]),
            "prior_query_count": int(prior_z.shape[0]),
            "train_support_source": config["dataset"]["train_support_source"],
            "heldout_source": config["dataset"]["heldout_source"],
            "manifest_sha256": data.manifest["manifest_hash"],
            "heldout_indices_sha256": data.manifest["confirmation_indices_sha256"],
        },
        "holdout": _summary_stats(holdout_distances),
        "prior": _summary_stats(prior_distances),
        "ratios": ratios,
        "bootstrap": bootstrap,
        "gate": {
            "name": "P4",
            "classification": p4_gate,
            "rule": "lower CI(R50)>1 and lower CI(R95)>1; or upper CI(R50)<=1 and upper CI(R95)<=1",
        },
        "descriptive": descriptive,
        "replay_anchor": {
            "checkpoint_step": 5000,
            "checkpoint_sha256": lineage["checkpoint_sha256"],
            "model_prefix": "slm_to_front_to_encoder_to_pre_ols_coefficients_to_normalized_power",
            "decoder_entered": False,
            "relay_after_pre_ols": False,
            "anchor_normalized_z_mean": anchor_mean,
            "heldout_normalized_z_mean": full_mean,
            "mra_confirmation_normalized_z_mean": lineage["mra_z_mean"],
            "prior_parameter_mean": 1.0,
            "prior_empirical_mean": float(prior_z.mean()),
        },
        "resource": {
            "device": str(device),
            "latent_dtype": str(heldout_z.dtype),
            "distance_dtype": str(holdout_distances.dtype),
            "free_cuda_bytes_before": free_before,
            "total_cuda_bytes": total_cuda,
            "peak_cuda_allocated_bytes": peak_cuda_bytes,
            "runtime_seconds": runtime_seconds,
        },
        "learned_state_freeze": {
            "state_digest_before": state_before,
            "state_digest_after": state_after,
            "state_unchanged": True,
            "optimizer_created": False,
            "backward_called": False,
            "refit": False,
            "checkpoint_created": False,
            "evaluator_calls": 0,
        },
        "governance": {
            "diagnostic_only": True,
            "generative_manifold_supported": False,
            "generation_readiness": "not_ready",
            "training_authorized": False,
            "formal_promotion": False,
        },
    }

    requested_output = _resolve(output_path)
    actual_output = next_write_once_output(requested_output)
    actual_output.mkdir(parents=True, exist_ok=False)
    distance_payload = {
        "schema": DISTANCES_SCHEMA,
        "task_id": TASK_ID,
        "checkpoint_step": 5000,
        "checkpoint_sha256": lineage["checkpoint_sha256"],
        "train_support": {"source": config["dataset"]["train_support_source"], "start": 0, "stop": 55000, "count": 55000},
        "heldout_query": {"source": config["dataset"]["heldout_source"], "index_sha256": data.manifest["confirmation_indices_sha256"], "count": 5000},
        "prior_query": {"seed": PRIOR_SEED, "kind": "truncated_normal", "count": 5000},
        "d_holdout_to_train": holdout_distances,
        "d_prior_to_train": prior_distances,
    }
    atomic_torch_save_new(actual_output / "support_distances.pt", distance_payload, artifact_kind="G1 distance artifact")
    _save_figures(actual_output, holdout_distances, prior_distances, metrics)
    _write_json_new(actual_output / "support_distance_distribution.alignment.json", _figure_alignment("support_distance_distribution"))
    _write_json_new(actual_output / "support_distance_summary.alignment.json", _figure_alignment("support_distance_summary"))
    _write_json_new(actual_output / "support_metrics.json", metrics)
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "result": p4_gate,
        "r50": ratios["r50"],
        "r50_ci": bootstrap["r50_ci"],
        "r95": ratios["r95"],
        "r95_ci": bootstrap["r95_ci"],
        "prior_exceedance_h95": descriptive["prior_exceedance_h95"],
        "authority": "G1 frozen support-accessibility observer; reconstruction/generation authority unchanged",
        "governance": metrics["governance"],
        "next_action": {
            "P4_SUPPORTED": "PLAN_GENERATIVE_TRAINING_FOR_SUPPORT_ALIGNMENT",
            "P4_NOT_SUPPORTED": "PLAN_G2_PRIOR_DECODER_ACCESSIBILITY",
            "P4_UNDERDETERMINED": "REVIEW_G1_SUPPORT_OBSERVER",
        }[p4_gate],
    }
    _write_json_new(actual_output / "compact_report.json", report)
    _assert_output_tree(actual_output)
    artifact_refs = _artifact_refs(actual_output)
    storage_bytes = sum(item["size_bytes"] for item in artifact_refs.values())
    if storage_bytes > MAX_STORAGE_BYTES:
        raise RuntimeError("G1 output exceeds the diagnostic storage contract")
    manifest = {
        "schema": OBSERVER_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "attempt": actual_output.name,
        "requested_output": str(requested_output),
        "output_directory": str(actual_output),
        "git": base._git_state(),
        "config": {
            "path": str(lineage["config_path"]),
            "file_sha256": lineage["config_file_sha256"],
            "canonical_sha256": lineage["config_sha256"],
        },
        "parent_mra": {
            "attempt": lineage["mra_attempt"],
            "config": str(_resolve(config["parent_mra"]["config"])),
            "review_contract": str(lineage["review_path"]),
            "audit": str(_resolve(config["parent_mra"]["audit"])),
            "compact_report": str(_resolve(config["parent_mra"]["compact_report"])),
            "panel": str(_resolve(config["parent_mra"]["panel"])),
            "panel_sha256": config["parent_mra"]["panel_sha256"],
            "status": lineage["audit"]["status"],
            "prior_status": lineage["audit"]["contract"]["prior_status"],
        },
        "lineage": {
            "checkpoint_path": str(lineage["checkpoint_path"]),
            "checkpoint_sha256": lineage["checkpoint_sha256"],
            "checkpoint_step": 5000,
            "calibration_path": str(lineage["calibration_path"]),
            "calibration_sha256": lineage["calibration_sha256"],
            "mra_config_canonical_sha256": config["parent_mra"]["config_canonical_sha256"],
            "latent_definition": "z_j=abs(c_j)^2/s_j",
        },
        "dataset": metrics["dataset"],
        "prior_sampling": config["prior_sampling"],
        "distance": config["distance"],
        "bootstrap": config["bootstrap"],
        "gate": metrics["gate"],
        "orientation": {"latent_matrix": "non_spatial_matrix_no_rotation", "distance_vectors": "one_dimensional"},
        "artifacts": artifact_refs,
        "storage_bytes_excluding_manifest": storage_bytes,
        "no_update_audit": metrics["learned_state_freeze"] | {
            "decoder_entered": False,
            "prior_decoded": False,
            "relay_after_pre_ols": False,
            "cmos_called": False,
            "mpcdar_called": False,
        },
        "governance": metrics["governance"],
        "next_action": report["next_action"],
    }
    _write_json_new(actual_output / "observer_manifest.json", manifest)
    return {"output_directory": str(actual_output), "metrics": metrics, "report": report, "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--review", default=str(DEFAULT_REVIEW_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--render-from-distance-artifact", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.render_from_distance_artifact is not None:
        result = render_from_distance_artifact(args.render_from_distance_artifact, args.output)
    else:
        result = run(args.config, args.review, args.output)
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
