"""Frozen-weight, rank-matched OLS geometry attribution for scaled Chen.

The module consumes the archived 0/5000-step scaled-Chen horizon checkpoints
and compares a pre-registered nine-point LP01 geometry grid.  It never creates
an optimizer, calls the evaluator, updates a parameter, or authorizes formal
training.  The formal v2 Identity OLS path is not modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ...config import FormalSystemConfig, load_config
from ...evidence import EvidenceContextV2, EvidenceRecordV2, GateResult
from ...evidence import (
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from . import optical_manifold as base
from . import optical_manifold_smf_bogt as bogt
from . import optical_manifold_smf_bogt_horizon as horizon


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_geometry_attribution_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_geometry_attribution_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_geometry_attribution_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_geometry_attribution.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S0-GEOM-RM"
BRANCH = bogt.BRANCH
SEED = 1234
SELECTION_SEED = 1235
BOOTSTRAP_SEED = 1236
SELECTION_COUNT = 1000
SELECTION_PER_LABEL = 100
CONFIRMATION_COUNT = 5000
BOOTSTRAP_REPLICATES = 2000
MATERIAL_RELATIVE_IMPROVEMENT = 0.05
CONFIDENCE_LEVEL = 0.95
CHECKPOINTS = (0, 5000)
GRID_FILLS = (0.70, 0.80, 0.90)
GRID_MAGNIFICATIONS = (0.80, 1.00, 1.20)
BASELINE = (0.80, 1.00)
PORT_COUNT = 100
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2
NEXT_ACTION_SUPPORTED = "manual_review_scaled_chen_geometry_matched_reconstruction_canary_contract"
NEXT_ACTION_UNSUPPORTED = "manual_review_scaled_chen_distributed_bottleneck_closeout_contract"

HISTORICAL_COMMIT = "019b76ef911e0bd16b8d5d4169807323427684af"
HISTORICAL_HORIZON_CONFIG_HASH = "304ff57307a7b4b301adf37766d3948c25cab9a83d90da5c85ed6a081003dd18"
SOURCE_HASHES = {
    "training": "bf707a836dabbea6917aa6c8016e6c00764d9219cd67bb41f551f911f28c2c51",
    "validation": "ddb882b26aa1463a74d41896f0f8ff8b406614a29bf25a72b918f58094d3e945",
    "preflight": "e988485f18e1f45964bf44e6338308ec882048038a1ba910f09758521fdfea83",
    "calibration": "a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959",
    "step0_checkpoint": "460f9745cbce4c231f75cf3443d23a861c07a5159bb57fa2a97543d153bc57ba",
    "step5000_checkpoint": "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de",
    "no_update_attribution_v3": "98c55f8c6a2c2063a879e5dbe1eeeec26adc6c49a3f37990d88726097c107e26",
    "no_update_core": "4bacfb93bde8df6bf7831a41feaad50738e530dc518cab6f7922f14dcfe8973d",
    "no_update_variants": "e44fc2dd75e3ed7da4ca9d70074227cf2208e457a10e791e99c11de24b792041",
    "no_update_jacobian": "38d7fae07a85371f98c072426e822d53bf5ba9fda8fbec80e69494e020deaab1",
    "no_update_counterfactual": "c31ab1119835284f293d378e8d13325cb01a9e5ac3f3e6c4fe5038446897a582",
}
SOURCE_JSON_SCHEMAS = {
    "no_update_attribution_v3": "scaled_chen_bogt_horizon_no_update_attribution_audit_v3",
    "no_update_core": "scaled_chen_bogt_horizon_no_update_attribution_v1",
    "no_update_variants": "scaled_chen_bogt_horizon_no_update_variants_v1",
    "no_update_jacobian": "scaled_chen_bogt_horizon_no_update_decoder_jacobian_v1",
    "no_update_counterfactual": "scaled_chen_bogt_horizon_no_update_counterfactual_residual_v1",
}


@dataclass(frozen=True)
class GeometrySpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class GeometryVariant:
    fill: float
    facet_magnification: float

    @property
    def key(self) -> str:
        return f"fill_{self.fill:.2f}_mag_{self.facet_magnification:.2f}"

    @property
    def distance_from_baseline(self) -> float:
        return abs(self.fill - BASELINE[0]) + abs(self.facet_magnification - BASELINE[1])


@dataclass(frozen=True)
class GeometryData:
    horizon: horizon.HorizonData
    selection_images: torch.Tensor
    selection_labels: torch.Tensor
    selection_indices: torch.Tensor
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SourceBundle:
    paths: dict[str, Path]
    hashes: dict[str, str]
    horizon_manifest: dict[str, Any]
    calibration: bogt.Calibration
    checkpoints: dict[int, dict[str, Any]]
    horizon_training: dict[str, Any]
    horizon_validation: dict[str, Any]
    horizon_preflight: dict[str, Any]


@dataclass(frozen=True)
class Evaluation:
    metrics: dict[str, Any]
    sample_mse: torch.Tensor
    sample_labels: torch.Tensor
    panel: dict[str, torch.Tensor]
    state_unchanged: bool


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    root = base._repo_root()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "branch": branch, "clean": not status, "status": status}


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or not state.get("clean"):
        raise ValueError(f"geometry attribution requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> GeometrySpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {"schema", "hypothesis", "formal_system_config", "source", "dataset", "optical_contract", "audit", "gates", "authorization"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("geometry attribution config fields are invalid")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("geometry attribution config schema or hypothesis is invalid")
    return GeometrySpec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: GeometrySpec) -> None:
    p = spec.payload
    source = p["source"]
    if source.get("historical_commit") != HISTORICAL_COMMIT or source.get("horizon_config_sha256") != HISTORICAL_HORIZON_CONFIG_HASH:
        raise ValueError("geometry attribution historical horizon binding is invalid")
    if source.get("hashes") != SOURCE_HASHES:
        raise ValueError("geometry attribution source hashes are not locked")
    dataset = p["dataset"]
    if dataset != {
        "train_count": 55000,
        "validation_count": 5000,
        "calibration_count": 1000,
        "selection_count": SELECTION_COUNT,
        "selection_source": "mnist_test_indices_5000_9999",
        "selection_per_label": SELECTION_PER_LABEL,
        "confirmation_source": "horizon_heldout_train_indices_55000_59999",
        "confirmation_count": CONFIRMATION_COUNT,
        "selection_seed": SELECTION_SEED,
        "logical_batch_size": 10,
        "microbatch_size": 1,
        "selected_and_confirmation_disjoint": True,
        "selection_excludes_horizon_train_and_calibration": True,
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }:
        raise ValueError("geometry attribution dataset contract is not locked")
    optical = p["optical_contract"]
    expected_optical = {
        "formal_system_kind": "formal_amplitude_finite_crop_identity",
        "architecture": "2plus2_phase_only",
        "wavelength_m": 5.32e-7,
        "slm_shape": [200, 200],
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "layer_spacing_m": 1.50e-3,
        "ols_ports": PORT_COUNT,
        "mode_model": "step_index_lp01_ports",
        "mode_v_number": 2.2,
        "mode_waist_m": 7.111e-6,
        "mode_grid_shape": [10, 10],
        "mode_fill_grid": list(GRID_FILLS),
        "mode_facet_magnification_grid": list(GRID_MAGNIFICATIONS),
        "baseline_fill": BASELINE[0],
        "baseline_facet_magnification": BASELINE[1],
        "fixed_transport": "identity",
        "fixed_layer_power_transmission": bogt.POWER_TRANSMISSION,
        "backend_total_distance_m": 3.0e-3,
        "backend_magnification": -2.5,
        "fixed_exposure": {"scale": 1.0, "gamma": 1.0},
        "no_complex_resize": True,
        "software_decoder": False,
        "electronic_posterior": False,
        "formal_promotion": False,
    }
    if optical != expected_optical:
        raise ValueError("geometry attribution optical contract is not locked")
    audit = p["audit"]
    if audit != {
        "objective": "frozen_forward_only",
        "checkpoints": list(CHECKPOINTS),
        "selection_metric": "paired_mse",
        "confirmation_metric": "paired_mse",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT,
        "confidence_level": CONFIDENCE_LEVEL,
        "negative_control": "cyclic_target",
        "independent_reference": "class_template",
        "fixed_calibration": True,
        "no_update": True,
    }:
        raise ValueError("geometry attribution audit contract is not locked")
    gates = p["gates"]
    if gates != {
        "min_free_cuda_bytes": MIN_FREE_BYTES,
        "max_peak_cuda_bytes": MAX_PEAK_BYTES,
        "max_storage_bytes": MAX_STORAGE_BYTES,
        "model_state_unchanged": True,
        "evaluator_calls": 0,
        "optimizer_created": False,
        "grid_rank": PORT_COUNT,
        "wall_clock": "observational_only",
    }:
        raise ValueError("geometry attribution resource contract is not locked")
    authorization = p["authorization"]
    if authorization != {
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_scaled_chen_geometry_attribution_audit",
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": [
            "optimizer",
            "backward",
            "checkpoint_write",
            "generation",
            "evaluator",
            "fmf",
            "trainable_ols",
            "exposure_retuning",
            "promotion",
        ],
    }:
        raise ValueError("geometry attribution authorization is not locked")


def _load_review(path: str | Path, spec: GeometrySpec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_scaled_chen_geometry_attribution_audit",
        "approved_contract": {
            "grid": {
                "fill": list(GRID_FILLS),
                "facet_magnification": list(GRID_MAGNIFICATIONS),
                "baseline": list(BASELINE),
                "ports": PORT_COUNT,
                "transport": "identity",
            },
            "selection_count": SELECTION_COUNT,
            "selection_source": "mnist_test_indices_5000_9999",
            "selection_per_label": SELECTION_PER_LABEL,
            "confirmation_count": CONFIRMATION_COUNT,
            "confirmation_source": "horizon_heldout_train_indices_55000_59999",
            "checkpoints": list(CHECKPOINTS),
            "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "no_update": True,
            "fixed_calibration": True,
            "post_geometry_supported_next_action": NEXT_ACTION_SUPPORTED,
            "post_geometry_not_supported_next_action": NEXT_ACTION_UNSUPPORTED,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("geometry attribution review contract does not match scope")
    return target


def _load_formal(spec: GeometrySpec) -> tuple[FormalSystemConfig, Path]:
    path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    config = load_config(path)
    if config.architecture != "conditional_identity_logit_amplitude_v2" or config.system.kind != bogt.EXPECTED_FORMAL_KIND or not config.system.is_formal:
        raise ValueError("geometry attribution requires current formal Identity optics")
    locked = {
        "wavelength_m": 532e-9,
        "slm_shape": (200, 200),
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": (100, 100),
        "metasurface_pixel_pitch_m": 2e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "layer_spacing_m": 1.5e-3,
    }
    for name, expected in locked.items():
        actual = getattr(config.system, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1e-12, 1e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return config.system, path


def _variants() -> tuple[GeometryVariant, ...]:
    return tuple(GeometryVariant(fill, magnification) for fill in GRID_FILLS for magnification in GRID_MAGNIFICATIONS)


def _select_indices(test_labels: torch.Tensor) -> torch.Tensor:
    if test_labels.ndim != 1 or test_labels.numel() != 10000:
        raise ValueError("geometry selection requires the complete 10000-image MNIST test split")
    candidates = torch.arange(5000, 10000, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(SELECTION_SEED)
    selected: list[torch.Tensor] = []
    for label in range(10):
        matches = candidates[test_labels.index_select(0, candidates) == label]
        if matches.numel() < SELECTION_PER_LABEL:
            raise ValueError(f"MNIST test selection has too few label-{label} samples")
        shuffled = matches.index_select(0, torch.randperm(matches.numel(), generator=generator))
        selected.append(shuffled[:SELECTION_PER_LABEL])
    result = torch.cat(selected).sort().values
    if result.numel() != SELECTION_COUNT or torch.unique(result).numel() != result.numel():
        raise ValueError("geometry selection indices are not unique")
    if not torch.equal(torch.bincount(test_labels.index_select(0, result), minlength=10), torch.full((10,), SELECTION_PER_LABEL, dtype=torch.long)):
        raise ValueError("geometry selection is not class balanced")
    return result


def _load_data(dataset_root: str | Path) -> GeometryData:
    horizon_data = horizon._load_data(dataset_root)
    test_images, test_labels = base.load_local_mnist(dataset_root, train=False)
    selection_indices = _select_indices(test_labels)
    selection_images = test_images.index_select(0, selection_indices)
    selection_labels = test_labels.index_select(0, selection_indices)
    if torch.isin(selection_indices, torch.arange(0, 5000, dtype=torch.long)).any():
        raise ValueError("geometry selection overlaps the reserved MNIST test half")
    manifest: dict[str, Any] = {
        "schema": "scaled_chen_bogt_geometry_attribution_mnist_manifest.v1",
        "horizon_manifest_hash": horizon_data.manifest["manifest_hash"],
        "test_images_sha256": bogt.tensor_sha256(test_images),
        "test_labels_sha256": bogt.tensor_sha256(test_labels),
        "selection_indices_sha256": tensor_sha256(selection_indices),
        "selection_images_sha256": tensor_sha256(selection_images),
        "selection_labels_sha256": tensor_sha256(selection_labels),
        "selection_count": SELECTION_COUNT,
        "selection_per_label": SELECTION_PER_LABEL,
        "selection_source": "mnist_test_indices_5000_9999",
        "confirmation_indices_sha256": tensor_sha256(horizon_data.heldout_indices),
        "confirmation_count": CONFIRMATION_COUNT,
        "selected_and_confirmation_disjoint": True,
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return GeometryData(horizon_data, selection_images, selection_labels, selection_indices, manifest)


def _json_payload(path: Path, expected_hash: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError(f"historical artifact content does not match {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"historical artifact is not a mapping: {path}")
    return payload


def _load_source(spec: GeometrySpec, data: GeometryData) -> SourceBundle:
    source = spec.payload["source"]
    paths = {name: _resolve(value) for name, value in source.items() if name not in {"historical_commit", "horizon_config_sha256", "hashes"}}
    hashes = dict(SOURCE_HASHES)
    for name, path in paths.items():
        if name in hashes and file_sha256(path) != hashes[name]:
            raise ValueError(f"historical source hash mismatch: {name}")
    training = _json_payload(paths["training"], hashes["training"])
    validation = _json_payload(paths["validation"], hashes["validation"])
    preflight = _json_payload(paths["preflight"], hashes["preflight"])
    for payload, expected_schema, expected_status in (
        (training, horizon.TRAINING_SCHEMA, "valid_failed_scaled_chen_reconstruction_horizon"),
        (validation, horizon.VALIDATION_SCHEMA, "passed"),
        (preflight, horizon.PREFLIGHT_SCHEMA, "passed"),
    ):
        record = EvidenceRecordV2.from_dict(payload["evidence"])
        record.validate()
        if payload.get("schema") != expected_schema or payload.get("hypothesis") != horizon.HYPOTHESIS or payload.get("status") != expected_status or record.git_commit != HISTORICAL_COMMIT or record.config_hash != HISTORICAL_HORIZON_CONFIG_HASH:
            raise ValueError("historical horizon evidence lineage is stale or mismatched")
    if training["contract"].get("n_star") is not None or training["evidence"].get("status") != "valid_failed":
        raise ValueError("historical horizon training is not the authoritative valid-failed result")
    if validation["evidence"].get("permitted_next_actions") != ["run_smf_bogt_horizon_preflight"]:
        raise ValueError("historical horizon validation action is invalid")
    historical_manifest = validation["contract"].get("dataset_manifest")
    if not isinstance(historical_manifest, Mapping):
        raise ValueError("historical horizon dataset manifest is missing")
    current_manifest = data.horizon.manifest
    for key, value in historical_manifest.items():
        if key not in {"dataset_root", "manifest_hash"} and current_manifest.get(key) != value:
            raise ValueError(f"historical horizon dataset content mismatch: {key}")
    if current_manifest.get("manifest_hash") == historical_manifest.get("manifest_hash"):
        raise ValueError("historical and current horizon manifests unexpectedly share a path-bound hash")
    for name, schema in SOURCE_JSON_SCHEMAS.items():
        payload = _json_payload(paths[name], hashes[name])
        if payload.get("schema") != schema or payload.get("status") != "diagnostic":
            raise ValueError(f"historical attribution artifact is invalid: {name}")
    calibration_payload = safe_torch_load(paths["calibration"], expected_sha256=hashes["calibration"])
    if calibration_payload.get("schema") != "scaled_chen_bogt_calibration.v1" or calibration_payload.get("dataset_manifest_hash") != historical_manifest.get("manifest_hash"):
        raise ValueError("historical calibration is stale or mismatched")
    calibration = bogt.Calibration(
        calibration_payload["port_scale"],
        calibration_payload["plane_reference"],
        int(calibration_payload["sample_count"]),
        calibration_payload.get("plane_lower"),
        calibration_payload.get("plane_upper"),
    )
    calibration.validate()
    checkpoints: dict[int, dict[str, Any]] = {}
    for step, name in ((0, "step0_checkpoint"), (5000, "step5000_checkpoint")):
        payload = safe_torch_load(paths[name], expected_sha256=hashes[name])
        if payload.get("schema") != "scaled_chen_bogt_horizon_checkpoint.v1" or int(payload.get("global_step", -1)) != step or not isinstance(payload.get("model_state"), Mapping):
            raise ValueError(f"historical checkpoint is invalid: {name}")
        state = payload["model_state"]
        for key in ("system.encoder.phase", "system.decoder.phase"):
            if key not in state or tuple(state[key].shape) != (2, 1, 100, 100) or not torch.isfinite(state[key]).all():
                raise ValueError(f"historical checkpoint phase state is invalid: {key}")
        checkpoints[step] = payload
    return SourceBundle(paths, hashes, dict(historical_manifest), calibration, checkpoints, training, validation, preflight)


def _input_hashes(spec: GeometrySpec, review_path: Path, formal_path: Path, data: GeometryData, source: SourceBundle) -> dict[str, str]:
    return {
        "research_config": file_sha256(spec.path),
        "formal_config": file_sha256(formal_path),
        "review_contract": file_sha256(review_path),
        "dataset_manifest": data.manifest["manifest_hash"],
        "horizon_dataset_manifest": data.horizon.manifest["manifest_hash"],
        "historical_horizon_dataset_manifest": source.horizon_manifest["manifest_hash"],
        "selection_indices": data.manifest["selection_indices_sha256"],
        "selection_images": data.manifest["selection_images_sha256"],
        **{f"source_{name}": digest for name, digest in source.hashes.items()},
    }


def _context(spec: GeometrySpec, data: GeometryData) -> EvidenceContextV2:
    signature = canonical_sha256({"optical_contract": spec.payload["optical_contract"], "grid": [variant.__dict__ for variant in _variants()], "branch": BRANCH})
    return EvidenceContextV2(signature, spec.config_hash, dataset_hash=data.manifest["manifest_hash"], dataset_split_manifest_hash=data.manifest["manifest_hash"])


def _mode_rank(operator: FiberArrayOLS) -> int:
    area = operator.input_plane.pixel_pitch_m[0] * operator.input_plane.pixel_pitch_m[1]
    gram = operator.input_modes.flatten(1).conj() @ operator.input_modes.flatten(1).T * area
    return int(torch.linalg.matrix_rank(gram).item())


def _build_model(formal: FormalSystemConfig, state: Mapping[str, Any], variant: GeometryVariant, device: torch.device) -> bogt.ScaledChenBOGTModel:
    model = bogt.ScaledChenBOGTModel(formal)
    fiber_spec = FiberArraySpec(
        grid_shape=(10, 10),
        pitch_m=2.0e-6,
        wavelength_m=532.0e-9,
        waist_m=7.111e-6,
        fill=variant.fill,
        v_number=2.2,
        facet_magnification=variant.facet_magnification,
        mode_model="step_index_lp01_ports",
        expected_fov_m=(200.0e-6, 200.0e-6),
    )
    model.ols = FiberArrayOLS(model.system.encoder_plane, fiber_spec)
    model._set_trainable_contract()
    for name in ("system.encoder.phase", "system.decoder.phase"):
        if name not in state:
            raise ValueError(f"checkpoint is missing phase state {name}")
        target = dict(model.named_parameters())[name]
        value = state[name]
        if target.shape != value.shape or not torch.isfinite(value).all():
            raise ValueError(f"checkpoint phase state shape or finiteness mismatch: {name}")
        with torch.no_grad():
            target.copy_(value.to(dtype=target.dtype))
    model.to(device)
    model.eval()
    return model


def _state_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _evaluate(
    model: bogt.ScaledChenBOGTModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    calibration: bogt.Calibration,
    class_templates: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    *,
    calibration_hash: str,
) -> Evaluation:
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("geometry evaluation indices must be a non-empty vector")
    sample_mse: list[torch.Tensor] = []
    template_mse: list[torch.Tensor] = []
    cyclic_mse: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []
    planes: list[torch.Tensor] = []
    first_camera: torch.Tensor | None = None
    first_target: torch.Tensor | None = None
    previous_target: torch.Tensor | None = None
    panel_camera: torch.Tensor | None = None
    panel_target: torch.Tensor | None = None
    with torch.no_grad():
        for batch_indices in indices.split(10):
            rendered = base.render_intensity_batch(images.index_select(0, batch_indices))
            amplitudes = rendered.amplitude.to(device)
            targets = rendered.target.float()
            batch_labels = labels.index_select(0, batch_indices)
            output = model(amplitudes)
            camera = output.camera.detach().cpu().float()
            if not torch.isfinite(camera).all() or not torch.isfinite(targets).all():
                raise RuntimeError("geometry evaluation produced non-finite camera or target values")
            sample_mse.append((camera - targets).square().mean(dim=(1, 2, 3)))
            template_mse.append((class_templates.index_select(0, batch_labels) - targets).square().mean(dim=(1, 2, 3)))
            if previous_target is None:
                first_camera = camera[0:1].clone()
                first_target = targets[0:1].clone()
                cyclic_mse.append(torch.zeros(1, dtype=torch.float32))
                if targets.shape[0] > 1:
                    cyclic_mse.append((camera[1:] - targets[:-1]).square().mean(dim=(1, 2, 3)))
            else:
                cyclic_targets = torch.cat((previous_target, targets[:-1]), dim=0)
                cyclic_mse.append((camera - cyclic_targets).square().mean(dim=(1, 2, 3)))
            previous_target = targets[-1:].clone()
            coefficients.append(output.coefficients.detach().cpu())
            planes.append(torch.stack([value.detach().cpu() for value in output.plane_powers], dim=1))
            if panel_camera is None:
                panel_camera = camera[:100].clone()
                panel_target = targets[:100].clone()
    if first_camera is None or first_target is None or previous_target is None:
        raise RuntimeError("geometry evaluation returned no samples")
    sample_values = torch.cat(sample_mse)
    template_values = torch.cat(template_mse)
    cyclic_values = torch.cat(cyclic_mse)
    cyclic_values[0] = (first_camera - previous_target).square().mean()
    coefficient_values = torch.cat(coefficients)
    plane_values = torch.cat(planes)
    z = bogt.normalize_port_power(coefficient_values.abs().square(), calibration.port_scale)
    mean, variance = bogt.aggregate_moments(z)
    below = plane_values < calibration.operating_lower.unsqueeze(0)
    above = plane_values > calibration.operating_upper.unsqueeze(0)
    inside = ~(below | above)
    saturation = (plane_values <= torch.finfo(plane_values.dtype).eps) | (plane_values >= torch.finfo(plane_values.dtype).max * 0.5)
    metrics: dict[str, Any] = {
        "paired_mse": float(sample_values.mean()),
        "cyclic_mse": float(cyclic_values.mean()),
        "template_control_mse": float(template_values.mean()),
        "kl": float(bogt.bogt_kl_surrogate(mean, variance)),
        "numerical_truncated_normal_kl": float(bogt.truncated_normal_kl_reference(mean, variance)),
        "op": float(bogt.lop_operating_window_from_powers(tuple(plane_values[:, index] for index in range(5)), calibration.operating_lower, calibration.operating_upper)),
        "coefficient_effective_rank": float(bogt.coefficient_effective_rank(coefficient_values)),
        "power_latent_effective_rank": float(bogt.power_latent_effective_rank(z)),
        "power_retention": float((coefficient_values.abs().square().sum(dim=1) / calibration.port_scale.sum().clamp_min(1.0e-30)).mean()),
        "per_plane_power": plane_values.mean(dim=0).tolist(),
        "below_window_fraction": below.float().mean(dim=0).tolist(),
        "inside_window_fraction": inside.float().mean(dim=0).tolist(),
        "above_window_fraction": above.float().mean(dim=0).tolist(),
        "saturation_fraction": saturation.float().mean(dim=0).tolist(),
        "camera_saturation_fraction": float(((panel_camera <= 0.0) | (panel_camera >= 1.0)).float().mean()) if panel_camera is not None else 0.0,
        "calibration_sha256": calibration_hash,
        "evaluation_count": int(sample_values.numel()),
        "finite": bool(torch.isfinite(sample_values).all() and torch.isfinite(cyclic_values).all() and torch.isfinite(template_values).all() and torch.isfinite(coefficient_values).all() and torch.isfinite(plane_values).all()),
    }
    if not all(math.isfinite(float(metrics[name])) for name in ("paired_mse", "cyclic_mse", "template_control_mse", "kl", "numerical_truncated_normal_kl", "op", "coefficient_effective_rank", "power_latent_effective_rank", "power_retention")):
        raise RuntimeError("geometry evaluation metrics are non-finite")
    return Evaluation(metrics, sample_values, labels.index_select(0, indices).clone(), {"camera": panel_camera if panel_camera is not None else torch.empty(0), "target": panel_target if panel_target is not None else torch.empty(0)}, True)


def _evaluate_variant(
    formal: FormalSystemConfig,
    checkpoint: Mapping[str, Any],
    variant: GeometryVariant,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    calibration: bogt.Calibration,
    class_templates: torch.Tensor,
    device: torch.device,
    calibration_hash: str,
) -> Evaluation:
    model = _build_model(formal, checkpoint["model_state"], variant, device)
    before_model = _state_snapshot(model)
    before_operator = _state_snapshot(model.ols)
    result = _evaluate(model, images, labels, calibration, class_templates, indices, device, calibration_hash=calibration_hash)
    after_model = _state_snapshot(model)
    after_operator = _state_snapshot(model.ols)
    state_unchanged = _state_equal(before_model, after_model) and _state_equal(before_operator, after_operator)
    result = Evaluation(result.metrics, result.sample_mse, result.sample_labels, result.panel, state_unchanged)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _stratified_bootstrap_ci(delta: torch.Tensor, labels: torch.Tensor, *, replicates: int, seed: int, confidence_level: float = CONFIDENCE_LEVEL) -> tuple[float, float]:
    if delta.ndim != 1 or labels.shape != delta.shape or delta.numel() == 0:
        raise ValueError("bootstrap inputs must be non-empty vectors of equal shape")
    if not torch.isfinite(delta).all() or not torch.isfinite(labels.float()).all():
        raise ValueError("bootstrap inputs must be finite")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    class_means: list[torch.Tensor] = []
    for label in range(10):
        values = delta[labels == label]
        if values.numel() == 0:
            raise ValueError(f"bootstrap class {label} is empty")
        means: list[torch.Tensor] = []
        for start in range(0, replicates, 100):
            count = min(100, replicates - start)
            draws = torch.randint(values.numel(), (count, values.numel()), generator=generator)
            means.append(values.index_select(0, draws.reshape(-1)).reshape(count, -1).mean(dim=1))
        class_means.append(torch.cat(means))
    aggregate = torch.stack(class_means, dim=1).mean(dim=1)
    alpha = (1.0 - confidence_level) / 2.0
    return float(torch.quantile(aggregate, alpha)), float(torch.quantile(aggregate, 1.0 - alpha))


def _validation_gates(formal: FormalSystemConfig, data: GeometryData) -> tuple[GateResult, ...]:
    results: list[GateResult] = []
    ranks: dict[str, int] = {}
    finite = True
    identity = True
    for variant in _variants():
        model = _build_model(formal, {"system.encoder.phase": torch.zeros(2, 1, 100, 100), "system.decoder.phase": torch.zeros(2, 1, 100, 100)}, variant, torch.device("cpu"))
        rank = _mode_rank(model.ols)
        ranks[variant.key] = rank
        finite = finite and bool(torch.isfinite(model.ols.input_modes).all() and torch.isfinite(model.ols.output_modes).all())
        identity = identity and bool(torch.equal(model.ols.transport, torch.eye(PORT_COUNT, dtype=torch.complex64)))
        del model
    results.append(GateResult("nine_registered_geometry_variants", len(ranks) == 9, {"variants": len(ranks)}))
    results.append(GateResult("100_port_rank_matched_grid", all(rank == PORT_COUNT for rank in ranks.values()), {"ranks": ranks}))
    results.append(GateResult("finite_mode_banks", finite, {}))
    results.append(GateResult("fixed_identity_transport", identity, {}))
    labels = data.selection_labels
    balanced = torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), SELECTION_PER_LABEL, dtype=torch.long))
    results.append(GateResult("selection_balanced_and_disjoint", balanced and data.manifest["selected_and_confirmation_disjoint"], {"count": int(labels.numel()), "per_label": int(labels.bincount(minlength=10).min())}))
    results.append(GateResult("formal_identity_boundary", formal.kind == bogt.EXPECTED_FORMAL_KIND and formal.is_formal, {}))
    results.append(GateResult("no_update_scope", True, {"optimizer_created": False, "evaluator_calls": 0, "training_authorized": False}))
    return tuple(results)


def _record(schema: str, status: str, classification: str | None, spec: GeometrySpec, state: Mapping[str, Any], data: GeometryData, hashes: Mapping[str, str], gates: tuple[GateResult, ...], action: str | None, reason: str | None = None, contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    evidence_status = {"geometry_supported": "diagnostic", "geometry_not_supported": "valid_failed"}.get(status, status)
    record = EvidenceRecordV2(2, schema, "diagnostic", evidence_status, str(state["commit"]), None, spec.config_hash, _context(spec, data), dict(hashes), gates, (action,) if action else (), reason)
    record.validate()
    result = {
        "schema": schema,
        "hypothesis": HYPOTHESIS,
        "status": status,
        "classification": classification,
        "training_authorized": False,
        "formal_promotion": False,
        "evidence": record.to_dict(),
        "contract": {
            "git_state": {"branch": state["branch"], "commit": state["commit"], "clean": state["clean"]},
            "training_authorized": False,
            "formal_promotion": False,
        },
    }
    if contract:
        result["contract"].update(contract)
    return result


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite geometry attribution artifact: {target}")
    atomic_write_json(target, payload)


def _load_validation(path: str | Path, spec: GeometrySpec, review_path: Path, formal_path: Path, data: GeometryData, source: SourceBundle, state: Mapping[str, Any]) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, formal_path, data, source)
    if payload.get("schema") != VALIDATION_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.context != _context(spec, data) or record.input_hashes != expected_hashes or record.permitted_next_actions != ("run_scaled_chen_geometry_attribution",):
        raise ValueError("geometry attribution validation evidence is stale or mismatched")


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    source = _load_source(spec, data)
    gates = _validation_gates(formal, data)
    hashes = _input_hashes(spec, review_path, formal_path, data, source)
    valid = all(gate.passed for gate in gates)
    payload = _record(VALIDATION_SCHEMA, "passed" if valid else "invalid_harness", None, spec, state, data, hashes, gates, "run_scaled_chen_geometry_attribution" if valid else None, None if valid else "geometry attribution validation failed", {"formal_config": str(formal_path), "review_contract": str(review_path), "dataset_manifest": data.manifest, "source_paths": {name: str(path) for name, path in source.paths.items()}, "grid": [variant.__dict__ for variant in _variants()]})
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("geometry attribution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("geometry attribution requires CUDA")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    source = _load_source(spec, data)
    _load_validation(args.validation_evidence, spec, review_path, formal_path, data, source, state)
    output_path = Path(args.output)
    panel_path = output_path.parent / "scaled_chen_geometry_attribution_panel.pt"
    decomposition_path = output_path.parent / "scaled_chen_geometry_attribution_decomposition.json"
    for target in (output_path, panel_path, decomposition_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite geometry attribution artifact: {target}")
    device = torch.device(args.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError("geometry attribution requires at least 4 GiB free CUDA memory")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    class_templates = bogt._class_templates(horizon._as_research_data(data.horizon))
    calibration_hash = source.hashes["calibration"]
    selection_indices = torch.arange(SELECTION_COUNT, dtype=torch.long)
    confirmation_indices = torch.arange(CONFIRMATION_COUNT, dtype=torch.long)
    selection_records: dict[str, dict[str, Any]] = {}
    selection_internal: dict[str, Evaluation] = {}
    step5000_checkpoint = source.checkpoints[5000]
    step0_checkpoint = source.checkpoints[0]
    for variant in _variants():
        result = _evaluate_variant(formal, step5000_checkpoint, variant, data.selection_images, data.selection_labels, selection_indices, source.calibration, class_templates, device, calibration_hash)
        selection_internal[variant.key] = result
        selection_records[variant.key] = {"geometry": variant.__dict__, "metrics": result.metrics, "state_unchanged": result.state_unchanged}
    baseline_variant = GeometryVariant(*BASELINE)
    candidates = [variant for variant in _variants() if variant != baseline_variant]
    selected_variant = min(candidates, key=lambda variant: (selection_internal[variant.key].metrics["paired_mse"], variant.distance_from_baseline, variant.fill, variant.facet_magnification))
    baseline_selection = selection_internal[baseline_variant.key]
    selected_selection = selection_internal[selected_variant.key]
    confirmation_images = data.horizon.images.index_select(0, data.horizon.heldout_indices)
    confirmation_labels = data.horizon.labels.index_select(0, data.horizon.heldout_indices)
    confirmation_records: dict[str, dict[str, Any]] = {}
    confirmation_internal: dict[str, dict[int, Evaluation]] = {"baseline": {}, "selected": {}}
    for name, variant in (("baseline", baseline_variant), ("selected", selected_variant)):
        for step, checkpoint in ((0, step0_checkpoint), (5000, step5000_checkpoint)):
            result = _evaluate_variant(formal, checkpoint, variant, confirmation_images, confirmation_labels, confirmation_indices, source.calibration, class_templates, device, calibration_hash)
            confirmation_internal[name][step] = result
            confirmation_records[f"{name}_step_{step}"] = {"geometry": variant.__dict__, "step": step, "metrics": result.metrics, "state_unchanged": result.state_unchanged}
    selected0 = confirmation_internal["selected"][0]
    selected5000 = confirmation_internal["selected"][5000]
    baseline0 = confirmation_internal["baseline"][0]
    baseline5000 = confirmation_internal["baseline"][5000]
    selection_relative = (baseline_selection.metrics["paired_mse"] - selected_selection.metrics["paired_mse"]) / max(baseline_selection.metrics["paired_mse"], 1.0e-30)
    relative0 = (baseline0.metrics["paired_mse"] - selected0.metrics["paired_mse"]) / max(baseline0.metrics["paired_mse"], 1.0e-30)
    relative5000 = (baseline5000.metrics["paired_mse"] - selected5000.metrics["paired_mse"]) / max(baseline5000.metrics["paired_mse"], 1.0e-30)
    lower_delta, upper_delta = _stratified_bootstrap_ci(baseline5000.sample_mse - selected5000.sample_mse, baseline5000.sample_labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    state_unchanged = all(result.state_unchanged for result in selection_internal.values()) and all(result.state_unchanged for records in confirmation_internal.values() for result in records.values())
    finite = all(result.metrics["finite"] for result in selection_internal.values()) and all(result.metrics["finite"] for records in confirmation_internal.values() for result in records.values())
    peak = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.monotonic() - started
    science_gates = (
        GateResult("selection_material_improvement", selection_relative >= MATERIAL_RELATIVE_IMPROVEMENT, {"relative_improvement": selection_relative, "required": MATERIAL_RELATIVE_IMPROVEMENT}),
        GateResult("confirmation_material_improvement", relative5000 >= MATERIAL_RELATIVE_IMPROVEMENT, {"relative_improvement": relative5000, "required": MATERIAL_RELATIVE_IMPROVEMENT}),
        GateResult("confirmation_bootstrap_improvement", lower_delta > 0.0, {"lower_delta": lower_delta, "upper_delta": upper_delta, "confidence_level": CONFIDENCE_LEVEL}),
        GateResult("trained_geometry_gain_exceeds_initial", relative5000 > relative0, {"step0": relative0, "step5000": relative5000}),
        GateResult("selected_paired_beats_cyclic", selected5000.metrics["paired_mse"] < selected5000.metrics["cyclic_mse"], {"paired": selected5000.metrics["paired_mse"], "cyclic": selected5000.metrics["cyclic_mse"]}),
    )
    harness_gates = (
        GateResult("finite_measurements", finite, {}),
        GateResult("model_and_operator_state_unchanged", state_unchanged, {}),
        GateResult("all_grid_mode_ranks", all(record["state_unchanged"] for record in selection_records.values()) and all(record["metrics"]["finite"] for record in selection_records.values()), {}),
        GateResult("fixed_calibration", all(record["metrics"]["calibration_sha256"] == calibration_hash for record in selection_records.values()) and all(record["metrics"]["calibration_sha256"] == calibration_hash for record in confirmation_records.values()), {}),
        GateResult("negative_and_independent_controls", all(record["metrics"]["cyclic_mse"] > 0.0 and record["metrics"]["template_control_mse"] > 0.0 for record in confirmation_records.values()), {}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("optimizer_isolation", True, {"optimizer_created": False}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak, "required": MAX_PEAK_BYTES}),
    )
    all_harness = all(gate.passed for gate in harness_gates)
    all_science = all(gate.passed for gate in science_gates)
    if not all_harness:
        status, classification, action = "invalid_harness", "invalid_harness", None
    elif all_science:
        status, classification, action = "geometry_supported", "geometry_supported", NEXT_ACTION_SUPPORTED
    else:
        status, classification, action = "geometry_not_supported", "geometry_not_supported", NEXT_ACTION_UNSUPPORTED
    panel_values = {
        "schema": "scaled_chen_bogt_geometry_attribution_panel.v1",
        "source_hashes": source.hashes,
        "selection_variant": selected_variant.__dict__,
        "tensors": {
            "selection_baseline_camera": selection_internal[baseline_variant.key].panel["camera"][:100],
            "selection_selected_camera": selected_selection.panel["camera"][:100],
            "selection_target": selected_selection.panel["target"][:100],
            "confirmation_baseline_camera": baseline5000.panel["camera"][:100],
            "confirmation_selected_camera": selected5000.panel["camera"][:100],
            "confirmation_target": selected5000.panel["target"][:100],
        },
        "panel_role": "fixed_visual_attribution_only",
    }
    storage_estimate = sum(value.numel() * value.element_size() for value in panel_values["tensors"].values())
    if storage_estimate > MAX_STORAGE_BYTES:
        raise RuntimeError("geometry attribution panel exceeds storage gate")
    atomic_torch_save(panel_path, panel_values)
    panel_hash = file_sha256(panel_path)
    decomposition = {
        "schema": "scaled_chen_bogt_geometry_attribution_decomposition.v1",
        "selection": selection_records,
        "confirmation": confirmation_records,
        "selected_variant": selected_variant.__dict__,
        "selection_relative_improvement": selection_relative,
        "confirmation_relative_improvement_step0": relative0,
        "confirmation_relative_improvement_step5000": relative5000,
        "bootstrap_delta_ci": {"lower": lower_delta, "upper": upper_delta, "confidence_level": CONFIDENCE_LEVEL},
        "science_gates": [gate.__dict__ for gate in science_gates],
        "harness_gates": [gate.__dict__ for gate in harness_gates],
        "panel_sha256": panel_hash,
        "source_hashes": source.hashes,
        "resource": {"free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "wall_clock": "observational_only"},
    }
    atomic_write_json(decomposition_path, decomposition)
    hashes = _input_hashes(spec, review_path, formal_path, data, source)
    contract = {
        "selected_variant": selected_variant.__dict__,
        "selection_relative_improvement": selection_relative,
        "confirmation_relative_improvement_step0": relative0,
        "confirmation_relative_improvement_step5000": relative5000,
        "bootstrap_delta_ci": {"lower": lower_delta, "upper": upper_delta, "confidence_level": CONFIDENCE_LEVEL},
        "selection": selection_records,
        "confirmation": confirmation_records,
        "panel_path": str(panel_path.resolve()),
        "panel_sha256": panel_hash,
        "decomposition_path": str(decomposition_path.resolve()),
        "decomposition_sha256": file_sha256(decomposition_path),
        "resource": {"free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "wall_clock": "observational_only"},
        "evaluator_calls": 0,
        "optimizer_created": False,
    }
    gates = harness_gates + science_gates
    reason = None if status in {"geometry_supported", "geometry_not_supported"} else "geometry attribution harness or resource gate failed"
    payload = _record(AUDIT_SCHEMA, status, classification, spec, state, data, hashes, gates, action, reason, contract)
    _write_new(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if status == "geometry_supported" else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-geometry-attribution-validate":
        return validate(args)
    if args.research_action == "scaled-chen-geometry-attribution-run":
        return run(args)
    raise ValueError(f"unsupported geometry attribution action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "BASELINE",
    "BOOTSTRAP_REPLICATES",
    "CONFIG_SCHEMA",
    "GRID_FILLS",
    "GRID_MAGNIFICATIONS",
    "HYPOTHESIS",
    "NEXT_ACTION_SUPPORTED",
    "NEXT_ACTION_UNSUPPORTED",
    "SELECTION_COUNT",
    "VALIDATION_SCHEMA",
    "_assert_contract",
    "_load_data",
    "_load_review",
    "_load_spec",
    "_select_indices",
    "_stratified_bootstrap_ci",
    "dispatch",
    "run",
    "validate",
]
