"""Run the LightGen-topology 4L linear/ReLU factorial comparison.

The experiment reuses the already-valid 2L A/B observations and adds only the
new C/D arms.  C is a four-layer phase-only generator with an identity facet;
D uses the same four-layer generator and the frozen ideal intensity-ReLU
facet.  The four generator layers are separated by the 3 mm inter-plane
spacing disclosed in Supplementary Figure S17 of the LightGen paper.  The
existing 2L encoder, project sampling, OLS and backend relay remain frozen.

This is an ideal algorithmic surrogate.  It is not a Te material model,
material validation, hardware-readiness result, or exact reproduction of the
paper's 400/600-pixel metasurface implementation.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import BACK_CLEAR_APERTURE_M, BACK_FOCAL_LENGTH_M, BACK_MAGNIFICATION, BACK_SEGMENTS_M, ComplexField, IntensityFrame, LightGen2DAlignmentRuntimeConfig, LightGen2DForwardResult, LightGen2DSemanticGenerationInputOLSAlignment, OperationProvenance, OLS_SHAPE, WAVELENGTH_M
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from ldm_project.v2.optics.relays import CoherentImagingRelay


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load research authority: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The completed nominal and strict executors are code/evidence authorities
# only.  New C/D observations are written by this consumer into a new run
# namespace.
ENGINE = _load_module(
    Path(__file__).with_name("execute_lightgen_2d_ideal_facet_relu_long_v1.py"),
    "lightgen_4l_nominal_code_authority",
)
STRICT = _load_module(
    Path(__file__).with_name("execute_lightgen_2d_strict_train_calibrated_relu90_gain_v2.py"),
    "lightgen_4l_strict_evidence_authority",
)
N6 = ENGINE.N6
N5 = ENGINE.N5
N4 = ENGINE.N4
V10 = ENGINE.V10
V11 = ENGINE.V11


TASK_ID = "LIGHTGEN_ALIGNED_4L_RELU_GAIN_V1"
DESIGN_AUTHORITY = "LIGHTGEN_ALIGNED_4L_RELU_GAIN_V1_CONTRACT"
OUTER_GATE = "LIGHTGEN_ALIGNED_4L_RELU_GAIN_V1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

RUN_ROOT = Path("outputs/20260915_lightgen_aligned_4l_relu_gain_v1/runs")
CONTRACT_DIR = RUN_ROOT / "lightgen_aligned_4l_relu_gain_v1_contract_attempt_02"
CONTRACT_PATH = CONTRACT_DIR / "lightgen_aligned_4l_relu_gain_v1_contract.v1.json"
PREFLIGHT_DIR = RUN_ROOT / "lightgen_aligned_4l_relu_gain_v1_preflight_attempt_05"
PREFLIGHT_REPORT = PREFLIGHT_DIR / "lightgen_aligned_4l_relu_gain_v1_preflight_report.v1.json"
PREFLIGHT_LINEAGE = PREFLIGHT_DIR / "lightgen_aligned_4l_relu_gain_v1_preflight_lineage.v1.json"
C_DIR = RUN_ROOT / "lightgen_aligned_4l_relu_gain_v1_c_linear_attempt_01"
D_DIR = RUN_ROOT / "lightgen_aligned_4l_relu_gain_v1_d_relu90_attempt_01"
AGGREGATE_DIR = RUN_ROOT / "lightgen_aligned_4l_relu_gain_v1_aggregate_attempt_01"

STRICT_AGGREGATE_PATH = Path(
    "outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/"
    "strict_train_calibrated_relu90_gain_v2_aggregate_attempt_03/"
    "strict_train_calibrated_relu90_gain_v2_aggregate_report.v2.json"
)
STRICT_AGGREGATE_SHA256 = "def0a60c7176b82bbb776b4c6989da8185b744c5d4504b1eec1995b9d4344ca7"
STRICT_PREFLIGHT_REPORT_SHA256 = "14ae74b7f7a0dedc7da8261b4bb475d758b46fc3202a78fa4d883d0c00dc71de"
STRICT_REUSE_AUDIT_SHA256 = "f6826f80e385e0305005b98b4ca0cd6c95c9584de6c9737cbc01fb05dc78778a"
STRICT_CALIBRATION_PATH = Path(
    "outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/"
    "strict_train_calibrated_relu90_gain_v2_train_calibration_attempt_03/"
    "strict_train_calibrated_relu90_gain_v2_train_calibration.v2.json"
)
STRICT_CALIBRATION_SHA256 = "6becf17a93d5c4617de9c33aef6ed9a2dd5474062b1f2afe21f32f6f4c903c1f"
PAPER_ROOT = Path("research_assets/lightgen/science_adv7434")
PAPER_MANIFEST = PAPER_ROOT / "manifest.json"
PAPER_PDF = PAPER_ROOT / "paper/sciencesm.pdf"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
ADAPTER_RANK = 9
LOGICAL_BATCH_SIZE = 10
MICRO_BATCH_SIZE = 2
PRODUCTION_GRID = ENGINE.PRODUCTION_GRID
K = 2000
STAGE_SWITCH = 1600
BLOCK_SIZE = 200
PROGRESS_EVAL_INTERVAL_ROWS = 500
SAMPLER_SEED = 1234
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
GENERATOR_LAYERS = 4
GENERATOR_LAYER_SPACING_M = 3.0e-3
ENCODER_LAYERS = 2
ENCODER_LAYER_SPACING_M = 1.50e-3
CALIBRATION_TARGET = 0.90
CALIBRATION_INTERVAL = (0.89, 0.91)
MATERIAL_EPSILON = ENGINE.MATERIAL_EPSILON
MATERIAL_GAIN = 1.0
MATERIAL_SLOPE = 1.0
LAMBDA_TRANSFER = 28.48464149920509
LAMBDA_IMAGE = 13.353282014834436
INITIAL_LR = 1.0e-3
MID_LR = 2.0e-4
FINAL_LR = 5.0e-5
ACTIVE_GPU_BUDGET_SECONDS = 6.0 * 3600.0
STAGE_A_FIREWALL = 0.937
MATERIAL_DELTA = 0.05
RECONSTRUCTION_MAX_DELTA = 0.005
ARCHITECTURE_EQUIVALENCE_TOLERANCE = 0.01

FACET_TAU_STRICT_V2 = 0.10682560942718147
STRICT_SUFFICIENT_SCHEMA = "strict_train_calibrated_relu90_gain_v2_sufficient_observations.v2"
CHECKPOINT_SCHEMA = "lightgen_aligned_4l_relu_gain_v1_checkpoint.v1"
SUFFICIENT_SCHEMA = "lightgen_aligned_4l_relu_gain_v1_sufficient_observations.v1"
DIAGNOSTIC_SCHEMA = "lightgen_aligned_4l_relu_gain_v1_diagnostics.v1"
RAW_SCHEMA = "lightgen_aligned_4l_relu_gain_v1_cyclic_raw_outputs.v1"
PROGRESS_SCHEMA = "lightgen_aligned_4l_relu_gain_v1_progress.v1"


class InvalidFourLayerRun(RuntimeError):
    """Invalid input, implementation, or lineage boundary."""


class HardBlockedFourLayerRun(RuntimeError):
    """A valid boundary blocks the authorized scientific execution."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidFourLayerRun(message)


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
    _require(_sha256(path) == expected.lower(), f"{name} SHA-256 mismatch")


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


def _configure_engine() -> None:
    ENGINE.K_MAX = K
    ENGINE.K_MIN = K
    ENGINE.BLOCK_SIZE = BLOCK_SIZE
    ENGINE.INITIAL_LR = INITIAL_LR
    ENGINE.MID_LR = MID_LR
    ENGINE.FINAL_LR = FINAL_LR


class LightGenTopologyAligned4LRuntime(LightGen2DSemanticGenerationInputOLSAlignment):
    """Project runtime with a real four-layer, 3 mm generator stack."""

    generator_layers = GENERATOR_LAYERS
    generator_spacing_m = GENERATOR_LAYER_SPACING_M

    def __init__(self, config: LightGen2DAlignmentRuntimeConfig) -> None:
        super().__init__(config)
        self.generator = FiniteCropPhaseStack(
            layers=GENERATOR_LAYERS,
            active_shape=OLS_SHAPE,
            propagation_shape=(config.propagation_grid_size,) * 2,
            spacing_m=GENERATOR_LAYER_SPACING_M,
        )
        generator_exit_z = self.generator_entrance_plane.z_m + GENERATOR_LAYERS * GENERATOR_LAYER_SPACING_M
        self.generator_exit_plane = self.generator_entrance_plane.with_plane(
            "lightgen_generator_exit_4l",
            z_m=generator_exit_z,
        )
        self.camera_plane = self.camera_plane.with_plane(
            "lightgen_raw_cmos_4l",
            z_m=generator_exit_z + sum(BACK_SEGMENTS_M),
        )
        self.output_relay = CoherentImagingRelay(
            self.generator_exit_plane,
            self.camera_plane,
            BACK_MAGNIFICATION,
            name="lightgen_output_b0_4l",
            segments_m=BACK_SEGMENTS_M,
            focal_lengths_m=(BACK_FOCAL_LENGTH_M,),
            clear_aperture_m=BACK_CLEAR_APERTURE_M,
        )

    def _run_generator(
        self,
        state: Any,
        *,
        image_input: Any,
        encoder_output: ComplexField | None,
        ols_input: ComplexField | None,
        encoder_records: tuple[ComplexField, ...],
    ) -> LightGen2DForwardResult:
        generator_input = self.w_ols(state)
        generator_output, generator_records, generator_stats = self.generator(
            generator_input,
            "lightgen_generator_4l",
        )
        generator_output = ComplexField(
            generator_output.tensor,
            self.generator_exit_plane,
            OperationProvenance(
                "lightgen_generator_exit_4l",
                generator_output.plane.plane_id,
                None,
                {
                    "generator_layers": GENERATOR_LAYERS,
                    "inter_layer_spacing_m": GENERATOR_LAYER_SPACING_M,
                    "real_propagation_separated_layers": True,
                },
            ),
        )
        camera_field = self.output_relay(generator_output)
        self._validate_relay_diagnostics(self.output_relay)
        _require(torch.isfinite(camera_field.tensor).all(), "4L raw CMOS field is nonfinite")
        raw_camera = IntensityFrame(
            camera_field.tensor.abs().square(),
            self.camera_plane,
            camera_field.provenance,
        )
        return LightGen2DForwardResult(
            raw_camera=raw_camera,
            z_ols=state,
            generator_input_field=generator_input,
            generator_output_field=generator_output,
            image_input=image_input,
            encoder_output_field=encoder_output,
            ols_input_field=ols_input,
            encoder_records=encoder_records,
            generator_records=generator_records,
            crop_statistics={
                "generator_crop_retention": generator_stats["crop_retention"],
                "generator_phase": generator_stats["phase"],
            },
        )


class FullFacetProbe(ENGINE.LongFacetProbe):
    """Capture complete facet intensities for the TRAIN-only threshold check."""

    def __init__(self, *, mode: str, tau: float) -> None:
        super().__init__(mode=mode, tau=tau, capture=False)
        self.full_intensities: list[torch.Tensor] = []

    def clear(self) -> None:
        super().clear()
        self.full_intensities.clear()

    def apply(self, field: Any) -> Any:
        intensity = field.tensor.abs().square().detach().cpu().to(torch.float32)
        _finite(intensity, "4L calibration facet intensity")
        _require(tuple(intensity.shape[-2:]) == OLS_SHAPE, "4L facet active shape changed")
        self.full_intensities.append(intensity)
        return super().apply(field)


def _install_probe(consumer: Any, *, mode: str, tau: float, full: bool = False) -> Any:
    generator = consumer.runtime.generator
    original_forward = generator.forward
    probe = FullFacetProbe(mode=mode, tau=tau) if full else ENGINE.LongFacetProbe(mode=mode, tau=tau, capture=False)

    def forward_with_facet(field: Any, prefix: str) -> Any:
        return original_forward(probe.apply(field), prefix)

    generator.forward = forward_with_facet
    return probe


def _paper_audit(root: Path) -> dict[str, Any]:
    manifest_path = _path(root, PAPER_MANIFEST)
    manifest = _read_json(manifest_path, "LightGen paper manifest")
    entries = manifest.get("files")
    _require(isinstance(entries, list) and entries, "LightGen paper manifest file list is missing")
    checked: list[dict[str, Any]] = []
    for entry in entries:
        item = _mapping(entry, "LightGen paper manifest entry")
        rel = item.get("path")
        expected_sha = item.get("sha256")
        expected_bytes = item.get("bytes")
        _require(isinstance(rel, str) and isinstance(expected_sha, str), "LightGen paper manifest entry is incomplete")
        path = _path(root, PAPER_ROOT / rel)
        _validate_sha(path, expected_sha, f"LightGen paper asset {rel}")
        if expected_bytes is not None:
            _require(path.stat().st_size == int(expected_bytes), f"LightGen paper asset byte count changed: {rel}")
        checked.append({"path": _relative(root, path), "sha256": expected_sha, "bytes": path.stat().st_size})
    _validate_sha(_path(root, PAPER_PDF), "1325cdfb438f55249224f660a90ec2c21438ba3ce8cffc2677001f2db4e6c211", "LightGen supplementary PDF")
    return {
        "manifest": {"path": _relative(root, manifest_path), "sha256": _sha256(manifest_path)},
        "files": checked,
        "source_facts": {
            "paper_identity": "Chen et al., LightGen, Science, DOI 10.1126/science.adv7434",
            "wavelength_m": 532.0e-9,
            "encoder_layers": 2,
            "generator_layers": 4,
            "inter_plane_spacing_m": 3.0e-3,
            "pixel_pitch_m": 3.0e-6,
            "layer_pixel_options": [[400, 400], [600, 600]],
            "generator_footprint_mm": [11.90, 2.70],
            "encoder_footprint_mm": [6.20, 2.70],
            "detector_distance_mm": 3.0,
            "support": [
                "Supplementary Figure S17, page 30: 3 mm between adjacent optical-plane centerlines",
                "Supplementary Text 1, page 10: 2 encoder and 4 generator layers",
                "Supplementary Tables S3/S4, pages 38-39: layer counts and footprints",
            ],
        },
        "interpretation": "paper-disclosed mechanical spacing; not an exact replication of project sampling or relay details",
    }


def _strict_authority_audit(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    aggregate_path = _path(root, STRICT_AGGREGATE_PATH)
    _validate_sha(aggregate_path, STRICT_AGGREGATE_SHA256, "strict 2L aggregate report")
    aggregate = _read_json(aggregate_path, "strict 2L aggregate report")
    _require(aggregate.get("result") == "VALID_COMPLETED" and aggregate.get("status") == "valid_completed", "strict 2L aggregate is not valid")
    _require(aggregate.get("outer_gate") == "STRICT_TRAIN_CALIBRATED_RELU90_GAIN_V2", "strict 2L aggregate gate changed")
    _require(set(_mapping(aggregate.get("candidates"), "strict 2L candidates")) == {"A_LINEAR_REUSED", "B_RELU90"}, "strict 2L candidate set changed")
    _require(aggregate.get("primary", {}).get("supported") is False, "strict 2L result unexpectedly changed to supported")
    preflight = _mapping(aggregate.get("preflight"), "strict 2L preflight")
    preflight_report = _path(root, _mapping(preflight.get("report"), "strict 2L preflight report ref")["path"])
    _validate_sha(preflight_report, STRICT_PREFLIGHT_REPORT_SHA256, "strict 2L preflight report")
    reuse_ref = _mapping(aggregate.get("reuse_audit"), "strict 2L A reuse ref")
    reuse_path = _path(root, reuse_ref["path"])
    _validate_sha(reuse_path, STRICT_REUSE_AUDIT_SHA256, "strict 2L A reuse audit")
    _require(_read_json(reuse_path, "strict 2L A reuse audit").get("result") == "PASS", "strict 2L A reuse audit is not PASS")

    # Reconstruct the exact A endpoint/checkpoint audit through the previous
    # consumer.  This checks the source checkpoint, all endpoint tensors, and
    # the old A report without modifying the existing strict namespace.
    reuse = STRICT._audit_previous_a(root, source, write=False)
    _require(reuse["ref"]["sha256"] == STRICT_REUSE_AUDIT_SHA256, "strict 2L A reuse audit digest changed")

    candidates = _mapping(aggregate["candidates"], "strict 2L candidates")
    b_report_ref = _mapping(candidates["B_RELU90"]["report"], "strict 2L B report ref")
    b_report_path = _path(root, b_report_ref["path"])
    _validate_sha(b_report_path, str(b_report_ref["sha256"]), "strict 2L B report")
    b_report = _read_json(b_report_path, "strict 2L B report")
    _require(b_report.get("result") == "VALID_COMPLETED" and b_report.get("candidate") == "B_RELU90", "strict 2L B report is not valid")
    b_endpoints: dict[str, dict[str, Any]] = {}
    for step in (0, K):
        sufficient_ref = _mapping(b_report.get("sufficient_observations", {}).get(str(step)), f"strict 2L B sufficient {step} ref")
        sufficient_path = _path(root, sufficient_ref["path"])
        _validate_sha(sufficient_path, str(sufficient_ref["sha256"]), f"strict 2L B sufficient {step}")
        sufficient = _mapping(N4.safe_torch_load(sufficient_path, expected_sha256=str(sufficient_ref["sha256"]), map_location="cpu"), f"strict 2L B sufficient {step}")
        _require(sufficient.get("schema") == STRICT_SUFFICIENT_SCHEMA and sufficient.get("candidate") == "B_RELU90", f"strict 2L B sufficient {step} identity changed")
        _require(torch.is_tensor(sufficient.get("dev_smooth_correct")) and tuple(sufficient["dev_smooth_correct"].shape) == (5_000,), f"strict 2L B sufficient {step} DEV shape changed")
        b_endpoints[str(step)] = {"sufficient": sufficient, "sufficient_ref": dict(sufficient_ref)}
    return {
        "aggregate": aggregate,
        "aggregate_ref": {"path": _relative(root, aggregate_path), "sha256": STRICT_AGGREGATE_SHA256},
        "reuse": reuse,
        "b_report": b_report,
        "b_report_ref": {"path": _relative(root, b_report_path), "sha256": str(b_report_ref["sha256"])},
        "b_endpoints": b_endpoints,
        "preflight_report_ref": {"path": _relative(root, preflight_report), "sha256": STRICT_PREFLIGHT_REPORT_SHA256},
        "reuse_ref": {"path": _relative(root, reuse_path), "sha256": STRICT_REUSE_AUDIT_SHA256},
    }


def _lift_generator_state(source_state: Mapping[str, Any]) -> dict[str, Any]:
    state = dict(source_state)
    phase = state.get("generator.phase")
    _require(torch.is_tensor(phase) and tuple(phase.shape) == (2, 1, 100, 100), "common source generator phase shape changed")
    lifted = torch.zeros((GENERATOR_LAYERS, 1, 100, 100), dtype=phase.dtype)
    lifted[:2].copy_(phase.detach().cpu())
    state["generator.phase"] = lifted
    return state


def _lift_optimizer_state(source_state: Mapping[str, Any]) -> dict[str, Any]:
    """Lift 2L Adam moments to 4L, zero-initializing the new planes."""
    state = _cpu_tree(source_state)
    groups = state.get("param_groups")
    opt_state = state.get("state")
    _require(isinstance(groups, list) and len(groups) == 1 and isinstance(opt_state, Mapping), "common source optimizer state is incomplete")
    _require(list(groups[0].get("params", [])) == [0, 1, 2, 3, 4], "common source optimizer parameter order changed")
    _require(set(opt_state) == {0, 1, 2, 3, 4}, "common source optimizer state keys changed")
    lifted = _cpu_tree(state)
    lifted_gen = lifted["state"][4]
    source_gen = opt_state[4]
    for key, value in source_gen.items():
        if torch.is_tensor(value) and tuple(value.shape) == (2, 1, 100, 100):
            target = torch.zeros((GENERATOR_LAYERS, 1, 100, 100), dtype=value.dtype)
            target[:2].copy_(value)
            lifted_gen[key] = target
        else:
            lifted_gen[key] = _cpu_tree(value)
    return lifted


def _new_consumer(
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    device: torch.device,
    *,
    mode: str,
    tau: float,
    trainable: bool,
    full_probe: bool = False,
    runtime_state: Mapping[str, Any] | None = None,
    adapter_state: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    _require(mode in {"linear", "relu"}, f"unknown 4L facet mode: {mode}")
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    if runtime_state is None:
        source_runtime = source["checkpoint"]["runtime_state"]
        lifted_state = _lift_generator_state(source_runtime)
        source_adapter = source["checkpoint"]["adapter_state"]
    else:
        lifted_state = _cpu_tree(runtime_state)
        source_adapter = adapter_state
        _require(source_adapter is not None, "4L checkpoint adapter state is missing")
    runtime = LightGenTopologyAligned4LRuntime(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=PRODUCTION_GRID)
    ).to(device)
    runtime.load_state_dict(lifted_state, strict=True)
    consumer.runtime = runtime
    adapter = V11.LowRankAdapter(device=device)
    adapter.load_state_dict(_cpu_tree(source_adapter), strict=True)
    _configure_candidate(consumer, adapter, trainable=trainable)
    probe = _install_probe(consumer, mode=mode, tau=tau, full=full_probe)
    return consumer, adapter, probe


def _candidate_named(consumer: Any, adapter: Any) -> dict[str, nn.Parameter]:
    runtime_named = dict(consumer.runtime.named_parameters())
    _require(list(runtime_named) == ["encoder.phase", "generator.phase"], "4L runtime parameter surface changed")
    named = {f"adapter.{name}": parameter for name, parameter in adapter.named_parameters()}
    _require("generator.phase" in runtime_named, "4L generator phase is missing")
    named["generator.phase"] = runtime_named["generator.phase"]
    expected = ["adapter.u_real", "adapter.u_imag", "adapter.v_real", "adapter.v_imag", "generator.phase"]
    _require(list(named) == expected, "4L trainable parameter order changed")
    return named


def _configure_candidate(consumer: Any, adapter: Any, *, trainable: bool) -> dict[str, Any]:
    runtime_named = dict(consumer.runtime.named_parameters())
    _require(list(runtime_named) == ["encoder.phase", "generator.phase"], "4L runtime parameter names changed")
    for parameter in runtime_named.values():
        parameter.requires_grad_(False)
        parameter.grad = None
    named = _candidate_named(consumer, adapter)
    for name, parameter in named.items():
        parameter.requires_grad_(bool(trainable))
        parameter.grad = None
    _require(all(parameter.requires_grad is bool(trainable) for parameter in named.values()), "4L trainable firewall failed")
    _require(not runtime_named["encoder.phase"].requires_grad, "4L encoder is not frozen")
    if trainable:
        consumer.runtime.train()
    else:
        consumer.runtime.eval()
    return {
        "trainable": list(named) if trainable else [],
        "runtime_parameter_names": list(runtime_named),
        "frozen_runtime": [name for name in runtime_named if name != "generator.phase"],
        "parameter_shapes": {name: list(parameter.shape) for name, parameter in named.items()},
        "parameter_count": int(sum(parameter.numel() for parameter in named.values())),
        "encoder_updated": False,
        "ols_updated": False,
        "w_ols_updated": False,
    }


def _optimizer(consumer: Any, adapter: Any) -> torch.optim.Adam:
    named = _candidate_named(consumer, adapter)
    parameters = list(named.values())
    _require(all(parameter.requires_grad for parameter in parameters), "4L optimizer received frozen parameters")
    optimizer = torch.optim.Adam(
        parameters,
        lr=INITIAL_LR,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    _require(list(optimizer.param_groups[0]["params"]) == parameters, "4L optimizer parameter order changed")
    return optimizer


def _restore_optimizer(optimizer: torch.optim.Adam, state: Mapping[str, Any]) -> None:
    optimizer.load_state_dict(_cpu_tree(state))
    _require(len(optimizer.state) == 5, "4L optimizer state restoration is incomplete")


def _schedule_lr(step: int) -> tuple[float, str]:
    if step <= STAGE_SWITCH:
        fraction = step / float(STAGE_SWITCH)
        return INITIAL_LR + fraction * (MID_LR - INITIAL_LR), "adapter_decoder_coadaptation"
    fraction = (step - STAGE_SWITCH) / float(K - STAGE_SWITCH)
    return MID_LR + fraction * (FINAL_LR - MID_LR), "decoder_only_refinement"


def _set_stage(consumer: Any, adapter: Any, *, coadapt: bool) -> str:
    named = _candidate_named(consumer, adapter)
    for name, parameter in named.items():
        parameter.requires_grad_(name.startswith("adapter.") and coadapt or name == "generator.phase")
        parameter.grad = None
    consumer.runtime.train()
    return "adapter_decoder_coadaptation" if coadapt else "decoder_only_refinement"


def _train_step(
    consumer: Any,
    adapter: Any,
    optimizer: torch.optim.Adam,
    z: torch.Tensor,
    batch: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    templates: torch.Tensor,
    *,
    step: int,
) -> dict[str, Any]:
    lr, stage = _schedule_lr(step)
    optimizer.param_groups[0]["lr"] = lr
    _set_stage(consumer, adapter, coadapt=stage == "adapter_decoder_coadaptation")
    named = _candidate_named(consumer, adapter)
    before = {name: parameter.detach().cpu().clone() for name, parameter in named.items()}
    before_runtime = {
        key: value.detach().cpu().clone()
        for key, value in consumer.runtime.state_dict().items()
        if torch.is_tensor(value)
    }
    optimizer.zero_grad(set_to_none=True)
    _require(batch.shape[0] == labels.shape[0] == z.shape[0], "4L logical batch shapes are inconsistent")
    _require(batch.shape[0] > 0, "4L logical batch is empty")
    metric_sums = {key: 0.0 for key in ("anchor_loss", "transfer_loss", "image_loss", "semantic_loss", "total_loss")}
    for start in range(0, int(batch.shape[0]), MICRO_BATCH_SIZE):
        stop = min(int(batch.shape[0]), start + MICRO_BATCH_SIZE)
        weight = float(stop - start) / float(batch.shape[0])
        components = ENGINE._components(
            consumer,
            adapter,
            z[start:stop],
            batch[start:stop],
            labels[start:stop],
            prototypes,
            templates,
            use_cache=False,
            require_grad=True,
        )
        micro_loss = components["total_loss"]
        _require(micro_loss.requires_grad and components["semantic_loss"].requires_grad and components["image_loss"].requires_grad, "4L objective path is detached")
        _finite(micro_loss, "4L microbatch total loss")
        for key in metric_sums:
            metric_sums[key] += float(components[key].detach().cpu()) * weight
        (micro_loss * weight).backward()
        del components, micro_loss
    gradient_norms: dict[str, float | None] = {}
    for name, parameter in named.items():
        if parameter.requires_grad:
            _require(parameter.grad is not None, f"4L {name} gradient is missing")
            gradient_norms[name] = ENGINE._norm(parameter.grad, f"4L {name} gradient")
        else:
            _require(parameter.grad is None, f"4L frozen {name} received a gradient")
            gradient_norms[name] = None
    _require(any(value is not None and value > 0.0 for value in gradient_norms.values()), "4L combined gradient is zero")
    _require(all(parameter.grad is None for name, parameter in consumer.runtime.named_parameters() if name != "generator.phase"), "4L frozen runtime received a gradient")
    optimizer.step()
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    updates = {name: float((parameter.detach().cpu() - before[name]).to(torch.float64).square().sum().sqrt()) for name, parameter in named.items()}
    for name, parameter in named.items():
        if parameter.requires_grad:
            _require(math.isfinite(updates[name]) and updates[name] > 0.0, f"4L trainable group did not update: {name}")
        else:
            _require(updates[name] == 0.0, f"4L frozen group updated: {name}")
    # Compare against the pre-step snapshots for the actual frozen keys.
    for key in ("encoder.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"):
        _require(torch.equal(consumer.runtime.state_dict()[key].detach().cpu(), before_runtime[key]), f"4L frozen state changed: {key}")
    return {
        "step": step,
        "global_step": 600 + step,
        "stage": stage,
        "learning_rate": lr,
        "L_anchor": metric_sums["anchor_loss"],
        "L_transfer": metric_sums["transfer_loss"],
        "L_image": metric_sums["image_loss"],
        "L_sem": metric_sums["semantic_loss"],
        "L_total": metric_sums["total_loss"],
        "gradient_norms": gradient_norms,
        "update_norms": updates,
        "trainable_parameter_names": [name for name, parameter in named.items() if parameter.requires_grad],
        "adapter_updated": stage == "adapter_decoder_coadaptation",
        "generator_updated": True,
        "encoder_updated": False,
        "ols_updated": False,
    }


def _source(root: Path) -> dict[str, Any]:
    return ENGINE._load_source(root)


def _authorities(root: Path) -> dict[str, Any]:
    return ENGINE._resolve_bundle(root)


def _placeholder_z(count: int) -> torch.Tensor:
    return torch.zeros((count, OLS_PORT_COUNT), dtype=torch.complex64)


def _strict_calibration(root: Path) -> dict[str, Any]:
    path = _path(root, STRICT_CALIBRATION_PATH)
    _validate_sha(path, STRICT_CALIBRATION_SHA256, "strict TRAIN calibration")
    payload = _read_json(path, "strict TRAIN calibration")
    _require(payload.get("result") == "PASS" and payload.get("status") == "valid_completed", "strict TRAIN calibration is not valid")
    _require(float(payload.get("tau_90")) == FACET_TAU_STRICT_V2, "strict-v2 tau authority changed")
    _require(tuple(payload.get("validation_interval", ())) == CALIBRATION_INTERVAL, "strict-v2 validation interval changed")
    return {"payload": payload, "ref": {"path": _relative(root, path), "sha256": STRICT_CALIBRATION_SHA256}}


def _capture_full_facet_intensity(
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    images: torch.Tensor,
    sample_ids: torch.Tensor,
    device: torch.device,
    *,
    tau: float,
) -> torch.Tensor:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, adapter, probe = _new_consumer(
        bundle,
        source,
        device,
        mode="relu",
        tau=tau,
        trainable=False,
        full_probe=True,
    )
    del adapter
    with torch.no_grad():
        for start in range(0, int(sample_ids.numel()), LOGICAL_BATCH_SIZE):
            ids = sample_ids[start : start + LOGICAL_BATCH_SIZE]
            consumer.runtime.forward_from_image(images.index_select(0, ids).to(device))
    expected = math.ceil(int(sample_ids.numel()) / LOGICAL_BATCH_SIZE)
    _require(len(probe.full_intensities) == expected, "4L calibration capture batch count changed")
    intensities = torch.cat(probe.full_intensities, dim=0).contiguous()
    _require(tuple(intensities.shape) == (int(sample_ids.numel()), 1, 100, 100), "4L calibration intensity shape changed")
    _finite(intensities, "4L captured facet intensity")
    del consumer, probe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return intensities


def _retention_summary(intensities: torch.Tensor, labels: torch.Tensor, tau: float) -> dict[str, Any]:
    values = intensities.to(torch.float64)
    flat = values.flatten(1)
    input_power = flat.sum(dim=1)
    output_intensity = torch.relu(values - tau)
    output_power = output_intensity.flatten(1).sum(dim=1)
    eta = output_power / input_power.clamp_min(MATERIAL_EPSILON)
    active = (values > tau).flatten(1).to(torch.float64).mean(dim=1)
    suppressed = (output_intensity <= MATERIAL_EPSILON).flatten(1).to(torch.float64).mean(dim=1)
    for name, value in (("input_power", input_power), ("output_power", output_power), ("eta", eta), ("active", active), ("suppressed", suppressed)):
        _finite(value, f"4L retention {name}")
    return {
        "sample_count": int(values.shape[0]),
        "eta": float(output_power.sum() / input_power.sum()),
        "eta_per_sample_mean": float(eta.mean()),
        "eta_per_sample_std": float(eta.std(unbiased=False)),
        "active_fraction": float(active.mean()),
        "suppressed_fraction": float(suppressed.mean()),
        "per_class_eta": [
            {
                "class": digit,
                "eta": float(eta[labels == digit].mean()),
                "active_fraction": float(active[labels == digit].mean()),
                "suppressed_fraction": float(suppressed[labels == digit].mean()),
            }
            for digit in range(CLASS_COUNT)
        ],
    }


def _tau_step0_check(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    calibration: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    payload = _mapping(calibration["payload"], "strict calibration payload")
    ids_value = payload.get("calibration_set", {}).get("ids")
    _require(isinstance(ids_value, list) and len(ids_value) == 1000, "strict calibration IDs are incomplete")
    ids = torch.tensor(ids_value, dtype=torch.long)
    calibration_labels = labels.index_select(0, ids)
    intensities = _capture_full_facet_intensity(
        bundle,
        source,
        images,
        ids,
        device,
        tau=FACET_TAU_STRICT_V2,
    )
    summary = _retention_summary(intensities, calibration_labels, FACET_TAU_STRICT_V2)
    _require(CALIBRATION_INTERVAL[0] <= summary["eta"] <= CALIBRATION_INTERVAL[1], "4L step-0 tau retention is outside the frozen [0.89,0.91] gate")
    return {
        "tau": FACET_TAU_STRICT_V2,
        "source_calibration_ids_sha256": V10.V6.tensor_sha256(ids),
        "sample_count": int(ids.numel()),
        "retention": summary,
        "validation_interval": list(CALIBRATION_INTERVAL),
        "status": "PASS",
    }


def _initial_state_facts(source: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(source["checkpoint"].get("runtime_state"), "common source runtime")
    adapter = _mapping(source["checkpoint"].get("adapter_state"), "common source adapter")
    _require(tuple(runtime["encoder.phase"].shape) == (2, 1, 100, 100), "common encoder phase shape changed")
    _require(tuple(runtime["generator.phase"].shape) == (2, 1, 100, 100), "common generator phase shape changed")
    _require(tuple(adapter["u_real"].shape) == (100, ADAPTER_RANK), "common adapter shape changed")
    lifted = _lift_generator_state(runtime)
    return {
        "source_checkpoint": source["checkpoint_ref"],
        "source_runtime_digest": _digest_tree(runtime),
        "source_adapter_digest": _digest_tree(adapter),
        "lifted_runtime_digest": _digest_tree(lifted),
        "lift_rule": "generator.phase[0:2] <- common 2L phase; generator.phase[2:4] <- zeros",
        "lift_is_function_preserving": False,
        "new_generator_layers_zero_initialized": bool(torch.equal(lifted["generator.phase"][2:], torch.zeros_like(lifted["generator.phase"][2:]))),
        "optimizer_lift_rule": "adapter moments copied; generator moments[0:2] copied; generator moments[2:4] zero",
    }


def _gradient_norm_map(parameters: tuple[nn.Parameter, ...], gradients: tuple[torch.Tensor, ...], names: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, parameter, gradient in zip(names, parameters, gradients, strict=True):
        _finite(gradient, f"4L negative-control gradient {name}")
        result[name] = float(gradient.detach().to(torch.float64).square().sum().sqrt().cpu())
    return result


def _negative_controls(
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
    *,
    mode: str,
    tau: float,
) -> dict[str, Any]:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, adapter, _probe = _new_consumer(bundle, source, device, mode=mode, tau=tau, trainable=True)
    indices = train_indices[:MICRO_BATCH_SIZE]
    batch = images.index_select(0, indices).to(device)
    batch_labels = labels.index_select(0, indices).to(device)
    components = ENGINE._components(
        consumer,
        adapter,
        _placeholder_z(MICRO_BATCH_SIZE).to(device),
        batch,
        batch_labels,
        bundle["authorities"]["prototypes"].to(device),
        bundle["authorities"]["templates"].to(device),
        use_cache=False,
        require_grad=True,
    )
    named = _candidate_named(consumer, adapter)
    parameters = tuple(named.values())
    names = list(named)
    wrong_labels = (batch_labels + 1) % CLASS_COUNT
    wrong_anchor = F.cross_entropy(components["anchor_logits"], wrong_labels)
    wrong_direction = F.softplus(components["delta"]).mean()
    anchor_grad = torch.autograd.grad(wrong_anchor, parameters, retain_graph=True, allow_unused=False)
    direction_grad = torch.autograd.grad(wrong_direction, parameters, allow_unused=False)
    anchor_norms = _gradient_norm_map(parameters, anchor_grad, names)
    direction_norms = _gradient_norm_map(parameters, direction_grad, names)
    _require(any(value > 0.0 for value in anchor_norms.values()), f"4L {mode} wrong-label negative control is zero")
    _require(any(value > 0.0 for value in direction_norms.values()), f"4L {mode} wrong-direction negative control is zero")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), f"4L {mode} negative control touched runtime gradients")
    del consumer, adapter, _probe, batch, components
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"wrong_label_anchor": anchor_norms, "wrong_direction_transfer": direction_norms, "runtime_gradient_none": True}


def _smoke_arm(
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
    *,
    mode: str,
    tau: float,
) -> dict[str, Any]:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, adapter, _probe = _new_consumer(bundle, source, device, mode=mode, tau=tau, trainable=True)
    optimizer = _optimizer(consumer, adapter)
    _restore_optimizer(optimizer, _lift_optimizer_state(source["checkpoint"]["optimizer_state"]))
    sampler = ENGINE.BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=K)
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    z = _placeholder_z(LOGICAL_BATCH_SIZE).to(device)
    traces: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, 4):
        indices, batch_labels = sampler.next_batch()
        batch = images.index_select(0, indices).to(device)
        traces.append(_train_step(consumer, adapter, optimizer, z, batch, batch_labels.to(device), prototypes, templates, step=step))
        del batch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    elapsed = time.perf_counter() - started
    _require(sampler.cursor == 3, f"4L {mode} smoke sampler cursor changed")
    _require(all(math.isfinite(float(trace["L_total"])) for trace in traces), f"4L {mode} smoke loss is nonfinite")
    result = {
        "mode": mode,
        "steps": 3,
        "scientific_updates": 0,
        "traces": traces,
        "sampler_cursor": sampler.cursor,
        "elapsed_seconds": elapsed,
        "seconds_per_update": elapsed / 3.0,
        "peak_memory_bytes": peak_memory,
        "trainable_surface": _configure_candidate(consumer, adapter, trainable=True),
    }
    del consumer, adapter, _probe, optimizer, sampler, z, prototypes, templates
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _stage2_smoke(
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
    *,
    mode: str,
    tau: float,
) -> dict[str, Any]:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, adapter, _probe = _new_consumer(bundle, source, device, mode=mode, tau=tau, trainable=True)
    optimizer = _optimizer(consumer, adapter)
    _restore_optimizer(optimizer, _lift_optimizer_state(source["checkpoint"]["optimizer_state"]))
    sampler = ENGINE.BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=K)
    indices, batch_labels = sampler.next_batch()
    batch = images.index_select(0, indices).to(device)
    before = {name: parameter.detach().cpu().clone() for name, parameter in _candidate_named(consumer, adapter).items()}
    trace = _train_step(
        consumer,
        adapter,
        optimizer,
        _placeholder_z(LOGICAL_BATCH_SIZE).to(device),
        batch,
        batch_labels.to(device),
        bundle["authorities"]["prototypes"].to(device),
        bundle["authorities"]["templates"].to(device),
        step=STAGE_SWITCH + 1,
    )
    after = {name: parameter.detach().cpu().clone() for name, parameter in _candidate_named(consumer, adapter).items()}
    adapter_updates = {
        name: float((after[name] - before[name]).to(torch.float64).square().sum().sqrt())
        for name in before
        if name.startswith("adapter.")
    }
    _require(all(value == 0.0 for value in adapter_updates.values()), f"4L {mode} stage-2 adapter updated")
    _require(float(trace["update_norms"]["generator.phase"]) > 0.0, f"4L {mode} stage-2 generator did not update")
    result = {
        "status": "PASS",
        "mode": mode,
        "step": STAGE_SWITCH + 1,
        "stage": trace["stage"],
        "adapter_update_norms": adapter_updates,
        "generator_update_norm": float(trace["update_norms"]["generator.phase"]),
        "scientific_updates": 0,
    }
    del consumer, adapter, _probe, optimizer, sampler, batch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _initialization_comparison(
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    device: torch.device,
    tau: float,
) -> dict[str, Any]:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    c_consumer, c_adapter, _c_probe = _new_consumer(bundle, source, device, mode="linear", tau=0.0, trainable=True)
    c_optimizer = _optimizer(c_consumer, c_adapter)
    _restore_optimizer(c_optimizer, _lift_optimizer_state(source["checkpoint"]["optimizer_state"]))
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    d_consumer, d_adapter, _d_probe = _new_consumer(bundle, source, device, mode="relu", tau=tau, trainable=True)
    d_optimizer = _optimizer(d_consumer, d_adapter)
    _restore_optimizer(d_optimizer, _lift_optimizer_state(source["checkpoint"]["optimizer_state"]))
    c_runtime = _cpu_tree(c_consumer.runtime.state_dict())
    d_runtime = _cpu_tree(d_consumer.runtime.state_dict())
    c_adapter = _cpu_tree(c_adapter.state_dict())
    d_adapter = _cpu_tree(d_adapter.state_dict())
    c_opt = _cpu_tree(c_optimizer.state_dict())
    d_opt = _cpu_tree(d_optimizer.state_dict())
    _require(_tree_equal(c_runtime, d_runtime), "C/D initial 4L runtime state differs")
    _require(_tree_equal(c_adapter, d_adapter), "C/D initial adapter state differs")
    _require(_tree_equal(c_opt, d_opt), "C/D initial optimizer state differs")
    result = {
        "runtime_digest": _digest_tree(c_runtime),
        "adapter_digest": _digest_tree(c_adapter),
        "optimizer_digest": _digest_tree(c_opt),
        "shared_runtime_exact": True,
        "shared_adapter_exact": True,
        "shared_optimizer_exact": True,
        "facet_is_only_arm_difference": True,
        "tau": tau,
    }
    del c_consumer, c_adapter, _c_probe, c_optimizer, d_consumer, d_adapter, _d_probe, d_optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _make_contract(
    root: Path,
    started: Mapping[str, Any],
    paper: Mapping[str, Any],
    source: Mapping[str, Any],
    source_facts: Mapping[str, Any],
    strict_audit: Mapping[str, Any],
    calibration: Mapping[str, Any],
    tau_check: Mapping[str, Any],
    initialization: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "lightgen_aligned_4l_relu_gain_v1_contract.v1",
        "version": 1,
        "task_id": TASK_ID,
        "design_authority": DESIGN_AUTHORITY,
        "status": "READY_FOR_BOUNDED_LONG_SCIENTIFIC_EXECUTION",
        "outer_gate": OUTER_GATE,
        "git_boundary": {"branch": started["branch"], "head": started["head"]},
        "paper_authority": paper,
        "geometry": {
            "variant": "LIGHTGEN_TOPOLOGY_ALIGNED_4L",
            "paper_disclosed": {
                "wavelength_m": WAVELENGTH_M,
                "encoder_layers": ENCODER_LAYERS,
                "generator_layers": GENERATOR_LAYERS,
                "inter_plane_spacing_m": GENERATOR_LAYER_SPACING_M,
                "pixel_pitch_m": 3.0e-6,
                "layer_pixel_options": [[400, 400], [600, 600]],
                "generator_footprint_mm": [11.90, 2.70],
                "encoder_footprint_mm": [6.20, 2.70],
                "detector_distance_mm": 3.0,
                "source": "Supplementary Figure S17 and Supplementary Text 1/Tables S3-S4",
            },
            "project_used": {
                "encoder_layers": ENCODER_LAYERS,
                "encoder_layer_spacing_m": ENCODER_LAYER_SPACING_M,
                "generator_layers": GENERATOR_LAYERS,
                "generator_layer_spacing_m": GENERATOR_LAYER_SPACING_M,
                "ols_grid": [10, 10],
                "ols_ports": OLS_PORT_COUNT,
                "ols_pixel_pitch_m": 2.0e-6,
                "propagation_grid": PRODUCTION_GRID,
                "wavelength_m": WAVELENGTH_M,
                "new_diffraction_plane": False,
            },
            "fallbacks": [
                "project 100x100 active OLS-plane sampling is retained; paper 400x400/600x600 layer pixels are not silently substituted",
                "project 2 um active pitch is retained; paper 3 um neuron pitch is recorded but not claimed",
                "project existing backend relay/CMOS representation is retained; paper's 3 mm lens-imaged detection geometry is not substituted",
                "existing 2L encoder phase and 1.5 mm encoder spacing remain frozen for A/B comparability",
            ],
            "real_propagation_separated_layers": True,
        },
        "source_common_start": {
            "authority": source["checkpoint_ref"],
            "facts": dict(source_facts),
            "strict_2l_aggregate": strict_audit["aggregate_ref"],
        },
        "reuse_audit": {
            "strict_2l_aggregate": strict_audit["aggregate_ref"],
            "strict_2l_preflight": strict_audit["preflight_report_ref"],
            "strict_2l_a_reuse": strict_audit["reuse_ref"],
            "decision": "A/B observations reused; no A/B retraining",
        },
        "arms": {
            "A": {"candidate": "A_LINEAR", "source": "REUSED_VALID_STRICT_2L_AUTHORITY"},
            "B": {"candidate": "B_RELU90", "source": "REUSED_VALID_STRICT_2L_AUTHORITY"},
            "C": {"candidate": "C_4L_LINEAR", "source": "NEW_4L_TRAINING", "facet": "identity"},
            "D": {"candidate": "D_4L_RELU90", "source": "NEW_4L_TRAINING", "facet": "strict_relu90"},
            "executed_new": ["C_4L_LINEAR", "D_4L_RELU90"],
            "primary": "D_4L_RELU90_vs_C_4L_LINEAR",
        },
        "facet": {
            "kind": "IDEAL_INTENSITY_RELU_SURROGATE",
            "formula": "E_out=E_in*sqrt((relu(|E_in|^2-tau)+1e-12)/(|E_in|^2+1e-12))",
            "tau_90": FACET_TAU_STRICT_V2,
            "gain": MATERIAL_GAIN,
            "slope": MATERIAL_SLOPE,
            "phase_shift": 0.0,
            "trainable": False,
            "additional_propagation": False,
            "placement": "after OLS coefficient transmission/modal re-emission and before the four-layer generator",
            "calibration_authority": calibration["ref"],
            "step0_retention_check": tau_check,
        },
        "objective": {
            "formula": "L=L_anchor+lambda_transfer*L_transfer+lambda_image*L_image",
            "lambda_transfer": LAMBDA_TRANSFER,
            "lambda_image": LAMBDA_IMAGE,
        },
        "training": {
            "K": K,
            "batch": LOGICAL_BATCH_SIZE,
            "micro_batch": MICRO_BATCH_SIZE,
            "stage_switch": STAGE_SWITCH,
            "stage_schedule": "first 80% adapter+decoder; last 20% decoder only",
            "optimizer": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "lr_start": INITIAL_LR,
            "lr_switch": MID_LR,
            "lr_end": FINAL_LR,
            "sampler": ENGINE.BalancedMultiEpochSampler.SCHEMA,
            "sampler_seed": SAMPLER_SEED,
            "block_size": BLOCK_SIZE,
            "evaluation_fractions": [0.0, 1.0],
            "C_D_initialization": initialization,
        },
        "preflight": {
            "benchmark": benchmark,
            "representative_real_update": True,
            "stage2_frozen_adapter_check": True,
            "negative_controls": "wrong-label anchor and wrong-direction transfer",
            "estimated_active_gpu_seconds": benchmark["estimated_active_gpu_seconds"],
            "budget_seconds": ACTIVE_GPU_BUDGET_SECONDS,
        },
        "factorial_estimands": {
            "delta_4L": "D-C",
            "delta_2L": "B-A",
            "interaction": "(D-C)-(B-A)",
            "architecture_linear": "C-A",
            "architecture_relu": "D-B",
            "approximation_tolerance": ARCHITECTURE_EQUIVALENCE_TOLERANCE,
        },
        "outer_gate_rules": {
            "delta_4L_min": MATERIAL_DELTA,
            "paired_bootstrap_ci_lower_min": 0.0,
            "image_delta_max": RECONSTRUCTION_MAX_DELTA,
            "cyclic_transfer_required": True,
            "raw_finite_nonconstant_structured_required": True,
        },
        "claim_boundary": {
            "ideal_simulation_only": True,
            "physical_te_validation": False,
            "formal_promotion": False,
            "stage_b": False,
            "sampled_generation": False,
            "protected": False,
            "official_test": False,
            "sealed": False,
        },
        "no_retry": True,
        "no_sweep": True,
    }


def _save_json(root: Path, path: Path, payload: Mapping[str, Any], artifact_kind: str) -> dict[str, str]:
    _require(not path.exists(), f"refusing to overwrite {artifact_kind}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    N4.atomic_write_new_json(path, dict(payload), artifact_kind=artifact_kind)
    return {"path": _relative(root, path), "sha256": _sha256(path)}


def _save_progress_event(
    root: Path,
    output_dir: Path,
    candidate: str,
    *,
    phase: str,
    status: str,
    step: int,
    rows_done: int | None = None,
    total_rows: int | None = None,
    elapsed_seconds: float | None = None,
    checkpoint_ref: Mapping[str, Any] | None = None,
    session: str | None = None,
    error: BaseException | None = None,
) -> dict[str, str]:
    """Persist an immutable progress event that survives interruption."""

    safe_phase = "".join(char if char.isalnum() else "_" for char in phase)
    token = f"{time.time_ns():019d}"
    path = output_dir / "progress" / f"{candidate.lower()}_{safe_phase}_{token}.v1.json"
    payload: dict[str, Any] = {
        "schema": PROGRESS_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "candidate": candidate,
        "phase": phase,
        "status": status,
        "step": int(step),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_done": int(rows_done) if rows_done is not None else None,
        "total_rows": int(total_rows) if total_rows is not None else None,
        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else None,
        "checkpoint": dict(checkpoint_ref) if checkpoint_ref is not None else None,
        "session": session,
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    return _save_json(root, path, payload, f"{candidate} progress event")


def _save_tensor(root: Path, path: Path, payload: Mapping[str, Any], artifact_kind: str) -> dict[str, str]:
    _require(not path.exists(), f"refusing to overwrite {artifact_kind}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    N4.atomic_torch_save_new(path, dict(payload), artifact_kind=artifact_kind)
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"{artifact_kind} reload")
    _require(loaded.get("schema") == payload.get("schema"), f"{artifact_kind} schema changed on reload")
    return {"path": _relative(root, path), "sha256": digest}


def _facet_observations(probe: Any, rows: int) -> dict[str, Any]:
    return ENGINE._facet_observations(probe, rows)


def _generator_diagnostics_4l(consumer: Any, initial_phase: torch.Tensor) -> dict[str, Any]:
    phase = consumer.runtime.state_dict()["generator.phase"].detach().cpu().to(torch.float64)
    initial = initial_phase.detach().cpu().to(torch.float64)
    _require(tuple(phase.shape) == (GENERATOR_LAYERS, 1, 100, 100), "4L generator diagnostic shape changed")
    _require(tuple(initial.shape) == tuple(phase.shape), "4L initial phase shape changed")
    delta = phase - initial
    layers = []
    for index in range(GENERATOR_LAYERS):
        layer_phase = phase[index]
        layer_delta = delta[index]
        layers.append(
            {
                "layer": index + 1,
                "phase_change_frobenius": float(torch.linalg.vector_norm(layer_delta).item()),
                "phase_change_max_abs": float(layer_delta.abs().max().item()),
                "phase_frobenius": float(torch.linalg.vector_norm(layer_phase).item()),
                "phase_max_abs": float(layer_phase.abs().max().item()),
                "phase_mean": float(layer_phase.mean().item()),
                "phase_std": float(layer_phase.std(unbiased=False).item()),
            }
        )
    return {
        "generator_layers": GENERATOR_LAYERS,
        "inter_layer_spacing_m": GENERATOR_LAYER_SPACING_M,
        "phase_change_frobenius": float(torch.linalg.vector_norm(delta).item()),
        "phase_change_max_abs": float(delta.abs().max().item()),
        "per_layer": layers,
        "real_propagation_separated_layers": True,
    }


def _train_metric(context: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, float]:
    sampler = ENGINE.BalancedMultiEpochSampler(
        context["train_indices"],
        context["labels"],
        seed=SAMPLER_SEED,
        steps=K,
    )
    indices, batch_labels = sampler.peek_batch()
    batch = context["images"].index_select(0, indices).to(context["device"])
    metric = ENGINE._train_metric(
        context["consumer"],
        context["adapter"],
        _placeholder_z(int(batch.shape[0])).to(context["device"]),
        batch,
        batch_labels.to(context["device"]),
        bundle["authorities"]["prototypes"].to(context["device"]),
        bundle["authorities"]["templates"].to(context["device"]),
        use_cache=False,
    )
    del batch
    return metric


def _endpoint(
    root: Path,
    context: Mapping[str, Any],
    bundle: Mapping[str, Any],
    dev_images: torch.Tensor,
    dev_labels: torch.Tensor,
    selection_images: torch.Tensor,
    selection_labels: torch.Tensor,
    selection_indices: torch.Tensor,
    *,
    candidate: str,
    facet_mode: str,
    step: int,
    train_metric: Mapping[str, Any],
) -> dict[str, Any]:
    consumer = context["consumer"]
    adapter = context["adapter"]
    probe = context["probe"]
    tau = float(context["tau"])
    contract_ref = context["contract_ref"]
    runtime_before = _digest_tree(consumer.runtime.state_dict())
    adapter_before = _digest_tree(adapter.state_dict())
    probe.clear()
    probe.capture = True
    progress_session = f"{time.time_ns():019d}"
    total_dev_rows = int(dev_images.shape[0])
    evaluation_started = time.perf_counter()
    _save_progress_event(
        root,
        context["output_dir"],
        candidate,
        phase="dev_evaluation",
        status="started",
        step=step,
        rows_done=0,
        total_rows=total_dev_rows,
        session=progress_session,
    )
    original_objective = N4._objective_components
    progress_rows = 0
    next_progress_rows = PROGRESS_EVAL_INTERVAL_ROWS

    def _progress_objective(*args: Any, **kwargs: Any) -> Any:
        nonlocal progress_rows, next_progress_rows
        result = original_objective(*args, **kwargs)
        batch = args[2] if len(args) > 2 else kwargs["batch"]
        progress_rows += int(batch.shape[0])
        if progress_rows >= next_progress_rows or progress_rows == total_dev_rows:
            _save_progress_event(
                root,
                context["output_dir"],
                candidate,
                phase="dev_evaluation",
                status="running",
                step=step,
                rows_done=progress_rows,
                total_rows=total_dev_rows,
                elapsed_seconds=time.perf_counter() - evaluation_started,
                session=progress_session,
            )
            while next_progress_rows <= progress_rows:
                next_progress_rows += PROGRESS_EVAL_INTERVAL_ROWS
        return result

    try:
        N4._objective_components = _progress_objective
        with torch.no_grad():
            evaluation = N4._evaluate(
                consumer,
                adapter,
                bundle["authorities"]["smooth"],
                dev_images,
                dev_labels,
                bundle["authorities"]["prototypes"],
                bundle["authorities"]["templates"],
                step=step,
                train_trace=None,
            )
    except BaseException as error:
        _save_progress_event(
            root,
            context["output_dir"],
            candidate,
            phase="dev_evaluation",
            status="failed",
            step=step,
            rows_done=progress_rows,
            total_rows=total_dev_rows,
            elapsed_seconds=time.perf_counter() - evaluation_started,
            session=progress_session,
            error=error,
        )
        raise
    finally:
        N4._objective_components = original_objective
    _save_progress_event(
        root,
        context["output_dir"],
        candidate,
        phase="dev_evaluation",
        status="completed",
        step=step,
        rows_done=total_dev_rows,
        total_rows=total_dev_rows,
        elapsed_seconds=time.perf_counter() - evaluation_started,
        session=progress_session,
    )

    cyclic_started = time.perf_counter()
    _save_progress_event(
        root,
        context["output_dir"],
        candidate,
        phase="cyclic_transfer",
        status="started",
        step=step,
        rows_done=0,
        total_rows=int(selection_images.shape[0]),
        session=progress_session,
    )
    try:
        with torch.no_grad():
            cyclic = V10._cyclic_transfer_v10(
                consumer,
                adapter,
                bundle["authorities"]["smooth"],
                selection_images,
                selection_labels,
                selection_indices,
                bundle["authorities"]["prototypes"],
                bundle["authorities"]["templates"],
            )
    except BaseException as error:
        _save_progress_event(
            root,
            context["output_dir"],
            candidate,
            phase="cyclic_transfer",
            status="failed",
            step=step,
            rows_done=0,
            total_rows=int(selection_images.shape[0]),
            elapsed_seconds=time.perf_counter() - cyclic_started,
            session=progress_session,
            error=error,
        )
        raise
    _save_progress_event(
        root,
        context["output_dir"],
        candidate,
        phase="cyclic_transfer",
        status="completed",
        step=step,
        rows_done=int(selection_images.shape[0]),
        total_rows=int(selection_images.shape[0]),
        elapsed_seconds=time.perf_counter() - cyclic_started,
        session=progress_session,
    )
    _require(runtime_before == _digest_tree(consumer.runtime.state_dict()), f"{candidate} evaluation changed runtime state at {step}")
    _require(adapter_before == _digest_tree(adapter.state_dict()), f"{candidate} evaluation changed adapter state at {step}")
    facet = _facet_observations(probe, int(dev_images.shape[0]))
    sufficient = N5._canonical_sufficient_payload(
        contract_ref,
        candidate,
        step,
        evaluation,
        cyclic,
        dev_labels,
        train_metric,
    )
    sufficient.update(
        {
            "schema": SUFFICIENT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "candidate": candidate,
            "global_step": 600 + step,
            "facet_observations": facet,
            "facet_material": {
                "kind": "IDEAL_INTENSITY_RELU_SURROGATE" if facet_mode == "relu" else "IDENTITY_LINEAR_FACET",
                "mode": facet_mode,
                "tau": tau,
                "gain": MATERIAL_GAIN,
                "slope": MATERIAL_SLOPE,
                "phase_shift": 0.0,
                "trainable": False,
            },
            "forward_mode": "4L real separated propagation; sufficient observations persisted before diagnostics",
            "progress_protocol": {
                "dev_evaluation_interval_rows": PROGRESS_EVAL_INTERVAL_ROWS,
                "immutable_events_dir": "progress/",
                "step0_bootstrap_checkpoint": True,
            },
            "generator_geometry": {
                "layers": GENERATOR_LAYERS,
                "inter_layer_spacing_m": GENERATOR_LAYER_SPACING_M,
                "new_diffraction_plane": False,
            },
        }
    )
    output_dir = context["output_dir"]
    prefix = candidate.lower()
    sufficient_path = output_dir / f"{prefix}_step_{step:05d}_sufficient_observations.v1.pt"
    sufficient_ref = _save_tensor(root, sufficient_path, sufficient, f"{candidate} sufficient observations step {step}")
    loaded = _mapping(
        N4.safe_torch_load(_path(root, sufficient_ref["path"]), expected_sha256=sufficient_ref["sha256"], map_location="cpu"),
        f"{candidate} sufficient observation reload {step}",
    )
    diagnostics = {
        "facet": ENGINE._facet_diagnostics(loaded["facet_observations"], mode=facet_mode, tau=tau),
        "adapter": N5._adapter_diagnostics(adapter),
        "generator": _generator_diagnostics_4l(consumer, context["initial_phase"]),
        "cyclic": N5._cyclic_diagnostics_from_observations(loaded),
        "direct_diversity": ENGINE._direct_diversity(cyclic["_observations"]["raw_base"], cyclic["_observations"]["labels"]),
    }
    diagnostic_path = output_dir / f"{prefix}_step_{step:05d}_diagnostics.v1.json"
    diagnostic_ref = _save_json(
        root,
        diagnostic_path,
        {
            "schema": DIAGNOSTIC_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "candidate": candidate,
            "step": step,
            "sufficient_observations": sufficient_ref,
            "diagnostics": diagnostics,
        },
        f"{candidate} endpoint diagnostics step {step}",
    )
    raw_ref = None
    if step == K:
        raw_path = output_dir / f"{prefix}_step_{step:05d}_cyclic_raw_outputs.v1.pt"
        raw_ref = _save_tensor(
            root,
            raw_path,
            {
                "schema": RAW_SCHEMA,
                "version": 1,
                "task_id": TASK_ID,
                "candidate": candidate,
                "contract": dict(contract_ref),
                "selection_indices": cyclic["_observations"]["selection_indices"],
                "labels": cyclic["_observations"]["labels"],
                "targets": cyclic["_observations"]["targets"],
                "raw_base": cyclic["_observations"]["raw_base"],
                "raw_shifted": cyclic["_observations"]["raw_shifted"],
            },
            f"{candidate} final cyclic raw outputs",
        )
    return {
        "step": step,
        "candidate": candidate,
        "facet_mode": facet_mode,
        "tau": tau,
        "global_step": 600 + step,
        "dev_labels": loaded["dev_labels"],
        "dev_correct": loaded["dev_smooth_correct"],
        "evaluation_public": loaded["evaluation_public"],
        "cyclic_public": loaded["cyclic_public"],
        "dev_image_loss": float(loaded["dev_image_loss"]),
        "train_metric": dict(train_metric),
        "sufficient_ref": sufficient_ref,
        "diagnostic_ref": diagnostic_ref,
        "facet_diagnostics": diagnostics["facet"],
        "diagnostics": diagnostics,
        "raw_ref": raw_ref,
    }


def _checkpoint_path(output_dir: Path, candidate: str, step: int) -> Path:
    return output_dir / f"{candidate.lower()}_step_{step:05d}_checkpoint.v1.pt"


def _checkpoint_payload(
    root: Path,
    contract_ref: Mapping[str, Any],
    source: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    candidate: str,
    step: int,
) -> dict[str, Any]:
    lr, stage = _schedule_lr(step)
    named = _candidate_named(context["consumer"], context["adapter"])
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "candidate": candidate,
        "contract": dict(contract_ref),
        "source_checkpoint": source["checkpoint_ref"],
        "facet_mode": context["facet_mode"],
        "facet_tau": float(context["tau"]),
        "checkpoint_step": step,
        "global_step": 600 + step,
        "total_steps": K,
        "runtime_state": _cpu_tree(context["consumer"].runtime.state_dict()),
        "adapter_state": _cpu_tree(context["adapter"].state_dict()),
        "optimizer_state": _cpu_tree(context["optimizer"].state_dict()),
        "sampler_state": _cpu_tree(context["sampler"].state_dict()),
        "rng_state": _cpu_tree(V10.V6._capture_rng()),
        "schedule_state": {"last_update": step, "stage": stage, "learning_rate": lr, "switch_step": STAGE_SWITCH},
        "trainable_parameter_names": [name for name, parameter in named.items() if parameter.requires_grad],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "facet", "Smooth evaluator", "templates", "prototypes"],
        "initial_generator_phase": _cpu_tree(context["initial_phase"]),
        "generator_geometry": {"layers": GENERATOR_LAYERS, "inter_layer_spacing_m": GENERATOR_LAYER_SPACING_M},
        "lift_rule": "new planes 3 and 4 are zero initialized at the common start; their optimizer moments are zero",
        "use_cache": False,
        "block_size": BLOCK_SIZE,
        "progress_protocol": {
            "bootstrap_checkpoint_before_step0_endpoint": True,
            "immutable_progress_events": True,
            "dev_evaluation_interval_rows": PROGRESS_EVAL_INTERVAL_ROWS,
            "training_block_checkpoint_before_terminal_endpoint": True,
        },
        "terminal_sampler_rule": "checkpoint step s saves sampler cursor=s and does not request s+1",
    }


def _save_checkpoint(root: Path, path: Path, payload: Mapping[str, Any], candidate: str, step: int) -> dict[str, Any]:
    ref = _save_tensor(root, path, payload, f"{candidate} checkpoint step {step}")
    loaded = _mapping(
        N4.safe_torch_load(path, expected_sha256=ref["sha256"], map_location="cpu"),
        f"{candidate} checkpoint reload {step}",
    )
    for key in (
        "runtime_state",
        "adapter_state",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "schedule_state",
    ):
        _require(_tree_equal(loaded[key], payload[key]), f"{candidate} checkpoint {key} changed on reload")
    _require(loaded.get("candidate") == candidate and loaded.get("checkpoint_step") == step, f"{candidate} checkpoint identity changed")
    _require(int(loaded["sampler_state"]["cursor"]) == step, f"{candidate} checkpoint sampler cursor changed")
    return {**ref, "step": step, "global_step": 600 + step, "reloaded": True}


def _load_checkpoint(root: Path, path: Path, contract_ref: Mapping[str, Any], candidate: str, step: int) -> dict[str, Any]:
    _require(path.is_file(), f"{candidate} checkpoint {step} is unavailable")
    digest = _sha256(path)
    checkpoint = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"{candidate} checkpoint {step}")
    _require(
        checkpoint.get("schema") == CHECKPOINT_SCHEMA
        and checkpoint.get("candidate") == candidate
        and checkpoint.get("checkpoint_step") == step
        and checkpoint.get("global_step") == 600 + step
        and checkpoint.get("total_steps") == K
        and checkpoint.get("contract") == dict(contract_ref),
        f"{candidate} checkpoint {step} identity changed",
    )
    _require(int(checkpoint["sampler_state"].get("cursor", -1)) == step, f"{candidate} checkpoint sampler cursor changed")
    _require(tuple(checkpoint["runtime_state"]["generator.phase"].shape) == (4, 1, 100, 100), f"{candidate} checkpoint generator shape changed")
    return {**dict(checkpoint), "_ref": {"path": _relative(root, path), "sha256": digest, "step": step, "global_step": 600 + step}}


def _context(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    contract_ref: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
    *,
    candidate: str,
    facet_mode: str,
    tau: float,
    output_dir: Path,
    step: int,
) -> dict[str, Any]:
    checkpoint_path = _checkpoint_path(output_dir, candidate, step)
    if step == 0 and not checkpoint_path.is_file():
        runtime_state = None
        adapter_state = None
        optimizer_state = _lift_optimizer_state(source["checkpoint"]["optimizer_state"])
        initial_phase = _lift_generator_state(source["checkpoint"]["runtime_state"])["generator.phase"]
        rng_state = source["checkpoint"]["rng_state"]
        checkpoint = source["checkpoint"]
    else:
        checkpoint = _load_checkpoint(root, checkpoint_path, contract_ref, candidate, step)
        runtime_state = checkpoint["runtime_state"]
        adapter_state = checkpoint["adapter_state"]
        optimizer_state = checkpoint["optimizer_state"]
        initial_phase = checkpoint["initial_generator_phase"]
        rng_state = checkpoint["rng_state"]
    V10.V6._restore_rng(_cpu_tree(rng_state))
    consumer, adapter, probe = _new_consumer(
        bundle,
        source,
        device,
        mode=facet_mode,
        tau=tau,
        trainable=True,
        runtime_state=runtime_state,
        adapter_state=adapter_state,
    )
    optimizer = _optimizer(consumer, adapter)
    _restore_optimizer(optimizer, optimizer_state)
    V10.V6._restore_rng(_cpu_tree(rng_state))
    sampler = ENGINE.BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=K)
    if step > 0:
        sampler.load_state_dict(_mapping(checkpoint["sampler_state"], f"{candidate} sampler state"))
    _require(sampler.cursor == step, f"{candidate} context sampler cursor changed")
    expected_lr, expected_stage = _schedule_lr(step)
    if step > 0:
        schedule = _mapping(checkpoint.get("schedule_state"), f"{candidate} schedule state")
        _require(schedule.get("stage") == expected_stage and math.isclose(float(schedule.get("learning_rate")), expected_lr, rel_tol=0.0, abs_tol=1.0e-15), f"{candidate} schedule changed at {step}")
    _set_stage(consumer, adapter, coadapt=expected_stage == "adapter_decoder_coadaptation")
    optimizer.param_groups[0]["lr"] = expected_lr
    return {
        "consumer": consumer,
        "adapter": adapter,
        "probe": probe,
        "optimizer": optimizer,
        "sampler": sampler,
        "checkpoint": checkpoint,
        "initial_phase": _cpu_tree(initial_phase),
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "bundle": bundle,
        "contract_ref": contract_ref,
        "device": device,
        "tau": tau,
        "facet_mode": facet_mode,
        "candidate": candidate,
        "output_dir": output_dir,
        "step": step,
    }


def _run_endpoint_for_context(
    root: Path,
    bundle: Mapping[str, Any],
    context: Mapping[str, Any],
    dev_indices: torch.Tensor,
    *,
    step: int,
) -> dict[str, Any]:
    dev_images = context["images"].index_select(0, dev_indices).contiguous()
    dev_labels = context["labels"].index_select(0, dev_indices).contiguous()
    selection_images, selection_labels, selection_indices = V10.V6._load_selection(
        root,
        context["images"],
        context["labels"],
        dev_indices,
    )
    train_metric = _train_metric(context, bundle)
    try:
        return _endpoint(
            root,
            context,
            bundle,
            dev_images,
            dev_labels,
            selection_images,
            selection_labels,
            selection_indices,
            candidate=context["candidate"],
            facet_mode=context["facet_mode"],
            step=step,
            train_metric=train_metric,
        )
    finally:
        del dev_images, dev_labels, selection_images, selection_labels, selection_indices


def _initialize_arm(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    contract_ref: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    device: torch.device,
    *,
    candidate: str,
    facet_mode: str,
    tau: float,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        _require(output_dir.is_dir() and not any(output_dir.iterdir()), f"{candidate} output directory is not a fresh empty boundary: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    context = _context(
        root,
        bundle,
        source,
        contract_ref,
        images,
        labels,
        train_indices,
        device,
        candidate=candidate,
        facet_mode=facet_mode,
        tau=tau,
        output_dir=output_dir,
        step=0,
    )
    checkpoint = _save_checkpoint(
        root,
        _checkpoint_path(output_dir, candidate, 0),
        _checkpoint_payload(root, contract_ref, source, context, candidate=candidate, step=0),
        candidate,
        0,
    )
    _save_progress_event(
        root,
        output_dir,
        candidate,
        phase="bootstrap_checkpoint",
        status="completed",
        step=0,
        rows_done=0,
        total_rows=int(dev_indices.numel()),
        checkpoint_ref=checkpoint,
    )
    endpoint = _run_endpoint_for_context(root, bundle, context, dev_indices, step=0)
    del context
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"endpoint": endpoint, "checkpoint": checkpoint}


def _initial_endpoint_exists(output_dir: Path, candidate: str) -> bool:
    prefix = candidate.lower()
    return (
        (output_dir / f"{prefix}_step_00000_sufficient_observations.v1.pt").is_file()
        and (output_dir / f"{prefix}_step_00000_diagnostics.v1.json").is_file()
    )


def _resume_initial_endpoint(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    contract_ref: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    device: torch.device,
    *,
    candidate: str,
    facet_mode: str,
    tau: float,
    output_dir: Path,
) -> dict[str, Any]:
    _require(_checkpoint_path(output_dir, candidate, 0).is_file(), f"{candidate} step-0 bootstrap checkpoint is unavailable")
    _require(not _initial_endpoint_exists(output_dir, candidate), f"{candidate} step-0 endpoint already exists; no endpoint resume is needed")
    context = _context(
        root,
        bundle,
        source,
        contract_ref,
        images,
        labels,
        train_indices,
        device,
        candidate=candidate,
        facet_mode=facet_mode,
        tau=tau,
        output_dir=output_dir,
        step=0,
    )
    try:
        endpoint = _run_endpoint_for_context(root, bundle, context, dev_indices, step=0)
    finally:
        del context
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return endpoint


def _run_block(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    contract_ref: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    device: torch.device,
    *,
    candidate: str,
    facet_mode: str,
    tau: float,
    output_dir: Path,
    current_step: int,
    target_step: int,
) -> dict[str, Any]:
    context = _context(
        root,
        bundle,
        source,
        contract_ref,
        images,
        labels,
        train_indices,
        device,
        candidate=candidate,
        facet_mode=facet_mode,
        tau=tau,
        output_dir=output_dir,
        step=current_step,
    )
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    z = _placeholder_z(LOGICAL_BATCH_SIZE).to(device)
    started = time.perf_counter()
    _save_progress_event(
        root,
        output_dir,
        candidate,
        phase="training_block",
        status="started",
        step=current_step,
        rows_done=current_step,
        total_rows=target_step,
        session=f"{time.time_ns():019d}",
    )
    try:
        for step in range(current_step + 1, target_step + 1):
            indices, batch_labels = context["sampler"].next_batch()
            batch = images.index_select(0, indices).to(device)
            _train_step(
                context["consumer"],
                context["adapter"],
                context["optimizer"],
                z,
                batch,
                batch_labels.to(device),
                prototypes,
                templates,
                step=step,
            )
            del batch
    except BaseException as error:
        _save_progress_event(
            root,
            output_dir,
            candidate,
            phase="training_block",
            status="failed",
            step=current_step,
            rows_done=int(context["sampler"].cursor),
            total_rows=target_step,
            elapsed_seconds=time.perf_counter() - started,
            error=error,
        )
        raise
    _require(context["sampler"].cursor == target_step, f"{candidate} block sampler cursor changed")
    checkpoint = _save_checkpoint(
        root,
        _checkpoint_path(output_dir, candidate, target_step),
        _checkpoint_payload(root, contract_ref, source, context, candidate=candidate, step=target_step),
        candidate,
        target_step,
    )
    block_elapsed = time.perf_counter() - started
    _save_progress_event(
        root,
        output_dir,
        candidate,
        phase="training_block",
        status="checkpointed",
        step=target_step,
        rows_done=target_step,
        total_rows=target_step,
        elapsed_seconds=block_elapsed,
        checkpoint_ref=checkpoint,
    )
    endpoint = None
    if target_step == K:
        endpoint = _run_endpoint_for_context(root, bundle, context, dev_indices, step=K)
    elapsed = time.perf_counter() - started
    del context, prototypes, templates, z
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"target_step": target_step, "checkpoint": checkpoint, "endpoint": endpoint, "training_seconds": elapsed}


def _load_new_endpoint(root: Path, output_dir: Path, candidate: str, step: int) -> dict[str, Any]:
    prefix = candidate.lower()
    sufficient_path = output_dir / f"{prefix}_step_{step:05d}_sufficient_observations.v1.pt"
    _require(sufficient_path.is_file(), f"{candidate} sufficient endpoint {step} is unavailable")
    sufficient_digest = _sha256(sufficient_path)
    sufficient = _mapping(N4.safe_torch_load(sufficient_path, expected_sha256=sufficient_digest, map_location="cpu"), f"{candidate} sufficient endpoint {step}")
    _require(sufficient.get("schema") == SUFFICIENT_SCHEMA and sufficient.get("candidate") == candidate and sufficient.get("global_step") == 600 + step, f"{candidate} sufficient endpoint {step} identity changed")
    diagnostic_path = output_dir / f"{prefix}_step_{step:05d}_diagnostics.v1.json"
    diagnostic = _read_json(diagnostic_path, f"{candidate} diagnostic endpoint {step}")
    _require(diagnostic.get("schema") == DIAGNOSTIC_SCHEMA and diagnostic.get("candidate") == candidate and int(diagnostic.get("step", -1)) == step, f"{candidate} diagnostic endpoint {step} identity changed")
    raw_ref = None
    if step == K:
        raw_path = output_dir / f"{prefix}_step_{step:05d}_cyclic_raw_outputs.v1.pt"
        _require(raw_path.is_file(), f"{candidate} final raw outputs are unavailable")
        raw_ref = {"path": _relative(root, raw_path), "sha256": _sha256(raw_path)}
    checkpoint_path = _checkpoint_path(output_dir, candidate, step)
    _require(checkpoint_path.is_file(), f"{candidate} checkpoint {step} is unavailable")
    return {
        "step": step,
        "candidate": candidate,
        "sufficient": sufficient,
        "sufficient_ref": {"path": _relative(root, sufficient_path), "sha256": sufficient_digest},
        "diagnostic": diagnostic,
        "diagnostic_ref": {"path": _relative(root, diagnostic_path), "sha256": _sha256(diagnostic_path)},
        "checkpoint_ref": {"path": _relative(root, checkpoint_path), "sha256": _sha256(checkpoint_path), "step": step, "global_step": 600 + step},
        "raw_ref": raw_ref,
        "dev_labels": sufficient["dev_labels"],
        "dev_correct": sufficient["dev_smooth_correct"],
        "evaluation_public": sufficient["evaluation_public"],
        "cyclic_public": sufficient["cyclic_public"],
        "dev_image_loss": float(sufficient["dev_image_loss"]),
        "train_metric": sufficient.get("train_metric", {}),
        "facet_diagnostics": diagnostic.get("diagnostics", {}).get("facet", {}),
        "diagnostics": diagnostic.get("diagnostics", {}),
    }


def _arm_report(
    root: Path,
    contract_ref: Mapping[str, Any],
    source: Mapping[str, Any],
    output_dir: Path,
    candidate: str,
    facet_mode: str,
    tau: float,
    *,
    training_seconds: float,
    resume_sessions: int,
) -> dict[str, Any]:
    initial = _load_new_endpoint(root, output_dir, candidate, 0)
    final = _load_new_endpoint(root, output_dir, candidate, K)
    labels = final["dev_labels"].to(torch.long)
    _require(torch.equal(labels, initial["dev_labels"].to(torch.long)), f"{candidate} endpoint labels are not matched")
    delta = final["dev_correct"].to(torch.float64) - initial["dev_correct"].to(torch.float64)
    ci = V10.V6._paired_bootstrap(delta, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    raw_validity = bool(
        final["evaluation_public"]["raw_validity"]["finite"]
        and final["evaluation_public"]["raw_validity"]["structured"]
        and final["evaluation_public"]["raw_validity"]["nonconstant"]
        and final["cyclic_public"]["raw_validity"]["base"]["finite"]
        and final["cyclic_public"]["raw_validity"]["shifted"]["finite"]
    )
    payload = {
        "schema": "lightgen_aligned_4l_relu_gain_v1_scientific_report.v1",
        "version": 1,
        "task_id": TASK_ID,
        "candidate": candidate,
        "facet_mode": facet_mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "VALID_COMPLETED",
        "status": "valid_completed",
        "classification": "C_4L_LINEAR_SCIENTIFIC_COMPLETED" if candidate.startswith("C_") else "D_4L_RELU90_SCIENTIFIC_COMPLETED",
        "outer_gate": OUTER_GATE,
        "contract": dict(contract_ref),
        "source_common_start": source["checkpoint_ref"],
        "tau": float(tau),
        "evaluations": {"0": initial["evaluation_public"], str(K): final["evaluation_public"]},
        "train_metrics": {"0": initial["train_metric"], str(K): final["train_metric"]},
        "sufficient_observations": {"0": initial["sufficient_ref"], str(K): final["sufficient_ref"]},
        "diagnostics": {"0": initial["diagnostic_ref"], str(K): final["diagnostic_ref"]},
        "checkpoints": {"0": initial["checkpoint_ref"], str(K): final["checkpoint_ref"]},
        "endpoint_raw_outputs": final["raw_ref"],
        "training_delta": {
            "R0": float(initial["evaluation_public"]["smooth"]["macro_recall"]),
            "RK": float(final["evaluation_public"]["smooth"]["macro_recall"]),
            "Delta_R": float(delta.mean()),
            "paired_bootstrap": {key: value for key, value in ci.items() if key != "delta_records"},
        },
        "reconstruction": {
            "DEV_L_image_R0": initial["dev_image_loss"],
            "DEV_L_image_RK": final["dev_image_loss"],
            "Delta_L_image_within_arm": final["dev_image_loss"] - initial["dev_image_loss"],
        },
        "cyclic_transfer": final["cyclic_public"],
        "facet_diagnostics": {"0": initial["facet_diagnostics"], str(K): final["facet_diagnostics"]},
        "raw_validity_pass": raw_validity,
        "timing": {"training_seconds_current_execution": training_seconds, "pause_resume_sessions": resume_sessions},
        "claim_boundary": {
            "scientific_training_executed": True,
            "scientific_data_consumed": True,
            "scientific_updates": K,
            "scientific_forward_evaluations": 4,
            "stage_b": False,
            "latent_prior": False,
            "sampled_generation": False,
            "protected": False,
            "official_test": False,
            "sealed": False,
            "formal_promotion": False,
            "physical_te_validation": False,
        },
        "governance": {
            "generator_layers": GENERATOR_LAYERS,
            "generator_layer_spacing_m": GENERATOR_LAYER_SPACING_M,
            "encoder_update": False,
            "ols_update": False,
            "w_ols_update": False,
            "material_trainable": False,
            "cache_used": False,
            "sampler": ENGINE.BalancedMultiEpochSampler.SCHEMA,
            "block_size": BLOCK_SIZE,
            "no_retry": True,
            "no_tau_sweep": True,
                "progress_events": {
                "immutable": True,
                "dev_evaluation_interval_rows": PROGRESS_EVAL_INTERVAL_ROWS,
                "checkpoint_every_steps": BLOCK_SIZE,
            },
        },
    }
    report_path = output_dir / f"{candidate.lower()}_scientific_report.v1.json"
    report_ref = _save_json(root, report_path, payload, f"{candidate} scientific report")
    lineage_path = output_dir / f"{candidate.lower()}_scientific_lineage.v1.json"
    lineage_ref = _save_json(
        root,
        lineage_path,
        {
            "schema": "lightgen_aligned_4l_relu_gain_v1_scientific_lineage.v1",
            "version": 1,
            "task_id": TASK_ID,
            "candidate": candidate,
            "result": payload["result"],
            "status": payload["status"],
            "classification": payload["classification"],
            "report": report_ref,
            "contract": dict(contract_ref),
            "claim_boundary": payload["claim_boundary"],
        },
        f"{candidate} scientific lineage",
    )
    return {"payload": payload, "report": report_ref, "lineage": lineage_ref}


def _cyclic_pass(cyclic: Mapping[str, Any]) -> bool:
    return bool(
        cyclic.get("template", {}).get("pass")
        and cyclic.get("smooth", {}).get("pass")
        and cyclic.get("broad")
        and not cyclic.get("pair_dominated")
    )


def _load_report(root: Path, ref: Mapping[str, Any], name: str) -> dict[str, Any]:
    path = _path(root, ref["path"])
    _validate_sha(path, str(ref["sha256"]), name)
    return _read_json(path, name)


def _report_endpoint(root: Path, report: Mapping[str, Any], step: int, name: str) -> dict[str, Any]:
    sufficient_ref = _mapping(report.get("sufficient_observations", {}).get(str(step)), f"{name} sufficient {step} ref")
    path = _path(root, sufficient_ref["path"])
    _validate_sha(path, str(sufficient_ref["sha256"]), f"{name} sufficient {step}")
    sufficient = _mapping(N4.safe_torch_load(path, expected_sha256=str(sufficient_ref["sha256"]), map_location="cpu"), f"{name} sufficient {step}")
    return {"sufficient": sufficient, "sufficient_ref": dict(sufficient_ref)}


def _bootstrap_record(delta: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    ci = V10.V6._paired_bootstrap(delta, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    return {key: value for key, value in ci.items() if key != "delta_records"}


def _approx_equal(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= ARCHITECTURE_EQUIVALENCE_TOLERANCE


def _aggregate(
    root: Path,
    contract_ref: Mapping[str, Any],
    source: Mapping[str, Any],
    strict_audit: Mapping[str, Any],
    c_report_ref: Mapping[str, Any],
    d_report_ref: Mapping[str, Any],
    *,
    training_seconds: float,
    resume_sessions: Mapping[str, int],
) -> dict[str, Any]:
    _require(not AGGREGATE_DIR.exists(), f"aggregate output already exists: {AGGREGATE_DIR}")
    c_report = _load_report(root, c_report_ref, "C scientific report")
    d_report = _load_report(root, d_report_ref, "D scientific report")
    c0 = _report_endpoint(root, c_report, 0, "C")
    ck = _report_endpoint(root, c_report, K, "C")
    d0 = _report_endpoint(root, d_report, 0, "D")
    dk = _report_endpoint(root, d_report, K, "D")
    a0 = strict_audit["reuse"]["endpoints"]["0"]["sufficient"]
    ak = strict_audit["reuse"]["endpoints"][str(K)]["sufficient"]
    b0 = strict_audit["b_endpoints"]["0"]["sufficient"]
    bk = strict_audit["b_endpoints"][str(K)]["sufficient"]

    labels = ak["dev_labels"].to(torch.long)
    for name, endpoint in (("B", bk), ("C", ck["sufficient"]), ("D", dk["sufficient"])):
        _require(torch.equal(labels, endpoint["dev_labels"].to(torch.long)), f"{name} endpoint DEV labels are not matched to A")
    delta2 = bk["dev_smooth_correct"].to(torch.float64) - ak["dev_smooth_correct"].to(torch.float64)
    delta4 = dk["sufficient"]["dev_smooth_correct"].to(torch.float64) - ck["sufficient"]["dev_smooth_correct"].to(torch.float64)
    interaction = delta4 - delta2
    architecture_linear = ck["sufficient"]["dev_smooth_correct"].to(torch.float64) - ak["dev_smooth_correct"].to(torch.float64)
    architecture_relu = dk["sufficient"]["dev_smooth_correct"].to(torch.float64) - bk["dev_smooth_correct"].to(torch.float64)
    r_a = float(ak["evaluation_public"]["smooth"]["macro_recall"])
    r_b = float(bk["evaluation_public"]["smooth"]["macro_recall"])
    r_c = float(ck["sufficient"]["evaluation_public"]["smooth"]["macro_recall"])
    r_d = float(dk["sufficient"]["evaluation_public"]["smooth"]["macro_recall"])
    image_delta4 = float(dk["sufficient"]["dev_image_loss"] - ck["sufficient"]["dev_image_loss"])
    cyclic = {
        "A_LINEAR": _cyclic_pass(ak["cyclic_public"]),
        "B_RELU90": _cyclic_pass(bk["cyclic_public"]),
        "C_4L_LINEAR": _cyclic_pass(ck["sufficient"]["cyclic_public"]),
        "D_4L_RELU90": _cyclic_pass(dk["sufficient"]["cyclic_public"]),
    }
    raw_validity = {
        "A_LINEAR": bool(strict_audit["reuse"]["report"].get("raw_validity_pass")),
        "B_RELU90": bool(strict_audit["b_report"].get("raw_validity_pass")),
        "C_4L_LINEAR": bool(c_report.get("raw_validity_pass")),
        "D_4L_RELU90": bool(d_report.get("raw_validity_pass")),
    }
    supported = bool(
        float(delta4.mean()) >= MATERIAL_DELTA
        and float(_bootstrap_record(delta4, labels)["ci_lower"]) > 0.0
        and image_delta4 <= RECONSTRUCTION_MAX_DELTA
        and cyclic["C_4L_LINEAR"]
        and cyclic["D_4L_RELU90"]
        and raw_validity["C_4L_LINEAR"]
        and raw_validity["D_4L_RELU90"]
    )
    delta4_ci = _bootstrap_record(delta4, labels)
    delta2_ci = _bootstrap_record(delta2, labels)
    interaction_ci = _bootstrap_record(interaction, labels)
    architecture_linear_ci = _bootstrap_record(architecture_linear, labels)
    architecture_relu_ci = _bootstrap_record(architecture_relu, labels)
    per_class = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        per_class.append(
            {
                "class": digit,
                "A_recall": float(ak["dev_smooth_correct"][mask].to(torch.float64).mean()),
                "B_recall": float(bk["dev_smooth_correct"][mask].to(torch.float64).mean()),
                "C_recall": float(ck["sufficient"]["dev_smooth_correct"][mask].to(torch.float64).mean()),
                "D_recall": float(dk["sufficient"]["dev_smooth_correct"][mask].to(torch.float64).mean()),
                "delta_2L_B_minus_A": float(delta2[mask].mean()),
                "delta_4L_D_minus_C": float(delta4[mask].mean()),
                "interaction": float(interaction[mask].mean()),
            }
        )
    matrix = {
        "tolerance": ARCHITECTURE_EQUIVALENCE_TOLERANCE,
        "C_approx_A": _approx_equal(r_c, r_a),
        "D_approx_C": _approx_equal(r_d, r_c),
        "D_approx_B": _approx_equal(r_d, r_b),
        "C_minus_A": r_c - r_a,
        "D_minus_C": r_d - r_c,
        "D_minus_B": r_d - r_b,
    }
    if matrix["C_approx_A"] and matrix["D_approx_C"]:
        interpretation = "4L architecture does not materially change the linear endpoint and ReLU does not materially change the 4L endpoint"
    elif matrix["C_approx_A"] and not matrix["D_approx_C"]:
        interpretation = "the 4L linear endpoint is near 2L linear, while the ReLU effect changes at 4L"
    elif not matrix["C_approx_A"] and matrix["D_approx_B"]:
        interpretation = "the architecture change is visible, while ReLU adds little beyond the 2L ReLU arm"
    else:
        interpretation = "both architecture and nonlinearity effects are present; inspect the interaction term"
    stage_a_best = max(r_a, r_b, r_c, r_d)
    stage_a = {
        "best_R": stage_a_best,
        "threshold": STAGE_A_FIREWALL,
        "status": "STAGE_A_FIREWALL_PASSED" if stage_a_best >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED",
        "stage_b_authorized": False,
    }
    c_facet0 = c_report["facet_diagnostics"]["0"]
    c_facetk = c_report["facet_diagnostics"][str(K)]
    d_facet0 = d_report["facet_diagnostics"]["0"]
    d_facetk = d_report["facet_diagnostics"][str(K)]
    payload = {
        "schema": "lightgen_aligned_4l_relu_gain_v1_aggregate.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "VALID_COMPLETED",
        "status": "valid_completed",
        "classification": "LIGHTGEN_ALIGNED_4L_RELU_GAIN_SUPPORTED" if supported else "LIGHTGEN_ALIGNED_4L_RELU_GAIN_NOT_SUPPORTED",
        "outer_gate": OUTER_GATE,
        "contract": dict(contract_ref),
        "source_common_start": source["checkpoint_ref"],
        "strict_2l_authority": strict_audit["aggregate_ref"],
        "candidates": {
            "A_LINEAR_REUSED": {"report": strict_audit["reuse"]["payload"]["previous_a"]["report"], "R2000": r_a},
            "B_RELU90_REUSED": {"report": strict_audit["b_report_ref"], "R2000": r_b},
            "C_4L_LINEAR": {"report": dict(c_report_ref), "R2000": r_c},
            "D_4L_RELU90": {"report": dict(d_report_ref), "R2000": r_d},
        },
        "primary": {
            "A": "A_LINEAR_REUSED",
            "B": "B_RELU90_REUSED",
            "C": "C_4L_LINEAR",
            "D": "D_4L_RELU90",
            "R_A2000": r_a,
            "R_B2000": r_b,
            "R_C2000": r_c,
            "R_D2000": r_d,
            "delta_4L_D_minus_C": float(delta4.mean()),
            "paired_bootstrap_delta_4L": delta4_ci,
            "image_delta_D_minus_C": image_delta4,
            "reconstruction_max_delta": RECONSTRUCTION_MAX_DELTA,
            "cyclic_transfer_C": cyclic["C_4L_LINEAR"],
            "cyclic_transfer_D": cyclic["D_4L_RELU90"],
            "raw_validity_C": raw_validity["C_4L_LINEAR"],
            "raw_validity_D": raw_validity["D_4L_RELU90"],
            "supported": supported,
        },
        "factorial": {
            "delta_4L_D_minus_C": {"point": float(delta4.mean()), "paired_bootstrap": delta4_ci},
            "delta_2L_B_minus_A": {"point": float(delta2.mean()), "paired_bootstrap": delta2_ci},
            "architecture_linear_C_minus_A": {"point": float(architecture_linear.mean()), "paired_bootstrap": architecture_linear_ci},
            "architecture_relu_D_minus_B": {"point": float(architecture_relu.mean()), "paired_bootstrap": architecture_relu_ci},
            "interaction_delta4_minus_delta2": {"point": float(interaction.mean()), "paired_bootstrap": interaction_ci},
            "definitions": {"delta_4L": "D-C", "delta_2L": "B-A", "interaction": "(D-C)-(B-A)", "architecture_linear": "C-A", "architecture_relu": "D-B"},
        },
        "per_class": per_class,
        "interpretation_matrix": {"metrics": matrix, "interpretation": interpretation},
        "cyclic_transfer": cyclic,
        "raw_validity": raw_validity,
        "facet_diagnostics": {
            "C_4L_LINEAR": {"0": c_facet0, str(K): c_facetk},
            "D_4L_RELU90": {"0": d_facet0, str(K): d_facetk},
        },
        "mechanism_diagnostics": {
            "tau": FACET_TAU_STRICT_V2,
            "C_power_retention_0": c_facet0.get("power_retention_mean"),
            "C_power_retention_2000": c_facetk.get("power_retention_mean"),
            "D_power_retention_0": d_facet0.get("power_retention_mean"),
            "D_power_retention_2000": d_facetk.get("power_retention_mean"),
            "D_activation_0": d_facet0.get("activation_mean"),
            "D_activation_2000": d_facetk.get("activation_mean"),
            "C_generator_phase": _load_report(root, c_report["diagnostics"][str(K)], "C final diagnostics").get("diagnostics", {}).get("generator", {}),
            "D_generator_phase": _load_report(root, d_report["diagnostics"][str(K)], "D final diagnostics").get("diagnostics", {}).get("generator", {}),
            "physical_te_validation": False,
        },
        "stage_a": stage_a,
        "timing": {"training_seconds_current_execution": training_seconds, "resume_sessions": dict(resume_sessions)},
        "claim_boundary": {
            "new_training_updates": {"C_4L_LINEAR": K, "D_4L_RELU90": K},
            "reused_A_B_updates": 0,
            "ideal_simulation_only": True,
            "physical_te_validation": False,
            "stage_b": False,
            "sampled_generation": False,
            "protected": False,
            "official_test": False,
            "sealed": False,
            "formal_promotion": False,
        },
        "governance": {
            "generator_layers": GENERATOR_LAYERS,
            "generator_layer_spacing_m": GENERATOR_LAYER_SPACING_M,
            "no_retry": True,
            "no_tau_sweep": True,
            "cache_used": False,
            "a_b_retrained": False,
        },
        "next": "one compact diagnostic visualization may consume this aggregate; Stage B, sampled generation and physical Te validation remain closed",
    }
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=False)
    report_path = AGGREGATE_DIR / "lightgen_aligned_4l_relu_gain_v1_aggregate_report.v1.json"
    report_ref = _save_json(root, report_path, payload, "4L aggregate report")
    lineage_ref = _save_json(
        root,
        AGGREGATE_DIR / "lightgen_aligned_4l_relu_gain_v1_aggregate_lineage.v1.json",
        {
            "schema": "lightgen_aligned_4l_relu_gain_v1_aggregate_lineage.v1",
            "version": 1,
            "task_id": TASK_ID,
            "result": payload["result"],
            "status": payload["status"],
            "classification": payload["classification"],
            "report": report_ref,
            "contract": dict(contract_ref),
            "claim_boundary": payload["claim_boundary"],
        },
        "4L aggregate lineage",
    )
    return {"payload": payload, "report": report_ref, "lineage": lineage_ref}


def _preflight_report_payload(
    started: Mapping[str, Any],
    *,
    result: str,
    status: str,
    classification: str,
    contract_ref: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "lightgen_aligned_4l_relu_gain_v1_preflight_report.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "status": status,
        "classification": classification,
        "outer_gate": OUTER_GATE,
        "contract": dict(contract_ref) if contract_ref is not None else None,
        "provenance": {"git_start": dict(started)},
        "claim_boundary": {
            "engineering_preflight_executed": result == "AUTHORIZABLE",
            "scientific_training_executed": False,
            "scientific_updates": 0,
            "stage_b": False,
            "sampled_generation": False,
            "protected_accessed": False,
            "official_test_accessed": False,
            "sealed_accessed": False,
            "formal_promotion": False,
        },
    }
    if details:
        payload.update(dict(details))
    if error is not None:
        payload["error"] = dict(error)
    return payload


def run_preflight(root: Path) -> dict[str, Any]:
    _configure_engine()
    _require(not PREFLIGHT_DIR.exists(), f"refusing to reuse 4L preflight directory: {PREFLIGHT_DIR}")
    _require(not CONTRACT_DIR.exists(), f"4L contract directory already exists: {CONTRACT_DIR}")
    _require(not C_DIR.exists() and not D_DIR.exists() and not AGGREGATE_DIR.exists(), "4L scientific output boundary is not fresh")
    started = V10.V6.git_facts(root)
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=False)
    contract_ref: dict[str, Any] | None = None
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from 4L authorization")
        paper = _paper_audit(root)
        source = _source(root)
        source_facts = _initial_state_facts(source)
        authority = _authorities(root)
        bundle = authority["bundle"]
        V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        _require(device.type == "cuda", "CUDA is required for the 4L preflight")
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, dev_indices, dataset = V10.V6._load_roles(root)
        _require(dataset.get("TRAIN", {}).get("rows") == 55_000 and dataset.get("DEV", {}).get("rows") == 5_000, "4L data roles changed")
        _require(dataset.get("protected", 0) in (0, [], None) and dataset.get("sealed", 0) in (0, [], None), "protected or sealed data role is nonempty")
        strict_audit = _strict_authority_audit(root, source)
        calibration = _strict_calibration(root)
        tau_check = _tau_step0_check(root, bundle, source, calibration, images, labels, device)
        initialization = _initialization_comparison(bundle, source, device, FACET_TAU_STRICT_V2)
        negative_controls = {
            "C_4L_LINEAR": _negative_controls(bundle, source, images, labels, train_indices, device, mode="linear", tau=0.0),
            "D_4L_RELU90": _negative_controls(bundle, source, images, labels, train_indices, device, mode="relu", tau=FACET_TAU_STRICT_V2),
        }
        smoke = {
            "C_4L_LINEAR": _smoke_arm(bundle, source, images, labels, train_indices, device, mode="linear", tau=0.0),
            "D_4L_RELU90": _smoke_arm(bundle, source, images, labels, train_indices, device, mode="relu", tau=FACET_TAU_STRICT_V2),
        }
        stage2 = {
            "C_4L_LINEAR": _stage2_smoke(bundle, source, images, labels, train_indices, device, mode="linear", tau=0.0),
            "D_4L_RELU90": _stage2_smoke(bundle, source, images, labels, train_indices, device, mode="relu", tau=FACET_TAU_STRICT_V2),
        }
        seconds_per_update = max(float(smoke[name]["seconds_per_update"]) for name in smoke)
        peak_memory = max(int(smoke[name]["peak_memory_bytes"]) for name in smoke)
        estimated_active = float(2 * K * seconds_per_update)
        benchmark = {
            "arms": {
                name: {
                    "seconds_per_update": float(smoke[name]["seconds_per_update"]),
                    "peak_memory_bytes": int(smoke[name]["peak_memory_bytes"]),
                }
                for name in smoke
            },
            "conservative_seconds_per_update": seconds_per_update,
            "estimated_active_gpu_seconds": estimated_active,
            "budget_seconds": ACTIVE_GPU_BUDGET_SECONDS,
            "peak_memory_bytes": peak_memory,
            "device_vram_bytes": int(device_facts.get("vram_bytes", 0)),
        }
        _require(estimated_active <= ACTIVE_GPU_BUDGET_SECONDS, "4L two-arm estimate exceeds the 6-hour active-GPU budget")
        if int(device_facts.get("vram_bytes", 0)) > 0:
            _require(peak_memory < int(device_facts["vram_bytes"]), "4L smoke peak VRAM exceeds the device capacity")
        contract = _make_contract(
            root,
            started,
            paper,
            source,
            source_facts,
            strict_audit,
            calibration,
            tau_check,
            initialization,
            benchmark,
        )
        contract_ref = _save_json(root, CONTRACT_PATH, contract, "4L experiment contract")
        status_after = V10.V6.git_facts(root)
        details = {
            "contract": contract_ref,
            "source_common_start": source["checkpoint_ref"],
            "paper": paper,
            "strict_2l_authority": {
                "aggregate": strict_audit["aggregate_ref"],
                "preflight": strict_audit["preflight_report_ref"],
                "reuse_audit": strict_audit["reuse_ref"],
                "decision": "PASS; reuse A/B and do not retrain them",
            },
            "tau_check": tau_check,
            "initialization": initialization,
            "negative_controls": negative_controls,
            "smoke": smoke,
            "stage2": stage2,
            "benchmark": benchmark,
            "runtime": {"device": device_facts, "production_grid": PRODUCTION_GRID},
            "data_roles": dataset,
            "provenance": {"git_start": started, "git_end": status_after, "branch_unchanged": started.get("branch") == status_after.get("branch"), "head_unchanged": started.get("head") == status_after.get("head")},
        }
        payload = _preflight_report_payload(
            started,
            result="AUTHORIZABLE",
            status="valid_completed",
            classification="LIGHTGEN_ALIGNED_4L_RELU_GAIN_PREFLIGHT_AUTHORIZABLE",
            contract_ref=contract_ref,
            details=details,
        )
    except HardBlockedFourLayerRun as error:
        payload = _preflight_report_payload(
            started,
            result="BLOCKED",
            status="valid_failed",
            classification=error.code,
            contract_ref=contract_ref,
            error={"type": type(error).__name__, "message": str(error), "details": error.details},
        )
    except Exception as error:
        payload = _preflight_report_payload(
            started,
            result="INVALID",
            status="invalid_harness",
            classification="INVALID_LIGHTGEN_ALIGNED_4L_RELU_GAIN_PREFLIGHT",
            contract_ref=contract_ref,
            error={"type": type(error).__name__, "message": str(error)},
        )
    report_ref = _save_json(root, PREFLIGHT_REPORT, payload, "4L preflight report")
    lineage_ref = _save_json(
        root,
        PREFLIGHT_LINEAGE,
        {
            "schema": "lightgen_aligned_4l_relu_gain_v1_preflight_lineage.v1",
            "version": 1,
            "task_id": TASK_ID,
            "result": payload["result"],
            "status": payload["status"],
            "classification": payload["classification"],
            "report": report_ref,
            "contract": contract_ref,
            "claim_boundary": payload["claim_boundary"],
        },
        "4L preflight lineage",
    )
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": lineage_ref, "contract": contract_ref}


def _load_preflight(root: Path) -> dict[str, Any]:
    report_path = _path(root, PREFLIGHT_REPORT)
    lineage_path = _path(root, PREFLIGHT_LINEAGE)
    report = _read_json(report_path, "4L preflight report")
    lineage = _read_json(lineage_path, "4L preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed", "4L preflight is not authorizable")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    _require(lineage.get("report") == report_ref, "4L preflight lineage binding changed")
    contract_ref = _mapping(report.get("contract"), "4L contract reference")
    contract_path = _path(root, contract_ref["path"])
    _validate_sha(contract_path, str(contract_ref["sha256"]), "4L contract")
    contract = _read_json(contract_path, "4L contract")
    _require(contract.get("task_id") == TASK_ID and contract.get("outer_gate") == OUTER_GATE, "4L contract identity changed")
    _require(contract.get("status") == "READY_FOR_BOUNDED_LONG_SCIENTIFIC_EXECUTION", "4L contract is not ready")
    geometry = _mapping(contract.get("geometry"), "4L contract geometry")
    _require(geometry.get("variant") == "LIGHTGEN_TOPOLOGY_ALIGNED_4L", "4L geometry variant changed")
    project_geometry = _mapping(geometry.get("project_used"), "4L project geometry")
    _require(project_geometry.get("generator_layers") == GENERATOR_LAYERS and float(project_geometry.get("generator_layer_spacing_m")) == GENERATOR_LAYER_SPACING_M, "4L generator geometry changed")
    training = _mapping(contract.get("training"), "4L training contract")
    for key, expected in (("K", K), ("batch", LOGICAL_BATCH_SIZE), ("stage_switch", STAGE_SWITCH), ("sampler_seed", SAMPLER_SEED), ("block_size", BLOCK_SIZE)):
        _require(training.get(key) == expected, f"4L training contract changed: {key}")
    _require(report.get("claim_boundary", {}).get("scientific_updates") == 0, "4L preflight scientific boundary changed")
    return {"report": report, "contract": contract, "contract_ref": dict(contract_ref), "report_ref": report_ref, "lineage_ref": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)} }


def _run_scientific_arm(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    contract_ref: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    device: torch.device,
    *,
    candidate: str,
    facet_mode: str,
    tau: float,
    output_dir: Path,
    resume_step: int | None,
) -> dict[str, Any]:
    _require(resume_step is None or (isinstance(resume_step, int) and 0 <= resume_step <= K and resume_step % BLOCK_SIZE == 0), f"{candidate} resume step must be a checkpoint boundary")
    if resume_step is None:
        _initialize_arm(
            root,
            bundle,
            source,
            contract_ref,
            images,
            labels,
            train_indices,
            dev_indices,
            device,
            candidate=candidate,
            facet_mode=facet_mode,
            tau=tau,
            output_dir=output_dir,
        )
        current_step = 0
        resume_sessions = 0
    else:
        _require(output_dir.is_dir(), f"{candidate} resume directory is unavailable")
        _require(_checkpoint_path(output_dir, candidate, resume_step).is_file(), f"{candidate} resume checkpoint {resume_step} is unavailable")
        current_step = resume_step
        resume_sessions = 1
        if current_step == 0 and not _initial_endpoint_exists(output_dir, candidate):
            _resume_initial_endpoint(
                root,
                bundle,
                source,
                contract_ref,
                images,
                labels,
                train_indices,
                dev_indices,
                device,
                candidate=candidate,
                facet_mode=facet_mode,
                tau=tau,
                output_dir=output_dir,
            )
        elif current_step > 0:
            _require(_initial_endpoint_exists(output_dir, candidate), f"{candidate} step-0 endpoint is required before resuming step {current_step}")
    training_seconds = 0.0
    while current_step < K:
        target_step = min(K, current_step + BLOCK_SIZE)
        block = _run_block(
            root,
            bundle,
            source,
            contract_ref,
            images,
            labels,
            train_indices,
            dev_indices,
            device,
            candidate=candidate,
            facet_mode=facet_mode,
            tau=tau,
            output_dir=output_dir,
            current_step=current_step,
            target_step=target_step,
        )
        current_step = target_step
        training_seconds += float(block["training_seconds"])
        print(
            json.dumps(
                {
                    "event": "lightgen_4l_progress",
                    "candidate": candidate,
                    "step": current_step,
                    "K": K,
                    "training_seconds_current_execution": training_seconds,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    report_path = output_dir / f"{candidate.lower()}_scientific_report.v1.json"
    if report_path.is_file():
        report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
        _load_report(root, report_ref, f"{candidate} existing scientific report")
    else:
        report = _arm_report(
            root,
            contract_ref,
            source,
            output_dir,
            candidate,
            facet_mode,
            tau,
            training_seconds=training_seconds,
            resume_sessions=resume_sessions,
        )
        report_ref = report["report"]
    return {"report": report_ref, "training_seconds": training_seconds, "resume_sessions": resume_sessions}


def run_scientific(
    root: Path,
    *,
    resume_c_step: int | None = None,
    resume_d_step: int | None = None,
) -> dict[str, Any]:
    _configure_engine()
    preflight = _load_preflight(root)
    source = _source(root)
    authority = _authorities(root)
    bundle = authority["bundle"]
    V10.V6._prepare_cuda_runtime_dll_search_path()
    device, device_facts = V10.V6._cuda_facts()
    _require(device.type == "cuda", "CUDA is required for 4L scientific execution")
    bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
    images, labels, train_indices, dev_indices, dataset = V10.V6._load_roles(root)
    _require(dataset.get("protected", 0) in (0, [], None) and dataset.get("sealed", 0) in (0, [], None), "protected or sealed role is nonempty")
    strict_audit = _strict_authority_audit(root, source)
    tau = float(preflight["contract"]["facet"]["tau_90"])
    _require(tau == FACET_TAU_STRICT_V2, "preflight tau binding changed")
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "scientific_start",
                "arms": ["C_4L_LINEAR", "D_4L_RELU90"],
                "K": K,
                "A_B": "REUSED_VALID_STRICT_2L_AUTHORITY",
                "generator_layers": GENERATOR_LAYERS,
                "generator_layer_spacing_m": GENERATOR_LAYER_SPACING_M,
                "progress_protocol": {
                    "dev_evaluation_interval_rows": PROGRESS_EVAL_INTERVAL_ROWS,
                    "checkpoint_every_steps": BLOCK_SIZE,
                    "terminal_checkpoint_before_endpoint": True,
                },
                "resume_c_step": resume_c_step,
                "resume_d_step": resume_d_step,
                "device": device_facts,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    c_result = _run_scientific_arm(
        root,
        bundle,
        source,
        preflight["contract_ref"],
        images,
        labels,
        train_indices,
        dev_indices,
        device,
        candidate="C_4L_LINEAR",
        facet_mode="linear",
        tau=0.0,
        output_dir=C_DIR,
        resume_step=resume_c_step,
    )
    d_result = _run_scientific_arm(
        root,
        bundle,
        source,
        preflight["contract_ref"],
        images,
        labels,
        train_indices,
        dev_indices,
        device,
        candidate="D_4L_RELU90",
        facet_mode="relu",
        tau=tau,
        output_dir=D_DIR,
        resume_step=resume_d_step,
    )
    aggregate = _aggregate(
        root,
        preflight["contract_ref"],
        source,
        strict_audit,
        c_result["report"],
        d_result["report"],
        training_seconds=float(c_result["training_seconds"] + d_result["training_seconds"]),
        resume_sessions={"C_4L_LINEAR": c_result["resume_sessions"], "D_4L_RELU90": d_result["resume_sessions"]},
    )
    aggregate["runtime"] = {
        "device": device_facts,
        "scientific_wall_seconds_current_execution": time.perf_counter() - started,
        "strict_2l_reuse": strict_audit["aggregate_ref"],
    }
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific", "all"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--resume-c-step", type=int, default=None, help="explicit last C checkpoint step")
    parser.add_argument("--resume-d-step", type=int, default=None, help="explicit last D checkpoint step")
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    if args.action == "preflight":
        _require(args.resume_c_step is None and args.resume_d_step is None, "preflight does not accept resume steps")
        result = run_preflight(root)
    elif args.action == "scientific":
        result = run_scientific(root, resume_c_step=args.resume_c_step, resume_d_step=args.resume_d_step)
    else:
        _require(args.resume_c_step is None and args.resume_d_step is None, "all does not accept resume steps")
        result = run_preflight(root)
        if result.get("result") != "AUTHORIZABLE":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2
        result = run_scientific(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("result") in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
