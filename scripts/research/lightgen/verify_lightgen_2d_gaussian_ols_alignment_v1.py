"""Zero-update integration verification for the explicit Gaussian OLS runtime."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import GaussianFiberArrayOLS, GaussianFiberGeometry
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import (
    GAUSSIAN_OLS_MODEL,
    TRAIN_SMOKE_ROLE,
    LightGen2DChenBOGTComplexOLSTrainingConsumer,
    collect_bogt_train_references,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import (
    SCALE_SMOKE_CLASSIFICATION,
    LightGen2DComplexOLSTrainingConsumerV2,
    extract_train_only_ols_scale,
    load_v2_training_contract,
    normalized_ols,
    physical_ols_from_normalized,
    sample_physical_prior,
    save_scale_artifact,
    load_scale_artifact,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    ComplexOLSState,
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.evidence import atomic_torch_save_new, safe_torch_load
from scripts.research.lightgen import verify_lightgen_2d_gaussian_fiber_geometry_v1 as geometry_validation


TASK_ID = "LIGHTGEN_GAUSSIAN_OLS_PAPER_STYLE_BASELINE_V1"
SEED = 0
SYNTHETIC_IMAGE_COUNT = 10
COEFFICIENT_TOLERANCE = 1.0e-5
POWER_TOLERANCE = 2.0e-5
ROLE = "NON_SCIENTIFIC_SMOKE"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)


def _numpy(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if np.iscomplexobj(array):
        return array.astype(np.complex128, copy=False)
    return array.astype(np.float64, copy=False)


def _state_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_unchanged(module: torch.nn.Module, before: Mapping[str, torch.Tensor]) -> bool:
    after = module.state_dict()
    return after.keys() == before.keys() and all(torch.equal(after[name].detach().cpu(), value) for name, value in before.items())


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _synthetic_images() -> torch.Tensor:
    return torch.linspace(0.05, 0.95, SYNTHETIC_IMAGE_COUNT * 28 * 28, dtype=torch.float32).reshape(
        SYNTHETIC_IMAGE_COUNT, 1, 28, 28
    )


def _independent_plane_powers(result: Any) -> tuple[torch.Tensor, list[Mapping[str, Any]]]:
    fields = (*result.encoder_records, result.generator_input_field, *result.generator_records)
    rows = []
    specs = []
    for field in fields:
        tensor = _numpy(field.tensor)
        axes = tuple(range(1, tensor.ndim))
        rows.append(np.sum(np.abs(tensor) ** 2, axis=axes) * math.prod(field.plane.pixel_pitch_m))
        specs.append(field.plane.to_dict())
    return torch.from_numpy(np.stack(rows, axis=1)), specs


def _independent_modes(plane: PlaneSpec, geometry: GaussianFiberGeometry) -> np.ndarray:
    y = (np.arange(plane.shape[0]) + 0.5) * plane.pixel_pitch_m[0] - plane.fov_m[0] / 2.0 + plane.origin_m[0]
    x = (np.arange(plane.shape[1]) + 0.5) * plane.pixel_pitch_m[1] - plane.fov_m[1] / 2.0 + plane.origin_m[1]
    centers = (np.arange(10) - 4.5) * geometry.port_pitch_m
    cy, cx = np.meshgrid(centers + plane.origin_m[0], centers + plane.origin_m[1], indexing="ij")
    radius_squared = (y[None, :, None] - cy.reshape(-1, 1, 1)) ** 2 + (x[None, None, :] - cx.reshape(-1, 1, 1)) ** 2
    return math.sqrt(2.0 / math.pi) / geometry.waist_m * np.exp(-radius_squared / geometry.waist_m**2)


def _component_preflight() -> dict[str, Any]:
    geometry = GaussianFiberGeometry()
    plane = PlaneSpec("gaussian_alignment_preflight", (100, 100), (2.0e-6, 2.0e-6), 532.0e-9, 0.0, "ols")
    operator = GaussianFiberArrayOLS(plane, geometry)
    fine_operator = GaussianFiberArrayOLS(geometry_validation.plane(1600), geometry)
    mode_checks, _ = geometry_validation.modal_checks(operator)
    coarse, _, _ = geometry_validation.tilt_checks(
        operator, geometry_validation.COARSE_ANGLES, precision="small_FP32"
    )
    refined, _, _ = geometry_validation.tilt_checks(
        fine_operator, geometry_validation.FINE_ANGLES, precision="refined_FP64"
    )
    gradient = geometry_validation.gradient_check(operator)
    old_negative = geometry_validation.negative_geometry_check(plane)

    theta = torch.tensor(math.radians(2.5), dtype=torch.float64)
    tilted = geometry_validation.tilt_input(operator, theta)
    field = ComplexField(tilted, plane, OperationProvenance("phase-loss-negative", None, None))
    coefficients = operator.project_coefficients(field).detach().cpu().numpy()
    phase_lost = torch.complex(tilted.abs(), torch.zeros_like(tilted.real))
    phase_lost_coefficients = operator.project_coefficients(
        ComplexField(phase_lost, plane, OperationProvenance("phase-loss-negative", None, None))
    ).detach().cpu().numpy()
    selected = geometry_validation.SELECTED_PORT
    phase_loss_error = float(abs(coefficients[0, selected] - phase_lost_coefficients[0, selected]))
    modes = _independent_modes(plane, geometry)
    source = _numpy(tilted)
    without_area = np.einsum("mhw,bhw->bm", modes.conj(), source)
    with_area = without_area * math.prod(plane.pixel_pitch_m)
    area_omission_error = float(abs(with_area[0, selected] - without_area[0, selected]))
    negatives = {
        "phase_discarded": {
            "executed": True,
            "passed": phase_loss_error > COEFFICIENT_TOLERANCE,
            "coefficient_error_from_production": phase_loss_error,
        },
        "pixel_area_omitted": {
            "executed": True,
            "passed": area_omission_error > COEFFICIENT_TOLERANCE,
            "coefficient_error_from_production": area_omission_error,
        },
    }
    checks: dict[str, Any] = {
        **mode_checks,
        "small_tilt": coarse,
        "refined_tilt": refined,
        "tilt_gradient": gradient,
        "old_geometry_rejected": old_negative,
        "phase_discarded_negative": negatives["phase_discarded"],
        "pixel_area_omitted_negative": negatives["pixel_area_omitted"],
    }
    return {
        "executed": True,
        "passed": all(value["executed"] and value["passed"] for value in checks.values()),
        "geometry": asdict(geometry),
        "tolerances": {
            "complex_coefficient_absolute": COEFFICIENT_TOLERANCE,
            "power_absolute": POWER_TOLERANCE,
            "coarse_angle_absolute": geometry_validation.COARSE_TILT_TOLERANCE,
            "fine_angle_absolute": geometry_validation.FINE_TILT_TOLERANCE,
            "tilt_gradient_relative": geometry_validation.GRADIENT_TOLERANCE,
        },
        "checks": checks,
        "scope": "independent component checks before the 2048-grid architecture forward",
    }


def _save_runtime_readout(
    path: Path,
    *,
    images: torch.Tensor,
    ols_input: torch.Tensor,
    encoder_output: torch.Tensor,
    z_ols: torch.Tensor,
    generator_input: torch.Tensor,
    raw_cmos: torch.Tensor,
    per_image_plane_power: torch.Tensor,
    plane_specs: list[Mapping[str, Any]],
    refs: Any,
    v2_scale: Any,
    runtime_identity: Mapping[str, Any],
    mode_geometry: Mapping[str, Any],
) -> None:
    payload = {
        "schema": "lightgen_gaussian_ols_alignment_readout.v1",
        "data_role": ROLE,
        "image": images.detach().cpu().clone(),
        "ols_input_field": {"tensor": ols_input.detach().cpu().clone(), "plane": dict(mode_geometry)},
        "ols_plane": dict(mode_geometry),
        "encoder_output_field": encoder_output.detach().cpu().clone(),
        "z_ols": z_ols.detach().cpu().clone(),
        "basis_id": GAUSSIAN_OLS_MODEL,
        "mode_geometry": dict(mode_geometry),
        "generator_input_field": generator_input.detach().cpu().clone(),
        "raw_cmos": raw_cmos.detach().cpu().clone(),
        "independent_per_image_plane_power": per_image_plane_power.detach().cpu().clone(),
        "power_plane_specs": [dict(record) for record in plane_specs],
        "bogt_scale": refs.scale.clone(),
        "bogt_power_reference": refs.power_reference.clone(),
        "v2_scale": v2_scale.values.clone(),
        "runtime_identity": dict(runtime_identity),
        "trainable_updates": 0,
        "classification": ROLE,
        "engineering_only": True,
    }
    atomic_torch_save_new(path, payload, artifact_kind="Gaussian OLS integration smoke readout")


def _independent_replay(readout_path: Path, expected_images: torch.Tensor, references: Any, v2_scale: Any) -> dict[str, Any]:
    payload = safe_torch_load(readout_path, map_location="cpu")
    if payload.get("schema") != "lightgen_gaussian_ols_alignment_readout.v1":
        raise ValueError("saved Gaussian readout schema mismatch")
    if payload.get("basis_id") != GAUSSIAN_OLS_MODEL or payload.get("data_role") != ROLE:
        raise ValueError("saved Gaussian readout modal basis/data role mismatch")
    saved_images_match = torch.equal(payload["image"], expected_images)
    geometry_record = payload["mode_geometry"]
    runtime_record = payload["runtime_identity"]
    tensor_shapes = {
        "image": tuple(payload["image"].shape),
        "ols_input_field": tuple(payload["ols_input_field"]["tensor"].shape),
        "encoder_output_field": tuple(payload["encoder_output_field"].shape),
        "z_ols": tuple(payload["z_ols"].shape),
        "generator_input_field": tuple(payload["generator_input_field"].shape),
        "raw_cmos": tuple(payload["raw_cmos"].shape),
        "independent_per_image_plane_power": tuple(payload["independent_per_image_plane_power"].shape),
        "bogt_scale": tuple(payload["bogt_scale"].shape),
        "bogt_power_reference": tuple(payload["bogt_power_reference"].shape),
        "v2_scale": tuple(payload["v2_scale"].shape),
    }
    tensors_well_formed = (
        tensor_shapes["image"] == (SYNTHETIC_IMAGE_COUNT, 1, 28, 28)
        and tensor_shapes["ols_input_field"] == (SYNTHETIC_IMAGE_COUNT, 1, *geometry_record["plane_shape"])
        and tensor_shapes["encoder_output_field"][0:2] == (SYNTHETIC_IMAGE_COUNT, 1)
        and tensor_shapes["z_ols"] == (SYNTHETIC_IMAGE_COUNT, 100)
        and tensor_shapes["generator_input_field"] == tensor_shapes["ols_input_field"]
        and tensor_shapes["raw_cmos"][0:2] == (SYNTHETIC_IMAGE_COUNT, 1)
        and tensor_shapes["independent_per_image_plane_power"] == (SYNTHETIC_IMAGE_COUNT, 5)
        and tensor_shapes["bogt_scale"] == (100,)
        and tensor_shapes["bogt_power_reference"] == (5,)
        and tensor_shapes["v2_scale"] == (100,)
    )
    records_bound = (
        payload["ols_input_field"]["plane"] == geometry_record
        and payload["ols_plane"] == geometry_record
        and runtime_record.get("ols_model") == GAUSSIAN_OLS_MODEL
        and runtime_record.get("mode_geometry") == geometry_record
        and runtime_record.get("active_propagation_grid") == 2048
        and len(payload["power_plane_specs"]) == 5
    )

    geometry = GaussianFiberGeometry()
    plane_geometry = payload["ols_plane"]
    ols_plane = PlaneSpec(
        "saved_ols_input",
        tuple(plane_geometry["plane_shape"]),
        tuple(plane_geometry["pixel_pitch_m"]),
        plane_geometry["wavelength_m"],
        0.0,
        "ols",
    )
    modes = _independent_modes(ols_plane, geometry)
    source = _numpy(payload["ols_input_field"]["tensor"])[:, 0]
    area = math.prod(ols_plane.pixel_pitch_m)
    projected = np.einsum("mhw,bhw->bm", modes.conj(), source) * area
    saved_z = _numpy(payload["z_ols"])
    coefficient_error = float(np.max(np.abs(projected - saved_z)))
    emitted = np.einsum("bm,mhw->bhw", saved_z, modes)
    saved_writer = _numpy(payload["generator_input_field"])[:, 0]
    writer_error = float(np.max(np.abs(emitted - saved_writer)))
    independent_scale = np.mean(np.abs(projected) ** 2, axis=0)
    independent_power_reference = np.mean(_numpy(payload["independent_per_image_plane_power"]), axis=0)
    scale_error_bogt = float(np.max(np.abs(independent_scale - _numpy(payload["bogt_scale"]))))
    scale_error_v2 = float(np.max(np.abs(independent_scale - _numpy(payload["v2_scale"]))))
    power_error = float(np.max(np.abs(independent_power_reference - _numpy(payload["bogt_power_reference"]))))
    saved_artifacts_error = max(
        float(torch.max(torch.abs(payload["bogt_scale"] - references.scale))),
        float(torch.max(torch.abs(payload["bogt_power_reference"] - references.power_reference))),
        float(torch.max(torch.abs(payload["v2_scale"] - v2_scale.values))),
    )
    normalized = normalized_ols(torch.from_numpy(saved_z.astype(np.complex64)), v2_scale)
    restored = physical_ols_from_normalized(normalized, v2_scale)
    inverse_error = float((restored - torch.from_numpy(saved_z.astype(np.complex64))).abs().max())
    return {
        "readback": {
            "executed": True,
            "passed": saved_images_match
            and tensors_well_formed
            and records_bound
            and payload["trainable_updates"] == 0
            and payload["engineering_only"] is True
            and saved_artifacts_error == 0.0,
            "image_count": int(payload["image"].shape[0]),
            "saved_runtime_model": runtime_record.get("ols_model"),
            "saved_classification": payload["classification"],
            "tensor_shapes": {name: list(shape) for name, shape in tensor_shapes.items()},
            "tensor_shapes_well_formed": tensors_well_formed,
            "runtime_and_mode_geometry_bound": records_bound,
            "saved_scale_and_reference_readback_max_absolute_error": saved_artifacts_error,
        },
        "numpy_complex_projection_with_pixel_area": {
            "executed": True,
            "passed": coefficient_error <= COEFFICIENT_TOLERANCE,
            "maximum_absolute_complex_error": coefficient_error,
        },
        "numpy_gaussian_reemission": {
            "executed": True,
            "passed": writer_error <= COEFFICIENT_TOLERANCE,
            "maximum_absolute_complex_error": writer_error,
        },
        "numpy_recomputed_ols_scale": {
            "executed": True,
            "passed": max(scale_error_bogt, scale_error_v2) <= POWER_TOLERANCE,
            "bogt_scale_max_absolute_error": scale_error_bogt,
            "v2_scale_max_absolute_error": scale_error_v2,
            "v2_consumer_accepted_rebuilt_scale": v2_scale.basis_id == GAUSSIAN_OLS_MODEL
            and v2_scale.mode_geometry == payload["mode_geometry"],
        },
        "numpy_recomputed_bogt_plane_power": {
            "executed": True,
            "passed": power_error <= POWER_TOLERANCE,
            "maximum_absolute_power_error": power_error,
            "bogt_consumer_accepted_rebuilt_references": references.basis_id == GAUSSIAN_OLS_MODEL
            and references.mode_geometry == payload["mode_geometry"],
        },
        "v2_normalized_inverse_mapping": {
            "executed": True,
            "passed": inverse_error <= COEFFICIENT_TOLERANCE,
            "max_absolute_complex_error": inverse_error,
        },
    }


def _report_md(decision: str, preflight: Mapping[str, Any], replay: Mapping[str, Any], runtime: Mapping[str, Any]) -> str:
    return (
        "# Gaussian OLS 论文式基线接入验证\n\n"
        "## 结果与状态\n\n"
        f"唯一判定：**{decision}**。训练更新数为 0；本次只执行了带工程标记的 10 张确定性合成图像。\n\n"
        "## 科学认识\n\n"
        "复系数投影、双端口相干、倾斜平面波、Gaussian 重发射和像素面积项均由独立参考核对。Gaussian 模式使用解析 S12 归一化，"
        "不做逐端口矩形裁切、离散重归一化或正交化。几何标签为 `PROJECT_GEOMETRY_UNMEASURED`。\n\n"
        "## 目标缺口\n\n"
        "这组零更新结果没有验证训练收益、正式 55,000 行 TRAIN 尺度、语义可读性、论文实物参数对应关系或光路部署可实现性。\n\n"
        "## 工程完成与验证\n\n"
        f"组件预检：`{'PASS' if preflight.get('passed') else 'FAIL'}`；2048 网格零更新生产运行与保存读出：`{'PASS' if bool(replay) and all(item.get('passed') for item in replay.values()) else 'FAIL'}`。"
        f"运行配置：`{runtime.get('ols_model')}`，端口序 `row-major j=10*r+c`，传输 `T=I_100`。\n\n"
        "机器可读证据见 `plan.json`、`preflight.json`、`runtime_readout.pt`、`independent_replay.json` 和 `verification.json`。\n\n"
        "## 决策影响与下一步\n\n"
        "通过只表示显式 Gaussian 研究变体与当前定义及依赖量重建机制一致；不据此启动训练、全量重算、云端部署或模型采用。"
        "若要形成正式尺度，仍需完整 55,000 行 TRAIN 输入；若要声称实物对应，还需测得 MFD、端口间距和中继成像参数。\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    target = args.output_dir.resolve()
    allowed_parent = root / "outputs" / "20261013_lightgen_gaussian_ols_alignment_v1" / "runs"
    if target.parent != allowed_parent or target.exists() or not target.name.startswith("diagnostic-only_attempt_"):
        raise ValueError("output must be a new diagnostic-only child of the named 20261013 run directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    device = torch.device(args.device)
    target.mkdir(parents=True)
    torch.set_num_threads(2)
    geometry = GaussianFiberGeometry()
    plan = {
        "task_id": TASK_ID,
        "question": "Does the explicit Gaussian S12 runtime consistently produce, bind, save, and independently reproduce modal latents and dependent scales?",
        "inputs": {
            "wavelength_m": 532.0e-9,
            "port_count": 100,
            "array_shape": [10, 10],
            "plane_shape": [100, 100],
            "pixel_pitch_m": [2.0e-6, 2.0e-6],
            "fov_m": [200.0e-6, 200.0e-6],
            "port_center_pitch_m": geometry.port_pitch_m,
            "gaussian_field_1e_radius_m": geometry.waist_m,
            "gaussian_mode_field_diameter_m": 2.0 * geometry.waist_m,
            "port_order": "row-major j=10*r+c",
            "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
            "initialization_seed": SEED,
            "images": "linspace(0.05,0.95,10*28*28).reshape(10,1,28,28)",
            "data_role": ROLE,
        },
        "action": "component references, then explicit 2048-grid image-to-OLS-to-Generator-to-CMOS zero-update producer runs and dependency reconstruction",
        "execution_device": str(device),
        "update_count": 0,
        "final_decision": "all required optical, integration, dependency-rebuild, and saved-readback checks must pass",
        "result_use": "implementation consistency only; no training gain, full TRAIN scale, readability, or physical deployment claim",
    }
    _write_json(target / "plan.json", plan)
    start = time.perf_counter()
    preflight = _component_preflight()
    _write_json(target / "preflight.json", preflight)
    if not preflight["passed"]:
        decision = "VALID_FAILED_COMPONENT_PRECHECK"
        verification = {"task_id": TASK_ID, "decision": decision, "preflight": preflight, "trainable_updates": 0}
        _write_json(target / "verification.json", verification)
        (target / "report.md").write_text(_report_md(decision, preflight, {}, {}), encoding="utf-8")
        return 2

    images = _synthetic_images()
    torch.manual_seed(SEED)
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=2048, ols_model=GAUSSIAN_OLS_MODEL)
    ).to(device).eval()
    runtime_before = _state_snapshot(runtime)
    direct_start = time.perf_counter()
    image_rows = []
    ols_input_rows = []
    encoder_output_rows = []
    z_rows = []
    generator_input_rows = []
    camera_rows = []
    plane_power_rows = []
    plane_specs: list[Mapping[str, Any]] | None = None
    writer_errors = []
    for image in images.split(1, dim=0):
        with torch.no_grad():
            result = runtime.forward_from_image(image.to(device))
        if result.ols_input_field is None or result.image_input is None or result.encoder_output_field is None:
            raise RuntimeError("Gaussian production image path omitted mandatory fields")
        image_rows.append(result.image_input.image.detach().cpu())
        ols_input_rows.append(result.ols_input_field.tensor.detach().cpu())
        encoder_output_rows.append(result.encoder_output_field.tensor.detach().cpu())
        z_rows.append(result.z_ols.coefficients.detach().cpu())
        generator_input_rows.append(result.generator_input_field.tensor.detach().cpu())
        camera_rows.append(result.raw_camera.tensor.detach().cpu())
        powers, current_specs = _independent_plane_powers(result)
        plane_power_rows.append(powers)
        if plane_specs is None:
            plane_specs = current_specs
        elif current_specs != plane_specs:
            raise RuntimeError("Gaussian runtime plane geometry changed between smoke rows")
        external_state = ComplexOLSState.from_external(
            result.z_ols.coefficients,
            basis_id=GAUSSIAN_OLS_MODEL,
            mode_geometry=runtime.ols.mode_identity(),
        )
        external_writer = runtime.w_ols(external_state)
        writer_errors.append(float((external_writer.tensor - result.generator_input_field.tensor).abs().max()))
        del result
    direct_seconds = time.perf_counter() - direct_start
    writer_error = max(writer_errors)
    runtime_images = torch.cat(image_rows, dim=0)
    ols_input = torch.cat(ols_input_rows, dim=0)
    encoder_output = torch.cat(encoder_output_rows, dim=0)
    z_ols = torch.cat(z_rows, dim=0)
    generator_input = torch.cat(generator_input_rows, dim=0)
    raw_cmos = torch.cat(camera_rows, dim=0)
    per_image_plane_power = torch.cat(plane_power_rows, dim=0)
    runtime_unchanged = _state_unchanged(runtime, runtime_before)
    direct_mode_geometry = runtime.ols.mode_identity()
    direct_propagation_grid_size = runtime.config.propagation_grid_size
    direct_encoder_layers = runtime.encoder.layers
    direct_generator_layers = runtime.generator.layers
    del runtime, runtime_before
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    bogt = LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device=device, propagation_grid_size=2048, ols_model=GAUSSIAN_OLS_MODEL
    )
    bogt_optimizer = bogt.create_optimizer()
    bogt_before = _state_snapshot(bogt)
    bogt_optimizer_before = copy.deepcopy(bogt_optimizer.state_dict())
    bogt_start = time.perf_counter()
    references = collect_bogt_train_references(
        bogt,
        images,
        data_role=TRAIN_SMOKE_ROLE,
        engineering_only=True,
        data_identity={"role": TRAIN_SMOKE_ROLE, "count": SYNTHETIC_IMAGE_COUNT, "purpose": ROLE},
    )
    bogt_seconds = time.perf_counter() - bogt_start
    bogt.bind_references(references)
    bogt_unchanged = _state_unchanged(bogt, bogt_before)
    bogt_optimizer_unchanged = _nested_equal(bogt_optimizer.state_dict(), bogt_optimizer_before)
    del bogt, bogt_optimizer, bogt_before, bogt_optimizer_before
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    v2 = LightGen2DComplexOLSTrainingConsumerV2(
        load_v2_training_contract(repo_root=root),
        device=device,
        propagation_grid_size=2048,
        ols_model=GAUSSIAN_OLS_MODEL,
    )
    v2_optimizer = v2.create_optimizer()
    v2_before = _state_snapshot(v2)
    v2_optimizer_before = copy.deepcopy(v2_optimizer.state_dict())
    v2_start = time.perf_counter()
    v2_scale = extract_train_only_ols_scale(
        v2,
        images,
        data_role=SCALE_SMOKE_CLASSIFICATION,
        checkpoint_identity={"role": ROLE, "initialization_seed": SEED, "purpose": "engineering_only"},
        data_identity={"role": ROLE, "count": SYNTHETIC_IMAGE_COUNT, "purpose": "engineering_only"},
        basis_id=GAUSSIAN_OLS_MODEL,
        mode_geometry=v2.runtime.ols.mode_identity(),
    )
    v2_seconds = time.perf_counter() - v2_start
    v2_scale_artifact = save_scale_artifact(target / "v2_scale.v2.pt", v2_scale)
    v2_scale_readback = load_scale_artifact(v2_scale_artifact.artifact_path)
    v2.set_scale(v2_scale_readback)
    external_prior_state, prior_sample = sample_physical_prior(
        v2_scale_readback,
        SYNTHETIC_IMAGE_COUNT,
        stage_local_step=1,
        device=device,
        role="gaussian_runtime_smoke",
    )
    prior_writer = v2.runtime.w_ols(external_prior_state)
    prior_modes = _independent_modes(v2.runtime.generator_entrance_plane, GaussianFiberGeometry())
    prior_expected = np.einsum("bm,mhw->bhw", _numpy(external_prior_state.coefficients), prior_modes)
    prior_writer_error = float(np.max(np.abs(_numpy(prior_writer.tensor)[:, 0] - prior_expected)))
    v2_unchanged = _state_unchanged(v2, v2_before)
    v2_optimizer_unchanged = _nested_equal(v2_optimizer.state_dict(), v2_optimizer_before)

    readout_path = target / "runtime_readout.pt"
    runtime_identity = v2.runtime_identity
    _save_runtime_readout(
        readout_path,
        images=runtime_images,
        ols_input=ols_input,
        encoder_output=encoder_output,
        z_ols=z_ols,
        generator_input=generator_input,
        raw_cmos=raw_cmos,
        per_image_plane_power=per_image_plane_power,
        plane_specs=plane_specs or [],
        refs=references,
        v2_scale=v2_scale,
        runtime_identity=runtime_identity,
        mode_geometry=direct_mode_geometry,
    )
    replay = _independent_replay(readout_path, images, references, v2_scale_readback)
    replay["v2_scale_artifact_roundtrip"] = {
        "executed": True,
        "passed": v2_scale_readback.basis_id == GAUSSIAN_OLS_MODEL
        and v2_scale_readback.mode_geometry == direct_mode_geometry
        and torch.equal(v2_scale_readback.values, v2_scale.values),
        "basis_id": v2_scale_readback.basis_id,
        "mode_geometry_matches_runtime": v2_scale_readback.mode_geometry == direct_mode_geometry,
        "artifact_path": str(v2_scale_readback.artifact_path),
    }
    replay["same_encoder_and_generator_writer"] = {
        "executed": True,
        "passed": writer_error <= COEFFICIENT_TOLERANCE and prior_writer_error <= COEFFICIENT_TOLERANCE,
        "maximum_absolute_field_error": writer_error,
        "basis_id": external_state.basis_id,
        "mode_geometry_matches_runtime": external_state.mode_geometry == direct_mode_geometry,
        "v2_generated_state_basis_id": external_prior_state.basis_id,
        "v2_generated_state_geometry_matches_runtime": external_prior_state.mode_geometry == direct_mode_geometry,
        "v2_generated_state_independent_writer_max_error": prior_writer_error,
        "v2_prior_sample_classification": prior_sample.role,
    }
    replay["zero_update_state_invariants"] = {
        "executed": True,
        "passed": runtime_unchanged and bogt_unchanged and v2_unchanged and bogt_optimizer_unchanged and v2_optimizer_unchanged,
        "runtime_parameters_and_buffers_unchanged": runtime_unchanged,
        "bogt_parameters_and_buffers_unchanged": bogt_unchanged,
        "v2_parameters_and_buffers_unchanged": v2_unchanged,
        "bogt_optimizer_unchanged": bogt_optimizer_unchanged,
        "v2_optimizer_unchanged": v2_optimizer_unchanged,
        "trainable_updates": 0,
    }
    _write_json(target / "independent_replay.json", replay)
    _write_json(
        target / "reconstructed_dependencies.json",
        {
            "classification": ROLE,
            "engineering_only": True,
            "formal_train_scale": False,
            "image_count": SYNTHETIC_IMAGE_COUNT,
            "initialization_seed": SEED,
            "basis_id": GAUSSIAN_OLS_MODEL,
            "mode_geometry": direct_mode_geometry,
            "bogt_scale": references.scale.tolist(),
            "bogt_power_reference": references.power_reference.tolist(),
            "v2_scale": v2_scale.values.tolist(),
            "v2_scale_identity": v2_scale.identity(),
            "bogt_reference_identity": references.identity(),
            "v2_inverse_map_max_absolute_error": replay["v2_normalized_inverse_mapping"]["max_absolute_complex_error"],
        },
    )
    runtime_section = {
        **runtime_identity,
        "execution_device": str(device),
        "propagation_grid_size": direct_propagation_grid_size,
        "encoder_layers": direct_encoder_layers,
        "generator_layers": direct_generator_layers,
        "architecture": "2+2",
    }
    verification = {
        "task_id": TASK_ID,
        "decision": "GAUSSIAN_INTEGRATION_CONSISTENT"
        if all(item.get("passed", False) for item in replay.values())
        else "VALID_FAILED_GAUSSIAN_INTEGRATION",
        "checks": replay,
        "preflight_passed": preflight["passed"],
        "runtime": runtime_section,
        "roles": {"classification": ROLE, "engineering_only": True, "formal_train_rows": 55_000},
        "updates": {"trainable_updates": 0, "full_train_rebuild": False},
        "timing_seconds_observed": {
            "direct_full_runtime": direct_seconds,
            "bogt_reference_producer": bogt_seconds,
            "v2_scale_producer": v2_seconds,
            "total_elapsed": time.perf_counter() - start,
        },
    }
    _write_json(target / "verification.json", verification)
    report = _report_md(verification["decision"], preflight, replay, runtime_section)
    (target / "report.md").write_text(report, encoding="utf-8")
    _write_json(target / "report.json", {"verification": verification, "report_markdown": report})
    return 0 if verification["decision"] == "GAUSSIAN_INTEGRATION_CONSISTENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
