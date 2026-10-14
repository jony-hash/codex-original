"""Zero-update CPU verification of an opt-in small Gaussian OLS geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mpmath
import numpy as np
import torch

from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import (
    GaussianFiberArrayOLS, GaussianFiberGeometry, MODE_NORM_TOLERANCE, PASSIVITY_TOLERANCE,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    GAUSSIAN_OLS_MODEL,
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec


TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_FIBER_GEOMETRY_V1"
COEFFICIENT_TOLERANCE = 1.0e-5
POWER_TOLERANCE = 2.0e-5
COARSE_TILT_TOLERANCE = 1.0e-4
FINE_TILT_TOLERANCE = 1.0e-6
GRADIENT_TOLERANCE = 5.0e-4
SOURCE_SIDE_M = 20.0e-6
SELECTED_PORT = 44
COARSE_ANGLES = (0.0, -0.5, 0.5, -1.0, 1.0, -2.5, 2.5, -5.0, 5.0)
FINE_ANGLES = (0.0, -2.5, 2.5, -5.0, 5.0, -7.5, 7.5, -10.0, 10.0, -20.0, 20.0, -40.0, 40.0, -60.0, 60.0, -80.0, 80.0)


def plane(size: int = 100) -> PlaneSpec:
    return PlaneSpec("gaussian_geometry_input", (size, size), (200.0e-6 / size,) * 2, 532.0e-9, 0.0, "ols")


def reference_gram(p: PlaneSpec, g: GaussianFiberGeometry) -> np.ndarray:
    """Independent finite-FOV analytical Gaussian product integral."""
    centers = (np.arange(10) - 4.5) * g.port_pitch_m
    axes = []
    for fov in p.fov_m:
        values = np.empty((10, 10))
        for i, ci in enumerate(centers):
            for j, cj in enumerate(centers):
                midpoint = (ci + cj) / 2.0
                fraction = 0.5 * (math.erf(math.sqrt(2.0) * (fov / 2.0 - midpoint) / g.waist_m) - math.erf(math.sqrt(2.0) * (-fov / 2.0 - midpoint) / g.waist_m))
                values[i, j] = math.exp(-(ci - cj)**2 / (2.0 * g.waist_m**2)) * fraction
        axes.append(values)
    return np.kron(*axes)


def reference_fields(p: PlaneSpec, g: GaussianFiberGeometry) -> np.ndarray:
    """Independent 2-D NumPy sampling; never reads the operator mode buffers."""
    y = (np.arange(p.shape[0]) + 0.5) * p.pixel_pitch_m[0] - p.fov_m[0] / 2.0
    x = (np.arange(p.shape[1]) + 0.5) * p.pixel_pitch_m[1] - p.fov_m[1] / 2.0
    cy, cx = np.meshgrid((np.arange(10) - 4.5) * g.port_pitch_m, (np.arange(10) - 4.5) * g.port_pitch_m, indexing="ij")
    radius = (y[None, :, None] - cy.reshape(-1, 1, 1))**2 + (x[None, None, :] - cx.reshape(-1, 1, 1))**2
    return math.sqrt(2.0 / math.pi) / g.waist_m * np.exp(-radius / g.waist_m**2)


def _field(tensor: torch.Tensor, p: PlaneSpec) -> ComplexField:
    return ComplexField(tensor, p, OperationProvenance("independent_gaussian_test_input", None, None))


def _measure(op: GaussianFiberArrayOLS, tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = _field(tensor, op.input_plane)
    coefficients = op.project_coefficients(field)
    output = op(field)
    area = math.prod(op.input_plane.pixel_pitch_m)
    pin = tensor.abs().square().sum((-2, -1)) * area
    pout = output.tensor.abs().square().sum((-2, -1)) * area
    return coefficients.detach().numpy(), pin.detach().numpy(), pout.detach().numpy()


def modal_checks(op: GaussianFiberArrayOLS) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    p, g = op.input_plane, op.geometry
    modes = reference_fields(p, g)
    gram = reference_gram(p, g)
    phase = np.exp(1j * (np.arange(100) * 0.37))
    weights = np.diag(phase / np.sqrt(gram.diagonal()))
    single = torch.from_numpy(modes * (phase / np.sqrt(gram.diagonal()))[:, None, None]).to(torch.complex64)
    expected_single = weights @ gram.T
    actual_single, single_pin, single_pout = _measure(op, single)
    pair_weights = np.zeros((3, 100), dtype=np.complex128)
    for row, phi in enumerate((0.0, math.pi / 2.0, math.pi)):
        pair_weights[row, SELECTED_PORT] = 0.6
        pair_weights[row, SELECTED_PORT + 1] = 0.8 * np.exp(1j * phi)
        pair_weights[row] /= math.sqrt(float(np.real(pair_weights[row].conj() @ gram @ pair_weights[row])))
    pair = torch.from_numpy(np.einsum("bm,mhw->bhw", pair_weights, modes)).to(torch.complex64)
    expected_pair = pair_weights @ gram.T
    actual_pair, pair_pin, pair_pout = _measure(op, pair)
    checks = {}
    for name, actual, expected, pin, pout in (
        ("single_mode_match", actual_single, expected_single, single_pin, single_pout),
        ("two_port_coherence", actual_pair, expected_pair, pair_pin, pair_pout),
    ):
        output_reference = np.real(np.einsum("bm,mn,bn->b", expected.conj(), gram, expected))
        coefficient_error = float(np.max(np.abs(actual - expected)))
        input_error = float(np.max(np.abs(pin - 1.0)))
        output_error = float(np.max(np.abs(pout - output_reference)))
        gain = float(np.max(pout / pin))
        checks[name] = {
            "executed": True,
            "passed": coefficient_error <= COEFFICIENT_TOLERANCE and input_error <= POWER_TOLERANCE and output_error <= POWER_TOLERANCE and gain <= 1.0 + PASSIVITY_TOLERANCE,
            "input_count": len(actual), "coefficient_max_absolute_error": coefficient_error,
            "input_power_max_absolute_error": input_error, "output_power_max_absolute_error": output_error,
            "max_measured_power_gain": gain,
        }
    readouts = {"single_coefficients": torch.from_numpy(actual_single), "single_reference": torch.from_numpy(expected_single),
                "pair_coefficients": torch.from_numpy(actual_pair), "pair_reference": torch.from_numpy(expected_pair)}
    return checks, readouts


def reference_tilt(theta_degrees: float, g: GaussianFiberGeometry, wavelength_m: float = 532.0e-9) -> complex:
    """S25-S27 evaluated with an independent arbitrary-precision complex erf."""
    with mpmath.workdps(60):
        w, side = mpmath.mpf(g.waist_m), mpmath.mpf(SOURCE_SIDE_M)
        kx = 2 * mpmath.pi / wavelength_m * mpmath.sin(mpmath.mpf(theta_degrees) * mpmath.pi / 180)
        a = side / (2 * w)
        amplitude = w * mpmath.sqrt(2 * mpmath.pi) / side * mpmath.erf(a) * mpmath.exp(-(kx * w)**2 / 4) * mpmath.re(mpmath.erf(a + 1j * kx * w / 2))
        return complex(amplitude * mpmath.exp(0.37j))


def tilt_input(op: GaussianFiberArrayOLS, theta_radians: torch.Tensor) -> torch.Tensor:
    p = op.input_plane
    cy, cx = op.centers_m[SELECTED_PORT]
    y = (torch.arange(p.shape[0], dtype=torch.float64) + 0.5) * p.pixel_pitch_m[0] - p.fov_m[0] / 2.0 - cy
    x = (torch.arange(p.shape[1], dtype=torch.float64) + 0.5) * p.pixel_pitch_m[1] - p.fov_m[1] / 2.0 - cx
    aperture_y = (y >= -SOURCE_SIDE_M / 2.0) & (y < SOURCE_SIDE_M / 2.0)
    aperture_x = (x >= -SOURCE_SIDE_M / 2.0) & (x < SOURCE_SIDE_M / 2.0)
    phase = 2.0 * math.pi / p.wavelength_m * theta_radians.sin() * x + 0.37
    return (aperture_y[:, None] * aperture_x[None, :] * torch.exp(1j * phase)[None, :] / SOURCE_SIDE_M).unsqueeze(0)


def tilt_checks(op: GaussianFiberArrayOLS, angles: tuple[float, ...], *, precision: str) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor]:
    tolerance = COARSE_TILT_TOLERANCE if precision == "small_FP32" else FINE_TILT_TOLERANCE
    rows = []
    coefficients = []
    for theta in angles:
        tensor = tilt_input(op, torch.tensor(math.radians(theta), dtype=torch.float64))
        if precision == "small_FP32":
            tensor = tensor.to(torch.complex64)
        actual, pin, pout = _measure(op, tensor)
        reference = reference_tilt(theta, op.geometry)
        c = complex(actual[0, SELECTED_PORT])
        eta, eta_ref = abs(c)**2 / float(pin[0]), abs(reference)**2
        rows.append({"precision": precision, "angle_degrees": theta, "coefficient_real": c.real, "coefficient_imag": c.imag,
                     "reference_real": reference.real, "reference_imag": reference.imag,
                     "coefficient_absolute_error": abs(c - reference), "efficiency": eta, "reference_efficiency": eta_ref,
                     "efficiency_absolute_error": abs(eta - eta_ref), "input_power": float(pin[0]), "output_power": float(pout[0])})
        coefficients.append(torch.from_numpy(actual))
    coefficient_error = max(row["coefficient_absolute_error"] for row in rows)
    efficiency_error = max(row["efficiency_absolute_error"] for row in rows)
    normalization_error = max(abs(row["input_power"] - 1.0) for row in rows)
    passed = coefficient_error <= tolerance and efficiency_error <= tolerance and normalization_error <= POWER_TOLERANCE
    return {"executed": True, "passed": passed, "input_count": len(rows),
            "coefficient_max_absolute_error": coefficient_error, "efficiency_max_absolute_error": efficiency_error,
            "input_power_max_absolute_error": normalization_error, "absolute_tolerance": tolerance}, rows, torch.cat(coefficients)


def gradient_check(op: GaussianFiberArrayOLS) -> dict[str, Any]:
    theta = torch.tensor(math.radians(2.5), dtype=torch.float64, requires_grad=True)
    coefficient = op.project_coefficients(_field(tilt_input(op, theta), op.input_plane))[0, SELECTED_PORT]
    efficiency = coefficient.abs().square()
    actual = float(torch.autograd.grad(efficiency, theta)[0])
    h = 1.0e-5
    values = []
    for sign in (-1, 1):
        perturbed = torch.tensor(float(theta.detach()) + sign * h, dtype=torch.float64)
        c = op.project_coefficients(_field(tilt_input(op, perturbed), op.input_plane))[0, SELECTED_PORT]
        values.append(float(c.abs().square()))
    finite_difference = (values[1] - values[0]) / (2.0 * h)
    with mpmath.workdps(60):
        w, side, wavelength = map(mpmath.mpf, (op.geometry.waist_m, SOURCE_SIDE_M, op.input_plane.wavelength_m))
        def analytical_efficiency(angle):
            kx = 2 * mpmath.pi / wavelength * mpmath.sin(angle)
            a = side / (2 * w)
            c = w * mpmath.sqrt(2 * mpmath.pi) / side * mpmath.erf(a) * mpmath.exp(-(kx * w)**2 / 4) * mpmath.re(mpmath.erf(a + 1j * kx * w / 2))
            return c**2
        reference = float(mpmath.diff(analytical_efficiency, mpmath.mpf(math.radians(2.5))))
    reference_error = abs(actual - reference) / abs(reference)
    fd_error = abs(actual - finite_difference) / abs(reference)
    return {"executed": True, "passed": reference_error <= GRADIENT_TOLERANCE and fd_error <= GRADIENT_TOLERANCE and actual < 0 and reference < 0,
            "autograd_d_eta_d_theta_radian": actual, "finite_difference": finite_difference, "analytical_derivative": reference,
            "reference_relative_error": reference_error, "finite_difference_relative_error": fd_error}


def negative_geometry_check(p: PlaneSpec) -> dict[str, Any]:
    bad = GaussianFiberGeometry(port_pitch_m=160.0e-6 / 9.0, waist_m=7.111e-6)
    try:
        GaussianFiberArrayOLS(p, bad)
    except ValueError as exc:
        if "non-passive" not in str(exc):
            raise
        s = math.exp(-bad.port_pitch_m**2 / (2.0 * bad.waist_m**2))
        return {"executed": True, "passed": True, "rejection": str(exc),
                "geometry": asdict(bad), "independent_two_port_power_gain": (1.0 + s)**2}
    return {"executed": True, "passed": False, "rejection": None}


def decision(checks: dict[str, Any]) -> str:
    required = {"single_mode_match", "two_port_coherence", "small_tilt", "refined_tilt", "tilt_gradient", "old_geometry_rejected", "small_runtime_writer"}
    if set(checks) != required:
        raise ValueError("final Gaussian geometry decision requires the complete measurement")
    return "GEOMETRY_NUMERICALLY_SUPPORTED" if all(c["executed"] and c["passed"] for c in checks.values()) else "GEOMETRY_NOT_SUPPORTED"


def small_writer_check(geometry: GaussianFiberGeometry) -> tuple[dict[str, Any], LightGen2DSemanticGenerationInputOLSAlignment, dict[str, torch.Tensor]]:
    legacy_runtime = LightGen2DSemanticGenerationInputOLSAlignment()
    legacy_mode_model = legacy_runtime.ols.operator.spec.mode_model
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(ols_model=GAUSSIAN_OLS_MODEL)
    )
    if runtime.ols.operator.geometry != geometry:
        raise ValueError("small writer check requires the explicit default Gaussian research geometry")
    frozen = {name: value.clone() for name, value in runtime.state_dict().items()}
    phase = complex(math.cos(0.37), math.sin(0.37))
    probe = torch.from_numpy(reference_fields(plane(), geometry)[[SELECTED_PORT]] * phase).to(torch.complex64)
    state = runtime.ols.project_state(_field(probe, runtime.ols_plane))
    emitted = runtime.w_ols(state)
    reference = reference_gram(plane(), geometry)[:, SELECTED_PORT] * phase
    error = float(np.max(np.abs(state.coefficients.detach().numpy()[0] - reference)))
    direct = runtime.ols.operator(_field(probe, runtime.ols_plane))
    writer_error = float((emitted.tensor - direct.tensor).abs().max())
    explicit_model = runtime.ols.basis_id == GAUSSIAN_OLS_MODEL and isinstance(runtime.ols.operator, GaussianFiberArrayOLS)
    writer = {"executed": True, "passed": explicit_model and error <= COEFFICIENT_TOLERANCE and writer_error == 0.0,
              "coefficient_max_absolute_error": error, "writer_tensor_max_absolute_error": writer_error,
              "explicit_gaussian_runtime": explicit_model,
              "propagation_grid": runtime.config.propagation_grid_size, "encoder_layers": runtime.encoder.layers,
              "generator_layers": runtime.generator.layers, "legacy_default_mode_model": legacy_mode_model,
              "scope": "OLS projection and generator-input writer only; no encoder/generator/CMOS forward"}
    return writer, runtime, frozen


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    target = args.output_dir.resolve()
    allowed = root / "outputs" / "20261012_lightgen_gaussian_fiber_geometry_v1" / "runs"
    if target.parent != allowed or target.exists() or not target.name.startswith("diagnostic-only_"):
        raise ValueError("output must be a new diagnostic-only child of the named versioned run directory")
    target.mkdir(parents=True)
    torch.set_num_threads(2)
    geometry = GaussianFiberGeometry()
    tolerances = {"mode_norm": MODE_NORM_TOLERANCE, "complete_operator_power_gain": PASSIVITY_TOLERANCE,
                  "modal_coefficient": COEFFICIENT_TOLERANCE, "power": POWER_TOLERANCE,
                  "small_tilt": COARSE_TILT_TOLERANCE, "refined_tilt": FINE_TILT_TOLERANCE, "gradient_relative": GRADIENT_TOLERANCE}
    plan = {"task_id": TASK_ID, "question": "Does a full Gaussian OLS have valid geometry, complex coupling, coherent re-emission, and angle response on the small interface?",
            "authorization": "User requested project geometry and three independent examples; no measured hardware parameters available.",
            "geometry": asdict(geometry), "small_plane": plane().to_dict(), "refined_plane": plane(1600).to_dict(),
            "action": "zero-update CPU component verification through the opt-in Gaussian producer",
            "tolerances": tolerances, "final_decision": "all required independently referenced measurements must pass",
            "result_use": {"supported": "Retain this uncalibrated Gaussian component candidate for later small-architecture research.",
                           "not_supported": "Reject this geometry for subsequent use; retain all measurements.",
                           "engineering_error": "No valid geometry conclusion; repair and recheck affected path."},
            "stop": "No training, checkpoint conversion, prior refit, model adoption, or switch of the existing LP01 default.",
            "documentation_write_set": [], "physical_parameter_status": "PROJECT_GEOMETRY_UNMEASURED"}
    _write_json(target / "plan.json", plan)
    op = GaussianFiberArrayOLS(plane(), geometry)
    fine = GaussianFiberArrayOLS(plane(1600), geometry)
    writer, runtime, frozen = small_writer_check(geometry)
    preflight = {"task_id": TASK_ID, "status": "READY" if writer["passed"] else "INVALID_HARNESS",
                 "executed": True, "passed": writer["passed"], "component_geometry": op.geometry_diagnostics,
                 "representative_small_writer": writer, "scientific_updates": 0}
    _write_json(target / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Gaussian component production preflight failed")
    checks, readouts = modal_checks(op)
    checks["small_tilt"], coarse_rows, readouts["small_tilt_coefficients"] = tilt_checks(op, COARSE_ANGLES, precision="small_FP32")
    checks["refined_tilt"], fine_rows, readouts["refined_tilt_coefficients"] = tilt_checks(fine, FINE_ANGLES, precision="refined_FP64")
    checks["tilt_gradient"] = gradient_check(op)
    checks["old_geometry_rejected"] = negative_geometry_check(plane())
    checks["small_runtime_writer"] = writer
    if any(not torch.equal(value, runtime.state_dict()[name]) for name, value in frozen.items()):
        raise RuntimeError("small runtime parameters or buffers changed during component verification")
    if any(parameter.grad is not None for parameter in runtime.parameters()):
        raise RuntimeError("small runtime acquired gradients")
    classification = decision(checks)
    rows = coarse_rows + fine_rows
    with (target / "tilt_rows.csv").open("x", encoding="utf-8", newline="") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    with (target / "port_centers.csv").open("x", encoding="utf-8", newline="") as stream:
        csv_writer = csv.writer(stream)
        csv_writer.writerow(("port", "row", "column", "y_m", "x_m"))
        for j, center in enumerate(op.centers_m.tolist()):
            csv_writer.writerow((j, j // 10, j % 10, *center))
    report = {"task_id": TASK_ID, "status": "DIAGNOSTIC_COMPLETED", "classification": classification,
              "geometry": asdict(geometry), "small_plane": plane().to_dict(), "refined_plane": plane(1600).to_dict(),
              "geometry_diagnostics": {"small": op.geometry_diagnostics, "refined": fine.geometry_diagnostics},
              "checks": checks, "tolerances": tolerances, "source_side_m": SOURCE_SIDE_M,
              "source_definition": "Unit-power square illumination centered on port 44; tilt exp(i*k*sin(theta)*(x-x44)); global phase 0.37 rad.",
              "source_side_role": "Test illumination width; independent of fiber length and cladding diameter.",
              "physical_parameter_status": "PROJECT_GEOMETRY_UNMEASURED", "model_adoption": False,
              "training_updates": 0, "encoder_generator_cmos_forwards": 0, "small_state_unchanged": True,
              "training_recommendation": "NONE", "next_action": "NONE"}
    readouts["small_plane_metadata"] = plane().to_dict()
    readouts["refined_plane_metadata"] = plane(1600).to_dict()
    readouts["geometry"] = asdict(geometry)
    with (target / "readouts.pt").open("xb") as stream:
        torch.save(readouts, stream)
    loaded = torch.load(target / "readouts.pt", map_location="cpu", weights_only=True)
    if any(not torch.equal(value, loaded[name]) for name, value in readouts.items() if isinstance(value, torch.Tensor)):
        raise RuntimeError("saved Gaussian coefficient evidence did not replay")
    _write_json(target / "report.json", report)
    with (target / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(f"**结果与状态：** {classification}；CPU、零更新、未标定项目几何。\n\n"
                     "**科学认识：** 20 μm端口间距、3.5 μm Gaussian 1/e振幅半径；保留S12解析归一化，无逐端口裁切。单模、相干双端口及倾斜平面波由独立解析公式核对。\n\n"
                     "**目标缺口：** 尚无实测MFD、端口间距或端面倍率；不代表论文实物复现、生成收益或物理验收。\n\n"
                     "**工程完成与验证：** 100个单模输入、3个相干双端口输入、9个原网格倾斜输入、17个局部细网格倾斜输入，及梯度、原几何负例和小架构OLS写入检查。数值与判定见report.json，角度逐行数据见tilt_rows.csv；源码测试另行记录。\n\n"
                     "**决策影响与下一步：** 所有检查通过才保留本未标定候选；未通过则不支持其后继使用。仅完成本轮组件验证；现有LP01默认及训练检查点未切换，无训练或采用授权。\n")
    print(json.dumps({"classification": classification, "output": str(target), "checks": checks}, ensure_ascii=False, allow_nan=False))
    return 0 if classification == "GEOMETRY_NUMERICALLY_SUPPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
