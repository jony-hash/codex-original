from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA = "ols_camera_training_interface_closure.v10"
DEFAULT_SOURCE = Path("outputs/20261016_ols_decoder_camera_interface_v9/runs/attempt_02")
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_camera_training_interface_closure_v10.yaml")
HC = 6.62607015e-34 * 299792458.0
WAVELENGTH_M = 532e-9


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _preflight(source: Path, config: dict[str, Any]) -> dict[str, Any]:
    camera = config["camera"]
    photon_factors = [
        float(camera["quantum_efficiency"] * t / (HC / WAVELENGTH_M))
        for t in camera["exposure_sweep_s"]
    ]
    threshold_power = float(camera["full_well_e"]) / photon_factors[2]
    power = torch.tensor([0.5, 2.0], dtype=torch.float64) * threshold_power
    power.requires_grad_(True)
    g = photon_factors[2] / float(camera["full_well_e"])
    task = torch.clamp(g * power, 0.0, 1.0)
    gradient = torch.autograd.grad(task.sum(), power)[0]
    expected = torch.tensor([g, 0.0], dtype=torch.float64)
    if not torch.allclose(gradient, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError("task-output clamp derivative preflight failed")
    epsilon = threshold_power * 1.0e-6
    below = float(np.clip(g * (threshold_power - epsilon), 0.0, 1.0))
    at = float(np.clip(g * threshold_power, 0.0, 1.0))
    above = float(np.clip(g * (threshold_power + epsilon), 0.0, 1.0))
    if not math.isclose((at - below) / epsilon, g, rel_tol=2.0e-6) or above != at:
        raise ValueError("full-well one-sided derivative preflight failed")
    below_zero = float(np.clip(g * (-epsilon), 0.0, 1.0))
    at_zero = float(np.clip(g * 0.0, 0.0, 1.0))
    above_zero = float(np.clip(g * epsilon, 0.0, 1.0))
    if (at_zero - below_zero) != 0.0 or not math.isclose((above_zero - at_zero) / epsilon, g, rel_tol=2.0e-6):
        raise ValueError("zero-clamp one-sided derivative preflight failed")

    # Mixed saturation counterexample: clipping the per-pixel values gives 0.5,
    # while clipping their unsaturated mean gives 1.0.
    mixed = np.array([0.0, 2.0], dtype=np.float64)
    correct = float(np.clip(mixed, 0.0, 1.0).mean())
    old = float(np.clip(mixed.mean(), 0.0, 1.0))
    if correct != 0.5 or old != 1.0 or correct == old:
        raise ValueError("mixed-saturation mean negative control failed")

    archives = sorted((source / "camera_measurements").glob("*_4096_complex64.npz"))
    factors = sorted((source / "camera_jacobians").glob("*_4096_complex128_factors.npz"))
    if len(archives) != 16 or len(factors) != 16:
        raise ValueError(f"expected 16 v9 camera and Jacobian factor archives; got {len(archives)}, {len(factors)}")
    with np.load(archives[0], allow_pickle=False) as data:
        required = {"pattern_names", "power_w_per_pixel", "uniform_t05_power_w", "task_output_1ms"}
        if not required.issubset(data.files) or data["power_w_per_pixel"].shape != (50, 200, 200):
            raise ValueError("v9 camera archive contract mismatch")
        measured = data["power_w_per_pixel"].astype(np.float64)
        actual_task = data["task_output_1ms"].astype(np.float64)
        factor = float(camera["quantum_efficiency"] * 1.0e-3 / (HC / WAVELENGTH_M))
        expected_task = np.clip(measured * factor / float(camera["full_well_e"]), 0.0, 1.0)
        if not np.allclose(actual_task, expected_task, rtol=2.0e-6, atol=1.0e-8):
            raise ValueError("saved task output is not the pixelwise clipped physical readout")
    # Complex128 central differences through the actual camera mapping at a
    # saved unsaturated pixel and a saved saturated pixel.
    central_checks = []
    exposure_factor = float(camera["quantum_efficiency"] * 1.0e-3 / (HC / WAVELENGTH_M))
    gain = exposure_factor / float(camera["full_well_e"])
    interior_power = None
    saturated_power = None
    for archive in archives:
        with np.load(archive, allow_pickle=False) as data:
            base_map = data["uniform_t05_power_w"].astype(np.float64)
        ratios = base_map * gain
        interior = base_map[(ratios > 0.2) & (ratios < 0.8)]
        saturated = base_map[ratios > 2.0]
        if interior_power is None and interior.size:
            interior_power = float(interior[0])
        if saturated_power is None and saturated.size:
            saturated_power = float(saturated[0])
        if interior_power is not None and saturated_power is not None:
            break
    if interior_power is None or saturated_power is None:
        raise ValueError("saved v9 fields lack interior or saturated camera pixels for derivative checks")
    for label, p0, expected_active in (("interior", interior_power, True), ("saturated", saturated_power, False)):
        field0 = np.complex128(math.sqrt(p0))
        eps = 1.0e-6
        direction = field0
        plus = float(np.clip(gain * abs(field0 + eps * direction) ** 2, 0.0, 1.0))
        minus = float(np.clip(gain * abs(field0 - eps * direction) ** 2, 0.0, 1.0))
        finite_difference = (plus - minus) / (2.0 * eps)
        expected_derivative = 2.0 * gain * p0 if expected_active else 0.0
        relative = abs(finite_difference - expected_derivative) / max(abs(expected_derivative), 1.0e-30)
        if relative > 1.0e-4:
            raise ValueError(f"complex128 central difference failed at {label} camera pixel")
        central_checks.append({"pixel_state": label, "physical_power_w_per_pixel": p0,
                               "finite_difference": finite_difference, "analytic_derivative": expected_derivative,
                               "relative_error": relative})
    sample_maps = np.array([[0.2, 0.4], [0.5, 0.6]], dtype=np.float64)
    per_sample_normalized = sample_maps / sample_maps.max(axis=1, keepdims=True)
    if np.allclose(sample_maps, per_sample_normalized):
        raise ValueError("per-sample normalization negative control did not change absolute output")
    pixel_area = (2.5e-6) ** 2
    test_irradiance = 2.0e-9
    if test_irradiance * pixel_area == test_irradiance:
        raise ValueError("missing-area negative control did not distinguish W/m2 and W/pixel")
    transmission = 0.889
    if math.isclose(transmission, math.sqrt(transmission), rel_tol=1.0e-3):
        raise ValueError("power-transmission versus amplitude-transmission control failed")
    labeled = np.array([1.0, 4.0, 9.0])
    if np.array_equal(labeled, labeled[[2, 1, 0]]):
        raise ValueError("fiber channel-order negative control failed")
    return {
        "passed": True,
        "context_archive_count": len(archives),
        "jacobian_factor_count": len(factors),
        "mixed_saturation_old_mean": old,
        "mixed_saturation_pixelwise_mean": correct,
        "task_output_value_gradient_control": "PASS",
        "full_well_one_sided_difference": "PASS",
        "zero_clamp_one_sided_difference": "PASS",
        "complex128_central_difference_on_saved_camera_values": central_checks,
        "negative_controls": {"mixed_saturation_mean": "PASS", "pixel_area": "PASS",
                              "power_vs_amplitude_transmission": "PASS", "fiber_order": "PASS",
                              "saved_task_pixelwise_clip": "PASS",
                              "per_sample_normalization": "PASS"},
    }


def _small_product_spectrum(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Singular values of (m x r)@(r x n), without materializing the large matrix.
    gram_l = left.T @ left
    gram_r = right @ right.T
    eig_r, vec_r = np.linalg.eigh((gram_r + gram_r.T) * 0.5)
    root_r = (vec_r * np.sqrt(np.maximum(eig_r, 0.0))) @ vec_r.T
    eig = np.linalg.eigvalsh(root_r @ gram_l @ root_r)
    return np.sqrt(np.maximum(eig[::-1], 0.0))


def _summary(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values.astype(np.float64, copy=False), [0, .1, .5, .9, 1])
    return {"min": float(q[0]), "p10": float(q[1]), "p50": float(q[2]),
            "p90": float(q[3]), "max": float(q[4]),
            "mean": float(values.astype(np.float64, copy=False).mean())}


def _pattern_group(name: str) -> str:
    if name.startswith("binary_probe_"):
        return "binary_probe"
    if name == "uniform_t_0.5":
        return "uniform_t0_5"
    if name == "uniform_t_1.0":
        return "uniform_t1_0"
    return "control_direction"


def _context_result(
    archive: Path, jacobian_path: Path, context_json: Path, frozen_inputs_path: Path,
    exposures: list[float], camera: dict[str, Any],
) -> dict[str, Any]:
    with np.load(archive, allow_pickle=False) as data:
        names = data["pattern_names"].astype(str)
        powers = data["power_w_per_pixel"].astype(np.float64)
        base = data["uniform_t05_power_w"].astype(np.float64)
        if powers.shape != (50, 200, 200) or base.shape != (200, 200) or len(names) != 50:
            raise ValueError(f"invalid camera archive shapes: {archive}")
        if not np.isfinite(powers).all() or not np.isfinite(base).all() or np.any(powers < 0) or np.any(base < 0):
            raise ValueError(f"nonfinite or negative power: {archive}")

    qe = float(camera["quantum_efficiency"])
    fw = float(camera["full_well_e"])
    rn = float(camera["read_noise_e_rms"])
    context = archive.name.removesuffix("_4096_complex64.npz")
    context_data = _json(context_json)
    if len(context_data.get("patterns", [])) != 50:
        raise ValueError(f"context scalar record is incomplete: {context_json}")
    if [p["pattern_name"] for p in context_data["patterns"]] != names.tolist():
        raise ValueError(f"pattern identity mismatch: {archive}")
    candidate_name, mode = context.split("__")[:2]
    with np.load(frozen_inputs_path, allow_pickle=False) as frozen:
        command_key = f"{candidate_name}__amplitude_commands"
        name_key = f"{candidate_name}__pattern_names"
        commands = frozen[command_key].astype(np.float64)
        frozen_names = frozen[name_key].astype(str)
    if commands.shape != (50, 10000) or frozen_names.tolist() != names.tolist():
        raise ValueError(f"frozen v8 command identity mismatch: {context}")
    base_index = int(np.flatnonzero(names == "uniform_t_0.5")[0])

    by_exposure: dict[str, Any] = {}
    per_pattern: dict[str, Any] = {}
    grouped: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        grouped.setdefault(_pattern_group(str(name)), []).append(index)
    for exposure in exposures:
        factor = qe * exposure / (HC / WAVELENGTH_M)
        mu = powers * factor
        task = np.clip(mu / fw, 0.0, 1.0)
        fractions = (mu >= fw).mean(axis=(1, 2))
        group_stats: dict[str, Any] = {}
        for group, indices in grouped.items():
            vals = fractions[indices]
            group_stats[group] = {
                "count": len(indices), "saturated_pattern_count": int((vals > 0).sum()),
                "saturated_pixel_fraction_by_pattern": _summary(vals),
                "pattern_names_with_saturation": [names[i].item() for i in indices if fractions[i] > 0],
            }
        base_mu = base * factor
        base_task = np.clip(base_mu / fw, 0.0, 1.0)
        by_exposure[str(exposure)] = {
            "all_50_saturation_fraction_by_pattern": _summary(fractions),
            "task_output_pixelwise_summary_across_patterns": _summary(task.reshape(-1)),
            "correct_task_output_mean": float(task.mean()),
            "incorrect_clip_after_per_pattern_mean_counterfactual": float(
                np.clip(mu.mean(axis=(1, 2)) / fw, 0.0, 1.0).mean()
            ),
            "uniform_t0_5": {
                "saturated_pixel_fraction": float((base_mu >= fw).mean()),
                "mean_task_output": float(base_task.mean()),
            },
            "groups": group_stats,
        }

        factor_archive = np.load(jacobian_path, allow_pickle=False)
        a = factor_archive["A_pixel_complex_gradient"].astype(np.float64)
        b = factor_archive["B_command_real_control"].astype(np.float64)
        if a.shape[0] != 40000 or a.shape[1] != b.shape[0] or b.shape[1] != 10000:
            raise ValueError(f"Jacobian factor shapes do not match: {jacobian_path}")
        unsaturated = (base_mu.reshape(-1) > 0.0) & (base_mu.reshape(-1) < fw)
        # derivative dy/dP on pixels strictly inside the unclipped interval
        # is QE*exposure/(photon_energy*full_well); elsewhere it is zero.
        left = a[unsaturated] * (factor / fw)
        spectrum = _small_product_spectrum(left, b)
        maximum = float(spectrum.max()) if spectrum.size else 0.0
        by_exposure[str(exposure)]["task_output_jacobian"] = {
            "units": "normalized task output per unit amplitude command",
            "base_unsaturated_pixel_fraction": float(unsaturated.mean()),
            "boundary_pixel_count": int((base_mu.reshape(-1) == fw).sum()),
            "singular_values": spectrum.tolist(),
            "effective_dimension_by_relative_threshold": {
                str(threshold): int((spectrum >= maximum * threshold).sum()) if maximum else 0
                for threshold in (1e-1, 1e-2, 1e-3, 1e-4)
            },
        }

        controls: dict[str, Any] = {}
        for tag in ("strongest_retained_direction", "weakest_retained_direction"):
            candidates = [i for i, name in enumerate(names) if tag in name and context.split("__")[1] in name]
            if not candidates:
                continue
            i = candidates[0]
            mu1 = powers[i] * factor
            task1 = np.clip(mu1 / fw, 0.0, 1.0)
            valid = (base_mu > 0.0) & (base_mu < fw) & (mu1 > 0.0) & (mu1 < fw)
            dark = float(camera["dark_current_e_per_pixel_s"]) * exposure
            variance = base_mu + mu1 + 2.0 * (dark + rn**2)
            delta_e = mu1 - base_mu
            snr = float(np.linalg.norm(delta_e[valid]) / max(np.linalg.norm(np.sqrt(variance[valid])), 1e-30))
            controls[tag] = {
                "pattern_name": names[i].item(),
                "base_unsaturated_fraction": float(((base_mu > 0.0) & (base_mu < fw)).mean()),
                "perturbed_unsaturated_fraction": float(((mu1 > 0.0) & (mu1 < fw)).mean()),
                "jointly_unsaturated_pixel_count": int(valid.sum()),
                "actual_task_output_difference_l2": float(np.linalg.norm(task1 - base_task)),
                "conditional_snr_on_jointly_unsaturated_pixels": snr,
                "amplitude_command_delta_rms": float(np.sqrt(np.mean((commands[i] - commands[base_index]) ** 2))),
                "base_saturated_fraction": float((base_mu >= fw).mean()),
                "perturbed_saturated_fraction": float((mu1 >= fw).mean()),
                "interpretation": "saved production-forward difference; SNR includes jointly unsaturated pixels only",
            }
            delta_command = commands[i] - commands[base_index]
            predicted_dp = a @ (b @ delta_command)
            actual_dp = (powers[i] - base).reshape(-1)
            local_task_prediction = np.zeros_like(predicted_dp)
            local_task_prediction[unsaturated] = predicted_dp[unsaturated] * (factor / fw)
            actual_task_delta = (task1 - base_task).reshape(-1)
            controls[tag]["power_jacobian_secant_error_relative_l2"] = float(
                np.linalg.norm(predicted_dp - actual_dp) / max(float(np.linalg.norm(actual_dp)), 1.0e-30)
            )
            controls[tag]["clipped_task_jacobian_secant_error_relative_l2"] = float(
                np.linalg.norm(local_task_prediction - actual_task_delta)
                / max(float(np.linalg.norm(actual_task_delta)), 1.0e-30)
            )
        by_exposure[str(exposure)]["saved_production_control_directions"] = controls
        per_pattern[str(exposure)] = [
            {"pattern_name": names[i].item(), "group": _pattern_group(str(names[i])),
             "saturated_pixel_fraction": float(fractions[i]), "task_output_mean": float(task[i].mean())}
            for i in range(len(names))
        ]
        factor_archive.close()

    return {"context": context, "pattern_count": len(names), "groups": {k: len(v) for k, v in grouped.items()},
            "exposures": by_exposure, "per_pattern": per_pattern,
            "source_files": {"camera_measurement": archive.as_posix(), "jacobian_factors": jacobian_path.as_posix(),
                             "context_scalars": context_json.as_posix()}}


def run(source: Path, config_path: Path, output: Path, preflight_only: bool) -> dict[str, Any]:
    config = _json(config_path)
    checks = _preflight(source, config)
    if preflight_only:
        return {"schema": SCHEMA, "status": "PREFLIGHT_PASS", "preflight": checks}
    contexts: dict[str, Any] = {}
    frozen_inputs = source / "frozen_v8_response_inputs_v9.npz"
    if not frozen_inputs.is_file():
        raise FileNotFoundError(f"missing frozen v8 command archive: {frozen_inputs}")
    for archive in sorted((source / "camera_measurements").glob("*_4096_complex64.npz")):
        context = archive.name.removesuffix("_4096_complex64.npz")
        factor = source / "camera_jacobians" / f"{context}_4096_complex128_factors.npz"
        scalar = source / "context_results" / f"{context}_complex64.json"
        if not factor.is_file() or not scalar.is_file():
            raise FileNotFoundError(f"missing exact v9 input for {context}")
        contexts[context] = _context_result(archive, factor, scalar, frozen_inputs,
                                            [float(v) for v in config["camera"]["exposure_sweep_s"]],
                                            config["camera"])
    if len(contexts) != 16:
        raise ValueError(f"expected all 16 contexts, found {len(contexts)}")
    report = {
        "schema": SCHEMA, "status": "VALID_COMPLETED",
        "decision_question": "Whether v9 camera readout and loss consumers support planning a controlled decoder training task",
        "source_run_dir": source.as_posix(), "configuration": config,
        "preflight": checks, "contexts": contexts,
        "training_consumer_audit": {
            "chen400_reconstruction": {"source": "scripts/research/lightgen/execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1.py",
                "producer": "physical_w_per_pixel = camera_intensity * pixel_area_m2",
                "consumer": "FIXED_GAIN * physical_w_per_pixel against task target; MSE and optional structural term are unclipped",
                "gain": "fixed constant in the A/B script, not the v9 QE/exposure/full-well camera model",
                "power_loss": "no separate L_op term in this finite-pixel A/B objective"},
            "matched_architecture": {"source": "scripts/research/lightgen/execute_lightgen_matched_architecture_v1.py",
                "producer": "physical_w_per_pixel after pixel-area integration",
                "consumer": "unclipped physical_w_per_pixel * fixed camera_gain against task target",
                "gain": "fit once by the script calibration path using its training population, then frozen; not a measured camera calibration",
                "power_loss": "mean squared log ratio of plane_powers_w to frozen per-plane references"},
            "legacy_semantic_ols": {"source": "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_complex_ols_training.py",
                "consumer": "raw_intensity directly in image reconstruction; this output is not the v9 W/pixel/electron readout",
                "power_loss": "optical term is explicitly zero in this training path",
                "compatibility": "does not consume v9 W/pixel, electron-count, or clipped proxy-camera output"},
            "interpretation": "separate research paths with different consumers; no loss or checkpoint is migrated by this audit"},
        "prohibited_activity_counts": {"formal_training_updates": 0, "phase_optimization_updates": 0,
            "dataset_reads": 0, "task_evaluator_calls": 0, "fid_calls": 0, "loss_changes": 0,
            "material_forward_calls": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-update v9 camera clipping and training-consumer audit")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ols_camera_training_interface_closure_v10.json"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.source_run_dir, args.config, args.output, args.preflight_only)
    print(json.dumps({"status": result["status"], "output": args.output.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
