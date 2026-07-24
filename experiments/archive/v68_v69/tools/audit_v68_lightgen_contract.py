"""Read-only audit for the user-confirmed LightGen-calibrated optical contract.

This tool does not construct a network, train, or modify a checkpoint.  It
checks the configuration-level contract before the v68 transparent diagnostic
path is allowed to run.  Paper attribution is deliberately separated from
project-scaled geometry: the exact distances are treated as a user-confirmed
calibration contract, not as numbers claimed to be printed verbatim in the
Science main text.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from ldm_project import load_training_config


REPORT_SCHEMA = "v68_lightgen_calibrated_contract_v1"
EXPECTED = {
    "slm_grid": 200,
    "slm_pixel_size_m": 2.5e-6,
    "d2nn_grid": 100,
    "d2nn_pixel_size_m": 2.0e-6,
    "layer_spacing_m": 1.5e-3,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "front_magnification": -0.4,
    "front_total_distance_m": 0.625,
    "front_min_pupil_m": 0.02375,
    "back_magnification": -2.5,
    "back_object_distance_m": 0.100,
    "back_image_distance_m": 0.250,
    "back_min_pupil_m": 0.0266,
    "dist_laser_to_m1_m": 0.350,
    "dist_m1_to_bs_m": 0.370,
    "dist_bs_to_slm_m": 0.025,
    "dist_slm_to_bs_m": 0.025,
    "dist_bs_to_m3_m": 0.125,
    "dist_m3_to_front_lens1_m": 0.07321428571428572,
    "dist_front_lens1_to_lens2_m": 0.3125,
    "dist_front_lens2_to_encoder_m": 0.0892857142857143,
    "dist_m3_to_lg_m": 0.475,
    "dist_lg_to_cmos_m": 0.350,
    "f_relay_m": 0.0714285714,
    "front_lens1_focal_length_m": 0.22321428571428573,
    "front_lens2_focal_length_m": 0.0892857142857143,
    "front_lens1_clear_aperture_m": 0.02375,
    "front_lens2_clear_aperture_m": 0.02375,
    "back_relay_clear_aperture_m": 0.0266,
}


def _close(actual: float, expected: float, tolerance: float = 1.0e-9) -> bool:
    return abs(float(actual) - expected) <= tolerance


def _formal_source_boundary_ok() -> dict[str, bool]:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/ldm_project/student/formal_model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []
    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
            imported_names.extend(alias.name for alias in node.names)
    # Check the import graph, not explanatory docstrings.  The formal module
    # may mention the experimental boundary in documentation while remaining
    # independent of its implementation.
    has_experimental_import = any(
        module.startswith("ldm_project.student.experimental")
        or module.startswith("student.experimental")
        for module in imported_modules
    )
    has_resize_import = "_resize_complex_field" in imported_names
    has_phase_amplitude_import = "PhaseAmplitudeDiffractiveStack" in imported_names
    return {
        "formal_source_has_no_generic_complex_resize": not has_resize_import,
        "formal_source_has_no_phase_amplitude_stack": not has_phase_amplitude_import,
        "formal_source_has_no_experimental_import": not has_experimental_import,
        # Formal output keeps fixed diagnostic fields for schema compatibility;
        # their presence is expected and does not activate an intervention.
        "formal_source_keeps_fixed_diagnostic_schema": "diagnostic_slm_override_active" in source and "diagnostic_slm_override_amplitude" in source,
        "formal_source_boundary_tokens_checked": not (
            has_resize_import or has_phase_amplitude_import or has_experimental_import
        ),
    }


def audit_config(config_path: str | Path) -> dict[str, Any]:
    config = load_training_config(config_path)
    physics = config.physics
    one_shot = config.one_shot
    d2nn = one_shot.d2nn
    if d2nn is None:
        raise ValueError("v68 LightGen contract requires one_shot.d2nn")
    macro = physics.macro_optics
    checks = {
        "amplitude_slm": one_shot.slm_residual_phase_scale_rad == 0.0,
        "front_path_model": physics.front_path_model == "coherent_imaging_4f",
        "back_path_model": physics.back_path_model == "free_space",
        "identity_ols": one_shot.ols_kind == "identity",
        "phase_only_d2nn": d2nn.modulation == "phase_only",
        "d2nn_geometry": (
            d2nn.grid_size == EXPECTED["d2nn_grid"]
            and _close(d2nn.pixel_size, EXPECTED["d2nn_pixel_size_m"], 1.0e-12)
            and _close(d2nn.layer_spacing, EXPECTED["layer_spacing_m"], 1.0e-12)
        ),
        "d2nn_layer_count": (
            d2nn.encoder_layers == EXPECTED["encoder_layers"]
            and d2nn.decoder_layers == EXPECTED["decoder_layers"]
        ),
        "slm_geometry": physics.geometry.grid_size == EXPECTED["slm_grid"] and _close(
            physics.optics.pixel_size, EXPECTED["slm_pixel_size_m"], 1.0e-12
        ),
        "macro_propagation_grid_default": macro.propagation_grid_size == 2048,
        "front_distance_contract": (
            _close(
                macro.dist_slm_to_bs + macro.dist_bs_to_m3 + macro.dist_m3_to_front_lens1,
                macro.front_lens1_focal_length,
            )
            and _close(
                macro.dist_front_lens1_to_lens2,
                macro.front_lens1_focal_length + macro.front_lens2_focal_length,
            )
            and _close(macro.dist_front_lens2_to_encoder, macro.front_lens2_focal_length)
            and _close(
                macro.dist_slm_to_bs
                + macro.dist_bs_to_m3
                + macro.dist_m3_to_front_lens1
                + macro.dist_front_lens1_to_lens2
                + macro.dist_front_lens2_to_encoder,
                EXPECTED["front_total_distance_m"],
            )
            and _close(
                macro.front_lens2_focal_length / macro.front_lens1_focal_length,
                abs(EXPECTED["front_magnification"]),
            )
        ),
        "front_pupil": macro.front_lens1_clear_aperture >= EXPECTED["front_min_pupil_m"] and macro.front_lens2_clear_aperture >= EXPECTED["front_min_pupil_m"],
        "back_distance_contract": _close(macro.dist_lg_to_relay_lens, EXPECTED["back_object_distance_m"], 1.0e-12) and _close(macro.dist_relay_lens_to_cmos, EXPECTED["back_image_distance_m"], 1.0e-12),
        "back_pupil": macro.back_relay_clear_aperture >= EXPECTED["back_min_pupil_m"],
        "macro_distance_and_pupil_values": all(
            _close(getattr(macro, field), EXPECTED[key], 1.0e-12)
            for field, key in (
                ("dist_laser_to_m1", "dist_laser_to_m1_m"),
                ("dist_m1_to_bs", "dist_m1_to_bs_m"),
                ("dist_bs_to_slm", "dist_bs_to_slm_m"),
                ("dist_slm_to_bs", "dist_slm_to_bs_m"),
                ("dist_bs_to_m3", "dist_bs_to_m3_m"),
                ("dist_m3_to_front_lens1", "dist_m3_to_front_lens1_m"),
                ("dist_front_lens1_to_lens2", "dist_front_lens1_to_lens2_m"),
                ("dist_front_lens2_to_encoder", "dist_front_lens2_to_encoder_m"),
                ("dist_m3_to_lg", "dist_m3_to_lg_m"),
                ("dist_lg_to_relay_lens", "back_object_distance_m"),
                ("dist_relay_lens_to_cmos", "back_image_distance_m"),
                ("dist_lg_to_cmos", "dist_lg_to_cmos_m"),
                ("f_relay", "f_relay_m"),
                ("front_lens1_focal_length", "front_lens1_focal_length_m"),
                ("front_lens2_focal_length", "front_lens2_focal_length_m"),
                ("front_lens1_clear_aperture", "front_lens1_clear_aperture_m"),
                ("front_lens2_clear_aperture", "front_lens2_clear_aperture_m"),
                ("back_relay_clear_aperture", "back_relay_clear_aperture_m"),
            )
        ),
        "physical_forward_enabled": physics.physical_forward_contract,
        "no_formal_diagnostic_switches": (
            one_shot.decoder_diagnostic_mode == "normal"
            and one_shot.decoder_diagnostic_modulation_override == "none"
            and one_shot.target_code_diagnostic_mode == "none"
            and one_shot.target_code_bridge_source == "none"
            and one_shot.ols_transfer_calibration == "none"
        ),
        "transparent_boundary": d2nn.exterior_boundary == "transparent_identity",
        "checkpointed_active_capture": (
            d2nn.transparent_memory_mode == "checkpointed_active_capture"
        ),
    }
    config_name = Path(config_path).name
    if config_name.endswith("_a.yaml"):
        checks["branch_a_camera_only_objective"] = (
            one_shot.camera_recon_weight == 1.0
            and one_shot.camera_readability_weight == 0.0
            and not one_shot.clean_target_enabled
            and one_shot.clean_foreground_dice_weight == 0.0
        )
    elif config_name.endswith("_b.yaml"):
        checks["branch_b_minimal_readability_objective"] = (
            one_shot.camera_recon_weight == 1.0
            and one_shot.camera_readability_weight == 0.25
            and one_shot.clean_target_enabled
            and one_shot.clean_foreground_dice_weight == 0.30
        )
    checks.update(_formal_source_boundary_ok())
    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "lightgen-calibrated-project-contract-valid" if passed else "contract-invalid",
        "config": str(config_path),
        "checks": checks,
        "source_attribution": {
            "science_lightgen": "encoder–OLS–photonic-generator organization and amplitude/phase optical latent representation",
            "user_confirmed_calibration": "current macro distances and relay placement are retained as the user's LightGen-referenced calibration contract",
            "project_scaled_geometry": "100x100 @ 2 um, 2+2 two-photon-printed phase layers, identity OLS",
            "nature_distillation": "DDPM teacher distillation and single-exposure generative training idea; not the SLM/relay replacement",
        },
        "contract": {
            "slm": "reflective_amplitude",
            "ols": "complex_identity_project_approximation_no_fiber",
            "boundary": "formal_finite_crop_or_experimental_transparent_identity",
            "fabrication": "ideal_diagnostic_only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_config(args.config)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if report["status"] != "lightgen-calibrated-project-contract-valid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
