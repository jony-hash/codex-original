"""Freeze the N5 true propagation-separated optical-depth comparison contract.

The contract defines two continuation arms from the same frozen N4 C1 step-300
endpoint.  Candidate B replaces one existing generator propagation segment by
two half-distance production ASM segments with a trainable midpoint phase
plane.  The midpoint has no crop; its zero state is therefore an identity on
the full production propagation grid.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen_2d_semantic_generation_input_ols_alignment import (
    BACK_CLEAR_APERTURE_M,
    BACK_FOCAL_LENGTH_M,
    BACK_MAGNIFICATION,
    BACK_SEGMENTS_M,
    LAYER_SPACING_M,
    OLS_PIXEL_PITCH_M,
    OLS_SHAPE,
    WAVELENGTH_M,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.evidence import atomic_write_new_json, file_sha256
from ldm_project.v2.optics.propagation import (
    angular_spectrum_propagate,
    clear_production_transfer_cache,
)
from ldm_project.v2.optics.relays import CoherentImagingRelay


TASK_ID = "DEFINE_LIGHTGEN_2D_TRUE_OPTICAL_DEPTH_COMPARISON_CONTRACT_V1"
CONTRACT_SCHEMA = "lightgen_2d_true_optical_depth_comparison_contract.v1"
CLASSIFICATION = "N5_TRUE_OPTICAL_DEPTH_COMPARISON_READY"
OUTER_GATE = "N5_TRUE_OPTICAL_DEPTH_COMPARISON_READY"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

N4_ENDPOINT_REPORT = Path(
    "outputs/20260914_n4_endpoint_attribution/runs/"
    "lightgen_2d_n4_endpoint_attribution_evaluation_v1_attempt_01/"
    "lightgen_2d_n4_endpoint_attribution_report.v1.json"
)
N4_ENDPOINT_REPORT_SHA256 = "18cef70e3ad1a2a7f81144663f6e55373d08265c4d387d2e39df2aec0c4a3388"
C1_STEP300_CHECKPOINT = Path(
    "outputs/20260913_n4/runs/"
    "lightgen_2d_ols_interface_synthesis_model_family_c1_attempt_01/"
    "lightgen_2d_n4_c1_step_300_checkpoint.v1.pt"
)
C1_STEP300_CHECKPOINT_SHA256 = "d7a5581df3fb460337e8655865a200886739cdeab0fb6074213cc4b2cc66c6e9"
V11_REPORT = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_report.v1.json"
)
V11_REPORT_SHA256 = "bc603b7a3b7ef1590bc354cb0b62464f3ececcb2390834d864d58bfbe9158184"
N3_AGGREGATE = Path(
    "outputs/20260912_n3/runs/"
    "n3_low_rank_model_order_aggregate_v1_attempt_01/"
    "n3_low_rank_model_order_aggregate.v1.json"
)
N3_AGGREGATE_SHA256 = "1b6bd6af18462f717fa19532e1a981e872a4e48fe6498ed561890f6570ae89de"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260914_n5/runs/"
    "lightgen_2d_true_optical_depth_comparison_contract_v1_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_true_optical_depth_comparison_contract.v1.json"

ACTIVE_SHAPE = tuple(OLS_SHAPE)
PROPAGATION_SHAPE = (2048, 2048)
PROPAGATION_DISTANCE_M = LAYER_SPACING_M
HALF_PROPAGATION_DISTANCE_M = PROPAGATION_DISTANCE_M / 2.0
CAMERA_SHAPE = (200, 200)
CAMERA_PIXEL_PITCH_M = (2.5e-6, 2.5e-6)
TOLERANCE = {"rtol": 1.0e-5, "atol": 1.0e-6}
DISTINCTNESS_MIN_MAX_ABS = 1.0e-4
DISTINCTNESS_MIN_RELATIVE_L2 = 1.0e-4
ADAPTER_PARAMETER_COUNT = 4 * 100 * 9
GENERATOR_PHASE_PARAMETER_COUNT = 2 * 100 * 100
DEPTH_PHASE_PARAMETER_COUNT = 100 * 100


class ContractBlocked(RuntimeError):
    """The N5 definition cannot be made technically valid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractBlocked(message)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractBlocked(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ContractBlocked(f"Git state cannot be resolved: {args}") from error
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_line_count": len(status.splitlines()) if status else 0,
    }


def _authority_ref(root: Path, path: Path, expected_sha256: str, name: str) -> dict[str, str]:
    target = (root / path).resolve()
    _require(target.is_file(), f"{name} is unavailable: {target}")
    observed = file_sha256(target)
    _require(observed == expected_sha256, f"{name} SHA mismatch: expected {expected_sha256}, observed {observed}")
    return {"path": path.as_posix(), "sha256": observed}


def _validate_authorities(root: Path) -> dict[str, Any]:
    git = _git_state(root)
    _require(git["branch"] == EXPECTED_BRANCH, f"unexpected branch: {git['branch']}")
    _require(git["head"] == EXPECTED_HEAD, f"unexpected HEAD: {git['head']}")

    n4_ref = _authority_ref(root, N4_ENDPOINT_REPORT, N4_ENDPOINT_REPORT_SHA256, "N4 endpoint-attribution report")
    c1_ref = _authority_ref(root, C1_STEP300_CHECKPOINT, C1_STEP300_CHECKPOINT_SHA256, "N4 C1 step-300 checkpoint")
    v11_ref = _authority_ref(root, V11_REPORT, V11_REPORT_SHA256, "V11 report")
    n3_ref = _authority_ref(root, N3_AGGREGATE, N3_AGGREGATE_SHA256, "N3 aggregate")

    n4 = _read_json((root / N4_ENDPOINT_REPORT).resolve(), "N4 endpoint-attribution report")
    _require(n4.get("result") == "VALID_COMPLETED", "N4 endpoint-attribution result changed")
    _require(n4.get("classification") == "GENERATOR_COADAPTATION_DOMINANT", "N4 attribution classification changed")
    _require(n4.get("outer_gate") == "N4_ENDPOINT_ATTRIBUTION_RESOLVED", "N4 endpoint-attribution gate changed")
    _require(n4.get("original_n4_status", {}).get("aggregate") == "UNRESOLVED", "original N4 aggregate must remain unresolved")
    _require(n4.get("training_updates") == 0 and n4.get("optimizer_updates") == 0, "N4 attribution was not evaluation-only")
    _require(math.isclose(float(n4["smooth"]["C1"]), 0.4838, rel_tol=0.0, abs_tol=1.0e-12), "C1 baseline Smooth changed")
    _require(math.isclose(float(n4["smooth"]["C2"]), 0.4844, rel_tol=0.0, abs_tol=1.0e-12), "C2 attribution Smooth changed")
    _require(n4.get("claim_boundary", {}).get("protected_accessed") is False, "protected data boundary changed")
    _require(n4.get("claim_boundary", {}).get("official_test_accessed") is False, "official test boundary changed")
    _require(n4.get("claim_boundary", {}).get("sealed_accessed") is False, "sealed data boundary changed")

    v11 = _read_json((root / V11_REPORT).resolve(), "V11 report")
    _require(v11.get("result") == "VALID_COMPLETED", "V11 authority is not valid")
    _require(v11.get("classification") == "V11_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_SUPPORTED", "V11 authority changed")
    n3 = _read_json((root / N3_AGGREGATE).resolve(), "N3 aggregate")
    _require(n3.get("result") == "VALID_COMPLETED", "N3 authority is not valid")
    _require(n3.get("classification") == "N3_LOW_RANK_MODEL_ORDER_RESOLVED", "N3 authority changed")

    return {
        "git": git,
        "n4_endpoint_attribution": n4_ref,
        "c1_step300_checkpoint": c1_ref,
        "v11": v11_ref,
        "n3": n3_ref,
        "facts": {
            "C0_Smooth": float(n4["smooth"]["C0"]),
            "C1_Smooth": float(n4["smooth"]["C1"]),
            "C2_Smooth": float(n4["smooth"]["C2"]),
            "original_n4_aggregate": "UNRESOLVED",
        },
    }


def _center_pad(value: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    height, width = value.shape[-2:]
    target_height, target_width = target_shape
    _require(target_height >= height and target_width >= width, "center padding target is smaller than source")
    delta_y = target_height - height
    delta_x = target_width - width
    _require(delta_y % 2 == 0 and delta_x % 2 == 0, "center padding is not symmetric")
    return F.pad(value, (delta_x // 2, delta_x // 2, delta_y // 2, delta_y // 2))


def _active_plane(plane_id: str, z_m: float) -> PlaneSpec:
    return PlaneSpec(plane_id, ACTIVE_SHAPE, (OLS_PIXEL_PITCH_M,) * 2, WAVELENGTH_M, z_m, "n5_generator_active")


def _make_field(tensor: torch.Tensor, plane: PlaneSpec, operation: str) -> ComplexField:
    return ComplexField(tensor, plane, OperationProvenance(operation, None, None))


def _phase_active(field: ComplexField, phase: torch.Tensor, plane_id: str) -> ComplexField:
    _require(tuple(phase.shape[-2:]) == ACTIVE_SHAPE, "active phase shape changed")
    transfer = torch.polar(torch.ones_like(phase), phase).to(field.tensor.dtype)
    plane = field.plane.with_plane(plane_id)
    return ComplexField(field.tensor * transfer, plane, OperationProvenance("phase_only_modulation", field.plane.plane_id, None, {"phase_support": "active_100x100"}))


def _pad_active(field: ComplexField, prefix: str) -> ComplexField:
    padded_plane = field.plane.with_plane(
        f"{prefix}_padded",
        shape=PROPAGATION_SHAPE,
        coordinate_frame="n5_production_propagation_grid",
    )
    return ComplexField(
        _center_pad(field.tensor, PROPAGATION_SHAPE),
        padded_plane,
        OperationProvenance("finite_crop_center_pad", field.plane.plane_id, None, {"active_shape": list(ACTIVE_SHAPE)}),
    )


def _crop_active(field: ComplexField, plane_id: str, z_m: float) -> ComplexField:
    crop_y = (PROPAGATION_SHAPE[0] - ACTIVE_SHAPE[0]) // 2
    crop_x = (PROPAGATION_SHAPE[1] - ACTIVE_SHAPE[1]) // 2
    cropped = field.tensor[..., crop_y : crop_y + ACTIVE_SHAPE[0], crop_x : crop_x + ACTIVE_SHAPE[1]]
    return ComplexField(
        cropped,
        _active_plane(plane_id, z_m),
        OperationProvenance("finite_crop_hard_aperture", field.plane.plane_id, PROPAGATION_DISTANCE_M, {"active_shape": list(ACTIVE_SHAPE)}),
    )


def _direct_layer(field: ComplexField, phase: torch.Tensor, prefix: str) -> ComplexField:
    modulated = _phase_active(field, phase, f"{prefix}_phase")
    padded = _pad_active(modulated, prefix)
    propagated = angular_spectrum_propagate(padded, PROPAGATION_DISTANCE_M, f"{prefix}_propagated")
    return _crop_active(propagated, f"{prefix}_active", field.plane.z_m + PROPAGATION_DISTANCE_M)


def _split_layer(field: ComplexField, phase: torch.Tensor, depth_phase: torch.Tensor, prefix: str) -> tuple[ComplexField, dict[str, ComplexField]]:
    modulated = _phase_active(field, phase, f"{prefix}_phase")
    padded = _pad_active(modulated, prefix)
    first_half = angular_spectrum_propagate(padded, HALF_PROPAGATION_DISTANCE_M, f"{prefix}_half_a")
    full_depth_phase = _center_pad(depth_phase, PROPAGATION_SHAPE)
    midpoint_plane = first_half.plane.with_plane(
        f"{prefix}_true_depth_phase_plane",
        coordinate_frame="n5_production_propagation_grid",
    )
    midpoint = ComplexField(
        first_half.tensor * torch.polar(torch.ones_like(full_depth_phase), full_depth_phase).to(first_half.tensor.dtype),
        midpoint_plane,
        OperationProvenance(
            "n5_true_depth_phase_modulation",
            first_half.plane.plane_id,
            None,
            {"phase_support": "active_100x100_embedded_in_2048_grid", "outside_support": "fixed_transparent", "crop": False},
        ),
    )
    second_half = angular_spectrum_propagate(midpoint, HALF_PROPAGATION_DISTANCE_M, f"{prefix}_half_b")
    output = _crop_active(second_half, f"{prefix}_active", field.plane.z_m + PROPAGATION_DISTANCE_M)
    return output, {"first_half": first_half, "midpoint": midpoint, "second_half": second_half}


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    difference = (left - right).abs()
    left_norm = torch.linalg.vector_norm(left.reshape(-1))
    relative_l2 = torch.linalg.vector_norm((left - right).reshape(-1)) / left_norm.clamp_min(1.0e-12)
    scale = max(float(left.abs().max().item()), float(right.abs().max().item()))
    return {
        "allclose": bool(torch.allclose(left, right, **TOLERANCE)),
        "max_abs_diff": float(difference.max().item()),
        "relative_l2": float(relative_l2.item()),
        "tolerance": dict(TOLERANCE),
        "scale": scale,
    }


def _synthetic_field(device: torch.device) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, ACTIVE_SHAPE[0], device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    amplitude = 0.7 + 0.15 * torch.cos(2.0 * torch.pi * xx) * torch.sin(3.0 * torch.pi * yy)
    phase = 0.4 * torch.sin(3.0 * torch.pi * xx + 0.2) + 0.27 * torch.cos(2.0 * torch.pi * yy)
    return torch.polar(amplitude, phase).unsqueeze(0).unsqueeze(0)


def _synthetic_phase(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, ACTIVE_SHAPE[0], device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    phase_a = (0.17 * torch.sin(3.0 * torch.pi * xx) * torch.cos(2.0 * torch.pi * yy)).unsqueeze(0).unsqueeze(0)
    phase_b = (0.23 * torch.cos(2.0 * torch.pi * xx + 0.3) * torch.sin(3.0 * torch.pi * yy)).unsqueeze(0).unsqueeze(0)
    depth = (0.19 * torch.sin(5.0 * torch.pi * xx + 0.4) * torch.cos(4.0 * torch.pi * yy)).unsqueeze(0).unsqueeze(0)
    return phase_a, phase_b, depth


def _synthetic_checks(device: torch.device) -> dict[str, Any]:
    clear_production_transfer_cache()
    tensor = _synthetic_field(device)
    phase_a, phase_b, nonzero_depth = _synthetic_phase(device)
    zero_depth = torch.zeros_like(nonzero_depth)
    active_input = _make_field(tensor, _active_plane("n5_synthetic_generator_entrance", 0.0), "synthetic_input")

    full_input = _pad_active(active_input, "n5_free_space")
    direct_free = angular_spectrum_propagate(full_input, PROPAGATION_DISTANCE_M, "n5_free_direct")
    first_half = angular_spectrum_propagate(full_input, HALF_PROPAGATION_DISTANCE_M, "n5_free_half_a")
    split_free = angular_spectrum_propagate(first_half, HALF_PROPAGATION_DISTANCE_M, "n5_free_half_b")
    free_comparison = _comparison(direct_free.tensor, split_free.tensor)

    a_layer1 = _direct_layer(active_input, phase_a, "n5_candidate_a_layer_1")
    a_layer2 = _direct_layer(a_layer1, phase_b, "n5_candidate_a_layer_2")
    b_layer1 = _direct_layer(active_input, phase_a, "n5_candidate_b_layer_1")
    b_layer2, b_records = _split_layer(b_layer1, phase_b, zero_depth, "n5_candidate_b_layer_2")
    layer1_comparison = _comparison(a_layer1.tensor, b_layer1.tensor)
    midpoint_identity = _comparison(b_records["first_half"].tensor, b_records["midpoint"].tensor)
    layer2_comparison = _comparison(a_layer2.tensor, b_layer2.tensor)

    exit_plane = _active_plane("n5_synthetic_generator_exit", 2.0 * PROPAGATION_DISTANCE_M)
    camera_plane = PlaneSpec(
        "n5_synthetic_raw_cmos",
        CAMERA_SHAPE,
        CAMERA_PIXEL_PITCH_M,
        WAVELENGTH_M,
        2.0 * PROPAGATION_DISTANCE_M + sum(BACK_SEGMENTS_M),
        "n5_camera",
    )
    relay = CoherentImagingRelay(
        exit_plane,
        camera_plane,
        BACK_MAGNIFICATION,
        name="n5_synthetic_output_b0",
        segments_m=BACK_SEGMENTS_M,
        focal_lengths_m=(BACK_FOCAL_LENGTH_M,),
        clear_aperture_m=BACK_CLEAR_APERTURE_M,
    )
    a_exit = ComplexField(a_layer2.tensor, exit_plane, a_layer2.provenance)
    b_exit = ComplexField(b_layer2.tensor, exit_plane, b_layer2.provenance)
    a_camera = relay(a_exit)
    b_camera = relay(b_exit)
    raw_comparison = _comparison(a_camera.tensor, b_camera.tensor)

    nonzero_layer2, nonzero_records = _split_layer(b_layer1, phase_b, nonzero_depth, "n5_candidate_b_nonzero_layer_2")
    nonzero_field_comparison = _comparison(a_layer2.tensor, nonzero_layer2.tensor)
    nonzero_exit = ComplexField(nonzero_layer2.tensor, exit_plane, nonzero_layer2.provenance)
    nonzero_camera = relay(nonzero_exit)
    nonzero_raw_comparison = _comparison(a_camera.tensor, nonzero_camera.tensor)
    nonzero_max_abs = nonzero_field_comparison["max_abs_diff"]
    nonzero_relative_l2 = nonzero_field_comparison["relative_l2"]

    metadata = {
        "active_shape": list(ACTIVE_SHAPE),
        "propagation_shape": list(PROPAGATION_SHAPE),
        "dtype": str(tensor.dtype),
        "wavelength_m": WAVELENGTH_M,
        "pixel_pitch_m": list((OLS_PIXEL_PITCH_M,) * 2),
        "distance_m": PROPAGATION_DISTANCE_M,
        "half_distance_m": HALF_PROPAGATION_DISTANCE_M,
        "input_z_m": active_input.plane.z_m,
        "midpoint_z_m": b_records["midpoint"].plane.z_m,
        "expected_midpoint_z_m": PROPAGATION_DISTANCE_M + HALF_PROPAGATION_DISTANCE_M,
        "exit_z_m": b_records["second_half"].plane.z_m,
        "expected_exit_z_m": 2.0 * PROPAGATION_DISTANCE_M,
        "midpoint_shape": list(b_records["midpoint"].plane.shape),
        "midpoint_crop": False,
        "midpoint_outside_support": "fixed_transparent",
    }
    metadata["coordinates_valid"] = bool(
        math.isclose(metadata["midpoint_z_m"], metadata["expected_midpoint_z_m"], abs_tol=1.0e-12)
        and math.isclose(metadata["exit_z_m"], metadata["expected_exit_z_m"], abs_tol=1.0e-12)
        and tuple(metadata["midpoint_shape"]) == PROPAGATION_SHAPE
    )
    equivalence = {
        "free_space_factorization": free_comparison,
        "full_path_step0": {
            "layer1_active_complex_field": layer1_comparison,
            "midpoint_zero_phase_identity": midpoint_identity,
            "layer2_active_complex_field": layer2_comparison,
            "raw_cmos": raw_comparison,
            "metadata": metadata,
        },
    }
    equivalence["all_step0_equivalence_checks_pass"] = bool(
        free_comparison["allclose"]
        and layer1_comparison["allclose"]
        and midpoint_identity["allclose"]
        and layer2_comparison["allclose"]
        and raw_comparison["allclose"]
        and metadata["coordinates_valid"]
    )
    distinctness = {
        "nonzero_depth_phase_max_abs": float(nonzero_depth.abs().max().item()),
        "nonzero_depth_vs_candidate_a_complex_field": nonzero_field_comparison,
        "nonzero_depth_vs_candidate_a_raw_cmos": nonzero_raw_comparison,
        "minimum_max_abs_difference": DISTINCTNESS_MIN_MAX_ABS,
        "minimum_relative_l2": DISTINCTNESS_MIN_RELATIVE_L2,
        "distinct_function_class_synthetic_check": bool(
            nonzero_max_abs >= DISTINCTNESS_MIN_MAX_ABS and nonzero_relative_l2 >= DISTINCTNESS_MIN_RELATIVE_L2
        ),
        "scientific_meaning": "none; deterministic synthetic expressivity check only",
    }
    return {
        "status": "PASS" if equivalence["all_step0_equivalence_checks_pass"] and distinctness["distinct_function_class_synthetic_check"] else "FAIL",
        "device": str(device),
        "production_operator": "ldm_project.v2.optics.propagation.angular_spectrum_propagate",
        "scientific_data_consumed": False,
        "scientific_forward": False,
        "equivalence": equivalence,
        "distinctness": distinctness,
    }


def _candidate_parameter_counts() -> dict[str, Any]:
    return {
        "CANDIDATE_A_CONTINUED_C1": {
            "trainable": ["adapter.u_real", "adapter.u_imag", "adapter.v_real", "adapter.v_imag", "generator.phase"],
            "adapter": ADAPTER_PARAMETER_COUNT,
            "existing_generator_phase": GENERATOR_PHASE_PARAMETER_COUNT,
            "total_trainable": ADAPTER_PARAMETER_COUNT + GENERATOR_PHASE_PARAMETER_COUNT,
        },
        "CANDIDATE_B_TRUE_DEPTH_C1": {
            "trainable": ["adapter.u_real", "adapter.u_imag", "adapter.v_real", "adapter.v_imag", "generator.phase", "generator.depth_phase"],
            "adapter": ADAPTER_PARAMETER_COUNT,
            "existing_generator_phase": GENERATOR_PHASE_PARAMETER_COUNT,
            "propagation_separated_depth_phase": DEPTH_PHASE_PARAMETER_COUNT,
            "total_trainable": ADAPTER_PARAMETER_COUNT + GENERATOR_PHASE_PARAMETER_COUNT + DEPTH_PHASE_PARAMETER_COUNT,
        },
    }


def build_contract(root: Path, authorities: dict[str, Any], synthetic: dict[str, Any]) -> dict[str, Any]:
    _require(synthetic["status"] == "PASS", "N5 synthetic propagation/step0 checks did not pass")
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "result": "READY",
        "status": "definition_ready",
        "classification": CLASSIFICATION,
        "outer_gate": OUTER_GATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": authorities["git"],
        "definition_entrypoint": {"path": Path(__file__).resolve().relative_to(root.resolve()).as_posix(), "sha256": file_sha256(Path(__file__).resolve())},
        "input_authorities": {
            "n4_endpoint_attribution": authorities["n4_endpoint_attribution"],
            "c1_step300_frozen_endpoint": authorities["c1_step300_checkpoint"],
            "v11_baseline": authorities["v11"],
            "n3_model_order": authorities["n3"],
        },
        "facts": {
            "N4_ENDPOINT_ATTRIBUTION_RESOLVED": True,
            "N4_attribution": "GENERATOR_COADAPTATION_DOMINANT",
            "C0_Smooth": authorities["facts"]["C0_Smooth"],
            "C1_Smooth": authorities["facts"]["C1_Smooth"],
            "C2_Smooth": authorities["facts"]["C2_Smooth"],
            "C1_minus_C0": "+0.1782; paired CI fully positive",
            "C2_minus_C1": "+0.0006; paired CI includes zero",
            "original_n4_aggregate": "UNRESOLVED",
        },
        "model_assumptions": [
            "The remaining gap above Smooth=0.4838 may arise from optimization horizon and/or insufficient propagation-separated synthesis expressivity.",
            "A nonzero phase plane inserted between two propagation halves is a relevant capacity test for true optical depth.",
        ],
        "design_decisions": [
            "Candidate A is a continuation, not a retry: exact C1 step300 state, rank9 adapter plus existing two-layer generator, encoder frozen.",
            "Candidate B starts from the same exact C1 step300 state and replaces the second generator segment P_d by P_(d/2) M_(E psi) P_(d/2).",
            "E embeds the trainable 100x100 depth phase into the 2048x2048 propagation grid; outside the active support is fixed transparent.",
            "There is no crop, pad, aperture, or resampling at the midpoint. The only crop is the existing final active-plane crop.",
            "psi0=0 and P_(d/2)P_(d/2) is verified using the production ASM, so B step0 is neutral.",
            "The split is applied to the second existing generator propagation interval; the first generator layer is unchanged.",
            "Candidate C2 is not used as a true-depth candidate and is not reinterpreted as a genuine separated layer.",
        ],
        "propagation_factorization": {
            "current_segment": "S_d(phi)=C_active P_d E_active M_phi",
            "candidate_b_segment": "S_depth(phi,psi)=C_active P_(d/2) M_(E_active psi) P_(d/2) E_active M_phi",
            "d_m": PROPAGATION_DISTANCE_M,
            "half_d_m": HALF_PROPAGATION_DISTANCE_M,
            "phase_operator": "M_(E_active psi) is unit-modulus phase on the full 2048 grid; E_active psi is zero outside the central 100x100 support",
            "neutral_identity": "psi0=0 => M_(E_active psi0)=I_full and P_(d/2)M_0P_(d/2)=P_d within frozen tolerance",
            "separation": "midpoint z is half a propagation interval after the second generator phase plane; midpoint crop=False",
        },
        "synthetic_preflight": synthetic,
        "candidates": {
            "A_CONTINUED_C1": {
                "label": "CONTINUED_C1",
                "initialization": "exact frozen C1 step300 checkpoint; restore shared adapter/generator state and optimizer/RNG state",
                "architecture": "rank9 OLS interface + existing 2-layer generator",
                "trainable": _candidate_parameter_counts()["CANDIDATE_A_CONTINUED_C1"]["trainable"],
                "frozen": ["encoder", "OLS", "W_OLS", "Smooth v2.1", "prototypes", "templates"],
                "parameter_counts": _candidate_parameter_counts()["CANDIDATE_A_CONTINUED_C1"],
            },
            "B_TRUE_DEPTH_C1": {
                "label": "TRUE_DEPTH_C1",
                "initialization": "same exact C1 step300 checkpoint; shared parameters identical to A and depth_phase=0",
                "architecture": "rank9 OLS interface + existing 2-layer generator with second propagation interval factorized by a separated midpoint phase plane",
                "trainable": _candidate_parameter_counts()["CANDIDATE_B_TRUE_DEPTH_C1"]["trainable"],
                "frozen": ["encoder", "OLS", "W_OLS", "Smooth v2.1", "prototypes", "templates"],
                "parameter_counts": _candidate_parameter_counts()["CANDIDATE_B_TRUE_DEPTH_C1"],
                "midpoint_phase_shape": [1, 1, 100, 100],
                "midpoint_phase_support": "central active support embedded in full propagation grid",
            },
        },
        "common_objective": {
            "formula": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image",
            "lambda_transfer": 28.48464149920509,
            "lambda_image": 13.353282014834436,
            "recalibration": False,
            "tuning": False,
        },
        "common_execution": {
            "optimizer": "Adam",
            "learning_rate": 1.0e-3,
            "batch": 10,
            "additional_updates": 300,
            "evaluation_steps_relative_to_continuation": [0, 100, 200, 300],
            "same_train_order": True,
            "same_rng_convention": True,
            "shared_initial_checkpoint": True,
            "shared_parameter_optimizer_state_restored": True,
            "new_depth_phase_optimizer_state": "zero-initialized Adam moments",
            "sampler": "fresh deterministic continuation view at global offset 300; no exhausted 300-step sampler reuse",
        },
        "metrics": {
            "primary": {
                "name": "Delta_depth",
                "formula": "R_B300-R_A300",
                "supported": "Delta_depth>=+0.05 AND paired CI lower>0 AND B L_image300-L_image0<=+0.005 AND cyclic template/Smooth transfer retained AND raw validity passes",
                "failure": "N5_TRUE_OPTICAL_DEPTH_NOT_SUPPORTED",
            },
            "optimization_horizon_diagnostic": {
                "formula": "R_A300-0.4838",
                "paired_ci": True,
                "is_outer_gate": False,
            },
            "reported": ["DEV image loss", "cyclic template transfer", "cyclic Smooth transfer", "raw/diversity", "adapter residual", "generator phase changes", "depth phase diagnostics", "per-class Smooth recall"],
            "bootstrap": {"replicates": 999, "seed": 271828, "unit": "paired identical DEV rows"},
        },
        "interpretation": {
            "A_material_and_B_approx_A": "OPTIMIZATION_HORIZON_DOMINANT_TRUE_DEPTH_NOT_SUPPORTED",
            "B_materially_exceeds_A": "TRUE_DEPTH_SUPPORTED",
            "neither_material": "LINEAR_INTERFACE_OPTICAL_SYNTHESIS_CEILING_OR_UNRESOLVED",
            "A_material_and_B_materially_exceeds_A": "TRUE_DEPTH_SUPPORTED",
        },
        "stage_a": {"firewall": "R300>=0.937", "independent": True, "stage_b_authorized_by_n5": False},
        "one_shot": {
            "preflight": 1,
            "candidate_a_scientific_attempts": 1,
            "candidate_b_scientific_attempts": 1,
            "retry": False,
            "second_arm": False,
            "poor_performance_is": "valid_scientific_outcome; not a harness blocker",
        },
        "hard_stop_conditions": [
            "required authority cannot be resolved",
            "production P_(d/2)P_(d/2) equivalence or full-path neutral equivalence fails",
            "true-depth midpoint cannot be represented without undefined crop/coordinate semantics",
            "data isolation or evaluator symmetry fails",
            "required statistic is undefined",
        ],
        "claim_boundary": {
            "scientific_training_executed": False,
            "scientific_forward_executed": False,
            "synthetic_forward_only": True,
            "stage_b": False,
            "generation": False,
            "protected": False,
            "official_test": False,
            "sealed": False,
            "formal_promotion": False,
            "physical_realizability_claim": False,
        },
        "future_authorization_scope": "EXECUTE_N5_TRUE_OPTICAL_DEPTH_COMPARISON_V1: one A/B 300-additional-update scientific comparison from the exact C1 step300 endpoint; no other arm, retry, rank/loss/lambda change, Stage B, generation or promotion",
        "next": "requires a separate explicit scientific execution authorization; this contract does not authorize training",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    target = (args.output_dir or root / DEFAULT_OUTPUT_DIR).resolve()
    try:
        _require(target.is_relative_to((root / "outputs").resolve()), "N5 contract output must remain under outputs/")
        _require(not target.exists(), f"refusing to overwrite N5 contract output: {target}")
        if args.device == "cuda":
            _require(torch.cuda.is_available(), "CUDA requested for N5 synthetic preflight but is unavailable")
        authorities = _validate_authorities(root)
        synthetic = _synthetic_checks(torch.device(args.device))
        payload = build_contract(root, authorities, synthetic)
        target.mkdir(parents=True, exist_ok=False)
        path = target / CONTRACT_NAME
        atomic_write_new_json(path, payload, artifact_kind="N5 true optical depth contract")
        result = {
            "result": payload["result"],
            "status": payload["status"],
            "classification": payload["classification"],
            "outer_gate": payload["outer_gate"],
            "contract": {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)},
            "synthetic_preflight": {"status": synthetic["status"], "device": synthetic["device"]},
            "scientific_training_executed": False,
            "training_authorized": False,
            "stage_b_authorized": False,
        }
    except ContractBlocked as error:
        result = {"result": "BLOCKED", "status": "blocked", "classification": "N5_TRUE_OPTICAL_DEPTH_COMPARISON_BLOCKED", "outer_gate": OUTER_GATE, "reason": str(error), "scientific_training_executed": False}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["result"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
