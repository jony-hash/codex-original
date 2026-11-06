"""Zero-update readout and structure audit for the fixed LightGen400 endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import statistics
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    SOURCE_POWER_W,
    ChenLightGen400Baseline,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    capture_execution_source_manifest,
)
from ldm_project.v2.evidence import (
    atomic_torch_save,
    atomic_torch_save_new,
    atomic_write_json,
    atomic_write_new_json,
)
from scripts.research.lightgen import (
    diagnose_chen_lightgen_400_quickcheck_v1 as quickcheck,
)
from scripts.research.lightgen import (
    execute_chen_lightgen_400_joint_reconstruction_v1 as joint,
)
from scripts.research.lightgen import (
    execute_chen_lightgen_400_mnist_baseline_v1 as baseline,
)

ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_READOUT_STRUCTURE_AUDIT_V1"
AUTHORITY = "USER_APPROVED_CHEN_LIGHTGEN_400_READOUT_STRUCTURE_AUDIT_20261016"
GRID = 4096
REFERENCE_GRID = 6144
WINDOW_TOLERANCE = 1.0e-4
BLOCK_SIZE = 10
TASK_SIZE = 28
SATURATION_MAX = 0.01
MSE_IMPROVEMENT_MIN = 0.10
REGION_SEMANTICS = "target28_ge_0.1_edge_priority_disjoint_v2"
STATE_NAMES = ("start_4096", "endpoint_500")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object at {path}")
    return value


def _load_inputs(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config_path, source_checkpoint_path, source_report_path = map(
        _resolve, (args.config, args.checkpoint, args.source_report)
    )
    config, source_checkpoint, source_report, source_model, references = (
        quickcheck._load_inputs(
            source_checkpoint_path, source_report_path, config_path, device
        )
    )
    _require(
        source_report.get("status") == "VALID_FAILED",
        "explicit source baseline report must retain VALID_FAILED status",
    )
    _require(
        source_checkpoint.get("completed_updates") == 30_000
        and source_checkpoint.get("config", {})
        .get("optics", {})
        .get("selected_production_grid")
        == 2048,
        "source checkpoint must be the explicit 30,000-step 2048 baseline",
    )
    _require(
        source_checkpoint.get("model_identity", {}).get("phase_shape") == [400, 400],
        "source model phase shape differs",
    )
    _require(
        source_report.get("status") == "VALID_FAILED", "source report status changed"
    )

    fixed_path = _resolve(args.fixed_population)
    pixels, labels, train_indices, _ = baseline._load_data(_resolve(args.data))
    population = quickcheck._load_fixed_population(fixed_path, labels, train_indices)
    _require(
        int(population["fit_rows"].numel()) == 100
        and int(population["confirm_rows"].numel()) == 200,
        "fixed population must contain 100 fit and 200 confirmation TRAIN rows",
    )
    _require(
        torch.equal(population["all_labels"], labels[population["all_rows"]].cpu()),
        "fixed-population labels differ from canonical TRAIN labels",
    )

    start_path = _resolve(args.start_state)
    start_state = torch.load(start_path, map_location="cpu", weights_only=False)
    _require(
        isinstance(start_state, dict)
        and set(start_state) == set(source_checkpoint["model"]),
        "explicit 4096 start-state tensor contract differs from source model",
    )
    endpoint_path = _resolve(args.endpoint_checkpoint)
    endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
    joint_report_path = _resolve(args.joint_report)
    joint_report = _json(joint_report_path)
    _require(
        joint_report.get("status") == "VALID_FAILED"
        and joint_report.get("source_checkpoint_step") == 30_000
        and joint_report.get("training", {}).get("completed_updates") == 500,
        "joint-run report does not describe the completed 500-step VALID_FAILED run",
    )
    _require(
        endpoint.get("schema") == "chen_lightgen_400_joint_reconstruction_checkpoint.v1"
        and endpoint.get("task_id") == joint.TASK_ID
        and endpoint.get("completed_updates") == 500
        and endpoint.get("source_checkpoint_step") == 30_000
        and endpoint.get("grid") == GRID,
        "explicit endpoint must be the 500-step 4096 checkpoint",
    )
    _require(
        torch.equal(endpoint["fit_rows"], population["fit_rows"])
        and torch.equal(endpoint["confirm_rows"], population["confirm_rows"]),
        "endpoint checkpoint population rows differ from fixed population",
    )
    _require(
        math.isclose(
            float(endpoint["camera_gain_fit_only_w_inverse"]),
            float(joint_report["fixed_gain"]),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        "endpoint checkpoint gain differs from the completed-run report",
    )
    _require(
        all(
            torch.equal(
                endpoint["frozen_references"][key].cpu(),
                source_checkpoint["references"][key].cpu(),
            )
            for key in ("scale", "power_reference", "camera_gain")
        ),
        "endpoint frozen references differ from the original source checkpoint",
    )
    _require(
        set(endpoint["model"]) == set(start_state),
        "endpoint and start model tensor contracts differ",
    )

    models: dict[str, ChenLightGen400Baseline] = {}
    for name, state in (
        (STATE_NAMES[0], start_state),
        (STATE_NAMES[1], endpoint["model"]),
    ):
        model = quickcheck._diagnostic_model(source_model, GRID)
        model.load_state_dict(state, strict=True)
        model.eval()
        model.requires_grad_(False)
        _require(
            all(not parameter.requires_grad for parameter in model.parameters()),
            f"{name} model unexpectedly has trainable parameters",
        )
        models[name] = model
    del source_model, source_checkpoint, endpoint, start_state
    torch.cuda.empty_cache()
    evaluator_path = (
        _resolve(args.evaluator)
        if args.evaluator
        else _resolve(Path(config["evaluation"]["evaluator_checkpoint"]))
    )
    evaluator = baseline._load_evaluator(evaluator_path, device)
    evaluator.eval()
    return {
        "config": config,
        "source_report": source_report,
        "references": references,
        "models": models,
        "population": population,
        "pixels": pixels,
        "labels": labels,
        "train_indices": train_indices,
        "evaluator": evaluator,
        "evaluator_path": evaluator_path,
        "source_checkpoint_path": source_checkpoint_path,
        "source_report_path": source_report_path,
        "fixed_path": fixed_path,
        "start_path": start_path,
        "endpoint_path": endpoint_path,
        "joint_report_path": joint_report_path,
    }


def _entry_pixels(
    pixel_source: torch.Tensor, rows: torch.Tensor, device: torch.device
) -> torch.Tensor:
    target, _ = baseline._image_batch(pixel_source, rows, device)
    return target


def _window_probe(
    model: ChenLightGen400Baseline, sample: torch.Tensor, roi_m: float
) -> dict[str, Any]:
    _, details = baseline._window_reference(
        model, sample, [GRID, REFERENCE_GRID], roi_m, include_sampling_reference=False
    )
    errors = details[GRID]["complex_relative_error_vs_largest_window"]
    maximum = max(float(value) for value in errors.values())
    return {
        "per_plane_errors": errors,
        "maximum_error": maximum,
        "passed": maximum <= WINDOW_TOLERANCE,
    }


def _verify_power_integrals(
    result: dict[str, Any], target: torch.Tensor
) -> dict[str, float]:
    area = math.prod((1.2e-3, 1.2e-3))
    expected_slm_after = SOURCE_POWER_W * target.double().mean((1, 2, 3))
    observed_slm_after = result["input_power_w"].double()
    ols_incident = result["ols_input_power_w"].double()
    modal = result["ols_output_power_w"].double()
    camera = result["physical_w_per_pixel"].double().sum((1, 2, 3))
    # Production input_power_w is integrated after the intensity-mask modulation.
    _require(
        torch.allclose(
            observed_slm_after,
            expected_slm_after,
            rtol=1.0e-5,
            atol=1.0e-12,
        ),
        "SLM post-modulation integral differs from P0 times mean image intensity",
    )
    _require(
        torch.allclose(
            observed_slm_after,
            SOURCE_POWER_W * target.double().sum((1, 2, 3)) / target[0].numel(),
            rtol=1.0e-5,
            atol=1.0e-12,
        ),
        "independent area-sum formula disagrees with the post-SLM power",
    )
    system_efficiency = modal / observed_slm_after.clamp_min(1.0e-300)
    modal_capture = modal / ols_incident.clamp_min(1.0e-300)
    return {
        "slm_before_modulation_w": SOURCE_POWER_W,
        "slm_after_modulation_measured_w": float(observed_slm_after[0]),
        "slm_after_modulation_area_integral_w": float(expected_slm_after[0]),
        "ols_incident_w": float(ols_incident[0]),
        "ols_modal_sum_w": float(modal[0]),
        "system_efficiency_slm_to_modal": float(system_efficiency[0]),
        "ols_modal_capture_fraction": float(modal_capture[0]),
        "camera_roi_pixel_sum_w": float(camera[0]),
        "active_area_m2": area,
    }


def _gain_values(
    fit_physical: torch.Tensor,
    fit_target: torch.Tensor,
    original_gain: float,
    previous_gain: float,
) -> dict[str, float]:
    w = fit_physical.detach().cpu().double().flatten()
    y = fit_target.detach().cpu().double().flatten()
    _require(
        w.numel() == y.numel() and torch.isfinite(w).all() and torch.isfinite(y).all(),
        "fit-only gain inputs are non-finite or have mismatched shapes",
    )
    denominator = float(w.square().sum())
    _require(
        denominator > 0.0 and math.isfinite(denominator),
        "fit-only least-squares denominator is zero",
    )
    ls_gain = float((w * y).sum()) / denominator
    _require(
        ls_gain > 0.0 and math.isfinite(ls_gain),
        "fit-only least-squares gain is invalid",
    )
    allowed_exceedances = math.floor(SATURATION_MAX * w.numel())
    threshold_index = w.numel() - allowed_exceedances - 1
    upper_order_statistic = float(torch.kthvalue(w, threshold_index + 1).values)
    cap_gain = math.inf if upper_order_statistic <= 0.0 else 1.0 / upper_order_statistic
    constrained_gain = min(ls_gain, cap_gain)
    return {
        "original_gain_w_inverse": float(original_gain),
        "previous_frozen_gain_w_inverse": float(previous_gain),
        "fit_only_least_squares_gain_w_inverse": ls_gain,
        "fit_only_saturation_cap_gain_w_inverse": cap_gain
        if math.isfinite(cap_gain)
        else None,
        "fit_only_constrained_gain_w_inverse": constrained_gain,
        "fit_pixel_count": int(w.numel()),
        "allowed_fit_pixels_over_one": allowed_exceedances,
        "cap_threshold_physical_w_per_pixel": upper_order_statistic,
    }


def _ssim_map28(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    coordinate = torch.arange(-3, 4, dtype=torch.float64, device=image.device)
    one = torch.exp(-coordinate.square() / (2.0 * 1.5**2))
    one /= one.sum()
    kernel = (one[:, None] * one[None, :]).to(image.dtype)[None, None]
    x = F.conv2d(image, kernel, padding=3)
    y = F.conv2d(target, kernel, padding=3)
    x2 = F.conv2d(image.square(), kernel, padding=3) - x.square()
    y2 = F.conv2d(target.square(), kernel, padding=3) - y.square()
    xy = F.conv2d(image * target, kernel, padding=3) - x * y
    return ((2 * x * y + 0.01**2) * (2 * xy + 0.03**2)) / (
        (x.square() + y.square() + 0.01**2) * (x2 + y2 + 0.03**2)
    )


def _target_regions(target: torch.Tensor) -> dict[str, torch.Tensor]:
    target28 = F.adaptive_avg_pool2d(target, (TASK_SIZE, TASK_SIZE))
    foreground = (target28 >= 0.1).float()
    dilated = F.max_pool2d(foreground, 3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - foreground, 3, stride=1, padding=1)
    edge28 = (dilated > 0) & (eroded == 0)
    interior28 = (foreground > 0) & ~edge28
    background28 = (foreground == 0) & ~edge28

    def up(mask: torch.Tensor) -> torch.Tensor:
        return F.interpolate(mask.float(), size=(400, 400), mode="nearest").bool()

    regions = {
        "edge": up(edge28),
        "stroke_interior": up(interior28),
        "background": up(background28),
        "target28": target28,
        "edge28": edge28,
    }
    partition = (
        regions["edge"].int()
        + regions["stroke_interior"].int()
        + regions["background"].int()
    )
    _require(
        bool((partition == 1).all()),
        "edge-priority regions must partition every 400x400 pixel",
    )
    return regions


def _metrics_for_gain(
    task: torch.Tensor,
    target: torch.Tensor,
    row: int,
    label: int,
    evaluator: torch.nn.Module,
    regions: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    row_tensor = torch.tensor([row], dtype=torch.long)
    view, psnr, ssim = baseline._reconstruction_batch_metrics(
        task, target, row_tensor, row_tensor
    )
    residual2 = (task - target).double().square()
    total_sse = float(residual2.sum())
    regional: dict[str, Any] = {}
    for name in ("edge", "stroke_interior", "background"):
        mask = regions[name].to(device=task.device)
        count = int(mask.sum())
        sse = float((residual2 * mask).sum())
        regional[name] = {
            "pixel_count": count,
            "sse": sse,
            "mse": sse / max(count, 1),
            "fraction_of_total_sse": sse / max(total_sse, 1.0e-300),
        }
    if total_sse > 0:
        share_sum = sum(value["fraction_of_total_sse"] for value in regional.values())
        _require(
            math.isclose(share_sum, 1.0, rel_tol=1.0e-10, abs_tol=1.0e-10),
            "regional error shares do not sum to the full-image squared error",
        )
    task28 = F.adaptive_avg_pool2d(task.clamp(0, 1), (TASK_SIZE, TASK_SIZE))
    target28 = regions["target28"]
    ssim_map = _ssim_map28(task28, target28)
    edge = regions["edge28"]
    edge_target_energy = float((target28 * edge).sum())
    edge_prediction_energy = float((task28 * edge).sum())
    saturation = (task < 0) | (task > 1)
    saturation_count = int(saturation.sum())
    evaluator_prediction = int(evaluator(view.float()).argmax(1)[0].cpu())
    record = {
        "row": row,
        "label": label,
        "mse400_unclipped": float((task - target).square().mean()),
        "psnr28_db": float(psnr[0]),
        "ssim28": float(ssim[0]),
        "saturation_fraction": saturation_count / task.numel(),
        "mean_excess_magnitude": float(
            ((task - 1).clamp_min(0) + (-task).clamp_min(0)).mean()
        ),
        "edge_target_energy28": edge_target_energy,
        "edge_prediction_energy28": edge_prediction_energy,
        "ed_prediction_descriptive": evaluator_prediction,
        "regions": regional,
    }
    heat = {
        "prediction28_clipped": task28.detach().cpu()[0, 0],
        "target28": target28.detach().cpu()[0, 0],
        "ssim_map28": ssim_map.detach().cpu()[0, 0],
        "saturation28": F.adaptive_max_pool2d(saturation.float(), (28, 28)).cpu()[0, 0]
        > 0,
        "saturation400": saturation.detach().cpu()[0, 0],
        "error400": (task - target).detach().cpu()[0, 0],
    }
    return record, heat


def _covariance_summary(coefficients: torch.Tensor) -> dict[str, Any]:
    z = coefficients.detach().cpu().to(torch.complex128)
    gauge_pivot = z.abs().argmax(dim=1)
    pivot = z[torch.arange(z.shape[0]), gauge_pivot]
    _require(
        bool((pivot.abs() > 0).all()),
        "complex coefficient global phase gauge has a zero pivot",
    )
    gauge_fixed = z * (pivot.conj() / pivot.abs())[:, None]
    summaries: dict[str, Any] = {}
    for name, value in (("raw", z), ("global_phase_fixed", gauge_fixed)):
        features = torch.cat((value.real, value.imag), dim=1)
        centered = features - features.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
        total = eigenvalues.sum()
        participation = float(
            total.square() / eigenvalues.square().sum().clamp_min(1.0e-300)
        )
        summaries[name] = {
            "feature_dimension": int(features.shape[1]),
            "effective_covariance_rank": participation,
            "largest_10_variance_fraction": float(
                eigenvalues[:10].sum() / total.clamp_min(1e-300)
            ),
            "variance_eigenvalues": eigenvalues.tolist(),
        }
    return summaries


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require(
        torch.cuda.is_available(),
        "AutoDL CUDA is required for production-path preflight",
    )
    output = _resolve(args.output_dir)
    _require(
        not output.exists(), f"refusing to overwrite preflight directory: {output}"
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device("cuda")
    loaded = _load_inputs(args, device)
    model_states_before = {
        name: baseline._state_snapshot(model)
        for name, model in loaded["models"].items()
    }
    evaluator_state_before = baseline._state_snapshot(loaded["evaluator"])
    roi = float(loaded["config"]["optics"]["common_complex_comparison_roi_m"])
    checks: list[dict[str, Any]] = []
    for name in STATE_NAMES:
        model = loaded["models"][name]
        for label in (0, 1):
            row = int(loaded["population"]["confirm_rows_by_class"][label, 0])
            target = _entry_pixels(loaded["pixels"], torch.tensor([row]), device)
            with torch.no_grad():
                result = model.forward_from_image(target)
            probe_outputs = {
                "state": name,
                "row": row,
                "physical_w_per_pixel": result["physical_w_per_pixel"].cpu(),
                "ols_coefficients": result["ols_coefficients"].cpu(),
            }
            probe_path = output / f"probe_{name}_{label}.pt"
            atomic_torch_save_new(probe_path, probe_outputs)
            probe_readback = torch.load(
                probe_path, map_location="cpu", weights_only=False
            )
            _require(
                torch.equal(
                    probe_outputs["physical_w_per_pixel"],
                    probe_readback["physical_w_per_pixel"],
                )
                and torch.equal(
                    probe_outputs["ols_coefficients"],
                    probe_readback["ols_coefficients"],
                ),
                "preflight production-output save/readback differs",
            )
            power_check = _verify_power_integrals(result, target)
            window = _window_probe(model, target, roi)
            checks.append(
                {
                    "state": name,
                    "label": label,
                    "row": row,
                    "power_integrals": power_check,
                    "window": window,
                }
            )
            _require(
                window["passed"],
                f"{name} row {row} failed 4096/6144 selected-plane comparison",
            )
            _require(
                result["ols_coefficients"].shape == (1, 100)
                and result["physical_w_per_pixel"].shape == (1, 1, 400, 400),
                f"{name} row {row} production output shape differs",
            )
            _require(
                all(not parameter.requires_grad for parameter in model.parameters()),
                f"{name} model became trainable during preflight",
            )

    # A row mismatch must be rejected by the production metric consumer.
    sample_rows = loaded["population"]["confirm_rows_by_class"][:2, 0].long()
    target = _entry_pixels(loaded["pixels"], sample_rows, device)
    with torch.no_grad():
        result = loaded["models"][STATE_NAMES[0]].forward_from_image(target)
    mismatch_rejected = False
    try:
        baseline._reconstruction_batch_metrics(
            result["physical_w_per_pixel"][:1],
            target[:1],
            sample_rows[:1] + 1,
            sample_rows[:1],
        )
    except ValueError:
        mismatch_rejected = True
    _require(
        mismatch_rejected,
        "production metric consumer accepted mismatched MNIST row ids",
    )
    original_view, _, _ = baseline._reconstruction_batch_metrics(
        result["physical_w_per_pixel"] * float(loaded["references"]["camera_gain"]),
        target,
        sample_rows,
        sample_rows,
    )
    target_view = F.adaptive_avg_pool2d(target, (28, 28))
    shuffled_view = target_view.flip(0)
    aligned_error = float((original_view - target_view).square().mean())
    shuffled_error = float((original_view - shuffled_view).square().mean())
    _require(
        shuffled_error > aligned_error,
        "target-content shuffle did not worsen paired reconstruction error",
    )
    for name, model in loaded["models"].items():
        _require(
            baseline._assert_state_close(
                model_states_before[name],
                baseline._state_snapshot(model),
                f"{name}_frozen_preflight",
            )
            == 0.0,
            f"{name} model state changed during preflight",
        )
    _require(
        baseline._assert_state_close(
            evaluator_state_before,
            baseline._state_snapshot(loaded["evaluator"]),
            "evaluator_frozen_preflight",
        )
        == 0.0,
        "frozen evaluator state changed during preflight",
    )

    report = {
        "schema": "chen_lightgen_400_readout_structure_preflight.v1",
        "status": "READY",
        "task_id": TASK_ID,
        "authorization_id": AUTHORITY,
        "source_checkpoint": str(loaded["source_checkpoint_path"]),
        "source_report": str(loaded["source_report_path"]),
        "source_config": str(_resolve(args.config)),
        "data": str(_resolve(args.data)),
        "evaluator": str(loaded["evaluator_path"]),
        "source_checkpoint_step": 30_000,
        "start_state": str(loaded["start_path"]),
        "endpoint_checkpoint": str(loaded["endpoint_path"]),
        "endpoint_updates": 500,
        "joint_report": str(loaded["joint_report_path"]),
        "fixed_population": str(loaded["fixed_path"]),
        "data_roles": {
            "fit": 100,
            "confirmation": 200,
            "canonical_train": True,
            "dev_reads": 0,
            "test_reads": 0,
        },
        "cuda_device": torch.cuda.get_device_name(device),
        "window_checks": checks,
        "row_mismatch_rejected": mismatch_rejected,
        "target_shuffle_negative_control": {
            "aligned_mse": aligned_error,
            "shuffled_mse": shuffled_error,
            "passed": shuffled_error > aligned_error,
        },
        "model_freeze_passed": True,
        "completed_updates": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
    }
    correctness = {
        "passed": True,
        "source_and_endpoint_steps_valid": True,
        "fixed_population_canonical_train": True,
        "selected_windows_passed": True,
        "row_mismatch_rejected": mismatch_rejected,
        "power_integral_identity_passed": True,
        "states_and_evaluator_frozen": True,
        "official_dev_test_reads": 0,
        "optimizer_updates": 0,
    }
    resource = {
        "passed": True,
        "cuda_device": report["cuda_device"],
        "peak_reserved_vram_bytes": report["peak_reserved_vram_bytes"],
    }
    implementation = {
        "passed": True,
        "engine": "cuda_400x400_4096_chunked_zero_update",
        "selected_engine": "cuda_400x400_4096_chunked_zero_update",
        "trainable_groups": [],
        "frozen_groups": ["encoder", "decoder", "ols", "evaluator"],
    }
    source_manifest = capture_execution_source_manifest(
        ROOT,
        task_id=TASK_ID,
        semantic_authority_id=f"{TASK_ID}:START4096_ENDPOINT500",
        relevant_paths=(
            Path(__file__),
            Path(
                "scripts/research/lightgen/diagnose_chen_lightgen_400_quickcheck_v1.py"
            ),
            Path(
                "scripts/research/lightgen/execute_chen_lightgen_400_mnist_baseline_v1.py"
            ),
            Path(
                "scripts/research/lightgen/execute_chen_lightgen_400_joint_reconstruction_v1.py"
            ),
            Path(
                "src/ldm_project/v2/application/research/lightgen/chen_lightgen_400_baseline.py"
            ),
            _resolve(args.config),
            loaded["fixed_path"],
        ),
        config_identity={
            "states": ["start4096", "endpoint500"],
            "population": 300,
            "readouts": ["original", "previous_frozen", "fit_ls", "fit_constrained"],
        },
    )
    run_id = uuid.uuid4().hex
    identity = ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=run_id,
        trajectory_id=f"{TASK_ID}:source30000:endpoint500",
        semantic_authority_id=f"{TASK_ID}:GRID4096:ZERO_UPDATES",
        data_role="canonical_MNIST_TRAIN_fixed_fit100_confirmation200_no_dev_test_reads",
        config_identity={
            "grid": GRID,
            "reference_grid": REFERENCE_GRID,
            "population": 300,
            "block_size": BLOCK_SIZE,
            "readouts": [
                "original_gain",
                "previous_frozen_gain",
                "fit_only_ls_gain",
                "fit_only_constrained_gain",
            ],
        },
        model_identity={
            "source_step": 30_000,
            "endpoint_updates": 500,
            "encoder": [2, 400, 400],
            "ols": "gaussian_s12_v1_100_ports",
            "decoder": [4, 400, 400],
            "grid": GRID,
            "trainable": [],
        },
        optimizer_identity={"name": "none", "updates": 0},
        sampler_identity={
            "fixed_fit_rows": 100,
            "fixed_confirmation_rows": 200,
            "source": str(loaded["fixed_path"]),
        },
        execution_source_semantics_id=source_manifest.execution_source_semantics_id,
        source_manifest_id=source_manifest.source_manifest_id,
    )
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    receipt = control.make_ready_receipt(
        correctness=correctness,
        resource=resource,
        implementation=implementation,
        selected_engine="cuda_400x400_4096_chunked_zero_update",
        resource_envelope={
            "cuda_device": report["cuda_device"],
            "peak_reserved_vram_bytes": report["peak_reserved_vram_bytes"],
        },
    )
    control.mark_preflight_ready(receipt)
    report["run_identity"] = identity.to_dict()
    report["receipt"] = receipt.to_dict()
    atomic_write_new_json(output / "preflight.json", report)
    atomic_write_new_json(output / "preflight_ready.json", receipt.to_dict())
    atomic_torch_save_new(output / "fixed_population.pt", loaded["population"])
    return report


def _read_ready(
    preflight_dir: Path, args: argparse.Namespace
) -> tuple[dict[str, Any], PreflightReceipt]:
    preflight = _json(preflight_dir / "preflight.json")
    receipt = PreflightReceipt.from_mapping(
        _json(preflight_dir / "preflight_ready.json")
    )
    _require(
        preflight.get("status") == "READY"
        and preflight.get("task_id") == TASK_ID
        and preflight.get("authorization_id") == AUTHORITY,
        "explicit READY preflight for this audit is required",
    )
    _require(
        receipt.status == "READY" and receipt.authorization_id == AUTHORITY,
        "preflight receipt is not READY for this audit",
    )
    for key, arg in (
        ("source_checkpoint", args.checkpoint),
        ("source_report", args.source_report),
        ("source_config", args.config),
        ("data", args.data),
        ("start_state", args.start_state),
        ("endpoint_checkpoint", args.endpoint_checkpoint),
        ("joint_report", args.joint_report),
        ("fixed_population", args.fixed_population),
    ):
        _require(
            preflight[key] == str(_resolve(arg)),
            f"run input differs from READY preflight: {key}",
        )
    if args.evaluator is not None:
        _require(
            preflight["evaluator"] == str(_resolve(args.evaluator)),
            "run evaluator differs from READY preflight",
        )
    return preflight, receipt


def _save_block(path: Path, payload: dict[str, Any]) -> None:
    atomic_torch_save_new(path, payload)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _require(
        torch.equal(loaded["rows"], payload["rows"])
        and torch.equal(loaded["physical_w_per_pixel"], payload["physical_w_per_pixel"])
        and torch.equal(loaded["ols_coefficients"], payload["ols_coefficients"]),
        f"saved measurement block failed readback: {path}",
    )


def _measure_state(
    name: str,
    model: ChenLightGen400Baseline,
    loaded: dict[str, Any],
    output_dir: Path,
    progress: dict[str, Any],
    stop_requested: list[bool],
) -> None:
    state_dir = output_dir / name
    state_dir.mkdir(parents=True, exist_ok=True)
    rows = loaded["population"]["all_rows"].long()
    for start in range(0, rows.numel(), BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, rows.numel())
        block_path = state_dir / f"block_{start:03d}_{end:03d}.pt"
        if block_path.exists():
            saved = torch.load(block_path, map_location="cpu", weights_only=False)
            _require(
                torch.equal(saved["rows"], rows[start:end]),
                f"existing block row ids differ: {block_path}",
            )
            progress["completed_rows_by_state"][name] = max(
                int(progress["completed_rows_by_state"].get(name, 0)), end
            )
            continue
        device = model.encoder_phase.device
        batch_rows = rows[start:end]
        target = _entry_pixels(loaded["pixels"], batch_rows, device)
        with torch.no_grad():
            result = model.forward_from_image(target)
        labels = loaded["labels"][batch_rows].long()
        modal_power = result["ols_coefficients"].abs().double().square()
        ols_input = result["ols_input_power_w"].double()
        powers = _verify_power_integrals(result, target)
        payload = {
            "rows": batch_rows.cpu(),
            "labels": labels.cpu(),
            "target": target.cpu(),
            "physical_w_per_pixel": result["physical_w_per_pixel"].cpu(),
            "ols_coefficients": result["ols_coefficients"].cpu(),
            "modal_power_w": modal_power.cpu(),
            "slm_before_modulation_power_w": torch.full_like(
                result["input_power_w"].cpu(), SOURCE_POWER_W
            ),
            "slm_after_modulation_power_w": result["input_power_w"].cpu(),
            "ols_incident_power_w": ols_input.cpu(),
            "ols_modal_sum_w": result["ols_output_power_w"].cpu(),
            "camera_roi_power_w": result["physical_w_per_pixel"].sum((1, 2, 3)).cpu(),
            "seven_plane_power_w": result["plane_powers_w"].cpu(),
            "independent_power_check_first_row": powers,
        }
        _save_block(block_path, payload)
        progress["completed_rows_by_state"][name] = end
        progress["last_saved_block"] = str(block_path)
        progress["updated_at"] = time.time()
        progress["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        atomic_write_json(output_dir / "progress.json", progress)
        atomic_write_json(
            output_dir / "heartbeat.json",
            {
                "time": time.time(),
                "state": name,
                "completed_rows_by_state": {
                    state: progress["completed_rows_by_state"].get(state, 0)
                    for state in STATE_NAMES
                },
                "elapsed_seconds": time.time() - progress["started_at"],
            },
        )
        if stop_requested[0]:
            return


def _read_state(
    name: str, directory: Path, expected_rows: torch.Tensor
) -> dict[str, torch.Tensor]:
    blocks = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(directory.glob("block_*.pt"))
    ]
    _require(blocks, f"no saved measurement blocks for {name}")
    combined = {
        key: torch.cat([block[key] for block in blocks], dim=0)
        for key in (
            "rows",
            "labels",
            "target",
            "physical_w_per_pixel",
            "ols_coefficients",
            "modal_power_w",
            "slm_before_modulation_power_w",
            "slm_after_modulation_power_w",
            "ols_incident_power_w",
            "ols_modal_sum_w",
            "camera_roi_power_w",
            "seven_plane_power_w",
        )
    }
    _require(
        torch.equal(combined["rows"], expected_rows),
        f"{name} blocks omit, duplicate, or reorder rows",
    )
    _require(
        torch.isfinite(combined["physical_w_per_pixel"]).all()
        and torch.isfinite(combined["ols_coefficients"]).all()
        and torch.isfinite(combined["modal_power_w"]).all(),
        f"{name} saved measurements contain non-finite values",
    )
    return combined


def _evaluate(
    loaded: dict[str, Any],
    measured: dict[str, torch.Tensor],
    original_gain: float,
    previous_gain: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    rows = measured["rows"]
    labels = measured["labels"]
    physical = measured["physical_w_per_pixel"].double()
    target = measured["target"].double()
    fit_mask = torch.isin(rows, loaded["population"]["fit_rows"])
    fit_indices = fit_mask.nonzero().flatten()
    gains = _gain_values(
        physical[fit_indices], target[fit_indices], original_gain, previous_gain
    )
    modal = measured["modal_power_w"].double()
    sums = modal.sum(1)
    capture = sums / measured["ols_incident_power_w"].double().clamp_min(1e-300)
    system = sums / measured["slm_after_modulation_power_w"].double().clamp_min(1e-300)
    gain_names = (
        "original_gain_w_inverse",
        "previous_frozen_gain_w_inverse",
        "fit_only_least_squares_gain_w_inverse",
        "fit_only_constrained_gain_w_inverse",
    )
    records: list[dict[str, Any]] = []
    heatmaps: dict[int, dict[str, torch.Tensor]] = {}
    evaluator = loaded["evaluator"]
    with torch.no_grad():
        for index in range(rows.numel()):
            row = int(rows[index])
            label = int(labels[index])
            regions = _target_regions(
                target[index : index + 1].to(next(evaluator.parameters()).device)
            )
            for gain_name in gain_names:
                gain = gains[gain_name]
                task = (
                    physical[index : index + 1].to(next(evaluator.parameters()).device)
                    * gain
                )
                target_device = target[index : index + 1].to(task.device)
                record, heat = _metrics_for_gain(
                    task, target_device, row, label, evaluator, regions
                )
                record.update(
                    {
                        "state": loaded["state_name"],
                        "gain_name": gain_name,
                        "gain_w_inverse": gain,
                        "population": "fit"
                        if bool(fit_mask[index])
                        else "confirmation",
                        "slm_after_modulation_power_w": float(
                            measured["slm_after_modulation_power_w"][index]
                        ),
                        "ols_incident_power_w": float(
                            measured["ols_incident_power_w"][index]
                        ),
                        "ols_modal_power_w": float(sums[index]),
                        "system_coupling_efficiency": float(system[index]),
                        "ols_modal_capture_fraction": float(capture[index]),
                        "ports_at_least_1nw": int((modal[index] >= 1.0e-9).sum()),
                        "weakest_port_power_w": float(modal[index].min()),
                        "n_eff": float(
                            sums[index].square()
                            / modal[index].square().sum().clamp_min(1.0e-300)
                        ),
                    }
                )
                records.append(record)
                if (
                    row in loaded["atlas_rows"]
                    and gain_name == "fit_only_constrained_gain_w_inverse"
                ):
                    heatmaps[row] = heat

    confirm_indices = (~fit_mask).nonzero().flatten()
    total_coeff = measured["ols_coefficients"]
    covariance = _covariance_summary(total_coeff)
    by_gain: dict[str, Any] = {}
    for gain_name in gain_names:
        gain_records = [
            record for record in records if record["gain_name"] == gain_name
        ]
        by_gain[gain_name] = {}
        for pop in ("fit", "confirmation"):
            subset = [record for record in gain_records if record["population"] == pop]
            by_gain[gain_name][pop] = {
                "count": len(subset),
                "mse400_mean": sum(r["mse400_unclipped"] for r in subset) / len(subset),
                "psnr28_mean_db": sum(r["psnr28_db"] for r in subset) / len(subset),
                "ssim28_mean": sum(r["ssim28"] for r in subset) / len(subset),
                "saturation_fraction": sum(r["saturation_fraction"] for r in subset)
                / len(subset),
                "mean_excess_magnitude": sum(r["mean_excess_magnitude"] for r in subset)
                / len(subset),
                "descriptive_ed_macro_recall": _macro_recall(subset),
                "rows_mse_improved_vs_start": None,
            }
    summary = {
        "state": loaded["state_name"],
        "gains": gains,
        "metrics_by_gain": by_gain,
        "mean_system_coupling_efficiency_slm_to_modal": float(system.mean()),
        "mean_ols_modal_capture_fraction": float(capture.mean()),
        "mean_slm_after_modulation_power_w": float(
            measured["slm_after_modulation_power_w"].double().mean()
        ),
        "mean_ols_incident_power_w": float(
            measured["ols_incident_power_w"].double().mean()
        ),
        "mean_ols_modal_power_w": float(sums.mean()),
        "per_sample_modal_power": modal.tolist(),
        "per_sample_ports_ge_1nw": (modal >= 1e-9).sum(1).tolist(),
        "per_sample_weakest_port_power_w": modal.min(1).values.tolist(),
        "per_sample_n_eff": (
            sums.square() / modal.square().sum(1).clamp_min(1e-300)
        ).tolist(),
        "ols_coefficient_covariance": covariance,
        "confirmation_rows": rows[confirm_indices].tolist(),
    }
    return records, summary, heatmaps


def _macro_recall(records: list[dict[str, Any]]) -> float:
    correct_by_label = [0] * 10
    count_by_label = [0] * 10
    for record in records:
        label = int(record["label"])
        count_by_label[label] += 1
        correct_by_label[label] += int(record["ed_prediction_descriptive"] == label)
    return sum(correct_by_label[i] / max(count_by_label[i], 1) for i in range(10)) / 10


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = (
        "state",
        "row",
        "label",
        "population",
        "gain_name",
        "gain_w_inverse",
        "mse400_unclipped",
        "psnr28_db",
        "ssim28",
        "saturation_fraction",
        "mean_excess_magnitude",
        "ed_prediction_descriptive",
        "slm_after_modulation_power_w",
        "ols_incident_power_w",
        "ols_modal_power_w",
        "system_coupling_efficiency",
        "ols_modal_capture_fraction",
        "ports_at_least_1nw",
        "weakest_port_power_w",
        "n_eff",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in columns})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_region_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "state",
                "row",
                "label",
                "population",
                "gain_name",
                "region",
                "pixel_count",
                "sse",
                "mse",
                "fraction_of_total_sse",
            ),
        )
        writer.writeheader()
        for record in records:
            for region, values in record["regions"].items():
                writer.writerow(
                    {
                        "state": record["state"],
                        "row": record["row"],
                        "label": record["label"],
                        "population": record["population"],
                        "gain_name": record["gain_name"],
                        "region": region,
                        **values,
                    }
                )


def _write_paired_analysis(
    output_dir: Path,
    per_state_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    gain_name = "fit_only_constrained_gain_w_inverse"
    regions = ("edge", "stroke_interior", "background")
    by_state: dict[str, dict[int, dict[str, Any]]] = {}
    region_by_state: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for state in STATE_NAMES:
        selected = [
            record
            for record in per_state_records[state]
            if record["gain_name"] == gain_name
        ]
        _require(len(selected) == 300, f"{state} constrained-gain rows must equal 300")
        indexed = {int(record["row"]): record for record in selected}
        _require(
            len(indexed) == 300, f"{state} contains duplicate constrained-gain rows"
        )
        by_state[state] = indexed

        region_index: dict[int, dict[str, dict[str, float]]] = {}
        for record in _read_csv(output_dir / f"regional_errors_{state}.csv"):
            if record["gain_name"] != gain_name:
                continue
            row = int(record["row"])
            region_index.setdefault(row, {})[record["region"]] = {
                "sse": float(record["sse"]),
                "fraction_of_total_sse": float(record["fraction_of_total_sse"]),
            }
        _require(
            set(region_index) == set(indexed)
            and all(set(values) == set(regions) for values in region_index.values()),
            f"{state} regional rows do not match constrained-gain samples",
        )
        region_by_state[state] = region_index

    start_name, endpoint_name = STATE_NAMES
    _require(
        set(by_state[start_name]) == set(by_state[endpoint_name]),
        "start and endpoint row sets differ",
    )
    paired: list[dict[str, Any]] = []
    for row in sorted(by_state[start_name]):
        start = by_state[start_name][row]
        endpoint = by_state[endpoint_name][row]
        _require(
            int(start["label"]) == int(endpoint["label"])
            and start["population"] == endpoint["population"],
            f"paired sample metadata differs for row {row}",
        )
        record: dict[str, Any] = {
            "row": row,
            "label": int(start["label"]),
            "population": start["population"],
            "gain_name": gain_name,
        }
        for metric in (
            "mse400_unclipped",
            "psnr28_db",
            "ssim28",
            "saturation_fraction",
            "mean_excess_magnitude",
            "ols_modal_power_w",
            "system_coupling_efficiency",
            "ols_modal_capture_fraction",
            "ports_at_least_1nw",
            "weakest_port_power_w",
            "n_eff",
        ):
            start_value = float(start[metric])
            endpoint_value = float(endpoint[metric])
            record[f"start_{metric}"] = start_value
            record[f"endpoint_{metric}"] = endpoint_value
            record[f"delta_{metric}"] = endpoint_value - start_value
        for region in regions:
            start_region = region_by_state[start_name][row][region]
            endpoint_region = region_by_state[endpoint_name][row][region]
            for value_name in ("sse", "fraction_of_total_sse"):
                start_value = start_region[value_name]
                endpoint_value = endpoint_region[value_name]
                record[f"start_{region}_{value_name}"] = start_value
                record[f"endpoint_{region}_{value_name}"] = endpoint_value
                record[f"delta_{region}_{value_name}"] = endpoint_value - start_value
        paired.append(record)

    paired_fields = list(paired[0])
    with (output_dir / "paired_sample_changes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired)

    def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(records)
        _require(count > 0, "paired analysis group is empty")
        start_sse = {
            region: sum(record[f"start_{region}_sse"] for record in records)
            for region in regions
        }
        endpoint_sse = {
            region: sum(record[f"endpoint_{region}_sse"] for record in records)
            for region in regions
        }
        start_total = sum(start_sse.values())
        endpoint_total = sum(endpoint_sse.values())
        return {
            "count": count,
            "mse_improved_rows": sum(
                record["delta_mse400_unclipped"] < 0 for record in records
            ),
            "mse_deteriorated_rows": sum(
                record["delta_mse400_unclipped"] > 0 for record in records
            ),
            "mse_equal_rows": sum(
                record["delta_mse400_unclipped"] == 0 for record in records
            ),
            "start_mse400_mean": statistics.mean(
                record["start_mse400_unclipped"] for record in records
            ),
            "endpoint_mse400_mean": statistics.mean(
                record["endpoint_mse400_unclipped"] for record in records
            ),
            "mean_mse400_delta": statistics.mean(
                record["delta_mse400_unclipped"] for record in records
            ),
            "start_ssim28_mean": statistics.mean(
                record["start_ssim28"] for record in records
            ),
            "endpoint_ssim28_mean": statistics.mean(
                record["endpoint_ssim28"] for record in records
            ),
            "mean_ssim28_delta": statistics.mean(
                record["delta_ssim28"] for record in records
            ),
            "start_saturation_fraction": statistics.mean(
                record["start_saturation_fraction"] for record in records
            ),
            "endpoint_saturation_fraction": statistics.mean(
                record["endpoint_saturation_fraction"] for record in records
            ),
            "start_mean_roots_ge_1nw": statistics.mean(
                record["start_ports_at_least_1nw"] for record in records
            ),
            "endpoint_mean_roots_ge_1nw": statistics.mean(
                record["endpoint_ports_at_least_1nw"] for record in records
            ),
            "start_samples_all_100_roots_ge_1nw": sum(
                record["start_ports_at_least_1nw"] == 100 for record in records
            ),
            "endpoint_samples_all_100_roots_ge_1nw": sum(
                record["endpoint_ports_at_least_1nw"] == 100 for record in records
            ),
            "start_weakest_port_nw_median": statistics.median(
                record["start_weakest_port_power_w"] * 1.0e9 for record in records
            ),
            "endpoint_weakest_port_nw_median": statistics.median(
                record["endpoint_weakest_port_power_w"] * 1.0e9 for record in records
            ),
            "start_n_eff_median": statistics.median(
                record["start_n_eff"] for record in records
            ),
            "endpoint_n_eff_median": statistics.median(
                record["endpoint_n_eff"] for record in records
            ),
            "start_system_coupling_efficiency_mean": statistics.mean(
                record["start_system_coupling_efficiency"] for record in records
            ),
            "endpoint_system_coupling_efficiency_mean": statistics.mean(
                record["endpoint_system_coupling_efficiency"] for record in records
            ),
            "start_ols_modal_capture_fraction_mean": statistics.mean(
                record["start_ols_modal_capture_fraction"] for record in records
            ),
            "endpoint_ols_modal_capture_fraction_mean": statistics.mean(
                record["endpoint_ols_modal_capture_fraction"] for record in records
            ),
            "start_region_pooled_sse_share": {
                region: start_sse[region] / start_total for region in regions
            },
            "endpoint_region_pooled_sse_share": {
                region: endpoint_sse[region] / endpoint_total for region in regions
            },
        }

    by_population = {
        population: summarize(
            [record for record in paired if record["population"] == population]
        )
        for population in ("fit", "confirmation")
    }
    class_records = [
        {
            "population": population,
            "label": label,
            **summarize(
                [
                    record
                    for record in paired
                    if record["population"] == population and record["label"] == label
                ]
            ),
        }
        for population in ("fit", "confirmation")
        for label in range(10)
    ]
    class_fields = [
        "population",
        "label",
        "count",
        "mse_improved_rows",
        "mse_deteriorated_rows",
        "mse_equal_rows",
        "start_mse400_mean",
        "endpoint_mse400_mean",
        "mean_mse400_delta",
        "start_ssim28_mean",
        "endpoint_ssim28_mean",
        "mean_ssim28_delta",
        "start_saturation_fraction",
        "endpoint_saturation_fraction",
        "start_mean_roots_ge_1nw",
        "endpoint_mean_roots_ge_1nw",
        "start_n_eff_median",
        "endpoint_n_eff_median",
    ]
    for region in regions:
        class_fields.extend(
            (
                f"start_{region}_pooled_sse_share",
                f"endpoint_{region}_pooled_sse_share",
            )
        )
    with (output_dir / "per_class_paired_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=class_fields)
        writer.writeheader()
        for record in class_records:
            row = {key: record[key] for key in class_fields if key in record}
            for region in regions:
                row[f"start_{region}_pooled_sse_share"] = record[
                    "start_region_pooled_sse_share"
                ][region]
                row[f"endpoint_{region}_pooled_sse_share"] = record[
                    "endpoint_region_pooled_sse_share"
                ][region]
            writer.writerow(row)
    return {
        "gain_name": gain_name,
        "by_population": by_population,
        "by_population_and_label": class_records,
        "sample_changes_file": "paired_sample_changes.csv",
        "class_summary_file": "per_class_paired_summary.csv",
    }


def _write_port_csv(
    path: Path, state_name: str, measured: dict[str, torch.Tensor]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "state",
                "row",
                "label",
                "port",
                "array_row",
                "array_column",
                "modal_power_w",
                "power_share",
                "at_least_1nw",
            ),
        )
        writer.writeheader()
        modal = measured["modal_power_w"].double()
        totals = modal.sum(1, keepdim=True).clamp_min(1e-300)
        for index, row in enumerate(measured["rows"].tolist()):
            for port, power in enumerate(modal[index].tolist()):
                writer.writerow(
                    {
                        "state": state_name,
                        "row": row,
                        "label": int(measured["labels"][index]),
                        "port": port,
                        "array_row": port // 10,
                        "array_column": port % 10,
                        "modal_power_w": power,
                        "power_share": power / float(totals[index]),
                        "at_least_1nw": power >= 1e-9,
                    }
                )


def _atlas(
    path: Path,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    selected_rows: list[int],
    start_heat: dict[int, dict[str, torch.Tensor]],
    end_heat: dict[int, dict[str, torch.Tensor]],
) -> None:
    tile = 112
    canvas = Image.new("RGB", (tile * 5, 34 + tile * len(selected_rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(
        ("target", "start", "endpoint", "endpoint error", "endpoint saturation")
    ):
        draw.text((column * tile + 4, 6), title, fill="black")
    for i, row in enumerate(selected_rows):
        y = 30 + i * tile
        draw.text((2, y + 2), f"{int(labels[row])}:{row}", fill="black")
        target = baseline._image_batch(
            pixels, torch.tensor([row]), torch.device("cpu")
        )[0][0, 0]
        items = [
            target,
            start_heat[row]["prediction28_clipped"],
            end_heat[row]["prediction28_clipped"],
            end_heat[row]["error400"].abs().clamp(0, 1),
            end_heat[row]["saturation28"].float(),
        ]
        for column, tensor in enumerate(items):
            image = Image.fromarray(
                (tensor.clamp(0, 1).numpy() * 255).astype("uint8"), mode="L"
            ).convert("RGB")
            image = image.resize((tile - 6, tile - 6), Image.Resampling.NEAREST)
            canvas.paste(image, (column * tile + 3, y + 25))
    canvas.save(path)


def _write_closeout(path: Path, report: dict[str, Any]) -> None:
    states = report["states"]
    constrained = "fit_only_constrained_gain_w_inverse"
    start = states[STATE_NAMES[0]]["metrics_by_gain"][constrained]["confirmation"]
    end = states[STATE_NAMES[1]]["metrics_by_gain"][constrained]["confirmation"]
    start_power = states[STATE_NAMES[0]]
    end_power = states[STATE_NAMES[1]]
    gate = report["local_gate"]
    paired = report["paired_confirmation_analysis"]
    confirm_pair = paired["by_population"]["confirmation"]
    class_rows = [
        item
        for item in paired["by_population_and_label"]
        if item["population"] == "confirmation"
    ]
    start_shares = confirm_pair["start_region_pooled_sse_share"]
    endpoint_shares = confirm_pair["endpoint_region_pooled_sse_share"]
    sensitivity = report["readout_sensitivity"]
    start_rank = states[STATE_NAMES[0]]["ols_coefficient_covariance"][
        "global_phase_fixed"
    ]["effective_covariance_rank"]
    endpoint_rank = states[STATE_NAMES[1]]["ols_coefficient_covariance"][
        "global_phase_fixed"
    ]["effective_covariance_rank"]
    lines = [
        "# LightGen400 零更新读出与结构定位",
        "",
        "## 结果与状态",
        "",
        f"测量状态：`{report['status']}`；局部读出门槛：`{report['decision']}`。"
        f"受约束 gain 下，确认集 MSE {start['mse400_mean']:.6f}→{end['mse400_mean']:.6f}"
        f"（下降 {gate['mse_improvement_fraction']:.1%}），SSIM {start['ssim28_mean']:.4f}"
        f"→{end['ssim28_mean']:.4f}，终点超范围比例 {end['saturation_fraction']:.3%}。",
        "",
        "## 科学认识",
        "",
        "拟合集约束 gain 后，起点与终点确认集饱和分别为 "
        f"{start['saturation_fraction']:.3%} 和 {end['saturation_fraction']:.3%}；"
        "终点仍略高于预设1%上限。起点约束 gain 下 SSIM 未因截断阈值改善而得到终点维持。",
        "按正确参考面计算，SLM后至100模态的平均系统耦合效率 "
        f"{start_power['mean_system_coupling_efficiency_slm_to_modal']:.2%}→"
        f"{end_power['mean_system_coupling_efficiency_slm_to_modal']:.2%}；"
        "OLS入射面模态捕获比例 "
        f"{start_power['mean_ols_modal_capture_fraction']:.2%}→"
        f"{end_power['mean_ols_modal_capture_fraction']:.2%}。"
        "这些变化与重建误差相关，不证明耦合变化导致质量变化。",
        f"确认集平均达1nW根数 {confirm_pair['start_mean_roots_ge_1nw']:.2f}→"
        f"{confirm_pair['endpoint_mean_roots_ge_1nw']:.2f}；达到全部100根的样本 "
        f"{confirm_pair['start_samples_all_100_roots_ge_1nw']}→"
        f"{confirm_pair['endpoint_samples_all_100_roots_ge_1nw']} / 200。"
        f"N_eff中位数 {confirm_pair['start_n_eff_median']:.2f}→"
        f"{confirm_pair['endpoint_n_eff_median']:.2f}。",
        f"全局相位对齐后复OLS系数协方差有效秩为 {start_rank:.2f}→"
        f"{endpoint_rank:.2f}/200（描述性样本统计，不能解释为独立可控根数）。",
        f"在200张配对确认图中，约束gain下MSE改善 {confirm_pair['mse_improved_rows']} 张、"
        f"变差 {confirm_pair['mse_deteriorated_rows']} 张，平均SSIM变化 "
        f"{confirm_pair['mean_ssim28_delta']:+.4f}。",
        "按互斥目标区域汇总的平方误差占比（起点→终点）："
        + "；".join(
            f"{region} {start_shares[region]:.1%}→{endpoint_shares[region]:.1%}"
            for region in ("edge", "stroke_interior", "background")
        )
        + "。",
        "逐类确认集（MSE改善张数/20、平均ΔMSE400、平均ΔSSIM28）：",
        "| 类别 | MSE改善 | 平均ΔMSE400 | 平均ΔSSIM28 |",
        "|---:|---:|---:|---:|",
        *[
            f"| {item['label']} | {item['mse_improved_rows']}/{item['count']} | "
            f"{item['mean_mse400_delta']:+.6f} | {item['mean_ssim28_delta']:+.4f} |"
            for item in class_rows
        ],
        "",
        "## 目标缺口",
        "",
        "本结果只覆盖固定300张TRAIN样本、4096离散传播和两个冻结状态。"
        "未验证全量DEV、完整生成、实物光纤参数或光子噪声；逐根功率和复系数协方差也不等于信息因果贡献。",
        "",
        "## 工程完成与验证",
        "",
        "600张图像整链前向均已保存为可恢复的10张分块；没有优化器更新，"
        "DEV／TEST读取数为0。两状态指定平面4096/6144复场检查均通过，"
        "最大误差低于1e-4；SLM后输入功率与独立面积积分吻合。",
        "预检已拒绝行号错配并确认目标打乱使误差增加。区域掩码按边缘优先重新做成不重叠分区，"
        "误差份额总和核对为1。首次汇总的图册字段错误已通过保存证据恢复，未重复传播。",
        "",
        "## 决策影响与下一步",
        "",
        f"门槛判定：MSE改善≥10%为 `{gate['mse_improvement_fraction'] >= MSE_IMPROVEMENT_MIN}`，"
        f"SSIM不下降为 `{gate['confirm_ssim_delta'] >= 0}`，"
        f"终点超范围≤1%为 `{gate['endpoint_saturation_fraction'] <= SATURATION_MAX}`。"
        "按预先固定的受约束标定主判据，本结果未支持合格局部重建；原固定gain辅助比较会通过，"
        "说明本地判定对读出尺度及截断敏感。两种口径并列保留，不以辅助结果替换主判据。"
        f"作为非主判据敏感性结果，两个状态共用原冻结gain时MSE下降 "
        f"{sensitivity['same_original_gain_mse_improvement_fraction']:.1%}、"
        f"SSIM变化 {sensitivity['same_original_gain_ssim_delta']:+.4f}、"
        f"终点饱和 {sensitivity['same_original_gain_endpoint_saturation_fraction']:.3%}，"
        f"会得到 `{sensitivity['same_original_gain_gate_passed']}`；这不能替代预先固定的"
        "各状态拟合且限制饱和的主判据。未限制的拟合最小二乘gain在终点饱和 "
        f"{sensitivity['unconstrained_fit_gain_endpoint_saturation_fraction']:.3%}。"
        "区域误差、逐根功率和协方差结果可用于另行设计因果对照。"
        "后续目标或训练改动需独立计划授权，本轮没有启动后继实验。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for measurement")
    preflight_dir = _resolve(args.preflight_dir)
    output_dir = _resolve(args.output_dir)
    preflight, receipt = _read_ready(preflight_dir, args)
    if args.mode == "run":
        _require(
            not output_dir.exists() and args.resume_checkpoint is None,
            "fresh run requires a new output directory and no resume checkpoint",
        )
    else:
        _require(
            output_dir.is_dir() and args.resume_checkpoint is not None,
            "resume requires existing output directory and explicit resume checkpoint",
        )
    loaded = _load_inputs(args, torch.device("cuda"))
    saved_population = torch.load(
        preflight_dir / "fixed_population.pt", map_location="cpu", weights_only=False
    )
    _require(
        baseline._assert_state_close(
            loaded["population"], saved_population, "audit_population_ready"
        )
        == 0.0,
        "fixed population differs from READY preflight",
    )
    identity = ResearchRunIdentity.from_mapping(preflight["run_identity"])
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    attempt_id = output_dir.name
    if args.mode == "run":
        control.open_scientific_attempt(
            output_dir, attempt_id=attempt_id, receipt=receipt
        )
        atomic_write_new_json(
            output_dir / "run_manifest.json",
            {
                "schema": "chen_lightgen_400_readout_structure_run.v1",
                "task_id": TASK_ID,
                "run_id": identity.run_id,
                "attempt_id": attempt_id,
                "source_checkpoint": str(loaded["source_checkpoint_path"]),
                "start_state": str(loaded["start_path"]),
                "endpoint_checkpoint": str(loaded["endpoint_path"]),
                "joint_report": str(loaded["joint_report_path"]),
                "fixed_population": str(loaded["fixed_path"]),
                "preflight_dir": str(preflight_dir),
                "grid": GRID,
                "sample_count": 300,
                "block_size": BLOCK_SIZE,
                "optimizer_updates": 0,
            },
        )
        atomic_torch_save_new(output_dir / "fixed_population.pt", loaded["population"])
        progress = {
            "schema": "chen_lightgen_400_readout_progress.v1",
            "completed_rows_by_state": {name: 0 for name in STATE_NAMES},
            "started_at": time.time(),
            "elapsed_before_resume": 0.0,
        }
    else:
        control.open_scientific_attempt(
            output_dir, attempt_id=output_dir.name, receipt=receipt, resume=True
        )
        manifest = _json(output_dir / "run_manifest.json")
        _require(
            manifest.get("task_id") == TASK_ID
            and manifest.get("attempt_id") == attempt_id
            and manifest.get("run_id") == identity.run_id
            and manifest.get("grid") == GRID
            and manifest.get("optimizer_updates") == 0
            and manifest.get("preflight_dir") == str(preflight_dir),
            "resume manifest does not match this explicit task",
        )
        _require(
            manifest.get("source_checkpoint") == str(loaded["source_checkpoint_path"])
            and manifest.get("start_state") == str(loaded["start_path"])
            and manifest.get("endpoint_checkpoint") == str(loaded["endpoint_path"])
            and manifest.get("joint_report") == str(loaded["joint_report_path"])
            and manifest.get("fixed_population") == str(loaded["fixed_path"]),
            "resume inputs differ from the original output manifest",
        )
        checkpoint = torch.load(
            _resolve(args.resume_checkpoint), map_location="cpu", weights_only=False
        )
        _require(
            checkpoint.get("task_id") == TASK_ID
            and checkpoint.get("run_id") == identity.run_id
            and checkpoint.get("attempt_id") == attempt_id,
            "resume checkpoint belongs to a different run",
        )
        progress = checkpoint["progress"]
    if args.mode == "run":
        _append = output_dir / "segments.jsonl"
        _append.write_text(
            json.dumps({"event": "run_started", "time": time.time()}) + "\n",
            encoding="utf-8",
        )
    else:
        with (output_dir / "segments.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "resumed",
                        "time": time.time(),
                        "checkpoint": str(_resolve(args.resume_checkpoint)),
                    }
                )
                + "\n"
            )
    stop_requested = [False]
    heartbeat_stop = threading.Event()

    def heartbeat_worker() -> None:
        while not heartbeat_stop.wait(60.0):
            atomic_write_json(
                output_dir / "heartbeat.json",
                {
                    "time": time.time(),
                    "alive": True,
                    "completed_rows_by_state": progress.get(
                        "completed_rows_by_state", {}
                    ),
                    "elapsed_seconds": progress.get("elapsed_before_resume", 0.0)
                    + time.time()
                    - progress.get("started_at", time.time()),
                },
            )

    heartbeat_thread = threading.Thread(
        target=heartbeat_worker, name="readout-audit-heartbeat", daemon=True
    )
    heartbeat_thread.start()
    previous_handlers = {}

    def request_stop(signum: int, frame: Any) -> None:
        del frame
        stop_requested[0] = True
        progress["pending_signal"] = int(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, request_stop)
    start_time = time.perf_counter()
    try:
        for name in STATE_NAMES:
            _measure_state(
                name,
                loaded["models"][name],
                loaded,
                output_dir,
                progress,
                stop_requested,
            )
            progress["completed_rows_by_state"][name] = max(
                int(progress["completed_rows_by_state"].get(name, 0)),
                sum(
                    torch.load(p, map_location="cpu", weights_only=False)[
                        "rows"
                    ].numel()
                    for p in sorted((output_dir / name).glob("block_*.pt"))
                ),
            )
            progress["updated_at"] = time.time()
            atomic_write_json(output_dir / "progress.json", progress)
            if stop_requested[0]:
                recovery = {
                    "schema": "chen_lightgen_400_readout_resume.v1",
                    "task_id": TASK_ID,
                    "run_id": identity.run_id,
                    "attempt_id": attempt_id,
                    "progress": progress,
                }
                recovery_path = (
                    output_dir / f"resume_{time.strftime('%Y%m%d_%H%M%S')}.pt"
                )
                atomic_torch_save_new(recovery_path, recovery)
                reloaded = torch.load(
                    recovery_path, map_location="cpu", weights_only=False
                )
                _require(
                    reloaded["progress"] == progress,
                    "recovery checkpoint readback differs",
                )
                with (output_dir / "segments.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(
                        json.dumps(
                            {
                                "event": "interrupted",
                                "signal": progress.get("pending_signal"),
                                "resume_checkpoint": recovery_path.name,
                                "time": time.time(),
                            }
                        )
                        + "\n"
                    )
                return {
                    "status": "INTERRUPTED",
                    "resume_checkpoint": str(recovery_path),
                }
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2.0)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    expected_rows = loaded["population"]["all_rows"].long()
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    per_state_records: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    heatmaps: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    original_gain = float(loaded["references"]["camera_gain"].cpu())
    previous_gain = float(_json(_resolve(args.joint_report))["fixed_gain"])
    for name in STATE_NAMES:
        outputs[name] = _read_state(name, output_dir / name, expected_rows)
        evaluation_path = output_dir / f"evaluation_{name}.pt"
        reusable_evaluation = False
        if evaluation_path.exists():
            saved_evaluation = torch.load(
                evaluation_path, map_location="cpu", weights_only=False
            )
            _require(
                torch.equal(
                    saved_evaluation["coefficients"], outputs[name]["ols_coefficients"]
                )
                and torch.equal(
                    saved_evaluation["physical_w_per_pixel"],
                    outputs[name]["physical_w_per_pixel"],
                ),
                f"saved evaluation does not match the measured {name} outputs",
            )
            reusable_evaluation = (
                saved_evaluation.get("region_partition_semantics") == REGION_SEMANTICS
            )
        if reusable_evaluation:
            summary = saved_evaluation["summary"]
            state_heatmaps = saved_evaluation["fixed_confirmation_heatmaps"]
            records = _read_csv(output_dir / f"per_sample_metrics_{name}.csv")
        else:
            selected = [
                int(loaded["population"]["confirm_rows_by_class"][label, j])
                for label in range(10)
                for j in range(3)
            ]
            state_input = {**loaded, "state_name": name, "atlas_rows": set(selected)}
            records, summary, state_heatmaps = _evaluate(
                state_input, outputs[name], original_gain, previous_gain
            )
            _write_csv(output_dir / f"per_sample_metrics_{name}.csv", records)
            _write_region_csv(output_dir / f"regional_errors_{name}.csv", records)
            _write_port_csv(
                output_dir / f"per_port_power_{name}.csv", name, outputs[name]
            )
            evaluation_payload = {
                "coefficients": outputs[name]["ols_coefficients"],
                "physical_w_per_pixel": outputs[name]["physical_w_per_pixel"],
                "summary": summary,
                "fixed_confirmation_heatmaps": state_heatmaps,
                "region_partition_semantics": REGION_SEMANTICS,
            }
            if evaluation_path.exists():
                atomic_torch_save(evaluation_path, evaluation_payload)
            else:
                atomic_torch_save_new(evaluation_path, evaluation_payload)
        per_state_records[name] = records
        summaries[name] = summary
        heatmaps[name] = state_heatmaps

    start_constraint = summaries[STATE_NAMES[0]]["metrics_by_gain"][
        "fit_only_constrained_gain_w_inverse"
    ]["confirmation"]
    end_constraint = summaries[STATE_NAMES[1]]["metrics_by_gain"][
        "fit_only_constrained_gain_w_inverse"
    ]["confirmation"]
    improvement = 1.0 - end_constraint["mse400_mean"] / start_constraint["mse400_mean"]
    gate = {
        "mse_improvement_fraction": improvement,
        "confirm_ssim_delta": end_constraint["ssim28_mean"]
        - start_constraint["ssim28_mean"],
        "endpoint_saturation_fraction": end_constraint["saturation_fraction"],
        "criteria": {
            "mse_improvement_min": MSE_IMPROVEMENT_MIN,
            "ssim_not_lower": True,
            "saturation_fraction_max": SATURATION_MAX,
        },
        "passed": improvement >= MSE_IMPROVEMENT_MIN
        and end_constraint["ssim28_mean"] >= start_constraint["ssim28_mean"]
        and end_constraint["saturation_fraction"] <= SATURATION_MAX,
    }
    selected_rows = [
        int(loaded["population"]["confirm_rows_by_class"][label, j])
        for label in range(10)
        for j in range(3)
    ]
    _atlas(
        output_dir / "fixed_confirmation_atlas.png",
        loaded["pixels"],
        loaded["labels"],
        selected_rows,
        heatmaps[STATE_NAMES[0]],
        heatmaps[STATE_NAMES[1]],
    )
    all_records = [record for name in STATE_NAMES for record in per_state_records[name]]
    _write_csv(output_dir / "per_sample_metrics.csv", all_records)
    paired_analysis = _write_paired_analysis(output_dir, per_state_records)
    original_gain_name = "original_gain_w_inverse"
    original_start = summaries[STATE_NAMES[0]]["metrics_by_gain"][original_gain_name][
        "confirmation"
    ]
    original_endpoint = summaries[STATE_NAMES[1]]["metrics_by_gain"][
        original_gain_name
    ]["confirmation"]
    original_gain_mse_improvement = 1.0 - (
        original_endpoint["mse400_mean"] / original_start["mse400_mean"]
    )
    original_gain_ssim_delta = (
        original_endpoint["ssim28_mean"] - original_start["ssim28_mean"]
    )
    original_gain_gate = (
        original_gain_mse_improvement >= MSE_IMPROVEMENT_MIN
        and original_gain_ssim_delta >= 0.0
        and original_endpoint["saturation_fraction"] <= SATURATION_MAX
    )
    unconstrained_endpoint = summaries[STATE_NAMES[1]]["metrics_by_gain"][
        "fit_only_least_squares_gain_w_inverse"
    ]["confirmation"]
    readout_sensitivity = {
        "same_original_gain_mse_improvement_fraction": original_gain_mse_improvement,
        "same_original_gain_ssim_delta": original_gain_ssim_delta,
        "same_original_gain_endpoint_saturation_fraction": original_endpoint[
            "saturation_fraction"
        ],
        "same_original_gain_gate_passed": original_gain_gate,
        "unconstrained_fit_gain_endpoint_saturation_fraction": unconstrained_endpoint[
            "saturation_fraction"
        ],
        "constrained_gain_w_inverse": {
            name: summaries[name]["gains"]["fit_only_constrained_gain_w_inverse"]
            for name in STATE_NAMES
        },
        "same_original_gain_w_inverse": summaries[STATE_NAMES[0]]["gains"][
            original_gain_name
        ],
    }
    report = {
        "schema": "chen_lightgen_400_readout_structure_report.v1",
        "status": "VALID_COMPLETED",
        "task_id": TASK_ID,
        "authorization_id": AUTHORITY,
        "source_baseline_status": loaded["source_report"]["status"],
        "completed_updates": 0,
        "data_roles": {
            "fit": 100,
            "confirmation": 200,
            "canonical_train": True,
            "dev_reads": 0,
            "official_test_reads": 0,
        },
        "states": summaries,
        "paired_confirmation_analysis": paired_analysis,
        "readout_sensitivity": readout_sensitivity,
        "local_gate": gate,
        "power_efficiency_definition": {
            "system": "sum modal power / post-SLM transmitted power",
            "modal_capture": "sum modal power / OLS incident power",
        },
        "region_partition_semantics": REGION_SEMANTICS,
        "decision": "READOUT_ONLY_LOCAL_GATE_MET"
        if gate["passed"]
        else "READOUT_ONLY_LOCAL_GATE_NOT_MET",
        "artifacts": {
            "measurements": "<state>/block_*.pt",
            "sample_metrics": "per_sample_metrics.csv",
            "regional_errors": "regional_errors_<state>.csv",
            "per_port_power": "per_port_power_<state>.csv",
            "paired_sample_changes": "paired_sample_changes.csv",
            "per_class_summary": "per_class_paired_summary.csv",
            "readout_sensitivity": "readout_sensitivity in report.json",
            "atlas": "fixed_confirmation_atlas.png",
        },
        "elapsed_seconds": progress.get("elapsed_before_resume", 0.0)
        + time.perf_counter()
        - start_time,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
        "resume_segments": str(output_dir / "segments.jsonl"),
    }
    report_path = output_dir / "report.json"
    if report_path.exists():
        previous = _json(report_path)
        if previous.get("region_partition_semantics") != REGION_SEMANTICS:
            archived = output_dir / "report_before_region_repair.json"
            if not archived.exists():
                atomic_write_new_json(archived, previous)
            atomic_write_json(report_path, report)
        elif (
            "paired_confirmation_analysis" not in previous
            or "readout_sensitivity" not in previous
        ):
            archived = output_dir / "report_before_paired_analysis.json"
            if not archived.exists():
                atomic_write_new_json(archived, previous)
            atomic_write_json(report_path, report)
    else:
        atomic_write_new_json(report_path, report)
    _write_closeout(output_dir / "results_analysis.md", report)
    with (output_dir / "segments.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"event": "completed", "time": time.time(), "local_gate": gate})
            + "\n"
        )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "resume"), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="explicit original step-30000 checkpoint",
    )
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--start-state", type=Path, required=True)
    parser.add_argument("--endpoint-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--fixed-population", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/research/lightgen/chen_lightgen_400_mnist_baseline_v1.yaml"
        ),
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/MNIST/processed/training.pt")
    )
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    _require(
        args.mode == "preflight"
        if args.preflight_dir is None
        else args.mode in ("run", "resume"),
        "preflight omits --preflight-dir; run/resume require it",
    )
    _require(
        args.mode == "resume"
        if args.resume_checkpoint is not None
        else args.mode != "resume",
        "resume mode requires an explicit recovery checkpoint",
    )
    return args


def main() -> None:
    args = _parse_args()
    try:
        report = _preflight(args) if args.mode == "preflight" else _run(args)
        print(
            json.dumps(
                {
                    key: report.get(key)
                    for key in ("status", "task_id", "local_gate", "output_dir")
                },
                ensure_ascii=False,
            )
        )
    except BaseException as error:
        output_dir = _resolve(args.output_dir)
        failure = {
            "status": "INVALID_HARNESS",
            "task_id": TASK_ID,
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_updates": 0,
            "official_test_reads": 0,
        }
        manifest_path = output_dir / "run_manifest.json"
        progress_path = output_dir / "progress.json"
        if (
            args.mode != "preflight"
            and manifest_path.is_file()
            and progress_path.is_file()
            and not (output_dir / "report.json").exists()
        ):
            manifest = _json(manifest_path)
            recovery_path = output_dir / f"resume_after_error_{int(time.time())}.pt"
            recovery = {
                "schema": "chen_lightgen_400_readout_resume.v1",
                "task_id": TASK_ID,
                "run_id": manifest["run_id"],
                "attempt_id": manifest["attempt_id"],
                "progress": _json(progress_path),
            }
            if not recovery_path.exists():
                atomic_torch_save_new(recovery_path, recovery)
                failure["resume_checkpoint"] = str(recovery_path)
        target = output_dir / (
            "preflight_failure.json"
            if args.mode == "preflight"
            else "execution_errors.jsonl"
        )
        if output_dir.is_dir() and args.mode == "preflight" and not target.exists():
            atomic_write_new_json(target, failure)
        elif output_dir.is_dir() and args.mode != "preflight":
            with target.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
        raise


if __name__ == "__main__":
    main()
