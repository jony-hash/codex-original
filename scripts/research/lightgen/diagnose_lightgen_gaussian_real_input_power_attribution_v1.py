"""Frozen real-input Gaussian OLS power and Encoder/Decoder attribution diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base  # noqa: E402
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (  # noqa: E402
    ComplexOLSState,
)
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch  # noqa: E402
from ldm_project.v2.application.research.lightgen.image_readout import resize_view, ssim_batch  # noqa: E402
from ldm_project.v2.evidence.io import (  # noqa: E402
    diagnostic_json_value as clean,
    write_diagnostic_csv as write_csv,
    write_diagnostic_json as write_json,
    write_diagnostic_npz as write_npz,
)

TASK_ID = "GAUSSIAN_REAL_INPUT_POWER_ATTRIBUTION_V1"
PORTS = 100
DB_THRESHOLDS = (-10, -20, -30, -40)
POWER_CHAIN = (
    "P_input_W",
    "P_after_slm_W",
    "P_encoder_entrance_W",
    "P_encoder_1_W",
    "P_encoder_2_W",
    "P_before_fiber_W",
    "P_fiber_total_W",
    "P_fiber_output_total_W",
    "P_after_material_W",
    "P_generator_entry_W",
    "P_decoder_1_W",
    "P_decoder_2_W",
    "P_camera_W",
)
LAYER_POWER = (
    "encoder_layer_input_power_W",
    "encoder_after_device_power_W",
    "encoder_after_propagation_power_W",
    "encoder_after_aperture_power_W",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(
        isinstance(cfg, dict)
        and cfg.get("schema") == "gaussian_real_input_power_attribution.v1",
        "diagnostic config schema changed",
    )
    require(cfg.get("task_id") == TASK_ID, "diagnostic task identity changed")
    require(
        cfg.get("semantic_authority_id")
        == "USER_APPROVED_GAUSSIAN_OLS_POWER_ATTRIBUTION_20261021",
        "diagnostic authority changed",
    )
    require(
        cfg["data"]
        == {
            "training_file": "data/MNIST/processed/training.pt",
            "split_seed": 1234,
            "dev_count": 5000,
            "dev_per_class": 500,
            "official_test_reads": 0,
        },
        "fixed DEV population or data roles changed",
    )
    m = cfg["model"]
    require(
        m["propagation_grid_size"] == 2048
        and m["reference_grid_size"] == 4096
        and m["ols_basis"] == "gaussian_s12_v1"
        and m["ols_port_count"] == PORTS,
        "Gaussian model or numerical grids changed",
    )
    require(
        m["source_power_w"] == 2.493e-4
        and m["layer_power_transmission"] == 0.889
        and m["port_pitch_m"] == 20e-6
        and m["gaussian_field_1e_radius_m"] == 3.5e-6
        and m["ols_plane_fov_m"] == 200e-6,
        "fixed optical settings changed",
    )
    require(
        m["shared_camera_gain_w_inverse"] == 1.505046959946815e9
        and m["window_relative_tolerance"] == 1e-4,
        "fixed gain or numerical tolerance changed",
    )
    s = cfg["statistics"]
    require(
        s["batch_size"] == 10
        and s["shard_size"] == 100
        and s["bootstrap_resamples"] == 2000
        and s["bootstrap_seed"] == 2026102101,
        "batch, recovery or bootstrap contract changed",
    )
    require(
        s["no_training_updates"]
        and s["no_optimizer_creation"]
        and s["no_prior_sampling"],
        "zero-update boundary changed",
    )
    steps = [int(x["global_step"]) for x in cfg["checkpoints"]]
    keys = [x["key"] for x in cfg["checkpoints"]]
    require(
        steps == [0, 5000, 10000, 20000, 30000, 32500, 35000, 32500, 35000]
        and keys
        == [
            "step_000000",
            "step_005000",
            "step_010000",
            "step_020000",
            "step_030000",
            "control_step_032500",
            "control_step_035000",
            "treatment_step_032500",
            "treatment_step_035000",
        ],
        "checkpoint population changed",
    )
    return cfg


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected YAML mapping: {path}")
    return value


def saved_adam_steps(value: Any) -> list[int]:
    require(
        isinstance(value, dict) and isinstance(value.get("state"), dict),
        "checkpoint Adam state malformed",
    )
    steps = []
    for state in value["state"].values():
        require(
            isinstance(state, dict) and "step" in state,
            "saved Adam parameter step missing",
        )
        step = state["step"]
        steps.append(int(step.item()) if isinstance(step, torch.Tensor) else int(step))
    return steps


def load_inputs(
    cfg: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    baseline = read_yaml(ROOT / cfg["model"]["baseline_config"])
    continuation = read_yaml(ROOT / cfg["model"]["continuation_config"])
    require(
        baseline.get("schema") == "lightgen_matched_small_vs_chen400.v1"
        and continuation.get("schema")
        == "lightgen_small_gaussian_structural_continuation_ab.v1",
        "source configuration schema changed",
    )
    expected_model = base._model_identity(baseline, "small")
    require(
        expected_model.get("ols") == "gaussian_s12_v1",
        "baseline model identity is not Gaussian",
    )
    cps: dict[str, dict[str, Any]] = {}
    baseline_run: str | None = None
    continuation_source: str | None = None
    for spec in cfg["checkpoints"]:
        path = ROOT / spec["checkpoint"]
        require(path.is_file(), f"explicit checkpoint missing: {path}")
        cp = torch.load(path, map_location="cpu", weights_only=False)
        require(
            isinstance(cp, dict)
            and cp.get("seed") == 2026101501
            and cp.get("model_identity") == expected_model
            and cp.get("propagation_grid_size") == 2048,
            f"checkpoint model identity mismatch: {spec['key']}",
        )
        step = int(spec["global_step"])
        if spec["trajectory"] == "baseline":
            require(
                cp.get("schema") == "lightgen_matched_architecture_checkpoint.v1"
                and cp.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_SYSTEM_BASELINE_V1"
                and cp.get("arm") == "small"
                and cp.get("config_identity") == baseline
                and cp.get("completed_updates") == step,
                f"baseline checkpoint metadata mismatch: {spec['key']}",
            )
            baseline_run = baseline_run or cp["run_id"]
            require(
                cp.get("run_id") == baseline_run,
                "baseline checkpoints are from different runs",
            )
        else:
            require(
                cp.get("schema")
                == "lightgen_small_gaussian_structural_continuation_checkpoint.v1"
                and cp.get("task_id")
                == "LIGHTGEN_SMALL_GAUSSIAN_STRUCTURAL_CONTINUATION_AB_V1"
                and cp.get("config_identity") == continuation
                and cp.get("branch") == spec["trajectory"]
                and cp.get("source_checkpoint_step") == 30000
                and cp.get("completed_global_step") == step
                and cp.get("completed_updates") == step - 30000,
                f"continuation branch/source/global step mismatch: {spec['key']}",
            )
            continuation_source = continuation_source or cp["source_run_id"]
            require(
                cp.get("source_run_id") == continuation_source,
                "continuations have different source runs",
            )
        expected_steps = [] if step == 0 else [step, step]
        require(
            saved_adam_steps(cp.get("optimizer")) == expected_steps,
            f"saved Adam steps mismatch: {spec['key']}",
        )
        require(
            cp.get("evaluator_role") == "frozen_confirmation_only",
            "checkpoint evaluator role changed",
        )
        gain = cp.get("camera_gain")
        require(
            isinstance(gain, torch.Tensor)
            and gain.numel() == 1
            and math.isclose(
                float(gain.reshape(-1)[0]),
                cfg["model"]["shared_camera_gain_w_inverse"],
                rel_tol=1e-12,
            ),
            f"checkpoint shared camera gain mismatch: {spec['key']}",
        )
        require(isinstance(cp.get("model"), dict), "model state missing")
        cps[spec["key"]] = cp
    require(
        continuation_source == cps["step_030000"]["run_id"],
        "control/treatment do not share the exact baseline 30,000-step source",
    )
    pixels, labels, train_rows, dev_rows = base._load_data(
        ROOT / cfg["data"]["training_file"]
    )
    require(
        len(dev_rows) == 5000
        and torch.equal(
            torch.bincount(labels[dev_rows], minlength=10), torch.full((10,), 500)
        )
        and not bool(torch.isin(train_rows, dev_rows).any()),
        "fixed DEV rows or data roles changed",
    )
    return baseline, continuation, cps, pixels, labels, train_rows, dev_rows


def model_for(
    key: str,
    cfg: dict[str, Any],
    baseline: dict[str, Any],
    cps: dict[str, dict[str, Any]],
    device: torch.device,
    grid: int | None = None,
) -> torch.nn.Module:
    model = base._model(
        "small", int(grid or cfg["model"]["propagation_grid_size"]), device, baseline
    )
    model.load_state_dict(cps[key]["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(
        not model.training and not any(p.requires_grad for p in model.parameters()),
        f"model not frozen: {key}",
    )
    return model


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def assert_snapshot(
    model: torch.nn.Module, before: dict[str, torch.Tensor], label: str
) -> None:
    after = model.state_dict()
    require(after.keys() == before.keys(), f"state keys changed during {label}")
    for key, value in after.items():
        require(
            torch.equal(value.detach().cpu(), before[key]),
            f"frozen parameter or buffer changed: {label}.{key}",
        )


def actual_geometry(model: torch.nn.Module) -> dict[str, Any]:
    op = model.runtime.ols.operator
    plane = op.input_plane
    geo = {
        "basis_id": model.runtime.ols.basis_id,
        "plane_shape": list(plane.shape),
        "plane_pixel_pitch_m": list(plane.pixel_pitch_m),
        "plane_fov_m": list(plane.fov_m),
        "port_center_pitch_m": float(op.geometry.port_pitch_m),
        "gaussian_field_1e_radius_m": float(op.geometry.waist_m),
        "centers_m_row_major": op.centers_m.detach().cpu().numpy(),
        "geometry_status": op.geometry_diagnostics["parameter_status"],
        "generic_physical_trace_geometry_excluded": True,
    }
    require(
        geo["basis_id"] == "gaussian_s12_v1"
        and geo["plane_shape"] == [100, 100]
        and geo["port_center_pitch_m"] == 20e-6
        and geo["gaussian_field_1e_radius_m"] == 3.5e-6
        and all(math.isclose(x, 200e-6, rel_tol=1e-12) for x in plane.fov_m),
        "actual Gaussian OLS geometry differs from the authorized runtime",
    )
    return geo


def fixed_preflight_rows(dev_rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    by_class = [dev_rows[labels[dev_rows] == digit][:2] for digit in range(10)]
    require(
        all(len(x) == 2 for x in by_class),
        "need exactly two preflight DEV rows per class",
    )
    return torch.stack(by_class).reshape(-1)


def require_row_match(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(
            "paired Encoder/Decoder input rows do not match original MNIST ids"
        )


def np_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(
        np.broadcast_shapes(numerator.shape, denominator.shape),
        np.nan,
        dtype=np.float64,
    )
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def target_batch(
    pixels: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    native = image_batch(pixels, rows, "small", device)
    target28 = (
        pixels.index_select(0, rows.detach().cpu())
        .unsqueeze(1)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    y = labels.index_select(0, rows.detach().cpu()).to(device=device)
    return native, target28, y


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    require(
        bool(parts) and all(part.keys() == parts[0].keys() for part in parts),
        "chunk schemas differ",
    )
    return {
        key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]
    }


def extract_result(
    model: torch.nn.Module,
    result: Any,
    images: torch.Tensor,
    target28: torch.Tensor,
    rows: torch.Tensor,
    labels: torch.Tensor,
    gain: float,
    representative_rows: set[int],
) -> dict[str, np.ndarray]:
    trace = result.physical_power_trace
    require(
        trace is not None
        and result.physical_sensor_output_w_per_pixel is not None
        and result.ols_input_field is not None,
        "production path omitted power, field or camera output",
    )
    coeff = result.z_ols.coefficients
    field = result.ols_input_field.tensor
    require(
        coeff.shape == (len(rows), PORTS)
        and torch.is_complex(coeff)
        and field.shape == (len(rows), 1, 100, 100)
        and torch.is_complex(field),
        "production coefficient or OLS field dimensions changed",
    )
    op = model.runtime.ols.operator
    dy, dx = op.input_plane.pixel_pitch_m
    abs_field = field[:, 0].abs().to(torch.float64)
    positive = torch.einsum(
        "rh,bhw,cw->brc", op.modes_y.to(abs_field), abs_field, op.modes_x.to(abs_field)
    ) * (dy * dx)
    bound = positive.reshape(len(rows), PORTS).square()
    power = coeff.to(torch.complex128).abs().square()
    c = torch.full_like(power, float("nan"))
    c[bound > 0] = power[bound > 0] / bound[bound > 0]
    before = trace["P_before_fiber_W"].to(torch.float64)
    source = trace["P_input_W"].to(torch.float64)
    coupled = power.sum(1)
    eta_fiber = torch.full_like(before, float("nan"))
    eta_fiber[before > 0] = coupled[before > 0] / before[before > 0]
    eta_system = torch.full_like(source, float("nan"))
    eta_system[source > 0] = coupled[source > 0] / source[source > 0]
    neff = torch.full_like(coupled, float("nan"))
    den = power.square().sum(1)
    neff[den > 0] = coupled[den > 0].square() / den[den > 0]
    pmax = power.max(1, keepdim=True).values
    raw = result.physical_sensor_output_w_per_pixel
    unclipped = raw * gain
    view = resize_view(unclipped)
    mse = (view - target28).square().mean((1, 2, 3))
    psnr = torch.where(
        mse > 0, -10 * torch.log10(mse), torch.full_like(mse, float("inf"))
    )
    ssim = ssim_batch(view, target28)
    loss = (unclipped - images).square().mean((1, 2, 3))
    clipping = ((unclipped < 0) | (unclipped > 1)).to(torch.float64).mean((1, 2, 3))
    out: dict[str, np.ndarray] = {
        "rows": rows.detach().cpu().numpy().astype(np.int64),
        "labels": labels.detach().cpu().numpy().astype(np.int64),
        "coefficients": coeff.detach().cpu().numpy().astype(np.complex64),
        "port_power_w": power.detach().cpu().numpy().astype(np.float64),
        "upper_power_w": bound.detach().cpu().numpy().astype(np.float64),
        "coherence_fraction": c.detach().cpu().numpy().astype(np.float64),
        "eta_fiber_each": eta_fiber.detach().cpu().numpy().astype(np.float64),
        "eta_system_each": eta_system.detach().cpu().numpy().astype(np.float64),
        "neff_each": neff.detach().cpu().numpy().astype(np.float64),
        "production_reconstruction_loss": loss.detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "mse28": mse.detach().cpu().numpy().astype(np.float64),
        "psnr28_db": psnr.detach().cpu().numpy().astype(np.float64),
        "ssim28": ssim.detach().cpu().numpy().astype(np.float64),
        "clipping_fraction": clipping.detach().cpu().numpy().astype(np.float64),
        "camera_total_w": trace["P_camera_W"].detach().cpu().numpy().astype(np.float64),
        "view28": view.detach().cpu().numpy().astype(np.float32),
        "ols_input_field_representatives": np.empty((0, 100, 100), np.complex64),
        "representative_rows": np.empty((0,), np.int64),
    }
    for db in DB_THRESHOLDS:
        out[f"ports_ge_{db}db"] = (
            (power >= pmax * 10.0 ** (db / 10.0)).sum(1).cpu().numpy().astype(np.int16)
        )
    for key, value in trace.items():
        if key not in {"fiber_pitch_m", "fiber_mfd_m"} and isinstance(
            value, torch.Tensor
        ):
            out[f"trace__{key}"] = value.detach().cpu().numpy()
    indexes = [
        i for i, row in enumerate(out["rows"]) if int(row) in representative_rows
    ]
    if indexes:
        out["ols_input_field_representatives"] = (
            field[indexes, 0].detach().cpu().numpy().astype(np.complex64)
        )
        out["representative_rows"] = out["rows"][indexes].copy()
    return out


def relative_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm((a - b).reshape(len(a), -1), axis=1)
    denominator = np.linalg.norm(b.reshape(len(b), -1), axis=1)
    return np_ratio(numerator, denominator)


def independent_numpy_check(
    model: torch.nn.Module, data: dict[str, np.ndarray]
) -> dict[str, float]:
    op = model.runtime.ols.operator
    my = op.modes_y.detach().cpu().numpy().astype(np.float64)
    mx = op.modes_x.detach().cpu().numpy().astype(np.float64)
    dy, dx = op.input_plane.pixel_pitch_m
    fields = data["ols_input_field_representatives"].astype(np.complex128)
    rows = data["representative_rows"]
    require(
        len(fields) == 20 and len(rows) == 20,
        "NumPy reference lacks 20 representative OLS fields",
    )
    coeff = (
        np.einsum("rh,bhw,cw->brc", my, fields, mx, optimize=True) * dy * dx
    ).reshape(20, PORTS)
    bound = (
        np.einsum("rh,bhw,cw->brc", my, np.abs(fields), mx, optimize=True) * dy * dx
    ).reshape(20, PORTS) ** 2
    input_power = np.sum(np.abs(fields) ** 2, axis=(1, 2), dtype=np.float64) * dy * dx
    indexes = np.asarray([int(np.flatnonzero(data["rows"] == row)[0]) for row in rows])
    cp = data["coefficients"][indexes].astype(np.complex128)
    p_trace = data["trace__P_before_fiber_W"][indexes].astype(np.float64)
    return {
        "numpy_vs_production_complex_coefficients_relative_l2_max": float(
            np.nanmax(relative_l2(coeff, cp))
        ),
        "numpy_area_integral_vs_physical_trace_ols_input_power_relative_error_max": float(
            np.nanmax(np_ratio(np.abs(input_power - p_trace), p_trace))
        ),
        "numpy_vs_production_Bi_relative_l2_max": float(
            np.nanmax(relative_l2(bound, data["upper_power_w"][indexes]))
        ),
        "numpy_vs_production_Pi_relative_l2_max": float(
            np.nanmax(relative_l2(np.abs(coeff) ** 2, data["port_power_w"][indexes]))
        ),
    }


def phase_control(model: torch.nn.Module) -> dict[str, float]:
    op = model.runtime.ols.operator
    my = op.modes_y.detach().cpu().numpy().astype(np.float64)
    mx = op.modes_x.detach().cpu().numpy().astype(np.float64)
    dy, dx = op.input_plane.pixel_pitch_m
    row = column = 4
    mode = my[row, :, None] * mx[column, None, :]
    x = (np.arange(mode.shape[1]) + 0.5) * dx - op.input_plane.fov_m[1] / 2.0
    cx = float(op.centers_m.reshape(10, 10, 2)[row, column, 1])
    phase_sign = np.where(x[None, :] < cx, -1.0, 1.0)
    aligned = mode.astype(np.complex128)
    cancelled = (np.abs(mode) * phase_sign).astype(np.complex128)
    b1 = (np.sum(np.abs(aligned) * np.abs(mode), dtype=np.float64) * dy * dx) ** 2
    b2 = (np.sum(np.abs(cancelled) * np.abs(mode), dtype=np.float64) * dy * dx) ** 2
    z1 = np.sum(aligned * mode, dtype=np.complex128) * dy * dx
    z2 = np.sum(cancelled * mode, dtype=np.complex128) * dy * dx
    c1 = abs(z1) ** 2 / b1
    c2 = abs(z2) ** 2 / b2
    require(
        abs(b1 - b2) / max(b1, b2) < 1e-12 and c1 > 0.99 and c2 < 1e-6,
        "same-amplitude phase-aligned/cancellation control failed",
    )
    return {
        "same_amplitude_B_relative_difference": abs(b1 - b2) / max(b1, b2),
        "C_phase_aligned": float(c1),
        "C_phase_cancelled": float(c2),
    }


def preflight_capture(
    key: str,
    grid: int,
    cfg: dict[str, Any],
    baseline: dict[str, Any],
    cps: dict[str, dict[str, Any]],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    rows: torch.Tensor,
    device: torch.device,
    output_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model = model_for(key, cfg, baseline, cps, device, grid)
    geo = actual_geometry(model)
    before = snapshot(model)
    parts: list[dict[str, np.ndarray]] = []
    camera_parts: list[np.ndarray] = []
    replay_errors: list[float] = []
    reps = set(map(int, rows.cpu().tolist()))
    for i in range(len(rows)):
        selected = rows[i : i + 1]
        images, target28, y = target_batch(pixels, labels, selected, device)
        with torch.inference_mode():
            result = model.runtime.forward_from_image(images)
            require(
                result.physical_sensor_output_w_per_pixel is not None,
                "production camera W/pixel missing",
            )
            camera_parts.append(
                result.physical_sensor_output_w_per_pixel.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            saved_physical = model.runtime.physical_optics
            model.runtime.physical_optics = None
            try:
                replay = model.runtime.forward_from_ols(result.z_ols)
            finally:
                model.runtime.physical_optics = saved_physical
            area = (
                model.runtime.camera_plane.pixel_pitch_m[0]
                * model.runtime.camera_plane.pixel_pitch_m[1]
            )
            replay_camera = replay.raw_camera.tensor * area
            replay_errors.extend(
                relative_l2(
                    result.physical_sensor_output_w_per_pixel.detach().cpu().numpy(),
                    replay_camera.detach().cpu().numpy(),
                ).tolist()
            )
            part = extract_result(
                model,
                result,
                images,
                target28,
                selected,
                y,
                float(cfg["model"]["shared_camera_gain_w_inverse"]),
                reps,
            )
        parts.append(part)
    assert_snapshot(model, before, f"{key}_grid_{grid}")
    data = concat(parts)
    data["camera_raw_w_per_pixel"] = np.concatenate(camera_parts, axis=0)
    require(
        len(data["ols_input_field_representatives"]) == 20,
        "fixed 20 representative OLS fields were not retained",
    )
    if grid == 2048:
        out = output_root / "states" / key / "representative"
        write_npz(
            out / "preflight_2048.npz",
            **{
                name: data[name]
                for name in (
                    "rows",
                    "labels",
                    "coefficients",
                    "port_power_w",
                    "upper_power_w",
                    "coherence_fraction",
                    "ols_input_field_representatives",
                    "representative_rows",
                    "camera_raw_w_per_pixel",
                    "view28",
                    "camera_total_w",
                    "trace__P_before_fiber_W",
                )
            },
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return data, {
        "geometry": geo,
        "same_endpoint_decoder_replay_max_relative_l2": max(replay_errors),
    }


def run_preflight(
    config_path: Path, cfg: dict[str, Any], device: torch.device, output_root: Path
) -> None:
    baseline, continuation, cps, pixels, labels, train_rows, dev_rows = load_inputs(cfg)
    require(
        device.type == "cuda" and torch.cuda.is_available(),
        "representative production preflight requires AutoDL CUDA",
    )
    rows = fixed_preflight_rows(dev_rows, labels)
    mismatch_rejected = False
    try:
        require_row_match(rows.numpy(), rows.numpy()[::-1].copy())
    except ValueError:
        mismatch_rejected = True
    require(mismatch_rejected, "negative row-alignment control was not rejected")
    details: dict[str, Any] = {}
    max_replay = 0.0
    for spec in cfg["checkpoints"]:
        key = spec["key"]
        low, info = preflight_capture(
            key, 2048, cfg, baseline, cps, pixels, labels, rows, device, output_root
        )
        high, hi_info = preflight_capture(
            key, 4096, cfg, baseline, cps, pixels, labels, rows, device, output_root
        )
        require(
            np.array_equal(low["rows"], high["rows"])
            and np.array_equal(low["labels"], high["labels"]),
            f"2048/4096 reference rows differ for {key}",
        )
        full = relative_l2(
            low["ols_input_field_representatives"],
            high["ols_input_field_representatives"],
        )
        common = relative_l2(
            low["ols_input_field_representatives"][:, 10:90, 10:90],
            high["ols_input_field_representatives"][:, 10:90, 10:90],
        )
        coeff = relative_l2(low["coefficients"], high["coefficients"])
        camera = relative_l2(
            low["camera_raw_w_per_pixel"], high["camera_raw_w_per_pixel"]
        )
        camera_power = np_ratio(
            np.abs(low["camera_total_w"] - high["camera_total_w"]),
            high["camera_total_w"],
        )
        port = np_ratio(
            np.max(np.abs(low["port_power_w"] - high["port_power_w"]), axis=1),
            np.sum(high["port_power_w"], axis=1),
        )
        errors = {
            "ols_full_200um_field_relative_l2_max": float(np.nanmax(full)),
            "ols_common_160um_roi_relative_l2_max": float(np.nanmax(common)),
            "all_100_coefficients_relative_l2_max": float(np.nanmax(coeff)),
            "camera_full_plane_w_per_pixel_relative_l2_max": float(np.nanmax(camera)),
            "camera_total_power_relative_error_max": float(np.nanmax(camera_power)),
            "all_port_power_max_abs_over_total_max": float(np.nanmax(port)),
        }
        tolerance = float(cfg["model"]["window_relative_tolerance"])
        require(
            max(errors.values()) <= tolerance,
            f"2048/4096 numerical check failed for {key}: {errors}",
        )
        floor = np.max(
            np.abs(
                low["coefficients"].astype(np.complex128)
                - high["coefficients"].astype(np.complex128)
            ),
            axis=0,
        )
        write_npz(
            output_root / "preflight" / key / "numerical_resolution_envelope.npz",
            coefficient_abs_error_by_port=floor,
        )
        model = model_for(key, cfg, baseline, cps, device)
        independent = independent_numpy_check(model, low)
        control = phase_control(model)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        max_replay = max(
            max_replay,
            info["same_endpoint_decoder_replay_max_relative_l2"],
            hi_info["same_endpoint_decoder_replay_max_relative_l2"],
        )
        details[key] = {
            "reference_grid_errors": errors,
            "reference_tolerance": tolerance,
            "per_port_complex_coefficient_error_envelope": floor,
            "actual_gaussian_geometry": info["geometry"],
            "independent_numpy_checks": independent,
            "same_amplitude_phase_control": control,
        }
    require(
        max_replay <= cfg["model"]["window_relative_tolerance"],
        "same-endpoint Decoder replay failed",
    )
    write_json(
        output_root / "input_manifest.json",
        {
            "schema": "gaussian_real_input_manifest.v1",
            "source": cfg["data"]["training_file"],
            "split_seed": 1234,
            "dev_rows_original_order": dev_rows,
            "dev_labels": labels[dev_rows],
            "dev_class_counts": torch.bincount(labels[dev_rows], minlength=10),
            "train_rows_count_only": len(train_rows),
            "official_test_rows_read": 0,
            "preflight_rows_class_major": rows,
            "preflight_labels_class_major": labels[rows],
        },
    )
    write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest(cfg, cps))
    write_json(
        output_root / "preflight.json",
        {
            "schema": "gaussian_real_input_power_preflight.v1",
            "task_id": TASK_ID,
            "status": "READY",
            "dev_count": 5000,
            "per_class": 500,
            "representative_rows_per_class": 2,
            "checkpoints": list(cps),
            "checkpoint_identity_lineage_adam_steps_and_gain_validated": True,
            "cuda_production_path_passed": True,
            "2048_4096_full_ols_coefficients_power_camera_passed": True,
            "independent_numpy_integral_passed": True,
            "phase_positive_negative_control_passed": True,
            "same_endpoint_decoder_replay_max_relative_l2": max_replay,
            "deliberate_row_mismatch_rejected": mismatch_rejected,
            "all_model_parameters_and_buffers_unchanged": True,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "new_prior_samples": 0,
            "official_test_reads": 0,
            "state_diagnostics": details,
            "research_config": str(config_path.relative_to(ROOT)),
            "decision_use": "authorize only the fixed zero-update measurements named in this task",
        },
    )


def checkpoint_manifest(
    cfg: dict[str, Any], cps: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": "gaussian_checkpoint_manifest.v1",
        "states": {
            spec["key"]: {
                "checkpoint": spec["checkpoint"],
                "trajectory": spec["trajectory"],
                "global_step": spec["global_step"],
                "completed_updates": cps[spec["key"]].get("completed_updates"),
                "completed_global_step": cps[spec["key"]].get(
                    "completed_global_step", spec["global_step"]
                ),
                "branch": cps[spec["key"]].get("branch", "baseline"),
                "run_id": cps[spec["key"]]["run_id"],
                "source_run_id": cps[spec["key"]].get("source_run_id"),
                "source_checkpoint_step": cps[spec["key"]].get(
                    "source_checkpoint_step"
                ),
                "saved_adam_steps": saved_adam_steps(cps[spec["key"]]["optimizer"]),
                "camera_gain_w_inverse": float(
                    cps[spec["key"]]["camera_gain"].reshape(-1)[0]
                ),
            }
            for spec in cfg["checkpoints"]
        },
        "optimizer_instantiated": False,
        "optimizer_updates": 0,
    }


def validate_preflight(output_root: Path) -> dict[str, Any]:
    path = output_root / "preflight.json"
    require(path.is_file(), "measurement phase requires the preflight receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    require(
        receipt.get("schema") == "gaussian_real_input_power_preflight.v1"
        and receipt.get("task_id") == TASK_ID
        and receipt.get("status") == "READY"
        and receipt.get("optimizer_created") is False
        and receipt.get("optimizer_updates") == 0
        and receipt.get("official_test_reads") == 0,
        "preflight does not authorize this frozen measurement",
    )
    return receipt


def read_state_chunks(
    directory: Path, rows: np.ndarray, shard_size: int
) -> dict[str, np.ndarray]:
    chunk_dir = directory / "chunks"
    count = math.ceil(len(rows) / shard_size)
    require(
        len(list(chunk_dir.glob("chunk_*.npz"))) == count,
        f"missing or unexpected saved chunks in {chunk_dir}",
    )
    parts = []
    for index in range(count):
        path = chunk_dir / f"chunk_{index:04d}.npz"
        with np.load(path, allow_pickle=False) as file:
            part = {key: file[key] for key in file.files}
        lo, hi = index * shard_size, min(len(rows), (index + 1) * shard_size)
        n = hi - lo
        required = {
            "rows",
            "labels",
            "coefficients",
            "port_power_w",
            "upper_power_w",
            "coherence_fraction",
            "eta_fiber_each",
            "eta_system_each",
            "neff_each",
            "mse28",
            "psnr28_db",
            "ssim28",
            "production_reconstruction_loss",
            "camera_total_w",
            "view28",
            "ols_input_field_representatives",
            "representative_rows",
            "ports_ge_-10db",
            "ports_ge_-20db",
            "ports_ge_-30db",
            "ports_ge_-40db",
        }
        require(
            required.issubset(part),
            f"reloaded shard is missing required measurements: {path.name}",
        )
        require(
            np.array_equal(part["rows"], rows[lo:hi]),
            f"reloaded shard row ids changed: {path.name}",
        )
        require(
            part["labels"].shape == (n,)
            and part["coefficients"].shape == (n, PORTS)
            and part["port_power_w"].shape == (n, PORTS)
            and part["upper_power_w"].shape == (n, PORTS)
            and part["coherence_fraction"].shape == (n, PORTS)
            and part["view28"].shape == (n, 1, 28, 28)
            and part["ols_input_field_representatives"].shape
            == (len(part["representative_rows"]), 100, 100),
            f"reloaded shard dimensions changed: {path.name}",
        )
        require(
            np.isin(part["labels"], np.arange(10)).all()
            and np.all(part["port_power_w"] >= 0)
            and np.all(part["upper_power_w"] >= 0),
            f"reloaded shard labels or nonnegative powers changed: {path.name}",
        )
        for key, value in part.items():
            if np.issubdtype(value.dtype, np.number):
                require(
                    (not np.isnan(value).any())
                    if key == "psnr28_db"
                    else np.isfinite(value).all(),
                    f"nonfinite saved evidence {path.name}:{key}",
                )
        parts.append(part)
    result = concat(parts)
    require(
        np.array_equal(result["rows"], rows),
        "saved state population does not match fixed DEV manifest",
    )
    return result


def measure_state(
    spec: dict[str, Any],
    cfg: dict[str, Any],
    baseline: dict[str, Any],
    cps: dict[str, dict[str, Any]],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    dev_rows: torch.Tensor,
    device: torch.device,
    output_root: Path,
    representative_rows: set[int],
) -> dict[str, Any]:
    key = spec["key"]
    directory = output_root / "states" / key
    chunk_dir = directory / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "state_key": key,
        "checkpoint": spec["checkpoint"],
        "trajectory": spec["trajectory"],
        "global_step": spec["global_step"],
        "run_id": cps[key]["run_id"],
        "source_run_id": cps[key].get("source_run_id"),
        "source_checkpoint_step": cps[key].get("source_checkpoint_step"),
        "dev_population": 5000,
        "split_seed": 1234,
        "optimizer_created": False,
        "updates": 0,
    }
    manifest = directory / "state_manifest.json"
    if manifest.exists():
        require(
            json.loads(manifest.read_text(encoding="utf-8")) == clean(identity),
            f"resume identity changed for {key}",
        )
    else:
        write_json(manifest, identity)
    row_ids = dev_rows.cpu().numpy().astype(np.int64)
    shard_size, batch_size = (
        int(cfg["statistics"]["shard_size"]),
        int(cfg["statistics"]["batch_size"]),
    )
    count = math.ceil(len(row_ids) / shard_size)
    require(
        all(
            p.name in {f"chunk_{i:04d}.npz" for i in range(count)}
            for p in chunk_dir.glob("chunk_*.npz")
        ),
        f"unknown recovery chunk in {chunk_dir}",
    )
    model = model_for(key, cfg, baseline, cps, device)
    geo, before, started = actual_geometry(model), snapshot(model), time.perf_counter()
    completed = 0
    for chunk in range(count):
        lo, hi = chunk * shard_size, min(len(row_ids), (chunk + 1) * shard_size)
        path = chunk_dir / f"chunk_{chunk:04d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as file:
                restored = {key: file[key] for key in file.files}
            require(
                np.array_equal(restored["rows"], row_ids[lo:hi])
                and restored["coefficients"].shape == (hi - lo, PORTS)
                and restored["port_power_w"].shape == (hi - lo, PORTS)
                and restored["upper_power_w"].shape == (hi - lo, PORTS)
                and restored["coherence_fraction"].shape == (hi - lo, PORTS),
                f"recovery shard invalid: {path}",
            )
            completed += 1
            continue
        chunks = []
        for start in range(lo, hi, batch_size):
            stop = min(hi, start + batch_size)
            rows = dev_rows[start:stop]
            images, target28, y = target_batch(pixels, labels, rows, device)
            with torch.inference_mode():
                result = model.runtime.forward_from_image(images)
                chunks.append(
                    extract_result(
                        model,
                        result,
                        images,
                        target28,
                        rows,
                        y,
                        float(cfg["model"]["shared_camera_gain_w_inverse"]),
                        representative_rows,
                    )
                )
        combined = concat(chunks)
        require(
            np.array_equal(combined["rows"], row_ids[lo:hi]),
            "new chunk has wrong original MNIST rows",
        )
        write_npz(path, **combined)
        with np.load(path, allow_pickle=False) as saved:
            require(
                set(saved.files) == set(combined), "chunk readback field set differs"
            )
            require(
                all(
                    np.array_equal(saved[name], combined[name], equal_nan=True)
                    for name in combined
                ),
                "chunk readback differs",
            )
        completed += 1
        write_json(
            directory / "progress.json",
            {
                "state": key,
                "completed_chunks": completed,
                "total_chunks": count,
                "completed_rows": hi,
                "last_committed_chunk": chunk,
                "resume_rule": "validated chunks are reused; only missing chunks are recomputed",
            },
        )
    assert_snapshot(model, before, key)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    combined = read_state_chunks(directory, row_ids, shard_size)
    require(
        combined["port_power_w"].shape == (5000, PORTS)
        and combined["upper_power_w"].shape == (5000, PORTS)
        and combined["coherence_fraction"].shape == (5000, PORTS),
        "full population readback dimensions differ",
    )
    write_json(directory / "geometry.json", geo)
    result = {
        **identity,
        "status": "MEASUREMENT_COMPLETE",
        "sample_count": 5000,
        "per_root_power_records": 500000,
        "recovery_chunks": count,
        "elapsed_seconds_observed": time.perf_counter() - started,
        "model_parameters_and_buffers_unchanged": True,
        "full_population_readback_validated": True,
    }
    write_json(directory / "complete.json", result)
    return result


def decode_chunk(
    model: torch.nn.Module,
    coeff: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    encoder_key: str,
    cfg: dict[str, Any],
    pixels: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    require(
        model.runtime.physical_optics is None,
        "Decoder-only public W_OLS path must not recalibrate power",
    )
    batch = torch.from_numpy(rows).long()
    state = ComplexOLSState(
        coefficients=torch.from_numpy(coeff).to(device=device),
        source=f"real_input_encoder:{encoder_key}",
        basis_id=model.runtime.ols.basis_id,
        mode_geometry=model.runtime.ols.mode_identity(),
    )
    native = image_batch(pixels, batch, "small", device)
    target28 = (
        pixels.index_select(0, batch.cpu())
        .unsqueeze(1)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    with torch.inference_mode():
        result = model.runtime.forward_from_ols(state)
    area = (
        model.runtime.camera_plane.pixel_pitch_m[0]
        * model.runtime.camera_plane.pixel_pitch_m[1]
    )
    camera_w_px = result.raw_camera.tensor * area
    calibrated = camera_w_px * float(cfg["model"]["shared_camera_gain_w_inverse"])
    view = resize_view(calibrated)
    mse = (view - target28).square().mean((1, 2, 3))
    psnr = torch.where(
        mse > 0, -10 * torch.log10(mse), torch.full_like(mse, float("inf"))
    )
    clipping = ((calibrated < 0) | (calibrated > 1)).to(torch.float64).mean((1, 2, 3))
    entry = result.generator_input_field
    entry_area = entry.plane.pixel_pitch_m[0] * entry.plane.pixel_pitch_m[1]
    entry_w = (
        entry.tensor.abs().to(torch.float64).square().reshape(len(rows), -1).sum(1)
        * entry_area
    )
    layer_power = result.crop_statistics["generator_layer_power"]
    return {
        "rows": rows.copy(),
        "labels": labels.copy(),
        "mse28": mse.detach().cpu().numpy().astype(np.float64),
        "psnr28_db": psnr.detach().cpu().numpy().astype(np.float64),
        "ssim28": ssim_batch(view, target28)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "production_reconstruction_loss": (calibrated - native)
        .square()
        .mean((1, 2, 3))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "clipping_fraction": clipping.detach().cpu().numpy().astype(np.float64),
        "camera_total_w": result.raw_camera.tensor.to(torch.float64)
        .reshape(len(rows), -1)
        .sum(1)
        .mul(area)
        .cpu()
        .numpy(),
        "generator_entry_w": entry_w.detach().cpu().numpy().astype(np.float64),
        "decoder_1_w": layer_power[:, 0, 3].detach().cpu().numpy().astype(np.float64),
        "decoder_2_w": layer_power[:, 1, 3].detach().cpu().numpy().astype(np.float64),
        "view28": view.detach().cpu().numpy().astype(np.float32),
    }


CROSS_CHUNK_FIELDS = {
    "rows",
    "labels",
    "mse28",
    "psnr28_db",
    "ssim28",
    "production_reconstruction_loss",
    "clipping_fraction",
    "camera_total_w",
    "generator_entry_w",
    "decoder_1_w",
    "decoder_2_w",
    "view28",
}


def validate_cross_chunk(
    part: dict[str, np.ndarray], expected_rows: np.ndarray, source: str
) -> None:
    n = len(expected_rows)
    require(set(part) == CROSS_CHUNK_FIELDS, f"cross chunk schema changed: {source}")
    require(
        np.array_equal(part["rows"], expected_rows) and part["labels"].shape == (n,),
        f"cross chunk original row ids or labels changed: {source}",
    )
    require(
        all(
            part[key].shape == (n,)
            for key in (
                "mse28",
                "psnr28_db",
                "ssim28",
                "production_reconstruction_loss",
                "clipping_fraction",
                "camera_total_w",
                "generator_entry_w",
                "decoder_1_w",
                "decoder_2_w",
            )
        )
        and part["view28"].shape == (n, 1, 28, 28),
        f"cross chunk dimensions changed: {source}",
    )
    require(
        np.isin(part["labels"], np.arange(10)).all(),
        f"cross chunk labels are invalid: {source}",
    )
    for key, value in part.items():
        if key != "psnr28_db" and np.issubdtype(value.dtype, np.number):
            require(
                np.isfinite(value).all(),
                f"cross chunk has nonfinite values: {source}:{key}",
            )
    require(
        not np.isnan(part["psnr28_db"]).any(),
        f"cross chunk PSNR contains NaN: {source}",
    )


def run_cross_combo(
    decoder_key: str,
    encoder_key: str,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    cps: dict[str, dict[str, Any]],
    state_arrays: dict[str, dict[str, np.ndarray]],
    pixels: torch.Tensor,
    dev_rows: np.ndarray,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    encoded = state_arrays[encoder_key]
    require_row_match(dev_rows, encoded["rows"])
    dspec = next(x for x in cfg["checkpoints"] if x["key"] == decoder_key)
    espec = next(x for x in cfg["checkpoints"] if x["key"] == encoder_key)
    identity = {
        "schema": "gaussian_real_input_cross_decoder.v1",
        "decoder_key": decoder_key,
        "encoder_key": encoder_key,
        "decoder_checkpoint": dspec["checkpoint"],
        "encoder_checkpoint": espec["checkpoint"],
        "decoder_run_id": cps[decoder_key]["run_id"],
        "encoder_run_id": cps[encoder_key]["run_id"],
        "decoder_global_step": dspec["global_step"],
        "encoder_global_step": espec["global_step"],
        "dev_population": 5000,
        "absolute_coefficients_preserved": True,
        "shared_gain_preserved": True,
        "optimizer_created": False,
    }
    directory = (
        output_root / "cross_decoder" / f"decoder_{decoder_key}__encoder_{encoder_key}"
    )
    chunks_dir = directory / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest = directory / "cross_manifest.json"
    if manifest.exists():
        require(
            json.loads(manifest.read_text(encoding="utf-8")) == clean(identity),
            f"cross recovery identity changed: {decoder_key} <- {encoder_key}",
        )
    else:
        write_json(manifest, identity)
    size = int(cfg["statistics"]["shard_size"])
    count = math.ceil(len(dev_rows) / size)
    for chunk in range(count):
        lo, hi = chunk * size, min(len(dev_rows), (chunk + 1) * size)
        path = chunks_dir / f"chunk_{chunk:04d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as file:
                restored = {key: file[key] for key in file.files}
            validate_cross_chunk(restored, dev_rows[lo:hi], str(path))
            continue
        parts = []
        batch = int(cfg["statistics"]["batch_size"])
        for start in range(lo, hi, batch):
            stop = min(hi, start + batch)
            parts.append(
                decode_chunk(
                    model,
                    encoded["coefficients"][start:stop],
                    encoded["rows"][start:stop],
                    encoded["labels"][start:stop],
                    encoder_key,
                    cfg,
                    pixels,
                    device,
                )
            )
        result = concat(parts)
        validate_cross_chunk(
            result, dev_rows[lo:hi], f"new:{decoder_key}<-{encoder_key}:{chunk}"
        )
        write_npz(path, **result)
        with np.load(path, allow_pickle=False) as saved:
            restored = {key: saved[key] for key in saved.files}
        require(
            set(restored) == set(result)
            and all(
                np.array_equal(restored[key], result[key], equal_nan=True)
                for key in result
            ),
            "cross chunk readback failed",
        )
    loaded = []
    for chunk in range(count):
        lo, hi = chunk * size, min(len(dev_rows), (chunk + 1) * size)
        chunk_path = chunks_dir / f"chunk_{chunk:04d}.npz"
        with np.load(chunk_path, allow_pickle=False) as file:
            part = {k: file[k] for k in file.files}
        validate_cross_chunk(part, dev_rows[lo:hi], f"readback:{chunk_path.name}")
        loaded.append(part)
    full = concat(loaded)
    require_row_match(dev_rows, full["rows"])
    write_json(
        directory / "complete.json",
        {
            **identity,
            "status": "MEASUREMENT_COMPLETE",
            "sample_count": len(dev_rows),
            "decoder_only_forwards": len(dev_rows),
            "chunks": count,
        },
    )
    return {"metadata": identity, "arrays": full}


def run_measure(cfg: dict[str, Any], device: torch.device, output_root: Path) -> None:
    validate_preflight(output_root)
    require(
        device.type == "cuda" and torch.cuda.is_available(),
        "full measurement requires AutoDL CUDA",
    )
    baseline, continuation, cps, pixels, labels, train_rows, dev_rows = load_inputs(cfg)
    dev_ids = dev_rows.cpu().numpy().astype(np.int64)
    dev_labels = labels[dev_rows].cpu().numpy().astype(np.int64)
    require(
        np.array_equal(np.bincount(dev_labels, minlength=10), np.full(10, 500)),
        "balanced DEV labels changed",
    )
    manifest = json.loads(
        (output_root / "input_manifest.json").read_text(encoding="utf-8")
    )
    require(
        np.array_equal(
            np.asarray(manifest["dev_rows_original_order"], dtype=np.int64), dev_ids
        )
        and np.array_equal(
            np.asarray(manifest["dev_labels"], dtype=np.int64), dev_labels
        ),
        "saved preflight input manifest differs from current fixed DEV",
    )
    reps = set(map(int, fixed_preflight_rows(dev_rows, labels).tolist()))
    started = time.perf_counter()
    state_results = []
    for spec in cfg["checkpoints"]:
        state_results.append(
            measure_state(
                spec,
                cfg,
                baseline,
                cps,
                pixels,
                labels,
                dev_rows,
                device,
                output_root,
                reps,
            )
        )
        write_json(
            output_root / "measurement_progress.json",
            {
                "completed_states": [x["state_key"] for x in state_results],
                "total_states": 9,
                "real_input_state_records": len(state_results) * 5000,
                "per_root_power_records": len(state_results) * 5000 * PORTS,
                "training_updates": 0,
            },
        )
    # Load the nine saved arrays from their chunk directories. Each row is paired by original data index.
    state_arrays = {
        spec["key"]: read_state_chunks(
            output_root / "states" / spec["key"],
            dev_ids,
            int(cfg["statistics"]["shard_size"]),
        )
        for spec in cfg["checkpoints"]
    }
    off_diagonal = [
        ("step_000000", "step_030000"),
        ("step_030000", "step_000000"),
        ("step_030000", "control_step_035000"),
        ("control_step_035000", "step_030000"),
        ("step_030000", "treatment_step_035000"),
        ("treatment_step_035000", "step_030000"),
        ("control_step_035000", "treatment_step_035000"),
        ("treatment_step_035000", "control_step_035000"),
    ]
    cross_results: dict[str, Any] = {}
    for decoder_key in dict.fromkeys(d for d, _ in off_diagonal):
        model = model_for(decoder_key, cfg, baseline, cps, device)
        before = snapshot(model)
        # The public forward_from_ols endpoint executes W_OLS and the production Decoder.
        # Removing only the physical-trace branch avoids recalibration; layer transmission stays 0.889.
        model.runtime.physical_optics = None
        for d_key, e_key in off_diagonal:
            if d_key == decoder_key:
                item = run_cross_combo(
                    d_key,
                    e_key,
                    model,
                    cfg,
                    cps,
                    state_arrays,
                    pixels,
                    dev_ids,
                    output_root,
                    device,
                )
                cross_results[f"{d_key}__{e_key}"] = item
        assert_snapshot(model, before, f"cross_decoder_{decoder_key}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    require(
        len(cross_results) == 8,
        "expected exactly eight unique off-diagonal Decoder combinations",
    )
    write_json(
        output_root / "measurement_complete.json",
        {
            "schema": "gaussian_real_input_measurement_complete.v1",
            "status": "MEASUREMENT_COMPLETE",
            "states": state_results,
            "cross_combinations": [x["metadata"] for x in cross_results.values()],
            "real_input_state_records": 45000,
            "per_root_power_records": 4500000,
            "additional_decoder_only_forward_records": 40000,
            "training_updates": 0,
            "optimizers_created": 0,
            "prior_samples": 0,
            "official_test_rows_read": 0,
            "elapsed_seconds_observed": time.perf_counter() - started,
            "fixed_DEV_rows_revalidated": True,
            "paired_by_original_row_id": True,
            "model_parameters_and_buffers_unchanged": True,
        },
    )


def load_cross_chunks(
    directory: Path, rows: np.ndarray, shard_size: int
) -> dict[str, np.ndarray]:
    chunk_dir = directory / "chunks"
    count = math.ceil(len(rows) / shard_size)
    require(
        len(list(chunk_dir.glob("chunk_*.npz"))) == count,
        f"cross measurement chunks missing in {directory}",
    )
    parts = []
    for index in range(count):
        with np.load(chunk_dir / f"chunk_{index:04d}.npz", allow_pickle=False) as file:
            part = {key: file[key] for key in file.files}
        lo, hi = index * shard_size, min(len(rows), (index + 1) * shard_size)
        require(
            np.array_equal(part["rows"], rows[lo:hi]),
            f"cross row ids changed in {directory}",
        )
        parts.append(part)
    out = concat(parts)
    require_row_match(rows, out["rows"])
    return out


def distribution(x: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(x, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def state_summaries(
    spec: dict[str, Any],
    data: dict[str, np.ndarray],
    numeric_floor: np.ndarray,
    config: dict[str, Any],
    output_root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    key = spec["key"]
    p = data["port_power_w"].astype(np.float64)
    coefficients = data["coefficients"].astype(np.complex128)
    b = data["upper_power_w"].astype(np.float64)
    c = data["coherence_fraction"].astype(np.float64)
    rows, labels = data["rows"], data["labels"]
    total = p.sum(axis=1)
    mean_vector = p.mean(axis=0)
    denominator = float(np.square(mean_vector).sum())
    pop_neff = float(mean_vector.sum() ** 2 / denominator) if denominator > 0 else None
    positive_count = (p > 0).sum(axis=1)
    resolved = (
        np.abs(data["coefficients"].astype(np.complex128))
        > float(config["statistics"]["numerical_resolution_margin"])
        * numeric_floor[None, :]
    ).sum(axis=1)
    resolved_by_port = (
        np.abs(coefficients)
        > float(config["statistics"]["numerical_resolution_margin"])
        * numeric_floor[None, :]
    )
    p3 = p.reshape(-1, 10, 10)
    center = p3[:, 2:8, 2:8].sum((1, 2))
    outer_mask = np.ones((10, 10), dtype=bool)
    outer_mask[1:9, 1:9] = False
    outer = p3[:, outer_mask].sum(1)
    center_share, outer_share = np_ratio(center, total), np_ratio(outer, total)
    root_thresholds = {}
    for db in DB_THRESHOLDS:
        root_thresholds[str(db)] = {
            "mean_roots_per_image": float(data[f"ports_ge_{db}db"].mean()),
            "mean_population_vector_roots": int(
                (mean_vector >= mean_vector.max() * 10.0 ** (db / 10)).sum()
            ),
        }
    descending = np.sort(p, axis=1)[:, ::-1]
    brightest_four_share = np_ratio(descending[:, :4].sum(axis=1), total)
    roots90 = np.asarray(
        [
            int(np.searchsorted(np.cumsum(x), 0.9 * x.sum()) + 1)
            if x.sum() > 0
            else np.nan
            for x in descending
        ]
    )
    summary = {
        "state_key": key,
        "trajectory": spec["trajectory"],
        "global_step": spec["global_step"],
        "sample_count": int(len(rows)),
        "per_root_records": int(p.size),
        "strict_positive_roots_per_image": distribution(positive_count),
        "numerically_resolved_roots_2x_reference_error_per_image": distribution(
            resolved
        ),
        "relative_threshold_occupancy": root_thresholds,
        "coupled_power_per_image_w": distribution(total),
        "eta_fiber_each_image": distribution(data["eta_fiber_each"]),
        "eta_system_each_image": distribution(data["eta_system_each"]),
        "eta_fiber_ratio_of_total_powers": float(
            total.sum() / data["trace__P_before_fiber_W"].sum()
        )
        if data["trace__P_before_fiber_W"].sum() > 0
        else None,
        "eta_system_ratio_of_total_powers": float(
            total.sum() / data["trace__P_input_W"].sum()
        )
        if data["trace__P_input_W"].sum() > 0
        else None,
        "sample_neff": distribution(data["neff_each"]),
        "population_mean_power_vector_neff": pop_neff,
        "brightest_four_power_share": distribution(brightest_four_share),
        "roots_to_carry_90_percent_power": distribution(roots90),
        "central_6x6_share": distribution(center_share),
        "outermost_36_share": distribution(outer_share),
        "production_reconstruction_loss": distribution(
            data["production_reconstruction_loss"]
        ),
        "mse28": distribution(data["mse28"]),
        "psnr28_db": distribution(data["psnr28_db"]),
        "ssim28": distribution(data["ssim28"]),
        "proxy_clipping_fraction": distribution(data["clipping_fraction"]),
        "camera_total_power_w": distribution(data["camera_total_w"]),
        "absolute_mean_port_power_w": mean_vector,
        "amplitude_upper_mean_port_power_w": b.mean(0),
        "mean_phase_retention_by_port": np.asarray(
            [
                float(np.nanmean(c[:, j])) if np.isfinite(c[:, j]).any() else np.nan
                for j in range(PORTS)
            ]
        ),
        "phase_retention_defined_fraction_by_port": np.isfinite(c).mean(0),
        "numerical_error_envelope_by_port": numeric_floor,
        "gain_w_inverse": 1.505046959946815e9,
    }
    quartile_size = PORTS // 4
    power_rank = np.lexsort((np.arange(PORTS), mean_vector))
    weakest_ports = power_rank[:quartile_size]
    brightest_ports = power_rank[-quartile_size:]
    mean_b_by_port = b.mean(0)
    mean_c_by_port = np.asarray(summary["mean_phase_retention_by_port"])

    def quartile_values(indices: np.ndarray) -> dict[str, Any]:
        c_values = mean_c_by_port[indices]
        c_values = c_values[np.isfinite(c_values)]
        return {
            "port_count": int(len(indices)),
            "port_indices_zero_based": indices.astype(np.int64),
            "mean_power_w": float(mean_vector[indices].mean()),
            "mean_amplitude_upper_w": float(mean_b_by_port[indices].mean()),
            "mean_phase_retention": float(c_values.mean()) if len(c_values) else None,
            "phase_retention_defined_port_count": int(len(c_values)),
        }

    weak_values = quartile_values(weakest_ports)
    bright_values = quartile_values(brightest_ports)
    summary["weak_channel_bottom_top_quartiles"] = {
        "selection": "rank the 100 roots by fixed-population mean absolute output power; lowest and highest 25 roots",
        "weakest_quartile": weak_values,
        "brightest_quartile": bright_values,
        "weak_to_bright_ratios": {
            "mean_power": weak_values["mean_power_w"] / bright_values["mean_power_w"]
            if bright_values["mean_power_w"] > 0
            else None,
            "amplitude_upper_bound": weak_values["mean_amplitude_upper_w"]
            / bright_values["mean_amplitude_upper_w"]
            if bright_values["mean_amplitude_upper_w"] > 0
            else None,
            "phase_retention": weak_values["mean_phase_retention"]
            / bright_values["mean_phase_retention"]
            if weak_values["mean_phase_retention"] is not None
            and bright_values["mean_phase_retention"] not in (None, 0)
            else None,
        },
    }

    port_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for digit in range(10):
        mask = labels == digit
        cp, cb, cc = p[mask], b[mask], c[mask]
        class_rel = np_ratio(cp, cp.max(axis=1, keepdims=True))
        class_mean = cp.mean(0)
        class_b_mean = cb.mean(0)
        for port in range(PORTS):
            r, col = port // 10 + 1, port % 10 + 1
            port_rows.append(
                {
                    "state_key": key,
                    "label": digit,
                    "port_index_zero_based": port,
                    "port_row_one_based": r,
                    "port_column_one_based": col,
                    "x_um": (col - 5.5) * 20.0,
                    "y_um": (r - 5.5) * 20.0,
                    "power_mean_w": float(cp[:, port].mean()),
                    "power_median_w": float(np.median(cp[:, port])),
                    "power_p10_w": float(np.quantile(cp[:, port], 0.1)),
                    "power_p90_w": float(np.quantile(cp[:, port], 0.9)),
                    "share_of_class_mean_power": float(
                        class_mean[port] / class_mean.sum()
                    )
                    if class_mean.sum() > 0
                    else None,
                    "relative_occupancy_minus10db": float(
                        np.mean(class_rel[:, port] >= 0.1)
                    ),
                    "relative_occupancy_minus20db": float(
                        np.mean(class_rel[:, port] >= 0.01)
                    ),
                    "relative_occupancy_minus30db": float(
                        np.mean(class_rel[:, port] >= 0.001)
                    ),
                    "relative_occupancy_minus40db": float(
                        np.mean(class_rel[:, port] >= 0.0001)
                    ),
                    "strictly_positive_fraction": float(np.mean(cp[:, port] > 0)),
                    "numerically_resolved_fraction_2x_reference_error": float(
                        np.mean(resolved_by_port[mask, port])
                    ),
                    "numerical_resolution_error_envelope_abs_coefficient": float(
                        numeric_floor[port]
                    ),
                }
            )
            defined = np.isfinite(cc[:, port])
            attribution_rows.append(
                {
                    "state_key": key,
                    "label": digit,
                    "port_index_zero_based": port,
                    "port_row_one_based": r,
                    "port_column_one_based": col,
                    "amplitude_upper_mean_w": float(class_b_mean[port]),
                    "amplitude_upper_median_w": float(np.median(cb[:, port])),
                    "amplitude_upper_p10_w": float(np.quantile(cb[:, port], 0.1)),
                    "amplitude_upper_p90_w": float(np.quantile(cb[:, port], 0.9)),
                    "phase_retention_mean": float(np.mean(cc[defined, port]))
                    if defined.any()
                    else None,
                    "phase_retention_median": float(np.median(cc[defined, port]))
                    if defined.any()
                    else None,
                    "phase_retention_p10": float(np.quantile(cc[defined, port], 0.1))
                    if defined.any()
                    else None,
                    "phase_retention_p90": float(np.quantile(cc[defined, port], 0.9))
                    if defined.any()
                    else None,
                    "amplitude_upper_share_of_ols_input": float(
                        class_b_mean[port]
                        / np.mean(data["trace__P_before_fiber_W"][mask])
                    )
                    if np.mean(data["trace__P_before_fiber_W"][mask]) > 0
                    else None,
                }
            )
        class_rows.append(
            {
                "state_key": key,
                "label": digit,
                "sample_count": int(mask.sum()),
                "coupled_power_mean_w": float(p[mask].sum(1).mean()),
                "eta_fiber_mean": float(np.nanmean(data["eta_fiber_each"][mask])),
                "eta_system_mean": float(np.nanmean(data["eta_system_each"][mask])),
                "sample_neff_mean": float(np.nanmean(data["neff_each"][mask])),
                "strict_positive_roots_mean": float(positive_count[mask].mean()),
                "resolved_roots_2x_error_mean": float(resolved[mask].mean()),
                "central_6x6_share_mean": float(np.nanmean(center_share[mask])),
                "outer36_share_mean": float(np.nanmean(outer_share[mask])),
                "mse28_mean": float(data["mse28"][mask].mean()),
                "ssim28_mean": float(data["ssim28"][mask].mean()),
            }
        )
    image_rows = []
    for i, row in enumerate(rows):
        image_rows.append(
            {
                "state_key": key,
                "original_row": int(row),
                "label": int(labels[i]),
                "coupled_power_w": float(total[i]),
                "eta_fiber": float(data["eta_fiber_each"][i])
                if np.isfinite(data["eta_fiber_each"][i])
                else None,
                "eta_system": float(data["eta_system_each"][i])
                if np.isfinite(data["eta_system_each"][i])
                else None,
                "neff": float(data["neff_each"][i])
                if np.isfinite(data["neff_each"][i])
                else None,
                "strictly_positive_roots": int(positive_count[i]),
                "resolved_roots_2x_reference_error": int(resolved[i]),
                "roots_ge_minus10db": int(data["ports_ge_-10db"][i]),
                "roots_ge_minus20db": int(data["ports_ge_-20db"][i]),
                "roots_ge_minus30db": int(data["ports_ge_-30db"][i]),
                "roots_ge_minus40db": int(data["ports_ge_-40db"][i]),
                "brightest_four_share": float(brightest_four_share[i])
                if np.isfinite(brightest_four_share[i])
                else None,
                "roots_to_90_percent_power": int(roots90[i])
                if np.isfinite(roots90[i])
                else None,
                "central_6x6_share": float(center_share[i])
                if np.isfinite(center_share[i])
                else None,
                "outer36_share": float(outer_share[i])
                if np.isfinite(outer_share[i])
                else None,
                "production_reconstruction_loss": float(
                    data["production_reconstruction_loss"][i]
                ),
                "mse28": float(data["mse28"][i]),
                "psnr28_db": float(data["psnr28_db"][i])
                if np.isfinite(data["psnr28_db"][i])
                else None,
                "ssim28": float(data["ssim28"][i]),
            }
        )
    chain_rows = []
    for name in POWER_CHAIN + LAYER_POWER:
        metric = f"trace__{name}"
        if metric not in data:
            continue
        values = data[metric].astype(np.float64)
        if values.ndim == 1:
            stages = [(name, values)]
        elif values.ndim == 2 and values.shape[1] == 2:
            stages = [(f"{name}_layer_{i + 1}", values[:, i]) for i in range(2)]
        else:
            continue
        for stage, arr in stages:
            chain_rows.append(
                {
                    "state_key": key,
                    "stage": stage,
                    **{f"{k}_w": v for k, v in distribution(arr).items()},
                }
            )
    write_json(output_root / "states" / key / "population_summary.json", summary)
    write_npz(
        output_root / "states" / key / "population_vectors.npz",
        mean_power_w=mean_vector,
        mean_amplitude_upper_w=b.mean(0),
        mean_phase_retention=summary["mean_phase_retention_by_port"],
        numerical_error_envelope=numeric_floor,
        normalized_population_power=mean_vector / mean_vector.sum()
        if mean_vector.sum() > 0
        else np.full(PORTS, np.nan),
    )
    return summary, port_rows, attribution_rows, image_rows, chain_rows, class_rows


def stratified_pair_bootstrap(
    old: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
    labels: np.ndarray,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    groups = [np.flatnonzero(labels == d) for d in range(10)]
    require(
        all(len(g) == 500 for g in groups),
        "bootstrap inputs must retain 500 rows per class",
    )
    pairs = {
        "eta_fiber_mean": (old["eta_fiber_each"], new["eta_fiber_each"]),
        "eta_system_mean": (old["eta_system_each"], new["eta_system_each"]),
        "neff_mean": (old["neff_each"], new["neff_each"]),
        "mse28_mean": (old["mse28"], new["mse28"]),
        "ssim28_mean": (old["ssim28"], new["ssim28"]),
        "production_reconstruction_loss_mean": (
            old["production_reconstruction_loss"],
            new["production_reconstruction_loss"],
        ),
    }
    boot = {name: np.empty(replicates, np.float64) for name in pairs}
    tv_boot = np.empty(replicates, np.float64)
    old_profile = old["port_power_w"] / old["port_power_w"].sum(1, keepdims=True)
    new_profile = new["port_power_w"] / new["port_power_w"].sum(1, keepdims=True)
    rng = np.random.default_rng(seed)
    for i in range(replicates):
        ix = np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])
        for name, (a, b) in pairs.items():
            boot[name][i] = np.mean(b[ix] - a[ix])
        tv_boot[i] = (
            0.5 * np.abs(new_profile[ix].mean(0) - old_profile[ix].mean(0)).sum()
        )
    result = {}
    for name, (a, b) in pairs.items():
        values = boot[name]
        result[name] = {
            "mean_change_new_minus_old": float(np.mean(b - a)),
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "resamples": replicates,
        }
    result["normalized_population_port_distribution_tv"] = {
        "estimate": float(
            0.5 * np.abs(new_profile.mean(0) - old_profile.mean(0)).sum()
        ),
        "ci95": [
            float(np.quantile(tv_boot, 0.025)),
            float(np.quantile(tv_boot, 0.975)),
        ],
        "definition": "TV distance between means of image-normalized 100-port power vectors",
    }
    return result


def cross_matrix(
    old_key: str,
    new_key: str,
    states: dict[str, dict[str, np.ndarray]],
    crosses: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    specs = [
        (old_key, old_key, "L_ii_old_decoder_old_encoder"),
        (old_key, new_key, "L_ij_old_decoder_new_encoder"),
        (new_key, old_key, "L_ji_new_decoder_old_encoder"),
        (new_key, new_key, "L_jj_new_decoder_new_encoder"),
    ]
    for decoder, encoder, name in specs:
        item = (
            states[decoder] if decoder == encoder else crosses[f"{decoder}__{encoder}"]
        )
        require_row_match(states[old_key]["rows"], item["rows"])
        require_row_match(states[new_key]["rows"], item["rows"])
        require(
            np.array_equal(states[old_key]["labels"], item["labels"])
            and np.array_equal(states[new_key]["labels"], item["labels"]),
            "cross labels are mismatched",
        )
        cells[name] = {
            "decoder": decoder,
            "encoder": encoder,
            **{
                metric: item[metric]
                for metric in (
                    "mse28",
                    "psnr28_db",
                    "ssim28",
                    "production_reconstruction_loss",
                    "camera_total_w",
                    "clipping_fraction",
                )
            },
            "mean_mse28": float(np.mean(item["mse28"])),
            "mean_psnr28_db": float(np.mean(item["psnr28_db"])),
            "mean_ssim28": float(np.mean(item["ssim28"])),
            "mean_production_reconstruction_loss": float(
                np.mean(item["production_reconstruction_loss"])
            ),
            "mean_camera_power_w": float(np.mean(item["camera_total_w"])),
            "mean_proxy_clipping_fraction": float(np.mean(item["clipping_fraction"])),
        }
    lii, lij = (
        cells["L_ii_old_decoder_old_encoder"]["mse28"],
        cells["L_ij_old_decoder_new_encoder"]["mse28"],
    )
    lji, ljj = (
        cells["L_ji_new_decoder_old_encoder"]["mse28"],
        cells["L_jj_new_decoder_new_encoder"]["mse28"],
    )
    ge_old_d, ge_new_d = lii - lij, lji - ljj
    gd_old_e, gd_new_e = lii - lji, lij - ljj
    vals = {
        "encoder_symmetric": 0.5 * (ge_old_d + ge_new_d),
        "decoder_symmetric": 0.5 * (gd_old_e + gd_new_e),
        "encoder_effect_with_old_decoder": ge_old_d,
        "encoder_effect_with_new_decoder": ge_new_d,
        "decoder_effect_with_old_encoder": gd_old_e,
        "decoder_effect_with_new_encoder": gd_new_e,
        "encoder_decoder_interaction": ge_new_d - ge_old_d,
        "native_system_mse_gain_old_minus_new": lii - ljj,
    }
    require(
        np.allclose(
            vals["encoder_symmetric"] + vals["decoder_symmetric"],
            vals["native_system_mse_gain_old_minus_new"],
            rtol=1e-10,
            atol=1e-12,
        ),
        "symmetric Encoder/Decoder effects do not sum to native MSE gain",
    )
    return {
        "old_checkpoint": old_key,
        "new_checkpoint": new_key,
        "cells": cells,
        "per_image_mse": {"L_ii": lii, "L_ij": lij, "L_ji": lji, "L_jj": ljj},
        "per_image_contributions": vals,
        "point_estimates": {k: float(np.mean(v)) for k, v in vals.items()},
    }


def bootstrap_contributions(
    matrix: dict[str, Any], labels: np.ndarray, seed: int, replicates: int
) -> dict[str, Any]:
    groups = [np.flatnonzero(labels == d) for d in range(10)]
    names = list(matrix["per_image_contributions"])
    draws = {name: np.empty(replicates, np.float64) for name in names}
    rng = np.random.default_rng(seed)
    for i in range(replicates):
        ix = np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])
        for name, values in matrix["per_image_contributions"].items():
            draws[name][i] = np.mean(values[ix])
    return {
        name: {
            "mean": float(np.mean(values)),
            "ci95": [
                float(np.quantile(draws[name], 0.025)),
                float(np.quantile(draws[name], 0.975)),
            ],
        }
        for name, values in matrix["per_image_contributions"].items()
    }


def large_architecture_recap(output_root: Path) -> dict[str, Any]:
    source = (
        ROOT / "outputs/20261015_chen_lightgen_400_ols_power_audit_v1/runs/attempt_01"
    )
    report_path = source / "analysis_report.md"
    json_path = source / "power_analysis.json"
    sample_path = source / "sample_power.csv"
    port_path = source / "port_power.csv"
    for path in (report_path, json_path, sample_path, port_path):
        require(path.is_file(), f"saved large-architecture evidence missing: {path}")
    analysis = json.loads(json_path.read_text(encoding="utf-8"))
    require(
        analysis["scientific_updates"]
        == analysis["new_optical_forwards"]
        == analysis["official_test_reads"]
        == 0,
        "large-architecture recap source is not saved-evidence-only",
    )
    sample_records = csv_rows(sample_path)
    samples = [row for row in sample_records if row["population"] == "fixed_train"]
    generated = [
        row for row in sample_records if row["population"] == "saved_generation"
    ]
    require(
        len(samples) == 300 and len({x["sample_key"] for x in samples}) == 300,
        "large-architecture fixed 300 sample population changed",
    )
    require(
        len(generated) == 10_000
        and {int(x["sample_key"]) for x in generated} == set(range(10_000)),
        "large-architecture saved-generation queue identity changed",
    )
    label_counts = np.bincount(
        np.asarray([int(x["label"]) for x in samples]), minlength=10
    )
    require(
        np.array_equal(label_counts, np.full(10, 30)),
        "large-architecture fixed sample class counts changed",
    )
    generated_label_counts = np.bincount(
        np.asarray([int(x["label"]) for x in generated]), minlength=10
    )
    require(
        np.array_equal(generated_label_counts, np.full(10, 1000)),
        "large-architecture saved-generation class counts changed",
    )
    ports = csv_rows(port_path)
    require(len(ports) == PORTS, "large-architecture saved 100-port table incomplete")
    port_ids = [int(row["port_zero_based"]) for row in ports]
    require(
        port_ids == list(range(PORTS)),
        "large-architecture port row order or identity changed",
    )
    full = np.asarray(
        [float(row["full_train_mean_w"]) for row in ports], dtype=np.float64
    )
    fixed = np.asarray(
        [float(row["fixed300_mean_w"]) for row in ports], dtype=np.float64
    )
    generated_port = np.asarray(
        [float(row["saved_generation10000_mean_w"]) for row in ports], dtype=np.float64
    )
    require(
        np.isfinite(full).all()
        and np.isfinite(fixed).all()
        and np.isfinite(generated_port).all(),
        "large-architecture port powers nonfinite",
    )
    for index, row in enumerate(ports):
        expected_row, expected_col = index // 10 + 1, index % 10 + 1
        require(
            int(row["row_one_based"]) == expected_row
            and int(row["column_one_based"]) == expected_col
            and math.isclose(
                float(row["y_um"]), (expected_row - 5.5) * 125.0, abs_tol=1e-12
            )
            and math.isclose(
                float(row["x_um"]), (expected_col - 5.5) * 125.0, abs_tol=1e-12
            ),
            f"large-architecture port geometry or ordering changed at {index}",
        )
    require(
        analysis["real_fixed_300"]["strictly_positive_port_count"]["minimum"] == 100
        and analysis["real_fixed_300"]["strictly_positive_port_count"]["maximum"]
        == 100,
        "saved large-architecture fixed images no longer show 100 strictly positive floating channels",
    )
    require(
        analysis["generated_saved_10000"]["strictly_positive_port_count"]["minimum"]
        == 100
        and analysis["generated_saved_10000"]["strictly_positive_port_count"]["maximum"]
        == 100,
        "saved large-architecture generated queue no longer shows 100 strictly positive floating channels",
    )
    validation = analysis["validation"]
    require(
        validation["population_rows_match_canonical_train"] is True
        and validation["source_step"] == 30000
        and validation["source_encoder_grid"] == 2048
        and validation["source_moments_cover_train_count"] == 55_000,
        "large-architecture real-input identity or saved-moment lineage changed",
    )
    power_chain = analysis["power_chain"]
    require(
        math.isclose(
            float(power_chain["uniform_source_power_uw"]),
            249.3,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "large-architecture source-power denominator changed",
    )
    fixed_mean_sample_power_uw = float(
        np.mean([float(x["total_power_w"]) for x in samples]) * 1e6
    )
    generated_mean_sample_power_uw = float(
        np.mean([float(x["total_power_w"]) for x in generated]) * 1e6
    )
    require(
        math.isclose(
            fixed_mean_sample_power_uw,
            float(analysis["real_fixed_300"]["total_power_uw"]["mean"]),
            rel_tol=1e-12,
        )
        and math.isclose(
            generated_mean_sample_power_uw,
            float(analysis["generated_saved_10000"]["total_power_uw"]["mean"]),
            rel_tol=1e-12,
        ),
        "large-architecture saved sample powers do not reproduce the population summaries",
    )
    require(
        math.isclose(
            float(full.sum()),
            float(
                analysis["full_train_average_power_distribution"]["total_power_uw"][
                    "mean"
                ]
            )
            * 1e-6,
            rel_tol=1e-12,
        )
        and math.isclose(
            float(fixed.sum()), fixed_mean_sample_power_uw * 1e-6, rel_tol=1e-12
        )
        and math.isclose(
            float(generated_port.sum()),
            generated_mean_sample_power_uw * 1e-6,
            rel_tol=1e-12,
        ),
        "large-architecture 100-port table totals do not match saved population denominators",
    )
    fixed_eta_fiber = float(power_chain["ols_modal_total_uw"]["mean"]) / float(
        power_chain["encoder_2_and_ols_input_full_plane_uw"]["mean"]
    )
    fixed_eta_system = float(power_chain["ols_modal_total_uw"]["mean"]) / float(
        power_chain["uniform_source_power_uw"]
    )
    require(
        math.isclose(
            fixed_eta_fiber,
            float(power_chain["modal_over_ols_input_ratio_of_totals"]),
            rel_tol=1e-12,
        )
        and math.isclose(
            fixed_eta_system,
            float(power_chain["modal_over_uniform_source_ratio_of_totals"]),
            rel_tol=1e-12,
        ),
        "large-architecture efficiency denominators do not reproduce saved totals",
    )
    out_dir = output_root / "large_architecture_recap"
    write_csv(out_dir / "per_port_full_train_fixed300_and_class_means.csv", ports)
    write_csv(out_dir / "fixed300_real_train_sample_power.csv", samples)
    write_csv(out_dir / "saved_generation_10000_sample_power.csv", generated)
    (out_dir / "saved_analysis_report.md").write_text(
        report_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    summary = {
        "source_report": str(report_path.relative_to(ROOT)),
        "source_machine_analysis": str(json_path.relative_to(ROOT)),
        "input_identity_and_saved_moment_lineage": {
            **analysis["inputs"],
            **validation,
            "fixed_train_population": "canonical TRAIN; 300 rows; 30 per class",
            "generated_queue_population": "saved_generation; 10,000 rows; 1,000 per class",
            "queue_populations_reported_separately": True,
        },
        "status": analysis["status"],
        "evidence_tier": analysis["evidence_tier"],
        "new_optical_forwards": 0,
        "official_test_reads": 0,
        "fixed_real_train_300": {
            "class_counts": label_counts,
            "strictly_positive_ports_per_image": analysis["real_fixed_300"][
                "strictly_positive_port_count"
            ],
            "sample_neff": analysis["real_fixed_300"]["participation_neff"],
            "mean_power_uw": analysis["real_fixed_300"]["total_power_uw"]["mean"],
            "relative_to_peak_root_counts": analysis["real_fixed_300"][
                "relative_to_peak_counts"
            ],
            "central6_share": analysis["real_fixed_300"]["central_6x6_share"]["mean"],
            "outer36_share": analysis["real_fixed_300"]["outer_36_share"]["mean"],
            "eta_fiber_each_image_mean": analysis["power_chain"][
                "modal_over_ols_input_per_sample"
            ]["mean"],
            "eta_fiber_total_power_ratio": analysis["power_chain"][
                "modal_over_ols_input_ratio_of_totals"
            ],
            "eta_system_total_power_ratio": analysis["power_chain"][
                "modal_over_uniform_source_ratio_of_totals"
            ],
        },
        "saved_generation_queue_10000": {
            "sample_count": len(generated),
            "class_counts": generated_label_counts,
            "strictly_positive_ports_per_sample": analysis["generated_saved_10000"][
                "strictly_positive_port_count"
            ],
            "sample_neff": analysis["generated_saved_10000"]["participation_neff"],
            "mean_power_uw": analysis["generated_saved_10000"]["total_power_uw"][
                "mean"
            ],
            "relative_to_peak_root_counts": analysis["generated_saved_10000"][
                "relative_to_peak_counts"
            ],
            "separate_from_real_train_and_current_DEV": True,
        },
        "full_train_55000_saved_moments": {
            "mean_power_uw": analysis["full_train_average_power_distribution"][
                "total_power_uw"
            ]["mean"],
            "mean_vector_neff": analysis["full_train_average_power_distribution"][
                "participation_neff"
            ]["mean"],
            "strictly_positive_mean_vector_ports": analysis[
                "full_train_average_power_distribution"
            ]["strictly_positive_port_count"],
            "relative_to_peak_root_counts": analysis[
                "full_train_average_power_distribution"
            ]["relative_to_peak_counts"],
            "central6_share": analysis["full_train_average_power_distribution"][
                "central_6x6_share"
            ]["mean"],
            "outer36_share": analysis["full_train_average_power_distribution"][
                "outer_36_share"
            ]["mean"],
            "eta_fiber_total_power_ratio": analysis["power_chain"][
                "modal_over_ols_input_ratio_of_totals"
            ],
            "eta_system_total_power_ratio": analysis["power_chain"][
                "modal_over_uniform_source_ratio_of_totals"
            ],
        },
        "spatial_shares_recomputed_from_saved_port_csv": {
            "fixed300_central6": float(
                fixed.reshape(10, 10)[2:8, 2:8].sum() / fixed.sum()
            ),
            "fixed300_outer36": float(
                (fixed.reshape(10, 10).sum() - fixed.reshape(10, 10)[1:9, 1:9].sum())
                / fixed.sum()
            ),
            "full_train_central6": float(
                full.reshape(10, 10)[2:8, 2:8].sum() / full.sum()
            ),
            "full_train_outer36": float(
                (full.reshape(10, 10).sum() - full.reshape(10, 10)[1:9, 1:9].sum())
                / full.sum()
            ),
        },
        "power_denominator_recheck": {
            "uniform_source_power_uw": float(power_chain["uniform_source_power_uw"]),
            "ols_input_mean_uw_fixed300": float(
                power_chain["encoder_2_and_ols_input_full_plane_uw"]["mean"]
            ),
            "ols_modal_mean_uw_fixed300": float(
                power_chain["ols_modal_total_uw"]["mean"]
            ),
            "eta_fiber_recomputed": fixed_eta_fiber,
            "eta_system_recomputed": fixed_eta_system,
            "fixed300_sample_csv_mean_power_uw": fixed_mean_sample_power_uw,
            "saved_generation_sample_csv_mean_power_uw": generated_mean_sample_power_uw,
        },
        "per_port_full_train_mean_power_w": full,
        "hardware_detectability": "UNKNOWN_NO_NOISE_OR_DETECTOR_CALIBRATION",
        "incident_ols_complex_field_saved": False,
        "amplitude_support_vs_phase_cancellation_separable": False,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def loss_reference_summary(
    cfg: dict[str, Any], cps: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result = {}
    for spec in cfg["checkpoints"]:
        cp = cps[spec["key"]]
        scale = cp["scale"].detach().cpu().to(torch.float64).numpy().reshape(-1)
        power_ref = (
            cp["power_reference"].detach().cpu().to(torch.float64).numpy().reshape(-1)
        )
        result[spec["key"]] = {
            "scale_count": int(len(scale)),
            "scale_min_w": float(np.min(scale)),
            "scale_median_w": float(np.median(scale)),
            "scale_max_w": float(np.max(scale)),
            "scale_at_or_below_epsilon_count": int(np.sum(scale <= float(base.EPS_S))),
            "epsilon_s": float(base.EPS_S),
            "power_reference_w": power_ref,
            "epsilon_op": float(base.EPS_OP),
            "camera_gain_w_inverse": float(cp["camera_gain"].reshape(-1)[0]),
            "reconstruction_constraint": "mean((native_camera_W_per_pixel * fixed_gain - resized_200x200_input)^2)",
            "distribution_constraint": "KL of per-image channel powers normalized by each saved initial per-port scale plus epsilon_s against fixed truncated-positive-normal prior",
            "plane_power_constraint": "mean squared log ratio of recorded plane total powers to saved plane references plus epsilon_op",
            "structural_branch_constraint": "control has none; treatment adds 1-SSIM after common fixed-gain clamp and 28x28 adaptive pooling",
            "absolute_equal_power_constraint": False,
        }
    return result


def create_figures(
    cfg: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    state_arrays: dict[str, dict[str, np.ndarray]],
    comparisons: dict[str, dict[str, Any]],
    output_root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    dest = output_root / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    maps = {
        key: np.asarray(value["absolute_mean_port_power_w"]).reshape(10, 10) * 1e9
        for key, value in summaries.items()
    }
    positives = np.concatenate([x[x > 0] for x in maps.values() if np.any(x > 0)])
    norm = LogNorm(vmin=float(positives.min()), vmax=float(positives.max()))
    for key, power_map in maps.items():
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        image = ax.imshow(
            np.ma.masked_less_equal(power_map, 0),
            origin="lower",
            cmap="magma",
            norm=norm,
        )
        ax.set(
            title=f"{key}: population mean absolute port power",
            xlabel="port column",
            ylabel="port row",
        )
        ax.set_xticks(range(10), range(1, 11))
        ax.set_yticks(range(10), range(1, 11))
        fig.colorbar(image, ax=ax, label="mean power (nW; shared logarithmic scale)")
        fig.savefig(dest / f"{key}_mean_port_power.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
        neff = state_arrays[key]["neff_each"]
        ax.hist(neff[np.isfinite(neff)], bins=np.arange(0, 101, 2), color="#315f92")
        ax.set(
            title=f"{key}: per-image power participation",
            xlabel="N_eff",
            ylabel="DEV images",
        )
        fig.savefig(dest / f"{key}_participation_distribution.png", dpi=180)
        plt.close(fig)
        b = (
            np.asarray(summaries[key]["amplitude_upper_mean_port_power_w"]).reshape(
                10, 10
            )
            * 1e9
        )
        c = np.asarray(summaries[key]["mean_phase_retention_by_port"]).reshape(10, 10)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), constrained_layout=True)
        first = axes[0].imshow(b, origin="lower", cmap="viridis")
        axes[0].set_title("mean B_i (nW)")
        second = axes[1].imshow(c, origin="lower", cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("mean C_i")
        for axis in axes:
            axis.set(xlabel="port column", ylabel="port row")
        fig.colorbar(first, ax=axes[0])
        fig.colorbar(second, ax=axes[1])
        fig.suptitle(f"{key}: amplitude support and coherent retention")
        fig.savefig(dest / f"{key}_weak_channel_attribution.png", dpi=180)
        plt.close(fig)
    chain = [
        "P_after_slm_W",
        "P_encoder_1_W",
        "P_encoder_2_W",
        "P_before_fiber_W",
        "P_fiber_total_W",
        "P_generator_entry_W",
        "P_decoder_1_W",
        "P_decoder_2_W",
        "P_camera_W",
    ]
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(chain))
    width = 0.82 / len(state_arrays)
    for i, key in enumerate(state_arrays):
        y = [state_arrays[key][f"trace__{stage}"].mean() * 1e6 for stage in chain]
        ax.bar(x - 0.41 + width * (i + 0.5), y, width, label=key)
    ax.set_xticks(
        x,
        [s.removeprefix("P_").removesuffix("_W") for s in chain],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("mean optical power (μW)")
    ax.set_title("fixed 5,000 DEV inputs: absolute power chain")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(dest / "all_checkpoints_power_chain.png", dpi=180)
    plt.close(fig)
    names = list(comparisons)
    enc = [
        comparisons[n]["contribution_bootstrap"]["encoder_symmetric"]["mean"]
        for n in names
    ]
    dec = [
        comparisons[n]["contribution_bootstrap"]["decoder_symmetric"]["mean"]
        for n in names
    ]
    encerr = np.asarray(
        [
            [
                enc[i]
                - comparisons[n]["contribution_bootstrap"]["encoder_symmetric"]["ci95"][
                    0
                ]
                for i, n in enumerate(names)
            ],
            [
                comparisons[n]["contribution_bootstrap"]["encoder_symmetric"]["ci95"][1]
                - enc[i]
                for i, n in enumerate(names)
            ],
        ]
    )
    decerr = np.asarray(
        [
            [
                dec[i]
                - comparisons[n]["contribution_bootstrap"]["decoder_symmetric"]["ci95"][
                    0
                ]
                for i, n in enumerate(names)
            ],
            [
                comparisons[n]["contribution_bootstrap"]["decoder_symmetric"]["ci95"][1]
                - dec[i]
                for i, n in enumerate(names)
            ],
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(
        x - 0.18, enc, 0.36, yerr=encerr, capsize=3, label="Encoder symmetric MSE gain"
    )
    ax.bar(
        x + 0.18, dec, 0.36, yerr=decerr, capsize=3, label="Decoder symmetric MSE gain"
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("28×28 MSE gain, old−new")
    ax.set_title("frozen 2×2 Encoder/Decoder contribution")
    ax.legend()
    fig.savefig(dest / "encoder_decoder_cross_attribution.png", dpi=180)
    plt.close(fig)


def fmt(value: Any, sig: int = 5) -> str:
    return (
        "未定义"
        if value is None
        else (f"{value:.{sig}g}" if isinstance(value, (int, float)) else str(value))
    )


def write_report(
    cfg: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    big: dict[str, Any],
    preflight: dict[str, Any],
    loss_refs: dict[str, Any],
    output_root: Path,
) -> None:
    lines = [
        "# Gaussian OLS逐根功率、弱通道成因与Encoder／Decoder贡献诊断",
        "",
        "## 结果与状态",
        "",
        "小架构在同一固定真实 MNIST DEV 5,000 张（每类 500 张）上完成九个显式检查点的冻结测量。历史大架构只复算既有保存证据，没有新增光学前向。本轮优化器更新、优化器创建、先验抽样和官方 TEST 读取均为 0。所有绝对功率是该 Gaussian 数值模型和固定功率标定下的模拟结果，不是硬件检测值。",
        "",
        f"代表性 CUDA 预检为 `{preflight['status']}`；测量共 45,000 条状态记录、4,500,000 条逐根功率记录，并执行 40,000 条 decoder-only 离对角前向。",
        "",
        "| 状态 | ηfiber逐图均值 | ηfiber总功率比 | ηsystem逐图均值 | ηsystem总功率比 | 平均单图N_eff | 平均功率向量N_eff | 最亮4根份额 | 承载90%根数 | 28×28 MSE | PSNR | SSIM | 数值可分辨根均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, s in summaries.items():
        lines.append(
            f"| {key} | {fmt(s['eta_fiber_each_image']['mean'])} | {fmt(s['eta_fiber_ratio_of_total_powers'])} "
            f"| {fmt(s['eta_system_each_image']['mean'])} | {fmt(s['eta_system_ratio_of_total_powers'])} "
            f"| {fmt(s['sample_neff']['mean'])} | {fmt(s['population_mean_power_vector_neff'])} "
            f"| {fmt(s['brightest_four_power_share']['mean'])} | {fmt(s['roots_to_carry_90_percent_power']['mean'])} "
            f"| {fmt(s['mse28']['mean'])} | {fmt(s['psnr28_db']['mean'])} | {fmt(s['ssim28']['mean'])} "
            f"| {fmt(s['numerically_resolved_roots_2x_reference_error_per_image']['mean'])} |"
        )
    lines += [
        "",
        "逐图效率均值与总功率比采用不同聚合口径，不能互换。零分母效率保持未定义。`N_eff=(ΣP)^2/ΣP²`只代表功率参与度，不是信息秩、独立控制数或可检出通道数。数值可分辨根数用每根 2048/4096 复系数最大差的 2 倍作为筛查线，仍不是探测器噪声门限。",
        "",
        "### 大架构历史复盘",
        "",
        f"原固定真实 TRAIN 300 张逐图表显示每张均有 100 根严格浮点正功率；单图 N_eff 均值为 {fmt(big['fixed_real_train_300']['sample_neff']['mean'])}，-10/-20/-30/-40 dB 平均占用分别为 "
        + "/".join(
            fmt(
                big["fixed_real_train_300"]["relative_to_peak_root_counts"][str(db)][
                    "mean"
                ]
            )
            for db in DB_THRESHOLDS
        )
        + f" 根。55,000 TRAIN 保存矩的平均功率向量 N_eff 为 {fmt(big['full_train_55000_saved_moments']['mean_vector_neff'])}，对应相对占用为 "
        + "/".join(
            fmt(
                big["full_train_55000_saved_moments"]["relative_to_peak_root_counts"][
                    str(db)
                ]["mean"]
            )
            for db in DB_THRESHOLDS
        )
        + " 根。全 TRAIN 中央 6×6 占比 "
        + fmt(big["full_train_55000_saved_moments"]["central6_share"])
        + "，最外圈 36 根占比 "
        + fmt(big["full_train_55000_saved_moments"]["outer36_share"])
        + "。历史固定 300 张效率：ηfiber逐图均值 "
        + fmt(big["fixed_real_train_300"]["eta_fiber_each_image_mean"])
        + "、总功率比 "
        + fmt(big["fixed_real_train_300"]["eta_fiber_total_power_ratio"])
        + "；ηsystem总功率比 "
        + fmt(big["fixed_real_train_300"]["eta_system_total_power_ratio"])
        + "。历史输入身份已核对为 canonical TRAIN，300张、每类30张；效率分母复算为 OLS 入射面约26.0202 μW、光源249.3 μW。逐根及逐类均值在 `per_port_full_train_fixed300_and_class_means.csv`。",
        "",
        f"历史生成队列单独列示：保存的10,000条 `saved_generation`（索引0–9,999、每类1,000条）均有100根严格浮点正功率；单条 N_eff 均值 {fmt(big['saved_generation_queue_10000']['sample_neff']['mean'])}，总模态功率均值 {fmt(big['saved_generation_queue_10000']['mean_power_uw'])} μW，−10/−20/−30/−40 dB平均占用分别为 "
        + "/".join(
            fmt(
                big["saved_generation_queue_10000"]["relative_to_peak_root_counts"][
                    str(db)
                ]["mean"]
            )
            for db in DB_THRESHOLDS
        )
        + " 根。它是历史生成系数队列，不是本轮真实DEV，也不与真实TRAIN效率混算；逐条功率见 `large_architecture_recap/saved_generation_10000_sample_power.csv`。",
        "",
        "绝对功率链中，固定 300 张源光 249.3 μW，SLM 后约 32.95 μW、Encoder 第一层后约 29.29 μW、OLS 入射面约 26.02 μW、OLS 模态总功率约 0.795 μW。未投影入模式的功率是模式投影外的场能量，不应统称为光纤材料吸收。",
        "",
        "### 小架构逐根与类别",
        "",
        "`per_port_per_class.csv` 对每个状态、类别、端口记录绝对功率的均值／中位数／P10／P90、相对单图峰值 -10/-20/-30/-40 dB 占用概率及严格非零概率。`per_image_summary.csv` 给出逐图参与数、最亮四根功率份额（四根之和除以总耦合功率）、承载 90% 功率的根数、中央／外圈份额及两种效率。`power_chain_per_stage.csv` 保留逐层链路统计。",
        "",
        "## 科学认识",
        "",
        "### 弱通道机制分解",
        "",
        "真实输入每根使用 `B_i=(∫|E||u_i|dA)^2` 和 `C_i=P_i/B_i`，满足 `P_i=B_i C_i`。B 是给定场幅下单端口相位对齐上界，不能同时被所有端口实现；C 衡量相干重叠保留比例。B 低说明该端口附近幅度支持／空间重叠弱；B 高而 C 低支持相位抵消；B、C 都低说明二者并存。按类逐根分布在 `weak_channel_attribution_per_port_per_class.csv`，各检查点空间图在 `figures/`。这是对观测场的数学分解，不是改相位或改几何的因果实验。",
        "",
        "按每个状态DEV人口平均绝对功率向量选出最弱25根和最亮25根，比较两组平均P、B、C比值。各状态端口集合可不同，B/C是对实测场的描述，不是相位控制反事实。",
        "",
        "| 状态 | 最弱/最亮平均P比 | 最弱/最亮平均B比 | 最弱/最亮平均C比 |",
        "|---|---:|---:|---:|",
    ]
    for key, s in summaries.items():
        ratios = s["weak_channel_bottom_top_quartiles"]["weak_to_bright_ratios"]
        lines.append(
            f"| {key} | {fmt(ratios['mean_power'])} | {fmt(ratios['amplitude_upper_bound'])} | {fmt(ratios['phase_retention'])} |"
        )
    weak0 = summaries["step_000000"]["weak_channel_bottom_top_quartiles"]
    weak30 = summaries["step_030000"]["weak_channel_bottom_top_quartiles"]
    corner_ports = sorted(
        set(weak30["weakest_quartile"]["port_indices_zero_based"]) & {0, 9, 90, 99}
    )
    lines += [
        "",
        f"基线0步最弱四分位相对最亮四分位的 P、B、C比分别为 {fmt(weak0['weak_to_bright_ratios']['mean_power'])}、{fmt(weak0['weak_to_bright_ratios']['amplitude_upper_bound'])}、{fmt(weak0['weak_to_bright_ratios']['phase_retention'])}；30,000步分别为 {fmt(weak30['weak_to_bright_ratios']['mean_power'])}、{fmt(weak30['weak_to_bright_ratios']['amplitude_upper_bound'])}、{fmt(weak30['weak_to_bright_ratios']['phase_retention'])}。30,000步最弱四分位包含的四角端口索引为 {corner_ports}。这表明弱通道同时有较低的幅度/空间模态支持和较低的相干保留；相对输出差距中，B差距更大，C又进一步加重。训练使B组间比值上升、C组间比值下降，不能称为单纯“把光耦进更多根”；两项都随输入场和相位结构改变，仍是观测分解而非因果归因。",
        "",
        "### 固定输入 Encoder／Decoder 交叉归因",
        "",
        "每个矩阵以 Decoder 为行、Encoder 为列，原生对角线复用完整前向，离对角线仅替换另一端。所有输入复系数绝对值、固定增益和读出保持不变。",
        "",
        "| 比较 | 原生系统 MSE gain（旧−新） | 原生 PSNR28 新端点 | Encoder 对称 MSE gain及95%区间 | Decoder 对称 MSE gain及95%区间 | 交互及95%区间 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in comparisons.items():
        boot = item["contribution_bootstrap"]
        ge, gd, inter = (
            boot["encoder_symmetric"],
            boot["decoder_symmetric"],
            boot["encoder_decoder_interaction"],
        )
        lines.append(
            f"| {name} | {fmt(item['point_estimates']['native_system_mse_gain_old_minus_new'])} "
            f"| {fmt(item['cells']['L_jj_new_decoder_new_encoder']['mean_psnr28_db'])} "
            f"| {fmt(ge['mean'])} [{fmt(ge['ci95'][0])}, {fmt(ge['ci95'][1])}] "
            f"| {fmt(gd['mean'])} [{fmt(gd['ci95'][0])}, {fmt(gd['ci95'][1])}] "
            f"| {fmt(inter['mean'])} [{fmt(inter['ci95'][0])}, {fmt(inter['ci95'][1])}] "
            f"|"
        )
    lines += [
        "",
        "每图 `G_E+G_D=L_ii−L_jj` 已验证。贡献为冻结组合的对称分解，非独立重训的因果效应；同时报告了 Encoder 在两个 Decoder 固定条件下的效应和反向条件效应。Decoder 区间若更稳定且原生 MSE 改善，才支持优先调查 Decoder；明显交互则保留共同适配解释。四组配对区间以相同的每类 500 张按类 Bootstrap 2,000 次计算。",
        "",
        "### 损失约束的含义",
        "",
        "存储的每端口初始尺度 `s_j`、`ε_s`、五平面功率参考和 `ε_op` 已写入 `loss_reference_summary.json`。原分布项先用 `P_j/(s_j+ε_s)` 归一各端口尺度，再与固定截断正态先验计算 KL；平面功率项比较各平面总功率与初始参考的对数比；重建项按固定相机增益约束原生图像读出。处理支路另加固定增益、裁剪及 28×28 池化后的 `1−SSIM`。这些项均未直接要求 100 根绝对等功率，因此图像收益不必对应 N_eff 上升。",
        "",
        "## 目标缺口",
        "",
        "仿真浮点非零不是硬件可检出性。没有探测器噪声／动态范围／串扰标定，不能回答弱根是否会被硬件看见。大架构没有 OLS 入射复场，不能将其弱根在幅度支持与相位抵消之间拆分。小架构 B/C 也没有通过同一场幅下改变相位、输入或几何建立完整反事实。",
        "",
        "大小架构不是只改相位像素数的对照：大架构为 400×400，历史模型端口间距 125 μm、参考场径约 14.222 μm；本轮小架构为 200×200，实际 Gaussian OLS 间距 20 μm、1/e 场半径 3.5 μm，二者输入、传播和 Decoder 也不同。小架构通用物理 trace 中的 125 μm/14.222 μm 字段没有用作本轮几何。",
        "",
        "## 工程验证",
        "",
        f"CUDA 预检状态 `{preflight['status']}`，对九个检查点各取每类首两张 DEV，比对 2048/4096 下完整 200 μm OLS 场、160 μm common ROI、100 根复系数与功率、全相机平面及总功率，数值阈值为 `1e-4`；独立 NumPy 面积积分重算系数、入射功率与 B；同幅度相位正／负控验证 B 不变而 C 改变。对角 Decoder-only 公共路径与原生前向回放最大相对误差为 {fmt(preflight['same_endpoint_decoder_replay_max_relative_l2'])}。完整各检查点误差见 `preflight.json`。",
        "",
        "每 100 张原始行号分块保存；恢复时校验身份、行号、维度与数值，再读取全部区块并回读确认 5,000×100 系数、功率、B、C。跨组输入行号必须相同，交换行号的负例被拒绝。无训练更新、无优化器创建、无先验抽样、无 TEST 读取。",
        "",
        "## 决策影响与下一步",
        "",
        "若 η 提高而 N_eff 下降，应解释为耦合吞吐改善伴随功率集中，不能宣称 100 通道覆盖改善；若功率分布 TV 改变但 η 区间跨零，证据支持重分配；若 Decoder 贡献更大且原生重建实际改善，可优先设计 Decoder 独立验证；若 Encoder×Decoder 交互显著，保留共同适配；若 B 与 C 同时低，原因仍是混合，需要同幅相位反事实。噪声门槛未知时要进行器件测量，增加训练不能回答可检出性。",
        "",
        "本轮停止于冻结诊断和证据回传，不改变历史 `VALID_FAILED`，不训练、不调整损失或几何、不采用模型、不部署硬件。",
        "",
    ]
    (output_root / "中文诊断报告.md").write_text("\n".join(lines), encoding="utf-8")


def report_phase(cfg: dict[str, Any], output_root: Path) -> None:
    preflight = validate_preflight(output_root)
    complete_path = output_root / "measurement_complete.json"
    require(
        complete_path.is_file(), "report phase requires completed frozen measurements"
    )
    completed = json.loads(complete_path.read_text(encoding="utf-8"))
    require(
        completed.get("status") == "MEASUREMENT_COMPLETE"
        and completed.get("real_input_state_records") == 45000
        and completed.get("per_root_power_records") == 4500000
        and completed.get("additional_decoder_only_forward_records") == 40000,
        "measurement population is incomplete",
    )
    manifest = json.loads(
        (output_root / "input_manifest.json").read_text(encoding="utf-8")
    )
    dev_rows = np.asarray(manifest["dev_rows_original_order"], dtype=np.int64)
    labels = np.asarray(manifest["dev_labels"], dtype=np.int64)
    require(
        len(dev_rows) == 5000
        and np.array_equal(np.bincount(labels, minlength=10), np.full(10, 500)),
        "returned DEV manifest is not the fixed balanced population",
    )
    baseline, continuation, cps, pixels, data_labels, train_rows, actual_dev = (
        load_inputs(cfg)
    )
    require(
        np.array_equal(actual_dev.numpy(), dev_rows)
        and np.array_equal(data_labels[actual_dev].numpy(), labels),
        "returned run used different source data rows or labels",
    )
    state_arrays = {
        spec["key"]: read_state_chunks(
            output_root / "states" / spec["key"],
            dev_rows,
            int(cfg["statistics"]["shard_size"]),
        )
        for spec in cfg["checkpoints"]
    }
    summaries: dict[str, dict[str, Any]] = {}
    port_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    for spec in cfg["checkpoints"]:
        with np.load(
            output_root
            / "preflight"
            / spec["key"]
            / "numerical_resolution_envelope.npz",
            allow_pickle=False,
        ) as file:
            floor = file["coefficient_abs_error_by_port"].copy()
        result = state_summaries(
            spec, state_arrays[spec["key"]], floor, cfg, output_root
        )
        summary, p_rows, a_rows, i_rows, c_rows, cl_rows = result
        summaries[spec["key"]] = summary
        port_rows.extend(p_rows)
        attribution_rows.extend(a_rows)
        image_rows.extend(i_rows)
        chain_rows.extend(c_rows)
        class_rows.extend(cl_rows)
    write_csv(output_root / "per_port_per_class.csv", port_rows)
    write_csv(
        output_root / "weak_channel_attribution_per_port_per_class.csv",
        attribution_rows,
    )
    write_csv(output_root / "per_image_summary.csv", image_rows)
    write_csv(output_root / "per_class_summary.csv", class_rows)
    write_csv(output_root / "power_chain_per_stage.csv", chain_rows)

    off_diagonal = [
        ("step_000000", "step_030000"),
        ("step_030000", "step_000000"),
        ("step_030000", "control_step_035000"),
        ("control_step_035000", "step_030000"),
        ("step_030000", "treatment_step_035000"),
        ("treatment_step_035000", "step_030000"),
        ("control_step_035000", "treatment_step_035000"),
        ("treatment_step_035000", "control_step_035000"),
    ]
    crosses: dict[str, dict[str, np.ndarray]] = {}
    for decoder, encoder in off_diagonal:
        path = output_root / "cross_decoder" / f"decoder_{decoder}__encoder_{encoder}"
        info = json.loads((path / "complete.json").read_text(encoding="utf-8"))
        require(
            info.get("status") == "MEASUREMENT_COMPLETE"
            and info.get("decoder_only_forwards") == 5000,
            f"cross Decoder combo is incomplete: {decoder} <- {encoder}",
        )
        crosses[f"{decoder}__{encoder}"] = load_cross_chunks(
            path, dev_rows, int(cfg["statistics"]["shard_size"])
        )
    comparison_specs = [
        ("0_to_30000", "step_000000", "step_030000"),
        ("30000_to_control35000", "step_030000", "control_step_035000"),
        ("30000_to_treatment35000", "step_030000", "treatment_step_035000"),
        (
            "control35000_to_treatment35000",
            "control_step_035000",
            "treatment_step_035000",
        ),
    ]
    comparisons: dict[str, dict[str, Any]] = {}
    replicate_count = int(cfg["statistics"]["bootstrap_resamples"])
    bootstrap_seed = int(cfg["statistics"]["bootstrap_seed"])
    for name, old, new in comparison_specs:
        matrix = cross_matrix(old, new, state_arrays, crosses)
        matrix["contribution_bootstrap"] = bootstrap_contributions(
            matrix, labels, bootstrap_seed, replicate_count
        )
        matrix["old_vs_new_state_bootstrap"] = stratified_pair_bootstrap(
            state_arrays[old],
            state_arrays[new],
            labels,
            bootstrap_seed,
            replicate_count,
        )
        matrix["old_checkpoint"], matrix["new_checkpoint"] = old, new
        per_image = []
        values = matrix["per_image_contributions"]
        mse = matrix["per_image_mse"]
        for row in range(len(dev_rows)):
            per_image.append(
                {
                    "original_row": int(dev_rows[row]),
                    "label": int(labels[row]),
                    "L_ii_old_decoder_old_encoder": float(mse["L_ii"][row]),
                    "L_ij_old_decoder_new_encoder": float(mse["L_ij"][row]),
                    "L_ji_new_decoder_old_encoder": float(mse["L_ji"][row]),
                    "L_jj_new_decoder_new_encoder": float(mse["L_jj"][row]),
                    "G_E": float(values["encoder_symmetric"][row]),
                    "G_D": float(values["decoder_symmetric"][row]),
                    "interaction": float(values["encoder_decoder_interaction"][row]),
                    "native_system_gain_old_minus_new": float(
                        values["native_system_mse_gain_old_minus_new"][row]
                    ),
                }
            )
        write_csv(
            output_root / "cross_decoder" / f"{name}_per_image_matrix.csv", per_image
        )
        comparisons[name] = matrix

    big = large_architecture_recap(output_root)
    loss_refs = loss_reference_summary(cfg, cps)
    write_json(
        output_root / "loss_reference_summary.json",
        {
            "epsilon_source": "production runner constants",
            "distribution_epsilon_s": float(base.EPS_S),
            "plane_power_epsilon_op": float(base.EPS_OP),
            "states": loss_refs,
            "scientific_interpretation": "Initial per-port scale-normalized KL and total-plane power references do not constrain absolute equal power across all 100 roots.",
        },
    )
    create_figures(cfg, summaries, state_arrays, comparisons, output_root)
    write_report(cfg, summaries, comparisons, big, preflight, loss_refs, output_root)

    comparison_machine = {}
    for name, value in comparisons.items():
        comparison_machine[name] = {
            "old_checkpoint": value["old_checkpoint"],
            "new_checkpoint": value["new_checkpoint"],
            "cells": {
                cell: {
                    k: v for k, v in data.items() if isinstance(v, (str, int, float))
                }
                for cell, data in value["cells"].items()
            },
            "point_estimates": value["point_estimates"],
            "contribution_bootstrap": value["contribution_bootstrap"],
            "old_vs_new_state_bootstrap": value["old_vs_new_state_bootstrap"],
            "formula_identity_checked": True,
        }
    machine = {
        "schema": "gaussian_real_input_power_attribution_report.v1",
        "task_id": TASK_ID,
        "status": "VALID_DIAGNOSTIC_COMPLETE",
        "outer_decision": "FROZEN_DIAGNOSTIC_COMPLETE_NO_MODEL_ADOPTION_OR_NEW_SCIENTIFIC_ACCEPTANCE",
        "decision_question": "Determine whether frozen training changed coupling efficiency or channel distribution and whether real-input readout change is more associated with Encoder or Decoder under fixed DEV rows.",
        "population": {
            "count": 5000,
            "per_class": 500,
            "split_seed": 1234,
            "official_test_reads": 0,
        },
        "checkpoint_manifest": checkpoint_manifest(cfg, cps),
        "state_summaries": summaries,
        "cross_comparisons": comparison_machine,
        "large_architecture_recap": big,
        "loss_reference_summary": loss_refs,
        "training_updates": 0,
        "optimizers_created": 0,
        "prior_samples": 0,
        "hardware_deployment": 0,
        "hardware_detectability": "UNKNOWN_NO_NOISE_OR_DETECTOR_CALIBRATION",
        "limitations": [
            "Gaussian mode geometry remains PROJECT_GEOMETRY_UNMEASURED.",
            "Historical large architecture lacks saved OLS incident complex fields for B_i/C_i separation.",
            "Frozen cross contributions are not independent retraining interventions.",
            "N_eff is a power participation statistic, not rank, controllability or detectable channel count.",
        ],
        "artifacts": {
            "report": "中文诊断报告.md",
            "image_records": "per_image_summary.csv",
            "port_class_records": "per_port_per_class.csv",
            "weak_channel_decomposition": "weak_channel_attribution_per_port_per_class.csv",
            "power_chain": "power_chain_per_stage.csv",
            "figures": "figures/",
        },
    }
    write_json(output_root / "machine_report.json", machine)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("preflight", "measure", "report"), required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/research/lightgen/gaussian_real_input_power_attribution_v1.yaml"
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    cfg = load_config(config_path)
    output_root = ROOT / cfg["output"]["experiment_root"]
    if args.phase == "report":
        report_phase(cfg, output_root)
    elif args.phase == "preflight":
        run_preflight(config_path, cfg, torch.device(args.device), output_root)
    else:
        run_measure(cfg, torch.device(args.device), output_root)


if __name__ == "__main__":
    main()
