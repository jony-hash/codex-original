"""Bounded, assumption-only process-aware height training for HW207 C3.4.

This branch reuses the C3.4 production wave-optics forward model and fixed
MNIST roles. It trains one physical height map through the same assumed print
operator used by C3.4 P4; it does not calibrate or validate a real printer.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.research.lightgen import execute_hw207_mnist4_c34_v1 as runner
from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as c33
from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34


PROCESS_NAME = "C3.4_PROCESS_AWARE_HEIGHT_TRAINING"
_FABRICATION_CONFIG = c34.load_config()["fabrication_diagnostic"]
SCENARIO = {
    "delta_n_primary": float(_FABRICATION_CONFIG["delta_n_scenarios"][0]),
    "delta_n_sensitivity": float(_FABRICATION_CONFIG["delta_n_scenarios"][1]),
    "psf_sigma_px": float(_FABRICATION_CONFIG["historical_psf_sigma_px"]),
    "minimum_feature_radius_px": int(_FABRICATION_CONFIG["minimum_feature_opening_radius_px"]),
    "height_levels": int(_FABRICATION_CONFIG["height_levels"]),
    "height_noise_nm": float(_FABRICATION_CONFIG["height_noise_sigma_nm"]),
    "edge_height_bias_nm": [float(v) for v in _FABRICATION_CONFIG["edge_height_error_scenarios_nm"]],
    "maximum_height_um": float(_FABRICATION_CONFIG["maximum_height_um"]),
    "validation_noise_seeds": [int(v) for v in _FABRICATION_CONFIG["height_noise_seeds"]],
}


def _noise_field(height: torch.Tensor, seed: int, sigma_nm: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(height.shape, generator=generator, dtype=torch.float32)
    return noise.to(device=height.device, dtype=height.dtype) * (float(sigma_nm) / 1000.0)


def process_height(
    design_height_um: torch.Tensor,
    *,
    psf_sigma_px: float,
    minimum_feature_radius_px: int,
    height_levels: int,
    maximum_height_um: float,
    noise_nm: float = 0.0,
    noise_seed: int = 0,
    edge_height_bias_nm: float = 0.0,
    noise_field: torch.Tensor | None = None,
    straight_through_quantization: bool = False,
) -> torch.Tensor:
    """Apply the P4 height-domain process, optionally with quantizer STE.

    The forward values remain those of the hard P4 operator. Only the
    derivative of round-to-levels is replaced by a straight-through estimate.
    """
    if (design_height_um.shape != (100, 100) or not torch.isfinite(design_height_um).all()
            or not math.isfinite(float(psf_sigma_px)) or psf_sigma_px < 0
            or minimum_feature_radius_px < 0 or height_levels < 0 or height_levels == 1
            or maximum_height_um <= 0 or noise_nm < 0
            or not math.isfinite(float(edge_height_bias_nm))):
        raise ValueError("invalid physical-height fabrication input or operator")
    height = design_height_um
    if psf_sigma_px > 0:
        height = c33.smooth_height(height, float(psf_sigma_px))
    if minimum_feature_radius_px > 0:
        height = c33.min_feature_open(height, int(minimum_feature_radius_px))
    if height_levels >= 2:
        quantized = c33.quantize_height(height, int(height_levels), float(maximum_height_um))
        height = (height + (quantized - height).detach()
                  if straight_through_quantization else quantized)
    if noise_nm > 0:
        if noise_field is None:
            noise_field = _noise_field(height, noise_seed, noise_nm)
        if noise_field.shape != height.shape or not torch.isfinite(noise_field).all():
            raise ValueError("fabrication noise field must be finite and match the height map")
        height = height + noise_field
    if edge_height_bias_nm != 0:
        delta = float(edge_height_bias_nm) / 1000.0
        height = height.clone()
        height[0, :] += delta
        height[-1, :] += delta
        height[:, 0] += delta
        height[:, -1] += delta
    return height.clamp(0.0, float(maximum_height_um))


def _process_phase_from_height(height: torch.Tensor, delta_n: float, **kwargs) -> torch.Tensor:
    processed_height = process_height(height, **kwargs)
    return c34.phase_from_height(processed_height, float(delta_n))


def _bounded_design_height(initial_phase: torch.Tensor, delta_n: float) -> torch.Tensor:
    maximum = min(float(SCENARIO["maximum_height_um"]), c34.WAVELENGTH_UM / float(delta_n))
    return c34.height_from_phase(initial_phase.detach(), float(delta_n)).float().clamp(0.0, maximum)


def _step_noise_seed(seed: int, epoch: int, batch_index: int) -> int:
    return int((1_000_003 * seed + 9_176 * epoch + 97 * batch_index + 34_011) % (2**31 - 1))


def _step_edge_bias(seed: int, epoch: int, batch_index: int) -> float:
    choices = SCENARIO["edge_height_bias_nm"]
    return float(choices[(seed + epoch + batch_index) % len(choices)])


def _process_options(*, training: bool, noise_seed: int, edge_bias_nm: float,
                     noise_field: torch.Tensor | None = None) -> dict[str, Any]:
    cfg = SCENARIO
    return {
        "psf_sigma_px": cfg["psf_sigma_px"],
        "minimum_feature_radius_px": cfg["minimum_feature_radius_px"],
        "height_levels": cfg["height_levels"],
        "maximum_height_um": cfg["maximum_height_um"],
        "noise_nm": cfg["height_noise_nm"],
        "noise_seed": noise_seed,
        "edge_height_bias_nm": edge_bias_nm,
        "noise_field": noise_field,
        "straight_through_quantization": training,
    }


def _validate_source_run(run_dir: Path) -> tuple[torch.Tensor, dict[str, Any], float]:
    phase, phase_source = runner._load_geometry_adapted_phase(run_dir)
    p3_root = run_dir / "p3_geometry_adapted_training"
    preflight_path = p3_root / "training_preflight.json"
    fast_path = p3_root / "fast_train_1000" / "fast_training_summary.json"
    if not preflight_path.is_file() or not fast_path.is_file():
        raise FileNotFoundError("process-aware training requires the completed C3.4 P3 preflight and fast gate")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    fast = json.loads(fast_path.read_text(encoding="utf-8"))
    if preflight.get("preflight_pass") is not True or fast.get("fast_training_pass") is not True:
        raise RuntimeError("process-aware training requires passing C3.4 P3 gates")
    floor = float(preflight["throughput_floor"]["efficiency_vs_input"])
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("C3.4 P3 throughput floor is missing or invalid")
    return phase, {"phase": phase_source, "p3_preflight": str(preflight_path),
                   "p3_fast_gate": str(fast_path)}, floor


def _make_context(args) -> dict[str, Any]:
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("process-aware C3.4 training requires the authorized CUDA host")
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    cache_path = args.field_cache_path.resolve()
    if os.name == "posix":
        system_root = Path("/root/hw207_c34_system").resolve()
        if not output_dir.is_relative_to(system_root):
            raise ValueError("process-aware run outputs must remain under /root/hw207_c34_system")
        if not cache_path.is_relative_to(system_root):
            raise ValueError("process-aware input-field cache must remain under /root/hw207_c34_system")
    if not run_dir.is_dir() or not cache_path.is_file():
        raise FileNotFoundError("explicit C3.4 source run or system-disk input cache is missing")
    initial_phase, source, throughput_floor = _validate_source_run(run_dir)
    images, labels, split = runner._split_and_data()
    device = torch.device(args.device)
    input_configuration = c33.build_configuration(52_000.0, str(device))
    input_configuration.base.screen.requires_grad_(False)
    cache_rows, cache_fields, cache_metadata = runner._load_or_prepare_formal_field_cache(
        images, split, input_configuration, device, cache_path
    )
    field_cache = (cache_rows, cache_fields)
    lens_pitch = 2.0
    model_audit_path = run_dir / "p0_p2" / "optical_model_audit.json"
    if model_audit_path.is_file():
        model_audit = json.loads(model_audit_path.read_text(encoding="utf-8"))
        if not model_audit.get("numerical_pass"):
            raise RuntimeError("process-aware training stopped because C3.4 P1 numerical audit did not pass")
        lens_pitch = float(model_audit["model_b"]["lens_plane_pitch_um"]["selected_for_next_stage_um"])
    return {"run_dir": run_dir, "output_dir": output_dir, "images": images,
            "labels": labels, "split": split, "device": device,
            "initial_phase": initial_phase.to(device), "source": source,
            "throughput_floor": throughput_floor, "input_configuration": input_configuration,
            "field_cache": field_cache, "field_cache_metadata": cache_metadata,
            "lens_pitch_um": lens_pitch,
            "propagation_cache": {}}


def _process_update(height: torch.nn.Parameter, optimizer, context: dict[str, Any],
                    rows: torch.Tensor, *, seed: int, epoch: int, batch_index: int,
                    microbatch: int) -> tuple[float, float]:
    total = int(len(rows))
    if total == 0:
        raise ValueError("process-aware update cannot use an empty logical batch")
    noise_seed = _step_noise_seed(seed, epoch, batch_index)
    noise_field = _noise_field(height, noise_seed, float(SCENARIO["height_noise_nm"]))
    edge_bias = _step_edge_bias(seed, epoch, batch_index)
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for start in range(0, total, microbatch):
        batch_rows = rows[start:start + microbatch]
        effective_phase = _process_phase_from_height(
            height, float(SCENARIO["delta_n_primary"]),
            **_process_options(training=True, noise_seed=noise_seed,
                               edge_bias_nm=edge_bias, noise_field=noise_field)
        )
        fields = runner._fields_for_rows(context["field_cache"], batch_rows)
        result = c34.model_b_forward(
            fields, effective_phase, 2.0, lens_pitch_um=context["lens_pitch_um"],
            plan_cache=context["propagation_cache"], return_camera=False
        )
        labels = context["labels"][batch_rows].to(context["device"])
        loss = runner._c34_loss(result, labels, effective_phase,
                                float(context["throughput_floor"]))
        (loss * (len(batch_rows) / total)).backward()
        losses.append(float(loss.detach()))
    if height.grad is None or not torch.isfinite(height.grad).all():
        raise RuntimeError("process-aware height gradient is absent or non-finite")
    gradient_norm = float(height.grad.detach().norm())
    if gradient_norm <= 0:
        raise RuntimeError("process-aware height gradient is zero")
    optimizer.step()
    max_design_height = min(float(SCENARIO["maximum_height_um"]),
                            c34.WAVELENGTH_UM / float(SCENARIO["delta_n_primary"]))
    with torch.no_grad():
        height.clamp_(0.0, max_design_height)
    return float(np.mean(losses)), gradient_norm


def _evaluate_phase(context: dict[str, Any], rows: torch.Tensor, phase: torch.Tensor,
                    *, batch_size: int) -> dict[str, Any]:
    fields = runner._fields_for_rows(context["field_cache"], rows)
    return runner._infer_fields_b(fields, context["labels"], rows, phase.detach(), 2.0,
                                  context["lens_pitch_um"], context["device"], batch_size,
                                  context["propagation_cache"])


def _evaluate_height(context: dict[str, Any], rows: torch.Tensor, height: torch.Tensor,
                     delta_n: float, *, noise_seed: int, edge_bias_nm: float,
                     batch_size: int, processed: bool = True,
                     noise_nm: float | None = None) -> dict[str, Any]:
    if processed:
        options = _process_options(training=False, noise_seed=noise_seed,
                                   edge_bias_nm=edge_bias_nm)
        if noise_nm is not None:
            options["noise_nm"] = float(noise_nm)
        output_height = process_height(height.detach(), **options)
    else:
        output_height = height.detach()
    phase = c34.phase_from_height(output_height, float(delta_n)).to(context["device"])
    return _evaluate_phase(context, rows, phase, batch_size=batch_size)


def _evaluate_ensemble(context: dict[str, Any], rows: torch.Tensor, height: torch.Tensor,
                       delta_n: float, *, batch_size: int,
                       save_arrays: dict[str, np.ndarray] | None = None,
                       prefix: str = "") -> dict[str, Any]:
    results = []
    for noise_seed in SCENARIO["validation_noise_seeds"]:
        for edge_bias in SCENARIO["edge_height_bias_nm"]:
            result = _evaluate_height(context, rows, height, delta_n,
                                      noise_seed=int(noise_seed), edge_bias_nm=float(edge_bias),
                                      batch_size=batch_size)
            metrics = result["metrics"]
            if not all(math.isfinite(float(metrics[k])) for k in (
                    "accuracy_all_samples", "macro_f1_classes_0_to_3",
                    "mean_full_sensor_efficiency_vs_input")):
                raise RuntimeError("process-aware validation produced non-finite metrics")
            results.append({"noise_seed": int(noise_seed), "edge_height_bias_nm": float(edge_bias),
                            "metrics": metrics})
            if save_arrays is not None:
                token = f"{prefix}_s{int(noise_seed)}_e{'p' if edge_bias > 0 else 'm'}{abs(int(edge_bias))}"
                save_arrays[f"{token}_scores"] = result["scores"].to(torch.float32).numpy()
                save_arrays[f"{token}_predictions"] = result["predictions"].numpy().astype(np.int8)
                save_arrays[f"{token}_detected_energy"] = result["detected"].to(torch.float32).numpy()
                save_arrays[f"{token}_crop_energy"] = result["crop_energy"].to(torch.float32).numpy()
    metric_names = ("accuracy_all_samples", "macro_f1_classes_0_to_3",
                    "mean_full_sensor_efficiency_vs_input", "mean_detected_energy_um2",
                    "mean_roi_outside_crop_energy_um2")
    summary = {f"mean_{name[5:] if name.startswith('mean_') else name}": float(
        np.mean([r["metrics"][name] for r in results]))
               for name in metric_names if all(name in r["metrics"] for r in results)}
    summary.update({f"sd_across_process_realizations_{name[5:] if name.startswith('mean_') else name}": float(np.std(
        [r["metrics"][name] for r in results], ddof=1))
        for name in metric_names if len(results) > 1 and all(name in r["metrics"] for r in results)})
    summary["process_realization_count"] = len(results)
    summary["scenarios"] = results
    return summary


def _phase_height_jump_summary(phase: torch.Tensor, delta_n: float) -> dict[str, float]:
    height = c34.height_from_phase(phase.detach().cpu(), delta_n).float()
    jumps = torch.cat([(height[:, 1:] - height[:, :-1]).abs().flatten(),
                       (height[1:, :] - height[:-1, :]).abs().flatten()])
    return {"height_min_um": float(height.min()), "height_max_um": float(height.max()),
            "neighbor_jump_max_um": float(jumps.max()),
            "neighbor_jump_q95_um": float(torch.quantile(jumps, 0.95)),
            "neighbor_jump_fraction_gt_0p2um": float((jumps > 0.2).float().mean())}


def _derivative_reference() -> dict[str, float]:
    generator = torch.Generator().manual_seed(340501)
    base = (0.2 + 0.7 * torch.rand((100, 100), generator=generator, dtype=torch.float64))
    weights = torch.randn((100, 100), generator=generator, dtype=torch.float64)
    coordinate = (45, 52)
    base.requires_grad_(True)
    smooth = process_height(base, psf_sigma_px=0.8, minimum_feature_radius_px=0,
                            height_levels=0, maximum_height_um=1.2,
                            straight_through_quantization=False)
    objective = (smooth * weights).sum()
    analytic = torch.autograd.grad(objective, base)[0][coordinate].item()
    epsilon = 1e-5
    with torch.no_grad():
        plus = base.detach().clone()
        minus = base.detach().clone()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        plus_value = (process_height(plus, psf_sigma_px=0.8, minimum_feature_radius_px=0,
                                     height_levels=0, maximum_height_um=1.2) * weights).sum()
        minus_value = (process_height(minus, psf_sigma_px=0.8, minimum_feature_radius_px=0,
                                      height_levels=0, maximum_height_um=1.2) * weights).sum()
        finite_difference = ((plus_value - minus_value) / (2 * epsilon)).item()
    relative_error = abs(analytic - finite_difference) / max(abs(analytic), abs(finite_difference), 1e-12)
    return {"analytic_derivative": analytic, "central_difference": finite_difference,
            "relative_error": relative_error}


def _checkpoint_payload(height: torch.Tensor, optimizer, *, seed: int, epoch: int,
                        generator: torch.Generator, history: list[dict[str, Any]],
                        best_height: torch.Tensor, best_accuracy: float, best_epoch: int,
                        train_rows: torch.Tensor, validation_rows: torch.Tensor) -> dict[str, Any]:
    return {"identity": "C3.4_PROCESS_AWARE_HEIGHT_CHECKPOINT", "seed": seed, "epoch": epoch,
            "height": height.detach().cpu(), "optimizer": optimizer.state_dict(),
            "generator_state": generator.get_state(), "history": history,
            "best_height": best_height.detach().cpu(), "best_accuracy": best_accuracy,
            "best_epoch": best_epoch, "train_rows": train_rows.cpu(),
            "validation_rows": validation_rows.cpu(),
            "process_assumptions": dict(SCENARIO),
            "session_source_id": c34.SESSION_SOURCE_ID}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _run_preflight(context: dict[str, Any], args) -> dict[str, Any]:
    output = context["output_dir"] / "preflight"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "preflight.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite process-aware preflight: {report_path}")
    phase = context["initial_phase"]
    initial_height = _bounded_design_height(phase, float(SCENARIO["delta_n_primary"]))

    expected = c34.fabricate_height(
        phase, float(SCENARIO["delta_n_primary"]),
        psf_sigma_px=float(SCENARIO["psf_sigma_px"]),
        minimum_feature_radius_px=int(SCENARIO["minimum_feature_radius_px"]),
        height_levels=int(SCENARIO["height_levels"]),
        height_noise_nm=float(SCENARIO["height_noise_nm"]), noise_seed=11,
        edge_error_nm=20.0, maximum_height_um=float(SCENARIO["maximum_height_um"]))
    forward = process_height(
        initial_height, **_process_options(training=True, noise_seed=11, edge_bias_nm=20.0))
    forward_error = float((expected - forward).abs().max())
    derivative = _derivative_reference()

    images, labels, split = context["images"], context["labels"], context["split"]
    pilot_rows = runner._balanced_rows(split["train"], labels, 256)
    pilot_rows = pilot_rows[torch.randperm(
        len(pilot_rows), generator=torch.Generator().manual_seed(340503))]
    validation_rows = runner._balanced_rows(split["validation"], labels, 16)
    gradient_height = torch.nn.Parameter(initial_height.clone())
    gradient_opt = torch.optim.Adam([gradient_height], lr=0.05)
    gradient_opt.zero_grad(set_to_none=True)
    gradient_values = []
    for start in range(0, len(pilot_rows), args.microbatch):
        batch_rows = pilot_rows[start:start + args.microbatch]
        noise_seed = _step_noise_seed(901, 0, start)
        noise = _noise_field(gradient_height, noise_seed, float(SCENARIO["height_noise_nm"]))
        effective_phase = _process_phase_from_height(
            gradient_height, float(SCENARIO["delta_n_primary"]),
            **_process_options(training=True, noise_seed=noise_seed,
                               edge_bias_nm=_step_edge_bias(901, 0, start), noise_field=noise))
        fields = runner._fields_for_rows(context["field_cache"], batch_rows)
        result = c34.model_b_forward(fields, effective_phase, 2.0,
                                     lens_pitch_um=context["lens_pitch_um"],
                                     plan_cache=context["propagation_cache"], return_camera=False)
        loss = runner._c34_loss(result, labels[batch_rows].to(context["device"]),
                                effective_phase, float(context["throughput_floor"]))
        (loss * (len(batch_rows) / len(pilot_rows))).backward()
        gradient_values.append(float(loss.detach()))
    if gradient_height.grad is None or not torch.isfinite(gradient_height.grad).all():
        raise RuntimeError("process-aware 256-row production-path gradient preflight failed")
    gradient_norm_256 = float(gradient_height.grad.detach().norm())
    if gradient_norm_256 <= 0:
        raise RuntimeError("process-aware 256-row gradient is zero")

    true_height = torch.nn.Parameter(initial_height.clone())
    true_opt = torch.optim.Adam([true_height], lr=0.05)
    true_updates = []
    for step in range(3):
        rows = pilot_rows[step * args.update_batch:(step + 1) * args.update_batch]
        loss, norm = _process_update(true_height, true_opt, context, rows,
                                     seed=701, epoch=0, batch_index=step,
                                     microbatch=args.microbatch)
        true_updates.append({"loss": loss, "gradient_norm": norm,
                             "train_row_count": int(len(rows))})
    parameter_update = float((true_height.detach() - initial_height).norm())

    negative_rows = pilot_rows[:args.update_batch]
    negative_labels = labels.clone()
    permutation = torch.randperm(len(negative_rows), generator=torch.Generator().manual_seed(340502))
    negative_labels[negative_rows] = labels[negative_rows][permutation]
    negative_height = torch.nn.Parameter(initial_height.clone())
    negative_opt = torch.optim.Adam([negative_height], lr=0.05)
    negative_updates = []
    original_labels = context["labels"]
    context["labels"] = negative_labels
    try:
        for step in range(3):
            loss, norm = _process_update(negative_height, negative_opt, context,
                                         negative_rows, seed=801, epoch=0,
                                         batch_index=step, microbatch=args.microbatch)
            negative_updates.append({"loss": loss, "gradient_norm": norm})
    finally:
        context["labels"] = original_labels
    negative_eval = _evaluate_ensemble(context, validation_rows, negative_height.detach(),
                                       float(SCENARIO["delta_n_primary"]),
                                       batch_size=args.batch_size)
    negative_accuracy = float(negative_eval["mean_accuracy_all_samples"])
    no_phase = _evaluate_phase(context, validation_rows,
                               torch.zeros_like(initial_height), batch_size=args.batch_size)

    recovery_dir = output / "checkpoint_recovery"
    recovery_dir.mkdir(exist_ok=True)
    payload = _checkpoint_payload(true_height, true_opt, seed=701, epoch=3,
                                  generator=torch.Generator().manual_seed(701),
                                  history=true_updates, best_height=true_height.detach(),
                                  best_accuracy=0.0, best_epoch=3,
                                  train_rows=negative_rows, validation_rows=validation_rows)
    checkpoint = recovery_dir / "explicit_checkpoint.pt"
    _atomic_torch_save(payload, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rows_next = pilot_rows[3 * args.update_batch:4 * args.update_batch]
    continuation_height = torch.nn.Parameter(loaded["height"].to(context["device"]).clone())
    resumed_height = torch.nn.Parameter(loaded["height"].to(context["device"]).clone())
    continuation_opt = torch.optim.Adam([continuation_height], lr=0.05)
    resumed_opt = torch.optim.Adam([resumed_height], lr=0.05)
    continuation_opt.load_state_dict(copy.deepcopy(loaded["optimizer"]))
    resumed_opt.load_state_dict(copy.deepcopy(loaded["optimizer"]))
    continued_loss, _ = _process_update(continuation_height, continuation_opt, context,
                                        rows_next, seed=701, epoch=1, batch_index=0,
                                        microbatch=args.microbatch)
    resumed_loss, _ = _process_update(resumed_height, resumed_opt, context,
                                      rows_next, seed=701, epoch=1, batch_index=0,
                                      microbatch=args.microbatch)
    recovery_error = float((continuation_height.detach() - resumed_height.detach()).abs().max())
    recovery_loss_error = abs(continued_loss - resumed_loss)

    checks = {
        "hard_forward_matches_p4": forward_error <= 1e-6,
        "independent_smooth_operator_derivative": derivative["relative_error"] <= 1e-4,
        "full_path_256_row_gradient_finite_positive": math.isfinite(gradient_norm_256) and gradient_norm_256 > 0,
        "three_real_updates_finite_positive": all(math.isfinite(r["gradient_norm"]) and r["gradient_norm"] > 0
                                                    for r in true_updates),
        "parameter_changed": math.isfinite(parameter_update) and parameter_update > 0,
        "random_label_negative_control": negative_accuracy < 0.40,
        "checkpoint_resume_matches_continuation": recovery_error <= 1e-6 and recovery_loss_error <= 1e-6,
        "validation_and_baseline_outputs_finite": math.isfinite(float(no_phase["metrics"]["accuracy_all_samples"])),
    }
    report = {
        "identity": "C3.4_PROCESS_AWARE_TRAINING_PREFLIGHT",
        "session_source_id": c34.SESSION_SOURCE_ID,
        "production_call_chain": {
            "entrypoint": str(Path(__file__).resolve()),
            "wave_optics_runner": str(Path(runner.__file__).resolve()),
            "C34_model": str(Path(c34.__file__).resolve()),
            "C33_fabrication_operators": str(Path(c33.__file__).resolve())},
        "status": "PASS" if all(checks.values()) else "INVALID_HARNESS",
        "preflight_pass": all(checks.values()),
        "outer_gate": "PROCESS_AWARE_PREFLIGHT_PASS" if all(checks.values()) else "INVALID_HARNESS",
        "source": context["source"], "assumptions": dict(SCENARIO),
        "population": {"gradient_role": "train", "gradient_rows": pilot_rows.tolist(),
                       "negative_control_role": "train", "negative_validation_role": "validation",
                       "negative_validation_rows": validation_rows.tolist(),
                       "official_mnist_test_loaded": False, "holdout_opened": False},
        "initial_phase_height_branch": _phase_height_jump_summary(
            phase, float(SCENARIO["delta_n_primary"])),
        "hard_forward_max_abs_height_error_um": forward_error,
        "smooth_height_derivative_reference": derivative,
        "gradient_norm_256_rows": gradient_norm_256,
        "mean_loss_256_rows": float(np.mean(gradient_values)),
        "three_real_updates": true_updates,
        "height_parameter_l2_update_um": parameter_update,
        "random_label_negative_control": {"three_updates": negative_updates,
                                          "mean_validation_accuracy": negative_accuracy,
                                          "acceptance": "mean accuracy below 0.40 on 64 balanced validation rows",
                                          "passed": checks["random_label_negative_control"]},
        "no_phase_validation_reference": no_phase["metrics"],
        "checkpoint_recovery": {"checkpoint": str(checkpoint),
                                 "max_height_error_um": recovery_error,
                                 "loss_abs_error": recovery_loss_error},
        "checks": checks,
        "data_contract": "C3.4 train/validation only; no holdout or official MNIST test read",
        "resource_plan": {"logical_batch": args.update_batch, "microbatch": args.microbatch,
                          "evaluation_batch": args.batch_size,
                          "input_cache_path": str(args.field_cache_path.resolve()),
                          "cache_metadata": context["field_cache_metadata"]},
    }
    c34.write_json(report_path, report)
    return report


def _run_fast(context: dict[str, Any], args) -> dict[str, Any]:
    preflight_path = context["output_dir"] / "preflight" / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8")).get("preflight_pass") is not True:
        raise RuntimeError("process-aware fast gate requires a passing saved preflight")
    output = context["output_dir"] / "fast_1000"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "fast_training_summary.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite process-aware fast gate: {report_path}")
    initial = _bounded_design_height(context["initial_phase"], float(SCENARIO["delta_n_primary"]))
    height = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([height], lr=0.05)
    train_rows = runner._balanced_rows(context["split"]["train"], context["labels"], 250)
    validation_rows = runner._balanced_rows(context["split"]["validation"], context["labels"], 16)
    zero = _evaluate_phase(context, validation_rows, torch.zeros_like(initial), batch_size=args.batch_size)
    zero_efficiency = float(zero["metrics"]["mean_full_sensor_efficiency_vs_input"])
    history: list[dict[str, Any]] = []
    all_gradient_norms: list[float] = []
    for epoch in range(3):
        order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(34_510 + epoch))
        losses = []
        for batch_index, start in enumerate(range(0, len(order), args.update_batch)):
            rows = train_rows[order[start:start + args.update_batch]]
            loss, norm = _process_update(height, optimizer, context, rows,
                                         seed=11, epoch=epoch, batch_index=batch_index,
                                         microbatch=args.microbatch)
            losses.append(loss)
            all_gradient_norms.append(norm)
        validation = _evaluate_ensemble(context, validation_rows, height.detach(),
                                        float(SCENARIO["delta_n_primary"]),
                                        batch_size=args.batch_size)
        efficiency = validation["mean_full_sensor_efficiency_vs_input"]
        entry = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                 "validation_accuracy_mean": validation["mean_accuracy_all_samples"],
                 "validation_macro_f1_mean": validation["mean_macro_f1_classes_0_to_3"],
                 "mean_full_sensor_efficiency_vs_input": efficiency,
                 "efficiency_fraction_of_zero_phase": efficiency / max(zero_efficiency, 1e-30),
                 "gradient_norm_last": all_gradient_norms[-1]}
        history.append(entry)
        print(json.dumps({"stage": "process_fast", **entry}), flush=True)
    loss_decreased = history[-1]["train_loss"] < history[0]["train_loss"]
    gradients_valid = bool(all_gradient_norms) and all(math.isfinite(v) and v > 0 for v in all_gradient_norms)
    energy_retained = history[-1]["efficiency_fraction_of_zero_phase"] >= 0.1
    passed = bool(loss_decreased and gradients_valid and energy_retained)
    report = {"identity": "C3.4_PROCESS_AWARE_FAST_TRAINING_GATE",
              "session_source_id": c34.SESSION_SOURCE_ID,
              "status": "PASS" if passed else "VALID_FAILED",
              "fast_training_pass": passed,
              "outer_gate": "PROCESS_AWARE_FAST_TRAINING_PASS" if passed else "VALID_FAILED",
              "train_rows": train_rows.tolist(),
              "class_counts": [int((context["labels"][train_rows] == cls).sum()) for cls in range(4)],
              "validation_rows": validation_rows.tolist(),
              "process_assumptions": dict(SCENARIO),
              "zero_phase_validation_efficiency": zero_efficiency,
              "gate_checks": {"loss_decreased": loss_decreased,
                              "all_gradients_finite_positive": gradients_valid,
                              "final_efficiency_at_least_10_percent_of_zero_phase": energy_retained},
              "history": history,
              "resource_configuration": {"logical_batch": args.update_batch,
                                         "microbatch": args.microbatch,
                                         "evaluation_batch": args.batch_size,
                                         "field_cache": str(args.field_cache_path.resolve())},
              "official_mnist_test_loaded": False, "holdout_opened": False}
    c34.write_json(report_path, report)
    torch.save({"height_um": height.detach().cpu(), "optimizer": optimizer.state_dict(),
                "process_assumptions": dict(SCENARIO), "session_source_id": c34.SESSION_SOURCE_ID},
               output / "fast_height.pt")
    return report


def _write_height_stl(height: np.ndarray, directory: Path, name: str,
                      *, design_role: str) -> dict[str, Any]:
    mesh_directory = directory / name
    mesh_directory.mkdir(parents=True, exist_ok=True)
    height = np.asarray(height, dtype=np.float64)
    np.save(mesh_directory / f"{name}_height_um.npy", height)
    triangles, mesh_meta = c34.mesh_200um(height, substrate_um=1.2)
    stl_path = mesh_directory / f"{name}.stl"
    c34.write_binary_stl(stl_path, triangles)
    c34.write_json(mesh_directory / "stl_metadata.json", {
        "identity": "C3.4_PROCESS_AWARE_PARAMETERIZED_DIAGNOSTIC_STL",
        "unit": "um", "design_role": design_role,
        "wavelength_um": c34.WAVELENGTH_UM,
        "delta_n_training_assumption": SCENARIO["delta_n_primary"],
        "printer_and_material_identity": "UNKNOWN",
        "fabrication_ready": False,
        "surface_mesh": mesh_meta,
        "substrate_thickness_um": 1.2,
        "substrate_thickness_status": "geometric placeholder, not measured or printer specified"})
    audit = c34.audit_stl_independent(stl_path)
    recovered = c34.recover_height_from_mesh_vertices(stl_path)
    error = float(np.max(np.abs(recovered - height)))
    if (audit["triangles"] != mesh_meta["triangles"]
            or not audit["closed_two_manifold_by_edge_incidence"]
            or not audit["opposite_directed_edge_pairing"]
            or audit["declared_unit_status"] != "SIDECAR_DECLARES_UM"
            or error > 6e-6):
        raise RuntimeError(f"process-aware STL round-trip audit failed: {stl_path}")
    record = {"stl_path": str(stl_path), "design_role": design_role,
              "mesh_generation_metadata": mesh_meta, "independent_stl_audit": audit,
              "height_recovery_max_abs_error_um": error,
              "status": "PARAMETERIZED_DIAGNOSTIC_ONLY"}
    c34.write_json(mesh_directory / "stl_independent_audit.json", record)
    return {"record": record, "recovered_height": recovered}


def _run_formal(context: dict[str, Any], args, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    fast_path = context["output_dir"] / "fast_1000" / "fast_training_summary.json"
    if not fast_path.is_file() or json.loads(fast_path.read_text(encoding="utf-8")).get("fast_training_pass") is not True:
        raise RuntimeError("process-aware formal training requires a passing saved fast gate")
    formal_dir = context["output_dir"] / "formal"
    formal_dir.mkdir(parents=True, exist_ok=True)
    final_summary_path = formal_dir / "training_summary.json"
    if final_summary_path.exists():
        raise FileExistsError(f"refusing to overwrite completed process-aware training: {final_summary_path}")
    train_rows = context["split"]["train"]
    validation_rows = context["split"]["validation"]
    initial = _bounded_design_height(context["initial_phase"], float(SCENARIO["delta_n_primary"]))
    seed_results = []
    resume_payload = None
    resume_seed = None
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"explicit process-aware resume checkpoint is missing: {resume_checkpoint}")
        resume_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        resume_seed = int(resume_payload.get("seed", -1))
        if (resume_payload.get("identity") != "C3.4_PROCESS_AWARE_HEIGHT_CHECKPOINT"
                or resume_seed not in (1, 2, 3)
                or not 1 <= int(resume_payload.get("epoch", 0)) < args.epochs
                or not torch.equal(resume_payload["train_rows"].long(), train_rows.cpu())
                or not torch.equal(resume_payload["validation_rows"].long(), validation_rows.cpu())):
            raise ValueError("explicit process-aware resume state does not match this fixed training population")
        expected_path = formal_dir / f"seed_{resume_seed}" / "epoch_state.pt"
        if resume_checkpoint != expected_path.resolve():
            raise ValueError("resume must name the exact seed epoch_state.pt path")

    for seed in (1, 2, 3):
        seed_dir = formal_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        summary_path = seed_dir / "training_summary.json"
        best_path = seed_dir / "best_height.pt"
        if summary_path.is_file() and best_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            seed_results.append({"seed": seed, "best_epoch": int(summary["best_epoch"]),
                                 "validation_accuracy_mean": float(summary["validation_accuracy_mean"]),
                                 "height_path": str(best_path)})
            continue
        state_path = seed_dir / "epoch_state.pt"
        if state_path.exists() and not (resume_payload is not None and seed == resume_seed):
            raise RuntimeError(f"partial process-aware seed requires an explicit checkpoint: {state_path}")
        generator = torch.Generator().manual_seed(seed)
        if resume_payload is not None and seed == resume_seed:
            height = torch.nn.Parameter(resume_payload["height"].to(context["device"]).clone())
            optimizer = torch.optim.Adam([height], lr=0.05)
            optimizer.load_state_dict(resume_payload["optimizer"])
            generator.set_state(resume_payload["generator_state"])
            start_epoch = int(resume_payload["epoch"])
            history = list(resume_payload["history"])
            best_height = resume_payload["best_height"].float().cpu().clone()
            best_accuracy = float(resume_payload["best_accuracy"])
            best_epoch = int(resume_payload["best_epoch"])
        else:
            height = torch.nn.Parameter(initial.clone())
            optimizer = torch.optim.Adam([height], lr=0.05)
            start_epoch, history, best_accuracy, best_epoch = 0, [], -1.0, 0
            best_height = None
        for epoch in range(start_epoch, args.epochs):
            started = time.perf_counter()
            order = torch.randperm(len(train_rows), generator=generator)
            losses, norms = [], []
            for batch_index, start in enumerate(range(0, len(order), args.update_batch)):
                rows = train_rows[order[start:start + args.update_batch]]
                loss, norm = _process_update(height, optimizer, context, rows,
                                             seed=seed, epoch=epoch, batch_index=batch_index,
                                             microbatch=args.microbatch)
                losses.append(loss)
                norms.append(norm)
            validation = _evaluate_ensemble(context, validation_rows, height.detach(),
                                            float(SCENARIO["delta_n_primary"]),
                                            batch_size=args.batch_size)
            accuracy = float(validation["mean_accuracy_all_samples"])
            record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                      "validation_accuracy_mean": accuracy,
                      "validation_macro_f1_mean": validation["mean_macro_f1_classes_0_to_3"],
                      "validation_efficiency_mean": validation["mean_full_sensor_efficiency_vs_input"],
                      "validation_efficiency_sd_across_process_realizations": validation[
                          "sd_across_process_realizations_full_sensor_efficiency_vs_input"],
                      "process_realization_count": validation["process_realization_count"],
                      "gradient_norm_min": float(min(norms)),
                      "gradient_norm_max": float(max(norms)),
                      "elapsed_s": float(time.perf_counter() - started)}
            history.append(record)
            print(json.dumps({"stage": "process_formal", "seed": seed, **record}), flush=True)
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch + 1
                best_height = height.detach().cpu().clone()
            state = _checkpoint_payload(height, optimizer, seed=seed, epoch=epoch + 1,
                                        generator=generator, history=history,
                                        best_height=best_height, best_accuracy=best_accuracy,
                                        best_epoch=best_epoch, train_rows=train_rows,
                                        validation_rows=validation_rows)
            _atomic_torch_save(state, state_path)
            if epoch + 1 - best_epoch >= args.patience:
                break
        if best_height is None:
            raise RuntimeError(f"process-aware training produced no selected height for seed {seed}")
        torch.save({"height_um": best_height, "seed": seed, "best_epoch": best_epoch,
                    "validation_accuracy_mean": best_accuracy,
                    "process_assumptions": dict(SCENARIO),
                    "session_source_id": c34.SESSION_SOURCE_ID}, best_path)
        c34.write_json(summary_path, {"seed": seed, "best_epoch": best_epoch,
                      "validation_accuracy_mean": best_accuracy, "history": history,
                      "train_count": len(train_rows), "validation_count": len(validation_rows),
                      "holdout_opened": False, "official_mnist_test_loaded": False,
                      "trainable_only": "one 100x100 physical height map"})
        seed_results.append({"seed": seed, "best_epoch": best_epoch,
                             "validation_accuracy_mean": best_accuracy,
                             "height_path": str(best_path)})
        if resume_payload is not None and seed == resume_seed:
            resume_payload = None

    seed_results.sort(key=lambda x: (-x["validation_accuracy_mean"], x["seed"]))
    selected = seed_results[0]
    selected_payload = torch.load(selected["height_path"], map_location="cpu", weights_only=False)
    selected_height_cpu = selected_payload["height_um"].float().reshape(100, 100)
    maximum_height = min(float(SCENARIO["maximum_height_um"]),
                         c34.WAVELENGTH_UM / float(SCENARIO["delta_n_primary"]))
    if (not torch.isfinite(selected_height_cpu).all()
            or float(selected_height_cpu.min()) < 0
            or float(selected_height_cpu.max()) > maximum_height + 1e-6):
        raise RuntimeError("selected physical height map violates the k=0 bounded height interval")
    np.save(formal_dir / "selected_height_um.npy", selected_height_cpu.numpy())
    torch.save({"height_um": selected_height_cpu, "selection": selected,
                "process_assumptions": dict(SCENARIO),
                "session_source_id": c34.SESSION_SOURCE_ID}, formal_dir / "selected_height.pt")

    arrays: dict[str, np.ndarray] = {"validation_rows": validation_rows.numpy(),
                                     "validation_labels": context["labels"][validation_rows].numpy()}
    comparisons = []
    for delta_n in (float(SCENARIO["delta_n_primary"]), float(SCENARIO["delta_n_sensitivity"])):
        baseline_height = _bounded_design_height(context["initial_phase"], delta_n)
        baseline = _evaluate_ensemble(context, validation_rows, baseline_height, delta_n,
                                      batch_size=args.batch_size, save_arrays=arrays,
                                      prefix=f"p3_baseline_dn{delta_n:g}")
        trained = _evaluate_ensemble(context, validation_rows, selected_height_cpu.to(context["device"]),
                                     delta_n, batch_size=args.batch_size, save_arrays=arrays,
                                     prefix=f"process_trained_dn{delta_n:g}")
        ideal_phase = c34.phase_from_height(selected_height_cpu.to(context["device"]), delta_n)
        ideal = _evaluate_phase(context, validation_rows, ideal_phase, batch_size=args.batch_size)
        comparisons.append({"delta_n_assumption": delta_n,
                            "p3_geometry_phase_under_matched_hard_process": baseline,
                            "process_aware_height_under_hard_process": trained,
                            "process_aware_height_ideal_no_process": ideal["metrics"],
                            "trained_height_is_same_physical_map_across_delta_n": True,
                            "validation_role": "previously used for P3 model selection; reused diagnostic, not independent"})
    np.savez_compressed(formal_dir / "validation_process_comparison.npz", **arrays)

    stl_dir = formal_dir / "stl_parameterized_diagnostic"
    design_np = selected_height_cpu.numpy().astype(np.float64)
    nominal_tensor = process_height(
        selected_height_cpu.to(context["device"]), psf_sigma_px=float(SCENARIO["psf_sigma_px"]),
        minimum_feature_radius_px=int(SCENARIO["minimum_feature_radius_px"]),
        height_levels=int(SCENARIO["height_levels"]),
        maximum_height_um=float(SCENARIO["maximum_height_um"]), noise_nm=0.0,
        noise_seed=0, edge_height_bias_nm=0.0, straight_through_quantization=False)
    nominal_np = nominal_tensor.detach().cpu().numpy().astype(np.float64)
    design_stl = _write_height_stl(design_np, stl_dir, "process_aware_design", design_role="optimized pre-process height map")
    nominal_stl = _write_height_stl(nominal_np, stl_dir, "process_aware_nominal_after_psf_minfeature_quantization",
                                    design_role="noise-free nominal transformed height under assumed process")
    stl_roundtrip = []
    for delta_n in (float(SCENARIO["delta_n_primary"]), float(SCENARIO["delta_n_sensitivity"])):
        for name, artifact in (("design", design_stl), ("nominal_processed", nominal_stl)):
            recovered = torch.as_tensor(artifact["recovered_height"], dtype=torch.float32,
                                        device=context["device"])
            result = _evaluate_height(context, validation_rows, recovered, delta_n,
                                      noise_seed=0, edge_bias_nm=0.0,
                                      batch_size=args.batch_size, processed=False)
            stl_roundtrip.append({"delta_n_assumption": delta_n, "height_representation": name,
                                  "metrics": result["metrics"],
                                  "height_recovery_max_abs_error_um": artifact["record"][
                                      "height_recovery_max_abs_error_um"]})
    c34.write_json(formal_dir / "stl_roundtrip_validation.json", {
        "identity": "C3.4_PROCESS_AWARE_STL_ROUNDTRIP_VALIDATION",
        "status": "PARAMETERIZED_DIAGNOSTIC_ONLY",
        "population": {"role": "C3.4 validation reused after P3 selection",
                       "count": len(validation_rows),
                       "official_mnist_test_loaded": False, "holdout_opened": False},
        "stl_records": [design_stl["record"], nominal_stl["record"]],
        "readouts": stl_roundtrip,
        "limitations": ["No physical printer response or material index was measured.",
                        "Generic triangle-triangle intersection testing was not run.",
                        "Validation data were previously used for P3 selection and are not independent."]})

    summary = {"identity": PROCESS_NAME,
               "session_source_id": c34.SESSION_SOURCE_ID,
               "status": "ASSUMPTION_ONLY_DIAGNOSTIC_COMPLETE",
               "production_call_chain": {
                   "entrypoint": str(Path(__file__).resolve()),
                   "wave_optics_runner": str(Path(runner.__file__).resolve()),
                   "C34_model": str(Path(c34.__file__).resolve()),
                   "C33_fabrication_operators": str(Path(c33.__file__).resolve())},
               "source": context["source"],
               "geometry": "C3.4 P(80 mm)L(30 mm)P(100 mm), D=2 mm aperture assumption",
               "assumptions": dict(SCENARIO),
               "initialization": "selected C3.4 P3 phase converted to the wrapped k=0 physical-height branch at delta_n=0.5",
               "trainable_only": "one 100x100 physical height map; no digital head, input, geometry or ROI updates",
               "data_roles": {"train_count": len(train_rows), "validation_count": len(validation_rows),
                              "train_validation_disjoint": not bool(set(train_rows.tolist()) & set(validation_rows.tolist())),
                              "validation_used_for_selection": True, "holdout_opened": False,
                              "official_mnist_test_loaded": False},
               "selected_seed": selected,
               "seed_results_selected_by_assumption_validation_mean_accuracy": seed_results,
               "validation_comparisons": comparisons,
               "stl_roundtrip_validation_path": str(formal_dir / "stl_roundtrip_validation.json"),
               "validation_arrays_path": str(formal_dir / "validation_process_comparison.npz"),
               "resource_configuration": {"logical_batch": args.update_batch,
                                          "microbatch": args.microbatch,
                                          "evaluation_batch": args.batch_size,
                                          "input_field_cache_path": str(args.field_cache_path.resolve()),
                                          "input_cache_validated_against_production_preprocessing": True,
                                          "output_on_system_disk": True},
               "resource_telemetry_path": str(context["output_dir"] / "resource_telemetry.csv"),
               "fabrication_ready": False,
               "scientific_boundary": "numerical optimization under assumed process operators only; not printer or HW207 validation"}
    c34.write_json(final_summary_path, summary)
    return summary


def _proc_cpu_ticks(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/stat")
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    end = text.rfind(")")
    fields = text[end + 2:].split()
    if len(fields) < 13:
        return None
    return int(fields[11]) + int(fields[12])


def _system_cpu_ticks() -> tuple[int, int] | None:
    path = Path("/proc/stat")
    if not path.is_file():
        return None
    values = path.read_text(encoding="utf-8").splitlines()[0].split()[1:]
    ticks = [int(v) for v in values]
    return sum(ticks), ticks[3] + (ticks[4] if len(ticks) > 4 else 0)


def _parse_cpu_range_list(value: str) -> int:
    cpus: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError("invalid cgroup CPU range")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError("invalid cgroup CPU index")
            cpus.add(cpu)
    return len(cpus)


def _cpu_quota_cores() -> float:
    path = Path("/sys/fs/cgroup/cpu.max")
    if path.is_file():
        quota, period = path.read_text(encoding="utf-8").split()[:2]
        if quota != "max":
            return max(float(quota) / float(period), 0.1)
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.is_file() and period_path.is_file():
        quota = float(quota_path.read_text(encoding="utf-8").strip())
        period = float(period_path.read_text(encoding="utf-8").strip())
        if quota > 0 and period > 0:
            return max(quota / period, 0.1)
    for cpuset_path in (Path("/sys/fs/cgroup/cpuset.cpus.effective"),
                        Path("/sys/fs/cgroup/cpuset/cpuset.cpus")):
        if cpuset_path.is_file():
            count = _parse_cpu_range_list(cpuset_path.read_text(encoding="utf-8").strip())
            if count > 0:
                return float(count)
    if hasattr(os, "sched_getaffinity"):
        affinity_count = len(os.sched_getaffinity(0))
        if affinity_count > 0:
            return float(affinity_count)
    return float(os.cpu_count() or 1)


def _gpu_snapshot() -> tuple[str, str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], check=True, capture_output=True,
            text=True, timeout=5)
        values = [v.strip() for v in result.stdout.splitlines()[0].split(",")]
        return values[0], values[1], values[2]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "", "", ""


def _resource_sampler(output_dir: Path, stop: threading.Event) -> None:
    path = output_dir / "resource_telemetry.csv"
    fields = ["timestamp_unix", "process_cpu_percent_of_container_quota", "container_cpu_quota_cores",
              "host_cpu_busy_percent", "gpu_utilization_percent", "gpu_memory_used_mib",
              "gpu_memory_total_mib", "system_disk_free_bytes", "data_disk_free_bytes",
              "host_mem_available_bytes"]
    hz = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0
    quota = _cpu_quota_cores()
    previous_time = time.perf_counter()
    previous_process = _proc_cpu_ticks(os.getpid())
    previous_system = _system_cpu_ticks()
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(fields)
    while not stop.wait(30.0):
        now = time.perf_counter()
        current_process = _proc_cpu_ticks(os.getpid())
        current_system = _system_cpu_ticks()
        cpu_pct = ""
        system_pct = ""
        elapsed = now - previous_time
        if current_process is not None and previous_process is not None and elapsed > 0:
            cpu_pct = 100.0 * ((current_process - previous_process) / hz) / elapsed / quota
        if current_system and previous_system:
            total_delta = current_system[0] - previous_system[0]
            idle_delta = current_system[1] - previous_system[1]
            system_pct = 100.0 * (total_delta - idle_delta) / total_delta if total_delta > 0 else ""
        previous_time, previous_process, previous_system = now, current_process, current_system
        gpu_util, gpu_used, gpu_total = _gpu_snapshot()
        try:
            system_free = shutil.disk_usage("/").free
            data_free = shutil.disk_usage("/root/autodl-tmp").free
        except OSError:
            system_free, data_free = "", ""
        mem_available = ""
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            line = next((v for v in meminfo.read_text(encoding="utf-8").splitlines()
                         if v.startswith("MemAvailable:")), None)
            if line:
                mem_available = int(line.split()[1]) * 1024
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([time.time(), cpu_pct, quota, system_pct, gpu_util,
                                    gpu_used, gpu_total, system_free, data_free, mem_available])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "fast", "formal", "all"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="explicit existing C3.4 run containing P3 phase, preflight and fast gate")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new C3.4 process-aware output directory on AutoDL system disk")
    parser.add_argument("--field-cache-path", type=Path, required=True,
                        help="explicit fixed C3.4 complex input-field cache on system disk")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--update-batch", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch_size, args.update_batch, args.microbatch, args.epochs, args.patience) < 1:
        raise ValueError("batch sizes, epochs and patience must be positive")
    if args.microbatch > args.update_batch:
        raise ValueError("microbatch cannot exceed the fixed logical update batch")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    sampler = threading.Thread(target=_resource_sampler, args=(args.output_dir, stop), daemon=True)
    sampler.start()
    try:
        context = _make_context(args)
        if args.stage in ("preflight", "all"):
            result = _run_preflight(context, args)
            if not result["preflight_pass"]:
                raise RuntimeError("C3.4 process-aware preflight failed; formal training stopped")
            if args.stage == "preflight":
                print(json.dumps({"stage": args.stage, "status": result["status"]}), flush=True)
                return
        if args.stage in ("fast", "all"):
            result = _run_fast(context, args)
            if not result["fast_training_pass"]:
                raise RuntimeError("C3.4 process-aware fast gate failed; formal training stopped")
            if args.stage == "fast":
                print(json.dumps({"stage": args.stage, "status": result["status"]}), flush=True)
                return
        if args.stage in ("formal", "all"):
            result = _run_formal(context, args, resume_checkpoint=args.resume_checkpoint)
            print(json.dumps({"stage": "formal", "status": result["status"],
                              "selected_seed": result["selected_seed"]}), flush=True)
    finally:
        stop.set()
        sampler.join(timeout=3)


if __name__ == "__main__":
    main()
