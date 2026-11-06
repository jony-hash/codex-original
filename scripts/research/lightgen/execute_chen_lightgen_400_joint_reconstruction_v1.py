"""Run a fixed-population, six-phase LightGen400 reconstruction diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    ChenLightGen400Baseline,
    LightGen400Config,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    ResearchRunControl,
    ResearchRunIdentity,
    capture_execution_source_manifest,
)
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_json, atomic_write_new_json
from scripts.research.lightgen import diagnose_chen_lightgen_400_quickcheck_v1 as quickcheck
from scripts.research.lightgen import execute_chen_lightgen_400_mnist_baseline_v1 as baseline
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import (
    EPS_OP,
    EPS_S,
    fixed_prior,
    q_from_batch,
    truncated_normal_positive_kl,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_JOINT_RECONSTRUCTION_V1"
AUTHORITY = "USER_APPROVED_CHEN_LIGHTGEN_400_JOINT_RECONSTRUCTION_20261016"
GRID = 4096
REFERENCE_GRID = 6144
SEED = 2026101501
FIT_PER_CLASS = 10
CONFIRM_PER_CLASS = 20
LOGICAL_BATCH = 10
UPDATES = 500
CHECKPOINT_INTERVAL = 100
WINDOW_TOLERANCE = 1.0e-4
MSE_IMPROVEMENT_MIN = 0.10
SATURATION_MAX = 0.01
TASK_SIZE = 28


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _state_snapshot(model: ChenLightGen400Baseline) -> dict[str, torch.Tensor]:
    return baseline._state_snapshot(model)


def _make_model(source: ChenLightGen400Baseline, device: torch.device) -> ChenLightGen400Baseline:
    candidate = ChenLightGen400Baseline(LightGen400Config(GRID, True)).to(device)
    with torch.no_grad():
        candidate.encoder_phase.copy_(source.encoder_phase)
        candidate.decoder_phase.copy_(source.decoder_phase)
    candidate.requires_grad_(False)
    candidate.encoder_phase.requires_grad_(True)
    candidate.decoder_phase.requires_grad_(True)
    return candidate


def _optimizer(model: ChenLightGen400Baseline) -> torch.optim.Adam:
    return torch.optim.Adam(
        [model.encoder_phase, model.decoder_phase],
        lr=1.0e-3,
        betas=baseline.TRAIN_BETAS,
        eps=baseline.TRAIN_EPSILON,
        weight_decay=0.0,
    )


def _fixed_gain(
    model: ChenLightGen400Baseline,
    population: dict[str, torch.Tensor],
    pixels: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    rows = population["fit_rows"]
    model.eval()
    with torch.no_grad():
        for row_batch in rows.split(1):
            target, _ = baseline._image_batch(pixels, row_batch, device)
            physical = model.forward_from_image(target)["physical_w_per_pixel"].double()
            numerator += (physical.cpu() * target.double().cpu()).sum()
            denominator += physical.square().sum().cpu()
    _require(float(denominator) > 0.0, "fit-only camera gain denominator is zero")
    gain = float(numerator / denominator)
    _require(math.isfinite(gain) and gain > 0.0, "fit-only camera gain is invalid")
    return {"gain_w_inverse": gain, "numerator": float(numerator), "denominator": float(denominator),
            "fit_rows": int(rows.numel()), "estimator": "float64_global_least_squares_at_4096_start"}


def _task_outputs(result: dict[str, Any], gain: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    physical = result["physical_w_per_pixel"]
    task = physical * torch.as_tensor(gain, dtype=physical.dtype, device=device)
    return task, task.clamp(0.0, 1.0)


def _forward_rows(
    model: ChenLightGen400Baseline,
    rows: torch.Tensor,
    pixels: torch.Tensor,
    device: torch.device,
    *,
    grad: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target, _ = baseline._image_batch(pixels, rows, device)
    if grad:
        result = model.forward_from_image(target)
    else:
        with torch.no_grad():
            result = model.forward_from_image(target)
    return target, result


def _fit_indices(population: dict[str, torch.Tensor], column: int) -> torch.Tensor:
    _require(0 <= column < FIT_PER_CLASS, "fit-cycle cursor is outside the fixed population")
    return population["fit_rows_by_class"][:, column].long()


def _logical_update(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    rows: torch.Tensor,
    pixels: torch.Tensor,
    device: torch.device,
    gain: float,
    beta_rec: float,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    *,
    microbatch_size: int = 1,
) -> dict[str, float]:
    _require(rows.numel() == LOGICAL_BATCH, "logical batch must contain one row per MNIST class")
    _require(microbatch_size in (1, 2), "preflight microbatch must be one or two")
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    plane_powers: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []
    for start in range(0, LOGICAL_BATCH, microbatch_size):
        part_rows = rows[start : start + microbatch_size]
        target, result = _forward_rows(model, part_rows, pixels, device, grad=True)
        task, _ = _task_outputs(result, gain, device)
        per_micro_loss = (task - target).square().mean()
        _require(torch.isfinite(per_micro_loss).item(), "reconstruction loss is nonfinite")
        (beta_rec * per_micro_loss * (part_rows.numel() / LOGICAL_BATCH)).backward()
        losses.append(float(per_micro_loss.detach().cpu()))
        plane_powers.append(result["plane_powers_w"].detach())
        coefficients.append(result["ols_coefficients"].detach())
        del result, task, target
    gradients: dict[str, float] = {}
    for name, parameter in (("encoder", model.encoder_phase), ("decoder", model.decoder_phase)):
        gradient = parameter.grad
        _require(gradient is not None and torch.isfinite(gradient).all().item(),
                 f"{name} phase gradient is missing or nonfinite")
        norms = [float(torch.linalg.vector_norm(layer.double()).cpu()) for layer in gradient]
        _require(all(math.isfinite(value) and value > 0.0 for value in norms),
                 f"one or more {name} phase planes received no gradient")
        gradients[name] = float(torch.linalg.vector_norm(gradient.double()).cpu())
    encoder_before = model.encoder_phase.detach().clone()
    decoder_before = model.decoder_phase.detach().clone()
    optimizer.step()
    _require(torch.isfinite(model.encoder_phase).all().item()
             and torch.isfinite(model.decoder_phase).all().item(),
             "an optimizer update produced a nonfinite phase")
    _require(bool((model.encoder_phase - encoder_before).abs().max() > 0)
             and bool((model.decoder_phase - decoder_before).abs().max() > 0),
             "an intended phase group did not update")
    powers = torch.cat(plane_powers).double()
    # The power and KL terms are readouts only; source references remain fixed.
    z = torch.cat(coefficients)
    u = (z.real.double().square() + z.imag.double().square()) / (scale.double().to(device) + EPS_S)
    l_dist = truncated_normal_positive_kl(q_from_batch(u), fixed_prior(device=device)).mean()
    l_op = torch.log((powers + EPS_OP) /
                     (power_reference.double().to(device).unsqueeze(0) + EPS_OP)).square().mean()
    _require(torch.isfinite(l_dist).item() and torch.isfinite(l_op).item()
             and torch.isfinite(powers).all().item() and torch.isfinite(z).all().item(),
             "diagnostic KL, power, or latent readout is nonfinite")
    return {"beta_l_rec": beta_rec * sum(losses) / len(losses),
            "l_rec_microbatch_mean": sum(losses) / len(losses),
            "l_dist_readout": float(l_dist.cpu()), "l_op_readout": float(l_op.cpu()),
            "encoder_gradient_norm": gradients["encoder"],
            "decoder_gradient_norm": gradients["decoder"],
            "encoder_phase_max_update": float((model.encoder_phase - encoder_before).abs().max().cpu()),
            "decoder_phase_max_update": float((model.decoder_phase - decoder_before).abs().max().cpu()),
            "mean_seven_plane_power_w": powers.mean(0).cpu().tolist(),
            "latent_coefficients_finite": bool(torch.isfinite(z).all().item())}


def _center_difference_check(
    model: ChenLightGen400Baseline,
    row: int,
    pixels: torch.Tensor,
    device: torch.device,
    gain: float,
    beta_rec: float,
) -> dict[str, Any]:
    original_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    target, result = _forward_rows(model, torch.tensor([row]), pixels, device, grad=True)
    task, _ = _task_outputs(result, gain, device)
    loss = beta_rec * (task - target).square().mean()
    gradient = torch.autograd.grad(loss, (model.encoder_phase, model.decoder_phase))
    checks: list[dict[str, Any]] = []
    epsilon = 1.0e-2
    try:
        for group_name, parameter, grad_value in zip(("encoder", "decoder"),
                                                     (model.encoder_phase, model.decoder_phase), gradient,
                                                     strict=True):
            for layer in range(parameter.shape[0]):
                flat = int(grad_value[layer].abs().flatten().argmax())
                y, x = divmod(flat, parameter.shape[-1])
                autodiff = float(grad_value[layer, 0, y, x].detach().cpu())
                with torch.no_grad():
                    center = parameter[layer, 0, y, x].item()
                    parameter[layer, 0, y, x] = center + epsilon
                plus_target, plus_result = _forward_rows(model, torch.tensor([row]), pixels, device, grad=False)
                plus_task, _ = _task_outputs(plus_result, gain, device)
                plus = float((beta_rec * (plus_task - plus_target).square().mean()).cpu())
                with torch.no_grad():
                    parameter[layer, 0, y, x] = center - epsilon
                minus_target, minus_result = _forward_rows(model, torch.tensor([row]), pixels, device, grad=False)
                minus_task, _ = _task_outputs(minus_result, gain, device)
                minus = float((beta_rec * (minus_task - minus_target).square().mean()).cpu())
                with torch.no_grad():
                    parameter[layer, 0, y, x] = center
                base = float(loss.detach().cpu())
                descent_value = plus if autodiff < 0.0 else minus
                finite_difference = (plus - minus) / (2.0 * epsilon)
                relative_error = abs(autodiff - finite_difference) / max(
                    abs(autodiff), abs(finite_difference), 1.0e-12
                )
                _require(math.isfinite(autodiff) and math.isfinite(finite_difference)
                         and autodiff * finite_difference > 0.0 and relative_error <= 0.25,
                         f"{group_name} layer {layer} finite difference disagrees with autodiff")
                checks.append({"group": group_name, "layer": layer, "index_yx": [y, x],
                               "autodiff": autodiff, "central_difference": finite_difference,
                               "relative_error": relative_error, "direction_agrees": True,
                               "adam_descent": descent_value < base})
    finally:
        with torch.no_grad():
            model.encoder_phase.grad = None
            model.decoder_phase.grad = None
        model.train(original_training)
    _require(len(checks) == 6 and all(item["adam_descent"] for item in checks),
             "phase finite-difference probe failed a value or descent-direction check")
    return {"epsilon_rad": epsilon, "rows": [row], "checks": checks}


def _window_checks(
    model: ChenLightGen400Baseline,
    population: dict[str, torch.Tensor],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    roi_m = float(config["optics"]["common_complex_comparison_roi_m"])
    entries: list[dict[str, Any]] = []
    for label in range(10):
        for sample_index, row_tensor in enumerate(population["confirm_rows_by_class"][label, :2]):
            row = int(row_tensor)
            sample, _ = baseline._image_batch(pixels, torch.tensor([row]), device)
            _, details = baseline._window_reference(model, sample, [GRID, REFERENCE_GRID], roi_m,
                                                    include_sampling_reference=False)
            fields = details[GRID]["complex_relative_error_vs_largest_window"]
            maximum = max(fields.values())
            entries.append({"label": label, "confirm_index_within_class": sample_index, "row": row,
                            "max_complex_error_4096_vs_6144": maximum,
                            "per_plane_errors": fields, "passed": maximum <= WINDOW_TOLERANCE})
    return {"candidate_grids": [GRID, REFERENCE_GRID], "selected_grid": GRID, "roi_m": roi_m,
            "global_phase_alignment": "single_phase_for_all_selected_planes",
            "tolerance": WINDOW_TOLERANCE, "samples": entries,
            "passed": all(item["passed"] for item in entries)}


def _evaluate(
    model: ChenLightGen400Baseline,
    population: dict[str, torch.Tensor],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    evaluator: torch.nn.Module,
    gain: float,
    original_gain: float,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[dict[str, Any]]]:
    model.eval()
    evaluator.eval()
    records: list[dict[str, Any]] = []
    ports: list[dict[str, Any]] = []
    examples: dict[int, torch.Tensor] = {}
    predictions: list[int] = []
    with torch.no_grad():
        for row_tensor in population["all_rows"].split(1):
            row = int(row_tensor.item())
            target, result = _forward_rows(model, row_tensor, pixels, device, grad=False)
            task, display = _task_outputs(result, gain, device)
            original_task, _ = _task_outputs(result, original_gain, device)
            view, psnr, ssim = baseline._reconstruction_batch_metrics(task, target, row_tensor, row_tensor)
            logits = evaluator(view)
            prediction = int(logits.argmax(1)[0].cpu())
            predictions.append(prediction)
            p = (result["ols_coefficients"].real.double().square() +
                 result["ols_coefficients"].imag.double().square())[0].cpu()
            total = float(p.sum())
            effective = float(p.sum().square() / p.square().sum().clamp_min(1.0e-300))
            ols_input = float(result["ols_input_power_w"][0].cpu())
            roots = int((p >= 1.0e-9).sum())
            label = int(labels[row])
            position = int((population["all_rows"] == row_tensor).nonzero()[0, 0])
            population_name = "fit" if position < 100 else "confirmation"
            records.append({"row": row, "label": label, "population": population_name,
                            "mse400_unclipped": float((task - target).square().mean().cpu()),
                            "mse400_unclipped_original_gain": float((original_task - target).square().mean().cpu()),
                            "psnr28_db": float(psnr[0].cpu()), "ssim28": float(ssim[0].cpu()),
                            "saturation_fraction": float((((task < 0) | (task > 1)).float().mean()).cpu()),
                            "ed_prediction_descriptive": prediction, "total_ols_mode_power_w": total,
                            "weakest_port_power_w": float(p.min()), "ports_at_least_1nw": roots,
                            "n_eff": effective, "ols_input_power_w": ols_input,
                            "system_coupling_efficiency": total / max(ols_input, 1.0e-300)})
            for j, value in enumerate(p.tolist()):
                ports.append({"row": row, "label": label, "population": population_name,
                              "port": j, "row_index": j // 10, "column_index": j % 10,
                              "modal_power_w": value})
            if bool(population["confirm_rows_by_class"][label, :3].eq(row).any()):
                examples[row] = torch.stack((_task_view28(target), _task_view28(display))).cpu()
    requested = labels[population["all_rows"]].cpu()
    predicted = torch.tensor(predictions)
    summaries: dict[str, Any] = {}
    for group, indices in (("fit", range(100)), ("confirmation", range(100, 300))):
        subset = [records[index] for index in indices]
        summary = baseline._classification_summary(requested[list(indices)], predicted[list(indices)])
        summaries[group] = {
            "count": len(subset),
            "mse400_unclipped_mean": sum(item["mse400_unclipped"] for item in subset) / len(subset),
            "mse400_unclipped_original_gain_mean": sum(item["mse400_unclipped_original_gain"] for item in subset) / len(subset),
            "psnr28_db_mean": sum(item["psnr28_db"] for item in subset) / len(subset),
            "ssim28_mean": sum(item["ssim28"] for item in subset) / len(subset),
            "saturation_fraction": sum(item["saturation_fraction"] for item in subset) / len(subset),
            "ed_macro_recall_descriptive": summary["macro_recall"],
            "ed_recall_by_class_descriptive": summary["recall_by_class"].tolist(),
            "ed_confusion_matrix_descriptive": summary["confusion_matrix"].tolist(),
            "mean_total_ols_mode_power_w": sum(item["total_ols_mode_power_w"] for item in subset) / len(subset),
            "mean_ports_at_least_1nw": sum(item["ports_at_least_1nw"] for item in subset) / len(subset),
            "all_ports_at_least_1nw_count": sum(item["ports_at_least_1nw"] == 100 for item in subset),
            "mean_n_eff": sum(item["n_eff"] for item in subset) / len(subset),
            "mean_system_coupling_efficiency": sum(item["system_coupling_efficiency"] for item in subset) / len(subset),
        }
    return {"summary": summaries, "per_sample": records}, examples, ports


def _task_view28(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.adaptive_avg_pool2d(tensor, (TASK_SIZE, TASK_SIZE))[0, 0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_atlas(path: Path, pixels: torch.Tensor, labels: torch.Tensor,
                 population: dict[str, torch.Tensor], examples: dict[int, torch.Tensor]) -> None:
    selected = population["confirm_rows_by_class"][:, :3].flatten().tolist()
    cell, margin, label_width = 28, 4, 22
    canvas = Image.new("L", (label_width + 2 * (cell + margin) + margin,
                             margin + len(selected) * (cell + margin)), color=255)
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected):
        target, _ = baseline._image_batch(pixels, torch.tensor([row]), torch.device("cpu"))
        y = margin + index * (cell + margin)
        draw.text((1, y + 8), str(int(labels[row])), fill=0)
        for col, image in enumerate((_task_view28(target), examples[row][1])):
            tile = Image.fromarray((image.clamp(0, 1).numpy() * 255).round().astype("uint8"), mode="L")
            canvas.paste(tile, (label_width + margin + col * (cell + margin), y))
    canvas.save(path, format="PNG")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    result = report["local_gate"]
    by = result["by_population"]
    initial = by["initial_4096"]["confirmation"]
    endpoint = by["endpoint_500"]["confirmation"]
    md = (
        "# LightGen400 六层联合重建快速验证\n\n"
        "## 结果与状态\n\n"
        f"本次状态为 `{report['status']}`，固定更新 {report['training']['completed_updates']} 步。"
        f"确认集 MSE 从 {initial['mse400_unclipped_mean']:.6g} 变为 "
        f"{endpoint['mse400_unclipped_mean']:.6g}；SSIM 从 {initial['ssim28_mean']:.4f} 变为 "
        f"{endpoint['ssim28_mean']:.4f}，终点饱和比例 {endpoint['saturation_fraction']:.4%}。\n\n"
        "## 科学认识\n\n"
        f"预先固定的局部改善判定为 `{report['decision']}`。该结果只覆盖当前离散几何、固定 300 张 TRAIN 人口、"
        "单一 500 步联合重建目标和冻结 gain；不证明生成采样、硬件功率或架构容量上限。\n\n"
        "## 目标缺口\n\n"
        "仍缺少完整 DEV 重建与生成验收、重复性测量以及真实器件参数和硬件验证。\n\n"
        "## 工程完成与验证\n\n"
        f"4096/6144 窗口检查、六层 CUDA 更新、微批次等价、独立恢复与信号边界检查结果见机器报告和预检回执。"
        f"官方 TEST 读取数为 {report['official_test_reads']}。\n\n"
        "## 决策影响与下一步\n\n"
        f"本轮证据建议：`{report['decision']}`。该建议只支持另行设计下一轮，不自动增加步数或启动后继训练。\n"
    )
    path.write_text(md, encoding="utf-8")


def _cpu_snapshot(value: Any) -> Any:
    return baseline._cpu_snapshot(value)


def _set_all_grads(model: ChenLightGen400Baseline, enabled: bool) -> None:
    model.requires_grad_(False)
    model.encoder_phase.requires_grad_(enabled)
    model.decoder_phase.requires_grad_(enabled)


def _source_inputs(args: argparse.Namespace, device: torch.device):
    config_path, checkpoint_path, report_path = map(_resolve, (args.config, args.checkpoint, args.source_report))
    config, checkpoint, report, source_model, references = quickcheck._load_inputs(
        checkpoint_path, report_path, config_path, device
    )
    _require(args.fixed_population is not None, "this task requires the explicit fixed-population file")
    pixels, labels, train_indices, _ = baseline._load_data(_resolve(args.data))
    population = quickcheck._load_fixed_population(_resolve(args.fixed_population), labels, train_indices)
    population_validation = quickcheck._verify_population_against_source_encoder(
        source_model, population, pixels, device
    )
    model = _make_model(source_model, device)
    evaluator_path = (_resolve(args.evaluator) if args.evaluator else
                      _resolve(Path(config["evaluation"]["evaluator_checkpoint"])))
    evaluator = baseline._load_evaluator(evaluator_path, device)
    return (config, checkpoint, report, references, source_model, model, population,
            population_validation, pixels, labels, train_indices, evaluator, evaluator_path)


def _gradient_norm_and_microbatch_check(
    model: ChenLightGen400Baseline,
    rows: torch.Tensor,
    pixels: torch.Tensor,
    device: torch.device,
    gain: float,
    beta_rec: float,
) -> dict[str, Any]:
    state = _state_snapshot(model)
    gradients: dict[int, dict[str, torch.Tensor]] = {}
    values: dict[int, float] = {}
    for micro in (1, 2):
        model.zero_grad(set_to_none=True)
        total = 0.0
        for start in range(0, LOGICAL_BATCH, micro):
            part = rows[start : start + micro]
            target, result = _forward_rows(model, part, pixels, device, grad=True)
            task, _ = _task_outputs(result, gain, device)
            value = (task - target).square().mean()
            (value * (part.numel() / LOGICAL_BATCH)).backward()
            total += float(value.detach().cpu()) * part.numel() / LOGICAL_BATCH
        values[micro] = total
        gradients[micro] = {"encoder": model.encoder_phase.grad.detach().clone(),
                            "decoder": model.decoder_phase.grad.detach().clone()}
        model.zero_grad(set_to_none=True)
        model.load_state_dict(state, strict=True)
    errors: dict[str, float] = {}
    for group in ("encoder", "decoder"):
        a, b = gradients[1][group].double(), gradients[2][group].double()
        errors[group] = float(torch.linalg.vector_norm(a - b) /
                              torch.linalg.vector_norm(a).clamp_min(1.0e-30))
    loss_error = abs(values[1] - values[2]) / max(abs(values[1]), 1.0e-30)
    _require(loss_error <= 1.0e-5 and all(value <= 1.0e-4 for value in errors.values()),
             "microbatch 1/2 logical-batch values or gradients are not equivalent")
    return {"loss_relative_error": loss_error, "gradient_relative_l2_error": errors,
            "tolerance": {"loss": 1.0e-5, "gradient": 1.0e-4}}


def _negative_control(
    model: ChenLightGen400Baseline,
    rows: torch.Tensor,
    pixels: torch.Tensor,
    device: torch.device,
    gain: float,
) -> dict[str, Any]:
    target, result = _forward_rows(model, rows, pixels, device, grad=False)
    task, _ = _task_outputs(result, gain, device)
    return quickcheck._validate_pairing_negative_control(task, target, rows, torch.arange(10, device=device))


def _make_identity(config: dict[str, Any], manifest: Any, run_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=run_id,
        trajectory_id=f"{TASK_ID}:source_step_30000:seed_{SEED}",
        semantic_authority_id=f"{TASK_ID}:SOURCE_STEP_30000:GRID_{GRID}:UPDATES_{UPDATES}",
        data_role="canonical_MNIST_TRAIN_fixed_fit_100_confirmation_200_no_test_reads",
        config_identity={"source_config": config, "grid": GRID, "reference_grid": REFERENCE_GRID,
                         "updates": UPDATES, "fit_per_class": FIT_PER_CLASS,
                         "confirm_per_class": CONFIRM_PER_CLASS, "gain": "fit_only_float64_ls_frozen"},
        model_identity={"source_step": 30_000, "encoder": [2, 400, 400],
                        "ols": "gaussian_s12_v1_100_ports", "decoder": [4, 400, 400],
                        "grid": GRID, "trainable": ["encoder_phase", "decoder_phase"]},
        optimizer_identity={"name": "Adam", "learning_rate": 1.0e-3,
                            "betas": baseline.TRAIN_BETAS, "epsilon": baseline.TRAIN_EPSILON,
                            "reset_at_start": True, "objective": "beta_rec_times_raw_task_mse"},
        sampler_identity={"seed": SEED, "logical_batch": LOGICAL_BATCH,
                          "fixed_fit_rows_per_class": FIT_PER_CLASS, "updates": UPDATES},
        execution_source_semantics_id=manifest.execution_source_semantics_id,
        source_manifest_id=manifest.source_manifest_id,
    )


def _save_state(path: Path, model: ChenLightGen400Baseline, optimizer: torch.optim.Optimizer,
                completed: int, gain: float, population: dict[str, torch.Tensor],
                references: dict[str, torch.Tensor], preflight_dir: Path, run_dir: Path,
                latest: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": "chen_lightgen_400_joint_reconstruction_checkpoint.v1",
        "task_id": TASK_ID, "authorization_id": AUTHORITY,
        "completed_updates": completed, "next_batch_cursor": completed % FIT_PER_CLASS,
        "grid": GRID, "source_checkpoint_step": 30_000,
        "model": _state_snapshot(model), "optimizer": _cpu_snapshot(optimizer.state_dict()),
        "rng_cpu": torch.random.get_rng_state().cpu(),
        "rng_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "camera_gain_fit_only_w_inverse": gain,
        "frozen_references": {name: value.detach().cpu().clone() for name, value in references.items()},
        "fit_rows": population["fit_rows"].clone(), "confirm_rows": population["confirm_rows"].clone(),
        "preflight_dir": str(preflight_dir.resolve()), "run_dir": str(run_dir.resolve()),
        "latest": latest,
    }
    atomic_torch_save_new(path, payload)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _require(loaded["completed_updates"] == completed and loaded["next_batch_cursor"] == completed % FIT_PER_CLASS,
             "checkpoint readback step or next-batch cursor changed")
    _require(baseline._assert_state_close(payload["model"], loaded["model"], "joint_model_readback") == 0.0,
             "checkpoint model readback differs")
    _require(baseline._assert_state_close(payload["optimizer"], loaded["optimizer"], "joint_optimizer_readback") == 0.0,
             "checkpoint optimizer readback differs")
    return payload


def _retained_checkpoints(run_dir: Path) -> None:
    recoveries = sorted(run_dir.glob("checkpoint_step_*.pt"))
    keep = {run_dir / "checkpoint_step_000500.pt"}
    keep.update(recoveries[-2:])
    for candidate in recoveries:
        if candidate not in keep and candidate.name != "checkpoint_step_000500.pt":
            candidate.unlink()


def _load_checkpoint(path: Path, model: ChenLightGen400Baseline,
                     optimizer: torch.optim.Optimizer, preflight_dir: Path,
                     run_dir: Path, population: dict[str, torch.Tensor],
                     references: dict[str, torch.Tensor]) -> tuple[int, float, dict[str, Any]]:
    _require(path.is_file(), f"explicit resume checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "chen_lightgen_400_joint_reconstruction_checkpoint.v1"
             and payload.get("task_id") == TASK_ID and payload.get("authorization_id") == AUTHORITY,
             "resume checkpoint belongs to another task or schema")
    _require(payload.get("grid") == GRID and payload.get("source_checkpoint_step") == 30_000
             and payload.get("preflight_dir") == str(preflight_dir.resolve())
             and payload.get("run_dir") == str(run_dir.resolve()),
             "resume checkpoint grid, source step, or run paths differ")
    _require(torch.equal(payload["fit_rows"], population["fit_rows"])
             and torch.equal(payload["confirm_rows"], population["confirm_rows"]),
             "resume checkpoint sample rows differ")
    _require(all(torch.equal(payload["frozen_references"][key], references[key].cpu()) for key in references),
             "resume checkpoint frozen references differ")
    completed = int(payload["completed_updates"])
    _require(0 <= completed <= UPDATES and payload["next_batch_cursor"] == completed % FIT_PER_CLASS,
             "resume checkpoint update or batch cursor is invalid")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if completed:
        baseline._validate_adam_state(optimizer, completed_updates=completed)
    torch.random.set_rng_state(payload["rng_cpu"])
    torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return completed, float(payload["camera_gain_fit_only_w_inverse"]), payload


def _update_from_payload(payload_path: Path, output_path: Path, ready_path: Path | None,
                         requested_signal: int | None) -> None:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model = ChenLightGen400Baseline(LightGen400Config(GRID, True)).to(device)
    _set_all_grads(model, True)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = _optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    torch.random.set_rng_state(payload["rng_cpu"])
    torch.cuda.set_rng_state_all(payload["rng_cuda"])
    signaled: list[int] = []
    old_handlers: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        signaled.append(signum)

    if requested_signal is not None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, handler)
        atomic_write_new_json(ready_path, {"ready": True, "completed_updates": payload["completed_updates"]})
        while not signaled:
            time.sleep(0.01)
    rows = payload["rows"].long()
    pixels = payload["pixels"]
    latest = _logical_update(model, optimizer, rows, pixels, device,
                             float(payload["gain"]), float(payload["beta_rec"]),
                             payload["scale"].to(device), payload["power_reference"].to(device),
                             microbatch_size=1)
    completed = int(payload["completed_updates"]) + 1
    state = {
        "model": _state_snapshot(model), "optimizer": _cpu_snapshot(optimizer.state_dict()),
        "rng_cpu": torch.random.get_rng_state().cpu(),
        "rng_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()],
        "completed_updates": completed, "next_batch_cursor": completed % FIT_PER_CLASS,
        "latest": latest, "requested_signal": requested_signal,
        "received_signals": signaled,
    }
    atomic_torch_save_new(output_path, state)
    for signum, old in old_handlers.items():
        signal.signal(signum, old)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_worker_probe(model: ChenLightGen400Baseline, optimizer: torch.optim.Optimizer,
                      pixels: torch.Tensor, rows: torch.Tensor, gain: float, beta: float,
                      scale: torch.Tensor, power_reference: torch.Tensor,
                      initial_updates: int, outdir: Path, label: str,
                      signal_number: int | None = None) -> dict[str, Any]:
    payload_path = outdir / f"worker_{label}_input.pt"
    output_path = outdir / f"worker_{label}_output.pt"
    ready_path = outdir / f"worker_{label}_ready.json"
    payload = {"model": _state_snapshot(model), "optimizer": _cpu_snapshot(optimizer.state_dict()),
               "rng_cpu": torch.random.get_rng_state().cpu(),
               "rng_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()],
               "completed_updates": initial_updates, "rows": rows.cpu(),
               "pixels": pixels, "gain": gain, "beta_rec": beta,
               "scale": scale.detach().cpu(), "power_reference": power_reference.detach().cpu()}
    atomic_torch_save_new(payload_path, payload)
    command = [sys.executable, "-m", "scripts.research.lightgen.execute_chen_lightgen_400_joint_reconstruction_v1",
               "--worker-payload", str(payload_path), "--worker-output", str(output_path)]
    if signal_number is not None:
        command.extend(("--worker-ready", str(ready_path), "--worker-signal", str(signal_number)))
    original_device = next(model.parameters()).device
    original_model = _state_snapshot(model)
    original_optimizer = _cpu_snapshot(optimizer.state_dict())
    model.to("cpu")
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.cpu()
    torch.cuda.synchronize(original_device)
    torch.cuda.empty_cache()
    try:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if signal_number is not None:
            deadline = time.monotonic() + 120.0
            while not ready_path.is_file():
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    raise RuntimeError(f"{label} recovery worker exited before signal boundary: {stdout[-500:]} {stderr[-1500:]}")
                _require(time.monotonic() < deadline, f"{label} recovery worker did not become ready")
                time.sleep(0.02)
            process.send_signal(signal_number)
        stdout, stderr = process.communicate()
        _require(process.returncode == 0, f"{label} recovery worker failed: {stdout[-500:]} {stderr[-2000:]}")
        observed = torch.load(output_path, map_location="cpu", weights_only=False)
    finally:
        model.to(original_device)
        model.load_state_dict(original_model, strict=True)
        optimizer.load_state_dict(original_optimizer)
        _require(baseline._assert_state_close(original_model, _state_snapshot(model),
                                              f"{label}_parent_state_restore") == 0.0,
                 f"{label} parent state changed during the independent worker")
    _require(observed["completed_updates"] == initial_updates + 1
             and observed["next_batch_cursor"] == (initial_updates + 1) % FIT_PER_CLASS,
             f"{label} worker duplicated or skipped an update")
    if signal_number is not None:
        _require(observed["requested_signal"] == signal_number
                 and observed["received_signals"] == [signal_number],
                 f"{label} worker did not record the requested signal")
    return observed


def _independent_recovery_checks(model: ChenLightGen400Baseline,
                                 optimizer: torch.optim.Optimizer,
                                 pixels: torch.Tensor, population: dict[str, torch.Tensor],
                                 gain: float, beta: float, scale: torch.Tensor,
                                 power_reference: torch.Tensor, outdir: Path) -> dict[str, Any]:
    initial_model = _state_snapshot(model)
    initial_optimizer = _cpu_snapshot(optimizer.state_dict())
    initial_cpu = torch.random.get_rng_state().cpu()
    initial_cuda = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    for step in range(3):
        rows = _fit_indices(population, step)
        _logical_update(model, optimizer, rows, pixels, torch.device("cuda"), gain, beta,
                        scale, power_reference, microbatch_size=1)
    step3_model = _state_snapshot(model)
    step3_optimizer = _cpu_snapshot(optimizer.state_dict())
    step3_cpu = torch.random.get_rng_state().cpu()
    step3_cuda = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    frozen_before_update = {name: value for name, value in step3_model.items()
                            if name not in {"encoder_phase", "decoder_phase"}}
    frozen_initial = {name: value for name, value in initial_model.items()
                      if name not in {"encoder_phase", "decoder_phase"}}
    _require(baseline._assert_state_close(frozen_initial, frozen_before_update,
                                          "frozen_optics_after_preflight_updates") == 0.0,
             "OLS or another frozen buffer changed during six-phase preflight updates")
    next_rows = _fit_indices(population, 3)
    continuous = _logical_update(model, optimizer, next_rows, pixels, torch.device("cuda"),
                                  gain, beta, scale, power_reference, microbatch_size=1)
    expected_model = _state_snapshot(model)
    expected_optimizer = _cpu_snapshot(optimizer.state_dict())
    expected_cpu = torch.random.get_rng_state().cpu()
    expected_cuda = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    model.load_state_dict(step3_model, strict=True)
    optimizer.load_state_dict(step3_optimizer)
    torch.random.set_rng_state(step3_cpu)
    torch.cuda.set_rng_state_all(step3_cuda)
    worker_model = _run_worker_probe(model, optimizer, pixels, next_rows, gain, beta,
                                     scale, power_reference, 3, outdir, "resume")
    model_error = baseline._assert_state_close(expected_model, worker_model["model"], "independent_worker_model")
    optimizer_error = baseline._assert_state_close(expected_optimizer, worker_model["optimizer"],
                                                    "independent_worker_optimizer")
    rng_cpu_error = torch.equal(expected_cpu, worker_model["rng_cpu"])
    rng_cuda_error = all(torch.equal(a, b) for a, b in
                         zip(expected_cuda, worker_model["rng_cuda"], strict=True))
    _require(rng_cpu_error and rng_cuda_error, "independent process changed CPU/CUDA RNG continuation")
    signal_records = []
    for name, signum in (("sigint", signal.SIGINT), ("sigterm", signal.SIGTERM)):
        torch.random.set_rng_state(step3_cpu)
        torch.cuda.set_rng_state_all(step3_cuda)
        model.load_state_dict(step3_model, strict=True)
        optimizer.load_state_dict(step3_optimizer)
        observed = _run_worker_probe(model, optimizer, pixels, next_rows, gain, beta,
                                     scale, power_reference, 3, outdir, name, signum)
        _require(baseline._assert_state_close(expected_model, observed["model"],
                                              f"{name}_signal_worker_model") == 0.0
                 and baseline._assert_state_close(expected_optimizer, observed["optimizer"],
                                                  f"{name}_signal_worker_optimizer") == 0.0,
                 f"{name} signal worker changed the resumed update")
        signal_records.append({"signal": name, "completed_updates": observed["completed_updates"],
                               "next_batch_cursor": observed["next_batch_cursor"],
                               "saved_after_current_update": True})
    model.load_state_dict(initial_model, strict=True)
    optimizer.load_state_dict(initial_optimizer)
    torch.random.set_rng_state(initial_cpu)
    torch.cuda.set_rng_state_all(initial_cuda)
    _require(baseline._assert_state_close(initial_model, _state_snapshot(model), "preflight_model_restore") == 0.0,
             "preflight did not restore original model state")
    _require(baseline._assert_state_close(initial_optimizer, _cpu_snapshot(optimizer.state_dict()),
                                          "preflight_optimizer_restore") == 0.0,
             "preflight did not restore original optimizer state")
    return {"three_cuda_updates": 3,
            "non_scientific_update_counts": {"three_step_preflight": 3,
                                              "continuous_recovery_reference": 1,
                                              "independent_resume_worker": 1,
                                              "sigint_worker": 1, "sigterm_worker": 1},
            "independent_process_next_step": worker_model["completed_updates"],
            "continuous_vs_restored_model_error": model_error,
            "continuous_vs_restored_optimizer_error": optimizer_error,
            "cpu_rng_continuation_equal": rng_cpu_error, "cuda_rng_continuation_equal": rng_cuda_error,
            "continuous_loss": continuous["l_rec_microbatch_mean"],
            "sigint_sigterm": signal_records,
            "rng_restored_after_preflight": torch.equal(initial_cpu, torch.random.get_rng_state())
            and all(torch.equal(a, b) for a, b in zip(initial_cuda, torch.cuda.get_rng_state_all(), strict=True)),
            "passed": True}


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for the LightGen400 joint update")
    output_dir = _resolve(args.output_dir)
    _require(not output_dir.exists(), f"refusing to overwrite preflight directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device("cuda")
    (config, checkpoint, source_report, references, source_model, model, population,
     population_validation, pixels, labels, train_indices, evaluator, evaluator_path) = _source_inputs(args, device)
    evaluator_state = _state_snapshot(evaluator) if hasattr(evaluator, "state_dict") else {}
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    initial_rng_cpu = torch.random.get_rng_state().cpu()
    initial_rng_cuda = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    gain = _fixed_gain(model, population, pixels, device)
    beta = float(checkpoint["bogt_weights"]["rec"])
    _require(math.isfinite(beta) and beta > 0.0, "source reconstruction weight is invalid")
    windows = _window_checks(model, population, pixels, labels, config, device)
    _require(windows["passed"], "4096 full-chain fields failed comparison against 6144")
    baseline_metrics, baseline_examples, baseline_ports = _evaluate(
        model, population, pixels, labels, evaluator, gain["gain_w_inverse"],
        float(references["camera_gain"]), device
    )
    first_rows = _fit_indices(population, 0)
    baseline._independent_physics_checks(model)
    microbatch = _gradient_norm_and_microbatch_check(
        model, first_rows, pixels, device, gain["gain_w_inverse"], beta
    )
    negative = _negative_control(model, first_rows, pixels, device, gain["gain_w_inverse"])
    gradient = _center_difference_check(model, int(first_rows[0]), pixels, device,
                                        gain["gain_w_inverse"], beta)
    evaluator_state_after = _state_snapshot(evaluator) if hasattr(evaluator, "state_dict") else {}
    _require(baseline._assert_state_close(evaluator_state, evaluator_state_after, "evaluator_frozen_preflight") == 0.0,
             "preflight changed frozen evaluator parameters")
    model_state = _state_snapshot(model)
    optimizer = _optimizer(model)
    recovery = _independent_recovery_checks(model, optimizer, pixels, population,
                                            gain["gain_w_inverse"], beta,
                                            references["scale"], references["power_reference"], output_dir)
    _require(recovery["passed"] and recovery["rng_restored_after_preflight"],
             "independent recovery or signal checks failed")
    _require(baseline._assert_state_close(model_state, _state_snapshot(model), "preflight_restore_after_recovery") == 0.0,
             "preflight model differs after restoring the starting state")
    peak_vram = torch.cuda.max_memory_reserved(device)
    manifest = capture_execution_source_manifest(
        ROOT, task_id=TASK_ID, semantic_authority_id=f"{TASK_ID}:STEP30000_GRID4096_500",
        relevant_paths=(Path(__file__), Path("scripts/research/lightgen/diagnose_chen_lightgen_400_quickcheck_v1.py"),
                        Path("scripts/research/lightgen/execute_chen_lightgen_400_mnist_baseline_v1.py"),
                        Path("src/ldm_project/v2/application/research/lightgen/chen_lightgen_400_baseline.py"),
                        Path("src/ldm_project/v2/application/research/lightgen/gaussian_fiber_baseline.py"),
                        Path("src/ldm_project/v2/optics/propagation.py"),
                        _resolve(args.config), _resolve(args.fixed_population)),
        config_identity={"task_id": TASK_ID, "grid": GRID, "updates": UPDATES,
                         "fit_rows": population["fit_rows"].tolist(),
                         "confirm_rows": population["confirm_rows"].tolist()},
    )
    run_id = uuid.uuid4().hex
    identity = _make_identity(config, manifest, run_id)
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    elapsed = time.perf_counter() - started
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "source_checkpoint_step": 30_000,
                    "fit_rows_are_canonical_train": True, "confirmation_rows_are_canonical_train": True,
                    "official_test_reads": 0, "all_six_phase_planes_updated": recovery["three_cuda_updates"] == 3,
                    "4096_vs_6144_all_six_layer_fields_passed": windows["passed"],
                    "microbatch_equivalence_passed": True, "center_difference_and_descent_passed": True,
                    "pairing_negative_control_passed": negative["passed"],
                    "independent_resume_sigint_sigterm_passed": recovery["passed"],
                    "frozen_evaluator_unchanged": True},
        resource={"passed": True, "cuda_device": torch.cuda.get_device_name(device),
                  "peak_reserved_vram_bytes": peak_vram, "preflight_seconds": elapsed},
        implementation={"passed": True, "engine": "cuda_400x400_six_phase_grid4096_microbatch1",
                        "selected_engine": "cuda_400x400_six_phase_grid4096_microbatch1",
                        "trainable_groups": ["encoder_phase", "decoder_phase"],
                        "frozen_groups": ["ols", "camera_gain", "references", "evaluator"]},
        selected_engine="cuda_400x400_six_phase_grid4096_microbatch1",
        resource_envelope={"cuda_device": torch.cuda.get_device_name(device),
                           "peak_reserved_vram_bytes": peak_vram},
    )
    control.mark_preflight_ready(receipt)
    preflight = {"schema": "chen_lightgen_400_joint_reconstruction_preflight.v1", "status": "READY",
                 "task_id": TASK_ID, "authorization_id": AUTHORITY, "run_identity": identity.to_dict(),
                 "receipt": receipt.to_dict(), "source_checkpoint": str(_resolve(args.checkpoint)),
                 "source_report": str(_resolve(args.source_report)),
                 "source_checkpoint_step": 30_000, "source_status": source_report["status"],
                 "source_config": str(_resolve(args.config)), "data": str(_resolve(args.data)),
                 "fixed_population": str(_resolve(args.fixed_population)), "evaluator": str(evaluator_path),
                 "data_roles": {"fit": 100, "confirmation": 200, "both_train": True, "test_reads": 0},
                 "training": {"grid": GRID, "logical_batch": LOGICAL_BATCH,
                              "microbatch": 1, "updates": UPDATES, "beta_rec": beta,
                              "optimizer": {"name": "Adam", "lr": 1.0e-3,
                                            "betas": baseline.TRAIN_BETAS,
                                            "eps": baseline.TRAIN_EPSILON}},
                 "fixed_population_rows": {"fit": population["fit_rows"].tolist(),
                                           "confirmation": population["confirm_rows"].tolist()},
                 "fixed_gain": gain, "original_gain_w_inverse": float(references["camera_gain"]),
                 "source_references": {key: value.cpu().tolist() for key, value in references.items()},
                 "population_source_validation": population_validation,
                 "window_checks": windows, "start_metrics": baseline_metrics["summary"],
                 "microbatch_equivalence": microbatch, "gradient_check": gradient,
                 "negative_control": negative, "recovery": recovery,
                 "peak_reserved_vram_bytes": peak_vram, "elapsed_seconds": elapsed,
                 "official_test_reads": 0}
    atomic_write_new_json(output_dir / "preflight.json", baseline._json_safe(preflight))
    atomic_write_new_json(output_dir / "preflight_ready.json", baseline._json_safe(receipt.to_dict()))
    atomic_torch_save_new(output_dir / "fixed_population.pt", population)
    atomic_torch_save_new(output_dir / "start_model_state.pt", model_state)
    atomic_torch_save_new(output_dir / "start_rng.pt", {"cpu": initial_rng_cpu, "cuda": initial_rng_cuda})
    _write_csv(output_dir / "start_per_sample_metrics.csv", baseline_metrics["per_sample"])
    _write_csv(output_dir / "start_per_port_power.csv", baseline_ports)
    _write_atlas(output_dir / "start_atlas.png", pixels, labels, population, baseline_examples)
    return preflight


def _read_ready(preflight_dir: Path, args: argparse.Namespace):
    preflight_dir = _resolve(preflight_dir)
    preflight = _load_json(preflight_dir / "preflight.json")
    ready = _load_json(preflight_dir / "preflight_ready.json")
    _require(preflight.get("status") == "READY" and preflight.get("task_id") == TASK_ID
             and preflight.get("authorization_id") == AUTHORITY,
             "an explicit READY preflight for this task is required")
    _require(preflight["receipt"] == ready and ready.get("authorization_id") == AUTHORITY
             and ready.get("status") == "READY", "READY receipt does not match preflight evidence")
    _require(preflight["source_checkpoint"] == str(_resolve(args.checkpoint))
             and preflight["source_report"] == str(_resolve(args.source_report))
             and preflight["fixed_population"] == str(_resolve(args.fixed_population))
             and preflight["source_config"] == str(_resolve(args.config))
             and preflight["data"] == str(_resolve(args.data)),
             "run inputs differ from explicit READY preflight inputs")
    if args.evaluator is not None:
        _require(preflight["evaluator"] == str(_resolve(args.evaluator)),
                 "run evaluator differs from READY preflight evaluator")
    return preflight, ready


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for the LightGen400 joint update")
    run_dir = _resolve(args.output_dir)
    preflight_dir = _resolve(args.preflight_dir)
    preflight, receipt = _read_ready(preflight_dir, args)
    if args.mode == "run":
        _require(not run_dir.exists(), f"refusing to overwrite run directory: {run_dir}")
        _require(args.resume_checkpoint is None, "fresh run cannot consume a recovery checkpoint")
    else:
        _require(run_dir.is_dir() and args.resume_checkpoint is not None,
                 "resume requires an existing run directory and explicit checkpoint")
    device = torch.device("cuda")
    (config, checkpoint, source_report, references, source_model, model, population,
     population_validation, pixels, labels, train_indices, evaluator, evaluator_path) = _source_inputs(args, device)
    saved_population = torch.load(preflight_dir / "fixed_population.pt", map_location="cpu", weights_only=False)
    _require(baseline._assert_state_close(population, saved_population, "run_fixed_population") == 0.0,
             "explicit fixed population differs from READY preflight population")
    model.load_state_dict(torch.load(preflight_dir / "start_model_state.pt", map_location="cpu", weights_only=False),
                          strict=True)
    original_gain = float(references["camera_gain"].cpu())
    gain = float(preflight["fixed_gain"]["gain_w_inverse"])
    beta = float(preflight["training"]["beta_rec"])
    references = {key: value.to(device) for key, value in references.items()}
    optimizer = _optimizer(model)
    identity = ResearchRunIdentity.from_mapping(preflight["run_identity"])
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    if args.mode == "run":
        attempt_id = run_dir.name
        control.open_scientific_attempt(run_dir, attempt_id=attempt_id, receipt=receipt)
        expected_manifest = {"schema": "chen_lightgen_400_joint_reconstruction_run.v1",
                             "task_id": TASK_ID, "run_id": identity.run_id,
                             "attempt_id": attempt_id,
                             "source_checkpoint": str(_resolve(args.checkpoint)),
                             "source_report": str(_resolve(args.source_report)),
                             "preflight_dir": str(preflight_dir),
                             "fixed_population": str(_resolve(args.fixed_population)),
                             "source_config": str(_resolve(args.config)),
                             "grid": GRID, "updates": UPDATES,
                             "evaluator": str(evaluator_path)}
        atomic_write_new_json(run_dir / "run_manifest.json", expected_manifest)
        _append_jsonl(run_dir / "segments.jsonl", {"event": "run_started", "time": time.time()})
        atomic_torch_save_new(run_dir / "fixed_population.pt", population)
        start = 0
        latest: dict[str, Any] = {}
        rng = torch.load(preflight_dir / "start_rng.pt", map_location="cpu", weights_only=False)
        torch.random.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state_all(rng["cuda"])
    else:
        expected_manifest = _load_json(run_dir / "run_manifest.json")
        _require(expected_manifest.get("task_id") == TASK_ID and expected_manifest.get("run_id") == identity.run_id
                 and expected_manifest.get("grid") == GRID and expected_manifest.get("updates") == UPDATES,
                 "existing run manifest differs from READY task")
        _require(expected_manifest.get("source_checkpoint") == str(_resolve(args.checkpoint))
                 and expected_manifest.get("source_report") == str(_resolve(args.source_report))
                 and expected_manifest.get("preflight_dir") == str(preflight_dir)
                 and expected_manifest.get("fixed_population") == str(_resolve(args.fixed_population))
                 and expected_manifest.get("source_config") == str(_resolve(args.config))
                 and expected_manifest.get("evaluator") == str(evaluator_path),
                 "resume inputs differ from the original run manifest")
        _require(expected_manifest.get("attempt_id") == run_dir.name,
                 "resume output directory does not match the original attempt identity")
        control.resume_scientific_attempt(run_dir, attempt_id=str(expected_manifest["attempt_id"]), receipt=receipt)
        start, gain, recovery = _load_checkpoint(_resolve(args.resume_checkpoint), model, optimizer,
                                                 preflight_dir, run_dir, population, references)
        _require(math.isclose(gain, float(preflight["fixed_gain"]["gain_w_inverse"]),
                              rel_tol=1.0e-12, abs_tol=0.0),
                 "resume checkpoint fit-only gain differs from READY preflight")
        latest = recovery.get("latest", {})
    evaluator_before = _state_snapshot(evaluator)
    start_time = time.perf_counter()
    last_heartbeat = start_time
    interrupted: list[int] = []
    previous_handlers: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        interrupted.append(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, handler)
    try:
        if start < UPDATES:
            for step0 in range(start, UPDATES):
                rows = _fit_indices(population, step0 % FIT_PER_CLASS)
                latest = _logical_update(model, optimizer, rows, pixels, device, gain, beta,
                                         references["scale"], references["power_reference"],
                                         microbatch_size=1)
                completed = step0 + 1
                now = time.perf_counter()
                if completed % 100 == 0:
                    record = {"step": completed, "rows": rows.tolist(), **latest,
                              "elapsed_seconds": now - start_time,
                              "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)}
                    with (run_dir / "training_history.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress = {"status": "RUNNING", "completed_updates": completed,
                                "next_batch_cursor": completed % FIT_PER_CLASS,
                                "elapsed_seconds": now - start_time}
                    atomic_write_json(run_dir / "progress.json", progress)
                if completed % CHECKPOINT_INTERVAL == 0 or interrupted:
                    filename = f"checkpoint_step_{completed:06d}.pt"
                    _save_state(run_dir / filename, model, optimizer, completed, gain, population,
                                references, preflight_dir, run_dir, latest)
                    if completed == UPDATES:
                        final = run_dir / "checkpoint_step_000500.pt"
                        if not final.exists():
                            _save_state(final, model, optimizer, completed, gain, population,
                                        references, preflight_dir, run_dir, latest)
                    _retained_checkpoints(run_dir)
                if now - last_heartbeat >= 60.0:
                    atomic_write_json(run_dir / "heartbeat.json", {"time": time.time(), "step": completed})
                    last_heartbeat = now
                if interrupted:
                    atomic_write_json(run_dir / "progress.json", {"status": "INTERRUPTED",
                                            "completed_updates": completed,
                                            "resume_checkpoint": f"checkpoint_step_{completed:06d}.pt",
                                            "signal": interrupted[-1]})
                    _append_jsonl(run_dir / "segments.jsonl", {"event": "interrupted",
                                 "completed_updates": completed, "signal": interrupted[-1],
                                 "time": time.time()})
                    control.interrupted()
                    return {"status": "INTERRUPTED", "completed_updates": completed}
        _require((run_dir / "checkpoint_step_000500.pt").is_file(),
                 "terminal 500-step checkpoint is missing")
        final_payload = torch.load(run_dir / "checkpoint_step_000500.pt", map_location="cpu", weights_only=False)
        _require(final_payload["completed_updates"] == UPDATES, "terminal checkpoint step is not 500")
        baseline._assert_state_close(final_payload["model"], _state_snapshot(model), "terminal_model_reload")
        baseline._assert_state_close(final_payload["optimizer"], _cpu_snapshot(optimizer.state_dict()),
                                     "terminal_optimizer_reload")
    finally:
        for sig, old in previous_handlers.items():
            signal.signal(sig, old)
    _require(baseline._assert_state_close(evaluator_before, _state_snapshot(evaluator), "frozen_evaluator_run") == 0.0,
             "training changed the frozen E_D evaluator")
    terminal_window = _window_checks(model, population, pixels, labels, config, device)
    _require(terminal_window["passed"], "trained 4096 fields failed terminal 6144 comparison")
    initial_model = _make_model(source_model, device)
    initial_model.load_state_dict(torch.load(preflight_dir / "start_model_state.pt", map_location="cpu", weights_only=False),
                                  strict=True)
    initial_metrics, initial_examples, initial_ports = _evaluate(
        initial_model, population, pixels, labels, evaluator, gain, original_gain, device
    )
    terminal_metrics, terminal_examples, terminal_ports = _evaluate(
        model, population, pixels, labels, evaluator, gain, original_gain, device
    )
    _write_csv(run_dir / "per_sample_metrics.csv", [
        {"state": state, **row}
        for state, block in (("initial_4096", initial_metrics), ("endpoint_500", terminal_metrics))
        for row in block["per_sample"]
    ])
    _write_csv(run_dir / "per_port_power.csv", [
        {"state": state, **row}
        for state, block in (("initial_4096", initial_ports), ("endpoint_500", terminal_ports))
        for row in block
    ])
    _write_atlas(run_dir / "initial_atlas.png", pixels, labels, population, initial_examples)
    _write_atlas(run_dir / "endpoint_atlas.png", pixels, labels, population, terminal_examples)
    initial_confirm = initial_metrics["summary"]["confirmation"]
    endpoint_confirm = terminal_metrics["summary"]["confirmation"]
    improvement = 1.0 - endpoint_confirm["mse400_unclipped_mean"] / initial_confirm["mse400_unclipped_mean"]
    gate_passed = (improvement >= MSE_IMPROVEMENT_MIN
                   and endpoint_confirm["ssim28_mean"] >= initial_confirm["ssim28_mean"]
                   and endpoint_confirm["saturation_fraction"] <= SATURATION_MAX)
    report = {"schema": "chen_lightgen_400_joint_reconstruction_report.v1",
              "task_id": TASK_ID, "authorization_id": AUTHORITY,
              "status": "VALID_COMPLETED" if gate_passed else "VALID_FAILED",
              "source_checkpoint_step": 30_000, "source_baseline_status": source_report["status"],
              "data_roles": {"fit": 100, "confirmation": 200, "canonical_train": True,
                             "dev_reads": 0, "official_test_reads": 0},
              "units": {"physical": "W/pixel", "task": "dimensionless after frozen scalar gain",
                        "display": "task output clipped to [0,1]"},
              "fixed_gain": gain, "original_gain_w_inverse": original_gain,
              "training": {"completed_updates": UPDATES, "latest": latest,
                           "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
                           "elapsed_seconds": time.perf_counter() - start_time},
              "window_checks_terminal": terminal_window,
              "local_gate": {"passed": gate_passed,
                             "criteria": {"confirm_mse_improvement_min": MSE_IMPROVEMENT_MIN,
                                          "ssim_not_lower": True, "saturation_max": SATURATION_MAX},
                             "confirm_mse_improvement_fraction": improvement,
                             "confirm_ssim_delta": endpoint_confirm["ssim28_mean"] - initial_confirm["ssim28_mean"],
                             "by_population": {"initial_4096": initial_metrics["summary"],
                                               "endpoint_500": terminal_metrics["summary"]}},
              "decision": "SIX_LAYER_JOINT_RECONSTRUCTION_LOCAL_IMPROVEMENT_SUPPORTED" if gate_passed
                          else "FIXED_SIX_LAYER_500_STEP_INTERVENTION_NOT_SUPPORTED",
              "official_test_reads": 0}
    atomic_write_new_json(run_dir / "report.json", baseline._json_safe(report))
    _write_report(run_dir / "results_analysis.md", report)
    atomic_write_json(run_dir / "progress.json", {"status": report["status"],
                        "completed_updates": UPDATES, "evaluation_complete": True})
    _append_jsonl(run_dir / "segments.jsonl", {"event": "completed", "time": time.time(),
                 "decision": report["decision"]})
    if gate_passed:
        control.complete()
    else:
        control.valid_failed()
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "resume"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--fixed-population", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/research/lightgen/chen_lightgen_400_mnist_baseline_v1.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/MNIST/processed/training.pt"))
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--worker-payload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ready", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-signal", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_payload is not None:
        _require(args.worker_output is not None, "worker output path is required")
        return args
    _require(args.mode is not None and args.checkpoint is not None and args.source_report is not None
             and args.fixed_population is not None and args.output_dir is not None,
             "explicit mode, checkpoint, report, fixed population and output directory are required")
    _require(args.resume_checkpoint is None if args.mode != "resume" else args.resume_checkpoint is not None,
             "resume mode requires exactly one explicit resume checkpoint")
    _require(args.preflight_dir is not None if args.mode in ("run", "resume") else args.preflight_dir is None,
             "run/resume require --preflight-dir; preflight must not consume a READY receipt")
    return args


def main() -> None:
    args = _parse_args()
    if args.worker_payload is not None:
        _update_from_payload(_resolve(args.worker_payload), _resolve(args.worker_output),
                             _resolve(args.worker_ready) if args.worker_ready else None,
                             args.worker_signal)
        return
    try:
        result = _preflight(args) if args.mode == "preflight" else _run(args)
        print(json.dumps({"status": result.get("status"), "task_id": TASK_ID,
                          "output_dir": str(_resolve(args.output_dir))}, ensure_ascii=False))
    except BaseException as error:
        output_dir = _resolve(args.output_dir)
        if output_dir.is_dir() and args.mode == "preflight" and not (output_dir / "preflight_failure.json").exists():
            atomic_write_new_json(output_dir / "preflight_failure.json", {
                "status": "INVALID_HARNESS", "task_id": TASK_ID,
                "error_type": type(error).__name__, "error": str(error),
                "completed_scientific_updates": 0, "official_test_reads": 0})
        elif output_dir.is_dir() and args.mode in ("run", "resume") and not (output_dir / "report.json").exists():
            with (output_dir / "execution_errors.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"status": "INVALID_HARNESS", "task_id": TASK_ID,
                                         "error_type": type(error).__name__, "error": str(error),
                                         "resume_checkpoint_required": True}, ensure_ascii=False) + "\n")
        raise


if __name__ == "__main__":
    main()
