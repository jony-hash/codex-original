"""Verify a frozen A4700 foreground-objective direction without updating the model."""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from scripts.research.lightgen import audit_lightgen_2d_generator_cross_port_joint_structure_v1 as frozen
from scripts.research.lightgen import audit_lightgen_2d_generator_foreground_shape_localization_v1 as foreground
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as parent_task
from scripts.research.lightgen import execute_lightgen_2d_generator_frozen_conditional_ols_generation_v1 as generation
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import execute_lightgen_2d_generator_semantic_target_comparison_v1 as semantic
from scripts.research.lightgen import execute_lightgen_2d_generator_stroke_fidelity_comparison_v1 as stroke


TASK_ID = "M0_GENERATOR_FOREGROUND_OBJECTIVE_GRADIENT_V1"
SOURCE_STEP = 4_700
FIT_ROWS = 200
CONFIRM_ROWS = 300
TOTAL_ROWS = FIT_ROWS + CONFIRM_ROWS
BATCH_SIZE = 10
FOCUS_CLASSES = (2, 4)
EPSILON = 1.0e-3
FD_RTOL = 0.1
FD_ATOL = 5.0e-7
REPLAY_RTOL = 2.0e-6
REPLAY_ATOL = 5.0e-8
GRADIENT_RTOL = 2.0e-5
GRADIENT_ATOL = 2.0e-7
DECISION_ATOL = 2.0e-7
DECISION_RTOL = 2.0e-5
CHUNK_SIZE = 50
SCENARIOS = ("baseline", "base_plus", "base_minus", "candidate_plus", "candidate_minus")
_NVRTC_DLL_HANDLE: Any | None = None
ROW_FIELDS = (
    "scenario",
    "position",
    "source_row_index",
    "label",
    "role",
    "q_fg",
    "q_bg",
    "q_raw200",
    "raw_mse",
    "raw_mse_fg",
    "raw_mse_bg",
    "ssim_28",
    "integral_intensity",
    "clipped_fraction",
    "e_d_prediction",
    "e_d_target_logit",
)


class GradientAuditError(RuntimeError):
    """Invalid input, a failed production invariant, or incomplete evidence."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise GradientAuditError(message)


def _runtime_dll_probe() -> dict[str, Any]:
    """Load the supported environment's NVRTC builtins in this process on Windows."""
    global _NVRTC_DLL_HANDLE
    if os.name != "nt":
        return {"status": "not_applicable", "scope": "current_process"}
    build = torch.version.cuda
    _require(isinstance(build, str) and len(build.split(".")) == 2 and all(part.isdigit() for part in build.split(".")), "PyTorch CUDA build cannot identify the NVRTC DLL")
    major, minor = build.split(".")
    environment_bin = (Path(sys.prefix) / "bin").resolve()
    dll_path = environment_bin / f"nvrtc-builtins64_{major}{minor}.dll"
    _require(dll_path.is_file(), f"the supported environment is missing {dll_path.name}")
    current_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    normalized = os.path.normcase(os.path.normpath(str(environment_bin)))
    present = any(os.path.normcase(os.path.normpath(entry)) == normalized for entry in current_entries)
    if not present:
        os.environ["PATH"] = str(environment_bin) + (os.pathsep + os.environ.get("PATH", "") if os.environ.get("PATH") else "")
    try:
        loader = getattr(ctypes, "WinDLL", None)
        _require(loader is not None, "Windows DLL loader is unavailable")
        _NVRTC_DLL_HANDLE = loader(str(dll_path))
    except OSError as error:
        raise GradientAuditError(f"CUDA NVRTC DLL load failed in the supported process environment: {error}") from error
    return {"status": "already_present" if present else "prepended", "scope": "current_process_only", "cuda_build": build, "environment_bin": str(environment_bin), "dll": dll_path.name}


def _cuda_backend_probe(device: torch.device) -> dict[str, Any]:
    complex_values = torch.zeros(16, dtype=torch.complex64, device=device)
    exponential = torch.exp(complex_values)
    spectrum = torch.fft.fft(exponential)
    _require(bool(torch.isfinite(exponential).all() and torch.isfinite(spectrum).all()), "CUDA complex exponential or FFT probe returned nonfinite values")
    torch.cuda.synchronize(device)
    return {"executed": True, "passed": True, "device": str(device), "complex_exp_rows": int(exponential.numel()), "fft_rows": int(spectrum.numel())}


def _qfg_components(raw: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute untruncated normalized foreground/background errors in float64."""
    _require(raw.shape == target.shape, "generated image and target geometry differs")
    _require(raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200), "images must be N×1×200×200")
    image = raw.to(torch.float64).flatten(1)
    truth = target.to(torch.float64).flatten(1)
    _require(bool(torch.isfinite(image).all() and torch.isfinite(truth).all()), "image or target contains nonfinite intensity")
    _require(bool((image >= 0.0).all() and (truth >= 0.0).all()), "image or target contains negative intensity")
    image_mass, target_mass = image.sum(1), truth.sum(1)
    _require(bool((image_mass > 0.0).all() and (target_mass > 0.0).all()), "image or target has zero intensity mass")
    p_image, p_target = image / image_mass[:, None], truth / target_mass[:, None]
    mask = truth > 0.1
    _require(bool(mask.any(1).all() and (~mask).any(1).all()), "target foreground or background is empty")
    denominator = p_target.square().sum(1)
    _require(bool((denominator > 0.0).all()), "target shape-error denominator is zero")
    squared = (p_image - p_target).square()
    q_fg = (squared * mask).sum(1) / denominator
    q_bg = (squared * (~mask)).sum(1) / denominator
    q_raw = squared.sum(1) / denominator
    _require(bool(torch.allclose(q_fg + q_bg, q_raw, rtol=1.0e-12, atol=1.0e-14)), "Q_raw200 does not decompose into foreground and background")
    return {"q_fg": q_fg, "q_bg": q_bg, "q_raw200": q_raw, "p_image": p_image, "p_target": p_target, "mask": mask, "denominator": denominator}


def _qfg_gradient_reference(raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Independent analytic image-space derivative, including total-mass normalization."""
    values = _qfg_components(raw, target)
    p_image, p_target, mask = values["p_image"], values["p_target"], values["mask"]
    a = 2.0 * mask * (p_image - p_target) / values["denominator"][:, None]
    image_mass = raw.to(torch.float64).flatten(1).sum(1)
    return ((a - (a * p_image).sum(1, keepdim=True)) / image_mass[:, None]).reshape_as(raw)


def _rms(vector: torch.Tensor) -> float:
    value = vector.detach().to(torch.float64)
    _require(value.numel() > 0 and bool(torch.isfinite(value).all()), "gradient contains nonfinite values or is empty")
    return float(value.square().mean().sqrt().cpu())


def _unit_descent(gradient: torch.Tensor, name: str) -> torch.Tensor:
    norm = _rms(gradient)
    _require(math.isfinite(norm) and norm > 0.0, f"{name} direction has zero or invalid RMS")
    direction = -gradient.detach().to(torch.float64) / norm
    _require(bool(torch.isfinite(direction).all()), f"{name} unit direction is nonfinite")
    return direction


def _candidate_generator_gradient(
    production_gradient: torch.Tensor,
    q_fg_gradient: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    """Add only the declared Q_fg term under the production reconstruction weight."""
    _require(production_gradient.shape == q_fg_gradient.shape, "production and Q_fg Generator gradients have different shapes")
    _require(math.isfinite(lambda_value) and lambda_value >= 0.0, "candidate Q_fg weight is negative or nonfinite")
    _require(bool(torch.isfinite(production_gradient).all() and torch.isfinite(q_fg_gradient).all()), "candidate Generator gradient contains nonfinite values")
    return production_gradient + semantic.BETA * lambda_value * q_fg_gradient


def _decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["role"] == "confirmation" and int(row["label"]) in FOCUS_CLASSES]
    _require(len(selected) == 2 * 30 * len(SCENARIOS), "primary decision requires 30 confirmation rows in each focus class and scenario")
    means: dict[str, float] = {}
    class_means: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        class_means[scenario] = {}
        for digit in FOCUS_CLASSES:
            values = [float(row["q_fg"]) for row in selected if row["scenario"] == scenario and int(row["label"]) == digit]
            _require(len(values) == 30 and all(math.isfinite(value) for value in values), f"primary class {digit} is incomplete or nonfinite for {scenario}")
            class_means[scenario][str(digit)] = math.fsum(values) / len(values)
        means[scenario] = math.fsum(class_means[scenario].values()) / len(FOCUS_CLASSES)
    difference = means["candidate_plus"] - means["base_plus"]
    tolerance = DECISION_ATOL + DECISION_RTOL * max(abs(means["candidate_plus"]), abs(means["base_plus"]))
    if difference < -tolerance:
        classification = "CANDIDATE_LOCAL_RELATIVE_IMPROVEMENT"
        interpretation = "候选方向在固定局部相位步幅下，相对原方向降低了类别2、4的等权前景误差；这只支持继续设计单因素训练验证。"
    elif difference > tolerance:
        classification = "CANDIDATE_LOCAL_RELATIVE_REGRESSION"
        interpretation = "候选方向在固定局部相位步幅下相对原方向增加了类别2、4的等权前景误差；当前候选不获推进依据。"
    else:
        classification = "NO_DIFFERENCE_BEYOND_NUMERICAL_TOLERANCE"
        interpretation = "两方向差异未超过预设数值容差；不能据此宣称方向等效。"
    baseline_mean = math.fsum(float(row["q_fg"]) for row in selected if row["scenario"] == "baseline") / 60
    for scenario in ("base_plus", "candidate_plus"):
        class_means[scenario]["difference_from_baseline"] = means[scenario] - baseline_mean
    return {
        "classification": classification,
        "interpretation": interpretation,
        "means_q_fg_by_scenario": means,
        "means_q_fg_by_focus_class": class_means,
        "baseline_mean_q_fg": baseline_mean,
        "difference_candidate_minus_base_plus": difference,
        "tolerance": tolerance,
        "focus_classes_equal_weighted": list(FOCUS_CLASSES),
        "confirmation_rows_per_class": 30,
        "candidate_is_absolutely_lower_than_baseline": means["candidate_plus"] < baseline_mean,
        "base_is_absolutely_lower_than_baseline": means["base_plus"] < baseline_mean,
    }


def _json_new(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _csv_new(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _make_plan() -> str:
    return """# 冻结A4700前景目标梯度核验\n\n唯一问题：在冻结A4700处，保留原生产目标并加入固定规则校准的归一化前景Q项，是否使相同RMS的临时Generator相位下降方向在类别2、4确认人口上优于原生产方向？\n\n使用能量对照attempt_03的A4700、readouts_start、500行人口和父报告。原200行拟合人口按原10类平衡批次确定λ；原300行确认人口用于冻结点局部探针，其中类别2、4各30行构成唯一主量。所有人口均属canonical TRAIN。\n\nQ_fg基于未截断200×200原始强度、目标掩码T>0.1和全图质量归一化。候选重建项为R+λQ_fg；生产总方向按原BETA权重加入该项，并保留光功率及规范化E_S/E_I/E_C方向。权重只由拟合人口的RMS梯度匹配确定。局部方向取负梯度并归一化到单位RMS相位，固定探针幅度为1e-3 rad。\n\n唯一主量为类别2、4确认行等权Q_fg(candidate_plus)−Q_fg(base_plus)，容差为2e-7+2e-5乘两者最大绝对均值。负差支持继续设计单因素训练验证；正差削弱当前候选；容差内不宣称等效；核验失败不作科学负结论。报告各方向相对未扰动基线的绝对变化，避免把相对较优误称为改善。\n\n保存A4700基线按父生产的10行微批复现，以保持与readouts_start一致；实测相位扰动使用单行微批，候选与原方向主比较采用相同前向批次方式。基线与单行局部探针的微小数值差单独记录。每个临时相位扰动均恢复A4700，不调用optimizer.step。报告五项反馈和原始分块读出；完成后停止，不训练、不采用模型、不增加人审，training_recommendation=NONE、next_action=NONE。\n"""


def _assert_same_tensor(left: torch.Tensor, right: torch.Tensor, name: str) -> None:
    _require(left.shape == right.shape and left.dtype == right.dtype, f"{name} shape/dtype changed")
    _require(torch.equal(left, right), f"{name} changed")


def _autograd_group(loss: torch.Tensor, parameters: Mapping[str, nn.Parameter], *, retain_graph: bool) -> dict[str, torch.Tensor]:
    names = ("encoder.phase", "generator.phase")
    grads = torch.autograd.grad(loss, tuple(parameters[name] for name in names), retain_graph=retain_graph, allow_unused=True)
    return {name: torch.zeros_like(parameters[name]) if grad is None else grad.detach() for name, grad in zip(names, grads, strict=True)}


def _single_microbatch_batch_gradients(
    consumer: Any,
    parameters: Mapping[str, nn.Parameter],
    evaluators: Mapping[str, nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Reproduce the production objective with one-row forwards and add Q_fg gradients."""
    _require(images.shape == (BATCH_SIZE, 1, 28, 28) and labels.shape == (BATCH_SIZE,), "gradient batch must contain ten images and labels")
    _require(torch.bincount(labels.detach().cpu(), minlength=10).tolist() == [1] * 10, "gradient batch must contain one row from each class")
    phase = parameters["generator.phase"]
    refs = consumer.references
    _require(refs is not None, "frozen BOGT references are not bound")
    z_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for index in range(BATCH_SIZE):
            result = consumer.runtime.forward_from_image(images[index : index + 1])
            z_rows.append(result.z_ols.coefficients.detach())
            del result
    coefficients = torch.cat(z_rows, dim=0)
    l_dist, d_dist = capacity._dist_loss_and_ols_gradient(coefficients, refs.scale)
    g_r = torch.zeros_like(phase)
    g_q = torch.zeros_like(phase)
    g_op = torch.zeros_like(phase)
    g_dist = torch.zeros_like(phase)
    g_sem = {name: {key: torch.zeros_like(value) for key, value in parameters.items()} for name in ("E_S", "E_I", "E_C")}
    metric_totals = {"raw_mse": 0.0, "ssim_loss": 0.0, "q_fg": 0.0, "l_op": 0.0, **{f"{name}_ce": 0.0 for name in ("E_S", "E_I", "E_C")}}
    for index in range(BATCH_SIZE):
        result = consumer.runtime.forward_from_image(images[index : index + 1])
        raw, target = result.raw_intensity, result.image_input.slm_intensity
        terms = stroke.reconstruction_terms(raw, target, structural=True)
        q_fg = _qfg_components(raw, target)["q_fg"].mean()
        planes = diagnostic.bogt.contracted_plane_powers(result)
        l_op = diagnostic.bogt.optical_power_loss_from_powers(planes, refs.power_reference)
        proxy = torch.real(torch.conj(d_dist[index : index + 1].to(result.z_ols.coefficients)) * result.z_ols.coefficients).sum()
        view = diagnostic.canonical.cmos_view(raw)
        ce_losses = {name: torch.nn.functional.cross_entropy(capacity._evaluator_logits(evaluators[name], view, name), labels[index : index + 1], reduction="mean") for name in ("E_S", "E_I", "E_C")}

        g_dist += _autograd_group(proxy, parameters, retain_graph=True)["generator.phase"] / BATCH_SIZE
        g_r += _autograd_group(terms["objective"], parameters, retain_graph=True)["generator.phase"] / BATCH_SIZE
        g_q += _autograd_group(q_fg, parameters, retain_graph=True)["generator.phase"] / BATCH_SIZE
        g_op += _autograd_group(l_op, parameters, retain_graph=True)["generator.phase"] / BATCH_SIZE
        for name, loss in ce_losses.items():
            grads = _autograd_group(loss, parameters, retain_graph=name != "E_C")
            for key in parameters:
                g_sem[name][key] += grads[key] / BATCH_SIZE

        metric_totals["raw_mse"] += float(terms["raw_mse"].detach().to(torch.float64).cpu()) / BATCH_SIZE
        metric_totals["ssim_loss"] += float(terms["ssim_loss"].detach().to(torch.float64).cpu()) / BATCH_SIZE
        metric_totals["q_fg"] += float(q_fg.detach().to(torch.float64).cpu()) / BATCH_SIZE
        metric_totals["l_op"] += float(l_op.detach().to(torch.float64).cpu()) / BATCH_SIZE
        for name, loss in ce_losses.items():
            metric_totals[f"{name}_ce"] += float(loss.detach().to(torch.float64).cpu()) / BATCH_SIZE
        del result, raw, target, terms, q_fg, planes, l_op, proxy, view, ce_losses

    canonical, unavailable = diagnostic._safe_canonical_gradient({key: g_sem[name] for name, key in zip(("E_S", "E_I", "E_C"), ("L_S", "L_I", "L_C"), strict=True)})
    _require(canonical is not None, f"canonical semantic direction unavailable: {unavailable}")
    _require(capacity._vector_norm(g_dist) == 0.0, "L_dist produced a Generator gradient despite the production structural zero")
    baseline = semantic.ALPHA * g_dist + semantic.BETA * g_r + semantic.GAMMA * g_op + semantic.KAPPA * canonical["generator.phase"]
    _require(bool(torch.isfinite(baseline).all() and torch.isfinite(g_q).all()), "production or Q_fg gradient is nonfinite")
    _require(all(parameter.grad is None for parameter in parameters.values()), "autograd populated a model parameter .grad")
    metric_totals["l_dist"] = float(l_dist.detach().to(torch.float64).cpu())
    metric_totals["gradient_rms"] = _rms(baseline)
    return baseline.detach(), g_r.detach(), g_q.detach(), {"metrics": metric_totals, "g_op_rms": _rms(g_op), "g_semantic_rms": _rms(canonical["generator.phase"]), "g_q_rms": _rms(g_q)}


def _qfg_phase_gradient(consumer: Any, parameters: Mapping[str, nn.Parameter], image: torch.Tensor) -> tuple[float, torch.Tensor]:
    result = consumer.runtime.forward_from_image(image)
    raw, target = result.raw_intensity, result.image_input.slm_intensity
    q = _qfg_components(raw, target)["q_fg"].mean()
    grad = torch.autograd.grad(q, parameters["generator.phase"], retain_graph=False)[0]
    return float(q.detach().to(torch.float64).cpu()), grad.detach()


def _metric_row(
    raw: torch.Tensor,
    target: torch.Tensor,
    view: torch.Tensor,
    ed_logits: torch.Tensor,
    *,
    scenario: str,
    position: int,
    source_row: int,
    label: int,
    role: str,
    pixel_area_m2: float,
) -> dict[str, Any]:
    pieces = _qfg_components(raw, target)
    mse = foreground._measure_components(raw, target)
    target_view = diagnostic.canonical.cmos_view(target)
    ssim = stroke.ssim_per_image(view, target_view)
    prediction = int(ed_logits.argmax(1).item())
    return {
        "scenario": scenario,
        "position": position,
        "source_row_index": source_row,
        "label": label,
        "role": role,
        "q_fg": float(pieces["q_fg"][0].cpu()),
        "q_bg": float(pieces["q_bg"][0].cpu()),
        "q_raw200": float(pieces["q_raw200"][0].cpu()),
        "raw_mse": float(mse["raw_mse"][0].cpu()),
        "raw_mse_fg": float(mse["raw_mse_fg"][0].cpu()),
        "raw_mse_bg": float(mse["raw_mse_bg"][0].cpu()),
        "ssim_28": float(ssim[0].to(torch.float64).cpu()),
        "integral_intensity": float(raw.to(torch.float64).sum().cpu()) * pixel_area_m2,
        "clipped_fraction": float((raw >= 1.0).to(torch.float64).mean().cpu()),
        "e_d_prediction": prediction,
        "e_d_target_logit": float(ed_logits[0, label].to(torch.float64).cpu()),
    }


def _forward_one(
    consumer: Any,
    evaluators: Mapping[str, nn.Module],
    image: torch.Tensor,
    *,
    scenario: str,
    position: int,
    inputs: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    result = consumer.runtime.forward_from_image(image)
    raw = result.raw_intensity.detach()
    target = result.image_input.slm_intensity.detach()
    view = diagnostic.canonical.cmos_view(raw)
    logits = capacity._evaluator_logits(evaluators["E_D"], view, "E_D")
    expected_target = inputs["readout"]["target_slm_intensity"][position : position + 1].to(target.device)
    _require(torch.allclose(target, expected_target, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"target intensity differs from saved evidence at position {position}")
    row = _metric_row(
        raw,
        target,
        view,
        logits,
        scenario=scenario,
        position=position,
        source_row=int(inputs["population"]["source_row_indices"][position]),
        label=int(inputs["labels"][position]),
        role=inputs["roles"][position],
        pixel_area_m2=float(inputs["pixel_area_m2"]),
    )
    return raw.cpu(), view.cpu(), logits.detach().cpu(), row


def _run_scenario(
    consumer: Any,
    evaluators: Mapping[str, nn.Module],
    inputs: Mapping[str, Any],
    output_dir: Path,
    *,
    scenario: str,
    images: torch.Tensor,
    device: torch.device,
    saved_raw: torch.Tensor,
    saved_view: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenario_dir = output_dir / "readouts" / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for chunk_index, start in enumerate(range(0, TOTAL_ROWS, CHUNK_SIZE)):
        stop = min(start + CHUNK_SIZE, TOTAL_ROWS)
        raw_parts, view_parts, logits_parts = [], [], []
        if scenario == "baseline":
            # The saved A4700 reference was produced by the parent's ten-row inference batch.
            for batch_start in range(start, stop, BATCH_SIZE):
                batch_stop = min(batch_start + BATCH_SIZE, stop)
                with torch.no_grad():
                    result = consumer.runtime.forward_from_image(images[batch_start:batch_stop].to(device))
                    raw_batch = result.raw_intensity.detach().cpu()
                    view_batch = diagnostic.canonical.cmos_view(result.raw_intensity).detach().cpu()
                    target_batch = result.image_input.slm_intensity.detach().cpu()
                    logits_batch = capacity._evaluator_logits(evaluators["E_D"], diagnostic.canonical.cmos_view(result.raw_intensity), "E_D").detach().cpu()
                _require(torch.allclose(target_batch, inputs["readout"]["target_slm_intensity"][batch_start:batch_stop], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"A4700 target replay differs at positions {batch_start}:{batch_stop}")
                for local, position in enumerate(range(batch_start, batch_stop)):
                    raw, view, logits = raw_batch[local : local + 1], view_batch[local : local + 1], logits_batch[local : local + 1]
                    _require(torch.allclose(raw, saved_raw[position : position + 1], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"A4700 raw replay differs at row position {position}")
                    _require(torch.allclose(view, saved_view[position : position + 1], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"A4700 production view replay differs at row position {position}")
                    rows.append(_metric_row(raw, target_batch[local : local + 1], view, logits, scenario=scenario, position=position, source_row=int(inputs["population"]["source_row_indices"][position]), label=int(inputs["labels"][position]), role=inputs["roles"][position], pixel_area_m2=float(inputs["pixel_area_m2"])))
                raw_parts.append(raw_batch)
                view_parts.append(view_batch)
                logits_parts.append(logits_batch)
                del result, raw_batch, view_batch, target_batch, logits_batch
        else:
            for position in range(start, stop):
                with torch.no_grad():
                    raw, view, logits, row = _forward_one(
                        consumer,
                        evaluators,
                        images[position : position + 1].to(device),
                        scenario=scenario,
                        position=position,
                        inputs=inputs,
                    )
                raw_parts.append(raw)
                view_parts.append(view)
                logits_parts.append(logits)
                rows.append(row)
        chunk_path = scenario_dir / f"chunk_{chunk_index:04d}.pt"
        _require(not chunk_path.exists(), f"refusing to overwrite {chunk_path}")
        torch.save(
            {
                "task_id": TASK_ID,
                "scenario": scenario,
                "positions": torch.arange(start, stop, dtype=torch.long),
                "source_row_indices": torch.tensor(inputs["population"]["source_row_indices"][start:stop], dtype=torch.long),
                "labels": inputs["labels"][start:stop].cpu(),
                "roles": inputs["roles"][start:stop],
                "raw_cmos": torch.cat(raw_parts),
                "production_view": torch.cat(view_parts),
                "e_d_logits": torch.cat(logits_parts),
            },
            chunk_path,
        )
        print(f"saved {scenario} rows {start}:{stop}", flush=True)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == len(SCENARIOS) * TOTAL_ROWS, "saved paired CSV row count differs")
    parsed: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        item = dict(row)
        item["position"] = int(item["position"])
        item["source_row_index"] = int(item["source_row_index"])
        item["label"] = int(item["label"])
        for name in ROW_FIELDS[5:14]:
            item[name] = float(item[name])
        item["e_d_prediction"] = int(item["e_d_prediction"])
        item["e_d_target_logit"] = float(item["e_d_target_logit"])
        key = (item["scenario"], item["position"])
        _require(item["scenario"] in SCENARIOS and key not in seen, "paired CSV contains an invalid scenario or duplicate row")
        seen.add(key)
        _require(0 <= item["position"] < TOTAL_ROWS and 0 <= item["label"] <= 9, "paired CSV contains an invalid source position or label")
        _require(all(math.isfinite(float(item[name])) for name in ROW_FIELDS[5:]), "paired CSV contains a nonfinite measurement")
        parsed.append(item)
    _require(len(seen) == len(SCENARIOS) * TOTAL_ROWS, "paired CSV is missing a scenario-position pair")
    return parsed


def _recheck_readout_chunks(output_dir: Path, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recompute saved row metrics from every persisted raw output chunk."""
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for chunk_index, start in enumerate(range(0, TOTAL_ROWS, CHUNK_SIZE)):
            stop = min(start + CHUNK_SIZE, TOTAL_ROWS)
            path = output_dir / "readouts" / scenario / f"chunk_{chunk_index:04d}.pt"
            _require(path.is_file(), f"saved readout chunk is missing: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            positions = torch.arange(start, stop, dtype=torch.long)
            expected_rows = torch.tensor(inputs["population"]["source_row_indices"][start:stop], dtype=torch.long)
            expected_labels = inputs["labels"][start:stop].cpu()
            _require(payload.get("scenario") == scenario and torch.equal(payload.get("positions"), positions), f"readout position mapping differs for {scenario} chunk {chunk_index}")
            _require(torch.equal(payload.get("source_row_indices"), expected_rows) and torch.equal(payload.get("labels"), expected_labels), f"readout source row/label mapping differs for {scenario} chunk {chunk_index}")
            _require(payload.get("roles") == inputs["roles"][start:stop], f"readout roles differ for {scenario} chunk {chunk_index}")
            raw, view, logits = payload["raw_cmos"], payload["production_view"], payload["e_d_logits"]
            _require(tuple(raw.shape) == (stop - start, 1, 200, 200) and tuple(view.shape) == (stop - start, 1, 28, 28) and tuple(logits.shape) == (stop - start, 10), f"readout chunk geometry differs for {scenario} chunk {chunk_index}")
            _require(bool(torch.isfinite(raw).all() and torch.isfinite(view).all() and torch.isfinite(logits).all()), f"readout chunk contains nonfinite values for {scenario} chunk {chunk_index}")
            _require(bool((raw >= 0.0).all() and (raw.flatten(1).sum(1) > 0.0).all()), f"readout chunk contains negative or zero-mass output for {scenario} chunk {chunk_index}")
            replay_view = diagnostic.canonical.cmos_view(raw)
            _require(torch.allclose(replay_view, view, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"saved readout production view fails replay for {scenario} chunk {chunk_index}")
            for local, position in enumerate(range(start, stop)):
                target = inputs["readout"]["target_slm_intensity"][position : position + 1].cpu()
                row = _metric_row(
                    raw[local : local + 1],
                    target,
                    view[local : local + 1],
                    logits[local : local + 1],
                    scenario=scenario,
                    position=position,
                    source_row=int(expected_rows[local]),
                    label=int(expected_labels[local]),
                    role=inputs["roles"][position],
                    pixel_area_m2=float(inputs["pixel_area_m2"]),
                )
                rows.append(row)
    _require(len(rows) == len(SCENARIOS) * TOTAL_ROWS, "raw readout replay did not cover all scenario rows")
    return rows


def _summaries(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_fields = ("q_fg", "q_bg", "q_raw200", "raw_mse", "raw_mse_fg", "raw_mse_bg", "ssim_28", "integral_intensity", "clipped_fraction")
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario"]), str(row["role"]), int(row["label"])), []).append(row)
    class_rows = []
    for scenario in SCENARIOS:
        for role in ("fit", "confirmation"):
            for digit in range(10):
                group = grouped[(scenario, role, digit)]
                expected = 20 if role == "fit" else 30
                _require(len(group) == expected, f"summary group has wrong size: {scenario}/{role}/{digit}")
                record: dict[str, Any] = {"scenario": scenario, "role": role, "label": digit, "rows": len(group)}
                for field in summary_fields:
                    record[f"mean_{field}"] = math.fsum(float(row[field]) for row in group) / len(group)
                record["e_d_correct"] = sum(int(row["e_d_prediction"]) == digit for row in group)
                class_rows.append(record)
    return class_rows, _decision(rows)


def _check_fd(
    consumer: Any,
    parameters: Mapping[str, nn.Parameter],
    image: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilon: float,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(image)
    raw0, target0 = result.raw_intensity, result.image_input.slm_intensity
    q0_tensor = _qfg_components(raw0, target0)["q_fg"].mean()
    gradient = torch.autograd.grad(q0_tensor, parameters["generator.phase"], retain_graph=True)[0]
    image_gradient = torch.autograd.grad(q0_tensor, raw0, retain_graph=False)[0]
    reference_gradient = _qfg_gradient_reference(raw0.detach(), target0.detach()).to(image_gradient)
    image_gradient_error = float((image_gradient.to(torch.float64) - reference_gradient.to(torch.float64)).abs().max().cpu())
    _require(torch.allclose(image_gradient, reference_gradient, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), "autograd image derivative differs from the independent normalized-Q_fg formula")
    q0 = float(q0_tensor.detach().to(torch.float64).cpu())
    del result, raw0, target0, q0_tensor, image_gradient, reference_gradient
    phase = parameters["generator.phase"]
    source = phase.detach().clone()
    device_direction = direction.to(device=phase.device, dtype=phase.dtype)
    predicted = float((gradient.to(torch.float64) * device_direction.to(torch.float64)).sum().cpu()) * epsilon
    try:
        with torch.no_grad():
            phase.copy_(source + epsilon * device_direction)
            result_plus = consumer.runtime.forward_from_image(image)
            q_plus = float(_qfg_components(result_plus.raw_intensity, result_plus.image_input.slm_intensity)["q_fg"][0].cpu())
            del result_plus
        with torch.no_grad():
            phase.copy_(source - epsilon * device_direction)
            result_minus = consumer.runtime.forward_from_image(image)
            q_minus = float(_qfg_components(result_minus.raw_intensity, result_minus.image_input.slm_intensity)["q_fg"][0].cpu())
            del result_minus
    finally:
        with torch.no_grad():
            phase.copy_(source)
    central = 0.5 * (q_plus - q_minus)
    tolerance = FD_ATOL + FD_RTOL * abs(predicted)
    _require(math.isfinite(q0) and math.isfinite(central) and abs(central - predicted) <= tolerance, "phase finite difference differs from analytic directional derivative")
    if max(abs(central), abs(predicted)) > FD_ATOL:
        _require(central == 0.0 or predicted == 0.0 or math.copysign(1.0, central) == math.copysign(1.0, predicted), "phase finite difference has the wrong response direction")
    return {"q_at_source": q0, "image_gradient_max_abs_difference": image_gradient_error, "predicted_epsilon_derivative": predicted, "central_half_difference": central, "absolute_error": abs(central - predicted), "tolerance": tolerance, "epsilon": epsilon, "passed": True}


def _state_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _make_report_markdown(report: Mapping[str, Any], validation: Mapping[str, Any]) -> str:
    decision = report["decision"]
    means = decision["means_q_fg_by_scenario"]
    focus = report["secondary"]["focus_class_summary"]
    focus_lines = [
        f"| {row['scenario']} | {row['label']} | {row['mean_q_fg']:.6f} | {row['mean_q_bg']:.6f} | {row['mean_raw_mse_fg']:.6f} | {row['mean_ssim_28']:.6f} | {row['mean_integral_intensity']:.6g} | {row['mean_clipped_fraction']:.4%} | {row['e_d_correct']}/30 |"
        for row in focus
        if row["scenario"] in ("baseline", "base_plus", "candidate_plus")
    ]
    focus_table = "\n".join(focus_lines)
    return f"""# 冻结A4700前景目标梯度核验结果

## 结果与状态

测量状态为`{report['status']}`，主判定为`{decision['classification']}`。类别2、4确认人口的主差`D_local={decision['difference_candidate_minus_base_plus']:.9g}`，容差`τ={decision['tolerance']:.9g}`。未扰动基线Q_fg均值为`{decision['baseline_mean_q_fg']:.9g}`；原方向探针为`{means['base_plus']:.9g}`，候选方向探针为`{means['candidate_plus']:.9g}`。候选相对原方向较好不等于候选低于未扰动A4700。

## 科学认识

{decision['interpretation']} 本结论仅覆盖冻结A4700、固定拟合人口校准的目标梯度和当前300行canonical TRAIN确认人口；类别2、4为主分析。它不证明任何训练收益、人眼可读性或独立泛化。

## 辅助结果

| 情景 | 类别 | Q_fg | Q_bg | 前景raw MSE | SSIM28 | 积分强度 | 截断像素 | E_D正确 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{focus_table}

类别、角色、五种情景的完整汇总见`class_summary.csv`；逐行E_D读数及raw读数保存在配对表和分块文件中。E_D仅为辅助分类读数，不作可读性证据。拟合人口、其他类别和负向探针也均保留在逐类表中。

## 目标缺口

仍缺单因素训练的实际轨迹、当前联合采样候选的人眼识读与覆盖检查，以及独立人口确认。实验一的单人审阅质控不足仍然有效。

## 工程完成与验证

生产CUDA路径冻结核验了500行×5种情景，共2500行读出。A4700基线raw与生产视图复现、单行生产方向与既有训练梯度参考、Q_fg解析导数与相位有限差分、扰动恢复、模型/评价器/优化器冻结及CSV回读均按validation记录核对。更新数为0，optimizer.step调用数为0，人审数为0。独立CSV复算结果为`{validation['primary_decision_independently_recomputed']['classification']}`。

## 决策影响与下一步

当前结果仅用于判断是否值得另行设计单因素训练。候选方向相对改善可保留后续训练设计依据；相对退化则限制当前权重候选；容差内不支持等效结论。无论结果如何，本阶段不授权训练或模型采用。`training_recommendation=NONE`，`next_action=NONE`。
"""


def _run(artifact_arg: Path, checkpoint_arg: Path, output_arg: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    start_time = time.perf_counter()
    output_dir = output_arg if output_arg.is_absolute() else root / output_arg
    output_dir = output_dir.resolve()
    required_parent = (root / "outputs" / "20261009_m0_generator_foreground_objective_gradient_v1" / "runs").resolve()
    _require(output_dir.parent == required_parent and re.fullmatch(r"attempt_\d{2}", output_dir.name) is not None, "output path must be a numbered run under the named isolated study")
    _require(not output_dir.exists(), f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "plan.md").write_text(_make_plan(), encoding="utf-8", newline="\n")

    artifact_dir = artifact_arg if artifact_arg.is_absolute() else root / artifact_arg
    inputs = frozen._load_inputs(root, artifact_dir.resolve(), checkpoint_arg)
    _json_new(output_dir / "input_population.json", inputs["population"])
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized frozen production-path gradient audit")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    disk_free = shutil.disk_usage(output_dir).free
    estimated_readout_bytes = len(SCENARIOS) * TOTAL_ROWS * (200 * 200 * 4 + 28 * 28 * 4 + 10 * 4)
    _require(free_bytes > 0 and total_bytes > 0, "CUDA device reports no usable memory")
    _require(disk_free > estimated_readout_bytes * 1.2, "insufficient observed output-disk space for the planned evidence chunks")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dll_probe = _runtime_dll_probe()
    backend_probe = _cuda_backend_probe(device)

    gen_inputs = inputs["generation_inputs"]
    consumer, optimizer, parameters = parent_task._new_endpoint(gen_inputs, device)
    evaluators = generation.diagnostic.load_evaluators(inputs["generation_paths"], device)
    for model in evaluators.values():
        model.eval()
        _require(all(not parameter.requires_grad for parameter in model.parameters()), "an evaluator parameter is not frozen")
    _require(consumer.runtime.config.propagation_grid_size == 2_048, "production optical propagation grid differs")
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == SOURCE_STEP, "loaded A4700 Generator Adam step differs")
    _require(parameters["generator.phase"].shape == (340_000,), "A4700 Generator phase geometry differs")
    _require(not optimizer.state[parameters["encoder.phase"]], "Encoder unexpectedly has optimizer state in the A4700 Generator-only optimizer")
    source_phase = parameters["generator.phase"].detach().clone()
    source_runtime = _state_snapshot(consumer.runtime)
    source_frozen = capacity._frozen_runtime_state(consumer)
    source_eval = {name: _state_snapshot(model) for name, model in evaluators.items()}
    source_optimizer = capacity._cpu_tree(optimizer.state[parameters["generator.phase"]])

    fit_images = gen_inputs["images"][:FIT_ROWS].to(device)
    fit_labels = inputs["labels"][:FIT_ROWS].to(device)
    preflight_checks: dict[str, Any] = {}
    first_batch = fit_images[:BATCH_SIZE]
    first_labels = fit_labels[:BATCH_SIZE]
    _require(torch.bincount(first_labels.cpu(), minlength=10).tolist() == [1] * 10, "fixed preflight batch is not class balanced")
    base_single, r_gradient, q_gradient, fit_preview = _single_microbatch_batch_gradients(consumer, parameters, evaluators, first_batch, first_labels)
    _, _, _, saved_combined = stroke._logical_batch_gradients(consumer, parameters, evaluators, first_batch, first_labels, structural=True)
    ref_baseline = saved_combined["generator.phase"]
    gradient_error = float((base_single.to(torch.float64) - ref_baseline.to(torch.float64)).abs().max().cpu())
    gradient_pass = torch.allclose(base_single, ref_baseline, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL)
    preflight_checks["single_microbatch_production_direction"] = {"executed": True, "passed": bool(gradient_pass), "rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL, "max_abs_difference": gradient_error, "reference_microbatch_size": stroke.MICROBATCH_SIZE}
    _require(gradient_pass, "single-row production objective direction differs from existing logical-batch gradient")

    lambda_preview = _rms(r_gradient) / _rms(q_gradient)
    provisional = _candidate_generator_gradient(base_single, q_gradient, lambda_preview)
    preflight_dirs = {"base": _unit_descent(base_single, "preflight base"), "candidate": _unit_descent(provisional, "preflight candidate")}
    sample_positions = list(range(BATCH_SIZE))
    _require(torch.bincount(inputs["labels"][:BATCH_SIZE], minlength=10).tolist() == [1] * 10, "saved reference microbatch is not class balanced")
    saved_raw = inputs["readout"]["raw_cmos"].cpu()
    saved_view = inputs["readout"]["evaluator_view"].cpu()
    replay_rows = []
    with torch.no_grad():
        replay_result = consumer.runtime.forward_from_image(first_batch)
        replay_raw = replay_result.raw_intensity.detach().cpu()
        replay_view = diagnostic.canonical.cmos_view(replay_result.raw_intensity).detach().cpu()
        replay_target = replay_result.image_input.slm_intensity.detach().cpu()
        replay_logits = capacity._evaluator_logits(evaluators["E_D"], diagnostic.canonical.cmos_view(replay_result.raw_intensity), "E_D").detach().cpu()
    _require(torch.allclose(replay_raw, saved_raw[:BATCH_SIZE], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "saved baseline raw replay differs for the original ten-row reference microbatch")
    _require(torch.allclose(replay_view, saved_view[:BATCH_SIZE], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "saved baseline view replay differs for the original ten-row reference microbatch")
    _require(torch.allclose(replay_target, inputs["readout"]["target_slm_intensity"][:BATCH_SIZE].to(device).cpu(), rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "saved baseline target replay differs for the original ten-row reference microbatch")
    for local, position in enumerate(sample_positions):
        replay_rows.append(_metric_row(replay_raw[local : local + 1], replay_target[local : local + 1], replay_view[local : local + 1], replay_logits[local : local + 1], scenario="preflight", position=position, source_row=int(inputs["population"]["source_row_indices"][position]), label=int(inputs["labels"][position]), role=inputs["roles"][position], pixel_area_m2=float(inputs["pixel_area_m2"])))
    del replay_result, replay_raw, replay_view, replay_target, replay_logits
    preflight_checks["saved_a4700_raw_and_view_replay"] = {"executed": True, "passed": True, "rows": len(replay_rows), "reference_microbatch_size": BATCH_SIZE, "rtol": REPLAY_RTOL, "atol": REPLAY_ATOL}
    one_row_max = 0.0
    one_row_mean_sum = 0.0
    one_row_elements = 0
    one_row_tolerance_failures = 0
    for position in sample_positions:
        with torch.no_grad():
            result = consumer.runtime.forward_from_image(gen_inputs["images"][position : position + 1].to(device))
            current = result.raw_intensity.detach().cpu()
        expected = saved_raw[position : position + 1]
        absolute = (current.to(torch.float64) - expected.to(torch.float64)).abs()
        local_tol = REPLAY_ATOL + REPLAY_RTOL * expected.abs().to(torch.float64)
        one_row_max = max(one_row_max, float(absolute.max()))
        one_row_mean_sum += float(absolute.sum())
        one_row_elements += absolute.numel()
        one_row_tolerance_failures += int((absolute > local_tol).sum())
        del result, current, absolute, local_tol
    preflight_checks["single_row_vs_saved_ten_row_baseline_numeric_observation"] = {"executed": True, "rows": len(sample_positions), "max_abs_difference": one_row_max, "mean_abs_difference": one_row_mean_sum / one_row_elements, "elements_outside_parent_replay_tolerance": one_row_tolerance_failures, "parent_saved_replay_tolerance": {"rtol": REPLAY_RTOL, "atol": REPLAY_ATOL}, "used_as_gate": False}

    fd_preflight: dict[str, list[dict[str, Any]]] = {"base": [], "candidate": []}
    try:
        for direction_name, direction in preflight_dirs.items():
            for epsilon in (EPSILON, EPSILON / 2):
                for position in sample_positions:
                    fd_preflight[direction_name].append(_check_fd(consumer, parameters, gen_inputs["images"][position : position + 1].to(device), direction, epsilon=epsilon))
    finally:
        with torch.no_grad():
            parameters["generator.phase"].copy_(source_phase)
    preflight_checks["image_derivative_and_phase_finite_difference"] = {"executed": True, "passed": True, "directions": fd_preflight, "rows_per_direction_and_scale": len(sample_positions), "epsilon": EPSILON, "finite_difference_rtol": FD_RTOL, "finite_difference_atol": FD_ATOL}
    preflight_checks["phase_restored_after_preflight"] = {"executed": True, "passed": torch.equal(parameters["generator.phase"].detach(), source_phase)}
    _require(preflight_checks["phase_restored_after_preflight"]["passed"], "A4700 phase failed to restore after preflight probe")
    runtime_still_equal = capacity._tree_equal(source_runtime, _state_snapshot(consumer.runtime))
    evaluator_still_equal = all(capacity._tree_equal(source_eval[name], _state_snapshot(model)) for name, model in evaluators.items())
    optimizer_still_equal = capacity._tree_equal(source_optimizer, capacity._cpu_tree(optimizer.state[parameters["generator.phase"]]))
    gradients_empty = all(parameter.grad is None for parameter in parameters.values()) and all(parameter.grad is None for model in evaluators.values() for parameter in model.parameters())
    frozen_state_equal = capacity._tree_equal(source_frozen, capacity._frozen_runtime_state(consumer))
    preflight_checks["frozen_model_evaluator_optimizer_and_grad_state"] = {"executed": True, "passed": runtime_still_equal and evaluator_still_equal and optimizer_still_equal and gradients_empty and frozen_state_equal}
    _require(preflight_checks["frozen_model_evaluator_optimizer_and_grad_state"]["passed"], "model, evaluator, optimizer, or gradient state changed during preflight")
    preflight = {
        "task_id": TASK_ID,
        "status": "READY" if all(item["passed"] for item in preflight_checks.values() if "passed" in item) else "INVALID_HARNESS",
        "checks": preflight_checks,
        "resource_observations": {"device": str(device), "cuda_free_bytes": int(free_bytes), "cuda_total_bytes": int(total_bytes), "disk_free_bytes": int(disk_free), "estimated_readout_bytes": estimated_readout_bytes, "tf32_disabled": True},
        "cuda_backend_probe": backend_probe,
        "runtime_dll_probe": dll_probe,
        "provisional_lambda_fit_batch_0_only": lambda_preview,
        "formal_measurement_started": False,
    }
    _json_new(output_dir / "preflight.json", preflight)
    _require(preflight["status"] == "READY", "production no-update preflight did not produce READY")

    batches: list[dict[str, Any]] = []
    base_batch_gradients: list[torch.Tensor] = []
    candidate_batch_gradients: list[torch.Tensor] = []
    r_batch_rms: list[float] = []
    q_batch_rms: list[float] = []
    gq_batches: list[torch.Tensor] = []
    baseline_totals: list[dict[str, Any]] = []
    for batch_index in range(FIT_ROWS // BATCH_SIZE):
        start = batch_index * BATCH_SIZE
        stop = start + BATCH_SIZE
        base, g_r_batch, g_q_batch, totals = _single_microbatch_batch_gradients(consumer, parameters, evaluators, fit_images[start:stop], fit_labels[start:stop])
        if batch_index == 0:
            _require(torch.allclose(base, ref_baseline, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), "READY preflight baseline changed on the first fit batch")
        base_batch_gradients.append(base.cpu())
        gq_batches.append(g_q_batch.cpu())
        r_batch_rms.append(_rms(g_r_batch))
        q_batch_rms.append(_rms(g_q_batch))
        baseline_totals.append(totals["metrics"])
        batches.append({"batch_index": batch_index, "positions": [start, stop], "source_row_indices": inputs["population"]["source_row_indices"][start:stop], "labels": inputs["labels"][start:stop].tolist(), "baseline_gradient_rms": _rms(base), "reconstruction_gradient_rms": _rms(g_r_batch), "q_fg_gradient_rms": _rms(g_q_batch)})
        print(f"fit gradient batch {batch_index + 1}/{FIT_ROWS // BATCH_SIZE}", flush=True)
    lambda_value = math.sqrt(math.fsum(value * value for value in r_batch_rms) / len(r_batch_rms)) / math.sqrt(math.fsum(value * value for value in q_batch_rms) / len(q_batch_rms))
    _require(math.isfinite(lambda_value) and lambda_value > 0.0, "fit-only Q_fg gradient matching produced an invalid lambda")
    for base, q_gradient in zip(base_batch_gradients, gq_batches, strict=True):
        candidate_batch_gradients.append(_candidate_generator_gradient(base, q_gradient, lambda_value))
    base_mean = torch.stack(base_batch_gradients).to(torch.float64).mean(0)
    q_mean = torch.stack(gq_batches).to(torch.float64).mean(0)
    candidate_mean = torch.stack(candidate_batch_gradients).to(torch.float64).mean(0)
    h_base = _unit_descent(base_mean, "fit-mean base")
    h_candidate = _unit_descent(candidate_mean, "fit-mean candidate")
    _torch_fit_path = output_dir / "fit_gradients.pt"
    torch.save({"task_id": TASK_ID, "lambda": lambda_value, "beta": semantic.BETA, "base_mean_gradient": base_mean, "q_fg_mean_gradient": q_mean, "candidate_mean_gradient": candidate_mean, "base_descent_direction": h_base, "candidate_descent_direction": h_candidate, "batch_records": batches, "baseline_metrics_by_batch": baseline_totals, "r_gradient_rms_by_batch": r_batch_rms, "q_fg_gradient_rms_by_batch": q_batch_rms}, _torch_fit_path)
    _json_new(output_dir / "fit_calibration.json", {"lambda": lambda_value, "lambda_rule": "sqrt(mean_b RMS(grad R_b)^2) / sqrt(mean_b RMS(grad Q_fg_b)^2)", "fit_batches": len(batches), "fit_rows": FIT_ROWS, "beta_weight_applied_to_added_q_fg_gradient": semantic.BETA, "mean_base_gradient_rms": _rms(base_mean), "mean_q_fg_gradient_rms": _rms(q_mean), "mean_candidate_gradient_rms": _rms(candidate_mean), "base_candidate_cosine": float(torch.nn.functional.cosine_similarity(base_mean, candidate_mean, dim=0)), "confirmation_rows_used_for_calibration": 0})

    final_positions: list[int] = []
    for label in range(10):
        matching = [position for position in range(FIT_ROWS, TOTAL_ROWS) if int(inputs["labels"][position]) == label]
        _require(matching, f"no confirmation row exists for final derivative check class {label}")
        final_positions.append(matching[0])
    final_fd_checks: dict[str, list[dict[str, Any]]] = {"base": [], "candidate": []}
    for direction_name, direction in (("base", h_base), ("candidate", h_candidate)):
        for epsilon in (EPSILON, EPSILON / 2):
            for position in final_positions:
                final_fd_checks[direction_name].append(_check_fd(consumer, parameters, gen_inputs["images"][position : position + 1].to(device), direction, epsilon=epsilon))
    _require(torch.equal(parameters["generator.phase"].detach(), source_phase), "final direction derivative checks failed to restore A4700")
    _json_new(output_dir / "final_direction_derivative_checks.json", {"task_id": TASK_ID, "executed": True, "passed": True, "positions": final_positions, "epsilon_values": [EPSILON, EPSILON / 2], "checks": final_fd_checks})

    images = gen_inputs["images"].to(device)
    raw_base_phase = parameters["generator.phase"].detach().clone()
    direction_by_scenario = {"base_plus": (h_base, EPSILON), "base_minus": (h_base, -EPSILON), "candidate_plus": (h_candidate, EPSILON), "candidate_minus": (h_candidate, -EPSILON)}
    all_rows: list[dict[str, Any]] = []
    runtime_before_formal = _state_snapshot(consumer.runtime)
    optimizer_before_formal = capacity._cpu_tree(optimizer.state[parameters["generator.phase"]])
    try:
        with torch.no_grad():
            parameters["generator.phase"].copy_(raw_base_phase)
        all_rows.extend(_run_scenario(consumer, evaluators, inputs, output_dir, scenario="baseline", images=images, device=device, saved_raw=saved_raw, saved_view=saved_view))
        for scenario, (direction, signed_epsilon) in direction_by_scenario.items():
            with torch.no_grad():
                parameters["generator.phase"].copy_(raw_base_phase + signed_epsilon * direction.to(device=device, dtype=raw_base_phase.dtype))
            all_rows.extend(_run_scenario(consumer, evaluators, inputs, output_dir, scenario=scenario, images=images, device=device, saved_raw=saved_raw, saved_view=saved_view))
    finally:
        with torch.no_grad():
            parameters["generator.phase"].copy_(raw_base_phase)
    _require(torch.equal(parameters["generator.phase"].detach(), source_phase), "formal phase probes failed to restore A4700")
    _require(capacity._tree_equal(runtime_before_formal, _state_snapshot(consumer.runtime)), "runtime changed during formal phase probes")
    _require(capacity._tree_equal(optimizer_before_formal, capacity._cpu_tree(optimizer.state[parameters["generator.phase"]])), "optimizer state changed during formal phase probes")
    _require(capacity._tree_equal(source_eval, {name: _state_snapshot(model) for name, model in evaluators.items()}), "evaluator state changed during formal phase probes")
    _require(all(parameter.grad is None for parameter in parameters.values()) and all(parameter.grad is None for model in evaluators.values() for parameter in model.parameters()), "formal readout populated parameter .grad")

    _csv_new(output_dir / "paired_rows.csv", all_rows, ROW_FIELDS)
    reread_rows = _read_csv(output_dir / "paired_rows.csv")
    readout_rows = _recheck_readout_chunks(output_dir, inputs)
    csv_by_key = {(row["scenario"], int(row["position"])): row for row in reread_rows}
    for raw_row in readout_rows:
        csv_row = csv_by_key[(raw_row["scenario"], int(raw_row["position"]))]
        for name in ROW_FIELDS:
            if name in {"scenario", "role"}:
                _require(raw_row[name] == csv_row[name], f"CSV metadata differs from raw readout for {name}")
            elif name in {"position", "source_row_index", "label", "e_d_prediction"}:
                _require(int(raw_row[name]) == int(csv_row[name]), f"CSV integer differs from raw readout for {name}")
            else:
                _require(math.isclose(float(raw_row[name]), float(csv_row[name]), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL), f"CSV metric differs from raw readout for {name}")
    class_rows, decision = _summaries(reread_rows)
    class_fields = tuple(class_rows[0].keys())
    _csv_new(output_dir / "class_summary.csv", class_rows, class_fields)
    independent_rows = _read_csv(output_dir / "paired_rows.csv")
    _class_rows_again, decision_again = _summaries(independent_rows)
    _require(decision_again == decision, "independent CSV decision recomputation differs")
    validation = {
        "task_id": TASK_ID,
        "status": "VALIDATED",
        "paired_rows": len(reread_rows),
        "raw_readout_rows_recomputed": len(readout_rows),
        "raw_chunks_and_paired_csv_agree": True,
        "scenario_rows": {scenario: sum(row["scenario"] == scenario for row in reread_rows) for scenario in SCENARIOS},
        "per_class_per_role_counts": {f"{scenario}/{role}/{digit}": sum(row["scenario"] == scenario and row["role"] == role and row["label"] == digit for row in reread_rows) for scenario in SCENARIOS for role in ("fit", "confirmation") for digit in range(10)},
        "primary_decision_independently_recomputed": decision_again,
        "primary_decision_matches_machine_report": True,
        "source_state_restored": True,
        "optimizer_step_calls": 0,
        "model_update_calls": 0,
        "gradient_calls_are_diagnostic_only": True,
    }
    _json_new(output_dir / "validation.json", validation)
    report = {
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": decision["classification"],
        "decision": decision,
        "scope": {"source_attempt": inputs["artifact_dir"].as_posix(), "checkpoint": inputs["checkpoint_path"].as_posix(), "checkpoint_step": SOURCE_STEP, "fit_rows": FIT_ROWS, "confirmation_rows": CONFIRM_ROWS, "total_rows": TOTAL_ROWS, "primary_focus_rows": 60, "canonical_train_population": True, "formal_forward_rows": len(SCENARIOS) * TOTAL_ROWS, "optimizer_step_calls": 0, "model_updates": 0, "new_human_ratings": 0},
        "measurement": {"reconstruction_R": "raw200 MSE + 0.1*(1 - production SSIM28)", "candidate": "R + lambda*Q_fg, with BETA*lambda*grad(Q_fg) added to the preserved production total gradient", "q_fg": "unclipped raw200 image and target each normalized by total intensity; fixed mask target > 0.1; float64 statistics", "lambda": lambda_value, "lambda_fit_only": True, "direction": "negative mean fit gradient, normalized to unit RMS over Generator phase", "probe_epsilon_rad_rms": EPSILON, "primary_population": "300 confirmation rows, classes 2 and 4 equally weighted", "tolerance": {"absolute": DECISION_ATOL, "relative": DECISION_RTOL}, "finite_difference_tolerance": {"relative": FD_RTOL, "absolute": FD_ATOL}, "baseline_microbatch_size": BATCH_SIZE, "phase_probe_microbatch_size": 1, "primary_direction_comparison_uses_matched_microbatch": True},
        "preflight": preflight,
        "final_direction_derivative_checks": "final_direction_derivative_checks.json",
        "validation": validation,
        "secondary": {"class_summary_csv": "class_summary.csv", "focus_class_summary": [row for row in class_rows if row["role"] == "confirmation" and row["label"] in FOCUS_CLASSES], "full_class_role_scenario_summary": class_rows, "paired_rows_csv": "paired_rows.csv", "readouts": "readouts/<scenario>/chunk_NNNN.pt", "gradient_record": "fit_gradients.pt", "baseline_replay": True, "classifier_is_auxiliary": True, "same_endpoint_direction_probe_is_not_training": True},
        "artifacts": {"plan": "plan.md", "input_population": "input_population.json", "preflight": "preflight.json", "fit_calibration": "fit_calibration.json", "fit_gradients": "fit_gradients.pt", "paired_rows": "paired_rows.csv", "class_summary": "class_summary.csv", "validation": "validation.json", "report": "report.json", "report_markdown": "report.md", "completed": "completed.json"},
        "elapsed_seconds_observation_only": time.perf_counter() - start_time,
        "training_recommendation": "NONE",
        "next_action": "NONE",
    }
    _json_new(output_dir / "report.json", report)
    markdown = _make_report_markdown(report, validation)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8", newline="\n")
    _json_new(output_dir / "completed.json", {"task_id": TASK_ID, "status": "COMPLETED", "classification": decision["classification"], "primary_effect": decision["difference_candidate_minus_base_plus"], "training_recommendation": "NONE", "next_action": "NONE"})
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[3]
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output = output.resolve()
    try:
        report = _run(args.artifact_dir, args.checkpoint, args.output_dir)
    except Exception as error:
        if output.is_dir() and not (output / "report.json").exists() and not (output / "failure.json").exists():
            try:
                _json_new(output / "failure.json", {"task_id": TASK_ID, "status": "ENGINEERING_ERROR", "error_type": type(error).__name__, "error": str(error), "scientific_classification": None, "training_recommendation": "NONE", "next_action": "NONE"})
            except Exception:
                pass
        raise
    print(json.dumps({"status": report["status"], "classification": report["classification"], "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
