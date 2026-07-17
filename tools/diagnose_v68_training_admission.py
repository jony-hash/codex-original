"""v68.2 CUDA gradient and memory admission (no optimizer, no training).

This tool compares the fixed 2048/3072 candidate with the streamed 3072/4096
reference using the complete ``one_shot_loss``.  It processes one sample at a
time and never updates parameters.  A/B training remains a separate user job.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ldm_project.student.objectives import one_shot_loss
from ldm_project.student.training_utils import _one_shot_loss_kwargs
try:
    from tools.diagnose_v68_transparent_memory import (
        _build_prototype_bank,
        _config_for_grid,
        _initialize_model,
    )
    from tools.diagnose_v674_transparent_operator import _set_phase_baseline
except ModuleNotFoundError:  # direct ``python tools/<script>.py`` compatibility
    from diagnose_v68_transparent_memory import (
        _build_prototype_bank,
        _config_for_grid,
        _initialize_model,
    )
    from diagnose_v674_transparent_operator import _set_phase_baseline


REPORT_SCHEMA = "v68_training_admission_v1"


def _parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "generator": list(model.generator_e.parameters()),
        "encoder": list(model.encoder.parameters()),
        "decoder": list(model.decoder.parameters()),
    }


def _flatten_gradients(
    gradients: tuple[torch.Tensor | None, ...], parameters: list[torch.nn.Parameter]
) -> torch.Tensor:
    values = []
    for gradient, parameter in zip(gradients, parameters):
        values.append(
            (torch.zeros_like(parameter) if gradient is None else gradient)
            .detach()
            .float()
            .reshape(-1)
        )
    return torch.cat(values)


def _compare_vectors(candidate: list[torch.Tensor], reference: list[torch.Tensor]) -> dict[str, float]:
    candidate_vector = torch.cat(candidate)
    reference_vector = torch.cat(reference)
    cosine = torch.nn.functional.cosine_similarity(
        candidate_vector[None], reference_vector[None], dim=1
    ).item()
    candidate_norm = candidate_vector.norm().item()
    reference_norm = reference_vector.norm().item()
    return {
        "cosine": float(cosine),
        "norm_ratio": float(candidate_norm / max(reference_norm, 1.0e-12)),
        "relative_l2": float(
            (candidate_vector - reference_vector).norm().item()
            / max(reference_norm, 1.0e-12)
        ),
    }


def _model_for_grid(base: Any, device: torch.device, state_grid: int, fft_grid: int):
    from ldm_project import OneShotOpticalGenerator

    model = OneShotOpticalGenerator(
        replace(base.physics, physical_forward_contract=True),
        _config_for_grid(base, state_grid, fft_grid),
        image_size=base.first_stage.image_size,
    ).to(device)
    model.train()
    _set_phase_baseline(model, "small")
    initialization = _initialize_model(
        model,
        base=base,
        prototype_bank=_build_prototype_bank(base),
        device=device,
        warm_start_mode="prototype_init",
        checkpoint_payload=None,
    )
    return model, initialization


def _evaluate_model(
    model: torch.nn.Module,
    *,
    codes: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
    loss_kwargs: dict[str, object],
    device: torch.device,
    max_samples: int,
) -> tuple[list[float], dict[str, list[torch.Tensor]], float | None, bool]:
    groups = _parameter_groups(model)
    losses: list[float] = []
    gradients: dict[str, list[torch.Tensor]] = {name: [] for name in groups}
    peak_memory = None
    finite = True
    baseline_reserved = (
        float(torch.cuda.memory_reserved(device)) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    for code, label, target in zip(
        codes[:max_samples], labels[:max_samples], targets[:max_samples]
    ):
        model.zero_grad(set_to_none=True)
        output = model(code[None].to(device), label[None].to(device))
        result = one_shot_loss(
            output,
            target[None].to(device),
            class_labels=label[None].to(device),
            **loss_kwargs,
        )
        loss = result["loss"]
        if not torch.isfinite(loss):
            finite = False
            raise FloatingPointError("non-finite one_shot_loss")
        losses.append(float(loss.detach().cpu()))
        for name, parameters in groups.items():
            values = torch.autograd.grad(
                loss, parameters, retain_graph=name != "decoder", allow_unused=True
            )
            flat = _flatten_gradients(values, parameters)
            if not torch.isfinite(flat).all():
                finite = False
                raise FloatingPointError(f"non-finite {name} gradient")
            gradients[name].append(flat.cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory = max(
                peak_memory or 0.0,
                float(torch.cuda.max_memory_reserved(device) / (1024**3))
                - baseline_reserved,
            )
        del output, result, loss
    return losses, gradients, peak_memory, finite


def run(
    *, output: str | Path, train_config: str | Path,
    sample_tensors: str | Path, device: str = "cuda", max_samples: int = 20,
) -> dict[str, Any]:
    resolved = torch.device(device)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "preflight-input-required",
        "parameter_update_count": 0,
        "optimizer_created": False,
        "candidate": {"state_grid": 2048, "fft_grid": 3072},
        "reference": {"state_grid": 3072, "fft_grid": 4096},
    }
    if resolved.type == "cuda" and not torch.cuda.is_available():
        report["status"] = "cuda-unavailable"
    else:
        from ldm_project import load_training_config

        base = load_training_config(train_config)
        payload = torch.load(sample_tensors, map_location="cpu", weights_only=False)
        codes = payload["codes"]
        labels = payload["class_labels"].long()
        targets = payload.get("reference_images", payload.get("teacher_camera"))
        if targets is None:
            raise ValueError("sample bundle requires reference_images or teacher_camera")
        candidate, candidate_init = _model_for_grid(base, resolved, 2048, 3072)
        reference, reference_init = _model_for_grid(base, resolved, 3072, 4096)
        try:
            loss_kwargs = _one_shot_loss_kwargs(base, None)
            startup_free_gib = None
            total_memory_gib = None
            if resolved.type == "cuda":
                free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)
                startup_free_gib = float(free_bytes / (1024**3))
                total_memory_gib = float(total_bytes / (1024**3))
            if resolved.type == "cuda":
                torch.cuda.reset_peak_memory_stats(resolved)
            candidate_losses, candidate_grads, candidate_memory, candidate_finite = _evaluate_model(
                candidate, codes=codes, labels=labels, targets=targets,
                loss_kwargs=loss_kwargs, device=resolved, max_samples=max_samples,
            )
            if resolved.type == "cuda":
                torch.cuda.reset_peak_memory_stats(resolved)
            reference_losses, reference_grads, reference_memory, reference_finite = _evaluate_model(
                reference, codes=codes, labels=labels, targets=targets,
                loss_kwargs=loss_kwargs, device=resolved, max_samples=max_samples,
            )
            metrics = {}
            for name in ("generator", "encoder", "decoder"):
                metrics[name] = _compare_vectors(
                    candidate_grads[name], reference_grads[name]
                )
            candidate_loss = sum(candidate_losses) / len(candidate_losses)
            reference_loss = sum(reference_losses) / len(reference_losses)
            report.update(
                {
                    "status": "gradient-memory-measured",
                    "initialization": {"candidate": candidate_init, "reference": reference_init},
                    "loss_relative_difference": abs(candidate_loss - reference_loss)
                    / max(abs(reference_loss), 1.0e-12),
                    "gradient_metrics": metrics,
                    "candidate_peak_reserved_gib": candidate_memory,
                    "reference_peak_reserved_gib": reference_memory,
                    "finite": bool(candidate_finite and reference_finite),
                }
            )
            if resolved.type == "cuda":
                # The model itself is constructed before the probe, so compare
                # the measured reserved peak against both explicit admission
                # limits and report the limits instead of silently treating a
                # successful CUDA allocation as a pass.
                memory_limits = {
                    "candidate_delta_reserved_gib": float(candidate_memory or 0.0),
                    "reference_delta_reserved_gib": float(reference_memory or 0.0),
                    "startup_free_gib": startup_free_gib,
                    "startup_free_85pct_gib": startup_free_gib * 0.85,
                    "total_90pct_gib": total_memory_gib * 0.90,
                }
                report["memory_limits"] = memory_limits
                report["memory_pass"] = bool(
                    candidate_memory is not None
                    and reference_memory is not None
                    and candidate_memory <= memory_limits["startup_free_85pct_gib"]
                    and candidate_memory <= memory_limits["total_90pct_gib"]
                )
            else:
                report["memory_limits"] = {"device": "cpu"}
                report["memory_pass"] = False
            report["gradient_pass"] = (
                report["loss_relative_difference"] <= 0.002
                and all(
                    metrics[name]["cosine"] >= 0.999
                    and 0.98 <= metrics[name]["norm_ratio"] <= 1.02
                    and metrics[name]["relative_l2"] <= 0.02
                    for name in metrics
                )
            )
            report["decision"] = (
                "ready-for-a-training"
                if report["gradient_pass"] and report["memory_pass"] and report["finite"]
                else "stop-before-a-training"
            )
        finally:
            del candidate, reference
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--sample-tensors", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", default=20, type=int)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
