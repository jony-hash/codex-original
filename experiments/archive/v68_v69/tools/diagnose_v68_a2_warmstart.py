"""Short, read-only v68.5 A2 warm-start and loss admission smoke.

This probe deliberately does not create an optimizer or update parameters.  It
loads only the A template logits and phase modules from the frozen v68 A
checkpoint, then evaluates four validation samples with the complete A2 loss.
Long CUDA training remains a user-run step after this gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ldm_project import load_training_config
from ldm_project.student.checkpoints import load_one_shot_checkpoint_subset
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from ldm_project.student.objectives import one_shot_loss
from ldm_project.student.training_utils import _one_shot_loss_kwargs

try:
    from tools.diagnose_v68_transparent_memory import _build_prototype_bank
except ModuleNotFoundError:  # direct ``python tools/<script>.py`` compatibility
    from diagnose_v68_transparent_memory import _build_prototype_bank


REPORT_SCHEMA = "v68_a2_warmstart_smoke_v2"
A2_ADMISSION_DELTA_RESERVED_LIMIT_GIB = 1.904296875
WARMSTART_STATE_KEYS = (
    "generator_e.template_logits",
    "encoder.phase",
    "decoder.phase",
)


def _component_hash(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if not key.startswith(prefixes):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_hash(model: torch.nn.Module) -> str:
    return _component_hash(model, ("",))


def _finite_state(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(value).all().item() for value in model.state_dict().values())


def _memory_gate(
    *,
    baseline_allocated_gib: float,
    baseline_reserved_gib: float,
    startup_free_gib: float,
    total_memory_gib: float,
    absolute_peak_allocated_gib: float,
    absolute_peak_reserved_gib: float,
) -> dict[str, float | bool]:
    """Compare A2 peak memory using the same delta-reserved basis as v68.2."""
    delta_allocated = max(absolute_peak_allocated_gib - baseline_allocated_gib, 0.0)
    delta_reserved = max(absolute_peak_reserved_gib - baseline_reserved_gib, 0.0)
    delta_against_admission = delta_reserved <= A2_ADMISSION_DELTA_RESERVED_LIMIT_GIB
    delta_capacity = delta_reserved <= startup_free_gib * 0.85
    absolute_capacity = absolute_peak_reserved_gib <= total_memory_gib * 0.90
    return {
        "baseline_allocated_gib": baseline_allocated_gib,
        "baseline_reserved_gib": baseline_reserved_gib,
        "startup_free_gib": startup_free_gib,
        "total_memory_gib": total_memory_gib,
        "absolute_peak_allocated_gib": absolute_peak_allocated_gib,
        "absolute_peak_reserved_gib": absolute_peak_reserved_gib,
        "delta_peak_allocated_gib": delta_allocated,
        "delta_peak_reserved_gib": delta_reserved,
        "delta_reserved_limit_gib": A2_ADMISSION_DELTA_RESERVED_LIMIT_GIB,
        "delta_reserved_against_v68_admission_pass": delta_against_admission,
        "delta_reserved_capacity_85pct_pass": delta_capacity,
        "absolute_reserved_capacity_90pct_pass": absolute_capacity,
        "memory_pass": delta_against_admission and delta_capacity and absolute_capacity,
    }


def _warmstart_coverage(
    *, model_state: dict[str, torch.Tensor], checkpoint_state: dict[str, torch.Tensor]
) -> dict[str, object]:
    expected = list(WARMSTART_STATE_KEYS)
    model_keys = [key for key in expected if key in model_state]
    checkpoint_keys = [key for key in expected if key in checkpoint_state]
    return {
        "expected_state_keys": expected,
        "model_state_keys": model_keys,
        "checkpoint_state_keys": checkpoint_keys,
        "complete": model_keys == expected and checkpoint_keys == expected,
    }


def _gradient_group_names(name: str) -> str | None:
    if name.startswith("generator_e."):
        return "generator"
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("decoder."):
        return "decoder"
    return None


def run(
    *,
    train_config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    output: str | Path,
    device: str = "cuda",
    max_samples: int = 4,
) -> dict[str, Any]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the A2 warm-start smoke")

    config = load_training_config(train_config)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dataset = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes = dataset["codes"][:max_samples]
    labels = dataset["class_labels"].long()[:max_samples]
    prototypes = _build_prototype_bank(config).float().contiguous()
    targets = prototypes.index_select(0, labels)
    if config.one_shot.diagnostic_camera_normalization != "fixed_exposure":
        raise ValueError("A2 warm-start smoke requires fixed_exposure normalization")
    exposure_scale = config.one_shot.diagnostic_camera_exposure_scale
    if exposure_scale is None or float(exposure_scale) <= 0.0:
        raise ValueError("A2 warm-start smoke requires a positive locked exposure scale")

    model = OneShotOpticalGenerator(
        config.physics,
        config.one_shot,
        image_size=config.first_stage.image_size,
    ).to(resolved)
    model.train()
    # A2 is explicitly a label-template model.  A residual/embedding module
    # would make this smoke test silently test the B architecture instead.
    forbidden_keys = [
        key
        for key in model.state_dict()
        if key.startswith("generator_e.label_embedding.")
        or key.startswith("generator_e.residual_net.")
    ]
    before_load = {
        "template": _component_hash(model, ("generator_e.template_logits",)),
        "phase": _component_hash(model, ("encoder.", "decoder.")),
    }
    generator_summary = load_one_shot_checkpoint_subset(
        model,
        checkpoint_payload,
        include_prefixes=("generator_e.template_logits",),
        description="v68 A2 template warm-start",
    )
    optical_summary = load_one_shot_checkpoint_subset(
        model,
        checkpoint_payload,
        include_prefixes=("encoder.", "decoder."),
        description="v68 A2 optical warm-start",
        expected_missing_prefixes=("encoder.amplitude_logits", "decoder.amplitude_logits"),
    )
    after_load = {
        "template": _component_hash(model, ("generator_e.template_logits",)),
        "phase": _component_hash(model, ("encoder.", "decoder.")),
    }
    coverage = _warmstart_coverage(
        model_state=model.state_dict(), checkpoint_state=checkpoint_payload["model_state_dict"]
    )
    parameter_before_forward = _state_hash(model)
    loss_kwargs = _one_shot_loss_kwargs(config)
    losses: list[float] = []
    component_losses: dict[str, list[float]] = {}
    finite = _finite_state(model)
    fields_finite = True
    named_parameters = tuple(model.named_parameters())
    gradient_records: dict[str, dict[str, float | int | bool]] = {
        group: {
            "parameter_count": 0,
            "observed_gradient_count": 0,
            "finite": True,
            "squared_norm_sum": 0.0,
            "sample_count": 0,
        }
        for group in ("generator", "encoder", "decoder")
    }
    for name, _ in named_parameters:
        group = _gradient_group_names(name)
        if group is not None:
            gradient_records[group]["parameter_count"] += 1
    baseline_memory: dict[str, float] | None = None
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
        free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)
        baseline_memory = {
            "baseline_allocated_gib": float(torch.cuda.memory_allocated(resolved) / (1024**3)),
            "baseline_reserved_gib": float(torch.cuda.memory_reserved(resolved) / (1024**3)),
            "startup_free_gib": float(free_bytes / (1024**3)),
            "total_memory_gib": float(total_bytes / (1024**3)),
        }
        torch.cuda.reset_peak_memory_stats(resolved)
    for code, label, target in zip(codes, labels, targets):
        model.zero_grad(set_to_none=True)
        output_values = model(code[None].to(resolved), label[None].to(resolved))
        result = one_shot_loss(
            output_values,
            target[None].to(resolved),
            class_labels=label[None].to(resolved),
            **loss_kwargs,
        )
        loss = result["loss"]
        finite = finite and bool(torch.isfinite(loss).item())
        finite = finite and all(
            bool(torch.isfinite(value).all().item())
            for value in result.values()
            if torch.is_tensor(value)
        )
        fields_finite = fields_finite and all(
            bool(torch.isfinite(value).all().item())
            for value in output_values.values()
            if torch.is_tensor(value)
        )
        losses.append(float(loss.detach().cpu().item()))
        for key, value in result.items():
            if torch.is_tensor(value) and value.ndim == 0:
                component_losses.setdefault(key, []).append(float(value.detach().cpu()))
        gradients = torch.autograd.grad(
            loss,
            tuple(parameter for _, parameter in named_parameters),
            allow_unused=True,
            retain_graph=False,
        )
        for (name, _), gradient in zip(named_parameters, gradients):
            group = _gradient_group_names(name)
            if group is None or gradient is None:
                continue
            group_record = gradient_records[group]
            group_record["observed_gradient_count"] += 1
            is_finite = bool(torch.isfinite(gradient).all().item())
            group_record["finite"] = bool(group_record["finite"]) and is_finite
            finite = finite and is_finite
            if is_finite:
                group_record["squared_norm_sum"] += float(
                    gradient.detach().float().square().sum().cpu()
                )
        for record in gradient_records.values():
            record["sample_count"] += 1
        del output_values, result, loss, gradients
    parameter_after_forward = _state_hash(model)
    memory: dict[str, float | bool] | None = None
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
        assert baseline_memory is not None
        memory = _memory_gate(
            **baseline_memory,
            absolute_peak_allocated_gib=float(
                torch.cuda.max_memory_allocated(resolved) / (1024**3)
            ),
            absolute_peak_reserved_gib=float(
                torch.cuda.max_memory_reserved(resolved) / (1024**3)
            ),
        )
    gradient_groups = {
        name: {
            "parameter_count": int(record["parameter_count"]),
            "observed_gradient_count": int(record["observed_gradient_count"]),
            "finite": bool(record["finite"]),
            "l2_norm": float(float(record["squared_norm_sum"]) ** 0.5),
        }
        for name, record in gradient_records.items()
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "sample_count": int(len(labels)),
        "checkpoint": str(checkpoint),
        "checkpoint_load_mode": "explicit_template_and_optical_subset",
        "generator_summary": generator_summary,
        "optical_summary": optical_summary,
        "warmstart_coverage": coverage,
        "forbidden_architecture_keys": forbidden_keys,
        "hashes": {
            "before_load": before_load,
            "after_load": after_load,
            "prototype_bank_sha256": _tensor_hash(prototypes),
            "state_before_forward": parameter_before_forward,
            "state_after_forward": parameter_after_forward,
        },
        "loss_mean": float(sum(losses) / max(len(losses), 1)),
        "loss_components_mean": {
            key: float(sum(values) / len(values)) for key, values in component_losses.items()
        },
        "finite": bool(finite),
        "output_fields_finite": bool(fields_finite),
        "gradient_groups": gradient_groups,
        "optimizer_created": False,
        "parameter_update_count": 0,
        "parameters_unchanged": parameter_before_forward == parameter_after_forward,
        "memory": memory,
    }
    report["memory_pass"] = memory is not None and bool(memory["memory_pass"])
    report["fixed_exposure_scale"] = float(exposure_scale)
    core_pass = (
        report["finite"]
        and report["output_fields_finite"]
        and report["parameters_unchanged"]
        and coverage["complete"]
        and not forbidden_keys
        and all(record["finite"] for record in gradient_groups.values())
        and all(record["observed_gradient_count"] > 0 for record in gradient_groups.values())
        and after_load["template"] != before_load["template"]
        and after_load["phase"] != before_load["phase"]
    )
    if core_pass and report["memory_pass"]:
        report["status"] = "smoke-complete"
        report["decision"] = "a2-warmstart-smoke-passed"
    elif core_pass:
        report["status"] = "memory-regression"
        report["decision"] = "a2-warmstart-memory-regression"
    else:
        report["status"] = "warmstart-or-numerical-failed"
        report["decision"] = "a2-warmstart-smoke-failed"
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
