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


REPORT_SCHEMA = "v68_a2_warmstart_smoke_v1"


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


def _state_hash(model: torch.nn.Module) -> str:
    return _component_hash(model, ("",))


def _finite_state(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(value).all().item() for value in model.state_dict().values())


def _grad_finite(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )


def run(
    *,
    train_config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    output: str | Path,
    device: str = "cuda",
    max_samples: int = 4,
    max_peak_reserved_gib: float = 1.904,
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
    parameter_before_forward = _state_hash(model)
    loss_kwargs = _one_shot_loss_kwargs(config)
    losses: list[float] = []
    component_losses: dict[str, list[float]] = {}
    finite = _finite_state(model)
    if resolved.type == "cuda":
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
        losses.append(float(loss.detach().cpu().item()))
        for key, value in result.items():
            if torch.is_tensor(value) and value.ndim == 0:
                component_losses.setdefault(key, []).append(float(value.detach().cpu()))
        gradients = torch.autograd.grad(
            loss,
            tuple(model.parameters()),
            allow_unused=True,
            retain_graph=False,
        )
        finite = finite and all(
            grad is None or bool(torch.isfinite(grad).all().item()) for grad in gradients
        )
        del output_values, result, loss, gradients
    parameter_after_forward = _state_hash(model)
    peak_reserved_gib = None
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
        peak_reserved_gib = float(torch.cuda.max_memory_reserved(resolved) / (1024**3))
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "sample_count": int(len(labels)),
        "checkpoint": str(checkpoint),
        "checkpoint_load_mode": "explicit_template_and_optical_subset",
        "generator_summary": generator_summary,
        "optical_summary": optical_summary,
        "forbidden_architecture_keys": forbidden_keys,
        "hashes": {
            "before_load": before_load,
            "after_load": after_load,
            "prototype_bank_sha256": _component_hash(
                torch.nn.ModuleDict({"prototype": torch.nn.Parameter(prototypes, requires_grad=False)}),
                ("prototype",),
            ),
            "state_before_forward": parameter_before_forward,
            "state_after_forward": parameter_after_forward,
        },
        "loss_mean": float(sum(losses) / max(len(losses), 1)),
        "loss_components_mean": {
            key: float(sum(values) / len(values)) for key, values in component_losses.items()
        },
        "finite": bool(finite),
        "optimizer_created": False,
        "parameter_update_count": 0,
        "parameters_unchanged": parameter_before_forward == parameter_after_forward,
        "peak_reserved_gib": peak_reserved_gib,
        "memory_limit_gib": float(max_peak_reserved_gib),
    }
    report["memory_pass"] = peak_reserved_gib is None or peak_reserved_gib <= float(max_peak_reserved_gib)
    report["fixed_exposure_scale"] = float(exposure_scale)
    report["decision"] = (
        "a2-warmstart-smoke-passed"
        if report["finite"]
        and report["parameters_unchanged"]
        and report["memory_pass"]
        and not forbidden_keys
        and after_load["template"] != before_load["template"]
        and after_load["phase"] != before_load["phase"]
        else "a2-warmstart-smoke-failed"
    )
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
    parser.add_argument("--max-peak-reserved-gib", type=float, default=1.904)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
