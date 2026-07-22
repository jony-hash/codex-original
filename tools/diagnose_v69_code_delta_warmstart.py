"""Read-only v69 B-to-code-delta warm-start smoke."""

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

REPORT_SCHEMA = "v69_code_delta_warmstart_v1"
FROZEN_PREFIXES = (
    "generator_e.template_logits",
    "generator_e.label_embedding.",
    "encoder.",
    "decoder.",
)


def _hash(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(prefixes):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(
    *,
    postmortem: str | Path,
    train_config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    output: str | Path,
    device: str = "cuda",
    max_samples: int = 4,
) -> dict[str, Any]:
    closure = json.loads(Path(postmortem).read_text())
    if closure.get("status") != "generator-injection-contract-insufficient":
        raise ValueError("v69 requires valid v68.7 closure")
    cfg = load_training_config(train_config)
    if (
        cfg.one_shot.generator_e_arch != "label_template_code_delta"
        or cfg.one_shot.code_residual_scale != 0.0
        or cfg.one_shot.distill_target_source != "teacher_camera"
    ):
        raise ValueError("invalid v69 code-delta contract")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v69 smoke")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    model = (
        OneShotOpticalGenerator(
            cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size
        )
        .to(dev)
        .train()
    )
    generator = load_one_shot_checkpoint_subset(
        model,
        payload,
        include_prefixes=("generator_e.",),
        description="v69 full B generator",
    )
    optical = load_one_shot_checkpoint_subset(
        model,
        payload,
        include_prefixes=("encoder.", "decoder."),
        description="v69 B phase",
        expected_missing_prefixes=(
            "encoder.amplitude_logits",
            "decoder.amplitude_logits",
        ),
    )
    frozen_before = _hash(model, FROZEN_PREFIXES)
    full_before = _hash(model, ("",))
    names = tuple(model.named_parameters())
    residual = [(n, p) for n, p in names if n.startswith("generator_e.residual_net.")]
    finite = True
    grad_sq = 0.0
    losses = []
    kwargs = _one_shot_loss_kwargs(cfg)
    for code, label, target in zip(
        data["codes"][:max_samples],
        data["class_labels"].long()[:max_samples],
        data["teacher_camera"][:max_samples],
    ):
        out = model(
            code[None].to(dev), label[None].to(dev), return_optical_fields=False
        )
        result = one_shot_loss(
            out,
            target[None].to(dev).clamp(0, 1),
            class_labels=label[None].to(dev),
            **kwargs,
        )
        loss = result["loss"]
        finite = (
            finite
            and bool(torch.isfinite(loss))
            and all(
                bool(torch.isfinite(v).all())
                for v in (*out.values(), *result.values())
                if torch.is_tensor(v)
            )
        )
        grads = torch.autograd.grad(
            loss, tuple(p for _, p in residual), allow_unused=True
        )
        for grad in grads:
            if grad is not None:
                grad_sq += float(grad.detach().float().square().sum())
                finite = finite and bool(torch.isfinite(grad).all())
        losses.append(float(loss.detach().cpu()))
    memory = None
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        memory = {
            "allocated_gib": float(torch.cuda.max_memory_allocated(dev) / 1024**3),
            "reserved_gib": float(torch.cuda.max_memory_reserved(dev) / 1024**3),
            "total_gib": float(
                torch.cuda.get_device_properties(dev).total_memory / 1024**3
            ),
        }
        memory["capacity_pass"] = memory["reserved_gib"] <= memory["total_gib"] * 0.90
    report = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "postmortem": str(postmortem),
        "generator_summary": generator,
        "optical_summary": optical,
        "fixed_exposure_scale": float(cfg.one_shot.diagnostic_camera_exposure_scale),
        "finite": finite,
        "loss_mean": sum(losses) / len(losses),
        "residual_gradient_l2": grad_sq**0.5,
        "frozen_hash_unchanged": frozen_before == _hash(model, FROZEN_PREFIXES),
        "parameters_unchanged": full_before == _hash(model, ("",)),
        "optimizer_created": False,
        "parameter_update_count": 0,
        "memory": memory,
    }
    report["status"] = (
        "v69-warmstart-smoke-passed"
        if finite
        and grad_sq > 0
        and report["frozen_hash_unchanged"]
        and report["parameters_unchanged"]
        and generator["loaded"] == 8
        and generator["missing"] == 2
        and optical["loaded"] == 2
        and optical["missing"] == 8
        and (memory is None or memory["capacity_pass"])
        else "harness-invalid"
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "postmortem",
        "train-config",
        "checkpoint",
        "validation-dataset",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=4)
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
