"""Read-only v68.6 B warm-start admission smoke."""

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
from tools.v68_b_acceptance_utils import validate_b_architecture
from tools.diagnose_v68_transparent_memory import _build_prototype_bank

REPORT_SCHEMA = "v68_b_warmstart_smoke_v1"


def _hash(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(prefixes):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _finite(values: Any) -> bool:
    return all(
        bool(torch.isfinite(v).all()) for v in values.values() if torch.is_tensor(v)
    )


def run(
    *,
    a2_decision: str | Path,
    train_config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    output: str | Path,
    device: str = "cuda",
    max_samples: int = 4,
) -> dict[str, Any]:
    linked = json.loads(Path(a2_decision).read_text(encoding="utf-8"))
    if (
        linked.get("status") != "a2-raw-readability-passed"
        or linked.get("b_allowed") is not True
    ):
        raise ValueError("B warm-start requires linked A2 decision with b_allowed=true")
    cfg = load_training_config(train_config)
    validate_b_architecture(cfg)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B warm-start smoke")
    torch.manual_seed(1234)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bundle = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes, labels = (
        bundle["codes"][:max_samples],
        bundle["class_labels"].long()[:max_samples],
    )
    if len(labels) != max_samples:
        raise ValueError("validation dataset has fewer than max_samples records")
    prototypes = _build_prototype_bank(cfg).float()
    model = (
        OneShotOpticalGenerator(
            cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size
        )
        .to(dev)
        .train()
    )
    fresh_residual = _hash(
        model, ("generator_e.label_embedding.", "generator_e.residual_net.")
    )
    gen = load_one_shot_checkpoint_subset(
        model,
        payload,
        include_prefixes=("generator_e.template_logits",),
        description="v68.6 B template",
        expected_missing_prefixes=(
            "generator_e.label_embedding.",
            "generator_e.residual_net.",
        ),
    )
    opt = load_one_shot_checkpoint_subset(
        model,
        payload,
        include_prefixes=("encoder.", "decoder."),
        description="v68.6 B phase",
        expected_missing_prefixes=(
            "encoder.amplitude_logits",
            "decoder.amplitude_logits",
        ),
    )
    after_load = _hash(model, ("",))
    residual_after_load = _hash(
        model, ("generator_e.label_embedding.", "generator_e.residual_net.")
    )
    names = tuple(model.named_parameters())
    groups = {"template": [], "residual": [], "encoder": [], "decoder": []}
    for name, param in names:
        if name.startswith("generator_e.template_logits"):
            groups["template"].append((name, param))
        elif name.startswith(
            ("generator_e.label_embedding.", "generator_e.residual_net.")
        ):
            groups["residual"].append((name, param))
        elif name.startswith("encoder."):
            groups["encoder"].append((name, param))
        elif name.startswith("decoder."):
            groups["decoder"].append((name, param))
    grad_norm = {key: 0.0 for key in groups}
    finite = True
    losses = []
    kwargs = _one_shot_loss_kwargs(cfg)
    for code, label in zip(codes, labels):
        model.zero_grad(set_to_none=True)
        out_values = model(
            code[None].to(dev), label[None].to(dev), return_optical_fields=False
        )
        result = one_shot_loss(
            out_values,
            prototypes[label][None].to(dev),
            class_labels=label[None].to(dev),
            **kwargs,
        )
        loss = result["loss"]
        finite = (
            finite
            and bool(torch.isfinite(loss))
            and _finite(result)
            and _finite(out_values)
        )
        grads = torch.autograd.grad(loss, tuple(p for _, p in names), allow_unused=True)
        for (name, _), grad in zip(names, grads):
            group = next(
                (
                    key
                    for key, values in groups.items()
                    if any(n == name for n, _ in values)
                ),
                None,
            )
            if group and grad is not None:
                finite = finite and bool(torch.isfinite(grad).all())
                grad_norm[group] += float(grad.detach().float().square().sum())
        losses.append(float(loss.detach().cpu()))
    state_after_load = after_load
    unchanged = state_after_load == _hash(model, ("",))
    memory: dict[str, Any] | None = None
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        memory = {
            "allocated_gib": float(torch.cuda.max_memory_allocated(dev) / (1024**3)),
            "reserved_gib": float(torch.cuda.max_memory_reserved(dev) / (1024**3)),
            "total_gib": float(
                torch.cuda.get_device_properties(dev).total_memory / (1024**3)
            ),
        }
        memory["capacity_pass"] = memory["reserved_gib"] <= memory["total_gib"] * 0.90
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "a2_decision": str(a2_decision),
        "a2_b_allowed": True,
        "sample_count": len(labels),
        "fixed_exposure_scale": float(cfg.one_shot.diagnostic_camera_exposure_scale),
        "generator_summary": gen,
        "optical_summary": opt,
        "coverage": {
            "generator_loaded": gen.get("loaded"),
            "generator_missing": gen.get("missing"),
            "optical_loaded": opt.get("loaded"),
            "optical_missing": opt.get("missing"),
        },
        "residual_embedding_seed_hash": fresh_residual,
        "residual_embedding_after_load_hash": residual_after_load,
        "residual_embedding_unchanged_on_load": fresh_residual == residual_after_load,
        "finite": finite,
        "loss_mean": sum(losses) / len(losses),
        "gradient_groups": {
            key: {"l2_norm": norm**0.5, "nonzero": norm > 0.0}
            for key, norm in grad_norm.items()
        },
        "optimizer_created": False,
        "parameter_update_count": 0,
        "parameters_unchanged": unchanged,
        "memory": memory,
    }
    core = (
        finite
        and unchanged
        and report["residual_embedding_unchanged_on_load"]
        and gen.get("loaded") == 1
        and gen.get("missing") == 9
        and opt.get("loaded") == 2
        and opt.get("missing") == 8
        and all(item["nonzero"] for item in report["gradient_groups"].values())
    )
    report["memory_pass"] = memory is None or bool(memory["capacity_pass"])
    report["status"] = (
        "b-warmstart-smoke-passed"
        if core and report["memory_pass"]
        else "harness-invalid"
    )
    report["decision"] = report["status"]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a2-decision", required=True, type=Path)
    p.add_argument("--train-config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--validation-dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=4)
    print(json.dumps(run(**vars(p.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
