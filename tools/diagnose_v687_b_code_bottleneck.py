"""Read-only v68.7 generator-injection postmortem for frozen B evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ldm_project import load_training_config
from ldm_project.student.checkpoints import load_one_shot_checkpoint
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from tools.v68_b_acceptance_utils import balanced_factor_decomposition

REPORT_SCHEMA = "v687_b_code_bottleneck_v1"


def _hash(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(prefixes):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _relative_change(
    checkpoint_state: dict[str, torch.Tensor],
    fresh_state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, float | int]:
    delta = reference = 0.0
    count = 0
    for key, value in checkpoint_state.items():
        if key.startswith(prefix):
            ref = fresh_state[key]
            delta += float((value.double() - ref.double()).square().sum())
            reference += float(ref.double().square().sum())
            count += value.numel()
    return {
        "parameter_count": count,
        "delta_l2": delta**0.5,
        "relative_delta_l2": (delta / max(reference, 1e-30)) ** 0.5,
    }


def run(
    *,
    config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    b_decision: str | Path,
    crossed_report: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    decision = json.loads(Path(b_decision).read_text())
    crossed = json.loads(Path(crossed_report).read_text())
    if decision.get("status") != "b-prototype-imaging-only":
        raise ValueError("v68.7 requires the valid B prototype-only decision")
    if crossed.get("schema") != "v68_b_crossed_raw_cmos_v1" or not crossed.get(
        "finite"
    ):
        raise ValueError("v68.7 requires the valid B crossed v2 report")
    cfg = load_training_config(config)
    if (
        cfg.one_shot.generator_e_arch != "label_template_code_residual"
        or float(cfg.one_shot.code_residual_scale) != 0.10
    ):
        raise ValueError("v68.7 requires the frozen B residual contract at scale 0.10")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    torch.manual_seed(1234)
    fresh = OneShotOpticalGenerator(
        cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size
    )
    model = OneShotOpticalGenerator(
        cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size
    )
    load_one_shot_checkpoint(model, payload)
    codes = data["codes"]
    if tuple(codes.shape) != (100, cfg.one_shot.code_dim):
        raise ValueError("v68.7 requires the 100-code validation split")
    labels = torch.arange(10).repeat(100)
    crossed_codes = codes[:, None, :].expand(100, 10, -1).reshape(1000, -1)
    with torch.no_grad():
        command, diagnostics = model.generator_e.forward_with_diagnostics(
            crossed_codes, labels
        )
    residual = diagnostics["generator_residual_map"].reshape(100, 10, 1, 200, 200)
    desired = diagnostics["generator_desired_intensity"].reshape(100, 10, 1, 200, 200)
    base = diagnostics["generator_base_command"].reshape(100, 10, 1, 200, 200)
    command = command.reshape(100, 10, 1, 200, 200)
    template_logits = model.generator_e.template_logits.detach()
    centered = residual - residual.mean(dim=0, keepdim=True)
    oracle = model.generator_e._to_amplitude_command(
        torch.sigmoid(template_logits[None] + centered.sign() * 0.10)
    )
    factors = {
        "residual_map": balanced_factor_decomposition(residual)[0],
        "desired_intensity": balanced_factor_decomposition(desired)[0],
        "slm_command": balanced_factor_decomposition(command)[0],
        "bounded_sign_code_oracle_slm": balanced_factor_decomposition(oracle)[0],
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "generator-injection-contract-insufficient"
        if factors["bounded_sign_code_oracle_slm"]["code_fraction"] < 0.25
        else "harness-invalid",
        "linked_reports": {
            "b_decision": str(b_decision),
            "crossed_report": str(crossed_report),
        },
        "checkpoint": str(checkpoint),
        "finite": all(item["finite"] for item in factors.values()),
        "factors": {
            **factors,
            "pre_encoder": crossed["factor_decomposition"]["pre_encoder"],
            "chip": crossed["factor_decomposition"]["chip"],
            "camera": crossed["factor_decomposition"]["camera"],
        },
        "command_minus_base_abs_mean": float((command - base).abs().mean()),
        "parameter_changes_from_seed": {
            "label_embedding": _relative_change(
                payload["model_state_dict"],
                fresh.state_dict(),
                "generator_e.label_embedding.",
            ),
            "residual_net": _relative_change(
                payload["model_state_dict"],
                fresh.state_dict(),
                "generator_e.residual_net.",
            ),
        },
        "hashes": {
            "generator": _hash(model, ("generator_e.",)),
            "state": _hash(model, ("",)),
        },
        "v68_closed": True,
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "checkpoint",
        "validation-dataset",
        "b-decision",
        "crossed-report",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
