"""Read-only v70 SLM code-transport upper-bound isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ldm_project import load_training_config
from ldm_project.student.checkpoints import checkpoint_boundary_contract, load_one_shot_checkpoint
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from tools.audit_v68_raw_cmos import fixed_exposure_view
from tools.diagnose_v68_transparent_memory import _build_prototype_bank
from tools.v68_b_acceptance_utils import metric_bundle, balanced_factor_decomposition
from tools.audit_v69_code_distill import _StreamingFactor

REPORT_SCHEMA = "v70_slm_transport_isolation_v1"
SCALE = 0.4913053810596466
TARGET_SLM_CODE_FACTOR = 0.25


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_lowfreq_code_pattern(codes: torch.Tensor, size: int = 200) -> torch.Tensor:
    """Match the existing experimental lowfreq_additive override exactly."""
    coords = torch.linspace(-torch.pi, torch.pi, size, dtype=codes.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    basis = torch.stack(
        [
            torch.sin(xx), torch.cos(xx), torch.sin(yy), torch.cos(yy),
            torch.sin(xx + yy), torch.cos(xx - yy), torch.sin(2.0 * xx + yy),
            torch.cos(xx + 2.0 * yy),
        ], dim=0,
    )
    features = torch.tanh(codes[:, :8]).reshape(-1, 8, 1, 1)
    pattern = (features * basis).sum(dim=1, keepdim=True)
    pattern = F.avg_pool2d(pattern, kernel_size=7, stride=1, padding=3)
    maximum = pattern.flatten(1).abs().amax(1).reshape(-1, 1, 1, 1).clamp_min(1e-6)
    return pattern / maximum


def calibrate_override_amplitude(
    base_command: torch.Tensor,
    pattern: torch.Tensor,
    *,
    iterations: int = 30,
    target: float = TARGET_SLM_CODE_FACTOR,
) -> dict[str, Any]:
    """Calibrate one amplitude using generator-only bisection, never optics."""
    if base_command.shape != pattern.shape or base_command.ndim != 5:
        raise ValueError("expected balanced [100,10,1,H,W] command and pattern")

    def evaluate(amplitude: float) -> tuple[float, torch.Tensor]:
        command = (base_command + amplitude * pattern).clamp(0.0, 1.0)
        factor = balanced_factor_decomposition(command)[0]["code_fraction"]
        return factor, command

    low, high = 0.0, 1.0
    high_factor, _ = evaluate(high)
    while high_factor < target and high < 16.0:
        high *= 2.0
        high_factor, _ = evaluate(high)
    if high_factor < target:
        raise ValueError("lowfreq override cannot reach the target SLM code factor")
    for _ in range(iterations):
        middle = (low + high) / 2.0
        factor, _ = evaluate(middle)
        if factor < target:
            low = middle
        else:
            high = middle
    factor, command = evaluate(high)
    return {
        "target_code_fraction": target,
        "amplitude": high,
        "iterations": iterations,
        "achieved_code_fraction": factor,
        "absolute_error": abs(factor - target),
        "finite": bool(torch.isfinite(command).all()),
        "zero_fraction": float((command == 0).float().mean()),
        "one_fraction": float((command == 1).float().mean()),
    }


def _load_generator(cfg: Any, checkpoint: str | Path) -> torch.nn.Module:
    from ldm_project.student.generator import ElectronicGeneratorE

    generator = ElectronicGeneratorE(
        image_size=cfg.first_stage.image_size,
        grid_size=cfg.physics.geometry.grid_size,
        code_dim=cfg.one_shot.code_dim,
        num_classes=cfg.one_shot.num_classes,
        label_embed_dim=cfg.one_shot.label_embed_dim,
        hidden_dim=cfg.one_shot.hidden_dim,
        arch=cfg.one_shot.generator_e_arch,
        code_residual_scale=cfg.one_shot.code_residual_scale,
        command_transform=cfg.one_shot.generator_command_transform,
        background_floor=cfg.one_shot.slm_background_floor,
        template_init=cfg.one_shot.generator_template_init,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    generator.load_state_dict(
        {key.removeprefix("generator_e."): value for key, value in state.items() if key.startswith("generator_e.")}
    )
    return generator.eval()


def _energy(outputs: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value.detach().float().mean())
        for key, value in outputs.items()
        if torch.is_tensor(value)
        and (key.endswith("energy_ratio") or key.endswith("guard_band_energy_ratio"))
    }


def run(
    *,
    config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    v69_decision: str | Path,
    v69_crossed_report: str | Path,
    output: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    decision = json.loads(Path(v69_decision).read_text(encoding="utf-8"))
    crossed = json.loads(Path(v69_crossed_report).read_text(encoding="utf-8"))
    if decision.get("status") != "v69-prototype-only":
        raise ValueError("v70 requires the frozen v69-prototype-only decision")
    if crossed.get("status") != "v69-crossed-audit-complete" or crossed.get("test_split_used"):
        raise ValueError("v70 requires a valid v69 crossed report")
    cfg = load_training_config(config)
    if cfg.one_shot.generator_e_arch != "label_template_code_delta":
        raise ValueError("v70 requires the frozen v69 code-delta config")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes, labels = data["codes"].float(), data["class_labels"].long()
    if codes.shape != (100, 128) or labels.shape != (100,) or set(labels.tolist()) != set(range(10)):
        raise ValueError("v70 requires the fixed 100-sample validation split")
    prototypes = _build_prototype_bank(cfg).float()
    generator = _load_generator(cfg, checkpoint)
    with torch.no_grad():
        all_codes = codes[:, None, :].expand(100, 10, 128).reshape(1000, 128)
        requested = torch.arange(10).repeat(100)
        _, diagnostics = generator.forward_with_diagnostics(all_codes, requested)
        base = diagnostics["generator_base_command"].reshape(100, 10, 1, 200, 200)
    pattern = build_lowfreq_code_pattern(codes).reshape(100, 1, 1, 200, 200).expand(100, 10, 1, 200, 200)
    calibration = calibrate_override_amplitude(base, pattern)
    overridden_one_shot = replace(
        cfg.one_shot,
        diagnostic_slm_override_mode="lowfreq_additive",
        diagnostic_slm_override_amplitude=calibration["amplitude"],
    )
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v70 isolation requires CUDA")
    model = OneShotOpticalGenerator(cfg.physics, overridden_one_shot, image_size=cfg.first_stage.image_size).to(dev).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_one_shot_checkpoint(model, payload)
    state_before = _state_hash(model)
    streams: dict[str, _StreamingFactor] = {}
    camera_raw = torch.empty((100, 10, 1, 200, 200), dtype=torch.float32)
    preview_raw = torch.empty((10, 10, 1, 200, 200), dtype=torch.float32)
    chosen = [int(torch.nonzero(labels == label, as_tuple=False)[0]) for label in range(10)]
    chosen_lookup = {index: row for row, index in enumerate(chosen)}
    energies: list[dict[str, float]] = []
    finite = True
    with torch.no_grad():
        for code_index, code in enumerate(codes):
            for requested_label in range(10):
                label = torch.tensor([requested_label], device=dev, dtype=torch.long)
                outputs = model(code[None].to(dev), label, return_optical_fields=False)
                finite = finite and all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_tensor(value))
                planes = {
                    "slm": outputs["slm_command"],
                    "pre_encoder": outputs["encoder_input_grid_amplitude"].square(),
                    "chip": outputs["chip_real"].square() + outputs["chip_imag"].square(),
                    "camera": outputs["physical_camera_intensity_raw"],
                }
                for name, value in planes.items():
                    streams.setdefault(name, _StreamingFactor(tuple(value.shape[1:]))).add(code_index, requested_label, value)
                raw = planes["camera"].detach().cpu()
                camera_raw[code_index, requested_label] = raw
                if code_index in chosen_lookup:
                    preview_raw[chosen_lookup[code_index], requested_label] = raw
                energies.append(_energy(outputs))
    camera = fixed_exposure_view(camera_raw, SCALE)
    targets = prototypes[None].expand(100, -1, *prototypes.shape[1:])
    crossed_metrics = metric_bundle(camera.flatten(0, 1), targets.flatten(0, 1), torch.arange(10).repeat(100), prototypes)
    diagonal = camera[torch.arange(100), labels]
    diagonal_metrics = metric_bundle(diagonal, prototypes[labels], labels, prototypes)
    factors = {name: stream.report() for name, stream in streams.items()}
    below = [name for name in ("pre_encoder", "chip", "camera") if factors[name]["code_fraction"] < TARGET_SLM_CODE_FACTOR]
    preview = fixed_exposure_view(preview_raw, SCALE)
    preview_path = output_path.with_name(output_path.stem + "_raw_preview.png")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(10, 10, figsize=(12, 12), squeeze=False)
        for row in range(10):
            for col in range(10):
                axes[row][col].imshow(preview[row, col, 0], cmap="gray", vmin=0, vmax=1)
                axes[row][col].axis("off")
        fig.tight_layout()
        fig.savefig(preview_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        preview_path = Path(str(preview_path) + ".unavailable")
        preview_path.write_text(str(exc), encoding="utf-8")
    energy_mean = {key: sum(item.get(key, 0.0) for item in energies) / len(energies) for key in sorted({key for item in energies for key in item})}
    energy_values = [value for key, value in energy_mean.items() if "guard" in key or "discarded" in key]
    report = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "v70-slm-intervention-complete",
        "v69_decision": str(v69_decision),
        "v69_crossed_report": str(v69_crossed_report),
        "test_split_used": False,
        "validation_sample_count": 100,
        "crossed_output_count": 1000,
        "batch_size": 1,
        "return_optical_fields": False,
        "override_mode": "lowfreq_additive",
        "exposure_mode": "locked",
        "exposure_scale": SCALE,
        "calibration": calibration,
        "factor_decomposition": factors,
        "slm_code_factor": factors["slm"]["code_fraction"],
        "camera_code_factor": factors["camera"]["code_fraction"],
        "first_plane_below_25_percent": below[0] if below else None,
        "crossed_metrics": crossed_metrics,
        "diagonal_metrics": diagonal_metrics,
        "energy_records_mean": energy_mean,
        "energy_pass": bool(energy_values) and max(energy_values) <= 1e-4,
        "finite": finite and all(item["finite"] for item in factors.values()),
        "parameters_unchanged": state_before == _state_hash(model),
        "optimizer_created": False,
        "parameter_update_count": 0,
        "checkpoint_boundary_contract": checkpoint_boundary_contract(payload),
        "preview": str(preview_path),
        "preview_shape": list(preview.shape),
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "validation-dataset", "v69-decision", "v69-crossed-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
