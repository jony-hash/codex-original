"""v67 diagnostic-only convergence metrics for the transparent thin-phase model.

The CUDA forward is intentionally opt-in.  The pure metric functions are kept
small and deterministic so the acceptance gate can be tested without a GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from dataclasses import replace


def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    value, reference = value.float(), reference.float()
    return float((value - reference).norm() / reference.norm().clamp_min(1e-12))


def correlation(value: torch.Tensor, reference: torch.Tensor) -> float:
    x, y = value.float().reshape(-1), reference.float().reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-12))


def guard_band_energy(field: torch.Tensor, guard_fraction: float = 0.05) -> float:
    """Return energy outside the centered active region of a full grid."""
    if field.ndim < 2:
        raise ValueError("field must have at least two spatial dimensions")
    height, width = field.shape[-2:]
    active_h = max(1, int(round(height * (1.0 - 2.0 * guard_fraction))))
    active_w = max(1, int(round(width * (1.0 - 2.0 * guard_fraction))))
    top, left = (height - active_h) // 2, (width - active_w) // 2
    mask = torch.ones((height, width), dtype=torch.bool, device=field.device)
    mask[top : top + active_h, left : left + active_w] = False
    energy = field.abs().square()
    return float(energy[..., mask].sum() / energy.sum().clamp_min(1e-12))


def center_crop(field: torch.Tensor, size: int) -> torch.Tensor:
    height, width = field.shape[-2:]
    top, left = (height - size) // 2, (width - size) // 2
    return field[..., top : top + size, left : left + size]


def choose_grid(results: dict[int, dict[str, float]], *, reference: int = 2048) -> dict[str, Any]:
    """Select the smallest grid meeting the v67 numerical gates."""
    for grid in (1024, 1536):
        item = results.get(grid, {})
        if (
            item.get("camera_intensity_corr", 0.0) >= 0.999
            and item.get("central_field_relative_l2", float("inf")) <= 1e-3
            and item.get("energy_relative_difference", float("inf")) <= 1e-3
            and item.get("guard_band_energy_ratio", float("inf")) <= 1e-4
        ):
            return {"status": "converged", "selected_grid": grid, "reference_grid": reference}
    return {"status": "grid-not-converged", "selected_grid": None, "reference_grid": reference}


def run(*, output: str | Path, device: str = "cuda", train_config: str | Path | None = None,
        checkpoint: str | Path | None = None, sample_tensors: str | Path | None = None) -> dict[str, Any]:
    """Run the CUDA gate when all frozen diagnostic inputs are supplied."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v67 full-geometry convergence")
    report: dict[str, Any] = {
        "status": "preflight-input-required" if not (train_config and checkpoint and sample_tensors) else "running",
        "diagnostic_only": True,
        "device": str(resolved),
        "grids": [1024, 1536, 2048],
        "gates": {
            "camera_intensity_corr": 0.999,
            "central_field_relative_l2": 1e-3,
            "energy_relative_difference": 1e-3,
            "guard_band_energy_ratio": 1e-4,
        },
        "decision": "do-not-train-until-grid-results-are-recorded",
    }
    if train_config and checkpoint and sample_tensors:
        from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint, load_training_config
        payload = torch.load(sample_tensors, map_location="cpu", weights_only=False)
        codes, labels = payload["codes"], payload["class_labels"]
        base = load_training_config(train_config)
        if base.physics is None or base.one_shot.d2nn is None:
            raise ValueError("v67 requires physics and nested d2nn config")
        reference: dict[str, dict[str, float]] = {}
        raw: dict[int, dict[str, torch.Tensor]] = {}
        for grid in (1024, 1536, 2048):
            physics = replace(base.physics, physical_forward_contract=False)
            d2nn = replace(base.one_shot.d2nn, propagation_grid_size=grid, exterior_boundary="transparent_identity")
            one_shot = replace(base.one_shot, physical_forward_contract=False, d2nn=d2nn)
            model = OneShotOpticalGenerator(physics, one_shot, image_size=base.first_stage.image_size).to(resolved).eval()
            load_one_shot_checkpoint(model, torch.load(checkpoint, map_location=resolved, weights_only=False))
            cameras, chips = [], []
            with torch.no_grad():
                for code, label in zip(codes, labels):
                    result = model(code[None].to(resolved), label[None].to(resolved), return_optical_fields=False)
                    cameras.append(result["physical_camera_plane"].detach().cpu())
                    chips.append(torch.complex(result["chip_real"], result["chip_imag"]).detach().cpu())
            raw[grid] = {"camera": torch.cat(cameras), "chip": torch.cat(chips)}
            del model
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
        ref = raw[2048]
        for grid, values in raw.items():
            reference[grid] = {
                "camera_intensity_corr": correlation(values["camera"], ref["camera"]),
                "central_field_relative_l2": relative_l2(
                    center_crop(values["chip"].abs(), min(values["chip"].shape[-2:])),
                    center_crop(ref["chip"].abs(), min(values["chip"].shape[-2:])),
                ),
                "energy_relative_difference": abs(float(values["chip"].abs().square().mean() / ref["chip"].abs().square().mean().clamp_min(1e-12) - 1.0)),
                "guard_band_energy_ratio": guard_band_energy(values["chip"]),
            }
        report["results"] = reference
        decision = choose_grid(reference)
        report.update(decision)
        report["status"] = decision["status"]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-tensors", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
