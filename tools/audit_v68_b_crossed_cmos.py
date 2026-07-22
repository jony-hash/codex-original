"""Streaming crossed code×label raw-CMOS audit for v68.6 B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ldm_project import load_training_config
from ldm_project.student.checkpoints import (
    checkpoint_boundary_contract,
    load_one_shot_checkpoint,
)
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from tools.audit_v68_raw_cmos import fixed_exposure_view
from tools.diagnose_v68_transparent_memory import _build_prototype_bank
from tools.v68_b_acceptance_utils import (
    balanced_factor_decomposition,
    metric_bundle,
    validate_b_architecture,
)

REPORT_SCHEMA = "v68_b_crossed_raw_cmos_v1"
SCALE = 0.4913053810596466


def _state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _factor(values: torch.Tensor) -> dict[str, Any]:
    factors, _ = balanced_factor_decomposition(values)
    return factors


def _energy(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
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
    a2_decision: str | Path,
    output: str | Path,
    device: str = "cuda",
    batch_size: int = 1,
) -> dict[str, Any]:
    a2 = json.loads(Path(a2_decision).read_text(encoding="utf-8"))
    if (
        a2.get("status") != "a2-raw-readability-passed"
        or a2.get("b_allowed") is not True
    ):
        raise ValueError("crossed audit requires A2 b_allowed=true")
    cfg = load_training_config(config)
    validate_b_architecture(cfg)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for crossed audit")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes, original_labels = data["codes"], data["class_labels"].long()
    if (
        codes.shape[0] != 100
        or codes.ndim != 2
        or original_labels.shape[0] != 100
        or set(original_labels.tolist()) != set(range(10))
    ):
        raise ValueError(
            "crossed audit requires the 100-sample validation split with all labels"
        )
    prototypes = _build_prototype_bank(cfg).float()
    model = (
        OneShotOpticalGenerator(
            cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size
        )
        .to(dev)
        .eval()
    )
    load_one_shot_checkpoint(model, payload)
    state_before = _state_hash(model)
    plane_names = ("slm", "pre_encoder", "chip", "camera")
    plane_batches: dict[str, list[torch.Tensor]] = {name: [] for name in plane_names}
    energies: list[dict[str, float]] = []
    finite = True
    if batch_size != 1:
        raise ValueError("crossed audit is intentionally streamed at batch_size=1")
    with torch.no_grad():
        for code in codes:
            for requested_label in range(10):
                label = torch.tensor([requested_label], dtype=torch.long, device=dev)
                out_values = model(
                    code[None].to(dev), label, return_optical_fields=False
                )
                finite = finite and all(
                    bool(torch.isfinite(v).all())
                    for v in out_values.values()
                    if torch.is_tensor(v)
                )
                raw = out_values["physical_camera_intensity_raw"].detach().cpu()
                slm = out_values["slm_command"].detach().cpu()
                pre = out_values["encoder_input_grid_amplitude"].detach().cpu().square()
                chip = (
                    out_values["chip_real"].detach().cpu().square()
                    + out_values["chip_imag"].detach().cpu().square()
                )
                for name, value in (
                    ("slm", slm),
                    ("pre_encoder", pre),
                    ("chip", chip),
                    ("camera", raw),
                ):
                    plane_batches[name].append(value)
                energies.append(_energy(out_values))
    # Lists contain only active diagnostic planes, never the model's full optical field.
    planes = {
        name: torch.cat(values).reshape(100, 10, *values[0].shape[1:])
        for name, values in plane_batches.items()
    }
    camera = fixed_exposure_view(planes["camera"], SCALE)
    targets = (
        prototypes[torch.arange(10)].unsqueeze(0).expand(100, -1, *prototypes.shape[1:])
    )
    crossed_metrics = metric_bundle(
        camera.flatten(0, 1),
        targets.flatten(0, 1),
        torch.arange(10).repeat(100),
        prototypes,
    )
    diagonal = camera[torch.arange(100), original_labels]
    diagonal_targets = prototypes[original_labels]
    diagonal_metrics = metric_bundle(
        diagonal, diagonal_targets, original_labels, prototypes
    )
    chosen = torch.tensor(
        [int(torch.nonzero(original_labels == label)[0]) for label in range(10)]
    )
    preview = camera[chosen]
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
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "b-crossed-audit-complete",
        "test_split_used": False,
        "validation_sample_count": 100,
        "crossed_output_count": 1000,
        "diagonal_output_count": 100,
        "exposure_mode": "locked",
        "exposure_scale": SCALE,
        "normalization_contract": "locked_scale_gamma_1_no_smoothing",
        "crossed_metrics": crossed_metrics,
        "diagonal_metrics": diagonal_metrics,
        "factor_decomposition": {name: _factor(planes[name]) for name in plane_names},
        "camera_code_factor": _factor(planes["camera"])["code_fraction"],
        "energy_records_mean": {
            key: sum(item.get(key, 0.0) for item in energies) / max(len(energies), 1)
            for key in sorted({key for item in energies for key in item})
        },
        "finite": finite,
        "optimizer_created": False,
        "parameter_update_count": 0,
        "parameters_unchanged": state_before == _state_hash(model),
        "checkpoint_boundary_contract": checkpoint_boundary_contract(payload),
        "preview": str(preview_path),
        "preview_shape": list(preview.shape),
    }
    energy_values = [
        value
        for key, value in report["energy_records_mean"].items()
        if "guard" in key or "discarded" in key
    ]
    report["energy_pass"] = bool(energy_values) and max(energy_values) <= 1e-4
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "validation-dataset", "a2-decision", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    print(json.dumps(run(**vars(p.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
