"""Streaming crossed raw-CMOS audit for the v69 code-distillation contract."""

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
from tools.v68_b_acceptance_utils import metric_bundle, sample_correlation, sample_dice

REPORT_SCHEMA = "v69_crossed_raw_cmos_v1"
SCALE = 0.4913053810596466


class _StreamingFactor:
    """Exact balanced two-factor ANOVA without retaining optical fields."""

    def __init__(self, sample_shape: tuple[int, ...]) -> None:
        self.total = torch.zeros(sample_shape, dtype=torch.float64)
        self.code = torch.zeros((100, *sample_shape), dtype=torch.float64)
        self.label = torch.zeros((10, *sample_shape), dtype=torch.float64)
        self.sum_sq = 0.0

    def add(self, code_index: int, label: int, value: torch.Tensor) -> None:
        x = value.detach().cpu().double().squeeze(0)
        self.total += x
        self.code[code_index] += x
        self.label[label] += x
        self.sum_sq += float(x.square().sum())

    def report(self) -> dict[str, Any]:
        grand = self.total / 1000.0
        code_mean = self.code / 10.0
        label_mean = self.label / 100.0
        total_ss = self.sum_sq - 1000.0 * float(grand.square().sum())
        code_ss = 10.0 * float((code_mean - grand).square().sum())
        label_ss = 100.0 * float((label_mean - grand).square().sum())
        interaction_ss = max(total_ss - code_ss - label_ss, 0.0)
        denom = max(total_ss, 1.0e-12)
        return {
            "total_energy": float(total_ss),
            "code_fraction": float(code_ss / denom),
            "label_fraction": float(label_ss / denom),
            "interaction_fraction": float(interaction_ss / denom),
            "finite": bool(
                torch.isfinite(self.total).all()
                and torch.isfinite(self.code).all()
                and torch.isfinite(self.label).all()
            ),
        }


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _energy(outputs: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value.detach().float().mean())
        for key, value in outputs.items()
        if torch.is_tensor(value)
        and (key.endswith("energy_ratio") or key.endswith("guard_band_energy_ratio"))
    }


def _validate_contract(cfg: Any) -> None:
    one_shot = cfg.one_shot
    if one_shot.generator_e_arch != "label_template_code_delta":
        raise ValueError("v69 audit requires label_template_code_delta")
    if float(one_shot.code_residual_scale) != 0.0:
        raise ValueError("v69 audit requires code_residual_scale=0.0")
    if one_shot.distill_target_source != "teacher_camera":
        raise ValueError("v69 audit requires teacher_camera targets")
    if one_shot.diagnostic_camera_normalization != "fixed_exposure":
        raise ValueError("v69 audit requires fixed exposure")
    if abs(float(one_shot.diagnostic_camera_exposure_scale) - SCALE) > 1e-12:
        raise ValueError("v69 audit requires the registered locked exposure scale")


def _paired_teacher_alignment(
    diagonal: torch.Tensor, teacher: torch.Tensor, labels: torch.Tensor
) -> dict[str, Any]:
    own = sample_correlation(diagonal, teacher)
    advantages: list[torch.Tensor] = []
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if len(indices) != 10:
            raise ValueError(
                "paired teacher gate requires ten validation codes per label"
            )
        student = diagonal.index_select(0, indices)
        targets = teacher.index_select(0, indices)
        matrix = sample_correlation(
            student[:, None].expand(-1, 10, -1, -1, -1).flatten(0, 1),
            targets[None].expand(10, -1, -1, -1, -1).flatten(0, 1),
        ).reshape(10, 10)
        shuffled_mean = (matrix.sum(1) - matrix.diag()) / 9.0
        advantages.append(matrix.diag() - shuffled_mean)
    advantage = torch.cat(advantages)
    generator = torch.Generator().manual_seed(1234)
    bootstrap = advantage[
        torch.randint(len(advantage), (10_000, len(advantage)), generator=generator)
    ].mean(1)
    return {
        "diagonal_teacher_corr": float(own.mean()),
        "diagonal_teacher_dice": float(sample_dice(diagonal, teacher).mean()),
        "teacher_advantage_mean": float(advantage.mean()),
        "teacher_advantage_ci95_lower": float(torch.quantile(bootstrap, 0.025)),
        "teacher_own_beats_shuffled_fraction": float((advantage > 0).float().mean()),
        "bootstrap_seed": 1234,
        "bootstrap_resamples": 10_000,
        "finite": bool(
            torch.isfinite(advantage).all() and torch.isfinite(bootstrap).all()
        ),
    }


def run(
    *,
    config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    warmstart_report: str | Path,
    output: str | Path,
    device: str = "cuda",
    batch_size: int = 1,
) -> dict[str, Any]:
    smoke = json.loads(Path(warmstart_report).read_text(encoding="utf-8"))
    if smoke.get("status") != "v69-warmstart-smoke-passed":
        raise ValueError("v69 crossed audit requires the passed v69 warm-start smoke")
    if batch_size != 1:
        raise ValueError("v69 crossed audit is intentionally streamed at batch_size=1")
    cfg = load_training_config(config)
    _validate_contract(cfg)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v69 crossed audit")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes, original_labels = data["codes"], data["class_labels"].long()
    teacher = data["teacher_camera"].float().clamp(0.0, 1.0)
    if (
        tuple(codes.shape[:1]) != (100,)
        or tuple(original_labels.shape) != (100,)
        or tuple(teacher.shape[:2]) != (100, 1)
        or set(original_labels.tolist()) != set(range(10))
    ):
        raise ValueError("v69 requires the fixed 100-sample validation split")
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
    camera_raw = torch.empty((100, 10, *teacher.shape[1:]), dtype=torch.float32)
    preview_raw = torch.empty((10, 10, *teacher.shape[1:]), dtype=torch.float32)
    factor_streams: dict[str, _StreamingFactor] = {}
    energies: list[dict[str, float]] = []
    finite = True
    chosen = [
        int(torch.nonzero(original_labels == label, as_tuple=False)[0])
        for label in range(10)
    ]
    chosen_lookup = {index: row for row, index in enumerate(chosen)}
    with torch.no_grad():
        for code_index, code in enumerate(codes):
            for requested_label in range(10):
                label = torch.tensor([requested_label], device=dev, dtype=torch.long)
                outputs = model(code[None].to(dev), label, return_optical_fields=False)
                finite = finite and all(
                    bool(torch.isfinite(value).all())
                    for value in outputs.values()
                    if torch.is_tensor(value)
                )
                planes = {
                    "code_delta_logits": outputs["generator_code_delta_logits"],
                    "slm": outputs["slm_command"],
                    "pre_encoder": outputs["encoder_input_grid_amplitude"].square(),
                    "chip": outputs["chip_real"].square()
                    + outputs["chip_imag"].square(),
                    "camera": outputs["physical_camera_intensity_raw"],
                }
                for name, value in planes.items():
                    factor_streams.setdefault(
                        name, _StreamingFactor(tuple(value.shape[1:]))
                    ).add(code_index, requested_label, value)
                raw = planes["camera"].detach().cpu()
                camera_raw[code_index, requested_label] = raw
                if code_index in chosen_lookup:
                    preview_raw[chosen_lookup[code_index], requested_label] = raw
                energies.append(_energy(outputs))
    camera = fixed_exposure_view(camera_raw, SCALE)
    targets = prototypes[None].expand(100, -1, *prototypes.shape[1:])
    crossed_metrics = metric_bundle(
        camera.flatten(0, 1),
        targets.flatten(0, 1),
        torch.arange(10).repeat(100),
        prototypes,
    )
    diagonal = camera[torch.arange(100), original_labels]
    diagonal_metrics = metric_bundle(
        diagonal, prototypes[original_labels], original_labels, prototypes
    )
    paired = _paired_teacher_alignment(diagonal, teacher, original_labels)
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
    except Exception as exc:  # Preview availability must not change raw statistics.
        preview_path = Path(str(preview_path) + ".unavailable")
        preview_path.write_text(str(exc), encoding="utf-8")
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "v69-crossed-audit-complete",
        "test_split_used": False,
        "validation_sample_count": 100,
        "crossed_output_count": 1000,
        "diagonal_output_count": 100,
        "batch_size": 1,
        "return_optical_fields": False,
        "exposure_mode": "locked",
        "exposure_scale": SCALE,
        "normalization_contract": "locked_scale_gamma_1_no_smoothing",
        "crossed_metrics": crossed_metrics,
        "diagonal_metrics": diagonal_metrics,
        "paired_teacher_alignment": paired,
        "factor_decomposition": {
            name: stream.report() for name, stream in factor_streams.items()
        },
        "camera_code_factor": factor_streams["camera"].report()["code_fraction"],
        "energy_records_mean": {
            key: sum(item.get(key, 0.0) for item in energies) / len(energies)
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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "checkpoint",
        "validation-dataset",
        "warmstart-report",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
