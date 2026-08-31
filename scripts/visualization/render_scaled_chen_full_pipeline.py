"""Render a read-only amplitude/phase plate for the frozen Scaled-Chen route.

The renderer is diagnostic only.  It loads the registered step-5000 checkpoint,
replays one confirmation-panel sample through the same fixed optical path, and
writes image exports plus compact provenance.  It never constructs an
optimizer, calls an evaluator, performs backward, samples the prior, or writes
model/checkpoint artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.application.research import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research import optical_manifold_smf_bogt_manifold_readiness as mra
from ldm_project.v2.contracts import ComplexField
from ldm_project.v2.evidence import canonical_sha256, file_sha256, safe_torch_load

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError as exc:  # pragma: no cover - exercised by the CLI environment
    raise RuntimeError(
        "add the nature-figure scripts directory to PYTHONPATH before rendering"
    ) from exc


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


STAGE_LABELS = (
    "SLM command",
    "SLM output",
    "Encoder input",
    "Encoder L1",
    "Pre-OLS",
    "Modal 10×10",
    "Post-OLS",
    "Decoder L1",
    "Decoder L2",
    "CMOS field",
)
STAGE_KEYS = (
    "slm_command",
    "slm_output",
    "encoder_input",
    "encoder_layer_1",
    "pre_ols",
    "modal_complex",
    "post_ols",
    "decoder_layer_1",
    "decoder_layer_2",
    "cmos_field",
)


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"root": str(root), "commit": commit, "clean": not status, "status": status}


def _load_formal_and_checkpoint(config_path: Path) -> tuple[Any, Path, dict[str, Any], Path]:
    spec = mra._load_spec(config_path)
    formal, _ = mra._load_formal(spec)
    paths = mra._source_paths(spec)
    checkpoint = paths["scaled_chen_step_5000"]
    payload = safe_torch_load(
        checkpoint,
        expected_sha256=mra.SOURCE_HASHES["scaled_chen_step_5000"],
        map_location="cpu",
    )
    if not isinstance(payload, Mapping):
        raise ValueError("registered Scaled-Chen checkpoint is not a mapping")
    if payload.get("schema") != "scaled_chen_bogt_horizon_checkpoint.v1" or int(payload.get("global_step", -1)) != 5000:
        raise ValueError("registered Scaled-Chen checkpoint schema or step is invalid")
    return formal, checkpoint, dict(payload), spec.path


def _trace_model(model: bogt.ScaledChenBOGTModel, amplitude: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, float]:
    """Replay the production research forward path and retain only one sample."""
    slm_field = model.system.slm(model.system.command(amplitude))
    front_field = model.system.front(slm_field)
    encoded_raw, encoder_records_raw, _ = model.system.encoder(front_field, "encoder")
    encoder_records = bogt._attenuate_records(encoder_records_raw)
    encoded = bogt._attenuate(encoded_raw, bogt.FIELD_TRANSMISSION**2, "fixed_encoder_group_transmission")
    ols_field = model.ols(encoded)
    coefficients = model._coefficients(encoded)
    decoded_raw, decoder_records_raw, _ = model.system.decoder(ols_field, "decoder")
    decoder_records = bogt._attenuate_records(decoder_records_raw)
    decoded = bogt._attenuate(decoded_raw, bogt.FIELD_TRANSMISSION**2, "fixed_decoder_group_transmission")
    chip = ComplexField(decoded.tensor, model.system.chip_plane, decoded.provenance)
    camera_field = model.chen_relay(chip)

    forward = model(amplitude, return_trace=True)
    consistency = float((forward.camera - camera_field.tensor.abs().square()).abs().max().detach().cpu())
    stages = {
        "slm_command": amplitude,
        "slm_output": slm_field.tensor,
        "encoder_input": front_field.tensor,
        "encoder_layer_1": encoder_records[0].tensor,
        "pre_ols": encoder_records[1].tensor,
        "modal_complex": coefficients,
        "post_ols": ols_field.tensor,
        "decoder_layer_1": decoder_records[0].tensor,
        "decoder_layer_2": decoder_records[1].tensor,
        "cmos_field": camera_field.tensor,
    }
    return stages, camera_field.tensor.abs().square(), consistency


def _to_2d(value: torch.Tensor, *, modal: bool = False) -> tuple[np.ndarray, bool]:
    tensor = value.detach()
    if tensor.ndim == 4:
        tensor = tensor[0, 0]
    elif tensor.ndim == 3:
        tensor = tensor[0]
    elif tensor.ndim == 2 and modal and tensor.shape == (1, 100):
        tensor = tensor[0].reshape(10, 10)
    elif tensor.ndim == 1 and modal and tensor.shape == (100,):
        tensor = tensor.reshape(10, 10)
    if tensor.ndim != 2:
        raise ValueError(f"visualization stage must reduce to 2D, got {tuple(tensor.shape)}")
    is_complex = bool(torch.is_complex(tensor))
    array = tensor.cpu().numpy() if is_complex else tensor.float().cpu().numpy()
    return array, is_complex


def _stage_views(value: torch.Tensor, *, modal: bool = False) -> tuple[np.ndarray, np.ndarray, float, float]:
    array, is_complex = _to_2d(value, modal=modal)
    amplitude = np.abs(array)
    vmax = float(np.quantile(amplitude, 0.995)) if np.any(amplitude > 0) else 1.0
    vmax = max(vmax, float(amplitude.max()), 1.0e-12)
    if is_complex:
        phase = np.angle(array)
        threshold = max(float(amplitude.max()) * 0.01, 1.0e-12)
        phase = np.where(amplitude >= threshold, phase, np.nan)
        valid_fraction = float(np.isfinite(phase).mean())
    else:
        phase = np.full_like(amplitude, np.nan, dtype=np.float32)
        valid_fraction = 0.0
    return np.clip(amplitude / vmax, 0.0, 1.0), phase, vmax, valid_fraction


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.0,
        1.0,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="black",
    )


def _save_figure_bundle(
    figure: plt.Figure,
    output_base: Path,
    *,
    axes: list[plt.Axes],
    panel_ids: list[str],
    row_groups: list[list[str]],
    column_groups: list[list[str]],
    colorbar_axes: list[plt.Axes],
) -> list[Path]:
    alignment_path = output_base.with_suffix(".alignment.json")
    alignment_overlay = output_base.with_suffix(".alignment.svg")
    require_matplotlib_panel_alignment(
        figure,
        axes=axes,
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        exclude_axes=colorbar_axes,
        json_out=str(alignment_path),
        overlay_svg=str(alignment_overlay),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    saved: list[Path] = []
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    svg_path = output_base.with_suffix(".svg")
    tiff_path = output_base.with_suffix(".tiff")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(tiff_path, dpi=600, bbox_inches="tight")
    saved.extend((png_path, pdf_path, svg_path, tiff_path))
    plt.close(figure)
    return saved


def _render_field_figure(
    views: list[tuple[np.ndarray, np.ndarray]],
    output_base: Path,
    *,
    sample_index: int,
    label: int,
    paired_mse: float,
) -> list[Path]:
    figure, axes = plt.subplots(2, len(views), figsize=(22.0, 5.4), squeeze=False)
    figure.subplots_adjust(left=0.018, right=0.958, bottom=0.16, top=0.82, wspace=0.07, hspace=0.26)
    for column, ((amplitude, phase), title) in enumerate(zip(views, STAGE_LABELS, strict=True)):
        upper, lower = axes[0, column], axes[1, column]
        upper.imshow(amplitude, cmap="magma", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
        lower.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower", interpolation="nearest")
        upper.set_title(title, fontsize=7, pad=4)
        for axis in (upper, lower):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    for index, axis in enumerate(axes.ravel()):
        _panel_label(axis, chr(ord("a") + index))
    axes[0, 0].set_ylabel("Amplitude\n(normalized per stage)", fontsize=7, labelpad=5)
    axes[1, 0].set_ylabel("Phase\n(radians)", fontsize=7, labelpad=5)
    figure.suptitle(
        f"Scaled-Chen step-5000 full optical path | confirmation train index {sample_index} | digit {label} | paired MSE {paired_mse:.5f}",
        fontsize=10,
        y=0.96,
    )
    figure.text(
        0.5,
        0.045,
        "Amplitude uses per-stage 99.5th-percentile display scaling; phase is masked below 1% of each stage maximum. Modal panel is the complex 100-port coefficient bank reshaped to 10×10.",
        ha="center",
        va="center",
        fontsize=6.5,
    )
    amp_cbar_ax = figure.add_axes([0.968, 0.53, 0.012, 0.23])
    phase_cbar_ax = figure.add_axes([0.968, 0.20, 0.012, 0.23])
    amp_cbar = figure.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma"), cax=amp_cbar_ax)
    phase_cbar = figure.colorbar(ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap="twilight"), cax=phase_cbar_ax)
    amp_cbar.set_label("display amplitude", fontsize=6)
    phase_cbar.set_label("phase [rad]", fontsize=6)
    for cbar in (amp_cbar, phase_cbar):
        cbar.ax.tick_params(labelsize=5, width=0.5, length=2)
    return _save_figure_bundle(
        figure,
        output_base,
        axes=list(axes.ravel()),
        panel_ids=[chr(ord("a") + index) for index in range(len(views) * 2)],
        row_groups=[[chr(ord("a") + index) for index in range(len(views))], [chr(ord("a") + len(views) + index) for index in range(len(views))]],
        column_groups=[[chr(ord("a") + column), chr(ord("a") + len(views) + column)] for column in range(len(views))],
        colorbar_axes=[amp_cbar.ax, phase_cbar.ax],
    )


def _render_mask_figure(masks: list[np.ndarray], output_base: Path) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(6.4, 6.2), squeeze=False)
    figure.subplots_adjust(left=0.07, right=0.88, bottom=0.14, top=0.84, wspace=0.18, hspace=0.24)
    labels = ("Encoder phase L1", "Encoder phase L2", "Decoder phase L1", "Decoder phase L2")
    for column, (mask, title) in enumerate(zip(masks, labels, strict=True)):
        axis = axes.ravel()[column]
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"phase mask must reduce to 2D, got {mask.shape}")
        axis.imshow(mask, cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower", interpolation="nearest")
        axis.set_title(title, fontsize=7, pad=4)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        _panel_label(axis, chr(ord("a") + column))
    figure.suptitle("Scaled-Chen step-5000 learned phase masks", fontsize=10, y=0.94)
    figure.text(0.5, 0.045, "Phase masks are shown in radians on the fixed 100×100 metasurface grid; these are learned optical parameters, not generated samples.", ha="center", va="center", fontsize=6.5)
    cbar_ax = figure.add_axes([0.91, 0.20, 0.022, 0.56])
    cbar = figure.colorbar(ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap="twilight"), cax=cbar_ax)
    cbar.set_label("phase [rad]", fontsize=6)
    cbar.ax.tick_params(labelsize=5, width=0.5, length=2)
    return _save_figure_bundle(
        figure,
        output_base,
        axes=list(axes.ravel()),
        panel_ids=["a", "b", "c", "d"],
        row_groups=[["a", "b"], ["c", "d"]],
        column_groups=[["a", "c"], ["b", "d"]],
        colorbar_axes=[cbar.ax],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--sample-index", type=int, default=55008, help="confirmation-panel train index; default is a held-out digit 3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/20260831_v2/visualizations/scaled_chen_step5000_full_pipeline_v1_attempt_01"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite visualization directory: {output_dir}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full optical visualization requires CUDA; CPU is not a substitute")
    if not 55000 <= args.sample_index < 60000:
        raise ValueError("sample index must belong to the registered confirmation panel 55000:60000")
    formal, checkpoint, payload, config_path = _load_formal_and_checkpoint(args.config)
    train_images, train_labels = base.load_local_mnist(args.dataset_root, train=True)
    image = train_images[args.sample_index : args.sample_index + 1]
    label = int(train_labels[args.sample_index])
    rendered = base.render_intensity_batch(image)
    device = torch.device(args.device)
    model = mra._load_scaled_model(formal, payload, device)
    amplitude = rendered.amplitude.to(device)
    with torch.no_grad():
        stages, camera_intensity, consistency = _trace_model(model, amplitude)
    target = rendered.target.to(device)
    paired_mse = float(torch.nn.functional.mse_loss(camera_intensity, target).detach().cpu())

    views: list[tuple[np.ndarray, np.ndarray]] = []
    stage_stats: dict[str, Any] = {}
    for key, value in zip(STAGE_KEYS, (stages[name] for name in STAGE_KEYS), strict=True):
        amplitude_view, phase_view, vmax, valid_fraction = _stage_views(value, modal=key == "modal_complex")
        views.append((amplitude_view, phase_view))
        stage_stats[key] = {
            "shape": list(value.shape),
            "complex": bool(torch.is_complex(value)),
            "display_amplitude_vmax": vmax,
            "phase_valid_fraction": valid_fraction,
            "mean_amplitude": float(value.detach().abs().float().mean().cpu()),
            "power": float(value.detach().abs().square().float().mean().cpu()),
        }

    masks = [
        phase.detach().float().cpu().numpy()
        for phase in (
            model.system.encoder.phase[0],
            model.system.encoder.phase[1],
            model.system.decoder.phase[0],
            model.system.decoder.phase[1],
        )
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    field_base = output_dir / "scaled_chen_step5000_full_pipeline_amplitude_phase"
    mask_base = output_dir / "scaled_chen_step5000_learned_phase_masks"
    saved = _render_field_figure(views, field_base, sample_index=args.sample_index, label=label, paired_mse=paired_mse)
    saved += _render_mask_figure(masks, mask_base)
    metadata = {
        "schema": "scaled_chen_step5000_full_pipeline_visualization.v1",
        "diagnostic_only": True,
        "checkpoint_step": 5000,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_path": str(config_path.resolve()),
        "config_sha256": canonical_sha256(mra._load_spec(args.config).payload),
        "dataset_root": str(args.dataset_root.resolve()),
        "sample_index": args.sample_index,
        "sample_split": "confirmation_panel_train_indices_55000_59999",
        "label": label,
        "device": str(device),
        "forward_consistency_max_abs": consistency,
        "paired_mse": paired_mse,
        "stage_order": list(STAGE_KEYS),
        "stage_stats": stage_stats,
        "phase_mask_shapes": [list(mask.shape) for mask in masks],
        "display_contract": {
            "amplitude": "absolute value, normalized independently by each stage 99.5th percentile then clipped to [0,1]",
            "phase": "angle in [-pi,pi], masked below 1% of stage maximum",
            "modal": "complex 100-port coefficients reshaped to 10x10",
        },
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "prior_sampled": False,
        "git": _git_state(),
        "outputs": [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in saved],
    }
    metadata_path = output_dir / "visualization_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "manifest": str(metadata_path.resolve()), "paired_mse": paired_mse, "sample_index": args.sample_index, "label": label}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
