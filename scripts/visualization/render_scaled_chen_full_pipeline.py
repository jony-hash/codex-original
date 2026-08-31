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
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

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


# The confirmation panel is fixed by the Scaled-Chen geometry audit.  These
# are the first occurrences of labels 0--9 in the registered held-out train
# interval; keeping the map here prevents a renderer from becoming an
# implicitly hand-picked sample selector.
DIGIT_CONFIRMATION_INDICES = {
    0: 55006,
    1: 55000,
    2: 55001,
    3: 55008,
    4: 55004,
    5: 55010,
    6: 55012,
    7: 55003,
    8: 55002,
    9: 55007,
}
FIBER_STAGE_LABELS = (
    "Encoder input",
    "Encoder L1",
    "Encoder L2 / fiber input",
    "LP01 input ports",
    "LP01 output ports",
    "Fiber output / decoder input",
    "Decoder L1",
    "Decoder L2",
    "CMOS incident field",
)
FIBER_STAGE_KEYS = (
    "encoder_input",
    "encoder_layer_1",
    "fiber_input",
    "port_input",
    "port_output",
    "fiber_output",
    "decoder_layer_1",
    "decoder_layer_2",
    "cmos_field",
)
FIBER_SPATIAL_KEYS = frozenset(
    {
        "encoder_input",
        "encoder_layer_1",
        "fiber_input",
        "fiber_output",
        "decoder_layer_1",
        "decoder_layer_2",
        "cmos_field",
    }
)
FIBER_MODAL_KEYS = frozenset({"port_input", "port_output"})
DISPLAY_ROTATED_KEYS = frozenset(
    {
        "encoder_input",
        "encoder_layer_1",
        "fiber_input",
        "port_input",
        "port_output",
        "fiber_output",
        "decoder_layer_1",
        "decoder_layer_2",
    }
)
CHECKPOINT_STEPS = (0, 1000, 5000)


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


def _load_formal_and_checkpoint(
    config_path: Path, checkpoint_step: int = 5000
) -> tuple[Any, Path, dict[str, Any], Path]:
    if checkpoint_step not in CHECKPOINT_STEPS:
        raise ValueError(f"checkpoint step must be one of {CHECKPOINT_STEPS}")
    spec = mra._load_spec(config_path)
    formal, _ = mra._load_formal(spec)
    paths = mra._source_paths(spec)
    checkpoint_key = f"scaled_chen_step_{checkpoint_step}"
    checkpoint = paths[checkpoint_key]
    payload = safe_torch_load(
        checkpoint,
        expected_sha256=mra.SOURCE_HASHES[checkpoint_key],
        map_location="cpu",
    )
    if not isinstance(payload, Mapping):
        raise ValueError("registered Scaled-Chen checkpoint is not a mapping")
    if payload.get("schema") != "scaled_chen_bogt_horizon_checkpoint.v1" or int(payload.get("global_step", -1)) != checkpoint_step:
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


def _module_state_digest(module: torch.nn.Module) -> str:
    """Return one stable digest for the frozen module state at this boundary."""
    return canonical_sha256(module.state_dict())


def _select_confirmation_digits(
    train_images: torch.Tensor, train_labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """Select the registered first-per-label confirmation samples."""
    if train_images.ndim != 4 or train_labels.ndim != 1 or train_images.shape[0] != train_labels.shape[0]:
        raise ValueError("MNIST train tensors have incompatible shapes")
    indices = tuple(DIGIT_CONFIRMATION_INDICES[digit] for digit in range(10))
    if len(set(indices)) != 10 or any(index < 55000 or index >= 60000 for index in indices):
        raise ValueError("confirmation digit indices are not distinct held-out train samples")
    index_tensor = torch.tensor(indices, dtype=torch.long)
    labels = train_labels.index_select(0, index_tensor)
    expected = torch.arange(10, dtype=torch.long)
    if not torch.equal(labels.cpu(), expected):
        raise ValueError(f"confirmation panel labels do not match 0--9: {labels.tolist()}")
    return train_images.index_select(0, index_tensor), labels, dict(DIGIT_CONFIRMATION_INDICES)


def _trace_fiber_batch(
    model: bogt.ScaledChenBOGTModel,
    amplitude: torch.Tensor,
    target: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor, float]:
    """Replay one fixed batch and capture the actual FiberArrayOLS diagnostics."""
    if amplitude.ndim != 4 or amplitude.shape[0] != 10:
        raise ValueError("fiber-chain replay requires the fixed ten-digit batch")
    slm_field = model.system.slm(model.system.command(amplitude))
    front_field = model.system.front(slm_field)
    encoded_raw, encoder_records_raw, _ = model.system.encoder(front_field, "encoder")
    encoder_records = bogt._attenuate_records(encoder_records_raw)
    encoded = bogt._attenuate(
        encoded_raw, bogt.FIELD_TRANSMISSION**2, "fixed_encoder_group_transmission"
    )
    ols_field = model.ols(encoded)
    diagnostics = dict(model.ols.last_diagnostics)
    c_in = diagnostics.get("port_input_coefficients")
    c_out = diagnostics.get("port_output_coefficients")
    if not torch.is_tensor(c_in) or not torch.is_tensor(c_out):
        raise ValueError("FiberArrayOLS did not expose port input/output coefficients")
    decoded_raw, decoder_records_raw, _ = model.system.decoder(ols_field, "decoder")
    decoder_records = bogt._attenuate_records(decoder_records_raw)
    decoded = bogt._attenuate(
        decoded_raw, bogt.FIELD_TRANSMISSION**2, "fixed_decoder_group_transmission"
    )
    chip = ComplexField(decoded.tensor, model.system.chip_plane, decoded.provenance)
    camera_field = model.chen_relay(chip)

    # The production forward call is a consistency check, not a second data
    # source.  It must match the explicit replay exactly and leaves the OLS
    # diagnostics populated by the same frozen path.
    forward = model(amplitude, return_trace=True)
    camera_intensity = camera_field.tensor.abs().square()
    consistency = float((forward.camera - camera_intensity).abs().max().detach().cpu())
    if not torch.isfinite(camera_intensity).all() or not torch.isfinite(target).all():
        raise ValueError("fiber-chain replay produced non-finite camera or target values")

    pre_values = encoded.tensor[:, 0]
    post_values = ols_field.tensor[:, 0]
    area = model.system.encoder_plane.pixel_pitch_m[0] * model.system.encoder_plane.pixel_pitch_m[1]
    input_energy = diagnostics["input_energy"].to(device=pre_values.device)
    output_energy = diagnostics["output_energy"].to(device=pre_values.device)
    coupling = diagnostics["coupling_efficiency"].to(device=pre_values.device)
    paired = torch.nn.functional.mse_loss(camera_intensity, target, reduction="none").flatten(1).mean(1)
    projection_residual = (
        (pre_values - post_values).abs().square().sum(dim=(-2, -1)) * area
        / input_energy.clamp_min(1.0e-30)
    )
    wrapped_phase = torch.angle(post_values * pre_values.conj())
    wrapped_phase_rms = wrapped_phase.square().mean(dim=(-2, -1)).sqrt()
    coefficient_delta = (c_out - c_in).norm(dim=1) / c_in.norm(dim=1).clamp_min(1.0e-30)
    power_retention = output_energy / input_energy.clamp_min(1.0e-30)
    values = {
        "encoder_input": front_field.tensor,
        "encoder_layer_1": encoder_records[0].tensor,
        "fiber_input": encoded.tensor,
        "port_input": c_in,
        "port_output": c_out,
        "fiber_output": ols_field.tensor,
        "decoder_layer_1": decoder_records[0].tensor,
        "decoder_layer_2": decoder_records[1].tensor,
        "cmos_field": camera_field.tensor,
    }
    metrics = {
        "coupling_efficiency": coupling.detach(),
        "power_retention": power_retention.detach(),
        "coefficient_delta_relative": coefficient_delta.detach(),
        "projection_residual": projection_residual.detach(),
        "wrapped_phase_rms": wrapped_phase_rms.detach(),
        "paired_mse": paired.detach(),
        "input_energy": input_energy.detach(),
        "output_energy": output_energy.detach(),
    }
    transport = model.ols.transport.detach()
    if tuple(transport.shape) != (100, 100) or not torch.is_complex(transport) or not torch.isfinite(transport).all():
        raise ValueError("FiberArrayOLS transport is not a finite complex 100x100 matrix")
    diagnostics_out: dict[str, Any] = {
        "transport": transport,
        "transport_singular_min": float(torch.linalg.svdvals(transport).min().cpu()),
        "transport_singular_max": float(torch.linalg.svdvals(transport).max().cpu()),
        "max_abs_coefficient_delta": float((c_out - c_in).abs().max().detach().cpu()),
        "max_abs_transport_minus_identity": float(
            (transport - torch.eye(100, dtype=transport.dtype, device=transport.device)).abs().max().cpu()
        ),
        "port_gram_rank": int(diagnostics["port_gram_rank"]),
        "port_gram_condition": float(diagnostics["port_gram_condition"]),
        "input_energy": input_energy.detach(),
        "output_energy": output_energy.detach(),
        "coupling_efficiency": coupling.detach(),
    }
    return values, {**diagnostics, **diagnostics_out, "metrics": metrics}, camera_intensity, consistency


def _display_array(
    value: torch.Tensor,
    sample_position: int,
    *,
    modal: bool = False,
    rotate: bool = False,
) -> np.ndarray:
    """Convert one batch member to a display-only 2-D array."""
    array, _ = _to_2d(value[sample_position : sample_position + 1], modal=modal)
    if rotate:
        array = np.rot90(array, 2).copy()
    return array


def _shared_p995(values: Sequence[np.ndarray]) -> float:
    finite_values = [np.abs(np.asarray(value)).reshape(-1) for value in values]
    joined = np.concatenate(finite_values) if finite_values else np.asarray([], dtype=np.float32)
    joined = joined[np.isfinite(joined)]
    if joined.size == 0:
        raise ValueError("cannot compute a shared display scale from empty/non-finite fields")
    return max(float(np.quantile(joined, 0.995)), 1.0e-12)


def _complex_views(
    values: Mapping[str, torch.Tensor], *, sample_positions: Sequence[int]
) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], float, float]:
    """Build shared-scale amplitude/phase views without mutating raw tensors."""
    spatial_arrays = {
        key: [_display_array(values[key], position, rotate=key in DISPLAY_ROTATED_KEYS) for position in sample_positions]
        for key in FIBER_SPATIAL_KEYS
    }
    modal_arrays = {
        key: [_display_array(values[key], position, modal=True, rotate=True) for position in sample_positions]
        for key in FIBER_MODAL_KEYS
    }
    spatial_scale = _shared_p995([item for batch in spatial_arrays.values() for item in batch])
    modal_scale = _shared_p995([item for batch in modal_arrays.values() for item in batch])
    views: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for key in FIBER_STAGE_KEYS:
        source = spatial_arrays if key in FIBER_SPATIAL_KEYS else modal_arrays
        per_sample: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for offset, raw in enumerate(source[key]):
            amplitude = np.abs(raw)
            scale = spatial_scale if key in FIBER_SPATIAL_KEYS else modal_scale
            phase = np.angle(raw)
            threshold = max(float(amplitude.max()) * 0.01, 1.0e-12)
            phase = np.where(amplitude >= threshold, phase, np.nan)
            per_sample[str(sample_positions[offset])] = (np.clip(amplitude / scale, 0.0, 1.0), phase)
        views[key] = per_sample
    return views, spatial_scale, modal_scale


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


def _panel_label(
    axis: plt.Axes,
    label: str,
    *,
    x: float = 0.0,
    y: float = 1.0,
    vertical_alignment: str = "bottom",
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va=vertical_alignment,
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
    require_panel_labels: bool = True,
    raster_dpi: int = 600,
    include_tiff: bool = True,
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
        require_panel_labels=require_panel_labels,
        strict=True,
    )
    saved: list[Path] = []
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    svg_path = output_base.with_suffix(".svg")
    tiff_path = output_base.with_suffix(".tiff")
    figure.savefig(png_path, dpi=raster_dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    saved.extend((png_path, pdf_path, svg_path))
    if include_tiff:
        figure.savefig(tiff_path, dpi=raster_dpi, bbox_inches="tight")
        saved.append(tiff_path)
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


def _imshow_phase(axis: plt.Axes, phase: np.ndarray) -> None:
    axis.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower", interpolation="nearest")


def _render_fiber_plate(
    values: Mapping[str, torch.Tensor],
    output_base: Path,
    *,
    positions: Sequence[int],
    labels: Sequence[int],
    sample_indices: Sequence[int],
    checkpoint_step: int,
    spatial_scale: float,
    modal_scale: float,
    views: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
) -> list[Path]:
    """Render the reference-style two-row-per-digit encoder/fiber/decoder plate."""
    if len(positions) != len(labels) or len(labels) != len(sample_indices):
        raise ValueError("fiber plate metadata lengths do not match")
    rows = len(positions) * 2
    columns = len(FIBER_STAGE_KEYS)
    figure, axes = plt.subplots(rows, columns, figsize=(19.0, max(7.0, rows * 0.92)), squeeze=False)
    figure.subplots_adjust(left=0.055, right=0.955, bottom=0.105, top=0.90, wspace=0.035, hspace=0.18)
    panel_ids: list[str] = []
    row_groups: list[list[str]] = []
    column_groups: list[list[str]] = [[] for _ in range(columns)]
    for row_position, (position, label, sample_index) in enumerate(zip(positions, labels, sample_indices, strict=True)):
        for phase_row in (False, True):
            row_ids: list[str] = []
            row = row_position * 2 + int(phase_row)
            for column, (key, title) in enumerate(zip(FIBER_STAGE_KEYS, FIBER_STAGE_LABELS, strict=True)):
                axis = axes[row, column]
                amplitude, phase = views[key][str(position)]
                if phase_row:
                    _imshow_phase(axis, phase)
                else:
                    axis.imshow(amplitude, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
                if row == 0:
                    axis.set_title(title, fontsize=6.4, pad=3)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                panel_id = f"digit{label}_{'phase' if phase_row else 'amplitude'}_{column}"
                panel_ids.append(panel_id)
                row_ids.append(panel_id)
                column_groups[column].append(panel_id)
                _panel_label(axis, chr(ord("a") + column), y=0.94, vertical_alignment="top")
            row_groups.append(row_ids)
        axes[row_position * 2, 0].set_ylabel(f"digit {label}\n|E|", fontsize=6.4, labelpad=3)
        axes[row_position * 2 + 1, 0].set_ylabel("phase", fontsize=6.4, labelpad=3)
        axes[row_position * 2, -1].text(
            1.10,
            0.50,
            f"idx {sample_index}",
            transform=axes[row_position * 2, -1].transAxes,
            rotation=-90,
            rotation_mode="anchor",
            va="center",
            ha="left",
            fontsize=5.5,
        )
    figure.suptitle(
        f"Scaled-Chen step-{checkpoint_step} encoder → LP01 fiber → decoder | digits {labels[0]}–{labels[-1]}",
        fontsize=10,
        y=0.955,
    )
    figure.text(
        0.5,
        0.035,
        "Spatial fields share one linear p99.5 amplitude scale across all digits and planes; LP01 ports use a common modal p99.5 scale. Phase is fixed to [−π, π] and masked below 1% of each field maximum. Intermediate planes and 10×10 port maps use a display-only 180° rotation; raw tensors and coordinates are unchanged.",
        ha="center",
        va="center",
        fontsize=6.0,
    )
    amp_cbar_ax = figure.add_axes([0.965, 0.57, 0.010, 0.22])
    phase_cbar_ax = figure.add_axes([0.965, 0.23, 0.010, 0.22])
    amp_cbar = figure.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="gray"), cax=amp_cbar_ax)
    phase_cbar = figure.colorbar(ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap="twilight"), cax=phase_cbar_ax)
    amp_cbar.set_label("shared |E| / p99.5", fontsize=5.5)
    phase_cbar.set_label("phase [rad]", fontsize=5.5)
    for cbar in (amp_cbar, phase_cbar):
        cbar.ax.tick_params(labelsize=5, width=0.5, length=2)
    return _save_figure_bundle(
        figure,
        output_base,
        axes=[axis for row in axes for axis in row],
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        colorbar_axes=[amp_cbar.ax, phase_cbar.ax],
        raster_dpi=300,
        include_tiff=False,
    )


def _phase_mask(array: np.ndarray) -> np.ndarray:
    amplitude = np.abs(array)
    threshold = max(float(amplitude.max()) * 0.01, 1.0e-12)
    return np.where(amplitude >= threshold, np.angle(array), np.nan)


def _render_fiber_cutaway(
    values: Mapping[str, torch.Tensor],
    diagnostics: Mapping[str, Any],
    output_base: Path,
    *,
    hero_position: int,
    hero_index: int,
    hero_label: int,
    checkpoint_step: int,
    spatial_scale: float,
    modal_scale: float,
    paired_mse: float,
    coupling_efficiency: float,
    power_retention: float,
    projection_residual: float,
    coefficient_delta: float,
    wrapped_phase_rms: float,
) -> list[Path]:
    """Render E_in → c_in → T → c_out → E_out → residual for the hero digit."""
    e_in = _display_array(values["fiber_input"], hero_position, rotate=True)
    c_in = _display_array(values["port_input"], hero_position, modal=True, rotate=True)
    c_out = _display_array(values["port_output"], hero_position, modal=True, rotate=True)
    e_out = _display_array(values["fiber_output"], hero_position, rotate=True)
    residual = e_in - e_out
    transport = np.asarray(diagnostics["transport"].detach().cpu().numpy())
    transport_amplitude = np.abs(transport)
    transport_phase = np.where(transport_amplitude >= 1.0e-12, np.angle(transport), np.nan)
    arrays = (e_in, c_in, transport, c_out, e_out, residual)
    titles = ("E_in", "c_in", "T", "c_out", "E_out", "E_in − E_out")
    figure, axes = plt.subplots(2, 6, figsize=(13.8, 4.6), squeeze=False)
    figure.subplots_adjust(left=0.045, right=0.94, bottom=0.22, top=0.80, wspace=0.14, hspace=0.25)
    amplitude_ids: list[str] = []
    phase_ids: list[str] = []
    for column, (array, title) in enumerate(zip(arrays, titles, strict=True)):
        amp_axis, phase_axis = axes[0, column], axes[1, column]
        if column == 2:
            amp = np.clip(transport_amplitude, 0.0, 1.0)
            phase = transport_phase
        else:
            amp = np.abs(array)
            phase = _phase_mask(array)
        scale = 1.0 if column == 2 else (modal_scale if column in {1, 3} else spatial_scale)
        amp_axis.imshow(np.clip(amp / max(scale, 1.0e-12), 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
        _imshow_phase(phase_axis, phase)
        amp_axis.set_title(title, fontsize=7, pad=4)
        for axis in (amp_axis, phase_axis):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        _panel_label(amp_axis, chr(ord("a") + column), y=0.94, vertical_alignment="top")
        _panel_label(phase_axis, chr(ord("a") + column), y=0.94, vertical_alignment="top")
        amplitude_ids.append(f"cutaway_{column}_amplitude")
        phase_ids.append(f"cutaway_{column}_phase")
    panel_ids = [*amplitude_ids, *phase_ids]
    axes[0, 0].set_ylabel("magnitude", fontsize=6.5, labelpad=3)
    axes[1, 0].set_ylabel("phase / residual", fontsize=6.5, labelpad=3)
    figure.suptitle(
        f"Digit {hero_label} (index {hero_index}) | Scaled-Chen step-{checkpoint_step} LP01 projection cutaway",
        fontsize=10,
        y=0.94,
    )
    figure.text(
        0.5,
        0.095,
        f"coupling η={coupling_efficiency:.4f}   pre→post power={power_retention:.4f}   normalized projection residual={projection_residual:.4f}   ||c_out−c_in||/||c_in||={coefficient_delta:.2e}   wrapped phase RMS={wrapped_phase_rms:.4f} rad   paired MSE={paired_mse:.5f}",
        ha="center",
        va="center",
        fontsize=6.4,
    )
    figure.text(
        0.5,
        0.045,
        "T is the registered 100×100 complex transport matrix (identity in this contract); the spatial residual is intentionally retained because LP01 projection/re-emission can discard non-modal field content. Port-map rotation is display-only.",
        ha="center",
        va="center",
        fontsize=5.9,
    )
    amp_cbar_ax = figure.add_axes([0.953, 0.50, 0.012, 0.24])
    phase_cbar_ax = figure.add_axes([0.953, 0.18, 0.012, 0.24])
    amp_cbar = figure.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="gray"), cax=amp_cbar_ax)
    phase_cbar = figure.colorbar(ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap="twilight"), cax=phase_cbar_ax)
    amp_cbar.set_label("shared/contract scale", fontsize=5.5)
    phase_cbar.set_label("phase [rad]", fontsize=5.5)
    for cbar in (amp_cbar, phase_cbar):
        cbar.ax.tick_params(labelsize=5, width=0.5, length=2)
    return _save_figure_bundle(
        figure,
        output_base,
        axes=list(axes.ravel()),
        panel_ids=panel_ids,
        row_groups=[amplitude_ids, phase_ids],
        column_groups=[[amplitude_ids[column], phase_ids[column]] for column in range(6)],
        colorbar_axes=[amp_cbar.ax, phase_cbar.ax],
        raster_dpi=600,
        include_tiff=True,
    )


def _render_fiber_metrics(
    metrics: Mapping[str, Sequence[float]], output_base: Path, *, checkpoint_step: int
) -> list[Path]:
    names = (
        ("coupling_efficiency", "LP01 coupling efficiency", (0.0, 1.0)),
        ("power_retention", "pre→post physical power retention", (0.0, 1.0)),
        ("coefficient_delta_relative", "‖c_out−c_in‖ / ‖c_in‖", None),
        ("projection_residual", "normalized projection residual", (0.0, 1.0)),
        ("wrapped_phase_rms", "wrapped phase RMS [rad]", (0.0, np.pi)),
        ("paired_mse", "paired reconstruction MSE", None),
    )
    figure, axes = plt.subplots(len(names), 1, figsize=(9.0, 9.0), sharex=True, squeeze=False)
    figure.subplots_adjust(left=0.19, right=0.97, bottom=0.08, top=0.91, hspace=0.34)
    panel_ids: list[str] = []
    x = np.arange(10)
    for row, (key, title, limits) in enumerate(names):
        axis = axes[row, 0]
        values = np.asarray(metrics[key], dtype=float)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError(f"metric {key} must contain ten finite values")
        axis.bar(x, values, color="#4c78a8", width=0.72)
        axis.set_ylabel(title, fontsize=6.7)
        if limits is not None:
            axis.set_ylim(*limits)
        elif float(np.max(np.abs(values))) == 0.0:
            axis.set_ylim(-1.0e-8, 1.0e-8)
            axis.axhline(0.0, color="#4c78a8", linewidth=0.8)
            axis.text(0.5, 0.56, "0 (identity transport)", transform=axis.transAxes, ha="center", va="center", fontsize=6.2)
        else:
            span = float(np.max(np.abs(values)))
            axis.set_ylim(min(0.0, float(values.min()) - 0.1 * span), float(values.max()) + 0.1 * span)
        axis.grid(axis="y", color="#dddddd", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.set_xticks(x)
        axis.set_xticklabels([str(digit) for digit in range(10)], fontsize=6)
        axis.tick_params(axis="y", labelsize=5.5, width=0.5, length=2)
        for spine in axis.spines.values():
            spine.set_visible(False)
        panel_id = f"metric_{key}"
        panel_ids.append(panel_id)
        _panel_label(axis, chr(ord("a") + row), x=-0.13, y=0.98, vertical_alignment="top")
    axes[-1, 0].set_xlabel("confirmation digit", fontsize=7)
    figure.suptitle(f"Scaled-Chen step-{checkpoint_step} encoder–fiber–decoder metrics", fontsize=10, y=0.96)
    figure.text(0.5, 0.025, "Values are measured from the same frozen ten-digit replay used by the amplitude/phase plates; no optimizer, evaluator, prior sample, or checkpoint write is involved.", ha="center", va="center", fontsize=6.0)
    return _save_figure_bundle(
        figure,
        output_base,
        axes=[axis[0] for axis in axes],
        panel_ids=panel_ids,
        row_groups=[],
        column_groups=[panel_ids],
        colorbar_axes=[],
        raster_dpi=300,
        include_tiff=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--sample-index", type=int, default=55008, help="confirmation-panel train index; default is a held-out digit 3")
    parser.add_argument("--view", choices=("single", "fiber-chain"), default="single", help="single legacy plate or the ten-digit encoder/fiber/decoder plate")
    parser.add_argument("--checkpoint-step", type=int, choices=CHECKPOINT_STEPS, default=5000)
    parser.add_argument("--hero-index", type=int, default=55008, help="fixed confirmation-panel index used by the fiber cutaway")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _default_output_dir(view: str) -> Path:
    if view == "fiber-chain":
        return Path("outputs/20260831_v2/visualizations/scaled_chen_encoder_fiber_decoder_step5000_v1_attempt_01")
    return Path("outputs/20260831_v2/visualizations/scaled_chen_step5000_full_pipeline_v1_attempt_01")


def _check_output_boundary(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite visualization directory: {output_dir}")
    drive = Path(output_dir.resolve().anchor or Path.cwd().anchor)
    free_bytes = shutil.disk_usage(drive).free
    if free_bytes <= 10 * 1024**3:
        raise RuntimeError(f"D: free space is below the 10 GiB visualization safety floor: {free_bytes} bytes")


def _json_metrics(metrics: Mapping[str, torch.Tensor]) -> dict[str, list[float]]:
    return {
        key: [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
        for key, value in metrics.items()
        if key in {"coupling_efficiency", "power_retention", "coefficient_delta_relative", "projection_residual", "wrapped_phase_rms", "paired_mse"}
    }


def _run_single(args: argparse.Namespace, output_dir: Path) -> int:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full optical visualization requires CUDA; CPU is not a substitute")
    if not 55000 <= args.sample_index < 60000:
        raise ValueError("sample index must belong to the registered confirmation panel 55000:60000")
    formal, checkpoint, payload, config_path = _load_formal_and_checkpoint(args.config, args.checkpoint_step)
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
    field_base = output_dir / f"scaled_chen_step{args.checkpoint_step}_full_pipeline_amplitude_phase"
    mask_base = output_dir / f"scaled_chen_step{args.checkpoint_step}_learned_phase_masks"
    saved = _render_field_figure(views, field_base, sample_index=args.sample_index, label=label, paired_mse=paired_mse)
    saved += _render_mask_figure(masks, mask_base)
    metadata = {
        "schema": "scaled_chen_full_pipeline_visualization.v1",
        "diagnostic_only": True,
        "checkpoint_step": args.checkpoint_step,
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


def _run_fiber_chain(args: argparse.Namespace, output_dir: Path) -> int:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full optical visualization requires CUDA; CPU is not a substitute")
    if args.hero_index not in DIGIT_CONFIRMATION_INDICES.values():
        raise ValueError("hero index must be one of the ten registered confirmation samples")
    _check_output_boundary(output_dir)
    formal, checkpoint, payload, config_path = _load_formal_and_checkpoint(args.config, args.checkpoint_step)
    train_images, train_labels = base.load_local_mnist(args.dataset_root, train=True)
    images, labels, fixed_indices = _select_confirmation_digits(train_images, train_labels)
    sample_indices = tuple(fixed_indices[digit] for digit in range(10))
    hero_position = sample_indices.index(args.hero_index)
    hero_label = int(labels[hero_position])
    rendered = base.render_intensity_batch(images)
    device = torch.device(args.device)
    model = mra._load_scaled_model(formal, payload, device)
    model_state_before = _module_state_digest(model)
    ols_state_before = _module_state_digest(model.ols)
    amplitude = rendered.amplitude.to(device)
    target = rendered.target.to(device)
    with torch.no_grad():
        values, diagnostics, camera_intensity, consistency = _trace_fiber_batch(model, amplitude, target)
    model_state_after = _module_state_digest(model)
    ols_state_after = _module_state_digest(model.ols)
    if model_state_before != model_state_after or ols_state_before != ols_state_after:
        raise RuntimeError("model or FiberArrayOLS state changed during read-only replay")
    if consistency > 1.0e-7:
        raise RuntimeError(f"forward consistency exceeded the visualization contract: {consistency}")
    if diagnostics["max_abs_coefficient_delta"] > 1.0e-7:
        raise RuntimeError("identity transport did not preserve LP01 coefficients")
    if abs(diagnostics["transport_singular_min"] - 1.0) > 1.0e-6 or abs(diagnostics["transport_singular_max"] - 1.0) > 1.0e-6:
        raise RuntimeError("FiberArrayOLS transport singular values are not identity")
    if torch.any(diagnostics["output_energy"] > diagnostics["input_energy"] * (1.0 + 1.0e-5) + 1.0e-12):
        raise RuntimeError("fiber output energy exceeds input energy")
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise RuntimeError("fiber-chain replay contains a non-finite field or coefficient")

    values_cpu = {key: value.detach().cpu() for key, value in values.items()}
    views, spatial_scale, modal_scale = _complex_views(values_cpu, sample_positions=range(10))
    metrics = diagnostics["metrics"]
    metric_json = _json_metrics(metrics)
    git_state = _git_state()
    script_path = Path(__file__).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    saved: list[Path] = []
    saved += _render_fiber_plate(
        values_cpu,
        output_dir / f"01_digits_0_4_encoder_fiber_decoder_amplitude_phase_step{args.checkpoint_step}",
        positions=range(5),
        labels=labels[:5].tolist(),
        sample_indices=sample_indices[:5],
        checkpoint_step=args.checkpoint_step,
        spatial_scale=spatial_scale,
        modal_scale=modal_scale,
        views=views,
    )
    saved += _render_fiber_plate(
        values_cpu,
        output_dir / f"02_digits_5_9_encoder_fiber_decoder_amplitude_phase_step{args.checkpoint_step}",
        positions=range(5, 10),
        labels=labels[5:].tolist(),
        sample_indices=sample_indices[5:],
        checkpoint_step=args.checkpoint_step,
        spatial_scale=spatial_scale,
        modal_scale=modal_scale,
        views=views,
    )
    saved += _render_fiber_cutaway(
        values_cpu,
        diagnostics,
        output_dir / f"03_digit_3_fiber_cutaway_step{args.checkpoint_step}",
        hero_position=hero_position,
        hero_index=args.hero_index,
        hero_label=hero_label,
        checkpoint_step=args.checkpoint_step,
        spatial_scale=spatial_scale,
        modal_scale=modal_scale,
        paired_mse=float(metrics["paired_mse"][hero_position].cpu()),
        coupling_efficiency=float(metrics["coupling_efficiency"][hero_position].cpu()),
        power_retention=float(metrics["power_retention"][hero_position].cpu()),
        projection_residual=float(metrics["projection_residual"][hero_position].cpu()),
        coefficient_delta=float(metrics["coefficient_delta_relative"][hero_position].cpu()),
        wrapped_phase_rms=float(metrics["wrapped_phase_rms"][hero_position].cpu()),
    )
    saved += _render_fiber_metrics(
        metric_json,
        output_dir / f"04_digits_0_9_fiber_metrics_step{args.checkpoint_step}",
        checkpoint_step=args.checkpoint_step,
    )
    masks = [
        phase.detach().float().cpu().numpy()
        for phase in (
            model.system.encoder.phase[0],
            model.system.encoder.phase[1],
            model.system.decoder.phase[0],
            model.system.decoder.phase[1],
        )
    ]
    saved += _render_mask_figure(
        masks, output_dir / f"05_scaled_chen_step{args.checkpoint_step}_learned_phase_masks"
    )

    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    output_bytes = sum(path.stat().st_size for path in output_files)
    if output_bytes > 512 * 1024**2:
        raise RuntimeError(f"visualization output exceeds the 512 MiB contract: {output_bytes} bytes")
    manifest = {
        "schema": "scaled_chen_encoder_fiber_decoder_visualization.v1",
        "view": "fiber-chain",
        "diagnostic_only": True,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_path": str(config_path.resolve()),
        "config_sha256": canonical_sha256(mra._load_spec(args.config).payload),
        "script_path": str(script_path),
        "script_sha256": file_sha256(script_path),
        "dataset_root": str(args.dataset_root.resolve()),
        "sample_split": "confirmation_panel_train_indices_55000_59999",
        "sample_indices_by_digit": {str(digit): fixed_indices[digit] for digit in range(10)},
        "labels": [int(label) for label in labels.tolist()],
        "hero": {"index": args.hero_index, "label": hero_label, "position": hero_position},
        "device": str(device),
        "stage_order": list(FIBER_STAGE_KEYS),
        "stage_labels": list(FIBER_STAGE_LABELS),
        "stage_shapes": {key: list(value.shape) for key, value in values_cpu.items()},
        "spatial_amplitude_shared_p995": spatial_scale,
        "modal_amplitude_shared_p995": modal_scale,
        "display_contract": {
            "spatial_amplitude": "absolute value, one shared linear p99.5 scale across all ten digits and spatial stages, clipped to [0,1]",
            "modal_amplitude": "absolute value, one shared modal p99.5 scale for c_in and c_out, clipped to [0,1]",
            "phase": "angle in [-pi,pi], masked below 1% of each field maximum",
            "rotation": "display-only 180-degree rotation for intermediate spatial planes and 10x10 port maps; raw tensors and physical coordinates unchanged",
            "transport": "100x100 complex matrix shown as magnitude and phase; numerical transport is not rotated or modified",
        },
        "transport_contract": {
            "shape": list(diagnostics["transport"].shape),
            "identity": True,
            "max_abs_transport_minus_identity": diagnostics["max_abs_transport_minus_identity"],
            "singular_min": diagnostics["transport_singular_min"],
            "singular_max": diagnostics["transport_singular_max"],
            "max_abs_c_out_minus_c_in": diagnostics["max_abs_coefficient_delta"],
            "port_gram_rank": diagnostics["port_gram_rank"],
            "port_gram_condition": diagnostics["port_gram_condition"],
        },
        "metrics_by_digit": metric_json,
        "forward_consistency_max_abs": consistency,
        "state_hashes": {
            "model_before": model_state_before,
            "model_after": model_state_after,
            "ols_before": ols_state_before,
            "ols_after": ols_state_after,
            "unchanged": True,
        },
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "prior_sampled": False,
        "output_bytes_before_manifest": output_bytes,
        "git": git_state,
        "outputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
    }
    metadata_path = output_dir / "visualization_manifest.json"
    metadata_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "manifest": str(metadata_path.resolve()), "checkpoint_step": args.checkpoint_step, "hero_index": args.hero_index, "hero_label": hero_label, "output_bytes": output_bytes}, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir(args.view)
    if args.view == "fiber-chain":
        return _run_fiber_chain(args, output_dir)
    _check_output_boundary(output_dir)
    return _run_single(args, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
