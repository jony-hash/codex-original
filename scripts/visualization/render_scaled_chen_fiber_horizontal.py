"""Render the approved frozen SMF/FMF/MMF-M6 horizontal VIS1 comparison.

This renderer is descriptive-only.  It reuses the registered S9V authority,
loads the frozen step-5000 Scaled-Chen model, computes one shared encoder
field, and sends that exact field through the native SMF, existing FMF-M2,
and synthetic nested MMF-M6 operators.  The decoder, CMOS relay, and frozen
MPCDAR R16 F32 readout remain unchanged for all three arms.

The MMF labels ETA1, XI2, XIETA, and ETA2 denote synthetic mode families in
this diagnostic sampled basis.  They are not claims about physical LP modes
or real multimode-fiber propagation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt
from ldm_project.v2.contracts import ComplexField
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
)
from ldm_project.v2.optics.few_mode_fiber_operator import (
    FewModeFiberArrayOLS,
    FewModeFiberArraySpec,
)
from ldm_project.v2.optics.nested_multimode_fiber_operator import (
    NestedMultimodeFiberArrayOLS,
)

import render_scaled_chen_full_pipeline as s9v


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


TASK_ID = "H-LOMA-CBOGT-FIBER-VIS1"
CLASSIFICATION = "descriptive_fiber_visualization_completed"
UNIQUE_NEXT_ACTION = "IMPLEMENT_H_LOMA_CBOGT_FIBER_VIS1"
EXPECTED_BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
EXPECTED_START_HEAD = "4d2933cb5b49302dd2114d49b56c905ef9d10ca7"
OUTPUT_ROOT = Path("outputs/20260903_v2/runs")
OUTPUT_PREFIX = "lightgen_scaled_chen_fiber_horizontal_visualization_v1_attempt_"
RASTER_DPI = 600
SEED = 0
HERO_LABEL = 3
HERO_POSITION = 3
PHASE_ALPHA = 0.01
PHASE_FLOOR = 1.0e-12

ARM_ORDER = ("SMF", "FMF", "MMF-M6")
ARM_TITLES = {
    "SMF": "SMF",
    "FMF": "FMF",
    "MMF-M6": "MMF-M6",
}
MMF_FAMILY_ORDER = ("LP01", "LP11a", "ETA1", "XI2", "XIETA", "ETA2")
MMF_SYNTHETIC_EXTRA_FAMILIES = ("ETA1", "XI2", "XIETA", "ETA2")


def _repo_path(value: str | Path) -> Path:
    return s9v._repo_path(value)


def _json_float_list(value: torch.Tensor) -> list[float]:
    detached = value.detach().cpu().reshape(-1).float()
    if not bool(torch.isfinite(detached).all()):
        raise RuntimeError("cannot serialize non-finite metric values")
    return [float(item) for item in detached.tolist()]


def _json_float(value: torch.Tensor) -> float:
    flattened = value.detach().cpu().reshape(-1).float()
    if flattened.numel() != 1 or not bool(torch.isfinite(flattened).all()):
        raise RuntimeError("cannot serialize non-finite scalar metric")
    return float(flattened.item())


def _tensor_cpu(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
        raise RuntimeError("visualization payload contains a non-finite tensor")
    return value.detach().cpu().contiguous()


def _state_digest_without_ols(model: torch.nn.Module) -> str:
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("ols.")
    }
    return canonical_sha256(state)


def _fixed_panel(contract: Mapping[str, Any]) -> dict[str, Any]:
    panel = contract.get("sample_panel")
    expected = {
        "source": "confirmation_panel_train_indices_55000_59999",
        "count": 10,
        "order": list(range(10)),
        "indices_by_label": {
            "0": 55006,
            "1": 55000,
            "2": 55001,
            "3": 55008,
            "4": 55004,
            "5": 55010,
            "6": 55012,
            "7": 55003,
            "8": 55002,
            "9": 55007,
        },
    }
    if not isinstance(panel, Mapping):
        raise RuntimeError("BLOCKED_SAMPLE_CONTRACT_CHANGED: S9V sample panel is missing")
    for key, value in expected.items():
        if panel.get(key) != value:
            raise RuntimeError(
                f"BLOCKED_SAMPLE_CONTRACT_CHANGED: S9V sample panel field {key} changed"
            )
    return dict(panel)


def _validate_git_boundary() -> dict[str, Any]:
    state = s9v._git_state()
    if state["branch"] != EXPECTED_BRANCH:
        raise RuntimeError(f"BLOCKED_START_BRANCH_MISMATCH: {state['branch']}")
    if state["commit"] != EXPECTED_START_HEAD:
        raise RuntimeError(f"BLOCKED_START_HEAD_MISMATCH: {state['commit']}")
    root = Path(state["root"]).resolve()
    expected_renderer = Path(__file__).resolve().relative_to(root).as_posix()
    expected_test = "tests/v2/test_scaled_chen_fiber_visualization.py"
    expected_paths = {expected_renderer, expected_test}
    unexpected: list[str] = []
    expected_task_changes: list[str] = []
    for raw in state["status"]:
        path = raw[3:] if len(raw) >= 3 else raw
        path = path.replace("\\", "/")
        if path in expected_paths:
            expected_task_changes.append(raw)
        else:
            unexpected.append(raw)
    if unexpected:
        raise RuntimeError(
            "BLOCKED_UNEXPECTED_DIRTY_PATH\n" + "\n".join(unexpected)
        )
    return {
        "root": str(root),
        "branch": state["branch"],
        "starting_head": state["commit"],
        "starting_worktree_clean_before_renderer": not expected_task_changes,
        "current_allowed_renderer_changes": expected_task_changes,
        "unexpected_dirty_paths": [],
    }


def _next_output_dir(requested: Path | None) -> Path:
    if requested is not None:
        return _repo_path(requested)
    root = _repo_path(OUTPUT_ROOT)
    pattern = re.compile(rf"^{re.escape(OUTPUT_PREFIX)}(\d+)$")
    numbers = [
        int(match.group(1))
        for path in root.glob(f"{OUTPUT_PREFIX}*")
        if (match := pattern.fullmatch(path.name)) and path.is_dir()
    ]
    attempt = max(numbers, default=0) + 1
    return root / f"{OUTPUT_PREFIX}{attempt:02d}"


def _total_mode_count(operator: torch.nn.Module) -> int:
    total = getattr(operator, "total_mode_count", None)
    if total is None:
        total = getattr(operator, "input_modes").shape[0]
    return int(total)


def _require_frozen_operator(operator: torch.nn.Module, expected_modes: int) -> None:
    if any(parameter.requires_grad for parameter in operator.parameters()):
        raise RuntimeError("fiber comparison operator unexpectedly exposes trainable parameters")
    if _total_mode_count(operator) != expected_modes:
        raise RuntimeError("fiber comparison operator mode count is not registered")
    if not bool(getattr(operator, "diagnostic_only", False)):
        raise RuntimeError("fiber comparison operator is missing diagnostic-only boundary")


def _shared_frontend(
    model: bogt.ScaledChenBOGTModel, amplitude: torch.Tensor
) -> tuple[dict[str, torch.Tensor], ComplexField]:
    slm_command = model.system.command(amplitude)
    slm_field = model.system.slm(slm_command)
    front_field = model.system.front(slm_field)
    encoded_raw, encoder_records_raw, _ = model.system.encoder(front_field, "encoder")
    if len(encoder_records_raw) != 2:
        raise RuntimeError("shared encoder replay did not produce the registered 2-layer path")
    encoder_records = bogt._attenuate_records(encoder_records_raw)
    encoded = bogt._attenuate(
        encoded_raw,
        bogt.FIELD_TRANSMISSION**2,
        "fixed_encoder_group_transmission",
    )
    values = {
        "slm_command": slm_command.tensor,
        "slm_output": slm_field.tensor,
        "encoder_input": front_field.tensor,
        "encoder_layer_1": encoder_records[0].tensor,
        "pre_ols": encoded.tensor,
    }
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise RuntimeError("shared encoder replay produced a non-finite field")
    return values, encoded


def _decode_camera(
    model: bogt.ScaledChenBOGTModel, post_fiber: ComplexField
) -> tuple[ComplexField, torch.Tensor]:
    decoded_raw, decoder_records_raw, _ = model.system.decoder(post_fiber, "decoder")
    if len(decoder_records_raw) != 2:
        raise RuntimeError("decoder replay did not produce the registered 2-layer path")
    decoded = bogt._attenuate(
        decoded_raw,
        bogt.FIELD_TRANSMISSION**2,
        "fixed_decoder_group_transmission",
    )
    chip = ComplexField(decoded.tensor, model.system.chip_plane, decoded.provenance)
    camera_field = model.chen_relay(chip)
    camera_intensity = camera_field.tensor.abs().square()
    if not torch.isfinite(camera_intensity).all():
        raise RuntimeError("decoder/CMOS replay produced a non-finite detector frame")
    return camera_field, camera_intensity


def _family_coefficients(
    arm: str,
    operator: torch.nn.Module,
    diagnostics: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], dict[str, Any]]:
    if arm == "SMF":
        flat = diagnostics.get("port_output_coefficients")
        if not torch.is_tensor(flat) or tuple(flat.shape[1:]) != (100,):
            raise RuntimeError("SMF diagnostics do not expose 100 output coefficients")
        family = flat.unsqueeze(1)
        names = ["LP01"]
        flat_names = ["LP01"] * 100
        ordering = "port-major flat ordering: LP01 for ports 0..99"
    elif arm == "FMF":
        flat = diagnostics.get("port_output_coefficients")
        family = diagnostics.get("mode_coefficients")
        if (
            not torch.is_tensor(flat)
            or not torch.is_tensor(family)
            or tuple(flat.shape[1:]) != (200,)
            or tuple(family.shape[1:]) != (100, 2)
        ):
            raise RuntimeError("FMF diagnostics do not expose the registered 200-mode ordering")
        family = family.permute(0, 2, 1).contiguous()
        names = list(operator.mode_names)  # type: ignore[attr-defined]
        flat_names = [name for _port in range(100) for name in names]
        ordering = "port-major flat ordering: LP01, LP11a interleaved for each port"
    elif arm == "MMF-M6":
        flat = diagnostics.get("coefficients")
        family = diagnostics.get("mode_family_coefficients")
        if (
            not torch.is_tensor(flat)
            or not torch.is_tensor(family)
            or tuple(flat.shape[1:]) != (600,)
            or tuple(family.shape[1:]) != (6, 100)
        ):
            raise RuntimeError("MMF-M6 diagnostics do not expose the registered flat/family ordering")
        metadata = operator.metadata  # type: ignore[attr-defined]
        flat_names = list(metadata["mode_family_names"])
        if flat_names[:200] != [name for _port in range(100) for name in ("LP01", "LP11a")]:
            raise RuntimeError("MMF-M6 FMF prefix ordering changed")
        if flat_names[200:] != [family_name for family_name in MMF_SYNTHETIC_EXTRA_FAMILIES for _port in range(100)]:
            raise RuntimeError("MMF-M6 synthetic family block ordering changed")
        names = list(MMF_FAMILY_ORDER)
        ordering = str(metadata["ordering_contract"])
    else:  # pragma: no cover - arm order is a local constant
        raise ValueError(f"unsupported fiber arm: {arm}")
    if not torch.isfinite(flat).all() or not torch.isfinite(family).all():
        raise RuntimeError(f"{arm} modal coefficients are non-finite")
    return flat, family, names, flat_names, {"ordering_contract": ordering}


def _transport_summary(
    operator: torch.nn.Module, diagnostics: Mapping[str, Any], flat_mode_count: int
) -> dict[str, Any]:
    transport = getattr(operator, "transport")
    expected = torch.eye(flat_mode_count, dtype=transport.dtype, device=transport.device)
    if not torch.equal(transport, expected):
        raise RuntimeError("VIS1 requires the registered identity transport for every arm")
    singular_min = diagnostics.get("transport_singular_min", 1.0)
    singular_max = diagnostics.get("transport_singular_max")
    if singular_max is None:
        singular_max = 1.0
    return {
        "shape": list(transport.shape),
        "dtype": str(transport.dtype),
        "kind": "identity",
        "singular_min": float(singular_min),
        "singular_max": float(singular_max),
    }


def _phase_view(value: torch.Tensor, position: int) -> tuple[np.ndarray, float, float]:
    raw = s9v._display_array(value, position, rotate=True)
    amplitude = np.abs(raw)
    threshold = max(float(amplitude.max()) * PHASE_ALPHA, PHASE_FLOOR)
    phase = np.where(amplitude >= threshold, np.angle(raw), np.nan)
    return phase, threshold, float(np.isfinite(phase).mean())


def _field_display(value: torch.Tensor, position: int) -> np.ndarray:
    return s9v._display_array(value, position, rotate=True)


def _save_figure_bundle(
    figure: plt.Figure,
    output_base: Path,
    *,
    axes: Sequence[plt.Axes],
    panel_ids: Sequence[str],
    row_groups: Sequence[Sequence[str]],
    column_groups: Sequence[Sequence[str]],
) -> list[Path]:
    alignment_path = output_base.with_suffix(".alignment.json")
    overlay_path = output_base.with_suffix(".alignment.svg")
    if alignment_path.exists() or overlay_path.exists():
        raise FileExistsError(f"refusing to overwrite figure QA artifact: {output_base}")
    s9v.require_matplotlib_panel_alignment(
        figure,
        axes=list(axes),
        panel_ids=list(panel_ids),
        row_groups=[list(group) for group in row_groups],
        column_groups=[list(group) for group in column_groups],
        exclude_axes=[],
        json_out=str(alignment_path),
        overlay_svg=str(overlay_path),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )
    saved = [alignment_path, overlay_path]
    formats = (
        (".png", "png", {"dpi": RASTER_DPI}),
        (".tiff", "tiff", {"dpi": RASTER_DPI}),
        (".pdf", "pdf", {}),
        (".svg", "svg", {}),
    )
    for suffix, file_format, kwargs in formats:
        target = output_base.with_suffix(suffix)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite figure: {target}")
        temporary = output_base.with_name(f".{output_base.name}{suffix}.tmp")
        try:
            figure.savefig(
                temporary,
                format=file_format,
                bbox_inches="tight",
                **kwargs,
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        saved.append(target)
    plt.close(figure)
    return saved


def _render_reconstruction(
    target: torch.Tensor,
    arm_predictions: Mapping[str, Mapping[str, torch.Tensor]],
    labels: torch.Tensor,
    output_dir: Path,
) -> list[Path]:
    columns: list[tuple[str, torch.Tensor]] = [("Target", target)]
    for arm in ARM_ORDER:
        columns.extend(
            (
                (f"{ARM_TITLES[arm]}\nCMOS", arm_predictions[arm]["cmos"]),
                (f"{ARM_TITLES[arm]}\nMPCDAR R16 F32", arm_predictions[arm]["mpcdar"]),
            )
        )
    figure, axes = plt.subplots(10, 7, figsize=(7.2, 10.5), squeeze=False)
    figure.subplots_adjust(left=0.065, right=0.995, bottom=0.055, top=0.935, wspace=0.035, hspace=0.19)
    panel_ids: list[str] = []
    for row, label in enumerate(labels.tolist()):
        for column, (title, value) in enumerate(columns):
            axis = axes[row, column]
            image = value[row, 0].detach().cpu().float().numpy()
            axis.imshow(np.clip(image, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
            if row == 0:
                axis.set_title(title, fontsize=7.0, pad=4.0)
            if column == 0:
                axis.set_ylabel(
                    f"digit {int(label)}",
                    rotation=0,
                    rotation_mode="anchor",
                    labelpad=27,
                    va="center",
                    fontsize=7.0,
                )
            panel_ids.append(f"A-r{row}-c{column}")
    figure.suptitle(
        "Figure A · frozen horizontal reconstruction comparison · fixed [0,1] display range",
        fontsize=10.0,
        y=0.972,
    )
    figure.text(
        0.5,
        0.018,
        "Fixed train confirmation panel, digits 0–9 · common display range · no per-image normalization · descriptive only",
        ha="center",
        va="bottom",
        fontsize=7.0,
    )
    row_groups = [panel_ids[row * 7 : (row + 1) * 7] for row in range(10)]
    column_groups = [[panel_ids[row * 7 + column] for row in range(10)] for column in range(7)]
    return _save_figure_bundle(
        figure,
        output_dir / "fiber_horizontal_reconstruction",
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
    )


def _render_ols_representation(
    pre_ols: torch.Tensor,
    arm_data: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    figure, axes = plt.subplots(4, 4, figsize=(7.2, 9.2), squeeze=False)
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.86, wspace=0.30, hspace=0.62)
    pre_arrays = [_field_display(pre_ols, position) for position in range(10)]
    pre_scale = s9v._shared_p995(pre_arrays)
    post_arrays = [
        _field_display(arm_data[arm]["post_fiber"], HERO_POSITION)
        for arm in ARM_ORDER
        for _position in range(1)
    ]
    post_scale = s9v._shared_p995(
        [
            _field_display(arm_data[arm]["post_fiber"], position)
            for arm in ARM_ORDER
            for position in range(10)
        ]
    )
    modal_scale = s9v._shared_p995(
        [arm_data[arm]["modal_power"].detach().cpu().float().numpy() for arm in ARM_ORDER]
    )
    panel_ids: list[str] = []

    def add_panel(axis: plt.Axes, panel_id: str, label: str) -> None:
        panel_ids.append(panel_id)

    pre_amplitude = np.clip(np.abs(pre_arrays[HERO_POSITION]) / max(pre_scale, PHASE_FLOOR), 0.0, 1.0)
    axes[0, 0].imshow(pre_amplitude, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    axes[0, 0].set_aspect("auto")
    axes[0, 0].set_title("Shared pre-OLS amplitude\nhero digit 3", fontsize=7.0, pad=3.0)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    add_panel(axes[0, 0], "B-r0-c0", "a")

    pre_phase, pre_threshold, pre_valid = _phase_view(pre_ols, HERO_POSITION)
    axes[0, 1].set_facecolor("#e8e8e8")
    axes[0, 1].imshow(np.ma.masked_invalid(pre_phase), cmap="twilight", norm=Normalize(-np.pi, np.pi), interpolation="nearest", aspect="auto")
    axes[0, 1].set_aspect("auto")
    axes[0, 1].set_title("Shared pre-OLS phase\nraw [−π,π]", fontsize=7.0, pad=3.0)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    add_panel(axes[0, 1], "B-r0-c1", "b")

    axes[0, 2].axis("off")
    axes[0, 2].set_aspect("auto")
    axes[0, 2].text(
        0.0,
        0.95,
        "Shared-field contract\n\n"
        "One encoder replay\n"
        "same tensor object to all arms\n"
        "pre-OLS = encoder layer 2\n"
        "no per-arm re-encoding\n\n"
        f"Amplitude p99.5 = {pre_scale:.4g}\n"
        f"Phase mask = {pre_threshold:.4g}\n"
        f"Valid phase = {pre_valid:.3f}",
        transform=axes[0, 2].transAxes,
        va="top",
        fontsize=7.0,
        linespacing=1.35,
    )
    add_panel(axes[0, 2], "B-r0-c2", "c")

    axes[0, 3].axis("off")
    axes[0, 3].set_aspect("auto")
    axes[0, 3].text(
        0.0,
        0.95,
        "Display contract\n\n"
        "Amplitude: common panel p99.5\n"
        "Phase: angle(E), [−π,π]\n"
        "Mask: 0.01·max(abs(E))\n"
        "Intermediate maps: 180° display rotation\n"
        "Decoder input = post-fiber alias\n"
        "MPCDAR: frozen R16 F32",
        transform=axes[0, 3].transAxes,
        va="top",
        fontsize=7.0,
        linespacing=1.35,
    )
    add_panel(axes[0, 3], "B-r0-c3", "d")

    for row, arm in enumerate(ARM_ORDER, start=1):
        data = arm_data[arm]
        power = data["modal_power"].detach().cpu().float().numpy()
        axis = axes[row, 0]
        axis.imshow(power.T, cmap="magma", vmin=0.0, vmax=modal_scale, aspect="auto", interpolation="nearest")
        axis.set_aspect("auto")
        axis.set_title(f"{ARM_TITLES[arm]} modal power\nshared p99.5 scale", fontsize=6.8, pad=3.0)
        axis.set_xticks(range(10), [str(index) for index in range(10)], fontsize=6.0)
        axis.set_xlabel("digit", fontsize=6.5)
        family_labels = list(data["family_labels"])
        axis.set_yticks(range(len(family_labels)), family_labels, fontsize=5.4)
        axis.tick_params(axis="y", pad=1.5)
        axis.set_ylabel("mode family", fontsize=6.5)
        add_panel(axis, f"B-r{row}-c0", chr(ord("a") + row * 4))

        post_amplitude = np.clip(np.abs(post_arrays[row - 1]) / max(post_scale, PHASE_FLOOR), 0.0, 1.0)
        axis = axes[row, 1]
        axis.imshow(post_amplitude, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
        axis.set_aspect("auto")
        axis.set_title(f"{ARM_TITLES[arm]} post-fiber amplitude\n= decoder input", fontsize=6.8, pad=3.0)
        axis.set_xticks([])
        axis.set_yticks([])
        add_panel(axis, f"B-r{row}-c1", chr(ord("a") + row * 4 + 1))

        post_phase, _threshold, _valid = _phase_view(data["post_fiber"], HERO_POSITION)
        axis = axes[row, 2]
        axis.set_facecolor("#e8e8e8")
        axis.imshow(np.ma.masked_invalid(post_phase), cmap="twilight", norm=Normalize(-np.pi, np.pi), interpolation="nearest", aspect="auto")
        axis.set_aspect("auto")
        axis.set_title(f"{ARM_TITLES[arm]} post-fiber phase\nraw [−π,π]", fontsize=6.8, pad=3.0)
        axis.set_xticks([])
        axis.set_yticks([])
        add_panel(axis, f"B-r{row}-c2", chr(ord("a") + row * 4 + 2))

        axis = axes[row, 3]
        axis.axis("off")
        axis.set_aspect("auto")
        ordering_label = {
            "SMF": "port-major LP01 × 100",
            "FMF": "port-major LP01/LP11a interleaved",
            "MMF-M6": "FMF prefix + four synthetic family blocks",
        }[arm]
        axis.text(
            0.0,
            0.95,
            f"{ARM_TITLES[arm]}\n\n"
            f"mode count: {data['mode_count']}\n"
            f"modes / port: {data['modes_per_port']}\n"
            f"transport: identity\n"
            f"flat ordering:\n{textwrap.fill(ordering_label, width=27)}\n\n"
            "post-fiber alias:\n"
            "decoder input\n\n"
            "synthetic diagnostic-only",
            transform=axis.transAxes,
            va="top",
            fontsize=6.5,
            linespacing=1.28,
        )
        add_panel(axis, f"B-r{row}-c3", chr(ord("a") + row * 4 + 3))

    figure.suptitle(
        "Figure B · common pre-OLS representation, modal power, and post-fiber fields",
        fontsize=9.0,
        y=0.98,
    )
    figure.text(
        0.5,
        0.018,
        "MMF-M6 extra rows are synthetic mode families; no physical LP11b/LP21/LP02 or guided-mode claim",
        ha="center",
        va="bottom",
        fontsize=7.0,
    )
    row_groups = [panel_ids[row * 4 : (row + 1) * 4] for row in range(4)]
    column_groups = [[panel_ids[row * 4 + column] for row in range(4)] for column in range(4)]
    return _save_figure_bundle(
        figure,
        output_dir / "fiber_horizontal_ols_representation",
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
    )


def _operator_contract(
    arm: str,
    operator: torch.nn.Module,
    diagnostics: Mapping[str, Any],
    post_fiber: ComplexField,
    pre_energy: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]:
    flat_count = {"SMF": 100, "FMF": 200, "MMF-M6": 600}[arm]
    flat, family_coefficients, family_names, flat_names, ordering = _family_coefficients(
        arm, operator, diagnostics
    )
    power = family_coefficients.abs().square().sum(dim=2)
    captured = power.sum(dim=1)
    input_energy = diagnostics["input_energy"].to(pre_energy)
    output_energy = diagnostics["output_energy"].to(pre_energy)
    if not torch.allclose(input_energy, pre_energy, atol=0.0, rtol=0.0):
        raise RuntimeError(f"{arm} did not receive the shared pre-OLS field energy")
    modal_metadata: dict[str, Any] = {
        "mode_family_order": family_names,
        "flat_mode_family_names": flat_names,
        "flat_mode_count": flat_count,
        "port_count": 100,
        "modes_per_port": flat_count // 100,
        "ordering_contract": ordering["ordering_contract"],
    }
    if arm == "MMF-M6":
        modal_metadata.update(
            {
                "synthetic_extra_families": list(MMF_SYNTHETIC_EXTRA_FAMILIES),
                "operator_metadata": operator.metadata,  # type: ignore[attr-defined]
            }
        )
    contract = {
        "mode_count": len(family_names),
        "modes_per_port": flat_count // 100,
        "total_mode_count": flat_count,
        "transport": _transport_summary(operator, diagnostics, flat_count),
        "provenance": post_fiber.provenance.to_dict(),
        "provenance_operation": post_fiber.provenance.operation,
        "modal_metadata": modal_metadata,
        "post_fiber_alias": "decoder_input",
        "diagnostic_only": True,
        "physical_parameter_status": "synthetic diagnostic-only",
    }
    metrics = {
        "captured_modal_power": {
            "per_sample": _json_float_list(captured),
            "mean": _json_float(captured.mean()),
        },
        "captured_fraction": {
            "per_sample": _json_float_list(captured / input_energy.clamp_min(1.0e-30)),
            "mean": _json_float((captured / input_energy.clamp_min(1.0e-30)).mean()),
        },
        "modal_power_by_family": {
            "family_order": family_names,
            "per_sample": [
                _json_float_list(row) for row in power.detach().cpu()
            ],
            "mean_by_family": _json_float_list(power.mean(dim=0)),
        },
        "pre_ols_power": _json_float_list(input_energy),
        "post_fiber_power": {
            "per_sample": _json_float_list(output_energy),
            "mean": _json_float(output_energy.mean()),
        },
    }
    return contract, metrics, flat, family_coefficients


def _build_arm_metrics(
    camera_intensity: torch.Tensor,
    target: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    reconstruction_metrics: Mapping[str, Any],
    z: torch.Tensor,
    fiber_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    camera = camera_intensity.detach().float()
    total = camera.flatten(1).sum(dim=1)
    mean = camera.flatten(1).mean(dim=1)
    mpcdar = predictions["mpcdar_rank16_float32"]
    mse_by_sample = (mpcdar - target.detach().cpu().float()).square().flatten(1).mean(dim=1)
    if not torch.allclose(mse_by_sample, torch.as_tensor(reconstruction_metrics["per_sample_mse"]["mpcdar_rank16_float32"]), atol=0.0, rtol=0.0):
        raise RuntimeError("MPCDAR per-sample MSE serialization is inconsistent")
    z_norm = z.norm(dim=1)
    z_second = z.square().mean(dim=1)
    z_variance = z.var(dim=1, unbiased=False)
    return {
        **fiber_metrics,
        "cmos_intensity_total": {"per_sample": _json_float_list(total), "mean": _json_float(total.mean())},
        "cmos_intensity_mean": {"per_sample": _json_float_list(mean), "mean": _json_float(mean.mean())},
        "z_norm": {"per_sample": _json_float_list(z_norm), "mean": _json_float(z_norm.mean())},
        "z_second_moment": {"per_sample": _json_float_list(z_second), "mean": _json_float(z_second.mean())},
        "z_variance": {"per_sample": _json_float_list(z_variance), "mean": _json_float(z_variance.mean())},
        "mpcdar_rank16_float32": {
            "mse_per_sample": _json_float_list(mse_by_sample),
            "mse_mean": _json_float(mse_by_sample.mean()),
            "readout": "existing frozen S9 MPCDAR rank16 float32",
        },
        "reconstruction_method_mse": reconstruction_metrics["method_mse"],
        "descriptive_only": True,
    }


def _detector_deltas(z_by_arm: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    native = z_by_arm["SMF"]
    result: dict[str, Any] = {"reference_arm": "SMF", "arms": {}}
    for arm in ("FMF", "MMF-M6"):
        value = z_by_arm[arm]
        delta = (value - native).norm(dim=1)
        cosine = torch.nn.functional.cosine_similarity(value, native, dim=1)
        result["arms"][arm] = {
            "delta_l2": {"per_sample": _json_float_list(delta), "mean": _json_float(delta.mean())},
            "cosine_like_similarity": {"per_sample": _json_float_list(cosine), "mean": _json_float(cosine.mean())},
            "alignment_or_remap": False,
            "refit": False,
        }
    return result


def _authority_summary(
    contract: Mapping[str, Any],
    review: Mapping[str, Any],
    config_path: Path,
    review_path: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    rmal_contract = contract["rmal"]
    return {
        "s9v_task_id": contract["task_id"],
        "s9v_config": str(config_path.resolve()),
        "s9v_config_sha256": file_sha256(config_path),
        "s9v_review": str(review_path.resolve()),
        "s9v_review_sha256": file_sha256(review_path),
        "s9v_review_decision": review["decision"],
        "s9v_registered_expected_head": contract["expected_head"],
        "rmal_config": str(_repo_path(rmal_contract["config"]).resolve()),
        "rmal_config_sha256": rmal_contract["config_sha256"],
        "checkpoint": str(Path(authority["rmal_checkpoint"]).resolve()),
        "checkpoint_sha256": rmal_contract["checkpoint_sha256"],
        "checkpoint_raw_sha256_verified": file_sha256(authority["rmal_checkpoint"]),
        "rmal_target": rmal_contract["target"],
        "rmal_target_shape": list(rmal_contract["target_shape"]),
        "parent_evidence": authority["parent_files"],
    }


def _panel_payload(
    images: torch.Tensor,
    labels: torch.Tensor,
    target: torch.Tensor,
    pre_ols: torch.Tensor,
    arm_data: Mapping[str, Mapping[str, Any]],
    fixed_panel: Mapping[str, Any],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        data = arm_data[arm]
        arms[arm] = {
            "cmos_detector_intensity": _tensor_cpu(data["camera_intensity"]),
            "post_fiber": _tensor_cpu(data["post_fiber"]),
            "flat_coefficients": _tensor_cpu(data["flat_coefficients"]),
            "mode_family_coefficients": _tensor_cpu(data["family_coefficients"]),
            "modal_power": _tensor_cpu(data["modal_power"]),
            "mpcdar_rank16_float32": _tensor_cpu(data["mpcdar"]),
            "z_256": _tensor_cpu(data["z"]),
            "mode_family_order": list(data["family_order"]),
            "flat_mode_family_names": list(data["flat_mode_family_names"]),
        }
    return {
        "schema": "lightgen_scaled_chen_fiber_horizontal_panel.v1",
        "task_id": TASK_ID,
        "classification": CLASSIFICATION,
        "fixed_panel": dict(fixed_panel),
        "images_28x28": _tensor_cpu(images),
        "labels": _tensor_cpu(labels),
        "target": _tensor_cpu(target),
        "shared_pre_ols": _tensor_cpu(pre_ols),
        "arms": arms,
        "full_mode_banks_stored": False,
        "mode_bank_note": "Only captured coefficients and family power are stored; no 600-mode bank is exported.",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--visualization-config", type=Path, default=s9v.FULL_PATH_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VIS1 fixed horizontal visualization requires CUDA")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("VIS1 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    git_boundary = _validate_git_boundary()
    output_dir = _next_output_dir(args.output_dir)
    s9v._check_output_boundary(output_dir)
    contract, review, config_path, review_path = s9v._load_full_path_contract(args.visualization_config)
    fixed_panel = _fixed_panel(contract)
    authority = s9v._load_full_path_authority(contract, args.dataset_root, torch.device(args.device))

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    train_images, train_labels = base.load_local_mnist(args.dataset_root, train=True)
    images, labels, fixed_indices = s9v._select_confirmation_digits(train_images, train_labels)
    expected_indices = {int(key): int(value) for key, value in fixed_panel["indices_by_label"].items()}
    if fixed_indices != expected_indices or labels.tolist() != list(range(10)):
        raise RuntimeError("BLOCKED_SAMPLE_CONTRACT_CHANGED: loaded panel differs from registered S9V panel")
    rendered = base.render_intensity_batch(images)
    device = torch.device(args.device)
    amplitude = rendered.amplitude.to(device)
    target = rendered.target.to(device)
    model = authority["model"]
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("loaded Scaled-Chen model is not frozen")
    model_state_before = _state_digest_without_ols(model)
    native = model.ols
    fmf = FewModeFiberArrayOLS(model.system.encoder_plane, FewModeFiberArraySpec()).to(device).eval()
    mmf = NestedMultimodeFiberArrayOLS(model.system.encoder_plane).to(device).eval()
    _require_frozen_operator(native, 100)
    _require_frozen_operator(fmf, 200)
    _require_frozen_operator(mmf, 600)
    operators: dict[str, torch.nn.Module] = {"SMF": native, "FMF": fmf, "MMF-M6": mmf}
    preflight_mode_counts = {arm: _total_mode_count(operator) for arm, operator in operators.items()}

    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.no_grad():
        shared_values, encoded = _shared_frontend(model, amplitude)
        pre_energy = encoded.tensor[:, 0].abs().square().sum(dim=(-2, -1)) * (
            model.system.encoder_plane.pixel_pitch_m[0] * model.system.encoder_plane.pixel_pitch_m[1]
        )
        if not torch.isfinite(pre_energy).all():
            raise RuntimeError("shared pre-OLS energy is non-finite")
        input_pointer = encoded.tensor.data_ptr()
        arm_data: dict[str, dict[str, Any]] = {}
        z_by_arm: dict[str, torch.Tensor] = {}
        for arm in ARM_ORDER:
            operator = operators[arm]
            if encoded.tensor.data_ptr() != input_pointer:
                raise RuntimeError("shared pre-OLS tensor identity changed before arm replay")
            post_fiber = operator(encoded)
            diagnostics = dict(operator.last_diagnostics)  # type: ignore[attr-defined]
            camera_field, camera_intensity = _decode_camera(model, post_fiber)
            flat_coefficients, family_coefficients, family_order, flat_names, ordering = _family_coefficients(
                arm, operator, diagnostics
            )
            modal_power = family_coefficients.abs().square().sum(dim=2)
            if not torch.isfinite(modal_power).all():
                raise RuntimeError(f"{arm} modal power is non-finite")
            values = {"cmos_detector_intensity": camera_intensity}
            predictions, reconstruction_metrics = s9v._replay_full_path_reconstructions(
                authority,
                values,
                images,
                labels,
                target,
            )
            z = s9v.mra._feature_projection(camera_intensity.detach().cpu().float(), complex_value=False).float()
            if tuple(z.shape) != (10, 256) or not torch.isfinite(z).all():
                raise RuntimeError(f"{arm} detector feature projection is not finite [10,256]")
            fiber_contract, fiber_metrics, _flat_unused, _family_unused = _operator_contract(
                arm,
                operator,
                diagnostics,
                post_fiber,
                pre_energy,
            )
            fiber_metrics["modal_power_by_family"]["family_order"] = family_order
            arm_metrics = _build_arm_metrics(
                camera_intensity,
                target,
                predictions,
                reconstruction_metrics,
                z,
                fiber_metrics,
            )
            arm_data[arm] = {
                "contract": fiber_contract,
                "metrics": arm_metrics,
                "post_fiber": post_fiber.tensor,
                "camera_field": camera_field.tensor,
                "camera_intensity": camera_intensity,
                "flat_coefficients": flat_coefficients,
                "family_coefficients": family_coefficients,
                "modal_power": modal_power,
                "family_order": family_order,
                "family_labels": [
                    (f"{name}\n(synthetic)" if arm == "MMF-M6" and name in MMF_SYNTHETIC_EXTRA_FAMILIES else name)
                    for name in family_order
                ],
                "flat_mode_family_names": flat_names,
                "ordering_contract": ordering["ordering_contract"],
                "mpcdar": predictions["mpcdar_rank16_float32"],
                "z": z,
                "predictions": predictions,
                "reconstruction_metrics": reconstruction_metrics,
                "provenance_operation": post_fiber.provenance.operation,
                "mode_count": len(family_order),
                "modes_per_port": _total_mode_count(operator) // 100,
            }
            z_by_arm[arm] = z
            if encoded.tensor.data_ptr() != input_pointer:
                raise RuntimeError("fiber arm replay mutated the shared pre-OLS tensor")

        model_state_after = _state_digest_without_ols(model)
        if model_state_before != model_state_after:
            raise RuntimeError("VIS1 changed learned model state")
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    elapsed_seconds = time.monotonic() - started

    output_dir.mkdir(parents=True, exist_ok=False)
    contracts_payload = {
        "schema": "lightgen_scaled_chen_fiber_horizontal_contracts.v1",
        "task_id": TASK_ID,
        "classification": CLASSIFICATION,
        "diagnostic_only": True,
        "unique_next_action": UNIQUE_NEXT_ACTION,
        "local_authority": _authority_summary(contract, review, config_path, review_path, authority),
        "git_preflight": git_boundary,
        "fixed_panel": fixed_panel,
        "fixed_state": {
            "seed": SEED,
            "device": str(device),
            "input": "fixed S9V Scaled-Chen confirmation panel rendered once",
            "shared_pre_ols": True,
            "same_input_tensor_data_ptr": True,
            "frontend_replayed_once": True,
            "decoder_relay_cmos_fixed": True,
            "mpcdar": "existing frozen S9 MPCDAR rank16 float32 readout",
            "rank_search": False,
            "refit": False,
            "alignment_or_remap": False,
            "training": False,
            "optimizer_created": False,
            "backward_called": False,
            "new_evaluator_classifier_probe_regression": False,
        },
        "display_contract": {
            "figure_a_columns": [
                "Target",
                "SMF CMOS",
                "SMF MPCDAR R16 F32",
                "FMF CMOS",
                "FMF MPCDAR R16 F32",
                "MMF-M6 CMOS",
                "MMF-M6 MPCDAR R16 F32",
            ],
            "reconstruction_range": [0.0, 1.0],
            "amplitude": "common panel-wide p99.5 scale per figure/stage",
            "phase": "angle(E), raw [-pi,pi]",
            "phase_mask": "max(0.01 * max(abs(E)), 1e-12) per stage and sample",
            "no_per_image_normalization": True,
            "intermediate_rotation": "display-only 180 degrees",
            "modal_metadata_ordering": "actual flat ordering; no reshape/pad/resize/interpolate/fake 10x10x6",
            "mmf_extra_family_label": "synthetic mode family",
        },
        "arm_order": list(ARM_ORDER),
        "arm_contracts": {arm: arm_data[arm]["contract"] for arm in ARM_ORDER},
        "mode_counts": preflight_mode_counts,
    }
    metrics_payload = {
        "schema": "lightgen_scaled_chen_fiber_horizontal_comparison_metrics.v1",
        "task_id": TASK_ID,
        "classification": CLASSIFICATION,
        "descriptive_only": True,
        "fixed_panel": fixed_panel,
        "shared_pre_ols": {
            "data_ptr_identity": True,
            "pre_ols_shape": list(shared_values["pre_ols"].shape),
            "pre_ols_power_per_sample": _json_float_list(pre_energy),
        },
        "arms": {arm: arm_data[arm]["metrics"] for arm in ARM_ORDER},
        "detector_distribution_deltas": _detector_deltas(z_by_arm),
        "interpretation_boundary": "Fixed downstream distribution mismatch descriptors only; no fiber worse/superior claim.",
    }
    panel_payload = _panel_payload(
        images,
        labels,
        target,
        shared_values["pre_ols"],
        arm_data,
        fixed_panel,
    )
    atomic_write_new_json(output_dir / "fiber_contracts.json", contracts_payload, artifact_kind="fiber contracts")
    atomic_write_new_json(output_dir / "comparison_metrics.json", metrics_payload, artifact_kind="comparison metrics")
    atomic_torch_save_new(output_dir / "panel.pt", panel_payload, artifact_kind="visualization panel")
    saved = _render_reconstruction(
        target.detach().cpu().float(),
        {
            arm: {
                "cmos": arm_data[arm]["camera_intensity"].detach().cpu().float(),
                "mpcdar": arm_data[arm]["mpcdar"].detach().cpu().float(),
            }
            for arm in ARM_ORDER
        },
        labels,
        output_dir,
    )
    saved += _render_ols_representation(shared_values["pre_ols"].detach().cpu(), arm_data, output_dir)
    output_files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "comparison_manifest.json"
    )
    output_bytes = sum(path.stat().st_size for path in output_files)
    if output_bytes > 512 * 1024**2:
        raise RuntimeError(f"VIS1 output exceeds 512 MiB: {output_bytes} bytes")
    final_git = s9v._git_state()
    manifest = {
        "schema": "lightgen_scaled_chen_fiber_horizontal_visualization.v1",
        "task_id": TASK_ID,
        "status": "completed",
        "classification": CLASSIFICATION,
        "diagnostic_only": True,
        "output_dir": str(output_dir.resolve()),
        "git": {
            "preflight": git_boundary,
            "after_render": final_git,
        },
        "authority": _authority_summary(contract, review, config_path, review_path, authority),
        "fixed_panel": fixed_panel,
        "figures": {
            "figure_a": "fiber_horizontal_reconstruction",
            "figure_b": "fiber_horizontal_ols_representation",
            "required_pngs": [
                str((output_dir / "fiber_horizontal_reconstruction.png").resolve()),
                str((output_dir / "fiber_horizontal_ols_representation.png").resolve()),
            ],
        },
        "learned_state_freeze": {
            "model_state_excludes": "ols.* operator buffers because the comparison intentionally changes the OLS arm",
            "model_before": model_state_before,
            "model_after": model_state_after,
            "unchanged": model_state_before == model_state_after,
            "all_model_parameters_requires_grad_false": True,
            "all_arm_operators_parameter_free": True,
        },
        "resources": {
            "device": str(device),
            "seed": SEED,
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_allocated_bytes": peak_cuda_bytes,
            "sample_count": 10,
            "full_dataset_sweep": False,
            "full_mode_bank_export": False,
            "cuda_run_is_descriptive_telemetry": True,
        },
        "authorization": {
            "no_training": True,
            "no_update": True,
            "no_optimizer": True,
            "no_backward": True,
            "no_rank_search": True,
            "no_refit": True,
            "no_new_evaluator": True,
            "formal_promotion": False,
            "generation": False,
        },
        "interpretation_boundary": "Use fixed downstream distribution mismatch wording only; this VIS1 does not establish fiber superiority, inferiority, physical MMF propagation, reconstruction manifold support, or generation readiness.",
        "outputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
        "output_bytes_before_manifest": output_bytes,
        "manifest_self_hash": "not embedded to avoid self-reference",
    }
    atomic_write_new_json(output_dir / "comparison_manifest.json", manifest, artifact_kind="comparison manifest")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "classification": CLASSIFICATION,
                "sample_count": 10,
                "figures": [str(path.resolve()) for path in saved if path.suffix == ".png"],
                "peak_cuda_memory_allocated_bytes": peak_cuda_bytes,
                "elapsed_seconds": elapsed_seconds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
