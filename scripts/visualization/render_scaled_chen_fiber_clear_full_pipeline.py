"""Render the VIS2 clear-input full-pipeline fiber visualization.

This diagnostic renderer has an explicit two-stage boundary.  Stage A reads
only the local MNIST train images, labels, and dataset indices, writes a
candidate contact sheet, and freezes one input-only 0--9 panel.  The full run
requires a separate review artifact for that frozen panel, then replays the
same ten inputs through the shared Scaled-Chen frontend and the native SMF,
existing FMF-M2, and synthetic nested MMF-M6 diagnostic arms.  It never
trains, refits, searches ranks or modes, or changes the VIS1 authority.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import time
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
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_manifold_readiness as mra
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    tensor_sha256,
)
from ldm_project.v2.optics.few_mode_fiber_operator import (
    FewModeFiberArrayOLS,
    FewModeFiberArraySpec,
)
from ldm_project.v2.optics.nested_multimode_fiber_operator import (
    NestedMultimodeFiberArrayOLS,
)

import render_scaled_chen_fiber_horizontal as vis1
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


TASK_ID = "H-LOMA-CBOGT-FIBER-VIS2"
CLASSIFICATION = "canonical_clear_full_pipeline_visualization_completed"
AWAITING_CLASSIFICATION = "AWAITING_CLEAR_PANEL_REVIEW"
UNIQUE_NEXT_ACTION = "manual_review_VIS2_figures"
AWAITING_NEXT_ACTION = "manual_review_clear_0_9_contact_sheet"
EXPECTED_BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
EXPECTED_START_HEAD = "4d2933cb5b49302dd2114d49b56c905ef9d10ca7"
THREE_PHOTO_EXPECTED_START_HEAD = "d07fdaa17aa91d9f3e5498e9b1f10d91db766777"
OUTPUT_ROOT = Path("outputs/20260903_v2/runs")
OUTPUT_PREFIX = "lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_"
VIS1_OUTPUT = Path(
    "outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_horizontal_visualization_v1_attempt_06"
)
RASTER_DPI = 300
SEED = 0
VIS2_LAYOUT_SCHEMA = "organized_compact_arm_separated_v3"
ARTIFACT_LAYOUT_SCHEMA = "vis2_content_folders_v1"
ARTIFACT_FOLDERS = {
    "digit_visualization": "digit_visualization",
    "pipeline_overview": "pipeline_overview",
    "fiber_information": "fiber_information",
    "metadata_qa": "metadata_qa",
}
FIGURE_EXPORT_SUFFIXES = (".png", ".pdf", ".svg")
THREE_PHOTO_LAYOUT_SCHEMA = "same_digit_three_arm_comparison_v1"
THREE_PHOTO_ARTIFACT_LAYOUT_SCHEMA = "vis2_three_photo_v1"
THREE_PHOTO_PAGE_RANGES = ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))
THREE_PHOTO_COLUMN_KEYS = ("standard_input", "SMF", "FMF", "MMF-M6")
THREE_PHOTO_OUTPUT_KEY = "mpcdar_rank16_float32"
THREE_PHOTO_FIGURE_EXPORT_SUFFIXES = (".png", ".pdf", ".svg")
THREE_PHOTO_PAGE_WIDTH_INCHES = 7.0
THREE_PHOTO_SPATIAL_ASPECT = "equal"
THREE_PHOTO_SOURCE_PANEL_SHA256 = "1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3"
THREE_PHOTO_FIGURE_PREFIX = "fiber_vis2_same_digit_comparison_page"
THREE_PHOTO_ALLOWED_DIRTY_PATHS = frozenset(
    {
        "docs/CURRENT_STATUS.md",
        "docs/migration_windows_native_20260830.md",
        "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_prior_support_accessibility_v1.yaml",
        "scripts/research/scaled_chen/measure_scaled_chen_g1_prior_support_accessibility.py",
        "tests/v2/test_scaled_chen_g1_prior_support_accessibility.py",
        "scripts/visualization/render_scaled_chen_fiber_clear_full_pipeline.py",
        "tests/v2/test_scaled_chen_fiber_clear_full_pipeline.py",
    }
)
ARM_COLORS = {
    "SMF": "#2f6db0",
    "FMF": "#2f8f5b",
    "MMF-M6": "#c77700",
}
DIGIT_PAGE_RANGES = (tuple(range(5)), tuple(range(5, 10)))
STAGE_GROUPS = {
    "frontend": (
        ("Input", "input", "input"),
        ("SLM command", "slm_command", "real"),
        ("SLM output", "slm_output", "complex"),
        ("Encoder input", "encoder_input", "complex"),
        ("Encoder L1", "encoder_layer_1", "complex"),
        ("Pre-OLS", "pre_ols", "complex"),
    ),
    "fiber_decoder": (
        ("Modal representation", "modal", "modal"),
        ("Post-fiber / decoder input", "post_fiber", "complex"),
        ("Decoder L1", "decoder_layer_1", "complex"),
        ("Decoder L2", "decoder_layer_2", "complex"),
    ),
    "detector_reconstruction": (
        ("CMOS incident field", "cmos_field", "complex"),
        ("CMOS intensity", "cmos_intensity", "intensity"),
        ("MPCDAR R16 F32", "mpcdar", "reconstruction"),
    ),
}
STAGE_GROUP_TITLES = {
    "frontend": "shared frontend → pre-OLS",
    "fiber_decoder": "fiber branch → decoder",
    "detector_reconstruction": "CMOS detector → frozen reconstruction",
}
PANEL_SCHEMA = "lightgen_scaled_chen_fiber_vis2_full_pipeline_panel.v1"

CANDIDATE_START = 55000
CANDIDATE_STOP = 60000
CANDIDATES_PER_DIGIT = 32
INPUT_THRESHOLD = 0.10
CENTROID_RADIUS_MAX = 0.28
OCCUPANCY_RANGE = (0.04, 0.55)
BBOX_HEIGHT_RANGE = (12, 27)
BBOX_WIDTH_RANGE = (4, 27)

ARM_ORDER = ("SMF", "FMF", "MMF-M6")
ARM_TOTAL_MODES = {"SMF": 100, "FMF": 200, "MMF-M6": 600}
MMF_SYNTHETIC_EXTRA_FAMILIES = ("ETA1", "XI2", "XIETA", "ETA2")

COMMON_STAGE_ORDER = (
    "input",
    "slm_command",
    "slm_output",
    "encoder_input",
    "encoder_layer_1",
    "pre_ols",
)
ARM_STAGE_ORDER = (
    "modal",
    "post_fiber",
    "decoder_layer_1",
    "decoder_layer_2",
    "cmos_field",
    "cmos_intensity",
    "mpcdar",
)
PROCESS_FLOW_LAYOUT_SCHEMA = "same_digit_interleaved_three_arm_process_flow_v1"
PROCESS_FLOW_ARTIFACT_LAYOUT_SCHEMA = "vis2_process_flow_interleaved_v1"
PROCESS_FLOW_PAGE_RANGES = THREE_PHOTO_PAGE_RANGES
PROCESS_FLOW_ARM_LANES = ARM_ORDER
PROCESS_FLOW_ROW_ORDER = "digit_then_arm_then_amplitude_phase"
PROCESS_FLOW_ROWS_PER_DIGIT = len(PROCESS_FLOW_ARM_LANES) * 2
PROCESS_FLOW_ROW_STRUCTURE = (
    "for each digit: SMF amplitude/phase, FMF amplitude/phase, "
    "MMF-M6 amplitude/phase; then the next digit"
)
PROCESS_FLOW_PAGE_WIDTH_INCHES = 12.0
PROCESS_FLOW_SPATIAL_ASPECT = "equal"
PROCESS_FLOW_FIGURE_PREFIX = "fiber_vis2_process_flow_page"
PROCESS_FLOW_STAGE_KEYS = COMMON_STAGE_ORDER + ARM_STAGE_ORDER
PROCESS_FLOW_STAGE_SPECS = (
    ("Input", "input", "input"),
    ("SLM command", "slm_command", "real"),
    ("SLM output", "slm_output", "complex"),
    ("Encoder in", "encoder_input", "complex"),
    ("Encoder L1", "encoder_layer_1", "complex"),
    ("Pre-OLS", "pre_ols", "complex"),
    ("Modal", "modal", "modal"),
    ("Post-fiber", "post_fiber", "complex"),
    ("Decoder L1", "decoder_layer_1", "complex"),
    ("Decoder L2", "decoder_layer_2", "complex"),
    ("CMOS field", "cmos_field", "complex"),
    ("CMOS intensity", "cmos_intensity", "intensity"),
    ("MPCDAR R16", "mpcdar", "reconstruction"),
)
PROCESS_FLOW_OUTPUT_FIELD = "mpcdar_rank16_float32"
COMPLEX_SPATIAL_KEYS = (
    "slm_output",
    "encoder_input",
    "encoder_layer_1",
    "pre_ols",
    "post_fiber",
    "decoder_layer_1",
    "decoder_layer_2",
    "cmos_field",
)
ROTATE_SPATIAL_KEYS = frozenset(
    {
        "slm_output",
        "encoder_input",
        "encoder_layer_1",
        "pre_ols",
        "post_fiber",
        "decoder_layer_1",
        "decoder_layer_2",
    }
)
ORIENTATION_POLICY = {
    "row_label_rotation_degrees": 0,
    "image_space": {
        "origin": "upper",
        "display_rotation_degrees": 0,
        "tensor_rotation": False,
    },
    "optical_spatial": {
        "origin": "lower",
        "display_rotation_degrees": 180,
        "display_rotation_keys": sorted(ROTATE_SPATIAL_KEYS),
        "unrotated_keys": sorted(set(COMPLEX_SPATIAL_KEYS) - ROTATE_SPATIAL_KEYS),
        "tensor_rotation": False,
    },
    "modal_family_port": {
        "strategy": "explicit_nonspatial_matrix",
        "origin": "upper",
        "spatial_rotation": False,
    },
}
IMAGE_ORIGIN = ORIENTATION_POLICY["image_space"]["origin"]
OPTICAL_ORIGIN = ORIENTATION_POLICY["optical_spatial"]["origin"]
MODAL_ORIGIN = ORIENTATION_POLICY["modal_family_port"]["origin"]
PHASE_ALPHA = 0.01
PHASE_FLOOR = 1.0e-12


def _repo_path(value: str | Path) -> Path:
    return s9v._repo_path(value)


def _json_float(value: float | torch.Tensor) -> float:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().reshape(-1).float()
        if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("non-finite scalar cannot be serialized")
        return float(tensor.item())
    if not math.isfinite(value):
        raise RuntimeError("non-finite scalar cannot be serialized")
    return float(value)


def _json_float_list(value: torch.Tensor) -> list[float]:
    tensor = value.detach().cpu().reshape(-1).float()
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("non-finite tensor cannot be serialized")
    return [float(item) for item in tensor.tolist()]


def _tensor_cpu(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
        raise RuntimeError("visualization payload contains a non-finite tensor")
    return value.detach().cpu().contiguous()


def _horizontal_row_label(
    axis: plt.Axes,
    label: str,
    *,
    fontsize: float = 6.0,
    color: str | None = None,
    x: float = -0.06,
) -> matplotlib.text.Text:
    """Place a readable, upright row label outside a panel."""

    return axis.text(
        x,
        0.5,
        label,
        transform=axis.transAxes,
        rotation=0,
        rotation_mode="anchor",
        ha="right",
        va="center",
        color=color,
        fontsize=fontsize,
        linespacing=1.2,
        clip_on=False,
    )


def _git_state() -> dict[str, Any]:
    return s9v._git_state()


def _validate_git_boundary(
    *,
    expected_start_head: str = EXPECTED_START_HEAD,
    expected_paths: frozenset[str] | None = None,
) -> dict[str, Any]:
    state = _git_state()
    if state["branch"] != EXPECTED_BRANCH:
        raise RuntimeError(f"BLOCKED_START_BRANCH_MISMATCH: {state['branch']}")
    if state["commit"] != expected_start_head:
        raise RuntimeError(f"BLOCKED_START_HEAD_MISMATCH: {state['commit']}")
    root = Path(state["root"]).resolve()
    allowed_paths = expected_paths or frozenset(
        {
            "docs/CURRENT_STATUS.md",
            "docs/migration_windows_native_20260830.md",
            "scripts/visualization/render_scaled_chen_fiber_horizontal.py",
            "tests/v2/test_scaled_chen_fiber_visualization.py",
            "scripts/visualization/render_scaled_chen_fiber_clear_full_pipeline.py",
            "tests/v2/test_scaled_chen_fiber_clear_full_pipeline.py",
        }
    )
    expected_task_changes: list[str] = []
    unexpected: list[str] = []
    for raw in state["status"]:
        path = raw[3:] if len(raw) >= 3 else raw
        path = path.replace("\\", "/")
        if path in allowed_paths:
            expected_task_changes.append(raw)
        else:
            unexpected.append(raw)
    if unexpected:
        raise RuntimeError("BLOCKED_UNEXPECTED_DIRTY_PATH\n" + "\n".join(unexpected))
    return {
        "root": str(root),
        "branch": state["branch"],
        "starting_head": state["commit"],
        "starting_worktree_clean_before_vis2": not expected_task_changes,
        "known_expected_dirty_paths": expected_task_changes,
        "unexpected_dirty_paths": [],
    }


def _validate_three_photo_git_boundary() -> dict[str, Any]:
    return _validate_git_boundary(
        expected_start_head=THREE_PHOTO_EXPECTED_START_HEAD,
        expected_paths=THREE_PHOTO_ALLOWED_DIRTY_PATHS,
    )


def _next_output_dir() -> Path:
    root = _repo_path(OUTPUT_ROOT)
    pattern = re.compile(rf"^{re.escape(OUTPUT_PREFIX)}(\d+)$")
    attempts = [
        int(match.group(1))
        for path in root.glob(f"{OUTPUT_PREFIX}*")
        if (match := pattern.fullmatch(path.name)) and path.is_dir()
    ]
    return root / f"{OUTPUT_PREFIX}{max(attempts, default=0) + 1:02d}"


def _check_disk_and_output_boundary(output_dir: Path) -> None:
    drive = Path(output_dir.resolve().anchor or Path.cwd().anchor)
    free_bytes = os.statvfs(drive).f_bavail * os.statvfs(drive).f_frsize if hasattr(os, "statvfs") else None
    if free_bytes is None:
        import shutil

        free_bytes = shutil.disk_usage(drive).free
    if free_bytes <= 10 * 1024**3:
        raise RuntimeError(f"D: free space is below the 10 GiB visualization safety floor: {free_bytes} bytes")


def _save_new_figure_file(figure: plt.Figure, target: Path, *, dpi: int | None = None) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite figure: {target}")
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        figure.savefig(temporary, format=target.suffix[1:], dpi=dpi, bbox_inches="tight")
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite figure: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure_bundle(
    figure: plt.Figure,
    output_base: Path,
    *,
    axes: Sequence[plt.Axes],
    panel_ids: Sequence[str],
    row_groups: Sequence[Sequence[str]],
    column_groups: Sequence[Sequence[str]],
    raster_dpi: int,
) -> list[Path]:
    alignment_json = output_base.with_suffix(".alignment.json")
    alignment_svg = output_base.with_suffix(".alignment.svg")
    if alignment_json.exists() or alignment_svg.exists():
        raise FileExistsError(f"refusing to overwrite figure alignment artifact: {output_base}")
    alignment_json_tmp = output_base.with_name(f".{output_base.name}.alignment.json.tmp")
    alignment_svg_tmp = output_base.with_name(f".{output_base.name}.alignment.svg.tmp")
    try:
        s9v.require_matplotlib_panel_alignment(
            figure,
            axes=list(axes),
            panel_ids=list(panel_ids),
            row_groups=[list(group) for group in row_groups],
            column_groups=[list(group) for group in column_groups],
            exclude_axes=[],
            json_out=str(alignment_json_tmp),
            overlay_svg=str(alignment_svg_tmp),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
        for temporary, target in (
            (alignment_json_tmp, alignment_json),
            (alignment_svg_tmp, alignment_svg),
        ):
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite figure alignment artifact: {target}") from exc
    finally:
        alignment_json_tmp.unlink(missing_ok=True)
        alignment_svg_tmp.unlink(missing_ok=True)
    saved = [alignment_json, alignment_svg]
    for suffix in FIGURE_EXPORT_SUFFIXES:
        dpi = raster_dpi if suffix == ".png" else None
        target = output_base.with_suffix(suffix)
        _save_new_figure_file(figure, target, dpi=dpi)
        saved.append(target)
    plt.close(figure)
    return saved


def _raw_input_quality(image: torch.Tensor) -> dict[str, Any]:
    if tuple(image.shape) != (1, 28, 28):
        raise RuntimeError(f"MNIST raw image shape changed: {tuple(image.shape)}")
    image_2d = image[0].float().cpu()
    mask = image_2d >= INPUT_THRESHOLD
    coordinates = torch.nonzero(mask, as_tuple=False)
    if coordinates.numel() == 0:
        return {
            "foreground_pixels": 0,
            "foreground_fraction": 0.0,
            "border_touch_pixels": 0,
            "bbox": None,
            "bbox_height": 0,
            "bbox_width": 0,
            "bbox_area_fraction": 0.0,
            "centroid_y": None,
            "centroid_x": None,
            "centroid_radius": None,
            "foreground_mean": 0.0,
            "hard_pass": False,
            "score": 0.0,
        }
    y = coordinates[:, 0]
    x = coordinates[:, 1]
    y_min, y_max = int(y.min()), int(y.max())
    x_min, x_max = int(x.min()), int(x.max())
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    foreground_pixels = int(mask.sum())
    foreground_fraction = foreground_pixels / float(image_2d.numel())
    border_touch = (
        int(mask[0].sum())
        + int(mask[-1].sum())
        + int(mask[:, 0].sum())
        + int(mask[:, -1].sum())
        - int(mask[0, 0])
        - int(mask[0, -1])
        - int(mask[-1, 0])
        - int(mask[-1, -1])
    )
    weighted = image_2d.clamp_min(0.0)
    mass = float(weighted.sum())
    if mass <= 0.0:
        centroid_y = float(y.float().mean())
        centroid_x = float(x.float().mean())
    else:
        yy, xx = torch.meshgrid(
            torch.arange(28, dtype=torch.float32),
            torch.arange(28, dtype=torch.float32),
            indexing="ij",
        )
        centroid_y = float((weighted * yy).sum() / mass)
        centroid_x = float((weighted * xx).sum() / mass)
    centroid_radius = math.sqrt(
        ((centroid_y / 27.0) - 0.5) ** 2 + ((centroid_x / 27.0) - 0.5) ** 2
    )
    bbox_area_fraction = height * width / float(image_2d.numel())
    foreground_mean = float(image_2d[mask].mean())
    margin = min(y_min, x_min, 27 - y_max, 27 - x_max)
    border_score = min(max(margin / 4.0, 0.0), 1.0)
    occupancy_score = math.exp(-abs(foreground_fraction - 0.16) / 0.16)
    centroid_score = max(0.0, 1.0 - centroid_radius / CENTROID_RADIUS_MAX)
    ink_score = min(max(foreground_mean / 0.65, 0.0), 1.0)
    score = border_score + occupancy_score + centroid_score + ink_score
    hard_pass = bool(
        border_touch == 0
        and OCCUPANCY_RANGE[0] <= foreground_fraction <= OCCUPANCY_RANGE[1]
        and BBOX_HEIGHT_RANGE[0] <= height <= BBOX_HEIGHT_RANGE[1]
        and BBOX_WIDTH_RANGE[0] <= width <= BBOX_WIDTH_RANGE[1]
        and centroid_radius <= CENTROID_RADIUS_MAX
    )
    return {
        "foreground_pixels": foreground_pixels,
        "foreground_fraction": foreground_fraction,
        "border_touch_pixels": border_touch,
        "bbox": [y_min, y_max, x_min, x_max],
        "bbox_height": height,
        "bbox_width": width,
        "bbox_area_fraction": bbox_area_fraction,
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
        "centroid_radius": centroid_radius,
        "foreground_mean": foreground_mean,
        "margin_pixels": margin,
        "principal_axis_note": "not used for selection; orientation remains an input-only visual review criterion",
        "hard_pass": hard_pass,
        "score": score,
    }


def _candidate_pool(
    images: torch.Tensor, labels: torch.Tensor
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    if tuple(images.shape) != (60000, 1, 28, 28) or tuple(labels.shape) != (60000,):
        raise RuntimeError("VIS2 requires the complete local MNIST train split")
    if not torch.isfinite(images).all() or images.min() < 0.0 or images.max() > 1.0:
        raise RuntimeError("local MNIST train images are not finite [0,1] tensors")
    candidates: dict[int, list[dict[str, Any]]] = {}
    selected: dict[int, dict[str, Any]] = {}
    for digit in range(10):
        indices = [
            index
            for index in range(CANDIDATE_START, CANDIDATE_STOP)
            if int(labels[index]) == digit
        ][:CANDIDATES_PER_DIGIT]
        if len(indices) != CANDIDATES_PER_DIGIT:
            raise RuntimeError(f"BLOCKED_NO_CLEAR_INPUT_CANDIDATE: digit {digit} has too few candidates")
        records: list[dict[str, Any]] = []
        for index in indices:
            quality = _raw_input_quality(images[index])
            records.append(
                {
                    "dataset_index": index,
                    "label": int(labels[index]),
                    "quality": quality,
                    "input_sha256": tensor_sha256(images[index]),
                }
            )
        candidates[digit] = records
        eligible = [record for record in records if record["quality"]["hard_pass"]]
        if not eligible:
            raise RuntimeError(f"BLOCKED_NO_CLEAR_INPUT_CANDIDATE: digit {digit}")
        selected[digit] = max(
            eligible,
            key=lambda record: (
                float(record["quality"]["score"]),
                -int(record["dataset_index"]),
            ),
        )
    return candidates, selected


def _clear_panel_manifest(
    images: torch.Tensor,
    labels: torch.Tensor,
    candidates: Mapping[int, Sequence[Mapping[str, Any]]],
    selected: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    selected_records: list[dict[str, Any]] = []
    for digit in range(10):
        record = selected[digit]
        index = int(record["dataset_index"])
        if int(labels[index]) != digit:
            raise RuntimeError("selected input label/index consistency check failed")
        quality = dict(record["quality"])
        selected_records.append(
            {
                "digit": digit,
                "dataset_index": index,
                "label": int(labels[index]),
                "input_shape": list(images[index].shape),
                "input_dtype": str(images[index].dtype),
                "input_sha256": tensor_sha256(images[index]),
                "selection_criteria": {
                    "input_threshold": INPUT_THRESHOLD,
                    "border_clear": quality["border_touch_pixels"] == 0,
                    "foreground_completeness_proxy": "non-empty foreground with complete in-frame bounding box",
                    "centroid_radius_max": CENTROID_RADIUS_MAX,
                    "occupancy_range": list(OCCUPANCY_RANGE),
                    "bbox_height_range": list(BBOX_HEIGHT_RANGE),
                    "bbox_width_range": list(BBOX_WIDTH_RANGE),
                    "orientation": "input-only visual review; not inferred from downstream output",
                    "human_readability": "input-only visual review of raw contact sheet",
                },
                "input_only_quality": quality,
                "deterministic_selection_score": float(quality["score"]),
            }
        )
    return {
        "schema": "lightgen_scaled_chen_fiber_vis2_clear_panel_manifest.v1",
        "task_id": TASK_ID,
        "classification": AWAITING_CLASSIFICATION,
        "dataset": "MNIST local train split",
        "split": "train",
        "download": False,
        "candidate_pool": {
            "index_range": [CANDIDATE_START, CANDIDATE_STOP],
            "count_per_digit": CANDIDATES_PER_DIGIT,
            "ordering": "dataset index ascending",
            "source_interval": "same confirmation train interval used by the current Scaled-Chen/S9V authority",
        },
        "selection_method": "input_only",
        "selection_rule": "highest deterministic input-only quality score among hard-pass candidates; dataset-index tie break",
        "selection_criteria": "border clearance, in-frame foreground completeness proxy, centroid, occupancy, and bounded foreground size",
        "selected_before_fiber_evaluation": True,
        "downstream_outputs_used_for_selection": False,
        "selected_panel_order": list(range(10)),
        "selected": selected_records,
        "candidate_records_by_digit": {
            str(digit): [dict(record) for record in candidates[digit]] for digit in range(10)
        },
    }


def _candidate_contact_sheet(
    images: torch.Tensor,
    labels: torch.Tensor,
    candidates: Mapping[int, Sequence[Mapping[str, Any]]],
    selected: Mapping[int, Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    figure, axes = plt.subplots(10, CANDIDATES_PER_DIGIT, figsize=(20.0, 8.0), squeeze=False)
    figure.subplots_adjust(left=0.032, right=0.998, bottom=0.07, top=0.89, wspace=0.10, hspace=0.85)
    panel_ids: list[str] = []
    for digit in range(10):
        selected_index = int(selected[digit]["dataset_index"])
        for column, record in enumerate(candidates[digit]):
            axis = axes[digit, column]
            index = int(record["dataset_index"])
            if int(labels[index]) != digit:
                raise RuntimeError("candidate contact sheet label/index mismatch")
            axis.imshow(
                images[index, 0].numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                origin=IMAGE_ORIGIN,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_title(f"{index} · L{digit}", fontsize=5.0, pad=2.5)
            if index == selected_index:
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#b2182b")
                    spine.set_linewidth(1.5)
            panel_ids.append(f"candidate-r{digit}-c{column}")
        _horizontal_row_label(axes[digit, 0], f"digit {digit}", fontsize=6.5)
    figure.suptitle(
        "VIS2 Stage A · input-only MNIST candidates · red frame = deterministic selection",
        fontsize=10.0,
        y=0.965,
    )
    figure.text(
        0.5,
        0.025,
        "Raw train images only · common grayscale [0,1] · candidate order is dataset-index ascending · no fiber, CMOS, reconstruction, or MPCDAR output was inspected",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )
    row_groups = [panel_ids[row * CANDIDATES_PER_DIGIT : (row + 1) * CANDIDATES_PER_DIGIT] for row in range(10)]
    column_groups = [
        [panel_ids[row * CANDIDATES_PER_DIGIT + column] for row in range(10)]
        for column in range(CANDIDATES_PER_DIGIT)
    ]
    return _save_figure_bundle(
        figure,
        output_dir / "clear_input_candidate_contact_sheet",
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _write_stage_a(output_dir: Path, dataset_root: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite VIS2 Stage A directory: {output_dir}")
    _check_disk_and_output_boundary(output_dir)
    train_images, train_labels = base.load_local_mnist(dataset_root, train=True)
    candidates, selected = _candidate_pool(train_images, train_labels)
    output_dir.mkdir(parents=True, exist_ok=False)
    _candidate_contact_sheet(train_images, train_labels, candidates, selected, output_dir)
    manifest = _clear_panel_manifest(train_images, train_labels, candidates, selected)
    atomic_write_new_json(
        output_dir / "clear_panel_manifest.json",
        manifest,
        artifact_kind="VIS2 clear panel manifest",
    )
    return {
        "status": AWAITING_CLASSIFICATION,
        "output_dir": str(output_dir.resolve()),
        "selected": [
            {"digit": item["digit"], "dataset_index": item["dataset_index"], "label": item["label"]}
            for item in manifest["selected"]
        ],
        "selection_method": "input_only",
        "downstream_outputs_used_for_selection": False,
        "next_action": AWAITING_NEXT_ACTION,
    }


def _load_clear_manifest(output_dir: Path, *, manifest_path: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path or output_dir / "clear_panel_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("BLOCKED_CLEAR_PANEL_MANIFEST_MISSING")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("BLOCKED_CLEAR_PANEL_MANIFEST_UNREADABLE") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("BLOCKED_CLEAR_PANEL_MANIFEST_INVALID")
    if (
        manifest.get("schema") != "lightgen_scaled_chen_fiber_vis2_clear_panel_manifest.v1"
        or manifest.get("task_id") != TASK_ID
        or manifest.get("selection_method") != "input_only"
        or manifest.get("selected_before_fiber_evaluation") is not True
        or manifest.get("downstream_outputs_used_for_selection") is not False
        or manifest.get("selected_panel_order") != list(range(10))
    ):
        raise RuntimeError("BLOCKED_CLEAR_PANEL_MANIFEST_CONTRACT_CHANGED")
    selected = manifest.get("selected")
    if not isinstance(selected, list) or len(selected) != 10:
        raise RuntimeError("BLOCKED_CLEAR_PANEL_MANIFEST_SELECTION_INVALID")
    for digit, item in enumerate(selected):
        if not isinstance(item, Mapping) or item.get("digit") != digit or item.get("label") != digit:
            raise RuntimeError("BLOCKED_CLEAR_PANEL_LABEL_INDEX_MISMATCH")
    return manifest


def _approve_clear_panel(output_dir: Path) -> dict[str, Any]:
    _validate_git_boundary()
    if not output_dir.is_dir():
        raise RuntimeError(f"BLOCKED_CLEAR_PANEL_DIRECTORY_MISSING: {output_dir}")
    manifest = _load_clear_manifest(output_dir)
    contact = output_dir / "clear_input_candidate_contact_sheet.png"
    if not contact.is_file():
        raise RuntimeError("BLOCKED_CLEAR_PANEL_CONTACT_SHEET_MISSING")
    review_path = output_dir / "clear_panel_review.json"
    if review_path.exists():
        raise FileExistsError(f"refusing to overwrite clear panel review: {review_path}")
    review = {
        "schema": "lightgen_scaled_chen_fiber_vis2_clear_panel_review.v1",
        "task_id": TASK_ID,
        "decision": "APPROVED",
        "classification": "canonical_clear_panel_frozen",
        "review_basis": "input-only contact sheet and raw selected images; no downstream outputs were available to the review",
        "selection_method": "input_only",
        "downstream_outputs_used_for_selection": False,
        "selected_before_fiber_evaluation": True,
        "clear_panel_manifest_sha256": file_sha256(output_dir / "clear_panel_manifest.json"),
        "selected": manifest["selected"],
        "reviewed_contact_sheet": str(contact.name),
        "human_readability_and_orientation": "approved from raw input display only",
    }
    atomic_write_new_json(review_path, review, artifact_kind="VIS2 clear panel review")
    return {
        "status": "clear_panel_review_approved",
        "output_dir": str(output_dir.resolve()),
        "next_action": "run VIS2 full CUDA visualization",
        "clear_panel_manifest_sha256": review["clear_panel_manifest_sha256"],
    }


def _validate_clear_panel_against_dataset(
    manifest: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    selected = manifest["selected"]
    indices = torch.tensor([int(item["dataset_index"]) for item in selected], dtype=torch.long)
    selected_labels = labels.index_select(0, indices)
    expected_labels = torch.arange(10, dtype=torch.long)
    if not torch.equal(selected_labels, expected_labels):
        raise RuntimeError("BLOCKED_CLEAR_PANEL_LABEL_INDEX_MISMATCH")
    selected_images = images.index_select(0, indices)
    for item, image, label in zip(selected, selected_images, selected_labels, strict=True):
        if tensor_sha256(image) != item["input_sha256"]:
            raise RuntimeError("BLOCKED_CLEAR_PANEL_INPUT_HASH_MISMATCH")
        if int(label) != int(item["digit"]):
            raise RuntimeError("BLOCKED_CLEAR_PANEL_LABEL_INDEX_MISMATCH")
    return selected_images, selected_labels, indices, {
        int(item["digit"]): int(item["dataset_index"]) for item in selected
    }


def _stage_plane_metadata(field: Any, *, representation: str, **extra: Any) -> dict[str, Any]:
    return {
        "representation": representation,
        "plane": field.plane.to_dict(),
        "provenance": field.provenance.to_dict(),
        **extra,
    }


def _capture_full_pipeline(
    model: bogt.ScaledChenBOGTModel,
    amplitude: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    target: torch.Tensor,
    authority: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Replay the common frontend once, then branch at the shared pre-OLS."""
    slm_command = model.system.command(amplitude)
    slm_field = model.system.slm(slm_command)
    encoder_input = model.system.front(slm_field)
    encoded_raw, encoder_records_raw, _ = model.system.encoder(encoder_input, "encoder")
    if len(encoder_records_raw) != 2:
        raise RuntimeError("VIS2 registered encoder path did not produce two layers")
    encoder_records = bogt._attenuate_records(encoder_records_raw)
    pre_ols = bogt._attenuate(
        encoded_raw,
        bogt.FIELD_TRANSMISSION**2,
        "fixed_encoder_group_transmission",
    )
    common = {
        "input": images,
        "slm_command": slm_command.tensor,
        "slm_output": slm_field.tensor,
        "encoder_input": encoder_input.tensor,
        "encoder_layer_1": encoder_records[0].tensor,
        "pre_ols": pre_ols.tensor,
    }
    if not all(torch.isfinite(value).all() for value in common.values()):
        raise RuntimeError("VIS2 common frontend produced a non-finite value")
    pre_energy = pre_ols.tensor[:, 0].abs().square().sum(dim=(-2, -1)) * (
        model.system.encoder_plane.pixel_pitch_m[0] * model.system.encoder_plane.pixel_pitch_m[1]
    )
    if not torch.isfinite(pre_energy).all():
        raise RuntimeError("VIS2 shared pre-OLS energy is non-finite")
    shared_pointer = pre_ols.tensor.data_ptr()

    native = model.ols
    fmf = FewModeFiberArrayOLS(model.system.encoder_plane, FewModeFiberArraySpec()).to(pre_ols.tensor.device).eval()
    mmf = NestedMultimodeFiberArrayOLS(model.system.encoder_plane).to(pre_ols.tensor.device).eval()
    operators: dict[str, torch.nn.Module] = {"SMF": native, "FMF": fmf, "MMF-M6": mmf}
    for arm, operator in operators.items():
        vis1._require_frozen_operator(operator, ARM_TOTAL_MODES[arm])

    arm_data: dict[str, dict[str, Any]] = {}
    z_by_arm: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for arm in ARM_ORDER:
            if pre_ols.tensor.data_ptr() != shared_pointer:
                raise RuntimeError("VIS2 shared pre-OLS tensor identity changed before arm replay")
            operator = operators[arm]
            post_fiber = operator(pre_ols)
            diagnostics = dict(operator.last_diagnostics)  # type: ignore[attr-defined]
            decoded_raw, decoder_records_raw, _ = model.system.decoder(post_fiber, "decoder")
            if len(decoder_records_raw) != 2:
                raise RuntimeError("VIS2 registered decoder path did not produce two layers")
            decoder_records = bogt._attenuate_records(decoder_records_raw)
            decoded = bogt._attenuate(
                decoded_raw,
                bogt.FIELD_TRANSMISSION**2,
                "fixed_decoder_group_transmission",
            )
            chip = vis1.ComplexField(decoded.tensor, model.system.chip_plane, decoded.provenance)
            camera_field = model.chen_relay(chip)
            camera_intensity = camera_field.tensor.abs().square()
            if not torch.isfinite(camera_intensity).all():
                raise RuntimeError(f"{arm} CMOS intensity is non-finite")
            flat, family, family_order, flat_names, ordering = vis1._family_coefficients(
                arm, operator, diagnostics
            )
            modal_power = family.abs().square().sum(dim=2)
            if not torch.isfinite(modal_power).all():
                raise RuntimeError(f"{arm} modal power is non-finite")
            predictions, reconstruction_metrics = s9v._replay_full_path_reconstructions(
                authority,
                {"cmos_detector_intensity": camera_intensity},
                images,
                labels,
                target,
            )
            z = mra._feature_projection(camera_intensity.detach().cpu().float(), complex_value=False).float()
            if tuple(z.shape) != (10, 256) or not torch.isfinite(z).all():
                raise RuntimeError(f"{arm} detector feature projection is not finite [10,256]")
            fiber_contract, fiber_metrics, _, _ = vis1._operator_contract(
                arm,
                operator,
                diagnostics,
                post_fiber,
                pre_energy,
            )
            fiber_metrics["modal_power_by_family"]["family_order"] = family_order
            arm_metrics = vis1._build_arm_metrics(
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
                "flat_coefficients": flat,
                "family_coefficients": family,
                "family_order": family_order,
                "flat_mode_family_names": flat_names,
                "ordering_contract": ordering["ordering_contract"],
                "modal_power": modal_power,
                "post_fiber": post_fiber.tensor,
                "decoder_layer_1": decoder_records[0].tensor,
                "decoder_layer_2": decoder_records[1].tensor,
                "cmos_field": camera_field.tensor,
                "cmos_intensity": camera_intensity,
                "mpcdar": predictions["mpcdar_rank16_float32"],
                "z": z,
                "predictions": predictions,
                "reconstruction_metrics": reconstruction_metrics,
                "stage_metadata": {
                    "modal": {
                        "representation": "complex_port_coefficients",
                        "display_representation": "family_power_and_family_phase",
                        "port_count": 100,
                        "mode_family_order": list(family_order),
                        "flat_mode_count": ARM_TOTAL_MODES[arm],
                        "flat_mode_family_names": list(flat_names),
                        "ordering_contract": ordering["ordering_contract"],
                        "synthetic_extra_families": list(MMF_SYNTHETIC_EXTRA_FAMILIES) if arm == "MMF-M6" else [],
                    },
                    "post_fiber": _stage_plane_metadata(
                        post_fiber,
                        representation="complex_field",
                        alias_of="decoder_input",
                        alias_note="Fiber output is the decoder input boundary; no extra propagation slice is inserted.",
                    ),
                    "decoder_layer_1": _stage_plane_metadata(
                        decoder_records[0], representation="complex_field"
                    ),
                    "decoder_layer_2": _stage_plane_metadata(
                        decoder_records[1], representation="complex_field"
                    ),
                    "cmos_field": _stage_plane_metadata(
                        camera_field,
                        representation="complex_field_upstream_of_detector",
                        detector_readout="intensity_only",
                        phase_note="CMOS incident phase is a model-side upstream field quantity, not a detector readout.",
                    ),
                    "cmos_intensity": {
                        "representation": "intensity_frame",
                        "formula": "I=|E|^2",
                        "phase_available": False,
                        "plane": camera_field.plane.to_dict(),
                        "provenance": camera_field.provenance.to_dict(),
                    },
                    "mpcdar": {
                        "representation": "frozen_reconstruction",
                        "range": [0.0, 1.0],
                        "readout": "existing frozen S9 MPCDAR rank16 float32",
                    },
                },
            }
            z_by_arm[arm] = z
            if pre_ols.tensor.data_ptr() != shared_pointer:
                raise RuntimeError("VIS2 fiber arm replay mutated the shared pre-OLS tensor")
    shared = {
        "pre_energy": pre_energy,
        "pre_ols_pointer_identity": True,
        "common_stage_metadata": {
            "input": {
                "representation": "raw_mnist_intensity",
                "shape": [1, 28, 28],
                "range": [0.0, 1.0],
                "phase_available": False,
            },
            "slm_command": {
                "representation": "real_amplitude_command",
                "phase_available": False,
                "calibration_id": model.system.slm.calibration_id,
                "plane": model.system.slm_plane.to_dict(),
            },
            "slm_output": _stage_plane_metadata(slm_field, representation="complex_field"),
            "encoder_input": _stage_plane_metadata(encoder_input, representation="complex_field"),
            "encoder_layer_1": _stage_plane_metadata(encoder_records[0], representation="complex_field"),
            "pre_ols": _stage_plane_metadata(
                pre_ols,
                representation="complex_field",
                alias_of="encoder_layer_2",
                alias_note="The second encoder record and OLS input share this boundary; fixed encoder group transmission is applied here.",
            ),
        },
        "shared_pre_ols_tensor_shape": list(pre_ols.tensor.shape),
    }
    return common, arm_data, shared, {"z_by_arm": z_by_arm, "operators": operators}


def _spatial_array(value: torch.Tensor, position: int, key: str) -> np.ndarray:
    return s9v._display_array(value, position, rotate=key in ROTATE_SPATIAL_KEYS)


def _phase_view(value: torch.Tensor, position: int, key: str) -> tuple[np.ndarray, float, float]:
    raw = _spatial_array(value, position, key)
    amplitude = np.abs(raw)
    threshold = max(float(amplitude.max()) * PHASE_ALPHA, PHASE_FLOOR)
    phase = np.where(amplitude >= threshold, np.angle(raw), np.nan)
    return phase, threshold, float(np.isfinite(phase).mean())


def _modal_arrays(arm_data: Mapping[str, Any], position: int) -> tuple[np.ndarray, np.ndarray]:
    family = arm_data["family_coefficients"][position].detach().cpu().numpy()
    power = np.abs(family) ** 2
    threshold = max(float(np.abs(family).max()) * PHASE_ALPHA, PHASE_FLOOR)
    phase = np.where(np.abs(family) >= threshold, np.angle(family), np.nan)
    return power, phase


def _shared_display_scales(common: Mapping[str, torch.Tensor], arm_data: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scales: dict[str, float] = {"input": 1.0, "slm_command": 1.0, "reconstruction": 1.0}
    sources: dict[str, str] = {
        "input": "fixed raw [0,1]",
        "slm_command": "fixed amplitude command [0,1]",
        "reconstruction": "fixed reconstruction range [0,1]",
    }
    for key in ("slm_output", "encoder_input", "encoder_layer_1", "pre_ols"):
        arrays = [_spatial_array(common[key], position, key) for position in range(10)]
        scales[key] = s9v._shared_p995(arrays)
        sources[key] = "common 10-sample panel"
    for key in ("post_fiber", "decoder_layer_1", "decoder_layer_2", "cmos_field"):
        arrays = [
            _spatial_array(arm_data[arm][key], position, key)
            for arm in ARM_ORDER
            for position in range(10)
        ]
        scales[key] = s9v._shared_p995(arrays)
        sources[key] = "joint SMF/FMF/MMF-M6 30-sample panel"
    modal_arrays = [
        _modal_arrays(arm_data[arm], position)[0]
        for arm in ARM_ORDER
        for position in range(10)
    ]
    scales["modal"] = s9v._shared_p995(modal_arrays)
    sources["modal"] = "joint SMF/FMF/MMF-M6 family-power panel"
    intensity_arrays = [
        arm_data[arm]["cmos_intensity"][position].detach().cpu().float().numpy()
        for arm in ARM_ORDER
        for position in range(10)
    ]
    scales["cmos_intensity"] = s9v._shared_p995(intensity_arrays)
    sources["cmos_intensity"] = "joint SMF/FMF/MMF-M6 detector panel"
    if any(not math.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise RuntimeError("shared display scale is invalid")
    return {
        "values": scales,
        "sources": sources,
        "amplitude_policy": "same stage uses one p99.5 display limit across all applicable arms and ten samples",
        "phase_policy": "raw angle(E) in [-pi,pi] with display-only mask max(0.01*max(abs(E)),1e-12) per stage and sample",
        "intensity_policy": "joint three-arm common p99.5 display limit; no per-image or per-arm normalization",
        "reconstruction_policy": "fixed [0,1]",
        "no_energy_equalization": True,
    }


def _imshow_amplitude(
    axis: plt.Axes,
    array: np.ndarray,
    scale: float,
    *,
    cmap: str = "magma",
    spatial: bool = True,
) -> None:
    axis.imshow(
        np.clip(np.abs(array) / max(scale, PHASE_FLOOR), 0.0, 1.0),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal" if spatial else "auto",
        origin=OPTICAL_ORIGIN if spatial else MODAL_ORIGIN,
    )


def _imshow_phase(axis: plt.Axes, phase: np.ndarray, *, spatial: bool = True) -> None:
    axis.set_facecolor("#e8e8e8")
    axis.imshow(
        np.ma.masked_invalid(phase),
        cmap="twilight",
        norm=Normalize(-np.pi, np.pi),
        interpolation="nearest",
        aspect="equal" if spatial else "auto",
        origin=OPTICAL_ORIGIN if spatial else MODAL_ORIGIN,
    )


def _imshow_na(axis: plt.Axes, message: str = "phase N/A") -> None:
    axis.set_facecolor("#e8e8e8")
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=5.2,
        color="#444444",
    )


def _finish_image_axis(axis: plt.Axes, *, spatial: bool = True) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal" if spatial else "auto", adjustable="datalim")
    axis.set_box_aspect(1.0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _render_clear_inputs(
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    output_dir: Path,
) -> list[Path]:
    figure, axes = plt.subplots(2, 5, figsize=(7.2, 4.2), squeeze=False)
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.78, wspace=0.16, hspace=0.42)
    panel_ids: list[str] = []
    for position, (label, index) in enumerate(zip(labels, indices, strict=True)):
        row, column = divmod(position, 5)
        axis = axes[row, column]
        axis.imshow(
            images[position, 0].detach().cpu().numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
            origin=IMAGE_ORIGIN,
        )
        axis.set_title(f"digit {int(label)} · idx {int(index)}", fontsize=6.8, pad=3.0)
        _finish_image_axis(axis)
        panel_ids.append(f"clear-input-r{row}-c{column}")
    figure.suptitle(
        "Figure 0 · canonical clear MNIST input panel · VIS2 input-only authority",
        fontsize=10.0,
        y=0.95,
    )
    figure.text(
        0.5,
        0.045,
        "Raw train images · grayscale [0,1] · no crop, rotation, recentering, enhancement, or downstream selection",
        ha="center",
        va="bottom",
        fontsize=6.2,
    )
    row_groups = [panel_ids[row * 5 : (row + 1) * 5] for row in range(2)]
    column_groups = [[panel_ids[row * 5 + column] for row in range(2)] for column in range(5)]
    return _save_figure_bundle(
        figure,
        output_dir / _figure_manifest_names()["figure_0"],
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _render_full_pipeline_page(
    common: Mapping[str, torch.Tensor],
    arm_data: Mapping[str, Mapping[str, Any]],
    target: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    scales: Mapping[str, Any],
    output_dir: Path,
    page_name: str,
    positions: Sequence[int],
) -> list[Path]:
    scale = scales["values"]
    columns: list[tuple[str, str, str | None, str]] = [
        ("Input", "input", None, "real"),
        ("SLM\ncommand", "slm_command", None, "real"),
        ("SLM output\namp", "slm_output", None, "amp"),
        ("SLM output\nphase", "slm_output", None, "phase"),
        ("Encoder input\namp", "encoder_input", None, "amp"),
        ("Encoder input\nphase", "encoder_input", None, "phase"),
        ("Encoder L1\namp", "encoder_layer_1", None, "amp"),
        ("Encoder L1\nphase", "encoder_layer_1", None, "phase"),
        ("Pre-OLS\namp", "pre_ols", None, "amp"),
        ("Pre-OLS\nphase", "pre_ols", None, "phase"),
    ]
    for arm in ARM_ORDER:
        columns.extend(
            [
                (f"{arm}\nmodal power", "modal", arm, "modal"),
                (f"{arm}\npost-fiber amp", "post_fiber", arm, "amp"),
                (f"{arm}\npost-fiber phase", "post_fiber", arm, "phase"),
                (f"{arm}\nDecoder L1 amp", "decoder_layer_1", arm, "amp"),
                (f"{arm}\nDecoder L1 phase", "decoder_layer_1", arm, "phase"),
                (f"{arm}\nDecoder L2 amp", "decoder_layer_2", arm, "amp"),
                (f"{arm}\nDecoder L2 phase", "decoder_layer_2", arm, "phase"),
                (f"{arm}\nCMOS inc. amp", "cmos_field", arm, "amp"),
                (f"{arm}\nCMOS inc. phase", "cmos_field", arm, "phase"),
                (f"{arm}\nCMOS intensity", "cmos_intensity", arm, "intensity"),
                (f"{arm}\nMPCDAR R16 F32", "mpcdar", arm, "reconstruction"),
            ]
        )
    figure, axes = plt.subplots(
        len(positions),
        len(columns),
        figsize=(max(24.0, len(columns) * 0.64), max(3.8, len(positions) * 0.95)),
        squeeze=False,
    )
    figure.subplots_adjust(left=0.03, right=0.998, bottom=0.13, top=0.78, wspace=0.045, hspace=0.22)
    panel_ids: list[str] = []
    for row, position in enumerate(positions):
        for column, (title, key, arm, kind) in enumerate(columns):
            axis = axes[row, column]
            if key == "input":
                value = common["input"][position, 0].numpy()
                axis.imshow(
                    value,
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                    aspect="equal",
                    origin=IMAGE_ORIGIN,
                )
            elif key == "slm_command":
                _imshow_amplitude(axis, common[key][position, 0].detach().cpu().numpy(), scale[key], cmap="gray")
            elif arm is None:
                raw = _spatial_array(common[key], position, key)
                if kind == "amp":
                    _imshow_amplitude(axis, raw, scale[key])
                else:
                    phase, _, _ = _phase_view(common[key], position, key)
                    _imshow_phase(axis, phase)
            elif key == "modal":
                power, _ = _modal_arrays(arm_data[arm], position)
                _imshow_amplitude(axis, power, scale["modal"])
            elif key == "cmos_intensity":
                value = arm_data[arm][key][position, 0].detach().cpu().numpy()
                _imshow_amplitude(axis, value, scale["cmos_intensity"], cmap="gray")
            elif key == "mpcdar":
                value = arm_data[arm][key][position, 0].detach().cpu().numpy()
                axis.imshow(
                    np.clip(value, 0.0, 1.0),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                    aspect="equal",
                    origin=IMAGE_ORIGIN,
                )
            else:
                value = arm_data[arm][key]
                if kind == "amp":
                    _imshow_amplitude(axis, _spatial_array(value, position, key), scale[key])
                else:
                    phase, _, _ = _phase_view(value, position, key)
                    _imshow_phase(axis, phase)
            _finish_image_axis(axis)
            if row == 0:
                axis.set_title(title, fontsize=5.2, pad=2.2, linespacing=1.2)
            panel_id = f"{page_name}-r{row}-c{column}"
            panel_ids.append(panel_id)
        _horizontal_row_label(
            axes[row, 0],
            f"digit {int(labels[position])}\nidx {int(indices[position])}",
            fontsize=6.1,
        )
    figure.suptitle(
        f"Figure A · VIS2 full-pipeline horizontal comparison · digits {int(labels[positions[0]])}–{int(labels[positions[-1]])}",
        fontsize=10.0,
        y=0.965,
    )
    figure.text(
        0.5,
        0.045,
        "Common input→SLM→encoder→pre-OLS appears once, then branches at the shared pre-OLS tensor. Amplitude uses same-stage joint p99.5 display limits; phase is raw [−π,π] with display-only 1% mask; CMOS incident phase is upstream field, detector phase is not measured. Post-fiber = decoder input. Synthetic MMF-M6 families are not physical guided modes.",
        ha="center",
        va="center",
        fontsize=5.6,
        wrap=True,
    )
    row_groups = [panel_ids[row * len(columns) : (row + 1) * len(columns)] for row in range(len(positions))]
    column_groups = [
        [panel_ids[row * len(columns) + column] for row in range(len(positions))]
        for column in range(len(columns))
    ]
    return _save_figure_bundle(
        figure,
        output_dir / page_name,
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _render_full_pipeline_index(
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    arm_data: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    columns = ("Input", "SMF\nreconstruction", "FMF\nreconstruction", "MMF-M6\nreconstruction")
    figure, axes = plt.subplots(10, 4, figsize=(6.5, 10.0), squeeze=False)
    figure.subplots_adjust(left=0.10, right=0.995, bottom=0.065, top=0.91, wspace=0.08, hspace=0.20)
    panel_ids: list[str] = []
    for row, (image, label, index) in enumerate(zip(images, labels, indices, strict=True)):
        axis = axes[row, 0]
        axis.imshow(
            image[0].numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            origin=IMAGE_ORIGIN,
        )
        for column, arm in enumerate(ARM_ORDER, start=1):
            axis = axes[row, column]
            axis.imshow(
                np.clip(arm_data[arm]["mpcdar"][row, 0].detach().cpu().numpy(), 0.0, 1.0),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                origin=IMAGE_ORIGIN,
            )
        for column in range(4):
            axis = axes[row, column]
            _finish_image_axis(axis)
            if row == 0:
                axis.set_title(columns[column], fontsize=6.6, pad=3.0)
            panel_ids.append(f"index-r{row}-c{column}")
        _horizontal_row_label(
            axes[row, 0],
            f"{int(label)}\nidx {int(index)}",
            fontsize=6.1,
        )
    figure.suptitle(
        "Figure A index · VIS2 clear-input full-pipeline split pages",
        fontsize=10.0,
        y=0.965,
    )
    figure.text(
        0.5,
        0.022,
        "Detailed full-pipeline pages: fiber_vis2_full_pipeline_digits_0_4 and fiber_vis2_full_pipeline_digits_5_9",
        ha="center",
        va="bottom",
        fontsize=6.0,
    )
    row_groups = [panel_ids[row * 4 : (row + 1) * 4] for row in range(10)]
    column_groups = [[panel_ids[row * 4 + column] for row in range(10)] for column in range(4)]
    return _save_figure_bundle(
        figure,
        output_dir / "fiber_vis2_full_pipeline_horizontal",
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _modal_labels(arm_data: Mapping[str, Any]) -> list[str]:
    arm = arm_data["arm"]
    labels = list(arm_data["family_order"])
    if arm == "MMF-M6":
        return [
            f"{name} [synthetic]" if name in MMF_SYNTHETIC_EXTRA_FAMILIES else name
            for name in labels
        ]
    return labels


def _render_fiber_closeup(
    common: Mapping[str, torch.Tensor],
    arm_data: Mapping[str, Mapping[str, Any]],
    labels: torch.Tensor,
    indices: Sequence[int],
    scales: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    columns: list[tuple[str, str, str | None]] = [
        ("Common pre-OLS\namplitude", "pre_ols_amp", None),
        ("Common pre-OLS\nphase", "pre_ols_phase", None),
    ]
    for arm in ARM_ORDER:
        columns.extend(
            [
                (f"{arm}\nmodal power", "modal", arm),
                (f"{arm}\npost-fiber amplitude", "post_amp", arm),
                (f"{arm}\npost-fiber phase", "post_phase", arm),
            ]
        )
    figure, axes = plt.subplots(10, len(columns), figsize=(14.5, 11.0), squeeze=False)
    figure.subplots_adjust(left=0.04, right=0.998, bottom=0.085, top=0.86, wspace=0.07, hspace=0.24)
    panel_ids: list[str] = []
    pre_phase_cache = [_phase_view(common["pre_ols"], position, "pre_ols")[0] for position in range(10)]
    for row, (label, index) in enumerate(zip(labels, indices, strict=True)):
        for column, (title, key, arm) in enumerate(columns):
            axis = axes[row, column]
            if key == "pre_ols_amp":
                _imshow_amplitude(axis, _spatial_array(common["pre_ols"], row, "pre_ols"), scales["values"]["pre_ols"])
            elif key == "pre_ols_phase":
                _imshow_phase(axis, pre_phase_cache[row])
            elif key == "modal":
                power, _ = _modal_arrays(arm_data[arm], row)
                _imshow_amplitude(axis, power, scales["values"]["modal"])
            elif key == "post_amp":
                _imshow_amplitude(
                    axis,
                    _spatial_array(arm_data[arm]["post_fiber"], row, "post_fiber"),
                    scales["values"]["post_fiber"],
                )
            else:
                _imshow_phase(axis, _phase_view(arm_data[arm]["post_fiber"], row, "post_fiber")[0])
            _finish_image_axis(axis)
            if key == "modal" and row == 0:
                family_labels = _modal_labels({**arm_data[arm], "arm": arm})
                axis.set_yticks(range(len(family_labels)), family_labels, fontsize=5.0)
                axis.yaxis.tick_right()
                axis.tick_params(
                    axis="y",
                    pad=2.0,
                    length=1.5,
                    left=False,
                    right=True,
                    labelleft=False,
                    labelright=True,
                )
            if row == 0:
                axis.set_title(title, fontsize=6.0, pad=3.0, linespacing=1.2)
            panel_ids.append(f"closeup-r{row}-c{column}")
        _horizontal_row_label(
            axes[row, 0],
            f"digit {int(label)}\nidx {int(index)}",
            fontsize=6.0,
        )
    figure.suptitle(
        "Figure B · VIS2 fiber close-up · one shared pre-OLS tensor, three modal branches",
        fontsize=10.0,
        y=0.965,
    )
    figure.text(
        0.5,
        0.035,
        "The two common pre-OLS panels are rendered once per digit and precede all arm-specific panels. Modal columns retain each arm's actual family×port representation; shared modal p99.5 scale, raw phase [−π,π], display-only mask. MMF-M6 extra families are synthetic.",
        ha="center",
        va="center",
        fontsize=5.8,
    )
    row_groups = [panel_ids[row * len(columns) : (row + 1) * len(columns)] for row in range(10)]
    column_groups = [[panel_ids[row * len(columns) + column] for row in range(10)] for column in range(len(columns))]
    return _save_figure_bundle(
        figure,
        output_dir / "fiber_vis2_fiber_closeup_horizontal",
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _render_cmos_reconstruction(
    target: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    arm_data: Mapping[str, Mapping[str, Any]],
    scales: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    columns = (
        ("Target", None, "target"),
        ("SMF CMOS", "SMF", "cmos_intensity"),
        ("SMF reconstruction", "SMF", "mpcdar"),
        ("FMF CMOS", "FMF", "cmos_intensity"),
        ("FMF reconstruction", "FMF", "mpcdar"),
        ("MMF-M6 CMOS", "MMF-M6", "cmos_intensity"),
        ("MMF-M6 reconstruction", "MMF-M6", "mpcdar"),
    )
    figure, axes = plt.subplots(10, len(columns), figsize=(7.2, 10.3), squeeze=False)
    figure.subplots_adjust(left=0.095, right=0.998, bottom=0.065, top=0.90, wspace=0.055, hspace=0.20)
    panel_ids: list[str] = []
    for row, (label, index) in enumerate(zip(labels, indices, strict=True)):
        for column, (title, arm, key) in enumerate(columns):
            axis = axes[row, column]
            if key == "target":
                array = target[row, 0].detach().cpu().numpy()
                axis.imshow(
                    np.clip(array, 0.0, 1.0),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                    origin=IMAGE_ORIGIN,
                )
            elif key == "cmos_intensity":
                _imshow_amplitude(
                    axis,
                    arm_data[arm][key][row, 0].detach().cpu().numpy(),
                    scales["values"]["cmos_intensity"],
                    cmap="gray",
                )
            else:
                array = arm_data[arm][key][row, 0].detach().cpu().numpy()
                axis.imshow(
                    np.clip(array, 0.0, 1.0),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                    origin=IMAGE_ORIGIN,
                )
            _finish_image_axis(axis)
            if row == 0:
                axis.set_title(title, fontsize=6.5, pad=3.0, linespacing=0.85)
            panel_ids.append(f"cmos-reconstruction-r{row}-c{column}")
        _horizontal_row_label(
            axes[row, 0],
            f"digit {int(label)}\nidx {int(index)}",
            fontsize=6.1,
        )
    figure.suptitle(
        "Figure C · VIS2 CMOS → frozen MPCDAR reconstruction comparison",
        fontsize=10.0,
        y=0.965,
    )
    figure.text(
        0.5,
        0.024,
        "CMOS intensity uses one joint SMF/FMF/MMF-M6 p99.5 display scale; target and reconstruction use fixed [0,1]. These ten clear samples are qualitative visualization evidence only, not overall performance statistics.",
        ha="center",
        va="bottom",
        fontsize=6.0,
    )
    row_groups = [panel_ids[row * len(columns) : (row + 1) * len(columns)] for row in range(10)]
    column_groups = [[panel_ids[row * len(columns) + column] for row in range(10)] for column in range(len(columns))]
    return _save_figure_bundle(
        figure,
        output_dir / "fiber_vis2_cmos_reconstruction_horizontal",
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _render_full_path_arm(
    common: Mapping[str, torch.Tensor],
    arm_data: Mapping[str, Mapping[str, Any]],
    arm: str,
    target: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    scales: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    columns = (
        ("Input", "input", "real"),
        ("SLM command", "slm_command", "real"),
        ("SLM output", "slm_output", "complex"),
        ("Encoder input", "encoder_input", "complex"),
        ("Encoder L1", "encoder_layer_1", "complex"),
        ("Pre-OLS / encoder L2", "pre_ols", "complex"),
        ("Modal representation", "modal", "modal"),
        ("Post-fiber / decoder input", "post_fiber", "complex"),
        ("Decoder L1", "decoder_layer_1", "complex"),
        ("Decoder L2", "decoder_layer_2", "complex"),
        ("CMOS incident field", "cmos_field", "complex"),
        ("CMOS intensity", "cmos_intensity", "intensity"),
        ("MPCDAR R16 F32", "mpcdar", "reconstruction"),
    )
    rows = 20
    figure, axes = plt.subplots(rows, len(columns), figsize=(23.5, 18.0), squeeze=False)
    figure.subplots_adjust(left=0.04, right=0.998, bottom=0.105, top=0.88, wspace=0.045, hspace=0.16)
    panel_ids: list[str] = []
    for position, (label, index) in enumerate(zip(labels, indices, strict=True)):
        for phase_row in (False, True):
            row = position * 2 + int(phase_row)
            for column, (title, key, kind) in enumerate(columns):
                axis = axes[row, column]
                if key == "input":
                    if phase_row:
                        _imshow_na(axis)
                    else:
                        axis.imshow(
                            common["input"][position, 0].numpy(),
                            cmap="gray",
                            vmin=0.0,
                            vmax=1.0,
                            interpolation="nearest",
                            origin=IMAGE_ORIGIN,
                        )
                elif key == "slm_command":
                    if phase_row:
                        _imshow_na(axis)
                    else:
                        _imshow_amplitude(axis, common[key][position, 0].detach().cpu().numpy(), scales["values"][key], cmap="gray")
                elif key in {"cmos_intensity", "mpcdar"}:
                    if phase_row:
                        _imshow_na(axis)
                    elif key == "cmos_intensity":
                        _imshow_amplitude(axis, arm_data[arm][key][position, 0].detach().cpu().numpy(), scales["values"][key], cmap="gray")
                    else:
                        axis.imshow(
                            np.clip(arm_data[arm][key][position, 0].detach().cpu().numpy(), 0.0, 1.0),
                            cmap="gray",
                            vmin=0.0,
                            vmax=1.0,
                            interpolation="nearest",
                            origin=IMAGE_ORIGIN,
                        )
                elif key == "modal":
                    if phase_row:
                        _, modal_phase = _modal_arrays(arm_data[arm], position)
                        _imshow_phase(axis, modal_phase)
                    else:
                        power, _ = _modal_arrays(arm_data[arm], position)
                        _imshow_amplitude(axis, power, scales["values"]["modal"])
                else:
                    value = common[key] if key in common else arm_data[arm][key]
                    if phase_row:
                        _imshow_phase(axis, _phase_view(value, position, key)[0])
                    else:
                        _imshow_amplitude(axis, _spatial_array(value, position, key), scales["values"][key])
                _finish_image_axis(axis)
                if key == "modal" and row == 0:
                    family_labels = _modal_labels({**arm_data[arm], "arm": arm})
                    axis.set_yticks(range(len(family_labels)), family_labels, fontsize=5.0)
                    axis.tick_params(axis="y", pad=3.0, length=1.5)
                if row == 0:
                    axis.set_title(title, fontsize=5.8, pad=3.0, linespacing=0.85)
                panel_ids.append(f"{arm}-r{row}-c{column}")
            _horizontal_row_label(
                axes[row, 0],
                f"digit {int(label)}\n{'phase' if phase_row else 'amplitude'}",
                fontsize=5.7,
            )
            if phase_row:
                axes[row, -1].text(
                    1.01,
                    0.5,
                    f"idx {int(index)}",
                    transform=axes[row, -1].transAxes,
                    rotation=0,
                    rotation_mode="anchor",
                    ha="left",
                    va="center",
                    fontsize=5.0,
                )
    figure.suptitle(
        f"Figure D · {arm} full path · VIS2 canonical clear digits 0–9",
        fontsize=10.0,
        y=0.955,
    )
    figure.text(
        0.5,
        0.035,
        "Amplitude rows use the same stage display limits across the three arms; phase rows use raw [−π,π] and the display-only 1% mask. Input, SLM command, CMOS intensity and reconstruction have no phase channel. Modal rows retain the actual family×port metadata. Post-fiber is the decoder-input alias; CMOS incident phase is upstream of square-law detection.",
        ha="center",
        va="center",
        fontsize=5.6,
    )
    row_groups = [panel_ids[row * len(columns) : (row + 1) * len(columns)] for row in range(rows)]
    column_groups = [[panel_ids[row * len(columns) + column] for row in range(rows)] for column in range(len(columns))]
    return _save_figure_bundle(
        figure,
        output_dir / f"fiber_vis2_full_path_{arm.lower().replace('-', '_')}",
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _arm_slug(arm: str) -> str:
    return arm.lower().replace("-", "_")


def _digit_page_label(positions: Sequence[int]) -> str:
    return f"{int(positions[0])}_{int(positions[-1])}"


def _figure_name(arm: str, group: str, positions: Sequence[int]) -> str:
    return Path(
        ARTIFACT_FOLDERS["fiber_information"],
        arm,
        group,
        f"fiber_vis2_{_arm_slug(arm)}_{group}_digits_{_digit_page_label(positions)}",
    ).as_posix()


def _figure_manifest_names() -> dict[str, Any]:
    return {
        "figure_0": Path(
            ARTIFACT_FOLDERS["digit_visualization"],
            "fiber_vis2_clear_0_9_inputs",
        ).as_posix(),
        "reconstruction_overview": [
            Path(
                ARTIFACT_FOLDERS["pipeline_overview"],
                f"fiber_vis2_reconstruction_overview_digits_{_digit_page_label(positions)}",
            ).as_posix()
            for positions in DIGIT_PAGE_RANGES
        ],
        "arm_stage_groups": {
            arm: {
                group: [_figure_name(arm, group, positions) for positions in DIGIT_PAGE_RANGES]
                for group in STAGE_GROUPS
            }
            for arm in ARM_ORDER
        },
    }


def _render_stage_panel(
    axis: plt.Axes,
    common: Mapping[str, torch.Tensor],
    arm_data: Mapping[str, Mapping[str, Any]],
    arm: str,
    position: int,
    key: str,
    kind: str,
    scales: Mapping[str, Any],
    phase_row: bool,
) -> None:
    scale = scales["values"]
    if key == "modal":
        power, phase = _modal_arrays(arm_data[arm], position)
        if phase_row:
            _imshow_phase(axis, phase, spatial=False)
        else:
            _imshow_amplitude(axis, power, scale["modal"], spatial=False)
        _finish_image_axis(axis, spatial=False)
        return
    if kind in {"input", "real", "intensity", "reconstruction"}:
        if phase_row:
            _imshow_na(axis)
        elif key == "input":
            axis.imshow(
                common[key][position, 0].detach().cpu().numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
                origin=IMAGE_ORIGIN,
            )
        elif kind == "real":
            _imshow_amplitude(
                axis,
                common[key][position, 0].detach().cpu().numpy(),
                scale[key],
                cmap="gray",
            )
        elif kind == "intensity":
            _imshow_amplitude(
                axis,
                arm_data[arm][key][position, 0].detach().cpu().numpy(),
                scale[key],
                cmap="gray",
            )
        else:
            axis.imshow(
                np.clip(arm_data[arm][key][position, 0].detach().cpu().numpy(), 0.0, 1.0),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
                origin=IMAGE_ORIGIN,
            )
        _finish_image_axis(axis)
        return
    value = common[key] if key in common else arm_data[arm][key]
    if phase_row:
        _imshow_phase(axis, _phase_view(value, position, key)[0])
    else:
        _imshow_amplitude(axis, _spatial_array(value, position, key), scale[key])
    _finish_image_axis(axis)


def _style_arm_axis(axis: plt.Axes, arm: str) -> None:
    color = ARM_COLORS[arm]
    for name, spine in axis.spines.items():
        spine.set_visible(name != "top")
        spine.set_color(color)
        spine.set_linewidth(0.8)


def _render_reconstruction_overview(
    target: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    arm_data: Mapping[str, Mapping[str, Any]],
    positions: Sequence[int],
    output_dir: Path,
) -> list[Path]:
    columns = (("Target", None),) + tuple((f"{arm} reconstruction", arm) for arm in ARM_ORDER)
    figure, axes = plt.subplots(len(positions), 4, figsize=(7.2, 8.4), squeeze=False)
    figure.subplots_adjust(left=0.19, right=0.99, bottom=0.13, top=0.82, wspace=0.11, hspace=0.33)
    panel_ids: list[str] = []
    for row, position in enumerate(positions):
        for column, (title, arm) in enumerate(columns):
            axis = axes[row, column]
            if arm is None:
                array = target[position, 0].detach().cpu().numpy()
                color = "#555555"
            else:
                array = arm_data[arm]["mpcdar"][position, 0].detach().cpu().numpy()
                color = ARM_COLORS[arm]
            axis.imshow(
                np.clip(array, 0.0, 1.0),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
                origin=IMAGE_ORIGIN,
            )
            _finish_image_axis(axis)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(0.8)
            if row == 0:
                axis.set_title(title, fontsize=6.8, color=color, pad=3.0)
            panel_ids.append(f"reconstruction-overview-r{row}-c{column}")
        _horizontal_row_label(
            axes[row, 0],
            f"digit {int(labels[position])}\nidx {int(indices[position])}",
            fontsize=6.5,
        )
    page_label = _digit_page_label(positions)
    figure.suptitle(
        f"Figure A · compact reconstruction overview · clear digits {page_label.replace('_', '–')}",
        fontsize=10.0,
        y=0.96,
    )
    figure.text(
        0.5,
        0.035,
        "SMF = blue · FMF = green · MMF-M6 = orange; fixed reconstruction range [0,1]. Qualitative ten-sample display only.",
        ha="center",
        va="bottom",
        fontsize=6.0,
    )
    row_groups = [panel_ids[row * 4 : (row + 1) * 4] for row in range(len(positions))]
    column_groups = [[panel_ids[row * 4 + column] for row in range(len(positions))] for column in range(4)]
    return _save_figure_bundle(
        figure,
        output_dir / Path(
            ARTIFACT_FOLDERS["pipeline_overview"],
            f"fiber_vis2_reconstruction_overview_digits_{page_label}",
        ),
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _render_arm_stage_group(
    common: Mapping[str, torch.Tensor],
    arm_data: Mapping[str, Mapping[str, Any]],
    arm: str,
    labels: torch.Tensor,
    indices: Sequence[int],
    positions: Sequence[int],
    scales: Mapping[str, Any],
    group: str,
    output_dir: Path,
) -> list[Path]:
    columns = STAGE_GROUPS[group]
    rows = len(positions) * 2
    figure, axes = plt.subplots(rows, len(columns), figsize=(7.2, 10.6), squeeze=False)
    figure.subplots_adjust(left=0.19, right=0.99, bottom=0.13, top=0.82, wspace=0.09, hspace=0.20)
    panel_ids: list[str] = []
    for position_index, position in enumerate(positions):
        for phase_row in (False, True):
            row = position_index * 2 + int(phase_row)
            for column, (title, key, kind) in enumerate(columns):
                axis = axes[row, column]
                _render_stage_panel(axis, common, arm_data, arm, position, key, kind, scales, phase_row)
                _style_arm_axis(axis, arm)
                if row == 0:
                    axis.set_title(title, fontsize=6.6, color=ARM_COLORS[arm], pad=3.0, linespacing=1.1)
                if key == "modal" and row == 0:
                    family_labels = _modal_labels({**arm_data[arm], "arm": arm})
                    axis.set_yticks(range(len(family_labels)), family_labels, fontsize=5.2)
                    axis.yaxis.tick_right()
                    axis.tick_params(
                        axis="y",
                        pad=2.0,
                        length=1.5,
                        left=False,
                        right=True,
                        labelleft=False,
                        labelright=True,
                    )
                panel_ids.append(f"{_arm_slug(arm)}-{group}-r{row}-c{column}")
            _horizontal_row_label(
                axes[row, 0],
                f"digit {int(labels[position])}\nidx {int(indices[position])}\n{'phase' if phase_row else 'amplitude'}",
                fontsize=6.0,
            )
    page_label = _digit_page_label(positions)
    figure.suptitle(
        f"Figure D · {arm} · {STAGE_GROUP_TITLES[group]} · digits {page_label.replace('_', '–')}",
        fontsize=10.0,
        color=ARM_COLORS[arm],
        y=0.95,
    )
    figure.text(
        0.5,
        0.04,
        f"{ARM_TOTAL_MODES[arm]} modes · arm-specific display group · shared p99.5 amplitude scales · raw phase [−π,π] with display-only mask. MMF-M6 synthetic families are explicitly labeled.",
        ha="center",
        va="center",
        fontsize=5.4,
        wrap=True,
    )
    figure.text(
        0.5,
        0.875,
        "same captured shared frontend; no additional optical replay",
        ha="center",
        va="center",
        fontsize=6.0,
        color=ARM_COLORS[arm],
        bbox={"facecolor": ARM_COLORS[arm], "alpha": 0.10, "edgecolor": ARM_COLORS[arm], "pad": 2.5},
    )
    row_groups = [panel_ids[row * len(columns) : (row + 1) * len(columns)] for row in range(rows)]
    column_groups = [[panel_ids[row * len(columns) + column] for row in range(rows)] for column in range(len(columns))]
    return _save_figure_bundle(
        figure,
        output_dir / _figure_name(arm, group, positions),
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _render_visualization_suite(
    common: Mapping[str, torch.Tensor],
    arm_data: Mapping[str, Mapping[str, Any]],
    target: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    scales: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    for folder in ARTIFACT_FOLDERS.values():
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    for arm in ARM_ORDER:
        for group in STAGE_GROUPS:
            (output_dir / ARTIFACT_FOLDERS["fiber_information"] / arm / group).mkdir(
                parents=True,
                exist_ok=True,
            )
    saved: list[Path] = []
    saved += _render_clear_inputs(images, labels, indices, output_dir)
    for positions in DIGIT_PAGE_RANGES:
        saved += _render_reconstruction_overview(
            target,
            labels,
            indices,
            arm_data,
            positions,
            output_dir,
        )
    for arm in ARM_ORDER:
        for group in STAGE_GROUPS:
            for positions in DIGIT_PAGE_RANGES:
                saved += _render_arm_stage_group(
                    common,
                    arm_data,
                    arm,
                    labels,
                    indices,
                    positions,
                    scales,
                    group,
                    output_dir,
                )
    return saved


def _three_photo_row_records(
    labels: torch.Tensor,
    indices: Sequence[int] | torch.Tensor,
    positions: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in positions:
        position_int = int(position)
        columns = {
            "standard_input": {
                "position": position_int,
                "source": "images_28x28",
            }
        }
        columns.update(
            {
                arm: {
                    "position": position_int,
                    "source": "arm_data[arm][mpcdar_rank16_float32]",
                }
                for arm in ARM_ORDER
            }
        )
        rows.append(
            {
                "position": position_int,
                "digit": int(labels[position_int]),
                "dataset_index": int(indices[position_int]),
                "columns": columns,
            }
        )
    return rows


def _three_photo_figure_base(
    output_dir: Path,
    page_number: int,
    positions: Sequence[int],
) -> Path:
    return output_dir / (
        f"{THREE_PHOTO_FIGURE_PREFIX}_{page_number:02d}_digits_"
        f"{_digit_page_label(positions)}"
    )


def _three_photo_figure_manifest() -> dict[str, Any]:
    titles = {
        "standard_input": "Standard digit input",
        "SMF": "SMF output",
        "FMF": "FMF output",
        "MMF-M6": "MMF-M6 output",
    }
    pages = []
    for page_number, positions in enumerate(THREE_PHOTO_PAGE_RANGES, start=1):
        base_name = _three_photo_figure_base(Path(), page_number, positions).name
        pages.append(
            {
                "page": page_number,
                "digits": list(positions),
                "base": base_name,
                "files": [
                    f"{base_name}{suffix}"
                    for suffix in THREE_PHOTO_FIGURE_EXPORT_SUFFIXES
                ]
                + [
                    f"{base_name}.alignment.json",
                    f"{base_name}.alignment.svg",
                ],
                "columns": [
                    {
                        "key": key,
                        "title": titles[key],
                        "color": "#555555" if key == "standard_input" else ARM_COLORS[key],
                    }
                    for key in THREE_PHOTO_COLUMN_KEYS
                ],
            }
        )
    return {
        "pages": pages,
        "page_count": len(pages),
        "column_order": list(THREE_PHOTO_COLUMN_KEYS),
        "input_field": "images_28x28",
        "output_field": THREE_PHOTO_OUTPUT_KEY,
    }


def _render_three_photo_page(
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    arm_data: Mapping[str, Mapping[str, Any]],
    positions: Sequence[int],
    page_number: int,
    output_dir: Path,
) -> list[Path]:
    columns = (
        ("Standard digit input", "standard_input"),
        ("SMF output", "SMF"),
        ("FMF output", "FMF"),
        ("MMF-M6 output", "MMF-M6"),
    )
    figure, axes = plt.subplots(
        len(positions),
        len(columns),
        figsize=(THREE_PHOTO_PAGE_WIDTH_INCHES, 4.8),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.16,
        right=0.99,
        bottom=0.13,
        top=0.83,
        wspace=0.08,
        hspace=0.24,
    )
    panel_ids: list[str] = []
    row_records = _three_photo_row_records(labels, indices, positions)
    for row, record in enumerate(row_records):
        position = record["position"]
        for column, (title, key) in enumerate(columns):
            axis = axes[row, column]
            if key == "standard_input":
                array = images[position, 0].detach().cpu().numpy()
                color = "#555555"
            else:
                array = arm_data[key]["mpcdar"][position, 0].detach().cpu().numpy()
                color = ARM_COLORS[key]
            axis.imshow(
                np.clip(array, 0.0, 1.0),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect=THREE_PHOTO_SPATIAL_ASPECT,
                origin=IMAGE_ORIGIN,
            )
            _finish_image_axis(axis)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(1.0)
            if row == 0:
                axis.set_title(title, fontsize=8.0, color=color, pad=4.0)
            panel_ids.append(f"page-{page_number:02d}-r{row}-c{column}")
        _horizontal_row_label(
            axes[row, 0],
            f"digit {record['digit']}",
            fontsize=8.0,
            x=-0.08,
        )
    page_label = _digit_page_label(positions).replace("_", "–")
    figure.suptitle(
        f"VIS2 · same digit input across three fiber arms · {page_label}",
        fontsize=11.0,
        y=0.965,
    )
    figure.text(
        0.5,
        0.895,
        "Each row is one identical MNIST input; only the fiber arm changes.",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#333333",
    )
    figure.text(
        0.5,
        0.035,
        "Output = frozen MPCDAR R16 F32 reconstruction; SMF blue · FMF green · MMF-M6 orange. Qualitative matched-input display only.",
        ha="center",
        va="bottom",
        fontsize=5.4,
        color="#333333",
    )
    row_groups = [
        panel_ids[row * len(columns) : (row + 1) * len(columns)]
        for row in range(len(positions))
    ]
    column_groups = [
        [panel_ids[row * len(columns) + column] for row in range(len(positions))]
        for column in range(len(columns))
    ]
    return _save_figure_bundle(
        figure,
        _three_photo_figure_base(output_dir, page_number, positions),
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _render_three_photo_suite(
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    arm_data: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    return [
        path
        for page_number, positions in enumerate(THREE_PHOTO_PAGE_RANGES, start=1)
        for path in _render_three_photo_page(
            images,
            labels,
            indices,
            arm_data,
            positions,
            page_number,
            output_dir,
        )
    ]


def _process_flow_row_records(
    labels: torch.Tensor,
    indices: Sequence[int] | torch.Tensor,
    positions: Sequence[int],
    arm: str,
) -> list[dict[str, Any]]:
    return [
        {
            "arm": arm,
            "position": int(position),
            "digit": int(labels[int(position)]),
            "dataset_index": int(indices[int(position)]),
            "stage_position": {
                key: int(position) for key in PROCESS_FLOW_STAGE_KEYS
            },
        }
        for position in positions
    ]


def _process_flow_render_row_records(
    labels: torch.Tensor,
    indices: Sequence[int] | torch.Tensor,
    positions: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in positions:
        for arm in PROCESS_FLOW_ARM_LANES:
            record = _process_flow_row_records(
                labels,
                indices,
                (position,),
                arm,
            )[0]
            for phase_row, phase in ((False, "amplitude"), (True, "phase")):
                rows.append(
                    {
                        **record,
                        "phase": phase,
                        "phase_row": phase_row,
                    }
                )
    return rows


def _process_flow_modal_display_labels(
    arm_data: Mapping[str, Any],
    arm: str,
) -> list[str]:
    labels = _modal_labels({**arm_data, "arm": arm})
    return [label.replace(" [synthetic]", "*") for label in labels]


def _process_flow_figure_base(
    output_dir: Path,
    page_number: int,
    positions: Sequence[int],
) -> Path:
    return output_dir / (
        f"{PROCESS_FLOW_FIGURE_PREFIX}_{page_number:02d}_digits_"
        f"{_digit_page_label(positions)}"
    )


def _process_flow_figure_manifest() -> dict[str, Any]:
    pages = []
    for page_number, positions in enumerate(PROCESS_FLOW_PAGE_RANGES, start=1):
        base_name = _process_flow_figure_base(Path(), page_number, positions).name
        pages.append(
            {
                "page": page_number,
                "digits": list(positions),
                "base": base_name,
                "files": [
                    f"{base_name}{suffix}"
                    for suffix in THREE_PHOTO_FIGURE_EXPORT_SUFFIXES
                ]
                + [
                    f"{base_name}.alignment.json",
                    f"{base_name}.alignment.svg",
                ],
                "arm_lanes": list(PROCESS_FLOW_ARM_LANES),
                "row_order": PROCESS_FLOW_ROW_ORDER,
                "rows_per_digit": PROCESS_FLOW_ROWS_PER_DIGIT,
                "row_sequence": [
                    {
                        "digit": int(position),
                        "position": int(position),
                        "arm_rows": [
                            {
                                "arm": arm,
                                "phase_rows": ["amplitude", "phase"],
                            }
                            for arm in PROCESS_FLOW_ARM_LANES
                        ],
                    }
                    for position in positions
                ],
                "columns": [
                    {
                        "key": key,
                        "title": title,
                        "kind": kind,
                        "shared_frontend": key in COMMON_STAGE_ORDER,
                    }
                    for title, key, kind in PROCESS_FLOW_STAGE_SPECS
                ],
            }
        )
    return {
        "pages": pages,
        "page_count": len(pages),
        "arm_lanes": list(PROCESS_FLOW_ARM_LANES),
        "stage_order": list(PROCESS_FLOW_STAGE_KEYS),
        "input_field": "images_28x28",
        "output_field": PROCESS_FLOW_OUTPUT_FIELD,
        "row_order": PROCESS_FLOW_ROW_ORDER,
        "rows_per_digit": PROCESS_FLOW_ROWS_PER_DIGIT,
        "row_structure": PROCESS_FLOW_ROW_STRUCTURE,
    }


def _render_process_flow_page(
    common: Mapping[str, torch.Tensor],
    arm_data: Mapping[str, Mapping[str, Any]],
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    positions: Sequence[int],
    page_number: int,
    scales: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    columns = PROCESS_FLOW_STAGE_SPECS
    render_rows = _process_flow_render_row_records(labels, indices, positions)
    total_rows = len(render_rows)
    figure_height = 1.9 + total_rows * 0.53
    figure, axes = plt.subplots(
        total_rows,
        len(columns),
        figsize=(PROCESS_FLOW_PAGE_WIDTH_INCHES, figure_height),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.065,
        top=0.89,
        wspace=0.025,
        hspace=0.28,
    )
    panel_ids: list[str] = []
    row_groups: list[list[str]] = []
    column_groups: list[list[str]] = [[] for _ in columns]
    for row, record in enumerate(render_rows):
        arm = record["arm"]
        phase_row = bool(record["phase_row"])
        row_ids: list[str] = []
        for column, (title, key, kind) in enumerate(columns):
            axis = axes[row, column]
            if key == "input":
                common_value = dict(common)
                common_value["input"] = images
            else:
                common_value = common
            _render_stage_panel(
                axis,
                common_value,
                arm_data,
                arm,
                record["position"],
                key,
                kind,
                scales,
                phase_row,
            )
            _style_arm_axis(axis, arm)
            for spine in axis.spines.values():
                spine.set_linewidth(0.65)
            if (
                key == "modal"
                and not phase_row
                and record["position"] == positions[0]
            ):
                family_labels = _process_flow_modal_display_labels(
                    arm_data[arm],
                    arm,
                )
                axis.yaxis.tick_right()
                axis.set_yticks(
                    range(len(family_labels)),
                    family_labels,
                    fontsize=5.0,
                )
                axis.tick_params(
                    axis="y",
                    pad=2.0,
                    length=1.2,
                    left=False,
                    right=True,
                    labelleft=False,
                    labelright=True,
                )
                axis.spines["right"].set_visible(False)
            panel_id = (
                f"{_arm_slug(arm)}-digit{record['digit']}-"
                f"{'phase' if phase_row else 'amplitude'}-{key}"
            )
            panel_ids.append(panel_id)
            row_ids.append(panel_id)
            column_groups[column].append(panel_id)
        row_groups.append(row_ids)
        row_label = (
            f"{arm} · digit {record['digit']}\n"
            f"{'Phase' if phase_row else 'Amplitude'}"
        )
        _horizontal_row_label(
            axes[row, 0],
            row_label,
            fontsize=5.8,
            color=ARM_COLORS[arm],
            x=-0.075,
        )

    for column, (title, _, _) in enumerate(columns):
        axis_position = axes[0, column].get_position()
        figure.text(
            (axis_position.x0 + axis_position.x1) / 2.0,
            axis_position.y1 + 0.012,
            title,
            ha="center",
            va="bottom",
            fontsize=5.4,
            color="#333333",
            linespacing=0.9,
        )

    page_label = _digit_page_label(positions).replace("_", "–")
    figure.suptitle(
        f"VIS2 process flow · matched digits {page_label} · SMF / FMF / MMF-M6",
        fontsize=11.0,
        y=0.965,
    )
    figure.text(
        0.5,
        0.925,
        "Reference-style amplitude/phase flow; each colored lane reuses the same input and shared frontend before its fiber branch.",
        ha="center",
        va="center",
        fontsize=6.3,
        color="#333333",
    )
    figure.text(
        0.5,
        0.020,
        "Shared frontend: input → SLM → encoder → pre-OLS. Branch: modal/fiber → decoder → CMOS → frozen MPCDAR R16 F32; phase N/A is explicit for real-valued input/command/intensity/reconstruction. In Modal, * marks synthetic MMF-M6 families.",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#333333",
    )
    return _save_figure_bundle(
        figure,
        _process_flow_figure_base(output_dir, page_number, positions),
        axes=axes.ravel().tolist(),
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        raster_dpi=RASTER_DPI,
    )


def _render_process_flow_suite(
    common: Mapping[str, torch.Tensor],
    arm_data: Mapping[str, Mapping[str, Any]],
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    scales: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    return [
        path
        for page_number, positions in enumerate(PROCESS_FLOW_PAGE_RANGES, start=1)
        for path in _render_process_flow_page(
            common,
            arm_data,
            images,
            labels,
            indices,
            positions,
            page_number,
            scales,
            output_dir,
        )
    ]


def _stage_payload(
    common: Mapping[str, torch.Tensor],
    arm_data: Mapping[str, Mapping[str, Any]],
    target: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    scales: Mapping[str, Any],
    clear_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        data = arm_data[arm]
        arms[arm] = {
            "flat_coefficients": _tensor_cpu(data["flat_coefficients"]),
            "mode_family_coefficients": _tensor_cpu(data["family_coefficients"]),
            "modal_power": _tensor_cpu(data["modal_power"]),
            "post_fiber": _tensor_cpu(data["post_fiber"]),
            "decoder_layer_1": _tensor_cpu(data["decoder_layer_1"]),
            "decoder_layer_2": _tensor_cpu(data["decoder_layer_2"]),
            "cmos_field": _tensor_cpu(data["cmos_field"]),
            "cmos_intensity": _tensor_cpu(data["cmos_intensity"]),
            "mpcdar_rank16_float32": _tensor_cpu(data["mpcdar"]),
            "z_256": _tensor_cpu(data["z"]),
            "mode_family_order": list(data["family_order"]),
            "flat_mode_family_names": list(data["flat_mode_family_names"]),
            "stage_metadata": data["stage_metadata"],
        }
    return {
        "schema": PANEL_SCHEMA,
        "task_id": TASK_ID,
        "classification": CLASSIFICATION,
        "clear_panel_manifest_sha256": file_sha256(_repo_path(clear_manifest["_path"])),
        "images_28x28": _tensor_cpu(images),
        "labels": _tensor_cpu(labels),
        "dataset_indices": _tensor_cpu(indices),
        "target_200x200": _tensor_cpu(target),
        "common_stages": {key: _tensor_cpu(value) for key, value in common.items()},
        "arms": arms,
        "display_scales": dict(scales),
        "full_mode_banks_stored": False,
        "mode_bank_note": "Only captured coefficients/family powers are stored; no complete 600-mode bank is exported.",
    }


def _read_json_mapping(path: Path, error_code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(error_code) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(error_code)
    return dict(payload)


def _normalize_manifest_relative_path(value: Any, error_code: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(error_code)
    relative = Path(value)
    if relative.is_absolute():
        raise RuntimeError(error_code)
    normalized = relative.as_posix()
    if normalized in {"", "."} or normalized == ".." or normalized.startswith("../"):
        raise RuntimeError(error_code)
    return normalized


def _manifest_artifact_path(
    source_dir: Path,
    relative_path: Any,
    error_code: str,
) -> Path:
    normalized = _normalize_manifest_relative_path(relative_path, error_code)
    root = source_dir.resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(error_code) from exc
    if not candidate.is_file():
        raise RuntimeError(error_code)
    return candidate


def _manifest_relative_path(source_dir: Path, path: Path, error_code: str) -> str:
    root = source_dir.resolve()
    candidate = path.resolve()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(error_code) from exc


def _resolve_panel_source_paths(source_dir: Path) -> dict[str, Path]:
    """Resolve the panel boundary for both flat and organized VIS2 outputs."""

    if not source_dir.is_dir():
        raise RuntimeError(f"BLOCKED_PANEL_SOURCE_DIRECTORY_MISSING: {source_dir}")
    visualization_path = source_dir / "visualization_manifest.json"
    visualization = _read_json_mapping(
        visualization_path,
        "BLOCKED_PANEL_SOURCE_VISUALIZATION_UNREADABLE",
    )
    comparison_binding = visualization.get("comparison_manifest")
    if not isinstance(comparison_binding, Mapping):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_VISUALIZATION_INVALID")
    comparison_path = _manifest_artifact_path(
        source_dir,
        comparison_binding.get("path"),
        "BLOCKED_PANEL_SOURCE_COMPARISON_MISSING",
    )
    comparison = _read_json_mapping(
        comparison_path,
        "BLOCKED_PANEL_SOURCE_COMPARISON_UNREADABLE",
    )
    clear_panel = comparison.get("clear_panel")
    if not isinstance(clear_panel, Mapping):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_CLEAR_MANIFEST_INVALID")
    clear_manifest_path = _manifest_artifact_path(
        source_dir,
        clear_panel.get("manifest"),
        "BLOCKED_PANEL_SOURCE_CLEAR_MANIFEST_MISSING",
    )
    review_path = _manifest_artifact_path(
        source_dir,
        clear_panel.get("review"),
        "BLOCKED_PANEL_SOURCE_REVIEW_MISSING",
    )
    output_entries = visualization.get("output_files")
    if not isinstance(output_entries, list):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_OUTPUT_MANIFEST_INVALID")
    panel_paths = [
        _manifest_artifact_path(source_dir, item.get("path"), "BLOCKED_PANEL_SOURCE_PANEL_MISSING")
        for item in output_entries
        if isinstance(item, Mapping)
        and isinstance(item.get("path"), str)
        and Path(item["path"]).name.lower() == "panel.pt"
    ]
    if len(panel_paths) != 1:
        raise RuntimeError("BLOCKED_PANEL_SOURCE_PANEL_MISSING")
    return {
        "visualization": visualization_path,
        "comparison": comparison_path,
        "clear_manifest": clear_manifest_path,
        "review": review_path,
        "panel": panel_paths[0],
    }


def _resolve_named_output_artifact(
    source_dir: Path,
    visualization_path: Path,
    filename: str,
) -> Path:
    visualization = _read_json_mapping(
        visualization_path,
        "BLOCKED_PANEL_SOURCE_VISUALIZATION_UNREADABLE",
    )
    output_entries = visualization.get("output_files")
    if not isinstance(output_entries, list):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_OUTPUT_MANIFEST_INVALID")
    candidates = [
        _manifest_artifact_path(source_dir, item.get("path"), "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING")
        for item in output_entries
        if isinstance(item, Mapping)
        and isinstance(item.get("path"), str)
        and Path(item["path"]).name == filename
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING: {filename}")
    return candidates[0]


def _validate_panel_source(source_dir: Path) -> dict[str, Any]:
    paths = _resolve_panel_source_paths(source_dir)
    panel_path = paths["panel"]
    clear_manifest_path = paths["clear_manifest"]
    review_path = paths["review"]
    comparison_path = paths["comparison"]
    visualization_path = paths["visualization"]
    source_relative = {
        name: _manifest_relative_path(source_dir, path, "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING")
        for name, path in paths.items()
    }

    clear_manifest = _load_clear_manifest(source_dir, manifest_path=clear_manifest_path)
    review = _read_json_mapping(review_path, "BLOCKED_PANEL_SOURCE_REVIEW_UNREADABLE")
    if (
        review.get("schema") != "lightgen_scaled_chen_fiber_vis2_clear_panel_review.v1"
        or review.get("task_id") != TASK_ID
        or review.get("decision") != "APPROVED"
        or review.get("selection_method") != "input_only"
        or review.get("downstream_outputs_used_for_selection") is not False
        or review.get("selected_before_fiber_evaluation") is not True
        or review.get("clear_panel_manifest_sha256") != file_sha256(clear_manifest_path)
    ):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_REVIEW_INVALID")

    comparison = _read_json_mapping(comparison_path, "BLOCKED_PANEL_SOURCE_COMPARISON_UNREADABLE")
    if (
        comparison.get("schema")
        not in {
            "lightgen_scaled_chen_fiber_vis2_comparison_manifest.v1",
            "lightgen_scaled_chen_fiber_vis2_comparison_manifest.v2",
        }
        or comparison.get("classification") != CLASSIFICATION
        or comparison.get("task_id") != TASK_ID
        or comparison.get("status") != "completed"
        or comparison.get("arm_order") != list(ARM_ORDER)
        or not isinstance(comparison.get("arm_contracts"), Mapping)
    ):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_COMPARISON_INVALID")
    comparison_clear_panel = comparison.get("clear_panel")
    if (
        not isinstance(comparison_clear_panel, Mapping)
        or _normalize_manifest_relative_path(
            comparison_clear_panel.get("manifest"),
            "BLOCKED_PANEL_SOURCE_CLEAR_MANIFEST_HASH_INVALID",
        )
        != source_relative["clear_manifest"]
        or _normalize_manifest_relative_path(
            comparison_clear_panel.get("review"),
            "BLOCKED_PANEL_SOURCE_CLEAR_MANIFEST_HASH_INVALID",
        )
        != source_relative["review"]
        or comparison_clear_panel.get("manifest_sha256") != file_sha256(clear_manifest_path)
        or comparison_clear_panel.get("review_sha256") != file_sha256(review_path)
        or comparison_clear_panel.get("review_decision") != "APPROVED"
        or comparison_clear_panel.get("selection_method") != "input_only"
        or comparison_clear_panel.get("selected_before_fiber_evaluation") is not True
        or comparison_clear_panel.get("downstream_outputs_used_for_selection") is not False
    ):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_CLEAR_MANIFEST_HASH_INVALID")

    visualization = _read_json_mapping(visualization_path, "BLOCKED_PANEL_SOURCE_VISUALIZATION_UNREADABLE")
    comparison_binding = visualization.get("comparison_manifest")
    if (
        visualization.get("schema")
        not in {
            "lightgen_scaled_chen_fiber_vis2_visualization_manifest.v1",
            "lightgen_scaled_chen_fiber_vis2_visualization_manifest.v2",
        }
        or visualization.get("classification") != CLASSIFICATION
        or visualization.get("task_id") != TASK_ID
        or visualization.get("status") != "completed"
        or not isinstance(comparison_binding, Mapping)
        or _normalize_manifest_relative_path(
            comparison_binding.get("path"),
            "BLOCKED_PANEL_SOURCE_VISUALIZATION_INVALID",
        )
        != source_relative["comparison"]
        or comparison_binding.get("sha256") != file_sha256(comparison_path)
    ):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_VISUALIZATION_INVALID")
    vis1_authority = visualization.get("vis1_authority")
    if (
        not isinstance(vis1_authority, Mapping)
        or vis1_authority.get("status") != "unchanged"
        or vis1_authority.get("not_overwritten") is not True
        or vis1_authority.get("not_recomputed") is not True
        or vis1_authority.get("not_modified") is not True
    ):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_VIS1_AUTHORITY_INVALID")
    output_entries = visualization.get("output_files")
    panel_entry = next(
        (
            item
            for item in output_entries
            if isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and _normalize_manifest_relative_path(
                item["path"], "BLOCKED_PANEL_SOURCE_PANEL_HASH_INVALID"
            )
            == source_relative["panel"]
        ),
        None,
    ) if isinstance(output_entries, list) else None
    panel_sha256 = file_sha256(panel_path)
    if not isinstance(panel_entry, Mapping) or panel_entry.get("sha256") != panel_sha256:
        raise RuntimeError("BLOCKED_PANEL_SOURCE_PANEL_HASH_INVALID")

    try:
        payload = torch.load(panel_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise RuntimeError("BLOCKED_PANEL_SOURCE_PAYLOAD_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_PAYLOAD_INVALID")
    if (
        payload.get("schema") != PANEL_SCHEMA
        or payload.get("task_id") != TASK_ID
        or payload.get("classification") != CLASSIFICATION
        or payload.get("clear_panel_manifest_sha256") != file_sha256(clear_manifest_path)
        or payload.get("full_mode_banks_stored") is not False
    ):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_PAYLOAD_CONTRACT_CHANGED")

    return {
        "source_dir": source_dir,
        "panel_path": panel_path,
        "clear_manifest": clear_manifest,
        "review": review,
        "comparison": comparison,
        "visualization": visualization,
        "paths": paths,
        "panel_sha256": panel_sha256,
        "payload": payload,
    }


def _load_panel_payload(source_dir: str | Path) -> dict[str, Any]:
    source = _validate_panel_source(_repo_path(source_dir))
    payload = source["payload"]
    common_payload = payload["common_stages"]
    if not isinstance(common_payload, Mapping):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_COMMON_STAGES_INVALID")
    common = {key: _tensor_cpu(common_payload[key]) for key in COMMON_STAGE_ORDER}
    images = _tensor_cpu(payload["images_28x28"])
    labels = _tensor_cpu(payload["labels"])
    indices = _tensor_cpu(payload["dataset_indices"])
    target = _tensor_cpu(payload["target_200x200"])
    if (
        tuple(images.shape) != (10, 1, 28, 28)
        or tuple(labels.shape) != (10,)
        or tuple(indices.shape) != (10,)
        or tuple(target.shape) != (10, 1, 200, 200)
        or labels.tolist() != list(range(10))
    ):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_SAMPLE_SHAPES_INVALID")

    comparison = source["comparison"]
    source_arms = payload["arms"]
    arm_contracts = comparison["arm_contracts"]
    arm_metrics = comparison["arm_metrics"]
    if not isinstance(source_arms, Mapping) or not isinstance(arm_contracts, Mapping) or not isinstance(arm_metrics, Mapping):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_ARMS_INVALID")
    arm_data: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        data = source_arms[arm]
        if not isinstance(data, Mapping):
            raise RuntimeError(f"BLOCKED_PANEL_SOURCE_ARM_INVALID: {arm}")
        arm_data[arm] = {
            "arm": arm,
            "contract": copy.deepcopy(arm_contracts[arm]),
            "metrics": copy.deepcopy(arm_metrics[arm]),
            "flat_coefficients": _tensor_cpu(data["flat_coefficients"]),
            "family_coefficients": _tensor_cpu(data["mode_family_coefficients"]),
            "modal_power": _tensor_cpu(data["modal_power"]),
            "post_fiber": _tensor_cpu(data["post_fiber"]),
            "decoder_layer_1": _tensor_cpu(data["decoder_layer_1"]),
            "decoder_layer_2": _tensor_cpu(data["decoder_layer_2"]),
            "cmos_field": _tensor_cpu(data["cmos_field"]),
            "cmos_intensity": _tensor_cpu(data["cmos_intensity"]),
            "mpcdar": _tensor_cpu(data["mpcdar_rank16_float32"]),
            "z": _tensor_cpu(data["z_256"]),
            "family_order": list(data["mode_family_order"]),
            "flat_mode_family_names": list(data["flat_mode_family_names"]),
            "stage_metadata": copy.deepcopy(data["stage_metadata"]),
        }

    shared_frontend = comparison["shared_frontend"]
    resources = comparison["resources"]
    if not isinstance(shared_frontend, Mapping) or not isinstance(resources, Mapping):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_SHARED_METADATA_INVALID")
    pre_energy = torch.tensor(shared_frontend["pre_ols_power_per_sample"], dtype=torch.float32)
    if tuple(pre_energy.shape) != (10,) or not bool(torch.isfinite(pre_energy).all()):
        raise RuntimeError("BLOCKED_PANEL_SOURCE_PRE_OLS_ENERGY_INVALID")
    shared = {
        "pre_energy": pre_energy,
        "pre_ols_pointer_identity": bool(shared_frontend["same_pre_ols_tensor_object_to_all_arms"]),
        "common_stage_metadata": copy.deepcopy(shared_frontend["common_stage_metadata"]),
        "shared_pre_ols_tensor_shape": list(shared_frontend["pre_ols_shape"]),
    }
    return {
        **source,
        "common": common,
        "arm_data": arm_data,
        "images": images,
        "labels": labels,
        "indices": indices,
        "target": target,
        "scales": copy.deepcopy(payload["display_scales"]),
        "shared": shared,
        "resources": copy.deepcopy(resources),
    }


def _copy_panel_source_artifacts(source: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    source_dir = source["source_dir"]
    paths = source["paths"]
    metadata_dir = output_dir / ARTIFACT_FOLDERS["metadata_qa"]
    metadata_dir.mkdir(parents=True, exist_ok=True)
    names_and_paths = {
        "clear_panel_manifest.json": paths["clear_manifest"],
        "clear_panel_review.json": paths["review"],
        "panel.pt": paths["panel"],
    }
    for name in (
        "clear_input_candidate_contact_sheet.png",
        "clear_input_candidate_contact_sheet.pdf",
        "clear_input_candidate_contact_sheet.svg",
        "clear_input_candidate_contact_sheet.alignment.json",
        "clear_input_candidate_contact_sheet.alignment.svg",
    ):
        names_and_paths[name] = _resolve_named_output_artifact(
            source_dir,
            paths["visualization"],
            name,
        )
    copied: dict[str, Path] = {}
    for name, source_path in names_and_paths.items():
        target = metadata_dir / name
        if not source_path.is_file():
            raise RuntimeError(f"BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING: {name}")
        shutil.copy2(source_path, target)
        copied[name] = target
    return copied


def _layout_manifest_contract() -> dict[str, Any]:
    return {
        "layout_schema": VIS2_LAYOUT_SCHEMA,
        "artifact_layout_schema": ARTIFACT_LAYOUT_SCHEMA,
        "artifact_folders": dict(ARTIFACT_FOLDERS),
        "arm_colors": dict(ARM_COLORS),
        "digit_page_ranges": [list(positions) for positions in DIGIT_PAGE_RANGES],
        "stage_groups": {
            group: [
                {"title": title, "key": key, "kind": kind}
                for title, key, kind in columns
            ]
            for group, columns in STAGE_GROUPS.items()
        },
        "spatial_aspect": "equal",
        "modal_aspect": "explicit_nonspatial_matrix",
        "figure_export_suffixes": list(FIGURE_EXPORT_SUFFIXES),
        "tiff_exported": False,
        "maximum_page_width_inches": 7.2,
        "orientation_policy": copy.deepcopy(ORIENTATION_POLICY),
        "relative_paths": True,
    }


def _three_photo_layout_manifest() -> dict[str, Any]:
    return {
        "layout_schema": THREE_PHOTO_LAYOUT_SCHEMA,
        "artifact_layout_schema": THREE_PHOTO_ARTIFACT_LAYOUT_SCHEMA,
        "figure_folder": ".",
        "metadata_folder": ARTIFACT_FOLDERS["metadata_qa"],
        "arm_colors": dict(ARM_COLORS),
        "digit_page_ranges": [list(positions) for positions in THREE_PHOTO_PAGE_RANGES],
        "column_order": list(THREE_PHOTO_COLUMN_KEYS),
        "input_field": "images_28x28",
        "output_field": THREE_PHOTO_OUTPUT_KEY,
        "spatial_aspect": THREE_PHOTO_SPATIAL_ASPECT,
        "modal_aspect": "not_rendered_in_three_photo_comparison",
        "figure_export_suffixes": list(THREE_PHOTO_FIGURE_EXPORT_SUFFIXES),
        "alignment_suffixes": [".alignment.json", ".alignment.svg"],
        "tiff_exported": False,
        "page_width_inches": THREE_PHOTO_PAGE_WIDTH_INCHES,
        "maximum_page_width_inches": 7.2,
        "row_label_policy": "digit only; dataset index remains in matched_input_comparison metadata",
        "relative_paths": True,
    }


def _three_photo_comparison_manifest(
    source: Mapping[str, Any],
    output_dir: Path,
    labels: torch.Tensor,
    indices: Sequence[int],
) -> dict[str, Any]:
    comparison = copy.deepcopy(source["comparison"])
    comparison["schema"] = "lightgen_scaled_chen_fiber_vis2_comparison_manifest.v3"
    comparison["output_dir"] = str(output_dir.resolve())
    comparison["git"] = {
        "source": copy.deepcopy(source["comparison"].get("git", {})),
        "after_render": _git_state(),
    }
    comparison["figures"] = _three_photo_figure_manifest()
    comparison["layout"] = _three_photo_layout_manifest()
    metadata_dir = Path(ARTIFACT_FOLDERS["metadata_qa"])
    target_clear_manifest = output_dir / metadata_dir / "clear_panel_manifest.json"
    target_review = output_dir / metadata_dir / "clear_panel_review.json"
    comparison["artifact_paths"] = {
        "clear_manifest": (metadata_dir / "clear_panel_manifest.json").as_posix(),
        "review": (metadata_dir / "clear_panel_review.json").as_posix(),
        "candidate_contact_sheet": (
            metadata_dir / "clear_input_candidate_contact_sheet.png"
        ).as_posix(),
        "panel": (metadata_dir / "panel.pt").as_posix(),
        "comparison_manifest": (metadata_dir / "comparison_manifest.json").as_posix(),
        "visualization_manifest": "visualization_manifest.json",
    }
    comparison["clear_panel"] = {
        **comparison["clear_panel"],
        "manifest": comparison["artifact_paths"]["clear_manifest"],
        "review": comparison["artifact_paths"]["review"],
        "manifest_sha256": file_sha256(target_clear_manifest),
        "review_sha256": file_sha256(target_review),
    }
    comparison["rendering"] = {
        "mode": "panel_only_three_photo_rerender",
        "source_output_dir": str(source["source_dir"].resolve()),
        "source_panel_sha256": source["panel_sha256"],
        "source_comparison_manifest_sha256": file_sha256(source["paths"]["comparison"]),
        "source_visualization_manifest_sha256": file_sha256(source["paths"]["visualization"]),
        "render_device": "cpu",
        "cuda_replay": False,
        "optical_data_changed": False,
    }
    resources = comparison.setdefault("resources", {})
    resources["render_only_from_panel"] = True
    resources["cuda_replay"] = False
    resources["optical_data_changed"] = False
    resources["render_device"] = "cpu"
    resources["source_panel_sha256"] = source["panel_sha256"]
    comparison["source_manifest_binding"] = {
        "source_visualization_manifest": _manifest_relative_path(
            source["source_dir"],
            source["paths"]["visualization"],
            "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING",
        ),
        "source_visualization_manifest_sha256": file_sha256(source["paths"]["visualization"]),
        "source_comparison_manifest": _manifest_relative_path(
            source["source_dir"],
            source["paths"]["comparison"],
            "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING",
        ),
        "source_comparison_manifest_sha256": file_sha256(source["paths"]["comparison"]),
        "source_panel": _manifest_relative_path(
            source["source_dir"],
            source["paths"]["panel"],
            "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING",
        ),
        "source_panel_sha256": source["panel_sha256"],
    }
    comparison["matched_input_comparison"] = {
        "row_alignment": "one shared panel position across standard input and all three fiber outputs",
        "input_field": "images_28x28",
        "output_field": THREE_PHOTO_OUTPUT_KEY,
        "pages": [
            {
                "page": page_number,
                "digits": [int(labels[position]) for position in positions],
                "dataset_indices": [int(indices[position]) for position in positions],
                "positions": list(positions),
            }
            for page_number, positions in enumerate(THREE_PHOTO_PAGE_RANGES, start=1)
        ],
    }
    comparison["supersedes"] = "attempt_10"
    comparison["interpretation_boundary"] = (
        "This three-photo VIS2 artifact is a qualitative matched-input comparison only. "
        "It does not rank fibers or support physical MMF, reconstruction-manifold, "
        "generation, training, or promotion claims."
    )
    return comparison


def _three_photo_visualization_manifest(
    output_dir: Path,
    comparison: Mapping[str, Any],
    output_files: Sequence[Path],
    governance: Mapping[str, Any],
) -> dict[str, Any]:
    if any(
        path.suffix.lower() in {".tif", ".tiff"}
        for path in output_files
    ):
        raise RuntimeError("VIS2 three-photo output must not contain TIFF artifacts")
    if any(
        path.name.startswith(".") or ".tmp" in path.name
        for path in output_files
    ):
        raise RuntimeError("VIS2 three-photo output must not contain hidden or temporary files")
    figures = comparison["figures"]
    artifact_paths = comparison["artifact_paths"]
    required_artifacts = [
        artifact_paths["clear_manifest"],
        artifact_paths["candidate_contact_sheet"],
        artifact_paths["review"],
        artifact_paths["panel"],
        artifact_paths["comparison_manifest"],
        *[
            filename
            for page in figures["pages"]
            for filename in page["files"]
        ],
    ]
    missing = [
        path
        for path in required_artifacts
        if not (output_dir / path).is_file()
    ]
    if missing:
        raise RuntimeError(
            "VIS2 three-photo output is missing required artifacts: "
            + ", ".join(missing)
        )
    comparison_path = artifact_paths["comparison_manifest"]
    return {
        "schema": "lightgen_scaled_chen_fiber_vis2_visualization_manifest.v3",
        "task_id": TASK_ID,
        "status": "completed",
        "classification": CLASSIFICATION,
        "layout_schema": THREE_PHOTO_LAYOUT_SCHEMA,
        "artifact_layout_schema": THREE_PHOTO_ARTIFACT_LAYOUT_SCHEMA,
        "folder_layout": {
            "figures": ".",
            "metadata_qa": ARTIFACT_FOLDERS["metadata_qa"],
        },
        "diagnostic_only": True,
        "output_dir": str(output_dir.resolve()),
        "comparison_manifest": {
            "path": comparison_path,
            "sha256": file_sha256(output_dir / comparison_path),
        },
        "orientation_policy": copy.deepcopy(ORIENTATION_POLICY),
        "supersedes": comparison.get("supersedes"),
        "required_artifacts": required_artifacts,
        "output_files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output_files)
        ],
        "source_panel_sha256": comparison["rendering"]["source_panel_sha256"],
        "source_manifest_binding": copy.deepcopy(
            comparison["source_manifest_binding"]
        ),
        "governance": dict(governance),
        "vis1_authority": {
            "status": "unchanged",
            "authority_output": str(VIS1_OUTPUT),
            "not_overwritten": True,
            "not_recomputed": True,
            "not_modified": True,
        },
        "unique_next_action": UNIQUE_NEXT_ACTION,
        "render_only_from_panel": True,
        "cuda_replay": False,
        "optical_data_changed": False,
        "manifest_self_hash": "not embedded to avoid self-reference",
    }


def _process_flow_layout_manifest() -> dict[str, Any]:
    return {
        "layout_schema": PROCESS_FLOW_LAYOUT_SCHEMA,
        "artifact_layout_schema": PROCESS_FLOW_ARTIFACT_LAYOUT_SCHEMA,
        "figure_folder": ".",
        "metadata_folder": ARTIFACT_FOLDERS["metadata_qa"],
        "arm_lanes": list(PROCESS_FLOW_ARM_LANES),
        "arm_colors": dict(ARM_COLORS),
        "digit_page_ranges": [
            list(positions) for positions in PROCESS_FLOW_PAGE_RANGES
        ],
        "stage_order": list(PROCESS_FLOW_STAGE_KEYS),
        "stage_columns": [
            {
                "key": key,
                "title": title,
                "kind": kind,
                "shared_frontend": key in COMMON_STAGE_ORDER,
            }
            for title, key, kind in PROCESS_FLOW_STAGE_SPECS
        ],
        "input_field": "images_28x28",
        "output_field": PROCESS_FLOW_OUTPUT_FIELD,
        "row_order": PROCESS_FLOW_ROW_ORDER,
        "rows_per_digit": PROCESS_FLOW_ROWS_PER_DIGIT,
        "row_structure": PROCESS_FLOW_ROW_STRUCTURE,
        "spatial_aspect": PROCESS_FLOW_SPATIAL_ASPECT,
        "modal_aspect": "explicit_nonspatial_matrix",
        "figure_export_suffixes": list(THREE_PHOTO_FIGURE_EXPORT_SUFFIXES),
        "alignment_suffixes": [".alignment.json", ".alignment.svg"],
        "tiff_exported": False,
        "page_width_inches": PROCESS_FLOW_PAGE_WIDTH_INCHES,
        "maximum_page_width_inches": PROCESS_FLOW_PAGE_WIDTH_INCHES,
        "minimum_text_size_pt": 5.0,
        "row_label_policy": "digit and amplitude/phase only; dataset index remains in matched_input_comparison metadata",
        "synthetic_label_policy": "MMF-M6 synthetic family labels remain explicit in the modal matrix metadata/display",
        "relative_paths": True,
    }


def _process_flow_comparison_manifest(
    source: Mapping[str, Any],
    output_dir: Path,
    labels: torch.Tensor,
    indices: Sequence[int],
) -> dict[str, Any]:
    comparison = copy.deepcopy(source["comparison"])
    comparison["schema"] = "lightgen_scaled_chen_fiber_vis2_comparison_manifest.v5"
    comparison["output_dir"] = str(output_dir.resolve())
    comparison["git"] = {
        "source": copy.deepcopy(source["comparison"].get("git", {})),
        "after_render": _git_state(),
    }
    comparison["layout_schema"] = PROCESS_FLOW_LAYOUT_SCHEMA
    comparison["figures"] = _process_flow_figure_manifest()
    comparison["layout"] = _process_flow_layout_manifest()
    metadata_dir = Path(ARTIFACT_FOLDERS["metadata_qa"])
    target_clear_manifest = output_dir / metadata_dir / "clear_panel_manifest.json"
    target_review = output_dir / metadata_dir / "clear_panel_review.json"
    comparison["artifact_paths"] = {
        "clear_manifest": (metadata_dir / "clear_panel_manifest.json").as_posix(),
        "review": (metadata_dir / "clear_panel_review.json").as_posix(),
        "candidate_contact_sheet": (
            metadata_dir / "clear_input_candidate_contact_sheet.png"
        ).as_posix(),
        "panel": (metadata_dir / "panel.pt").as_posix(),
        "comparison_manifest": (metadata_dir / "comparison_manifest.json").as_posix(),
        "visualization_manifest": "visualization_manifest.json",
    }
    comparison["clear_panel"] = {
        **comparison["clear_panel"],
        "manifest": comparison["artifact_paths"]["clear_manifest"],
        "review": comparison["artifact_paths"]["review"],
        "manifest_sha256": file_sha256(target_clear_manifest),
        "review_sha256": file_sha256(target_review),
    }
    source_comparison_sha256 = file_sha256(source["paths"]["comparison"])
    source_visualization_sha256 = file_sha256(source["paths"]["visualization"])
    comparison["rendering"] = {
        "mode": "panel_only_interleaved_process_flow_rerender",
        "source_output_dir": str(source["source_dir"].resolve()),
        "source_panel_sha256": source["panel_sha256"],
        "source_comparison_manifest_sha256": source_comparison_sha256,
        "source_visualization_manifest_sha256": source_visualization_sha256,
        "render_device": "cpu",
        "cuda_replay": False,
        "optical_data_changed": False,
    }
    resources = comparison.setdefault("resources", {})
    resources["device"] = "cpu"
    resources["render_device"] = "cpu"
    resources["render_only_from_panel"] = True
    resources["cuda_replay"] = False
    resources["optical_data_changed"] = False
    resources["source_panel_sha256"] = source["panel_sha256"]
    comparison["source_manifest_binding"] = {
        "source_visualization_manifest": _manifest_relative_path(
            source["source_dir"],
            source["paths"]["visualization"],
            "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING",
        ),
        "source_visualization_manifest_sha256": source_visualization_sha256,
        "source_comparison_manifest": _manifest_relative_path(
            source["source_dir"],
            source["paths"]["comparison"],
            "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING",
        ),
        "source_comparison_manifest_sha256": source_comparison_sha256,
        "source_panel": _manifest_relative_path(
            source["source_dir"],
            source["paths"]["panel"],
            "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING",
        ),
        "source_panel_sha256": source["panel_sha256"],
    }
    comparison["matched_input_comparison"] = {
        "row_alignment": "for each digit, SMF, FMF, and MMF-M6 amplitude/phase pairs are adjacent before the next digit begins",
        "row_order": PROCESS_FLOW_ROW_ORDER,
        "row_structure": PROCESS_FLOW_ROW_STRUCTURE,
        "input_field": "images_28x28",
        "output_field": PROCESS_FLOW_OUTPUT_FIELD,
        "process_stage_order": list(PROCESS_FLOW_STAGE_KEYS),
        "arm_lanes": list(PROCESS_FLOW_ARM_LANES),
        "pages": [
            {
                "page": page_number,
                "digits": [int(labels[position]) for position in positions],
                "dataset_indices": [int(indices[position]) for position in positions],
                "positions": list(positions),
                "row_records": [
                    {
                        "digit": int(row["digit"]),
                        "arm": row["arm"],
                        "phase": row["phase"],
                        "position": int(row["position"]),
                        "dataset_index": int(row["dataset_index"]),
                    }
                    for row in _process_flow_render_row_records(
                        labels,
                        indices,
                        positions,
                    )
                ],
            }
            for page_number, positions in enumerate(
                PROCESS_FLOW_PAGE_RANGES, start=1
            )
        ],
    }
    comparison["supersedes"] = "attempt_14"
    comparison["interpretation_boundary"] = (
        "This process-flow VIS2 artifact is a qualitative matched-input display only. "
        "The three colored arm lanes expose stage-wise amplitude/phase differences "
        "for the same digits; they do not rank fibers or support physical MMF, "
        "reconstruction-manifold, generation, training, or promotion claims."
    )
    return comparison


def _process_flow_visualization_manifest(
    output_dir: Path,
    comparison: Mapping[str, Any],
    output_files: Sequence[Path],
    governance: Mapping[str, Any],
) -> dict[str, Any]:
    if any(path.suffix.lower() in {".tif", ".tiff"} for path in output_files):
        raise RuntimeError("VIS2 process-flow output must not contain TIFF artifacts")
    if any(
        path.name.startswith(".") or ".tmp" in path.name
        for path in output_files
    ):
        raise RuntimeError(
            "VIS2 process-flow output must not contain hidden or temporary files"
        )
    figures = comparison["figures"]
    artifact_paths = comparison["artifact_paths"]
    required_artifacts = [
        artifact_paths["clear_manifest"],
        artifact_paths["candidate_contact_sheet"],
        artifact_paths["review"],
        artifact_paths["panel"],
        artifact_paths["comparison_manifest"],
        *[
            filename
            for page in figures["pages"]
            for filename in page["files"]
        ],
    ]
    missing = [
        path
        for path in required_artifacts
        if not (output_dir / path).is_file()
    ]
    if missing:
        raise RuntimeError(
            "VIS2 process-flow output is missing required artifacts: "
            + ", ".join(missing)
        )
    comparison_path = artifact_paths["comparison_manifest"]
    return {
        "schema": "lightgen_scaled_chen_fiber_vis2_visualization_manifest.v5",
        "task_id": TASK_ID,
        "status": "completed",
        "classification": CLASSIFICATION,
        "layout_schema": PROCESS_FLOW_LAYOUT_SCHEMA,
        "artifact_layout_schema": PROCESS_FLOW_ARTIFACT_LAYOUT_SCHEMA,
        "folder_layout": {
            "figures": ".",
            "metadata_qa": ARTIFACT_FOLDERS["metadata_qa"],
        },
        "diagnostic_only": True,
        "output_dir": str(output_dir.resolve()),
        "comparison_manifest": {
            "path": comparison_path,
            "sha256": file_sha256(output_dir / Path(comparison_path)),
        },
        "orientation_policy": copy.deepcopy(ORIENTATION_POLICY),
        "supersedes": comparison.get("supersedes"),
        "required_artifacts": required_artifacts,
        "output_files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output_files)
        ],
        "source_panel_sha256": comparison["rendering"]["source_panel_sha256"],
        "source_manifest_binding": copy.deepcopy(
            comparison["source_manifest_binding"]
        ),
        "governance": dict(governance),
        "vis1_authority": {
            "status": "unchanged",
            "authority_output": str(VIS1_OUTPUT),
            "not_overwritten": True,
            "not_recomputed": True,
            "not_modified": True,
        },
        "unique_next_action": UNIQUE_NEXT_ACTION,
        "render_only_from_panel": True,
        "cuda_replay": False,
        "optical_data_changed": False,
        "manifest_self_hash": "not embedded to avoid self-reference",
    }


def _layout_comparison_manifest(source: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    comparison = copy.deepcopy(source["comparison"])
    comparison["schema"] = "lightgen_scaled_chen_fiber_vis2_comparison_manifest.v2"
    comparison["output_dir"] = str(output_dir.resolve())
    comparison["git"] = {
        "source": copy.deepcopy(source["comparison"].get("git", {})),
        "after_render": _git_state(),
    }
    comparison["figures"] = _figure_manifest_names()
    comparison["layout"] = _layout_manifest_contract()
    metadata_dir = Path(ARTIFACT_FOLDERS["metadata_qa"])
    target_clear_manifest = output_dir / metadata_dir / "clear_panel_manifest.json"
    target_review = output_dir / metadata_dir / "clear_panel_review.json"
    comparison["artifact_paths"] = {
        "clear_manifest": (metadata_dir / "clear_panel_manifest.json").as_posix(),
        "review": (metadata_dir / "clear_panel_review.json").as_posix(),
        "candidate_contact_sheet": (metadata_dir / "clear_input_candidate_contact_sheet.png").as_posix(),
        "panel": (metadata_dir / "panel.pt").as_posix(),
        "comparison_manifest": (metadata_dir / "comparison_manifest.json").as_posix(),
        "visualization_manifest": "visualization_manifest.json",
    }
    comparison["clear_panel"] = {
        **comparison["clear_panel"],
        "manifest": comparison["artifact_paths"]["clear_manifest"],
        "review": comparison["artifact_paths"]["review"],
        "manifest_sha256": file_sha256(target_clear_manifest),
        "review_sha256": file_sha256(target_review),
    }
    comparison["rendering"] = {
        "mode": "panel_only_rerender",
        "source_output_dir": str(source["source_dir"].resolve()),
        "source_panel_sha256": source["panel_sha256"],
        "source_comparison_manifest_sha256": file_sha256(source["paths"]["comparison"]),
        "source_visualization_manifest_sha256": file_sha256(source["paths"]["visualization"]),
        "cuda_replay": False,
        "optical_data_changed": False,
    }
    comparison.setdefault("resources", {})["render_only_from_panel"] = True
    comparison["resources"]["cuda_replay"] = False
    comparison["resources"]["source_panel_sha256"] = source["panel_sha256"]
    comparison["source_manifest_binding"] = {
        "source_visualization_manifest": _manifest_relative_path(
            source["source_dir"], source["paths"]["visualization"], "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING"
        ),
        "source_visualization_manifest_sha256": file_sha256(source["paths"]["visualization"]),
        "source_comparison_manifest": _manifest_relative_path(
            source["source_dir"], source["paths"]["comparison"], "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING"
        ),
        "source_comparison_manifest_sha256": file_sha256(source["paths"]["comparison"]),
        "source_panel": _manifest_relative_path(
            source["source_dir"], source["paths"]["panel"], "BLOCKED_PANEL_SOURCE_ARTIFACT_MISSING"
        ),
        "source_panel_sha256": source["panel_sha256"],
    }
    comparison["supersedes"] = "attempt_09"
    return comparison


def _authority_summary(
    contract: Mapping[str, Any],
    review: Mapping[str, Any],
    config_path: Path,
    review_path: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    rmal_contract = contract["rmal"]
    s9_parent = contract["parents"]["s9"]
    f32_manifest = _repo_path(s9_parent["artifact_manifest"])
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
        "frozen_mpcdar_s9_f32_manifest": str(f32_manifest.resolve()),
        "frozen_mpcdar_s9_f32_manifest_sha256": file_sha256(f32_manifest),
        "frozen_mpcdar_s9_f32_manifest_registered_sha256": s9_parent["artifact_manifest_sha256"],
        "frozen_mpcdar_s9_f32_rank": authority["f32_reader"].manifest["rank"],
        "rmal_target": rmal_contract["target"],
        "rmal_target_shape": rmal_contract["target_shape"],
        "parent_evidence": authority["parent_files"],
    }


def _comparison_manifest(
    output_dir: Path,
    git_boundary: Mapping[str, Any],
    final_git: Mapping[str, Any],
    clear_manifest: Mapping[str, Any],
    review: Mapping[str, Any],
    authority_summary: Mapping[str, Any],
    shared: Mapping[str, Any],
    arm_data: Mapping[str, Mapping[str, Any]],
    scales: Mapping[str, Any],
    model_before: str,
    model_after: str,
    elapsed_seconds: float,
    peak_cuda_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": "lightgen_scaled_chen_fiber_vis2_comparison_manifest.v2",
        "task_id": TASK_ID,
        "status": "completed",
        "classification": CLASSIFICATION,
        "diagnostic_only": True,
        "output_dir": str(output_dir.resolve()),
        "git": {"preflight": dict(git_boundary), "after_render": dict(final_git)},
        "clear_panel": {
            "manifest": "clear_panel_manifest.json",
            "review": "clear_panel_review.json",
            "manifest_sha256": file_sha256(output_dir / "clear_panel_manifest.json"),
            "review_sha256": file_sha256(output_dir / "clear_panel_review.json"),
            "selection_method": clear_manifest["selection_method"],
            "selected_before_fiber_evaluation": clear_manifest["selected_before_fiber_evaluation"],
            "downstream_outputs_used_for_selection": clear_manifest["downstream_outputs_used_for_selection"],
            "review_decision": review["decision"],
        },
        "authority": dict(authority_summary),
        "shared_frontend": {
            "stage_order": list(COMMON_STAGE_ORDER),
            "pre_ols_shape": shared["shared_pre_ols_tensor_shape"],
            "frontend_replayed_once": True,
            "same_pre_ols_tensor_object_to_all_arms": shared["pre_ols_pointer_identity"],
            "pre_ols_power_per_sample": _json_float_list(shared["pre_energy"]),
            "common_stage_metadata": shared["common_stage_metadata"],
        },
        "arm_order": list(ARM_ORDER),
        "arm_contracts": {arm: arm_data[arm]["contract"] for arm in ARM_ORDER},
        "arm_stage_order": list(ARM_STAGE_ORDER),
        "arm_metrics": {arm: arm_data[arm]["metrics"] for arm in ARM_ORDER},
        "modal_contract": {
            "SMF": {"mode_count": 100, "mode_families": ["LP01"], "ordering": "native FiberArrayOLS flat port order"},
            "FMF": {"mode_count": 200, "mode_families": ["LP01", "LP11a"], "ordering": "native FewModeFiberArrayOLS interleaved port×family order"},
            "MMF-M6": {
                "mode_count": 600,
                "mode_families": ["LP01", "LP11a", "ETA1", "XI2", "XIETA", "ETA2"],
                "synthetic_extra_families": list(MMF_SYNTHETIC_EXTRA_FAMILIES),
                "ordering": "NestedMultimodeFiberArrayOLS metadata flat ordering; FMF prefix followed by four candidate blocks",
                "physical_guided_mode_claim": False,
            },
        },
        "display_scales": dict(scales),
        "figures": _figure_manifest_names(),
        "layout": _layout_manifest_contract(),
        "artifact_paths": {
            "clear_manifest": "clear_panel_manifest.json",
            "review": "clear_panel_review.json",
            "candidate_contact_sheet": "clear_input_candidate_contact_sheet.png",
            "panel": "panel.pt",
            "comparison_manifest": "comparison_manifest.json",
            "visualization_manifest": "visualization_manifest.json",
        },
        "learned_state_freeze": {
            "model_state_excludes": "ols.* because the comparison intentionally changes the OLS arm",
            "model_before": model_before,
            "model_after": model_after,
            "unchanged": model_before == model_after,
            "all_model_parameters_requires_grad_false": True,
            "all_arm_operators_parameter_free": True,
            "optimizer_created": False,
            "backward_called": False,
            "refit": False,
        },
        "resources": {
            "device": "cuda",
            "seed": SEED,
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_allocated_bytes": peak_cuda_bytes,
            "sample_count": 10,
            "full_dataset_sweep": False,
            "rank_search": False,
            "mode_search": False,
            "seed_sweep": False,
            "render_only_from_panel": False,
            "cuda_replay": True,
        },
        "interpretation_boundary": "VIS2 is a qualitative clear-input full-pipeline display authority only; its ten-sample descriptors do not replace VIS1 fixed-panel quantitative descriptive authority and do not support overall performance, fiber ranking, physical MMF, reconstruction-manifold, generation, or promotion claims.",
    }


def _visualization_manifest(
    output_dir: Path,
    comparison: Mapping[str, Any],
    output_files: Sequence[Path],
    governance: Mapping[str, Any],
) -> dict[str, Any]:
    figures = comparison["figures"]
    artifact_paths = comparison.get("artifact_paths", {})
    clear_manifest_path = artifact_paths.get("clear_manifest", "clear_panel_manifest.json")
    contact_sheet_path = artifact_paths.get(
        "candidate_contact_sheet",
        "clear_input_candidate_contact_sheet.png",
    )
    review_path = artifact_paths.get("review", "clear_panel_review.json")
    panel_path = artifact_paths.get("panel", "panel.pt")
    comparison_path = artifact_paths.get("comparison_manifest", "comparison_manifest.json")
    required_artifacts = [
        clear_manifest_path,
        contact_sheet_path,
        review_path,
        f"{figures['figure_0']}.png",
        *(f"{name}.png" for name in figures["reconstruction_overview"]),
    ]
    for arm in ARM_ORDER:
        for group in STAGE_GROUPS:
            required_artifacts.extend(f"{name}.png" for name in figures["arm_stage_groups"][arm][group])
    required_artifacts.extend((comparison_path, "visualization_manifest.json", panel_path))
    tiff_suffix = "." + "tif" + "f"
    if any(path.suffix.lower() == tiff_suffix for path in output_files):
        raise RuntimeError("VIS2 compact output must not contain TIFF artifacts")
    return {
        "schema": "lightgen_scaled_chen_fiber_vis2_visualization_manifest.v2",
        "task_id": TASK_ID,
        "status": "completed",
        "classification": CLASSIFICATION,
        "layout_schema": VIS2_LAYOUT_SCHEMA,
        "artifact_layout_schema": ARTIFACT_LAYOUT_SCHEMA,
        "folder_layout": dict(ARTIFACT_FOLDERS),
        "diagnostic_only": True,
        "output_dir": str(output_dir.resolve()),
        "comparison_manifest": {
            "path": comparison_path,
            "sha256": file_sha256(output_dir / Path(comparison_path)),
        },
        "orientation_policy": copy.deepcopy(ORIENTATION_POLICY),
        "supersedes": comparison.get("supersedes"),
        "required_artifacts": required_artifacts,
        "output_files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output_files)
        ],
        "source_panel_sha256": comparison.get("rendering", {}).get("source_panel_sha256"),
        "source_manifest_binding": copy.deepcopy(comparison.get("source_manifest_binding")),
        "governance": dict(governance),
        "vis1_authority": {
            "status": "unchanged",
            "authority_output": str(VIS1_OUTPUT),
            "not_overwritten": True,
            "not_recomputed": True,
            "not_modified": True,
        },
        "unique_next_action": UNIQUE_NEXT_ACTION,
        "render_only_from_panel": bool(comparison.get("rendering", {}).get("cuda_replay") is False),
        "manifest_self_hash": "not embedded to avoid self-reference",
    }


def _run_full(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VIS2 full visualization requires CUDA; CPU is not a substitute")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("VIS2 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    git_boundary = _validate_git_boundary()
    if args.output_dir is None:
        raise RuntimeError("BLOCKED_CLEAR_PANEL_REVIEW_REQUIRED: pass the approved Stage A output directory")
    output_dir = _repo_path(args.output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(f"BLOCKED_CLEAR_PANEL_DIRECTORY_MISSING: {output_dir}")
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    allowed_stage_a = {
        "clear_panel_manifest.json",
        "clear_panel_review.json",
        "clear_input_candidate_contact_sheet.png",
        "clear_input_candidate_contact_sheet.pdf",
        "clear_input_candidate_contact_sheet.svg",
        "clear_input_candidate_contact_sheet.alignment.json",
        "clear_input_candidate_contact_sheet.alignment.svg",
    }
    if not existing.issubset(allowed_stage_a):
        raise RuntimeError("BLOCKED_OUTPUT_DIR_HAS_UNEXPECTED_FILES: full run must start from an untouched approved Stage A directory")
    clear_manifest = _load_clear_manifest(output_dir)
    review_path = output_dir / "clear_panel_review.json"
    if not review_path.is_file():
        raise RuntimeError("BLOCKED_CLEAR_PANEL_REVIEW_REQUIRED")
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("BLOCKED_CLEAR_PANEL_REVIEW_UNREADABLE") from exc
    if (
        not isinstance(review, Mapping)
        or review.get("schema") != "lightgen_scaled_chen_fiber_vis2_clear_panel_review.v1"
        or review.get("decision") != "APPROVED"
        or review.get("selection_method") != "input_only"
        or review.get("downstream_outputs_used_for_selection") is not False
        or review.get("selected_before_fiber_evaluation") is not True
        or review.get("clear_panel_manifest_sha256") != file_sha256(output_dir / "clear_panel_manifest.json")
    ):
        raise RuntimeError("BLOCKED_CLEAR_PANEL_REVIEW_CONTRACT_CHANGED")

    dataset_root = args.dataset_root
    train_images, train_labels = base.load_local_mnist(dataset_root, train=True)
    images, labels, indices_tensor, indices_by_label = _validate_clear_panel_against_dataset(
        clear_manifest, train_images, train_labels
    )
    rendered = base.render_intensity_batch(images)
    device = torch.device(args.device)

    contract, s9v_review, config_path, s9v_review_path = s9v._load_full_path_contract(args.visualization_config)
    authority = s9v._load_full_path_authority(contract, dataset_root, device)
    if not VIS1_OUTPUT.is_dir() or not (VIS1_OUTPUT / "comparison_manifest.json").is_file():
        raise RuntimeError("BLOCKED_VIS1_AUTHORITY_MISSING")

    model = authority["model"]
    model.eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("loaded Scaled-Chen model is not frozen")
    model_before = vis1._state_digest_without_ols(model)
    amplitude = rendered.amplitude.to(device)
    target = rendered.target.to(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.no_grad():
        common, arm_data, shared, _runtime = _capture_full_pipeline(
            model,
            amplitude,
            images,
            labels,
            target,
            authority,
        )
        model_after = vis1._state_digest_without_ols(model)
        if model_before != model_after:
            raise RuntimeError("VIS2 changed learned model state")
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    elapsed_seconds = time.monotonic() - started

    # The raw input tensors are kept CPU-side for display, while all optical
    # tensors are converted only after the no-grad CUDA replay has completed.
    common_cpu = {key: _tensor_cpu(value) for key, value in common.items()}
    arm_cpu: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        arm_cpu[arm] = dict(arm_data[arm])
        for key in (
            "flat_coefficients",
            "family_coefficients",
            "modal_power",
            "post_fiber",
            "decoder_layer_1",
            "decoder_layer_2",
            "cmos_field",
            "cmos_intensity",
            "mpcdar",
            "z",
        ):
            arm_cpu[arm][key] = _tensor_cpu(arm_data[arm][key])
        arm_cpu[arm]["arm"] = arm
    scales = _shared_display_scales(common_cpu, arm_cpu)
    target_cpu = _tensor_cpu(target)
    labels_cpu = _tensor_cpu(labels)
    indices_cpu = _tensor_cpu(indices_tensor)

    saved = _render_visualization_suite(
        common_cpu,
        arm_cpu,
        target_cpu,
        images,
        labels_cpu,
        indices_cpu.tolist(),
        scales,
        output_dir,
    )

    # Bind the panel payload to the already-reviewed manifest without storing
    # a second copy of the raw candidate pool or any complete mode bank.
    clear_manifest_for_payload = dict(clear_manifest)
    clear_manifest_for_payload["_path"] = str(output_dir / "clear_panel_manifest.json")
    panel_payload = _stage_payload(
        common_cpu,
        arm_cpu,
        target_cpu,
        images,
        labels,
        indices_cpu,
        scales,
        clear_manifest_for_payload,
    )
    panel_path = output_dir / "panel.pt"
    atomic_torch_save_new(panel_path, panel_payload, artifact_kind="VIS2 panel")
    saved.append(panel_path)

    authority_summary = _authority_summary(
        contract,
        s9v_review,
        config_path,
        s9v_review_path,
        authority,
    )
    comparison = _comparison_manifest(
        output_dir,
        git_boundary,
        _git_state(),
        clear_manifest,
        review,
        authority_summary,
        shared,
        arm_cpu,
        scales,
        model_before,
        model_after,
        elapsed_seconds,
        peak_cuda_bytes,
    )
    comparison_path = output_dir / "comparison_manifest.json"
    atomic_write_new_json(comparison_path, comparison, artifact_kind="VIS2 comparison manifest")
    saved.append(comparison_path)
    governance = {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
        "closed_task_relevant_bottleneck_not_supported": True,
        "bridge": "invalid / cloud_sources_sync_blocked",
        "vis1_authority": "quantitative_fixed_panel_descriptive_authority_unchanged",
        "vis2_authority": "qualitative_clear_input_full_pipeline_visualization_only",
    }
    output_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "visualization_manifest.json"
    )
    output_bytes = sum(path.stat().st_size for path in output_files)
    if output_bytes > 512 * 1024**2:
        raise RuntimeError(f"VIS2 output exceeds 512 MiB: {output_bytes} bytes")
    final_git = _git_state()
    visualization = _visualization_manifest(output_dir, comparison, output_files, governance)
    atomic_write_new_json(
        output_dir / "visualization_manifest.json",
        visualization,
        artifact_kind="VIS2 visualization manifest",
    )
    return {
        "status": "completed",
        "classification": CLASSIFICATION,
        "output_dir": str(output_dir.resolve()),
        "branch": final_git["branch"],
        "starting_head": git_boundary["starting_head"],
        "ending_head": final_git["commit"],
        "selected": [{"digit": digit, "dataset_index": indices_by_label[digit], "label": digit} for digit in range(10)],
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_allocated_bytes": peak_cuda_bytes,
        "model_state_unchanged": model_before == model_after,
        "output_bytes_before_visualization_manifest": output_bytes,
        "unique_next_action": UNIQUE_NEXT_ACTION,
    }


def _render_from_panel(source_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    git_boundary = _validate_git_boundary()
    source = _load_panel_payload(source_dir)
    target_dir = _repo_path(output_dir)
    if target_dir.exists():
        raise FileExistsError(f"refusing to overwrite panel rerender directory: {target_dir}")
    _check_disk_and_output_boundary(target_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    _copy_panel_source_artifacts(source, target_dir)

    indices = source["indices"].tolist()
    saved = _render_visualization_suite(
        source["common"],
        source["arm_data"],
        source["target"],
        source["images"],
        source["labels"],
        indices,
        source["scales"],
        target_dir,
    )
    if not saved:
        raise RuntimeError("VIS2 panel rerender produced no figures")

    comparison = _layout_comparison_manifest(source, target_dir)
    comparison_path = target_dir / Path(comparison["artifact_paths"]["comparison_manifest"])
    atomic_write_new_json(
        comparison_path,
        comparison,
        artifact_kind="VIS2 compact comparison manifest",
    )
    output_files = sorted(
        path
        for path in target_dir.rglob("*")
        if path.is_file() and path.name != "visualization_manifest.json"
    )
    output_bytes = sum(path.stat().st_size for path in output_files)
    if output_bytes > 512 * 1024**2:
        raise RuntimeError(f"VIS2 compact output exceeds 512 MiB: {output_bytes} bytes")
    final_git = _git_state()
    governance = {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
        "closed_task_relevant_bottleneck_not_supported": True,
        "bridge": "invalid / cloud_sources_sync_blocked",
        "vis1_authority": "quantitative_fixed_panel_descriptive_authority_unchanged",
        "vis2_authority": "qualitative_clear_input_full_pipeline_visualization_only",
    }
    visualization = _visualization_manifest(target_dir, comparison, output_files, governance)
    atomic_write_new_json(
        target_dir / "visualization_manifest.json",
        visualization,
        artifact_kind="VIS2 compact visualization manifest",
    )
    return {
        "status": "completed",
        "classification": CLASSIFICATION,
        "layout_schema": VIS2_LAYOUT_SCHEMA,
        "render_only_from_panel": True,
        "source_output_dir": str(source["source_dir"].resolve()),
        "source_panel_sha256": source["panel_sha256"],
        "output_dir": str(target_dir.resolve()),
        "branch": final_git["branch"],
        "starting_head": git_boundary["starting_head"],
        "ending_head": final_git["commit"],
        "selected": [
            {"digit": int(label), "dataset_index": int(index), "label": int(label)}
            for label, index in zip(source["labels"].tolist(), indices, strict=True)
        ],
        "output_bytes_before_visualization_manifest": output_bytes,
        "unique_next_action": UNIQUE_NEXT_ACTION,
    }


def _render_three_photo_from_panel(
    source_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    git_boundary = _validate_three_photo_git_boundary()
    source = _load_panel_payload(source_dir)
    if source["panel_sha256"].lower() != THREE_PHOTO_SOURCE_PANEL_SHA256.lower():
        raise RuntimeError("BLOCKED_THREE_PHOTO_SOURCE_PANEL_HASH_UNEXPECTED")
    target_dir = _repo_path(output_dir)
    if target_dir.exists():
        raise FileExistsError(f"refusing to overwrite three-photo directory: {target_dir}")
    _check_disk_and_output_boundary(target_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    _copy_panel_source_artifacts(source, target_dir)

    indices = source["indices"].tolist()
    saved = _render_three_photo_suite(
        source["images"],
        source["labels"],
        indices,
        source["arm_data"],
        target_dir,
    )
    expected_figure_count = len(THREE_PHOTO_PAGE_RANGES) * (
        len(THREE_PHOTO_FIGURE_EXPORT_SUFFIXES) + 2
    )
    if len(saved) != expected_figure_count:
        raise RuntimeError(
            "VIS2 three-photo renderer produced an unexpected figure count: "
            f"{len(saved)} != {expected_figure_count}"
        )

    comparison = _three_photo_comparison_manifest(
        source,
        target_dir,
        source["labels"],
        indices,
    )
    comparison_path = target_dir / Path(comparison["artifact_paths"]["comparison_manifest"])
    atomic_write_new_json(
        comparison_path,
        comparison,
        artifact_kind="VIS2 three-photo comparison manifest",
    )
    output_files = sorted(
        path
        for path in target_dir.rglob("*")
        if path.is_file() and path.name != "visualization_manifest.json"
    )
    output_bytes = sum(path.stat().st_size for path in output_files)
    if output_bytes > 512 * 1024**2:
        raise RuntimeError(
            f"VIS2 three-photo output exceeds 512 MiB: {output_bytes} bytes"
        )
    governance = {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
        "closed_task_relevant_bottleneck_not_supported": True,
        "bridge": "invalid / cloud_sources_sync_blocked",
        "vis1_authority": "quantitative_fixed_panel_descriptive_authority_unchanged",
        "vis2_authority": "qualitative_same_digit_three_fiber_comparison_only",
    }
    visualization = _three_photo_visualization_manifest(
        target_dir,
        comparison,
        output_files,
        governance,
    )
    atomic_write_new_json(
        target_dir / "visualization_manifest.json",
        visualization,
        artifact_kind="VIS2 three-photo visualization manifest",
    )
    final_boundary = _validate_three_photo_git_boundary()
    return {
        "status": "completed",
        "classification": CLASSIFICATION,
        "layout_schema": THREE_PHOTO_LAYOUT_SCHEMA,
        "render_only_from_panel": True,
        "cuda_replay": False,
        "optical_data_changed": False,
        "source_output_dir": str(source["source_dir"].resolve()),
        "source_panel_sha256": source["panel_sha256"],
        "output_dir": str(target_dir.resolve()),
        "branch": final_boundary["branch"],
        "starting_head": git_boundary["starting_head"],
        "ending_head": final_boundary["starting_head"],
        "pages": [
            {
                "page": page_number,
                "digits": list(positions),
                "figure_base": _three_photo_figure_base(
                    target_dir,
                    page_number,
                    positions,
                ).relative_to(target_dir).as_posix(),
            }
            for page_number, positions in enumerate(THREE_PHOTO_PAGE_RANGES, start=1)
        ],
        "output_bytes_before_visualization_manifest": output_bytes,
        "unique_next_action": UNIQUE_NEXT_ACTION,
    }


def _render_process_flow_from_panel(
    source_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    git_boundary = _validate_three_photo_git_boundary()
    source = _load_panel_payload(source_dir)
    if source["panel_sha256"].lower() != THREE_PHOTO_SOURCE_PANEL_SHA256.lower():
        raise RuntimeError("BLOCKED_PROCESS_FLOW_SOURCE_PANEL_HASH_UNEXPECTED")
    target_dir = _repo_path(output_dir)
    if target_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite process-flow directory: {target_dir}"
        )
    _check_disk_and_output_boundary(target_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    _copy_panel_source_artifacts(source, target_dir)

    indices = source["indices"].tolist()
    saved = _render_process_flow_suite(
        source["common"],
        source["arm_data"],
        source["images"],
        source["labels"],
        indices,
        source["scales"],
        target_dir,
    )
    expected_figure_count = len(PROCESS_FLOW_PAGE_RANGES) * (
        len(THREE_PHOTO_FIGURE_EXPORT_SUFFIXES) + 2
    )
    if len(saved) != expected_figure_count:
        raise RuntimeError(
            "VIS2 process-flow renderer produced an unexpected figure count: "
            f"{len(saved)} != {expected_figure_count}"
        )

    comparison = _process_flow_comparison_manifest(
        source,
        target_dir,
        source["labels"],
        indices,
    )
    comparison_path = target_dir / Path(
        comparison["artifact_paths"]["comparison_manifest"]
    )
    atomic_write_new_json(
        comparison_path,
        comparison,
        artifact_kind="VIS2 process-flow comparison manifest",
    )
    output_files = sorted(
        path
        for path in target_dir.rglob("*")
        if path.is_file() and path.name != "visualization_manifest.json"
    )
    output_bytes = sum(path.stat().st_size for path in output_files)
    if output_bytes > 512 * 1024**2:
        raise RuntimeError(
            f"VIS2 process-flow output exceeds 512 MiB: {output_bytes} bytes"
        )
    governance = {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
        "closed_task_relevant_bottleneck_not_supported": True,
        "bridge": "invalid / cloud_sources_sync_blocked",
        "vis1_authority": "quantitative_fixed_panel_descriptive_authority_unchanged",
        "vis2_authority": "qualitative_same_digit_three_fiber_process_flow_only",
    }
    visualization = _process_flow_visualization_manifest(
        target_dir,
        comparison,
        output_files,
        governance,
    )
    atomic_write_new_json(
        target_dir / "visualization_manifest.json",
        visualization,
        artifact_kind="VIS2 process-flow visualization manifest",
    )
    final_boundary = _validate_three_photo_git_boundary()
    return {
        "status": "completed",
        "classification": CLASSIFICATION,
        "layout_schema": PROCESS_FLOW_LAYOUT_SCHEMA,
        "artifact_layout_schema": PROCESS_FLOW_ARTIFACT_LAYOUT_SCHEMA,
        "row_order": PROCESS_FLOW_ROW_ORDER,
        "rows_per_digit": PROCESS_FLOW_ROWS_PER_DIGIT,
        "supersedes": "attempt_14",
        "render_only_from_panel": True,
        "cuda_replay": False,
        "optical_data_changed": False,
        "source_output_dir": str(source["source_dir"].resolve()),
        "source_panel_sha256": source["panel_sha256"],
        "source_comparison_manifest_sha256": file_sha256(
            source["paths"]["comparison"]
        ),
        "source_visualization_manifest_sha256": file_sha256(
            source["paths"]["visualization"]
        ),
        "output_dir": str(target_dir.resolve()),
        "branch": final_boundary["branch"],
        "starting_head": git_boundary["starting_head"],
        "ending_head": final_boundary["starting_head"],
        "pages": [
            {
                "page": page_number,
                "digits": [int(source["labels"][position]) for position in positions],
                "positions": list(positions),
                "figure_base": _process_flow_figure_base(
                    target_dir,
                    page_number,
                    positions,
                ).relative_to(target_dir).as_posix(),
            }
            for page_number, positions in enumerate(
                PROCESS_FLOW_PAGE_RANGES, start=1
            )
        ],
        "output_bytes_before_visualization_manifest": output_bytes,
        "unique_next_action": UNIQUE_NEXT_ACTION,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--visualization-config", type=Path, default=s9v.FULL_PATH_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage-a-only", action="store_true")
    parser.add_argument("--approve-clear-panel", action="store_true")
    parser.add_argument("--render-from-panel", type=Path, default=None)
    parser.add_argument("--render-three-photo-from-panel", type=Path, default=None)
    parser.add_argument("--render-process-flow-from-panel", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected_actions = sum(
        bool(value)
        for value in (
            args.stage_a_only,
            args.approve_clear_panel,
            args.render_from_panel is not None,
            args.render_three_photo_from_panel is not None,
            args.render_process_flow_from_panel is not None,
        )
    )
    if selected_actions > 1:
        raise RuntimeError("choose only one VIS2 stage action")
    if args.stage_a_only:
        _validate_git_boundary()
        output_dir = _repo_path(args.output_dir) if args.output_dir is not None else _next_output_dir()
        result = _write_stage_a(output_dir, args.dataset_root)
    elif args.approve_clear_panel:
        if args.output_dir is None:
            raise RuntimeError("--approve-clear-panel requires --output-dir")
        result = _approve_clear_panel(_repo_path(args.output_dir))
    elif args.render_from_panel is not None:
        if args.output_dir is None:
            raise RuntimeError("--render-from-panel requires --output-dir")
        result = _render_from_panel(args.render_from_panel, args.output_dir)
    elif args.render_three_photo_from_panel is not None:
        if args.output_dir is None:
            raise RuntimeError("--render-three-photo-from-panel requires --output-dir")
        result = _render_three_photo_from_panel(
            args.render_three_photo_from_panel,
            args.output_dir,
        )
    elif args.render_process_flow_from_panel is not None:
        if args.output_dir is None:
            raise RuntimeError("--render-process-flow-from-panel requires --output-dir")
        result = _render_process_flow_from_panel(
            args.render_process_flow_from_panel,
            args.output_dir,
        )
    else:
        result = _run_full(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
