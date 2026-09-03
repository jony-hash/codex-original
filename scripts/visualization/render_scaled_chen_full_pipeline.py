"""Render a read-only amplitude/phase plate for the frozen Scaled-Chen route.

The renderer is diagnostic only.  It loads the registered step-5000 checkpoint,
replays the fixed ten-digit confirmation panel through the same optical path, and
writes image exports plus compact provenance.  The full-path view is an eight-stage
amplitude/phase plate with an explicit display-only upright orientation policy.  It
never constructs an optimizer, calls an evaluator, performs backward, samples the
prior, or writes model/checkpoint artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ldm_project.v2.application.current import frozen_mpcdar_rank16_float32_readout as mpcdar_f32_runtime
from ldm_project.v2.application.current import frozen_mpcdar_rank16_readout as mpcdar_f64_runtime
from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.application.research import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research import optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery as mpcdar
from ldm_project.v2.application.research import optical_manifold_smf_bogt_manifold_readiness as mra
from ldm_project.v2.application.research import optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization as rmal
from ldm_project.v2.application.research import optical_manifold_smf_bogt_reconstruction_manifold_failure as rmfa
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
    "Encoder input\n(complex field)",
    "Encoder L1",
    "Encoder L2 =\nfiber input",
    "LP01 input ports",
    "LP01 output ports",
    "Fiber output =\ndecoder input",
    "Decoder L1",
    "Decoder L2",
    "CMOS incident field\n(pre-detection)",
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

FULL_PATH_TASK = "H-LOMA-CBOGT-S9V-FULL-PATH-VIS"
FULL_PATH_CLASSIFICATION = "descriptive_visualization_completed"
FULL_PATH_CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_full_path_visualization_v2"
FULL_PATH_LAYOUT_SCHEMA = "full_optical_path_eight_stage_upright_v1"
FULL_PATH_EXPECTED_HEAD = "d07fdaa17aa91d9f3e5498e9b1f10d91db766777"
FULL_PATH_CONFIG = Path(
    "configs/research/lightgen_optical_manifold_smf_bogt_full_path_visualization_v2.yaml"
)
FULL_PATH_REVIEW = Path(
    "configs/research/lightgen_optical_manifold_smf_bogt_full_path_visualization_review_v1.json"
)
FULL_PATH_RMAL_CONFIG = Path(
    "configs/research/lightgen_optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization_v1.yaml"
)
FULL_PATH_S7_CONFIG = Path(
    "configs/research/lightgen_optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery_v1.yaml"
)
FULL_PATH_OUTPUT = Path(
    "outputs/20260902_v2/visualizations/scaled_chen_full_path_reconstruction_v1_attempt_02"
)
FULL_PATH_PREVIOUS_OUTPUT = Path(
    "outputs/20260902_v2/visualizations/scaled_chen_full_path_reconstruction_v1_attempt_01"
)
FULL_PATH_PREVIOUS_MANIFEST = FULL_PATH_PREVIOUS_OUTPUT / "visualization_manifest.json"
FULL_PATH_PREVIOUS_MANIFEST_SHA256 = "ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749"
FULL_PATH_DETAIL_LABELS = (0, 2, 3, 5, 8)
FULL_PATH_PHASE_ALPHA = 0.01
FULL_PATH_PHASE_FLOOR = 1.0e-12
FULL_PATH_RECON_RANGE = (0.0, 1.0)
FULL_PATH_ERROR_RANGE = (0.0, 1.0)
FULL_PATH_F32_DIFF_RANGE = (0.0, 2.0e-4)
FULL_PATH_F32_DIFF_AMPLIFICATION = 1.0e6
FULL_PATH_RANK = 16
FULL_PATH_FULL_RANK = 256
FULL_PATH_RECON_METHODS = (
    "target",
    "class_template",
    "physical_output_camera",
    "raw_svd_rank16",
    "mean_preserving_raw_svd_rank16",
    "mpcdar_rank16_float64",
    "mpcdar_rank16_float32",
    "frozen_full_rank256_rmal",
)
FULL_PATH_STAGE_KEYS = (
    "slm_output",
    "encoder_input",
    "encoder_layer_1",
    "pre_ols",
    "post_ols",
    "decoder_layer_1",
    "decoder_layer_2",
    "cmos_field",
)
FULL_PATH_STAGE_LABELS = (
    "SLM output\n200×200, 2.5×2.5 μm\nMODEL=φ=0",
    "Front 4f\n100×100, 2×2 μm",
    "Encoder layer 1\n100×100, 2×2 μm",
    "Encoder layer 2\n100×100, 2×2 μm\nPre-OLS",
    "OLS output\n100×100, 2×2 μm",
    "Decoder layer 1\n100×100, 2×2 μm",
    "Decoder layer 2\n100×100, 2×2 μm",
    "Camera incident field\n200×200, 2.5×2.5 μm",
)
FULL_PATH_PAGE_DIGITS = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9),
)
FULL_PATH_REAL_KEYS = frozenset()
FULL_PATH_MODAL_KEYS = frozenset()
FULL_PATH_COMPLEX_KEYS = frozenset(FULL_PATH_STAGE_KEYS) - FULL_PATH_REAL_KEYS
FULL_PATH_STAGE_ALIASES = {
    "pre_ols": {
        "alias_of": "encoder_layer_2",
        "display_label": "Pre-OLS / Fiber input",
    },
    "post_ols": {
        "alias_of": "decoder_input",
        "display_label": "Post-OLS / Decoder input",
    },
}
FULL_PATH_DISPLAY_ROTATED_ORDER = (
    "encoder_input",
    "encoder_layer_1",
    "pre_ols",
    "post_ols",
    "decoder_layer_1",
    "decoder_layer_2",
)
FULL_PATH_DISPLAY_ROTATED_KEYS = frozenset(FULL_PATH_DISPLAY_ROTATED_ORDER)
FULL_PATH_NATIVE_DISPLAY_KEYS = ("slm_output", "cmos_field")
FULL_PATH_MAIN_FIGURE_WIDTH = 7.2

# These are deliberately explicit because a visualization must not imply a
# phase measurement where the source is real-valued or the sensor has already
# performed square-law detection.  The complex incident field immediately
# upstream of CMOS remains available in the numerical model; the detector
# readout is intensity-only.
FIBER_STAGE_AVAILABILITY = {
    "encoder_input": {
        "representation": "complex_field",
        "amplitude_available": True,
        "phase_available": True,
    },
    "encoder_layer_1": {
        "representation": "complex_field",
        "amplitude_available": True,
        "phase_available": True,
    },
    "fiber_input": {
        "representation": "complex_field",
        "amplitude_available": True,
        "phase_available": True,
        "alias_of": "encoder_layer_2",
    },
    "port_input": {
        "representation": "complex_port_coefficients",
        "amplitude_available": True,
        "phase_available": True,
    },
    "port_output": {
        "representation": "complex_port_coefficients",
        "amplitude_available": True,
        "phase_available": True,
    },
    "fiber_output": {
        "representation": "complex_field",
        "amplitude_available": True,
        "phase_available": True,
        "alias_of": "decoder_input",
    },
    "decoder_layer_1": {
        "representation": "complex_field",
        "amplitude_available": True,
        "phase_available": True,
    },
    "decoder_layer_2": {
        "representation": "complex_field",
        "amplitude_available": True,
        "phase_available": True,
    },
    "cmos_field": {
        "representation": "complex_field_upstream_of_detector",
        "amplitude_available": True,
        "phase_available": True,
        "detector_readout": "intensity_only",
    },
}


def _stage_plane_metadata(
    field: ComplexField, *, representation: str, **extra: Any
) -> dict[str, Any]:
    """Serialize the physical plane and operation that produced a field."""
    return {
        "representation": representation,
        "plane": field.plane.to_dict(),
        "provenance": field.provenance.to_dict(),
        **extra,
    }


def _render_phase_unavailable(axis: plt.Axes, message: str) -> None:
    """Make unavailable phase explicit instead of leaving a misleading blank."""
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("#e8e8e8")
    axis.text(
        0.5,
        0.5,
        textwrap.fill(message, width=14),
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color="#444444",
        linespacing=1.25,
    )


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
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
    return {"root": str(root), "branch": branch, "commit": commit, "clean": not status, "status": status}


def _verify_allowed_dirty_paths(status: Sequence[str], allowed_paths: Sequence[str]) -> list[str]:
    allowed = {str(Path(path).as_posix()) for path in allowed_paths}
    unexpected = []
    for line in status:
        path = line[3:].split(" -> ", 1)[-1].replace("\\", "/")
        if path not in allowed:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            "BLOCKED_UNEXPECTED_DIRTY_PATH: " + "; ".join(unexpected)
        )
    return sorted(allowed)


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
    slm_command = model.system.command(amplitude)
    slm_field = model.system.slm(slm_command)
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
    slm_command = model.system.command(amplitude)
    slm_field = model.system.slm(slm_command)
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
        "slm_command": slm_command.tensor,
        "slm_output": slm_field.tensor,
        "encoder_input": front_field.tensor,
        "encoder_layer_1": encoder_records[0].tensor,
        "fiber_input": encoded.tensor,
        "port_input": c_in,
        "port_output": c_out,
        "fiber_output": ols_field.tensor,
        "decoder_layer_1": decoder_records[0].tensor,
        "decoder_layer_2": decoder_records[1].tensor,
        "cmos_field": camera_field.tensor,
        "cmos_detector_intensity": camera_intensity,
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
        "stage_metadata": {
            "slm_command": {
                "representation": "real_amplitude_command",
                "amplitude_available": True,
                "phase_available": False,
                "phase_note": "IdealAmplitudeSLM command is real-valued; no independent phase channel is measured.",
                "calibration_id": slm_command.calibration_id,
                "plane": slm_command.plane.to_dict(),
            },
            "slm_output": _stage_plane_metadata(
                slm_field, representation="complex_field"
            ),
            "encoder_input": _stage_plane_metadata(
                front_field, representation="complex_field"
            ),
            "encoder_layer_1": _stage_plane_metadata(
                encoder_records[0], representation="complex_field"
            ),
            "fiber_input": _stage_plane_metadata(
                encoded,
                representation="complex_field",
                alias_of="encoder_layer_2",
                alias_note="The second encoder record and OLS input share the encoder_layer_2 plane; the fixed group transmission is applied at this boundary.",
            ),
            "port_input": {
                "representation": "complex_port_coefficients",
                "amplitude_available": True,
                "phase_available": True,
                "port_count": 100,
                "display_shape": [10, 10],
                "mode_model": "step_index_lp01_ports",
                "source_plane": encoded.plane.to_dict(),
            },
            "port_output": {
                "representation": "complex_port_coefficients",
                "amplitude_available": True,
                "phase_available": True,
                "port_count": 100,
                "display_shape": [10, 10],
                "mode_model": "step_index_lp01_ports",
                "transport": "100x100 complex identity",
            },
            "fiber_output": _stage_plane_metadata(
                ols_field,
                representation="complex_field",
                alias_of="decoder_input",
                alias_note="The FiberArrayOLS output is the decoder input boundary; no extra propagation slice is inserted.",
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
                phase_note="Phase is a model-side upstream field quantity, not a CMOS detector readout.",
            ),
            "cmos_detector_intensity": {
                "representation": "intensity_frame",
                "amplitude_available": False,
                "phase_available": False,
                "intensity_available": True,
                "formula": "I=|E|^2",
                "plane": camera_field.plane.to_dict(),
                "provenance": camera_field.provenance.to_dict(),
            },
        },
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
        "Spatial fields share one linear p99.5 amplitude scale across all digits and planes; LP01 ports use a common modal p99.5 scale. Phase is fixed to [−π, π] and masked below 1% of each field maximum. Encoder L2/fiber input and fiber output/decoder input are boundary aliases, not extra propagation slices. CMOS incident phase is a model-side upstream quantity; the detector itself records intensity only (see the separate detector panel). Intermediate planes and 10×10 port maps use a display-only 180° rotation; raw tensors and coordinates are unchanged.",
        ha="center",
        va="center",
        fontsize=5.8,
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


def _render_real_reference_plate(
    values: Mapping[str, torch.Tensor],
    output_base: Path,
    *,
    positions: Sequence[int],
    labels: Sequence[int],
    sample_indices: Sequence[int],
    value_key: str,
    title: str,
    top_label: str,
    unavailable_phase_label: str,
    scale_label: str,
    display_scale: float | None = None,
) -> list[Path]:
    """Render a real-valued command/readout with an explicit N/A phase row."""
    if not (len(positions) == len(labels) == len(sample_indices)):
        raise ValueError("reference plate metadata lengths do not match")
    arrays = [_display_array(values[value_key], position) for position in positions]
    scale = _shared_p995(arrays) if display_scale is None else float(display_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("reference plate display scale must be finite and positive")
    columns = len(arrays)
    figure, axes = plt.subplots(2, columns, figsize=(max(10.0, columns * 1.6), 3.45), squeeze=False)
    figure.subplots_adjust(left=0.055, right=0.94, bottom=0.18, top=0.77, wspace=0.10, hspace=0.30)
    panel_grid = [["" for _ in range(columns)] for _ in range(2)]
    for column, (position, label, sample_index, array) in enumerate(
        zip(positions, labels, sample_indices, arrays, strict=True)
    ):
        amplitude_axis, phase_axis = axes[0, column], axes[1, column]
        amplitude_axis.imshow(
            np.clip(np.abs(array) / max(scale, 1.0e-12), 0.0, 1.0),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            interpolation="nearest",
        )
        amplitude_axis.set_title(f"digit {label}\nidx {sample_index}", fontsize=6.2, pad=3)
        _render_phase_unavailable(phase_axis, unavailable_phase_label)
        if column == 0:
            amplitude_axis.set_ylabel("amplitude", fontsize=6.0, labelpad=2)
            phase_axis.set_ylabel("phase", fontsize=6.0, labelpad=2)
        for row, axis in enumerate((amplitude_axis, phase_axis)):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            panel_id = f"{value_key}_{'amplitude' if row == 0 else 'phase_na'}_{label}"
            panel_grid[row][column] = panel_id
            _panel_label(
                axis,
                chr(ord("a") + row * columns + column),
                y=0.94,
                vertical_alignment="top",
            )
    panel_ids = [panel_id for row in panel_grid for panel_id in row]
    row_groups = [list(row) for row in panel_grid]
    column_groups = [
        [panel_grid[row][column] for row in range(2)] for column in range(columns)
    ]
    figure.suptitle(title, fontsize=10, y=0.91)
    figure.text(
        0.5,
        0.065,
        f"{top_label}; shared linear p99.5 scale = {scale:.6g}.  {unavailable_phase_label}",
        ha="center",
        va="center",
        fontsize=6.0,
    )
    cbar_ax = figure.add_axes([0.952, 0.31, 0.014, 0.36])
    cbar = figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="gray"), cax=cbar_ax
    )
    cbar.set_label(scale_label, fontsize=5.5)
    cbar.ax.tick_params(labelsize=5, width=0.5, length=2)
    return _save_figure_bundle(
        figure,
        output_base,
        axes=[axis for row in axes for axis in row],
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        colorbar_axes=[cbar.ax],
        raster_dpi=300,
        include_tiff=False,
    )


def _phase_mask(array: np.ndarray) -> np.ndarray:
    amplitude = np.abs(array)
    threshold = max(float(amplitude.max()) * 0.01, 1.0e-12)
    return np.where(amplitude >= threshold, np.angle(array), np.nan)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _read_exact_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} is missing, stale, or tampered: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not readable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a mapping: {path}")
    return payload


def _load_full_path_contract(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    resolved_config = _repo_path(config_path)
    expected_config = _repo_path(FULL_PATH_CONFIG)
    if resolved_config != expected_config:
        raise RuntimeError(f"full-path visualization requires the registered config: {expected_config}")
    try:
        payload = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"full-path visualization config is unreadable: {resolved_config}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("full-path visualization config is not a mapping")
    if (
        payload.get("schema") != FULL_PATH_CONFIG_SCHEMA
        or payload.get("task_id") != FULL_PATH_TASK
        or payload.get("classification") != "descriptive_visualization_checkpoint"
        or payload.get("branch") != "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
        or payload.get("expected_head") != FULL_PATH_EXPECTED_HEAD
    ):
        raise RuntimeError("full-path visualization contract identity is mismatched")
    if (
        payload.get("layout_schema") != FULL_PATH_LAYOUT_SCHEMA
        or payload.get("supersedes") != "attempt_01"
        or str(payload.get("supersedes_manifest", "")).replace("\\", "/")
        != str(FULL_PATH_PREVIOUS_MANIFEST).replace("\\", "/")
        or payload.get("supersedes_manifest_sha256") != FULL_PATH_PREVIOUS_MANIFEST_SHA256
    ):
        raise RuntimeError("full-path visualization layout lineage is mismatched")
    sample_panel = payload.get("sample_panel")
    if (
        not isinstance(sample_panel, Mapping)
        or sample_panel.get("count") != 10
        or sample_panel.get("order") != list(range(10))
        or sample_panel.get("detail_labels") != list(FULL_PATH_DETAIL_LABELS)
    ):
        raise RuntimeError("full-path visualization sample panel contract is mismatched")
    indices = sample_panel.get("indices_by_label")
    expected_indices = {str(label): DIGIT_CONFIRMATION_INDICES[label] for label in range(10)}
    if indices != expected_indices:
        raise RuntimeError("full-path visualization sample indices are mismatched")
    if payload.get("stages") != list(FULL_PATH_STAGE_KEYS):
        raise RuntimeError("full-path visualization stage contract is mismatched")
    if payload.get("methods") != list(FULL_PATH_RECON_METHODS):
        raise RuntimeError("full-path visualization method contract is mismatched")
    display = payload.get("display")
    if not isinstance(display, Mapping):
        raise RuntimeError("full-path visualization display contract is missing")
    if (
        display.get("phase") != "angle(E), raw, range [-pi, pi]"
        or display.get("phase_mask") != "max(0.01 * max(abs(E)), 1e-12), per stage and sample"
        or display.get("reconstruction_range") != list(FULL_PATH_RECON_RANGE)
        or display.get("target_error_range") != list(FULL_PATH_ERROR_RANGE)
        or display.get("f64_f32_difference_range") != list(FULL_PATH_F32_DIFF_RANGE)
        or display.get("f64_f32_amplification") != FULL_PATH_F32_DIFF_AMPLIFICATION
        or display.get("layout_schema") != FULL_PATH_LAYOUT_SCHEMA
        or display.get("display_rotation") != "upright_common_object_frame_v1"
        or display.get("display_rotation_keys") != list(FULL_PATH_DISPLAY_ROTATED_ORDER)
        or display.get("native_coordinate_keys") != list(FULL_PATH_NATIVE_DISPLAY_KEYS)
        or display.get("front_relay_magnification") != -0.4
        or display.get("back_relay_magnification") != -2.5
        or display.get("net_magnification") != 1.0
    ):
        raise RuntimeError("full-path visualization display contract is mismatched")
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        authorization.get(name) is not True
        for name in ("no_training", "no_update", "no_optimizer", "no_backward", "no_new_basis", "no_rank_search")
    ) or authorization.get("visualization_is_scientific_gate") is not False:
        raise RuntimeError("full-path visualization authorization boundary is mismatched")

    review_path = _repo_path(FULL_PATH_REVIEW)
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"full-path visualization review is unreadable: {review_path}") from exc
    if not isinstance(review, dict):
        raise RuntimeError("full-path visualization review is not a mapping")
    if (
        review.get("schema") != "lightgen_optical_manifold_smf_bogt_full_path_visualization_review.v1"
        or review.get("task_id") != FULL_PATH_TASK
        or review.get("decision") != "APPROVED WITH REQUIRED DISPLAY CONTRACT FREEZE"
        or review.get("classification") != "descriptive_visualization_checkpoint"
    ):
        raise RuntimeError("full-path visualization review approval is mismatched")
    approved = review.get("approved_contract")
    if (
        not isinstance(approved, Mapping)
        or approved.get("sample_count") != 10
        or approved.get("sample_source") != "confirmation_panel_train_indices_55000_59999"
        or approved.get("phase_range") != [-np.pi, np.pi]
        or approved.get("phase_mask") != "max(0.01 * max(abs(E)), 1e-12), per stage and sample"
        or approved.get("reconstruction_range") != list(FULL_PATH_RECON_RANGE)
        or approved.get("target_error_range") != list(FULL_PATH_ERROR_RANGE)
        or approved.get("f64_f32_difference_range") != list(FULL_PATH_F32_DIFF_RANGE)
        or approved.get("f64_f32_amplification") != FULL_PATH_F32_DIFF_AMPLIFICATION
        or approved.get("no_phase_shift") is not True
        or approved.get("no_per_sample_reconstruction_normalization") is not True
        or approved.get("visualization_is_scientific_gate") is not False
    ):
        raise RuntimeError("full-path visualization approved display contract is mismatched")
    if any(
        review.get(name) is not False
        for name in ("training_authorized", "formal_promotion", "generation")
    ):
        raise RuntimeError("full-path visualization authorization flags are mismatched")
    return payload, review, resolved_config, review_path


def _verify_full_path_previous_attempt(contract: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _repo_path(contract["supersedes_manifest"])
    if manifest_path != _repo_path(FULL_PATH_PREVIOUS_MANIFEST):
        raise RuntimeError("full-path visualization superseded manifest path is mismatched")
    manifest = _read_exact_json(
        manifest_path,
        str(contract["supersedes_manifest_sha256"]),
        "previous full-path visualization manifest",
    )
    if (
        manifest.get("schema") != "lightgen_optical_manifold_smf_bogt_full_path_visualization.v1"
        or manifest.get("task_id") != FULL_PATH_TASK
        or manifest.get("status") != "completed"
    ):
        raise RuntimeError("previous full-path visualization manifest is not the registered attempt_01")
    return {
        "path": str(manifest_path),
        "sha256": str(contract["supersedes_manifest_sha256"]),
        "schema": manifest["schema"],
        "git": manifest.get("git"),
    }


def _verify_full_path_parent_files(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    parents = contract.get("parents")
    if not isinstance(parents, Mapping):
        raise RuntimeError("full-path visualization parent contract is missing")
    verified: dict[str, dict[str, Any]] = {}
    for name in ("s7", "s8", "s9"):
        parent = parents.get(name)
        if not isinstance(parent, Mapping):
            raise RuntimeError(f"full-path visualization parent {name} is missing")
        records: dict[str, Any] = {"id": parent.get("id"), "status": parent.get("status"), "classification": parent.get("classification")}
        for field in ("validation", "audit", "compact"):
            path = _repo_path(parent[field])
            value = _read_exact_json(path, str(parent[f"{field}_sha256"]), f"S{name[1:]} {field}")
            if value.get("status") != "passed":
                raise RuntimeError(f"S{name[1:]} {field} is not passed")
            records[field] = {"path": str(path), "sha256": str(parent[f"{field}_sha256"]), "status": value.get("status")}
            if field == "compact" and value.get("classification") != parent.get("classification"):
                raise RuntimeError(f"S{name[1:]} compact classification is mismatched")
        for field in ("artifact_manifest", "artifact_payload"):
            if field in parent:
                path = _repo_path(parent[field])
                expected = str(parent[f"{field}_sha256"])
                if not path.is_file() or file_sha256(path) != expected:
                    raise RuntimeError(f"S{name[1:]} {field} is missing, stale, or tampered: {path}")
                records[field] = {"path": str(path), "sha256": expected}
        verified[name] = records
    return verified


def _load_full_path_authority(
    contract: Mapping[str, Any], dataset_root: Path, device: torch.device
) -> dict[str, Any]:
    parent_files = _verify_full_path_parent_files(contract)
    rmal_contract = contract.get("rmal")
    if not isinstance(rmal_contract, Mapping):
        raise RuntimeError("RMAL authority contract is missing")
    rmal_config = _repo_path(rmal_contract["config"])
    if rmal_config != _repo_path(FULL_PATH_RMAL_CONFIG) or file_sha256(rmal_config) != rmal_contract["config_sha256"]:
        raise RuntimeError("RMAL config identity is mismatched")
    checkpoint = _repo_path(rmal_contract["checkpoint"])
    if not checkpoint.is_file() or file_sha256(checkpoint) != rmal_contract["checkpoint_sha256"]:
        raise RuntimeError("RMAL checkpoint identity is mismatched")

    s7_spec = mpcdar._load_spec(_repo_path(FULL_PATH_S7_CONFIG))
    upstream = mpcdar._load_upstream(s7_spec, dataset_root)
    spectral = mpcdar._spectral_lock(upstream)
    runtime = upstream["runtime"]
    source = rmfa._load_source(runtime.rmal_consumer.bundle, load_checkpoints=True)
    formal = runtime.rmal_consumer.bundle.parent.parent.formal
    checkpoint_from_source = source.checkpoints[rmal.PRIMARY_CHECKPOINT]
    if source.hashes.get(rmal.PRIMARY_CHECKPOINT) != rmal_contract["checkpoint_sha256"]:
        raise RuntimeError("loaded RMAL checkpoint is not the registered primary checkpoint")
    model = mra._load_scaled_model(formal, checkpoint_from_source, device)

    s8_parent = contract["parents"]["s8"]
    s9_parent = contract["parents"]["s9"]
    f64_manifest = _repo_path(s8_parent["artifact_manifest"])
    f32_manifest = _repo_path(s9_parent["artifact_manifest"])
    f64_reader = mpcdar_f64_runtime.load_frozen_mpcdar_rank16_readout(f64_manifest)
    f32_reader = mpcdar_f32_runtime.load_frozen_mpcdar_rank16_float32_readout(f32_manifest)
    if f64_reader.manifest.get("rank") != FULL_PATH_RANK or f32_reader.manifest.get("rank") != FULL_PATH_RANK:
        raise RuntimeError("frozen MPCDAR artifact rank is not 16")
    return {
        "parent_files": parent_files,
        "rmal_config": str(rmal_config),
        "rmal_checkpoint": str(checkpoint),
        "upstream": upstream,
        "spectral": spectral,
        "model": model,
        "formal": formal,
        "f64_reader": f64_reader,
        "f32_reader": f32_reader,
    }


def _full_path_views(
    values: Mapping[str, torch.Tensor], *, sample_positions: Sequence[int]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, float]]:
    views: dict[str, dict[str, dict[str, Any]]] = {}
    scales: dict[str, float] = {}
    for key in FULL_PATH_STAGE_KEYS:
        value = values[key]
        arrays = [
            _display_array(
                value,
                position,
                modal=key in FULL_PATH_MODAL_KEYS,
                rotate=key in FULL_PATH_DISPLAY_ROTATED_KEYS,
            )
            for position in sample_positions
        ]
        scale = 1.0 if key == "target" else _shared_p995(arrays)
        scales[key] = scale
        per_sample: dict[str, dict[str, Any]] = {}
        for position, raw in zip(sample_positions, arrays, strict=True):
            amplitude = np.abs(raw)
            if key in FULL_PATH_REAL_KEYS:
                phase = None
                threshold = None
                valid_fraction = 0.0
            else:
                phase = np.angle(raw)
                threshold = max(float(amplitude.max()) * FULL_PATH_PHASE_ALPHA, FULL_PATH_PHASE_FLOOR)
                phase = np.where(amplitude >= threshold, phase, np.nan)
                valid_fraction = float(np.isfinite(phase).mean())
            per_sample[str(position)] = {
                "amplitude": np.clip(amplitude / max(scale, FULL_PATH_PHASE_FLOOR), 0.0, 1.0),
                "phase": phase,
                "phase_threshold": threshold,
                "phase_valid_fraction": valid_fraction,
            }
        views[key] = per_sample
    return views, scales


def _as_reconstruction_images(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 2 and tuple(value.shape[1:]) == (40000,):
        result = value.reshape(-1, 1, 200, 200)
    elif value.ndim == 4 and tuple(value.shape[1:]) == (1, 200, 200):
        result = value
    else:
        raise RuntimeError(f"{name} has an invalid reconstruction shape: {tuple(value.shape)}")
    if not torch.is_floating_point(result) or torch.is_complex(result) or not bool(torch.isfinite(result).all()):
        raise RuntimeError(f"{name} is not a finite real reconstruction")
    return result.detach().cpu().float()


def _apply_frozen_rmal_readout(probe: Any, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or tuple(features.shape[1:]) != (256,) or not torch.is_floating_point(features):
        raise RuntimeError("RMAL full-readout features have an invalid shape")
    if tuple(probe.full_weights.shape) != (257, 40000) or not torch.is_floating_point(probe.full_weights):
        raise RuntimeError("RMAL frozen full-readout weights have an invalid shape")
    augmented = torch.cat(
        (features.float(), torch.ones((features.shape[0], 1), dtype=torch.float32)),
        dim=1,
    )
    result = augmented.mm(probe.full_weights).reshape(-1, 1, 200, 200)
    if not torch.isfinite(result).all():
        raise RuntimeError("RMAL frozen full-readout output is not finite")
    return result


def _replay_full_path_reconstructions(
    authority: Mapping[str, Any],
    values: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    target: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    camera = values["cmos_detector_intensity"].detach().cpu().float()
    features = mra._feature_projection(camera, complex_value=False).float()
    if tuple(features.shape) != (10, 256):
        raise RuntimeError(f"visualization feature projection shape changed: {tuple(features.shape)}")
    probe = authority["upstream"]["runtime"].rmal_probe
    spectral = authority["spectral"]
    W64 = probe.weights.detach().cpu().to(torch.float64).contiguous()
    mu_fit = authority["f64_reader"].mu_fit.detach().cpu().to(torch.float64).contiguous()
    predictions = {
        "target": target.detach().cpu().float(),
        "class_template": mpcdar._target_templates(images, labels).detach().cpu().float(),
        "physical_output_camera": camera,
        "raw_svd_rank16": _as_reconstruction_images(
            mpcdar._old_raw_prediction(features, spectral["U"], W64, probe.bias, FULL_PATH_RANK),
            "raw_svd_rank16",
        ),
        "mean_preserving_raw_svd_rank16": _as_reconstruction_images(
            mpcdar._mean_preserving_raw_prediction(
                features, spectral["U"], W64, probe.bias, mu_fit, FULL_PATH_RANK
            ),
            "mean_preserving_raw_svd_rank16",
        ),
        "mpcdar_rank16_float64": _as_reconstruction_images(
            authority["f64_reader"].predict(features), "mpcdar_rank16_float64"
        ),
        "mpcdar_rank16_float32": _as_reconstruction_images(
            authority["f32_reader"].predict(features), "mpcdar_rank16_float32"
        ),
        "frozen_full_rank256_rmal": _as_reconstruction_images(
            _apply_frozen_rmal_readout(probe, features),
            "frozen_full_rank256_rmal",
        ),
    }
    target_cpu = predictions["target"]
    if any(value.shape != target_cpu.shape for value in predictions.values()):
        raise RuntimeError("reconstruction methods do not share the target shape")
    method_mse = {
        name: float((value - target_cpu).square().mean()) for name, value in predictions.items()
    }
    per_sample_mse = {
        name: [float(item) for item in (value - target_cpu).square().flatten(1).mean(dim=1)]
        for name, value in predictions.items()
    }
    metrics = {
        "method_mse": method_mse,
        "per_sample_mse": per_sample_mse,
        "feature_shape": list(features.shape),
        "feature_contract": "output.camera -> mra._feature_projection(real, complex_value=False)",
        "reshape": "row_major_[N,40000] -> [N,1,200,200]",
    }
    return predictions, metrics


def _render_full_path_plate(
    views: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_base: Path,
    *,
    positions: Sequence[int],
    labels: Sequence[int],
    sample_indices: Sequence[int],
    page_label: str,
) -> list[Path]:
    if not (len(positions) == len(labels) == len(sample_indices)):
        raise ValueError("full-path plate metadata lengths do not match")
    if not positions:
        raise ValueError("full-path plate requires at least one digit")
    columns = len(FULL_PATH_STAGE_KEYS)
    rows = len(positions) * 2
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(FULL_PATH_MAIN_FIGURE_WIDTH, max(7.4, rows * 0.72 + 1.2)),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.955,
        bottom=0.095,
        top=0.84,
        wspace=0.035,
        hspace=0.20,
    )
    panel_ids: list[str] = []
    row_groups: list[list[str]] = []
    column_groups: list[list[str]] = [[] for _ in range(columns)]
    for row_position, (position, label, sample_index) in enumerate(
        zip(positions, labels, sample_indices, strict=True)
    ):
        row_ids_amplitude: list[str] = []
        row_ids_phase: list[str] = []
        for phase_row in (False, True):
            row = row_position * 2 + int(phase_row)
            row_ids = row_ids_phase if phase_row else row_ids_amplitude
            for column, (key, title) in enumerate(zip(FULL_PATH_STAGE_KEYS, FULL_PATH_STAGE_LABELS, strict=True)):
                axis = axes[row, column]
                view = views[key][str(position)]
                if phase_row:
                    if key in FULL_PATH_REAL_KEYS:
                        _render_phase_unavailable(axis, "phase N/A")
                    else:
                        axis.imshow(
                            view["phase"],
                            cmap="twilight",
                            vmin=-np.pi,
                            vmax=np.pi,
                            origin="lower",
                            interpolation="nearest",
                            aspect="equal",
                        )
                else:
                    axis.imshow(
                        view["amplitude"],
                        cmap="magma",
                        vmin=0.0,
                        vmax=1.0,
                        origin="lower",
                        interpolation="nearest",
                        aspect="equal",
                    )
                axis.set_aspect("equal", adjustable="box")
                if row == 0:
                    axis.set_title(title, fontsize=5.0, pad=2)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                panel_id = f"{page_label}_digit{label}_{'phase' if phase_row else 'amplitude'}_{key}"
                panel_ids.append(panel_id)
                row_ids.append(panel_id)
                column_groups[column].append(panel_id)

        row_groups.extend((row_ids_amplitude, row_ids_phase))
        axes[row_position * 2, 0].set_ylabel(f"digit {label}\n|E|", fontsize=5.2, labelpad=2)
        axes[row_position * 2 + 1, 0].set_ylabel("phase", fontsize=5.0, labelpad=2)
    figure.suptitle(
        f"{FULL_PATH_TASK} | eight-stage full optical path | {page_label}",
        fontsize=8.0,
        y=0.955,
    )
    figure.text(
        0.5,
        0.035,
        "Amplitude: shared linear p99.5 scale per stage. Phase: raw [−π,π], masked below 1% of each sample maximum. Upright display is CPU-only normalization; SLM output and camera field retain native coordinates.",
        ha="center",
        va="center",
        fontsize=5.0,
    )
    amp_cbar_ax = figure.add_axes([0.968, 0.55, 0.010, 0.22])
    phase_cbar_ax = figure.add_axes([0.968, 0.24, 0.010, 0.22])
    amp_cbar = figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma"), cax=amp_cbar_ax
    )
    phase_cbar = figure.colorbar(
        ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap="twilight"), cax=phase_cbar_ax
    )
    amp_cbar.set_label("display |E|", fontsize=5.0)
    phase_cbar.set_label("phase [rad]", fontsize=5.0)
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
        require_panel_labels=False,
        raster_dpi=300,
        include_tiff=False,
    )


RECONSTRUCTION_LABELS = {
    "target": "Target",
    "class_template": "Class template",
    "physical_output_camera": "Physical",
    "raw_svd_rank16": "Raw-SVD R16",
    "mean_preserving_raw_svd_rank16": "MP-RAW R16",
    "mpcdar_rank16_float64": "MPCDAR R16 F64",
    "mpcdar_rank16_float32": "MPCDAR R16 F32",
    "frozen_full_rank256_rmal": "Full R256",
}


def _render_reconstruction_grid(
    predictions: Mapping[str, torch.Tensor],
    output_base: Path,
    *,
    positions: Sequence[int],
    labels: Sequence[int],
    sample_indices: Sequence[int],
    method_keys: Sequence[str],
    method_mse: Mapping[str, float],
) -> list[Path]:
    figure, axes = plt.subplots(
        len(positions),
        len(method_keys),
        figsize=(16.0, max(7.5, len(positions) * 1.65)),
        squeeze=False,
    )
    figure.subplots_adjust(left=0.045, right=0.955, bottom=0.09, top=0.88, wspace=0.035, hspace=0.18)
    panel_ids: list[str] = []
    row_groups: list[list[str]] = []
    column_groups: list[list[str]] = [[] for _ in method_keys]
    for row, (position, label, sample_index) in enumerate(zip(positions, labels, sample_indices, strict=True)):
        row_ids: list[str] = []
        for column, key in enumerate(method_keys):
            axis = axes[row, column]
            array = _display_array(predictions[key], position)
            axis.imshow(
                np.clip(array, 0.0, 1.0),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            if row == 0:
                axis.set_title(f"{RECONSTRUCTION_LABELS[key]}\nMSE {method_mse[key]:.5g}", fontsize=6.4, pad=3)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            panel_id = f"reconstruction_digit{label}_{key}"
            panel_ids.append(panel_id)
            row_ids.append(panel_id)
            column_groups[column].append(panel_id)
        row_groups.append(row_ids)
        axes[row, 0].set_ylabel(f"digit {label}\nidx {sample_index}", fontsize=6.0, labelpad=2)
    figure.suptitle(
        f"{FULL_PATH_TASK} | reconstruction family | fixed [0,1] display range | digits {labels[0]}–{labels[-1]}",
        fontsize=10,
        y=0.95,
    )
    figure.text(
        0.5,
        0.03,
        "All methods use the same target, row-major [N,40000]→[N,1,200,200] reshape, orientation, colormap, and display range. MSE values are descriptive visualization-panel metrics.",
        ha="center",
        va="center",
        fontsize=5.8,
    )
    cbar_ax = figure.add_axes([0.963, 0.33, 0.010, 0.34])
    cbar = figure.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="gray"), cax=cbar_ax)
    cbar.set_label("fixed display range", fontsize=5.5)
    cbar.ax.tick_params(labelsize=5, width=0.5, length=2)
    return _save_figure_bundle(
        figure,
        output_base,
        axes=[axis for row in axes for axis in row],
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        colorbar_axes=[cbar.ax],
        require_panel_labels=False,
        raster_dpi=300,
        include_tiff=False,
    )


def _render_repair_detail(
    predictions: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    output_base: Path,
    *,
    positions: Sequence[int],
    labels: Sequence[int],
    sample_indices: Sequence[int],
) -> list[Path]:
    prediction_keys = (
        "target",
        "raw_svd_rank16",
        "mean_preserving_raw_svd_rank16",
        "mpcdar_rank16_float32",
        "frozen_full_rank256_rmal",
    )
    error_keys = prediction_keys[1:]
    columns = len(prediction_keys) + len(error_keys)
    figure, axes = plt.subplots(
        len(positions),
        columns,
        figsize=(18.0, max(5.0, len(positions) * 1.95)),
        squeeze=False,
    )
    figure.subplots_adjust(left=0.05, right=0.955, bottom=0.11, top=0.84, wspace=0.035, hspace=0.20)
    panel_ids: list[str] = []
    row_groups: list[list[str]] = []
    column_groups: list[list[str]] = [[] for _ in range(columns)]
    for row, (position, label, sample_index) in enumerate(zip(positions, labels, sample_indices, strict=True)):
        row_ids: list[str] = []
        for column, key in enumerate(prediction_keys):
            axis = axes[row, column]
            array = _display_array(predictions[key], position)
            axis.imshow(
                np.clip(array, 0.0, 1.0),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            if row == 0:
                axis.set_title(RECONSTRUCTION_LABELS[key], fontsize=6.2, pad=3)
            panel_id = f"detail_digit{label}_{key}"
            panel_ids.append(panel_id)
            row_ids.append(panel_id)
            column_groups[column].append(panel_id)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        for offset, key in enumerate(error_keys, start=len(prediction_keys)):
            axis = axes[row, offset]
            difference = (predictions[key][position] - target[position]).abs().squeeze(0).numpy()
            axis.imshow(
                np.clip(difference, 0.0, 1.0),
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            if row == 0:
                axis.set_title(f"|{RECONSTRUCTION_LABELS[key]}−target|", fontsize=6.0, pad=3)
            panel_id = f"detail_digit{label}_{key}_error"
            panel_ids.append(panel_id)
            row_ids.append(panel_id)
            column_groups[offset].append(panel_id)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        row_groups.append(row_ids)
        axes[row, 0].set_ylabel(f"digit {label}\nidx {sample_index}", fontsize=6.0, labelpad=2)
    figure.suptitle(
        f"{FULL_PATH_TASK} | descriptive repair/error detail | fixed classes {list(FULL_PATH_DETAIL_LABELS)}",
        fontsize=10,
        y=0.94,
    )
    figure.text(
        0.5,
        0.035,
        "Prediction panels use fixed [0,1]; target-error panels use one shared fixed [0,1] error range. No per-method contrast adjustment is applied.",
        ha="center",
        va="center",
        fontsize=5.8,
    )
    image_cbar_ax = figure.add_axes([0.963, 0.50, 0.010, 0.24])
    error_cbar_ax = figure.add_axes([0.963, 0.18, 0.010, 0.24])
    image_cbar = figure.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="gray"), cax=image_cbar_ax)
    error_cbar = figure.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma"), cax=error_cbar_ax)
    image_cbar.set_label("image [0,1]", fontsize=5.5)
    error_cbar.set_label("absolute error [0,1]", fontsize=5.5)
    for cbar in (image_cbar, error_cbar):
        cbar.ax.tick_params(labelsize=5, width=0.5, length=2)
    return _save_figure_bundle(
        figure,
        output_base,
        axes=[axis for row in axes for axis in row],
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        colorbar_axes=[image_cbar.ax, error_cbar.ax],
        require_panel_labels=False,
        raster_dpi=300,
        include_tiff=False,
    )


def _render_f64_f32_detail(
    predictions: Mapping[str, torch.Tensor],
    output_base: Path,
    *,
    positions: Sequence[int],
    labels: Sequence[int],
    sample_indices: Sequence[int],
) -> list[Path]:
    columns = 4
    figure, axes = plt.subplots(
        len(positions),
        columns,
        figsize=(8.8, max(4.0, len(positions) * 1.85)),
        squeeze=False,
    )
    figure.subplots_adjust(left=0.055, right=0.925, bottom=0.12, top=0.83, wspace=0.08, hspace=0.22)
    panel_ids: list[str] = []
    row_groups: list[list[str]] = []
    column_groups: list[list[str]] = [[] for _ in range(columns)]
    for row, (position, label, sample_index) in enumerate(zip(positions, labels, sample_indices, strict=True)):
        row_ids: list[str] = []
        f64 = _display_array(predictions["mpcdar_rank16_float64"], position)
        f32 = _display_array(predictions["mpcdar_rank16_float32"], position)
        difference = np.abs(f64 - f32)
        arrays = (
            (np.clip(f64, 0.0, 1.0), "MPCDAR R16 F64", "gray", 1.0),
            (np.clip(f32, 0.0, 1.0), "MPCDAR R16 F32", "gray", 1.0),
            (np.clip(difference, 0.0, FULL_PATH_F32_DIFF_RANGE[1]), "abs(F64−F32) [0,2e−4]", "magma", FULL_PATH_F32_DIFF_RANGE[1]),
            (np.clip(difference * FULL_PATH_F32_DIFF_AMPLIFICATION, 0.0, FULL_PATH_F32_DIFF_RANGE[1] * FULL_PATH_F32_DIFF_AMPLIFICATION), "display-only ×10⁶", "magma", FULL_PATH_F32_DIFF_RANGE[1] * FULL_PATH_F32_DIFF_AMPLIFICATION),
        )
        for column, (array, title, cmap, vmax) in enumerate(arrays):
            axis = axes[row, column]
            axis.imshow(
                array,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            if row == 0:
                axis.set_title(title, fontsize=6.0, pad=3)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            panel_id = f"f64_f32_digit{label}_{column}"
            panel_ids.append(panel_id)
            row_ids.append(panel_id)
            column_groups[column].append(panel_id)
        row_groups.append(row_ids)
        axes[row, 0].set_ylabel(f"digit {label}\nidx {sample_index}", fontsize=6.0, labelpad=2)
    figure.suptitle(
        f"{FULL_PATH_TASK} | MPCDAR rank-16 F64/F32 descriptive comparison | fixed classes {list(FULL_PATH_DETAIL_LABELS)}",
        fontsize=9.5,
        y=0.94,
    )
    figure.text(
        0.5,
        0.035,
        "Raw difference uses fixed [0,2e−4] from the S9 registered elementwise absolute contract. The ×10⁶ panel is display-only and never used for metrics.",
        ha="center",
        va="center",
        fontsize=5.8,
    )
    image_cbar_ax = figure.add_axes([0.94, 0.49, 0.014, 0.25])
    diff_cbar_ax = figure.add_axes([0.94, 0.17, 0.014, 0.25])
    image_cbar = figure.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="gray"), cax=image_cbar_ax)
    diff_cbar = figure.colorbar(ScalarMappable(norm=Normalize(0.0, FULL_PATH_F32_DIFF_RANGE[1]), cmap="magma"), cax=diff_cbar_ax)
    image_cbar.set_label("image [0,1]", fontsize=5.5)
    diff_cbar.set_label("raw |Δ| [0,2e−4]", fontsize=5.5)
    for cbar in (image_cbar, diff_cbar):
        cbar.ax.tick_params(labelsize=5, width=0.5, length=2)
    return _save_figure_bundle(
        figure,
        output_base,
        axes=[axis for row in axes for axis in row],
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        colorbar_axes=[image_cbar.ax, diff_cbar.ax],
        require_panel_labels=False,
        raster_dpi=300,
        include_tiff=False,
    )


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
    parser.add_argument("--visualization-config", type=Path, default=FULL_PATH_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--sample-index", type=int, default=55008, help="confirmation-panel train index; default is a held-out digit 3")
    parser.add_argument("--view", choices=("single", "fiber-chain", "full-path-reconstruction"), default="single", help="single legacy plate, encoder/fiber/decoder plate, or the S9V full-path/reconstruction checkpoint")
    parser.add_argument("--checkpoint-step", type=int, choices=CHECKPOINT_STEPS, default=5000)
    parser.add_argument("--hero-index", type=int, default=55008, help="fixed confirmation-panel index used by the fiber cutaway")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-dirty-path",
        action="append",
        default=[],
        help="repo-relative dirty path allowed by the visualization preflight; repeat for each path",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _default_output_dir(view: str) -> Path:
    if view == "full-path-reconstruction":
        return FULL_PATH_OUTPUT
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
    if torch.is_complex(values["slm_command"]) or torch.is_complex(values["cmos_detector_intensity"]):
        raise RuntimeError("SLM command and CMOS detector readout must remain real-valued")
    if tuple(values["slm_command"].shape) != tuple(amplitude.shape):
        raise RuntimeError("SLM command shape no longer matches the registered amplitude input")
    if tuple(values["cmos_detector_intensity"].shape) != tuple(camera_intensity.shape):
        raise RuntimeError("CMOS detector intensity shape no longer matches the camera readout")

    values_cpu = {key: value.detach().cpu() for key, value in values.items()}
    views, spatial_scale, modal_scale = _complex_views(values_cpu, sample_positions=range(10))
    command_scale = _shared_p995(
        [_display_array(values_cpu["slm_command"], position) for position in range(10)]
    )
    detector_scale = _shared_p995(
        [_display_array(values_cpu["cmos_detector_intensity"], position) for position in range(10)]
    )
    metrics = diagnostics["metrics"]
    metric_json = _json_metrics(metrics)
    git_state = _git_state()
    script_path = Path(__file__).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    saved: list[Path] = []
    saved += _render_real_reference_plate(
        values_cpu,
        output_dir / f"00_digits_0_4_slm_amplitude_command_reference_step{args.checkpoint_step}",
        positions=range(5),
        labels=labels[:5].tolist(),
        sample_indices=sample_indices[:5],
        value_key="slm_command",
        title=f"Scaled-Chen step-{args.checkpoint_step} registered input reference | SLM amplitude command",
        top_label="This is the real-valued amplitude command sent to the ideal SLM; it is the stage that should retain the MNIST digit silhouette.",
        unavailable_phase_label="phase N/A — real amplitude command; no independent phase channel",
        scale_label="shared command amplitude / p99.5",
        display_scale=command_scale,
    )
    saved += _render_real_reference_plate(
        values_cpu,
        output_dir / f"00_digits_5_9_slm_amplitude_command_reference_step{args.checkpoint_step}",
        positions=range(5, 10),
        labels=labels[5:].tolist(),
        sample_indices=sample_indices[5:],
        value_key="slm_command",
        title=f"Scaled-Chen step-{args.checkpoint_step} registered input reference | SLM amplitude command | digits 5–9",
        top_label="This is the real-valued amplitude command sent to the ideal SLM; it is the stage that should retain the MNIST digit silhouette.",
        unavailable_phase_label="phase N/A — real amplitude command; no independent phase channel",
        scale_label="shared command amplitude / p99.5",
        display_scale=command_scale,
    )
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
    saved += _render_real_reference_plate(
        values_cpu,
        output_dir / f"06_digits_0_4_cmos_detector_intensity_step{args.checkpoint_step}",
        positions=range(5),
        labels=labels[:5].tolist(),
        sample_indices=sample_indices[:5],
        value_key="cmos_detector_intensity",
        title=f"Scaled-Chen step-{args.checkpoint_step} CMOS detector readout",
        top_label="The detector panel is the predicted intensity readout I=|E|² at the camera plane; it is not a complex-field measurement.",
        unavailable_phase_label="phase N/A — square-law CMOS readout; see the main plate for the modeled upstream incident field",
        scale_label="detector intensity / p99.5",
        display_scale=detector_scale,
    )
    saved += _render_real_reference_plate(
        values_cpu,
        output_dir / f"06_digits_5_9_cmos_detector_intensity_step{args.checkpoint_step}",
        positions=range(5, 10),
        labels=labels[5:].tolist(),
        sample_indices=sample_indices[5:],
        value_key="cmos_detector_intensity",
        title=f"Scaled-Chen step-{args.checkpoint_step} CMOS detector readout | digits 5–9",
        top_label="The detector panel is the predicted intensity readout I=|E|² at the camera plane; it is not a complex-field measurement.",
        unavailable_phase_label="phase N/A — square-law CMOS readout; see the main plate for the modeled upstream incident field",
        scale_label="detector intensity / p99.5",
        display_scale=detector_scale,
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
        "sample_correspondence": {
            "selection_rule": "registered confirmation panel first occurrence per label in train indices 55000:60000",
            "labels_verified": True,
            "batch_order": [
                {"position": position, "index": sample_indices[position], "label": int(labels[position])}
                for position in range(10)
            ],
            "expected_labels": list(range(10)),
            "no_reindex_after_replay": True,
        },
        "stage_order": list(FIBER_STAGE_KEYS),
        "stage_labels": list(FIBER_STAGE_LABELS),
        "stage_shapes": {key: list(value.shape) for key, value in values_cpu.items()},
        "stage_availability": {
            **FIBER_STAGE_AVAILABILITY,
            "slm_command": {
                "representation": "real_amplitude_command",
                "amplitude_available": True,
                "phase_available": False,
                "phase_note": "IdealAmplitudeSLM command is real-valued; no independent phase channel is measured.",
            },
            "cmos_detector_intensity": {
                "representation": "intensity_frame",
                "amplitude_available": False,
                "phase_available": False,
                "intensity_available": True,
                "formula": "I=|E|^2",
            },
        },
        "stage_metadata": diagnostics["stage_metadata"],
        "stage_aliases": {
            "encoder_layer_2_equals_fiber_input": True,
            "fiber_output_equals_decoder_input": True,
            "note": "The slash/equality labels identify shared boundary tensors; they are not independent measurements.",
        },
        "spatial_amplitude_shared_p995": spatial_scale,
        "modal_amplitude_shared_p995": modal_scale,
        "slm_command_amplitude_shared_p995": command_scale,
        "cmos_detector_intensity_shared_p995": detector_scale,
        "display_contract": {
            "spatial_amplitude": "absolute value, one shared linear p99.5 scale across all ten digits and spatial stages, clipped to [0,1]",
            "modal_amplitude": "absolute value, one shared modal p99.5 scale for c_in and c_out, clipped to [0,1]",
            "phase": "angle in [-pi,pi], masked below 1% of each field maximum",
            "real_input": "SLM amplitude command is plotted as a real-valued amplitude with an explicit phase N/A panel; it is not treated as a complex measurement",
            "detector": "CMOS detector readout is plotted as intensity I=|E|^2 with an explicit phase N/A panel; the modeled upstream incident field remains in the main chain plate",
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


def _run_full_path(args: argparse.Namespace, output_dir: Path) -> int:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-path visualization requires CUDA; CPU is not a substitute")
    _check_output_boundary(output_dir)
    contract, _review, config_path, review_path = _load_full_path_contract(args.visualization_config)
    previous_attempt = _verify_full_path_previous_attempt(contract)
    git_state = _git_state()
    if (
        git_state["root"] != str(_repo_path("."))
        or git_state["branch"] != contract["branch"]
        or git_state["commit"] != contract["expected_head"]
    ):
        raise RuntimeError("full-path visualization Git identity is mismatched")
    allowed_dirty_paths = _verify_allowed_dirty_paths(
        git_state["status"], args.allow_dirty_path
    )

    device = torch.device(args.device)
    authority = _load_full_path_authority(contract, args.dataset_root, device)
    train_images, train_labels = base.load_local_mnist(args.dataset_root, train=True)
    images, labels, fixed_indices = _select_confirmation_digits(train_images, train_labels)
    expected_indices = {
        digit: int(contract["sample_panel"]["indices_by_label"][str(digit)])
        for digit in range(10)
    }
    if fixed_indices != expected_indices or labels.tolist() != list(range(10)):
        raise RuntimeError("locked visualization sample panel does not match the approved contract")
    locked_panel = tuple((int(label), int(fixed_indices[int(label)])) for label in range(10))
    rendered = base.render_intensity_batch(images)
    amplitude = rendered.amplitude.to(device)
    target = rendered.target.to(device)
    model = authority["model"]
    model_state_before = _module_state_digest(model)
    ols_state_before = _module_state_digest(model.ols)
    with torch.no_grad():
        values, diagnostics, camera_intensity, consistency = _trace_fiber_batch(model, amplitude, target)
    model_state_after = _module_state_digest(model)
    ols_state_after = _module_state_digest(model.ols)
    if model_state_before != model_state_after or ols_state_before != ols_state_after:
        raise RuntimeError("full-path visualization changed frozen model or OLS state")
    if consistency > 1.0e-7:
        raise RuntimeError(f"full-path forward consistency exceeded contract: {consistency}")
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise RuntimeError("full-path visualization contains a non-finite stage")
    if not torch.equal(values["cmos_detector_intensity"], camera_intensity):
        raise RuntimeError("CMOS intensity stage is not the explicit traced detector readout")

    values_cpu = {key: value.detach().cpu() for key, value in values.items()}
    target_cpu = target.detach().cpu().float()
    stage_values = {
        "slm_output": values_cpu["slm_output"],
        "encoder_input": values_cpu["encoder_input"],
        "encoder_layer_1": values_cpu["encoder_layer_1"],
        "pre_ols": values_cpu["fiber_input"],
        "post_ols": values_cpu["fiber_output"],
        "decoder_layer_1": values_cpu["decoder_layer_1"],
        "decoder_layer_2": values_cpu["decoder_layer_2"],
        "cmos_field": values_cpu["cmos_field"],
    }
    views, scales = _full_path_views(stage_values, sample_positions=range(10))
    predictions, reconstruction_metrics = _replay_full_path_reconstructions(
        authority, values_cpu, images, labels, target_cpu
    )
    label_to_position = {int(label): position for position, label in enumerate(labels.tolist())}
    detail_positions = tuple(label_to_position[label] for label in FULL_PATH_DETAIL_LABELS)
    detail_indices = tuple(int(fixed_indices[label]) for label in FULL_PATH_DETAIL_LABELS)

    output_dir.mkdir(parents=True, exist_ok=False)
    saved: list[Path] = []
    for page_digits in FULL_PATH_PAGE_DIGITS:
        page_positions = tuple(label_to_position[digit] for digit in page_digits)
        page_label = f"digits {page_digits[0]}–{page_digits[-1]}"
        saved += _render_full_path_plate(
            views,
            output_dir / f"01_full_optical_path_digits_{page_digits[0]}_{page_digits[-1]}",
            positions=page_positions,
            labels=page_digits,
            sample_indices=tuple(int(fixed_indices[digit]) for digit in page_digits),
            page_label=page_label,
        )
    saved += _render_reconstruction_grid(
        predictions,
        output_dir / "02_reconstruction_family_comparison",
        positions=range(10),
        labels=labels.tolist(),
        sample_indices=tuple(int(fixed_indices[int(label)]) for label in labels.tolist()),
        method_keys=FULL_PATH_RECON_METHODS,
        method_mse=reconstruction_metrics["method_mse"],
    )
    saved += _render_repair_detail(
        predictions,
        target_cpu,
        output_dir / "03_repair_detail_and_error_maps",
        positions=detail_positions,
        labels=FULL_PATH_DETAIL_LABELS,
        sample_indices=detail_indices,
    )
    saved += _render_f64_f32_detail(
        predictions,
        output_dir / "04_mpcdar_rank16_f64_f32_difference",
        positions=detail_positions,
        labels=FULL_PATH_DETAIL_LABELS,
        sample_indices=detail_indices,
    )

    raw_stage_metadata = diagnostics["stage_metadata"]
    stage_metadata = {
        "slm_output": raw_stage_metadata["slm_output"],
        "encoder_input": raw_stage_metadata["encoder_input"],
        "encoder_layer_1": raw_stage_metadata["encoder_layer_1"],
        "pre_ols": raw_stage_metadata["fiber_input"] | {
            "display_name": "pre_ols / encoder_layer_2 / fiber_input",
        },
        "modal_complex": raw_stage_metadata["port_output"] | {
            "display_name": "modal coefficients — 10×10 display layout",
            "spatial_plane": False,
        },
        "post_ols": raw_stage_metadata["fiber_output"] | {
            "display_name": "post_ols / decoder_input",
        },
        "decoder_layer_1": raw_stage_metadata["decoder_layer_1"],
        "decoder_layer_2": raw_stage_metadata["decoder_layer_2"],
        "cmos_field": raw_stage_metadata["cmos_field"],
    }
    auxiliary_stage_metadata = {
        "target": {
            "representation": "intensity_frame",
            "semantic": "camera_intensity_same_fov",
            "shape": [1, 200, 200],
            "display_range": list(FULL_PATH_RECON_RANGE),
        },
        "slm_command": raw_stage_metadata["slm_command"],
        "modal_complex": raw_stage_metadata["port_output"] | {
            "display_name": "modal coefficients — 10×10 display layout",
            "spatial_plane": False,
        },
        "cmos_detector_intensity": raw_stage_metadata["cmos_detector_intensity"],
    }
    stage_stats: dict[str, Any] = {}
    for key in FULL_PATH_STAGE_KEYS:
        value = stage_values[key]
        stage_stats[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "complex": bool(torch.is_complex(value)),
            "display_scale": scales[key],
            "phase_thresholds": [
                views[key][str(position)]["phase_threshold"]
                for position in range(10)
            ],
            "phase_valid_fractions": [
                views[key][str(position)]["phase_valid_fraction"]
                for position in range(10)
            ],
        }
    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    output_bytes = sum(path.stat().st_size for path in output_files)
    if output_bytes > 512 * 1024**2:
        raise RuntimeError(f"full-path visualization output exceeds 512 MiB: {output_bytes}")
    parent_contract = contract["parents"]
    artifact_identities = {
        "s8_float64": {
            "manifest": str(_repo_path(parent_contract["s8"]["artifact_manifest"])),
            "manifest_sha256": parent_contract["s8"]["artifact_manifest_sha256"],
            "payload": str(_repo_path(parent_contract["s8"]["artifact_payload"])),
            "payload_sha256": parent_contract["s8"]["artifact_payload_sha256"],
            "dtype": "torch.float64",
            "rank": FULL_PATH_RANK,
        },
        "s9_float32": {
            "manifest": str(_repo_path(parent_contract["s9"]["artifact_manifest"])),
            "manifest_sha256": parent_contract["s9"]["artifact_manifest_sha256"],
            "payload": str(_repo_path(parent_contract["s9"]["artifact_payload"])),
            "payload_sha256": parent_contract["s9"]["artifact_payload_sha256"],
            "dtype": "torch.float32",
            "rank": FULL_PATH_RANK,
        },
    }
    manifest = {
        "schema": "lightgen_optical_manifold_smf_bogt_full_path_visualization.v2",
        "task_id": FULL_PATH_TASK,
        "status": "completed",
        "classification": FULL_PATH_CLASSIFICATION,
        "diagnostic_only": True,
        "layout_schema": FULL_PATH_LAYOUT_SCHEMA,
        "supersedes": "attempt_01",
        "supersedes_manifest": previous_attempt,
        "cuda_replay": True,
        "optical_data_changed": False,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "review_path": str(review_path),
        "review_sha256": file_sha256(review_path),
        "git": git_state,
        "preflight": {
            "expected_branch": contract["branch"],
            "expected_head": contract["expected_head"],
            "starting_worktree_clean": git_state["clean"],
            "dirty_path_policy": "only explicitly allowed repository paths may be dirty",
            "allowed_dirty_paths": allowed_dirty_paths,
            "unexpected_dirty_paths": [],
        },
        "parent_evidence": authority["parent_files"],
        "rmal": {
            "config": str(_repo_path(contract["rmal"]["config"])),
            "config_sha256": contract["rmal"]["config_sha256"],
            "checkpoint": authority["rmal_checkpoint"],
            "checkpoint_sha256": contract["rmal"]["checkpoint_sha256"],
            "probe_hash": str(authority["upstream"]["runtime"].rmal_probe.artifact_sha256),
            "target": contract["rmal"]["target"],
            "target_shape": contract["rmal"]["target_shape"],
        },
        "artifacts": artifact_identities,
        "sample_panel": {
            "source": contract["sample_panel"]["source"],
            "count": 10,
            "order": list(range(10)),
            "indices_by_label": {str(label): index for label, index in locked_panel},
            "labels": labels.tolist(),
            "locked_before_prediction_replay": True,
            "error_selection": False,
            "random_sampling": False,
            "detail_labels": list(FULL_PATH_DETAIL_LABELS),
            "detail_indices": list(detail_indices),
        },
        "stage_order": list(FULL_PATH_STAGE_KEYS),
        "stage_aliases": FULL_PATH_STAGE_ALIASES,
        "stage_metadata": stage_metadata,
        "auxiliary_trace_stages": auxiliary_stage_metadata,
        "stage_stats": stage_stats,
        "display_contract": {
            "spatial_amplitude": "abs(E), one panel-wide linear p99.5 scale per named stage over all ten samples",
            "phase": "angle(E), raw, fixed [-pi,pi], no shift/unwrap/gauge normalization",
            "phase_mask": "max(0.01 * max(abs(E)), 1e-12), per stage and sample; masked values are NaN/explicit N/A",
            "reconstruction_range": list(FULL_PATH_RECON_RANGE),
            "target_error_range": list(FULL_PATH_ERROR_RANGE),
            "f64_f32_difference_range": list(FULL_PATH_F32_DIFF_RANGE),
            "f64_f32_amplification": FULL_PATH_F32_DIFF_AMPLIFICATION,
            "amplitude_colormap": "magma",
            "phase_colormap": "twilight",
            "reconstruction_colormap": "gray",
            "error_colormap": "magma",
            "layout_schema": FULL_PATH_LAYOUT_SCHEMA,
            "display_rotation": "upright_common_object_frame_v1",
            "display_rotation_keys": list(FULL_PATH_DISPLAY_ROTATED_ORDER),
            "native_coordinate_keys": list(FULL_PATH_NATIVE_DISPLAY_KEYS),
            "front_relay_magnification": -0.4,
            "back_relay_magnification": -2.5,
            "net_magnification": 1.0,
            "rotation": "display-only 180 degrees for named intermediate optical maps; no raw tensor or reconstruction rotation",
        },
        "reconstruction": {
            "methods": list(FULL_PATH_RECON_METHODS),
            "ranks": {"raw_svd": FULL_PATH_RANK, "mean_preserving_raw_svd": FULL_PATH_RANK, "mpcdar_f64": FULL_PATH_RANK, "mpcdar_f32": FULL_PATH_RANK, "full_rmal": FULL_PATH_FULL_RANK},
            "feature_contract": reconstruction_metrics["feature_contract"],
            "reshape": reconstruction_metrics["reshape"],
            "display_range": list(FULL_PATH_RECON_RANGE),
            "error_range": list(FULL_PATH_ERROR_RANGE),
            "method_mse_descriptive_panel": reconstruction_metrics["method_mse"],
            "per_sample_mse_descriptive_panel": reconstruction_metrics["per_sample_mse"],
        },
        "forward_consistency_max_abs": consistency,
        "state_hashes": {
            "model_before": model_state_before,
            "model_after": model_state_after,
            "ols_before": ols_state_before,
            "ols_after": ols_state_after,
            "unchanged": True,
        },
        "renderer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "dtype": "frozen optical model and transient CPU display copies; F64/F32 readouts preserve artifact dtype",
            "device": str(device),
        },
        "authorization": {
            "no_training": True,
            "no_update": True,
            "no_optimizer": True,
            "no_backward": True,
            "no_new_basis": True,
            "no_rank_search": True,
            "visualization_is_scientific_gate": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation": False,
        },
        "outputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
        "output_bytes_before_manifest": output_bytes,
    }
    metadata_path = output_dir / "visualization_manifest.json"
    if metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite visualization manifest: {metadata_path}")
    metadata_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "manifest": str(metadata_path.resolve()), "classification": FULL_PATH_CLASSIFICATION, "sample_count": 10, "output_bytes": output_bytes}, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir(args.view)
    if args.view == "full-path-reconstruction":
        return _run_full_path(args, output_dir)
    if args.view == "fiber-chain":
        return _run_fiber_chain(args, output_dir)
    _check_output_boundary(output_dir)
    return _run_single(args, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
