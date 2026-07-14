from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch


STRICT_OPTICAL_FIELD_SCHEMA_VERSION = 1
PHASE_INTENSITY_FLOOR_RATIO = 1.0e-6  # -60 dB relative intensity


_FIELD_LABELS = {
    "slm_output": "SLM output",
    "pre_encoder_relay_image": "pre-encoder relay image",
    "encoder_input": "encoder input",
    "encoder_grid": "encoder grid",
    "encoder_output": "encoder output",
    "ols_input": "OLS input",
    "ols_output": "OLS output",
    "decoder_input": "decoder input",
    "decoder_output": "decoder output",
    "chip": "chip",
    "camera_field": "camera field",
}


def _field_label(name: str) -> str:
    if name in _FIELD_LABELS:
        return _FIELD_LABELS[name]
    if name.startswith("encoder_layer_"):
        return f"encoder field after L{name.rsplit('_', maxsplit=1)[-1]}"
    if name.startswith("decoder_layer_"):
        return f"decoder field after L{name.rsplit('_', maxsplit=1)[-1]}"
    return name.replace("_", " ")


def build_optical_field_record(
    field: torch.Tensor,
    *,
    upstream: str | None,
    operation: str,
    phase_available: bool,
    source_kind: str = "propagated_complex_field",
    sampling: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Record an available model complex field without deriving display tensors."""
    if not torch.is_complex(field):
        raise ValueError("strict optical-field records require a complex tensor")
    return {
        "real": field.real,
        "imag": field.imag,
        "shape": list(field.shape),
        "upstream": upstream,
        "operation": operation,
        "phase_available": bool(phase_available),
        "source_kind": source_kind,
        "sampling": dict(sampling) if sampling is not None else None,
    }


def detach_optical_fields(fields: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    detached: dict[str, dict[str, object]] = {}
    for name, record in fields.items():
        detached[name] = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in record.items()
        }
    return detached


def _record_field(name: str, record: Mapping[str, object]) -> torch.Tensor:
    real = record.get("real")
    imag = record.get("imag")
    if not torch.is_tensor(real) or not torch.is_tensor(imag):
        raise ValueError(f"optical field {name!r} is missing real/imag tensors")
    if real.shape != imag.shape:
        raise ValueError(f"optical field {name!r} real/imag shapes differ: {tuple(real.shape)} vs {tuple(imag.shape)}")
    if not torch.isfinite(real).all() or not torch.isfinite(imag).all():
        raise ValueError(f"optical field {name!r} contains non-finite values")
    return torch.complex(real, imag)


def _allclose_record_pair(
    fields: Mapping[str, Mapping[str, object]],
    left: str,
    right: str,
) -> bool:
    return torch.allclose(_record_field(left, fields[left]), _record_field(right, fields[right]), rtol=0.0, atol=0.0)


def _pre_intervention_planes(fields: Mapping[str, Mapping[str, object]]) -> list[str]:
    names = list(fields)
    if "decoder_input" not in fields:
        return names
    return names[: names.index("decoder_input")]


def validate_optical_field_chain(
    fields: Mapping[str, Mapping[str, object]],
    *,
    front_path_model: str,
    target_code_active: bool,
    bridge_active: bool,
) -> dict[str, object]:
    """Validate provenance before a field can be called a continuous optical chain."""
    malformed: list[str] = []
    if not fields:
        malformed.append("no optical field records were captured")
    for name, record in fields.items():
        try:
            _record_field(name, record)
        except ValueError as exc:
            malformed.append(str(exc))
        sampling = record.get("sampling")
        if sampling is not None:
            shape = sampling.get("shape") if isinstance(sampling, Mapping) else None
            if shape is None or tuple(shape) != tuple(record["real"].shape[-2:]):
                malformed.append(f"optical field {name!r} sampling shape does not match tensor shape")
            for key in ("pixel_size_x", "pixel_size_y", "fov_x", "fov_y"):
                if not isinstance(sampling, Mapping) or not isinstance(sampling.get(key), (int, float)) or sampling[key] <= 0:
                    malformed.append(f"optical field {name!r} has invalid sampling {key}")
        upstream = record.get("upstream")
        if upstream in fields and isinstance(sampling, Mapping):
            upstream_sampling = fields[upstream].get("sampling")
            if isinstance(upstream_sampling, Mapping):
                same_grid = (
                    sampling.get("shape") == upstream_sampling.get("shape")
                    and sampling.get("pixel_size_x") == upstream_sampling.get("pixel_size_x")
                    and sampling.get("pixel_size_y") == upstream_sampling.get("pixel_size_y")
                )
                if not same_grid and record.get("operation") not in {
                    "physical_aperture_sampling",
                    "macro_free_space",
                    "free_space",
                    "scaled_imaging",
                    "coherent_camera_propagation",
                    "coherent_imaging_4f",
                }:
                    malformed.append(
                        f"optical field {name!r} changes physical sampling without an explicit physical mapping: "
                        f"{record.get('operation')!r}"
                    )

    reasons: list[str] = []
    if front_path_model == "relay_intensity_psf":
        reasons.append("intensity-only-front-relay")
    if target_code_active:
        reasons.append("target-code-intervention")
    if bridge_active:
        reasons.append("target-code-bridge-intervention")

    if front_path_model != "relay_intensity_psf" and not target_code_active and not bridge_active:
        required = ("slm_output", "encoder_output", "ols_input", "ols_output", "decoder_input", "chip")
        missing = [name for name in required if name not in fields]
        if missing:
            malformed.append(f"missing required continuous-chain fields: {missing}")

    for left, right, reason in (
        ("encoder_output", "ols_input", "encoder-output-to-ols-input-discontinuity"),
        ("ols_output", "decoder_input", "ols-output-to-decoder-input-discontinuity"),
    ):
        if left in fields and right in fields and not _allclose_record_pair(fields, left, right):
            reasons.append(reason)

    if not reasons:
        unavailable = [name for name, record in fields.items() if not bool(record.get("phase_available", False))]
        unavailable = [name for name in unavailable if name != "camera_field"]
        if unavailable:
            reasons.extend(f"phase-unavailable:{name}" for name in unavailable)

    if malformed:
        status = "harness-invalid"
    elif reasons:
        status = "field-chain-invalid"
    else:
        status = "valid"
    full_chain_available = status == "valid"
    render_plane_names = list(fields) if full_chain_available else _pre_intervention_planes(fields)
    render_plane_names = [
        name for name in render_plane_names if bool(fields[name].get("phase_available", False))
    ]
    return {
        "schema_version": STRICT_OPTICAL_FIELD_SCHEMA_VERSION,
        "status": status,
        "full_chain_available": full_chain_available,
        "reasons": reasons,
        "errors": malformed,
        "render_plane_names": render_plane_names,
        "phase_intensity_floor_ratio": PHASE_INTENSITY_FLOOR_RATIO,
        "amplitude_units": "simulation_field_a.u.",
    }


def _phase_validity_mask(amplitude: torch.Tensor) -> torch.Tensor:
    intensity = amplitude.square()
    peak = intensity.flatten(start_dim=1).amax(dim=1).reshape(-1, *([1] * (intensity.ndim - 1)))
    return intensity >= peak * PHASE_INTENSITY_FLOOR_RATIO


def build_strict_optical_field_views(
    fields: Mapping[str, Mapping[str, object]],
    chain: Mapping[str, object],
) -> dict[str, object]:
    """Derive raw |E| and arg(E) views directly from validated real/imag records."""
    amplitude: dict[str, torch.Tensor] = {}
    phase: dict[str, torch.Tensor] = {}
    phase_validity: dict[str, torch.Tensor] = {}
    sampling: dict[str, Mapping[str, object] | None] = {}
    for name in chain["render_plane_names"]:
        field = _record_field(name, fields[name])
        field_amplitude = field.abs()
        validity = _phase_validity_mask(field_amplitude)
        label = _field_label(name)
        amplitude[f"{label} |E|"] = field_amplitude
        phase[f"{label} phase [rad]"] = torch.where(validity, torch.angle(field), torch.full_like(field_amplitude, float("nan")))
        phase_validity[label] = validity
        sampling[label] = fields[name].get("sampling")
    amplitude_vmax = max((float(value.detach().amax().item()) for value in amplitude.values()), default=1.0)
    return {
        "amplitude": amplitude,
        "phase": phase,
        "phase_validity": phase_validity,
        "sampling": sampling,
        "amplitude_vmax": max(amplitude_vmax, 1.0e-12),
        "phase_limits_rad": [-math.pi, math.pi],
        "status": chain["status"],
    }


def save_strict_optical_field_grids(
    run_manager,
    views: Mapping[str, object],
    destination: str | Path,
    *,
    title_prefix: str,
    max_items: int,
    row_labels: list[str] | tuple[str, ...] | None,
    common_scale: bool = False,
) -> dict[str, Path]:
    """Save raw-scale amplitude and radian-phase grids without display normalization."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for strict optical-field visualization") from exc

    base = Path(destination)
    labels = list(row_labels or [])
    saved: dict[str, Path] = {}
    for kind, cmap in (("amplitude", "gray"), ("phase", "twilight")):
        columns = views[kind]
        if not columns:
            continue
        images_by_column = {
            name: value.detach().float().cpu()[:max_items, 0].numpy()
            for name, value in columns.items()
        }
        max_fov_um = max(
            [
                float(sampling["fov_x"]) * 1.0e6
                for label, sampling in views.get("sampling", {}).items()
                if label in images_by_column and isinstance(sampling, Mapping)
            ]
            or [1.0]
        )
        rows = max(image.shape[0] for image in images_by_column.values())
        figure, axes = plt.subplots(
            rows,
            len(images_by_column),
            squeeze=False,
            figsize=(1.8 * len(images_by_column), 1.45 * rows + 0.9),
            constrained_layout=True,
        )
        for column_index, (label, images) in enumerate(images_by_column.items()):
            sampling = views.get("sampling", {}).get(label)
            if isinstance(sampling, Mapping):
                shape = sampling.get("shape", [images.shape[-2], images.shape[-1]])
                dx_um = float(sampling.get("pixel_size_x", 0.0)) * 1.0e6
                fov_um = float(sampling.get("fov_x", 0.0)) * 1.0e6
                title = f"{label}\n{shape[1]}×{shape[0]}, dx={dx_um:g} µm, FOV={fov_um:g} µm"
            else:
                title = label
            axes[0][column_index].set_title(title, fontsize=8)
            for row_index in range(rows):
                axis = axes[row_index][column_index]
                axis.set_xticks([])
                axis.set_yticks([])
                if column_index == 0 and row_index < len(labels):
                    axis.set_ylabel(labels[row_index], rotation=0, labelpad=10, fontsize=8)
                if row_index >= images.shape[0]:
                    axis.axis("off")
                    continue
                if isinstance(sampling, Mapping):
                    fov_x = float(sampling["fov_x"]) * 1.0e6
                    fov_y = float(sampling["fov_y"]) * 1.0e6
                    extent = (-fov_x / 2.0, fov_x / 2.0, -fov_y / 2.0, fov_y / 2.0)
                else:
                    extent = None
                if kind == "amplitude":
                    axis.imshow(images[row_index], cmap=cmap, vmin=0.0, vmax=float(views["amplitude_vmax"]), interpolation="nearest", extent=extent)
                else:
                    axis.imshow(
                        images[row_index],
                        cmap=cmap,
                        vmin=-math.pi,
                        vmax=math.pi,
                        interpolation="nearest",
                        extent=extent,
                    )
                if common_scale:
                    axis.set_xlim(-max_fov_um / 2.0, max_fov_um / 2.0)
                    axis.set_ylim(-max_fov_um / 2.0, max_fov_um / 2.0)
                    axis.set_xlabel("µm", fontsize=6)
                    axis.set_ylabel("µm", fontsize=6)
        figure.suptitle(
            f"{title_prefix} {'|E| [simulation a.u., shared linear scale]' if kind == 'amplitude' else 'arg(E) [rad, valid above -60 dB]'}",
            fontsize=10,
        )
        if kind == "amplitude":
            scalar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=float(views["amplitude_vmax"])))
            figure.colorbar(scalar, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.75, pad=0.03, label="|E| [simulation a.u.]")
        else:
            scalar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-math.pi, vmax=math.pi))
            colorbar = figure.colorbar(scalar, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.75, pad=0.03)
            colorbar.set_ticks((-math.pi, 0.0, math.pi))
            colorbar.set_ticklabels(("−π", "0", "π"))
            colorbar.set_label("arg(E) [rad]")
        suffix = f"_common_scale_{kind}" if common_scale else f"_{kind}"
        path = base.with_name(f"{base.stem}{suffix}{base.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=120)
        plt.close(figure)
        saved[kind] = path
    return saved
