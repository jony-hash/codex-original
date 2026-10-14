"""Build shared-scale 10-node amplitude/phase figures and a PDF atlas."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from scripts.research.lightgen import capture_lightgen_best_endpoints_10node_v1 as run_capture
from scripts.research.lightgen import visualize_lightgen_architecture_comparison_v1 as capture


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = run_capture.OUTPUT
FIGURES = OUTPUT / "figures"
FONT_FAMILY = "Microsoft YaHei"
PHASE_CMAP = "twilight"
AMPLITUDE_CMAP = "magma"
SMALL_NAME = run_capture.SMALL_NAME
LARGE_NAME = run_capture.LARGE_NAME
NODE_MAP = run_capture.DISPLAY_NODE_MAP
PRESENTATION_SAMPLES = (1, 7)  # first saved class-1 and class-7 rows
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 8,
})


def _read_json(path: Path) -> dict[str, Any]:
    capture.require(path.is_file(), f"required visualization input is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    capture.require(isinstance(result, dict), f"expected an object in {path}")
    return result


def _write_result_json(path: Path, value: dict[str, Any], *, replace_existing: bool) -> None:
    if path.exists():
        if _read_json(path) == value:
            return
        capture.require(replace_existing, f"refusing to overwrite an existing result manifest: {path}")
        temporary = path.with_name(f"{path.stem}.rebuild{path.suffix}")
        capture.require(not temporary.exists(), f"refusing to overwrite temporary result manifest: {temporary}")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return
    capture._write_json(path, value)


def _require_physical_coordinates(metadata: dict[str, Any], context: str) -> None:
    required = ("coordinate_frame", "plane_id", "z_m", "origin_m", "pixel_pitch_m", "plane_shape", "fov_m", "wavelength_m", "origin_convention", "pixel_center_formula")
    missing = [name for name in required if metadata.get(name) is None]
    capture.require(not missing, f"{context}: physical coordinates are incomplete: {missing}")
    capture.require(len(metadata["origin_m"]) == len(metadata["pixel_pitch_m"]) == len(metadata["plane_shape"]) == len(metadata["fov_m"]) == 2, f"{context}: physical coordinate dimensions are invalid")
    expected_fov = [float(metadata["plane_shape"][axis]) * float(metadata["pixel_pitch_m"][axis]) for axis in range(2)]
    capture.require(all(math.isclose(float(got), expected, rel_tol=1.0e-12, abs_tol=1.0e-15) for got, expected in zip(metadata["fov_m"], expected_fov, strict=True)), f"{context}: FOV does not match native shape and pixel pitch")


def _resolve_small_coordinates(metadata: dict[str, Any], source: dict[str, Any], node: str) -> dict[str, Any]:
    capture.require(metadata.get("field_kind") != "complex_ols_coefficients", f"{node}: OLS port coordinates must not be resolved as a metric plane")
    compare_keys = ("coordinate_frame", "plane_id", "pixel_pitch_m", "plane_shape", "wavelength_m", "fov_m")
    for key in compare_keys:
        if metadata.get(key) is not None:
            capture.require(metadata[key] == source.get(key), f"{node}: saved A1000 {key} differs from authoritative coordinate metadata")
    if metadata.get("z_m") is not None:
        capture.require(math.isclose(float(metadata["z_m"]), float(source["z_m"]), rel_tol=0.0, abs_tol=1.0e-12), f"{node}: saved A1000 axial coordinate differs from authoritative coordinate metadata")
    if metadata.get("origin_m") is not None:
        capture.require(metadata["origin_m"] == source.get("origin_m"), f"{node}: saved A1000 origin differs from authoritative coordinate metadata")
    resolved = {**source, **metadata}
    for key in ("origin_m", "pixel_pitch_m", "plane_shape", "fov_m", "coordinate_frame", "plane_id", "wavelength_m", "origin_convention", "pixel_center_formula", "z_m"):
        resolved[key] = source[key]
    _require_physical_coordinates(resolved, f"{SMALL_NAME}/{node}")
    return resolved


def _complete_capture_coordinate_convention(metadata: dict[str, Any], convention: dict[str, Any], node: str) -> dict[str, Any]:
    origin = convention.get("origin_m")
    capture.require(origin is not None and metadata.get("origin_m") == origin, f"{LARGE_NAME}/{node}: captured origin differs from the documented PlaneSpec convention")
    resolved = dict(metadata)
    resolved["origin_convention"] = convention["origin_semantics"]
    resolved["pixel_center_formula"] = convention["pixel_center_formula"]
    _require_physical_coordinates(resolved, f"{LARGE_NAME}/{node}")
    return resolved


def _load_samples() -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, Any]]]]:
    selection = _read_json(OUTPUT / "sample_selection.json")
    manifest = _read_json(OUTPUT / "field_capture_manifest.json")
    small_coordinate_source = _read_json(run_capture.OLD_OUTPUT / "field_coordinate_metadata.json")
    small_capture_source = _read_json(run_capture.OLD_OUTPUT / "field_capture_manifest.json")
    capture.require(small_coordinate_source.get("coordinate_system") == "PlaneSpec physical grid; no field interpolation or resampling", "A1000 coordinate source does not document the required physical grid convention")
    source_coordinates = small_coordinate_source.get("endpoints", {}).get(SMALL_NAME, {})
    coordinate_convention = small_capture_source.get("coordinate_convention", {})
    capture.require(coordinate_convention.get("origin_m") == [0.0, 0.0] and coordinate_convention.get("pixel_center_formula"), "authoritative PlaneSpec coordinate convention is incomplete")
    selected_rows = list(map(int, selection["selected_source_rows"]))
    labels = list(map(int, selection["selected_labels"]))
    source_positions = list(map(int, selection["selected_positions"]))
    capture.require(len(source_positions) == len(selected_rows) == len(labels) == 20, "selection manifest must contain 20 paired rows")
    samples = [
        {"position": i, "source_population_position": source_positions[i], "source_row_index": selected_rows[i], "label": labels[i], "selection_role": "A1000 confirmation"}
        for i in range(20)
    ]
    capture.require([labels.count(label) for label in range(10)] == [2] * 10, "selection manifest class balance differs")
    manifest_endpoints = {item["endpoint"]: item for item in manifest["endpoints"]}
    datasets: dict[str, dict[int, dict[str, Any]]] = {SMALL_NAME: {}, LARGE_NAME: {}}
    expected_by_position = {sample["position"]: sample for sample in samples}
    for endpoint in (SMALL_NAME, LARGE_NAME):
        record = manifest_endpoints[endpoint]
        required_nodes = [name for name, _display in NODE_MAP[endpoint]]
        for chunk_record in record["chunks"]:
            chunk_path = OUTPUT / chunk_record["path"]
            chunk = capture._load_field_chunk(chunk_path, expected_endpoint=endpoint)
            for local, position in enumerate(chunk["positions"].tolist()):
                capture.require(position in expected_by_position, f"{endpoint}: unselected display position {position} was saved")
                sample = expected_by_position[position]
                row = int(chunk["source_row_indices"][local])
                label = int(chunk["labels"][local])
                capture.require((row, label) == (sample["source_row_index"], sample["label"]), f"{endpoint}: selected source row or label differs")
                capture.require(position not in datasets[endpoint], f"{endpoint}: duplicate selected display position {position}")
                missing = [node for node in required_nodes if node not in chunk["fields"]]
                capture.require(not missing, f"{endpoint}: required display nodes are missing: {missing}")
                node_metadata = {node: dict(chunk["field_metadata"][node]) for node in required_nodes}
                for node in required_nodes:
                    metadata = node_metadata[node]
                    if metadata.get("field_kind") == "complex_ols_coefficients":
                        capture.require(metadata.get("coordinate_frame") == "row_major_ols_port_order" and metadata.get("port_order") == "row-major j=10*r+c", f"{endpoint}/{node}: OLS port order metadata is invalid")
                        capture.require(metadata.get("plane_shape") == [10, 10], f"{endpoint}/{node}: OLS field must be 10×10")
                    elif endpoint == SMALL_NAME:
                        capture.require(node in source_coordinates, f"{endpoint}/{node}: authoritative physical coordinate metadata is missing")
                        node_metadata[node] = _resolve_small_coordinates(metadata, source_coordinates[node], node)
                    else:
                        node_metadata[node] = _complete_capture_coordinate_convention(metadata, coordinate_convention, node)
                datasets[endpoint][position] = {
                    "fields": {node: chunk["fields"][node][local].numpy().copy() for node in required_nodes},
                    "metadata": node_metadata,
                    "source_row_index": row,
                    "label": label,
                    "field_chunk_path": chunk_record["path"],
                }
            del chunk
        capture.require(set(datasets[endpoint]) == set(expected_by_position), f"{endpoint}: saved chunks do not cover all selected rows")
    for sample in samples:
        pos = sample["position"]
        _validate_pair_sample(sample, datasets[SMALL_NAME][pos], datasets[LARGE_NAME][pos])
        for index, (_display, small_node, large_node) in enumerate(_node_specs()):
            small = datasets[SMALL_NAME][pos]["metadata"][small_node]
            large = datasets[LARGE_NAME][pos]["metadata"][large_node]
            capture.require(small["field_kind"] == large["field_kind"], f"node {index + 1}: paired field kinds differ")
            capture.require(small["plane_shape"] == large["plane_shape"], f"node {index + 1}: paired plane shapes differ")
            if small["field_kind"] == "complex_ols_coefficients":
                capture.require(small["coordinate_frame"] == large["coordinate_frame"] and small.get("port_order") == large.get("port_order"), f"node {index + 1}: paired OLS port order differs")
            else:
                capture.require(small["origin_m"] == large["origin_m"], f"node {index + 1}: paired origins differ")
                capture.require(small["pixel_pitch_m"] == large["pixel_pitch_m"], f"node {index + 1}: paired pixel pitch differs")
            capture.require(math.isclose(float(small["z_m"]), float(large["z_m"]), abs_tol=1.0e-12), f"node {index + 1}: paired axial coordinates differ")
    _write_coordinate_metadata(samples, datasets, coordinate_convention)
    _write_field_source_manifest(samples, datasets, manifest)
    return samples, datasets


def _write_coordinate_metadata(
    samples: list[dict[str, Any]],
    datasets: dict[str, dict[int, dict[str, Any]]],
    coordinate_convention: dict[str, Any],
) -> None:
    endpoints: dict[str, dict[str, Any]] = {SMALL_NAME: {}, LARGE_NAME: {}}
    for endpoint in (SMALL_NAME, LARGE_NAME):
        for index, (_display, small_node, large_node) in enumerate(_node_specs(), start=1):
            node = small_node if endpoint == SMALL_NAME else large_node
            records = [datasets[endpoint][sample["position"]]["metadata"][node] for sample in samples]
            first = records[0]
            for other in records[1:]:
                coordinate_keys = ("coordinate_frame", "plane_id", "z_m", "origin_m", "pixel_pitch_m", "plane_shape", "fov_m", "wavelength_m", "port_order")
                capture.require(all(other.get(key) == first.get(key) for key in coordinate_keys), f"{endpoint}/{node}: coordinate metadata changes between samples")
            if first.get("field_kind") == "complex_ols_coefficients":
                endpoints[endpoint][node] = {key: first[key] for key in ("coordinate_frame", "port_order", "z_m", "plane_shape")}
            else:
                endpoints[endpoint][node] = {key: first[key] for key in ("coordinate_frame", "plane_id", "z_m", "origin_m", "pixel_pitch_m", "plane_shape", "fov_m", "wavelength_m", "origin_convention", "pixel_center_formula")}
    coordinate_manifest = {
        "schema": "lightgen_best_endpoints_10node_coordinate_metadata.v1",
        "coordinate_system": "PlaneSpec physical grid; no field interpolation or resampling",
        "coordinate_convention": coordinate_convention,
        "source_provenance": {
            "small_A1000": "outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/field_coordinate_metadata.json, endpoint small_A1000; checked against each saved field chunk's coordinate frame, plane, z and pixel pitch",
            "large_B5400_fg": "field_metadata in the current frozen replay capture chunks",
            "source_role": "canonical TRAIN confirmation/display replay",
        },
        "endpoints": endpoints,
    }
    coordinate_path = OUTPUT / "field_coordinate_metadata.json"
    if coordinate_path.exists():
        capture.require(_read_json(coordinate_path) == coordinate_manifest, "refusing to replace a different field-coordinate manifest")
    else:
        capture._write_json(coordinate_path, coordinate_manifest)
    resolution = {
        "schema": "lightgen_best_endpoints_10node_coordinate_resolution.v1",
        "small_A1000_source": "outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/field_coordinate_metadata.json",
        "small_A1000_nodes_resolved_from_source": list(endpoints[SMALL_NAME]),
        "large_B5400_fg_nodes_from_capture_metadata": list(endpoints[LARGE_NAME]),
        "sample_count": len(samples),
        "validated_against_each_saved_field_record": True,
        "fov_independently_matches_native_shape_times_pixel_pitch": True,
    }
    resolution_path = OUTPUT / "coordinate_metadata_resolution.json"
    if resolution_path.exists():
        capture.require(_read_json(resolution_path) == resolution, "refusing to replace a different coordinate-resolution report")
    else:
        capture._write_json(resolution_path, resolution)


def _write_field_source_manifest(
    samples: list[dict[str, Any]],
    datasets: dict[str, dict[int, dict[str, Any]]],
    capture_manifest: dict[str, Any],
) -> None:
    endpoints = {item["endpoint"]: item for item in capture_manifest["endpoints"]}
    path = OUTPUT / "field_source_manifest.csv"
    rows = []
    for sample in samples:
        for node_index, (display_name, small_node, large_node) in enumerate(_node_specs(), start=1):
            for endpoint, node in ((SMALL_NAME, small_node), (LARGE_NAME, large_node)):
                data = datasets[endpoint][sample["position"]]
                metadata = data["metadata"][node]
                rows.append({
                    "endpoint": endpoint,
                    "checkpoint": endpoints[endpoint]["checkpoint"],
                    "checkpoint_step": endpoints[endpoint]["checkpoint_step"],
                    "display_position": sample["position"],
                    "source_population_position": sample["source_population_position"],
                    "source_row_index": sample["source_row_index"],
                    "label": sample["label"],
                    "source_role": "A1000 confirmation; canonical TRAIN display sample" if endpoint == SMALL_NAME else "same A1000 confirmation row; B5400 frozen shared-input replay, outside B5400 parent fit/confirmation",
                    "field_data_role": json.dumps(metadata.get("data_roles"), ensure_ascii=False),
                    "node_index": node_index,
                    "display_name": display_name,
                    "source_node": node,
                    "field_kind": metadata["field_kind"],
                    "shape": json.dumps(metadata["shape"]),
                    "complex_dtype": metadata["complex_dtype"],
                    "plane_id": metadata.get("plane_id"),
                    "coordinate_frame": metadata["coordinate_frame"],
                    "origin_m": json.dumps(metadata.get("origin_m")),
                    "pixel_pitch_m": json.dumps(metadata.get("pixel_pitch_m")),
                    "plane_shape": json.dumps(metadata["plane_shape"]),
                    "fov_m": json.dumps(metadata.get("fov_m")),
                    "z_m": metadata["z_m"],
                    "wavelength_m": metadata.get("wavelength_m"),
                    "phase_position": metadata.get("phase_position"),
                    "field_chunk": data["field_chunk_path"],
                })
    capture.require(len(rows) == 400, "field source manifest must map 2 endpoints × 20 rows × 10 nodes")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue()
    if path.exists():
        existing = path.read_text(encoding="utf-8-sig")
        capture.require(existing.replace("\r\n", "\n") == content.replace("\r\n", "\n"), f"existing field source manifest differs from the frozen field records: {path}")
    else:
        path.write_text(content, encoding="utf-8-sig", newline="")


def _validate_pair_sample(sample: dict[str, Any], small_data: dict[str, Any], large_data: dict[str, Any]) -> None:
    for endpoint, record in ((SMALL_NAME, small_data), (LARGE_NAME, large_data)):
        capture.require(record.get("source_row_index") == sample["source_row_index"] and record.get("label") == sample["label"], f"{endpoint}: selected source row or label differs")
        required = [node for node, _display in NODE_MAP[endpoint]]
        missing = [node for node in required if node not in record.get("fields", {}) or node not in record.get("metadata", {})]
        capture.require(not missing, f"{endpoint}: required display nodes are missing: {missing}")


def _node_specs() -> list[tuple[str, str, str]]:
    manifest = _read_json(OUTPUT / "field_capture_manifest.json")
    return [
        (item["display_name"], item["small_source_node"], item["large_source_node"])
        for item in manifest["nodes"]
    ]


def _crop_display(field: np.ndarray, metadata: dict[str, Any]) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    capture.require(np.iscomplexobj(field) and np.isfinite(field.real).all() and np.isfinite(field.imag).all(), "display input must be a finite complex field")
    if field.ndim == 3 and field.shape[0] == 1:
        field = field[0]
    capture.require(field.ndim == 2, "display field must be one native 2D plane or a singleton-leading-dimension plane")
    capture.require(list(field.shape) == metadata["plane_shape"], "display field shape differs from native coordinate metadata")
    rows, cols = field.shape[-2:]
    if metadata["field_kind"] == "complex_ols_coefficients":
        return field, (-0.5, cols - 0.5, -0.5, rows - 0.5), "10×10 row-major OLS ports; not a metric plane"
    crop_note = "native full plane"
    r0 = c0 = 0
    if rows == cols == capture.GRID:
        size = 400
        r0 = c0 = (capture.GRID - size) // 2
        field = field[r0 : r0 + size, c0 : c0 + size]
        rows = cols = size
        crop_note = "center 400×400 native pixels; no interpolation"
    dy, dx = map(float, metadata["pixel_pitch_m"])
    oy, ox = map(float, metadata["origin_m"])
    native_rows, native_cols = metadata["plane_shape"]
    x0 = ox + (c0 - (native_cols - 1) / 2) * dx
    x1 = ox + (c0 + cols - 1 - (native_cols - 1) / 2) * dx
    y0 = oy + (r0 - (native_rows - 1) / 2) * dy
    y1 = oy + (r0 + rows - 1 - (native_rows - 1) / 2) * dy
    extent = (x0 - dx / 2, x1 + dx / 2, y0 - dy / 2, y1 + dy / 2)
    return field, extent, crop_note


def _shared_scales(samples: list[dict[str, Any]], datasets: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    scales: dict[str, Any] = {"percentile": 99.5, "phase_range_rad": [-math.pi, math.pi], "phase_cmap": PHASE_CMAP, "amplitude_cmap": AMPLITUDE_CMAP, "phase_blank_below_fraction_of_shared_P99_5": 0.01, "nodes": {}}
    for index, (display_name, small_node, large_node) in enumerate(_node_specs(), start=1):
        values = []
        for endpoint, node in ((SMALL_NAME, small_node), (LARGE_NAME, large_node)):
            for sample in samples:
                field = datasets[endpoint][sample["position"]]["fields"][node]
                metadata = datasets[endpoint][sample["position"]]["metadata"][node]
                display, _extent, _crop_note = _crop_display(field, metadata)
                values.append(np.abs(display).astype(np.float32, copy=False).reshape(-1))
        all_values = np.concatenate(values)
        scale = float(np.percentile(all_values, 99.5, method="linear"))
        capture.require(math.isfinite(scale) and scale > 0.0, f"node {index}: shared P99.5 amplitude scale is invalid")
        per_endpoint: dict[str, Any] = {}
        for endpoint, node in ((SMALL_NAME, small_node), (LARGE_NAME, large_node)):
            rows = []
            for sample in samples:
                field = datasets[endpoint][sample["position"]]["fields"][node]
                metadata = datasets[endpoint][sample["position"]]["metadata"][node]
                display, _extent, crop_note = _crop_display(field, metadata)
                amplitude = np.abs(display)
                rows.append({
                    "position": sample["position"],
                    "source_row_index": sample["source_row_index"],
                    "label": sample["label"],
                    "display_clipped_fraction": float(np.mean(amplitude > scale)),
                    "phase_mask_fraction": float(np.mean(amplitude < 0.01 * scale)),
                    "crop_note": crop_note,
                })
            per_endpoint[endpoint] = rows
        scales["nodes"][str(index)] = {
            "display_name": display_name,
            "small_source_node": small_node,
            "large_source_node": large_node,
            "shared_P99_5_amplitude": scale,
            "small_A1000": per_endpoint[SMALL_NAME],
            "large_B5400_fg": per_endpoint[LARGE_NAME],
        }
    return scales


def _node_title(index: int, display_name: str, endpoint: str, metadata: dict[str, Any], scale: float | None) -> str:
    if metadata["field_kind"] == "complex_ols_coefficients":
        dimensions = "10×10端口系数"
    else:
        dimensions = _window_description(metadata)
    device_note = ""
    if index == 6:
        device_note = "\n自由传播参照（无器件）" if endpoint == SMALL_NAME else "\n新增面后完整 Et"
    if index == 8:
        device_note = "\n自由传播参照（无器件）" if endpoint == SMALL_NAME else "\n新增面后完整 Et"
    scale_note = f"\n共同 P99.5={scale:.3g}" if scale is not None else ""
    return f"{index}. {display_name}\n{dimensions}{device_note}{scale_note}"


def _window_description(metadata: dict[str, Any]) -> str:
    rows, cols = map(int, metadata["plane_shape"])
    dy, dx = map(float, metadata["pixel_pitch_m"])
    displayed_rows = 400 if (rows, cols) == (capture.GRID, capture.GRID) else rows
    displayed_cols = 400 if (rows, cols) == (capture.GRID, capture.GRID) else cols
    displayed_fov_mm = (displayed_cols * dx * 1e3, displayed_rows * dy * 1e3)
    if (displayed_rows, displayed_cols) != (rows, cols):
        full_fov = metadata["fov_m"]
        full_fov_mm = (full_fov[1] * 1e3, full_fov[0] * 1e3)
        return f"中心显示窗 {displayed_fov_mm[0]:.2g}×{displayed_fov_mm[1]:.2g} mm · 全幅 {full_fov_mm[0]:.2g}×{full_fov_mm[1]:.2g} mm"
    return f"FOV {displayed_fov_mm[0]:.3g}×{displayed_fov_mm[1]:.3g} mm"


def _paired_node_caption(
    index: int,
    display_name: str,
    small_metadata: dict[str, Any],
    large_metadata: dict[str, Any],
    *,
    kind: str,
    scale: float,
) -> str:
    if small_metadata["field_kind"] == "complex_ols_coefficients":
        window = "10×10端口序列"
    else:
        window = _window_description(small_metadata)
    device = ""
    if index in (6, 8):
        small_position = small_metadata.get("phase_position") or ""
        large_position = large_metadata.get("phase_position") or ""
        capture.require("no device" in small_position, f"node {index}: small endpoint is not marked as a free propagation reference")
        capture.require("full E*t" in large_position, f"node {index}: large endpoint is not marked as the full post-phase field")
        device = "\nA1000自由传播 · B5400相位后 Et"
    display = f"{index}. {display_name}\n{window}{device}"
    if kind == "amplitude":
        return f"{display}\n共同 P99.5={scale:.3g}"
    return f"{display}\narg(E) · [−π, π)"


def _render_node_image(field: np.ndarray, metadata: dict[str, Any], *, kind: str, scale: float, path: Path) -> float:
    display, extent, _crop_note = _crop_display(field, metadata)
    fig, ax = plt.subplots(figsize=(3.3, 3.3), constrained_layout=True)
    if kind == "amplitude":
        amplitude = np.abs(display)
        clipped = float(np.mean(amplitude > scale))
        ax.imshow(np.minimum(amplitude, scale), origin="lower", cmap=AMPLITUDE_CMAP, vmin=0.0, vmax=scale, extent=extent, interpolation="nearest", aspect="equal")
    else:
        amplitude = np.abs(display)
        phase = np.angle(display)
        shown = np.ma.array(phase, mask=amplitude < 0.01 * scale)
        ax.imshow(shown, origin="lower", cmap=PHASE_CMAP, vmin=-math.pi, vmax=math.pi, extent=extent, interpolation="nearest", aspect="equal")
        clipped = float(np.mean(amplitude < 0.01 * scale))
    ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return clipped


def _save_figure_triplet(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)


def _atlas_page(
    sample: dict[str, Any],
    node_indices: range,
    datasets: dict[str, dict[int, dict[str, Any]]],
    scales: dict[str, Any],
) -> plt.Figure:
    fig, axes = plt.subplots(4, 5, figsize=(16.0, 9.2), squeeze=False)
    row_specs = ((SMALL_NAME, "amplitude"), (LARGE_NAME, "amplitude"), (SMALL_NAME, "phase"), (LARGE_NAME, "phase"))
    row_labels = ("小架构 A1000 · 振幅 |E|", "大架构 B5400 · 振幅 |E|", "小架构 A1000 · 相位", "大架构 B5400 · 相位")
    for row, ((endpoint, kind), row_label) in enumerate(zip(row_specs, row_labels, strict=True)):
        axes[row, 0].set_ylabel(row_label, fontsize=8)
        for col, index in enumerate(node_indices):
            node_info = _node_specs()[index - 1]
            node = node_info[1] if endpoint == SMALL_NAME else node_info[2]
            sample_data = datasets[endpoint][sample["position"]]
            field = sample_data["fields"][node]
            metadata = sample_data["metadata"][node]
            display, extent, _note = _crop_display(field, metadata)
            scale = scales["nodes"][str(index)]["shared_P99_5_amplitude"]
            if kind == "amplitude":
                axes[row, col].imshow(np.minimum(np.abs(display), scale), origin="lower", cmap=AMPLITUDE_CMAP, vmin=0.0, vmax=scale, extent=extent, interpolation="nearest", aspect="equal")
            else:
                phase = np.angle(display)
                mask = np.abs(display) < 0.01 * scale
                phase_image = axes[row, col].imshow(np.ma.array(phase, mask=mask), origin="lower", cmap=PHASE_CMAP, vmin=-math.pi, vmax=math.pi, extent=extent, interpolation="nearest", aspect="equal")
            if row == 0:
                small_meta = datasets[SMALL_NAME][sample["position"]]["metadata"][node_info[1]]
                large_meta = datasets[LARGE_NAME][sample["position"]]["metadata"][node_info[2]]
                axes[row, col].set_title(_paired_node_caption(index, node_info[0], small_meta, large_meta, kind="amplitude", scale=scale), fontsize=7)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f"类别{sample['label']} · TRAIN行 {sample['source_row_index']} · 节点{node_indices.start}–{node_indices.stop - 1}", fontsize=13)
    fig.text(0.50, 0.015, "相位范围 [−π, π)，环形色图；低于该节点共同振幅 P99.5 的1%留白。P99.5 在两端点全部20张样本的共同显示窗上计算。", ha="center", fontsize=8)
    phase_bar = fig.colorbar(phase_image, ax=axes[2:, :].ravel().tolist(), orientation="horizontal", fraction=0.035, pad=0.05, ticks=[-math.pi, 0.0, math.pi])
    phase_bar.ax.set_xticklabels(["−π", "0", "π"])
    phase_bar.set_label("相位 (rad)", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.89, bottom=0.12, hspace=0.30, wspace=0.20)
    return fig


def _build_presentation_panels(
    samples: list[dict[str, Any]],
    datasets: dict[str, dict[int, dict[str, Any]]],
    scales: dict[str, Any],
) -> dict[str, Any]:
    slides = []
    panel_root = FIGURES / "ppt_panels"
    for sample_index in PRESENTATION_SAMPLES:
        sample = samples[sample_index]
        capture.require(sample["label"] in {1, 7}, "representative PPT rows must be the first saved class-1 and class-7 samples")
        for kind in ("amplitude", "phase"):
            for start in (1, 6):
                indices = range(start, start + 5)
                slug = f"class_{sample['label']}_row_{sample['source_row_index']}_{kind}_nodes_{start}_{start + 4}"
                fig, axes = plt.subplots(2, 5, figsize=(15.8, 5.0), squeeze=False)
                panel_records = []
                row_endpoints = (SMALL_NAME, LARGE_NAME)
                for row, endpoint in enumerate(row_endpoints):
                    axes[row, 0].set_ylabel("小架构 A1000" if endpoint == SMALL_NAME else "大架构 B5400", fontsize=9)
                    for col, index in enumerate(indices):
                        node_info = _node_specs()[index - 1]
                        node = node_info[1] if endpoint == SMALL_NAME else node_info[2]
                        sample_data = datasets[endpoint][sample["position"]]
                        field = sample_data["fields"][node]
                        metadata = sample_data["metadata"][node]
                        scale = scales["nodes"][str(index)]["shared_P99_5_amplitude"]
                        display, extent, _crop_note = _crop_display(field, metadata)
                        if kind == "amplitude":
                            clipped = float(np.mean(np.abs(display) > scale))
                            axes[row, col].imshow(np.minimum(np.abs(display), scale), origin="lower", cmap=AMPLITUDE_CMAP, vmin=0.0, vmax=scale, extent=extent, interpolation="nearest", aspect="equal")
                        else:
                            mask = np.abs(display) < 0.01 * scale
                            clipped = float(mask.mean())
                            phase_image = axes[row, col].imshow(np.ma.array(np.angle(display), mask=mask), origin="lower", cmap=PHASE_CMAP, vmin=-math.pi, vmax=math.pi, extent=extent, interpolation="nearest", aspect="equal")
                        if row == 0:
                            small_meta = datasets[SMALL_NAME][sample["position"]]["metadata"][node_info[1]]
                            large_meta = datasets[LARGE_NAME][sample["position"]]["metadata"][node_info[2]]
                            caption = _paired_node_caption(index, node_info[0], small_meta, large_meta, kind=kind, scale=scale)
                            axes[row, col].set_title(caption, fontsize=8)
                        axes[row, col].set_xticks([])
                        axes[row, col].set_yticks([])
                        image_path = panel_root / f"class_{sample['label']}_row_{sample['source_row_index']}" / kind / f"nodes_{start}_{start+4}" / f"node_{index:02d}_{endpoint}.png"
                        _render_node_image(field, metadata, kind=kind, scale=scale, path=image_path)
                        panel_records.append({"node_index": index, "node_label": caption if row == 0 else None, "alt": f"{endpoint} {node_info[0]} {kind}, row {sample['source_row_index']}", "endpoint": endpoint, "image": image_path.relative_to(OUTPUT).as_posix(), "source_node": node, "scale": scale, "suppressed_fraction": clipped})
                if kind == "phase":
                    phase_bar = fig.colorbar(phase_image, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.045, pad=0.11, ticks=[-math.pi, 0.0, math.pi])
                    phase_bar.ax.set_xticklabels(["−π", "0", "π"])
                fig.suptitle(f"类别{sample['label']}首样本 · {'振幅' if kind == 'amplitude' else '相位'} · 节点{start}–{start+4}", fontsize=14)
                fig.text(0.5, 0.01, f"来源行 {sample['source_row_index']}；相位低振幅留白阈值=节点共同P99.5的1%；样本为canonical TRAIN展示行。", ha="center", fontsize=8)
                fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.10, hspace=0.28, wspace=0.20)
                _save_figure_triplet(fig, FIGURES / "presentation_sources" / slug)
                plt.close(fig)
                slides.append({
                    "sample_position": sample["position"],
                    "source_row_index": sample["source_row_index"],
                    "label": sample["label"],
                    "kind": kind,
                    "node_start": start,
                    "node_end": start + 4,
                    "title": f"类别{sample['label']}首样本 · {'振幅' if kind == 'amplitude' else '相位'} · 节点{start}–{start+4}",
                    "panels": panel_records,
                })
    return {"schema": "lightgen_best_endpoints_10node_presentation_panels.v1", "slides": slides}


def build(*, resume_partial: bool = False, rebuild_derived: bool = False) -> dict[str, Any]:
    capture.require(OUTPUT.is_dir(), "frozen field capture directory is missing")
    atlas_path = FIGURES / "complex_field_atlas.pdf"
    report_path = OUTPUT / "visualization_build_report.json"
    if FIGURES.exists():
        capture.require(resume_partial or rebuild_derived, "refusing to overwrite existing derived figures without an explicit resume/rebuild option")
    if report_path.exists():
        capture.require(rebuild_derived, "refusing to overwrite a completed figure-build report")
    if atlas_path.exists():
        capture.require((resume_partial or rebuild_derived) and atlas_path.stat().st_size > 0, "refusing to overwrite an existing atlas PDF")
    samples, datasets = _load_samples()
    scales = _shared_scales(samples, datasets)
    FIGURES.mkdir(parents=True, exist_ok=resume_partial or rebuild_derived)
    scaling_path = OUTPUT / "display_scaling.json"
    if scaling_path.exists():
        capture.require((resume_partial or rebuild_derived) and _read_json(scaling_path) == scales, "existing display scales differ from the frozen field data")
    else:
        capture._write_json(scaling_path, scales)
    page_manifest = []
    page_number = 0
    for sample in samples:
        for start in (1, 6):
            page_number += 1
            page_manifest.append({"page": page_number, "source_row_index": sample["source_row_index"], "label": sample["label"], "node_start": start, "node_end": start + 4})
    if atlas_path.exists() and resume_partial:
        capture.require(len(page_manifest) == 40 and atlas_path.stat().st_size > 0, "partial-resume atlas is empty or does not match the 40-page plan")
    else:
        build_path = atlas_path.with_name("complex_field_atlas.rebuild.pdf") if atlas_path.exists() else atlas_path
        with PdfPages(build_path) as pdf:
            for sample in samples:
                for start in (1, 6):
                    fig = _atlas_page(sample, range(start, start + 5), datasets, scales)
                    pdf.savefig(fig, dpi=200, bbox_inches="tight")
                    plt.close(fig)
        if build_path != atlas_path:
            build_path.replace(atlas_path)
    presentation_manifest = _build_presentation_panels(samples, datasets, scales)
    _write_result_json(OUTPUT / "presentation_panels.json", presentation_manifest, replace_existing=rebuild_derived)
    _write_result_json(OUTPUT / "field_atlas_manifest.json", {"pdf": "figures/complex_field_atlas.pdf", "page_count": page_number, "pages": page_manifest, "nodes_per_sample": 10, "rows_per_endpoint": 20}, replace_existing=rebuild_derived)
    result = {"status": "FIGURES_BUILT", "atlas_pages": page_number, "slides": len(presentation_manifest["slides"]), "node_count": 10, "sample_count_per_endpoint": 20, "field_source_manifest_rows": 400, "presentation_source_figures": len(presentation_manifest["slides"])}
    _write_result_json(report_path, result, replace_existing=rebuild_derived)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-partial", action="store_true", help="resume this attempt after an incomplete figure build")
    parser.add_argument("--rebuild-derived", action="store_true", help="rebuild plots from the same frozen capture after a verified presentation correction")
    args = parser.parse_args()
    print(json.dumps(build(resume_partial=args.resume_partial, rebuild_derived=args.rebuild_derived), ensure_ascii=False, indent=2))
