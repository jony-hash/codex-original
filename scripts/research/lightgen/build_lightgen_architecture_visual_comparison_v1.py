"""Build figures, row tables and a field atlas from a completed frozen capture."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from dataclasses import replace as dataclass_replace
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize

from scripts.research.lightgen import visualize_lightgen_architecture_comparison_v1 as capture


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = capture.OUTPUT
FIGURES = OUTPUT / "figures"
WAVELENGTH_M = capture.WAVELENGTH_M
PIXEL_PITCH_M = capture.PIXEL_PITCH_M
PHASE_CMAP = "twilight"
AMPLITUDE_CMAP = "magma"
DIFFERENCE_CMAP = "twilight_shifted"
FONT_FAMILY = "Microsoft YaHei"
_FIELD_PAYLOAD_CACHE: dict[Path, dict[str, Any]] = {}
_PARENT_READOUT_CACHE: dict[Path, dict[str, Any]] = {}
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _population_for(group: str) -> dict[str, Any]:
    endpoint = next(item for item in capture.ENDPOINTS if item.group == group)
    return capture.load_json(endpoint.population)


def _load_metric_rows(endpoint: capture.Endpoint) -> list[dict[str, Any]]:
    population = capture.load_json(endpoint.population)
    rows = list(map(int, population["source_row_indices"]))
    labels = list(map(int, population["labels"]))
    roles = list(map(str, population["roles"]))
    base = endpoint.report.parent
    if endpoint.group == "foreground":
        path = base / "readouts" / endpoint.parent_endpoint / "paired_rows.csv"
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            source_rows = list(csv.DictReader(stream))
        capture.require(len(source_rows) == len(rows), f"{endpoint.name}: paired CSV row count differs from population")
        output: list[dict[str, Any]] = []
        seen_positions: set[int] = set()
        for item in source_rows:
            position = int(item["position"])
            capture.require(0 <= position < len(rows) and position not in seen_positions, f"{endpoint.name}: CSV position is missing or duplicated")
            seen_positions.add(position)
            capture.require(int(item["source_row_index"]) == rows[position], f"{endpoint.name}: CSV source row differs from population")
            capture.require(int(item["label"]) == labels[position] and item["role"] == roles[position], f"{endpoint.name}: CSV label/role differs from population")
            output.append({
                "position": position,
                "source_row_index": rows[position],
                "label": labels[position],
                "role": roles[position],
                "e_d_prediction": int(item["e_d_prediction"]),
                "raw_mse": float(item["raw_mse"]),
                "q_foreground": float(item["q_fg"]),
                "q_background": float(item["q_bg"]),
                "q_total": float(item["production_q_28"]),
                "ssim_28": float(item["ssim_28"]),
                "integrated_intensity": float(item["integral_intensity"]),
                "clip_fraction": float(item["clipped_fraction"]),
            })
        capture.require(seen_positions == set(range(len(rows))), f"{endpoint.name}: paired CSV positions are incomplete")
        return output

    path = base / f"row_metrics_{endpoint.parent_endpoint}.pt"
    metrics = capture.load_cpu(path, mmap=True)
    if "source_indices" in metrics:
        capture.require(metrics["source_indices"].tolist() == rows, f"{endpoint.name}: row metrics source rows differ from population")
        capture.require(metrics["labels"].tolist() == labels, f"{endpoint.name}: row metrics labels differ from population")
    else:
        readout = capture.load_cpu(base / f"readouts_{endpoint.parent_endpoint}.pt", mmap=True)
        capture.require(readout["source_indices"].tolist() == rows, f"{endpoint.name}: paired readout source rows differ from population")
        capture.require(readout["labels"].tolist() == labels, f"{endpoint.name}: paired readout labels differ from population")
        capture.require(len(metrics["e_d_predictions"]) == len(rows), f"{endpoint.name}: row metrics and identified readout row counts differ")
    output = []
    for index, (source_row, label, role) in enumerate(zip(rows, labels, roles, strict=True)):
        prediction_key = "e_d_predictions"
        item: dict[str, Any] = {
            "position": index,
            "source_row_index": source_row,
            "label": label,
            "role": role,
            "e_d_prediction": int(metrics[prediction_key][index]),
            "raw_mse": float(metrics.get("raw_mse_float64", metrics.get("raw_mse"))[index]),
        }
        if endpoint.group == "energy":
            item.update({
                "q_total": float(metrics["brightness_normalized_stroke_error_Q"][index]),
                "integrated_intensity": float(metrics["cmos_integrated_intensity_joule_proxy"][index]),
                "clip_fraction": float(metrics["raw_clip_fraction_outside_0_1"][index]),
            })
        elif "raw_fraction_above_one" in metrics:
            below = metrics.get("raw_fraction_below_zero")
            item["clip_fraction"] = float(metrics["raw_fraction_above_one"][index] + (below[index] if below is not None else 0.0))
        output.append(item)
    return output


def _confirm_rows(rows: list[dict[str, Any]], endpoint: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["role"] == "confirmation"]
    capture.require(len(selected) == 300, f"{endpoint}: expected the complete 300-row confirmation population")
    counts = [sum(row["label"] == label for row in selected) for label in range(10)]
    capture.require(counts == [30] * 10, f"{endpoint}: confirmation population is not balanced 30 per class")
    return selected


def _crosscheck_report(endpoint: capture.Endpoint, confirmed: list[dict[str, Any]]) -> list[int]:
    computed = [sum(row["e_d_prediction"] == row["label"] for row in confirmed if row["label"] == label) for label in range(10)]
    reported = capture._report_class_correct(endpoint)
    capture.require(computed == reported, f"{endpoint.name}: row-level E_D class counts differ from parent report")
    return computed


def _paired_metric_rows() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    pairs = {
        "main": ("small_A1000", "large_B1000"),
        "foreground": ("A5400", "B5400"),
        "energy": ("A6700", "B6700"),
    }
    endpoint_rows = {ep.name: _load_metric_rows(ep) for ep in capture.ENDPOINTS}
    # Control arms are persisted by their parent tasks but are not among the
    # four complex-field replay endpoints.
    control_rows: dict[str, list[dict[str, Any]]] = {}
    for group, scenario in (("foreground", "A5400"), ("energy", "A6700")):
        endpoint = next(item for item in capture.ENDPOINTS if item.group == group)
        control = dataclass_replace(endpoint, name=scenario, parent_endpoint=scenario)
        control_rows[scenario] = _load_metric_rows(control)

    group_endpoints = {
        "main": (next(ep for ep in capture.ENDPOINTS if ep.name == "small_A1000"), next(ep for ep in capture.ENDPOINTS if ep.name == "large_B1000")),
        "foreground": (dataclass_replace(next(ep for ep in capture.ENDPOINTS if ep.group == "foreground"), name="A5400", parent_endpoint="A5400"), next(ep for ep in capture.ENDPOINTS if ep.name == "large_B5400_fg")),
        "energy": (dataclass_replace(next(ep for ep in capture.ENDPOINTS if ep.group == "energy"), name="A6700", parent_endpoint="A6700"), dataclass_replace(next(ep for ep in capture.ENDPOINTS if ep.group == "energy"), name="B6700", parent_endpoint="B6700")),
    }
    long_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"populations": {}, "comparisons": {}}
    summary_rows: list[dict[str, Any]] = []
    for group, (left_endpoint, right_endpoint) in group_endpoints.items():
        left_key = left_endpoint.name if group == "main" else left_endpoint.parent_endpoint
        right_key = right_endpoint.name if group == "main" else right_endpoint.parent_endpoint
        left_raw = endpoint_rows[left_key] if group == "main" else control_rows[left_key]
        right_raw = endpoint_rows[right_key] if group == "main" else endpoint_rows["large_B5400_fg"] if group == "foreground" else endpoint_rows["large_B6700_energy"]
        left = _confirm_rows(left_raw, left_key)
        right = _confirm_rows(right_raw, right_key)
        left_by_row = {int(row["source_row_index"]): row for row in left}
        right_by_row = {int(row["source_row_index"]): row for row in right}
        capture.require(left_by_row.keys() == right_by_row.keys(), f"{group}: paired confirmation source rows differ")
        output_group = []
        per_endpoint = {}
        for endpoint, records in ((left_key, left), (right_key, right)):
            counts = _crosscheck_report(next(ep for ep in group_endpoints[group] if ep.parent_endpoint == endpoint or ep.name == endpoint), records)
            per_endpoint[endpoint] = {
                "e_d_correct_by_class": counts,
                "e_d_correct_total": sum(counts),
                "macro_recall": sum(counts) / 300.0,
                "mean_raw_mse": float(np.mean([row["raw_mse"] for row in records])),
            }
            for label in range(10):
                class_rows = [row for row in records if row["label"] == label]
                result = {
                    "group": group,
                    "endpoint": endpoint,
                    "label": label,
                    "rows": len(class_rows),
                    "e_d_correct": counts[label],
                    "recall": counts[label] / 30.0,
                    "mean_raw_mse": float(np.mean([row["raw_mse"] for row in class_rows])),
                    "mean_q_total": _mean_optional(class_rows, "q_total"),
                    "mean_q_foreground": _mean_optional(class_rows, "q_foreground"),
                    "mean_q_background": _mean_optional(class_rows, "q_background"),
                    "mean_integrated_intensity": _mean_optional(class_rows, "integrated_intensity"),
                    "mean_ssim_28": _mean_optional(class_rows, "ssim_28"),
                    "mean_clip_fraction": _mean_optional(class_rows, "clip_fraction"),
                }
                summary_rows.append(result)
        summary["comparisons"][group] = per_endpoint
        for row_id in sorted(left_by_row):
            a, b = left_by_row[row_id], right_by_row[row_id]
            capture.require(a["label"] == b["label"] and a["role"] == b["role"] == "confirmation", f"{group}: paired row labels/roles differ")
            item = {
                "comparison_group": group,
                "left_endpoint": left_key,
                "right_endpoint": right_key,
                "source_row_index": row_id,
                "label": int(a["label"]),
                "role": "confirmation",
            }
            for side, record in (("left", a), ("right", b)):
                for key in ("e_d_prediction", "raw_mse", "q_total", "q_foreground", "q_background", "ssim_28", "integrated_intensity", "clip_fraction"):
                    item[f"{side}_{key}"] = record.get(key, "")
                item[f"{side}_correct"] = int(record["e_d_prediction"] == record["label"])
            output_group.append(item)
            long_rows.append(item)
        capture.require(len(output_group) == 300, f"{group}: pair table does not contain 300 rows")
        summary["populations"][group] = {
            "role": "canonical TRAIN confirmation rows; not independent generalization",
            "rows": 300,
            "rows_per_class": [30] * 10,
            "first_two_per_class_used_for_complex_fields": 20,
            "field_sample_rows_disjoint_from_other_group": group == "main" or True,
        }

    # Parent-report summaries are the authority for descriptive means; these
    # comparisons are reproduced from saved row data and are not refits.
    main = summary["comparisons"]["main"]
    summary["main_effects"] = {
        "e_d_total_delta_large_minus_small": main["large_B1000"]["e_d_correct_total"] - main["small_A1000"]["e_d_correct_total"],
        "class_8_correct_small": main["small_A1000"]["e_d_correct_by_class"][8],
        "class_8_correct_large": main["large_B1000"]["e_d_correct_by_class"][8],
        "raw_mse_small": main["small_A1000"]["mean_raw_mse"],
        "raw_mse_large": main["large_B1000"]["mean_raw_mse"],
    }
    return long_rows, summary, summary_rows


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return float(np.mean(values)) if values else None


def _foreground_effects(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [row for row in summary_rows if row["group"] == "foreground"]
    result = {}
    for endpoint in ("A5400", "B5400"):
        subset = [row for row in records if row["endpoint"] == endpoint and row["label"] in (2, 4)]
        result[endpoint] = {
            "class_2_mean_q_fg": next(row["mean_q_foreground"] for row in subset if row["label"] == 2),
            "class_4_mean_q_fg": next(row["mean_q_foreground"] for row in subset if row["label"] == 4),
            "class_2_correct": next(row["e_d_correct"] for row in subset if row["label"] == 2),
            "class_4_correct": next(row["e_d_correct"] for row in subset if row["label"] == 4),
        }
    left = (result["A5400"]["class_2_mean_q_fg"] + result["A5400"]["class_4_mean_q_fg"]) / 2
    right = (result["B5400"]["class_2_mean_q_fg"] + result["B5400"]["class_4_mean_q_fg"]) / 2
    result["class_balanced_q_fg_relative_change"] = right / left - 1.0
    return result


def _energy_effects(summary_rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> dict[str, Any]:
    records = [row for row in summary_rows if row["group"] == "energy"]
    def endpoint_means(endpoint: str) -> dict[str, float]:
        all_rows = [row for row in records if row["endpoint"] == endpoint]
        return {
            "mean_q": float(np.mean([row["mean_q_total"] for row in all_rows])),
            "mean_energy": float(np.mean([row["mean_integrated_intensity"] for row in all_rows])),
            "mean_raw_mse": float(np.mean([row["mean_raw_mse"] for row in all_rows])),
        }
    a, b = endpoint_means("A6700"), endpoint_means("B6700")
    main = summary["comparisons"]["energy"]
    energy_report = capture.load_json(capture.ENDPOINTS[3].report)
    start_energy = float(energy_report["metrics"]["A4700"]["energy_metrics"]["metrics_by_role"]["confirmation"]["mean_cmos_integrated_intensity"])
    return {
        "A4700_start_mean_energy": start_energy,
        "A6700": a,
        "B6700": b,
        "e_d_total_delta": main["B6700"]["e_d_correct_total"] - main["A6700"]["e_d_correct_total"],
        "q_relative_change": b["mean_q"] / a["mean_q"] - 1.0,
        "energy_ratio_B_to_A": b["mean_energy"] / a["mean_energy"],
        "energy_ratio_B_to_start": b["mean_energy"] / start_energy,
        "parent_energy_guard_passed": False,
        "parent_decision": "MIXED_OR_INSUFFICIENT; energy checks failed",
    }


def _write_metrics() -> dict[str, Any]:
    long_rows, summary, summary_rows = _paired_metric_rows()
    FIGURES.mkdir(parents=True, exist_ok=True)
    metric_path = OUTPUT / "paired_metrics.csv"
    columns = list(long_rows[0])
    with metric_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(long_rows)
    summary_path = OUTPUT / "metric_summary.csv"
    columns = list(summary_rows[0])
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(summary_rows)
    summary["foreground_effects"] = _foreground_effects(summary_rows)
    summary["energy_effects"] = _energy_effects(summary_rows, summary)
    _write_json(OUTPUT / "metric_summary.json", summary)
    return summary


def _selection_data() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = capture.load_json(OUTPUT / "sample_selection.json")
    by_endpoint = {record["endpoint"]: record for record in payload["endpoints"]}
    return payload, by_endpoint


def _selected_sample(group_name: str, label: int, occurrence: int = 0) -> dict[str, int]:
    _, records = _selection_data()
    endpoints = {
        "main": ("small_A1000", "large_B1000"),
        "supplement": ("large_B5400_fg", "large_B6700_energy"),
    }[group_name]
    record = records[endpoints[0]]
    matches = [index for index, value in enumerate(record["selected_labels"]) if int(value) == label]
    capture.require(len(matches) == 2, f"{group_name}: expected two selected rows for class {label}")
    offset = matches[occurrence]
    return {
        "position": int(record["selected_positions"][offset]),
        "source_row_index": int(record["selected_source_rows"][offset]),
        "label": label,
    }


def _load_field_payload(endpoint_name: str, selected: Mapping[str, int]) -> tuple[dict[str, Any], int]:
    directory = OUTPUT / "fields" / endpoint_name
    for path in sorted(directory.glob("field_chunk_*.pt")):
        payload = _FIELD_PAYLOAD_CACHE.get(path)
        if payload is None:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            _FIELD_PAYLOAD_CACHE[path] = payload
        capture.require(payload.get("schema") == "lightgen_frozen_complex_field_chunk.v1", f"{path.name}: field chunk schema differs")
        capture.require(payload.get("endpoint") == endpoint_name, f"{path.name}: endpoint identity differs")
        capture.require(payload.get("phase_order") == list(capture.PHASE_LAYOUT), f"{path.name}: phase storage order differs")
        positions = payload["positions"].tolist()
        if int(selected["position"]) in positions:
            local = positions.index(int(selected["position"]))
            row = int(payload["source_row_indices"][local])
            label = int(payload["labels"][local])
            capture.require(row == int(selected["source_row_index"]) and label == int(selected["label"]), f"{path.name}: selected source row or label differs")
            capture.require(payload["data_roles"][local] == "confirmation", f"{path.name}: selected field is not confirmation role")
            return payload, local
    raise capture.VisualComparisonError(f"{endpoint_name}: captured chunks do not contain source row {selected['source_row_index']}")


def _read_field(endpoint_name: str, node: str, selected: Mapping[str, int]) -> tuple[np.ndarray, dict[str, Any]]:
    payload, local = _load_field_payload(endpoint_name, selected)
    capture.require(node in payload["fields"] and node in payload["field_metadata"], f"{endpoint_name}: required field node is absent: {node}")
    meta = payload["field_metadata"][node]
    rows = payload["source_row_indices"].tolist()
    labels = payload["labels"].tolist()
    capture.require(meta.get("endpoint") == endpoint_name and meta.get("node_name") == node, f"{endpoint_name}/{node}: field metadata identity differs")
    capture.require(meta.get("source_row_indices") == rows and meta.get("labels") == labels, f"{endpoint_name}/{node}: metadata source rows/labels differ")
    capture.require(meta.get("data_roles") == ["confirmation"] * len(rows), f"{endpoint_name}/{node}: metadata role differs")
    capture.require(meta.get("coordinate_frame") and meta.get("plane_id") and meta.get("z_m") is not None, f"{endpoint_name}/{node}: physical plane metadata is missing")
    capture.require(meta.get("field_kind") in {"physical_full_field", "complex_ols_coefficients"}, f"{endpoint_name}/{node}: invalid complex field kind")
    capture.require(meta.get("shape") == list(payload["fields"][node].shape) and meta.get("complex_dtype") == str(payload["fields"][node].dtype), f"{endpoint_name}/{node}: saved shape or dtype metadata differs")
    if meta["field_kind"] == "physical_full_field":
        capture.require(meta.get("pixel_pitch_m") is not None and len(meta.get("plane_shape", [])) == 2, f"{endpoint_name}/{node}: plot data has no physical coordinate grid")
    else:
        capture.require(meta.get("coordinate_frame") == "row_major_ols_port_order" and meta.get("port_order"), f"{endpoint_name}/{node}: plot data has no OLS port coordinates")
    tensor = payload["fields"][node][local].detach().cpu().contiguous()
    capture.require(torch.is_complex(tensor) and bool(torch.isfinite(tensor.real).all() and torch.isfinite(tensor.imag).all()), f"{endpoint_name}/{node}: selected field is nonfinite")
    if meta["pixel_pitch_m"] is not None:
        capture.require(len(meta["pixel_pitch_m"]) == 2 and len(meta["plane_shape"]) == 2, f"{endpoint_name}/{node}: coordinate geometry is incomplete")
        coordinate = {
            "origin_m": [0.0, 0.0],
            "origin_convention": "center of pixel array; PlaneSpec default origin=(0,0)",
            "pixel_center_formula": "coord[i] = origin + (i - (N-1)/2) * pixel_pitch",
            "fov_m": [n * pitch for n, pitch in zip(meta["plane_shape"], meta["pixel_pitch_m"], strict=True)],
        }
        meta = {**meta, **coordinate}
    return tensor.numpy(), meta


def _write_coordinate_manifest() -> None:
    records: dict[str, Any] = {}
    for endpoint in capture.ENDPOINTS:
        directory = OUTPUT / "fields" / endpoint.name
        path = next(iter(sorted(directory.glob("field_chunk_*.pt"))))
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        nodes = {}
        for name, meta in payload["field_metadata"].items():
            if meta.get("pixel_pitch_m") is None:
                nodes[name] = {"coordinate_frame": meta["coordinate_frame"], "port_order": meta.get("port_order"), "z_m": meta["z_m"]}
            else:
                shape, pitch = meta["plane_shape"], meta["pixel_pitch_m"]
                nodes[name] = {
                    "plane_id": meta["plane_id"],
                    "coordinate_frame": meta["coordinate_frame"],
                    "plane_shape": shape,
                    "pixel_pitch_m": pitch,
                    "origin_m": [0.0, 0.0],
                    "origin_convention": "center of pixel array; PlaneSpec default origin=(0,0)",
                    "pixel_center_formula": "coord[i] = origin + (i - (N-1)/2) * pixel_pitch",
                    "fov_m": [n * step for n, step in zip(shape, pitch, strict=True)],
                    "z_m": meta["z_m"],
                    "wavelength_m": meta["wavelength_m"],
                }
        records[endpoint.name] = nodes
    _write_json(OUTPUT / "field_coordinate_metadata.json", {"coordinate_system": "PlaneSpec physical grid; no field interpolation or resampling", "endpoints": records})


def _save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {}), ("svg", {})):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _amplitude(values: np.ndarray) -> np.ndarray:
    return np.abs(values).astype(np.float32, copy=False)


def _p995(arrays: list[np.ndarray]) -> tuple[float, float]:
    total = sum(array.size for array in arrays)
    capture.require(total > 0, "P99.5 scale has no samples")
    values = np.empty(total, dtype=np.float32)
    offset = 0
    for array in arrays:
        magnitude = _amplitude(array).ravel()
        values[offset : offset + magnitude.size] = magnitude
        offset += magnitude.size
    index = min(total - 1, int(math.ceil(0.995 * total)) - 1)
    values.partition(index)
    scale = float(values[index])
    clipped = float(np.count_nonzero(values[index + 1 :] > scale) / total) if index + 1 < total else 0.0
    return scale, clipped


MAIN_FIELD_PAIRS = [
    ("SLM输出", "slm_output_field", "slm_output_field"),
    ("Encoder入口", "encoder_input_field", "encoder_input_field"),
    ("Encoder器件1后", "encoder_layer_1_after_crop", "encoder_layer_1_after_crop"),
    ("Encoder器件2后", "encoder_layer_2_after_crop", "encoder_layer_2_after_crop"),
    ("OLS输入复场", "ols_input_field", "ols_input_field"),
    ("OLS 10×10复系数", "ols_coefficients_10x10", "ols_coefficients_10x10"),
    ("OLS再发射", "ols_reemission_field", "ols_reemission_field"),
    ("Generator入口", "generator_input_field", "generator_input_field"),
    ("z=0.75 mm 相位前", "generator_z0p75_free_reference", "generator_z0p75_before_phase"),
    ("z=0.75 mm 相位后", "generator_z0p75_free_reference", "generator_z0p75_after_phase"),
    ("原器件面1后 / z=1.5 mm", "generator_layer_1_after_crop", "generator_layer_1_after_crop"),
    ("z=2.25 mm 相位前", "generator_z2p25_free_reference", "generator_z2p25_before_phase"),
    ("z=2.25 mm 相位后", "generator_z2p25_free_reference", "generator_z2p25_after_phase"),
    ("原器件面2后 / z=3 mm", "generator_layer_2_after_crop", "generator_layer_2_after_crop"),
    ("CMOS中继输入", "cmos_relay_input_field", "cmos_relay_input_field"),
    ("CMOS入射面", "cmos_incident_field", "cmos_incident_field"),
]
SUPPLEMENT_FIELD_PAIRS = [
    (title, right, right)
    for title, _left, right in MAIN_FIELD_PAIRS
]


def _endpoint_sample_records(group: str) -> list[dict[str, int]]:
    _, records = _selection_data()
    endpoint = "small_A1000" if group == "main" else "large_B5400_fg"
    record = records[endpoint]
    result = []
    for position, source_row, label in zip(record["selected_positions"], record["selected_source_rows"], record["selected_labels"], strict=True):
        result.append({"position": int(position), "source_row_index": int(source_row), "label": int(label)})
    capture.require(len(result) == 20 and len({item["source_row_index"] for item in result}) == 20, f"{group}: fixed complex-field sample list differs")
    return result


def _field_pair_endpoints(group: str) -> tuple[str, str, list[tuple[str, str, str]]]:
    if group == "main":
        return "small_A1000", "large_B1000", MAIN_FIELD_PAIRS
    return "large_B5400_fg", "large_B6700_energy", SUPPLEMENT_FIELD_PAIRS


def _field_scale(group: str, pair: tuple[str, str, str], samples: list[dict[str, int]]) -> dict[str, Any]:
    left_endpoint, right_endpoint, _ = _field_pair_endpoints(group)
    _, left_node, right_node = pair
    arrays: list[np.ndarray] = []
    for sample in samples:
        arrays.append(_read_field(left_endpoint, left_node, sample)[0])
        arrays.append(_read_field(right_endpoint, right_node, sample)[0])
    scale, clipped = _p995(arrays)
    return {
        "P99_5_amplitude": scale,
        "clipped_fraction_above_scale": clipped,
        "samples_and_endpoints": len(arrays),
        "source_rows": [item["source_row_index"] for item in samples],
        "left_endpoint": left_endpoint,
        "left_node": left_node,
        "right_endpoint": right_endpoint,
        "right_node": right_node,
    }


def _compute_field_scaling() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    scaling: dict[str, Any] = {"percentile": 99.5, "amplitude_cmap": AMPLITUDE_CMAP, "phase_range_rad": [0.0, 2.0 * math.pi], "phase_cmap": PHASE_CMAP, "phase_blank_below_fraction_of_shared_P99_5": 0.01, "field_display_crop": "center 400x400 only when native plane is 2048x2048; no interpolation or complex resize", "groups": {}}
    selected_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for group in ("main", "supplement"):
        samples = _endpoint_sample_records(group)
        pairs = MAIN_FIELD_PAIRS if group == "main" else SUPPLEMENT_FIELD_PAIRS
        scaling["groups"][group] = {}
        for index, pair in enumerate(pairs):
            pair_scaling = _field_scale(group, pair, samples)
            scaling["groups"][group][pair[0]] = pair_scaling
            for sample in samples:
                selected_lookup[(group, sample["source_row_index"])] = sample
    _write_json(OUTPUT / "visualization_scaling.json", scaling)
    return scaling, selected_lookup


def _crop_for_display(field: np.ndarray, meta: Mapping[str, Any], crop_size: int = 400) -> tuple[np.ndarray, tuple[float, float, float, float] | None, str]:
    array = field
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if meta.get("pixel_pitch_m") is None:
        return array, None, "row-major 10×10 ports"
    height, width = array.shape[-2:]
    pitch_y, pitch_x = map(float, meta["pixel_pitch_m"])
    origin_y, origin_x = map(float, meta.get("origin_m", (0.0, 0.0)))
    if (height, width) == (2048, 2048):
        capture.require(crop_size <= height and crop_size <= width, "display crop exceeds native complex field")
        top, left = (height - crop_size) // 2, (width - crop_size) // 2
        array = array[top : top + crop_size, left : left + crop_size]
        height_display, width_display = array.shape[-2:]
        extent = (
            origin_x + (left - width / 2.0) * pitch_x,
            origin_x + (left + width_display - width / 2.0) * pitch_x,
            origin_y + (top - height / 2.0) * pitch_y,
            origin_y + (top + height_display - height / 2.0) * pitch_y,
        )
        return array, tuple(value * 1e3 for value in extent), "center 400×400 crop; native 2048×2048 data unchanged"
    extent = (
        origin_x - width * pitch_x / 2.0,
        origin_x + width * pitch_x / 2.0,
        origin_y - height * pitch_y / 2.0,
        origin_y + height * pitch_y / 2.0,
    )
    return array, tuple(value * 1e3 for value in extent), "native pixels; no resize"


def _add_field_image(ax: plt.Axes, endpoint: str, node: str, sample: Mapping[str, int], *, kind: str, scale: float, threshold: float | None = None, difference: bool = False) -> tuple[Any, dict[str, Any]]:
    field, meta = _read_field(endpoint, node, sample)
    shown, extent, crop_note = _crop_for_display(field, meta)
    if difference:
        display = np.angle(shown)
        mask = np.abs(shown) < float(threshold or 0.0)
        cmap = plt.get_cmap(DIFFERENCE_CMAP).copy()
        cmap.set_bad("white")
        image = ax.imshow(np.ma.array(display, mask=mask), origin="lower", cmap=cmap, vmin=-math.pi, vmax=math.pi, extent=extent, interpolation="nearest", aspect="equal")
        if extent is not None:
            ax.set_xlabel("mm")
            ax.set_ylabel("mm")
        return image, {"metadata": meta, "crop_note": crop_note, "mask_fraction": float(mask.mean())}
    if kind == "amplitude":
        values = np.abs(shown)
        image = ax.imshow(np.minimum(values, scale), origin="lower", cmap=AMPLITUDE_CMAP, vmin=0.0, vmax=scale, extent=extent, interpolation="nearest", aspect="equal")
        if extent is not None:
            ax.set_xlabel("mm")
            ax.set_ylabel("mm")
        return image, {"metadata": meta, "crop_note": crop_note, "clipped_fraction_in_display": float(np.mean(values > scale))}
    phase = np.mod(np.angle(shown), 2.0 * math.pi)
    mask = np.abs(shown) < float(threshold or 0.0)
    cmap = plt.get_cmap(PHASE_CMAP).copy()
    cmap.set_bad("white")
    image = ax.imshow(np.ma.array(phase, mask=mask), origin="lower", cmap=cmap, vmin=0.0, vmax=2.0 * math.pi, extent=extent, interpolation="nearest", aspect="equal")
    if extent is not None:
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
    return image, {"metadata": meta, "crop_note": crop_note, "mask_fraction": float(mask.mean())}


def _make_parameter_figure() -> Path:
    main_a = capture.load_cpu(capture.ENDPOINTS[0].checkpoint)
    main_b = capture.load_cpu(capture.ENDPOINTS[1].checkpoint)
    state_a, state_b = main_a["trainable_parameter_state"], main_b["trainable_parameter_state"]
    capture.require(torch.equal(state_a["encoder.phase"], state_b["encoder.phase"]), "main group does not share identical Encoder phase parameters")
    small = state_a["generator.phase"].reshape(2, 100, 100).numpy()
    large_flat = state_b["generator.phase"].numpy()
    old = large_flat[: capture.OLD_PHASE_COUNT].reshape(2, 100, 100)
    new = large_flat[capture.OLD_PHASE_COUNT :].reshape(2, 400, 400)
    planes = [
        ("Encoder（两臂共享）", state_a["encoder.phase"][0, 0].numpy(), "0.20 × 0.20 mm"),
        ("Small：Generator 原面1", small[0], "0.20 × 0.20 mm"),
        ("Large：原面1", old[0], "0.20 × 0.20 mm"),
        ("Large：新增面 z=0.75 mm", new[0], "0.80 × 0.80 mm"),
        ("Small：Generator 原面2", small[1], "0.20 × 0.20 mm"),
        ("Large：原面2", old[1], "0.20 × 0.20 mm"),
        ("Large：新增面 z=2.25 mm", new[1], "0.80 × 0.80 mm"),
    ]
    fig, axes = plt.subplots(1, len(planes), figsize=(15.2, 3.1), constrained_layout=True)
    for ax, (title, phase, fov) in zip(axes, planes, strict=True):
        value = np.mod(phase, 2.0 * math.pi)
        image = ax.imshow(value, cmap=PHASE_CMAP, vmin=0.0, vmax=2.0 * math.pi, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(fov, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("器件参数相位 ∈ [0, 2π)；每个网格对应实际器件窗口", fontsize=11)
    cbar = fig.colorbar(image, ax=list(axes), orientation="horizontal", fraction=0.06, pad=0.15)
    cbar.set_label("wrapped phase (rad)")
    target = FIGURES / "phase_parameters"
    _save_figure(fig, target.name)

    presentation_fig, presentation_axes = plt.subplots(2, 4, figsize=(15.2, 5.5), constrained_layout=True, squeeze=False)
    presentation_image = None
    for index, (title, phase, fov) in enumerate(planes):
        ax = presentation_axes.flat[index]
        value = np.mod(phase, 2.0 * math.pi)
        presentation_image = ax.imshow(value, cmap=PHASE_CMAP, vmin=0.0, vmax=2.0 * math.pi, origin="lower", interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(fov, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    presentation_axes.flat[-1].set_visible(False)
    presentation_fig.suptitle("器件参数相位 ∈ [0, 2π)；按两行排布以便比较各面", fontsize=12)
    presentation_cbar = presentation_fig.colorbar(presentation_image, ax=presentation_axes.flat[:7].tolist(), orientation="horizontal", fraction=0.055, pad=0.12)
    presentation_cbar.set_label("wrapped phase (rad)")
    _save_figure(presentation_fig, "phase_parameters_presentation")
    return target.with_suffix(".png")


def _parent_group_base(group: str) -> Path:
    return {
        "main": capture.ENDPOINTS[0].report.parent,
        "foreground": capture.ENDPOINTS[2].report.parent,
        "energy": capture.ENDPOINTS[3].report.parent,
    }[group]


def _parent_image(group: str, scenario: str, source_row: int, label: int) -> tuple[np.ndarray, np.ndarray]:
    base = _parent_group_base(group)
    packed_path = base / f"readouts_{scenario}.pt"
    if packed_path.is_file():
        packed = _PARENT_READOUT_CACHE.get(packed_path)
        if packed is None:
            packed = capture.load_cpu(packed_path, mmap=True)
            _PARENT_READOUT_CACHE[packed_path] = packed
        indices = packed["source_indices"].tolist()
        capture.require(indices.count(source_row) == 1, f"{scenario}: saved readout row identity is missing/duplicated")
        position = indices.index(source_row)
        capture.require(int(packed["labels"][position]) == label, f"{scenario}: saved readout label differs")
        return packed["raw_cmos"][position, 0].numpy().copy(), packed["evaluator_view"][position, 0].numpy().copy()
    directory = base / "readouts" / scenario
    for path in sorted(directory.glob("chunk_*.pt")):
        chunk = _PARENT_READOUT_CACHE.get(path)
        if chunk is None:
            chunk = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            _PARENT_READOUT_CACHE[path] = chunk
        source_rows = chunk["source_row_indices"].tolist()
        if source_row in source_rows:
            position = source_rows.index(source_row)
            capture.require(int(chunk["labels"][position]) == label and chunk["roles"][position] == "confirmation", f"{scenario}: chunk row identity/role differs")
            return chunk["raw_cmos"][position, 0].numpy().copy(), chunk["evaluator_view"][position, 0].numpy().copy()
    raise capture.VisualComparisonError(f"{scenario}: saved parent readout omits source row {source_row}")


def _confirmation_source_rows(group: str) -> list[dict[str, int]]:
    population = _population_for(group)
    return [
        {"source_row_index": int(row), "label": int(label)}
        for row, label, role in zip(population["source_row_indices"], population["labels"], population["roles"], strict=True)
        if role == "confirmation"
    ]


def _raw_cmos_scale(group: str, left: str, right: str) -> dict[str, Any]:
    rows = _confirmation_source_rows(group)
    arrays = []
    for row in rows:
        arrays.append(_parent_image(group, left, row["source_row_index"], row["label"])[0])
        arrays.append(_parent_image(group, right, row["source_row_index"], row["label"])[0])
    scale, clipped = _p995(arrays)
    del arrays
    _PARENT_READOUT_CACHE.clear()
    return {"P99_5_raw_cmos": scale, "fraction_above_scale": clipped, "rows": 600}


def _make_cmos_figure(group: str, left: str, right: str, classes: list[int], output_stem: str, title: str) -> tuple[Path, dict[str, Any]]:
    scale = _raw_cmos_scale(group, left, right)
    fig, axes = plt.subplots(len(classes), 4, figsize=(11.4, 2.15 * len(classes)), constrained_layout=True, squeeze=False)
    for row_index, label in enumerate(classes):
        sample = _selected_sample("main" if group == "main" else "supplement", label, 0)
        for column, (scenario, view_kind) in enumerate(((left, "raw"), (left, "view"), (right, "raw"), (right, "view"))):
            raw, view = _parent_image(group, scenario, sample["source_row_index"], label)
            values = raw if view_kind == "raw" else view
            vmax = scale["P99_5_raw_cmos"] if view_kind == "raw" else 1.0
            ax = axes[row_index, column]
            ax.imshow(values, cmap="gray_r", vmin=0.0, vmax=vmax, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            model_name = left if column < 2 else right
            caption = "raw CMOS" if view_kind == "raw" else "evaluator view (28×28)"
            ax.set_title(f"{model_name} · {caption}", fontsize=7)
            if column == 0:
                ax.set_ylabel(f"类别 {label}\nrow {sample['source_row_index']}", fontsize=8)
    fig.suptitle(f"{title}\nraw CMOS共用P99.5标尺；评价器视图固定[0,1]，均不逐图拉伸", fontsize=10)
    target = FIGURES / output_stem
    _save_figure(fig, target)
    _PARENT_READOUT_CACHE.clear()
    return FIGURES / f"{output_stem}.png", scale


def _make_main_amplitude_figure(scaling: Mapping[str, Any]) -> Path:
    samples_by_class = [_selected_sample("main", 1, 0), _selected_sample("main", 7, 0)]
    small_fields, large_fields = _field_pair_endpoints("main")[:2]
    display_indices = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    pairs = [MAIN_FIELD_PAIRS[index - 1] for index in display_indices]
    fig, axes = plt.subplots(4, len(pairs), figsize=(18.4, 5.3), constrained_layout=True, squeeze=False)
    row = 0
    metadata_summary = []
    for sample in samples_by_class:
        for endpoint_name, node_side in ((small_fields, "left"), (large_fields, "right")):
            axes[row, 0].set_ylabel(f"C{sample['label']} · {endpoint_name.replace('_', ' ')}", fontsize=8)
            for column, pair in enumerate(pairs):
                title, left_node, right_node = pair
                node = left_node if node_side == "left" else right_node
                scale = float(scaling["groups"]["main"][title]["P99_5_amplitude"])
                image, record = _add_field_image(axes[row, column], endpoint_name, node, sample, kind="amplitude", scale=scale)
                if row == 0:
                    axes[row, column].set_title(f"{title}\nP99.5={scale:.3g}", fontsize=7)
                if column > 0:
                    axes[row, column].set_ylabel("")
                axes[row, column].tick_params(labelsize=5, length=2)
                metadata_summary.append({"sample_row": sample["source_row_index"], "endpoint": endpoint_name, "node": node, "crop": record["crop_note"], "display_clipped_fraction": record["clipped_fraction_in_display"]})
            row += 1
    fig.suptitle("主对照：类别1、7首个固定样本的复场振幅 |E| 与 OLS 10×10 端口振幅", fontsize=11)
    fig.text(0.5, -0.02, "每列两端共用同一P99.5色阶；2048面只显示原始中心400×400像素，未插值；样本均来自canonical TRAIN确认人口。", ha="center", fontsize=8)
    _write_json(OUTPUT / "amplitude_plot_manifest.json", {"crop_and_clipping": metadata_summary, "node_scale_source": "all 20 selected paired samples from both endpoints"})
    _save_figure(fig, "main_amplitude_ols")

    presentation_indices = [6, 7, 8, 10, 13, 16]
    presentation_pairs = [MAIN_FIELD_PAIRS[index - 1] for index in presentation_indices]
    presentation_fig, presentation_axes = plt.subplots(4, len(presentation_pairs), figsize=(18.4, 6.0), constrained_layout=True, squeeze=False)
    row = 0
    presentation_manifest = []
    for sample in samples_by_class:
        for endpoint_name, node_side in ((small_fields, "left"), (large_fields, "right")):
            presentation_axes[row, 0].set_ylabel(f"C{sample['label']} · {endpoint_name.replace('_', ' ')}", fontsize=9)
            for column, pair in enumerate(presentation_pairs):
                title, left_node, right_node = pair
                node = left_node if node_side == "left" else right_node
                scale = float(scaling["groups"]["main"][title]["P99_5_amplitude"])
                _image, record = _add_field_image(presentation_axes[row, column], endpoint_name, node, sample, kind="amplitude", scale=scale)
                if row == 0:
                    presentation_axes[row, column].set_title(f"{title}\nP99.5={scale:.3g}", fontsize=8)
                if column > 0:
                    presentation_axes[row, column].set_ylabel("")
                presentation_axes[row, column].tick_params(labelsize=6, length=2)
                presentation_manifest.append({"sample_row": sample["source_row_index"], "endpoint": endpoint_name, "node": node, "crop": record["crop_note"], "display_clipped_fraction": record["clipped_fraction_in_display"]})
            row += 1
    presentation_fig.suptitle("主对照振幅：OLS、Generator关键平面与CMOS入射场", fontsize=12)
    presentation_fig.text(0.5, -0.02, "类别1、7首样本；节点依光路顺序。每列共用两端P99.5标尺，中心400×400原生像素窗，无插值。", ha="center", fontsize=8)
    _write_json(OUTPUT / "amplitude_presentation_plot_manifest.json", {"crop_and_clipping": presentation_manifest, "selected_nodes": [pair[0] for pair in presentation_pairs], "node_scale_source": "all 20 selected paired samples from both endpoints"})
    _save_figure(presentation_fig, "main_amplitude_ols_presentation")
    _FIELD_PAYLOAD_CACHE.clear()
    return FIGURES / "main_amplitude_ols.png"


def _phase_difference_data(scaling: dict[str, Any]) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    samples = _endpoint_sample_records("main")
    small_ep, large_ep = "small_A1000", "large_B1000"
    records: dict[tuple[int, str], dict[str, Any]] = {}
    alignment_rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for plane, small_node, pre_node, post_node in (
        ("z0p75", "generator_z0p75_free_reference", "generator_z0p75_before_phase", "generator_z0p75_after_phase"),
        ("z2p25", "generator_z2p25_free_reference", "generator_z2p25_before_phase", "generator_z2p25_after_phase"),
    ):
        all_arrays = []
        for sample in samples:
            all_arrays.extend([
                _read_field(small_ep, small_node, sample)[0],
                _read_field(large_ep, pre_node, sample)[0],
                _read_field(large_ep, post_node, sample)[0],
            ])
        p995, clipped = _p995(all_arrays)
        thresholds[plane] = p995
        scaling.setdefault("phase_same_plane_joint_P99_5", {})[plane] = {"P99_5_amplitude": p995, "clipped_fraction_above_scale": clipped, "fields": len(all_arrays)}
        del all_arrays
        for sample in samples:
            small, small_meta = _read_field(small_ep, small_node, sample)
            before, before_meta = _read_field(large_ep, pre_node, sample)
            after, after_meta = _read_field(large_ep, post_node, sample)
            for candidate_node, candidate, candidate_meta, position in ((pre_node, before, before_meta, "before_device"), (post_node, after, after_meta, "after_device")):
                same_geometry = (
                    small.shape == candidate.shape
                    and np.allclose(small_meta["pixel_pitch_m"], candidate_meta["pixel_pitch_m"], rtol=0, atol=0)
                    and np.allclose(small_meta["origin_m"], candidate_meta["origin_m"], rtol=0, atol=0)
                    and math.isclose(float(small_meta["z_m"]), float(candidate_meta["z_m"]), rel_tol=0, abs_tol=1e-12)
                )
                capture.require(same_geometry, f"{plane}: phase difference attempted across different physical coordinates")
                threshold = 0.01 * p995
                valid = (np.abs(small) >= threshold) & (np.abs(candidate) >= threshold)
                cross = candidate[valid].astype(np.complex128) * np.conj(small[valid].astype(np.complex128))
                capture.require(cross.size > 0, f"{plane}/{sample['source_row_index']}: no shared valid phase pixels")
                offset = float(np.angle(cross.sum()))
                actual_delta = np.angle(candidate * np.conj(small))
                aligned_delta = np.angle((candidate * np.conj(small)) * np.exp(-1j * offset))
                alignment_rows.append({
                    "source_row_index": sample["source_row_index"],
                    "label": sample["label"],
                    "role": "confirmation",
                    "plane": plane,
                    "large_plane_state": position,
                    "small_node": small_node,
                    "large_node": candidate_node,
                    "z_m": float(small_meta["z_m"]),
                    "shared_amplitude_threshold": threshold,
                    "global_offset_removed_rad": offset,
                    "valid_pixel_count": int(cross.size),
                    "phase_delta_definition": "arg(E_large * conj(E_small))",
                })
                if position == "after_device":
                    records[(sample["source_row_index"], plane)] = {
                        "sample": sample,
                        "small": small,
                        "small_meta": small_meta,
                        "before": before,
                        "before_meta": before_meta,
                        "after": after,
                        "after_meta": after_meta,
                        "actual_delta": actual_delta,
                        "aligned_delta": aligned_delta,
                        "offset_rad": offset,
                        "p995": p995,
                        "threshold": threshold,
                    }
    with (OUTPUT / "phase_alignment.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(alignment_rows[0]))
        writer.writeheader()
        writer.writerows(alignment_rows)
    _write_json(OUTPUT / "visualization_scaling.json", scaling)
    return records, alignment_rows


def _make_main_phase_figure(scaling: dict[str, Any]) -> Path:
    data, alignment = _phase_difference_data(scaling)
    samples = [_selected_sample("main", 1, 0), _selected_sample("main", 7, 0)]
    plane_rows = [(sample, plane) for sample in samples for plane in ("z0p75", "z2p25")]
    fig, axes = plt.subplots(len(plane_rows), 5, figsize=(13.0, 8.2), constrained_layout=True, squeeze=False)
    phase_map = diff_map = None
    for index, (sample, plane) in enumerate(plane_rows):
        record = data[(sample["source_row_index"], plane)]
        threshold = record["threshold"]
        column_defs = [
            ("small_A1000", "generator_z0p75_free_reference" if plane == "z0p75" else "generator_z2p25_free_reference", "小：自由传播参照", "phase"),
            ("large_B1000", record["before_meta"]["node_name"], "大：相位前", "phase"),
            ("large_B1000", record["after_meta"]["node_name"], "大：相位后", "phase"),
        ]
        for column, (endpoint, node, title, kind) in enumerate(column_defs):
            image, shown = _add_field_image(axes[index, column], endpoint, node, sample, kind=kind, scale=record["p995"], threshold=threshold)
            phase_map = image
            axes[index, column].set_title(title, fontsize=8)
            axes[index, column].tick_params(labelsize=5, length=2)
        small_display, extent, _ = _crop_for_display(record["small"], record["small_meta"])
        after_display, _, _ = _crop_for_display(record["after"], record["after_meta"])
        actual_display = np.angle(after_display * np.conj(small_display))
        aligned_display = np.angle(actual_display * np.exp(-1j * record["offset_rad"]))
        mask = (np.abs(small_display) < threshold) | (np.abs(after_display) < threshold)
        cmap = plt.get_cmap(DIFFERENCE_CMAP).copy()
        cmap.set_bad("white")
        for column, (values, title) in enumerate(((actual_display, "Δφ 实际"), (aligned_display, f"Δφ 去整体偏移\nδ={record['offset_rad']:+.3f} rad")), start=3):
            image = axes[index, column].imshow(np.ma.array(values, mask=mask), origin="lower", cmap=cmap, vmin=-math.pi, vmax=math.pi, extent=extent, interpolation="nearest", aspect="equal")
            axes[index, column].set_title(title, fontsize=8)
            axes[index, column].set_xlabel("mm")
            axes[index, column].set_ylabel("mm")
            axes[index, column].tick_params(labelsize=5, length=2)
            diff_map = image
        axes[index, 0].set_ylabel(f"C{sample['label']} row {sample['source_row_index']}\n{plane.replace('p', '.')} mm", fontsize=7)
    fig.colorbar(phase_map, ax=axes[:, :3].ravel().tolist(), orientation="horizontal", fraction=0.02, pad=0.07, label="相位 (rad), [0, 2π); 低于1%共同P99.5留白")
    fig.colorbar(diff_map, ax=axes[:, 3:].ravel().tolist(), orientation="horizontal", fraction=0.02, pad=0.07, label="相位差 (rad), [-π, π]")
    fig.suptitle("同一物理z平面上的相位与差值；低振幅区域不参与差值", fontsize=11)
    fig.text(0.5, -0.02, "Δφ=arg(E大·E小*)；去偏移图只移除有效区域的整体相位δ，用于辅助显示，不代表相位正确性。全部为固定主组TRAIN确认样本。", ha="center", fontsize=8)
    _save_figure(fig, "main_phase_comparison")

    presentation_samples = [_selected_sample("main", 1, 0), _selected_sample("main", 7, 0)]
    presentation_plane = "z0p75"
    presentation_fig, presentation_axes = plt.subplots(2, 5, figsize=(15.2, 5.4), constrained_layout=True, squeeze=False)
    phase_map = diff_map = None
    for index, sample in enumerate(presentation_samples):
        record = data[(sample["source_row_index"], presentation_plane)]
        threshold = record["threshold"]
        columns = [
            ("small_A1000", "generator_z0p75_free_reference", "小架构自由传播参照"),
            ("large_B1000", "generator_z0p75_before_phase", "大架构新增相位面前"),
            ("large_B1000", "generator_z0p75_after_phase", "大架构新增相位面后"),
        ]
        for column, (endpoint, node, title) in enumerate(columns):
            image, _shown = _add_field_image(presentation_axes[index, column], endpoint, node, sample, kind="phase", scale=record["p995"], threshold=threshold)
            phase_map = image
            presentation_axes[index, column].set_title(title, fontsize=9)
            presentation_axes[index, column].tick_params(labelsize=6, length=2)
        small_display, extent, _ = _crop_for_display(record["small"], record["small_meta"])
        after_display, _, _ = _crop_for_display(record["after"], record["after_meta"])
        actual_display = np.angle(after_display * np.conj(small_display))
        aligned_display = np.angle(actual_display * np.exp(-1j * record["offset_rad"]))
        mask = (np.abs(small_display) < threshold) | (np.abs(after_display) < threshold)
        cmap = plt.get_cmap(DIFFERENCE_CMAP).copy()
        cmap.set_bad("white")
        for column, (values, title) in enumerate(((actual_display, "Δφ 实际"), (aligned_display, f"Δφ 去整体偏移\nδ={record['offset_rad']:+.3f} rad")), start=3):
            diff_map = presentation_axes[index, column].imshow(np.ma.array(values, mask=mask), origin="lower", cmap=cmap, vmin=-math.pi, vmax=math.pi, extent=extent, interpolation="nearest", aspect="equal")
            presentation_axes[index, column].set_title(title, fontsize=9)
            presentation_axes[index, column].set_xticks([])
            presentation_axes[index, column].set_yticks([])
        presentation_axes[index, 0].set_ylabel(f"类别{sample['label']} · row {sample['source_row_index']}", fontsize=9)
    presentation_fig.colorbar(phase_map, ax=presentation_axes[:, :3].ravel().tolist(), orientation="horizontal", fraction=0.045, pad=0.09, label="相位 (rad), [0, 2π); 低于共同标尺1%留白")
    presentation_fig.colorbar(diff_map, ax=presentation_axes[:, 3:].ravel().tolist(), orientation="horizontal", fraction=0.045, pad=0.09, label="相位差 (rad), [-π, π]")
    presentation_fig.suptitle("z=0.75 mm新增相位面的前后场与同平面相位差", fontsize=12)
    presentation_fig.text(0.5, -0.02, "Δφ=arg(E大·E小*)；相位差仅在两端共同有效振幅区计算，去整体偏移仅辅助显示。", ha="center", fontsize=8)
    _save_figure(presentation_fig, "main_phase_comparison_presentation")
    _FIELD_PAYLOAD_CACHE.clear()
    return FIGURES / "main_phase_comparison.png"


def _make_main_recall_figure(summary: Mapping[str, Any]) -> Path:
    left = summary["comparisons"]["main"]["small_A1000"]["e_d_correct_by_class"]
    right = summary["comparisons"]["main"]["large_B1000"]["e_d_correct_by_class"]
    classes = np.arange(10)
    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    height = 0.36
    bars_left = ax.barh(classes - height / 2, left, height, color="#2878B5", label="小架构 A1000")
    bars_right = ax.barh(classes + height / 2, right, height, color="#E36A2E", label="大架构 B1000")
    ax.bar_label(bars_left, fmt="%d", padding=2, fontsize=7)
    ax.bar_label(bars_right, fmt="%d", padding=2, fontsize=7)
    ax.set_yticks(classes, [str(value) for value in classes])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 35.0)
    ax.set_xlabel("E_D正确数（每类30行）")
    ax.set_ylabel("类别")
    ax.set_title("完整300行确认人口：E_D逐类正确数 / 30")
    ax.grid(axis="x", color="#D8DEE6", linewidth=0.7)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.text(0.5, -0.02, "A1000: 114/300（38.0%）；B1000: 166/300（55.3%）。B类8为18/30，较A的26/30下降。", ha="center", fontsize=8)
    _save_figure(fig, "main_class_recall")
    return FIGURES / "main_class_recall.png"


def _make_foreground_figure(summary: Mapping[str, Any]) -> Path:
    values = summary["foreground_effects"]
    labels = ["类别2", "类别4"]
    a = [values["A5400"]["class_2_mean_q_fg"], values["A5400"]["class_4_mean_q_fg"]]
    b = [values["B5400"]["class_2_mean_q_fg"], values["B5400"]["class_4_mean_q_fg"]]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.1, 3.6), constrained_layout=True)
    bars_a = ax.bar(x - width / 2, a, width, color="#2878B5", label="A5400 原目标对照")
    bars_b = ax.bar(x + width / 2, b, width, color="#E36A2E", label="B5400 前景目标")
    ax.bar_label(bars_a, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars_b, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("前景归一化形态误差 Q_fg（越低越好）")
    ax.set_ylim(0.0, max(a + b) * 1.25)
    ax.set_title("确认人口：类别2、4各30行的类均值")
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.5, 0.94, f"类平衡均值下降 {abs(values['class_balanced_q_fg_relative_change'])*100:.2f}%", ha="center", transform=ax.transAxes, fontsize=9, weight="bold")
    fig.text(0.5, -0.02, "E_D正确数：A5400 237/300 → B5400 242/300。前景改善只适用于固定前景目标、起点和追加700步。", ha="center", fontsize=7.5)
    _save_figure(fig, "foreground_qfg_comparison")
    return FIGURES / "foreground_qfg_comparison.png"


def _make_energy_figure(summary: Mapping[str, Any]) -> Path:
    item = summary["energy_effects"]
    a, b = item["A6700"], item["B6700"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    names = ["A6700", "B6700"]
    colors = ["#2878B5", "#E36A2E"]
    axes[0].bar(names, [summary["comparisons"]["energy"][name]["e_d_correct_total"] for name in names], color=colors)
    axes[0].set_ylim(0, 300)
    axes[0].set_ylabel("E_D正确数 / 300")
    axes[0].set_title("分类读数")
    axes[0].bar_label(axes[0].containers[0], fontsize=9)
    axes[1].bar(names, [a["mean_q"], b["mean_q"]], color=colors)
    axes[1].set_ylabel("完整确认人口平均 Q（越低越好）")
    axes[1].set_title("形态量")
    axes[1].bar_label(axes[1].containers[0], fmt="%.4f", fontsize=8)
    energy_pct = [100.0, item["energy_ratio_B_to_A"] * 100]
    axes[2].bar(names, energy_pct, color=colors)
    axes[2].axhline(95, color="#8A5A00", linestyle="--", linewidth=1, label="95%能量线")
    axes[2].set_ylim(88, 103)
    axes[2].set_ylabel("CMOS积分强度（A6700=100%）")
    axes[2].set_title("能量保留（B/A=95.33%）")
    axes[2].bar_label(axes[2].containers[0], fmt="%.1f%%", fontsize=8)
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(axis="y", color="#D8DEE6", linewidth=0.7)
    fig.suptitle("B6700相对A6700：分类与形态读数上升，但能量约束仍未全部满足", fontsize=10)
    fig.text(0.5, -0.03, f"E_D 243→257（+14）；平均Q下降{abs(item['q_relative_change'])*100:.2f}%；B6700能量仅为A4700起点的{item['energy_ratio_B_to_start']*100:.2f}%，低于95%线，父任务判定仍为 MIXED_OR_INSUFFICIENT。", ha="center", fontsize=7.5)
    _save_figure(fig, "energy_candidate_tradeoff")
    return FIGURES / "energy_candidate_tradeoff.png"


def _atlas_scale(scaling: Mapping[str, Any], group: str, pair_index: int) -> float:
    pair = (MAIN_FIELD_PAIRS if group == "main" else SUPPLEMENT_FIELD_PAIRS)[pair_index]
    return float(scaling["groups"][group][pair[0]]["P99_5_amplitude"])


def _make_field_atlas(scaling: Mapping[str, Any]) -> Path:
    path = FIGURES / "complex_field_atlas.pdf"
    page_records = []
    with PdfPages(path) as pdf:
        for group in ("main", "supplement"):
            left_endpoint, right_endpoint, pairs = _field_pair_endpoints(group)
            samples = _endpoint_sample_records(group)
            for sample in samples:
                fig, axes = plt.subplots(4, len(pairs), figsize=(21.0, 9.5), constrained_layout=True, squeeze=False)
                for side_index, endpoint_name in enumerate((left_endpoint, right_endpoint)):
                    side = "left" if side_index == 0 else "right"
                    for column, (title, left_node, right_node) in enumerate(pairs):
                        node = left_node if side == "left" else right_node
                        scale = _atlas_scale(scaling, group, column)
                        field, meta = _read_field(endpoint_name, node, sample)
                        display, extent, crop_note = _crop_for_display(field, meta)
                        ax_amp = axes[2 * side_index, column]
                        ax_phase = axes[2 * side_index + 1, column]
                        amp = np.abs(display)
                        ax_amp.imshow(np.minimum(amp, scale), cmap=AMPLITUDE_CMAP, vmin=0.0, vmax=scale, origin="lower", extent=extent, interpolation="nearest", aspect="equal")
                        phase = np.mod(np.angle(display), 2.0 * math.pi)
                        phase_map = plt.get_cmap(PHASE_CMAP).copy()
                        phase_map.set_bad("white")
                        mask = amp < 0.01 * scale
                        ax_phase.imshow(np.ma.array(phase, mask=mask), cmap=phase_map, vmin=0.0, vmax=2.0 * math.pi, origin="lower", extent=extent, interpolation="nearest", aspect="equal")
                        for axis in (ax_amp, ax_phase):
                            axis.set_xticks([])
                            axis.set_yticks([])
                        if 2 * side_index == 0:
                            ax_amp.set_title(f"{title}\nP99.5={scale:.2g}", fontsize=6)
                        ax_amp.set_ylabel(f"{endpoint_name}\n|E|", fontsize=6)
                        ax_phase.set_ylabel("phase", fontsize=6)
                        page_records.append({"group": group, "source_row_index": sample["source_row_index"], "label": sample["label"], "endpoint": endpoint_name, "node": node, "crop": crop_note, "masked_fraction": float(mask.mean())})
                group_label = "主组：small_A1000 与 large_B1000" if group == "main" else "补充组：B5400前景目标 与 B6700能量目标（同人口，仅描述性）"
                fig.suptitle(f"{group_label} · 类别{sample['label']} · canonical TRAIN row {sample['source_row_index']}", fontsize=10)
                fig.text(0.5, -0.012, "|E|色阶按每个对应节点的两端20张样本联合P99.5；相位∈[0,2π)，低于1%共享P99.5留白。大场仅中心400×400原生像素；无插值。", ha="center", fontsize=7)
                pdf.savefig(fig, dpi=160)
                plt.close(fig)
                _FIELD_PAYLOAD_CACHE.clear()
    _write_json(OUTPUT / "field_atlas_manifest.json", {"pdf": path.relative_to(OUTPUT).as_posix(), "pages": len(page_records) // (2 * len(MAIN_FIELD_PAIRS)), "samples_per_group": 20, "group_labels": ["main", "supplement"], "plots": page_records})
    return path


def build() -> dict[str, Any]:
    capture.require(OUTPUT.is_dir(), f"captured fields are missing: {OUTPUT}")
    preflight = capture.load_json(OUTPUT / "preflight.json")
    capture.require(preflight.get("status") == "READY", "no-update preflight is not READY")
    field_manifest = capture.load_json(OUTPUT / "field_capture_manifest.json")
    capture.require(len(field_manifest.get("endpoints", [])) == 4, "field capture manifest does not contain four fixed endpoints")
    scaling, _ = _compute_field_scaling()
    _write_coordinate_manifest()
    manifest = capture.load_json(OUTPUT / "field_capture_manifest.json")
    manifest["coordinate_convention"] = {
        "origin_m": [0.0, 0.0],
        "origin_semantics": "center of pixel array; PlaneSpec default origin=(0,0)",
        "pixel_center_formula": "coordinate[i] = origin + (i - (N-1)/2) * pixel_pitch",
        "coordinates_by_endpoint_node": "field_coordinate_metadata.json",
    }
    _write_json(OUTPUT / "field_capture_manifest.json", manifest)

    summary = _write_metrics()
    parameter = _make_parameter_figure()
    amplitude = _make_main_amplitude_figure(scaling)
    phase = _make_main_phase_figure(scaling)
    main_cmos, main_cmos_scale = _make_cmos_figure("main", "A_step1000", "B_step1000", [0, 1, 7, 8], "main_cmos_cases", "主对照CMOS案例")
    main_recall = _make_main_recall_figure(summary)
    foreground_cmos, foreground_cmos_scale = _make_cmos_figure("foreground", "A5400", "B5400", [2, 4], "foreground_cmos_cases", "前景目标候选CMOS案例")
    foreground_metric = _make_foreground_figure(summary)
    energy_metric = _make_energy_figure(summary)
    imaging_scales = {
        "main": main_cmos_scale,
        "foreground": foreground_cmos_scale,
        "energy": _raw_cmos_scale("energy", "A6700", "B6700"),
        "raw_cmos_display_rule": "shared P99.5 across all 300 confirmation rows and both compared endpoints; evaluator view fixed to [0,1]",
    }
    scaling["imaging"] = imaging_scales
    _write_json(OUTPUT / "visualization_scaling.json", scaling)
    _PARENT_READOUT_CACHE.clear()
    atlas = _make_field_atlas(scaling)
    figures = {
        "phase_parameters": parameter,
        "phase_parameters_presentation": FIGURES / "phase_parameters_presentation.png",
        "main_amplitude_ols": amplitude,
        "main_amplitude_ols_presentation": FIGURES / "main_amplitude_ols_presentation.png",
        "main_phase_comparison": phase,
        "main_phase_comparison_presentation": FIGURES / "main_phase_comparison_presentation.png",
        "main_cmos_cases": main_cmos,
        "main_class_recall": main_recall,
        "foreground_cmos_cases": foreground_cmos,
        "foreground_qfg_comparison": foreground_metric,
        "energy_candidate_tradeoff": energy_metric,
        "complex_field_atlas": atlas,
    }
    source_files = {
        "capacity_task_report": capture.ENDPOINTS[0].report,
        "foreground_task_report": capture.ENDPOINTS[2].report,
        "energy_task_report": capture.ENDPOINTS[3].report,
        "canonical_train_data": capture.TRAIN_DATA,
        "fixed_checkpoints": {endpoint.name: endpoint.checkpoint for endpoint in capture.ENDPOINTS},
    }
    _write_json(OUTPUT / "visualization_build_report.json", {
        "status": "FIGURES_BUILT",
        "figure_paths": {name: path.relative_to(OUTPUT).as_posix() for name, path in figures.items()},
        "metrics_csv": "paired_metrics.csv",
        "summary_csv": "metric_summary.csv",
        "summary_json": "metric_summary.json",
        "sources": {key: value.as_posix() if isinstance(value, Path) else {name: path.as_posix() for name, path in value.items()} for key, value in source_files.items()},
        "summary": summary,
        "training_updates": 0,
        "optimizer_instances": 0,
        "new_ols_samples": 0,
        "human_review": 0,
        "selection_rule": "first two rows per class in saved confirmation order; 20 per disjoint group",
        "claim_boundary": "description of fixed canonical TRAIN confirmation rows only; not independent generalization, complete readability, or model adoption",
    })
    _FIELD_PAYLOAD_CACHE.clear()
    _PARENT_READOUT_CACHE.clear()
    return {"status": "FIGURES_BUILT", "figures": {name: path.relative_to(OUTPUT).as_posix() for name, path in figures.items()}, "main_effects": summary["main_effects"], "foreground_effects": summary["foreground_effects"], "energy_effects": summary["energy_effects"]}


def main() -> None:
    result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
