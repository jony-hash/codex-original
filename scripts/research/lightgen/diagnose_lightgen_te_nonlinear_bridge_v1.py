"""ADMA2026 original-condition Te response proxies and G1 input-plane audit.

This is a diagnostic-only evidence consumer. It does not run the G1 forward
model, train a mapper, or infer a 532 nm complex nonlinear transfer function.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pymupdf


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "lightgen_te_nonlinear_bridge.v1"
FIELD_SCHEMA = "lightgen_g1_te_inserted_path_complex_fields.v1"
EXPECTED_DOI = "10.1002/adma.202522820"
EXPECTED_TITLE = "A Phase-Transition-Driven All-Optical Neuron with Sub-Nanosecond Nonlinear Activation"


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(content)


def _write_json_new(path: Path, data: dict[str, Any]) -> None:
    _write_new(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _write_csv_new(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA:
        raise ValueError("Unsupported Te nonlinear bridge configuration")
    if cfg["paper"]["doi"] != EXPECTED_DOI or cfg["paper"]["title"] != EXPECTED_TITLE:
        raise ValueError("Source identity must remain bound to ADMA2026 DOI 10.1002/adma.202522820")
    if cfg["s9"]["pump_wavelength_nm"] != 1530 or cfg["s9"]["probe_wavelength_nm"] != 1520:
        raise ValueError("SI Fig. S9 uses a 1530 nm pump and 1520 nm CW probe")
    if cfg["s9"]["ordinate"] != "normalized probe transmission" or cfg["s9"]["output_phase"] != "not reported":
        raise ValueError("S9 supports normalized intensity response only; phase is unavailable")
    if cfg["s11"]["pump_wavelength_nm"] != 800 or cfg["s11"]["pulse_width_fs"] != 100:
        raise ValueError("SI Fig. S11 identity must remain 800 nm, 100 fs")
    if cfg["s11"]["ordinate"] != "normalized transmission (a.u.)":
        raise ValueError("S11 ordinate must remain normalized transmission")
    if cfg["g1"]["expected_plane_pitch_m"] != [2e-6, 2e-6] or cfg["g1"]["wavelength_m"] != 532e-9:
        raise ValueError("G1 membrane-plane input must use its saved 2 um sampling and 532 nm wavelength")
    panels = (cfg["s9"]["panel_c"], cfg["s9"]["panel_d"], cfg["s9"]["panel_e"], cfg["s11"]["panel_a"])
    for panel in panels:
        for axis_name in ("x_axis", "y_axis", "x_axis_method_b", "y_axis_method_b"):
            axis = panel[axis_name]
            px, vals = axis["pixels"], axis["values"]
            if len(px) != len(vals) or len(px) < 2 or any(not math.isfinite(float(v)) for v in px + vals):
                raise ValueError("Plot calibration must have paired finite pixel and physical coordinates")
            if not all(b > a for a, b in zip(px, px[1:])):
                raise ValueError("Tick pixel coordinates must be ordered left-to-right or top-to-bottom")
            if not (all(b > a for a, b in zip(vals, vals[1:])) or all(b < a for a, b in zip(vals, vals[1:]))):
                raise ValueError("Plot tick values must be strictly monotonic")
        if not all(b > a for axis_name in ("x_axis", "x_axis_method_b")
                   for a, b in zip(panel[axis_name]["values"], panel[axis_name]["values"][1:])):
            raise ValueError("Pump energy / delay / k horizontal axes must increase left-to-right")
        if not all(b < a for axis_name in ("y_axis", "y_axis_method_b")
                   for a, b in zip(panel[axis_name]["values"], panel[axis_name]["values"][1:])):
            raise ValueError("Transmission and extinction ordinate values must increase upward, not downward")
    anchors = cfg["s11"]["panel_a"]["method_b_anchor_y_pixels_by_fluence"]
    if len(anchors) != len(cfg["s11"]["pump_fluences"]):
        raise ValueError("S11 method B needs one source-selected identity anchor per pump fluence")
    if "unknown" not in cfg["paper"]["sample_identities"]["waveguide_device"].casefold():
        raise ValueError("The S9 device active length is not directly sourced and must remain unknown")


def _save_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace a different existing evidence file: {path}")
        return
    with path.open("xb") as f:
        f.write(payload)


def _save_or_verify_rgb(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        from io import BytesIO
        expected = np.asarray(Image.open(BytesIO(payload)).convert("RGB"))
        present = np.asarray(Image.open(path).convert("RGB"))
        if expected.shape != present.shape or not np.array_equal(expected, present):
            raise FileExistsError(f"Refusing to replace a different existing rendered image: {path}")
        return
    with path.open("xb") as f:
        f.write(payload)


def _resolve_sources(cfg: dict[str, Any]) -> dict[str, Path]:
    return {k: _path(v) for k, v in cfg["sources"].items()}


def _plane_entry(meta: dict[str, Any], grid: str, arm: str) -> dict[str, Any]:
    return meta["grids"][grid]["arms"][arm]


def preflight(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    out = _path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = _resolve_sources(cfg)
    for name, p in src.items():
        if not p.is_file():
            raise FileNotFoundError(f"Required source is missing: {name}: {p}")

    pdf = pymupdf.open(src["article_pdf"])
    article_text = " ".join(page.get_text() for page in pdf)
    article_text = re.sub(r"\s+", " ", article_text)
    if EXPECTED_DOI not in article_text or EXPECTED_TITLE not in article_text:
        raise ValueError("The local article PDF does not match the expected title and DOI")
    with zipfile.ZipFile(src["si_docx"]) as zf:
        members = set(zf.namelist())
        required_media = cfg["paper"]["si_figure_media_members"]
        for member in required_media.values():
            if member not in members:
                raise ValueError(f"Required SI image is absent: {member}")
        document_xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
        if "Supplementary" not in document_xml and "Supplementary" not in zf.read("docProps/core.xml").decode("utf-8", errors="replace"):
            raise ValueError("SI DOCX identity text could not be confirmed")
        media_data = {k: zf.read(v) for k, v in required_media.items()}

    for label, payload in media_data.items():
        _save_or_verify(out / "source_pages" / f"adma2026_si_fig_{label}.png", payload)
        with Image.open(out / "source_pages" / f"adma2026_si_fig_{label}.png") as im:
            rgba = im.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            from io import BytesIO
            buf = BytesIO()
            white.convert("RGB").save(buf, format="PNG")
            white_bytes = buf.getvalue()
        _save_or_verify_rgb(out / "source_pages" / f"adma2026_si_fig_{label}_white.png", white_bytes)

    meta = json.loads(src["g1_metadata"].read_text(encoding="utf-8"))
    if meta.get("schema") != FIELD_SCHEMA or meta.get("complex_dtype") != "complex128":
        raise ValueError("Attempt 04 complex-field metadata schema or precision differs")
    if len(meta["grids"]["2048"]["arms"]["direct"]["labels"]) != 100:
        raise ValueError("Attempt 04 must contain all 100 primary sample identities")
    plane = _plane_entry(meta, "2048", "direct")["planes"]["generator_entry"]["plane"]
    if plane["shape"] != [100, 100] or plane["pixel_pitch_m"] != [2e-6, 2e-6] or plane["wavelength_m"] != 532e-9:
        raise ValueError("Attempt 04 generator-entry PlaneSpec differs from 100x100 at 2 um / 532 nm")
    if plane["plane_id"] != "lightgen_generator_entrance":
        raise ValueError("Unexpected saved G1 membrane-plane identity")
    prior_measurement = json.loads(src["g1_prior_budget_measurement"].read_text(encoding="utf-8"))
    prior_review = json.loads(src["g1_prior_budget_independent_review"].read_text(encoding="utf-8"))
    prior_metrics = _read_csv(src["g1_prior_budget_metrics"])
    if prior_measurement.get("status") != "COMPLETE_DIAGNOSTIC_ONLY":
        raise ValueError("Attempt 01 did not publish a complete saved G1 input-budget measurement")
    if prior_review.get("status") != "REVIEW_COMPLETE_DIAGNOSTIC_ONLY":
        raise ValueError("The prior G1 input-budget review is not complete")
    if len(prior_metrics) != 220 or prior_review["g1_membrane_plane"].get("sample_rows_independently_recomputed") != 220:
        raise ValueError("The previously reviewed G1 power budget does not contain the 220-row 2048/4096 population")
    if prior_review["g1_membrane_plane"].get("max_relative_power_error_vs_saved_metric_csv", 1.0) > 1e-12:
        raise ValueError("The previously reviewed G1 power-budget comparison no longer satisfies its saved consumer check")
    expected_g1_attempt = str(Path(cfg["g1"]["attempt"]).as_posix())
    prior_g1_attempt = str(prior_measurement["g1_membrane_plane"]["source_attempt"]).replace("\\", "/")
    if expected_g1_attempt not in prior_g1_attempt.replace("\\", "/"):
        raise ValueError("The reused G1 budget does not point to the configured attempt 04 source")
    if not src["g1_fields"].is_file() or not src["g1_sample_metrics"].is_file():
        raise FileNotFoundError("Attempt 04 complex fields and sample metrics remain required as provenance inputs")

    binding = {
        "config_path": str(Path(config_path).resolve()),
        "config_semantics": cfg,
        "resolved_sources": {name: str(path.resolve()) for name, path in src.items()},
        "verified_article_identity": {"doi": EXPECTED_DOI, "title": EXPECTED_TITLE, "pdf_pages": pdf.page_count},
        "verified_si_media_members": required_media,
        "verified_g1_plane": plane,
    }
    existing_receipt = out / "preflight.json"
    if existing_receipt.exists():
        previous = json.loads(existing_receipt.read_text(encoding="utf-8"))
        if previous.get("input_binding") != binding:
            raise ValueError("Existing preflight receipt is stale for the current config, sources, or physical plane")
        return previous

    result = {
        "schema": f"{SCHEMA}.preflight",
        "status": "READY_DIAGNOSTIC_ONLY",
        "input_binding": binding,
        "source_identity": {"doi": EXPECTED_DOI, "title": EXPECTED_TITLE, "pdf_pages": pdf.page_count,
                            "si_figure_members": required_media},
        "g1_input": {"attempt": str(src["g1_fields"]), "complex_field_metadata": str(src["g1_metadata"]),
                      "primary_samples": 100, "reference_grid_samples": 10, "plane_spec": plane,
                      "budget_reused_from_attempt_01": True,
                      "prior_budget_measurement": str(src["g1_prior_budget_measurement"]),
                      "prior_budget_independent_review": str(src["g1_prior_budget_independent_review"]),
                      "prior_budget_row_count": len(prior_metrics)},
        "nonlinear_g1_admission": {
            "eligible": False,
            "reason": [
                "S9 is a 1530 nm pump / 1520 nm CW probe normalized intensity response, with no output phase.",
                "S11 is an 800 nm, 100 fs normalized pump-probe response for a distinct 50 nm Te / 20 nm SiO2 planar sample.",
                "The bounded 2024-2026 preferred-venue review recorded no source-matched external-intensity to complex-transmission relation at 532 nm and the target CW condition; this is a reviewed literature decision, not a result computed by the preflight.",
            ],
            "decision": "Do not run the conditional frozen G1 nonlinear branch; retain original-condition proxies only.",
        },
        "artifact_scope": "diagnostic-only; no model, mapper, evaluator, training, or hardware execution",
    }
    _write_json_new(out / "preflight.json", result)
    return result


def _fit_axis(axis: dict[str, Any]) -> dict[str, float]:
    x = np.asarray(axis["pixels"], dtype=np.float64)
    y = np.asarray(axis["values"], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return {"slope_value_per_pixel": float(slope), "intercept_value": float(intercept),
            "tick_fit_rms_pixel": float(np.sqrt(np.mean((residual / slope) ** 2))),
            "tick_fit_max_pixel": float(np.max(np.abs(residual / slope))),
            "r_squared": float(1.0 - np.sum(residual**2) / np.sum((y - np.mean(y))**2))}


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def _trace_per_column(image: np.ndarray, panel: dict[str, Any], color: list[int], tolerance: float,
                      exclude_legend_after_x: int | None = None, legend_y_cut: int | None = None) -> list[dict[str, float]]:
    x0, y0, x1, y1 = [int(v) for v in panel["plot_bounds_pixels"]]
    crop = image[y0:y1 + 1, x0:x1 + 1]
    dist = np.linalg.norm(crop.astype(np.float64) - np.asarray(color, dtype=np.float64), axis=2)
    mask = dist <= tolerance
    rows: list[dict[str, float]] = []
    for xi, x_pixel in enumerate(range(x0, x1 + 1)):
        ys = np.flatnonzero(mask[:, xi]) + y0
        if exclude_legend_after_x is not None and legend_y_cut is not None and x_pixel >= exclude_legend_after_x:
            ys = ys[ys < legend_y_cut]
        if ys.size:
            rows.append({"x_pixel": float(x_pixel), "y_median_pixel": float(np.median(ys)),
                         "y_mean_pixel": float(np.mean(ys)), "line_pixel_count": int(ys.size),
                         "y_span_pixel": float(np.max(ys) - np.min(ys))})
    return rows


def _coordinate(rows: list[dict[str, float]], x_axis: dict[str, Any], y_axis: dict[str, Any]) -> list[dict[str, float]]:
    xf, yf = _fit_axis(x_axis), _fit_axis(y_axis)
    result = []
    for row in rows:
        x_val = xf["slope_value_per_pixel"] * row["x_pixel"] + xf["intercept_value"]
        y_med = yf["slope_value_per_pixel"] * row["y_median_pixel"] + yf["intercept_value"]
        y_mean = yf["slope_value_per_pixel"] * row["y_mean_pixel"] + yf["intercept_value"]
        err_px = max(0.5, row["y_span_pixel"] / 2, abs(row["y_median_pixel"] - row["y_mean_pixel"])) + yf["tick_fit_rms_pixel"]
        result.append({**row, "x_value": float(x_val), "y_median_value": float(y_med), "y_mean_value": float(y_mean),
                       "extraction_envelope_y": float(abs(yf["slope_value_per_pixel"]) * err_px)})
    return result


def _coordinate_dual(rows: list[dict[str, float]], panel: dict[str, Any]) -> list[dict[str, float]]:
    method_a = _coordinate(rows, panel["x_axis"], panel["y_axis"])
    method_b = _coordinate(rows, panel["x_axis_method_b"], panel["y_axis_method_b"])
    paired = []
    for a, b in zip(method_a, method_b, strict=True):
        if a["x_pixel"] != b["x_pixel"]:
            raise ValueError("Independent coordinate calibrations lost source-pixel alignment")
        paired.append({**a,
                       "x_value_method_a": a["x_value"], "y_value_method_a": a["y_median_value"],
                       "x_value_method_b": b["x_value"], "y_value_method_b": b["y_median_value"],
                       "x_calibration_difference": b["x_value"] - a["x_value"],
                       "y_calibration_difference": b["y_median_value"] - a["y_median_value"]})
    return paired


def _review_axis_calibrations(panel: dict[str, Any], digitization: dict[str, Any]) -> dict[str, Any]:
    x0, y0, x1, y1 = [float(v) for v in panel["plot_bounds_pixels"]]
    method_a = {axis: _fit_axis(panel[f"{axis}_axis"]) for axis in ("x", "y")}
    method_b = {axis: _fit_axis(panel[f"{axis}_axis_method_b"]) for axis in ("x", "y")}
    results = {}
    for axis, lo, hi in (("x", x0, x1), ("y", y0, y1)):
        pixels = np.linspace(lo, hi, 17, dtype=np.float64)
        slope_a, intercept_a = method_a[axis]["slope_value_per_pixel"], method_a[axis]["intercept_value"]
        slope_b, intercept_b = method_b[axis]["slope_value_per_pixel"], method_b[axis]["intercept_value"]
        mapped_a = slope_a * pixels + intercept_a
        mapped_b = slope_b * pixels + intercept_b
        difference_equivalent_px = np.abs(mapped_a - mapped_b) / abs(slope_a)
        maximum = float(np.max(difference_equivalent_px))
        results[axis] = {"method_a": method_a[axis], "method_b": method_b[axis],
                         "max_difference_equivalent_pixels": maximum}
        if maximum > float(digitization["calibration_pair_max_disagreement_equivalent_pixels"]):
            raise ValueError(f"Independent {axis}-axis calibrations disagree beyond the predeclared pixel envelope")
    return results


def _column_clusters(mask: np.ndarray, x_origin: int, y_origin: int) -> dict[int, list[dict[str, float]]]:
    """Return disjoint vertical color runs by source x pixel; no cross-gap fill."""
    clusters: dict[int, list[dict[str, float]]] = {}
    for xi in range(mask.shape[1]):
        ys = np.flatnonzero(mask[:, xi]) + y_origin
        if not ys.size:
            continue
        groups: list[list[int]] = [[int(ys[0])]]
        for y in ys[1:]:
            if int(y) - groups[-1][-1] <= 1:
                groups[-1].append(int(y))
            else:
                groups.append([int(y)])
        candidates = []
        for group in groups:
            if len(group) < 2:
                continue
            candidates.append({"x_pixel": float(x_origin + xi), "y_pixel": float(np.median(group)),
                               "line_pixel_count": len(group), "y_span_pixel": float(group[-1] - group[0])})
        if candidates:
            clusters[x_origin + xi] = candidates
    return clusters


def _exclusive_s11_clusters(image: np.ndarray, panel: dict[str, Any], fluences: list[dict[str, Any]],
                            digitization: dict[str, Any], *, core_only: bool) -> dict[str, dict[int, list[dict[str, float]]]]:
    x0, y0, x1, y1 = [int(v) for v in panel["plot_bounds_pixels"]]
    crop = image[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
    palette = np.asarray([item["color_rgb"] for item in fluences], dtype=np.float64)
    distance = np.linalg.norm(crop[:, :, None, :] - palette[None, None, :, :], axis=3)
    order = np.argsort(distance, axis=2)
    nearest = order[:, :, 0]
    nearest_distance = np.take_along_axis(distance, nearest[:, :, None], axis=2)[:, :, 0]
    separation = np.take_along_axis(distance, order[:, :, 1:2], axis=2)[:, :, 0] - nearest_distance
    legend = digitization["s11_legend_exclusion_box_pixels"]
    legend_mask = np.zeros(nearest.shape, dtype=bool)
    lx0, ly0, lx1, ly1 = [int(v) for v in legend]
    legend_mask[max(0, ly0 - y0):min(nearest.shape[0], ly1 - y0 + 1),
                max(0, lx0 - x0):min(nearest.shape[1], lx1 - x0 + 1)] = True
    result: dict[str, dict[int, list[dict[str, float]]]] = {}
    for index, item in enumerate(fluences):
        if core_only:
            mask = ((nearest == index) & (distance[:, :, index] <= float(digitization["s11_core_color_tolerance_rgb"])))
        else:
            mask = ((nearest == index)
                    & (nearest_distance <= float(digitization["s11_color_tolerance"]))
                    & (separation >= float(digitization["s11_color_confidence_margin_rgb"])))
        mask &= ~legend_mask
        result[str(item["pJ_per_um2"])] = _column_clusters(mask, x0, y0)
    return result


def _trace_locally_from_anchor(clusters: dict[int, list[dict[str, float]]], x_range: tuple[int, int],
                               anchor_x: int, anchor_y: float, max_center_px: float) -> dict[int, dict[str, float]]:
    anchor_candidates = clusters.get(anchor_x, [])
    if not anchor_candidates:
        raise ValueError("S11 method B source anchor has no core-color pixels")
    anchor = min(anchor_candidates, key=lambda row: abs(row["y_pixel"] - anchor_y))
    if abs(anchor["y_pixel"] - anchor_y) > max_center_px:
        raise ValueError("S11 method B source anchor does not identify its labeled color trace")
    traced = {anchor_x: {**anchor, "trace_segment_id": 0, "trace_origin": "manual_source_anchor"}}
    for direction in (-1, 1):
        previous: dict[str, float] | None = anchor
        previous_previous: dict[str, float] | None = None
        segment_id = 0
        gap_active = False
        xs = range(anchor_x + direction, x_range[0] - 1, -1) if direction < 0 else range(anchor_x + 1, x_range[1] + 1)
        for x_pixel in xs:
            candidates = clusters.get(x_pixel, [])
            chosen = None
            if len(candidates) == 1:
                chosen = candidates[0]
            elif len(candidates) > 1 and previous is not None:
                predicted = previous["y_pixel"]
                if previous_previous is not None:
                    predicted += previous["y_pixel"] - previous_previous["y_pixel"]
                ordered = sorted(candidates, key=lambda row: abs(row["y_pixel"] - predicted))
                best = abs(ordered[0]["y_pixel"] - predicted)
                second = abs(ordered[1]["y_pixel"] - predicted)
                if second - best > 2.0:
                    chosen = ordered[0]
            if chosen is None:
                previous = None
                previous_previous = None
                if not gap_active:
                    segment_id += 1
                gap_active = True
                continue
            traced[x_pixel] = {**chosen, "trace_segment_id": segment_id,
                               "trace_origin": "local_color_continuation" if previous is not None else "new_visible_color_segment"}
            previous_previous, previous = previous, chosen
            gap_active = False
    return traced


def _trace_s11_dual_method(image: np.ndarray, panel: dict[str, Any], fluences: list[dict[str, Any]],
                           digitization: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    method_a = _exclusive_s11_clusters(image, panel, fluences, digitization, core_only=False)
    method_b = _exclusive_s11_clusters(image, panel, fluences, digitization, core_only=True)
    anchors = panel["method_b_anchor_y_pixels_by_fluence"]
    anchor_x = int(panel["method_b_anchor_x_pixel"])
    max_center = float(digitization["s11_max_method_center_difference_px"])
    max_delay = float(digitization["s11_max_calibrated_delay_difference_ps"])
    max_tr = float(digitization["s11_max_calibrated_transmission_difference"])
    curves: dict[str, list[dict[str, Any]]] = {}
    missing: list[dict[str, Any]] = []
    anchor_checks = []
    max_observed = {"center_difference_px": 0.0, "delay_difference_ps": 0.0,
                    "transmission_difference": 0.0}
    x0, _, x1, _ = [int(v) for v in panel["plot_bounds_pixels"]]
    for index, item in enumerate(fluences):
        key = str(item["pJ_per_um2"])
        rows: list[dict[str, Any]] = []
        expected_y = float(anchors[index])
        local_b = _trace_locally_from_anchor(method_b[key], (x0, x1), anchor_x, expected_y, max_center)
        anchor = local_b[anchor_x]
        anchor_checks.append({"pump_fluence_pJ_um2": float(item["pJ_per_um2"]), "x_pixel": anchor_x,
                              "selected_y_pixel": anchor["y_pixel"], "independently_selected_y_pixel": expected_y,
                              "absolute_difference_px": abs(anchor["y_pixel"] - expected_y), "status": "source_anchor_confirmed"})
        segment_id = -1
        previous_x: int | None = None
        for x_pixel in range(x0, x1 + 1):
            a_candidates = method_a[key].get(x_pixel, [])
            b_candidates = [local_b[x_pixel]] if x_pixel in local_b else []
            reason = ""
            if len(a_candidates) != 1 or len(b_candidates) != 1:
                reason = "missing_or_ambiguous_method_a_or_b_trace" if (a_candidates or b_candidates) else "no_visible_color_pixel"
            else:
                a, b = a_candidates[0], b_candidates[0]
                center_delta = abs(a["y_pixel"] - b["y_pixel"])
                a_coord = _coordinate([{"x_pixel": a["x_pixel"], "y_median_pixel": a["y_pixel"],
                                        "y_mean_pixel": a["y_pixel"], "y_span_pixel": a["y_span_pixel"]}],
                                      panel["x_axis"], panel["y_axis"])[0]
                b_coord = _coordinate([{"x_pixel": b["x_pixel"], "y_median_pixel": b["y_pixel"],
                                        "y_mean_pixel": b["y_pixel"], "y_span_pixel": b["y_span_pixel"]}],
                                      panel["x_axis_method_b"], panel["y_axis_method_b"])[0]
                delay_delta = abs(a_coord["x_value"] - b_coord["x_value"])
                transmission_delta = abs(a_coord["y_median_value"] - b_coord["y_median_value"])
                max_observed["center_difference_px"] = max(max_observed["center_difference_px"], center_delta)
                max_observed["delay_difference_ps"] = max(max_observed["delay_difference_ps"], delay_delta)
                max_observed["transmission_difference"] = max(max_observed["transmission_difference"], transmission_delta)
                if center_delta > max_center or delay_delta > max_delay or transmission_delta > max_tr:
                    reason = "independent_trace_or_calibration_difference_exceeds_fixed_envelope"
                else:
                    if previous_x is None or x_pixel != previous_x + 1:
                        segment_id += 1
                    rows.append({"pump_fluence_pJ_um2": float(item["pJ_per_um2"]),
                                 "delay_ps": float(a_coord["x_value"]),
                                 "delay_ps_method_a": float(a_coord["x_value"]),
                                 "delay_ps_method_b": float(b_coord["x_value"]),
                                 "normalized_transmission": float(a_coord["y_median_value"]),
                                 "normalized_transmission_method_a": float(a_coord["y_median_value"]),
                                 "normalized_transmission_method_b": float(b_coord["y_median_value"]),
                                 "method_center_difference_px": float(center_delta),
                                 "method_delay_difference_ps": float(delay_delta),
                                 "method_transmission_difference": float(transmission_delta),
                                 "extraction_envelope_a": float(a_coord["extraction_envelope_y"]),
                                 "extraction_envelope_b": float(b_coord["extraction_envelope_y"]),
                                 "x_pixel": x_pixel, "y_pixel_method_a": float(a["y_pixel"]),
                                 "y_pixel_method_b": float(b["y_pixel"]),
                                 "method_a_line_pixel_count": int(a["line_pixel_count"]),
                                 "method_b_line_pixel_count": int(b["line_pixel_count"]),
                                 "method_b_trace_segment_id": int(b["trace_segment_id"]),
                                 "method_b_trace_origin": b["trace_origin"],
                                 "segment_id": segment_id, "source_figure": "SI Fig. S11a",
                                 "source_role": "digitized curve position; not an independent experimental observation",
                                 "visible_data": True})
                    previous_x = x_pixel
                    continue
            previous_x = None
            missing.append({"pump_fluence_pJ_um2": float(item["pJ_per_um2"]), "x_pixel": x_pixel,
                            "reason": reason})
        curves[key] = rows
    summary = {"anchor_checks": anchor_checks, "max_method_differences": max_observed,
               "fixed_envelope": {"center_difference_px": max_center, "delay_difference_ps": max_delay,
                                  "transmission_difference": max_tr},
               "exclusive_pixel_assignment": True, "overlapping_palette_masks_allowed": False,
               "interpolation_across_segments": False}
    return curves, missing, summary


def _segment_ids(rows: list[dict[str, Any]], x_field: str, *, max_adjacent_gap_px: float = 1.0) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[x_field]))
    segment = -1
    previous: float | None = None
    out = []
    for row in ordered:
        x_value = float(row[x_field])
        if previous is None or x_value - previous > max_adjacent_gap_px:
            segment += 1
        out.append({**row, "segment_id": segment})
        previous = x_value
    return out


def _interpolate_within_segment(rows: list[dict[str, Any]], x_value: float, segment_id: int,
                                x_field: str, y_field: str) -> float:
    segment = sorted((row for row in rows if int(row["segment_id"]) == int(segment_id)),
                     key=lambda row: float(row[x_field]))
    if not segment or x_value < float(segment[0][x_field]) or x_value > float(segment[-1][x_field]):
        raise ValueError("Interpolation query is outside the selected visible curve segment")
    return float(np.interp(x_value, [float(row[x_field]) for row in segment],
                           [float(row[y_field]) for row in segment]))


def _missing_intervals(rows: list[dict[str, Any]], fluence: float, x_field: str,
                       plot_x_range: tuple[int, int], reason_by_x: dict[int, str]) -> list[dict[str, Any]]:
    observed = {int(row[x_field]) for row in rows}
    if not observed:
        return []
    visible_lo = max(plot_x_range[0], min(observed))
    visible_hi = min(plot_x_range[1], max(observed))
    missing_x = [x for x in range(visible_lo, visible_hi + 1) if x not in observed]
    if not missing_x:
        return []
    groups: list[list[int]] = [[missing_x[0]]]
    for x in missing_x[1:]:
        if x == groups[-1][-1] + 1:
            groups[-1].append(x)
        else:
            groups.append([x])
    result = []
    for group in groups:
        reasons = sorted({reason_by_x.get(x, "outside_visible_trace") for x in group})
        result.append({"pump_fluence_pJ_um2": fluence, "x_pixel_start": group[0], "x_pixel_end": group[-1],
                       "pixel_count": len(group), "reason": "; ".join(reasons)})
    return result


def _reject_local_overlay_outliers(rows: list[dict[str, float]], window: int, threshold_px: float) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Flag isolated colored columns inconsistent with their neighboring trace."""
    values = np.asarray([r["y_median_pixel"] for r in rows], dtype=np.float64)
    accepted, excluded = [], []
    for i, row in enumerate(rows):
        lo, hi = max(0, i - window), min(len(rows), i + window + 1)
        neighbors = np.concatenate((values[lo:i], values[i + 1:hi]))
        local = float(np.median(neighbors)) if neighbors.size else float(values[i])
        deviation = abs(float(values[i]) - local)
        enriched = {**row, "local_trace_median_pixel": local, "local_deviation_pixel": deviation}
        if deviation > threshold_px:
            excluded.append({**enriched, "use_status": "excluded_overlay_or_unresolved_pixel",
                             "reason": "isolated column is inconsistent with adjacent curve pixels"})
        else:
            accepted.append({**enriched, "use_status": "accepted_visible_curve"})
    return accepted, excluded


def _components(mask: np.ndarray, origin_x: int, origin_y: int) -> list[dict[str, float]]:
    foreground = np.asarray(mask, dtype=bool)
    seen = np.zeros(foreground.shape, dtype=bool)
    out = []
    height, width = foreground.shape
    for sy, sx in np.argwhere(foreground):
        if seen[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if foreground[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        area = len(points)
        ys, xs = zip(*points, strict=True)
        h, w = max(ys) - min(ys) + 1, max(xs) - min(xs) + 1
        if area < 100 or w > 34 or h > 34:
            continue
        out.append({"x_pixel": float(origin_x + np.mean(xs)),
                    "y_pixel": float(origin_y + np.mean(ys)),
                    "pixel_count": area, "bbox_width_px": int(w), "bbox_height_px": int(h)})
    return out


def _save_source_overlay(image_path: Path, curves: dict[str, list[dict[str, float]]], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(image)
    for name, rows in curves.items():
        if rows:
            ax.scatter([r["x_pixel"] for r in rows],
                       [r["y_median_pixel"] if "y_median_pixel" in r else r["y_pixel"] for r in rows],
                       s=3, label=name)
    ax.set_axis_off()
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_plot_outputs(out: Path, s9c: list[dict[str, Any]],
                        s9_bridge: list[dict[str, Any]], s11: dict[str, list[dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for segment_id in sorted({int(r["segment_id"]) for r in s9c}):
        rows = sorted((r for r in s9c if int(r["segment_id"]) == segment_id), key=lambda row: row["pump_energy_pJ"])
        ax.plot([r["pump_energy_pJ"] for r in rows], [r["tr_norm_relative_to_Eref"] for r in rows],
                color="#7650c7", lw=1.4, label="S9c r(E)/r(Eref)" if segment_id == 0 else "_nolegend_")
    for segment_id in sorted({int(r["s9e_segment_id"]) for r in s9_bridge}):
        rows = sorted((r for r in s9_bridge if int(r["s9e_segment_id"]) == segment_id),
                      key=lambda row: row["pump_energy_pJ"])
        ax.plot([r["pump_energy_pJ"] for r in rows], [r["Tk_relative_to_kref"] for r in rows],
                color="#e6534b", lw=1.1, ls="--",
                label="PWL S9d T(k(E))/T(kref)" if segment_id == min(int(r["s9e_segment_id"]) for r in s9_bridge) else "_nolegend_")
    ax.set(xlabel="Pump energy (pJ)", ylabel="Response normalized at shared Eref (a.u.)",
           title="ADMA2026 SI Fig. S9; original 1530/1520 nm conditions; gapped source segments")
    ax.grid(alpha=.2); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / "s9_response_reconstruction.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ordered_s11 = sorted(s11.items(), key=lambda kv: -float(kv[0]))
    curve_colors = plt.get_cmap("tab10")
    for curve_index, (fluence, rows) in enumerate(ordered_s11):
        curve_color = curve_colors(curve_index % 10)
        segment_ids = sorted({int(row["segment_id"]) for row in rows})
        for index, segment_id in enumerate(segment_ids):
            segment = sorted((row for row in rows if int(row["segment_id"]) == segment_id),
                             key=lambda row: row["delay_ps"])
            ax.plot([r["delay_ps"] for r in segment], [r["normalized_transmission"] for r in segment],
                    color=curve_color, lw=1,
                    label=f"{fluence} pJ/µm²" if index == 0 else "_nolegend_")
    ax.set(xlabel="Delay (ps)", ylabel="Normalized transmission (a.u.)",
           title="S11a visible trace segments · 800 nm, 100 fs")
    ax.set_ylim(.89, 1.015); ax.grid(alpha=.2); ax.legend(ncol=3, fontsize=7); fig.tight_layout()
    fig.savefig(out / "s11_dynamic_response.png", dpi=200); plt.close(fig)


def _sample_metric_rows(src: dict[str, Path]) -> dict[tuple[str, str, int], dict[str, str]]:
    table = {}
    for row in _read_csv(src["g1_sample_metrics"]):
        key = (row["arm"], row["state"], int(row["population_position"]))
        table[key] = row
    return table


def _intensity_metrics(field: np.ndarray, pixel_area: float) -> dict[str, float]:
    intensity = np.abs(np.asarray(field, dtype=np.complex128)) ** 2
    power = float(np.sum(intensity, dtype=np.float64) * pixel_area)
    area = float(intensity.size * pixel_area)
    denominator = float(np.sum(intensity**2, dtype=np.float64) * pixel_area)
    if power <= 0 or denominator <= 0:
        raise ValueError("Saved field has nonpositive power or effective-area denominator")
    effective_area = power**2 / denominator
    return {
        "total_power_W": power,
        "geometric_area_m2": area,
        "geometric_mean_irradiance_W_m2": power / area,
        "peak_irradiance_W_m2": float(np.max(intensity)),
        "p50_irradiance_W_m2": float(np.quantile(intensity, .50)),
        "p95_irradiance_W_m2": float(np.quantile(intensity, .95)),
        "p99_irradiance_W_m2": float(np.quantile(intensity, .99)),
        "effective_area_m2": effective_area,
        "effective_mean_irradiance_W_m2": power / effective_area,
    }


def _measurement(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    out = _path(output_dir)
    preflight_path = out / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("Run preflight first; measurement cannot bypass the input checks")
    if (out / "measurement.json").exists():
        raise FileExistsError("Refusing to overwrite an existing measurement result")
    # Re-check the receipt against the current explicit config, sources and plane.
    preflight_result = preflight(config_path, output_dir)
    if preflight_result.get("status") != "READY_DIAGNOSTIC_ONLY":
        raise ValueError("The current inputs are not covered by a ready diagnostic preflight")
    src = _resolve_sources(cfg)
    s9white = out / "source_pages" / "adma2026_si_fig_s9_white.png"
    s11white = out / "source_pages" / "adma2026_si_fig_s11_white.png"
    img9, img11 = _rgb(s9white), _rgb(s11white)

    digitization = cfg["digitization"]
    c_cfg, d_cfg, e_cfg = cfg["s9"]["panel_c"], cfg["s9"]["panel_d"], cfg["s9"]["panel_e"]
    calibration_reviews = {
        "S9c": _review_axis_calibrations(c_cfg, digitization),
        "S9d": _review_axis_calibrations(d_cfg, digitization),
        "S9e": _review_axis_calibrations(e_cfg, digitization),
        "S11a": _review_axis_calibrations(cfg["s11"]["panel_a"], digitization),
    }
    s9c_pixels = _trace_per_column(img9, c_cfg, c_cfg["trace_color_rgb"], digitization["color_tolerance"])
    s9e_pixels = _trace_per_column(img9, e_cfg, e_cfg["trace_color_rgb"], digitization["color_tolerance"])
    s9c_raw = _coordinate_dual(s9c_pixels, c_cfg)
    s9e_raw = _coordinate_dual(s9e_pixels, e_cfg)
    s9e_kept, s9e_rejected = _reject_local_overlay_outliers(
        s9e_raw, int(digitization["s9e_local_window_columns"]),
        float(digitization["s9e_local_outlier_threshold_px"]))
    occlusions = cfg["s9"]["panel_c"]["annotation_occlusion_energy_windows_pJ"]
    s9c_accepted = [
        row for row in s9c_raw
        if not any(lo <= row["x_value_method_a"] <= hi for lo, hi in occlusions)
    ]
    s9c_segmented = _segment_ids(s9c_accepted, "x_pixel")
    s9e_segmented = _segment_ids(s9e_kept, "x_pixel")
    if len(s9c_segmented) < 100 or len(s9e_segmented) < 100:
        raise ValueError("S9 has insufficient source-visible curve support after explicit occlusion handling")

    dx0, dy0, dx1, dy1 = [int(v) for v in d_cfg["plot_bounds_pixels"]]
    d_crop = img9[dy0:dy1 + 1, dx0:dx1 + 1]
    d_mask = np.linalg.norm(
        d_crop.astype(np.float64) - np.asarray(cfg["s9"]["panel_d"]["marker_color_rgb"], dtype=np.float64),
        axis=2,
    ) <= cfg["digitization"]["marker_color_tolerance"]
    d_centers = _components(d_mask, dx0, dy0)
    d_coordinate_rows = _coordinate_dual([
        {"x_pixel": row["x_pixel"], "y_median_pixel": row["y_pixel"],
         "y_mean_pixel": row["y_pixel"], "y_span_pixel": 1.0}
        for row in d_centers
    ], d_cfg)
    s9d = [
        {"x_pixel": source["x_pixel"], "y_pixel": source["y_pixel"],
         "k": coordinate["x_value_method_a"], "transmission": coordinate["y_value_method_a"],
         "k_method_b": coordinate["x_value_method_b"], "transmission_method_b": coordinate["y_value_method_b"],
         "x_calibration_difference": coordinate["x_calibration_difference"],
         "y_calibration_difference": coordinate["y_calibration_difference"],
         "pixel_count": source["pixel_count"], "bbox_width_px": source["bbox_width_px"],
         "bbox_height_px": source["bbox_height_px"]}
        for source, coordinate in zip(d_centers, d_coordinate_rows, strict=True)
    ]
    if len(s9d) < 5:
        raise ValueError(f"S9d marker extraction found only {len(s9d)} usable source markers")
    s9d_sorted = sorted(s9d, key=lambda row: row["k"])
    k_values = [row["k"] for row in s9d_sorted]
    t_values = [row["transmission"] for row in s9d_sorted]
    marker_slope, _ = np.polyfit(k_values, t_values, 1)
    slope_difference = float(marker_slope - cfg["s9"]["reported_dTr_dk"])
    k_min, k_max = min(k_values), max(k_values)

    def s9c_value_at(energy: float) -> tuple[int, float] | None:
        for segment_id in sorted({int(row["segment_id"]) for row in s9c_segmented}):
            try:
                value = _interpolate_within_segment(
                    s9c_segmented, energy, segment_id, "x_value_method_a", "y_value_method_a")
                return segment_id, value
            except ValueError:
                continue
        return None

    common_candidates = []
    bridge_exclusions = []
    for row in sorted(s9e_segmented, key=lambda item: item["x_value_method_a"]):
        energy = float(row["x_value_method_a"])
        k_value = float(row["y_value_method_a"])
        if not k_min <= k_value <= k_max:
            bridge_exclusions.append({
                "pump_energy_pJ": energy, "k_inferred_author": k_value, "segment_id": row["segment_id"],
                "reason": "S9e k(E) lies outside the source-supported S9d marker domain; no extrapolation",
            })
            continue
        s9c_match = s9c_value_at(energy)
        if s9c_match is None:
            bridge_exclusions.append({
                "pump_energy_pJ": energy, "k_inferred_author": k_value, "segment_id": row["segment_id"],
                "reason": "energy falls in an S9c annotation or source-pixel gap; no cross-segment interpolation",
            })
            continue
        common_candidates.append((row, s9c_match[0], s9c_match[1]))
    if not common_candidates:
        raise ValueError("S9c and S9e have no common source-supported energy and k domain")

    anchor_row, s9c_anchor_segment, r_reference = min(
        common_candidates, key=lambda item: float(item[0]["x_value_method_a"]))
    e_reference = float(anchor_row["x_value_method_a"])
    k_reference = float(anchor_row["y_value_method_a"])
    t_reference = float(np.interp(k_reference, k_values, t_values))
    if r_reference == 0 or t_reference == 0:
        raise ValueError("S9 shared normalization anchor must have nonzero source-supported values")

    s9c_rows = []
    s9c_segment_by_pixel = {int(row["x_pixel"]): row for row in s9c_segmented}
    for row in s9c_raw:
        x = int(row["x_pixel"])
        accepted = s9c_segment_by_pixel.get(x)
        is_occluded = any(lo <= row["x_value_method_a"] <= hi for lo, hi in occlusions)
        s9c_rows.append({
            "pump_energy_pJ": row["x_value_method_a"],
            "pump_energy_pJ_method_b": row["x_value_method_b"],
            "tr_norm_original_initial_reference": row["y_value_method_a"],
            "tr_norm_method_b": row["y_value_method_b"],
            "tr_norm_relative_to_Eref": (
                float(row["y_value_method_a"]) / r_reference if accepted is not None else ""),
            "extraction_envelope_method_a": row["extraction_envelope_y"],
            "x_calibration_difference": row["x_calibration_difference"],
            "y_calibration_difference": row["y_calibration_difference"],
            "x_pixel": row["x_pixel"], "y_pixel": row["y_median_pixel"],
            "line_pixel_count": row["line_pixel_count"],
            "segment_id": accepted["segment_id"] if accepted is not None else "",
            "source_figure": "SI Fig. S9c",
            "source_role": "digitized curve pixel coordinate; not an independent experimental observation",
            "use_status": "excluded_annotation_occlusion" if is_occluded else "accepted_visible_curve",
        })

    s9e_segment_by_pixel = {int(row["x_pixel"]): row for row in s9e_segmented}
    rejected_by_pixel = {int(row["x_pixel"]): row for row in s9e_rejected}
    s9e_rows = []
    for row in s9e_raw:
        x = int(row["x_pixel"])
        accepted = s9e_segment_by_pixel.get(x)
        rejected = rejected_by_pixel.get(x)
        k_value = float(row["y_value_method_a"])
        in_s9d_domain = k_min <= k_value <= k_max
        s9e_rows.append({
            "pump_energy_pJ": row["x_value_method_a"],
            "pump_energy_pJ_method_b": row["x_value_method_b"],
            "k_inferred_author": k_value,
            "k_method_b": row["y_value_method_b"],
            "extraction_envelope_method_a": row["extraction_envelope_y"],
            "x_calibration_difference": row["x_calibration_difference"],
            "y_calibration_difference": row["y_calibration_difference"],
            "x_pixel": row["x_pixel"], "y_pixel": row["y_median_pixel"],
            "line_pixel_count": row["line_pixel_count"],
            "segment_id": accepted["segment_id"] if accepted is not None else "",
            "source_figure": "SI Fig. S9e",
            "source_role": "author-inferred effective k(E); derived from transmission and FDTD, not direct ellipsometry",
            "use_status": (
                "excluded_overlay_or_unresolved_pixel" if rejected is not None else
                "accepted_visible_curve" if accepted is not None else "unresolved_gap"
            ),
            "within_S9d_marker_domain": in_s9d_domain,
        })

    bridge = []
    for row, c_segment, r_value in common_candidates:
        energy = float(row["x_value_method_a"])
        k_value = float(row["y_value_method_a"])
        t_value = float(np.interp(k_value, k_values, t_values))
        bridge.append({
            "pump_energy_pJ": energy,
            "pump_energy_pJ_method_b": float(row["x_value_method_b"]),
            "s9e_segment_id": int(row["segment_id"]),
            "s9c_segment_id": c_segment,
            "k_inferred_author": k_value,
            "T_k_absolute_FDTD_marker_interpolation": t_value,
            "T_k_reference_at_shared_Eref": t_reference,
            "Tk_relative_to_kref": t_value / t_reference,
            "tr_norm_S9c_original_initial_reference": r_value,
            "tr_norm_relative_to_Eref": r_value / r_reference,
            "shared_reference_energy_pJ": e_reference,
            "shared_reference_k": k_reference,
            "residual_composite_minus_S9c": t_value / t_reference - r_value / r_reference,
            "interpolation": "piecewise linear within S9d marker domain and one visible S9c/S9e segment",
            "independent_validation": False,
        })

    for row in s9e_segmented:
        energy, k_value = float(row["x_value_method_a"]), float(row["y_value_method_a"])
        if k_min <= k_value <= k_max and s9c_value_at(energy) is not None:
            continue
        if not any(float(item["pump_energy_pJ"]) == energy for item in bridge_exclusions):
            bridge_exclusions.append({
                "pump_energy_pJ": energy, "k_inferred_author": k_value, "segment_id": row["segment_id"],
                "reason": "not admitted to composite response because source support is incomplete",
            })

    s9d_rows = [
        {**row,
         "source_role": "digitized FDTD marker coordinate; marker fit is a figure-derived model check",
         "paper_annotated_dTr_dk": cfg["s9"]["reported_dTr_dk"]}
        for row in s9d
    ]

    s11cfg = cfg["s11"]["panel_a"]
    s11curves, s11_missing_pixels, s11_summary = _trace_s11_dual_method(
        img11, s11cfg, cfg["s11"]["pump_fluences"], digitization)
    all_s11 = [row for rows in s11curves.values() for row in rows]
    missing_intervals = []
    x0, _, x1, _ = [int(v) for v in s11cfg["plot_bounds_pixels"]]
    for fluence in cfg["s11"]["pump_fluences"]:
        value = float(fluence["pJ_per_um2"])
        key = str(fluence["pJ_per_um2"])
        reason_by_x = {
            int(row["x_pixel"]): row["reason"]
            for row in s11_missing_pixels if float(row["pump_fluence_pJ_um2"]) == value
        }
        missing_intervals.extend(
            _missing_intervals(s11curves[key], value, "x_pixel", (x0, x1), reason_by_x)
        )
    curve_identity_checks = s11_summary["anchor_checks"]
    for value in cfg["s11"]["pump_fluences"]:
        key = str(value["pJ_per_um2"])
        rows = s11curves[key]
        if rows:
            s11_summary.setdefault("curve_domains", {})[key] = {
                "accepted_rows": len(rows),
                "segment_count": len({int(row["segment_id"]) for row in rows}),
                "visible_delay_range_ps": [min(row["delay_ps"] for row in rows), max(row["delay_ps"] for row in rows)],
            }
        else:
            s11_summary.setdefault("curve_domains", {})[key] = {
                "accepted_rows": 0, "segment_count": 0, "visible_delay_range_ps": None,
            }

    # G1 power and irradiance are re-consumed from the already reviewed attempt 01 result;
    # no complex-field propagation or field-wide recalculation is repeated here.
    prior_measurement = json.loads(src["g1_prior_budget_measurement"].read_text(encoding="utf-8"))
    prior_metrics = _read_csv(src["g1_prior_budget_metrics"])
    g1_membrane = dict(prior_measurement["g1_membrane_plane"])
    g1_membrane.update({
        "budget_reused_from_attempt": str(src["g1_prior_budget_measurement"]),
        "per_sample_metrics_reused_from": str(src["g1_prior_budget_metrics"]),
        "rows_reused": len(prior_metrics),
        "independent_review_path": str(src["g1_prior_budget_independent_review"]),
        "recomputed_in_attempt_02": False,
        "physical_facet_mapping_status": "unresolved; numeric 100x100, 2um plane is not the 125/127um target SMF facet",
    })

    s9_bridge_rms = float(np.sqrt(np.mean([row["residual_composite_minus_S9c"] ** 2 for row in bridge])))
    s9_bridge_max = float(np.max(np.abs([row["residual_composite_minus_S9c"] for row in bridge])))
    s9_k_span = [min(row["k_inferred_author"] for row in bridge), max(row["k_inferred_author"] for row in bridge)]
    s9c_values = [row["pump_energy_pJ"] for row in s9c_rows if row["use_status"] == "accepted_visible_curve"]
    s9e_values = [row["pump_energy_pJ"] for row in s9e_rows if row["use_status"] == "accepted_visible_curve"]
    s11_ranges = {
        key: (None if not rows else [min(row["delay_ps"] for row in rows), max(row["delay_ps"] for row in rows)])
        for key, rows in s11curves.items()
    }

    s9c_overlay = [{"x_pixel": row["x_pixel"], "y_median_pixel": row["y_pixel"]} for row in s9c_rows]
    s9e_overlay = [
        {"x_pixel": row["x_pixel"], "y_median_pixel": row["y_pixel"]}
        for row in s9e_rows if row["use_status"] == "accepted_visible_curve"
    ]
    s9d_overlay = [{"x_pixel": row["x_pixel"], "y_pixel": row["y_pixel"]} for row in s9d_rows]
    _write_csv_new(out / "s9c_normalized_probe_transmission.csv", s9c_rows)
    _write_csv_new(out / "s9d_simulated_T_of_k.csv", s9d_rows)
    _write_csv_new(out / "s9e_author_inferred_k_of_energy.csv", s9e_rows)
    _write_csv_new(out / "s9_normalized_response_bridge.csv", bridge)
    _write_csv_new(out / "s9_bridge_exclusions.csv", bridge_exclusions,
                   ["pump_energy_pJ", "k_inferred_author", "segment_id", "reason"])
    _write_csv_new(out / "s11a_delay_curves.csv", all_s11)
    _write_csv_new(out / "s11a_missing_intervals.csv", missing_intervals,
                   ["pump_fluence_pJ_um2", "x_pixel_start", "x_pixel_end", "pixel_count", "reason"])
    _write_csv_new(out / "s11a_curve_identity_checks.csv", curve_identity_checks)
    _save_source_overlay(s9white, {"S9c source trace": s9c_overlay, "S9e author-inferred k(E)": s9e_overlay,
                                   "S9d FDTD markers": s9d_overlay}, out / "s9_digitization_overlay.png")
    _save_source_overlay(s11white, {
        f"{key} pJ/um2": [{"x_pixel": row["x_pixel"], "y_pixel": row["y_pixel_method_a"]} for row in rows]
        for key, rows in s11curves.items()
    }, out / "s11a_digitization_overlay.png")
    _write_plot_outputs(out, [row for row in s9c_rows if row["segment_id"] != ""], bridge, s11curves)

    measurement = {
        "schema": f"{SCHEMA}.measurement",
        "status": "PENDING_FINAL_OUTPUT_VALIDATION",
        "source": {
            "doi": EXPECTED_DOI, "article": str(src["article_pdf"]), "si": str(src["si_docx"]),
            "figure_locations": {"S9c": "SI Fig. S9c", "S9d": "SI Fig. S9d",
                                 "S9e": "SI Fig. S9e", "S11a": "SI Fig. S11a"},
            "sample_identity": cfg["paper"]["sample_identities"],
            "S9_waveguide_device_active_length": "unknown; no directly reviewed source location states this length for the S9 response device",
        },
        "s9": {
            "pump_wavelength_nm": cfg["s9"]["pump_wavelength_nm"],
            "pump_temporal_condition": cfg["s9"]["pump_temporal_condition"],
            "probe_wavelength_nm": cfg["s9"]["probe_wavelength_nm"],
            "probe_temporal_condition": cfg["s9"]["probe_temporal_condition"],
            "ordinate": cfg["s9"]["ordinate"],
            "absolute_phase": cfg["s9"]["output_phase"],
            "output_role": "normalized intensity response r_probe(E_pump), not absolute output energy or complex transmission",
            "digitized_pixel_positions_are_independent_experiments": False,
            "s9c_original_normalization": "normalized to paper's initial transmission reference",
            "shared_comparison_reference": {"energy_pJ": e_reference, "k_author_inferred": k_reference,
                                            "s9c_at_reference": r_reference, "FDTD_T_at_reference": t_reference},
            "s9c_visible_energy_domain_pJ": [float(min(s9c_values)), float(max(s9c_values))],
            "s9e_visible_energy_domain_pJ": [float(min(s9e_values)), float(max(s9e_values))],
            "composite_source_supported_energy_domain_pJ": [float(min(row["pump_energy_pJ"] for row in bridge)),
                                                            float(max(row["pump_energy_pJ"] for row in bridge))],
            "s9d_detected_marker_count": len(s9d_rows),
            "s9d_marker_fit_dTr_dk": float(marker_slope),
            "s9d_paper_annotated_dTr_dk": float(cfg["s9"]["reported_dTr_dk"]),
            "s9d_marker_minus_annotation": slope_difference,
            "s9d_relative_slope_difference": float(abs(slope_difference) / abs(cfg["s9"]["reported_dTr_dk"])),
            "s9d_main_reconstruction_method": "piecewise-linear interpolation through digitized S9d FDTD marker coordinates, without k-domain extrapolation",
            "s9d_annotated_slope_role": "secondary figure-consistency check only; not used to replace marker data",
            "k_role": "S9e author-inferred effective parameter derived from S9c transmission and FDTD; not direct ellipsometry and not independent validation",
            "anchor": {"energy_pJ": e_reference, "k": k_reference,
                       "normalization_identity_r_S9c": 1.0, "normalization_identity_Tk": 1.0,
                       "s9c_segment_id": s9c_anchor_segment, "s9e_segment_id": int(anchor_row["segment_id"])},
            "composite_trace_rms_difference": s9_bridge_rms,
            "composite_trace_max_abs_difference": s9_bridge_max,
            "composite_trace_difference_scope": "processing-consistency comparison between dependent figure products; not independent experimental validation or a pass threshold",
            "k_span_within_common_domain": [float(min(s9_k_span)), float(max(s9_k_span))],
            "bridge_exclusion_count": len(bridge_exclusions),
            "annotation_occlusion_windows_pJ": occlusions,
            "digitization_sample_count_is_not_experiment_n": True,
            "coordinate_calibrations": calibration_reviews,
        },
        "s11": {
            "pump_wavelength_nm": cfg["s11"]["pump_wavelength_nm"],
            "pump_pulse_width_fs": cfg["s11"]["pulse_width_fs"],
            "repetition_rate_hz": cfg["s11"]["repetition_rate_hz"],
            "sample_stack": cfg["s11"]["sample_stack"],
            "probe_fluence_pJ_um2": cfg["s11"]["probe_fluence_pJ_um2"],
            "polarization": cfg["s11"]["polarization"],
            "pump_fluences_pJ_um2": [float(item["pJ_per_um2"]) for item in cfg["s11"]["pump_fluences"]],
            "visible_delay_range_by_fluence_ps": s11_ranges,
            "coordinate_calibrations": calibration_reviews["S11a"],
            "trace_methods": s11_summary,
            "accepted_pixel_positions_per_fluence": {key: len(rows) for key, rows in s11curves.items()},
            "missing_interval_count": len(missing_intervals),
            "missing_intervals_csv": "s11a_missing_intervals.csv",
            "curve_identity_checks_csv": "s11a_curve_identity_checks.csv",
            "color_assignment": "exclusive nearest-palette winner with a predeclared confidence margin; no pixel can belong to multiple fluence masks",
            "interpolation_policy": "no interpolation is performed for saved curves; optional interpolation is only accepted within one saved segment and rejects gaps/out-of-domain queries",
            "red_50_percent_recovery_fit": "extrapolation beyond measured data; excluded from measured recovery conclusions",
            "fluence_conversion": "not converted to peak irradiance because pulse temporal profile is not specified",
            "digitization_pixels_are_independent_experiments": False,
        },
        "g1_membrane_plane": g1_membrane,
        "g1_nonlinear_admission": preflight_result["nonlinear_g1_admission"],
        "model_design": {
            "paper_condition_proxy": "S9 normalized probe response and S11 source-visible delay segments remain separate, intensity-only proxies under their respective paper sample/wavelength/time conditions",
            "532nm_target": "not admitted for quantitative CW G1 prediction",
            "required_g1_comparison_if_future_source_admitted": [
                "saved bare G1 path",
                "explicitly parameterized nonlinear Te hypothesis with intensity scale and response form",
                "equal-output-power spatially uniform linear attenuation control",
            ],
            "unresolved_parameters": [
                "mapping from the numeric 100x100 2um generator plane to the physical 125/127um-pitch SMF facet",
                "532nm CW external irradiance to absolute Te transmission relation for the matched sputtered sample",
                "complex output phase or an independently justified phase assumption",
                "membrane-plane reference conversion, temporal envelope, mode overlap, reflection and internal-field basis",
            ],
            "g1_nonlinear_forward_run": False,
            "training_updates": 0,
            "hardware_work": False,
        },
        "figures": {"s9_overlay": "s9_digitization_overlay.png",
                    "s11_overlay": "s11a_digitization_overlay.png",
                    "s9_response": "s9_response_reconstruction.png",
                    "s11_response": "s11_dynamic_response.png"},
        "reused_prior_result": {
            "attempt_01_g1_budget": str(src["g1_prior_budget_measurement"]),
            "attempt_01_independent_review": str(src["g1_prior_budget_independent_review"]),
            "attempt_04_complex_fields_reprocessed": False,
        },
    }

    expected_outputs = [
        "preflight.json", "s9c_normalized_probe_transmission.csv", "s9d_simulated_T_of_k.csv",
        "s9e_author_inferred_k_of_energy.csv", "s9_normalized_response_bridge.csv",
        "s9_bridge_exclusions.csv", "s11a_delay_curves.csv", "s11a_missing_intervals.csv",
        "s11a_curve_identity_checks.csv", "s9_digitization_overlay.png",
        "s11a_digitization_overlay.png", "s9_response_reconstruction.png",
        "s11_dynamic_response.png", "report.md",
    ]
    report_text = _render_report(measurement)
    _write_new(out / "report.md", report_text)
    missing_files = [name for name in expected_outputs if not (out / name).is_file() or (out / name).stat().st_size == 0]
    if missing_files:
        raise ValueError(f"Required attempt 02 evidence is incomplete: {missing_files}")
    for image_name in ("s9_digitization_overlay.png", "s11a_digitization_overlay.png",
                       "s9_response_reconstruction.png", "s11_dynamic_response.png"):
        with Image.open(out / image_name) as picture:
            picture.verify()
    saved_s11_rows = _read_csv(out / "s11a_delay_curves.csv")
    expected_fluences = {str(item["pJ_per_um2"]) for item in cfg["s11"]["pump_fluences"]}
    found_fluences = {row["pump_fluence_pJ_um2"] for row in saved_s11_rows}
    if found_fluences != expected_fluences:
        raise ValueError("Saved S11 curve consumer does not cover all nine configured fluence identities")
    for row in all_s11:
        if (float(row["method_center_difference_px"]) > digitization["s11_max_method_center_difference_px"]
                or float(row["method_delay_difference_ps"]) > digitization["s11_max_calibrated_delay_difference_ps"]
                or float(row["method_transmission_difference"]) > digitization["s11_max_calibrated_transmission_difference"]):
            raise ValueError("A saved S11 row exceeds the predeclared two-method extraction envelope")
    for key, rows in s11curves.items():
        for segment_id in {int(row["segment_id"]) for row in rows}:
            segment = sorted((row for row in rows if int(row["segment_id"]) == segment_id),
                             key=lambda row: row["x_pixel"])
            if any(float(b["x_pixel"]) - float(a["x_pixel"]) != 1.0 for a, b in zip(segment, segment[1:])):
                raise ValueError(f"S11 curve {key} bridges a source-pixel gap inside segment {segment_id}")
    if abs(bridge[0]["tr_norm_relative_to_Eref"] - 1.0) > 1e-12 or abs(bridge[0]["Tk_relative_to_kref"] - 1.0) > 1e-12:
        raise ValueError("S9 common-reference normalization did not consume the shared anchor")
    measurement["output_consumer_verification"] = {
        "status": "CHECKED_FROM_SAVED_PRODUCTION_OUTPUTS",
        "required_files_present_and_nonempty": expected_outputs,
        "curve_identity_rows": len(curve_identity_checks),
        "s11_saved_row_count": len(saved_s11_rows),
        "s11_fluence_identity_set_matches_config": True,
        "s11_all_saved_rows_inside_predeclared_envelope": True,
        "s11_segments_do_not_bridge_missing_pixels": True,
        "s9_shared_reference_consumed_by_both_curves": True,
        "plots_opened_and_verified": True,
        "measurement_status_published_after_required_artifacts": True,
    }
    measurement["status"] = "COMPLETE_DIAGNOSTIC_ONLY"
    _write_json_new(out / "measurement.json", measurement)
    return measurement


def _render_report(m: dict[str, Any]) -> str:
    s9, s11, g1 = m["s9"], m["s11"], m["g1_membrane_plane"]
    domain_lines = "\n".join(
        f"- {fluence} pJ/µm²: {data['accepted_rows']} accepted pixels, {data['segment_count']} visible segments, "
        f"delay domain {data['visible_delay_range_ps'] or 'unavailable'} ps."
        for fluence, data in s11["trace_methods"]["curve_domains"].items()
    )
    reasons = " ".join(m["g1_nonlinear_admission"]["reason"])
    return f"""# Te nonlinear bridge diagnostic, corrected attempt 02

## 结果与状态

本报告先于最终状态文件写出。只有同目录 measurement.json 最终记录 COMPLETE_DIAGNOSTIC_ONLY，才表示所列文件均生成并通过本次输出消费者检查。attempt 01 原始内容保留，不能再作为 S11 曲线准入依据。

本次只纠正 S9/S11 图像提取、归一化和缺失段管理，复用已经独立复审的 G1 膜前功率预算。没有运行 G1 非线性传播、训练或实物工作。

## 科学认识

- **S9 原条件经验响应：**{s9['pump_wavelength_nm']} nm、{s9['pump_temporal_condition']} 泵浦和 {s9['probe_wavelength_nm']} nm CW 探测。输出是论文初始参考下的归一化探测透射；比较曲线统一到 Eref={s9['anchor']['energy_pJ']:.5f} pJ、作者反推 k={s9['anchor']['k']:.6f}。S9d 主重建使用可辨认 FDTD 标记间的分段线性 T(k)，不使用论文标注斜率替代标记；标记斜率与论文标注的相对差为 {100*s9['s9d_relative_slope_difference']:.2f}%。S9e 的 k(E) 来自透射和 FDTD 反推，因此复合残差不是独立实验验证。
- **S11 原条件延迟曲线：**800 nm、100 fs、1 kHz；样品为 {s11['sample_stack']}，固定探测能流密度 {s11['probe_fluence_pJ_um2']} pJ/µm²，泵浦/探测偏振{s11['polarization']}。只保存经过颜色排他识别和两种中心线/坐标复核的可见像素段；未连接缺失像素，不拟合统一动力学，也不使用图 S11b 外推恢复时间。
{domain_lines}
- 两个文献代理没有共同的 532 nm CW 光强—复透射合同。S9 的 S9 器件长度在已审阅来源位置中未被直接说明，记录为未知；不能沿用其他图的 3.3 µm。
- G1 膜面输入预算从 attempt 01 直接重用：{g1['rows_reused']} 条既有记录及其独立复审。该数值入口为 100×100、2 µm 采样的空气侧参考平面，不是 125/127 µm 节距真实光纤端面强度。

## 目标缺口

仍缺少与目标磁控溅射样品、层序、532 nm CW 和绝对外部入射强度匹配的 Te 非线性输入—输出关系；相位、局部端面参考平面到 SMF 阵列的映射、模式重叠、反射反馈和实际功率标定亦未建立。现有 S9/S11 结果只能作为原论文条件下的归一化强度代理。

## 工程完成与验证

- S11 每个源像素只归属到最近的一个色板类别；颜色距离和置信边界不满足条件的像素留空。手工选定的 x=258 px 标签锚点分别复核九条曲线身份，重点区分 2.23 与 1.91 pJ/µm²。
- S11 两种颜色提取中心、独立坐标标定的最大差异分别为 {s11['trace_methods']['max_method_differences']['center_difference_px']:.3f} px、{s11['trace_methods']['max_method_differences']['delay_difference_ps']:.4f} ps 和 {s11['trace_methods']['max_method_differences']['transmission_difference']:.6f} a.u.；阈值预先保存在配置中。
- S9c 与 S9e 使用共同有效能量锚点。无法在同一可见曲线段内插值、或 k 超出 S9d 标记范围的行被登记为缺口/排除，不外推。逐列像素仅表示图像数字化位置，不表示独立实验样本。
- attempt 01 中 G1 预算的 220 行复场/功率比较最大相对误差为 {g1['2048_power_reproduction_max_relative_error_vs_saved_csv']:.3e}；本次没有重新计算全链场或材料响应。
- 必需图像、CSV、叠加图与曲线图经文件存在性、非空、PNG读取、曲线身份、双方法容差、段内像素连续性及 S9 共同归一化消费者核对。独立复审记录与文献参数适用性结论另见研究包新增文件。

## 决策影响与下一步

本轮准入 S9 的 1530/1520 nm 归一化强度响应经验代理和 S11 的 800 nm 超快延迟曲线段，限于各自论文样品及条件。它们不构成 532 nm CW 的 Te 薄膜预测，也不允许复场接入 G1。{reasons}

后续建议是先按本包模型设计表明确列出的假设、外部输入量纲和等损耗控制，待获得匹配来源后另行设计 G1 传播。该建议不是当前运行授权。本轮停止于可靠的原条件代理、G1 输入合同和训练断点说明。
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "measure"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        preflight(args.config, args.output_dir)
    else:
        _measurement(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
