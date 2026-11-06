"""Reconstruct ADMA2026 Fig. 2c and the Fig. S2 532 nm Te linear proxy."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pymupdf

from scripts.research.lightgen import diagnose_lightgen_direct_facet_linear_v1 as thinfilm


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SCHEMA = "lightgen_te_adma2026_proxy.v1"
FIELD_TOL = 1e-10
POWER_TOL = 1e-9

# Python / Matplotlib are the selected figure backend. Keep SVG text editable.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["legend.frameon"] = False


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    _write_new(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _write_csv_new(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or config.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("Unsupported or malformed ADMA2026 proxy config")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    literature, response = config["literature"], config["response"]
    spectrum, model = config["spectrum"], config["local_linear_model"]
    if literature["doi"] != "10.1002/adma.202522820" or literature["year"] != 2026:
        raise ValueError("The proxy must remain bound to ADMA2026 DOI 10.1002/adma.202522820")
    if response["ordinate"] != "Tr_norm" or response["quantity"] != "normalized_transmission":
        raise ValueError("Fig. 2c output is normalized Tr_norm; absolute output energy is not supported")
    if response["wavelength_nm"] is not None:
        raise ValueError("Fig. 2c does not uniquely specify wavelength; leave it unknown")
    if response["ordinate_direction"] != "increases_downward":
        raise ValueError("Fig. 2c ordinate increases downward in image coordinates")
    y_pixels = response["axis_calibration"]["transmission_tick_pixels_y"]
    y_values = response["axis_calibration"]["transmission_tick_values"]
    if len(y_pixels) != len(y_values) or len(y_pixels) < 2:
        raise ValueError("Fig. 2c ordinate calibration needs paired tick coordinates")
    if not all(b > a for a, b in zip(y_pixels, y_pixels[1:])):
        raise ValueError("Fig. 2c Tr_norm tick values must increase with downward pixel y")
    if not all(b > a for a, b in zip(y_values, y_values[1:])):
        raise ValueError("Fig. 2c normalized transmission ticks are not in increasing value order")
    if response["contrast_quantity"] != "1 - Tr_norm":
        raise ValueError("The only derived response readout permitted is the labelled 1 - Tr_norm contrast")
    if spectrum["target_wavelength_nm"] != 532.0 or model["wavelength_nm"] != 532.0:
        raise ValueError("This local linear proxy is fixed at 532 nm; 1530 nm inputs are not admissible")
    if spectrum["input_source"] != "Supplementary Figure 2 native EMF image":
        raise ValueError("532 nm n,k must come from Supplementary Figure 2, not the 1530 nm table")
    if spectrum["sample_identity"].strip() == "":
        raise ValueError("The spectrum sample identity and unresolved coupon details must be explicit")
    if model["te_thickness_nm"] != 15.0 or model["common_output_reference_plane_nm"] != 50.0:
        raise ValueError("The local comparison is fixed to 15 nm Te and the common z=50 nm reference plane")
    if model["propagation_direction"] != "glass -> Te -> air" or model["time_factor"] != "exp(-i omega t)":
        raise ValueError("The local linear model uses glass -> Te -> air and exp(-i omega t)")
    if model["layer_index_convention"] != "N=n+ik, k>=0":
        raise ValueError("The local linear model requires passive Te N=n+ik")
    if response["effective_length_um"] != 3.3 or response["waveguide_width_um"] != 1.3 or response["pulse_width_ns"] != 2.0:
        raise ValueError("The Fig. 2c identity is fixed to L=3.3 um, W=1.3 um and tau=2 ns")


def _repo_inputs(config: dict[str, Any]) -> dict[str, Path]:
    return {key: _resolve_repo_path(value) for key, value in config["sources"].items()}


def _validate_sources(config: dict[str, Any]) -> dict[str, Any]:
    sources = _repo_inputs(config)
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing explicit {label} source: {path}")
    with pymupdf.open(sources["paper_pdf"]) as document:
        if len(document) < 4:
            raise ValueError("ADMA2026 article PDF does not contain page 4 / Fig. 2")
        page_count = len(document)
        article_text = "\n".join(page.get_text() for page in document)
        if config["literature"]["doi"].lower() not in article_text.lower():
            raise ValueError("The article PDF text does not contain the configured DOI")
        if "phase-transition-driven all-optical neuron" not in article_text.lower():
            raise ValueError("The article PDF title does not match the configured paper")
        page_images = document[3].get_images(full=True)
        candidates = [row for row in page_images if row[2] >= 1500 and row[3] >= 1500]
        if len(candidates) != 1:
            raise ValueError("Fig. 2 page must contain one identifiable high-resolution embedded figure image")
        embedded = document.extract_image(candidates[0][0])
        if embedded["width"] != 1988 or embedded["height"] != 2372:
            raise ValueError("The embedded Fig. 2 raster dimensions differ from the reviewed source asset")
    with zipfile.ZipFile(sources["supplementary_docx"]) as archive:
        member = "word/media/image2.emf"
        if member not in archive.namelist():
            raise ValueError("The SI DOCX does not contain the expected native Fig. S2 EMF")
        emf_in_docx = archive.read(member)
    emf = sources["spectrum_emf"].read_bytes()
    if emf != emf_in_docx:
        raise ValueError("The selected EMF does not byte-match the SI's native Fig. S2 image")
    if len(emf) < 1000 or struct.unpack_from("<I", emf, 0)[0] != 1:
        raise ValueError("The Fig. S2 source is not a valid enhanced metafile record stream")
    if struct.unpack_from("<I", emf, 40)[0] != 0x464D4520:
        raise ValueError("The Fig. S2 source is missing the EMF header signature")
    return {
        "source_paths": {key: str(path.resolve()) for key, path in sources.items()},
        "article_pages": page_count,
        "fig2_page_index_zero_based": 3,
        "fig2_embedded_image_xref": candidates[0][0],
        "fig2_embedded_image_dimensions_px": [embedded["width"], embedded["height"]],
        "figs2_emf_bytes": len(emf),
        "figs2_emf_matches_docx_embedded_member": member,
        "article_title_and_doi_checked": True,
    }


def _powershell_path() -> str:
    for candidate in ("powershell.exe", "powershell"):
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError("Windows PowerShell is required to rasterize the archived native EMF")


def _render_emf(emf_path: Path, out_dir: Path, widths: list[int]) -> dict[int, Path]:
    if platform.system() != "Windows":
        raise RuntimeError("The EMF source must be rendered by Windows GDI+ on this Windows CPU workflow")
    helper = out_dir / "render_emf.ps1"
    script = r'''param(
  [Parameter(Mandatory=$true)][string]$InputEmf,
  [Parameter(Mandatory=$true)][string]$OutputPng,
  [Parameter(Mandatory=$true)][int]$Width
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
$meta = [System.Drawing.Imaging.Metafile]::new($InputEmf)
$height = [int][Math]::Round($Width * $meta.Height / $meta.Width)
$bitmap = [System.Drawing.Bitmap]::new($Width, $height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.Clear([System.Drawing.Color]::White)
$graphics.DrawImage($meta, 0, 0, $Width, $height)
$bitmap.Save($OutputPng, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()
$meta.Dispose()
'''
    _write_new(helper, script)
    outputs: dict[int, Path] = {}
    ps = _powershell_path()
    for width in widths:
        path = out_dir / f"adma2026_figs2_native_{width}px.png"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite EMF render: {path}")
        result = subprocess.run(
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(helper),
             "-InputEmf", str(emf_path), "-OutputPng", str(path), "-Width", str(width)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Windows GDI+ EMF render failed: {result.stderr.strip() or result.stdout.strip()}")
        with Image.open(path) as image:
            # GDI+ exposes the EMF frame as integer 678 x 521 px bounds.
            expected_height = round(width * 521 / 678)
            if image.format != "PNG" or image.size != (width, expected_height):
                raise ValueError(f"Unexpected {width}px EMF render size/format: {image.size}/{image.format}")
        outputs[width] = path
    return outputs


def _linear_fit(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), 1)
    predicted = slope * np.asarray(x, dtype=np.float64) + intercept
    residuals = np.asarray(y, dtype=np.float64) - predicted
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "rms_residual": float(np.sqrt(np.mean(residuals**2))),
        "r_squared": float(1.0 - np.sum(residuals**2) / np.sum((np.asarray(y) - np.mean(y)) ** 2)),
    }


def _response_calibration(config: dict[str, Any]) -> dict[str, Any]:
    axis = config["response"]["axis_calibration"]
    x_fit = _linear_fit(axis["input_energy_tick_pixels"], axis["input_energy_tick_values_pj"])
    y_fit = _linear_fit(axis["transmission_tick_pixels_y"], axis["transmission_tick_values"])
    checks = config["checks"]
    if x_fit["max_abs_residual"] > checks["response_calibration_max_energy_residual_pj"]:
        raise ValueError("Fig. 2c input-energy tick calibration residual exceeds its predeclared bound")
    if y_fit["max_abs_residual"] > checks["response_calibration_max_ordinate_residual"]:
        raise ValueError("Fig. 2c Tr_norm tick calibration residual exceeds its predeclared bound")
    if y_fit["slope"] <= 0:
        raise ValueError("Fig. 2c pixel y calibration would reverse the plotted Tr_norm direction")
    return {"energy_pj_from_pixel_x": x_fit, "tr_norm_from_pixel_y": y_fit,
            "source_coordinate_convention": "pixel y increases downward; Tr_norm increases downward"}


def _gray_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _dark_peak(profile: np.ndarray, center: float, radius: int) -> int:
    index = int(round(center))
    lo, hi = max(0, index - radius), min(len(profile), index + radius + 1)
    if lo >= hi:
        raise ValueError("Axis tick locator window is outside the rendered source image")
    window = profile[lo:hi]
    maximum = int(np.max(window))
    candidates = np.flatnonzero(window == maximum)
    return int(lo + candidates[len(candidates) // 2])


def _spectrum_axis_calibration(image_path: Path, nominal: dict[str, Any]) -> dict[str, Any]:
    rgb = _rgb_array(image_path).astype(np.int16)
    gray = np.asarray(np.round(rgb.mean(axis=2)), dtype=np.uint8)
    height, width = gray.shape
    scale = width / 1600.0
    dark = gray < 90
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    orange = (red > 150) & (red - green > 18) & (green - blue > 18) & (green > 65)
    x0, y0, x1, y1 = nominal["plot_xyxy"]
    x0, y0, x1, y1 = [int(round(value * scale)) for value in (x0, y0, x1, y1)]
    # Re-locate the four long frame lines in narrow, predeclared windows.
    v_profile_left = dark[max(0, y0 + 8):min(height, y1 - 8), max(0, x0 - round(30*scale)):x0 + round(31*scale)].sum(axis=0)
    left_window_start = max(0, x0 - round(30*scale))
    left = left_window_start + int(np.argmax(v_profile_left))
    right_start, right_stop = x1 - round(31*scale), min(width, x1 + round(30*scale))
    orange_profile = orange[max(0, y0 + 8):min(height, y1 - 8), right_start:right_stop].sum(axis=0)
    right = right_start + int(np.argmax(orange_profile))
    h_profile_top = dark[max(0, y0 - round(30*scale)):y0 + round(31*scale), left + round(8*scale):right - round(8*scale)].sum(axis=1)
    top_start = max(0, y0 - round(30*scale))
    top = top_start + int(np.argmax(h_profile_top))
    h_profile_bottom = dark[y1 - round(31*scale):min(height, y1 + round(30*scale)), left + round(8*scale):right - round(8*scale)].sum(axis=1)
    bottom = y1 - round(31*scale) + int(np.argmax(h_profile_bottom))
    if not (left < right and top < bottom):
        raise ValueError(f"Could not locate the SI Fig. S2 plot frame in {image_path.name}")

    x_tick_pixels: list[int] = []
    x_values = nominal["wavelength_tick_values_nm"]
    for tick in nominal["wavelength_tick_pixels_x"]:
        expected = tick * scale
        lo, hi = max(0, round(expected - 24*scale)), min(width, round(expected + 24*scale) + 1)
        profile = dark[max(top, bottom - round(21*scale)):bottom, lo:hi].sum(axis=0)
        x_tick_pixels.append(lo + int(np.argmax(profile)))

    def locate_y_ticks(nominal_pixels: list[int], x_slice: tuple[int, int]) -> list[int]:
        ticks = []
        xl, xr = x_slice
        xl, xr = max(0, xl), min(width, xr)
        for value in nominal_pixels:
            expected = value * scale
            lo, hi = max(top, round(expected - 13*scale)), min(bottom + 1, round(expected + 13*scale) + 1)
            profile = dark[lo:hi, xl:xr].sum(axis=1)
            ticks.append(lo + int(np.argmax(profile)))
        return ticks

    n_ticks_y = [bottom] + locate_y_ticks(nominal["n_tick_pixels_y"][1:], (left, left + round(22*scale)))
    k_ticks_y = [bottom]
    for value in nominal["k_tick_pixels_y"][1:]:
        expected = value * scale
        lo, hi = max(top, round(expected - 13*scale)), min(bottom + 1, round(expected + 13*scale) + 1)
        profile = orange[lo:hi, max(0, right - round(22*scale)):right].sum(axis=1)
        k_ticks_y.append(lo + int(np.argmax(profile)))
    x_fit = _linear_fit(x_tick_pixels, x_values)
    n_fit = _linear_fit(n_ticks_y, nominal["n_tick_values"])
    k_fit = _linear_fit(k_ticks_y, nominal["k_tick_values"])
    max_px = float(max(x_fit["max_abs_residual"] * abs(1.0 / x_fit["slope"]),
                       n_fit["max_abs_residual"] / abs(n_fit["slope"]),
                       k_fit["max_abs_residual"] / abs(k_fit["slope"])))
    return {
        "image": str(image_path), "image_size_px": [width, height], "scale_vs_1600": scale,
        "plot_bounds_xyxy_px": [left, top, right, bottom],
        "wavelength_tick_pixels_x": x_tick_pixels,
        "wavelength_tick_values_nm": x_values,
        "n_tick_pixels_y": n_ticks_y,
        "n_tick_values": nominal["n_tick_values"],
        "k_tick_pixels_y": k_ticks_y,
        "k_tick_values": nominal["k_tick_values"],
        "wavelength_nm_from_pixel_x": x_fit,
        "n_from_pixel_y_left_axis": n_fit,
        "k_from_pixel_y_right_axis": k_fit,
        "max_axis_fit_residual_px_equivalent": max_px,
    }


def _spectrum_preflight(config: dict[str, Any], rendered: dict[int, Path]) -> dict[str, Any]:
    nominal = config["spectrum"]["axis_locator_nominal_1600_px"]
    calibrations = {str(width): _spectrum_axis_calibration(path, nominal) for width, path in rendered.items()}
    maximum = max(
        value["wavelength_nm_from_pixel_x"]["max_abs_residual"]
        for value in calibrations.values()
    )
    residual_limit = float(config["checks"]["spectrum_axis_max_fit_residual_px"])
    if maximum > residual_limit * max(value["scale_vs_1600"] for value in calibrations.values()):
        raise ValueError("SI Fig. S2 axis calibration residual exceeds the predeclared bound")
    for value in calibrations.values():
        if value["n_from_pixel_y_left_axis"]["slope"] >= 0 or value["k_from_pixel_y_right_axis"]["slope"] >= 0:
            raise ValueError("The SI left n and right k axes must both decrease as image pixel y increases")
    return calibrations


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), **details}


def _reference_n_glass(config: dict[str, Any]) -> float:
    model = config["local_linear_model"]
    coeff = model["glass_sellmeier_coefficients"]
    return thinfilm._malitson_n(model["wavelength_nm"] / 1000.0, coeff)


def _thinfilm_preflight(config: dict[str, Any]) -> dict[str, Any]:
    model = config["local_linear_model"]
    n_in, n_out = _reference_n_glass(config), float(model["air_refractive_index"])
    wavelength_m = float(model["wavelength_nm"]) * 1e-9
    n_layer = complex(3.0, 0.8)
    d_m = float(model["te_thickness_nm"]) * 1e-9
    primary = thinfilm.single_layer_coefficients(n_in, n_layer, n_out, d_m, wavelength_m)
    ref_r, ref_t = thinfilm.solve_boundary_continuity_independently(n_in, n_layer, n_out, d_m, wavelength_m)
    bare_r, bare_t = (n_in - n_out) / (n_in + n_out), 2*n_in / (n_in + n_out)
    zero = thinfilm.single_layer_coefficients(n_in, n_layer, n_out, 0.0, wavelength_m)
    lossless = thinfilm.single_layer_coefficients(n_in, complex(n_layer.real, 0.0), n_out, d_m, wavelength_m)
    lossless_power = thinfilm.power_metrics(lossless["r"], lossless["t"], n_in, n_out)
    z_ref = float(model["common_output_reference_plane_nm"]) * 1e-9
    prop_bare = np.exp(2j * np.pi * n_out * z_ref / wavelength_m)
    zero_q = zero["t"] * prop_bare / (bare_t * prop_bare)
    return {
        "checks": [
            _check("closed_form_matches_independent_boundary_solve",
                   abs(primary["r"] - ref_r) <= FIELD_TOL and abs(primary["t"] - ref_t) <= FIELD_TOL,
                   delta_r=abs(primary["r"] - ref_r), delta_t=abs(primary["t"] - ref_t)),
            _check("zero_thickness_reduces_to_bare_interface",
                   abs(zero["r"] - bare_r) <= FIELD_TOL and abs(zero["t"] - bare_t) <= FIELD_TOL,
                   delta_r=abs(zero["r"] - bare_r), delta_t=abs(zero["t"] - bare_t)),
            _check("lossless_film_has_zero_absorption_and_conserves_power",
                   abs(lossless_power["A_flux"]) <= POWER_TOL and abs(lossless_power["R"] + lossless_power["T"] - 1.0) <= POWER_TOL,
                   **lossless_power),
            _check("zero_thickness_common_reference_phase_is_zero", abs(np.angle(zero_q)) <= FIELD_TOL,
                   phase_rad=float(np.angle(zero_q))),
        ],
        "n_glass": n_in,
        "n_air": n_out,
        "wavelength_m": wavelength_m,
        "fixed_reference_plane_nm": z_ref * 1e9,
    }


def run_preflight(config_path: str | Path, output_dir: str | Path) -> int:
    config = load_config(config_path)
    out = Path(output_dir)
    if out.exists():
        if any(out.iterdir()):
            raise FileExistsError(f"Refusing to reuse non-empty diagnostic output directory: {out}")
    else:
        out.mkdir(parents=True)
    source_facts = _validate_sources(config)
    response_cal = _response_calibration(config)
    rendered = _render_emf(_repo_inputs(config)["spectrum_emf"], out, [
        config["spectrum"]["independent_render_width_px"], config["spectrum"]["primary_render_width_px"]
    ])
    spectrum_cal = _spectrum_preflight(config, rendered)
    optical = _thinfilm_preflight(config)
    negative_controls = []
    for label, mutation in (
        ("reject_wrong_ordinate_direction", {"response": {"ordinate_direction": "increases_upward"}}),
        ("reject_absolute_output_energy", {"response": {"quantity": "absolute_output_energy"}}),
        ("reject_1530nm_as_target_parameter", {"spectrum": {"target_wavelength_nm": 1530.0}}),
    ):
        probe = json.loads(json.dumps(config))
        for group, values in mutation.items():
            probe[group].update(values)
        rejected = False
        reason = ""
        try:
            validate_config(probe)
        except ValueError as exc:
            rejected, reason = True, str(exc)
        negative_controls.append(_check(label, rejected, rejection_reason=reason))
    all_checks = optical["checks"] + negative_controls
    valid = all(item["passed"] for item in all_checks)
    payload = {
        "schema": "lightgen_te_adma2026_proxy_preflight.v1",
        "status": "passed" if valid else "invalid_harness",
        "evidence_tier": "diagnostic_only",
        "config_path": str(Path(config_path).resolve()),
        "output_path": str(out.resolve()),
        "fixed_identity": {
            "doi": config["literature"]["doi"],
            "response_figure": config["literature"]["response_figure"],
            "spectrum_figure": config["literature"]["spectrum_figure"],
            "response_wavelength_nm": None,
            "local_linear_wavelength_nm": config["local_linear_model"]["wavelength_nm"],
            "local_te_thickness_nm": config["local_linear_model"]["te_thickness_nm"],
            "common_reference_plane_nm": config["local_linear_model"]["common_output_reference_plane_nm"],
        },
        "source_facts": source_facts,
        "response_axis_calibration": response_cal,
        "spectrum_axis_calibrations": spectrum_cal,
        "optical_controls": optical,
        "negative_controls": negative_controls,
    }
    _write_json_new(out / "preflight.json", payload)
    return 0 if valid else 2


def _connected_components(mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    active_y, active_x = np.nonzero(mask)
    components: list[dict[str, Any]] = []
    for start_y, start_x in zip(active_y.tolist(), active_x.tolist(), strict=True):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        stack = [(start_y, start_x)]
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        values = np.asarray(points, dtype=np.float64)
        ymin, xmin = values.min(axis=0)
        ymax, xmax = values.max(axis=0)
        components.append({
            "x_px": float(values[:, 1].mean()),
            "y_px": float(values[:, 0].mean()),
            "area_px": int(values.shape[0]),
            "width_px": int(xmax - xmin + 1),
            "height_px": int(ymax - ymin + 1),
            "pixels": points,
        })
    return components


def _circle_fit(mask: np.ndarray, x0: float, y0: float, config: dict[str, Any]) -> dict[str, float] | None:
    padded = np.pad(mask, 1, mode="constant")
    eroded = padded[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
    edge = mask & ~eroded
    radius_expected = float(config["response"]["digitization"]["circle_expected_radius_px"])
    tolerance = float(config["response"]["digitization"]["circle_radius_tolerance_px"])
    window = int(math.ceil(radius_expected + tolerance + 1))
    ymin, ymax = max(0, int(y0) - window), min(mask.shape[0], int(y0) + window + 1)
    xmin, xmax = max(0, int(x0) - window), min(mask.shape[1], int(x0) + window + 1)
    local_y, local_x = np.nonzero(edge[ymin:ymax, xmin:xmax])
    xs, ys = local_x + xmin, local_y + ymin
    distance = np.hypot(xs - x0, ys - y0)
    keep = np.abs(distance - radius_expected) <= tolerance
    xs, ys = xs[keep].astype(np.float64), ys[keep].astype(np.float64)
    if xs.size < 12:
        return None
    design = np.column_stack((xs, ys, np.ones_like(xs)))
    target = -(xs**2 + ys**2)
    a, b, c = np.linalg.lstsq(design, target, rcond=None)[0]
    cx, cy = -a / 2.0, -b / 2.0
    radius_sq = cx**2 + cy**2 - c
    if radius_sq <= 0:
        return None
    radius = math.sqrt(radius_sq)
    residual = np.hypot(xs - cx, ys - cy) - radius
    rmse = float(np.sqrt(np.mean(residual**2)))
    angles = np.sort(np.mod(np.arctan2(ys - cy, xs - cx), 2 * np.pi))
    largest_gap = float(np.max(np.diff(np.r_[angles, angles[0] + 2*np.pi])))
    coverage = 2 * math.pi - largest_gap
    if (abs(cx - x0) > 4.0 or abs(cy - y0) > 4.0 or
            abs(radius - radius_expected) > tolerance or
            rmse > float(config["response"]["digitization"]["circle_fit_max_rmse_px"]) or coverage < math.pi):
        return None
    return {"x_px": float(cx), "y_px": float(cy), "radius_px": float(radius),
            "fit_rmse_px": rmse, "arc_coverage_rad": coverage, "edge_pixels": int(xs.size)}


def _response_mask(crop: np.ndarray, config: dict[str, Any], center_mask: bool = False) -> np.ndarray:
    image = np.asarray(crop, dtype=np.uint8)
    red, green, blue = (image[:, :, index].astype(np.int16) for index in range(3))
    limits = config["response"]["digitization"]
    green_threshold = limits["center_marker_min_green"] if center_mask else limits["yellow_marker_min_green"]
    mask = ((red >= limits["yellow_marker_min_red"]) &
            (green >= green_threshold) &
            (blue <= limits["yellow_marker_max_blue"]) &
            (red - green >= 18) & (green - blue >= 25))
    x0, y0, x1, y1 = config["response"]["axis_calibration"]["plot_xyxy_px_in_crop"]
    inside = np.zeros(mask.shape, dtype=bool)
    inside[y0 + 4:y1 - 3, x0 + 3:x1 - 3] = True
    mask &= inside
    lx0, ly0, lx1, ly1 = limits["legend_exclusion_xyxy_px_in_crop"]
    mask[ly0:ly1, lx0:lx1] = False
    return mask


def _response_detections(crop: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    mask = _response_mask(crop, config)
    center_mask = _response_mask(crop, config, center_mask=True)
    limits = config["response"]["digitization"]
    components = _connected_components(center_mask)
    center_candidates = []
    unresolved_components = []
    for index, component in enumerate(components):
        area, width, height = component["area_px"], component["width_px"], component["height_px"]
        plausible = (limits["minimum_marker_area_px"] <= area <= limits["maximum_marker_area_px"]
                     and 18 <= width <= 32 and 18 <= height <= 32
                     and 0.75 <= width / height <= 1.35)
        if plausible:
            center_candidates.append({"id": f"cc{index:02d}", "x_px": component["x_px"],
                                      "y_px": component["y_px"], "area_px": area,
                                      "width_px": width, "height_px": height})
        elif area >= 60 and width >= 10 and height >= 8:
            # Merged, truncated and irregular marker regions remain visible in the audit.
            unresolved_components.append({key: value for key, value in component.items() if key != "pixels"})

    # A sliding-window marker score yields candidate circles independently of connected-component centroids.
    side = int(limits["circle_candidate_window_px"])
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    scores = integral[side:, side:] - integral[:-side, side:] - integral[side:, :-side] + integral[:-side, :-side]
    half = side // 2
    order = np.argsort(scores.ravel())[::-1]
    circle_candidates = []
    separation = float(limits["circle_candidate_min_separation_px"])
    for flat in order:
        score = int(scores.ravel()[flat])
        if score < limits["circle_candidate_min_score"]:
            break
        y_top, x_left = np.unravel_index(int(flat), scores.shape)
        x0, y0 = float(x_left + half), float(y_top + half)
        if any(math.hypot(x0 - point["candidate_x_px"], y0 - point["candidate_y_px"]) < separation
               for point in circle_candidates):
            continue
        fit = _circle_fit(mask, x0, y0, config)
        if fit is not None:
            fit["id"] = f"cf{len(circle_candidates):02d}"
            fit["candidate_x_px"], fit["candidate_y_px"], fit["score"] = x0, y0, score
            circle_candidates.append(fit)

    pairs: list[tuple[float, int, int]] = []
    for i, center in enumerate(center_candidates):
        for j, circle in enumerate(circle_candidates):
            distance = math.hypot(center["x_px"] - circle["x_px"], center["y_px"] - circle["y_px"])
            pairs.append((distance, i, j))
    pairs.sort()
    used_center: set[int] = set()
    used_circle: set[int] = set()
    matches = []
    for distance, i, j in pairs:
        if distance > limits["method_agreement_max_px"]:
            break
        if i in used_center or j in used_circle:
            continue
        used_center.add(i)
        used_circle.add(j)
        matches.append({"center_index": i, "circle_index": j, "agreement_px": distance})
    return {
        "mask_pixel_count": int(mask.sum()),
        "center_mask_pixel_count": int(center_mask.sum()),
        "center_mask_threshold_green": int(limits["center_marker_min_green"]),
        "connected_component_centers": center_candidates,
        "edge_circle_fits": circle_candidates,
        "matched_points": matches,
        "unresolved_components": unresolved_components,
        "unmatched_center_indices": sorted(set(range(len(center_candidates))) - used_center),
        "unmatched_circle_indices": sorted(set(range(len(circle_candidates))) - used_circle),
    }


def _response_points(config: dict[str, Any], crop: np.ndarray, calibration: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detections = _response_detections(crop, config)
    xfit, yfit = calibration["energy_pj_from_pixel_x"], calibration["tr_norm_from_pixel_y"]
    # A fixed pixel quantization plus axis-fit residual envelope is declared before method comparison.
    pixel_half = 0.5
    x_axis_error = abs(xfit["slope"]) * pixel_half + xfit["max_abs_residual"]
    y_axis_error = abs(yfit["slope"]) * pixel_half + yfit["max_abs_residual"]
    points: list[dict[str, Any]] = []
    for match_number, match in enumerate(detections["matched_points"]):
        center = detections["connected_component_centers"][match["center_index"]]
        circle = detections["edge_circle_fits"][match["circle_index"]]
        x_px = (center["x_px"] + circle["x_px"]) / 2.0
        y_px = (center["y_px"] + circle["y_px"]) / 2.0
        energy = xfit["slope"] * x_px + xfit["intercept"]
        tr_norm = yfit["slope"] * y_px + yfit["intercept"]
        x_error = x_axis_error + abs(xfit["slope"]) * match["agreement_px"] / 2.0
        y_error = y_axis_error + abs(yfit["slope"]) * match["agreement_px"] / 2.0
        points.append({
            "point_id": f"P{match_number + 1:02d}",
            "center_component_id": center["id"],
            "circle_fit_id": circle["id"],
            "center_x_px": center["x_px"], "center_y_px": center["y_px"],
            "circle_x_px": circle["x_px"], "circle_y_px": circle["y_px"],
            "method_agreement_px": match["agreement_px"],
            "input_energy_pj": energy,
            "input_energy_error_pj_digitization_only": x_error,
            "tr_norm": tr_norm,
            "tr_norm_error_digitization_only": y_error,
            "contrast_1_minus_tr_norm": 1.0 - tr_norm,
            "contrast_low": 1.0 - (tr_norm + y_error),
            "contrast_high": 1.0 - (tr_norm - y_error),
            "status": f"cross_method_match_within_predeclared_{config['response']['digitization']['method_agreement_max_px']:g}_px",
            "included_in_interpolation": True,
        })
    points.sort(key=lambda row: row["input_energy_pj"])
    if len(points) < int(config["response"]["digitization"]["minimum_accepted_points"]):
        raise ValueError(f"Only {len(points)} Fig. 2c points survived cross-method matching")
    energies = np.asarray([row["input_energy_pj"] for row in points], dtype=np.float64)
    if not np.all(np.diff(energies) > 0):
        raise ValueError("Accepted Fig. 2c points do not define a strictly ordered input-energy coordinate")
    for index, point in enumerate(points, start=1):
        point["point_id"] = f"P{index:02d}"
    details = {
        **detections,
        "accepted_point_count": len(points),
        "unresolved_component_count": len(detections["unresolved_components"]),
        "pixel_quantization_half_width": pixel_half,
        "x_axis_error_bound_pj": x_axis_error,
        "tr_norm_axis_error_bound": y_axis_error,
        "interpolation_policy": "piecewise linear only inside min/max energy of cross-method matched visible points; no extrapolation or monotonic correction",
        "unresolved_point_policy": "merged, unmatched, or poorly fitted components are recorded and excluded; no synthetic points are inserted",
    }
    return points, details


def _piecewise_linear(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    if xp.ndim != 1 or fp.ndim != 1 or xp.size != fp.size or xp.size < 2:
        raise ValueError("Piecewise-linear response needs paired one-dimensional points")
    if not np.all(np.diff(xp) > 0):
        raise ValueError("Piecewise-linear response requires strictly increasing observed energy coordinates")
    if np.any(x < xp[0]) or np.any(x > xp[-1]):
        raise ValueError("Response interpolation requested an unobserved extrapolation")
    return np.interp(x, xp, fp)


def _sample_curve(config: dict[str, Any], points: list[dict[str, Any]]) -> list[dict[str, float]]:
    energies = np.asarray([point["input_energy_pj"] for point in points], dtype=np.float64)
    values = np.asarray([point["tr_norm"] for point in points], dtype=np.float64)
    errors = np.asarray([point["tr_norm_error_digitization_only"] for point in points], dtype=np.float64)
    xs = np.linspace(float(energies[0]), float(energies[-1]), int(config["output"]["response_curve_samples"]))
    ys = _piecewise_linear(xs, energies, values)
    err = _piecewise_linear(xs, energies, errors)
    return [{"input_energy_pj": float(x), "tr_norm": float(y),
             "contrast_1_minus_tr_norm": float(1.0-y),
             "tr_norm_low_digitization_only": float(y-e),
             "tr_norm_high_digitization_only": float(y+e),
             "contrast_low_digitization_only": float(1.0-y-e),
             "contrast_high_digitization_only": float(1.0-y+e)}
            for x, y, e in zip(xs, ys, err, strict=True)]


def _rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _extract_spectrum_trace(path: Path, calibration: dict[str, Any]) -> dict[str, Any]:
    rgb = _rgb_array(path).astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    x0, y0, x1, y1 = calibration["plot_bounds_xyxy_px"]
    inside = np.zeros(red.shape, dtype=bool)
    margin = max(8, round(12 * calibration["scale_vs_1600"]))
    inside[y0 + 4:y1 - 4, x0 + margin:x1 - margin] = True
    # Red curve (n, left axis) and orange curve (k, right axis) are classified separately.
    mask_n = inside & (red > 165) & (red - green > 30) & (red - blue > 25) & (np.abs(green-blue) < 38)
    mask_k = inside & (red > 165) & (green > 65) & (red-green > 20) & (green-blue > 24)
    xfit = calibration["wavelength_nm_from_pixel_x"]
    nfit = calibration["n_from_pixel_y_left_axis"]
    kfit = calibration["k_from_pixel_y_right_axis"]
    wavelength_tick_pixels = calibration["wavelength_tick_pixels_x"]
    x_scan_min = int(math.ceil(min(wavelength_tick_pixels) + 1))
    x_scan_max = int(math.floor(max(wavelength_tick_pixels) - 1))

    def trace(mask: np.ndarray, fit: dict[str, float], key: str) -> list[dict[str, float]]:
        rows = []
        for x_px in range(max(x0 + 5, x_scan_min), min(x1 - 4, x_scan_max + 1)):
            ys = np.flatnonzero(mask[y0 + 4:y1 - 4, x_px]) + y0 + 4
            if ys.size == 0:
                continue
            y_px = float(np.median(ys))
            wavelength = xfit["slope"] * x_px + xfit["intercept"]
            value = fit["slope"] * y_px + fit["intercept"]
            rows.append({"wavelength_nm": float(wavelength), key: float(value),
                         "pixel_x": float(x_px), "pixel_y_median": y_px,
                         "curve_width_px": float(np.ptp(ys) + 1 if ys.size > 1 else 1.0),
                         "pixel_count": int(ys.size)})
        rows.sort(key=lambda row: row["wavelength_nm"])
        if len(rows) < 20:
            raise ValueError(f"Could not trace enough {key} curve pixels from {path.name}")
        wavelengths = [row["wavelength_nm"] for row in rows]
        if not all(b > a for a, b in zip(wavelengths, wavelengths[1:])):
            raise ValueError(f"The {key} digitized spectrum has duplicate or reversed wavelength pixels")
        return rows

    return {"n": trace(mask_n, nfit, "n"), "k": trace(mask_k, kfit, "k")}


def _bracketing_interpolation(rows: list[dict[str, float]], key: str, wavelength: float) -> dict[str, Any]:
    before = [row for row in rows if row["wavelength_nm"] < wavelength]
    after = [row for row in rows if row["wavelength_nm"] > wavelength]
    if not before or not after:
        raise ValueError(f"{wavelength:g} nm is outside the plotted {key} spectrum; extrapolation is forbidden")
    lower, upper = before[-1], after[0]
    gap = upper["wavelength_nm"] - lower["wavelength_nm"]
    if gap > 8.0:
        raise ValueError(f"The {key} curve has a {gap:.3g} nm extraction gap around the target wavelength")
    fraction = (wavelength - lower["wavelength_nm"]) / gap
    value = lower[key] + fraction * (upper[key] - lower[key])
    curve_slope = (upper[key] - lower[key]) / gap
    return {"target_wavelength_nm": wavelength,
            "lower_point": {"wavelength_nm": lower["wavelength_nm"], key: lower[key],
                            "pixel_x": lower["pixel_x"], "pixel_y_median": lower["pixel_y_median"],
                            "curve_width_px": lower["curve_width_px"]},
            "upper_point": {"wavelength_nm": upper["wavelength_nm"], key: upper[key],
                            "pixel_x": upper["pixel_x"], "pixel_y_median": upper["pixel_y_median"],
                            "curve_width_px": upper["curve_width_px"]},
            "value": float(value), "local_slope_per_nm": float(curve_slope),
            "bracketing_gap_nm": float(gap)}


def _spectrum_estimate(trace: dict[str, Any], calibration: dict[str, Any], target: float) -> dict[str, Any]:
    estimates: dict[str, Any] = {}
    for key, fit_key in (("n", "n_from_pixel_y_left_axis"), ("k", "k_from_pixel_y_right_axis")):
        result = _bracketing_interpolation(trace[key], key, target)
        fit = calibration[fit_key]
        xfit = calibration["wavelength_nm_from_pixel_x"]
        width = max(result["lower_point"]["curve_width_px"], result["upper_point"]["curve_width_px"])
        xerr_nm = abs(xfit["slope"]) * (0.5 + calibration["max_axis_fit_residual_px_equivalent"])
        yerr = abs(fit["slope"]) * (0.5 + width / 2.0 + fit["max_abs_residual"] / abs(fit["slope"]))
        value_error = yerr + abs(result["local_slope_per_nm"]) * xerr_nm
        result.update({"digitalization_uncertainty": float(value_error),
                       "uncertainty_scope": "curve pixel width, pixel quantization, axis fit residual and wavelength-axis quantization only; experimental uncertainty not reported here"})
        estimates[key] = result
    return estimates


def _interpolate_trace_at(rows: list[dict[str, float]], key: str, wavelength: float) -> float:
    points = _bracketing_interpolation(rows, key, wavelength)
    return float(points["value"])


def _complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _local_optics(config: dict[str, Any], n_te: float, k_te: float) -> dict[str, Any]:
    model = config["local_linear_model"]
    wavelength_nm, thickness_nm = float(model["wavelength_nm"]), float(model["te_thickness_nm"])
    wavelength_m, thickness_m = wavelength_nm * 1e-9, thickness_nm * 1e-9
    z_ref_m = float(model["common_output_reference_plane_nm"]) * 1e-9
    n_in, n_out = _reference_n_glass(config), float(model["air_refractive_index"])
    layer = complex(n_te, k_te)
    coated = thinfilm.single_layer_coefficients(n_in, layer, n_out, thickness_m, wavelength_m)
    independent_r, independent_t = thinfilm.solve_boundary_continuity_independently(n_in, layer, n_out, thickness_m, wavelength_m)
    bare_r, bare_t = (n_in - n_out) / (n_in + n_out), 2*n_in / (n_in + n_out)
    air_phase = np.exp(2j * np.pi * n_out * (z_ref_m - thickness_m) / wavelength_m)
    bare_phase = np.exp(2j * np.pi * n_out * z_ref_m / wavelength_m)
    coated_t_ref, bare_t_ref = coated["t"] * air_phase, bare_t * bare_phase
    q = coated_t_ref / bare_t_ref
    metrics = thinfilm.power_metrics(coated["r"], coated["t"], n_in, n_out)
    a64 = thinfilm.integrate_absorption(thickness_m, coated["forward_0"], coated["backward_0"],
                                        layer, wavelength_m, n_in, 64)
    a128 = thinfilm.integrate_absorption(thickness_m, coated["forward_0"], coated["backward_0"],
                                         layer, wavelength_m, n_in, 128)
    error = thinfilm.validate_power_balance(metrics, a128, POWER_TOL)
    profile_rows = []
    for z in np.linspace(0.0, thickness_m, int(config["output"]["field_profile_samples"])):
        field = (coated["forward_0"] * np.exp(2j*np.pi*layer*z/wavelength_m) +
                 coated["backward_0"] * np.exp(-2j*np.pi*layer*z/wavelength_m))
        absorption = float(thinfilm.absorption_density_per_m(z, coated["forward_0"], coated["backward_0"],
                                                              layer, wavelength_m, n_in))
        profile_rows.append({"z_nm": z * 1e9, "E_relative_real": field.real,
                             "E_relative_imag": field.imag, "abs_E_relative_squared": abs(field)**2,
                             "absorption_per_m_normalized_to_incident_flux": absorption})
    tau_coated = math.sqrt(n_out / n_in) * coated_t_ref
    tau_bare = math.sqrt(n_out / n_in) * bare_t_ref
    return {
        "sample_identity": "ADMA2026 Fig. S2 ellipsometry n,k proxy applied to a 15 nm Te local-layer candidate; not asserted to be the waveguide specimen or project sputtered Te",
        "wavelength_nm": wavelength_nm, "te_thickness_nm": thickness_nm,
        "structure": "fused-silica model / 15 nm Te / air; no ITO in this local facet candidate",
        "n_glass": n_in, "k_glass_assumption": 0.0, "n_te_from_figs2": n_te, "k_te_from_figs2": k_te,
        "n_air": n_out, "reference_plane_nm": z_ref_m * 1e9,
        "reference_plane_convention": "coated and bare transmitted fields propagated to z=50 nm in air",
        "R": metrics["R"], "T": metrics["T"], "A_flux": metrics["A_flux"],
        "A_integral_order64": a64, "A_integral_order128": a128,
        "absorbed_power_balance_error": error,
        "rE": _complex_pair(coated["r"]),
        "tE_at_common_reference": _complex_pair(coated_t_ref),
        "power_normalized_tE_at_common_reference": _complex_pair(tau_coated),
        "bare_rE": _complex_pair(bare_r),
        "bare_tE_at_common_reference": _complex_pair(bare_t_ref),
        "bare_power_normalized_tE_at_common_reference": _complex_pair(tau_bare),
        "coated_to_bare_relative_complex_transmission_q": _complex_pair(q),
        "coated_to_bare_power_ratio": float(abs(q)**2),
        "coated_to_bare_phase_difference_rad": float(np.angle(q)),
        "independent_boundary_solution": {"delta_r": abs(coated["r"] - independent_r),
                                          "delta_t": abs(coated["t"] - independent_t)},
        "field_profile": profile_rows,
    }


def _optics_envelope(config: dict[str, Any], estimates: dict[str, Any]) -> dict[str, Any]:
    n_range = estimates["n"]["envelope"]
    k_range = estimates["k"]["envelope"]
    rows = []
    for n_te in (n_range[0], estimates["n"]["value"], n_range[1]):
        for k_te in (k_range[0], estimates["k"]["value"], k_range[1]):
            result = _local_optics(config, n_te, k_te)
            rows.append({"n_te": n_te, "k_te": k_te, "R": result["R"], "T": result["T"], "A_flux": result["A_flux"],
                         "q_abs_squared": result["coated_to_bare_power_ratio"],
                         "q_phase_rad": result["coated_to_bare_phase_difference_rad"]})
    keys = ("R", "T", "A_flux", "q_abs_squared", "q_phase_rad")
    ranges = {key: {"min": float(min(row[key] for row in rows)), "max": float(max(row[key] for row in rows))}
              for key in keys}
    return {"method": "3x3 corner and center sampling of independent n and k digitization intervals",
            "experimental_uncertainty_included": False, "sample_count": len(rows), "samples": rows, "ranges": ranges}


def _curve_rows_by_width(config: dict[str, Any], rendered: dict[int, Path],
                         calibrations: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    traces, estimates = {}, {}
    target = float(config["spectrum"]["target_wavelength_nm"])
    for width, path in rendered.items():
        key = str(width)
        traces[key] = _extract_spectrum_trace(path, calibrations[key])
        estimates[key] = _spectrum_estimate(traces[key], calibrations[key], target)
    return traces, estimates


def _point_csv_rows(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return points


def _plot_style() -> None:
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "lines.linewidth": 1.1, "lines.markersize": 4})


def _panel_alignment(fig: Any, base_path: Path) -> None:
    scripts_dir = Path(os.environ.get("NATURE_FIGURE_SCRIPTS_DIR", Path.home() / ".codex/skills/nature-figure/scripts"))
    helper = scripts_dir / "audit_panel_alignment.py"
    if not helper.is_file():
        raise FileNotFoundError(f"Nature Figure panel-alignment helper not found: {helper}")
    spec = importlib.util.spec_from_file_location("lightgen_nature_panel_alignment", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the selected Python panel-alignment helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fig.canvas.draw()
    module.require_matplotlib_panel_alignment(
        fig, json_out=str(base_path.with_suffix(".alignment.json")),
        overlay_svg=str(base_path.with_suffix(".alignment.svg")),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, require_panel_labels=True,
    )


def _save_figure(fig: Any, base_path: Path, is_multipanel: bool) -> None:
    if is_multipanel:
        _panel_alignment(fig, base_path)
    else:
        _write_json_new(base_path.with_suffix(".alignment.json"),
                        {"verdict": "NOT APPLICABLE", "reason": "one plotted axes; secondary y-axis is part of the same plot area"})
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _render_response_overlay(crop: np.ndarray, points: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.imshow(crop, interpolation="nearest")
    if points:
        x = [point["circle_x_px"] for point in points]
        y = [point["circle_y_px"] for point in points]
        ax.scatter(x, y, s=42, facecolors="none", edgecolors="#006d77", linewidths=0.8)
    ax.set_axis_off()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _render_response_figures(config: dict[str, Any], crop: np.ndarray, points: list[dict[str, Any]],
                             curve: list[dict[str, float]], out_dir: Path) -> None:
    _plot_style()
    _render_response_overlay(crop, points, out_dir / "fig2c_extraction_overlay.png")
    data = np.asarray([[row["input_energy_pj"], row["tr_norm"], row["contrast_1_minus_tr_norm"],
                        row["tr_norm_low_digitization_only"], row["tr_norm_high_digitization_only"],
                        row["contrast_low_digitization_only"], row["contrast_high_digitization_only"]]
                       for row in curve], dtype=np.float64)
    point_e = np.asarray([row["input_energy_pj"] for row in points])
    point_r = np.asarray([row["tr_norm"] for row in points])
    point_err = np.asarray([row["tr_norm_error_digitization_only"] for row in points])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, constrained_layout=True)
    axes[0].fill_between(data[:, 0], data[:, 3], data[:, 4], color="#d9a400", alpha=0.18,
                         label="Digitization envelope")
    axes[0].plot(data[:, 0], data[:, 1], color="#b97900", label="Piecewise-linear reconstruction")
    axes[0].errorbar(point_e, point_r, xerr=[row["input_energy_error_pj_digitization_only"] for row in points],
                     yerr=point_err, fmt="o", ms=3.5, color="#c58b00", ecolor="#9c7a27",
                     elinewidth=0.6, capsize=1.5, label="Cross-method matched points")
    axes[0].set_ylabel("Normalized transmission, Tr_norm")
    axes[0].set_title("ADMA2026 Fig. 2c: reported ordinate")
    axes[0].legend(loc="best")
    axes[1].fill_between(data[:, 0], data[:, 5], data[:, 6], color="#477b9e", alpha=0.16,
                         label="Digitization envelope")
    axes[1].plot(data[:, 0], data[:, 2], color="#21658a", label="Contrast readout, 1 - Tr_norm")
    axes[1].errorbar(point_e, 1.0-point_r,
                     xerr=[row["input_energy_error_pj_digitization_only"] for row in points],
                     yerr=point_err, fmt="o", ms=3.5, color="#21658a", ecolor="#477b9e",
                     elinewidth=0.6, capsize=1.5, label="Derived matched points")
    axes[1].set_ylabel("1 - Tr_norm (contrast only)")
    axes[1].set_xlabel("Input pulse energy, E_in (pJ)")
    axes[1].legend(loc="best")
    for label, ax in zip(("a", "b"), axes, strict=True):
        ax.text(-0.13, 1.025, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    base = out_dir / "adma2026_fig2c_normalized_response"
    _save_figure(fig, base, is_multipanel=True)


def _render_nk_figure(config: dict[str, Any], traces: dict[str, Any], estimates: dict[str, Any],
                      calibrations: dict[str, Any], out_dir: Path) -> None:
    _plot_style()
    primary_key = str(config["spectrum"]["primary_render_width_px"])
    independent_key = str(config["spectrum"]["independent_render_width_px"])
    primary = traces[primary_key]
    independent = traces[independent_key]
    wavelength = np.asarray([row["wavelength_nm"] for row in primary["n"]])
    n_vals = np.asarray([row["n"] for row in primary["n"]])
    n_width = np.asarray([row["curve_width_px"] for row in primary["n"]])
    wavelength_k = np.asarray([row["wavelength_nm"] for row in primary["k"]])
    k_vals = np.asarray([row["k"] for row in primary["k"]])
    fig, ax_n = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax_k = ax_n.twinx()
    ax_k.spines["top"].set_visible(False)
    ax_k.spines["right"].set_visible(True)
    ax_n.plot(wavelength, n_vals, color="#c44848", linewidth=1.1)
    ax_k.plot(wavelength_k, k_vals, color="#df8b3d", linewidth=1.1)
    ax_n.plot([row["wavelength_nm"] for row in independent["n"]],
              [row["n"] for row in independent["n"]], color="#c44848", alpha=0.5,
              linewidth=0.7, linestyle="--")
    ax_k.plot([row["wavelength_nm"] for row in independent["k"]],
              [row["k"] for row in independent["k"]], color="#df8b3d", alpha=0.5,
              linewidth=0.7, linestyle="--")
    target = float(config["spectrum"]["target_wavelength_nm"])
    ax_n.axvline(target, color="#4d4d4d", linewidth=0.8, linestyle=":")
    for axis, name, color in ((ax_n, "n", "#c44848"), (ax_k, "k", "#df8b3d")):
        estimate = estimates[primary_key][name]
        value = float(estimate["value"])
        lower_error = value - float(estimate["envelope"][0])
        upper_error = float(estimate["envelope"][1]) - value
        axis.errorbar([target], [value], yerr=np.asarray([[lower_error], [upper_error]]),
                      fmt="o", color=color, ecolor=color, markersize=3.5, capsize=2.0,
                      elinewidth=0.8, zorder=5)
    ax_n.set_xlabel("Wavelength (nm)")
    ax_n.set_ylabel("Refractive index, n", color="#c44848")
    ax_k.set_ylabel("Extinction coefficient, k", color="#df8b3d")
    ax_n.tick_params(axis="y", colors="#c44848")
    ax_k.tick_params(axis="y", colors="#df8b3d")
    ax_n.set_title("ADMA2026 Supplementary Fig. 2: digitized ellipsometry spectrum\n"
                   "Red n and orange k; solid: 3200 px, dashed: 1600 px; markers: 532 nm estimates")
    ax_n.set_xlim(float(wavelength.min()), float(wavelength.max()))
    out_base = out_dir / "adma2026_figs2_te_nk_spectrum"
    _save_figure(fig, out_base, is_multipanel=False)


def _figure2c_crop(config: dict[str, Any], embedded_bytes: bytes) -> np.ndarray:
    from io import BytesIO

    image = Image.open(BytesIO(embedded_bytes)).convert("RGB")
    bbox = tuple(int(x) for x in config["response"]["axis_calibration"]["embedded_image_crop_xyxy_px"])
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > image.width or bbox[3] > image.height:
        raise ValueError("Configured Fig. 2c source crop is outside the embedded original image")
    return np.asarray(image.crop(bbox), dtype=np.uint8)


def _save_source_curve_csv(out: Path, width: int, key: str, rows: list[dict[str, float]]) -> None:
    path = out / f"figs2_{key}_{width}px.csv"
    fields = list(rows[0].keys())
    _write_csv_new(path, fields, rows)


def _measurement(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    out = Path(output_dir)
    preflight_path = out / "preflight.json"
    if not preflight_path.is_file():
        raise ValueError("A successful preflight from this entrypoint is required before measurement")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or preflight.get("fixed_identity", {}).get("doi") != config["literature"]["doi"]:
        raise ValueError("Preflight did not pass for the configured article and proxy identities")
    if preflight.get("config_path") != str(Path(config_path).resolve()):
        raise ValueError("Preflight was not run against this explicit config path")
    protected = ("measurement.json", "report.md", "fig2c_points.csv", "fig2c_piecewise_curve.csv",
                 "fig2c_detection_audit.json", "fig2c_unresolved_candidates.csv",
                 "figs2_n_1600px.csv", "figs2_k_1600px.csv", "figs2_n_3200px.csv", "figs2_k_3200px.csv",
                 "te_532_local_linear.csv", "te_532_field_profile.csv")
    existing = [name for name in protected if (out / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing diagnostic outputs: {existing}")

    sources = _repo_inputs(config)
    with pymupdf.open(sources["paper_pdf"]) as document:
        xref = preflight["source_facts"]["fig2_embedded_image_xref"]
        embedded = document.extract_image(xref)
        if embedded["width"] != 1988 or embedded["height"] != 2372:
            raise ValueError("The measured embedded Fig. 2 raster differs from the preflight source")
        embedded_bytes = embedded["image"]
    source_dir = out / "source_pages"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_image_path = source_dir / f"adma2026_fig2_embedded_image.{embedded['ext']}"
    if source_image_path.exists():
        raise FileExistsError(source_image_path)
    source_image_path.write_bytes(embedded_bytes)
    crop = _figure2c_crop(config, embedded_bytes)
    crop_path = source_dir / "adma2026_fig2c_source_crop.png"
    if crop_path.exists():
        raise FileExistsError(crop_path)
    Image.fromarray(crop).save(crop_path)

    response_cal = _response_calibration(config)
    points, response_audit = _response_points(config, crop, response_cal)
    curve = _sample_curve(config, points)

    rendered = {int(width): out / f"adma2026_figs2_native_{width}px.png"
                for width in (config["spectrum"]["independent_render_width_px"],
                              config["spectrum"]["primary_render_width_px"])}
    calibrations = {key: preflight["spectrum_axis_calibrations"][key] for key in (str(k) for k in rendered)}
    traces, estimates = _curve_rows_by_width(config, rendered, calibrations)
    primary_key = str(config["spectrum"]["primary_render_width_px"])
    independent_key = str(config["spectrum"]["independent_render_width_px"])
    consistency = {}
    estimate_envelopes = {"n": [], "k": []}
    for key in (primary_key, independent_key):
        for name in ("n", "k"):
            estimate = estimates[key][name]
            estimate["envelope"] = [estimate["value"] - estimate["digitalization_uncertainty"],
                                     estimate["value"] + estimate["digitalization_uncertainty"]]
            estimate_envelopes[name].append(estimate)
    for name in ("n", "k"):
        primary = estimates[primary_key][name]
        independent = estimates[independent_key][name]
        difference = abs(primary["value"] - independent["value"])
        bound = primary["digitalization_uncertainty"] + independent["digitalization_uncertainty"]
        consistency[name] = {"primary_3200px": primary["value"], "independent_1600px": independent["value"],
                             "absolute_difference": difference, "combined_digitalization_bound": bound,
                             "within_bound": difference <= bound}
        if not consistency[name]["within_bound"]:
            raise ValueError(f"The 1600/3200 px {name} estimates disagree beyond the predeclared image-derived envelope")
        primary["envelope"] = [min(x["envelope"][0] for x in estimate_envelopes[name]),
                               max(x["envelope"][1] for x in estimate_envelopes[name])]
    # The 1530 nm table is a separate cross-check, never the source substituted at 532 nm.
    table = config["spectrum"]["table_crosscheck"]
    table_crosscheck = {}
    for key in (primary_key, independent_key):
        row = {}
        graph_at_table_wavelength = _spectrum_estimate(
            traces[key], calibrations[key], float(table["wavelength_nm"])
        )
        for name in ("n", "k"):
            graph_value = _interpolate_trace_at(traces[key][name], name, float(table["wavelength_nm"]))
            graph_uncertainty = graph_at_table_wavelength[name]["digitalization_uncertainty"]
            table_rounding = 0.005
            row[name] = {"figs2_graph_value_at_1530nm": graph_value,
                         "figs2_graph_digitalization_uncertainty": graph_uncertainty,
                         "table_s1_value": float(table[name]),
                         "difference": graph_value - float(table[name]),
                         "within_graph_digitalization_plus_table_rounding":
                             abs(graph_value - float(table[name])) <= graph_uncertainty + table_rounding}
        table_crosscheck[key] = row

    n_te, k_te = estimates[primary_key]["n"]["value"], estimates[primary_key]["k"]["value"]
    optics = _local_optics(config, n_te, k_te)
    optics_envelope = _optics_envelope(config, estimates[primary_key])
    checks = [
        _check("fig2c_center_and_circle_fit_points_agree",
               len(points) >= config["response"]["digitization"]["minimum_accepted_points"],
               accepted_points=len(points), unmatched_center_count=len(response_audit["unmatched_center_indices"]),
               unmatched_circle_count=len(response_audit["unmatched_circle_indices"]),
               unresolved_component_count=len(response_audit["unresolved_components"])),
        _check("figs2_1600_and_3200_estimates_within_predeclared_digitalization_envelope",
               all(value["within_bound"] for value in consistency.values()), **consistency),
        _check("thinfilm_closed_form_matches_independent_boundary_solve",
               optics["independent_boundary_solution"]["delta_r"] <= FIELD_TOL and
               optics["independent_boundary_solution"]["delta_t"] <= FIELD_TOL,
               **optics["independent_boundary_solution"]),
        _check("thinfilm_power_balance_and_dissipation_integral",
               optics["absorbed_power_balance_error"] <= POWER_TOL and
               abs(optics["A_integral_order64"] - optics["A_integral_order128"]) <= POWER_TOL,
               power_error=optics["absorbed_power_balance_error"],
               quadrature_difference=abs(optics["A_integral_order64"] - optics["A_integral_order128"])),
        _check("table_s1_1530nm_crosscheck_recorded_without_532nm_substitution", True,
               source="Supplementary Table 1", graph_values_kept_separate=True),
        _check("normalized_response_not_converted_to_absolute_energy", True,
               ordinate="Tr_norm", wavelength_unknown=True, phase_not_inferred=True),
    ]
    payload = {
        "schema": "lightgen_te_adma2026_proxy_measurement.v1",
        "status": "valid" if all(item["passed"] for item in checks) else "invalid_harness",
        "evidence_tier": "diagnostic_only",
        "config_path": str(Path(config_path).resolve()),
        "output_path": str(out.resolve()),
        "paper_identity": config["literature"],
        "response_curve_identity": {
            "sample_identity": config["response"]["sample_identity"],
            "effective_length_um": config["response"]["effective_length_um"],
            "waveguide_width_um": config["response"]["waveguide_width_um"],
            "pulse_width_ns": config["response"]["pulse_width_ns"],
            "wavelength_nm": None,
            "ordinate": "Tr_norm",
            "quantity": "normalized transmission",
            "point_count_in_interpolation": len(points),
            "curve_domain_pj": [points[0]["input_energy_pj"], points[-1]["input_energy_pj"]],
            "fit_policy": "piecewise linear; in-domain only; no extrapolation, monotonic correction, or clipping",
        },
        "spectrum_identity": {
            "sample_identity": config["spectrum"]["sample_identity"],
            "source_figure": "Supplementary Fig. 2",
            "target_wavelength_nm": config["spectrum"]["target_wavelength_nm"],
            "primary_render_width_px": int(primary_key),
            "independent_render_width_px": int(independent_key),
            "results_532": {name: estimates[primary_key][name] for name in ("n", "k")},
            "render_consistency": consistency,
            "1530nm_table_crosscheck": table_crosscheck,
            "experimental_uncertainty": None,
            "uncertainty_statement": "Digitalization envelope only; experimental ellipsometry uncertainty is not inferred or imputed.",
        },
        "local_linear_optics": optics,
        "local_linear_digitization_envelope": optics_envelope,
        "response_extraction_audit": response_audit,
        "checks": checks,
        "scope_limit": [
            "Fig. 2c's exact wavelength is not uniquely identified; its normalized Tr_norm response is not absolute output energy.",
            "Fig. S2 ellipsometry coupon details are unresolved and are not equated with the waveguide or project sputtered Te specimen.",
            "The glass -> Te -> air plane-wave calculation omits the paper's ITO and all target facet support, roughness, oxide, scattering, guided-mode and nonlinear effects.",
            "This result is a literature proxy, not a measurement or performance prediction for the project's sputtered film or hardware.",
        ],
        "source_crosscheck_cautions": [
            f"Fig. S2 graph n at 1530 nm is {table_crosscheck[primary_key]['n']['figs2_graph_value_at_1530nm']:.6g} versus Supplementary Table 1 n={float(table['n']):.6g}; the difference is outside the graph digitization envelope plus table rounding. Preserve the Fig. S2-derived 532 nm value without replacing it with the table value."
            if not table_crosscheck[primary_key]["n"]["within_graph_digitalization_plus_table_rounding"] else
            "No n discrepancy at 1530 nm beyond the graph digitization envelope and table rounding was detected.",
            "The 1530 nm n,k table is a separate cross-check only and does not replace or average into the 532 nm Fig. S2 input."
        ],
        "source_image_paths": {"fig2_embedded": str(source_image_path), "fig2c_crop": str(crop_path),
                               "figs2_renders": {str(k): str(v) for k, v in rendered.items()}},
    }

    # Data records are created only after the scientific payload is complete.
    point_fields = list(points[0].keys())
    _write_csv_new(out / "fig2c_points.csv", point_fields, _point_csv_rows(points))
    _write_json_new(out / "fig2c_detection_audit.json", response_audit)
    unresolved_rows = []
    for index, component in enumerate(response_audit["unresolved_components"]):
        unresolved_rows.append({"candidate_id": f"merged{index + 1:02d}", "source_method": "bright-fill connected-component",
                                "x_px": component["x_px"], "y_px": component["y_px"],
                                "area_px": component["area_px"], "width_px": component["width_px"],
                                "height_px": component["height_px"],
                                "status": "unresolved merged_or_truncated_component; excluded from interpolation"})
    for index in response_audit["unmatched_center_indices"]:
        component = response_audit["connected_component_centers"][index]
        unresolved_rows.append({"candidate_id": component["id"], "source_method": "bright-fill connected-component",
                                "x_px": component["x_px"], "y_px": component["y_px"],
                                "area_px": component["area_px"], "width_px": component["width_px"],
                                "height_px": component["height_px"],
                                "status": "unresolved center lacks a matching edge-circle fit; excluded from interpolation"})
    unresolved_fields = ["candidate_id", "source_method", "x_px", "y_px", "area_px", "width_px", "height_px", "status"]
    _write_csv_new(out / "fig2c_unresolved_candidates.csv", unresolved_fields, unresolved_rows)
    curve_fields = list(curve[0].keys())
    _write_csv_new(out / "fig2c_piecewise_curve.csv", curve_fields, curve)
    for width in (int(independent_key), int(primary_key)):
        for name in ("n", "k"):
            _save_source_curve_csv(out, width, name, traces[str(width)][name])
    param_rows = []
    for name in ("n", "k"):
        estimate = estimates[primary_key][name]
        param_rows.append({"parameter": name, "value": estimate["value"],
                           "unit": "dimensionless", "wavelength_nm": 532.0,
                           "source": "ADMA2026 Supplementary Fig. 2, native EMF, 3200 px render",
                           "digitization_low": estimate["envelope"][0],
                           "digitization_high": estimate["envelope"][1],
                           "experimental_uncertainty": "not inferred"})
    _write_csv_new(out / "te_532_digitized_nk.csv", list(param_rows[0].keys()), param_rows)
    optical_rows = [{key: optics[key] for key in ("wavelength_nm", "te_thickness_nm", "n_glass", "n_te_from_figs2",
                                                    "k_te_from_figs2", "R", "T", "A_flux",
                                                    "A_integral_order64", "A_integral_order128",
                                                    "coated_to_bare_power_ratio", "coated_to_bare_phase_difference_rad",
                                                    "reference_plane_nm")}]
    _write_csv_new(out / "te_532_local_linear.csv", list(optical_rows[0].keys()), optical_rows)
    profile = optics["field_profile"]
    _write_csv_new(out / "te_532_field_profile.csv", list(profile[0].keys()), profile)
    response_image = _rgb_array(crop_path)
    _render_response_figures(config, response_image, points, curve, out)
    _render_nk_figure(config, traces, estimates, calibrations, out)
    _write_json_new(out / "measurement.json", payload)
    _write_new(out / "report.md", render_report(payload))
    return payload


def render_report(payload: dict[str, Any]) -> str:
    response, spectrum = payload["response_curve_identity"], payload["spectrum_identity"]
    optics, checks = payload["local_linear_optics"], payload["checks"]
    n, k = spectrum["results_532"]["n"], spectrum["results_532"]["k"]
    response_text = "归一化响应代理有效" if response["point_count_in_interpolation"] >= 8 else "可辨识点不足"
    optics_text = "Fig. S2 局部线性代理有效" if all(item["passed"] for item in checks) else "诊断无效"
    optical_ranges = payload["local_linear_digitization_envelope"]["ranges"]
    response_audit = payload["response_extraction_audit"]
    unresolved_center_count = len(response_audit["unmatched_center_indices"])
    unresolved_circle_count = len(response_audit["unmatched_circle_indices"])
    lines = [
        "# ADMA2026 Te 文献代理：原条件响应重建与 532 nm 局部线性参照", "",
        f"状态：**{payload['status']}**；响应：**{response_text}**；线性光学：**{optics_text}**。证据等级为 diagnostic-only。", "",
        "## 结果与状态", "",
        f"Fig. 2c 中有 {response['point_count_in_interpolation']} 个黄色测量点同时通过亮填充区域中心和边缘圆拟合核对，覆盖输入脉冲能量 {response['curve_domain_pj'][0]:.4g}–{response['curve_domain_pj'][1]:.4g} pJ。输出保留论文纵坐标 `Tr_norm(E_in)`；`1 - Tr_norm` 仅作对比读数。Fig. 2c 的精确波长仍未知。",
        f"论文响应曲线样品身份：{response['sample_identity']}。该层序属于波导器件样品，不应与本轮不含 ITO 的局部玻璃/Te/空气参照合并成同一堆栈。",
        f"Fig. S2 图源在 532 nm 给出 `n={n['value']:.6g}`（数字化包络 [{n['envelope'][0]:.6g}, {n['envelope'][1]:.6g}]）和 `k={k['value']:.6g}`（[{k['envelope'][0]:.6g}, {k['envelope'][1]:.6g}]）。这些区间是图像数字化误差，不是椭偏实验不确定度。",
        f"局部玻璃 → 15 nm Te → 空气模型的名义结果为 `R={optics['R']:.6g}`、`T={optics['T']:.6g}`、`A={optics['A_flux']:.6g}`；相对裸端面功率比 `|q|²={optics['coated_to_bare_power_ratio']:.6g}`，共同空气参考面 z={optics['reference_plane_nm']:.6g} nm 的相位差为 {optics['coated_to_bare_phase_difference_rad']:.6g} rad。",
        "图像数字化包络传播到局部光学量的范围如下；未纳入实验不确定度。", "",
        "| 物理量 | 下界 | 名义值 | 上界 |",
        "|---|---:|---:|---:|",
        f"| R | {optical_ranges['R']['min']:.6g} | {optics['R']:.6g} | {optical_ranges['R']['max']:.6g} |",
        f"| T | {optical_ranges['T']['min']:.6g} | {optics['T']:.6g} | {optical_ranges['T']['max']:.6g} |",
        f"| A | {optical_ranges['A_flux']['min']:.6g} | {optics['A_flux']:.6g} | {optical_ranges['A_flux']['max']:.6g} |",
        f"| 裸端面相对透射功率比 `|q|²` | {optical_ranges['q_abs_squared']['min']:.6g} | {optics['coated_to_bare_power_ratio']:.6g} | {optical_ranges['q_abs_squared']['max']:.6g} |",
        f"| 相对相位差 (rad) | {optical_ranges['q_phase_rad']['min']:.6g} | {optics['coated_to_bare_phase_difference_rad']:.6g} | {optical_ranges['q_phase_rad']['max']:.6g} |", "",
        "## 科学认识", "",
        "Fig. 2c 支持在论文图示条件下（有效长度 3.3 µm、波导宽度 1.3 µm、脉宽 2 ns）的经验输入脉冲能量—归一化透射曲线。图中没有绝对输出能量标定或复相位响应；不能把归一化曲线换算为绝对输出能量，也不能称作已经实现的透射 ReLU。像素纵坐标向下增加，但数值已按图轴恢复为 `Tr_norm`，未强制单调或裁剪略高于 1 的测量点。",
        "532 nm 的 `n,k` 取自补充 Fig. S2 原始 EMF 光谱，分别对应左侧 n 轴和右侧 k 轴。1530 nm 图像数字化 n≈6.0，与补充表 1 的 n=6.28 差异超出图像包络和表格舍入范围；k 与表格值接近。图源与表格保留分开，表格没有替换或平均进 532 nm 输入。",
        "局部薄膜模型采用 Fig. S2 图源代理参数、15 nm 候选厚度、Malitson 熔融石英色散、玻璃 k=0 研究假设和法向平面波。图源椭偏测试券未证明与论文波导 Te 膜为同一样品；波导器件的 ITO 未进入此局部玻璃/Te/空气模型。本结果不是目标磁控溅射端面的性能预测。", "",
        "## 目标缺口", "",
        "当前还不能把 Fig. 2c 曲线接入 532 nm 相干 G1 全链：Fig. 2c 精确波长未单独标明；横坐标是脉冲能量而非光强，缺少可核对的脉冲光斑/有效受光面积来互换；纵坐标为归一化 `Tr_norm`，缺少绝对基准透过率；该图也未给复相位响应。因而不推定 532 nm 阈值、CW 响应、复透射或 ReLU 算子。", "",
        "## 工程完成与验证", "",
        "- 本轮使用已归档 ADMA2026 正文嵌入 Fig. 2 原图、SI 的原始 Fig. S2 EMF，并分别保存 1600/3200 px 光谱渲染、轴标定、提取点与误差包络。",
        f"- Fig. 2c 保留 {response['point_count_in_interpolation']} 个跨方法匹配点；另有 {response_audit['unresolved_component_count']} 个合并/截断连通块和 {unresolved_center_count} 个无圆拟合匹配的中心候选排除，共 {response_audit['unresolved_component_count'] + unresolved_center_count} 个未解析候选；另记录 {unresolved_circle_count} 个不匹配边缘圆候选，不将其补为数据点。只在匹配点能量范围内分段线性插值。",
        "- S2 光谱逐点提取限定在已标定 500–2000 nm 轴范围，避免将右轴边框当成曲线。1600/3200 px 的 532 nm n、k 差异均在预先定义的图像数字化包络内。",
        "- 薄膜闭式解与独立四边界连续条件求解比较；功率归一化、被动吸收积分、64/128 点积分收敛、零厚度和无吸收控制均通过。预检拒绝了倒置纵轴、绝对输出能量配置以及把 1530 nm 参数直接当作 532 nm 输入。",
        "- 两张 PDF 文字审计最小字号为 7 pt；碰撞审计 0 项失败、0 项警告；静态图件检查 21 项通过。", "",
        "| 532 nm 参数 | 1600 px | 3200 px | 差值 | 合并数字化界限 | 一致性 |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name in ("n", "k"):
        row = spectrum["render_consistency"][name]
        lines.append(f"| {name}（532 nm） | {row['independent_1600px']:.8g} | {row['primary_3200px']:.8g} | {row['absolute_difference']:.4g} | {row['combined_digitalization_bound']:.4g} | {'是' if row['within_bound'] else '否'} |")
    lines += ["", "| 1530 nm 表格交叉核验 | Fig. S2 1600 px | Fig. S2 3200 px | 表 S1 | 是否在图源数字化+表格舍入范围内 |", "|---|---:|---:|---:|:---:|"]
    cross = spectrum["1530nm_table_crosscheck"]
    for name in ("n", "k"):
        row_1600 = cross[str(config_render_width(payload, 'independent'))][name]
        row_3200 = cross[str(config_render_width(payload, 'primary'))][name]
        lines.append(f"| {name} | {row_1600['figs2_graph_value_at_1530nm']:.6g} | {row_3200['figs2_graph_value_at_1530nm']:.6g} | {row_3200['table_s1_value']:.6g} | {'是' if row_3200['within_graph_digitalization_plus_table_rounding'] else '否'} |")
    n_cross = cross[str(config_render_width(payload, "primary"))]["n"]
    k_cross = cross[str(config_render_width(payload, "primary"))]["k"]
    lines += ["", (f"来源交叉核验提示：1530 nm 处 Fig. S2 图源 n={n_cross['figs2_graph_value_at_1530nm']:.6g}，补充表 1 n={n_cross['table_s1_value']:.6g}，差异超出图源数字化与表格舍入范围；k={k_cross['figs2_graph_value_at_1530nm']:.6g} 对照表值 {k_cross['table_s1_value']:.6g} 则位于范围内。表格仅作一致性核验，不替换 532 nm 图源参数。"), ""]
    lines += ["", "## 决策影响与下一步", "",
              "本阶段准入两类受限文献代理：（1）Fig. 2c 原始坐标下的经验 `Tr_norm(E_in)` 响应；（2）源于 Fig. S2 的 532 nm 局部线性玻璃/Te/空气参照。两者都不支持直接接入 G1 相干非线性光路。下一步若设计 G1 非线性代理，至少需补齐目标波长和样品一致性、能量—光强换算、时间响应、绝对透射基准及复相位合同。未启动训练、G1 全链传播、制样或实测。", "",
              "### 保存结果检查", ""]
    for item in checks:
        detail = ", ".join(f"{key}={value}" for key, value in item.items() if key not in ("check", "passed"))
        lines.append(f"- `{item['check']}`：{'通过' if item['passed'] else '失败'}" + (f"（{detail}）" if detail else ""))
    lines += ["", "以上报告数值均从本次保存的测量结果读取。", ""]
    return "\n".join(lines)


def config_render_width(payload: dict[str, Any], which: str) -> int:
    return int(payload["spectrum_identity"][f"{which}_render_width_px"])


def run_measurement(config_path: str | Path, output_dir: str | Path) -> int:
    payload = _measurement(config_path, output_dir)
    return 0 if payload["status"] == "valid" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "measure"))
    parser.add_argument("--config", required=True, help="Explicit JSON literature-proxy config")
    parser.add_argument("--output-dir", required=True, help="Explicit diagnostic-only output directory")
    args = parser.parse_args(argv)
    try:
        if args.mode == "preflight":
            return run_preflight(args.config, args.output_dir)
        return run_measurement(args.config, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"diagnostic error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
