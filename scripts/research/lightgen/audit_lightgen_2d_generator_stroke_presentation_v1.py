"""Audit the saved CMOS-to-review presentation path without new optical work."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from scripts.research.lightgen import execute_lightgen_2d_generator_stroke_fidelity_comparison_v1 as stroke


TASK_ID = "M0_GENERATOR_STROKE_PRESENTATION_LOCALIZATION_V1"
SOURCE_TASK_ID = stroke.TASK_ID
DEFAULT_ARTIFACT_DIR = Path("outputs/20261008_m0_generator_stroke_fidelity_comparison_v1/runs/attempt_08")
DEFAULT_RATINGS = Path(r"C:\Users\24567\Downloads\lightgen_stroke_fidelity_ratings.csv")
DEFAULT_OUTPUT = Path("outputs/20261008_m0_generator_stroke_presentation_localization_v1/runs/attempt_01")
ENDPOINT_FILES = {"A4700": "readouts_start.pt", "A5400": "readouts_A5400.pt", "B5400": "readouts_B5400.pt"}
REVIEW_ENDPOINTS = {"A4700": "start", "A5400": "A", "B5400": "B"}
FOCUS_CLASSES = (2, 4)
CONTROL_CLASSES = (1, 7, 9)
DISPLAY_CLASSES = FOCUS_CLASSES + CONTROL_CLASSES


class PresentationAuditError(RuntimeError):
    """Invalid source evidence or an incomplete display-path audit."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PresentationAuditError(message)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _image_stages(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    _require(raw.ndim == 4 and raw.shape[1:] == (1, 200, 200), "raw CMOS must be [N,1,200,200]")
    _require(bool(torch.isfinite(raw).all()) and bool((raw >= 0).all()), "raw CMOS must be finite nonnegative intensity")
    raw_pool = F.adaptive_avg_pool2d(raw, (28, 28))
    production = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))
    pool_then_clamp = raw_pool.clamp(0.0, 1.0)
    quantized = production.clamp(0.0, 1.0).mul(255.0).round().div(255.0)
    return {
        "raw_pool_unclipped": raw_pool,
        "production_clamp_then_pool": production,
        "counterfactual_pool_then_clamp": pool_then_clamp,
        "page_uint8_roundtrip": quantized,
    }


def _independent_adaptive_average(image: Sequence[Sequence[float]], output_size: int = 28) -> list[list[float]]:
    height, width = len(image), len(image[0])
    result: list[list[float]] = []
    for out_y in range(output_size):
        row = []
        top = math.floor(out_y * height / output_size)
        bottom = math.ceil((out_y + 1) * height / output_size)
        for out_x in range(output_size):
            left = math.floor(out_x * width / output_size)
            right = math.ceil((out_x + 1) * width / output_size)
            values = [float(image[y][x]) for y in range(top, bottom) for x in range(left, right)]
            row.append(math.fsum(values) / len(values))
        result.append(row)
    return result


def _review_pngs(html: str) -> tuple[dict[str, bytes], dict[str, str]]:
    marker = "const items="
    start = html.find(marker)
    _require(start >= 0, "review page has no embedded image list")
    start += len(marker)
    end_marker = "];const pageSize=20;"
    end = html.find(end_marker, start)
    _require(end >= 0, "review page image list has no production terminator")
    items = json.loads(html[start : end + 1])
    _require(isinstance(items, list) and len(items) == stroke.REVIEW_TOTAL, "review page item count differs from 680")
    pngs: dict[str, bytes] = {}
    uris: dict[str, str] = {}
    for item in items:
        item_id, uri = str(item.get("item_id", "")), str(item.get("image", ""))
        _require(item_id and item_id not in pngs and uri.startswith("data:image/png;base64,"), "review page contains an invalid or duplicate image item")
        data = base64.b64decode(uri.partition(",")[2], validate=True)
        with Image.open(io.BytesIO(data)) as image:
            _require(image.mode == "L" and image.size == (28, 28), "review page image is not the declared 28x28 grayscale view")
        pngs[item_id] = data
        uris[item_id] = uri
    return pngs, uris


def _validate_population_and_readouts(
    population: Mapping[str, Any],
    readouts: Mapping[str, Mapping[str, Any]],
    mapping: Sequence[Mapping[str, Any]],
    pngs: Mapping[str, bytes],
    png_uris: Mapping[str, str],
) -> dict[str, Any]:
    rows = torch.as_tensor(population.get("source_row_indices"), dtype=torch.long)
    labels = torch.as_tensor(population.get("labels"), dtype=torch.long)
    roles = population.get("roles")
    _require(rows.shape == labels.shape == (500,) and len(roles or []) == 500, "source population must retain its 200 fit and 300 confirmation rows")
    _require(roles == ["fit"] * 200 + ["confirmation"] * 300, "source population role order differs")
    _require(torch.unique(rows).numel() == 500, "source population row indices are not unique")
    stroke._validate_review_mapping(mapping)
    _require(set(pngs) == {str(row["item_id"]) for row in mapping}, "review page IDs differ from its private mapping")
    reference = readouts["A4700"]
    expected_shapes = {"source_indices": (500,), "labels": (500,), "raw_cmos": (500, 1, 200, 200), "evaluator_view": (500, 1, 28, 28), "target_slm_intensity": (500, 1, 200, 200), "z": (500, 100)}
    for endpoint, readout in readouts.items():
        for key, shape in expected_shapes.items():
            _require(tuple(readout[key].shape) == shape, f"{endpoint} {key} shape differs")
        _require(torch.equal(readout["source_indices"].to(torch.long), rows), f"{endpoint} row mapping differs from input population")
        _require(torch.equal(readout["labels"].to(torch.long), labels), f"{endpoint} labels differ from input population")
        finite_fields = ("raw_cmos", "evaluator_view", "target_slm_intensity", "z", "e_s_logits", "e_i_logits", "e_c_logits", "e_d_logits")
        _require(all(torch.isfinite(readout[key].real if readout[key].is_complex() else readout[key]).all().item() for key in finite_fields), f"{endpoint} readout contains nonfinite values")
        production = F.adaptive_avg_pool2d(readout["raw_cmos"].clamp(0.0, 1.0), (28, 28))
        _require(torch.equal(_image_stages(readout["raw_cmos"])["production_clamp_then_pool"], production), f"{endpoint} audit producer differs from the production clamp/pool order")
        _require(torch.equal(production, readout["evaluator_view"]), f"{endpoint} saved view differs from the production raw-to-view conversion")
        if endpoint != "A4700":
            _require(torch.equal(readout["target_slm_intensity"], reference["target_slm_intensity"]), f"{endpoint} target rows differ")
            _require(torch.equal(readout["z"], reference["z"]), f"{endpoint} OLS rows differ")

    endpoint_name = {"start": "A4700", "A": "A5400", "B": "B5400"}
    primary = [row for row in mapping if row["kind"] == "primary_generated"]
    for row in primary:
        key = endpoint_name[str(row["endpoint"])]
        position = int(row["measurement_position"])
        readout = readouts[key]
        _require(int(rows[position]) == int(row["source_row_index"]), "review mapping source row differs from population")
        _require(int(labels[position]) == int(row["true_class"]), "review mapping class differs from population")
        expected_uri = stroke._image_data_uri(readout["evaluator_view"][position, 0])
        _require(png_uris[str(row["item_id"])] == expected_uri, "review PNG differs from the saved endpoint view")
    repeats = [row for row in mapping if row["kind"] == "hidden_repeat"]
    for row in repeats:
        original = str(row["repeat_of"])
        _require(pngs[str(row["item_id"])] == pngs[original], "hidden repeat PNG differs from its primary image")
    return {
        "population_rows": 500,
        "fit_rows": 200,
        "confirmation_rows": 300,
        "row_label_role_alignment": True,
        "saved_production_view_exact_replay": True,
        "review_item_ids_match_private_mapping": True,
        "primary_generated_pngs_match_saved_views": len(primary),
        "hidden_repeat_pngs_pixel_identical": len(repeats),
        "target_and_ols_identical_across_endpoints": True,
    }


def _masked_average(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (value * mask).flatten(1).sum(1)
    denominator = mask.flatten(1).sum(1)
    return numerator / denominator.clamp_min(1.0)


def _audit_row_batch(raw: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    stages = _image_stages(raw)
    target_view = F.adaptive_avg_pool2d(target.clamp(0.0, 1.0), (28, 28))
    raw_error = (raw - target).square()
    view_error = (stages["production_clamp_then_pool"] - target_view).square()
    raw_foreground = target > 0.1
    raw_background = ~raw_foreground
    view_foreground = F.adaptive_max_pool2d(raw_foreground.to(torch.float32), (28, 28)).to(torch.bool)
    view_background = ~view_foreground
    clipped = (raw - raw.clamp(0.0, 1.0)).clamp_min(0.0)
    raw_sum = raw.flatten(1).sum(1).clamp_min(1e-30)
    foreground_sum = (raw * raw_foreground).flatten(1).sum(1).clamp_min(1e-30)
    view_base, view_unclipped, view_alt, view_quant = (
        stages["production_clamp_then_pool"],
        stages["raw_pool_unclipped"],
        stages["counterfactual_pool_then_clamp"],
        stages["page_uint8_roundtrip"],
    )
    return {
        "raw_foreground_mse": _masked_average(raw_error, raw_foreground),
        "raw_background_mse": _masked_average(raw_error, raw_background),
        "raw_foreground_mean": _masked_average(raw, raw_foreground),
        "target_foreground_mean": _masked_average(target, raw_foreground),
        "raw_background_mean": _masked_average(raw, raw_background),
        "view_foreground_mse": _masked_average(view_error, view_foreground),
        "view_background_mse": _masked_average(view_error, view_background),
        "raw_clipped_pixel_fraction": (raw > 1.0).flatten(1).to(torch.float64).mean(1),
        "raw_clipped_foreground_pixel_fraction": ((raw > 1.0) & raw_foreground).flatten(1).sum(1).to(torch.float64) / raw_foreground.flatten(1).sum(1).clamp_min(1),
        "clipped_intensity_fraction_all": clipped.flatten(1).sum(1) / raw_sum,
        "clipped_intensity_fraction_foreground": (clipped * raw_foreground).flatten(1).sum(1) / foreground_sum,
        "production_vs_pool_then_clamp_mean_abs": (view_base - view_alt).abs().flatten(1).mean(1),
        "production_vs_pool_then_clamp_max_abs": (view_base - view_alt).abs().flatten(1).amax(1),
        "rawpool_vs_production_mean_abs": (view_unclipped - view_base).abs().flatten(1).mean(1),
        "page_quantization_rmse": (view_quant - view_base).square().flatten(1).mean(1).sqrt(),
        "production_view_ssim": stroke.ssim_per_image(view_base, target_view),
        "raw_zero": raw.flatten(1).amax(1) == 0,
        "raw_constant": (raw.flatten(1).amax(1) - raw.flatten(1).amin(1)) == 0,
    }


def _image_pil(value: torch.Tensor, *, size: int, clamp: bool = True) -> Image.Image:
    image = value.detach().cpu().to(torch.float32).squeeze()
    if clamp:
        image = image.clamp(0.0, 1.0)
    pixels = image.mul(255.0).round().to(torch.uint8).tolist()
    rendered = Image.frombytes("L", (len(pixels[0]), len(pixels)), bytes(pixel for row in pixels for pixel in row))
    return rendered.resize((size, size), resample=Image.Resampling.NEAREST) if size != rendered.width else rendered


def _write_contact_sheets(
    target_dir: Path,
    population: Mapping[str, Any],
    mapping: Sequence[Mapping[str, Any]],
    answers: Mapping[str, str],
    readouts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = torch.as_tensor(population["source_row_indices"], dtype=torch.long).tolist()
    labels = torch.as_tensor(population["labels"], dtype=torch.long).tolist()
    role_names = list(population["roles"])
    display_rows = [row for row in mapping if row["kind"] == "primary_generated" and int(row["true_class"]) in DISPLAY_CLASSES]
    manifest = []
    for digit in DISPLAY_CLASSES:
        for endpoint, readout_key in (("start", "A4700"), ("A", "A5400"), ("B", "B5400")):
            subset = sorted((row for row in display_rows if row["endpoint"] == endpoint and int(row["true_class"]) == digit), key=lambda row: int(row["measurement_position"]))
            _require(len(subset) == stroke.REVIEW_ROWS_PER_CLASS, f"review page class {digit} endpoint {endpoint} has an unexpected row count")
            page = Image.new("RGB", (1_360, 20 + len(subset) * 230), "white")
            draw = ImageDraw.Draw(page)
            for x, title in ((10, "source / answer"), (175, "target 200"), (390, "raw CMOS, display capped"), (605, "production view"), (790, "raw pool*"), (975, "pool then clip"), (1_160, "page uint8")):
                draw.text((x, 4), title, fill="black")
            draw.text((10, page.height - 16), "*Unclipped raw pool statistics are preserved separately; image panels use a fixed [0,1] display range.", fill="black")
            readout = readouts[readout_key]
            for index, row in enumerate(subset):
                y = 22 + index * 230
                position = int(row["measurement_position"])
                source_row = int(row["source_row_index"])
                _require(rows[position] == source_row and labels[position] == digit, "contact sheet row differs from review mapping")
                raw = readout["raw_cmos"][position]
                stages = _image_stages(raw.unsqueeze(0))
                target200 = readout["target_slm_intensity"][position]
                panels = (
                    (175, _image_pil(target200, size=200)),
                    (390, _image_pil(raw, size=200)),
                    (605, _image_pil(stages["production_clamp_then_pool"], size=140)),
                    (790, _image_pil(stages["raw_pool_unclipped"], size=140)),
                    (975, _image_pil(stages["counterfactual_pool_then_clamp"], size=140)),
                    (1_160, _image_pil(stages["page_uint8_roundtrip"], size=140)),
                )
                for x, image in panels:
                    page.paste(image.convert("RGB"), (x, y))
                answer = answers.get(str(row["item_id"]), "not-reviewed")
                e_d = int(readout["e_d_logits"][position].to(torch.float64).argmax())
                draw.text((10, y + 18), f"row={source_row}", fill="black")
                draw.text((10, y + 40), f"class={digit} role={role_names[position]}", fill="black")
                draw.text((10, y + 62), f"human={answer}", fill="black")
                draw.text((10, y + 84), f"E_D={e_d}", fill="black")
                manifest.append({"endpoint": endpoint, "class": digit, "population_position": position, "source_row_index": source_row, "review_item_id": row["item_id"], "human_answer": answer, "e_d_prediction": e_d})
            filename = f"class_{digit}_{endpoint}.png"
            path = target_dir / filename
            _require(not path.exists(), f"refusing to overwrite contact sheet: {path}")
            page.save(path, format="PNG", optimize=True)
    return {
        "class_endpoints": len(DISPLAY_CLASSES) * 3,
        "reviewed_rows": len(manifest),
        "panels_per_row": 6,
        "files": [f"class_{digit}_{endpoint}.png" for digit in DISPLAY_CLASSES for endpoint in ("start", "A", "B")],
        "row_manifest": manifest,
        "raw_pool_panel_display_cap": 1.0,
    }


def _summarize_endpoint(
    endpoint: str,
    readout: Mapping[str, torch.Tensor],
    population: Mapping[str, Any],
    mapping: Sequence[Mapping[str, Any]],
    answers: Mapping[str, str],
    rows_writer: csv.DictWriter,
) -> dict[str, Any]:
    labels = readout["labels"].to(torch.long)
    roles = list(population["roles"])
    source_rows = readout["source_indices"].to(torch.long).tolist()
    positions_for = {(int(row["measurement_position"]), str(row["endpoint"])): row for row in mapping if row["kind"] == "primary_generated"}
    by_role_class: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    totals: dict[str, list[float]] = defaultdict(list)
    maxima: dict[str, float] = {}
    zero_images = constant_images = rows_with_clip = 0
    finite = True
    with torch.no_grad():
        for start in range(0, 500, 25):
            stop = min(start + 25, 500)
            raw = readout["raw_cmos"][start:stop].to(torch.float32)
            target = readout["target_slm_intensity"][start:stop].to(torch.float32)
            values = _audit_row_batch(raw, target)
            finite = finite and all(bool(torch.isfinite(value).all()) for value in values.values() if value.dtype != torch.bool)
            zero_images += int(values["raw_zero"].sum())
            constant_images += int(values["raw_constant"].sum())
            for offset in range(stop - start):
                position = start + offset
                digit = int(labels[position])
                role = roles[position]
                prediction = int(readout["e_d_logits"][position].to(torch.float64).argmax())
                human_row = positions_for.get((position, REVIEW_ENDPOINTS[endpoint]))
                answer = "" if human_row is None else answers[str(human_row["item_id"])]
                item: dict[str, Any] = {
                    "endpoint": endpoint,
                    "population_position": position,
                    "source_row_index": source_rows[position],
                    "role": role,
                    "true_class": digit,
                    "e_d_prediction": prediction,
                    "review_item_id": "" if human_row is None else human_row["item_id"],
                    "human_answer": answer,
                    "human_correct": "" if not answer else int(answer == str(digit)),
                    "human_unclear": int(answer == "unclear"),
                }
                for name, tensor in values.items():
                    item[name] = bool(tensor[offset]) if tensor.dtype == torch.bool else float(tensor[offset])
                    if tensor.dtype != torch.bool:
                        totals[name].append(float(tensor[offset]))
                        maxima[name] = max(maxima.get(name, -math.inf), float(tensor[offset]))
                rows_writer.writerow(item)
                by_role_class[(role, digit)].append(item)
                rows_with_clip += int(item["raw_clipped_pixel_fraction"] > 0)
    role_summary = {}
    for role in sorted(set(roles)):
        role_summary[role] = {}
        for digit in range(10):
            group = by_role_class[(role, digit)]
            if not group:
                continue
            role_summary[role][str(digit)] = {
                "count": len(group),
                "human_reviewed": sum(bool(row["review_item_id"]) for row in group),
                "human_correct": sum(row["human_answer"] == str(digit) for row in group),
                "human_unclear": sum(row["human_answer"] == "unclear" for row in group),
                "e_d_correct": sum(row["e_d_prediction"] == digit for row in group),
                "raw_foreground_mse_mean": math.fsum(row["raw_foreground_mse"] for row in group) / len(group),
                "raw_background_mse_mean": math.fsum(row["raw_background_mse"] for row in group) / len(group),
                "view_foreground_mse_mean": math.fsum(row["view_foreground_mse"] for row in group) / len(group),
                "view_background_mse_mean": math.fsum(row["view_background_mse"] for row in group) / len(group),
                "production_view_ssim_mean": math.fsum(row["production_view_ssim"] for row in group) / len(group),
                "raw_clipped_pixel_fraction_mean": math.fsum(row["raw_clipped_pixel_fraction"] for row in group) / len(group),
                "clipped_intensity_fraction_foreground_mean": math.fsum(row["clipped_intensity_fraction_foreground"] for row in group) / len(group),
                "production_vs_pool_then_clamp_mean_abs": math.fsum(row["production_vs_pool_then_clamp_mean_abs"] for row in group) / len(group),
            }
    per_class_human = {}
    for digit in range(10):
        group = by_role_class[("confirmation", digit)]
        reviewed = [row for row in group if row["review_item_id"]]
        class_answers = [row["human_answer"] for row in reviewed]
        per_class_human[str(digit)] = {
            "human_reviewed": len(reviewed),
            "human_correct": sum(answer == str(digit) for answer in class_answers),
            "human_unclear": sum(answer == "unclear" for answer in class_answers),
            "human_predicted_as_9": sum(answer == "9" for answer in class_answers),
            "e_d_correct_all_confirmation": sum(row["e_d_prediction"] == digit for row in group),
            "confirmation_rows": len(group),
            "raw_foreground_mse_mean": math.fsum(row["raw_foreground_mse"] for row in group) / len(group),
            "raw_background_mse_mean": math.fsum(row["raw_background_mse"] for row in group) / len(group),
            "view_foreground_mse_mean": math.fsum(row["view_foreground_mse"] for row in group) / len(group),
            "view_background_mse_mean": math.fsum(row["view_background_mse"] for row in group) / len(group),
            "production_view_ssim_mean": math.fsum(row["production_view_ssim"] for row in group) / len(group),
            "raw_clipped_pixel_fraction_mean": math.fsum(row["raw_clipped_pixel_fraction"] for row in group) / len(group),
            "clipped_intensity_fraction_foreground_mean": math.fsum(row["clipped_intensity_fraction_foreground"] for row in group) / len(group),
            "production_vs_pool_then_clamp_mean_abs": math.fsum(row["production_vs_pool_then_clamp_mean_abs"] for row in group) / len(group),
        }
    return {
        "endpoint": endpoint,
        "finite": finite,
        "rows": len(labels),
        "raw_zero_images": zero_images,
        "raw_constant_images": constant_images,
        "rows_with_any_clipped_pixel": rows_with_clip,
        "all_rows_maxima": maxima,
        "confirmation_by_class": per_class_human,
        "by_role_and_class": role_summary,
        "all_rows_means": {key: math.fsum(values) / len(values) for key, values in totals.items() if values},
    }


def _load_sources(artifact_dir: Path, ratings_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Mapping[str, Any]], dict[str, bytes], dict[str, str], dict[str, Any]]:
    report_path = artifact_dir / "report.json"
    _require(report_path.is_file(), "stroke-fidelity report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("task_id") == SOURCE_TASK_ID, "source report belongs to another task")
    stroke._finalize_ratings(artifact_dir, ratings_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rating_payload = json.loads((artifact_dir / "ratings_result.json").read_text(encoding="utf-8"))
    population = json.loads((artifact_dir / "input_population.json").read_text(encoding="utf-8"))
    mapping_payload = json.loads((artifact_dir / "review_mapping.json").read_text(encoding="utf-8"))
    _require(mapping_payload.get("task_id") == SOURCE_TASK_ID and rating_payload.get("task_id") == SOURCE_TASK_ID, "review or rating artifacts belong to another task")
    answers = rating_payload.get("answers")
    mapping = mapping_payload.get("items")
    stroke._validate_review_mapping(mapping)
    _require(stroke._parse_ratings(ratings_path, mapping) == answers, "explicit rating CSV differs from the saved answers")
    html = (artifact_dir / "blind_review.html").read_text(encoding="utf-8")
    pngs, png_uris = _review_pngs(html)
    readouts = {name: torch.load(artifact_dir / filename, map_location="cpu", weights_only=True, mmap=True) for name, filename in ENDPOINT_FILES.items()}
    validation = _validate_population_and_readouts(population, readouts, mapping, pngs, png_uris)
    return report, {"mapping": mapping, "answers": answers, "population": population, "rating_result": rating_payload}, readouts, pngs, png_uris, validation


def audit(artifact_dir: Path, ratings_path: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), "refusing to overwrite an existing localization output directory")
    source_report, evidence, readouts, _pngs, _png_uris, validation = _load_sources(artifact_dir, ratings_path)
    mapping, answers, population = evidence["mapping"], evidence["answers"], evidence["population"]
    output_dir.mkdir(parents=True, exist_ok=False)
    row_path = output_dir / "derived_rows.csv"
    endpoint_summaries = {}
    with row_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "endpoint", "population_position", "source_row_index", "role", "true_class", "e_d_prediction", "review_item_id", "human_answer", "human_correct", "human_unclear",
            "raw_foreground_mse", "raw_background_mse", "raw_foreground_mean", "target_foreground_mean", "raw_background_mean", "view_foreground_mse", "view_background_mse",
            "raw_clipped_pixel_fraction", "raw_clipped_foreground_pixel_fraction", "clipped_intensity_fraction_all", "clipped_intensity_fraction_foreground",
            "production_vs_pool_then_clamp_mean_abs", "production_vs_pool_then_clamp_max_abs", "rawpool_vs_production_mean_abs", "page_quantization_rmse", "production_view_ssim", "raw_zero", "raw_constant",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for endpoint, readout in readouts.items():
            endpoint_summaries[endpoint] = _summarize_endpoint(endpoint, readout, population, mapping, answers, writer)
    sheets = _write_contact_sheets(output_dir, population, mapping, answers, readouts)
    human_review = source_report.get("human_review_result", {})
    inconclusive = source_report.get("status") == "REVIEW_INCONCLUSIVE" or not human_review.get("review_quality", {}).get("passed", False)
    report = {
        "task_id": TASK_ID,
        "status": "EVIDENCE_REVIEW_COMPLETED",
        "source_task_id": SOURCE_TASK_ID,
        "source_report_status": source_report.get("status"),
        "human_review_decision": {"status": source_report.get("status"), "classification": source_report.get("classification"), "quality": human_review.get("review_quality"), "scientific_claim_eligible": not inconclusive},
        "scope": {"model_updates": 0, "optical_forwards": 0, "evaluator_forwards": 0, "gradients": 0, "new_samples": 0, "new_human_ratings": 0, "source_population_rows": 500, "endpoint_rows": 1500, "canonical_train_not_independent_generalization": True},
        "evidence_chain_checks": validation,
        "display_definitions": {"production": "clamp raw CMOS to [0,1], then adaptive average pool to 28x28", "counterfactual": "adaptive average pool unclamped raw CMOS to 28x28, then clamp to [0,1]", "page_pixels": "round(production_view*255)/255", "foreground": "target_slm_intensity > 0.1 at the matching 200x200 raw size", "target_note": "saved 200x200 interpolated target intensity; source image files were not reloaded"},
        "endpoint_summaries": endpoint_summaries,
        "human_error_localization": {endpoint: {str(digit): endpoint_summaries[endpoint]["confirmation_by_class"][str(digit)] for digit in DISPLAY_CLASSES} for endpoint in ENDPOINT_FILES},
        "repeat_disagreements": [
            {"repeat_item_id": row["item_id"], "primary_item_id": row["repeat_of"], "endpoint": row["endpoint"], "class": int(row["true_class"]), "repeat_answer": answers[str(row["item_id"])], "primary_answer": answers[str(row["repeat_of"])], "agree": False}
            for row in mapping if row["kind"] == "hidden_repeat" and answers[str(row["item_id"])] != answers[str(row["repeat_of"])]
        ],
        "derived_rows_csv": row_path.name,
        "contact_sheets": sheets,
        "decision": {"training_recommendation": "NONE", "next_action": "NONE", "conclusion": "If the stored view and review PNG match, clipping-order and raw-image errors remain descriptive evidence and cannot alone establish the cause of human unreadability.", "stop": True},
    }
    _write_json(output_dir / "report.json", report)
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = stroke._root()
    artifact_dir = args.artifact_dir.resolve() if args.artifact_dir.is_absolute() else (root / args.artifact_dir).resolve()
    ratings = args.ratings.resolve() if args.ratings.is_absolute() else (root / args.ratings).resolve()
    output = args.output_dir.resolve() if args.output_dir.is_absolute() else (root / args.output_dir).resolve()
    outputs = (root / "outputs").resolve()
    _require(output != outputs and outputs in output.parents, "output must be a child of outputs/")
    parts = output.relative_to(outputs).parts
    _require(len(parts) == 3 and parts[1] == "runs" and parts[0].startswith("20261008_"), "output must use the declared versioned runs path")
    _require(not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in parts), "protected, restored, or temporary output namespace is forbidden")
    result = audit(artifact_dir, ratings, output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
