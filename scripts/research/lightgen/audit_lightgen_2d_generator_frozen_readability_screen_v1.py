"""Prepare and score a blinded, single-reviewer LightGen readability screen."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image


TASK_ID = "M0_GENERATOR_FROZEN_READABILITY_SCREEN_V1"
PARENT_TASK_ID = "M0_GENERATOR_FROZEN_CONDITIONAL_OLS_GENERATION_V1"
PARENT_REPORT_SCHEMA = "m0_generator_frozen_conditional_ols_generation_report.v1"
DEFAULT_GENERATION_DIR = Path(
    "outputs/20261008_m0_generator_frozen_conditional_ols_generation_v1/runs/attempt_02_microbatch1"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20261008_m0_generator_frozen_readability_screen_v1/runs/attempt_01"
)
SELECTION_SEED = 2026100815
ORDER_SEED = 2026100816
GENERATED_PER_CLASS = 20
EMPIRICAL_PER_CLASS = 5
REAL_TARGET_PER_CLASS = 5
DUPLICATES_PER_CLASS = 1
GENERATED_PER_CLASS_TOTAL = 200
ITEM_COUNT = 310
CLASS_COUNT = 10
IMAGE_SIDE = 28
PRIMARY_TOTAL_MIN = 150
PRIMARY_PER_CLASS_MIN = 14
ANCHOR_TOTAL_MIN = 45
REPEAT_AGREEMENT_MIN = 9


class ReadabilityScreenError(RuntimeError):
    """Invalid saved input, blinded review, or rating evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadabilityScreenError(message)


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else root / value).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _load_torch(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required tensor input is missing: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    _require(isinstance(value, dict), f"expected tensor mapping: {path}")
    return value


def _resolve_a4700_target_readout(generation_dir: Path, population: Mapping[str, Any]) -> Path:
    checkpoint_value = population.get("source_checkpoint")
    _require(isinstance(checkpoint_value, str) and bool(checkpoint_value), "A4700 source checkpoint is missing from the saved lineage")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = generation_dir / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    _require(checkpoint_path.name == "arm_a_step4700.pt", "saved generation source checkpoint is not A4700")
    return checkpoint_path.parent / "readouts_A4700.pt"


def _atomic_write(path: Path, data: bytes) -> None:
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pending_", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))


def _load_inputs(generation_dir: Path) -> dict[str, Any]:
    report = _load_json(generation_dir / "report.json")
    _require(report.get("schema") == PARENT_REPORT_SCHEMA, "generation report schema differs")
    _require(
        report.get("task_id") == PARENT_TASK_ID
        and report.get("status") == "VALID_COMPLETED"
        and report.get("classification") == "CONDITIONAL_OLS_GENERATION_ENTRY_SUPPORTED",
        "source is not the named completed frozen conditional OLS generation",
    )
    population = _load_json(generation_dir / "corrected_input_population.json")
    _require(population.get("task_id") == "M0_GENERATOR_TRAINING_POPULATION_EXPANSION_COMPARISON_V1", "corrected source population identity differs")
    source_rows = population.get("source_row_indices")
    labels = population.get("labels")
    _require(isinstance(source_rows, list) and isinstance(labels, list) and len(source_rows) == len(labels) == 3_300, "corrected TRAIN population must contain 3,300 rows")
    _require(len(set(source_rows)) == 3_300 and all(0 <= int(label) < CLASS_COUNT for label in labels), "source population rows or labels are invalid")
    _require(population.get("role_positions", {}).get("coverage_confirmation") == [3_000, 3_300], "saved real-image anchors are not the declared coverage-confirmation role")
    anchor_labels = torch.tensor(labels[3_000:3_300], dtype=torch.long)
    _require(torch.bincount(anchor_labels, minlength=CLASS_COUNT).tolist() == [30] * CLASS_COUNT, "real-image anchor population is not 30 per class")

    metrics = _load_torch(generation_dir / "sample_row_metrics.pt")
    samples = metrics.get("sample_evaluator_view")
    requested = metrics.get("sample_requested_class")
    sample_index = metrics.get("sample_index_within_class")
    e_d_prediction = metrics.get("e_d_predictions_diagonal_gaussian_sample")
    _require(isinstance(samples, torch.Tensor) and tuple(samples.shape) == (2_000, 1, IMAGE_SIDE, IMAGE_SIDE), "saved sampled views must be 2,000 fixed 28x28 images")
    _require(isinstance(requested, torch.Tensor) and tuple(requested.shape) == (2_000,), "saved requested labels are incomplete")
    _require(isinstance(sample_index, torch.Tensor) and tuple(sample_index.shape) == (2_000,), "saved within-class sample indices are incomplete")
    _require(isinstance(e_d_prediction, torch.Tensor) and tuple(e_d_prediction.shape) == (2_000,), "saved E_D predictions are incomplete")
    expected_requested = torch.arange(CLASS_COUNT).repeat_interleave(200)
    expected_index = torch.arange(200).repeat(CLASS_COUNT)
    _require(torch.equal(requested.to(torch.long), expected_requested) and torch.equal(sample_index.to(torch.long), expected_index), "sample order is not class-major with 200 rows per requested class")
    _require(bool(torch.isfinite(samples).all()) and float(samples.min()) >= 0.0 and float(samples.max()) <= 1.0, "sample views are nonfinite or outside fixed [0,1] display range")
    e_d_prediction = e_d_prediction.to(torch.long)
    _require(bool(((e_d_prediction >= 0) & (e_d_prediction < CLASS_COUNT)).all()), "saved E_D class predictions are outside 0..9")
    expected_correct = [int(((requested == digit) & (e_d_prediction == digit)).sum()) for digit in range(CLASS_COUNT)]
    reported = report.get("results", {}).get("primary_e_d_sample_screen", {}).get("per_class_correct")
    _require(reported == expected_correct, "saved E_D predictions do not reproduce the parent report")

    empirical_views: list[torch.Tensor] = []
    empirical_labels: list[torch.Tensor] = []
    empirical_rows: list[torch.Tensor] = []
    for start, stop in ((0, 100), (100, 200)):
        chunk_path = generation_dir / f"readout_rows_{start:04d}_{stop:04d}.pt"
        chunk = _load_torch(chunk_path)
        _require(tuple(chunk.get("measurement_positions", torch.empty(0)).shape) == (100,), f"empirical readout positions are incomplete in {chunk_path.name}")
        _require(chunk.get("kind") == ["empirical_fit_ols"] * 100, f"unexpected readout kind in {chunk_path.name}")
        _require(torch.equal(chunk["measurement_positions"].to(torch.long), torch.arange(start, stop)), f"empirical readout order differs in {chunk_path.name}")
        empirical_views.append(chunk["evaluator_view"].to(torch.float32))
        empirical_labels.append(chunk["requested_class"].to(torch.long))
        empirical_rows.append(chunk["source_row_indices"].to(torch.long))
    empirical_views_tensor = torch.cat(empirical_views)
    empirical_labels_tensor = torch.cat(empirical_labels)
    empirical_rows_tensor = torch.cat(empirical_rows)
    _require(torch.equal(empirical_labels_tensor, torch.tensor(labels[:200], dtype=torch.long)), "empirical decoded labels differ from the saved fit population")
    _require(torch.equal(empirical_rows_tensor, torch.tensor(source_rows[:200], dtype=torch.long)), "empirical decoded source rows differ from the saved fit population")
    _require(bool(torch.isfinite(empirical_views_tensor).all()) and float(empirical_views_tensor.min()) >= 0.0 and float(empirical_views_tensor.max()) <= 1.0, "empirical views are nonfinite or outside [0,1]")

    target_readout_path = _resolve_a4700_target_readout(generation_dir, population)
    target_readout = _load_torch(target_readout_path)
    _require(torch.equal(target_readout.get("source_indices", torch.empty(0)).to(torch.long), torch.tensor(source_rows, dtype=torch.long)), "saved A4700 target readout source rows differ from corrected population")
    _require(torch.equal(target_readout.get("labels", torch.empty(0)).to(torch.long), torch.tensor(labels, dtype=torch.long)), "saved A4700 target readout labels differ from corrected population")
    target_raw = target_readout.get("target_slm_intensity")
    _require(isinstance(target_raw, torch.Tensor) and tuple(target_raw.shape) == (3_300, 1, 200, 200), "saved TRAIN target images have an unexpected shape")
    target_views = torch.empty((3_300, 1, IMAGE_SIDE, IMAGE_SIDE), dtype=torch.float32)
    for start in range(0, 3_300, 100):
        stop = min(3_300, start + 100)
        target_views[start:stop] = F.adaptive_avg_pool2d(target_raw[start:stop].to(torch.float32), (IMAGE_SIDE, IMAGE_SIDE))
    _require(bool(torch.isfinite(target_views).all()) and float(target_views.min()) >= 0.0 and float(target_views.max()) <= 1.0, "saved TRAIN target views are invalid")

    return {
        "report": report,
        "population": population,
        "source_rows": torch.tensor(source_rows, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_views": samples.to(torch.float32),
        "sample_requested": requested.to(torch.long),
        "sample_index": sample_index.to(torch.long),
        "sample_e_d_prediction": e_d_prediction,
        "empirical_views": empirical_views_tensor,
        "empirical_labels": empirical_labels_tensor,
        "empirical_rows": empirical_rows_tensor,
        "target_views": target_views,
        "generation_dir": generation_dir,
        "target_readout_path": target_readout_path,
    }


def _image_data_uri(view: torch.Tensor) -> str:
    pixels = bytes(view.detach().cpu().clamp(0.0, 1.0).mul(255).round().to(torch.uint8).flatten().tolist())
    image = Image.frombytes("L", (IMAGE_SIDE, IMAGE_SIDE), pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _balanced_positions(labels: torch.Tensor, candidates: Sequence[int], per_class: int, rng: random.Random) -> list[int]:
    result: list[int] = []
    for digit in range(CLASS_COUNT):
        positions = [position for position in candidates if int(labels[position]) == digit]
        _require(len(positions) >= per_class, f"class {digit} has too few rows for the fixed blind sample")
        result.extend(rng.sample(positions, per_class))
    return result


def _build_records(data: Mapping[str, Any], id_factory: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if id_factory is None:
        import secrets

        def id_factory() -> str:
            return secrets.token_hex(6)
    selection_rng = random.Random(SELECTION_SEED)
    generated_positions = _balanced_positions(
        data["sample_requested"], list(range(2_000)), GENERATED_PER_CLASS, selection_rng
    )
    empirical_positions = _balanced_positions(
        data["empirical_labels"], list(range(200)), EMPIRICAL_PER_CLASS, selection_rng
    )
    target_positions = _balanced_positions(
        data["labels"], list(range(3_000, 3_300)), REAL_TARGET_PER_CLASS, selection_rng
    )

    records: list[dict[str, Any]] = []
    primary_by_class: dict[int, list[dict[str, Any]]] = {digit: [] for digit in range(CLASS_COUNT)}

    def add_record(view: torch.Tensor, *, source_type: str, true_class: int, requested_class: int | None = None, sample_index: int | None = None, source_row: int | None = None, e_d_prediction: int | None = None, repeat_of: str | None = None) -> dict[str, Any]:
        record = {
            "item_id": id_factory(),
            "source_type": source_type,
            "true_class": int(true_class),
            "requested_class": None if requested_class is None else int(requested_class),
            "sample_index": None if sample_index is None else int(sample_index),
            "source_row_index": None if source_row is None else int(source_row),
            "e_d_prediction": None if e_d_prediction is None else int(e_d_prediction),
            "repeat_of": repeat_of,
            "_image_data": _image_data_uri(view),
        }
        records.append(record)
        return record

    for position in generated_positions:
        digit = int(data["sample_requested"][position])
        sample_index = int(data["sample_index"][position])
        record = add_record(
            data["sample_views"][position, 0],
            source_type="generated",
            true_class=digit,
            requested_class=digit,
            sample_index=sample_index,
            e_d_prediction=int(data["sample_e_d_prediction"][position]),
        )
        primary_by_class[digit].append(record)

    for position in empirical_positions:
        digit = int(data["empirical_labels"][position])
        add_record(
            data["empirical_views"][position, 0],
            source_type="empirical_ols",
            true_class=digit,
            requested_class=digit,
            sample_index=position,
            source_row=int(data["empirical_rows"][position]),
        )

    for position in target_positions:
        digit = int(data["labels"][position])
        add_record(
            data["target_views"][position, 0],
            source_type="real_target",
            true_class=digit,
            source_row=int(data["source_rows"][position]),
        )

    for digit in range(CLASS_COUNT):
        chosen = selection_rng.choice(primary_by_class[digit])
        add_record(
            data["sample_views"][digit * 200 + int(chosen["sample_index"]), 0],
            source_type="generated_repeat",
            true_class=digit,
            requested_class=digit,
            sample_index=int(chosen["sample_index"]),
            e_d_prediction=int(chosen["e_d_prediction"]),
            repeat_of=str(chosen["item_id"]),
        )

    rng_order = random.Random(ORDER_SEED)
    rng_order.shuffle(records)
    public_items = [{"item_id": record["item_id"], "image": record["_image_data"]} for record in records]
    mapping = [{key: value for key, value in record.items() if key != "_image_data"} for record in records]
    counts = {kind: sum(record["source_type"] == kind for record in mapping) for kind in ("generated", "empirical_ols", "real_target", "generated_repeat")}
    _require(len(mapping) == ITEM_COUNT and len({row["item_id"] for row in mapping}) == ITEM_COUNT, "blind packet does not contain 310 unique item IDs")
    _require(counts == {"generated": 200, "empirical_ols": 50, "real_target": 50, "generated_repeat": 10}, "blind packet source counts differ from the fixed review design")
    _require(all(sum(row["source_type"] == "generated" and row["true_class"] == c for row in mapping) == 20 for c in range(CLASS_COUNT)), "generated review sample is not 20 per requested class")
    return public_items, mapping


def _render_html(public_items: Sequence[Mapping[str, str]], review_id: str) -> str:
    payload = json.dumps(list(public_items), ensure_ascii=False, separators=(",", ":"))
    review_token = json.dumps(review_id)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LightGen 数字盲读</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:20px;background:#f4f4f4;color:#222}}
header{{position:sticky;top:0;background:#fff;padding:12px;z-index:2;border:1px solid #ccc}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px;margin-top:16px}}
.card{{background:#fff;border:1px solid #ddd;padding:10px;text-align:center}}
.card img{{width:196px;height:196px;image-rendering:pixelated;background:#fff;object-fit:contain}}
.id{{font-family:monospace;color:#555;margin:5px}}
select,button{{font-size:16px;padding:7px;margin:4px}}button{{cursor:pointer}}
@media(max-width:850px){{.grid{{grid-template-columns:repeat(2,minmax(140px,1fr))}}.card img{{width:150px;height:150px}}}}
</style></head><body>
<header><strong>LightGen 数字盲读</strong>
<p>只看图像本身，选择你读到的数字；无法辨认时选“看不清”。图像采用统一灰度范围，不做单图增强。不要查看其他带标签图库，完成后再提交评分。</p>
<span id="progress"></span><div><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button><button id="export">导出全部评分 CSV</button></div>
<span id="message">每页20张。全部310项完成后才能导出。</span></header>
<main id="grid" class="grid"></main>
<script>
const items={payload}; const reviewId={review_token}; const pageSize=20; let page=0; const answers={{}};
try{{const saved=localStorage.getItem('lightgen-readability-'+reviewId);if(saved)Object.assign(answers,JSON.parse(saved));}}catch{{}}
const grid=document.getElementById('grid');
function render(){{grid.innerHTML='';const start=page*pageSize;const visible=items.slice(start,start+pageSize);
for(const item of visible){{const card=document.createElement('section');card.className='card';
const image=document.createElement('img');image.src=item.image;image.alt='';image.draggable=false;
const id=document.createElement('div');id.className='id';id.textContent='编号 '+item.item_id;
const select=document.createElement('select');select.setAttribute('aria-label','编号 '+item.item_id+' 的读数');
const blank=document.createElement('option');blank.value='';blank.textContent='请选择';select.appendChild(blank);
for(let digit=0;digit<10;digit++){{const option=document.createElement('option');option.value=String(digit);option.textContent=String(digit);select.appendChild(option);}}
const unclear=document.createElement('option');unclear.value='unclear';unclear.textContent='看不清';select.appendChild(unclear);
select.value=answers[item.item_id]||'';select.onchange=()=>{{if(select.value)answers[item.item_id]=select.value;else delete answers[item.item_id];try{{localStorage.setItem('lightgen-readability-'+reviewId,JSON.stringify(answers));}}catch{{}}updateProgress();}};
card.append(image,id,select);grid.appendChild(card);}}
document.getElementById('page').textContent=`第 ${{page+1}} / ${{Math.ceil(items.length/pageSize)}} 页`;
document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=start+pageSize>=items.length;updateProgress();}}
function updateProgress(){{const count=Object.keys(answers).length;document.getElementById('progress').textContent=`已完成 ${{count}} / ${{items.length}} 项`;
document.getElementById('export').disabled=count!==items.length;document.getElementById('message').textContent=count===items.length?'已完成，可以导出评分 CSV。':'每页20张。全部310项完成后才能导出。';}}
document.getElementById('prev').onclick=()=>{{page--;render();}};document.getElementById('next').onclick=()=>{{page++;render();}};
document.getElementById('export').onclick=()=>{{if(Object.keys(answers).length!==items.length)return;
const lines=['item_id,answer',...items.map(item=>`${{item.item_id}},${{answers[item.item_id]}}`)];
const blob=new Blob(['\\ufeff'+lines.join('\\r\\n')],{{type:'text/csv;charset=utf-8'}});const link=document.createElement('a');
link.href=URL.createObjectURL(blob);link.download=`lightgen_readability_ratings_${{reviewId}}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);}};
render();
</script></body></html>"""


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _parse_ratings(path: Path, mapping: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    _require(path.is_file(), f"explicit ratings CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == ["item_id", "answer"], "ratings CSV header must be exactly item_id,answer")
        rows = list(reader)
    expected = {str(record["item_id"]) for record in mapping}
    answers: dict[str, str] = {}
    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        answer = str(row.get("answer", "")).strip().lower()
        _require(item_id in expected, f"ratings CSV contains an unknown item ID: {item_id}")
        _require(item_id not in answers, f"ratings CSV repeats an item ID: {item_id}")
        _require(answer in {str(digit) for digit in range(10)} | {"unclear"}, f"invalid rating for item {item_id}")
        answers[item_id] = answer
    _require(set(answers) == expected, "ratings CSV is missing one or more item IDs")
    return answers


def _validate_mapping(mapping: Sequence[Mapping[str, Any]]) -> None:
    _require(len(mapping) == ITEM_COUNT, "unblinding map does not contain exactly 310 review items")
    _require(all(isinstance(record, Mapping) for record in mapping), "unblinding map contains a non-object item")
    _require(
        all(isinstance(record.get("item_id"), str) and bool(record["item_id"]) for record in mapping),
        "unblinding map item IDs must be nonempty strings",
    )
    by_id = {str(record.get("item_id", "")): record for record in mapping}
    _require(len(by_id) == ITEM_COUNT and "" not in by_id, "unblinding map IDs are missing or duplicated")
    expected_counts = {"generated": 200, "empirical_ols": 50, "real_target": 50, "generated_repeat": 10}
    counts = {kind: sum(record.get("source_type") == kind for record in mapping) for kind in expected_counts}
    _require(counts == expected_counts, "unblinding map source counts differ from the fixed review design")
    for digit in range(CLASS_COUNT):
        generated = [record for record in mapping if record.get("source_type") == "generated" and record.get("true_class") == digit]
        empirical = [record for record in mapping if record.get("source_type") == "empirical_ols" and record.get("true_class") == digit]
        targets = [record for record in mapping if record.get("source_type") == "real_target" and record.get("true_class") == digit]
        repeats = [record for record in mapping if record.get("source_type") == "generated_repeat" and record.get("true_class") == digit]
        _require(len(generated) == GENERATED_PER_CLASS and len(empirical) == EMPIRICAL_PER_CLASS and len(targets) == REAL_TARGET_PER_CLASS and len(repeats) == DUPLICATES_PER_CLASS, f"unblinding map class {digit} counts differ")
        generated_indices = [record.get("sample_index") for record in generated]
        _require(
            all(isinstance(index, int) and 0 <= index < 200 for index in generated_indices)
            and len(set(generated_indices)) == GENERATED_PER_CLASS,
            f"unblinding map class {digit} generated indices are missing, duplicated, or out of range",
        )
        _require(
            all(isinstance(record.get("source_row_index"), int) for record in targets),
            f"unblinding map class {digit} real target source rows are missing",
        )
        _require(
            len({record["source_row_index"] for record in targets}) == REAL_TARGET_PER_CLASS,
            f"unblinding map class {digit} real target source rows repeat",
        )
    empirical_indices = [record.get("sample_index") for record in mapping if record.get("source_type") == "empirical_ols"]
    _require(
        all(isinstance(index, int) and 0 <= index < 200 for index in empirical_indices)
        and len(set(empirical_indices)) == EMPIRICAL_PER_CLASS * CLASS_COUNT,
        "unblinding map empirical OLS positions are missing, duplicated, or out of range",
    )
    target_rows = [record["source_row_index"] for record in mapping if record.get("source_type") == "real_target"]
    _require(len(set(target_rows)) == REAL_TARGET_PER_CLASS * CLASS_COUNT, "unblinding map real target source rows repeat")
    primary_by_id = {str(record["item_id"]): record for record in mapping if record["source_type"] == "generated"}
    for record in mapping:
        kind = record.get("source_type")
        digit = record.get("true_class")
        _require(
            isinstance(kind, str)
            and kind in expected_counts
            and type(digit) is int
            and 0 <= digit < CLASS_COUNT,
            "unblinding map contains an invalid source or true label",
        )
        if kind in {"generated", "generated_repeat", "empirical_ols"}:
            _require(
                type(record.get("requested_class")) is int and record.get("requested_class") == digit,
                "decoded OLS item requested class differs from its saved label",
            )
        if kind in {"generated", "generated_repeat"}:
            _require(
                type(record.get("e_d_prediction")) is int and 0 <= record["e_d_prediction"] < CLASS_COUNT,
                "generated item E_D prediction is invalid",
            )
        if kind == "generated_repeat":
            source = primary_by_id.get(str(record.get("repeat_of", "")))
            _require(
                source is not None
                and source.get("true_class") == digit
                and source.get("sample_index") == record.get("sample_index")
                and source.get("e_d_prediction") == record.get("e_d_prediction"),
                "hidden repeat does not point to its matching generated item",
            )


def _score_review(mapping: Sequence[Mapping[str, Any]], answers: Mapping[str, str]) -> dict[str, Any]:
    _validate_mapping(mapping)
    _require(len(answers) == ITEM_COUNT and {str(row["item_id"]) for row in mapping} == set(answers), "rating map does not cover the blinded packet")
    allowed_answers = {str(digit) for digit in range(CLASS_COUNT)} | {"unclear"}
    _require(
        all(isinstance(answer, str) and answer in allowed_answers for answer in answers.values()),
        "rating map contains an invalid answer",
    )
    primary = [record for record in mapping if record["source_type"] == "generated"]
    anchors = [record for record in mapping if record["source_type"] == "real_target"]
    repeats = [record for record in mapping if record["source_type"] == "generated_repeat"]
    empirical = [record for record in mapping if record["source_type"] == "empirical_ols"]
    class_rows: dict[str, Any] = {}
    human_correct_by_id: dict[str, bool] = {}
    for digit in range(CLASS_COUNT):
        rows = [record for record in primary if int(record["true_class"]) == digit]
        correct = sum(answers[str(record["item_id"])] == str(digit) for record in rows)
        ed_correct = sum(int(record["e_d_prediction"]) == digit for record in rows)
        class_rows[str(digit)] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows), "wilson_95": _wilson(correct, len(rows)), "e_d_correct_same_rows": ed_correct}
        for record in rows:
            item_id = str(record["item_id"])
            human_correct_by_id[item_id] = answers[item_id] == str(digit)

    anchor_correct = sum(answers[str(record["item_id"])] == str(record["true_class"]) for record in anchors)
    repeat_agreement = sum(answers[str(record["item_id"])] == answers[str(record["repeat_of"])] for record in repeats)
    reviewer_quality = {
        "real_target_anchor_correct": anchor_correct,
        "real_target_anchor_total": len(anchors),
        "real_target_anchor_passed": anchor_correct >= ANCHOR_TOTAL_MIN,
        "duplicate_agreement": repeat_agreement,
        "duplicate_total": len(repeats),
        "duplicate_consistency_passed": repeat_agreement >= REPEAT_AGREEMENT_MIN,
    }
    reviewer_quality["passed"] = reviewer_quality["real_target_anchor_passed"] and reviewer_quality["duplicate_consistency_passed"]
    total_correct = sum(int(class_rows[str(digit)]["correct"]) for digit in range(CLASS_COUNT))
    per_class_pass = all(int(class_rows[str(digit)]["correct"]) >= PRIMARY_PER_CLASS_MIN for digit in range(CLASS_COUNT))
    main_pass = total_correct >= PRIMARY_TOTAL_MIN and per_class_pass

    human_ed_pairs = {"both_correct": 0, "human_only_correct": 0, "e_d_only_correct": 0, "both_incorrect": 0}
    for record in primary:
        item_id = str(record["item_id"])
        human_ok = human_correct_by_id[item_id]
        ed_ok = int(record["e_d_prediction"]) == int(record["true_class"])
        key = "both_correct" if human_ok and ed_ok else "human_only_correct" if human_ok else "e_d_only_correct" if ed_ok else "both_incorrect"
        human_ed_pairs[key] += 1

    def reference_summary(rows: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for digit in range(CLASS_COUNT):
            selected = [record for record in rows if int(record["true_class"]) == digit]
            result[str(digit)] = {"correct": sum(answers[str(record["item_id"])] == str(digit) for record in selected), "total": len(selected)}
        return {"kind": kind, "per_class": result, "primary_gate": False}

    qc_pass = bool(reviewer_quality["passed"])
    if not qc_pass:
        status, classification = "REVIEW_INCONCLUSIVE", "REVIEW_EVIDENCE_INSUFFICIENT"
    elif main_pass:
        status, classification = "VALID_COMPLETED", "SINGLE_REVIEWER_READABILITY_SCREEN_SUPPORTED"
    else:
        status, classification = "VALID_FAILED", "TEN_CLASS_READABILITY_SCREEN_NOT_SUPPORTED"
    return {
        "task_id": TASK_ID,
        "reviewer_count": 1,
        "reviewer_quality": reviewer_quality,
        "generated_primary": {"correct_total": total_correct, "total": len(primary), "accuracy": total_correct / len(primary), "wilson_95": _wilson(total_correct, len(primary)), "per_class": class_rows, "minimum_per_class_correct": min(int(class_rows[str(digit)]["correct"]) for digit in range(CLASS_COUNT))},
        "human_vs_e_d_same_generated_rows": human_ed_pairs,
        "empirical_fit_reference": reference_summary(empirical, "empirical_OLS_fit_outputs_5_per_class"),
        "real_target_anchor": reference_summary(anchors, "saved_canonical_TRAIN_targets_5_per_class"),
        "decision": {
            "status": status,
            "classification": classification,
            "conditions": {
                "review_quality_passed": qc_pass,
                "generated_total_at_least_150_of_200": total_correct >= PRIMARY_TOTAL_MIN,
                "each_generated_class_at_least_14_of_20": per_class_pass,
            },
            "interpretation": "single-reviewer rapid screen only; not formal human-readability acceptance or complete semantic-generation success",
            "training_recommendation": "NONE",
            "next_action": "NONE",
            "stop": True,
        },
    }


def _diversity_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    samples = data["sample_views"].flatten(1).to(torch.float32)
    requested = data["sample_requested"].to(torch.long)
    targets = data["target_views"].flatten(1).to(torch.float32)
    target_labels = data["labels"].to(torch.long)
    source_rows = data["source_rows"].to(torch.long)
    quantized = torch.round(samples.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    signatures = [bytes(row.tolist()) for row in quantized]
    class_metrics: dict[str, Any] = {}
    closest_pairs: dict[str, list[dict[str, Any]]] = {}
    for digit in range(CLASS_COUNT):
        positions = torch.nonzero(requested == digit, as_tuple=False).flatten()
        group = samples.index_select(0, positions)
        squared = group.square().sum(1, keepdim=True) + group.square().sum(1).unsqueeze(0) - 2.0 * (group @ group.T)
        squared.clamp_min_(0.0)
        squared.fill_diagonal_(float("inf"))
        nearest_sq, nearest_idx = squared.min(1)
        nearest_rmse = nearest_sq.sqrt() / math.sqrt(group.shape[1])
        class_target_positions = torch.nonzero(target_labels == digit, as_tuple=False).flatten()
        target_group = targets.index_select(0, class_target_positions)
        target_sq = group.square().sum(1, keepdim=True) + target_group.square().sum(1).unsqueeze(0) - 2.0 * (group @ target_group.T)
        target_sq.clamp_min_(0.0)
        nearest_target_sq, nearest_target_local = target_sq.min(1)
        target_rmse = nearest_target_sq.sqrt() / math.sqrt(group.shape[1])
        pairs: list[tuple[float, int, int]] = []
        for source_local, destination_local, distance in zip(range(len(positions)), nearest_idx.tolist(), nearest_rmse.tolist(), strict=True):
            left, right = sorted((source_local, int(destination_local)))
            pairs.append((float(distance), left, right))
        unique_pairs = sorted({(distance, left, right) for distance, left, right in pairs})[:5]
        closest_pairs[str(digit)] = [
            {"sample_index_a": left, "sample_index_b": right, "nearest_view_rmse": distance}
            for distance, left, right in unique_pairs
        ]
        class_signatures = [signatures[int(position)] for position in positions.tolist()]
        class_metrics[str(digit)] = {
            "sample_count": int(positions.numel()),
            "unique_8bit_views": len(set(class_signatures)),
            "exact_8bit_duplicate_excess": int(len(class_signatures) - len(set(class_signatures))),
            "mean_pixel_variance": float(group.var(0, unbiased=False).mean()),
            "nearest_generated_same_class_rmse": {"mean": float(nearest_rmse.mean()), "median": float(nearest_rmse.median()), "p10": float(torch.quantile(nearest_rmse, 0.10)), "p90": float(torch.quantile(nearest_rmse, 0.90))},
            "nearest_saved_TRAIN_target_same_class_rmse": {"mean": float(target_rmse.mean()), "median": float(target_rmse.median()), "p10": float(torch.quantile(target_rmse, 0.10)), "p90": float(torch.quantile(target_rmse, 0.90))},
            "nearest_target_source_row": [int(source_rows[int(class_target_positions[index])]) for index in nearest_target_local.tolist()],
        }
    return {
        "scope": "descriptive_only_2000_samples_and_3300_saved_canonical_TRAIN_targets",
        "per_class": class_metrics,
        "five_closest_generated_pairs_per_class": closest_pairs,
        "novelty_or_nonmemorization_claim": False,
    }


def _write_blind_packet(output_dir: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    public_items, mapping = _build_records(data)
    import secrets

    review_id = secrets.token_hex(8)
    packet = _render_html(public_items, review_id)
    template = io.StringIO(newline="")
    writer = csv.writer(template, lineterminator="\n")
    writer.writerow(["item_id", "answer"])
    writer.writerows((item["item_id"], "") for item in public_items)
    _atomic_write(output_dir / "blind_review.html", packet.encode("utf-8"))
    _atomic_write(output_dir / "ratings_template.csv", template.getvalue().encode("utf-8-sig"))
    _write_json(output_dir / "analysis" / "review_mapping.json", {"review_id": review_id, "items": mapping})
    diversity = _diversity_summary(data)
    _write_json(output_dir / "analysis" / "diversity_summary.json", diversity)
    prepared = {
        "task_id": TASK_ID,
        "state": "PREPARED_AWAITING_RATINGS",
        "source_generation_dir": str(data["generation_dir"]),
        "target_readout_path": str(data["target_readout_path"]),
        "sample_counts": {"generated_primary": 200, "empirical_OLS_reference": 50, "real_TRAIN_targets": 50, "hidden_generated_repeats": 10, "total_review_items": ITEM_COUNT},
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "reviewer_count": 1,
        "display": {"shape": [IMAGE_SIDE, IMAGE_SIDE], "range": [0, 1], "per_image_rescaling": False},
        "diversity_artifact": "analysis/diversity_summary.json",
        "rating_import": "pass the exported CSV explicitly with --ratings",
    }
    _write_json(output_dir / "prepared.json", prepared)
    return prepared


def _finalize(output_dir: Path, ratings_path: Path) -> dict[str, Any]:
    prepared = _load_json(output_dir / "prepared.json")
    _require(prepared.get("task_id") == TASK_ID and prepared.get("state") == "PREPARED_AWAITING_RATINGS", "output directory is not awaiting ratings for this task")
    mapping_payload = _load_json(output_dir / "analysis" / "review_mapping.json")
    mapping = mapping_payload.get("items")
    _require(isinstance(mapping, list) and len(mapping) == ITEM_COUNT, "saved unblinding map is incomplete")
    _require(not (output_dir / "report.json").exists(), "final report already exists; refusing to overwrite completed review")
    answers = _parse_ratings(ratings_path, mapping)
    scored = _score_review(mapping, answers)
    scored["source_generation_dir"] = prepared["source_generation_dir"]
    scored["reviewer_count"] = 1
    scored["blinded_items"] = ITEM_COUNT
    scored["selection_seed"] = prepared["selection_seed"]
    scored["diversity_artifact"] = prepared["diversity_artifact"]
    scored["diversity"] = _load_json(output_dir / prepared["diversity_artifact"])
    _write_json(output_dir / "report.json", scored)
    return scored


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", default=str(DEFAULT_GENERATION_DIR), help="completed frozen conditional OLS generation attempt")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="new readability-screen output directory")
    parser.add_argument("--ratings", help="explicit completed ratings CSV; omit to prepare the blind review package")
    parser.add_argument("--target-readout", help="optional explicit A4700 readout containing saved TRAIN targets")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    root = Path(__file__).resolve().parents[3]
    generation_dir = _resolve(root, args.generation_dir)
    output_dir = _resolve(root, args.output_dir)
    if args.ratings:
        result = _finalize(output_dir, _resolve(root, args.ratings))
        print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
        return 0

    _require(not output_dir.exists(), f"refusing to overwrite an existing output directory: {output_dir}")
    data = _load_inputs(generation_dir)
    if args.target_readout:
        explicit_readout = _resolve(root, args.target_readout)
        _require(explicit_readout == data["target_readout_path"], "explicit target readout differs from the saved source lineage")
    output_dir.mkdir(parents=True, exist_ok=False)
    prepared = _write_blind_packet(output_dir, data)
    print(json.dumps(prepared, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
