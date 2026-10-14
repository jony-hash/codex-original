"""Prepare and score a one-reviewer classifier-proxy validity audit."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import secrets
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_frozen_readability_screen_v1 as saved_evidence,
)


TASK_ID = "M0_GENERATOR_CLASSIFIER_PROXY_VALIDITY_V1"
PARENT_TASK_ID = "M0_GENERATOR_FROZEN_CONDITIONAL_OLS_GENERATION_V1"
DEFAULT_GENERATION_DIR = saved_evidence.DEFAULT_GENERATION_DIR
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20261009_m0_generator_classifier_proxy_validity_v1/runs/attempt_01"
)
SELECTION_SEED = 2026100901
ORDER_SEED = 2026100902
GENERATED_COUNT = 200
GENERATED_CLASS_COUNTS = [19, 25, 19, 23, 27, 24, 14, 12, 18, 19]
EMPIRICAL_PER_CLASS = 5
ANCHOR_PER_CLASS = 5
REPEAT_PER_CLASS = 3
BLANK_COUNT = 10
POPULATION_POSITIVES = 1_513
ITEM_COUNT = 340
BATCH_SIZE = 85
PROXY_MISMATCH_LIMIT = 0.10
CONFIDENCE_LEVEL = 0.95
CLASS_COUNT = 10
IMAGE_SIDE = 28
MIN_REPEAT_SPACING = 50


class ProxyValidityError(RuntimeError):
    """Invalid saved inputs, review mapping, or rating evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProxyValidityError(message)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _atomic_write(path: Path, payload: bytes) -> None:
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pending_", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
        os.rename(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        ),
    )


def _load_inputs(generation_dir: Path) -> dict[str, Any]:
    data = saved_evidence._load_inputs(generation_dir)
    metrics = saved_evidence._load_torch(generation_dir / "sample_row_metrics.pt")
    predictions = metrics.get("e_d_predictions_empirical_fit_ols")
    positions = metrics.get("e_d_positions_empirical_fit_ols")
    _require(
        isinstance(predictions, torch.Tensor)
        and tuple(predictions.shape) == (200,)
        and isinstance(positions, torch.Tensor)
        and torch.equal(positions.to(torch.long), torch.arange(200)),
        "saved empirical E_D predictions or positions are incomplete",
    )
    predictions = predictions.to(torch.long)
    _require(
        bool(((predictions >= 0) & (predictions < CLASS_COUNT)).all()),
        "saved empirical E_D predictions are outside 0..9",
    )
    positive_count = int(
        (data["sample_e_d_prediction"] == data["sample_requested"]).sum()
    )
    _require(
        positive_count == POPULATION_POSITIVES,
        "saved generated E_D-positive population differs from the fixed report count",
    )
    report = data["report"]
    expected_per_class = (
        report.get("results", {})
        .get("primary_e_d_sample_screen", {})
        .get("per_class_correct")
    )
    _require(
        isinstance(expected_per_class, list)
        and len(expected_per_class) == CLASS_COUNT
        and sum(int(value) for value in expected_per_class) == POPULATION_POSITIVES,
        "source report does not support the fixed E_D-positive population size",
    )
    data["empirical_e_d_prediction"] = predictions
    data["population_positive_count"] = positive_count
    return data


def _balanced_positions(
    labels: torch.Tensor, candidates: Sequence[int], per_class: int, rng: random.Random
) -> list[int]:
    selected: list[int] = []
    for digit in range(CLASS_COUNT):
        positions = [
            position for position in candidates if int(labels[position]) == digit
        ]
        _require(
            len(positions) >= per_class,
            f"class {digit} has too few rows for the fixed blind sample",
        )
        selected.extend(rng.sample(positions, per_class))
    return selected


def _insert_repeats_with_spacing(
    records: list[dict[str, Any]], repeats: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    ordered = list(records)
    rng.shuffle(ordered)
    for repeat in repeats:
        original_id = str(repeat["repeat_of"])
        original_position = next(
            index
            for index, record in enumerate(ordered)
            if record["item_id"] == original_id
        )
        candidates = [
            position
            for position in range(len(ordered) + 1)
            if abs(position - (original_position + (position <= original_position)))
            >= MIN_REPEAT_SPACING
        ]
        _require(
            bool(candidates), "could not place a hidden repeat at the required distance"
        )
        insertion = rng.choice(candidates)
        ordered.insert(insertion, repeat)
    return ordered


def _validate_mapping(mapping: Sequence[Mapping[str, Any]]) -> None:
    _require(len(mapping) == ITEM_COUNT, "blind map must contain exactly 340 items")
    _require(
        all(isinstance(row, Mapping) for row in mapping),
        "blind map contains a non-object",
    )
    identifiers = [row.get("item_id") for row in mapping]
    _require(
        all(isinstance(item_id, str) and item_id for item_id in identifiers)
        and len(set(identifiers)) == ITEM_COUNT,
        "blind map IDs are missing or duplicated",
    )
    by_id = {str(row["item_id"]): row for row in mapping}
    expected_counts = {
        "generated": GENERATED_COUNT,
        "empirical_ols": EMPIRICAL_PER_CLASS * CLASS_COUNT,
        "real_target": ANCHOR_PER_CLASS * CLASS_COUNT,
        "generated_repeat": REPEAT_PER_CLASS * CLASS_COUNT,
        "blank": BLANK_COUNT,
    }
    _require(
        {
            kind: sum(row.get("source_type") == kind for row in mapping)
            for kind in expected_counts
        }
        == expected_counts,
        "blind map source counts differ from the fixed review design",
    )
    generated_by_class = [
        sum(
            row.get("source_type") == "generated"
            and row.get("requested_class") == digit
            for row in mapping
        )
        for digit in range(CLASS_COUNT)
    ]
    _require(
        generated_by_class == GENERATED_CLASS_COUNTS,
        "blind map class counts differ from the fixed random sample",
    )
    empirical_by_class = [
        sum(
            row.get("source_type") == "empirical_ols"
            and row.get("requested_class") == digit
            for row in mapping
        )
        for digit in range(CLASS_COUNT)
    ]
    anchors_by_class = [
        sum(
            row.get("source_type") == "real_target" and row.get("true_class") == digit
            for row in mapping
        )
        for digit in range(CLASS_COUNT)
    ]
    repeats_by_class = [
        sum(
            row.get("source_type") == "generated_repeat"
            and row.get("requested_class") == digit
            for row in mapping
        )
        for digit in range(CLASS_COUNT)
    ]
    _require(
        empirical_by_class == [EMPIRICAL_PER_CLASS] * CLASS_COUNT
        and anchors_by_class == [ANCHOR_PER_CLASS] * CLASS_COUNT
        and repeats_by_class == [REPEAT_PER_CLASS] * CLASS_COUNT,
        "reference, positive-control, or repeat class counts differ from the fixed design",
    )
    for digit in range(CLASS_COUNT):
        generated_indices = [
            row.get("sample_index")
            for row in mapping
            if row.get("source_type") == "generated"
            and row.get("requested_class") == digit
        ]
        empirical_indices = [
            row.get("sample_index")
            for row in mapping
            if row.get("source_type") == "empirical_ols"
            and row.get("requested_class") == digit
        ]
        anchor_rows = [
            row.get("source_row_index")
            for row in mapping
            if row.get("source_type") == "real_target"
            and row.get("true_class") == digit
        ]
        _require(
            len(set(generated_indices)) == len(generated_indices)
            and len(set(empirical_indices)) == EMPIRICAL_PER_CLASS
            and len(set(anchor_rows)) == ANCHOR_PER_CLASS,
            f"class {digit} contains a duplicated sampled row",
        )
    for row in mapping:
        kind = row.get("source_type")
        if kind in {"generated", "generated_repeat", "empirical_ols"}:
            _require(
                isinstance(row.get("requested_class"), int)
                and 0 <= row["requested_class"] < CLASS_COUNT,
                "generated or empirical item is missing its requested class",
            )
            _require(
                isinstance(row.get("e_d_prediction"), int)
                and 0 <= row["e_d_prediction"] < CLASS_COUNT,
                "generated or empirical item is missing its saved E_D prediction",
            )
        if kind in {"generated", "generated_repeat"}:
            _require(
                isinstance(row.get("sample_index"), int)
                and 0 <= row["sample_index"] < 200,
                "generated item has an invalid within-class sample index",
            )
        if kind == "empirical_ols":
            _require(
                isinstance(row.get("sample_index"), int)
                and 0 <= row["sample_index"] < 200
                and isinstance(row.get("source_row_index"), int),
                "empirical item has an invalid saved sample position or source row",
            )
        if kind in {"real_target", "generated_repeat"}:
            _require(
                isinstance(row.get("true_class"), int)
                and 0 <= row["true_class"] < CLASS_COUNT,
                "positive-control or repeat item is missing its reference class",
            )
        if kind == "real_target":
            _require(
                isinstance(row.get("source_row_index"), int),
                "real target item is missing its saved source row",
            )
        if kind == "generated_repeat":
            source = by_id.get(str(row.get("repeat_of")))
            _require(
                source is not None
                and source.get("source_type") == "generated"
                and source.get("requested_class") == row.get("requested_class")
                and source.get("sample_index") == row.get("sample_index")
                and source.get("e_d_prediction") == row.get("e_d_prediction"),
                "hidden repeat does not match its selected generated item",
            )
    repeat_pairs = [
        (str(row["repeat_of"]), str(row["item_id"]))
        for row in mapping
        if row.get("source_type") == "generated_repeat"
    ]
    positions = {str(row["item_id"]): index for index, row in enumerate(mapping)}
    _require(
        all(
            abs(positions[original] - positions[repeat]) >= MIN_REPEAT_SPACING
            for original, repeat in repeat_pairs
        ),
        "hidden repeats are closer than 50 items to their originals",
    )


def _build_records(
    data: Mapping[str, Any], id_factory: Any = None
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if id_factory is None:

        def id_factory() -> str:
            return secrets.token_hex(6)

    selection_rng = random.Random(SELECTION_SEED)
    sample_positions = selection_rng.sample(range(2_000), GENERATED_COUNT)
    sampled_class_counts = [
        sum(
            int(data["sample_requested"][position]) == digit
            for position in sample_positions
        )
        for digit in range(CLASS_COUNT)
    ]
    _require(
        sampled_class_counts == GENERATED_CLASS_COUNTS,
        "fixed selection seed no longer produces the declared class coverage",
    )

    records: list[dict[str, Any]] = []
    generated_by_class: dict[int, list[dict[str, Any]]] = {
        digit: [] for digit in range(CLASS_COUNT)
    }

    def add(
        view: torch.Tensor,
        *,
        source_type: str,
        requested_class: int | None = None,
        true_class: int | None = None,
        e_d_prediction: int | None = None,
        sample_index: int | None = None,
        source_row_index: int | None = None,
        repeat_of: str | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "item_id": str(id_factory()),
            "source_type": source_type,
            "_image_data": saved_evidence._image_data_uri(view),
        }
        for name, value in (
            ("requested_class", requested_class),
            ("true_class", true_class),
            ("e_d_prediction", e_d_prediction),
            ("sample_index", sample_index),
            ("source_row_index", source_row_index),
            ("repeat_of", repeat_of),
        ):
            if value is not None:
                row[name] = value
        records.append(row)
        return row

    for position in sample_positions:
        requested = int(data["sample_requested"][position])
        index = int(data["sample_index"][position])
        row = add(
            data["sample_views"][position, 0],
            source_type="generated",
            requested_class=requested,
            e_d_prediction=int(data["sample_e_d_prediction"][position]),
            sample_index=index,
        )
        generated_by_class[requested].append(row)

    empirical_positions = _balanced_positions(
        data["empirical_labels"], list(range(200)), EMPIRICAL_PER_CLASS, selection_rng
    )
    for position in empirical_positions:
        digit = int(data["empirical_labels"][position])
        add(
            data["empirical_views"][position, 0],
            source_type="empirical_ols",
            requested_class=digit,
            e_d_prediction=int(data["empirical_e_d_prediction"][position]),
            sample_index=position,
            source_row_index=int(data["empirical_rows"][position]),
        )

    target_candidates = list(range(3_000, 3_300))
    target_positions = _balanced_positions(
        data["labels"], target_candidates, ANCHOR_PER_CLASS, selection_rng
    )
    for position in target_positions:
        digit = int(data["labels"][position])
        add(
            data["target_views"][position, 0],
            source_type="real_target",
            true_class=digit,
            source_row_index=int(data["source_rows"][position]),
        )

    repeats: list[dict[str, Any]] = []
    for digit in range(CLASS_COUNT):
        for source in selection_rng.sample(generated_by_class[digit], REPEAT_PER_CLASS):
            repeats.append(
                add(
                    data["sample_views"][digit * 200 + int(source["sample_index"]), 0],
                    source_type="generated_repeat",
                    requested_class=digit,
                    true_class=digit,
                    e_d_prediction=int(source["e_d_prediction"]),
                    sample_index=int(source["sample_index"]),
                    repeat_of=str(source["item_id"]),
                )
            )

    blank = torch.zeros((1, IMAGE_SIDE, IMAGE_SIDE), dtype=torch.float32)
    for _ in range(BLANK_COUNT):
        add(blank, source_type="blank")

    order_rng = random.Random(ORDER_SEED)
    ordered = _insert_repeats_with_spacing(
        [row for row in records if row["source_type"] != "generated_repeat"],
        repeats,
        order_rng,
    )
    mapping = [
        {key: value for key, value in row.items() if key != "_image_data"}
        for row in ordered
    ]
    public_items = [
        {"item_id": str(row["item_id"]), "image": str(row["_image_data"])}
        for row in ordered
    ]
    _validate_mapping(mapping)
    _require(
        len(public_items) == ITEM_COUNT, "blind packet item count differs from 340"
    )
    return public_items, mapping


def _render_html(public_items: Sequence[Mapping[str, str]], review_id: str) -> str:
    payload = json.dumps(list(public_items), ensure_ascii=False, separators=(",", ":"))
    review_token = json.dumps(review_id)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LightGen 数字盲判</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:20px;background:#f4f4f4;color:#222}}
header{{position:sticky;top:0;background:#fff;padding:12px;z-index:2;border:1px solid #ccc}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px;margin-top:16px}}
.card{{background:#fff;border:1px solid #ddd;padding:10px;text-align:center}}
.card img{{width:224px;height:224px;image-rendering:pixelated;background:#fff;object-fit:contain}}
.id{{font-family:monospace;color:#555;margin:5px}}
select,button{{font-size:16px;padding:7px;margin:4px}}button{{cursor:pointer}}
@media(max-width:850px){{.grid{{grid-template-columns:repeat(2,minmax(140px,1fr))}}.card img{{width:140px;height:140px}}}}
</style></head><body>
<header><strong>LightGen 数字盲判</strong>
<p>只看图像本身，选择你读到的数字；无法辨认时选“无法辨认”。图像使用统一灰度范围，不做单图增强。每组 85 题，可分四次完成。请不要查看带标签的文件或中途查看结果。</p>
<label>恢复进度：<input id="import" type="file" accept=".csv,text/csv"></label>
<div><button id="prev">上一组</button><span id="page"></span><button id="next">下一组</button>
<button id="exportProgress">导出进度 CSV</button><button id="exportFinal">导出完整评分 CSV</button></div>
<span id="progress"></span><p id="message">可随时导出进度；完成全部 340 题后导出完整评分。</p></header>
<main id="grid" class="grid"></main>
<script>
const items={payload}; const reviewId={review_token}; const pageSize={BATCH_SIZE}; const answers={{}};
const allowed=new Set([...Array(10).keys()].map(String).concat(["unclear"]));
const storageKey='lightgen-proxy-validity-'+reviewId;
try{{const saved=localStorage.getItem(storageKey);if(saved)Object.assign(answers,JSON.parse(saved));}}catch{{}}
const grid=document.getElementById('grid');let page=0;
function persist(){{try{{localStorage.setItem(storageKey,JSON.stringify(answers));}}catch{{}}updateProgress();}}
function render(){{grid.innerHTML='';const start=page*pageSize;const visible=items.slice(start,start+pageSize);
for(const item of visible){{const card=document.createElement('section');card.className='card';
const image=document.createElement('img');image.src=item.image;image.alt='';image.draggable=false;
const id=document.createElement('div');id.className='id';id.textContent='编号 '+item.item_id;
const select=document.createElement('select');select.setAttribute('aria-label','编号 '+item.item_id+' 的读数');
const blank=document.createElement('option');blank.value='';blank.textContent='请选择';select.appendChild(blank);
for(let digit=0;digit<10;digit++){{const option=document.createElement('option');option.value=String(digit);option.textContent=String(digit);select.appendChild(option);}}
const unclear=document.createElement('option');unclear.value='unclear';unclear.textContent='无法辨认';select.appendChild(unclear);
select.value=answers[item.item_id]||'';select.onchange=()=>{{if(select.value)answers[item.item_id]=select.value;else delete answers[item.item_id];persist();}};
card.append(image,id,select);grid.appendChild(card);}}
const first=start+1;const last=Math.min(start+pageSize,items.length);
document.getElementById('page').textContent=`第 ${{page+1}} / ${{Math.ceil(items.length/pageSize)}} 组（${{first}}–${{last}}）`;
document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=last>=items.length;updateProgress();}}
function updateProgress(){{const count=Object.keys(answers).length;document.getElementById('progress').textContent=`已完成 ${{count}} / ${{items.length}} 项`;document.getElementById('exportFinal').disabled=count!==items.length;}}
function downloadRatings(requireComplete){{if(requireComplete&&Object.keys(answers).length!==items.length)return;
const lines=['item_id,answer',...items.map(item=>`${{item.item_id}},${{answers[item.item_id]||''}}`)];
const blob=new Blob(['\\ufeff'+lines.join('\\r\\n')],{{type:'text/csv;charset=utf-8'}});const link=document.createElement('a');
link.href=URL.createObjectURL(blob);link.download=`lightgen_proxy_validity_${{requireComplete?'ratings':'progress'}}_${{reviewId}}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);}}
document.getElementById('prev').onclick=()=>{{page--;render();}};document.getElementById('next').onclick=()=>{{page++;render();}};
document.getElementById('exportProgress').onclick=()=>downloadRatings(false);
document.getElementById('exportFinal').onclick=()=>downloadRatings(true);
document.getElementById('import').onchange=event=>{{const file=event.target.files[0];if(!file)return;const reader=new FileReader();reader.onload=()=>{{
try{{const lines=String(reader.result).replace(/^\\uFEFF/,'').trim().split(/\\r?\\n/);if(lines.shift()!=='item_id,answer')throw Error('评分文件表头不正确');
const known=new Set(items.map(item=>item.item_id));const incoming={{}};for(const line of lines){{const fields=line.split(',');if(fields.length!==2)throw Error('评分文件行格式不正确');
const [id,answer]=fields;if(!known.has(id)||Object.hasOwn(incoming,id))throw Error('评分编号未知或重复');if(answer&&!allowed.has(answer))throw Error('评分值无效');if(answer)incoming[id]=answer;}}
Object.assign(answers,incoming);persist();render();document.getElementById('message').textContent='进度已载入。';
}}catch(error){{document.getElementById('message').textContent='载入失败：'+error.message;}}}};reader.readAsText(file);}};
render();
</script></body></html>"""


def _write_blind_packet(output_dir: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    public_items, mapping = _build_records(data)
    review_id = secrets.token_hex(8)
    page = _render_html(public_items, review_id)
    template_lines = ["item_id,answer"]
    template_lines.extend(str(item["item_id"]) + "," for item in public_items)
    _atomic_write(output_dir / "blind_review.html", page.encode("utf-8"))
    _atomic_write(
        output_dir / "ratings_template.csv",
        ("\ufeff" + "\n".join(template_lines) + "\n").encode("utf-8"),
    )
    _write_json(
        output_dir / "analysis" / "review_mapping.json",
        {"review_id": review_id, "items": mapping},
    )
    instructions = (
        "# 实验一：盲判说明\n\n"
        "打开 `blind_review.html`，只根据图像选择 0–9 或“无法辨认”。每组 85 题，共四组。\n\n"
        "页面会自动保存进度；也可随时导出进度 CSV，之后用页面顶部的“恢复进度”载入。"
        "完成全部 340 题后，点击“导出完整评分 CSV”。请保留原始评分文件。\n\n"
        "评分期间不要打开 `analysis/review_mapping.json`，也不要查看中间统计。"
        "完成后在仓库根目录运行：\n\n"
        "```powershell\n"
        "python -B -m scripts.research.lightgen.audit_lightgen_2d_generator_classifier_proxy_validity_v1 "
        "--output-dir outputs/20261009_m0_generator_classifier_proxy_validity_v1/runs/attempt_01 "
        '--ratings "<导出的完整评分 CSV 路径>"\n'
        "```\n"
    )
    _atomic_write(output_dir / "review_instructions.md", instructions.encode("utf-8"))
    prepared = {
        "task_id": TASK_ID,
        "state": "PREPARED_AWAITING_RATINGS",
        "source_task_id": PARENT_TASK_ID,
        "source_generation_dir": str(data["generation_dir"]),
        "target_readout_path": str(data["target_readout_path"]),
        "population_positive_count": int(data["population_positive_count"]),
        "sample_counts": {
            "generated_primary": GENERATED_COUNT,
            "empirical_OLS_reference": EMPIRICAL_PER_CLASS * CLASS_COUNT,
            "real_TRAIN_positive_control": ANCHOR_PER_CLASS * CLASS_COUNT,
            "hidden_generated_repeats": REPEAT_PER_CLASS * CLASS_COUNT,
            "all_zero_negative_controls": BLANK_COUNT,
            "total_review_items": ITEM_COUNT,
        },
        "generated_primary_class_counts": GENERATED_CLASS_COUNTS,
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "reviewer_count": 1,
        "batch_size": BATCH_SIZE,
        "minimum_repeat_spacing": MIN_REPEAT_SPACING,
        "display": {
            "shape": [IMAGE_SIDE, IMAGE_SIDE],
            "range": [0, 1],
            "per_image_rescaling": False,
        },
        "rating_answers": [str(digit) for digit in range(CLASS_COUNT)] + ["unclear"],
        "analysis_map": "analysis/review_mapping.json",
        "rating_import": "Run this script with --ratings and the exported complete CSV.",
        "stop_after_this_review": True,
    }
    _write_json(output_dir / "prepared.json", prepared)
    return prepared


def _hypergeom_cdf(
    x: int, population_size: int, positive_count: int, draws: int
) -> float:
    _require(
        population_size > 0
        and 0 <= positive_count <= population_size
        and 0 <= draws <= population_size,
        "hypergeometric interval inputs are out of range",
    )
    lower = max(0, draws - (population_size - positive_count))
    upper = min(x, positive_count, draws)
    if upper < lower:
        return 0.0
    numerator = sum(
        math.comb(positive_count, value)
        * math.comb(population_size - positive_count, draws - value)
        for value in range(lower, upper + 1)
    )
    return numerator / math.comb(population_size, draws)


def _hypergeom_interval(
    observed_mismatches: int,
    draws: int,
    positive_population: int = POPULATION_POSITIVES,
) -> list[float]:
    _require(
        draws > 0 and 0 <= observed_mismatches <= draws,
        "hypergeometric interval requires at least one proxy-positive sample",
    )
    alpha_tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    population_size = positive_population
    lower_support = observed_mismatches
    upper_support = population_size - draws + observed_mismatches
    _require(
        draws <= population_size and upper_support >= lower_support,
        "sample contains more proxy-positive rows than the fixed population",
    )

    low, high = lower_support, upper_support
    while low < high:
        middle = (low + high) // 2
        upper_tail = 1.0 - _hypergeom_cdf(
            observed_mismatches - 1, population_size, middle, draws
        )
        if upper_tail >= alpha_tail:
            high = middle
        else:
            low = middle + 1
    lower_count = low

    low, high = lower_support, upper_support
    while low < high:
        middle = (low + high + 1) // 2
        lower_tail = _hypergeom_cdf(observed_mismatches, population_size, middle, draws)
        if lower_tail >= alpha_tail:
            low = middle
        else:
            high = middle - 1
    upper_count = low
    return [lower_count / population_size, upper_count / population_size]


def _parse_ratings(path: Path, mapping: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    _require(path.is_file(), f"explicit ratings CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(
            reader.fieldnames == ["item_id", "answer"],
            "ratings CSV header must be item_id,answer",
        )
        rows = list(reader)
    expected = {str(row["item_id"]) for row in mapping}
    answers: dict[str, str] = {}
    allowed = {str(digit) for digit in range(CLASS_COUNT)} | {"unclear"}
    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        answer = str(row.get("answer", "")).strip().lower()
        _require(
            item_id in expected, f"ratings CSV contains an unknown item ID: {item_id}"
        )
        _require(item_id not in answers, f"ratings CSV repeats an item ID: {item_id}")
        _require(answer in allowed, f"invalid rating for item {item_id}")
        answers[item_id] = answer
    _require(set(answers) == expected, "ratings CSV is missing one or more item IDs")
    return answers


def _score_review(
    mapping: Sequence[Mapping[str, Any]], answers: Mapping[str, str]
) -> dict[str, Any]:
    _validate_mapping(mapping)
    expected_ids = {str(row["item_id"]) for row in mapping}
    _require(
        set(answers) == expected_ids, "ratings do not cover the blind packet exactly"
    )
    allowed = {str(digit) for digit in range(CLASS_COUNT)} | {"unclear"}
    _require(
        all(answer in allowed for answer in answers.values()),
        "ratings contain an invalid answer",
    )

    generated = [row for row in mapping if row["source_type"] == "generated"]
    proxy_positive = [
        row for row in generated if row["e_d_prediction"] == row["requested_class"]
    ]
    proxy_positive_mismatches = sum(
        answers[str(row["item_id"])] != str(row["requested_class"])
        for row in proxy_positive
    )
    primary_interval = (
        _hypergeom_interval(proxy_positive_mismatches, len(proxy_positive))
        if proxy_positive
        else [0.0, 1.0]
    )

    anchors = [row for row in mapping if row["source_type"] == "real_target"]
    anchor_correct = sum(
        answers[str(row["item_id"])] == str(row["true_class"]) for row in anchors
    )
    repeats = [row for row in mapping if row["source_type"] == "generated_repeat"]
    repeat_agreement = sum(
        answers[str(row["item_id"])] == answers[str(row["repeat_of"])]
        for row in repeats
    )
    blanks = [row for row in mapping if row["source_type"] == "blank"]
    blank_correct = sum(answers[str(row["item_id"])] == "unclear" for row in blanks)
    quality = {
        "real_target_correct": anchor_correct,
        "real_target_total": len(anchors),
        "real_target_passed": anchor_correct >= 45,
        "repeat_answers_agree": repeat_agreement,
        "repeat_total": len(repeats),
        "repeat_consistency_passed": repeat_agreement >= 27,
        "all_zero_answered_unclear": blank_correct,
        "all_zero_total": len(blanks),
        "all_zero_passed": blank_correct >= 9,
    }
    quality["passed"] = all(
        quality[key]
        for key in (
            "real_target_passed",
            "repeat_consistency_passed",
            "all_zero_passed",
        )
    )

    def summarize(
        rows: Sequence[Mapping[str, Any]], expected_key: str
    ) -> dict[str, Any]:
        total = len(rows)
        human_matches = sum(
            answers[str(row["item_id"])] == str(row[expected_key]) for row in rows
        )
        ed_matches = sum(
            row.get("e_d_prediction") == row.get(expected_key)
            for row in rows
            if isinstance(row.get("e_d_prediction"), int)
        )
        paired = {
            "both_match_requested": 0,
            "human_only_matches_requested": 0,
            "e_d_only_matches_requested": 0,
            "neither_matches_requested": 0,
        }
        confusion = [[0 for _ in range(CLASS_COUNT + 1)] for _ in range(CLASS_COUNT)]
        for row in rows:
            requested = int(row[expected_key])
            human = answers[str(row["item_id"])]
            confusion[requested][CLASS_COUNT if human == "unclear" else int(human)] += 1
            human_ok = human == str(requested)
            ed_ok = row.get("e_d_prediction") == requested
            label = (
                "both_match_requested"
                if human_ok and ed_ok
                else "human_only_matches_requested"
                if human_ok
                else "e_d_only_matches_requested"
                if ed_ok
                else "neither_matches_requested"
            )
            paired[label] += 1
        by_class: dict[str, Any] = {}
        for digit in range(CLASS_COUNT):
            class_rows = [row for row in rows if int(row[expected_key]) == digit]
            by_class[str(digit)] = {
                "total": len(class_rows),
                "human_matches_requested": sum(
                    answers[str(row["item_id"])] == str(digit) for row in class_rows
                ),
                "e_d_matches_requested": sum(
                    row.get("e_d_prediction") == digit
                    for row in class_rows
                    if isinstance(row.get("e_d_prediction"), int)
                ),
                "human_answers": {
                    str(answer): sum(
                        answers[str(row["item_id"])] == str(answer)
                        for row in class_rows
                    )
                    for answer in [str(value) for value in range(CLASS_COUNT)]
                    + ["unclear"]
                },
            }
        return {
            "total": total,
            "human_matches_requested": human_matches,
            "human_match_rate": human_matches / total if total else None,
            "e_d_matches_requested": ed_matches,
            "e_d_match_rate": ed_matches / total if total else None,
            "direct_human_e_d_answer_agreement": sum(
                answers[str(row["item_id"])] == str(row.get("e_d_prediction"))
                for row in rows
                if isinstance(row.get("e_d_prediction"), int)
            ),
            "paired_counts": paired,
            "confusion_requested_rows_human_answer_columns_0_to_9_unclear": confusion,
            "per_class": by_class,
        }

    primary = summarize(generated, "requested_class")
    positive_summary = {
        "e_d_positive_count_in_sample": len(proxy_positive),
        "human_mismatch_count": proxy_positive_mismatches,
        "mismatch_rate": proxy_positive_mismatches / len(proxy_positive)
        if proxy_positive
        else None,
        "exact_hypergeometric_95_percent_interval": primary_interval,
        "finite_population_size": POPULATION_POSITIVES,
        "confidence_interval_method": "equal-tailed exact hypergeometric inversion, without replacement",
        "tolerance": PROXY_MISMATCH_LIMIT,
    }
    references = {
        "empirical_ols": summarize(
            [row for row in mapping if row["source_type"] == "empirical_ols"],
            "requested_class",
        ),
        "real_target_positive_control": {
            "human_matches_reference": anchor_correct,
            "total": len(anchors),
            "answers": {
                str(answer): sum(
                    answers[str(row["item_id"])] == str(answer) for row in anchors
                )
                for answer in [str(value) for value in range(CLASS_COUNT)] + ["unclear"]
            },
        },
        "all_zero_negative_control": {
            "answered_unclear": blank_correct,
            "total": len(blanks),
            "answers": {
                str(answer): sum(
                    answers[str(row["item_id"])] == str(answer) for row in blanks
                )
                for answer in [str(value) for value in range(CLASS_COUNT)] + ["unclear"]
            },
        },
    }

    if not quality["passed"]:
        status = "REVIEW_INCONCLUSIVE"
        classification = "REVIEW_QUALITY_CHECK_FAILED"
    elif primary_interval[1] <= PROXY_MISMATCH_LIMIT:
        status = "VALID_COMPLETED"
        classification = "CLASSIFIER_PROXY_LOCALLY_SUPPORTED"
    elif primary_interval[0] > PROXY_MISMATCH_LIMIT:
        status = "VALID_FAILED"
        classification = "CLASSIFIER_PROXY_MISMATCH_SUPPORTED"
    else:
        status = "VALID_INCONCLUSIVE"
        classification = "CLASSIFIER_PROXY_EVIDENCE_INSUFFICIENT"
    return {
        "task_id": TASK_ID,
        "source_task_id": PARENT_TASK_ID,
        "reviewer_count": 1,
        "review_quality": quality,
        "generated_primary": primary,
        "proxy_positive_validity": positive_summary,
        "reference_groups": references,
        "decision": {
            "status": status,
            "classification": classification,
            "interpretation": "one-reviewer validity of E_D as positive evidence that an A4700 frozen output matches its requested class; not population-level human validity or complete semantic-generation acceptance",
            "training_recommendation": "NONE",
            "next_action": "NONE",
            "stop": True,
        },
    }


def _paired_rows_csv(
    mapping: Sequence[Mapping[str, Any]], answers: Mapping[str, str]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "item_id",
            "sample_population_position",
            "requested_class",
            "e_d_prediction",
            "human_answer",
            "e_d_matches_requested",
            "human_matches_requested",
            "human_e_d_answer_agrees",
        ]
    )
    for row in mapping:
        if row["source_type"] != "generated":
            continue
        item_id = str(row["item_id"])
        requested = int(row["requested_class"])
        e_d = int(row["e_d_prediction"])
        human = answers[item_id]
        writer.writerow(
            [
                item_id,
                requested * 200 + int(row["sample_index"]),
                requested,
                e_d,
                human,
                int(e_d == requested),
                int(human == str(requested)),
                int(human == str(e_d)),
            ]
        )
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def _render_closeout_markdown(result: Mapping[str, Any]) -> str:
    decision = result["decision"]
    validity = result["proxy_positive_validity"]
    quality = result["review_quality"]
    interval = validity["exact_hypergeometric_95_percent_interval"]
    classification = decision["classification"]
    if classification == "CLASSIFIER_PROXY_LOCALLY_SUPPORTED":
        learning = (
            "在本次 A4700 输出总体、当前单人盲判和 10% 失配上限下，"
            "E_D 判为请求数字的样本得到了局部支持。"
        )
        advice = "建议仅把 E_D 保留为该冻结输出总体的辅助肯定证据。"
    elif classification == "CLASSIFIER_PROXY_MISMATCH_SUPPORTED":
        learning = (
            "当前冻结输出中，E_D 判为请求数字的图像仍有明确比例未获人工认可；"
            "分类预测不能独立代表可识读性。"
        )
        advice = "建议暂停用 E_D 单独支持当前输出的可识读性主张。"
    elif classification == "REVIEW_QUALITY_CHECK_FAILED":
        learning = (
            "审阅质控未通过，本轮效度结论证据不足；失配率及区间仅作描述性结果。"
        )
        advice = "按审阅质控未通过收尾，不据此认定代理失效或改变科学判断。"
    else:
        learning = "本轮区间跨越预设 10% 容忍线，证据不足以支持或否定代理效度。"
        advice = "建议保留未决状态，不据此改变科学判断。"

    mismatch_rate = validity["mismatch_rate"]
    rate_text = "未估计" if mismatch_rate is None else f"{mismatch_rate:.1%}"
    mismatch_count = validity["human_mismatch_count"]
    positive_count = validity["e_d_positive_count_in_sample"]
    interval_text = f"[{interval[0]:.1%}, {interval[1]:.1%}]"
    quality_text = (
        f"真实数字 {quality['real_target_correct']}/{quality['real_target_total']}，"
        f"重复一致 {quality['repeat_answers_agree']}/{quality['repeat_total']}，"
        f"全零图选‘无法辨认’ {quality['all_zero_answered_unclear']}/{quality['all_zero_total']}"
    )
    return (
        "# 实验一：分类代理在当前输出分布上的效度\n\n"
        f"结果与状态：{decision['status']}；{classification}。\n\n"
        f"科学认识：{learning} 主指标失配率 q={rate_text} "
        f"（{mismatch_count}/{positive_count}），95% 有限总体区间 "
        f"{interval_text}。结论只适用于 A4700 保存输出和本次单人审阅。\n\n"
        f"目标缺口：隐藏重复答案一致为 {quality['repeat_answers_agree']}/"
        f"{quality['repeat_total']}，低于预设 27/30；因此单人效度主张未被建立。"
        "本结果不验证其他审阅者、后续 checkpoint、完整语义生成或物理部署。\n\n"
        f"工程完成与验证：完成 340 项盲判，主样本 200 张，其中 E_D 判为请求数字的 "
        f"{validity['e_d_positive_count_in_sample']} 张；审阅质控为"
        f"{('通过' if quality['passed'] else '未通过')}（{quality_text}）。"
        "样本来自已保存 A4700 证据，未训练或重跑生成。\n\n"
        f"决策影响与下一步：{advice} 本轮授权止于这一次盲判和计分；本报告不授权训练、"
        "采用模型或启动后续实验。\n"
    )


def _finalize(output_dir: Path, ratings_path: Path) -> dict[str, Any]:
    prepared = saved_evidence._load_json(output_dir / "prepared.json")
    _require(
        prepared.get("task_id") == TASK_ID
        and prepared.get("state") == "PREPARED_AWAITING_RATINGS",
        "output directory is not awaiting this review's ratings",
    )
    mapping_payload = saved_evidence._load_json(
        output_dir / "analysis" / "review_mapping.json"
    )
    mapping = mapping_payload.get("items")
    _require(isinstance(mapping, list), "saved blind mapping is incomplete")
    _require(
        not (output_dir / "report.json").exists(),
        "final report already exists; refusing to overwrite completed review",
    )
    answers = _parse_ratings(ratings_path, mapping)
    result = _score_review(mapping, answers)
    raw_path = output_dir / "analysis" / "raw_answers.csv"
    _require(
        not raw_path.exists(), "raw answer archive already exists; refusing overwrite"
    )
    result.update(
        {
            "source_generation_dir": prepared["source_generation_dir"],
            "population_positive_count": prepared["population_positive_count"],
            "blinded_items": ITEM_COUNT,
            "sample_counts": prepared["sample_counts"],
            "selection_seed": SELECTION_SEED,
            "order_seed": ORDER_SEED,
            "raw_answers_artifact": "analysis/raw_answers.csv",
            "paired_rows_artifact": "analysis/paired_rows.csv",
        }
    )
    paired_rows = _paired_rows_csv(mapping, answers)
    closeout = _render_closeout_markdown(result).encode("utf-8")
    _atomic_write(raw_path, ratings_path.read_bytes())
    _atomic_write(output_dir / "analysis" / "paired_rows.csv", paired_rows)
    _write_json(output_dir / "report.json", result)
    _atomic_write(output_dir / "report.md", closeout)
    final_state = dict(prepared)
    final_state["state"] = result["decision"]["status"]
    final_state["report_artifact"] = "report.json"
    _write_json(output_dir / "completed.json", final_state)
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dir",
        default=str(DEFAULT_GENERATION_DIR),
        help="explicit saved A4700 conditional OLS generation attempt",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="new experiment output directory",
    )
    parser.add_argument(
        "--ratings",
        help="completed blind-rating CSV; omit to prepare the review packet",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    root = Path(__file__).resolve().parents[3]
    output_dir = _resolve(root, args.output_dir)
    if args.ratings:
        result = _finalize(output_dir, _resolve(root, args.ratings))
        print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
        return 0
    _require(
        not output_dir.exists(),
        f"refusing to overwrite existing output directory: {output_dir}",
    )
    data = _load_inputs(_resolve(root, args.generation_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    prepared = _write_blind_packet(output_dir, data)
    print(json.dumps(prepared, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
