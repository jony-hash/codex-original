"""Prepare and score a 100-item single-reviewer measurement-stability pilot."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import secrets
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_classifier_proxy_validity_v1 as proxy,
)

TASK_ID = "M0_GENERATOR_MEASUREMENT_STABILITY_V1"
DEFAULT_GENERATION_DIR = proxy.DEFAULT_GENERATION_DIR
DEFAULT_PRIOR_REVIEW_DIR = proxy.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20261009_m0_generator_measurement_stability_v1/runs/attempt_01"
)
HISTORICAL_310_DIR = Path(
    "outputs/20261008_m0_generator_frozen_readability_screen_v1/runs/attempt_01"
)
STROKE_REVIEW_DIR = Path(
    "outputs/20261008_m0_generator_stroke_fidelity_comparison_v1/runs/attempt_08"
)
SELECTION_SEED = 2026100951
REPEAT_SEED = 2026100952
ORDER_SEED = 2026100953
CLASS_COUNT = 10
GENERATED_PER_CLASS = 3
EMPIRICAL_PER_CLASS = 3
ANCHOR_PER_CLASS = 1
REPEAT_PER_SOURCE = 10
BLACK_COUNT = 10
ITEM_COUNT = 100
PAGE_SIZE = 50
MIN_REPEAT_SPACING = 50
ALLOWED_ANSWERS = {str(value) for value in range(10)} | {"unclear"}


class StabilityError(RuntimeError):
    """Invalid source, blind packet, or rating evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StabilityError(message)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _atomic_write(path: Path, payload: bytes) -> None:
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pending_", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"),
    )


def _read_mapping(path: Path, *, stroke: bool = False) -> list[dict[str, Any]]:
    _require(path.is_file(), f"historical review mapping is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("items")
    _require(isinstance(rows, list) and rows, f"historical review mapping is malformed: {path}")
    if not stroke:
        for row in rows:
            kind = row.get("source_type")
            _require(kind in {"generated", "generated_repeat", "empirical_ols", "real_target", "blank"}, f"unknown historical review item type in {path}")
            if kind in {"generated", "generated_repeat"}:
                _require(isinstance(row.get("requested_class"), int) and isinstance(row.get("sample_index"), int), "historical sampled image identity is incomplete")
            if kind == "empirical_ols":
                _require(isinstance(row.get("source_row_index"), int), "historical OLS image identity is incomplete")
            if kind == "real_target":
                _require(isinstance(row.get("source_row_index"), int), "historical target image identity is incomplete")
    else:
        for row in rows:
            kind = row.get("kind")
            _require(kind in {"primary_generated", "real_target_anchor", "hidden_repeat"}, f"unknown stroke review item type in {path}")
            if kind in {"primary_generated", "hidden_repeat"}:
                _require(isinstance(row.get("endpoint"), str), "historical generated image endpoint is missing")
                if row.get("endpoint") == "A":
                    _require(isinstance(row.get("source_row_index"), int), "historical A4700 image identity is incomplete")
            if kind == "real_target_anchor":
                _require(isinstance(row.get("source_row_index"), int), "historical target anchor identity is incomplete")
    return rows


def _excluded_identities(
    prior_review_dir: Path,
    historical_310_dir: Path,
    stroke_review_dir: Path,
) -> tuple[dict[str, set[Any]], dict[str, int]]:
    sets: dict[str, set[Any]] = {
        "conditional": set(),
        "empirical_ols": set(),
        "real_target": set(),
    }
    counts: Counter[str] = Counter()
    mapping_paths = [
        (prior_review_dir / "analysis" / "review_mapping.json", False),
        (historical_310_dir / "analysis" / "review_mapping.json", False),
        (stroke_review_dir / "review_mapping.json", True),
    ]
    for path, stroke in mapping_paths:
        rows = _read_mapping(path, stroke=stroke)
        for row in rows:
            if not stroke:
                kind = row["source_type"]
                if kind in {"generated", "generated_repeat"}:
                    sets["conditional"].add((int(row["requested_class"]), int(row["sample_index"])))
                    counts["conditional"] += 1
                elif kind == "empirical_ols":
                    sets["empirical_ols"].add(int(row["source_row_index"]))
                    counts["empirical_ols"] += 1
                elif kind == "real_target":
                    sets["real_target"].add(int(row["source_row_index"]))
                    counts["real_target"] += 1
            else:
                kind = row["kind"]
                if kind in {"primary_generated", "hidden_repeat"} and row.get("endpoint") == "A":
                    sets["empirical_ols"].add(int(row["source_row_index"]))
                    counts["empirical_ols"] += 1
                elif kind == "real_target_anchor":
                    sets["real_target"].add(int(row["source_row_index"]))
                    counts["real_target"] += 1
    return sets, {"mapping_rows_considered": int(sum(counts.values())), **{f"{key}_mapping_rows": counts[key] for key in sets}, **{f"{key}_unique_excluded": len(value) for key, value in sets.items()}}


def _image_record(view: torch.Tensor, item_id: str, **fields: Any) -> dict[str, Any]:
    tensor = view.detach().to(device="cpu", dtype=torch.float32).contiguous()
    _require(tuple(tensor.shape) == (1, 28, 28), "review image must be the saved 28×28 view")
    _require(bool(torch.isfinite(tensor).all() and (tensor >= 0).all() and (tensor <= 1).all()), "review image is outside its saved fixed [0,1] view")
    uri = proxy.saved_evidence._image_data_uri(tensor[0])
    encoded = uri.partition(",")[2]
    _require(uri.startswith("data:image/png;base64,") and encoded, "could not encode saved view as PNG")
    png = base64.b64decode(encoded, validate=True)
    row = {"item_id": item_id, "_image_data": uri, "image_sha256": hashlib.sha256(png).hexdigest(), **fields}
    return row


def _select_records(data: Mapping[str, Any], excluded: Mapping[str, set[Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for field, shape in (("sample_views", (2000, 1, 28, 28)), ("empirical_views", (200, 1, 28, 28)), ("target_views", (3300, 1, 28, 28))):
        _require(tuple(data[field].shape) == shape, f"saved {field} geometry differs from the fixed source")
    _require(torch.equal(data["sample_index"], torch.arange(200).repeat(10)), "conditional sample indices differ from the fixed 10×200 layout")
    _require(torch.equal(data["sample_requested"], torch.arange(10).repeat_interleave(200)), "conditional sample class layout differs from the fixed source")
    _require(torch.equal(torch.bincount(data["empirical_labels"], minlength=10), torch.full((10,), 20)), "saved empirical reference does not contain 20 rows per class")
    _require(torch.equal(torch.bincount(data["labels"][3000:3300], minlength=10), torch.full((10,), 30)), "saved real-target pool does not contain 30 rows per class")

    conditional_candidates: dict[int, list[int]] = defaultdict(list)
    empirical_candidates: dict[int, list[int]] = defaultdict(list)
    target_candidates: dict[int, list[int]] = defaultdict(list)
    for position in range(2000):
        key = (int(data["sample_requested"][position]), int(data["sample_index"][position]))
        if key not in excluded["conditional"]:
            conditional_candidates[key[0]].append(position)
    for position in range(200):
        digit = int(data["empirical_labels"][position])
        source_row = int(data["empirical_rows"][position])
        if source_row not in excluded["empirical_ols"]:
            empirical_candidates[digit].append(position)
    for position in range(3000, 3300):
        digit = int(data["labels"][position])
        source_row = int(data["source_rows"][position])
        if source_row not in excluded["real_target"]:
            target_candidates[digit].append(position)

    available = {
        "conditional_by_class": {str(digit): len(conditional_candidates[digit]) for digit in range(10)},
        "empirical_ols_by_class": {str(digit): len(empirical_candidates[digit]) for digit in range(10)},
        "real_target_by_class": {str(digit): len(target_candidates[digit]) for digit in range(10)},
    }
    for digit in range(10):
        _require(len(conditional_candidates[digit]) >= GENERATED_PER_CLASS, f"class {digit} has too few eligible conditional samples")
        _require(len(empirical_candidates[digit]) >= EMPIRICAL_PER_CLASS, f"class {digit} has too few eligible empirical OLS images")
        _require(len(target_candidates[digit]) >= ANCHOR_PER_CLASS, f"class {digit} has too few eligible real-target controls")

    selection_rng = random.Random(SELECTION_SEED)
    selected: list[dict[str, Any]] = []
    repeated_sources: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for digit in range(10):
        for position in selection_rng.sample(sorted(conditional_candidates[digit]), GENERATED_PER_CLASS):
            row = _image_record(
                data["sample_views"][position], secrets.token_hex(6), source_type="conditional_sample",
                requested_class=digit, sample_index=int(data["sample_index"][position]),
                e_d_prediction=int(data["sample_e_d_prediction"][position]),
            )
            selected.append(row)
            repeated_sources[("conditional_sample", digit)].append(row)
        for position in selection_rng.sample(sorted(empirical_candidates[digit]), EMPIRICAL_PER_CLASS):
            row = _image_record(
                data["empirical_views"][position], secrets.token_hex(6), source_type="empirical_ols",
                requested_class=digit, sample_index=position, source_row_index=int(data["empirical_rows"][position]),
                e_d_prediction=int(data["empirical_e_d_prediction"][position]),
            )
            selected.append(row)
            repeated_sources[("empirical_ols", digit)].append(row)
        position = selection_rng.choice(sorted(target_candidates[digit]))
        selected.append(_image_record(
            data["target_views"][position], secrets.token_hex(6), source_type="real_target",
            true_class=digit, source_row_index=int(data["source_rows"][position]),
        ))

    repeat_rng = random.Random(REPEAT_SEED)
    repeats: list[dict[str, Any]] = []
    for digit in range(10):
        for source_type in ("conditional_sample", "empirical_ols"):
            source = repeat_rng.choice(repeated_sources[(source_type, digit)])
            repeats.append({
                key: value for key, value in _image_record(
                    data["sample_views"][digit * 200 + int(source["sample_index"])] if source_type == "conditional_sample" else data["empirical_views"][int(source["sample_index"])],
                    secrets.token_hex(6), source_type=f"{source_type}_repeat", requested_class=digit,
                    sample_index=int(source["sample_index"]), source_row_index=source.get("source_row_index"),
                    e_d_prediction=source.get("e_d_prediction"), repeat_of=source["item_id"],
                ).items() if value is not None
            })
    black = torch.zeros((1, 28, 28), dtype=torch.float32)
    for _ in range(BLACK_COUNT):
        selected.append(_image_record(black, secrets.token_hex(6), source_type="black_control"))
    _require(len(selected) == 80 and len(repeats) == 20, "fixed pilot sample composition changed")
    selected.extend(repeats)

    order_rng = random.Random(ORDER_SEED)
    repeat_original_ids = {str(row["repeat_of"]) for row in repeats}
    first_group = [row for row in selected if row["item_id"] in repeat_original_ids]
    other_rows = [row for row in selected if row["item_id"] not in repeat_original_ids and not row["source_type"].endswith("_repeat")]
    order_rng.shuffle(first_group)
    order_rng.shuffle(other_rows)
    primary_order = first_group + other_rows
    _require(len(primary_order) == 80 and len(primary_order) >= 40, "primary blind sample assembly changed")
    ordered_first_group = primary_order[:40]
    repeats_by_parent = {str(row["repeat_of"]): row for row in repeats}
    repeat_positions: dict[str, int] = {}
    free_slots = list(range(50, ITEM_COUNT))
    for original_position in sorted(range(40), key=lambda position: position, reverse=True):
        original = ordered_first_group[original_position]
        repeat = repeats_by_parent.get(str(original["item_id"]))
        if repeat is None:
            continue
        choices = [slot for slot in free_slots if slot - original_position >= MIN_REPEAT_SPACING]
        _require(bool(choices), "no legal hidden-repeat slot satisfies the 50-question spacing")
        slot = order_rng.choice(choices)
        free_slots.remove(slot)
        repeat_positions[str(repeat["item_id"])] = slot
    remaining_slots = set(range(40, ITEM_COUNT)) - set(repeat_positions.values())
    for repeat in repeats:
        if str(repeat["item_id"]) not in repeat_positions:
            choices = sorted(remaining_slots)
            _require(bool(choices), "blind order has no remaining repeat slot")
            slot = order_rng.choice(choices)
            remaining_slots.remove(slot)
            repeat_positions[str(repeat["item_id"])] = slot
    ordered: list[dict[str, Any] | None] = [None] * ITEM_COUNT
    for position, primary in enumerate(ordered_first_group):
        ordered[position] = primary
    remaining_primary_positions = sorted(set(range(40, ITEM_COUNT)) - set(repeat_positions.values()))
    order_rng.shuffle(remaining_primary_positions)
    for position, primary in zip(remaining_primary_positions, primary_order[40:], strict=True):
        ordered[position] = primary
    for repeat in repeats:
        ordered[repeat_positions[str(repeat["item_id"])]] = repeat
    _require(all(row is not None for row in ordered), "blind order contains an unfilled question slot")
    ordered = [row for row in ordered if row is not None]
    for repeat in repeats:
        _require(str(repeat["item_id"]) in repeat_positions, "a hidden repeat has no reserved position")
    _require(len(ordered) == ITEM_COUNT, "ordered review does not contain 100 questions")
    order_index = {str(row["item_id"]): index for index, row in enumerate(ordered)}
    for repeat in repeats:
        distance = order_index[str(repeat["item_id"])] - order_index[str(repeat["repeat_of"])]
        _require(distance >= MIN_REPEAT_SPACING, "hidden repeat is too close to its original")
        source = next(row for row in ordered if row["item_id"] == repeat["repeat_of"])
        _require(source["image_sha256"] == repeat["image_sha256"], "hidden repeat image bytes differ from the original")
    _require(len({row["item_id"] for row in ordered}) == ITEM_COUNT, "question IDs are not unique")

    public = [{"item_id": str(row["item_id"]), "image": str(row["_image_data"])} for row in ordered]
    private = [{key: value for key, value in row.items() if key != "_image_data"} for row in ordered]
    return (public, private), {"available_candidates": available, "selected_counts": {
        "conditional_sample": 30, "empirical_ols": 30, "hidden_repeats": 20,
        "real_target_controls": 10, "black_controls": 10, "total": 100,
    }}


def _render_html(public_items: Sequence[Mapping[str, str]], review_id: str) -> str:
    payload = json.dumps(list(public_items), ensure_ascii=False, separators=(",", ":"))
    token = json.dumps(review_id)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>图像识读稳定性盲判</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:18px;background:#f4f4f4;color:#222}}header{{position:sticky;top:0;background:white;padding:12px;border:1px solid #ccc;z-index:2}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px;margin-top:14px}}.card{{background:white;border:1px solid #ddd;padding:8px;text-align:center}}
.card img{{width:224px;height:224px;image-rendering:pixelated;object-fit:contain;background:white}}.id{{font-family:monospace;color:#555;margin:4px}}
select,button{{font-size:16px;padding:6px;margin:3px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><strong>单人图像识读稳定性检查</strong>
<p>仅根据图像作答：只有能辨认时才选择 0–9；不能辨认或不能稳定辨认时请选择“无法辨认”。图像均为保存的 28×28 视图，灰度范围固定为 [0,1]，放大 8 倍；没有逐图增强或修饰。</p>
<p>共两组，每组 50 题。请完成全部 100 题后再导出评分，不要查看标签文件、答案或中间统计。</p>
<label>恢复进度：<input id="import" type="file" accept=".csv,text/csv"></label><div><button id="prev">上一组</button><span id="page"></span><button id="next">下一组</button>
<button id="exportProgress">导出进度</button><button id="exportFinal">导出完整评分</button></div><span id="progress"></span><p id="message">进度保存在当前浏览器；可导出并在之后载入。</p></header>
<main id="grid" class="grid"></main><script>
const items={payload},reviewId={token},pageSize={PAGE_SIZE},answers={{}};const allowed=new Set([...Array(10).keys()].map(String).concat(['unclear']));
const storageKey='lightgen-measurement-stability-'+reviewId;try{{const old=localStorage.getItem(storageKey);if(old)Object.assign(answers,JSON.parse(old));}}catch{{}}
const grid=document.getElementById('grid');let page=0;function persist(){{try{{localStorage.setItem(storageKey,JSON.stringify(answers));}}catch{{}}updateProgress();}}
function render(){{grid.innerHTML='';const start=page*pageSize,visible=items.slice(start,start+pageSize);for(const item of visible){{const card=document.createElement('section');card.className='card';const image=document.createElement('img');image.src=item.image;image.alt='';image.draggable=false;const id=document.createElement('div');id.className='id';id.textContent='编号 '+item.item_id;const select=document.createElement('select');select.setAttribute('aria-label','编号 '+item.item_id+' 的读数');const empty=document.createElement('option');empty.value='';empty.textContent='请选择';select.appendChild(empty);for(let d=0;d<10;d++){{const option=document.createElement('option');option.value=String(d);option.textContent=String(d);select.appendChild(option);}}const unclear=document.createElement('option');unclear.value='unclear';unclear.textContent='无法辨认';select.appendChild(unclear);select.value=answers[item.item_id]||'';select.onchange=()=>{{if(select.value)answers[item.item_id]=select.value;else delete answers[item.item_id];persist();}};card.append(image,id,select);grid.appendChild(card);}}
const first=start+1,last=Math.min(start+pageSize,items.length);document.getElementById('page').textContent=`第 ${{page+1}} / 2 组（${{first}}–${{last}}）`;document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=page===1;updateProgress();}}
function updateProgress(){{const count=Object.keys(answers).filter(id=>items.some(x=>x.item_id===id)&&allowed.has(answers[id])).length;document.getElementById('progress').textContent=`已完成 ${{count}} / 100 项`;document.getElementById('exportFinal').disabled=count!==100;}}
function download(requireComplete){{const valid=items.filter(x=>answers[x.item_id]&&allowed.has(answers[x.item_id])).length;if(requireComplete&&valid!==100)return;const lines=['item_id,answer',...items.map(x=>`${{x.item_id}},${{answers[x.item_id]||''}}`)];const blob=new Blob(['\\ufeff'+lines.join('\\r\\n')],{{type:'text/csv;charset=utf-8'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`lightgen_measurement_stability_${{requireComplete?'ratings':'progress'}}_${{reviewId}}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}}
document.getElementById('prev').onclick=()=>{{page--;render();}};document.getElementById('next').onclick=()=>{{page++;render();}};document.getElementById('exportProgress').onclick=()=>download(false);document.getElementById('exportFinal').onclick=()=>download(true);
document.getElementById('import').onchange=e=>{{const file=e.target.files[0];if(!file)return;const reader=new FileReader();reader.onload=()=>{{try{{const lines=String(reader.result).replace(/^\\uFEFF/,'').trim().split(/\\r?\\n/);if(lines.shift()!=='item_id,answer')throw Error('CSV 表头错误');const known=new Set(items.map(x=>x.item_id)),incoming={{}};for(const line of lines){{const f=line.split(',');if(f.length!==2||!known.has(f[0])||Object.hasOwn(incoming,f[0]))throw Error('评分行无效');const [id,answer]=f;if(answer&&!allowed.has(answer))throw Error('评分值无效');incoming[id]=answer;}}for(const [id,answer] of Object.entries(incoming)){{if(answer)answers[id]=answer;else delete answers[id];}}persist();render();document.getElementById('message').textContent='进度已载入。';}}catch(err){{document.getElementById('message').textContent='载入失败：'+err.message;}}}};reader.readAsText(file);}};render();
</script></body></html>"""


def _prepare_output(root: Path, output_dir: Path) -> Path:
    output_root = (root / "outputs").resolve()
    target = output_dir.resolve()
    _require(target != output_root and output_root in target.parents, "output must be a child of outputs/")
    parts = target.relative_to(output_root).parts
    _require(len(parts) == 3 and parts[0].startswith("20261009_") and parts[1] == "runs" and parts[2].startswith("attempt_"), "output must use outputs/YYYYMMDD_<version>/runs/attempt_N")
    _require(not target.exists(), f"output already exists; refusing to overwrite: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _prepare(generation_dir: Path, prior_review_dir: Path, output_arg: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    output_dir = _prepare_output(root, output_arg)
    data = proxy._load_inputs(generation_dir.resolve())
    excluded, exclusion_counts = _excluded_identities(
        prior_review_dir.resolve(), _resolve(root, HISTORICAL_310_DIR), _resolve(root, STROKE_REVIEW_DIR)
    )
    (public, mapping), sample_counts = _select_records(data, excluded)
    review_id = secrets.token_hex(8)
    _atomic_write(output_dir / "blind_review.html", _render_html(public, review_id).encode("utf-8"))
    template = "\ufeffitem_id,answer\n" + "\n".join(f"{item['item_id']}," for item in public) + "\n"
    _atomic_write(output_dir / "ratings_template.csv", template.encode("utf-8"))
    _write_json(output_dir / "analysis" / "review_mapping.json", {"task_id": TASK_ID, "review_id": review_id, "items": mapping})
    exclusions = {
        "source_generation_dir": str(data["generation_dir"]),
        "review_mapping_paths": [
            str(prior_review_dir.resolve() / "analysis" / "review_mapping.json"),
            str(_resolve(root, HISTORICAL_310_DIR) / "analysis" / "review_mapping.json"),
            str(_resolve(root, STROKE_REVIEW_DIR) / "review_mapping.json"),
        ],
        "excluded_identity_counts": exclusion_counts,
        "selection_seeds": {"sample": SELECTION_SEED, "repeat": REPEAT_SEED, "order": ORDER_SEED},
        "sample_counts": sample_counts["selected_counts"],
        "available_candidates_after_exclusion": sample_counts["available_candidates"],
        "display": {"view": "saved 28×28", "range": [0, 1], "integer_scale": 8, "per_image_enhancement": False},
        "blindness": {"public_fields": ["item_id", "image"], "answers_before_complete": False, "intermediate_statistics": False},
        "repeat_spacing_minimum_questions": MIN_REPEAT_SPACING,
        "repeat_spacing_observed_minimum": min(
            mapping.index(next(row for row in mapping if row["item_id"] == repeat["item_id"])) - mapping.index(next(row for row in mapping if row["item_id"] == repeat["repeat_of"]))
            for repeat in mapping if str(repeat["source_type"]).endswith("_repeat")
        ),
    }
    _write_json(output_dir / "eligibility.json", exclusions)
    instructions = (
        "# 实验：测量稳定性检查\n\n"
        "请打开 `blind_review.html`，只依据图像回答 0–9 或“无法辨认”。只有能辨认时才选数字；不能辨认或无法稳定辨认时选择“无法辨认”。\n\n"
        "共两组、每组50题。完成全部100题后导出完整CSV。评分期间不要查看 `analysis/review_mapping.json` 或中间统计。\n\n"
        "评分完成后，将完整CSV提供给评分入口：\n\n"
        "```powershell\n"
        "python -B -m scripts.research.lightgen.audit_lightgen_2d_generator_measurement_stability_v1 score "
        f"--output-dir {output_arg.as_posix()} --ratings \"<评分CSV路径>\"\n"
        "```\n"
    )
    _atomic_write(output_dir / "review_instructions.md", instructions.encode("utf-8"))
    prepared = {
        "task_id": TASK_ID,
        "state": "PREPARED_AWAITING_RATINGS",
        "source_generation_dir": str(data["generation_dir"]),
        "reviewer_count": 1,
        "item_count": ITEM_COUNT,
        "page_size": PAGE_SIZE,
        "sample_composition": sample_counts["selected_counts"],
        "selection_seeds": {"sample": SELECTION_SEED, "repeat": REPEAT_SEED, "order": ORDER_SEED},
        "historical_exclusions_verified": True,
        "training_recommendation": "NONE",
        "next_action": "WAIT_FOR_RATINGS",
    }
    _write_json(output_dir / "prepared.json", prepared)
    return prepared


def _parse_ratings(path: Path, mapping: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], list[dict[str, str]]]:
    _require(path.is_file(), f"explicit ratings CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == ["item_id", "answer"], "ratings CSV header must be item_id,answer")
        raw_rows = list(reader)
    expected = {str(row["item_id"]) for row in mapping}
    answers: dict[str, str] = {}
    preserved: list[dict[str, str]] = []
    for row in raw_rows:
        item_id, answer = str(row.get("item_id", "")).strip(), str(row.get("answer", "")).strip().lower()
        _require(item_id in expected, f"unknown ratings item ID: {item_id}")
        _require(item_id not in answers, f"duplicate ratings item ID: {item_id}")
        _require(answer in ALLOWED_ANSWERS, f"missing or invalid rating for item {item_id}")
        answers[item_id] = answer
        preserved.append({"item_id": item_id, "answer": answer})
    _require(set(answers) == expected and len(preserved) == ITEM_COUNT, "ratings CSV must contain every item exactly once")
    return answers, preserved


def _score(mapping: Sequence[Mapping[str, Any]], answers: Mapping[str, str]) -> dict[str, Any]:
    _require(len(mapping) == ITEM_COUNT and len({str(row["item_id"]) for row in mapping}) == ITEM_COUNT, "private review map is incomplete or has duplicate IDs")
    _require(set(answers) == {str(row["item_id"]) for row in mapping}, "ratings do not cover the packet exactly")
    by_id = {str(row["item_id"]): row for row in mapping}
    generated = [row for row in mapping if row["source_type"] == "conditional_sample"]
    empirical = [row for row in mapping if row["source_type"] == "empirical_ols"]
    anchors = [row for row in mapping if row["source_type"] == "real_target"]
    blacks = [row for row in mapping if row["source_type"] == "black_control"]
    repeats = [row for row in mapping if str(row["source_type"]).endswith("_repeat")]
    _require((len(generated), len(empirical), len(anchors), len(blacks), len(repeats)) == (30, 30, 10, 10, 20), "private review map composition differs from plan")
    for source_type in ("conditional_sample", "empirical_ols"):
        _require(Counter(int(row["requested_class"]) for row in mapping if row["source_type"] == source_type) == Counter({digit: 3 for digit in range(10)}), f"{source_type} class balance differs")
    _require(Counter(int(row["true_class"]) for row in anchors) == Counter({digit: 1 for digit in range(10)}), "true-control class balance differs")
    anchor_correct = sum(answers[str(row["item_id"])] == str(row["true_class"]) for row in anchors)
    black_unclear = sum(answers[str(row["item_id"])] == "unclear" for row in blacks)
    repeat_by_source = {"conditional_sample": [], "empirical_ols": []}
    repeat_transitions: list[dict[str, Any]] = []
    for row in repeats:
        parent = by_id.get(str(row.get("repeat_of")))
        _require(parent is not None, "hidden repeat references an unknown original")
        source_type = str(row["source_type"]).removesuffix("_repeat")
        _require(source_type in repeat_by_source and parent["source_type"] == source_type, "hidden repeat source type differs")
        agree = answers[str(row["item_id"])] == answers[str(parent["item_id"])]
        repeat_by_source[source_type].append(bool(agree))
        repeat_transitions.append({
            "source_type": source_type,
            "class": int(row["requested_class"]),
            "original_item_id": str(parent["item_id"]),
            "repeat_item_id": str(row["item_id"]),
            "original_answer": answers[str(parent["item_id"])],
            "repeat_answer": answers[str(row["item_id"])],
            "agree": bool(agree),
        })
    repeat_quality = {kind: {"agree": sum(values), "total": len(values), "passed": sum(values) >= 9} for kind, values in repeat_by_source.items()}
    quality = {
        "real_target_correct": anchor_correct,
        "real_target_total": 10,
        "real_target_passed": anchor_correct >= 9,
        "black_answered_unclear": black_unclear,
        "black_total": 10,
        "black_passed": black_unclear >= 9,
        "repeat_consistency_by_source": repeat_quality,
    }
    quality["passed"] = bool(quality["real_target_passed"] and quality["black_passed"] and all(item["passed"] for item in repeat_quality.values()))
    source_results: dict[str, Any] = {}
    for source_type, rows in (("conditional_sample", generated), ("empirical_ols", empirical)):
        correct = sum(answers[str(row["item_id"])] == str(row["requested_class"]) for row in rows)
        unclear = sum(answers[str(row["item_id"])] == "unclear" for row in rows)
        confusion = [[0 for _ in range(11)] for _ in range(10)]
        for row in rows:
            answer = answers[str(row["item_id"])]
            confusion[int(row["requested_class"])][10 if answer == "unclear" else int(answer)] += 1
        source_results[source_type] = {
            "n": len(rows), "matches_requested_class": correct, "match_fraction": correct / len(rows),
            "unclear": unclear, "unclear_fraction": unclear / len(rows),
            "per_class": {
                str(digit): {
                    "n": sum(int(row["requested_class"]) == digit for row in rows),
                    "matches": sum(int(row["requested_class"]) == digit and answers[str(row["item_id"])] == str(digit) for row in rows),
                    "unclear": sum(int(row["requested_class"]) == digit and answers[str(row["item_id"])] == "unclear" for row in rows),
                } for digit in range(10)
            },
            "confusion_requested_rows_answer_columns_0_to_9_unclear": confusion,
        }
    return {
        "classification": "PILOT_QUALITY_CHECKS_PASSED" if quality["passed"] else "PILOT_QUALITY_CHECKS_FAILED",
        "quality_checks": quality,
        "source_reading": source_results,
        "repeat_transitions": repeat_transitions,
        "scope_limit": "One reviewer and these 100 items only; does not establish population-wide stability or E_D validity.",
        "confidence_interval": None,
        "proxy_validity_gate_applied": False,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows) and not path.exists(), f"cannot write CSV or artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(report: Mapping[str, Any]) -> str:
    decision = report["classification"]
    quality = report["results"]["quality_checks"]
    stability_text = "达到本次预设内部检查" if quality["passed"] else "未达到本次预设内部检查"
    reading = report["results"]["source_reading"]
    return f"""# 单人图像识读稳定性检查

## 结果与状态

状态：`{decision}`。本次审阅质量检查{stability_text}：真实数字 {quality['real_target_correct']}/10，全黑图“无法辨认” {quality['black_answered_unclear']}/10；条件采样重复一致 {quality['repeat_consistency_by_source']['conditional_sample']['agree']}/10，经验OLS重复一致 {quality['repeat_consistency_by_source']['empirical_ols']['agree']}/10。

## 科学认识

在这一位审阅者和这100题上，条件采样图请求类别识读 {reading['conditional_sample']['matches_requested_class']}/30，无法辨认 {reading['conditional_sample']['unclear']}/30；经验OLS图识读 {reading['empirical_ols']['matches_requested_class']}/30，无法辨认 {reading['empirical_ols']['unclear']}/30。重复转移与逐类计数见 `report.json` 和 `analysis/paired_rows.csv`。

## 目标缺口

这是小规模单人测量稳定性检查，不估计跨审阅者一致性，也不证明总体识读稳定性。其通过不代表分类代理效度；失败也不等于生成模型失效。未使用实验一的10%门槛或超几何区间。

## 工程完成与验证

已从固定A4700条件采样生成与经验OLS保存视图抽样；依据类别和组内索引、OLS与真实图源行号排除了实验一、历史310题及笔画审阅映射中的已审图像。盲页仅含匿名编号与固定28×28视图，采用统一[0,1]灰度和8倍整数放大。完整回答行数 {report['validation']['answer_rows']}，盲映射与CSV逐题匹配通过。

## 决策影响与下一步

本次检查仅描述这位审阅者对此100题的测量稳定性。不得扩展为人群效度或E_D效度结论；本任务已停止，`training_recommendation=NONE`、`next_action=NONE`。
"""


def _score_output(output_dir: Path, ratings_path: Path) -> dict[str, Any]:
    map_path = output_dir / "analysis" / "review_mapping.json"
    _require(map_path.is_file(), "prepared private review map is missing")
    prepared = json.loads((output_dir / "prepared.json").read_text(encoding="utf-8"))
    _require(prepared.get("state") == "PREPARED_AWAITING_RATINGS" and prepared.get("task_id") == TASK_ID, "review packet is not awaiting ratings")
    _require(not (output_dir / "completed.json").exists(), "review has already been completed")
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    _require(payload.get("task_id") == TASK_ID, "private review map task identity differs")
    mapping = payload["items"]
    answers, preserved = _parse_ratings(ratings_path.resolve(), mapping)
    results = _score(mapping, answers)
    _write_csv(output_dir / "analysis" / "raw_answers.csv", preserved)
    paired = []
    for row in mapping:
        paired.append({key: value for key, value in {
            "item_id": str(row["item_id"]), "source_type": row["source_type"],
            "requested_class": row.get("requested_class", ""), "true_class": row.get("true_class", ""),
            "sample_index": row.get("sample_index", ""), "source_row_index": row.get("source_row_index", ""),
            "e_d_prediction": row.get("e_d_prediction", ""), "repeat_of": row.get("repeat_of", ""),
            "image_sha256": row["image_sha256"], "answer": answers[str(row["item_id"])],
        }.items()})
    _write_csv(output_dir / "analysis" / "paired_rows.csv", paired)
    validation = {"passed": True, "answer_rows": len(preserved), "mapping_rows": len(mapping), "all_item_ids_unique": len(answers) == ITEM_COUNT, "complete_and_valid": True}
    report = {
        "task_id": TASK_ID,
        "status": "COMPLETED",
        "classification": results["classification"],
        "results": results,
        "validation": validation,
        "scope": {"reviewers": 1, "items": 100, "source": str(prepared["source_generation_dir"]), "groups": {"conditional_sample": 30, "empirical_ols": 30, "real_target_control": 10, "black_control": 10, "hidden_repeats": 20}},
        "training_recommendation": "NONE",
        "next_action": "NONE",
    }
    _write_json(output_dir / "report.json", report)
    _atomic_write(output_dir / "report.md", _report_markdown(report).encode("utf-8"))
    _write_json(output_dir / "completed.json", {"task_id": TASK_ID, "status": "COMPLETED", "classification": results["classification"], "training_recommendation": "NONE", "next_action": "NONE"})
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--generation-dir", required=True, type=Path)
    prepare.add_argument("--prior-review-dir", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    score = commands.add_parser("score")
    score.add_argument("--output-dir", required=True, type=Path)
    score.add_argument("--ratings", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.command == "prepare":
        result = _prepare(args.generation_dir, args.prior_review_dir, args.output_dir)
    else:
        result = _score_output(args.output_dir.resolve(), args.ratings)
    print(json.dumps({"state": result.get("state", result.get("status")), "classification": result.get("classification"), "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
