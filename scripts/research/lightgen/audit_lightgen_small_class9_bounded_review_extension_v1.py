"""Prepare one fixed saved-image review extension after the mixed class-9 audit."""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import time
import uuid
from typing import Any

import numpy as np

from scripts.research.lightgen import audit_lightgen_small_class9_saved_defect_diagnosis_v1 as base


TASK_ID = "LIGHTGEN_SMALL_CLASS9_BOUNDED_REVIEW_EXTENSION_V1"
SELECTION_SEED = 2026102105
SHUFFLE_SEED = 2026102106
STRATA = (("requested9_ed4", 9, 4, 10), ("requested9_ed8", 9, 8, 10),
          ("requested9_ed9", 9, 9, 4), ("requested4_unfiltered", 4, None, 2),
          ("requested8_unfiltered", 8, None, 2))
CATEGORIES = ("readable9", "readable_other", "unreadable")
FILES = ("preflight.json", "review_contract.json", "blind_review_manifest.json", "blind_review.html",
         "report.json", "report.html", "unblinded_comparison.html", "validation_receipt.json", "closeout_report.md")


def _category(rating: str) -> str:
    return "unreadable" if rating == "unreadable" else "readable9" if rating == "9" else "readable_other"


def _load_parent(parent_dir: Path, inputs: base.Inputs) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    report = base._read_json(parent_dir / "report.json")
    manifest = base._read_json(parent_dir / "blind_review_manifest.json")
    payload = base._read_json(parent_dir / "human_ratings.json")
    receipt = base._read_json(parent_dir / "human_finalization_verification.json")
    base._require(report.get("task_id") == base.TASK_ID and report.get("status") == "COMPLETE_DIAGNOSTIC"
                  and report.get("measurement_complete") is True
                  and report.get("decision", {}).get("code") == "MIXED_OR_UNRESOLVED",
                  "parent must be the completed mixed class-9 review")
    base._require(report["inputs"]["attribution_dir"] == str(inputs.attribution_dir)
                  and report["inputs"]["ab_evidence_dir"] == str(inputs.ab_evidence_dir),
                  "extension input paths differ from the completed parent")
    ratings = base._validate_ratings(payload, manifest)
    diagnosed = base._human_diagnosis(manifest, ratings)
    base._require(diagnosed == report.get("human_diagnosis"), "parent diagnosis does not match its preserved ratings")
    base._require(receipt.get("status") == "PASSED" and receipt.get("review_id") == manifest["review_id"]
                  and receipt.get("decision") == "MIXED_OR_UNRESOLVED", "parent independent verification is absent")
    return report, manifest, ratings


def _select(inputs: base.Inputs, parent_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    used = {(e["requested_label"], e["proposal_id"]) for e in parent_manifest["entries"]}
    base._require(len(used) == 60, "parent selection must have 60 unique class/proposal IDs")
    rng = np.random.default_rng(SELECTION_SEED)
    entries = []
    for cell in ("AA", "BB"):
        for stratum, label, ed_prediction, count in STRATA:
            block = inputs.blocks[(cell, label)]
            condition = block["conditions"][f"{cell}_g1p00"]
            logits = condition["logits"].numpy()
            predictions, _, _ = base._recompute_predictions(logits, np.full(1000, label, dtype=np.int64))
            base._require(np.array_equal(predictions, condition["predicted_labels"].numpy()),
                          f"{cell}/class{label} saved predictions differ from logits")
            candidates = np.flatnonzero(predictions == ed_prediction) if label == 9 else np.arange(1000)
            candidates = [int(i) for i in candidates if (label, int(inputs.proposal_ids[i])) not in used]
            base._require(len(candidates) >= count, f"fixed extension stratum is insufficient: {cell}/{stratum}")
            for index in rng.choice(candidates, size=count, replace=False).tolist():
                proposal = int(inputs.proposal_ids[index])
                used.add((label, proposal))
                entries.append({"selection_index": len(entries), "cell": cell, "condition_id": f"{cell}_g1p00",
                                "requested_label": label, "proposal_id": proposal, "cohort_index": index,
                                "ed_prediction": int(predictions[index]), "stratum": stratum})
    base._require(len(entries) == 56 and len(used) == 116, "extension selection lost or repeated rows")
    for display, selected in enumerate(np.random.default_rng(SHUFFLE_SEED).permutation(56), 1):
        entries[int(selected)]["blind_id"] = f"extension_{display:03d}"
        entries[int(selected)]["display_order"] = display
    return sorted(entries, key=lambda e: e["display_order"])


def _validate_ratings(payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str]:
    base._require(payload.get("schema") == "lightgen_blind_ratings.v1" and payload.get("task_id") == TASK_ID
                  and payload.get("review_id") == manifest["review_id"], "ratings belong to a different task or review")
    rows = payload.get("ratings")
    base._require(isinstance(rows, list) and len(rows) == 56, "extension requires exactly 56 ratings")
    expected = {entry["blind_id"] for entry in manifest["entries"]}
    result = {}
    for row in rows:
        base._require(isinstance(row, dict) and set(row) == {"blind_id", "rating"}, "rating row fields differ")
        card, label = row["blind_id"], row["rating"]
        base._require(isinstance(card, str) and card in expected and card not in result,
                      "extension ratings have an unknown or duplicate card")
        base._require(isinstance(label, str) and label in base.RATING_LABELS, "extension rating label is invalid")
        result[card] = label
    base._require(set(result) == expected, "extension ratings are incomplete")
    return result


def _diagnose(parent_manifest: dict[str, Any], parent_ratings: dict[str, str],
              manifest: dict[str, Any], ratings: dict[str, str]) -> dict[str, Any]:
    counts = {cell: Counter() for cell in ("AA", "BB")}
    wave_counts = {"parent20": Counter(), "extension40": Counter()}
    for wave, mapping, labels in (("parent20", parent_manifest, parent_ratings),
                                  ("extension40", manifest, ratings)):
        for entry in mapping["entries"]:
            if entry["cell"] in counts and entry["requested_label"] == 9 and entry["ed_prediction"] in (4, 8):
                category = _category(labels[entry["blind_id"]])
                counts[entry["cell"]][category] += 1
                wave_counts[wave][category] += 1
    base._require(sum(wave_counts["parent20"].values()) == 20
                  and sum(wave_counts["extension40"].values()) == 40
                  and all(sum(c.values()) == 30 for c in counts.values()), "primary60 population is incomplete")
    pooled = {category: sum(counts[cell][category] for cell in counts) for category in CATEGORIES}
    intervals = {category: base._wilson(pooled[category], 60) for category in CATEGORIES}
    qualifying = [category for category in CATEGORIES
                  if all(counts[cell][category] > 15 for cell in counts) and intervals[category][0] > 0.5]
    base._require(len(qualifying) <= 1, "majority rule cannot have multiple qualifying categories")
    direction = qualifying[0] if qualifying else None
    recommendation = {
        "readable9": "优先核验独立身份评价，不依据E_D错判增加训练损失。",
        "readable_other": "优先设计固定预算身份/笔画保真训练对照，禁止使用确认E_D作为训练目标。",
        "unreadable": "优先设计固定增益下的可视强度与空间对比质量改动。",
        None: "固定扩展后仍混合，保留可辨性与评价不一致并存；停止追加评阅，只设计独立质量评价核验。",
    }[direction]
    return {"primary_cards": 60, "per_cell_cards": 30,
            "counts": pooled, "wilson_95": intervals,
            "per_cell": {cell: {k: counts[cell][k] for k in CATEGORIES} for cell in counts},
            "per_wave": {wave: {k: counter[k] for k in CATEGORIES} for wave, counter in wave_counts.items()},
            "decision_category": direction, "recommendation": recommendation,
            "interpretation": "equal E_D4/E_D8 strata; descriptive single-review evidence, not full-population accuracy or paired A/B quality effect"}


def _contract(parent_dir: Path, inputs: base.Inputs) -> dict[str, Any]:
    return {"task_id": TASK_ID, "question": "One bounded extension to distinguish dominant class-9 readability issue or retain a mixture.",
            "parent_dir": str(parent_dir), "attribution_dir": str(inputs.attribution_dir),
            "ab_evidence_dir": str(inputs.ab_evidence_dir), "gain_multiplier": 1.0,
            "additional_cards": 56, "additional_primary": 40, "combined_primary": 60,
            "cells": ["AA", "BB"], "per_cell_strata": [list(row) for row in STRATA],
            "selection_seed": SELECTION_SEED, "shuffle_seed": SHUFFLE_SEED,
            "selection_excludes": "all 60 parent class/proposal pairs; no human-label, brightness or appearance filtering",
            "decision": "Same category strict majority in both AA/BB primary30, and pooled60 Wilson95 lower >0.5.",
            "scope": "saved 28x28 images only; zero model/optical/evaluator/sampling/gradient/optimizer and zero new data reads",
            "stop": "Exactly one extension; if mixed, preserve mixture and stop, never add cards to obtain a majority."}


def _html(report: dict[str, Any]) -> str:
    message = html.escape(report["decision"]["recommendation"])
    result = "<p>新标签尚未返回。</p>"
    if "human_diagnosis" in report:
        diagnosis = report["human_diagnosis"]
        result = "<p>合并主要60例：</p><ul>" + "".join(
            f"<li>{category}：{diagnosis['counts'][category]}/60；95% Wilson {diagnosis['wilson_95'][category]}</li>"
            for category in CATEGORIES) + "</ul>"
    return ("<!doctype html><html lang='zh'><meta charset='utf-8'><title>类别9一次固定追加评阅</title>"
            "<style>body{max-width:900px;margin:28px auto;font-family:system-ui,'Microsoft YaHei',sans-serif;line-height:1.7}</style>"
            f"<h1>类别9：一次固定追加评阅</h1><p>状态：{report['status']}</p>"
            "<p>上一轮主要20例：10例9、10例不可辨认、0例其他数字。本轮新增56张未评阅的保存图，主样本40张；原增益、原视图和参数固定。</p>"
            f"{result}<p>{message}</p><p><a href='blind_review.html'>开始56张盲评</a> | <a href='unblinded_comparison.html'>揭盲对照</a></p>"
            "<p>只追加一次。标签齐全后合并主要60例判断，仍混合即按并存结论收尾，不继续加样。</p></html>")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    parent_dir, output_dir = base._resolve(args.parent_dir), base._resolve(args.output_dir)
    inputs = base._load_inputs(base._resolve(args.attribution_dir), base._resolve(args.ab_evidence_dir))
    parent_report, parent_manifest, parent_ratings = _load_parent(parent_dir, inputs)
    entries = _select(inputs, parent_manifest)
    contract = _contract(parent_dir, inputs)
    if args.preflight_only:
        result = {"status": "PREFLIGHT_READY", "task_id": TASK_ID, "selected_cards": len(entries), "contract": contract}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    existing = output_dir.exists()
    base._require(existing == bool(args.resume), "existing output requires --resume; --resume requires existing complete output")
    if existing:
        base._require(all((output_dir / name).is_file() for name in FILES), "extension output is incomplete; refusing resume")
        base._require(base._read_json(output_dir / "review_contract.json") == contract, "saved extension contract differs")
        manifest = base._read_json(output_dir / "blind_review_manifest.json")
        base._require(manifest.get("task_id") == TASK_ID and manifest.get("entries") == entries
                      and manifest.get("parent_review_id") == parent_manifest["review_id"], "saved extension manifest differs")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {"schema": "lightgen_class9_bounded_extension_manifest.v1", "task_id": TASK_ID,
                    "review_id": uuid.uuid4().hex, "parent_review_id": parent_manifest["review_id"],
                    "cards": 56, "entries": entries}
        base._atomic_json(output_dir / "review_contract.json", contract)
        base._atomic_json(output_dir / "blind_review_manifest.json", manifest)
        base._atomic_json(output_dir / "preflight.json", {"status": "PASSED", "task_id": TASK_ID,
                           "parent_status": parent_report["status"], "fixed_inputs": inputs.preflight,
                           "new_cards": 56, "primary_new": 40, "parent_overlap": 0})
        cards = base._public_cards(entries, inputs.blocks)
        (output_dir / "blind_review.html").write_text(base._review_html(
            manifest["review_id"], cards, task_id=TASK_ID, download_name="lightgen_class9_extension_ratings.json"), encoding="utf-8")
    ratings = None
    payload = None
    if args.ratings_file:
        payload = base._read_json(base._resolve(args.ratings_file))
        ratings = _validate_ratings(payload, manifest)
    if existing:
        prior = base._read_json(output_dir / "report.json")
        if prior.get("status") == "COMPLETE_DIAGNOSTIC":
            saved = base._read_json(output_dir / "human_ratings.json")
            _validate_ratings(saved, manifest)
            base._require(payload is None or payload == saved, "completed labels cannot be replaced")
            return prior
    report: dict[str, Any] = {"task_id": TASK_ID, "status": "AWAITING_HUMAN_RATINGS" if ratings is None else "COMPLETE_DIAGNOSTIC",
                             "measurement_complete": ratings is not None, "contract": contract,
                             "review_id": manifest["review_id"], "parent_review_id": parent_manifest["review_id"],
                             "parent_primary20": parent_report["human_diagnosis"]["primary_20_AA_BB_ED_errors"],
                             "decision": {"code": "PENDING_HUMAN_RATINGS", "recommendation": "等待固定56张新图标签；本次只扩展一次。"},
                             "execution": {"model_constructions": 0, "optical_forwards": 0, "encoder_forwards": 0,
                                           "evaluator_forwards": 0, "sampling": 0, "gradients": 0, "optimizer_steps": 0,
                                           "new_train_dev_test_reads": 0},
                             "system_quality_gate": "NOT_REEVALUATED; A/B VALID_FAILED preserved"}
    if ratings is not None:
        diagnosis = _diagnose(parent_manifest, parent_ratings, manifest, ratings)
        report["human_diagnosis"] = diagnosis
        report["decision"] = {"code": diagnosis["decision_category"] or "MIXED_OR_UNRESOLVED", "recommendation": diagnosis["recommendation"]}
        base._atomic_json(output_dir / "human_ratings.json", payload)
    report["execution"]["elapsed_seconds_this_invocation"] = time.perf_counter() - started
    base._atomic_json(output_dir / "report.json", report)
    (output_dir / "report.html").write_text(_html(report), encoding="utf-8")
    (output_dir / "unblinded_comparison.html").write_text(
        base._build_unblinded_html(report, manifest, ratings, inputs.blocks), encoding="utf-8")
    science = "父主要20例为10例9和10例不可辨认，0例其他数字；新增评阅尚未完成。"
    gap = "补齐预定56张标签后，合并主要60例判断；抽样比例不外推全人口。"
    if ratings is not None:
        science = f"固定主要60例人工计数为{report['human_diagnosis']['counts']}；两个原组合分层结果及Wilson区间见机器报告。"
        gap = "固定评阅完成；单一评阅人口不能代表完整生成准确率，系统质量与真实CMOS仍未新增验证。"
    (output_dir / "closeout_report.md").write_text(
        f"结果与状态：{report['status']}；父任务已完成，本次固定56张追加卡片，原历史科学状态不变。\n\n"
        f"科学认识：{science}\n\n"
        f"目标缺口：{gap}\n\n"
        "工程完成与验证：固定原增益、原视图、未重复proposal；所有模型前向、梯度、采样、更新与新数据读取为0。\n\n"
        f"决策影响与下一步：{report['decision']['recommendation']} 不启动训练或调读出；混合不继续追加。\n", encoding="utf-8")
    base._atomic_json(output_dir / "validation_receipt.json", {"task_id": TASK_ID, "status": "PASSED",
                      "parent_ratings_validated": 60, "selected_cards": 56, "new_primary40": 40,
                      "excluded_all_parent_proposals": True, "selection_rows_readback_passed": base._read_json(
                          output_dir / "blind_review_manifest.json")["entries"] == entries,
                      "human_ratings_complete": ratings is not None, "measurement_complete": ratings is not None,
                      "input_and_readout_validation": inputs.preflight["status"]})
    size = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    base._require(size <= 64 * 1024 * 1024, "extension output exceeded 64 MiB")
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir), "cards": 56,
                      "output_bytes": size, "decision": report["decision"]}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-dir", "attribution-dir", "ab-evidence-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ratings-file", type=Path)
    args = parser.parse_args()
    if args.preflight_only and (args.resume or args.ratings_file):
        parser.error("preflight-only cannot finalize or resume")
    if args.ratings_file and not args.resume:
        parser.error("ratings-file requires the saved manifest and --resume")
    try:
        run(args)
    except base.EvidenceError as error:
        raise SystemExit(f"INVALID_HARNESS: {error}") from error


if __name__ == "__main__":
    main()
