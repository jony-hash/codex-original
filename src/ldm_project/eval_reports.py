from __future__ import annotations

import json
from pathlib import Path

_RUN_META_FILE = "run_meta.json"
_EVAL_METRICS_FILE = "artifacts/one_shot_distill_eval_metrics.json"
_SAMPLE_METRICS_FILE = "artifacts/one_shot_sample_metrics.json"

# 保守默认阈值，优先用 baseline JSON 中的 tolerances。
_DEFAULT_TOLERANCES: dict[str, float] = {
    "eval_mse": 0.005,
    "eval_corr": 0.03,
    "eval_dice": 0.03,
    "eval_sat": 0.02,
    "eval_slm_tv": 0.10,
    "eval_diversity_ratio": 0.15,
    "sample_corr": 0.05,
    "sample_dice": 0.05,
    "best_loss": 0.02,
}


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _extract_overall(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    overall = metrics.get("overall", {})
    return {
        "count": overall.get("count"),
        "mse": overall.get("paired_mse"),
        "corr": overall.get("paired_corr", overall.get("corr")),
        "dice": overall.get("paired_dice", overall.get("dice")),
        "sat": overall.get("student_saturation_ratio"),
        "slm_tv": overall.get("slm_tv"),
        "diversity_ratio": overall.get("diversity_ratio"),
    }


def summarize_runs(run_dirs: list[str]) -> dict:
    """汇总一个或多个 outputs/runs/<run> 目录的 one-shot 评估结果。

    返回 dict，key 为 run 目录名，value 包含：
      - run_dir: 绝对路径
      - stage: 从 run_meta.mode 提取
      - checkpoint: one_shot_checkpoint 路径
      - distill_dataset: teacher distill dataset 路径
      - best_loss: one_shot_distill_best_loss（distill 训练 run）
      - eval_metrics: 来自 one_shot_distill_eval_metrics.json 的 overall 字段
      - sample_metrics: 来自 one_shot_sample_metrics.json 的 overall 字段
      - errors: 读取过程中的错误列表

    optional artifact 缺失时对应字段为 null。
    """
    result: dict = {}
    for raw in run_dirs:
        run_path = Path(raw).resolve()
        run_name = run_path.name
        entry: dict = {
            "run_dir": str(run_path),
            "stage": None,
            "checkpoint": None,
            "distill_dataset": None,
            "best_loss": None,
            "eval_metrics": None,
            "sample_metrics": None,
            "errors": [],
        }

        meta = _read_json(run_path / _RUN_META_FILE)
        if meta is None:
            entry["errors"].append(f"missing {_RUN_META_FILE}")
        else:
            entry["stage"] = meta.get("mode")
            entry["checkpoint"] = meta.get("one_shot_checkpoint")
            entry["distill_dataset"] = meta.get("distill_dataset")
            if meta.get("one_shot_distill_best_loss") is not None:
                entry["best_loss"] = meta["one_shot_distill_best_loss"]

        eval_metrics = _read_json(run_path / _EVAL_METRICS_FILE)
        entry["eval_metrics"] = _extract_overall(eval_metrics)

        sample_metrics = _read_json(run_path / _SAMPLE_METRICS_FILE)
        entry["sample_metrics"] = _extract_overall(sample_metrics)

        result[run_name] = entry

    return result


def print_summary(run_dirs: list[str]) -> None:
    """打印 JSON 汇总到 stdout。"""
    summary = summarize_runs(run_dirs)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# 指标合并：多个 run 按 best-of 合并（eval run 提供 eval 指标，distill run
# 提供 best_loss，sample run 提供 sample 指标）。同一指标跨 run 冲突时：
# lower-is-better → min，higher-is-better → max。
# ---------------------------------------------------------------------------

_LOWER_METRICS = frozenset({"mse", "sat", "slm_tv", "best_loss"})
_LOWER_COMPARE_METRICS = frozenset({"eval_mse", "eval_sat", "eval_slm_tv", "best_loss"})

_METRIC_SOURCE_MAP: dict[str, tuple[str, str]] = {
    "eval_mse": ("eval_metrics", "mse"),
    "eval_corr": ("eval_metrics", "corr"),
    "eval_dice": ("eval_metrics", "dice"),
    "eval_sat": ("eval_metrics", "sat"),
    "eval_slm_tv": ("eval_metrics", "slm_tv"),
    "eval_diversity_ratio": ("eval_metrics", "diversity_ratio"),
    "sample_corr": ("sample_metrics", "corr"),
    "sample_dice": ("sample_metrics", "dice"),
}


def _merge_candidate_metrics(candidates: dict) -> dict:
    """跨 run 合并 best-of 指标。"""
    merged_eval: dict[str, float] = {}
    merged_sample: dict[str, float] = {}
    best_losses: list[float] = []

    for entry in candidates.values():
        em = entry.get("eval_metrics")
        if em:
            for key, val in em.items():
                if val is None or key == "count":
                    continue
                prev = merged_eval.get(key)
                if prev is None:
                    merged_eval[key] = val
                elif key in _LOWER_METRICS:
                    merged_eval[key] = min(prev, val)
                else:
                    merged_eval[key] = max(prev, val)

        sm = entry.get("sample_metrics")
        if sm:
            for key, val in sm.items():
                if val is None or key == "count":
                    continue
                prev = merged_sample.get(key)
                if prev is None:
                    merged_sample[key] = val
                else:
                    merged_sample[key] = max(prev, val)

        bl = entry.get("best_loss")
        if bl is not None:
            best_losses.append(bl)

    return {
        "eval_metrics": merged_eval,
        "sample_metrics": merged_sample,
        "best_loss": min(best_losses) if best_losses else None,
    }


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def compare_runs(
    candidate_dirs: list[str],
    baseline: dict,
    tolerances: dict | None = None,
) -> dict:
    """比较候选 run 与 baseline 指标，返回逐项检查结果。

    baseline 格式:
        {"metrics": {"eval_mse": 0.020, ...}, "tolerances": {"eval_mse": 0.005, ...}}
    tolerances 参数会覆盖 baseline 内的 tolerances。

    多个 candidate dir 的指标按 best-of 合并后比较。

    返回:
        {"passed": bool, "candidate": str, "baseline_ref": str,
         "checks": [{"metric": str, "candidate": float|null, "baseline": float,
                      "tolerance": float, "passed": bool, "direction": str}, ...]}
    """
    tolerances = dict(tolerances) if tolerances is not None else {}
    baseline_metrics = baseline.get("metrics", {})
    baseline_tolerances = dict(baseline.get("tolerances", {}))
    baseline_tolerances.update(tolerances)

    candidates = summarize_runs(candidate_dirs)
    merged = _merge_candidate_metrics(candidates)

    all_checks: list[dict] = []

    for metric_key, baseline_value in baseline_metrics.items():
        if baseline_value is None:
            continue

        source_key, field = _METRIC_SOURCE_MAP.get(metric_key, (None, metric_key))
        if metric_key == "best_loss":
            candidate_value = merged.get("best_loss")
        elif source_key is not None:
            source = merged.get(source_key, {}) if merged else {}
            candidate_value = source.get(field) if source else None
        else:
            candidate_value = None

        tolerance = baseline_tolerances.get(metric_key, _DEFAULT_TOLERANCES.get(metric_key, 0))

        if candidate_value is None:
            all_checks.append({
                "metric": metric_key,
                "candidate": None,
                "baseline": baseline_value,
                "tolerance": tolerance,
                "passed": False,
                "direction": "missing",
            })
            continue

        if metric_key in _LOWER_COMPARE_METRICS:
            direction = "lower-is-better"
            ok = candidate_value <= baseline_value + tolerance
        else:
            direction = "higher-is-better"
            ok = candidate_value >= baseline_value - tolerance

        all_checks.append({
            "metric": metric_key,
            "candidate": candidate_value,
            "baseline": baseline_value,
            "tolerance": tolerance,
            "passed": ok,
            "direction": direction,
        })

    passed = len(all_checks) > 0 and all(c["passed"] for c in all_checks)

    return {
        "passed": passed,
        "candidate_runs": list(candidates.keys()),
        "baseline_ref": baseline.get("description", baseline.get("commit", "")),
        "checks": all_checks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m ldm_project.eval_reports")
    sub = parser.add_subparsers(dest="command")

    summary_parser = sub.add_parser("summary", help="print JSON summary for run dirs")
    summary_parser.add_argument("run_dirs", nargs="+", help="one or more outputs/runs/<run> directories")

    compare_parser = sub.add_parser("compare", help="compare run dirs against baseline JSON")
    compare_parser.add_argument("run_dirs", nargs="+", help="one or more outputs/runs/<run> directories")
    compare_parser.add_argument(
        "--baseline", required=True, help="path to baseline JSON (e.g. docs/ldm_migration/acceptance_baseline_c99f559.json)"
    )

    args = parser.parse_args()

    if args.command == "summary":
        print_summary(args.run_dirs)
    elif args.command == "compare":
        baseline = _read_json(Path(args.baseline))
        if baseline is None:
            print(f"error: cannot read baseline file: {args.baseline}")
            raise SystemExit(1)
        result = compare_runs(args.run_dirs, baseline)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        if not result["passed"]:
            raise SystemExit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
