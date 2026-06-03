from __future__ import annotations

import json
from pathlib import Path

_RUN_META_FILE = "run_meta.json"
_EVAL_METRICS_FILE = "artifacts/one_shot_distill_eval_metrics.json"
_SAMPLE_METRICS_FILE = "artifacts/one_shot_sample_metrics.json"


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
