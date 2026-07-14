#!/usr/bin/env python3
"""v38 formal CMOS-only manual review package.

This offline diagnostic consumes the v37.1 candidate report and renders only
the true final CMOS plane for human review. It deliberately excludes target
code and decoder intermediate planes from the primary review package.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch


from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output
from tools.diagnose_v34_code_coupling_recovery import _load_sample_bundle


FINAL_PLANE = "physical_camera_plane"


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v38_formal_cmos_manual_review_{timestamp}.json"


def _default_template_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v38_manual_review_template_{timestamp}.json"


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float().cpu()
    while value.ndim > 4:
        value = value[:, 0]
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4:
        raise ValueError(f"expected CMOS tensor with 2-4 dims, got {tuple(tensor.shape)}")
    mn = value.amin(dim=(-2, -1), keepdim=True)
    mx = value.amax(dim=(-2, -1), keepdim=True)
    return (value - mn) / (mx - mn).clamp_min(1.0e-6)


def _render_cmos_contact_sheet(
    *,
    values: torch.Tensor,
    label: str,
    output_dir: Path,
    filename: str,
    max_items: int = 20,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = _normalize(values)
    count = min(max_items, int(normalized.shape[0]))
    cols = min(5, max(1, count))
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.9))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[axis] for axis in axes]
    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx < count:
            ax.imshow(normalized[idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"sample {idx}", fontsize=8)
        else:
            ax.axis("off")
    fig.suptitle(f"v38 CMOS-only review: {label}", fontsize=10)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.as_posix()


def _render_comparison_sheet(records: list[dict[str, Any]], output_dir: Path) -> str | None:
    if not records:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    max_items = min(10, min(int(record["cmos_tensor"].shape[0]) for record in records))
    fig, axes = plt.subplots(max_items, len(records), figsize=(len(records) * 2.0, max_items * 1.6))
    if max_items == 1 and len(records) == 1:
        axes = [[axes]]
    elif max_items == 1:
        axes = [axes]
    elif len(records) == 1:
        axes = [[axis] for axis in axes]
    normalized = [(record["label"], _normalize(record["cmos_tensor"])) for record in records]
    for col, (label, values) in enumerate(normalized):
        axes[0][col].set_title(label, fontsize=8)
        for row in range(max_items):
            ax = axes[row][col]
            ax.imshow(values[row, 0], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("v38 CMOS-only A/B/C comparison | final CMOS is the only review plane", fontsize=10)
    fig.tight_layout()
    path = output_dir / "v38_cmos_only_probe_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.as_posix()


def _entries_from_v37_report(report: dict[str, Any]) -> list[tuple[str, Path]]:
    entries = []
    for record in report.get("runs", []):
        run = record.get("run")
        if not run:
            continue
        label = str(record.get("label") or Path(str(run)).name)
        entries.append((label, Path(run)))
    return entries


def _manual_review_items(*, label: str, run: Path, cmos: torch.Tensor) -> list[dict[str, Any]]:
    normalized = _normalize(cmos)
    items = []
    for index in range(int(normalized.shape[0])):
        items.append(
            {
                "label": label,
                "run": run.as_posix(),
                "sample_index": index,
                "final_human_gate_plane": FINAL_PLANE,
                "visible_digit_topology": None,
                "suggested_digit": None,
                "confidence": None,
                "notes": "",
            }
        )
    return items


def run_v38_formal_cmos_manual_review(
    *,
    v37_report: Path,
    output: Path | None = None,
    template_output: Path | None = None,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    template_path = template_output or _default_template_output()
    _require_report_output(output_path)
    _require_report_output(template_path)
    source = _load_json(v37_report)
    if source is None:
        raise FileNotFoundError(f"missing v37 report: {v37_report}")
    review_output_dir = review_dir or (output_path.parent / output_path.stem)
    entries = _entries_from_v37_report(source)
    records_for_render: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    manual_items: list[dict[str, Any]] = []
    for label, run in entries:
        try:
            bundle = _load_sample_bundle(run)
            cmos = bundle.get(FINAL_PLANE)
            if not torch.is_tensor(cmos):
                raise ValueError(f"sample bundle missing {FINAL_PLANE}")
            normalized = _normalize(cmos)
            image = _render_cmos_contact_sheet(
                values=cmos,
                label=label,
                output_dir=review_output_dir,
                filename=f"{label}_cmos_only_review.png",
            )
            run_record = {
                "label": label,
                "run": run.as_posix(),
                "status": "manual-review-required",
                "review_planes": [FINAL_PLANE],
                "final_human_gate_plane": FINAL_PLANE,
                "cmos_only_review_image": image,
                "physical_camera_plane_shape": list(cmos.shape),
                "sample_count": int(normalized.shape[0]),
            }
            runs.append(run_record)
            records_for_render.append({"label": label, "run": run, "cmos_tensor": cmos})
            manual_items.extend(_manual_review_items(label=label, run=run, cmos=cmos))
        except Exception as exc:
            errors.append({"label": label, "run": run.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
            runs.append(
                {
                    "label": label,
                    "run": run.as_posix(),
                    "status": "harness-invalid",
                    "reason": "CMOS-only review package generation failed",
                    "review_planes": [FINAL_PLANE],
                    "final_human_gate_plane": FINAL_PLANE,
                }
            )
    comparison = _render_comparison_sheet(records_for_render, review_output_dir)
    status = "harness-invalid" if errors or not runs else "manual-review-required"
    result = {
        "version": "v38",
        "objective": "formal_cmos_manual_review_candidate",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "status": status,
        "source_v37_report": v37_report.as_posix(),
        "source_v37_status": source.get("status"),
        "review_policy": "cmos-only",
        "final_human_gate_plane": FINAL_PLANE,
        "human_readability_required": True,
        "human_gate_note": "Only true final CMOS intensity / physical_camera_plane can satisfy manual review.",
        "review_dir": review_output_dir.as_posix(),
        "comparison_review_image": comparison,
        "manual_review_template": template_path.as_posix(),
        "runs": runs,
        "errors": errors,
    }
    template = {
        "version": "v38",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "status": "manual-review-required" if not errors else "harness-invalid",
        "source_report": output_path.as_posix(),
        "final_human_gate_plane": FINAL_PLANE,
        "manual_review_items": manual_items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    template_path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v37-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--template-output", type=Path)
    parser.add_argument("--review-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v38_formal_cmos_manual_review(
        v37_report=args.v37_report,
        output=args.output,
        template_output=args.template_output,
        review_dir=args.review_dir,
    )
    print(json.dumps({key: result[key] for key in ("status", "accepted_summary_written", "review_dir")}, indent=2))


if __name__ == "__main__":
    main()
