"""C3.3 P5 aggregation: per-seed results, baselines, ROI mapping sensitivity, energy ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as core

ROI_PERMUTATIONS = {"standard": (0, 1, 2, 3), "rotate_180": (3, 2, 1, 0)}


def load_runs(root: Path) -> list[dict]:
    records = []
    for directory in sorted(root.glob("train_*")):
        summary = directory / "training_summary.json"
        payload = directory / "phase.pt"
        if not (summary.exists() and payload.exists()):
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        data["directory"] = directory.name
        data["phase_path"] = str(payload)
        records.append(data)
    return records


def mapping_accuracy(scores: torch.Tensor, labels: torch.Tensor, permutation) -> float:
    mapping = torch.tensor(permutation)
    predicted = scores.argmax(1)
    remapped = mapping[predicted]
    return float((remapped == labels).double().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=250)
    args = parser.parse_args()

    runs = load_runs(args.root)
    if not runs:
        raise SystemExit("no completed runs")
    images_all, labels_all = core.load_mnist()
    first = torch.load(runs[0]["phase_path"], map_location="cpu", weights_only=False)
    holdout = first["holdout_rows"].clone()
    validation = first["validation_rows"].clone()
    holdout_images = images_all[holdout].to(args.device)
    holdout_labels = labels_all[holdout].to(args.device)
    validation_images = images_all[validation].to(args.device)
    validation_labels = labels_all[validation].to(args.device)

    baselines = {}
    for defocus in (0.0, 52000.0):
        configuration = core.build_configuration(defocus, args.device)
        zeros = torch.zeros(1, 1, core.GRID, core.GRID, device=args.device)
        generator = torch.Generator().manual_seed(1)
        random_phase = (torch.rand((1, 1, core.GRID, core.GRID), generator=generator)
                        .to(args.device) * 6.283185307179586)
        entry = {}
        for name, phase in (("no_screen", zeros), ("random_phase", random_phase)):
            evaluation = core.evaluate(holdout_images, holdout_labels, configuration, phase, args.batch)
            entry[name] = {"accuracy": evaluation["metrics"]["accuracy"],
                           "macro_f1": evaluation["metrics"]["macro_f1"],
                           "mean_detected_energy": evaluation["metrics"]["mean_detected_energy"]}
        baselines[f"defocus_{defocus:g}"] = entry

    per_run = []
    for run in runs:
        configuration = core.build_configuration(run["defocus_um"], args.device)
        payload = torch.load(run["phase_path"], map_location="cpu", weights_only=False)
        phase = payload.get("effective", payload["phase"]).to(args.device)
        raw = payload["phase"].to(args.device)
        holdout_eval = core.evaluate(holdout_images, holdout_labels, configuration, phase, args.batch)
        validation_eval = core.evaluate(validation_images, validation_labels, configuration, phase, args.batch)
        per_run.append({
            "directory": run["directory"],
            "init": run["init"],
            "seed": run["seed"],
            "defocus_um": run["defocus_um"],
            "smooth_sigma": run.get("smooth_sigma", 0.0),
            "best_epoch": run["best_epoch"],
            "validation_accuracy": validation_eval["metrics"]["accuracy"],
            "validation_macro_f1": validation_eval["metrics"]["macro_f1"],
            "holdout_accuracy": holdout_eval["metrics"]["accuracy"],
            "holdout_macro_f1": holdout_eval["metrics"]["macro_f1"],
            "holdout_recall": holdout_eval["metrics"]["recall"],
            "holdout_confusion": holdout_eval["metrics"]["confusion_4x4"],
            "correct_roi_fraction": holdout_eval["metrics"]["correct_roi_fraction"],
            "max_wrong_roi_fraction": holdout_eval["metrics"]["max_wrong_roi_fraction"],
            "outside_fraction": holdout_eval["metrics"]["outside_fraction"],
            "mean_detected_energy": holdout_eval["metrics"]["mean_detected_energy"],
            "roi_energy_matrix": holdout_eval["metrics"]["roi_energy_matrix"],
            "mapping_accuracy": {name: mapping_accuracy(holdout_eval["scores"], holdout_labels.cpu(), perm)
                                 for name, perm in ROI_PERMUTATIONS.items()},
            "raw_holdout_accuracy": core.evaluate(holdout_images, holdout_labels, configuration, raw, args.batch)["metrics"]["accuracy"],
            "phase_tv": float(core.phase_tv(phase)),
            "phase_l2": float(phase.norm()),
            "phase_fingerprint": core.tensor_sha256(phase),
        })

    groups: dict[str, list[dict]] = {}
    for record in per_run:
        key = f"dz{int(record['defocus_um'])}_sigma{record['smooth_sigma']}_{record['init']}"
        groups.setdefault(key, []).append(record)
    summary = {}
    for key, items in groups.items():
        holdout_values = torch.tensor([item["holdout_accuracy"] for item in items])
        summary[key] = {
            "count": len(items),
            "holdout_accuracy_mean": float(holdout_values.mean()),
            "holdout_accuracy_sd": float(holdout_values.std(unbiased=True)) if len(items) > 1 else 0.0,
            "validation_accuracy_mean": float(torch.tensor([i["validation_accuracy"] for i in items]).mean()),
            "macro_f1_mean": float(torch.tensor([i["holdout_macro_f1"] for i in items]).mean()),
            "seeds": [item["seed"] for item in items],
            "accuracy_range": [float(holdout_values.min()), float(holdout_values.max())],
        }

    report = {"identity": "HW207_C33_P5_EVALUATION", "population": {
        "holdout_rows": int(len(holdout)), "validation_rows": int(len(validation)),
        "note": "phase_holdout is disjoint from this run's training split only; it is not globally unseen"},
        "baselines": baselines, "per_run": per_run, "groups": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    core.write_json(args.out, report)
    print(json.dumps({"runs": len(per_run), "groups": summary,
                      "baselines": baselines}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()