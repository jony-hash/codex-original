"""Independently recompute the saved C3.4 frozen-phase decisions and trigger."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _predictions(scores: np.ndarray, positive_energy: np.ndarray) -> np.ndarray:
    maximum = scores.max(axis=1, keepdims=True)
    tied = np.isclose(scores, maximum, rtol=1e-12, atol=0).sum(axis=1) != 1
    predictions = scores.argmax(axis=1).astype(np.int64)
    predictions[(positive_energy <= 0) | tied] = -1
    return predictions


def _metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray,
             detected: np.ndarray, fraction_denominator: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((4, 5), dtype=np.int64)
    for truth, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        if truth not in range(4) or predicted not in (-1, 0, 1, 2, 3):
            raise ValueError("saved labels or predictions are outside the four-class contract")
        confusion[truth, 4 if predicted == -1 else predicted] += 1
    true_positive = np.diag(confusion)[:4].astype(np.float64)
    precision = true_positive / np.maximum(confusion[:, :4].sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    correct = (predictions == labels) & (predictions >= 0)
    correct_fraction = scores[np.arange(len(labels)), labels] / np.maximum(fraction_denominator, 1e-30)
    wrong = scores.copy()
    wrong[np.arange(len(labels)), labels] = -np.inf
    strongest_wrong = wrong.max(axis=1) / np.maximum(fraction_denominator, 1e-30)
    decided = predictions >= 0
    return {
        "sample_count": int(len(labels)),
        "accuracy_all_samples": float(correct.mean()),
        "macro_f1_classes_0_to_3": float(f1.mean()),
        "coverage": float(decided.mean()),
        "accuracy_when_decided": float(correct[decided].mean()) if decided.any() else 0.0,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "confusion_4x5_with_abstention_column": confusion.tolist(),
        "correct_roi_energy_fraction_mean": float(correct_fraction.mean()),
        "strongest_wrong_roi_energy_fraction_mean": float(strongest_wrong.mean()),
        "correct_minus_strongest_wrong_energy_fraction_mean": float((correct_fraction - strongest_wrong).mean()),
        "mean_detected_energy_um2": float(detected.mean()),
        "mean_roi_energy_um2": scores.mean(axis=0).tolist(),
    }


def _mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict[str, Any]:
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    discordant = a_only + b_only
    p_value = (sum(math.comb(discordant, k) for k in range(a_only, discordant + 1))
               / (1 << discordant)) if discordant else 1.0
    return {"A_correct_B_wrong": a_only, "A_wrong_B_correct": b_only,
            "discordant_pairs": discordant,
            "one_sided_exact_binomial_pvalue_B_worse": float(p_value),
            "trigger": bool(p_value < 0.05)}


def audit_p2(run_dir: Path, c33_root: Path) -> dict[str, Any]:
    p2_dir = run_dir / "p2_frozen_inference"
    comparison = json.loads((p2_dir / "frozen_phase_comparison.json").read_text(encoding="utf-8"))
    if comparison.get("identity") != "C3.4_FROZEN_PHASE_PAIRED_INFERENCE":
        raise ValueError("unexpected saved P2 report identity")
    selected_run = comparison["phase_source"]["run"]
    c33_checkpoint = torch.load(c33_root / selected_run / "phase.pt", map_location="cpu", weights_only=False)
    expected_rows = c33_checkpoint["holdout_rows"].detach().cpu().numpy().astype(np.int64)
    a = np.load(p2_dir / "model_a_fixed_roi_scores.npz")
    b = np.load(p2_dir / "model_b_aperture_2mm_scores.npz")
    a_rows, b_rows = a["original_indices"].astype(np.int64), b["original_indices"].astype(np.int64)
    a_labels, b_labels = a["labels"].astype(np.int64), b["labels"].astype(np.int64)
    if not np.array_equal(a_rows, expected_rows) or not np.array_equal(b_rows, expected_rows):
        raise ValueError("P2 row order differs from the selected C3.3 checkpoint holdout")
    if not np.array_equal(a_labels, b_labels) or not np.array_equal(a_rows, b_rows):
        raise ValueError("Model A and B P2 rows/labels are not paired")
    if len(a_rows) != 2000 or not np.array_equal(np.bincount(a_labels, minlength=4), [500] * 4):
        raise ValueError("P2 population does not contain 2,000 balanced holdout rows")

    a_scores = a["roi_scores"].astype(np.float64)
    b_scores = b["roi_scores"].astype(np.float64)
    a_detected = a["detected_energy"].astype(np.float64)
    b_detected = b["full_sensor_energy"].astype(np.float64)
    b_crop = b["crop_energy"].astype(np.float64)
    a_predictions = _predictions(a_scores, a_detected)
    b_predictions = _predictions(b_scores, b_crop)
    if not np.array_equal(a_predictions, a["predictions"]) or not np.array_equal(b_predictions, b["predictions"]):
        raise ValueError("saved P2 predictions do not follow fixed ROI argmax and abstention rules")
    a_metrics = _metrics(a_labels, a_predictions, a_scores, a_detected, a_detected)
    b_metrics = _metrics(b_labels, b_predictions, b_scores, b_detected, b_crop)
    paired = _mcnemar((a_predictions == a_labels) & (a_predictions >= 0),
                      (b_predictions == b_labels) & (b_predictions >= 0))
    report_a = comparison["model_a_c33"]["metrics"]
    report_b = comparison["model_b_estimated_geometry"]["aperture_assumptions_mm"]["2.0"]
    report_pair = comparison["paired_adaptation_decision"]
    checks = {
        "model_a_accuracy_matches_report": math.isclose(a_metrics["accuracy_all_samples"], report_a["accuracy_all_samples"], abs_tol=1e-12),
        "model_a_confusion_matches_report": a_metrics["confusion_4x5_with_abstention_column"] == report_a["confusion_4x5_with_abstention_column"],
        "model_b_accuracy_matches_report": math.isclose(b_metrics["accuracy_all_samples"], report_b["accuracy_all_samples"], abs_tol=1e-12),
        "model_b_confusion_matches_report": b_metrics["confusion_4x5_with_abstention_column"] == report_b["confusion_4x5_with_abstention_column"],
        "paired_discordances_match_report": paired["A_correct_B_wrong"] == report_pair["A_correct_B_wrong"] and paired["A_wrong_B_correct"] == report_pair["A_wrong_B_correct"],
        "paired_p_value_matches_report": math.isclose(paired["one_sided_exact_binomial_pvalue_B_worse"], report_pair["one_sided_exact_binomial_pvalue_B_worse"], rel_tol=1e-12, abs_tol=1e-300),
        "trigger_matches_report": paired["trigger"] == report_pair["trigger"],
    }
    result = {
        "identity": "C3.4_P2_INDEPENDENT_DECISION_AUDIT",
        "population_rows": len(a_rows),
        "population_class_counts": np.bincount(a_labels, minlength=4).tolist(),
        "model_a_recomputed_metrics": a_metrics,
        "model_b_2mm_recomputed_metrics": b_metrics,
        "paired_decision_recomputed": paired,
        "checks": checks,
        "independent_audit_pass": all(checks.values()),
        "scope": "saved numerical-model outputs; historical C3.3 holdout is not an independent project-level test",
    }
    (p2_dir / "p2_independent_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not result["independent_audit_pass"]:
        raise RuntimeError("independent C3.4 P2 decision audit failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--c33-root", type=Path, required=True)
    args = parser.parse_args()
    result = audit_p2(args.run_dir.resolve(), args.c33_root.resolve())
    print(json.dumps({"identity": result["identity"], "pass": result["independent_audit_pass"],
                      "run_dir": str(args.run_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
