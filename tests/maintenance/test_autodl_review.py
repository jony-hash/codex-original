from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.maintenance import autodl_review as review


def _review_fixture(root: Path) -> Path:
    evidence = root / "evidence"
    (evidence / "training_history").mkdir(parents=True)
    (evidence / "checkpoints").mkdir()
    (evidence / "readouts" / "A4700").mkdir(parents=True)
    seeds = [2026101201, 2026101202, 2026101203]
    scenarios = ["A4700", *[f"seed_{seed}_{arm}_step4702" for seed in seeds for arm in ("A", "B")]]
    means = {"A4700": 0.5}
    for index, seed in enumerate(seeds):
        means[f"seed_{seed}_A_step4702"] = 0.4 + index * 0.001
        means[f"seed_{seed}_B_step4702"] = 0.3 + index * 0.001
    paired_path = evidence / "paired_rows.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("scenario", "position", "label", "q_fg"))
        writer.writeheader()
        for scenario in scenarios:
            for label in range(10):
                value = means.get(scenario, 0.5)
                for repeat in range(100):
                    writer.writerow({"scenario": scenario, "position": label * 100 + repeat, "label": label, "q_fg": value})
    class_path = evidence / "dev_class_summary.csv"
    with class_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("scenario", "label", "mean_q_fg", "mean_raw_cmos_integral"))
        writer.writeheader()
        for scenario in scenarios:
            for label in range(10):
                writer.writerow({"scenario": scenario, "label": label, "mean_q_fg": means.get(scenario, 0.5) + label / 1000, "mean_raw_cmos_integral": 2e-8 + label / 1e9})
    seed_decisions = {}
    for seed in seeds:
        a = means[f"seed_{seed}_A_step4702"]
        b = means[f"seed_{seed}_B_step4702"]
        seed_decisions[str(seed)] = {"A": a, "B": b, "Q0": means["A4700"]}
        for arm in ("A", "B"):
            records = [{"task_local_update": step, "q_fg": means[f"seed_{seed}_{arm}_step4702"], "raw_mse": 0.01 / step} for step in (1, 2)]
            (evidence / "training_history" / f"seed_{seed}_arm_{arm.lower()}.json").write_text(json.dumps({"seed": seed, "arm": arm, "records": records}), encoding="utf-8")
            (evidence / "checkpoints" / f"seed_{seed}_arm_{arm.lower()}_step4702.pt").write_bytes(b"terminal")
            (evidence / "checkpoints" / f"seed_{seed}_arm_{arm.lower()}_step4701.pt").write_bytes(b"intermediate")
    for name in (
        "validation.json", "completed.json", "closeout.md", "fixed_parameters.json", "batch_orders.json",
        "execution_runtime.json", "preflight.json", "preflight_trials.json", "progress.json", "consolidation.json",
        "conditional_sample_rows.csv", "conditional_sample_class_summary.csv", "conditional_sample_summary.json",
        "conditional_sample_bank.pt", "fitted_joint_prior.pt",
    ):
        (evidence / name).write_text("{}\n", encoding="utf-8")
    (evidence / "input_population.json").write_text(json.dumps({"source_checkpoint": "D:\\codex-original\\outputs\\start\\runs\\attempt_01\\arm_a_step4700.pt"}), encoding="utf-8")
    report = {
        "status": "VALID_COMPLETED", "training_seeds": seeds, "source_checkpoint_step": 4700,
        "updates_per_arm_per_seed": 2, "decision": {"shared_start_mean_q_fg": means["A4700"], "seed_decisions": seed_decisions},
    }
    (evidence / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (evidence / "readouts" / "A4700" / "chunk_000.pt").write_bytes(b"raw-audit")
    return evidence


def test_review_package_contains_terminal_evidence_but_omits_intermediate_and_raw(tmp_path: Path) -> None:
    evidence = _review_fixture(tmp_path)
    output = tmp_path / "review"
    package = review._package(evidence, output, "review")
    manifest = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))

    assert package["status"] == "PACKAGED"
    assert package["raw_pixel_audit_complete_in_bundle"] is False
    assert manifest["required_local_dependencies"] == ["outputs/start/runs/attempt_01/arm_a_step4700.pt"]
    assert any("step4702.pt" in path for path in manifest["files"])
    assert not any("step4701.pt" in path or "readouts/" in path for path in manifest["files"])
    figures = output / "review_bundle" / "figures"
    assert len(list(figures.glob("*.png"))) == 3
    assert package["review_png_validation"]["total_png_bytes"] <= 2 * 1024 * 1024


def test_audit_profile_requires_and_includes_raw_readout_chunks(tmp_path: Path) -> None:
    evidence = _review_fixture(tmp_path)
    selected, _dependencies = review._review_files(evidence, "audit")

    assert any(path.relative_to(evidence).as_posix() == "readouts/A4700/chunk_000.pt" for path in selected)


def test_cloud_posix_source_checkpoint_maps_to_repository_relative_path_on_windows() -> None:
    path = "/root/autodl-tmp/codex-original/outputs/start/runs/attempt_01/arm_a_step4700.pt"

    assert review._repo_relative_source(path) == "outputs/start/runs/attempt_01/arm_a_step4700.pt"


@pytest.mark.parametrize("value", ["../report.json", "outputs/a/*.pt", "review bundle/file.json", "/etc/passwd"])
def test_manifest_rejects_traversal_wildcards_spaces_and_absolute_paths(value: str) -> None:
    with pytest.raises(review.ReviewError):
        review._validate_relative(value)


def test_review_manifest_rejects_missing_terminal_checkpoint(tmp_path: Path) -> None:
    evidence = _review_fixture(tmp_path)
    (evidence / "checkpoints" / "seed_2026101201_arm_a_step4702.pt").unlink()

    with pytest.raises(review.ReviewError, match="review bundle is incomplete"):
        review._review_files(evidence, "review")
