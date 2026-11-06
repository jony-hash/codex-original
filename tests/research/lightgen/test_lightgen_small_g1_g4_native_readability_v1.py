from __future__ import annotations

import json
import argparse
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.research.lightgen import execute_lightgen_small_g1_g4_native_readability_v1 as task


def _cohorts() -> dict[str, dict[str, torch.Tensor | int | str]]:
    labels = torch.arange(10).repeat_interleave(1_000)
    rows = torch.arange(10_000)
    return {
        method: {
            "requested_labels": labels.clone(), "proposal_ids": rows.clone(),
            "component_ids": torch.zeros(10_000, dtype=torch.long), "stream": 0,
        }
        for method in task.METHODS
    }


def _review_fixture(run_dir: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cohorts: dict[str, dict[str, torch.Tensor | int | str]] = {}
    selected: dict[str, dict[str, torch.Tensor]] = {}
    manifest_items: list[dict[str, int | str]] = []
    for method_index, method in enumerate(task.METHODS):
        rows = []
        images = []
        for label in task.CLASSES:
            for offset in range(task.SELECT_PER_CLASS):
                row = label * 1_000 + offset
                rows.append(row)
                proposal_id = row
                manifest_items.append({
                    "method": method, "requested_label": label, "stream": 0,
                    "cohort_row": row, "proposal_id": proposal_id, "component_id": 0,
                })
                images.append(torch.full((1, 200, 200), 0.2 + method_index * 0.1 + label / 100.0
                                         + offset / 1_000.0, dtype=torch.float32))
        selected[method] = {
            "cohort_rows": torch.tensor(rows), "native200": torch.stack(images),
            "view28": torch.zeros((100, 1, 28, 28)),
        }
        cohorts[method] = {
            "ed_predictions": torch.zeros(10_000, dtype=torch.long),
            "ec_predictions": torch.zeros(10_000, dtype=torch.long),
        }
    identity = {
        "task_id": task.TASK_ID, "run_id": "test-run", "source_run_id": "source-C",
        "source_step": 45000, "source_quality_status_preserved": "VALID_FAILED",
    }
    selection = {
        "schema": "lightgen_small_g1_g4_native_readability_selection.v1",
        "task_id": task.TASK_ID, "run_id": "test-run", "source_run_id": "source-C",
        "source_step": task.SOURCE_STEP, "stream": 0, "selection_seeds": task.SELECTION_SEEDS,
        "classifier_brightness_or_appearance_used_for_selection": False,
        "items": manifest_items,
    }
    rows = torch.arange(20)
    labels = torch.arange(10).repeat_interleave(2)
    examples = {
        "rows": rows, "labels": labels,
        "targets28": torch.zeros((20, 1, 28, 28), dtype=torch.float32),
    }
    key, summary = task._write_blind_review(
        run_dir, identity, selection, selected, cohorts, examples, None, [], 0,
    )
    machine = {
        "task_id": task.TASK_ID, "run_id": "test-run", "status": "AWAITING_HUMAN_RATINGS",
        "measurement_complete": True,
        "source": {"source_status_preserved": "VALID_FAILED", "run_id": "source-C", "global_step": 45000},
        "quality_gate": {"historical_system_status": "VALID_FAILED", "status_changed": False},
        "decision": {"result": "AWAITING_HUMAN_RATINGS", "human_rating_received": False},
        "artifacts": {}, "execution": {"completed_blocks": ["preflight_20_samples"],
                                        "optical_forward_sample_count": 220}, "output": {},
    }
    (run_dir / "machine_report.json").write_text(json.dumps(machine), encoding="utf-8")
    (run_dir / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    with (run_dir / "selected_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "requested_label", "proposal_id", "stream",
                                                   "E_D_prediction_saved", "E_C_prediction_saved"])
        writer.writeheader()
        for card in key["cards"]:
            if card["kind"] == "generated":
                writer.writerow({"method": card["method"], "requested_label": card["requested_label"],
                                 "proposal_id": card["proposal_id"], "stream": 0,
                                 "E_D_prediction_saved": card["requested_label"],
                                 "E_C_prediction_saved": card["requested_label"]})
    submission = _complete_ratings(key)
    return key, submission, {"machine": machine, "summary": summary}


def _complete_ratings(key: dict[str, object]) -> dict[str, object]:
    cards = key["cards"]
    return {
        "schema": task.RATINGS_SCHEMA, "task_id": task.TASK_ID, "review_id": key["review_id"],
        "cards": [
            {"card_id": card["card_id"], "rating": card.get("requested_label")
             if card["kind"] in ("generated", "duplicate") else card["target_label"]}
            for card in cards
        ],
    }


def test_independent_readout_matches_production_and_rejects_wrong_order() -> None:
    rng = np.random.default_rng(17)
    raw = rng.uniform(0.0, 2.2, size=(200, 200)).astype(np.float32)
    gain = 0.73
    native, pooled = task._independent_readout(raw, gain)
    production_native = torch.tensor(raw, dtype=torch.float32).mul(gain).clamp(0, 1)
    production_view = F.adaptive_avg_pool2d(production_native[None, None], (28, 28))[0, 0]
    assert np.allclose(native, production_native.numpy(), rtol=0, atol=2e-7)
    assert np.allclose(pooled, production_view.numpy(), rtol=2e-6, atol=5e-8)

    wrong_pool_before_clamp = np.clip(task._numpy_adaptive_pool_28(raw * gain), 0.0, 1.0)
    assert not np.allclose(pooled, wrong_pool_before_clamp, rtol=2e-6, atol=5e-8)
    clamp_then_scale = np.clip(raw, 0, 1) * 0.5
    scale_then_clamp = np.clip(raw * 0.5, 0, 1)
    assert not np.array_equal(clamp_then_scale, scale_then_clamp)


def test_selection_is_fixed_class_balanced_and_rejects_wrong_row() -> None:
    cohorts = _cohorts()
    identity = {"run_id": "fixed", "source_run_id": "C"}
    first = task._build_selection_manifest(cohorts, identity)
    second = task._build_selection_manifest(cohorts, identity)
    assert first == second
    task._validate_selection_manifest(first, cohorts, identity)
    assert len(first["items"]) == 200
    damaged = json.loads(json.dumps(first))
    damaged["items"][0]["cohort_row"] += 1
    with pytest.raises(RuntimeError, match="row, class, proposal"):
        task._validate_selection_manifest(damaged, cohorts, identity)


def test_first_start_preserves_only_the_autodl_prelaunch_telemetry(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_01"
    run_dir.mkdir()
    telemetry = run_dir / "resource_telemetry.csv"
    original = (task.WATCHER_TELEMETRY_HEADER + "\n2026-10-22T00:00:00Z,run,STARTING,,,,,,,,,,,,\n").encode()
    telemetry.write_bytes(original)
    identity = {"task_id": task.TASK_ID, "run_id": "fixed"}
    loaded, resumed = task._prepare_or_resume(run_dir, identity, None)
    assert loaded == identity and resumed is False
    assert telemetry.read_bytes() == original
    assert json.loads((run_dir / "run_identity.json").read_text()) == identity

    for name, files in (
        ("extra", {"resource_telemetry.csv": original, "unexpected.txt": b"evidence"}),
        ("bad_header", {"resource_telemetry.csv": b"not,the,watcher,header\n"}),
    ):
        damaged_dir = tmp_path / name
        damaged_dir.mkdir()
        for filename, content in files.items():
            (damaged_dir / filename).write_bytes(content)
        with pytest.raises(RuntimeError):
            task._prepare_or_resume(damaged_dir, identity, None)
        assert not (damaged_dir / "run_identity.json").exists()


def test_production_review_page_and_rating_consumer_validate_all_240_cards(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_01"
    key, submission, _details = _review_fixture(run_dir)
    page = (run_dir / "blind_review.html").read_text(encoding="utf-8")
    assert len(key["cards"]) == 240
    assert "requested_label" not in page and "proposal_id" not in page and "method" not in page
    assert "localStorage.setItem" in page and "第" in page
    ratings_path = run_dir / "ratings.json"
    ratings_path.write_text(json.dumps(submission), encoding="utf-8")
    validated = task._validate_ratings(run_dir, ratings_path)
    assert validated["duplicate_consistency_count"] == 20
    assert validated["anchor_correct_count"] == 20
    assert len([card for card in validated["cards"] if card["kind"] == "generated"]) == 200

    # The submitted labels never affect the 200-card primary denominator.
    metrics = task._method_metrics(validated["cards"], validated["rating_by_id"], "G1")
    assert metrics["unique_generated_denominator"] == 100
    assert metrics["equal_class_requested_digit_rate"] == 1.0


def test_rating_consumer_rejects_wrong_ids_mapping_missing_duplicate_and_label(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_01"
    key, submission, _details = _review_fixture(run_dir)
    ratings_path = run_dir / "ratings.json"

    for mutation in ("wrong_task", "wrong_review", "missing", "duplicate", "illegal"):
        damaged = json.loads(json.dumps(submission))
        if mutation == "wrong_task":
            damaged["task_id"] = "OTHER_TASK"
        elif mutation == "wrong_review":
            damaged["review_id"] = "OTHER_REVIEW"
        elif mutation == "missing":
            damaged["cards"].pop()
        elif mutation == "duplicate":
            damaged["cards"][-1]["card_id"] = damaged["cards"][0]["card_id"]
        else:
            damaged["cards"][0]["rating"] = 10
        ratings_path.write_text(json.dumps(damaged), encoding="utf-8")
        with pytest.raises((RuntimeError, ValueError)):
            task._validate_ratings(run_dir, ratings_path)

    page_path = run_dir / "blind_review.html"
    page = page_path.read_text(encoding="utf-8")
    page_path.write_text(page.replace('"card_id": "R0001"', '"card_id": "R0002"', 1), encoding="utf-8")
    ratings_path.write_text(json.dumps(submission), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mapping"):
        task._validate_ratings(run_dir, ratings_path)
    page_path.write_text(page, encoding="utf-8")
    csv_path = run_dir / "selected_per_image.csv"
    original_csv = csv_path.read_text(encoding="utf-8")
    lines = original_csv.splitlines()
    csv_path.write_text("\n".join(lines[:-1] + [lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="classifier rows"):
        task._validate_ratings(run_dir, ratings_path)
    csv_path.write_text(original_csv, encoding="utf-8")


def test_transferred_run_finalizes_production_report_without_optical_loading(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "autodl_backup_attempt_01"
    _key, submission, _details = _review_fixture(run_dir)
    source = tmp_path / "source.pt"
    source.write_bytes(b"not-loaded-in-rating-finalization")
    baseline = tmp_path / "baseline"
    replication = tmp_path / "replication"
    baseline.mkdir()
    replication.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixed configuration", encoding="utf-8")
    config = {"source": {"checkpoint": str(source), "baseline_evidence_dir": str(baseline),
                         "replication_evidence_dir": str(replication)},
              "output": {"experiment_root": str(tmp_path / "attempt_01"),
                         "formal_result_budget_bytes": 134217728},
              "human_review": {"minimum_duplicate_consistency": 18, "minimum_anchor_correct": 18}}
    identity_path = run_dir / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    identity.update(config=config, config_path="/root/autodl-tmp/codex-original/config.yaml",
                    source_checkpoint="/cloud/source.pt", baseline_evidence_dir="/cloud/baseline",
                    replication_evidence_dir="/cloud/replication")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    transfer = {"status": "COMPLETE", "destination": str(run_dir),
                "remote_run_dir": "/cloud/attempt_01",
                "files": [{"status": "COPIED", "scp_exit_code": 0}]}
    (run_dir / "bundle_transfer_receipt.json").write_text(json.dumps(transfer), encoding="utf-8")
    (run_dir / "bundle_extraction_receipt.json").write_text(
        json.dumps({"status": "VERIFIED", "size_inventory_match": True}), encoding="utf-8")
    ratings = tmp_path / "ratings.json"
    ratings.write_text(json.dumps(submission), encoding="utf-8")
    monkeypatch.setattr(task, "_read_config", lambda _path: (config, {}))
    def reject_optics(*args, **kwargs):
        raise AssertionError("rating finalization must not load optical inputs")
    monkeypatch.setattr(task, "_load_inputs", reject_optics)
    args = argparse.Namespace(config=config_path, output_dir=run_dir, source_checkpoint=source,
                              baseline_evidence_dir=baseline, replication_dir=replication,
                              ratings_file=ratings, preflight_only=False, resume_state=None)
    wrong = argparse.Namespace(**vars(args))
    wrong.source_checkpoint = tmp_path / "wrong.pt"
    with pytest.raises((ValueError, RuntimeError)):
        task._run(wrong)
    task._run(args)
    result = json.loads((run_dir / "final_report.json").read_text())
    assert result["status"] == "COMPLETE_DIAGNOSTIC"
    assert result["decision"]["human_rating_received"] is True
    assert result["decision"]["result"] == "HUMAN_READABILITY_INDEPENDENT_EFFECT_UNCLEAR"
    assert result["human_review"]["primary_unique_generated_count"] == 200
    assert result["human_review"]["quality_checks"]["duplicate_consistency"]["count"] == 20
    assert json.loads((run_dir / "progress.json").read_text())["optical_forward_samples_total"] == 220
    assert json.loads(identity_path.read_text()) == identity
    with pytest.raises(FileExistsError, match="already finalized"):
        task._run(args)


def test_failed_repeat_quality_blocks_even_a_maximal_primary_gain(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt_01"
    key, submission, _details = _review_fixture(run_dir)
    mapping = {card["card_id"]: card for card in key["cards"]}
    spoiled_repeats = 0
    for item in submission["cards"]:
        card = mapping[item["card_id"]]
        if card.get("method") == "G1":
            item["rating"] = "unreadable"
        elif card["kind"] == "duplicate" and spoiled_repeats < 3:
            item["rating"] = "unreadable"
            spoiled_repeats += 1
    ratings = tmp_path / "ratings.json"
    ratings.write_text(json.dumps(submission), encoding="utf-8")
    config = {"human_review": {"minimum_duplicate_consistency": 18, "minimum_anchor_correct": 18},
              "output": {"formal_result_budget_bytes": 134217728}}
    task._finalize_ratings(run_dir, ratings, config)
    report = json.loads((run_dir / "final_report.json").read_text())
    assert report["human_review"]["G4_minus_G1_primary_effect"]["ci_95"] == [1.0, 1.0]
    assert report["human_review"]["quality_checks"]["duplicate_consistency"]["count"] == 17
    assert report["status"] == report["decision"]["result"] == "REVIEW_INCONCLUSIVE"
