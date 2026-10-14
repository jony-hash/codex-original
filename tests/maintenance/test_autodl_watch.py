from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.maintenance import autodl_watch


def test_watch_records_resources_and_stops_when_the_bound_process_exits(tmp_path: Path, monkeypatch) -> None:
    control = tmp_path / "control"
    run = tmp_path / "run"
    run.mkdir()
    (run / "progress.json").write_text(json.dumps({"status": "RUNNING", "current_seed": 2026101201, "current_arm": "B", "current_task_local_update": 100, "current_optimizer_step": 4800}), encoding="utf-8")
    alive = iter((True, True, True, False, False, False))
    monkeypatch.setattr(autodl_watch, "_gpu_sample", lambda: ("55", "12000", "32760"))
    monkeypatch.setattr(autodl_watch, "_host_available_kib", lambda: 8_000_000)
    monkeypatch.setattr(autodl_watch, "_training_process", lambda _pid: (456, 900_000))
    monkeypatch.setattr(autodl_watch, "_process_alive", lambda _pid: next(alive))
    monkeypatch.setattr(autodl_watch.time, "sleep", lambda _seconds: None)

    result = autodl_watch.watch("r" * 32, control, run, 123, interval_seconds=60)

    rows = list(csv.DictReader((control / "telemetry.csv").open(encoding="utf-8")))
    assert result["state"] == "STOPPED"
    assert len(rows) == 2
    assert rows[0]["run_id"] == "r" * 32
    assert rows[0]["training_pid"] == "456"
    assert rows[0]["training_rss_kib"] == "900000"
    assert rows[0]["gpu_utilization_percent"] == "55"
    assert rows[0]["gpu_memory_used_mib"] == "12000"
    assert rows[0]["task_local_update"] == "100"
    assert json.loads((control / "telemetry_status.json").read_text(encoding="utf-8"))["state"] == "STOPPED"


def test_collector_error_is_separate_from_training_progress(tmp_path: Path, monkeypatch) -> None:
    control = tmp_path / "control"
    run = tmp_path / "run"
    run.mkdir()
    (run / "progress.json").write_text(json.dumps({"status": "FAILED"}), encoding="utf-8")
    monkeypatch.setattr(autodl_watch, "_gpu_sample", lambda: (_ for _ in ()).throw(RuntimeError("gpu query failed")))
    monkeypatch.setattr(autodl_watch, "_host_available_kib", lambda: 8_000_000)
    monkeypatch.setattr(autodl_watch, "_training_process", lambda _pid: (None, None))
    monkeypatch.setattr(autodl_watch, "_process_alive", lambda _pid: False)

    result = autodl_watch.watch("a" * 32, control, run, 123)

    row = next(csv.DictReader((control / "telemetry.csv").open(encoding="utf-8")))
    assert result["collector_failures"] == 1
    assert row["progress_state"] == "FAILED"
    assert row["run_state"] == "PROCESS_EXITED"
    assert "gpu:RuntimeError" in row["sample_error"]
