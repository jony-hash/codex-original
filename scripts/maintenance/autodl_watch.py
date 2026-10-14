"""Sample AutoDL run resources without changing or restarting the training process."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


FIELDS = (
    "timestamp_utc", "run_id", "run_state", "progress_state", "training_pid", "training_rss_kib",
    "gpu_utilization_percent", "gpu_memory_used_mib", "gpu_memory_total_mib", "host_mem_available_kib",
    "seed", "arm", "task_local_update", "optimizer_step", "sample_error",
)


def _proc_snapshot() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            status = (proc / "status").read_text(encoding="utf-8")
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
            ppid = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("PPid:"))
            rss = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
            result[int(proc.name)] = {"ppid": ppid, "rss_kib": rss, "cmdline": cmdline}
        except (OSError, StopIteration, ValueError):
            continue
    return result


def _training_process(root_pid: int) -> tuple[int | None, int | None]:
    snapshot = _proc_snapshot()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, item in snapshot.items():
            if item["ppid"] in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    candidates = [
        (pid, item) for pid, item in snapshot.items()
        if pid in descendants and ("python" in item["cmdline"].casefold() or pid == root_pid)
    ]
    if not candidates:
        return None, None
    pid, item = max(candidates, key=lambda pair: ("python" in pair[1]["cmdline"].casefold(), pair[0]))
    return pid, int(item["rss_kib"])


def _gpu_sample() -> tuple[str, str, str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits", "--id=0"],
        text=True,
        capture_output=True,
        check=True,
    )
    first = completed.stdout.strip().splitlines()[0]
    values = [value.strip() for value in first.split(",")]
    if len(values) != 3:
        raise RuntimeError("nvidia-smi returned an unexpected field count")
    return values[0], values[1], values[2]


def _host_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _json_or_empty(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def watch(run_id: str, control_dir: Path, run_dir: Path, process_pid: int, interval_seconds: int = 60) -> dict[str, Any]:
    if not run_id or interval_seconds < 1 or process_pid < 1:
        raise ValueError("run id, process id, and sample interval must be explicit positive values")
    telemetry = control_dir / "telemetry.csv"
    status_path = control_dir / "telemetry_status.json"
    stop_path = control_dir / "telemetry.stop"
    control_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    samples = 0
    while True:
        if stop_path.exists():
            break
        progress = _json_or_empty(run_dir / "progress.json")
        collector_errors: list[str] = []
        try:
            gpu_util, gpu_used, gpu_total = _gpu_sample()
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            gpu_util = gpu_used = gpu_total = ""
            collector_errors.append(f"gpu:{type(error).__name__}:{error}")
        try:
            host_available = str(_host_available_kib())
        except (OSError, ValueError, RuntimeError) as error:
            host_available = ""
            collector_errors.append(f"host_memory:{type(error).__name__}:{error}")
        train_pid, rss = _training_process(process_pid)
        progress_state = str(progress.get("status", "UNKNOWN"))
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": run_id,
            "run_state": "RUNNING" if _process_alive(process_pid) else "PROCESS_EXITED",
            "progress_state": progress_state,
            "training_pid": train_pid or "",
            "training_rss_kib": rss if rss is not None else "",
            "gpu_utilization_percent": gpu_util,
            "gpu_memory_used_mib": gpu_used,
            "gpu_memory_total_mib": gpu_total,
            "host_mem_available_kib": host_available,
            "seed": progress.get("current_seed", ""),
            "arm": progress.get("current_arm", ""),
            "task_local_update": progress.get("current_task_local_update", ""),
            "optimizer_step": progress.get("current_optimizer_step", ""),
            "sample_error": " | ".join(collector_errors),
        }
        with telemetry.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        samples += 1
        failures += bool(collector_errors)
        status = {"run_id": run_id, "state": "COLLECTING", "samples": samples, "collector_failures": failures, "last_sample_utc": row["timestamp_utc"], "last_error": row["sample_error"] or None}
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, status_path)
        if not _process_alive(process_pid):
            break
        for _ in range(interval_seconds):
            if stop_path.exists() or not _process_alive(process_pid):
                break
            time.sleep(1)
    final = {"run_id": run_id, "state": "STOPPED", "samples": samples, "collector_failures": failures, "telemetry": telemetry.name}
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, status_path)
    return final


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--process-pid", required=True, type=int)
    parser.add_argument("--interval-seconds", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(watch(**vars(_arguments())), ensure_ascii=False, sort_keys=True))
