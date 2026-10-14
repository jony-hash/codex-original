from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = ROOT / "scripts" / "maintenance" / "autodl_run.ps1"
PULL_SCRIPT = ROOT / "scripts" / "maintenance" / "autodl_pull.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("pwsh.exe")


@pytest.fixture
def ssh_config(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    config.write_text("Host autodl-test\n  HostName 127.0.0.1\n", encoding="utf-8")
    return config


def invoke(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("PowerShell is required to validate the AutoDL scripts.")
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def assert_rejected(result: subprocess.CompletedProcess[str], phrase: str) -> None:
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert phrase.lower() in combined.lower()


def test_start_rejects_traversal_and_protected_output(ssh_config: Path) -> None:
    prefix = ["-Action", "Start", "-SshConfig", str(ssh_config), "-TaskName", "test"]
    traversal = invoke(RUN_SCRIPT, *prefix, "-RunDir", "outputs/../data/run")
    protected = invoke(RUN_SCRIPT, *prefix, "-RunDir", "outputs/20260101_test/runs/final_pt")

    assert_rejected(traversal, "versioned outputs")
    assert_rejected(protected, "protected output namespace")


def test_status_rejects_missing_or_malformed_run_id(ssh_config: Path) -> None:
    result = invoke(
        RUN_SCRIPT,
        "-Action",
        "Status",
        "-SshConfig",
        str(ssh_config),
        "-TaskName",
        "test",
        "-RunId",
        "latest",
    )

    assert_rejected(result, "32-character identifier")


def test_task_control_directory_cannot_overlap_project(ssh_config: Path) -> None:
    result = invoke(
        RUN_SCRIPT,
        "-Action",
        "Check",
        "-SshConfig",
        str(ssh_config),
        "-RemoteProjectRoot",
        "/root/autodl-tmp/project",
        "-ControlRoot",
        "/root/autodl-tmp/project/control",
    )

    assert_rejected(result, "must be separate directories")


def test_runner_preserves_the_collector_csv_in_the_registered_run_directory() -> None:
    runner_source = RUN_SCRIPT.read_text(encoding="utf-8")

    assert 'telemetry_output="$run_output/resource_telemetry.csv"' in runner_source
    assert 'cp -- "$job_dir/telemetry.csv" "$telemetry_output"' in runner_source
    assert 'if [ -f "$job_dir/telemetry.csv" ] && [ ! -e "$telemetry_output" ]; then' in runner_source


def test_pull_rejects_traversal_and_protected_paths(ssh_config: Path) -> None:
    destination = ROOT / "outputs" / "20261012_autodl_workflow_test_v1" / "runs" / "autodl_backup_test"
    common = [
        "-SshConfig",
        str(ssh_config),
        "-RemoteRunDir",
        "outputs/20261012_test/runs/attempt_01",
        "-Destination",
        str(destination),
    ]
    traversal = invoke(PULL_SCRIPT, *common, "-Files", "../report.json")
    protected = invoke(
        PULL_SCRIPT,
        "-SshConfig",
        str(ssh_config),
        "-RemoteRunDir",
        "outputs/20261012_test/runs/final_pt",
        "-Destination",
        str(destination),
        "-Files",
        "report.json",
    )

    assert_rejected(traversal, "invalid relative file path")
    assert_rejected(protected, "protected output namespace")


def test_pull_refuses_existing_backup_without_contacting_host(ssh_config: Path) -> None:
    version_name = "20261012_autodl_workflow_test_" + uuid.uuid4().hex[:10]
    version_root = ROOT / "outputs" / version_name
    backup = version_root / "runs" / ("autodl_backup_" + uuid.uuid4().hex[:10])
    backup.mkdir(parents=True)
    try:
        result = invoke(
            PULL_SCRIPT,
            "-SshConfig",
            str(ssh_config),
            "-RemoteRunDir",
            "outputs/20261012_test/runs/attempt_01",
            "-Files",
            "report.json",
            "-Destination",
            str(backup),
        )
        assert_rejected(result, "already exists")
    finally:
        resolved_root = version_root.resolve()
        outputs_root = (ROOT / "outputs").resolve()
        assert resolved_root.parent == outputs_root
        shutil.rmtree(resolved_root)


def test_pull_manifest_rejects_traversal_and_missing_local_dependency(ssh_config: Path, tmp_path: Path) -> None:
    version_name = "20261012_autodl_manifest_test_" + uuid.uuid4().hex[:10]
    backup = ROOT / "outputs" / version_name / "runs" / ("autodl_backup_" + uuid.uuid4().hex[:10])
    manifest = tmp_path / "manifest.json"
    common = [
        "-SshConfig", str(ssh_config), "-RemoteRunDir", "outputs/20261012_test/runs/attempt_01",
        "-Destination", str(backup), "-ManifestFile", str(manifest),
    ]
    try:
        manifest.write_text(json.dumps({"schema": "autodl_review_manifest_v1", "files": ["../report.json"], "required_local_dependencies": []}), encoding="utf-8")
        traversal = invoke(PULL_SCRIPT, *common)
        assert_rejected(traversal, "invalid relative file path")

        manifest.write_text(json.dumps({"schema": "autodl_review_manifest_v1", "files": ["review_bundle/report.json"], "required_local_dependencies": ["outputs/missing/start.pt"]}), encoding="utf-8")
        missing_dependency = invoke(PULL_SCRIPT, *common)
        assert_rejected(missing_dependency, "required local start dependency is missing")
    finally:
        version_root = ROOT / "outputs" / version_name
        if version_root.exists():
            assert version_root.resolve().parent == (ROOT / "outputs").resolve()
            shutil.rmtree(version_root)


def test_pull_manifest_keeps_large_file_list_on_standard_input(ssh_config: Path, tmp_path: Path) -> None:
    version_name = "20261012_autodl_large_manifest_test_" + uuid.uuid4().hex[:10]
    backup = ROOT / "outputs" / version_name / "runs" / ("autodl_backup_" + uuid.uuid4().hex[:10])
    manifest = tmp_path / "large-manifest.json"
    manifest.write_text(
        json.dumps({
            "schema": "autodl_review_manifest_v1",
            "files": [f"review_bundle/files/{index:05d}.json" for index in range(12000)],
            "required_local_dependencies": [],
        }),
        encoding="utf-8",
    )
    try:
        result = invoke(
            PULL_SCRIPT,
            "-SshConfig", str(ssh_config),
            "-SshAlias", "autodl-test",
            "-RemoteRunDir", "outputs/20261012_test/runs/attempt_01",
            "-ManifestFile", str(manifest),
            "-Destination", str(backup),
        )
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "Remote preflight failed" in combined
        receipts = list((ROOT / "outputs" / version_name / "runs").rglob("transfer_receipt.json"))
        assert len(receipts) == 1
        receipt = json.loads(receipts[0].read_text(encoding="utf-8-sig"))
        assert "SSH exited 255" in receipt["error"]
        assert "too long" not in receipt["error"].lower()
    finally:
        version_root = ROOT / "outputs" / version_name
        if version_root.exists():
            assert version_root.resolve().parent == (ROOT / "outputs").resolve()
            shutil.rmtree(version_root)
