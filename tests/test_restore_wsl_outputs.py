"""Tests for the fail-closed WSL outputs restore boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "maintenance" / "restore_wsl_outputs.py"
SPEC = importlib.util.spec_from_file_location("restore_wsl_outputs_under_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load restore tool: {SCRIPT}")
RESTORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RESTORE
SPEC.loader.exec_module(RESTORE)


class RestoreWslOutputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="restore-wsl-outputs-")
        self.root = Path(self.tempdir.name)
        self.archive_root = self.root / "archive"
        self.target = self.root / "project" / "outputs"
        self.archive_root.mkdir()
        self.target.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_tar(self, name: str, members: list[tuple[str, bytes | None, str]]) -> Path:
        path = self.archive_root / name
        with tarfile.open(path, "w") as archive:
            for member_name, content, kind in members:
                info = tarfile.TarInfo(member_name)
                if kind == "directory":
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    archive.addfile(info)
                elif kind == "file":
                    payload = content or b""
                    info.size = len(payload)
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(payload))
                else:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "outputs/target"
                    archive.addfile(info)
        return path

    def _write_index_and_manifest(self, archive_name: str, top_level: str, file_count: int, source_bytes: int) -> None:
        archive_path = self.archive_root / archive_name
        archive_bytes = archive_path.stat().st_size
        sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest().upper()
        manifest = {
            "index": 1,
            "linux_path": f"/home/user/project/outputs/{top_level}",
            "relative_path": f"outputs/{top_level}",
            "archive_name": archive_name,
            "file_count": file_count,
            "bytes": source_bytes,
            "apparent_bytes": source_bytes,
            "archive_bytes": archive_bytes,
            "sha256": sha256,
        }
        (self.archive_root / RESTORE.MANIFEST_FILENAME).write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        index = {
            "schema_version": 1,
            "top_level_count": 1,
            "archived_count": 1,
            "source_apparent_bytes": source_bytes,
        }
        (self.archive_root / RESTORE.INDEX_FILENAME).write_text(
            json.dumps(index), encoding="utf-8"
        )

    def test_trailing_dot_and_long_path_are_mapped_deterministically(self) -> None:
        trailing = "outputs/run./result.txt"
        long_directory = "x" * 180
        long_member = f"outputs/long/runs/{long_directory}/{long_directory}/result.txt"
        target_root = Path("C:/") / ("x" * 1) / "outputs"
        mapped_trailing, trailing_reasons = RESTORE._map_components(
            tuple(trailing.removeprefix("outputs/").split("/")),
            target_root,
            "run./result.txt",
        )
        mapped_long, long_reasons = RESTORE._map_components(
            tuple(long_member.removeprefix("outputs/").split("/")),
            target_root,
            long_member.removeprefix("outputs/"),
        )
        self.assertTrue(mapped_trailing[0].endswith(RESTORE.TRAILING_COMPONENT_SUFFIX))
        self.assertIn("trailing_dot_or_space", trailing_reasons)
        self.assertIn("path_length", long_reasons)
        self.assertLessEqual(len(str(target_root.joinpath(*mapped_long))), RESTORE.MAX_WINDOWS_PATH)

    def test_path_traversal_is_rejected(self) -> None:
        self._write_tar("archive_0001.tar", [("outputs/demo/../evil.txt", b"x", "file")])
        self._write_index_and_manifest("archive_0001.tar", "demo", 1, 1)
        with self.assertRaises(RESTORE.RestoreError):
            RESTORE.build_plan(self.archive_root, self.target)

    def test_symlink_is_rejected(self) -> None:
        self._write_tar("archive_0001.tar", [("outputs/demo/link", None, "symlink")])
        self._write_index_and_manifest("archive_0001.tar", "demo", 0, 0)
        with self.assertRaises(RESTORE.RestoreError):
            RESTORE.build_plan(self.archive_root, self.target)

    def test_case_insensitive_collision_is_rejected(self) -> None:
        members = [
            ("outputs/demo/File.txt", b"a", "file"),
            ("outputs/demo/file.txt", b"b", "file"),
        ]
        self._write_tar("archive_0001.tar", members)
        self._write_index_and_manifest("archive_0001.tar", "demo", 2, 2)
        with self.assertRaises(RESTORE.RestoreError):
            RESTORE.build_plan(self.archive_root, self.target)

    def test_dry_run_apply_and_verify_preserve_identical_existing_file(self) -> None:
        content = b"existing checkpoint\n"
        existing = self.target / "demo" / "checkpoint.pt"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(content)
        self._write_tar("archive_0001.tar", [("outputs/demo/checkpoint.pt", content, "file")])
        self._write_index_and_manifest("archive_0001.tar", "demo", 1, len(content))
        plan_file = self.root / "plan.json"
        state_root = self.root / "state"
        staging_root = self.root / "staging"
        plan = RESTORE.build_plan(self.archive_root, self.target)
        RESTORE._write_json_atomic(plan_file, plan)
        result = RESTORE.apply_plan(plan_file, state_root, staging_root)
        self.assertEqual(result["status"], "apply_ok")
        verified = RESTORE.verify_plan(plan_file, state_root)
        self.assertEqual(verified["status"], "verify_ok")
        self.assertEqual(existing.read_bytes(), content)
        self.assertFalse(staging_root.exists() and any(staging_root.iterdir()))

    def test_different_existing_file_is_rejected_without_overwrite(self) -> None:
        existing = self.target / "demo" / "checkpoint.pt"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"current\n")
        archived = b"historical\n"
        self._write_tar("archive_0001.tar", [("outputs/demo/checkpoint.pt", archived, "file")])
        self._write_index_and_manifest("archive_0001.tar", "demo", 1, len(archived))
        plan_file = self.root / "plan.json"
        RESTORE._write_json_atomic(plan_file, RESTORE.build_plan(self.archive_root, self.target))
        with self.assertRaises(RESTORE.RestoreError):
            RESTORE.apply_plan(plan_file, self.root / "state", self.root / "staging")
        self.assertEqual(existing.read_bytes(), b"current\n")

    def test_archive_source_changes_are_rejected_after_plan(self) -> None:
        content = b"data\n"
        archive_path = self._write_tar("archive_0001.tar", [("outputs/demo/data.txt", content, "file")])
        self._write_index_and_manifest("archive_0001.tar", "demo", 1, len(content))
        plan_file = self.root / "plan.json"
        RESTORE._write_json_atomic(plan_file, RESTORE.build_plan(self.archive_root, self.target))
        archive_path.write_bytes(archive_path.read_bytes() + b"tamper")
        with self.assertRaises(RESTORE.RestoreError):
            RESTORE.apply_plan(plan_file, self.root / "state", self.root / "staging")

    def test_free_space_gate_rejects_below_safety_floor(self) -> None:
        actual = shutil.disk_usage(self.target)
        low_free = type(actual)(actual.total, actual.used, RESTORE.MIN_FREE_BYTES - 1)
        with patch.object(RESTORE.shutil, "disk_usage", return_value=low_free):
            with self.assertRaises(RESTORE.RestoreError):
                RESTORE._check_free_space(self.target, 0, initial=False)

    def test_installed_state_can_finish_after_interruption(self) -> None:
        content = b"resumable\n"
        self._write_tar("archive_0001.tar", [("outputs/demo/data.txt", content, "file")])
        self._write_index_and_manifest("archive_0001.tar", "demo", 1, len(content))
        plan_file = self.root / "plan.json"
        state_root = self.root / "state"
        staging_root = self.root / "staging"
        RESTORE._write_json_atomic(plan_file, RESTORE.build_plan(self.archive_root, self.target))
        first = RESTORE.apply_plan(plan_file, state_root, staging_root)
        self.assertEqual(first["status"], "apply_ok")
        state_path = state_root / "restore-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["archive_status"]["archive_0001.tar"]["status"] = "installed"
        RESTORE._write_json_atomic(state_path, state)
        (staging_root / "archive_0001.tar").mkdir(parents=True)
        resumed = RESTORE.apply_plan(plan_file, state_root, staging_root)
        self.assertEqual(resumed["status"], "apply_ok")
        self.assertTrue((self.target / "demo" / "data.txt").is_file())


if __name__ == "__main__":
    unittest.main()
