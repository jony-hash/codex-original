"""Isolated tests for the local Codex history/project binding boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import uuid


SCRIPT = Path(__file__).parents[1] / "scripts" / "maintenance" / "codex_history_migration.py"
SPEC = importlib.util.spec_from_file_location("codex_history_migration_under_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load migration script: {SCRIPT}")
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)


class ProjectBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="codex-project-bind-test-")
        self.root = Path(self.tempdir.name)
        self.codex_home = self.root / "codex-home"
        self.codex_home.mkdir()
        self.project_root = self.root / "repo"
        self.project_root.mkdir()
        self.archive = self.root / "legacy.tar"
        self.archive.write_bytes(b"immutable test archive")
        self.state = self.codex_home / "state_5.sqlite"
        self._create_state()
        self.migration_manifest = self.root / "migration-manifest.json"
        self._create_migration_manifest()
        self.args = type("Args", (), {})()
        self.args.archive = self.archive
        self.args.codex_home = self.codex_home
        self.args.project_root = self.project_root
        self.args.manifest_file = self.migration_manifest
        self.args.project_name = "codex-original · Windows"
        self.args.binding_snapshot_root = self.root / "binding-snapshot"
        self.args.binding_manifest = None
        self.original_require_apps_closed = MIGRATION.require_apps_closed
        MIGRATION.require_apps_closed = lambda: None

    def tearDown(self) -> None:
        MIGRATION.require_apps_closed = self.original_require_apps_closed
        self.tempdir.cleanup()

    def _create_state(self) -> None:
        connection = sqlite3.connect(self.state)
        connection.executescript(
            """
            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                position INTEGER NOT NULL DEFAULT 0,
                created_at_ms INTEGER NOT NULL DEFAULT 0,
                updated_at_ms INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE project_roots (
                project_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                path TEXT NOT NULL,
                PRIMARY KEY(project_id, position)
            );
            CREATE TABLE threads (id TEXT PRIMARY KEY, project_id TEXT, cwd TEXT);
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT NOT NULL,
                child_thread_id TEXT NOT NULL
            );
            """
        )
        project_id = "local-project-1"
        connection.execute(
            "INSERT INTO projects (id, name) VALUES (?, ?)",
            (project_id, "codex-original · Windows"),
        )
        connection.execute(
            "INSERT INTO project_roots (project_id, position, path) VALUES (?, 0, ?)",
            (project_id, str(self.project_root)),
        )
        self.base_id = str(uuid.uuid4())
        connection.execute(
            "INSERT INTO threads (id, project_id, cwd) VALUES (?, NULL, ?)",
            (self.base_id, str(self.root / "misc")),
        )
        self.candidate_ids: list[str] = []
        for _ in range(214):
            session_id = str(uuid.uuid4())
            self.candidate_ids.append(session_id)
            connection.execute(
                "INSERT INTO threads (id, project_id, cwd) VALUES (?, NULL, ?)",
                (session_id, str(self.project_root)),
            )
        self.misc_ids: list[str] = []
        for _ in range(3):
            session_id = str(uuid.uuid4())
            self.misc_ids.append(session_id)
            connection.execute(
                "INSERT INTO threads (id, project_id, cwd) VALUES (?, NULL, ?)",
                (session_id, str(self.root / "misc")),
            )
        connection.execute(
            "INSERT INTO thread_spawn_edges (parent_thread_id, child_thread_id) VALUES (?, ?)",
            (self.candidate_ids[0], self.candidate_ids[1]),
        )
        connection.commit()
        connection.close()

    def _create_migration_manifest(self) -> None:
        sessions = self.codex_home / "sessions"
        archived = self.codex_home / "archived_sessions"
        sessions.mkdir()
        archived.mkdir()
        records: list[dict[str, object]] = []
        for index, session_id in enumerate(self.candidate_ids):
            visible = index < 144
            destination = sessions if visible else archived
            target = destination / f"{session_id}.jsonl"
            target.write_text('{"event":"test"}\n', encoding="utf-8")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            records.append(
                {
                    "id": session_id,
                    "visible": visible,
                    "archived": not visible,
                    "project_cwd": True,
                    "metadata_cwd_rewritten": True,
                    "target_cwd": str(self.project_root),
                    "target_path": str(target),
                    "transformed_sha256": digest,
                }
            )
        for session_id in self.misc_ids:
            target = sessions / f"{session_id}.jsonl"
            target.write_text('{"event":"misc"}\n', encoding="utf-8")
            records.append(
                {
                    "id": session_id,
                    "visible": True,
                    "archived": False,
                    "project_cwd": False,
                    "metadata_cwd_rewritten": False,
                    "target_cwd": str(self.root / "misc"),
                    "target_path": str(target),
                    "transformed_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        plan = {
            "format_version": 1,
            "archive": str(self.archive),
            "codex_home": str(self.codex_home),
            "project_root": str(self.project_root),
            "current_before": {"thread_ids": [self.base_id]},
        }
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        manifest = {
            "format_version": 1,
            "phase": "complete",
            "archive": str(self.archive),
            "codex_home": str(self.codex_home),
            "project_root": str(self.project_root),
            "plan_file": str(plan_path),
            "imported_ids": self.candidate_ids + self.misc_ids,
            "file_records": records,
        }
        self.migration_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    def test_apply_verify_and_rollback(self) -> None:
        preview = MIGRATION.project_binding_dry_run(self.args)
        self.assertTrue(preview["ready_to_bind"])
        self.assertEqual(preview["candidate_threads"], 214)
        self.assertEqual(preview["misc_visible_unbound"], 3)

        applied = MIGRATION.apply_project_binding(self.args)
        self.assertEqual(applied["bound_threads"], 214)
        self.assertEqual(applied["visible_threads"], 144)
        self.assertEqual(applied["archived_threads"], 70)

        verified = MIGRATION.verify_project_binding(self.args)
        self.assertEqual(verified["bound_threads"], 214)
        self.assertEqual(verified["parent_child_edges"], 1)
        self.assertEqual(verified["session_check"]["file_count"], 217)

        rolled_back = MIGRATION.rollback_project_binding(self.args)
        self.assertEqual(rolled_back["unbound_threads"], 214)
        after = MIGRATION.project_binding_dry_run(self.args)
        self.assertTrue(after["ready_to_bind"])
        self.assertEqual(after["already_bound"], 0)

    def test_dry_run_rejects_non_project_cwd(self) -> None:
        connection = sqlite3.connect(self.state)
        connection.execute(
            "UPDATE threads SET cwd = ? WHERE id = ?",
            ("/home/dingzhen/projects/codex-original", self.candidate_ids[0]),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(MIGRATION.MigrationError, "non-project cwd"):
            MIGRATION.project_binding_dry_run(self.args)


if __name__ == "__main__":
    unittest.main()
