"""Isolated tests for the local Codex history/project binding boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from contextlib import redirect_stdout
from io import StringIO
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
        self.original_migration_manifest = self.migration_manifest.read_bytes()
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
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                cwd TEXT,
                rollout_path TEXT,
                source TEXT,
                thread_source TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                archived_at TEXT,
                title TEXT
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT NOT NULL,
                child_thread_id TEXT NOT NULL
            );
            """
        )
        self.base_id = str(uuid.uuid4())
        connection.execute(
            "INSERT INTO threads "
            "(id, project_id, cwd, source, thread_source, archived, archived_at, title) "
            "VALUES (?, NULL, ?, ?, ?, 0, NULL, ?)",
            (
                self.base_id,
                str(self.root / "misc"),
                "vscode",
                "vscode",
                "base thread",
            ),
        )

        visible_ids = list(MIGRATION.PROJECT_BIND_CANARY_IDS[:2])
        visible_ids.extend(str(uuid.uuid4()) for _ in range(142))
        archived_ids = [MIGRATION.PROJECT_BIND_CANARY_IDS[2]]
        archived_ids.extend(str(uuid.uuid4()) for _ in range(69))
        self.candidate_ids = visible_ids + archived_ids
        for index, session_id in enumerate(self.candidate_ids):
            visible = index < 144
            source = "subagent" if session_id == MIGRATION.PROJECT_BIND_CANARY_IDS[2] else "vscode"
            thread_source = "subagent" if session_id == MIGRATION.PROJECT_BIND_CANARY_IDS[2] else "vscode"
            connection.execute(
                "INSERT INTO threads "
                "(id, project_id, cwd, source, thread_source, archived, archived_at, title) "
                "VALUES (?, NULL, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    MIGRATION.OLD_PROJECT_CWD if index == 0 else str(self.project_root),
                    source,
                    thread_source,
                    0 if visible else 1,
                    None if visible else "2026-08-29T00:00:00Z",
                    f"candidate {index}",
                ),
            )

        self.misc_ids = [str(uuid.uuid4()) for _ in range(3)]
        for index, session_id in enumerate(self.misc_ids):
            connection.execute(
                "INSERT INTO threads "
                "(id, project_id, cwd, source, thread_source, archived, archived_at, title) "
                "VALUES (?, NULL, ?, ?, ?, 0, NULL, ?)",
                (
                    session_id,
                    str(self.root / "misc"),
                    "vscode",
                    "vscode",
                    f"misc {index}",
                ),
            )
        connection.execute(
            "INSERT INTO thread_spawn_edges (parent_thread_id, child_thread_id) VALUES (?, ?)",
            (self.candidate_ids[0], self.candidate_ids[2]),
        )
        connection.commit()
        connection.close()

    def _candidate_lines(self, session_id: str, index: int, visible: bool) -> list[dict[str, object]]:
        source = "subagent" if session_id == MIGRATION.PROJECT_BIND_CANARY_IDS[2] else "vscode"
        records: list[dict[str, object]] = [
            {
                "type": "session_meta",
                "payload": {
                    "id": session_id,
                    "cwd": str(self.project_root),
                    "source": source,
                    "thread_source": source,
                    "archived": not visible,
                    "title": f"candidate {index}",
                },
            }
        ]
        if session_id in {
            MIGRATION.PROJECT_BIND_CANARY_IDS[0],
            MIGRATION.PROJECT_BIND_CANARY_IDS[2],
        }:
            records.append(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": session_id,
                        "cwd": MIGRATION.OLD_PROJECT_CWD,
                        "source": source,
                        "thread_source": source,
                        "archived": not visible,
                        "title": f"candidate {index}",
                        "later": True,
                    },
                }
            )
        records.append(
            {
                "type": "turn_context",
                "payload": {
                    "cwd": MIGRATION.OLD_PROJECT_CWD,
                    "turn": index,
                    "context": {"preserve": True},
                },
            }
        )
        records.append(
            {
                "type": "event_msg",
                "payload": {
                    "cwd": MIGRATION.OLD_PROJECT_CWD,
                    "command": f"echo candidate-{index}",
                    "stdout": f"historical-{index}\n",
                },
            }
        )
        records.append(
            {
                "type": "response_item",
                "payload": {"text": f"unchanged output {index}"},
            }
        )
        return records

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
            lines = self._candidate_lines(session_id, index, visible)
            target.write_text(
                "".join(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n" for line in lines),
                encoding="utf-8",
                newline="",
            )
            records.append(
                {
                    "id": session_id,
                    "visible": visible,
                    "archived": not visible,
                    "project_cwd": True,
                    "metadata_cwd_rewritten": True,
                    "target_cwd": str(self.project_root),
                    "target_path": str(target),
                    "transformed_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        for index, session_id in enumerate(self.misc_ids):
            target = sessions / f"{session_id}.jsonl"
            target.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"id": session_id, "cwd": str(self.root / "misc")},
                    },
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
                newline="",
            )
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

    def _read_jsonl(self, session_id: str) -> list[dict[str, object]]:
        path = self.codex_home / "sessions" / f"{session_id}.jsonl"
        if not path.is_file():
            path = self.codex_home / "archived_sessions" / f"{session_id}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def _thread_snapshot(self, session_ids: list[str]) -> dict[str, tuple[object, ...]]:
        connection = sqlite3.connect(self.state)
        rows = connection.execute(
            "SELECT id, project_id, cwd, source, thread_source, archived, archived_at, title "
            f"FROM threads WHERE id IN ({','.join('?' for _ in session_ids)})",
            session_ids,
        ).fetchall()
        connection.close()
        return {str(row[0]).lower(): tuple(row[1:]) for row in rows}

    def test_dry_run_reports_folder_bound_state_without_project_tables(self) -> None:
        preview = MIGRATION.project_binding_dry_run(self.args)
        self.assertEqual(preview["status"], "project_bind_dry_run_ok")
        self.assertEqual(preview["mode"], "folder_cwd_repair")
        self.assertEqual(preview["candidate_threads"], 214)
        self.assertEqual(preview["visible_threads"], 144)
        self.assertEqual(preview["archived_threads"], 70)
        self.assertEqual(preview["misc_visible_unbound"], 3)
        self.assertEqual(preview["candidate_cwd_old_wsl"], 1)
        self.assertEqual(preview["candidate_cwd_target"], 213)
        self.assertGreater(preview["metadata_session_meta_pending"], 0)
        self.assertEqual(preview["metadata_turn_context_pending"], 214)
        self.assertEqual(preview["event_msg_old_cwd_negative_control"], 214)
        self.assertEqual(preview["file_count"], 217)
        self.assertEqual(preview["sqlite_integrity"], "ok")
        self.assertTrue(preview["ready_to_bind"])

    def test_top_level_verify_registers_project_binding_arguments(self) -> None:
        binding_snapshot_root = self.root / "missing-binding-snapshot"
        cli_args = [
            "verify",
            "--archive",
            str(self.archive),
            "--codex-home",
            str(self.codex_home),
            "--project-root",
            str(self.project_root),
            "--manifest-file",
            str(self.migration_manifest),
            "--binding-snapshot-root",
            str(binding_snapshot_root),
        ]
        with redirect_stdout(StringIO()):
            self.assertEqual(MIGRATION.main(cli_args), 0)

    def test_verify_accepts_equivalent_extended_rollout_path(self) -> None:
        session_id = self.candidate_ids[0]
        rollout_path = self.root / "rollout.jsonl"
        connection = sqlite3.connect(self.state)
        connection.execute(
            "UPDATE threads SET rollout_path = ? WHERE id = ?",
            (str(rollout_path), session_id),
        )
        connection.commit()
        connection.close()

        MIGRATION.apply_project_binding(self.args, "canary")
        connection = sqlite3.connect(self.state)
        connection.execute(
            "UPDATE threads SET rollout_path = ? WHERE id = ?",
            ("\\\\?\\" + str(rollout_path), session_id),
        )
        connection.commit()
        connection.close()

        verified = MIGRATION.verify_project_binding(self.args)
        self.assertEqual(verified["phase"], "canary_complete")

    def test_apply_canary_complete_verify_and_rollback_preserves_invariants(self) -> None:
        before_apply = self._thread_snapshot(self.candidate_ids + self.misc_ids)
        canary = MIGRATION.apply_project_binding(self.args, "canary")
        self.assertEqual(canary["phase"], "canary_complete")
        self.assertEqual(canary["repaired_this_run"], 3)
        self.assertEqual(canary["repaired_total"], 3)

        canary_verified = MIGRATION.verify_project_binding(self.args)
        self.assertEqual(canary_verified["phase"], "canary_complete")
        self.assertEqual(canary_verified["repaired_threads"], 3)
        self.assertEqual(canary_verified["session_check"]["file_count"], 217)
        self.assertEqual(MIGRATION.verify_import(self.args)["file_count"], 217)

        complete = MIGRATION.apply_project_binding(self.args, "complete")
        self.assertEqual(complete["phase"], "complete")
        self.assertEqual(complete["repaired_this_run"], 211)
        self.assertEqual(complete["repaired_total"], 214)

        verified = MIGRATION.verify_project_binding(self.args)
        self.assertEqual(verified["phase"], "complete")
        self.assertEqual(verified["repaired_threads"], 214)
        self.assertEqual(verified["parent_child_edges"], 1)
        self.assertEqual(verified["session_check"]["file_count"], 217)
        self.assertEqual(MIGRATION.verify_import(self.args)["file_count"], 217)
        after_complete = self._thread_snapshot(self.candidate_ids + self.misc_ids)
        for session_id in before_apply:
            self.assertEqual(
                before_apply[session_id][0:1] + before_apply[session_id][2:],
                after_complete[session_id][0:1] + after_complete[session_id][2:],
            )

        new_thread = str(uuid.uuid4())
        connection = sqlite3.connect(self.state)
        connection.execute(
            "INSERT INTO threads "
            "(id, project_id, cwd, source, thread_source, archived, archived_at, title) "
            "VALUES (?, NULL, ?, 'vscode', 'vscode', 0, NULL, 'new after snapshot')",
            (new_thread, str(self.project_root)),
        )
        connection.commit()
        connection.close()

        rolled_back = MIGRATION.rollback_project_binding(self.args)
        self.assertEqual(rolled_back["phase"], "rolled_back")
        self.assertEqual(rolled_back["restored_threads"], 214)
        self.assertEqual(rolled_back["restored_files"], 214)
        self.assertTrue(Path(rolled_back["quarantine"]).is_dir())
        self.assertEqual(self._thread_snapshot(self.candidate_ids + self.misc_ids), before_apply)

        connection = sqlite3.connect(self.state)
        self.assertIsNotNone(connection.execute("SELECT id FROM threads WHERE id = ?", (new_thread,)).fetchone())
        project_ids = connection.execute(
            "SELECT project_id FROM threads WHERE id IN "
            f"({','.join('?' for _ in self.candidate_ids)})",
            self.candidate_ids,
        ).fetchall()
        connection.close()
        self.assertTrue(all(row[0] is None for row in project_ids))
        after = MIGRATION.project_binding_dry_run(self.args)
        self.assertEqual(after["phase"], "rolled_back")
        self.assertEqual(after["already_repaired"], 0)
        self.assertEqual(self.migration_manifest.read_bytes(), self.original_migration_manifest)

    def test_metadata_repair_changes_only_cwd_and_preserves_event_msg_bytes(self) -> None:
        session_id = MIGRATION.PROJECT_BIND_CANARY_IDS[1]
        path = self.codex_home / "sessions" / f"{session_id}.jsonl"
        before_lines = path.read_bytes().splitlines(keepends=True)
        before_records = self._read_jsonl(session_id)
        before_threads = self._thread_snapshot([session_id])

        MIGRATION.apply_project_binding(self.args, "canary")

        after_lines = path.read_bytes().splitlines(keepends=True)
        after_records = self._read_jsonl(session_id)
        event_index = next(index for index, line in enumerate(before_lines) if b'"type":"event_msg"' in line)
        self.assertEqual(before_lines[event_index], after_lines[event_index])
        self.assertEqual(before_records[0], after_records[0])
        self.assertEqual(before_records[1]["type"], after_records[1]["type"])
        self.assertEqual(before_records[1]["payload"]["cwd"], MIGRATION.OLD_PROJECT_CWD)
        self.assertEqual(after_records[1]["payload"]["cwd"], str(self.project_root))
        before_without_cwd = dict(before_records[1])
        after_without_cwd = dict(after_records[1])
        before_payload = dict(before_without_cwd["payload"])
        after_payload = dict(after_without_cwd["payload"])
        before_payload.pop("cwd")
        after_payload.pop("cwd")
        before_without_cwd["payload"] = before_payload
        after_without_cwd["payload"] = after_payload
        self.assertEqual(before_without_cwd, after_without_cwd)
        self.assertEqual(self._thread_snapshot([session_id]), before_threads)

    def test_canary_and_complete_phase_rejections_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(MIGRATION.MigrationError, "successful canary"):
            MIGRATION.apply_project_binding(self.args, "complete")
        MIGRATION.apply_project_binding(self.args, "canary")
        with self.assertRaisesRegex(MIGRATION.MigrationError, "requires snapshot_created"):
            MIGRATION.apply_project_binding(self.args, "canary")

    def test_canary_rollback_restores_only_canary_files_and_rows(self) -> None:
        before = self._thread_snapshot(self.candidate_ids + self.misc_ids)
        MIGRATION.apply_project_binding(self.args, "canary")
        rolled_back = MIGRATION.rollback_project_binding(self.args)
        self.assertEqual(rolled_back["phase"], "rolled_back")
        self.assertEqual(rolled_back["restored_files"], 3)
        self.assertEqual(self._thread_snapshot(self.candidate_ids + self.misc_ids), before)
        self.assertTrue(MIGRATION.project_binding_dry_run(self.args)["ready_to_bind"])

    def test_unknown_database_cwd_is_rejected_but_exact_wsl_cwd_is_allowed(self) -> None:
        preview = MIGRATION.project_binding_dry_run(self.args)
        self.assertEqual(preview["candidate_cwd_old_wsl"], 1)
        connection = sqlite3.connect(self.state)
        connection.execute(
            "UPDATE threads SET cwd = ? WHERE id = ?",
            (str(self.root / "unexpected"), self.candidate_ids[0]),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(MIGRATION.MigrationError, "unknown candidate cwd"):
            MIGRATION.project_binding_dry_run(self.args)

    def test_unknown_metadata_cwd_is_rejected(self) -> None:
        session_id = MIGRATION.PROJECT_BIND_CANARY_IDS[1]
        path = self.codex_home / "sessions" / f"{session_id}.jsonl"
        records = self._read_jsonl(session_id)
        records[1]["payload"]["cwd"] = str(self.root / "unexpected-metadata")
        path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
            newline="",
        )
        with self.assertRaisesRegex(MIGRATION.MigrationError, "unknown metadata cwd"):
            MIGRATION.audit_project_binding_file(path, self.args, session_id)

    def test_verify_rejects_hash_tampering_and_does_not_change_migration_manifest(self) -> None:
        MIGRATION.apply_project_binding(self.args, "canary")
        untouched_id = next(
            session_id
            for session_id in self.candidate_ids
            if session_id not in MIGRATION.PROJECT_BIND_CANARY_IDS
        )
        path = self.codex_home / "sessions" / f"{untouched_id}.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(MIGRATION.MigrationError, "changed imported session file"):
            MIGRATION.verify_project_binding(self.args)
        self.assertEqual(self.migration_manifest.read_bytes(), self.original_migration_manifest)

    def test_extended_drive_path_is_equivalent(self) -> None:
        plain = str(self.project_root)
        extended = "\\\\?\\" + plain
        self.assertEqual(MIGRATION.canonical_path(plain), MIGRATION.canonical_path(extended))


if __name__ == "__main__":
    unittest.main()
