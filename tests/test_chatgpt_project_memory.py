"""Regression tests for the controlled ChatGPT Project Sources boundary."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "maintenance"
    / "chatgpt_project_memory.py"
)
SPEC = importlib.util.spec_from_file_location("chatgpt_project_memory_under_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Sources tool: {SCRIPT}")
MEMORY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MEMORY
SPEC.loader.exec_module(MEMORY)


class ChatGPTProjectMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="codex-original-test-temp-")
        self.root = Path(self.tempdir.name)
        self.project = self.root / "项目"
        self.history = self.root / "history-context"
        self.output = self.history / "chatgpt-sources"
        (self.project / "docs" / "status").mkdir(parents=True)
        (self.project / "docs" / "architecture").mkdir(parents=True)
        (self.history / "summaries").mkdir(parents=True)
        self._write_sources()
        self._init_git()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_sources(self) -> None:
        project_sources = {
            "docs/PROJECT_MEMORY.md": "# 项目记忆\n真实工作区：测试项目。\n",
            "docs/CURRENT_STATUS.md": "# 当前状态\n测试状态：待验证。\n",
            "docs/status/current_handoff.md": "# 交接\n唯一动作：人工审查。\n",
            "docs/architecture/v2_runtime.md": "# v2\n正式边界：单次曝光。\n",
        }
        for relative, content in project_sources.items():
            target = self.project / Path(*relative.split("/"))
            target.write_text(content, encoding="utf-8")
        evidence = self.project / "docs" / "test-evidence.json"
        evidence.write_text(
            json.dumps({"schema": "test.evidence.v1", "status": "valid"}) + "\n",
            encoding="utf-8",
        )
        state = {
            "schema_version": 1,
            "kind": "research_governance_state",
            "active_objective": "TEST_OBJECTIVE",
            "authorization_tier": "R1",
            "hard_blocker": None,
            "evidence_aliases": {
                "TEST_EVIDENCE_V1": {
                    "kind": "machine_evidence",
                    "path": "docs/test-evidence.json",
                    "role": "test evidence",
                    "schema": "test.evidence.v1",
                    "sha256": MEMORY.sha256_bytes(evidence.read_bytes()),
                    "verification": "verified",
                },
                "TEST_MODEL_ASSUMPTION_V1": {
                    "kind": "model_assumption",
                    "path": "docs/PROJECT_MEMORY.md",
                    "role": "test model assumption",
                    "verification": "not_required",
                },
            },
        }
        (self.project / "docs" / "status" / "research_governance_state.json").write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        (self.history / "HISTORY_CONTEXT.md").write_text(
            "# 历史上下文\n仅作导航。\n", encoding="utf-8"
        )
        (self.history / "CHAT_INDEX.md").write_text(
            "# Chat index\nabc-session\n", encoding="utf-8"
        )
        (self.history / "summaries" / "abc-session.md").write_text(
            "# abc-session\n这是按需选择的历史摘要。\n", encoding="utf-8"
        )

    def _init_git(self) -> None:
        commands = [
            ["git", "-C", str(self.project), "init"],
            [
                "git",
                "-C",
                str(self.project),
                "config",
                "user.email",
                "tests@example.invalid",
            ],
            [
                "git",
                "-C",
                str(self.project),
                "config",
                "user.name",
                "Sources tests",
            ],
            ["git", "-C", str(self.project), "add", "docs"],
            [
                "git",
                "-C",
                str(self.project),
                "commit",
                "-m",
                "test sources boundary",
            ],
        ]
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, text=True)

    def _invoke(self, *arguments: str) -> tuple[int, dict[str, object] | None, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = MEMORY.main(list(arguments))
        result = stdout.getvalue().strip()
        parsed = json.loads(result) if result else None
        return code, parsed, stderr.getvalue()

    def _common_arguments(self, command: str) -> list[str]:
        return [
            command,
            "--project-root",
            str(self.project),
            "--history-root",
            str(self.history),
            "--output-root",
            str(self.output),
        ]

    def _build(self, *session_ids: str) -> dict[str, object]:
        arguments = self._common_arguments("build")
        for session_id in session_ids:
            arguments.extend(["--session-id", session_id])
        code, result, error = self._invoke(*arguments)
        self.assertEqual((code, error), (0, ""))
        self.assertIsNotNone(result)
        return result or {}

    def _verify(self, expect_success: bool = True) -> tuple[int, str]:
        code, _, error = self._invoke(*self._common_arguments("verify"))
        if expect_success:
            self.assertEqual((code, error), (0, ""))
        else:
            self.assertEqual(code, 2)
            self.assertTrue(error.startswith("PROJECT MEMORY STOPPED:"))
        return code, error

    def _status(self) -> dict[str, object]:
        code, result, error = self._invoke(*self._common_arguments("status"))
        self.assertEqual((code, error), (0, ""))
        self.assertIsNotNone(result)
        return result or {}

    def _governance(self, command: str, *arguments: str) -> tuple[int, dict[str, object] | None, str]:
        return self._invoke(
            command,
            "--project-root",
            str(self.project),
            *arguments,
        )

    def test_refresh_emits_snapshot_and_status_is_current(self) -> None:
        code, result, error = self._invoke(*self._common_arguments("refresh"))
        self.assertEqual((code, error), (0, ""))
        self.assertIsNotNone(result)
        payload = result or {}
        self.assertEqual(payload["status"], "refresh_ok")
        self.assertEqual(payload["build_status"], "build_ok")
        self.assertEqual(payload["verification_status"], "verify_ok")
        self.assertRegex(payload["snapshot_id"], r"^[0-9a-f]{12}-")
        self.assertEqual(payload["source_count"], 6)
        package_memory = (self.output / "PROJECT_MEMORY.md").read_text(encoding="utf-8")
        self.assertIn("<!-- Sources bridge snapshot", package_memory)
        self.assertIn(f"snapshot_id: {payload['snapshot_id']}", package_memory)
        self.assertIn(f"git_head: {payload['git_head']}", package_memory)
        self.assertEqual(self._status()["status"], "current")

    def test_governance_summary_and_alias_verification_are_compact_and_fail_closed(self) -> None:
        code, result, error = self._governance("summary")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(result["branch"], MEMORY._run_git(self.project, "branch", "--show-current").strip())
        self.assertEqual(result["worktree"], "clean")
        self.assertEqual(result["active_objective"], "TEST_OBJECTIVE")
        self.assertEqual(
            result["evidence_aliases"],
            ["TEST_EVIDENCE_V1", "TEST_MODEL_ASSUMPTION_V1"],
        )
        self.assertNotIn("sha256", result)

        code, result, error = self._governance("verify-evidence", "TEST_EVIDENCE_V1")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(result["status"], "verified")
        self.assertNotIn("sha256", result)

        code, result, error = self._governance(
            "verify-evidence", "TEST_MODEL_ASSUMPTION_V1"
        )
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(result["status"], "not_required")
        self.assertIn("MODEL_ASSUMPTION", result["claim_boundary"])

        code, result, error = self._governance(
            "verify-evidence", "TEST_EVIDENCE_V1", "--verbose"
        )
        self.assertEqual((code, error), (0, ""))
        self.assertRegex(result["sha256"], r"^[0-9a-f]{64}$")

        evidence = self.project / "docs" / "test-evidence.json"
        evidence.write_text(evidence.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
        code, _, error = self._governance("verify-evidence", "TEST_EVIDENCE_V1")
        self.assertEqual(code, 2)
        self.assertIn("semantic evidence hash mismatch", error)

    def test_status_distinguishes_missing_stale_and_invalid(self) -> None:
        self.assertEqual(self._status()["status"], "missing")
        self._build()

        memory_path = self.project / "docs" / "PROJECT_MEMORY.md"
        original = memory_path.read_bytes()
        memory_path.write_bytes(original + "新事实\n".encode("utf-8"))
        self.assertEqual(self._status()["status"], "stale")
        memory_path.write_bytes(original)
        self.assertEqual(self._status()["status"], "current")

        output_path = self.output / "CURRENT_STATUS.md"
        output_original = output_path.read_bytes()
        output_path.write_bytes(output_original + b"tamper\n")
        self.assertEqual(self._status()["status"], "invalid")

    def test_clean_worktree_gate_requires_explicit_dirty_override(self) -> None:
        untracked = self.project / "review-note.txt"
        untracked.write_text("reviewed local note\n", encoding="utf-8")
        code, _, error = self._invoke(*self._common_arguments("build"))
        self.assertEqual(code, 2)
        self.assertIn("clean Git worktree", error)

        code, result, error = self._invoke(
            *self._common_arguments("build"),
            "--allow-dirty",
        )
        self.assertEqual((code, error), (0, ""))
        self.assertIsNotNone(result)
        self.assertTrue(result["allow_dirty"])
        self.assertEqual(self._status()["status"], "current")
        self.assertIn(
            "worktree: dirty-allowed",
            (self.output / "PROJECT_MEMORY.md").read_text(encoding="utf-8"),
        )
        untracked.unlink()
        self.assertEqual(self._status()["status"], "stale")

    def test_default_package_is_exact_and_verifies(self) -> None:
        source_bytes = {
            path: path.read_bytes()
            for path in (
                self.project / "docs" / "PROJECT_MEMORY.md",
                self.project / "docs" / "CURRENT_STATUS.md",
                self.project / "docs" / "status" / "current_handoff.md",
                self.project / "docs" / "architecture" / "v2_runtime.md",
                self.history / "HISTORY_CONTEXT.md",
                self.history / "CHAT_INDEX.md",
            )
        }
        result = self._build()
        self.assertEqual(result["source_count"], 6)
        self.assertEqual(result["selected_session_ids"], [])
        actual = {
            path.relative_to(self.output).as_posix() for path in self.output.rglob("*")
        }
        expected = {
            MEMORY.MANIFEST_FILENAME,
            "PROJECT_MEMORY.md",
            "CURRENT_STATUS.md",
            "CURRENT_HANDOFF.md",
            "V2_RUNTIME.md",
            "HISTORY_CONTEXT.md",
            "CHAT_INDEX.md",
        }
        self.assertEqual(actual, expected)
        self.assertNotIn("CHAT_CATALOG.jsonl", actual)
        self.assertFalse(any(path.startswith("summaries/") for path in actual))
        self._verify()
        self.assertEqual(
            {path: path.read_bytes() for path in source_bytes},
            source_bytes,
        )

    def test_selected_session_is_the_only_summary_and_unknown_id_rejects(self) -> None:
        result = self._build("abc-session")
        self.assertEqual(result["source_count"], 7)
        self.assertEqual(result["selected_session_ids"], ["abc-session"])
        self.assertTrue((self.output / "summaries" / "abc-session.md").is_file())
        self._verify()

        code, _, error = self._invoke(
            *self._common_arguments("build"),
            "--session-id",
            "missing-session",
        )
        self.assertEqual(code, 2)
        self.assertIn("source file summaries/missing-session.md is missing", error)

    def test_extended_path_prefix_is_equivalent(self) -> None:
        extended_project = "\\\\?\\" + str(self.project)
        extended_history = "\\\\?\\" + str(self.history)
        extended_output = "\\\\?\\" + str(self.output)
        result = self._invoke(
            "build",
            "--project-root",
            extended_project,
            "--history-root",
            extended_history,
            "--output-root",
            extended_output,
        )
        self.assertEqual((result[0], result[2]), (0, ""))
        self.assertEqual(
            MEMORY.canonical_path_key(extended_project),
            MEMORY.canonical_path_key(self.project),
        )
        self._verify()

    def test_source_change_output_tamper_and_git_state_change_fail_closed(self) -> None:
        self._build()
        memory_path = self.project / "docs" / "PROJECT_MEMORY.md"
        original = memory_path.read_bytes()
        memory_path.write_bytes(original + "新事实\n".encode("utf-8"))
        self._verify(expect_success=False)
        memory_path.write_bytes(original)
        self._verify()

        output_path = self.output / "PROJECT_MEMORY.md"
        output_original = output_path.read_bytes()
        output_path.write_bytes(output_original + b"tamper\n")
        self._verify(expect_success=False)
        code, _, error = self._invoke(*self._common_arguments("build"))
        self.assertEqual(code, 2)
        self.assertIn("output hash mismatch", error)
        output_path.write_bytes(output_original)
        self._verify()

        (self.project / "untracked.txt").write_text("untracked\n", encoding="utf-8")
        self._verify(expect_success=False)

    def test_secret_and_unknown_output_are_rejected(self) -> None:
        memory_path = self.project / "docs" / "PROJECT_MEMORY.md"
        memory_path.write_text("secret = do-not-publish\n", encoding="utf-8")
        code, _, error = self._invoke(*self._common_arguments("build"))
        self.assertEqual(code, 2)
        self.assertIn("secret scan failed", error)
        memory_path.write_text("# 项目记忆\n真实工作区：测试项目。\n", encoding="utf-8")

        self._build()
        (self.output / "unknown.txt").write_text("unexpected\n", encoding="utf-8")
        self._verify(expect_success=False)
        code, _, error = self._invoke(*self._common_arguments("build"))
        self.assertEqual(code, 2)
        self.assertIn("file set mismatch", error)

    def test_incomplete_manifest_and_path_traversal_are_rejected(self) -> None:
        self._build()
        manifest_path = self.output / MEMORY.MANIFEST_FILENAME
        original_manifest = manifest_path.read_bytes()
        manifest_path.write_text("{}\n", encoding="utf-8")
        self._verify(expect_success=False)
        code, _, error = self._invoke(*self._common_arguments("build"))
        self.assertEqual(code, 2)
        self.assertIn("unsupported or missing Sources manifest schema", error)
        manifest_path.write_bytes(original_manifest)
        self._verify()

        code, _, error = self._invoke(
            *self._common_arguments("build"),
            "--session-id",
            "../escape",
        )
        self.assertEqual(code, 2)
        self.assertIn("invalid session ID", error)

    def test_old_project_copy_and_history_noise_are_not_packaged(self) -> None:
        (self.history / "CHAT_CATALOG.jsonl").write_text(
            '{"id":"old"}\n', encoding="utf-8"
        )
        (self.history / "old-project-summary.md").write_text(
            "旧项目副本，不得上传。\n", encoding="utf-8"
        )
        noisy_temp = self.history / ".pytest-tmp" / "generated.txt"
        noisy_temp.parent.mkdir()
        noisy_temp.write_text("test artifact\n", encoding="utf-8")
        self._build()
        package_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in self.output.rglob("*.md")
        )
        self.assertNotIn("旧项目副本", package_text)
        self.assertFalse((self.output / "CHAT_CATALOG.jsonl").exists())
        self.assertFalse((self.output / ".pytest-tmp").exists())


if __name__ == "__main__":
    unittest.main()
