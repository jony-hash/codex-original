from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "experiments" / "archive" / "test_governance_v1" / "behavior_mapping.yaml"


def test_governance_archive_mapping_covers_every_archived_only_nodeid() -> None:
    payload = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    entries = payload["entries"]
    nodeids = [entry["nodeid"] for entry in entries]
    assert len(nodeids) == len(set(nodeids))
    replacements = {
        replacement
        for entry in entries
        for replacement in entry["replacement_nodeids"]
    }
    assert all(entry.get("reason_code") for entry in entries)
    assert all(entry.get("historical_reference") for entry in entries)
    assert all(
        "version-specific historical diagnostic or launcher flow" not in entry["reason"]
        for entry in entries
    )
    if replacements:
        collected = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", *sorted(replacements)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert collected.returncode == 0, collected.stdout + collected.stderr
        assert all(replacement in collected.stdout for replacement in replacements)
    expected: set[str] = set()
    trees: dict[Path, dict[str, ast.ClassDef]] = {}
    for entry in entries:
        source, class_name, method_name = entry["nodeid"].split("::")
        path = ROOT / source
        if path not in trees:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            trees[path] = {
                node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
            }
        classes = trees[path]
        assert class_name in classes
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == method_name
            for node in classes[class_name].body
        )
        assert entry["replacement_nodeids"] or entry["disposition"] == "archive_only"
        expected.add(entry["nodeid"])
    active_files = {
        "test_train_stages.py": [ROOT / "tests/application/test_train_stages.py"],
        "test_student_config.py": list((ROOT / "tests/student").glob("test_student_*.py")),
    }
    discovered: set[str] = set()
    for source, paths in active_files.items():
        archive_path = ROOT / "experiments/archive/test_governance_v1/tests" / source
        archive_tree = ast.parse(archive_path.read_text(encoding="utf-8"))
        active_names = set()
        for active_path in paths:
            tree = ast.parse(active_path.read_text(encoding="utf-8"))
            active_names |= {
                node.name
                for candidate in tree.body
                if isinstance(candidate, ast.ClassDef)
                for node in candidate.body
                if isinstance(node, ast.FunctionDef)
            }
            active_names |= {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
        for candidate in archive_tree.body:
            if not isinstance(candidate, ast.ClassDef):
                continue
            for node in candidate.body:
                if isinstance(node, ast.FunctionDef) and node.name not in active_names:
                    discovered.add(
                        f"experiments/archive/test_governance_v1/tests/{source}::"
                        f"{candidate.name}::{node.name}"
                    )
    assert expected == discovered
