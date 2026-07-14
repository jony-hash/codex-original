#!/usr/bin/env python3
"""Rename managed output version directories by earliest surviving run time."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from ldm_project.output_layout import (
    earliest_run_timestamp,
    extract_prefixed_version,
    version_directory_name,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_SKIP_VERSION_DIRS = {"final_pt", "reports"}
_OUTPUT_PREFIX_RE = re.compile(r"(?<![A-Za-z0-9_])outputs/(?P<name>[^/\s`\"']+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs", type=Path, help="Outputs root to reorganize.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Write a manifest without mutating directories.")
    mode.add_argument("--apply", action="store_true", help="Rename directories, rewrite tracked references, and update the index.")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _manifest_path(root: Path, *, dry_run: bool) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = "dry_run" if dry_run else "apply"
    return root / "reports" / f"output_version_reorder_{suffix}_{timestamp}.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, ensure_ascii=False)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _canonical_version(name: str) -> str:
    prefixed = extract_prefixed_version(name)
    if prefixed is not None:
        return prefixed[1]
    return name


def _count_runs(version_dir: Path) -> int:
    runs_dir = version_dir / "runs"
    if not runs_dir.is_dir():
        return 0
    return sum(1 for path in runs_dir.iterdir() if path.is_dir())


def _version_entry(path: Path) -> dict[str, Any]:
    version = _canonical_version(path.name)
    first_run = earliest_run_timestamp(path)
    return {
        "directory": path.name,
        "version": version,
        "path": path,
        "runs_path": path / "runs",
        "run_count": _count_runs(path),
        "earliest_run_timestamp": first_run,
        "earliest_run_date": first_run.split("_", 1)[0] if first_run is not None else None,
    }


def _version_order_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    timestamp = entry["earliest_run_timestamp"] or "99999999_999999"
    return (timestamp, entry["version"], entry["directory"])


def build_version_index(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in root.iterdir() if root.exists() else ():
        if not path.is_dir() or path.name in _SKIP_VERSION_DIRS:
            continue
        entries.append(_version_entry(path))
    return sorted(entries, key=_version_order_key)


def write_version_index(root: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    index = build_version_index(root)
    json_path = root / "version_index.json"
    markdown_path = root / "version_index.md"
    payload = {"schema_version": 2, "versions_oldest_first": index}
    _write_text(json_path, json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Output Versions",
        "",
        "| Directory | Version | Earliest Run | Runs | Path |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for entry in index:
        lines.append(
            "| `{directory}` | `{version}` | `{earliest}` | {runs} | `{path}` |".format(
                directory=entry["directory"],
                version=entry["version"],
                earliest=entry["earliest_run_timestamp"] or "missing",
                runs=entry["run_count"],
                path=entry["runs_path"],
            )
        )
    _write_text(markdown_path, "\n".join(lines) + "\n")
    return json_path, markdown_path, index


def _resolve_version_index(root: Path, *, dry_run: bool) -> tuple[Path, Path, list[dict[str, Any]]]:
    if dry_run:
        return root / "version_index.json", root / "version_index.md", build_version_index(root)
    return write_version_index(root)


def _tracked_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [repo_root / line for line in result.stdout.splitlines() if line]


def _collect_replacements(version_actions: list[dict[str, Any]]) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for action in version_actions:
        if action["status"] != "planned_rename":
            continue
        replacements.append((f"outputs/{action['source_name']}", f"outputs/{action['target_name']}"))
    return replacements


def _replace_outputs_reference(text: str, old: str, new: str) -> tuple[str, int]:
    count = 0

    def _swap(match: re.Match[str]) -> str:
        nonlocal count
        if match.group(0) != old:
            return match.group(0)
        count += 1
        return new

    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?=(?:/|[`\"'\s)]|$))")
    updated = pattern.sub(_swap, text)
    return updated, count


def rewrite_tracked_references(
    replacements: list[tuple[str, str]],
    *,
    tracked_files: list[Path],
    dry_run: bool,
) -> list[dict[str, Any]]:
    actions = []
    for path in tracked_files:
        try:
            original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        updated = original
        replacements_in_file = []
        for old, new in replacements:
            updated, count = _replace_outputs_reference(updated, old, new)
            if count == 0:
                continue
            replacements_in_file.append({"old": old, "new": new, "count": count})
        if not replacements_in_file:
            continue
        actions.append(
            {
                "file": path,
                "status": "planned_rewrite" if dry_run else "rewritten",
                "replacements": replacements_in_file,
            }
        )
        if not dry_run:
            path.write_text(updated, encoding="utf-8")
    return actions


def _plan_version_dir(path: Path, root: Path) -> dict[str, Any]:
    version = _canonical_version(path.name)
    first_run = earliest_run_timestamp(path)
    if not (path / "runs").is_dir():
        return {
            "source_name": path.name,
            "source_path": path,
            "version": version,
            "status": "blocker_missing_runs_dir",
        }
    if first_run is None:
        return {
            "source_name": path.name,
            "source_path": path,
            "version": version,
            "status": "blocker_missing_timestamped_runs",
        }
    target_name = version_directory_name(version, date_prefix=first_run.split("_", 1)[0])
    target_path = root / target_name
    status = "already_named" if path.name == target_name else "planned_rename"
    return {
        "source_name": path.name,
        "source_path": path,
        "version": version,
        "earliest_run_timestamp": first_run,
        "target_name": target_name,
        "target_path": target_path,
        "status": status,
    }


def plan_reorder(root: Path) -> list[dict[str, Any]]:
    actions = [
        _plan_version_dir(path, root)
        for path in root.iterdir()
        if path.is_dir() and path.name not in _SKIP_VERSION_DIRS
    ]
    actions.sort(
        key=lambda item: (
            item.get("earliest_run_timestamp") or "99999999_999999",
            item["version"],
            item["source_name"],
        )
    )

    target_to_sources: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        target_name = action.get("target_name")
        if target_name is None:
            continue
        target_to_sources.setdefault(target_name, []).append(action)

    for target_name, grouped in target_to_sources.items():
        if len(grouped) > 1:
            for action in grouped:
                action["status"] = "conflict_duplicate_target"
                action["conflict_target_name"] = target_name

    for action in actions:
        if action["status"] != "planned_rename":
            continue
        target_path = Path(action["target_path"])
        source_path = Path(action["source_path"])
        if target_path.exists() and target_path.resolve() != source_path.resolve():
            action["status"] = "conflict_target_exists"

    return actions


def _apply_renames(actions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    updated = []
    for action in actions:
        copied = dict(action)
        if copied["status"] != "planned_rename":
            updated.append(copied)
            continue
        source = Path(copied["source_path"])
        target = Path(copied["target_path"])
        source.rename(target)
        copied["status"] = "renamed"
        updated.append(copied)
    return updated


def _counts(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    materialized = list(items)
    return {
        status: sum(1 for item in materialized if item["status"] == status)
        for status in sorted({item["status"] for item in materialized})
    }


def reorganize_outputs(
    root: Path,
    *,
    dry_run: bool,
    tracked_files: list[Path] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    root = root.resolve()
    version_actions = plan_reorder(root)
    blockers = [
        action
        for action in version_actions
        if action["status"].startswith("blocker_") or action["status"].startswith("conflict_")
    ]
    tracked = tracked_files if tracked_files is not None else _tracked_files(repo_root)
    replacements = _collect_replacements(version_actions)

    applied_version_actions = version_actions
    if not dry_run:
        if blockers:
            raise RuntimeError(f"refusing to rename outputs with blockers: {len(blockers)}")
        applied_version_actions = _apply_renames(version_actions)

    rewrite_actions = rewrite_tracked_references(replacements, tracked_files=tracked, dry_run=dry_run)
    index_json, index_md, version_index = _resolve_version_index(root, dry_run=dry_run)
    manifest = {
        "schema_version": 3,
        "action": "reorder_by_time",
        "dry_run": dry_run,
        "root": root,
        "version_index_json": index_json,
        "version_index_markdown": index_md,
        "versions_oldest_first": version_index,
        "version_actions": applied_version_actions,
        "rewrite_actions": rewrite_actions,
        "counts": {
            **{f"versions_{key}": value for key, value in _counts(applied_version_actions).items()},
            "rewrite_files": len(rewrite_actions),
            "rewrite_replacements": sum(
                sum(replacement["count"] for replacement in action["replacements"])
                for action in rewrite_actions
            ),
        },
    }
    path = _manifest_path(root, dry_run=dry_run)
    _write_json(path, manifest)
    manifest["manifest_path"] = path
    return manifest


def main() -> int:
    args = _parse_args()
    manifest = reorganize_outputs(args.root, dry_run=args.dry_run)
    print(f"manifest={manifest['manifest_path']}")
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
