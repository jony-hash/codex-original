"""Managed v2 run directories and evidence provenance."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..evidence import atomic_write_json, canonical_sha256, file_sha256


def _worktree_fingerprint(repo_root: Path) -> str | None:
    """Return a stable digest for all non-ignored worktree changes.

    ``git diff HEAD`` covers tracked staged and unstaged changes, but Git does
    not include untracked files in that patch.  Untracked source files can
    affect a run just as much as a tracked edit, so include their relative
    paths and content hashes in the same provenance value.  Ignored files are
    deliberately excluded by ``git ls-files --others --exclude-standard``.
    """
    repo_root = Path(
        subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not status:
        return None
    diff = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked_raw = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
        text=False,
    ).stdout
    untracked = [name for name in untracked_raw.decode("utf-8").split("\0") if name]
    records: list[dict[str, Any]] = []
    for name in sorted(untracked):
        target = repo_root / name
        if not target.is_file():
            records.append({"path": name, "kind": "non_regular"})
            continue
        records.append({"path": name, "sha256": file_sha256(target)})
    return canonical_sha256({"tracked_diff": diff, "untracked": records})


@dataclass(frozen=True)
class V2Run:
    root: Path
    config_hash: str
    git_commit: str
    dirty_patch_hash: str | None

    @classmethod
    def create(cls, *, output_root: str, run_name: str, config_payload: dict[str, object]) -> "V2Run":
        now = datetime.now()
        root = Path(output_root) / f"{now:%Y%m%d}_v2" / "runs" / f"{now:%Y%m%d_%H%M%S_%f}_{run_name}"
        root.mkdir(parents=True, exist_ok=False)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = _worktree_fingerprint(Path.cwd())
        config_hash = canonical_sha256(config_payload)
        run = cls(root, config_hash, commit, dirty)
        atomic_write_json(root / "run_meta.json", {"schema_version": 2, "config_hash": config_hash, "git_commit": commit, "dirty_patch_hash": dirty})
        return run
