"""Managed v2 run directories and evidence provenance."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..evidence import atomic_write_json, canonical_sha256


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
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], check=True, capture_output=True, text=True).stdout
        dirty = canonical_sha256(diff) if diff else None
        config_hash = canonical_sha256(config_payload)
        run = cls(root, config_hash, commit, dirty)
        atomic_write_json(root / "run_meta.json", {"schema_version": 2, "config_hash": config_hash, "git_commit": commit, "dirty_patch_hash": dirty})
        return run
