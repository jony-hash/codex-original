"""Path migration and YAML loading for the public v2 configuration facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import yaml

if TYPE_CHECKING:
    from ..config import V2Config


def _resolve_config_path(path: str | Path) -> Path:
    requested = Path(path)
    if requested.is_file():
        return requested
    manifest_candidates = [
        Path("configs/v2/path_migration_manifest.yaml"),
        Path(__file__).resolve().parents[4] / "configs/v2/path_migration_manifest.yaml",
    ]
    for manifest_path in manifest_candidates:
        if not manifest_path.is_file():
            continue
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        for entry in payload.get("entries", []):
            if entry.get("old_path") == requested.as_posix():
                target = Path(str(entry["new_path"]))
                if not target.is_file():
                    raise FileNotFoundError(f"migrated v2 config target is missing: {target}")
                warnings.warn(
                    f"deprecated v2 config path {requested}; use {target}",
                    UserWarning,
                    stacklevel=3,
                )
                return target
        break
    raise FileNotFoundError(f"v2 config does not exist: {requested}")


def load_config(path: str | Path) -> V2Config:
    """Load via the public model without importing an execution runtime."""
    from ..config import V2Config

    resolved = _resolve_config_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("v2 configuration must be a YAML mapping")
    return V2Config.model_validate(payload)
