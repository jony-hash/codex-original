"""Explicit trusted loader for immutable v1/legacy compatibility formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _trusted_legacy_torch_load(
    path: str | Path, *, map_location: Any = "cpu"
) -> Any:
    """Load a historical pickle only inside the legacy compatibility boundary."""
    return torch.load(path, map_location=map_location, weights_only=False)
