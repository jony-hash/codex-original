"""Fixed-exposure, content-bound human-review artifacts for raw-CMOS gates."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ..evidence import atomic_write_json, file_sha256, tensor_sha256


@dataclass(frozen=True)
class ReviewArtifactsV2:
    preview_path: Path
    preview_sha256: str
    template_path: Path
    raw_tensor_sha256: str


def _atomic_png(path: Path, figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp.png", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=150, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_review_artifacts(*, root: str | Path, stem: str, kind: str, raw_cmos: torch.Tensor, exposure_scale: float, labels: torch.Tensor | None = None, targets: torch.Tensor | None = None) -> ReviewArtifactsV2:
    """Render one deterministic representative per label with no adaptive normalization."""
    if raw_cmos.ndim != 4 or raw_cmos.shape[1] != 1 or exposure_scale <= 0.0:
        raise ValueError("raw-CMOS review requires [N, 1, H, W] and positive fixed exposure")
    raw = raw_cmos.detach().cpu()
    if labels is None:
        if raw.shape[0] != 10:
            raise ValueError("unlabelled review requires exactly ten raw-CMOS rows")
        rows = torch.arange(10)
    else:
        labels = labels.detach().cpu().long()
        rows = torch.stack([torch.nonzero(labels == label, as_tuple=False).flatten()[0] for label in range(10)])
    columns = 2 if targets is not None else 1
    figure, axes = plt.subplots(10, columns, figsize=(3 * columns, 20))
    axes = axes.reshape(10, columns)
    for label, row in enumerate(rows.tolist()):
        if targets is not None:
            target = targets[label if targets.shape[0] == 10 else row, 0].detach().cpu()
            axes[label, 0].imshow(target, cmap="gray", vmin=0.0, vmax=1.0)
            axes[label, 0].set_title(f"label {label} target")
        raw_axis = axes[label, columns - 1]
        raw_axis.imshow((raw[row, 0] / exposure_scale).clamp(0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
        raw_axis.set_title(f"label {label} raw CMOS / fixed exposure")
        for axis in axes[label]:
            axis.axis("off")
    figure.tight_layout()
    root = Path(root)
    preview = root / f"{stem}_fixed_exposure.png"
    _atomic_png(preview, figure)
    plt.close(figure)
    raw_hash, preview_hash = tensor_sha256(raw), file_sha256(preview)
    template = root / f"{stem}_manual_review_template.json"
    atomic_write_json(template, {"schema_version": 2, "review_kind": kind, "reviewed_tensor_sha256": raw_hash, "reviewed_preview_sha256": preview_hash, "readable_labels": [], "reviewer": "REQUIRED_REVIEWER"})
    return ReviewArtifactsV2(preview, preview_hash, template, raw_hash)
