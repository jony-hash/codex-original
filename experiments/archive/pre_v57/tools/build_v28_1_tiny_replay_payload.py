#!/usr/bin/env python3
"""Build deterministic v28.1 same-batch replay distillation payload."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


import torch


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v28_1_tiny_replay_payload_{timestamp}.pt"


def _require_output(path: Path) -> None:
    normalized = path.as_posix()
    allowed = normalized.startswith("outputs/reports/") or normalized.startswith("outputs/20260702_v28_1/artifacts/")
    if path.is_absolute() or not allowed:
        raise ValueError(
            "v28.1 replay payload output must be relative under outputs/reports/ "
            "or outputs/20260702_v28_1/artifacts/"
        )


def _torch_load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"distillation payload must be a dict: {path}")
    for key in ("codes", "class_labels", "teacher_camera"):
        if key not in payload or not torch.is_tensor(payload[key]):
            raise ValueError(f"distillation payload is missing required tensor: {key}")
    return payload


def _sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _select_indices(labels: torch.Tensor, *, per_class: int, num_classes: int) -> list[int]:
    selected: list[int] = []
    for label in range(num_classes):
        matches = torch.nonzero(labels.cpu().long() == label, as_tuple=False).flatten().tolist()
        if len(matches) < per_class:
            raise ValueError(f"class {label} has {len(matches)} samples, need {per_class}")
        selected.extend(int(index) for index in matches[:per_class])
    return selected


def build_tiny_replay_payload(
    *,
    source: Path,
    output: Path | None = None,
    per_class: int = 5,
    num_classes: int = 10,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_output(output_path)
    if per_class <= 0:
        raise ValueError(f"per_class must be positive, got {per_class}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    payload = _torch_load(source)
    labels = payload["class_labels"].detach().cpu().long()
    source_size = int(labels.shape[0])
    selected = _select_indices(labels, per_class=per_class, num_classes=num_classes)
    index_tensor = torch.tensor(selected, dtype=torch.long)
    replay: dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == source_size:
            replay[key] = value.index_select(0, index_tensor).clone()
        else:
            replay[key] = value
    source_index_tensor = payload.get("readability_filter_indices")
    if torch.is_tensor(source_index_tensor) and source_index_tensor.ndim == 1 and int(source_index_tensor.shape[0]) == source_size:
        original_indices = source_index_tensor.index_select(0, index_tensor).detach().cpu().long()
    else:
        original_indices = index_tensor.detach().cpu().long()
    replay["readability_filter_indices"] = original_indices.clone()
    reference = replay["teacher_camera"].detach().cpu()
    metadata = {
        "diagnostic_only": True,
        "version": "v28.1",
        "source_dataset": source.as_posix(),
        "source_size": source_size,
        "per_class": per_class,
        "num_classes": num_classes,
        "selected_source_positions": selected,
        "selected_source_indices": [int(item) for item in original_indices.tolist()],
        "selected_labels": [int(item) for item in replay["class_labels"].detach().cpu().long().tolist()],
        "reference_key": "teacher_camera",
        "reference_sha256": [_sha256_tensor(reference[index]) for index in range(int(reference.shape[0]))],
        "payload_sha256": _sha256_tensor(reference),
        "acceptance_eligible": False,
    }
    replay["v28_1_replay_metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(replay, output_path)
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output": output_path.as_posix(), "metadata": metadata, "sidecar": sidecar.as_posix()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Readable train distillation dataset.")
    parser.add_argument("--output", type=Path, default=None, help="Replay payload path.")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_tiny_replay_payload(
        source=args.source,
        output=args.output,
        per_class=args.per_class,
        num_classes=args.num_classes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
