"""Materialize pointwise predictions for the frozen FOSEA validation corpus.

This task replays only the already processed validation views through the
registered frozen evaluator on CPU.  It does not run the optical model, train
the evaluator, or perform the downstream OSERA statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator


TASK_ID = "H-LOMA-GEN-S2-OSERA-VPIM"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_output_semantic_evaluator_prediction_identity_v1"
ARTIFACT_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_prediction_identity_manifest.v1"
)
PREDICTION_SCHEMA = "lightgen_scaled_chen_output_semantic_evaluator_predictions.v1"
LOGITS_SCHEMA = "lightgen_scaled_chen_output_semantic_evaluator_logits.v1"
NULL_SCHEMA = "lightgen_scaled_chen_output_semantic_evaluator_null_permutation.v1"
SAMPLE_ORDER_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_sample_order.v1"
)
AGGREGATE_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_aggregate_replay.v1"
)
NULL_CONTRACT_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_null_identity.v1"
)
STATE_SCHEMA = "lightgen_scaled_chen_output_semantic_evaluator_state_isolation.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_output_semantic_evaluator_vpim_report.v1"

FOSEA_CLASSIFICATION = "FROZEN_OUTPUT_SEMANTIC_EVALUATOR_NOT_QUALIFIED"
VPIM_CLASSIFICATION = "OSERA_VALIDATION_PREDICTION_IDENTITY_READY"
REPLAY_GATE = "FROZEN_PREDICTION_REPLAY_PASS"
AGGREGATE_GATE = "OSERA_PREDICTION_AGGREGATE_EXACT_MATCH"
STATE_GATE = "OSERA_VPIM_STATE_ISOLATION_PASS"
NEXT_ACTION = "IMPLEMENT_OUTPUT_SEMANTIC_EVALUATOR_REQUALIFICATION_AUTHORITY"

FOSEA_MANIFEST_SHA256 = (
    "c8e144da2dd02e9f4ee78f7e8829782800651ec835adb1ae6fd781b1921c1d0f"
)
CHECKPOINT_SHA256 = (
    "a6ac0b4644a459bad21bc7cf3071465d5306af31ad2081fe7acb6f81e87c47e6"
)
CORPUS_SHA256 = (
    "787b1b4e256f54ccebb55bb5582308f92148dc595d4cab4f632d3e87d72907b7"
)
OUTPUT_IDENTITY_SHA256 = (
    "db8f03d977a55e2c82ec557b8b359f55f27048c413e31e04c2f76134dfcbb04d"
)
QUALIFICATION_SHA256 = (
    "2fed6db88c0b5003458077198375f746eb995a88be4a9350c2d70cd38acb8568"
)
DATA_IDENTITY_SHA256 = (
    "8ae23bf1f247124a1f1f9689a2a4cfb507860c49704d334adec158808d093dc4"
)
OUTPUT_CONTRACT_SHA256 = (
    "515137bd66c71ba6b1b1f4e21fa99d4f2527aca286e11ff079b4b7dfd2fbce56"
)
TRAINING_CONTRACT_SHA256 = (
    "e84ad034e5fc726810f8813409dd438242d38aafdae735bfba83c14c53487f25"
)
VALIDATION_INDICES_SHA256 = (
    "5370b35f304a779cde7f978df6392b9974faaf4296e9edf75f9edf8202db02fa"
)
VALIDATION_LABELS_SHA256 = (
    "13d90706eef49c3c9f17ccd39c002dc18a79eca1378e780548ea6b29adc6ec29"
)

VALIDATION_COUNT = 5000
VALIDATION_SHAPE = (VALIDATION_COUNT, 1, 28, 28)
NULL_SEED = 2053
BOOTSTRAP_SEED = 1236
BOOTSTRAP_REPLICATES = 2000
LOGIT_TOLERANCE = 1.0e-7
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / (
    "configs/research/scaled_chen/"
    "lightgen_scaled_chen_output_semantic_evaluator_prediction_identity_v1.yaml"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / (
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_output_semantic_evaluator_prediction_identity_v1_attempt_01"
)


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else REPO_ROOT / target


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _require_hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value.lower()) is None:
        raise ValueError(f"OSERA_VPIM_AUTHORITY_BINDING_FAILED: invalid {name}")
    return value.lower()


def _load_json(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"OSERA_VPIM_AUTHORITY_BINDING_FAILED: {role}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"OSERA_VPIM_AUTHORITY_BINDING_FAILED: {role} schema")
    return payload


def next_write_once_output(base_path: str | Path) -> Path:
    """Return the first unused numbered output path without touching disk."""
    requested = Path(base_path)
    match = re.fullmatch(r"(.+)_attempt_(\d+)", requested.name)
    if match is None:
        raise ValueError("write-once output must end with _attempt_<number>")
    prefix, number_text = match.groups()
    number = int(number_text)
    while True:
        candidate = requested.parent / f"{prefix}_attempt_{number:02d}"
        if not candidate.exists():
            return candidate
        number += 1


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VPIM config must be a YAML mapping")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("task_id") != TASK_ID:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: config schema")
    _assert_config(payload)
    return payload


def _assert_config(config: Mapping[str, Any]) -> None:
    source = config.get("source")
    validation = config.get("validation")
    replay = config.get("replay")
    if not isinstance(source, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: config sections")
    if not isinstance(replay, Mapping):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: replay section")

    expected_sources = {
        "fosea_manifest_sha256": FOSEA_MANIFEST_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "output_identity_manifest_sha256": OUTPUT_IDENTITY_SHA256,
        "qualification_metrics_sha256": QUALIFICATION_SHA256,
        "data_identity_manifest_sha256": DATA_IDENTITY_SHA256,
        "output_domain_contract_sha256": OUTPUT_CONTRACT_SHA256,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
    }
    for name, expected in expected_sources.items():
        if source.get(name) != expected:
            raise ValueError(f"OSERA_VPIM_AUTHORITY_BINDING_FAILED: config {name}")

    if validation.get("count") != VALIDATION_COUNT:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: validation count")
    if tuple(validation.get("shape", ())) != VALIDATION_SHAPE:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: validation shape")
    if validation.get("indices_sha256") != VALIDATION_INDICES_SHA256:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: validation indices")
    if validation.get("labels_sha256") != VALIDATION_LABELS_SHA256:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: validation labels")
    if replay.get("null_seed") != NULL_SEED:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: null seed")
    if replay.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: bootstrap seed")
    if replay.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: bootstrap replicates")
    if replay.get("device") != "cpu" or replay.get("shuffle") is not False:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: CPU replay contract")


def _validate_source_manifest(
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    if manifest.get("classification") != FOSEA_CLASSIFICATION:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: FOSEA classification")
    evaluator = manifest.get("evaluator")
    storage = manifest.get("artifact_storage")
    dataset = manifest.get("dataset")
    if not isinstance(evaluator, Mapping) or not isinstance(storage, Mapping):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: FOSEA manifest schema")
    if not isinstance(dataset, Mapping):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: FOSEA dataset")
    if evaluator.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: checkpoint reference")
    if storage.get("corpus_sha256") != CORPUS_SHA256:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: corpus reference")
    if evaluator.get("schema") != "frozen_mnist_evaluator_v1":
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: evaluator schema")
    if dataset.get("validation_count") != VALIDATION_COUNT:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: validation reference")
    if dataset.get("s2_confirmation_consumed") is not False:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: confirmation consumed")
    if dataset.get("s2_selection_consumed") is not False:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: selection consumed")
    for name, expected in {
        "fosea_manifest_sha256": FOSEA_MANIFEST_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "corpus_sha256": CORPUS_SHA256,
    }.items():
        _require_hex(source[name], name)
        if source[name] != expected:
            raise ValueError(f"OSERA_VPIM_AUTHORITY_BINDING_FAILED: {name}")


def _load_corpus(
    path: Path,
    expected_sha256: str,
) -> dict[str, torch.Tensor | str]:
    payload = safe_torch_load(path, expected_sha256=expected_sha256, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: corpus schema")
    expected_keys = {
        "schema",
        "fit_views",
        "validation_views",
        "fit_labels",
        "validation_labels",
        "fit_indices",
        "validation_indices",
    }
    if set(payload) != expected_keys or payload.get("schema") != (
        "lightgen_scaled_chen_frozen_output_domain_corpus.v1"
    ):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: corpus keys")
    validation_views = payload["validation_views"]
    validation_indices = payload["validation_indices"]
    validation_labels = payload["validation_labels"]
    if (
        not torch.is_tensor(validation_views)
        or tuple(validation_views.shape) != VALIDATION_SHAPE
        or validation_views.dtype != torch.float32
        or not torch.isfinite(validation_views).all()
        or not torch.is_tensor(validation_indices)
        or validation_indices.dtype != torch.long
        or tuple(validation_indices.shape) != (VALIDATION_COUNT,)
        or not torch.is_tensor(validation_labels)
        or validation_labels.dtype != torch.long
        or tuple(validation_labels.shape) != (VALIDATION_COUNT,)
    ):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: validation corpus")
    if tensor_sha256(validation_indices) != VALIDATION_INDICES_SHA256:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: validation index hash")
    if tensor_sha256(validation_labels) != VALIDATION_LABELS_SHA256:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: validation label hash")
    expected_indices = torch.arange(50000, 55000, dtype=torch.long)
    if not torch.equal(validation_indices, expected_indices):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: validation index order")
    return payload


def validate_record_order(
    records: list[Mapping[str, Any]],
    validation_indices: torch.Tensor,
    validation_labels: torch.Tensor,
) -> bool:
    """Check the identity/order portion of a validation identity manifest."""
    if len(records) != validation_indices.numel() or validation_labels.numel() != len(records):
        return False
    for record, source_index, label in zip(
        records, validation_indices.tolist(), validation_labels.tolist()
    ):
        if (
            record.get("source_dataset") != "mnist_train"
            or record.get("source_index") != int(source_index)
            or record.get("label") != int(label)
        ):
            return False
    return True


def _validate_identity_manifest(
    identity: Mapping[str, Any],
    corpus: Mapping[str, torch.Tensor | str],
    expected_sha256: str,
) -> dict[str, Any]:
    if identity.get("schema") != "lightgen_scaled_chen_frozen_output_identity_manifest.v1":
        raise ValueError("OSERA_VPIM_SAMPLE_IDENTITY_MISMATCH: identity schema")
    if identity.get("corpus_sha256") != CORPUS_SHA256:
        raise ValueError("OSERA_VPIM_SAMPLE_IDENTITY_MISMATCH: corpus reference")
    if identity.get("record_count") != 55000:
        raise ValueError("OSERA_VPIM_SAMPLE_IDENTITY_MISMATCH: record count")
    if identity.get("raw_camera_persisted") is not False:
        raise ValueError("OSERA_VPIM_SAMPLE_IDENTITY_MISMATCH: raw camera policy")
    if identity.get("s2_confirmation_records") != 0 or identity.get("s2_selection_records") != 0:
        raise ValueError("OSERA_VPIM_SAMPLE_IDENTITY_MISMATCH: S2 records")
    records_value = identity.get("records")
    if not isinstance(records_value, list):
        raise ValueError("OSERA_VPIM_SAMPLE_IDENTITY_MISMATCH: records")
    validation_records = [
        record
        for record in records_value
        if isinstance(record, Mapping) and record.get("split") == "validation"
    ]
    validation_indices = corpus["validation_indices"]
    validation_labels = corpus["validation_labels"]
    validation_views = corpus["validation_views"]
    if not torch.is_tensor(validation_indices) or not torch.is_tensor(validation_labels):
        raise ValueError("OSERA_VPIM_SAMPLE_IDENTITY_MISMATCH: corpus identity tensors")
    if not isinstance(validation_views, torch.Tensor):
        raise ValueError("OSERA_VPIM_SAMPLE_IDENTITY_MISMATCH: corpus views")
    if len(validation_records) != VALIDATION_COUNT or not validate_record_order(
        validation_records, validation_indices, validation_labels
    ):
        raise ValueError("OSERA_VPIM_SAMPLE_IDENTITY_MISMATCH: source order")

    processed_hashes: list[str] = []
    raw_hashes: list[str] = []
    for row_index, record in enumerate(validation_records):
        processed_hash = _require_hex(record.get("processed_view_sha256"), "view hash")
        raw_hash = _require_hex(record.get("raw_camera_sha256"), "raw hash")
        expected_view_hash = tensor_sha256(validation_views[row_index])
        if processed_hash != expected_view_hash:
            raise ValueError(
                "OSERA_VPIM_SAMPLE_IDENTITY_MISMATCH: processed view hash"
            )
        processed_hashes.append(processed_hash)
        raw_hashes.append(raw_hash)

    source_index_hash = tensor_sha256(validation_indices)
    label_hash = tensor_sha256(validation_labels)
    sample_order_hash = canonical_sha256(
        [
            {
                "row_index": row_index,
                "source_dataset": record["source_dataset"],
                "source_index": record["source_index"],
            }
            for row_index, record in enumerate(validation_records)
        ]
    )
    return {
        "manifest_sha256": expected_sha256,
        "validation_records": validation_records,
        "source_index_hash": source_index_hash,
        "label_hash": label_hash,
        "sample_order_hash": sample_order_hash,
        "processed_view_hash_sequence": canonical_sha256(processed_hashes),
        "raw_camera_hash_sequence": canonical_sha256(raw_hashes),
    }


def evaluator_state_digest(model: FrozenMNISTEvaluator) -> str:
    return canonical_sha256(
        {
            name: tensor_sha256(value.detach().cpu())
            for name, value in model.state_dict().items()
        }
    )


def replay_predictions(
    model: FrozenMNISTEvaluator,
    validation_views: torch.Tensor,
    *,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen evaluator on CPU in stored order, without gradients."""
    if validation_views.device.type != "cpu":
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: replay device")
    if batch_size <= 0:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: batch size")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: evaluator device")
    model.eval()
    logits_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, validation_views.shape[0], batch_size):
            _, logits = model(validation_views[start : start + batch_size])
            if (
                logits.ndim != 2
                or logits.shape[1] != 10
                or not torch.isfinite(logits).all()
            ):
                raise RuntimeError("OSERA_PREDICTION_REPLAY_INVALID: logits")
            logits_parts.append(logits.detach().cpu())
    logits = torch.cat(logits_parts, dim=0)
    return logits, logits.argmax(dim=1)


def aggregate_predictions(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, Any]:
    if (
        labels.ndim != 1
        or predictions.ndim != 1
        or labels.shape != predictions.shape
        or labels.dtype != torch.long
        or predictions.dtype != torch.long
        or labels.numel() == 0
        or int(labels.min()) < 0
        or int(labels.max()) > 9
        or int(predictions.min()) < 0
        or int(predictions.max()) > 9
    ):
        raise ValueError("OSERA_PREDICTION_AGGREGATE_MISMATCH: prediction tensors")
    confusion = torch.zeros((10, 10), dtype=torch.long)
    confusion.index_put_((labels, predictions), torch.ones_like(labels), accumulate=True)
    counts = torch.bincount(labels, minlength=10)
    correct = confusion.diag()
    per_class = correct.to(torch.float64) / counts.to(torch.float64)
    correctness = predictions.eq(labels)
    return {
        "sample_count": int(labels.numel()),
        "overall_accuracy": float(correctness.to(torch.float64).mean().item()),
        "error_count": int((~correctness).sum().item()),
        "class_counts": [int(value) for value in counts.tolist()],
        "correct_counts": [int(value) for value in correct.tolist()],
        "per_class_accuracy": [float(value) for value in per_class.tolist()],
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_sha256": tensor_sha256(confusion),
    }


def build_null_permutation(labels: torch.Tensor, *, seed: int = NULL_SEED) -> torch.Tensor:
    if labels.ndim != 1 or labels.dtype != torch.long or labels.device.type != "cpu":
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: null labels")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(labels.numel(), generator=generator)


def _aggregate_matches_historical(
    actual: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> bool:
    validation = historical.get("validation")
    if not isinstance(validation, Mapping):
        return False
    confusion = validation.get("confusion_matrix")
    if not isinstance(confusion, list):
        return False
    expected_errors = int(validation["sample_count"]) - sum(
        int(confusion[index][index]) for index in range(10)
    )
    return (
        actual["sample_count"] == validation["sample_count"]
        and actual["overall_accuracy"] == validation["overall_accuracy"]
        and actual["error_count"] == expected_errors
        and actual["class_counts"] == validation["class_counts"]
        and actual["correct_counts"]
        == [int(confusion[index][index]) for index in range(10)]
        and actual["per_class_accuracy"] == validation["per_class_accuracy"]
        and actual["confusion_matrix"] == confusion
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _artifact_refs(root: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        refs[relative] = {
            "path": relative,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return refs


def _assert_final_tree(root: Path) -> None:
    expected = {
        "aggregate_replay.json",
        "compact_report.json",
        "null_identity_contract.json",
        "null_permutation_indices.pt",
        "prediction_identity_manifest.json",
        "sample_order_authority.json",
        "state_isolation.json",
        "validation_logits.pt",
        "validation_predictions.jsonl",
    }
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected:
        raise RuntimeError(f"OSERA_VPIM_MATERIALIZATION_INCOMPLETE: {sorted(actual)}")
    for path in root.iterdir():
        if path.name.startswith(".") or path.suffix.lower() in {".tmp", ".tif", ".tiff"}:
            raise RuntimeError(f"OSERA_VPIM_MATERIALIZATION_INCOMPLETE: {path.name}")


def _source_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    source = config["source"]
    return {
        name: _resolve(source[name])
        for name in (
            "fosea_manifest",
            "checkpoint",
            "corpus",
            "output_identity_manifest",
            "qualification_metrics",
            "data_identity_manifest",
            "output_domain_contract",
            "training_contract",
        )
    }


def run(
    config_path: str | Path = CONFIG_PATH,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    config = load_config(config_file)
    config_hash = file_sha256(config_file)
    config_canonical_hash = canonical_sha256(config)
    source = config["source"]
    paths = _source_paths(config)

    fosea_manifest = _load_json(
        paths["fosea_manifest"], FOSEA_MANIFEST_SHA256, "FOSEA manifest"
    )
    _validate_source_manifest(fosea_manifest, source)
    qualification = _load_json(
        paths["qualification_metrics"], QUALIFICATION_SHA256, "qualification metrics"
    )
    if qualification.get("classification") != FOSEA_CLASSIFICATION:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: qualification")
    _load_json(
        paths["data_identity_manifest"], DATA_IDENTITY_SHA256, "data identity"
    )
    _load_json(
        paths["output_domain_contract"], OUTPUT_CONTRACT_SHA256, "output contract"
    )
    _load_json(
        paths["training_contract"], TRAINING_CONTRACT_SHA256, "training contract"
    )
    identity = _load_json(
        paths["output_identity_manifest"], OUTPUT_IDENTITY_SHA256, "output identity"
    )
    corpus = _load_corpus(paths["corpus"], CORPUS_SHA256)
    identity_info = _validate_identity_manifest(
        identity,
        corpus,
        OUTPUT_IDENTITY_SHA256,
    )

    checkpoint_payload = safe_torch_load(
        paths["checkpoint"], expected_sha256=CHECKPOINT_SHA256, map_location="cpu"
    )
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: checkpoint payload")
    if (
        checkpoint_payload.get("schema")
        != "lightgen_scaled_chen_frozen_output_evaluator_checkpoint.v1"
        or checkpoint_payload.get("architecture_kind") != "frozen_mnist_evaluator_v1"
        or checkpoint_payload.get("validation_accuracy") != 0.9712
        or checkpoint_payload.get("fit_split_only") is not True
        or checkpoint_payload.get("s2_confirmation_used") is not False
        or checkpoint_payload.get("s2_selection_used") is not False
    ):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: checkpoint contract")
    for key, expected in {
        "data_identity_manifest_sha256": DATA_IDENTITY_SHA256,
        "output_domain_contract_sha256": OUTPUT_CONTRACT_SHA256,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
    }.items():
        if checkpoint_payload.get(key) != expected:
            raise ValueError(f"OSERA_VPIM_AUTHORITY_BINDING_FAILED: checkpoint {key}")
    if checkpoint_payload.get("qualification") != FOSEA_CLASSIFICATION:
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: checkpoint qualification")

    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: checkpoint state")
    model = FrozenMNISTEvaluator()
    model.load_state_dict(state_dict, strict=True)
    model.freeze()
    before_checkpoint_hash = file_sha256(paths["checkpoint"])
    before_state_digest = evaluator_state_digest(model)

    validation_views = corpus["validation_views"]
    validation_labels = corpus["validation_labels"]
    if not isinstance(validation_views, torch.Tensor) or not isinstance(
        validation_labels, torch.Tensor
    ):
        raise ValueError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: validation tensors")

    batch_size = int(config["replay"]["batch_size"])
    first_logits, first_predictions = replay_predictions(
        model, validation_views, batch_size=batch_size
    )
    second_logits, second_predictions = replay_predictions(
        model, validation_views, batch_size=batch_size
    )
    max_abs_difference = float((first_logits - second_logits).abs().max().item())
    replay_pass = bool(
        torch.equal(first_predictions, second_predictions)
        and max_abs_difference <= float(config["replay"]["logit_tolerance"])
    )
    if not replay_pass:
        raise RuntimeError("OSERA_PREDICTION_REPLAY_INVALID")

    aggregate = aggregate_predictions(validation_labels, first_predictions)
    aggregate_pass = _aggregate_matches_historical(aggregate, qualification)
    if not aggregate_pass:
        raise RuntimeError("OSERA_PREDICTION_AGGREGATE_MISMATCH")

    correctness = first_predictions.eq(validation_labels)
    null_permutation = build_null_permutation(
        validation_labels, seed=int(config["replay"]["null_seed"])
    )
    permuted_labels = validation_labels.index_select(0, null_permutation)
    class_counts_preserved = torch.equal(
        torch.bincount(validation_labels, minlength=10),
        torch.bincount(permuted_labels, minlength=10),
    )
    if not class_counts_preserved:
        raise RuntimeError("OSERA_VPIM_MATERIALIZATION_INCOMPLETE: null counts")

    after_checkpoint_hash = file_sha256(paths["checkpoint"])
    after_state_digest = evaluator_state_digest(model)
    state_pass = bool(
        before_checkpoint_hash == after_checkpoint_hash
        and before_state_digest == after_state_digest
    )
    if not state_pass:
        raise RuntimeError("OSERA_VPIM_AUTHORITY_BINDING_FAILED: evaluator state changed")

    records = []
    for row_index, (identity_record, prediction, is_correct) in enumerate(
        zip(identity_info["validation_records"], first_predictions.tolist(), correctness.tolist())
    ):
        records.append(
            {
                "row_index": row_index,
                "source_dataset": identity_record["source_dataset"],
                "source_index": identity_record["source_index"],
                "ground_truth_label": identity_record["label"],
                "prediction": int(prediction),
                "correct": bool(is_correct),
                "processed_view_sha256": identity_record["processed_view_sha256"],
                "raw_camera_sha256": identity_record["raw_camera_sha256"],
            }
        )

    prediction_hash = tensor_sha256(first_predictions)
    correctness_hash = tensor_sha256(correctness)
    logits_hash = tensor_sha256(first_logits)
    null_permutation_hash = tensor_sha256(null_permutation)
    prediction_file_name = "validation_predictions.jsonl"
    logits_file_name = "validation_logits.pt"
    null_file_name = "null_permutation_indices.pt"

    requested_output = _resolve(output_path or config["output"]["path"])
    actual_output = next_write_once_output(requested_output)
    actual_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{actual_output.name}.", dir=str(actual_output.parent))
    )
    try:
        _write_jsonl(temporary / prediction_file_name, records)
        atomic_torch_save_new(
            temporary / logits_file_name,
            {
                "schema": LOGITS_SCHEMA,
                "logits": first_logits,
                "sample_order_hash": identity_info["sample_order_hash"],
                "checkpoint_sha256": CHECKPOINT_SHA256,
            },
            artifact_kind="validation logits",
        )
        atomic_torch_save_new(
            temporary / null_file_name,
            {
                "schema": NULL_SCHEMA,
                "permutation_indices": null_permutation,
                "seed": NULL_SEED,
                "sample_order_hash": identity_info["sample_order_hash"],
            },
            artifact_kind="null permutation",
        )

        sample_order_payload = {
            "schema": SAMPLE_ORDER_SCHEMA,
            "source_identity_manifest_sha256": OUTPUT_IDENTITY_SHA256,
            "corpus_sha256": CORPUS_SHA256,
            "validation_count": VALIDATION_COUNT,
            "source_index_hash": identity_info["source_index_hash"],
            "label_hash": identity_info["label_hash"],
            "sample_order_hash": identity_info["sample_order_hash"],
            "processed_view_hash_sequence": identity_info["processed_view_hash_sequence"],
            "raw_camera_hash_sequence": identity_info["raw_camera_hash_sequence"],
            "order_verified": True,
        }
        atomic_write_new_json(
            temporary / "sample_order_authority.json",
            sample_order_payload,
            artifact_kind="sample order authority",
        )

        aggregate_payload = {
            "schema": AGGREGATE_SCHEMA,
            "source_qualification_metrics_sha256": QUALIFICATION_SHA256,
            "sample_count": aggregate["sample_count"],
            "top1_accuracy": aggregate["overall_accuracy"],
            "error_count": aggregate["error_count"],
            "class_counts": aggregate["class_counts"],
            "correct_counts": aggregate["correct_counts"],
            "per_class_accuracy": aggregate["per_class_accuracy"],
            "confusion_matrix": aggregate["confusion_matrix"],
            "confusion_matrix_sha256": aggregate["confusion_matrix_sha256"],
            "historical_exact_match": aggregate_pass,
            "gate": AGGREGATE_GATE,
        }
        atomic_write_new_json(
            temporary / "aggregate_replay.json",
            aggregate_payload,
            artifact_kind="aggregate replay",
        )

        null_payload = {
            "schema": NULL_CONTRACT_SCHEMA,
            "seed": NULL_SEED,
            "algorithm": "torch_cpu_randperm_labels_only",
            "input_order_hash": identity_info["sample_order_hash"],
            "permutation_hash": null_permutation_hash,
            "permutation_artifact": null_file_name,
            "predictions_unchanged": True,
            "sample_order_unchanged": True,
            "class_count_multiset_preserved": class_counts_preserved,
            "significance_computed": False,
        }
        atomic_write_new_json(
            temporary / "null_identity_contract.json",
            null_payload,
            artifact_kind="null identity contract",
        )

        state_payload = {
            "schema": STATE_SCHEMA,
            "device": "cpu",
            "model_eval": True,
            "no_grad": True,
            "checkpoint_sha256_before": before_checkpoint_hash,
            "checkpoint_sha256_after": after_checkpoint_hash,
            "evaluator_state_digest_before": before_state_digest,
            "evaluator_state_digest_after": after_state_digest,
            "checkpoint_unchanged": before_checkpoint_hash == after_checkpoint_hash,
            "evaluator_state_unchanged": before_state_digest == after_state_digest,
            "cuda_used": False,
            "optical_forward_called": False,
            "optical_checkpoint_touched": False,
            "encoder_touched": False,
            "decoder_touched": False,
            "ols_touched": False,
            "controller_touched": False,
            "optimizer_created": False,
            "backward_calls": 0,
            "training_steps": 0,
            "s2_confirmation_consumed": False,
            "s2_selection_consumed": False,
            "status": STATE_GATE,
        }
        atomic_write_new_json(
            temporary / "state_isolation.json",
            state_payload,
            artifact_kind="state isolation",
        )

        report_payload = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "classification": VPIM_CLASSIFICATION,
            "purpose": "pointwise frozen evaluator prediction identity only",
            "prediction_count": VALIDATION_COUNT,
            "prediction_hash": prediction_hash,
            "correctness_hash": correctness_hash,
            "logits_saved": True,
            "logits_hash": logits_hash,
            "replay_gate": REPLAY_GATE,
            "aggregate_gate": AGGREGATE_GATE,
            "state_isolation_gate": STATE_GATE,
            "s2_executed": False,
            "osera_bootstrap_executed": False,
            "osera_null_significance_executed": False,
            "next_action": NEXT_ACTION,
        }
        atomic_write_new_json(
            temporary / "compact_report.json",
            report_payload,
            artifact_kind="VPIM report",
        )

        artifact_refs = _artifact_refs(temporary)
        manifest_payload = {
            "schema": ARTIFACT_SCHEMA,
            "task_id": TASK_ID,
            "branch": BRANCH,
            "classification": VPIM_CLASSIFICATION,
            "purpose": "S2_MEASUREMENT_INSTRUMENT_ONLY",
            "config": {
                "path": _repo_relative(config_file),
                "sha256": config_hash,
                "canonical_sha256": config_canonical_hash,
            },
            "source_fosea": {
                "manifest_path": _repo_relative(paths["fosea_manifest"]),
                "manifest_sha256": FOSEA_MANIFEST_SHA256,
                "classification": FOSEA_CLASSIFICATION,
                "checkpoint_path": _repo_relative(paths["checkpoint"]),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "corpus_path": _repo_relative(paths["corpus"]),
                "corpus_sha256": CORPUS_SHA256,
                "output_identity_manifest_sha256": OUTPUT_IDENTITY_SHA256,
                "qualification_metrics_sha256": QUALIFICATION_SHA256,
                "data_identity_manifest_sha256": DATA_IDENTITY_SHA256,
                "output_domain_contract_sha256": OUTPUT_CONTRACT_SHA256,
                "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            },
            "evaluator": {
                "schema": "frozen_mnist_evaluator_v1",
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "validation_accuracy_reference": 0.9712,
                "frozen": True,
            },
            "validation": {
                "source_dataset": "mnist_train",
                "source_index_start": 50000,
                "source_index_stop": 55000,
                "count": VALIDATION_COUNT,
                "shape": list(VALIDATION_SHAPE),
                "source_index_hash": identity_info["source_index_hash"],
                "label_hash": identity_info["label_hash"],
                "sample_order_hash": identity_info["sample_order_hash"],
                "processed_view_hash_sequence": identity_info["processed_view_hash_sequence"],
                "raw_camera_hash_sequence": identity_info["raw_camera_hash_sequence"],
            },
            "pointwise": {
                "prediction_file": prediction_file_name,
                "prediction_count": VALIDATION_COUNT,
                "prediction_hash": prediction_hash,
                "correctness_hash": correctness_hash,
                "logits_file": logits_file_name,
                "logits_hash": logits_hash,
                "logits_shape": list(first_logits.shape),
                "logits_dtype": str(first_logits.dtype),
            },
            "replay": {
                "device": "cpu",
                "batch_size": batch_size,
                "shuffle": False,
                "model_eval": True,
                "no_grad": True,
                "deterministic_replay": True,
                "max_abs_logit_difference": max_abs_difference,
                "logit_tolerance": float(config["replay"]["logit_tolerance"]),
                "predictions_equal": True,
                "gate": REPLAY_GATE,
            },
            "aggregate_replay": {
                "artifact": "aggregate_replay.json",
                "top1_accuracy": aggregate["overall_accuracy"],
                "error_count": aggregate["error_count"],
                "confusion_matrix_sha256": aggregate["confusion_matrix_sha256"],
                "historical_exact_match": aggregate_pass,
                "gate": AGGREGATE_GATE,
            },
            "bootstrap_material": {
                "sufficient": True,
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "final_bootstrap_executed": False,
            },
            "null_identity": {
                "artifact": "null_identity_contract.json",
                "permutation_artifact": null_file_name,
                "seed": NULL_SEED,
                "permutation_hash": null_permutation_hash,
                "class_count_multiset_preserved": class_counts_preserved,
                "final_significance_executed": False,
            },
            "state_isolation": {
                "artifact": "state_isolation.json",
                "gate": STATE_GATE,
                "checkpoint_unchanged": True,
                "evaluator_state_unchanged": True,
                "cuda_used": False,
                "optical_forward_called": False,
                "training_steps": 0,
            },
            "artifacts": artifact_refs,
            "authorization": {
                "evaluator_replay": "cpu_only",
                "evaluator_training": False,
                "optical_forward": False,
                "s2_execution": False,
                "osera_final_gate": False,
                "formal_promotion": False,
            },
            "governance": {
                "mainline_state": "closed_at_current_evidence_boundary",
                "reconstruction_manifold_supported": False,
                "generative_manifold_supported": False,
                "generation_readiness": "not_ready",
                "training_authorized": False,
                "formal_promotion": False,
            },
            "next_action": NEXT_ACTION,
        }
        atomic_write_new_json(
            temporary / "prediction_identity_manifest.json",
            manifest_payload,
            artifact_kind="prediction identity manifest",
        )
        _assert_final_tree(temporary)
        if actual_output.exists():
            raise FileExistsError(f"refusing to overwrite VPIM output: {actual_output}")
        os.replace(temporary, actual_output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    manifest = json.loads(
        (actual_output / "prediction_identity_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "task_id": TASK_ID,
        "classification": manifest["classification"],
        "output": _repo_relative(actual_output),
        "prediction_count": VALIDATION_COUNT,
        "top1_accuracy": aggregate["overall_accuracy"],
        "error_count": aggregate["error_count"],
        "prediction_hash": prediction_hash,
        "correctness_hash": correctness_hash,
        "logits_hash": logits_hash,
        "replay_gate": REPLAY_GATE,
        "aggregate_gate": AGGREGATE_GATE,
        "state_isolation_gate": STATE_GATE,
        "s2_executed": False,
        "next_action": NEXT_ACTION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
