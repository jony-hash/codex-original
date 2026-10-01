"""Requalify the frozen output evaluator for the paired-S2 measurement scope.

This consumer reads the immutable VPIM pointwise prediction authority and the
historical FOSEA evidence.  It performs only CPU tensor/statistical work: no
evaluator replay, optical execution, training, calibration, or prediction
correction is part of this task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ldm_project.v2.evidence import (
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)


TASK_ID = "H-LOMA-GEN-S2-OSERA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_output_semantic_evaluator_requalification_v1"
ARTIFACT_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_requalification_manifest.v1"
)
AUTHORITY_BINDING_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_requalification_binding.v1"
)
PER_CLASS_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_per_class_identifiability.v1"
)
BOOTSTRAP_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_bootstrap_metrics.v1"
)
GLOBAL_NULL_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_global_null_metrics.v1"
)
ERROR_MODEL_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_error_model.v1"
)
GATE_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_requalification_gate.v1"
)
REPORT_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_requalification_report.v1"
)

VPIM_SCHEMA = (
    "lightgen_scaled_chen_output_semantic_evaluator_prediction_identity_manifest.v1"
)
VPIM_TASK_ID = "H-LOMA-GEN-S2-OSERA-VPIM"
VPIM_CLASSIFICATION = "OSERA_VALIDATION_PREDICTION_IDENTITY_READY"
VPIM_REPLAY_GATE = "FROZEN_PREDICTION_REPLAY_PASS"
VPIM_AGGREGATE_GATE = "OSERA_PREDICTION_AGGREGATE_EXACT_MATCH"
VPIM_STATE_GATE = "OSERA_VPIM_STATE_ISOLATION_PASS"
FOSEA_CLASSIFICATION = "FROZEN_OUTPUT_SEMANTIC_EVALUATOR_NOT_QUALIFIED"
FOSEA_THRESHOLD = 0.985
EVALUATOR_SCHEMA = "frozen_mnist_evaluator_v1"

PASS_GATE = "OUTPUT_SEMANTIC_EVALUATOR_REQUALIFICATION_PASS"
PASS_CLASSIFICATION = "FROZEN_OUTPUT_SEMANTIC_EVALUATOR_REQUALIFIED_FOR_PAIRED_S2"
NOT_SUPPORTED_GATE = "OUTPUT_SEMANTIC_EVALUATOR_REQUALIFICATION_NOT_SUPPORTED"
BINDING_FAILURE = "OSERA_REQUALIFICATION_AUTHORITY_BINDING_FAILED"
NEXT_ACTION_PASS = "IMPLEMENT_S2_DIRECT_OLS_SEMANTIC_DECODE_CANARY"
NEXT_ACTION_FAIL = "REVIEW_SEMANTIC_MEASUREMENT_INSTRUMENT_LIMITATION"

FOSEA_MANIFEST_SHA256 = (
    "c8e144da2dd02e9f4ee78f7e8829782800651ec835adb1ae6fd781b1921c1d0f"
)
FOSEA_CHECKPOINT_SHA256 = (
    "a6ac0b4644a459bad21bc7cf3071465d5306af31ad2081fe7acb6f81e87c47e6"
)
FOSEA_CORPUS_SHA256 = (
    "787b1b4e256f54ccebb55bb5582308f92148dc595d4cab4f632d3e87d72907b7"
)
FOSEA_OUTPUT_IDENTITY_SHA256 = (
    "db8f03d977a55e2c82ec557b8b359f55f27048c413e31e04c2f76134dfcbb04d"
)
FOSEA_QUALIFICATION_SHA256 = (
    "2fed6db88c0b5003458077198375f746eb995a88be4a9350c2d70cd38acb8568"
)
FOSEA_LEAKAGE_SHA256 = (
    "e541dcbb9c507c8fb8afc4672e6ab4dbb00adbbd5e543badb599000baa60a283"
)
FOSEA_FREEZE_REPLAY_SHA256 = (
    "b5229df43117180ce46b562227f77eb5388d2f5ca28454cb39eb7380e513c30c"
)
FOSEA_EXPOSURE_SHA256 = (
    "e3a9ac15b30c5dd1074914a9f3c05b5f8f21bc3b8349418d2345efc3fb9687f4"
)
FOSEA_OPTICAL_STATE_SHA256 = (
    "11979aef44dd56407ae530727016f2480e5bfc82344e5cf5bf20850aa29cc5f6"
)
FOSEA_DATA_IDENTITY_SHA256 = (
    "8ae23bf1f247124a1f1f9689a2a4cfb507860c49704d334adec158808d093dc4"
)
FOSEA_OUTPUT_CONTRACT_SHA256 = (
    "515137bd66c71ba6b1b1f4e21fa99d4f2527aca286e11ff079b4b7dfd2fbce56"
)
FOSEA_TRAINING_CONTRACT_SHA256 = (
    "e84ad034e5fc726810f8813409dd438242d38aafdae735bfba83c14c53487f25"
)

VPIM_MANIFEST_SHA256 = (
    "eff78d5417961d908c148c08084f55485d0c68cd4abc53e2101c811c3eb84336"
)
VPIM_PREDICTIONS_SHA256 = (
    "0fd6c6d65eed05a9750b80d372167cd57e87a2174adc650d38ca5b8d0cc662ee"
)
VPIM_SAMPLE_ORDER_SHA256 = (
    "12226c3f96c5429b6358eaf6d0473ae0572e1b5836764021578e7dafd6eb271c"
)
VPIM_AGGREGATE_SHA256 = (
    "a6b1c8d71f83fd9ebdec9df902ddd36563caa19b9f02cc138f69d6440bbbd961"
)
VPIM_NULL_CONTRACT_SHA256 = (
    "f0d8e3aefdd4465f2fb785e71277dc17101e5002d9afe23ff2dafa4ee7f1e6ee"
)
VPIM_NULL_PERMUTATION_SHA256 = (
    "34d6437efe47261183ce870d6b91123094c41c6fc72cc4b48b16b0bb39ce135a"
)
VPIM_STATE_SHA256 = (
    "a10f19709068eb76b51467746ba00e8df36a8a8579abe37b859790c426dc9718"
)
VPIM_PREDICTION_HASH = (
    "572e270d17c5e0a1dbb1aaf4010fd1d06d03739ce86f06b945f956d27dfa484a"
)
VPIM_CORRECTNESS_HASH = (
    "e8936b2fbe33ac185099c61ae9c6065a1a4f2ade70376033ca8846f7faf04362"
)
VPIM_SAMPLE_ORDER_HASH = (
    "aedcb0f58651c77518ae01ca6f908ad2126a81395a7a431a233bd0497b60601e"
)
VPIM_NULL_PERMUTATION_TENSOR_HASH = (
    "1b3fa37d45fdd646b5b2478a904e5029a402c7b7061d2ecc42a8eeab21a80794"
)
VPIM_LABEL_HASH = (
    "13d90706eef49c3c9f17ccd39c002dc18a79eca1378e780548ea6b29adc6ec29"
)
VPIM_INDEX_HASH = (
    "5370b35f304a779cde7f978df6392b9974faaf4296e9edf75f9edf8202db02fa"
)

VALIDATION_COUNT = 5000
CLASS_COUNT = 10
VALIDATION_INDEX_START = 50000
VALIDATION_INDEX_STOP = 55000
BOOTSTRAP_SEED = 1236
BOOTSTRAP_REPLICATES = 2000
NULL_SEED = 2053
CONFIDENCE_LEVEL = 0.95
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / (
    "configs/research/scaled_chen/"
    "lightgen_scaled_chen_output_semantic_evaluator_requalification_v1.yaml"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / (
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_output_semantic_evaluator_requalification_v1_attempt_01"
)


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else REPO_ROOT / target


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _output_reference(path: Path) -> str:
    try:
        return _repo_relative(path)
    except ValueError:
        return str(path.resolve())


def _require_hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value.lower()) is None:
        raise ValueError(f"{BINDING_FAILURE}: invalid {name}")
    return value.lower()


def _load_json(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"{BINDING_FAILURE}: {role}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{BINDING_FAILURE}: {role} schema")
    return payload


def _verify_file(path: Path, expected_sha256: str, role: str) -> None:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"{BINDING_FAILURE}: {role}")


def next_write_once_output(base_path: str | Path) -> Path:
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
        raise ValueError("OSERA requalification config must be a mapping")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("task_id") != TASK_ID:
        raise ValueError(f"{BINDING_FAILURE}: config schema")
    _assert_config(payload)
    return payload


def _assert_config(config: Mapping[str, Any]) -> None:
    source = config.get("source")
    statistics = config.get("statistics")
    authorization = config.get("authorization")
    if not isinstance(source, Mapping) or not isinstance(statistics, Mapping):
        raise ValueError(f"{BINDING_FAILURE}: config sections")
    if not isinstance(authorization, Mapping):
        raise ValueError(f"{BINDING_FAILURE}: authorization section")

    expected_source_hashes = {
        "vpim_manifest_sha256": VPIM_MANIFEST_SHA256,
        "vpim_predictions_sha256": VPIM_PREDICTIONS_SHA256,
        "vpim_sample_order_sha256": VPIM_SAMPLE_ORDER_SHA256,
        "vpim_aggregate_sha256": VPIM_AGGREGATE_SHA256,
        "vpim_null_contract_sha256": VPIM_NULL_CONTRACT_SHA256,
        "vpim_null_permutation_sha256": VPIM_NULL_PERMUTATION_SHA256,
        "vpim_state_sha256": VPIM_STATE_SHA256,
        "fosea_manifest_sha256": FOSEA_MANIFEST_SHA256,
        "fosea_checkpoint_sha256": FOSEA_CHECKPOINT_SHA256,
        "fosea_qualification_sha256": FOSEA_QUALIFICATION_SHA256,
        "fosea_leakage_sha256": FOSEA_LEAKAGE_SHA256,
        "fosea_freeze_replay_sha256": FOSEA_FREEZE_REPLAY_SHA256,
        "fosea_exposure_sha256": FOSEA_EXPOSURE_SHA256,
        "fosea_optical_state_sha256": FOSEA_OPTICAL_STATE_SHA256,
        "fosea_data_identity_sha256": FOSEA_DATA_IDENTITY_SHA256,
        "fosea_output_contract_sha256": FOSEA_OUTPUT_CONTRACT_SHA256,
        "fosea_training_contract_sha256": FOSEA_TRAINING_CONTRACT_SHA256,
    }
    for name, expected in expected_source_hashes.items():
        if source.get(name) != expected:
            raise ValueError(f"{BINDING_FAILURE}: config {name}")

    expected_statistics = {
        "class_count": CLASS_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "null_seed": NULL_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "resampling": "within_ground_truth_class_with_replacement",
        "interval": "percentile",
    }
    for name, expected in expected_statistics.items():
        if statistics.get(name) != expected:
            raise ValueError(f"{BINDING_FAILURE}: statistics {name}")
    if statistics.get("absolute_accuracy_threshold") is not None:
        raise ValueError(f"{BINDING_FAILURE}: new accuracy threshold")

    for name in (
        "model_execution",
        "evaluator_replay",
        "optical_execution",
        "s2_execution",
        "training",
        "formal_promotion",
    ):
        if authorization.get(name) is not False:
            raise ValueError(f"{BINDING_FAILURE}: authorization {name}")


def _source_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    source = config["source"]
    return {
        name: _resolve(source[name])
        for name in (
            "vpim_manifest",
            "vpim_predictions",
            "vpim_sample_order",
            "vpim_aggregate",
            "vpim_null_contract",
            "vpim_null_permutation",
            "vpim_state",
            "fosea_manifest",
            "fosea_checkpoint",
            "fosea_qualification",
            "fosea_leakage",
            "fosea_freeze_replay",
            "fosea_exposure",
            "fosea_optical_state",
            "fosea_data_identity",
            "fosea_output_contract",
            "fosea_training_contract",
        )
    }


def _validate_historical_provenance(
    fosea: Mapping[str, Any],
    qualification: Mapping[str, Any],
    leakage: Mapping[str, Any],
    freeze_replay: Mapping[str, Any],
    exposure: Mapping[str, Any],
    optical_state: Mapping[str, Any],
    output_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        fosea.get("schema") != "lightgen_scaled_chen_frozen_output_semantic_evaluator_manifest.v1"
        or fosea.get("classification") != FOSEA_CLASSIFICATION
        or fosea.get("task_id") != "H-LOMA-GEN-S2-FOSEA-BUILD"
        or qualification.get("schema")
        != "lightgen_scaled_chen_frozen_output_evaluator_qualification.v1"
        or qualification.get("classification") != FOSEA_CLASSIFICATION
        or qualification.get("threshold") != FOSEA_THRESHOLD
    ):
        raise ValueError(f"{BINDING_FAILURE}: historical FOSEA identity")

    evaluator = fosea.get("evaluator")
    dataset = fosea.get("dataset")
    fosea_qualification = fosea.get("qualification")
    fixed_preprocessing = output_contract.get("fixed_preprocessing")
    storage_policy = output_contract.get("storage_policy")
    if not isinstance(evaluator, Mapping) or not isinstance(dataset, Mapping):
        raise ValueError(f"{BINDING_FAILURE}: historical FOSEA sections")
    if not isinstance(fosea_qualification, Mapping):
        raise ValueError(f"{BINDING_FAILURE}: historical qualification section")
    if not isinstance(fixed_preprocessing, Mapping) or not isinstance(
        storage_policy, Mapping
    ):
        raise ValueError(f"{BINDING_FAILURE}: historical output contract sections")

    validation = qualification.get("validation")
    gates = qualification.get("gates")
    if not isinstance(validation, Mapping) or not isinstance(gates, Mapping):
        raise ValueError(f"{BINDING_FAILURE}: historical validation section")

    clean_gates = {
        "zero_confirmation_leakage": bool(
            dataset.get("s2_confirmation_consumed") is False
            and dataset.get("s2_selection_consumed") is False
            and leakage.get("valid") is True
            and leakage.get("s2_confirmation_consumed_by_evaluator") is False
            and leakage.get("s2_selection_consumed_by_evaluator") is False
            and leakage.get("validation_confirmation_overlap_zero") is True
        ),
        "zero_selection_leakage": bool(
            leakage.get("validation_selection_overlap_zero") is True
            and leakage.get("underlying_source_identity_overlap_zero") is True
        ),
        "output_domain_match": bool(
            output_contract.get("schema")
            == "lightgen_scaled_chen_frozen_output_domain_contract.v1"
            and fixed_preprocessing.get("per_sample_normalization") is False
            and storage_policy.get("raw_camera_saved") is False
            and storage_policy.get("s2_confirmation_forwarded") is False
            and storage_policy.get("s2_selection_forwarded") is False
        ),
        "frozen_checkpoint": bool(
            evaluator.get("schema") == EVALUATOR_SCHEMA
            and evaluator.get("checkpoint_sha256") == FOSEA_CHECKPOINT_SHA256
            and qualification.get("evaluator_checkpoint_sha256")
            == FOSEA_CHECKPOINT_SHA256
        ),
        "fixed_preprocessing": bool(
            exposure.get("status") == "EXPOSURE_SCALE_AUTHORITY_PASS"
            and exposure.get("fit_from_s2_confirmation") is False
            and exposure.get("fit_from_s2_selection") is False
            and exposure.get("tuned_on_s2_performance") is False
        ),
        "optical_state_isolation": bool(
            optical_state.get("status") == "OPTICAL_STATE_ISOLATION_PASS"
            and optical_state.get("state_unchanged") is True
            and optical_state.get("phase_updated") is False
            and optical_state.get("s2_preview_executed") is False
        ),
        "freeze_replay": bool(
            freeze_replay.get("status") == "EVALUATOR_FREEZE_REPLAY_PASS"
            and freeze_replay.get("deterministic_replay") is True
            and freeze_replay.get("logits_exact_equal") is True
            and freeze_replay.get("reloaded_parameters_frozen") is True
        ),
        "historical_metrics_finite": bool(
            validation.get("finite") is True
            and validation.get("sample_count") == VALIDATION_COUNT
            and validation.get("overall_accuracy") == 0.9712
            and fosea_qualification.get("s2_confirmation_evaluated") is False
            and fosea_qualification.get("s2_selection_evaluated") is False
            and all(gates.get(name) is True for name in gates)
        ),
    }
    return {
        "schema": "lightgen_scaled_chen_output_semantic_evaluator_historical_provenance.v1",
        "gates": clean_gates,
        "all_pass": all(clean_gates.values()),
        "historical_classification": FOSEA_CLASSIFICATION,
        "historical_accuracy": 0.9712,
        "historical_threshold": FOSEA_THRESHOLD,
    }


def _validate_vectors(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    *,
    class_count: int = CLASS_COUNT,
) -> None:
    if (
        labels.ndim != 1
        or predictions.ndim != 1
        or labels.shape != predictions.shape
        or labels.dtype != torch.long
        or predictions.dtype != torch.long
        or labels.device.type != "cpu"
        or predictions.device.type != "cpu"
        or labels.numel() == 0
        or class_count < 2
        or int(labels.min()) < 0
        or int(labels.max()) >= class_count
        or int(predictions.min()) < 0
        or int(predictions.max()) >= class_count
    ):
        raise ValueError(f"{BINDING_FAILURE}: classification vectors")
    counts = torch.bincount(labels, minlength=class_count)
    if (counts == 0).any():
        raise ValueError(f"{BINDING_FAILURE}: empty semantic class")


def _confusion_matrix(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    *,
    class_count: int = CLASS_COUNT,
) -> torch.Tensor:
    _validate_vectors(labels, predictions, class_count=class_count)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    confusion.index_put_((labels, predictions), torch.ones_like(labels), accumulate=True)
    return confusion


def compute_per_class_margins(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    *,
    class_count: int = CLASS_COUNT,
) -> dict[str, Any]:
    confusion = _confusion_matrix(labels, predictions, class_count=class_count)
    counts = confusion.sum(dim=1)
    classes: dict[str, dict[str, Any]] = {}
    for label in range(class_count):
        row = confusion[label]
        diagonal = int(row[label].item())
        wrong_counts = row.clone()
        wrong_counts[label] = 0
        max_wrong_count = int(wrong_counts.max().item())
        dominant_prediction = int(row.argmax().item())
        dominant_wrong_class = (
            int(wrong_counts.argmax().item()) if max_wrong_count > 0 else None
        )
        denominator = float(counts[label].item())
        p_correct = diagonal / denominator
        p_wrong_max = max_wrong_count / denominator
        classes[str(label)] = {
            "sample_count": int(counts[label].item()),
            "correct_count": diagonal,
            "p_correct": p_correct,
            "p_wrong_max": p_wrong_max,
            "margin": p_correct - p_wrong_max,
            "dominant_prediction": dominant_prediction,
            "dominant_wrong_class": dominant_wrong_class,
            "dominant_wrong_count": max_wrong_count,
            "dominant_wrong_probability": p_wrong_max,
            "diagonal_unique_maximum": diagonal > max_wrong_count,
        }
    return {
        "schema": PER_CLASS_SCHEMA,
        "class_count": class_count,
        "class_order": list(range(class_count)),
        "confusion_matrix": confusion.tolist(),
        "classes": classes,
        "point_gate": all(value["margin"] > 0.0 for value in classes.values()),
        "class_collapse": any(
            not value["diagonal_unique_maximum"] for value in classes.values()
        ),
    }


def _percentile_ci(values: torch.Tensor, confidence_level: float) -> tuple[float, float]:
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("bootstrap values must be finite and one-dimensional")
    alpha = (1.0 - confidence_level) / 2.0
    quantiles = torch.quantile(
        values.to(torch.float64),
        torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64),
    )
    return float(quantiles[0].item()), float(quantiles[1].item())


def _stratified_draws(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    *,
    reps: int,
    seed: int,
    class_count: int,
) -> torch.Tensor:
    _validate_vectors(labels, predictions, class_count=class_count)
    if reps < 1:
        raise ValueError("bootstrap replicates must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.empty((reps, class_count), dtype=torch.float64)
    for label in range(class_count):
        positions = torch.nonzero(labels == label, as_tuple=False).flatten()
        draws = torch.randint(
            positions.numel(),
            (reps, positions.numel()),
            generator=generator,
        )
        sampled_positions = positions.index_select(0, draws.reshape(-1)).reshape(
            reps, -1
        )
        sampled_predictions = predictions.index_select(
            0, sampled_positions.reshape(-1)
        ).reshape(reps, -1)
        row_counts = torch.zeros((reps, class_count), dtype=torch.float64)
        row_counts.scatter_add_(
            1,
            sampled_predictions,
            torch.ones_like(sampled_predictions, dtype=torch.float64),
        )
        wrong_counts = row_counts.clone()
        wrong_counts[:, label] = 0.0
        values[:, label] = (
            row_counts[:, label] - wrong_counts.max(dim=1).values
        ) / positions.numel()
    return values


def stratified_margin_bootstrap(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    *,
    reps: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    point = compute_per_class_margins(labels, predictions)
    samples = _stratified_draws(
        labels,
        predictions,
        reps=reps,
        seed=seed,
        class_count=CLASS_COUNT,
    )
    classes: dict[str, dict[str, Any]] = {}
    for label in range(CLASS_COUNT):
        lower95, upper95 = _percentile_ci(samples[:, label], confidence_level)
        point_value = point["classes"][str(label)]
        classes[str(label)] = {
            "point_margin": point_value["margin"],
            "lower95": lower95,
            "upper95": upper95,
            "lower95_positive": lower95 > 0.0,
        }
    minimum_lower95 = min(value["lower95"] for value in classes.values())
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "statistic": "per_class_margin",
        "resampling": "within_ground_truth_class_with_replacement",
        "interval": "percentile",
        "confidence_level": confidence_level,
        "seed": seed,
        "replicates": reps,
        "classes": classes,
        "minimum_lower95": minimum_lower95,
        "all_lower95_positive": minimum_lower95 > 0.0,
    }


def _validate_permutation(
    permutation: torch.Tensor,
    sample_count: int,
) -> None:
    if (
        permutation.ndim != 1
        or permutation.dtype != torch.long
        or permutation.device.type != "cpu"
        or permutation.numel() != sample_count
        or not torch.equal(
            torch.sort(permutation).values,
            torch.arange(sample_count, dtype=torch.long),
        )
    ):
        raise ValueError(f"{BINDING_FAILURE}: null permutation")


def compute_global_shuffled_null(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    permutation: torch.Tensor,
    *,
    reps: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    _validate_vectors(labels, predictions)
    _validate_permutation(permutation, labels.numel())
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    shuffled_labels = labels.index_select(0, permutation)
    class_count_multiset_preserved = torch.equal(
        torch.bincount(labels, minlength=CLASS_COUNT),
        torch.bincount(shuffled_labels, minlength=CLASS_COUNT),
    )
    if not class_count_multiset_preserved:
        raise ValueError(f"{BINDING_FAILURE}: null class counts")

    real_correct = predictions.eq(labels).to(torch.float64)
    shuffled_correct = predictions.eq(shuffled_labels).to(torch.float64)
    delta = real_correct - shuffled_correct
    bootstrap_values = torch.zeros(reps, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_count = float(labels.numel())
    for label in range(CLASS_COUNT):
        positions = torch.nonzero(labels == label, as_tuple=False).flatten()
        draws = torch.randint(
            positions.numel(), (reps, positions.numel()), generator=generator
        )
        sampled_positions = positions.index_select(0, draws.reshape(-1)).reshape(
            reps, -1
        )
        bootstrap_values += (
            delta.index_select(0, sampled_positions.reshape(-1))
            .reshape(reps, -1)
            .mean(dim=1)
            * (positions.numel() / total_count)
        )
    lower95, upper95 = _percentile_ci(bootstrap_values, confidence_level)
    real_accuracy = float(real_correct.mean().item())
    shuffled_accuracy = float(shuffled_correct.mean().item())
    return {
        "schema": GLOBAL_NULL_SCHEMA,
        "statistic": "accuracy_real_minus_accuracy_label_shuffled",
        "label_only_permutation": True,
        "predictions_unchanged": True,
        "sample_order_unchanged": True,
        "class_count_multiset_preserved": class_count_multiset_preserved,
        "null_seed": NULL_SEED,
        "bootstrap_seed": seed,
        "bootstrap_replicates": reps,
        "confidence_level": confidence_level,
        "interval": "stratified_percentile_bootstrap",
        "real_accuracy": real_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "point_delta": real_accuracy - shuffled_accuracy,
        "lower95": lower95,
        "upper95": upper95,
        "lower95_positive": lower95 > 0.0,
    }


def build_error_model(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, Any]:
    point = compute_per_class_margins(labels, predictions)
    confusion = torch.tensor(point["confusion_matrix"], dtype=torch.long)
    counts = confusion.sum(dim=1)
    correct = confusion.diag()
    per_class_recall = [
        float(value.item())
        for value in (correct.to(torch.float64) / counts.to(torch.float64))
    ]
    dominant_wrong = {
        label: {
            "class": values["dominant_wrong_class"],
            "count": values["dominant_wrong_count"],
            "probability": values["dominant_wrong_probability"],
        }
        for label, values in point["classes"].items()
    }
    correctness = predictions.eq(labels)
    return {
        "schema": ERROR_MODEL_SCHEMA,
        "name": "SEMANTIC_INSTRUMENT_ERROR_MODEL_V1",
        "raw_predictions_authoritative": True,
        "prediction_correction_applied": False,
        "posthoc_calibration_applied": False,
        "confusion_matrix": point["confusion_matrix"],
        "class_counts": [int(value.item()) for value in counts],
        "correct_counts": [int(value.item()) for value in correct],
        "overall_accuracy": float(correctness.to(torch.float64).mean().item()),
        "overall_error_rate": float((~correctness).to(torch.float64).mean().item()),
        "per_class_recall": per_class_recall,
        "dominant_wrong_class": dominant_wrong,
        "per_class_margin": {
            label: values["margin"] for label, values in point["classes"].items()
        },
        "frozen": True,
    }


def _finite_metrics(
    per_class: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    global_null: Mapping[str, Any],
    error_model: Mapping[str, Any],
) -> bool:
    values: list[float] = []
    values.extend(
        float(item["margin"])
        for item in per_class["classes"].values()
    )
    for item in bootstrap["classes"].values():
        values.extend(
            [float(item["point_margin"]), float(item["lower95"]), float(item["upper95"])]
        )
    values.extend(
        float(global_null[name])
        for name in ("real_accuracy", "shuffled_accuracy", "point_delta", "lower95", "upper95")
    )
    values.extend(
        [float(error_model["overall_accuracy"]), float(error_model["overall_error_rate"])]
    )
    return all(math.isfinite(value) for value in values)


def classify_requalification(
    *,
    authority_binding: bool,
    clean_provenance: bool,
    per_class_gate: bool,
    global_null_gate: bool,
    metrics_finite: bool,
    error_model_frozen: bool,
    evaluator_state_unchanged: bool,
) -> dict[str, Any]:
    if not authority_binding:
        return {
            "gate": BINDING_FAILURE,
            "classification": BINDING_FAILURE,
            "reason": "AUTHORITY_BINDING",
            "new_absolute_accuracy_threshold": "NONE",
        }
    if not clean_provenance:
        reason = "HISTORICAL_PROVENANCE"
    elif not per_class_gate:
        reason = "PER_CLASS_IDENTIFIABILITY"
    elif not global_null_gate:
        reason = "GLOBAL_NULL_SEPARATION"
    elif not metrics_finite:
        reason = "NONFINITE_METRICS"
    elif not error_model_frozen:
        reason = "ERROR_MODEL_NOT_FROZEN"
    elif not evaluator_state_unchanged:
        reason = "EVALUATOR_STATE_CHANGED"
    else:
        return {
            "gate": PASS_GATE,
            "classification": PASS_CLASSIFICATION,
            "reason": None,
            "new_absolute_accuracy_threshold": "NONE",
        }
    return {
        "gate": NOT_SUPPORTED_GATE,
        "classification": NOT_SUPPORTED_GATE,
        "reason": reason,
        "new_absolute_accuracy_threshold": "NONE",
    }


def _read_prediction_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{BINDING_FAILURE}: prediction record schema")
        records.append(value)
    return records


def _validate_prediction_records(
    records: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if len(records) != VALIDATION_COUNT:
        raise ValueError(f"{BINDING_FAILURE}: prediction row count")
    labels: list[int] = []
    predictions: list[int] = []
    correctness: list[bool] = []
    order_payload: list[dict[str, Any]] = []
    required_keys = {
        "row_index",
        "source_dataset",
        "source_index",
        "ground_truth_label",
        "prediction",
        "correct",
        "processed_view_sha256",
        "raw_camera_sha256",
    }
    for row_index, record in enumerate(records):
        if set(record) != required_keys:
            raise ValueError(f"{BINDING_FAILURE}: prediction record fields")
        label = record["ground_truth_label"]
        prediction = record["prediction"]
        correct = record["correct"]
        if (
            record["row_index"] != row_index
            or record["source_dataset"] != "mnist_train"
            or record["source_index"] != VALIDATION_INDEX_START + row_index
            or not isinstance(label, int)
            or not 0 <= label < CLASS_COUNT
            or not isinstance(prediction, int)
            or not 0 <= prediction < CLASS_COUNT
            or type(correct) is not bool
            or correct != (label == prediction)
            or HEX64.fullmatch(str(record["processed_view_sha256"]).lower()) is None
            or HEX64.fullmatch(str(record["raw_camera_sha256"]).lower()) is None
        ):
            raise ValueError(f"{BINDING_FAILURE}: prediction identity")
        labels.append(label)
        predictions.append(prediction)
        correctness.append(correct)
        order_payload.append(
            {
                "row_index": row_index,
                "source_dataset": record["source_dataset"],
                "source_index": record["source_index"],
            }
        )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    prediction_tensor = torch.tensor(predictions, dtype=torch.long)
    correctness_tensor = torch.tensor(correctness, dtype=torch.bool)
    return (
        label_tensor,
        prediction_tensor,
        correctness_tensor,
        canonical_sha256(order_payload),
    )


def _validate_vpim_authority(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    vpim_manifest: Mapping[str, Any],
    vpim_sample_order: Mapping[str, Any],
    vpim_aggregate: Mapping[str, Any],
    vpim_null_contract: Mapping[str, Any],
    vpim_state: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        vpim_manifest.get("schema") != VPIM_SCHEMA
        or vpim_manifest.get("task_id") != VPIM_TASK_ID
        or vpim_manifest.get("classification") != VPIM_CLASSIFICATION
        or vpim_manifest.get("purpose") != "S2_MEASUREMENT_INSTRUMENT_ONLY"
    ):
        raise ValueError(f"{BINDING_FAILURE}: VPIM manifest identity")
    source_fosea = vpim_manifest.get("source_fosea")
    evaluator = vpim_manifest.get("evaluator")
    validation = vpim_manifest.get("validation")
    replay = vpim_manifest.get("replay")
    aggregate = vpim_manifest.get("aggregate_replay")
    null_identity = vpim_manifest.get("null_identity")
    state = vpim_manifest.get("state_isolation")
    pointwise = vpim_manifest.get("pointwise")
    if not all(
        isinstance(value, Mapping)
        for value in (
            source_fosea,
            evaluator,
            validation,
            replay,
            aggregate,
            null_identity,
            state,
            pointwise,
        )
    ):
        raise ValueError(f"{BINDING_FAILURE}: VPIM sections")
    if (
        source_fosea["manifest_sha256"] != FOSEA_MANIFEST_SHA256
        or source_fosea["checkpoint_sha256"] != FOSEA_CHECKPOINT_SHA256
        or source_fosea["corpus_sha256"] != FOSEA_CORPUS_SHA256
        or source_fosea["output_identity_manifest_sha256"]
        != FOSEA_OUTPUT_IDENTITY_SHA256
        or source_fosea["qualification_metrics_sha256"]
        != FOSEA_QUALIFICATION_SHA256
        or source_fosea["data_identity_manifest_sha256"]
        != FOSEA_DATA_IDENTITY_SHA256
        or source_fosea["output_domain_contract_sha256"]
        != FOSEA_OUTPUT_CONTRACT_SHA256
        or source_fosea["training_contract_sha256"]
        != FOSEA_TRAINING_CONTRACT_SHA256
        or source_fosea["classification"] != FOSEA_CLASSIFICATION
        or evaluator["schema"] != EVALUATOR_SCHEMA
        or evaluator["checkpoint_sha256"] != FOSEA_CHECKPOINT_SHA256
        or evaluator["frozen"] is not True
        or validation["count"] != VALIDATION_COUNT
        or validation["source_index_start"] != VALIDATION_INDEX_START
        or validation["source_index_stop"] != VALIDATION_INDEX_STOP
        or validation["source_index_hash"] != VPIM_INDEX_HASH
        or validation["label_hash"] != VPIM_LABEL_HASH
        or validation["sample_order_hash"] != VPIM_SAMPLE_ORDER_HASH
        or replay["gate"] != VPIM_REPLAY_GATE
        or replay["device"] != "cpu"
        or replay["shuffle"] is not False
        or aggregate["gate"] != VPIM_AGGREGATE_GATE
        or aggregate["historical_exact_match"] is not True
        or null_identity["seed"] != NULL_SEED
        or null_identity["permutation_hash"] != VPIM_NULL_PERMUTATION_TENSOR_HASH
        or null_identity["final_significance_executed"] is not False
        or state["gate"] != VPIM_STATE_GATE
        or state["checkpoint_unchanged"] is not True
        or state["evaluator_state_unchanged"] is not True
        or state["cuda_used"] is not False
        or state["training_steps"] != 0
        or pointwise["prediction_count"] != VALIDATION_COUNT
        or pointwise["prediction_hash"] != VPIM_PREDICTION_HASH
        or pointwise["correctness_hash"] != VPIM_CORRECTNESS_HASH
    ):
        raise ValueError(f"{BINDING_FAILURE}: VPIM binding values")
    bootstrap_material = vpim_manifest.get("bootstrap_material")
    authorization = vpim_manifest.get("authorization")
    if not isinstance(bootstrap_material, Mapping) or not isinstance(
        authorization, Mapping
    ):
        raise ValueError(f"{BINDING_FAILURE}: VPIM authorization sections")
    if (
        bootstrap_material.get("seed") != BOOTSTRAP_SEED
        or bootstrap_material.get("replicates") != BOOTSTRAP_REPLICATES
        or bootstrap_material.get("final_bootstrap_executed") is not False
        or authorization.get("evaluator_training") is not False
        or authorization.get("optical_forward") is not False
        or authorization.get("s2_execution") is not False
        or authorization.get("osera_final_gate") is not False
    ):
        raise ValueError(f"{BINDING_FAILURE}: VPIM authorization values")
    if vpim_sample_order.get("order_verified") is not True:
        raise ValueError(f"{BINDING_FAILURE}: sample-order authority")
    if (
        vpim_sample_order.get("sample_order_hash") != VPIM_SAMPLE_ORDER_HASH
        or vpim_aggregate.get("gate") != VPIM_AGGREGATE_GATE
        or vpim_aggregate.get("sample_count") != VALIDATION_COUNT
        or vpim_aggregate.get("top1_accuracy") != 0.9712
        or vpim_aggregate.get("error_count") != 144
        or vpim_null_contract.get("schema")
        != "lightgen_scaled_chen_output_semantic_evaluator_null_identity.v1"
        or vpim_null_contract.get("seed") != NULL_SEED
        or vpim_null_contract.get("predictions_unchanged") is not True
        or vpim_null_contract.get("sample_order_unchanged") is not True
        or vpim_null_contract.get("class_count_multiset_preserved") is not True
        or vpim_null_contract.get("significance_computed") is not False
        or vpim_state.get("status") != VPIM_STATE_GATE
        or vpim_state.get("cuda_used") is not False
        or vpim_state.get("training_steps") != 0
    ):
        raise ValueError(f"{BINDING_FAILURE}: VPIM artifact values")
    return {
        "schema": AUTHORITY_BINDING_SCHEMA,
        "binding_gate": True,
        "vpim_manifest_sha256": VPIM_MANIFEST_SHA256,
        "vpim_classification": VPIM_CLASSIFICATION,
        "prediction_count": VALIDATION_COUNT,
        "prediction_hash": VPIM_PREDICTION_HASH,
        "correctness_hash": VPIM_CORRECTNESS_HASH,
        "aggregate_gate": VPIM_AGGREGATE_GATE,
        "replay_gate": VPIM_REPLAY_GATE,
        "state_gate": VPIM_STATE_GATE,
        "source_artifacts": {
            name: {
                "path": _repo_relative(paths[name]),
                "sha256": config["source"][f"{name}_sha256"],
            }
            for name in (
                "vpim_manifest",
                "vpim_predictions",
                "vpim_sample_order",
                "vpim_aggregate",
                "vpim_null_contract",
                "vpim_null_permutation",
                "vpim_state",
            )
        },
    }


def _artifact_refs(root: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for path in sorted(root.iterdir()):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            refs[relative] = {
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    return refs


def _assert_final_tree(root: Path) -> None:
    expected = {
        "authority_binding.json",
        "bootstrap_metrics.json",
        "compact_report.json",
        "global_null_metrics.json",
        "osera_manifest.json",
        "per_class_identifiability.json",
        "requalification_gate.json",
        "semantic_instrument_error_model.json",
    }
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected:
        raise RuntimeError(f"OSERA_REQUALIFICATION_MATERIALIZATION_INCOMPLETE: {sorted(actual)}")
    for path in root.iterdir():
        if path.name.startswith(".") or path.suffix.lower() in {".tmp", ".tif", ".tiff"}:
            raise RuntimeError(f"OSERA_REQUALIFICATION_MATERIALIZATION_INCOMPLETE: {path.name}")


def run(
    config_path: str | Path = CONFIG_PATH,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    config = load_config(config_file)
    paths = _source_paths(config)

    fosea_manifest = _load_json(
        paths["fosea_manifest"], FOSEA_MANIFEST_SHA256, "FOSEA manifest"
    )
    qualification = _load_json(
        paths["fosea_qualification"], FOSEA_QUALIFICATION_SHA256, "qualification"
    )
    leakage = _load_json(paths["fosea_leakage"], FOSEA_LEAKAGE_SHA256, "leakage")
    freeze_replay = _load_json(
        paths["fosea_freeze_replay"], FOSEA_FREEZE_REPLAY_SHA256, "freeze replay"
    )
    exposure = _load_json(paths["fosea_exposure"], FOSEA_EXPOSURE_SHA256, "exposure")
    optical_state = _load_json(
        paths["fosea_optical_state"], FOSEA_OPTICAL_STATE_SHA256, "state isolation"
    )
    output_contract = _load_json(
        paths["fosea_output_contract"], FOSEA_OUTPUT_CONTRACT_SHA256, "output contract"
    )
    _load_json(paths["fosea_data_identity"], FOSEA_DATA_IDENTITY_SHA256, "data identity")
    _load_json(
        paths["fosea_training_contract"],
        FOSEA_TRAINING_CONTRACT_SHA256,
        "training contract",
    )
    historical = _validate_historical_provenance(
        fosea_manifest,
        qualification,
        leakage,
        freeze_replay,
        exposure,
        optical_state,
        output_contract,
    )
    if not historical["all_pass"]:
        raise ValueError(f"{BINDING_FAILURE}: historical clean provenance")

    vpim_manifest = _load_json(
        paths["vpim_manifest"], VPIM_MANIFEST_SHA256, "VPIM manifest"
    )
    vpim_sample_order = _load_json(
        paths["vpim_sample_order"], VPIM_SAMPLE_ORDER_SHA256, "VPIM sample order"
    )
    vpim_aggregate = _load_json(
        paths["vpim_aggregate"], VPIM_AGGREGATE_SHA256, "VPIM aggregate"
    )
    vpim_null_contract = _load_json(
        paths["vpim_null_contract"], VPIM_NULL_CONTRACT_SHA256, "VPIM null contract"
    )
    vpim_state = _load_json(paths["vpim_state"], VPIM_STATE_SHA256, "VPIM state")
    vpim_binding = _validate_vpim_authority(
        config,
        paths,
        vpim_manifest,
        vpim_sample_order,
        vpim_aggregate,
        vpim_null_contract,
        vpim_state,
    )

    _verify_file(paths["vpim_predictions"], VPIM_PREDICTIONS_SHA256, "VPIM predictions")
    records = _read_prediction_records(paths["vpim_predictions"])
    labels, predictions, correctness, computed_order_hash = _validate_prediction_records(
        records
    )
    if (
        computed_order_hash != VPIM_SAMPLE_ORDER_HASH
        or tensor_sha256(labels) != VPIM_LABEL_HASH
        or tensor_sha256(predictions) != VPIM_PREDICTION_HASH
        or tensor_sha256(correctness) != VPIM_CORRECTNESS_HASH
    ):
        raise ValueError(f"{BINDING_FAILURE}: pointwise identity hashes")

    null_payload = safe_torch_load(
        paths["vpim_null_permutation"],
        expected_sha256=VPIM_NULL_PERMUTATION_SHA256,
        map_location="cpu",
    )
    if not isinstance(null_payload, Mapping):
        raise ValueError(f"{BINDING_FAILURE}: null permutation payload")
    permutation = null_payload.get("permutation_indices")
    if not isinstance(permutation, torch.Tensor):
        raise ValueError(f"{BINDING_FAILURE}: null permutation tensor")
    _validate_permutation(permutation, VALIDATION_COUNT)
    if (
        null_payload.get("schema")
        != "lightgen_scaled_chen_output_semantic_evaluator_null_permutation.v1"
        or null_payload.get("seed") != NULL_SEED
        or null_payload.get("sample_order_hash") != VPIM_SAMPLE_ORDER_HASH
        or tensor_sha256(permutation) != VPIM_NULL_PERMUTATION_TENSOR_HASH
    ):
        raise ValueError(f"{BINDING_FAILURE}: null permutation identity")

    per_class = compute_per_class_margins(labels, predictions)
    bootstrap = stratified_margin_bootstrap(
        labels,
        predictions,
        reps=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        confidence_level=CONFIDENCE_LEVEL,
    )
    global_null = compute_global_shuffled_null(
        labels,
        predictions,
        permutation,
        reps=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        confidence_level=CONFIDENCE_LEVEL,
    )
    error_model = build_error_model(labels, predictions)
    metrics_finite = _finite_metrics(per_class, bootstrap, global_null, error_model)
    per_class_gate = bool(
        per_class["point_gate"]
        and not per_class["class_collapse"]
        and bootstrap["all_lower95_positive"]
    )
    global_null_gate = bool(global_null["lower95_positive"])
    evaluator_state_unchanged = bool(
        vpim_state.get("checkpoint_unchanged") is True
        and vpim_state.get("evaluator_state_unchanged") is True
        and vpim_state.get("status") == VPIM_STATE_GATE
    )
    error_model_frozen = bool(
        error_model["frozen"] is True
        and error_model["raw_predictions_authoritative"] is True
        and error_model["prediction_correction_applied"] is False
        and error_model["posthoc_calibration_applied"] is False
    )
    gate = classify_requalification(
        authority_binding=bool(vpim_binding["binding_gate"] and historical["all_pass"]),
        clean_provenance=bool(historical["all_pass"]),
        per_class_gate=per_class_gate,
        global_null_gate=global_null_gate,
        metrics_finite=metrics_finite,
        error_model_frozen=error_model_frozen,
        evaluator_state_unchanged=evaluator_state_unchanged,
    )
    next_action = NEXT_ACTION_PASS if gate["classification"] == PASS_CLASSIFICATION else NEXT_ACTION_FAIL

    authority_binding_payload = {
        **vpim_binding,
        "historical_fosea": {
            "manifest_sha256": FOSEA_MANIFEST_SHA256,
            "classification": FOSEA_CLASSIFICATION,
            "threshold": FOSEA_THRESHOLD,
            "accuracy": 0.9712,
            "clean_provenance_gates": historical["gates"],
            "all_clean_gates_pass": historical["all_pass"],
        },
        "identity_verified": True,
        "model_execution": False,
    }
    requalification_gate = {
        "schema": GATE_SCHEMA,
        "checks": {
            "authority_binding": True,
            "historical_clean_provenance": historical["all_pass"],
            "all_point_margins_positive": per_class["point_gate"],
            "all_class_diagonal_dominant": not per_class["class_collapse"],
            "all_bootstrap_lower95_positive": bootstrap["all_lower95_positive"],
            "global_null_lower95_positive": global_null_gate,
            "metrics_finite": metrics_finite,
            "error_model_frozen": error_model_frozen,
            "evaluator_state_unchanged": evaluator_state_unchanged,
        },
        "gate": gate["gate"],
        "classification": gate["classification"],
        "reason": gate["reason"],
        "new_absolute_accuracy_threshold": "NONE",
        "scope": "S2_MEASUREMENT_INSTRUMENT_ONLY",
        "next_action": next_action,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "classification": gate["classification"],
        "primary_gate": gate["gate"],
        "reason": gate["reason"],
        "historical_fosea_classification": FOSEA_CLASSIFICATION,
        "vpim_classification": VPIM_CLASSIFICATION,
        "prediction_count": VALIDATION_COUNT,
        "accuracy": float(error_model["overall_accuracy"]),
        "error_count": int((~correctness).sum().item()),
        "per_class_gate": per_class_gate,
        "global_null_gate": global_null_gate,
        "new_absolute_accuracy_threshold": "NONE",
        "model_execution": False,
        "evaluator_state_unchanged": evaluator_state_unchanged,
        "error_model": "SEMANTIC_INSTRUMENT_ERROR_MODEL_V1",
        "scope": "S2_MEASUREMENT_INSTRUMENT_ONLY",
        "next_action": next_action,
    }

    requested_output = _resolve(output_path or config["output"]["path"])
    actual_output = next_write_once_output(requested_output)
    actual_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{actual_output.name}.", dir=str(actual_output.parent))
    )
    try:
        atomic_write_new_json(
            temporary / "authority_binding.json",
            authority_binding_payload,
            artifact_kind="authority binding",
        )
        atomic_write_new_json(
            temporary / "per_class_identifiability.json",
            per_class,
            artifact_kind="per-class identifiability",
        )
        atomic_write_new_json(
            temporary / "bootstrap_metrics.json",
            bootstrap,
            artifact_kind="bootstrap metrics",
        )
        atomic_write_new_json(
            temporary / "global_null_metrics.json",
            {
                **global_null,
                "permutation_artifact": _repo_relative(paths["vpim_null_permutation"]),
                "permutation_file_sha256": VPIM_NULL_PERMUTATION_SHA256,
                "permutation_tensor_sha256": VPIM_NULL_PERMUTATION_TENSOR_HASH,
            },
            artifact_kind="global null metrics",
        )
        atomic_write_new_json(
            temporary / "semantic_instrument_error_model.json",
            error_model,
            artifact_kind="semantic error model",
        )
        atomic_write_new_json(
            temporary / "requalification_gate.json",
            requalification_gate,
            artifact_kind="requalification gate",
        )
        atomic_write_new_json(
            temporary / "compact_report.json",
            report,
            artifact_kind="requalification report",
        )

        config_hash = file_sha256(config_file)
        config_canonical_hash = canonical_sha256(config)
        artifact_refs = _artifact_refs(temporary)
        manifest_payload = {
            "schema": ARTIFACT_SCHEMA,
            "task_id": TASK_ID,
            "branch": BRANCH,
            "scope": "S2_MEASUREMENT_INSTRUMENT_ONLY",
            "classification": gate["classification"],
            "primary_gate": gate["gate"],
            "reason": gate["reason"],
            "config": {
                "path": _repo_relative(config_file),
                "sha256": config_hash,
                "canonical_sha256": config_canonical_hash,
            },
            "historical_fosea": {
                "manifest_path": _repo_relative(paths["fosea_manifest"]),
                "manifest_sha256": FOSEA_MANIFEST_SHA256,
                "classification": FOSEA_CLASSIFICATION,
                "historical_threshold": FOSEA_THRESHOLD,
                "historical_accuracy": 0.9712,
                "qualification_metrics_sha256": FOSEA_QUALIFICATION_SHA256,
            },
            "vpim": {
                "manifest_path": _repo_relative(paths["vpim_manifest"]),
                "manifest_sha256": VPIM_MANIFEST_SHA256,
                "classification": VPIM_CLASSIFICATION,
                "prediction_path": _repo_relative(paths["vpim_predictions"]),
                "prediction_sha256": VPIM_PREDICTIONS_SHA256,
                "prediction_hash": VPIM_PREDICTION_HASH,
                "correctness_hash": VPIM_CORRECTNESS_HASH,
                "sample_order_hash": VPIM_SAMPLE_ORDER_HASH,
                "aggregate_gate": VPIM_AGGREGATE_GATE,
                "replay_gate": VPIM_REPLAY_GATE,
                "state_gate": VPIM_STATE_GATE,
            },
            "evaluator": {
                "schema": EVALUATOR_SCHEMA,
                "checkpoint_sha256": FOSEA_CHECKPOINT_SHA256,
                "model_execution": False,
            },
            "statistics": {
                "class_count": CLASS_COUNT,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "null_seed": NULL_SEED,
                "confidence_level": CONFIDENCE_LEVEL,
                "interval": "percentile",
                "resampling": "within_ground_truth_class_with_replacement",
                "new_absolute_accuracy_threshold": "NONE",
            },
            "validation": {
                "source_dataset": "mnist_train",
                "source_index_start": VALIDATION_INDEX_START,
                "source_index_stop": VALIDATION_INDEX_STOP,
                "count": VALIDATION_COUNT,
                "accuracy": float(error_model["overall_accuracy"]),
                "errors": int((~correctness).sum().item()),
                "sample_order_hash": VPIM_SAMPLE_ORDER_HASH,
            },
            "gates": requalification_gate,
            "error_model": {
                "artifact": "semantic_instrument_error_model.json",
                "name": "SEMANTIC_INSTRUMENT_ERROR_MODEL_V1",
                "raw_predictions_authoritative": True,
                "correction_applied": False,
            },
            "authorization": {
                "model_execution": False,
                "evaluator_replay": False,
                "optical_execution": False,
                "s2_execution": False,
                "training": False,
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
            "artifacts": artifact_refs,
            "next_action": next_action,
        }
        atomic_write_new_json(
            temporary / "osera_manifest.json",
            manifest_payload,
            artifact_kind="OSERA manifest",
        )
        _assert_final_tree(temporary)
        if actual_output.exists():
            raise FileExistsError(f"refusing to overwrite OSERA output: {actual_output}")
        os.replace(temporary, actual_output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)

    manifest = json.loads(
        (actual_output / "osera_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "task_id": TASK_ID,
        "classification": manifest["classification"],
        "primary_gate": manifest["primary_gate"],
        "reason": manifest["reason"],
        "output": _output_reference(actual_output),
        "prediction_count": VALIDATION_COUNT,
        "accuracy": float(error_model["overall_accuracy"]),
        "error_count": int((~correctness).sum().item()),
        "per_class_minimum_lower95": bootstrap["minimum_lower95"],
        "global_null": {
            "real_accuracy": global_null["real_accuracy"],
            "shuffled_accuracy": global_null["shuffled_accuracy"],
            "point_delta": global_null["point_delta"],
            "lower95": global_null["lower95"],
            "upper95": global_null["upper95"],
        },
        "next_action": next_action,
        "model_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
