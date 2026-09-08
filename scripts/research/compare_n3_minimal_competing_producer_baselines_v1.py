"""Compare two minimal unconditional N3 amplitude producer families.

This consumer uses only the existing 1000-row N3 development pool.  It does
not read the reserved confirmation source, create OLS states, invoke an
optical/evaluator path, or use CUDA.  Candidate parameters are fit inside a
deterministic five-fold development assignment and are never fit from a
held-out row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.evidence import (
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research.validate_n3_amplitude_producer_baseline import (
    _energy_distance as validated_energy_distance,
)


TASK_ID = "COMPARE_N3_MINIMAL_COMPETING_PRODUCER_BASELINES_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_minimal_competing_producer_baselines_report.v1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"

DATA_DIR = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_ols_producer_data_authority_v1_attempt_01"
)
SOURCE_MANIFEST = DATA_DIR / "ols_producer_source_manifest.v1.json"
PARTITION_MANIFEST = DATA_DIR / "ols_producer_partition_manifest.v1.json"
FIREWALL_MANIFEST = DATA_DIR / "ols_producer_data_firewall_manifest.v1.json"
FIT_BANK = DATA_DIR / "ols_producer_fit_bank.v1.pt"
INTERNAL_VALIDATION_BANK = DATA_DIR / "ols_producer_internal_validation_bank.v1.pt"
HISTORICAL_VALIDATION_REPORT = Path(
    "outputs/20260908_v2/runs/"
    "lightgen_n3_amplitude_producer_validation_v1_attempt_01/"
    "n3_amplitude_producer_validation_report.v1.json"
)
SEALED_CONFIRMATION_MANIFEST = Path(
    "outputs/20260908_v2/runs/"
    "n3_sealed_confirmation_source_v1_attempt_01/"
    "n3_sealed_confirmation_source_manifest.v1.json"
)

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "d07fc3fa927796e97254fa09a8c0eceb18d6555a99590db43217a685bf0d1687"
)
EXPECTED_PARTITION_MANIFEST_SHA256 = (
    "409b598aacae921da631adea553b339b56560231f9cb2cf75ce8d7e5b439242b"
)
EXPECTED_FIREWALL_MANIFEST_SHA256 = (
    "b711e9d3443f3a86c803ff1f8cae1ff266634439ebe0c0d31c04e807c3841837"
)
EXPECTED_FIT_BANK_SHA256 = (
    "21b989a13cbea8f24d3298424b200eb1727de8d04809bfbc77e1901daa888f77"
)
EXPECTED_INTERNAL_VALIDATION_BANK_SHA256 = (
    "8397c886db73bb260723debd6d4eab332b6172972a055b28f770e7ca2bf3ee25"
)
EXPECTED_HISTORICAL_VALIDATION_REPORT_SHA256 = (
    "16435dd72babf1f6abe537aa766c27fad79065f1cdf0ebfa6a9f54c4837aa6a0"
)
EXPECTED_SEALED_CONFIRMATION_MANIFEST_SHA256 = (
    "6475474f157d31591a3b0252f27320323d84ea93803c61ee6a365420a8ebb08e"
)
EXPECTED_SOURCE_ARTIFACT_SHA256 = (
    "9375b08597f03fee3798e05798779556c3e7534db6a9d39033232f7278d32279"
)
EXPECTED_SOURCE_INDEX_SHA256 = (
    "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d"
)
EXPECTED_SOURCE_LABEL_SHA256 = (
    "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
)
EXPECTED_SOURCE_R3_SHA256 = (
    "dcade4dca73f86f7bdc8d9be8fe183a8eda6a5f3fe2b810986d846ddf4c12fd3"
)
EXPECTED_PARTITION_IDENTITY_SHA256 = (
    "d1eaf777c27b98eb607e98ca41d2360115eb0b070156b29d21805c7332981d6d"
)

CLASS_COUNT = 10
DIMENSION = 100
POOL_COUNT = 1000
ROWS_PER_CLASS = 100
FOLD_COUNT = 5
FIT_COUNT_PER_FOLD = 800
HELDOUT_COUNT_PER_FOLD = 200
ROWS_PER_CLASS_PER_FOLD = 20
FIT_ROWS_PER_CLASS_PER_FOLD = 80
HELDOUT_ROWS_PER_CLASS_PER_FOLD = 20

FOLD_SALT = "N3_MINIMAL_COMPETING_PRODUCER_BASELINES_V1"
RNG_NAMESPACE = "N3_MINIMAL_COMPETING_PRODUCER_BASELINES_V1_CPU"
TARGET_ACCEPTED = 200
PROPOSAL_BUDGET = 1_000_000
PROPOSAL_BATCH_SIZE = 4096
DATA_VS_MODEL_REPLICATES = 4
MODEL_VS_MODEL_REPLICATES = 8
Q95 = 0.95

CANDIDATE_A = "SHRINKAGE_ELLIPTICAL_GAUSSIAN_AMPLITUDE_V1"
CANDIDATE_B = "RADIAL_SCALE_MIXTURE_AMPLITUDE_V1"
CLASSIFICATION_SELECTED = "N3_MINIMAL_PRODUCER_CANDIDATE_SELECTED"
CLASSIFICATION_NOT_SUPPORTED = "N3_MINIMAL_PRODUCER_BASELINES_NOT_SUPPORTED"
CLASSIFICATION_INCONCLUSIVE = "N3_MINIMAL_PRODUCER_COMPARISON_INCONCLUSIVE"

REPORT_NAME = "n3_minimal_competing_producer_baselines_report.v1.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _read_json_exact(repo_root: Path, path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{name} is missing: {target}")
    _require(file_sha256(target) == expected_sha256, f"{name} hash changed: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    return dict(_mapping(value, name))


def _git_state(repo_root: Path) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "branch": branch,
        "head": head,
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
        "worktree_clean": not status,
    }


def _verify_historical_boundaries(repo_root: Path) -> dict[str, Any]:
    historical = _read_json_exact(
        repo_root,
        HISTORICAL_VALIDATION_REPORT,
        EXPECTED_HISTORICAL_VALIDATION_REPORT_SHA256,
        "historical Gaussian validation report",
    )
    _require(
        historical.get("classification") == "N3_AMPLITUDE_PRODUCER_BASELINE_NOT_SUPPORTED",
        "historical Gaussian producer classification changed",
    )
    sealed = _read_json_exact(
        repo_root,
        SEALED_CONFIRMATION_MANIFEST,
        EXPECTED_SEALED_CONFIRMATION_MANIFEST_SHA256,
        "sealed confirmation reservation",
    )
    _require(sealed.get("status") == "RESERVED_NOT_MATERIALIZED", "sealed confirmation is materialized")
    _require(
        sealed.get("semantic_authority_id") == "N3_SEALED_CONFIRMATION_SOURCE_V1",
        "sealed confirmation authority changed",
    )
    return {
        "historical_gaussian": {
            "path": _relative(repo_root, repo_root / HISTORICAL_VALIDATION_REPORT),
            "sha256": EXPECTED_HISTORICAL_VALIDATION_REPORT_SHA256,
            "classification": historical["classification"],
        },
        "sealed_confirmation": {
            "path": _relative(repo_root, repo_root / SEALED_CONFIRMATION_MANIFEST),
            "sha256": EXPECTED_SEALED_CONFIRMATION_MANIFEST_SHA256,
            "status": sealed["status"],
            "semantic_authority_id": sealed["semantic_authority_id"],
            "sample_values_loaded": False,
            "materialized_by_this_consumer": False,
        },
    }


def _verify_authorities(repo_root: Path) -> dict[str, Any]:
    source = _read_json_exact(
        repo_root,
        SOURCE_MANIFEST,
        EXPECTED_SOURCE_MANIFEST_SHA256,
        "N3 source manifest",
    )
    partition = _read_json_exact(
        repo_root,
        PARTITION_MANIFEST,
        EXPECTED_PARTITION_MANIFEST_SHA256,
        "N3 partition manifest",
    )
    firewall = _read_json_exact(
        repo_root,
        FIREWALL_MANIFEST,
        EXPECTED_FIREWALL_MANIFEST_SHA256,
        "N3 firewall manifest",
    )
    _require(source.get("schema") == "ols_producer_source_manifest.v1", "source schema changed")
    _require(source.get("producer_lineage") == "OLS_LATENT_PRODUCER_V1", "producer lineage changed")
    _require(source.get("source_coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX", "source coordinate changed")
    source_artifact = _mapping(source.get("source_artifact"), "source artifact")
    _require(source_artifact.get("count") == POOL_COUNT, "N3 development pool count changed")
    _require(source_artifact.get("shape") == [POOL_COUNT, DIMENSION], "N3 development pool shape changed")
    _require(source_artifact.get("per_class") == ROWS_PER_CLASS, "N3 development pool class balance changed")
    _require(source_artifact.get("sha256") == EXPECTED_SOURCE_ARTIFACT_SHA256, "source artifact identity changed")
    _require(source_artifact.get("index_sha256") == EXPECTED_SOURCE_INDEX_SHA256, "source index authority changed")
    _require(source_artifact.get("label_sha256") == EXPECTED_SOURCE_LABEL_SHA256, "source label authority changed")
    _require(source_artifact.get("r3_complex_sha256") == EXPECTED_SOURCE_R3_SHA256, "source R3 authority changed")
    _require(partition.get("schema") == "ols_producer_partition_manifest.v1", "partition schema changed")
    _require(
        partition.get("partition_identity", {}).get("semantic_identity_sha256") == EXPECTED_PARTITION_IDENTITY_SHA256,
        "partition identity changed",
    )
    _require(partition.get("runtime_resplit") is False, "historical runtime resplit is enabled")
    _require(partition.get("runtime_reselection") is False, "historical runtime reselection is enabled")
    _require(firewall.get("schema") == "ols_producer_data_firewall_manifest.v1", "firewall schema changed")
    _require(firewall.get("firewall_pass") is True, "historical data firewall did not pass")
    future_consumer = _mapping(firewall.get("future_consumer"), "future consumer firewall")
    _require(future_consumer.get("must_verify_target_bank_identity") is True, "future identity check is disabled")
    _require(future_consumer.get("runtime_membership_mutation") is False, "runtime membership mutation is enabled")
    _require(future_consumer.get("runtime_reselection") is False, "runtime reselection is enabled")
    _require(future_consumer.get("runtime_resplit") is False, "runtime resplit is enabled")
    protected = firewall.get("protected_identities")
    _require(isinstance(protected, list) and protected, "protected identity registry is missing")
    for identity in protected:
        item = _mapping(identity, "protected identity")
        if item.get("independently_required") is True or item.get("status") == "PASS_ZERO_OVERLAP":
            _require(item.get("status") in {"PASS_ZERO_OVERLAP", "SUBSUMED"}, "protected identity firewall changed")
            if item.get("domain") == "SOURCE_IMAGE_INDEX_SPACE":
                _require(item.get("fit_overlap_count") == 0, "development fit overlaps protected source")
                _require(item.get("validation_overlap_count") == 0, "development validation overlaps protected source")
            if item.get("domain") == "CONTROLLER_CODE_SPACE_AND_SAVED_COMPLEX_OLS_STATE_SPACE":
                _require(item.get("fit_overlap_count") == 0, "development pool overlaps S3 code/state domain")
                _require(item.get("validation_overlap_count") == 0, "development pool overlaps S3 code/state domain")
    return {
        "source_manifest": {
            "path": _relative(repo_root, repo_root / SOURCE_MANIFEST),
            "sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "schema": source["schema"],
            "source_artifact_sha256": EXPECTED_SOURCE_ARTIFACT_SHA256,
        },
        "partition_manifest": {
            "path": _relative(repo_root, repo_root / PARTITION_MANIFEST),
            "sha256": EXPECTED_PARTITION_MANIFEST_SHA256,
            "semantic_identity_sha256": EXPECTED_PARTITION_IDENTITY_SHA256,
        },
        "firewall_manifest": {
            "path": _relative(repo_root, repo_root / FIREWALL_MANIFEST),
            "sha256": EXPECTED_FIREWALL_MANIFEST_SHA256,
            "firewall_pass": True,
            "new_sample_set_introduced": False,
            "protected_source_overlap_rechecked_from_authority": True,
        },
        "fit_bank": {
            "path": _relative(repo_root, repo_root / FIT_BANK),
            "sha256": EXPECTED_FIT_BANK_SHA256,
        },
        "internal_validation_bank": {
            "path": _relative(repo_root, repo_root / INTERNAL_VALIDATION_BANK),
            "sha256": EXPECTED_INTERNAL_VALIDATION_BANK_SHA256,
        },
        "coordinate": "R2_AMPLITUDE",
        "coordinate_definition": "q=abs(c/sqrt(s))=abs(R3_COMPLEX)",
        "producer_lineage": "OLS_LATENT_PRODUCER_V1",
        "protected_identity_count": len(protected),
    }


def _load_bank(repo_root: Path, path: Path, expected_sha256: str, role: str) -> dict[str, torch.Tensor]:
    payload = safe_torch_load(
        repo_root / path,
        expected_sha256=expected_sha256,
        map_location="cpu",
    )
    bank = _mapping(payload, role)
    _require(bank.get("schema") == "ols_producer_bank.v1", f"{role} schema changed")
    _require(bank.get("producer_lineage") == "OLS_LATENT_PRODUCER_V1", f"{role} lineage changed")
    _require(bank.get("source_coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX", f"{role} coordinate changed")
    _require(bank.get("count") == POOL_COUNT // 2 and bank.get("N") == DIMENSION, f"{role} count changed")
    _require(bank.get("dtype") == "torch.complex64", f"{role} dtype changed")
    _require(
        bank.get("partition", {}).get("semantic_identity_sha256") == EXPECTED_PARTITION_IDENTITY_SHA256,
        f"{role} partition identity changed",
    )
    result: dict[str, torch.Tensor] = {}
    for name, dtype, shape in (
        ("source_row_positions", torch.int64, (POOL_COUNT // 2,)),
        ("source_indices", torch.int64, (POOL_COUNT // 2,)),
        ("labels", torch.int64, (POOL_COUNT // 2,)),
        ("r3_complex", torch.complex64, (POOL_COUNT // 2, DIMENSION)),
    ):
        value = bank.get(name)
        _require(torch.is_tensor(value), f"{role} {name} is missing")
        _require(tuple(value.shape) == shape and value.dtype == dtype, f"{role} {name} shape/dtype changed")
        _require(bool(torch.isfinite(value).all()), f"{role} {name} is non-finite")
        result[name] = value.detach().cpu().contiguous()
    return result


def _load_development_pool(repo_root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    fit = _load_bank(repo_root, FIT_BANK, EXPECTED_FIT_BANK_SHA256, "fit bank")
    validation = _load_bank(
        repo_root,
        INTERNAL_VALIDATION_BANK,
        EXPECTED_INTERNAL_VALIDATION_BANK_SHA256,
        "internal validation bank",
    )
    positions = torch.cat([fit["source_row_positions"], validation["source_row_positions"]])
    indices = torch.cat([fit["source_indices"], validation["source_indices"]])
    labels = torch.cat([fit["labels"], validation["labels"]])
    r3 = torch.cat([fit["r3_complex"], validation["r3_complex"]])
    order = torch.argsort(positions, stable=True)
    positions = positions.index_select(0, order).contiguous()
    indices = indices.index_select(0, order).contiguous()
    labels = labels.index_select(0, order).contiguous()
    r3 = r3.index_select(0, order).contiguous()
    _require(torch.equal(positions, torch.arange(POOL_COUNT, dtype=torch.int64)), "development rows are not complete")
    _require(torch.unique(indices).numel() == POOL_COUNT, "development source indices are not unique")
    _require(tensor_sha256(indices) == EXPECTED_SOURCE_INDEX_SHA256, "development source index identity changed")
    _require(tensor_sha256(labels) == EXPECTED_SOURCE_LABEL_SHA256, "development label identity changed")
    _require(tensor_sha256(r3) == EXPECTED_SOURCE_R3_SHA256, "development R3 identity changed")
    _require(
        torch.equal(torch.bincount(labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), ROWS_PER_CLASS, dtype=torch.int64)),
        "development class balance changed",
    )
    q = r3.abs().to(dtype=torch.float64).contiguous()
    _require(tuple(q.shape) == (POOL_COUNT, DIMENSION), "development q shape changed")
    _require(bool(torch.isfinite(q).all()) and bool(q.ge(0.0).all()), "development q is invalid")
    return q, indices, labels, {
        "count": POOL_COUNT,
        "dimension": DIMENSION,
        "per_class": ROWS_PER_CLASS,
        "source_indices_sha256": tensor_sha256(indices),
        "labels_sha256": tensor_sha256(labels),
        "r3_complex_sha256": tensor_sha256(r3),
        "q_r2_amplitude_sha256": tensor_sha256(q),
        "source_row_positions_sha256": tensor_sha256(positions),
        "fit_and_internal_validation_union": True,
        "historical_partition_identities_unchanged": True,
    }


def _stable_row_digest(source_index: int, label: int) -> str:
    material = f"mnist_train|{source_index}|{label}|{FOLD_SALT}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _assign_folds(source_indices: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    _require(tuple(source_indices.shape) == (POOL_COUNT,), "fold source-index shape is invalid")
    _require(tuple(labels.shape) == (POOL_COUNT,), "fold label shape is invalid")
    fold_ids = torch.full((POOL_COUNT,), -1, dtype=torch.int64)
    for label in range(CLASS_COUNT):
        positions = (labels == label).nonzero(as_tuple=False).reshape(-1).tolist()
        ordered = sorted(
            positions,
            key=lambda position: (
                _stable_row_digest(int(source_indices[position]), label),
                int(source_indices[position]),
            ),
        )
        for rank, position in enumerate(ordered):
            fold_ids[position] = rank % FOLD_COUNT
    _require(bool(fold_ids.ge(0).all()), "fold assignment is incomplete")
    _require(torch.unique(fold_ids).numel() == FOLD_COUNT, "fold assignment has wrong fold count")
    fold_hash_payload = {
        "algorithm": "within_class_sha256_sort_round_robin_v1",
        "dataset": "mnist_train",
        "salt": FOLD_SALT,
        "source_indices": [int(value) for value in source_indices.tolist()],
        "labels": [int(value) for value in labels.tolist()],
        "fold_id_by_source_row": [int(value) for value in fold_ids.tolist()],
    }
    folds: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        held_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1)
        fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
        held_labels = labels.index_select(0, held_positions)
        fit_labels = labels.index_select(0, fit_positions)
        _require(held_positions.numel() == HELDOUT_COUNT_PER_FOLD, "held-out fold count changed")
        _require(fit_positions.numel() == FIT_COUNT_PER_FOLD, "fit fold count changed")
        _require(
            torch.equal(
                torch.bincount(held_labels, minlength=CLASS_COUNT),
                torch.full((CLASS_COUNT,), HELDOUT_ROWS_PER_CLASS_PER_FOLD, dtype=torch.int64),
            ),
            "held-out fold class balance changed",
        )
        _require(
            torch.equal(
                torch.bincount(fit_labels, minlength=CLASS_COUNT),
                torch.full((CLASS_COUNT,), FIT_ROWS_PER_CLASS_PER_FOLD, dtype=torch.int64),
            ),
            "fit fold class balance changed",
        )
        folds.append(
            {
                "fold": fold,
                "fit_count": int(fit_positions.numel()),
                "heldout_count": int(held_positions.numel()),
                "fit_per_class": FIT_ROWS_PER_CLASS_PER_FOLD,
                "heldout_per_class": HELDOUT_ROWS_PER_CLASS_PER_FOLD,
                "fit_source_index_sha256": tensor_sha256(source_indices.index_select(0, fit_positions)),
                "heldout_source_index_sha256": tensor_sha256(source_indices.index_select(0, held_positions)),
                "fit_label_sha256": tensor_sha256(labels.index_select(0, fit_positions)),
                "heldout_label_sha256": tensor_sha256(labels.index_select(0, held_positions)),
                "fit_positions_sha256": tensor_sha256(fit_positions),
                "heldout_positions_sha256": tensor_sha256(held_positions),
                "fit_heldout_disjoint": True,
            }
        )
    return fold_ids, {
        "algorithm": "within_class_sha256_sort_round_robin_v1",
        "salt": FOLD_SALT,
        "source_identity": "dataset-qualified mnist_train source index plus label",
        "fold_count": FOLD_COUNT,
        "fold_id_by_source_row": [int(value) for value in fold_ids.tolist()],
        "assignment_sha256": canonical_sha256(fold_hash_payload),
        "folds": folds,
        "q_values_used": False,
        "ols_values_used": False,
        "evaluator_values_used": False,
        "image_values_used": False,
    }


def _fit_shrinkage(q_fit: torch.Tensor) -> dict[str, Any]:
    values = q_fit.to(dtype=torch.float64).contiguous()
    _require(values.ndim == 2 and values.shape[1] == DIMENSION and values.shape[0] >= 2, "fit q shape is invalid")
    _require(bool(torch.isfinite(values).all()) and bool(values.ge(0.0).all()), "fit q is invalid")
    n = values.shape[0]
    mu = values.mean(dim=0)
    centered = values - mu
    covariance = centered.transpose(0, 1).mm(centered).div(float(n - 1))
    tau = torch.trace(covariance).div(float(DIMENSION))
    identity = torch.eye(DIMENSION, dtype=torch.float64)
    target = tau * identity
    quadratic = (centered.mm(covariance) * centered).sum(dim=1)
    numerator = (
        centered.square().sum(dim=1).square().sum()
        - 2.0 * quadratic.sum()
        + float(n) * covariance.square().sum()
    ).div(float(n * n))
    _require(bool(torch.isfinite(numerator)) and bool(torch.isfinite(tau)), "shrinkage plug-in is non-finite")
    numerator = numerator.clamp_min(0.0)
    delta = (covariance - target).square().sum()
    _require(bool(torch.isfinite(delta)), "shrinkage target distance is non-finite")
    if float(delta) == 0.0:
        alpha = torch.zeros((), dtype=torch.float64)
    else:
        alpha = (numerator / delta).clamp(0.0, 1.0)
    sigma = ((1.0 - alpha) * covariance + alpha * target).contiguous()
    sigma = (sigma + sigma.transpose(0, 1)).mul(0.5).contiguous()
    chol, info = torch.linalg.cholesky_ex(sigma, check_errors=False)
    _require(int(info.item()) == 0, "shrinkage covariance is not positive definite")
    eigenvalues = torch.linalg.eigvalsh(sigma)
    _require(bool(torch.isfinite(eigenvalues).all()) and bool(eigenvalues.gt(0.0).all()), "shrinkage covariance spectrum is invalid")
    return {
        "mu": mu,
        "sample_covariance_unbiased": covariance,
        "tau": tau,
        "alpha": alpha,
        "sigma": sigma,
        "chol": chol,
        "fit_count": int(n),
        "shrinkage_formula": "alpha=clip(beta/delta,0,1); beta=n^-2 sum_i ||x_i x_i^T-S||_F^2; delta=||S-tau I||_F^2",
    }


def _fit_radial_extension(fit: Mapping[str, Any]) -> dict[str, Any]:
    q_fit = fit["q_fit"]  # type: ignore[index]
    values = q_fit - fit["mu"]  # type: ignore[operator]
    whitened = torch.linalg.solve_triangular(fit["chol"], values.transpose(0, 1), upper=False).transpose(0, 1)  # type: ignore[index]
    radial = whitened.square().sum(dim=1)
    _require(bool(torch.isfinite(radial).all()) and bool(radial.gt(0.0).all()), "fit radial statistic is invalid")
    radial_scale = torch.tensor(float(DIMENSION), dtype=torch.float64).div(radial.mean())
    normalized = radial * radial_scale
    radial_mean = normalized.mean()
    radial_variance = normalized.var(unbiased=True)
    _require(bool(torch.isfinite(radial_mean)) and bool(torch.isfinite(radial_variance)), "fit radial moments are non-finite")
    if float(radial_variance) <= float(2 * DIMENSION):
        return {
            "status": "RADIAL_EXTENSION_NOT_IDENTIFIED_ON_FIT",
            "radial_mean_before_scale": float(radial.mean()),
            "radial_scale": float(radial_scale),
            "radial_mean_after_scale": float(radial_mean),
            "radial_variance": float(radial_variance),
            "degrees_of_freedom": None,
        }
    denominator = radial_variance - float(2 * DIMENSION)
    numerator = 4.0 * radial_variance + 2.0 * DIMENSION * (DIMENSION - 2.0)
    nu = numerator / denominator
    _require(torch.isfinite(torch.tensor(nu)) and nu > 4.0, "radial degrees of freedom is invalid")
    return {
        "status": "RADIAL_EXTENSION_IDENTIFIED_ON_FIT",
        "radial_mean_before_scale": float(radial.mean()),
        "radial_scale": float(radial_scale),
        "radial_mean_after_scale": float(radial_mean),
        "radial_variance": float(radial_variance),
        "degrees_of_freedom": float(nu),
    }


def _fit_fold(q_fit: torch.Tensor) -> dict[str, Any]:
    fitted = _fit_shrinkage(q_fit)
    fitted["q_fit"] = q_fit.to(dtype=torch.float64).contiguous()
    fitted["radial"] = _fit_radial_extension(fitted)
    return fitted


def _seed(candidate: str, role: str, fold: int, replicate: int) -> int:
    material = f"{RNG_NAMESPACE}|{candidate}|{role}|fold={fold}|replicate={replicate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _sample_positive(
    fit: Mapping[str, Any],
    candidate: str,
    *,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mu = fit["mu"]
    chol = fit["chol"]
    radial = fit["radial"]
    nu = radial.get("degrees_of_freedom")
    accepted: list[torch.Tensor] = []
    accepted_count = 0
    proposals = 0
    termination = "proposal_budget_exhausted"
    while proposals < PROPOSAL_BUDGET and accepted_count < TARGET_ACCEPTED:
        block_count = min(PROPOSAL_BATCH_SIZE, PROPOSAL_BUDGET - proposals)
        gaussian = torch.randn((block_count, DIMENSION), generator=generator, dtype=torch.float64)
        q = mu + gaussian.mm(chol.transpose(0, 1))
        if candidate == CANDIDATE_B:
            _require(nu is not None, "radial candidate has no identified degrees of freedom")
            shape = torch.full((block_count,), float(nu) / 2.0, dtype=torch.float64)
            chi_square = 2.0 * torch._standard_gamma(shape, generator=generator)
            q = mu + torch.sqrt((float(nu) - 2.0) / chi_square).unsqueeze(1) * gaussian.mm(chol.transpose(0, 1))
        valid = torch.isfinite(q).all(dim=1) & q.ge(0.0).all(dim=1)
        positions = valid.nonzero(as_tuple=False).reshape(-1)
        needed = TARGET_ACCEPTED - accepted_count
        if positions.numel() >= needed:
            selected = positions[:needed]
            accepted.append(q.index_select(0, selected).contiguous())
            proposals += int(selected[-1].item()) + 1
            accepted_count += needed
            termination = "target_accepted_reached"
            break
        if positions.numel() > 0:
            accepted.append(q.index_select(0, positions).contiguous())
            accepted_count += int(positions.numel())
        proposals += block_count
    result = torch.cat(accepted, dim=0) if accepted else torch.empty((0, DIMENSION), dtype=torch.float64)
    _require(tuple(result.shape) == (accepted_count, DIMENSION), "accepted q shape changed")
    if accepted_count:
        _require(bool(torch.isfinite(result).all()) and bool(result.ge(0.0).all()), "accepted q violates positivity")
    return result, {
        "seed": seed,
        "rng": "torch.Generator(device='cpu')",
        "target_accepted": TARGET_ACCEPTED,
        "proposal_budget": PROPOSAL_BUDGET,
        "proposal_batch_size": PROPOSAL_BATCH_SIZE,
        "proposal_count": proposals,
        "accepted_count": accepted_count,
        "rejected_count": proposals - accepted_count,
        "acceptance_fraction": float(accepted_count / proposals) if proposals else 0.0,
        "termination": termination,
        "feasible": accepted_count == TARGET_ACCEPTED,
    }


def _covariance(value: torch.Tensor) -> torch.Tensor:
    view = value.to(dtype=torch.float64).contiguous()
    _require(view.ndim == 2 and view.shape[0] >= 2, "metric covariance input is invalid")
    centered = view - view.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1).mm(centered).div(float(view.shape[0] - 1))
    _require(bool(torch.isfinite(covariance).all()), "metric covariance is non-finite")
    return covariance


def _pair_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    left = first.to(dtype=torch.float64).contiguous()
    right = second.to(dtype=torch.float64).contiguous()
    _require(left.shape == right.shape and left.ndim == 2, "metric sample shapes differ")
    left_cov = _covariance(left)
    right_cov = _covariance(right)
    right_cov_norm = right_cov.square().sum()
    _require(bool(torch.isfinite(right_cov_norm)) and bool(right_cov_norm.gt(0.0)), "metric reference covariance is zero")
    left_norm = left.norm(dim=1).unsqueeze(1)
    right_norm = right.norm(dim=1).unsqueeze(1)
    left_energy = left.square().sum(dim=1).unsqueeze(1)
    right_energy = right.square().sum(dim=1).unsqueeze(1)
    return {
        "energy_distance": validated_energy_distance(left, right),
        "covariance_relative_frobenius": float((left_cov - right_cov).square().sum() / right_cov_norm),
        "norm_energy_distance": validated_energy_distance(left_norm, right_norm),
        "squared_norm_energy_distance": validated_energy_distance(left_energy, right_energy),
        "marginal_mean_l2": float((left.mean(dim=0) - right.mean(dim=0)).norm()),
        "marginal_std_l2": float((left.std(dim=0, unbiased=True) - right.std(dim=0, unbiased=True)).norm()),
    }


def _mean_metrics(metrics: list[Mapping[str, float]]) -> dict[str, float]:
    _require(metrics, "metric list is empty")
    names = tuple(metrics[0])
    return {name: float(sum(float(item[name]) for item in metrics) / len(metrics)) for name in names}


def _quantile_metrics(metrics: list[Mapping[str, float]], probability: float) -> dict[str, float]:
    _require(metrics, "reference metric list is empty")
    names = tuple(metrics[0])
    return {
        name: float(torch.quantile(torch.tensor([float(item[name]) for item in metrics], dtype=torch.float64), probability))
        for name in names
    }


def _evaluate_candidate(
    q: torch.Tensor,
    fold_ids: torch.Tensor,
    fold_fits: list[dict[str, Any]],
    candidate: str,
) -> dict[str, Any]:
    fold_results: list[dict[str, Any]] = []
    for fold, fit in enumerate(fold_fits):
        heldout_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1)
        heldout = q.index_select(0, heldout_positions).contiguous()
        fit_summary = {
            "mu_sha256": tensor_sha256(fit["mu"]),
            "sigma_A_sha256": tensor_sha256(fit["sigma"]),
            "alpha": float(fit["alpha"]),
            "tau": float(fit["tau"]),
            "sigma_trace": float(torch.trace(fit["sigma"])),
            "sigma_min_eigenvalue": float(torch.linalg.eigvalsh(fit["sigma"]).min()),
            "fit_count": fit["fit_count"],
            "heldout_count": int(heldout_positions.numel()),
        }
        if candidate == CANDIDATE_B:
            radial = fit["radial"]
            if radial["status"] != "RADIAL_EXTENSION_IDENTIFIED_ON_FIT":
                fold_results.append(
                    {
                        "fold": fold,
                        "operationally_valid": False,
                        "compatibility": False,
                        "fit": fit_summary,
                        "radial": radial,
                        "sampling": [],
                        "reason": radial["status"],
                    }
                )
                continue
        sampling_records: list[dict[str, Any]] = []
        data_metrics: list[dict[str, float]] = []
        for replicate in range(DATA_VS_MODEL_REPLICATES):
            model_q, sampling = _sample_positive(
                fit,
                candidate,
                seed=_seed(candidate, "data-vs-model", fold, replicate),
            )
            sampling_records.append({"role": "data-vs-model", "replicate": replicate, **sampling})
            if not sampling["feasible"]:
                continue
            data_metrics.append(_pair_metrics(heldout, model_q))
        model_banks: list[torch.Tensor] = []
        for replicate in range(MODEL_VS_MODEL_REPLICATES):
            model_q, sampling = _sample_positive(
                fit,
                candidate,
                seed=_seed(candidate, "model-vs-model", fold, replicate),
            )
            sampling_records.append({"role": "model-vs-model", "replicate": replicate, **sampling})
            if sampling["feasible"]:
                model_banks.append(model_q)
        operationally_valid = (
            len(data_metrics) == DATA_VS_MODEL_REPLICATES
            and len(model_banks) == MODEL_VS_MODEL_REPLICATES
        )
        if not operationally_valid:
            fold_results.append(
                {
                    "fold": fold,
                    "operationally_valid": False,
                    "compatibility": False,
                    "fit": fit_summary,
                    "radial": fit["radial"],
                    "sampling": sampling_records,
                    "reason": "FIXED_POSITIVITY_BANK_NOT_REACHED",
                }
            )
            continue
        reference_pairs = [
            _pair_metrics(model_banks[index], model_banks[index + 1])
            for index in range(0, MODEL_VS_MODEL_REPLICATES, 2)
        ]
        data_mean = _mean_metrics(data_metrics)
        reference_q95 = _quantile_metrics(reference_pairs, Q95)
        guardrail_names = (
            "energy_distance",
            "covariance_relative_frobenius",
            "norm_energy_distance",
            "squared_norm_energy_distance",
        )
        component_checks = {
            name: data_mean[name] <= reference_q95[name]
            for name in guardrail_names
        }
        fold_results.append(
            {
                "fold": fold,
                "operationally_valid": True,
                "compatibility": all(component_checks.values()),
                "fit": fit_summary,
                "radial": fit["radial"],
                "sampling": sampling_records,
                "data_vs_model_replicates": data_metrics,
                "data_vs_model_mean": data_mean,
                "model_vs_model_reference_pairs": reference_pairs,
                "model_vs_model_reference_q95": reference_q95,
                "compatibility_rule": "fixed-replicate mean DATA_VS_MODEL <= q95 MODEL_VS_MODEL for primary plus secondary guardrails",
                "compatibility_components": component_checks,
            }
        )
    valid_folds = [item for item in fold_results if item["operationally_valid"]]
    compatibility_count = sum(bool(item["compatibility"]) for item in fold_results)
    primary_values = [item["data_vs_model_mean"]["energy_distance"] for item in valid_folds]
    covariance_values = [item["data_vs_model_mean"]["covariance_relative_frobenius"] for item in valid_folds]
    norm_values = [
        item["data_vs_model_mean"]["norm_energy_distance"]
        + item["data_vs_model_mean"]["squared_norm_energy_distance"]
        for item in valid_folds
    ]
    return {
        "candidate": candidate,
        "operationally_valid": len(valid_folds) == FOLD_COUNT,
        "compatible": len(valid_folds) == FOLD_COUNT and compatibility_count == FOLD_COUNT,
        "compatible_fold_count": compatibility_count,
        "primary_mean": float(sum(primary_values) / len(primary_values)) if primary_values else None,
        "covariance_mean": float(sum(covariance_values) / len(covariance_values)) if covariance_values else None,
        "norm_guardrail_mean": float(sum(norm_values) / len(norm_values)) if norm_values else None,
        "folds": fold_results,
    }


def _select_candidate(candidate_a: Mapping[str, Any], candidate_b: Mapping[str, Any]) -> dict[str, Any]:
    a_compatible = bool(candidate_a["compatible"])
    b_compatible = bool(candidate_b["compatible"])
    if a_compatible and not b_compatible:
        return {
            "classification": CLASSIFICATION_SELECTED,
            "selected_candidate": CANDIDATE_A,
            "rationale": "exactly one candidate satisfies all fixed fold compatibility checks",
        }
    if b_compatible and not a_compatible:
        return {
            "classification": CLASSIFICATION_SELECTED,
            "selected_candidate": CANDIDATE_B,
            "rationale": "exactly one candidate satisfies all fixed fold compatibility checks",
        }
    if a_compatible and b_compatible:
        a_primary = float(candidate_a["primary_mean"])
        b_primary = float(candidate_b["primary_mean"])
        b_primary_improves = sum(
            float(b_item["data_vs_model_mean"]["energy_distance"])
            < float(a_item["data_vs_model_mean"]["energy_distance"])
            for a_item, b_item in zip(candidate_a["folds"], candidate_b["folds"])
            if a_item.get("operationally_valid") and b_item.get("operationally_valid")
        )
        guardrails_not_worse = (
            float(candidate_b["covariance_mean"]) <= float(candidate_a["covariance_mean"])
            and float(candidate_b["norm_guardrail_mean"]) <= float(candidate_a["norm_guardrail_mean"])
        )
        if b_primary_improves >= 4 and b_primary < a_primary and guardrails_not_worse:
            return {
                "classification": CLASSIFICATION_SELECTED,
                "selected_candidate": CANDIDATE_B,
                "rationale": "B improves primary in at least four folds, has lower aggregate primary, and has no aggregate secondary guardrail degradation",
                "b_primary_improving_fold_count": b_primary_improves,
                "b_guardrails_not_worse": guardrails_not_worse,
            }
        return {
            "classification": CLASSIFICATION_SELECTED,
            "selected_candidate": CANDIDATE_A,
            "rationale": "A is the simpler default because B does not meet the preregistered clear-dominance rule",
            "b_primary_improving_fold_count": b_primary_improves,
            "b_guardrails_not_worse": guardrails_not_worse,
        }
    partial_compatibility = (
        any(bool(item["compatibility"]) for item in candidate_a["folds"])
        or any(bool(item["compatibility"]) for item in candidate_b["folds"])
    )
    if partial_compatibility and candidate_a["operationally_valid"] and candidate_b["operationally_valid"]:
        return {
            "classification": CLASSIFICATION_INCONCLUSIVE,
            "selected_candidate": None,
            "rationale": "both operationally valid candidates have mixed cross-fold compatibility and neither is globally compatible",
        }
    return {
        "classification": CLASSIFICATION_NOT_SUPPORTED,
        "selected_candidate": None,
        "rationale": "no candidate satisfies the fixed five-fold compatibility contract",
    }


def _candidate_specifications() -> dict[str, Any]:
    return {
        "candidate_a": {
            "family": CANDIDATE_A,
            "complexity_role": "simpler_default",
            "fit_inputs": "fold fit q only",
            "parameters": "mu=fit mean; S=unbiased covariance; tau=trace(S)/d; Sigma_A=(1-alpha)S+alpha*tau*I",
            "shrinkage_alpha": "analytic fit-only plug-in alpha=clip(beta/delta,0,1), no held-out tuning",
            "proposal": "q~Normal(mu,Sigma_A)",
            "acceptance": "finite(q) and q>=0 elementwise; no nonlinear repair",
        },
        "candidate_b": {
            "family": CANDIDATE_B,
            "fit_inputs": "same fold fit q, mu and Sigma_A as candidate A",
            "radial_fit": "whiten centered fit rows with Sigma_A; rescale radial mean to d; unbiased radial variance V_R",
            "degrees_of_freedom": "nu=[4 V_R + 2 d(d-2)]/[V_R-2d] when V_R>2d and finite nu>4",
            "not_identified": "V_R<=2d -> RADIAL_EXTENSION_NOT_IDENTIFIED_ON_FIT; no parameter invented",
            "proposal": "q=mu+sqrt((nu-2)/u)g; g~N(0,Sigma_A), u~ChiSquare(nu)",
            "acceptance": "same finite(q) and q>=0 elementwise predicate; no nonlinear repair",
        },
        "shared": {
            "coordinate": "R2_AMPLITUDE",
            "dimension": DIMENSION,
            "labels_as_model_inputs": False,
            "fit_data_only": True,
            "optimizer": False,
            "likelihood_tuning": False,
        },
    }


def _write_report(repo_root: Path, output_dir: Path, report: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(temporary / REPORT_NAME, report, artifact_kind="N3 producer comparison report")
        if output_dir.exists():
            raise FileExistsError(f"write-once output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / REPORT_NAME


def execute(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 research authority branch")
    authorities = _verify_authorities(repo_root)
    boundaries = _verify_historical_boundaries(repo_root)
    q, source_indices, labels, pool_identity = _load_development_pool(repo_root)
    fold_ids, fold_identity = _assign_folds(source_indices, labels)
    fold_fits: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
        fit = _fit_fold(q.index_select(0, fit_positions).contiguous())
        fit["fit_positions"] = fit_positions
        fold_fits.append(fit)
    candidate_a = _evaluate_candidate(q, fold_ids, fold_fits, CANDIDATE_A)
    candidate_b = _evaluate_candidate(q, fold_ids, fold_fits, CANDIDATE_B)
    selection = _select_candidate(candidate_a, candidate_b)
    script_path = Path(__file__).resolve()
    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": selection["classification"],
        "selection": selection,
        "git": {
            **git,
            "consumer_script": _relative(repo_root, script_path),
            "consumer_script_sha256": file_sha256(script_path),
        },
        "authorities": authorities,
        "historical_boundaries": boundaries,
        "development_pool": pool_identity,
        "development_fold_identity": fold_identity,
        "candidate_specifications": _candidate_specifications(),
        "rng": {
            "namespace": RNG_NAMESPACE,
            "device": "cpu",
            "generator": "torch.Generator(device='cpu')",
            "ambient_rng_used": False,
            "independent_substreams": ["candidate", "fold", "data-vs-model replicate", "model-vs-model replicate"],
            "seed_derivation": "sha256(namespace|candidate|role|fold|replicate) first 64 bits masked to signed 63-bit",
        },
        "sampling_contract": {
            "target_accepted": TARGET_ACCEPTED,
            "proposal_budget": PROPOSAL_BUDGET,
            "proposal_batch_size": PROPOSAL_BATCH_SIZE,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "budget_changed_after_acceptance_inspection": False,
        },
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "comparison": {
            "primary_metric": "validated_biased_empirical_energy_distance_euclidean_q_space",
            "primary_metric_provenance": "scripts.research.validate_n3_amplitude_producer_baseline._energy_distance",
            "secondary_metrics": [
                "covariance_relative_frobenius",
                "norm_energy_distance",
                "squared_norm_energy_distance",
                "marginal_mean_l2",
                "marginal_std_l2",
                "positivity_acceptance_fraction",
            ],
            "compatibility_rule": "fixed-replicate mean DATA_VS_MODEL <= q95 MODEL_VS_MODEL for primary plus covariance/norm guardrails",
            "model_vs_model_role": "finite-sample fluctuation reference only",
            "candidate_b_clear_dominance": "B improves primary in at least 4/5 folds, lower aggregate primary, and aggregate covariance/norm guardrails no worse; otherwise A default",
            "absolute_thresholds_used": False,
        },
        "claim_boundary": {
            "development_comparison_supported": True,
            "selected_candidate_is_development_only": selection["selected_candidate"] is not None,
            "new_family_scientifically_confirmed": False,
            "independent_confirmation": False,
            "intrinsic_ols_dimension": False,
            "full_complex_ols_distribution": False,
            "amplitude_phase_compatibility": False,
            "semantic_generation": False,
            "generation_readiness": "not_ready",
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "selection_authority": "DEVELOPMENT_ONLY",
        },
        "sealed_confirmation": {
            "status": "RESERVED_NOT_MATERIALIZED",
            "consumed": False,
            "materialized": False,
            "future_use_requires_separate_frozen_producer_confirmation_contract": True,
        },
        "not_executed": [
            "sealed confirmation source loading or materialization",
            "historical r=99 refit or reselection",
            "selector V3",
            "label-conditioned producer",
            "local mixture/MFA",
            "ICA",
            "normalizing flow",
            "deep producer training",
            "phase-source work",
            "novel complex OLS",
            "optical generator",
            "evaluator",
            "CUDA",
            "optimizer",
            "backward",
            "training",
        ],
    }
    report_path = _write_report(repo_root, output_dir, report)
    return {
        "classification": report["classification"],
        "selected_candidate": selection["selected_candidate"],
        "report": _relative(repo_root, report_path),
        "report_sha256": file_sha256(report_path),
        "git_head": git["head"],
        "worktree_clean_at_run": git["worktree_clean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/20260908_v2/runs/"
            "n3_minimal_competing_producer_baselines_v1_attempt_01"
        ),
        help="fresh write-once output directory",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
