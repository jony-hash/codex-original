"""Resolve the frozen V2 training-source authority without scientific execution.

The consumer is deliberately limited to provenance and non-protected source
metadata.  It may publish a distinct correction authority and an equivalent
corrected contract only after earlier exact-source authorities establish one
canonical digest.  It never edits the original contract or the source file.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256, tensor_sha256


TASK_ID = "RESOLVE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_TRAINING_SOURCE_AUTHORITY_V1"
CLASSIFICATION = "TRAINING_SOURCE_AUTHORITY_CONTRACT_TRANSCRIPTION_ERROR_CONFIRMED"
CONTRACT_SCHEMA = "lightgen_2d_semantic_complex_ols_training_contract.v2"
ORIGINAL_CONTRACT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_contract_v2_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2.json"
)
ORIGINAL_CONTRACT_SHA256 = "b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53"
TRAINING_SOURCE = Path("data/MNIST/processed/training.pt")
MALFORMED_SHA256 = "8151bb33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
CANONICAL_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
SOURCE_DATA_IDENTITY_MANIFEST = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_frozen_output_semantic_evaluator_v1_attempt_01/"
    "data_identity_manifest.json"
)
SOURCE_DATA_IDENTITY_MANIFEST_SHA256 = "8ae23bf1f247124a1f1f9689a2a4cfb507860c49704d334adec158808d093dc4"
N3_SOURCE_MANIFEST = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_source_manifest_v1_attempt_01/"
    "n3_development_bank_v2_source_manifest.v1.json"
)
N3_SOURCE_MANIFEST_SHA256 = "4bc251af91b3a6cd0235fe113117443d871a564bd05634fa9e61b5ab0b36f812"
V1_CONTRACT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_contract_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v1.json"
)
V1_CONTRACT_SHA256 = "7d384fe9a73acc8f883badc4973926ab0e58bfce737067d4764deef10ee1d16f"
V1_PREFLIGHT_INPUT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_consumer_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_preflight_input.v1.json"
)
V1_PREFLIGHT_INPUT_SHA256 = "c16dc2b5a1ed1c8a4f0ba0fc2e5a3e90154620e3f053d54d79342cc434e85aa7"
V1_PREFLIGHT_REPORT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight.v1.json"
)
V1_PREFLIGHT_REPORT_SHA256 = "365573edb0a3b4525bb571e31241bba6ba4243b9b2a3f591a8d1aea722c48d8a"
V1_TRAINING_REPORT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_report.v1.json"
)
V1_TRAINING_REPORT_SHA256 = "f3f5fa876567513ec70cac67a6888aad6b55d140299992a535d0f9656bb4aa06"
V1_TRAINING_LINEAGE = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_lineage.v1.json"
)
V1_TRAINING_LINEAGE_SHA256 = "9c6eacb921b59788062236d9ba8354529f97bcf86de23813ba594e0fb7582c12"
POST_TRAINING_CONTRACT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_post_training_evaluation_contract_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_post_training_evaluation_contract.v1.json"
)
POST_TRAINING_CONTRACT_SHA256 = "b750bafd9c3866707a7d492c55c12ee08ab4775da9f6878805317feccb937a1d"

OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01"
)
CORRECTION_NAME = "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction.v1.json"
CORRECTED_CONTRACT_NAME = "lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_FIELD = "data_contract.source_artifacts.training.sha256"


class AuthorityResolutionBlocked(RuntimeError):
    """The current provenance cannot establish a unique canonical source."""


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityResolutionBlocked(message)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(root: Path, relative: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    path = root / relative
    _require(path.is_file(), f"{name} is unavailable: {relative.as_posix()}")
    actual = file_sha256(path)
    _require(actual == expected_sha256, f"{name} SHA mismatch: {actual}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthorityResolutionBlocked(f"{name} is not valid JSON") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _at(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        _require(isinstance(value, Mapping) and key in value, f"missing provenance field: {'.'.join(path)}")
        value = value[key]
    return value


def _git_facts(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_line_count": len(status.splitlines()),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _contract_identity(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / ORIGINAL_CONTRACT
    original = _read_json(root, ORIGINAL_CONTRACT, ORIGINAL_CONTRACT_SHA256, "original V2 contract")
    value = _at(original, ("data_contract", "source_artifacts", "training", "sha256"))
    _require(isinstance(value, str), "V2 training source authority is not a string")
    return original, {
        "path": _relative(root, path),
        "sha256": ORIGINAL_CONTRACT_SHA256,
        "schema": original.get("schema"),
        "field": SOURCE_FIELD,
        "value": value,
        "character_count": len(value),
        "sha256_regex_valid": bool(SHA256_RE.fullmatch(value)),
    }


def _training_identity(root: Path) -> dict[str, Any]:
    path = root / TRAINING_SOURCE
    _require(path.is_file(), f"training source is unavailable: {TRAINING_SOURCE.as_posix()}")
    before_sha256 = file_sha256(path)
    _require(before_sha256 == CANONICAL_SHA256, f"current training source differs from expected canonical SHA: {before_sha256}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise AuthorityResolutionBlocked("training source cannot be read for schema corroboration") from error
    _require(isinstance(payload, (tuple, list)) and len(payload) == 2, "training source container schema changed")
    images, labels = payload
    _require(torch.is_tensor(images) and torch.is_tensor(labels), "training source tensors are missing")
    _require(tuple(images.shape) == (60_000, 28, 28) and images.dtype == torch.uint8, "training image tensor schema changed")
    _require(tuple(labels.shape) == (60_000,) and labels.dtype == torch.int64, "training label tensor schema changed")
    _require(int(images.min()) == 0 and int(images.max()) == 255, "training image value range changed")
    _require(int(labels.min()) == 0 and int(labels.max()) == 9, "training label range changed")
    label_sha256 = tensor_sha256(labels)
    _require(label_sha256 == "5b15b6a6d7cf9f282941280ffeb569aca7640d0d7e78be939a9b1723e4f83549", "training label tensor identity changed")
    after_sha256 = file_sha256(path)
    _require(before_sha256 == after_sha256, "training source changed during read-only inspection")
    stat = path.stat()
    return {
        "path": _relative(root, path),
        "sha256": after_sha256,
        "size_bytes": stat.st_size,
        "last_write_time_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "dataset": "mnist_train",
        "row_count": 60_000,
        "images": {"shape": list(images.shape), "dtype": str(images.dtype), "value_range": [0, 255]},
        "labels": {"shape": list(labels.shape), "dtype": str(labels.dtype), "value_range": [0, 9], "tensor_sha256": label_sha256},
        "unchanged_before_after": True,
    }


def _authority_ref(root: Path, relative: Path, expected_sha256: str, *, authority_id: str, kind: str, claim_path: tuple[str, ...], claim_field: str, exact_source_role: bool, predates_v2: bool) -> dict[str, Any]:
    payload = _read_json(root, relative, expected_sha256, authority_id)
    claim = _at(payload, claim_path)
    _require(isinstance(claim, str) and claim == CANONICAL_SHA256, f"{authority_id} does not bind the canonical training source")
    _require(bool(SHA256_RE.fullmatch(claim)), f"{authority_id} claim is not a valid SHA-256")
    path = root / relative
    return {
        "authority_id": authority_id,
        "kind": kind,
        "path": _relative(root, path),
        "artifact_sha256": expected_sha256,
        "claim_path": ".".join(claim_path),
        "claim_field": claim_field,
        "claim_sha256": claim,
        "claim_syntax_valid": True,
        "exact_source_role": exact_source_role,
        "predates_v2_contract_materialization": predates_v2,
    }


def _supporting_source_identity(root: Path) -> dict[str, Any]:
    payload = _read_json(root, SOURCE_DATA_IDENTITY_MANIFEST, SOURCE_DATA_IDENTITY_MANIFEST_SHA256, "source data identity manifest")
    source = _at(payload, ("source_dataset_hashes",))
    _require(source.get("train_images_sha256") == "cc86acbde125e1596a9d290165c4345cff2294032e1bb5cf4aeaf8c6467322f0", "source image identity changed")
    _require(source.get("train_labels_sha256") == "5b15b6a6d7cf9f282941280ffeb569aca7640d0d7e78be939a9b1723e4f83549", "source label identity changed")
    return {
        "authority_id": "LIGHTGEN_SCALED_CHEN_SOURCE_DATA_IDENTITY_V1",
        "kind": "underlying raw dataset identity corroboration",
        "path": _relative(root, root / SOURCE_DATA_IDENTITY_MANIFEST),
        "artifact_sha256": SOURCE_DATA_IDENTITY_MANIFEST_SHA256,
        "dataset": "mnist_train",
        "train_images_sha256": source["train_images_sha256"],
        "train_labels_sha256": source["train_labels_sha256"],
        "exact_processed_source_sha256_claim": False,
    }


def _search_source_claims(root: Path) -> dict[str, Any]:
    scopes = ["outputs", "scripts", "configs", "docs"]
    command = [
        "rg",
        "-uuu",
        "-l",
        "-F",
        TRAINING_SOURCE.as_posix(),
        *scopes,
        "--glob",
        "*.json",
        "--glob",
        "*.py",
        "--glob",
        "*.yaml",
        "--glob",
        "*.yml",
        "--glob",
        "*.md",
    ]
    completed = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    files = [Path(line) for line in completed.stdout.splitlines() if line.strip()]
    observations: list[dict[str, Any]] = []
    observed_values: set[str] = set()

    def record(path: Path, value: str, claim_field: str) -> None:
        normalized = value.lower()
        observed_values.add(normalized)
        observations.append({"path": path.as_posix(), "claim_field": claim_field, "observed_sha256_value": normalized, "syntax_valid": bool(SHA256_RE.fullmatch(normalized))})

    def walk_json(value: Any, path: Path) -> None:
        if isinstance(value, Mapping):
            if value.get("path") == TRAINING_SOURCE.as_posix():
                for field in ("sha256", "actual_sha256", "expected_sha256"):
                    claim = value.get(field)
                    if isinstance(claim, str) and re.fullmatch(r"[0-9a-f]{63,64}", claim, re.IGNORECASE):
                        record(path, claim, field)
            for child in value.values():
                walk_json(child, path)
        elif isinstance(value, list):
            for child in value:
                walk_json(child, path)

    source_constant_pattern = re.compile(
        r"(?i)(?:TRAINING_DATA_SHA256|TRAINING_SOURCE_SHA256|SOURCE_LABELS_FILE_SHA256|labels_artifact_sha256)\s*(?:=|:)\s*[\"']?([0-9a-f]{63,64})"
    )
    for relative in files:
        path = root / relative
        if path.suffix.lower() == ".json":
            try:
                walk_json(json.loads(path.read_text(encoding="utf-8")), relative)
            except json.JSONDecodeError as error:
                raise AuthorityResolutionBlocked(f"source claim file is not valid JSON: {relative.as_posix()}") from error
        elif path.suffix.lower() in {".py", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8", errors="strict")
            if TRAINING_SOURCE.as_posix() in text:
                for line in text.splitlines():
                    match = source_constant_pattern.search(line)
                    if match:
                        record(relative, match.group(1), "source constant")
    unexpected_valid = sorted(value for value in observed_values if SHA256_RE.fullmatch(value) and value != CANONICAL_SHA256)
    _require(not unexpected_valid, f"competing valid source digests observed: {unexpected_valid}")
    return {
        "scopes": scopes,
        "files_with_exact_source_path": [path.as_posix() for path in files],
        "observed_sha256_values": sorted(observed_values),
        "observations": observations,
        "unexpected_valid_digests": unexpected_valid,
        "malformed_observed_values": sorted(value for value in observed_values if not SHA256_RE.fullmatch(value)),
        "competing_valid_canonical_digests": [],
    }


def _semantic_normal_form(payload: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(payload))
    value.pop("authority_correction", None)
    value["data_contract"]["source_artifacts"]["training"]["sha256"] = "<TRAINING_SOURCE_SHA256>"
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _corrected_contract(original: Mapping[str, Any], *, correction_path: str, provenance: list[dict[str, Any]], canonical: Mapping[str, Any], corrected_at: str) -> tuple[dict[str, Any], dict[str, Any]]:
    corrected = copy.deepcopy(dict(original))
    corrected["data_contract"]["source_artifacts"]["training"]["sha256"] = CANONICAL_SHA256
    corrected["authority_correction"] = {
        "correction_id": TASK_ID,
        "correction_classification": CLASSIFICATION,
        "correction_only": True,
        "corrected_at_utc": corrected_at,
        "original_contract": {"path": ORIGINAL_CONTRACT.as_posix(), "sha256": ORIGINAL_CONTRACT_SHA256},
        "corrected_source_field": SOURCE_FIELD,
        "original_malformed_value": MALFORMED_SHA256,
        "original_value_character_count": len(MALFORMED_SHA256),
        "canonical_training_source": {"path": TRAINING_SOURCE.as_posix(), "sha256": CANONICAL_SHA256},
        "provenance_authorities": [
            {"authority_id": item["authority_id"], "path": item["path"], "artifact_sha256": item["artifact_sha256"], "claim_sha256": item["claim_sha256"]}
            for item in provenance
        ],
        "source_file_unchanged": True,
        "scientific_semantics_unchanged": True,
        "original_contract_preserved": True,
        "correction_authority_path": correction_path,
        "canonical_source_size_bytes": canonical["size_bytes"],
    }
    comparison = {
        "original_semantic_normal_form_sha256": _semantic_normal_form(original),
        "corrected_semantic_normal_form_sha256": _semantic_normal_form(corrected),
        "changed_paths": [SOURCE_FIELD, "authority_correction"],
        "all_other_fields_unchanged": _semantic_normal_form(original) == _semantic_normal_form(corrected),
    }
    _require(comparison["all_other_fields_unchanged"], "corrected contract changed a field outside the bounded correction")
    return corrected, comparison


def resolve(repo_root: Path | None = None) -> dict[str, Any]:
    root = _root() if repo_root is None else repo_root.resolve()
    output_dir = root / OUTPUT_DIR
    _require(not output_dir.exists(), f"correction output already exists; refusing overwrite: {OUTPUT_DIR.as_posix()}")
    git_start = _git_facts(root)
    original, original_identity = _contract_identity(root)
    _require(original_identity["value"] == MALFORMED_SHA256, "materialized V2 contract does not contain the reported malformed value")
    _require(original_identity["character_count"] == 63 and not original_identity["sha256_regex_valid"], "V2 authority is not the expected malformed 63-character value")
    _require(original_identity["schema"] == CONTRACT_SCHEMA, "original V2 contract schema changed")
    canonical = _training_identity(root)
    provenance = [
        _authority_ref(root, N3_SOURCE_MANIFEST, N3_SOURCE_MANIFEST_SHA256, authority_id="N3_DEVELOPMENT_BANK_V2", kind="frozen exact-source machine manifest", claim_path=("source_domain", "source_labels_artifact", "sha256"), claim_field="source_domain.source_labels_artifact.sha256", exact_source_role=True, predates_v2=True),
        _authority_ref(root, V1_CONTRACT, V1_CONTRACT_SHA256, authority_id="DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V1_READY", kind="prior training contract", claim_path=("data_contract", "source_artifacts", "data/MNIST/processed/training.pt", "sha256"), claim_field="data_contract.source_artifacts[data/MNIST/processed/training.pt].sha256", exact_source_role=True, predates_v2=True),
        _authority_ref(root, V1_PREFLIGHT_INPUT, V1_PREFLIGHT_INPUT_SHA256, authority_id="LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V1_IMPLEMENTED", kind="prior consumer input authority", claim_path=("dataset", "source_artifacts", "training", "actual_sha256"), claim_field="dataset.source_artifacts.training.actual_sha256", exact_source_role=True, predates_v2=True),
        _authority_ref(root, V1_PREFLIGHT_REPORT, V1_PREFLIGHT_REPORT_SHA256, authority_id="LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_AUTHORIZABLE", kind="prior accepted target-device preflight", claim_path=("dataset", "source_artifacts", "training", "actual_sha256"), claim_field="dataset.source_artifacts.training.actual_sha256", exact_source_role=True, predates_v2=True),
        _authority_ref(root, V1_TRAINING_REPORT, V1_TRAINING_REPORT_SHA256, authority_id="LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1_VALID_COMPLETED", kind="prior completed training report", claim_path=("data", "source_artifacts", "training", "actual_sha256"), claim_field="data.source_artifacts.training.actual_sha256", exact_source_role=True, predates_v2=True),
        _authority_ref(root, V1_TRAINING_LINEAGE, V1_TRAINING_LINEAGE_SHA256, authority_id="LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1_LINEAGE", kind="prior completed training lineage", claim_path=("execution", "data", "source_artifacts", "training", "actual_sha256"), claim_field="execution.data.source_artifacts.training.actual_sha256", exact_source_role=True, predates_v2=True),
        _authority_ref(root, POST_TRAINING_CONTRACT, POST_TRAINING_CONTRACT_SHA256, authority_id="DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_CONTRACT_V1_READY", kind="downstream evaluation contract", claim_path=("bound_training_authority", "source_artifacts", "training", "sha256"), claim_field="bound_training_authority.source_artifacts.training.sha256", exact_source_role=True, predates_v2=True),
    ]
    supporting = _supporting_source_identity(root)
    search = _search_source_claims(root)
    correction_path = _relative(root, output_dir / CORRECTION_NAME)
    corrected, comparison = _corrected_contract(original, correction_path=correction_path, provenance=provenance, canonical=canonical, corrected_at=dt.datetime.now(dt.timezone.utc).isoformat())
    corrected_path = output_dir / CORRECTED_CONTRACT_NAME
    atomic_write_new_json(corrected_path, corrected, artifact_kind="corrected V2 training contract")
    corrected_sha256 = file_sha256(corrected_path)
    _require(corrected_sha256 != ORIGINAL_CONTRACT_SHA256, "corrected contract unexpectedly has original artifact identity")
    original_after_sha256 = file_sha256(root / ORIGINAL_CONTRACT)
    source_after_sha256 = file_sha256(root / TRAINING_SOURCE)
    _require(original_after_sha256 == ORIGINAL_CONTRACT_SHA256, "original V2 contract changed during correction")
    _require(source_after_sha256 == CANONICAL_SHA256, "training source changed during correction")
    correction = {
        "schema": "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction.v1",
        "version": 1,
        "task_id": TASK_ID,
        "gate": TASK_ID,
        "status": "valid_completed",
        "result": "CONFIRMED",
        "classification": CLASSIFICATION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "original_v2_contract": original_identity,
        "current_source": canonical,
        "provenance": {
            "decisive_exact_source_authorities": provenance,
            "supporting_source_identity": supporting,
            "repository_search": search,
            "competing_valid_canonical_sha256": [],
            "canonical_conclusion": "current training.pt is the established canonical source; the V2 contract field is a malformed transcription",
        },
        "corrected_contract": {
            "path": _relative(root, corrected_path),
            "sha256": corrected_sha256,
            "schema": corrected.get("schema"),
            "version": corrected.get("version"),
            "classification": corrected.get("classification"),
            "source_sha256": corrected["data_contract"]["source_artifacts"]["training"]["sha256"],
        },
        "semantic_equivalence": comparison,
        "correction_boundary": {
            "original_v2_contract_overwritten": False,
            "training_source_modified": False,
            "scientific_semantics_changed": False,
            "dataset_rows_or_split_changed": False,
            "consumer_rebind_required": True,
            "protected_test_read": False,
            "sealed_read": False,
            "cuda_executed": False,
            "optimizer_steps": 0,
            "scientific_training_executed": False,
            "generation_executed": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "training_authorized": False,
        },
        "provenance_integrity": {
            "original_contract_sha256_after": original_after_sha256,
            "training_source_sha256_after": source_after_sha256,
            "all_exact_source_claims_canonical": True,
            "canonical_sha256_syntax_valid": True,
        },
        "git": {"start": git_start, "end": _git_facts(root)},
        "artifact_self_hash": "not_embedded_to_avoid_self_reference",
    }
    atomic_write_new_json(output_dir / CORRECTION_NAME, correction, artifact_kind="V2 source authority correction")
    correction_sha256 = file_sha256(output_dir / CORRECTION_NAME)
    ending = _git_facts(root)
    _require(ending["branch"] == git_start["branch"] and ending["head"] == git_start["head"], "Git branch or HEAD changed during correction")
    return {
        "result": "VALID_COMPLETED",
        "classification": CLASSIFICATION,
        "correction_authority": {"path": str(output_dir / CORRECTION_NAME), "sha256": correction_sha256},
        "corrected_contract": {"path": str(corrected_path), "sha256": corrected_sha256},
        "original_contract_preserved": True,
        "training_source_preserved": True,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "next": "RERUN_PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_AFTER_CONSUMER_REBIND",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = resolve(args.repo_root)
    except AuthorityResolutionBlocked as error:
        print(json.dumps({"result": "BLOCKED", "classification": "TRAINING_SOURCE_AUTHORITY_UNRESOLVED", "message": str(error), "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
