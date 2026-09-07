"""Materialize the immutable N3 OLS-producer data authority.

This module only copies already-saved complex OLS rows into a fixed,
class-balanced fit/internal-validation partition.  It deliberately does not
construct an optical model, run a forward pass, create an optimizer, or train
anything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)


BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
EXPECTED_STARTING_HEAD = "9cb768006e525a5c84eb1faa9d3d7cf3863e2c6d"

SOURCE_MANIFEST_SCHEMA = "ols_producer_source_manifest.v1"
BANK_SCHEMA = "ols_producer_bank.v1"
PARTITION_MANIFEST_SCHEMA = "ols_producer_partition_manifest.v1"
FIREWALL_MANIFEST_SCHEMA = "ols_producer_data_firewall_manifest.v1"
S1_MANIFEST_SCHEMA = "lightgen_scaled_chen_ols_semantic_representation_audit.v1"
S1_EMBEDDINGS_SCHEMA = "lightgen_scaled_chen_ols_semantic_embeddings.v1"
S2_CORPUS_SCHEMA = "lightgen_scaled_chen_frozen_output_domain_corpus.v1"
S3_MANIFEST_SCHEMA = "lightgen_scaled_chen_controller_to_ols_writability_audit.v1"
S3_PANEL_SCHEMA = "lightgen_scaled_chen_controller_panel_identity.v1"
S3_STATE_SCHEMA = "lightgen_scaled_chen_controller_ols_states.v1"
CODE8_MANIFEST_SCHEMA = "lightgen_scaled_chen_code8_coverage_review.v1"

SOURCE_COORDINATE = "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX"
SOURCE_ARTIFACT_SHA256 = (
    "9375b08597f03fee3798e05798779556c3e7534db6a9d39033232f7278d32279"
)
SOURCE_SEMANTIC_MANIFEST_SHA256 = (
    "b0199040926618b17373edc4f167cc8bc4788f39f64de174e1a3fb225781833e"
)
SOURCE_INDEX_SHA256 = (
    "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d"
)
SOURCE_LABEL_SHA256 = (
    "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
)
SOURCE_R3_SHA256 = (
    "dcade4dca73f86f7bdc8d9be8fe183a8eda6a5f3fe2b810986d846ddf4c12fd3"
)
MODE_BANK_SHA256 = (
    "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9"
)
CHECKPOINT_SHA256 = (
    "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"
)
CALIBRATION_MANIFEST_SHA256 = (
    "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6"
)
CALIBRATION_PAYLOAD_SHA256 = (
    "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07"
)

S1_SELECTION_INDEX_SHA256 = (
    "11c076affa8511e883f595a0d1ee91e40f6d94747a351e21269520344d3f594f"
)
S1_CONFIRMATION_INDEX_SHA256 = (
    "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e7e5e2218a"
)
S2_VALIDATION_INDEX_SHA256 = (
    "5370b35f304a779cde7f978df6392b9974faaf4296e9edf75f9edf8202db02fa"
)
S2_VALIDATION_LABEL_SHA256 = (
    "13d90706eef49c3c9f17ccd39c002dc18a79eca1378e780548ea6b29adc6ec29"
)
S2_CORPUS_SHA256 = (
    "787b1b4e256f54ccebb55bb5582308f92148dc595d4cab4f632d3e87d72907b7"
)
S2_DATA_IDENTITY_FILE_SHA256 = (
    "8ae23bf1f247124a1f1f9689a2a4cfb507860c49704d334adec158808d093dc4"
)

FIT_ROW_POSITION_SHA256 = (
    "54fe39a51036f6b69cf4cb65cfc614e4043a92fb8a601c75f529da8f87122243"
)
VALIDATION_ROW_POSITION_SHA256 = (
    "964724eddd503d596ec870ceea3b0145f151ade64f5903307352297372763a75"
)
FIT_SOURCE_INDEX_SHA256 = (
    "99285b32a2c098de1788cd405bc317c3c7877bbd07bd1b63260fd9fd1cc93d18"
)
VALIDATION_SOURCE_INDEX_SHA256 = (
    "96a367399d53a48078d7a3328a6063174c2f58031177da0f14953191620b5d80"
)
PARTITION_LABEL_SHA256 = (
    "c8bacf260ae894e7da623586b9e04c67d95f628c16f52766cee0994ca2ceac19"
)
FIT_R3_SHA256 = (
    "f507dcaad007963178a07fcb13b55c267529b2adecc745df6c465316aa8732c0"
)
VALIDATION_R3_SHA256 = (
    "0c3887331553c474f5a39ab3653c8b0094ce7db8f416cff279627e001d68102c"
)
PARTITION_IDENTITY_SHA256 = (
    "d1eaf777c27b98eb607e98ca41d2360115eb0b070156b29d21805c7332981d6d"
)

S3_HISTORICAL_MANIFEST_SHA256 = (
    "0779d8d0ffb0a9958064bb3c05d451058cdd7a670a732d6ae1d4ba49c11eb5e6"
)
S3_HISTORICAL_STATE_SHA256 = (
    "45f9b2b18c0cc67bf784ee206059d97164abaab08f0304892c368f53d218b19a"
)
S3_POST_MANIFEST_SHA256 = (
    "e13603745ed9fc70ae4f47975241d7653629000090e89aef5cb38b5357e5018c"
)
S3_POST_STATE_SHA256 = (
    "1dcf9ed0b10fd49eedf54e07c52d64e4bf2c994e48588e75e4def49fa4c29800"
)
S3_PANEL_IDENTITY_SHA256 = (
    "8a9dd1c7e3e2b674a152e6c96ac8b47976ff1f8d27428a18477e89795b940f47"
)
S3_PANEL_FILE_SHA256 = (
    "517d2a826ebfcff44473d33dcf6abb15da68d69c9debe59b761179aa211ce5ec"
)
S3_BASE_CODE_SHA256 = (
    "2c83f8fca3d1fde7d5c694f69b602dbc3d43b4e41cfb8159d1af1914efef8c86"
)
S3_PANEL_CODE_SHA256 = (
    "24d72ca2926a707506f597c0cfc946a925f61bf623fbbb4a1d7ce2cd3bf30ca6"
)
S3_PANEL_LABEL_SHA256 = (
    "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
)
S3_PANEL_SPLIT_SHA256 = (
    "9a8a55618b3035b7244b2b522bd5a6f642b3ee0f386bf6207fd6d9ebd7a767bb"
)

CODE8_REVIEW_MANIFEST_SHA256 = (
    "d8f073987eda234ac9b85731dad66e9d9ffc92eaf896f700ff84ba4de17263fd"
)
INTERNAL_CODE_BASE_SHA256 = (
    "d6ff3077066bd0b7d809925ff94f7cc0e602fadc933a3246af80d0853945f356"
)
INTERNAL_CODE_PANEL_SHA256 = (
    "4a11c82e4235521c3181585220e8df1e92d34c49adaad29e6a7502f217f58695"
)
INTERNAL_CODE_LABEL_SHA256 = (
    "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
)
INTERNAL_CODE_SPLIT_SHA256 = (
    "9a8a55618b3035b7244b2b522bd5a6f642b3ee0f386bf6207fd6d9ebd7a767bb"
)

CLASS_COUNT = 10
SOURCE_ROWS_PER_CLASS = 100
PARTITION_ROWS_PER_CLASS = 50
PORT_COUNT = 100
PARTITION_ROW_COUNT = CLASS_COUNT * PARTITION_ROWS_PER_CLASS

S1_DIR = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01"
)
S2_DIR = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_frozen_output_semantic_evaluator_v1_attempt_01"
)
CALIBRATION_DIR = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02"
)
CHECKPOINT_PATH = Path(
    "outputs/20260829_v2/runs/"
    "lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/"
    "scaled_chen_horizon_step_5000.pt"
)
S3_HISTORICAL_DIR = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_controller_to_ols_writability_audit_v1_attempt_03"
)
S3_POST_DIR = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_controller_semantic_geometry_s3_post_step_492_v1_attempt_01"
)
CODE8_DIR = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_code8_coverage_review_v1_attempt_01"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_ols_producer_data_authority_v1_attempt_01"
)


@dataclass(frozen=True)
class SourceBundle:
    source_indices: torch.Tensor
    source_labels: torch.Tensor
    r3_complex: torch.Tensor
    selection_indices: torch.Tensor
    confirmation_indices: torch.Tensor
    source_manifest: Mapping[str, Any]
    checkpoint_path: Path
    calibration_path: Path
    calibration_manifest_path: Path
    calibration_scale: torch.Tensor


@dataclass(frozen=True)
class Partition:
    fit_positions: torch.Tensor
    validation_positions: torch.Tensor
    fit_indices: torch.Tensor
    validation_indices: torch.Tensor
    fit_labels: torch.Tensor
    validation_labels: torch.Tensor
    fit_r3: torch.Tensor
    validation_r3: torch.Tensor
    local_material_sha256: str


@dataclass(frozen=True)
class S3PanelEvidence:
    manifest_path: Path
    manifest_sha256: str
    state_path: Path
    state_sha256: str
    normalized_row_hashes: frozenset[str]
    panel_identity: Mapping[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _relative(repo_root: Path, value: str | Path) -> str:
    path = _resolve(repo_root, value)
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
    finite: bool = True,
) -> torch.Tensor:
    _require(torch.is_tensor(value), f"{name} is not a tensor")
    tensor = value.detach().cpu().contiguous()
    if shape is not None:
        _require(tuple(tensor.shape) == shape, f"{name} shape is {tuple(tensor.shape)}")
    if dtype is not None:
        _require(tensor.dtype == dtype, f"{name} dtype is {tensor.dtype}")
    if finite:
        _require(bool(torch.isfinite(tensor).all()), f"{name} is not finite")
    return tensor


def _read_json_exact(repo_root: Path, path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    target = _resolve(repo_root, path)
    _require(target.is_file(), f"{name} is missing: {target}")
    actual = file_sha256(target)
    _require(actual == expected_sha256, f"{name} SHA mismatch: {actual}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{name} is not a mapping")
    return payload


def _verify_manifest_tree(
    repo_root: Path,
    directory: Path,
    manifest_name: str,
    manifest: Mapping[str, Any],
    name: str,
) -> None:
    root = _resolve(repo_root, directory)
    artifacts = _mapping(manifest.get("artifacts"), f"{name}.artifacts")
    expected = set(artifacts) | {manifest_name}
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    }
    _require(actual == expected, f"{name} artifact tree is not exact")
    for artifact_name, raw_reference in artifacts.items():
        reference = _mapping(raw_reference, f"{name}.{artifact_name}")
        relative = Path(str(artifact_name))
        _require(not relative.is_absolute() and ".." not in relative.parts, f"{name} path escapes run")
        target = root / relative
        _require(reference.get("path") == str(artifact_name), f"{name} path binding changed")
        _require(target.is_file(), f"{name} artifact is missing: {target}")
        _require(
            file_sha256(target) == reference.get("sha256"),
            f"{name} artifact SHA changed: {artifact_name}",
        )
        _require(
            target.stat().st_size == int(reference.get("size_bytes", -1)),
            f"{name} artifact size changed: {artifact_name}",
        )


def _git_state(repo_root: Path) -> tuple[str, str, list[str]]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    branch = run("branch", "--show-current")
    head = run("rev-parse", "HEAD")
    status = [line for line in run("status", "--porcelain").splitlines() if line]
    return branch, head, status


def _load_calibration(repo_root: Path) -> tuple[Path, Path, torch.Tensor]:
    calibration_manifest_path = _resolve(repo_root, CALIBRATION_DIR / "calibration_manifest.json")
    calibration_manifest = _read_json_exact(
        repo_root,
        calibration_manifest_path,
        CALIBRATION_MANIFEST_SHA256,
        "calibration manifest",
    )
    _require(
        calibration_manifest.get("schema")
        == "lightgen_scaled_chen_p1_calibration_contract_repair_manifest.v1",
        "calibration manifest schema changed",
    )
    payload_path = _resolve(repo_root, CALIBRATION_DIR / "repaired_calibration.pt")
    _require(
        file_sha256(payload_path) == CALIBRATION_PAYLOAD_SHA256,
        "calibration payload SHA changed",
    )
    payload = safe_torch_load(payload_path, expected_sha256=CALIBRATION_PAYLOAD_SHA256)
    payload_map = _mapping(payload, "calibration payload")
    scale = _require_tensor(
        payload_map.get("port_scale"),
        name="calibration.port_scale",
        shape=(PORT_COUNT,),
        dtype=torch.float32,
    )
    raw_mean = _require_tensor(
        payload_map.get("raw_port_mean"),
        name="calibration.raw_port_mean",
        shape=(PORT_COUNT,),
        dtype=torch.float32,
    )
    _require(torch.equal(scale, raw_mean), "calibration scale is not raw_port_mean")
    _require(bool((scale > 0).all()), "calibration scale is not positive")
    _require(
        calibration_manifest.get("mode_bank_sha256") == MODE_BANK_SHA256
        or _mapping(calibration_manifest.get("lineage"), "calibration.lineage").get(
            "mode_bank_sha256"
        )
        == MODE_BANK_SHA256,
        "calibration mode-bank binding changed",
    )
    return payload_path, calibration_manifest_path, scale


def _load_source(repo_root: Path) -> SourceBundle:
    source_manifest_path = _resolve(repo_root, S1_DIR / "semantic_manifest.json")
    source_manifest = _read_json_exact(
        repo_root,
        source_manifest_path,
        SOURCE_SEMANTIC_MANIFEST_SHA256,
        "S1 semantic manifest",
    )
    _require(source_manifest.get("schema") == S1_MANIFEST_SCHEMA, "S1 manifest schema changed")
    _verify_manifest_tree(
        repo_root,
        S1_DIR,
        "semantic_manifest.json",
        source_manifest,
        "S1 authority",
    )
    data_identity = _mapping(source_manifest.get("data_identity"), "S1 data_identity")
    source = _mapping(source_manifest.get("source"), "S1 source")
    _require(source_manifest.get("mode_bank_sha256") == MODE_BANK_SHA256, "S1 mode-bank identity changed")
    _require(source.get("mode_bank_sha256") == MODE_BANK_SHA256, "S1 source mode-bank identity changed")
    _require(source.get("dtype") == "torch.complex64", "S1 source dtype changed")
    _require(source.get("checkpoint_sha256") == CHECKPOINT_SHA256, "S1 checkpoint binding changed")
    _require(source.get("semantic_field") == "output.coefficients", "S1 source field changed")
    _require(source.get("stage") == "encoder_final_field_to_fixed_ols_modal_projection", "S1 source stage changed")

    source_artifact_path = _resolve(repo_root, S1_DIR / "ols_embeddings.pt")
    _require(
        file_sha256(source_artifact_path) == SOURCE_ARTIFACT_SHA256,
        "S1 source artifact SHA changed",
    )
    payload = safe_torch_load(source_artifact_path, expected_sha256=SOURCE_ARTIFACT_SHA256)
    payload_map = _mapping(payload, "S1 embeddings")
    _require(payload_map.get("schema") == S1_EMBEDDINGS_SCHEMA, "S1 embeddings schema changed")
    indices_map = _mapping(payload_map.get("indices"), "S1 indices")
    labels_map = _mapping(payload_map.get("labels"), "S1 labels")
    representations = _mapping(payload_map.get("representations"), "S1 representations")

    expected_split_hashes = {
        "fit": (SOURCE_INDEX_SHA256, SOURCE_LABEL_SHA256, 1000),
        "selection": (S1_SELECTION_INDEX_SHA256, None, 1000),
        "confirmation": (S1_CONFIRMATION_INDEX_SHA256, None, 5000),
    }
    split_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split, (expected_index_sha, expected_label_sha, expected_count) in expected_split_hashes.items():
        indices = _require_tensor(
            indices_map.get(split),
            name=f"S1 indices.{split}",
            shape=(expected_count,),
            dtype=torch.int64,
        )
        labels = _require_tensor(
            labels_map.get(split),
            name=f"S1 labels.{split}",
            shape=(expected_count,),
            dtype=torch.int64,
        )
        _require(tensor_sha256(indices) == expected_index_sha, f"S1 {split} index identity changed")
        identity = _mapping(data_identity.get(split), f"S1 data_identity.{split}")
        _require(identity.get("count") == expected_count, f"S1 {split} count changed")
        _require(identity.get("indices_sha256") == expected_index_sha, f"S1 {split} manifest index changed")
        manifest_label_sha = identity.get("labels_sha256")
        _require(isinstance(manifest_label_sha, str), f"S1 {split} label binding is missing")
        _require(tensor_sha256(labels) == manifest_label_sha, f"S1 {split} label identity changed")
        if expected_label_sha is not None:
            _require(manifest_label_sha == expected_label_sha, f"S1 {split} manifest label changed")
        split_tensors[split] = (indices, labels)

    r3 = _require_tensor(
        _mapping(representations.get("fit"), "S1 fit representations").get("R3_COMPLEX"),
        name="S1 fit R3_COMPLEX",
        shape=(1000, PORT_COUNT),
        dtype=torch.complex64,
    )
    _require(tensor_sha256(r3) == SOURCE_R3_SHA256, "S1 fit R3 identity changed")
    checkpoint_path = _resolve(repo_root, CHECKPOINT_PATH)
    _require(file_sha256(checkpoint_path) == CHECKPOINT_SHA256, "canonical checkpoint SHA changed")
    calibration_path, calibration_manifest_path, calibration_scale = _load_calibration(repo_root)
    return SourceBundle(
        source_indices=split_tensors["fit"][0],
        source_labels=split_tensors["fit"][1],
        r3_complex=r3,
        selection_indices=split_tensors["selection"][0],
        confirmation_indices=split_tensors["confirmation"][0],
        source_manifest=source_manifest,
        checkpoint_path=checkpoint_path,
        calibration_path=calibration_path,
        calibration_manifest_path=calibration_manifest_path,
        calibration_scale=calibration_scale,
    )


def build_fixed_partition(
    source_indices: torch.Tensor,
    source_labels: torch.Tensor,
    r3_complex: torch.Tensor,
) -> Partition:
    """Build the registered label-major 50/50 partition without RNG."""
    indices = _require_tensor(source_indices, name="source_indices", shape=(1000,), dtype=torch.int64)
    labels = _require_tensor(source_labels, name="source_labels", shape=(1000,), dtype=torch.int64)
    r3 = _require_tensor(
        r3_complex,
        name="source R3_COMPLEX",
        shape=(1000, PORT_COUNT),
        dtype=torch.complex64,
    )
    fit_positions_list: list[int] = []
    validation_positions_list: list[int] = []
    for label in range(CLASS_COUNT):
        class_positions = [
            int(position)
            for position in torch.nonzero(labels == label, as_tuple=False).flatten().tolist()
        ]
        _require(len(class_positions) == SOURCE_ROWS_PER_CLASS, f"class {label} source count changed")
        class_positions.sort(key=lambda position: int(indices[position]))
        fit_positions_list.extend(class_positions[:PARTITION_ROWS_PER_CLASS])
        validation_positions_list.extend(class_positions[PARTITION_ROWS_PER_CLASS:])

    fit_positions = torch.tensor(fit_positions_list, dtype=torch.int64)
    validation_positions = torch.tensor(validation_positions_list, dtype=torch.int64)
    _require(
        set(fit_positions.tolist()).isdisjoint(validation_positions.tolist()),
        "fit/validation row positions overlap",
    )
    _require(
        set(fit_positions.tolist()) | set(validation_positions.tolist()) == set(range(1000)),
        "fit/validation row positions do not cover source",
    )
    fit_indices = indices.index_select(0, fit_positions).contiguous()
    validation_indices = indices.index_select(0, validation_positions).contiguous()
    fit_labels = labels.index_select(0, fit_positions).contiguous()
    validation_labels = labels.index_select(0, validation_positions).contiguous()
    fit_r3 = r3.index_select(0, fit_positions).contiguous()
    validation_r3 = r3.index_select(0, validation_positions).contiguous()
    for selected_labels, name in ((fit_labels, "fit"), (validation_labels, "validation")):
        counts = torch.bincount(selected_labels, minlength=CLASS_COUNT)
        _require(torch.equal(counts, torch.full((CLASS_COUNT,), PARTITION_ROWS_PER_CLASS)), f"{name} class balance changed")

    component_material = {
        "schema": "ols_producer_partition_identity_components.v1",
        "algorithm": "label_major; within_class_source_index_ascending; first_50_fit_last_50_validation",
        "class_order": list(range(CLASS_COUNT)),
        "rng": "none",
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "source_index_sha256": SOURCE_INDEX_SHA256,
        "source_label_sha256": SOURCE_LABEL_SHA256,
        "fit_row_position_sha256": tensor_sha256(fit_positions),
        "validation_row_position_sha256": tensor_sha256(validation_positions),
        "fit_source_index_sha256": tensor_sha256(fit_indices),
        "validation_source_index_sha256": tensor_sha256(validation_indices),
        "fit_label_sha256": tensor_sha256(fit_labels),
        "validation_label_sha256": tensor_sha256(validation_labels),
        "fit_r3_sha256": tensor_sha256(fit_r3),
        "validation_r3_sha256": tensor_sha256(validation_r3),
        "fit_count_per_class": PARTITION_ROWS_PER_CLASS,
        "validation_count_per_class": PARTITION_ROWS_PER_CLASS,
    }
    return Partition(
        fit_positions=fit_positions,
        validation_positions=validation_positions,
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        fit_labels=fit_labels,
        validation_labels=validation_labels,
        fit_r3=fit_r3,
        validation_r3=validation_r3,
        local_material_sha256=canonical_sha256(component_material),
    )


def _validate_partition(partition: Partition) -> None:
    expected = {
        "fit_positions": FIT_ROW_POSITION_SHA256,
        "validation_positions": VALIDATION_ROW_POSITION_SHA256,
        "fit_indices": FIT_SOURCE_INDEX_SHA256,
        "validation_indices": VALIDATION_SOURCE_INDEX_SHA256,
        "fit_labels": PARTITION_LABEL_SHA256,
        "validation_labels": PARTITION_LABEL_SHA256,
        "fit_r3": FIT_R3_SHA256,
        "validation_r3": VALIDATION_R3_SHA256,
    }
    for name, expected_sha in expected.items():
        _require(tensor_sha256(getattr(partition, name)) == expected_sha, f"{name} hash is not registered")
    _require(partition.fit_positions.numel() == PARTITION_ROW_COUNT, "fit count changed")
    _require(partition.validation_positions.numel() == PARTITION_ROW_COUNT, "validation count changed")
    _require(
        set(partition.fit_positions.tolist()).isdisjoint(partition.validation_positions.tolist()),
        "fit/validation row positions overlap",
    )
    _require(
        set(partition.fit_positions.tolist()) | set(partition.validation_positions.tolist())
        == set(range(1000)),
        "fit/validation row positions do not cover source",
    )


def _load_s2_validation_indices(repo_root: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    identity_path = _resolve(repo_root, S2_DIR / "data_identity_manifest.json")
    identity = _read_json_exact(
        repo_root,
        identity_path,
        S2_DATA_IDENTITY_FILE_SHA256,
        "S2 data identity manifest",
    )
    _require(identity.get("schema") == "lightgen_scaled_chen_frozen_output_data_identity.v1", "S2 data identity schema changed")
    _require(identity.get("output_domain_corpus_sha256") == S2_CORPUS_SHA256, "S2 corpus binding changed")
    validation_identity = _mapping(_mapping(identity.get("splits"), "S2 splits").get("validation"), "S2 validation")
    _require(validation_identity.get("dataset") == "mnist_train", "S2 validation dataset changed")
    _require(validation_identity.get("count") == 5000, "S2 validation count changed")
    _require(validation_identity.get("indices_sha256") == S2_VALIDATION_INDEX_SHA256, "S2 validation index binding changed")
    _require(validation_identity.get("labels_sha256") == S2_VALIDATION_LABEL_SHA256, "S2 validation label binding changed")
    corpus_path = _resolve(repo_root, S2_DIR / "output_domain_corpus.pt")
    corpus = safe_torch_load(corpus_path, expected_sha256=S2_CORPUS_SHA256)
    corpus_map = _mapping(corpus, "S2 output domain corpus")
    _require(corpus_map.get("schema") == S2_CORPUS_SCHEMA, "S2 corpus schema changed")
    indices = _require_tensor(
        corpus_map.get("validation_indices"),
        name="S2 validation indices",
        shape=(5000,),
        dtype=torch.int64,
    )
    labels = _require_tensor(
        corpus_map.get("validation_labels"),
        name="S2 validation labels",
        shape=(5000,),
        dtype=torch.int64,
    )
    _require(tensor_sha256(indices) == S2_VALIDATION_INDEX_SHA256, "S2 validation indices changed")
    _require(tensor_sha256(labels) == S2_VALIDATION_LABEL_SHA256, "S2 validation labels changed")
    return indices, identity


def _load_s3_panel(repo_root: Path, directory: Path, manifest_sha: str, state_sha: str) -> S3PanelEvidence:
    manifest_path = _resolve(repo_root, directory / "s3_manifest.json")
    manifest = _read_json_exact(repo_root, manifest_path, manifest_sha, "S3 manifest")
    _require(manifest.get("schema") == S3_MANIFEST_SCHEMA, "S3 manifest schema changed")
    _verify_manifest_tree(repo_root, directory, "s3_manifest.json", manifest, "S3 authority")
    panel_path = _resolve(repo_root, directory / "controller_panel_identity.json")
    _require(file_sha256(panel_path) == S3_PANEL_FILE_SHA256, "S3 panel file SHA changed")
    panel = _read_json_exact(repo_root, panel_path, S3_PANEL_FILE_SHA256, "S3 panel identity")
    _require(panel.get("schema") == S3_PANEL_SCHEMA, "S3 panel schema changed")
    _require(panel.get("panel_identity_sha256") == S3_PANEL_IDENTITY_SHA256, "S3 panel identity changed")
    _require(panel.get("base_code_count") == 100, "S3 base code count changed")
    _require(panel.get("class_count") == CLASS_COUNT, "S3 class count changed")
    _require(panel.get("code_dim") == 8, "S3 code dimension changed")
    _require(panel.get("distribution") == "uniform[-1,1]^8", "S3 code distribution changed")
    _require(_mapping(panel.get("base_codes"), "S3 base_codes").get("sha256") == S3_BASE_CODE_SHA256, "S3 base code identity changed")
    _require(_mapping(panel.get("codes"), "S3 codes").get("sha256") == S3_PANEL_CODE_SHA256, "S3 panel code identity changed")
    _require(_mapping(panel.get("labels"), "S3 labels").get("sha256") == S3_PANEL_LABEL_SHA256, "S3 panel label identity changed")
    _require(_mapping(panel.get("split"), "S3 split").get("sha256") == S3_PANEL_SPLIT_SHA256, "S3 panel split identity changed")
    state_path = _resolve(repo_root, directory / "controller_ols_states.pt")
    states = safe_torch_load(state_path, expected_sha256=state_sha)
    state = _mapping(states, "S3 state artifact")
    _require(state.get("schema") == S3_STATE_SCHEMA, "S3 state schema changed")
    _require(state.get("panel_identity_sha256") == S3_PANEL_IDENTITY_SHA256, "S3 state panel identity changed")
    codes = _require_tensor(state.get("codes"), name="S3 codes", shape=(1000, 8), dtype=torch.float32)
    labels = _require_tensor(state.get("labels"), name="S3 labels", shape=(1000,), dtype=torch.int64)
    split = _require_tensor(state.get("split"), name="S3 split", shape=(1000,), dtype=torch.int64)
    coefficients = _require_tensor(
        state.get("coefficients"),
        name="S3 coefficients",
        shape=(1000, PORT_COUNT),
        dtype=torch.complex64,
    )
    _require(tensor_sha256(codes) == S3_PANEL_CODE_SHA256, "S3 state code tensor changed")
    _require(tensor_sha256(labels) == S3_PANEL_LABEL_SHA256, "S3 state label tensor changed")
    _require(tensor_sha256(split) == S3_PANEL_SPLIT_SHA256, "S3 state split tensor changed")
    internal_validation = manifest.get("internal_validation")
    if internal_validation is not None:
        internal = _mapping(internal_validation, "S3 internal validation")
        _require(internal.get("schema") == "controller_semantic_geometry_internal_validation.v1", "internal validation schema changed")
        _require(internal.get("descriptive_only") is True, "internal validation is not descriptive-only")
        if "complex_states_saved" in internal:
            _require(internal.get("complex_states_saved") is False, "internal validation complex-state status changed")
        _require(internal.get("base_code_count") == 100, "internal validation base-code count changed")
        _require(internal.get("base_codes_sha256") == INTERNAL_CODE_BASE_SHA256, "internal validation base-code identity changed")
        _require(internal.get("panel_code_sha256") == INTERNAL_CODE_PANEL_SHA256, "internal validation panel-code identity changed")
        _require(internal.get("internal_validation_labels_sha256") == INTERNAL_CODE_LABEL_SHA256, "internal validation label identity changed")
    scale_path, _, scale = _load_calibration(repo_root)
    del scale_path
    normalized = (coefficients / torch.sqrt(scale)).contiguous()
    row_hashes = frozenset(tensor_sha256(row) for row in normalized)
    return S3PanelEvidence(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        state_path=state_path,
        state_sha256=state_sha,
        normalized_row_hashes=row_hashes,
        panel_identity=panel,
    )


def _load_code8_manifest(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    path = _resolve(repo_root, CODE8_DIR / "code8_coverage_review_manifest.json")
    manifest = _read_json_exact(repo_root, path, CODE8_REVIEW_MANIFEST_SHA256, "Code8 review manifest")
    _require(manifest.get("schema") == CODE8_MANIFEST_SCHEMA, "Code8 review schema changed")
    _verify_manifest_tree(repo_root, CODE8_DIR, "code8_coverage_review_manifest.json", manifest, "Code8 review")
    binding = _mapping(manifest.get("authority_binding"), "Code8 authority binding")
    s3 = _mapping(binding.get("s3"), "Code8 S3 binding")
    _require(s3.get("manifest_sha256") == S3_HISTORICAL_MANIFEST_SHA256, "Code8 S3 binding changed")
    _require(s3.get("panel_identity_sha256") == S3_PANEL_IDENTITY_SHA256, "Code8 panel binding changed")
    return path, manifest


def _dataset_index_set(dataset: str, indices: torch.Tensor) -> set[tuple[str, int]]:
    return {(dataset, int(index)) for index in indices.tolist()}


def _overlap_count(left: set[tuple[str, int]], right: set[tuple[str, int]]) -> int:
    return len(left & right)


def validate_firewall_entries(entries: list[Mapping[str, Any]]) -> None:
    """Reject every independently required entry that is not a zero-overlap pass."""
    independent = [entry for entry in entries if entry.get("independently_required")]
    _require(
        all(
            entry.get("status") == "PASS_ZERO_OVERLAP"
            and entry.get("fit_overlap_count") == 0
            and entry.get("validation_overlap_count") == 0
            for entry in independent
        ),
        "BLOCKED_OLS_PRODUCER_DATA_FIREWALL",
    )


def _make_index_entry(
    *,
    canonical_name: str,
    role: str,
    dataset: str,
    identity: Mapping[str, Any],
    fit_overlap: int,
    validation_overlap: int,
    independently_required: bool,
    status: str,
    comparison_method: str,
    subsumed_by: str | None = None,
) -> dict[str, Any]:
    _require(status in {"PASS_ZERO_OVERLAP", "SUBSUMED", "NOT_APPLICABLE"}, "invalid firewall status")
    return {
        "canonical_name": canonical_name,
        "role": role,
        "domain": "SOURCE_IMAGE_INDEX_SPACE",
        "artifact_source_identity": dict(identity),
        "comparison_method": comparison_method,
        "dataset_coordinate": dataset,
        "fit_overlap_count": fit_overlap,
        "validation_overlap_count": validation_overlap,
        "independently_required": independently_required,
        "subsumed_by": subsumed_by,
        "status": status,
    }


def _build_firewall(
    repo_root: Path,
    source: SourceBundle,
    partition: Partition,
    s2_validation_indices: torch.Tensor,
    s2_identity: Mapping[str, Any],
    s3_historical: S3PanelEvidence,
    s3_post: S3PanelEvidence,
    code8_path: Path,
    code8_manifest: Mapping[str, Any],
    fit_bank_sha: str,
    validation_bank_sha: str,
) -> dict[str, Any]:
    fit_train = _dataset_index_set("mnist_train", partition.fit_indices)
    validation_train = _dataset_index_set("mnist_train", partition.validation_indices)
    selection = _dataset_index_set("mnist_test", source.selection_indices)
    confirmation = _dataset_index_set("mnist_train", source.confirmation_indices)
    s2_validation = _dataset_index_set("mnist_train", s2_validation_indices)

    entries = [
        _make_index_entry(
            canonical_name="S1_SELECTION",
            role="protected scientific selection",
            dataset="mnist_test",
            identity={
                "source": "mnist_test_indices_5000_9999",
                "index_sha256": S1_SELECTION_INDEX_SHA256,
                "count": 1000,
            },
            fit_overlap=_overlap_count(fit_train, selection),
            validation_overlap=_overlap_count(validation_train, selection),
            independently_required=True,
            status="PASS_ZERO_OVERLAP",
            comparison_method="exact dataset-qualified source-index set overlap",
        ),
        _make_index_entry(
            canonical_name="S1_CONFIRMATION",
            role="protected scientific confirmation",
            dataset="mnist_train",
            identity={
                "source": "horizon_heldout_train_indices_55000_59999",
                "index_sha256": S1_CONFIRMATION_INDEX_SHA256,
                "count": 5000,
            },
            fit_overlap=_overlap_count(fit_train, confirmation),
            validation_overlap=_overlap_count(validation_train, confirmation),
            independently_required=True,
            status="PASS_ZERO_OVERLAP",
            comparison_method="exact dataset-qualified source-index set overlap",
        ),
        _make_index_entry(
            canonical_name="S2_SELECTION_REFERENCE",
            role="S2 selection/reference alias",
            dataset="mnist_test",
            identity={
                "alias_of": "S1_SELECTION",
                "index_sha256": S1_SELECTION_INDEX_SHA256,
            },
            fit_overlap=0,
            validation_overlap=0,
            independently_required=False,
            status="SUBSUMED",
            comparison_method="alias identity; no independent membership",
            subsumed_by="S1_SELECTION",
        ),
        _make_index_entry(
            canonical_name="S2_CONFIRMATION_REFERENCE",
            role="S2 confirmation/reference alias",
            dataset="mnist_train",
            identity={
                "alias_of": "S1_CONFIRMATION",
                "index_sha256": S1_CONFIRMATION_INDEX_SHA256,
            },
            fit_overlap=0,
            validation_overlap=0,
            independently_required=False,
            status="SUBSUMED",
            comparison_method="alias identity; no independent membership",
            subsumed_by="S1_CONFIRMATION",
        ),
        _make_index_entry(
            canonical_name="S2_VALIDATION",
            role="frozen evaluator validation",
            dataset="mnist_train",
            identity={
                "source": "frozen_output_semantic_evaluator.validation",
                "index_sha256": S2_VALIDATION_INDEX_SHA256,
                "label_sha256": S2_VALIDATION_LABEL_SHA256,
                "count": 5000,
                "identity_manifest_hash": s2_identity.get("identity_manifest_hash"),
                "data_identity_manifest_sha256": file_sha256(
                    _resolve(repo_root, S2_DIR / "data_identity_manifest.json")
                ),
            },
            fit_overlap=_overlap_count(fit_train, s2_validation),
            validation_overlap=_overlap_count(validation_train, s2_validation),
            independently_required=True,
            status="PASS_ZERO_OVERLAP",
            comparison_method="exact dataset-qualified source-index set overlap",
        ),
    ]

    fit_rows = {tensor_sha256(row) for row in partition.fit_r3}
    validation_rows = {tensor_sha256(row) for row in partition.validation_r3}
    s3_rows = s3_historical.normalized_row_hashes | s3_post.normalized_row_hashes
    fit_s3_overlap = len(fit_rows & s3_rows)
    validation_s3_overlap = len(validation_rows & s3_rows)
    s3_identity = {
        "formal_panel_identity_sha256": S3_PANEL_IDENTITY_SHA256,
        "base_code_sha256": S3_BASE_CODE_SHA256,
        "panel_code_sha256": S3_PANEL_CODE_SHA256,
        "panel_label_sha256": S3_PANEL_LABEL_SHA256,
        "panel_split_sha256": S3_PANEL_SPLIT_SHA256,
        "historical_manifest_sha256": s3_historical.manifest_sha256,
        "historical_state_sha256": s3_historical.state_sha256,
        "post_training_manifest_sha256": s3_post.manifest_sha256,
        "post_training_state_sha256": s3_post.state_sha256,
    }
    entries.append(
        {
            "canonical_name": "FORMAL_S3_PROTECTED_PANEL",
            "role": "external scientific evaluation panel",
            "domain": "CONTROLLER_CODE_SPACE_AND_SAVED_COMPLEX_OLS_STATE_SPACE",
            "artifact_source_identity": s3_identity,
            "comparison_method": "exact panel identity plus exact normalized complex row hash",
            "fit_overlap_count": fit_s3_overlap,
            "validation_overlap_count": validation_s3_overlap,
            "independently_required": True,
            "subsumed_by": None,
            "status": "PASS_ZERO_OVERLAP",
        }
    )
    entries.extend(
        [
            {
                "canonical_name": "HISTORICAL_S3_MANIFEST",
                "role": "historical S3 provenance",
                "domain": "CONTROLLER_CODE_SPACE_AND_SAVED_COMPLEX_OLS_STATE_SPACE",
                "artifact_source_identity": {
                    "manifest_sha256": s3_historical.manifest_sha256,
                    "state_sha256": s3_historical.state_sha256,
                    "panel_identity_sha256": S3_PANEL_IDENTITY_SHA256,
                },
                "comparison_method": "provenance subsumed by FORMAL_S3_PROTECTED_PANEL",
                "fit_overlap_count": 0,
                "validation_overlap_count": 0,
                "independently_required": False,
                "subsumed_by": "FORMAL_S3_PROTECTED_PANEL",
                "status": "SUBSUMED",
            },
            {
                "canonical_name": "POST_S3_MANIFEST",
                "role": "post-training S3 provenance",
                "domain": "CONTROLLER_CODE_SPACE_AND_SAVED_COMPLEX_OLS_STATE_SPACE",
                "artifact_source_identity": {
                    "manifest_sha256": s3_post.manifest_sha256,
                    "state_sha256": s3_post.state_sha256,
                    "panel_identity_sha256": S3_PANEL_IDENTITY_SHA256,
                },
                "comparison_method": "provenance subsumed by FORMAL_S3_PROTECTED_PANEL",
                "fit_overlap_count": 0,
                "validation_overlap_count": 0,
                "independently_required": False,
                "subsumed_by": "FORMAL_S3_PROTECTED_PANEL",
                "status": "SUBSUMED",
            },
            {
                "canonical_name": "CODE8_REVIEW",
                "role": "Code8 coverage review provenance",
                "domain": "CONTROLLER_CODE_SPACE",
                "artifact_source_identity": {
                    "manifest_path": _relative(repo_root, code8_path),
                    "manifest_sha256": CODE8_REVIEW_MANIFEST_SHA256,
                    "formal_panel_identity_sha256": S3_PANEL_IDENTITY_SHA256,
                    "review_status": code8_manifest.get("status"),
                },
                "comparison_method": "same formal S3 panel identity; provenance duplicate",
                "fit_overlap_count": 0,
                "validation_overlap_count": 0,
                "independently_required": False,
                "subsumed_by": "FORMAL_S3_PROTECTED_PANEL",
                "status": "SUBSUMED",
            },
            {
                "canonical_name": "CONTROLLER_REPAIR_INTERNAL_CODE_PANEL",
                "role": "controller repair descriptive internal code panel",
                "domain": "CONTROLLER_CODE_SPACE",
                "artifact_source_identity": {
                    "base_code_sha256": INTERNAL_CODE_BASE_SHA256,
                    "panel_code_sha256": INTERNAL_CODE_PANEL_SHA256,
                    "label_sha256": INTERNAL_CODE_LABEL_SHA256,
                    "split_sha256": INTERNAL_CODE_SPLIT_SHA256,
                    "code_seed": 100000,
                    "complex_states_saved": False,
                },
                "comparison_method": "not a source-index or saved-complex producer data domain",
                "fit_overlap_count": None,
                "validation_overlap_count": None,
                "independently_required": False,
                "subsumed_by": None,
                "status": "NOT_APPLICABLE",
            },
        ]
    )
    validate_firewall_entries(entries)
    return {
        "schema": FIREWALL_MANIFEST_SCHEMA,
        "version": 1,
        "authority": "OLS_PRODUCER_DATA_AUTHORITY",
        "producer_lineage": "OLS_LATENT_PRODUCER_V1",
        "source_coordinate": SOURCE_COORDINATE,
        "fit_bank": {
            "path": "ols_producer_fit_bank.v1.pt",
            "file_sha256": fit_bank_sha,
            "partition_identity_sha256": PARTITION_IDENTITY_SHA256,
        },
        "internal_validation_bank": {
            "path": "ols_producer_internal_validation_bank.v1.pt",
            "file_sha256": validation_bank_sha,
            "partition_identity_sha256": PARTITION_IDENTITY_SHA256,
        },
        "protected_identities": entries,
        "unregistered_external_candidates_excluded": True,
        "firewall_pass": True,
        "firewall_rule": "any independent overlap or identity mismatch is invalid_harness",
        "future_consumer": {
            "must_verify_target_bank_identity": True,
            "runtime_resplit": False,
            "runtime_reselection": False,
            "runtime_membership_mutation": False,
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "source_references": {
            "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
            "source_semantic_manifest_sha256": SOURCE_SEMANTIC_MANIFEST_SHA256,
            "s1_selection_index_sha256": S1_SELECTION_INDEX_SHA256,
            "s1_confirmation_index_sha256": S1_CONFIRMATION_INDEX_SHA256,
            "s2_validation_index_sha256": S2_VALIDATION_INDEX_SHA256,
            "s2_data_identity_manifest_sha256": file_sha256(
                _resolve(repo_root, S2_DIR / "data_identity_manifest.json")
            ),
            "s3_formal_panel_identity_sha256": S3_PANEL_IDENTITY_SHA256,
            "code8_review_manifest_sha256": CODE8_REVIEW_MANIFEST_SHA256,
        },
    }


def _bank_payload(
    *,
    role: str,
    positions: torch.Tensor,
    indices: torch.Tensor,
    labels: torch.Tensor,
    r3: torch.Tensor,
    source: SourceBundle,
) -> dict[str, Any]:
    return {
        "schema": BANK_SCHEMA,
        "version": 1,
        "role": role,
        "producer_lineage": "OLS_LATENT_PRODUCER_V1",
        "source_coordinate": SOURCE_COORDINATE,
        "count": int(r3.shape[0]),
        "N": PORT_COUNT,
        "dtype": str(r3.dtype),
        "shape": list(r3.shape),
        "label_role": ["class_balance", "partition_integrity", "source_provenance"],
        "source": {
            "artifact_path": _relative(_repo_root(), S1_DIR / "ols_embeddings.pt"),
            "artifact_sha256": SOURCE_ARTIFACT_SHA256,
            "semantic_manifest_sha256": SOURCE_SEMANTIC_MANIFEST_SHA256,
            "source_index_sha256": SOURCE_INDEX_SHA256,
            "source_label_sha256": SOURCE_LABEL_SHA256,
            "source_r3_sha256": SOURCE_R3_SHA256,
        },
        "mode_bank_sha256": MODE_BANK_SHA256,
        "checkpoint": {
            "path": _relative(_repo_root(), CHECKPOINT_PATH),
            "sha256": CHECKPOINT_SHA256,
        },
        "calibration": {
            "manifest_path": _relative(_repo_root(), source.calibration_manifest_path),
            "manifest_sha256": CALIBRATION_MANIFEST_SHA256,
            "payload_path": _relative(_repo_root(), source.calibration_path),
            "payload_sha256": CALIBRATION_PAYLOAD_SHA256,
        },
        "partition": {
            "semantic_identity_sha256": PARTITION_IDENTITY_SHA256,
            "algorithm": "label_major; within_class_source_index_ascending; first_50_fit_last_50_validation",
            "rng": "none",
            "class_order": list(range(CLASS_COUNT)),
        },
        "source_row_positions": positions,
        "source_indices": indices,
        "labels": labels,
        "r3_complex": r3,
    }


def _validate_bank_payload(
    payload: Mapping[str, Any],
    *,
    role: str,
    expected_positions_sha: str,
    expected_indices_sha: str,
    expected_r3_sha: str,
) -> None:
    _require(payload.get("schema") == BANK_SCHEMA, f"{role} schema changed")
    _require(payload.get("role") == role, f"{role} role changed")
    _require(payload.get("source_coordinate") == SOURCE_COORDINATE, f"{role} coordinate changed")
    _require(payload.get("count") == PARTITION_ROW_COUNT, f"{role} count changed")
    _require(payload.get("N") == PORT_COUNT, f"{role} N changed")
    _require(payload.get("dtype") == "torch.complex64", f"{role} dtype changed")
    _require(payload.get("shape") == [PARTITION_ROW_COUNT, PORT_COUNT], f"{role} shape changed")
    _require(payload.get("mode_bank_sha256") == MODE_BANK_SHA256, f"{role} mode bank changed")
    _require(_mapping(payload.get("checkpoint"), f"{role}.checkpoint").get("sha256") == CHECKPOINT_SHA256, f"{role} checkpoint changed")
    _require(_mapping(payload.get("calibration"), f"{role}.calibration").get("payload_sha256") == CALIBRATION_PAYLOAD_SHA256, f"{role} calibration changed")
    _require(_mapping(payload.get("partition"), f"{role}.partition").get("semantic_identity_sha256") == PARTITION_IDENTITY_SHA256, f"{role} partition changed")
    positions = _require_tensor(payload.get("source_row_positions"), name=f"{role}.source_row_positions", shape=(PARTITION_ROW_COUNT,), dtype=torch.int64)
    indices = _require_tensor(payload.get("source_indices"), name=f"{role}.source_indices", shape=(PARTITION_ROW_COUNT,), dtype=torch.int64)
    labels = _require_tensor(payload.get("labels"), name=f"{role}.labels", shape=(PARTITION_ROW_COUNT,), dtype=torch.int64)
    r3 = _require_tensor(payload.get("r3_complex"), name=f"{role}.r3_complex", shape=(PARTITION_ROW_COUNT, PORT_COUNT), dtype=torch.complex64)
    _require(tensor_sha256(positions) == expected_positions_sha, f"{role} row positions changed")
    _require(tensor_sha256(indices) == expected_indices_sha, f"{role} source indices changed")
    _require(tensor_sha256(labels) == PARTITION_LABEL_SHA256, f"{role} labels changed")
    _require(tensor_sha256(r3) == expected_r3_sha, f"{role} R3 changed")
    counts = torch.bincount(labels, minlength=CLASS_COUNT)
    _require(torch.equal(counts, torch.full((CLASS_COUNT,), PARTITION_ROWS_PER_CLASS)), f"{role} class balance changed")


def _source_manifest(
    repo_root: Path,
    *,
    git_branch: str,
    git_head: str,
    output_dir: Path,
    partition: Partition,
    fit_bank_sha: str,
    validation_bank_sha: str,
) -> dict[str, Any]:
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "version": 1,
        "role": "N3 OLS producer source",
        "producer_lineage": "OLS_LATENT_PRODUCER_V1",
        "source_coordinate": SOURCE_COORDINATE,
        "source_artifact": {
            "path": _relative(repo_root, S1_DIR / "ols_embeddings.pt"),
            "sha256": SOURCE_ARTIFACT_SHA256,
            "semantic_manifest_sha256": SOURCE_SEMANTIC_MANIFEST_SHA256,
            "dataset": "mnist_train",
            "source": "mnist_train_indices_0_54999",
            "count": 1000,
            "per_class": 100,
            "index_sha256": SOURCE_INDEX_SHA256,
            "label_sha256": SOURCE_LABEL_SHA256,
            "r3_complex_sha256": SOURCE_R3_SHA256,
            "shape": [1000, PORT_COUNT],
            "dtype": "torch.complex64",
        },
        "mode_bank_sha256": MODE_BANK_SHA256,
        "checkpoint": {
            "path": _relative(repo_root, CHECKPOINT_PATH),
            "sha256": CHECKPOINT_SHA256,
        },
        "calibration": {
            "manifest_path": _relative(repo_root, CALIBRATION_DIR / "calibration_manifest.json"),
            "manifest_sha256": CALIBRATION_MANIFEST_SHA256,
            "payload_path": _relative(repo_root, CALIBRATION_DIR / "repaired_calibration.pt"),
            "payload_sha256": CALIBRATION_PAYLOAD_SHA256,
        },
        "lineage": {
            "source_historical_role": "S1 fit reference",
            "source_reuse_authority": "S1_FIT_SOURCE_REUSE_AUTHORIZED_FOR_N3",
            "historical_s1_meaning_unchanged": True,
            "copy_semantics": "exact R3_COMPLEX row copy; no reorder, conjugate, rescale, or re-encode",
        },
        "partition": {
            "semantic_identity_sha256": PARTITION_IDENTITY_SHA256,
            "local_component_material_sha256": partition.local_material_sha256,
            "fit_bank_file_sha256": fit_bank_sha,
            "internal_validation_bank_file_sha256": validation_bank_sha,
        },
        "git": {
            "branch": git_branch,
            "head": git_head,
            "expected_starting_head": EXPECTED_STARTING_HEAD,
            "worktree_clean_at_materialization": True,
        },
        "attempt": {
            "directory": _relative(repo_root, output_dir),
            "write_once": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
    }


def _partition_manifest(
    repo_root: Path,
    *,
    git_branch: str,
    git_head: str,
    output_dir: Path,
    partition: Partition,
    fit_bank_sha: str,
    validation_bank_sha: str,
) -> dict[str, Any]:
    fit_material = {
        "role": "OLS_PRODUCER_FIT_BANK_V1",
        "row_positions": partition.fit_positions.tolist(),
        "source_indices": partition.fit_indices.tolist(),
        "labels": partition.fit_labels.tolist(),
        "r3_sha256": FIT_R3_SHA256,
    }
    validation_material = {
        "role": "OLS_PRODUCER_INTERNAL_VALIDATION_BANK_V1",
        "row_positions": partition.validation_positions.tolist(),
        "source_indices": partition.validation_indices.tolist(),
        "labels": partition.validation_labels.tolist(),
        "r3_sha256": VALIDATION_R3_SHA256,
    }
    return {
        "schema": PARTITION_MANIFEST_SCHEMA,
        "version": 1,
        "producer_lineage": "OLS_LATENT_PRODUCER_V1",
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "source_coordinate": SOURCE_COORDINATE,
        "algorithm": "label_major; within_class_source_index_ascending; first_50_fit_last_50_validation",
        "class_order": list(range(CLASS_COUNT)),
        "fit_count_per_class": PARTITION_ROWS_PER_CLASS,
        "validation_count_per_class": PARTITION_ROWS_PER_CLASS,
        "rng": "none",
        "runtime_resplit": False,
        "runtime_reselection": False,
        "runtime_membership_mutation": False,
        "fit": {
            "count": PARTITION_ROW_COUNT,
            "row_positions": partition.fit_positions.tolist(),
            "row_position_sha256": FIT_ROW_POSITION_SHA256,
            "source_indices": partition.fit_indices.tolist(),
            "source_index_sha256": FIT_SOURCE_INDEX_SHA256,
            "labels": partition.fit_labels.tolist(),
            "label_sha256": PARTITION_LABEL_SHA256,
            "r3_complex_sha256": FIT_R3_SHA256,
            "bank_path": "ols_producer_fit_bank.v1.pt",
            "bank_file_sha256": fit_bank_sha,
            "component_material_sha256": canonical_sha256(fit_material),
        },
        "internal_validation": {
            "count": PARTITION_ROW_COUNT,
            "row_positions": partition.validation_positions.tolist(),
            "row_position_sha256": VALIDATION_ROW_POSITION_SHA256,
            "source_indices": partition.validation_indices.tolist(),
            "source_index_sha256": VALIDATION_SOURCE_INDEX_SHA256,
            "labels": partition.validation_labels.tolist(),
            "label_sha256": PARTITION_LABEL_SHA256,
            "r3_complex_sha256": VALIDATION_R3_SHA256,
            "bank_path": "ols_producer_internal_validation_bank.v1.pt",
            "bank_file_sha256": validation_bank_sha,
            "component_material_sha256": canonical_sha256(validation_material),
        },
        "partition_identity": {
            "semantic_identity_sha256": PARTITION_IDENTITY_SHA256,
            "local_component_material_sha256": partition.local_material_sha256,
            "identity_recipe_status": "frozen_plan_semantic_authority; component bindings are independently verified",
        },
        "overlap": {
            "fit_validation_row_position_count": 0,
            "fit_validation_source_index_count": 0,
            "union_source_row_count": 1000,
        },
        "git": {"branch": git_branch, "head": git_head},
        "attempt_directory": _relative(repo_root, output_dir),
    }


def materialize(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    branch, head, dirty_paths = _git_state(repo_root)
    _require(branch == BRANCH, f"unexpected branch: {branch}")
    _require(not dirty_paths, f"worktree must be clean: {dirty_paths}")
    source = _load_source(repo_root)
    partition = build_fixed_partition(
        source.source_indices,
        source.source_labels,
        source.r3_complex,
    )
    _validate_partition(partition)
    s2_validation_indices, s2_identity = _load_s2_validation_indices(repo_root)
    s3_historical = _load_s3_panel(
        repo_root,
        S3_HISTORICAL_DIR,
        S3_HISTORICAL_MANIFEST_SHA256,
        S3_HISTORICAL_STATE_SHA256,
    )
    s3_post = _load_s3_panel(
        repo_root,
        S3_POST_DIR,
        S3_POST_MANIFEST_SHA256,
        S3_POST_STATE_SHA256,
    )
    code8_path, code8_manifest = _load_code8_manifest(repo_root)
    _require(INTERNAL_CODE_BASE_SHA256 != S3_BASE_CODE_SHA256, "internal code panel unexpectedly equals S3 base codes")
    _require(INTERNAL_CODE_PANEL_SHA256 != S3_PANEL_CODE_SHA256, "internal code panel unexpectedly equals S3 panel codes")
    _require(bool(head), "git HEAD is unavailable")
    output = _resolve(repo_root, output_dir)
    _require(not output.exists(), f"write-once attempt directory already exists: {output}")

    fit_payload = _bank_payload(
        role="OLS_PRODUCER_FIT_BANK_V1",
        positions=partition.fit_positions,
        indices=partition.fit_indices,
        labels=partition.fit_labels,
        r3=partition.fit_r3,
        source=source,
    )
    validation_payload = _bank_payload(
        role="OLS_PRODUCER_INTERNAL_VALIDATION_BANK_V1",
        positions=partition.validation_positions,
        indices=partition.validation_indices,
        labels=partition.validation_labels,
        r3=partition.validation_r3,
        source=source,
    )
    fit_path = output / "ols_producer_fit_bank.v1.pt"
    validation_path = output / "ols_producer_internal_validation_bank.v1.pt"
    atomic_torch_save_new(fit_path, fit_payload, artifact_kind="fit bank")
    atomic_torch_save_new(validation_path, validation_payload, artifact_kind="internal-validation bank")
    fit_bank_sha = file_sha256(fit_path)
    validation_bank_sha = file_sha256(validation_path)

    source_manifest = _source_manifest(
        repo_root,
        git_branch=branch,
        git_head=head,
        output_dir=output,
        partition=partition,
        fit_bank_sha=fit_bank_sha,
        validation_bank_sha=validation_bank_sha,
    )
    partition_manifest = _partition_manifest(
        repo_root,
        git_branch=branch,
        git_head=head,
        output_dir=output,
        partition=partition,
        fit_bank_sha=fit_bank_sha,
        validation_bank_sha=validation_bank_sha,
    )
    firewall_manifest = _build_firewall(
        repo_root,
        source,
        partition,
        s2_validation_indices,
        s2_identity,
        s3_historical,
        s3_post,
        code8_path,
        code8_manifest,
        fit_bank_sha,
        validation_bank_sha,
    )
    atomic_write_new_json(
        output / "ols_producer_source_manifest.v1.json",
        source_manifest,
        artifact_kind="source manifest",
    )
    atomic_write_new_json(
        output / "ols_producer_partition_manifest.v1.json",
        partition_manifest,
        artifact_kind="partition manifest",
    )
    atomic_write_new_json(
        output / "ols_producer_data_firewall_manifest.v1.json",
        firewall_manifest,
        artifact_kind="data firewall manifest",
    )

    fit_loaded = safe_torch_load(fit_path, expected_sha256=fit_bank_sha)
    validation_loaded = safe_torch_load(validation_path, expected_sha256=validation_bank_sha)
    _validate_bank_payload(
        _mapping(fit_loaded, "reloaded fit bank"),
        role="OLS_PRODUCER_FIT_BANK_V1",
        expected_positions_sha=FIT_ROW_POSITION_SHA256,
        expected_indices_sha=FIT_SOURCE_INDEX_SHA256,
        expected_r3_sha=FIT_R3_SHA256,
    )
    _validate_bank_payload(
        _mapping(validation_loaded, "reloaded internal-validation bank"),
        role="OLS_PRODUCER_INTERNAL_VALIDATION_BANK_V1",
        expected_positions_sha=VALIDATION_ROW_POSITION_SHA256,
        expected_indices_sha=VALIDATION_SOURCE_INDEX_SHA256,
        expected_r3_sha=VALIDATION_R3_SHA256,
    )
    for filename, expected_schema in (
        ("ols_producer_source_manifest.v1.json", SOURCE_MANIFEST_SCHEMA),
        ("ols_producer_partition_manifest.v1.json", PARTITION_MANIFEST_SCHEMA),
        ("ols_producer_data_firewall_manifest.v1.json", FIREWALL_MANIFEST_SCHEMA),
    ):
        manifest = json.loads((output / filename).read_text(encoding="utf-8"))
        _require(manifest.get("schema") == expected_schema, f"{filename} schema changed after write")
    _require(firewall_manifest.get("firewall_pass") is True, "firewall did not close")
    return {
        "classification": "OLS_PRODUCER_DATA_AUTHORITY_MATERIALIZED",
        "output_directory": _relative(repo_root, output),
        "source_manifest_sha256": file_sha256(output / "ols_producer_source_manifest.v1.json"),
        "fit_bank_sha256": fit_bank_sha,
        "validation_bank_sha256": validation_bank_sha,
        "partition_manifest_sha256": file_sha256(output / "ols_producer_partition_manifest.v1.json"),
        "firewall_manifest_sha256": file_sha256(output / "ols_producer_data_firewall_manifest.v1.json"),
        "partition_identity_sha256": PARTITION_IDENTITY_SHA256,
        "partition_component_material_sha256": partition.local_material_sha256,
        "fit_tensor_sha256": FIT_R3_SHA256,
        "validation_tensor_sha256": VALIDATION_R3_SHA256,
        "firewall_pass": True,
        "optimizer_created": False,
        "backward_called": False,
        "forward_called": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "next_gate": "PLAN_LOW_RANK_JOINT_AMPLITUDE_PRODUCER_CONTRACT",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="fresh write-once attempt directory",
    )
    args = parser.parse_args()
    result = materialize(_repo_root(), args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
