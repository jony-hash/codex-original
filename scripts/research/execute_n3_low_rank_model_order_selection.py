"""Execute the frozen one-shot N3 low-rank model-order contract.

The consumer reads only the immutable producer fit and internal-validation
banks.  It does not construct an optical model, invoke a generator or
evaluator, sample scores, or train anything.
"""

from __future__ import annotations

import argparse
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
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)


TASK_ID = "EXECUTE_N3_LOW_RANK_MODEL_ORDER_SELECTION_V1"
GATE = "N3_LOW_RANK_MODEL_ORDER_SELECTION_V1"
SCHEMA = "n3_low_rank_model_order_selection_result.v1"
CLASSIFICATION = "N3_LOW_RANK_MODEL_ORDER_SELECTED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"

DATA_DIR = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_ols_producer_data_authority_v1_attempt_01"
)
SOURCE_MANIFEST = DATA_DIR / "ols_producer_source_manifest.v1.json"
PARTITION_MANIFEST = DATA_DIR / "ols_producer_partition_manifest.v1.json"
FIREWALL_MANIFEST = DATA_DIR / "ols_producer_data_firewall_manifest.v1.json"
FIT_BANK = DATA_DIR / "ols_producer_fit_bank.v1.pt"
VALIDATION_BANK = DATA_DIR / "ols_producer_internal_validation_bank.v1.pt"

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
EXPECTED_VALIDATION_BANK_SHA256 = (
    "8397c886db73bb260723debd6d4eab332b6172972a055b28f770e7ca2bf3ee25"
)
EXPECTED_PARTITION_IDENTITY_SHA256 = (
    "d1eaf777c27b98eb607e98ca41d2360115eb0b070156b29d21805c7332981d6d"
)
EXPECTED_SOURCE_ARTIFACT_SHA256 = (
    "9375b08597f03fee3798e05798779556c3e7534db6a9d39033232f7278d32279"
)
EXPECTED_SOURCE_SEMANTIC_MANIFEST_SHA256 = (
    "b0199040926618b17373edc4f167cc8bc4788f39f64de174e1a3fb225781833e"
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
EXPECTED_FIT_R3_SHA256 = (
    "f507dcaad007963178a07fcb13b55c267529b2adecc745df6c465316aa8732c0"
)
EXPECTED_VALIDATION_R3_SHA256 = (
    "0c3887331553c474f5a39ab3653c8b0094ce7db8f416cff279627e001d68102c"
)
EXPECTED_FIT_ROW_POSITION_SHA256 = (
    "54fe39a51036f6b69cf4cb65cfc614e4043a92fb8a601c75f529da8f87122243"
)
EXPECTED_VALIDATION_ROW_POSITION_SHA256 = (
    "964724eddd503d596ec870ceea3b0145f151ade64f5903307352297372763a75"
)
EXPECTED_FIT_SOURCE_INDEX_SHA256 = (
    "99285b32a2c098de1788cd405bc317c3c7877bbd07bd1b63260fd9fd1cc93d18"
)
EXPECTED_VALIDATION_SOURCE_INDEX_SHA256 = (
    "96a367399d53a48078d7a3328a6063174c2f58031177da0f14953191620b5d80"
)
EXPECTED_PARTITION_LABEL_SHA256 = (
    "c8bacf260ae894e7da623586b9e04c67d95f628c16f52766cee0994ca2ceac19"
)

CLASS_COUNT = 10
DIMENSION = 100
FIT_COUNT = 500
VALIDATION_COUNT = 500
ROWS_PER_CLASS = 50
FIRST_RANK = 1
LAST_RANK = 99
ORTHOGONALITY_RTOL = 1.0e-9
ORTHOGONALITY_ATOL = 1.0e-9


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _read_json_exact(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    actual_sha256 = file_sha256(path)
    _require(actual_sha256 == expected_sha256, f"{label} hash changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(_mapping(value, label))


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


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
    dirty_paths = [line[3:] if len(line) >= 4 else line for line in status]
    return {"branch": branch, "head": head, "dirty_paths": dirty_paths}


def _verify_authorities(repo_root: Path) -> dict[str, Any]:
    source_path = repo_root / SOURCE_MANIFEST
    partition_path = repo_root / PARTITION_MANIFEST
    firewall_path = repo_root / FIREWALL_MANIFEST
    source = _read_json_exact(
        source_path, EXPECTED_SOURCE_MANIFEST_SHA256, "producer source manifest"
    )
    partition = _read_json_exact(
        partition_path,
        EXPECTED_PARTITION_MANIFEST_SHA256,
        "producer partition manifest",
    )
    firewall = _read_json_exact(
        firewall_path,
        EXPECTED_FIREWALL_MANIFEST_SHA256,
        "producer firewall manifest",
    )

    _require(source.get("schema") == "ols_producer_source_manifest.v1", "source schema changed")
    _require(source.get("producer_lineage") == "OLS_LATENT_PRODUCER_V1", "producer lineage changed")
    _require(source.get("source_coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX", "source coordinate changed")
    source_artifact = _mapping(source.get("source_artifact"), "source artifact")
    _require(source_artifact.get("sha256") == EXPECTED_SOURCE_ARTIFACT_SHA256, "source artifact changed")
    _require(source_artifact.get("semantic_manifest_sha256") == EXPECTED_SOURCE_SEMANTIC_MANIFEST_SHA256, "S1 semantic manifest changed")
    _require(source_artifact.get("index_sha256") == EXPECTED_SOURCE_INDEX_SHA256, "source index changed")
    _require(source_artifact.get("label_sha256") == EXPECTED_SOURCE_LABEL_SHA256, "source label identity changed")
    _require(source_artifact.get("r3_complex_sha256") == EXPECTED_SOURCE_R3_SHA256, "source R3 identity changed")
    _require(source_artifact.get("dataset") == "mnist_train", "source dataset changed")
    _require(source_artifact.get("count") == 1000 and source_artifact.get("per_class") == 100, "source cardinality changed")
    _require(source_artifact.get("shape") == [1000, DIMENSION], "source shape changed")
    _require(source_artifact.get("dtype") == "torch.complex64", "source dtype changed")

    partition_section = _mapping(source.get("partition"), "source partition")
    _require(partition_section.get("semantic_identity_sha256") == EXPECTED_PARTITION_IDENTITY_SHA256, "partition identity changed")
    _require(partition_section.get("fit_bank_file_sha256") == EXPECTED_FIT_BANK_SHA256, "fit bank binding changed")
    _require(partition_section.get("internal_validation_bank_file_sha256") == EXPECTED_VALIDATION_BANK_SHA256, "validation bank binding changed")
    _require(partition.get("schema") == "ols_producer_partition_manifest.v1", "partition schema changed")
    _require(partition.get("source_artifact_sha256") == EXPECTED_SOURCE_ARTIFACT_SHA256, "partition source binding changed")
    _require(partition.get("source_coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX", "partition coordinate changed")
    _require(partition.get("algorithm") == "label_major; within_class_source_index_ascending; first_50_fit_last_50_validation", "partition algorithm changed")
    _require(partition.get("rng") == "none", "partition RNG changed")
    _require(partition.get("runtime_resplit") is False, "runtime resplit is enabled")
    _require(partition.get("runtime_reselection") is False, "runtime reselection is enabled")
    _require(partition.get("fit_count_per_class") == ROWS_PER_CLASS, "fit class balance changed")
    _require(partition.get("validation_count_per_class") == ROWS_PER_CLASS, "validation class balance changed")
    _require(_mapping(partition.get("partition_identity"), "partition identity").get("semantic_identity_sha256") == EXPECTED_PARTITION_IDENTITY_SHA256, "partition semantic identity changed")

    _require(firewall.get("schema") == "ols_producer_data_firewall_manifest.v1", "firewall schema changed")
    _require(firewall.get("firewall_pass") is True, "producer data firewall did not pass")
    _require(firewall.get("unregistered_external_candidates_excluded") is True, "unregistered candidates were not excluded")
    future_consumer = _mapping(firewall.get("future_consumer"), "firewall future consumer")
    _require(future_consumer.get("must_verify_target_bank_identity") is True, "consumer identity check is not required")
    _require(future_consumer.get("runtime_membership_mutation") is False, "runtime membership mutation is enabled")
    _require(future_consumer.get("runtime_reselection") is False, "runtime reselection is enabled in firewall")
    _require(future_consumer.get("runtime_resplit") is False, "runtime resplit is enabled in firewall")
    protected = {
        item.get("canonical_name"): item
        for item in firewall.get("protected_identities", [])
        if isinstance(item, Mapping)
    }
    for name in ("S1_SELECTION", "S1_CONFIRMATION", "S2_VALIDATION", "FORMAL_S3_PROTECTED_PANEL"):
        entry = _mapping(protected.get(name), f"protected identity {name}")
        _require(entry.get("status") == "PASS_ZERO_OVERLAP", f"protected identity {name} is not isolated")
        _require(entry.get("fit_overlap_count") == 0, f"fit bank overlaps {name}")
        _require(entry.get("validation_overlap_count") == 0, f"validation bank overlaps {name}")

    return {
        "source_manifest": {
            "path": _relative(repo_root, source_path),
            "sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "schema": source["schema"],
        },
        "partition_manifest": {
            "path": _relative(repo_root, partition_path),
            "sha256": EXPECTED_PARTITION_MANIFEST_SHA256,
            "schema": partition["schema"],
            "semantic_identity_sha256": EXPECTED_PARTITION_IDENTITY_SHA256,
        },
        "firewall_manifest": {
            "path": _relative(repo_root, firewall_path),
            "sha256": EXPECTED_FIREWALL_MANIFEST_SHA256,
            "schema": firewall["schema"],
            "firewall_pass": True,
        },
        "coordinate": "R2_AMPLITUDE",
        "coordinate_definition": "abs(c/sqrt(s))=abs(R3_COMPLEX)",
        "producer_lineage": source["producer_lineage"],
        "fit_bank": {
            "path": _relative(repo_root, repo_root / FIT_BANK),
            "sha256": EXPECTED_FIT_BANK_SHA256,
            "count": FIT_COUNT,
            "per_class": ROWS_PER_CLASS,
        },
        "internal_validation_bank": {
            "path": _relative(repo_root, repo_root / VALIDATION_BANK),
            "sha256": EXPECTED_VALIDATION_BANK_SHA256,
            "count": VALIDATION_COUNT,
            "per_class": ROWS_PER_CLASS,
        },
    }


def _validate_bank(
    payload: Any,
    *,
    role: str,
    expected_r3_sha256: str,
    expected_row_position_sha256: str,
    expected_source_index_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bank = _mapping(payload, role)
    _require(bank.get("schema") == "ols_producer_bank.v1", f"{role} schema changed")
    _require(bank.get("role") == role, f"{role} role changed")
    _require(bank.get("producer_lineage") == "OLS_LATENT_PRODUCER_V1", f"{role} lineage changed")
    _require(bank.get("source_coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX", f"{role} coordinate changed")
    _require(bank.get("count") == FIT_COUNT, f"{role} count changed")
    _require(bank.get("N") == DIMENSION, f"{role} dimension changed")
    _require(bank.get("dtype") == "torch.complex64", f"{role} dtype changed")
    _require(bank.get("shape") == [FIT_COUNT, DIMENSION], f"{role} shape changed")
    partition = _mapping(bank.get("partition"), f"{role} partition")
    _require(partition.get("semantic_identity_sha256") == EXPECTED_PARTITION_IDENTITY_SHA256, f"{role} partition changed")

    r3 = bank.get("r3_complex")
    indices = bank.get("source_indices")
    positions = bank.get("source_row_positions")
    labels = bank.get("labels")
    _require(torch.is_tensor(r3), f"{role} R3 tensor is missing")
    _require(torch.is_tensor(indices), f"{role} source indices are missing")
    _require(torch.is_tensor(positions), f"{role} row positions are missing")
    _require(torch.is_tensor(labels), f"{role} labels are missing")
    _require(tuple(r3.shape) == (FIT_COUNT, DIMENSION) and r3.dtype == torch.complex64, f"{role} R3 tensor shape/dtype changed")
    _require(tuple(indices.shape) == (FIT_COUNT,) and indices.dtype == torch.int64, f"{role} indices shape/dtype changed")
    _require(tuple(positions.shape) == (FIT_COUNT,) and positions.dtype == torch.int64, f"{role} positions shape/dtype changed")
    _require(tuple(labels.shape) == (FIT_COUNT,) and labels.dtype == torch.int64, f"{role} labels shape/dtype changed")
    _require(bool(torch.isfinite(r3).all()), f"{role} R3 tensor is non-finite")
    _require(tensor_sha256(r3) == expected_r3_sha256, f"{role} R3 tensor identity changed")
    _require(tensor_sha256(indices) == expected_source_index_sha256, f"{role} source index identity changed")
    _require(tensor_sha256(positions) == expected_row_position_sha256, f"{role} row position identity changed")
    _require(tensor_sha256(labels) == EXPECTED_PARTITION_LABEL_SHA256, f"{role} label identity changed")
    counts = torch.bincount(labels, minlength=CLASS_COUNT)
    _require(torch.equal(counts, torch.full((CLASS_COUNT,), ROWS_PER_CLASS, dtype=torch.int64)), f"{role} class balance changed")
    return r3.detach().cpu().contiguous(), indices.detach().cpu().contiguous(), positions.detach().cpu().contiguous(), labels.detach().cpu().contiguous()


def _select_rank(
    fit_r3: torch.Tensor,
    validation_r3: torch.Tensor,
) -> dict[str, Any]:
    fit_q = fit_r3.abs().to(dtype=torch.float64).contiguous()
    validation_q = validation_r3.abs().to(dtype=torch.float64).contiguous()
    _require(bool(torch.isfinite(fit_q).all()), "fit R2 values are non-finite")
    _require(bool(torch.isfinite(validation_q).all()), "validation R2 values are non-finite")
    _require(tuple(fit_q.shape) == (FIT_COUNT, DIMENSION), "fit R2 shape changed")
    _require(tuple(validation_q.shape) == (VALIDATION_COUNT, DIMENSION), "validation R2 shape changed")

    mu_fit = fit_q.mean(dim=0)
    centered_fit = fit_q - mu_fit
    _require(bool(torch.isfinite(mu_fit).all()), "fit mean is non-finite")
    _require(bool(torch.isfinite(centered_fit).all()), "centered fit matrix is non-finite")

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        left, singular_values, right_transposed = torch.linalg.svd(
            centered_fit, full_matrices=True
        )
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)

    _require(tuple(left.shape) == (FIT_COUNT, FIT_COUNT), "full SVD left basis shape changed")
    _require(tuple(singular_values.shape) == (DIMENSION,), "full SVD spectrum shape changed")
    _require(tuple(right_transposed.shape) == (DIMENSION, DIMENSION), "full SVD right basis shape changed")
    _require(bool(torch.isfinite(left).all()), "SVD left basis is non-finite")
    _require(bool(torch.isfinite(singular_values).all()), "SVD spectrum is non-finite")
    _require(bool(torch.isfinite(right_transposed).all()), "SVD right basis is non-finite")
    _require(bool(singular_values.ge(0.0).all()), "SVD spectrum is negative")

    right_basis = right_transposed.transpose(0, 1).contiguous()
    gram = right_basis.transpose(0, 1).mm(right_basis)
    identity = torch.eye(DIMENSION, dtype=torch.float64)
    orthogonality_passed = bool(
        torch.allclose(
            gram,
            identity,
            rtol=ORTHOGONALITY_RTOL,
            atol=ORTHOGONALITY_ATOL,
        )
    )
    _require(orthogonality_passed, "SVD right basis is not orthonormal")

    validation_centered = validation_q - mu_fit
    losses: dict[int, torch.Tensor] = {}
    errors: list[dict[str, Any]] = []
    for rank in range(FIRST_RANK, LAST_RANK + 1):
        basis = right_basis[:, :rank]
        reconstructed = mu_fit + validation_centered.mm(basis).mm(basis.transpose(0, 1))
        loss = (validation_q - reconstructed).square().sum(dim=1).div(float(DIMENSION))
        _require(bool(torch.isfinite(loss).all()), f"validation loss is non-finite at rank {rank}")
        mean_loss = loss.mean()
        _require(bool(torch.isfinite(mean_loss)), f"validation mean loss is non-finite at rank {rank}")
        _require(bool(mean_loss.ge(0.0)), f"validation mean loss is negative at rank {rank}")
        losses[rank] = loss
        errors.append({"rank": rank, "E_r": float(mean_loss.item())})

    error_tensor = torch.tensor([item["E_r"] for item in errors], dtype=torch.float64)
    _require(bool(torch.isfinite(error_tensor).all()), "candidate validation errors are non-finite")
    minimum = error_tensor.min()
    minimum_ranks = [
        rank
        for rank, value in zip(range(FIRST_RANK, LAST_RANK + 1), error_tensor)
        if bool(value == minimum)
    ]
    _require(minimum_ranks, "minimum validation rank is undefined")
    r_min = min(minimum_ranks)
    E_min = error_tensor[r_min - FIRST_RANK]
    sample_std = losses[r_min].std(unbiased=True)
    SE_min = sample_std.div(torch.sqrt(torch.tensor(float(VALIDATION_COUNT), dtype=torch.float64)))
    _require(bool(torch.isfinite(SE_min)), "SE_min is non-finite")
    _require(bool(SE_min.ge(0.0)), "SE_min is negative")
    acceptable = [
        rank
        for rank, value in zip(range(FIRST_RANK, LAST_RANK + 1), error_tensor)
        if bool(value <= E_min + SE_min)
    ]
    _require(acceptable, "one-standard-error acceptable set is empty")
    r_selected = min(acceptable)
    _require(r_selected in range(FIRST_RANK, LAST_RANK + 1), "selected rank is outside frozen domain")

    return {
        "fit": {
            "count": FIT_COUNT,
            "dimension": DIMENSION,
            "dtype": "torch.float64",
            "mean_definition": "global_fit_mean_only",
            "mu_fit_sha256": tensor_sha256(mu_fit),
            "spectrum": [float(value) for value in singular_values.tolist()],
            "spectrum_definition": "singular_values_of_centered_fit_q",
        },
        "validation": {
            "count": VALIDATION_COUNT,
            "dimension": DIMENSION,
            "loss_definition": "||q_i-q_hat_i(r)||_2^2/d",
            "errors": errors,
        },
        "selection": {
            "candidate_domain": list(range(FIRST_RANK, LAST_RANK + 1)),
            "r_min": r_min,
            "E_min": float(E_min.item()),
            "SE_min": float(SE_min.item()),
            "sample_std_ddof": 1,
            "acceptable_ranks": acceptable,
            "r_selected": r_selected,
            "tie_break": "smallest_rank",
            "rule": "E_r <= E_min + SE_min; select smallest acceptable rank",
        },
        "numerical_checks": {
            "inputs_finite": True,
            "svd_finite": True,
            "right_basis_orthonormal": True,
            "orthogonality_rtol": ORTHOGONALITY_RTOL,
            "orthogonality_atol": ORTHOGONALITY_ATOL,
            "all_candidate_errors_finite": True,
            "SE_min_finite_nonnegative": True,
            "acceptable_set_nonempty": True,
            "exactly_one_selected_rank": True,
        },
        "execution": {
            "device": "cpu",
            "dtype": "torch.float64",
            "full_svd": True,
            "full_svd_calls": 1,
            "rng": "NONE",
            "adaptive_feedback": False,
        },
    }


def _write_result(repo_root: Path, output_dir: Path, payload: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        target = temporary / "model_order_result.v1.json"
        atomic_write_new_json(target, payload, artifact_kind="N3 model-order result")
        if sorted(path.name for path in temporary.iterdir()) != [target.name]:
            raise RuntimeError("model-order temporary artifact layout changed")
        if output_dir.exists():
            raise FileExistsError(f"write-once output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / target.name


def execute(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    state = _git_state(repo_root)
    _require(state["branch"] == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    _require(not state["dirty_paths"], f"worktree must be clean: {state['dirty_paths']}")
    script_path = Path(__file__).resolve()
    authorities = _verify_authorities(repo_root)
    fit_payload = safe_torch_load(repo_root / FIT_BANK, expected_sha256=EXPECTED_FIT_BANK_SHA256)
    validation_payload = safe_torch_load(
        repo_root / VALIDATION_BANK,
        expected_sha256=EXPECTED_VALIDATION_BANK_SHA256,
    )
    fit_r3, fit_indices, fit_positions, fit_labels = _validate_bank(
        fit_payload,
        role="OLS_PRODUCER_FIT_BANK_V1",
        expected_r3_sha256=EXPECTED_FIT_R3_SHA256,
        expected_row_position_sha256=EXPECTED_FIT_ROW_POSITION_SHA256,
        expected_source_index_sha256=EXPECTED_FIT_SOURCE_INDEX_SHA256,
    )
    validation_r3, validation_indices, validation_positions, validation_labels = _validate_bank(
        validation_payload,
        role="OLS_PRODUCER_INTERNAL_VALIDATION_BANK_V1",
        expected_r3_sha256=EXPECTED_VALIDATION_R3_SHA256,
        expected_row_position_sha256=EXPECTED_VALIDATION_ROW_POSITION_SHA256,
        expected_source_index_sha256=EXPECTED_VALIDATION_SOURCE_INDEX_SHA256,
    )
    _require(set(fit_indices.tolist()).isdisjoint(validation_indices.tolist()), "fit/validation source indices overlap")
    _require(set(fit_positions.tolist()).isdisjoint(validation_positions.tolist()), "fit/validation row positions overlap")
    _require(
        set(fit_positions.tolist()) | set(validation_positions.tolist()) == set(range(1000)),
        "fit/validation row positions do not cover the source bank",
    )
    _require(
        len(set(fit_indices.tolist()) | set(validation_indices.tolist()))
        == FIT_COUNT + VALIDATION_COUNT,
        "fit/validation source indices do not cover exactly the authorized 1000-row bank",
    )
    _require(torch.equal(fit_labels.unique(sorted=True), validation_labels.unique(sorted=True)), "fit/validation class domains differ")

    computed = _select_rank(fit_r3, validation_r3)
    result = {
        "schema": SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": CLASSIFICATION,
        "semantic_inputs": {
            "producer_data": "OLS_PRODUCER_DATA_V1",
            "model_order_contract": "DEFINE_N3_LOW_RANK_MODEL_ORDER_CONTRACT_V1_READY",
            "coordinate": "R2_AMPLITUDE",
            "score_baseline": "DIAGONAL_GAUSSIAN_LOW_RANK_SCORE_BASELINE_V1",
            "phase_lineage": "OBSERVED_RELATIVE_PHASE_INHERITANCE_V1",
            "authorities": authorities,
        },
        "contract": {
            "ambient_dimension": DIMENSION,
            "candidate_domain": [FIRST_RANK, LAST_RANK],
            "family": "q=mu_fit+W_r h; strict positive-dimensional proper subspace",
            "fit_algorithm": "one centered deterministic full SVD on fit bank only",
            "validation_statistic": "canonical R2 q-space mean squared reconstruction error",
            "complexity_rule": "one_standard_error_parsimony",
            "positivity_rule": "generated legal state requires finite q>=0; not a rank-selection objective",
            "labels": "balance_and_provenance_only",
            "tie_break": "smallest_rank",
            "rng": "NONE",
            "adaptive_feedback": False,
        },
        "git": {
            "branch": state["branch"],
            "head": state["head"],
            "dirty_paths": state["dirty_paths"],
            "consumer_script": _relative(repo_root, script_path),
            "consumer_script_sha256": file_sha256(script_path),
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "not_executed": [
            "generator",
            "evaluator",
            "score_sampling",
            "positivity_acceptance_validation",
            "scientific_panels",
            "optical_forward",
            "optimizer",
            "backward",
            "training",
            "CUDA",
        ],
        **computed,
    }
    artifact = _write_result(repo_root, output_dir, result)
    return {
        "status": result["status"],
        "classification": result["classification"],
        "r_selected": result["selection"]["r_selected"],
        "artifact": _relative(repo_root, artifact),
        "artifact_sha256": file_sha256(artifact),
        "git_head": state["head"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/20260907_v2/runs/"
            "lightgen_n3_low_rank_model_order_selection_v1_attempt_01"
        ),
        help="fresh write-once output directory",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
