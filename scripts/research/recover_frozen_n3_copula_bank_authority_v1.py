"""Recover the exact generated banks from one completed N3 copula run.

The consumer first searches registered local N3 output roots for a complete
artifact carrying the historical report identity.  If none exists, it performs
one deterministic rematerialization through the original frozen consumer
helpers and accepts it only when every recorded q tensor hash matches exactly.
It never computes scientific metrics, changes the historical report, reads the
sealed confirmation source, or runs an optical/training path.
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
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import execute_n3_empirical_marginal_gaussian_copula_producer_v1 as original


TASK_ID = "RECOVER_FROZEN_N3_COPULA_BANK_AUTHORITY_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_empirical_marginal_gaussian_copula_producer_report.v1"
MANIFEST_SCHEMA = "n3_frozen_copula_generated_banks_manifest.v1"
BANK_SCHEMA = "n3_frozen_copula_generated_banks.v1"
SEMANTIC_AUTHORITY_ID = "FROZEN_N3_COPULA_GENERATED_BANKS_V1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_CONTRACT_SHA256 = "3edc9fee006f68eff6ba41205b7e4e6b6cbea3205080c4cb333689e554fba7a4"
EXPECTED_REPORT_SHA256 = "78b8e40f21015af8d704a89a83bb6b4043c0208386032b41b1ced4dd9bdeba7d"
EXPECTED_FOLD_ASSIGNMENT_SHA256 = "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32"
HISTORICAL_REPORT_PATH = Path(
    "outputs/20260908_v2/runs/"
    "n3_empirical_marginal_gaussian_copula_producer_v1_attempt_01/"
    "n3_empirical_marginal_gaussian_copula_producer_report.v1.json"
)
OUTPUT_DIR = Path("outputs/20260908_v2/runs/n3_frozen_copula_bank_recovery_v1_attempt_01")
BANK_ARTIFACT_NAME = "n3_frozen_copula_generated_banks.v1.pt"
MANIFEST_NAME = "n3_frozen_copula_generated_banks_manifest.v1.json"
SEARCH_ROOTS = (
    Path("outputs/20260908_v2/runs"),
    Path("outputs/20260907_v2/runs"),
)


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
    return dict(_mapping(json.loads(target.read_text(encoding="utf-8")), name))


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


def _historical_bank_records(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    folds = report.get("fold_execution")
    _require(isinstance(folds, list) and len(folds) == original.FOLD_COUNT, "historical fold report is incomplete")
    for fold_result in folds:
        fold = int(fold_result["fold"])
        _require(0 <= fold < original.FOLD_COUNT, "historical fold identity is invalid")
        for role in ("data_sampling", "model_sampling"):
            sampling = fold_result.get(role)
            _require(isinstance(sampling, list), f"historical {role} records are missing")
            expected_count = original.DATA_VS_MODEL_REPLICATES if role == "data_sampling" else original.MODEL_VS_MODEL_REPLICATES
            _require(len(sampling) == expected_count, f"historical {role} count changed for fold {fold}")
            role_name = "data-vs-model" if role == "data_sampling" else "model-vs-model"
            for item in sampling:
                item_map = _mapping(item, f"historical {role} item")
                replicate = int(item_map["replicate"])
                _require(item_map.get("role") == role_name, f"historical bank role changed for fold {fold}")
                _require(int(item_map.get("fold")) == fold, f"historical bank fold changed for {role_name}")
                _require(0 <= replicate < expected_count, f"historical replicate index is invalid for {role_name}")
                _require(int(item_map.get("bank_size")) == original.GENERATED_BANK_SIZE, "historical bank size changed")
                q_sha256 = item_map.get("q_sha256")
                _require(isinstance(q_sha256, str) and len(q_sha256) == 64, "historical q hash is missing")
                records.append(
                    {
                        "fold": fold,
                        "role": role_name,
                        "replicate": replicate,
                        "seed": int(item_map["seed"]),
                        "bank_size": int(item_map["bank_size"]),
                        "expected_q_sha256": q_sha256,
                    }
                )
    records.sort(key=lambda item: (item["fold"], item["role"], item["replicate"]))
    keys = [(item["fold"], item["role"], item["replicate"]) for item in records]
    _require(len(records) == original.FOLD_COUNT * (original.DATA_VS_MODEL_REPLICATES + original.MODEL_VS_MODEL_REPLICATES), "historical bank count changed")
    _require(len(set(keys)) == len(keys), "historical bank identities are duplicated")
    _require(len({item["expected_q_sha256"] for item in records}) == len(records), "historical q hashes are not unique")
    return records


def _verify_historical_inputs(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    report = _read_json_exact(repo_root, HISTORICAL_REPORT_PATH, EXPECTED_REPORT_SHA256, "historical copula execution report")
    _require(report.get("schema") == REPORT_SCHEMA, "historical report schema changed")
    _require(report.get("classification") == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED", "historical report classification changed")
    _require(report.get("status") == "valid_completed", "historical report is not a valid completed execution")
    contract = _mapping(report.get("contract"), "historical report contract")
    _require(contract.get("sha256") == EXPECTED_CONTRACT_SHA256, "historical contract identity changed")
    fold_identity = _mapping(report.get("development_fold_identity"), "historical fold identity")
    _require(fold_identity.get("assignment_sha256") == EXPECTED_FOLD_ASSIGNMENT_SHA256, "historical fold assignment changed")
    sealed = _mapping(report.get("sealed_confirmation"), "historical sealed status")
    _require(sealed.get("status") == "RESERVED_NOT_MATERIALIZED" and sealed.get("consumed") is False, "historical sealed boundary changed")
    settings = _mapping(report.get("execution_settings"), "historical execution settings")
    for key, expected in {
        "dtype": "float64",
        "generated_bank_size": original.GENERATED_BANK_SIZE,
        "data_vs_model_replicates": original.DATA_VS_MODEL_REPLICATES,
        "model_vs_model_replicates": original.MODEL_VS_MODEL_REPLICATES,
        "rng_namespace": original.RNG_NAMESPACE,
    }.items():
        _require(settings.get(key) == expected, f"historical execution setting changed: {key}")
    records = _historical_bank_records(report)
    return report, dict(contract), records


def _search_existing_artifacts(repo_root: Path) -> dict[str, Any]:
    candidates: list[str] = []
    for root in SEARCH_ROOTS:
        target = repo_root / root
        if not target.is_dir():
            continue
        for path in target.rglob("*.pt"):
            lower = path.as_posix().lower()
            if "n3" in lower or "copula" in lower or "empirical" in lower:
                candidates.append(_relative(repo_root, path))
    candidates = sorted(set(candidates))
    exact_matches: list[str] = []
    for relative in candidates:
        path = repo_root / relative
        try:
            payload = safe_torch_load(path, map_location="cpu")
        except (OSError, RuntimeError, ValueError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if payload.get("semantic_authority_id") != SEMANTIC_AUTHORITY_ID:
            continue
        source = _mapping(payload.get("source_execution_report"), "candidate source execution report")
        if source.get("sha256") != EXPECTED_REPORT_SHA256:
            continue
        banks = payload.get("banks")
        if isinstance(banks, Mapping):
            exact_matches.append(relative)
    return {
        "searched_roots": [_relative(repo_root, repo_root / root) for root in SEARCH_ROOTS],
        "candidate_tensor_artifacts": candidates,
        "complete_exact_historical_matches": exact_matches,
        "complete_existing_artifact_available": len(exact_matches) == 1,
        "partial_or_unverified_candidates_consumed": False,
    }


def _validate_existing_payload(
    repo_root: Path,
    path: Path,
    records: list[dict[str, Any]],
) -> dict[str, torch.Tensor] | None:
    payload = safe_torch_load(repo_root / path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("semantic_authority_id") != SEMANTIC_AUTHORITY_ID:
        return None
    if _mapping(payload.get("source_execution_report"), "candidate source execution report").get("sha256") != EXPECTED_REPORT_SHA256:
        return None
    banks = payload.get("banks")
    if not isinstance(banks, Mapping):
        return None
    recovered: dict[str, torch.Tensor] = {}
    for item in records:
        key = _bank_key(item)
        value = banks.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != (original.GENERATED_BANK_SIZE, original.DIMENSION) or str(value.dtype) != "torch.float64":
            return None
        if tensor_sha256(value) != item["expected_q_sha256"]:
            return None
        recovered[key] = value.detach().cpu().contiguous()
    return recovered


def _bank_key(item: Mapping[str, Any]) -> str:
    return f"fold={int(item['fold'])}|role={item['role']}|replicate={int(item['replicate'])}"


def _rematerialize(
    repo_root: Path,
    contract: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    _require(contract.get("classification") == "DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1_READY", "frozen contract is not ready")
    q, _source_indices, _labels, fold_ids, _pool_identity, fold_identity = original._verify_development_identity(repo_root, contract)
    _require(fold_identity.get("assignment_sha256") == EXPECTED_FOLD_ASSIGNMENT_SHA256, "runtime fold assignment changed")
    expected_by_key = {_bank_key(item): item for item in records}
    banks: dict[str, torch.Tensor] = {}
    observed: list[dict[str, Any]] = []
    with torch.no_grad():
        fits: list[dict[str, Any]] = []
        for fold in range(original.FOLD_COUNT):
            fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
            fits.append(original._fit_fold(q.index_select(0, fit_positions).contiguous()))
        for fold, fit in enumerate(fits):
            for role, count in (("data-vs-model", original.DATA_VS_MODEL_REPLICATES), ("model-vs-model", original.MODEL_VS_MODEL_REPLICATES)):
                for replicate in range(count):
                    key = f"fold={fold}|role={role}|replicate={replicate}"
                    expected = expected_by_key[key]
                    bank, sampling = original._sample_bank(fit, role=role, fold=fold, replicate=replicate)
                    _require(int(sampling["seed"]) == int(expected["seed"]), f"seed changed for {key}")
                    _require(tuple(bank.shape) == (original.GENERATED_BANK_SIZE, original.DIMENSION), f"shape changed for {key}")
                    _require(str(bank.dtype) == "torch.float64", f"dtype changed for {key}")
                    _require(bool(torch.isfinite(bank).all()) and bool(bank.ge(0.0).all()), f"bank values invalid for {key}")
                    observed_hash = tensor_sha256(bank)
                    _require(observed_hash == expected["expected_q_sha256"], f"historical q hash mismatch for {key}")
                    banks[key] = bank.detach().cpu().contiguous()
                    observed.append(
                        {
                            **expected,
                            "bank_key": key,
                            "observed_q_sha256": observed_hash,
                            "shape": list(bank.shape),
                            "dtype": str(bank.dtype),
                            "seed_verified": True,
                            "hash_equal": True,
                        }
                    )
    _require(len(banks) == len(records) == 60, "recovered bank count is incomplete")
    return banks, observed


def _write_recovery(
    repo_root: Path,
    output_dir: Path,
    *,
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    records: list[dict[str, Any]],
    banks: Mapping[str, torch.Tensor],
    observed: list[dict[str, Any]],
    search: Mapping[str, Any],
    recovery_mode: str,
    existing_artifact: str | None,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once recovery output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    published = False
    try:
        artifact_path: Path | None = None
        artifact_sha256: str | None = None
        if existing_artifact is None:
            artifact_path = temporary / BANK_ARTIFACT_NAME
            payload = {
                "schema": BANK_SCHEMA,
                "version": 1,
                "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
                "source_contract": {
                    "path": _relative(repo_root, repo_root / original.CONTRACT_PATH),
                    "sha256": EXPECTED_CONTRACT_SHA256,
                },
                "source_execution_report": {
                    "path": _relative(repo_root, repo_root / HISTORICAL_REPORT_PATH),
                    "sha256": EXPECTED_REPORT_SHA256,
                },
                "fold_assignment_sha256": EXPECTED_FOLD_ASSIGNMENT_SHA256,
                "coordinate": "R2_AMPLITUDE",
                "dtype": "torch.float64",
                "bank_shape": [original.GENERATED_BANK_SIZE, original.DIMENSION],
                "bank_count": len(banks),
                "banks": dict(banks),
            }
            atomic_torch_save_new(artifact_path, payload, artifact_kind="frozen N3 copula bank artifact")
            artifact_sha256 = file_sha256(artifact_path)
            artifact_reference = {
                "path": _relative(repo_root, output_dir / BANK_ARTIFACT_NAME),
                "sha256": artifact_sha256,
                "schema": BANK_SCHEMA,
            }
        else:
            artifact_reference = {
                "path": existing_artifact,
                "sha256": file_sha256(repo_root / existing_artifact),
                "schema": BANK_SCHEMA,
            }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "gate": GATE,
            "status": "valid_completed",
            "classification": "FROZEN_N3_COPULA_BANK_AUTHORITY_RECOVERED",
            "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
            "git": {
                **git,
                "recovery_script": _relative(repo_root, Path(__file__).resolve()),
                "recovery_script_sha256": file_sha256(Path(__file__).resolve()),
            },
            "source_contract": {
                "path": _relative(repo_root, repo_root / original.CONTRACT_PATH),
                "sha256": EXPECTED_CONTRACT_SHA256,
                "classification": contract["classification"],
            },
            "source_execution_report": {
                "path": _relative(repo_root, repo_root / HISTORICAL_REPORT_PATH),
                "sha256": EXPECTED_REPORT_SHA256,
                "classification": report["classification"],
            },
            "fold_assignment_sha256": EXPECTED_FOLD_ASSIGNMENT_SHA256,
            "recovery_mode": recovery_mode,
            "existing_artifact_search": dict(search),
            "artifact": artifact_reference,
            "bank_count": len(records),
            "expected_bank_count": len(records),
            "recovered_bank_count": len(banks),
            "per_bank_identity": observed,
            "per_bank_hash_equality": all(item["hash_equal"] for item in observed),
            "shape_dtype_contract": {
                "shape": [original.GENERATED_BANK_SIZE, original.DIMENSION],
                "dtype": "torch.float64",
                "all_verified": all(item["shape"] == [original.GENERATED_BANK_SIZE, original.DIMENSION] and item["dtype"] == "torch.float64" for item in observed),
            },
            "historical_evidence_unchanged": True,
            "scientific_semantics": {
                "historical_sample_identity_recovered": True,
                "new_scientific_evidence": False,
                "historical_metrics_recomputed": False,
                "producer_rerun_as_scientific_execution": False,
            },
            "sealed_confirmation": {
                "status": "RESERVED_NOT_MATERIALIZED",
                "consumed": False,
                "materialized": False,
                "sample_values_loaded": False,
            },
            "claim_boundary": {
                "exact_historical_copula_generated_bank_authority_recovered": True,
                "new_producer_execution_result": False,
                "copula_scientifically_supported": False,
                "marginal_dependence_attribution_completed": False,
                "independent_confirmation": False,
                "full_complex_OLS": False,
                "novel_OLS": False,
                "semantic_generation": False,
                "generation_readiness": "not_ready",
            },
            "not_executed": [
                "marginal-vs-dependence attribution",
                "scientific compatibility metrics",
                "historical report modification",
                "alternative bank generation",
                "producer-family selection",
                "selector refinement",
                "sealed confirmation source loading or materialization",
                "phase-source work",
                "novel complex OLS",
                "generator",
                "evaluator",
                "CUDA",
                "optimizer",
                "backward",
                "training",
            ],
        }
        atomic_write_new_json(temporary / MANIFEST_NAME, manifest, artifact_kind="frozen N3 copula bank recovery manifest")
        if output_dir.exists():
            raise FileExistsError(f"recovery output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
        published = True
        manifest_path = output_dir / MANIFEST_NAME
        return {
            "classification": manifest["classification"],
            "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
            "recovery_mode": recovery_mode,
            "artifact": _relative(repo_root, output_dir / BANK_ARTIFACT_NAME) if existing_artifact is None else existing_artifact,
            "artifact_sha256": artifact_sha256 if existing_artifact is None else artifact_reference["sha256"],
            "manifest": _relative(repo_root, manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "bank_count": len(banks),
            "git_head": git["head"],
            "worktree_clean_at_run": git["worktree_clean"],
        }
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 research authority branch")
    _require(git["worktree_clean"], "recovery must start from a clean committed state")
    report, contract_ref, records = _verify_historical_inputs(repo_root)
    contract = original._load_contract(repo_root)
    _require(contract.get("classification") == contract_ref.get("classification"), "runtime contract differs from historical contract")
    search = _search_existing_artifacts(repo_root)

    existing_artifact: str | None = None
    banks: dict[str, torch.Tensor] | None = None
    observed: list[dict[str, Any]] = []
    if len(search["complete_exact_historical_matches"]) == 1:
        existing_artifact = str(search["complete_exact_historical_matches"][0])
        banks = _validate_existing_payload(repo_root, Path(existing_artifact), records)
        _require(banks is not None and len(banks) == len(records), "existing exact artifact is incomplete")
        observed = [
            {
                **item,
                "bank_key": _bank_key(item),
                "observed_q_sha256": item["expected_q_sha256"],
                "shape": [original.GENERATED_BANK_SIZE, original.DIMENSION],
                "dtype": "torch.float64",
                "seed_verified": True,
                "hash_equal": True,
            }
            for item in records
        ]
        recovery_mode = "EXISTING_ARTIFACT_RECOVERY"
    else:
        banks, observed = _rematerialize(repo_root, contract, records)
        recovery_mode = "EXACT_DETERMINISTIC_REMATERIALIZATION"
    return _write_recovery(
        repo_root,
        output_dir,
        report=report,
        contract=contract,
        records=records,
        banks=banks,
        observed=observed,
        search=search,
        recovery_mode=recovery_mode,
        existing_artifact=existing_artifact,
        git=git,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="fresh write-once recovery output directory")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
