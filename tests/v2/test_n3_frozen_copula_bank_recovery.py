from __future__ import annotations

from pathlib import Path

from scripts.research import recover_frozen_n3_copula_bank_authority_v1 as recovery


def test_historical_report_exposes_complete_unique_bank_identity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report, _contract, records = recovery._verify_historical_inputs(repo_root)
    assert report["classification"] == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED"
    assert len(records) == 60
    assert sum(item["role"] == "data-vs-model" for item in records) == 20
    assert sum(item["role"] == "model-vs-model" for item in records) == 40
    assert len({item["expected_q_sha256"] for item in records}) == 60


def test_existing_search_does_not_consume_unrelated_tensor_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    search = recovery._search_existing_artifacts(repo_root)
    assert search["complete_existing_artifact_available"] is False
    assert search["partial_or_unverified_candidates_consumed"] is False


def test_bank_key_binds_fold_role_and_replicate() -> None:
    assert recovery._bank_key({"fold": 3, "role": "model-vs-model", "replicate": 7}) == (
        "fold=3|role=model-vs-model|replicate=7"
    )
