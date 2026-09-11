"""Regression checks for the bounded V2 training-source authority correction."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import scripts.research.resolve_lightgen_2d_semantic_complex_ols_v2_training_source_authority_v1 as resolver
from ldm_project.v2.evidence import file_sha256


ROOT = Path(__file__).resolve().parents[2]
CORRECTION_AUTHORITY = ROOT / resolver.OUTPUT_DIR / resolver.CORRECTION_NAME
CORRECTED_CONTRACT = ROOT / resolver.OUTPUT_DIR / resolver.CORRECTED_CONTRACT_NAME


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_reported_v2_value_is_malformed_but_current_value_is_valid() -> None:
    assert len(resolver.MALFORMED_SHA256) == 63
    assert resolver.SHA256_RE.fullmatch(resolver.MALFORMED_SHA256) is None
    assert len(resolver.CANONICAL_SHA256) == 64
    assert resolver.SHA256_RE.fullmatch(resolver.CANONICAL_SHA256)


def test_corrected_contract_changes_only_source_authority() -> None:
    original = _load_json(ROOT / resolver.ORIGINAL_CONTRACT)
    original_snapshot = copy.deepcopy(original)
    corrected, comparison = resolver._corrected_contract(
        original,
        correction_path=(resolver.OUTPUT_DIR / resolver.CORRECTION_NAME).as_posix(),
        provenance=[
            {
                "authority_id": "TEST_EXACT_SOURCE_AUTHORITY",
                "path": "outputs/test/exact_source.json",
                "artifact_sha256": "a" * 64,
                "claim_sha256": resolver.CANONICAL_SHA256,
            }
        ],
        canonical={"size_bytes": 47_521_438},
        corrected_at="2026-09-10T00:00:00+00:00",
    )

    assert original == original_snapshot
    assert corrected["schema"] == original["schema"] == resolver.CONTRACT_SCHEMA
    assert corrected["version"] == original["version"] == 2
    assert corrected["classification"] == original["classification"]
    assert corrected["data_contract"]["source_artifacts"]["training"]["sha256"] == resolver.CANONICAL_SHA256
    assert comparison["all_other_fields_unchanged"] is True
    assert comparison["original_semantic_normal_form_sha256"] == comparison["corrected_semantic_normal_form_sha256"]
    assert comparison["changed_paths"] == [resolver.SOURCE_FIELD, "authority_correction"]
    assert corrected["authority_correction"]["correction_only"] is True
    assert corrected["authority_correction"]["source_file_unchanged"] is True
    assert corrected["authority_correction"]["scientific_semantics_unchanged"] is True


def test_published_correction_artifacts_are_hash_bound_and_governance_closed() -> None:
    assert CORRECTION_AUTHORITY.is_file()
    assert CORRECTED_CONTRACT.is_file()
    assert sorted(path.name for path in CORRECTION_AUTHORITY.parent.iterdir()) == sorted(
        [resolver.CORRECTION_NAME, resolver.CORRECTED_CONTRACT_NAME]
    )
    assert file_sha256(CORRECTION_AUTHORITY) == "fdf03a0a8568db947011549e7869ff11ab972e19c1e01fbc3e7d2e9d850dbeb0"
    assert file_sha256(CORRECTED_CONTRACT) == "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"

    authority = _load_json(CORRECTION_AUTHORITY)
    corrected = _load_json(CORRECTED_CONTRACT)
    assert authority["result"] == "CONFIRMED"
    assert authority["classification"] == resolver.CLASSIFICATION
    assert authority["original_v2_contract"]["value"] == resolver.MALFORMED_SHA256
    assert authority["original_v2_contract"]["sha256_regex_valid"] is False
    assert authority["current_source"]["sha256"] == resolver.CANONICAL_SHA256
    assert authority["provenance"]["competing_valid_canonical_sha256"] == []
    assert authority["semantic_equivalence"]["all_other_fields_unchanged"] is True
    assert authority["correction_boundary"]["consumer_rebind_required"] is True
    assert authority["correction_boundary"]["training_authorized"] is False
    assert authority["correction_boundary"]["formal_promotion"] is False
    assert authority["correction_boundary"]["generation_readiness"] == "not_ready"
    assert authority["correction_boundary"]["cuda_executed"] is False
    assert authority["correction_boundary"]["scientific_training_executed"] is False
    assert authority["correction_boundary"]["protected_test_read"] is False
    assert authority["correction_boundary"]["sealed_read"] is False

    assert corrected["data_contract"]["source_artifacts"]["training"]["sha256"] == resolver.CANONICAL_SHA256
    assert corrected["authority_correction"]["original_contract_preserved"] is True
    assert corrected["authority_correction"]["correction_only"] is True
    assert file_sha256(ROOT / resolver.ORIGINAL_CONTRACT) == resolver.ORIGINAL_CONTRACT_SHA256
    assert file_sha256(ROOT / resolver.TRAINING_SOURCE) == resolver.CANONICAL_SHA256
