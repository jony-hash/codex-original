"""Regression tests for trust-boundary hash-domain handling."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.research.diagnose_lightgen_2d_fixed_ols_generator_conditionality_v1 as diagnosis


def _smooth_authorities(file_sha256: str) -> dict[str, object]:
    return {
        "evaluators": {
            "smooth_checkpoint": {
                "path": "outputs/example/smooth_mnist_evaluator_v21.pt",
                "role": "Smooth v2.1 evaluator checkpoint",
                "sha256": file_sha256,
            }
        }
    }


def test_smooth_file_sha_and_loaded_state_tree_sha_are_distinct_domains() -> None:
    state_tree_sha256 = "b" * 64

    identity = diagnosis._validate_smooth_evaluator_identity(
        _smooth_authorities(diagnosis.SMOOTH_EVALUATOR_SHA256),
        {"smooth_state_sha256": state_tree_sha256},
    )

    assert identity["file_sha256"] == diagnosis.SMOOTH_EVALUATOR_SHA256
    assert identity["state_dict_tree_sha256"] == state_tree_sha256
    assert identity["file_sha256"] != identity["state_dict_tree_sha256"]


def test_required_same_domain_file_sha_mismatch_still_fails() -> None:
    with pytest.raises(diagnosis.InvalidDiagnosis, match="test artifact SHA mismatch"):
        diagnosis._read_json_ref(Path(__file__).resolve().parents[2], "AGENTS.md", "0" * 64, "test artifact")
