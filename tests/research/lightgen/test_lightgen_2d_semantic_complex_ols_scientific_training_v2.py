from __future__ import annotations

import torch
import pytest

from scripts.research.lightgen.execute_lightgen_2d_semantic_complex_ols_scientific_training_v2 import CONTRACT_RELATIVE_PATH, CONTRACT_SHA256, DEFAULT_OUTPUT_DIR, DEVELOPMENT_SELECTION_SEED, PREFLIGHT_INPUT_SHA256, PREFLIGHT_LINEAGE_SHA256, PREFLIGHT_REPORT_SHA256, SUCCESS_CLASSIFICATION, TASK_ID, _fixed_development_selection


def test_scientific_executor_is_bound_to_corrected_attempt07_authority() -> None:
    assert TASK_ID == "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2"
    assert SUCCESS_CLASSIFICATION == "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2_VALID_COMPLETED"
    assert CONTRACT_RELATIVE_PATH.as_posix().endswith("lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json")
    assert CONTRACT_SHA256 == "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"
    assert PREFLIGHT_INPUT_SHA256 == "2e090926b057f70ed9cacc63473d16786336aa39b9371da2dcfdbccfead94b1d"
    assert PREFLIGHT_REPORT_SHA256 == "b7bc2482d55af70ad8fc7a1b204c636425e9bca64448feda9e11e0111b5d64a4"
    assert PREFLIGHT_LINEAGE_SHA256 == "30913cd07ad318fbd94d2e5a4a346da6ca8390087e47a0b145d2c08b54b2d3bd"
    assert DEFAULT_OUTPUT_DIR.as_posix().startswith("outputs/")


def test_fixed_development_selection_is_deterministic_and_balanced() -> None:
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    development_indices = torch.cat([torch.arange(digit, 60_000, 10)[:500] for digit in range(10)])
    selected_a = _fixed_development_selection(labels, development_indices)
    selected_b = _fixed_development_selection(labels, development_indices)
    assert DEVELOPMENT_SELECTION_SEED == 4321
    assert torch.equal(selected_a, selected_b)
    assert selected_a.numel() == 1_000
    assert torch.unique(selected_a).numel() == 1_000
    assert torch.equal(torch.bincount(labels.index_select(0, selected_a), minlength=10), torch.full((10,), 100, dtype=torch.long))


def test_fixed_development_selection_rejects_unbalanced_development_role() -> None:
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    development_indices = torch.cat([torch.arange(digit, 60_000, 10)[:500] for digit in range(9)])
    with pytest.raises(Exception, match="balanced"):
        _fixed_development_selection(labels, development_indices)
