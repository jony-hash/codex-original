"""Contract tests for the frozen S3 controller-to-OLS writability audit."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.scaled_chen.run_scaled_chen_controller_to_ols_writability_audit import CONFIG_SCHEMA, MANIFEST_SCHEMA, TASK_ID, build_controller_panel, classify_rank, classify_s3, fixed_label_permutation, load_config, _assert_output_tree, validate_amplitude_output, validate_controller_authority_payload, validate_power_observer


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/research/scaled_chen/lightgen_scaled_chen_controller_to_ols_writability_audit_v1.yaml"


def test_frozen_controller_panel_is_label_major_and_reused() -> None:
    panel = build_controller_panel()

    assert panel["codes"].shape == (1000, 8)
    assert panel["labels"].shape == (1000,)
    assert panel["split"].shape == (1000,)
    assert torch.equal(panel["codes"][:100], panel["codes"][100:200])
    assert torch.equal(panel["labels"], torch.arange(10).repeat_interleave(100))
    assert torch.equal(panel["split"][:50], torch.zeros(50, dtype=torch.long))
    assert torch.equal(panel["split"][50:100], torch.ones(50, dtype=torch.long))


def test_panel_and_null_are_deterministic() -> None:
    first = build_controller_panel()
    second = build_controller_panel()
    assert torch.equal(first["base_codes"], second["base_codes"])
    assert torch.equal(first["codes"], second["codes"])
    assert torch.equal(first["labels"], second["labels"])
    null_first = fixed_label_permutation(first["labels"])
    null_second = fixed_label_permutation(second["labels"])
    assert torch.equal(null_first, null_second)
    assert torch.equal(torch.sort(null_first).values, torch.sort(first["labels"]).values)


def test_boundary_validators_reject_wrong_shapes_and_nonfinite_values() -> None:
    amplitude = torch.full((2, 1, 200, 200), 0.5)
    assert validate_amplitude_output(amplitude).shape == amplitude.shape
    with pytest.raises(ValueError, match="amplitude"):
        validate_amplitude_output(torch.ones(2, 200, 200))
    with pytest.raises(ValueError, match="amplitude"):
        validate_amplitude_output(torch.full_like(amplitude, float("nan")))

    coefficients = torch.ones(2, 100, dtype=torch.complex64)
    scale = torch.ones(100)
    assert validate_power_observer(coefficients, scale).shape == (2, 100)
    with pytest.raises(ValueError, match="coefficients"):
        validate_power_observer(torch.ones(2, 99, dtype=torch.complex64), scale)
    with pytest.raises(ValueError, match="scale"):
        validate_power_observer(coefficients, torch.zeros(100))


def test_effective_rank_classification_is_fail_closed() -> None:
    assert classify_rank(torch.tensor([2.0, 1.0, 0.0, 0.0])) == "CODE8_LOCAL_CONTROL_PARTIAL_RANK"
    assert classify_rank(torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])) == "CODE8_LOCAL_CONTROL_FULL_RANK"
    assert classify_rank(torch.zeros(8)) == "CODE8_LOCAL_CONTROL_DEGENERATE"


def test_s3_classification_requires_all_registered_gates() -> None:
    kwargs = dict(
        authority=True,
        interface=True,
        targeting=True,
        no_collapse=True,
        reachability=True,
        code_effect=True,
        isolation=True,
        finite=True,
    )
    assert classify_s3(**kwargs) == "S3_CONTROL_TO_OLS_WRITABILITY_SUPPORTED"
    assert classify_s3(**{**kwargs, "reachability": False}) == (
        "S3_LABEL_DIRECTION_SUPPORTED_BUT_OLS_MANIFOLD_NOT_REACHED"
    )
    assert classify_s3(**{**kwargs, "targeting": False, "reachability": True}) == (
        "S3_OLS_REACHABLE_BUT_LABEL_SEMANTICS_NOT_CONTROLLED"
    )


def test_config_and_authority_reject_schema_or_task_mismatch(tmp_path: Path) -> None:
    path, payload, _, _ = load_config(CONFIG_PATH)
    assert path == CONFIG_PATH
    assert payload["schema"] == CONFIG_SCHEMA
    assert payload["task_id"] == TASK_ID

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema: wrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_config(invalid)

    authority = {
        "schema": "wrong",
        "task_id": TASK_ID,
    }
    with pytest.raises(ValueError, match="schema"):
        validate_controller_authority_payload(authority)


def test_manifest_tree_contract_rejects_tiff_and_unknown_files(tmp_path: Path) -> None:
    (tmp_path / "unknown.tif").write_bytes(b"not-a-figure")
    with pytest.raises(ValueError, match="tree"):
        _assert_output_tree(tmp_path)
    assert MANIFEST_SCHEMA.startswith("lightgen_scaled_chen")
