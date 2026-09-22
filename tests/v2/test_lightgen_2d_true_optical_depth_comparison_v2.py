"""Focused contract and persistence-order checks for N5 V2."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from scripts.research import define_lightgen_2d_true_optical_depth_comparison_v2_contract as contract
from scripts.research import execute_lightgen_2d_true_optical_depth_comparison_v2 as v2


def test_successor_contract_freezes_the_two_v2_arms_without_v1_observation_reuse() -> None:
    candidates = contract._candidate_definitions()

    assert candidates["A_CONTINUED_C1"]["parameter_counts"]["total_trainable"] == 23_600
    assert candidates["B_TRUE_DEPTH_C1"]["parameter_counts"]["total_trainable"] == 33_600
    assert candidates["B_TRUE_DEPTH_C1"]["midpoint_phase_shape"] == [1, 1, 100, 100]
    assert contract.TASK_ID == "DEFINE_N5_SUCCESSOR_TRUE_OPTICAL_DEPTH_COMPARISON_V2_CONTRACT"


def test_cyclic_diagnostics_are_derivable_without_a_forward() -> None:
    sufficient = {
        "cyclic_observations": {
            "template_base_margin": torch.tensor([0.0, 0.2]),
            "template_shifted_margin": torch.tensor([0.1, 0.1]),
            "smooth_base_margin": torch.tensor([0.0, -0.1]),
            "smooth_shifted_margin": torch.tensor([0.2, 0.0]),
        }
    }

    diagnostics = v2._cyclic_diagnostics_from_observations(sufficient)

    assert diagnostics["rows"] == 2
    assert diagnostics["computed_without_forward_from_persisted_observations"] is True
    assert diagnostics["template_delta_mean"] == 0.0
    assert diagnostics["smooth_delta_mean"] == pytest.approx(0.15)


def test_sufficient_observation_persistence_precedes_optional_diagnostics() -> None:
    source = Path(v2.__file__).read_text(encoding="utf-8")

    assert source.index("sufficient_ref = _save_sufficient") < source.index("adapter_diag = _adapter_diagnostics")
    assert source.index("sufficient_ref = _save_sufficient") < source.index("generator_diag = _generator_diagnostics")
    assert "V1._adapter_observation" in source or "_adapter_diagnostics" in source
    assert hasattr(v2.V10, "_adapter_observation")
    assert not hasattr(v2.N4, "_adapter_observation")


def test_terminal_checkpoint_validation_uses_cursor_only() -> None:
    source = Path(v2.__file__).read_text(encoding="utf-8")
    validator = source[source.index("def _validate_checkpoint_reload"):source.index("def _save_tensor_artifact")]

    assert "V1._validate_checkpoint_reload" in validator
    v1_source = Path(v2.V1.__file__).read_text(encoding="utf-8")
    terminal = v1_source[v1_source.index("def _validate_checkpoint_reload"):v1_source.index("def _save_checkpoint")]
    assert "if int(loaded[\"relative_step\"]) < SCIENTIFIC_STEPS:" in terminal
    assert "sampler_clone.cursor == CONTINUATION_TOTAL_STEPS" in terminal
