from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.research.measure_scaled_chen_repaired_coordinate_bogt_weight_validity import (
    ALPHA,
    CONFIG_SCHEMA,
    MAX_RATIO,
    MIN_RATIO,
    PANEL_INDICES_SHA256,
    RC_ALPHA_TOO_STRONG,
    RC_ALPHA_TOO_WEAK,
    RC_ALPHA_UNDERDETERMINED,
    RC_ALPHA_VALID,
    alpha_gate,
    classify_alpha_ratios,
    effective_ratio,
    gradient_norm_from_gradients,
    load_config,
    next_write_once_output,
    optimizer_state_binding,
    state_digest,
    validate_gradient_parameter_subset,
)


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_v1.yaml"
MEASUREMENT_SCRIPT = ROOT / "scripts/research/measure_scaled_chen_repaired_coordinate_bogt_weight_validity.py"


class _Phase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(2, 1, 100, 100))


class _System(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Phase()
        self.decoder = _Phase()


class _PhaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.system = _System()


def test_registered_config_binds_repaired_coordinate_and_three_panels() -> None:
    _, payload, _, _ = load_config(CONFIG)
    assert payload["schema"] == CONFIG_SCHEMA
    assert payload["objective"]["alpha"] == ALPHA
    assert payload["repaired_calibration"]["schema"] == "scaled_chen_bogt_calibration.v2"
    assert payload["repaired_calibration"]["contract_version"] == "unit_aware_raw_mean_v2"
    assert payload["dataset"]["panel_indices_sha256"] == list(PANEL_INDICES_SHA256)
    assert payload["dataset"]["panel_count"] == 3
    assert payload["dataset"]["logical_batch_size"] == 10
    assert payload["dataset"]["microbatch_size"] == 1


def test_config_rejects_schema_or_intervention_changes(tmp_path: Path) -> None:
    import yaml

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    invalid_schema = dict(payload)
    invalid_schema["schema"] = "wrong"
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(invalid_schema, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_config(invalid_path)

    invalid_objective = dict(payload)
    invalid_objective["objective"] = {**payload["objective"], "alpha": 0.003}
    invalid_path.write_text(yaml.safe_dump(invalid_objective, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="objective"):
        load_config(invalid_path)


def test_effective_ratio_and_gradient_norm_use_registered_formulas() -> None:
    gradients = (torch.tensor([3.0, 4.0]), torch.tensor([12.0]))
    assert gradient_norm_from_gradients(gradients) == pytest.approx(13.0)
    assert effective_ratio(0.25, 2.0, 4.0) == pytest.approx(0.125)
    with pytest.raises(ValueError, match="finite positive"):
        effective_ratio(0.25, 0.0, 4.0)
    with pytest.raises(ValueError, match="finite"):
        gradient_norm_from_gradients((torch.tensor([float("nan")]),))


def test_alpha_gate_is_inclusive_and_classifies_mixed_panels() -> None:
    assert classify_alpha_ratios([MIN_RATIO, 0.10, MAX_RATIO]) == RC_ALPHA_VALID
    assert alpha_gate([MIN_RATIO, 0.10, MAX_RATIO])["all_inclusive"] is True
    assert classify_alpha_ratios([MIN_RATIO - 1.0e-6, 0.10, 0.10]) == RC_ALPHA_TOO_WEAK
    assert classify_alpha_ratios([0.10, MAX_RATIO + 1.0e-6, 0.10]) == RC_ALPHA_TOO_STRONG
    assert classify_alpha_ratios([MIN_RATIO - 1.0e-6, MAX_RATIO + 1.0e-6, 0.10]) == RC_ALPHA_UNDERDETERMINED
    assert classify_alpha_ratios([0.10, 0.10]) == "invalid_harness"


def test_gradient_parameter_subset_is_exactly_the_two_phase_parameters() -> None:
    model = _PhaseModel()
    selected = validate_gradient_parameter_subset(model)
    assert tuple(name for name, _ in selected) == (
        "system.encoder.phase",
        "system.decoder.phase",
    )

    model.system.decoder.phase.requires_grad_(False)
    with pytest.raises(ValueError, match="subset"):
        validate_gradient_parameter_subset(model)


def test_optimizer_state_binding_is_read_only_and_shape_bound() -> None:
    shape = (2, 1, 100, 100)
    valid = {
        "optimizer_state": {
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": 0.001,
                    "betas": (0.9, 0.999),
                    "eps": 1.0e-8,
                    "weight_decay": 0,
                }
            ],
            "state": {
                0: {"exp_avg": torch.zeros(shape), "exp_avg_sq": torch.ones(shape)},
                1: {"exp_avg": torch.zeros(shape), "exp_avg_sq": torch.ones(shape)},
            },
        }
    }
    result = optimizer_state_binding(valid, [shape, shape])
    assert result["status"] == "AVAILABLE_AND_BOUND"
    assert result["parameter_ids"] == [0, 1]

    invalid = {
        **valid,
        "optimizer_state": {
            **valid["optimizer_state"],
            "state": {
                **valid["optimizer_state"]["state"],
                1: {"exp_avg": torch.zeros(1), "exp_avg_sq": torch.ones(1)},
            },
        },
    }
    assert optimizer_state_binding(invalid, [shape, shape])["status"] == "UNRESOLVED"


def test_state_digest_and_write_once_preserve_isolation(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    before = state_digest(model)
    _ = model(torch.ones(1, 3))
    assert state_digest(model) == before

    requested = tmp_path / "weight_validity_attempt_01"
    first = next_write_once_output(requested)
    first.mkdir()
    second = next_write_once_output(requested)
    assert first.name.endswith("attempt_01")
    assert second.name.endswith("attempt_02")


def test_measurement_script_contains_no_update_or_optimizer_step() -> None:
    source = MEASUREMENT_SCRIPT.read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert ".step(" not in source
    assert "torch.optim" not in source
