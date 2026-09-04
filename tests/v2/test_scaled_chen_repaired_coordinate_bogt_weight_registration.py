from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.research.register_scaled_chen_repaired_coordinate_bogt_weight import (
    ALPHA_OLD,
    CONFIG_SCHEMA,
    EXPECTED_ALPHA_DECIMAL,
    MAX_RATIO,
    MIN_RATIO,
    NEW_CALIBRATION_CONTRACT,
    NEW_CALIBRATION_SCHEMA,
    PANEL_INDICES_SHA256,
    REGISTRATION_ALPHA,
    REGISTRATION_GATE,
    classify_registration_ratios,
    classify_set_provenance,
    derive_registration,
    load_config,
    load_rcwv_authority,
    next_write_once_output,
)


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_v1.yaml"
REGISTRATION_SCRIPT = ROOT / "scripts/research/register_scaled_chen_repaired_coordinate_bogt_weight.py"


def test_registered_config_binds_v2_identity_and_single_alpha() -> None:
    _, payload, _, _ = load_config(CONFIG)
    assert payload["schema"] == CONFIG_SCHEMA
    assert payload["registration"]["alpha"] == REGISTRATION_ALPHA
    assert payload["historical_registration"]["target_ratio"] == 0.10
    assert payload["historical_registration"]["acceptance_interval"] == [MIN_RATIO, MAX_RATIO]
    repaired = payload["repaired_calibration"]
    assert repaired["schema"] == NEW_CALIBRATION_SCHEMA
    assert repaired["contract_version"] == NEW_CALIBRATION_CONTRACT
    assert payload["dataset"]["panel_indices_sha256"] == list(PANEL_INDICES_SHA256)
    assert payload["registration"]["candidate_count"] == 1


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("repaired_calibration", "payload_sha256", "calibration identity"),
        ("checkpoint", "sha256", "checkpoint identity"),
        ("rcwv", "gradient_panel_metrics_sha256", "RCWV"),
    ],
)
def test_config_rejects_wrong_boundary_identity(
    tmp_path: Path, section: str, field: str, message: str
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload[section][field] = "0" * 64
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(invalid_path)


def test_rcwv_provenance_classifies_set_b_as_authority_and_set_a_as_context() -> None:
    authority = load_rcwv_authority(load_config(CONFIG)[1])
    assert classify_set_provenance(authority) == {
        "set_a": "A_IS_REPLAY_CONTEXT_ONLY",
        "set_b": "B_IS_REGISTRATION_AUTHORITY",
    }


def test_registration_reconstructs_k_alpha_and_predicted_ratios_from_raw_artifact() -> None:
    authority = load_rcwv_authority(load_config(CONFIG)[1])
    result = derive_registration(
        authority["repaired_panels"],
        old_alpha=ALPHA_OLD,
        target_ratio=0.10,
        lower=MIN_RATIO,
        upper=MAX_RATIO,
    )
    assert result["gate"] == REGISTRATION_GATE
    assert result["alpha"] == pytest.approx(REGISTRATION_ALPHA, rel=0.0, abs=1.0e-18)
    assert result["alpha_decimal"] == format(EXPECTED_ALPHA_DECIMAL, "f")
    assert result["k_values"] == pytest.approx(
        [651.3518921854, 516.9095296851, 525.7839101253], rel=0.0, abs=1.0e-10
    )
    assert result["predicted_ratios"] == pytest.approx(
        [0.1166303093, 0.0925572168, 0.0941462530], rel=0.0, abs=1.0e-10
    )
    assert result["gate_details"]["all_inclusive"] is True


def test_registration_gate_is_inclusive() -> None:
    assert classify_registration_ratios([MIN_RATIO, 0.10, MAX_RATIO]) == REGISTRATION_GATE
    assert classify_registration_ratios([MIN_RATIO - 1.0e-8, 0.10, MAX_RATIO]) != REGISTRATION_GATE
    assert classify_registration_ratios([MIN_RATIO, 0.10, MAX_RATIO + 1.0e-8]) != REGISTRATION_GATE


def test_old_alpha_is_immutable_and_only_one_candidate_is_allowed() -> None:
    authority = load_rcwv_authority(load_config(CONFIG)[1])
    panels = copy.deepcopy(authority["repaired_panels"])
    with pytest.raises(ValueError, match="old alpha"):
        derive_registration(
            panels,
            old_alpha=ALPHA_OLD + 1.0e-6,
            target_ratio=0.10,
            lower=MIN_RATIO,
            upper=MAX_RATIO,
        )
    _, payload, _, _ = load_config(CONFIG)
    assert payload["registration"]["candidate_count"] == 1
    assert payload["registration"]["selection"] == "analytic_only_no_sweep"


def test_write_once_output_never_reuses_an_existing_attempt(tmp_path: Path) -> None:
    requested = tmp_path / "lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_v1_attempt_01"
    first = next_write_once_output(requested)
    first.mkdir()
    second = next_write_once_output(requested)
    assert first.name.endswith("attempt_01")
    assert second.name.endswith("attempt_02")


def test_registration_source_has_no_execution_update_path() -> None:
    source = REGISTRATION_SCRIPT.read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert ".step(" not in source
    assert "torch.optim" not in source
    assert "torch.autograd" not in source
