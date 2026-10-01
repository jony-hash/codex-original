from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from scripts.research.scaled_chen.build_scaled_chen_frozen_output_semantic_evaluator import CONFIG_PATH, CONFIG_SCHEMA, EXPOSURE_SCALE, FROZEN_OUTPUT_SEMANTIC_EVALUATOR_NOT_QUALIFIED, FROZEN_OUTPUT_SEMANTIC_EVALUATOR_READY, CHECKPOINT_SHA256, _optical_state_isolation, audit_identity_leakage, fixed_output_view, load_config, next_write_once_output, qualification_classification, registered_split_indices, training_order, validate_checkpoint_identity


def test_registered_splits_are_exact_and_pairwise_disjoint() -> None:
    config = load_config(CONFIG_PATH)
    splits = registered_split_indices(config)
    assert splits["fit"].tolist() == list(range(50000))
    assert splits["validation"].tolist() == list(range(50000, 55000))
    assert splits["s2_confirmation"].tolist() == list(range(55000, 60000))
    assert splits["s2_selection"].numel() == 1000
    audit = audit_identity_leakage(
        splits["fit"],
        splits["validation"],
        splits["s2_confirmation"],
        splits["s2_selection"],
    )
    assert audit["all_pairwise_index_intersections_zero"] is True
    assert audit["underlying_source_identity_overlap_zero"] is True
    assert audit["s2_confirmation_consumed_by_evaluator"] is False


def test_loader_rejects_identity_and_preview_contract_changes(tmp_path: Path) -> None:
    source = Path(CONFIG_PATH)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert payload["schema"] == CONFIG_SCHEMA
    for key, value in (("schema", "wrong"), ("task_id", "wrong")):
        candidate = dict(payload)
        candidate[key] = value
        target = tmp_path / f"{key}.yaml"
        target.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(target)
    candidate = yaml.safe_load(source.read_text(encoding="utf-8"))
    candidate["authorization"] = dict(candidate["authorization"])
    candidate["authorization"]["s2_preview_forbidden"] = False
    target = tmp_path / "preview.yaml"
    target.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="preview"):
        load_config(target)


def test_exposure_view_uses_one_global_scale_and_fixed_pooling() -> None:
    raw = torch.tensor(
        [[[[0.0, EXPOSURE_SCALE], [2.0 * EXPOSURE_SCALE, -1.0]]]],
        dtype=torch.float32,
    )
    viewed = fixed_output_view(raw, EXPOSURE_SCALE, output_size=2)
    expected = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    assert torch.equal(viewed, expected)
    assert viewed.shape == (1, 1, 2, 2)


def test_training_order_is_epoch_seeded_and_deterministic() -> None:
    first = training_order(32, epoch=0, seed=1234)
    second = training_order(32, epoch=0, seed=1234)
    next_epoch = training_order(32, epoch=1, seed=1234)
    assert torch.equal(first, second)
    assert not torch.equal(first, next_epoch)
    assert sorted(first.tolist()) == list(range(32))


def test_checkpoint_identity_is_fail_closed() -> None:
    expected = {
        "path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "sha256": CHECKPOINT_SHA256,
        "global_step": 5000,
    }
    assert validate_checkpoint_identity(expected, expected) is True
    wrong = dict(expected)
    wrong["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_checkpoint_identity(wrong, expected)


def test_qualification_requires_validation_threshold_and_valid_contract() -> None:
    assert (
        qualification_classification(
            validation_accuracy=0.985,
            validation_metrics_finite=True,
            contract_valid=True,
            leakage_free=True,
            optical_isolated=True,
            freeze_replay=True,
        )
        == FROZEN_OUTPUT_SEMANTIC_EVALUATOR_READY
    )
    assert (
        qualification_classification(
            validation_accuracy=0.9849,
            validation_metrics_finite=True,
            contract_valid=True,
            leakage_free=True,
            optical_isolated=True,
            freeze_replay=True,
        )
        == FROZEN_OUTPUT_SEMANTIC_EVALUATOR_NOT_QUALIFIED
    )


def test_failure_is_a_qualification_failure_without_unlocking_s2_or_training() -> None:
    config = load_config(CONFIG_PATH)
    assert config["training"]["validation_model_selection"] is False
    assert config["training"]["checkpoint_selection"] == "final_epoch"
    assert config["authorization"]["s2_execution"] is False
    assert config["authorization"]["s2_preview_forbidden"] is True
    assert config["authorization"]["training_authorized"] is False
    assert config["forbidden_operations"][:7] == [
        "optical_training",
        "optical_backward",
        "optical_phase_update",
        "prior_sampling",
        "prior_generation",
        "s2_direct_replay",
        "s2_semantic_measurement",
    ]


def test_optical_state_isolation_rejects_any_bound_state_change() -> None:
    before = {
        "checkpoint_digest": "checkpoint",
        "model_state_digest": "model",
        "encoder_phase_digest": "encoder",
        "decoder_phase_digest": "decoder",
        "ols_state_config_digest": "ols",
        "controller_config_digest": "controller",
    }
    passed = _optical_state_isolation(before, dict(before))
    assert passed["state_unchanged"] is True
    changed = dict(before)
    changed["decoder_phase_digest"] = "changed"
    failed = _optical_state_isolation(before, changed)
    assert failed["state_unchanged"] is False
    assert failed["status"] == "BLOCKED_OPTICAL_STATE_ISOLATION"


def test_numeric_index_coincidence_across_mnist_splits_is_not_identity_leakage() -> None:
    audit = audit_identity_leakage(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
        torch.tensor([4, 5], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert audit["underlying_source_identity_overlap_zero"] is True
    assert audit["s2_confirmation_consumed_by_evaluator"] is False


def test_write_once_output_never_overwrites(tmp_path: Path) -> None:
    requested = tmp_path / "frozen_evaluator_attempt_01"
    first = next_write_once_output(requested)
    first.mkdir()
    second = next_write_once_output(requested)
    assert first.name.endswith("attempt_01")
    assert second.name.endswith("attempt_02")
