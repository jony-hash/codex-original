from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import _cli_research_optical_manifold as research_router
from ldm_project.v2.application import cli
from ldm_project.v2.application.research import optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit as audit


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit_v1.yaml"
REVIEW = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit_review_v1.json"


def _synthetic_probe(*, b_nonzero: bool = False) -> dict[str, torch.Tensor | int | float]:
    weights = torch.zeros((257, 40000), dtype=torch.float32)
    weights[0, 6498:] = 1.0
    if b_nonzero:
        weights[-1, 3] = 1.0
    return {
        "W": weights[:-1],
        "b": weights[-1],
        "full_weights": weights,
        "dynamic_mass": weights[:-1].abs().to(torch.float64).sum(dim=0),
        "zero_mask": torch.arange(40000) < 6498,
        "active_mask": torch.arange(40000) >= 6498,
    }


def _fake_stats(
    name: str,
    *,
    exact_constant: bool = True,
    first_value: float = 0.0,
    nonzero_indices: tuple[int, ...] = (),
) -> audit.PanelStats:
    nonzero = torch.zeros(40000, dtype=torch.int64)
    if nonzero_indices:
        nonzero[list(nonzero_indices)] = 1
    exact = torch.full((40000,), exact_constant, dtype=torch.bool)
    first = torch.full((40000,), first_value, dtype=torch.float32)
    return audit.PanelStats(
        name,
        2,
        torch.zeros(40000, dtype=torch.float64),
        torch.ones(40000, dtype=torch.float64),
        torch.full((40000,), first_value, dtype=torch.float64),
        torch.full((40000,), 0.25, dtype=torch.float64),
        nonzero,
        nonzero.to(torch.float64).div(2),
        exact,
        first,
        {"count": 0, "mse": 0.0},
    )


def _two_sample_panel() -> audit.Panel:
    images = torch.cat(
        (
            torch.zeros((1, 1, 28, 28), dtype=torch.float32),
            torch.ones((1, 1, 28, 28), dtype=torch.float32),
        ),
        dim=0,
    )
    return audit.Panel("fit", "synthetic", torch.tensor([0, 1]), images, torch.tensor([0, 1]))


def test_exact_rmal_and_rwsa_consumer_locks() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec) == REVIEW
    identity = audit._verify_rwsa_identity(spec)
    assert identity["audit"].get("status") == "invalid_harness"
    assert identity["audit"].get("classification") == "invalid_harness"
    assert audit.RMAL_CONSUMER["probe_sha256"] == "3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed"


def test_stale_hash_rejection_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    with pytest.raises(audit.ZCSAHarnessFailure, match="stale"):
        audit._read_json_hash(path, "0" * 64, "test evidence")


def test_probe_shapes_finite_zero_identity_and_active_telemetry() -> None:
    weights = _synthetic_probe()
    probe = type("Probe", (), {"full_weights": weights["full_weights"], "artifact_sha256": "synthetic"})()
    identity = audit._probe_identity(probe)
    assert identity["W"].shape == (256, 40000)
    assert identity["b"].shape == (40000,)
    assert identity["W"].dtype == torch.float32
    assert identity["b"].dtype == torch.float32
    assert identity["zero_count"] == 6498
    assert identity["active_count"] == 33502
    assert identity["zero_dynamic_entries_nonzero_count"] == 0
    assert identity["active_minimum_mass"] == 1.0
    assert "epsilon" not in inspect.getsource(audit._probe_identity).lower()


def test_nonfinite_w_or_b_is_rejected() -> None:
    weights = _synthetic_probe()["full_weights"].clone()
    weights[0, 6498] = float("nan")
    probe = type("Probe", (), {"full_weights": weights, "artifact_sha256": "synthetic"})()
    with pytest.raises(audit.ZCSAHarnessFailure, match="non-finite"):
        audit._probe_identity(probe)


def test_renderer_is_exact_and_deterministic() -> None:
    result = audit._renderer_identity()
    assert result["source"] == "base.render_intensity_batch"
    assert result["output_shape"] == [200, 200]
    assert result["align_corners"] is False
    assert result["exact_repeat"] is True


def test_panel_contracts_and_candidate_prefix_are_locked() -> None:
    assert audit.FIT_PANEL["count"] == 1000
    assert audit.FIT_PANEL["indices_sha256"] == "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d"
    assert audit.EVALUATION_PANELS["selection"]["source"] == "mnist_test_indices_5000_9999"
    assert audit.EVALUATION_PANELS["selection"]["count"] == 1000
    assert audit.EVALUATION_PANELS["confirmation"]["source"] == "horizon_heldout_train_indices_55000_59999"
    assert audit.EVALUATION_PANELS["confirmation"]["count"] == 5000
    assert audit.EVALUATION_PANELS["candidate_prefix"]["consumed"] is False


def test_panel_target_statistics_use_exact_constant_population_variance_and_counts() -> None:
    stats = audit._panel_stats(_two_sample_panel(), torch.zeros(40000, dtype=torch.bool), torch.zeros(40000))
    assert stats.count == 2
    assert torch.equal(stats.minimum, torch.zeros(40000, dtype=torch.float64))
    assert torch.equal(stats.maximum, torch.ones(40000, dtype=torch.float64))
    assert torch.equal(stats.mean, torch.full((40000,), 0.5, dtype=torch.float64))
    assert torch.equal(stats.variance, torch.full((40000,), 0.25, dtype=torch.float64))
    assert torch.equal(stats.nonzero_count, torch.ones(40000, dtype=torch.int64))
    assert torch.equal(stats.positive_fraction, torch.full((40000,), 0.5, dtype=torch.float64))
    assert not bool(stats.exact_constant.any())


def test_taxonomy_contains_all_fixed_categories() -> None:
    probe = _synthetic_probe(b_nonzero=True)
    fit = _fake_stats("fit")
    fit.exact_constant[1] = True
    fit.first_value[1] = 0.5
    fit.exact_constant[2] = False
    fit.first_value[2] = 0.5
    fit.exact_constant[3] = True
    fit.first_value[3] = 0.0
    categories = audit._taxonomy(probe, fit)
    assert set(categories) == set(audit.TAXONOMY["priority"])
    assert categories["dynamic_active"].sum() == 33502
    assert categories["target_zero_static"][0]
    assert categories["bias_only_constant"][1]
    assert categories["bias_only_variable_target"][2]
    assert categories["zero_W_constant_target_bias_mismatch"][3]
    assert sum(int(mask.sum()) for mask in categories.values()) == 40000


def test_taxonomy_is_mutually_exclusive_and_exhaustive() -> None:
    probe = _synthetic_probe(b_nonzero=True)
    fit = _fake_stats("fit")
    fit.exact_constant[1] = True
    fit.first_value[1] = 0.5
    fit.exact_constant[2] = False
    fit.exact_constant[3] = True
    fit.first_value[3] = 0.0
    categories = audit._taxonomy(probe, fit)
    masks = list(categories.values())
    assert not any(bool((left & right).any()) for index, left in enumerate(masks) for right in masks[index + 1 :])
    union = torch.zeros(40000, dtype=torch.bool)
    for mask in masks:
        union |= mask
    assert union.all()


def test_cross_panel_support_and_zero_to_active_transitions() -> None:
    probe = _synthetic_probe()
    fit = _fake_stats("fit", nonzero_indices=(0,))
    selection = _fake_stats("selection", nonzero_indices=(1,))
    confirmation = _fake_stats("confirmation", nonzero_indices=(2,))
    categories = audit._taxonomy(probe, fit)
    result = audit._cross_panel_semantics(
        categories,
        {"fit": fit, "selection": selection, "confirmation": confirmation},
        probe["zero_mask"],
    )
    assert result["zero_w_intersections"]["zero_w_fit_always_zero"] == 6497
    assert result["zero_w_intersections"]["zero_w_selection_ever_nonzero"] == 1
    assert result["zero_w_intersections"]["zero_w_confirmation_ever_nonzero"] == 1
    assert result["transitions"]["fit_zero_to_selection_active"] == 1
    assert result["transitions"]["fit_zero_to_confirmation_active"] == 1


def test_bias_residual_uses_frozen_b_not_target_mean() -> None:
    probe = _synthetic_probe(b_nonzero=True)
    fit = audit._panel_stats(_two_sample_panel(), probe["zero_mask"], probe["b"])
    selection = _fake_stats("selection")
    confirmation = _fake_stats("confirmation")
    result = audit._bias_semantics(probe, {"fit": fit, "selection": selection, "confirmation": confirmation})
    assert result["b_exact_zero_count"] == 6497
    assert result["b_nonzero_count"] == 1
    assert result["b_minus_fit_target_mean"]["maximum"] == pytest.approx(0.5)
    assert result["panel_bias_only"]["fit"]["mse"] == pytest.approx(0.5)


def test_row_major_mask_edge_definition_components_and_support_overlap() -> None:
    zero = torch.zeros(40000, dtype=torch.bool)
    zero[[0, 1, 200, 202]] = True
    fit = _fake_stats("fit", nonzero_indices=(0, 202))
    selection = _fake_stats("selection", nonzero_indices=(202,))
    confirmation = _fake_stats("confirmation", nonzero_indices=(1,))
    result = audit._spatial_semantics(zero, {"fit": fit, "selection": selection, "confirmation": confirmation})
    assert result["mask_shape"] == [200, 200]
    assert result["bbox"] == {"row_min": 0, "row_max": 1, "column_min": 0, "column_max": 2}
    assert result["edge_count"] == 3
    assert result["interior_count"] == 1
    assert result["connected_components"]["count"] == 2
    assert result["connected_components"]["largest_size"] == 3
    assert result["support_overlap"]["fit"]["zero_w_overlap_count"] == 2
    assert result["support_overlap"]["selection"]["zero_w_overlap_count"] == 1
    assert result["support_overlap"]["confirmation"]["zero_w_overlap_count"] == 1
    assert audit.SPATIAL["edge_definition"] == "outermost one-pixel image boundary"
    assert audit.SPATIAL["morphology"] is None


def test_component_algorithm_is_deterministic_4_connectivity() -> None:
    mask = torch.zeros((200, 200), dtype=torch.bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[1, 0] = True
    mask[1, 1] = True
    mask[3, 3] = True
    first = audit._component_summary(mask)
    second = audit._component_summary(mask)
    assert first == second
    assert first["connectivity"] == "4-connected"
    assert first["count"] == 2
    assert first["largest_size"] == 4


def test_no_optimizer_backward_evaluator_or_optical_replay_contract() -> None:
    source = inspect.getsource(audit)
    assert audit.GATES["optimizer_created"] is False
    assert audit.GATES["backward_calls"] == 0
    assert audit.GATES["evaluator_calls"] == 0
    assert audit.MEASUREMENT["no_optical_forward"] is True
    assert audit.MEASUREMENT["no_cmos_replay"] is True
    assert "_panel_replay" not in source
    assert "_load_scaled_model" not in source


def test_cli_schema_and_lazy_route_are_registered() -> None:
    parser = cli._parser()
    validate = parser.parse_args([
        "research", "optical-manifold",
        "scaled-chen-detector-readout-zero-column-semantics-audit-validate",
        "--config", "config.yaml", "--review-contract", "review.json",
        "--dataset-root", "data", "--output", "validation.json",
    ])
    run = parser.parse_args([
        "research", "optical-manifold",
        "scaled-chen-detector-readout-zero-column-semantics-audit-run",
        "--config", "config.yaml", "--review-contract", "review.json",
        "--validation-evidence", "validation.json", "--dataset-root", "data", "--output", "audit.json",
    ])
    assert validate.research_action.endswith("-validate")
    assert run.research_action.endswith("-run")
    route_source = inspect.getsource(research_router.dispatch)
    assert "optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit" in route_source


def test_descriptive_only_terminal_interpretation_is_locked() -> None:
    assert audit.CLASSIFICATION == "zero_column_semantics_attribution_completed"
    assert audit.NEXT_ACTION_RESULT_REVIEW == "manual_review_ZCSA_result"
    assert audit.AUTHORIZATION["training_authorized"] is False
    assert audit.AUTHORIZATION["formal_promotion"] is False


def test_failed_rwsa_identity_is_invalid_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = audit._load_spec(CONFIG)
    monkeypatch.setitem(audit.RWSA_IDENTITY, "config_sha256", "0" * 64)
    with pytest.raises(audit.ZCSAHarnessFailure, match="RWSA config identity"):
        audit._verify_rwsa_identity(spec)


def test_probe_and_prior_artifacts_are_not_mutated() -> None:
    probe_path = ROOT / audit.RMAL_CONSUMER["probe"]
    before = audit.file_sha256(probe_path)
    weights = _synthetic_probe()
    probe = type("Probe", (), {"full_weights": weights["full_weights"], "artifact_sha256": "synthetic"})()
    audit._probe_identity(probe)
    assert audit.file_sha256(probe_path) == before
