from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt_detector_readout_structure_attribution_active_domain as audit,
)


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / (
    "configs/research/"
    "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_"
    "active_domain_v1.yaml"
)
REVIEW = ROOT / (
    "configs/research/"
    "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_"
    "active_domain_review_v1.json"
)
RMAL_PROBE = ROOT / audit.RMAL_CONSUMER["probe"]


def _probe() -> audit.ReadoutProbe:
    payload = audit.safe_torch_load(
        RMAL_PROBE,
        expected_sha256=audit.RMAL_CONSUMER["probe_sha256"],
        map_location="cpu",
    )
    indices = payload["fit_indices"].detach().cpu().to(torch.long)
    return audit._load_rmal_probe(
        payload, indices, audit.RMAL_CONSUMER["probe_sha256"]
    )


def _active_mask(count: int = 4) -> torch.Tensor:
    mask = torch.zeros(audit.TOTAL_COLUMNS, dtype=torch.bool)
    mask[:count] = True
    return mask


def _local_weights(active_mask: torch.Tensor) -> torch.Tensor:
    weights = torch.zeros((257, audit.TOTAL_COLUMNS), dtype=torch.float32)
    indices = torch.where(active_mask)[0]
    rows = torch.div(indices, 200, rounding_mode="floor")
    columns = torch.remainder(indices, 200)
    bins = audit._pool_bins()
    membership = audit._bin_membership(200, bins)
    source_rows = membership[:, rows].to(torch.long).argmax(dim=0)
    source_columns = membership[:, columns].to(torch.long).argmax(dim=0)
    source = source_rows * 16 + source_columns
    weights[source, indices] = 1.0
    return weights


def test_config_and_review_are_exactly_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec) == REVIEW
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert review["hypothesis"] == audit.HYPOTHESIS
    assert review["approved_contract"]["active_domain"]["active_count"] == 33502


def test_amendment_lineage_and_parent_are_explicit() -> None:
    assert audit.HYPOTHESIS == "H-LOMA-CBOGT-S5-RWSA-ACTIVE"
    assert audit.LINEAGE["parent_hypothesis"] == "H-LOMA-CBOGT-S5-RWSA"
    assert audit.LINEAGE["old_rwsa_attempt_02_immutable"] is True
    assert audit.LINEAGE["zcsa_passed_evidence_required"] is True
    assert audit.LINEAGE["attempt_name"].endswith("active_domain_v1_attempt_01")


def test_old_rwsa_and_zcsa_identity_bindings_are_immutable() -> None:
    assert audit.RWSA_IDENTITY["audit_status"] == "invalid_harness"
    assert audit.RWSA_IDENTITY["result_git_commit"] == (
        "0dead1b73655b3556450cedbb5d0a2c579385209"
    )
    assert audit.ZCSA_IDENTITY["audit_status"] == "passed"
    assert audit.ZCSA_IDENTITY["audit_classification"] == (
        "zero_column_semantics_attribution_completed"
    )
    assert audit.ZCSA_IDENTITY["compact_report_sha256"] == (
        "772fb739b39d029c213ab42e8f552648782607b5562b6350d4d2c9744b73b9e6"
    )


def test_frozen_probe_has_registered_active_and_inactive_counts() -> None:
    identity = audit._probe_identity(_probe())
    assert identity["active_count"] == 33502
    assert identity["inactive_count"] == 6498
    assert identity["zero_dynamic_entries_nonzero_count"] == 0
    assert identity["zero_bias_entries_nonzero_count"] == 0
    assert identity["active_minimum_mass"] > 0


def test_active_normalization_has_no_epsilon_and_excludes_inactive_columns() -> None:
    weights = torch.zeros((257, audit.TOTAL_COLUMNS), dtype=torch.float32)
    weights[0, 0] = 2.0
    weights[1, 1] = -3.0
    mask = _active_mask(2)
    probabilities, mass = audit._normalize_active_abs_columns(weights, mask)
    assert probabilities.shape == (2, 256)
    assert mass.tolist() == [2.0, 3.0]
    assert torch.equal(probabilities.sum(dim=1), torch.ones(2, dtype=torch.float64))
    source = inspect.getsource(audit._normalize_active_abs_columns).lower()
    assert "eps" not in source
    assert "active_weights = w[:, active_mask]" in source


def test_nonpositive_active_mass_fails_closed() -> None:
    with pytest.raises(audit.RWSAActiveHarnessFailure, match="non-positive"):
        audit._normalize_active_abs_columns(
            torch.zeros((257, audit.TOTAL_COLUMNS), dtype=torch.float32),
            _active_mask(2),
        )


def test_nonfinite_active_mass_fails_closed() -> None:
    weights = torch.zeros((257, audit.TOTAL_COLUMNS), dtype=torch.float32)
    weights[0, 0] = float("nan")
    with pytest.raises(audit.RWSAActiveHarnessFailure, match="non-finite"):
        audit._normalize_active_abs_columns(weights, _active_mask(2))


def test_permutation_null_preserves_the_same_active_mask() -> None:
    mask = _active_mask(2)
    probabilities = torch.zeros((2, 256), dtype=torch.float64)
    probabilities[0, 0] = 1.0
    probabilities[1, 1] = 1.0
    denominator = torch.ones(2, dtype=torch.float64)
    support_indices = torch.tensor([[0, 1, 16, 17], [0, 1, 16, 17]])
    support_mask = torch.ones_like(support_indices, dtype=torch.bool)
    result = audit._permutation_null_summary(
        probabilities,
        denominator,
        support_indices,
        support_mask,
        mask,
        1.0,
    )
    assert result["domain"] == "dynamic_active"
    assert result["conditional_on_dynamic_active_domain"] is True
    assert result["active_mask_preserved"] is True
    assert result["active_mask_sha256"] == audit.tensor_sha256(mask)
    assert result["count"] == 256


def test_structural_metrics_are_conditional_and_have_required_families() -> None:
    weights = _local_weights(torch.ones(audit.TOTAL_COLUMNS, dtype=torch.bool))
    report = audit._active_structural_metrics(
        weights,
        weights[-1],
        torch.ones(audit.TOTAL_COLUMNS, dtype=torch.bool),
    )
    assert report["domain"] == "dynamic_active"
    assert report["conditional_on_dynamic_active_domain"] is True
    assert report["L_active"] > 0.95
    for key in (
        "argmax_source_cell",
        "centroid",
        "displacement",
        "top_1_mass",
        "top_k_mass",
        "expected_support_mass",
        "coarse_radius_locality_mass",
        "spatial_variance_m2",
        "rms_support_radius_um",
        "effective_support",
        "normalized_entropy",
        "row_concentration",
        "column_concentration",
        "d4_coordinate_consistency",
        "translation_field",
        "locality_vs_target_position",
        "positive_mass",
        "negative_mass",
        "cancellation",
        "permutation_null",
    ):
        assert key in report


def test_structural_metric_domain_is_not_extrapolated_to_inactive_columns() -> None:
    weights = torch.zeros((257, audit.TOTAL_COLUMNS), dtype=torch.float32)
    weights[:-1].fill_(1.0)
    active = torch.ones(audit.TOTAL_COLUMNS, dtype=torch.bool)
    report = audit._active_structural_metrics(weights, weights[-1], active)
    assert report["active_column_count"] == audit.TOTAL_COLUMNS
    assert report["inactive_output_columns_excluded"] == 0
    assert report["permutation_null"]["domain"] == "dynamic_active"


def test_adaptive_pooling_and_d4_contract_are_inherited_exactly() -> None:
    assert audit._pool_bins()[0] == (0, 13)
    assert audit._pool_bins()[1] == (12, 25)
    assert audit.D4_CONTROLS == (
        "identity",
        "x_flip",
        "y_flip",
        "xy_flip",
        "transpose",
        "transpose_x_flip",
        "transpose_y_flip",
        "transpose_xy_flip",
    )
    rows = torch.tensor([2, 5])
    columns = torch.tensor([7, 9])
    assert torch.equal(
        audit._transform_coordinates(rows, columns, "identity")[0], rows
    )
    assert audit._transform_coordinates(rows, columns, "x_flip")[1].tolist() == [
        192,
        190,
    ]


def test_pixelwise_decomposition_retains_all_active_and_zero_w_domains() -> None:
    target = torch.tensor([[1.0, 2.0, 0.0, 1.0], [2.0, 1.0, 1.0, 0.0]])
    physical = torch.zeros_like(target)
    full = target.clone()
    dynamic = torch.zeros_like(target)
    bias = torch.zeros_like(target)
    domains = {
        "all_40000": torch.ones(4, dtype=torch.bool),
        "dynamic_active": torch.tensor([True, True, False, False]),
        "zero_W": torch.tensor([False, False, True, True]),
    }
    result = audit._pixelwise_decomposition(
        physical, full, dynamic, bias, target, domains
    )
    assert set(result) == {"all_40000", "dynamic_active", "zero_W"}
    assert result["dynamic_active"]["domain"] == "dynamic_active"
    assert result["dynamic_active"]["conditional_on_dynamic_active_domain"] is True
    assert result["zero_W"]["column_count"] == 2
    assert result["zero_W"]["masked_class_template_metrics"].startswith(
        "not_evaluated"
    )
    assert result["all_40000"]["full_mse"] == pytest.approx(0.0)


def test_active_domain_transition_and_decomposition_contract_is_explicit() -> None:
    assert audit.BIAS_ATTRIBUTION["decomposition_domains"] == [
        "all_40000",
        "dynamic_active",
        "zero_W",
    ]
    assert (
        audit.BIAS_ATTRIBUTION["masked_class_template_metrics"]
        == "not_evaluated_without_unambiguous_contract"
    )
    assert audit.ACTIVE_DOMAIN["inactive_columns_retained_in_evidence"] is True
    assert audit.ACTIVE_DOMAIN["epsilon_normalization"] is False


def test_no_update_and_replay_restrictions_are_visible() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "evaluator_calls" in source
    assert "checkpoint_written" in source
    assert audit.MEASUREMENT["exact_panel_cmos_only_replay"] is True
    assert audit.MEASUREMENT["no_large_feature_bank_artifact"] is True


def test_physical_anchor_and_readiness_boundaries_are_preserved() -> None:
    assert audit.PHYSICAL_ANCHORS["selection"]["paired_mse"] == pytest.approx(
        0.06204846128821373
    )
    assert audit.PHYSICAL_ANCHORS["confirmation"]["paired_mse"] == pytest.approx(
        0.05778951570391655
    )
    assert audit.CLASSIFICATION == "descriptive_structural_attribution_completed"
    assert audit.NEXT_ACTION_RESULT_REVIEW == "manual_review_RWSA_active_domain_result"
    assert audit.AUTHORIZATION["training_authorized"] is False
    assert audit.AUTHORIZATION["formal_promotion"] is False


def test_cli_actions_are_registered_and_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-detector-readout-structure-active-domain-validate",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--dataset-root",
            "data",
            "--output",
            "validation.json",
        ]
    )
    run = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-detector-readout-structure-active-domain-run",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--validation-evidence",
            "validation.json",
            "--dataset-root",
            "data",
            "--output",
            "audit.json",
        ]
    )
    assert validate.research_action.endswith("active-domain-validate")
    assert run.device == "cuda"
    router = (
        ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py"
    ).read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_detector_readout_structure_attribution_active_domain" in router
    assert (
        "import optical_manifold_smf_bogt_detector_readout_structure_attribution_active_domain"
        not in router
    )
