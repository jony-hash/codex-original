from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from ldm_project.v2.application import cli
from ldm_project.v2.application import _cli_research_optical_manifold as lazy_router
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt_detector_data_metric_spectral_attribution as dmsa,
)


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / (
    "configs/research/"
    "lightgen_optical_manifold_smf_bogt_detector_data_metric_spectral_attribution_v1.yaml"
)
REVIEW = ROOT / (
    "configs/research/"
    "lightgen_optical_manifold_smf_bogt_detector_data_metric_spectral_attribution_review_v1.json"
)


def test_config_review_and_contract_are_exactly_locked() -> None:
    spec = dmsa._load_spec(CONFIG)
    dmsa._assert_contract(spec)
    assert dmsa._load_review(REVIEW, spec) == REVIEW
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert review["hypothesis"] == dmsa.HYPOTHESIS
    assert review["descriptive_only"] is True
    assert review["training_authorized"] is False
    assert review["approved_contract"]["fit_panel"] == dmsa.FIT_PANEL


def test_lineage_and_all_frozen_consumers_are_registered() -> None:
    assert dmsa.LINEAGE["lineage_start_head"] == (
        "39b0ac449986df31b9102c0b1296dda912f7b39c"
    )
    assert dmsa.RMAL_CONSUMER["probe_sha256"] == (
        "3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed"
    )
    assert (dmsa.FEATURE_DIM, dmsa.TOTAL_COLUMNS) == (256, 40000)
    assert dmsa.LSDA_CONSUMER["classification"] == (
        "linear_readout_spectral_attribution_completed"
    )
    assert dmsa.LRRC_CONSUMER["first_confirmed_candidate_rank"] == 256
    assert dmsa.DSFA_CONSUMER["classification"] == (
        "data_weighted_spectral_functional_attribution_completed"
    )


def test_panel_contract_keeps_fit_reference_and_evaluation_panels_separate() -> None:
    assert tuple(list(dmsa.PANEL_CONTRACT)[:3]) == (
        "fit",
        "selection",
        "confirmation",
    )
    assert dmsa.FIT_PANEL["indices_sha256"] == (
        "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d"
    )
    assert dmsa.PANEL_CONTRACT["fit_is_reference_only"] is True
    assert dmsa.PANEL_CONTRACT["selection"]["count"] == 1000
    assert dmsa.PANEL_CONTRACT["confirmation"]["count"] == 5000
    assert dmsa.PANEL_CONTRACT["selection_confirmation_independent"] is True
    assert dmsa.PANEL_CONTRACT["pooled_feature_bank_saved"] is False
    assert dmsa.PANEL_CONTRACT["target_bank_saved"] is False
    assert dmsa.PANEL_CONTRACT["raw_cmos_bank_saved"] is False
    assert dmsa.PANEL_CONTRACT["internal_stage_capture"] is False


def test_config_mutation_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["measurement"]["no_rank_search"] = False
    mutated = tmp_path / CONFIG.name
    mutated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="contract is not locked"):
        dmsa._assert_contract(dmsa._load_spec(mutated))


def test_review_mutation_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(REVIEW.read_text(encoding="utf-8"))
    payload["training_authorized"] = True
    mutated = tmp_path / REVIEW.name
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    spec = dmsa._load_spec(CONFIG)
    with pytest.raises(ValueError, match="review contract"):
        dmsa._load_review(mutated, spec)


def test_moment_accumulator_uses_population_denominator_and_chan_merge() -> None:
    values = torch.tensor(
        [[1.0, -2.0, 0.5], [2.0, 1.0, -1.0], [-3.0, 0.25, 2.0], [4.0, 3.0, 1.5]],
        dtype=torch.float32,
    )
    accumulator = dmsa.MomentAccumulator(feature_dim=3)
    accumulator.update(values[:1])
    accumulator.update(values[1:3])
    accumulator.update(values[3:])
    result = accumulator.finalize("fixture")
    values64 = values.to(torch.float64)
    mean = values64.mean(dim=0)
    second = values64.T.mm(values64) / values64.shape[0]
    centered = values64 - mean
    covariance = centered.T.mm(centered) / values64.shape[0]
    assert result.count == values.shape[0]
    assert torch.allclose(result.mu, mean, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(result.second, second, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(result.centered, covariance, atol=1.0e-12, rtol=1.0e-12)
    assert result.identity["passed"] is True
    assert dmsa.MOMENT_CONTRACT["population_denominator"] == "N"
    assert dmsa.MOMENT_CONTRACT["unbiased_N_minus_1"] is False


def test_moment_accumulator_rejects_empty_or_wrong_shape_batches() -> None:
    accumulator = dmsa.MomentAccumulator(feature_dim=3)
    with pytest.raises(ValueError, match="wrong shape"):
        accumulator.update(torch.zeros((2, 4)))
    with pytest.raises(ValueError, match="empty"):
        accumulator.update(torch.zeros((0, 3)))
    with pytest.raises(dmsa.DMSAHarnessFailure, match="no features"):
        accumulator.finalize("empty")


def test_moment_identity_and_symmetry_telemetry_are_recorded_before_symmetrization() -> None:
    second = torch.eye(3, dtype=torch.float64)
    centered = torch.diag(torch.tensor([0.5, 0.75, 0.25], dtype=torch.float64))
    mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    identity = dmsa._moment_identity(second, centered, mu)
    assert identity["passed"] is False
    matrix = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    symmetry = dmsa._matrix_symmetry(matrix)
    assert symmetry["max_abs_antisymmetry"] == pytest.approx(0.0)
    assert symmetry["frobenius_antisymmetry_residual"] == pytest.approx(0.0)


def test_energy_decomposition_matches_mean_plus_centered_variance() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]], dtype=torch.float64)
    accumulator = dmsa.MomentAccumulator(feature_dim=2)
    accumulator.update(values)
    result = accumulator.finalize("fixture")
    gram = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    decomposition = dmsa._energy_decomposition(result, gram)
    assert decomposition["identity_passed"] is True
    assert decomposition["identity_residual"] == pytest.approx(0.0, abs=1.0e-12)
    assert decomposition["E_total"] == pytest.approx(
        decomposition["E_mean"] + decomposition["E_var"], abs=1.0e-12
    )


def test_psd_square_root_clamps_only_tiny_negative_eigenvalues() -> None:
    tiny = torch.diag(torch.tensor([-1.0e-14, 1.0, 4.0], dtype=torch.float64))
    root, telemetry = dmsa._psd_square_root(tiny, "tiny")
    assert root.dtype == torch.float64
    assert telemetry["tiny_negative_count"] == 1
    assert telemetry["rank_cutoff_used"] is False
    assert telemetry["square_root_passed"] is True

    material = torch.diag(torch.tensor([-1.0e-3, 1.0, 4.0], dtype=torch.float64))
    with pytest.raises(dmsa.DMSAHarnessFailure, match="material negative"):
        dmsa._psd_square_root(material, "material")


def test_off_diagonal_telemetry_preserves_frobenius_identity() -> None:
    matrix = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    telemetry = dmsa._off_diagonal(matrix)
    assert telemetry["total_frobenius_squared"] == pytest.approx(15.0)
    assert telemetry["diagonal_frobenius_squared"] == pytest.approx(13.0)
    assert telemetry["offdiagonal_frobenius_squared"] == pytest.approx(2.0)
    assert telemetry["diagonal_frobenius_squared"] + telemetry[
        "offdiagonal_frobenius_squared"
    ] == pytest.approx(telemetry["total_frobenius_squared"])


def test_spectrum_report_exposes_registered_landmarks_without_recomputing_modes() -> None:
    class Summary:
        cumulative_energy = torch.linspace(0.01, 1.0, dmsa.FEATURE_DIM, dtype=torch.float64)

        @staticmethod
        def compact() -> dict[str, object]:
            return {"sentinel": True}

    report = dmsa._spectrum_report(Summary())
    assert report["sentinel"] is True
    assert set(report["cumulative_energy_landmarks"]) == {str(rank) for rank in dmsa.LANDMARKS}
    assert report["cumulative_energy_landmarks"]["256"] == pytest.approx(1.0)


def test_selection_confirmation_spectrum_comparison_is_descriptive_only() -> None:
    def spectrum(offset: float) -> dict[str, object]:
        return {
            "effective_rank_entropy": 10.0 + offset,
            "stable_rank": 4.0 + offset,
            "cumulative_energy_landmarks": {str(rank): 0.5 + offset for rank in dmsa.LANDMARKS},
            "sigma_over_sigma1": [1.0 + offset] * dmsa.FEATURE_DIM,
            "singular_values": [1.0 + offset] * dmsa.FEATURE_DIM,
            "cumulative_energy": [1.0 + offset] * dmsa.FEATURE_DIM,
        }

    selection = {
        "raw_spectrum": spectrum(0.0),
        "second_moment_spectrum": spectrum(0.0),
        "centered_variance_spectrum": spectrum(0.0),
    }
    confirmation = {
        "raw_spectrum": spectrum(0.0),
        "second_moment_spectrum": spectrum(0.5),
        "centered_variance_spectrum": spectrum(1.0),
    }
    comparison = dmsa._spectrum_comparison(selection, confirmation)
    assert comparison["second_moment_spectrum"][
        "absolute_effective_rank_difference"
    ] == pytest.approx(0.5)
    assert comparison["centered_variance_spectrum"][
        "absolute_stable_rank_difference"
    ] == pytest.approx(1.0)
    assert comparison["raw_panel_identity"]["singular_values_equal"] is True
    assert comparison["panel_agreement_gate"] is False


def test_cross_identity_fixture_decomposes_second_moment_into_centered_and_mean() -> None:
    U = torch.eye(3, dtype=torch.float64)
    mu = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    centered = torch.diag(torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    second = centered + torch.outer(mu, mu)
    result = dmsa._dsfa_cross_identity("fit", U, second, centered, mu, None)
    assert result["modewise_decomposition_passed"] is True
    assert result["dsfa_identity_passed"] is True
    assert result["dsfa_reference"].startswith("not_evaluated")


def test_fraction_and_zero_denominator_semantics_are_explicit() -> None:
    assert dmsa._fraction(2.0, 4.0) == pytest.approx(0.5)
    assert dmsa._fraction(2.0, 0.0) is None
    assert dmsa._fraction(float("nan"), 1.0) is None
    assert dmsa.OFF_DIAGONAL_CONTRACT["zero_denominator"] == "undefined"
    assert dmsa.DECOMPOSITION_CONTRACT["no_scientific_dominance_gate"] is True


def test_source_has_no_training_refit_rank_search_or_persistent_feature_bank_path() -> None:
    source = inspect.getsource(dmsa)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer.step" not in source
    assert "model.train(" not in source
    assert "return_trace=True" not in source
    assert "torch.cov" not in source
    assert "40000 * 40000" not in source
    assert dmsa.MEASUREMENT["no_training"] is True
    assert dmsa.MEASUREMENT["no_refit"] is True
    assert dmsa.MEASUREMENT["no_rank_search"] is True
    assert dmsa.MEASUREMENT["no_rank_truncated_reconstruction"] is True
    assert dmsa.SPECTRAL_REPRESENTATION["rank_truncation"] is False
    assert dmsa.RESOURCE_CONTRACT["feature_bank_saved"] is False
    assert dmsa.RESOURCE_CONTRACT["large_target_gram"] is False


def test_run_computes_final_terminal_status_before_write_once_compact_slot() -> None:
    source = inspect.getsource(dmsa.run)
    assert source.index("audit_gates_without_resource") < source.index(
        "_write_new(compact_path"
    )
    assert source.count("_write_new(compact_path") == 1
    assert "compact slot is write-once" not in source


def test_write_once_validation_slot_and_run_slot_contract(tmp_path: Path) -> None:
    validation = tmp_path / "attempt" / "validation.json"
    dmsa._prepare_validation_slot(validation)
    with pytest.raises(FileExistsError):
        dmsa._prepare_validation_slot(validation)
    validation.write_text("{}", encoding="utf-8")
    audit = validation.with_name("audit.json")
    assert dmsa._check_run_slot(audit, validation) is None


def test_cli_parses_dmsa_validate_and_run_actions() -> None:
    validate = cli._parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-detector-data-metric-spectral-attribution-validate",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--output",
            "validation.json",
        ]
    )
    run = cli._parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-detector-data-metric-spectral-attribution-run",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--validation-evidence",
            "validation.json",
            "--output",
            "audit.json",
        ]
    )
    assert validate.research_action.endswith("-validate")
    assert validate.dataset_root == "data"
    assert run.research_action.endswith("-run")
    assert run.device == "cuda"


def test_lazy_router_contains_only_the_registered_dmsa_route() -> None:
    source = inspect.getsource(lazy_router.dispatch)
    assert "scaled-chen-detector-data-metric-spectral-attribution-validate" in source
    assert "scaled-chen-detector-data-metric-spectral-attribution-run" in source
    assert (
        "optical_manifold_smf_bogt_detector_data_metric_spectral_attribution"
        in source
    )


def test_authorization_and_next_action_remain_descriptive_only() -> None:
    assert dmsa.AUTHORIZATION["descriptive_only"] is True
    assert dmsa.AUTHORIZATION["formal_promotion"] is False
    assert dmsa.AUTHORIZATION["training_authorized"] is False
    assert dmsa.AUTHORIZATION["formal_training_authorized"] is False
    assert dmsa.AUTHORIZATION["mechanism_evidence_only"] is True
    assert dmsa.NEXT_ACTION_RESULT_REVIEW == "manual_review_DMSA_result"
