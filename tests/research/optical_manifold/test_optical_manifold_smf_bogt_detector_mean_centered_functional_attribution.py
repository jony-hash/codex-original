import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from ldm_project.v2.application import _cli_research_optical_manifold as lazy_router
from ldm_project.v2.application import cli
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_detector_mean_centered_functional_attribution as mcfa


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / (
    "configs/research/optical_manifold/"
    "lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution_v1.yaml"
)
REVIEW = ROOT / (
    "configs/research/optical_manifold/"
    "lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution_review_v1.json"
)


def test_config_review_and_contract_are_exactly_locked() -> None:
    spec = mcfa._load_spec(CONFIG)
    mcfa._assert_contract(spec)
    assert mcfa._load_review(REVIEW, spec) == REVIEW
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert review["hypothesis"] == mcfa.HYPOTHESIS
    assert review["approved_contract"] == mcfa._contract_payload()
    assert review["descriptive_only"] is True
    assert review["training_authorized"] is False


def test_lineage_and_frozen_consumers_are_registered() -> None:
    assert mcfa.LINEAGE["lineage_start_head"] == (
        "3f1c01859a238986b263c0b040d21f96f531cb52"
    )
    assert mcfa.RMAL_CONSUMER["probe_sha256"] == (
        "3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed"
    )
    assert mcfa.DSFA_CONSUMER["classification"] == (
        "data_weighted_spectral_functional_attribution_completed"
    )
    assert mcfa.DMSA_CONSUMER["classification"] == (
        "detector_data_metric_spectral_attribution_completed"
    )
    assert mcfa.LRRC_CONSUMER["first_confirmed_candidate_rank"] == 256


def test_panel_roles_counts_and_attribution_boundary_are_locked() -> None:
    assert tuple(list(mcfa.PANEL_CONTRACT)[:3]) == ("fit", "selection", "confirmation")
    assert mcfa.PANEL_CONTRACT["fit"]["role"] == "probe_fit_origin_reference"
    assert mcfa.PANEL_CONTRACT["fit"]["count"] == 1000
    assert mcfa.PANEL_CONTRACT["selection"]["role"] == "independent_registered_selection"
    assert mcfa.PANEL_CONTRACT["selection"]["count"] == 1000
    assert mcfa.PANEL_CONTRACT["confirmation"]["role"] == "independent_registered_confirmation"
    assert mcfa.PANEL_CONTRACT["confirmation"]["count"] == 5000
    assert mcfa.PANEL_CONTRACT["selection_confirmation_independent"] is True
    assert mcfa.PANEL_CONTRACT["fit_is_reference_only"] is True
    assert mcfa.PANEL_CONTRACT["panel_mean_is_attribution_only"] is True
    assert mcfa.PANEL_CONTRACT["deploy_time_mean_estimator"] is False


def test_mean_accumulator_uses_population_mean_and_cpu_float64() -> None:
    values = torch.tensor([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]], dtype=torch.float32)
    accumulator = mcfa.MeanAccumulator(feature_dim=2)
    accumulator.update(values[:1])
    accumulator.update(values[1:])
    result = accumulator.finalize("fixture")
    assert result.count == 3
    assert result.mu.dtype == torch.float64
    assert result.mu.device.type == "cpu"
    assert torch.allclose(result.mu, values.to(torch.float64).mean(0))
    assert mcfa.NUMERICAL_CONTRACT["population_denominator"] == "N"
    assert mcfa.NUMERICAL_CONTRACT["no_N_minus_1"] is True


def test_mean_accumulator_rejects_empty_wrong_shape_and_nonfinite_batches() -> None:
    accumulator = mcfa.MeanAccumulator(feature_dim=2)
    with pytest.raises(ValueError, match="shape"):
        accumulator.update(torch.zeros((2, 3)))
    with pytest.raises(ValueError, match="empty"):
        accumulator.update(torch.zeros((0, 2)))
    with pytest.raises(mcfa.MCFAHarnessFailure, match="non-finite"):
        accumulator.update(torch.tensor([[float("nan"), 0.0]]))
    with pytest.raises(mcfa.MCFAHarnessFailure, match="no features"):
        mcfa.MeanAccumulator(feature_dim=2).finalize("empty")


def test_two_pass_contract_freezes_mean_and_has_registered_visit_count() -> None:
    assert mcfa.TWO_PASS_CONTRACT["pass_order"] == [
        "pass_1_mean",
        "pass_2_functional",
    ]
    assert mcfa.TWO_PASS_CONTRACT["total_sample_visits"] == 14000
    assert mcfa.TWO_PASS_CONTRACT["pass_2"]["mu_mutated"] is False
    assert mcfa.TWO_PASS_CONTRACT["pass_2"]["centered_feature_bank_saved"] is False
    source = inspect.getsource(mcfa._replay_pass2)
    assert source.index("mu") < source.index("accumulator.update")
    assert "mean_result" not in source


def test_panel_means_are_independent_values_not_a_shared_global() -> None:
    values_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    values_b = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    first = mcfa.MeanAccumulator(feature_dim=2)
    second = mcfa.MeanAccumulator(feature_dim=2)
    first.update(values_a)
    second.update(values_b)
    mean_a = first.finalize("selection")
    mean_b = second.finalize("confirmation")
    assert not torch.equal(mean_a.mu, mean_b.mu)
    assert mean_a.panel == "selection"
    assert mean_b.panel == "confirmation"


def _fixture_functional_report() -> dict[str, object]:
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, 1.0, 4.0], [4.0, 3.0, 2.0]], dtype=torch.float32)
    mu = features.to(torch.float64).mean(0)
    W64 = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]], dtype=torch.float64)
    b64 = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float64)
    mean_response = mu.unsqueeze(0).mm(W64).reshape(3)
    accumulator = mcfa.FunctionalAccumulator(feature_dim=2, output_dim=3)
    accumulator.update(features, target, mu, W64, b64, mean_response)
    return accumulator.finalize("fixture", mu, W64, mean_response)


def test_functional_accumulator_checks_dynamic_decomposition_and_centered_means() -> None:
    report = _fixture_functional_report()
    assert report["finite"] is True
    assert report["dynamic_decomposition_max_abs_residual"] == pytest.approx(0.0)
    assert report["dynamic_decomposition_passed"] is True
    assert report["centered_feature_mean_norm"] == pytest.approx(0.0)
    assert report["centered_dynamic_mean_norm"] == pytest.approx(0.0)
    assert report["centered_feature_mean_passed"] is True
    assert report["centered_dynamic_mean_passed"] is True
    assert report["mean_centered_output_cross"] == pytest.approx(0.0)
    assert report["mean_centered_output_cross_passed"] is True


def test_output_energy_identity_and_dmsa_replay_shape() -> None:
    report = _fixture_functional_report()
    energy = {
        "E_total": report["E_total"],
        "E_mean": report["E_mean"],
        "E_centered": report["E_centered"],
    }
    dmsa_energy = {
        "E_total": report["E_total"],
        "E_mean": report["E_mean"],
        "E_var": report["E_centered"],
        "mean_fraction": report["E_mean"] / report["E_total"],
        "variance_fraction": report["E_centered"] / report["E_total"],
    }
    output, replay = mcfa._energy_report(energy, dmsa_energy)
    assert output["identity_passed"] is True
    assert output["mean_fraction"] + output["centered_fraction"] == pytest.approx(1.0)
    assert replay["passed"] is True


def test_functional_alignment_penalty_and_mse_identities_are_recorded() -> None:
    report = _fixture_functional_report()
    assert report["mean_functional_gain"] == pytest.approx(
        report["mean_alignment_benefit"] - report["mean_self_penalty"]
    )
    assert report["centered_functional_gain"] == pytest.approx(
        report["centered_alignment_benefit"] - report["centered_self_penalty"]
    )
    assert report["total_functional_gain"] == pytest.approx(
        report["mean_functional_gain"] + report["centered_functional_gain"]
    )
    mse = mcfa._mse_report(report)
    assert mse["bias_plus_mean_attribution_only"] is True
    assert mse["mean_gain_identity"]["passed"] is True
    assert mse["centered_gain_identity"]["passed"] is True
    assert mse["total_gain_identity"]["passed"] is True


def test_signed_gain_shares_preserve_negative_and_greater_than_one_values() -> None:
    shares = mcfa._gain_shares(2.0, -1.0, 1.0)
    assert shares["mean_gain_share"] == pytest.approx(2.0)
    assert shares["centered_gain_share"] == pytest.approx(-1.0)
    assert shares["share_sum"] == pytest.approx(1.0)
    assert shares["share_identity_passed"] is True


def test_zero_or_nonfinite_total_gain_is_not_applicable_without_epsilon() -> None:
    for total in (0.0, float("nan"), float("inf")):
        shares = mcfa._gain_shares(1.0, -1.0, total)
        assert shares["mean_gain_share"] == "not_applicable"
        assert shares["centered_gain_share"] == "not_applicable"
        assert shares["share_identity_passed"] is True
    assert mcfa.FUNCTIONAL_CONTRACT["no_abs_clamp_epsilon_or_renormalization"] is True


def test_mse_cross_evidence_identity_is_not_evaluated_for_fit() -> None:
    report = _fixture_functional_report()
    result = mcfa._dsfa_replay(report, None)
    assert result["reference"].startswith("not_evaluated_fit")
    assert result["passed"] is True


def test_forbidden_paths_and_state_controls_are_explicitly_disabled() -> None:
    for name in (
        "no_training",
        "no_optimizer",
        "no_backward",
        "no_refit",
        "no_decoder",
        "no_whitening",
        "no_centering_deployment",
        "no_weighted_truncation",
        "no_rank_search",
        "no_generation",
    ):
        assert mcfa.MEASUREMENT[name] is True
    assert mcfa.MEASUREMENT["descriptive_only"] is True
    assert mcfa.AUTHORIZATION["formal_promotion"] is False
    assert mcfa.AUTHORIZATION["training_authorized"] is False
    source = inspect.getsource(mcfa.run)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "torch.linalg.eigh" not in source
    assert "rank_search" not in source


def test_no_banks_and_no_physical_forward_controls_are_recorded() -> None:
    for name in (
        "feature_bank_saved",
        "centered_feature_bank_saved",
        "target_bank_saved",
        "prediction_bank_saved",
        "raw_cmos_bank_saved",
        "internal_stage_capture",
        "optical_replay_executed",
    ):
        assert mcfa.PANEL_CONTRACT.get(name, False) is False
    assert mcfa.MEASUREMENT["no_physical_forward_modification"] is True


def test_write_once_validation_slot_and_run_slot_contract(tmp_path: Path) -> None:
    validation = tmp_path / "attempt" / "validation.json"
    mcfa._prepare_validation_slot(validation)
    with pytest.raises(FileExistsError):
        mcfa._prepare_validation_slot(validation)
    validation.write_text("{}", encoding="utf-8")
    audit = validation.with_name("audit.json")
    assert mcfa._check_run_slot(audit, validation) is None
    audit.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        mcfa._check_run_slot(audit, validation)


def test_cli_parses_mcfa_validate_and_run_actions() -> None:
    parser = cli._parser()
    validate = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-detector-mean-centered-functional-attribution-validate",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--output",
            "validation.json",
        ]
    )
    run = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-detector-mean-centered-functional-attribution-run",
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


def test_lazy_router_contains_only_the_registered_mcfa_route() -> None:
    source = inspect.getsource(lazy_router.dispatch)
    assert "scaled-chen-detector-mean-centered-functional-attribution-validate" in source
    assert "scaled-chen-detector-mean-centered-functional-attribution-run" in source
    assert "optical_manifold_smf_bogt_detector_mean_centered_functional_attribution" in source


def test_fit_is_reference_only_and_next_action_is_manual_review() -> None:
    assert mcfa.PANEL_CONTRACT["fit"]["role"] == "probe_fit_origin_reference"
    assert mcfa.PANEL_CONTRACT["fit_is_reference_only"] is True
    assert mcfa.NEXT_ACTION_RESULT_REVIEW == "manual_review_MCFA_result"
    assert mcfa.NEXT_ACTION_RUN == "run_scaled_chen_detector_mean_centered_functional_attribution"


def test_config_mutation_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["measurement"]["no_training"] = False
    mutated = tmp_path / CONFIG.name
    mutated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="contract is not locked"):
        mcfa._assert_contract(mcfa._load_spec(mutated))


def test_review_mutation_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(REVIEW.read_text(encoding="utf-8"))
    payload["training_authorized"] = True
    mutated = tmp_path / REVIEW.name
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    spec = mcfa._load_spec(CONFIG)
    with pytest.raises(ValueError, match="review contract"):
        mcfa._load_review(mutated, spec)
