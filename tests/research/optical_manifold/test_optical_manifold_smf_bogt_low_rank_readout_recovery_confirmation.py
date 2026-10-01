from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from ldm_project.v2.application import _cli_research_optical_manifold as lazy_router
from ldm_project.v2.application.cli_schema import _parser
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation as lrrc


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation_v1.yaml"
REVIEW = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation_review_v1.json"


def _operator_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor([[1.0, 2.0, 0.0, -1.0] + [0.0] * 252])
    angle = 0.37
    U = torch.eye(lrrc.FEATURE_DIM, dtype=torch.float64)
    U[0, 0] = torch.cos(torch.tensor(angle, dtype=torch.float64))
    U[1, 0] = torch.sin(torch.tensor(angle, dtype=torch.float64))
    U[0, 1] = -torch.sin(torch.tensor(angle, dtype=torch.float64))
    U[1, 1] = torch.cos(torch.tensor(angle, dtype=torch.float64))
    W = torch.zeros((lrrc.FEATURE_DIM, lrrc.TOTAL_COLUMNS), dtype=torch.float32)
    W[:4, :4] = torch.tensor(
        [[0.3, -0.2, 0.1, 0.4], [0.7, 0.2, -0.4, 0.5], [0.1, 0.8, 0.2, -0.3], [0.2, -0.1, 0.6, 0.9]]
    )
    bias = torch.zeros(lrrc.TOTAL_COLUMNS, dtype=torch.float32)
    bias[:4] = torch.tensor([0.1, -0.2, 0.3, 0.4])
    return features, U, W, bias


def test_config_and_review_are_exactly_locked() -> None:
    spec = lrrc._load_spec(CONFIG)
    lrrc._assert_contract(spec)
    assert lrrc._load_review(REVIEW, spec) == REVIEW
    assert spec.payload["rank_candidates"] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert spec.payload["measurement"]["no_candidate_family_mutation"] is True


def test_upstream_consumer_hashes_and_shapes_are_registered() -> None:
    assert lrrc.RMAL_CONSUMER["probe_sha256"] == "3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed"
    assert lrrc.LSDA_CONSUMER["config_sha256"] == "d5093fc9f26ccf19eaa1bb74c3867fb25023e5837c00e96c5432facd30d9339d"
    assert lrrc.RWSA_ACTIVE_IDENTITY["audit_classification"] == "descriptive_structural_attribution_completed"
    assert lrrc.ZCSA_IDENTITY["audit_classification"] == "zero_column_semantics_attribution_completed"
    assert (lrrc.INPUT_DIM, lrrc.FEATURE_DIM, lrrc.TOTAL_COLUMNS) == (257, 256, 40000)
    assert lrrc.TRUNCATION["spectrum_source"].startswith("LSDA ordered CPU float64")


def test_config_mutation_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["rank_candidates"] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    mutated = tmp_path / CONFIG.name
    mutated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate ranks|rank_candidates"):
        lrrc._assert_contract(lrrc._load_spec(mutated))


def test_candidate_family_and_controls_are_immutable() -> None:
    assert lrrc.RANK_CANDIDATES == (1, 2, 4, 8, 16, 32, 64, 128, 256)
    assert lrrc.SEARCH_RANKS == lrrc.RANK_CANDIDATES[:-1]
    assert lrrc.RANK0 == 0
    assert set(lrrc.CONTROLS) == {"rank_0_bias_only", "rank_256_full_positive_anchor"}
    assert lrrc.EVALUATION["candidate_prefix"]["consumed"] is False


def test_rank0_control_is_bias_only_and_does_not_mutate_bias() -> None:
    features, _, _, bias = _operator_fixture()
    before = bias.clone()
    output = lrrc._rank0_prediction(features, bias)
    assert output.shape == (1, lrrc.TOTAL_COLUMNS)
    assert torch.equal(output[0], bias.to(torch.float64))
    assert torch.equal(bias, before)


def test_explicit_and_factorized_truncation_match_fixture() -> None:
    features, U, W, bias = _operator_fixture()
    explicit = lrrc._explicit_prediction(features, U, W, bias, 2)
    factorized = lrrc._factorized_prediction(features, U, W, bias, 2)
    manual = features.to(torch.float64).mm(U[:, :2].mm(U[:, :2].T)).mm(W.to(torch.float64))
    manual = manual.add(bias.to(torch.float64))
    assert torch.allclose(explicit, manual, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(factorized, explicit, atol=1.0e-12, rtol=1.0e-12)


def test_rank256_operator_identity_replays_full_operator() -> None:
    features, U, W, bias = _operator_fixture()
    identity = lrrc._operator_identity(features, U, W, bias)
    assert identity["explicit_factorized_close"] is True
    assert identity["explicit_factorized_max_abs_residual"] < 1.0e-12


def test_expected_anchor_consumes_the_frozen_cmos_stage_not_final_rmal_stage() -> None:
    compact = {
        "positive_anchor_replay": {"selection": {"probe": {"paired_mse": 0.003}}},
        "stage_results": {
            "cmos_intensity": {
                "selection": {
                    "native_physical_anchor": {
                        "paired_mse": 0.01,
                        "class_template_mse": 0.02,
                        "cyclic_mse": 0.03,
                    },
                    "probe": {"paired_mse": 0.04},
                }
            }
        },
    }
    physical, probe = lrrc._expected_anchor(compact, "selection")
    assert physical == {"paired_mse": 0.01, "class_template_mse": 0.02, "cyclic_mse": 0.03}
    assert probe == pytest.approx(0.04)


def test_invalid_middle_rank_is_rejected() -> None:
    features, U, W, bias = _operator_fixture()
    with pytest.raises(ValueError, match="registered"):
        lrrc._factorized_prediction(features, U, W, bias, 12)


def test_rank_boundary_telemetry_is_descriptive_and_complete() -> None:
    sigma = torch.linspace(256.0, 1.0, lrrc.FEATURE_DIM, dtype=torch.float64)
    squared = sigma.square()
    full = SimpleNamespace(
        singular_values=sigma,
        squared_singular_values=squared,
        cumulative_energy=torch.cumsum(squared, 0) / squared.sum(),
    )
    telemetry = lrrc._rank_boundary_telemetry(16, {"full": full})
    assert telemetry["rank"] == 16
    assert telemetry["sigma_k"] == pytest.approx(float(sigma[15]))
    assert telemetry["sigma_k_plus_1"] == pytest.approx(float(sigma[16]))
    assert telemetry["relative_gap"] == pytest.approx(float((sigma[15] - sigma[16]) / sigma[15]))
    assert telemetry["descriptive_only"] is True
    assert lrrc._rank_boundary_telemetry(256, {"full": full})["sigma_k_plus_1"] is None


def test_registered_gap_telemetry_is_locked_and_not_a_gate() -> None:
    assert lrrc.SPECTRAL_BOUNDARY_GAPS == {
        1: 0.131054483,
        2: 0.178902324,
        4: 0.034630756,
        8: 0.105670493,
        16: 0.104494812,
        32: 0.061605380,
        64: 0.043964196,
        128: 0.052600464,
    }
    assert lrrc.LSDA_SMALLEST_FULL_SPECTRUM_GAP == {"rank": 36, "relative_gap": 0.0025793411}
    assert lrrc.GATE_CONTRACT["no_retained_energy_success_gate"] is True


def test_gate_inherits_physical_template_cyclic_instance_and_bootstrap_checks() -> None:
    report = {
        "probe": {"paired_mse": 0.1},
        "bootstrap": {"ci": [0.01, 0.2]},
        "native_replay": {"paired_mse": True, "class_template_mse": True, "cyclic_mse": True},
        "relative_improvement": 0.1,
        "class_template_mse": 0.2,
        "cyclic_mse": 0.3,
        "instance_sensitive": True,
        "state_unchanged": True,
        "ols_state_unchanged": True,
        "square_law_consistency": True,
        "feature_contract_valid": True,
        "probe_contract_valid": True,
        "finite": True,
    }
    gate = lrrc._panel_gate(report, require_bootstrap=True)
    assert gate["passed"] is True
    assert set(lrrc.INHERITED_GATE_NAMES).issubset(set(gate["checks"]))


def test_bootstrap_wrapper_preserves_rmal_float32_helper_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, torch.dtype] = {}

    def helper(delta: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
        captured["delta"] = delta.dtype
        captured["labels"] = labels.dtype
        return (0.0, 1.0)

    monkeypatch.setattr(lrrc.rmal, "_stratified_bootstrap_ci", helper)
    lrrc._bootstrap_ci(torch.ones(10, dtype=torch.float64), torch.zeros(10, dtype=torch.long))
    assert captured == {"delta": torch.float32, "labels": torch.int64}


def test_sequential_search_selection_failure_skips_confirmation() -> None:
    events: list[tuple[str, int]] = []

    def selection(rank: int) -> dict[str, object]:
        events.append(("selection", rank))
        return {"selection_gate": {"passed": rank == 2}}

    def confirmation(rank: int) -> dict[str, object]:
        events.append(("confirmation", rank))
        return {"selection_gate": {"passed": True}, "confirmation_gate": {"passed": True}}

    result = lrrc._sequential_search(selection, confirmation, anchor_passed=True)
    assert events == [("selection", 1), ("selection", 2), ("confirmation", 2)]
    assert result["evaluated_ranks"] == [1, 2]
    assert result["rank_results"]["1"]["confirmation_not_run"] is True
    assert result["first_confirmed_candidate_rank"] == 2
    assert result["previous_registered_candidate"] == 1


def test_sequential_search_confirmation_failure_continues_and_stops_on_first_double_pass() -> None:
    events: list[tuple[str, int]] = []

    def selection(rank: int) -> dict[str, object]:
        events.append(("selection", rank))
        return {"selection_gate": {"passed": True}}

    def confirmation(rank: int) -> dict[str, object]:
        events.append(("confirmation", rank))
        return {
            "selection_gate": {"passed": True},
            "confirmation_gate": {"passed": rank == 4},
        }

    result = lrrc._sequential_search(selection, confirmation, anchor_passed=True)
    assert result["evaluated_ranks"] == [1, 2, 4]
    assert events[-1] == ("confirmation", 4)
    assert result["rank_results"]["2"]["confirmation_pass"] is False
    assert result["first_confirmed_candidate_rank"] == 4
    assert result["stopped_after_first_double_pass"] is True


def test_sequential_search_all_truncated_candidates_reuses_rank256_anchor() -> None:
    seen: list[int] = []

    def selection(rank: int) -> dict[str, object]:
        seen.append(rank)
        return {"selection_gate": {"passed": False}}

    result = lrrc._sequential_search(selection, lambda rank: {}, anchor_passed=True)
    assert seen == [1, 2, 4, 8, 16, 32, 64, 128]
    assert result["first_confirmed_candidate_rank"] == 256
    assert result["rank_256_reused_as_fallback"] is True
    assert result["previous_registered_candidate"] == 128


def test_anchor_failure_stops_before_any_candidate_search() -> None:
    called = False

    def selection(rank: int) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    result = lrrc._sequential_search(selection, lambda rank: {}, anchor_passed=False)
    assert called is False
    assert result["anchor_failure_stop"] is True
    assert result["first_confirmed_candidate_rank"] is None


def test_run_orders_rank256_anchor_before_sequential_search() -> None:
    source = inspect.getsource(lrrc.run)
    assert source.index("_rank256_anchor(") < source.index("_sequential_search(")
    assert "_rank256_anchor(" in source
    assert "if not anchor[\"passed\"]" in source


def test_source_has_no_update_or_new_decoder_path() -> None:
    source = inspect.getsource(lrrc)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer.step" not in source
    assert "return_trace=True" not in source
    assert "decoder(" not in source
    assert 'payload["cmos_only_replay_executed"] = True' in source
    assert lrrc.AUTHORIZATION["formal_promotion"] is False
    assert lrrc.AUTHORIZATION["training_authorized"] is False
    assert lrrc.TRUNCATION["permanent_W_k_saved"] is False
    assert lrrc.EVALUATION["pooled_feature_bank_saved"] is False
    assert lrrc.EVALUATION["target_bank_saved"] is False


def test_result_language_is_not_mathematical_minimum() -> None:
    source = inspect.getsource(lrrc)
    assert "minimum sufficient" not in source.lower()
    assert "intrinsic rank" not in source.lower()
    assert "first_confirmed_candidate_rank" in source
    assert lrrc.NEXT_ACTION_RESULT_REVIEW == "manual_review_LRRC_result"


def test_write_once_and_invalid_harness_are_explicit() -> None:
    source = inspect.getsource(lrrc)
    assert "atomic_write_new_json" in source
    assert "refusing to reuse LRRC validation attempt" in source
    assert "rank-256 full positive anchor failed" in source
    assert '"invalid_harness"' in source


def test_cli_parses_validate_and_run_actions() -> None:
    validate = _parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-low-rank-readout-recovery-confirmation-validate",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--output",
            "validation.json",
        ]
    )
    run = _parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-low-rank-readout-recovery-confirmation-run",
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
    assert validate.research_action.endswith("validate")
    assert validate.dataset_root == "data"
    assert run.research_action.endswith("run")
    assert run.device == "cuda"


def test_lazy_router_contains_only_registered_lrrc_route() -> None:
    source = inspect.getsource(lazy_router.dispatch)
    assert "scaled-chen-low-rank-readout-recovery-confirmation-validate" in source
    assert "scaled-chen-low-rank-readout-recovery-confirmation-run" in source
    assert "optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation" in source


def test_compact_report_filter_drops_sample_tensors() -> None:
    report = {"rank": 4, "sample_mse": torch.ones(2), "target_first": torch.zeros(1), "finite": True}
    assert lrrc._json_report(report) == {"rank": 4, "finite": True}


def test_recorded_flags_keep_lrrc_diagnostic_only() -> None:
    assert lrrc.MEASUREMENT["mechanism_evidence_only"] is True
    assert lrrc.MEASUREMENT["no_training"] is True
    assert lrrc.MEASUREMENT["no_refit"] is True
    assert lrrc.MEASUREMENT["no_physical_forward_modification"] is True
    assert lrrc.AUTHORIZATION["mechanism_evidence_only"] is True
    assert lrrc.AUTHORIZATION["formal_training_authorized"] is False


def test_review_file_is_valid_json_and_has_no_candidate_mutation() -> None:
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert review["schema"] == lrrc.REVIEW_SCHEMA
    assert review["approved_contract"]["rank_candidates"] == list(lrrc.RANK_CANDIDATES)
    assert review["approved_contract"]["no_rank_family_mutation"] is True
