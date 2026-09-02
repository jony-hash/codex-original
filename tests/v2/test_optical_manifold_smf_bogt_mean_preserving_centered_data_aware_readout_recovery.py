from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from ldm_project.v2.application import _cli_research_optical_manifold as lazy_router
from ldm_project.v2.application.cli_schema import _parser
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery as mpcdar,
)


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery_v1.yaml"
REVIEW = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery_review_v1.json"


def _full_row_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    W = torch.zeros((mpcdar.FEATURE_DIM, mpcdar.TOTAL_COLUMNS), dtype=torch.float64)
    W[:, :mpcdar.FEATURE_DIM] = torch.eye(mpcdar.FEATURE_DIM, dtype=torch.float64)
    bias = torch.zeros(mpcdar.TOTAL_COLUMNS, dtype=torch.float32)
    mu = torch.linspace(-0.2, 0.2, mpcdar.FEATURE_DIM, dtype=torch.float64)
    features = torch.zeros((2, mpcdar.FEATURE_DIM), dtype=torch.float64)
    features[0, :4] = torch.tensor([1.0, 2.0, 0.0, -1.0])
    features[1, :4] = torch.tensor([-0.5, 0.3, 1.0, 0.2])
    V = torch.zeros((mpcdar.TOTAL_COLUMNS, mpcdar.FEATURE_DIM), dtype=torch.float64)
    V[:mpcdar.FEATURE_DIM] = torch.eye(mpcdar.FEATURE_DIM, dtype=torch.float64)
    return features, W, bias, mu, V


def test_config_and_review_are_exactly_locked() -> None:
    spec = mpcdar._load_spec(CONFIG)
    mpcdar._assert_contract(spec)
    assert mpcdar._load_review(REVIEW, spec) == REVIEW
    assert spec.payload["candidate_family"] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert spec.payload["data_aware"]["V_VT_materialized"] is False


def test_authority_and_candidate_family_are_frozen() -> None:
    assert mpcdar.RMAL_CONSUMER["probe_sha256"] == "3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed"
    assert mpcdar.RANK_CANDIDATES == (1, 2, 4, 8, 16, 32, 64, 128, 256)
    assert mpcdar.SEARCH_RANKS == mpcdar.RANK_CANDIDATES[:-1]
    assert mpcdar.RANK0 == 0
    assert mpcdar.RAW_COMPARATOR_CONTRACT["primary_search"] is False
    assert mpcdar.RAW_COMPARATOR_CONTRACT["confirmation_trigger"] is False


def test_config_mutation_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["candidate_family"] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    mutated = tmp_path / CONFIG.name
    mutated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate family|candidate_family"):
        mpcdar._assert_contract(mpcdar._load_spec(mutated))


def test_data_aware_boundary_uses_its_own_spectrum_and_labels() -> None:
    da_sigma = torch.tensor([10.0, 9.0, 2.0, 1.0] + [0.5] * 252, dtype=torch.float64)
    raw_sigma = torch.tensor([10.0, 8.0, 7.0, 6.0] + [0.5] * 252, dtype=torch.float64)
    da_squared = da_sigma.square()
    raw_squared = raw_sigma.square()
    da = SimpleNamespace(
        singular_values=da_sigma,
        squared_singular_values=da_squared,
        cumulative_energy=torch.cumsum(da_squared, 0) / da_squared.sum(),
    )
    raw = SimpleNamespace(
        singular_values=raw_sigma,
        squared_singular_values=raw_squared,
        cumulative_energy=torch.cumsum(raw_squared, 0) / raw_squared.sum(),
    )
    da_telemetry = mpcdar._data_aware_boundary_telemetry(2, da)
    raw_telemetry = mpcdar._raw_boundary_telemetry(2, {"full": raw})
    assert da_telemetry["sigma_DA_k"] == pytest.approx(9.0)
    assert da_telemetry["sigma_DA_k_plus_1"] == pytest.approx(2.0)
    assert da_telemetry["gap_source"] == "A_fit/G_fit data-aware spectrum"
    assert raw_telemetry["relative_gap_raw"] == pytest.approx(1.0 / 8.0)
    assert raw_telemetry["gap_source"] == "LSDA raw-W comparator telemetry"
    assert "relative_gap_DA" not in raw_telemetry


def test_data_aware_basis_is_orthogonal_and_replays_A_and_W_projectors() -> None:
    _, W, _, _, _ = _full_row_fixture()
    C = torch.diag(torch.linspace(0.5, 2.0, mpcdar.FEATURE_DIM, dtype=torch.float64))
    basis = mpcdar._build_data_aware_basis(W, C)
    assert basis["V"].shape == (mpcdar.TOTAL_COLUMNS, mpcdar.FEATURE_DIM)
    assert basis["orthogonality"]["passed"] is True
    assert mpcdar._projector_identity(basis["A_fit"], basis["V"])["max_abs_residual"] < 1.0e-8
    assert mpcdar._projector_identity(W, basis["V"])["max_abs_residual"] < 1.0e-8


def test_sign_flip_leaves_projector_unchanged() -> None:
    _, W, _, _, V = _full_row_fixture()
    result = mpcdar._sign_flip_projector_invariance(W, V, rank=4)
    assert result["passed"] is True
    assert result["max_abs_residual"] == 0.0


def test_raw_lrcc_projector_equals_raw_right_basis_projector() -> None:
    _, W, _, _, V = _full_row_fixture()
    U = torch.eye(mpcdar.FEATURE_DIM, dtype=torch.float64)
    result = mpcdar._raw_projector_equivalence(W, U, V, 4)
    assert result["passed"] is True
    assert result["max_abs_residual"] == 0.0


def test_mean_preserving_raw_uses_the_same_fit_mean_and_centered_input() -> None:
    features, W, bias, mu, V = _full_row_fixture()
    U = torch.eye(mpcdar.FEATURE_DIM, dtype=torch.float64)
    actual = mpcdar._mean_preserving_raw_prediction(features, U, W, bias, mu, 4)
    expected = (
        mu.unsqueeze(0).mm(W)
        + (features - mu).mm(U[:, :4]).mm(U[:, :4].T.mm(W))
    )
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_data_aware_factorized_prediction_matches_explicit_fixture() -> None:
    features, W, bias, mu, V = _full_row_fixture()
    rank = 4
    V_k = V[:, :rank]
    explicit = (features - mu).mm(W.mm(V_k)).mm(V_k.T) + mu.unsqueeze(0).mm(W) + bias
    factorized = mpcdar._data_aware_prediction(features, V, W, bias, mu, rank)
    assert torch.allclose(factorized, explicit, atol=1.0e-12, rtol=1.0e-12)


def test_sequential_search_keeps_mp_raw_outside_stop_rule() -> None:
    events: list[tuple[str, int]] = []

    def selection(rank: int) -> dict[str, object]:
        events.append(("selection", rank))
        return {"selection_gate": {"passed": rank == 4}, "matched_control": {"passed": True}}

    def confirmation(rank: int) -> dict[str, object]:
        events.append(("confirmation", rank))
        return {"selection_gate": {"passed": True}, "confirmation_gate": {"passed": True}, "matched_control": {"passed": False}}

    result = mpcdar._sequential_search(selection, confirmation, anchor_passed=True)
    assert events == [("selection", 1), ("selection", 2), ("selection", 4), ("confirmation", 4)]
    assert result["first_confirmed_data_aware_candidate_rank"] == 4
    assert result["matched_control_participated"] is False


def test_anchor_failure_stops_before_data_aware_search() -> None:
    called = False

    def selection(rank: int) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    result = mpcdar._sequential_search(selection, lambda rank: {}, anchor_passed=False)
    assert called is False
    assert result["anchor_failure_stop"] is True
    assert result["first_confirmed_data_aware_candidate_rank"] is None


def test_no_target_or_update_path_and_primary_gate_inheritance() -> None:
    source = inspect.getsource(mpcdar)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "return_trace=True" not in source
    assert mpcdar.DATA_AWARE_CONTRACT["target_labels_used"] is False
    assert mpcdar.DATA_AWARE_CONTRACT["target_values_used"] is False
    assert mpcdar.GATE_CONTRACT["no_energy_gate"] is True
    assert mpcdar.AUTHORIZATION["formal_promotion"] is False
    assert mpcdar.AUTHORIZATION["training_authorized"] is False


def test_cli_parses_validate_and_run_actions() -> None:
    validate = _parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-mean-preserving-centered-data-aware-readout-recovery-validate",
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
            "scaled-chen-mean-preserving-centered-data-aware-readout-recovery-run",
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


def test_lazy_router_contains_only_registered_mpcdar_route() -> None:
    source = inspect.getsource(lazy_router.dispatch)
    assert "scaled-chen-mean-preserving-centered-data-aware-readout-recovery-validate" in source
    assert "scaled-chen-mean-preserving-centered-data-aware-readout-recovery-run" in source
    assert "optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery" in source


def test_result_language_and_next_action_are_bounded() -> None:
    source = inspect.getsource(mpcdar)
    assert "intrinsic rank" not in source.lower()
    assert "minimum rank" not in source.lower()
    assert mpcdar.NEXT_ACTION_RESULT_REVIEW == "manual_review_MPCDAR_result"
