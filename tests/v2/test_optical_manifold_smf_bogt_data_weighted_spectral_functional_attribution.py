from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution as dsfa,
)


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / (
    "configs/research/"
    "lightgen_optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution_v1.yaml"
)
REVIEW = ROOT / (
    "configs/research/"
    "lightgen_optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution_review_v1.json"
)


def test_config_review_and_upstream_contract_are_locked() -> None:
    spec = dsfa._load_spec(CONFIG)
    dsfa._assert_contract(spec)
    assert dsfa._load_review(REVIEW, spec) == REVIEW
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert review["hypothesis"] == dsfa.HYPOTHESIS
    assert review["approved_contract"]["measurement"]["registered_ranks"] == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]


def test_registered_ranks_and_fixed_mode_bands_are_not_expanded() -> None:
    assert dsfa.LRRC_RANKS == (1, 2, 4, 8, 16, 32, 64, 128, 256)
    assert dsfa.REGISTERED_RANKS == tuple(range(1, dsfa.FEATURE_DIM + 1))
    assert dsfa.REGISTERED_BANDS["B128"] == (65, 128)
    assert dsfa.REGISTERED_BANDS["B256"] == (129, 256)
    assert tuple(dsfa.REGISTERED_BANDS) == (
        "B1",
        "B2",
        "B4",
        "B8",
        "B16",
        "B32",
        "B64",
        "B128",
        "B256",
    )


def test_q_from_w_transpose_u_preserves_operator_energy_on_fixture() -> None:
    W = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))
    U = torch.eye(3, dtype=torch.float64)
    q = W.transpose(0, 1).mm(U)
    sigma = torch.linalg.svdvals(W)
    assert torch.allclose(q.square().sum(dim=0), sigma.square())
    assert torch.allclose(q, W.transpose(0, 1).mm(U))


def test_factorized_mode_terms_match_explicit_small_fixture() -> None:
    activations = torch.tensor(
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]], dtype=torch.float64
    )
    residual = torch.tensor(
        [[2.0, -1.0, 0.5], [0.0, 4.0, -2.0], [1.0, 0.5, 3.0]],
        dtype=torch.float64,
    )
    q = torch.tensor([[1.5, -0.5], [0.0, 2.0], [-1.0, 0.25]], dtype=torch.float64)
    explicit = dsfa._explicit_mode_terms(activations, residual, q)
    factorized = dsfa._factorized_mode_terms(activations, residual, q)
    for actual, expected in zip(factorized, explicit):
        assert torch.allclose(actual, expected)


def test_activation_second_moment_variance_identity_is_exact_fixture() -> None:
    activations = torch.tensor(
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]], dtype=torch.float64
    )
    mean = activations.mean(dim=0)
    second = activations.square().mean(dim=0)
    variance = second - mean.square()
    assert torch.allclose(second, variance + mean.square())
    assert torch.all(variance >= 0)


def test_signed_gain_is_preserved_without_clamp_or_absolute_value() -> None:
    activations = torch.ones((1, 2), dtype=torch.float64)
    residual = torch.tensor([[-1.0, 3.0]], dtype=torch.float64)
    q = torch.eye(2, dtype=torch.float64)
    _, _, gain = dsfa._factorized_mode_terms(activations, residual, q)
    total = gain.sum()
    shares = gain / total
    assert gain[0] < 0
    assert gain[1] > 0
    assert shares[0] < 0
    assert shares[1] > 1


def test_quantile_summary_keeps_registered_summary_fields() -> None:
    summary = dsfa._quantile_summary(torch.arange(1, 6, dtype=torch.float64))
    assert summary["count"] == 5
    assert summary["sum"] == pytest.approx(15.0)
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["median"] == pytest.approx(3.0)
    assert summary["q05"] == pytest.approx(1.2)
    assert summary["q95"] == pytest.approx(4.8)


def test_no_update_path_and_streaming_memory_contract_are_present() -> None:
    source = inspect.getsource(dsfa)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "for rank in range(129, 256)" not in source
    assert "[N,256,40000]" not in source
    assert dsfa.MEASUREMENT["no_training"] is True
    assert dsfa.MEASUREMENT["no_optimizer"] is True
    assert dsfa.MEASUREMENT["no_backward"] is True
    assert dsfa.MEASUREMENT["no_new_rank_search"] is True
    assert dsfa.PANEL_CONTRACT["streaming"] is True
    assert dsfa.PANEL_CONTRACT["pooled_feature_bank_saved"] is False
    assert dsfa.PANEL_CONTRACT["target_bank_saved"] is False
    assert dsfa.PANEL_CONTRACT["raw_cmos_bank_saved"] is False
    assert dsfa.PANEL_CONTRACT["internal_stage_capture"] is False


def test_write_once_slots_and_lazy_cli_route() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "attempt" / "validation.json"
        dsfa._prepare_validation_slot(path)
        with pytest.raises(FileExistsError):
            dsfa._prepare_validation_slot(path)

        path.write_text("{}", encoding="utf-8")
        audit_path = path.with_name("audit.json")
        assert dsfa._check_run_slot(audit_path, path) is None

    parser = cli._parser()
    validate = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-data-weighted-spectral-functional-attribution-validate",
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
            "scaled-chen-data-weighted-spectral-functional-attribution-run",
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
    router = (
        ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py"
    ).read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution" in router
    assert "from .research.optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution import dispatch" in router


def test_compact_authorization_remains_descriptive_only() -> None:
    assert dsfa.AUTHORIZATION["descriptive_only"] is True
    assert dsfa.AUTHORIZATION["formal_promotion"] is False
    assert dsfa.AUTHORIZATION["training_authorized"] is False
    assert dsfa.AUTHORIZATION["formal_training_authorized"] is False
    assert "scientific_rank_classification" in dsfa.AUTHORIZATION["prohibited"]
    assert dsfa.NEXT_ACTION_RESULT_REVIEW == "manual_review_DSFA_result"
