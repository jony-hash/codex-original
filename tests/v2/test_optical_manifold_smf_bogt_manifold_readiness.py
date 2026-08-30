from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research import optical_manifold_smf_bogt_manifold_readiness as audit


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1.yaml"
REVIEW = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_manifold_readiness_review_v1.json"


def test_contract_and_review_are_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec).name.endswith("manifold_readiness_review_v1.json")
    assert spec.payload["hypothesis"] == "H-LOMA-CBOGT-S0-MRA"


def test_contract_rejects_missing_prior_or_gate_mutation() -> None:
    spec = audit._load_spec(CONFIG)
    with pytest.raises(ValueError, match="parent prior"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "parent_prior": {}}, spec.config_hash))
    with pytest.raises(ValueError, match="gate contract"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "gates": {**spec.payload["gates"], "rank_retention": 0.5}}, spec.config_hash))


def test_selection_is_deterministic_balanced_and_tail_only() -> None:
    labels = torch.arange(10000, dtype=torch.long) % 10
    first = audit._select_indices(labels)
    second = audit._select_indices(labels)
    assert torch.equal(first, second)
    assert first.shape == (1000,)
    assert int(first.min()) >= 5000 and int(first.max()) < 10000
    assert torch.equal(torch.bincount(labels.index_select(0, first), minlength=10), torch.full((10,), 100))


def test_rank_metrics_include_permuted_control_and_retrieval() -> None:
    values = torch.randn(100, 1, 8, 8, generator=torch.Generator().manual_seed(7))
    labels = torch.arange(100) % 10
    metrics = audit._rank_metrics(values, labels)
    assert metrics["finite"] is True
    assert metrics["entropy_effective_rank"] > 0.0
    assert metrics["participation_rank"] > 0.0
    assert "true_label_distance" in metrics and "permuted_label_distance" in metrics


def test_bootstrap_is_deterministic() -> None:
    labels = torch.arange(1000) % 10
    delta = torch.ones(1000)
    assert audit._stratified_bootstrap_ci(delta, labels, replicates=20, seed=11) == (1.0, 1.0)


def test_classification_boundaries() -> None:
    good = {"retention": {name: {"geometric_mean_retention": 0.5} for name, *_ in audit.SEGMENTS}}
    assert audit._classify_attribution(good, good) == "distributed_bottleneck_supported"
    unknown = {"retention": {"slm_to_pre_ols": {"geometric_mean_retention": 0.8}}}
    assert audit._classify_attribution(unknown, unknown) == "attribution_underdetermined"


def test_prior_accessibility_requires_both_kl_directions_and_reconstruction() -> None:
    step0 = {"ols": {"bogt_kl_surrogate": 2.0, "truncated_normal_numerical_kl": 2.0, "complex_effective_rank": 100.0, "power_effective_rank": 100.0}}
    step1000 = {"ols": {"bogt_kl_surrogate": 1.2, "truncated_normal_numerical_kl": 1.2, "complex_effective_rank": 100.0, "power_effective_rank": 100.0}}
    step5000 = {"ols": {"bogt_kl_surrogate": 1.0, "truncated_normal_numerical_kl": 1.0, "complex_effective_rank": 100.0, "power_effective_rank": 100.0}}
    assert audit._classify_prior_accessibility(step0, step1000, step5000, True)[0] == "prior_accessible"
    failed = {"ols": {"bogt_kl_surrogate": 2.0, "truncated_normal_numerical_kl": 2.0, "complex_effective_rank": 100.0, "power_effective_rank": 100.0}}
    assert audit._classify_prior_accessibility(step0, step1000, failed, True)[0] != "prior_accessible"


def test_cli_actions_are_registered_and_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args(["research", "optical-manifold", "scaled-chen-manifold-readiness-validate", "--config", "config.yaml", "--review-contract", "review.json", "--dataset-root", "data", "--output", "validation.json"])
    run = parser.parse_args(["research", "optical-manifold", "scaled-chen-manifold-readiness-run", "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json", "--dataset-root", "data", "--output", "audit.json"])
    assert validate.research_action.endswith("manifold-readiness-validate")
    assert run.device == "cuda"
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_manifold_readiness" in router
    assert "import optical_manifold_smf_bogt_manifold_readiness" not in router


def test_no_optimizer_or_evaluator_in_audit_source() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "evaluator_calls" in source

