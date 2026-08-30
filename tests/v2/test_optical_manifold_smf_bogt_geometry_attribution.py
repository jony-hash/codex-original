from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research import optical_manifold_smf_bogt_geometry_attribution as audit


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_geometry_attribution_v1.yaml"
REVIEW = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_geometry_attribution_review_v1.json"


def test_geometry_contract_and_review_are_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    review = audit._load_review(REVIEW, spec)
    assert review.name.endswith("geometry_attribution_review_v1.json")
    assert audit._variants()[-1].key == "fill_0.90_mag_1.20"


def test_geometry_contract_rejects_grid_mutation() -> None:
    spec = audit._load_spec(CONFIG)
    invalid = audit.GeometrySpec(
        spec.path,
        {**spec.payload, "audit": {**spec.payload["audit"], "material_relative_improvement": 0.01}},
        spec.config_hash,
    )
    with pytest.raises(ValueError, match="audit contract"):
        audit._assert_contract(invalid)


def test_selection_is_deterministic_balanced_and_uses_test_tail() -> None:
    labels = torch.arange(10000, dtype=torch.long) % 10
    first = audit._select_indices(labels)
    second = audit._select_indices(labels)
    assert torch.equal(first, second)
    assert first.shape == (1000,)
    assert int(first.min()) >= 5000
    assert int(first.max()) < 10000
    assert torch.equal(torch.bincount(labels.index_select(0, first), minlength=10), torch.full((10,), 100, dtype=torch.long))


def test_bootstrap_ci_is_stratified_and_deterministic() -> None:
    labels = torch.arange(1000, dtype=torch.long) % 10
    delta = torch.ones(1000)
    first = audit._stratified_bootstrap_ci(delta, labels, replicates=50, seed=77)
    second = audit._stratified_bootstrap_ci(delta, labels, replicates=50, seed=77)
    assert first == second
    assert first == pytest.approx((1.0, 1.0))


def test_historical_json_hash_is_checked_before_consumption() -> None:
    with pytest.raises(ValueError, match="does not match"):
        audit._json_payload(CONFIG, "0" * 64)


def test_cli_router_and_schema_expose_lazy_geometry_actions() -> None:
    parser = cli._parser()
    validate = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-geometry-attribution-validate",
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
            "scaled-chen-geometry-attribution-run",
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
    assert validate.research_action == "scaled-chen-geometry-attribution-validate"
    assert run.research_action == "scaled-chen-geometry-attribution-run"
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_geometry_attribution" in router
    assert "import optical_manifold_smf_bogt_geometry_attribution" not in router
