from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_failure_attribution as audit


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_failure_attribution_v1.yaml"
REVIEW = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_failure_attribution_review_v1.json"


def test_failure_attribution_contract_and_review_bind_source() -> None:
    spec = audit._load_spec(CONFIG)
    review = audit._load_review(REVIEW, spec)
    assert review.exists()
    assert spec.payload["dose"]["multipliers"] == [0.5, 1.0, 2.0, 4.0]


def test_candidate_grid_requires_both_weights_above_registered_dose() -> None:
    kl_norm = 0.1 / audit.PREFLIGHT_ALPHA
    op_norm = 0.1 / audit.PREFLIGHT_GAMMA
    vector = {
        "mse": torch.tensor([1.0, 0.0]),
        "kl": torch.tensor([kl_norm, 0.0]),
        "op": torch.tensor([op_norm, 0.0]),
    }
    base_report = {"vectors": vector}
    weak = audit._candidate_report(base_report, audit.PREFLIGHT_ALPHA, audit.PREFLIGHT_GAMMA)
    strong = audit._candidate_report(base_report, 2.0 * audit.PREFLIGHT_ALPHA, 2.0 * audit.PREFLIGHT_GAMMA)
    assert weak["compatible"]
    assert strong["compatible"]
    assert audit._select_candidate([weak]) is None
    assert audit._select_candidate([strong]) == strong


def test_candidate_report_rejects_conflicting_objective_direction() -> None:
    base_report = {
        "vectors": {
            "mse": torch.tensor([1.0, 0.0]),
            "kl": torch.tensor([-1.0, 0.0]),
            "op": torch.tensor([1.0, 0.0]),
        }
    }
    report = audit._candidate_report(base_report, 2.0 * audit.PREFLIGHT_ALPHA, 2.0 * audit.PREFLIGHT_GAMMA)
    assert report["dots"]["kl"] < 0.0
    assert not report["compatible"]


def test_fresh_panels_are_balanced_and_disjoint() -> None:
    labels = torch.arange(60000, dtype=torch.long) % 10
    data = bogt.ResearchData(
        images=torch.zeros(60000, 1, 28, 28),
        labels=labels,
        calibration_indices=torch.arange(1000),
        schedule=torch.arange(1000, 2000).reshape(100, 10),
        heldout_indices=torch.arange(55000, 60000),
        panels=tuple(torch.arange(2000 + index * 10, 2010 + index * 10) for index in range(3)),
        manifest={"manifest_hash": "synthetic"},
    )
    selection, confirmation = audit._fresh_panels(data)
    all_panels = torch.cat((*selection, *confirmation))
    assert all_panels.numel() == 600
    assert torch.unique(all_panels).numel() == 600
    assert not torch.isin(all_panels, torch.cat((data.calibration_indices, data.schedule.reshape(-1), data.heldout_indices, *data.panels))).any()
    assert torch.equal(labels.index_select(0, selection[0]), torch.arange(10).repeat(10))


def test_review_tampering_is_rejected_before_source_use(tmp_path: Path) -> None:
    spec = audit._load_spec(CONFIG)
    payload = json.loads(REVIEW.read_text(encoding="utf-8"))
    payload["source_commit"] = "stale"
    path = tmp_path / "review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="review contract"):
        audit._load_review(path, spec)


def test_lazy_router_contains_failure_attribution_actions() -> None:
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    schema = (ROOT / "src/ldm_project/v2/application/_cli_schema_research.py").read_text(encoding="utf-8")
    for action in ("smf-bogt-failure-attribution-validate", "smf-bogt-failure-attribution-run"):
        assert action in router and action in schema
    assert "optical_manifold_smf_bogt_failure_attribution" in router
