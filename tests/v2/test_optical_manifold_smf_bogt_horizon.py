from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import optical_manifold_smf_bogt_horizon as horizon


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_horizon_v2.yaml"
REVIEW = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_horizon_review_v2.json"


def test_horizon_contract_is_observational_for_wall_clock() -> None:
    spec = horizon._load_spec(CONFIG)
    horizon._assert_contract(spec)
    review = horizon._load_review(REVIEW, spec)
    assert review["approved_contract"]["wall_clock"] == "observational_only"
    assert "max_wall_clock_s" not in CONFIG.read_text(encoding="utf-8")
    assert "training_wall_clock_gate" not in CONFIG.read_text(encoding="utf-8")


def test_horizon_schedule_uses_five_thousand_balanced_steps() -> None:
    labels = torch.arange(55000, dtype=torch.long) % 10
    calibration, schedule = horizon._horizon_schedule(labels)
    assert calibration.shape == (1000,)
    assert schedule.shape == (5000, 10)
    assert not torch.isin(calibration, schedule).any()
    assert torch.equal(labels.index_select(0, schedule.reshape(-1)), torch.arange(10).repeat(5000))
    assert schedule[:, 5].unique().numel() == 4863
    assert all(schedule[:, label].unique().numel() == 5000 for label in range(10) if label != 5)


def test_horizon_contract_rejects_hard_wall_gate() -> None:
    spec = horizon._load_spec(CONFIG)
    invalid = horizon.HorizonSpec(spec.path, {**spec.payload, "training": {**spec.payload["training"], "max_wall_clock_s": 30}}, spec.config_hash)
    with pytest.raises(ValueError, match="hard training wall-clock"):
        horizon._assert_contract(invalid)


def test_horizon_phase_histogram_reports_registered_levels() -> None:
    formal, _ = horizon._load_formal(horizon._load_spec(CONFIG))
    model = horizon.bogt.ScaledChenBOGTModel(formal)
    report = horizon._phase_histogram(model)
    assert set(report) == {"encoder", "decoder"}
    for value in report.values():
        assert len(value["level_counts"]) == 8
        assert sum(value["level_counts"]) == 2 * 100 * 100


def test_horizon_router_is_lazy() -> None:
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    schema = (ROOT / "src/ldm_project/v2/application/_cli_schema_research.py").read_text(encoding="utf-8")
    for action in ("smf-bogt-horizon-validate", "smf-bogt-horizon-preflight", "smf-bogt-horizon-train"):
        assert action in router or action in schema
