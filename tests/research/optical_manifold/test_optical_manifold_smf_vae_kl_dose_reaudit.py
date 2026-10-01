import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_vae_kl_dose_reaudit as audit


def test_config_locks_corrected_beta_dose_contract() -> None:
    root = Path(__file__).parents[3]
    spec = audit._load_spec(
        root / "configs/research/optical_manifold/lightgen_optical_manifold_smf_vae_kl_dose_reaudit_v1.yaml"
    )
    audit._assert_contract(spec)
    review = audit._load_review(
        root / "configs/research/optical_manifold/lightgen_optical_manifold_smf_vae_kl_dose_reaudit_review_v1.json",
        spec,
    )
    assert review.name.endswith("review_v1.json")
    assert audit.TRAINING_BETA == pytest.approx(0.0018256452619527068)
    assert audit.DOSES == (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)


def test_dose_report_multiplies_registered_beta() -> None:
    rec = {"system.encoder.phase": torch.tensor([1.0, 0.0])}
    kl = {"system.encoder.phase": torch.tensor([0.0, 1.0])}
    report = audit._dose_report(rec, kl, 4.0)
    assert report["effective_beta"] == pytest.approx(4.0 * audit.TRAINING_BETA)
    assert report["combined_dot_reconstruction"] == pytest.approx(1.0)
    assert report["combined_dot_kl"] == pytest.approx(4.0 * audit.TRAINING_BETA)


def test_old_unscaled_formula_is_not_the_registered_effective_beta() -> None:
    rec = {"system.encoder.phase": torch.tensor([1.0, 0.0])}
    kl = {"system.encoder.phase": torch.tensor([0.0, 1.0])}
    report = audit._dose_report(rec, kl, 1.0)
    assert report["effective_beta"] != 1.0
    assert report["effective_beta"] == pytest.approx(audit.TRAINING_BETA)


@pytest.mark.historical_artifact
def test_attempt_06_outcome_is_consumed_from_contract_layer() -> None:
    root = Path(__file__).parents[3]
    source = root / "outputs/20260828_v2/runs/lightgen_optical_manifold_smf_vae_kl_attribution_audit_v1_attempt_06/audit.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    assert record.get("outcome") is None
    assert record["contract"]["outcome"] == "diagnostic_completed_parameterization_limited"


def test_gradient_boundary_reads_stacked_phase_layers() -> None:
    rec = {
        "system.encoder.phase": torch.ones(2, 3),
        "system.decoder.phase": torch.full((2, 3), 2.0),
    }
    kl = {
        "system.encoder.phase": torch.ones(2, 3),
        "system.decoder.phase": torch.zeros(2, 3),
        "posterior.trunk.weight": torch.ones(2, 2),
    }
    checks = audit._gradient_boundary_checks(rec, kl)
    assert checks["four_phase_reconstruction"] is True
    assert checks["kl_encoder_posterior_only"] is True
    assert checks["reconstruction_encoder_layer_norms"] == pytest.approx([3**0.5, 3**0.5])


def test_gradient_boundary_rejects_missing_or_zero_phase_layer() -> None:
    rec = {
        "system.encoder.phase": torch.cat((torch.ones(1, 3), torch.zeros(1, 3))),
        "system.decoder.phase": torch.ones(2, 3),
    }
    kl = {
        "system.encoder.phase": torch.ones(2, 3),
        "system.decoder.phase": torch.zeros(2, 3),
        "posterior.trunk.weight": torch.ones(2, 2),
    }
    checks = audit._gradient_boundary_checks(rec, kl)
    assert checks["four_phase_reconstruction"] is False
