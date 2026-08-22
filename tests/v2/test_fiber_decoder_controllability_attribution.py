from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research import fiber_decoder_controllability_attribution as decoder


def _args(action: str) -> list[str]:
    values = ["research", "fiber-decoder-controllability-attribution", action]
    if action in {"validate", "run"}:
        values.extend(
            [
                "--config", "config.yaml", "--dataset-root", "data",
                "--primary-evaluator-checkpoint", "primary.pt",
                "--secondary-evaluator-checkpoint", "secondary.pt",
                "--primary-qualification-evidence", "primary.json",
                "--secondary-qualification-evidence", "secondary.json",
                "--research-initialization-checkpoint", "init.pt",
                "--source-training-evidence", "source.json", "--v3-closeout", "v3.json",
                "--attribution-measurement", "attribution.json",
                "--interface-validation", "interface-validation.json",
                "--interface-measurement", "interface-measurement.json",
                "--interface-closeout", "interface-closeout.json",
                "--loss-exposure-validation", "loss-validation.json",
                "--loss-exposure-measurement", "loss-measurement.json",
                "--loss-exposure-presentation", "loss-presentation.json",
                "--loss-exposure-closeout", "loss-closeout.json",
            ]
        )
    if action == "validate":
        values.extend(["--lp01-fixture", "fixture.json"])
    if action == "run":
        values.extend(["--lp01-fixture", "fixture.json", "--validation-evidence", "validation.json"])
    if action == "report":
        values.extend(["--measurement-evidence", "measurement.json"])
    if action == "closeout":
        values.extend(["--measurement-evidence", "measurement.json", "--presentation-evidence", "presentation.json"])
    values.extend(["--output", "out.json"])
    return values


def _rows(*, shared_gain: float = 0.0, shared_min: float = 0.0, oracle_gain: float = 0.0, oracle_min: float = 0.0) -> dict:
    rows = {variant: {} for variant in decoder.VARIANTS}
    for variant in decoder.VARIANTS:
        for panel in decoder.PANELS:
            rows[variant][str(panel)] = {
                "primary_accuracy": 0.5,
                "secondary_accuracy": 0.1,
                "secondary_min_recall": shared_min if variant == "Dshared+" else oracle_min if variant == "Dclass-oracle" else 0.0,
                "saturation": 0.01,
            }
    return rows


def _classification_payload(*, shared_gain: float = 0.0, shared_min: float = 0.0, oracle_gain: float = 0.0, oracle_min: float = 0.0, negative_fraction: float = 0.0, candidate_selected: bool = False) -> dict:
    rows = _rows(shared_gain=shared_gain, shared_min=shared_min, oracle_gain=oracle_gain, oracle_min=oracle_min)
    bootstrap = {}
    for variant in ("Dshared+", "Dshared-", "Dscrambled", "Dclass-oracle"):
        gain = shared_gain if variant == "Dshared+" else oracle_gain if variant == "Dclass-oracle" else 0.0
        bootstrap[variant] = {
            "observed_gain": gain,
            "ci95": [0.01 if gain else -0.01, 0.03],
            "panel_observed_gain": {str(panel): gain for panel in decoder.PANELS},
        }
    return {
        "rows": rows,
        "bootstrap": bootstrap,
        "design": {
            "candidate_selected": candidate_selected,
            "local_consistency": {"cosine": 0.95, "odd_error": 0.05},
            "gradient_conflict": {"negative_fraction_below_minus_0_10": negative_fraction},
        },
    }


def test_decoder_cli_has_four_lifecycle_actions() -> None:
    parser = _parser()
    for action in ("validate", "run", "report", "closeout"):
        parsed = parser.parse_args(_args(action))
        assert parsed.research_program == "fiber-decoder-controllability-attribution"
        assert parsed.research_action == action


def test_direction_normalization_and_scramble_preserve_rms_and_distribution() -> None:
    direction = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) - 11.5
    normalized = decoder._normalize_direction(direction)
    scrambled = decoder._scramble_direction(normalized)
    assert decoder._phase_rms(normalized) == pytest.approx(1.0)
    assert decoder._phase_rms(scrambled) == pytest.approx(1.0)
    assert torch.equal(torch.sort(normalized.flatten()).values, torch.sort(scrambled.flatten()).values)
    assert torch.isfinite(scrambled).all()


def test_scale_selection_uses_only_design_rows_and_breaks_ties_by_smallest_scale() -> None:
    rows = [
        {"scale": 0.025, "primary_macro_ce": 0.4, "eligible": True},
        {"scale": 0.005, "primary_macro_ce": 0.4, "eligible": True},
        {"scale": 0.05, "primary_macro_ce": 0.1, "eligible": False},
    ]
    assert decoder._select_scale(rows) == (0.005, True)
    assert decoder._select_scale([{**rows[0], "eligible": False}]) == (0.005, False)


def test_phase_only_transfer_has_unit_amplitude() -> None:
    phase = torch.randn(2, 1, 10, 10, generator=torch.Generator().manual_seed(9))
    transfer = torch.polar(torch.ones_like(phase), phase)
    assert float((transfer.abs() - 1.0).abs().max()) <= 1.0e-6


def test_gradient_conflict_reports_negative_pairs() -> None:
    gradients = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    result = decoder._gradient_conflict(gradients)
    assert result["negative_fraction_below_minus_0_10"] == pytest.approx(2 / 6)
    assert result["minimum_off_diagonal_cosine"] == pytest.approx(-1.0)


def test_classification_has_four_fail_closed_outcomes() -> None:
    assert decoder._classification(_classification_payload(shared_gain=0.02, shared_min=0.1, candidate_selected=True))["conclusion"] == "shared_phase_only_decoder_recovery_candidate"
    assert decoder._classification(_classification_payload(oracle_gain=0.02, oracle_min=0.1, negative_fraction=0.4))["conclusion"] == "fixed_decoder_class_gradient_conflict"
    assert decoder._classification(_classification_payload())["conclusion"] == "zero_phase_decoder_local_reachability_failed"
    inconclusive = _classification_payload(oracle_gain=0.02, oracle_min=0.1, candidate_selected=True)
    assert decoder._classification(inconclusive)["conclusion"] == "decoder_controllability_inconclusive"


def test_tampered_artifact_reference_is_rejected(tmp_path) -> None:
    artifact = tmp_path / "tensor.pt"
    decoder.atomic_torch_save(artifact, torch.zeros(2))
    reference = decoder._ref(artifact, tmp_path, torch.zeros(2))
    decoder.atomic_torch_save(artifact, torch.ones(2))
    with pytest.raises(ValueError, match="hash-mismatched"):
        decoder._resolve(reference, tmp_path)


def test_dirty_validation_fails_closed_before_scientific_payload(tmp_path, monkeypatch) -> None:
    output = tmp_path / "validation.json"
    monkeypatch.setattr(decoder, "_dirty", lambda: True)
    assert decoder.validate(SimpleNamespace(output=str(output))) == 2
    assert '"gate": "invalid_harness"' in output.read_text(encoding="utf-8")
