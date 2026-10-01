from __future__ import annotations

import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.optical_manifold import fiber_loss_exposure_attribution as exposure
from ldm_project.v2.student.conditional import fixed_exposure_view


def _args(action: str) -> list[str]:
    values = ["research", "fiber-loss-exposure-attribution", action]
    if action in {"validate", "run"}:
        values.extend(
            [
                "--config", "config.yaml",
                "--dataset-root", "data",
                "--primary-evaluator-checkpoint", "primary.pt",
                "--secondary-evaluator-checkpoint", "secondary.pt",
                "--primary-qualification-evidence", "primary.json",
                "--secondary-qualification-evidence", "secondary.json",
                "--research-initialization-checkpoint", "init.pt",
                "--source-training-evidence", "source.json",
                "--v3-closeout", "v3.json",
                "--attribution-measurement", "attribution.json",
                "--interface-validation", "interface-validation.json",
                "--interface-measurement", "interface-measurement.json",
                "--interface-closeout", "interface-closeout.json",
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


def test_loss_exposure_cli_has_four_lifecycle_actions() -> None:
    parser = _parser()
    for action in ("validate", "run", "report", "closeout"):
        parsed = parser.parse_args(_args(action))
        assert parsed.research_program == "fiber-loss-exposure-attribution"
        assert parsed.research_action == action


def test_calibration_ratio_and_fixed_exposure_equivalence() -> None:
    input_energy = torch.tensor([2.0, 3.0, 5.0])
    projected_energy = input_energy * 0.75
    rho, gain = exposure._calibration_ratio(input_energy, projected_energy)
    assert rho == 0.75
    assert gain == 1.0 / 0.75

    raw = torch.rand((2, 1, 8, 8), generator=torch.Generator().manual_seed(3))
    nominal = fixed_exposure_view(raw * gain, 0.5)
    calibrated = fixed_exposure_view(raw, 0.5 / gain)
    assert torch.allclose(nominal, calibrated, atol=1.0e-6, rtol=1.0e-6)


def _classification_payload(
    *,
    global_gain: float,
    oracle_gain: float,
    shape_gain: float,
    ci_low: float,
    recovered_panels: int,
) -> dict:
    rows = {variant: {} for variant in exposure.BASE_VARIANTS + tuple(
        f"S0-gain-{int(round(value * 100)):03d}" for value in exposure.GAIN_GRID
    )}
    for variant in rows:
        for index, panel in enumerate(exposure.PANELS):
            rows[variant][str(panel)] = {
                "secondary_accuracy": 0.1 + (global_gain if variant == "S0-Ecal" else 0.0),
                "secondary_min_recall": 0.1 if variant == "S0-Ecal" and index < recovered_panels else 0.0,
                "primary_accuracy": 0.9,
                "saturation": 0.01,
            }
    bootstrap = {}
    for variant, gain in (
        ("S0-Ecal", global_gain),
        ("S0-per-sample-oracle", oracle_gain),
        ("I0-Acal", shape_gain),
    ):
        bootstrap[variant] = {
            "observed_gain": gain,
            "ci95": [ci_low if gain else -0.01, 0.05],
            "panel_observed_gain": {str(panel): gain for panel in exposure.PANELS},
        }
    return {"rows": rows, "bootstrap": bootstrap}


def test_closeout_classification_covers_four_outcomes() -> None:
    assert exposure._classification(
        _classification_payload(global_gain=0.02, oracle_gain=0.02, shape_gain=0.0, ci_low=0.001, recovered_panels=2)
    )["conclusion"] == "global_exposure_semantic_recovery_candidate"
    assert exposure._classification(
        _classification_payload(global_gain=0.02, oracle_gain=0.02, shape_gain=0.0, ci_low=0.001, recovered_panels=0)
    )["conclusion"] == "global_exposure_explains_oracle_only"
    assert exposure._classification(
        _classification_payload(global_gain=0.0, oracle_gain=0.02, shape_gain=0.02, ci_low=0.001, recovered_panels=0)
    )["conclusion"] == "field_shape_dominant_after_power_match"
    assert exposure._classification(
        _classification_payload(global_gain=0.0, oracle_gain=0.0, shape_gain=0.0, ci_low=-0.01, recovered_panels=0)
    )["conclusion"] == "loss_exposure_inconclusive"


def test_oracle_is_marked_nonphysical_and_storage_gate_is_fail_closed() -> None:
    assert "S0-per-sample-oracle" in exposure.BASE_VARIANTS
    assert exposure.STORAGE_LIMIT_BYTES == 2 * 1024**3
    assert "S0-field-gain-reference" in exposure.BASE_VARIANTS
