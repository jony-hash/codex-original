from __future__ import annotations

from pathlib import Path

import pytest

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research import electronic_o0_prior_cycle_composition as ccf


def _values(l00: float, l10: float, l01: float, l11: float) -> dict[str, dict[str, float]]:
    return {
        name: {metric: value for metric in ccf.METRICS}
        for name, value in {
            "source_source": l00,
            "final_source": l10,
            "source_final": l01,
            "final_final": l11,
        }.items()
    }


def _route(classification: str) -> dict:
    total = 1.0
    if classification == "encoder_dominant_cycle_drift":
        effect = {"encoder_shapley": 0.8, "decoder_shapley": 0.1, "interaction_effect": 0.1}
    elif classification == "decoder_dominant_cycle_drift":
        effect = {"encoder_shapley": 0.1, "decoder_shapley": 0.8, "interaction_effect": 0.1}
    elif classification == "composition_interaction_cycle_drift":
        effect = {"encoder_shapley": 0.2, "decoder_shapley": 0.2, "interaction_effect": 0.6}
    elif classification == "component_attribution_underdetermined":
        effect = {"encoder_shapley": -0.1, "decoder_shapley": 0.2, "interaction_effect": -0.1}
    else:
        effect = {"encoder_shapley": 0.4, "decoder_shapley": 0.35, "interaction_effect": 0.25}
    historical_effect = {"total_drift": total, **effect}
    bank_effect = {"total_drift": total, **effect}
    return {
        "historical": {"effects": {metric: historical_effect.copy() for metric in ccf.METRICS}},
        "stability": [
            {"effects": {metric: bank_effect.copy() for metric in ccf.METRICS}}
            for _ in range(4)
        ],
    }


def test_config_and_cli_contract_are_registered() -> None:
    root = Path(__file__).parents[2]
    config = ccf._config(root / "configs/research/electronic_o0_prior_cycle_composition_v1.yaml")
    assert config["cycle_composition_seed"] == 8761
    assert config["factorial_components"] == list(ccf.COMPONENTS)
    for action in ("validate", "preflight", "audit", "report", "closeout"):
        args = _parser().parse_args(
            ["research", "electronic-o0-prior-cycle-composition", action, "--run-manifest", "manifest.yaml"]
        )
        assert args.research_action == action


def test_factorial_identity_and_shapley_decomposition() -> None:
    effects = ccf._effect_metrics(_values(1.0, 2.0, 3.0, 6.0))
    row = effects["cycle"]
    assert row["encoder_effect"] == pytest.approx(1.0)
    assert row["decoder_effect"] == pytest.approx(2.0)
    assert row["interaction_effect"] == pytest.approx(2.0)
    assert row["total_drift"] == pytest.approx(5.0)
    assert row["encoder_shapley"] + row["decoder_shapley"] == pytest.approx(row["total_drift"])
    assert ccf._factorial_identity(_values(1.0, 2.0, 3.0, 6.0))


def test_classification_routes_and_underdetermined_are_fail_closed() -> None:
    for expected in (
        "encoder_dominant_cycle_drift",
        "decoder_dominant_cycle_drift",
        "composition_interaction_cycle_drift",
        "mixed_component_cycle_drift",
    ):
        classification, _, decision = ccf._classify({"swae": _route(expected), "sced": _route(expected)})
        assert classification == expected
        assert decision["alignment_estimator_independent"] is True
    under = {"swae": _route("component_attribution_underdetermined"), "sced": _route("component_attribution_underdetermined")}
    classification, next_action, _ = ccf._classify(under)
    assert classification == "cycle_composition_attribution_underdetermined"
    assert next_action == "close_prior_native_attribution_family"


def test_null_swap_tolerance_is_strict() -> None:
    assert ccf.NULL_EFFECT_TOLERANCE == 1.0e-10
    with pytest.raises(ValueError):
        ccf._effect_metrics({"source_source": {"cycle": 1.0}})
