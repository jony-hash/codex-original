"""Pure decision checks for the N4 endpoint-attribution evaluator."""

from __future__ import annotations

from scripts.research import execute_lightgen_2d_n4_endpoint_attribution_evaluation_v1 as execution


def _pair(mean: float, lower: float, upper: float, material: bool) -> dict[str, object]:
    return {"observed_delta": mean, "material": material, "bootstrap": {"ci_lower": lower, "ci_upper": upper}}


def test_attribution_classifies_generator_coadaptation_when_extra_increment_is_small() -> None:
    result = execution._classify_attribution(
        {
            "C1_minus_C0": _pair(0.12, 0.10, 0.14, True),
            "C2_minus_C0": _pair(0.13, 0.11, 0.15, True),
            "C2_minus_C1": _pair(0.01, -0.01, 0.03, False),
        }
    )

    assert result["classification"] == "GENERATOR_COADAPTATION_DOMINANT"


def test_attribution_classifies_extra_phase_or_mixed_cases() -> None:
    extra = execution._classify_attribution(
        {
            "C1_minus_C0": _pair(0.01, -0.01, 0.03, False),
            "C2_minus_C0": _pair(0.12, 0.10, 0.14, True),
            "C2_minus_C1": _pair(0.11, 0.09, 0.13, True),
        }
    )
    mixed = execution._classify_attribution(
        {
            "C1_minus_C0": _pair(0.12, 0.10, 0.14, True),
            "C2_minus_C0": _pair(0.14, 0.01, 0.20, False),
            "C2_minus_C1": _pair(0.02, -0.02, 0.08, False),
        }
    )

    assert extra["classification"] == "EXTRA_PHASE_CONTROL_DOMINANT"
    assert mixed["classification"] == "ATTRIBUTION_MIXED_OR_UNRESOLVED"
