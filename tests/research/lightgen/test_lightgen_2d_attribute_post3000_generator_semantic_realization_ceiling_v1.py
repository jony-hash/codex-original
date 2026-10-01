from __future__ import annotations

import math

import torch

from scripts.research.lightgen.execute_lightgen_2d_attribute_post3000_generator_semantic_realization_ceiling_v1 import CLASSIFICATION_TWO_EVALUATOR, _classify, _normalize_rms, _stable_descent, _three_way_geometry


def _step_geometry(*, output_alignment: float = 0.8, phase_alignment: float = 0.5) -> dict[str, object]:
    return {
        "output_space": {
            "common_alignment_index": output_alignment,
            "pairwise_cosines": {"E_S_E_I": 0.2, "E_S_E_C": 0.1, "E_I_E_C": 0.1},
        },
        "phase_space": {
            "common_alignment_index": phase_alignment,
            "pairwise_cosines": {"E_S_E_I": 0.8, "E_S_E_C": -0.2, "E_I_E_C": -0.2},
        },
        "historical_SI": {"cos_g_C_vs_v_SI": -0.1},
    }


def _probe(*, c_descent: bool) -> dict[str, object]:
    def records(metrics: dict[str, float]) -> list[dict[str, object]]:
        return [
            {
                "epsilon": epsilon,
                "plus": {"deltas": metrics},
                "minus": {"deltas": {name: -value for name, value in metrics.items()}},
            }
            for epsilon in (1.0e-4, 3.0e-4, 1.0e-3)
        ]

    return {
        "directions": {
            "d_SI": records({"L_S": -0.1, "L_I": -0.1, "L_C": 0.1}),
            "d_C": records({"L_S": 0.1, "L_I": 0.1, "L_C": -0.1 if c_descent else 0.1}),
            "d_SIC": records({"L_S": -0.1, "L_I": -0.1, "L_C": -0.1 if c_descent else 0.1}),
        }
    }


def test_normalize_rms_uses_exact_rms_phase_convention() -> None:
    value = torch.arange(1.0, 9.0).reshape(2, 2, 2)
    direction = _normalize_rms(value, "test direction")
    assert math.isclose(float(direction.square().mean().sqrt()), 1.0, rel_tol=1.0e-6)
    assert torch.equal(direction > 0.0, torch.zeros_like(direction, dtype=torch.bool))


def test_three_way_geometry_reports_pairwise_and_common_alignment() -> None:
    values = {
        "E_S": torch.tensor([1.0, 0.0]),
        "E_I": torch.tensor([1.0, 0.0]),
        "E_C": torch.tensor([0.0, 1.0]),
    }
    result, resultant, units = _three_way_geometry(values, "test")
    assert result["common_direction_defined"] is True
    assert math.isclose(result["pairwise_cosines"]["E_S_E_I"], 1.0)
    assert torch.equal(units["E_S"], torch.tensor([1.0, 0.0]))
    assert torch.equal(resultant, torch.tensor([2.0, 1.0]))


def test_stable_descent_requires_all_fixed_epsilons() -> None:
    probe = _probe(c_descent=True)
    assert _stable_descent(probe, "d_SI", "L_S") is True
    assert _stable_descent(probe, "d_SIC", "L_C") is True
    probe["directions"]["d_SIC"][1]["plus"]["deltas"]["L_C"] = 0.1
    assert _stable_descent(probe, "d_SIC", "L_C") is False


def test_classification_identifies_two_evaluator_ceiling() -> None:
    classification, details, successor = _classify(_step_geometry(), _probe(c_descent=False), {"projection": {"g_rec": {"projected": False}, "g_op": {"projected": False}}})
    assert classification == CLASSIFICATION_TWO_EVALUATOR
    assert details["flags"]["step3000_d_SI_improves_E_S_and_E_I"] is True
    assert successor == "DEFINE_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1"
