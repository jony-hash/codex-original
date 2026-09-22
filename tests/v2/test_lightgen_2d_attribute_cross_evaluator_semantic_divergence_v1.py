from __future__ import annotations

import torch

from scripts.research.execute_lightgen_2d_attribute_cross_evaluator_semantic_divergence_v1 import (
    CLASSIFICATION_OFF_MANIFOLD,
    CLASSIFICATION_SHARED,
    ROWS_PER_CLASS_PER_HALF,
    _classify,
    _direction_bundle,
    _select_diagnostic_bank,
    _stable_descent,
)


def _probes(*, joint_both: bool, cross: bool) -> dict[str, object]:
    def direction(smooth: bool, independent: bool) -> list[dict[str, object]]:
        records = []
        for epsilon in (1.0e-4, 3.0e-4, 1.0e-3):
            records.append(
                {
                    "epsilon": epsilon,
                    "plus": {"deltas": {"L_S": -0.1 if smooth else 0.1, "L_I": -0.1 if independent else 0.1, "L_rec": 0.0, "L_op": 0.0}},
                    "minus": {"deltas": {"L_S": 0.1 if smooth else -0.1, "L_I": 0.1 if independent else -0.1, "L_rec": 0.0, "L_op": 0.0}},
                }
            )
        return records

    return {
        "directions": {
            "d_S": direction(True, not cross),
            "d_I": direction(not cross, True),
            "d_joint": direction(joint_both, joint_both),
        },
        "joint_bisector": {"status": "DEFINED"},
    }


def _step(cos_h: float, cos_g: float) -> dict[str, object]:
    return {"output_space": {"cos_h_S_h_I": cos_h}, "parameter_space": {"cos_g_S_g_I": cos_g}}


def _interpolation() -> dict[str, object]:
    return {"summary": {"cos_at_lambda_0.25": 0.8, "cos_at_lambda_1.0": 0.1}}


def test_diagnostic_bank_is_deterministic_disjoint_and_class_balanced() -> None:
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    train_indices = torch.arange(55_000, dtype=torch.long)
    first = _select_diagnostic_bank(train_indices, labels)
    second = _select_diagnostic_bank(train_indices, labels)
    for name in ("a_indices", "a_labels", "b_indices", "b_labels"):
        assert torch.equal(first[name], second[name])
    assert first["a_indices"].numel() == 10 * ROWS_PER_CLASS_PER_HALF
    assert first["b_indices"].numel() == 10 * ROWS_PER_CLASS_PER_HALF
    assert torch.unique(torch.cat((first["a_indices"], first["b_indices"]))).numel() == 40
    assert torch.bincount(first["a_labels"], minlength=10).tolist() == [ROWS_PER_CLASS_PER_HALF] * 10
    assert torch.bincount(first["b_labels"], minlength=10).tolist() == [ROWS_PER_CLASS_PER_HALF] * 10


def test_joint_bisector_is_equal_weight_and_detects_degeneracy() -> None:
    gradients = {"g_S": torch.ones((2, 1, 2, 2)), "g_I": torch.ones((2, 1, 2, 2))}
    directions, joint = _direction_bundle(gradients)
    assert joint["status"] == "DEFINED"
    assert torch.isclose(directions["d_joint"].square().mean().sqrt(), torch.tensor(1.0))
    _, degenerate = _direction_bundle({"g_S": gradients["g_S"], "g_I": -gradients["g_I"]})
    assert degenerate["status"] == "JOINT_BISECTOR_DEGENERATE"


def test_stable_descent_requires_all_fixed_epsilons() -> None:
    probes = _probes(joint_both=True, cross=False)
    assert _stable_descent(probes, "d_S", "L_S") is True
    assert _stable_descent(probes, "d_S", "L_I") is True
    probes["directions"]["d_S"][1]["plus"]["deltas"]["L_S"] = 0.1
    assert _stable_descent(probes, "d_S", "L_S") is False


def test_classification_accepts_shared_direction_without_cosine_cutoff() -> None:
    classification, details, _, _ = _classify(_step(0.01, -0.01), _step(0.02, -0.02), _probes(joint_both=True, cross=False), _probes(joint_both=True, cross=False), _interpolation(), _interpolation())
    assert classification == CLASSIFICATION_SHARED
    assert details["flags"]["step0_joint_direction_improves_both"] is True


def test_classification_accepts_evaluator_specific_cross_effects() -> None:
    classification, _, _, _ = _classify(_step(0.2, 0.3), _step(0.2, 0.3), _probes(joint_both=False, cross=True), _probes(joint_both=False, cross=True), _interpolation(), _interpolation())
    assert classification == CLASSIFICATION_OFF_MANIFOLD
