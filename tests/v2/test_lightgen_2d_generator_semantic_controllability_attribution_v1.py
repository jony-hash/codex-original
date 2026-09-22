from __future__ import annotations

from scripts.research.execute_lightgen_2d_generator_semantic_controllability_attribution_v1 import (
    ATTRIBUTION_CONFLICT,
    ATTRIBUTION_CONTROLLABILITY,
    ATTRIBUTION_MISMATCH,
    ATTRIBUTION_MIXED,
    PERTURBATION_EPSILONS,
    _classify,
)


def _step(*, train_smooth: float, rec_smooth: float) -> dict[str, object]:
    return {
        "cosines": {
            "C_train_smooth": train_smooth,
            "C_rec_smooth": rec_smooth,
            "C_rec_train": 0.2,
        }
    }


def _direction(*, smooth_plus: float, smooth_minus: float, rec_plus: float = -0.1, rec_minus: float = 0.1) -> list[dict[str, object]]:
    return [
        {
            "epsilon": epsilon,
            "plus": {"deltas": {"L_rec": rec_plus, "L_sem_train": -0.1, "L_sem_smooth": smooth_plus}},
            "minus": {"deltas": {"L_rec": rec_minus, "L_sem_train": 0.1, "L_sem_smooth": smooth_minus}},
            "central_slope_per_relative_parameter_norm": {
                "L_rec": 100.0,
                "L_sem_train": 100.0,
                "L_sem_smooth": abs(smooth_plus - smooth_minus) * 10.0,
            },
        }
        for epsilon in PERTURBATION_EPSILONS
    ]


def _probes(*, train: tuple[float, float], smooth: tuple[float, float], reconstruction: tuple[float, float]) -> dict[str, object]:
    return {"directions": {"semantic_train": _direction(smooth_plus=train[0], smooth_minus=train[1]), "semantic_smooth": _direction(smooth_plus=smooth[0], smooth_minus=smooth[1]), "reconstruction": _direction(smooth_plus=reconstruction[0], smooth_minus=reconstruction[1])}}


def test_attribution_classification_uses_directional_signs() -> None:
    primary, _, secondary, _, _ = _classify(
        _step(train_smooth=-0.2, rec_smooth=0.2),
        _step(train_smooth=-0.1, rec_smooth=0.3),
        _probes(train=(0.1, -0.1), smooth=(-0.1, 0.1), reconstruction=(-0.1, 0.1)),
        {"classification": "GENERATOR_SEMANTIC_RECOVERY_NOT_SUPPORTED"},
    )
    assert primary == ATTRIBUTION_MISMATCH
    assert secondary == "NONE"

    primary, _, secondary, _, _ = _classify(
        _step(train_smooth=0.4, rec_smooth=0.2),
        _step(train_smooth=0.3, rec_smooth=0.2),
        _probes(train=(-0.1, 0.1), smooth=(-0.01, 0.01), reconstruction=(-0.1, 0.1)),
        {"classification": "GENERATOR_SEMANTIC_RECOVERY_NOT_SUPPORTED"},
    )
    assert primary == ATTRIBUTION_CONTROLLABILITY
    assert secondary == "NONE"

    primary, _, secondary, _, _ = _classify(
        _step(train_smooth=0.4, rec_smooth=-0.3),
        _step(train_smooth=0.3, rec_smooth=-0.2),
        _probes(train=(-0.1, 0.1), smooth=(-0.01, 0.01), reconstruction=(0.1, -0.1)),
        {"classification": "GENERATOR_SEMANTIC_RECOVERY_NOT_SUPPORTED"},
    )
    assert primary == ATTRIBUTION_CONTROLLABILITY
    assert secondary == ATTRIBUTION_CONFLICT

    primary, _, secondary, _, _ = _classify(
        _step(train_smooth=0.1, rec_smooth=0.2),
        _step(train_smooth=0.2, rec_smooth=0.3),
        _probes(train=(0.1, -0.1), smooth=(-0.01, 0.01), reconstruction=(-0.1, 0.1)),
        {"classification": "GENERATOR_SEMANTIC_RECOVERY_NOT_SUPPORTED"},
    )
    assert primary == ATTRIBUTION_MIXED
    assert secondary == "NONE"
