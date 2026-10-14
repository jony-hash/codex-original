from __future__ import annotations

import math

import pytest
import torch

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_foreground_shape_localization_v1 as audit,
)
from scripts.research.lightgen import (
    execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as energy_task,
)


def _target() -> torch.Tensor:
    result = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    result[0, 0, 0, :4] = torch.tensor([0.8, 0.2, 0.001, 0.003])
    return result


def _image(values: tuple[float, float, float, float]) -> torch.Tensor:
    result = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    result[0, 0, 0, :4] = torch.tensor(values, dtype=torch.float64)
    return result


def _primary_rows(*, improve: bool) -> list[dict[str, object]]:
    target = _target()
    imperfect = _image((0.65, 0.25, 0.1, 0.0))
    left, right = (imperfect, target) if improve else (target, imperfect)
    left_metrics = audit._measure_components(left, target)
    right_metrics = audit._measure_components(right, target)
    rows: list[dict[str, object]] = []
    for label in (2, 4):
        for _ in range(30):
            rows.append(
                {
                    "label": label,
                    "role": "confirmation",
                    "A6700_q_fg": float(left_metrics["q_fg"][0]),
                    "B6700_q_fg": float(right_metrics["q_fg"][0]),
                }
            )
    # These rows must not enter the scientific estimand.
    rows.extend(
        [
            {"label": 2, "role": "fit", "A6700_q_fg": 10.0, "B6700_q_fg": 0.0},
            {"label": 0, "role": "confirmation", "A6700_q_fg": 10.0, "B6700_q_fg": 0.0},
        ]
    )
    return rows


def test_foreground_and_background_components_partition_raw_shape_and_mse() -> None:
    target = _target()
    measured = audit._measure_components(_image((0.65, 0.25, 0.1, 0.0)), target)

    assert measured["q_raw200"].item() == pytest.approx(
        measured["q_fg"].item() + measured["q_bg"].item(), rel=1e-12, abs=1e-14
    )
    assert measured["raw_mse"].item() == pytest.approx(
        measured["raw_mse_fg_contribution"].item() + measured["raw_mse_bg_contribution"].item(),
        rel=1e-12,
        abs=1e-14,
    )
    assert measured["foreground_area_fraction"].item() == pytest.approx(2 / 40000)


def test_background_only_change_does_not_change_foreground_component() -> None:
    target = _target()
    image_a = _image((0.75, 0.15, 0.004, 0.0))
    image_b = _image((0.75, 0.15, 0.002, 0.002))
    a = audit._measure_components(image_a, target)
    b = audit._measure_components(image_b, target)

    assert a["q_fg"].item() == pytest.approx(b["q_fg"].item(), abs=1e-15)
    assert b["q_bg"].item() < a["q_bg"].item()


def test_foreground_can_improve_while_background_error_worsens() -> None:
    target = _target()
    image_a = _image((0.65, 0.25, 0.1, 0.0))
    image_b = _image((0.68, 0.17, 0.15, 0.0))
    a = audit._measure_components(image_a, target)
    b = audit._measure_components(image_b, target)

    assert b["q_fg"].item() < a["q_fg"].item()
    assert b["q_bg"].item() > a["q_bg"].item()


def test_normalized_shape_components_ignore_uniform_brightness_scaling() -> None:
    target = _target()
    image = _image((0.6, 0.3, 0.08, 0.02))
    original = audit._measure_components(image, target)
    scaled = audit._measure_components(image * 0.2, target)

    for name in ("q_raw200", "q_fg", "q_bg", "normalized_fg_underfill_mass", "normalized_fg_overfill_mass", "normalized_bg_generated_mass"):
        assert scaled[name].item() == pytest.approx(original[name].item(), rel=1e-12, abs=1e-14)


@pytest.mark.parametrize(
    ("improve", "expected"),
    [
        (True, "FOREGROUND_SHAPE_ERROR_REDUCED"),
        (False, "FOREGROUND_SHAPE_ERROR_INCREASED"),
    ],
)
def test_measured_primary_effect_reaches_decision_and_excludes_other_rows(improve: bool, expected: str) -> None:
    decision = audit._primary_decision(_primary_rows(improve=improve))

    assert decision["classification"] == expected
    assert decision["primary_rows"] == 60
    assert decision["by_class"]["2"]["rows"] == 30
    assert decision["by_class"]["4"]["rows"] == 30


def test_primary_decision_reports_within_tolerance_for_identical_measured_images() -> None:
    target = _target()
    metrics = audit._measure_components(target, target)
    rows = [
        {"label": label, "role": "confirmation", "A6700_q_fg": float(metrics["q_fg"][0]), "B6700_q_fg": float(metrics["q_fg"][0])}
        for label in (2, 4)
        for _ in range(30)
    ]

    result = audit._primary_decision(rows)

    assert result["classification"] == "FOREGROUND_SHAPE_CHANGE_WITHIN_TOLERANCE"
    assert result["difference_q_fg_B6700_minus_A6700"] == 0.0


def test_final_report_renders_from_the_machine_result_schema() -> None:
    decision = audit._primary_decision(_primary_rows(improve=True))
    decision["csv_recheck"] = {"primary_classification_recomputed": decision["classification"]}
    rendered = audit._render_report(
        {
            "decision": decision,
            "population": {},
            "validation": {"csv_roundtrip": decision["csv_recheck"]},
        }
    )

    assert "## 结果与状态" in rendered
    assert "## 科学认识" in rendered
    assert "## 目标缺口" in rendered
    assert "## 工程完成与验证" in rendered
    assert "## 决策影响与下一步" in rendered


@pytest.mark.parametrize("bad_value", [math.nan, -0.1])
def test_nonfinite_and_negative_intensities_are_rejected(bad_value: float) -> None:
    image = _image((bad_value, 0.2, 0.0, 0.0))

    with pytest.raises(audit.ForegroundLocalizationError):
        audit._measure_components(image, _target())


def test_zero_mass_and_empty_target_regions_are_rejected() -> None:
    with pytest.raises(audit.ForegroundLocalizationError, match="zero intensity mass"):
        audit._measure_components(torch.zeros_like(_target()), _target())
    with pytest.raises(audit.ForegroundLocalizationError, match="foreground or background is empty"):
        audit._measure_components(_image((0.5, 0.5, 0.0, 0.0)), torch.full_like(_target(), 0.2))


def test_population_row_identity_and_production_view_mismatches_are_rejected() -> None:
    source_rows = list(range(audit.TOTAL_ROWS))
    labels = [label for repeat in (20, 30) for label in range(10) for _ in range(repeat)]
    roles = ["fit"] * audit.FIT_ROWS + ["confirmation"] * audit.CONFIRM_ROWS
    population = {
        "source_row_indices": source_rows,
        "labels": labels,
        "roles": roles,
        "role_positions": {"fit": [0, 200], "confirmation": [200, 500]},
    }
    audit._validate_population(population)
    population["source_row_indices"][-1] = population["source_row_indices"][0]
    with pytest.raises(audit.ForegroundLocalizationError, match="duplicate row identities"):
        audit._validate_population(population)

    readout = {
        "source_indices": torch.arange(audit.TOTAL_ROWS, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    mismatched = dict(readout)
    mismatched["source_indices"] = readout["source_indices"].clone()
    mismatched["source_indices"][0] += 1
    population["source_row_indices"] = list(range(audit.TOTAL_ROWS))
    with pytest.raises(audit.ForegroundLocalizationError, match="source row IDs differ"):
        audit._verify_source_row_identity(mismatched, population, "test")

    raw = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    replay = energy_task.diagnostic.canonical.cmos_view(raw)
    with pytest.raises(audit.ForegroundLocalizationError, match="production view differs"):
        audit._verify_production_view({"raw_cmos": raw, "evaluator_view": replay + 0.1}, "test")
