from __future__ import annotations

import csv
import math
from fractions import Fraction

import pytest
import torch

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_shape_energy_counterfactual_v1 as audit,
)
from scripts.research.lightgen import (
    execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as energy_task,
)


def _halves(left: float, right: float, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    result = torch.empty((1, 1, 200, 200), dtype=dtype)
    result[..., :100] = left
    result[..., 100:] = right
    return result


def _metadata() -> dict[str, object]:
    return {"camera_plane": {"pixel_pitch_m": [2.5e-6, 2.5e-6]}}


def test_clipping_attenuation_improvement_disappears_when_energy_is_restored() -> None:
    target = _halves(0.8, 0.2)
    image_a = _halves(2.0, 0.5)
    image_b = image_a * 0.4

    cells, _ = audit._measure_batch(image_a, image_b, target, torch.tensor([0]), _metadata(), 1.0)

    expected = float(Fraction(8, 153))
    assert cells["AA"]["q"].item() == pytest.approx(expected, abs=1e-14)
    assert cells["BB"]["q"].item() == pytest.approx(0.0, abs=1e-14)
    assert cells["BA"]["q"].item() == pytest.approx(expected, abs=1e-14)
    assert cells["AB"]["q"].item() == pytest.approx(0.0, abs=1e-14)
    buggy_unrestored = audit._classify_difference(cells["BB"]["q"].item() - cells["AA"]["q"].item(), 0.0, expected)
    assert buggy_unrestored["classification"] == "FIXED_ENERGY_Q_IMPROVEMENT_RETAINED"
    correct = audit._classify_difference(cells["BA"]["q"].item() - cells["AA"]["q"].item(), expected, expected)
    assert correct["classification"] == "FIXED_ENERGY_Q_CHANGE_WITHIN_TOLERANCE"


def test_fixed_energy_true_shape_change_reaches_the_primary_direction() -> None:
    target = _halves(0.8, 0.2)
    image_a = _halves(0.6, 0.4)
    image_b = target.clone()

    cells, energies = audit._measure_batch(image_a, image_b, target, torch.tensor([0]), _metadata(), 1.0)
    decision = audit._classify_difference(
        float(cells["BA"]["q"].item() - cells["AA"]["q"].item()),
        float(cells["BA"]["q"].mean()),
        float(cells["AA"]["q"].mean()),
    )

    assert energies["energy_aa"].item() == pytest.approx(energies["energy_bb"].item())
    assert cells["AA"]["q"].item() == pytest.approx(float(Fraction(2, 17)), abs=1e-14)
    assert cells["BA"]["q"].item() == pytest.approx(0.0, abs=1e-14)
    assert decision["classification"] == "FIXED_ENERGY_Q_IMPROVEMENT_RETAINED"


def test_unclipped_brightness_scaling_leaves_shape_q_unchanged() -> None:
    target = _halves(0.8, 0.2)
    image = _halves(0.6, 0.4)

    q_before = audit._normalized_shape_q(image, target)
    q_after = audit._normalized_shape_q(image * 0.4, target)

    assert torch.allclose(q_before, q_after, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    ("difference", "expected"),
    [
        (-1.0e-4, "FIXED_ENERGY_Q_IMPROVEMENT_RETAINED"),
        (1.0e-4, "FIXED_ENERGY_Q_IMPROVEMENT_REVERSED"),
        (1.0e-8, "FIXED_ENERGY_Q_CHANGE_WITHIN_TOLERANCE"),
    ],
)
def test_csv_population_consumer_drives_each_final_decision(difference: float, expected: str, tmp_path) -> None:
    labels = [digit for _ in range(50) for digit in range(10)]
    roles = ["fit"] * audit.FIT_ROWS + ["confirmation"] * audit.CONFIRM_ROWS
    source_indices = list(range(2_000, 2_000 + audit.TOTAL_ROWS))
    population = {"source_row_indices": source_indices, "labels": labels, "roles": roles}
    rows = []
    for index, (label, role) in enumerate(zip(labels, roles, strict=True)):
        q_aa = 0.1
        q_ba = q_aa + difference
        q_ab = q_aa - difference
        q_bb = q_aa
        d_shape = 0.5 * ((q_ba - q_aa) + (q_bb - q_ab))
        d_energy = 0.5 * ((q_ab - q_aa) + (q_bb - q_ba))
        rows.append(
            {
                "position": index,
                "source_row_index": source_indices[index],
                "label": label,
                "role": role,
                "AA_q": q_aa,
                "AB_q": q_ab,
                "BA_q": q_ba,
                "BB_q": q_bb,
                "d_fixed_a_energy": q_ba - q_aa,
                "d_shape_symmetric": d_shape,
                "d_energy_symmetric": d_energy,
                "q_total_bb_minus_aa": q_bb - q_aa,
            }
        )
    path = tmp_path / "paired_rows.csv"
    audit._write_csv(path, rows)
    expected_decision = audit._classify_difference(difference, 0.1 + difference, 0.1)

    result = audit._recheck_csv(path, population, expected_decision)

    assert result["passed"] is True
    assert result["primary_decision"]["classification"] == expected
    assert result["primary_confirmation_rows"] == 300


@pytest.mark.parametrize("bad_value", [0.0, -0.1, math.nan])
def test_zero_negative_and_nonfinite_source_images_are_rejected(bad_value: float) -> None:
    invalid = _halves(bad_value, 0.5)
    if bad_value == 0.0:
        invalid.zero_()

    with pytest.raises(audit.CounterfactualError):
        audit._four_images(invalid, _halves(0.5, 0.5), 1.0)


def test_wrong_source_rows_and_saved_view_are_rejected() -> None:
    population, readouts = saved_evidence_fixture()
    readouts["B6700"]["source_indices"] = readouts["B6700"]["source_indices"].clone()
    readouts["B6700"]["source_indices"][0] += 1
    with pytest.raises(saved_evidence_error(), match="source row indices differ"):
        audit.saved_evidence._verify_readouts(population, readouts)
    readouts["B6700"]["source_indices"] = torch.arange(audit.TOTAL_ROWS, dtype=torch.float64) + 0.5
    with pytest.raises(audit.CounterfactualError, match="not stored as integers"):
        audit._verify_row_identity(readouts["B6700"], population, "B6700")

    broken = {
        "raw_cmos": _halves(0.5, 0.5),
        "evaluator_view": torch.zeros((1, 1, 28, 28), dtype=torch.float64),
    }
    with pytest.raises(audit.CounterfactualError, match="production view"):
        audit._verify_production_view(broken, "synthetic")


def saved_evidence_fixture():
    from scripts.research.lightgen import audit_lightgen_2d_generator_saved_evidence_v1 as saved

    rows = torch.arange(saved.TOTAL_ROWS)
    labels = torch.tensor([digit for _ in range(50) for digit in range(10)])
    shared = {
        "z": torch.zeros((saved.TOTAL_ROWS, 100), dtype=torch.complex64),
        "raw_cmos": torch.ones((saved.TOTAL_ROWS, 1, 200, 200), dtype=torch.float32),
        "evaluator_view": torch.ones((saved.TOTAL_ROWS, 1, 28, 28), dtype=torch.float32),
        "target_slm_intensity": torch.ones((saved.TOTAL_ROWS, 1, 200, 200), dtype=torch.float32),
        "power_by_plane": torch.zeros((saved.TOTAL_ROWS, 5), dtype=torch.float64),
        "source_indices": rows.clone(),
        "labels": labels.clone(),
        **{f"{evaluator.lower()}_logits": torch.zeros((saved.TOTAL_ROWS, 10)) for evaluator in saved.EVALUATORS},
        "new_phase_plane_power": torch.zeros((saved.TOTAL_ROWS, 2), dtype=torch.float64),
        "plane_metadata": {"camera_plane": {"pixel_pitch_m": [2.5e-6, 2.5e-6]}},
    }
    population = {"source_row_indices": rows.tolist(), "labels": labels.tolist()}
    return population, {name: dict(shared) for name in saved.ENDPOINTS}


def saved_evidence_error():
    from scripts.research.lightgen.audit_lightgen_2d_generator_saved_evidence_v1 import SavedEvidenceAuditError

    return SavedEvidenceAuditError
