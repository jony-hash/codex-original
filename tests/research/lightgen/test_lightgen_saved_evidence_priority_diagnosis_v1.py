from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ldm_project.v2.application.research.lightgen.canonical_semantics import cmos_view
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    PixelMLPConfirmationEvaluatorV1,
)
from scripts.research.lightgen import audit_lightgen_2d_generator_identity_stroke_localization_v1 as identity
from scripts.research.lightgen import audit_lightgen_saved_evidence_priority_diagnosis_v1 as diagnosis


def test_camera_replay_and_saturated_error_partition_use_the_camera_producer() -> None:
    wavelength_m = 532e-9
    exposure_s = 1.0
    qe = 0.5
    full_well_e = 10.0
    requested_electrons = np.array([[5.0, 10.0], [15.0, 2.0]], dtype=np.float64)
    photon_energy = 6.62607015e-34 * 299_792_458.0 / wavelength_m
    power = requested_electrons * photon_energy / (exposure_s * qe)

    electrons, task_output = diagnosis._camera_replay(power, wavelength_m, exposure_s, qe, full_well_e)
    np.testing.assert_allclose(electrons, requested_electrons, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(task_output, np.array([[0.5, 1.0], [1.0, 0.2]]), rtol=0, atol=1e-14)

    target = np.array([[0.1, 0.7], [0.2, 0.4]], dtype=np.float64)
    partition = diagnosis._camera_error_partition(target, electrons, task_output, full_well_e)
    expected_error = np.square(task_output - target)
    saturated = electrons >= full_well_e
    assert partition["at_or_above_full_well_pixel_fraction"] == 0.5
    assert math.isclose(partition["at_or_above_full_well_sq_error_per_image"], float(expected_error[saturated].sum()))
    assert math.isclose(partition["below_full_well_sq_error_per_image"], float(expected_error[~saturated].sum()))
    assert math.isclose(partition["task_mse"], float(expected_error.mean()))


def test_camera_candidates_must_keep_the_same_rows_roles_labels_and_targets() -> None:
    reference = {
        "rows": np.array([10, 11]),
        "labels": np.array([2, 3]),
        "role": np.array(["fit", "confirmation"]),
        "target": np.zeros((2, 1, 2, 2), dtype=np.float32),
    }
    diagnosis._require_same_camera_population(reference, dict(reference), "same")
    wrong = dict(reference)
    wrong["target"] = reference["target"].copy()
    wrong["target"][1, 0, 0, 0] = 1.0
    with pytest.raises(diagnosis.EvidenceError, match="target differs"):
        diagnosis._require_same_camera_population(reference, wrong, "wrong target")


def test_sorted_power_projection_matches_original_bisection_for_feasible_and_short_budgets() -> None:
    target = np.array([[0.9, 0.7, 0.5, 0.2]], dtype=np.float64)
    full_well = 10.0
    sufficient = full_well * float(target.sum())
    insufficient = full_well * 0.6

    assert diagnosis._power_lower_bound_sorted(target, sufficient, full_well) == 0.0
    assert diagnosis._power_lower_bound_bisection(target, sufficient, full_well) == 0.0
    sorted_bound = diagnosis._power_lower_bound_sorted(target, insufficient, full_well)
    bisection_bound = diagnosis._power_lower_bound_bisection(target, insufficient, full_well)
    assert sorted_bound > 0.0
    assert math.isclose(sorted_bound, bisection_bound, rel_tol=0.0, abs_tol=1e-12)


def test_energy_match_is_preclip_and_rejects_zero_mass() -> None:
    a = torch.full((1, 1, 200, 200), 0.5, dtype=torch.float32)
    b = torch.full((1, 1, 200, 200), 2.0, dtype=torch.float32)
    zero = torch.zeros_like(b)
    scale = diagnosis._energy_match_scale(a, b)
    assert scale == 0.25

    production_matched = cmos_view(b * scale)
    wrong_scale_after_clip = cmos_view(b) * scale
    assert not torch.allclose(production_matched, wrong_scale_after_clip)
    with pytest.raises(diagnosis.EvidenceError, match="zero-mass"):
        diagnosis._energy_match_scale(a, zero)


def test_cpu_evaluator_is_frozen_eval_and_matches_production_cmos_view() -> None:
    torch.manual_seed(7)
    evaluator = PixelMLPConfirmationEvaluatorV1().cpu().freeze().eval()
    raw = torch.rand((2, 1, 200, 200), dtype=torch.float32) * 1.4
    before = {name: value.detach().clone() for name, value in evaluator.state_dict().items()}

    actual = diagnosis._run_evaluator(evaluator, raw)
    with torch.no_grad():
        expected = evaluator(cmos_view(raw))

    assert not evaluator.training
    assert torch.allclose(actual, expected.cpu(), rtol=0.0, atol=0.0)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in evaluator.parameters())
    assert all(torch.equal(before[name], value) for name, value in evaluator.state_dict().items())


def test_selected_pair_table_rejects_missing_and_duplicate_samples() -> None:
    def row(sample_key: str, population: str) -> dict[str, str]:
        return {
            "sample_key": sample_key,
            "population": population,
            "target_class": "3",
            "a_scenario": "A",
            "b_scenario": "B",
            "a_position": "0",
            "b_position": "0",
            "a_e_d_prediction": "3",
            "b_e_d_prediction": "3",
            "a_e_d_identity_margin": "1",
            "b_e_d_identity_margin": "1",
            "e_d_transition": "both_correct",
        }

    expected = {"empirical_dev": 1, "conditional_sample": 1}
    complete = [row("one", "empirical_dev"), row("two", "conditional_sample")]
    diagnosis._validate_selected_pair_table(
        complete, expected_count=2, expected_population_counts=expected
    )
    with pytest.raises(diagnosis.EvidenceError, match="2 unique complete samples"):
        diagnosis._validate_selected_pair_table(
            complete[:1], expected_count=2, expected_population_counts=expected
        )
    with pytest.raises(diagnosis.EvidenceError, match="2 unique complete samples"):
        diagnosis._validate_selected_pair_table(
            [complete[0], dict(complete[0])], expected_count=2,
            expected_population_counts={"empirical_dev": 2},
        )


def _identity_csv_row(target_class: int) -> dict[str, object]:
    row: dict[str, object] = {
        "population": "empirical_dev",
        "label": target_class,
        "requested_class": "",
        "e_d_identity_margin": 1.0,
    }
    for evaluator in identity.EVALUATORS:
        row[f"{evaluator.lower()}_prediction"] = target_class
    return row


def _write_identity_chunk(path: Path, scenario: str, source_row_index: int) -> None:
    raw = torch.full((identity.RAW_CHUNK_ROWS, 1, 200, 200), 0.25, dtype=torch.float32)
    logits: dict[str, torch.Tensor] = {}
    for evaluator in identity.EVALUATORS:
        values = torch.zeros((identity.RAW_CHUNK_ROWS, 10), dtype=torch.float32)
        values[:, 3] = 1.0
        logits[evaluator] = values
    torch.save(
        {
            "task_id": "M0_GENERATOR_CLOUD_FOREGROUND_DEV_MULTISEED_V1",
            "scenario": scenario,
            "raw_cmos": raw,
            "evaluator_view": cmos_view(raw),
            "logits": logits,
            "positions": torch.arange(identity.RAW_CHUNK_ROWS),
            "source_row_indices": torch.full((identity.RAW_CHUNK_ROWS,), source_row_index),
            "labels": torch.full((identity.RAW_CHUNK_ROWS,), 3),
            "roles": ["dev"] * identity.RAW_CHUNK_ROWS,
            "target_slm_intensity": torch.zeros((identity.RAW_CHUNK_ROWS, 1, 200, 200)),
        },
        path,
    )


def _identity_pair(tmp_path: Path, source_row_index: int) -> dict[str, object]:
    target_class = 3
    pair: dict[str, object] = {
        "population": "empirical_dev",
        "sample_key": "empirical_dev:seed:row",
        "target_class": target_class,
        "source_row_index": source_row_index,
        "sample_seed": "",
        "draw_index": "",
        "a_scenario": "scenario_A",
        "b_scenario": "scenario_B",
        "a_position": 0,
        "b_position": 0,
        "_a_csv_row": _identity_csv_row(target_class),
        "_b_csv_row": _identity_csv_row(target_class),
    }
    root = tmp_path / "evidence"
    for scenario, source in (("scenario_A", source_row_index), ("scenario_B", source_row_index)):
        path = root / "readouts" / scenario / "chunk_0000.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_identity_chunk(path, scenario, source)
    return pair


def test_production_raw_loader_rejects_row_identity_mismatch(tmp_path: Path) -> None:
    pair = _identity_pair(tmp_path, source_row_index=101)
    evidence_dir = tmp_path / "evidence"
    valid = identity._read_selected_raw(evidence_dir, [pair])
    assert set(valid) == {(str(pair["sample_key"]), "A"), (str(pair["sample_key"]), "B")}
    assert tuple(valid[(str(pair["sample_key"]), "A")]["saved_e_d_logits"].shape) == (10,)

    b_path = evidence_dir / "readouts" / "scenario_B" / "chunk_0000.pt"
    _write_identity_chunk(b_path, "scenario_B", source_row_index=999)
    with pytest.raises(identity.EvidenceError, match="source row differs"):
        identity._read_selected_raw(evidence_dir, [pair])
