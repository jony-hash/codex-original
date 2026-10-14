from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
import torch

from scripts.research.lightgen import audit_lightgen_2d_generator_saved_evidence_v1 as audit


def test_classifier_rows_match_known_ce_margin_prediction_and_wrong_class_ties() -> None:
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    logits = torch.tensor(
        [
            [3.0, 1.0, 1.0, -2.0],
            [2.0, 2.0, 0.0, -1.0],
            [0.0, 2.0, 2.0, 2.0],
        ],
        dtype=torch.float64,
    )
    readout = {"labels": labels, **{f"{name.lower()}_logits": logits for name in audit.EVALUATORS}}

    rows = audit._classification_rows(readout)

    assert rows["e_d_predictions"].tolist() == [0, 0, 1]
    assert rows["e_d_margin_float64"].tolist() == pytest.approx([2.0, 0.0, -2.0])
    expected_ce = torch.logsumexp(logits, dim=1) - logits.gather(1, labels[:, None]).squeeze(1)
    assert torch.allclose(rows["e_d_ce_float64"], expected_ce)
    assert rows["e_d_max_wrong_logit_tie_count"].tolist() == [2, 1, 3]


def test_normalized_q_is_brightness_invariant_while_integrated_energy_changes() -> None:
    target = torch.zeros((1, 1, 28, 28), dtype=torch.float64)
    target[..., 5:20, 11:16] = 1.0
    generated = target.clone()
    generated[..., 10:13, 11:16] = 0.5

    q_before = audit._normalized_q(generated, target)
    q_scaled = audit._normalized_q(generated * 0.4, target)
    energy_before = audit._integrated_energy(generated, 2.5e-12)
    energy_scaled = audit._integrated_energy(generated * 0.4, 2.5e-12)

    assert torch.allclose(q_before, q_scaled, rtol=0.0, atol=1e-14)
    assert energy_scaled.item() == pytest.approx(0.4 * energy_before.item())


def test_zero_mass_rejects_undefined_q() -> None:
    image = torch.zeros((1, 1, 28, 28), dtype=torch.float64)
    with pytest.raises(audit.SavedEvidenceAuditError, match="Q is undefined"):
        audit._normalized_q(image, image)


def _full_population_arrays() -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor, list[str]]:
    labels = torch.tensor([digit for _ in range(50) for digit in range(10)], dtype=torch.long)
    roles = ["fit"] * audit.FIT_ROWS + ["confirmation"] * audit.CONFIRM_ROWS
    base: dict[str, torch.Tensor] = {}
    for metric in audit.METRIC_KEYS:
        if metric.endswith("_predictions"):
            base[metric] = labels.clone()
        elif metric == "raw_zero_image" or metric == "raw_constant_image":
            base[metric] = torch.zeros(audit.TOTAL_ROWS, dtype=torch.bool)
        else:
            base[metric] = torch.ones(audit.TOTAL_ROWS, dtype=torch.float64)
    base["e_d_margin_float64"] = torch.zeros(audit.TOTAL_ROWS, dtype=torch.float64)
    base["brightness_normalized_stroke_error_Q"] = torch.ones(audit.TOTAL_ROWS, dtype=torch.float64)
    base["integrated_energy"] = torch.ones(audit.TOTAL_ROWS, dtype=torch.float64)
    after = {key: value.clone() for key, value in base.items()}
    after["e_d_margin_float64"][200:210] = 1.0
    after["brightness_normalized_stroke_error_Q"][200:210] = 0.5
    after["integrated_energy"][200:210] = 0.8
    after["e_d_predictions"][210] = (int(labels[210]) + 1) % 10
    after["e_d_predictions"][211] = (int(labels[211]) + 1) % 10
    after["e_d_predictions"][12] = labels[12]
    return {"A4700": base, "B6700": after}, labels, roles


def test_complete_population_pairing_covers_roles_classes_and_report() -> None:
    arrays, labels, roles = _full_population_arrays()
    pair = audit._pair_analysis("A4700", "B6700", arrays, labels, roles)
    decision = audit._interpret({"A4700_to_B6700": pair})

    assert pair["by_role"]["fit"]["rows"] == 200
    assert pair["by_role"]["confirmation"]["rows"] == 300
    assert [pair["by_class"][str(i)]["confirmation"]["rows"] for i in range(10)] == [30] * 10
    assert pair["by_role"]["confirmation"]["E_D_correctness_transitions"] == {
        "rows": 300,
        "wrong_to_correct": 0,
        "correct_to_wrong": 2,
        "correct_to_correct": 298,
        "wrong_to_wrong": 0,
    }
    assert decision["classification"] == "JOINT_SCORE_AND_STROKE_IMPROVEMENT_WITH_POSSIBLE_ENERGY_TRADEOFF"
    assert decision["training_recommendation"] == "NONE"


def test_paired_csv_writes_margin_ties_for_each_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class CaptureBuffer(io.StringIO):
        def close(self) -> None:
            self.flush()

    arrays, labels, roles = _full_population_arrays()
    for endpoint in audit.ENDPOINTS:
        arrays[endpoint] = {key: value.clone() for key, value in arrays["A4700"].items()}
        for evaluator_index, tie_key in enumerate(audit.TIE_KEYS):
            arrays[endpoint][tie_key] = torch.full((audit.TOTAL_ROWS,), evaluator_index + 1, dtype=torch.long)
    readouts = {"A4700": {"source_indices": torch.arange(audit.TOTAL_ROWS), "labels": labels}}
    buffer = CaptureBuffer()
    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: buffer)

    audit._write_paired_rows(Path("in-memory.csv"), readouts, arrays, roles)

    records = list(csv.DictReader(io.StringIO(buffer.getvalue())))
    assert len(records) == audit.TOTAL_ROWS
    assert records[0]["A4700_e_s_max_wrong_logit_tie_count"] == "1"
    assert records[0]["A4700_e_d_max_wrong_logit_tie_count"] == "4"
    assert records[0]["A6700_e_d_max_wrong_logit_tie_count"] == "4"
    assert records[0]["B6700_e_d_max_wrong_logit_tie_count"] == "4"


def test_duplicate_or_misassigned_population_rows_are_rejected() -> None:
    rows = list(range(audit.TOTAL_ROWS))
    rows[-1] = rows[0]
    population = {
        "source_row_indices": rows,
        "labels": [digit for _ in range(50) for digit in range(10)],
        "roles": ["fit"] * audit.FIT_ROWS + ["confirmation"] * audit.CONFIRM_ROWS,
        "role_positions": {"fit": [0, audit.FIT_ROWS], "confirmation": [audit.FIT_ROWS, audit.TOTAL_ROWS]},
        "fit_rows": audit.FIT_ROWS,
        "confirmation_rows": audit.CONFIRM_ROWS,
        "canonical_train_not_claimed_unseen": True,
        "source_generator_adam_step": 4_700,
        "final_generator_adam_step": 6_700,
        "updates_per_arm": audit.UPDATES_PER_ARM,
        "energy_scale_coefficients": [1.0] * audit.FIT_ROWS,
        "energy_scale_source_rows": list(range(audit.FIT_ROWS)),
        "energy_scale_labels": [digit for _ in range(20) for digit in range(10)],
    }
    report = {
        "task_id": audit.SOURCE_TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": "MIXED_OR_INSUFFICIENT",
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "source_step": 4_700,
        "updates_per_arm": audit.UPDATES_PER_ARM,
        "final_generator_adam_step": 6_700,
        "stop": True,
        "population": population,
        "preflight": {"status": "READY"},
        "source_A4700_replay": {"passed": True},
    }
    population["task_id"] = audit.SOURCE_TASK_ID

    with pytest.raises(audit.SavedEvidenceAuditError, match="duplicates"):
        audit._verify_attempt(population, report)


def _minimal_readouts() -> tuple[dict[str, object], dict[str, dict[str, torch.Tensor | dict[str, object]]]]:
    labels = torch.tensor([digit for _ in range(50) for digit in range(10)], dtype=torch.long)
    rows = torch.arange(audit.TOTAL_ROWS, dtype=torch.long)
    shared = {
        "z": torch.zeros((audit.TOTAL_ROWS, 100), dtype=torch.complex64),
        "raw_cmos": torch.zeros((audit.TOTAL_ROWS, 1, 200, 200), dtype=torch.float32),
        "evaluator_view": torch.zeros((audit.TOTAL_ROWS, 1, 28, 28), dtype=torch.float32),
        "target_slm_intensity": torch.zeros((audit.TOTAL_ROWS, 1, 200, 200), dtype=torch.float32),
        "power_by_plane": torch.zeros((audit.TOTAL_ROWS, 5), dtype=torch.float64),
        "source_indices": rows,
        "labels": labels,
        **{f"{evaluator.lower()}_logits": torch.zeros((audit.TOTAL_ROWS, 10), dtype=torch.float32) for evaluator in audit.EVALUATORS},
        "new_phase_plane_power": torch.zeros((audit.TOTAL_ROWS, 2), dtype=torch.float64),
        "plane_metadata": {"camera_plane": {"pixel_pitch_m": [2.5e-6, 2.5e-6]}},
    }
    return {"source_row_indices": rows.tolist(), "labels": labels.tolist()}, {name: dict(shared) for name in audit.ENDPOINTS}


@pytest.mark.parametrize("field", ["source_indices", "labels", "raw_cmos"])
def test_readout_population_mismatch_and_nonfinite_values_are_rejected(field: str) -> None:
    population, readouts = _minimal_readouts()
    if field == "source_indices":
        readouts["B6700"][field] = readouts["B6700"][field].clone()
        readouts["B6700"][field][0] = 1
        message = "source row indices differ"
    elif field == "labels":
        readouts["B6700"][field] = readouts["B6700"][field].clone()
        readouts["B6700"][field][0] = 1
        message = "labels differ"
    else:
        readouts["B6700"][field] = readouts["B6700"][field].clone()
        readouts["B6700"][field][0, 0, 0, 0] = float("nan")
        message = "nonfinite"

    with pytest.raises(audit.SavedEvidenceAuditError, match=message):
        audit._verify_readouts(population, readouts)


def test_audit_refuses_existing_output_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path("existing-source")
    output = Path("existing-output")
    monkeypatch.setattr(Path, "is_dir", lambda self: self == source)
    monkeypatch.setattr(Path, "exists", lambda self: self == output)

    with pytest.raises(audit.SavedEvidenceAuditError, match="refusing to overwrite"):
        audit._run_audit(source, output)
