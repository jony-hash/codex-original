from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.research.lightgen.diagnose_lightgen_g1_te_inserted_path_v1 import (
    DEFAULT_CONFIG,
    DiagnosticStop,
    _assert_unchanged,
    _compose_labeled_gallery,
    _load_config,
    _load_transfer_table,
    _metric_row,
    _json_transfer_record,
    _require_transfer_observation,
    _transfer_field,
    _transfer_observation,
    _validate_preflight_receipt,
    _write_json_new,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec


def test_saved_paper_transfer_matches_power_ratio_and_reference_phase() -> None:
    config = _load_config(DEFAULT_CONFIG)
    source, records = _load_transfer_table(config)
    assert source["measurement"]["status"] == "valid"
    assert source["bare_power"] > 0.0
    for thickness in (10.0, 50.0):
        for state in ("solid_25C", "quasi_liquid_80C"):
            record = records[thickness, state]
            q = record["q"]
            assert math.isclose(abs(q) ** 2, record["power_ratio"], rel_tol=0.0, abs_tol=1e-9)
            assert 0.0 <= record["R"] <= 1.0
            assert 0.0 <= record["T"] <= 1.0
            assert 0.0 <= record["A_flux"] <= 1.0


def test_production_transfer_acceptance_rejects_power_ratio_used_as_field() -> None:
    plane = PlaneSpec("test-plane", (4, 4), (1.0e-6, 1.0e-6), 532.0e-9, 0.0, "test-frame")
    provenance = OperationProvenance("test-source", None, None, {})
    base_tensor = torch.arange(1, 17, dtype=torch.float64).reshape(1, 1, 4, 4).to(torch.complex128)
    base = ComplexField(base_tensor, plane, provenance)
    baseline = {"entry": base, "camera": base, "raw": base_tensor.abs().square()}
    q = 0.3 + 0.4j
    expected_power = abs(q) ** 2
    correct = ComplexField(base_tensor * torch.as_tensor(q, dtype=torch.complex128), plane, provenance)
    correct_actual = {"entry": correct, "camera": correct, "raw": correct.tensor.abs().square()}
    accepted = _transfer_observation(baseline, correct_actual, q, expected_power)
    assert accepted["passed"] is True
    _require_transfer_observation(accepted, "correct complex field transfer")

    wrong_q = complex(abs(q) ** 2 * np.cos(np.angle(q)), abs(q) ** 2 * np.sin(np.angle(q)))
    wrong_field = _transfer_field(base, wrong_q, "wrong-power-as-field-negative-control")
    wrong_actual = {"entry": wrong_field, "camera": wrong_field, "raw": wrong_field.tensor.abs().square()}
    rejected = _transfer_observation(baseline, wrong_actual, q, expected_power)
    assert rejected["passed"] is False
    with pytest.raises(DiagnosticStop, match="transfer acceptance rejected"):
        _require_transfer_observation(rejected, "wrong power used as field amplitude")


def test_frozen_state_guard_rejects_parameter_drift() -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    snapshot = {key: value.detach().clone() for key, value in module.state_dict().items()}
    _assert_unchanged(module, snapshot, "linear test module")
    with torch.no_grad():
        module.weight[0, 0].add_(1.0)
    with pytest.raises(DiagnosticStop, match="frozen linear test module changed"):
        _assert_unchanged(module, snapshot, "linear test module")


def test_preflight_receipt_rejects_changed_sample_or_proxy_identity() -> None:
    identity = {"sample_indices": [0, 1000], "proxy_q": {"solid": [0.3, -0.1]}}
    receipt = {"status": "PREFLIGHT_PASSED", "evidence_tier": "diagnostic_only",
               "model_identity": identity}
    _validate_preflight_receipt(receipt, identity)
    changed = {**identity, "sample_indices": [0, 1001]}
    with pytest.raises(DiagnosticStop, match="preflight receipt does not match"):
        _validate_preflight_receipt(receipt, changed)


def test_gallery_consumer_maps_each_28_by_28_frame_without_row_broadcast() -> None:
    frames = np.arange(100 * 28 * 28, dtype=np.float32).reshape(100, 28, 28)
    labels = [label for label in range(10) for _ in range(10)]
    gallery, positions, error = _compose_labeled_gallery(frames, labels, columns=10)
    assert gallery.shape == (280, 280)
    assert len(positions) == 100
    assert error == 0.0
    for position in positions:
        index = position["sample_position"]
        row, column = position["row"], position["column"]
        tile = gallery[row * 28:(row + 1) * 28, column * 28:(column + 1) * 28]
        np.testing.assert_array_equal(tile, frames[index])
    with pytest.raises(DiagnosticStop, match="one 2D frame and label"):
        _compose_labeled_gallery(frames[:, 0], labels, columns=10)


def test_preflight_rows_drop_array_payloads_before_json_serialization() -> None:
    record = {
        "condition": "solid_25C",
        "q_real": 0.3,
        "view28": np.zeros((28, 28), dtype=np.float32),
        "raw_cmos_w_m2": np.zeros((64, 64), dtype=np.float64),
        "camera_field": np.zeros((64, 64), dtype=np.complex128),
        "generator_entry_field": np.zeros((64, 64), dtype=np.complex128),
    }
    try:
        json.dumps(record, allow_nan=False)
    except TypeError:
        pass
    else:
        raise AssertionError("the unsanitized production record should expose the serialization defect")

    serialized = json.dumps(_metric_row({}, record), allow_nan=False)
    assert json.loads(serialized) == {"condition": "solid_25C", "q_real": 0.3}


def test_measurement_transfer_record_serializes_complex_field_as_components() -> None:
    transfer = {"q": 0.3 + 0.4j, "T": 0.25, "state": "solid_25C"}
    serialized = json.dumps(_json_transfer_record(transfer), allow_nan=False)
    assert json.loads(serialized) == {
        "T": 0.25, "state": "solid_25C", "q_real": 0.3, "q_imag": 0.4,
    }



def test_json_writer_validates_before_opening_output(monkeypatch) -> None:
    opened = []

    def forbidden_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("output should not be opened before JSON validation")

    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(TypeError):
        _write_json_new(Path("unused.json"), {"unsupported_complex": 1.0 + 2.0j})
    assert not opened
