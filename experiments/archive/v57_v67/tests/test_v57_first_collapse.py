from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v57_first_collapse_complex_field.py"


def _module():
    spec = importlib.util.spec_from_file_location("v57_first_collapse", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_distance_preserves_complex_phase():
    module = _module()
    field = torch.tensor([[1.0 + 2.0j, 2.0 - 1.0j]], dtype=torch.complex64)
    rotated = field * torch.exp(torch.tensor(0.7j))
    distance = module._pair_distance(field, rotated)
    assert distance["complex"] == pytest.approx(0.0, abs=1e-6)
    assert distance["amplitude"] == pytest.approx(0.0, abs=1e-6)
    assert distance["phase_mae_rad"] == pytest.approx(0.0, abs=1e-6)


def _payloads():
    module = _module()
    labels = torch.arange(10).repeat_interleave(2)
    fields = {}
    for plane_index, plane in enumerate(module.PLANE_ORDER):
        values = torch.zeros((20, 1, 2, 2), dtype=torch.complex64)
        for sample, label in enumerate(labels):
            values[sample] = (float(label) + 1.0) * (plane_index + 1) + 1j * (sample + 1)
        fields[plane] = {
            "real": values.real,
            "imag": values.imag,
            "operation": "propagation",
        }
    return (
        {"status": "manual-review-required", "fields": fields},
        {"field_chain_status": "valid", "diagnostic_only": True, "accepted_summary_written": False},
        {"class_labels": labels},
    )


def test_streaming_report_covers_all_planes_and_writes_progress(tmp_path: Path):
    module = _module()
    fields, provenance, labels = _payloads()
    fields_path = tmp_path / "fields.pt"
    provenance_path = tmp_path / "provenance.json"
    labels_path = tmp_path / "labels.pt"
    output_path = tmp_path / "report.json"
    torch.save(fields, fields_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    torch.save(labels, labels_path)

    report = module.run(fields_path, provenance_path, labels_path, output_path)

    assert report["status"] == "manual-review-required"
    assert len(report["planes"]) == len(module.PLANE_ORDER)
    assert report["planes"][0]["same_class_pair_count"] == 10
    assert report["planes"][0]["cross_class_pair_count"] == 180
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(written["planes"]) == len(module.PLANE_ORDER)
    assert written["accepted_summary_written"] is False


def test_missing_class_is_harness_invalid(tmp_path: Path):
    module = _module()
    fields, provenance, labels = _payloads()
    labels["class_labels"][0] = 1
    fields_path = tmp_path / "fields.pt"
    provenance_path = tmp_path / "provenance.json"
    labels_path = tmp_path / "labels.pt"
    output_path = tmp_path / "report.json"
    torch.save(fields, fields_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    torch.save(labels, labels_path)

    report = module.run(fields_path, provenance_path, labels_path, output_path)

    assert report["status"] == "harness-invalid"
    assert "exactly two samples" in report["reasons"][0]
