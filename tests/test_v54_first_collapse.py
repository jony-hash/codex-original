import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v54", ROOT / "tools" / "diagnose_v54_first_collapse_complex_field.py")
V54 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V54)


def _record(values, *, upstream=None, operation="test"):
    z = torch.as_tensor(values, dtype=torch.complex64)
    return {
        "real": z.real,
        "imag": z.imag,
        "sampling": {"shape": list(z.shape[-2:]), "pixel_size_x": 1.0, "pixel_size_y": 1.0, "fov_x": float(z.shape[-1]), "fov_y": float(z.shape[-2])},
        "phase_available": True,
        "upstream": upstream,
        "operation": operation,
    }


def test_global_phase_alignment_and_phase_metric():
    a = torch.ones(1, 2, 2, dtype=torch.complex64)
    b = a * torch.exp(torch.tensor(1j * 0.7))
    result = V54._pair_metrics(a, b)
    assert result["complex_relative_l2"] < 1e-6
    assert result["amplitude_corr"] == 0.0


def test_complex_loader_preserves_imaginary_component():
    record = _record([[[[3.0 + 4.0j]]]])
    loaded = V54._complex(record)
    assert loaded.is_complex()
    assert torch.allclose(loaded.imag, torch.full_like(loaded.imag, 4.0))


def test_class_coverage_is_strict():
    bundle = {"class_labels": torch.tensor([0] * 20)}
    try:
        V54._labels(bundle)
    except ValueError as exc:
        assert "expected classes" in str(exc)
    else:
        raise AssertionError("missing class coverage must fail")


def test_validation_rejects_ols_discontinuity():
    fields = {name: _record(torch.ones(20, 1, 2, 2), upstream=None) for name in V54.PLANES}
    fields["ols_input"] = _record(torch.ones(20, 1, 2, 2), upstream="encoder_output")
    fields["ols_output"] = _record(torch.zeros(20, 1, 2, 2), upstream="ols_input")
    provenance = {"field_chain_status": "valid", "full_chain_available": True, "accepted_summary_written": False}
    errors = V54._validate({"fields": fields}, provenance, {"class_labels": torch.tensor([i // 2 for i in range(20)])})
    assert any("OLS input/output" in error for error in errors)


def test_streaming_report_flushes_and_writes_csv(tmp_path):
    labels = torch.tensor([i // 2 for i in range(20)])
    fields = {}
    for index, name in enumerate(V54.PLANES):
        upstream = V54.PLANES[index - 1] if index else None
        fields[name] = _record(torch.ones(20, 1, 2, 2), upstream=upstream)
    provenance = {"field_chain_status": "valid", "full_chain_available": True, "accepted_summary_written": False}
    field_path, provenance_path, tensor_path = [tmp_path / name for name in ("fields.pt", "provenance.json", "tensors.pt")]
    torch.save({"fields": fields}, field_path)
    provenance_path.write_text(json.dumps(provenance))
    torch.save({"class_labels": labels}, tensor_path)
    output = tmp_path / "report.json"
    report = V54.run_diagnostic(field_path, provenance_path, tensor_path, output, tmp_path / "progress.json")
    assert report["progress"]["complete"] is True
    assert output.with_suffix(".csv").exists()
