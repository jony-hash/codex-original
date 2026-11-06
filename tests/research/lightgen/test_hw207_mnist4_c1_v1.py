"""Checks of the C1 input and field measurement functions."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "research" / "lightgen"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location("hw207_c1", SCRIPT_DIR / "hw207_mnist4_c1_v1.py")
c1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c1)
runner_spec = importlib.util.spec_from_file_location("hw207_c1_runner", SCRIPT_DIR / "hw207_mnist4_c1_run_v1.py")
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)
audit_spec = importlib.util.spec_from_file_location("hw207_c1_audit", SCRIPT_DIR / "hw207_mnist4_c1_audit_v1.py")
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)
readout_spec = importlib.util.spec_from_file_location("hw207_c1_readout", SCRIPT_DIR / "hw207_mnist4_c1_readout_v1.py")
readout = importlib.util.module_from_spec(readout_spec)
readout_spec.loader.exec_module(readout)


def test_roi_values_and_inset():
    intensity = torch.zeros((1, 100, 100))
    intensity[0, 10, 10] = 1
    intensity[0, 10, 60] = 2
    intensity[0, 60, 10] = 3
    intensity[0, 60, 60] = 4
    intensity[0, 0, 0] = 5
    assert c1.roi_scores(intensity, 0).tolist() == [[6, 2, 3, 4]]
    assert c1.roi_scores(intensity, 5).tolist() == [[1, 2, 3, 4]]


def test_identity_and_propagation_energy():
    amplitude = torch.zeros((1, 100, 100))
    amplitude[:, 40:60, 30:70] = 1
    assert torch.equal(c1.propagate(amplitude, 0, 128).real, amplitude)
    output = c1.propagate(amplitude, 0.001, 128)
    assert output.shape == (1, 128, 128)
    assert torch.allclose(output.abs().square().sum(), amplitude.square().sum(), rtol=1e-5)
    with pytest.raises(ValueError):
        c1.propagate(amplitude, 0.001, 101)


def test_propagation_preserves_requested_precision():
    amplitude = torch.zeros((1, 100, 100), dtype=torch.float64)
    amplitude[:, 40:60, 30:70] = 1
    assert c1.propagate(amplitude, 0, 128).dtype == torch.complex128
    assert c1.propagate(amplitude, 0.001, 128).dtype == torch.complex128
    assert c1.propagate(amplitude.float(), 0.001, 128).dtype == torch.complex64


def test_window_and_precision_comparisons():
    amplitude = torch.zeros((2, 100, 100))
    amplitude[:, 40:60, 30:70] = 1
    image_error, roi_error = c1.window_comparison(amplitude, 0.001, 128, 256)
    assert image_error.shape == roi_error.shape == (2,)
    assert torch.isfinite(image_error).all() and torch.isfinite(roi_error).all()
    complex_error, energy_error = c1.precision_comparison(amplitude, 0.001, 128)
    assert complex_error.max() < 1e-5
    assert energy_error.max() < 1e-5
    with pytest.raises(ValueError):
        c1.window_comparison(torch.zeros_like(amplitude), 0.001, 128, 256)


def test_chunk_identity_and_atomic_no_overwrite(tmp_path):
    images = torch.zeros((2, 28, 28), dtype=torch.uint8)
    images[0, 3:9, 4:11] = 255
    images[1, 15:20, 16:21] = 255
    labels, rows = torch.tensor([0, 1]), torch.tensor([22, 35])
    chunk = c1.produce_chunk(images, labels, rows, True, 0, 100, torch.device("cpu"), 1)
    assert torch.equal(chunk["rows"], rows)
    assert torch.equal(chunk["labels"], labels)
    torch.testing.assert_close(chunk["input_energy"], chunk["full_energy"])
    torch.testing.assert_close(chunk["crop_energy"], chunk["roi_a"].sum(-1))
    path = tmp_path / "chunk.pt"
    c1.save_chunk(path, chunk)
    saved = torch.load(path, weights_only=True)
    torch.testing.assert_close(saved["intensity"], chunk["intensity"])
    checked = c1.checked_chunk(path, True, 0, 100, labels, rows)
    torch.testing.assert_close(checked["roi_a"], chunk["roi_a"])
    with pytest.raises(ValueError, match="identity"):
        c1.checked_chunk(path, True, 0, 100, labels, rows.flip(0))
    damaged = dict(chunk)
    damaged["roi_a"] = damaged["roi_a"] + 1
    torch.save(damaged, tmp_path / "damaged.pt")
    with pytest.raises(ValueError, match="ROI"):
        c1.checked_chunk(tmp_path / "damaged.pt", True, 0, 100, labels, rows)
    with pytest.raises(FileExistsError):
        c1.save_chunk(path, chunk)


def test_numerical_probes_reject_sign_and_axis_errors():
    result = c1.numerical_probes(torch.device("cpu"))
    assert result["passed"], result
    assert result["production_reverse_fault_l2"] > 1e-5
    assert result["production_axis_fault_l2"] > 1e-5


def test_readout_normalization_and_real_updates():
    intensity = torch.zeros((8, 100, 100))
    intensity[1:, 30:70, 30:70] = torch.arange(1, 8)[:, None, None]
    raw = runner.readout_features(intensity, 25, False)
    normalized = runner.readout_features(intensity, 25, True)
    assert raw.shape == normalized.shape == (8, 625)
    assert torch.count_nonzero(normalized[0]) == 0
    torch.testing.assert_close(normalized[1], normalized[7])
    assert raw[7].sum() > raw[1].sum()
    labels = torch.arange(4).repeat(2)
    model = runner.readout_fit(raw, labels, 11, 0.0, 3, 2, steps=3)
    assert torch.isfinite(model.weight).all()


def test_independent_roi_audit_from_committed_chunk(tmp_path):
    images = torch.zeros((4, 28, 28), dtype=torch.uint8)
    images[:, 5:10, 5:10] = 255
    labels = torch.arange(4)
    rows = torch.arange(4) + 10
    chunk = c1.produce_chunk(images, labels, rows, True, 0, 100, torch.device("cpu"), 2, "test")
    directory = tmp_path / "chunks" / "binary" / "0" / "test"
    directory.mkdir(parents=True)
    c1.save_chunk(directory / "000000.pt", chunk)
    result = audit.aggregate(tmp_path, "binary", True, 0, 100, "test", labels, rows)
    assert result["count"] == 4
    assert result["roi"]["a"]["coverage"] == 1
    assert result["roi"]["a"]["accuracy"] == 0.25
    assert result["counts"] == [1, 1, 1, 1]


def test_full_readout_candidate_records_validation_history():
    inputs = torch.eye(4).repeat(2, 1)
    labels = torch.arange(4).repeat(2)
    candidate = readout.fit_candidate(inputs, labels, inputs, labels, 11, 0.0, torch.device("cpu"))
    assert len(candidate["history"]) == 30
    assert candidate["history"][-1]["validation"]["accuracy"] > 0.25
    assert all(torch.isfinite(value).all() for value in candidate["state"].values())


def test_measure_resumes_committed_chunks_and_rejects_mismatched_rows(tmp_path):
    images = torch.zeros((4, 28, 28), dtype=torch.uint8)
    images[:, 5:10, 5:10] = 255
    labels = torch.arange(4)
    rows = torch.arange(4) + 10
    samples = {role: (images, labels, rows) for role in ("train", "validation", "test")}
    windows = {str(d): (100 if d == 0 else None) for d in (0, 0.0005, 0.001, 0.002)}
    receipt = {"windows": {"gray": windows, "binary": windows}}
    path = tmp_path / "chunks" / "gray" / "0" / "train" / "000000.pt"
    path.parent.mkdir(parents=True)
    first = c1.produce_chunk(images, labels, rows, False, 0, 100, torch.device("cpu"), 2, "train")
    c1.save_chunk(path, first)
    before = path.stat().st_mtime_ns
    runner.measure(samples, receipt, tmp_path, torch.device("cpu"))
    assert path.stat().st_mtime_ns == before
    assert (tmp_path / "chunks" / "binary" / "0" / "test" / "000000.pt").exists()
    invalid = dict(samples)
    invalid["train"] = (images, labels, rows.flip(0))
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.measure(invalid, receipt, tmp_path, torch.device("cpu"))
