from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ldm_project.optics.reference import measure_circular_pupil_first_zero


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "validate_v58_3_b0_relay_acceptance.py"


def _module():
    spec = importlib.util.spec_from_file_location("v58_3_relay", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v58_3_requires_full_formal_sampling_and_reports_measured_psf():
    report = _module().run(ROOT / "configs" / "validate_v55_coherent_relay.yaml", "cpu")
    assert report["status"] == "physical-forward-model-valid"
    assert report["validation_revision"] == "v58.3-full-fov-b0-relay-acceptance"
    formal = report["formal_full_fov_validation"]
    assert formal["input_sampling"] == {"shape": [200, 200], "pixel_size_m": 2.5e-6, "fov_m": 500e-6}
    assert formal["output_sampling"]["shape"] == [100, 100]
    assert formal["output_sampling"]["fov_m"] == pytest.approx(200e-6)
    assert report["measured_pupil_psf_validation"]["wide_psf"]["measured_first_zero_index"] > 0
    assert report["measured_pupil_psf_validation"]["pass_frequency_cycles_m"] == 80_000.0
    assert report["measured_pupil_psf_validation"]["stop_frequency_cycles_m"] == 120_000.0
    assert all(report["independent_analytic_validation"]["fault_injection"].values())
    assert report["report_contract"]["front_abcd"] == {"A": -0.4, "B": 0.0, "C": 0.0}
    assert report["report_contract"]["fault_keys"] == sorted(report["independent_analytic_validation"]["fault_injection"])
    assert report["sampling_and_independence_audit"]["production_has_no_generic_resize"] is True


def test_measured_pupil_zero_fails_for_insufficient_radial_extent():
    import pytest
    import torch

    with pytest.raises(ValueError, match="no measurable"):
        measure_circular_pupil_first_zero(
            source_size=200, input_pixel_size=2.5e-6, magnification=-0.4,
            wavelength=532e-9, pupil_diameter=0.02375, pupil_focal_length=0.0892857142857143,
            radial_step=0.05e-6, radial_extent=0.5e-6, device=torch.device("cpu"),
        )


def test_faults_keep_inversion_and_magnification_independent():
    module = _module()
    source = TOOL.read_text(encoding="utf-8")
    assert "wrong_inversion =" in source
    assert source.count('_metrics(wrong_magnification, expected)') == 1
    assert all(module._fault_injection(__import__("torch").device("cpu")).values())
