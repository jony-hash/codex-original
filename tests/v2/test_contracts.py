from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from ldm_project.v2.config import V2Config
from ldm_project.v2.contracts import (
    AmplitudeCommand,
    ComplexField,
    DisplayFrame,
    ExposureCalibration,
    IntensityFrame,
    OperationProvenance,
    PlaneSpec,
    render_fixed_exposure,
)


def _plane() -> PlaneSpec:
    return PlaneSpec("slm", (4, 4), (2.5e-6, 2.5e-6), 532e-9, 0.0, "slm")


def _config() -> dict[str, object]:
    return {
        "schema_version": 2,
        "system": {"kind": "formal_amplitude_finite_crop_identity"},
        "run": {"run_name": "unit", "device": "cpu"},
    }


def test_plane_records_full_physical_coordinates() -> None:
    plane = _plane()
    assert plane.fov_m == (1.0e-5, 1.0e-5)
    assert plane.to_dict()["wavelength_m"] == 532e-9


def test_complex_field_and_intensity_are_typed() -> None:
    plane = _plane()
    provenance = OperationProvenance("identity", "source", None)
    field = ComplexField(torch.ones(1, 1, 4, 4, dtype=torch.complex64), plane, provenance)
    assert isinstance(field.intensity, IntensityFrame)
    with pytest.raises(TypeError):
        ComplexField(torch.ones(1, 1, 4, 4), plane, provenance)


def test_commands_and_display_cannot_be_confused() -> None:
    plane = _plane()
    provenance = OperationProvenance("detector", "camera_field", None)
    raw = IntensityFrame(torch.ones(1, 1, 4, 4), plane, provenance, "locked")
    display = render_fixed_exposure(raw, ExposureCalibration("locked", 2.0))
    assert isinstance(display, DisplayFrame)
    with pytest.raises(TypeError):
        AmplitudeCommand(torch.ones(1, 1, 4, 4, dtype=torch.complex64), plane, "slm-v1")


def test_propagation_requires_distance() -> None:
    with pytest.raises(ValueError, match="propagation_distance_m"):
        OperationProvenance("propagate_asm", "slm", None)


def test_v2_config_rejects_unknown_and_geometry_drift() -> None:
    with pytest.raises(ValidationError):
        V2Config.model_validate({**_config(), "unknown": True})
    payload = _config()
    payload["system"] = {"kind": "formal_amplitude_finite_crop_identity", "layer_spacing_m": 1e-3}
    with pytest.raises(ValidationError, match="layer_spacing_m"):
        V2Config.model_validate(payload)


def test_small_diagnostic_system_is_never_formal() -> None:
    config = V2Config.model_validate({"schema_version": 2, "system": {"kind": "diagnostic_amplitude_finite_crop_identity", "slm_shape": [8, 8], "metasurface_shape": [4, 4], "propagation_grid_size": 8}, "run": {"run_name": "unit", "device": "cpu"}})
    assert not config.system.is_formal


def test_shape_guard_contract_locks_raw_shape_weights() -> None:
    payload = _config()
    payload["prototype_calibration"] = {
        "prototype_contract_kind": "formal_prototype_shape_guard_v1",
        "epochs": 5,
        "learning_rate": 5.0e-4,
        "fixed_exposure_scale": 1.0,
        "initialization": "clean_intensity_to_amplitude",
        "optimizer_steps_per_epoch": 40,
        "foreground_weight": 3.0,
        "background_weight": 1.0,
        "background_energy_weight": 0.10,
    }
    payload["capacity_audit"] = {"dct_modes": 64, "selected_basis_dim": 8, "codes_per_label": 100}
    assert V2Config.model_validate(payload).prototype_calibration is not None
    payload["prototype_calibration"]["foreground_weight"] = 2.0
    with pytest.raises(ValidationError, match="foreground_weight"):
        V2Config.model_validate(payload)
