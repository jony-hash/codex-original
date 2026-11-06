from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.research.lightgen import diagnose_lightgen_gaussian_j_frozen_attribution_phase_density_v1 as diag


def test_equal_camera_power_keeps_zero_source_undefined() -> None:
    camera = torch.ones((2, 2, 2), dtype=torch.complex64)
    camera[1].zero_()
    pixels = torch.zeros((20, 28, 28), dtype=torch.uint8)
    geometry = {"pixel_pitch_m": [1.0, 1.0]}
    output, _view, defined = diag._equal_camera_readout(
        camera, geometry, 1.0, np.asarray([8.0, 0.0]), pixels, np.asarray([11, 12]),
    )
    assert defined.tolist() == [True, False]
    assert output[0, 4] == 8.0
    assert np.isnan(output[1]).all()


def test_equal_camera_power_rejects_misaligned_target_population() -> None:
    camera = torch.ones((2, 2, 2), dtype=torch.complex64)
    with pytest.raises(RuntimeError, match="finite, nonnegative, and row-aligned"):
        diag._equal_camera_readout(
            camera, {"pixel_pitch_m": [1.0, 1.0]}, 1.0,
            np.asarray([1.0]), None, None,
        )


def test_equal_camera_power_rejects_intensity_frame() -> None:
    intensity = torch.full((1, 2, 2), 0.5, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="complex camera field, not an intensity frame"):
        diag._equal_camera_readout(
            intensity, {"pixel_pitch_m": [1.0, 1.0]}, 1.0,
            np.asarray([2.0]), None, None,
        )


def test_native_dev_equal_power_uses_camera_field_not_intensity(tmp_path, monkeypatch) -> None:
    camera_field = torch.full((1, 1, 2, 2), 0.5**0.5, dtype=torch.complex64)
    camera_intensity = camera_field.abs().square()
    geometry = {"plane_id": "camera", "coordinate_frame": "test", "shape": [2, 2],
                "pixel_pitch_m": [1.0, 1.0], "origin_m": [0.0, 0.0], "z_m": 0.0,
                "geometric_area_m2": 4.0, "measurement_boundary": "test"}
    fields = [(name, SimpleNamespace(tensor=camera_field)) for name in diag.DEV_PLANES]
    fields[-1] = (diag.DECODER_PLANES[-1], SimpleNamespace(tensor=camera_field))
    result = SimpleNamespace(
        z_ols=SimpleNamespace(coefficients=torch.zeros((1, 100), dtype=torch.complex64)),
        physical_sensor_output_w_per_pixel=camera_intensity,
        raw_camera=SimpleNamespace(tensor=camera_intensity),
        physical_power_trace={"P_camera_W": torch.tensor([2.0], dtype=torch.float64)},
    )
    model = SimpleNamespace(runtime=SimpleNamespace(forward_from_image=lambda _image: result))

    monkeypatch.setattr(diag, "_model", lambda *_args: model)
    monkeypatch.setattr(diag, "image_batch", lambda *_args: torch.zeros((1, 1, 2, 2)))
    monkeypatch.setattr(diag, "small_image_result_fields", lambda *_args: (fields, {}))
    monkeypatch.setattr(diag, "field_plane_metrics", lambda _field: (
        np.asarray([[2.0, 4.0, 4.0, 0.5, 0.5, 1.0, 1.0]], dtype=np.float64), geometry,
    ))
    monkeypatch.setattr(diag, "field_intensity_distribution", lambda _field: [
        {key: 0.5 for key in diag.DIST_FIELDS}
    ])
    monkeypatch.setattr(diag, "resize_view", lambda value: torch.zeros((len(value), 1, 28, 28)))
    monkeypatch.setattr(diag, "ssim_batch", lambda left, _right: torch.ones(len(left)))

    d0_metrics = np.zeros((1, len(diag.DEV_PLANES), 7), dtype=np.float64)
    d0_metrics[0, 9, 0] = 8.0
    fixed = {
        "dev_rows": torch.tensor([0]), "labels": torch.tensor([0]),
        "pixels": torch.zeros((1, 28, 28), dtype=torch.uint8),
        "device": torch.device("cpu"),
        "checkpoints": {"J50000": {"camera_gain": torch.tensor([1.0])}},
    }
    diag._measure_dev("J50000", "production", 2048, tmp_path, fixed,
                      {"metrics": d0_metrics})

    with np.load(tmp_path / "populations/dev/J50000/production/chunk_0000.npz",
                 allow_pickle=False) as saved:
        equal = saved["equal_camera_power_reconstruction"]
    assert equal[0, 4] == pytest.approx(8.0)
    assert equal[0, 0] == pytest.approx(0.0)


def test_npz_readback_handles_string_metadata_and_nan_values(tmp_path) -> None:
    path = tmp_path / "mixed_arrays.npz"
    diag._write_npz(path, endpoint=np.asarray("J50000"), values=np.asarray([1.0, np.nan]))
    with np.load(path, allow_pickle=False) as saved:
        assert saved["endpoint"].item() == "J50000"
        assert saved["values"][0] == 1.0 and np.isnan(saved["values"][1])


def test_plane_order_recovers_missing_index_from_saved_geometry(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = {"planes": [{"plane_key": "first"}, {"plane_key": "second"}]}
    diag._validated_plane_order(metadata, ("first", "second"), metadata_path)
    assert metadata["plane_order"] == ["first", "second"]
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["plane_order"] == ["first", "second"]


def test_plane_order_rejects_geometry_order_mismatch(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="plane geometry order"):
        diag._validated_plane_order(
            {"planes": [{"plane_key": "second"}, {"plane_key": "first"}]},
            ("first", "second"), tmp_path / "metadata.json",
        )


def test_class_bootstrap_reports_equal_class_mean_and_interval() -> None:
    labels = np.repeat(np.arange(10), 2)
    values = np.repeat(np.arange(10, dtype=np.float64), 2)
    result = diag._class_bootstrap(values, labels)
    assert result["mean_equal_class"] == 4.5
    assert result["n"] == 20
    assert len(result["ci95"]) == 2


def test_factorial_encoder_decoder_effects_reconstruct_native_change() -> None:
    l00 = np.asarray([0.9, 0.7, 0.5])
    l01 = np.asarray([0.8, 0.6, 0.45])
    l10 = np.asarray([0.7, 0.65, 0.4])
    l11 = np.asarray([0.55, 0.50, 0.35])
    effects = diag._factorial_mse_effects(l00, l01, l10, l11)
    np.testing.assert_allclose(
        effects["encoder_average_mse_improvement"] + effects["decoder_average_mse_improvement"],
        l00-l11, rtol=0.0, atol=1e-15,
    )
    np.testing.assert_allclose(
        effects["interaction_L00_minus_L01_minus_L10_plus_L11"], l00-l01-l10+l11,
        rtol=0.0, atol=0.0,
    )


def test_complex_moments_preserve_imaginary_cross_terms() -> None:
    coefficients = np.asarray([
        [1.0+1.0j, 0.4+0.2j, 0.2-0.7j],
        [0.3+0.9j, 1.1+0.1j, -0.2+0.8j],
        [0.8-0.4j, 0.2+0.9j, 0.6+0.5j],
        [0.5+0.6j, 0.8-0.3j, -0.5+0.2j],
    ], dtype=np.complex128)
    second, covariance, gauged = diag._complex_moment_matrices(coefficients)
    independent_second = sum(np.outer(row, row.conj()) for row in gauged) / len(gauged)
    independent_covariance = independent_second - np.outer(gauged.mean(0), gauged.mean(0).conj())
    np.testing.assert_allclose(second, independent_second, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(covariance, independent_covariance, rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(np.imag(covariance))) > 1e-8
    np.testing.assert_allclose(covariance, covariance.conj().T, rtol=0.0, atol=1e-12)


def test_phase_conditions_use_requested_per_image_rms(monkeypatch) -> None:
    y, x = np.mgrid[-4:6, -4:6]
    centers = np.stack((x.reshape(-1), y.reshape(-1)), axis=1).astype(np.float64)
    fake_model = SimpleNamespace(runtime=SimpleNamespace(ols=SimpleNamespace(
        operator=SimpleNamespace(centers_m=torch.as_tensor(centers)),
    )))
    monkeypatch.setattr(diag, "_model", lambda _key, _grid, _fixed: fake_model)
    labels = np.repeat(np.arange(10, dtype=np.int64), 20)
    conditions = diag._phase_conditions(
        {"device": torch.device("cpu")}, {"phase": {
            "seed": 2026102303, "relative_rms_rad": [0.01, 0.03, 0.1, 0.3],
            "families": ["independent_port", "linear_ramp"], "repeats_per_class": 8,
        }}, labels,
    )
    assert len(conditions) == 66
    for condition in conditions:
        if condition["family"] in {"independent_port", "linear_ramp"}:
            rms = np.sqrt(np.mean(np.square(condition["phase"]), axis=1))
            np.testing.assert_allclose(rms, condition["sigma"], rtol=0.0, atol=1e-12)
    common = conditions[-1]
    assert common["family"] == "common_phase_control"
    np.testing.assert_array_equal(common["phase"], np.full((200, 100), 0.3))
