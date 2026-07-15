from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v61_offline_decodability_upper_bound.py"


def _module():
    spec = importlib.util.spec_from_file_location("v61_decodability", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _labels(count_per_class: int) -> torch.Tensor:
    return torch.arange(10).repeat_interleave(count_per_class)


def _targets(labels: torch.Tensor) -> torch.Tensor:
    values = torch.zeros(labels.numel(), 1, 200, 200)
    for index, label in enumerate(labels.tolist()):
        top = 10 + int(label) * 15
        values[index, 0, top : top + 8, 20:60] = 1.0
    return values


def _split_payload(split: str) -> dict[str, object]:
    module = _module()
    count = module.SPLIT_COUNTS[split]
    labels = _labels(module.SPLIT_CLASS_COUNTS[split])
    assert labels.numel() == count
    return {
        "codes": torch.zeros(count, 128),
        "class_labels": labels,
        "teacher_camera": _targets(labels),
        "readability_split": split,
        "config": {"filter": {"teacher_filter": "readability_strict_split"}},
        "label_audit_nearest_label": labels.clone(),
        "label_audit_margin": torch.ones(count),
    }


def _record(field: torch.Tensor, *, upstream: str | None, operation: str, sampling: dict[str, object]) -> dict[str, object]:
    return {
        "real": field.real,
        "imag": field.imag,
        "shape": list(field.shape),
        "upstream": upstream,
        "operation": operation,
        "phase_available": True,
        "source_kind": "propagated_complex_field",
        "sampling": sampling,
    }


def _sampling(shape: int, pitch: float) -> dict[str, object]:
    return {
        "shape": [shape, shape],
        "pixel_size_x": pitch,
        "pixel_size_y": pitch,
        "fov_x": shape * pitch,
        "fov_y": shape * pitch,
    }


def _strict_outputs(module) -> dict[str, object]:
    config = module.load_training_config(module.DEFAULT_TRAIN_CONFIG)
    slm = torch.ones(1, 1, 200, 200, dtype=torch.complex64)
    field = torch.ones(1, 1, 100, 100, dtype=torch.complex64)
    camera = torch.ones(1, 1, 200, 200, dtype=torch.complex64)
    chip_sampling, camera_sampling = _sampling(100, 2.0e-6), _sampling(200, 2.5e-6)
    fields = {
        "slm_output": _record(slm, upstream=None, operation="slm_output_field", sampling=_sampling(200, 2.5e-6)),
        "pre_encoder_relay_image": _record(field, upstream="slm_output", operation="coherent_imaging_4f", sampling=chip_sampling),
        "encoder_input": _record(field, upstream="pre_encoder_relay_image", operation="coherent_relay_alias", sampling=chip_sampling),
        "encoder_grid": _record(field, upstream="encoder_input", operation="coherent_relay_alias", sampling=chip_sampling),
        "encoder_layer_1": _record(field, upstream="encoder_grid", operation="encoder_diffractive_layer", sampling=chip_sampling),
        "encoder_layer_2": _record(field, upstream="encoder_layer_1", operation="encoder_diffractive_layer", sampling=chip_sampling),
        "encoder_output": _record(field, upstream="encoder_layer_2", operation="encoder_output", sampling=chip_sampling),
        "ols_input": _record(field, upstream="encoder_output", operation="ols_input_alias", sampling=chip_sampling),
        "ols_output": _record(field, upstream="ols_input", operation="complex_ols:identity", sampling=chip_sampling),
        "decoder_input": _record(field, upstream="ols_output", operation="decoder_input", sampling=chip_sampling),
        "decoder_layer_1": _record(field, upstream="decoder_input", operation="decoder_diffractive_layer", sampling=chip_sampling),
        "decoder_layer_2": _record(field, upstream="decoder_layer_1", operation="decoder_diffractive_layer", sampling=chip_sampling),
        "decoder_output": _record(field, upstream="decoder_layer_2", operation="decoder_output", sampling=chip_sampling),
        "chip": _record(field, upstream="decoder_output", operation="chip_plane", sampling=chip_sampling),
        "camera_field": _record(camera, upstream="chip", operation="coherent_camera_propagation", sampling=camera_sampling),
    }
    raw = camera.abs().square()
    return {
        "optical_fields": fields,
        "physical_camera_intensity_raw": raw,
        "physical_camera_plane": module.normalize_real_intensity(
            raw, config.physics.readout.normalize_percentile, clamp=False
        ).clamp(0.0, 1.0),
        "target_code_diagnostic_active": torch.tensor(False),
        "target_code_bridge_active": torch.tensor(False),
    }


def test_readable_split_validation_requires_strict_balanced_contract():
    module = _module()
    assert module._validate_readable_split(_split_payload("val"), "val") == []
    payload = _split_payload("val")
    payload["readability_split"] = "train"
    assert any("readability_split" in reason for reason in module._validate_readable_split(payload, "val"))


def test_strict_forward_uses_raw_field_intensity_and_detector_plane():
    module = _module()
    config = module.load_training_config(module.DEFAULT_TRAIN_CONFIG)
    features, errors = module._validate_forward(_strict_outputs(module), config=config)
    assert set(features) == set(module.MODALITIES)
    assert tuple(features["chip_intensity"].shape) == (1, 1, 100, 100)
    assert tuple(features["camera_complex_oracle"].shape) == (1, 2, 200, 200)
    assert errors["raw_camera_intensity_field_max_error"] == pytest.approx(0.0)
    outputs = _strict_outputs(module)
    outputs["physical_camera_intensity_raw"] = outputs["physical_camera_intensity_raw"] + 0.1
    with pytest.raises(ValueError, match=r"\|E_camera\|\^2"):
        module._validate_forward(outputs, config=config)


def test_complex_oracle_decodes_phase_coded_targets_but_intensity_does_not():
    module = _module()
    labels = {split: _labels(2) for split in ("train", "val", "test")}
    targets = {split: _targets(value) for split, value in labels.items()}
    complex_features = {
        split: torch.nn.functional.one_hot(value, num_classes=10).float()
        for split, value in labels.items()
    }
    intensity_features = {split: torch.zeros(value.shape[0], 10) for split, value in labels.items()}
    complex_report, _ = module._decode_modality(
        complex_features, targets, labels, compute_device=torch.device("cpu")
    )
    intensity_report, _ = module._decode_modality(
        intensity_features, targets, labels, compute_device=torch.device("cpu")
    )
    assert complex_report["test"]["decodable_at_preregistered_thresholds"] is True
    assert intensity_report["test"]["decodable_at_preregistered_thresholds"] is False


def test_decision_prioritizes_detector_then_phase_then_backend():
    module = _module()

    def modality(*, decodable: bool, label_accuracy: float = 0.0):
        return {
            "test": {
                "decodable_at_preregistered_thresholds": decodable,
                "prototype_label_accuracy": label_accuracy,
            }
        }

    base = {name: modality(decodable=False) for name in module.MODALITIES}
    assert module._decision(base)["status"] == "objective-representation-mismatch-confirmed"
    phase = dict(base)
    phase["camera_complex_oracle"] = modality(decodable=True)
    assert module._decision(phase)["status"] == "camera-phase-to-intensity-representation-mismatch"
    backend = dict(base)
    backend["chip_intensity"] = modality(decodable=True)
    assert module._decision(backend)["status"] == "backend-relay-intensity-mismatch-candidate"
    class_only = dict(base)
    class_only["camera_complex_oracle"] = modality(decodable=False, label_accuracy=0.8)
    assert module._decision(class_only)["status"] == "class-only-information-insufficient-for-image-generation"


def test_v58_3_report_requires_complete_current_revision(tmp_path: Path):
    module = _module()
    path = tmp_path / "reference.json"
    payload = {
        "status": "physical-forward-model-valid",
        "validation_revision": "v58.3-full-fov-b0-relay-acceptance",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        **{name: {} for name in module.REQUIRED_V58_3_SECTIONS},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert module._validate_v58_3_report(path) == []
    payload.pop("measured_pupil_psf_validation")
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert module._validate_v58_3_report(path)


def test_state_snapshot_detects_any_parameter_update():
    module = _module()
    model = torch.nn.Linear(2, 2)
    before = module._state_snapshot(model)
    assert module._state_changes(model, before) == []
    with torch.no_grad():
        model.weight.add_(1.0)
    assert module._state_changes(model, before) == ["weight"]


def test_tool_has_no_resize_or_grid_sample_probe_path():
    source = TOOL.read_text(encoding="utf-8")
    assert "grid_sample" not in source
    assert "interpolate(" not in source
    assert "adaptive_avg_pool" not in source
    assert "TinyUNet" not in source


def test_native_feature_memory_maps_match_the_pre_registered_sampling(tmp_path: Path):
    module = _module()
    maps = module._create_feature_memmaps(tmp_path)
    expected = module._feature_dimensions()
    for modality, split_maps in maps.items():
        for split, values in split_maps.items():
            assert tuple(values.shape) == (module.SPLIT_COUNTS[split], expected[modality])
            assert values.dtype == np.dtype("float32")
