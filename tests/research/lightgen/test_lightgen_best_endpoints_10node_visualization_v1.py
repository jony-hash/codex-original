from __future__ import annotations

import pytest
import numpy as np

from scripts.research.lightgen import build_lightgen_best_endpoints_10node_visualization_v1 as plots
from scripts.research.lightgen import capture_lightgen_best_endpoints_10node_v1 as capture
from scripts.research.lightgen import visualize_lightgen_architecture_comparison_v1 as field_capture


def _pair_records() -> tuple[dict[str, object], dict[str, object], dict[str, int]]:
    sample = {"position": 3, "source_row_index": 1042, "label": 7}
    small_fields = {node: object() for node, _display in capture.DISPLAY_NODE_MAP[capture.SMALL_NAME]}
    large_fields = {node: object() for node, _display in capture.DISPLAY_NODE_MAP[capture.LARGE_NAME]}
    small_meta = {
        node: {"field_kind": "physical_full_field", "plane_shape": [4, 4], "origin_m": [0.0, 0.0], "pixel_pitch_m": [1.0e-6, 1.0e-6], "z_m": 0.2}
        for node, _display in capture.DISPLAY_NODE_MAP[capture.SMALL_NAME]
    }
    large_meta = {
        node: {"field_kind": "physical_full_field", "plane_shape": [4, 4], "origin_m": [0.0, 0.0], "pixel_pitch_m": [1.0e-6, 1.0e-6], "z_m": 0.2}
        for node, _display in capture.DISPLAY_NODE_MAP[capture.LARGE_NAME]
    }
    small = {"source_row_index": 1042, "label": 7, "fields": small_fields, "metadata": small_meta}
    large = {"source_row_index": 1042, "label": 7, "fields": large_fields, "metadata": large_meta}
    return small, large, sample


def test_display_contract_has_ten_ordered_nodes_and_phase_semantics() -> None:
    small = capture.DISPLAY_NODE_MAP[capture.SMALL_NAME]
    large = capture.DISPLAY_NODE_MAP[capture.LARGE_NAME]

    assert len(small) == len(large) == 10
    assert small[5][0] == "generator_z0p75_free_reference"
    assert large[5][0] == "generator_z0p75_after_phase"
    assert small[7][0] == "generator_z2p25_free_reference"
    assert large[7][0] == "generator_z2p25_after_phase"
    assert small[3][0] == large[3][0] == "ols_coefficients_10x10"


def test_pair_consumer_accepts_matching_source_row_label_and_nodes() -> None:
    small, large, sample = _pair_records()

    plots._validate_pair_sample(sample, small, large)


def test_pair_consumer_rejects_source_row_or_label_mismatch() -> None:
    small, large, sample = _pair_records()
    large["source_row_index"] = 1043

    with pytest.raises(field_capture.VisualComparisonError, match="selected source row or label differs"):
        plots._validate_pair_sample(sample, small, large)


def test_pair_consumer_rejects_missing_required_display_node() -> None:
    small, large, sample = _pair_records()
    del large["fields"]["generator_z2p25_after_phase"]

    with pytest.raises(field_capture.VisualComparisonError, match="required display nodes are missing"):
        plots._validate_pair_sample(sample, small, large)


def test_small_coordinate_resolution_adds_authoritative_centered_grid() -> None:
    source = {
        "coordinate_frame": "encoder",
        "plane_id": "encoder_layer_1",
        "z_m": 0.2,
        "origin_m": [0.0, 0.0],
        "pixel_pitch_m": [2.0e-6, 2.0e-6],
        "plane_shape": [4, 4],
        "fov_m": [8.0e-6, 8.0e-6],
        "wavelength_m": 532e-9,
        "origin_convention": "center of pixel array; PlaneSpec default origin=(0,0)",
        "pixel_center_formula": "coordinate[i] = origin + (i - (N-1)/2) * pixel_pitch",
    }
    saved = {
        "field_kind": "physical_full_field",
        "coordinate_frame": "encoder",
        "plane_id": "encoder_layer_1",
        "z_m": 0.2,
        "pixel_pitch_m": [2.0e-6, 2.0e-6],
        "plane_shape": [4, 4],
    }

    resolved = plots._resolve_small_coordinates(saved, source, "encoder_layer_1")

    assert resolved["origin_m"] == [0.0, 0.0]
    assert resolved["fov_m"] == [8.0e-6, 8.0e-6]
    assert resolved["pixel_center_formula"] == source["pixel_center_formula"]


def test_small_coordinate_resolution_rejects_mismatched_saved_grid() -> None:
    source = {
        "coordinate_frame": "encoder",
        "plane_id": "encoder_layer_1",
        "z_m": 0.2,
        "origin_m": [0.0, 0.0],
        "pixel_pitch_m": [2.0e-6, 2.0e-6],
        "plane_shape": [4, 4],
        "fov_m": [8.0e-6, 8.0e-6],
        "wavelength_m": 532e-9,
        "origin_convention": "center of pixel array; PlaneSpec default origin=(0,0)",
        "pixel_center_formula": "coordinate[i] = origin + (i - (N-1)/2) * pixel_pitch",
    }
    saved = {
        "field_kind": "physical_full_field",
        "coordinate_frame": "encoder",
        "plane_id": "encoder_layer_1",
        "z_m": 0.2,
        "pixel_pitch_m": [2.0e-6, 2.0e-6],
        "plane_shape": [5, 4],
    }

    with pytest.raises(field_capture.VisualComparisonError, match="plane_shape differs"):
        plots._resolve_small_coordinates(saved, source, "encoder_layer_1")


def test_complete_physical_coordinate_contract_rejects_missing_origin() -> None:
    metadata = {
        "coordinate_frame": "camera",
        "plane_id": "camera",
        "z_m": 0.2,
        "pixel_pitch_m": [1e-6, 1e-6],
        "plane_shape": [4, 4],
        "fov_m": [4e-6, 4e-6],
        "wavelength_m": 532e-9,
        "origin_convention": "center",
        "pixel_center_formula": "formula",
    }

    with pytest.raises(field_capture.VisualComparisonError, match="origin_m"):
        plots._require_physical_coordinates(metadata, "test")


def test_display_crop_squeezes_only_singleton_leading_field_axis() -> None:
    field = (1.0 + 2.0j) * np.ones((1, 5, 6), dtype=np.complex64)
    metadata = {
        "field_kind": "physical_full_field",
        "plane_shape": [5, 6],
        "pixel_pitch_m": [1.0e-6, 2.0e-6],
        "origin_m": [0.0, 0.0],
    }

    displayed, extent, note = plots._crop_display(field, metadata)

    assert displayed.shape == (5, 6)
    assert np.allclose(extent, (-6.0e-6, 6.0e-6, -2.5e-6, 2.5e-6), rtol=0.0, atol=1.0e-18)
    assert note == "native full plane"


def test_display_crop_rejects_non_singleton_leading_field_axis() -> None:
    field = np.ones((2, 5, 6), dtype=np.complex64)
    metadata = {"plane_shape": [5, 6], "pixel_pitch_m": [1.0e-6, 2.0e-6], "origin_m": [0.0, 0.0]}

    with pytest.raises(field_capture.VisualComparisonError, match="native 2D plane"):
        plots._crop_display(field, metadata)


def test_generator_midpoint_caption_reports_display_crop_and_full_plane() -> None:
    metadata = {
        "plane_shape": [2048, 2048],
        "pixel_pitch_m": [2.0e-6, 2.0e-6],
        "fov_m": [4.096e-3, 4.096e-3],
    }

    caption = plots._window_description(metadata)

    assert caption == "中心显示窗 0.8×0.8 mm · 全幅 4.1×4.1 mm"
