from __future__ import annotations

import copy

import pytest
import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from scripts.research.lightgen import build_lightgen_architecture_visual_comparison_v1 as plots
from scripts.research.lightgen import visualize_lightgen_architecture_comparison_v1 as capture


def _collector() -> tuple[capture.FieldCollector, torch.Tensor]:
    endpoint = capture.ENDPOINTS[0]
    collector = capture.FieldCollector.create(endpoint, [1234], [7])
    plane = PlaneSpec(
        plane_id="test_physical_plane",
        shape=(4, 4),
        pixel_pitch_m=(2.0e-6, 2.0e-6),
        wavelength_m=capture.WAVELENGTH_M,
        z_m=0.62875,
        coordinate_frame="test_centered_grid",
        origin_m=(0.0, 0.0),
    )
    values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).to(torch.complex64) + 1j
    field = ComplexField(values, plane, OperationProvenance("test_field", None, None, {}))
    collector.add("test_node", field, phase_position="after device")
    return collector, values


def _payload(collector: capture.FieldCollector, values: torch.Tensor, *, phase_order: list[str] | None = None, source_row: int = 1234) -> dict[str, object]:
    return {
        "schema": "lightgen_frozen_complex_field_chunk.v1",
        "endpoint": collector.endpoint.name,
        "positions": torch.tensor([2], dtype=torch.long),
        "source_row_indices": torch.tensor([source_row], dtype=torch.long),
        "labels": torch.tensor([7], dtype=torch.long),
        "data_roles": ["confirmation"],
        "phase_order": phase_order or list(capture.PHASE_LAYOUT),
        "fields": {"test_node": values.clone()},
        "field_metadata": copy.deepcopy(collector.metadata),
    }


def test_field_record_preserves_physical_coordinates_and_complex_values() -> None:
    collector, values = _collector()
    metadata = collector.metadata["test_node"]

    assert metadata["origin_m"] == [0.0, 0.0]
    assert metadata["fov_m"] == [8.0e-6, 8.0e-6]
    assert metadata["pixel_pitch_m"] == [2.0e-6, 2.0e-6]
    assert metadata["field_kind"] == "physical_full_field"
    capture._validate_field_record("test_node", values, metadata, [1234], [7])


def test_field_record_rejects_missing_physical_coordinates() -> None:
    collector, values = _collector()
    metadata = dict(collector.metadata["test_node"])
    metadata.pop("pixel_pitch_m")

    with pytest.raises(capture.VisualComparisonError, match="physical pixel coordinates"):
        capture._validate_field_record("test_node", values, metadata, [1234], [7])


def test_field_record_rejects_row_rebinding_and_incremental_field() -> None:
    collector, values = _collector()
    with pytest.raises(capture.VisualComparisonError, match="source row/label binding"):
        capture._validate_field_record("test_node", values, collector.metadata["test_node"], [9999], [7])

    endpoint = capture.ENDPOINTS[0]
    plane = PlaneSpec("delta_plane", (4, 4), (2e-6, 2e-6), capture.WAVELENGTH_M, 0.2, "delta")
    delta = ComplexField(values, plane, OperationProvenance("test_delta", None, None, {}))
    with pytest.raises(capture.VisualComparisonError, match="deviation field"):
        capture.FieldCollector.create(endpoint, [1234], [7]).add("delta", delta, field_kind="deviation_field")


def test_saved_loader_rejects_wrong_phase_storage_order(tmp_path) -> None:
    collector, values = _collector()
    path = tmp_path / "bad_order.pt"
    torch.save(_payload(collector, values, phase_order=["new_0p75mm", "old_0mm", "old_1p5mm", "new_2p25mm"]), path)

    with pytest.raises(capture.VisualComparisonError, match="phase storage order"):
        capture._load_field_chunk(path, expected_endpoint=collector.endpoint.name)


def test_plot_consumer_checks_saved_row_and_uses_saved_coordinate_metadata(tmp_path, monkeypatch) -> None:
    collector, values = _collector()
    directory = tmp_path / "fields" / collector.endpoint.name
    directory.mkdir(parents=True)
    path = directory / "field_chunk_000.pt"
    torch.save(_payload(collector, values), path)
    monkeypatch.setattr(plots, "OUTPUT", tmp_path)
    plots._FIELD_PAYLOAD_CACHE.clear()

    sample = {"position": 2, "source_row_index": 1234, "label": 7}
    field, metadata = plots._read_field(collector.endpoint.name, "test_node", sample)
    display, extent, _ = plots._crop_for_display(field, metadata)

    assert display.shape == (4, 4)
    assert extent == pytest.approx((-0.004, 0.004, -0.004, 0.004))
    assert metadata["origin_m"] == [0.0, 0.0]

    bad_sample = {**sample, "source_row_index": 1235}
    with pytest.raises(capture.VisualComparisonError, match="selected source row or label differs"):
        plots._read_field(collector.endpoint.name, "test_node", bad_sample)
    plots._FIELD_PAYLOAD_CACHE.clear()
