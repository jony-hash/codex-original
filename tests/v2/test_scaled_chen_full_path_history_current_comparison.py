from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/visualization/render_scaled_chen_full_pipeline.py"
NATURE_SCRIPTS = Path(r"C:\Users\24567\.codex\skills\nature-figure\scripts")


@pytest.fixture(scope="module")
def renderer():
    sys.path.insert(0, str(NATURE_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "scaled_chen_history_current_test_module", SCRIPT
        )
        if spec is None or spec.loader is None:
            raise AssertionError("failed to load visualization renderer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(NATURE_SCRIPTS))


def _fake_views(renderer, positions: range) -> dict[str, dict[str, dict[int, dict[str, np.ndarray]]]]:
    views: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]] = {}
    for state in renderer.HISTORY_CURRENT_STATES:
        views[state] = {}
        for stage_id in renderer.HISTORY_CURRENT_STAGE_KEYS:
            views[state][stage_id] = {}
            for position in positions:
                values = np.ones((4, 4), dtype=np.float32)
                views[state][stage_id][position] = {
                    "amplitude": values,
                    "phase": np.zeros_like(values),
                    "phase_available": True,
                }
    return views


def test_history_current_contract_and_page_order(renderer) -> None:
    assert renderer.HISTORY_CURRENT_LAYOUT_SCHEMA == (
        "full_optical_path_historical_current_side_by_side_v1"
    )
    assert renderer.HISTORY_CURRENT_STAGE_KEYS == (
        "slm_output",
        "encoder_input",
        "encoder_layer_1",
        "pre_ols",
        "post_ols",
        "decoder_layer_1",
        "decoder_layer_2",
        "cmos_field",
    )
    assert renderer.HISTORY_CURRENT_STAGE_ALIASES == {
        "pre_ols": "encoder_layer_2",
        "post_ols": "ols_output",
        "cmos_field": "camera",
    }
    assert renderer.HISTORY_CURRENT_PAGE_DIGITS == (
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8, 9),
    )
    assert [digit for page in renderer.HISTORY_CURRENT_PAGE_DIGITS for digit in page] == list(
        range(10)
    )
    assert renderer.HISTORY_CURRENT_STATES == ("historical", "current")


def test_historical_trace_selector_uses_two_times_digit_and_preserves_stage_order(renderer) -> None:
    positions = renderer._historical_trace_positions()
    assert positions == {digit: 2 * digit for digit in range(10)}

    fields = {
        stage_id: torch.full((1, 1, 4, 4), complex(digit, 0), dtype=torch.complex64)
        for digit, stage_id in enumerate(renderer.HISTORY_CURRENT_STAGE_KEYS)
    }
    trace = [{"fields": [{"plane": {"plane_id": key}, "tensor": value} for key, value in fields.items()]} for _ in range(20)]
    selected, records = renderer._select_historical_trace_digits(trace)
    assert list(selected) == list(range(10))
    assert [record["trace_position"] for record in records] == [2 * digit for digit in range(10)]
    assert all(tuple(selected_fields) == renderer.HISTORY_CURRENT_STAGE_KEYS for selected_fields in selected.values())


def test_slm_phase_is_fixed_zero_and_display_rotation_is_pure(renderer) -> None:
    values = {
        key: torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).to(torch.complex64)
        for key in renderer.HISTORY_CURRENT_STAGE_KEYS
    }
    original = {key: value.clone() for key, value in values.items()}
    displayed = renderer._history_current_display_values(values)
    assert torch.equal(values["slm_output"], original["slm_output"])
    assert torch.equal(values["cmos_field"], original["cmos_field"])
    assert torch.equal(
        displayed["encoder_input"], original["encoder_input"].flip((-2, -1))
    )
    assert all(torch.equal(values[key], original[key]) for key in values)

    phase = renderer._history_current_phase(values["slm_output"])
    assert torch.allclose(phase, torch.zeros_like(phase))


def test_history_current_plate_is_side_by_side_equal_aspect_and_compact(renderer, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    positions = range(2)
    captured: dict[str, object] = {}

    def capture(fig, output_base, **kwargs):
        captured["figure"] = fig
        captured["axes"] = kwargs["axes"]
        captured["include_tiff"] = kwargs["include_tiff"]
        return []

    monkeypatch.setattr(renderer, "_save_figure_bundle", capture)
    renderer._render_history_current_plate(
        _fake_views(renderer, positions),
        output_base=ROOT / "unused",
        positions=positions,
        labels=(0, 1),
        historical_records=({"digit": 0, "trace_position": 0}, {"digit": 1, "trace_position": 2}),
        current_indices=(55006, 55000),
        page_label="digits 0–1",
    )
    figure = captured["figure"]
    assert figure.get_figwidth() <= renderer.HISTORY_CURRENT_FIGURE_WIDTH_IN
    assert renderer.HISTORY_CURRENT_PANEL_WIDTH_IN <= 7.2
    assert all(float(axis.get_aspect()) == 1.0 for axis in captured["axes"])
    assert captured["include_tiff"] is False
    plt.close(figure)


def test_source_manifest_hash_guard_and_output_policy(renderer, tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"schema":"ok"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="source manifest hash"):
        renderer._load_history_current_source_manifest(source, "0" * 64)

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    for name in ("figure.png", "figure.pdf", "figure.svg", "figure.alignment.json", "figure.alignment.svg", "comparison_manifest.json"):
        (allowed / name).write_text("x", encoding="utf-8")
    renderer._verify_history_current_output_files(allowed)
    (allowed / "unexpected.tiff").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden output"):
        renderer._verify_history_current_output_files(allowed)


def test_cli_exposes_history_current_comparison(renderer, monkeypatch) -> None:
    original = sys.argv
    try:
        sys.argv = [
            "render_scaled_chen_full_pipeline.py",
            "--view",
            "full-path-history-current-comparison",
        ]
        args = renderer._parse_args()
    finally:
        monkeypatch.setattr(sys, "argv", original)
    assert args.view == "full-path-history-current-comparison"
