from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/visualization/render_scaled_chen_full_pipeline.py"
NATURE_SCRIPTS = Path(r"C:\Users\24567\.codex\skills\nature-figure\scripts")


@pytest.fixture(scope="module")
def renderer():
    sys.path.insert(0, str(NATURE_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("scaled_chen_visualization_test_module", SCRIPT)
        if spec is None or spec.loader is None:
            raise AssertionError("failed to load visualization renderer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(NATURE_SCRIPTS))


def test_fixed_confirmation_panel_is_balanced_and_ordered(renderer) -> None:
    assert renderer.DIGIT_CONFIRMATION_INDICES == {
        0: 55006,
        1: 55000,
        2: 55001,
        3: 55008,
        4: 55004,
        5: 55010,
        6: 55012,
        7: 55003,
        8: 55002,
        9: 55007,
    }
    assert len(set(renderer.DIGIT_CONFIRMATION_INDICES.values())) == 10
    assert len(renderer.FIBER_STAGE_KEYS) == len(renderer.FIBER_STAGE_LABELS) == 9
    assert renderer.FIBER_STAGE_KEYS == (
        "encoder_input",
        "encoder_layer_1",
        "fiber_input",
        "port_input",
        "port_output",
        "fiber_output",
        "decoder_layer_1",
        "decoder_layer_2",
        "cmos_field",
    )


def test_selection_rejects_wrong_registered_labels(renderer) -> None:
    images = torch.zeros((60000, 1, 28, 28))
    labels = torch.zeros(60000, dtype=torch.long)
    for digit, index in renderer.DIGIT_CONFIRMATION_INDICES.items():
        labels[index] = digit
    selected, selected_labels, indices = renderer._select_confirmation_digits(images, labels)
    assert selected.shape == (10, 1, 28, 28)
    assert torch.equal(selected_labels, torch.arange(10))
    assert tuple(indices[digit] for digit in range(10)) == tuple(renderer.DIGIT_CONFIRMATION_INDICES.values())
    labels[renderer.DIGIT_CONFIRMATION_INDICES[3]] = 8
    with pytest.raises(ValueError, match="confirmation panel labels"):
        renderer._select_confirmation_digits(images, labels)


def test_shared_scale_mask_and_display_rotation_are_pure(renderer) -> None:
    complex_field = torch.ones((10, 1, 4, 4), dtype=torch.complex64)
    complex_field[0, 0, 0, 0] = 0.001 + 0.0j
    complex_field[1, 0, 0, 0] = 10.0 + 0.0j
    ports = torch.ones((10, 100), dtype=torch.complex64)
    values = {
        key: (ports if key in renderer.FIBER_MODAL_KEYS else complex_field)
        for key in renderer.FIBER_STAGE_KEYS
    }
    original = values["fiber_input"].clone()
    views, spatial_scale, modal_scale = renderer._complex_views(values, sample_positions=range(10))
    assert spatial_scale > 1.0
    assert modal_scale == pytest.approx(1.0)
    assert torch.equal(values["fiber_input"], original)
    amplitude, phase = views["fiber_input"]["0"]
    assert amplitude.shape == (4, 4)
    assert np_isnan(phase[-1, -1])
    rotated = renderer._display_array(complex_field, 0, rotate=True)
    assert rotated[0, 0] == pytest.approx(complex_field[0, 0, -1, -1].abs().item())


def test_output_boundary_refuses_existing_directory(renderer) -> None:
    existing = ROOT / "scripts"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        renderer._check_output_boundary(existing)


def test_cli_contract_keeps_cuda_only_and_read_only_flags(renderer) -> None:
    parser = renderer._parse_args
    original = sys.argv
    try:
        sys.argv = ["render_scaled_chen_full_pipeline.py", "--view", "fiber-chain"]
        args = parser()
    finally:
        sys.argv = original
    assert args.view == "fiber-chain"
    assert args.checkpoint_step == 5000
    assert args.hero_index == 55008
    source = SCRIPT.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert '"prior_sampled": False' in source


def np_isnan(value: float) -> bool:
    return bool(torch.isnan(torch.tensor(value)))
