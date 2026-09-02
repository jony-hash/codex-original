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
    assert renderer.FIBER_STAGE_AVAILABILITY["cmos_field"]["phase_available"] is True
    assert renderer.FIBER_STAGE_AVAILABILITY["cmos_field"]["detector_readout"] == "intensity_only"
    assert renderer.FIBER_STAGE_AVAILABILITY["fiber_input"]["alias_of"] == "encoder_layer_2"
    assert renderer.FIBER_STAGE_AVAILABILITY["fiber_output"]["alias_of"] == "decoder_input"


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


def test_real_command_and_detector_are_not_treated_as_complex_phase(renderer) -> None:
    real = torch.zeros((10, 1, 4, 4), dtype=torch.float32)
    real[:, :, 1:3, 1:3] = 1.0
    assert not torch.is_complex(real)
    assert renderer.FIBER_STAGE_AVAILABILITY["cmos_field"]["detector_readout"] == "intensity_only"
    assert renderer._stage_views(real)[1].shape == (4, 4)
    assert torch.isnan(torch.as_tensor(renderer._stage_views(real)[1])).all()


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


def test_full_path_contract_locks_registered_parents_and_panel(renderer) -> None:
    contract, review, config_path, review_path = renderer._load_full_path_contract(
        renderer.FULL_PATH_CONFIG
    )
    parents = renderer._verify_full_path_parent_files(contract)
    assert config_path.name == "lightgen_optical_manifold_smf_bogt_full_path_visualization_v1.yaml"
    assert review_path.name == "lightgen_optical_manifold_smf_bogt_full_path_visualization_review_v1.json"
    assert review["decision"] == "APPROVED WITH REQUIRED DISPLAY CONTRACT FREEZE"
    assert sorted(parents) == ["s7", "s8", "s9"]
    assert contract["sample_panel"]["order"] == list(range(10))
    assert contract["sample_panel"]["detail_labels"] == [0, 2, 3, 5, 8]
    assert contract["display"]["reconstruction_range"] == [0.0, 1.0]
    assert contract["display"]["f64_f32_difference_range"] == [0.0, 2.0e-4]


def test_full_path_views_use_stage_shared_scales_and_exact_mask(renderer) -> None:
    real = torch.zeros((10, 1, 4, 4), dtype=torch.float32)
    real[:, :, 1:3, 1:3] = 1.0
    values = {}
    for key in renderer.FULL_PATH_STAGE_KEYS:
        if key in renderer.FULL_PATH_REAL_KEYS:
            values[key] = real.clone()
        elif key in renderer.FULL_PATH_MODAL_KEYS:
            values[key] = torch.ones((10, 100), dtype=torch.complex64)
        else:
            values[key] = torch.ones((10, 1, 4, 4), dtype=torch.complex64)
    values["encoder_input"][0, 0, 0, 0] = 0.001 + 0.0j
    values["encoder_input"][1, 0, 0, 0] = 10.0 + 0.0j
    original = values["encoder_input"].clone()
    views, scales = renderer._full_path_views(values, sample_positions=range(10))
    assert scales["target"] == 1.0
    assert scales["encoder_input"] > 1.0
    assert views["encoder_input"]["1"]["phase_threshold"] == pytest.approx(0.1)
    assert bool(torch.isnan(torch.tensor(views["encoder_input"]["0"]["phase"][3, 3])))
    assert torch.equal(values["encoder_input"], original)


def test_full_path_route_and_reconstruction_reshape_are_locked(renderer) -> None:
    original = sys.argv
    try:
        sys.argv = [
            "render_scaled_chen_full_pipeline.py",
            "--view",
            "full-path-reconstruction",
        ]
        args = renderer._parse_args()
    finally:
        sys.argv = original
    assert args.view == "full-path-reconstruction"
    assert args.visualization_config == renderer.FULL_PATH_CONFIG
    flat = torch.zeros((2, 40000), dtype=torch.float64)
    image = renderer._as_reconstruction_images(flat, "synthetic")
    assert image.shape == (2, 1, 200, 200)
    with pytest.raises(RuntimeError, match="invalid reconstruction shape"):
        renderer._as_reconstruction_images(torch.zeros((2, 12)), "bad")


def np_isnan(value: float) -> bool:
    return bool(torch.isnan(torch.tensor(value)))
