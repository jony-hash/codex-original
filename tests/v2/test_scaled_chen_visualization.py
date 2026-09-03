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
    previous = renderer._verify_full_path_previous_attempt(contract)
    assert config_path.name == "lightgen_optical_manifold_smf_bogt_full_path_visualization_v2.yaml"
    assert review_path.name == "lightgen_optical_manifold_smf_bogt_full_path_visualization_review_v1.json"
    assert review["decision"] == "APPROVED WITH REQUIRED DISPLAY CONTRACT FREEZE"
    assert sorted(parents) == ["s7", "s8", "s9"]
    assert previous["sha256"] == renderer.FULL_PATH_PREVIOUS_MANIFEST_SHA256
    assert contract["sample_panel"]["order"] == list(range(10))
    assert contract["layout_schema"] == "full_optical_path_eight_stage_upright_v1"
    assert contract["sample_panel"]["detail_labels"] == [0, 2, 3, 5, 8]
    assert contract["display"]["reconstruction_range"] == [0.0, 1.0]
    assert contract["display"]["f64_f32_difference_range"] == [0.0, 2.0e-4]


def test_full_path_uses_reference_eight_stage_order_and_pages(renderer) -> None:
    assert renderer.FULL_PATH_STAGE_KEYS == (
        "slm_output",
        "encoder_input",
        "encoder_layer_1",
        "pre_ols",
        "post_ols",
        "decoder_layer_1",
        "decoder_layer_2",
        "cmos_field",
    )
    assert len(renderer.FULL_PATH_STAGE_KEYS) == len(renderer.FULL_PATH_STAGE_LABELS) == 8
    assert renderer.FULL_PATH_PAGE_DIGITS == (
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8, 9),
    )
    assert sorted(digit for page in renderer.FULL_PATH_PAGE_DIGITS for digit in page) == list(range(10))


def test_full_path_upright_display_rotation_does_not_rotate_slm_or_camera(renderer) -> None:
    assert "slm_output" not in renderer.FULL_PATH_DISPLAY_ROTATED_KEYS
    assert "cmos_field" not in renderer.FULL_PATH_DISPLAY_ROTATED_KEYS
    assert renderer.FULL_PATH_DISPLAY_ROTATED_KEYS == frozenset(
        {
            "encoder_input",
            "encoder_layer_1",
            "pre_ols",
            "post_ols",
            "decoder_layer_1",
            "decoder_layer_2",
        }
    )


def test_full_path_main_plate_is_compact_and_equal_aspect(renderer, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    field = torch.ones((10, 1, 4, 4), dtype=torch.complex64)
    values = {key: field.clone() for key in renderer.FULL_PATH_STAGE_KEYS}
    views, _ = renderer._full_path_views(values, sample_positions=range(10))
    captured = {}

    def capture(fig, output_base, **kwargs):
        captured["figure"] = fig
        captured["axes"] = kwargs["axes"]
        captured["include_tiff"] = kwargs["include_tiff"]
        return []

    monkeypatch.setattr(renderer, "_save_figure_bundle", capture)
    renderer._render_full_path_plate(
        views,
        renderer.Path("unused"),
        positions=range(5),
        labels=range(5),
        sample_indices=range(5),
        page_label="digits 0–4",
    )
    figure = captured["figure"]
    assert figure.get_figwidth() <= 7.2
    assert all(float(axis.get_aspect()) == 1.0 for axis in captured["axes"])
    assert captured["include_tiff"] is False
    plt.close(figure)


def test_full_path_contract_rejects_identity_and_review_mutations(renderer, tmp_path, monkeypatch) -> None:
    original_config = renderer.FULL_PATH_CONFIG
    source = original_config.read_text(encoding="utf-8")
    bad_schema = tmp_path / "bad_schema.yaml"
    bad_schema.write_text(
        source.replace(
            "schema: lightgen_optical_manifold_smf_bogt_full_path_visualization_v2",
            "schema: wrong",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(renderer, "FULL_PATH_CONFIG", bad_schema)
    with pytest.raises(RuntimeError, match="identity is mismatched"):
        renderer._load_full_path_contract(bad_schema)

    bad_head = tmp_path / "bad_head.yaml"
    bad_head.write_text(source.replace("expected_head: d07fdaa", "expected_head: deadbeef"), encoding="utf-8")
    monkeypatch.setattr(renderer, "FULL_PATH_CONFIG", bad_head)
    with pytest.raises(RuntimeError, match="identity is mismatched"):
        renderer._load_full_path_contract(bad_head)

    monkeypatch.setattr(renderer, "FULL_PATH_CONFIG", original_config)
    review_source = renderer.FULL_PATH_REVIEW.read_text(encoding="utf-8")
    bad_review = tmp_path / "bad_review.json"
    bad_review.write_text(review_source.replace('"decision": "APPROVED WITH REQUIRED DISPLAY CONTRACT FREEZE"', '"decision": "REJECTED"'), encoding="utf-8")
    monkeypatch.setattr(renderer, "FULL_PATH_REVIEW", bad_review)
    with pytest.raises(RuntimeError, match="review approval is mismatched"):
        renderer._load_full_path_contract(renderer.FULL_PATH_CONFIG)


def test_full_path_parent_and_dirty_path_guards_fail_closed(renderer) -> None:
    contract, _review, _config_path, _review_path = renderer._load_full_path_contract(
        renderer.FULL_PATH_CONFIG
    )
    contract["parents"]["s7"]["validation_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="S7 validation is missing"):
        renderer._verify_full_path_parent_files(contract)
    assert renderer._verify_allowed_dirty_paths([" M docs/CURRENT_STATUS.md"], ["docs/CURRENT_STATUS.md"]) == ["docs/CURRENT_STATUS.md"]
    with pytest.raises(RuntimeError, match="BLOCKED_UNEXPECTED_DIRTY_PATH"):
        renderer._verify_allowed_dirty_paths([" M docs/UNEXPECTED.md"], ["docs/CURRENT_STATUS.md"])


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
