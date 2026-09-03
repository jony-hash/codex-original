from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/visualization/render_scaled_chen_fiber_clear_full_pipeline.py"
VISUALIZATION_SCRIPTS = ROOT / "scripts/visualization"
NATURE_SCRIPTS = Path(r"C:\Users\24567\.codex\skills\nature-figure\scripts")


@pytest.fixture(scope="module")
def renderer():
    sys.path.insert(0, str(NATURE_SCRIPTS))
    sys.path.insert(0, str(VISUALIZATION_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "scaled_chen_fiber_clear_full_pipeline_test_module", SCRIPT
        )
        if spec is None or spec.loader is None:
            raise AssertionError("failed to load VIS2 renderer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(VISUALIZATION_SCRIPTS))
        sys.path.remove(str(NATURE_SCRIPTS))


def test_vis2_contract_and_arm_order(renderer) -> None:
    assert renderer.TASK_ID == "H-LOMA-CBOGT-FIBER-VIS2"
    assert renderer.CLASSIFICATION == "canonical_clear_full_pipeline_visualization_completed"
    assert renderer.EXPECTED_BRANCH == "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
    assert renderer.EXPECTED_START_HEAD == "4d2933cb5b49302dd2114d49b56c905ef9d10ca7"
    assert renderer.ARM_ORDER == ("SMF", "FMF", "MMF-M6")
    assert renderer.ARM_TOTAL_MODES == {"SMF": 100, "FMF": 200, "MMF-M6": 600}
    assert renderer.COMMON_STAGE_ORDER == (
        "input",
        "slm_command",
        "slm_output",
        "encoder_input",
        "encoder_layer_1",
        "pre_ols",
    )


def test_input_quality_is_input_only_and_rejects_border_contact(renderer) -> None:
    image = torch.zeros((1, 28, 28), dtype=torch.float32)
    image[0, 5:23, 10:15] = 1.0
    result = renderer._raw_input_quality(image)
    assert result["hard_pass"] is True
    assert result["border_touch_pixels"] == 0
    assert result["bbox_height"] == 18
    assert result["bbox_width"] == 5
    touching = image.clone()
    touching[0, 0, 10] = 1.0
    assert renderer._raw_input_quality(touching)["hard_pass"] is False


def test_clear_manifest_locks_input_only_selection(renderer) -> None:
    images = torch.zeros((60000, 1, 28, 28), dtype=torch.float32)
    labels = torch.arange(60000, dtype=torch.long) % 10
    images[:, 0, 5:23, 10:15] = 1.0
    candidates, selected = renderer._candidate_pool(images, labels)
    # The synthetic fixture is not a visualization input; this test only
    # verifies that the contract function requires the complete split and
    # that downstream values cannot enter the manifest schema.
    assert set(candidates) == set(range(10))
    assert set(selected) == set(range(10))
    manifest = renderer._clear_panel_manifest(images, labels, candidates, selected)
    assert manifest["selection_method"] == "input_only"
    assert manifest["selected_before_fiber_evaluation"] is True
    assert manifest["downstream_outputs_used_for_selection"] is False
    assert all("mpcdar" not in item for item in manifest["selected"])


def test_modal_metadata_keeps_family_axis_and_synthetic_labels(renderer) -> None:
    power = torch.ones((6, 100), dtype=torch.float32)
    family = torch.ones((6, 100), dtype=torch.complex64)
    labels = renderer._modal_labels(
        {
            "arm": "MMF-M6",
            "family_order": ["LP01", "LP11a", "ETA1", "XI2", "XIETA", "ETA2"],
        }
    )
    assert tuple(power.shape) == (6, 100)
    assert tuple(family.shape) == (6, 100)
    assert labels[0] == "LP01"
    assert labels[2] == "ETA1 [synthetic]"
    assert labels[-1] == "ETA2 [synthetic]"


def test_phase_mask_and_shared_scale_contract(renderer) -> None:
    value = torch.ones((10, 1, 4, 4), dtype=torch.complex64)
    value[0, 0, 0, 0] = 0.001 + 0.0j
    phase, threshold, valid = renderer._phase_view(value, 0, "pre_ols")
    assert threshold == pytest.approx(0.01)
    assert np.isnan(phase[-1, -1])
    assert valid == pytest.approx(15 / 16)
    assert np.nanmin(phase) >= -np.pi
    assert np.nanmax(phase) <= np.pi


def test_compact_layout_contract_separates_arms_and_digit_pages(renderer) -> None:
    assert renderer.VIS2_LAYOUT_SCHEMA == "organized_compact_arm_separated_v3"
    assert renderer.ARM_COLORS["SMF"] != renderer.ARM_COLORS["FMF"]
    assert renderer.ARM_COLORS["FMF"] != renderer.ARM_COLORS["MMF-M6"]
    assert renderer.ARM_COLORS["SMF"] != renderer.ARM_COLORS["MMF-M6"]
    assert renderer.DIGIT_PAGE_RANGES == ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
    assert tuple(renderer.STAGE_GROUPS) == ("frontend", "fiber_decoder", "detector_reconstruction")
    assert all(len(columns) <= 6 for columns in renderer.STAGE_GROUPS.values())
    assert renderer.FIGURE_EXPORT_SUFFIXES == (".png", ".pdf", ".svg")


def test_three_photo_layout_contract_matches_same_digit_comparison(renderer) -> None:
    assert renderer.THREE_PHOTO_LAYOUT_SCHEMA == "same_digit_three_arm_comparison_v1"
    assert renderer.THREE_PHOTO_PAGE_RANGES == ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))
    assert tuple(renderer.THREE_PHOTO_COLUMN_KEYS) == (
        "standard_input",
        "SMF",
        "FMF",
        "MMF-M6",
    )
    assert renderer.THREE_PHOTO_OUTPUT_KEY == "mpcdar_rank16_float32"
    assert [position for page in renderer.THREE_PHOTO_PAGE_RANGES for position in page] == list(range(10))
    assert max(map(len, renderer.THREE_PHOTO_PAGE_RANGES)) - min(
        map(len, renderer.THREE_PHOTO_PAGE_RANGES)
    ) <= 1
    assert all(len(renderer.THREE_PHOTO_COLUMN_KEYS) == 4 for _ in renderer.THREE_PHOTO_PAGE_RANGES)


def test_three_photo_rows_match_one_input_position_across_all_arms(renderer) -> None:
    labels = torch.arange(10, dtype=torch.long)
    indices = torch.arange(10, dtype=torch.long) + 55000
    rows = renderer._three_photo_row_records(labels, indices, renderer.THREE_PHOTO_PAGE_RANGES[0])
    assert [row["digit"] for row in rows] == [0, 1, 2, 3]
    assert [row["dataset_index"] for row in rows] == [55000, 55001, 55002, 55003]
    for row, position in zip(rows, renderer.THREE_PHOTO_PAGE_RANGES[0], strict=True):
        assert row["position"] == position
        assert {column["position"] for column in row["columns"].values()} == {position}
        assert row["columns"]["standard_input"]["source"] == "images_28x28"
        for arm in renderer.ARM_ORDER:
            assert row["columns"][arm]["source"] == "arm_data[arm][mpcdar_rank16_float32]"


def test_three_photo_render_contract_is_compact_and_spatially_equal(renderer) -> None:
    assert renderer.THREE_PHOTO_PAGE_WIDTH_INCHES <= 7.2
    assert renderer.THREE_PHOTO_SPATIAL_ASPECT == "equal"
    assert renderer.ARM_COLORS["SMF"] != renderer.ARM_COLORS["FMF"]
    assert renderer.ARM_COLORS["FMF"] != renderer.ARM_COLORS["MMF-M6"]
    assert renderer.ARM_COLORS["SMF"] != renderer.ARM_COLORS["MMF-M6"]
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'aspect="equal"' in source


def test_three_photo_output_contract_excludes_tiff(renderer) -> None:
    assert renderer.THREE_PHOTO_FIGURE_EXPORT_SUFFIXES == (".png", ".pdf", ".svg")
    assert all(
        suffix.lower() not in {".tif", ".tiff"}
        for suffix in renderer.THREE_PHOTO_FIGURE_EXPORT_SUFFIXES
    )


def test_process_flow_layout_uses_reference_style_stages_and_three_arm_lanes(renderer) -> None:
    assert renderer.PROCESS_FLOW_LAYOUT_SCHEMA == "same_digit_interleaved_three_arm_process_flow_v1"
    assert renderer.PROCESS_FLOW_ARTIFACT_LAYOUT_SCHEMA == "vis2_process_flow_interleaved_v1"
    assert renderer.PROCESS_FLOW_PAGE_RANGES == ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))
    assert renderer.PROCESS_FLOW_ARM_LANES == renderer.ARM_ORDER
    assert renderer.PROCESS_FLOW_ROW_ORDER == "digit_then_arm_then_amplitude_phase"
    assert renderer.PROCESS_FLOW_ROWS_PER_DIGIT == 6
    assert renderer.PROCESS_FLOW_STAGE_KEYS == (
        "input",
        "slm_command",
        "slm_output",
        "encoder_input",
        "encoder_layer_1",
        "pre_ols",
        "modal",
        "post_fiber",
        "decoder_layer_1",
        "decoder_layer_2",
        "cmos_field",
        "cmos_intensity",
        "mpcdar",
    )
    assert len(renderer.PROCESS_FLOW_STAGE_KEYS) == 13
    assert renderer.PROCESS_FLOW_PAGE_WIDTH_INCHES <= 12.0
    assert renderer.PROCESS_FLOW_SPATIAL_ASPECT == "equal"


def test_process_flow_rows_keep_digit_position_and_arm_identity(renderer) -> None:
    labels = torch.arange(10, dtype=torch.long)
    indices = torch.arange(10, dtype=torch.long) + 55000
    rows = renderer._process_flow_row_records(
        labels,
        indices,
        renderer.PROCESS_FLOW_PAGE_RANGES[1],
        "FMF",
    )
    assert [row["digit"] for row in rows] == [4, 5, 6]
    assert [row["position"] for row in rows] == [4, 5, 6]
    assert [row["dataset_index"] for row in rows] == [55004, 55005, 55006]
    assert all(row["arm"] == "FMF" for row in rows)


def test_process_flow_render_rows_interleave_arms_within_each_digit(renderer) -> None:
    labels = torch.arange(10, dtype=torch.long)
    indices = torch.arange(10, dtype=torch.long) + 55000
    rows = renderer._process_flow_render_row_records(
        labels,
        indices,
        renderer.PROCESS_FLOW_PAGE_RANGES[1],
    )
    assert [
        (row["digit"], row["arm"], row["phase"])
        for row in rows
    ] == [
        (4, "SMF", "amplitude"),
        (4, "SMF", "phase"),
        (4, "FMF", "amplitude"),
        (4, "FMF", "phase"),
        (4, "MMF-M6", "amplitude"),
        (4, "MMF-M6", "phase"),
        (5, "SMF", "amplitude"),
        (5, "SMF", "phase"),
        (5, "FMF", "amplitude"),
        (5, "FMF", "phase"),
        (5, "MMF-M6", "amplitude"),
        (5, "MMF-M6", "phase"),
        (6, "SMF", "amplitude"),
        (6, "SMF", "phase"),
        (6, "FMF", "amplitude"),
        (6, "FMF", "phase"),
        (6, "MMF-M6", "amplitude"),
        (6, "MMF-M6", "phase"),
    ]
    assert [row["dataset_index"] for row in rows] == [
        55004,
        55004,
        55004,
        55004,
        55004,
        55004,
        55005,
        55005,
        55005,
        55005,
        55005,
        55005,
        55006,
        55006,
        55006,
        55006,
        55006,
        55006,
    ]
    assert all(
        set(row["stage_position"].values()) == {row["position"]}
        for row in rows
    )


def test_process_flow_manifest_has_three_pages_and_all_stage_columns(renderer) -> None:
    manifest = renderer._process_flow_figure_manifest()
    assert [page["digits"] for page in manifest["pages"]] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert all(
        len(page["columns"]) == len(renderer.PROCESS_FLOW_STAGE_KEYS)
        for page in manifest["pages"]
    )
    assert manifest["arm_lanes"] == list(renderer.PROCESS_FLOW_ARM_LANES)
    assert manifest["output_field"] == "mpcdar_rank16_float32"
    assert manifest["input_field"] == "images_28x28"
    assert manifest["row_order"] == renderer.PROCESS_FLOW_ROW_ORDER
    assert manifest["rows_per_digit"] == renderer.PROCESS_FLOW_ROWS_PER_DIGIT
    assert manifest["row_structure"] == renderer.PROCESS_FLOW_ROW_STRUCTURE


def test_process_flow_entrypoint_and_layout_manifest_are_declared(renderer) -> None:
    assert callable(renderer._render_process_flow_from_panel)
    layout = renderer._process_flow_layout_manifest()
    assert layout["layout_schema"] == "same_digit_interleaved_three_arm_process_flow_v1"
    assert layout["arm_lanes"] == ["SMF", "FMF", "MMF-M6"]
    assert layout["input_field"] == "images_28x28"
    assert layout["output_field"] == "mpcdar_rank16_float32"
    assert layout["tiff_exported"] is False
    assert layout["spatial_aspect"] == "equal"
    assert layout["row_order"] == renderer.PROCESS_FLOW_ROW_ORDER
    assert layout["rows_per_digit"] == renderer.PROCESS_FLOW_ROWS_PER_DIGIT
    assert layout["row_structure"] == renderer.PROCESS_FLOW_ROW_STRUCTURE
    assert [column["key"] for column in layout["stage_columns"]] == list(
        renderer.PROCESS_FLOW_STAGE_KEYS
    )


def test_organized_layout_and_orientation_contract(renderer) -> None:
    assert renderer.ARTIFACT_FOLDERS == {
        "digit_visualization": "digit_visualization",
        "pipeline_overview": "pipeline_overview",
        "fiber_information": "fiber_information",
        "metadata_qa": "metadata_qa",
    }
    policy = renderer.ORIENTATION_POLICY
    assert policy["row_label_rotation_degrees"] == 0
    assert policy["image_space"]["origin"] == "upper"
    assert policy["optical_spatial"]["origin"] == "lower"
    assert policy["optical_spatial"]["display_rotation_degrees"] == 180
    assert policy["modal_family_port"]["strategy"] == "explicit_nonspatial_matrix"
    assert callable(renderer._horizontal_row_label)
    figure, axis = renderer.plt.subplots()
    try:
        label = renderer._horizontal_row_label(axis, "digit 0\nidx 1\namplitude")
        assert label.get_rotation() == 0
    finally:
        renderer.plt.close(figure)


def test_figure_manifest_uses_content_relative_paths(renderer) -> None:
    figures = renderer._figure_manifest_names()
    assert figures["figure_0"].startswith("digit_visualization/")
    assert all(
        name.startswith("pipeline_overview/")
        for name in figures["reconstruction_overview"]
    )
    for arm in renderer.ARM_ORDER:
        for group in renderer.STAGE_GROUPS:
            assert all(
                name.startswith(f"fiber_information/{arm}/{group}/")
                for name in figures["arm_stage_groups"][arm][group]
            )
    assert all("\\" not in name for name in _flatten_figure_names(figures))


def test_renderer_labels_and_images_have_no_implicit_vertical_orientation(renderer) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "set_ylabel(" not in source
    assert "rotation=-90" not in source
    assert source.count("origin=IMAGE_ORIGIN") >= 8
    assert source.count("origin=OPTICAL_ORIGIN if spatial else MODAL_ORIGIN") == 2


def _flatten_figure_names(figures: dict) -> list[str]:
    names = [figures["figure_0"], *figures["reconstruction_overview"]]
    for arm_groups in figures["arm_stage_groups"].values():
        for group_names in arm_groups.values():
            names.extend(group_names)
    return names


def test_spatial_display_axes_keep_equal_geometry(renderer) -> None:
    figure, axis = renderer.plt.subplots()
    try:
        renderer._finish_image_axis(axis)
        assert axis.get_aspect() == 1.0
        assert axis.get_box_aspect() == pytest.approx(1.0)
    finally:
        renderer.plt.close(figure)


def test_panel_rerender_entrypoint_is_declared(renderer) -> None:
    assert callable(renderer._load_panel_payload)
    assert callable(renderer._render_from_panel)


def test_panel_source_path_resolver_accepts_organized_source(renderer, tmp_path: Path) -> None:
    source_dir = tmp_path / "organized-source"
    metadata_dir = source_dir / "metadata_qa"
    metadata_dir.mkdir(parents=True)
    for name in (
        "clear_panel_manifest.json",
        "clear_panel_review.json",
        "comparison_manifest.json",
        "panel.pt",
    ):
        path = metadata_dir / name
        if path.suffix == ".pt":
            torch.save({"fixture": True}, path)
        else:
            path.write_text("{}", encoding="utf-8")
    (source_dir / "visualization_manifest.json").write_text(
        json.dumps(
            {
                "comparison_manifest": {"path": "metadata_qa/comparison_manifest.json"},
                "output_files": [{"path": "metadata_qa/panel.pt"}],
            }
        ),
        encoding="utf-8",
    )
    (metadata_dir / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "clear_panel": {
                    "manifest": "metadata_qa/clear_panel_manifest.json",
                    "review": "metadata_qa/clear_panel_review.json",
                }
            }
        ),
        encoding="utf-8",
    )
    paths = renderer._resolve_panel_source_paths(source_dir)
    assert paths["panel"] == metadata_dir / "panel.pt"
    assert paths["clear_manifest"] == metadata_dir / "clear_panel_manifest.json"
    assert paths["review"] == metadata_dir / "clear_panel_review.json"
    assert paths["comparison"] == metadata_dir / "comparison_manifest.json"


def _write_minimal_panel_source(
    renderer,
    source_dir: Path,
    *,
    review_decision: str = "APPROVED",
    panel_schema: str | None = None,
    panel_sha256: str | None = None,
) -> None:
    source_dir.mkdir()
    clear_manifest = {
        "schema": "lightgen_scaled_chen_fiber_vis2_clear_panel_manifest.v1",
        "task_id": renderer.TASK_ID,
        "selection_method": "input_only",
        "selected_before_fiber_evaluation": True,
        "downstream_outputs_used_for_selection": False,
        "selected_panel_order": list(range(10)),
        "selected": [
            {"digit": digit, "label": digit, "dataset_index": digit}
            for digit in range(10)
        ],
    }
    clear_manifest_path = source_dir / "clear_panel_manifest.json"
    clear_manifest_path.write_text(json.dumps(clear_manifest), encoding="utf-8")
    review = {
        "schema": "lightgen_scaled_chen_fiber_vis2_clear_panel_review.v1",
        "task_id": renderer.TASK_ID,
        "decision": review_decision,
        "selection_method": "input_only",
        "downstream_outputs_used_for_selection": False,
        "selected_before_fiber_evaluation": True,
        "clear_panel_manifest_sha256": renderer.file_sha256(clear_manifest_path),
    }
    (source_dir / "clear_panel_review.json").write_text(json.dumps(review), encoding="utf-8")
    panel_path = source_dir / "panel.pt"
    torch.save(
        {
            "schema": panel_schema or renderer.PANEL_SCHEMA,
            "task_id": renderer.TASK_ID,
            "classification": renderer.CLASSIFICATION,
            "clear_panel_manifest_sha256": renderer.file_sha256(clear_manifest_path),
            "full_mode_banks_stored": False,
        },
        panel_path,
    )
    comparison = {
        "schema": "lightgen_scaled_chen_fiber_vis2_comparison_manifest.v1",
        "task_id": renderer.TASK_ID,
        "status": "completed",
        "classification": renderer.CLASSIFICATION,
        "clear_panel": {
            "manifest": "clear_panel_manifest.json",
            "review": "clear_panel_review.json",
            "manifest_sha256": renderer.file_sha256(clear_manifest_path),
            "review_sha256": renderer.file_sha256(source_dir / "clear_panel_review.json"),
            "review_decision": "APPROVED",
            "selection_method": "input_only",
            "selected_before_fiber_evaluation": True,
            "downstream_outputs_used_for_selection": False,
        },
        "arm_order": list(renderer.ARM_ORDER),
        "arm_contracts": {},
    }
    comparison_path = source_dir / "comparison_manifest.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    visualization = {
        "schema": "lightgen_scaled_chen_fiber_vis2_visualization_manifest.v1",
        "task_id": renderer.TASK_ID,
        "status": "completed",
        "classification": renderer.CLASSIFICATION,
        "comparison_manifest": {
            "path": "comparison_manifest.json",
            "sha256": renderer.file_sha256(comparison_path),
        },
        "vis1_authority": {
            "status": "unchanged",
            "not_overwritten": True,
            "not_recomputed": True,
            "not_modified": True,
        },
        "output_files": [
            {
                "path": "panel.pt",
                "sha256": panel_sha256 or renderer.file_sha256(panel_path),
            }
        ],
    }
    (source_dir / "visualization_manifest.json").write_text(
        json.dumps(visualization), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("case", "expected_error"),
    (
        ("review", "BLOCKED_PANEL_SOURCE_REVIEW_INVALID"),
        ("panel_hash", "BLOCKED_PANEL_SOURCE_PANEL_HASH_INVALID"),
        ("panel_schema", "BLOCKED_PANEL_SOURCE_PAYLOAD_CONTRACT_CHANGED"),
    ),
)
def test_panel_loader_rejects_review_hash_and_schema_mismatch(
    renderer,
    tmp_path: Path,
    case: str,
    expected_error: str,
) -> None:
    source_dir = tmp_path / case
    kwargs = {
        "review_decision": "REJECTED" if case == "review" else "APPROVED",
        "panel_sha256": "tampered" if case == "panel_hash" else None,
        "panel_schema": "wrong.schema" if case == "panel_schema" else None,
    }
    _write_minimal_panel_source(renderer, source_dir, **kwargs)
    with pytest.raises(RuntimeError, match=expected_error):
        renderer._load_panel_payload(source_dir)


@pytest.mark.parametrize(
    ("case", "expected_error"),
    (
        ("clear_hash", "BLOCKED_PANEL_SOURCE_CLEAR_MANIFEST_HASH_INVALID"),
        ("source_binding", "BLOCKED_PANEL_SOURCE_VISUALIZATION_INVALID"),
    ),
)
def test_panel_loader_rejects_clear_manifest_and_source_binding_mismatch(
    renderer,
    tmp_path: Path,
    case: str,
    expected_error: str,
) -> None:
    source_dir = tmp_path / case
    _write_minimal_panel_source(renderer, source_dir)
    if case == "clear_hash":
        path = source_dir / "comparison_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["clear_panel"]["manifest_sha256"] = "tampered"
    else:
        path = source_dir / "visualization_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["comparison_manifest"]["sha256"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match=expected_error):
        renderer._load_panel_payload(source_dir)


def test_renderer_does_not_contain_training_or_visualization_bypass(renderer) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "optimizer.step" not in source
    assert "loss.backward" not in source
    assert "generate_from_prior" not in source
    assert "downstream_outputs_used_for_selection" in source
    assert "full_mode_banks_stored" in source


def test_vis2_preserves_vis1_authority_and_governance(renderer) -> None:
    assert renderer.VIS1_OUTPUT.name == "lightgen_scaled_chen_fiber_horizontal_visualization_v1_attempt_06"
    assert renderer.UNIQUE_NEXT_ACTION == "manual_review_VIS2_figures"
    assert renderer.AWAITING_NEXT_ACTION == "manual_review_clear_0_9_contact_sheet"
