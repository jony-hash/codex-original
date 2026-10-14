from __future__ import annotations

import torch
from torch import nn

from ldm_project.v2.evidence import atomic_torch_save_new
from scripts.research.lightgen import execute_lightgen_2d_existing_400_frozen_replay_v1 as replay


def test_400_view_clamps_then_pools_with_one_fixed_transform() -> None:
    raw = torch.full((2, 1, 400, 400), 0.5)
    raw[0, :, :200, :200] = 2.0
    raw[1, :, 200:, 200:] = -1.0

    view = replay.raw_400_to_view(raw)

    assert view.shape == (2, 1, 28, 28)
    assert torch.equal(view[0, 0, :14, :14], torch.ones(14, 14))
    assert torch.equal(view[0, 0, 14:, 14:], torch.full((14, 14), 0.5))
    assert torch.equal(view[1, 0, 14:, 14:], torch.zeros(14, 14))
    assert replay.historical_raw_indices().tolist() == [
        200 * digit + offset for digit in range(10) for offset in range(5)
    ]


def test_generated_payload_rejects_wrong_label_order_and_accepts_original_layout() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    ols = {
        "schema": "lightgen_400_mnist_generated_ols.v1",
        "coefficients": torch.zeros((2000, 100), dtype=torch.complex64),
        "requested_labels": labels,
    }
    visuals = {
        "schema": "lightgen_400_mnist_generated_visuals.v1",
        "evaluator_views_28x28": torch.zeros((2000, 1, 28, 28)),
        "raw_cmos_examples_400x400": torch.zeros((50, 1, 400, 400)),
        "requested_labels": labels.clone(),
        "examples_per_class": 5,
    }
    assert replay.validate_generated_payloads(ols, visuals)[0].shape == (2000, 100)
    ols["requested_labels"] = labels.roll(1)
    try:
        replay.validate_generated_payloads(ols, visuals)
    except replay.ReplayError as exc:
        assert "original 200-per-class order" in str(exc)
    else:
        raise AssertionError("out-of-order class labels were accepted")


def test_comparison_uses_fixed_tolerance_and_reports_row_mismatches() -> None:
    reference = torch.zeros((3, 1, 2, 2))
    close = reference.clone()
    close[1, 0, 0, 0] = replay.ATOL / 2
    result = replay.compare_tensors(close, reference)
    assert result["matches"] is True
    assert result["matching_rows"] == 3

    changed = close.clone()
    changed[2, 0, 0, 0] = 0.1
    result = replay.compare_tensors(changed, reference)
    assert result["matches"] is False
    assert result["mismatch_indices"] == [2]
    assert abs(result["max_abs_error"] - 0.1) < 1.0e-7


def test_confusion_metrics_include_counts_prediction_mass_and_wilson_intervals() -> None:
    labels = torch.arange(10).repeat_interleave(10)
    predictions = labels.clone()
    predictions[0] = 1

    metrics = replay.confusion_metrics(predictions, labels)

    assert metrics["count"] == 100
    assert metrics["correct_by_class"][0] == 9
    assert metrics["predicted_class_counts"][0] == 9
    assert metrics["per_class"][0]["wilson_95"][0] < 0.9
    assert metrics["per_class"][0]["wilson_95"][1] > 0.9
    assert len(metrics["confusion"]) == 10


def test_evaluator_forward_can_be_checked_without_parameter_or_buffer_changes() -> None:
    class ReadOnlyEvaluator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 10)
            self.register_buffer("scale", torch.tensor(2.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x.flatten(1)) * self.scale

    model = ReadOnlyEvaluator().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = replay._tree_snapshot(model)

    with torch.no_grad():
        logits = replay.evaluator_logits(model, torch.ones((3, 1, 2, 2)))

    assert logits.shape == (3, 10)
    replay._assert_module_frozen(model, before, "toy evaluator")


def test_duplicate_and_nearest_generation_statistics_are_exact_and_descriptive() -> None:
    values = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
        ]
    )
    nearest, indices = replay._pairwise_rmse(values)
    assert replay._duplicates(values) == 1
    assert torch.equal(nearest, torch.tensor([0.0, 0.0, 1.0]))
    assert indices.tolist()[:2] == [1, 0]

    nearest_targets = replay._nearest_to_targets(
        values,
        torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 1.0], [1.0, 1.0]]]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1, 1]),
    )
    assert nearest_targets["nearest_any_target_index"].tolist() == [0, 0, 1]
    assert nearest_targets["nearest_requested_class_target_index"].tolist() == [0, 1, 1]


def test_plane_power_and_crop_retention_are_kept_separate_and_combined_by_row() -> None:
    plane = {
        "plane_id": "sensor",
        "shape": [2, 2],
        "pixel_pitch_m": [3.0e-6, 3.0e-6],
        "wavelength_m": 532.0e-9,
        "z_m": 0.002,
        "power_proxy_by_row": [1.0, 3.0],
    }
    first = replay._combine_plane_measurements(
        [
            {
                "planes": [plane],
                "crop_retention": {"decoder_crop_retention": [[0.8, 0.7], [0.6, 0.5]]},
            }
        ]
    )
    second = replay._combine_plane_measurements(
        [
            {
                "planes": [{**plane, "power_proxy_by_row": [5.0]}],
                "crop_retention": {"decoder_crop_retention": [[0.4, 0.3]]},
            }
        ]
    )

    merged = replay._merge_chunk_plane_measurements(
        [
            {"optical_plane_measurement": first},
            {"optical_plane_measurement": second},
        ]
    )

    assert merged["planes"][0]["power_proxy_by_row"] == [1.0, 3.0, 5.0]
    assert merged["planes"][0]["power_proxy_mean"] == 3.0
    assert merged["crop_retention"]["decoder_crop_retention"]["per_row_by_layer"] == [
        [0.8, 0.7],
        [0.6, 0.5],
        [0.4, 0.3],
    ]


def test_plane_power_reduces_only_spatial_axes_and_keeps_per_row_scalars() -> None:
    class Plane:
        pixel_pitch_m = (2.0, 3.0)

        def to_dict(self) -> dict[str, object]:
            return {
                "plane_id": "toy",
                "shape": [2, 2],
                "pixel_pitch_m": [2.0, 3.0],
                "z_m": 0.0,
                "wavelength_m": 532.0e-9,
            }

    class Field:
        plane = Plane()
        tensor = torch.tensor(
            [[[[1.0 + 0.0j, 0.0j], [0.0j, 0.0j]]], [[[2.0 + 0.0j, 0.0j], [0.0j, 0.0j]]]],
            dtype=torch.complex64,
        )

    class Result:
        optical_trace = (Field(),)
        crop_statistics = {"decoder_crop_retention": torch.tensor([[0.8], [0.6]])}

    measured = replay._plane_measurements(Result())

    assert measured["planes"][0]["power_proxy_by_row"] == [6.0, 24.0]
    assert torch.allclose(
        torch.tensor(measured["crop_retention"]["decoder_crop_retention"]),
        torch.tensor([[0.8], [0.6]]),
    )


def test_class_1_and_7_focus_metrics_include_margin_and_cross_evaluator_disagreement() -> None:
    labels = torch.tensor([1, 1, 7, 7])
    first = torch.zeros((4, 10))
    first[0, 1] = 2.0
    first[1, 2] = 2.0
    first[2, 7] = 3.0
    first[3, 7] = 1.0
    second = first.clone()
    second[1, 1] = 3.0

    focus = replay._focus_class_rows({"E_D": first, "E_I": second}, labels)

    assert focus["classes"]["1"]["evaluator_summary"]["E_D"]["correct"] == 1
    assert focus["classes"]["1"]["pairwise_evaluator_disagreement_rows"] == 1
    assert focus["classes"]["7"]["evaluator_summary"]["E_D"]["mean_true_vs_best_other_logit_margin"] == 2.0


def test_chunk_resume_recovers_published_chunk_and_rejects_wrong_fixed_rows(tmp_path) -> None:
    labels = torch.arange(10).repeat_interleave(10)
    chunk_labels = labels[:100]
    atomic_torch_save_new(
        tmp_path / "generated_chunk_0000.pt",
        {
            "schema": "generated_chunk.v1",
            "row_start": 0,
            "row_stop": 100,
            "row_indices": torch.arange(100),
            "labels": chunk_labels,
            "ols": torch.zeros((100, 100), dtype=torch.complex64),
        },
    )
    manifest: dict[str, object] = {"completed_generated_chunks": []}

    chunks = replay._load_existing_chunks(tmp_path, "generated", 2000, labels, manifest)

    assert len(chunks) == 1
    assert manifest["completed_generated_chunks"] == [0]

    wrong_manifest: dict[str, object] = {"completed_generated_chunks": [1]}
    try:
        replay._load_existing_chunks(tmp_path, "generated", 2000, labels, wrong_manifest)
    except replay.ReplayError as exc:
        assert "missing chunk" in str(exc) or "row order" in str(exc)
    else:
        raise AssertionError("manifest referring to an absent row block was accepted")


def test_decision_has_confirmed_discrepancy_and_invalid_control_exits() -> None:
    metrics = {"macro_recall": 0.80, "recall_by_class": [0.8] * 10}
    raw = {
        "finite": True,
        "nonzero_images": 2000,
        "constant_images": 0,
        "fraction_pixels_above_1": 0.005,
    }
    assert replay._decision(True, True, metrics, raw, True, True, True) == (
        "VALID_COMPLETED",
        "FROZEN_REPLAY_CONFIRMED",
        True,
    )
    assert replay._decision(False, True, metrics, raw, True, True, True)[1] == "FROZEN_REPLAY_DISCREPANCY"
    assert replay._decision(True, True, metrics, raw, True, True, False) == (
        "INVALID_HARNESS",
        None,
        False,
    )


def test_step600_configuration_is_locked_to_historical_400_proxy() -> None:
    root = replay.ROOT
    config = replay.load_config(root / replay.DEFAULT_CONFIG)
    observed = replay.validate_system_config(config)
    assert observed["propagation_grid_size"] == 512
    assert observed["slm_shape"] == [400, 400]
    assert observed["encoder_layers"] == 2
    assert observed["decoder_layers"] == 4


def test_reference_population_rejects_nontrain_rows_and_label_mismatch() -> None:
    all_labels = torch.arange(10).repeat_interleave(6000)
    train_rows, dev_rows = replay.stratified_mnist_indices(
        all_labels, seed=1234, validation_per_label=500
    )
    row_ids = torch.cat(
        [train_rows[all_labels[train_rows] == digit][:30] for digit in range(10)]
    )
    labels = all_labels[row_ids].clone()

    replay._validate_reference_population(row_ids, labels, all_labels)

    nontrain_rows = row_ids.clone()
    nontrain_rows[0] = dev_rows[all_labels[dev_rows] == 0][0]
    try:
        replay._validate_reference_population(nontrain_rows, labels, all_labels)
    except replay.ReplayError as exc:
        assert "noncanonical TRAIN" in str(exc)
    else:
        raise AssertionError("a DEV row was accepted in the fixed TRAIN reference")

    wrong_labels = labels.clone()
    wrong_labels[0] = 1
    try:
        replay._validate_reference_population(row_ids, wrong_labels, all_labels)
    except replay.ReplayError as exc:
        assert "labels differ" in str(exc)
    else:
        raise AssertionError("a mismatched reference label was accepted")


def test_completed_resume_checks_all_chunks_and_returns_without_forward(tmp_path) -> None:
    class FrozenNoForward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1), requires_grad=False)

        def forward(self, *_args: torch.Tensor) -> torch.Tensor:
            raise AssertionError("completed resume must not run another forward")

    output_dir = tmp_path / "attempt"
    output_dir.mkdir()
    model = FrozenNoForward()
    evaluator = FrozenNoForward()
    evaluators = {"E_D": evaluator}
    sampler = {"class_mean": torch.ones((10, 100)), "seed": 20260928}
    atomic_torch_save_new(output_dir / "frozen_model_state.pt", model.state_dict())
    atomic_torch_save_new(
        output_dir / "frozen_evaluator_states.pt",
        {"E_D": evaluator.state_dict()},
    )
    atomic_torch_save_new(output_dir / "frozen_sampler_state.pt", sampler)
    generated_labels = torch.arange(10).repeat_interleave(200)
    generated_ols = torch.zeros((2000, 100), dtype=torch.complex64)
    for index in range(20):
        start, stop = index * 100, (index + 1) * 100
        atomic_torch_save_new(
            output_dir / f"generated_chunk_{index:04d}.pt",
            {
                "schema": "generated_chunk.v1",
                "row_start": start,
                "row_stop": stop,
                "row_indices": torch.arange(start, stop),
                "labels": generated_labels[start:stop],
                "ols": generated_ols[start:stop],
            },
        )
    reference_labels = torch.arange(10).repeat_interleave(30)
    reference_ids = torch.arange(300) + 10_000
    reference_images = torch.zeros((300, 1, 28, 28))
    for index in range(3):
        start, stop = index * 100, (index + 1) * 100
        atomic_torch_save_new(
            output_dir / f"reference_chunk_{index:04d}.pt",
            {
                "schema": "reference_chunk.v1",
                "row_start": start,
                "row_stop": stop,
                "row_indices": torch.arange(start, stop),
                "labels": reference_labels[start:stop],
                "source_row_indices": reference_ids[start:stop],
                "target_image_28x28": reference_images[start:stop],
            },
        )
    (output_dir / "report.json").write_text(
        '{"status":"VALID_COMPLETED","classification":"FROZEN_REPLAY_CONFIRMED"}',
        encoding="utf-8",
    )
    manifest = {
        "status": "VALID_COMPLETED",
        "classification": "FROZEN_REPLAY_CONFIRMED",
        "completed_generated_chunks": list(range(20)),
        "completed_reference_chunks": list(range(3)),
    }

    report_path = replay._verify_terminal_replay_resume(
        output_dir,
        manifest,
        generated_ols,
        generated_labels,
        reference_ids,
        reference_labels,
        reference_images,
        model,
        evaluators,
        sampler,
    )

    assert report_path == output_dir / "report.json"


def test_harness_incidents_are_append_only_and_deduplicated(tmp_path) -> None:
    first = replay._record_harness_incident(tmp_path, "missing NVRTC runtime")
    repeated = replay._record_harness_incident(tmp_path, "missing NVRTC runtime")
    second = replay._record_harness_incident(tmp_path, "invalid batch telemetry")

    assert first == repeated == "harness_incident_0001.json"
    assert second == "harness_incident_0002.json"
    assert "missing NVRTC runtime" in (tmp_path / first).read_text(encoding="utf-8")
    assert "invalid batch telemetry" in (tmp_path / second).read_text(encoding="utf-8")
