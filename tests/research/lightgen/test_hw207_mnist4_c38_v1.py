"""Focused C3.8 regressions for the shared one-phase CMOS classifier path."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from scripts.research.lightgen import hw207_mnist4_c37_v1 as c37
from scripts.research.lightgen import execute_hw207_mnist4_c38_v1 as c38


def _config():
    return c38.load_config()


def _result(intensity: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    pixel_area = c37.CMOS_PITCH_UM**2
    scores = torch.stack([
        intensity[:, r0:r1, c0:c1].sum((1, 2)) * pixel_area
        for r0, r1, c0, c1 in c37.roi_bounds()
    ], dim=1)
    detected = intensity.sum((1, 2)) * pixel_area
    return {
        "camera_intensity": intensity,
        "roi_scores": scores,
        "detected_energy": detected,
        "outside_roi_full_cmos_energy": (detected - scores.sum(1)).clamp_min(0),
        "input_energy": torch.full_like(detected, 10_000.0),
    }


def test_config_freezes_model_b_roi_core_and_five_losses():
    config = _config()
    assert config["identity"] == "HW207_MNIST4_C38_JOINT_PHASE_SCREEN"
    assert config["physics_source"] == "configs/research/lightgen/hw207_mnist4_c37_v1.json"
    assert config["numerics"]["lens_to_cmos_fresnel_chirp_compute_dtype"] == "complex128"
    assert tuple(config["training"]["loss_weights"]) == (
        "classification", "core", "peak", "energy", "spread"
    )
    assert config["core"]["fraction_of_roi_side"] == 0.4
    assert c37.roi_bounds() == ((921, 1024, 921, 1024), (921, 1024, 1024, 1127),
                                (1024, 1127, 921, 1024), (1024, 1127, 1024, 1127))


def test_formal_training_rows_use_the_declared_data_role_count():
    labels = torch.arange(4).repeat_interleave(5)
    rows = torch.arange(len(labels))
    context = {"split": {"train": rows}, "labels": labels}
    selected = c38._formal_training_rows(
        context, {"data": {"train_rows_per_class_formal": 2}}
    )
    assert len(selected) == 8
    assert torch.bincount(labels[selected], minlength=4).tolist() == [2, 2, 2, 2]
    assert c38.load_config()["data"]["train_rows_per_class_formal"] == 4000


def test_completed_training_run_is_not_silently_extended_on_resume(tmp_path):
    run_dir = tmp_path / "attempt" / "p3" / "pilots" / "completed"
    run_dir.mkdir(parents=True)
    phase_path = run_dir / "phase_rad.npy"
    np.save(phase_path, np.zeros((100, 100), dtype=np.float32), allow_pickle=False)
    (run_dir / "checkpoint.pt").write_bytes(b"preserved test checkpoint")
    train_rows = torch.tensor([0, 1, 2, 3])
    validation_rows = torch.tensor([4, 5, 6, 7])
    c38.atomic_savez(run_dir / "sample_roles.npz", train_rows=train_rows.numpy(),
                     validation_rows=validation_rows.numpy())
    summary = {
        "identity": "C38_SHARED_SINGLE_PHASE_TRAINING_RUN", "arm": "T4",
        "seed": 11, "initialization": "C35_P1",
        "active_losses": ["classification"],
        "training_rows": train_rows.tolist(), "validation_rows": validation_rows.tolist(),
        "phase_path": str(phase_path), "requested_max_epochs": 3,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary), encoding="utf-8")

    actual = c38._train_one(
        {}, {"device": torch.device("cpu")}, run_dir, {}, {},
        arm="T4", initialization="C35_P1", seed=11,
        train_rows=train_rows, validation_rows=validation_rows,
        active_losses=("classification",), max_epochs=3, resume=True,
    )
    assert actual == summary
    with pytest.raises(ValueError, match="different epoch budget"):
        c38._train_one(
            {}, {"device": torch.device("cpu")}, run_dir, {}, {},
            arm="T4", initialization="C35_P1", seed=11,
            train_rows=train_rows, validation_rows=validation_rows,
            active_losses=("classification",), max_epochs=5, resume=True,
        )


def test_four_cores_are_equal_area_inside_the_frozen_rois():
    config = _config()
    rois = c37.roi_bounds()
    cores = c37.core_bounds(config["core"]["fraction_of_roi_side"])
    assert c37.core_side_px(0.4) == 41
    assert all((r1 - r0, c1 - c0) == (41, 41) for r0, r1, c0, c1 in cores)
    for roi, core in zip(rois, cores):
        assert roi[0] <= core[0] < core[1] <= roi[1]
        assert roi[2] <= core[2] < core[3] <= roi[3]
    assert config["core"]["core_side_um"] == pytest.approx(266.5)


def test_physical_roi_coordinates_preserve_row_y_column_x_orientation():
    left_low = c38._physical_box(c37.roi_bounds()[0])
    right_low = c38._physical_box(c37.roi_bounds()[1])
    assert left_low[0] == pytest.approx(-669.5)
    assert left_low[1] == pytest.approx(-669.5)
    assert left_low[2:] == pytest.approx((669.5, 669.5))
    assert right_low[0] == pytest.approx(0.0)


def test_phase_transmission_is_periodic_and_unit_amplitude():
    phase = torch.linspace(-math.pi, 4 * math.pi, 10_000).reshape(100, 100)
    actual = c38.phase_transmission(phase)
    shifted = c38.phase_transmission(phase + 2 * math.pi)
    assert actual.shape == (100, 100)
    assert torch.allclose(actual.abs(), torch.ones_like(phase), atol=1e-6, rtol=0)
    assert torch.allclose(actual, shifted, atol=2e-6, rtol=0)


def test_five_joint_losses_are_finite_and_differentiable():
    generator = torch.Generator().manual_seed(3801)
    intensity = (torch.rand((2, c37.CMOS, c37.CMOS), generator=generator) + 0.01)
    intensity.requires_grad_(True)
    labels = torch.tensor([0, 3])
    config = _config()
    result = _result(intensity, labels)
    phase = torch.rand((100, 100), generator=generator, requires_grad=True)
    eta_reference = torch.tensor([0.45, 0.52])

    loss, components = c38.loss_components(
        result, labels, phase, {"device": torch.device("cpu")}, config,
        eta_reference=eta_reference,
    )
    assert set(components) == {"classification", "core", "peak", "energy", "spread"}
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    intensity_grad = torch.autograd.grad(loss, intensity)[0]
    assert torch.isfinite(intensity_grad).all()
    assert float(intensity_grad.norm()) > 0


def test_ablation_mask_changes_the_real_weighted_objective():
    config = _config()
    labels = torch.tensor([0])
    intensity = torch.ones((1, c37.CMOS, c37.CMOS), requires_grad=True)
    phase = torch.zeros((100, 100), requires_grad=True)
    result = _result(intensity, labels)
    total, terms = c38.loss_components(
        result, labels, phase, {"device": torch.device("cpu")}, config,
        eta_reference=torch.tensor([0.45]), active_losses=("classification",),
    )
    assert torch.allclose(total, terms["classification"])
    assert not torch.allclose(total, sum(terms.values()))


def test_roi_integrals_are_nonnegative_and_close_the_cmos_energy_account():
    image = torch.zeros((1, c37.CMOS, c37.CMOS))
    for index, (r0, r1, c0, c1) in enumerate(c37.roi_bounds()):
        image[0, r0 + 2, c0 + 2] = float(index + 1)
    result = _result(image, torch.tensor([3]))
    assert result["roi_scores"].shape == (1, 4)
    assert torch.all(result["roi_scores"] >= 0)
    assert result["roi_scores"].argmax(1).tolist() == [3]
    assert torch.allclose(result["roi_scores"].sum(1) +
                          result["outside_roi_full_cmos_energy"], result["detected_energy"])


def test_peak_term_rewards_a_high_intensity_peak_inside_the_target_core():
    config = _config()
    labels = torch.tensor([0])
    base = torch.full((1, c37.CMOS, c37.CMOS), 1e-4)
    target = c37.core_bounds(config["core"]["fraction_of_roi_side"])[0]
    wrong = c37.core_bounds(config["core"]["fraction_of_roi_side"])[3]
    phase = torch.zeros((100, 100), requires_grad=True)

    losses = []
    for row, col in ((target[0] + 10, target[2] + 10), (wrong[0] + 10, wrong[2] + 10)):
        intensity = base.clone()
        intensity[0, row, col] = 1.0
        result = _result(intensity, labels)
        _, parts = c38.loss_components(
            result, labels, phase, {"device": torch.device("cpu")}, config,
            eta_reference=torch.tensor([0.45]),
        )
        losses.append(float(parts["peak"]))
    assert losses[0] < losses[1]


def test_frozen_prediction_uses_only_four_fixed_roi_integrals():
    scores = torch.tensor([[9.0, 2.0, 1.0, 0.5], [0.0, 4.0, 5.0, 1.0]])
    assert c38.predict_from_roi_scores(scores).tolist() == [0, 2]
    with pytest.raises(ValueError, match="four ROI"):
        c38.predict_from_roi_scores(torch.ones((2, 5)))


def test_frozen_production_path_aggregates_p4_all_and_correct_cmos_means(monkeypatch):
    labels = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    predicted_rois = (0, 2, 1, 2)

    def fake_input_fields(context, rows):
        return torch.as_tensor(rows, dtype=torch.float32)

    def fake_forward_field(field, phase, context, *, lens_pitch_um=None):
        row_ids = field.to(torch.long).tolist()
        images = torch.zeros((len(row_ids), c37.CMOS, c37.CMOS), dtype=torch.float32)
        for local, row in enumerate(row_ids):
            r0, r1, c0, c1 = c37.roi_bounds()[predicted_rois[row]]
            images[local, (r0 + r1) // 2, (c0 + c1) // 2] = float(row + 1)
        return _result(images, labels[torch.tensor(row_ids)])

    monkeypatch.setattr(c37, "input_fields", fake_input_fields)
    monkeypatch.setattr(c38, "forward_field", fake_forward_field)
    context = {"device": torch.device("cpu"), "labels": labels, "lens_pitch_um": 2.0}
    result = c38.evaluate_frozen(
        context, _config(), torch.zeros((100, 100)), torch.arange(4), batch_size=2,
        aggregate_group_means=True,
    )

    grouped = result["group_mean_maps"]
    assert grouped["all_class_counts"].tolist() == [1, 2, 1, 0]
    assert grouped["correct_class_counts"].tolist() == [1, 1, 1, 0]
    assert grouped["predictions"].tolist() == [0, 2, 1, 2]
    assert grouped["all_class_means"].shape == (4, c37.CMOS, c37.CMOS)
    assert grouped["correct_class_means"].shape == (4, c37.CMOS, c37.CMOS)
    assert np.isfinite(grouped["overall_mean"]).all()
    assert grouped["overall_mean"].sum() == pytest.approx(
        result["arrays"]["cmos_energy_um2"].sum()
        / (len(labels) * c37.CMOS_PITCH_UM**2),
        rel=1e-6,
    )
    assert np.array_equal(grouped["correct_class_means"][3],
                          np.zeros((c37.CMOS, c37.CMOS), dtype=np.float32))
    cls_one_roi_two = c37.roi_bounds()[2]
    cls_one_roi_one = c37.roi_bounds()[1]
    assert grouped["all_class_means"][1, (cls_one_roi_two[0] + cls_one_roi_two[1]) // 2,
                                      (cls_one_roi_two[2] + cls_one_roi_two[3]) // 2] == pytest.approx(1.0)
    assert grouped["all_class_means"][1, (cls_one_roi_one[0] + cls_one_roi_one[1]) // 2,
                                      (cls_one_roi_one[2] + cls_one_roi_one[3]) // 2] == pytest.approx(1.5)


def test_cmos_summary_plot_keeps_fixed_roi_core_coordinates_and_linear_scale():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.zeros((c37.CMOS, c37.CMOS), dtype=np.float32)
    center_px = int(round(c37.FRAME_CENTER_PX))
    image[center_px, center_px] = 1.0
    fig, ax = plt.subplots()
    artist = c38._draw_cmos_summary(
        ax, image, 0.125, title="test", target_label=2,
        core_fraction=_config()["core"]["fraction_of_roi_side"],
    )
    try:
        assert artist.get_clim() == (0.0, 0.125)
        assert len(ax.patches) == 8  # four fixed ROIs and four fixed cores
        assert len(ax.collections) == 2  # peak and centroid markers
        r0, r1, c0, c1 = c37.roi_bounds()[0]
        expected_xy = ((c0 - c37.FRAME_CENTER_PX - 0.5) * c37.CMOS_PITCH_UM,
                       (r0 - c37.FRAME_CENTER_PX - 0.5) * c37.CMOS_PITCH_UM)
        assert ax.patches[0].get_xy() == pytest.approx(expected_xy)
        assert ax.patches[0].get_width() == pytest.approx((c1 - c0) * c37.CMOS_PITCH_UM)
        assert ax.patches[0].get_height() == pytest.approx((r1 - r0) * c37.CMOS_PITCH_UM)
    finally:
        plt.close(fig)


def test_pareto_front_drops_dominated_candidates_but_keeps_tradeoffs():
    records = [
        {"name": "a", "accuracy": 0.91, "eta_mean": 0.50, "core_input_mean": 0.04},
        {"name": "b", "accuracy": 0.90, "eta_mean": 0.55, "core_input_mean": 0.05},
        {"name": "dominated", "accuracy": 0.89, "eta_mean": 0.49, "core_input_mean": 0.03},
    ]
    assert {item["name"] for item in c38.pareto_front(records)} == {"a", "b"}


def test_candidate_score_uses_the_declared_peak_core_target():
    config = _config()
    targets = config["evaluation"]
    baseline = {
        "eta_phase_to_cmos": {"mean": 0.5, "p10": 0.25},
        "correct_core_input_fraction": {"mean": 0.04},
    }
    metrics = {
        "accuracy": targets["classification_accuracy_target"],
        "macro_f1": targets["macro_f1_target"],
        "peak_in_correct_roi_rate": targets["peak_in_correct_roi_target"],
        "peak_in_correct_core_rate": targets["peak_in_correct_core_target"],
        "eta_phase_to_cmos": {
            "mean": baseline["eta_phase_to_cmos"]["mean"] *
            targets["efficiency_reference_retention_target"],
            "p10": baseline["eta_phase_to_cmos"]["p10"] *
            targets["efficiency_reference_retention_target"],
        },
        "correct_core_input_fraction": {
            "mean": baseline["correct_core_input_fraction"]["mean"] * 1.25,
        },
    }
    assert c38._candidate_score(metrics, baseline, config) == pytest.approx(1.0)


def test_sample_selection_keeps_correct_examples_and_representative_errors():
    arrays = {
        "rows": np.arange(8),
        "labels": np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        "predictions": np.array([0, 1, 1, 1, 2, 0, 3, 2]),
        "roi_scores": np.array([
            [4, 1, 0, 0], [1, 4, 0, 0], [0, 1, 0, 0], [0, 4, 0, 0],
            [0, 0, 4, 0], [4, 0, 1, 0], [0, 0, 0, 4], [0, 0, 4, 0],
        ], dtype=np.float32),
    }
    selection = c38._sample_selection(arrays)
    assert selection["typical_correct"]["0"]
    assert selection["representative_errors"]["0"] == [1]
    assert selection["most_common_error_destination"]["0"] == 1


def test_shared_forward_matches_frozen_c37_model_b_on_cpu():
    context = {
        "device": torch.device("cpu"),
        "lens_pitch_um": 4.0,
        "screen_to_lens_um": 80_000.0,
        "lens_to_cmos_um": 100_000.0,
        "aperture_mm": 2.0,
        "propagation_cache": {},
    }
    fields = torch.ones((1, 100, 100), dtype=torch.complex64)
    phase = torch.zeros((100, 100))
    actual = c38.forward_field(fields, phase, context)
    expected = c37.forward_batch(
        fields, phase, context, lens_to_cmos_chirp_dtype=torch.complex128
    )
    assert torch.equal(actual["roi_scores"], expected["roi_scores"])
    assert torch.equal(actual["camera_intensity"], expected["camera_intensity"])
    assert torch.all(actual["camera_intensity"] >= 0)


def test_single_frequency_transfer_check_runs_and_passes_on_cpu():
    result = c38._single_frequency_check(torch.device("cpu"))
    assert result["passed"]
    assert result["relative_l2_vs_independent_finite_aperture_reference"] <= result["tolerance"]
    assert result["reverse_propagation_rejected"]
    assert result["reverse_propagation_relative_l2"] > result["reverse_propagation_rejection_threshold"]


def test_baseline_source_arrays_are_compatible_and_historical_records_align():
    config = _config()
    sources = c38.resolve_phase_sources(config)
    assert set(sources) == {"C35_P1", "C37_E2_seed11", "C37_E3_seed11",
                            "C37_E3_seed23", "C37_E7_seed23"}
    assert all(tuple(phase.shape) == (100, 100) and torch.isfinite(phase).all()
               for phase in sources.values())
    run = c38.ROOT / config["run"]["version_directory"] / "runs" / config["run"]["attempt"]
    references = c38._resolve_history_sources(config, run)
    with np.load(references["C35_P1"]) as baseline:
        reference_rows, reference_labels = baseline["rows"], baseline["labels"]
        reference_input = baseline["input_energy_um2"]
    for key, path in references.items():
        with np.load(path) as saved:
            assert np.array_equal(reference_rows, saved["rows"]), key
            assert np.array_equal(reference_labels, saved["labels"]), key
            assert np.array_equal(reference_input, saved["input_energy_um2"]), key


def _minimal_completion_audit_run(run_dir, *, formal_loss_sets):
    config = _config()
    for stage, payload in (
        ("p0/c38_p0_history_audit.json", {"all_historical_prediction_replays_match": True}),
        ("p1/c38_p1_numerical_validation.json", {"numerical_core_pass": True}),
        ("p2/c38_preflight.json", {"passed": True}),
    ):
        c38.write_json(run_dir / stage, payload)
    formal_runs = []
    for index, losses in enumerate(formal_loss_sets):
        summary_path = run_dir / "p3" / f"formal_{index}" / "training_summary.json"
        c38.write_json(summary_path, {
            "active_losses": list(losses), "epochs_completed": 3, "training_updates": 12,
        })
        formal_runs.append({
            "name": f"candidate_seed{index}",
            "summary_path": summary_path.relative_to(c38.ROOT).as_posix(),
        })
    c38.write_json(run_dir / "c38_training_summary.json", {
        "formal_run_count": 6, "formal_runs": formal_runs,
    })
    group_path = run_dir / "p4" / "group_means.npz"
    group_path.parent.mkdir(parents=True, exist_ok=True)
    group_path.write_bytes(b"test group means")
    c38.write_json(run_dir / "c38_evaluation.json", {
        "selected_for_visualization": "candidate_seed0",
        "group_mean_maps_candidate": "candidate_seed0",
        "group_mean_maps_path": "p4/group_means.npz",
        "all_research_targets_pass": True,
    })
    c38.write_json(run_dir / "p5" / "c38_visualization_manifest.json", {
        "programmatic_qa": {
            "all_formats_present": True,
            "single_shared_cmos_scale": True,
            "all_sample_maps_positive_and_finite": True,
            "all_group_mean_maps_positive_and_finite": True,
            "p4_group_means_cover_all_validation_rows": True,
            "p4_selected_rows_replay_saved_roi_statistics": True,
        },
    })
    return config


def test_completion_audit_requires_spread_to_reach_formal_training(tmp_path, monkeypatch):
    monkeypatch.setattr(c38, "ROOT", tmp_path)
    required = tuple(_config()["training"]["loss_order"])
    full_arm = tmp_path / "outputs" / "full_joint_arm" / "runs" / "attempt_01"
    full_config = _minimal_completion_audit_run(
        full_arm, formal_loss_sets=[required] * 3 + [required[:-1]] * 3,
    )
    full_result = c38.run_completion_audit(full_config, full_arm)
    assert full_result["status"] == "C3.8_PASS"
    assert full_result["stage_checks"]["P3_formal_runs"]
    assert full_result["stage_checks"]["P3_full_joint_loss_in_formal_training"]

    no_spread_arm = tmp_path / "outputs" / "no_spread_arm" / "runs" / "attempt_01"
    no_spread_config = _minimal_completion_audit_run(
        no_spread_arm, formal_loss_sets=[required[:-1]] * 6,
    )
    no_spread_result = c38.run_completion_audit(no_spread_config, no_spread_arm)
    assert no_spread_result["stage_checks"]["P3_formal_runs"]
    assert not no_spread_result["stage_checks"]["P3_full_joint_loss_in_formal_training"]
    assert no_spread_result["status"] == "C3.8_PARTIAL_PASS"
