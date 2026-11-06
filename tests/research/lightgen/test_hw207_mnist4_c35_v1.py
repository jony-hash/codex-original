from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.research.lightgen import execute_hw207_mnist4_c35_v1 as c35
from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34


def test_eta_phase_to_cmos_uses_full_sensor_over_phase_input_energy() -> None:
    detected = np.asarray([10.0, 12.0])
    phase_input = np.asarray([20.0, 16.0])
    np.testing.assert_allclose(c35._eta_phase_to_cmos(detected, phase_input), [0.5, 0.75])


def test_eta_phase_to_cmos_rejects_invalid_input_energy() -> None:
    with pytest.raises(ValueError, match="positive input energy"):
        c35._eta_phase_to_cmos(np.asarray([1.0]), np.asarray([0.0]))


def test_fixed_roi_metrics_match_the_argmax_decision() -> None:
    metrics = c35._classification_metrics(np.asarray([0, 1, 2, 3]),
                                         np.asarray([0, 2, 2, 3]))
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    ]
    assert metrics["recall"] == [1.0, 0.0, 1.0, 1.0]


def test_shuffled_label_negative_control_preserves_classes_and_rejects_alignment() -> None:
    labels = np.repeat(np.arange(4, dtype=np.int64), 64)
    shuffled, metrics = c35._shuffled_label_negative_control(labels, labels.copy(), 991)
    np.testing.assert_array_equal(np.bincount(shuffled, minlength=4), [64, 64, 64, 64])
    assert metrics["sample_count"] == 256
    assert metrics["accuracy"] < 0.40


def test_energy_feasibility_precedes_accuracy_in_model_selection() -> None:
    feasible = {"accuracy": 0.91, "macro_f1": 0.90, "recall": [0.8, 0.9, 0.9, 0.9],
                "eta_phase_to_cmos": {"mean": 0.08, "p10": 0.04}}
    infeasible_but_more_accurate = {
        "accuracy": 0.99, "macro_f1": 0.99, "recall": [0.99] * 4,
        "eta_phase_to_cmos": {"mean": 0.07, "p10": 0.001},
    }
    assert c35._selection_key(feasible, 0.077) < c35._selection_key(infeasible_but_more_accurate, 0.077)


def test_lagrange_multiplier_tracks_both_sides_of_energy_constraint() -> None:
    assert c35._update_lagrange_multiplier(0.0, 0.1, 0.05, 10.0) == 0.5
    assert c35._update_lagrange_multiplier(1.0, 0.1, 0.2, 10.0) == 0.0


def test_restored_optimizer_states_do_not_alias_checkpoint_tensors() -> None:
    source = torch.nn.Parameter(torch.tensor([1.0]))
    source_optimizer = torch.optim.Adam([source], lr=0.1)
    source.grad = torch.tensor([0.25])
    source_optimizer.step()
    checkpoint_state = source_optimizer.state_dict()

    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    first_optimizer = torch.optim.Adam([first], lr=0.1)
    second_optimizer = torch.optim.Adam([second], lr=0.1)
    c35._restore_optimizer_state(first_optimizer, checkpoint_state)
    c35._restore_optimizer_state(second_optimizer, checkpoint_state)

    first.grad = torch.tensor([0.25])
    first_optimizer.step()
    second.grad = torch.tensor([0.25])
    second_optimizer.step()

    torch.testing.assert_close(first, second)
    assert first_optimizer.state[first]["step"].item() == 2
    assert second_optimizer.state[second]["step"].item() == 2
    assert checkpoint_state["state"][0]["step"].item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA state-tensor aliasing is AutoDL-only")
def test_cuda_optimizer_checkpoint_restore_replays_independent_updates() -> None:
    device = torch.device("cuda")
    source = torch.nn.Parameter(torch.tensor([1.0], device=device))
    source_optimizer = torch.optim.Adam([source], lr=0.1)
    source.grad = torch.tensor([0.25], device=device)
    source_optimizer.step()
    checkpoint_state = source_optimizer.state_dict()

    first = torch.nn.Parameter(torch.tensor([1.0], device=device))
    second = torch.nn.Parameter(torch.tensor([1.0], device=device))
    first_optimizer = torch.optim.Adam([first], lr=0.1)
    second_optimizer = torch.optim.Adam([second], lr=0.1)
    c35._restore_optimizer_state(first_optimizer, checkpoint_state)
    c35._restore_optimizer_state(second_optimizer, checkpoint_state)
    first.grad = torch.tensor([0.25], device=device)
    second.grad = torch.tensor([0.25], device=device)
    first_optimizer.step()
    second_optimizer.step()

    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_objective_measures_roi_leakage_against_full_detected_energy() -> None:
    result = {
        "roi_scores": torch.tensor([[2.0, 1.0, 0.5, 0.5]]),
        "comparison_crop_energy": torch.tensor([4.0]),
        "outside_roi_full_cmos_energy": torch.tensor([2.0]),
        "detected_energy": torch.tensor([10.0]),
        "input_energy": torch.tensor([10.0]),
    }
    config = {"training": {"classification_margin_weight": 0.0,
                           "roi_leakage_weight": 1.0, "phase_smoothness_weight": 0.0}}
    _, components = c35._objective(result, torch.tensor([0]), torch.zeros((2, 2)),
                                   target_efficiency=0.0, multiplier=0.0, rho=1.0,
                                   config=config)
    assert components["leakage"].item() == pytest.approx(0.2)


def test_config_rejects_geometry_that_the_c34_forward_would_not_use(tmp_path) -> None:
    config = json.loads(c35.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["geometry"]["lens_to_cmos_mm"] = 99.0
    path = tmp_path / "mismatched_geometry.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the C3.4 forward model"):
        c35._load_config(path)


def test_resume_preserves_best_epoch_selected_before_interruption(tmp_path, monkeypatch) -> None:
    config = {
        "identity": c35.IDENTITY,
        "training": {"epochs": 2, "logical_batch_size": 2, "microbatch_size": 2,
                     "learning_rate": 0.1},
        "energy": {"adaptive_lagrange_rho": 1.0},
        "fabrication_assumptions": {"edge_height_bias_nm": [-20.0, 20.0],
                                    "height_noise_seeds": [11]},
    }
    context = {
        "device": torch.device("cpu"),
        "split": {"train": torch.tensor([0, 1, 2, 3]), "validation": torch.tensor([4])},
        "labels": torch.zeros(5, dtype=torch.long),
    }
    update_state = {"count": 0, "interrupt": True}

    def fake_update(variable, optimizer, update_context, rows, **kwargs):
        update_state["count"] += 1
        if update_state["interrupt"] and update_state["count"] == 3:
            raise RuntimeError("simulated process interruption")
        with torch.no_grad():
            variable.add_(1.0)
        return {"loss": 1.0, "classification": 1.0, "margin": 0.0, "leakage": 0.0,
                "constraint": 0.0, "eta_mean": 0.1, "eta_shortfall_mean": 0.0,
                "gradient_norm": 1.0, "parameter_delta_l2": 1.0, "microbatch_used": 2.0}

    def fake_effective_phase(variable, mode, current_config, **kwargs):
        return variable

    def fake_evaluate(eval_context, phase, rows, tag, output_path, **kwargs):
        accuracy = 0.9 if float(phase.mean()) <= 2.0 else 0.5
        return {"accuracy": accuracy, "macro_f1": accuracy, "recall": [accuracy] * 4,
                "eta_phase_to_cmos": {"mean": 0.1, "p10": 0.08}}

    monkeypatch.setattr(c35, "_update_batch", fake_update)
    monkeypatch.setattr(c35, "_effective_phase", fake_effective_phase)
    monkeypatch.setattr(c35, "_evaluate_phase", fake_evaluate)

    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        c35._train_one(context, config, mode="phase", seed=11, target=0.1,
                       initial=torch.zeros((2, 2)), output_dir=interrupted_dir)
    state_path = interrupted_dir / "epoch_state.pt"
    saved_state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert saved_state["best_epoch"] == 1

    update_state.update({"count": 3, "interrupt": False})
    resumed = c35._train_one(context, config, mode="phase", seed=11, target=0.1,
                              initial=torch.zeros((2, 2)), output_dir=interrupted_dir,
                              resume_checkpoint=state_path)

    update_state.update({"count": 0, "interrupt": False})
    continuous = c35._train_one(context, config, mode="phase", seed=11, target=0.1,
                                initial=torch.zeros((2, 2)), output_dir=tmp_path / "continuous")
    assert resumed["best_epoch"] == continuous["best_epoch"] == 1
    torch.testing.assert_close(resumed["best_variable"], continuous["best_variable"])
    assert resumed["best_metrics"] == continuous["best_metrics"]


def test_phase_height_roundtrip_respects_2pi_equivalence() -> None:
    phase = torch.tensor([[0.0, 2 * torch.pi], [-torch.pi, 3 * torch.pi]])
    height = c34.height_from_phase(phase, 0.5)
    recovered = c34.phase_from_height(height, 0.5)
    torch.testing.assert_close(torch.remainder(recovered, 2 * torch.pi),
                               torch.remainder(phase, 2 * torch.pi), atol=1e-6, rtol=0)


def test_c0_population_has_no_unseen_mnist_0_to_3_rows() -> None:
    config = c35._load_config(c35.DEFAULT_CONFIG)
    result = c35._indices_and_counts(config)
    assert result["counts"] == {"train": 22279, "validation": 2475, "test": 4157}
    assert result["mnist_train_0_to_3"] == 24754
    assert result["mnist_official_test_0_to_3"] == 4157
    assert result["unused_0_to_3_samples"] == 0
    assert result["independent_eval_status"] == "INDEPENDENT_EVAL_BLOCKED"


def test_data_usage_ledger_covers_the_audited_c0_rows(tmp_path) -> None:
    config = c35._load_config(c35.DEFAULT_CONFIG)
    result = c35._indices_and_counts(config, ledger_dir=tmp_path)
    ledger_path = tmp_path / "c35_data_usage_ledger.csv"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 22_279 + 2_475 + 4_157
    assert result["row_level_ledger_rows"] == 28_911
    assert result["independent_eval_status"] == "INDEPENDENT_EVAL_BLOCKED"
    assert "independent_for_c35" in lines[0]
    assert "UNKNOWN_NOT_SAVED_IN_C33_SUMMARIES" in "\n".join(lines)


def test_process_sensitivity_varies_one_factor_around_the_nominal_operator(
    tmp_path, monkeypatch
) -> None:
    process_calls = []
    evaluated_phases = {}

    def fake_processed_height(height, config, *, dn, psf_sigma_px,
                              minimum_feature_radius_px, height_levels,
                              noise_seed, noise_nm, edge_bias_nm=0.0):
        process_calls.append({
            "dn": dn,
            "psf": psf_sigma_px,
            "minimum_feature": minimum_feature_radius_px,
            "levels": height_levels,
            "noise_seed": noise_seed,
            "noise": noise_nm,
            "edge": edge_bias_nm,
        })
        offset = (dn + psf_sigma_px + minimum_feature_radius_px
                  + height_levels / 1000.0 + noise_nm / 1000.0
                  + edge_bias_nm / 1000.0)
        return height + offset

    def fake_evaluate(context, phase, rows, tag, output_path, **kwargs):
        phase = phase.detach().cpu().clone()
        evaluated_phases[tag] = phase
        row_ids = rows.detach().cpu().numpy()
        labels = context["labels"][rows].cpu().numpy()
        predictions = labels.copy()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, original_rows=row_ids, labels=labels,
                 predictions=predictions)
        accuracy = 0.9 if float(phase.mean()) > 3.0 else 0.1
        return {
            "accuracy": accuracy,
            "macro_f1": accuracy,
            "recall": [accuracy] * 4,
            "eta_phase_to_cmos": {"mean": 0.1, "p10": 0.08},
        }

    monkeypatch.setattr(c35, "_load_control_phases", lambda config: {})
    monkeypatch.setattr(c35, "_processed_height", fake_processed_height)
    monkeypatch.setattr(c35, "_evaluate_phase", fake_evaluate)

    p1_path = tmp_path / "p1_energy_constrained" / "phase_energy_constrained.npy"
    p2_root = tmp_path / "p2_process_aware"
    p1_path.parent.mkdir(parents=True)
    p2_root.mkdir(parents=True)
    np.save(p1_path, np.zeros((100, 100), dtype=np.float32))
    p2_height = np.ones((100, 100), dtype=np.float32)
    np.save(p2_root / "height_map_um.npy", p2_height)
    np.save(p2_root / "phase_after_nominal_process.npy",
            np.full((100, 100), -1.0, dtype=np.float32))

    config = {
        "fabrication_assumptions": {
            "delta_n_scenarios": [0.5, 0.6],
            "psf_sigma_px": 0.8,
            "minimum_feature_radius_px": 1,
            "height_levels": 64,
            "height_noise_nm": 20.0,
            "height_noise_seeds": [11],
            "edge_height_bias_nm": [-20.0, 20.0],
            "maximum_height_um": 1.2,
        },
        "sensitivity": {
            "validation_rows_per_class": 1,
            "delta_n": [0.5, 0.6],
            "psf_sigma_px": [0.0, 0.8],
            "minimum_feature_radius_px": [0, 1],
            "height_noise_seeds": [11],
            "apertures_mm": [0.5],
        },
    }
    context = {
        "work_dir": tmp_path,
        "split": {"validation": torch.arange(8)},
        "labels": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
    }

    summary = c35.run_sensitivity(context, config)

    assert process_calls
    expected_p2_baseline = p2_height + (0.5 + 0.8 + 1 + 64 / 1000.0)
    torch.testing.assert_close(
        evaluated_phases["c35_p2_process_aware_baseline_nominal_process"],
        torch.from_numpy(expected_p2_baseline),
    )

    delta_n_spec = next(
        spec for spec in summary["scenario_specs"]
        if spec["name"] == "delta_n_0.6_nominal_process"
    )
    assert delta_n_spec == {
        "family": "delta_n", "name": "delta_n_0.6_nominal_process",
        "dn": 0.6, "psf": 0.8, "min_feature": 1, "levels": 64,
        "noise": 0.0, "noise_seed": 0, "edge": 0.0, "aperture": 2.0,
    }
    expected_p2_delta_n = p2_height + (0.6 + 0.8 + 1 + 64 / 1000.0)
    torch.testing.assert_close(
        evaluated_phases["c35_p2_process_aware_delta_n_0.6_nominal_process"],
        torch.from_numpy(expected_p2_delta_n),
    )
    delta_n_record = next(
        item for item in summary["candidates"]["c35_p2_process_aware"]
        if item["scenario"] == "delta_n_0.6_nominal_process"
    )
    assert delta_n_record["metrics"]["accuracy"] == 0.9
