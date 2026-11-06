from __future__ import annotations

import torch
import pytest

from scripts.research.lightgen import diagnose_lightgen_fullpath_zero_update_attribution_v1 as attribution


def _summary(labels: torch.Tensor, predictions: torch.Tensor, diversity: float = 1.0) -> dict:
    per_class = []
    for label in range(10):
        rows = labels == label
        per_class.append(float((predictions[rows] == labels[rows]).double().mean()))
    score = {
        "macro_recall": sum(per_class) / 10,
        "per_class_recall": per_class,
        "predicted_labels": predictions.tolist(),
    }
    return {
        "conditions": {"direct_decoder": {"evaluators": {"ED": score, "EC": score}}},
        "same_class_diversity": {
            "direct_decoder": {
                "by_class": {str(label): {"mean_pairwise_normalized_l2": diversity}
                             for label in range(10)}
            }
        },
    }


def _factorial(labels: torch.Tensor, predictions: dict[str, torch.Tensor]) -> dict[str, dict]:
    return {name: _summary(labels, values) for name, values in predictions.items()}


def test_restore_encoder_only_routes_to_encoder_priority() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    poor = labels.clone()
    poor[::10] = (poor[::10] + 1) % 10
    summaries = _factorial(labels, {
        "E0G0": labels, "E1G1": poor, "E0G1": labels, "E1G0": poor,
    })

    result = attribution._attribution_decision(labels, summaries, resamples=100)

    assert result["route"] == "ENCODER_PRIORITY"
    assert result["restoration_support"] == {
        "restore_old_encoder": True,
        "restore_old_generator": False,
    }


def test_restore_generator_only_routes_to_generator_priority() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    poor = labels.clone()
    poor[::10] = (poor[::10] + 1) % 10
    summaries = _factorial(labels, {
        "E0G0": labels, "E1G1": poor, "E0G1": poor, "E1G0": labels,
    })

    result = attribution._attribution_decision(labels, summaries, resamples=100)

    assert result["route"] == "GENERATOR_PRIORITY"
    assert result["restoration_support"] == {
        "restore_old_encoder": False,
        "restore_old_generator": True,
    }


def test_both_restore_paths_are_reported_as_mixed() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    poor = labels.clone()
    poor[::10] = (poor[::10] + 1) % 10
    summaries = _factorial(labels, {
        "E0G0": labels, "E1G1": poor, "E0G1": labels, "E1G0": labels,
    })

    result = attribution._attribution_decision(labels, summaries, resamples=100)

    assert result["route"] == "MIXED_EFFECT"
    assert all(result["restoration_support"].values())


def test_no_supported_restoration_stays_unresolved() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    predictions = {name: labels for name in attribution.CONDITION_ORDER}

    result = attribution._attribution_decision(
        labels, _factorial(labels, predictions), resamples=100
    )

    assert result["route"] == "UNRESOLVED"
    assert not any(result["restoration_support"].values())


def test_saved_population_consumer_rejects_missing_or_reordered_rows() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    proposal_ids = torch.arange(2_000)
    command_shape = (2_000, 1, 200, 200)
    commands = torch.arange(2_000, dtype=torch.uint8).reshape(2_000, 1, 1, 1).expand(command_shape)

    attribution._validate_saved_population_identity(
        labels, proposal_ids, commands, labels, proposal_ids, commands
    )

    with pytest.raises(RuntimeError, match="incomplete or misaligned"):
        attribution._validate_saved_population_identity(
            labels[:-1], proposal_ids[:-1], commands[:-1],
            labels[:-1], proposal_ids[:-1], commands[:-1]
        )

    reordered = labels.roll(1)
    with pytest.raises(RuntimeError, match="incomplete or misaligned"):
        attribution._validate_saved_population_identity(
            labels, proposal_ids, commands, reordered, proposal_ids, commands
        )
    with pytest.raises(RuntimeError, match="incomplete or misaligned"):
        attribution._validate_saved_population_identity(
            labels, proposal_ids, commands, labels, proposal_ids, commands.roll(1, dims=0)
        )


def test_replay_controls_take_first_two_saved_rows_from_every_class() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    rows = attribution._first_two_rows_per_class(labels)

    assert rows.tolist() == [label * 200 + offset
                             for label in range(10) for offset in range(2)]


def test_development_reconstruction_rows_are_balanced_and_from_dev_only() -> None:
    labels = torch.arange(10).repeat_interleave(20)
    assets = {"runtime": {"labels": labels, "dev_indices": torch.arange(200)}}

    rows = attribution._fixed_development_rows(assets)

    assert rows.tolist() == list(range(200))


def test_cross_conditions_change_only_the_registered_phase_group() -> None:
    def checkpoint(encoder: float, generator: float) -> dict:
        return {
            "model": {
                "runtime.encoder.phase": torch.full((2, 1, 100, 100), encoder),
                "runtime.generator.phase": torch.full((2, 1, 100, 100), generator),
                "runtime.camera.fixed": torch.tensor([7.0]),
            },
            "optimizer": {"step": 50_000},
        }

    assets = {
        "checkpoints": {0: checkpoint(0.0, 0.0), 2_000: checkpoint(0.5, 1.0)},
    }

    e0g1 = attribution._condition_checkpoint(assets, "E0G1")
    e1g0 = attribution._condition_checkpoint(assets, "E1G0")

    assert torch.equal(e0g1["model"]["runtime.encoder.phase"], torch.zeros(2, 1, 100, 100))
    assert torch.equal(e0g1["model"]["runtime.generator.phase"], torch.ones(2, 1, 100, 100))
    assert torch.equal(e1g0["model"]["runtime.encoder.phase"], torch.full((2, 1, 100, 100), 0.5))
    assert torch.equal(e1g0["model"]["runtime.generator.phase"], torch.zeros(2, 1, 100, 100))
    assert torch.equal(e0g1["model"]["runtime.camera.fixed"], torch.tensor([7.0]))


def test_saved_condition_consumer_rejects_wrong_phase_source() -> None:
    def checkpoint(encoder: float, generator: float) -> dict:
        return {"model": {
            "runtime.encoder.phase": torch.full((2, 1, 100, 100), encoder),
            "runtime.generator.phase": torch.full((2, 1, 100, 100), generator),
            "runtime.camera.fixed": torch.tensor([7.0]),
        }}

    assets = {"run_id": "a" * 32,
              "checkpoints": {0: checkpoint(0.0, 0.0),
                              2_000: checkpoint(0.5, 1.0)}}
    bundle = {
        "phase_sources": {
            name: {"encoder_step": steps["encoder"],
                   "generator_step": steps["generator"],
                   "source_training_run_id": assets["run_id"]}
            for name, steps in attribution.CONDITION_PHASE_STEPS.items()
        },
        "phase_states": {
            name: {
                "encoder": attribution._condition_checkpoint(assets, name)["model"]
                    ["runtime.encoder.phase"].clone(),
                "generator": attribution._condition_checkpoint(assets, name)["model"]
                    ["runtime.generator.phase"].clone(),
            } for name in attribution.CONDITION_ORDER
        },
    }

    attribution._validate_condition_phase_provenance(bundle, assets)
    bundle["phase_states"]["E0G1"]["encoder"].fill_(0.5)

    with pytest.raises(RuntimeError, match="wrong phase source"):
        attribution._validate_condition_phase_provenance(bundle, assets)


def _positive_control_assets() -> dict:
    return {
        "previous_config": {},
        "runtime": {"device": torch.device("cpu")},
        "checkpoints": {
            0: {"model": {
                "runtime.encoder.phase": torch.zeros(1),
                "runtime.generator.phase": torch.zeros(1),
            }},
            2_000: {"model": {
                "runtime.encoder.phase": torch.ones(1),
                "runtime.generator.phase": torch.ones(1),
            }},
        },
    }


def _install_positive_control_forward(monkeypatch, *, cross_grid_delta: float) -> None:
    target = torch.ones((1, 100), dtype=torch.complex128)
    controls = [{
        "target": target,
        "command": torch.zeros((1, 1, 2, 2), dtype=torch.float64),
        "label": index // 2,
        "proposal_id": index,
    } for index in range(20)]
    monkeypatch.setattr(
        attribution.recent, "_load_solver_positive_controls", lambda _config: controls
    )
    monkeypatch.setattr(
        attribution.recent, "_new_endpoint_model",
        lambda _checkpoint, grid, _runtime: {"grid": grid},
    )

    def forward_amplitude(model, _command):
        grid = model["grid"]
        coefficients = target.clone()
        coefficients[0, 0] = 2.0 + (cross_grid_delta if grid == 4096 else 0.0)
        raw_value = 1.1 + (cross_grid_delta if grid == 4096 else 0.0)
        return {
            "coefficients": coefficients,
            "raw_cmos": torch.full((1, 1, 2, 2), raw_value, dtype=torch.float64),
        }

    monkeypatch.setattr(attribution.projected, "_forward_amplitude", forward_amplitude)
    monkeypatch.setattr(
        attribution.projected, "_forward_target",
        lambda _model, _target: {"raw_cmos": torch.ones((1, 1, 2, 2), dtype=torch.float64)},
    )


def test_cross_grid_positive_control_does_not_gate_target_reachability_error(monkeypatch) -> None:
    _install_positive_control_forward(monkeypatch, cross_grid_delta=1e-5)

    result = attribution._positive_controls(_positive_control_assets())

    assert len(result["rows"]) == 40
    metrics = result["rows"][0]["metrics"]
    assert metrics["coefficient_2048"] > attribution.CROSS_GRID_TOLERANCE
    assert metrics["native_cmos_2048"] > attribution.CROSS_GRID_TOLERANCE
    assert metrics["cross_grid_coefficient"] <= attribution.CROSS_GRID_TOLERANCE
    assert metrics["cross_grid_native_cmos"] <= attribution.CROSS_GRID_TOLERANCE


def test_cross_grid_positive_control_rejects_excess_cross_grid_error(monkeypatch) -> None:
    _install_positive_control_forward(monkeypatch, cross_grid_delta=1e-2)

    with pytest.raises(RuntimeError, match="E0G1 positive control exceeded"):
        attribution._positive_controls(_positive_control_assets())


def test_finite_difference_uses_the_production_bogt_ssim_objective(monkeypatch) -> None:
    class PhaseModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.tensor([0.3], dtype=torch.float64))
            self.decoder_phase = torch.nn.Parameter(torch.tensor([-0.4], dtype=torch.float64))

    model = PhaseModel()

    def loss_parts(current_model, _pixels, _rows, _device, _scale, _power, _gain):
        encoder = current_model.encoder_phase
        generator = current_model.decoder_phase
        zero = (encoder + generator) * 0.0
        return ({"dist": encoder.square() + generator.square(),
                 "rec": zero, "op": zero}, zero, zero, {})

    monkeypatch.setattr(attribution.decoder, "_loss_parts", loss_parts)
    original = (model.encoder_phase.detach().clone(), model.decoder_phase.detach().clone())

    result = attribution._finite_difference_checks(
        model,
        torch.zeros((10, 28, 28), dtype=torch.uint8),
        torch.arange(10),
        {"device": torch.device("cpu")},
        {"scale": torch.ones(1), "power_reference": torch.ones(1),
         "camera_gain": torch.ones(1), "weights": {"dist": 1.0, "rec": 0.0, "op": 0.0},
         "structural_weight": 0.0},
    )

    assert len(result) == 4
    assert all(row["relative_error"] <= 0.05 for row in result)
    assert torch.equal(model.encoder_phase, original[0])
    assert torch.equal(model.decoder_phase, original[1])


def test_dev_reconstruction_uses_fixed_native_image_and_readout_functions(monkeypatch) -> None:
    labels = torch.arange(10).repeat_interleave(20)
    pixels = torch.zeros((60_000, 28, 28), dtype=torch.uint8)
    assets = {
        "runtime": {
            "labels": labels,
            "dev_indices": torch.arange(200),
            "pixels": pixels,
            "device": torch.device("cpu"),
            "camera_gain": 1.0,
        },
        "checkpoints": {
            0: {"rec_value": 0.0, "weights": {"rec": 2.0}},
            2_000: {"rec_value": 1.0, "weights": {"rec": 2.0}},
        },
    }
    monkeypatch.setattr(
        attribution.recent, "_new_endpoint_model",
        lambda checkpoint, _grid, _runtime: {"rec_value": checkpoint["rec_value"]},
    )
    monkeypatch.setattr(
        attribution, "image_batch",
        lambda _pixels, rows, _arm, _device:
            torch.zeros((len(rows), 1, 4, 4), dtype=torch.float32),
    )
    monkeypatch.setattr(
        attribution.base, "_forward",
        lambda model, target: {"physical_w_per_pixel":
                               torch.full_like(target, model["rec_value"])},
    )
    monkeypatch.setattr(
        attribution, "lightgen_resize_view",
        lambda image: torch.ones((len(image), 1, 28, 28), dtype=torch.float32),
    )
    monkeypatch.setattr(
        attribution, "lightgen_ssim_batch",
        lambda view, _target: torch.ones(len(view), dtype=torch.float32),
    )

    result = attribution._development_reconstruction_diagnostic(assets)

    assert result["rows"] == 200
    assert result["rows_per_class"] == 20
    assert result["comparison"]["mean_rec_delta_terminal_minus_start"] > 0.0


def test_saved_decision_consumer_accepts_only_floating_point_roundoff() -> None:
    actual = {"route": "UNRESOLVED", "effects": {"diversity": [0.9919936159496007]},
              "support": {"encoder": False}}
    rounded = {"route": "UNRESOLVED", "effects": {"diversity": [0.9919936159496008]},
               "support": {"encoder": False}}
    changed = {"route": "UNRESOLVED", "effects": {"diversity": [0.9921]},
               "support": {"encoder": False}}

    assert attribution._saved_decision_values_match(actual, rounded)
    assert not attribution._saved_decision_values_match(actual, changed)
    assert not attribution._saved_decision_values_match(
        {"route": "UNRESOLVED", "support": {"encoder": False}},
        {"route": "UNRESOLVED", "support": {"encoder": 0}},
    )

