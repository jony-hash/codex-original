from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.research.lightgen import execute_lightgen_recent_optical_fullpath_semantic_v1 as route


def _semantic_summary(ed: torch.Tensor, ec: torch.Tensor,
                      labels: torch.Tensor, diversity: float) -> dict:
    def evaluator(predicted: torch.Tensor) -> dict:
        recalls = []
        for label in range(10):
            mask = labels == label
            recalls.append(float((predicted[mask] == labels[mask]).float().mean()))
        confusion = torch.zeros((10, 10), dtype=torch.long)
        for actual, predicted_label in zip(labels, predicted, strict=True):
            confusion[int(actual), int(predicted_label)] += 1
        return {
            "macro_recall": sum(recalls) / 10,
            "per_class_recall": recalls,
            "predicted_labels": predicted.tolist(),
            "confusion_matrix": confusion.tolist(),
        }

    return {
        "conditions": {"direct_decoder": {"evaluators": {
            "ED": evaluator(ed), "EC": evaluator(ec),
        }}},
        "same_class_diversity": {"direct_decoder": {"by_class": {
            str(label): {"mean_pairwise_normalized_l2": diversity}
            for label in range(10)
        }}},
    }


def _endpoint_summary(ed_recall: float, ec_recall: float) -> dict:
    return {"conditions": {"direct_decoder": {"evaluators": {
        "ED": {"macro_recall": ed_recall},
        "EC": {"macro_recall": ec_recall},
    }}}}


def test_endpoint_choice_and_tie_order_are_fixed() -> None:
    summaries = {
        "C45k": _endpoint_summary(0.80, 0.82),
        "J": _endpoint_summary(0.84, 0.84),
        "D0": _endpoint_summary(0.85, 0.83),
        "Dgrad": _endpoint_summary(0.84, 0.84),
    }
    assert route._choose_endpoint(summaries) == "J"


def test_dataset_roles_accept_real_mnist_class_imbalance_and_reject_overlap() -> None:
    counts = torch.tensor([5_923, 6_742, 5_958, 6_131, 5_842,
                           5_421, 5_918, 6_265, 5_851, 5_949])
    labels = torch.repeat_interleave(torch.arange(10), counts)
    dev_indices = torch.cat([
        torch.nonzero(labels == label, as_tuple=False).flatten()[:500]
        for label in range(10)
    ])
    dev_set = torch.zeros(labels.numel(), dtype=torch.bool)
    dev_set[dev_indices] = True
    train_indices = torch.nonzero(~dev_set, as_tuple=False).flatten()

    route._validate_dataset_roles(labels, train_indices, dev_indices)
    with pytest.raises(RuntimeError, match="roles changed or overlap"):
        route._validate_dataset_roles(
            labels, torch.cat((train_indices[:-1], dev_indices[:1])), dev_indices
        )


def test_endpoint_identity_binds_run_lineage_without_inventing_j_branch_metadata() -> None:
    spec = {
        "task_id": "joint_task",
        "checkpoint_branch": None,
        "checkpoint_run_id": "endpoint_run",
        "source_run_id": "source_run",
        "global_step": 50_000,
    }
    checkpoint = {
        "task_id": "joint_task",
        "run_id": "endpoint_run",
        "source_run_id": "source_run",
        "completed_global_step": 50_000,
        "model_identity": {
            "encoder": [2, 100, 100],
            "decoder": [2, 100, 100],
            "ols": "gaussian_s12_v1",
        },
        "model": {
            "runtime.encoder.phase": torch.zeros((2, 1, 100, 100)),
            "runtime.generator.phase": torch.zeros((2, 1, 100, 100)),
        },
    }
    route._validate_endpoint_identity("J", spec, checkpoint)
    checkpoint["run_id"] = "different_run"
    with pytest.raises(RuntimeError, match="run lineage"):
        route._validate_endpoint_identity("J", spec, checkpoint)


def test_saved_mapper_replay_uses_registered_project_tolerance(tmp_path, monkeypatch) -> None:
    expected = torch.zeros((10, 1, 200, 200), dtype=torch.float32)
    replayed = expected + 1.0e-6
    monkeypatch.setattr(route, "_make_commands", lambda *_args: (replayed, {}))
    runtime = {
        "mapper_replay": {
            "target_coefficients": torch.zeros((10, 100), dtype=torch.complex128),
            "commands": expected,
        },
        "mapper_replay_path": tmp_path / "shard_0000.pt",
    }
    result = route._verify_mapper_replay(nn.Identity(), runtime)
    assert result["max_abs_command_error"] == pytest.approx(1.0e-6)
    assert result["tolerance"] == 5.0e-6

    replayed.fill_(6.0e-6)
    with pytest.raises(RuntimeError, match="saved-command replay"):
        route._verify_mapper_replay(nn.Identity(), runtime)


def test_saved_solver_positive_controls_allow_proposal_ids_reused_across_classes(
    tmp_path, monkeypatch
) -> None:
    rows = tmp_path / "rows"
    rows.mkdir()
    command = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    target = torch.zeros(100, dtype=torch.complex128)
    for label in range(10):
        for proposal_id in range(2):
            torch.save({
                "task_id": "LIGHTGEN_G1_FIXED_FRAME_TRANSFER_DIAGNOSIS_V1",
                "scientific_updates": 0,
                "mapper_updates": 0,
                "optical_updates": 0,
                "base_row": {
                    "label": label,
                    "proposal_id": proposal_id,
                    "target_coefficients": target,
                },
                "solver_results": {"fista_fixed_terminal_phase": {
                    "command": command,
                    "condition": "fista_fixed_terminal_phase",
                }},
            }, rows / f"row_{2 * label + proposal_id:04d}.pt")
    monkeypatch.setattr(route, "_path", lambda _config, _value: rows)

    result = route._load_solver_positive_controls({"sources": {"solver_rows": "unused"}})

    assert len(result) == 20
    assert {label: sum(row["label"] == label for row in result)
            for label in range(10)} == {label: 2 for label in range(10)}


def test_recovery_worker_runs_as_project_module_and_surfaces_saved_result(
    tmp_path, monkeypatch
) -> None:
    recovery_input = tmp_path / "input.pt"
    recovery_output = tmp_path / "output.pt"
    recovery_input.touch()

    def fake_run(command, *, cwd, capture_output, text):
        assert command[0] == route.sys.executable
        assert command[1:3] == [
            "-m", "scripts.research.lightgen.execute_lightgen_recent_optical_fullpath_semantic_v1"
        ]
        assert cwd == route.ROOT
        assert capture_output and text
        torch.save({
            "task_id": route.TASK_ID,
            "completed_updates": 4,
        }, recovery_output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(route.subprocess, "run", fake_run)
    result = route._run_recovery_worker(
        route.DEFAULT_CONFIG, recovery_input, recovery_output
    )
    assert result["completed_updates"] == 4


def test_source_receipt_preserves_absent_checkpoint_branch_metadata(tmp_path) -> None:
    source = tmp_path / "source.pt"
    path_fields = {
        "mapper_checkpoint": str(source),
        "physical_phase_prior": str(source),
        "fixed_response_projector": str(source),
        "mnist_training_file": str(source),
    }
    endpoint = {
        "path": source,
        "checkpoint": {
            "run_id": "joint_run",
            "source_run_id": "source_run",
            "task_id": "joint_task",
            "completed_global_step": 50_000,
        },
        "optimizer_steps": (50_000, 50_000),
    }
    route._save_source_receipt(
        {"sources": {**path_fields, "endpoints": {"J": {}}}}, tmp_path,
        {"J": endpoint},
        {
            "mapper_checkpoint": {"completed_updates": 15_000, "lineage_mapper_updates": 25_000},
            "train_indices": torch.arange(55_000),
            "dev_indices": torch.arange(5_000),
        },
    )
    saved = json.loads((tmp_path / "asset_reuse_receipt.json").read_text(encoding="utf-8"))
    assert saved["sources"]["J"]["checkpoint_branch_field_present"] is False
    assert saved["sources"]["J"]["checkpoint_branch"] is None


def test_frozen_snapshot_detects_fixed_projection_scale_changes() -> None:
    mapper = nn.Linear(1, 1)
    runtime = {
        "prior": {"scale": torch.ones(2), "fit": {0: torch.ones(1)}},
        "scale": torch.ones(2),
        "matrix": torch.eye(2),
        "inverse": torch.eye(2),
        "camera_gain": torch.tensor(1.0),
        "input_bundle": {
            "response_matrix": torch.eye(2),
            "inverse": torch.eye(2),
            "evaluator_ed": nn.Linear(1, 1),
            "evaluator_ec": nn.Linear(1, 1),
        },
    }
    snapshot = route._snapshot_frozen(runtime, mapper)
    runtime["scale"][0] = 2.0
    with pytest.raises(RuntimeError, match="frozen prior, projection, or camera gain"):
        route._assert_frozen_unchanged(snapshot, runtime, mapper)


def test_endpoint_rule_skips_only_for_two_point_average_gain_without_regression() -> None:
    baseline = _endpoint_summary(0.80, 0.80)
    improved = _endpoint_summary(0.82, 0.82)
    skip, evidence = route._should_skip_training(
        "J", {"C45k": baseline, "J": improved, "D0": baseline, "Dgrad": baseline}
    )
    assert skip
    assert evidence["average_macro_recall_delta"] >= 0.02

    regressed = _endpoint_summary(0.84, 0.79)
    skip, evidence = route._should_skip_training(
        "J", {"C45k": baseline, "J": regressed, "D0": baseline, "Dgrad": baseline}
    )
    assert not skip
    assert evidence["EC_macro_recall_delta"] < 0


def test_final_gate_accepts_only_all_three_conditions() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    baseline_ed = labels.clone()
    baseline_ec = labels.clone()
    candidate_ed = labels.clone()
    candidate_ec = labels.clone()
    for label in range(10):
        rows = torch.nonzero(labels == label, as_tuple=False).flatten()
        baseline_ed[rows[:40]] = (label + 1) % 10
        candidate_ed[rows[:20]] = (label + 1) % 10
        baseline_ec[rows[:20]] = (label + 1) % 10
        candidate_ec[rows[:22]] = (label + 1) % 10
    baseline = {"semantic_summary": _semantic_summary(
        baseline_ed, baseline_ec, labels, diversity=1.0
    )}
    candidate = {
        "condition": "final_candidate",
        "semantic_summary": _semantic_summary(
            candidate_ed, candidate_ec, labels, diversity=0.9
        ),
    }
    config = {"measurement": {
        "bootstrap_seed": 202610300970,
        "bootstrap_resamples": 2_000,
        "final_ed_paired_macro_recall_lower_bound_min": 0.0,
        "final_ec_macro_recall_drop_max": 0.02,
        "final_class_diversity_retention_min": 0.8,
    }}
    result = route._final_gate(labels, baseline, candidate, config)
    assert result["passed"]
    assert result["criteria"] == {
        "E_D_paired_bootstrap_lower_bound_gt_zero": True,
        "E_C_macro_recall_drop_at_most_2pp": True,
        "all_classes_retain_at_least_80pct_diversity": True,
    }


def test_final_gate_rejects_ec_drop_and_one_class_diversity_loss() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    baseline_prediction = labels.clone()
    candidate_prediction = labels.clone()
    for label in range(10):
        rows = torch.nonzero(labels == label, as_tuple=False).flatten()
        candidate_prediction[rows[:30]] = (label + 1) % 10
    baseline_summary = _semantic_summary(
        baseline_prediction, baseline_prediction, labels, diversity=1.0
    )
    candidate_summary = _semantic_summary(
        candidate_prediction, candidate_prediction, labels, diversity=1.0
    )
    candidate_summary["same_class_diversity"]["direct_decoder"]["by_class"]["3"][
        "mean_pairwise_normalized_l2"
    ] = 0.79
    candidate = {"condition": "final_candidate", "semantic_summary": candidate_summary}
    result = route._final_gate(labels, {"semantic_summary": baseline_summary}, candidate, {
        "measurement": {
            "bootstrap_seed": 202610300970,
            "bootstrap_resamples": 2_000,
            "final_ed_paired_macro_recall_lower_bound_min": 0.0,
            "final_ec_macro_recall_drop_max": 0.02,
            "final_class_diversity_retention_min": 0.8,
        }
    })
    assert not result["passed"]
    assert not result["criteria"]["E_C_macro_recall_drop_at_most_2pp"]
    assert not result["criteria"]["all_classes_retain_at_least_80pct_diversity"]


def test_final_consumer_rejects_incomplete_population_before_measurement(tmp_path) -> None:
    cohort = {
        "labels": torch.arange(10).repeat_interleave(200)[:-1],
        "coefficients": torch.zeros((1_999, 100), dtype=torch.complex128),
        "proposal_ids": torch.arange(1_999),
    }
    torch.save({
        "schema": f"{route.TASK_ID}.final_population.v1",
        "task_id": route.TASK_ID,
        "seed_base": 202610300930,
        "cohort": cohort,
        "commands": torch.zeros((1_999, 1, 200, 200)),
    }, tmp_path / "final_population_and_commands.pt")
    with pytest.raises(RuntimeError, match="incomplete or malformed"):
        route._consume_final_measurements({}, {}, tmp_path)


def test_final_consumer_reloads_saved_commands_and_full_path_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(route, "COMMAND_SHAPE", (1, 1, 1))
    monkeypatch.setattr(
        route, "_save_torch",
        lambda path, payload, new=False: torch.save(payload, path),
    )
    labels = torch.arange(10).repeat_interleave(200)
    cohort = {
        "role": "new_final_population_not_consumed_during_route_or_training",
        "labels": labels,
        "coefficients": torch.zeros((2_000, 100), dtype=torch.complex128),
        "proposal_ids": torch.arange(2_000),
    }
    commands = torch.ones((2_000, 1, 1, 1))
    torch.save({
        "schema": f"{route.TASK_ID}.final_population.v1",
        "task_id": route.TASK_ID,
        "seed_base": 202610300930,
        "cohort": cohort,
        "commands": commands,
    }, tmp_path / "final_population_and_commands.pt")

    definitions = {
        "C45k": ("baseline", 45_000),
        "selected_initial": ("initial", 50_000),
        "final_candidate": ("terminal", 52_000),
    }
    registry = {"task_id": route.TASK_ID, "conditions": {}}
    for name, (artifact_condition, step) in definitions.items():
        checkpoint_path = tmp_path / f"{name}_checkpoint.pt"
        torch.save({"run_id": name, "completed_global_step": step}, checkpoint_path)
        artifact_name = f"final_{artifact_condition}.pt"
        torch.save({
            "task_id": route.TASK_ID,
            "artifact_condition": artifact_condition,
            "checkpoint_run_id": name,
            "checkpoint_step": step,
            "checkpoint_path": str(checkpoint_path),
            "labels": labels,
            "proposal_ids": cohort["proposal_ids"],
            "commands": commands,
            "raw_cmos": torch.zeros((2_000, 1, 1, 1)),
            "coefficients": torch.zeros((2_000, 100), dtype=torch.complex128),
            "readouts": torch.full((2_000, 1), float(step)),
        }, tmp_path / artifact_name)
        registry["conditions"][name] = {
            "artifact": artifact_name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_run_id": name,
            "checkpoint_step": step,
        }
    (tmp_path / "final_condition_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    replay_commands = []

    def measure(checkpoint, saved_commands, runtime):
        replay_commands.append(saved_commands.clone())
        return {
            "raw_cmos": torch.zeros((2_000, 1, 1, 1)),
            "coefficients": torch.zeros((2_000, 100), dtype=torch.complex128),
            "readouts": torch.full(
                (2_000, 1), float(checkpoint["completed_global_step"])
            ),
        }

    def summary_from_readouts(_inputs, saved_cohort, readouts):
        step = int(readouts[0, 0].item())
        misses = 20 if step == 52_000 else 40
        ed, ec = labels.clone(), labels.clone()
        for label in range(10):
            rows = torch.nonzero(labels == label, as_tuple=False).flatten()
            ed[rows[:misses]] = (label + 1) % 10
            ec[rows[:20]] = (label + 1) % 10
        return _semantic_summary(ed, ec, saved_cohort["labels"], diversity=1.0)

    monkeypatch.setattr(route, "_measure_commands", measure)
    monkeypatch.setattr(route.frame, "_summary_from_readouts", summary_from_readouts)
    result, reconsumed = route._consume_final_measurements({
        "measurement": {
            "bootstrap_seed": 202610300970,
            "bootstrap_resamples": 2_000,
            "final_ed_paired_macro_recall_lower_bound_min": 0.0,
            "final_ec_macro_recall_drop_max": 0.02,
            "final_class_diversity_retention_min": 0.8,
        }
    }, {"input_bundle": {}}, tmp_path)
    assert result["same_saved_slm_commands_for_all_conditions"]
    assert result["replayed_full_production_path_from_saved_commands"]
    assert result["final_gate"]["passed"]
    assert len(replay_commands) == 3
    assert all(torch.equal(value, commands) for value in replay_commands)
    assert reconsumed["final_candidate"]["checkpoint_step"] == 52_000


def test_slm_command_contract_and_mapper_only_optimizer_negative_control() -> None:
    route._assert_legal_commands(torch.zeros((1, 1, 200, 200)))
    route._assert_legal_commands(torch.ones((1, 1, 200, 200)))
    with pytest.raises(RuntimeError, match="SLM commands"):
        route._assert_legal_commands(torch.full((1, 1, 200, 200), 1.01))

    class OpticalPhases(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_phase = nn.Parameter(torch.zeros(2))
            self.decoder_phase = nn.Parameter(torch.zeros(2))

    class ElectronicMapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

    model = OpticalPhases()
    mapper = ElectronicMapper()
    route._assert_optical_optimizer(model, [model.encoder_phase, model.decoder_phase])
    with pytest.raises(RuntimeError, match="exactly Encoder and Generator"):
        route._assert_optical_optimizer(model, list(mapper.parameters()))


def test_resume_checkpoint_requires_the_exact_complete_schedule_and_adam_steps() -> None:
    schedule = torch.arange(2_000 * 10, dtype=torch.long).reshape(2_000, 10)
    config = {"training": {"schedule_seed": 202610300950}}
    selected_checkpoint = {
        "run_id": "selected-run",
        "completed_global_step": 50_000,
        "model_identity": {"encoder": [2, 100, 100], "decoder": [2, 100, 100]},
        "model": {
            "runtime.encoder.phase": torch.zeros((2, 1, 1, 1)),
            "runtime.generator.phase": torch.zeros((2, 1, 1, 1)),
            "frozen.weight": torch.ones(1),
        },
        "weights": {"bogt": 0.2},
        "structural_weight": 0.3,
        "scale": torch.ones(100, dtype=torch.float64),
        "power_reference": torch.tensor(1.0),
        "camera_gain": torch.tensor(2.0),
    }
    selected = {
        "name": "J",
        "path": Path("J.pt"),
        "checkpoint": selected_checkpoint,
        "optimizer_steps": (50_000, 50_000),
    }
    completed = 500
    model_state = {
        "runtime.encoder.phase": torch.ones((2, 1, 1, 1)),
        "runtime.generator.phase": torch.ones((2, 1, 1, 1)),
        "frozen.weight": torch.ones(1),
    }
    history = [{
        "completed_update": index + 1,
        "sample_rows": [int(value) for value in schedule[index].tolist()],
        "labels": list(range(10)),
    } for index in range(completed)]
    checkpoint = {
        "schema": f"{route.TASK_ID}.optical_checkpoint.v1",
        "task_id": route.TASK_ID,
        "run_id": "resume-run",
        "selected_endpoint": "J",
        "source_checkpoint": "J.pt",
        "source_run_id": "selected-run",
        "source_global_step": 50_000,
        "completed_updates": completed,
        "completed_global_step": 50_000 + completed,
        "trainable_groups": ["encoder_phase", "generator_phase"],
        "model_identity": selected_checkpoint["model_identity"],
        "config_identity": config,
        "schedule_seed": config["training"]["schedule_seed"],
        "schedule": schedule,
        "history_count": completed,
        "history": history,
        "optimizer_steps": {
            "encoder_adam_step": 50_000 + completed,
            "generator_adam_step": 50_000 + completed,
        },
        "model": model_state,
        "weights": selected_checkpoint["weights"],
        "structural_weight": selected_checkpoint["structural_weight"],
        "scale": selected_checkpoint["scale"],
        "power_reference": selected_checkpoint["power_reference"],
        "camera_gain": selected_checkpoint["camera_gain"],
        "rng_state": {
            "cpu": torch.zeros(10, dtype=torch.uint8),
            "cuda": [torch.zeros(10, dtype=torch.uint8)],
        },
    }
    assert route._validate_training_checkpoint(checkpoint, selected, schedule, config,
                                              "resume-run") == completed

    invalid = {**checkpoint, "optimizer_steps": {
        "encoder_adam_step": 50_000 + completed,
        "generator_adam_step": 50_001 + completed,
    }}
    with pytest.raises(RuntimeError, match="Adam counters changed"):
        route._validate_training_checkpoint(invalid, selected, schedule, config,
                                            "resume-run")
