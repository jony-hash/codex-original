from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.research.lightgen import (
    diagnose_lightgen_fullpath_evidence_closure_v1 as study,
)
from scripts.research.lightgen import (
    execute_lightgen_small_joint_update_readability_pilot_v1 as joint,
)


def test_registered_closure_config_keeps_both_paths_and_zero_update_budget() -> None:
    config = study._read_config(study.DEFAULT_CONFIG)

    assert config["measurement"]["fullpath_conditions"] == ["P0_E1G1", "P1_E1G1"]
    assert config["output"]["direct_ols_2048_rows"] == 2_000
    assert config["output"]["optimizer_instances"] == 0
    assert config["output"]["optimizer_updates"] == 0
    assert config["measurement"]["bootstrap_seed"] == 202611021170


def test_repair_config_preserves_invalid_first_attempt() -> None:
    config = study._read_repair_config(study.REPAIR_CONFIG)

    assert config["output"]["run_dir"].endswith("runs/attempt_02")
    assert config["output"]["backup_dir"].endswith("runs/autodl_backup_attempt_02")
    assert config["output"]["review_dir"].endswith("runs/windows_review_02")


def test_historical_training_preflight_is_bound_to_d0_not_terminal_e1g1() -> None:
    payload = {
        "status": "READY",
        "selected_endpoint": "D0",
        "selected_global_step": 50_000,
        "selected_checkpoint": "/saved/D0/checkpoint_step_005000.pt",
        "initial_optimizer_steps": [45_000, 50_000],
        "preflight_rolled_back": True,
        "mapper_only_optimizer_rejected": True,
        "scientific_updates": 0,
        "official_test_reads": 0,
        "recovery": {"passed": True},
        "frozen_adapter_projection_prior_camera_and_evaluators_unchanged": True,
        "cmos_only_phase_gradient_norms": {
            "encoder_cmos_gradient_norm": 0.5,
            "generator_cmos_gradient_norm": 0.25,
        },
    }

    scope = study._validate_historical_d0_training_preflight(payload)
    assert scope["endpoint"] == "D0"
    assert scope["global_step"] == 50_000

    wrong_endpoint = dict(payload, selected_endpoint="E1G1", selected_global_step=52_000)
    with pytest.raises(RuntimeError, match="D0 step 50000"):
        study._validate_historical_d0_training_preflight(wrong_endpoint)


def test_terminal_gradient_diagnostics_require_e1g1_source_identity() -> None:
    groups = {
        group: {
            "total_gradient_norm": 1.0,
            "components": {"rec": {"norm": 0.5}, "ssim": {"norm": 0.25}},
        }
        for group in ("encoder", "generator")
    }
    batches = [{"phase_groups": groups} for _ in range(10)]
    finite_difference = []
    for group in ("encoder", "generator"):
        for requested in (0.005, 0.01):
            analytic = 2.0
            central = 1.99
            finite_difference.append(
                {
                    "group": group,
                    "requested_rad": requested,
                    "analytic_directional_derivative": analytic,
                    "central_difference": central,
                    "relative_error": abs(analytic - central) / abs(analytic),
                }
            )
    gradient = {
        "task_id": study.attribution.TASK_ID,
        "optimizer_updates": 0,
        "endpoints": {
            "step_02000": {
                "model_unchanged": True,
                "batches": batches,
                "finite_difference": finite_difference,
            }
        },
    }
    identity = {
        "task_id": study.attribution.TASK_ID,
        "terminal_run_id": "training-run",
        "terminal_global_step": 52_000,
        "optimizer_steps": {
            "encoder_adam_step": 47_000,
            "generator_adam_step": 52_000,
        },
    }

    scope = study._validate_terminal_gradient_diagnostics(
        gradient, identity, expected_training_run_id="training-run"
    )
    assert scope["bound_global_step"] == 52_000
    assert scope["batches"] == 10

    wrong_identity = dict(identity, terminal_global_step=50_000)
    with pytest.raises(RuntimeError, match="not bound to the registered E1G1"):
        study._validate_terminal_gradient_diagnostics(
            gradient, wrong_identity, expected_training_run_id="training-run"
        )


def test_repair_source_failure_does_not_create_partial_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "attempt_02"
    monkeypatch.setattr(study, "REPAIR_OUTPUT_DIR", artifact_dir)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def missing_source(*_args, **_kwargs):
        raise FileNotFoundError("registered source input is missing")

    monkeypatch.setattr(study, "_repair_inputs", missing_source)

    with pytest.raises(FileNotFoundError, match="registered source input"):
        study._run_repair_evidence(
            {"sources": {"base_config": "missing_base.yaml"}},
            tmp_path / "repair.yaml",
            artifact_dir,
        )

    assert not artifact_dir.exists()


def test_direct_fourth_call_log_reconciles_only_from_saved_branch_and_witness(
    tmp_path: Path,
) -> None:
    schedule = torch.arange(40, dtype=torch.long).reshape(4, 10)
    calls = [
        {
            "call_number": index,
            "branch": "fourth_direct" if index == 4 else "three_continuous",
            "status": "STARTED" if index == 4 else "COMPLETED",
            "schedule_row_index": index - 1,
            "sample_rows": schedule[index - 1].tolist(),
        }
        for index in range(1, 5)
    ]
    artifact_dir = tmp_path
    witness_dir = artifact_dir / "training_witnesses"
    witness_dir.mkdir()
    direct = {
        "task_id": study.REPAIR_TASK_ID,
        "branch": "fourth_direct",
        "diagnostic_only_no_candidate": True,
        "source_checkpoint_run_id": "training-run",
        "source_global_step": 52_000,
        "completed_global_step": 52_004,
        "schedule_row_index": 3,
        "rows": schedule[3],
        "optimizer_steps": (47_004, 52_004),
        "loss": {"dist": 1.0, "rec": 0.1, "op": 0.2, "ssim_loss": 0.3, "total": 0.4},
    }
    witness = {
        "source_training_run_id": "training-run",
        "completed_update": 4,
        "forward_global_step": 52_003,
        "completed_global_step": 52_004,
        "optimizer_steps_before": {"encoder": 47_003, "generator": 52_003},
        "optimizer_steps_after": {"encoder": 47_004, "generator": 52_004},
        "source_rows": schedule[3],
    }
    torch.save(direct, artifact_dir / "engineering_direct_branch.pt")
    torch.save(witness, witness_dir / "training_path_witness_update_00004.pt")

    reconciliation = study._reconcile_saved_direct_fourth_call(
        artifact_dir,
        calls,
        {"run_id": "training-run"},
        schedule,
    )

    assert reconciliation["completion_source"] == "saved_direct_branch_and_pre_update_witness"
    assert calls[3]["status"] == "COMPLETED"
    assert calls[3]["optimizer_step_delta"] == {"encoder": 1, "generator": 1}


def test_optimizer_event_consumer_counts_failed_worker_instance_without_step() -> None:
    execution = {"optimizer_events": {"optimizer_instances": 3, "optimizer_updates": 5}}
    events = {
        "parent_process": {"optimizer_instances": 1, "optimizer_updates": 4},
        "failed_recovery_process": {
            "optimizer_instances": 1,
            "optimizer_updates": 0,
            "status": "FAILED_BEFORE_UPDATE",
            "trace_file": "recovery_worker_stderr.txt",
        },
        "independent_recovery_process": {
            "optimizer_instances": 1,
            "optimizer_updates": 1,
        },
        "total": {"optimizer_instances": 3, "optimizer_updates": 5},
        "recovery_worker_failure_reconciled": True,
    }

    assert study._validate_repair_optimizer_event_record(events, execution) == {
        "optimizer_instances": 3,
        "optimizer_updates": 5,
    }

    events["failed_recovery_process"] = {
        **events["failed_recovery_process"],
        "optimizer_updates": 1,
    }
    with pytest.raises(RuntimeError, match="failed recovery-worker optimizer event"):
        study._validate_repair_optimizer_event_record(events, execution)


def test_ready_receipt_uses_the_implementation_engine_declared_by_its_consumer() -> None:
    config = study._read_config(study.DEFAULT_CONFIG)
    control = study.ResearchRunControl(
        study._make_identity(config), authorization_id=study.AUTHORIZATION_ID,
    )
    implementation = {
        "passed": True,
        "status": "PASSED",
        "selected_engine": "frozen_Gaussian_OLS_and_fullpath_CUDA_2048",
        "optimizer_instances": 0,
        "optimizer_updates": 0,
    }

    receipt = study._make_preflight_receipt(
        control,
        correctness={"passed": True, "status": "PASSED"},
        resource={"passed": True, "status": "PASSED"},
        implementation=implementation,
    )

    assert receipt.selected_engine == implementation["selected_engine"]


def test_measurement_cli_forwards_explicit_p1_source_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attempt_dir = tmp_path / "attempt_01"
    p1_source = tmp_path / "terminal_projection_attempt_01"
    p1_review = tmp_path / "independent_review.json"
    training_preflight = tmp_path / "training_preflight.json"
    gradient_diagnostics = tmp_path / "loss_gradient_diagnostics.json"
    captured: dict[str, object] = {}
    monkeypatch.setattr(study, "OUTPUT_DIR", attempt_dir)
    monkeypatch.setattr(study, "_read_config", lambda _path: {"sources": {}})
    monkeypatch.setattr(
        study,
        "_measure",
        lambda config, artifact_dir, **kwargs: captured.update(
            config=config, artifact_dir=artifact_dir, **kwargs
        ),
    )

    assert study.main(
        [
            "--measure",
            "--artifact-dir",
            str(attempt_dir),
            "--projection-source-dir",
            str(p1_source),
            "--projection-review-file",
            str(p1_review),
            "--training-preflight-file",
            str(training_preflight),
            "--gradient-diagnostics-file",
            str(gradient_diagnostics),
        ]
    ) == 0
    assert captured["artifact_dir"] == attempt_dir.resolve()
    assert captured["projection_source_dir"] == p1_source
    assert captured["projection_review_path"] == p1_review
    assert captured["training_preflight_path"] == training_preflight
    assert captured["gradient_diagnostics_path"] == gradient_diagnostics


def test_direct_backend_identity_accepts_resolved_default_cuda_device_and_rejects_other_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = SimpleNamespace(
        global_fiber_geometry=False,
        fiber_attenuation_db_per_km=0.0,
        material=None,
    )
    ols = SimpleNamespace(
        basis_id="gaussian_s12_v1",
        mode_identity=lambda: {"transport": "T=I_100 coefficient space"},
    )
    model_device = torch.device("cuda:0")
    model = SimpleNamespace(
        runtime=SimpleNamespace(
            physical_optics=physical,
            ols=ols,
            config=SimpleNamespace(propagation_grid_size=2048),
            camera_plane=SimpleNamespace(shape=(200, 200)),
        ),
        eval=lambda: None,
        parameters=lambda: iter([SimpleNamespace(device=model_device)]),
    )
    monkeypatch.setattr(study, "_terminal_checkpoint", lambda _assets: {})
    monkeypatch.setattr(study.recent, "_new_endpoint_model", lambda *_args: model)

    assets = {"runtime": {}}
    assert study._new_direct_model(assets, 2048, torch.device("cuda")) is model
    with pytest.raises(RuntimeError, match="registered zero-attenuation"):
        study._new_direct_model(assets, 2048, torch.device("cuda:1"))


def test_path_retention_decision_covers_pass_insufficient_degradation_and_diversity() -> (
    None
):
    ratios = {str(label): 0.9 for label in range(10)}
    passing = {"ED": [-0.01, 0.01], "EC": [-0.015, 0.0]}
    insufficient = {"ED": [-0.03, 0.01], "EC": [-0.01, 0.01]}
    degradation = {"ED": [-0.05, -0.03], "EC": [-0.01, 0.01]}
    poor_diversity = {**ratios, "9": 0.79}

    assert study._path_retention_decision(passing, ratios) == (
        "WITHIN_PREDECLARED_MARGIN_ON_CONSUMED_POPULATION"
    )
    assert study._path_retention_decision(insufficient, ratios) == (
        "EVIDENCE_INSUFFICIENT_AT_NONINFERIORITY_MARGIN"
    )
    assert study._path_retention_decision(degradation, ratios) == (
        "DEGRADATION_BEYOND_MARGIN_SUPPORTED"
    )
    assert (
        study._path_retention_decision(passing, poor_diversity)
        == "DIVERSITY_PROTECTION_FAILED"
    )
    assert (
        study._overall_path_decision(
            {
                "P0_E1G1": "WITHIN_PREDECLARED_MARGIN_ON_CONSUMED_POPULATION",
                "P1_E1G1": "EVIDENCE_INSUFFICIENT_AT_NONINFERIORITY_MARGIN",
            }
        )
        == "ONLY_ONE_FULL_PATH_WITHIN_PREDECLARED_MARGIN"
    )


def test_numpy_fixed_display_matches_torch_adaptive_pooling_from_raw_cmos() -> None:
    rng = np.random.default_rng(2048)
    raw = rng.uniform(0.0, 2.0, size=(3, 1, 200, 200)).astype(np.float64)
    area, gain = 2.5e-7, 3.0e5

    measured = study._numpy_readout(raw, area, gain)
    reference = (
        F.adaptive_avg_pool2d(
            torch.from_numpy(np.clip(raw * area * gain, 0.0, 1.0)),
            (28, 28),
        )
        .numpy()
        .astype(np.float32)
    )

    np.testing.assert_allclose(measured, reference, rtol=0.0, atol=2.0e-7)


def test_p1_population_reconstructs_commands_before_full_command_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    population_rows = 40
    labels = torch.arange(10).repeat_interleave(4)
    proposals = torch.arange(population_rows)
    targets = torch.ones((population_rows, 100), dtype=torch.complex128)
    commands = torch.rand(
        (population_rows, 1, 200, 200),
        generator=torch.Generator().manual_seed(84),
    )
    controls = torch.tensor(
        [label * 4 + offset for label in range(10) for offset in range(2)],
        dtype=torch.long,
    )
    manifest = {
        "task_id": study.alignment.TASK_ID,
        "population_role": "consumed_train_diagnostic",
        "population_rows": population_rows,
        "candidate_projection": "R1_terminal_E1G1_plus_P1",
        "labels": labels,
        "proposal_ids": proposals,
        "prior_targets": targets,
    }
    generation = {
        "labels": labels,
        "proposal_ids": proposals,
        "saved_prior_targets": targets,
    }
    assets = {"generation": generation}
    control_payload = {
        "actual_slm_commands_forwarded": commands.index_select(0, controls),
    }
    cross_grid_payload = {
        "task_id": study.alignment.TASK_ID,
        "population_indices": controls,
        "labels": labels.index_select(0, controls),
        "proposal_ids": proposals.index_select(0, controls),
        "actual_slm_commands_forwarded": commands.index_select(0, controls),
        "coefficients_2048": targets.index_select(0, controls),
        "raw_cmos_2048": torch.zeros((len(controls), 1, 200, 200), dtype=torch.float64),
        "coefficients_4096": targets.index_select(0, controls),
        "raw_cmos_4096": torch.zeros((len(controls), 1, 200, 200), dtype=torch.float64),
    }
    remaining = torch.tensor(
        [index for index in range(population_rows) if index not in set(controls.tolist())],
        dtype=torch.long,
    )
    shard = {
        "population_indices": remaining,
        "actual_slm_commands_forwarded": commands.index_select(0, remaining),
    }
    validation_calls: list[bool] = []

    def load_tensor(path: Path):
        if path.name == "population_manifest.pt":
            return manifest
        if path.name == "controls_2048.pt":
            return control_payload
        if path.name == "cross_grid_controls.pt":
            return cross_grid_payload
        return shard

    def validate_shard(
        saved_shard, indices, reconstructed_commands, _assets, *, compare_commands=True
    ):
        validation_calls.append(compare_commands)
        assert torch.equal(saved_shard["population_indices"], indices)

    def load_candidate(_artifact_dir, _assets, reconstructed_commands):
        assert torch.equal(reconstructed_commands, commands)
        return {
            "labels": labels,
            "proposal_ids": proposals,
            "targets": targets,
            "commands": reconstructed_commands,
            "coefficients": targets.clone(),
            "raw_cmos": torch.zeros((population_rows, 1, 200, 200), dtype=torch.float64),
        }

    monkeypatch.setattr(study, "POPULATION_ROWS", population_rows)
    monkeypatch.setattr(study, "CONTROL_ROWS", controls)
    monkeypatch.setattr(study, "_load_torch", load_tensor)
    monkeypatch.setattr(study.alignment, "CONTROL_ROWS", controls)
    monkeypatch.setattr(study.alignment, "_load_torch", load_tensor)
    monkeypatch.setattr(study.alignment, "_validate_shard", validate_shard)
    monkeypatch.setattr(study.alignment, "_load_candidate_output", load_candidate)

    output = study._load_p1_output(tmp_path, assets)

    assert validation_calls == [False]
    assert torch.equal(output["commands"], commands)
    for field in ("population_indices", "labels", "actual_slm_commands_forwarded", "coefficients_2048"):
        bad_cross_grid = dict(cross_grid_payload)
        bad_cross_grid[field] = bad_cross_grid[field].clone()
        if field == "actual_slm_commands_forwarded":
            bad_cross_grid[field][0, 0, 0, 0] += 0.01
        elif field in {"labels", "population_indices"}:
            bad_cross_grid[field][0] += 1
        else:
            bad_cross_grid[field][0, 0] += 1.0

        def load_corrupt_cross_grid(path: Path):
            return bad_cross_grid if path.name == "cross_grid_controls.pt" else load_tensor(path)

        monkeypatch.setattr(study, "_load_torch", load_corrupt_cross_grid)
        with pytest.raises(RuntimeError, match="cross-grid controls"):
            study._load_p1_output(tmp_path, assets)
        monkeypatch.setattr(study, "_load_torch", load_tensor)


def _training_forward_fixture(monkeypatch: pytest.MonkeyPatch):
    rows = torch.arange(10)
    pixels = torch.zeros((20, 1, 28, 28), dtype=torch.uint8)
    target = torch.full((10, 1, 200, 200), 0.5, dtype=torch.float32)
    labels = torch.full((20,), -1, dtype=torch.long)
    labels[:10] = torch.arange(10)
    train_indices = torch.arange(10)
    monkeypatch.setattr(joint, "image_batch", lambda *_args: target.clone())

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.runtime = SimpleNamespace(
                slm_plane=SimpleNamespace(shape=(200, 200)),
                camera_plane=SimpleNamespace(pixel_pitch_m=(1.0, 1.0)),
                ols=SimpleNamespace(
                    basis_id="test_gaussian", project_state=lambda _field: state
                ),
            )

    coefficients = torch.ones((10, 100), dtype=torch.complex64)
    state = SimpleNamespace(basis_id="test_gaussian", coefficients=coefficients)
    model = FakeModel()
    slm_intensity = target
    slm_amplitude = slm_intensity.sqrt()
    field = SimpleNamespace(
        tensor=torch.complex(slm_amplitude, torch.zeros_like(slm_amplitude))
    )
    encoder_fields = (field, field)
    ols_input = SimpleNamespace(tensor=field.tensor)
    generator_entry = SimpleNamespace(
        tensor=coefficients[:, :1].reshape(10, 1, 1, 1).expand(10, 1, 200, 200)
    )
    generator_exit = SimpleNamespace(tensor=generator_entry.tensor)
    generator_fields = (generator_entry, generator_exit)
    raw = target
    result = {
        "image_input": SimpleNamespace(
            image=target,
            slm_intensity=slm_intensity,
            slm_amplitude=slm_amplitude,
            slm_field=field,
        ),
        "encoder_fields": encoder_fields,
        "decoder_fields": generator_fields,
        "ols_input_field": ols_input,
        "generator_entry_field": generator_entry,
        "generator_exit_field": generator_exit,
        "ols_coefficients": coefficients,
        "plane_powers_w": torch.ones((10, 5)),
        "raw_intensity_w_per_m2": raw,
        "physical_w_per_pixel": raw,
    }
    source = {
        "training_labels": labels,
        "training_indices": train_indices,
        "training_pixels": pixels,
    }
    scale = torch.ones((100,), dtype=torch.float64)
    power_reference = torch.ones((5,), dtype=torch.float64)
    camera_gain = torch.tensor(1.0)
    parts = joint.base._loss_parts_from_result(
        result, target, scale, power_reference, camera_gain,
    )
    checkpoint = {
        "scale": scale,
        "power_reference": power_reference,
        "camera_gain": camera_gain,
    }
    ssim = torch.tensor(0.25)
    monkeypatch.setattr(joint, "resize_view", lambda value: value)
    monkeypatch.setattr(
        joint,
        "ssim_batch",
        lambda _view, _target: torch.full((10,), 0.75),
    )
    return model, pixels, rows, source, result, parts, checkpoint, target, ssim


def test_fullpath_training_guard_accepts_bound_native_slm_to_cmos_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, pixels, rows, source, result, parts, checkpoint, _target, ssim = (
        _training_forward_fixture(monkeypatch)
    )

    witness = joint._validate_fullpath_training_forward(
        model,
        pixels,
        rows,
        source,
        result,
        parts,
        checkpoint,
        ssim,
    )

    assert witness["source_rows"].tolist() == list(range(10))
    assert witness["source_labels"].tolist() == list(range(10))
    assert witness["path"].startswith("TRAIN image -> native amplitude SLM")


def test_fullpath_training_guard_rejects_direct_ols_and_wrong_source_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, pixels, rows, source, result, parts, checkpoint, target, ssim = (
        _training_forward_fixture(monkeypatch)
    )
    direct_ols = dict(result)
    direct_ols["image_input"] = None
    with pytest.raises(RuntimeError, match="actual amplitude-SLM input"):
        joint._validate_fullpath_training_forward(
            model,
            pixels,
            rows,
            source,
            direct_ols,
            parts,
            checkpoint,
            ssim,
        )

    wrong_pixels = pixels.clone()
    wrong_pixels[0].fill_(255)
    monkeypatch.setattr(
        joint,
        "image_batch",
        lambda source_pixels, *_args: (
            target.clone() if source_pixels is pixels else target.clone() + 0.1
        ),
    )
    with pytest.raises(
        RuntimeError, match="not bound to the registered source TRAIN rows"
    ):
        joint._validate_fullpath_training_forward(
            model,
            wrong_pixels,
            rows,
            source,
            result,
            parts,
            checkpoint,
            ssim,
        )


def test_fullpath_training_guard_rejects_wrong_labels_and_disconnected_encoder_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, pixels, rows, source, result, parts, checkpoint, _target, ssim = (
        _training_forward_fixture(monkeypatch)
    )
    wrong_labels = {**source, "training_labels": source["training_labels"].clone()}
    wrong_labels["training_labels"][0] = 1
    with pytest.raises(RuntimeError, match="registered class-balanced TRAIN batch"):
        joint._validate_fullpath_training_forward(
            model,
            pixels,
            rows,
            wrong_labels,
            result,
            parts,
            checkpoint,
            ssim,
        )

    disconnected = dict(result)
    fields = list(result["encoder_fields"])
    fields[-1] = SimpleNamespace(tensor=fields[-1].tensor + 1.0)
    disconnected["encoder_fields"] = tuple(fields)
    with pytest.raises(RuntimeError, match="saved layer traces are disconnected"):
        joint._validate_fullpath_training_forward(
            model,
            pixels,
            rows,
            source,
            disconnected,
            parts,
            checkpoint,
            ssim,
        )


def _direct_shard_fixture() -> tuple[dict, torch.Tensor, dict]:
    indices = torch.tensor([0, 1], dtype=torch.long)
    labels = torch.tensor([0, 1], dtype=torch.long)
    proposals = torch.tensor([7, 8], dtype=torch.long)
    targets = torch.ones((2, 100), dtype=torch.complex128)
    targets[1] *= 2.0
    raw = torch.ones((2, 1, 200, 200), dtype=torch.float64)
    area = 1.0e-12
    shard = {
        "schema": "lightgen_direct_ols_population_shard.v1",
        "task_id": study.TASK_ID,
        "condition": "D_DIRECT_OLS",
        "population_role": "consumed_train_diagnostic",
        "endpoint_source_run_id": "d88d5bfc2b29488884a88512152815d6",
        "endpoint_global_step": 52_000,
        "encoder_adam_step": 47_000,
        "generator_adam_step": 52_000,
        "grid_size": 2048,
        "ols_basis_id": "gaussian_s12_v1",
        "ols_transport": "T=I_100 coefficient space",
        "modal_coefficient_scale": "registered_native_complex_OLS_scale",
        "raw_cmos_unit": "W/m^2",
        "w_per_pixel_unit": "W/pixel",
        "readout_unit": "dimensionless_[0,1]",
        "camera_pixel_area_m2": area,
        "camera_gain_w_inverse": 1.0,
        "population_indices": indices,
        "labels": labels,
        "proposal_ids": proposals,
        "targets": targets,
        "coefficients": targets.clone(),
        "raw_cmos": raw,
        "w_per_pixel": raw * area,
        "readouts": torch.zeros((2, 1, 28, 28)),
        "modal_power_native": targets.abs().square().sum(1),
    }
    assets = {
        "generation": {
            "labels": labels,
            "proposal_ids": proposals,
            "saved_prior_targets": targets,
        },
        "runtime": {"camera_area": area, "camera_gain": 1.0},
        "alignment": {
            "assets": {
                "checkpoints": {
                    2_000: {
                        "run_id": "d88d5bfc2b29488884a88512152815d6",
                        "completed_global_step": 52_000,
                        "optimizer_steps": {
                            "encoder_adam_step": 47_000,
                            "generator_adam_step": 52_000,
                        },
                    }
                }
            }
        },
    }
    return shard, indices, assets


def test_direct_ols_consumer_accepts_valid_shard_and_rejects_wrong_units_or_basis() -> (
    None
):
    shard, indices, assets = _direct_shard_fixture()
    study._validate_direct_shard(shard, indices, assets)

    for key, value in (("raw_cmos_unit", "W/pixel"), ("ols_basis_id", "lp01_ports_v1")):
        invalid = {**shard, key: value}
        with pytest.raises(RuntimeError, match="incomplete, duplicated, or bound"):
            study._validate_direct_shard(invalid, indices, assets)

    wrong_power = {**shard, "w_per_pixel": shard["w_per_pixel"] * 1_000.0}
    with pytest.raises(RuntimeError, match=r"W/m\^2 to W/pixel"):
        study._validate_direct_shard(wrong_power, indices, assets)


def test_direct_population_producer_keeps_physical_pixel_power_from_controls_and_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    control_rows = torch.tensor([0, 1], dtype=torch.long)
    labels = torch.arange(4, dtype=torch.long)
    proposals = torch.arange(10, 14, dtype=torch.long)
    targets = torch.ones((4, 100), dtype=torch.complex128)
    raw_control = torch.ones((2, 1, 200, 200), dtype=torch.float64)
    area = 2.0e-8
    control = {
        "population_indices": control_rows,
        "labels": labels.index_select(0, control_rows),
        "proposal_ids": proposals.index_select(0, control_rows),
        "targets": targets.index_select(0, control_rows),
        "raw_cmos": raw_control,
        "w_per_pixel": raw_control * area,
        "readouts": torch.full((2, 1, 28, 28), 0.1),
    }
    captured = {}

    def direct_batch(_model, batch_targets, camera_area, camera_gain):
        count = len(batch_targets)
        raw = torch.full((count, 1, 200, 200), 2.0, dtype=torch.float64)
        return {
            "raw_cmos": raw,
            "w_per_pixel": raw * camera_area,
            "readouts": torch.full((count, 1, 28, 28), 0.2),
        }

    monkeypatch.setattr(study, "POPULATION_ROWS", 4)
    monkeypatch.setattr(study, "CHUNK_ROWS", 4)
    monkeypatch.setattr(study, "CONTROL_ROWS", control_rows)
    monkeypatch.setattr(study, "_new_direct_model", lambda *_args: object())
    monkeypatch.setattr(study, "_direct_batch", direct_batch)
    monkeypatch.setattr(study, "_validate_direct_shard", lambda *_args: None)
    monkeypatch.setattr(
        study,
        "_save_direct_shard",
        lambda _path, indices, measured, _assets: captured.update(
            indices=indices.clone(),
            **{key: value.clone() for key, value in measured.items()},
        ),
    )
    assets = {
        "generation": {
            "labels": labels,
            "proposal_ids": proposals,
            "saved_prior_targets": targets,
        },
        "runtime": {"camera_area": area, "camera_gain": 1.0},
    }

    study._measure_direct_population(
        tmp_path, assets, {"direct_2048": control}, torch.device("cpu")
    )

    assert torch.equal(captured["indices"], torch.arange(4))
    assert torch.equal(captured["w_per_pixel"][:2], control["w_per_pixel"])
    assert torch.equal(
        captured["w_per_pixel"][2:],
        torch.full((2, 1, 200, 200), 2.0 * area, dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "field", ["population_indices", "labels", "proposal_ids", "targets"]
)
def test_direct_ols_consumer_rejects_misbound_population_rows(field: str) -> None:
    shard, indices, assets = _direct_shard_fixture()
    invalid = {**shard, field: shard[field].flip(0)}

    with pytest.raises(RuntimeError, match="incomplete, duplicated, or bound"):
        study._validate_direct_shard(invalid, indices, assets)


def test_direct_ols_population_consumer_rejects_missing_chunk_then_accepts_complete_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "POPULATION_ROWS", 4)
    monkeypatch.setattr(study, "CHUNK_ROWS", 2)
    shard, _indices, assets = _direct_shard_fixture()
    labels = torch.arange(4, dtype=torch.long)
    proposals = torch.arange(10, 14, dtype=torch.long)
    targets = torch.ones((4, 100), dtype=torch.complex128)
    assets["generation"] = {
        "labels": labels,
        "proposal_ids": proposals,
        "saved_prior_targets": targets,
    }

    def make_chunk(start: int, end: int) -> dict:
        rows = torch.arange(start, end, dtype=torch.long)
        values = {**shard}
        raw = torch.ones((len(rows), 1, 200, 200), dtype=torch.float64)
        values.update(
            {
                "population_indices": rows,
                "labels": labels[rows],
                "proposal_ids": proposals[rows],
                "targets": targets[rows],
                "coefficients": targets[rows],
                "raw_cmos": raw,
                "w_per_pixel": raw * assets["runtime"]["camera_area"],
                "readouts": torch.zeros((len(rows), 1, 28, 28)),
                "modal_power_native": targets[rows].abs().square().sum(1),
            }
        )
        return values

    root = tmp_path / "direct_population"
    root.mkdir()
    torch.save(make_chunk(0, 2), root / "direct_ols_2048_0000_0002.pt")
    with pytest.raises(FileNotFoundError, match="required evidence is missing"):
        study._load_direct_population(root, assets)

    torch.save(make_chunk(2, 4), root / "direct_ols_2048_0002_0004.pt")
    result = study._load_direct_population(root, assets)
    assert result["coefficients"].shape == (4, 100)
    assert torch.equal(result["labels"], labels)


def _cross_grid_consumer_fixture() -> tuple[dict, dict, dict]:
    direct_coefficients = torch.arange(1, 2_001, dtype=torch.float64).reshape(20, 100)
    targets = direct_coefficients.to(torch.complex128)
    direct_raw = torch.ones((20, 1, 200, 200), dtype=torch.float64)
    direct_raw *= torch.arange(1, 21, dtype=torch.float64).reshape(20, 1, 1, 1)
    direct_wpixel = direct_raw * 2.0
    row_scale = torch.linspace(1.000001, 1.00002, 20, dtype=torch.float64)
    direct4096 = {
        "coefficients": targets.clone(),
        "raw_cmos": direct_raw * row_scale.reshape(20, 1, 1, 1),
        "w_per_pixel": direct_wpixel * row_scale.reshape(20, 1, 1, 1),
    }
    p1_coefficients = torch.ones((20, 100), dtype=torch.complex128)
    p1_raw = direct_raw * 1.5
    p1_row_scale = torch.linspace(1.000001, 1.00004, 20, dtype=torch.float64)
    p1_cross = {
        "coefficients_2048": p1_coefficients,
        "coefficients_4096": p1_coefficients * p1_row_scale.reshape(20, 1),
        "raw_cmos_2048": p1_raw,
        "raw_cmos_4096": p1_raw * (p1_row_scale + 1.0e-5).reshape(20, 1, 1, 1),
    }
    saved = {
        "direct_2048": {
            "coefficients": targets,
            "raw_cmos": direct_raw,
            "w_per_pixel": direct_wpixel,
        },
        "direct_4096": direct4096,
    }
    assets = {
        "generation": {"saved_prior_targets": targets},
        "p1": {"cross_grid_controls": p1_cross},
    }
    return assets, saved, {
        "direct2048_coefficients": direct_coefficients.numpy(),
        "direct4096_coefficients": direct4096["coefficients"].numpy(),
        "direct2048_raw": direct_raw.numpy(),
        "direct4096_raw": direct4096["raw_cmos"].numpy(),
        "direct2048_wpixel": direct_wpixel.numpy(),
        "direct4096_wpixel": direct4096["w_per_pixel"].numpy(),
    }


def test_saved_cross_grid_consumer_reports_per_row_maxima_and_pooled_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    control_rows = torch.arange(20, dtype=torch.long)
    monkeypatch.setattr(study, "CONTROL_ROWS", control_rows)
    monkeypatch.setattr(study, "CROSS_GRID_TOLERANCE", 1.0e-4)
    assets, saved, arrays = _cross_grid_consumer_fixture()
    monkeypatch.setattr(study, "_load_torch", lambda _path: saved)
    monkeypatch.setattr(study, "_validate_direct_shard", lambda *_args: None)

    result = study._recompute_saved_cross_grid_controls(assets, tmp_path)

    assert result["passed"] is True
    direct = result["D_DIRECT_OLS"]
    direct_coeff_rows = study._numpy_row_relative_l2(
        arrays["direct2048_coefficients"], arrays["direct4096_coefficients"]
    )
    assert direct["coefficient_per_row"] == pytest.approx(direct_coeff_rows.tolist())
    assert direct["coefficient_max_relative_error"] == pytest.approx(0.0)
    assert direct["coefficient_aggregate_relative_error"] == pytest.approx(0.0)
    assert direct["raw_cmos_aggregate_relative_error"] == pytest.approx(
        study._numpy_aggregate_relative_l2(
            arrays["direct2048_raw"], arrays["direct4096_raw"]
        )
    )
    p1 = result["P1_E1G1"]
    assert p1["coefficient_aggregate_relative_error"] == pytest.approx(
        study._numpy_aggregate_relative_l2(
            assets["p1"]["cross_grid_controls"]["coefficients_2048"].numpy(),
            assets["p1"]["cross_grid_controls"]["coefficients_4096"].numpy(),
        )
    )
    assert p1["raw_cmos_aggregate_relative_error"] == pytest.approx(
        study._numpy_aggregate_relative_l2(
            assets["p1"]["cross_grid_controls"]["raw_cmos_2048"].numpy(),
            assets["p1"]["cross_grid_controls"]["raw_cmos_4096"].numpy(),
        )
    )


def test_saved_cross_grid_consumer_rejects_one_bad_p1_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    control_rows = torch.arange(20, dtype=torch.long)
    monkeypatch.setattr(study, "CONTROL_ROWS", control_rows)
    assets, saved, _arrays = _cross_grid_consumer_fixture()
    assets["p1"]["cross_grid_controls"]["raw_cmos_4096"][1] *= 1.001
    monkeypatch.setattr(study, "_load_torch", lambda _path: saved)
    monkeypatch.setattr(study, "_validate_direct_shard", lambda *_args: None)

    with pytest.raises(RuntimeError, match="above 1e-4"):
        study._recompute_saved_cross_grid_controls(assets, tmp_path)


def _joint_update_fixture(monkeypatch: pytest.MonkeyPatch, fault: str | None = None):
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Parameter(torch.tensor(0.7))
            self.generator = torch.nn.Parameter(torch.tensor(-0.4))

    model = ToyModel()
    optimizer = torch.optim.Adam((model.encoder, model.generator), lr=1.0e-2)
    optimizer.zero_grad(set_to_none=True)
    (model.encoder.square() + model.generator.square()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    monkeypatch.setattr(
        joint.base,
        "_phase_groups",
        lambda current: (current.encoder, current.generator),
    )
    monkeypatch.setattr(
        joint,
        "_validate_fullpath_training_forward",
        lambda *_args: {"path": "bound full path"},
    )
    source = {
        "checkpoint": {
            "weights": {"dist": 0.1, "rec": 1.0, "op": 0.1},
            "structural_weight": 0.2,
            "scale": torch.ones(100),
            "power_reference": torch.ones(5),
            "camera_gain": torch.tensor(1.0),
        }
    }
    observed: dict[str, float] = {}

    def loss_parts(*_args, **_kwargs):
        raw = model.encoder * model.generator
        rec = (raw - 0.3).square()
        ssim = raw.square()
        parts = {
            "dist": (model.encoder + model.generator).square(),
            "rec": rec,
            "op": (model.encoder - model.generator).square(),
        }
        if fault in {"detach_rec", "detach_rec_and_ssim"}:
            parts["rec"] = parts["rec"].detach()
        if fault in {"detach_ssim", "detach_rec_and_ssim"}:
            ssim = ssim.detach()
        if fault == "detach_rec_and_ssim":
            remaining = joint.decoder_ref._objective(
                parts, ssim, None, source["checkpoint"]["weights"], 0.2, None,
            )
            gradients = torch.autograd.grad(
                remaining, (model.encoder, model.generator), retain_graph=True,
            )
            observed["non_cmos_total_gradient_norm"] = sum(
                float(torch.linalg.vector_norm(value.detach())) for value in gradients
            )
        return parts, ssim, None, {
            "physical_w_per_pixel": raw,
            "rec": rec,
            "ssim": raw.square(),
            "plane_powers_w": torch.ones((10, 5)),
        }

    monkeypatch.setattr(joint.decoder_ref, "_loss_parts", loss_parts)
    monkeypatch.setattr(
        joint,
        "_expected_cmos_terms",
        lambda result, *_args: (result["rec"], result["ssim"]),
    )
    monkeypatch.setattr(joint, "_save_training_path_witness_if_selected", lambda *_args, **_kwargs: None)
    pixels = torch.zeros((10, 1, 28, 28), dtype=torch.uint8)
    rows = torch.arange(10)
    return model, optimizer, pixels, rows, source, observed


@pytest.mark.parametrize("fault", ["detach_rec", "detach_ssim", "detach_rec_and_ssim"])
def test_joint_update_rejects_detached_cmos_terms_before_optimizer_step(
    monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    model, optimizer, pixels, rows, source, observed = _joint_update_fixture(
        monkeypatch, fault,
    )
    before_model = {key: value.clone() for key, value in model.state_dict().items()}
    before_adam = copy.deepcopy(optimizer.state_dict())
    step_calls = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    message = "reconstruction CMOS loss is detached" if fault != "detach_ssim" else "SSIM CMOS loss is detached"
    with pytest.raises(RuntimeError, match=message):
        joint._joint_update(model, optimizer, pixels, rows, torch.device("cpu"), source)

    assert step_calls == 0
    assert all(torch.equal(before_model[key], value) for key, value in model.state_dict().items())
    after_adam = optimizer.state_dict()
    assert before_adam["param_groups"] == after_adam["param_groups"]
    for parameter_id, before_state in before_adam["state"].items():
        for key, value in before_state.items():
            actual = after_adam["state"][parameter_id][key]
            assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
    if fault == "detach_rec_and_ssim":
        assert observed["non_cmos_total_gradient_norm"] > 0.0


def test_joint_update_consumes_cmos_gradients_for_both_phase_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, optimizer, pixels, rows, source, _observed = _joint_update_fixture(monkeypatch)
    before = (model.encoder.detach().clone(), model.generator.detach().clone())
    steps_before = joint._optimizer_steps(optimizer, model.encoder, model.generator)

    loss = joint._joint_update(model, optimizer, pixels, rows, torch.device("cpu"), source)

    assert torch.isfinite(torch.tensor(loss["total"]))
    assert not torch.equal(before[0], model.encoder)
    assert not torch.equal(before[1], model.generator)
    assert joint._optimizer_steps(optimizer, model.encoder, model.generator) == (
        steps_before[0] + 1,
        steps_before[1] + 1,
    )


def test_joint_update_rejects_single_group_freeze_before_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, optimizer, pixels, rows, source, _observed = _joint_update_fixture(monkeypatch)
    model.generator.requires_grad_(False)
    before_model = {key: value.clone() for key, value in model.state_dict().items()}
    before_adam = copy.deepcopy(optimizer.state_dict())
    step_calls = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    with pytest.raises(RuntimeError, match="joint phase groups"):
        joint._joint_update(model, optimizer, pixels, rows, torch.device("cpu"), source)

    assert step_calls == 0
    assert all(torch.equal(before_model[key], value) for key, value in model.state_dict().items())
    after_adam = optimizer.state_dict()
    assert before_adam["param_groups"] == after_adam["param_groups"]
    for parameter_id, before_state in before_adam["state"].items():
        for key, value in before_state.items():
            actual = after_adam["state"][parameter_id][key]
            assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual


def test_independent_numpy_path_decision_covers_margin_and_diversity_cases() -> None:
    ratios = {str(label): 0.9 for label in range(10)}
    passing = {"ED": [-0.01, 0.0], "EC": [-0.02, 0.01]}
    insufficient = {"ED": [-0.021, 0.01], "EC": [-0.01, 0.01]}
    degradation = {"ED": [-0.04, -0.03], "EC": [-0.01, 0.01]}
    assert study._numpy_path_retention_decision(passing, ratios) == (
        "WITHIN_PREDECLARED_MARGIN_ON_CONSUMED_POPULATION"
    )
    assert study._numpy_path_retention_decision(insufficient, ratios) == (
        "EVIDENCE_INSUFFICIENT_AT_NONINFERIORITY_MARGIN"
    )
    assert study._numpy_path_retention_decision(degradation, ratios) == (
        "DEGRADATION_BEYOND_MARGIN_SUPPORTED"
    )
    poor_diversity = {**ratios, "9": 0.79}
    assert study._numpy_path_retention_decision(passing, poor_diversity) == (
        "DIVERSITY_PROTECTION_FAILED"
    )


def test_saved_optical_adam_review_does_not_construct_an_optimizer() -> None:
    state = {
        index: {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.zeros((2, 1, 100, 100)),
            "exp_avg_sq": torch.ones((2, 1, 100, 100)),
        }
        for index, step in enumerate((47_000, 52_000))
    }
    payload = {
        "state": state,
        "param_groups": [{
            "params": [0, 1],
            "lr": 1.0e-3,
            "betas": (0.9, 0.999),
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
        }],
    }
    audit = study._OptimizerEventAudit()
    with audit:
        study.recent._validate_saved_optical_adam(payload, (47_000, 52_000))
        audit.require_zero("read-only saved Adam validation")
    assert audit.snapshot() == {"optimizer_instances": 0, "optimizer_updates": 0}


def test_endpoint_loader_validates_saved_adam_without_constructing_optimizers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    encoder_steps = {"C45k": 45_000, "J": 47_000, "D0": 45_000, "Dgrad": 45_000}
    generator_steps = {"C45k": 45_000, "J": 52_000, "D0": 47_000, "Dgrad": 52_000}
    specs = {}
    checkpoint_models = {}
    model_identity = {"encoder": [2, 100, 100], "decoder": [2, 100, 100], "ols": "gaussian_s12_v1"}
    for name in study.recent.ENDPOINT_ORDER:
        encoder = torch.zeros((2, 1, 100, 100))
        generator = torch.zeros((2, 1, 100, 100))
        if name == "J":
            encoder.fill_(0.1)
            generator.fill_(0.2)
        elif name in {"D0", "Dgrad"}:
            generator.fill_(0.3 if name == "D0" else 0.4)
        model = {
            "runtime.encoder.phase": encoder,
            "runtime.generator.phase": generator,
            "runtime.fixed_state": torch.ones((1,)),
        }
        checkpoint_models[name] = model
        checkpoint_path = tmp_path / f"{name}.pt"
        spec = {
            "checkpoint": checkpoint_path,
            "checkpoint_branch": None,
            "task_id": f"task-{name}",
            "checkpoint_run_id": f"run-{name}",
            "source_run_id": "source-run",
            "global_step": 50_000,
            "encoder_adam_step": encoder_steps[name],
            "generator_adam_step": generator_steps[name],
        }
        specs[name] = spec
        state = {
            parameter_id: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.zeros((2, 1, 100, 100)),
                "exp_avg_sq": torch.ones((2, 1, 100, 100)),
            }
            for parameter_id, step in enumerate((encoder_steps[name], generator_steps[name]))
        }
        torch.save(
            {
                "task_id": spec["task_id"],
                "run_id": spec["checkpoint_run_id"],
                "source_run_id": spec["source_run_id"],
                "completed_global_step": spec["global_step"],
                "model_identity": model_identity,
                "model": model,
                "evaluator_role": "frozen_confirmation_only",
                "scale": torch.ones((100,)),
                "power_reference": torch.ones((5,)),
                "camera_gain": torch.tensor(1.0),
                "weights": {"dist": 1.0, "rec": 1.0, "op": 1.0},
                "structural_weight": 0.1,
                "optimizer": {
                    "state": state,
                    "param_groups": [{
                        "params": [0, 1],
                        "lr": 1.0e-3,
                        "betas": (0.9, 0.999),
                        "eps": 1.0e-8,
                        "weight_decay": 0.0,
                        "amsgrad": False,
                    }],
                },
            },
            checkpoint_path,
        )
    monkeypatch.setattr(study.recent, "_path", lambda _config, path: Path(path))
    monkeypatch.setattr(torch.optim, "Adam", lambda *_args, **_kwargs: pytest.fail("read-only endpoint loading constructed Adam"))
    runtime = {"input_bundle": {"checkpoint": {"model": checkpoint_models["C45k"]}}}

    endpoints = study.recent._load_endpoints({"sources": {"endpoints": specs}}, runtime)

    assert set(endpoints) == set(study.recent.ENDPOINT_ORDER)
    assert endpoints["D0"]["optimizer_steps"] == (45_000, 47_000)


def test_optimizer_event_audit_counts_real_construction_and_step() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    audit = study._OptimizerEventAudit()
    with audit:
        optimizer = torch.optim.Adam([parameter], lr=1.0e-2)
        optimizer.zero_grad(set_to_none=True)
        parameter.square().backward()
        optimizer.step()
    assert audit.snapshot() == {"optimizer_instances": 1, "optimizer_updates": 1}


def test_optimizer_event_audit_does_not_count_a_failed_step_as_an_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class FailingOptimizer(torch.optim.Optimizer):
        def __init__(self, params: object) -> None:
            super().__init__(params, {"lr": 1.0})

        def step(self, closure: object = None) -> None:
            raise RuntimeError("injected optimizer failure")

    audit = study._OptimizerEventAudit()
    with audit:
        optimizer = FailingOptimizer([parameter])
        with pytest.raises(RuntimeError, match="injected optimizer failure"):
            optimizer.step()
    assert audit.snapshot() == {"optimizer_instances": 1, "optimizer_updates": 0}


def test_training_path_witness_binds_forward_step_and_adam_counters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = {
        "checkpoint": {
            "run_id": "source-run",
            "completed_global_step": 52_000,
        },
        "training_witness_dir": tmp_path,
        "active_training_update": 1,
        "training_capture_updates": [1],
    }
    witness = {
        "source_rows": torch.arange(10),
        "source_labels": torch.arange(10),
        "slm_intensity": torch.zeros((10, 1, 2, 2)),
        "slm_amplitude": torch.zeros((10, 1, 2, 2)),
        "ols_coefficients": torch.zeros((10, 100), dtype=torch.complex64),
        "plane_powers_w": torch.zeros((10, 5), dtype=torch.float64),
        "raw_cmos_w_per_m2": torch.zeros((10, 1, 200, 200), dtype=torch.float64),
    }
    monkeypatch.setattr(
        joint,
        "atomic_torch_save_new",
        lambda path, payload: torch.save(payload, path),
    )
    joint._save_training_path_witness_if_selected(
        source, witness, steps_before=(47_000, 52_000), steps_after=(47_001, 52_001),
    )
    saved = torch.load(
        tmp_path / "training_path_witness_update_00001.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["forward_global_step"] == 52_000
    assert saved["completed_global_step"] == 52_001
    assert saved["optimizer_steps_before"] == {"encoder": 47_000, "generator": 52_000}
    assert saved["optimizer_steps_after"] == {"encoder": 47_001, "generator": 52_001}
