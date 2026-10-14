"""Focused contract, numerical, and bounded-resume checks for Chen BOGT."""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as chen
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as executor
from scripts.research.lightgen import execute_lightgen_2d_semantic_complex_ols_training_v1 as complex_ols_executor


def test_resume_input_comparison_ignores_digests_but_keeps_execution_fields() -> None:
    before = {
        "source_path": "data/MNIST/processed/training.pt",
        "source_sha256": "old-content",
        "split_seed": 1234,
        "train_indices_sha256": "old-rows",
        "role": "TRAIN",
    }
    after = {**before, "source_sha256": "new-content", "train_indices_sha256": "new-rows"}

    assert executor._resume_identity(before) == executor._resume_identity(after)
    assert executor._resume_identity(before) != executor._resume_identity({**after, "split_seed": 999})


@pytest.mark.parametrize("changed", [None, "images", "labels", "train_indices", "development_indices", "evaluator_state"])
def test_resume_checks_saved_actual_inputs(tmp_path: Path, changed: str | None) -> None:
    evaluator = torch.nn.Linear(2, 2)
    inputs = {"images": torch.zeros(4, 1, 2, 2), "labels": torch.tensor([0, 1, 0, 1]), "train_indices": torch.tensor([0, 1]), "development_indices": torch.tensor([2, 3])}
    executor._bind_resume_inputs(tmp_path, evaluator=evaluator, resume=False, **inputs)
    if changed == "evaluator_state":
        with torch.no_grad():
            evaluator.weight.add_(1)
    elif changed is not None:
        inputs[changed] = inputs[changed].clone()
        inputs[changed].reshape(-1)[0] += 1
    if changed is None:
        executor._bind_resume_inputs(tmp_path, evaluator=evaluator, resume=True, **inputs)
    else:
        with pytest.raises(executor.RunControlError, match="resume actual input changed"):
            executor._bind_resume_inputs(tmp_path, evaluator=evaluator, resume=True, **inputs)


def test_legacy_resume_cannot_create_a_new_input_baseline(tmp_path: Path) -> None:
    with pytest.raises(executor.RunControlError, match="input snapshot is missing"):
        executor._bind_resume_inputs(tmp_path, images=torch.zeros(1), labels=torch.zeros(1, dtype=torch.long), train_indices=torch.tensor([0]), development_indices=torch.tensor([], dtype=torch.long), evaluator=torch.nn.Linear(1, 1), resume=True)
    assert not list(tmp_path.iterdir())


def test_gaussian_bogt_reference_producer_records_basis_and_independent_values() -> None:
    torch.manual_seed(0)
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu",
        propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID,
        ols_model=chen.GAUSSIAN_OLS_MODEL,
    ).eval()
    images = torch.linspace(0.05, 0.95, 10 * 28 * 28, dtype=torch.float32).reshape(10, 1, 28, 28)
    with torch.no_grad():
        result = consumer.runtime.forward_from_image(images)
    references = chen.collect_bogt_train_references(
        consumer,
        images,
        data_role=chen.TRAIN_SMOKE_ROLE,
        engineering_only=True,
        data_identity={"role": chen.TRAIN_SMOKE_ROLE, "purpose": "engineering_only"},
    )

    independent_scale = np.mean(np.abs(result.z_ols.coefficients.detach().cpu().numpy().astype(np.complex128)) ** 2, axis=0)
    np.testing.assert_allclose(references.scale.numpy(), independent_scale, rtol=1.0e-12, atol=1.0e-18)
    fields = (*result.encoder_records, result.generator_input_field, *result.generator_records)
    independent_powers = []
    for field in fields:
        value = field.tensor.detach().cpu().numpy().astype(np.complex128)
        area = math.prod(field.plane.pixel_pitch_m)
        independent_powers.append(np.sum(np.abs(value) ** 2, axis=tuple(range(1, value.ndim))) * area)
    np.testing.assert_allclose(
        references.power_reference.numpy(), np.mean(np.stack(independent_powers, axis=1), axis=0), rtol=1.0e-12, atol=1.0e-18
    )
    assert references.data_role == chen.TRAIN_SMOKE_ROLE
    assert references.engineering_only is True
    assert references.count == 10
    assert references.basis_id == chen.GAUSSIAN_OLS_MODEL
    assert references.mode_geometry == consumer.runtime.ols.mode_identity()
    restored = chen.FrozenBogtReferences.from_payload(references.to_payload())
    consumer.bind_references(restored)
    with pytest.raises(chen.CheckpointError, match="another OLS modal basis"):
        consumer.bind_references(replace(references, basis_id=chen.LP01_OLS_MODEL, mode_geometry=None))
    with pytest.raises(chen.TrainingContractError, match="explicit ComplexOLSState"):
        consumer.generated_forward(result.z_ols.coefficients)


@pytest.mark.parametrize("changed", [None, "images", "evaluator", "missing"])
def test_scientific_resume_consumes_actual_input_check_before_model_construction(monkeypatch, tmp_path: Path, changed: str | None) -> None:
    target = tmp_path / "outputs/20261006_bogt/runs/attempt_01"
    target.mkdir(parents=True)
    images, labels = torch.zeros(4, 1, 2, 2), torch.tensor([0, 1, 0, 1])
    train, dev = torch.tensor([0, 1]), torch.tensor([2, 3])
    evaluator = torch.nn.Linear(2, 2)
    if changed != "missing":
        executor._bind_resume_inputs(target, images=images, labels=labels, train_indices=train, development_indices=dev, evaluator=evaluator, resume=False)
    if changed == "images":
        images[0, 0, 0, 0] = 0.5
    if changed == "evaluator":
        with torch.no_grad():
            evaluator.weight.add_(1)
    identity = executor.ResearchRunIdentity(executor.NEXT_TASK, "unit", "unit", executor.SEMANTIC_AUTHORITY_ID, executor.TRAIN_ROLE, {}, {}, {}, {}, scientific_attempt_id="unit:attempt_01")
    manifest = {"schema": executor.RUN_MANIFEST_SCHEMA, "source_manifest": {"source_manifest_id": "unit"}, "run_identity": identity.to_dict(), "authorization_id": "unit", "data_identity": {"source_sha256": "historical"}, "evaluator_identity": {}, "run_config": {}}
    (target / "execution_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(executor, "_production_cuda_memory", lambda *_: {})
    monkeypatch.setattr(executor, "_load_scientific_dataset", lambda *_: (images, labels, train, dev, {"source_sha256": "current"}))
    monkeypatch.setattr(executor, "_load_evaluator", lambda *_: (evaluator, {}))
    monkeypatch.setattr(executor, "_make_run_config", lambda *_: {})
    def construction_boundary(**kwargs):
        raise RuntimeError("model construction reached")
    monkeypatch.setattr(executor, "LightGen2DChenBOGTComplexOLSTrainingConsumer", construction_boundary)
    expected = "model construction reached" if changed is None else ("snapshot is missing" if changed == "missing" else "actual input changed")
    with pytest.raises((RuntimeError, executor.RunControlError), match=expected):
        executor.run_scientific_execution(repo_root=tmp_path, resume=target)
    assert not list(target.glob("scientific_step_*.pt"))


def test_resume_theta0_identity_ignores_git_and_moved_runtime_module_provenance() -> None:
    before = {
        "config": {"propagation_grid_size": 2048, "phase_zero": True},
        "git": {"branch": "research/old", "head": "old-head", "worktree_dirty": True},
        "runtime_module": "src/old/location/runtime.py",
    }
    after = {
        "config": {"propagation_grid_size": 2048, "phase_zero": True},
        "git": {"branch": "research/current", "head": "new-head", "worktree_dirty": False},
        "runtime_module": "src/new/location/runtime.py",
    }

    assert chen._resume_identity(before) == chen._resume_identity(after)
    assert chen._resume_identity(before) != chen._resume_identity({**after, "config": {"propagation_grid_size": 1024, "phase_zero": True}})


def test_train_loader_records_changed_file_digest_without_rejecting_valid_roles(monkeypatch, tmp_path: Path) -> None:
    train_path = tmp_path / "data" / "MNIST" / "processed" / "training.pt"
    train_path.parent.mkdir(parents=True)
    train_path.write_bytes(b"local dataset payload differs from historical contract")

    images = torch.zeros((1, 1, 28, 28), dtype=torch.uint8).expand(60_000, -1, -1, -1)
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    monkeypatch.setattr(complex_ols_executor, "load_local_mnist", lambda *_args, **_kwargs: (images, labels))
    contract_payload = {
        "data_contract": {
            "root": "data/MNIST",
            "source_artifacts": {
                "data/MNIST/processed/training.pt": {
                    "path": "data/MNIST/processed/training.pt",
                    "sha256": "historical-digest",
                },
                "data/MNIST/processed/test.pt": {
                    "path": "data/MNIST/processed/test.pt",
                    "sha256": "heldout-digest",
                },
            },
        }
    }

    loaded_images, loaded_labels, train_rows, dev_rows, roles = complex_ols_executor._load_train_role(
        tmp_path, contract_payload
    )

    assert loaded_images.shape == (60_000, 1, 28, 28)
    assert loaded_labels.shape == (60_000,)
    assert train_rows.numel() == 55_000
    assert dev_rows.numel() == 5_000
    assert roles["source_artifacts"]["training"]["contract_sha256"] == "historical-digest"
    assert roles["source_artifacts"]["training"]["actual_sha256"] != "historical-digest"
    assert roles["source_artifacts"]["official_test"]["read"] is False


def test_preflight_records_git_changes_without_rejecting_them(monkeypatch, tmp_path: Path) -> None:
    train_path = tmp_path / "data" / "MNIST" / "processed" / "training.pt"
    train_path.parent.mkdir(parents=True)
    train_path.write_bytes(b"training source")
    runtime_path = tmp_path / complex_ols_executor.ALIGNMENT_RUNTIME_RELATIVE_PATH
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_text("runtime provenance fixture", encoding="utf-8")
    images = torch.zeros((1, 1, 28, 28), dtype=torch.uint8).expand(60_000, -1, -1, -1)
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    monkeypatch.setattr(complex_ols_executor, "load_local_mnist", lambda *_args, **_kwargs: (images, labels))
    contract_payload = {
        "data_contract": {
            "root": "data/MNIST",
            "source_artifacts": {
                "data/MNIST/processed/training.pt": {
                    "path": "data/MNIST/processed/training.pt",
                    "sha256": "historical-digest",
                },
                "data/MNIST/processed/test.pt": {
                    "path": "data/MNIST/processed/test.pt",
                    "sha256": "heldout-digest",
                },
            },
        }
    }
    fake_contract = SimpleNamespace(
        payload=contract_payload,
        identity=lambda _root: {"contract": "fixture"},
    )
    git_records = iter(
        (
            {"branch": "before", "head": "one", "worktree_dirty": False},
            {"branch": "after", "head": "two", "worktree_dirty": True},
        )
    )
    monkeypatch.setattr(complex_ols_executor, "load_frozen_training_contract", lambda repo_root: fake_contract)
    monkeypatch.setattr(complex_ols_executor, "git_facts", lambda _root: next(git_records))
    monkeypatch.setattr(complex_ols_executor, "validate_legacy_exclusion", lambda: {})
    monkeypatch.setattr(complex_ols_executor, "_smoke", lambda **_kwargs: {"parameter_manifest": {"fixture": True}})

    result = complex_ols_executor.run_preflight(tmp_path, tmp_path / "preflight")
    payload = json.loads(Path(result["artifact"]).read_text(encoding="utf-8"))

    assert payload["git"]["branch_unchanged"] is False
    assert payload["git"]["head_unchanged"] is False
    assert payload["git"]["starting"]["head"] == "one"
    assert payload["git"]["ending"]["head"] == "two"


@pytest.mark.parametrize("free_bytes", [0, 1024**3, 128 * 1024**2])
def test_execute_does_not_impose_cuda_memory_quota(monkeypatch, tmp_path: Path, free_bytes: int) -> None:
    target = tmp_path / "outputs" / "runs" / "preflight-blocked"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (free_bytes, 8 * 1024**3))
    def stop_at_input_validation(*_args, **_kwargs):
        raise executor.TrainingContractError("input validation reached")

    monkeypatch.setattr(
        executor,
        "_load_scientific_dataset",
        stop_at_input_validation,
    )

    with pytest.raises(executor.TrainingContractError, match="input validation reached"):
        executor.run_scientific_execution(repo_root=tmp_path, output_dir=target)

    assert not target.exists()
    assert not (tmp_path / "outputs").exists()


def test_execute_rejects_cuda_unavailability_before_loading_inputs(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "20261002_bogt" / "runs" / "attempt_01"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        executor, "_load_scientific_dataset",
        lambda *_args, **_kwargs: pytest.fail("CUDA failure must stop input loading"),
    )
    with pytest.raises(executor.RunControlError, match="CUDA"):
        executor.run_scientific_execution(repo_root=tmp_path, output_dir=target)
    assert not target.exists()


def test_unopened_fresh_preflight_failure_removes_scientific_artifacts_and_keeps_engineering_record(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "20261002_bogt" / "runs" / "attempt_01"
    target.mkdir(parents=True)
    (target / "frozen_references.pt").write_bytes(b"preflight-only artifact")
    failure = executor.RunControlError("PREFLIGHT_BLOCKED(resource): synthetic test")

    executor._discard_unopened_fresh_attempt_after_preflight_failure(
        repo_root=tmp_path,
        target=target,
        error=failure,
    )

    assert not target.exists()
    records = list(target.with_name("attempt_01.preflight_failures").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["status"] == "PREFLIGHT_BLOCKED(resource)"
    assert record["scientific_updates"] == 0
    assert record["scientific_attempt_opened"] is False
    assert record["fresh_attempt_artifacts_removed"] is True


def test_execute_preparation_oom_is_engineering_failure_without_a_scientific_attempt(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "20261002_bogt" / "runs" / "attempt_01"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (128 * 1024**2, 8 * 1024**3))
    monkeypatch.setattr(executor, "_load_scientific_dataset", lambda *_args: (None, None, None, None, {}))
    monkeypatch.setattr(executor, "_load_evaluator", lambda *_args: (None, {}))
    monkeypatch.setattr(executor, "_make_run_config", lambda *_args: {})
    monkeypatch.setattr(executor, "git_identity", lambda *_args: {"head": "unit", "branch": "unit"})
    manifest = {"execution_source_semantics_id": "unit", "source_manifest_id": "unit"}
    monkeypatch.setattr(executor, "capture_execution_source_manifest", lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: manifest))
    real_save = executor._save_new_json

    def fail_during_preparation(path, payload):
        if path.name == "source_manifest.json":
            raise torch.cuda.OutOfMemoryError("actual allocation failure boundary")
        return real_save(path, payload)

    monkeypatch.setattr(executor, "_save_new_json", fail_during_preparation)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        executor.run_scientific_execution(repo_root=tmp_path, output_dir=target)
    assert not target.exists()
    record = json.loads(next(target.with_name("attempt_01.preflight_failures").glob("*.json")).read_text(encoding="utf-8"))
    assert record["status"] == "PREFLIGHT_BLOCKED(resource)"
    assert record["scientific_updates"] == 0
    assert record["scientific_attempt_opened"] is False


def test_execute_rejection_preserves_existing_attempt_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "20261002_bogt" / "runs" / "attempt_01"
    target.mkdir(parents=True)
    evidence = target / "existing.pt"
    evidence.write_bytes(b"existing evidence")
    with pytest.raises(executor.RunControlError, match="exists"):
        executor.run_scientific_execution(repo_root=tmp_path, output_dir=target)
    assert evidence.read_bytes() == b"existing evidence"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--production-preflight"],
        ["--production-preflight", "--checkpoint", "checkpoint.pt"],
        ["--production-preflight", "--output-dir", "outputs/20261003_engineering/runs/diagnostic-only_test"],
        ["--execute", "--checkpoint", "checkpoint.pt"],
        ["--preflight", "--output-dir", "outputs/20261003_engineering/runs/diagnostic-only_test"],
        ["--execute", "--production-preflight", "--checkpoint", "checkpoint.pt", "--output-dir", "outputs/20261003_engineering/runs/diagnostic-only_test"],
    ],
)
def test_production_preflight_cli_rejects_missing_or_misplaced_inputs(monkeypatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["bogt_executor", *arguments])

    with pytest.raises(SystemExit) as error:
        executor.main()

    assert error.value.code == 2


@pytest.mark.parametrize("status,expected_exit", [("READY", 0), ("PREFLIGHT_BLOCKED(resource)", 1)])
def test_production_preflight_cli_dispatches_diagnostic_and_never_science(
    monkeypatch, capsys, status: str, expected_exit: int
) -> None:
    checkpoint = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01/scientific_step_000100.pt")
    output = Path("outputs/20261003_engineering_governance/runs/diagnostic-only_step000100")
    monkeypatch.setattr(sys, "argv", ["bogt_executor", "--production-preflight", "--checkpoint", str(checkpoint), "--output-dir", str(output)])
    monkeypatch.setattr(
        executor,
        "run_production_checkpoint_preflight",
        lambda **kwargs: {
            "status": status,
            "diagnostic_only": True,
            "scientific_updates": 0,
            "received": {key: str(value) for key, value in kwargs.items()},
        },
    )
    monkeypatch.setattr(
        executor,
        "run_scientific_execution",
        lambda **_kwargs: pytest.fail("production preflight mode must exit before scientific execution"),
    )

    assert executor.main() == expected_exit
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == status
    assert result["diagnostic_only"] is True
    assert result["scientific_updates"] == 0
    assert result["received"] == {"checkpoint": str(checkpoint), "output_dir": str(output)}


def test_production_preflight_missing_checkpoint_records_engineering_failure_without_science(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "20261003_engineering_governance" / "runs" / "diagnostic-only_missing_checkpoint"

    result = executor.run_production_checkpoint_preflight(
        repo_root=tmp_path,
        checkpoint="outputs/source/scientific_step_000000.pt",
        output_dir=target,
    )

    assert result["status"] == "INVALID_HARNESS"
    assert result["diagnostic_only"] is True
    assert result["scientific_updates"] == 0
    assert result["scientific_attempt_opened"] is False
    assert json.loads((target / "preflight_report.json").read_text(encoding="utf-8"))["status"] == "INVALID_HARNESS"
    assert "INVALID_HARNESS" in (target / "preflight_progress.jsonl").read_text(encoding="utf-8")


def test_production_preflight_refuses_existing_diagnostic_output_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "20261003_engineering_governance" / "runs" / "diagnostic-only_existing"
    target.mkdir(parents=True)
    marker = target / "keep.txt"
    marker.write_text("existing diagnostic evidence", encoding="utf-8")

    with pytest.raises(executor.RunControlError, match="already exists"):
        executor.run_production_checkpoint_preflight(
            repo_root=tmp_path,
            checkpoint="outputs/source/scientific_step_000000.pt",
            output_dir=target,
        )

    assert marker.read_text(encoding="utf-8") == "existing diagnostic evidence"


def test_cpu_engineering_preflight_exercises_and_fully_restores_three_step_trial() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    result = executor.run_engineering_preflight(repo_root=repo_root)

    correctness = result["preflight_receipt"]["checks"]["CORRECTNESS_INVARIANTS"]
    assert result["preflight_status"] == "READY"
    assert result["scientific_updates"] == 0
    assert correctness["passed"] is True
    assert correctness["three_real_updates"] == 3
    assert correctness["checkpoint_model_state_equivalent"] is True
    assert correctness["checkpoint_optimizer_state_equivalent"] is True
    assert correctness["checkpoint_sampler_state_equivalent"] is True
    assert correctness["checkpoint_loss_trace_equivalent"] is True
    assert correctness["loaded_model_optimizer_rng_sampler_restored"] is True


def _images() -> torch.Tensor:
    return torch.linspace(0.05, 0.95, 10 * 28 * 28, dtype=torch.float32).reshape(10, 1, 28, 28)


def _theta0_identity() -> dict[str, str]:
    return {"state_identity_sha256": "fixture-theta0"}


def _fixture_references(consumer: chen.LightGen2DChenBOGTComplexOLSTrainingConsumer) -> chen.FrozenBogtReferences:
    return chen.FrozenBogtReferences(
        torch.ones(100, dtype=torch.float64),
        torch.ones(5, dtype=torch.float64),
        chen.SYNTHETIC_ROLE,
        10,
        True,
        consumer.initialization_fingerprint(),
        {"source": "unit fixture"},
    )


def _prepared_consumer() -> tuple[
    chen.LightGen2DChenBOGTComplexOLSTrainingConsumer,
    chen.FrozenBogtReferences,
    chen.BogtLossWeights,
    torch.Tensor,
    chen.DataReadLedger,
]:
    images = _images()
    ledger = chen.DataReadLedger()
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    references = chen.collect_bogt_train_references(
        consumer,
        images,
        data_role=chen.SYNTHETIC_ROLE,
        engineering_only=True,
        source_indices=torch.arange(10, dtype=torch.long),
        data_identity={"source": "unit fixture"},
        ledger=ledger,
    )
    weights = chen.calibrate_initial_gradient_weights(
        consumer,
        images,
        references,
        data_role=chen.SYNTHETIC_ROLE,
        calibration_batch_identity={
            "source": "unit fixture",
            "seed": chen.SCHEDULE_SEED,
            "batch_size": chen.LOGICAL_BATCH_SIZE,
            "class_balanced": True,
        },
        engineering_only=True,
        ledger=ledger,
    )
    consumer.bind_weights(weights)
    return consumer, references, weights, images, ledger


def _phase_fixture() -> tuple[torch.Tensor, chen.FrozenBogtReferences, torch.Tensor]:
    references = chen.FrozenBogtReferences(
        torch.ones(100, dtype=torch.float64),
        torch.ones(5, dtype=torch.float64),
        chen.SYNTHETIC_ROLE,
        24,
        True,
        _theta0_identity(),
        {"source": "phase unit fixture"},
    )
    rows = 24
    u = torch.full((rows, 100), 0.25, dtype=torch.float64)
    u[:, 0] = torch.linspace(2.0, 3.5, rows, dtype=torch.float64)
    u[:, 1] = torch.linspace(0.7, 1.7, rows, dtype=torch.float64)
    phase = torch.zeros((rows, 100), dtype=torch.float64)
    phase[:, 1] = 0.1 * torch.log(u[:, 0])
    phase[:, 2] = -0.2 * torch.log(u[:, 1])
    phase[:, 3] = 0.05 * torch.log(u[:, 0] * u[:, 1])
    amplitude = u.sqrt()
    z = torch.complex(amplitude * phase.cos(), amplitude * phase.sin()).to(torch.complex64)
    return z, references, u


def test_parameter_firewall_and_optimizer_contract() -> None:
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    named = list(consumer.runtime.named_parameters())
    assert [name for name, _ in named] == list(chen.ALLOWED_TRAINABLE_PARAMETER_NAMES)
    assert all(torch.equal(parameter, torch.zeros_like(parameter)) for _, parameter in named)
    assert all(not buffer.requires_grad for buffer in consumer.runtime.buffers())
    manifest = consumer.parameter_manifest()
    assert manifest["optimizer_parameter_names"] == list(chen.ALLOWED_TRAINABLE_PARAMETER_NAMES)
    assert manifest["parameter_shapes"] == {name: list(shape) for name, shape in chen.EXPECTED_PARAMETER_SHAPES.items()}

    optimizer = consumer.create_optimizer()
    optimizer_manifest = consumer.optimizer_manifest(optimizer)
    assert optimizer_manifest["name"] == "Adam"
    assert optimizer_manifest["learning_rate"] == chen.LEARNING_RATE
    assert tuple(optimizer_manifest["betas"]) == chen.ADAM_BETAS
    assert optimizer_manifest["weight_decay"] == 0.0
    assert optimizer_manifest["amsgrad"] is False

    with pytest.raises(chen.TrainingContractError):
        consumer.create_optimizer([parameter for _, parameter in named] + [torch.nn.Parameter(torch.ones(1))])


def test_statistical_variable_is_normalized_modal_power_and_gauge_invariant() -> None:
    scale = torch.arange(1, 101, dtype=torch.float64)
    references = chen.FrozenBogtReferences(
        scale,
        torch.ones(5, dtype=torch.float64),
        chen.SYNTHETIC_ROLE,
        2,
        True,
        _theta0_identity(),
        {"source": "u unit fixture"},
    )
    z = torch.zeros((2, 100), dtype=torch.complex64)
    z[0, 0] = torch.complex(torch.tensor(2.0), torch.tensor(0.0))
    z[0, 1] = torch.complex(torch.tensor(0.0), torch.tensor(3.0))
    z[1, 0] = torch.complex(torch.tensor(4.0), torch.tensor(0.0))
    z[1, 1] = torch.complex(torch.tensor(0.0), torch.tensor(5.0))
    u = chen.statistical_variable(z, references)
    assert u[0, 0].item() == pytest.approx(4.0)
    assert u[0, 1].item() == pytest.approx(9.0 / 2.0)
    angle = torch.tensor(0.63)
    rotation = torch.complex(angle.cos(), angle.sin())
    assert torch.allclose(u, chen.statistical_variable(z * rotation, references))


def _numeric_tn_positive_kl(q_loc: float, q_variance: float, p_loc: float, p_variance: float) -> float:
    dtype = torch.float64
    q_sigma = math.sqrt(q_variance)
    p_sigma = math.sqrt(p_variance)
    upper = max(50.0, q_loc + 14.0 * q_sigma, p_loc + 14.0 * p_sigma)
    x = torch.linspace(0.0, upper, 120_001, dtype=dtype)
    sqrt_two = math.sqrt(2.0)
    q_z = 0.5 * (1.0 + math.erf(q_loc / (q_sigma * sqrt_two)))
    p_z = 0.5 * (1.0 + math.erf(p_loc / (p_sigma * sqrt_two)))
    q_log_density = (
        -0.5 * ((x - q_loc) / q_sigma).square()
        - math.log(q_sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(q_z)
    )
    p_log_density = (
        -0.5 * ((x - p_loc) / p_sigma).square()
        - math.log(p_sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(p_z)
    )
    return float(torch.trapezoid(torch.exp(q_log_density) * (q_log_density - p_log_density), x))


@pytest.mark.parametrize(
    ("q_loc", "q_variance", "p_loc", "p_variance"),
    (
        (0.0, 0.25, 1.0, 0.25),
        (0.5, 0.5, 1.0, 0.25),
        (1.0, 0.01, 1.0, 0.25),
        (1.3, 0.5, 1.0, 0.25),
        (1.0, 0.25, 1.0, 0.25),
        (3.0, 4.0, 1.0, 0.25),
    ),
)
def test_positive_truncated_normal_kl_matches_independent_quadrature(
    q_loc: float,
    q_variance: float,
    p_loc: float,
    p_variance: float,
) -> None:
    analytic = chen.tn_positive_kl(
        torch.tensor(q_loc),
        torch.tensor(q_variance),
        torch.tensor(p_loc),
        torch.tensor(p_variance),
    )
    numeric = _numeric_tn_positive_kl(q_loc, q_variance, p_loc, p_variance)
    assert torch.isfinite(analytic)
    assert float(analytic) >= -1.0e-9
    assert float(analytic) == pytest.approx(numeric, abs=3.0e-4)


def test_q_uses_population_moments_and_variance_floor() -> None:
    constant = torch.full((10, 100), 2.0, dtype=torch.float64)
    moments = chen.q_batch_moments(constant)
    assert torch.equal(moments.mean, torch.full((100,), 2.0, dtype=torch.float64))
    assert torch.equal(moments.variance, torch.full((100,), chen.VARIANCE_FLOOR, dtype=torch.float64))

    varying = constant.clone()
    varying[:, 0] = torch.arange(10, dtype=torch.float64)
    varying_moments = chen.q_from_batch(varying)
    assert varying_moments.loc[0].item() == pytest.approx(4.5)
    assert varying_moments.variance[0].item() == pytest.approx(8.25)


def test_optical_power_loss_uses_five_plane_relative_log_power() -> None:
    reference = torch.ones(5, dtype=torch.float64)
    equal = torch.ones((2, 5), dtype=torch.float64)
    assert chen.optical_power_loss_from_powers(equal, reference).item() == pytest.approx(0.0, abs=1.0e-12)
    high = torch.full((1, 5), 2.0, dtype=torch.float64)
    low = torch.full((1, 5), 0.5, dtype=torch.float64)
    assert chen.optical_power_loss_from_powers(high, reference).item() > 0.0
    assert chen.optical_power_loss_from_powers(low, reference).item() > 0.0
    near_zero = torch.zeros((1, 5), dtype=torch.float64)
    assert torch.isfinite(chen.optical_power_loss_from_powers(near_zero, reference))


def test_reference_collector_is_engineering_only_and_float64() -> None:
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    ledger = chen.DataReadLedger()
    references = chen.collect_bogt_train_references(
        consumer,
        _images(),
        data_role=chen.SYNTHETIC_ROLE,
        engineering_only=True,
        source_indices=torch.arange(10, dtype=torch.long),
        data_identity={"source": "collector unit fixture"},
        ledger=ledger,
    )
    assert references.engineering_only is True
    assert references.count == 10
    assert references.scale.dtype == torch.float64
    assert references.power_reference.dtype == torch.float64
    assert torch.isfinite(references.scale).all()
    assert torch.isfinite(references.power_reference).all()
    assert ledger.counts[chen.SYNTHETIC_ROLE] == 10


@pytest.mark.parametrize("bad_value", ("zero", "nan", "inf"))
def test_weight_calibration_rejects_invalid_initial_gradients(
    bad_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    images = _images()
    references = _fixture_references(consumer)

    def invalid_gradients(outputs: object, inputs: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        del outputs, kwargs
        parameters = list(inputs)  # type: ignore[arg-type]
        fill = {"zero": 0.0, "nan": float("nan"), "inf": float("inf")}[bad_value]
        return tuple(torch.full_like(parameter, fill) for parameter in parameters)

    monkeypatch.setattr(torch.autograd, "grad", invalid_gradients)
    with pytest.raises(chen.TrainingContractError):
        chen.calibrate_initial_gradient_weights(
            consumer,
            images,
            references,
            data_role=chen.SYNTHETIC_ROLE,
            calibration_batch_identity={
                "seed": chen.SCHEDULE_SEED,
                "batch_size": chen.LOGICAL_BATCH_SIZE,
                "class_balanced": True,
            },
            engineering_only=True,
        )


def test_loss_components_are_finite_and_real_update_changes_only_allowed_phases() -> None:
    consumer, references, weights, images, ledger = _prepared_consumer()
    before_frozen = consumer.frozen_state_sha256
    breakdown = consumer.forward_loss(images, data_role=chen.SYNTHETIC_ROLE)
    assert all(torch.isfinite(value) for value in (breakdown.L_dist, breakdown.L_rec, breakdown.L_op, breakdown.L_total))
    expected_total = weights.alpha * breakdown.L_dist + weights.beta * breakdown.L_rec + weights.gamma * breakdown.L_op
    assert torch.allclose(breakdown.L_total, expected_total)

    optimizer = consumer.create_optimizer()
    update = consumer.bogt_training_step(
        images,
        optimizer,
        global_step=0,
        data_role=chen.SYNTHETIC_ROLE,
        ledger=ledger,
    )
    assert update.autograd.projection_active is True
    assert update.autograd.writer_active is True
    assert update.autograd.forbidden_gradients_absent is True
    assert all(value > 0.0 for value in update.parameter_update_norms.values())
    assert update.frozen_state_unchanged is True
    assert consumer.frozen_state_sha256 == before_frozen
    assert ledger.counts[chen.SYNTHETIC_ROLE] == 30


def _sampler() -> chen.DeterministicTrainRoleSampler:
    return chen.DeterministicTrainRoleSampler(
        torch.arange(chen.TRAIN_COUNT, dtype=torch.long),
        torch.arange(60_000, dtype=torch.long).remainder(10),
    )


@pytest.mark.parametrize("failure_mode", ["none", "frozen_violation", "midpoint_reload"])
@pytest.mark.parametrize("completed_updates", [1, 2])
def test_shared_preflight_continues_loaded_state_and_restores_after_success_or_failure(
    monkeypatch, failure_mode: str, completed_updates: int
) -> None:
    consumer, _references, _weights, images, ledger = _prepared_consumer()
    optimizer = consumer.create_optimizer()
    sampler = _sampler()
    for _ in range(completed_updates):
        sampler.next_batch()
    first_update = consumer.bogt_training_step(images, optimizer, global_step=0, data_role=chen.SYNTHETIC_ROLE)
    for state in optimizer.state.values():
        state["step"].fill_(completed_updates)
    model_before = {name: value.detach().clone() for name, value in consumer.runtime.state_dict().items()}
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in consumer.runtime.named_parameters()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    sampler_before = copy.deepcopy(sampler.state_dict())
    rng_before = executor._rng_snapshot()
    real_step = optimizer.step
    states_seen = []
    batches_seen = []
    global_steps_seen = []
    real_training_step = consumer.bogt_training_step

    def capture_training_step(batch, *args, **kwargs):
        batches_seen.append(batch.detach().clone())
        global_steps_seen.append(kwargs["global_step"])
        return real_training_step(batch, *args, **kwargs)

    def step_with_optional_frozen_violation(*args, **kwargs):
        states_seen.append(copy.deepcopy(optimizer.state_dict()))
        result = real_step(*args, **kwargs)
        if failure_mode == "frozen_violation":
            with torch.no_grad():
                next(consumer.runtime.buffers()).add_(1)
        return result

    monkeypatch.setattr(optimizer, "step", step_with_optional_frozen_violation)
    monkeypatch.setattr(consumer, "bogt_training_step", capture_training_step)
    if failure_mode == "midpoint_reload":
        real_load_checkpoint = executor.load_checkpoint

        def fail_midpoint_reload(path, *args, **kwargs):
            if Path(path).name == "after_first_update.pt":
                raise chen.TrainingContractError("injected midpoint reload failure")
            return real_load_checkpoint(path, *args, **kwargs)

        monkeypatch.setattr(executor, "load_checkpoint", fail_midpoint_reload)
    identity = executor.ResearchRunIdentity(
        task_id="isolated-preflight-test", run_id="unit", trajectory_id="unit",
        semantic_authority_id="unit", data_role=chen.SYNTHETIC_ROLE,
        config_identity={"grid": 128}, model_identity="bogt", optimizer_identity="Adam",
        sampler_identity={"seed": chen.SCHEDULE_SEED},
    )
    arguments = dict(
        consumer=consumer, optimizer=optimizer, sampler=sampler,
        images=images.repeat(6_000, 1, 1, 1),
        labels=torch.arange(60_000, dtype=torch.long).remainder(10),
        train_indices=torch.arange(55_000, dtype=torch.long),
        run_identity=identity, source_manifest={}, authorization_id="unit",
        data_role=chen.SYNTHETIC_ROLE, engineering_only=True,
        completed_updates=completed_updates,
        training_history=[first_update.metrics()] * completed_updates,
        ledger=ledger,
    )
    if failure_mode != "none":
        message = "frozen runtime state changed" if failure_mode == "frozen_violation" else "injected midpoint reload failure"
        with pytest.raises(chen.TrainingContractError, match=message) as error:
            executor._run_isolated_preflight_trial(**arguments)
        restoration = getattr(error.value, "production_preflight_restoration")
        assert restoration["verified"] is True
    else:
        result = executor._run_isolated_preflight_trial(**arguments)
        assert result["checks"]["passed"] is True
        assert result["checks"]["three_real_updates"] == 3
        assert result["checks"]["checkpoint_rng_state_equivalent"] is True
        assert result["loaded_completed_updates"] == completed_updates
        assert result["trial_completed_updates"] == completed_updates + 3
        assert result["optimizer_global_step_indices"] == list(range(completed_updates, completed_updates + 3))
        assert result["completed_update_steps"] == list(range(completed_updates + 1, completed_updates + 4))
        assert result["sampler_cursor_start"] == completed_updates
        assert result["sampler_cursor_end"] == completed_updates + 3
        assert result["restoration"]["verified"] is True
        assert result["checks"]["checkpoint_step_sampler_adam_history_consistent"] is True
        assert global_steps_seen == [
            *range(completed_updates, completed_updates + 3),
            *range(completed_updates + 1, completed_updates + 3),
        ]
        assert executor._tree_close(batches_seen[1:3], batches_seen[3:5])

    expected_trial = _sampler()
    expected_trial.load_state_dict(sampler_before)
    expected_indices, _ = expected_trial.next_batch()
    if failure_mode == "none":
        assert result["first_sampler_batch_indices"] == expected_indices.tolist()
    assert torch.equal(
        batches_seen[0], arguments["images"].index_select(0, expected_indices)
    )

    assert executor._tree_close(states_seen[0], optimizer_before)
    assert executor._tree_close(model_before, consumer.runtime.state_dict())
    gradients_after = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in consumer.runtime.named_parameters()
    }
    assert executor._tree_close(gradients_before, gradients_after)
    assert executor._tree_close(optimizer_before, optimizer.state_dict())
    assert executor._tree_close(sampler_before, sampler.state_dict())
    assert executor._tree_close(rng_before, executor._rng_snapshot())
    expected_next = _sampler()
    expected_next.load_state_dict(sampler_before)
    assert executor._tree_close(sampler.next_batch(), expected_next.next_batch())


@pytest.mark.parametrize("corruption", ["adam_step", "adam_moment", "sampler_cursor"])
def test_shared_preflight_rejects_invalid_loaded_state_before_updates(monkeypatch, corruption: str) -> None:
    consumer, _references, _weights, images, ledger = _prepared_consumer()
    optimizer = consumer.create_optimizer()
    sampler = _sampler()
    sampler.next_batch()
    first = consumer.bogt_training_step(images, optimizer, global_step=0, data_role=chen.SYNTHETIC_ROLE)
    state = next(iter(optimizer.state.values()))
    if corruption == "adam_step":
        state["step"].add_(1)
    elif corruption == "adam_moment":
        state["exp_avg"].flatten()[0] = float("nan")
    else:
        sampler.next_batch()
    monkeypatch.setattr(consumer, "bogt_training_step", lambda *_args, **_kwargs: pytest.fail("invalid checkpoint must not reach an update"))
    identity = executor.ResearchRunIdentity(
        task_id="loaded-state-test", run_id="unit", trajectory_id="unit",
        semantic_authority_id="unit", data_role=chen.SYNTHETIC_ROLE,
        config_identity={"grid": 128}, model_identity="bogt", optimizer_identity="Adam",
        sampler_identity={"seed": chen.SCHEDULE_SEED},
    )
    with pytest.raises(chen.TrainingContractError, match="loaded checkpoint"):
        executor._run_isolated_preflight_trial(
            consumer=consumer, optimizer=optimizer, sampler=sampler,
            images=images, labels=torch.arange(60_000, dtype=torch.long).remainder(10),
            train_indices=torch.arange(55_000, dtype=torch.long),
            run_identity=identity, source_manifest={}, authorization_id="unit",
            data_role=chen.SYNTHETIC_ROLE, engineering_only=True,
            completed_updates=1, training_history=[first.metrics()], ledger=ledger,
        )


def test_checkpoint_resume_matches_uninterrupted_engineering_trajectory() -> None:
    consumer_a, references, weights, images, _ = _prepared_consumer()
    optimizer_a = consumer_a.create_optimizer()
    sampler_a = _sampler()
    trace_a: list[dict[str, object]] = []
    for step in range(3):
        sampler_a.next_batch()
        trace_a.append(
            consumer_a.bogt_training_step(
                images,
                optimizer_a,
                global_step=step,
                data_role=chen.SYNTHETIC_ROLE,
            ).metrics()
        )

    consumer_b = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    consumer_b.bind_references(references)
    consumer_b.bind_weights(weights)
    optimizer_b = consumer_b.create_optimizer()
    sampler_b = _sampler()
    sampler_b.next_batch()
    first = consumer_b.bogt_training_step(
        images,
        optimizer_b,
        global_step=0,
        data_role=chen.SYNTHETIC_ROLE,
    )

    with tempfile.TemporaryDirectory(prefix="chen-bogt-test-") as temporary:
        checkpoint = Path(temporary) / "step_000001.engineering.pt"
        chen.save_checkpoint(
            checkpoint,
            chen.checkpoint_payload(
                consumer_b,
                optimizer_b,
                completed_updates=1,
                sampler_state=sampler_b.state_dict(),
                training_history=[first.metrics()],
                engineering_only=True,
            ),
        )
        consumer_c = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
            device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
        )
        optimizer_c = consumer_c.create_optimizer()
        loaded = chen.load_checkpoint(checkpoint, consumer_c, optimizer_c, expected_engineering_only=True)
        sampler_c = _sampler()
        sampler_c.load_state_dict(loaded.sampler_state)
        trace_c = [first.metrics()]
        for step in range(1, 3):
            sampler_c.next_batch()
            trace_c.append(
                consumer_c.bogt_training_step(
                    images,
                    optimizer_c,
                    global_step=step,
                    data_role=chen.SYNTHETIC_ROLE,
                ).metrics()
            )

        assert sampler_c.cursor == sampler_a.cursor == 3
        for left, right in zip(trace_a, trace_c, strict=True):
            assert left["completed_updates"] == right["completed_updates"]
            assert left["loss"] == pytest.approx(right["loss"], abs=1.0e-7)
        for name, value in consumer_a.trainable_state_dict().items():
            assert torch.allclose(value, consumer_c.trainable_state_dict()[name], rtol=1.0e-5, atol=1.0e-6)

        provenance_payload = chen.checkpoint_payload(
            consumer_b,
            optimizer_b,
            completed_updates=1,
            sampler_state=sampler_b.state_dict(),
            training_history=[first.metrics()],
            engineering_only=True,
        )
        provenance_payload["contracts"]["model"]["sha256"] = "changed-contract-digest"
        provenance_payload["runtime"]["module_sha256"] = "changed-source-digest"
        provenance_payload["initialization"]["state_identity_sha256"] = "changed-theta0-digest"
        provenance_payload["frozen_runtime_state_sha256"] = "changed-frozen-state-digest"
        provenance_payload["references"]["theta0_identity"]["state_identity_sha256"] = "changed-reference-digest"
        provenance_payload["weights"]["references_identity_sha256"] = "changed-reference-binding"
        provenance_checkpoint = Path(temporary) / "provenance-only.engineering.pt"
        torch.save(provenance_payload, provenance_checkpoint)
        consumer_d = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
            device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
        )
        optimizer_d = consumer_d.create_optimizer()
        provenance_loaded = chen.load_checkpoint(
            provenance_checkpoint,
            consumer_d,
            optimizer_d,
            expected_engineering_only=True,
        )
        assert provenance_loaded.completed_updates == 1

        bad_payload = chen.checkpoint_payload(
            consumer_b,
            optimizer_b,
            completed_updates=1,
            sampler_state=sampler_b.state_dict(),
            training_history=[first.metrics()],
            engineering_only=True,
        )
        bad_payload["selection_rule"] = "changed"
        bad_checkpoint = Path(temporary) / "tampered.engineering.pt"
        torch.save(bad_payload, bad_checkpoint)
        with pytest.raises(chen.CheckpointError):
            chen.load_checkpoint(bad_checkpoint, consumer_c, optimizer_c, expected_engineering_only=True)


def test_data_firewall_rejects_non_train_roles_before_read() -> None:
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    ledger = chen.DataReadLedger()
    for role in (chen.DEV_ROLE, chen.PROTECTED_ROLE, chen.TEST_ROLE, chen.SEALED_ROLE):
        with pytest.raises(chen.DataRoleError):
            chen.collect_bogt_train_references(
                consumer,
                _images(),
                data_role=role,
                engineering_only=True,
                ledger=ledger,
            )
    with pytest.raises(chen.DataRoleError):
        chen.collect_bogt_train_references(
            consumer,
            _images(),
            data_role=chen.SYNTHETIC_ROLE,
            engineering_only=False,
            ledger=ledger,
        )
    ledger.assert_forbidden_zero()
    assert all(ledger.counts[role] == 0 for role in (chen.DEV_ROLE, chen.PROTECTED_ROLE, chen.TEST_ROLE, chen.SEALED_ROLE))


def test_phase_fit_is_gauged_wrapped_and_deterministic() -> None:
    z, references, u = _phase_fixture()
    model_a = chen.fit_wrapped_conditional_phase_model(
        z,
        references,
        data_role=chen.SYNTHETIC_ROLE,
        engineering_only=True,
    )
    model_b = chen.fit_wrapped_conditional_phase_model(
        z,
        references,
        data_role=chen.SYNTHETIC_ROLE,
        engineering_only=True,
    )
    assert model_a.coefficients.shape == (101, 100)
    assert torch.allclose(model_a.coefficients, model_b.coefficients)
    assert torch.allclose(model_a.residual_mean, model_b.residual_mean)
    assert torch.allclose(model_a.covariance, model_a.covariance.T, atol=1.0e-12)
    assert torch.isfinite(model_a.cholesky).all()
    assert torch.allclose(model_a.cholesky, torch.tril(model_a.cholesky))
    assert torch.isfinite(model_a.conditional_mean_phase(u[:3])).all()
    wrapped = chen.wrap_phase(torch.tensor([math.pi - 1.0e-5, -math.pi + 1.0e-5], dtype=torch.float64))
    circular_gap = torch.atan2(torch.sin(wrapped[0] - wrapped[1]), torch.cos(wrapped[0] - wrapped[1]))
    assert abs(circular_gap.item()) < 3.0e-5

    gauged, _, indices = chen.max_q_gauge(z, references)
    angle = torch.tensor(0.71)
    rotated, _, rotated_indices = chen.max_q_gauge(
        z * torch.complex(angle.cos(), angle.sin()), references
    )
    assert torch.equal(indices, rotated_indices)
    assert torch.allclose(gauged, rotated, atol=2.0e-6, rtol=2.0e-6)

    tie = torch.ones((1, 100), dtype=torch.complex64)
    _, _, tie_indices = chen.max_q_gauge(tie, references)
    assert tie_indices.item() == 0

    sampled = chen.construct_sampled_ols(
        u[:3],
        references,
        model_a,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )
    assert sampled.coefficients.shape == (3, 100)
    assert sampled.coefficients.dtype == torch.complex64
    assert torch.isfinite(sampled.coefficients).all()
    assert torch.allclose(chen.statistical_variable(sampled.coefficients, references), u[:3], atol=2.0e-5, rtol=2.0e-5)

    stale_digest_model = chen.WrappedConditionalPhaseModel(
        model_a.coefficients,
        model_a.residual_mean,
        model_a.covariance,
        model_a.cholesky,
        "historical-digest-only",
        model_a.fit_data_role,
        model_a.fit_count,
        model_a.engineering_only,
        model_a.data_identity,
    )
    same_sample = chen.construct_sampled_ols(
        u[:3],
        references,
        stale_digest_model,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )
    assert torch.equal(sampled.coefficients, same_sample.coefficients)


def test_real_and_generated_paths_share_the_same_w_ols_writer() -> None:
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    real = consumer.runtime.forward_from_image(_images())
    generated = consumer.generated_forward(real.z_ols)
    assert torch.allclose(
        real.generator_input_field.tensor,
        generated.generator_input_field.tensor,
        atol=1.0e-6,
        rtol=1.0e-5,
    )
    assert real.generator_input_field.provenance.operation == generated.generator_input_field.provenance.operation


def test_scientific_graph_is_static_and_requires_explicit_execution() -> None:
    from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as execution

    with tempfile.TemporaryDirectory(prefix="chen-bogt-graph-") as temporary:
        root = Path(temporary)
        graph = execution.build_scientific_execution_graph(
            repo_root=root,
            output_dir=root / "outputs" / "attempt_01",
        )
    assert graph["task_id"] == execution.NEXT_TASK
    assert graph["scientific_execution_started"] is False
    assert graph["implementation_only"] is False
    assert graph["cli_execution"] == "--execute"
    assert graph["updates"] == chen.SCIENTIFIC_UPDATES
    assert graph["checkpoint_steps"] == list(chen.CHECKPOINT_STEPS)
    assert graph["forbidden_roles"] == ["dev", "protected", "test", "sealed"]
    assert graph["steps"][-1] == "publish write-once generation bundle"
