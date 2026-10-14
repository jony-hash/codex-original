from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_balanced_reconstruction_v1 as task


def _toy_target() -> torch.Tensor:
    target = torch.zeros((2, 1, 200, 200), dtype=torch.float32)
    target[0, :, :, :50] = 0.5
    target[1, 0, 0, 0] = 0.2
    return target


def test_balanced_loss_weights_each_image_foreground_and_background_equally() -> None:
    target = _toy_target()
    raw = torch.zeros_like(target, requires_grad=True)
    with torch.no_grad():
        raw[0, :, :, :50] = 1.5
        raw[1].fill_(2.0)
        raw[1, 0, 0, 0] = 3.2

    loss = task.balanced_reconstruction_loss(raw, target)
    assert loss.item() == pytest.approx(3.5)
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[0, :, :, :50].abs().sum() > 0


@pytest.mark.parametrize("target", [torch.zeros((1, 1, 200, 200)), torch.ones((1, 1, 200, 200))])
def test_balanced_loss_rejects_empty_foreground_or_background(target: torch.Tensor) -> None:
    with pytest.raises(task.ReconstructionRunError):
        task.balanced_reconstruction_loss(torch.zeros_like(target), target)


def test_training_batch_selects_only_fixed_fit_rows_and_rejects_other_roles() -> None:
    images = torch.arange(300).reshape(300, 1, 1, 1)
    labels = torch.arange(300) % 10
    batch, batch_labels, positions = task.training_batch(images, labels, 19)
    assert positions.tolist() == list(range(190, 200))
    assert batch.flatten().tolist() == list(range(190, 200))
    assert batch_labels.tolist() == list(range(10))
    with pytest.raises(task.ReconstructionRunError, match="canonical TRAIN"):
        task.training_batch(images, labels, 0, data_role="dev")
    with pytest.raises(task.ReconstructionRunError):
        task.training_batch(images[:200], labels[:200], 0)


def test_generator_only_adam_update_leaves_encoder_parameters_and_moments_frozen() -> None:
    encoder = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    generator = torch.nn.Parameter(torch.tensor([0.75, 0.1]))
    parameters = {"encoder.phase": encoder, "generator.phase": generator}
    optimizer = torch.optim.Adam([encoder, generator], lr=1e-3)
    encoder_before = encoder.detach().clone()
    optimizer.state[encoder] = {"step": torch.tensor(2200.0), "exp_avg": torch.tensor([0.1, 0.2]), "exp_avg_sq": torch.tensor([0.3, 0.4])}
    optimizer.state[generator] = {"step": torch.tensor(2200.0), "exp_avg": torch.tensor([0.1, 0.2]), "exp_avg_sq": torch.tensor([0.3, 0.4])}
    encoder_moments = task._cpu_tree(optimizer.state[encoder])
    generator_before = generator.detach().clone()

    task.apply_generator_only_adam_update(optimizer, parameters, torch.tensor([0.5, -0.25]))

    assert torch.equal(encoder, encoder_before)
    assert task._tree_equal(task._cpu_tree(optimizer.state[encoder]), encoder_moments)
    assert torch.equal(optimizer.state[encoder]["step"], torch.tensor(2200.0))
    assert float(optimizer.state[generator]["step"]) == 2201.0
    assert not torch.equal(generator, generator_before)
    assert encoder.grad is None and generator.grad is None


def test_joint_semantic_gradient_keeps_current_three_evaluator_normalization() -> None:
    losses = {
        "L_S": {"encoder.phase": torch.tensor([3.0]), "generator.phase": torch.tensor([0.0])},
        "L_I": {"encoder.phase": torch.tensor([0.0]), "generator.phase": torch.tensor([4.0])},
        "L_C": {"encoder.phase": torch.tensor([3.0]), "generator.phase": torch.tensor([4.0])},
    }
    normalized, reason = diagnostic._safe_canonical_gradient(losses)
    assert reason is None
    assert normalized is not None
    # Unit vectors are summed, then jointly RMS-normalized and scaled by delta_SIC.
    rms = ((1.6**2 + 1.8**2) / 2.0) ** 0.5
    assert normalized["encoder.phase"].item() == pytest.approx(1.6 / rms * task.DELTA_SIC)
    assert normalized["generator.phase"].item() == pytest.approx(1.8 / rms * task.DELTA_SIC)


def test_raw_cmos_and_clamped_evaluator_view_remain_distinct() -> None:
    raw = torch.full((1, 1, 200, 200), 2.0)
    view = canonical.cmos_view(raw)
    assert raw.max().item() == 2.0
    assert view.shape == (1, 1, 28, 28)
    assert view.max().item() == 1.0


def test_resource_telemetry_uses_this_task_identity_and_retains_separate_memory_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        task.diagnostic,
        "_resource_snapshot",
        lambda device: {"device": str(device), "global_used_fraction": 0.4, "pytorch_allocated_bytes": 12, "pytorch_reserved_bytes": 34},
    )
    sample = task._sample_resource_telemetry(tmp_path, torch.device("cpu"), "test_phase", progress={"step": 100})
    saved = json.loads((tmp_path / task.RESOURCE_TELEMETRY_NAME).read_text(encoding="utf-8"))
    assert sample["task_id"] == task.TASK_ID
    assert saved["task_id"] == task.TASK_ID
    assert saved["resource"]["pytorch_allocated_bytes"] == 12
    assert saved["resource"]["pytorch_reserved_bytes"] == 34


def test_a_a_cuda_repeatability_allows_only_declared_float32_roundoff() -> None:
    equal, detail = task._repeatability_difference(
        {"generator.phase": torch.tensor([0.3])},
        {"generator.phase": torch.tensor([0.3 + 2e-7])},
    )
    assert equal
    assert detail["atol"] == 5e-7
    unequal, _ = task._repeatability_difference(
        {"generator.phase": torch.tensor([0.3])},
        {"generator.phase": torch.tensor([0.3 + 2e-5])},
    )
    assert not unequal


def _fake_endpoint() -> tuple[Any, dict[str, torch.nn.Parameter], torch.optim.Adam]:
    encoder = torch.nn.Parameter(torch.tensor([0.3]))
    generator = torch.nn.Parameter(torch.tensor([0.7]))
    parameters = {"encoder.phase": encoder, "generator.phase": generator}
    optimizer = torch.optim.Adam([encoder, generator], lr=1e-3)
    for parameter in (encoder, generator):
        optimizer.state[parameter] = {
            "step": torch.tensor(2200.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }

    class Consumer:
        device = torch.device("cpu")

        def forward_loss(self, images: torch.Tensor, *, data_role: str) -> Any:
            assert data_role == "train"
            target = F.interpolate(images, size=(200, 200), mode="nearest")
            raw = target * generator.reshape(1, 1, 1, 1) + 0.05
            result = SimpleNamespace(raw_intensity=raw, image_input=SimpleNamespace(slm_intensity=target))
            zero = raw.mean() * 0.0
            return SimpleNamespace(L_dist=zero, L_op=zero, result=result)

    return Consumer(), parameters, optimizer


def _fake_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.zeros((300, 1, 28, 28), dtype=torch.float32)
    for row in range(300):
        images[row, 0, 4:20, 4 + (row % 12) : 8 + (row % 12)] = 0.6
    labels = torch.arange(300) % 10
    return images, labels


def _patch_training_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    def losses(bundle: dict[str, Any], images: torch.Tensor, smooth: Any, independent: Any, ec: Any) -> dict[str, torch.Tensor]:
        raw_mean = bundle["result"].raw_intensity.mean()
        return {"L_S": raw_mean, "L_I": 0.5 * raw_mean, "L_C": 0.25 * raw_mean}

    def canonical_direction(gradients: dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], None]:
        return {name: torch.zeros_like(value) for name, value in gradients["L_S"].items()}, None

    monkeypatch.setattr(task.diagnostic, "_loss_reductions", losses)
    monkeypatch.setattr(task.diagnostic, "_safe_canonical_gradient", canonical_direction)


def _train_fake_step(monkeypatch: pytest.MonkeyPatch, endpoint: tuple[Any, dict[str, torch.nn.Parameter], torch.optim.Adam], images: torch.Tensor, labels: torch.Tensor, update: int, arm: str) -> dict[str, Any]:
    consumer, parameters, optimizer = endpoint
    return task._train_step(
        consumer,
        optimizer,
        parameters,
        {"E_S": object(), "E_I": object(), "E_C": object()},
        images,
        labels,
        update,
        arm=arm,
    )


def test_arm_updates_are_isolated_and_resume_matches_continuous_training(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_training_directions(monkeypatch)
    images, labels = _fake_inputs()
    start_generator = torch.tensor([0.7])
    start_encoder = torch.tensor([0.3])

    arm_a = _fake_endpoint()
    arm_b = _fake_endpoint()
    a_record = _train_fake_step(monkeypatch, arm_a, images, labels, 0, "A_original")
    b_record = _train_fake_step(monkeypatch, arm_b, images, labels, 0, "B_balanced")
    assert torch.equal(arm_a[1]["encoder.phase"].detach(), start_encoder)
    assert torch.equal(arm_b[1]["encoder.phase"].detach(), start_encoder)
    assert a_record["training_reconstruction_loss"] != pytest.approx(b_record["training_reconstruction_loss"])
    assert arm_a[1]["generator.phase"].data_ptr() != arm_b[1]["generator.phase"].data_ptr()
    assert arm_a[1]["generator.phase"].item() != start_generator.item()
    assert arm_b[1]["generator.phase"].item() != start_generator.item()

    continuous = _fake_endpoint()
    _train_fake_step(monkeypatch, continuous, images, labels, 0, "A_original")
    _train_fake_step(monkeypatch, continuous, images, labels, 1, "A_original")

    interrupted = _fake_endpoint()
    _train_fake_step(monkeypatch, interrupted, images, labels, 0, "A_original")
    checkpoint_parameters = {name: value.detach().clone() for name, value in interrupted[1].items()}
    checkpoint_optimizer = copy.deepcopy(interrupted[2].state_dict())
    resumed = _fake_endpoint()
    with torch.no_grad():
        for name, parameter in resumed[1].items():
            parameter.copy_(checkpoint_parameters[name])
    resumed[2].load_state_dict(checkpoint_optimizer)
    _train_fake_step(monkeypatch, resumed, images, labels, 1, "A_original")
    assert all(torch.equal(continuous[1][name], resumed[1][name]) for name in continuous[1])
    assert task._tree_equal(continuous[2].state_dict(), resumed[2].state_dict())


def _metrics(macro: float, counts: list[int]) -> dict[str, Any]:
    return {"macro_recall": macro, "per_class_correct": counts}


def _image_stats(overall: float, foreground: float, clipping: float = 0.0) -> dict[str, Any]:
    by_class = {}
    for class_id in range(10):
        by_class[str(class_id)] = {
            "observation": {"raw_foreground_mse": foreground if class_id in (1, 7) else 1.0, "raw_fraction_above_one": clipping},
            "fit": {"raw_foreground_mse": foreground, "raw_fraction_above_one": clipping},
            "combined": {"raw_foreground_mse": foreground, "raw_fraction_above_one": clipping},
        }
    return {"overall": {"observation": {"raw_mse": overall}, "fit": {"raw_mse": overall}, "combined": {"raw_mse": overall}}, "by_class": by_class}


def test_decision_support_and_evidence_insufficient_exit() -> None:
    target = {"observation": _metrics(0.95, [10, 9, 9, 9, 9, 9, 9, 9, 9, 9])}
    equal_endpoints = {
        "step2200": {"observation": _metrics(0.5, [5] * 10)},
        "A_original": {"observation": _metrics(0.5, [5] * 10)},
        "B_balanced": {"observation": _metrics(0.5, [5] * 10)},
    }
    equal_images = {name: _image_stats(0.1, 1.0) for name in equal_endpoints}
    equal_tests = {"holm_adjusted_p": {"1": 1.0, "7": 1.0}}
    unsupported = task._decision(target, equal_endpoints, equal_images, equal_tests)
    assert unsupported["classification"] == "NOT_SUPPORTED_WITHIN_SCOPE"
    assert unsupported["training_recommendation"] == "NONE"

    improved_endpoints = copy.deepcopy(equal_endpoints)
    improved_endpoints["B_balanced"]["observation"] = _metrics(0.6, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
    improved_images = {
        "step2200": _image_stats(0.1, 1.0),
        "A_original": _image_stats(0.1, 1.0),
        "B_balanced": _image_stats(0.1, 0.8, 0.001),
    }
    supported = task._decision(target, improved_endpoints, improved_images, {"holm_adjusted_p": {"1": 0.02, "7": 0.03}})
    assert supported["classification"] == "BALANCED_RECONSTRUCTION_SUPPORTED"
    assert "Generator-only" in supported["training_recommendation"]

    unqualified = task._decision({"observation": _metrics(0.89, [10] * 10)}, improved_endpoints, improved_images, {"holm_adjusted_p": {"1": 0.02, "7": 0.03}})
    assert unqualified["classification"] is None
    assert unqualified["training_recommendation"] == "NONE"


def test_exact_mcnemar_and_holm_treat_identical_arms_as_no_evidence() -> None:
    identical = torch.tensor([True] * 10)
    comparison = task._exact_mcnemar_better(identical, identical)
    assert comparison["discordant_pairs"] == 0
    assert comparison["one_sided_exact_p"] == 1.0
    assert task._holm({"1": 1.0, "7": 1.0}) == {"1": 1.0, "7": 1.0}


def test_resume_requires_explicit_current_checkpoint_and_completed_resume_is_noop(tmp_path: Path) -> None:
    population = {"rows": [1, 2, 3]}
    payload = {
        "schema": "m0_generator_balanced_reconstruction_checkpoint.v1",
        "task_id": task.TASK_ID,
        "arm": "B_balanced",
        "completed_updates": 1000,
        "next_batch_index": 0,
        "source_checkpoint_step": 2200,
        "population": population,
    }
    final_checkpoint = tmp_path / "arm_b_step1000.pt"
    torch.save(payload, final_checkpoint)
    loaded = task._read_checkpoint(final_checkpoint, tmp_path, population)
    report = {"task_id": task.TASK_ID, "status": "VALID_COMPLETED", "updates": 2000}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    assert task._completed_resume_result(tmp_path, loaded) == report
    assert len(list(tmp_path.glob("arm_*_step*.pt"))) == 1


def test_explicit_older_checkpoint_is_rejected_when_later_artifact_exists(tmp_path: Path) -> None:
    population = {"rows": [1, 2, 3]}
    early = {
        "schema": "m0_generator_balanced_reconstruction_checkpoint.v1",
        "task_id": task.TASK_ID,
        "arm": "A_original",
        "completed_updates": 100,
        "next_batch_index": 0,
        "source_checkpoint_step": 2200,
        "population": population,
    }
    torch.save(early, tmp_path / "arm_a_step0100.pt")
    later = dict(early, completed_updates=200)
    torch.save(later, tmp_path / "arm_a_step0200.pt")
    with pytest.raises(task.ReconstructionRunError, match="older than already saved"):
        task._read_checkpoint(tmp_path / "arm_a_step0100.pt", tmp_path, population)
@pytest.mark.parametrize("relative", ["outputs/final_pt/probe", "outputs/_wsl_restore_metadata_v4/probe", "outputs/runs/probe", "outputs/20261006_v2/probe", "outputs/20261006_v2/runs"])
def test_output_rejects_protected_and_non_run_paths(tmp_path: Path, relative: str) -> None:
    target = tmp_path / relative
    with pytest.raises(RuntimeError):
        task._prepare_output(tmp_path, target, resume=False)
    assert not target.exists()


def test_output_accepts_named_versioned_run(tmp_path: Path) -> None:
    target = tmp_path / "outputs/20261006_v2/runs/diagnostic-only_probe"
    task._prepare_output(tmp_path, target, resume=False)
    task._prepare_output(tmp_path, target, resume=True)
    assert target.is_dir()
