from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.research.lightgen import diagnose_lightgen_2d_generator_foreground_objective_gradient_v1 as gradient
from scripts.research.lightgen import execute_lightgen_2d_generator_foreground_objective_training_comparison_v1 as training


def _images_and_target() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    target[0, 0, 80, 90] = 0.8
    target[0, 0, 80, 110] = 0.4
    baseline = target.clone()
    baseline[0, 0, 80, 90] = 0.6
    baseline[0, 0, 80, 110] = 0.6
    return baseline, target


def _decision_rows(candidate: torch.Tensor) -> list[dict[str, object]]:
    baseline, target = _images_and_target()
    q_start = float(gradient._qfg_components(baseline, target)["q_fg"][0])
    q_candidate = float(gradient._qfg_components(candidate, target)["q_fg"][0])
    rows: list[dict[str, object]] = []
    for scenario, q_value in (("A4700", q_start), ("A5400", q_start), ("B5400", q_candidate)):
        for label in training.FOCUS_CLASSES:
            rows.extend({"scenario": scenario, "role": "confirmation", "label": label, "q_fg": q_value} for _ in range(30))
    rows.append({"scenario": "B5400", "role": "fit", "label": 2, "q_fg": 1.0e6})
    rows.append({"scenario": "B5400", "role": "confirmation", "label": 0, "q_fg": 1.0e6})
    return rows


def test_real_adam_path_uses_only_the_declared_fixed_qfg_addition(monkeypatch) -> None:
    base = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    reconstruction = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32)
    qfg = torch.tensor([-0.25, 3.0, 2.0], dtype=torch.float32)
    detail = {"metrics": {"raw_mse": 0.1, "ssim_loss": 0.2, "q_fg": 0.3, "l_op": 0.4, "l_dist": 0.0, "E_S_ce": 0.5, "E_I_ce": 0.6, "E_C_ce": 0.7}, "g_q_rms": 1.0}
    monkeypatch.setattr(gradient, "_single_microbatch_batch_gradients", lambda *_args: (base.clone(), reconstruction.clone(), qfg.clone(), copy.deepcopy(detail)))
    images = torch.zeros((training.FIT_ROWS, 1, 28, 28), dtype=torch.float32)
    labels = torch.arange(10).repeat(20)
    inputs = {"generation_inputs": {"images": images}, "labels": labels, "population": {"source_row_indices": list(range(training.TOTAL_ROWS))}}

    def make_optimizer() -> tuple[torch.nn.Parameter, torch.optim.Adam, dict[str, torch.nn.Parameter]]:
        phase = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        encoder = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([phase], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0)
        optimizer.state[phase] = {"step": torch.tensor(float(training.SOURCE_STEP)), "exp_avg": torch.zeros_like(phase), "exp_avg_sq": torch.zeros_like(phase)}
        return phase, optimizer, {"generator.phase": phase, "encoder.phase": encoder}

    captured: list[torch.Tensor] = []
    original_step = torch.optim.Adam.step

    def capture_step(self, *args, **kwargs):
        captured.append(self.param_groups[0]["params"][0].grad.detach().clone())
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", capture_step)
    consumer = SimpleNamespace(device=torch.device("cpu"))
    evaluators = {}
    phase_a, optimizer_a, params_a = make_optimizer()
    record_a = training._train_step(consumer, optimizer_a, params_a, evaluators, inputs, "A", 0)
    gradient_a = captured[-1]
    phase_b, optimizer_b, params_b = make_optimizer()
    record_b = training._train_step(consumer, optimizer_b, params_b, evaluators, inputs, "B", 0)
    gradient_b = captured[-1]

    expected = base + training.semantic.BETA * training.FIXED_LAMBDA * qfg
    assert torch.equal(gradient_a, base)
    assert torch.allclose(gradient_b, expected, rtol=0.0, atol=1e-7)
    assert record_a["optimizer_step"] == record_b["optimizer_step"] == training.SOURCE_STEP + 1
    assert record_b["effective_qfg_lambda"] == training.FIXED_LAMBDA
    assert record_a["effective_qfg_lambda"] == 0.0
    assert not torch.equal(phase_a.detach(), phase_b.detach())
    assert torch.equal(params_a["encoder.phase"].detach(), params_b["encoder.phase"].detach())


def test_zero_weight_control_reproduces_a_arm_adam_update(monkeypatch) -> None:
    base = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    reconstruction = torch.zeros_like(base)
    qfg = torch.tensor([9.0, 8.0, 7.0], dtype=torch.float32)
    detail = {"metrics": {"raw_mse": 0.1, "ssim_loss": 0.2, "q_fg": 0.3, "l_op": 0.4, "l_dist": 0.0, "E_S_ce": 0.5, "E_I_ce": 0.6, "E_C_ce": 0.7}, "g_q_rms": 1.0}
    monkeypatch.setattr(gradient, "_single_microbatch_batch_gradients", lambda *_args: (base.clone(), reconstruction.clone(), qfg.clone(), copy.deepcopy(detail)))
    inputs = {"generation_inputs": {"images": torch.zeros((training.FIT_ROWS, 1, 28, 28))}, "labels": torch.arange(10).repeat(20), "population": {"source_row_indices": list(range(training.TOTAL_ROWS))}}
    consumer = SimpleNamespace(device=torch.device("cpu"))

    def trial(arm: str, weight: float | None):
        phase = torch.nn.Parameter(torch.zeros(3))
        encoder = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([phase], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8)
        optimizer.state[phase] = {"step": torch.tensor(float(training.SOURCE_STEP)), "exp_avg": torch.zeros_like(phase), "exp_avg_sq": torch.zeros_like(phase)}
        params = {"generator.phase": phase, "encoder.phase": encoder}
        record = training._train_step(consumer, optimizer, params, {}, inputs, arm, 0, qfg_lambda=weight)
        return phase.detach().clone(), optimizer.state_dict(), record

    a_phase, a_optimizer, _ = trial("A", None)
    b_phase, b_optimizer, _ = trial("B", 0.0)
    assert torch.equal(a_phase, b_phase)
    assert torch.equal(a_optimizer["state"][0]["exp_avg"], b_optimizer["state"][0]["exp_avg"])
    assert torch.equal(a_optimizer["state"][0]["exp_avg_sq"], b_optimizer["state"][0]["exp_avg_sq"])


def test_actual_qfg_values_drive_improvement_regression_and_tie_decisions() -> None:
    _baseline, target = _images_and_target()
    improved = target.clone()
    improved_decision = training._decision(_decision_rows(improved))
    assert improved_decision["classification"] == "FOREGROUND_TRAINING_RELATIVE_IMPROVEMENT"
    assert improved_decision["difference_B5400_minus_A5400"] < -improved_decision["tolerance"]

    regressed = target.clone()
    regressed[0, 0, 80, 90] = 0.4
    regressed[0, 0, 80, 110] = 0.8
    regression_decision = training._decision(_decision_rows(regressed))
    assert regression_decision["classification"] == "FOREGROUND_TRAINING_RELATIVE_REGRESSION"

    baseline, _ = _images_and_target()
    tie_decision = training._decision(_decision_rows(baseline))
    assert tie_decision["classification"] == "NO_DIFFERENCE_BEYOND_NUMERICAL_TOLERANCE"


def test_invalid_resume_checkpoint_identity_is_rejected(tmp_path) -> None:
    path = tmp_path / "arm_a_step4701.pt"
    path.write_bytes(b"placeholder")
    with pytest.raises(training.TrainingComparisonError, match="another task"):
        training._validate_checkpoint({"task_id": "OTHER"}, path, tmp_path, {})


def test_prior_engineering_failure_is_preserved_before_a_recheck(tmp_path) -> None:
    (tmp_path / "invalid_harness.json").write_text('{"failure":"first"}', encoding="utf-8")
    (tmp_path / "engineering_failure_01.json").write_text('{"failure":"older"}', encoding="utf-8")
    training._preserve_prior_failure(tmp_path)
    assert not (tmp_path / "invalid_harness.json").exists()
    assert (tmp_path / "engineering_failure_02.json").read_text(encoding="utf-8") == '{"failure":"first"}'
