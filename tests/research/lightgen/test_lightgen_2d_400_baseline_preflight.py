"""CPU checks of the baseline's preflight and receipt boundary, not CUDA acceptance."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.research.lightgen import execute_lightgen_2d_400_mnist_semantic_generation_baseline_v1 as baseline
from ldm_project.v2.application.research.shared.research_run_control import RunControlError


class _BoundaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))

    def command(self, amplitude):
        return amplitude

    def forward(self, amplitude):
        raw = amplitude.square() * self.gain.sigmoid()
        return SimpleNamespace(raw_camera=SimpleNamespace(tensor=raw), ols=SimpleNamespace(coefficients=amplitude))

    def forward_from_ols(self, coefficients):
        return self(coefficients)


class _FrozenClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.centers = nn.Parameter(torch.arange(10).float() / 9, requires_grad=False)

    def forward(self, image):
        brightness = image.mean(dim=(1, 2, 3)).unsqueeze(1)
        return brightness, -100 * (brightness - self.centers).square()


@pytest.mark.parametrize("damage_evaluator", [False, True])
def test_baseline_preflight_computes_checks_and_rolls_back_without_memory_quota(monkeypatch, damage_evaluator):
    model = _BoundaryModel()
    evaluator = _FrozenClassifier()
    images = (torch.arange(10).float() / 9).reshape(10, 1, 1, 1).expand(-1, 1, 28, 28)
    labels = torch.arange(10)
    rows = labels.expand(3, -1)
    model_before = model.gain.detach().clone()
    evaluator_before = evaluator.centers.detach().clone()
    rng_before = torch.random.get_rng_state().clone()
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(baseline, "_memory_fraction", lambda _device: 0.99)
    real_step = baseline._step
    calls = []

    def update(*args, **kwargs):
        result = real_step(*args, **kwargs)
        calls.append(result)
        torch.rand(1)
        if damage_evaluator:
            with torch.no_grad():
                evaluator.centers.add_(0.001)
        return result

    monkeypatch.setattr(baseline, "_step", update)
    if damage_evaluator:
        with pytest.raises(RuntimeError, match="training evaluator changed"):
            baseline._preflight(model, images, labels, rows, evaluator, torch.device("cpu"))
    else:
        result = baseline._preflight(model, images, labels, rows, evaluator, torch.device("cpu"))
        assert all(result[key] is True for key in ("three_updates", "negative_control", "frozen_evaluator", "model_restored", "rng_restored"))
        assert result["peak_vram_fraction"] == 0.99
    assert len(calls) == 3
    assert torch.equal(model.gain, model_before)
    assert torch.equal(evaluator.centers, evaluator_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("negative_control_passed", [True, False])
def test_baseline_entry_consumes_measured_flags_before_opening_attempt(monkeypatch, tmp_path, negative_control_passed):
    monkeypatch.setattr(baseline, "RUN_DIR", tmp_path / "attempt")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_args: None)
    config = SimpleNamespace(system=SimpleNamespace(kind="diagnostic_lightgen_amplitude_fiber_ols_400"))
    monkeypatch.setattr(baseline, "load_config", lambda *_args: config)
    monkeypatch.setattr(baseline, "_load_inputs", lambda *_args: (None,) * 6)
    monkeypatch.setattr(baseline, "_balanced_rows", lambda *_args: None)
    model = SimpleNamespace(train=lambda: None)
    model.to = lambda _device: model
    monkeypatch.setattr(baseline, "LightGenAmplitudeFiberOLSSystemV2", lambda *_args: model)
    manifest = SimpleNamespace(execution_source_semantics_id="unit-provenance", source_manifest_id="unit-manifest")
    monkeypatch.setattr(baseline, "capture_execution_source_manifest", lambda *_args, **_kwargs: manifest)
    preflight = {key: True for key in ("three_updates", "negative_control", "frozen_evaluator", "model_restored", "rng_restored")}
    preflight.update(negative_control=negative_control_passed, peak_vram_fraction=0.99, estimated_execution_seconds=100000)
    monkeypatch.setattr(baseline, "_preflight", lambda *_args: preflight)
    opened = []

    class StopAtAttempt(Exception):
        pass

    def capture_open(_self, *_args, receipt, **_kwargs):
        opened.append(receipt)
        raise StopAtAttempt

    monkeypatch.setattr(baseline.ResearchRunControl, "open_scientific_attempt", capture_open)
    if negative_control_passed:
        with pytest.raises(StopAtAttempt):
            baseline.main()
        assert opened[0].checks[baseline.CORRECTNESS_INVARIANTS]["negative_control"] is True
    else:
        with pytest.raises(RunControlError, match="did not pass"):
            baseline.main()
        assert not opened
    assert not baseline.RUN_DIR.exists()
