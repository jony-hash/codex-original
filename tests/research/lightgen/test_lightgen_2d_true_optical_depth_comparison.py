"""Focused contract tests for the N5 execution boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[3] / "scripts/research/lightgen/execute_lightgen_2d_true_optical_depth_comparison_v1.py"


def _load_execution():
    spec = importlib.util.spec_from_file_location("test_n5_true_depth_execution", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_continuation_sampler_starts_at_offset_and_stops_at_terminal_cursor() -> None:
    execution = _load_execution()
    train_indices = torch.arange(55_000, dtype=torch.long)
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    sampler = execution.ContinuationSampler(train_indices, labels)
    assert sampler.cursor == 300
    first_indices, first_labels = sampler.next_batch()
    assert first_indices.shape == (10,)
    assert first_labels.equal(torch.arange(10, dtype=torch.long))
    for _ in range(299):
        sampler.next_batch()
    assert sampler.cursor == 600
    with pytest.raises(StopIteration):
        sampler.next_batch()


def test_true_depth_parameter_surface_is_fixed_and_neutral() -> None:
    execution = _load_execution()
    model = execution.TrueDepthGenerator(torch.zeros((2, 1, 100, 100)), torch.device("cpu"))
    assert list(model.state_dict()) == ["phase", "depth_phase"]
    assert list(model.named_parameters())[0][0] == "phase"
    assert list(model.named_parameters())[1][0] == "depth_phase"
    assert int(sum(value.numel() for value in model.parameters())) == 30_000
    assert torch.equal(model.depth_phase, torch.zeros_like(model.depth_phase))
