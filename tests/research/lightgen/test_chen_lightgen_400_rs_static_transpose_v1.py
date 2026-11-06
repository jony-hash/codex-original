from __future__ import annotations

import torch
import pytest

from scripts.research.lightgen import execute_chen_lightgen_400_rs_static_transpose_v1 as task


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.propagator = type("Propagator", (), {
            "operator": torch.nn.Linear(1, 1),
        })()


class _FakeProfiler:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def key_averages(self):
        return self.events


class _DeviceEvent:
    key = "aten::matmul"
    count = 2
    self_device_time_total = 42.5
    device_time_total = 50.0
    self_cpu_time_total = 4.0


class _LegacyEvent:
    key = "aten::matmul"
    count = 2
    self_cpu_time_total = 4.0


def test_corrected_profile_reads_the_supported_device_time_field(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: _FakeProfiler([_DeviceEvent()]))
    trace = task._CorrectedTrace(_FakeModel(), profile=True)
    assert trace._profile_call("forward", lambda *_a, **_k: "ok", None, (), {}) == "ok"
    profile = trace.profile_summaries[0]
    assert profile["status"] == "DEVICE_TIMING_CAPTURED"
    assert profile["device_time_field"] == "self_device_time_total"
    assert profile["operators"][0]["self_device_time_us"] == 42.5


def test_corrected_profile_marks_an_absent_device_field_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: _FakeProfiler([_LegacyEvent()]))
    trace = task._CorrectedTrace(_FakeModel(), profile=True)
    trace._profile_call("forward", lambda *_a, **_k: "ok", None, (), {})
    profile = trace.profile_summaries[0]
    assert profile["status"] == "DEVICE_TIMING_UNAVAILABLE"
    assert profile["device_time_field"] is None
    assert profile["operators"] == []


@pytest.mark.parametrize("variant", ["original", "optimized"])
def test_saved_cost_consumer_rejects_missing_predicate_or_schedule_counts(tmp_path, variant):
    materializations = 90_720
    counts = {
        "rs_forward_calls": 35, "rs_adjoint_calls": 35,
        "linear_propagation_forward_calls": 35, "kernel_cache_hits": 70,
        "kernel_cache_misses": 0, "pair_spectrum_materializations": materializations,
        "center_observation_requests": 0, "full_window_observation_requests": 0,
    }
    counts.update({"legacy_device_pair_predicates": materializations} if variant == "original" else {
        "pair_blocks_with_transpose": 12_000, "pair_blocks_without_transpose": 78_720,
    })
    payload = {
        "schema": "chen_lightgen_400_rs_static_transpose_batch10.v1",
        "task_id": task.TASK_ID, "case": f"{variant}_1", "variant": variant,
        "rows": torch.arange(10), "loss": 0.5,
        "gradients": {"encoder_phase": torch.zeros(2, 1, 400, 400),
                      "decoder_phase": torch.zeros(4, 1, 400, 400)},
        "w_per_pixel": torch.zeros(10, 1, 400, 400),
        "operator_call_counts": counts,
    }
    path = tmp_path / "measurements" / f"{variant}_1.pt"
    path.parent.mkdir()
    torch.save(payload, path)
    loaded = task._load_run_case(tmp_path, f"{variant}_1", torch.arange(10))
    assert loaded["operator_call_counts"] == counts

    if variant == "original":
        del loaded["operator_call_counts"]["legacy_device_pair_predicates"]
    else:
        del loaded["operator_call_counts"]["pair_blocks_without_transpose"]
    torch.save(loaded, path)
    with pytest.raises(task.HarnessError, match="instrumentation is incomplete"):
        task._load_run_case(tmp_path, f"{variant}_1", torch.arange(10))
