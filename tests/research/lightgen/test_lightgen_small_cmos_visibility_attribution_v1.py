from __future__ import annotations

import math
from functools import partial

import numpy as np
import pytest
import torch

from scripts.research.lightgen import execute_lightgen_small_cmos_visibility_attribution_v1 as attribution


def _numpy_clamp_pool(image: np.ndarray, gain: float) -> np.ndarray:
    scaled = np.clip(image * gain, 0.0, 1.0)
    pooled = np.empty((28, 28), dtype=np.float64)
    for y in range(28):
        y0 = math.floor(y * 200 / 28)
        y1 = math.ceil((y + 1) * 200 / 28)
        for x in range(28):
            x0 = math.floor(x * 200 / 28)
            x1 = math.ceil((x + 1) * 200 / 28)
            pooled[y, x] = scaled[y0:y1, x0:x1].mean()
    return pooled


def test_production_readout_matches_independent_numpy_and_rejects_wrong_order() -> None:
    raw = torch.linspace(0.0, 2.0, 40_000, dtype=torch.float32).reshape(1, 1, 200, 200)
    gain = torch.tensor(1.0)
    multiplier = 0.85
    produced = attribution._production_view(raw, gain, multiplier)[0, 0].numpy()
    expected = _numpy_clamp_pool(raw[0, 0].numpy(), multiplier)
    np.testing.assert_allclose(produced, expected, rtol=2e-6, atol=5e-8)

    pooled_then_clamped = torch.nn.functional.adaptive_avg_pool2d(raw * multiplier, (28, 28)).clamp(0, 1)
    clipped_before_scale = attribution._production_view(raw.clamp(0, 1), gain, multiplier)
    assert not np.allclose(produced, pooled_then_clamped[0, 0].numpy(), rtol=1e-5, atol=1e-6)
    assert not np.allclose(produced, clipped_before_scale[0, 0].numpy(), rtol=1e-5, atol=1e-6)


def test_evaluator_counter_hook_preserves_model_output() -> None:
    evaluator = torch.nn.Identity()
    calls = {"samples": 0, "forward_calls": 0}
    evaluator.register_forward_hook(partial(attribution._count_evaluator_forward, calls=calls))
    values = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    result = evaluator(values)
    assert torch.equal(result, values)
    assert calls == {"samples": 3, "forward_calls": 1}


def _valid_block() -> tuple[dict[str, object], dict[str, object], torch.Tensor, torch.Tensor]:
    expected = {
        "schema": "lightgen_small_cmos_visibility_block.v1",
        "task_id": attribution.TASK_ID,
        "cell": "AA",
        "input_source": "A",
        "decoder": "A",
        "class_id": 0,
        "sample_count": attribution.SAMPLES_PER_CLASS,
        "input_run_id": "run-a",
        "decoder_run_id": "run-a",
        "conditions": ["AA_g0p85", "AA_g1p00", "AA_g1p15"],
    }
    labels = torch.zeros(attribution.SAMPLES_PER_CLASS, dtype=torch.long)
    proposals = torch.arange(attribution.SAMPLES_PER_CLASS)
    variants: dict[str, dict[str, torch.Tensor]] = {}
    for name in expected["conditions"]:
        variants[name] = {
            "images": torch.zeros((attribution.SAMPLES_PER_CLASS, 1, 28, 28)),
            "logits": torch.zeros((attribution.SAMPLES_PER_CLASS, 10)),
            "predicted_labels": torch.zeros(attribution.SAMPLES_PER_CLASS, dtype=torch.long),
            "identity_margin": torch.zeros(attribution.SAMPLES_PER_CLASS),
            "view_mean": torch.zeros(attribution.SAMPLES_PER_CLASS),
            "camera_total_power_w": torch.ones(attribution.SAMPLES_PER_CLASS),
            "peak_w_per_pixel": torch.ones(attribution.SAMPLES_PER_CLASS),
            "scaled_peak": torch.ones(attribution.SAMPLES_PER_CLASS),
            "clipping_area_fraction": torch.zeros(attribution.SAMPLES_PER_CLASS),
            "clipping_region_power_share": torch.zeros(attribution.SAMPLES_PER_CLASS),
            "excess_removed_power_share": torch.zeros(attribution.SAMPLES_PER_CLASS),
        }
    payload: dict[str, object] = {
        "identity": expected.copy(),
        "requested_labels": labels.clone(),
        "proposal_ids": proposals.clone(),
        "conditions": variants,
        "native_raw_examples": torch.zeros((10, 1, 200, 200)),
        "native_replay_check": {
            "passed": True,
            "max_abs_view_difference": 0.0,
            "max_abs_view_scale": 1.0,
            "max_abs_clipping_fraction_difference": 0.0,
        },
        "timing_seconds": {"optical_seconds": 1.0, "evaluator_seconds": 1.0},
    }
    return payload, expected, labels, proposals


def test_block_consumer_accepts_complete_zero_output_with_null_power_ratios() -> None:
    payload, expected, labels, proposals = _valid_block()
    for condition in payload["conditions"].values():
        condition["camera_total_power_w"].zero_()
        condition["peak_w_per_pixel"].zero_()
        condition["scaled_peak"].zero_()
        condition["clipping_region_power_share"].fill_(float("nan"))
        condition["excess_removed_power_share"].fill_(float("nan"))
    attribution._validate_block(payload, expected, labels, proposals)


def test_synchronized_equal_class_bootstrap_consumes_all_fixed_conditions() -> None:
    labels = torch.arange(10).repeat_interleave(attribution.SAMPLES_PER_CLASS)
    names = [
        "AA_g0p85", "AA_g1p00", "AA_g1p15", "BA_g1p00",
        "AB_g1p00", "BB_g0p85", "BB_g1p00", "BB_g1p15",
    ]
    values = {
        name: {
            "predicted_labels": labels.clone(),
            "identity_margin": torch.ones(attribution.COHORT_SIZE),
            "clipping_area_fraction": torch.zeros(attribution.COHORT_SIZE),
        }
        for name in names
    }
    result = attribution._paired_bootstrap(values, labels)
    assert result["resamples"] == 2_000
    assert result["seed"] == 2026102101
    for contrast in result["contrasts"].values():
        for metric in contrast.values():
            assert metric["estimate"] == 0.0
            assert metric["ci_95"] == [0.0, 0.0]


@pytest.mark.parametrize("corruption", ["decoder", "class", "duplicate_proposal", "missing_condition", "nonfinite"])
def test_block_consumer_rejects_wrong_identity_rows_or_incomplete_values(corruption: str) -> None:
    payload, expected, labels, proposals = _valid_block()
    if corruption == "decoder":
        payload["identity"]["decoder_run_id"] = "wrong-decoder"
    elif corruption == "class":
        payload["requested_labels"][0] = 1
    elif corruption == "duplicate_proposal":
        payload["proposal_ids"][1] = 0
    elif corruption == "missing_condition":
        del payload["conditions"]["AA_g0p85"]
    elif corruption == "nonfinite":
        payload["conditions"]["AA_g1p00"]["logits"][0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        attribution._validate_block(payload, expected, labels, proposals)
