from __future__ import annotations

import math

import pytest
import torch

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_frozen_joint_prior_comparison_v1 as task,
)


def _fit_bank() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(731)
    values = torch.randn((200, 100), generator=generator, dtype=torch.float64)
    # Give ports 1 and 2 a strong shared component so dropping cross-port
    # covariance has an independently checkable effect.
    values[:, 1] = values[:, 0] * 0.8 + values[:, 1] * 0.2
    z = torch.complex(values[:, 0::1][:, :100], torch.randn((200, 100), generator=generator, dtype=torch.float64))
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(20)
    return z, labels


def _primary_rows(block_mean: float, joint_mean: float) -> list[dict[str, object]]:
    rows = []
    for seed in task.SEEDS:
        for digit in range(10):
            rows.append({"arm": "block", "seed": seed, "label": digit, "target_role": "confirmation", "shape_size": 28, "energy_distance": block_mean})
            rows.append({"arm": "joint", "seed": seed, "label": digit, "target_role": "confirmation", "shape_size": 28, "energy_distance": joint_mean})
    return rows


def test_fit_priors_preserve_each_port_block_and_retain_cross_port_term() -> None:
    z, labels = _fit_bank()
    priors = task._fit_class_priors(z, labels)
    assert priors["reconstruction_error"].max() <= 1e-12
    for digit in range(10):
        for port in range(task.PORTS):
            pair = slice(2 * port, 2 * port + 2)
            assert torch.equal(priors["block_correlation"][digit, pair, pair], priors["joint_correlation"][digit, pair, pair])
    assert priors["block_correlation"][0, 0, 2] == 0
    assert abs(float(priors["joint_correlation"][0, 0, 2])) > 0.1
    banks, noise = task._sample_banks(priors)
    assert banks["block"].shape == banks["joint"].shape == (5, 10, 30, 100)
    assert noise.shape == (5, 10, 30, 200)
    assert not torch.equal(banks["block"], banks["joint"])


def test_identical_marginals_without_cross_port_terms_give_identical_paired_draws() -> None:
    eye = torch.eye(task.REAL_DIM, dtype=torch.float64).expand(10, -1, -1).clone()
    priors = {
        "means": torch.zeros((10, task.REAL_DIM), dtype=torch.float64),
        "scales": torch.ones((10, task.REAL_DIM), dtype=torch.float64),
        "block_cholesky": eye.clone(),
        "joint_cholesky": eye.clone(),
    }
    banks, noise = task._sample_banks(priors)
    assert torch.equal(banks["block"], banks["joint"])
    assert torch.equal(noise[0], noise[0])


def test_covariance_fitting_rejects_nonfinite_and_zero_variance_coefficients() -> None:
    z, labels = _fit_bank()
    nonfinite = z.clone()
    nonfinite[0, 0] = complex(float("nan"), 0.0)
    with pytest.raises(task.PriorComparisonError, match="nonfinite"):
        task._fit_class_priors(nonfinite, labels)
    constant = z.clone()
    constant[labels == 0, 0] = 1.0 + 0.0j
    with pytest.raises(task.PriorComparisonError, match="nonpositive"):
        task._fit_class_priors(constant, labels)


def test_interleaving_accepts_sampling_draw_population_and_preserves_coordinate_order() -> None:
    rows = torch.tensor([[1.0 + 2.0j] * task.PORTS] * 30, dtype=torch.complex128)
    values = task._to_interleaved(rows)
    assert values.shape == (30, 200)
    assert torch.equal(values[0, :4], torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float64))


def test_saved_readout_validator_checks_row_identity_and_production_view() -> None:
    rows = []
    for flat in range(task.CHUNK_ROWS):
        digit, draw = divmod(flat, task.SAMPLE_COUNT_PER_CLASS)
        rows.append({
            "arm": "block", "seed": task.SEEDS[0], "label": digit, "draw_index": draw,
            "counterfactual_pair_id": f"{task.SEEDS[0]}:{digit}:{draw}",
        })
    raw = torch.ones((task.CHUNK_ROWS, 1, 200, 200), dtype=torch.float32)
    payload = {
        "task_id": task.TASK_ID, "arm": "block", "seed": task.SEEDS[0],
        "flat_start": 0, "rows": rows, "raw_cmos": raw,
        "evaluator_view": task.generation.capacity.canonical.cmos_view(raw),
        "e_d_logits": torch.zeros((task.CHUNK_ROWS, 10)),
        "generator_input_power": torch.ones(task.CHUNK_ROWS),
        "camera_integrated_intensity": torch.ones(task.CHUNK_ROWS),
        "raw_clip_fraction": torch.zeros(task.CHUNK_ROWS),
    }
    task._validate_chunk_payload(payload, "block", task.SEEDS[0], 0)
    wrong_order = {**payload, "rows": [dict(row) for row in rows]}
    wrong_order["rows"][0]["label"] = 1
    with pytest.raises(task.PriorComparisonError, match="row mapping"):
        task._validate_chunk_payload(wrong_order, "block", task.SEEDS[0], 0)
    wrong_view = {**payload, "evaluator_view": payload["evaluator_view"] + 0.01}
    with pytest.raises(task.PriorComparisonError, match="production view"):
        task._validate_chunk_payload(wrong_view, "block", task.SEEDS[0], 0)


def test_energy_distance_detects_same_mean_shape_collapse_and_scale_invariant_view() -> None:
    distributed = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    assert torch.equal(distributed.mean(0), collapsed.mean(0))
    assert task.frozen._energy_distance(distributed, distributed) == 0.0
    assert task.frozen._energy_distance(collapsed, distributed) > 0.0
    raw = torch.rand((2, 1, 200, 200), generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    scaled = raw * 0.37
    assert torch.allclose(task.frozen._shape_vectors(raw, 28), task.frozen._shape_vectors(scaled, 28), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("joint", "expected"),
    [
        (0.9, "JOINT_PRIOR_NUMERIC_IMPROVEMENT"),
        (1.1, "JOINT_PRIOR_NUMERIC_DETERIORATION"),
        (1.0 + 0.5 * (task.TOLERANCE_ATOL + task.TOLERANCE_RTOL), "NO_GAIN_BEYOND_NUMERICAL_TOLERANCE"),
    ],
)
def test_final_decision_consumes_class_and_seed_shape_distances(joint: float, expected: str) -> None:
    decision = task._main_decision(_primary_rows(1.0, joint))
    assert decision["classification"] == expected
    assert math.isfinite(decision["primary_effect_joint_minus_block"])
