from __future__ import annotations

import pytest
import torch

from scripts.research.lightgen import (
    execute_lightgen_2d_chen_bogt_latent_prior_support_repair_v1 as support_repair,
)


def test_empirical_quantile_batch_interpolates_and_clamps() -> None:
    sorted_values = torch.arange(4, dtype=torch.float64)[:, None].repeat(1, 100)
    probabilities = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    probabilities = probabilities[:, None].repeat(1, 100)

    result = support_repair._empirical_quantile_batch(sorted_values, probabilities)

    expected = torch.tensor([0.0, 0.5, 1.5, 2.5, 3.0], dtype=torch.float64)
    assert torch.equal(result[:, 0], expected)
    assert torch.equal(result, result[:, :1].expand_as(result))


def test_empirical_quantile_batch_rejects_negative_fit_values() -> None:
    sorted_values = torch.arange(4, dtype=torch.float64)[:, None].repeat(1, 100) - 2.0
    probabilities = torch.full((1, 100), 0.5, dtype=torch.float64)

    with pytest.raises(support_repair.RepairError, match="empirical quantile output is negative"):
        support_repair._empirical_quantile_batch(sorted_values, probabilities)
