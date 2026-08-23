from __future__ import annotations

import json
from argparse import Namespace

import pytest
import torch

from ldm_project.v2.application.research.electronic_o0_latent_transport import (
    LatentTransportFlow,
    _stats,
    _state_equal,
    _transport,
    classify_attribution,
)


def test_realnvp_forward_inverse_and_logdet_are_finite() -> None:
    torch.manual_seed(4)
    flow = LatentTransportFlow()
    codes = torch.randn(30, 8)
    labels = torch.arange(10).repeat_interleave(3)
    transformed = flow(codes, labels)
    recovered, logdet = flow.inverse(transformed, labels)
    assert transformed.shape == (30, 8)
    assert torch.allclose(recovered, codes, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(logdet).all()
    assert sum(parameter.numel() for parameter in flow.parameters()) <= 500_000


def test_classwise_cholesky_transport_is_deterministic_and_preserves_shape() -> None:
    labels = torch.arange(10).repeat_interleave(20)
    codes = torch.randn(200, 8)
    statistics = _stats(codes, labels)
    epsilon = torch.randn(20, 8)
    target_labels = torch.arange(10).repeat_interleave(2)
    first = _transport(epsilon, target_labels, statistics, "Aclass")
    second = _transport(epsilon, target_labels, statistics, "Aclass")
    assert first.shape == (20, 8)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_attribution_classification_requires_both_empirical_upper_bounds() -> None:
    row = {"passed": False}
    payload = {
        "branches": {
            "Aclass": row,
            "Eposterior": {"passed": True},
            "Rencoded": {"passed": True},
        }
    }
    assert classify_attribution(payload)[0] == "nonlinear_latent_transport_candidate"
    payload["branches"]["Rencoded"] = {"passed": False}
    assert classify_attribution(payload)[0] == "decoder_posterior_manifold_failed"


def test_stale_wae_v1_closeout_is_not_a_valid_upstream(tmp_path) -> None:
    from ldm_project.v2.application.research.electronic_o0_latent_transport import _validate_upstream

    path = tmp_path / "closeout.json"
    path.write_text(
        json.dumps({
            "schema": "electronic_o0_wae_closeout.v1",
            "status": "valid",
            "classification": "wae_image_manifold_failed",
        }),
        encoding="utf-8",
    )
    args = Namespace(wae_confirm_evidence=str(path))
    with pytest.raises(ValueError):
        _validate_upstream(args)


def test_state_equal_rejects_changed_flow_parameter() -> None:
    left = {"value": torch.zeros(2)}
    right = {"value": torch.zeros(2)}
    assert _state_equal(left, right)
    right["value"][0] = 1
    assert not _state_equal(left, right)
