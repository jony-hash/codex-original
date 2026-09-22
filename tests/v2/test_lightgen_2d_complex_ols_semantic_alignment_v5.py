"""Focused mathematical and gate tests for the V5 preflight core."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/research/execute_lightgen_2d_complex_ols_semantic_alignment_preflight_v5.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("lightgen_v5_preflight_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gauge_compatible_prototype_distance_and_loss_are_differentiable() -> None:
    entrypoint = _load_entrypoint()
    torch.manual_seed(7)
    real = torch.randn(10, 100)
    imag = torch.randn(10, 100)
    query = torch.complex(real, imag).requires_grad_()
    prototypes = torch.complex(torch.randn(10, 100), torch.randn(10, 100))
    scale, summary = entrypoint.prototype_scale(prototypes)
    assert summary["pair_count"] == 45
    assert scale > 0.0

    angle = torch.tensor(0.73)
    rotated = query * torch.complex(torch.cos(angle), torch.sin(angle))
    original_distance = entrypoint.prototype_distance(query, prototypes)
    rotated_distance = entrypoint.prototype_distance(rotated, prototypes)
    assert torch.allclose(original_distance, rotated_distance, atol=2.0e-5, rtol=2.0e-5)

    labels = torch.arange(10, dtype=torch.long)
    loss = entrypoint.prototype_loss(query, labels, prototypes, scale)
    assert torch.isfinite(loss)
    gradient = torch.autograd.grad(loss, query)[0]
    assert torch.isfinite(gradient.real).all()
    assert torch.isfinite(gradient.imag).all()
    assert float(gradient.abs().sum()) > 0.0


def test_lambda_rule_is_one_shot_and_fail_closed() -> None:
    entrypoint = _load_entrypoint()
    assert entrypoint.lambda_proto_from_gradients(2.0, 4.0) == pytest.approx(0.5, abs=1.0e-12)
    with pytest.raises(entrypoint.InvalidPreflight):
        entrypoint.lambda_proto_from_gradients(2.0, 0.0)
    with pytest.raises(entrypoint.InvalidPreflight):
        entrypoint.lambda_proto_from_gradients(float("nan"), 1.0)


def test_v5_gate_requires_every_frozen_criterion() -> None:
    entrypoint = _load_entrypoint()
    metrics = {
        "smooth_delta": 0.06,
        "smooth_ci_lower": 0.01,
        "ols_margin_delta_mean": 0.02,
        "ols_margin_ci_lower": 0.005,
        "smooth_per_class_delta": [0.0] * 10,
        "image_loss_delta": 0.001,
        "raw_cmos_valid": True,
    }
    supported = entrypoint.classify_v5_gate(metrics)
    assert supported["all_frozen_criteria_pass"] is True
    assert supported["classification"] == "V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED"

    metrics["ols_margin_ci_lower"] = 0.0
    failed = entrypoint.classify_v5_gate(metrics)
    assert failed["all_frozen_criteria_pass"] is False
    assert failed["classification"] == "V5_OLS_SEMANTIC_ALIGNMENT_NOT_SUPPORTED"


def test_contract_constants_keep_objective_and_stage_a_firewall_separate() -> None:
    entrypoint = _load_entrypoint()
    assert entrypoint.CONTRACT_PATH.as_posix().endswith("lightgen_2d_complex_ols_semantic_alignment_contract.v5.json")
    assert entrypoint.EXPECTED_HEAD == "9a72e0bcf2de12ab9a0caa7151f273de76768dc7"
    assert entrypoint.STAGE_A_FIREWALL == 0.937
    assert entrypoint.PRODUCTION_GRID == 2048
    source = SCRIPT.read_text(encoding="utf-8")
    assert "L_candidate = L_image + lambda_proto * L_proto" in source or "L_image+lambda_proto*L_proto" in source
    assert "V3 output-feature objective" not in source
