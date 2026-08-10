"""Contracts for the diagnostic-only capacity-map command surface."""

from __future__ import annotations

import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.capacity_map import _latest_evidence, _objective_loss, _optical_output
from ldm_project.v2.application.research.measurement import MEASUREMENT_CONTRACT, fixed_panels, panel_hashes
from ldm_project.v2.application.research.models import ShapeMNISTEvaluator, SpatialSeedDecoder
from ldm_project.v2.application.research.spec import load_capacity_map_spec


def test_capacity_map_config_is_pre_registered() -> None:
    spec = load_capacity_map_spec("configs/research/conditional_capacity_map_v1.yaml")
    assert spec.variant("E0:O2")[0].code_dim == 8
    assert spec.variant("E4:O5")[1].diagnostic_only
    assert len(spec.electronic) * len(spec.optical) == 30
    dual = load_capacity_map_spec("configs/research/conditional_capacity_map_v2_2_dual.yaml")
    assert dual.objective_id == "dual_bundle"
    assert dual.measurement_contract == MEASUREMENT_CONTRACT


def test_capacity_map_cli_is_lazy_and_separate_from_v2_config_loader() -> None:
    args = _parser().parse_args(
        [
            "research", "capacity-map", "screen", "--config", "capacity.yaml",
            "--variant", "E2:O0", "--device", "cpu",
        ]
    )
    assert args.domain == "research"
    assert args.research_program == "capacity-map"
    assert args.research_action == "screen"
    assert args.variant == "E2:O0"
    diagnose = _parser().parse_args(
        [
            "research", "capacity-map", "diagnose", "--config", "capacity.yaml",
            "--variant", "E2:O0", "--checkpoint", "model.pt", "--source-evidence", "source.json", "--device", "cpu",
        ]
    )
    assert diagnose.research_action == "diagnose"
    assert diagnose.checkpoint == "model.pt"
    shape = _parser().parse_args(["research", "capacity-map", "qualify-shape", "--config", "capacity.yaml", "--device", "cpu"])
    assert shape.seed == 7319
    closeout = _parser().parse_args(
        ["research", "capacity-map", "closeout", "--config", "dual.yaml", "--comparison-config", "primary.yaml", "--evidence-root", "outputs"]
    )
    assert closeout.device == "cpu"
    assert closeout.variant is None


def test_spatial_seed_decoder_preserves_exact_zero_code_anchor_and_budget() -> None:
    model = SpatialSeedDecoder(32)
    labels = torch.tensor([0, 4, 9])
    zero = torch.zeros(3, 32)
    output = model(zero, labels)
    carrier = torch.sigmoid(model.class_carrier_logits.index_select(0, labels))
    assert torch.equal(output, carrier)
    assert sum(parameter.numel() for parameter in model.parameters()) <= 1_500_000
    output.sum().backward()
    assert model.projection.weight.grad is not None


def test_upper_bound_slots_do_not_silently_substitute_an_optical_model() -> None:
    spec = load_capacity_map_spec("configs/research/conditional_capacity_map_v1.yaml")
    _, optical = spec.variant("E2:O5")
    try:
        _optical_output(optical, torch.zeros(1, 1, 200, 200), device=torch.device("cpu"))
    except RuntimeError as error:
        assert "requires CUDA full-grid propagation" in str(error)
    else:  # pragma: no cover - the failure is the contract
        raise AssertionError("O5 must not silently become a CPU/amplitude-only proxy")


def test_panels_are_fixed_and_shape_evaluator_is_distinct() -> None:
    first = fixed_panels(device=torch.device("cpu"), code_dim=8)
    assert tuple(first) == ("axis", "uniform_1234", "uniform_5678", "holdout_9012")
    assert all(codes.shape == (170, 8) and labels.shape == (170,) for codes, labels in first.values())
    assert panel_hashes(device=torch.device("cpu"), code_dim=8) == panel_hashes(device=torch.device("cpu"), code_dim=8)
    shape = ShapeMNISTEvaluator()
    features, logits = shape(torch.zeros(2, 1, 28, 28))
    assert features.shape == (2, 128)
    assert logits.shape == (2, 10)
    assert shape.architecture_kind != "smooth_mnist_evaluator_v2"


def test_dual_bundle_loss_backpropagates_through_both_frozen_evaluators() -> None:
    primary, smooth = ShapeMNISTEvaluator(), ShapeMNISTEvaluator()
    for evaluator in (primary, smooth):
        evaluator.freeze()
    generated = torch.rand(20, 1, 28, 28, requires_grad=True)
    real = torch.rand(20, 1, 28, 28)
    labels = torch.arange(10).repeat_interleave(2)
    loss, detail = _objective_loss(
        generated, real, labels, primary=primary, smooth=smooth,
        objective_id="dual_bundle", primary_baseline=torch.tensor(1.0), smooth_baseline=torch.tensor(1.0),
    )
    loss.backward()
    assert generated.grad is not None and generated.grad.abs().sum() > 0
    assert all(parameter.grad is None for evaluator in (primary, smooth) for parameter in evaluator.parameters())
    assert "smooth_semantic_loss" in detail


def test_same_revision_content_conflict_is_ambiguous_and_invalid_is_retained() -> None:
    base = {
        "config_hash": "config", "dataset": "mnist", "variant": "E2:O0", "action": "diagnose", "seed": 1234,
        "evidence_revision": 9,
    }
    first = base | {"content_hash": "a", "path": "invalid.json", "state": "invalid_harness", "reason": "shape qualification failed"}
    second = base | {"content_hash": "b", "path": "valid.json", "state": "screen_passed", "reason": None}
    selected, ambiguous = _latest_evidence([first, second])
    assert selected == []
    assert ambiguous == [{
        "key": ["config", "mnist", "E2:O0", "diagnose", 1234], "evidence_revision": 9,
        "content_hashes": ["a", "b"], "paths": ["invalid.json", "valid.json"], "classification": "ambiguous_revision",
    }]
