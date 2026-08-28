from __future__ import annotations

import inspect

import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research import optical_manifold_smf_integration_canary as canary
from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec


CONFIG = "configs/research/lightgen_optical_manifold_identity_smf_integration_canary_v1.yaml"
REVIEW = "configs/research/lightgen_optical_manifold_identity_smf_integration_canary_review_v1.json"


def _plane() -> PlaneSpec:
    return PlaneSpec("encoder", (100, 100), (2.0e-6, 2.0e-6), 532.0e-9, 0.0, "encoder")


def test_config_review_and_lazy_cli_contract() -> None:
    spec = canary._load_spec(CONFIG)
    canary._assert_contract(spec)
    review, _ = canary._load_review(REVIEW, spec)
    assert review["diagnostic_training_authorized"] is True
    assert review["training_authorized"] is False
    parser = _parser()
    args = parser.parse_args(
        [
            "research", "optical-manifold", "smf-integration-canary-train",
            "--config", CONFIG, "--review-contract", REVIEW,
            "--validation-evidence", "validation.json",
            "--preflight-evidence", "preflight.json",
            "--dataset-root", "data", "--output", "training.json",
        ]
    )
    assert args.research_action == "smf-integration-canary-train"
    assert args.resume_checkpoint is None


def test_canary_isolated_from_formal_and_evaluator_runtime() -> None:
    source = inspect.getsource(canary)
    assert "SmoothMNISTEvaluator" not in source
    assert "load_frozen_evaluator" not in source
    assert base.EXPECTED_TRAINABLE == ("system.encoder.phase", "system.decoder.phase")


def test_research_smf_has_100_ports_and_passive_zero_transport_control() -> None:
    operator = FiberArrayOLS(_plane(), FiberArraySpec())
    field = ComplexField(
        torch.ones(1, 1, 100, 100, dtype=torch.complex64),
        _plane(),
        OperationProvenance("fixture", None, None),
    )
    output = operator(field)
    assert output.tensor.shape == (1, 1, 100, 100)
    assert output.provenance.operation == "synthetic_smf_lp01_port_transport"
    assert operator.last_diagnostics["port_input_coefficients"].shape == (1, 100)
    zero = FiberArrayOLS(_plane(), FiberArraySpec(), transport=torch.zeros(100, 100, dtype=torch.complex64))
    zero_output = zero(field)
    assert torch.equal(zero_output.tensor, torch.zeros_like(zero_output.tensor))
    assert zero.last_diagnostics["output_energy"].max() <= 1.0e-12


def test_rank_replay_is_per_direction_permutation() -> None:
    scores = torch.arange(10 * canary.RANK, dtype=torch.float32).reshape(10, canary.RANK)
    directions = torch.zeros(canary.DIRECTIONS, canary.RANK)
    directions[:, 0] = 1.0
    directions[:, 1] = 1.0
    coefficients, ranks = canary._rank_coefficients(scores, directions)
    assert coefficients.shape == (10, canary.DIRECTIONS)
    expected = torch.arange(10).reshape(-1, 1).expand_as(ranks)
    assert torch.equal(torch.sort(ranks, dim=0).values, expected)


def test_canary_uses_pilot_split_manifest_not_representative_manifest(monkeypatch) -> None:
    def fail_representative(*_args, **_kwargs):
        raise AssertionError("representative manifest must not be consumed")

    monkeypatch.setattr(base, "_load_data_manifest", fail_representative)
    _images, _labels, manifest = canary._load_authoritative_pilot_data("data")
    assert manifest["manifest_hash"] == canary.SOURCE_MANIFEST_HASH
    assert manifest["train_count"] == 55000
    assert manifest["validation_count"] == 5000


def test_layer_gates_cannot_pass_on_empty_or_partial_metadata() -> None:
    assert not canary._four_phase_nonzero({})
    assert not canary._swd_encoder_only({})
    assert not canary._four_phase_nonzero({"system.encoder.phase[0]": 1.0})
    values = {
        "system.encoder.phase[0]": 1.0,
        "system.encoder.phase[1]": 1.0,
        "system.decoder.phase[0]": 0.0,
        "system.decoder.phase[1]": 0.0,
    }
    assert canary._swd_encoder_only(values)


def test_activation_checkpoint_is_part_of_mse_production_path() -> None:
    source = inspect.getsource(canary._mse_gradients)
    assert "checkpoint(" in source
    assert "use_reentrant=False" in source


def test_completed_training_uses_compatible_evidence_status() -> None:
    source = inspect.getsource(canary.train)
    assert 'status = "diagnostic"' in source
    assert 'outcome = "diagnostic_completed"' in source
