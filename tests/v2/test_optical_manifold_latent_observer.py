from __future__ import annotations

import inspect

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research import optical_manifold_latent_observer as observer
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec


CONFIG = "configs/research/lightgen_optical_manifold_identity_latent_observer_v1.yaml"
REVIEW = "configs/research/lightgen_optical_manifold_identity_latent_observer_review_v1.json"


def test_observer_config_review_and_lazy_import_contract() -> None:
    spec = observer._load_spec(CONFIG)
    review, _ = observer._load_review(REVIEW, spec)
    assert spec.payload["source"]["attribution_evidence_sha256"] == observer.SOURCE_AUDIT_HASH
    assert review["no_update_observer_authorized"] is True
    assert review["training_authorized"] is False
    assert "torch.optim" not in inspect.getsource(observer)


def test_observer_cli_actions_are_registered() -> None:
    parser = _parser()
    validate = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "latent-observer-validate",
            "--config",
            CONFIG,
            "--review-contract",
            REVIEW,
            "--dataset-root",
            "data",
            "--output",
            "validation.json",
        ]
    )
    assert validate.research_action == "latent-observer-validate"
    run = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "latent-observer-run",
            "--config",
            CONFIG,
            "--review-contract",
            REVIEW,
            "--validation-evidence",
            "validation.json",
            "--dataset-root",
            "data",
            "--output",
            "observer.json",
        ]
    )
    assert run.research_action == "latent-observer-run"
    assert run.device == "cuda"


def test_trace_lookup_uses_plane_and_operation_not_position() -> None:
    plane = PlaneSpec("encoder_layer_2", (2, 2), (1.0, 1.0), 1.0, 0.0, "encoder")
    pre = ComplexField(
        torch.ones(1, 1, 2, 2, dtype=torch.complex64),
        plane,
        OperationProvenance("finite_crop_hard_aperture", "encoder_layer_1", 1.0),
    )
    post = ComplexField(
        pre.tensor,
        plane.with_plane("ols_output"),
        OperationProvenance("identity_ols", "encoder_layer_2", None),
    )
    assert observer._find_trace_field((post, pre), "encoder_layer_2", "finite_crop_hard_aperture") is pre
    with pytest.raises(ValueError, match="found 0"):
        observer._find_trace_field((post,), "encoder_layer_2", "finite_crop_hard_aperture")
    with pytest.raises(ValueError, match="found 2"):
        observer._find_trace_field((pre, pre), "encoder_layer_2", "finite_crop_hard_aperture")


def test_structure_metrics_are_finite_and_report_true_and_permuted_ratios() -> None:
    values = torch.tensor(
        [
            [[[[1.0, 0.0]]]],
            [[[[1.1, 0.0]]]],
            [[[[5.0, 0.0]]]],
            [[[[5.1, 0.0]]]],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    permuted = torch.tensor([0, 1, 0, 1])
    metrics, scores = observer._structure_metrics(values, labels, permuted)
    assert scores.shape == (4, min(observer.PCA_COMPONENTS, 4))
    assert metrics["finite"] is True
    assert metrics["entropy_effective_rank"] > 0.0
    assert metrics["participation_rank"] > 0.0
    assert metrics["true_label_distance"]["inter_intra_ratio"] > 1.0


def test_observer_does_not_modify_source_or_formal_runtime_symbols() -> None:
    source = inspect.getsource(observer)
    assert "FormalForwardResult" not in source
    assert "return_optical_trace=True" in source
    assert "encoder_layer_2" in source
    assert "identity_ols" in source
