from __future__ import annotations

import copy
import json
import inspect
from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.optical_manifold import optical_manifold_latent_observer as observer
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec


CONFIG = "configs/research/optical_manifold/lightgen_optical_manifold_identity_latent_observer_v1.yaml"
REVIEW = "configs/research/optical_manifold/lightgen_optical_manifold_identity_latent_observer_review_v1.json"


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


def test_observer_requires_process_level_cublas_determinism(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        observer._require_deterministic_cuda_runtime()

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        observer._require_deterministic_cuda_runtime()

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    observer._require_deterministic_cuda_runtime()


def test_observer_rejects_missing_cublas_config_before_run_creation(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    output = tmp_path / "observer.json"
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        observer.run(SimpleNamespace(device="cuda", output=str(output)))
    assert not output.exists()


def test_observer_restores_deterministic_algorithm_state_on_exception() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError, match="sentinel"):
        with observer._deterministic_algorithms():
            assert torch.are_deterministic_algorithms_enabled() is True
            raise RuntimeError("sentinel")
    assert torch.are_deterministic_algorithms_enabled() is previous


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


def test_endpoint_mse_requires_exact_camera_shape() -> None:
    raw = torch.ones(1, 1, 2, 2)
    endpoint_raw = torch.ones(2, 1, 2, 2)
    assert observer._endpoint_mse(raw, endpoint_raw, 0) == 0.0
    with pytest.raises(ValueError, match="shape mismatch"):
        observer._endpoint_mse(raw.squeeze(0), endpoint_raw, 0)


def test_observer_does_not_modify_source_or_formal_runtime_symbols() -> None:
    source = inspect.getsource(observer)
    assert "FormalForwardResult" not in source
    assert "return_optical_trace=True" in source
    assert "encoder_layer_2" in source
    assert "identity_ols" in source


@pytest.mark.historical_artifact
def test_observer_validation_is_distinct_from_attribution_validation(tmp_path) -> None:
    spec = observer._load_spec(CONFIG)
    formal, _ = observer._load_formal(spec)
    source_validation = json.loads(
        observer._resolve(spec.payload["source"]["attribution_validation"])
        .read_text(encoding="utf-8")
    )
    observer_validation_path = observer._resolve(
        "outputs/20260827_v2/runs/"
        "lightgen_optical_manifold_identity_latent_observer_v1_attempt_01/"
        "latent_observer_validation.json"
    )
    observer_validation = json.loads(observer_validation_path.read_text(encoding="utf-8"))
    source_hashes = dict(observer_validation["evidence"]["input_hashes"])
    data = SimpleNamespace(
        manifest={"full_epoch_manifest_hash": source_validation["evidence"]["context"]["dataset_hash"]}
    )
    state = {"commit": source_validation["evidence"]["git_commit"]}

    with pytest.raises(ValueError, match="invalid schema"):
        observer._load_observer_validation(
            spec.payload["source"]["attribution_validation"],
            spec,
            formal,
            data,
            state,
            source_hashes,
        )

    observer_validation = copy.deepcopy(observer_validation)
    observer_validation["evidence"]["git_commit"] = state["commit"]
    path = tmp_path / "observer_validation.json"
    path.write_text(json.dumps(observer_validation), encoding="utf-8")
    loaded, digest = observer._load_observer_validation(
        path,
        spec,
        formal,
        data,
        state,
        source_hashes,
    )
    assert loaded["schema"] == observer.VALIDATION_SCHEMA
    assert digest == observer.file_sha256(path)

    observer_validation["evidence"]["git_commit"] = "0" * 40
    path.write_text(json.dumps(observer_validation), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        observer._load_observer_validation(path, spec, formal, data, state, source_hashes)


@pytest.mark.historical_artifact
def test_observer_validation_rejects_incomplete_source_hashes(tmp_path) -> None:
    spec = observer._load_spec(CONFIG)
    formal, _ = observer._load_formal(spec)
    payload = json.loads(
        observer._resolve(
            "outputs/20260827_v2/runs/"
            "lightgen_optical_manifold_identity_latent_observer_v1_attempt_01/"
            "latent_observer_validation.json"
        ).read_text(encoding="utf-8")
    )
    source_hashes = dict(payload["evidence"]["input_hashes"])
    missing = next(iter(source_hashes))
    del source_hashes[missing]
    data = SimpleNamespace(
        manifest={"full_epoch_manifest_hash": payload["evidence"]["context"]["dataset_hash"]}
    )
    state = {"commit": payload["evidence"]["git_commit"]}
    path = tmp_path / "observer_validation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source hashes"):
        observer._load_observer_validation(path, spec, formal, data, state, source_hashes)
