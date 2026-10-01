from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_task_relevant_bottleneck as audit


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_task_relevant_bottleneck_v1.yaml"
REVIEW = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_task_relevant_bottleneck_review_v1.json"


def _representations(count: int = 20, dimension: int = 12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(11)
    values = torch.randn((count, dimension), generator=generator) + 1j * torch.randn((count, dimension), generator=generator)
    output = values.clone()
    labels = torch.arange(count, dtype=torch.long) % 10
    return values.to(torch.complex64), output.to(torch.complex64), labels


def test_contract_and_review_are_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec).name.endswith("task_relevant_bottleneck_review_v1.json")
    assert spec.payload["hypothesis"] == audit.HYPOTHESIS
    assert spec.payload["measurement"]["raw_context_not_primary"] is True
    assert audit.PARENT_HFRR["validation"]


def test_contract_rejects_parent_or_primary_metric_mutation() -> None:
    spec = audit._load_spec(CONFIG)
    with pytest.raises(ValueError, match="RMRI parent"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "parent_rmri": {}}, spec.config_hash))
    with pytest.raises(ValueError, match="measurement contract"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "measurement": {**spec.payload["measurement"], "raw_context_not_primary": False}}, spec.config_hash))


def test_normalized_gram_is_transform_invariant_and_dimension_agnostic() -> None:
    values, _, labels = _representations()
    report = audit._control_report(values, labels)
    assert report["passed"] is True
    transformed = audit._control_transforms(values)
    identity = audit._geometry_metrics(values, values, labels)
    padded = torch.cat((values, torch.zeros((values.shape[0], 7), dtype=values.dtype)), dim=1)
    padded_report = audit._geometry_metrics(padded, padded, labels)
    assert identity["geometry_retention"] == pytest.approx(1.0, abs=1e-7)
    assert padded_report["geometry_retention"] == pytest.approx(identity["geometry_retention"], abs=1e-7)
    assert transformed["C1"].shape == values.shape


def test_blockwise_geometry_matches_single_block() -> None:
    values, _, labels = _representations(count=24, dimension=9)
    generator = torch.Generator(device="cpu").manual_seed(12)
    output = (values + 0.1 * (torch.randn(values.shape, generator=generator) + 1j * torch.randn(values.shape, generator=generator))).to(torch.complex64)
    small = audit._geometry_metrics(values, output, labels, block_size=5)
    large = audit._geometry_metrics(values, output, labels, block_size=64)
    assert small["geometry_retention"] == pytest.approx(large["geometry_retention"], abs=1e-6)
    assert small["task_advantage_degradation"] == pytest.approx(large["task_advantage_degradation"], abs=1e-6)
    assert small["retrieval_degradation"] == pytest.approx(large["retrieval_degradation"], abs=1e-6)


def _segment(primary: float, task: float, retrieval: float, *, controls: bool = True) -> dict[str, object]:
    return {
        "finite": True,
        "controls_pass": controls,
        "primary": {"geometry_degradation": primary},
        "task_geometry": {
            "task_advantage_degradation": task,
            "retrieval_degradation": retrieval,
            "geometry_degradation_bootstrap": {"ci": [0.001, primary + 0.01]},
        },
    }


def test_segment_and_overall_classification_boundaries() -> None:
    supported = _segment(0.06, 0.06, 0.0)
    assert audit._segment_classification(supported, supported)[0] == "supported"
    generic = _segment(0.06, 0.0, 0.0)
    assert audit._segment_classification(generic, generic)[0] == "generic_geometry_change_only"
    unsupported = _segment(0.01, 0.01, 0.01)
    assert audit._segment_classification(unsupported, unsupported)[0] == "not_supported"
    assert audit._overall_classification({audit.OLS_SEGMENT: "supported", audit.DECODER_SEGMENT: "not_supported"})[0] == "task_relevant_bottleneck_ols_only_supported"
    assert audit._overall_classification({audit.OLS_SEGMENT: "not_supported", audit.DECODER_SEGMENT: "supported"})[0] == "task_relevant_bottleneck_decoder_only_supported"
    assert audit._overall_classification({audit.OLS_SEGMENT: "supported", audit.DECODER_SEGMENT: "supported"})[0] == "task_relevant_distributed_bottleneck_both_supported"
    assert audit._overall_classification({audit.OLS_SEGMENT: "not_supported", audit.DECODER_SEGMENT: "not_supported"})[0] == "task_relevant_bottleneck_not_supported"
    assert audit._overall_classification({audit.OLS_SEGMENT: "underdetermined", audit.DECODER_SEGMENT: "not_supported"})[0] == "task_relevant_bottleneck_attribution_underdetermined"
    assert audit._overall_classification({audit.OLS_SEGMENT: "invalid_harness", audit.DECODER_SEGMENT: "not_supported"})[0] == "invalid_harness"


def test_panel_contract_and_write_once(tmp_path: Path) -> None:
    assert audit.PANELS["selection"]["count"] == 1000
    assert audit.PANELS["selection"]["per_label"] == 100
    assert audit.PANELS["confirmation"]["count"] == 5000
    assert audit.PANELS["candidate_prefix"]["consumed"] is False
    target = tmp_path / "validation.json"
    audit._write_new(target, {"ok": True}, artifact_kind="test evidence")
    with pytest.raises(FileExistsError):
        audit._write_new(target, {"ok": False}, artifact_kind="test evidence")


def test_json_report_omits_panel_tensors() -> None:
    values, _, _ = _representations(count=4, dimension=3)
    report = {"finite": True, "compact_per_sample": {"geometry": values}, "scalar": 1.0}
    compact = audit._json_report(report)
    assert compact == {"finite": True, "scalar": 1.0}


def test_cli_actions_are_registered_and_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args(["research", "optical-manifold", "scaled-chen-task-relevant-bottleneck-validate", "--config", "config.yaml", "--review-contract", "review.json", "--dataset-root", "data", "--output", "validation.json"])
    run = parser.parse_args(["research", "optical-manifold", "scaled-chen-task-relevant-bottleneck-run", "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json", "--dataset-root", "data", "--output", "audit.json"])
    assert validate.research_action.endswith("task-relevant-bottleneck-validate")
    assert run.device == "cuda"
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_task_relevant_bottleneck" in router
    assert "import optical_manifold_smf_bogt_task_relevant_bottleneck" not in router


def test_no_update_and_relay_canonicalizer_boundaries() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "evaluator_calls" in source
    assert "prior_sampled" in source
    assert "HFRR._canonicalize_field" not in source
    assert audit.MEASUREMENT["segments"] == [
        {"id": audit.OLS_SEGMENT, "input": "pre_ols", "output": "modal_complex"},
        {"id": audit.DECODER_SEGMENT, "input": "decoder_input", "output": "decoder_layer_2"},
    ]
