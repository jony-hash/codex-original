from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ldm_project.v2.application import cli
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_relay_metric_reinterpretation as audit


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_relay_metric_reinterpretation_v1.yaml"
REVIEW = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_relay_metric_reinterpretation_review_v1.json"


def _label_signal(drop: float = 0.0) -> dict[str, float]:
    return {"relative_inter_intra_drop": drop, "retrieval_drop": drop}


def _dcm(relay: float, detector: float, canonical: float | None = None, label_drop: float = 0.0) -> dict[str, object]:
    return {
        "retention": {
            audit.RELAY_SEGMENT: {"geometric_mean_retention": relay},
            audit.READOUT_SEGMENT: {"geometric_mean_retention": detector},
        },
        "canonicalized_retention": {"geometric_mean_retention": relay if canonical is None else canonical},
        "label_signal": _label_signal(label_drop),
        "canonicalized_label_signal": _label_signal(label_drop),
    }


def _mra_retention(pre_modal: float, decoder_internal: float) -> dict[str, object]:
    return {
        "retention": {
            "pre_ols_to_modal": {"geometric_mean_retention": pre_modal},
            "decoder_input_to_decoder_layer_2": {"geometric_mean_retention": decoder_internal},
        }
    }


def _cell(raw: float, canonical: float) -> dict[str, object]:
    return {"r_prop": raw, "r_prop_canonicalized": canonical, "r_readout": 0.9, "r_det": 0.8, "power_retention": 1.0}


def test_contract_and_review_are_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec).name.endswith("relay_metric_reinterpretation_review_v1.json")
    assert spec.payload["hypothesis"] == audit.HYPOTHESIS
    assert spec.payload["canonicalization"]["valid_segments"] == [audit.RELAY_SEGMENT]


def test_contract_rejects_parent_or_forbidden_segment_mutation() -> None:
    spec = audit._load_spec(CONFIG)
    with pytest.raises(ValueError, match="MRA parent"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "parent_mra": {}}, spec.config_hash))
    with pytest.raises(ValueError, match="canonicalization"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "canonicalization": {**spec.payload["canonicalization"], "valid_segments": ["incident_to_intensity"]}}, spec.config_hash))


def test_canonicalization_contract_is_analytic_only() -> None:
    contract = audit.CANONICALIZATION
    assert contract["implementation"] == "HFRR._canonicalize_field"
    assert contract["fit"] is False
    assert contract["physical_forward_modified"] is False
    assert audit.RELAY_SEGMENT in contract["valid_segments"]
    assert audit.READOUT_SEGMENT in contract["forbidden_segments"]


def test_threshold_transfer_and_no_update_source_contract() -> None:
    assert audit.RANK_RETENTION_THRESHOLD == 0.90
    assert audit.MATERIAL_GAP == 0.05
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "evaluator_calls" in source
    assert "prior_sampled" in source
    assert "physical_forward_modified" in source


def test_dcm_reinterpretation_d1_d2_d3_boundaries() -> None:
    identity = _dcm(0.95, 0.95)
    d1 = audit._classify_dcm(_dcm(0.60, 0.95, canonical=0.95), _dcm(0.60, 0.95, canonical=0.95), identity)
    assert d1[0] == "relay_propagation_bottleneck_reinterpreted_metric_confound"
    d2 = audit._classify_dcm(_dcm(0.60, 0.95, canonical=0.80), _dcm(0.60, 0.95, canonical=0.80), identity)
    assert d2[0] == "relay_propagation_bottleneck_supported_canonicalized"
    d3 = audit._classify_dcm(_dcm(0.60, 0.60, canonical=0.80), _dcm(0.60, 0.60, canonical=0.80), identity)
    assert d3[0] == "relay_propagation_reinterpretation_underdetermined"


def test_mra_reinterpretation_m1_and_m2_boundaries() -> None:
    m1 = audit._classify_mra(_mra_retention(0.80, 0.80), _mra_retention(0.80, 0.80), "relay_propagation_bottleneck_reinterpreted_metric_confound")
    assert m1[0] == "distributed_bottleneck_supported_canonicalized"
    m2 = audit._classify_mra(_mra_retention(0.80, 0.95), _mra_retention(0.80, 0.95), "relay_propagation_bottleneck_reinterpreted_metric_confound")
    assert m2[0] == "distributed_bottleneck_evidence_reduced"


def test_bosa_reinterpretation_b1_b2_and_no_confirmation() -> None:
    cells = {name: _cell(0.9, 0.9) for name in audit.CELL_ORDER}
    b1, details = audit._bosa_reinterpretation(cells)
    assert b1 == "backend_observable_subspace_mismatch_not_supported_reinterpreted"
    assert details["confirmation_executed"] is False
    cells = {
        "I_to_I": _cell(0.9, 0.9),
        "I_to_S": _cell(0.8, 0.8),
        "S_to_I": _cell(0.8, 0.7),
        "S_to_S": _cell(0.8, 0.6),
    }
    b2, details = audit._bosa_reinterpretation(cells)
    assert b2 == "backend_observable_subspace_reinterpretation_underdetermined"
    assert details["confirmation_executed"] is False


def test_raw_anchor_helpers_are_fail_closed() -> None:
    mra_metrics = {
        "retention": {name: {"geometric_mean_retention": 0.8} for name in (name for name, *_ in audit.mra.SEGMENTS)},
        "reconstruction": {"paired_mse": 0.1},
    }
    parent_mra = {"contract": {"selection": {audit.SCALED: {"retention": {name: {"geometric_mean_retention": 0.8} for name, *_ in audit.mra.SEGMENTS}, "reconstruction": {"paired_mse": 0.1}}}}}
    assert audit._mra_anchor_gate(mra_metrics, "selection", parent_mra)["passed"] is True
    cells = {name: _cell(0.8, 0.8) for name in audit.CELL_ORDER}
    parent_bosa = {"contract": {"selection": {"cells": {name: {"r_prop": 0.8, "r_readout": 0.9, "r_det": 0.8, "power_retention": 1.0} for name in audit.CELL_ORDER}}}}
    assert audit._bosa_anchor_gate(cells, parent_bosa)["passed"] is True
    cells["S_to_S"]["r_prop"] = 0.7
    assert audit._bosa_anchor_gate(cells, parent_bosa)["passed"] is False


def test_panels_are_registered_without_new_samples() -> None:
    assert audit.PANELS["no_new_samples"] is True
    assert audit.PANELS["bosa_selection"]["confirmation_consumed"] is False
    assert audit.PANELS["candidate_prefix"]["consumed"] is False


def test_run_directory_is_write_once(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    validation = root / "validation.json"
    validation.write_text("{}", encoding="utf-8")
    audit._ensure_run_directory(root / "audit.json", validation)
    (root / "other.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit._ensure_run_directory(root / "audit.json", validation)


def test_cli_actions_are_registered_and_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args(["research", "optical-manifold", "scaled-chen-relay-metric-reinterpretation-validate", "--config", "config.yaml", "--review-contract", "review.json", "--dataset-root", "data", "--output", "validation.json"])
    run = parser.parse_args(["research", "optical-manifold", "scaled-chen-relay-metric-reinterpretation-run", "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json", "--dataset-root", "data", "--output", "audit.json"])
    assert validate.research_action.endswith("-validate")
    assert run.device == "cuda"
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_relay_metric_reinterpretation" in router
    assert "import optical_manifold_smf_bogt_relay_metric_reinterpretation" not in router
