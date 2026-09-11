from __future__ import annotations

import itertools
import json
from pathlib import Path

from ldm_project.v2.evidence import file_sha256
from scripts.research import define_n3_copula_mixed_component_swap_diagnostic_contract_v1 as contract


EXPECTED_ARTIFACT_SHA256 = "d46c4d8e8d8dbc41f325f2e1d7f001d645479a60deef08e5d514a80bada22763"


def _load_artifact() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / contract.OUTPUT_DIR / contract.CONTRACT_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def test_artifact_is_immutable_definition_authority() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / contract.OUTPUT_DIR / contract.CONTRACT_NAME
    artifact = _load_artifact()

    assert file_sha256(path) == EXPECTED_ARTIFACT_SHA256
    assert artifact["schema"] == contract.CONTRACT_SCHEMA
    assert artifact["semantic_authority_id"] == contract.TASK_ID
    assert artifact["classification"] == contract.READY_CLASSIFICATION
    assert artifact["claim_classification"] == contract.CLAIM_CLASSIFICATION
    assert artifact["definition_only"] is True
    assert artifact["contract_totality"]["status"] == "passed"
    assert artifact["contract_totality"]["future_runtime_math_choices"] == []
    assert artifact["git"]["contract_script_sha256"] == file_sha256(Path(contract.__file__))


def test_pairing_rank_oracles_and_original_raw_metric_are_total() -> None:
    artifact = _load_artifact()
    pairing = artifact["pairing"]

    assert pairing["historical_pairing_only"] is True
    assert pairing["model_banks"]["replicates"] == [0, 1, 2, 3]
    assert pairing["model_reference_banks"]["replicates"] == list(range(8))
    assert pairing["model_reference_banks"]["reference_pairs"] == [
        {"pair_index": 0, "left_replicate": 0, "right_replicate": 1},
        {"pair_index": 1, "left_replicate": 2, "right_replicate": 3},
        {"pair_index": 2, "left_replicate": 4, "right_replicate": 5},
        {"pair_index": 3, "left_replicate": 6, "right_replicate": 7},
    ]
    assert len(pairing["folds"]) == contract.FOLD_COUNT
    for fold, record in enumerate(pairing["folds"]):
        assert record["fold"] == fold
        assert len(record["data_vs_model"]) == contract.DATA_VS_MODEL_REPLICATES
        assert record["original_raw_q95_metric_field"] == (
            f"fold_execution[{fold}].model_vs_model_reference_q95.energy_distance"
        )
        for replicate, pair in enumerate(record["data_vs_model"]):
            assert pair["replicate"] == replicate
            assert pair["model_bank_key"] == (
                f"fold={fold}|role=data-vs-model|replicate={replicate}"
            )
            assert pair["original_raw_metric_field"] == (
                f"fold_execution[{fold}].data_vs_model_replicates[{replicate}].energy_distance"
            )

    assert artifact["ordinal_rank_representation"]["rank_index_base"] == 0
    assert artifact["ordinal_rank_representation"]["total_order_key"] == (
        "(numeric X[i,j] ascending, original row index i ascending for exact ties)"
    )
    assert artifact["marginal_oracle"]["equation"] == "H_MARG[i,j] = sorted_D[r_M(i,j),j]"
    assert artifact["dependence_oracle"]["equation"] == "H_DEP[i,j] = sorted_M[r_D(i,j),j]"

    metric = artifact["primary_diagnostic"]["raw_energy_distance_implementation"]
    assert metric["metric_name"] == "validated_biased_empirical_energy_distance_euclidean_q_space"
    assert metric["negative_policy"] == "require raw >= -1e-10; otherwise invalid_harness; return max(0.0,raw)"
    assert artifact["primary_diagnostic"]["frozen_model_reference_q95"]["oracle_hybrid_values_in_q95"] is False


def test_pair_fold_and_aggregate_mappings_cover_every_outcome() -> None:
    expected_pairs = {
        (False, False): "NEITHER_ORACLE",
        (False, True): "DEPENDENCE_ORACLE_ONLY",
        (True, False): "MARGINAL_ORACLE_ONLY",
        (True, True): "BOTH_ORACLES",
    }
    for marginal, dependence in itertools.product((False, True), repeat=2):
        assert contract.pair_sufficiency_label(marginal, dependence) == expected_pairs[(marginal, dependence)]

    for label in contract.PAIR_LABEL_VALUES:
        assert contract.fold_sufficiency_label([]) == contract.FOLD_CONTROL
        assert contract.fold_sufficiency_label([label, label]) == label
        other = next(value for value in contract.PAIR_LABEL_VALUES if value != label)
        assert contract.fold_sufficiency_label([label, other]) == contract.FOLD_MIXED

        control_folds = {fold: contract.FOLD_CONTROL for fold in range(contract.FOLD_COUNT)}
        control_folds[0] = label
        assert contract.aggregate_oracle_sufficiency_label(control_folds, [label, label]) == contract.AGGREGATE_LABELS[label]
        assert contract.aggregate_oracle_sufficiency_label(control_folds, [label, "NEITHER_ORACLE" if label != "NEITHER_ORACLE" else "BOTH_ORACLES"]) == contract.AGGREGATE_MIXED

    all_control = {fold: contract.FOLD_CONTROL for fold in range(contract.FOLD_COUNT)}
    assert contract.aggregate_oracle_sufficiency_label(all_control, []) == contract.AGGREGATE_MIXED
    mixed_folds = dict(all_control)
    mixed_folds[0] = contract.FOLD_MIXED
    assert contract.aggregate_oracle_sufficiency_label(mixed_folds, ["MARGINAL_ORACLE_ONLY"]) == contract.AGGREGATE_MIXED


def test_definition_consumer_has_no_tensor_load_or_oracle_metric_path() -> None:
    source = Path(contract.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch.load" not in source
    assert "safe_torch_load" not in source
    artifact = _load_artifact()
    boundary = artifact["execution_boundary"]
    assert boundary["q_tensor_loaded_by_this_definition_consumer"] is False
    assert boundary["component_swaps_executed_by_this_definition_consumer"] is False
    assert boundary["oracle_metrics_computed_by_this_definition_consumer"] is False
    assert boundary["sealed_confirmation_read"] is False
    assert boundary["cuda"] is False
    assert boundary["training"] is False
