"""Contract tests for the frozen semantic geometry alignment diagnostic."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.scaled_chen.run_scaled_chen_controller_to_ols_semantic_geometry_alignment import CLASS_COUNT, CONFIG_SCHEMA, MANIFEST_SCHEMA, PORT_COUNT, TASK_ID, _assert_output_tree, _centroid_geometry, _covariance_stats, _mechanism_decision, _nearest_components, apply_class_translation, fit_class_translation, load_config


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/research/scaled_chen/lightgen_scaled_chen_controller_to_ols_semantic_geometry_alignment_v1.yaml"


def test_config_binds_frozen_s3_and_s1_authorities() -> None:
    path, payload, _, _ = load_config(CONFIG_PATH)

    assert path == CONFIG_PATH
    assert payload["schema"] == CONFIG_SCHEMA
    assert payload["task_id"] == TASK_ID
    assert payload["observer"]["representation"] == "R1_POWER"
    assert payload["execution"]["device"] == "cpu"
    assert payload["authorization"]["s3_authority_mutated"] is False


def test_selection_translation_is_classwise_and_confirmation_safe() -> None:
    labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(10)
    real_selection = torch.zeros((CLASS_COUNT * 10, PORT_COUNT), dtype=torch.float64)
    controller_selection = torch.zeros_like(real_selection)
    real_selection[:, 0] = labels.to(torch.float64) * 3.0
    controller_selection[:, 0] = labels.to(torch.float64) * 3.0
    real_selection[labels == 1, 0] += 2.0
    real_labels = labels
    controller_labels = labels.clone()

    translation = fit_class_translation(
        real_selection, real_labels, controller_selection, controller_labels
    )
    assert translation.shape == (CLASS_COUNT, PORT_COUNT)
    assert torch.equal(translation[0], torch.zeros(PORT_COUNT, dtype=torch.float64))
    assert translation[1, 0].item() == pytest.approx(2.0)

    confirmation = torch.zeros((2, PORT_COUNT), dtype=torch.float64)
    original = confirmation.clone()
    aligned = apply_class_translation(
        confirmation, torch.tensor([0, 1]), translation
    )
    assert torch.equal(confirmation, original)
    assert aligned[1, 0].item() == pytest.approx(2.0)


def test_nearest_components_keep_intended_class_and_confusion() -> None:
    reference_labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(10)
    reference = torch.zeros((CLASS_COUNT * 10, PORT_COUNT), dtype=torch.float64)
    reference[:, 0] = reference_labels.to(torch.float64) * 4.0
    labels = torch.arange(CLASS_COUNT, dtype=torch.long)
    queries = torch.zeros((CLASS_COUNT, PORT_COUNT), dtype=torch.float64)
    queries[:, 0] = labels.to(torch.float64) * 4.0

    result = _nearest_components(queries, labels, reference, reference_labels)

    assert result["predictions"].tolist() == list(range(CLASS_COUNT))
    assert result["target"].tolist() == pytest.approx([0.0] * CLASS_COUNT)
    assert result["margin"].tolist() == pytest.approx([16.0] * CLASS_COUNT)


def test_covariance_spectrum_is_deterministic_and_full() -> None:
    values = torch.arange(40, dtype=torch.float64).reshape(10, 4)
    first = _covariance_stats(values, 1.0e-6)
    second = _covariance_stats(values, 1.0e-6)

    assert first == second
    assert len(first["eigenvalues_descending"]) == 4
    assert first["finite"] is True


def test_interclass_geometry_has_fixed_class_order_and_no_fit() -> None:
    values = torch.zeros((CLASS_COUNT * 2, PORT_COUNT), dtype=torch.float64)
    labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(2)
    values[:, 0] = labels.to(torch.float64) * 3.0
    payload = _centroid_geometry(
        values,
        labels,
        values,
        labels,
        {"minimum_spearman": 0.8, "minimum_nearest_relation_agreement": 0.8},
    )

    assert len(payload["real_distance_matrix"]) == CLASS_COUNT
    assert payload["finite"] is True
    assert payload["qualitative_status"] in {
        "INTER_CLASS_SEMANTIC_GEOMETRY_PRESERVED",
        "INTER_CLASS_SEMANTIC_GEOMETRY_DISTORTED",
    }


def test_mechanism_decision_does_not_create_s3_gate() -> None:
    reachability = {
        "confirmation": {
            "q50_shrinkage_positive_fraction": 1.0,
            "q95_shrinkage_positive_fraction": 1.0,
            "per_class": {
                str(label): {
                    "q50_gap_after": 0.0,
                    "delta_q50_after": {"upper95": -0.01},
                    "q95_gap_after": 0.0,
                    "delta_q95_after": {"upper95": -0.01},
                }
                for label in range(CLASS_COUNT)
            },
        }
    }
    targeting = {"confirmation": {"all_class_non_degradation": False}}
    centroid = {
        "selection": {"qualitative_status": "INTER_CLASS_SEMANTIC_GEOMETRY_DISTORTED"},
        "confirmation": {"qualitative_status": "INTER_CLASS_SEMANTIC_GEOMETRY_DISTORTED"},
    }
    coverage = {"confirmation": {"compressed": True}}
    result = _mechanism_decision(
        reachability,
        targeting,
        centroid,
        coverage,
        {
            "translation_effective_min_q50_shrinkage_fraction": 0.8,
            "translation_effective_min_q95_shrinkage_fraction": 0.8,
        },
    )

    assert result["primary_mechanism"] == "S3_GEOMETRY_ANCHOR_PLUS_TANGENT_MISMATCH"
    assert result["new_s3_gate_created"] is False


def test_f2_uses_registered_confirmation_hard_criterion_not_point_gap_alone() -> None:
    base_per_class = {
        str(label): {
            "q50_gap_after": 0.1,
            "delta_q50_after": {"upper95": -0.01},
            "q95_gap_after": 0.1,
            "delta_q95_after": {"upper95": -0.01},
        }
        for label in range(CLASS_COUNT)
    }
    reachability = {
        "confirmation": {
            "q50_shrinkage_positive_fraction": 1.0,
            "q95_shrinkage_positive_fraction": 1.0,
            "per_class": base_per_class,
        }
    }
    targeting = {"confirmation": {"all_class_non_degradation": True}}
    centroid = {
        "selection": {"qualitative_status": "INTER_CLASS_SEMANTIC_GEOMETRY_PRESERVED"},
        "confirmation": {"qualitative_status": "INTER_CLASS_SEMANTIC_GEOMETRY_PRESERVED"},
    }
    coverage = {"confirmation": {"compressed": False}}
    anchor_result = _mechanism_decision(
        reachability,
        targeting,
        centroid,
        coverage,
        {
            "translation_effective_min_q50_shrinkage_fraction": 0.8,
            "translation_effective_min_q95_shrinkage_fraction": 0.8,
        },
    )
    assert anchor_result["higher_order_mismatch_supported"] is False

    base_per_class["3"]["delta_q50_after"]["upper95"] = 0.01
    f2_result = _mechanism_decision(
        reachability,
        targeting,
        centroid,
        coverage,
        {
            "translation_effective_min_q50_shrinkage_fraction": 0.8,
            "translation_effective_min_q95_shrinkage_fraction": 0.8,
        },
    )
    assert f2_result["higher_order_mismatch_supported"] is True
    assert f2_result["primary_mechanism"] == "S3_GEOMETRY_ANCHOR_PLUS_TANGENT_MISMATCH"
    assert f2_result["secondary_mechanism"] == "S3_GEOMETRY_HIGHER_ORDER_MISMATCH_PRIMARY"


def test_output_tree_rejects_tiff_and_unknown_files(tmp_path: Path) -> None:
    expected = {
        "authority_binding.json",
        "centroid_translation.json",
        "selection_confirmation_isolation.json",
        "confirmation_before_after.json",
        "anchor_offset_stability.json",
        "interclass_centroid_geometry.json",
        "covariance_geometry.json",
        "coverage_metrics.json",
        "aligned_residual_metrics.json",
        "class9_diagnostic.json",
        "compact_report.json",
        "geometry_alignment_manifest.json",
    }
    for name in expected:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    _assert_output_tree(tmp_path)
    (tmp_path / "unexpected.tiff").write_bytes(b"x")
    with pytest.raises(ValueError, match="tree"):
        _assert_output_tree(tmp_path)

    assert MANIFEST_SCHEMA.startswith("lightgen_scaled_chen_controller_to_ols")
