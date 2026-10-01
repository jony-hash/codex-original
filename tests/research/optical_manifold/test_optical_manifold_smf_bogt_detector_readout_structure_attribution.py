from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_detector_readout_structure_attribution as audit


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_v1.yaml"
REVIEW = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_review_v1.json"
RMAL_PROBE = ROOT / audit.RMAL_CONSUMER["probe"]


def _probe() -> audit.ReadoutProbe:
    payload = audit.safe_torch_load(RMAL_PROBE, expected_sha256=audit.RMAL_CONSUMER["probe_sha256"], map_location="cpu")
    indices = payload["fit_indices"].detach().cpu().to(torch.long)
    return audit._load_rmal_probe(payload, indices, audit.RMAL_CONSUMER["probe_sha256"])


def _synthetic_weights(kind: str) -> torch.Tensor:
    weights = torch.zeros((257, 40000), dtype=torch.float32)
    if kind == "broad":
        weights[:-1].fill_(1.0)
        return weights
    if kind != "local":
        raise ValueError(kind)
    bins = audit._pool_bins()
    membership = audit._bin_membership(200, bins)
    rows = torch.arange(200).repeat_interleave(200)
    columns = torch.arange(200).repeat(200)
    source_rows = membership[:, rows].to(torch.long).argmax(dim=0)
    source_columns = membership[:, columns].to(torch.long).argmax(dim=0)
    source = source_rows * 16 + source_columns
    weights[source, torch.arange(40000)] = 1.0
    return weights


def test_rmal_consumer_exact_hash_lock() -> None:
    assert audit.RMAL_CONSUMER["config_sha256"] == "c50b5b2d12830c40fbe619f137aca40f36f799f277e637d1ec20ac7e3403859f"
    assert audit.RMAL_CONSUMER["review_sha256"] == "73ba944ac5766c88591e0f5125cc031a1c9c55bd84aa2cbc30ea457bd4fb30e8"
    assert audit.RMAL_CONSUMER["validation_sha256"] == "affb4961626a2caaf4dc2c8bb04c06ac4f8de51d1d4a1ab418858eaa11e536ec"
    assert audit.RMAL_CONSUMER["audit_sha256"] == "5ba51e041a266865fc0443b729d31a0c97c35f2b86d6ef45f11dacf6617a15fa"
    assert audit.RMAL_CONSUMER["compact_report_sha256"] == "c335f75124666e9423d66b7a4537c74ff642ef2d71044bda86be16f9333a2137"
    assert audit.RMAL_CONSUMER["probe_sha256"] == "3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed"
    assert audit.file_sha256(RMAL_PROBE) == audit.RMAL_CONSUMER["probe_sha256"]


def test_config_and_review_are_exactly_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec) == REVIEW
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert review["approved_contract"]["permutation_null"]["count"] == 256
    assert review["approved_contract"]["permutation_null"]["seed"] == 0


def test_rmal_probe_shapes_are_locked() -> None:
    probe = _probe()
    assert probe.weights.shape == (256, 40000)
    assert probe.bias.shape == (40000,)
    assert probe.full_weights.shape == (257, 40000)


def test_source_and_target_order_are_row_major() -> None:
    centers = audit._physical_cell_centers()
    assert centers[0, 0] == centers[1, 0]
    assert centers[0, 1] != centers[1, 1]
    assert centers[15, 0] == centers[0, 0]
    assert centers[16, 0] != centers[15, 0]
    rows = torch.tensor([0, 0, 1], dtype=torch.long)
    columns = torch.tensor([0, 1, 0], dtype=torch.long)
    indices, mask = audit._expected_support_indices(rows, columns)
    assert mask.shape[0] == 3
    assert indices[0, 0].item() == 0
    assert indices[1, 0].item() == 0
    assert indices[2, 0].item() == 0


def test_exact_adaptive_pooling_bin_construction() -> None:
    bins = audit._pool_bins()
    assert bins[0] == (0, 13)
    assert bins[1] == (12, 25)
    assert bins[-1] == (187, 200)
    assert bins == tuple(((i * 200) // 16, ((i + 1) * 200 + 15) // 16) for i in range(16))


def test_overlapping_bin_semantics() -> None:
    bins = audit._pool_bins()
    membership = audit._bin_membership(200, bins)
    assert bool(membership[:, 12].sum() == 2)
    assert bool(membership[:, 25].sum() == 1)


def test_expected_support_is_constructed_from_membership() -> None:
    rows = torch.tensor([12], dtype=torch.long)
    columns = torch.tensor([12], dtype=torch.long)
    indices, mask = audit._expected_support_indices(rows, columns)
    assert mask.sum().item() == 4
    assert set(indices[0, mask[0]].tolist()) == {0, 1, 16, 17}
    dense = audit._expected_support(rows, columns)
    assert dense.shape == (1, 256)
    assert dense[0, indices[0, mask[0]]].all()


def test_physical_cell_centers_use_actual_raw_pixel_centers() -> None:
    centers = audit._physical_cell_centers()
    raw = (torch.arange(200, dtype=torch.float64) - 99.5) * 2.5e-6
    assert torch.isclose(centers[0, 0], raw[:13].mean())
    assert torch.isclose(centers[0, 1], raw[:13].mean())
    assert torch.isclose(centers[17, 0], raw[12:25].mean())


def test_zero_abs_weight_denominator_fails_closed() -> None:
    weights = torch.zeros((257, 40000), dtype=torch.float32)
    with pytest.raises(audit.RWSAHarnessFailure, match="denominator"):
        audit._structural_metrics(weights)


def test_nonfinite_abs_weight_denominator_fails_closed() -> None:
    weights = _synthetic_weights("broad")
    weights[0, 0] = float("nan")
    with pytest.raises(audit.RWSAHarnessFailure, match="non-finite"):
        audit._structural_metrics(weights)


def test_structural_metrics_local_fixture_is_concentrated() -> None:
    report = audit._structural_metrics(_synthetic_weights("local"))
    assert report["expected_support_locality_mass"]["mean"] > 0.95
    assert report["effective_support_size"]["mean"] < 1.1
    assert report["normalized_weight_entropy"]["mean"] < 0.01


def test_structural_metrics_broad_fixture_is_broad() -> None:
    report = audit._structural_metrics(_synthetic_weights("broad"))
    assert report["expected_support_locality_mass"]["mean"] < 0.02
    assert report["effective_support_size"]["mean"] == pytest.approx(256.0)
    assert report["normalized_weight_entropy"]["mean"] == pytest.approx(1.0)


def test_d4_coordinate_controls_are_fixed() -> None:
    assert audit.D4_CONTROLS == (
        "identity", "x_flip", "y_flip", "xy_flip", "transpose",
        "transpose_x_flip", "transpose_y_flip", "transpose_xy_flip",
    )


def test_d4_coordinate_controls_transform_coordinates() -> None:
    rows = torch.tensor([2, 5], dtype=torch.long)
    columns = torch.tensor([7, 9], dtype=torch.long)
    identity_rows, identity_columns = audit._transform_coordinates(rows, columns, "identity")
    assert torch.equal(identity_rows, rows)
    assert torch.equal(identity_columns, columns)
    assert audit._transform_coordinates(rows, columns, "x_flip")[1].tolist() == [192, 190]
    assert audit._transform_coordinates(rows, columns, "y_flip")[0].tolist() == [197, 194]
    assert audit._transform_coordinates(rows, columns, "transpose")[0].tolist() == [7, 9]


def test_permutation_null_count_is_fixed() -> None:
    assert audit.PERMUTATION_NULL_COUNT == 256
    assert audit.PERMUTATION_NULL["count"] == 256


def test_permutation_null_seed_is_fixed() -> None:
    assert audit.PERMUTATION_NULL_SEED == 0
    assert audit.PERMUTATION_NULL["seed"] == 0


def test_permutation_bank_is_deterministic() -> None:
    assert torch.equal(audit._permutation_bank(), audit._permutation_bank())


def test_permutation_bank_preserves_weight_marginals() -> None:
    generator = torch.Generator(device="cpu").manual_seed(23)
    weights = torch.randn((256, 40000), generator=generator)
    permutation = audit._permutation_bank()[17]
    permuted = audit._apply_source_row_permutation(weights, permutation)
    assert torch.equal(weights.abs().sort(dim=0).values, permuted.abs().sort(dim=0).values)


def test_null_computation_does_not_mutate_weights() -> None:
    weights = _synthetic_weights("local")
    before = weights.clone()
    report = audit._structural_metrics(weights)
    assert report["permutation_null"]["count"] == 256
    assert torch.equal(weights, before)


def test_full_dynamic_bias_decomposition_is_exact() -> None:
    generator = torch.Generator(device="cpu").manual_seed(7)
    full_weights = torch.randn((257, 40000), generator=generator)
    probe = audit.ReadoutProbe(full_weights, full_weights[:-1], full_weights[-1], torch.arange(1000), {}, "probe")
    features = torch.randn((3, 256), generator=generator)
    full, dynamic, bias = audit._decompose_readout(features, probe)
    assert torch.equal(dynamic, features.mm(probe.weights))
    assert torch.equal(bias, probe.bias.unsqueeze(0).expand(3, -1))
    assert torch.equal(full, dynamic + bias)


def test_dynamic_is_z_times_w_without_bias() -> None:
    generator = torch.Generator(device="cpu").manual_seed(8)
    full_weights = torch.randn((257, 40000), generator=generator)
    probe = audit.ReadoutProbe(full_weights, full_weights[:-1], full_weights[-1], torch.arange(1000), {}, "probe")
    features = torch.randn((2, 256), generator=generator)
    _, dynamic, _ = audit._decompose_readout(features, probe)
    assert torch.equal(dynamic, features.float().mm(full_weights[:-1]))


def test_bias_is_broadcast_frozen_b() -> None:
    generator = torch.Generator(device="cpu").manual_seed(9)
    full_weights = torch.randn((257, 40000), generator=generator)
    probe = audit.ReadoutProbe(full_weights, full_weights[:-1], full_weights[-1], torch.arange(1000), {}, "probe")
    _, _, bias = audit._decompose_readout(torch.zeros((4, 256)), probe)
    assert torch.equal(bias[0], full_weights[-1])
    assert torch.equal(bias[0], bias[3])


def test_no_refit_or_postprocessing_contract() -> None:
    assert audit.BIAS_ATTRIBUTION["refit"] is False
    assert audit.BIAS_ATTRIBUTION["postprocessing"] == "none"
    source = inspect.getsource(audit._decompose_readout)
    assert "pinv" not in source
    assert "clamp" not in source


def test_bias_constant_output_degeneracy_is_explicit() -> None:
    assert audit.BIAS_ATTRIBUTION["instance_constant_label"] == "not_applicable_degenerate_constant_output"
    assert audit._bias_metrics(
        torch.ones(10), torch.ones(10), torch.ones(10), torch.arange(10) % 10, torch.ones(40000)
    )["instance_sensitive_metrics"] == "not_applicable_degenerate_constant_output"


def test_exact_rmal_panels_are_locked() -> None:
    assert audit.EVALUATION_PANELS["selection"]["source"] == "mnist_test_indices_5000_9999"
    assert audit.EVALUATION_PANELS["confirmation"]["source"] == "horizon_heldout_train_indices_55000_59999"
    assert audit.EVALUATION_PANELS["candidate_prefix"]["consumed"] is False
    assert audit.MEASUREMENT["selection_count"] == 1000
    assert audit.MEASUREMENT["confirmation_count"] == 5000


def test_physical_anchor_contract_is_inherited() -> None:
    assert audit.PHYSICAL_ANCHORS["selection"]["paired_mse"] == pytest.approx(0.06204846128821373)
    assert audit.PHYSICAL_ANCHORS["confirmation"]["paired_mse"] == pytest.approx(0.05778951570391655)
    assert audit.PHYSICAL_BASELINE_TOLERANCE == 1.0e-6


def test_no_update_boundary_is_visible_in_source() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "evaluator_calls" in source
    assert "physical_forward_modified" in source
    assert audit.AUTHORIZATION["mechanism_evidence_only"] is True


def test_cli_actions_are_registered_and_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args([
        "research", "optical-manifold", "scaled-chen-detector-readout-structure-attribution-validate",
        "--config", "config.yaml", "--review-contract", "review.json", "--dataset-root", "data", "--output", "validation.json",
    ])
    run = parser.parse_args([
        "research", "optical-manifold", "scaled-chen-detector-readout-structure-attribution-run",
        "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json",
        "--dataset-root", "data", "--output", "audit.json",
    ])
    assert validate.research_action.endswith("detector-readout-structure-attribution-validate")
    assert run.device == "cuda"
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_detector_readout_structure_attribution" in router
    assert "import optical_manifold_smf_bogt_detector_readout_structure_attribution" not in router


def test_invalid_probe_consumer_fails_closed() -> None:
    payload = {"schema": audit.RMAL_PROBE_SCHEMA, "fit_indices": torch.arange(1000), "probes": {}, "fit_count": 1000, "mechanism_evidence_only": True, "physical_forward_modified": False}
    with pytest.raises(ValueError, match="fields|entry"):
        audit._load_rmal_probe(payload, torch.arange(1000), "probe")


def test_invalid_decomposition_features_fail_closed() -> None:
    probe = _probe()
    with pytest.raises(ValueError, match="features"):
        audit._decompose_readout(torch.zeros((1, 255)), probe)


def test_success_classification_is_descriptive_only() -> None:
    assert audit.CLASSIFICATION == "descriptive_structural_attribution_completed"
    assert audit.NEXT_ACTION_RESULT_REVIEW == "manual_review_RWSA_structural_result"
    assert audit.AUTHORIZATION["training_authorized"] is False
    assert audit.AUTHORIZATION["formal_promotion"] is False


def test_no_structural_significance_threshold_is_exposed() -> None:
    source = inspect.getsource(audit._structural_metrics)
    assert "p_value" not in source
    assert "threshold" not in source
    assert audit.PERMUTATION_NULL["statistic"] == "identity_expected_support_locality_mass"


def test_no_raw_or_large_feature_bank_artifact_is_declared() -> None:
    assert audit.MEASUREMENT["no_raw_cmos_bank"] is True
    assert audit.MEASUREMENT["no_large_feature_bank_artifact"] is True
    source = inspect.getsource(audit._panel_replay)
    assert "output.camera" in source
    assert "return_trace=False" in source


def test_structural_summary_contains_required_families() -> None:
    report = audit._structural_metrics(_synthetic_weights("broad"))
    for key in (
        "argmax_source_coarse_cell", "absolute_weight_centroid", "centroid_displacement",
        "top_1_mass", "top_k_mass", "expected_support_locality_mass",
        "coarse_radius_locality_mass", "spatial_variance_m2", "rms_support_radius_um",
        "effective_support_size", "normalized_weight_entropy", "source_row_concentration",
        "source_column_concentration", "identity_coordinate_consistency", "d4_control_consistency",
        "translation_field", "locality_vs_target_spatial_position", "positive_weight_mass",
        "negative_weight_mass", "signed_cancellation", "bias_spatial_summary", "permutation_null",
    ):
        assert key in report
