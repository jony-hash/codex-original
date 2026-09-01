from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research import (
    optical_manifold as base,
    optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit as audit,
)


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / (
    "configs/research/"
    "lightgen_optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit_v1.yaml"
)
REVIEW = ROOT / (
    "configs/research/"
    "lightgen_optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit_review_v1.json"
)


def _fixture_w() -> torch.Tensor:
    return torch.tensor(
        [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def _fixture_spectrum() -> audit.SpectralSummary:
    return audit._spectrum_from_gram(audit._gram_matrix(_fixture_w()), "fixture")


def test_config_review_and_identity_contracts_are_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec) == REVIEW
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert review["hypothesis"] == audit.HYPOTHESIS
    assert review["approved_contract"]["mode_preview"]["k"] == 16


def test_registered_upstream_identity_and_probe_hash() -> None:
    spec = audit._load_spec(CONFIG)
    upstream = audit._upstream_bundle(spec)
    assert upstream["rmal"]["compact"].get("generation_readiness") == "not_ready"
    assert upstream["rwsa"]["audit"]["classification"] == "invalid_harness"
    assert upstream["zcsa"]["audit"]["classification"] == audit.ZCSA_IDENTITY["audit_classification"]
    assert upstream["rwsa_active"]["audit"]["classification"] == audit.RWSA_ACTIVE_IDENTITY["audit_classification"]
    probe, fit_indices, path = audit._load_probe()
    identity = audit._probe_identity(probe)
    assert path == ROOT / audit.RMAL_CONSUMER["probe"]
    assert fit_indices.numel() == audit.FIT_COUNT
    assert identity["full_shape"] == [257, 40000]
    assert identity["W_shape"] == [256, 40000]
    assert identity["bias_shape"] == [40000]
    assert identity["storage_dtype"] == "torch.float32"
    assert identity["source_order"] == "row-major i_then_j"
    assert identity["target_order"] == "row-major r_then_c"
    assert identity["active_count"] == 33502
    assert identity["inactive_count"] == 6498
    assert identity["active_mask_sha256"] == audit.ACTIVE_MASK_SHA256
    assert identity["inactive_mask_sha256"] == audit.INACTIVE_MASK_SHA256


def test_source_float32_is_immutable_and_analysis_is_float64() -> None:
    probe, _, _ = audit._load_probe()
    identity = audit._probe_identity(probe)
    before = audit.tensor_sha256(identity["full"])
    W64, full, active, _ = audit._spectral_analysis(
        identity["W"], identity["active_mask"]
    )
    assert identity["full"].dtype == torch.float32
    assert W64.dtype == torch.float64
    assert W64.device.type == "cpu"
    assert full.dimension == active.dimension == 256
    assert audit.tensor_sha256(identity["full"]) == before


def test_gram_matches_direct_svd_and_zero_columns_preserve_spectrum() -> None:
    W = _fixture_w()
    summary = _fixture_spectrum()
    direct = torch.linalg.svdvals(W)
    assert torch.allclose(summary.singular_values, direct)
    padded = torch.cat((W, torch.zeros((3, 2), dtype=torch.float64)), dim=1)
    mask = torch.tensor([True, True, True, False, False])
    full = audit._spectrum_from_gram(audit._gram_matrix(padded), "full")
    active = audit._spectrum_from_gram(audit._gram_matrix(padded, mask), "active")
    assert torch.allclose(full.singular_values, active.singular_values)


def test_psd_tolerance_sorts_and_clamps_only_tiny_negative_values() -> None:
    tiny = torch.diag(torch.tensor([1.0, -1.0e-15, 0.0], dtype=torch.float64))
    result = audit._spectrum_from_gram(tiny)
    assert result.psd["negative_eigenvalue_count"] == 1
    assert result.psd["clamped_tiny_negative_count"] == 1
    assert result.psd["maximum_negative_magnitude"] == pytest.approx(1.0e-15)
    assert torch.equal(result.singular_values, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    material = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64))
    with pytest.raises(audit.LSDAHarnessFailure, match="material negative"):
        audit._spectrum_from_gram(material)


def test_spectrum_arrays_cumulative_effective_stable_and_condition_profile() -> None:
    result = _fixture_spectrum()
    assert result.singular_values.tolist() == [3.0, 2.0, 1.0]
    assert torch.all(result.cumulative_energy[1:] >= result.cumulative_energy[:-1])
    assert result.effective_rank == pytest.approx(
        float(torch.exp(-(result.normalized_energy * result.normalized_energy.clamp_min(1e-30).log()).sum()))
    )
    assert result.stable_rank == pytest.approx(14.0 / 9.0)
    assert result.condition_number == pytest.approx(3.0)
    assert len(result.compact()["sigma_over_sigma1"]) == 3
    assert len(result.compact()["adjacent_relative_gaps"]) == 2


def test_mode_preview_gap_and_basis_dependence_are_frozen() -> None:
    assert audit.MODE_PREVIEW_K == 16
    assert audit.MODE_PREVIEW["basis_dependent_mode_descriptor"] is True
    assert audit.MODE_PREVIEW["no_numerical_rank_cutoff"] is True
    assert audit.SPECTRAL_CONTRACT["numerical_rank_classification"] is False
    assert audit.SPECTRAL_CONTRACT["randomized_svd"] is False


def test_sign_canonicalization_uses_lowest_row_major_tie() -> None:
    source = torch.tensor([[-2.0, 1.0], [2.0, -3.0]], dtype=torch.float64)
    target = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64)
    canonical, paired, indices, signs = audit._canonicalize_source_modes(source, target)
    assert indices == [0, 1]
    assert signs == [-1, -1]
    assert canonical[0, 0] > 0 and canonical[1, 1] > 0
    assert torch.equal(paired[:, 0], -target[:, 0])
    assert audit.SIGN_CANONICALIZATION["name"] == "largest_abs_source_entry_positive"


def test_frequency_coordinates_and_fixed_radial_bands() -> None:
    fx, fy, rho = audit._frequency_coordinates((4, 4))
    assert fx.tolist() == [-0.5, -0.25, 0.0, 0.25]
    assert fy.tolist() == [-0.5, -0.25, 0.0, 0.25]
    assert float(rho.min()) == pytest.approx(0.0)
    mode = torch.zeros((4, 4), dtype=torch.float64)
    mode[0, 0] = 1.0
    bands = audit._radial_fft_energy(mode)
    assert set(bands["fractions"]) == {"low", "mid", "high"}
    assert sum(bands["fractions"].values()) == pytest.approx(1.0)
    assert bands["fixed_bands"]["low"] == [0.0, 0.25]


def test_spatial_masks_and_mode_metrics_use_fixed_coordinates() -> None:
    coordinates = audit._pixel_coordinate_grid((16, 16), 2.5e-6)
    _, masks = audit._spatial_masks(coordinates)
    assert bool(torch.stack(tuple(masks.values())).sum(dim=0).eq(1).all())
    metrics = audit._mode_metrics(
        torch.ones((16, 16), dtype=torch.float64),
        mode_index=0,
        domain="source",
        coordinates=coordinates,
    )
    assert metrics["reshape"] == [16, 16]
    assert metrics["basis_dependent_mode_descriptor"] is True
    assert set(metrics["spatial"]["squared_mode_energy_fractions"]) == {
        "center",
        "middle",
        "edge",
    }
    assert metrics["spatial"]["participation_size"] == pytest.approx(256.0)


def test_source_coordinates_and_target_reshape_contract() -> None:
    assert audit._source_coordinate_grid().shape == (16, 16, 2)
    assert audit._pixel_coordinate_grid((200, 200), 2.5e-6).shape == (200, 200, 2)
    source = torch.zeros((16, 16), dtype=torch.float64)
    target = torch.zeros((200, 200), dtype=torch.float64)
    assert audit._mode_metrics(
        source, mode_index=0, domain="source", coordinates=audit._source_coordinate_grid()
    )["reshape"] == [16, 16]
    assert audit._mode_metrics(
        target, mode_index=0, domain="target", coordinates=audit._pixel_coordinate_grid((200, 200), 2.5e-6)
    )["reshape"] == [200, 200]


def test_d4_residuals_are_sign_invariant_and_orthogonality_is_relative() -> None:
    symmetric = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    residuals = audit._d4_sign_invariant_residual(symmetric)
    assert residuals["identity"] == pytest.approx(0.0)
    assert residuals["transpose"] == pytest.approx(0.0)
    assert audit._d4_sign_invariant_residual(-symmetric)["identity"] == pytest.approx(0.0)
    assert audit._orthogonality_residual(torch.eye(4, dtype=torch.float64)) == pytest.approx(0.0)
    assert audit._orthogonality_residual(torch.zeros((4, 0), dtype=torch.float64)) == 0.0


def test_target_modes_follow_v_equals_w_transpose_u_over_sigma_and_zero_safe() -> None:
    W = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    spectrum = audit._spectrum_from_gram(audit._gram_matrix(W))
    target, available = audit._target_modes(W, spectrum, spectrum.eigenvectors)
    assert available == [0, 1]
    expected = W.T @ spectrum.eigenvectors[:, :2]
    expected = expected / spectrum.singular_values[:2]
    assert torch.allclose(target[:, :2], expected)
    assert target.shape == (2, 3)
    assert audit._target_modes(W, spectrum, spectrum.eigenvectors)[0].shape[0] != 40000


def test_global_mean_is_streaming_renderer_output_without_labels_or_eval_panels() -> None:
    images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
    images[1, 0, 10:18, 10:18] = 1.0
    registered_images = images.repeat((500, 1, 1, 1))
    mean, summary = audit._global_target_mean(registered_images)
    expected = base.render_intensity_batch(images).target.reshape(2, -1).double().mean(dim=0)
    assert torch.equal(mean, expected)
    assert summary["count"] == 1000
    assert summary["target_bank_saved"] is False
    assert summary["labels_used"] is False
    assert summary["selection_used"] is False
    assert summary["confirmation_used"] is False
    source = inspect.getsource(audit._global_target_mean)
    assert "labels.index_select" not in source


def test_global_mean_and_bias_projections_are_descriptive_only() -> None:
    modes = torch.eye(4, dtype=torch.float64)
    mean = torch.tensor([1.0, 2.0, 0.0, 0.0], dtype=torch.float64)
    projection = audit._global_mean_projections(modes, [0, 1, 2, 3], mean)
    assert projection["class_template_reference"] == "not_evaluated"
    assert projection["modes"][0]["dot"] == pytest.approx(1.0)
    bias = audit._bias_projection(modes, [0, 1, 2, 3], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert bias["refit"] is False
    assert len(bias["coefficients"]) == 16
    assert bias["residual_bias_energy_outside_preview"] == pytest.approx(0.0)
    zero_bias = audit._bias_projection(modes, [0, 1, 2, 3], torch.zeros(4))
    assert zero_bias["zero_bias_descriptor"] == "not_applicable_zero_bias"


def test_no_model_update_decoder_physical_or_rank_truncated_path_is_present() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "_load_scaled_model" not in source
    assert "ScaledChenBOGTModel" not in source
    assert audit.MEASUREMENT["no_training"] is True
    assert audit.MEASUREMENT["no_optimizer"] is True
    assert audit.MEASUREMENT["no_backward"] is True
    assert audit.MEASUREMENT["no_probe_refit"] is True
    assert audit.MEASUREMENT["no_decoder"] is True
    assert audit.MEASUREMENT["no_physical_forward"] is True
    assert audit.MEASUREMENT["no_rank_truncated_reconstruction"] is True


def test_compact_schema_is_bounded_and_does_not_save_raw_mode_banks() -> None:
    source = inspect.getsource(audit)
    assert "40000, 40000" not in source
    assert "TOTAL_COLUMNS, TOTAL_COLUMNS" not in source
    assert audit.MAX_WORKING_SET_BYTES == 512 * 1024**2
    assert audit._estimated_working_set_bytes() < audit.MAX_WORKING_SET_BYTES
    assert audit._compact_report.__name__ == "_compact_report"
    assert audit._compact_report.__doc__ is None


def test_write_once_slot_and_lazy_cli_route() -> None:
    target = Path("temporary") / "validation.json"
    # The helper only creates the requested temporary slot; pytest's cwd is
    # not used so this remains a pure path-contract test.
    with pytest.raises(FileNotFoundError):
        audit._check_run_slot(target, target)
    # Use pytest's managed temporary directory for the write-once check.
    # (The first call creates the directory; the second call must reject it.)
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "attempt" / "validation.json"
        audit._prepare_validation_slot(path)
        with pytest.raises(FileExistsError):
            audit._prepare_validation_slot(path)
    parser = cli._parser()
    validate = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-linear-readout-structure-decomposition-validate",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--output",
            "validation.json",
        ]
    )
    run = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-linear-readout-structure-decomposition-run",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--validation-evidence",
            "validation.json",
            "--output",
            "audit.json",
        ]
    )
    assert validate.dataset_root == "data"
    assert not hasattr(validate, "device")
    assert not hasattr(run, "device")
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit" in router
    assert "import optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit" not in router
