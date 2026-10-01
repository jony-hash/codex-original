from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt
from ldm_project.v2.evidence import file_sha256
from scripts.research.scaled_chen.repair_scaled_chen_calibration_unit_aware import R2_CALIBRATION_SCHEMA, build_repaired_calibration_payload, calibration_z_sanity, load_registered_unit_aware_calibration, next_write_once_output


def _calibration() -> bogt.UnitAwareCalibration:
    raw = torch.linspace(1.0e-12, 6.0e-10, bogt.PORT_COUNT, dtype=torch.float32)
    raw[1] = 1.0e-10
    plane = torch.ones(5, dtype=torch.float32)
    return bogt.build_unit_aware_calibration(raw, plane, sample_count=1000)


def _metadata() -> dict[str, object]:
    return {
        "task_id": "H-LOMA-GEN-P1-R2-UACR",
        "contract_version": "unit_aware_raw_mean_v2",
        "dataset_manifest_hash": "dataset-hash",
        "calibration_indices_sha256": "indices-hash",
        "zero_phase_reference": "independent_train_only_zero_phase",
        "mode_bank_sha256": "mode-bank-hash",
        "plane_identity": "pre_ols_plane_power_reference_v1",
        "raw_scale_dtype": "torch.float32",
        "registered_scale_dtype": "torch.float32",
        "raw_scale_statistics": {"count": 100, "finite": True, "positive": True},
        "floor_policy": "none_for_positive_finite_raw_means",
        "parent_v1_artifact_sha256": "parent-v1-hash",
        "parent_v1_schema": "scaled_chen_bogt_calibration.v1",
        "source_c6_manifest_sha256": "c6-manifest-hash",
        "source_c6_payload_sha256": "c6-payload-hash",
        "producer_git": {"branch": bogt.BRANCH, "measurement_commit": "measurement"},
        "prior_contract": {"kind": "truncated_normal", "mean": 1.0, "variance": 0.25, "lower": 0.0, "unconditional": True},
        "optical_forward_dtype": "torch.float32/torch.complex64",
    }


def test_positive_sub_eps_scale_is_preserved_without_absolute_floor() -> None:
    calibration = _calibration()

    assert calibration.port_scale[0].item() == pytest.approx(1.0e-12, rel=1.0e-6)
    assert calibration.port_scale[1].item() == pytest.approx(1.0e-10, rel=1.0e-6)
    assert torch.equal(calibration.raw_port_mean, calibration.port_scale)
    assert torch.unique(calibration.port_scale).numel() == bogt.PORT_COUNT
    assert not torch.all(calibration.port_scale == torch.finfo(torch.float32).eps)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_unit_aware_calibration_rejects_nonpositive_or_nonfinite_raw_means(
    bad_value: float,
) -> None:
    raw = torch.ones(bogt.PORT_COUNT, dtype=torch.float32)
    raw[0] = bad_value

    with pytest.raises(ValueError):
        bogt.build_unit_aware_calibration(raw, torch.ones(5), sample_count=1000)


def test_raw_reference_z_sanity_preserves_global_and_per_port_mean() -> None:
    calibration = _calibration()
    powers = calibration.raw_port_mean.unsqueeze(0).repeat(1000, 1)

    sanity = calibration_z_sanity(powers, calibration.port_scale)

    assert sanity["finite"] is True
    assert sanity["global_mean"] == pytest.approx(1.0)
    assert min(sanity["per_port_mean"]) == pytest.approx(1.0)
    assert max(sanity["per_port_mean"]) == pytest.approx(1.0)


def test_v2_payload_roundtrips_only_with_explicit_bindings(tmp_path: Path) -> None:
    target = tmp_path / "repaired_calibration.pt"
    payload = build_repaired_calibration_payload(_calibration(), metadata=_metadata())
    torch.save(payload, target)

    loaded = load_registered_unit_aware_calibration(
        target,
        expected_sha256=file_sha256(target),
        expected_dataset_manifest_hash="dataset-hash",
        expected_calibration_indices_sha256="indices-hash",
        expected_mode_bank_sha256="mode-bank-hash",
        expected_parent_v1_sha256="parent-v1-hash",
    )

    assert loaded.schema == R2_CALIBRATION_SCHEMA
    assert torch.equal(loaded.raw_port_mean, loaded.port_scale)
    assert loaded.sample_count == 1000


def test_v2_consumer_rejects_historical_v1_payload(tmp_path: Path) -> None:
    target = tmp_path / "historical_v1.pt"
    torch.save(
        {
            "schema": "scaled_chen_bogt_calibration.v1",
            "port_scale": torch.ones(bogt.PORT_COUNT),
            "plane_reference": torch.ones(5),
            "plane_lower": torch.full((5,), 0.8),
            "plane_upper": torch.full((5,), 1.2),
            "sample_count": 1000,
            "dataset_manifest_hash": "dataset-hash",
        },
        target,
    )

    with pytest.raises(ValueError, match="scaled_chen_bogt_calibration.v2"):
        load_registered_unit_aware_calibration(
            target,
            expected_sha256=file_sha256(target),
            expected_dataset_manifest_hash="dataset-hash",
            expected_calibration_indices_sha256="indices-hash",
            expected_mode_bank_sha256="mode-bank-hash",
            expected_parent_v1_sha256="parent-v1-hash",
        )


def test_v2_consumer_rejects_mismatched_lineage_binding(tmp_path: Path) -> None:
    target = tmp_path / "repaired_calibration.pt"
    torch.save(build_repaired_calibration_payload(_calibration(), metadata=_metadata()), target)

    with pytest.raises(ValueError, match="dataset manifest"):
        load_registered_unit_aware_calibration(
            target,
            expected_sha256=file_sha256(target),
            expected_dataset_manifest_hash="wrong-dataset-hash",
            expected_calibration_indices_sha256="indices-hash",
            expected_mode_bank_sha256="mode-bank-hash",
            expected_parent_v1_sha256="parent-v1-hash",
        )


def test_historical_v1_reader_contract_remains_readable() -> None:
    calibration = bogt.Calibration(
        port_scale=torch.ones(bogt.PORT_COUNT),
        plane_reference=torch.ones(5),
        plane_lower=torch.full((5,), 0.8),
        plane_upper=torch.full((5,), 1.2),
        sample_count=1000,
    )
    calibration.validate()


def test_write_once_output_never_reuses_existing_attempt(tmp_path: Path) -> None:
    base = tmp_path / "lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_01"
    first = next_write_once_output(base)
    first.mkdir()
    second = next_write_once_output(base)

    assert second.name.endswith("attempt_02")
    assert second != first
