from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scripts.research.lightgen.execute_lightgen_c45k_phase_chart_ab_v1 import (
    ExperimentError,
    GENERATION_FIELDS,
    _canonicalize_template,
    _assert_saved_fits_match,
    _numpy_template_and_transform,
    _output_identity,
    _template_and_transform,
    _transform_with_template,
    _validate_generation_rows,
    _validate_new_cohort,
    _write_json,
)


def _features(values: np.ndarray) -> torch.Tensor:
    pairs = np.stack((values.real, values.imag), axis=-1)
    return torch.from_numpy(pairs.reshape(values.shape[0], 200).copy()).to(torch.float64)


def _row() -> dict[str, object]:
    row: dict[str, object] = {name: 0 for name in GENERATION_FIELDS}
    row.update({"representation": "template", "prior": "G4", "capture_power_w": 0.1,
                "raw_mean_w_per_pixel": 0.01, "raw_peak_w_per_pixel": 0.02,
                "view_mean_intensity": 0.5, "proxy_clipping_area_fraction": 0.0,
                "proxy_clipped_excess_fraction": 0.0})
    return row


def test_template_transform_matches_independent_numpy_and_removes_common_phase() -> None:
    rng = np.random.default_rng(20261023)
    latent = rng.normal(size=(160, 1)) + 1j * rng.normal(size=(160, 1))
    direction = rng.normal(size=100) + 1j * rng.normal(size=100)
    direction /= np.linalg.norm(direction)
    values = 3.0 * latent * direction[None, :] + 0.15 * (
        rng.normal(size=(160, 100)) + 1j * rng.normal(size=(160, 100))
    )
    features = _features(values)

    record, transformed = _template_and_transform(features, torch.device("cpu"))
    reference_np, transformed_np, independent = _numpy_template_and_transform(features)
    transformed_complex = transformed

    assert np.allclose(record["reference"].numpy(), reference_np, rtol=1e-10, atol=1e-12)
    assert np.allclose(transformed_complex.numpy(), transformed_np, rtol=1e-10, atol=1e-12)
    assert math.isclose(record["top_eigenvalue_relative_gap"],
                        independent["top_eigenvalue_relative_gap"], rel_tol=1e-10, abs_tol=1e-12)
    projections = transformed_complex @ record["reference"].conj()
    assert torch.all(projections.real > 0)
    assert torch.max(projections.imag.abs()) <= 1e-12

    rotation = torch.polar(torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.731, dtype=torch.float64))
    rotated = torch.view_as_complex(features.reshape(-1, 100, 2).contiguous()) * rotation
    rotated_features = torch.view_as_real(rotated).reshape(-1, 200)
    charted_again = _transform_with_template(rotated_features, record["reference"], torch.device("cpu"))
    charted_again_complex = torch.view_as_complex(charted_again.reshape(-1, 100, 2).contiguous())
    assert torch.allclose(transformed, charted_again_complex, rtol=1e-10, atol=1e-12)


def test_template_canonicalization_uses_first_tied_maximum_and_zero_projection_fails() -> None:
    reference = torch.zeros(100, dtype=torch.complex128)
    reference[0] = 1.0j
    reference[1] = -1.0j
    canonical, channel, tied = _canonicalize_template(reference)
    assert channel == 0
    assert tied
    assert canonical[0].real > 0 and abs(float(canonical[0].imag)) <= 1e-12

    zero_projection = torch.zeros((1, 100), dtype=torch.complex128)
    zero_projection[0, 1] = 1.0
    with pytest.raises(ExperimentError, match="zero or near-zero template projection"):
        _transform_with_template(torch.view_as_real(zero_projection).reshape(1, 200),
                                 torch.eye(100, dtype=torch.complex128)[0], torch.device("cpu"))


def test_generation_consumers_reject_wrong_endpoint_duplicate_and_mixed_rows() -> None:
    wrong_endpoint = {"task_id": "LIGHTGEN_C45K_PHASE_CHART_AB_V1", "source_global_step": 44_999}
    with pytest.raises(ExperimentError, match="endpoint, representation, method or stream"):
        _validate_new_cohort(wrong_endpoint, "G4", 0)

    row = _row()
    _validate_generation_rows([row])
    with pytest.raises(ExperimentError, match="duplicate generation row"):
        _validate_generation_rows([row, row.copy()])
    missing_metric = row.copy()
    missing_metric.pop("proxy_clipping_area_fraction")
    with pytest.raises(ExperimentError, match="missing, extra, or mixed statistic fields"):
        _validate_generation_rows([missing_metric])


def test_preflight_recovery_accepts_only_the_incomplete_engineering_failure(tmp_path) -> None:
    semantic = {"task_id": "LIGHTGEN_C45K_PHASE_CHART_AB_V1", "fixed": True}
    _write_json(tmp_path / "run_config.json", semantic)
    failure = {
        "task_id": "LIGHTGEN_C45K_PHASE_CHART_AB_V1",
        "status": "ENGINEERING_INVALID",
        "source_global_step": 45_000,
        "scientific_result": None,
        "error_type": "ValueError",
        "error": "incomplete preflight failure",
    }
    _write_json(tmp_path / "engineering_error.json", failure)
    _write_json(tmp_path / "engineering_error_0002.json", {**failure, "error": "second incomplete preflight failure"})
    (tmp_path / "resource_telemetry.csv").write_text("timestamp\n", encoding="utf-8")

    assert _output_identity(tmp_path, semantic, resume=True,
                            recover_incomplete_preflight=True) == failure
    assert (tmp_path / "engineering_error.json").is_file()
    assert (tmp_path / "engineering_error_0002.json").is_file()

    template_path = tmp_path / "phase_templates_and_assignments.pt"
    prior_path = tmp_path / "template_priors_G1_G4.pt"
    template_path.write_bytes(b"preserve-template")
    prior_path.write_bytes(b"preserve-prior")
    assert _output_identity(tmp_path, semantic, resume=True,
                            recover_incomplete_preflight=True) == failure

    prior_path.unlink()
    with pytest.raises(ExperimentError, match="partial scientific artifacts"):
        _output_identity(tmp_path, semantic, resume=True,
                         recover_incomplete_preflight=True)
    prior_path.write_bytes(b"preserve-prior")
    (tmp_path / "phase_templates.pt").write_bytes(b"unexpected")
    with pytest.raises(ExperimentError, match="partial scientific artifacts"):
        _output_identity(tmp_path, semantic, resume=True,
                         recover_incomplete_preflight=True)


def test_saved_fit_recovery_consumer_checks_all_fixed_classes() -> None:
    templates: dict[int, dict[str, object]] = {}
    saved_templates: dict[int, dict[str, object]] = {}
    assignments: dict[int, torch.Tensor] = {}
    fitted: dict[str, dict[int, dict[str, object]]] = {"G1": {}, "G4": {}}
    saved_fits: dict[int, dict[str, dict[str, object]]] = {}
    for label in range(10):
        reference = torch.full((100,), 0.1 + 0j, dtype=torch.complex128)
        eigenvalues = torch.full((100,), float(label), dtype=torch.float64)
        template = {"reference": reference, "eigenvalues": eigenvalues,
                    "top_eigenvalue_relative_gap": 0.5,
                    "minimum_normalized_reference_projection": 0.1,
                    "canonical_reference_channel": 0,
                    "canonical_reference_exact_tie": False}
        templates[label] = template
        saved_templates[label] = template.copy()
        assignments[label] = torch.tensor([0, 1], dtype=torch.long)
        g1 = {"marker": torch.tensor(float(label), dtype=torch.float64)}
        g4 = {"marker": torch.tensor(float(label), dtype=torch.float64),
              "assignments": assignments[label]}
        fitted["G1"][label] = g1
        fitted["G4"][label] = g4
        saved_fits[label] = {"G1": g1.copy(), "G4": g4.copy()}

    _assert_saved_fits_match(templates, fitted, assignments, saved_templates, saved_fits)
    saved_fits[0]["G1"]["marker"] = torch.tensor(-1.0, dtype=torch.float64)
    with pytest.raises(ExperimentError, match="saved G1 recovery changed class 0 marker"):
        _assert_saved_fits_match(templates, fitted, assignments, saved_templates, saved_fits)



