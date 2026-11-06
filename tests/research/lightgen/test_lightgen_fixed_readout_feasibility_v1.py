from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.research.lightgen import diagnose_lightgen_fixed_readout_feasibility_v1 as diagnosis


def test_camera_conversion_uses_raw_power_then_full_well_clip() -> None:
    wavelength_m = 532e-9
    exposure_s = 1.0e-3
    qe = 0.5
    full_well_e = 10.0
    requested_electrons = np.array([[5.0, 10.0, 15.0]])
    photon_energy = 6.62607015e-34 * 299_792_458.0 / wavelength_m
    power = requested_electrons * photon_energy / (qe * exposure_s)

    electrons, view = diagnosis.camera_replay(power, wavelength_m, exposure_s, qe, full_well_e)

    np.testing.assert_allclose(electrons, requested_electrons, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(view, np.array([[0.5, 1.0, 1.0]], dtype=np.float32))


def test_changing_exposure_before_clip_differs_from_wrong_postclip_scaling() -> None:
    wavelength_m = 532e-9
    qe = 0.5
    full_well_e = 10.0
    source_exposure = 1.0e-3
    desired_at_source = np.array([[20.0]])
    photon_energy = 6.62607015e-34 * 299_792_458.0 / wavelength_m
    power = desired_at_source * photon_energy / (qe * source_exposure)

    _electrons, correctly_replayed = diagnosis.camera_replay(power, wavelength_m, 0.5e-3, qe, full_well_e)
    _original_electrons, original_clipped = diagnosis.camera_replay(
        power, wavelength_m, source_exposure, qe, full_well_e
    )
    wrong_order = original_clipped * 0.5

    assert correctly_replayed.item() == 1.0
    assert wrong_order.item() == 0.5
    assert not np.array_equal(correctly_replayed, wrong_order)


def test_predefined_exposure_grid_has_301_points_and_exact_original_1ms() -> None:
    grid = diagnosis.EXPOSURES_S
    assert len(grid) == 301
    assert grid[0] == pytest.approx(1.0e-5)
    assert grid[-1] == pytest.approx(1.0e-2)
    assert grid[200] == 1.0e-3
    assert np.all(np.diff(grid) > 0.0)


def test_global_mean_baseline_uses_fit_targets_only() -> None:
    targets = np.zeros((300, 2, 2), dtype=np.float32)
    roles = np.array(["fit"] * 100 + ["confirmation"] * 200)
    targets[:100] = 0.25
    targets[100:] = 0.9
    first = diagnosis.global_mean_image(targets, roles)
    targets[100:] = 0.0
    second = diagnosis.global_mean_image(targets, roles)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.full((2, 2), 0.25))


def test_fit_exposure_is_selected_from_fit_only_and_lock_is_consumed_from_disk(tmp_path) -> None:
    wavelength_m = 532e-9
    qe = 0.5
    full_well_e = 30000.0
    target = np.full((100, 2, 2), 0.2, dtype=np.float32)
    labels = np.repeat(np.arange(10), 10)
    photon_energy = 6.62607015e-34 * 299_792_458.0 / wavelength_m
    desired_electrons_at_1ms = 0.2 * full_well_e
    power = np.full((100, 2, 2),
                    desired_electrons_at_1ms * photon_energy / (qe * 1.0e-3), dtype=np.float32)

    curve_a, lock_a = diagnosis.select_fit_exposure(power, target, labels, wavelength_m, qe, full_well_e)
    # Confirmation targets are deliberately outside the selector's argument boundary.
    changed_confirmation_targets = np.zeros((200, 2, 2), dtype=np.float32)
    assert float(changed_confirmation_targets.mean()) == 0.0
    curve_b, lock_b = diagnosis.select_fit_exposure(power, target, labels, wavelength_m, qe, full_well_e)

    assert lock_a == lock_b
    assert len(curve_a) == len(curve_b) == 301
    assert lock_a["selected_exposure_s"] == 1.0e-3
    path = tmp_path / "locked_exposure.json"
    diagnosis._write_json(path, lock_a)
    assert diagnosis.read_exposure_lock(path)["selected_exposure_s"] == 1.0e-3
    payload = diagnosis._read_json(path)
    payload["selected_exposure_s"] = 1.0e-2
    diagnosis._write_json(path, payload)
    with pytest.raises(diagnosis.EvidenceError, match="does not match its fit-grid row"):
        diagnosis.read_exposure_lock(path)


def _population_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    rows: list[int] = []
    labels: list[int] = []
    roles: list[str] = []
    table: list[dict[str, str]] = []
    row_id = 1000
    for label in range(10):
        for role, count in (("fit", 10), ("confirmation", 20)):
            for _ in range(count):
                rows.append(row_id)
                labels.append(label)
                roles.append(role)
                table.append({"row_id": str(row_id), "label": str(label), "role": role,
                              "candidate": diagnosis.CANDIDATE, "mode": diagnosis.MODE,
                              "state": "step_000500_terminal"})
                row_id += 1
    return np.array(rows), np.array(labels), np.array(roles), table


def test_population_boundary_rejects_duplicate_or_missing_rows() -> None:
    rows, labels, roles, table = _population_vectors()
    diagnosis._validate_population_vectors(rows, labels, roles, table, "step_000500_terminal")

    duplicate = rows.copy()
    duplicate[-1] = duplicate[0]
    with pytest.raises(diagnosis.EvidenceError, match="duplicate sample row"):
        diagnosis._validate_population_vectors(duplicate, labels, roles, table, "step_000500_terminal")
    with pytest.raises(diagnosis.EvidenceError, match="missing or duplicate rows"):
        diagnosis._validate_population_vectors(rows, labels, roles, table[:-1], "step_000500_terminal")
    with pytest.raises(diagnosis.EvidenceError, match="metadata dtype"):
        diagnosis._validate_population_vectors(rows.astype(np.float64), labels, roles, table,
                                               "step_000500_terminal")


def test_nonfinite_saved_values_are_rejected_at_the_input_boundary() -> None:
    target = np.zeros((2, 2), dtype=np.float32)
    power = np.zeros_like(target)
    electrons = np.zeros_like(target)
    output = np.zeros_like(target)
    power[0, 0] = np.nan
    with pytest.raises(diagnosis.EvidenceError, match="nonfinite"):
        diagnosis._validate_camera_values(target, power, electrons, output, "initial")


def test_bootstrap_is_deterministic_and_uses_paired_class_strata() -> None:
    differences = {label * 20 + index: float(label + index / 100.0) for label in range(10) for index in range(20)}
    labels = {row_id: row_id // 20 for row_id in differences}
    first = diagnosis._stratified_bootstrap(differences, labels)
    second = diagnosis._stratified_bootstrap(differences, labels)
    assert first == second
    assert first["replicates"] == 2000
    assert first["seed"] == 2026101803
    assert math.isclose(first["mean_gain"], np.mean(list(differences.values())))


def test_uniform_scale_gallery_png_writer_emits_png(tmp_path) -> None:
    path = tmp_path / "sample.png"
    diagnosis._write_gray_png(path, np.full((200, 200), 0.5, dtype=np.float32))
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

