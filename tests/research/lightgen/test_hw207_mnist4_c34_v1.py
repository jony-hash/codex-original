"""Focused numerical contracts for the HW207 C3.4 diagnostic model."""

import json
import math

import numpy as np
import pytest
import torch

from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34
from scripts.research.lightgen import execute_hw207_mnist4_c34_v1 as c34_runner
from scripts.research.lightgen import audit_hw207_mnist4_c34_p2_v1 as c34_p2_audit


def _direct_fresnel(field, input_pitch_um, output_shape, output_pitch_um, distance_um,
                    wavelength_um=0.532):
    _, ny, nx = field.shape
    oy, ox = output_shape
    xin = (torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2) * input_pitch_um
    yin = (torch.arange(ny, dtype=torch.float64) - (ny - 1) / 2) * input_pitch_um
    xout = (torch.arange(ox, dtype=torch.float64) - (ox - 1) / 2) * output_pitch_um
    yout = (torch.arange(oy, dtype=torch.float64) - (oy - 1) / 2) * output_pitch_um
    xx, yy = torch.meshgrid(xin, yin, indexing="xy")
    xo, yo = torch.meshgrid(xout, yout, indexing="xy")
    result = torch.empty((oy, ox), dtype=torch.complex128)
    for row in range(oy):
        for col in range(ox):
            phase = math.pi / (wavelength_um * distance_um) * (
                (xo[row, col] - xx).square() + (yo[row, col] - yy).square()
            )
            result[row, col] = (-1j * input_pitch_um**2 / (wavelength_um * distance_um)) * (
                field[0] * torch.exp(1j * phase)
            ).sum()
    return result


def test_scaled_fresnel_czt_matches_independent_quadrature():
    generator = torch.Generator().manual_seed(34)
    field = torch.complex(torch.randn((1, 3, 4), generator=generator, dtype=torch.float64),
                          torch.randn((1, 3, 4), generator=generator, dtype=torch.float64))
    actual = c34.scaled_fresnel(field, 2.0, (4, 5), 3.5, 1000.0)
    expected = _direct_fresnel(field, 2.0, (4, 5), 3.5, 1000.0)
    assert torch.allclose(actual[0], expected, rtol=1e-11, atol=1e-14)


def test_high_precision_fresnel_chirps_reduce_large_phase_roundoff():
    generator = torch.Generator().manual_seed(348)
    real = torch.randn((1, 8, 8), generator=generator, dtype=torch.float32)
    imag = torch.randn((1, 8, 8), generator=generator, dtype=torch.float32)
    field = torch.complex(real, imag)
    arguments = (2.0, (16, 17), 1000.0, 10_000.0)
    fp32 = c34.scaled_fresnel(field, *arguments)
    precise_chirps = c34.scaled_fresnel(
        field, *arguments, phase_dtype=torch.complex128
    )
    fp64 = c34.scaled_fresnel(field.to(torch.complex128), *arguments)

    def intensity_error(actual, expected):
        actual_i = actual.abs().square().to(torch.float64)
        expected_i = expected.abs().square()
        return float(torch.linalg.vector_norm(actual_i - expected_i) /
                     torch.linalg.vector_norm(expected_i).clamp_min(1e-30))

    assert precise_chirps.dtype == torch.complex64
    assert intensity_error(precise_chirps, fp64) < intensity_error(fp32, fp64)


def test_complex_precision_promotion_preserves_imaginary_component():
    field = torch.tensor([[[1.0 + 2.5j, -0.5 - 3.0j]]], dtype=torch.complex64)
    promoted = c34.to_complex128(field)
    assert promoted.dtype == torch.complex128
    assert torch.equal(promoted.real, field.real.to(torch.float64))
    assert torch.equal(promoted.imag, field.imag.to(torch.float64))
    with pytest.raises(ValueError, match="field must be complex"):
        c34.to_complex128(field.real)


def test_mcnemar_trigger_uses_exact_one_sided_binomial_tail():
    model_a = torch.tensor([True, True, True, False])
    model_b = torch.tensor([False, False, True, True])
    result = c34_runner._mcnemar_worse_pvalue(model_a, model_b)
    assert result["A_correct_B_wrong"] == 2
    assert result["A_wrong_B_correct"] == 1
    assert result["one_sided_exact_binomial_pvalue_B_worse"] == pytest.approx(0.5)
    assert not result["trigger"]

    decisive = c34_runner._mcnemar_worse_pvalue(torch.ones(10, dtype=torch.bool),
                                                torch.zeros(10, dtype=torch.bool))
    assert decisive["one_sided_exact_binomial_pvalue_B_worse"] == pytest.approx(1 / 1024)
    assert decisive["trigger"]
    no_discordance = c34_runner._mcnemar_worse_pvalue(torch.ones(4, dtype=torch.bool),
                                                      torch.ones(4, dtype=torch.bool))
    assert no_discordance["one_sided_exact_binomial_pvalue_B_worse"] == 1.0


def test_training_preflight_gate_rejects_failed_random_label_control():
    updates = [(1.0, 0.2), (0.8, 0.1), (0.6, 0.05), (1.1, 0.2)]
    assert c34_runner._training_preflight_pass(0.3, updates, 0.4, 0.39, 0.7)
    assert not c34_runner._training_preflight_pass(0.3, updates, 0.4, 0.40, 0.7)
    assert not c34_runner._training_preflight_pass(float("nan"), updates, 0.4, 0.20, 0.7)


def test_fast_training_gate_requires_loss_drop_gradient_and_energy_retention():
    history = [
        {"train_loss": 1.2, "efficiency_fraction_of_zero_phase_validation": 0.5},
        {"train_loss": 0.8, "efficiency_fraction_of_zero_phase_validation": 0.12},
    ]
    assert c34_runner._fast_training_pass(history, [0.3, 0.2])["fast_training_pass"]
    history[-1]["efficiency_fraction_of_zero_phase_validation"] = 0.099
    assert not c34_runner._fast_training_pass(history, [0.3, 0.2])["fast_training_pass"]
    history[-1]["efficiency_fraction_of_zero_phase_validation"] = 0.5
    history[-1]["train_loss"] = 1.2
    assert not c34_runner._fast_training_pass(history, [0.3, 0.2])["fast_training_pass"]
    assert not c34_runner._fast_training_pass([], [0.3])["fast_training_pass"]


def test_resume_shuffle_generator_matches_uninterrupted_epoch_order():
    seed = 7
    train_count = 257
    completed_epochs = 4
    uninterrupted = torch.Generator().manual_seed(seed)
    torch.rand((100, 100), generator=uninterrupted)
    for _ in range(completed_epochs):
        torch.randperm(train_count, generator=uninterrupted)
    expected_next = torch.randperm(train_count, generator=uninterrupted)

    resumed = c34_runner._shuffle_generator_after_completed_epochs(
        seed, completed_epochs, train_count
    )
    actual_next = torch.randperm(train_count, generator=resumed)
    assert torch.equal(actual_next, expected_next)


def test_formal_input_field_cache_round_trip_and_role_row_validation(tmp_path, monkeypatch):
    images = torch.zeros((12, 28, 28), dtype=torch.uint8)
    for row in range(len(images)):
        images[row].fill_(row)

    def deterministic_input_field(batch, _configuration):
        real = batch[:, :1, :1].to(torch.float32).expand(-1, 100, 100)
        return torch.complex(real, torch.zeros_like(real))

    monkeypatch.setattr(c34, "input_field", deterministic_input_field)
    split = {"train": torch.tensor([1, 5, 9]),
             "validation": torch.tensor([3]),
             "holdout": torch.tensor([7])}
    rows, fields, metadata = c34_runner._prepare_field_cache(
        images, torch.cat(list(split.values())), object(), torch.device("cpu"), chunk_size=2
    )
    path = tmp_path / "input_field_cache.pt"
    torch.save({"identity": "C34_FIXED_BINARY_INPUT_FIELD_CACHE", "rows": rows,
                "fields": fields, "metadata": metadata}, path)
    loaded_rows, loaded_fields, loaded_metadata = c34_runner._load_or_prepare_formal_field_cache(
        images, split, object(), torch.device("cpu"), path
    )
    assert torch.equal(loaded_rows, rows)
    assert torch.equal(loaded_fields, fields)
    assert loaded_metadata["loaded_from_system_disk"]
    assert torch.equal(c34_runner._fields_for_rows((loaded_rows, loaded_fields), torch.tensor([7]))[0],
                       fields[torch.searchsorted(rows, torch.tensor(7))])

    split["holdout"] = torch.tensor([8])
    with pytest.raises(ValueError, match="rows, shape, dtype"):
        c34_runner._load_or_prepare_formal_field_cache(
            images, split, object(), torch.device("cpu"), path
        )


def test_p3_requires_the_registered_p2_adaptation_trigger():
    assert c34_runner._p2_training_triggered({
        "paired_adaptation_decision": {"trigger": True,
                                        "one_sided_exact_binomial_pvalue_B_worse": 0.01}
    })
    assert not c34_runner._p2_training_triggered({
        "paired_adaptation_decision": {"trigger": False,
                                        "one_sided_exact_binomial_pvalue_B_worse": 0.01}
    })
    assert not c34_runner._p2_training_triggered({
        "paired_adaptation_decision": {"trigger": True,
                                        "one_sided_exact_binomial_pvalue_B_worse": float("nan")}
    })


def test_p4_loads_selected_phase_from_formal_training_directory(tmp_path):
    formal_dir = tmp_path / "p3_geometry_adapted_training" / "formal"
    formal_dir.mkdir(parents=True)
    phase = torch.linspace(0.0, 1.0, 10_000, dtype=torch.float32).reshape(100, 100)
    seed_rows = [
        {"seed": seed, "best_epoch": epoch, "validation_accuracy": accuracy,
         "phase_path": str(formal_dir / f"seed_{seed}" / "phase.pt")}
        for seed, epoch, accuracy in ((1, 10, 0.9605), (2, 11, 0.9605), (3, 15, 0.961))
    ]
    selected = seed_rows[-1]
    (formal_dir / "training_summary.json").write_text(
        json.dumps({
            "identity": "C3.4_GEOMETRY_ADAPTED_PHASE_TRAINING",
            "official_mnist_test_loaded": False,
            "holdout_opened_once_after_selection": True,
            "seed_results_selected_by_validation_only": seed_rows,
            "selected_seed": selected,
        }),
        encoding="utf-8",
    )
    np.save(formal_dir / "phase_rad.npy", phase.numpy())
    torch.save({"phase": phase, "selection": selected}, formal_dir / "selected_phase.pt")

    loaded, metadata = c34_runner._load_geometry_adapted_phase(tmp_path)

    assert torch.equal(loaded, phase)
    assert metadata["selected_seed"] == selected


def test_p4_normalizes_c33_singleton_phase_dimensions(tmp_path):
    run = tmp_path / "train_reference"
    run.mkdir()
    phase = torch.linspace(0.0, 1.0, 10_000, dtype=torch.float32).reshape(1, 1, 100, 100)
    torch.save({"phase": phase}, run / "phase.pt")
    (run / "training_summary.json").write_text(
        json.dumps({"defocus_um": 52_000, "best_epoch": 2,
                    "validation": {"accuracy": 0.9},
                    "history": [{"epoch": 2, "val_loss": 0.4}]}),
        encoding="utf-8",
    )

    selected, source = c34_runner._select_c33_fabrication_arm(tmp_path, 0.0)

    assert selected.shape == (100, 100)
    assert torch.equal(selected, phase.reshape(100, 100))
    assert not selected.requires_grad
    assert source["run"] == "train_reference"


def test_p4_accuracy_plot_consumes_saved_assumption_field(tmp_path):
    records = [
        {"arm": arm, "delta_n_assumption": 0.5, "scenario": scenario,
         "metrics": {"accuracy_all_samples": 0.75, "macro_f1_classes_0_to_3": 0.7}}
        for arm in ("c33_naive", "c33_bandwidth_constrained", "c34_geometry_adapted")
        for scenario in ("ideal_phase", "psf_0.8px",
                         "combined_psf0p8_minfeature1_q64_noise20nm")
    ]

    c34_runner._plot_p4_accuracy(tmp_path, records)

    figure = tmp_path / "p4_matched_fabrication_accuracy.png"
    assert figure.is_file()
    assert figure.stat().st_size > 0


def test_independent_p2_audit_recomputes_roi_ties_and_paired_test():
    scores = np.array([[4.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]])
    predictions = c34_p2_audit._predictions(scores, np.array([1.0, 1.0, 0.0]))
    np.testing.assert_array_equal(predictions, [0, -1, -1])
    metrics = c34_p2_audit._metrics(np.array([0, 1, 2]), predictions, scores,
                                    np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]))
    assert metrics["confusion_4x5_with_abstention_column"] == [
        [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]
    ]
    paired = c34_p2_audit._mcnemar(np.array([True] * 10), np.array([False] * 10))
    assert paired["one_sided_exact_binomial_pvalue_B_worse"] == pytest.approx(1 / 1024)
    assert paired["trigger"]


def test_scaled_fresnel_preserves_global_phase_as_a_global_phase():
    generator = torch.Generator().manual_seed(35)
    real = torch.randn((1, 5, 6), generator=generator)
    imag = torch.randn((1, 5, 6), generator=generator)
    field = torch.complex(real, imag)
    phase = 0.731
    reference = c34.scaled_fresnel(field, 2.0, (7, 8), 3.25, 5000.0)
    shifted = c34.scaled_fresnel(field * torch.exp(torch.tensor(1j * phase)), 2.0,
                                  (7, 8), 3.25, 5000.0)
    assert torch.allclose(shifted.abs().square(), reference.abs().square(), rtol=2e-5, atol=1e-7)


def test_abcd_matches_tape_estimate_without_claiming_calibration():
    matrix = c34.abcd_one_lens(80_000.0, 30_000.0, 100_000.0)
    assert matrix["A"] == pytest.approx(-2.3333333333)
    assert matrix["B_um"] == pytest.approx(-86_666.6666667)
    assert matrix["C_per_um"] == pytest.approx(-1 / 30_000)
    assert matrix["D"] == pytest.approx(-1.6666666667)
    assert matrix["determinant"] == pytest.approx(1.0)
    assert matrix["conjugate_image_distance_after_lens_um"] == pytest.approx(48_000.0)
    assert matrix["magnification_at_conjugate"] == pytest.approx(-0.6)
    assert matrix["camera_defocus_after_conjugate_um"] == pytest.approx(52_000.0)


def test_phase_height_round_trip_and_periodicity():
    phase = torch.linspace(-0.5, 8.0, 10000, dtype=torch.float64).reshape(100, 100)
    for delta_n in (0.5, 0.6):
        height = c34.height_from_phase(phase, delta_n)
        wrapped = c34.phase_from_height(height, delta_n)
        assert float(height.min()) >= 0
        assert float(height.max()) <= c34.WAVELENGTH_UM / delta_n + 1e-12
        assert torch.allclose(torch.exp(1j * phase), torch.exp(1j * wrapped), atol=1e-12, rtol=0)
        assert torch.allclose(c34.height_from_phase(phase + 2 * math.pi, delta_n), height,
                              atol=1e-12, rtol=0)


def test_fabrication_operator_preserves_phase_when_all_operators_are_off():
    phase = torch.linspace(0.0, 2 * math.pi, 10000, dtype=torch.float32).reshape(100, 100)
    for delta_n in (0.5, 0.6):
        height = c34.fabricate_height(phase, delta_n, psf_sigma_px=0,
                                      minimum_feature_radius_px=0, height_levels=0)
        recovered = c34.phase_from_height(height, delta_n)
        assert torch.allclose(torch.exp(1j * recovered), torch.exp(1j * phase),
                              atol=1e-6, rtol=0)


def test_mesh_uses_200um_cell_boundaries_and_independent_stl_read(tmp_path):
    height = np.linspace(0.0, 1.0, 10000, dtype=np.float64).reshape(100, 100)
    triangles, metadata = c34.mesh_200um(height, substrate_um=1.2)
    assert metadata["footprint_um"] == [200.0, 200.0]
    assert metadata["sample_center_first_um"] == 1.0
    assert metadata["sample_center_last_um"] == 199.0
    assert metadata["triangles"] == len(triangles)
    path = tmp_path / "phase.stl"
    c34.write_binary_stl(path, triangles)
    sidecar = tmp_path / "stl_metadata.json"
    sidecar.write_text(json.dumps({"unit": "um"}), encoding="utf-8")
    audit = c34.audit_stl_independent(path)
    assert audit["triangles"] == len(triangles)
    assert audit["closed_two_manifold_by_edge_incidence"]
    assert audit["opposite_directed_edge_pairing"]
    assert audit["declared_normal_agreement_fraction"] == 1.0
    assert audit["bbox_numeric_without_intrinsic_unit"][0] == [0.0, 200.0]
    assert audit["bbox_numeric_without_intrinsic_unit"][1] == [0.0, 200.0]
    assert audit["declared_unit_status"] == "SIDECAR_DECLARES_UM"
    assert audit["unit_evidence_path"] == str(sidecar.resolve())
    recovered = c34.recover_height_from_mesh_vertices(path)
    np.testing.assert_allclose(recovered, height, atol=6e-6, rtol=0)


def test_binary_input_field_matches_c33_production_preprocessing():
    configuration = c34.c33.build_configuration(52_000.0, "cpu")
    images = torch.zeros((2, 28, 28), dtype=torch.uint8)
    images[0, 8:20, 10:14] = 255
    images[1, 10:18, 7:21] = 255
    with torch.no_grad():
        new_input = c34.input_field(images, configuration)
        old_result = c34.c33.forward_field(
            images, configuration, torch.zeros((1, 1, 100, 100))
        )
    assert torch.allclose(new_input, old_result["lightgen_input_field"],
                          rtol=5e-4, atol=2e-7)


def test_model_b_fixed_roi_phase_frozen_and_energy_ledger():
    field = torch.zeros((1, 100, 100), dtype=torch.complex64)
    field[0, 49, 50] = 1.0 + 0.25j
    phase = torch.zeros((100, 100), dtype=torch.float32)
    original_phase = phase.clone()
    with torch.no_grad():
        base = c34.model_b_forward(field, phase, 0.5, lens_pitch_um=4.0)
    assert base["camera_intensity"].shape == (1, 2048, 2048)
    assert base["roi_scores"].shape == (1, 4)
    assert torch.equal(phase, original_phase)
    assert torch.allclose(base["roi_scores"].sum(1) + base["outside_roi_energy"],
                          base["comparison_crop_energy"], rtol=1e-5, atol=1e-7)
    assert base["camera_intensity"].isfinite().all()


def test_c34_split_has_exact_counts_and_no_overlap():
    labels = torch.empty(60000, dtype=torch.long)
    class_capacity = 7000
    selected_parts = []
    for cls, count in enumerate([5331, 6068, 5362, 5518]):
        start = cls * class_capacity
        labels[start:start + class_capacity] = cls
        selected_parts.append(torch.arange(start, start + count))
    selected = torch.cat(selected_parts)
    split = c34.stratified_c34_split(selected, labels)
    assert [len(split[key]) for key in ("train", "validation", "holdout")] == [18279, 2000, 2000]
    assert len(torch.unique(torch.cat(list(split.values())))) == 22279
    assert all((labels[split[key]] == cls).sum().item() == expected
               for key, expected_by_class in (("validation", [500] * 4), ("holdout", [500] * 4))
               for cls, expected in enumerate(expected_by_class))
