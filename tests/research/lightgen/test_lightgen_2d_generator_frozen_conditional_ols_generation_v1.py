from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from scripts.research.lightgen import execute_lightgen_2d_generator_frozen_conditional_ols_generation_v1 as task
from scripts.research.lightgen import execute_lightgen_2d_generator_training_population_expansion_comparison_v1 as expansion


def _toy_fit() -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(20)
    row = torch.arange(200, dtype=torch.float64).view(200, 1)
    port = torch.arange(100, dtype=torch.float64).view(1, 100)
    real = row * 0.001 + port * 0.02
    imag = row * -0.003 + port * 0.01
    z = torch.complex(real, imag).to(torch.complex64)
    return z, labels


def test_corrected_role_partition_covers_3300_rows_once_and_rejects_gap() -> None:
    positions, counts, roles = expansion._role_partition()
    population = {"role_positions": positions, "role_counts": counts, "roles": roles}
    expansion._validate_role_partition(population)
    assert sum(counts.values()) == 3_300
    assert counts == {
        "fit": 200,
        "historic": 100,
        "ce_confirmation": 300,
        "presentation_confirmation": 300,
        "class9_confirmation": 300,
        "expanded_training": 1_800,
        "coverage_confirmation": 300,
    }
    pages = expansion._role_page_positions(population)
    flattened = [row for group in positions for page in pages[group] for row in page]
    assert flattened == list(range(3_300))
    bad = dict(population)
    bad["role_positions"] = dict(positions)
    bad["role_positions"]["class9_confirmation"] = [900, 1199]
    with pytest.raises(expansion.PopulationExpansionError, match="intervals"):
        expansion._validate_role_partition(bad)


def test_diagonal_gaussian_matches_independent_float64_population_reference() -> None:
    z, labels = _toy_fit()
    bank = task._fit_diagonal_gaussian(z, labels)
    values = z.to(torch.complex128)
    means = torch.stack([values[labels == c].mean(0) for c in range(10)])
    real_std = torch.stack([values[labels == c].real.var(0, unbiased=False).sqrt() for c in range(10)]).clamp_min(task.STD_FLOOR)
    imag_std = torch.stack([values[labels == c].imag.var(0, unbiased=False).sqrt() for c in range(10)]).clamp_min(task.STD_FLOOR)
    rng = torch.Generator(device="cpu").manual_seed(task.SEED)
    eps_real = torch.randn((10, 200, 100), generator=rng, dtype=torch.float64)
    eps_imag = torch.randn((10, 200, 100), generator=rng, dtype=torch.float64)
    expected = torch.complex(means.real[:, None] + real_std[:, None] * eps_real, means.imag[:, None] + imag_std[:, None] * eps_imag).to(torch.complex64).reshape(2_000, 100)
    assert torch.equal(bank["mean"], means)
    assert torch.allclose(bank["real_std"], real_std, rtol=1.0e-14, atol=1.0e-16)
    assert torch.allclose(bank["imag_std"], imag_std, rtol=1.0e-14, atol=1.0e-16)
    assert torch.allclose(bank["ols"], expected, rtol=1.0e-6, atol=1.0e-7)
    assert bank["noise_draw_order"] == ["all_real_epsilon_[10,200,100]", "all_imag_epsilon_[10,200,100]"]
    task._verify_bank_against_independent_reference(z, labels, bank)


def test_diagonal_gaussian_floor_and_wrong_noise_order_are_detected() -> None:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(20)
    z = torch.zeros((200, 100), dtype=torch.complex64)
    bank = task._fit_diagonal_gaussian(z, labels)
    assert bank["real_floor_dimension_count"] == 1_000
    assert bank["imag_floor_dimension_count"] == 1_000
    assert float(bank["real_std"].min()) == task.STD_FLOOR
    changed = dict(bank)
    changed["noise_draw_order"] = ["all_imag_epsilon_[10,200,100]", "all_real_epsilon_[10,200,100]"]
    with pytest.raises(task.FrozenGenerationError, match="draw order"):
        task._verify_bank_against_independent_reference(z, labels, changed)


def test_report_sampling_parameters_serialize_complex_mean_as_real_and_imaginary() -> None:
    bank = {
        "mean": torch.tensor([[1 + 2j, 3 - 4j]], dtype=torch.complex64),
        "real_std": torch.tensor([[0.5, 0.25]]),
        "imag_std": torch.tensor([[0.75, 0.125]]),
        "real_floor_dimension_count": 0,
        "imag_floor_dimension_count": 0,
        "noise_draw_order": ["real", "imaginary"],
    }
    serialized = json.dumps(task._sampling_parameters(bank))
    restored = json.loads(serialized)
    assert restored["mean_real"] == [[1.0, 3.0]]
    assert restored["mean_imag"] == [[2.0, -4.0]]


def test_resume_loads_the_saved_bank_without_drawing_again(tmp_path, monkeypatch) -> None:
    z, labels = _toy_fit()
    bank = task._fit_diagonal_gaussian(z, labels)
    path = tmp_path / "conditional_ols_bank.pt"
    torch.save(bank, path)
    monkeypatch.setattr(task, "_fit_diagonal_gaussian", lambda *_args, **_kwargs: pytest.fail("resume must not redraw the saved bank"))
    restored = task._bank_payload(path, z, labels)
    assert torch.equal(restored["ols"], bank["ols"])


def test_new_artifact_writer_roundtrips_and_refuses_overwrite(tmp_path) -> None:
    json_path = tmp_path / "population.json"
    torch_path = tmp_path / "bank.pt"
    expected_json = {"rows": [1, 2, 3], "role": "TRAIN"}
    expected_torch = {"z": torch.tensor([1 + 2j], dtype=torch.complex64)}
    task._write_json(json_path, expected_json, "unit test")
    task._write_torch(torch_path, expected_torch, "unit test")
    assert json.loads(json_path.read_text(encoding="utf-8")) == expected_json
    assert torch.equal(torch.load(torch_path, map_location="cpu", weights_only=True)["z"], expected_torch["z"])
    with pytest.raises(task.FrozenGenerationError, match="overwrite"):
        task._write_json(json_path, {"rows": []}, "unit test")


def test_resume_validates_an_existing_summary_tensor_without_overwriting(tmp_path) -> None:
    path = tmp_path / "sample_row_metrics.pt"
    saved = {"score": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    task._write_torch(path, saved, "unit test")
    task._write_or_validate_torch(path, saved, "unit test")
    with pytest.raises(task.FrozenGenerationError, match="differs"):
        task._write_or_validate_torch(path, {"score": torch.tensor([1.0, 3.0], dtype=torch.float64)}, "unit test")


def test_output_preparation_accepts_only_the_known_population_correction_pages(tmp_path) -> None:
    task._write_json(tmp_path / "corrected_input_population.json", {"rows": 3_300}, "unit test")
    task._write_json(tmp_path / "population_evidence_review.json", {"covered": True}, "unit test")
    page = tmp_path / "corrected_class9_confirmation_000.png"
    page.write_bytes(b"page")
    task._prepare_target(tmp_path, resume=False)

    (tmp_path / "readout_rows_000000_000100.pt").write_bytes(b"unexpected")
    with pytest.raises(task.FrozenGenerationError, match="unstaged"):
        task._prepare_target(tmp_path, resume=False)


def test_formal_measurement_freeze_check_consumes_runtime_and_evaluator_state() -> None:
    runtime = torch.nn.Linear(2, 2)
    evaluator = torch.nn.Linear(2, 2)
    runtime.requires_grad_(False)
    evaluator.requires_grad_(False)
    runtime_before = task._snapshot_state(runtime)
    evaluator_before = {"E_D": task.capacity._state_snapshot(evaluator)}
    assert task._verify_measurement_freeze(runtime, runtime_before, {"E_D": evaluator}, evaluator_before)["passed"]

    with torch.no_grad():
        evaluator.weight[0, 0] += 1.0
    changed = task._verify_measurement_freeze(runtime, runtime_before, {"E_D": evaluator}, evaluator_before)
    assert changed["executed"] and not changed["passed"]
    assert not changed["evaluator_parameters_and_buffers_unchanged"]


def test_readout_payload_preserves_chunk_order_with_single_row_optical_microbatches() -> None:
    class Plane:
        pixel_pitch_m = (1.0e-6, 1.0e-6)

        def to_dict(self) -> dict[str, object]:
            return {"pixel_pitch_m": list(self.pixel_pitch_m)}

    class Provenance:
        def to_dict(self) -> dict[str, object]:
            return {"operation": "test"}

    class Field:
        def __init__(self, batch: int):
            self.tensor = torch.ones((batch, 1, 2, 2), dtype=torch.complex64)
            self.plane = Plane()
            self.provenance = Provenance()

    class Runtime:
        def __init__(self) -> None:
            self.generator = SimpleNamespace(last_extra_plane_powers=None)
            self.seen_batches: list[int] = []

        def forward_from_ols(self, z: torch.Tensor) -> SimpleNamespace:
            self.seen_batches.append(int(z.shape[0]))
            field = Field(int(z.shape[0]))
            self.generator.last_extra_plane_powers = torch.ones((z.shape[0], 2))
            raw = torch.ones((z.shape[0], 1, 200, 200)) * (0.1 + z.real.reshape(-1, 1, 1, 1) * 0.01)
            return SimpleNamespace(raw_intensity=raw, generator_input_field=field, generator_records=[field, field], raw_camera=field)

    runtime = Runtime()
    rows = {
        "z": torch.tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype=torch.complex64).reshape(3, 1),
        "requested_class": torch.tensor([2, 4, 6]),
        "kind": ["sample"] * 3,
        "source_row_indices": torch.full((3,), -1),
        "source_positions": torch.full((3,), -1),
        "sample_index_within_class": torch.arange(3),
    }
    payload = task._readout_payload(runtime, {}, rows, 0, 3, device=torch.device("cpu"))
    assert runtime.seen_batches == [1, 1, 1]
    assert payload["requested_class"].tolist() == [2, 4, 6]
    assert payload["raw_cmos"].shape == (3, 1, 200, 200)
    assert payload["new_phase_plane_power"].shape == (3, 2)
    assert payload["field_metadata"]["camera_plane"]["pixel_pitch_m"] == [1.0e-6, 1.0e-6]


def test_diversity_reader_uses_aligned_chunks_when_samples_start_inside_a_chunk(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(task, "FIT_ROWS", 2)
    monkeypatch.setattr(task, "MEAN_COUNT", 1)
    monkeypatch.setattr(task, "SAMPLE_COUNT", 3)
    monkeypatch.setattr(task, "TOTAL_OUTPUTS", 6)
    monkeypatch.setattr(task, "CHUNK_ROWS", 2)
    for start, stop in ((0, 2), (2, 4), (4, 6)):
        value = torch.zeros((stop - start, 1, 28, 28))
        for index in range(start, stop):
            value[index - start, 0, index, index] = 1.0
        torch.save({"evaluator_view": value}, tmp_path / task._chunk_name(start, stop))
    inputs = {"images": torch.zeros((5, 784)), "source_indices": torch.arange(100, 105)}
    diversity = task._descriptive_diversity(tmp_path, inputs, {})
    assert diversity["scope"].endswith("3_generated_views_and_3300_canonical_TRAIN_targets")
    assert len(diversity["nearest_target_source_row"]) == 3
    assert len(diversity["nearest_generated_sample_index"]) == 3


def test_gallery_prediction_reader_uses_aligned_chunks_when_samples_start_inside_a_chunk(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(task, "FIT_ROWS", 2)
    monkeypatch.setattr(task, "MEAN_COUNT", 1)
    monkeypatch.setattr(task, "SAMPLE_COUNT", 3)
    monkeypatch.setattr(task, "TOTAL_OUTPUTS", 6)
    monkeypatch.setattr(task, "CHUNK_ROWS", 2)
    for start, stop in ((0, 2), (2, 4), (4, 6)):
        logits = torch.zeros((stop - start, 10))
        for index in range(start, stop):
            logits[index - start, index] = 1.0
        torch.save({"e_d_logits": logits}, tmp_path / task._chunk_name(start, stop))
    assert task._sample_predictions(tmp_path).tolist() == [3, 4, 5]


def test_output_row_mapping_keeps_empirical_means_and_random_classes_separate() -> None:
    z, labels = _toy_fit()
    bank = task._fit_diagonal_gaussian(z, labels)
    rows = task._make_output_rows(z, labels, torch.arange(10_000, 10_200), bank)
    assert rows["z"].shape == (2_210, 100)
    assert rows["kind"][:200] == ["empirical_fit_ols"] * 200
    assert rows["kind"][200:210] == ["class_mean_ols"] * 10
    assert rows["kind"][210:] == ["diagonal_gaussian_sample"] * 2_000
    assert rows["requested_class"][210:].tolist() == torch.arange(10).repeat_interleave(200).tolist()
    assert rows["source_row_indices"][:200].tolist() == list(range(10_000, 10_200))
    assert bool((rows["source_row_indices"][200:] == -1).all())


def test_float64_score_consumer_uses_requested_class_for_ce_and_margin() -> None:
    logits = torch.full((2, 10), -1.0, dtype=torch.float32)
    logits[0, 0], logits[0, 1] = 2.0, 1.0
    logits[1, 1], logits[1, 2] = 3.0, 2.0
    labels = torch.tensor([0, 1])
    ce, margin = task._score_for_labels(logits, labels)
    assert ce.dtype == torch.float64 and margin.dtype == torch.float64
    assert margin.tolist() == [1.0, 1.0]
    wrong_ce, wrong_margin = task._score_for_labels(logits, (labels + 1) % 10)
    assert not torch.equal(ce, wrong_ce)
    assert not torch.equal(margin, wrong_margin)
    summary = task._classification_summary(labels, logits)
    assert summary["per_class_correct"][:2] == [1, 1]
    assert summary["confusion_matrix_actual_rows_predicted_columns"][0][0] == 1


def _small_decision_summary(per_class: list[int], *, clipping: float = 0.0) -> dict:
    total = sum(per_class)
    recalls = [value / 2 for value in per_class]
    summary = {
        "primary_e_d_sample_screen": {
            "correct_total": sum(per_class),
            "macro_recall": sum(recalls) / 10,
            "per_class_correct": per_class,
            "per_class_recall": recalls,
            "raw_clipping_fraction": clipping,
            "zero_image_count": 0,
            "constant_image_count": 0,
            "all_finite": True,
            "passed": sum(recalls) / 10 >= 0.50 and min(recalls) >= 0.10 and clipping <= 0.01,
            "rows": total,
        }
    }
    return summary


def test_decision_consumes_production_summary_and_preserves_screen_boundary() -> None:
    supported = task._decision(_small_decision_summary([2] * 10), valid=True)
    assert supported["classification"] == "CONDITIONAL_OLS_GENERATION_ENTRY_SUPPORTED"
    below_class_floor = task._decision(_small_decision_summary([2] * 9 + [0]), valid=True)
    assert below_class_floor["classification"] == "CONDITIONAL_OLS_GENERATION_NOT_SUPPORTED_WITHIN_SCOPE"
    invalid = task._decision(_small_decision_summary([2] * 10), valid=False)
    assert invalid["status"] == "INVALID_HARNESS" and invalid["classification"] is None
    assert invalid["training_recommendation"] == "NONE" and invalid["next_action"] == "NONE"


def test_chunk_validator_rejects_ols_or_source_row_remapping() -> None:
    z, labels = _toy_fit()
    bank = task._fit_diagonal_gaussian(z, labels)
    rows = task._make_output_rows(z, labels, torch.arange(10_000, 10_200), bank)
    payload = {
        "measurement_positions": torch.arange(0, 2),
        "z": rows["z"][:2].clone(),
        "requested_class": rows["requested_class"][:2].clone(),
        "source_row_indices": rows["source_row_indices"][:2].clone(),
        "source_positions": rows["source_positions"][:2].clone(),
        "sample_index_within_class": rows["sample_index_within_class"][:2].clone(),
        "raw_cmos": torch.zeros((2, 1, 200, 200)),
        "evaluator_view": torch.zeros((2, 1, 28, 28)),
        **{f"{name.lower()}_logits": torch.zeros((2, 10)) for name in task.EVALUATOR_NAMES},
    }
    task._validate_chunk(payload, rows, 0, 2)
    payload["source_row_indices"][0] += 1
    with pytest.raises(task.FrozenGenerationError, match="mapping differs"):
        task._validate_chunk(payload, rows, 0, 2)


def test_corrected_population_roles_are_distinct_from_historical_strings() -> None:
    old = {
        "source_row_indices": list(range(3_300)),
        "labels": [i % 10 for i in range(3_300)],
        "roles": ["new_confirmation"] * 3_300,
        "role_counts": {"new_confirmation": 3_300},
        "role_positions": {"new_confirmation": [3_000, 3_300]},
    }
    fixed = task._corrected_population(old)
    assert fixed["role_positions"]["class9_confirmation"] == [900, 1_200]
    assert fixed["role_positions"]["coverage_confirmation"] == [3_000, 3_300]
    assert fixed["roles"][900] == "class9_confirmation"
    assert fixed["roles"][3_000] == "coverage_confirmation"
    assert old["role_positions"] == {"new_confirmation": [3_000, 3_300]}


def test_saved_production_readout_aggregation_flows_into_the_screen_decision(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(task, "FIT_ROWS", 1)
    monkeypatch.setattr(task, "MEAN_COUNT", 1)
    monkeypatch.setattr(task, "SAMPLE_COUNT", 20)
    monkeypatch.setattr(task, "TOTAL_OUTPUTS", 22)
    monkeypatch.setattr(task, "CHUNK_ROWS", 10)
    requested = torch.cat((torch.tensor([0, 1]), torch.arange(10).repeat_interleave(2)))
    bank = {"real_floor_dimension_count": 0, "imag_floor_dimension_count": 0, "sample_index_within_class": torch.arange(2).repeat(10)}
    flat = torch.arange(40_000, dtype=torch.float32).reshape(1, 1, 200, 200) / 1_000_000.0 + 0.1
    view = torch.nn.functional.adaptive_avg_pool2d(flat, (28, 28))
    for start in range(0, 22, 10):
        stop = min(22, start + 10)
        labels = requested[start:stop]
        logits = torch.full((stop - start, 10), -1.0)
        logits[torch.arange(stop - start), labels] = 2.0
        raw = flat.expand(stop - start, -1, -1, -1).clone()
        payload = {
            "evaluator_view": view.expand(stop - start, -1, -1, -1).clone(),
            "raw_cmos": raw,
            "raw_zero_image": (raw.flatten(1).abs().sum(1) == 0),
            "raw_constant_image": (raw.flatten(1).std(1, unbiased=False) == 0),
            **{f"{name.lower()}_logits": logits.clone() for name in task.EVALUATOR_NAMES},
        }
        torch.save(payload, tmp_path / task._chunk_name(start, stop))
    output_rows = {"requested_class": requested}
    summary, row_metrics = task._summary_and_rows(tmp_path, output_rows, bank)
    decision = task._decision(summary, valid=True)
    assert decision["classification"] == "CONDITIONAL_OLS_GENERATION_ENTRY_SUPPORTED"
    assert summary["primary_e_d_sample_screen"]["per_class_correct"] == [2] * 10
    assert row_metrics["e_d_predictions_diagonal_gaussian_sample"].tolist() == torch.arange(10).repeat_interleave(2).tolist()

    wrong = torch.load(tmp_path / task._chunk_name(20, 22), map_location="cpu", weights_only=True)
    wrong["e_d_logits"][0] = torch.roll(wrong["e_d_logits"][0], 1)
    wrong["e_d_logits"][1] = torch.roll(wrong["e_d_logits"][1], 1)
    torch.save(wrong, tmp_path / task._chunk_name(20, 22))
    failed_summary, _ = task._summary_and_rows(tmp_path, output_rows, bank)
    failed_decision = task._decision(failed_summary, valid=True)
    assert failed_decision["classification"] == "CONDITIONAL_OLS_GENERATION_NOT_SUPPORTED_WITHIN_SCOPE"
