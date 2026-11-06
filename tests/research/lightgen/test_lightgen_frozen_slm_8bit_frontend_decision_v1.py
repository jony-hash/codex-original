from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts.research.lightgen import execute_lightgen_frozen_slm_8bit_frontend_decision_v1 as task


def test_a8_i8_decode_order_and_invalid_gray_rejection() -> None:
    gray = np.array([0, 64, 255], dtype=np.uint8)
    a8 = task._decode_gray(gray, "A8_amplitude_linear")
    i8 = task._decode_gray(gray, "I8_intensity_linear")
    assert a8[0] == 0.0 and a8[-1] == 1.0
    assert i8[0] == 0.0 and i8[-1] == 1.0
    assert np.array_equal(task._intensity_from_amplitude(a8), np.square(gray.astype(np.float64) / 255.0))
    assert np.allclose(task._intensity_from_amplitude(i8), gray.astype(np.float64) / 255.0, rtol=0, atol=1e-15)
    assert not np.allclose(a8[1] ** 2, i8[1] ** 2)
    with pytest.raises(ValueError):
        task._decode_gray(np.array([256], dtype=np.int64), "A8_amplitude_linear")
    with pytest.raises(ValueError):
        task._decode_gray(np.array([np.nan], dtype=np.float64), "I8_intensity_linear")
    with pytest.raises(ValueError):
        task._intensity_from_amplitude(np.array([np.inf], dtype=np.float64))


def test_coordinate_candidate_matches_exhaustive_256_code_search() -> None:
    rng = np.random.default_rng(20261026)
    exhaustive_codes = np.arange(256, dtype=np.uint8)
    for condition in task.CONDITIONS:
        for _ in range(50):
            current = rng.integers(0, 256, size=8, dtype=np.uint8)
            cross = rng.normal(0.0, 0.2, size=8)
            norm_sq = float(rng.uniform(0.1, 2.0))
            chosen, old_amp, chosen_amp, _ = task._best_coordinate_codes(
                current, cross, norm_sq, condition, np.ones(8, dtype=np.float64),
            )
            all_amp = task._decode_gray(exhaustive_codes, condition)
            for index in range(len(current)):
                all_delta = all_amp - old_amp[index]
                all_cost = 2.0 * all_delta * cross[index] + np.square(all_delta) * norm_sq
                chosen_delta = chosen_amp[index] - old_amp[index]
                chosen_cost = 2.0 * chosen_delta * cross[index] + chosen_delta**2 * norm_sq
                assert chosen_cost <= float(all_cost.min()) + 1e-12
                assert 0 <= int(chosen[index]) <= 255


def test_public_phase_equivalence_and_frozen_discrete_solver_controls(tmp_path: Path) -> None:
    matrix = np.zeros((task.PORTS, task.PIXEL_COUNT), dtype=np.complex128)
    matrix[0, 0] = 1.0 + 0.0j
    matrix[1, 1] = 0.25 + 0.5j
    target_codes = np.zeros((1, task.PIXEL_COUNT), dtype=np.uint8)
    target_codes[0, :2] = [83, 177]
    target_amp = task._decode_gray(target_codes, "A8_amplitude_linear")
    target = target_amp @ matrix.T
    phase = 1.37
    aligned_phase, phase_error = task._phase_error(target[0] * np.exp(1j * phase), target[0])
    assert phase_error < 1e-14
    assert abs(abs(aligned_phase) - abs(phase)) < 1e-12 or np.isclose(
        np.exp(1j * aligned_phase), np.exp(1j * phase), atol=1e-12,
    )

    initial = np.zeros_like(target_codes)
    initial[0, :2] = [4, 9]
    identity = [{"sample_key": "known", "method": "G4", "label": 3}]
    solved, records = task._coordinate_optimize(
        matrix, target, initial, "A8_amplitude_linear", identity, [None],
        tmp_path / "positive.pt", tmp_path / "positive_progress.json", "positive-run", resume=True,
    )
    assert np.array_equal(solved[0, :2], target_codes[0, :2])
    assert records[0]["final_matrix_coefficient_error"] <= task.COEFFICIENT_STOP
    assert records[0]["skipped_input"] is False
    resumed, _ = task._coordinate_optimize(
        matrix, target, initial, "A8_amplitude_linear", identity, [None],
        tmp_path / "positive.pt", tmp_path / "positive_progress.json", "positive-run", resume=True,
    )
    assert np.array_equal(resumed, solved)
    saved_state_path = tmp_path / "positive.pt"
    saved_state = torch.load(saved_state_path, map_location="cpu", weights_only=False)
    saved_state["residual"][0, 0] += 1.0
    torch.save(saved_state, saved_state_path)
    with pytest.raises(RuntimeError, match="saved coordinate residual"):
        task._coordinate_optimize(
            matrix, target, initial, "A8_amplitude_linear", identity, [None],
            saved_state_path, tmp_path / "positive_progress.json", "positive-run", resume=True,
        )
    with pytest.raises(ValueError, match="initial commands"):
        task._coordinate_optimize(
            matrix, target, initial.astype(np.float32), "A8_amplitude_linear", identity, [None],
            tmp_path / "wrong-initial.pt", tmp_path / "wrong-initial-progress.json", "bad-run", resume=True,
        )

    impossible_matrix = np.zeros_like(matrix)
    impossible_matrix[0, 0] = 1.0
    impossible_target = np.zeros((1, task.PORTS), dtype=np.complex128)
    impossible_target[0, :2] = [0.5, 0.5]
    rejected, rejection = task._coordinate_optimize(
        impossible_matrix, impossible_target, initial, "A8_amplitude_linear", identity, [None],
        tmp_path / "negative.pt", tmp_path / "negative_progress.json", "negative-run", resume=True,
    )
    del rejected
    assert rejection[0]["final_matrix_coefficient_error"] > 0.5
    assert rejection[0]["termination"] == "full_sweep_no_gray_change"


def test_identity_population_and_decoded_command_tamper_rejections() -> None:
    identities = [{"sample_key": f"S{i:03d}", "label": i % 10} for i in range(200)]
    task._validate_identity_population(identities, identities)
    with pytest.raises(ValueError):
        task._validate_identity_population(identities, identities[:-1])
    duplicated = list(identities)
    duplicated[-1] = dict(duplicated[-2])
    with pytest.raises(ValueError):
        task._validate_identity_population(identities, duplicated)

    codes = np.array([64, 255], dtype=np.uint8)
    decoded = task._decode_gray(codes, "A8_amplitude_linear")
    task._validate_decoded_command(codes, decoded, "A8_amplitude_linear")
    tampered = codes.copy()
    tampered[0] += 1
    with pytest.raises(ValueError):
        task._validate_decoded_command(tampered, decoded, "A8_amplitude_linear")


def test_final_output_consumer_rejects_nonfinite_tampering(tmp_path: Path) -> None:
    identity = {"sample_key": "G4_00001", "method": "G4", "cohort_row": 1,
                "label": 0, "proposal_id": 1, "component_id": 0}
    output = {
        "schema": "lightgen_frozen_slm_8bit_frontend_output.v1", "task_id": task.TASK_ID,
        "run_id": "test-run", "condition": task.CONDITIONS[0], "grid": 2048,
        "sample_identity": identity,
        "actual_coefficients": torch.ones(task.PORTS, dtype=torch.complex128),
        "raw_cmos_intensity_w_per_m2": torch.ones(task.IMAGE_SHAPE, dtype=torch.float64),
        "power_trace_w": {key: 1e-6 for key in task.POWER_FIELDS},
        "camera_pixel_area_m2": 1e-8,
        "camera_total_power_w": 4e-4,
    }
    path = tmp_path / "output.pt"
    torch.save(output, path)
    task._validate_final_output(path, "test-run", identity, task.CONDITIONS[0], 2048)
    output["actual_coefficients"][0] = complex(float("nan"), 0.0)
    torch.save(output, path)
    with pytest.raises(RuntimeError):
        task._validate_final_output(path, "test-run", identity, task.CONDITIONS[0], 2048)


def _synthetic_ratings(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    generated: dict[tuple[str, str], dict[str, Any]] = {}
    sample_labels: dict[str, int] = {}
    for label in range(10):
        for row in range(10):
            sample_key = f"G4_{label}_{row}"
            sample_labels[sample_key] = label
            for condition in task.REVIEW_CONDITIONS:
                card_id = f"R{len(cards) + 1:04d}"
                card = {"card_id": card_id, "kind": "generated", "condition": condition,
                        "sample_key": sample_key, "method": "G4", "requested_label": label,
                        "proposal_id": row, "component_id": 0, "identity_group": sample_key,
                        "duplicate_group": None}
                cards.append(card)
                generated[(sample_key, condition)] = card
    repeat_keys = [f"G4_{label}_{row}" for label in range(10) for row in range(2)]
    repeat_conditions = ["continuous"] * 7 + ["A8_amplitude_linear"] * 7 + ["I8_intensity_linear"] * 6
    for number, (sample_key, condition) in enumerate(zip(repeat_keys, repeat_conditions, strict=True), start=1):
        original = generated[(sample_key, condition)]
        group = f"D{number:02d}"
        original["duplicate_group"] = group
        cards.append({**original, "card_id": f"R{len(cards) + 1:04d}", "kind": "hidden_duplicate",
                      "duplicate_group": group})
    for anchor in range(20):
        cards.append({"card_id": f"R{len(cards) + 1:04d}", "kind": "real_anchor",
                      "target_label": anchor // 2, "duplicate_group": None})
    key = {"schema": "lightgen_frozen_slm_8bit_frontend_review_key.v1", "task_id": task.TASK_ID,
           "run_id": "test-run", "review_id": task.REVIEW_ID, "cards": cards}
    (tmp_path / "blind_review_key.json").write_text(json.dumps(key), encoding="utf-8")
    rating_rows: list[dict[str, Any]] = []
    for card in cards:
        if card["kind"] in {"generated", "hidden_duplicate"}:
            rating = sample_labels[card["sample_key"]]
        else:
            rating = int(card["target_label"])
        rating_rows.append({"card_id": card["card_id"], "rating": rating})
    ratings = {"schema": "lightgen_frozen_slm_8bit_frontend_ratings.v1", "task_id": task.TASK_ID,
               "review_id": task.REVIEW_ID, "cards": rating_rows}
    path = tmp_path / "ratings.json"
    path.write_text(json.dumps(ratings), encoding="utf-8")
    return path, ratings


def test_review_ratings_assignment_qc_and_exact_binomial_boundaries(tmp_path: Path) -> None:
    ratings_path, _ = _synthetic_ratings(tmp_path)
    result = task._consume_ratings(tmp_path, ratings_path)
    assert result["quality_passed"] is True
    assert result["quality_control"]["duplicate_consistent_pairs"] == 20
    assert result["quality_control"]["anchor_correct"] == 20
    for condition in task.CONDITIONS:
        loss = result["conditions"][condition]["semantic_loss_decision"]
        assert loss["denominator"] == 100 and loss["errors"] == 0
        assert loss["decision"] == "SUPPORTS_SEMANTIC_RETENTION"
        assert loss["one_sided_95_percent_clopper_pearson"]["upper"] == pytest.approx(
            1.0 - 0.05 ** (1.0 / 100.0), abs=1e-12,
        )
    assert task._semantic_loss_classification(0, 0)["decision"] == "UNDECIDABLE_NO_BASELINE_CORRECT_SAMPLES"
    assert task._semantic_loss_classification(1, 89)["decision"] == "INCONCLUSIVE"
    assert task._semantic_loss_classification(12, 100)["decision"] == "SUPPORTS_SEMANTIC_LOSS_ABOVE_LIMIT"

    with pytest.raises(FileNotFoundError):
        task._consume_ratings(tmp_path, tmp_path / "missing-ratings.json")
    malformed = json.loads(ratings_path.read_text(encoding="utf-8"))
    malformed["cards"][1]["card_id"] = malformed["cards"][0]["card_id"]
    malformed_path = tmp_path / "duplicated-rating-id.json"
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated card identity"):
        task._consume_ratings(tmp_path, malformed_path)


def test_fixed_population_review_schedule_is_blind_and_separated() -> None:
    config = task._config(task.DEFAULT_CONFIG)
    inputs = task._load_inputs(config)
    schedule = task._review_schedule(inputs)
    cards = schedule["cards"]
    assert len(cards) == 340
    assert Counter(card["kind"] for card in cards) == {
        "generated": 300, "real_anchor": 20, "hidden_duplicate": 20,
    }
    generated = [card for card in cards if card["kind"] == "generated"]
    assert Counter(card["condition"] for card in generated) == {
        "continuous": 100, "A8_amplitude_linear": 100, "I8_intensity_linear": 100,
    }
    positions: dict[str, list[int]] = defaultdict(list)
    for position, card in enumerate(cards):
        if card.get("identity_group"):
            positions[card["identity_group"]].append(position)
    assert len(positions) == 100
    assert all(min(b - a for a, b in zip(items, items[1:])) >= 40 for items in positions.values())
    import io
    from PIL import Image
    generated_card = next(card for card in cards if card["kind"] == "generated")
    encoded = task._review_image_bytes(inputs, generated_card)
    source_index, condition = generated_card["image_source"]
    source_output = task._old_output(inputs, int(source_index), condition, 2048)
    expected_pixels = np.rint(
        (source_output["raw_cmos_intensity_w_per_m2"].double()
         * float(source_output["camera_pixel_area_m2"]) * float(inputs["camera_gain"])).clamp(0.0, 1.0).numpy()
        * 255.0
    ).astype(np.uint8)
    with Image.open(io.BytesIO(encoded)) as decoded:
        assert decoded.mode == "L" and decoded.size == (200, 200)
        assert np.array_equal(np.asarray(decoded), expected_pixels)
    anchor = next(card for card in cards if card["kind"] == "real_anchor")
    assert task._review_image_bytes(inputs, anchor) == (
        inputs["native_review_dir"] / anchor["image_source"][1]
    ).read_bytes()
    page = task._review_page(task.REVIEW_ID,
                            [{"card_id": card["card_id"], "image": card["image"]} for card in cards])
    assert "/ 340 张" in page and "/ 17 段" in page
    assert "A8_amplitude_linear" not in page and "sample_key" not in page


def test_reachability_identity_alignment_uses_saved_selection_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    config = task._config(task.DEFAULT_CONFIG)
    replay_dir = task.ROOT / config["source"]["replay_attempt_dir"]
    native_dir = task.ROOT / config["source"]["native_review_attempt_dir"]
    identities = json.loads((replay_dir / "input_authority.json").read_text(encoding="utf-8"))["sample_identities"]
    entries = json.loads((native_dir / "selection_manifest.json").read_text(encoding="utf-8"))["items"]
    fake_inputs = {
        "checkpoint": {"completed_global_step": 45_000, "branch": "C"},
        "entries": entries,
    }
    monkeypatch.setattr(task.reachability, "_load_yaml", lambda _path: {})
    monkeypatch.setattr(task.reachability, "_validate_config", lambda *_args: None)
    monkeypatch.setattr(task.reachability, "_load_inputs", lambda _config: fake_inputs)

    assert task._load_reachability_inputs(config, identities) is fake_inputs

    changed_entries = [dict(item) for item in entries]
    changed_entries[0]["component_id"] += 1
    fake_inputs["entries"] = changed_entries
    with pytest.raises(RuntimeError, match="sample identity order differs"):
        task._load_reachability_inputs(config, identities)


def test_saved_original_outputs_are_loaded_once_per_identity_condition_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {"sample_key": "G4_00001"}
    calls: list[tuple[str, int]] = []

    def load_output(_inputs: Any, _identity: Any, condition: str, grid: int) -> dict[str, Any]:
        calls.append((condition, grid))
        return {"condition": condition, "grid": grid}

    monkeypatch.setattr(task.attribution, "_load_output", load_output)
    inputs = {"identities": [identity], "_verified_old_output_cache": {}}
    first = task._old_output(inputs, 0, "continuous", 2048)
    repeated = task._old_output(inputs, 0, "continuous", 2048)
    other_grid = task._old_output(inputs, 0, "continuous", 4096)

    assert first is repeated
    assert other_grid is not first
    assert calls == [("continuous", 2048), ("continuous", 4096)]
