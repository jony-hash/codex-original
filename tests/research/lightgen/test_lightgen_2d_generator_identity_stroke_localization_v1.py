from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.lightgen import audit_lightgen_2d_generator_identity_stroke_localization_v1 as audit


def _prediction_pair(target: int, transition_index: int) -> tuple[int, int]:
    wrong_a, wrong_b = (target + 1) % 10, (target + 2) % 10
    return (
        (target, target),
        (target, wrong_b),
        (wrong_a, target),
        (wrong_a, wrong_b),
    )[transition_index]


def _analysis_row(
    population: str,
    seed: int,
    scenario: str,
    target: int,
    endpoint: str,
    transition_index: int,
    *,
    source_row_index: int | str = "",
    sample_seed: int | str = "",
    draw_index: int | str = "",
) -> dict[str, str]:
    row: dict[str, str] = {
        "population": population,
        "training_seed": str(seed),
        "scenario": scenario,
        "sample_seed": str(sample_seed),
        "requested_class": str(target if population == "conditional_sample" else ""),
        "draw_index": str(draw_index),
        "source_row_index": str(source_row_index),
        "label": str(target),
        "q_fg": "0.25" if endpoint == "A" else "0.30",
        "raw_cmos_integral": "1e-8" if endpoint == "A" else "0.8e-8",
        "clipped_fraction": "0.0",
        "e_d_identity_margin": "1.0" if _prediction_pair(target, (transition_index + 3) % 4)[0 if endpoint == "A" else 1] == target else "-1.0",
    }
    for evaluator_index, evaluator in enumerate(audit.EVALUATORS):
        a_prediction, b_prediction = _prediction_pair(target, (transition_index + evaluator_index) % 4)
        row[f"{evaluator.lower()}_prediction"] = str(a_prediction if endpoint == "A" else b_prediction)
    return row


def _full_analysis_fixture() -> tuple[list[dict[str, str]], dict[tuple[str, int, int], dict[str, int]]]:
    rows: list[dict[str, str]] = []
    positions: dict[tuple[str, int, int], dict[str, int]] = {}
    for seed in (101, 102, 103):
        for position in range(audit.DEV_ROWS):
            target = position // audit.DEV_ROWS_PER_CLASS
            transition_index = position % 4
            source_index = 50_000 + position
            for endpoint in ("A", "B"):
                scenario = f"seed_{seed}_{endpoint}_step6700"
                rows.append(
                    _analysis_row(
                        "empirical_dev", seed, scenario, target, endpoint, transition_index,
                        source_row_index=source_index,
                    )
                )
                positions[(scenario, seed, source_index)] = {"position": position, "label": target}
        for sample_index in range(5):
            sample_seed = 900 + sample_index
            for target in range(10):
                for draw_index in range(audit.SAMPLE_ROWS_PER_CLASS):
                    transition_index = draw_index % 4
                    for endpoint in ("A", "B"):
                        scenario = f"seed_{seed}_{endpoint}_step6700"
                        rows.append(
                            _analysis_row(
                                "conditional_sample", seed, scenario, target, endpoint, transition_index,
                                sample_seed=sample_seed, draw_index=draw_index,
                            )
                        )
    rows.extend({"scenario": "A4700"} for _ in range(2_500))
    return rows, positions


def _source_confusions(pairs: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {
            "population": str(row["population"]),
            "training_seed": str(row["training_seed"]),
            "evaluator": str(row["evaluator"]),
            "true_or_requested_class": str(row["target_or_requested_class"]),
            "A_prediction": str(row["a_prediction"]),
            "B_prediction": str(row["b_prediction"]),
            "paired_outcome": str(row["correctness_transition"]),
            "count": str(row["pairs"]),
        }
        for row in audit._confusion_rows(pairs)
    ]


def test_full_population_pairing_recomputes_all_confusions_and_selection() -> None:
    source_rows, positions = _full_analysis_fixture()
    pairs = audit._pair_rows(source_rows, positions)

    assert len(source_rows) == 17_500
    assert len(pairs) == 7_500
    assert audit._verify_source_confusions(pairs, _source_confusions(pairs))["exact_match"] is True
    assert {row["e_d_transition"] for row in pairs} == set(audit.TRANSITIONS)

    selected_a, strata_a = audit._select_pairs(pairs)
    selected_b, strata_b = audit._select_pairs(pairs)
    assert len(selected_a) == len(selected_b) == 288
    assert selected_a == selected_b
    assert strata_a == strata_b
    assert len(strata_a) == 2 * 3 * len(audit.FOCUS_CLASSES) * len(audit.TRANSITIONS)


def test_duplicate_and_mismatched_dev_keys_are_rejected() -> None:
    source_rows, positions = _full_analysis_fixture()
    first_dev = next(row for row in source_rows if row.get("population") == "empirical_dev")
    with pytest.raises(audit.EvidenceError, match="duplicate A/B row key"):
        audit._pair_rows([*source_rows, first_dev], positions)

    mismatched = [dict(row) for row in source_rows]
    a_row = next(row for row in mismatched if row.get("population") == "empirical_dev" and row["scenario"].endswith("_A_step6700"))
    b_row = next(
        row for row in mismatched
        if row.get("population") == "empirical_dev"
        and row["scenario"].endswith("_B_step6700")
        and row["source_row_index"] == a_row["source_row_index"]
        and row["training_seed"] == a_row["training_seed"]
    )
    b_row["label"] = str((int(b_row["label"]) + 1) % 10)
    with pytest.raises(audit.EvidenceError, match="A/B target class differs"):
        audit._pair_rows(mismatched, positions)


def test_unit_mass_shape_separates_scalar_energy_from_spatial_change() -> None:
    a = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    a[:, :, 40:80, 40:80] = 0.1
    a[:, :, 100:120, 130:160] = 0.05
    scaled_b = 2.0 * a
    shape_a, mass_a = audit._normalized(a)
    shape_scaled, mass_scaled = audit._normalized(scaled_b)
    assert mass_scaled == 2.0 * mass_a
    assert torch.allclose(shape_a, shape_scaled)
    assert torch.count_nonzero(shape_a - shape_scaled) == 0
    view_a = audit.cmos_view(a.to(torch.float32))
    view_scaled = audit.cmos_view(scaled_b.to(torch.float32))
    assert float((view_a - view_scaled).abs().mean()) > 0.0
    assert torch.allclose(audit.cmos_view((scaled_b * 0.5).to(torch.float32)), view_a)

    changed_b = torch.roll(a, shifts=45, dims=-1)
    shape_changed, mass_changed = audit._normalized(changed_b)
    assert mass_changed == mass_a
    assert float((shape_a - shape_changed).abs().sum()) > 0.0
    assert float((view_a - audit.cmos_view(changed_b.to(torch.float32))).abs().mean()) > 0.0


def _write_dev_chunk(
    root: Path,
    scenario: str,
    raw_row: torch.Tensor,
    target: torch.Tensor,
    *,
    saved_view_override: torch.Tensor | None = None,
) -> None:
    raw = torch.zeros((audit.RAW_CHUNK_ROWS, 1, 200, 200), dtype=torch.float32)
    targets = torch.zeros_like(raw)
    offset = 7
    raw[offset] = raw_row[0]
    targets[offset] = target[0]
    logits = {name: torch.zeros((audit.RAW_CHUNK_ROWS, 10), dtype=torch.float32) for name in audit.EVALUATORS}
    for values in logits.values():
        values[offset, 1] = 2.0
    view = audit.cmos_view(raw)
    if saved_view_override is not None:
        view[offset] = saved_view_override[0]
    path = root / "readouts" / scenario / "chunk_0000.pt"
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "task_id": "M0_GENERATOR_CLOUD_FOREGROUND_DEV_MULTISEED_V1",
            "scenario": scenario,
            "raw_cmos": raw,
            "evaluator_view": view,
            "logits": logits,
            "positions": torch.arange(audit.RAW_CHUNK_ROWS),
            "source_row_indices": torch.zeros(audit.RAW_CHUNK_ROWS, dtype=torch.long).index_fill(0, torch.tensor([offset]), 12_345),
            "labels": torch.ones(audit.RAW_CHUNK_ROWS, dtype=torch.long),
            "roles": ["dev"] * audit.RAW_CHUNK_ROWS,
            "target_slm_intensity": targets,
        },
        path,
    )


def _selected_dev_pair(raw_a: torch.Tensor, raw_b: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    serialized_target = target.to(torch.float32).to(torch.float64)
    q_a = audit._qfg_numpy(raw_a.to(torch.float32).to(torch.float64), serialized_target)["q_fg"]
    q_b = audit._qfg_numpy(raw_b.to(torch.float32).to(torch.float64), serialized_target)["q_fg"]
    row_a = {
        "population": "empirical_dev", "training_seed": "101", "scenario": "seed_101_A_step6700",
        "sample_seed": "", "requested_class": "", "draw_index": "", "source_row_index": "12345", "label": "1",
        "q_fg": str(q_a), "e_d_identity_margin": "2.0",
    }
    row_a.update({f"{evaluator.lower()}_prediction": "1" for evaluator in audit.EVALUATORS})
    row_b = {**row_a, "scenario": "seed_101_B_step6700", "q_fg": str(q_b)}
    return {
        "population": "empirical_dev", "training_seed": 101, "target_class": 1, "source_row_index": "12345",
        "sample_seed": "", "draw_index": "", "sample_key": "empirical_dev:101:12345",
        "a_scenario": row_a["scenario"], "b_scenario": row_b["scenario"], "a_position": 7, "b_position": 7,
        "a_q_fg": q_a, "b_q_fg": q_b, "e_d_transition": "both_correct", "selection_rank": 1,
        "_a_csv_row": row_a, "_b_csv_row": row_b,
    }


def test_saved_raw_chunk_replays_production_view_logits_and_metrics(tmp_path: Path) -> None:
    target = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    target[:, :, 50:150, 90:110] = 0.8
    raw_a, raw_b = target * 1.0e-6, target * 2.0e-6
    selected = _selected_dev_pair(raw_a, raw_b, target)
    _write_dev_chunk(tmp_path, str(selected["a_scenario"]), raw_a, target)
    _write_dev_chunk(tmp_path, str(selected["b_scenario"]), raw_b, target)

    replayed = audit._read_selected_raw(tmp_path, [selected])
    metrics, panels = audit._pair_metrics_and_panels(
        selected,
        replayed[(str(selected["sample_key"]), "A")],
        replayed[(str(selected["sample_key"]), "B")],
    )
    assert metrics["q_fg_saved_replay_error_a"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["q_fg_saved_replay_error_b"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["b_to_a_energy_match_factor"] == pytest.approx(0.5)
    assert metrics["production_view_a_energy_matched_b_mean_absolute_gap"] == pytest.approx(0.0)
    assert metrics["unit_mass_shape_total_variation_a_b"] == pytest.approx(0.0, abs=1e-12)
    assert len(panels) == 9
    assert all(panel["data"] == "" or panel["data"].startswith("data:image/png;base64,") for panel in panels)


def test_saved_raw_chunk_replay_rejects_wrong_production_view(tmp_path: Path) -> None:
    target = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    target[:, :, 50:150, 90:110] = 0.8
    raw = target * 1.0e-6
    selected = _selected_dev_pair(raw, raw, target)
    wrong_view = torch.ones((1, 1, 28, 28), dtype=torch.float32)
    _write_dev_chunk(tmp_path, str(selected["a_scenario"]), raw, target, saved_view_override=wrong_view)
    _write_dev_chunk(tmp_path, str(selected["b_scenario"]), raw, target)

    with pytest.raises(audit.EvidenceError, match="raw-to-production view does not replay"):
        audit._read_selected_raw(tmp_path, [selected])
