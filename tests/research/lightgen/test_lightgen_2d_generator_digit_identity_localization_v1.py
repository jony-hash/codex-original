from __future__ import annotations

import uuid
from pathlib import Path

import torch
import pytest

from scripts.research.lightgen import audit_lightgen_2d_generator_digit_identity_localization_v1 as localization


def _basis(index: int, width: int = 12) -> torch.Tensor:
    value = torch.zeros(width, dtype=torch.float64)
    value[index] = 1.0
    return value


def _target_groups() -> dict[int, torch.Tensor]:
    return {digit: _basis(digit).repeat(2, 1) for digit in range(localization.CLASS_COUNT)}


def _class_output(digit: int, requested_weight: float) -> torch.Tensor:
    competitor = (digit + 1) % localization.CLASS_COUNT
    value = requested_weight * _basis(digit) + (1.0 - requested_weight) * _basis(competitor)
    return value.repeat(3, 1)


def _decision_rows(weight_a: float, weight_b: float) -> list[dict[str, object]]:
    targets = _target_groups()
    rows = []
    for seed in localization.SEEDS:
        for digit in localization.FOCUS_CLASSES:
            for arm, weight in (("A5400", weight_a), ("B5400", weight_b)):
                margin, nearest, distances = localization._class_margins(_class_output(digit, weight), targets, digit)
                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "requested_class": digit,
                    "target_role": "confirmation",
                    "view": "shape28",
                    "margin": margin,
                    "nearest_class": nearest,
                    "own_class_distance": distances[digit]["energy_distance"],
                })
    return rows


def test_saved_shape_producer_drives_improvement_reversal_and_tolerance_decisions() -> None:
    better = localization._decision(_decision_rows(0.5, 0.9))
    worse = localization._decision(_decision_rows(0.9, 0.5))
    unchanged = localization._decision(_decision_rows(0.5, 0.5))

    assert better["classification"] == "CLASS_SEPARATION_INCREASED"
    assert better["difference_B_minus_A"] > better["tolerance"]
    assert worse["classification"] == "CLASS_SEPARATION_DECREASED"
    assert worse["difference_B_minus_A"] < -worse["tolerance"]
    assert unchanged["classification"] == "CLASS_SEPARATION_WITHIN_TOLERANCE"
    assert unchanged["difference_B_minus_A"] == 0.0


def test_counterexample_own_class_distance_improves_while_separation_worsens() -> None:
    target_index = {0: 2, 1: 1, 2: 0, **{digit: digit for digit in range(3, 10)}}
    targets = {digit: _basis(target_index[digit]).repeat(2, 1) for digit in range(10)}
    a = (0.4 * _basis(0) + 0.6 * _basis(10)).repeat(3, 1)
    b = (0.45 * _basis(0) + 0.45 * _basis(1) + 0.1 * _basis(10)).repeat(3, 1)

    margin_a, _, distances_a = localization._class_margins(a, targets, 2)
    margin_b, _, distances_b = localization._class_margins(b, targets, 2)

    assert distances_b[2]["energy_distance"] < distances_a[2]["energy_distance"]
    assert margin_b < margin_a


def test_output_within_distance_cancels_and_requires_all_nine_competitors() -> None:
    targets = _target_groups()
    output = _class_output(2, 0.8)
    margin, nearest, table = localization._class_margins(output, targets, 2)
    without_output_within = min(
        2.0 * table[digit]["cross_distance"] - table[digit]["within_target_distance"]
        for digit in range(10)
        if digit != 2
    ) - (2.0 * table[2]["cross_distance"] - table[2]["within_target_distance"])

    assert nearest in set(range(10)) - {2}
    assert len(table) == 10
    assert margin == pytest.approx(without_output_within, abs=1e-12)


def test_brightness_scaling_and_same_mean_shape_collapse_are_distinguished() -> None:
    image = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    image[0, 0, 12, 16] = 0.7
    image[0, 0, 90, 121] = 0.3
    for view, size in (("shape28", 28), ("shape200", 200)):
        original = localization._vectorize(image, size, view)
        scaled = localization._vectorize(image * 0.2, size, view)
        assert torch.allclose(original, scaled, rtol=0.0, atol=1e-15)

    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    collapsed = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)
    assert torch.allclose(target.mean(0), collapsed.mean(0))
    assert localization._energy_distance(target, target)["energy_distance"] == pytest.approx(0.0, abs=1e-12)
    assert localization._energy_distance(collapsed, target)["energy_distance"] > 0.0


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros((1, 1, 200, 200)),
        torch.full((1, 1, 200, 200), float("nan")),
        torch.full((1, 1, 200, 200), -0.1),
    ],
)
def test_invalid_image_mass_and_intensity_are_rejected(value: torch.Tensor) -> None:
    with pytest.raises(localization.LocalizationError):
        localization._vectorize(value, 28, "shape28")


def test_incomplete_or_duplicate_primary_rows_are_rejected() -> None:
    rows = _decision_rows(0.5, 0.9)
    with pytest.raises(localization.LocalizationError):
        localization._decision(rows[:-1])
    with pytest.raises(localization.LocalizationError):
        localization._decision(rows + [rows[0]])


def test_sampled_paired_decomposition_rejects_missing_pair_and_changed_targets() -> None:
    rows = []
    for seed in localization.SEEDS:
        for role in ("fit", "confirmation"):
            for view in ("shape28", "shape200", "display28"):
                for digit in range(10):
                    for arm, own, cross, within_x in (("A5400", 0.2, 0.4, 0.1), ("B5400", 0.18, 0.38, 0.08)):
                        rows.append({
                            "arm": arm, "seed": seed, "target_role": role, "view": view,
                            "requested_class": digit, "own_class_distance": own,
                            "own_cross_distance": cross, "own_within_output_distance": within_x,
                            "own_within_target_distance": 0.5,
                        })

    assert len(localization._sampled_decomposition(rows)) == 300
    with pytest.raises(localization.LocalizationError):
        localization._sampled_decomposition(rows[:-1])
    rows[-1]["own_within_target_distance"] = 0.4
    with pytest.raises(localization.LocalizationError):
        localization._sampled_decomposition(rows)


def test_gallery_uses_public_png_record_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    image = torch.ones((1, 1, 28, 28), dtype=torch.float32)
    monkeypatch.setattr(localization.readability, "_png_record", lambda value: ("data:image/png;base64,AA==", "digest"))
    captured: dict[str, str] = {}

    def capture_write_text(path: Path, content: str, **kwargs: object) -> None:
        captured["path"] = str(path)
        captured["content"] = content

    monkeypatch.setattr(Path, "write_text", capture_write_text)

    output = Path("gallery.html")
    localization._gallery(output, [{"endpoint": "A5400", "position": 4, "label": 2}], {("A5400", 4): image}, {})

    assert captured["path"] == "gallery.html"
    assert "data:image/png;base64,AA==" in captured["content"]
    assert "class 2" in captured["content"]


def test_next_explicit_attempt_path_is_accepted() -> None:
    root = Path.cwd() / "test_scratch" / f"output-validator-{uuid.uuid4().hex}"
    output = root / "outputs" / "20261010_m0_generator_digit_identity_localization_v1" / "runs" / "attempt_07"

    localization._validate_output(root, output)


def _representative_preflight_fixture(monkeypatch: pytest.MonkeyPatch, *, break_input: bool = False, break_view: bool = False):
    saved_raw = torch.ones((500, 1, 200, 200), dtype=torch.float32)
    labels = [digit for digit in range(10) for _ in range(20)] + [digit for digit in range(10) for _ in range(30)]
    population = {"labels": labels, "roles": ["fit"] * 200 + ["confirmation"] * 300, "source_row_indices": list(range(500))}
    source_readouts = {"A4700": {"raw_cmos": saved_raw, "target_slm_intensity": saved_raw}}
    targets = {
        "fit": {digit: torch.ones((2, 1, 200, 200), dtype=torch.float32) for digit in range(10)},
        "confirmation": {digit: torch.ones((2, 1, 200, 200), dtype=torch.float32) for digit in range(10)},
    }
    bank = torch.zeros((5, 10, 30, 100), dtype=torch.complex64)
    fixture = {}
    training_fixture = {}
    label_tensor = torch.tensor(labels, dtype=torch.long)
    for arm in ("A5400", "B5400"):
        for chunk_index in range(6):
            start = chunk_index * 50
            raw = torch.ones((50, 1, 200, 200), dtype=torch.float32)
            view = localization.generation.capacity.canonical.cmos_view(raw)
            z = bank[0].reshape(300, 100)[start : start + 50].clone()
            rows = []
            for offset in range(50):
                flat = start + offset
                digit, draw = divmod(flat, 30)
                rows.append({"label": digit, "draw_index": draw, "pair_id": f"{localization.SEEDS[0]}:{digit}:{draw}"})
            if break_input and arm == "B5400" and chunk_index == 0:
                z[0, 0] = 1.0 + 1.0j
            if break_view and arm == "B5400" and chunk_index == 0:
                view = view.clone()
                view[0, 0, 0, 0] = 0.0
            fixture[(arm, chunk_index)] = {
                "task_id": localization.readability.TASK_ID,
                "arm": arm,
                "seed": localization.SEEDS[0],
                "flat_start": start,
                "rows": rows,
                "raw_cmos": raw,
                "evaluator_view": view,
                "input_z": z,
            }
            training_fixture[(arm, chunk_index)] = {
                "task_id": localization.training.TASK_ID,
                "scenario": arm,
                "positions": torch.arange(start, start + 50),
                "source_row_indices": torch.arange(start, start + 50),
                "labels": label_tensor[start : start + 50],
                "roles": population["roles"][start : start + 50],
                "raw_cmos": raw,
                "evaluator_view": view,
                "target_slm_intensity": saved_raw[start : start + 50],
            }
    for arm in ("A5400", "B5400"):
        for chunk_index in range(6, 10):
            start = chunk_index * 50
            raw = torch.ones((50, 1, 200, 200), dtype=torch.float32)
            training_fixture[(arm, chunk_index)] = {
                "task_id": localization.training.TASK_ID,
                "scenario": arm,
                "positions": torch.arange(start, start + 50),
                "source_row_indices": torch.arange(start, start + 50),
                "labels": label_tensor[start : start + 50],
                "roles": population["roles"][start : start + 50],
                "raw_cmos": raw,
                "evaluator_view": localization.generation.capacity.canonical.cmos_view(raw),
                "target_slm_intensity": saved_raw[start : start + 50],
            }

    def fake_load(path, *args, **kwargs):
        path = str(path)
        arm = "A5400" if "A5400" in path else "B5400"
        chunk_index = int(path.rsplit("_", 1)[1].split(".")[0])
        if "seed_" in path:
            return fixture[(arm, chunk_index)]
        return training_fixture[(arm, chunk_index)]

    monkeypatch.setattr(localization.torch, "load", fake_load)
    return bank, population, source_readouts, targets


def test_representative_preflight_consumes_saved_inputs_and_view(monkeypatch: pytest.MonkeyPatch) -> None:
    bank, population, source_readouts, targets = _representative_preflight_fixture(monkeypatch)
    result = localization._preflight_samples(Path("unused"), Path("unused"), bank, population, source_readouts, targets)
    assert result["passed"] is True
    assert result["rows"] == 50


@pytest.mark.parametrize("break_input,break_view", [(True, False), (False, True)])
def test_representative_preflight_rejects_input_or_saved_view_mismatch(monkeypatch: pytest.MonkeyPatch, break_input: bool, break_view: bool) -> None:
    bank, population, source_readouts, targets = _representative_preflight_fixture(monkeypatch, break_input=break_input, break_view=break_view)
    with pytest.raises(localization.LocalizationError):
        localization._preflight_samples(Path("unused"), Path("unused"), bank, population, source_readouts, targets)
