from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.research.lightgen import execute_lightgen_small_foreground_balance_ab_v1 as research


def _native_target(seed: int = 21) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 0.05 + 0.9 * torch.rand((1, 1, 200, 200), generator=generator)


def test_balanced_loss_matches_numpy_and_keeps_unclipped_gradient() -> None:
    target = _native_target()
    output = (0.8 * target + 0.1).requires_grad_()
    with torch.no_grad():
        output[0, 0, 0, 0] = 1.2
        output[0, 0, 0, 1] = -0.2

    loss = research._balanced_reconstruction_loss(target, output)
    stroke, background = research._balanced_reconstruction_parts(target, output)
    error = output.detach().numpy() - target.numpy()
    x = target.numpy()
    foreground_mass = x.sum(axis=(1, 2, 3))
    background_mass = (1.0 - x).sum(axis=(1, 2, 3))
    expected_stroke = ((x * error**2).sum(axis=(1, 2, 3)) / foreground_mass)
    expected_background = (((1.0 - x) * error**2).sum(axis=(1, 2, 3)) / background_mass)
    expected_loss = float((0.5 * (expected_stroke + expected_background)).mean())
    assert float(loss) == pytest.approx(expected_loss, rel=1e-6, abs=1e-8)
    assert float(stroke[0]) == pytest.approx(float(expected_stroke[0]), rel=1e-6)
    assert float(background[0]) == pytest.approx(float(expected_background[0]), rel=1e-6)

    gradient = torch.autograd.grad(loss, output)[0]
    expected_gradient = (target * (output.detach() - target) / target.sum()
                         + (1.0 - target) * (output.detach() - target) / (1.0 - target).sum())
    assert torch.allclose(gradient, expected_gradient, rtol=2e-6, atol=1e-8)
    clipped = output.detach().clone().requires_grad_()
    clipped_loss = research._balanced_reconstruction_loss(target, clipped.clamp(0.0, 1.0))
    clipped_gradient = torch.autograd.grad(clipped_loss, clipped)[0]
    assert gradient[0, 0, 0, 0].abs() > 0
    assert clipped_gradient[0, 0, 0, 0] == 0


def test_balanced_loss_rejects_zero_foreground_or_background_mass() -> None:
    output = torch.zeros((1, 1, 200, 200))
    with pytest.raises(RuntimeError, match="denominator"):
        research._balanced_reconstruction_loss(torch.zeros_like(output), output)
    with pytest.raises(RuntimeError, match="denominator"):
        research._balanced_reconstruction_loss(torch.ones_like(output), output)


def test_production_objective_preserves_c_and_replaces_only_v_reconstruction() -> None:
    parts = {
        "dist": torch.tensor(2.0, requires_grad=True),
        "rec": torch.tensor(3.0, requires_grad=True),
        "balanced_rec": torch.tensor(5.0, requires_grad=True),
        "op": torch.tensor(7.0, requires_grad=True),
    }
    structure = torch.tensor(11.0, requires_grad=True)
    weights = {"dist": 0.2, "rec": 0.3, "op": 0.4}
    lambda_ssim = 0.5
    c_value = research._total_loss(parts, structure, weights, lambda_ssim, "C", None)
    v_value = research._total_loss(parts, structure, weights, lambda_ssim, "V", 0.6)
    assert float(c_value) == pytest.approx(0.2 * 2 + 0.3 * 3 + 0.4 * 7 + 0.5 * 11)
    assert float(v_value) == pytest.approx(0.2 * 2 + 0.3 * 0.6 * 5 + 0.4 * 7 + 0.5 * 11)
    c_grads = torch.autograd.grad(c_value, (parts["rec"], parts["balanced_rec"]),
                                  retain_graph=True, allow_unused=True)
    v_grads = torch.autograd.grad(v_value, (parts["rec"], parts["balanced_rec"]), allow_unused=True)
    assert float(c_grads[0]) == pytest.approx(weights["rec"])
    assert c_grads[1] is None
    assert v_grads[0] is None
    assert float(v_grads[1]) == pytest.approx(weights["rec"] * 0.6)


def test_direction_check_rejects_wrong_derivative_sign() -> None:
    with pytest.raises(RuntimeError, match="finite difference failed"):
        research._validate_direction_values("balanced", "encoder", analytic=0.5, finite=-0.5)
    assert research._validate_direction_values("balanced", "encoder", 0.5, 0.501) < 0.01


def test_schedule_consumer_rejects_wrong_rows_and_non_train_rows() -> None:
    labels = torch.arange(40).remainder(10)
    train = torch.arange(30)
    schedule = torch.stack([torch.arange(10) + 10 * (step % 3) for step in range(2)])
    research._validate_schedule_rows(labels, train, schedule, expected_steps=2)
    wrong_label = schedule.clone()
    wrong_label[0, 0] = 1
    with pytest.raises(RuntimeError, match="wrong class row"):
        research._validate_schedule_rows(labels, train, wrong_label, expected_steps=2)
    wrong_role = schedule.clone()
    wrong_role[0, 0] = 30
    with pytest.raises(RuntimeError, match="outside TRAIN"):
        research._validate_schedule_rows(labels, train, wrong_role, expected_steps=2)


def test_review_sample_selection_is_fixed_per_class_and_repeatable() -> None:
    requested = torch.arange(10).repeat_interleave(1000)
    proposals = torch.arange(10_000) + 200
    first, capture_first = research._select_review_samples(requested, proposals, "C", 2026102112)
    second, capture_second = research._select_review_samples(requested, proposals, "C", 2026102112)
    assert first == second
    assert capture_first == capture_second
    assert len(first) == 100
    assert len(capture_first) == 200
    assert [sum(row["requested_label"] == label for row in first) for label in range(10)] == [10] * 10


def test_generation_cohort_consumer_rejects_wrong_population_and_power() -> None:
    requested = torch.arange(10).repeat_interleave(1000)
    proposal_ids = torch.arange(1000).repeat(10)
    samples = torch.zeros((10_000, 100), dtype=torch.complex64)
    accounting = {
        str(label): {"proposal_ids": list(range(1000)), "noise_seed": 2026101601 + label,
                     "retained_samples": 1000, "proposal_limit": 1_000_000, "trials": 1024,
                     "rejection_fraction": 0.0}
        for label in range(10)
    }
    research._validate_sampled_cohort(samples, requested, proposal_ids, accounting, 1_000_000)
    wrong_labels = requested.clone()
    wrong_labels[0] = 1
    with pytest.raises(RuntimeError, match="1,000 samples per class"):
        research._validate_sampled_cohort(samples, wrong_labels, proposal_ids, accounting, 1_000_000)
    over_power = samples.clone()
    over_power[0, 0] = complex(float(research.base.MAX_MODAL_POWER_W**0.5 * 2), 0.0)
    with pytest.raises(RuntimeError, match="power bound"):
        research._validate_sampled_cohort(over_power, requested, proposal_ids, accounting, 1_000_000)


def _review_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    cards_dir = root / "review_cards"
    cards_dir.mkdir()
    cards: list[dict[str, object]] = []
    pixel = np.zeros((200, 200), dtype=np.uint8)

    def add_card(record: dict[str, object]) -> None:
        card_id = f"R{len(cards) + 1:04d}"
        filename = f"card_{len(cards) + 1:04d}.png"
        Image.fromarray(pixel, mode="L").save(cards_dir / filename)
        cards.append({**record, "card_id": card_id, "image": f"review_cards/{filename}"})

    originals: dict[tuple[str, int], dict[str, object]] = {}
    for branch in research.BRANCHES:
        for label in range(10):
            for sample in range(10):
                key = (branch, label)
                duplicate_group = f"duplicate_{branch}_{label}" if sample == 0 else None
                record: dict[str, object] = {
                    "kind": "generated", "branch": branch, "requested_label": label,
                    "sample_index": label * 100 + sample, "proposal_id": 10_000 + label * 100 + sample,
                    "duplicate_group": duplicate_group,
                }
                add_card(record)
                if sample == 0:
                    originals[key] = cards[-1]
    for (branch, label), original in originals.items():
        add_card({key: value for key, value in original.items()
                  if key not in {"card_id", "image"}} | {"kind": "duplicate"})
    for label in range(10):
        for anchor in range(2):
            add_card({"kind": "real_anchor", "target_label": label, "row": label * 2 + anchor,
                      "branch": None, "requested_label": None, "sample_index": None,
                      "proposal_id": None, "duplicate_group": None})
    assert len(cards) == 240
    key = {"schema": research.REVIEW_SCHEMA, "task_id": research.TASK_ID, "review_id": "review-test",
           "registered_card_count": 240, "cards": cards}
    manifest = {"schema": research.REVIEW_SCHEMA, "task_id": research.TASK_ID,
                "review_id": "review-test", "card_count": 240}
    for name, value in (("blind_review_key.json", key), ("blind_review_public_manifest.json", manifest)):
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    (root / "machine_report.json").write_text(json.dumps({
        "task_id": research.TASK_ID, "status": "AWAITING_HUMAN_RATINGS", "measurement_complete": True,
    }), encoding="utf-8")
    for branch in research.BRANCHES:
        (root / branch).mkdir()
        (root / branch / "endpoint_summary.json").write_text("{}", encoding="utf-8")
    ratings = []
    for card in cards:
        rating = card.get("target_label", card.get("requested_label"))
        ratings.append({"card_id": card["card_id"], "rating": rating})
    ratings_payload = {"schema": research.RATINGS_SCHEMA, "task_id": research.TASK_ID,
                       "review_id": "review-test", "cards": list(reversed(ratings))}
    ratings_path = root / "ratings.json"
    ratings_path.write_text(json.dumps(ratings_payload), encoding="utf-8")
    exposed = [{"card_id": card["card_id"], "image": card["image"]} for card in cards]
    (root / "blind_review.html").write_text(
        research._render_blind_review_page("review-test", exposed), encoding="utf-8"
    )
    return root, ratings_path


def test_ratings_consumer_requires_complete_identity_and_keeps_unordered_input(tmp_path: Path) -> None:
    run_dir, ratings_path = _review_fixture(tmp_path / "attempt")
    machine, key, validated = research._validate_review_inputs(run_dir, ratings_path)
    assert machine["status"] == "AWAITING_HUMAN_RATINGS"
    assert len(key["cards"]) == 240
    assert validated["duplicate_consistency_count"] == 20
    assert validated["anchor_correct_count"] == 20
    c = research._human_class_metrics(validated["cards"], validated["rating_by_id"], "C")
    v = research._human_class_metrics(validated["cards"], validated["rating_by_id"], "V")
    effect = research._human_effect_bootstrap(c, v, seed=2026102115, resamples=100)
    assert effect["observed_difference"] == 0.0
    assert effect["ci_95"] == [0.0, 0.0]


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "wrong_task", "wrong_page_review", "wrong_page_mapping",
])
def test_ratings_consumer_rejects_incomplete_or_mismatched_evidence(tmp_path: Path, mutation: str) -> None:
    run_dir, ratings_path = _review_fixture(tmp_path / "attempt")
    if mutation in {"missing", "duplicate", "wrong_task"}:
        payload = json.loads(ratings_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        payload["cards"].pop()
        ratings_path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "duplicate":
        payload["cards"][-1]["card_id"] = payload["cards"][0]["card_id"]
        ratings_path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "wrong_task":
        payload["task_id"] = "OTHER_TASK"
        ratings_path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "wrong_page_review":
        page_path = run_dir / "blind_review.html"
        page_path.write_text(page_path.read_text(encoding="utf-8").replace(
            'REVIEW_ID="review-test"', 'REVIEW_ID="other-review"', 1), encoding="utf-8")
    else:
        page_path = run_dir / "blind_review.html"
        page_path.write_text(page_path.read_text(encoding="utf-8").replace(
            "card_0001.png", "card_0002.png", 1), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        research._validate_review_inputs(run_dir, ratings_path)


def test_review_qc_keeps_disagreement_as_a_measurement_not_silent_drop(tmp_path: Path) -> None:
    run_dir, ratings_path = _review_fixture(tmp_path / "attempt")
    payload = json.loads(ratings_path.read_text(encoding="utf-8"))
    duplicate = next(card for card in json.loads((run_dir / "blind_review_key.json").read_text())
                     ["cards"] if card["kind"] == "duplicate")
    original_id = duplicate["duplicate_group"].replace("duplicate_", "")
    branch, label_text = original_id.rsplit("_", 1)
    key = json.loads((run_dir / "blind_review_key.json").read_text(encoding="utf-8"))
    original = next(card for card in key["cards"] if card["kind"] == "generated"
                    and card["branch"] == branch and card["requested_label"] == int(label_text)
                    and card["sample_index"] == 100 * int(label_text))
    by_id = {card["card_id"]: card for card in payload["cards"]}
    by_id[duplicate["card_id"]]["rating"] = (int(by_id[duplicate["card_id"]]["rating"]) + 1) % 10
    ratings_path.write_text(json.dumps(payload), encoding="utf-8")
    _, _, validated = research._validate_review_inputs(run_dir, ratings_path)
    assert validated["duplicate_consistency_count"] == 19
    assert original["duplicate_group"] == duplicate["duplicate_group"]
