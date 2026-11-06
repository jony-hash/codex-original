from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.research.lightgen.execute_lightgen_small_decoder_native_gradient_ab_v1 as audit


def test_native_image_preprocessing_reaches_reconstruction_value_and_gradient() -> None:
    class TinyCamera(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.factor = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
            self.seen_images = None

        def forward_from_image(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
            self.seen_images = images.detach().clone()
            return {
                "physical_w_per_pixel": self.factor * images,
                "ols_coefficients": torch.ones((images.shape[0], 100), dtype=torch.complex64),
                "plane_powers_w": torch.ones((images.shape[0], 5), dtype=torch.float64),
            }

    gray_levels = (0, 64, 128, 255)
    pixels = torch.stack([torch.full((28, 28), level, dtype=torch.uint8) for level in gray_levels])
    rows = torch.tensor([3, 0, 2, 1, 3, 2, 1, 0, 3, 2], dtype=torch.long)
    model = TinyCamera()
    scale = torch.ones(100, dtype=torch.float64)
    power_reference = torch.ones(5, dtype=torch.float64)
    gain = torch.tensor(1.0, dtype=torch.float32)

    parts, _ssim, _gradient, _result = audit._loss_parts(
        model, pixels, rows, torch.device("cpu"), scale, power_reference, gain,
    )
    derivative = torch.autograd.grad(parts["rec"], model.factor)[0]

    ordered_intensity = torch.tensor([gray_levels[index] / 255.0 for index in rows.tolist()], dtype=torch.float64)
    mean_square = float(ordered_intensity.square().mean())
    factor = 0.7
    expected_value = (factor - 1.0) ** 2 * mean_square
    expected_derivative = 2.0 * (factor - 1.0) * mean_square
    assert model.seen_images is not None
    assert model.seen_images.shape == (10, 1, 200, 200)
    torch.testing.assert_close(
        model.seen_images[:, 0, 0, 0].to(torch.float64), ordered_intensity,
        rtol=0.0, atol=1e-7,
    )
    assert parts["rec"].item() == pytest.approx(expected_value, rel=2e-6, abs=1e-7)
    assert derivative.item() == pytest.approx(expected_derivative, rel=2e-6, abs=1e-7)
    assert derivative.item() < 0.0
    assert (0.7 + 1e-3 - 1.0) ** 2 * mean_square < expected_value


def test_native_gradient_loss_matches_independent_numpy_and_keeps_overrange_gradient() -> None:
    prediction = torch.tensor(
        [[[[0.0, 1.2, 0.4], [0.3, 0.1, 0.8], [0.9, 0.2, 0.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor(
        [[[[0.0, 0.1, 0.4], [0.3, 0.1, 0.8], [0.9, 0.2, 0.0]]]],
        dtype=torch.float64,
    )
    actual = audit._native_gradient_loss(prediction, target)
    expected = audit._native_gradient_loss_numpy(prediction.detach().numpy(), target.numpy())
    assert actual.item() == pytest.approx(expected, rel=1e-14, abs=1e-14)

    actual.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 0, 1].abs().item() > 0.0
    assert torch.isfinite(prediction.grad).all()

    # Independent central differences check edge pixels as well as interior pixels.
    epsilon = 1e-6
    for row, column in ((0, 0), (0, 2), (1, 1), (2, 2)):
        plus = prediction.detach().clone()
        minus = prediction.detach().clone()
        plus[0, 0, row, column] += epsilon
        minus[0, 0, row, column] -= epsilon
        finite = (audit._native_gradient_loss_numpy(plus.numpy(), target.numpy())
                  - audit._native_gradient_loss_numpy(minus.numpy(), target.numpy())) / (2 * epsilon)
        assert prediction.grad[0, 0, row, column].item() == pytest.approx(finite, rel=2e-6, abs=2e-8)


def test_native_gradient_loss_has_zero_value_for_equal_constant_images() -> None:
    image = torch.full((2, 1, 4, 5), 1.25, dtype=torch.float64, requires_grad=True)
    loss = audit._native_gradient_loss(image, image.detach().clone())
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(image.grad, torch.zeros_like(image))


def test_direction_check_rejects_wrong_autograd_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Parameter(torch.tensor([1.0]))
            self.decoder = torch.nn.Parameter(torch.tensor([0.7, -0.4]))

    model = TinyModel()
    monkeypatch.setattr(audit.base, "_phase_groups", lambda value: (value.encoder, value.decoder))

    class WrongSign(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value: torch.Tensor) -> torch.Tensor:  # noqa: N805
            ctx.save_for_backward(value)
            return value.square().sum()

        @staticmethod
        def backward(ctx, _gradient: torch.Tensor) -> tuple[torch.Tensor]:
            (value,) = ctx.saved_tensors
            return (-2.0 * value,)

    with pytest.raises(RuntimeError, match="finite difference failed"):
        audit._direction_check(model, lambda: WrongSign.apply(model.decoder), "wrong_sign")


def test_restore_keeps_encoder_adam_at_45000_and_rejects_wrong_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Parameter(torch.tensor([1.0]))
            self.decoder = torch.nn.Parameter(torch.tensor([2.0]))

    model = TinyModel()
    monkeypatch.setattr(audit.base, "_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(audit.base, "_phase_groups", lambda value: (value.encoder, value.decoder))
    optimizer = torch.optim.Adam((model.encoder, model.decoder), lr=1e-3)
    for parameter in (model.encoder, model.decoder):
        optimizer.state[parameter] = {
            "step": torch.tensor(45_000.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "optimizer_steps": {"encoder_adam_step": 45_000, "decoder_adam_step": 45_000},
    }
    restored, restored_optimizer = audit._restore_training_state(payload, {}, torch.device("cpu"))
    assert not restored.encoder.requires_grad
    assert restored.decoder.requires_grad
    assert audit._optimizer_phase_steps(restored_optimizer, restored.encoder, restored.decoder) == (45_000, 45_000)

    bad = dict(payload)
    bad["optimizer_steps"] = {"encoder_adam_step": 50_000, "decoder_adam_step": 50_000}
    with pytest.raises(RuntimeError, match="Adam state or parameter mapping"):
        audit._restore_training_state(bad, {}, torch.device("cpu"))


def test_resume_rejects_progress_ahead_of_last_complete_checkpoint() -> None:
    audit._require_progress_matches_checkpoint({"branch_completed_updates": {"D0": 0}}, "D0", 0)
    with pytest.raises(RuntimeError, match="progress claims 500 updates.*checkpoint has 0"):
        audit._require_progress_matches_checkpoint({"branch_completed_updates": {"D0": 500}}, "D0", 0)


def test_paired_generation_summary_consumes_production_csv_rows_and_checks_identity() -> None:
    d0 = [{
        "requested_label": "1", "proposal_id": str(row), "component_id": str(row % 4),
        "ed_prediction": "0", "ec_prediction": "0",
        "proxy_clipping_area_fraction": "0.2", "view_mean_intensity": "0.1",
    } for row in range(10_000)]
    dgrad = [{
        "requested_label": "1", "proposal_id": str(row), "component_id": str(row % 4),
        "ed_prediction": "1", "ec_prediction": "1",
        "proxy_clipping_area_fraction": "0.1", "view_mean_intensity": "0.3",
    } for row in range(10_000)]

    summary = audit._paired_generation_summary(d0, dgrad)
    assert summary["same_ols_coefficients_and_proposal_ids"] is True
    assert summary["ed_correctness_rate_difference_Dgrad_minus_D0"] == pytest.approx(1.0)
    assert summary["ec_correctness_rate_difference_Dgrad_minus_D0"] == pytest.approx(1.0)
    assert summary["proxy_clipping_area_fraction_difference_Dgrad_minus_D0"] == pytest.approx(-0.1)
    assert summary["view_mean_intensity_difference_Dgrad_minus_D0"] == pytest.approx(0.2)

    mismatched = [dict(row) for row in dgrad]
    mismatched[9_999]["proposal_id"] = "99999"
    with pytest.raises(RuntimeError, match="fixed G4 paired generation identities differ"):
        audit._paired_generation_summary(d0, mismatched)


def test_elapsed_summary_accumulates_saved_phases_after_resume_and_validates_times() -> None:
    endpoints = {
        "D0": {"training": {"elapsed_seconds_training": 10.0}, "elapsed_seconds_evaluation": 2.0},
        "Dgrad": {"training": {"elapsed_seconds_training": 20.0}, "elapsed_seconds_evaluation": 3.0},
    }
    resumed = audit._elapsed_seconds_summary({"elapsed_seconds": 15.0}, endpoints, 5.0, [{}])
    assert resumed["elapsed_seconds_total"] == pytest.approx(55.0)
    assert resumed["elapsed_seconds_components"] == {
        "preflight": 15.0, "branch_training": 30.0,
        "branch_evaluation": 5.0, "final_process": 5.0,
    }
    assert "excludes failed uncheckpointed work" in resumed["elapsed_seconds_total_scope"]

    uninterrupted = audit._elapsed_seconds_summary({"elapsed_seconds": 15.0}, endpoints, 42.0, None)
    assert uninterrupted["elapsed_seconds_total"] == 42.0
    with pytest.raises(RuntimeError, match="elapsed-time components"):
        audit._elapsed_seconds_summary({"elapsed_seconds": float("nan")}, endpoints, 5.0, [{}])


def _valid_review_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "attempt_01"
    card_root = run_dir / "review_cards"
    card_root.mkdir(parents=True)
    (run_dir / "D0").mkdir()
    (run_dir / "Dgrad").mkdir()
    for branch in audit.BRANCHES:
        (run_dir / branch / "endpoint_summary.json").write_text("{}", encoding="utf-8")

    cards: list[dict[str, object]] = []
    duplicate_sources: list[dict[str, object]] = []
    card_index = 0
    for branch in audit.BRANCHES:
        for label in range(10):
            for within_class in range(10):
                card_index += 1
                card_id = f"R{card_index:04d}"
                cohort_row = label * 100 + within_class
                relative_image = f"review_cards/{card_id}.png"
                image = np.full((200, 200), (label * 17 + within_class) % 256, dtype=np.uint8)
                from PIL import Image

                Image.fromarray(image, mode="L").save(run_dir / relative_image)
                card: dict[str, object] = {
                    "card_id": card_id, "kind": "generated", "branch": branch,
                    "requested_label": label, "cohort_row": cohort_row,
                    "proposal_id": cohort_row + 1000, "image": relative_image,
                    "duplicate_group": None,
                }
                cards.append(card)
                if within_class == 0:
                    duplicate_sources.append(card)

    for source in duplicate_sources:
        card_index += 1
        card_id = f"R{card_index:04d}"
        relative_image = f"review_cards/{card_id}.png"
        original_path = run_dir / str(source["image"])
        (run_dir / relative_image).write_bytes(original_path.read_bytes())
        group_name = f"dup-{source['branch']}-{source['requested_label']}"
        source["duplicate_group"] = group_name
        cards.append({
            "card_id": card_id, "kind": "duplicate", "branch": source["branch"],
            "requested_label": source["requested_label"], "cohort_row": source["cohort_row"],
            "proposal_id": source["proposal_id"], "image": relative_image,
            "duplicate_group": group_name,
        })
    for label in range(10):
        for replicate in range(2):
            card_index += 1
            card_id = f"R{card_index:04d}"
            relative_image = f"review_cards/{card_id}.png"
            from PIL import Image

            Image.fromarray(np.full((200, 200), label * 20 + replicate, dtype=np.uint8), mode="L").save(
                run_dir / relative_image)
            cards.append({"card_id": card_id, "kind": "real_anchor", "target_label": label,
                          "image": relative_image, "duplicate_group": None})
    assert len(cards) == 240

    review_id = "review-fixture"
    exposed = [{"card_id": card["card_id"], "image": card["image"]} for card in cards]
    (run_dir / "blind_review.html").write_text(audit._render_blind_page(review_id, exposed), encoding="utf-8")
    (run_dir / "machine_report.json").write_text(json.dumps({
        "task_id": audit.TASK_ID, "status": "AWAITING_HUMAN_RATINGS", "measurement_complete": True,
        "run_id": "run-fixture",
    }), encoding="utf-8")
    (run_dir / "blind_review_public_manifest.json").write_text(json.dumps({
        "schema": audit.REVIEW_SCHEMA, "task_id": audit.TASK_ID, "review_id": review_id, "card_count": 240,
    }), encoding="utf-8")
    (run_dir / "blind_review_key.json").write_text(json.dumps({
        "schema": audit.REVIEW_SCHEMA, "task_id": audit.TASK_ID, "review_id": review_id,
        "registered_card_count": 240, "minimum_duplicate_card_distance": 40, "cards": cards,
    }), encoding="utf-8")
    ratings_path = tmp_path / "ratings.json"
    ratings_path.write_text(json.dumps({
        "schema": audit.RATINGS_SCHEMA, "task_id": audit.TASK_ID, "review_id": review_id,
        "cards": [{"card_id": card["card_id"],
                   "rating": card.get("target_label", card.get("requested_label"))}
                  for card in cards],
    }), encoding="utf-8")
    return run_dir, ratings_path


def test_ratings_consumer_accepts_complete_mapping_and_rejects_duplicate_card(tmp_path: Path) -> None:
    run_dir, ratings_path = _valid_review_fixture(tmp_path)
    _machine, _key, validated = audit._validate_review_inputs(run_dir, ratings_path)
    assert validated["duplicate_consistency_count"] == 20
    assert validated["anchor_correct_count"] == 20
    assert len(validated["rating_by_id"]) == 240

    payload = json.loads(ratings_path.read_text(encoding="utf-8"))
    payload["cards"][-1] = dict(payload["cards"][0])
    bad_ratings = tmp_path / "duplicate_ratings.json"
    bad_ratings.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="every registered card exactly once"):
        audit._validate_review_inputs(run_dir, bad_ratings)


def test_fixed_selection_excludes_prior_g4_proposal_ids_deterministically() -> None:
    requested = torch.arange(10).repeat_interleave(1000)
    proposals = torch.arange(10_000)
    previous = {label: set(range(label * 1000, label * 1000 + 10)) for label in range(10)}
    selected_a = audit._select_review_cohort(requested, proposals, previous, 2026102292)
    selected_b = audit._select_review_cohort(requested, proposals, previous, 2026102292)
    assert selected_a == selected_b
    assert len(selected_a) == 100
    assert all(item["proposal_id"] not in previous[item["requested_label"]] for item in selected_a)
    assert [sum(item["requested_label"] == label for item in selected_a) for label in range(10)] == [10] * 10
