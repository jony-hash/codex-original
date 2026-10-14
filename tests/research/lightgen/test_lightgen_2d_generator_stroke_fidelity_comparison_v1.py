from __future__ import annotations

import csv
import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.research.lightgen import execute_lightgen_2d_generator_stroke_fidelity_comparison_v1 as stroke


def _independent_ssim(image: list[list[float]], target: list[list[float]]) -> float:
    coordinate = [float(index - 5) for index in range(11)]
    one_d = [math.exp(-(value**2) / (2.0 * 1.5**2)) for value in coordinate]
    total = math.fsum(one_d)
    one_d = [value / total for value in one_d]
    window = [[one_d[row] * one_d[column] for column in range(11)] for row in range(11)]
    values = []
    for row in range(18):
        for column in range(18):
            x = [image[row + dy][column + dx] for dy in range(11) for dx in range(11)]
            y = [target[row + dy][column + dx] for dy in range(11) for dx in range(11)]
            weights = [window[dy][dx] for dy in range(11) for dx in range(11)]
            mu_x = math.fsum(weight * value for weight, value in zip(weights, x, strict=True))
            mu_y = math.fsum(weight * value for weight, value in zip(weights, y, strict=True))
            var_x = math.fsum(weight * (value - mu_x) ** 2 for weight, value in zip(weights, x, strict=True))
            var_y = math.fsum(weight * (value - mu_y) ** 2 for weight, value in zip(weights, y, strict=True))
            covariance = math.fsum(weight * (left - mu_x) * (right - mu_y) for weight, left, right in zip(weights, x, y, strict=True))
            numerator = (2 * mu_x * mu_y + 1e-4) * (2 * covariance + 9e-4)
            denominator = (mu_x**2 + mu_y**2 + 1e-4) * (var_x + var_y + 9e-4)
            values.append(numerator / denominator)
    return math.fsum(values) / len(values)


def test_ssim_matches_independent_weighted_window_reference_and_identity() -> None:
    generator = torch.Generator(device="cpu").manual_seed(314159)
    image_tensor = 0.05 + 0.9 * torch.rand((28, 28), generator=generator, dtype=torch.float64)
    target_tensor = 0.05 + 0.9 * torch.rand((28, 28), generator=generator, dtype=torch.float64)
    image = image_tensor.tolist()
    target = target_tensor.tolist()
    observed = stroke.ssim_per_image(image_tensor[None, None], target_tensor[None, None])
    assert observed.dtype == torch.float64
    assert float(observed[0]) == pytest.approx(_independent_ssim(image, target), abs=2e-12)
    identical = stroke.ssim_per_image(image_tensor[None, None], image_tensor[None, None])
    assert float(identical[0]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ssim_derivative_matches_independent_central_difference(dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cpu").manual_seed(2026100817)
    image = (0.25 + 0.5 * torch.rand((1, 1, 28, 28), generator=generator, dtype=dtype)).requires_grad_(True)
    target = 0.2 + 0.6 * torch.rand((1, 1, 28, 28), generator=generator, dtype=dtype)
    direction = torch.randn(image.shape, generator=generator, dtype=torch.float64)
    direction /= direction.norm()
    loss = 1.0 - stroke.ssim_per_image(image, target).mean()
    gradient = torch.autograd.grad(loss, image)[0].to(torch.float64)
    epsilon = 1e-5

    def value(offset: float) -> float:
        candidate = image.detach().to(torch.float64) + offset * direction
        return float((1.0 - stroke.ssim_per_image(candidate, target.to(torch.float64)).mean()).detach())

    observed = (value(epsilon) - value(-epsilon)) / (2 * epsilon)
    predicted = float((gradient * direction).sum())
    assert predicted == pytest.approx(observed, rel=2e-4, abs=2e-7)


def test_reconstruction_gradient_accumulates_microbatches_before_normalization() -> None:
    generator = torch.Generator(device="cpu").manual_seed(2026100818)
    target = torch.rand((10, 1, 200, 200), generator=generator, dtype=torch.float64) * 0.45 + 0.2
    base = torch.rand((10, 1, 200, 200), generator=generator, dtype=torch.float64) * 0.1 + 0.35
    pattern = torch.rand((10, 1, 200, 200), generator=generator, dtype=torch.float64) * 0.1 - 0.05
    theta = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)

    def generated(parameter: torch.Tensor) -> torch.Tensor:
        return base + parameter * pattern

    full_a = stroke.reconstruction_terms(generated(theta), target, structural=False)["objective"]
    grad_a = torch.autograd.grad(full_a, theta, retain_graph=True)[0]
    full_b = stroke.reconstruction_terms(generated(theta), target, structural=True)["objective"]
    grad_b = torch.autograd.grad(full_b, theta, retain_graph=True)[0]
    accumulated = torch.zeros_like(theta)
    for start in range(0, 10, stroke.MICROBATCH_SIZE):
        stop = start + stroke.MICROBATCH_SIZE
        piece = stroke.reconstruction_terms(generated(theta)[start:stop], target[start:stop], structural=True)["objective"]
        accumulated += ((stop - start) / 10) * torch.autograd.grad(piece, theta, retain_graph=True)[0]
    assert torch.allclose(accumulated, grad_b, rtol=1e-10, atol=1e-12)
    assert not torch.allclose(grad_a, grad_b, rtol=1e-4, atol=1e-8)


def test_frozen_encoder_tensor_check_accepts_exact_copy_and_rejects_mutation() -> None:
    original = torch.zeros((2, 1, 100, 100), dtype=torch.float32)
    stroke._assert_tensor_unchanged(original, original.clone(), "Encoder")
    mutated = original.clone()
    mutated[0, 0, 0, 0] = 1e-7
    with pytest.raises(stroke.StrokeFidelityError, match="Encoder tensor changed"):
        stroke._assert_tensor_unchanged(original, mutated, "Encoder")


def test_aa_repeatability_uses_declared_float_tolerance_and_rejects_real_drift() -> None:
    reference = {
        "phase": torch.zeros(20_000, dtype=torch.float32),
        "optimizer": {"state": {0: {"step": torch.tensor(4_703.0), "exp_avg": torch.zeros(20_000), "exp_avg_sq": torch.zeros(20_000)}}, "param_groups": [{"params": [0], "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8}]},
        "history": [{"raw_mse": 0.1, "optimizer_step": 4_701}],
    }
    replay = {
        "phase": torch.zeros(20_000, dtype=torch.float32),
        "optimizer": {"state": {0: {"step": torch.tensor(4_703.0), "exp_avg": torch.zeros(20_000), "exp_avg_sq": torch.zeros(20_000)}}, "param_groups": [{"params": [0], "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8}]},
        "history": [{"raw_mse": 0.1 + 1e-8, "optimizer_step": 4_701}],
    }
    replay["phase"][0] = 1e-8
    replay["optimizer"]["state"][0]["exp_avg"][0] = 4e-9
    assert stroke._trial_repeatability(reference, replay)["passed"] is True
    replay["optimizer"]["state"][0]["exp_avg"][0] = 2e-8
    optimizer_drift = stroke._trial_repeatability(reference, replay)
    assert optimizer_drift["passed"] is False
    assert optimizer_drift["comparisons"]["optimizer_exp_avg"]["passed"] is False
    replay["optimizer"]["state"][0]["exp_avg"][0] = 0.0
    replay["history"][0]["optimizer_step"] = 4_702
    result = stroke._trial_repeatability(reference, replay)
    assert result["passed"] is False
    assert result["comparisons"]["training_history"]["passed"] is False


def test_aa_preflight_record_exposes_actual_result_to_ready_consumer() -> None:
    state = {
        "phase": torch.zeros(20_000),
        "optimizer": {"state": {0: {"step": torch.tensor(4_703.0), "exp_avg": torch.zeros(20_000), "exp_avg_sq": torch.zeros(20_000)}}, "param_groups": [{"params": [0], "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8}]},
        "history": [{"optimizer_step": 4_701}],
    }
    record = stroke._aa_preflight_record(state, state, updates=3)
    assert record["executed"] is True
    assert record["passed"] is True
    assert record["A_A_repeatability"]["passed"] is True


def _toy_training_inputs(monkeypatch):
    encoder = torch.nn.Parameter(torch.zeros((2, 1, 100, 100), dtype=torch.float32))
    generator = torch.nn.Parameter(torch.zeros((20_001,), dtype=torch.float32))
    parameters = {"encoder.phase": encoder, "generator.phase": generator}
    labels = torch.arange(200, dtype=torch.long) % 10
    images = torch.zeros((200, 1, 28, 28), dtype=torch.float32)
    for index, label in enumerate(labels.tolist()):
        images[index, 0, label, (label + index // 10) % 28] = 1.0
    rows = torch.arange(200, dtype=torch.long) + 4_000
    fit = {"images": images, "labels": labels, "source_indices": rows}

    class ToyRuntime:
        def __init__(self):
            self.generator = SimpleNamespace()
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, 200), torch.linspace(-1, 1, 200), indexing="ij")
            self.spatial = (0.04 * (xx + yy)).view(1, 1, 200, 200)
            self.port_pattern = torch.linspace(-0.2, 0.2, 100).view(1, 100)

        def forward_from_image(self, batch):
            target = F.interpolate(batch, size=(200, 200), mode="bilinear", align_corners=False)
            offset = 0.001 * encoder.mean() + 0.002 * generator[0] + 0.003 * generator[20_000]
            raw = 0.05 + 0.8 * target + offset + self.spatial.to(batch.device)
            signal = (batch.flatten(1) * torch.linspace(-1, 1, 784, device=batch.device)).sum(1, keepdim=True)
            encoder_offset = 0.001 * encoder.mean()
            real = 0.2 * signal + encoder_offset + self.port_pattern.to(batch.device)
            imag = 0.1 * signal + 0.5 * encoder_offset + self.port_pattern.flip(1).to(batch.device)
            coefficients = torch.complex(real.expand(-1, 100), imag.expand(-1, 100))
            planes = raw.mean((1, 2, 3)).view(-1, 1).expand(-1, 5)
            return SimpleNamespace(raw_intensity=raw, image_input=SimpleNamespace(slm_intensity=target), z_ols=SimpleNamespace(coefficients=coefficients), plane_powers=planes)

    consumer = SimpleNamespace(runtime=ToyRuntime(), device=torch.device("cpu"), references=SimpleNamespace(scale=torch.ones(100), power_reference=torch.full((5,), 0.2)))
    evaluators = {}
    for name, seed in zip(stroke.TRAIN_EVALUATORS, (1, 2, 3), strict=True):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = torch.nn.Linear(28 * 28, 10, bias=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        evaluators[name] = model
    monkeypatch.setattr(stroke.capacity, "_evaluator_logits", lambda model, view, _name: model(view.flatten(1)))
    monkeypatch.setattr(stroke.diagnostic.bogt, "contracted_plane_powers", lambda result: result.plane_powers)
    monkeypatch.setattr(stroke.diagnostic.bogt, "optical_power_loss_from_powers", lambda powers, reference: (powers - reference).square().mean())
    return consumer, parameters, evaluators, fit


def test_logical_batch_producer_consumes_ssim_gradient_and_generator_update(monkeypatch) -> None:
    consumer, parameters, evaluators, fit = _toy_training_inputs(monkeypatch)
    images = fit["images"][:10]
    labels = fit["labels"][:10]
    _ref_planes, _ref_losses, _ref_gradients, reference_a = stroke.capacity._large_microbatch_loss_gradients(consumer, parameters, evaluators, images, labels)
    _powers, _losses, gradients, combined_b = stroke._logical_batch_gradients(consumer, parameters, evaluators, images, labels, structural=True, verify_ssim_vjp=True)
    for name in ("encoder.phase", "generator.phase"):
        expected_difference = stroke.capacity.semantic.BETA * stroke.SSIM_WEIGHT * gradients["ssim"][name]
        actual_difference = combined_b[name] - reference_a[name]
        assert torch.allclose(actual_difference, expected_difference, rtol=2e-3, atol=2e-7)
    assert all(parameter.grad is None for parameter in parameters.values())

    optimizer = torch.optim.Adam([parameters["generator.phase"]], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    phase = parameters["generator.phase"]
    optimizer.state[phase] = {"step": torch.tensor(float(stroke.SOURCE_STEP)), "exp_avg": torch.zeros_like(phase), "exp_avg_sq": torch.zeros_like(phase)}
    record = stroke._train_step(consumer, optimizer, parameters, evaluators, fit, stroke.ARM_B, 0)
    assert record["optimizer_step"] == stroke.SOURCE_STEP + 1
    assert record["added_400x400_gradient_norm"] > 0.0
    assert record["ssim_gradient_norm"] > 0.0
    assert all(parameter.grad is None for parameter in parameters.values())


def test_logical_batch_fixed_reconstruction_target_changes_only_paired_reconstruction(monkeypatch) -> None:
    consumer, parameters, evaluators, fit = _toy_training_inputs(monkeypatch)
    images = fit["images"][:10]
    labels = fit["labels"][:10]
    base = consumer.runtime.forward_from_image(images)
    original_targets = base.image_input.slm_intensity.detach()
    del base
    scales = torch.linspace(0.45, 0.75, 10, dtype=torch.float32).view(10, 1, 1, 1)
    scaled_targets = (original_targets * scales).detach()

    # s=1 is the control: the production consumer must exactly recover the
    # ordinary paired-target path, including its complete combined direction.
    _, losses_a, _, gradient_a = stroke._logical_batch_gradients(
        consumer, parameters, evaluators, images, labels, structural=True
    )
    _, losses_identity, _, gradient_identity = stroke._logical_batch_gradients(
        consumer,
        parameters,
        evaluators,
        images,
        labels,
        structural=True,
        reconstruction_targets=original_targets,
    )
    assert losses_identity == pytest.approx(losses_a)
    for name in gradient_a:
        assert torch.equal(gradient_identity[name], gradient_a[name])

    _, losses_b, _, gradient_b = stroke._logical_batch_gradients(
        consumer,
        parameters,
        evaluators,
        images,
        labels,
        structural=True,
        reconstruction_targets=scaled_targets,
    )
    assert all(losses_b[f"{name}_ce"] == pytest.approx(losses_a[f"{name}_ce"], abs=0.0) for name in stroke.TRAIN_EVALUATORS)
    assert losses_b["l_dist"] == pytest.approx(losses_a["l_dist"], abs=0.0)

    def reconstruction_gradient(targets: torch.Tensor) -> dict[str, torch.Tensor]:
        accumulated = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
        for start in range(0, stroke.BATCH_SIZE, stroke.MICROBATCH_SIZE):
            stop = start + stroke.MICROBATCH_SIZE
            result = consumer.runtime.forward_from_image(images[start:stop])
            objective = stroke.reconstruction_terms(
                result.raw_intensity,
                targets[start:stop],
                structural=True,
            )["objective"]
            gradients = torch.autograd.grad(
                objective,
                tuple(parameters.values()),
                retain_graph=False,
                allow_unused=True,
            )
            fraction = (stop - start) / stroke.BATCH_SIZE
            for (name, parameter), gradient in zip(parameters.items(), gradients, strict=True):
                accumulated[name] += fraction * (torch.zeros_like(parameter) if gradient is None else gradient.detach())
            del result, objective, gradients
        return accumulated

    expected_delta = reconstruction_gradient(scaled_targets)
    original_delta = reconstruction_gradient(original_targets)
    for name in gradient_a:
        observed = gradient_b[name] - gradient_a[name]
        expected = stroke.capacity.semantic.BETA * (expected_delta[name] - original_delta[name])
        assert torch.allclose(observed, expected, rtol=2e-4, atol=2e-8)
    assert all(parameter.grad is None for parameter in parameters.values())


@pytest.mark.parametrize(
    "bad_target",
    [
        torch.empty((10, 1, 199, 200), dtype=torch.float32),
        torch.full((10, 1, 200, 200), float("nan"), dtype=torch.float32),
        torch.ones((10, 1, 200, 200), dtype=torch.float64),
    ],
)
def test_logical_batch_fixed_reconstruction_target_rejects_invalid_mapping_shape_or_values(monkeypatch, bad_target) -> None:
    consumer, parameters, evaluators, fit = _toy_training_inputs(monkeypatch)
    with pytest.raises(stroke.StrokeFidelityError, match="fixed reconstruction target"):
        stroke._logical_batch_gradients(
            consumer,
            parameters,
            evaluators,
            fit["images"][:10],
            fit["labels"][:10],
            structural=True,
            reconstruction_targets=bad_target,
        )


def _synthetic_inputs_and_readouts() -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    labels = torch.tensor(list(range(10)) * 20 + list(range(10)) * 30, dtype=torch.long)
    rows = torch.arange(500, dtype=torch.long) + 1000
    image_bank = torch.zeros((500, 1, 28, 28), dtype=torch.float32)
    for index, label in enumerate(labels.tolist()):
        image_bank[index, 0, label, label] = 1.0
    readouts = {}
    for name in ("A4700", "A5400", "B5400"):
        logits = torch.zeros((500, 10), dtype=torch.float32)
        logits[torch.arange(500), labels] = 10.0
        readouts[name] = {"source_indices": rows.clone(), "labels": labels.clone(), "evaluator_view": image_bank.clone(), "e_d_logits": logits}
    return {"labels": labels, "rows": rows, "images": image_bank}, readouts


def _synthetic_endpoint_summaries() -> dict[str, dict[str, object]]:
    by_class = {str(digit): {"raw_fraction_above_one": 0.0} for digit in range(10)}
    metrics = {
        "raw_mse": 0.1,
        "raw_fraction_above_one": 0.0,
        "raw_zero_image": 0,
        "raw_constant_image": 0,
        "by_class": by_class,
    }
    return {name: {"image_metrics_by_role": {"confirmation": dict(metrics, by_class=dict(by_class))}, "all_finite": True} for name in ("A4700", "A5400", "B5400")}


def test_production_review_packet_blinds_mapping_and_scores_complete_valid_ratings(tmp_path) -> None:
    inputs, readouts = _synthetic_inputs_and_readouts()
    packet = stroke._prepare_review(tmp_path, inputs, readouts)
    mapping_payload = json.loads((tmp_path / "review_mapping.json").read_text(encoding="utf-8"))
    mapping = mapping_payload["items"]
    html = (tmp_path / "blind_review.html").read_text(encoding="utf-8")
    assert packet["item_count"] == stroke.REVIEW_TOTAL == 680
    assert sum(row["kind"] == "primary_generated" for row in mapping) == 600
    assert sum(row["kind"] == "real_target_anchor" for row in mapping) == 50
    assert sum(row["kind"] == "hidden_repeat" for row in mapping) == 30
    assert all(sum(row["kind"] == "primary_generated" and row["endpoint"] == endpoint and row["true_class"] == digit for row in mapping) == 20 for endpoint in ("start", "A", "B") for digit in range(10))
    assert all(field not in html for field in ("true_class", "endpoint", "e_d_prediction", "measurement_position", "source_row_index"))
    stroke._validate_review_mapping(mapping)

    answers: dict[str, str] = {}
    primary = [row for row in mapping if row["kind"] == "primary_generated"]
    for endpoint in ("start", "A", "B"):
        for digit in range(10):
            class_rows = [row for row in primary if row["endpoint"] == endpoint and row["true_class"] == digit]
            for index, row in enumerate(class_rows):
                answers[row["item_id"]] = str(digit if endpoint == "B" or index < 10 else (digit + 1) % 10)
    for row in mapping:
        if row["kind"] == "real_target_anchor":
            answers[row["item_id"]] = str(row["true_class"])
    for row in mapping:
        if row["kind"] == "hidden_repeat":
            answers[row["item_id"]] = answers[str(row["repeat_of"])]
    ratings = tmp_path / "ratings.csv"
    with ratings.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("item_id", "answer"))
        writer.writerows((item_id, answer) for item_id, answer in answers.items())
    parsed = stroke._parse_ratings(ratings, mapping)
    result = stroke._score_review(mapping, parsed, readouts, _synthetic_endpoint_summaries())
    assert result["review_quality"]["passed"] is True
    assert result["human_primary_by_endpoint"]["B"]["human_correct_total"] == 200
    assert result["decision"]["classification"] == "STROKE_FIDELITY_GAIN_SUPPORTED"

    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text("item_id,answer\n" + "\n".join(f"{key},{value}" for key, value in list(answers.items())[:-1]), encoding="utf-8")
    with pytest.raises(stroke.StrokeFidelityError, match="do not cover all 680"):
        stroke._parse_ratings(incomplete, mapping)


def test_review_mapping_rejects_hidden_repeat_with_wrong_endpoint_or_source(tmp_path) -> None:
    inputs, readouts = _synthetic_inputs_and_readouts()
    stroke._prepare_review(tmp_path, inputs, readouts)
    mapping = json.loads((tmp_path / "review_mapping.json").read_text(encoding="utf-8"))["items"]
    duplicate = next(row for row in mapping if row["kind"] == "hidden_repeat")
    duplicate["repeat_of"] = "missing-item"
    with pytest.raises(stroke.StrokeFidelityError, match="hidden repeat"):
        stroke._validate_review_mapping(mapping)


def test_finalize_ratings_consumes_produced_start_filename_and_report_summaries(tmp_path) -> None:
    inputs, readouts = _synthetic_inputs_and_readouts()
    stroke._prepare_review(tmp_path, inputs, readouts)
    (tmp_path / "review_mapping.json").write_text(
        json.dumps({"task_id": stroke.TASK_ID, "items": json.loads((tmp_path / "review_mapping.json").read_text(encoding="utf-8"))["items"]}),
        encoding="utf-8",
    )
    for name, filename in (("A4700", "readouts_start.pt"), ("A5400", "readouts_A5400.pt"), ("B5400", "readouts_B5400.pt")):
        torch.save(readouts[name], tmp_path / filename)
    assert not (tmp_path / "readouts_A4700.pt").exists()
    summaries = _synthetic_endpoint_summaries()
    (tmp_path / "report.json").write_text(
        json.dumps({"task_id": stroke.TASK_ID, "status": "AWAITING_RATINGS", "endpoint_metrics": summaries}),
        encoding="utf-8",
    )

    mapping = json.loads((tmp_path / "review_mapping.json").read_text(encoding="utf-8"))["items"]
    answers = {row["item_id"]: str(row["true_class"]) for row in mapping}
    ratings = tmp_path / "ratings.csv"
    with ratings.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("item_id", "answer"))
        writer.writerows(answers.items())

    result = stroke._finalize_ratings(tmp_path, ratings)
    assert result["status"] == "VALID_FAILED"
    assert result["human_review_result"]["review_quality"]["passed"] is True
    assert (tmp_path / "report_awaiting_ratings.json").is_file()
    saved = json.loads((tmp_path / "ratings_result.json").read_text(encoding="utf-8"))
    assert saved["answers"] == answers
    assert stroke._finalize_ratings(tmp_path, ratings) == result

    changed = dict(answers)
    changed[mapping[0]["item_id"]] = "unclear" if changed[mapping[0]["item_id"]] != "unclear" else "0"
    alternate = tmp_path / "alternate.csv"
    with alternate.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("item_id", "answer"))
        writer.writerows(changed.items())
    with pytest.raises(stroke.StrokeFidelityError, match="different answers"):
        stroke._finalize_ratings(tmp_path, alternate)
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == result
