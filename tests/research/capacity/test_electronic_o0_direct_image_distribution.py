from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.capacity import electronic_o0_direct_image_distribution as didm


def _images(seed: int = 3, per_class: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, labels = [], []
    for label in range(10):
        base = torch.zeros(1, 28, 28)
        base[:, label : label + 3, label : label + 3] = 0.8
        rows.append((base + 0.02 * torch.randn((per_class, 1, 28, 28), generator=generator)).clamp(0, 1))
        labels.append(torch.full((per_class,), label, dtype=torch.long))
    return torch.cat(rows), torch.cat(labels)


def _naive(projected: torch.Tensor) -> torch.Tensor:
    n = projected.shape[0]
    values = torch.abs(projected[:, None, :] - projected[None, :, :])
    return values.sum(dim=(0, 1)) / (n * (n - 1))


def test_config_and_manifest_cli_contract() -> None:
    root = Path(__file__).parents[3]
    config = didm._config(root / "configs/research/capacity/electronic_o0_direct_image_distribution_v1.yaml")
    assert config["direct_image_distribution_seed"] == didm.HARNESS_SEED
    assert config["views"] == list(didm.VIEWS)
    for action in ("validate", "audit", "preflight", "screen", "report", "closeout"):
        args = _parser().parse_args([
            "research", "electronic-o0-direct-image-distribution", action,
            "--run-manifest", "manifest.yaml",
        ])
        assert args.research_action == action


def test_ordered_u_statistic_matches_naive_and_gradients() -> None:
    values = torch.randn((6, 4), generator=torch.Generator().manual_seed(9), requires_grad=True)
    fast = didm._ordered_pairwise_absolute_mean(values)
    slow = _naive(values)
    assert torch.allclose(fast, slow, atol=1e-6, rtol=1e-6)
    fast.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_two_sample_energy_matches_naive_formula_and_allows_negative_estimates() -> None:
    x = torch.zeros((3, 2), requires_grad=True)
    y = torch.ones((3, 2), requires_grad=True)
    projections = torch.eye(2)
    result = didm.two_sample_sliced_energy(x, y, projections)
    cross = (x[:, None, :] - y[None, :, :]).abs().mean((0, 1))
    expected = (2 * cross - _naive(x) - _naive(y)).mean()
    assert torch.allclose(result, expected)
    result.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
    negative = didm.two_sample_sliced_energy(torch.zeros((3, 2)), torch.zeros((3, 2)), projections)
    assert float(negative) == pytest.approx(0.0)


def test_formula_chunk_and_projection_controls_are_deterministic_on_cpu() -> None:
    controls = didm._formula_and_chunk_controls(device=torch.device("cpu"))
    assert all(controls.values())
    images, labels = _images(per_class=20)
    banks = {
        seed: didm.multiscale_projection_banks(seed, device=torch.device("cpu"), dtype=torch.float32)
        for seed in didm.PROJECTION_SEEDS
    }
    result = didm._control_measurement(images, labels, banks)
    assert result["passed"]
    assert all(row["passed"] for row in result["banks"])


def test_registered_schedule_has_625_steps() -> None:
    _, _, schedule = didm._training_data("data", seed=didm.SCREEN_SEED)
    assert len(schedule) == didm.EXPECTED_STEPS == 625


def test_views_and_projection_banks_are_deterministic_and_chunk_independent() -> None:
    images, _ = _images()
    first = didm.multiscale_views(images)
    second = didm.multiscale_views(images)
    assert all(torch.equal(first[name], second[name]) for name in didm.VIEWS)
    banks_a = didm.multiscale_projection_banks(271829, device=torch.device("cpu"), dtype=torch.float32)
    banks_b = didm.multiscale_projection_banks(271829, device=torch.device("cpu"), dtype=torch.float32)
    assert all(torch.equal(banks_a[name], banks_b[name]) for name in didm.VIEWS)
    assert all(torch.allclose(banks_a[name].norm(dim=1), torch.ones(256)) for name in didm.VIEWS)


def test_classwise_distribution_loss_has_finite_decoder_gradients() -> None:
    images, labels = _images()
    decoder = didm.ConditionalWAEDecoder("O0-I")
    codes = torch.randn((labels.numel(), 8), generator=torch.Generator().manual_seed(12))
    generated = decoder(codes, labels)
    banks = didm.multiscale_projection_banks(271829, device=torch.device("cpu"), dtype=torch.float32)
    loss = didm.classwise_multiscale_energy(generated, images, labels, banks)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in decoder.parameters())


def test_invalid_shapes_fail_fast() -> None:
    with pytest.raises(ValueError):
        didm.multiscale_views(torch.zeros((4, 28, 28)))
    with pytest.raises(ValueError):
        didm.two_sample_sliced_energy(torch.zeros((1, 2)), torch.zeros((2, 2)), torch.eye(2))


def test_audit_classification_is_fail_closed() -> None:
    valid = {
        "all_gradients_finite": True,
        "all_groups_nonzero": True,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "direction_pass_fraction": 0.75,
        "reverse_negative_control_pass_fraction": 0.0,
        "controls": {"passed": True},
        "formula_and_chunk_controls": {
            "value_matches_naive": True,
            "gradient_matches_naive": True,
            "chunk_matches_full": True,
            "finite": True,
        },
        "projection_determinism": {"training_repeat": True, "reference_repeat": True},
    }
    assert didm._audit_classification(valid)[0] == "direct_image_distribution_direction_admitted"
    valid["direction_pass_fraction"] = 0.5
    assert didm._audit_classification(valid)[0] == "direct_image_distribution_direction_not_admitted"
    invalid = {"all_gradients_finite": False}
    assert didm._audit_classification(invalid)[0] == "invalid_harness"


def test_screen_classification_is_fail_closed() -> None:
    assert didm._screen_classification(True, True) == (
        "direct_image_distribution_admitted", "register_direct_image_distribution_confirm"
    )
    assert didm._screen_classification(True, False) == (
        "direct_image_distribution_failed", "close_current_direct_image_distribution_route"
    )
    assert didm._screen_classification(False, False)[0] == "invalid_harness"


def test_no_evaluator_or_legacy_dispatcher_import() -> None:
    source = Path(didm.__file__).read_text(encoding="utf-8")
    assert "load_frozen_evaluator" not in source
    assert "electronic_o0_prior_native_swae" not in source
