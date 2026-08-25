from __future__ import annotations

import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.electronic_distribution_recovery import (
    CODE_DIM,
    _chi_square_radius,
    _config,
    _deterministic_adaptive_pool,
    _effective_rank,
    _geometry_fixture,
    _partition,
    _project_codes,
    _semantic_gate,
)


def test_package_a_cli_registers_all_fail_closed_stages() -> None:
    parser = _parser()
    for action in ("validate", "gate-audit", "preflight", "audit", "report", "closeout"):
        argv = ["research", "electronic-distribution-recovery", action]
        if action in {"validate", "gate-audit", "preflight", "audit"}:
            argv.extend(
                [
                    "--config", "config.yaml",
                    "--dataset-root", "data",
                    "--generator-checkpoint", "generator.pt",
                    "--primary-evaluator-checkpoint", "primary.pt",
                    "--secondary-evaluator-checkpoint", "secondary.pt",
                    "--primary-qualification-evidence", "primary.json",
                    "--secondary-qualification-evidence", "secondary.json",
                    "--output", "output.json",
                ]
            )
        else:
            argv.extend(["--config", "config.yaml", "--output", "output.json"])
        if action == "gate-audit":
            argv.extend(["--validation-evidence", "validation.json"])
        if action == "preflight":
            argv.extend(["--validation-evidence", "validation.json", "--gate-audit-evidence", "gate.json"])
        if action == "audit":
            argv.extend(["--gate-audit-evidence", "gate.json", "--preflight-evidence", "preflight.json"])
        if action == "report":
            argv.extend(["--audit-evidence", "audit.json"])
        if action == "closeout":
            argv.extend(["--report-evidence", "report.json"])
        parsed = parser.parse_args(argv)
        assert parsed.research_program == "electronic-distribution-recovery"
        assert parsed.research_action == action


def test_package_a_config_is_registered() -> None:
    config = _config("configs/research/electronic_distribution_recovery_v1.yaml")
    assert config["chi_square_probability"] == 0.99
    assert config["bank_samples_per_class"] == 250
    assert config["formal_promotion"] is False
    assert config["optical_training_authorized"] is False


def test_package_a_partition_is_strictly_mutually_exclusive() -> None:
    images = torch.arange(5000, dtype=torch.float32).reshape(5000, 1, 1, 1)
    labels = torch.arange(10).repeat_interleave(500)
    first, second = _partition(images, labels, seed=32452843, samples_per_class=250)
    first_values = set(first[0].flatten().tolist())
    second_values = set(second[0].flatten().tolist())
    assert first[0].shape == second[0].shape == (2500, 1, 1, 1)
    assert first_values.isdisjoint(second_values)
    assert first[1].tolist() == second[1].tolist()


def test_package_a_chi_square_projection_respects_registered_radius() -> None:
    radius = _chi_square_radius(0.99)
    codes = torch.full((32, CODE_DIM), 10.0)
    projected = _project_codes(codes, radius)
    assert torch.all(projected.square().sum(dim=1).sqrt() <= radius + 1.0e-5)
    assert radius > 0.0


def test_package_a_deterministic_pool_matches_formal_adaptive_average_pool() -> None:
    value = torch.rand(2, 1, 200, 200)
    expected = torch.nn.functional.adaptive_avg_pool2d(value, (28, 28))
    actual = _deterministic_adaptive_pool(value)
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_package_a_jacobian_fixture_has_known_spectrum_and_zero_negative_control() -> None:
    fixture = _geometry_fixture()
    assert fixture["known_spectrum"] == [4.0, 3.0, 2.0, 1.0]
    assert fixture["known_effective_rank"]["nominal_rank"] == 4
    assert fixture["constant_output_nominal_rank"] == 0


def test_package_a_semantic_gate_rejects_label_shift_negative_control() -> None:
    metrics = {
        "agreement": 1.0,
        "primary": {"accuracy": 0.10, "minimum_recall": 0.0, "diversity": [1.0] * 10, "coverage": 1.0},
        "secondary": {"accuracy": 0.10, "minimum_recall": 0.0, "diversity": [1.0] * 10, "coverage": 1.0},
    }
    config = _config("configs/research/electronic_distribution_recovery_v1.yaml")
    assert not _semantic_gate(metrics, config)


def test_package_a_effective_rank_uses_nominal_singular_values() -> None:
    result = _effective_rank(torch.tensor([4.0, 2.0, 0.01]), 1.0e-3)
    assert result["nominal_rank"] == 3
    assert result["effective_rank"] > 1.0
