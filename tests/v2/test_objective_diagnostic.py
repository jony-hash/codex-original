from __future__ import annotations

import torch
import pytest

from ldm_project.v2.application.cli_schema import _parser
from ldm_project.v2.application.diagnostic.objective import _code_seed, _companion_indices, _effective_gradient_gate, _gradient_cosine, _scheduled_batches, _warmup_lineage_gate, OBJECTIVE_DIAGNOSTIC_CONTRACT_SPEC
from ldm_project.v2.config import ConditionalGenerationConfig
from ldm_project.v2.student.conditional import active_objective_kind, deterministic_companion_indices, objective_contract_payload, relative_diversity_worst_class_loss
from ldm_project.v2.evidence import canonical_sha256
from ldm_project.v2.student.conditional import coverage_ratio_upper_loss


def test_companion_schedule_is_deterministic_and_disjoint() -> None:
    labels = torch.arange(10).repeat_interleave(20)
    primary = torch.cat([torch.nonzero(labels == label, as_tuple=False).flatten()[:5] for label in range(10)])
    first = _companion_indices(labels, primary, seed=9012)
    second = _companion_indices(labels, primary, seed=9012)
    assert torch.equal(first, second)
    assert not torch.isin(first, primary).any()
    assert torch.bincount(labels.index_select(0, first), minlength=10).tolist() == [5] * 10


def test_registered_confirmation_seeds_reproduce_formal_code_schedule() -> None:
    assert [_code_seed(seed, 0) for seed in (1234, 1235, 1236)] == [100000, 200000, 300000]
    assert _code_seed(1234, 99) == 100099


def test_segmented_confirmation_schedule_is_contiguous_and_reproducible() -> None:
    labels = torch.arange(10).repeat_interleave(2500)
    whole = _scheduled_batches(labels, batch_size=50, seed=1234, start_step=0, steps=400)
    segmented = _scheduled_batches(labels, batch_size=50, seed=1234, start_step=0, steps=100)
    segmented += _scheduled_batches(labels, batch_size=50, seed=1234, start_step=100, steps=300)
    assert canonical_sha256(whole) == canonical_sha256(segmented)
    assert all(torch.equal(left, right) for left, right in zip(whole, segmented, strict=True))


def test_confirmation_primary_schedule_is_shared_by_candidates() -> None:
    labels = torch.arange(10).repeat_interleave(2500)
    a = _scheduled_batches(labels, batch_size=50, seed=1235, start_step=0, steps=400)
    winner = _scheduled_batches(labels, batch_size=50, seed=1235, start_step=0, steps=100)
    winner += _scheduled_batches(labels, batch_size=50, seed=1235, start_step=100, steps=300)
    assert canonical_sha256(a) == canonical_sha256(winner)


def test_coverage_companion_contract_uses_matching_primary_counts() -> None:
    torch.manual_seed(7)
    labels = torch.arange(10).repeat_interleave(5)
    generated_features = torch.randn(50, 8)
    real_features = torch.randn(50, 8)
    generated_view = torch.rand(50, 1, 8, 8)
    real_view = torch.rand(50, 1, 8, 8)
    companion_features = torch.randn(50, 8)
    companion_view = torch.rand(50, 1, 8, 8)
    value = coverage_ratio_upper_loss(
        generated_features,
        generated_view,
        real_features,
        real_view,
        labels,
        epsilon=0.05,
        iterations=4,
        companion_features=companion_features,
        companion_view=companion_view,
        companion_labels=labels,
    )
    assert torch.isfinite(value)


def test_gradient_cosine_reports_alignment_and_zero_as_unavailable() -> None:
    assert _gradient_cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == 0.0
    assert _gradient_cosine(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == 1.0
    assert _gradient_cosine(torch.zeros(2), torch.ones(2)) is None


def test_effective_gradient_gate_is_invariant_to_raw_loss_units() -> None:
    statuses = {
        name: {
            group: "inactive_neutral"
            for group in ("class_carrier", "global_modes", "code_projection", "class_heads", "encoder_phase", "decoder_phase")
        }
        for name in ("camera", "pairwise", "coverage")
    }
    diagnostics = {
        "batches": [
            {"weighted_gradient_ratios": {"camera": 0.25, "pairwise": 0.25, "coverage": 0.25}, "combined_gradient_ratios": {"B": 0.40, "C": 0.55}, "gradient_group_status": statuses}
            for _ in range(4)
        ],
        "effective_gradients": {
            "weighted_gradient_ratios": {"camera": 0.25, "pairwise": 0.25, "coverage": 0.25},
            "combined_gradient_ratios": {"B": 0.40, "C": 0.55},
        },
    }
    passed, report = _effective_gradient_gate(diagnostics)
    assert passed and report["aggregate_safe"]
    diagnostics["effective_gradients"]["weighted_gradient_ratios"]["camera"] = 0.60
    assert not _effective_gradient_gate(diagnostics)[0]


def test_effective_gradient_gate_rejects_spikes_and_missing_components() -> None:
    diagnostics = {
        "batches": [
            {"weighted_gradient_ratios": {"camera": 0.25, "pairwise": 0.25, "coverage": 0.25}, "combined_gradient_ratios": {"B": 0.40, "C": 0.55}},
            {"weighted_gradient_ratios": {"camera": 0.51, "pairwise": 0.25, "coverage": 0.25}, "combined_gradient_ratios": {"B": 0.40, "C": 0.55}},
        ],
        "effective_gradients": {
            "weighted_gradient_ratios": {"camera": 0.25, "pairwise": 0.25, "coverage": 0.25},
            "combined_gradient_ratios": {"B": 0.40, "C": 0.55},
        },
    }
    with pytest.raises(KeyError):
        _effective_gradient_gate(diagnostics)
    with pytest.raises(KeyError):
        _effective_gradient_gate({})


def test_effective_gradient_gate_uses_auxiliary_delta_not_base_plus_delta() -> None:
    statuses = {
        name: {
            group: "inactive_neutral"
            for group in ("class_carrier", "global_modes", "code_projection", "class_heads", "encoder_phase", "decoder_phase")
        }
        for name in ("camera", "pairwise", "coverage")
    }
    diagnostics = {
        "batches": [
            {"weighted_gradient_ratios": {"camera": 0.25, "pairwise": 0.25, "coverage": 0.25}, "combined_gradient_ratios": {"B": 0.35, "C": 0.45}, "gradient_group_status": statuses}
            for _ in range(4)
        ],
        "effective_gradients": {
            "weighted_gradient_ratios": {"camera": 0.25, "pairwise": 0.25, "coverage": 0.25},
            "combined_gradient_ratios": {"B": 0.35, "C": 0.45},
        },
    }
    assert _effective_gradient_gate(diagnostics)[0]


def test_objective_commands_are_content_bound() -> None:
    parser = _parser()
    screen = parser.parse_args(
        [
            "student", "objective-alignment-screen", "--config", "c", "--dataset-root", "d",
            "--evaluator-checkpoint", "p", "--acceptance-evaluator-checkpoint", "s",
            "--acceptance-measurement-bundle", "b", "--acceptance-measurement-evidence", "m",
            "--initialization-checkpoint", "i", "--injection-evidence", "j", "--weight-calibration", "w", "--weight-calibration-evidence", "we",
        ]
    )
    assert screen.weight_calibration == "w"
    assert screen.weight_calibration_evidence == "we"
    assert not hasattr(screen, "baseline_measurement")
    confirm = parser.parse_args(
        [
            "student", "objective-alignment-confirm", "--config", "c", "--dataset-root", "d",
            "--evaluator-checkpoint", "p", "--acceptance-evaluator-checkpoint", "s",
            "--acceptance-measurement-bundle", "b", "--acceptance-measurement-evidence", "m",
            "--initialization-checkpoint", "i", "--injection-evidence", "j", "--screen-evidence", "e",
        ]
    )
    assert confirm.screen_evidence == "e"
    assert not hasattr(confirm, "candidate")


def test_late_objective_contract_and_schema_are_registered() -> None:
    config = ConditionalGenerationConfig(
        objective_kind="D",
        objective_activation_step=400,
        camera_code_safety_weight=1.0,
        pairwise_diversity_weight=1.0,
        coverage_ratio_weight=1.0,
    )
    assert config.objective_kind == "D"
    assert config.objective_activation_step == 400
    assert OBJECTIVE_DIAGNOSTIC_CONTRACT_SPEC["activation_step"] == 400


@pytest.mark.parametrize(
    "kwargs",
    [
        {"objective_activation_step": 399},
        {"camera_code_safety_weight": 0.0},
        {"pairwise_diversity_weight": 0.0},
        {"coverage_ratio_weight": 0.0},
    ],
)
def test_late_objective_schema_rejects_unregistered_contract(kwargs: dict[str, object]) -> None:
    values = {
        "objective_kind": "D",
        "objective_activation_step": 400,
        "camera_code_safety_weight": 1.0,
        "pairwise_diversity_weight": 1.0,
        "coverage_ratio_weight": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        ConditionalGenerationConfig(**values)


def test_late_objective_worst_class_penalty_is_finite_and_differentiable() -> None:
    torch.manual_seed(9)
    features = torch.randn(50, 8, requires_grad=True)
    real = torch.randn(50, 8)
    labels = torch.arange(10).repeat_interleave(5)
    value = relative_diversity_worst_class_loss(features, real, labels)
    value.backward()
    assert torch.isfinite(value)
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_late_objective_activation_boundary_and_contract_binding() -> None:
    assert active_objective_kind("D", 400, 399) == "A"
    assert active_objective_kind("D", 400, 400) == "D"
    base = objective_contract_payload("D", 400, 1.0, 2.0, 3.0)
    changed = objective_contract_payload("D", 401, 1.0, 2.0, 3.0)
    assert canonical_sha256(base) != canonical_sha256(changed)


def test_formal_and_diagnostic_companion_schedule_share_contract() -> None:
    labels = torch.arange(10).repeat_interleave(20)
    primary = torch.cat([torch.nonzero(labels == label, as_tuple=False).flatten()[:5] for label in range(10)])
    formal = deterministic_companion_indices(labels, primary, seed=1234 * 100000 + 400)
    diagnostic = _companion_indices(labels, primary, seed=1234 * 100000 + 400)
    assert torch.equal(formal, diagnostic)
    assert not torch.isin(formal, primary).any()


def test_warmup_lineage_gate_compares_declared_hash_fields() -> None:
    hashes = {name: f"{name}-hash" for name in ("state_hash", "optimizer_hash", "primary_schedule_hash", "companion_schedule_hash")}
    milestone = {"state_hash": hashes["state_hash"], "optimizer_hash": hashes["optimizer_hash"]}
    report = {
        "A": {"update_status": {"milestones": {"400": milestone}}, "warmup_primary_schedule_hash": hashes["primary_schedule_hash"], "warmup_companion_schedule_hash": hashes["companion_schedule_hash"]},
        "D": {"update_status": {"milestones": {"400": dict(milestone)}}, "warmup_primary_schedule_hash": hashes["primary_schedule_hash"], "warmup_companion_schedule_hash": hashes["companion_schedule_hash"]},
    }
    calibration = {"diagnostics": {"warmup": {"state_hash": hashes["state_hash"], "optimizer_hash": hashes["optimizer_hash"], "schedule_hash": hashes["primary_schedule_hash"], "companion_schedule_hash": hashes["companion_schedule_hash"]}}}
    assert _warmup_lineage_gate(report, calibration)[0]
    report["D"]["update_status"]["milestones"]["400"]["state_hash"] = "tampered"
    assert not _warmup_lineage_gate(report, calibration)[0]


@pytest.mark.parametrize("field", ["optimizer_hash", "warmup_primary_schedule_hash", "warmup_companion_schedule_hash"])
def test_warmup_lineage_gate_rejects_each_schedule_or_optimizer_tamper(field: str) -> None:
    hashes = {name: f"{name}-hash" for name in ("state_hash", "optimizer_hash", "primary_schedule_hash", "companion_schedule_hash")}
    milestone = {"state_hash": hashes["state_hash"], "optimizer_hash": hashes["optimizer_hash"]}
    report = {
        "A": {"update_status": {"milestones": {"400": dict(milestone)}}, "warmup_primary_schedule_hash": hashes["primary_schedule_hash"], "warmup_companion_schedule_hash": hashes["companion_schedule_hash"]},
        "D": {"update_status": {"milestones": {"400": dict(milestone)}}, "warmup_primary_schedule_hash": hashes["primary_schedule_hash"], "warmup_companion_schedule_hash": hashes["companion_schedule_hash"]},
    }
    calibration = {"diagnostics": {"warmup": {"state_hash": hashes["state_hash"], "optimizer_hash": hashes["optimizer_hash"], "schedule_hash": hashes["primary_schedule_hash"], "companion_schedule_hash": hashes["companion_schedule_hash"]}}}
    if field in {"state_hash", "optimizer_hash"}:
        report["D"]["update_status"]["milestones"]["400"][field] = "tampered"
    else:
        report["D"][field] = "tampered"
    assert not _warmup_lineage_gate(report, calibration)[0]
