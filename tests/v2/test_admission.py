from __future__ import annotations

import pytest
import torch

from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, file_sha256, load_manual_review, require_evidence_action, require_passed_evidence, tensor_sha256
from ldm_project.v2.student.admission import CapacityMetrics, PrototypeMetrics, balanced_code_fraction, capacity_gate, prototype_gate, prototype_measurement_gate
from ldm_project.v2.student.capacity import build_code_basis, capacity_metrics, supported_dct_candidates
from ldm_project.v2.student.capacity import finite_difference_response_matrices


def _prototype_pass() -> GateResult:
    return prototype_gate(PrototypeMetrics(0.8, 0.8, 1.0, 0.01, True, True, True, True), list(range(10)))


def test_prototype_gate_requires_every_manual_label() -> None:
    assert _prototype_pass().passed
    failed = prototype_gate(PrototypeMetrics(0.8, 0.8, 1.0, 0.01, True, True, True, True), list(range(9)))
    assert not failed.passed


def test_prototype_measurement_gate_stops_before_manual_review() -> None:
    failed = prototype_measurement_gate(PrototypeMetrics(0.69, 0.8, 1.0, 0.01, True, True, True, True))
    assert not failed.passed
    assert "manual_readability" not in failed.metrics


def test_capacity_gate_blocks_when_code_is_not_transportable() -> None:
    gate = capacity_gate(CapacityMetrics(0.25, 0.24, 0.95, 0.08, True, False, True), _prototype_pass())
    assert not gate.passed
    assert gate.metrics["raw_camera_code_fraction_value"] == 0.24
    passed = capacity_gate(CapacityMetrics(0.25, 0.25, 0.95, 0.08, True, False, True), _prototype_pass())
    assert passed.passed


def test_dct_svd_basis_and_capacity_metrics_are_label_fixed() -> None:
    support = torch.ones(10, 8, 8)
    candidates = supported_dct_candidates(support[0], candidate_count=8, pixel_pitch_m=(62.5e-6, 62.5e-6), max_cycles_per_mm=20.0)
    assert candidates.directions.shape == (8, 8, 8)
    assert torch.all(candidates.frequencies_cycles_per_mm <= 20.0)
    response = torch.randn(10, 8, 16)
    artifact = build_code_basis(response, support, candidate_count=8, seed=1234, source_hash="source", pixel_pitch_m=(62.5e-6, 62.5e-6), max_cycles_per_mm=20.0)
    assert artifact.basis.shape == (10, 8, 8, 8)
    assert artifact.normalizer.shape == (10, 1, 1)
    commands = torch.rand(10, 4, 8, 8)
    camera = commands.square()
    values = capacity_metrics(commands, commands.square(), camera, support)
    assert all(0.0 <= value <= 1.0 for value in values)
    assert balanced_code_fraction(commands) >= 0.0


def test_physical_dct_cutoff_rejects_an_underresolved_fov() -> None:
    with pytest.raises(ValueError, match="physical"):
        supported_dct_candidates(torch.ones(8, 8), candidate_count=8, pixel_pitch_m=(2.5e-6, 2.5e-6), max_cycles_per_mm=20.0)


def test_evidence_state_machine_blocks_diagnostic_and_dirty_acceptance() -> None:
    gate = GateResult("unit", True, {})
    context = EvidenceContextV2("system", "config")
    diagnostic = EvidenceRecordV2(2, "capacity", "diagnostic", "diagnostic", "commit", "patch", "config", context, {}, (gate,), ())
    with pytest.raises(ValueError, match="passed"):
        require_passed_evidence(diagnostic, kind="capacity")
    dirty = EvidenceRecordV2(2, "capacity", "acceptance", "passed", "commit", "patch", "config", context, {}, (gate,), ("train",))
    with pytest.raises(ValueError, match="clean"):
        dirty.validate()


def test_evidence_action_requires_exact_context_and_named_diagnostic_permission() -> None:
    context = EvidenceContextV2("system", "config", dataset_hash="dataset")
    record = EvidenceRecordV2(2, "prototype", "diagnostic", "diagnostic", "commit", "patch", "config", context, {}, (GateResult("unit", False, {}),), ("diagnostic_capacity_smoke",))
    require_evidence_action(record, kind="prototype", action="diagnostic_capacity_smoke", expected_context=EvidenceContextV2("system", "config", dataset_hash="dataset"), allow_diagnostic=True)
    with pytest.raises(ValueError, match="context"):
        require_evidence_action(record, kind="prototype", action="diagnostic_capacity_smoke", expected_context=EvidenceContextV2("system", "config", dataset_hash="other"), allow_diagnostic=True)


def test_finite_difference_responses_are_frozen_and_finite() -> None:
    class System(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def command(self, value: torch.Tensor) -> torch.Tensor:
            return value

        def forward(self, value: torch.Tensor):
            return type("Result", (), {"raw_camera": type("Raw", (), {"tensor": value * self.scale})()})()

    responses, unchanged, before_hash, after_hash = finite_difference_response_matrices(System(), torch.full((10, 1, 4, 4), 0.5), torch.ones(10, 4, 4), candidate_count=8, epsilon=1.0e-3, pixel_pitch_m=(125e-6, 125e-6), max_cycles_per_mm=20.0)
    assert responses.shape == (10, 8, 16)
    assert torch.isfinite(responses).all() and unchanged and before_hash == after_hash


def test_review_requires_the_exact_fixed_exposure_preview(tmp_path) -> None:
    raw = torch.zeros(10, 1, 2, 2)
    preview = tmp_path / "preview.png"
    preview.write_bytes(b"fixed-exposure")
    review = tmp_path / "review.json"
    review.write_text('{"schema_version":2,"review_kind":"prototype","reviewed_tensor_sha256":"%s","reviewed_preview_sha256":"%s","readable_labels":[0,1,2,3,4,5,6,7,8,9],"reviewer":"unit"}' % (tensor_sha256(raw), file_sha256(preview)), encoding="utf-8")
    load_manual_review(review, kind="prototype", tensor_hash=tensor_sha256(raw), preview_hash=file_sha256(preview))
    preview.write_bytes(b"different")
    with pytest.raises(ValueError, match="preview"):
        load_manual_review(review, kind="prototype", tensor_hash=tensor_sha256(raw), preview_hash=file_sha256(preview))


def test_review_can_record_a_failed_readability_subset(tmp_path) -> None:
    raw = torch.zeros(10, 1, 2, 2)
    preview = tmp_path / "preview.png"
    preview.write_bytes(b"fixed-exposure")
    review = tmp_path / "review.json"
    review.write_text('{"schema_version":2,"review_kind":"prototype","reviewed_tensor_sha256":"%s","reviewed_preview_sha256":"%s","readable_labels":[0,2],"reviewer":"unit"}' % (tensor_sha256(raw), file_sha256(preview)), encoding="utf-8")
    parsed, _ = load_manual_review(review, kind="prototype", tensor_hash=tensor_sha256(raw), preview_hash=file_sha256(preview))
    assert not prototype_gate(PrototypeMetrics(0.8, 0.8, 1.0, 0.01, True, True, True, True), parsed.readable_labels).passed
