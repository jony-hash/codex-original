from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_back_relay_axial_conjugate as audit
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_back_relay_axial_conjugate_v1.yaml"
REVIEW = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_back_relay_axial_conjugate_review_v1.json"


def test_contract_and_review_are_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec).name.endswith("back_relay_axial_conjugate_review_v1.json")
    assert spec.payload["hypothesis"] == audit.HYPOTHESIS


def test_contract_rejects_intervention_or_parent_mutation() -> None:
    spec = audit._load_spec(CONFIG)
    with pytest.raises(ValueError, match="intervention"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "intervention": {}}, spec.config_hash))
    with pytest.raises(ValueError, match="DCM parent"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "parent_dcm": {}}, spec.config_hash))


def test_nominal_and_axial_abcd_contract() -> None:
    nominal = audit._axial_abcd(0.0)
    assert nominal.a == pytest.approx(-2.5, abs=1.0e-12)
    assert nominal.b_m == pytest.approx(0.0, abs=1.0e-9)
    assert nominal.c_per_m == pytest.approx(-1633.3333333333333, rel=1.0e-12)
    assert nominal.d == pytest.approx(-0.4, abs=1.0e-12)
    assert nominal.determinant == pytest.approx(1.0, abs=1.0e-12)
    negative = audit._axial_abcd(-audit.DELTA_Z_M)
    positive = audit._axial_abcd(audit.DELTA_Z_M)
    assert negative.b_m > 0.0
    assert positive.b_m < 0.0
    assert positive.a - nominal.a == pytest.approx(audit.DELTA_Z_M * nominal.c_per_m, rel=1.0e-12)
    assert positive.b_m - nominal.b_m == pytest.approx(audit.DELTA_Z_M * nominal.d, rel=1.0e-12)
    for candidate in (negative, positive):
        assert candidate.determinant == pytest.approx(1.0, abs=1.0e-12)
        assert abs(candidate.a / audit.BACK_MAGNIFICATION - 1.0) < audit.MAGNIFICATION_DRIFT_TOLERANCE


def test_signed_asm_zero_inverse_and_frequency_gate() -> None:
    plane = PlaneSpec(
        "chen_camera",
        audit.TARGET_SHAPE,
        audit.TARGET_PIXEL_PITCH_M,
        audit.WAVELENGTH_M,
        0.0,
        "optical_axis_xy",
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    field_tensor = torch.polar(
        torch.rand((2, 1, *audit.TARGET_SHAPE), generator=generator) + 0.2,
        torch.rand((2, 1, *audit.TARGET_SHAPE), generator=generator) * 6.0 - 3.0,
    )
    field = ComplexField(field_tensor, plane, OperationProvenance("test_input", None, None))
    zero = audit._signed_asm(field, 0.0, "chen_camera_baseline")
    assert torch.equal(zero.tensor, field.tensor)
    positive = audit._signed_asm(field, audit.DELTA_Z_M, "chen_camera_positive")
    round_trip = audit._signed_asm(positive, -audit.DELTA_Z_M, "chen_camera_roundtrip")
    assert float((positive.tensor - field.tensor).abs().max()) > 0.0
    assert float((round_trip.tensor - field.tensor).abs().max()) < 1.0e-4
    gate = audit._frequency_gate(plane, torch.device("cpu"))
    assert gate["all_propagating"] is True
    assert gate["minimum_propagating_argument"] > 0.0


def test_candidate_panel_is_balanced_and_disjoint() -> None:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(500)
    labels = torch.cat((labels, labels))
    data = SimpleNamespace(
        test_labels=labels,
        selection_indices=torch.arange(5000, 6000, dtype=torch.long),
    )
    panel = audit._candidate_panel(data)
    assert panel.indices.numel() == 4560
    assert torch.equal(torch.bincount(panel.labels, minlength=10), torch.full((10,), 456, dtype=torch.long))
    assert int(panel.indices.max()) < 5000
    assert not bool(torch.isin(panel.indices, data.selection_indices).any())
    assert panel.mechanism_evidence_only is True


def test_threshold_and_tie_selection_are_deterministic() -> None:
    def evaluation(arm: str, retention: float, mse: float) -> SimpleNamespace:
        return SimpleNamespace(
            arm=arm,
            delta_z_m=audit.ARMS[arm],
            metrics={
                "retention": {
                    audit.RELAY_SEGMENT: {"geometric_mean_retention": retention},
                    audit.DETECTOR_SEGMENT: {"geometric_mean_retention": 0.95},
                },
                "reconstruction": {"paired_mse": mse},
                "power_retention": 1.0,
            },
        )

    baseline = evaluation("baseline", 0.60, 1.0)
    candidate = evaluation("negative", 0.65, 0.94)
    passed, details = audit._selection_gate(candidate, baseline)
    assert passed is True
    assert details["primary_retention_improvement"] == pytest.approx(0.05)
    not_passed, _ = audit._selection_gate(evaluation("negative", 0.649, 0.94), baseline)
    assert not_passed is False


def test_cli_actions_are_registered_and_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args(["research", "optical-manifold", "scaled-chen-back-relay-axial-conjugate-validate", "--config", "config.yaml", "--review-contract", "review.json", "--dataset-root", "data", "--output", "validation.json"])
    preflight = parser.parse_args(["research", "optical-manifold", "scaled-chen-back-relay-axial-conjugate-preflight", "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json", "--dataset-root", "data", "--output", "preflight.json"])
    run = parser.parse_args(["research", "optical-manifold", "scaled-chen-back-relay-axial-conjugate-run", "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json", "--preflight-evidence", "preflight.json", "--dataset-root", "data", "--output", "audit.json"])
    report = parser.parse_args(["research", "optical-manifold", "scaled-chen-back-relay-axial-conjugate-report", "--config", "config.yaml", "--review-contract", "review.json", "--audit-evidence", "audit.json", "--dataset-root", "data", "--output", "report.json"])
    closeout = parser.parse_args(["research", "optical-manifold", "scaled-chen-back-relay-axial-conjugate-closeout", "--config", "config.yaml", "--review-contract", "review.json", "--report-evidence", "report.json", "--dataset-root", "data", "--output", "closeout.json"])
    assert validate.research_action.endswith("-validate")
    assert preflight.device == run.device == "cuda"
    assert report.research_action.endswith("-report")
    assert closeout.research_action.endswith("-closeout")
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_back_relay_axial_conjugate" in router
    assert "import optical_manifold_smf_bogt_back_relay_axial_conjugate" not in router


def test_write_once_and_no_update_source_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="codex-brac-") as temporary:
        output = Path(temporary) / "audit.json"
        output.write_text("{}", encoding="utf-8")
        with pytest.raises(FileExistsError):
            audit._ensure_absent(output)
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "evaluator_calls" in source
    assert "prior_sampled" in source


def test_baseline_replay_precedes_signed_candidate_evaluation() -> None:
    source = inspect.getsource(audit._run_measurement)
    assert source.index("baseline_selection_replay") < source.index("selection_evals =")
