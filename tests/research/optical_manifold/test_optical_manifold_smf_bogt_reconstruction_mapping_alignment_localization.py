from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization as audit


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization_v1.yaml"
REVIEW = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization_review_v1.json"


def _report(
    *,
    improvement: float = 0.10,
    probe: float = 0.02,
    template: float = 0.03,
    cyclic: float = 0.08,
    replay: bool = True,
    square_law: bool = True,
    feature_contract: bool = True,
    probe_contract: bool = True,
) -> dict[str, object]:
    return {
        "physical": {"paired_mse": 0.06},
        "probe": {"paired_mse": probe, "same_label_top1": 0.40},
        "class_template_mse": template,
        "cyclic_mse": cyclic,
        "relative_improvement": improvement,
        "bootstrap": {"ci": [0.01, 0.02]},
        "native_replay": {"paired_mse": replay, "class_template_mse": replay, "cyclic_mse": replay},
        "finite": True,
        "instance_sensitive": True,
        "state_unchanged": True,
        "ols_state_unchanged": True,
        "square_law_consistency": square_law,
        "feature_contract_valid": feature_contract,
        "probe_contract_valid": probe_contract,
    }


def test_config_and_review_are_exactly_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec) == REVIEW
    assert spec.payload["hypothesis"] == audit.HYPOTHESIS
    assert spec.payload["probe"]["lambda"] is None
    assert spec.payload["stage_order"] == list(audit.STAGE_ORDER)
    assert spec.payload["stages"]["cmos_intensity"]["input_dim"] == 257
    assert spec.payload["stages"]["modal_complex_100"]["input_dim"] == 201
    assert spec.payload["canonicalization"] == audit.CANONICALIZATION


def test_contract_rejects_mutation_and_preserves_scope() -> None:
    spec = audit._load_spec(CONFIG)
    with pytest.raises(ValueError, match="stage_order"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "stage_order": list(reversed(spec.payload["stage_order"]))}, spec.config_hash))
    with pytest.raises(ValueError, match="canonicalization"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "canonicalization": {**spec.payload["canonicalization"], "fit": True}}, spec.config_hash))


def test_stage_order_and_feature_dimensions_are_fixed() -> None:
    assert audit.STAGE_ORDER == (
        "cmos_intensity",
        "cmos_incident_complex_field",
        "decoder_layer_2",
        "decoder_input",
        "modal_complex_100",
        "pre_ols",
    )
    assert [audit.STAGE_CONTRACT[name]["input_dim"] for name in audit.STAGE_ORDER] == [257, 513, 513, 513, 201, 513]


def test_mra_feature_projection_dimensions_are_used() -> None:
    real = torch.zeros((2, 1, 200, 200), dtype=torch.float32)
    complex_field = torch.zeros((2, 1, 100, 100), dtype=torch.complex64)
    modal = torch.zeros((2, 100), dtype=torch.complex64)
    assert audit.mra._feature_projection(real, complex_value=False).shape == (2, 256)
    assert audit.mra._feature_projection(complex_field, complex_value=True).shape == (2, 512)
    assert audit.mra._feature_projection(modal, complex_value=True).shape == (2, 200)


def test_feature_view_canonicalizes_only_incident_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_canonicalize(value: torch.Tensor, backend: str) -> torch.Tensor:
        calls.append((backend, tuple(value.shape)))
        return torch.zeros((value.shape[0], 1, 100, 100), dtype=torch.complex128)

    monkeypatch.setattr(audit.hfrr, "_canonicalize_field", fake_canonicalize)
    values = {
        "cmos_intensity": torch.zeros((2, 1, 200, 200), dtype=torch.float32),
        "cmos_incident_complex_field": torch.zeros((2, 1, 200, 200), dtype=torch.complex64),
        "decoder_layer_2": torch.zeros((2, 1, 100, 100), dtype=torch.complex64),
        "decoder_input": torch.zeros((2, 1, 100, 100), dtype=torch.complex64),
        "modal_complex_100": torch.zeros((2, 100), dtype=torch.complex64),
        "pre_ols": torch.zeros((2, 1, 100, 100), dtype=torch.complex64),
    }
    raw = {name: {"shape": list(value.shape[1:]), "dtype": str(value.dtype), "plane": {}} for name, value in values.items()}
    capture = audit.CapturedStages(values, {}, raw, 0.0, True)
    formal = object()
    incident = audit._feature_for_stage(capture, "cmos_incident_complex_field", formal)  # type: ignore[arg-type]
    decoder = audit._feature_for_stage(capture, "decoder_layer_2", formal)  # type: ignore[arg-type]
    assert incident.values.shape == (2, 512)
    assert decoder.values.shape == (2, 512)
    assert calls == [("scaled_chen_backend", (2, 1, 200, 200))]
    assert incident.metadata["canonicalization"] == audit.CANONICALIZATION
    assert decoder.metadata["canonicalization"] is None


def test_minimum_norm_solver_is_deterministic_and_fixed() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    features = torch.randn((4, 512), generator=generator)
    targets = torch.randn((4, 40000), generator=generator)
    first, first_meta = audit._solve_minimum_norm(features, targets)
    second, second_meta = audit._solve_minimum_norm(features, targets)
    assert torch.equal(first, second)
    assert first_meta == second_meta
    assert first.shape == (513, 40000)
    assert audit.PROBE["regularization"] == "none_parameter_free"
    assert audit.PROBE["adaptive_rank"] is False
    assert audit.PROBE["rcond"] == 1.0e-12


def test_bootstrap_is_inherited_and_deterministic() -> None:
    labels = torch.arange(100) % 10
    delta = torch.linspace(0.01, 0.20, 100)
    assert audit._stratified_bootstrap_ci(delta, labels, replicates=20) == audit._stratified_bootstrap_ci(delta, labels, replicates=20)


def test_panel_gate_inherits_rmfa_and_fail_closes_harness_controls() -> None:
    passed = _report()
    passed_gate = audit._panel_gate(passed, require_bootstrap=False)
    assert passed_gate["passed"] is True
    assert passed_gate["checks"]["square_law_consistency"] is True
    assert passed_gate["checks"]["feature_contract"] is True
    assert passed_gate["checks"]["probe_contract"] is True
    for key, kwargs in {
        "relative_improvement": {"improvement": 0.01},
        "template_crossing": {"probe": 0.04, "template": 0.03},
        "cyclic_control": {"cyclic": 0.01},
        "square_law": {"square_law": False},
        "feature_contract": {"feature_contract": False},
    }.items():
        report = _report(**kwargs)
        gate = audit._panel_gate(report, require_bootstrap=False)
        assert gate["passed"] is False, key


def test_confirmation_bootstrap_and_result_classification() -> None:
    selection = _report()
    selection["selection_gate"] = audit._panel_gate(selection, require_bootstrap=False)
    confirmation = _report()
    confirmation["selection_gate"] = audit._panel_gate(confirmation, require_bootstrap=False)
    confirmation["confirmation_gate"] = audit._panel_gate(confirmation, require_bootstrap=True)
    assert confirmation["confirmation_gate"]["passed"] is True
    assert audit._stage_classification("cmos_intensity") == "bounded_recoverability_established_at_detector"
    assert audit._stage_classification("decoder_input") == "bounded_recoverability_established_at_decoder_input"
    failed = _report(improvement=0.01)
    failed["selection_gate"] = audit._panel_gate(failed, require_bootstrap=False)
    assert failed["selection_gate"]["passed"] is False


def test_write_once_validation_slot(tmp_path: Path) -> None:
    output = tmp_path / "attempt" / "validation.json"
    audit._prepare_validation_slot(output)
    assert output.parent.is_dir()
    with pytest.raises(FileExistsError):
        audit._prepare_validation_slot(output)


def test_cli_actions_are_registered_and_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args([
        "research", "optical-manifold", "scaled-chen-reconstruction-mapping-alignment-localization-validate",
        "--config", "config.yaml", "--review-contract", "review.json", "--dataset-root", "data", "--output", "validation.json",
    ])
    run = parser.parse_args([
        "research", "optical-manifold", "scaled-chen-reconstruction-mapping-alignment-localization-run",
        "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json",
        "--dataset-root", "data", "--output", "audit.json",
    ])
    assert validate.research_action.endswith("reconstruction-mapping-alignment-localization-validate")
    assert run.device == "cuda"
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization" in router
    assert "import optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization" not in router


def test_capture_contract_is_wired_to_frozen_physical_objects() -> None:
    source = inspect.getsource(audit._capture)
    assert "output.ols_field.tensor" in source
    assert "output.coefficients" in source
    assert "model.chen_relay(chip)" in source
    assert "expected_intensity = incident.tensor.abs().square()" in source


def test_sequential_stop_and_failure_semantics_are_visible() -> None:
    source = inspect.getsource(audit.run)
    assert "for stage in STAGE_ORDER" in source
    assert "if not selection[\"selection_gate\"][\"passed\"]" in source
    assert '"outcome": "bounded_recoverability_not_established"' in source
    assert '"outcome": "bounded_recoverability_not_established_on_confirmation"' in source
    assert "winner = stage" in source
    assert "break" in source
    assert "stages_skipped_by_stop_rule" in source


def test_no_update_boundary_is_visible_in_source() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "evaluator_calls" in source
    assert "physical_forward_modified" in source
    assert audit.AUTHORIZATION["mechanism_evidence_only"] is True
