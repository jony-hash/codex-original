from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_reconstruction_manifold_failure as audit


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_reconstruction_manifold_failure_v1.yaml"
REVIEW = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_reconstruction_manifold_failure_review_v1.json"


def _report(*, improvement: float = 0.10, probe: float = 0.02, template: float = 0.03, cyclic: float = 0.08, replay: bool = True) -> dict[str, object]:
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
    }


def test_config_and_review_are_exactly_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec) == REVIEW
    assert spec.payload["hypothesis"] == audit.HYPOTHESIS
    assert spec.payload["probe"]["lambda"] is None
    assert spec.payload["fit_panel"]["indices_sha256"] == audit.FIT_PANEL["indices_sha256"]


def test_contract_rejects_mutation_and_preserves_primary_scope() -> None:
    spec = audit._load_spec(CONFIG)
    with pytest.raises(ValueError, match="fit_panel"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "fit_panel": {**spec.payload["fit_panel"], "per_label": 99}}, spec.config_hash))
    with pytest.raises(ValueError, match="probe"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "probe": {**spec.payload["probe"], "adaptive_rank": True}}, spec.config_hash))


def test_minimum_norm_solver_is_deterministic_and_fixed() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    features = torch.randn((12, 512), generator=generator)
    targets = torch.randn((12, 40000), generator=generator)
    first, first_meta = audit._solve_minimum_norm(features, targets)
    second, second_meta = audit._solve_minimum_norm(features, targets)
    assert torch.equal(first, second)
    assert first_meta == second_meta
    assert first.shape == (513, 40000)
    assert audit.PROBE["regularization"] == "none_parameter_free"
    assert audit.PROBE["adaptive_rank"] is False


def test_bootstrap_is_stratified_and_deterministic() -> None:
    labels = torch.arange(100) % 10
    delta = torch.linspace(0.01, 0.20, 100)
    assert audit._stratified_bootstrap_ci(delta, labels, replicates=20) == audit._stratified_bootstrap_ci(delta, labels, replicates=20)


def test_selection_and_confirmation_boundaries() -> None:
    selection = _report()
    selection["selection_gate"] = audit._panel_gate(selection, require_bootstrap=False)
    assert selection["selection_gate"]["passed"] is True
    confirmation = _report()
    confirmation["selection_gate"] = audit._panel_gate(confirmation, require_bootstrap=False)
    confirmation["confirmation_gate"] = audit._panel_gate(confirmation, require_bootstrap=True)
    assert audit._classify(selection, confirmation) == ("reconstruction_mapping_misalignment_supported", audit.NEXT_ACTION_R1)
    failed = _report(improvement=0.01)
    failed["selection_gate"] = audit._panel_gate(failed, require_bootstrap=False)
    assert audit._classify(failed) == ("bounded_reconstruction_recoverability_not_supported", audit.NEXT_ACTION_R2)
    under = _report()
    under["probe_contract_valid"] = False
    under["selection_gate"] = audit._panel_gate(under, require_bootstrap=False)
    assert audit._classify(under) == ("reconstruction_failure_attribution_underdetermined", audit.NEXT_ACTION_R3)


def test_template_crossing_and_control_fail_closed() -> None:
    not_crossed = _report(probe=0.04, template=0.03)
    not_crossed["selection_gate"] = audit._panel_gate(not_crossed, require_bootstrap=False)
    assert not_crossed["selection_gate"]["checks"]["strict_template_crossing"] is False
    not_replayed = _report(replay=False)
    not_replayed["selection_gate"] = audit._panel_gate(not_replayed, require_bootstrap=False)
    assert not_replayed["selection_gate"]["passed"] is False


def test_write_once_validation_slot(tmp_path: Path) -> None:
    output = tmp_path / "attempt" / "validation.json"
    audit._prepare_validation_slot(output)
    assert output.parent.is_dir()
    with pytest.raises(FileExistsError):
        audit._prepare_validation_slot(output)


def test_cli_actions_are_registered_and_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args(["research", "optical-manifold", "scaled-chen-reconstruction-manifold-failure-validate", "--config", "config.yaml", "--review-contract", "review.json", "--dataset-root", "data", "--output", "validation.json"])
    run = parser.parse_args(["research", "optical-manifold", "scaled-chen-reconstruction-manifold-failure-run", "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json", "--dataset-root", "data", "--output", "audit.json"])
    assert validate.research_action.endswith("reconstruction-manifold-failure-validate")
    assert run.device == "cuda"
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_reconstruction_manifold_failure" in router
    assert "import optical_manifold_smf_bogt_reconstruction_manifold_failure" not in router


def test_no_update_boundary_is_visible_in_source() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "evaluator_calls" in source
    assert "physical_forward_modified" in source
    assert audit.AUTHORIZATION["mechanism_evidence_only"] is True
