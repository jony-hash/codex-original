from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research import optical_manifold_smf_bogt_backend_observable_subspace as audit


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_backend_observable_subspace_v1.yaml"
REVIEW = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_backend_observable_subspace_review_v1.json"


def _cell(prop: float, detector: float, power: float = 1.0) -> dict[str, object]:
    return {
        "r_prop": prop,
        "r_det": detector,
        "power_retention": power,
        "finite": True,
        "native_replay": {"passed": True},
    }


def _cells(prop_gap: float = 0.10, detector_gap: float = 0.10, power: float = 1.0) -> dict[str, dict[str, object]]:
    return {
        "I_to_I": _cell(0.90, 0.90, power),
        "I_to_S": _cell(0.90, 0.90, power),
        "S_to_I": _cell(0.90 - prop_gap, 0.90 - detector_gap, power),
        "S_to_S": _cell(0.90 - prop_gap, 0.90 - detector_gap, power),
    }


def test_contract_and_review_are_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec).name.endswith("backend_observable_subspace_review_v1.json")
    assert spec.payload["hypothesis"] == "H-LOMA-CBOGT-S1-BOSA"
    assert tuple(spec.payload["measurement"]["models"]) == audit.MODEL_NAMES
    assert tuple(spec.payload["measurement"]["cells"]) == tuple(dict(audit.CELL_REGISTRATION[name]) for name in audit.CELL_ORDER)


def test_contract_rejects_parent_reference_or_gate_mutation() -> None:
    spec = audit._load_spec(CONFIG)
    with pytest.raises(ValueError, match="parent DCM"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "parent_dcm": {}}, spec.config_hash))
    with pytest.raises(ValueError, match="gate contract"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "gates": {**spec.payload["gates"], "material_gap": 0.01}}, spec.config_hash))


def test_backend_plane_contract_is_physical_and_observable() -> None:
    formal = audit.mra._load_formal(audit.mra._load_spec(ROOT / audit.PARENT_MRA["config"]))[0]
    details, gates = audit._backend_validation(formal)
    assert all(gate.passed for gate in gates)
    assert details["source_plane"]["plane_id"] == "chip"
    assert details["source_plane"]["shape"] == [100, 100]
    assert details["target_observable_equivalence"] is True
    assert details["no_resampling"] is True


def test_four_cell_contrasts_and_backend_interaction() -> None:
    cells = _cells(0.10, 0.20)
    contrasts = audit._representation_contrasts(cells)
    assert contrasts == {
        "delta_repr_prop_Ibackend": pytest.approx(0.10),
        "delta_repr_prop_Sbackend": pytest.approx(0.10),
        "delta_repr_det_Ibackend": pytest.approx(0.20),
        "delta_repr_det_Sbackend": pytest.approx(0.20),
    }
    backend = audit._backend_contrasts(cells)
    assert backend["interaction_propagation"] == pytest.approx(0.0)
    assert backend["interaction_detector"] == pytest.approx(0.0)


def test_selection_gate_has_five_point_boundary_and_direction() -> None:
    passed, details = audit._selection_gate(_cells(0.05, 0.05))
    assert passed is True
    assert details["material"] is True
    failed, details = audit._selection_gate(_cells(0.049, 0.05))
    assert failed is False
    assert "representation_gap_insufficient" in details["failure_reasons"]
    power_confounded = _cells(0.10, 0.10, power=1.0)
    power_confounded["S_to_I"]["power_retention"] = 1.10
    failed, details = audit._selection_gate(power_confounded)
    assert failed is False
    assert details["power_controls"]["power_confound_present"] is True


def test_selection_gate_rejects_missing_native_or_finite_control() -> None:
    cells = _cells()
    cells["I_to_I"]["native_replay"] = {"passed": False}
    passed, details = audit._selection_gate(cells)
    assert passed is False
    assert "native_or_finite_control_failed" in details["failure_reasons"]


def test_power_controls_are_backend_local() -> None:
    cells = _cells(power=1.0)
    cells["S_to_I"]["power_retention"] = 0.98
    controls = audit._power_controls(cells)
    assert controls["representation_gap_by_backend"]["identity_backend"] == pytest.approx(0.02)
    assert controls["power_confound_present"] is False
    cells["S_to_S"]["power_retention"] = 0.94
    assert audit._power_controls(cells)["power_confound_present"] is True


def test_bootstrap_is_stratified_and_deterministic() -> None:
    labels = torch.arange(10).repeat_interleave(4)
    delta = torch.arange(labels.numel(), dtype=torch.float32) / 100.0
    first = audit._bootstrap_ci(delta, labels)
    second = audit._bootstrap_ci(delta, labels)
    assert first == second
    assert first[0] <= first[1]


def test_run_directory_rejects_existing_content(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "other.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit._ensure_run_directory(root / "audit.json", root / "validation.json")


def test_cli_actions_are_registered_and_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args(["research", "optical-manifold", "scaled-chen-backend-observable-subspace-validate", "--config", "config.yaml", "--review-contract", "review.json", "--dataset-root", "data", "--output", "validation.json"])
    run = parser.parse_args(["research", "optical-manifold", "scaled-chen-backend-observable-subspace-run", "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json", "--dataset-root", "data", "--output", "audit.json"])
    assert validate.research_action.endswith("backend-observable-subspace-validate")
    assert run.device == "cuda"
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_backend_observable_subspace" in router
    assert "import optical_manifold_smf_bogt_backend_observable_subspace" not in router


def test_no_update_source_contract() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "evaluator_calls" in source
    assert "generate_from_prior" not in source
    assert "torch.save(" not in source
