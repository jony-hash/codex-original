from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research import optical_manifold_smf_bogt_decoder_cmos_attribution as audit


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_decoder_cmos_attribution_v1.yaml"
REVIEW = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_decoder_cmos_attribution_review_v1.json"


def _retention(relay: float, detector: float) -> dict[str, object]:
    return {
        "retention": {
            audit.SEGMENTS[0][0]: {"geometric_mean_retention": relay},
            audit.SEGMENTS[1][0]: {"geometric_mean_retention": detector},
        },
        "label_signal": {
            "relative_inter_intra_drop": 0.10,
            "retrieval_drop": 0.10,
        },
    }


def _identity(relay: float, detector: float) -> dict[str, object]:
    return _retention(relay, detector)


def test_contract_and_review_are_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec).name.endswith("decoder_cmos_attribution_review_v1.json")
    assert spec.payload["hypothesis"] == "H-LOMA-CBOGT-S0-DCM-OBS"


def test_contract_rejects_parent_or_gate_mutation() -> None:
    spec = audit._load_spec(CONFIG)
    with pytest.raises(ValueError, match="parent MRA"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "parent_mra": {}}, spec.config_hash))
    with pytest.raises(ValueError, match="gate contract"):
        audit._assert_contract(audit.Spec(spec.path, {**spec.payload, "gates": {**spec.payload["gates"], "material_gap": 0.01}}, spec.config_hash))


def test_maximum_discontinuity_requires_same_panel_and_five_points() -> None:
    selection = _retention(0.60, 0.85)
    confirmation = _retention(0.61, 0.86)
    result = audit._maximum_discontinuity(selection, confirmation)
    assert result["status"] == "unique"
    assert result["segment"] == audit.SEGMENTS[0][0]
    assert audit._maximum_discontinuity(_retention(0.70, 0.70), _retention(0.70, 0.70))["status"] == "joint"


def test_readout_classification_boundaries() -> None:
    relay = _retention(0.70, 0.95)
    assert audit._classify_readout(relay, relay, _identity(0.95, 0.95))[0] == "relay_propagation_bottleneck_supported"
    detector = _retention(0.95, 0.70)
    assert audit._classify_readout(detector, detector, _identity(0.95, 0.95))[0] == "phase_to_intensity_readout_bottleneck_supported"
    joint = _retention(0.70, 0.70)
    assert audit._classify_readout(joint, joint, _identity(0.95, 0.95))[0] == "joint_decoder_readout_bottleneck_supported"
    under = _retention(0.95, 0.70)
    under["label_signal"] = {"relative_inter_intra_drop": 0.0, "retrieval_drop": 0.0}
    assert audit._classify_readout(under, under, _identity(0.95, 0.95))[0] == "readout_attribution_underdetermined"


def test_phase_controls_preserve_square_law_intensity() -> None:
    generator = torch.Generator(device="cpu").manual_seed(3)
    incident = torch.polar(torch.rand((2, 1, 8, 8), generator=generator) + 0.1, torch.rand((2, 1, 8, 8), generator=generator) * 6.0 - 3.0)
    intensity = incident.abs().square()
    controls = audit._phase_controls(incident, intensity, torch.Generator(device="cpu").manual_seed(4))
    assert controls["global_phase_max_abs_error"] <= audit.PHASE_INVARIANCE_TOLERANCE
    assert controls["pointwise_phase_max_abs_error"] <= audit.PHASE_INVARIANCE_TOLERANCE


def test_cli_actions_are_registered_and_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args(["research", "optical-manifold", "scaled-chen-decoder-cmos-attribution-validate", "--config", "config.yaml", "--review-contract", "review.json", "--dataset-root", "data", "--output", "validation.json"])
    run = parser.parse_args(["research", "optical-manifold", "scaled-chen-decoder-cmos-attribution-run", "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json", "--dataset-root", "data", "--output", "audit.json"])
    assert validate.research_action.endswith("decoder-cmos-attribution-validate")
    assert run.device == "cuda"
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_decoder_cmos_attribution" in router
    assert "import optical_manifold_smf_bogt_decoder_cmos_attribution" not in router


def test_run_directory_rejects_existing_content(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "other.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit._ensure_run_directory(root / "audit.json", root / "validation.json")


def test_no_update_source_contract() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "evaluator_calls" in source
    assert "generate_from_prior" not in source
