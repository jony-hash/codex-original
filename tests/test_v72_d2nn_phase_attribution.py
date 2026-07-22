from __future__ import annotations

import json

import torch

from tools.audit_v72_d2nn_phase_attribution import gauge_align_fields
from tools.decide_v72_d2nn_phase_attribution import decide


def test_gauge_alignment_removes_global_phase_variation() -> None:
    base = torch.zeros((100, 10, 2, 1, 4, 4), dtype=torch.float32)
    base[:, :, 0] = 1.0
    phases = torch.linspace(-0.3, 0.3, 100).view(100, 1, 1, 1, 1)
    base[:, :, 0] = torch.cos(phases)
    base[:, :, 1] = torch.sin(phases)
    aligned = gauge_align_fields(base)
    assert float(aligned[:, :, 0].std()) < float(base[:, :, 0].std())
    assert float(aligned[:, :, 1].std()) < float(base[:, :, 1].std())
    assert torch.allclose(aligned.square().sum(2), torch.ones_like(aligned[:, :, 0]), atol=1e-5)


def _write_case(tmp_path, *, cameras: dict[str, float], readable: dict[str, list[int]], valid: bool = True) -> dict:
    tmp_path.mkdir()
    metrics = {"clean_prototype_corr": 0.9, "clean_prototype_dice": 0.9, "prototype_accuracy": 1.0, "high_saturation": 0.01}
    arms = {}
    for name, camera in cameras.items():
        arms[name] = {
            "camera_code_factor": camera,
            "finite": True,
            "energy_pass": True,
            "non_target_parameters_unchanged": valid,
            "targeted_parameters_zeroed": True,
            "optimizer_created": False,
            "parameter_update_count": 0,
            "crossed_metrics": metrics,
            "diagonal_metrics": metrics,
            "gauge_aligned_complex_factor_decomposition": {"decoder_layer_2": {"code_effect_ss": 120.0 if name != "baseline" else 100.0}},
        }
    prior = {"status": "v69-prototype-only"}
    v70 = {"status": "post-slm-code-factor-attenuated"}
    v71 = {"status": "d2nn-code-factor-attenuated"}
    report = {
        "status": "v72-phase-attribution-complete",
        "diagnostic_only": True,
        "test_split_used": False,
        "return_optical_fields": False,
        "batch_size": 1,
        "calibration_semantics": "slm_amplitude_squared",
        "calibration_replay_pass": True,
        "finite": True,
        "parameters_unchanged": valid,
        "optimizer_created": False,
        "parameter_update_count": 0,
        "arms": arms,
    }
    paths = {}
    for name, value in (("v69", prior), ("v70", v70), ("v71", v71), ("report", report), ("manual", {"arms": {key: {"readable_labels": value} for key, value in readable.items()}})):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return decide(
        v69_decision=paths["v69"], v70_decision=paths["v70"], v71_decision=paths["v71"],
        isolation_report=paths["report"], manual_review=paths["manual"], output=tmp_path / "decision.json",
    )


def test_decision_branches(tmp_path) -> None:
    arms = {"baseline": list(range(10)), "encoder2_identity": list(range(10)), "decoder1_identity": list(range(10)), "joint_identity": list(range(10))}
    cameras = {"baseline": 0.22, "encoder2_identity": 0.30, "decoder1_identity": 0.23, "joint_identity": 0.24}
    assert _write_case(tmp_path / "single", cameras=cameras, readable=arms)["status"] == "single-layer-trained-phase-suppression-confirmed"
    cameras["encoder2_identity"] = 0.20
    cameras["joint_identity"] = 0.30
    assert _write_case(tmp_path / "joint", cameras=cameras, readable=arms)["status"] == "multi-layer-trained-phase-suppression-confirmed"
    arms["joint_identity"] = [0, 1]
    assert _write_case(tmp_path / "tradeoff", cameras=cameras, readable=arms)["status"] == "phase-readability-code-tradeoff"
    cameras = {key: 0.20 for key in cameras}
    cameras["joint_identity"] = 0.215
    arms["joint_identity"] = list(range(10))
    assert _write_case(tmp_path / "mixed", cameras=cameras, readable=arms)["status"] == "mixed-phase-propagation-attenuation"
    cameras = {key: 0.20 for key in cameras}
    assert _write_case(tmp_path / "prop", cameras=cameras, readable=arms)["status"] == "propagation-or-active-window-attenuation"
    assert _write_case(tmp_path / "invalid", cameras={key: 0.3 for key in cameras}, readable=arms, valid=False)["status"] == "harness-invalid"
