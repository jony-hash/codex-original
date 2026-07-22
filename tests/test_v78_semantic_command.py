from __future__ import annotations

import json

import pytest
import torch

from ldm_project import load_training_config
from ldm_project.student.experimental.semantic_command import (
    semantic_basis,
    semantic_command_residual,
    project_residual_to_basis,
)
from ldm_project.student.formal_contract import validate_formal_student_config
from ldm_project.student.generator import ElectronicGeneratorE
from tools.decide_v78_semantic_command_contract import decide


def test_basis_projection_reconstructs_basis_component_and_is_deterministic():
    basis = semantic_basis(16)
    coefficients = torch.arange(8, dtype=torch.float32)
    residual = torch.einsum("k,khw->hw", coefficients, basis)[None, None]
    pattern, recovered = project_residual_to_basis(residual, basis)
    assert torch.allclose(
        pattern,
        torch.einsum("bk,khw->bhw", recovered, basis)[:, None],
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.equal(basis, semantic_basis(16))


def test_semantic_command_weighted_zero_mean_and_fixed_amplitude():
    template = torch.full((1, 1, 16, 16), -1.0)
    residual = torch.randn_like(template)
    command, diagnostics = semantic_command_residual(template, residual)
    support = torch.sigmoid(template)
    weighted = (support * diagnostics["generator_semantic_pattern"]).sum()
    assert abs(float(weighted)) < 1.0e-5
    assert torch.all((command >= 0.0) & (command <= 1.0))
    assert torch.allclose(
        diagnostics["generator_command_delta"],
        diagnostics["generator_semantic_pattern"] * 0.4027685085311532,
    )


def test_semantic_architecture_reuses_delta_state_keys_and_formal_rejects():
    delta = ElectronicGeneratorE(8, 8, 4, 2, 3, 5, arch="label_template_code_delta", command_transform="intensity_to_amplitude", template_init="mnist_prototype")
    semantic = ElectronicGeneratorE(8, 8, 4, 2, 3, 5, arch="label_template_semantic_command_residual", command_transform="intensity_to_amplitude", template_init="mnist_prototype")
    assert set(delta.state_dict()) == set(semantic.state_dict())
    cfg = load_training_config("configs/train_v78_semantic_command_contract.yaml")
    with pytest.raises(ValueError, match="semantic command"):
        validate_formal_student_config(cfg.physics, cfg.one_shot)


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_v78_decision_metric_and_harness_branches(tmp_path):
    v77 = _write(tmp_path / "v77.json", {"status": "phase-aware-local-co-design-admissible"})
    admission = {
        "status": "v78-cpu-admission-passed",
        "metrics": {
            "physical_slm_intensity_code_factor": 0.3,
            "support_energy_fraction": 0.96,
            "low_frequency_energy_fraction": 0.995,
            "one_clipping_fraction": 0.01,
            "finite": True,
        },
        "checks": {
            "parameters_unchanged": True,
            "v77_status": True,
            "validation_contract": True,
            "checkpoint_contract": True,
            "checkpoint_state_compatible": True,
            "formula": True,
            "finite": True,
            "optimizer_created": False,
            "calibration_or_sweep": False,
        },
    }
    cpu = _write(tmp_path / "cpu.json", admission)
    smoke = _write(tmp_path / "smoke.json", {"status": "v78-semantic-command-smoke-passed", "finite": True, "template_embedding_frozen": True})
    out = tmp_path / "decision.json"
    assert decide(v77_decision=v77, admission=cpu, smoke=smoke, output=out)["status"] == "v78-semantic-command-contract-admitted"
    admission["metrics"]["physical_slm_intensity_code_factor"] = 0.1
    cpu.write_text(json.dumps(admission), encoding="utf-8")
    assert decide(v77_decision=v77, admission=cpu, smoke=None, output=out)["status"] == "v78-command-code-factor-insufficient"
    admission["metrics"]["physical_slm_intensity_code_factor"] = 0.3
    admission["metrics"]["finite"] = False
    cpu.write_text(json.dumps(admission), encoding="utf-8")
    assert decide(v77_decision=v77, admission=cpu, smoke=None, output=out)["status"] == "harness-invalid"
