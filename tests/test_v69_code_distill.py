from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from ldm_project.configuration.validation import _validate_one_shot_generator_config
from ldm_project.student.generator import ElectronicGeneratorE
from tools.audit_v69_code_distill import _StreamingFactor, _paired_teacher_alignment
from tools.decide_v69_code_distill import run as decide_v69


def test_code_delta_adds_raw_logits_without_tanh_or_scale():
    model = ElectronicGeneratorE(
        image_size=2,
        grid_size=2,
        code_dim=1,
        num_classes=1,
        label_embed_dim=1,
        hidden_dim=1,
        arch="label_template_code_delta",
        code_residual_scale=0.0,
        command_transform="direct_amplitude",
        template_init="mnist_prototype",
    )
    with torch.no_grad():
        model.template_logits.zero_()
        model.label_embedding.weight.zero_()
        for parameter in model.residual_net.parameters():
            parameter.zero_()
        model.residual_net[-1].bias.fill_(2.0)
    command, diagnostics = model.forward_with_diagnostics(
        torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)
    )
    assert torch.allclose(
        diagnostics["generator_code_delta_logits"], torch.full_like(command, 2.0)
    )
    assert torch.allclose(command, torch.sigmoid(torch.full_like(command, 2.0)))
    assert torch.allclose(
        diagnostics["generator_residual_map"], torch.full_like(command, 2.0)
    )


def test_code_delta_contract_rejects_nonzero_residual_scale():
    from ldm_project import load_training_config

    cfg = load_training_config("experiments/archive/v68_v69/configs/train_v69_code_distill.yaml")
    changed = replace(cfg, one_shot=replace(cfg.one_shot, code_residual_scale=0.1))
    with pytest.raises(ValueError, match="code delta is unscaled"):
        _validate_one_shot_generator_config(changed)


def test_streaming_factor_and_teacher_paired_advantage_are_code_sensitive():
    stream = _StreamingFactor((1, 1, 1))
    for code in range(100):
        for label in range(10):
            stream.add(code, label, torch.tensor([[[[float(code)]]]]))
    assert stream.report()["code_fraction"] > 0.99
    diagonal = torch.zeros(100, 1, 2, 2)
    teacher = torch.zeros_like(diagonal)
    labels = torch.arange(10).repeat_interleave(10)
    for index in range(100):
        diagonal[index, 0, index % 2, (index // 2) % 2] = 1.0
        teacher[index] = diagonal[index]
    report = _paired_teacher_alignment(diagonal, teacher, labels)
    assert report["teacher_advantage_ci95_lower"] >= 0.0
    assert report["bootstrap_seed"] == 1234


def test_v69_decision_all_status_branches(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    closure = write(
        "closure.json",
        {"status": "generator-injection-contract-insufficient", "v68_closed": True},
    )
    smoke = write(
        "smoke.json",
        {
            "schema": "v69_code_delta_warmstart_v1",
            "status": "v69-warmstart-smoke-passed",
            "finite": True,
            "frozen_hash_unchanged": True,
            "parameters_unchanged": True,
        },
    )
    history = write(
        "history.json",
        {
            "one_shot_distill_validation": [
                {"epoch": 4, "paired_corr": 0.8, "paired_dice": 0.8},
                {"epoch": 5, "paired_corr": 0.9, "paired_dice": 0.9},
            ]
        },
    )
    manual = write("manual.json", {"readable_labels": list(range(10)), "notes": "ok"})

    def crossed(*, code=0.3, teacher_advantage=0.1, corr=0.8):
        metrics = {
            "clean_prototype_corr": corr,
            "clean_prototype_dice": 0.8,
            "prototype_accuracy": 0.9,
            "high_saturation": 0.01,
        }
        return {
            "schema": "v69_crossed_raw_cmos_v1",
            "status": "v69-crossed-audit-complete",
            "test_split_used": False,
            "return_optical_fields": False,
            "finite": True,
            "parameters_unchanged": True,
            "camera_code_factor": code,
            "energy_records_mean": {"guard_energy_ratio": 1e-6},
            "crossed_metrics": metrics,
            "diagonal_metrics": metrics,
            "paired_teacher_alignment": {
                "diagonal_teacher_corr": 0.8,
                "diagonal_teacher_dice": 0.8,
                "teacher_advantage_mean": teacher_advantage,
                "teacher_advantage_ci95_lower": teacher_advantage,
                "bootstrap_seed": 1234,
                "bootstrap_resamples": 10000,
            },
        }

    output = tmp_path / "decision.json"
    args = dict(
        postmortem=closure,
        warmstart_report=smoke,
        training_history=history,
        manual_review=manual,
        output=output,
    )
    assert (
        decide_v69(crossed_report=write("passed.json", crossed()), **args)["status"]
        == "v69-code-distillation-passed"
    )
    assert (
        decide_v69(crossed_report=write("prototype.json", crossed(code=0.1)), **args)[
            "status"
        ]
        == "v69-prototype-only"
    )
    assert (
        decide_v69(
            crossed_report=write("unaligned.json", crossed(teacher_advantage=-0.01)),
            **args,
        )["status"]
        == "v69-code-transport-unaligned"
    )
    assert (
        decide_v69(crossed_report=write("readability.json", crossed(corr=0.2)), **args)[
            "status"
        ]
        == "v69-readability-regression"
    )
    bad_smoke = write("bad_smoke.json", {"schema": "wrong"})
    assert (
        decide_v69(
            crossed_report=write("invalid.json", crossed()),
            warmstart_report=bad_smoke,
            **{key: value for key, value in args.items() if key != "warmstart_report"},
        )["status"]
        == "harness-invalid"
    )
