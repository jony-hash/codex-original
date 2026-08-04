from __future__ import annotations

import importlib.util

import pytest
import torch


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v63_code_conditioning_provenance_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("v63_code_provenance", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crossed_codes_and_labels_are_deterministic_and_balanced():
    module = _module()
    codes = torch.arange(100 * 128, dtype=torch.float32).reshape(100, 128)
    crossed_codes, crossed_labels = module.crossed_codes_and_labels(codes)
    assert tuple(crossed_codes.shape) == (1000, 128)
    assert tuple(crossed_labels.shape) == (1000,)
    assert torch.equal(crossed_codes[:10], codes[0].expand(10, -1))
    assert torch.equal(crossed_labels[:10], torch.arange(10))
    assert {label: int((crossed_labels == label).sum()) for label in range(10)} == {label: 100 for label in range(10)}


def test_crossed_design_rejects_non_native_test_code_shape():
    module = _module()
    with pytest.raises(ValueError, match="requires test codes"):
        module.crossed_codes_and_labels(torch.zeros(99, 128))


def test_balanced_factor_decomposition_separates_code_label_and_interaction():
    module = _module()
    code = torch.arange(100, dtype=torch.float32).view(100, 1, 1, 1, 1)
    label = torch.arange(10, dtype=torch.float32).view(1, 10, 1, 1, 1)
    code_only = code.expand(100, 10, 1, 1, 1)
    label_only = label.expand(100, 10, 1, 1, 1)
    interaction = (code * label).expand(100, 10, 1, 1, 1)
    factors, _ = module.balanced_factor_decomposition(code_only)
    assert factors["code_fraction"] == pytest.approx(1.0)
    assert factors["label_fraction"] == pytest.approx(0.0)
    factors, _ = module.balanced_factor_decomposition(label_only)
    assert factors["label_fraction"] == pytest.approx(1.0)
    factors, _ = module.balanced_factor_decomposition(interaction)
    assert factors["interaction_fraction"] > 0.1
    assert factors["fraction_sum"] == pytest.approx(1.0)


def test_factor_decomposition_requires_the_pre_registered_cross_shape():
    module = _module()
    with pytest.raises(ValueError, match="factor values"):
        module.balanced_factor_decomposition(torch.zeros(99, 10, 1, 2, 2))


def test_decision_selects_localized_drop_before_representation_confirmation():
    module = _module()

    def factors(*, front_code: float, camera_code: float):
        return {
            "slm_command": {"code_fraction": 0.8, "label_fraction": 0.1},
            "pre_encoder_intensity": {"code_fraction": front_code, "label_fraction": 0.1},
            "chip_intensity": {"code_fraction": camera_code, "label_fraction": 0.1},
            "camera_intensity": {"code_fraction": camera_code, "label_fraction": 0.1},
        }

    local = module._decision(factors(front_code=0.01, camera_code=0.01), v62_residual_valid=True)
    assert local["status"] == "localized-code-factor-attenuation-candidate"
    confirmed = module._decision(factors(front_code=0.5, camera_code=0.4), v62_residual_valid=True)
    assert confirmed["status"] == "native-code-conditioned-subvisual-representation-confirmed"


def test_v62_residual_gate_requires_matching_and_permuted_control():
    module = _module()
    assert module._v62_residual_is_valid({}) is False
    assert module._v62_residual_is_valid(
        {
            "matched_residual": {"corr_mean": 0.2},
            "residual_corr_vs_within_class_target_rotation": {"ci_low": 0.01},
        }
    ) is True


def test_state_snapshot_detects_any_parameter_update():
    module = _module()
    model = torch.nn.Linear(2, 2)
    before = module.V62._state_snapshot(model)
    assert module.V62._state_changes(model, before) == []
    with torch.no_grad():
        model.weight.add_(1.0)
    assert module.V62._state_changes(model, before) == ["weight"]


def test_tool_uses_no_resize_or_optimizer_path():
    source = TOOL.read_text(encoding="utf-8")
    assert "interpolate(" not in source
    assert "adaptive_avg_pool" not in source
    assert "grid_sample" not in source
    assert "optimizer" not in source.lower().replace("optimizer_created", "")
