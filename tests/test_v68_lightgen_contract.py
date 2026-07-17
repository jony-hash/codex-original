from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.configuration.schema import D2NNConfig
from ldm_project.configuration.schema import OneShotConfig, PhysicsConfig
from ldm_project.student.formal_contract import validate_formal_student_config
from ldm_project.student.checkpoints import (
    load_one_shot_checkpoint,
    load_one_shot_checkpoint_subset,
)
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from ldm_project.student.contracts import ONE_SHOT_MODEL_VERSION
from tools.diagnose_v68_transparent_memory import (
    _build_prototype_bank,
    _compare,
    _tensor_hash,
)
from ldm_project.student.experimental.transparent_phase import (
    TransparentIdentityDiffractiveStack,
)
from tools.audit_v68_lightgen_contract import audit_config
from tools.audit_v68_raw_cmos import (
    _class_metrics,
    fixed_exposure_view,
    global_exposure_scale,
    per_sample_normalized_view,
)
from ldm_project.student.training import _save_transparent_active_artifact


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name,branch_key",
    [
        ("train_v68_transparent_readability_a.yaml", "branch_a_camera_only_objective"),
        ("train_v68_transparent_readability_b.yaml", "branch_b_minimal_readability_objective"),
    ],
)
def test_v68_lightgen_contract_audit_is_read_only_and_valid(
    name: str, branch_key: str
) -> None:
    report = audit_config(ROOT / "configs" / name)
    assert report["diagnostic_only"] is True
    assert report["status"] == "lightgen-calibrated-project-contract-valid"
    assert report["checks"]["amplitude_slm"] is True
    assert report["checks"]["transparent_boundary"] is True
    assert report["checks"]["checkpointed_active_capture"] is True
    assert report["checks"][branch_key] is True


def test_v68_memory_mode_is_explicit_and_validated() -> None:
    assert D2NNConfig().transparent_memory_mode == "legacy_full_capture"
    with pytest.raises(ValueError, match="transparent_memory_mode"):
        D2NNConfig(transparent_memory_mode="not-a-mode")


def test_formal_contract_rejects_checkpointed_transparent_memory_mode() -> None:
    config = OneShotConfig(
        optical_core_arch="diffractive_phase_only",
        physical_forward_contract=True,
        d2nn=D2NNConfig(transparent_memory_mode="checkpointed_active_capture"),
    )
    with pytest.raises(ValueError, match="transparent_memory_mode"):
        validate_formal_student_config(
            PhysicsConfig(physical_forward_contract=True), config
        )


def test_formal_contract_rejects_fixed_exposure_diagnostic_mode() -> None:
    config = OneShotConfig(
        optical_core_arch="diffractive_phase_only",
        physical_forward_contract=True,
        diagnostic_camera_normalization="fixed_exposure",
        diagnostic_camera_exposure_scale=0.5,
        d2nn=D2NNConfig(),
    )
    with pytest.raises(ValueError, match="diagnostic_camera_normalization"):
        validate_formal_student_config(
            PhysicsConfig(physical_forward_contract=True), config
        )


def test_fixed_exposure_uses_one_scale_for_all_samples() -> None:
    raw = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[2.0, 4.0], [6.0, 8.0]]]])
    scale = global_exposure_scale(raw)
    view = fixed_exposure_view(raw, scale)
    assert scale == pytest.approx(float(torch.quantile(raw.flatten(), 0.995)))
    assert torch.allclose(view[0], raw[0] / scale)
    assert view[1, 0, -1, -1].item() == 1.0
    assert torch.allclose(view[0, ..., :1, :], view[1, ..., :1, :] / 2.0)


def test_locked_exposure_requires_fixed_exposure_config() -> None:
    from ldm_project import load_training_config

    a = load_training_config(ROOT / "configs" / "train_v68_transparent_readability_a.yaml")
    a2 = load_training_config(ROOT / "configs" / "train_v68_transparent_readability_a2.yaml")
    b = load_training_config(ROOT / "configs" / "train_v68_transparent_readability_b.yaml")
    assert a.one_shot.diagnostic_camera_normalization == "per_sample_percentile"
    assert a2.one_shot.diagnostic_camera_normalization == "fixed_exposure"
    assert a2.one_shot.diagnostic_camera_exposure_scale == pytest.approx(0.4913053810596466)
    assert b.one_shot.diagnostic_camera_normalization == "fixed_exposure"
    assert b.one_shot.checkpoint_selection_metric == "strict_readability"


def test_raw_audit_helpers_keep_clean_and_teacher_targets_separate() -> None:
    prediction = torch.zeros(4, 1, 2, 2)
    prediction[:, :, 0, 0] = 1.0
    target = prediction.clone()
    labels = torch.tensor([0, 0, 1, 1])
    normalized = per_sample_normalized_view(prediction + 0.1)
    metrics = _class_metrics(prediction, target, labels)
    assert set(metrics) == {"0", "1"}
    assert metrics["0"]["corr"] == pytest.approx(1.0)
    assert normalized.shape == prediction.shape


def test_transparent_checkpoint_is_rejected_by_formal_target() -> None:
    config = OneShotConfig(
        optical_core_arch="diffractive_phase_only",
        physical_forward_contract=True,
        d2nn=D2NNConfig(),
    )
    payload = {
        "model_version": ONE_SHOT_MODEL_VERSION,
        "model_state_dict": {},
        "optical_boundary_contract": "ideal_transparent_thin_phase_v2",
    }

    target = type("_Target", (), {"config": config})()
    with pytest.raises(ValueError, match="optical boundary contract"):
        load_one_shot_checkpoint(target, payload)


def test_v68_a2_config_declares_fixed_exposure_and_compact_mode() -> None:
    from ldm_project import load_training_config

    config = load_training_config(
        ROOT / "configs" / "train_v68_transparent_readability_a2.yaml"
    )
    assert config.one_shot.diagnostic_camera_normalization == "fixed_exposure"
    assert config.one_shot.diagnostic_camera_exposure_scale > 0.0
    assert config.one_shot.checkpoint_selection_metric == "strict_readability"
    assert config.one_shot.d2nn.transparent_memory_mode == "checkpointed_active_capture"


def test_compact_transparent_artifact_excludes_full_field_bundle() -> None:
    class _Capture:
        def __init__(self) -> None:
            self.json_payload = None
            self.tensor_payload = None

        def save_artifact_json(self, _name, payload):
            self.json_payload = payload

        def save_tensor_bundle(self, _name, payload):
            self.tensor_payload = payload

    capture = _Capture()
    _save_transparent_active_artifact(
        capture,
        {
            "optical_boundary_contract": "ideal_transparent_thin_phase_v2",
            "ideal_phase_provenance": {
                "transparent_artifact_view": "center_crop_from_internal_field"
            },
            "physical_camera_intensity_raw": torch.ones(1, 1, 4, 4),
            "optical_fields": {"full_field": torch.ones(1, 1, 2048, 2048)},
        },
        artifact_prefix="epoch_0001",
    )
    assert capture.json_payload["full_field_saved"] is False
    assert capture.json_payload["active_view_only"] is True
    assert set(capture.tensor_payload) == {"physical_camera_intensity_raw"}


def test_v68_prototype_bank_hash_is_reproducible() -> None:
    from ldm_project import load_training_config

    config = load_training_config(
        ROOT / "configs" / "train_v68_transparent_readability_a.yaml"
    )
    assert _tensor_hash(_build_prototype_bank(config)) == _tensor_hash(
        _build_prototype_bank(config)
    )


def test_b_warmstart_loads_template_and_phase_but_not_residual() -> None:
    common = dict(
        optical_core_arch="diffractive_phase_only",
        physical_forward_contract=False,
        code_dim=8,
        label_embed_dim=4,
        hidden_dim=8,
        num_classes=3,
        generator_command_transform="intensity_to_amplitude",
        generator_template_init="mnist_prototype",
        distill_target_source="mnist_prototype",
        d2nn=D2NNConfig(
            grid_size=4,
            propagation_grid_size=8,
            pixel_size=2.0e-6,
            layer_spacing=1.0e-5,
            exterior_boundary="transparent_identity",
        ),
    )
    source_cfg = OneShotConfig(generator_e_arch="label_template", **common)
    target_cfg = OneShotConfig(
        generator_e_arch="label_template_code_residual",
        code_residual_scale=0.10,
        **common,
    )
    source = OneShotOpticalGenerator(PhysicsConfig(), source_cfg, image_size=8)
    target = OneShotOpticalGenerator(PhysicsConfig(), target_cfg, image_size=8)
    with torch.no_grad():
        source.generator_e.template_logits.fill_(0.25)
    residual_before = target.generator_e.residual_net[0].weight.detach().clone()
    summary = load_one_shot_checkpoint_subset(
        target,
        {"model_version": ONE_SHOT_MODEL_VERSION, "model_state_dict": source.state_dict()},
        include_prefixes=("generator_e.",),
        description="v68 B template warm-start",
        expected_missing_prefixes=(
            "generator_e.label_embedding.",
            "generator_e.residual_net.",
        ),
    )
    assert summary["loaded"] == 1
    assert torch.equal(
        target.generator_e.template_logits, source.generator_e.template_logits
    )
    assert torch.equal(target.generator_e.residual_net[0].weight, residual_before)


def test_v68_admission_report_is_not_success_when_a_plane_gate_fails() -> None:
    torch.manual_seed(1234)
    fields = {
        name: torch.randn(1, 1, 100, 100, dtype=torch.complex64)
        for name in (
            "encoder_layer_1",
            "encoder_layer_2",
            "encoder_output",
            "decoder_layer_1",
            "decoder_layer_2",
            "chip",
        )
    }
    base = {
        "fields": fields,
        "camera": torch.ones(1, 1, 8, 8),
        "energy_records": [
            {
                "encoder_propagation_energy_ratio": 1.0,
                "encoder_retained_energy": 1.0,
                "encoder_guard_band_energy_ratio": 0.0,
                "encoder_discarded_energy_ratio": 0.0,
            }
        ],
        "initialization": {},
    }
    assert _compare(base, base)["all_plane_gates_pass"] is True
    failed = {**base, "fields": {**fields, "chip": fields["chip"] * 1.1}}
    assert _compare(failed, base)["all_plane_gates_pass"] is False


def _make_stack(memory_mode: str) -> TransparentIdentityDiffractiveStack:
    return TransparentIdentityDiffractiveStack(
        num_layers=2,
        grid_size=4,
        layer_spacing=1.0e-5,
        wavelength=532.0e-9,
        pixel_size=2.0e-6,
        propagation_grid_size=8,
        phase_init="small",
        transparent_propagation_method="padded_band_limited_asm",
        transparent_fft_grid_size=16,
        transparent_memory_mode=memory_mode,
    )


def test_checkpointed_capture_keeps_forward_and_active_artifact_contract() -> None:
    torch.manual_seed(1234)
    legacy = _make_stack("legacy_full_capture").eval()
    checkpointed = _make_stack("checkpointed_active_capture").eval()
    with torch.no_grad():
        checkpointed.phase.copy_(legacy.phase)
    field = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    with torch.no_grad():
        full_a, layers_a, _ = legacy(field, capture_full_layers=True)
        full_b, layers_b, _ = checkpointed(field, capture_full_layers=True)
        active_b, active_layers_b, diagnostics_b = checkpointed(
            field, capture_full_layers=False
        )
    assert torch.equal(full_a, full_b)
    assert torch.equal(layers_a, layers_b)
    assert full_b.shape[-2:] == (8, 8)
    assert layers_b.shape == (1, 2, 1, 8, 8)
    assert active_b.shape[-2:] == (8, 8)
    assert active_layers_b.shape == (1, 2, 1, 4, 4)
    assert diagnostics_b["transparent_full_layer_capture"] is False


def test_checkpointed_capture_backward_is_finite() -> None:
    stack = _make_stack("checkpointed_active_capture").train()
    field = torch.randn(1, 1, 4, 4, dtype=torch.complex64, requires_grad=True)
    output, _, _ = stack(field, capture_full_layers=False)
    output.abs().square().mean().backward()
    assert stack.phase.grad is not None
    assert torch.isfinite(stack.phase.grad).all()
