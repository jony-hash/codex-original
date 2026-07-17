from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.configuration.schema import D2NNConfig
from ldm_project.configuration.schema import OneShotConfig, PhysicsConfig
from ldm_project.student.formal_contract import validate_formal_student_config
from ldm_project.student.experimental.transparent_phase import (
    TransparentIdentityDiffractiveStack,
)
from tools.audit_v68_lightgen_contract import audit_config


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
