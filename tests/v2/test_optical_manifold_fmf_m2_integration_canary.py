from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import (
    optical_manifold_fmf_m2_integration_canary as canary,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from ldm_project.v2.optics.few_mode_fiber_operator import (
    FewModeFiberArrayOLS,
    FewModeFiberArraySpec,
)


def _plane() -> PlaneSpec:
    return PlaneSpec(
        "fiber_input", (100, 100), (2.0e-6, 2.0e-6), 532.0e-9, 0.0, "encoder"
    )


def test_nested_fmf_preserves_smf_lp01_and_has_200_modes() -> None:
    plane = _plane()
    smf = FiberArrayOLS(plane, FiberArraySpec())
    fmf = FewModeFiberArrayOLS(plane, FewModeFiberArraySpec())
    assert fmf.mode_names == ("LP01", "LP11a")
    assert fmf.input_modes.shape == (200, 100, 100)
    assert torch.equal(fmf.input_modes[0::2], smf.input_modes)
    assert torch.equal(fmf.output_modes[0::2], smf.output_modes)


def test_fmf_bank_is_finite_orthonormal_and_full_rank() -> None:
    fmf = FewModeFiberArrayOLS(_plane())
    area = 4.0e-12
    gram = fmf.input_modes.flatten(1).conj() @ fmf.input_modes.flatten(1).T * area
    assert torch.isfinite(gram).all()
    assert float((gram - torch.eye(200, dtype=gram.dtype)).abs().max()) <= 2.0e-4
    assert int(torch.linalg.matrix_rank(gram)) == 200


def test_fmf_forward_exposes_200_coefficients_and_mode_power() -> None:
    fmf = FewModeFiberArrayOLS(_plane())
    field = ComplexField(
        torch.randn(2, 1, 100, 100, dtype=torch.complex64),
        _plane(),
        OperationProvenance("fixture", None, None),
    )
    output = fmf(field)
    assert output.tensor.shape == (2, 1, 100, 100)
    assert (
        output.provenance.operation
        == "synthetic_fmf_nested_lp01_lp11a_identity_transport"
    )
    assert fmf.last_diagnostics["port_input_coefficients"].shape == (2, 200)
    assert fmf.last_diagnostics["mode_coefficients"].shape == (2, 100, 2)
    assert torch.isfinite(fmf.last_diagnostics["mode_power"]).all()


def test_fmf_zero_transport_is_finite_zero_and_passive() -> None:
    fmf = FewModeFiberArrayOLS(
        _plane(), transport=torch.zeros(200, 200, dtype=torch.complex64)
    )
    field = ComplexField(
        torch.ones(1, 1, 100, 100, dtype=torch.complex64),
        _plane(),
        OperationProvenance("fixture", None, None),
    )
    output = fmf(field)
    assert torch.isfinite(output.tensor).all()
    assert float(output.tensor.abs().max()) == 0.0
    assert float(fmf.last_diagnostics["output_energy"].max()) == 0.0


def test_fmf_rejects_nonpassive_or_unsupported_contract() -> None:
    with pytest.raises(ValueError, match="non-passive"):
        FewModeFiberArrayOLS(
            _plane(), transport=torch.eye(200, dtype=torch.complex64) * 1.01
        )
    with pytest.raises(ValueError, match="supported"):
        FewModeFiberArrayOLS(_plane(), FewModeFiberArraySpec(modes=("LP01", "LP11b")))


def test_fmf_config_and_review_are_exactly_registered() -> None:
    root = Path(__file__).parents[2]
    spec = canary._load_spec(
        root
        / "configs/research/lightgen_optical_manifold_fmf_m2_integration_canary_v1.yaml"
    )
    review, _ = canary._load_review(
        root
        / "configs/research/lightgen_optical_manifold_fmf_m2_integration_canary_review_v1.json",
        spec,
    )
    assert spec.payload["optical_contract"]["m2_modes"] == ["LP01", "LP11a"]
    assert review["approved_contract"]["m2_transport"] == "identity_200"


def test_comparison_panel_materializes_only_the_fixed_100_image_panel(
    tmp_path: Path,
) -> None:
    visual = torch.arange(100)
    data = canary.CanaryData(
        images=torch.zeros(1000, 1, 28, 28),
        labels=torch.arange(1000) % 10,
        train_indices=torch.arange(1000),
        heldout_indices=torch.arange(1000),
        schedule=torch.zeros(20, 10, dtype=torch.long),
        visual_indices=visual,
        manifest={},
    )
    full_camera = torch.zeros(1000, 1, 20, 20)
    full_target = torch.zeros(1000, 1, 20, 20)
    full_features = torch.zeros(1000, 100, dtype=torch.complex64)
    values = {"camera": full_camera, "target": full_target, "features": full_features}
    path = tmp_path / "panel.pt"
    canary._save_panel(path, data, {}, values, values, values, values, {})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert all(
        payload["tensors"][name].shape[0] == 100
        for name in ("target_200", "smf_m1_initial", "smf_m1_features", "fmf_m2_final")
    )
    assert path.stat().st_size < 3_000_000


def test_fmf_cli_dispatch_is_lazy() -> None:
    root = Path(__file__).parents[2]
    source = (root / "src/ldm_project/v2/application/cli.py").read_text()
    router = (
        root / "src/ldm_project/v2/application/_cli_research_optical_manifold.py"
    ).read_text()
    assert "_cli_research_optical_manifold" in source
    assert "optical_manifold_fmf_m2_integration_canary" in router
    assert "importlib.import_module" in source
