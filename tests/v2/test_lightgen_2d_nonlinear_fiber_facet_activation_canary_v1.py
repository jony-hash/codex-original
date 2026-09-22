from __future__ import annotations

import importlib.util
from pathlib import Path
from dataclasses import dataclass
import unittest

import torch


@dataclass(frozen=True)
class _FakeProvenance:
    operation: str
    upstream_plane: str | None
    propagation_distance_m: float | None
    parameters: dict[str, object]


class _FakeField:
    def __init__(self, tensor: torch.Tensor, plane: object, provenance: _FakeProvenance) -> None:
        self.tensor = tensor
        self.plane = plane
        self.provenance = provenance


class _FakePlane:
    plane_id = "fiber_output"
    shape = (2, 2)
    coordinate_frame = "fiber_output_frame"


def _load_execution():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "research" / "execute_lightgen_2d_nonlinear_fiber_facet_activation_canary_v1.py"
    spec = importlib.util.spec_from_file_location("n6_canary_test_execution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NonlinearFiberFacetCanaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.n6 = _load_execution()
        cls.plane = _FakePlane()
        cls.provenance = _FakeProvenance(
            operation="lightgen_ols_coefficient_transmission_and_modal_reemission",
            upstream_plane="ols",
            propagation_distance_m=None,
            parameters={"transport": "T=I_100_coefficient_space"},
        )

    def test_fixed_surrogate_is_non_identity_and_preserves_facet_geometry(self) -> None:
        field = _FakeField(torch.tensor([[[[0.1, 0.2], [0.01, 0.0]]]], dtype=torch.complex64), self.plane, self.provenance)
        probe = self.n6.FacetMaterialProbe(nonlinear=True, capture=True)
        output = probe.apply(field)
        self.assertEqual(tuple(output.tensor.shape[-2:]), (2, 2))
        self.assertEqual(output.plane, field.plane)
        self.assertEqual(output.provenance.operation, "n6_fixed_nonlinear_fiber_facet_activation")
        self.assertTrue(torch.isfinite(output.tensor.real).all())
        self.assertTrue(torch.isfinite(output.tensor.imag).all())
        self.assertLess(float(output.tensor.abs()[0, 0, 0, 0]), float(field.tensor.abs()[0, 0, 0, 0]))
        self.assertEqual(len(probe.records), 1)
        self.assertFalse(probe.placements[0]["crop"])
        self.assertFalse(probe.placements[0]["pad"])
        self.assertFalse(probe.placements[0]["resample"])
        self.assertFalse(probe.placements[0]["new_diffraction_plane"])

    def test_linear_control_is_exact_identity(self) -> None:
        tensor = torch.tensor([[[[0.1 + 0.2j, 0.01 - 0.3j]]]], dtype=torch.complex64)
        field = _FakeField(tensor, self.plane, self.provenance)
        probe = self.n6.FacetMaterialProbe(nonlinear=False, capture=True)
        output = probe.apply(field)
        self.assertIs(output, field)
        self.assertTrue(torch.equal(output.tensor, tensor))
        self.assertFalse(probe.placements[0]["material_applied"])

    def test_surrogate_gradient_is_finite_on_suppressed_branch(self) -> None:
        tensor = torch.tensor([[[[0.01 + 0.0j, 0.2 + 0.0j]]]], dtype=torch.complex64, requires_grad=True)
        field = _FakeField(tensor, self.plane, self.provenance)
        probe = self.n6.FacetMaterialProbe(nonlinear=True)
        output = probe.apply(field)
        gradient = torch.autograd.grad(output.tensor.abs().square().sum(), tensor)[0]
        self.assertTrue(torch.isfinite(gradient.real).all())
        self.assertTrue(torch.isfinite(gradient.imag).all())

    def test_contract_freezes_n5_closure_and_stage_a_firewall(self) -> None:
        self.assertTrue(self.n6.N5_GLOBAL_OFFSET == 600)
        self.assertTrue(self.n6.N6_GLOBAL_TERMINAL == 900)
        self.assertFalse(self.n6.CONTRACT_PATH.is_absolute())
        self.assertEqual(self.n6.STAGE_A_FIREWALL, 0.937)
        self.assertEqual(self.n6.ADAPTER_RANK, 9)


if __name__ == "__main__":
    unittest.main()
