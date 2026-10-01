from __future__ import annotations

import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class _Plane:
    plane_id: str = "fiber_output"
    shape: tuple[int, int] = (1, 4)
    coordinate_frame: str = "fiber_output_frame"


@dataclass(frozen=True)
class _Provenance:
    operation: str = "lightgen_ols_coefficient_transmission_and_modal_reemission"
    upstream_plane: str | None = "ols"
    propagation_distance_m: float | None = None
    parameters: dict[str, object] | None = None


class _Field:
    def __init__(self, tensor: torch.Tensor, plane: _Plane, provenance: _Provenance) -> None:
        self.tensor = tensor
        self.plane = plane
        self.provenance = provenance


def _load_execution():
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "research" / "lightgen" / "execute_lightgen_2d_ideal_facet_relu_long_v1.py"
    spec = importlib.util.spec_from_file_location("ideal_facet_relu_long_test_execution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IdealFacetReluLongTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.execution = _load_execution()
        cls.provenance = _Provenance(parameters={"transport": "T=I_100_coefficient_space"})
        cls.plane = _Plane()

    def test_relu_formula_and_linear_identity(self) -> None:
        tensor = torch.tensor([[[[0.1 + 0.0j, 0.5 + 0.0j, 0.01 + 0.0j, 0.0 + 0.0j]]]], dtype=torch.complex64)
        field = _Field(tensor, self.plane, self.provenance)
        probe = self.execution.LongFacetProbe(mode="relu", tau=0.1, capture=True)
        output = probe.apply(field)
        expected_high = (0.5**2 - 0.1) + self.execution.MATERIAL_EPSILON
        self.assertAlmostEqual(float(output.tensor.abs().square()[0, 0, 0, 1]), expected_high, places=6)
        self.assertLess(float(output.tensor.abs().square()[0, 0, 0, 2]), 1.0e-10)
        self.assertEqual(output.provenance.operation, "long_fixed_ideal_facet_relu")
        identity = self.execution.LongFacetProbe(mode="linear", capture=True)
        identity_output = identity.apply(field)
        self.assertIs(identity_output, field)
        self.assertTrue(torch.equal(identity_output.tensor, tensor))
        self.assertFalse(identity.placements[0]["material_applied"])

    def test_sampler_balanced_epoch_and_resume(self) -> None:
        indices = torch.arange(30, dtype=torch.long)
        labels = torch.arange(30, dtype=torch.long) % self.execution.CLASS_COUNT
        sampler = self.execution.BalancedMultiEpochSampler(indices, labels, seed=17, steps=8)
        batches = [sampler.next_batch() for _ in range(4)]
        self.assertEqual(sampler.epoch, 1)
        self.assertEqual(sampler.position, 1)
        state = sampler.state_dict()
        resumed = self.execution.BalancedMultiEpochSampler(indices, labels, seed=17, steps=8)
        resumed.load_state_dict(state)
        for _ in range(4):
            left = sampler.next_batch()
            right = resumed.next_batch()
            self.assertTrue(torch.equal(left[0], right[0]))
            self.assertTrue(torch.equal(left[1], right[1]))
        self.assertEqual(sampler.cursor, resumed.cursor)
        self.assertEqual(sampler.cursor, 8)
        self.assertTrue(all(torch.equal(labels.index_select(0, batch_indices), batch_labels) for batch_indices, batch_labels in batches))

    def test_schedule_and_cache_position_are_frozen(self) -> None:
        lr0, stage0 = self.execution._schedule_lr(0, 3000)
        lr_switch, stage_switch = self.execution._schedule_lr(2400, 3000)
        lr_end, stage_end = self.execution._schedule_lr(3000, 3000)
        self.assertEqual((lr0, stage0), (self.execution.INITIAL_LR, "adapter_decoder_coadaptation"))
        self.assertAlmostEqual(lr_switch, self.execution.MID_LR, places=15)
        self.assertEqual(stage_switch, "adapter_decoder_coadaptation")
        self.assertAlmostEqual(lr_end, self.execution.FINAL_LR, places=15)
        self.assertEqual(stage_end, "decoder_only_refinement")
        role = torch.tensor([9, 3, 7], dtype=torch.long)
        self.assertTrue(torch.equal(self.execution._cache_positions(self.execution._train_position_map(role), torch.tensor([7, 9], dtype=torch.long)), torch.tensor([2, 0], dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
