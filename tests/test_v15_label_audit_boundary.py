from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


def _load_tool_module():
    path = Path("tools/diagnose_v15_label_audit_boundary.py")
    spec = importlib.util.spec_from_file_location("diagnose_v15_label_audit_boundary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V15LabelAuditBoundaryTests(unittest.TestCase):
    def test_class_average_audit_reports_top_labels_and_pair_margin(self) -> None:
        tool = _load_tool_module()
        prototypes = torch.zeros(3, 1, 4, 4)
        prototypes[0, :, :2, :2] = 1.0
        prototypes[1, :, 2:, 2:] = 1.0
        prototypes[2, :, :2, 2:] = 1.0
        samples = torch.stack([prototypes[0], prototypes[1]], dim=0)
        labels = torch.tensor([0, 1], dtype=torch.long)

        result = tool.class_average_audit(samples, labels, prototypes, class_labels=(0, 1, 2))
        summary = tool.per_class_summary(result, labels, class_labels=(0, 1), pairs=((0, 1),))

        self.assertEqual(result.nearest_labels.tolist(), [0, 1])
        self.assertEqual(result.top_labels[0][0], 0)
        self.assertAlmostEqual(result.correct_ratio, 1.0)
        self.assertGreater(summary["pairs"]["0->1"]["target_minus_negative_mean"], 0.0)
        self.assertTrue(summary["pairs"]["0->1"]["fixed_by_this_audit"])

    def test_multi_prototype_vote_can_correct_class_average_boundary(self) -> None:
        tool = _load_tool_module()
        samples = torch.zeros(1, 1, 4, 4)
        samples[:, :, :2, :2] = 1.0
        labels = torch.tensor([3], dtype=torch.long)
        prototype_samples = torch.zeros(4, 1, 4, 4)
        prototype_samples[0, :, :2, :2] = 1.0
        prototype_samples[1, :, :2, :2] = 0.95
        prototype_samples[2, :, 2:, 2:] = 1.0
        prototype_samples[3, :, 2:, 2:] = 0.95
        prototype_labels = torch.tensor([3, 3, 5, 5], dtype=torch.long)

        result = tool.multi_prototype_vote_audit(
            samples,
            labels,
            prototype_samples,
            prototype_labels,
            class_labels=(3, 5),
            neighbors=3,
        )
        summary = tool.per_class_summary(result, labels, class_labels=(3, 5), pairs=((3, 5),))

        self.assertEqual(result.nearest_labels.tolist(), [3])
        self.assertAlmostEqual(result.correct_ratio, 1.0)
        self.assertTrue(summary["pairs"]["3->5"]["fixed_by_this_audit"])

    def test_edge_map_and_center_crop_preserve_batch_shapes(self) -> None:
        tool = _load_tool_module()
        tensor = torch.zeros(2, 1, 10, 8)
        tensor[:, :, 3:7, 2:6] = 1.0

        edge = tool.edge_map(tensor)
        cropped = tool.center_crop(tensor, crop_fraction=0.5)

        self.assertEqual(tuple(edge.shape), tuple(tensor.shape))
        self.assertEqual(tuple(cropped.shape), (2, 1, 5, 4))
        self.assertGreater(float(edge.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
