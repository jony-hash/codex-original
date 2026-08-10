# ruff: noqa: F401,F403,F405,F811
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from tests.tests_support import (
    _assert_nonblank_phase_columns_are_labeled,
    _cuda_device,
    _dummy_one_shot_outputs,
)
@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for one-shot tests")
@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for one-shot tests")
class TestStudentDataAndTargets(unittest.TestCase):
        def _minmax_norm(tensor):
            mn = tensor.amin(dim=(-2, -1), keepdim=True)
            mx = tensor.amax(dim=(-2, -1), keepdim=True)
            return (tensor - mn) / (mx - mn).clamp_min(1e-6)

        def test_distillation_train_subset_reduces_payload_deterministically(self) -> None:
            import torch

            from ldm_project.config import (
                DataConfig,
                DistillationConfig,
                FirstStageConfig,
                OneShotConfig,
                TrainingConfig,
            )
            from ldm_project.one_shot_distill_data import _subset_distill_payload_for_training

            payload = {
                "codes": torch.arange(30, dtype=torch.float32).reshape(10, 3),
                "class_labels": torch.arange(10, dtype=torch.long) % 2,
                "teacher_camera": torch.arange(10 * 16, dtype=torch.float32).reshape(10, 1, 4, 4),
                "teacher_initial_noise": torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1),
                "config": {"source": "unit-test"},
            }
            train_cfg = TrainingConfig(
                data=DataConfig(subset_seed=123, validation_fraction=0.2),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=3, num_classes=2),
                distillation=DistillationConfig(train_subset_size=4),
            )

            subset, source_size, effective_size = _subset_distill_payload_for_training(torch, payload, train_cfg)
            expected_indices = torch.randperm(10, generator=torch.Generator().manual_seed(123))[:4]

            self.assertEqual(source_size, 10)
            self.assertEqual(effective_size, 4)
            self.assertTrue(torch.equal(subset["codes"], payload["codes"].index_select(0, expected_indices)))
            self.assertTrue(
                torch.equal(
                    subset["teacher_camera"],
                    payload["teacher_camera"].index_select(0, expected_indices),
                )
            )
            self.assertEqual(subset["config"], payload["config"])

        def test_distillation_train_subset_rejects_invalid_sizes(self) -> None:
            import torch

            from ldm_project.config import DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _subset_distill_payload_for_training

            payload = {
                "codes": torch.randn(3, 2),
                "class_labels": torch.tensor([0, 1, 0]),
                "teacher_camera": torch.randn(3, 1, 4, 4),
            }
            for invalid_size in (0, 4):
                train_cfg = TrainingConfig(
                    first_stage=FirstStageConfig(image_size=4),
                    one_shot=OneShotConfig(code_dim=2, num_classes=2),
                    distillation=DistillationConfig(train_subset_size=invalid_size),
                )
                with self.assertRaisesRegex(ValueError, "distillation.train_subset_size"):
                    _subset_distill_payload_for_training(torch, payload, train_cfg)

        def test_distill_target_source_defaults_to_teacher_camera(self) -> None:
            import torch

            from ldm_project.config import FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _resolve_distill_targets

            teacher_camera = torch.rand(3, 1, 4, 4)
            payload = {
                "codes": torch.randn(3, 2),
                "class_labels": torch.tensor([0, 1, 0]),
                "teacher_camera": teacher_camera,
            }
            train_cfg = TrainingConfig(
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=2, num_classes=2),
            )

            targets = _resolve_distill_targets(torch, payload, train_cfg)

            self.assertTrue(torch.equal(targets, teacher_camera))

        def test_distill_target_source_mnist_prototype_indexes_normalized_labels(self) -> None:
            import torch

            from ldm_project.config import FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _resolve_distill_targets

            payload = {
                "codes": torch.randn(4, 2),
                "class_labels": torch.tensor([1, 0, 1, 0]),
                "teacher_camera": torch.zeros(4, 1, 4, 4),
            }
            prototypes = torch.stack(
                [
                    torch.linspace(-1.0, 1.0, steps=16).reshape(1, 4, 4),
                    torch.linspace(2.0, 6.0, steps=16).reshape(1, 4, 4).flip(-1),
                ],
                dim=0,
            )
            train_cfg = TrainingConfig(
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=2, num_classes=2, distill_target_source="mnist_prototype"),
            )

            targets = _resolve_distill_targets(torch, payload, train_cfg, mnist_prototypes=prototypes)

            self.assertEqual(tuple(targets.shape), (4, 1, 4, 4))
            self.assertGreaterEqual(float(targets.min().item()), 0.0)
            self.assertLessEqual(float(targets.max().item()), 1.0)
            self.assertTrue(torch.equal(targets[0], targets[2]))
            self.assertTrue(torch.equal(targets[1], targets[3]))
            self.assertFalse(torch.equal(targets[0], targets[1]))

        def test_distill_loader_uses_resolved_mnist_prototype_targets(self) -> None:
            import torch

            from ldm_project.config import DataConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _build_distill_loaders

            labels = torch.tensor([0, 1], dtype=torch.long)
            payload = {
                "codes": torch.randn(2, 2),
                "class_labels": labels,
                "teacher_camera": torch.zeros(2, 1, 4, 4),
            }
            prototypes = torch.zeros(2, 1, 4, 4)
            prototypes[0, :, :2, :] = 1.0
            prototypes[1, :, :, :2] = 1.0
            train_cfg = TrainingConfig(
                batch_size=2,
                data=DataConfig(validation_fraction=0.0),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=2, num_classes=2, distill_target_source="mnist_prototype"),
            )

            train_loader, _, _, _, _ = _build_distill_loaders(
                torch,
                payload,
                train_cfg,
                use_pin_memory=False,
                mnist_prototypes=prototypes,
            )
            _, batch_labels, batch_targets = next(iter(train_loader))

            for index, label in enumerate(batch_labels.tolist()):
                self.assertTrue(torch.equal(batch_targets[index], prototypes[int(label)]))

        def test_distill_loader_keeps_aux_detail_target_separate_from_main_target(self) -> None:
            import torch

            from ldm_project.config import DataConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _build_distill_loaders
            from ldm_project.one_shot_losses import _build_clean_readable_target

            labels = torch.tensor([0, 1], dtype=torch.long)
            teacher_camera = torch.zeros(2, 1, 4, 4)
            teacher_camera[:, :, 1:3, 1:3] = 1.0
            payload = {
                "codes": torch.randn(2, 2),
                "class_labels": labels,
                "teacher_camera": teacher_camera,
            }
            prototypes = torch.zeros(2, 1, 4, 4)
            prototypes[0, :, :2, :] = 1.0
            prototypes[1, :, :, :2] = 1.0
            train_cfg = TrainingConfig(
                batch_size=2,
                data=DataConfig(validation_fraction=0.0),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(
                    code_dim=2,
                    num_classes=2,
                    distill_target_source="mnist_prototype",
                    aux_detail_target_source="clean_teacher_camera",
                ),
            )

            train_loader, _, _, _, _ = _build_distill_loaders(
                torch,
                payload,
                train_cfg,
                use_pin_memory=False,
                mnist_prototypes=prototypes,
            )
            _, batch_labels, batch_targets, batch_aux_targets = next(iter(train_loader))
            expected_aux = _build_clean_readable_target(teacher_camera)["clean_target"]

            for index, label in enumerate(batch_labels.tolist()):
                self.assertTrue(torch.equal(batch_targets[index], prototypes[int(label)]))
            self.assertTrue(torch.allclose(batch_aux_targets, expected_aux))

        def test_distill_target_source_mnist_sample_is_deterministic_and_per_sample(self) -> None:
            import sys
            import types
            from unittest.mock import patch

            import torch

            from ldm_project.config import DataConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _resolve_distill_targets

            class FakeMNIST:
                def __init__(self, root, train, download, transform):
                    del root, train, download
                    self.transform = transform
                    self.targets = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
                    self.images = [
                        torch.full((1, 4, 4), 0.10),
                        torch.full((1, 4, 4), 0.20),
                        torch.full((1, 4, 4), 0.30),
                        torch.full((1, 4, 4), 0.70),
                        torch.full((1, 4, 4), 0.80),
                        torch.full((1, 4, 4), 0.90),
                    ]

                def __iter__(self):
                    for image, label in zip(self.images, self.targets.tolist(), strict=True):
                        yield self.transform(image), label

            class IdentityTransform:
                def __call__(self, image):
                    return image

            fake_torchvision = types.ModuleType("torchvision")
            fake_datasets = types.SimpleNamespace(MNIST=FakeMNIST)
            fake_transforms = types.SimpleNamespace(
                Compose=lambda transforms: IdentityTransform(),
                Resize=lambda size: IdentityTransform(),
                ToTensor=lambda: IdentityTransform(),
            )
            fake_torchvision.datasets = fake_datasets
            fake_torchvision.transforms = fake_transforms

            payload = {
                "codes": torch.randn(4, 2),
                "class_labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                "teacher_camera": torch.zeros(4, 1, 4, 4),
                "readability_filter_indices": torch.tensor([0, 1, 3, 4], dtype=torch.long),
            }
            train_cfg = TrainingConfig(
                data=DataConfig(root="unused"),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=2, num_classes=2, distill_target_source="mnist_sample"),
            )

            with patch.dict(
                sys.modules,
                {
                    "torchvision": fake_torchvision,
                    "torchvision.datasets": fake_datasets,
                    "torchvision.transforms": fake_transforms,
                },
            ):
                targets_a = _resolve_distill_targets(torch, payload, train_cfg)
                targets_b = _resolve_distill_targets(torch, payload, train_cfg)

            self.assertEqual(tuple(targets_a.shape), (4, 1, 4, 4))
            self.assertGreaterEqual(float(targets_a.min().item()), 0.0)
            self.assertLessEqual(float(targets_a.max().item()), 1.0)
            self.assertTrue(torch.equal(targets_a, targets_b))
            self.assertFalse(torch.equal(targets_a[0], targets_a[1]))
            self.assertFalse(torch.equal(targets_a[2], targets_a[3]))
            self.assertAlmostEqual(float(targets_a[0].mean().item()), 0.10, places=6)
            self.assertAlmostEqual(float(targets_a[1].mean().item()), 0.20, places=6)
            self.assertAlmostEqual(float(targets_a[2].mean().item()), 0.70, places=6)
            self.assertAlmostEqual(float(targets_a[3].mean().item()), 0.80, places=6)

        def test_distill_loader_uses_resolved_mnist_sample_targets(self) -> None:
            import sys
            import types
            from unittest.mock import patch

            import torch

            from ldm_project.config import DataConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _build_distill_loaders

            class FakeMNIST:
                def __init__(self, root, train, download, transform):
                    del root, train, download
                    self.transform = transform
                    self.targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
                    self.images = [
                        torch.full((1, 4, 4), 0.15),
                        torch.full((1, 4, 4), 0.25),
                        torch.full((1, 4, 4), 0.75),
                        torch.full((1, 4, 4), 0.85),
                    ]

                def __iter__(self):
                    for image, label in zip(self.images, self.targets.tolist(), strict=True):
                        yield self.transform(image), label

            class IdentityTransform:
                def __call__(self, image):
                    return image

            fake_torchvision = types.ModuleType("torchvision")
            fake_datasets = types.SimpleNamespace(MNIST=FakeMNIST)
            fake_transforms = types.SimpleNamespace(
                Compose=lambda transforms: IdentityTransform(),
                Resize=lambda size: IdentityTransform(),
                ToTensor=lambda: IdentityTransform(),
            )
            fake_torchvision.datasets = fake_datasets
            fake_torchvision.transforms = fake_transforms

            payload = {
                "codes": torch.randn(4, 2),
                "class_labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
                "teacher_camera": torch.zeros(4, 1, 4, 4),
                "readability_filter_indices": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            }
            train_cfg = TrainingConfig(
                batch_size=4,
                data=DataConfig(root="unused", validation_fraction=0.0),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(code_dim=2, num_classes=2, distill_target_source="mnist_sample"),
            )

            with patch.dict(
                sys.modules,
                {
                    "torchvision": fake_torchvision,
                    "torchvision.datasets": fake_datasets,
                    "torchvision.transforms": fake_transforms,
                },
            ):
                train_loader, _, _, _, _ = _build_distill_loaders(
                    torch,
                    payload,
                    train_cfg,
                    use_pin_memory=False,
                )
                _, batch_labels, batch_targets = next(iter(train_loader))

            means_by_label: dict[int, list[float]] = {0: [], 1: []}
            for label, target in zip(batch_labels.tolist(), batch_targets, strict=True):
                means_by_label[int(label)].append(round(float(target.mean().item()), 6))
                self.assertFalse(torch.equal(target, torch.zeros_like(target)))
            self.assertEqual(sorted(means_by_label[0]), [0.15, 0.25])
            self.assertEqual(sorted(means_by_label[1]), [0.75, 0.85])

        def test_distill_target_source_mnist_nearest_teacher_image_selects_aligned_targets(self) -> None:
            import sys
            import types
            from unittest.mock import patch

            import torch

            from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _resolve_distill_targets

            horizontal = torch.zeros(1, 4, 4)
            horizontal[:, :2, :] = 1.0
            vertical = torch.zeros(1, 4, 4)
            vertical[:, :, :2] = 1.0
            diagonal = torch.eye(4).view(1, 4, 4)
            anti_diagonal = torch.flip(diagonal, dims=(-1,))

            class FakeMNIST:
                def __init__(self, root, train, download, transform):
                    del root, train, download
                    self.transform = transform
                    self.targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
                    self.images = [horizontal, vertical, diagonal, anti_diagonal]

                def __iter__(self):
                    for image, label in zip(self.images, self.targets.tolist(), strict=True):
                        yield self.transform(image), label

            class IdentityTransform:
                def __call__(self, image):
                    return image

            fake_torchvision = types.ModuleType("torchvision")
            fake_datasets = types.SimpleNamespace(MNIST=FakeMNIST)
            fake_transforms = types.SimpleNamespace(
                Compose=lambda transforms: IdentityTransform(),
                Resize=lambda size: IdentityTransform(),
                ToTensor=lambda: IdentityTransform(),
            )
            fake_torchvision.datasets = fake_datasets
            fake_torchvision.transforms = fake_transforms

            payload = {
                "codes": torch.randn(4, 2),
                "class_labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                "teacher_camera": torch.zeros(4, 1, 4, 4),
                "teacher_image": torch.stack([horizontal * 2.0 - 0.5, vertical * 3.0 - 1.0, diagonal, anti_diagonal], dim=0),
            }
            train_cfg = TrainingConfig(
                data=DataConfig(root="unused"),
                distillation=DistillationConfig(label_audit_prototypes_per_class=2),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(
                    code_dim=2,
                    num_classes=2,
                    distill_target_source="mnist_nearest_teacher_image",
                ),
            )

            with patch.dict(
                sys.modules,
                {
                    "torchvision": fake_torchvision,
                    "torchvision.datasets": fake_datasets,
                    "torchvision.transforms": fake_transforms,
                },
            ):
                targets_a = _resolve_distill_targets(torch, payload, train_cfg)
                targets_b = _resolve_distill_targets(torch, payload, train_cfg)

            self.assertEqual(tuple(targets_a.shape), (4, 1, 4, 4))
            self.assertGreaterEqual(float(targets_a.min().item()), 0.0)
            self.assertLessEqual(float(targets_a.max().item()), 1.0)
            self.assertTrue(torch.equal(targets_a, targets_b))
            self.assertTrue(torch.equal(targets_a[0], horizontal))
            self.assertTrue(torch.equal(targets_a[1], vertical))
            self.assertTrue(torch.equal(targets_a[2], diagonal))
            self.assertTrue(torch.equal(targets_a[3], anti_diagonal))

        def test_distill_target_source_mnist_nearest_teacher_image_requires_teacher_image(self) -> None:
            import torch

            from ldm_project.config import FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _resolve_distill_targets

            payload = {
                "codes": torch.randn(2, 2),
                "class_labels": torch.tensor([0, 1], dtype=torch.long),
                "teacher_camera": torch.zeros(2, 1, 4, 4),
            }
            train_cfg = TrainingConfig(
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(
                    code_dim=2,
                    num_classes=2,
                    distill_target_source="mnist_nearest_teacher_image",
                ),
            )

            with self.assertRaisesRegex(ValueError, "teacher_image"):
                _resolve_distill_targets(torch, payload, train_cfg)

        def test_distill_loader_uses_resolved_mnist_nearest_teacher_image_targets(self) -> None:
            import sys
            import types
            from unittest.mock import patch

            import torch

            from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _build_distill_loaders

            horizontal = torch.zeros(1, 4, 4)
            horizontal[:, :2, :] = 1.0
            vertical = torch.zeros(1, 4, 4)
            vertical[:, :, :2] = 1.0

            class FakeMNIST:
                def __init__(self, root, train, download, transform):
                    del root, train, download
                    self.transform = transform
                    self.targets = torch.tensor([0, 0], dtype=torch.long)
                    self.images = [horizontal, vertical]

                def __iter__(self):
                    for image, label in zip(self.images, self.targets.tolist(), strict=True):
                        yield self.transform(image), label

            class IdentityTransform:
                def __call__(self, image):
                    return image

            fake_torchvision = types.ModuleType("torchvision")
            fake_datasets = types.SimpleNamespace(MNIST=FakeMNIST)
            fake_transforms = types.SimpleNamespace(
                Compose=lambda transforms: IdentityTransform(),
                Resize=lambda size: IdentityTransform(),
                ToTensor=lambda: IdentityTransform(),
            )
            fake_torchvision.datasets = fake_datasets
            fake_torchvision.transforms = fake_transforms

            payload = {
                "codes": torch.randn(2, 2),
                "class_labels": torch.tensor([0, 0], dtype=torch.long),
                "teacher_camera": torch.zeros(2, 1, 4, 4),
                "teacher_image": torch.stack([horizontal, vertical], dim=0),
            }
            train_cfg = TrainingConfig(
                batch_size=2,
                data=DataConfig(root="unused", validation_fraction=0.0),
                distillation=DistillationConfig(label_audit_prototypes_per_class=2),
                first_stage=FirstStageConfig(image_size=4),
                one_shot=OneShotConfig(
                    code_dim=2,
                    num_classes=1,
                    distill_target_source="mnist_nearest_teacher_image",
                ),
            )

            with patch.dict(
                sys.modules,
                {
                    "torchvision": fake_torchvision,
                    "torchvision.datasets": fake_datasets,
                    "torchvision.transforms": fake_transforms,
                },
            ):
                train_loader, _, _, _, _ = _build_distill_loaders(
                    torch,
                    payload,
                    train_cfg,
                    use_pin_memory=False,
                )
                _, _, batch_targets = next(iter(train_loader))

            target_sums = sorted(float(target.sum().item()) for target in batch_targets)
            self.assertEqual(target_sums, [8.0, 8.0])
            self.assertFalse(torch.equal(batch_targets[0], batch_targets[1]))

        def test_paired_class_distill_loader_enables_diversity_pairs(self) -> None:
            import torch

            from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _build_distill_loaders
            from ldm_project.training import one_shot_loss

            payload = {
                "codes": torch.randn(4, 3),
                "class_labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                "teacher_camera": torch.tensor(
                    [
                        [[[0.0, 0.0], [0.0, 0.0]]],
                        [[[1.0, 0.0], [0.0, 0.0]]],
                        [[[0.0, 1.0], [0.0, 0.0]]],
                        [[[0.0, 0.0], [1.0, 0.0]]],
                    ],
                    dtype=torch.float32,
                ),
            }
            train_cfg = TrainingConfig(
                batch_size=1,
                data=DataConfig(validation_fraction=0.0, subset_seed=123),
                first_stage=FirstStageConfig(image_size=2),
                one_shot=OneShotConfig(code_dim=3, num_classes=2),
                distillation=DistillationConfig(paired_class_batches=True),
            )

            train_loader, _, _, _, _ = _build_distill_loaders(torch, payload, train_cfg, use_pin_memory=False)
            _, labels, targets = next(iter(train_loader))

            self.assertGreaterEqual(int(labels.shape[0]), 2)
            self.assertLess(int(labels.unique().numel()), int(labels.numel()))

            losses = one_shot_loss(
                {
                    "camera_plane": torch.zeros_like(targets),
                    "slm_amplitude": torch.ones_like(targets),
                },
                targets,
                class_labels=labels,
                code_diversity_weight=1.0,
                sample_code_diversity_weight=0.5,
                teacher_pairwise_diversity_weight=1.0,
                diversity_min_teacher_mse=1.0e-8,
            )
            self.assertGreater(float(losses["diversity_pair_count"].item()), 0.0)
            self.assertGreater(float(losses["code_diversity_loss"].item()), 0.0)
            self.assertGreater(float(losses["sample_code_diversity_loss"].item()), 0.0)
            self.assertGreater(float(losses["teacher_pairwise_mse"].item()), 0.0)

        def test_multiclass_distill_loader_enables_cross_class_pairs(self) -> None:
            import torch

            from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_data import _build_distill_loaders
            from ldm_project.training import one_shot_loss

            labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
            target = torch.zeros(6, 1, 2, 2, dtype=torch.float32)
            for index in range(6):
                target[index, 0, index // 2 % 2, index % 2] = 1.0
            payload = {
                "codes": torch.randn(6, 3),
                "class_labels": labels,
                "teacher_camera": target,
            }
            train_cfg = TrainingConfig(
                batch_size=3,
                data=DataConfig(validation_fraction=0.0, subset_seed=321),
                first_stage=FirstStageConfig(image_size=2),
                one_shot=OneShotConfig(code_dim=3, num_classes=3),
                distillation=DistillationConfig(multiclass_batches=True),
            )

            train_loader, _, _, _, _ = _build_distill_loaders(torch, payload, train_cfg, use_pin_memory=False)
            first_batch = next(iter(train_loader))
            second_loader, _, _, _, _ = _build_distill_loaders(torch, payload, train_cfg, use_pin_memory=False)
            second_first_batch = next(iter(second_loader))
            _, batch_labels, batch_targets = first_batch

            self.assertEqual(batch_labels.tolist(), second_first_batch[1].tolist())
            self.assertGreaterEqual(int(batch_labels.unique().numel()), 2)

            collapsed = torch.full_like(batch_targets, 0.5, requires_grad=True)
            losses = one_shot_loss(
                {
                    "camera_plane": collapsed,
                    "slm_amplitude": torch.rand_like(batch_targets),
                },
                batch_targets,
                class_labels=batch_labels,
                cross_class_pairwise_weight=0.1,
                diversity_min_teacher_mse=1.0e-8,
            )

            self.assertGreater(float(losses["cross_class_pair_count"].item()), 0.0)
            self.assertGreater(float(losses["cross_class_pairwise_loss"].item()), 0.0)
            self.assertGreater(float(losses["cross_class_teacher_pairwise_mse"].item()), 0.0)
            self.assertEqual(float(losses["cross_class_student_pairwise_mse"].item()), 0.0)
            self.assertEqual(float(losses["cross_class_diversity_ratio"].item()), 0.0)
            losses["loss"].backward()
            self.assertIsNotNone(collapsed.grad)

        def test_one_shot_distill_warm_start_latest_and_crop_history(self) -> None:
            import torch

            from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_training import run_one_shot_distill_training
            from ldm_project.one_shot_training_utils import _one_shot_d2nn_crop_retention_metrics
            from ldm_project.stage_constants import ONE_SHOT_DISTILL_LATEST_CHECKPOINT
            from ldm_project.training import one_shot_loss

            class DummyOneShot(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.scale = torch.nn.Parameter(torch.tensor(0.0))

                def forward(self, codes, labels, *, return_optical_fields=False):
                    outputs = _dummy_one_shot_outputs(batch_size=int(codes.shape[0]), image_size=4, grid_size=2)
                    for key in (
                        "slm_phase",
                        "encoder_input_phase",
                        "encoder_input_grid_phase",
                        "encoded_phase",
                        "ols_input_phase",
                        "ols_output_phase",
                        "ols_phase",
                        "decoder_input_phase",
                        "decoder_output_phase",
                        "chip_phase",
                        "camera_phase",
                        "diffractive_encoder_layer_amplitude",
                        "diffractive_encoder_layer_phase",
                        "diffractive_decoder_layer_amplitude",
                        "diffractive_decoder_layer_phase",
                        "diffractive_encoder_mask_amplitude",
                        "diffractive_encoder_mask_phase",
                        "diffractive_decoder_mask_amplitude",
                        "diffractive_decoder_mask_phase",
                    ):
                        outputs.pop(key, None)
                    outputs["camera_plane"] = torch.sigmoid(self.scale) * outputs["camera_plane"]
                    if return_optical_fields:
                        outputs["optical_fields"] = {}
                    return outputs

            class RecordingRunManager:
                def __init__(self, run_dir: Path) -> None:
                    self.run_dir = run_dir
                    self.images_dir = run_dir / "images"
                    self.artifacts_dir = run_dir / "artifacts"
                    self.configs_dir = run_dir / "configs"
                    self.events = []
                    self.history = {}
                    self.meta_payload = {}
                    self.checkpoints = []

                def save_config_snapshot(self, name, data):
                    return self.configs_dir / f"{name}.json"

                def append_history(self, name, payload):
                    self.history.setdefault(name, []).append(dict(payload))

                def save_tensor_grid(self, tensor_map, destination, title="", max_items=4, **kwargs):
                    self.events.append(("grid", Path(destination).name))
                    return Path(destination)

                def save_artifact_json(self, name, payload):
                    self.events.append(("artifact", name))
                    return self.artifacts_dir / f"{name}.json"

                def save_tensor_bundle(self, name, payload):
                    self.events.append(("bundle", name))
                    return self.artifacts_dir / name

                def save_checkpoint(self, name, payload):
                    path = self.run_dir / "checkpoints" / name
                    record = (name, dict(payload), path)
                    self.events.append(("checkpoint", name))
                    self.checkpoints.append(record)
                    return path

                def write_meta(self, meta):
                    self.meta_payload = dict(meta)

            with tempfile.TemporaryDirectory() as tmpdir:
                run_dir = Path(tmpdir)
                distill_path = run_dir / "distill.pt"
                warm_start_path = run_dir / "one_shot.pt"
                labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
                torch.save(
                    {
                        "codes": torch.randn(4, 3),
                        "class_labels": labels,
                        "teacher_camera": torch.rand(4, 1, 4, 4),
                    },
                    distill_path,
                )
                warm_start_payload = {"model_state_dict": {"sentinel": torch.tensor(1.0)}}
                torch.save(warm_start_payload, warm_start_path)
                train_cfg = TrainingConfig(
                    batch_size=2,
                    step_log_interval=1,
                    visual_save_epoch_interval=5,
                    validation_epoch_interval=1,
                    one_shot_epochs=2,
                    data=DataConfig(validation_fraction=0.5, validation_split_seed=11),
                    first_stage=FirstStageConfig(image_size=4),
                    one_shot=OneShotConfig(code_dim=3, num_classes=2),
                    distillation=DistillationConfig(),
                )
                run_manager = RecordingRunManager(run_dir)
                loaded_payloads = []

                def load_checkpoint(model, payload):
                    loaded_payloads.append(payload)

                run_one_shot_distill_training(
                    torch,
                    one_shot_model=DummyOneShot(),
                    one_shot_loss_fn=one_shot_loss,
                    run_manager=run_manager,
                    train_cfg=train_cfg,
                    run_meta={},
                    physics={"unit": "test"},
                    device=torch.device("cpu"),
                    use_pin_memory=False,
                    one_shot_model_version="unit-test",
                    distill_dataset=str(distill_path),
                    one_shot_checkpoint=str(warm_start_path),
                    load_one_shot_checkpoint_fn=load_checkpoint,
                )

                self.assertEqual(len(loaded_payloads), 1)
                self.assertEqual(loaded_payloads[0]["model_state_dict"]["sentinel"].item(), 1.0)
                latest_checkpoints = [
                    item for item in run_manager.checkpoints if item[0] == ONE_SHOT_DISTILL_LATEST_CHECKPOINT
                ]
                self.assertEqual(len(latest_checkpoints), 2)
                latest_payload = latest_checkpoints[-1][1]
                self.assertEqual(latest_payload["epoch"], 2)
                self.assertEqual(latest_payload["stage_step"], 2)
                self.assertEqual(latest_payload["total_steps"], 2)
                self.assertEqual(latest_payload["distill_dataset"], str(distill_path))
                self.assertEqual(run_manager.meta_payload["one_shot_distill_warm_start_checkpoint"], str(warm_start_path))
                self.assertEqual(run_manager.meta_payload["one_shot_distill_latest_checkpoint"], latest_checkpoints[-1][2])
                self.assertEqual(run_manager.meta_payload["one_shot_distill_latest_epoch"], 2)
                self.assertEqual(run_manager.meta_payload["one_shot_distill_latest_step"], 2)
                first_checkpoint_event = next(index for index, event in enumerate(run_manager.events) if event[0] == "checkpoint")
                first_grid_event = next(index for index, event in enumerate(run_manager.events) if event[0] == "grid")
                self.assertLess(first_checkpoint_event, first_grid_event)
                train_history = run_manager.history["one_shot_distill"]
                validation_history = run_manager.history["one_shot_distill_validation"]
                for payload in (train_history[0], validation_history[0]):
                    self.assertAlmostEqual(payload["crop_energy_retention_mean"], 0.25, places=6)
                    self.assertAlmostEqual(payload["encoder_crop_energy_retention_mean"], 0.25, places=6)
                    self.assertAlmostEqual(payload["decoder_crop_energy_retention_mean"], 0.25, places=6)
                    self.assertAlmostEqual(payload["encoder_l1_crop_energy_retention"], 0.25, places=6)
                    self.assertAlmostEqual(payload["decoder_l2_crop_energy_retention"], 0.25, places=6)
                    self.assertNotIn("encoder_crop_input_energy", payload)
                    self.assertNotIn("decoder_crop_cropped_energy", payload)
                self.assertEqual(_one_shot_d2nn_crop_retention_metrics({"camera_plane": torch.ones(1, 1, 2, 2)}), {})

            with self.assertRaisesRegex(ValueError, "warm-start"):
                run_one_shot_distill_training(
                    torch,
                    one_shot_model=DummyOneShot(),
                    one_shot_loss_fn=one_shot_loss,
                    run_manager=RecordingRunManager(Path(".")),
                    train_cfg=train_cfg,
                    run_meta={},
                    physics={},
                    device=torch.device("cpu"),
                    use_pin_memory=False,
                    one_shot_model_version="unit-test",
                    distill_dataset=str(distill_path),
                    one_shot_checkpoint=str(warm_start_path),
                    load_one_shot_checkpoint_fn=None,
                )

        def test_one_shot_distill_quality_selection_score(self) -> None:
            from ldm_project.one_shot_quality import (
                _one_shot_distill_balanced_quality_score,
                _one_shot_distill_quality_score,
                _one_shot_distill_strict_readability_score,
                _one_shot_distill_strict_balanced_quality_score,
                _resolve_one_shot_distill_selection,
            )
            from ldm_project.stage_constants import (
                ONE_SHOT_DISTILL_BALANCED_QUALITY_METRIC,
                ONE_SHOT_DISTILL_BEST_METRIC,
                ONE_SHOT_DISTILL_QUALITY_METRIC,
                ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_METRIC,
                ONE_SHOT_DISTILL_STRICT_READABILITY_METRIC,
            )

            one_shot = type("OneShotSelectionConfig", (), {"checkpoint_selection_metric": "quality"})()
            train_cfg = type("TrainingSelectionConfig", (), {"one_shot": one_shot})()
            metrics = {
                "loss": 0.5,
                "camera_recon_loss": 0.1,
                "camera_foreground_dice": 0.8,
                "camera_saturation_ratio": 0.4,
                "foreground_hole_ratio": 0.25,
                "background_speckle_ratio": 0.20,
                "background_hotspot_ratio": 0.35,
                "background_highfreq_energy": 0.004,
                "readable_contrast": 4.2,
                "slm_tv": 0.2,
                "diversity_ratio": 0.30,
                "student_label_correct_ratio": 0.75,
                "student_label_margin_mean": 0.01,
                "paired_dice": 0.80,
                "paired_corr": 0.85,
                "camera_high_saturation_ratio": 0.02,
            }

            score, metric = _resolve_one_shot_distill_selection(train_cfg, metrics)

            self.assertEqual(metric, ONE_SHOT_DISTILL_QUALITY_METRIC)
            self.assertAlmostEqual(score, 0.191, places=6)
            self.assertAlmostEqual(metrics["quality_score"], 0.191, places=6)
            self.assertAlmostEqual(_one_shot_distill_quality_score(metrics), 0.191, places=6)

            one_shot.checkpoint_selection_metric = "loss"
            score, metric = _resolve_one_shot_distill_selection(train_cfg, metrics)

            self.assertEqual(metric, ONE_SHOT_DISTILL_BEST_METRIC)
            self.assertAlmostEqual(score, 0.5, places=6)

            one_shot.checkpoint_selection_metric = "balanced_quality"
            score, metric = _resolve_one_shot_distill_selection(train_cfg, metrics)

            self.assertEqual(metric, ONE_SHOT_DISTILL_BALANCED_QUALITY_METRIC)
            self.assertAlmostEqual(score, 0.1575, places=6)
            self.assertAlmostEqual(metrics["balanced_quality_score"], 0.1575, places=6)
            self.assertAlmostEqual(_one_shot_distill_balanced_quality_score(metrics), 0.1575, places=6)

            one_shot.checkpoint_selection_metric = "strict_balanced_quality"
            score, metric = _resolve_one_shot_distill_selection(train_cfg, metrics)

            self.assertEqual(metric, ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_METRIC)
            self.assertAlmostEqual(score, 0.1695, places=6)
            self.assertAlmostEqual(metrics["strict_balanced_quality_score"], 0.1695, places=6)
            self.assertAlmostEqual(_one_shot_distill_strict_balanced_quality_score(metrics), 0.1695, places=6)

            one_shot.checkpoint_selection_metric = "strict_readability"
            score, metric = _resolve_one_shot_distill_selection(train_cfg, metrics)

            self.assertEqual(metric, ONE_SHOT_DISTILL_STRICT_READABILITY_METRIC)
            self.assertAlmostEqual(score, 0.1425, places=6)
            self.assertAlmostEqual(metrics["strict_readability_score"], 0.1425, places=6)
            self.assertAlmostEqual(_one_shot_distill_strict_readability_score(metrics), 0.1425, places=6)
            for required_metric in ("paired_corr", "paired_dice", "camera_high_saturation_ratio"):
                incomplete = dict(metrics)
                incomplete.pop(required_metric)
                with self.assertRaisesRegex(ValueError, required_metric):
                    _one_shot_distill_strict_readability_score(incomplete)

        def test_one_shot_saturation_split_not_penalizing_black_background(self) -> None:
            from ldm_project.training_loss_common import (
                _camera_low_saturation_ratio,
                _camera_high_saturation_ratio,
                _camera_saturation_ratio,
            )
            from ldm_project.one_shot_quality import _one_shot_distill_strict_readability_score
            import torch

            # Pure black background: all pixels at 0.0 → low saturation high, high saturation zero.
            black = torch.zeros(1, 1, 200, 200)
            self.assertAlmostEqual(float(_camera_low_saturation_ratio(black).item()), 1.0, places=6)
            self.assertAlmostEqual(float(_camera_high_saturation_ratio(black).item()), 0.0, places=6)
            self.assertAlmostEqual(float(_camera_saturation_ratio(black).item()), 1.0, places=6)

            # Pure white foreground: all pixels at 1.0 → high saturation high, low saturation zero.
            white = torch.ones(1, 1, 200, 200)
            self.assertAlmostEqual(float(_camera_low_saturation_ratio(white).item()), 0.0, places=6)
            self.assertAlmostEqual(float(_camera_high_saturation_ratio(white).item()), 1.0, places=6)

            # Strict readability score should NOT penalize black background (low sat only).
            base_metrics = {
                "camera_recon_loss": 0.1,
                "camera_foreground_dice": 0.8,
                "paired_dice": 0.80,
                "paired_corr": 0.85,
                "camera_saturation_ratio": 1.0,
                "camera_low_saturation_ratio": 1.0,
                "camera_high_saturation_ratio": 0.0,
                "foreground_hole_ratio": 0.06,
                "background_speckle_ratio": 0.07,
                "background_hotspot_ratio": 0.12,
                "background_highfreq_energy": 0.002,
                "readable_contrast": 4.5,
                "slm_tv": 0.01,
                "student_label_correct_ratio": 0.90,
                "student_label_margin_mean": 0.03,
            }
            score_black = _one_shot_distill_strict_readability_score(base_metrics)
            # Identical metrics but low_saturation=0 → same score because high_saturation=0 in both.
            low_sat_only = dict(base_metrics, camera_low_saturation_ratio=0.0)
            score_no_low_sat = _one_shot_distill_strict_readability_score(low_sat_only)
            self.assertAlmostEqual(score_black, score_no_low_sat, places=6,
                                   msg="strict_readability should not distinguish low saturation levels (black bg is valid).")

            # High saturation increases strict_readability score (worse).
            high_sat_metrics = dict(base_metrics, camera_high_saturation_ratio=0.50)
            score_high_sat = _one_shot_distill_strict_readability_score(high_sat_metrics)
            self.assertGreater(score_high_sat, score_black,
                               msg="high saturation should worsen (increase) strict_readability score.")

        def test_one_shot_strict_selection_prefers_better_paired_corr_dice(self) -> None:
            from ldm_project.one_shot_quality import _one_shot_distill_strict_readability_score

            base = {
                "camera_recon_loss": 0.1,
                "camera_foreground_dice": 0.8,
                "paired_dice": 0.72,
                "paired_corr": 0.78,
                "camera_saturation_ratio": 0.02,
                "camera_low_saturation_ratio": 0.01,
                "camera_high_saturation_ratio": 0.01,
                "foreground_hole_ratio": 0.06,
                "background_speckle_ratio": 0.07,
                "background_hotspot_ratio": 0.12,
                "background_highfreq_energy": 0.002,
                "readable_contrast": 4.5,
                "slm_tv": 0.01,
                "student_label_correct_ratio": 0.90,
                "student_label_margin_mean": 0.03,
            }
            score_ok = _one_shot_distill_strict_readability_score(base)
            # Worse paired_dice → higher score (worse).
            bad_dice = dict(base, paired_dice=0.50)
            score_bad_dice = _one_shot_distill_strict_readability_score(bad_dice)
            self.assertGreater(score_bad_dice, score_ok,
                               msg="lower paired_dice must worsen (increase) strict_readability score.")
            # Worse paired_corr → higher score (worse).
            bad_corr = dict(base, paired_corr=0.50)
            score_bad_corr = _one_shot_distill_strict_readability_score(bad_corr)
            self.assertGreater(score_bad_corr, score_ok,
                               msg="lower paired_corr must worsen (increase) strict_readability score.")

        def test_one_shot_foreground_dice_not_used_in_strict_selection(self) -> None:
            from ldm_project.one_shot_quality import _one_shot_distill_strict_readability_score

            base = {
                "camera_recon_loss": 0.1,
                "camera_foreground_dice": 0.50,  # poor, but should NOT affect strict score
                "paired_dice": 0.80,
                "paired_corr": 0.85,
                "camera_saturation_ratio": 0.02,
                "camera_low_saturation_ratio": 0.01,
                "camera_high_saturation_ratio": 0.01,
                "foreground_hole_ratio": 0.06,
                "background_speckle_ratio": 0.07,
                "background_hotspot_ratio": 0.12,
                "background_highfreq_energy": 0.002,
                "readable_contrast": 4.5,
                "slm_tv": 0.01,
                "student_label_correct_ratio": 0.90,
                "student_label_margin_mean": 0.03,
            }
            score_poor_fg_dice = _one_shot_distill_strict_readability_score(base)
            good_fg_dice = dict(base, camera_foreground_dice=0.95)
            score_good_fg_dice = _one_shot_distill_strict_readability_score(good_fg_dice)
            # camera_foreground_dice is no longer in the formula → scores must be identical.
            self.assertAlmostEqual(score_poor_fg_dice, score_good_fg_dice, places=6,
                                   msg="camera_foreground_dice should NOT affect strict_readability score.")

            diagnostics_changed = dict(
                base,
                student_label_correct_ratio=0.0,
                student_label_margin_mean=-1.0,
                foreground_hole_ratio=1.0,
                background_speckle_ratio=1.0,
                readable_contrast=0.0,
            )
            self.assertAlmostEqual(
                score_poor_fg_dice,
                _one_shot_distill_strict_readability_score(diagnostics_changed),
                places=6,
            )

        def test_one_shot_forward_exposes_encoder_diagnostics_on_cpu(self) -> None:
            import torch

            from ldm_project.config import OneShotConfig, PhysicsConfig
            from ldm_project.one_shot import OneShotOpticalGenerator

            config = OneShotConfig(
                code_dim=4,
                label_embed_dim=4,
                hidden_dim=8,
                num_classes=2,
                metasurface_grid_size=4,
            )
            model = OneShotOpticalGenerator(PhysicsConfig(), config, image_size=8)
            outputs = model(torch.randn(2, config.code_dim), torch.tensor([0, 1]))

            full_size_keys = (
                "encoder_input_amplitude",
                "encoder_input_phase",
                "chip_amplitude",
                "chip_phase",
            )
            grid_size_keys = (
                "encoder_input_grid_amplitude",
                "encoder_input_grid_phase",
                "encoded_amplitude",
                "encoded_phase",
                "ols_input_amplitude",
                "ols_input_phase",
                "ols_output_amplitude",
                "ols_output_phase",
                "ols_amplitude",
                "ols_phase",
                "decoder_input_amplitude",
                "decoder_input_phase",
                "decoder_output_amplitude",
                "decoder_output_phase",
            )
            for key in full_size_keys:
                self.assertEqual(tuple(outputs[key].shape), (2, 1, 8, 8))
                self.assertTrue(torch.isfinite(outputs[key]).all().item(), key)
            for key in grid_size_keys:
                self.assertEqual(tuple(outputs[key].shape), (2, 1, 4, 4))
                self.assertTrue(torch.isfinite(outputs[key]).all().item(), key)

        def test_teacher_distill_filter_preserves_bundle_and_balances_classes(self) -> None:
            import torch

            from ldm_project.config import (
                DataConfig,
                DiffusionConfig,
                DistillationConfig,
                FirstStageConfig,
                OneShotConfig,
                TrainingConfig,
            )
            from ldm_project.teacher_distill import run_teacher_distill_filter

            class RecordingRunManager:
                def __init__(self, run_dir: Path) -> None:
                    self.run_dir = run_dir
                    self.images_dir = run_dir / "images"
                    self.artifacts_dir = run_dir / "artifacts"
                    self.configs_dir = run_dir / "configs"
                    self.meta_payload = None
                    self.metrics_payload = None
                    self.bundle_path = None

                def write_meta(self, meta):
                    self.meta_payload = dict(meta)

                def save_config_snapshot(self, name, data):
                    path = self.configs_dir / f"{name}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    return path

                def save_tensor_bundle(self, name, payload):
                    path = self.artifacts_dir / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(payload), path)
                    self.bundle_path = path
                    return path

                def save_artifact_json(self, name, payload):
                    self.metrics_payload = payload
                    path = self.artifacts_dir / f"{name}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
                    return path

                def save_tensor_grid(self, tensor_map, destination, title="", max_items=4):
                    path = Path(destination)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    return path

            with tempfile.TemporaryDirectory() as tmpdir:
                run_dir = Path(tmpdir)
                source_path = run_dir / "source_distill.pt"
                torch.manual_seed(7)
                labels = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
                teacher_camera = torch.rand(20, 1, 8, 8, dtype=torch.float32) * 0.8 + 0.1
                payload = {
                    "codes": torch.randn(20, 4),
                    "class_labels": labels,
                    "teacher_camera": teacher_camera,
                    "teacher_image": teacher_camera.clone(),
                    "teacher_latents": torch.randn(20, 1, 1, 1),
                    "teacher_initial_noise": torch.randn(20, 1, 1, 1),
                    "guidance_scale": torch.tensor(2.0),
                    "latent_scale": torch.tensor(1.0),
                    "config": {"source": "unit-test"},
                }
                torch.save(payload, source_path)
                train_cfg = TrainingConfig(
                    data=DataConfig(validation_fraction=0.1),
                    first_stage=FirstStageConfig(image_size=8, latent_channels=1),
                    diffusion=DiffusionConfig(num_classes=2),
                    one_shot=OneShotConfig(code_dim=4, num_classes=2),
                    distillation=DistillationConfig(
                        samples_per_class=10,
                        class_labels=(0, 1),
                        filter_keep_per_class=4,
                        filter_foreground_threshold=0.2,
                    ),
                )
                run_manager = RecordingRunManager(run_dir)
                run_meta = {}

                run_teacher_distill_filter(
                    torch,
                    train_cfg=train_cfg,
                    run_manager=run_manager,
                    run_meta=run_meta,
                    distill_dataset=str(source_path),
                )

                filtered = torch.load(run_manager.bundle_path, map_location="cpu", weights_only=False)
                self.assertEqual(tuple(filtered["codes"].shape), (8, 4))
                self.assertEqual(tuple(filtered["teacher_camera"].shape), (8, 1, 8, 8))
                self.assertEqual(tuple(filtered["teacher_initial_noise"].shape), (8, 1, 1, 1))
                self.assertIn("filter", filtered["config"])
                self.assertEqual((filtered["class_labels"] == 0).sum().item(), 4)
                self.assertEqual((filtered["class_labels"] == 1).sum().item(), 4)
                self.assertEqual(run_manager.metrics_payload["class_counts"], {"0": 4, "1": 4})
                self.assertEqual(run_manager.meta_payload["teacher_distill_filter_sample_count"], 8)

        def test_teacher_distill_label_audit_and_strict_selection_use_requested_label(self) -> None:
            import torch

            from ldm_project.config import DistillationConfig, OneShotConfig, TrainingConfig
            from ldm_project.teacher_distill import (
                _build_label_prototype_bank_from_tensors,
                _select_teacher_distill_label_strict_indices,
                compute_teacher_distill_label_audit,
            )

            proto0 = torch.zeros(1, 4, 4)
            proto0[:, :, :2] = 1.0
            proto1 = torch.zeros(1, 4, 4)
            proto1[:, :, 2:] = 1.0
            prototype_images = torch.stack([proto0, proto0, proto1, proto1], dim=0)
            prototype_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
            prototypes = _build_label_prototype_bank_from_tensors(
                torch,
                prototype_images,
                prototype_labels,
                class_labels=(0, 1),
                image_size=4,
            )

            teacher_camera = torch.stack([proto0, proto0 * 0.9, proto1, proto1, proto1, proto0], dim=0)
            labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
            audit = compute_teacher_distill_label_audit(
                torch,
                teacher_camera,
                labels,
                prototypes,
                class_labels=(0, 1),
                foreground_threshold=0.2,
            )
            self.assertEqual(tuple(prototypes.shape), (2, 1, 4, 4))
            self.assertEqual(audit["nearest_label"].tolist(), [0, 0, 1, 1, 1, 0])
            self.assertTrue(audit["label_correct"][0].item())
            self.assertFalse(audit["label_correct"][2].item())
            self.assertFalse(audit["label_correct"][5].item())

            train_cfg = TrainingConfig(
                one_shot=OneShotConfig(num_classes=2),
                distillation=DistillationConfig(class_labels=(0, 1), label_strict_keep_per_class=2),
            )
            selected, summary = _select_teacher_distill_label_strict_indices(torch, labels, audit, train_cfg)
            self.assertEqual(len(selected), 4)
            self.assertEqual((labels[torch.tensor(selected)] == 0).sum().item(), 2)
            self.assertEqual((labels[torch.tensor(selected)] == 1).sum().item(), 2)
            self.assertFalse(summary["per_class"]["0"]["relaxed_label_filter"])
            self.assertFalse(summary["per_class"]["1"]["relaxed_label_filter"])
            self.assertEqual(summary["per_class"]["0"]["selected_label_correct"], 2)
            self.assertEqual(summary["per_class"]["1"]["selected_label_correct"], 2)

            relaxed_cfg = TrainingConfig(
                one_shot=OneShotConfig(num_classes=2),
                distillation=DistillationConfig(class_labels=(0, 1), label_strict_keep_per_class=3),
            )
            relaxed_selected, relaxed_summary = _select_teacher_distill_label_strict_indices(
                torch,
                labels,
                audit,
                relaxed_cfg,
            )
            self.assertEqual(len(relaxed_selected), 6)
            self.assertTrue(relaxed_summary["per_class"]["0"]["relaxed_label_filter"])
            self.assertTrue(relaxed_summary["per_class"]["1"]["relaxed_label_filter"])

        def test_teacher_distill_label_audit_batched_matches_full_audit(self) -> None:
            import torch

            from ldm_project.teacher_distill import (
                compute_teacher_distill_label_audit,
                compute_teacher_distill_label_audit_batched,
            )

            proto0 = torch.zeros(1, 4, 4)
            proto0[:, :, :2] = 1.0
            proto1 = torch.zeros(1, 4, 4)
            proto1[:, :, 2:] = 1.0
            prototypes = torch.stack([proto0, proto1], dim=0)
            teacher_camera = torch.stack([proto0, proto1, proto0 * 0.8, proto1 * 0.7, proto0], dim=0)
            labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)

            full = compute_teacher_distill_label_audit(
                torch,
                teacher_camera,
                labels,
                prototypes,
                class_labels=(0, 1),
                foreground_threshold=0.2,
            )
            batched = compute_teacher_distill_label_audit_batched(
                torch,
                teacher_camera,
                labels,
                prototypes,
                class_labels=(0, 1),
                batch_size=2,
                device=torch.device("cpu"),
                foreground_threshold=0.2,
            )

            self.assertEqual(set(batched), set(full))
            self.assertEqual(batched["nearest_label"].tolist(), full["nearest_label"].tolist())
            self.assertEqual(batched["label_correct"].tolist(), full["label_correct"].tolist())
            torch.testing.assert_close(batched["label_margin"], full["label_margin"])
            torch.testing.assert_close(batched["teacher_readable"], full["teacher_readable"])

        def test_teacher_distill_label_strict_relaxed_filter_preserves_all_correct_samples(self) -> None:
            import torch

            from ldm_project.config import DistillationConfig, OneShotConfig, TrainingConfig
            from ldm_project.teacher_distill import _select_teacher_distill_label_strict_indices

            labels = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
            audit = {
                "label_correct": torch.tensor([True, True, False, False, False]),
                "label_margin": torch.tensor([0.01, 0.02, 10.0, 5.0, 1.0]),
                "requested_label_corr": torch.tensor([0.7, 0.8, 0.95, 0.9, 0.85]),
                "background_speckle_ratio": torch.zeros(5),
                "readable_contrast": torch.zeros(5),
                "foreground_hole_ratio": torch.zeros(5),
            }
            train_cfg = TrainingConfig(
                one_shot=OneShotConfig(num_classes=1),
                distillation=DistillationConfig(class_labels=(0,), label_strict_keep_per_class=3),
            )

            selected, summary = _select_teacher_distill_label_strict_indices(torch, labels, audit, train_cfg)

            self.assertEqual(selected, [0, 1, 2])
            self.assertTrue(summary["per_class"]["0"]["relaxed_label_filter"])
            self.assertEqual(summary["per_class"]["0"]["label_correct_available"], 2)
            self.assertEqual(summary["per_class"]["0"]["selected_label_correct"], 2)
            self.assertEqual(summary["per_class"]["0"]["kept"], 3)

        def test_teacher_distill_label_strict_filter_writes_balanced_bundle_and_audit(self) -> None:
            import torch

            from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.teacher_distill import run_teacher_distill_label_strict_filter

            proto0 = torch.zeros(1, 4, 4)
            proto0[:, :, :2] = 1.0
            proto1 = torch.zeros(1, 4, 4)
            proto1[:, :, 2:] = 1.0
            prototypes = torch.stack([proto0, proto1], dim=0)

            class RecordingRunManager:
                def __init__(self, run_dir: Path) -> None:
                    self.run_dir = run_dir
                    self.images_dir = run_dir / "images"
                    self.artifacts_dir = run_dir / "artifacts"
                    self.configs_dir = run_dir / "configs"
                    self.meta_payload = None
                    self.bundle_path = None
                    self.artifact_payloads = {}
                    self.config_payloads = {}
                    self.grid_calls = []

                def write_meta(self, meta):
                    self.meta_payload = dict(meta)

                def save_config_snapshot(self, name, data):
                    self.config_payloads[name] = dict(data)
                    return self.configs_dir / f"{name}.json"

                def save_tensor_bundle(self, name, payload):
                    path = self.artifacts_dir / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(payload), path)
                    self.bundle_path = path
                    return path

                def save_artifact_json(self, name, payload):
                    self.artifact_payloads[name] = payload
                    path = self.artifacts_dir / f"{name}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
                    return path

                def save_tensor_grid(
                    self,
                    tensor_map,
                    destination,
                    title="",
                    max_items=4,
                    row_labels=None,
                    cmap="gray",
                    phase_colorbar=False,
                ):
                    self.grid_calls.append(
                        (
                            Path(destination).name,
                            list(tensor_map.keys()),
                            {
                                "row_labels": row_labels,
                                "cmap": cmap,
                                "phase_colorbar": phase_colorbar,
                            },
                        )
                    )
                    path = Path(destination)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    return path

            with tempfile.TemporaryDirectory() as tmpdir:
                run_dir = Path(tmpdir)
                source_path = run_dir / "source_distill.pt"
                labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
                teacher_camera = torch.stack([proto0, proto0, proto1, proto1, proto1, proto0], dim=0)
                torch.save(
                    {
                        "codes": torch.arange(18, dtype=torch.float32).reshape(6, 3),
                        "class_labels": labels,
                        "teacher_camera": teacher_camera,
                        "teacher_image": teacher_camera.clone(),
                    },
                    source_path,
                )
                train_cfg = TrainingConfig(
                    data=DataConfig(validation_fraction=0.0),
                    first_stage=FirstStageConfig(image_size=4),
                    one_shot=OneShotConfig(code_dim=3, num_classes=2),
                    distillation=DistillationConfig(
                        class_labels=(0, 1),
                        label_strict_keep_per_class=2,
                        label_audit_prototypes_per_class=2,
                    ),
                )
                run_manager = RecordingRunManager(run_dir)
                with unittest.mock.patch(
                    "ldm_project.teacher_distill._build_mnist_label_prototype_bank",
                    return_value=prototypes,
                ):
                    run_teacher_distill_label_strict_filter(
                        torch,
                        train_cfg=train_cfg,
                        run_manager=run_manager,
                        run_meta={},
                        distill_dataset=str(source_path),
                    )

                filtered = torch.load(run_manager.bundle_path, map_location="cpu", weights_only=False)
                self.assertEqual(run_manager.bundle_path.name, "teacher_distill_dataset_label_strict.pt")
                self.assertEqual(tuple(filtered["codes"].shape), (4, 3))
                self.assertEqual((filtered["class_labels"] == 0).sum().item(), 2)
                self.assertEqual((filtered["class_labels"] == 1).sum().item(), 2)
                self.assertEqual(filtered["config"]["filter"]["teacher_filter"], "label_strict")
                self.assertIn("label_audit_nearest_label", filtered)
                metrics = run_manager.artifact_payloads["teacher_distill_label_strict_filter_metrics"]
                self.assertEqual(metrics["teacher_filter"], "label_strict")
                self.assertEqual(metrics["class_counts"], {"0": 2, "1": 2})
                self.assertIn("label_margin", metrics["per_class"]["0"])
                audit = run_manager.artifact_payloads["teacher_distill_label_audit"]
                self.assertIn("per_sample", audit)
                self.assertIn("nearest_label", audit["per_sample"][0])
                self.assertEqual(run_manager.meta_payload["teacher_distill_filter"], "label_strict")
                self.assertIn("teacher_distill_label_audit.png", [call[0] for call in run_manager.grid_calls])

        def test_teacher_distill_readable_split_requires_readable_targets(self) -> None:
            import torch

            from ldm_project.config import DistillationConfig, OneShotConfig, TrainingConfig
            from ldm_project.teacher_distill import _split_readable_teacher_indices

            labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
            audit = {
                "label_correct": torch.tensor([True, True, True, True, False, True, True, True, True, False]),
                "label_margin": torch.tensor([0.04, 0.05, 0.06, 0.07, 0.90, 0.04, 0.05, 0.06, 0.07, 0.90]),
                "requested_label_corr": torch.ones(10),
                "nearest_label": labels.clone(),
                "background_speckle_ratio": torch.tensor([0.10, 0.12, 0.11, 0.13, 0.01, 0.10, 0.12, 0.11, 0.13, 0.01]),
                "readable_contrast": torch.tensor([0.80, 0.81, 0.82, 0.83, 0.95, 0.80, 0.81, 0.82, 0.83, 0.95]),
                "foreground_hole_ratio": torch.tensor([0.01, 0.02, 0.03, 0.04, 0.01, 0.01, 0.02, 0.03, 0.04, 0.01]),
            }
            train_cfg = TrainingConfig(
                one_shot=OneShotConfig(num_classes=2),
                distillation=DistillationConfig(
                    class_labels=(0, 1),
                    readable_train_per_class=2,
                    readable_val_per_class=1,
                    readable_test_per_class=1,
                    readable_split_seed=7,
                ),
            )

            summary = _split_readable_teacher_indices(torch, labels, audit, train_cfg)

            self.assertFalse(summary["failed"])
            self.assertEqual(len(summary["indices"]["train"]), 4)
            self.assertEqual(len(summary["indices"]["val"]), 2)
            self.assertEqual(len(summary["indices"]["test"]), 2)
            self.assertEqual(summary["per_class"]["0"]["readable_available"], 4)
            self.assertEqual(summary["per_class"]["1"]["readable_available"], 4)
            all_indices = (
                summary["indices"]["train"] + summary["indices"]["val"] + summary["indices"]["test"]
            )
            self.assertEqual(len(all_indices), len(set(all_indices)))

            fail_cfg = TrainingConfig(
                one_shot=OneShotConfig(num_classes=2),
                distillation=DistillationConfig(
                    class_labels=(0, 1),
                    readable_train_per_class=3,
                    readable_val_per_class=1,
                    readable_test_per_class=1,
                ),
            )
            failed = _split_readable_teacher_indices(torch, labels, audit, fail_cfg)
            self.assertTrue(failed["failed"])
            self.assertTrue(failed["per_class"]["0"]["readability_filter_failed"])
            self.assertEqual(failed["indices"]["train"], [])

        def test_teacher_distill_readable_split_writes_train_val_test_bundles(self) -> None:
            import torch

            from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.teacher_distill import run_teacher_distill_readable_split

            proto0 = torch.zeros(1, 4, 4)
            proto0[:, :, :2] = 1.0
            proto1 = torch.zeros(1, 4, 4)
            proto1[:, :, 2:] = 1.0
            prototypes = torch.stack([proto0, proto1], dim=0)

            class RecordingRunManager:
                def __init__(self, run_dir: Path) -> None:
                    self.run_dir = run_dir
                    self.images_dir = run_dir / "images"
                    self.artifacts_dir = run_dir / "artifacts"
                    self.configs_dir = run_dir / "configs"
                    self.meta_payload = None
                    self.bundle_paths = {}
                    self.artifact_payloads = {}
                    self.config_payloads = {}
                    self.grid_calls = []

                def write_meta(self, meta):
                    self.meta_payload = dict(meta)

                def save_config_snapshot(self, name, data):
                    self.config_payloads[name] = dict(data)
                    return self.configs_dir / f"{name}.json"

                def save_tensor_bundle(self, name, payload):
                    path = self.artifacts_dir / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(payload), path)
                    self.bundle_paths[name] = path
                    return path

                def save_artifact_json(self, name, payload):
                    self.artifact_payloads[name] = payload
                    path = self.artifacts_dir / f"{name}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
                    return path

                def save_tensor_grid(
                    self,
                    tensor_map,
                    destination,
                    title="",
                    max_items=4,
                    row_labels=None,
                    cmap="gray",
                    phase_colorbar=False,
                ):
                    self.grid_calls.append((Path(destination).name, list(tensor_map.keys()), row_labels))
                    path = Path(destination)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    return path

            with tempfile.TemporaryDirectory() as tmpdir:
                run_dir = Path(tmpdir)
                source_path = run_dir / "source_distill.pt"
                labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
                teacher_camera = torch.stack([proto0, proto0, proto0, proto0, proto1, proto1, proto1, proto1], dim=0)
                torch.save(
                    {
                        "codes": torch.arange(24, dtype=torch.float32).reshape(8, 3),
                        "class_labels": labels,
                        "teacher_camera": teacher_camera,
                        "teacher_image": teacher_camera.clone(),
                        "config": {"source": "unit"},
                    },
                    source_path,
                )
                train_cfg = TrainingConfig(
                    data=DataConfig(validation_fraction=0.0),
                    first_stage=FirstStageConfig(image_size=4),
                    one_shot=OneShotConfig(code_dim=3, num_classes=2),
                    distillation=DistillationConfig(
                        class_labels=(0, 1),
                        label_audit_prototypes_per_class=2,
                        readable_min_label_margin=-1.0,
                        readable_min_contrast=0.0,
                        readable_max_background_speckle=1.0,
                        readable_max_foreground_hole=1.0,
                        readable_train_per_class=2,
                        readable_val_per_class=1,
                        readable_test_per_class=1,
                        readable_split_seed=11,
                    ),
                )
                run_manager = RecordingRunManager(run_dir)
                with unittest.mock.patch(
                    "ldm_project.teacher_distill._build_mnist_label_prototype_bank",
                    return_value=prototypes,
                ):
                    run_teacher_distill_readable_split(
                        torch,
                        train_cfg=train_cfg,
                        run_manager=run_manager,
                        run_meta={},
                        distill_dataset=str(source_path),
                        device=torch.device("cpu"),
                    )

                self.assertFalse(run_manager.artifact_payloads["teacher_distill_readable_split_metrics"]["readability_filter_failed"])
                self.assertEqual(set(run_manager.bundle_paths), {
                    "teacher_distill_dataset_readable_train.pt",
                    "teacher_distill_dataset_readable_val.pt",
                    "teacher_distill_dataset_readable_test.pt",
                })
                train_payload = torch.load(
                    run_manager.bundle_paths["teacher_distill_dataset_readable_train.pt"],
                    map_location="cpu",
                    weights_only=False,
                )
                val_payload = torch.load(
                    run_manager.bundle_paths["teacher_distill_dataset_readable_val.pt"],
                    map_location="cpu",
                    weights_only=False,
                )
                test_payload = torch.load(
                    run_manager.bundle_paths["teacher_distill_dataset_readable_test.pt"],
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(tuple(train_payload["codes"].shape), (4, 3))
                self.assertEqual(tuple(val_payload["codes"].shape), (2, 3))
                self.assertEqual(tuple(test_payload["codes"].shape), (2, 3))
                self.assertEqual(train_payload["config"]["filter"]["teacher_filter"], "readability_strict_split")
                self.assertIn("label_audit_margin", train_payload)
                split_indices = [
                    *train_payload["readability_filter_indices"].tolist(),
                    *val_payload["readability_filter_indices"].tolist(),
                    *test_payload["readability_filter_indices"].tolist(),
                ]
                self.assertEqual(len(split_indices), len(set(split_indices)))
                self.assertEqual(run_manager.meta_payload["teacher_distill_readability_filter_failed"], False)
                self.assertIn("teacher_distill_readable_split_before_after.png", [call[0] for call in run_manager.grid_calls])

        def test_prepare_distill_loaders_uses_external_validation_dataset(self) -> None:
            import torch

            from ldm_project.config import DataConfig, DistillationConfig, FirstStageConfig, OneShotConfig, TrainingConfig
            from ldm_project.one_shot_distill_training import _prepare_distill_loaders

            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                train_path = root / "train.pt"
                val_path = root / "val.pt"
                torch.save(
                    {
                        "codes": torch.arange(12, dtype=torch.float32).reshape(4, 3),
                        "class_labels": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                        "teacher_camera": torch.rand(4, 1, 4, 4),
                    },
                    train_path,
                )
                torch.save(
                    {
                        "codes": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                        "class_labels": torch.tensor([0, 1], dtype=torch.long),
                        "teacher_camera": torch.rand(2, 1, 4, 4),
                    },
                    val_path,
                )
                train_cfg = TrainingConfig(
                    batch_size=2,
                    data=DataConfig(validation_fraction=0.5),
                    first_stage=FirstStageConfig(image_size=4),
                    one_shot=OneShotConfig(code_dim=3, num_classes=2),
                    distillation=DistillationConfig(),
                )

                loaders = _prepare_distill_loaders(
                    torch,
                    distill_dataset=str(train_path),
                    distill_validation_dataset=str(val_path),
                    train_cfg=train_cfg,
                    use_pin_memory=False,
                )

                self.assertEqual(loaders.train_size, 4)
                self.assertEqual(loaders.validation_size, 2)
                self.assertEqual(loaders.total_size, 6)
                self.assertEqual(loaders.validation_source, "external_dataset")
                self.assertEqual(loaders.validation_dataset_path, str(val_path))
