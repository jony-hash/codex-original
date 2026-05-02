from __future__ import annotations

from typing import cast
from unittest import mock
import unittest


class DatasetSplitTests(unittest.TestCase):
    def _build_fake_dataset(self, size: int = 20):
        import torch
        from torch.utils.data import TensorDataset

        images = torch.arange(size, dtype=torch.float32).view(size, 1, 1, 1)
        labels = torch.arange(size, dtype=torch.long)
        return TensorDataset(images, labels)

    def test_build_train_validation_loaders_splits_and_subsets_deterministically(self) -> None:
        from torch.utils.data import Subset

        from ldm_project.config import DataConfig, FirstStageConfig
        from ldm_project.data import build_train_validation_loaders

        dataset = self._build_fake_dataset(size=20)
        data_config = DataConfig(
            subset_fraction=0.5,
            subset_seed=3,
            validation_fraction=0.2,
            validation_split_seed=11,
            pin_memory=False,
        )

        with mock.patch("ldm_project.data._build_mnist_dataset", return_value=dataset):
            first = build_train_validation_loaders(data_config, FirstStageConfig(), batch_size=4)
            second = build_train_validation_loaders(data_config, FirstStageConfig(), batch_size=4)

        self.assertEqual(first.total_size, 20)
        self.assertEqual(first.train.total_size, 16)
        self.assertEqual(first.train.effective_size, 8)
        self.assertIsNotNone(first.validation)
        assert first.validation is not None
        self.assertIsNotNone(second.validation)
        assert second.validation is not None
        first_train_subset = cast(Subset, first.train.dataset)
        second_train_subset = cast(Subset, second.train.dataset)
        first_validation_subset = cast(Subset, first.validation.dataset)
        second_validation_subset = cast(Subset, second.validation.dataset)
        self.assertEqual(first.validation.total_size, 4)
        self.assertEqual(first.validation.effective_size, 4)
        self.assertEqual(first_train_subset.indices, second_train_subset.indices)
        self.assertEqual(first_validation_subset.indices, second_validation_subset.indices)
        self.assertTrue(set(first_train_subset.indices).isdisjoint(first_validation_subset.indices))

    def test_validation_fraction_zero_keeps_train_only_behavior(self) -> None:
        from ldm_project.config import DataConfig, FirstStageConfig
        from ldm_project.data import build_train_validation_loaders

        dataset = self._build_fake_dataset(size=12)
        data_config = DataConfig(
            subset_fraction=0.5,
            validation_fraction=0.0,
            pin_memory=False,
        )

        with mock.patch("ldm_project.data._build_mnist_dataset", return_value=dataset):
            splits = build_train_validation_loaders(data_config, FirstStageConfig(), batch_size=3)

        self.assertEqual(splits.total_size, 12)
        self.assertEqual(splits.train.total_size, 12)
        self.assertEqual(splits.train.effective_size, 6)
        self.assertIsNone(splits.validation)

    def test_subset_size_overrides_fraction_for_train_split(self) -> None:
        from torch.utils.data import Subset

        from ldm_project.config import DataConfig, FirstStageConfig
        from ldm_project.data import build_train_validation_loaders

        dataset = self._build_fake_dataset(size=30)
        data_config = DataConfig(
            subset_fraction=0.5,
            subset_size=7,
            subset_seed=5,
            validation_fraction=0.2,
            validation_split_seed=13,
            pin_memory=False,
        )

        with mock.patch("ldm_project.data._build_mnist_dataset", return_value=dataset):
            first = build_train_validation_loaders(data_config, FirstStageConfig(), batch_size=4)
            second = build_train_validation_loaders(data_config, FirstStageConfig(), batch_size=4)

        self.assertEqual(first.train.total_size, 24)
        self.assertEqual(first.train.effective_size, 7)
        self.assertIsNotNone(first.validation)
        assert first.validation is not None
        first_train_subset = cast(Subset, first.train.dataset)
        second_train_subset = cast(Subset, second.train.dataset)
        self.assertEqual(first_train_subset.indices, second_train_subset.indices)


if __name__ == "__main__":
    unittest.main()
