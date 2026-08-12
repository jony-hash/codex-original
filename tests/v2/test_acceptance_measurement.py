from __future__ import annotations

import pytest
import torch

from ldm_project.v2.application.current import measurement


def test_official_test_manifest_requires_exact_ten_thousand_samples() -> None:
    images = torch.zeros(10_000, 1, 28, 28)
    labels = torch.arange(10_000) % 10
    manifest = measurement._test_manifest(images, labels)
    assert manifest["kind"] == "official_mnist_test_10000_v1"
    assert manifest["count"] == 10_000


@pytest.mark.parametrize("count", [5, 9_999, 10_001])
def test_non_official_test_counts_are_rejected_before_manifest_write(count: int) -> None:
    images = torch.zeros(count, 1, 28, 28)
    labels = torch.arange(count) % 10
    with pytest.raises(ValueError, match="exactly 10000"):
        measurement._require_official_test_split(images, labels)
