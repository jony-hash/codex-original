from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.capacity import electronic_o0_direct_prior_capacity_canary as dpgc


def _synthetic(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, labels = [], []
    for label in range(10):
        base = torch.zeros(1, 28, 28)
        base[:, label : label + 3, label : label + 3] = 0.7
        rows.append((base + 0.03 * torch.randn((12, 1, 28, 28), generator=generator)).clamp(0, 1))
        labels.append(torch.full((12,), label, dtype=torch.long))
    return torch.cat(rows), torch.cat(labels)


def test_config_and_manifest_cli_are_registered() -> None:
    root = Path(__file__).parents[3]
    config = dpgc._config(root / "configs/research/capacity/electronic_o0_direct_prior_capacity_canary_v1.yaml")
    assert config["direct_prior_capacity_canary_seed"] == dpgc.HARNESS_SEED
    assert config["rank"] == dpgc.CODE_DIM == 8
    for action in ("validate", "preflight", "audit", "report", "closeout"):
        args = _parser().parse_args([
            "research", "electronic-o0-direct-prior-capacity-canary", action,
            "--run-manifest", "manifest.yaml",
        ])
        assert args.research_action == action


def test_rank8_canary_is_deterministic_and_chunk_invariant() -> None:
    images, labels = _synthetic()
    first = dpgc._fit_rank8(images, labels)
    second = dpgc._fit_rank8(images, labels)
    assert torch.equal(first.means, second.means)
    assert torch.equal(first.bases, second.bases)
    assert torch.equal(first.scales, second.scales)
    codes = torch.randn((20, 8), generator=torch.Generator().manual_seed(99))
    sample_labels = torch.arange(10).repeat_interleave(2)
    assert torch.equal(first.sample(sample_labels, codes, chunk_size=3), first.sample(sample_labels, codes, chunk_size=64))
    assert torch.isfinite(first.sample(sample_labels, codes)).all()


def test_canary_rejects_bad_shapes_and_nonfinite_codes() -> None:
    images, labels = _synthetic()
    canary = dpgc._fit_rank8(images, labels)
    with pytest.raises(ValueError):
        canary.sample(labels[:3], torch.zeros((3, 7)))
    bad = torch.zeros((3, 8))
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        canary.sample(labels[:3], bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device test")
def test_memorization_accepts_cuda_generated_images() -> None:
    images, labels = _synthetic()
    generated = images[:20].to("cuda")
    result = dpgc._memorization(generated, labels[:20].to("cuda"), images, labels, images, labels)
    assert result["exact_duplicate_count"] >= 0


def test_canonical_basis_has_deterministic_signs() -> None:
    value = torch.tensor([[0.0, -2.0], [-1.0, 0.5]])
    result = dpgc._canonical_basis(value)
    assert result[0, 1] > 0
    assert result[1, 0] > 0


def _measurement(classification: str) -> dict:
    good = {
        "primary": {"accuracy": 0.99, "minimum_recall": 0.98},
        "secondary": {"accuracy": 0.99, "minimum_recall": 0.98},
        "agreement": 0.99,
        "camera_code_fraction": 0.5,
        "saturation": 0.01,
        "label_shuffled_primary_accuracy": 0.1,
        "diversity_ratio": {"primary": [1.0] * 10, "secondary": [1.0] * 10},
        "coverage": {"primary": 1.0, "secondary": 1.0},
        "memorization": {"exact_duplicate_count": 0, "pass_fraction": 0.99},
        "no_update_replay_exact": True,
    }
    if classification == "semantic":
        good["primary"]["minimum_recall"] = 0.5
    if classification == "diversity":
        good["memorization"]["pass_fraction"] = 0.5
    return {"panels": {str(i): dict(good) for i in range(3)}, "jacobian": {"rank_1e-3_pass_fraction": 1.0}}


def test_classification_is_fail_closed() -> None:
    assert dpgc._classify(_measurement("good"))[0] == "direct_prior_linear_capacity_admitted"
    assert dpgc._classify(_measurement("semantic"))[0] == "direct_prior_linear_semantic_capacity_failed"
    assert dpgc._classify(_measurement("diversity"))[0] == "direct_prior_linear_diversity_capacity_failed"
    invalid = {"status": "invalid_harness", "measurement_valid": False}
    assert dpgc._classify(invalid) == ("invalid_harness", "repair_and_revalidate_on_a_new_clean_commit")


def test_no_optimizer_symbol_is_used() -> None:
    source = Path(dpgc.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "optimizer.step" not in source
