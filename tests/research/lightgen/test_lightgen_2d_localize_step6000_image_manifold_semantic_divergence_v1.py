from __future__ import annotations

import torch

from scripts.research.lightgen.execute_lightgen_2d_localize_step6000_image_manifold_semantic_divergence_v1 import _fit_pca, _nearest_distances, _select_stratum_examples, _strata


def test_consensus_strata_are_disjoint_and_cover_all_rows() -> None:
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    predictions = {
        "E_S": torch.tensor([0, 1, 2, 3, 4, 0, 6, 7]),
        "E_I": torch.tensor([0, 1, 2, 3, 4, 1, 6, 0]),
        "E_C": torch.tensor([0, 1, 2, 3, 0, 2, 6, 1]),
        "E_D": torch.tensor([0, 1, 9, 8, 4, 3, 5, 2]),
    }
    strata = _strata(predictions, labels)
    masks = [strata[name] for name in ("G1", "G2", "G3", "G4")]
    assert torch.equal(torch.stack(masks).to(torch.int64).sum(dim=0), torch.ones(labels.numel(), dtype=torch.int64))
    assert strata["G1"].tolist() == [True, True, False, False, False, False, False, False]
    assert strata["G2"].tolist() == [False, False, True, True, True, False, True, False]


def test_stratum_selection_is_first_sorted_position_per_class() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    strata = {
        "G1": torch.tensor([True, False, True, False, False, False]),
        "G2": torch.tensor([False, True, False, True, False, False]),
        "G3": torch.tensor([False, False, False, False, True, False]),
        "G4": torch.tensor([False, False, False, False, False, True]),
    }
    selected = _select_stratum_examples(strata, labels, per_class=1)
    assert selected == {"G1": [0, 2], "G2": [1, 3], "G3": [4], "G4": [5]}


def test_nearest_distances_bind_requested_and_competing_classes() -> None:
    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    reference_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    query = torch.tensor([[0.9, 0.0], [0.0, 1.8]])
    requested = torch.tensor([0, 0], dtype=torch.long)
    distances = _nearest_distances(query, reference, reference_labels, requested, device=torch.device("cpu"), dimension=2)
    assert distances["nearest_requested_index"].tolist() == [1, 0]
    assert distances["nearest_other_index"].tolist() == [2, 2]
    assert distances["nearest_any_index"].tolist() == [1, 2]


def test_pca_is_fit_without_labels_and_reports_rank() -> None:
    torch.manual_seed(123)
    images = torch.rand((12, 28 * 28), dtype=torch.float32)
    pca = _fit_pca(images, torch.device("cpu"))
    assert pca["rank"] >= 1
    assert pca["rank"] <= 256
    assert pca["components"].shape == (28 * 28, pca["rank"])
    assert pca["train_scores"].shape == (12, pca["rank"])
