"""Formal/experimental loss composition for the single-shot student.

The individual metric implementations stay in :mod:`student.objectives` for
checkpoint and CLI compatibility.  This module owns only the final contract:
camera reconstruction is the formal term; every other term is explicitly
experimental and must be enabled by a non-zero configuration weight.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch


# The order is intentional: it matches the historical one_shot_loss addition
# order so moving the composer does not change floating-point accumulation.
FORMAL_TERM_KEYS = ("camera_recon_loss",)
EXPERIMENTAL_TERM_KEYS = (
    "shape_loss",
    "edge_loss",
    "slm_smoothness_loss",
    "slm_contrast_loss",
    "camera_saturation_loss",
    "camera_topology_loss",
    "camera_artifact_loss",
    "camera_readability_loss",
    "camera_background_speckle_loss",
    "camera_background_hotspot_loss",
    "camera_background_highfreq_loss",
    "camera_readable_contrast_loss",
    "student_label_margin_loss",
    "student_prototype_ce_loss",
    "student_negative_label_margin_loss",
    "student_roi_prototype_ce_loss",
    "student_roi_negative_label_margin_loss",
    "aux_semantic_loss",
    "aux_topology_loss",
    "code_diversity_loss",
    "sample_code_diversity_loss",
    "teacher_pairwise_diversity_loss",
    "cross_class_pairwise_loss",
    "clean_foreground_dice_loss",
    "clean_foreground_hole_loss",
    "clean_background_speckle_loss",
    "clean_background_hotspot_loss",
    "clean_background_highfreq_loss",
    "encoder_input_topology_loss",
    "student_pair_contrast_loss",
    "display_speckle_loss",
    "aux_detail_shape_loss",
    "aux_detail_edge_loss",
    "residual_l1_loss",
    "residual_background_loss",
)


def compose_one_shot_total(terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Compose the formal and explicitly experimental objective terms.

    ``terms`` contains already-weighted tensors.  Missing experimental terms
    are treated as zero, while a missing formal camera term is an error; this
    keeps accidental diagnostic-only objectives from becoming the training
    contract silently.
    """
    try:
        total = terms[FORMAL_TERM_KEYS[0]]
    except KeyError as exc:
        raise KeyError("formal objective requires camera_recon_loss") from exc
    for key in EXPERIMENTAL_TERM_KEYS:
        value = terms.get(key)
        if value is not None:
            total = total + value
    return total


__all__ = [
    "EXPERIMENTAL_TERM_KEYS",
    "FORMAL_TERM_KEYS",
    "compose_one_shot_total",
]
