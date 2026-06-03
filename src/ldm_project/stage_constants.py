from __future__ import annotations


FIRST_STAGE_BEST_OPTICAL_CHECKPOINT = "first_stage_best_optical.pt"
FIRST_STAGE_BEST_OPTICAL_METRIC = "camera_complex_balanced_mnist_score"
DIFFUSION_BEST_CHECKPOINT = "diffusion_best.pt"
DIFFUSION_BEST_METRIC = "diffusion_validation.loss"
ONE_SHOT_BEST_CHECKPOINT = "one_shot_best.pt"
ONE_SHOT_BEST_METRIC = "one_shot_validation.loss"
ONE_SHOT_DISTILL_BEST_CHECKPOINT = "one_shot_distill_best.pt"
ONE_SHOT_DISTILL_LATEST_CHECKPOINT = "one_shot_distill_latest.pt"
ONE_SHOT_DISTILL_BEST_METRIC = "one_shot_distill_validation.loss"
ONE_SHOT_DISTILL_QUALITY_METRIC = "one_shot_distill_validation.quality"
ONE_SHOT_DISTILL_BALANCED_QUALITY_METRIC = "one_shot_distill_validation.balanced_quality"
ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_METRIC = "one_shot_distill_validation.strict_balanced_quality"
ONE_SHOT_DISTILL_QUALITY_SATURATION_TARGET = 0.325
ONE_SHOT_DISTILL_BALANCED_QUALITY_SATURATION_TARGET = 0.39
ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_SATURATION_TARGET = 0.36
