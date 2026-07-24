from __future__ import annotations

from .distillation.dataset import (
    _build_distill_loaders,
    _select_distill_eval_indices,
    _subset_distill_payload_for_training,
)
from .student.evaluation import (
    run_one_shot_distill_eval_diagnostics_stage,
    run_one_shot_distill_eval_stage,
    run_one_shot_sample_stage,
)
from .student.artifacts import (
    _save_one_shot_d2nn_focus_visualization,
    _save_one_shot_slm_to_encoder_focus_visualization,
)
from .student.metrics import compute_one_shot_distill_eval_metrics
from .stage_constants import (
    DIFFUSION_BEST_CHECKPOINT,
    DIFFUSION_BEST_METRIC,
    FIRST_STAGE_BEST_OPTICAL_CHECKPOINT,
    FIRST_STAGE_BEST_OPTICAL_METRIC,
    ONE_SHOT_BEST_CHECKPOINT,
    ONE_SHOT_BEST_METRIC,
    ONE_SHOT_DISTILL_BALANCED_QUALITY_METRIC,
    ONE_SHOT_DISTILL_BEST_CHECKPOINT,
    ONE_SHOT_DISTILL_BEST_METRIC,
    ONE_SHOT_DISTILL_LATEST_CHECKPOINT,
    ONE_SHOT_DISTILL_QUALITY_METRIC,
    ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_METRIC,
)
from .student.training import run_one_shot_distill_training
from .student.quality import (
    _one_shot_distill_balanced_quality_score,
    _one_shot_distill_quality_score,
    _one_shot_distill_strict_balanced_quality_score,
    _resolve_one_shot_distill_selection,
)
from .student.baseline_training import run_one_shot_training
from .student.training_utils import (
    _one_shot_d2nn_crop_retention_metrics,
)
from .teacher_stages import (
    compute_teacher_distill_filter_metrics,
    run_diffusion_training,
    run_first_stage_training,
    run_sample_stage,
    run_teacher_diffusion_latent_audit,
    run_teacher_distill_export,
    run_teacher_distill_filter,
    run_teacher_distill_label_audit,
    run_teacher_distill_label_strict_filter,
    run_teacher_distill_readable_split,
    run_teacher_first_stage_oracle_audit,
)

__all__ = [
    "DIFFUSION_BEST_CHECKPOINT",
    "DIFFUSION_BEST_METRIC",
    "FIRST_STAGE_BEST_OPTICAL_CHECKPOINT",
    "FIRST_STAGE_BEST_OPTICAL_METRIC",
    "ONE_SHOT_BEST_CHECKPOINT",
    "ONE_SHOT_BEST_METRIC",
    "ONE_SHOT_DISTILL_BALANCED_QUALITY_METRIC",
    "ONE_SHOT_DISTILL_BEST_CHECKPOINT",
    "ONE_SHOT_DISTILL_BEST_METRIC",
    "ONE_SHOT_DISTILL_LATEST_CHECKPOINT",
    "ONE_SHOT_DISTILL_QUALITY_METRIC",
    "ONE_SHOT_DISTILL_STRICT_BALANCED_QUALITY_METRIC",
    "compute_one_shot_distill_eval_metrics",
    "compute_teacher_distill_filter_metrics",
    "run_one_shot_distill_eval_diagnostics_stage",
    "run_one_shot_distill_eval_stage",
    "run_diffusion_training",
    "run_first_stage_training",
    "run_one_shot_distill_training",
    "run_one_shot_sample_stage",
    "run_one_shot_training",
    "run_sample_stage",
    "run_teacher_diffusion_latent_audit",
    "run_teacher_distill_export",
    "run_teacher_distill_filter",
    "run_teacher_distill_label_audit",
    "run_teacher_distill_label_strict_filter",
    "run_teacher_distill_readable_split",
    "run_teacher_first_stage_oracle_audit",
    "_build_distill_loaders",
    "_one_shot_d2nn_crop_retention_metrics",
    "_one_shot_distill_balanced_quality_score",
    "_one_shot_distill_quality_score",
    "_one_shot_distill_strict_balanced_quality_score",
    "_resolve_one_shot_distill_selection",
    "_save_one_shot_d2nn_focus_visualization",
    "_save_one_shot_slm_to_encoder_focus_visualization",
    "_select_distill_eval_indices",
    "_subset_distill_payload_for_training",
]
