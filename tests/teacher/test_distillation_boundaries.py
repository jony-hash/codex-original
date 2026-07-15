from __future__ import annotations

from ldm_project.generation import project_codes_to_teacher_noise
from ldm_project.teacher import distill
from ldm_project.teacher.distill_support import (
    balanced_preview_indices,
    metric_summary,
    project_codes_to_teacher_noise as scoped_project_codes_to_teacher_noise,
)
from ldm_project.teacher.export import run_teacher_distill_export


def test_teacher_public_entrypoints_are_canonical() -> None:
    assert distill.run_teacher_distill_export is run_teacher_distill_export
    assert distill._project_codes_to_teacher_noise is project_codes_to_teacher_noise
    assert scoped_project_codes_to_teacher_noise is project_codes_to_teacher_noise


def test_teacher_support_private_aliases_preserve_legacy_identity() -> None:
    from ldm_project.teacher import distill_support

    assert distill_support._metric_summary is metric_summary
    assert distill_support._balanced_preview_indices is balanced_preview_indices
