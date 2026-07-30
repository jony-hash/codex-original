"""Explicit v2-to-v1 compatibility adapters.

This module is deliberately absent from the formal identity import path.  It
is loaded only by legacy teacher-target commands that must validate a historic
v1 export configuration.
"""

from __future__ import annotations

from ..evidence import file_sha256


def legacy_export_contract(path: str, config: object) -> str:
    from ldm_project.v1.config import load_training_config

    legacy = load_training_config(path)
    expected = (
        "teacher_distill_export",
        8,
        100,
        1234,
        2026,
        200,
        4.0,
        tuple(range(10)),
        False,
    )
    actual = (
        legacy.mode,
        legacy.one_shot.code_dim,
        legacy.distillation.samples_per_class,
        legacy.distillation.code_seed,
        legacy.distillation.code_projection_seed,
        legacy.distillation.sampling_steps,
        legacy.diffusion.guidance_scale,
        tuple(legacy.distillation.class_labels),
        legacy.data.download,
    )
    teacher = getattr(config, "teacher")
    if (
        actual != expected
        or teacher is None
        or teacher.seed != 1234
        or teacher.guidance_scale != 4.0
    ):
        raise ValueError(
            "formal teacher export requires the locked 8D/100-per-label legacy export contract"
        )
    return file_sha256(path)
