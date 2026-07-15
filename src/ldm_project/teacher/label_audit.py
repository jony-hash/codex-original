"""Teacher label-audit APIs."""
from .distill_legacy import (
    compute_teacher_distill_label_audit,
    compute_teacher_distill_label_audit_batched,
    run_teacher_distill_label_audit,
)

__all__ = [
    "compute_teacher_distill_label_audit",
    "compute_teacher_distill_label_audit_batched",
    "run_teacher_distill_label_audit",
]
