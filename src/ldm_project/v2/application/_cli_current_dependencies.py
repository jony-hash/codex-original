# ruff: noqa: F401
"""Dependencies required by the formal identity CLI commands only."""

from __future__ import annotations

from typing import Any


def install(namespace: dict[str, Any]) -> None:
    from ..config import V2Config, load_config
    from .current.admission import (
        conditional_control_isolation,
        conditional_injection_admission,
        conditional_observability,
        qualify_evaluator,
        sample_conditional,
        train_evaluator,
    )
    from .current.dataset import (
        admit_dataset,
        audit_memorization,
        evaluate_utility,
        export_conditional_dataset,
    )
    from .current.evaluation import (
        admit_conditional_evaluation,
        conditional_evaluate,
    )
    from .current.recovery import controllability_smoke, semantic_recovery_smoke
    from .current.training import conditional_train, conditional_training_completion

    namespace.update(locals())
