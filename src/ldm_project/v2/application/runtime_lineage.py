"""Component-scoped provenance for v2 evidence consumers."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..evidence import canonical_sha256


RUNTIME_COMPONENT_REGISTRY_VERSION = "v2-runtime-components-2"

_FULL_CURRENT = ("src/ldm_project/v2/", "configs/v2/current/", "pyproject.toml")
_COMPONENT_PATHS = {
    "foundation": (
        "src/ldm_project/v2/evidence/",
        "src/ldm_project/v2/application/runs.py",
        "src/ldm_project/v2/application/_runtime_shared.py",
        "src/ldm_project/v2/application/runtime_lineage.py",
        "src/ldm_project/v2/config.py",
        "pyproject.toml",
    ),
    "data_evaluator": (
        "src/ldm_project/v2/student/evaluator.py",
        "src/ldm_project/v2/application/current/runtime.py",
    ),
    "optical_generator": (
        "src/ldm_project/v2/optics/",
        "src/ldm_project/v2/student/identity_generator.py",
        "src/ldm_project/v2/student/checkpoints.py",
    ),
    "control_admission": (
        "src/ldm_project/v2/application/_conditional_admission_shared.py",
        "src/ldm_project/v2/application/current/admission.py",
    ),
    "recovery": (
        "src/ldm_project/v2/application/_conditional_recovery_shared.py",
        "src/ldm_project/v2/application/current/recovery.py",
    ),
    "formal_objective_training": (
        "src/ldm_project/v2/student/conditional.py",
        "src/ldm_project/v2/application/current/training.py",
        "src/ldm_project/v2/application/current/admission.py",
    ),
    "acceptance_measurement": ("src/ldm_project/v2/application/current/measurement.py",),
    "formal_evaluation": (
        "src/ldm_project/v2/application/current/evaluation.py",
        "src/ldm_project/v2/application/current/admission.py",
    ),
    "dataset_admission": (
        "src/ldm_project/v2/application/current/dataset.py",
        "src/ldm_project/v2/application/current/admission.py",
        "src/ldm_project/v2/application/current/measurement.py",
    ),
    "objective_diagnostic": (
        "src/ldm_project/v2/application/diagnostic/objective.py",
        "src/ldm_project/v2/student/conditional.py",
        "src/ldm_project/v2/application/current/training.py",
        "src/ldm_project/v2/application/current/recovery.py",
        "src/ldm_project/v2/application/_conditional_recovery_shared.py",
        "src/ldm_project/v2/application/current/admission.py",
    ),
}

_KIND_COMPONENTS = {
    "evaluator_qualification": ("foundation", "data_evaluator"),
    "acceptance_evaluator_training": ("foundation", "data_evaluator", "acceptance_measurement"),
    "conditional_acceptance_measurement": ("foundation", "data_evaluator", "acceptance_measurement"),
    "conditional_injection_admission": ("foundation", "data_evaluator", "optical_generator", "control_admission"),
    "conditional_observability": ("foundation", "data_evaluator", "optical_generator", "control_admission"),
    "conditional_control_isolation": ("foundation", "data_evaluator", "optical_generator", "control_admission"),
    "conditional_semantic_localization": ("foundation", "data_evaluator", "optical_generator", "control_admission"),
    "controllability_smoke": ("foundation", "data_evaluator", "optical_generator", "recovery"),
    "conditional_semantic_recovery_smoke": ("foundation", "data_evaluator", "optical_generator", "recovery"),
    "conditional_training": ("foundation", "data_evaluator", "optical_generator", "formal_objective_training", "recovery"),
    "conditional_training_completion": ("foundation", "data_evaluator", "optical_generator", "formal_objective_training", "recovery"),
    "conditional_acceptance": ("foundation", "data_evaluator", "optical_generator", "formal_evaluation"),
    "conditional_generation_admission": ("foundation", "data_evaluator", "optical_generator", "formal_evaluation"),
    "conditional_dataset_candidate": ("foundation", "data_evaluator", "optical_generator", "dataset_admission"),
    "conditional_memorization_audit": ("foundation", "data_evaluator", "dataset_admission"),
    "conditional_utility": ("foundation", "data_evaluator", "dataset_admission"),
    "conditional_dataset_accepted": ("foundation", "data_evaluator", "dataset_admission"),
    "conditional_objective_weight_calibration": ("foundation", "data_evaluator", "optical_generator", "objective_diagnostic"),
    "conditional_objective_alignment_screen": ("foundation", "data_evaluator", "optical_generator", "objective_diagnostic", "acceptance_measurement"),
    "conditional_objective_alignment_confirm": ("foundation", "data_evaluator", "optical_generator", "objective_diagnostic", "acceptance_measurement"),
}


def components_for_kind(evidence_kind: str) -> tuple[str, ...]:
    return _KIND_COMPONENTS.get(evidence_kind, ("full_current",))


def _paths_for_component(component: str) -> tuple[str, ...]:
    if component == "full_current":
        return _FULL_CURRENT
    return _COMPONENT_PATHS[component]


def component_fingerprints(repo_root: Path, commit: str, evidence_kind: str) -> dict[str, str]:
    """Hash Git's tracked tree entries at a commit, independent of worktree state."""
    components = components_for_kind(evidence_kind)
    result = {}
    for component in components:
        paths = _paths_for_component(component)
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-tree", "-r", commit, "--", *paths],
            check=True,
            capture_output=True,
            text=True,
        )
        result[component] = canonical_sha256({"component": component, "paths": paths, "tree": completed.stdout})
    return result


def lineage_inputs(repo_root: Path, commit: str, evidence_kind: str) -> dict[str, object]:
    components = components_for_kind(evidence_kind)
    return {
        "runtime_component_registry": RUNTIME_COMPONENT_REGISTRY_VERSION,
        "runtime_components": components,
        "runtime_component_fingerprints": component_fingerprints(repo_root, commit, evidence_kind),
    }
