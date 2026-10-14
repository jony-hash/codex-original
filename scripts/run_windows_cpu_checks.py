"""Run the Windows CPU Actions regression, Ruff, and compile checks locally or in CI."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTEST_PATHS = (
    "tests/test_architecture_boundaries.py",
    "tests/test_environment_locks.py",
    "tests/v2/test_import_isolation.py",
    "tests/research/lightgen/test_research_run_control.py",
    "tests/research/lightgen/test_joint_structural_semantic_continuation.py",
    "tests/test_strict_optical_fields.py",
    "tests/research/lightgen/test_canonical_semantics.py",
    "tests/research/lightgen/test_lightgen_2d_chen_bogt_complex_ols_training.py",
    "tests/research/lightgen/test_lightgen_2d_400_baseline_preflight.py",
)
RUFF_PATHS = (
    "src/ldm_project/v2/application/research/shared/research_run_control.py",
    "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
    "scripts/research/lightgen/execute_lightgen_2d_chen_bogt_complex_ols_training_v1.py",
    "scripts/verify_environment_locks.py",
    "scripts/run_windows_cpu_checks.py",
    "scripts/research/lightgen/diagnose_lightgen_2d_cmos_metric_audit_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_400_mnist_semantic_generation_baseline_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_chen_bogt_phase_completion_repair_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_local_compute_bounded_joint_canary_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_class_balanced_joint_canary_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_evaluator_agnostic_structural_image_alignment_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_evaluator_robust_semantic_training_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_attribute_canonical_semantic_realization_failure_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_attribute_cross_evaluator_semantic_divergence_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_attribute_step9000_structural_classifier_semantic_tradeoff_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_minimal_near_identity_nonlinear_generator_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_generator_semantic_controllability_attribution_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_generator_target_image_semantic_alignment_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_train_canonical_semantic_alignment_with_independent_confirmation_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_attribute_post3000_generator_semantic_realization_ceiling_v1.py",
    "scripts/research/lightgen/execute_lightgen_2d_joint_structural_semantic_continuation_v1.py",
    "scripts/research/capacity/execute_lightgen_2d_minimal_generator_capacity_expansion_v1.py",
    "tests/test_architecture_boundaries.py",
    "tests/test_environment_locks.py",
    "tests/research/lightgen/test_research_run_control.py",
    "tests/research/lightgen/test_local_compute_bounded_joint_canary_executor.py",
    "tests/research/lightgen/test_joint_structural_semantic_continuation.py",
    "tests/research/lightgen/test_lightgen_2d_chen_bogt_complex_ols_training.py",
    "tests/research/lightgen/test_lightgen_2d_400_baseline_preflight.py",
)


def run(command: list[str]) -> None:
    print("$ " + subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> int:
    if not Path(sys.executable).is_file():
        raise SystemExit("the current Python executable is unavailable")
    runner_temp = os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    with tempfile.TemporaryDirectory(prefix="ldm-project-pytest-", dir=runner_temp) as test_temp:
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *PYTEST_PATHS,
                "--basetemp",
                test_temp,
            ]
        )
    run([sys.executable, "-m", "ruff", "check", "--select", "E4,E9,F821", *RUFF_PATHS])
    run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "src",
            "tests",
            "scripts",
            "train.py",
        ]
    )
    print("Windows CPU Actions-equivalent checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
