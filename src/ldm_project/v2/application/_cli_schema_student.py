from __future__ import annotations

import argparse


def register_student(root: argparse._SubParsersAction) -> None:
    student = root.add_parser("student").add_subparsers(dest="action", required=True)
    calibrate = student.add_parser("calibrate-prototypes")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--clean-target-manifest", required=True)
    calibrate.add_argument("--reachability-evidence")
    clean = student.add_parser("export-clean-targets")
    for name in ("config", "dataset_root", "output", "manifest"):
        clean.add_argument("--" + name.replace("_", "-"), required=True)
    admit = student.add_parser("admit-prototypes")
    for name in (
        "config",
        "calibration_evidence",
        "prototype_checkpoint",
        "prototype_raw_cmos",
        "prototype_preview",
        "measurements",
        "clean_target_manifest",
        "manual_review",
    ):
        admit.add_argument("--" + name.replace("_", "-"), required=True)
    train = student.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--contract")
    train.add_argument("--run-root")
    train.add_argument("--resume-checkpoint")
    for name in (
        "prototype_evidence",
        "capacity_evidence",
        "code_basis",
        "prototype_checkpoint",
        "teacher_target_manifest",
        "dataset_root",
        "evaluator_checkpoint",
        "observability_evidence",
        "controllability_evidence",
        "semantic_recovery_evidence",
        "initialization_checkpoint",
        "injection_evidence",
    ):
        train.add_argument("--" + name.replace("_", "-"))
    sample = student.add_parser("sample")
    sample.add_argument("--config", required=True)
    for name in ("code_basis", "checkpoint", "codes", "labels", "output"):
        sample.add_argument(
            "--" + name.replace("_", "-"),
            required=name in {"checkpoint", "codes", "labels", "output"},
        )
    evaluator = student.add_parser("train-evaluator")
    evaluator.add_argument("--config", required=True)
    evaluator.add_argument("--contract")
    evaluator.add_argument("--run-root")
    evaluator.add_argument("--dataset-root", required=True)
    evaluator.add_argument("--epochs", type=int, default=20)
    evaluator.add_argument("--batch-size", type=int, default=128)
    qualify = student.add_parser("qualify-evaluator")
    qualify.add_argument("--config", required=True)
    qualify.add_argument("--dataset-root", required=True)
    qualify.add_argument("--evaluator-checkpoint", required=True)
    acceptance_evaluator = student.add_parser("train-acceptance-evaluator")
    acceptance_evaluator.add_argument("--config", required=True)
    acceptance_evaluator.add_argument("--contract")
    acceptance_evaluator.add_argument("--run-root")
    acceptance_evaluator.add_argument("--dataset-root", required=True)
    acceptance_evaluator.add_argument("--epochs", type=int, default=10)
    acceptance_evaluator.add_argument("--batch-size", type=int, default=256)
    qualify_measurement = student.add_parser("qualify-acceptance-measurement")
    for name in ("config", "dataset_root", "evaluator_checkpoint", "acceptance_evaluator_checkpoint", "acceptance_evaluator_training_evidence"):
        qualify_measurement.add_argument("--" + name.replace("_", "-"), required=True)
    objective_screen = student.add_parser("objective-alignment-screen")
    objective_screen.add_argument("--config", required=True)
    for name in ("dataset_root", "evaluator_checkpoint", "acceptance_evaluator_checkpoint", "acceptance_measurement_bundle", "acceptance_measurement_evidence", "initialization_checkpoint", "injection_evidence", "weight_calibration", "weight_calibration_evidence", "warmup_checkpoint"):
        objective_screen.add_argument("--" + name.replace("_", "-"), required=True)
    objective_screen.add_argument("--resume-checkpoint")
    objective_screen.add_argument("--contract")
    objective_screen.add_argument("--run-root")
    objective_confirm = student.add_parser("objective-alignment-confirm")
    objective_confirm.add_argument("--config", required=True)
    objective_confirm.add_argument("--contract")
    objective_confirm.add_argument("--run-root")
    for name in ("dataset_root", "evaluator_checkpoint", "acceptance_evaluator_checkpoint", "acceptance_measurement_bundle", "acceptance_measurement_evidence", "initialization_checkpoint", "injection_evidence", "screen_evidence"):
        objective_confirm.add_argument("--" + name.replace("_", "-"), required=True)
    controllability = student.add_parser("controllability-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "observability_evidence",
        "checkpoint",
        "injection_evidence",
        "isolation_evidence",
        "semantic_localization_evidence",
    ):
        controllability.add_argument(
            "--" + name.replace("_", "-"),
            required=name
            in {
                "config",
                "dataset_root",
                "evaluator_checkpoint",
                "observability_evidence",
            },
        )
    controllability.add_argument("--steps", type=int, default=100)
    recovery = student.add_parser("semantic-recovery-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "initialization_checkpoint",
        "injection_evidence",
        "observability_evidence",
        "isolation_evidence",
        "semantic_localization_evidence",
        "failed_controllability_evidence",
    ):
        recovery.add_argument("--" + name.replace("_", "-"), required=True)
    diversity_smoke = student.add_parser("diversity-recovery-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "training_completion_evidence",
        "diversity_observability_evidence",
    ):
        diversity_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    diversity_train = student.add_parser("train-diversity-recovery")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "diversity_recovery_evidence",
    ):
        diversity_train.add_argument("--" + name.replace("_", "-"), required=True)
    aligned_smoke = student.add_parser("diversity-aligned-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "alignment_admission_evidence",
    ):
        aligned_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    aligned_train = student.add_parser("train-diversity-aligned")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "aligned_smoke_evidence",
    ):
        aligned_train.add_argument("--" + name.replace("_", "-"), required=True)
    reanchor_smoke = student.add_parser("identity-reanchor-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "observability_evidence",
    ):
        reanchor_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    reanchor_train = student.add_parser("train-identity-reanchor")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "smoke_evidence",
    ):
        reanchor_train.add_argument("--" + name.replace("_", "-"), required=True)
    class_mode_smoke = student.add_parser("class-mode-recovery-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "observability_evidence",
    ):
        class_mode_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    class_mode_train = student.add_parser("train-class-modes")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "smoke_evidence",
    ):
        class_mode_train.add_argument("--" + name.replace("_", "-"), required=True)
    parity_smoke = student.add_parser("parity-stabilizer-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "observability_evidence",
    ):
        parity_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    parity_train = student.add_parser("train-parity-stabilizer")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "smoke_evidence",
    ):
        parity_train.add_argument("--" + name.replace("_", "-"), required=True)
    multi_anchor_smoke_parser = student.add_parser("multi-anchor-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "observability_evidence",
    ):
        multi_anchor_smoke_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    multi_anchor_train = student.add_parser("train-multi-anchor")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "smoke_evidence",
    ):
        multi_anchor_train.add_argument("--" + name.replace("_", "-"), required=True)
    pca_transport_smoke_parser = student.add_parser("pca-transport-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "observability_evidence",
    ):
        pca_transport_smoke_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    pca_transport_train = student.add_parser("train-pca-transport")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "smoke_evidence",
    ):
        pca_transport_train.add_argument("--" + name.replace("_", "-"), required=True)
    export_dataset = student.add_parser("export-dataset")
    for name in ("config", "checkpoint", "acceptance_evidence"):
        export_dataset.add_argument("--" + name.replace("_", "-"), required=True)
    export_dataset.add_argument("--samples-per-label", type=int)
    export_dataset.add_argument("--seed", type=int, default=1234)
    admit_conditional = student.add_parser("admit-conditional-evaluation")
    for name in (
        "config",
        "evaluation_evidence",
        "evaluation_raw_cmos",
        "evaluation_labels",
        "evaluation_preview",
        "manual_review",
    ):
        admit_conditional.add_argument("--" + name.replace("_", "-"), required=True)
    utility = student.add_parser("evaluate-utility")
    for name in ("config", "candidate_evidence", "candidate_dataset", "dataset_root"):
        utility.add_argument("--" + name.replace("_", "-"), required=True)
    admit_dataset_parser = student.add_parser("admit-dataset")
    for name in (
        "config",
        "candidate_dataset",
        "generation_admission_evidence",
        "memorization_evidence",
        "utility_evidence",
    ):
        admit_dataset_parser.add_argument("--" + name.replace("_", "-"), required=True)
    student_memorization = student.add_parser("audit-memorization")
    for name in (
        "config",
        "candidate_evidence",
        "candidate_dataset",
        "dataset_root",
        "evaluator_checkpoint",
    ):
        student_memorization.add_argument("--" + name.replace("_", "-"), required=True)
    preflight = student.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    evaluate = student.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    for name in (
        "training_evidence",
        "code_basis",
        "checkpoint",
        "teacher_target_manifest",
        "output",
        "dataset_root",
        "evaluator_checkpoint",
    ):
        evaluate.add_argument("--" + name.replace("_", "-"))
    evaluate.add_argument("--acceptance-evaluator-checkpoint")
    evaluate.add_argument("--acceptance-measurement-bundle")
    evaluate.add_argument("--acceptance-measurement-evidence")
    admit_evaluation = student.add_parser("admit-evaluation")
    for name in (
        "config",
        "evaluation_evidence",
        "evaluation_raw_cmos",
        "evaluation_preview",
        "manual_review",
        "code_basis",
        "checkpoint",
        "teacher_target_manifest",
        "output",
    ):
        admit_evaluation.add_argument("--" + name.replace("_", "-"), required=True)
    short_smoke = student.add_parser("short-spacing-self-imaging-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "initialization_checkpoint",
        "target_admission_evidence",
        "optical_admission_evidence",
        "observability_evidence",
    ):
        short_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    formal_smoke = student.add_parser("formal-spacing-self-imaging-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "observability_evidence",
    ):
        formal_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    formal_train = student.add_parser("train-formal-spacing-self-imaging")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "smoke_evidence",
    ):
        formal_train.add_argument("--" + name.replace("_", "-"), required=True)
    evaluation_qualified_smoke = student.add_parser("evaluation-qualified-transport-smoke")
    for name in (
        "config", "dataset_root", "primary_evaluator_checkpoint",
        "smooth_evaluator_checkpoint", "evaluator_bundle",
        "evaluator_qualification_evidence", "admission_evidence", "tangent_evidence", "checkpoint",
    ):
        evaluation_qualified_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    near_smoke = student.add_parser("near-conjugate-relay-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "observability_evidence",
    ):
        near_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    near_train = student.add_parser("train-near-conjugate-relay")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "smoke_evidence",
    ):
        near_train.add_argument("--" + name.replace("_", "-"), required=True)
    transport_smoke = student.add_parser("transport-aligned-manifold-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "tangent_evidence",
    ):
        transport_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    phase_capacity_smoke = student.add_parser("transport-aligned-phase-capacity-smoke")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "tangent_evidence",
    ):
        phase_capacity_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    transport_train = student.add_parser("train-transport-aligned-manifold")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "smoke_evidence",
    ):
        transport_train.add_argument("--" + name.replace("_", "-"), required=True)
