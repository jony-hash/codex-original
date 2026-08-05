"""Side-effect-free v2 command-line schema.

This module intentionally imports only :mod:`argparse`; execution routing lives
in :mod:`.cli` and implementation modules are imported after parsing.
"""

from __future__ import annotations

import argparse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ldm_project.v2")
    root = parser.add_subparsers(dest="domain", required=True)
    research = root.add_parser("research", help="diagnostic-only pre-registered research programs")
    research_programs = research.add_subparsers(dest="research_program", required=True)
    capacity_map = research_programs.add_parser("capacity-map")
    capacity_actions = capacity_map.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "screen", "train-smoke", "confirm", "summarize", "diagnose", "qualify-shape", "closeout"):
        command = capacity_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument("--dataset", choices=("mnist", "controlled_shapes", "fashion_mnist"), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cpu" if action == "closeout" else "cuda")
        if action not in {"validate", "summarize", "qualify-shape", "closeout"}:
            command.add_argument("--variant", required=True)
        else:
            command.add_argument("--variant")
        if action in {"screen", "train-smoke", "confirm", "diagnose", "qualify-shape"}:
            command.add_argument("--seed", type=int, default=7319 if action == "qualify-shape" else 1234)
        if action == "screen":
            command.add_argument("--preflight-only", action="store_true")
            command.add_argument("--stage", choices=("audit", "probe"), default="audit")
            command.add_argument("--prior-evidence")
        if action in {"train-smoke", "confirm"}:
            command.add_argument("--prior-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "diagnose":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--source-evidence", required=True)
            command.add_argument("--shape-evaluator-checkpoint")
            command.add_argument("--shape-qualification-evidence")
        if action == "summarize":
            command.add_argument("--evidence-root")
        if action == "closeout":
            command.add_argument("--comparison-config", required=True)
            command.add_argument("--evidence-root", required=True)
    o0 = research_programs.add_parser("o0-feasibility")
    o0_actions = o0.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "preflight", "controls", "probe", "smoke", "confirm", "final", "summarize", "closeout"):
        command = o0_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument("--dataset", choices=("mnist",), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--candidate", choices=("D0", "C0", "C1", "C2", "C3"), required=action in {"preflight", "probe", "smoke", "confirm", "final"})
        if action in {"preflight", "probe", "smoke", "confirm"}:
            command.add_argument("--seed", type=int, default=1234)
        if action in {"probe", "smoke", "confirm"}:
            command.add_argument("--prior-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "final":
            command.add_argument("--checkpoint", action="append", required=True)
            command.add_argument("--source-evidence", action="append", required=True)
        if action in {"summarize", "closeout"}:
            command.add_argument("--evidence-root", required=True)
    ident = research_programs.add_parser("o0-identifiability")
    ident_actions = ident.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "preflight", "audit-update", "probe", "summarize", "closeout"):
        command = ident_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument("--dataset", choices=("mnist",), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--candidate", choices=("D0",), default="D0")
        if action in {"preflight", "audit-update", "probe"}:
            command.add_argument("--seed", type=int, default=1234)
        if action == "probe":
            command.add_argument("--prior-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action in {"summarize", "closeout"}:
            command.add_argument("--evidence-root", required=True)
    teacher = root.add_parser("teacher").add_subparsers(dest="action", required=True)
    for action, stage in (
        ("train-first-stage", "first_stage"),
        ("train-diffusion", "diffusion"),
    ):
        command = teacher.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--legacy-train-config", required=True)
    export = teacher.add_parser("export-targets")
    for name in (
        "config",
        "legacy_train_config",
        "teacher_manifest",
        "output",
        "target_manifest",
    ):
        export.add_argument("--" + name.replace("_", "-"), required=True)
    manifest = teacher.add_parser("manifest")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--split", default="train")
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
    evaluator.add_argument("--dataset-root", required=True)
    evaluator.add_argument("--epochs", type=int, default=20)
    evaluator.add_argument("--batch-size", type=int, default=128)
    qualify = student.add_parser("qualify-evaluator")
    qualify.add_argument("--config", required=True)
    qualify.add_argument("--dataset-root", required=True)
    qualify.add_argument("--evaluator-checkpoint", required=True)
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
    audit = root.add_parser("audit").add_subparsers(dest="action", required=True)
    conditional_observability_parser = audit.add_parser("conditional-observability")
    conditional_observability_parser.add_argument("--config", required=True)
    for name in (
        "checkpoint",
        "injection_evidence",
        "dataset_root",
        "evaluator_checkpoint",
    ):
        conditional_observability_parser.add_argument(
            "--" + name.replace("_", "-"), required=False
        )
    conditional_injection_parser = audit.add_parser("conditional-injection-admission")
    for name in ("config", "dataset_root", "evaluator_checkpoint"):
        conditional_injection_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    conditional_control_parser = audit.add_parser("conditional-control-isolation")
    conditional_control_parser.add_argument("--config", required=True)
    conditional_control_parser.add_argument("--dataset-root", required=True)
    conditional_control_parser.add_argument("--evaluator-checkpoint", required=True)
    conditional_control_parser.add_argument("--checkpoint", required=False)
    conditional_control_parser.add_argument("--injection-evidence", required=False)
    conditional_control_parser.add_argument("--observability-evidence", required=False)
    conditional_control_parser.add_argument("--output")
    conditional_localization_parser = audit.add_parser(
        "conditional-semantic-localization"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "injection_evidence",
        "observability_evidence",
        "isolation_evidence",
    ):
        conditional_localization_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    completion_parser = audit.add_parser("conditional-training-completion")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "training_evidence",
    ):
        completion_parser.add_argument("--" + name.replace("_", "-"), required=True)
    diversity_observability_parser = audit.add_parser(
        "conditional-diversity-observability"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "training_completion_evidence",
    ):
        diversity_observability_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    alignment_admission_parser = audit.add_parser(
        "conditional-diversity-alignment-admission"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "failed_recovery_evidence",
        "failed_recovery_history",
    ):
        alignment_admission_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    reanchor_parser = audit.add_parser("conditional-identity-reanchor")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "aligned_smoke_evidence",
    ):
        reanchor_parser.add_argument("--" + name.replace("_", "-"), required=True)
    reanchor_obs_parser = audit.add_parser("conditional-reanchor-observability")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "reanchor_admission_evidence",
    ):
        reanchor_obs_parser.add_argument("--" + name.replace("_", "-"), required=True)
    conflict_parser = audit.add_parser("conditional-shared-mode-conflict")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "reanchor_smoke_evidence",
    ):
        conflict_parser.add_argument("--" + name.replace("_", "-"), required=True)
    class_mode_admission_parser = audit.add_parser("conditional-class-mode-admission")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "conflict_evidence",
    ):
        class_mode_admission_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    class_mode_obs_parser = audit.add_parser("conditional-class-mode-observability")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "admission_evidence",
    ):
        class_mode_obs_parser.add_argument("--" + name.replace("_", "-"), required=True)
    parity_conflict_parser = audit.add_parser("conditional-parity-conflict")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "failed_smoke_evidence",
    ):
        parity_conflict_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    parity_admission_parser = audit.add_parser("conditional-parity-mode-admission")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "conflict_evidence",
    ):
        parity_admission_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    parity_obs_parser = audit.add_parser("conditional-parity-observability")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "admission_evidence",
    ):
        parity_obs_parser.add_argument("--" + name.replace("_", "-"), required=True)
    tangent_parser = audit.add_parser("conditional-semantic-tangent")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "class_mode_admission_evidence",
        "failed_class_mode_smoke_evidence",
        "failed_parity_evidence",
    ):
        tangent_parser.add_argument("--" + name.replace("_", "-"), required=True)
    multi_capacity_parser = audit.add_parser("conditional-multi-anchor-capacity")
    for name in ("config", "dataset_root", "evaluator_checkpoint"):
        multi_capacity_parser.add_argument("--" + name.replace("_", "-"), required=True)
    multi_admission_parser = audit.add_parser("conditional-multi-anchor-admission")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "semantic_checkpoint",
        "capacity_evidence",
    ):
        multi_admission_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    multi_obs_parser = audit.add_parser("conditional-multi-anchor-observability")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "admission_evidence",
    ):
        multi_obs_parser.add_argument("--" + name.replace("_", "-"), required=True)
    continuous_capacity_parser = audit.add_parser(
        "conditional-continuous-mode-capacity"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "semantic_checkpoint",
    ):
        continuous_capacity_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    continuous_reachability_parser = audit.add_parser(
        "conditional-continuous-mode-reachability"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "semantic_checkpoint",
        "capacity_evidence",
    ):
        continuous_reachability_parser.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    pca_transport_admission = audit.add_parser("conditional-pca-transport-admission")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "semantic_checkpoint",
        "safe_scale_evidence",
    ):
        pca_transport_admission.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    pca_transport_observability = audit.add_parser(
        "conditional-pca-transport-observability"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "admission_evidence",
    ):
        pca_transport_observability.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    pca_revalidation = audit.add_parser("conditional-pca-transport-smoke-revalidation")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "failed_smoke_evidence",
        "smoke_history",
        "smoke_run",
    ):
        pca_revalidation.add_argument("--" + name.replace("_", "-"), required=True)
    feasibility = audit.add_parser("conditional-optical-generation-feasibility")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "revalidation_evidence",
    ):
        feasibility.add_argument("--" + name.replace("_", "-"), required=True)
    short_target = audit.add_parser("conditional-short-spacing-target-admission")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "semantic_checkpoint",
    ):
        short_target.add_argument("--" + name.replace("_", "-"), required=True)
    short_optical = audit.add_parser("conditional-short-spacing-optical-admission")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "target_admission_evidence",
    ):
        short_optical.add_argument("--" + name.replace("_", "-"), required=True)
    short_supersession = audit.add_parser(
        "conditional-short-spacing-optical-supersession"
    )
    for name in ("config", "old_optical_evidence"):
        short_supersession.add_argument("--" + name.replace("_", "-"), required=True)
    short_obs = audit.add_parser("conditional-short-spacing-observability")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "initialization_checkpoint",
        "optical_admission_evidence",
    ):
        short_obs.add_argument("--" + name.replace("_", "-"), required=True)
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
    formal_admission = audit.add_parser("conditional-formal-spacing-admission")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "target_admission_evidence",
        "electronic_initialization",
    ):
        formal_admission.add_argument("--" + name.replace("_", "-"), required=True)
    formal_tangent = audit.add_parser("conditional-formal-spacing-semantic-tangent")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "admission_evidence",
    ):
        formal_tangent.add_argument("--" + name.replace("_", "-"), required=True)
    formal_observability = audit.add_parser("conditional-formal-spacing-observability")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "tangent_evidence",
    ):
        formal_observability.add_argument("--" + name.replace("_", "-"), required=True)
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
    near_equivalence = audit.add_parser("conditional-near-conjugate-relay-equivalence")
    near_equivalence.add_argument("--config", required=True)
    near_admission = audit.add_parser("conditional-near-conjugate-admission")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "relay_equivalence_evidence",
        "target_admission_evidence",
        "electronic_initialization",
    ):
        near_admission.add_argument("--" + name.replace("_", "-"), required=True)
    near_tangent = audit.add_parser("conditional-near-conjugate-semantic-tangent")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "admission_evidence",
    ):
        near_tangent.add_argument("--" + name.replace("_", "-"), required=True)
    near_obs = audit.add_parser("conditional-near-conjugate-observability")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "tangent_evidence",
    ):
        near_obs.add_argument("--" + name.replace("_", "-"), required=True)
    near_replay = audit.add_parser("conditional-near-conjugate-smoke-replay")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "smoke_evidence",
        "history",
        "initialization_checkpoint",
        "diagnostic_final",
        "observability_evidence",
    ):
        near_replay.add_argument("--" + name.replace("_", "-"), required=True)
    near_distribution_tangent = audit.add_parser(
        "conditional-near-conjugate-phase-distribution-tangent"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "replay_evidence",
        "initialization_checkpoint",
        "diagnostic_final",
    ):
        near_distribution_tangent.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    expanded_capacity = audit.add_parser(
        "conditional-nonlinear-expanded-manifold-capacity"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "electronic_initialization",
    ):
        expanded_capacity.add_argument("--" + name.replace("_", "-"), required=True)
    expanded_admission = audit.add_parser(
        "conditional-nonlinear-expanded-manifold-admission"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "electronic_initialization",
        "capacity_evidence",
        "basis_artifact",
    ):
        expanded_admission.add_argument("--" + name.replace("_", "-"), required=True)
    transport_design = audit.add_parser("conditional-transport-aligned-manifold-design")
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "electronic_initialization",
        "v7_capacity_evidence",
        "v7_basis_artifact",
    ):
        transport_design.add_argument("--" + name.replace("_", "-"), required=True)
    transport_admission = audit.add_parser(
        "conditional-transport-aligned-manifold-admission"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "design_evidence",
        "design_artifact",
    ):
        transport_admission.add_argument("--" + name.replace("_", "-"), required=True)
    transport_tangent = audit.add_parser(
        "conditional-transport-aligned-manifold-tangent"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "admission_evidence",
    ):
        transport_tangent.add_argument("--" + name.replace("_", "-"), required=True)
    phase_capacity_design = audit.add_parser(
        "conditional-transport-aligned-phase-capacity-design"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "electronic_initialization",
        "v7_capacity_evidence",
        "v7_basis_artifact",
        "v71_failure_evidence",
    ):
        phase_capacity_design.add_argument("--" + name.replace("_", "-"), required=True)
    curvature_attribution = audit.add_parser(
        "conditional-transport-curvature-attribution"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "electronic_initialization",
        "v7_capacity_evidence",
        "v7_basis_artifact",
        "v71_failure_evidence",
        "v72_failure_evidence",
    ):
        curvature_attribution.add_argument("--" + name.replace("_", "-"), required=True)
    evaluator_v2 = audit.add_parser("conditional-evaluator-qualification-v2")
    for name in ("config", "dataset_root", "primary_evaluator_checkpoint"):
        evaluator_v2.add_argument("--" + name.replace("_", "-"), required=True)
    evaluator_calibration_v21 = audit.add_parser(
        "conditional-evaluator-calibration-v21"
    )
    for name in (
        "config",
        "dataset_root",
        "primary_evaluator_checkpoint",
        "evaluator_v2_evidence",
    ):
        evaluator_calibration_v21.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    evaluator_v21 = audit.add_parser("conditional-evaluator-qualification-v21")
    for name in (
        "config",
        "dataset_root",
        "primary_evaluator_checkpoint",
        "calibration_evidence",
    ):
        evaluator_v21.add_argument("--" + name.replace("_", "-"), required=True)
    evaluator_curvature = audit.add_parser("conditional-evaluator-curvature-baseline")
    for name in (
        "config",
        "dataset_root",
        "primary_evaluator_checkpoint",
        "smooth_evaluator_checkpoint",
        "evaluator_bundle",
        "evaluator_qualification_evidence",
        "electronic_initialization",
        "v7_capacity_evidence",
        "v7_basis_artifact",
    ):
        evaluator_curvature.add_argument("--" + name.replace("_", "-"), required=True)
    evaluation_qualified_design = audit.add_parser(
        "conditional-evaluation-qualified-transport-design"
    )
    for name in (
        "config",
        "dataset_root",
        "primary_evaluator_checkpoint",
        "smooth_evaluator_checkpoint",
        "evaluator_bundle",
        "evaluator_qualification_evidence",
        "curvature_evidence",
        "electronic_initialization",
        "v7_capacity_evidence",
        "v7_basis_artifact",
    ):
        evaluation_qualified_design.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    evaluation_qualified_admission = audit.add_parser(
        "conditional-evaluation-qualified-transport-admission"
    )
    for name in (
        "config", "dataset_root", "primary_evaluator_checkpoint",
        "smooth_evaluator_checkpoint", "evaluator_bundle",
        "evaluator_qualification_evidence", "design_evidence", "design_artifact",
    ):
        evaluation_qualified_admission.add_argument("--" + name.replace("_", "-"), required=True)
    evaluation_qualified_tangent = audit.add_parser(
        "conditional-evaluation-qualified-transport-tangent"
    )
    for name in (
        "config", "dataset_root", "primary_evaluator_checkpoint",
        "smooth_evaluator_checkpoint", "evaluator_bundle",
        "evaluator_qualification_evidence", "admission_evidence", "checkpoint",
    ):
        evaluation_qualified_tangent.add_argument("--" + name.replace("_", "-"), required=True)
    evaluation_qualified_smoke = student.add_parser("evaluation-qualified-transport-smoke")
    for name in (
        "config", "dataset_root", "primary_evaluator_checkpoint",
        "smooth_evaluator_checkpoint", "evaluator_bundle",
        "evaluator_qualification_evidence", "admission_evidence", "tangent_evidence", "checkpoint",
    ):
        evaluation_qualified_smoke.add_argument("--" + name.replace("_", "-"), required=True)
    phase_capacity_admission = audit.add_parser(
        "conditional-transport-aligned-phase-capacity-admission"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "design_evidence",
        "design_artifact",
    ):
        phase_capacity_admission.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    phase_capacity_tangent = audit.add_parser(
        "conditional-transport-aligned-phase-capacity-tangent"
    )
    for name in (
        "config",
        "dataset_root",
        "evaluator_checkpoint",
        "checkpoint",
        "admission_evidence",
    ):
        phase_capacity_tangent.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
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
    memorization = audit.add_parser("memorization")
    for name in (
        "config",
        "candidate_evidence",
        "candidate_dataset",
        "dataset_root",
        "evaluator_checkpoint",
    ):
        memorization.add_argument("--" + name.replace("_", "-"), required=True)
    capacity = audit.add_parser("capacity")
    for name in ("config", "prototype_evidence", "prototype_checkpoint"):
        capacity.add_argument("--" + name.replace("_", "-"), required=True)
    reachability = audit.add_parser("prototype-reachability")
    for name in (
        "config",
        "clean_target_manifest",
        "failed_calibration_evidence",
        "failed_prototype_checkpoint",
    ):
        reachability.add_argument("--" + name.replace("_", "-"), required=True)
    topology = audit.add_parser("prototype-topology")
    for name in ("config", "clean_target_manifest", "prototype_checkpoint"):
        topology.add_argument("--" + name.replace("_", "-"), required=True)
    geometry = audit.add_parser("prototype-geometry-upper-bound")
    for name in (
        "config",
        "clean_target_manifest",
        "shape_evidence",
        "shape_checkpoint",
    ):
        geometry.add_argument("--" + name.replace("_", "-"), required=True)
    oracle = audit.add_parser("prototype-control-oracle")
    for name in (
        "config",
        "clean_target_manifest",
        "geometry_evidence",
        "shape_checkpoint",
    ):
        oracle.add_argument("--" + name.replace("_", "-"), required=True)
    replay = audit.add_parser("direct-template-replay")
    for name in (
        "config",
        "clean_target_manifest",
        "control_evidence",
        "control_report",
        "amplitude_commands",
        "oracle_raw_cmos",
        "shape_checkpoint",
    ):
        replay.add_argument("--" + name.replace("_", "-"), required=True)
    admit_replay = audit.add_parser("admit-direct-template-replay")
    for name in (
        "config",
        "replay_evidence",
        "command_manifest",
        "replay_raw_cmos",
        "replay_preview",
        "clean_target_manifest",
        "shape_checkpoint",
        "manual_review",
    ):
        admit_replay.add_argument("--" + name.replace("_", "-"), required=True)
    freeze_template = audit.add_parser("freeze-direct-template-prototype")
    for name in (
        "config",
        "admission_evidence",
        "replay_evidence",
        "command_manifest",
        "commands",
        "raw_cmos",
        "clean_target_manifest",
        "shape_checkpoint",
    ):
        freeze_template.add_argument("--" + name.replace("_", "-"), required=True)
    local_capacity = audit.add_parser("direct-template-local-capacity")
    for name in (
        "config",
        "prototype_evidence",
        "prototype_artifact",
        "clean_target_manifest",
        "shape_checkpoint",
    ):
        local_capacity.add_argument("--" + name.replace("_", "-"), required=True)
    lightgen_gap = audit.add_parser("lightgen-contract-gap")
    for name in ("config", "reference_contract"):
        lightgen_gap.add_argument("--" + name.replace("_", "-"), required=True)
    lightgen_observability = audit.add_parser("lightgen-ols-observability")
    for name in ("config", "reference_contract"):
        lightgen_observability.add_argument(
            "--" + name.replace("_", "-"), required=True
        )
    evidence = root.add_parser("evidence").add_subparsers(dest="action", required=True)
    summarize = evidence.add_parser("summarize")
    summarize.add_argument("--run", required=True)
    rebind = evidence.add_parser("rebind-manifest")
    rebind.add_argument("--config", required=True)
    rebind.add_argument(
        "--kind", choices=("clean-target", "teacher-target"), required=True
    )
    rebind.add_argument("--source-manifest", required=True)
    rebind.add_argument("--output", required=True)
    rebind.add_argument("--dataset", default="")
    rebind.add_argument("--teacher-manifest", default="")
    rebind.add_argument("--legacy-train-config", default="")
    supersede = evidence.add_parser("supersede-reachability")
    supersede.add_argument("--config", required=True)
    supersede.add_argument("--old-evidence", required=True)
    return parser
