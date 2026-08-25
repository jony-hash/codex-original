from __future__ import annotations

import argparse


def register_audit(root: argparse._SubParsersAction) -> None:
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
    for name in ("config", "dataset_root", "evaluator_checkpoint", "acceptance_measurement_evidence"):
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
    objective_calibration = audit.add_parser("conditional-objective-weight-calibration")
    objective_calibration.add_argument("--config", required=True)
    for name in ("dataset_root", "evaluator_checkpoint", "initialization_checkpoint", "injection_evidence"):
        objective_calibration.add_argument("--" + name.replace("_", "-"), required=True)
    spatial_electronic_preflight = audit.add_parser("spatial-electronic-preflight")
    for name in ("config", "dataset_root", "evaluator_checkpoint", "output"):
        spatial_electronic_preflight.add_argument("--" + name.replace("_", "-"), required=True)
    spatial_electronic_admit = audit.add_parser("admit-spatial-electronic")
    for name in ("config", "dataset_root", "evaluator_checkpoint", "secondary_evaluator_checkpoint", "electronic_checkpoint", "training_evidence", "output"):
        spatial_electronic_admit.add_argument("--" + name.replace("_", "-"), required=True)
    spatial_quadrature_preflight = audit.add_parser("spatial-quadrature-preflight")
    for name in ("config", "dataset_root", "evaluator_checkpoint", "electronic_checkpoint", "electronic_admission_evidence", "output"):
        spatial_quadrature_preflight.add_argument("--" + name.replace("_", "-"), required=True)
