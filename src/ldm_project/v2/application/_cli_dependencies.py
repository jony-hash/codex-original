# ruff: noqa: F401
"""Deferred execution dependencies for the compatibility v2 CLI.

Parser construction must remain side-effect free: importing a command-line
module must not initialize legacy teachers, capacity artifacts, candidate
generators, or diagnostic optics.  The legacy command implementations still
share the historic global names in :mod:`.cli`; ``install`` populates those
names only after argparse selected a command.
"""

from __future__ import annotations

from typing import Any


def install(namespace: dict[str, Any]) -> None:
    from ..config import LightGenSystemConfig, V2Config, load_config
    from ..contracts import ComplexField, OperationProvenance, PlaneSpec
    from ..evidence import (
        EvidenceContextV2,
        EvidenceRecordV2,
        GateResult,
        atomic_torch_save,
        atomic_write_json,
        canonical_sha256,
        file_sha256,
        load_manual_review,
        require_evidence_action,
        tensor_artifact_ref,
        tensor_sha256,
    )
    from ..optics import (
        FormalAmplitudeFiniteCropIdentitySystem,
        LightGenAmplitudeFiberOLSSystemV2,
        reference_project_single_mode_fibers,
    )
    from ..lightgen_contract import LightGenReferenceContractV2
    from ..optics.d2nn import FiniteCropPhaseStack
    from ..optics.devices import identity_ols
    from ..optics.relays import CoherentImagingRelay
    from ..student.admission import (
        CapacityMetrics,
        PrototypeMetrics,
        capacity_gate,
        prototype_gate,
        prototype_measurement_gate,
    )
    from ..student.capacity import (
        CodeBasisArtifactV2,
        build_code_basis,
        capacity_metrics,
        finite_difference_response_matrices,
        replay_capacity,
        supported_dct_candidates,
    )
    from ..student.checkpoints import (
        CheckpointManifestV2,
        load_exact_checkpoint,
        load_prototype_warm_start,
        save_exact_checkpoint,
        state_content_hash,
        state_signature,
    )
    from ..student.clean_targets import CleanTargetManifestV2, build_clean_targets
    from ..student.direct_template import DirectTemplatePrototypeArtifactV2
    from ..student.evaluation import (
        CodeAcceptanceMetrics,
        acceptance_metrics,
        code_acceptance_gate,
        paired_advantage_lower_bound,
    )
    from ..student.generator import LabelPrototypeBasisGenerator
    from ..student.metrics import (
        background_speckle_ratio,
        correlation,
        foreground_dice,
        foreground_hole_ratio,
        high_saturation_ratio,
        prototype_correlation_matrix,
        prototype_margins,
    )
    from ..student.review_artifacts import write_review_artifacts
    from ..student.training import (
        _differentiable_correlation,
        calibrate_prototypes,
        train_code_conditioned,
    )
    from ..teacher import (
        TeacherPairManifestV2,
        TeacherTargetManifestV2,
        build_teacher_pair_manifest,
        build_teacher_target_manifest,
    )
    from .runs import V2Run
    from .conditional_admission import (
        conditional_control_isolation,
        conditional_injection_admission,
        conditional_observability,
        conditional_semantic_localization,
        controllability_smoke,
        qualify_evaluator,
        semantic_recovery_smoke,
        train_evaluator,
        conditional_training_completion,
        conditional_diversity_observability,
        conditional_diversity_alignment_admission,
        conditional_identity_reanchor,
        conditional_reanchor_observability,
        conditional_shared_mode_conflict,
        conditional_class_mode_admission,
        conditional_class_mode_observability,
        conditional_parity_conflict,
        conditional_parity_mode_admission,
        conditional_parity_observability,
        conditional_semantic_tangent,
    )
    from .conditional_training import (
        admit_conditional_evaluation,
        conditional_evaluate,
        conditional_train,
        sample_conditional,
        diversity_recovery_smoke,
        train_diversity_recovery,
        diversity_aligned_smoke,
        train_diversity_aligned,
        identity_reanchor_smoke,
        train_identity_reanchor,
        class_mode_recovery_smoke,
        train_class_modes,
        parity_stabilizer_smoke,
        train_parity_stabilizer,
    )
    from .conditional_runtime import (
        conditional_multi_anchor_capacity,
        conditional_multi_anchor_admission,
        conditional_multi_anchor_observability,
        multi_anchor_smoke,
        train_multi_anchor,
    )
    from .conditional_dataset import (
        admit_dataset,
        audit_memorization,
        evaluate_utility,
        export_conditional_dataset,
    )
    from .diagnostic.continuous_audit import conditional_continuous_mode_capacity
    from .diagnostic.continuous_reachability import conditional_continuous_mode_reachability
    from .diagnostic.pca_transport import (
        conditional_pca_transport_admission,
        conditional_pca_transport_observability,
        conditional_pca_transport_smoke_revalidation,
        pca_transport_smoke,
        train_pca_transport,
    )
    from .diagnostic.optical_feasibility import conditional_optical_generation_feasibility
    from .diagnostic.short_spacing import (
        conditional_short_spacing_target_admission,
        conditional_short_spacing_optical_admission,
        conditional_short_spacing_observability,
        conditional_short_spacing_evidence_supersession,
        short_spacing_self_imaging_smoke,
    )
    from .diagnostic.formal_spacing import (
        conditional_formal_spacing_admission,
        conditional_formal_spacing_semantic_tangent,
        conditional_formal_spacing_observability,
        formal_spacing_self_imaging_smoke,
        train_formal_spacing_self_imaging,
    )
    from .diagnostic.near_conjugate import (
        conditional_near_conjugate_relay_equivalence,
        conditional_near_conjugate_admission,
        conditional_near_conjugate_semantic_tangent,
        conditional_near_conjugate_observability,
        conditional_near_conjugate_smoke_replay,
        conditional_near_conjugate_phase_distribution_tangent,
        near_conjugate_relay_smoke,
        train_near_conjugate_relay,
    )
    from .diagnostic.transport_aligned_phase_capacity import (
        conditional_transport_aligned_phase_capacity_admission,
        conditional_transport_aligned_phase_capacity_design,
        conditional_transport_aligned_phase_capacity_tangent,
        transport_aligned_phase_capacity_smoke,
    )
    from .diagnostic.transport_curvature import conditional_transport_curvature_attribution
    from .diagnostic.evaluator_qualification_v2 import (
        conditional_evaluator_calibration_v21,
        conditional_evaluator_qualification_v2,
        conditional_evaluator_qualification_v21,
    )
    from .diagnostic.expanded_manifold import (
        conditional_nonlinear_expanded_manifold_admission,
        conditional_nonlinear_expanded_manifold_capacity,
    )
    from .diagnostic.transport_aligned_manifold import (
        conditional_transport_aligned_manifold_admission,
        conditional_transport_aligned_manifold_design,
        conditional_transport_aligned_manifold_tangent,
        train_transport_aligned_manifold,
        transport_aligned_manifold_smoke,
    )

    namespace.update(locals())
