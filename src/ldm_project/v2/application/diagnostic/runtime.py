# ruff: noqa: F401
"""Compatibility runtime for historical v2 conditional candidates.

The formal identity path lives in :mod:`.identity_runtime`.  This module is
kept for legacy command and import compatibility only; it must not be imported
by formal identity CLI handlers.
"""

from __future__ import annotations

import json
import subprocess
import copy
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from ...config import V2Config
from ...contracts import ComplexField
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_conditional_manual_review,
    tensor_sha256,
)
from ...optics import FormalAmplitudeFiniteCropIdentitySystem
from ...student.checkpoints import (
    ConditionalCheckpointManifestV2,
    load_conditional_checkpoint,
    save_conditional_checkpoint,
    state_content_hash,
    state_signature,
)
from ...student.admission import balanced_code_fraction
from ...student.conditional import (
    camera_code_fraction,
    camera_code_fraction_loss,
    conditional_loss,
    conditional_sinkhorn_loss,
    fixed_exposure_view,
    raw_from_generator_batched,
    relative_diversity_loss,
    relative_diversity_ratios,
    log_sinkhorn,
    sinkhorn_cost_view,
    phase_total_variation,
)
from ...student.conditional_acceptance import (
    conditional_coverage_ratio,
    load_candidate_dataset,
    memorization_statistics,
    paired_bootstrap_lower_bound,
)
from ...student.evaluator import (
    FrozenMNISTEvaluator,
    MNISTSplitManifest,
    balanced_batches,
    evaluator_checkpoint_hash,
    load_frozen_evaluator,
    load_local_mnist,
    save_evaluator_checkpoint,
    stratified_mnist_indices,
)
from ...student.identity_generator import IdentityAnchoredConditionalAmplitudeGenerator
from ...evidence.records import require_evidence_action
from ...student.review_artifacts import (
    write_conditional_review_artifacts,
    write_review_artifacts,
)
from .._runtime_shared import (
    build_context as _context,
    carrier_logits as _carrier_logits,
    new_run as _new_run,
    random_codes as _random_codes,
    split_mnist as _split,
    system_signature as _system_signature,
    write_evidence as _write_evidence,
)
from ..runs import V2Run


CONDITIONAL_CONTROL_METRIC_CONTRACT = (
    "sample_centered_anova_v1|sinkhorn_feature_l2_v1|coverage_disjoint_real_ab_v1"
)
# This admission contract is deliberately separate from the immutable loss/
# checkpoint metric contract above.  Existing initialization checkpoints retain
# the latter; new isolation evidence cannot be mistaken for the old gate.
CONDITIONAL_CONTROL_ADMISSION_CONTRACT = "sample_centered_anova_v1|sinkhorn_feature_l2_v1|coverage_disjoint_real_ab_v1|physical_command_semantics_v2"
CONDITIONAL_SEMANTIC_LOCALIZATION_CONTRACT = "identity_semantic_localization_v1|zero_code_vs_formal_gradient_cosine_v1|no_optimizer_state_change_v1"
IDENTITY_ARCHITECTURE = "conditional_identity_logit_amplitude_v2"
CLASS_MODE_ARCHITECTURE = "conditional_class_conditioned_modes_v3"
PARITY_ARCHITECTURE = "conditional_parity_stabilized_modes_v4"
MULTI_ANCHOR_ARCHITECTURE = "conditional_multi_anchor_local_modes_v5"
PCA_TRANSPORT_ARCHITECTURE = "conditional_pca_transport_codesign_v6"
SHORT_SPACING_ARCHITECTURE = "conditional_pca_self_imaging_transport_v6_2"
FORMAL_SPACING_ARCHITECTURE = "conditional_pca_formal_spacing_self_imaging_v6_3"
NEAR_CONJUGATE_ARCHITECTURE = "conditional_pca_near_conjugate_relay_v6_4"
EXPANDED_MANIFOLD_ARCHITECTURE = "conditional_nonlinear_expanded_manifold_v7"
TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE = "conditional_transport_aligned_manifold_v7_1"
TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE = "conditional_transport_aligned_phase_capacity_v7_2"
IDENTITY_GENERATOR_CONTRACT = canonical_sha256(
    {
        "architecture": IDENTITY_ARCHITECTURE,
        "formula": "h=SiLU(Wz*z);r=Wy*h;c=tanh(z+r);A=sigmoid(By+0.5*sum(c*P_hat))",
        "gain": 0.5,
        "bias_free": True,
    }
)
CLASS_MODE_GENERATOR_CONTRACT = canonical_sha256(
    {
        "architecture": CLASS_MODE_ARCHITECTURE,
        "formula": "c=tanh(z);A=sigmoid(By+0.5*sum(c*P_hat_y))",
        "gain": 0.5,
        "class_modes_shape": [10, 8, 200, 200],
    }
)
PARITY_GENERATOR_CONTRACT = canonical_sha256(
    {
        "architecture": PARITY_ARCHITECTURE,
        "formula": "t=tanh(z);rho=1-exp(-sum(t^2));w=t^2/sum(t^2);A=sigmoid(By+0.5*(sum(t*P_hat_y)+rho*sum(w*tanh(Q_y))))",
        "gain": 0.5,
        "class_modes_shape": [10, 8, 200, 200],
        "stabilizer_shape": [10, 8, 200, 200],
    }
)
MULTI_ANCHOR_GENERATOR_CONTRACT = canonical_sha256(
    {
        "architecture": MULTI_ANCHOR_ARCHITECTURE,
        "formula": "m=2*I(z0>0)+I(z1>0);v=z2:8;A=sigmoid(B_y_m+0.25*sum(tanh(v)*P_hat_y_m))",
        "gain": 0.25,
        "anchor_shape": [10, 4, 200, 200],
        "local_modes_shape": [10, 4, 6, 200, 200],
    }
)
PCA_TRANSPORT_GENERATOR_CONTRACT = canonical_sha256(
    {
        "architecture": PCA_TRANSPORT_ARCHITECTURE,
        "formula": "t=tanh(z);P_hat=lowpass20cyclesmm(P);s=.20*sigmoid(g);A=sigmoid(B+sum(s*t*P_hat))",
        "scale_bounds": [0.0, 0.20],
        "initial_scale": 0.05,
        "per_class_l2_budget": 0.30,
    }
)
SHORT_SPACING_GENERATOR_CONTRACT = canonical_sha256(
    {
        "architecture": SHORT_SPACING_ARCHITECTURE,
        "formula": "A=sigmoid(B+sum((.05+.45*sigmoid(g))*tanh(z)*lowpass20(P)))",
        "scale_bounds": [0.05, 0.50],
        "initial_scale": 0.05,
        "self_imaging": True,
    }
)
FORMAL_SPACING_GENERATOR_CONTRACT = canonical_sha256(
    {
        "architecture": FORMAL_SPACING_ARCHITECTURE,
        "formula": "A=sigmoid(B+sum((.05+.45*sigmoid(g))*tanh(z)*lowpass20(P)))",
        "scale_bounds": [0.05, 0.50],
        "initial_scale": 0.05,
        "self_imaging": True,
        "layer_spacing_m": 1.50e-3,
    }
)
NEAR_CONJUGATE_GENERATOR_CONTRACT = canonical_sha256(
    {
        "architecture": NEAR_CONJUGATE_ARCHITECTURE,
        "formula": "A=sigmoid(B+sum((.05+.45*sigmoid(g))*tanh(z)*lowpass20(P)))",
        "scale_bounds": [0.05, 0.50],
        "initial_scale": 0.05,
        "generator_frozen": True,
        "illumination_intensity_transmission": 0.80,
        "near_conjugate_abcd": [-1.0, 0.00025, 0.0, -1.0],
    }
)
TRAINING_SCHEDULE_CONTRACT = canonical_sha256(
    {
        "epochs": 5,
        "logical_batch_size": 50,
        "steps": 5500,
        "seed": 1234,
        "code_seed": "(epoch+1)*100000+batch_number",
    }
)
BALANCED_TRAINING_SCHEDULE = {
    "epochs": 5,
    "logical_batch_size": 50,
    "batch_policy": "balanced_min_class_exhaustion",
    "seed": 1234,
    "code_seed": "(epoch+1)*100000+batch_number",
}
BALANCED_TRAINING_SCHEDULE_CONTRACT = canonical_sha256(BALANCED_TRAINING_SCHEDULE)
DIVERSITY_LOSS_CONTRACT = canonical_sha256(
    {
        "sinkhorn": 1.0,
        "classification": 0.5,
        "relative_diversity": 1.0,
        "saturation": 0.1,
        "mode_orthogonality": 1.0e-3,
        "feature_normalization": "l2",
        "bounds": [0.70, 1.30],
    }
)
DIVERSITY_RECOVERY_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_diversity_recovery_smoke",
        "steps": 400,
        "probes": [0, 100, 200, 400],
        "trainable_groups": ["global_modes", "code_projection", "class_heads"],
    }
)
ALIGNED_DIVERSITY_METRIC_CONTRACT = "conditional_diversity_metric_v2|feature_l2_normalized_median_pdist|raw_cmos_sample_centered_anova"
ALIGNED_DIVERSITY_LOSS_CONTRACT = canonical_sha256(
    {
        "sinkhorn": 1.0,
        "classification": 0.5,
        "relative_diversity": 1.0,
        "camera_fraction": 2.0,
        "saturation": 0.1,
        "mode_orthogonality": 1.0e-3,
        "camera_target": 0.25,
    }
)
ALIGNED_DIVERSITY_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_diversity_aligned_smoke",
        "steps": 400,
        "thresholds": {
            "camera": 0.175,
            "camera_delta_200_400": 0.026,
            "ratio": 0.42,
            "ratio_delta_200_400": 0.10,
        },
        "code_schedule": "zero_plus_two_antithetic_pairs",
    }
)
REANCHOR_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_identity_reanchor_v1",
        "zeroed": ["generator.class_heads"],
        "trainable": ["generator.global_modes"],
        "gain": 0.5,
    }
)
REANCHOR_SMOKE_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_identity_reanchor_smoke",
        "steps": 400,
        "code_schedule": "axis_plus_uniform_antithetic",
        "trainable": ["generator.global_modes"],
    }
)
CLASS_MODE_SMOKE_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_class_mode_recovery_smoke",
        "steps": 400,
        "panels": ["axis", "uniform_1234", "uniform_5678"],
        "trainable": ["generator.class_modes"],
    }
)
PARITY_SMOKE_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_parity_stabilizer_smoke",
        "steps": 400,
        "panels": ["axis", "uniform_1234", "uniform_5678"],
        "trainable": ["generator.semantic_stabilizers"],
        "stop_after_failure": True,
    }
)
MULTI_ANCHOR_SMOKE_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_multi_anchor_smoke",
        "steps": 400,
        "panels": ["anchor_centers", "local_axis", "uniform_1234", "uniform_5678"],
        "trainable": [
            "generator.anchor_logits",
            "generator.local_modes",
            "system.encoder.phase",
            "system.decoder.phase",
        ],
        "stop_after_failure": True,
    }
)
SEMANTIC_TANGENT_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_semantic_tangent_v1",
        "panels": ["axis", "uniform_1234", "uniform_5678"],
        "margin_threshold": 0.5,
        "probe_raw_rms": 0.01,
        "logit_rms": 0.05,
        "phase_rms": 0.005,
        "optimizer": False,
    }
)
SEMANTIC_RECOVERY_STEPS = 400
SEMANTIC_RECOVERY_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_semantic_recovery_smoke",
        "steps": SEMANTIC_RECOVERY_STEPS,
        "batch_seed": 1234,
        "code_seed": "100000+batch_number",
        "initialization_only": True,
    }
)


class ConditionalBundle(nn.Module):
    def __init__(
        self,
        config: V2Config,
        *,
        carrier_logits: torch.Tensor | None = None,
        anchor_logits: torch.Tensor | None = None,
        local_modes: torch.Tensor | None = None,
        class_modes: torch.Tensor | None = None,
        scale_logits: torch.Tensor | None = None,
        manifold_scales: torch.Tensor | None = None,
        transport_mixing: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if (
            config.architecture
            not in {
                "conditional_global_amplitude_v1",
                IDENTITY_ARCHITECTURE,
                CLASS_MODE_ARCHITECTURE,
                PARITY_ARCHITECTURE,
                MULTI_ANCHOR_ARCHITECTURE,
                PCA_TRANSPORT_ARCHITECTURE,
                SHORT_SPACING_ARCHITECTURE,
                FORMAL_SPACING_ARCHITECTURE,
                NEAR_CONJUGATE_ARCHITECTURE,
                EXPANDED_MANIFOLD_ARCHITECTURE,
                TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE,
                TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE,
            }
            or config.conditional_generation is None
        ):
            raise ValueError(
                "conditional bundle requires a supported conditional architecture config"
            )
        settings = config.conditional_generation
        self.system = FormalAmplitudeFiniteCropIdentitySystem(config.system)
        if config.architecture == IDENTITY_ARCHITECTURE:
            generator_type = IdentityAnchoredConditionalAmplitudeGenerator
        else:
            # Candidate generators are intentionally outside the formal import
            # surface.  Historical commands retain the same behavior once they
            # explicitly construct a non-identity architecture.
            from ...student import conditional_generator as legacy_generators

            generator_type = {
                CLASS_MODE_ARCHITECTURE: legacy_generators.ConditionalClassConditionedModesGenerator,
                PARITY_ARCHITECTURE: legacy_generators.ConditionalParityStabilizedModesGenerator,
                MULTI_ANCHOR_ARCHITECTURE: legacy_generators.ConditionalMultiAnchorLocalModesGenerator,
                PCA_TRANSPORT_ARCHITECTURE: legacy_generators.ConditionalPCATransportCodesignGenerator,
                SHORT_SPACING_ARCHITECTURE: legacy_generators.ConditionalPCASelfImagingTransportGenerator,
                FORMAL_SPACING_ARCHITECTURE: legacy_generators.ConditionalPCASelfImagingTransportGenerator,
                NEAR_CONJUGATE_ARCHITECTURE: legacy_generators.ConditionalPCASelfImagingTransportGenerator,
                EXPANDED_MANIFOLD_ARCHITECTURE: legacy_generators.ConditionalNonlinearExpandedManifoldGenerator,
                TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE: legacy_generators.ConditionalTransportAlignedManifoldGenerator,
                TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE: legacy_generators.ConditionalTransportAlignedPhaseCapacityGenerator,
            }.get(config.architecture, legacy_generators.ConditionalGlobalAmplitudeGenerator)
        kwargs = {
            "shape": config.system.slm_shape,
            "code_dim": settings.code_dim,
            "hidden_dim": settings.hidden_dim,
            "command_gain": settings.command_gain,
        }
        if config.architecture == MULTI_ANCHOR_ARCHITECTURE:
            kwargs.update(anchor_logits=anchor_logits, local_modes=local_modes)
        else:
            kwargs["class_carrier_logits"] = carrier_logits
        if config.architecture in {
            PCA_TRANSPORT_ARCHITECTURE,
            SHORT_SPACING_ARCHITECTURE,
            FORMAL_SPACING_ARCHITECTURE,
            NEAR_CONJUGATE_ARCHITECTURE,
        }:
            ctor = (
                generator_type
                if config.architecture == PCA_TRANSPORT_ARCHITECTURE
                else generator_type
            )
            kwargs.update(
                command_gain=ctor.initial_scale,
                pixel_pitch_m=config.system.slm_pixel_pitch_m,
                class_modes=class_modes,
                scale_logits=scale_logits,
            )
        if config.architecture == EXPANDED_MANIFOLD_ARCHITECTURE:
            kwargs.update(class_modes=class_modes, manifold_scales=manifold_scales)
        if config.architecture in {
            TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE,
            TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE,
        }:
            kwargs.update(
                candidate_modes=class_modes,
                manifold_scales=manifold_scales,
                transport_mixing=transport_mixing,
            )
        self.generator = generator_type(**kwargs)


def _require_conditional(config: V2Config) -> None:
    if (
        config.architecture
        not in {
            "conditional_global_amplitude_v1",
            IDENTITY_ARCHITECTURE,
            CLASS_MODE_ARCHITECTURE,
            PARITY_ARCHITECTURE,
            MULTI_ANCHOR_ARCHITECTURE,
            PCA_TRANSPORT_ARCHITECTURE,
            SHORT_SPACING_ARCHITECTURE,
            FORMAL_SPACING_ARCHITECTURE,
            NEAR_CONJUGATE_ARCHITECTURE,
            EXPANDED_MANIFOLD_ARCHITECTURE,
            TRANSPORT_ALIGNED_MANIFOLD_ARCHITECTURE,
            TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE,
        }
        or config.conditional_generation is None
    ):
        raise ValueError("this command requires a supported conditional architecture")


def _require_identity(config: V2Config) -> None:
    _require_conditional(config)
    if config.architecture != IDENTITY_ARCHITECTURE:
        raise ValueError(
            "this command requires architecture: conditional_identity_logit_amplitude_v2"
        )


def _require_class_modes(config: V2Config) -> None:
    _require_conditional(config)
    if config.architecture != CLASS_MODE_ARCHITECTURE:
        raise ValueError(
            "this command requires architecture: conditional_class_conditioned_modes_v3"
        )


def _require_parity(config: V2Config) -> None:
    _require_conditional(config)
    if config.architecture != PARITY_ARCHITECTURE:
        raise ValueError(
            "this command requires architecture: conditional_parity_stabilized_modes_v4"
        )


def _require_multi_anchor(config: V2Config) -> None:
    _require_conditional(config)
    if config.architecture != MULTI_ANCHOR_ARCHITECTURE:
        raise ValueError(
            "this command requires architecture: conditional_multi_anchor_local_modes_v5"
        )


def _require_current_acceptance(config: V2Config) -> None:
    if (
        config.run.evidence_tier == "acceptance"
        and config.architecture != IDENTITY_ARCHITECTURE
    ):
        raise ValueError("legacy v2 architectures cannot authorize current acceptance")


def _load_bundle(
    config: V2Config,
    checkpoint: str,
    *,
    device: torch.device,
    evaluator_hash: str | None = None,
    split_hash: str | None = None,
    allow_balanced_schedule: bool = False,
) -> tuple[ConditionalBundle, ConditionalCheckpointManifestV2]:
    _require_conditional(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
        raise ValueError("invalid conditional checkpoint")
    manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    if (
        manifest.architecture_kind != config.architecture
        or manifest.system_signature != _system_signature(config)
        or manifest.config_hash != canonical_sha256(config.model_dump(mode="json"))
    ):
        raise ValueError(
            "conditional checkpoint belongs to a different architecture/system/config"
        )
    if (
        evaluator_hash is not None
        and manifest.evaluator_checkpoint_hash != evaluator_hash
    ):
        raise ValueError("conditional checkpoint uses another evaluator")
    if split_hash is not None and manifest.dataset_split_manifest_hash != split_hash:
        raise ValueError("conditional checkpoint uses another MNIST split")
    bundle = ConditionalBundle(config).to(device)
    if manifest.architecture_kind == IDENTITY_ARCHITECTURE:
        valid_schedules = {TRAINING_SCHEDULE_CONTRACT}
        if allow_balanced_schedule:
            valid_schedules.add(BALANCED_TRAINING_SCHEDULE_CONTRACT)
        if (
            manifest.generator_contract_hash != IDENTITY_GENERATOR_CONTRACT
            or manifest.metric_contract_hash
            != canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT)
            or manifest.training_schedule_hash not in valid_schedules
            or manifest.initialization_seed != 1234
        ):
            raise ValueError(
                "identity checkpoint has an incomplete or mismatched contract"
            )
    if manifest.architecture_kind == CLASS_MODE_ARCHITECTURE:
        if (
            manifest.generator_contract_hash != CLASS_MODE_GENERATOR_CONTRACT
            or manifest.metric_contract_hash
            != canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT)
            or manifest.training_schedule_hash != BALANCED_TRAINING_SCHEDULE_CONTRACT
            or manifest.initialization_seed != 1234
        ):
            raise ValueError(
                "class-mode checkpoint has an incomplete or mismatched contract"
            )
    if manifest.architecture_kind == PARITY_ARCHITECTURE:
        if (
            manifest.generator_contract_hash != PARITY_GENERATOR_CONTRACT
            or manifest.metric_contract_hash
            != canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT)
            or manifest.training_schedule_hash != BALANCED_TRAINING_SCHEDULE_CONTRACT
            or manifest.initialization_seed != 1234
        ):
            raise ValueError(
                "parity checkpoint has an incomplete or mismatched contract"
            )
    if manifest.architecture_kind == MULTI_ANCHOR_ARCHITECTURE:
        if (
            manifest.generator_contract_hash != MULTI_ANCHOR_GENERATOR_CONTRACT
            or manifest.metric_contract_hash
            != canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT)
            or manifest.training_schedule_hash != BALANCED_TRAINING_SCHEDULE_CONTRACT
            or manifest.initialization_seed != 1234
        ):
            raise ValueError(
                "multi-anchor checkpoint has an incomplete or mismatched contract"
            )
    if manifest.architecture_kind == SHORT_SPACING_ARCHITECTURE:
        from ..conditional_short_spacing import (
            SHORT_SPACING_GENERATOR_CONTRACT,
            SHORT_SPACING_METRIC_CONTRACT,
        )

        if (
            manifest.generator_contract_hash != SHORT_SPACING_GENERATOR_CONTRACT
            or manifest.metric_contract_hash != SHORT_SPACING_METRIC_CONTRACT
            or manifest.initialization_seed != 1234
        ):
            raise ValueError("v6.2 checkpoint has an incomplete or mismatched contract")
    if manifest.architecture_kind == FORMAL_SPACING_ARCHITECTURE:
        from ..conditional_formal_spacing import (
            FORMAL_SPACING_GENERATOR_CONTRACT,
            FORMAL_SPACING_METRIC_CONTRACT,
        )

        if (
            manifest.generator_contract_hash != FORMAL_SPACING_GENERATOR_CONTRACT
            or manifest.metric_contract_hash != FORMAL_SPACING_METRIC_CONTRACT
            or manifest.initialization_seed != 1234
        ):
            raise ValueError("v6.3 checkpoint has an incomplete or mismatched contract")
    if manifest.architecture_kind == NEAR_CONJUGATE_ARCHITECTURE:
        from ..conditional_near_conjugate import NEAR_CONJUGATE_METRIC_CONTRACT

        if (
            manifest.generator_contract_hash != NEAR_CONJUGATE_GENERATOR_CONTRACT
            or manifest.metric_contract_hash != NEAR_CONJUGATE_METRIC_CONTRACT
            or manifest.initialization_seed != 1234
        ):
            raise ValueError("v6.4 checkpoint has an incomplete or mismatched contract")
    load_conditional_checkpoint(checkpoint, bundle, manifest)
    return bundle, manifest


def _load_parity_source(
    config: V2Config, checkpoint: str, *, device: torch.device
) -> tuple[ConditionalBundle, dict]:
    """Explicitly migrate the v3 physical state into a zero-Q v4 bundle."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("manifest"), dict)
        or not isinstance(payload.get("state_dict"), dict)
    ):
        raise ValueError("invalid source checkpoint for parity migration")
    source = ConditionalCheckpointManifestV2(**payload["manifest"])
    if (
        source.architecture_kind != CLASS_MODE_ARCHITECTURE
        or source.system_signature != _system_signature(config)
    ):
        raise ValueError(
            "parity migration requires a class-mode v3 checkpoint with the same optical system"
        )
    state = payload["state_dict"]
    bundle = ConditionalBundle(config).to(device)
    target = bundle.state_dict()
    for key in (
        "system.encoder.phase",
        "system.decoder.phase",
        "generator.class_carrier_logits",
        "generator.class_modes",
    ):
        if (
            key not in state
            or key not in target
            or state[key].shape != target[key].shape
        ):
            raise ValueError(f"source checkpoint lacks exact migration tensor {key}")
        target[key].copy_(state[key])
    bundle.load_state_dict(target, strict=True)
    return bundle, payload
