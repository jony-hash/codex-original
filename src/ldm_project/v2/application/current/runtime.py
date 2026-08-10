# ruff: noqa: F401
"""Formal identity runtime primitives.

This module contains only the current identity bundle, its checkpoint contract,
and neutral runtime helpers. Candidate architectures live in
:mod:`application.diagnostic`.
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
    safe_torch_load,
    tensor_sha256,
)
from ...optics import FormalAmplitudeFiniteCropIdentitySystem
from ...student.checkpoints import (
    ConditionalCheckpointManifestV2,
    load_conditional_checkpoint_payload,
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
from ...evidence.records import require_evidence_action, require_input_hashes
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
IDENTITY_GENERATOR_CONTRACT = canonical_sha256(
    {
        "architecture": IDENTITY_ARCHITECTURE,
        "formula": "h=SiLU(Wz*z);r=Wy*h;c=tanh(z+r);A=sigmoid(By+0.5*sum(c*P_hat))",
        "gain": 0.5,
        "bias_free": True,
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
    """Exact formal identity checkpoint topology.

    Attribute names intentionally match the historical conditional bundle so
    state-dict keys and checkpoint signatures remain unchanged.
    """

    def __init__(
        self, config: V2Config, *, carrier_logits: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        if config.architecture != IDENTITY_ARCHITECTURE or config.conditional_generation is None:
            raise ValueError("formal current bundle requires conditional_identity_logit_amplitude_v2")
        settings = config.conditional_generation
        self.system = FormalAmplitudeFiniteCropIdentitySystem(config.system)
        self.generator = IdentityAnchoredConditionalAmplitudeGenerator(
            shape=config.system.slm_shape,
            code_dim=settings.code_dim,
            hidden_dim=settings.hidden_dim,
            command_gain=settings.command_gain,
            class_carrier_logits=carrier_logits,
        )


def _seeded_conditional_bundle(
    config: V2Config,
    *,
    carrier_logits: torch.Tensor | None = None,
) -> ConditionalBundle:
    """Construct one identity bundle from the scoped configuration seed."""
    if config.conditional_generation is None:
        raise ValueError("conditional initialization requires conditional_generation")
    seed = config.conditional_generation.initialization_seed
    if seed < 0:
        raise ValueError("conditional initialization seed must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ConditionalBundle(config, carrier_logits=carrier_logits)

def _require_identity(config: V2Config) -> None:
    if config.architecture != IDENTITY_ARCHITECTURE or config.conditional_generation is None:
        raise ValueError("this command requires architecture: conditional_identity_logit_amplitude_v2")

_require_conditional = _require_identity

def _require_current_acceptance(config: V2Config) -> None:
    """Current evidence can only be issued by the formal identity runtime."""
    _require_identity(config)


def _validate_initialization_payload(
    config: V2Config,
    manifest: ConditionalCheckpointManifestV2,
    extra: object,
) -> None:
    settings = config.conditional_generation
    if settings is None:
        raise ValueError("identity checkpoint requires conditional_generation")
    if manifest.initialization_seed != settings.initialization_seed:
        raise ValueError("identity checkpoint initialization seed is inconsistent")
    if manifest.initialization_state_hash is None:
        raise ValueError("identity checkpoint lacks initialization state hash")
    if not isinstance(extra, dict) or not extra.get("initialization"):
        return
    if manifest.state_content_hash != manifest.initialization_state_hash:
        raise ValueError("initialization checkpoint state hash is inconsistent")
    if extra.get("seed") != manifest.initialization_seed:
        raise ValueError("initialization checkpoint seed is inconsistent")


def _load_bundle(
    config: V2Config,
    checkpoint: str,
    *,
    device: torch.device,
    evaluator_hash: str | None = None,
    split_hash: str | None = None,
    allow_balanced_schedule: bool = False,
    checkpoint_sha256: str | None = None,
) -> tuple[ConditionalBundle, ConditionalCheckpointManifestV2]:
    bundle, manifest, _ = _load_bundle_with_payload(
        config,
        checkpoint,
        device=device,
        evaluator_hash=evaluator_hash,
        split_hash=split_hash,
        allow_balanced_schedule=allow_balanced_schedule,
        checkpoint_sha256=checkpoint_sha256,
    )
    return bundle, manifest


def _load_bundle_with_payload(
    config: V2Config,
    checkpoint: str,
    *,
    device: torch.device,
    evaluator_hash: str | None = None,
    split_hash: str | None = None,
    allow_balanced_schedule: bool = False,
    checkpoint_sha256: str | None = None,
) -> tuple[
    ConditionalBundle,
    ConditionalCheckpointManifestV2,
    dict[str, object],
]:
    _require_identity(config)
    payload = safe_torch_load(checkpoint, expected_sha256=checkpoint_sha256)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
        raise ValueError("invalid conditional checkpoint")
    manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    if (
        manifest.architecture_kind != IDENTITY_ARCHITECTURE
        or manifest.system_signature != _system_signature(config)
        or manifest.config_hash != canonical_sha256(config.model_dump(mode="json"))
    ):
        raise ValueError("conditional checkpoint belongs to a different architecture/system/config")
    if evaluator_hash is not None and manifest.evaluator_checkpoint_hash != evaluator_hash:
        raise ValueError("conditional checkpoint uses another evaluator")
    if split_hash is not None and manifest.dataset_split_manifest_hash != split_hash:
        raise ValueError("conditional checkpoint uses another MNIST split")
    if (
        config.conditional_generation is None
        or manifest.initialization_seed
        != config.conditional_generation.initialization_seed
    ):
        raise ValueError("identity checkpoint initialization seed mismatch")
    valid_schedules = {TRAINING_SCHEDULE_CONTRACT}
    if allow_balanced_schedule:
        valid_schedules.add(BALANCED_TRAINING_SCHEDULE_CONTRACT)
    if (
        manifest.generator_contract_hash != IDENTITY_GENERATOR_CONTRACT
        or manifest.metric_contract_hash != canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT)
        or manifest.training_schedule_hash not in valid_schedules
        or config.conditional_generation is None
    ):
        raise ValueError("identity checkpoint has an incomplete or mismatched contract")
    _validate_initialization_payload(config, manifest, payload.get("extra"))
    bundle = _seeded_conditional_bundle(config).to(device)
    load_conditional_checkpoint_payload(payload, bundle, manifest)
    return bundle, manifest, payload
