"""Static handler registry retained behind the small v2 CLI dispatcher.

The schema is deliberately separate from execution.  A command target is only
imported after argparse and v2 configuration validation have completed.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CallKind = Literal["args", "args+config", "architecture-dispatch"]
IDENTITY_ARCHITECTURE = "conditional_identity_logit_amplitude_v2"
# Public monkeypatch seam retained for the diagnostic prototype oracle.  Values
# are forwarded only if that legacy command is selected.
_CONTROL_ORACLE_STEPS = 400
_CONTROL_ORACLE_LR = 1.0e-2
_CONTROL_ORACLE_PROGRESS_INTERVAL = 40


def _parser() -> argparse.ArgumentParser:
    """Construct the public schema without making it a dispatcher dependency."""
    from .cli_schema import _parser as build_parser

    return build_parser()


@dataclass(frozen=True)
class HandlerSpec:
    """Deferred command target; ``target`` is ``module:function``."""

    target: str
    call_kind: CallKind


# Formal commands resolve directly to the current runtime.  All remaining
# commands retain their historical parser and implementation behind the legacy
# boundary, so neither parser construction nor identity execution loads them.
_IDENTITY_HANDLERS = {
    ("student", "train"): HandlerSpec(
        ".current.training:conditional_train", "args+config"
    ),
    ("student", "sample"): HandlerSpec(
        ".current.evaluation:sample_conditional", "args+config"
    ),
    ("student", "evaluate"): HandlerSpec(
        ".current.evaluation:conditional_evaluate", "args+config"
    ),
    ("student", "train-evaluator"): HandlerSpec(
        ".current.admission:train_evaluator", "args+config"
    ),
    ("student", "qualify-evaluator"): HandlerSpec(
        ".current.admission:qualify_evaluator", "args+config"
    ),
    ("student", "train-acceptance-evaluator"): HandlerSpec(
        ".current.measurement:train_acceptance_evaluator", "args+config"
    ),
    ("student", "qualify-acceptance-measurement"): HandlerSpec(
        ".current.measurement:qualify_acceptance_measurement", "args+config"
    ),
    ("student", "controllability-smoke"): HandlerSpec(
        ".current.recovery:controllability_smoke", "args+config"
    ),
    ("student", "semantic-recovery-smoke"): HandlerSpec(
        ".current.recovery:semantic_recovery_smoke", "args+config"
    ),
    ("student", "export-dataset"): HandlerSpec(
        ".current.dataset:export_conditional_dataset", "args+config"
    ),
    ("student", "admit-conditional-evaluation"): HandlerSpec(
        ".current.evaluation:admit_conditional_evaluation", "args+config"
    ),
    ("student", "evaluate-utility"): HandlerSpec(
        ".current.dataset:evaluate_utility", "args+config"
    ),
    ("audit", "conditional-observability"): HandlerSpec(
        ".current.admission:conditional_observability", "args+config"
    ),
    ("audit", "conditional-injection-admission"): HandlerSpec(
        ".current.admission:conditional_injection_admission", "args+config"
    ),
    ("audit", "conditional-control-isolation"): HandlerSpec(
        ".current.admission:conditional_control_isolation", "args+config"
    ),
    ("audit", "conditional-semantic-localization"): HandlerSpec(
        ".current.admission:_identity_semantic_localization", "args+config"
    ),
    ("audit", "conditional-training-completion"): HandlerSpec(
        ".current.training:conditional_training_completion", "args+config"
    ),
    ("audit", "memorization"): HandlerSpec(
        ".current.dataset:audit_memorization", "args+config"
    ),
    ("student", "admit-dataset"): HandlerSpec(
        ".current.dataset:admit_dataset", "args+config"
    ),
    ("student", "audit-memorization"): HandlerSpec(
        ".current.dataset:audit_memorization", "args+config"
    ),
}

_NON_IDENTITY_HANDLERS = {
    ("evidence", "summarize"): HandlerSpec(".cli:_summarize", "args"),
    ("student", "train"): HandlerSpec(".legacy.templates:_dispatch_train", "args"),
    ("student", "sample"): HandlerSpec(".legacy.templates:_dispatch_sample", "args"),
    ("student", "evaluate"): HandlerSpec(
        ".legacy.templates:_dispatch_evaluate", "args"
    ),
    ("teacher", "train-first-stage"): HandlerSpec(
        ".legacy.teacher:train_first_stage", "args+config"
    ),
    ("teacher", "train-diffusion"): HandlerSpec(
        ".legacy.teacher:train_diffusion", "args+config"
    ),
    ("teacher", "export-targets"): HandlerSpec(
        ".legacy.teacher:export_targets", "args+config"
    ),
    ("teacher", "manifest"): HandlerSpec(".legacy.teacher:manifest", "args+config"),
    ("student", "calibrate-prototypes"): HandlerSpec(
        ".legacy.prototypes:_calibrate", "args+config"
    ),
    ("student", "export-clean-targets"): HandlerSpec(
        ".legacy.teacher:_export_clean_targets", "args+config"
    ),
    ("student", "admit-prototypes"): HandlerSpec(
        ".legacy.prototypes:_admit_prototypes", "args+config"
    ),
    ("student", "preflight"): HandlerSpec(
        ".legacy.prototypes:_preflight", "args+config"
    ),
    ("student", "admit-evaluation"): HandlerSpec(
        ".legacy.prototypes:_admit_evaluation", "args+config"
    ),
    ("audit", "capacity"): HandlerSpec(".legacy.prototypes:_capacity", "args+config"),
    ("audit", "prototype-reachability"): HandlerSpec(
        ".legacy.prototypes:_reachability", "args+config"
    ),
    ("audit", "prototype-topology"): HandlerSpec(
        ".legacy.geometry:_prototype_topology", "args+config"
    ),
    ("audit", "prototype-geometry-upper-bound"): HandlerSpec(
        ".legacy.geometry:_prototype_geometry_upper_bound", "args+config"
    ),
    ("audit", "prototype-control-oracle"): HandlerSpec(
        ".legacy.geometry:_prototype_control_oracle", "args+config"
    ),
    ("audit", "direct-template-replay"): HandlerSpec(
        ".legacy.templates:_direct_template_replay", "args+config"
    ),
    ("audit", "admit-direct-template-replay"): HandlerSpec(
        ".legacy.templates:_admit_direct_template_replay", "args+config"
    ),
    ("audit", "freeze-direct-template-prototype"): HandlerSpec(
        ".legacy.templates:_freeze_direct_template_prototype", "args+config"
    ),
    ("audit", "direct-template-local-capacity"): HandlerSpec(
        ".legacy.templates:_direct_template_local_capacity", "args+config"
    ),
    ("evidence", "rebind-manifest"): HandlerSpec(
        ".legacy.templates:_rebind_manifest", "args+config"
    ),
    ("evidence", "supersede-reachability"): HandlerSpec(
        ".legacy.templates:_supersede_reachability", "args+config"
    ),
    # v3-v5 diagnostic candidates.  These targets remain diagnostic-only; the
    # mapping merely avoids reparsing the already validated command line.
    ("student", "diversity-recovery-smoke"): HandlerSpec(
        ".diagnostic.legacy_training:diversity_recovery_smoke", "args+config"
    ),
    ("student", "train-diversity-recovery"): HandlerSpec(
        ".diagnostic.legacy_training:train_diversity_recovery", "args+config"
    ),
    ("student", "diversity-aligned-smoke"): HandlerSpec(
        ".diagnostic.legacy_training:diversity_aligned_smoke", "args+config"
    ),
    ("student", "train-diversity-aligned"): HandlerSpec(
        ".diagnostic.legacy_training:train_diversity_aligned", "args+config"
    ),
    ("student", "identity-reanchor-smoke"): HandlerSpec(
        ".diagnostic.diversity:identity_reanchor_smoke", "args+config"
    ),
    ("student", "train-identity-reanchor"): HandlerSpec(
        ".diagnostic.diversity:train_identity_reanchor", "args+config"
    ),
    ("student", "class-mode-recovery-smoke"): HandlerSpec(
        ".diagnostic.parity:class_mode_recovery_smoke", "args+config"
    ),
    ("student", "train-class-modes"): HandlerSpec(
        ".diagnostic.parity:train_class_modes", "args+config"
    ),
    ("student", "parity-stabilizer-smoke"): HandlerSpec(
        ".diagnostic.parity:parity_stabilizer_smoke", "args+config"
    ),
    ("student", "train-parity-stabilizer"): HandlerSpec(
        ".diagnostic.parity:train_parity_stabilizer", "args+config"
    ),
    ("student", "multi-anchor-smoke"): HandlerSpec(
        ".diagnostic.multi_anchor:multi_anchor_smoke", "args+config"
    ),
    ("student", "train-multi-anchor"): HandlerSpec(
        ".diagnostic.multi_anchor:train_multi_anchor", "args+config"
    ),
    ("audit", "conditional-diversity-observability"): HandlerSpec(
        ".diagnostic.diversity:conditional_diversity_observability", "args+config"
    ),
    ("audit", "conditional-diversity-alignment-admission"): HandlerSpec(
        ".diagnostic.diversity:conditional_diversity_alignment_admission", "args+config"
    ),
    ("audit", "conditional-identity-reanchor"): HandlerSpec(
        ".diagnostic.diversity:conditional_identity_reanchor", "args+config"
    ),
    ("audit", "conditional-reanchor-observability"): HandlerSpec(
        ".diagnostic.diversity:conditional_reanchor_observability", "args+config"
    ),
    ("audit", "conditional-shared-mode-conflict"): HandlerSpec(
        ".diagnostic.class_modes:conditional_shared_mode_conflict", "args+config"
    ),
    ("audit", "conditional-class-mode-admission"): HandlerSpec(
        ".diagnostic.class_modes:conditional_class_mode_admission", "args+config"
    ),
    ("audit", "conditional-class-mode-observability"): HandlerSpec(
        ".diagnostic.class_modes:conditional_class_mode_observability", "args+config"
    ),
    ("audit", "conditional-semantic-tangent"): HandlerSpec(
        ".diagnostic.class_modes:conditional_semantic_tangent", "args+config"
    ),
    ("audit", "conditional-parity-conflict"): HandlerSpec(
        ".diagnostic.parity:conditional_parity_conflict", "args+config"
    ),
    ("audit", "conditional-parity-mode-admission"): HandlerSpec(
        ".diagnostic.parity:conditional_parity_mode_admission", "args+config"
    ),
    ("audit", "conditional-parity-observability"): HandlerSpec(
        ".diagnostic.parity:conditional_parity_observability", "args+config"
    ),
    ("audit", "conditional-multi-anchor-capacity"): HandlerSpec(
        ".diagnostic.multi_anchor:conditional_multi_anchor_capacity", "args+config"
    ),
    ("audit", "conditional-multi-anchor-admission"): HandlerSpec(
        ".diagnostic.multi_anchor:conditional_multi_anchor_admission", "args+config"
    ),
    ("audit", "conditional-multi-anchor-observability"): HandlerSpec(
        ".diagnostic.multi_anchor:conditional_multi_anchor_observability", "args+config"
    ),
    # v6-v6.4 diagnostic candidates.
    ("student", "pca-transport-smoke"): HandlerSpec(
        ".diagnostic.pca_transport:pca_transport_smoke", "args+config"
    ),
    ("student", "train-pca-transport"): HandlerSpec(
        ".diagnostic.pca_transport:train_pca_transport", "args+config"
    ),
    ("audit", "conditional-pca-transport-admission"): HandlerSpec(
        ".diagnostic.pca_transport:conditional_pca_transport_admission", "args+config"
    ),
    ("audit", "conditional-pca-transport-observability"): HandlerSpec(
        ".diagnostic.pca_transport:conditional_pca_transport_observability",
        "args+config",
    ),
    ("audit", "conditional-pca-transport-smoke-revalidation"): HandlerSpec(
        ".diagnostic.pca_transport:conditional_pca_transport_smoke_revalidation",
        "args+config",
    ),
    ("student", "short-spacing-self-imaging-smoke"): HandlerSpec(
        ".diagnostic.short_spacing:short_spacing_self_imaging_smoke", "args+config"
    ),
    ("audit", "conditional-short-spacing-target-admission"): HandlerSpec(
        ".diagnostic.short_spacing:conditional_short_spacing_target_admission",
        "args+config",
    ),
    ("audit", "conditional-short-spacing-optical-admission"): HandlerSpec(
        ".diagnostic.short_spacing:conditional_short_spacing_optical_admission",
        "args+config",
    ),
    ("audit", "conditional-short-spacing-optical-supersession"): HandlerSpec(
        ".diagnostic.short_spacing:conditional_short_spacing_evidence_supersession",
        "args+config",
    ),
    ("audit", "conditional-short-spacing-observability"): HandlerSpec(
        ".diagnostic.short_spacing:conditional_short_spacing_observability",
        "args+config",
    ),
    ("student", "formal-spacing-self-imaging-smoke"): HandlerSpec(
        ".diagnostic.formal_spacing:formal_spacing_self_imaging_smoke", "args+config"
    ),
    ("student", "train-formal-spacing-self-imaging"): HandlerSpec(
        ".diagnostic.formal_spacing:train_formal_spacing_self_imaging", "args+config"
    ),
    ("audit", "conditional-formal-spacing-admission"): HandlerSpec(
        ".diagnostic.formal_spacing:conditional_formal_spacing_admission", "args+config"
    ),
    ("audit", "conditional-formal-spacing-semantic-tangent"): HandlerSpec(
        ".diagnostic.formal_spacing:conditional_formal_spacing_semantic_tangent",
        "args+config",
    ),
    ("audit", "conditional-formal-spacing-observability"): HandlerSpec(
        ".diagnostic.formal_spacing:conditional_formal_spacing_observability",
        "args+config",
    ),
    ("student", "near-conjugate-relay-smoke"): HandlerSpec(
        ".diagnostic.near_conjugate:near_conjugate_relay_smoke", "args+config"
    ),
    ("student", "train-near-conjugate-relay"): HandlerSpec(
        ".diagnostic.near_conjugate:train_near_conjugate_relay", "args+config"
    ),
    ("audit", "conditional-near-conjugate-relay-equivalence"): HandlerSpec(
        ".diagnostic.near_conjugate:conditional_near_conjugate_relay_equivalence",
        "args+config",
    ),
    ("audit", "conditional-near-conjugate-admission"): HandlerSpec(
        ".diagnostic.near_conjugate:conditional_near_conjugate_admission", "args+config"
    ),
    ("audit", "conditional-near-conjugate-semantic-tangent"): HandlerSpec(
        ".diagnostic.near_conjugate:conditional_near_conjugate_semantic_tangent",
        "args+config",
    ),
    ("audit", "conditional-near-conjugate-observability"): HandlerSpec(
        ".diagnostic.near_conjugate:conditional_near_conjugate_observability",
        "args+config",
    ),
    ("audit", "conditional-near-conjugate-smoke-replay"): HandlerSpec(
        ".diagnostic.near_conjugate:conditional_near_conjugate_smoke_replay",
        "args+config",
    ),
    ("audit", "conditional-near-conjugate-phase-distribution-tangent"): HandlerSpec(
        ".diagnostic.near_conjugate:conditional_near_conjugate_phase_distribution_tangent",
        "args+config",
    ),
    # v7-v7.2 and LightGen diagnostics.
    ("audit", "conditional-nonlinear-expanded-manifold-capacity"): HandlerSpec(
        ".diagnostic.expanded_manifold:conditional_nonlinear_expanded_manifold_capacity",
        "args+config",
    ),
    ("audit", "conditional-nonlinear-expanded-manifold-admission"): HandlerSpec(
        ".diagnostic.expanded_manifold:conditional_nonlinear_expanded_manifold_admission",
        "args+config",
    ),
    ("audit", "conditional-transport-aligned-manifold-design"): HandlerSpec(
        ".diagnostic.transport_aligned_manifold:conditional_transport_aligned_manifold_design",
        "args+config",
    ),
    ("audit", "conditional-transport-aligned-manifold-admission"): HandlerSpec(
        ".diagnostic.transport_aligned_manifold:conditional_transport_aligned_manifold_admission",
        "args+config",
    ),
    ("audit", "conditional-transport-aligned-manifold-tangent"): HandlerSpec(
        ".diagnostic.transport_aligned_manifold:conditional_transport_aligned_manifold_tangent",
        "args+config",
    ),
    ("student", "transport-aligned-manifold-smoke"): HandlerSpec(
        ".diagnostic.transport_aligned_manifold:transport_aligned_manifold_smoke",
        "args+config",
    ),
    ("student", "train-transport-aligned-manifold"): HandlerSpec(
        ".diagnostic.transport_aligned_manifold:train_transport_aligned_manifold",
        "args+config",
    ),
    ("audit", "conditional-transport-aligned-phase-capacity-design"): HandlerSpec(
        ".diagnostic.transport_aligned_phase_capacity:conditional_transport_aligned_phase_capacity_design",
        "args+config",
    ),
    ("audit", "conditional-transport-curvature-attribution"): HandlerSpec(
        ".diagnostic.transport_curvature:conditional_transport_curvature_attribution",
        "args+config",
    ),
    ("audit", "conditional-evaluator-qualification-v2"): HandlerSpec(
        ".diagnostic.evaluator_qualification_v2:conditional_evaluator_qualification_v2",
        "args+config",
    ),
    ("audit", "conditional-evaluator-calibration-v21"): HandlerSpec(
        ".diagnostic.evaluator_qualification_v2:conditional_evaluator_calibration_v21",
        "args+config",
    ),
    ("audit", "conditional-evaluator-qualification-v21"): HandlerSpec(
        ".diagnostic.evaluator_qualification_v2:conditional_evaluator_qualification_v21",
        "args+config",
    ),
    ("audit", "conditional-evaluator-curvature-baseline"): HandlerSpec(
        ".diagnostic.evaluation_qualified_transport:conditional_evaluator_curvature_baseline",
        "args+config",
    ),
    ("audit", "conditional-evaluation-qualified-transport-design"): HandlerSpec(
        ".diagnostic.evaluation_qualified_transport:conditional_evaluation_qualified_transport_design",
        "args+config",
    ),
    ("audit", "conditional-evaluation-qualified-transport-admission"): HandlerSpec(
        ".diagnostic.evaluation_qualified_transport:conditional_evaluation_qualified_transport_admission",
        "args+config",
    ),
    ("audit", "conditional-evaluation-qualified-transport-tangent"): HandlerSpec(
        ".diagnostic.evaluation_qualified_transport:conditional_evaluation_qualified_transport_tangent",
        "args+config",
    ),
    ("student", "evaluation-qualified-transport-smoke"): HandlerSpec(
        ".diagnostic.evaluation_qualified_transport:evaluation_qualified_transport_smoke",
        "args+config",
    ),
    ("audit", "lightgen-contract-gap"): HandlerSpec(
        ".legacy.teacher:_lightgen_contract_gap", "args+config"
    ),
    ("audit", "lightgen-ols-observability"): HandlerSpec(
        ".legacy.teacher:_lightgen_ols_observability", "args+config"
    ),
    ("audit", "conditional-transport-aligned-phase-capacity-admission"): HandlerSpec(
        ".diagnostic.transport_aligned_phase_capacity:conditional_transport_aligned_phase_capacity_admission",
        "args+config",
    ),
    ("audit", "conditional-transport-aligned-phase-capacity-tangent"): HandlerSpec(
        ".diagnostic.transport_aligned_phase_capacity:conditional_transport_aligned_phase_capacity_tangent",
        "args+config",
    ),
    ("student", "transport-aligned-phase-capacity-smoke"): HandlerSpec(
        ".diagnostic.transport_aligned_phase_capacity:transport_aligned_phase_capacity_smoke",
        "args+config",
    ),
    ("audit", "conditional-continuous-mode-capacity"): HandlerSpec(
        ".diagnostic.continuous_audit:conditional_continuous_mode_capacity",
        "args+config",
    ),
    ("audit", "conditional-continuous-mode-reachability"): HandlerSpec(
        ".diagnostic.continuous_reachability:conditional_continuous_mode_reachability",
        "args+config",
    ),
    ("audit", "conditional-optical-generation-feasibility"): HandlerSpec(
        ".diagnostic.optical_feasibility:conditional_optical_generation_feasibility",
        "args+config",
    ),
}

# Diagnostic commands remain available through the public parser, but they are
# deliberately a separate registry from legacy teacher/prototype commands.
_DIAGNOSTIC_HANDLERS = {
    key: spec
    for key, spec in _NON_IDENTITY_HANDLERS.items()
    if ".diagnostic." in spec.target
}
_LEGACY_HANDLERS = {
    key: spec
    for key, spec in _NON_IDENTITY_HANDLERS.items()
    if key not in _DIAGNOSTIC_HANDLERS
}


def _load_config(args: argparse.Namespace):
    path = getattr(args, "config", None)
    if path is None:
        return None
    from ..config import load_config

    return load_config(path)


def _resolve(spec: HandlerSpec):
    module_name, separator, name = spec.target.partition(":")
    if not separator:
        raise ValueError(f"invalid CLI target: {spec.target}")
    module = importlib.import_module(module_name, __package__)
    return getattr(module, name)


def _summarize(args: argparse.Namespace) -> int:
    print(Path(args.run).read_text(encoding="utf-8"))
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    config = _load_config(args)
    spec = _IDENTITY_HANDLERS.get((args.domain, args.action))
    if config is not None and config.architecture == IDENTITY_ARCHITECTURE:
        if spec is None:
            raise RuntimeError(
                f"identity architecture has no current handler for "
                f"{args.domain}/{args.action}"
            )
        target = _resolve(spec)
        return int(target(args, config) or 0)
    spec = _DIAGNOSTIC_HANDLERS.get((args.domain, args.action))
    if spec is None:
        spec = _LEGACY_HANDLERS.get((args.domain, args.action))
    if spec is not None and spec.call_kind == "args":
        return int(_resolve(spec)(args) or 0)
    if spec is not None and config is not None:
        target = _resolve(spec)
        module = importlib.import_module(spec.target.partition(":")[0], __package__)
        for name in (
            "_CONTROL_ORACLE_STEPS",
            "_CONTROL_ORACLE_PROGRESS_INTERVAL",
            "_CONTROL_ORACLE_LR",
        ):
            if hasattr(module, name):
                setattr(module, name, globals()[name])
        return int(target(args, config) or 0)
    raise RuntimeError(f"no static v2 command handler for {args.domain}/{args.action}")


def __getattr__(name: str):
    """Compatibility access for diagnostic helpers without eager imports."""
    if name == "_lightgen_seeds":
        module = importlib.import_module(".legacy.teacher", __package__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else None
    args = _parser().parse_args(arguments)
    return _dispatch(args)
