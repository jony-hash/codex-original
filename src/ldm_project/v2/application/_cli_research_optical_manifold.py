"""Lazy router for optical-manifold research actions."""

from __future__ import annotations

import argparse


def dispatch(args: argparse.Namespace) -> int:
    action = args.research_action
    if action in {
        "adaptive-projection-audit-validate",
        "adaptive-projection-audit-run",
    }:
        from .research.optical_manifold_adaptive_projection_audit import dispatch
    elif action in {
        "smf-integration-canary-validate",
        "smf-integration-canary-preflight",
        "smf-integration-canary-train",
    }:
        from .research.optical_manifold_smf_integration_canary import dispatch
    elif action in {
        "smf-reconstruction-validate",
        "smf-reconstruction-preflight",
        "smf-reconstruction-train",
    }:
        from .research.optical_manifold_smf_reconstruction_pilot import dispatch
    elif action in {
        "fmf-m2-integration-canary-validate",
        "fmf-m2-integration-canary-preflight",
        "fmf-m2-integration-canary-train",
    }:
        from .research.optical_manifold_fmf_m2_integration_canary import dispatch
    elif action in {
        "smf-vae-pilot-validate",
        "smf-vae-pilot-preflight",
        "smf-vae-pilot-train",
    }:
        from .research.optical_manifold_smf_vae_pilot import dispatch
    elif action in {
        "smf-kl-chain-audit-validate",
        "smf-kl-chain-audit-run",
    }:
        from .research.optical_manifold_smf_vae_kl_attribution_audit import dispatch
    elif action in {
        "smf-kl-dose-reaudit-validate",
        "smf-kl-dose-reaudit-run",
    }:
        from .research.optical_manifold_smf_vae_kl_dose_reaudit import dispatch
    elif action in {
        "smf-kl-effect-pilot-validate",
        "smf-kl-effect-pilot-preflight",
        "smf-kl-effect-pilot-train",
    }:
        from .research.optical_manifold_smf_vae_kl_effect import dispatch
    elif action in {
        "smf-ols-loss-recovery-validate",
        "smf-ols-loss-recovery-preflight",
        "smf-ols-loss-recovery-train",
    }:
        from .research.optical_manifold_smf_ols_loss_recovery import dispatch
    elif action in {
        "smf-bottleneck-attribution-validate",
        "smf-bottleneck-attribution-run",
    }:
        from .research.optical_manifold_smf_bottleneck_attribution import dispatch
    elif action in {
        "scaled-chen-geometry-attribution-validate",
        "scaled-chen-geometry-attribution-run",
    }:
        from .research.optical_manifold_smf_bogt_geometry_attribution import dispatch
    elif action in {
        "scaled-chen-manifold-readiness-validate",
        "scaled-chen-manifold-readiness-run",
    }:
        from .research.optical_manifold_smf_bogt_manifold_readiness import dispatch
    elif action in {
        "scaled-chen-decoder-cmos-attribution-validate",
        "scaled-chen-decoder-cmos-attribution-run",
    }:
        from .research.optical_manifold_smf_bogt_decoder_cmos_attribution import dispatch
    elif action in {
        "scaled-chen-backend-observable-subspace-validate",
        "scaled-chen-backend-observable-subspace-run",
    }:
        from .research.optical_manifold_smf_bogt_backend_observable_subspace import dispatch
    elif action in {
        "scaled-chen-high-fidelity-relay-reference-validate",
        "scaled-chen-high-fidelity-relay-reference-run",
    }:
        from .research.optical_manifold_smf_bogt_high_fidelity_relay_reference import dispatch
    elif action in {
        "scaled-chen-relay-metric-reinterpretation-validate",
        "scaled-chen-relay-metric-reinterpretation-run",
    }:
        from .research.optical_manifold_smf_bogt_relay_metric_reinterpretation import dispatch
    elif action in {
        "scaled-chen-task-relevant-bottleneck-validate",
        "scaled-chen-task-relevant-bottleneck-run",
    }:
        from .research.optical_manifold_smf_bogt_task_relevant_bottleneck import dispatch
    elif action in {
        "scaled-chen-reconstruction-manifold-failure-validate",
        "scaled-chen-reconstruction-manifold-failure-run",
    }:
        from .research.optical_manifold_smf_bogt_reconstruction_manifold_failure import dispatch
    elif action in {
        "scaled-chen-reconstruction-mapping-alignment-localization-validate",
        "scaled-chen-reconstruction-mapping-alignment-localization-run",
    }:
        from .research.optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization import dispatch
    elif action in {
        "scaled-chen-detector-readout-structure-attribution-validate",
        "scaled-chen-detector-readout-structure-attribution-run",
    }:
        from .research.optical_manifold_smf_bogt_detector_readout_structure_attribution import dispatch
    elif action in {
        "scaled-chen-detector-readout-zero-column-semantics-audit-validate",
        "scaled-chen-detector-readout-zero-column-semantics-audit-run",
    }:
        from .research.optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit import dispatch
    elif action in {
        "scaled-chen-detector-readout-structure-active-domain-validate",
        "scaled-chen-detector-readout-structure-active-domain-run",
    }:
        from .research.optical_manifold_smf_bogt_detector_readout_structure_attribution_active_domain import dispatch
    elif action in {
        "scaled-chen-linear-readout-structure-decomposition-validate",
        "scaled-chen-linear-readout-structure-decomposition-run",
    }:
        from .research.optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit import dispatch
    elif action in {
        "scaled-chen-low-rank-readout-recovery-confirmation-validate",
        "scaled-chen-low-rank-readout-recovery-confirmation-run",
    }:
        from .research.optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation import dispatch
    elif action in {
        "scaled-chen-data-weighted-spectral-functional-attribution-validate",
        "scaled-chen-data-weighted-spectral-functional-attribution-run",
    }:
        from .research.optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution import dispatch
    elif action in {
        "scaled-chen-back-relay-axial-conjugate-validate",
        "scaled-chen-back-relay-axial-conjugate-preflight",
        "scaled-chen-back-relay-axial-conjugate-run",
        "scaled-chen-back-relay-axial-conjugate-report",
        "scaled-chen-back-relay-axial-conjugate-closeout",
    }:
        from .research.optical_manifold_smf_bogt_back_relay_axial_conjugate import dispatch
    elif action in {
        "smf-bogt-validate",
        "smf-bogt-preflight",
        "smf-bogt-pilot-train",
    }:
        from .research.optical_manifold_smf_bogt import dispatch
    elif action in {
        "smf-bogt-failure-attribution-validate",
        "smf-bogt-failure-attribution-run",
    }:
        from .research.optical_manifold_smf_bogt_failure_attribution import dispatch
    elif action in {
        "smf-bogt-horizon-validate",
        "smf-bogt-horizon-preflight",
        "smf-bogt-horizon-train",
    }:
        from .research.optical_manifold_smf_bogt_horizon import dispatch
    else:
        from .research.optical_manifold import dispatch
    return int(dispatch(args) or 0)
