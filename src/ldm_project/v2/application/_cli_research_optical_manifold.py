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
