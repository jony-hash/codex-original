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
    else:
        from .research.optical_manifold import dispatch
    return int(dispatch(args) or 0)
