from __future__ import annotations

import argparse


def register_research(root: argparse._SubParsersAction) -> None:
    research = root.add_parser(
        "research", help="diagnostic-only pre-registered research programs"
    )
    research_programs = research.add_subparsers(dest="research_program", required=True)
    optical_manifold = research_programs.add_parser(
        "optical-manifold", help="LightGen-like Identity optical manifold preflight"
    )
    optical_actions = optical_manifold.add_subparsers(
        dest="research_action", required=True
    )
    optical_validate = optical_actions.add_parser("validate")
    optical_validate.add_argument("--config", required=True)
    optical_validate.add_argument("--dataset-root", required=True)
    optical_validate.add_argument("--output", required=True)
    optical_preflight = optical_actions.add_parser("preflight")
    optical_preflight.add_argument("--validate-evidence", required=True)
    optical_preflight.add_argument("--config", required=True)
    optical_preflight.add_argument("--dataset-root", required=True)
    optical_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    optical_preflight.add_argument("--output", required=True)
    bogt_validate = optical_actions.add_parser("smf-bogt-validate", help="validate the independent scaled Chen-BOGT contract")
    bogt_validate.add_argument("--config", required=True)
    bogt_validate.add_argument("--review-contract", required=True)
    bogt_validate.add_argument("--attribution-evidence")
    bogt_validate.add_argument("--dataset-root", required=True)
    bogt_validate.add_argument("--output", required=True)
    bogt_preflight = optical_actions.add_parser("smf-bogt-preflight", help="run the gated CUDA preflight for scaled Chen-BOGT")
    bogt_preflight.add_argument("--validate-evidence", required=True)
    bogt_preflight.add_argument("--review-contract", required=True)
    bogt_preflight.add_argument("--attribution-evidence")
    bogt_preflight.add_argument("--config", required=True)
    bogt_preflight.add_argument("--dataset-root", required=True)
    bogt_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    bogt_preflight.add_argument("--output", required=True)
    bogt_train = optical_actions.add_parser("smf-bogt-pilot-train", help="run the fail-closed four-arm scaled Chen-BOGT pilot")
    bogt_train.add_argument("--config", required=True)
    bogt_train.add_argument("--review-contract", required=True)
    bogt_train.add_argument("--pilot-review-contract", required=True)
    bogt_train.add_argument("--attribution-evidence")
    bogt_train.add_argument("--validation-evidence", required=True)
    bogt_train.add_argument("--preflight-evidence", required=True)
    bogt_train.add_argument("--dataset-root", required=True)
    bogt_train.add_argument("--device", choices=("cuda",), default="cuda")
    bogt_train.add_argument("--output", required=True)
    bogt_attr_validate = optical_actions.add_parser("smf-bogt-failure-attribution-validate", help="validate the no-update scaled Chen-BOGT failure-attribution contract")
    bogt_attr_validate.add_argument("--config", required=True)
    bogt_attr_validate.add_argument("--review-contract", required=True)
    bogt_attr_validate.add_argument("--dataset-root", required=True)
    bogt_attr_validate.add_argument("--output", required=True)
    bogt_attr_run = optical_actions.add_parser("smf-bogt-failure-attribution-run", help="run the no-update scaled Chen-BOGT dose attribution")
    bogt_attr_run.add_argument("--config", required=True)
    bogt_attr_run.add_argument("--review-contract", required=True)
    bogt_attr_run.add_argument("--validation-evidence", required=True)
    bogt_attr_run.add_argument("--dataset-root", required=True)
    bogt_attr_run.add_argument("--device", choices=("cuda",), default="cuda")
    bogt_attr_run.add_argument("--output", required=True)
    bogt_horizon_validate = optical_actions.add_parser("smf-bogt-horizon-validate", help="validate the MSE-only scaled Chen reconstruction-horizon contract")
    bogt_horizon_validate.add_argument("--config", required=True)
    bogt_horizon_validate.add_argument("--review-contract", required=True)
    bogt_horizon_validate.add_argument("--dataset-root", required=True)
    bogt_horizon_validate.add_argument("--output", required=True)
    bogt_horizon_preflight = optical_actions.add_parser("smf-bogt-horizon-preflight", help="run the CUDA preflight for the MSE-only scaled Chen reconstruction horizon")
    bogt_horizon_preflight.add_argument("--config", required=True)
    bogt_horizon_preflight.add_argument("--review-contract", required=True)
    bogt_horizon_preflight.add_argument("--validate-evidence", required=True)
    bogt_horizon_preflight.add_argument("--dataset-root", required=True)
    bogt_horizon_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    bogt_horizon_preflight.add_argument("--output", required=True)
    bogt_horizon_train = optical_actions.add_parser("smf-bogt-horizon-train", help="run the MSE-only scaled Chen reconstruction horizon")
    bogt_horizon_train.add_argument("--config", required=True)
    bogt_horizon_train.add_argument("--review-contract", required=True)
    bogt_horizon_train.add_argument("--validation-evidence", required=True)
    bogt_horizon_train.add_argument("--preflight-evidence", required=True)
    bogt_horizon_train.add_argument("--dataset-root", required=True)
    bogt_horizon_train.add_argument("--device", choices=("cuda",), default="cuda")
    bogt_horizon_train.add_argument("--output", required=True)
    pilot_validate = optical_actions.add_parser("train-validate")
    pilot_validate.add_argument("--config", required=True)
    pilot_validate.add_argument("--review-contract", required=True)
    pilot_validate.add_argument("--source-validation-evidence", required=True)
    pilot_validate.add_argument("--source-preflight-evidence", required=True)
    pilot_validate.add_argument("--dataset-root", required=True)
    pilot_validate.add_argument("--output", required=True)
    pilot_preflight = optical_actions.add_parser("train-preflight")
    pilot_preflight.add_argument("--config", required=True)
    pilot_preflight.add_argument("--review-contract", required=True)
    pilot_preflight.add_argument("--training-validation-evidence", required=True)
    pilot_preflight.add_argument("--dataset-root", required=True)
    pilot_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    pilot_preflight.add_argument("--output", required=True)
    pilot_train = optical_actions.add_parser("train")
    pilot_train.add_argument("--config", required=True)
    pilot_train.add_argument("--review-contract", required=True)
    pilot_train.add_argument("--training-validation-evidence", required=True)
    pilot_train.add_argument("--training-preflight-evidence", required=True)
    pilot_train.add_argument("--dataset-root", required=True)
    pilot_train.add_argument("--device", choices=("cuda",), default="cuda")
    pilot_train.add_argument("--output", required=True)
    pilot_train.add_argument("--resume-checkpoint")
    objective_validate = optical_actions.add_parser(
        "manifold-regularization-objective-audit-validate"
    )
    objective_validate.add_argument("--config", required=True)
    objective_validate.add_argument("--review-contract", required=True)
    objective_validate.add_argument("--dataset-root", required=True)
    objective_validate.add_argument("--output", required=True)
    objective_run = optical_actions.add_parser(
        "manifold-regularization-objective-audit-run"
    )
    objective_run.add_argument("--config", required=True)
    objective_run.add_argument("--review-contract", required=True)
    objective_run.add_argument("--validation-evidence", required=True)
    objective_run.add_argument("--dataset-root", required=True)
    objective_run.add_argument("--device", choices=("cuda",), default="cuda")
    objective_run.add_argument("--output", required=True)
    adaptive_validate = optical_actions.add_parser("adaptive-projection-audit-validate")
    adaptive_validate.add_argument("--config", required=True)
    adaptive_validate.add_argument("--review-contract", required=True)
    adaptive_validate.add_argument("--dataset-root", required=True)
    adaptive_validate.add_argument("--output", required=True)
    adaptive_run = optical_actions.add_parser("adaptive-projection-audit-run")
    adaptive_run.add_argument("--config", required=True)
    adaptive_run.add_argument("--review-contract", required=True)
    adaptive_run.add_argument("--validation-evidence", required=True)
    adaptive_run.add_argument("--dataset-root", required=True)
    adaptive_run.add_argument("--device", choices=("cuda",), default="cuda")
    adaptive_run.add_argument("--output", required=True)
    smf_validate = optical_actions.add_parser("smf-integration-canary-validate")
    smf_validate.add_argument("--config", required=True)
    smf_validate.add_argument("--review-contract", required=True)
    smf_validate.add_argument("--dataset-root", required=True)
    smf_validate.add_argument("--output", required=True)
    smf_preflight = optical_actions.add_parser("smf-integration-canary-preflight")
    smf_preflight.add_argument("--config", required=True)
    smf_preflight.add_argument("--review-contract", required=True)
    smf_preflight.add_argument("--validation-evidence", required=True)
    smf_preflight.add_argument("--dataset-root", required=True)
    smf_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    smf_preflight.add_argument("--output", required=True)
    smf_train = optical_actions.add_parser("smf-integration-canary-train")
    smf_train.add_argument("--config", required=True)
    smf_train.add_argument("--review-contract", required=True)
    smf_train.add_argument("--validation-evidence", required=True)
    smf_train.add_argument("--preflight-evidence", required=True)
    smf_train.add_argument("--dataset-root", required=True)
    smf_train.add_argument("--device", choices=("cuda",), default="cuda")
    smf_train.add_argument("--output", required=True)
    smf_train.add_argument("--resume-checkpoint")
    reconstruction_validate = optical_actions.add_parser("smf-reconstruction-validate")
    reconstruction_validate.add_argument("--config", required=True)
    reconstruction_validate.add_argument("--review-contract", required=True)
    reconstruction_validate.add_argument("--dataset-root", required=True)
    reconstruction_validate.add_argument("--output", required=True)
    reconstruction_preflight = optical_actions.add_parser(
        "smf-reconstruction-preflight"
    )
    reconstruction_preflight.add_argument("--config", required=True)
    reconstruction_preflight.add_argument("--review-contract", required=True)
    reconstruction_preflight.add_argument("--validation-evidence", required=True)
    reconstruction_preflight.add_argument("--dataset-root", required=True)
    reconstruction_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    reconstruction_preflight.add_argument("--output", required=True)
    reconstruction_train = optical_actions.add_parser("smf-reconstruction-train")
    reconstruction_train.add_argument("--config", required=True)
    reconstruction_train.add_argument("--review-contract", required=True)
    reconstruction_train.add_argument("--validation-evidence", required=True)
    reconstruction_train.add_argument("--preflight-evidence", required=True)
    reconstruction_train.add_argument("--dataset-root", required=True)
    reconstruction_train.add_argument("--device", choices=("cuda",), default="cuda")
    reconstruction_train.add_argument("--output", required=True)
    reconstruction_train.add_argument("--resume-checkpoint")
    fmf_validate = optical_actions.add_parser("fmf-m2-integration-canary-validate")
    fmf_validate.add_argument("--config", required=True)
    fmf_validate.add_argument("--review-contract", required=True)
    fmf_validate.add_argument("--dataset-root", required=True)
    fmf_validate.add_argument("--output", required=True)
    fmf_preflight = optical_actions.add_parser("fmf-m2-integration-canary-preflight")
    fmf_preflight.add_argument("--config", required=True)
    fmf_preflight.add_argument("--review-contract", required=True)
    fmf_preflight.add_argument("--validation-evidence", required=True)
    fmf_preflight.add_argument("--dataset-root", required=True)
    fmf_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    fmf_preflight.add_argument("--output", required=True)
    fmf_train = optical_actions.add_parser("fmf-m2-integration-canary-train")
    fmf_train.add_argument("--config", required=True)
    fmf_train.add_argument("--review-contract", required=True)
    fmf_train.add_argument("--validation-evidence", required=True)
    fmf_train.add_argument("--preflight-evidence", required=True)
    fmf_train.add_argument("--dataset-root", required=True)
    fmf_train.add_argument("--device", choices=("cuda",), default="cuda")
    fmf_train.add_argument("--output", required=True)
    fmf_train.add_argument("--resume-checkpoint")
    vae_validate = optical_actions.add_parser("smf-vae-pilot-validate")
    vae_validate.add_argument("--config", required=True)
    vae_validate.add_argument("--review-contract", required=True)
    vae_validate.add_argument("--dataset-root", required=True)
    vae_validate.add_argument("--output", required=True)
    vae_preflight = optical_actions.add_parser("smf-vae-pilot-preflight")
    vae_preflight.add_argument("--config", required=True)
    vae_preflight.add_argument("--review-contract", required=True)
    vae_preflight.add_argument("--validation-evidence", required=True)
    vae_preflight.add_argument("--dataset-root", required=True)
    vae_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    vae_preflight.add_argument("--output", required=True)
    vae_train = optical_actions.add_parser("smf-vae-pilot-train")
    vae_train.add_argument("--config", required=True)
    vae_train.add_argument("--review-contract", required=True)
    vae_train.add_argument("--validation-evidence", required=True)
    vae_train.add_argument("--preflight-evidence", required=True)
    vae_train.add_argument("--dataset-root", required=True)
    vae_train.add_argument("--device", choices=("cuda",), default="cuda")
    vae_train.add_argument("--output", required=True)
    vae_train.add_argument("--resume-checkpoint")
    kl_validate = optical_actions.add_parser("smf-kl-chain-audit-validate")
    kl_validate.add_argument("--config", required=True)
    kl_validate.add_argument("--review-contract", required=True)
    kl_validate.add_argument("--dataset-root", required=True)
    kl_validate.add_argument("--output", required=True)
    kl_run = optical_actions.add_parser("smf-kl-chain-audit-run")
    kl_run.add_argument("--config", required=True)
    kl_run.add_argument("--review-contract", required=True)
    kl_run.add_argument("--validation-evidence", required=True)
    kl_run.add_argument("--dataset-root", required=True)
    kl_run.add_argument("--device", choices=("cuda",), default="cuda")
    kl_run.add_argument("--output", required=True)
    dose_validate = optical_actions.add_parser("smf-kl-dose-reaudit-validate")
    dose_validate.add_argument("--config", required=True)
    dose_validate.add_argument("--review-contract", required=True)
    dose_validate.add_argument("--dataset-root", required=True)
    dose_validate.add_argument("--output", required=True)
    dose_run = optical_actions.add_parser("smf-kl-dose-reaudit-run")
    dose_run.add_argument("--config", required=True)
    dose_run.add_argument("--review-contract", required=True)
    dose_run.add_argument("--validation-evidence", required=True)
    dose_run.add_argument("--dataset-root", required=True)
    dose_run.add_argument("--device", choices=("cuda",), default="cuda")
    dose_run.add_argument("--output", required=True)
    effect_validate = optical_actions.add_parser("smf-kl-effect-pilot-validate")
    effect_validate.add_argument("--config", required=True)
    effect_validate.add_argument("--review-contract", required=True)
    effect_validate.add_argument("--dataset-root", required=True)
    effect_validate.add_argument("--output", required=True)
    effect_preflight = optical_actions.add_parser("smf-kl-effect-pilot-preflight")
    effect_preflight.add_argument("--config", required=True)
    effect_preflight.add_argument("--review-contract", required=True)
    effect_preflight.add_argument("--validation-evidence", required=True)
    effect_preflight.add_argument("--dataset-root", required=True)
    effect_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    effect_preflight.add_argument("--output", required=True)
    effect_train = optical_actions.add_parser("smf-kl-effect-pilot-train")
    effect_train.add_argument("--config", required=True)
    effect_train.add_argument("--review-contract", required=True)
    effect_train.add_argument("--validation-evidence", required=True)
    effect_train.add_argument("--preflight-evidence", required=True)
    effect_train.add_argument("--dataset-root", required=True)
    effect_train.add_argument("--device", choices=("cuda",), default="cuda")
    effect_train.add_argument("--output", required=True)
    effect_train.add_argument("--resume-checkpoint")
    ols_validate = optical_actions.add_parser("smf-ols-loss-recovery-validate")
    ols_validate.add_argument("--config", required=True)
    ols_validate.add_argument("--review-contract", required=True)
    ols_validate.add_argument("--dataset-root", required=True)
    ols_validate.add_argument("--output", required=True)
    ols_preflight = optical_actions.add_parser("smf-ols-loss-recovery-preflight")
    ols_preflight.add_argument("--config", required=True)
    ols_preflight.add_argument("--review-contract", required=True)
    ols_preflight.add_argument("--validation-evidence", required=True)
    ols_preflight.add_argument("--dataset-root", required=True)
    ols_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    ols_preflight.add_argument("--output", required=True)
    ols_train = optical_actions.add_parser("smf-ols-loss-recovery-train")
    ols_train.add_argument("--config", required=True)
    ols_train.add_argument("--review-contract", required=True)
    ols_train.add_argument("--validation-evidence", required=True)
    ols_train.add_argument("--preflight-evidence", required=True)
    ols_train.add_argument("--dataset-root", required=True)
    ols_train.add_argument("--device", choices=("cuda",), default="cuda")
    ols_train.add_argument("--output", required=True)
    ols_train.add_argument("--resume-checkpoint")
    bottleneck_validate = optical_actions.add_parser("smf-bottleneck-attribution-validate")
    bottleneck_validate.add_argument("--config", required=True)
    bottleneck_validate.add_argument("--review-contract", required=True)
    bottleneck_validate.add_argument("--dataset-root", required=True)
    bottleneck_validate.add_argument("--output", required=True)
    bottleneck_run = optical_actions.add_parser("smf-bottleneck-attribution-run")
    bottleneck_run.add_argument("--config", required=True)
    bottleneck_run.add_argument("--review-contract", required=True)
    bottleneck_run.add_argument("--validation-evidence", required=True)
    bottleneck_run.add_argument("--dataset-root", required=True)
    bottleneck_run.add_argument("--device", choices=("cuda",), default="cuda")
    bottleneck_run.add_argument("--output", required=True)
    geometry_validate = optical_actions.add_parser("scaled-chen-geometry-attribution-validate")
    geometry_validate.add_argument("--config", required=True)
    geometry_validate.add_argument("--review-contract", required=True)
    geometry_validate.add_argument("--dataset-root", required=True)
    geometry_validate.add_argument("--output", required=True)
    geometry_run = optical_actions.add_parser("scaled-chen-geometry-attribution-run")
    geometry_run.add_argument("--config", required=True)
    geometry_run.add_argument("--review-contract", required=True)
    geometry_run.add_argument("--validation-evidence", required=True)
    geometry_run.add_argument("--dataset-root", required=True)
    geometry_run.add_argument("--device", choices=("cuda",), default="cuda")
    geometry_run.add_argument("--output", required=True)
    readiness_validate = optical_actions.add_parser("scaled-chen-manifold-readiness-validate")
    readiness_validate.add_argument("--config", required=True)
    readiness_validate.add_argument("--review-contract", required=True)
    readiness_validate.add_argument("--dataset-root", required=True)
    readiness_validate.add_argument("--output", required=True)
    readiness_run = optical_actions.add_parser("scaled-chen-manifold-readiness-run")
    readiness_run.add_argument("--config", required=True)
    readiness_run.add_argument("--review-contract", required=True)
    readiness_run.add_argument("--validation-evidence", required=True)
    readiness_run.add_argument("--dataset-root", required=True)
    readiness_run.add_argument("--device", choices=("cuda",), default="cuda")
    readiness_run.add_argument("--output", required=True)
    decoder_cmos_validate = optical_actions.add_parser("scaled-chen-decoder-cmos-attribution-validate")
    decoder_cmos_validate.add_argument("--config", required=True)
    decoder_cmos_validate.add_argument("--review-contract", required=True)
    decoder_cmos_validate.add_argument("--dataset-root", required=True)
    decoder_cmos_validate.add_argument("--output", required=True)
    decoder_cmos_run = optical_actions.add_parser("scaled-chen-decoder-cmos-attribution-run")
    decoder_cmos_run.add_argument("--config", required=True)
    decoder_cmos_run.add_argument("--review-contract", required=True)
    decoder_cmos_run.add_argument("--validation-evidence", required=True)
    decoder_cmos_run.add_argument("--dataset-root", required=True)
    decoder_cmos_run.add_argument("--device", choices=("cuda",), default="cuda")
    decoder_cmos_run.add_argument("--output", required=True)
    backend_observable_validate = optical_actions.add_parser("scaled-chen-backend-observable-subspace-validate")
    backend_observable_validate.add_argument("--config", required=True)
    backend_observable_validate.add_argument("--review-contract", required=True)
    backend_observable_validate.add_argument("--dataset-root", required=True)
    backend_observable_validate.add_argument("--output", required=True)
    backend_observable_run = optical_actions.add_parser("scaled-chen-backend-observable-subspace-run")
    backend_observable_run.add_argument("--config", required=True)
    backend_observable_run.add_argument("--review-contract", required=True)
    backend_observable_run.add_argument("--validation-evidence", required=True)
    backend_observable_run.add_argument("--dataset-root", required=True)
    backend_observable_run.add_argument("--device", choices=("cuda",), default="cuda")
    backend_observable_run.add_argument("--output", required=True)
    hfrr_validate = optical_actions.add_parser("scaled-chen-high-fidelity-relay-reference-validate")
    hfrr_validate.add_argument("--config", required=True)
    hfrr_validate.add_argument("--review-contract", required=True)
    hfrr_validate.add_argument("--dataset-root", required=True)
    hfrr_validate.add_argument("--output", required=True)
    hfrr_run = optical_actions.add_parser("scaled-chen-high-fidelity-relay-reference-run")
    hfrr_run.add_argument("--config", required=True)
    hfrr_run.add_argument("--review-contract", required=True)
    hfrr_run.add_argument("--validation-evidence", required=True)
    hfrr_run.add_argument("--dataset-root", required=True)
    hfrr_run.add_argument("--device", choices=("cuda",), default="cuda")
    hfrr_run.add_argument("--output", required=True)
    rmri_validate = optical_actions.add_parser("scaled-chen-relay-metric-reinterpretation-validate")
    rmri_validate.add_argument("--config", required=True)
    rmri_validate.add_argument("--review-contract", required=True)
    rmri_validate.add_argument("--dataset-root", required=True)
    rmri_validate.add_argument("--output", required=True)
    rmri_run = optical_actions.add_parser("scaled-chen-relay-metric-reinterpretation-run")
    rmri_run.add_argument("--config", required=True)
    rmri_run.add_argument("--review-contract", required=True)
    rmri_run.add_argument("--validation-evidence", required=True)
    rmri_run.add_argument("--dataset-root", required=True)
    rmri_run.add_argument("--device", choices=("cuda",), default="cuda")
    rmri_run.add_argument("--output", required=True)
    trba_validate = optical_actions.add_parser("scaled-chen-task-relevant-bottleneck-validate")
    trba_validate.add_argument("--config", required=True)
    trba_validate.add_argument("--review-contract", required=True)
    trba_validate.add_argument("--dataset-root", required=True)
    trba_validate.add_argument("--output", required=True)
    trba_run = optical_actions.add_parser("scaled-chen-task-relevant-bottleneck-run")
    trba_run.add_argument("--config", required=True)
    trba_run.add_argument("--review-contract", required=True)
    trba_run.add_argument("--validation-evidence", required=True)
    trba_run.add_argument("--dataset-root", required=True)
    trba_run.add_argument("--device", choices=("cuda",), default="cuda")
    trba_run.add_argument("--output", required=True)
    rmfa_validate = optical_actions.add_parser("scaled-chen-reconstruction-manifold-failure-validate")
    rmfa_validate.add_argument("--config", required=True)
    rmfa_validate.add_argument("--review-contract", required=True)
    rmfa_validate.add_argument("--dataset-root", required=True)
    rmfa_validate.add_argument("--output", required=True)
    rmfa_run = optical_actions.add_parser("scaled-chen-reconstruction-manifold-failure-run")
    rmfa_run.add_argument("--config", required=True)
    rmfa_run.add_argument("--review-contract", required=True)
    rmfa_run.add_argument("--validation-evidence", required=True)
    rmfa_run.add_argument("--dataset-root", required=True)
    rmfa_run.add_argument("--device", choices=("cuda",), default="cuda")
    rmfa_run.add_argument("--output", required=True)
    rmal_validate = optical_actions.add_parser("scaled-chen-reconstruction-mapping-alignment-localization-validate")
    rmal_validate.add_argument("--config", required=True)
    rmal_validate.add_argument("--review-contract", required=True)
    rmal_validate.add_argument("--dataset-root", required=True)
    rmal_validate.add_argument("--output", required=True)
    rmal_run = optical_actions.add_parser("scaled-chen-reconstruction-mapping-alignment-localization-run")
    rmal_run.add_argument("--config", required=True)
    rmal_run.add_argument("--review-contract", required=True)
    rmal_run.add_argument("--validation-evidence", required=True)
    rmal_run.add_argument("--dataset-root", required=True)
    rmal_run.add_argument("--device", choices=("cuda",), default="cuda")
    rmal_run.add_argument("--output", required=True)
    rwsa_validate = optical_actions.add_parser("scaled-chen-detector-readout-structure-attribution-validate")
    rwsa_validate.add_argument("--config", required=True)
    rwsa_validate.add_argument("--review-contract", required=True)
    rwsa_validate.add_argument("--dataset-root", required=True)
    rwsa_validate.add_argument("--output", required=True)
    rwsa_run = optical_actions.add_parser("scaled-chen-detector-readout-structure-attribution-run")
    rwsa_run.add_argument("--config", required=True)
    rwsa_run.add_argument("--review-contract", required=True)
    rwsa_run.add_argument("--validation-evidence", required=True)
    rwsa_run.add_argument("--dataset-root", required=True)
    rwsa_run.add_argument("--device", choices=("cuda",), default="cuda")
    rwsa_run.add_argument("--output", required=True)
    zcsa_validate = optical_actions.add_parser("scaled-chen-detector-readout-zero-column-semantics-audit-validate")
    zcsa_validate.add_argument("--config", required=True)
    zcsa_validate.add_argument("--review-contract", required=True)
    zcsa_validate.add_argument("--dataset-root", required=True)
    zcsa_validate.add_argument("--output", required=True)
    zcsa_run = optical_actions.add_parser("scaled-chen-detector-readout-zero-column-semantics-audit-run")
    zcsa_run.add_argument("--config", required=True)
    zcsa_run.add_argument("--review-contract", required=True)
    zcsa_run.add_argument("--validation-evidence", required=True)
    zcsa_run.add_argument("--dataset-root", required=True)
    zcsa_run.add_argument("--output", required=True)
    rwsa_active_validate = optical_actions.add_parser(
        "scaled-chen-detector-readout-structure-active-domain-validate"
    )
    rwsa_active_validate.add_argument("--config", required=True)
    rwsa_active_validate.add_argument("--review-contract", required=True)
    rwsa_active_validate.add_argument("--dataset-root", required=True)
    rwsa_active_validate.add_argument("--output", required=True)
    rwsa_active_run = optical_actions.add_parser(
        "scaled-chen-detector-readout-structure-active-domain-run"
    )
    rwsa_active_run.add_argument("--config", required=True)
    rwsa_active_run.add_argument("--review-contract", required=True)
    rwsa_active_run.add_argument("--validation-evidence", required=True)
    rwsa_active_run.add_argument("--dataset-root", required=True)
    rwsa_active_run.add_argument("--device", choices=("cuda",), default="cuda")
    rwsa_active_run.add_argument("--output", required=True)
    lsda_validate = optical_actions.add_parser(
        "scaled-chen-linear-readout-structure-decomposition-validate"
    )
    lsda_validate.add_argument("--config", required=True)
    lsda_validate.add_argument("--review-contract", required=True)
    lsda_validate.add_argument("--dataset-root", default="data")
    lsda_validate.add_argument("--output", required=True)
    lsda_run = optical_actions.add_parser(
        "scaled-chen-linear-readout-structure-decomposition-run"
    )
    lsda_run.add_argument("--config", required=True)
    lsda_run.add_argument("--review-contract", required=True)
    lsda_run.add_argument("--validation-evidence", required=True)
    lsda_run.add_argument("--dataset-root", default="data")
    lsda_run.add_argument("--output", required=True)
    lrrc_validate = optical_actions.add_parser(
        "scaled-chen-low-rank-readout-recovery-confirmation-validate"
    )
    lrrc_validate.add_argument("--config", required=True)
    lrrc_validate.add_argument("--review-contract", required=True)
    lrrc_validate.add_argument("--dataset-root", default="data")
    lrrc_validate.add_argument("--output", required=True)
    lrrc_run = optical_actions.add_parser(
        "scaled-chen-low-rank-readout-recovery-confirmation-run"
    )
    lrrc_run.add_argument("--config", required=True)
    lrrc_run.add_argument("--review-contract", required=True)
    lrrc_run.add_argument("--validation-evidence", required=True)
    lrrc_run.add_argument("--dataset-root", default="data")
    lrrc_run.add_argument("--device", choices=("cuda",), default="cuda")
    lrrc_run.add_argument("--output", required=True)
    dsfa_validate = optical_actions.add_parser(
        "scaled-chen-data-weighted-spectral-functional-attribution-validate"
    )
    dsfa_validate.add_argument("--config", required=True)
    dsfa_validate.add_argument("--review-contract", required=True)
    dsfa_validate.add_argument("--dataset-root", default="data")
    dsfa_validate.add_argument("--output", required=True)
    dsfa_run = optical_actions.add_parser(
        "scaled-chen-data-weighted-spectral-functional-attribution-run"
    )
    dsfa_run.add_argument("--config", required=True)
    dsfa_run.add_argument("--review-contract", required=True)
    dsfa_run.add_argument("--validation-evidence", required=True)
    dsfa_run.add_argument("--dataset-root", default="data")
    dsfa_run.add_argument("--device", choices=("cuda",), default="cuda")
    dsfa_run.add_argument("--output", required=True)
    dmsa_validate = optical_actions.add_parser(
        "scaled-chen-detector-data-metric-spectral-attribution-validate"
    )
    dmsa_validate.add_argument("--config", required=True)
    dmsa_validate.add_argument("--review-contract", required=True)
    dmsa_validate.add_argument("--dataset-root", default="data")
    dmsa_validate.add_argument("--output", required=True)
    dmsa_run = optical_actions.add_parser(
        "scaled-chen-detector-data-metric-spectral-attribution-run"
    )
    dmsa_run.add_argument("--config", required=True)
    dmsa_run.add_argument("--review-contract", required=True)
    dmsa_run.add_argument("--validation-evidence", required=True)
    dmsa_run.add_argument("--dataset-root", default="data")
    dmsa_run.add_argument("--device", choices=("cuda",), default="cuda")
    dmsa_run.add_argument("--output", required=True)
    mcfa_validate = optical_actions.add_parser(
        "scaled-chen-detector-mean-centered-functional-attribution-validate"
    )
    mcfa_validate.add_argument("--config", required=True)
    mcfa_validate.add_argument("--review-contract", required=True)
    mcfa_validate.add_argument("--dataset-root", default="data")
    mcfa_validate.add_argument("--output", required=True)
    mcfa_run = optical_actions.add_parser(
        "scaled-chen-detector-mean-centered-functional-attribution-run"
    )
    mcfa_run.add_argument("--config", required=True)
    mcfa_run.add_argument("--review-contract", required=True)
    mcfa_run.add_argument("--validation-evidence", required=True)
    mcfa_run.add_argument("--dataset-root", default="data")
    mcfa_run.add_argument("--device", choices=("cuda",), default="cuda")
    mcfa_run.add_argument("--output", required=True)
    mpcdar_validate = optical_actions.add_parser(
        "scaled-chen-mean-preserving-centered-data-aware-readout-recovery-validate"
    )
    mpcdar_validate.add_argument("--config", required=True)
    mpcdar_validate.add_argument("--review-contract", required=True)
    mpcdar_validate.add_argument("--dataset-root", default="data")
    mpcdar_validate.add_argument("--output", required=True)
    mpcdar_run = optical_actions.add_parser(
        "scaled-chen-mean-preserving-centered-data-aware-readout-recovery-run"
    )
    mpcdar_run.add_argument("--config", required=True)
    mpcdar_run.add_argument("--review-contract", required=True)
    mpcdar_run.add_argument("--validation-evidence", required=True)
    mpcdar_run.add_argument("--dataset-root", default="data")
    mpcdar_run.add_argument("--device", choices=("cuda",), default="cuda")
    mpcdar_run.add_argument("--output", required=True)
    mpcdar_rank16_build = optical_actions.add_parser(
        "scaled-chen-mpcdar-rank16-integration-qualification-build"
    )
    mpcdar_rank16_build.add_argument("--config", required=True)
    mpcdar_rank16_build.add_argument("--review-contract", required=True)
    mpcdar_rank16_build.add_argument("--dataset-root", default="data")
    mpcdar_rank16_build.add_argument("--device", choices=("cuda",), default="cuda")
    mpcdar_rank16_build.add_argument("--artifact-dir", required=True)
    mpcdar_rank16_qualify = optical_actions.add_parser(
        "scaled-chen-mpcdar-rank16-integration-qualification-qualify"
    )
    mpcdar_rank16_qualify.add_argument("--config", required=True)
    mpcdar_rank16_qualify.add_argument("--review-contract", required=True)
    mpcdar_rank16_qualify.add_argument("--dataset-root", default="data")
    mpcdar_rank16_qualify.add_argument("--device", choices=("cuda",), default="cuda")
    mpcdar_rank16_qualify.add_argument("--artifact-dir", required=True)
    mpcdar_rank16_qualify.add_argument("--output", required=True)
    mpcdar_rank16_f32_build = optical_actions.add_parser(
        "scaled-chen-mpcdar-rank16-float32-numerical-qualification-build"
    )
    mpcdar_rank16_f32_build.add_argument("--config", required=True)
    mpcdar_rank16_f32_build.add_argument("--review-contract", required=True)
    mpcdar_rank16_f32_build.add_argument("--parent-manifest", required=True)
    mpcdar_rank16_f32_build.add_argument("--artifact-dir", required=True)
    mpcdar_rank16_f32_qualify = optical_actions.add_parser(
        "scaled-chen-mpcdar-rank16-float32-numerical-qualification-qualify"
    )
    mpcdar_rank16_f32_qualify.add_argument("--config", required=True)
    mpcdar_rank16_f32_qualify.add_argument("--review-contract", required=True)
    mpcdar_rank16_f32_qualify.add_argument("--dataset-root", default="data")
    mpcdar_rank16_f32_qualify.add_argument("--device", choices=("cuda",), default="cuda")
    mpcdar_rank16_f32_qualify.add_argument("--artifact-dir", required=True)
    mpcdar_rank16_f32_qualify.add_argument("--output", required=True)
    runtime_pq_validate = optical_actions.add_parser(
        "scaled-chen-mpcdar-rank16-runtime-performance-qualification-validate"
    )
    runtime_pq_validate.add_argument("--config", required=True)
    runtime_pq_validate.add_argument("--review-contract", required=True)
    runtime_pq_validate.add_argument("--artifact-dir", required=True)
    runtime_pq_validate.add_argument("--output", required=True)
    runtime_pq_run = optical_actions.add_parser(
        "scaled-chen-mpcdar-rank16-runtime-performance-qualification-run"
    )
    runtime_pq_run.add_argument("--config", required=True)
    runtime_pq_run.add_argument("--review-contract", required=True)
    runtime_pq_run.add_argument("--validation-evidence", required=True)
    runtime_pq_run.add_argument("--artifact-dir", required=True)
    runtime_pq_run.add_argument("--device", choices=("cpu",), default="cpu")
    runtime_pq_run.add_argument("--output", required=True)
    back_relay_validate = optical_actions.add_parser("scaled-chen-back-relay-axial-conjugate-validate")
    back_relay_validate.add_argument("--config", required=True)
    back_relay_validate.add_argument("--review-contract", required=True)
    back_relay_validate.add_argument("--dataset-root", required=True)
    back_relay_validate.add_argument("--output", required=True)
    back_relay_preflight = optical_actions.add_parser("scaled-chen-back-relay-axial-conjugate-preflight")
    back_relay_preflight.add_argument("--config", required=True)
    back_relay_preflight.add_argument("--review-contract", required=True)
    back_relay_preflight.add_argument("--validation-evidence", required=True)
    back_relay_preflight.add_argument("--dataset-root", required=True)
    back_relay_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    back_relay_preflight.add_argument("--output", required=True)
    back_relay_run = optical_actions.add_parser("scaled-chen-back-relay-axial-conjugate-run")
    back_relay_run.add_argument("--config", required=True)
    back_relay_run.add_argument("--review-contract", required=True)
    back_relay_run.add_argument("--validation-evidence", required=True)
    back_relay_run.add_argument("--preflight-evidence", required=True)
    back_relay_run.add_argument("--dataset-root", required=True)
    back_relay_run.add_argument("--device", choices=("cuda",), default="cuda")
    back_relay_run.add_argument("--output", required=True)
    back_relay_report = optical_actions.add_parser("scaled-chen-back-relay-axial-conjugate-report")
    back_relay_report.add_argument("--config", required=True)
    back_relay_report.add_argument("--review-contract", required=True)
    back_relay_report.add_argument("--audit-evidence", required=True)
    back_relay_report.add_argument("--dataset-root", required=True)
    back_relay_report.add_argument("--output", required=True)
    back_relay_closeout = optical_actions.add_parser("scaled-chen-back-relay-axial-conjugate-closeout")
    back_relay_closeout.add_argument("--config", required=True)
    back_relay_closeout.add_argument("--review-contract", required=True)
    back_relay_closeout.add_argument("--report-evidence", required=True)
    back_relay_closeout.add_argument("--dataset-root", required=True)
    back_relay_closeout.add_argument("--output", required=True)
    full_validate = optical_actions.add_parser("full-epoch-validate")
    full_validate.add_argument("--config", required=True)
    full_validate.add_argument("--review-contract", required=True)
    full_validate.add_argument("--pilot-training-evidence", required=True)
    full_validate.add_argument("--pilot-checkpoint", required=True)
    full_validate.add_argument("--dataset-root", required=True)
    full_validate.add_argument("--output", required=True)
    full_preflight = optical_actions.add_parser("full-epoch-preflight")
    full_preflight.add_argument("--config", required=True)
    full_preflight.add_argument("--review-contract", required=True)
    full_preflight.add_argument("--epoch-validation-evidence", required=True)
    full_preflight.add_argument("--pilot-training-evidence", required=True)
    full_preflight.add_argument("--pilot-checkpoint", required=True)
    full_preflight.add_argument("--dataset-root", required=True)
    full_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    full_preflight.add_argument("--output", required=True)
    full_train = optical_actions.add_parser("full-epoch-train")
    full_train.add_argument("--config", required=True)
    full_train.add_argument("--review-contract", required=True)
    full_train.add_argument("--epoch-validation-evidence", required=True)
    full_train.add_argument("--epoch-preflight-evidence", required=True)
    full_train.add_argument("--pilot-training-evidence", required=True)
    full_train.add_argument("--pilot-checkpoint", required=True)
    full_train.add_argument("--dataset-root", required=True)
    full_train.add_argument("--device", choices=("cuda",), default="cuda")
    full_train.add_argument("--output", required=True)
    full_train.add_argument("--resume-checkpoint")
    audit_validate = optical_actions.add_parser("reconstruction-audit-validate")
    audit_validate.add_argument("--config", required=True)
    audit_validate.add_argument("--review-contract", required=True)
    audit_validate.add_argument("--source-training-evidence", required=True)
    audit_validate.add_argument("--source-checkpoint", required=True)
    audit_validate.add_argument("--dataset-root", required=True)
    audit_validate.add_argument("--output", required=True)
    audit_run = optical_actions.add_parser("reconstruction-audit-run")
    audit_run.add_argument("--config", required=True)
    audit_run.add_argument("--review-contract", required=True)
    audit_run.add_argument("--validation-evidence", required=True)
    audit_run.add_argument("--source-training-evidence", required=True)
    audit_run.add_argument("--source-checkpoint", required=True)
    audit_run.add_argument("--dataset-root", required=True)
    audit_run.add_argument("--device", choices=("cuda",), default="cuda")
    audit_run.add_argument("--output", required=True)
    observer_validate = optical_actions.add_parser("latent-observer-validate")
    observer_validate.add_argument("--config", required=True)
    observer_validate.add_argument("--review-contract", required=True)
    observer_validate.add_argument("--dataset-root", required=True)
    observer_validate.add_argument("--output", required=True)
    observer_run = optical_actions.add_parser("latent-observer-run")
    observer_run.add_argument("--config", required=True)
    observer_run.add_argument("--review-contract", required=True)
    observer_run.add_argument("--validation-evidence", required=True)
    observer_run.add_argument("--dataset-root", required=True)
    observer_run.add_argument("--device", choices=("cuda",), default="cuda")
    observer_run.add_argument("--output", required=True)
    regularization_validate = optical_actions.add_parser(
        "manifold-regularization-audit-validate"
    )
    regularization_validate.add_argument("--config", required=True)
    regularization_validate.add_argument("--review-contract", required=True)
    regularization_validate.add_argument("--dataset-root", required=True)
    regularization_validate.add_argument("--output", required=True)
    regularization_run = optical_actions.add_parser("manifold-regularization-audit-run")
    regularization_run.add_argument("--config", required=True)
    regularization_run.add_argument("--review-contract", required=True)
    regularization_run.add_argument("--validation-evidence", required=True)
    regularization_run.add_argument("--dataset-root", required=True)
    regularization_run.add_argument("--device", choices=("cuda",), default="cuda")
    regularization_run.add_argument("--output", required=True)
    pilot_validate = optical_actions.add_parser(
        "manifold-regularization-pilot-validate"
    )
    pilot_validate.add_argument("--config", required=True)
    pilot_validate.add_argument("--review-contract", required=True)
    pilot_validate.add_argument("--dataset-root", required=True)
    pilot_validate.add_argument("--output", required=True)
    pilot_preflight = optical_actions.add_parser(
        "manifold-regularization-pilot-preflight"
    )
    pilot_preflight.add_argument("--config", required=True)
    pilot_preflight.add_argument("--review-contract", required=True)
    pilot_preflight.add_argument("--training-validation-evidence", required=True)
    pilot_preflight.add_argument("--dataset-root", required=True)
    pilot_preflight.add_argument("--device", choices=("cuda",), default="cuda")
    pilot_preflight.add_argument("--output", required=True)
    pilot_train = optical_actions.add_parser("manifold-regularization-pilot-train")
    pilot_train.add_argument("--config", required=True)
    pilot_train.add_argument("--review-contract", required=True)
    pilot_train.add_argument("--training-validation-evidence", required=True)
    pilot_train.add_argument("--training-preflight-evidence", required=True)
    pilot_train.add_argument("--dataset-root", required=True)
    pilot_train.add_argument("--device", choices=("cuda",), default="cuda")
    pilot_train.add_argument("--output", required=True)
    pilot_train.add_argument("--resume-checkpoint")
    fiber = research_programs.add_parser(
        "fiber-robustness", help="synthetic multi-matrix fiber diagnostics"
    )
    fiber_actions = fiber.add_subparsers(dest="research_action", required=True)
    validate = fiber_actions.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--lp01-fixture", required=True)
    validate.add_argument("--dataset-root", required=True)
    validate.add_argument("--primary-evaluator-checkpoint", required=True)
    validate.add_argument("--secondary-evaluator-checkpoint", required=True)
    validate.add_argument("--primary-qualification-evidence", required=True)
    validate.add_argument("--secondary-qualification-evidence", required=True)
    validate.add_argument("--measurement-bundle-output", required=True)
    validate.add_argument("--measurement-evidence-output", required=True)
    validate.add_argument("--output", required=True)
    preflight = fiber_actions.add_parser("preflight")
    preflight.add_argument("--validate-evidence", required=True)
    preflight.add_argument("--config", required=True)
    preflight.add_argument(
        "--research-initialization-checkpoint",
        "--checkpoint",
        dest="research_initialization_checkpoint",
        required=True,
    )
    preflight.add_argument("--source-training-evidence", required=True)
    preflight.add_argument("--measurement-bundle", required=True)
    preflight.add_argument("--measurement-evidence", required=True)
    preflight.add_argument("--primary-qualification-evidence", required=True)
    preflight.add_argument("--secondary-qualification-evidence", required=True)
    preflight.add_argument("--dataset-root", required=True)
    preflight.add_argument(
        "--primary-evaluator-checkpoint",
        "--evaluator-checkpoint",
        dest="primary_evaluator_checkpoint",
        required=True,
    )
    preflight.add_argument("--secondary-evaluator-checkpoint", required=True)
    preflight.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    preflight.add_argument("--output", required=True)
    screen = fiber_actions.add_parser("screen")
    screen.add_argument("--preflight-evidence", required=True)
    screen.add_argument("--config", required=True)
    screen.add_argument(
        "--research-initialization-checkpoint",
        "--checkpoint",
        dest="research_initialization_checkpoint",
        required=True,
    )
    screen.add_argument("--source-training-evidence", required=True)
    screen.add_argument("--measurement-bundle", required=True)
    screen.add_argument("--measurement-evidence", required=True)
    screen.add_argument("--primary-qualification-evidence", required=True)
    screen.add_argument("--secondary-qualification-evidence", required=True)
    screen.add_argument("--dataset-root", required=True)
    screen.add_argument(
        "--primary-evaluator-checkpoint",
        "--evaluator-checkpoint",
        dest="primary_evaluator_checkpoint",
        required=True,
    )
    screen.add_argument("--secondary-evaluator-checkpoint", required=True)
    screen.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    screen.add_argument("--output", required=True)
    select = fiber_actions.add_parser("select")
    select.add_argument("--screen-evidence", required=True)
    select.add_argument("--config", required=True)
    select.add_argument(
        "--research-initialization-checkpoint",
        "--checkpoint",
        dest="research_initialization_checkpoint",
        required=True,
    )
    select.add_argument("--source-training-evidence", required=True)
    select.add_argument("--measurement-bundle", required=True)
    select.add_argument("--measurement-evidence", required=True)
    select.add_argument("--primary-qualification-evidence", required=True)
    select.add_argument("--secondary-qualification-evidence", required=True)
    select.add_argument("--dataset-root", required=True)
    select.add_argument(
        "--primary-evaluator-checkpoint",
        "--evaluator-checkpoint",
        dest="primary_evaluator_checkpoint",
        required=True,
    )
    select.add_argument("--secondary-evaluator-checkpoint", required=True)
    select.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    select.add_argument("--output", required=True)
    closeout = fiber_actions.add_parser("closeout")
    closeout.add_argument("--selection-evidence", required=True)
    closeout.add_argument("--output", required=True)

    image = research_programs.add_parser(
        "fiber-image", help="paired synthetic fiber image-generation diagnostics"
    )
    image_actions = image.add_subparsers(dest="research_action", required=True)

    def image_common(command, *, validation=False):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if validation:
            command.add_argument("--lp01-fixture", required=True)
        else:
            command.add_argument(
                "--research-initialization-checkpoint",
                "--checkpoint",
                dest="research_initialization_checkpoint",
                required=True,
            )
            command.add_argument("--source-training-evidence", required=True)
            command.add_argument("--measurement-bundle", required=True)
            command.add_argument("--measurement-evidence", required=True)

    image_validate = image_actions.add_parser("validate")
    image_common(image_validate, validation=True)
    image_validate.add_argument("--measurement-bundle-output", required=True)
    image_validate.add_argument("--measurement-evidence-output", required=True)
    image_validate.add_argument("--output", required=True)
    image_baseline = image_actions.add_parser("baseline")
    image_common(image_baseline)
    image_baseline.add_argument("--validate-evidence", required=True)
    image_baseline.add_argument("--output", required=True)
    for action, upstream in (
        ("preflight", "validate-evidence"),
        ("screen", "preflight-evidence"),
    ):
        command = image_actions.add_parser(action)
        image_common(command)
        command.add_argument(
            f"--{upstream}", dest=upstream.replace("-", "_"), required=True
        )
        if action == "preflight":
            command.add_argument("--baseline-evidence", required=True)
        if action in {"preflight", "screen"}:
            command.add_argument(
                "--function-id", choices=("S0", "S1a", "S1b", "S1c"), required=True
            )
        if action == "screen":
            command.add_argument(
                "--seed", type=int, choices=(5201, 5202, 5203), required=True
            )
            command.add_argument("--resume-checkpoint")
        command.add_argument("--output", required=True)
    image_select = image_actions.add_parser("select")
    image_common(image_select)
    image_select.add_argument("--screen-evidence", action="append", required=True)
    image_select.add_argument("--output", required=True)
    image_closeout = image_actions.add_parser("closeout")
    image_closeout.add_argument("--selection-evidence", required=True)
    image_closeout.add_argument("--output", required=True)

    attribution = research_programs.add_parser(
        "fiber-attribution",
        help="no-update synthetic fiber attribution and presentation diagnostics",
    )
    attribution_actions = attribution.add_subparsers(
        dest="research_action", required=True
    )

    def attribution_lineage(command, *, fixture=False):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--research-initialization-checkpoint", required=True)
        command.add_argument("--source-training-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--v3-closeout", required=True)
        if fixture:
            command.add_argument("--lp01-fixture", required=True)

    attribution_validate = attribution_actions.add_parser("validate")
    attribution_lineage(attribution_validate, fixture=True)
    attribution_validate.add_argument("--output", required=True)
    attribution_run = attribution_actions.add_parser("run")
    attribution_lineage(attribution_run)
    attribution_run.add_argument("--validation-evidence", required=True)
    attribution_run.add_argument("--output", required=True)
    attribution_report = attribution_actions.add_parser("report")
    attribution_report.add_argument("--measurement-evidence", required=True)
    attribution_report.add_argument("--v3-selection", required=True)
    attribution_report.add_argument(
        "--digit-pipeline-variant", choices=("S0",), default="S0"
    )
    attribution_report.add_argument("--digit-pipeline-only", action="store_true")
    attribution_report.add_argument("--output", required=True)
    attribution_closeout = attribution_actions.add_parser("closeout")
    attribution_closeout.add_argument("--measurement-evidence", required=True)
    attribution_closeout.add_argument("--presentation-evidence", required=True)
    attribution_closeout.add_argument("--output", required=True)

    interface = research_programs.add_parser(
        "fiber-interface-attribution",
        help="no-update synthetic modal-interface and optical-readout attribution",
    )
    interface_actions = interface.add_subparsers(dest="research_action", required=True)

    def interface_lineage(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument(
            "--research-initialization-checkpoint",
            "--checkpoint",
            dest="research_initialization_checkpoint",
            required=True,
        )
        command.add_argument("--source-training-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--v3-closeout", required=True)
        command.add_argument("--attribution-measurement", required=True)

    interface_validate = interface_actions.add_parser("validate")
    interface_lineage(interface_validate)
    interface_validate.add_argument("--lp01-fixture", required=True)
    interface_validate.add_argument("--output", required=True)

    interface_run = interface_actions.add_parser("run")
    interface_lineage(interface_run)
    interface_run.add_argument("--lp01-fixture", required=True)
    interface_run.add_argument("--validation-evidence", required=True)
    interface_run.add_argument("--output", required=True)

    interface_report = interface_actions.add_parser("report")
    interface_report.add_argument("--measurement-evidence", required=True)
    interface_report.add_argument("--output", required=True)

    interface_closeout = interface_actions.add_parser("closeout")
    interface_closeout.add_argument("--measurement-evidence", required=True)
    interface_closeout.add_argument("--presentation-evidence", required=True)
    interface_closeout.add_argument("--output", required=True)

    exposure = research_programs.add_parser(
        "fiber-loss-exposure-attribution",
        help="no-update synthetic fiber loss and exposure-budget attribution",
    )
    exposure_actions = exposure.add_subparsers(dest="research_action", required=True)

    def exposure_lineage(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument(
            "--research-initialization-checkpoint",
            "--checkpoint",
            dest="research_initialization_checkpoint",
            required=True,
        )
        command.add_argument("--source-training-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--v3-closeout", required=True)
        command.add_argument("--attribution-measurement", required=True)
        command.add_argument("--interface-validation", required=True)
        command.add_argument("--interface-measurement", required=True)
        command.add_argument("--interface-closeout", required=True)

    exposure_validate = exposure_actions.add_parser("validate")
    exposure_lineage(exposure_validate)
    exposure_validate.add_argument("--lp01-fixture", required=True)
    exposure_validate.add_argument("--output", required=True)

    exposure_run = exposure_actions.add_parser("run")
    exposure_lineage(exposure_run)
    exposure_run.add_argument("--lp01-fixture", required=True)
    exposure_run.add_argument("--validation-evidence", required=True)
    exposure_run.add_argument("--output", required=True)

    exposure_report = exposure_actions.add_parser("report")
    exposure_report.add_argument("--measurement-evidence", required=True)
    exposure_report.add_argument("--output", required=True)

    exposure_closeout = exposure_actions.add_parser("closeout")
    exposure_closeout.add_argument("--measurement-evidence", required=True)
    exposure_closeout.add_argument("--presentation-evidence", required=True)
    exposure_closeout.add_argument("--output", required=True)

    decoder = research_programs.add_parser(
        "fiber-decoder-controllability-attribution",
        help="no-update synthetic zero-phase decoder controllability attribution",
    )
    decoder_actions = decoder.add_subparsers(dest="research_action", required=True)

    def decoder_lineage(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument(
            "--research-initialization-checkpoint",
            "--checkpoint",
            dest="research_initialization_checkpoint",
            required=True,
        )
        command.add_argument("--source-training-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--v3-closeout", required=True)
        command.add_argument("--attribution-measurement", required=True)
        command.add_argument("--interface-validation", required=True)
        command.add_argument("--interface-measurement", required=True)
        command.add_argument("--interface-closeout", required=True)
        command.add_argument("--loss-exposure-validation", required=True)
        command.add_argument("--loss-exposure-measurement", required=True)
        command.add_argument("--loss-exposure-presentation", required=True)
        command.add_argument("--loss-exposure-closeout", required=True)

    decoder_validate = decoder_actions.add_parser("validate")
    decoder_lineage(decoder_validate)
    decoder_validate.add_argument("--lp01-fixture", required=True)
    decoder_validate.add_argument("--output", required=True)

    decoder_run = decoder_actions.add_parser("run")
    decoder_lineage(decoder_run)
    decoder_run.add_argument("--lp01-fixture", required=True)
    decoder_run.add_argument("--validation-evidence", required=True)
    decoder_run.add_argument("--output", required=True)

    decoder_report = decoder_actions.add_parser("report")
    decoder_report.add_argument("--measurement-evidence", required=True)
    decoder_report.add_argument("--output", required=True)

    decoder_closeout = decoder_actions.add_parser("closeout")
    decoder_closeout.add_argument("--measurement-evidence", required=True)
    decoder_closeout.add_argument("--presentation-evidence", required=True)
    decoder_closeout.add_argument("--output", required=True)

    capacity_map = research_programs.add_parser("capacity-map")
    capacity_actions = capacity_map.add_subparsers(
        dest="research_action", required=True
    )
    for action in (
        "validate",
        "screen",
        "train-smoke",
        "confirm",
        "summarize",
        "diagnose",
        "qualify-shape",
        "closeout",
    ):
        command = capacity_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument(
            "--dataset",
            choices=("mnist", "controlled_shapes", "fashion_mnist"),
            default="mnist",
        )
        command.add_argument(
            "--device",
            choices=("cuda", "cpu"),
            default="cpu" if action == "closeout" else "cuda",
        )
        if action not in {"validate", "summarize", "qualify-shape", "closeout"}:
            command.add_argument("--variant", required=True)
        else:
            command.add_argument("--variant")
        if action in {"screen", "train-smoke", "confirm", "diagnose", "qualify-shape"}:
            command.add_argument(
                "--seed", type=int, default=7319 if action == "qualify-shape" else 1234
            )
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
    for action in (
        "validate",
        "preflight",
        "controls",
        "probe",
        "smoke",
        "confirm",
        "final",
        "summarize",
        "closeout",
    ):
        command = o0_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument("--dataset", choices=("mnist",), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument(
            "--candidate",
            choices=("D0", "C0", "C1", "C2", "C3"),
            required=action in {"preflight", "probe", "smoke", "confirm", "final"},
        )
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
    for action in (
        "validate",
        "preflight",
        "audit-update",
        "probe",
        "summarize",
        "closeout",
    ):
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
    o0e = research_programs.add_parser("objective-o0")
    o0e_actions = o0e.add_subparsers(dest="research_action", required=True)
    for action in (
        "validate",
        "preflight",
        "calibrate",
        "screen",
        "confirm",
        "locked-final",
        "closeout",
    ):
        command = o0e_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--dataset", choices=("mnist",), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if action != "validate":
            command.add_argument("--evaluator-checkpoint", required=True)
            command.add_argument("--primary-qualification-evidence", required=True)
            command.add_argument("--secondary-evaluator-checkpoint", required=True)
            command.add_argument("--secondary-training-evidence", required=True)
            command.add_argument("--measurement-bundle", required=True)
            command.add_argument("--measurement-evidence", required=True)
        if action in {"preflight", "calibrate", "screen", "confirm", "locked-final"}:
            command.add_argument("--seed", type=int, default=1234)
        if action == "preflight":
            command.add_argument("--validate-evidence", required=True)
            command.add_argument("--o0-i-closeout")
        if action == "calibrate":
            command.add_argument("--preflight-evidence", required=True)
        if action == "screen":
            command.add_argument("--calibration-artifact", required=True)
            command.add_argument("--calibration-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "confirm":
            command.add_argument("--screen-evidence", required=True)
            command.add_argument("--calibration-artifact", required=True)
            command.add_argument("--calibration-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "locked-final":
            command.add_argument("--confirm-evidence", required=True)
            command.add_argument("--calibration-artifact", required=True)
            command.add_argument("--calibration-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "closeout":
            command.add_argument("--source-evidence", required=True)

    wae = research_programs.add_parser(
        "electronic-o0-wae",
        help="teacher-free WAE electronic O0 lifecycle",
    )
    wae_actions = wae.add_subparsers(dest="research_action", required=True)

    def wae_lineage(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--candidate", choices=("O0-I", "O0-G"), required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--o0-i-closeout")

    for action in (
        "validate",
        "preflight",
        "screen",
        "confirm",
        "locked-final",
        "closeout",
    ):
        command = wae_actions.add_parser(action)
        wae_lineage(command)
        if action in {"preflight", "screen", "confirm", "locked-final"}:
            command.add_argument("--seed", type=int, default=7301)
        if action == "preflight":
            command.add_argument("--validation-evidence", required=True)
        if action == "screen":
            command.add_argument("--validation-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "confirm":
            command.add_argument("--screen-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "locked-final":
            command.add_argument("--confirm-evidence", required=True)
            command.add_argument("--checkpoint", required=True)
        if action == "closeout":
            command.add_argument("--locked-evidence", required=True)

    transport = research_programs.add_parser(
        "electronic-o0-latent-transport",
        help="diagnostic class-conditional latent transport lifecycle",
    )
    transport_actions = transport.add_subparsers(dest="research_action", required=True)

    def transport_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--posterior-artifact")
        command.add_argument("--wae-training-checkpoint")
        command.add_argument("--wae-decoder-checkpoint")
        command.add_argument("--wae-confirm-evidence")
        command.add_argument("--latent-transport-confirm-evidence")
        command.add_argument("--mixture-closeout")
        command.add_argument("--primary-evaluator-checkpoint")
        command.add_argument("--secondary-evaluator-checkpoint")
        command.add_argument("--primary-qualification-evidence")
        command.add_argument("--secondary-qualification-evidence")

    for action in (
        "validate",
        "encode-posterior",
        "attribute",
        "preflight",
        "fit-flow",
        "screen",
        "measure-flow",
        "confirm",
        "locked-final",
        "closeout",
        "canary",
    ):
        command = transport_actions.add_parser(action)
        transport_lineage(command)
        if action in {"preflight", "screen", "confirm", "locked-final"}:
            command.add_argument("--seed", type=int, choices=(8401, 8402, 8403))
        if action == "attribute":
            command.add_argument("--validation-evidence")
        if action == "preflight":
            command.add_argument("--attribution-evidence")
        if action == "screen":
            command.add_argument("--preflight-evidence")
            command.add_argument("--fit-evidence")
            command.add_argument("--resume-checkpoint")
        if action == "fit-flow":
            command.add_argument("--preflight-evidence")
            command.add_argument("--resume-checkpoint")
        if action == "measure-flow":
            command.add_argument("--fit-evidence")
        if action == "confirm":
            command.add_argument("--screen-evidence")
            command.add_argument("--resume-checkpoint")
        if action == "locked-final":
            command.add_argument("--confirm-evidence")
            command.add_argument("--peer-confirm-evidence", action="append", default=[])
            command.add_argument("--flow-checkpoint")
        if action == "closeout":
            command.add_argument("--locked-evidence")
            command.add_argument("--measurement-evidence")

    mixture = research_programs.add_parser(
        "electronic-o0-mixture-transport",
        help="diagnostic regularized class-conditional Gaussian-mixture transport lifecycle",
    )
    mixture_actions = mixture.add_subparsers(dest="research_action", required=True)

    def mixture_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--posterior-artifact")
        command.add_argument("--wae-training-checkpoint")
        command.add_argument("--wae-decoder-checkpoint")
        command.add_argument("--wae-confirm-evidence")
        command.add_argument("--latent-transport-attribution")
        command.add_argument("--latent-transport-confirm-evidence")
        command.add_argument("--primary-evaluator-checkpoint")
        command.add_argument("--secondary-evaluator-checkpoint")
        command.add_argument("--primary-qualification-evidence")
        command.add_argument("--secondary-qualification-evidence")

    for action in (
        "validate",
        "encode-posterior",
        "preflight",
        "fit-mixture",
        "calibrate-temperature",
        "screen-measure",
        "screen",
        "confirm",
        "locked-final",
        "report",
        "closeout",
        "canary",
    ):
        command = mixture_actions.add_parser(action)
        mixture_lineage(command)
        if action in {"preflight", "screen", "confirm", "locked-final"}:
            command.add_argument("--seed", type=int, choices=(8501, 8502, 8503))
        if action == "preflight":
            command.add_argument("--validation-evidence")
        if action == "screen":
            command.add_argument("--preflight-evidence")
            command.add_argument("--calibration-evidence")
            command.add_argument("--resume-checkpoint")
        if action == "fit-mixture":
            command.add_argument("--preflight-evidence")
            command.add_argument("--fit-evidence")
        if action == "calibrate-temperature":
            command.add_argument("--fit-evidence")
        if action == "screen-measure":
            command.add_argument("--calibration-evidence")
        if action == "confirm":
            command.add_argument("--screen-evidence")
            command.add_argument("--resume-checkpoint")
        if action == "locked-final":
            command.add_argument("--confirm-evidence")
            command.add_argument("--peer-confirm-evidence", action="append", default=[])
            command.add_argument("--mixture-checkpoint")
        if action == "report":
            command.add_argument("--measurement-evidence")
        if action == "closeout":
            command.add_argument("--locked-evidence")
            command.add_argument("--screen-evidence")
            command.add_argument("--presentation-evidence")

    swae = research_programs.add_parser(
        "electronic-o0-prior-native-swae",
        help="diagnostic prior-native sliced-Wasserstein electronic O0 lifecycle",
    )
    swae_actions = swae.add_subparsers(dest="research_action", required=True)

    def swae_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--wae-training-checkpoint")
        command.add_argument("--wae-decoder-checkpoint")
        command.add_argument("--wae-confirm-evidence")
        command.add_argument("--anchor-confirm-evidence")
        command.add_argument("--primary-evaluator-checkpoint")
        command.add_argument("--secondary-evaluator-checkpoint")
        command.add_argument("--primary-qualification-evidence")
        command.add_argument("--secondary-qualification-evidence")

    for action in (
        "validate",
        "audit",
        "preflight",
        "screen",
        "confirm",
        "locked-final",
        "report",
        "closeout",
    ):
        command = swae_actions.add_parser(action)
        swae_lineage(command)
        if action in {"preflight", "screen", "confirm", "locked-final"}:
            command.add_argument("--seed", type=int, choices=(8601, 8602, 8603))
        if action == "preflight":
            command.add_argument("--audit-evidence")
        if action == "screen":
            command.add_argument("--preflight-evidence")
            command.add_argument("--resume-checkpoint")
        if action == "confirm":
            command.add_argument("--screen-evidence")
            command.add_argument("--resume-checkpoint")
        if action == "locked-final":
            command.add_argument("--confirm-evidence")
            command.add_argument("--peer-confirm-evidence", action="append", default=[])
            command.add_argument("--checkpoint")
        if action == "report":
            command.add_argument("--measurement-evidence")
        if action == "closeout":
            command.add_argument("--measurement-evidence")

    sced = research_programs.add_parser(
        "electronic-o0-prior-native-sced",
        help="diagnostic prior-native sliced Cramer/energy electronic O0 lifecycle",
    )
    sced_actions = sced.add_subparsers(dest="research_action", required=True)

    def sced_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--wae-training-checkpoint")
        command.add_argument("--wae-decoder-checkpoint")
        command.add_argument("--wae-confirm-evidence")
        command.add_argument("--latent-transport-confirm-evidence")
        command.add_argument("--mixture-closeout")
        command.add_argument("--h-swae-closeout")
        command.add_argument("--anchor-confirm-evidence")
        command.add_argument("--primary-evaluator-checkpoint")
        command.add_argument("--secondary-evaluator-checkpoint")
        command.add_argument("--primary-qualification-evidence")
        command.add_argument("--secondary-qualification-evidence")

    for action in (
        "validate",
        "audit",
        "preflight",
        "screen",
        "confirm",
        "locked-final",
        "report",
        "closeout",
    ):
        command = sced_actions.add_parser(action)
        sced_lineage(command)
        if action in {"preflight", "screen", "confirm", "locked-final"}:
            command.add_argument("--seed", type=int, choices=(8601, 8602, 8603))
        if action == "audit":
            command.add_argument("--validation-evidence")
        if action == "preflight":
            command.add_argument("--audit-evidence")
        if action == "screen":
            command.add_argument("--preflight-evidence")
            command.add_argument("--resume-checkpoint")
        if action == "confirm":
            command.add_argument("--screen-evidence")
            command.add_argument("--resume-checkpoint")
        if action == "locked-final":
            command.add_argument("--confirm-evidence")
            command.add_argument("--peer-confirm-evidence", action="append", default=[])
            command.add_argument("--checkpoint")
        if action == "report":
            command.add_argument("--measurement-evidence")
        if action == "closeout":
            command.add_argument("--measurement-evidence")

    pgc = research_programs.add_parser(
        "electronic-o0-prior-gradient-conflict",
        help="read-only prior-gradient conflict attribution for electronic O0",
    )
    pgc_actions = pgc.add_subparsers(dest="research_action", required=True)

    def pgc_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--wae-training-checkpoint")
        command.add_argument("--wae-decoder-checkpoint")
        command.add_argument("--swae-training-checkpoint")
        command.add_argument("--swae-screen-evidence")
        command.add_argument("--swae-closeout-evidence")
        command.add_argument("--sced-training-checkpoint")
        command.add_argument("--sced-screen-evidence")
        command.add_argument("--sced-closeout-evidence")

    pgc_validate = pgc_actions.add_parser("validate")
    pgc_lineage(pgc_validate)
    pgc_audit = pgc_actions.add_parser("audit")
    pgc_lineage(pgc_audit)
    pgc_audit.add_argument("--validation-evidence")
    pgc_report = pgc_actions.add_parser("report")
    pgc_report.add_argument("--run-manifest")
    pgc_report.add_argument("--config")
    pgc_report.add_argument("--dataset-root")
    pgc_report.add_argument("--output")
    pgc_report.add_argument("--measurement-evidence")
    pgc_closeout = pgc_actions.add_parser("closeout")
    pgc_closeout.add_argument("--run-manifest")
    pgc_closeout.add_argument("--config")
    pgc_closeout.add_argument("--dataset-root")
    pgc_closeout.add_argument("--output")
    pgc_closeout.add_argument("--measurement-evidence")

    tca = research_programs.add_parser(
        "electronic-o0-prior-trajectory-curvature",
        help="deterministic paired prior trajectory and curvature attribution for electronic O0",
    )
    tca_actions = tca.add_subparsers(dest="research_action", required=True)

    def tca_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--wae-training-checkpoint")
        command.add_argument("--wae-decoder-checkpoint")
        command.add_argument("--swae-training-checkpoint")
        command.add_argument("--swae-screen-evidence")
        command.add_argument("--swae-closeout-evidence")
        command.add_argument("--sced-training-checkpoint")
        command.add_argument("--sced-screen-evidence")
        command.add_argument("--sced-closeout-evidence")
        command.add_argument("--h-pgc-audit-evidence")
        command.add_argument("--h-pgc-report-evidence")
        command.add_argument("--h-pgc-closeout-evidence")

    for action in ("validate", "preflight", "audit"):
        command = tca_actions.add_parser(action)
        tca_lineage(command)
        if action == "preflight":
            command.add_argument("--validation-evidence")
        if action == "audit":
            command.add_argument("--preflight-evidence")
    for action in ("report", "closeout"):
        command = tca_actions.add_parser(action)
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--output")
        command.add_argument("--measurement-evidence")

    dlta = research_programs.add_parser(
        "electronic-o0-prior-dense-local-attribution",
        help="dense local prior trajectory attribution for electronic O0",
    )
    dlta_actions = dlta.add_subparsers(dest="research_action", required=True)

    def dlta_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--wae-training-checkpoint")
        command.add_argument("--wae-decoder-checkpoint")
        command.add_argument("--swae-training-checkpoint")
        command.add_argument("--swae-screen-evidence")
        command.add_argument("--swae-closeout-evidence")
        command.add_argument("--sced-training-checkpoint")
        command.add_argument("--sced-screen-evidence")
        command.add_argument("--sced-closeout-evidence")
        command.add_argument("--h-pgc-audit-evidence")
        command.add_argument("--h-pgc-report-evidence")
        command.add_argument("--h-pgc-closeout-evidence")
        command.add_argument("--h-tca-audit-evidence")
        command.add_argument("--h-tca-report-evidence")
        command.add_argument("--h-tca-closeout-evidence")
        command.add_argument("--resume-checkpoint")

    for action in ("validate", "preflight", "audit"):
        command = dlta_actions.add_parser(action)
        dlta_lineage(command)
        if action == "preflight":
            command.add_argument("--validation-evidence")
        if action == "audit":
            command.add_argument("--preflight-evidence")
    for action in ("report", "closeout"):
        command = dlta_actions.add_parser(action)
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--output")
        command.add_argument("--measurement-evidence")

    ccf = research_programs.add_parser(
        "electronic-o0-prior-cycle-composition",
        help="factorial prior-cycle composition attribution for electronic O0",
    )
    ccf_actions = ccf.add_subparsers(dest="research_action", required=True)

    def ccf_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--wae-training-checkpoint")
        command.add_argument("--wae-decoder-checkpoint")
        command.add_argument("--swae-training-checkpoint")
        command.add_argument("--swae-screen-evidence")
        command.add_argument("--swae-closeout-evidence")
        command.add_argument("--sced-training-checkpoint")
        command.add_argument("--sced-screen-evidence")
        command.add_argument("--sced-closeout-evidence")
        command.add_argument("--h-dlta-audit-evidence")
        command.add_argument("--h-dlta-report-evidence")
        command.add_argument("--h-dlta-closeout-evidence")

    for action in ("validate", "preflight", "audit"):
        command = ccf_actions.add_parser(action)
        ccf_lineage(command)
        if action == "preflight":
            command.add_argument("--validation-evidence")
        if action == "audit":
            command.add_argument("--preflight-evidence")
    for action in ("report", "closeout"):
        command = ccf_actions.add_parser(action)
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--output")
        command.add_argument("--measurement-evidence")

    dpgc = research_programs.add_parser(
        "electronic-o0-direct-prior-capacity-canary",
        help="no-update direct-prior capacity canary for electronic O0",
    )
    dpgc_actions = dpgc.add_subparsers(dest="research_action", required=True)

    def dpgc_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--primary-evaluator-checkpoint")
        command.add_argument("--secondary-evaluator-checkpoint")
        command.add_argument("--primary-qualification-evidence")
        command.add_argument("--secondary-qualification-evidence")
        command.add_argument("--h-ccf-closeout-evidence")

    for action in ("validate", "preflight", "audit"):
        command = dpgc_actions.add_parser(action)
        dpgc_lineage(command)
        if action == "preflight":
            command.add_argument("--validation-evidence")
        if action == "audit":
            command.add_argument("--preflight-evidence")
    for action in ("report", "closeout"):
        command = dpgc_actions.add_parser(action)
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--output")
        command.add_argument("--measurement-evidence")

    didm = research_programs.add_parser(
        "electronic-o0-direct-image-distribution",
        help="direct classwise multiscale image-distribution matching for electronic O0",
    )
    didm_actions = didm.add_subparsers(dest="research_action", required=True)

    def didm_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--wae-decoder-checkpoint")
        command.add_argument("--h-dpgc-closeout-evidence")
        command.add_argument("--primary-evaluator-checkpoint")
        command.add_argument("--secondary-evaluator-checkpoint")
        command.add_argument("--primary-qualification-evidence")
        command.add_argument("--secondary-qualification-evidence")

    for action in ("validate", "audit", "preflight", "screen"):
        command = didm_actions.add_parser(action)
        didm_lineage(command)
        if action == "audit":
            command.add_argument("--validation-evidence")
        elif action == "preflight":
            command.add_argument("--audit-evidence")
        elif action == "screen":
            command.add_argument("--preflight-evidence")
            command.add_argument("--resume-checkpoint")
    for action in ("report", "closeout"):
        command = didm_actions.add_parser(action)
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--output")
        command.add_argument("--measurement-evidence")

    semantic_factorial = research_programs.add_parser(
        "electronic-o0-semantic-code-factorial",
        help="no-update decoder/code-source semantic factorial for electronic O0",
    )
    semantic_factorial_actions = semantic_factorial.add_subparsers(
        dest="research_action", required=True
    )

    def semantic_factorial_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--wae-training-checkpoint")
        command.add_argument("--wae-decoder-checkpoint")
        command.add_argument("--wae-confirm-evidence")
        command.add_argument("--latent-transport-attribution")
        command.add_argument("--h-scf-final-decoder-checkpoint")
        command.add_argument("--h-didm-screen-evidence")
        command.add_argument("--h-didm-report-evidence")
        command.add_argument("--h-didm-closeout-evidence")
        command.add_argument("--primary-evaluator-checkpoint")
        command.add_argument("--secondary-evaluator-checkpoint")
        command.add_argument("--primary-qualification-evidence")
        command.add_argument("--secondary-qualification-evidence")

    for action in ("validate", "preflight", "audit"):
        command = semantic_factorial_actions.add_parser(action)
        semantic_factorial_lineage(command)
        if action == "preflight":
            command.add_argument("--validation-evidence")
        elif action == "audit":
            command.add_argument("--preflight-evidence")
    for action in ("report", "closeout"):
        command = semantic_factorial_actions.add_parser(action)
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--output")
        command.add_argument("--measurement-evidence")

    manifold = research_programs.add_parser(
        "electronic-o0-nonlinear-direct-manifold-capacity",
        help="no-update nonlinear direct manifold capacity lifecycle for electronic O0",
    )
    manifold_actions = manifold.add_subparsers(dest="research_action", required=True)

    def manifold_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--h-dpgc-closeout-evidence")
        command.add_argument("--h-scf-closeout-evidence")
        command.add_argument("--primary-evaluator-checkpoint")
        command.add_argument("--secondary-evaluator-checkpoint")
        command.add_argument("--primary-qualification-evidence")
        command.add_argument("--secondary-qualification-evidence")

    for action in ("validate", "preflight", "audit"):
        command = manifold_actions.add_parser(action)
        manifold_lineage(command)
        if action == "preflight":
            command.add_argument("--validation-evidence")
        elif action == "audit":
            command.add_argument("--preflight-evidence")
    for action in ("report", "closeout"):
        command = manifold_actions.add_parser(action)
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--output")
        command.add_argument("--measurement-evidence")

    jqoc = research_programs.add_parser(
        "joint-quadrature-optical-capacity",
        help="no-update electronic and single-exposure quadrature capacity lifecycle",
    )
    jqoc_actions = jqoc.add_subparsers(dest="research_action", required=True)

    def jqoc_lineage(command):
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--device", choices=("cuda", "cpu"))
        command.add_argument("--output")
        command.add_argument("--v69-checkpoint")
        command.add_argument("--v75-report")
        command.add_argument("--v75-decision")
        command.add_argument("--v76-admission")
        command.add_argument("--v76-report")
        command.add_argument("--v76-decision")
        command.add_argument("--h-ndmc-closeout-evidence")
        command.add_argument("--primary-evaluator-checkpoint")
        command.add_argument("--secondary-evaluator-checkpoint")
        command.add_argument("--primary-qualification-evidence")
        command.add_argument("--secondary-qualification-evidence")

    for action in ("validate", "preflight", "audit"):
        command = jqoc_actions.add_parser(action)
        jqoc_lineage(command)
        if action == "preflight":
            command.add_argument("--validation-evidence")
        elif action == "audit":
            command.add_argument("--preflight-evidence")
    for action in ("report", "closeout"):
        command = jqoc_actions.add_parser(action)
        command.add_argument("--run-manifest")
        command.add_argument("--config")
        command.add_argument("--dataset-root")
        command.add_argument("--output")
        command.add_argument("--measurement-evidence")

    recovery = research_programs.add_parser(
        "electronic-distribution-recovery",
        help="fail-closed v3 electronic range diagnosis",
    )
    recovery_actions = recovery.add_subparsers(dest="research_action", required=True)

    def recovery_common(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--generator-checkpoint", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--output", required=True)

    validate = recovery_actions.add_parser("validate")
    recovery_common(validate)
    gate_audit = recovery_actions.add_parser("gate-audit")
    recovery_common(gate_audit)
    gate_audit.add_argument("--validation-evidence", required=True)
    preflight = recovery_actions.add_parser("preflight")
    recovery_common(preflight)
    preflight.add_argument("--validation-evidence", required=True)
    preflight.add_argument("--gate-audit-evidence", required=True)
    audit = recovery_actions.add_parser("audit")
    recovery_common(audit)
    audit.add_argument("--gate-audit-evidence", required=True)
    audit.add_argument("--preflight-evidence", required=True)
    report = recovery_actions.add_parser("report")
    report.add_argument("--config", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--audit-evidence", required=True)
    closeout = recovery_actions.add_parser("closeout")
    closeout.add_argument("--config", required=True)
    closeout.add_argument("--output", required=True)
    closeout.add_argument("--report-evidence", required=True)

    teacher = research_programs.add_parser(
        "teacher-distribution-admission",
        help="fail-closed admission of frozen v1 DDPM-VAE teacher banks",
    )
    teacher_actions = teacher.add_subparsers(dest="research_action", required=True)

    def teacher_common(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--teacher-bank-manifest", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--package-a-gate-evidence", required=True)
        command.add_argument("--package-a-closeout-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--output", required=True)

    for action in ("validate", "audit"):
        command = teacher_actions.add_parser(action)
        teacher_common(command)
        if action == "audit":
            command.add_argument("--validation-evidence", required=True)
    report = teacher_actions.add_parser("report")
    teacher_common(report)
    report.add_argument("--audit-evidence", required=True)
    closeout = teacher_actions.add_parser("closeout")
    teacher_common(closeout)
    closeout.add_argument("--report-evidence", required=True)

    attribution = research_programs.add_parser(
        "teacher-failure-attribution",
        help="no-update attribution of the locked Package B teacher failure",
    )
    attribution_actions = attribution.add_subparsers(
        dest="research_action", required=True
    )

    def attribution_common(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--attribution-artifact", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--output", required=True)

    for action in ("validate", "audit"):
        command = attribution_actions.add_parser(action)
        attribution_common(command)
        if action == "audit":
            command.add_argument("--validation-evidence", required=True)
    report = attribution_actions.add_parser("report")
    attribution_common(report)
    report.add_argument("--audit-evidence", required=True)
    closeout = attribution_actions.add_parser("closeout")
    attribution_common(closeout)
    closeout.add_argument("--report-evidence", required=True)

    first_stage = research_programs.add_parser(
        "teacher-first-stage-recovery",
        help="Package B3 software first-stage reconstruction admission",
    )
    first_stage_actions = first_stage.add_subparsers(
        dest="research_action", required=True
    )

    def first_stage_common(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--reconstruction-manifest", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--output", required=True)

    validate = first_stage_actions.add_parser("validate")
    first_stage_common(validate)
    audit = first_stage_actions.add_parser("audit")
    first_stage_common(audit)
    audit.add_argument("--validation-evidence", required=True)
    report = first_stage_actions.add_parser("report")
    first_stage_common(report)
    report.add_argument("--audit-evidence", required=True)
    closeout = first_stage_actions.add_parser("closeout")
    first_stage_common(closeout)
    closeout.add_argument("--report-evidence", required=True)

    smf = research_programs.add_parser(
        "smf-modal-density-attribution",
        help="synthetic single-mode port-density upper-bound lifecycle",
    )
    smf_actions = smf.add_subparsers(dest="research_action", required=True)

    def smf_lineage(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--output", required=True)
        command.add_argument("--o0-g-decoder", required=True)
        command.add_argument("--o0-i-closeout", required=True)
        command.add_argument("--o0-g-closeout", required=True)
        command.add_argument("--fiber-closeout", required=True)
        command.add_argument("--lp01-fixture", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)

    smf_validate = smf_actions.add_parser("validate")
    smf_lineage(smf_validate)
    smf_validate.add_argument("--formal-config", required=True)
    smf_run = smf_actions.add_parser("run")
    smf_lineage(smf_run)
    smf_run.add_argument("--validation-evidence", required=True)
    for action in ("report", "closeout"):
        command = smf_actions.add_parser(action)
        command.add_argument("--measurement-evidence", required=True)
        if action == "closeout":
            command.add_argument("--presentation-evidence", required=True)
        command.add_argument("--output", required=True)
