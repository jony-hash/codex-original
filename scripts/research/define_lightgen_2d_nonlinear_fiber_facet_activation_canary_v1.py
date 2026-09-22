"""Materialize the frozen N6 nonlinear fiber-facet canary contract.

This module only writes the definition artifact.  It never loads scientific
data, constructs a CUDA runtime, or performs a forward/update.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ldm_project.v2.evidence import atomic_write_new_json


TASK_ID = "DEFINE_N6_NONLINEAR_FIBER_FACET_ACTIVATION_CANARY_V1"
OUTER_GATE = "N6_NONLINEAR_FIBER_FACET_ACTIVATION_SUPPORTED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

DEFAULT_PATH = Path(
    "outputs/20260914_n6/runs/"
    "lightgen_2d_nonlinear_fiber_facet_activation_canary_contract_v1_attempt_02/"
    "lightgen_2d_nonlinear_fiber_facet_activation_canary_contract.v1.json"
)


def _contract() -> dict[str, Any]:
    return {
        "schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_contract.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_BOUNDED_PREFLIGHT",
        "outer_gate": OUTER_GATE,
        "git_boundary": {
            "branch": EXPECTED_BRANCH,
            "head": EXPECTED_HEAD,
            "identity_rule": "branch plus HEAD is source-code identity",
        },
        "closure": {
            "n5_v2_terminal": True,
            "statement": "N5-V2 is the terminal experiment for the current linear true-optical-depth branch; no additional linear optical-depth candidate is authorized.",
            "n5_v2_classification": "OPTIMIZATION_HORIZON_DOMINANT_TRUE_DEPTH_NOT_SUPPORTED",
            "n5_v2_a_smooth_r300": 0.5526,
            "n5_v2_b_smooth_r300": 0.5504,
            "n5_v2_delta_depth": -0.0022,
            "n5_v2_delta_depth_ci": [-0.0042, -0.0002],
        },
        "authorities": {
            "n5_v2_contract": "DEFINE_N5_SUCCESSOR_TRUE_OPTICAL_DEPTH_COMPARISON_V2_CONTRACT",
            "n5_a_endpoint": {
                "path": "outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_candidate_a_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_a_continued_c1_step_300_checkpoint.v1.pt",
                "checkpoint_step": 300,
                "global_step": 600,
                "sha256": "5ac782bde83e7f33c34df872c2b907005f0b27200ff8c3b06386019a7294be6e",
                "role": "exact current best linear endpoint and N6 common starting state",
            },
            "v11_objective": "V11 frozen semantic-plus-image objective",
            "rank": 9,
            "encoder": "V3 CONTROL frozen",
            "generator": "existing LightGen-aligned two-layer diffraction generator frozen in depth and trainable in phase",
            "ols": "100-port complex OLS with coefficient-space T=I_100 and shared W_OLS",
        },
        "model_assumption": {
            "statement": "A fixed nonlinear material response at the physical fiber-output facet may provide semantic capacity beyond equal continuation of the current best linear architecture.",
            "physical_validation": False,
        },
        "material": {
            "kind": "ALGORITHMIC_NONLINEAR_FACET_CANARY",
            "measured_transfer_curve": False,
            "phase_measurement_available": False,
            "phase_assumption": "Delta_phi_material(I)=0 exactly; this is an explicit model assumption, not a measurement.",
            "placement": "physical fiber-output spatial field, immediately after W_OLS/modal re-emission and immediately before the existing generator; no new diffraction plane",
            "input": "E_in, the ComplexField generator_input_field produced by the canonical W_OLS writer",
            "fixed_transfer": {
                "intensity": "I_out = max(I_in - 0.05, 0.0)",
                "threshold": 0.05,
                "slope": 1.0,
                "gain": 1.0,
                "epsilon": 1.0e-12,
                "field": "E_out = sqrt((I_out+epsilon)/(I_in+epsilon))*E_in; epsilon is the fixed differentiability stabilizer",
            },
            "trainable": False,
            "tuning": False,
            "sweep": False,
            "resampling": False,
            "crop_or_pad": False,
            "coefficient_space_direct_nonlinearity": False,
        },
        "candidates": {
            "A_LINEAR_CONTINUATION_CONTROL": {
                "starting_state": "exact N5-A step300 checkpoint",
                "facet": "identity; no nonlinear material",
                "trainable": ["adapter.u_real", "adapter.u_imag", "adapter.v_real", "adapter.v_imag", "generator.phase"],
                "frozen": ["encoder.phase", "OLS", "T=I_100", "W_OLS", "material"],
            },
            "B_NONLINEAR_FACET": {
                "starting_state": "bitwise-equivalent A starting state",
                "facet": "fixed N6 material transfer at the physical fiber-output field",
                "trainable": ["adapter.u_real", "adapter.u_imag", "adapter.v_real", "adapter.v_imag", "generator.phase"],
                "frozen": ["encoder.phase", "OLS", "T=I_100", "W_OLS", "material"],
            },
        },
        "common_protocol": {
            "objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image",
            "lambda_transfer": 28.48464149920509,
            "lambda_image": 13.353282014834436,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "batch_size": 10,
            "additional_updates": 300,
            "global_offset": 600,
            "global_terminal_step": 900,
            "evaluation_steps": [0, 100, 200, 300],
            "same_checkpoint": True,
            "same_train_order": True,
            "same_rng_restore": True,
            "same_optimizer_state": True,
            "same_evaluator_and_bootstrap": True,
        },
        "preflight": {
            "scientific_updates": 0,
            "required": [
                "resolve exact N5-A endpoint and all upstream authorities",
                "prove A/B runtime and adapter starting-state equality",
                "prove material receives the canonical physical fiber-output field",
                "prove no crop/pad/resample or new diffraction plane is introduced",
                "prove finite complex output and finite power/activation diagnostics",
                "prove finite nonzero gradients through the material to adapter and generator phase",
                "prove encoder/OLS/material freeze firewall",
                "run disposable three-update CUDA smoke",
                "checkpoint/reload/resume without consuming a post-budget scientific batch",
                "run exact-core synthetic persistence chain before diagnostics",
            ],
        },
        "statistics": {
            "primary": {
                "delta": "R_B300-R_A300",
                "supported_if": [
                    "delta >= +0.05",
                    "paired bootstrap CI lower > 0",
                    "B DEV image loss step300 minus B step0 <= +0.005",
                    "cyclic template transfer retained",
                    "cyclic Smooth transfer retained",
                    "raw validity passes",
                ],
            },
            "descriptive": ["R_A300-0.5526", "R_B300-0.5526"],
            "facet": [
                "input/output intensity distributions",
                "transition-region fraction",
                "suppressed fraction",
                "power in/out and ratio",
                "local transfer slope distribution",
                "per-class activation",
                "adapter residual and factor singular values",
                "generator phase change",
            ],
        },
        "stage_a": {
            "firewall": "R300>=0.937",
            "independent_of_n6_gate": True,
            "stage_b_authorized_by_n6": False,
        },
        "data_boundary": {
            "TRAIN": True,
            "DEV": True,
            "protected": False,
            "official_test": False,
            "sealed": False,
            "prior_generation": False,
        },
        "one_shot": {
            "preflight_attempts_may_be_repaired": True,
            "candidate_A_attempts": 1,
            "candidate_B_attempts": 1,
            "extra_candidate": False,
            "retry": False,
            "adaptive_change": False,
        },
        "hard_stops": [
            "required authority unresolved",
            "material transfer undefined or placement inconsistent",
            "nonfinite output or required statistic undefined",
            "all upstream gradients structurally zero",
            "trainable/frozen firewall failure",
            "CUDA/checkpoint/persistence failure",
            "data leakage",
        ],
        "claim_boundary": {
            "new_optical_depth": False,
            "rank": 9,
            "stage_b": False,
            "generation": False,
            "protected": False,
            "official_test": False,
            "sealed": False,
            "formal_promotion": False,
            "physical_realizability_claim": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    path = (args.output or root / DEFAULT_PATH).resolve()
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=False)
    atomic_write_new_json(path, _contract(), artifact_kind="N6 nonlinear fiber-facet canary contract")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
