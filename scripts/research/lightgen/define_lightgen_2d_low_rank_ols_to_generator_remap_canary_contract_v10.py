"""Freeze the one-shot V10 low-rank OLS-to-generator remap canary contract.

V10 consumes the validated V9 authority chain and inserts one bounded,
rank-nine coordinate remap between the frozen encoded OLS coefficients and the
unchanged W_OLS/frozen-generator path.  The contract is deliberately
write-once: the adapter rank, factor initialization, objective, optimizer,
data roles, metrics, and gates are fixed before any execution.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training import file_sha256
from ldm_project.v2.evidence import atomic_write_new_json


TASK_ID = "DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10"
CONTRACT_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10"
CLASSIFICATION = "DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10_READY"
OUTER_GATE = "V10_LOW_RANK_OLS_TO_GENERATOR_REMAP_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

V9_CONTRACT_PATH = Path(
    "outputs/20260912_v9/runs/"
    "lightgen_2d_joint_semantic_remapping_warmup_contract_v9_attempt_01/"
    "lightgen_2d_joint_semantic_remapping_warmup_contract.v9.json"
)
V9_CONTRACT_SHA256 = "4d0fb81bfa6152995aad4484c83487c20335616283097b394757f4616bf41fca"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260912_v10/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10.json"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
ADAPTER_RANK = 9
INIT_SCALE = 1.0e-2
S_TAU = 0.02919053753794475
LAMBDA_TRANSFER = 28.48464149920509


class ContractError(RuntimeError):
    """A frozen V10 authority cannot be resolved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    _require(not value.is_absolute(), f"{relative} must be relative")
    return (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _load_v9_parent(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _path(root, V9_CONTRACT_PATH)
    _require(path.is_file(), "V9 contract authority is unavailable")
    _require(file_sha256(path) == V9_CONTRACT_SHA256, "V9 contract authority SHA changed")
    parent = _read_json(path, "V9 contract")
    _require(
        parent.get("schema") == "lightgen_2d_joint_semantic_remapping_warmup_contract.v9"
        and parent.get("result") == "READY"
        and parent.get("status") == "valid_completed"
        and parent.get("outer_gate") == "V9_JOINT_SEMANTIC_REMAPPING_SUPPORTED",
        "V9 parent contract is not READY",
    )
    git = parent.get("git")
    _require(isinstance(git, Mapping), "V9 parent Git identity is missing")
    _require(git.get("branch") == EXPECTED_BRANCH and git.get("head") == EXPECTED_HEAD, "V9 parent Git identity changed")
    definition = parent.get("definition_entrypoint")
    _require(isinstance(definition, Mapping), "V9 definition entrypoint is missing")
    definition_path = _path(root, str(definition.get("path")))
    _require(file_sha256(definition_path) == definition.get("sha256"), "V9 definition entrypoint SHA changed")
    objective = parent.get("objective")
    _require(isinstance(objective, Mapping), "V9 objective is missing")
    _require(float(objective.get("s_tau", 0.0)) == S_TAU, "V9 s_tau authority changed")
    _require(float(objective.get("lambda_transfer", 0.0)) == LAMBDA_TRANSFER, "V9 lambda authority changed")
    return (
        {
            "path": V9_CONTRACT_PATH.as_posix(),
            "sha256": V9_CONTRACT_SHA256,
            "semantic_authority": "DEFINE_LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_CONTRACT_V9",
        },
        parent,
    )


def build_contract(root: Path) -> dict[str, Any]:
    parent_ref, parent = _load_v9_parent(root)
    authorities = copy.deepcopy(parent["authorities"])
    authorities["v9_joint_remapping"] = {
        "role": "validated V9 step-300 scientific arm; parent bounded authority only"
    }
    authorities["v10_low_rank_adapter"] = {
        "role": "new canary adapter; no historical checkpoint authority",
        "matrix": "T_adapt=I_100+U V^H",
        "rank": ADAPTER_RANK,
        "factor_shapes": {"U": [OLS_PORT_COUNT, ADAPTER_RANK], "V": [OLS_PORT_COUNT, ADAPTER_RANK]},
        "logical_dtype": "torch.complex64",
    }

    definition_path = Path(__file__).resolve()
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "result": "READY",
        "status": "valid_completed",
        "classification": CLASSIFICATION,
        "outer_gate": OUTER_GATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"branch": EXPECTED_BRANCH, "head": EXPECTED_HEAD},
        "definition_entrypoint": {"path": _relative(root, definition_path), "sha256": file_sha256(definition_path)},
        "parent_contract": parent_ref,
        "authorities": authorities,
        "model_assumption": (
            "The V9 limitation may include insufficient coordinate/mapping flexibility between "
            "encoded OLS coefficients and the frozen generator; a bounded low-rank remap may "
            "improve semantic reconstruction without changing optical architecture."
        ),
        "objective": {
            "question": (
                "Can a fixed-rank remap between frozen real-path encoder coefficients and the "
                "unchanged W_OLS/frozen-generator path improve semantic reconstruction?"
            ),
            "training_objective": "L_V10=L_anchor+lambda_transfer*L_transfer",
            "anchor": {
                "logits": "b_k(raw_base)=-d_image(raw_base,tau_k)/(s_tau+1e-12)",
                "loss": "cross_entropy([b_0,...,b_9], y)",
                "output": "raw_base=G(W_OLS(T_adapt z))",
                "representation": "frozen historical TRAIN templates tau_0...tau_9",
            },
            "transfer": {
                "loss": "softplus(-Delta_template)",
                "delta_template": "M_template(raw_shift)-M_template(raw_base)",
                "margin": "M_template(r)=d_image(r,tau_y)-d_image(r,tau_t)",
                "latent_shift": "z_shift=z-mu_y+mu_t; z_shift_adapt=T_adapt z_shift",
                "target_rule": "t=(y+1) mod 10",
            },
            "image_distance": "mean squared pixel distance after clamp [0,1] and adaptive_avg_pool2d to 28x28",
            "s_tau": S_TAU,
            "lambda_transfer": LAMBDA_TRANSFER,
            "lambda_rule": {
                "formula": "reuse exact frozen V9/V8 lambda_transfer",
                "source": "V9 parent authority; no V10 recalibration",
                "recompute": False,
                "clipping_or_tuning": False,
            },
            "gradient_contributions": {
                "image_reconstruction": "ZERO; evaluation-only",
                "prototype_cross_entropy": "ZERO; prototypes are frozen shift support only",
                "Smooth_evaluator": "ZERO; evaluation-only",
            },
        },
        "adapter": {
            "path": "z -> T_adapt -> fixed W_OLS -> frozen generator",
            "matrix_formula": "T_adapt=I_100+U V^H",
            "rank": ADAPTER_RANK,
            "application_convention": "row-vector runtime application is z_adapt=z @ T_adapt.T; T_adapt is the column-vector contract matrix",
            "parameterization": "U=u_real+i*u_imag; V=v_real+i*v_imag; four real trainable Parameter tensors",
            "parameter_names": ["adapter.u_real", "adapter.u_imag", "adapter.v_real", "adapter.v_imag"],
            "factor_shapes": {"U": [OLS_PORT_COUNT, ADAPTER_RANK], "V": [OLS_PORT_COUNT, ADAPTER_RANK]},
            "initialization": {
                "scale": INIT_SCALE,
                "u_real": "INIT_SCALE*I_9 in rows 0..8, zero elsewhere",
                "v_real": "INIT_SCALE*I_9 in rows 9..17, zero elsewhere",
                "u_imag": "zero",
                "v_imag": "zero",
                "support_rows_U": list(range(0, ADAPTER_RANK)),
                "support_rows_V": list(range(ADAPTER_RANK, 2 * ADAPTER_RANK)),
                "residual_at_step0": "exact zero in float32 construction; T_adapt=I_100",
            },
            "identity_tolerances": {
                "residual_max_abs": 0.0,
                "forward_raw_allclose_atol": 1.0e-6,
                "forward_raw_allclose_rtol": 1.0e-5,
            },
            "diagnostics": [
                "||T_adapt-I||_F",
                "spectral norm of residual via complex128 svdvals",
                "torch.linalg.matrix_rank(residual) with default deterministic tolerance",
                "all residual singular values",
                "per-class DEV centroid displacement ||mean(z_adapt)-mean(z)||_2",
                "all 45 pairwise DEV centroid-distance distortions",
            ],
        },
        "design_decisions": [
            "Use exact V3 CONTROL step-300 encoder and exact V4 valid step-300 generator as frozen initial authorities.",
            "Insert exactly one rank-nine U/V remap; do not change OLS ports, W_OLS, optical layers, or generator layers.",
            "Use orthogonal-support U/V initialization so both factors are nonzero while U V^H is exactly zero at step 0.",
            "Reuse exact V9/V8 s_tau and lambda_transfer; never recalibrate either constant.",
            "Train only adapter.u_real, adapter.u_imag, adapter.v_real, and adapter.v_imag with Adam lr=1e-3 for 300 updates.",
            "Use step-0 identity adapter as the matched baseline; no second arm, rank sweep, hyperparameter sweep, retry, or budget extension.",
            "Keep the independent Smooth gate and the 0.937 Stage-A firewall unchanged.",
        ],
        "protocol": {
            "trainable": ["adapter.u_real", "adapter.u_imag", "adapter.v_real", "adapter.v_imag"],
            "frozen": [
                "encoder.phase",
                "generator.phase",
                "OLS definition",
                "T=I_100 coefficient-space transport",
                "shared W_OLS",
                "mandatory modal projection/re-emission",
                "TRAIN templates",
                "TRAIN prototypes",
                "SmoothMNISTEvaluator",
                "all remaining runtime parameters",
            ],
            "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False},
            "logical_batch_size": 10,
            "updates": 300,
            "checkpoints": [0, 100, 200, 300],
            "propagation_grid": 2048,
            "rng_convention": {"initialization_seed": 1234, "schedule_seed": 1234},
            "cyclic_transfer": {"target_rule": "t=(y+1) mod 10", "selection_rows": 1000, "selection_per_class": 100, "selection_seed": 4321, "no_target_search": True, "no_magnitude_sweep": True},
            "bootstrap": {"replicates": 999, "seed": 271828, "type": "paired class-stratified"},
        },
        "metrics": {
            "primary_natural_semantic": {"name": "Delta_R_Smooth", "formula": "R300-R0 using frozen SmoothMNISTEvaluator v2.1", "pass": "Delta >= +0.05 and paired bootstrap 95% CI lower > 0"},
            "primary_transfer": {"template": "Delta_template=M_shift-M_base", "smooth": "Delta_smooth=(score_t-score_y)_shift-(score_t-score_y)_base", "known_transfer_pass": "template pass and Smooth pass and broad positive pair coverage"},
            "descriptive": [
                "TRAIN anchor/transfer/total semantic losses at 0/100/200/300",
                "TRAIN/DEV image loss",
                "Smooth macro/per-class recall",
                "template-anchor macro/per-class recall and correct-class margin",
                "OLS prototype margin",
                "step-0/step-300 raw collapse/diversity diagnostics",
                "all ten cyclic-pair summaries",
                "adapter Frobenius/spectral/rank/singular-value/centroid diagnostics",
                "adapter gradient and update diagnostics",
            ],
            "raw_validity": "finite, expected shape, structured, nonconstant at all checkpoints and cyclic base/shift",
            "stage_a_firewall": "R300 >= 0.937, independently reported and does not authorize Stage B",
        },
        "data_roles": {"TRAIN": {"rows": 55000, "used_for": "adapter optimization"}, "DEV": {"rows": 5000, "used_for": "frozen evaluation and balanced cyclic selection"}, "calibration": 0, "protected": 0, "official_test": 0, "sealed": 0, "labels_as_model_input": False},
        "engineering_preflight": {
            "required": [
                "resolve exact V3 encoder, V4 generator, V5 prototypes, historical templates, Smooth, V2 runtime, and V9 parent authorities",
                "verify rank-nine adapter materialization and exact step-0 T_adapt=I_100 identity",
                "verify only adapter U/V real-imag parameters are trainable and encoder/generator/runtime remain frozen",
                "verify adapted real-path forward, anchor/transfer objective, nonzero adapter gradients, and zero forbidden gradients",
                "perform disposable three-step CUDA update through the exact adapter/base+shift/objective/evaluation path",
                "checkpoint/save/reload/resume adapter, optimizer, RNG, sampler, and frozen runtime identity",
                "exact-core synthetic forward/objective/update/evaluation/Smooth/cyclic/adapter metrics/diversity/bootstrap/report reload",
            ],
            "scientific_meaning": "none for synthetic path",
        },
        "acceptance": {
            "supported": ["natural Smooth Delta_R >= +0.05", "paired Smooth bootstrap CI lower > 0", "known cyclic transfer retained", "raw validity passes"],
            "not_supported": "any supported criterion fails after valid scientific execution",
            "stage_a_firewall": "R300 >= 0.937",
            "generation_readiness": "NOT_READY",
        },
        "governance": {"one_shot": True, "scientific_retry": False, "second_arm": False, "rank_sweep": False, "hyperparameter_sweep": False, "training_authorized": False, "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False, "protected_read": False, "test_read": False, "sealed_read": False, "hash_policy": "HASH_ON_TRUST_BOUNDARY_V2"},
        "claim_boundary": {"contract_only": True, "scientific_training_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "generation_executed": False, "formal_promotion": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    target = (args.output_dir or root / DEFAULT_OUTPUT_DIR).resolve()
    outputs = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs), "V10 contract output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite V10 contract output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    payload = build_contract(root)
    path = target / CONTRACT_NAME
    atomic_write_new_json(path, payload, artifact_kind="V10 low-rank remap canary contract")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": payload["outer_gate"], "contract": {"path": _relative(root, path), "sha256": file_sha256(path)}, "rank": ADAPTER_RANK, "scientific_training_executed": False, "training_authorized": False, "stage_b_authorized": False}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
