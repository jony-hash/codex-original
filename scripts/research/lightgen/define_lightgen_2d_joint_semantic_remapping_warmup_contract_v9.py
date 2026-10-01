"""Freeze the one-shot V9 joint semantic-remapping warmup contract.

V9 consumes the validated V8 authority chain and changes exactly one main
degree of freedom: ``generator.phase`` becomes trainable alongside
``encoder.phase``.  The semantic objective, constants, data roles, runtime,
and 300-update budget remain fixed.
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


TASK_ID = "DEFINE_LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_CONTRACT_V9"
CONTRACT_SCHEMA = "lightgen_2d_joint_semantic_remapping_warmup_contract.v9"
CLASSIFICATION = "DEFINE_LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_CONTRACT_V9_READY"
OUTER_GATE = "V9_JOINT_SEMANTIC_REMAPPING_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

V8_CONTRACT_PATH = Path(
    "outputs/20260911_v8/runs/"
    "lightgen_2d_absolute_anchor_plus_transfer_warmup_contract_v8_attempt_01/"
    "lightgen_2d_absolute_anchor_plus_transfer_warmup_contract.v8.json"
)
V8_CONTRACT_SHA256 = "02cef2918ff89eafaa53cf52f7a7492b3091e84acba3c351b6142b94b703a85d"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260912_v9/runs/"
    "lightgen_2d_joint_semantic_remapping_warmup_contract_v9_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_joint_semantic_remapping_warmup_contract.v9.json"

S_TAU = 0.02919053753794475
LAMBDA_TRANSFER = 28.48464149920509


class ContractError(RuntimeError):
    """A frozen V9 authority cannot be resolved."""


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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _load_v8_parent(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _path(root, V8_CONTRACT_PATH)
    _require(path.is_file(), "V8 contract authority is unavailable")
    _require(file_sha256(path) == V8_CONTRACT_SHA256, "V8 contract authority SHA changed")
    parent = _read_json(path, "V8 contract")
    _require(
        parent.get("schema") == "lightgen_2d_absolute_anchor_plus_transfer_warmup_contract.v8"
        and parent.get("result") == "READY"
        and parent.get("status") == "valid_completed"
        and parent.get("outer_gate") == "V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER_SUPPORTED",
        "V8 parent contract is not READY",
    )
    git = parent.get("git")
    _require(isinstance(git, Mapping), "V8 parent Git identity is missing")
    _require(git.get("branch") == EXPECTED_BRANCH and git.get("head") == EXPECTED_HEAD, "V8 parent Git identity changed")
    definition = parent.get("definition_entrypoint")
    _require(isinstance(definition, Mapping), "V8 definition entrypoint is missing")
    definition_path = _path(root, str(definition.get("path")))
    _require(file_sha256(definition_path) == definition.get("sha256"), "V8 definition entrypoint SHA changed")
    objective = parent.get("objective")
    _require(isinstance(objective, Mapping), "V8 objective is missing")
    _require(float(objective.get("s_tau", 0.0)) == S_TAU, "V8 s_tau authority changed")
    return (
        {
            "path": V8_CONTRACT_PATH.as_posix(),
            "sha256": V8_CONTRACT_SHA256,
            "semantic_authority": "DEFINE_LIGHTGEN_2D_ABSOLUTE_ANCHOR_PLUS_TRANSFER_WARMUP_CONTRACT_V8",
        },
        parent,
    )


def build_contract(root: Path) -> dict[str, Any]:
    parent_ref, parent = _load_v8_parent(root)
    authorities = copy.deepcopy(parent["authorities"])
    authorities["v3_control"]["initialization_role"] = "exact V3 CONTROL step-300 encoder; not V5/V6/V7/V8 final"
    authorities["v4_generator"]["initialization_role"] = "exact V4 valid step-300 generator; not V5/V6/V7/V8 final"
    authorities["smooth_evaluator"]["role"] = "frozen evaluation-only SmoothMNISTEvaluator v2.1; no training gradient"

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
        "definition_entrypoint": {
            "path": _relative(root, definition_path),
            "sha256": file_sha256(definition_path),
        },
        "parent_contract": parent_ref,
        "authorities": authorities,
        "objective": {
            "question": (
                "Can making generator.phase trainable, while holding the V8 semantic objective "
                "and all optical/training conditions fixed, improve real-path semantic remapping?"
            ),
            "training_objective": "L_V9=L_anchor+lambda_transfer*L_transfer",
            "anchor": {
                "logits": "b_k(raw_base)=-d_image(raw_base,tau_k)/(s_tau+1e-12)",
                "loss": "cross_entropy([b_0,...,b_9], y)",
                "output": "natural raw CMOS output raw_base=G(E(x))",
                "representation": "frozen historical TRAIN templates tau_0...tau_9",
            },
            "transfer": {
                "loss": "softplus(-Delta_template)",
                "delta_template": "M_template(raw_shift)-M_template(raw_base)",
                "margin": "M_template(r)=d_image(r,tau_y)-d_image(r,tau_t)",
                "latent_shift": "z_shift=z-mu_y+mu_t",
                "target_rule": "t=(y+1) mod 10",
            },
            "image_distance": "mean squared pixel distance after clamp [0,1] and adaptive_avg_pool2d to 28x28",
            "s_tau": S_TAU,
            "lambda_transfer": LAMBDA_TRANSFER,
            "lambda_rule": {
                "formula": "reuse exact frozen V8 lambda_transfer",
                "source": "V8 initial-gradient rule; no V9 recalibration",
                "recompute": False,
                "clipping_or_tuning": False,
            },
            "gradient_contributions": {
                "image_reconstruction": "ZERO; evaluation-only",
                "prototype_cross_entropy": "ZERO; prototypes are frozen shift support only",
                "Smooth_evaluator": "ZERO; evaluation-only",
            },
        },
        "model_assumption": (
            "The remaining limitation after V8 may be insufficient adaptability of the frozen "
            "OLS-to-generator mapping; jointly updating encoder.phase and generator.phase may "
            "improve semantic remapping without changing the optical architecture."
        ),
        "design_decisions": [
            "Use exact V3 CONTROL step-300 encoder and exact V4 valid step-300 generator as step-0.",
            "Change only generator.phase from frozen to trainable relative to V8.",
            "Reuse exact V8 s_tau and lambda_transfer; never recalibrate either constant.",
            "Train exactly encoder.phase and generator.phase with Adam lr=1e-3 for 300 updates.",
            "Use step-0 as the matched baseline; no separate control arm, sweep, retry, or budget extension.",
            "Keep the independent Smooth gate and the 0.937 Stage-A firewall unchanged.",
        ],
        "protocol": {
            "trainable": ["encoder.phase", "generator.phase"],
            "frozen": [
                "OLS definition",
                "T=I_100",
                "shared W_OLS",
                "TRAIN templates",
                "TRAIN prototypes",
                "SmoothMNISTEvaluator",
                "all remaining model/runtime parameters",
            ],
            "optimizer": {
                "name": "Adam",
                "learning_rate": 1.0e-3,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
            "logical_batch_size": 10,
            "updates": 300,
            "checkpoints": [0, 100, 200, 300],
            "propagation_grid": 2048,
            "rng_convention": {"initialization_seed": 1234, "schedule_seed": 1234},
            "cyclic_transfer": {
                "target_rule": "t=(y+1) mod 10",
                "selection_rows": 1000,
                "selection_per_class": 100,
                "selection_seed": 4321,
                "no_target_search": True,
                "no_magnitude_sweep": True,
            },
            "bootstrap": {"replicates": 999, "seed": 271828, "type": "paired class-stratified"},
        },
        "metrics": {
            "primary_natural_semantic": {
                "name": "Delta_R_Smooth",
                "formula": "R300-R0 using frozen SmoothMNISTEvaluator v2.1",
                "pass": "Delta >= +0.05 and paired bootstrap 95% CI lower > 0",
            },
            "primary_transfer": {
                "template": "Delta_template=M_shift-M_base",
                "smooth": "Delta_smooth=(score_t-score_y)_shift-(score_t-score_y)_base",
                "known_transfer_pass": "template pass and Smooth pass and broad positive pair coverage",
            },
            "descriptive": [
                "TRAIN anchor/transfer/total semantic losses at 0/100/200/300",
                "TRAIN/DEV image loss",
                "Smooth macro/per-class recall",
                "template-anchor macro/per-class recall and correct-class margin",
                "OLS prototype margin",
                "step-0/step-300 raw collapse/diversity diagnostics",
                "all ten cyclic-pair summaries",
                "encoder/generator gradient and update diagnostics",
            ],
            "raw_validity": "finite, expected shape, structured, nonconstant at all checkpoints and cyclic base/shift",
            "stage_a_firewall": "R300 >= 0.937, independently reported and does not authorize Stage B",
        },
        "data_roles": {
            "TRAIN": {"rows": 55000, "used_for": "optimization and fixed initial-gradient validation"},
            "DEV": {"rows": 5000, "used_for": "frozen evaluation and balanced cyclic selection"},
            "calibration": 0,
            "protected": 0,
            "official_test": 0,
            "sealed": 0,
            "labels_as_model_input": False,
        },
        "engineering_preflight": {
            "required": [
                "resolve exact V3 encoder, V4 generator, V5 prototypes, historical templates, Smooth, and V2 runtime authorities",
                "verify exact s_tau and exact reused V8 lambda_transfer",
                "verify exactly encoder.phase and generator.phase are trainable",
                "verify L_anchor gradients are finite/nonzero for both trainable parameters",
                "verify L_transfer gradients are finite/nonzero where mathematically expected",
                "verify image/prototype/Smooth gradients do not enter the training objective",
                "perform disposable three-step CUDA joint-update smoke through the real entrypoint",
                "checkpoint/save/reload/resume encoder, generator, optimizer, RNG, sampler, and runtime",
                "exact-core synthetic end-to-end forward/objective/update/evaluation/statistics/report reload",
            ],
            "scientific_meaning": "none for synthetic path",
        },
        "acceptance": {
            "supported": [
                "natural Smooth Delta_R >= +0.05",
                "paired Smooth bootstrap CI lower > 0",
                "known cyclic transfer retained",
                "raw validity passes",
            ],
            "not_supported": "any supported criterion fails after valid scientific execution",
            "stage_a_firewall": "R300 >= 0.937",
            "generation_readiness": "NOT_READY",
        },
        "governance": {
            "one_shot": True,
            "scientific_retry": False,
            "second_arm": False,
            "second_hyperparameter_attempt": False,
            "training_authorized": False,
            "stage_b_authorized": False,
            "generation_readiness": "not_ready",
            "formal_promotion": False,
            "protected_read": False,
            "test_read": False,
            "sealed_read": False,
            "hash_policy": "HASH_ON_TRUST_BOUNDARY_V2",
        },
        "claim_boundary": {
            "contract_only": True,
            "scientific_training_executed": False,
            "stage_b_executed": False,
            "scale_extraction_executed": False,
            "prior_shaping_executed": False,
            "generation_executed": False,
            "formal_promotion": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    target = (args.output_dir or root / DEFAULT_OUTPUT_DIR).resolve()
    outputs = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs), "V9 contract output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite V9 contract output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    payload = build_contract(root)
    path = target / CONTRACT_NAME
    atomic_write_new_json(path, payload, artifact_kind="V9 joint semantic-remapping contract")
    result = {
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "outer_gate": payload["outer_gate"],
        "contract": {"path": _relative(root, path), "sha256": file_sha256(path)},
        "s_tau": S_TAU,
        "lambda_transfer": LAMBDA_TRANSFER,
        "scientific_training_executed": False,
        "training_authorized": False,
        "stage_b_authorized": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
