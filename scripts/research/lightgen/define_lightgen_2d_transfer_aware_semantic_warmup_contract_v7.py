"""Freeze the one-shot V7 transfer-aware semantic-warmup contract.

The V7 contract consumes the already validated V3/V4/V5 and historical
class-direction authorities.  It performs no TRAIN/DEV read, forward pass,
optimizer construction, or scientific execution.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training import file_sha256
from ldm_project.v2.evidence import atomic_write_new_json


TASK_ID = "DEFINE_LIGHTGEN_2D_TRANSFER_AWARE_SEMANTIC_WARMUP_CONTRACT_V7"
CONTRACT_SCHEMA = "lightgen_2d_transfer_aware_semantic_warmup_contract.v7"
CLASSIFICATION = "DEFINE_LIGHTGEN_2D_TRANSFER_AWARE_SEMANTIC_WARMUP_CONTRACT_V7_READY"
OUTER_GATE = "V7_TRANSFER_AWARE_SEMANTIC_WARMUP_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

V6_DEFINITION_PATH = Path(
    "scripts/research/lightgen/define_lightgen_2d_proto_only_semantic_warmup_contract_v6.py"
)
V6_CONTRACT_PATH = Path(
    "outputs/20260911_v6/runs/"
    "lightgen_2d_proto_only_semantic_warmup_contract_v6_attempt_01/"
    "lightgen_2d_proto_only_semantic_warmup_contract.v6.json"
)
V6_CONTRACT_SHA256 = "4b849632b67bd68f4075816e061bb3db0664abe31ab757a7137bfcd4c8f9f441"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v7/runs/"
    "lightgen_2d_transfer_aware_semantic_warmup_contract_v7_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_transfer_aware_semantic_warmup_contract.v7.json"


class ContractError(RuntimeError):
    """Raised when a frozen V7 authority cannot be resolved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    _require(not value.is_absolute(), f"{relative} must be relative")
    return (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_v6_definition() -> Any:
    path = Path(__file__).with_name(
        "define_lightgen_2d_proto_only_semantic_warmup_contract_v6.py"
    )
    spec = importlib.util.spec_from_file_location("lightgen_v6_definition_authority", path)
    _require(spec is not None and spec.loader is not None, "V6 definition authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_v6_parent(root: Path, v6: Any) -> dict[str, Any]:
    _require(v6._validate_git(root).get("branch") == EXPECTED_BRANCH, "V6 parent branch changed")
    _require(v6._validate_git(root).get("head") == EXPECTED_HEAD, "V6 parent HEAD changed")
    v6_path = _path(root, V6_CONTRACT_PATH)
    _require(v6_path.is_file(), "V6 contract authority is unavailable")
    _require(file_sha256(v6_path) == V6_CONTRACT_SHA256, "V6 contract authority SHA changed")
    parent = json.loads(v6_path.read_text(encoding="utf-8"))
    _require(
        parent.get("result") == "READY"
        and parent.get("status") == "valid_completed"
        and parent.get("outer_gate") == "V6_PROTO_ONLY_SEMANTIC_WARMUP_SUPPORTED",
        "V6 parent contract is not READY",
    )
    return {
        "path": V6_CONTRACT_PATH.as_posix(),
        "sha256": V6_CONTRACT_SHA256,
        "semantic_authority": "DEFINE_LIGHTGEN_2D_PROTO_ONLY_SEMANTIC_WARMUP_CONTRACT_V6",
    }


def _adapt_authorities(v6_authorities: Mapping[str, Any]) -> dict[str, Any]:
    authorities = copy.deepcopy(dict(v6_authorities))
    authorities["v3_control"]["initialization_role"] = (
        "V7 initial encoder; exact V3 CONTROL step-300 only; never V5/V6 final"
    )
    authorities["v4_generator"]["generator_phase"]["role"] = (
        "exact frozen V4 step-300 generator for V7"
    )
    authorities["v5_prototypes"]["source_role"] = (
        "TRAIN-only frozen prototypes; reused without recomputation"
    )
    authorities["v5_prototypes"]["recompute_during_v7"] = False
    authorities["historical_class_direction"]["semantic_authority"] = (
        "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED"
    )
    authorities["smooth_evaluator"]["role"] = (
        "frozen evaluation-only SmoothMNISTEvaluator v2.1; no gradient"
    )
    return authorities


def build_contract(root: Path) -> dict[str, Any]:
    v6 = _load_v6_definition()
    parent = _validate_v6_parent(root, v6)
    authorities = _adapt_authorities(v6._validate_authorities(root))
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
        "git": v6._validate_git(root),
        "definition_entrypoint": {
            "path": _relative(root, definition_path),
            "sha256": file_sha256(definition_path),
        },
        "parent_contract": parent,
        "authorities": authorities,
        "objective": {
            "question": (
                "Can a frozen generator-aware transfer objective improve natural "
                "semantic reconstruction and retain known cyclic class-direction transfer?"
            ),
            "training_objective": "L_V7=L_transfer=mean(softplus(-Delta_template))",
            "delta_template": (
                "M_template(raw_shift)-M_template(raw_base), where "
                "M_template=d(raw,tau_y)-d(raw,tau_t)"
            ),
            "latent_shift": "z_shift=z-mu_y+mu_t; t=(y+1) mod 10",
            "image_loss_gradient_contribution": "ZERO",
            "prototype_loss_gradient_contribution": "ZERO",
            "smooth_gradient_contribution": "ZERO",
            "image_loss_role": "evaluation-only",
            "prototype_margin_role": "evaluation-only/descriptive",
            "smooth_role": "evaluation-only",
            "template_distance": (
                "mean squared pixel distance after clamp [0,1] and "
                "adaptive_avg_pool2d to 28x28"
            ),
        },
        "model_assumption": (
            "A generator-aware transfer objective may make the frozen optical "
            "path respond to historical class-direction latent shifts, unlike "
            "static prototype-distance alignment."
        ),
        "design_decisions": [
            "Use exact V3 CONTROL step-300 encoder as the only initial encoder.",
            "Use exact V4 step-300 generator fully frozen.",
            "Reuse exact frozen V5 TRAIN-only prototypes and s_mu.",
            "Reuse exact historical TRAIN templates and cyclic target rule.",
            "Train only encoder.phase with no separate control arm, lambda, margin, or sweep.",
            "Use step-0 state as the matched baseline for all scientific deltas.",
        ],
        "protocol": {
            "trainable": ["encoder.phase"],
            "frozen": [
                "generator.phase",
                "prototypes",
                "TRAIN templates",
                "OLS",
                "T=I_100",
                "W_OLS",
                "SmoothMNISTEvaluator",
                "all other parameters",
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
                "known_transfer_pass": (
                    "template pass and Smooth pass and not solely one isolated pair"
                ),
            },
            "secondary": [
                "TRAIN transfer loss at step 0/100/200/300",
                "TRAIN/DEV image reconstruction loss",
                "OLS prototype margin",
                "per-class Smooth recall",
                "ten cyclic-pair summaries",
            ],
            "raw_validity": "finite, expected shape, structured, nonconstant at step 0/100/200/300 and cyclic base/shift",
            "stage_a_firewall": "R300 >= 0.937, independently reported and does not authorize Stage B",
        },
        "data_roles": {
            "TRAIN": {"rows": 55000, "used_for": "optimization"},
            "DEV": {"rows": 5000, "used_for": "frozen evaluation and balanced cyclic selection"},
            "protected": 0,
            "official_test": 0,
            "sealed": 0,
            "labels_as_model_input": False,
        },
        "engineering_preflight": {
            "required": [
                "resolve exact V3/V4/V5/historical/Smooth/runtime authorities",
                "verify V3 CONTROL initial encoder and V4 generator frozen",
                "verify encoder.phase sole trainable parameter",
                "prove image, prototype, Smooth, and frozen/generator gradients are absent",
                "prove finite nonzero transfer-loss gradient",
                "three-step CUDA smoke through the real entrypoint",
                "checkpoint/reload/resume identity",
                "exact-core synthetic train/eval/transfer/bootstrap/sufficient/report/lineage reload",
            ],
            "scientific_meaning": "none for synthetic path",
        },
        "governance": {
            "one_shot": True,
            "scientific_retry": False,
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
    _require(target.is_relative_to(outputs), "V7 contract output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite V7 contract output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    payload = build_contract(root)
    path = target / CONTRACT_NAME
    atomic_write_new_json(path, payload, artifact_kind="V7 transfer-aware semantic-warmup contract")
    result = {
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "outer_gate": payload["outer_gate"],
        "contract": {"path": _relative(root, path), "sha256": file_sha256(path)},
        "scientific_training_executed": False,
        "training_authorized": False,
        "stage_b_authorized": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
