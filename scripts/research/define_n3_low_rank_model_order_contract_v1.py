"""Freeze the bounded N3 low-rank model-order selection contract."""

from __future__ import annotations

import copy
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_v11_definition() -> Any:
    path = Path(__file__).with_name("define_lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract_v11.py")
    spec = importlib.util.spec_from_file_location("lightgen_v11_definition_authority_n3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V11 definition authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V11 = _load_v11_definition()

TASK_ID = "DEFINE_N3_LOW_RANK_MODEL_ORDER_CONTRACT_V1"
CONTRACT_SCHEMA = "n3_low_rank_model_order_contract.v1"
CLASSIFICATION = "N3_LOW_RANK_MODEL_ORDER_CONTRACT_READY"
OUTER_GATE = "N3_LOW_RANK_MODEL_ORDER_RESOLVED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

V11_CONTRACT_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract_v11_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract.v11.json"
)
V11_CONTRACT_SHA256 = "6c49a0cff4e08622351438529cebafe9388537d713a5b78b9fabbc953904efff"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260912_n3/runs/"
    "n3_low_rank_model_order_contract_v1_attempt_01"
)
CONTRACT_NAME = "n3_low_rank_model_order_contract.v1.json"
CANDIDATE_RANKS = list(range(1, 10))
NEW_SCIENTIFIC_RANKS = list(range(1, 9))
REUSED_RANKS = [9]
LAMBDA_TRANSFER = 28.48464149920509
LAMBDA_IMAGE = 13.353282014834436


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract(root: Path) -> dict[str, Any]:
    payload = copy.deepcopy(V11.build_contract(root))
    payload.update(
        {
            "schema": CONTRACT_SCHEMA,
            "task_id": TASK_ID,
            "classification": CLASSIFICATION,
            "outer_gate": OUTER_GATE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"branch": EXPECTED_BRANCH, "head": EXPECTED_HEAD},
            "definition_entrypoint": {
                "path": Path(__file__).resolve().relative_to(root.resolve()).as_posix(),
                "sha256": _file_sha256(Path(__file__).resolve()),
            },
            "parent_contract": {
                "semantic_authority": "DEFINE_LIGHTGEN_2D_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_CONTRACT_V11",
                "path": V11_CONTRACT_PATH.as_posix(),
                "sha256": V11_CONTRACT_SHA256,
            },
            "supersedes": {
                "semantic_authority": "DEFINE_LIGHTGEN_2D_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_CONTRACT_V11",
                "path": V11_CONTRACT_PATH.as_posix(),
                "sha256": V11_CONTRACT_SHA256,
                "scope": "freeze a bounded nested low-rank candidate family and select the minimum adequate order",
            },
        }
    )

    payload["objective"].update(
        {
            "question": "What is the minimum low-rank order that retains V11 semantic-reconstruction compatibility?",
            "training_objective": "L_N3=L_sem+13.353282014834436*L_image",
            "semantic_objective": "L_sem=L_anchor+28.48464149920509*L_transfer",
            "lambda_transfer": LAMBDA_TRANSFER,
            "lambda_image_value": LAMBDA_IMAGE,
            "lambda_policy": "reuse exact frozen V11 values for every rank; no rank-specific recalibration",
        }
    )
    payload["adapter"].update(
        {
            "rank": "r in {1,2,3,4,5,6,7,8,9}",
            "factor_shapes": {"U": [100, "r"], "V": [100, "r"]},
            "matrix_formula": "T_r=I_100+U_r V_r^H",
            "rank_semantics": "factor width r; structural rank(U_r V_r^H) <= r; numerical rank is descriptive only",
            "nested_initialization": {
                "U0": "0.01*[e0,...,e_(r-1)]",
                "V0": "0",
                "residual_at_step0": "U0 V0^H=0; T_r(0)=I_100",
                "support_rows_U": "0..r-1",
                "support_rows_V": [],
            },
        }
    )
    payload["model_order"] = {
        "family": "T_r=I_100+U_r V_r^H with U_r,V_r in C^(100 x r)",
        "candidate_ranks": CANDIDATE_RANKS,
        "new_scientific_ranks": NEW_SCIENTIFIC_RANKS,
        "reused_ranks": REUSED_RANKS,
        "rank_zero_allowed": False,
        "rank_above_nine_allowed": False,
        "adaptive_expansion": False,
        "selection_rule": "minimum integer r with MODEL_ADEQUATE_r=true",
        "performance_maximization_selection": False,
        "complexity_penalty": False,
        "rationale": "the centered ten-class prototype subspace has maximum dimension nine",
        "rationale_type": "DESIGN_DECISION; not a claim that physical semantic rank equals nine",
    }
    payload["protocol"].update(
        {
            "rank_candidates": CANDIDATE_RANKS,
            "candidate_execution_order": NEW_SCIENTIFIC_RANKS,
            "rank9_authority_reuse": "REUSED_VALID_AUTHORITY only after exact V11 equivalence verification",
            "same_objective_across_ranks": True,
            "same_lambdas_across_ranks": True,
            "same_train_order_rng_across_ranks": True,
        }
    )
    payload["metrics"]["model_order_selection"] = {
        "semantic_pass": "Smooth Delta >= +0.05 and paired bootstrap 95% CI lower > 0",
        "reconstruction_pass": "DEV L_image(step300)-DEV L_image(step0) <= +0.005",
        "transfer_pass": "cyclic template and Smooth means/paired CIs positive, broad, and not pair-dominated",
        "raw_pass": "finite, expected shape, structured, nonconstant",
        "model_adequate": "semantic_pass AND reconstruction_pass AND transfer_pass AND raw_pass",
        "selection": "minimum integer rank among adequate candidates",
        "stage_a_firewall": "R300 >= 0.937, independent for every rank",
    }
    payload["engineering_preflight"] = {
        "required": [
            "verify exact V11 contract, input, report and lineage authority for rank-9 reuse",
            "instantiate every rank 1..9 and verify nested T_r(0)=I_100",
            "verify adapter-only trainable set and frozen encoder/generator/OLS/W_OLS/templates/prototypes/Smooth",
            "verify finite usable V11 gradients and identical frozen lambda values for every rank",
            "execute one disposable three-step real TRAIN CUDA smoke and checkpoint/reload/resume for every new rank",
            "execute exact-core synthetic end-to-end for representative low/high ranks through the shared generic path",
            "verify report/input/lineage schemas carry candidate rank and model-order scope",
        ],
        "scientific_meaning": "none for preflight synthetic or disposable updates",
    }
    payload["acceptance"] = {
        "candidate_adequate": "MODEL_ADEQUATE_r under frozen V11 thresholds",
        "resolved": "all required ranks 1..8 valid, rank-9 reuse verified, and minimum adequate rank selected",
        "not_resolved": "no candidate adequate after all required valid outcomes",
        "outer_gate": OUTER_GATE,
        "stage_a_firewall": "reported independently; does not authorize Stage B",
        "generation_readiness": "NOT_READY",
    }
    payload["design_decisions"] = [
        "Freeze exactly ranks 1 through 9; do not expand the domain or add rank zero.",
        "Use nested U/V identity-preserving initialization for every candidate.",
        "Reuse exact V11 semantic-reconstruction objective, optimizer, data roles and evaluation semantics.",
        "Do not recalibrate, tune or sweep either lambda by rank.",
        "Reuse V11 rank-9 scientific authority only after exact equivalence verification; do not rerun rank 9.",
        "Select feasibility first, then minimum integer rank; do not maximize Smooth or add a complexity penalty.",
        "Keep the independent 0.937 Stage-A firewall and all downstream closures unchanged.",
    ]
    payload["governance"].update(
        {
            "n3_model_order_selection": True,
            "candidate_rank_domain": CANDIDATE_RANKS,
            "rank_domain_expansion": False,
            "rank_specific_lambda_recalibration": False,
            "rank9_reuse_required": True,
            "scientific_retry": False,
            "second_arm": False,
            "stage_b_authorized": False,
            "generation_readiness": "not_ready",
            "formal_promotion": False,
        }
    )
    payload["claim_boundary"].update(
        {
            "n3_contract_only": True,
            "scientific_training_executed": False,
            "stage_b_executed": False,
            "generation_executed": False,
            "protected_accessed": False,
            "official_test_accessed": False,
            "sealed_accessed": False,
            "formal_promotion": False,
        }
    )
    payload["result"] = "READY"
    payload["status"] = "valid_completed"
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    target = (args.output_dir or root / DEFAULT_OUTPUT_DIR).resolve()
    if not target.is_relative_to((root / "outputs").resolve()):
        raise RuntimeError("N3 contract output must remain under outputs/")
    if target.exists():
        raise RuntimeError(f"refusing to overwrite N3 contract output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    payload = build_contract(root)
    path = target / CONTRACT_NAME
    V11.V10_1.V10.atomic_write_new_json(path, payload, artifact_kind="N3 low-rank model-order contract")
    result = {
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "outer_gate": payload["outer_gate"],
        "contract": {"path": path.relative_to(root).as_posix(), "sha256": _file_sha256(path)},
        "candidate_ranks": CANDIDATE_RANKS,
        "new_scientific_ranks": NEW_SCIENTIFIC_RANKS,
        "reused_ranks": REUSED_RANKS,
        "scientific_training_executed": False,
        "training_authorized": False,
        "stage_b_authorized": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
