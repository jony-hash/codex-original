"""Freeze the V11 semantic/reconstruction compatibility contract.

V11 supersedes only the V10.1 training objective by adding one deterministic,
step-0 gradient-balanced reconstruction term.  The rank-nine adapter,
identity initialization, optical runtime, semantic objective, data roles and
all scientific conditions remain inherited from V10.1.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_v10_1_definition() -> Any:
    path = Path(__file__).with_name("define_lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_1.py")
    spec = importlib.util.spec_from_file_location("lightgen_v10_1_definition_authority_v11", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V10.1 definition authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V10_1 = _load_v10_1_definition()

TASK_ID = "DEFINE_LIGHTGEN_2D_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_CONTRACT_V11"
CONTRACT_SCHEMA = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract.v11"
CLASSIFICATION = "DEFINE_LIGHTGEN_2D_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_CONTRACT_V11_READY"
OUTER_GATE = "V11_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

V10_1_CONTRACT_PATH = Path(
    "outputs/20260912_v10_1/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_1_attempt_01/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10_1.json"
)
V10_1_CONTRACT_SHA256 = "45210d92e00ba01255a4340dfc837c514c110640097aa20821d8d5b6e0efab3d"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract_v11_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract.v11.json"


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract(root: Path) -> dict[str, Any]:
    payload = copy.deepcopy(V10_1.build_contract(root))
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
            "supersedes": {
                "semantic_authority": "DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10_1",
                "path": V10_1_CONTRACT_PATH.as_posix(),
                "sha256": V10_1_CONTRACT_SHA256,
                "scope": "add one deterministic reconstruction-compatibility term; no optical or rank redesign",
            },
        }
    )

    objective = payload["objective"]
    objective["question"] = (
        "Can the supported rank-nine OLS-to-generator remapping preserve its semantic gain "
        "while preserving reconstruction quality?"
    )
    objective["semantic_objective"] = "L_sem=L_anchor+lambda_transfer*L_transfer"
    objective["training_objective"] = "L_V11=L_sem+lambda_image*L_image"
    objective["reconstruction_compatibility"] = {
        "loss": "mean squared pixel error between adapted raw CMOS and image target",
        "formula": "L_image=mean((raw_base-target)^2)",
        "gradient_role": "differentiated adapter gradient; image loss is no longer evaluation-only",
    }
    objective["lambda_image"] = {
        "formula": "g_sem/(g_image+1e-12)",
        "g_sem": "||grad_adapter L_sem||",
        "g_image": "||grad_adapter L_image||",
        "batch": "same deterministic initial TRAIN batch as the preflight gradient probe",
        "freeze_step": 0,
        "clipping": False,
        "tuning": False,
        "sweep": False,
        "recalibration": False,
    }
    objective["gradient_contributions"] = {
        "semantic_anchor_transfer": "differentiated",
        "image_reconstruction": "differentiated with frozen lambda_image",
        "prototype_cross_entropy": "ZERO; prototypes are frozen shift support only",
        "Smooth_evaluator": "ZERO; evaluation-only",
    }

    payload["metrics"]["primary_reconstruction_compatibility"] = {
        "name": "Delta_L_image_DEV",
        "formula": "DEV L_image(step300)-DEV L_image(step0)",
        "pass": "Delta <= +0.005",
    }
    payload["metrics"]["primary_natural_semantic"]["name"] = "Delta_R_Smooth"
    payload["metrics"]["primary_natural_semantic"]["pass"] = "Delta >= +0.05 and paired bootstrap 95% CI lower > 0"
    payload["metrics"]["primary_compatibility_gate"] = {
        "all": [
            "Smooth Delta_R >= +0.05",
            "paired Smooth bootstrap CI lower > 0",
            "DEV Delta_L_image <= +0.005",
            "known cyclic template transfer retained",
            "known cyclic Smooth transfer retained",
            "raw validity passes",
        ],
        "failure_classification": "V11_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_NOT_SUPPORTED",
    }

    payload["design_decisions"] = [
        decision
        for decision in payload["design_decisions"]
        if "V10" not in decision or "adapter" in decision.lower()
    ] + [
        "Keep the V10.1 rank-nine asymmetric identity initialization U0=0.01*I_9 and V0=0.",
        "Add only lambda_image*L_image to the unchanged V10.1 semantic objective.",
        "Freeze lambda_image exactly once at step zero from the same deterministic initial TRAIN batch using g_sem/(g_image+1e-12).",
        "Do not clip, tune, sweep, recalibrate, or select lambda_image after scientific outcomes.",
        "Keep the independent 0.937 Stage-A firewall unchanged; V11 compatibility support does not authorize Stage B.",
    ]

    payload["engineering_preflight"] = {
        **payload["engineering_preflight"],
        "required": [
            "resolve exact V10.1/V3/V4/V5/V9 authorities and existing DEV/cyclic selection authorities",
            "verify U0=0.01*I_9, V0=0, U0 V0^H=0 and exact step-0 T_adapt=I_100",
            "compute g_sem, g_image and freeze lambda_image from the same initial TRAIN batch",
            "verify all required gradients finite, g_sem/g_image finite and positive, and total V11 adapter gradient usable",
            "verify image loss is in the differentiated objective while Smooth/prototype/evaluator paths remain excluded",
            "verify only adapter U/V real-imag parameters are trainable and encoder/generator/runtime remain frozen",
            "perform disposable three-step CUDA update smoke, checkpoint/save/reload/resume, and exact-core synthetic end-to-end",
        ],
        "scientific_meaning": "none for synthetic path",
    }

    payload["acceptance"] = {
        "supported": [
            "Smooth R300-R0 >= +0.05",
            "paired Smooth bootstrap CI lower > 0",
            "DEV L_image300-L_image0 <= +0.005",
            "known cyclic template transfer retained",
            "known cyclic Smooth transfer retained",
            "raw validity passes",
        ],
        "not_supported": "any supported criterion fails after valid scientific execution",
        "stage_a_firewall": "R300 >= 0.937",
        "generation_readiness": "NOT_READY",
    }
    payload["governance"].update(
        {
            "v11_reconstruction_compatibility": True,
            "lambda_image_frozen_at_step0": True,
            "lambda_image_clipping": False,
            "lambda_image_tuning": False,
            "lambda_image_sweep": False,
            "rank": V10_1.V10.ADAPTER_RANK,
            "rank_sweep": False,
            "scientific_retry": False,
        }
    )
    payload["claim_boundary"].update({"v11_contract_only": True, "scientific_training_executed": False})
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    target = (args.output_dir or root / DEFAULT_OUTPUT_DIR).resolve()
    if not target.is_relative_to((root / "outputs").resolve()):
        raise RuntimeError("V11 contract output must remain under outputs/")
    if target.exists():
        raise RuntimeError(f"refusing to overwrite V11 contract output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    payload = build_contract(root)
    path = target / CONTRACT_NAME
    V10_1.V10.atomic_write_new_json(path, payload, artifact_kind="V11 semantic reconstruction compatibility contract")
    result = {
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "outer_gate": payload["outer_gate"],
        "contract": {"path": path.relative_to(root).as_posix(), "sha256": _file_sha256(path)},
        "supersedes": payload["supersedes"],
        "rank": V10_1.V10.ADAPTER_RANK,
        "scientific_training_executed": False,
        "training_authorized": False,
        "stage_b_authorized": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
