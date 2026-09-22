"""Freeze the superseding V10.1 low-rank remap canary contract.

V10.1 changes only the impossible V10 step-0 factor initialization.  The
optical runtime, OLS interface, objective, optimizer, data roles, metrics and
outer gate are inherited unchanged from the V10/V9 authority chain.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_v10_definition() -> Any:
    path = Path(__file__).with_name("define_lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10.py")
    spec = importlib.util.spec_from_file_location("lightgen_v10_definition_authority_v10_1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V10 definition authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V10 = _load_v10_definition()

TASK_ID = "DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10_1"
CONTRACT_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10_1"
CLASSIFICATION = "DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10_1_READY"
OUTER_GATE = "V10_LOW_RANK_OLS_TO_GENERATOR_REMAP_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

V10_CONTRACT_PATH = Path(
    "outputs/20260912_v10/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_attempt_01/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10.json"
)
V10_CONTRACT_SHA256 = "5efeb9d4b8e5e1e29fee3e3c445a52a201fbfabdd7f696a183284c71832ca0f1"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260912_v10_1/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_1_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10_1.json"


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract(root: Path) -> dict[str, Any]:
    payload = copy.deepcopy(V10.build_contract(root))
    payload.update(
        {
            "schema": CONTRACT_SCHEMA,
            "task_id": TASK_ID,
            "classification": CLASSIFICATION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"branch": EXPECTED_BRANCH, "head": EXPECTED_HEAD},
            "definition_entrypoint": {
                "path": Path(__file__).resolve().relative_to(root.resolve()).as_posix(),
                "sha256": _file_sha256(Path(__file__).resolve()),
            },
            "supersedes": {
                "semantic_authority": "DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10",
                "path": V10_CONTRACT_PATH.as_posix(),
                "sha256": V10_CONTRACT_SHA256,
                "scope": "only the impossible step-0 factor initialization clause",
            },
        }
    )

    adapter = payload["adapter"]
    adapter["rank_semantics"] = "r=9 is factor width and maximum residual rank; exact identity may have residual rank 0"
    adapter["initialization"] = {
        "scale": V10.INIT_SCALE,
        "u_real": "INIT_SCALE*I_9 in rows 0..8, zero elsewhere",
        "v_real": "all zeros",
        "u_imag": "zero",
        "v_imag": "zero",
        "support_rows_U": list(range(V10.ADAPTER_RANK)),
        "support_rows_V": [],
        "residual_at_step0": "exact zero in float32 construction; T_adapt=I_100",
    }
    adapter["gradient_semantics"] = {
        "at_step0": "grad_U may be exactly zero while grad_V is nonzero for R=UV^H with V0=0",
        "required": "all relevant gradients finite; total adapter gradient finite and nonzero; one optimizer step departs from identity",
        "per_factor_nonzero_required": False,
    }

    payload["design_decisions"] = [
        decision
        for decision in payload["design_decisions"]
        if "orthogonal-support U/V initialization" not in decision
    ] + [
        "Use asymmetric initialization U0=INIT_SCALE*I_9 in rows 0..8 and V0=0 so U0 V0^H=0 exactly.",
        "Treat rank nine as factor width and maximum residual rank; residual rank zero at exact identity is expected.",
        "Permit zero grad_U at exact bilinear identity initialization; require finite nonzero total adapter gradient and a one-step departure.",
    ]

    payload["engineering_preflight"] = {
        **payload["engineering_preflight"],
        "required": [
            "resolve exact V3 encoder, V4 generator, V5 prototypes, historical templates, Smooth, V2 runtime, and V9 parent authorities",
            "verify U0=INIT_SCALE*I_9, V0=0, U0 V0^H=0 and exact step-0 T_adapt=I_100",
            "verify arbitrary synthetic and production synthetic complex coefficients satisfy identity",
            "verify canonical downstream and identity-adapter downstream paths agree",
            "verify only adapter U/V real-imag parameters are trainable and encoder/generator/runtime remain frozen",
            "verify all relevant gradients are finite, total adapter gradient is finite and nonzero, and report grad_U/grad_V separately",
            "perform exactly one disposable optimizer step that moves T_adapt away from identity, followed by three-step CUDA update smoke",
            "checkpoint/save/reload/resume adapter, optimizer, RNG, sampler, and frozen runtime identity",
            "exact-core synthetic forward/objective/update/evaluation/Smooth/cyclic/adapter metrics/diversity/bootstrap/report reload",
        ],
    }
    payload["governance"].update(
        {
            "supersedes_v10_initialization_only": True,
            "rank": V10.ADAPTER_RANK,
            "rank_sweep": False,
            "scientific_retry": False,
        }
    )
    payload["claim_boundary"].update({"supersedes_v10_only": True, "scientific_training_executed": False})
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    target = (args.output_dir or root / DEFAULT_OUTPUT_DIR).resolve()
    outputs = (root / "outputs").resolve()
    if not target.is_relative_to(outputs):
        raise RuntimeError("V10.1 contract output must remain under outputs/")
    if target.exists():
        raise RuntimeError(f"refusing to overwrite V10.1 contract output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    payload = build_contract(root)
    path = target / CONTRACT_NAME
    V10.atomic_write_new_json(path, payload, artifact_kind="V10.1 low-rank remap canary contract")
    result = {
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "outer_gate": payload["outer_gate"],
        "contract": {"path": path.relative_to(root).as_posix(), "sha256": _file_sha256(path)},
        "supersedes": payload["supersedes"],
        "rank": V10.ADAPTER_RANK,
        "scientific_training_executed": False,
        "training_authorized": False,
        "stage_b_authorized": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
