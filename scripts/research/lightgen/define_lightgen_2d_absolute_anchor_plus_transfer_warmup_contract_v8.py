"""Freeze the one-shot V8 absolute-anchor plus transfer-warmup contract.

This definition consumes the validated V7 authority chain without reading
TRAIN/DEV rows or constructing a runtime.  It freezes the historical image
templates, the absolute-anchor distance scale, and the deterministic
initial-gradient rule before the scientific arm is allowed to run.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training import file_sha256
from ldm_project.v2.evidence import atomic_write_new_json, safe_torch_load


TASK_ID = "DEFINE_LIGHTGEN_2D_ABSOLUTE_ANCHOR_PLUS_TRANSFER_WARMUP_CONTRACT_V8"
CONTRACT_SCHEMA = "lightgen_2d_absolute_anchor_plus_transfer_warmup_contract.v8"
CLASSIFICATION = "DEFINE_LIGHTGEN_2D_ABSOLUTE_ANCHOR_PLUS_TRANSFER_WARMUP_CONTRACT_V8_READY"
OUTER_GATE = "V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

V7_CONTRACT_PATH = Path(
    "outputs/20260911_v7/runs/"
    "lightgen_2d_transfer_aware_semantic_warmup_contract_v7_attempt_01/"
    "lightgen_2d_transfer_aware_semantic_warmup_contract.v7.json"
)
V7_CONTRACT_SHA256 = "974d6dc560d2f4427cc75e5dc3d3404140d6e2a84439d800f6af10e299bc7766"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v8/runs/"
    "lightgen_2d_absolute_anchor_plus_transfer_warmup_contract_v8_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_absolute_anchor_plus_transfer_warmup_contract.v8.json"


class ContractError(RuntimeError):
    """Raised when a frozen V8 authority cannot be resolved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    _require(not value.is_absolute(), f"{relative} must be relative")
    return (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_v7_definition() -> Any:
    path = Path(__file__).with_name(
        "define_lightgen_2d_transfer_aware_semantic_warmup_contract_v7.py"
    )
    spec = importlib.util.spec_from_file_location("lightgen_v7_definition_authority", path)
    _require(spec is not None and spec.loader is not None, "V7 definition authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_v7_parent(root: Path) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    v7 = _load_v7_definition()
    v6 = v7._load_v6_definition()
    v7_git = v6._validate_git(root)
    _require(v7_git.get("branch") == EXPECTED_BRANCH, "V7 parent branch changed")
    _require(v7_git.get("head") == EXPECTED_HEAD, "V7 parent HEAD changed")
    v7_path = _path(root, V7_CONTRACT_PATH)
    _require(v7_path.is_file(), "V7 contract authority is unavailable")
    _require(file_sha256(v7_path) == V7_CONTRACT_SHA256, "V7 contract authority SHA changed")
    parent = json.loads(v7_path.read_text(encoding="utf-8"))
    _require(
        parent.get("result") == "READY"
        and parent.get("status") == "valid_completed"
        and parent.get("outer_gate") == "V7_TRANSFER_AWARE_SEMANTIC_WARMUP_SUPPORTED",
        "V7 parent contract is not READY",
    )
    definition = parent.get("definition_entrypoint")
    _require(isinstance(definition, Mapping), "V7 definition entrypoint is missing")
    _require(
        file_sha256(_path(root, definition["path"])) == definition["sha256"],
        "V7 definition entrypoint SHA changed",
    )
    return (
        {
            "path": V7_CONTRACT_PATH.as_posix(),
            "sha256": V7_CONTRACT_SHA256,
            "semantic_authority": "DEFINE_LIGHTGEN_2D_TRANSFER_AWARE_SEMANTIC_WARMUP_CONTRACT_V7",
        },
        v7,
        parent,
    )


def _image_template_support(templates: torch.Tensor) -> torch.Tensor:
    _require(
        torch.is_tensor(templates)
        and tuple(templates.shape) == (10, 1, 28, 28)
        and templates.dtype == torch.float32,
        "historical template shape/dtype changed",
    )
    support = F.adaptive_avg_pool2d(templates.clamp(0.0, 1.0), (28, 28)).to(torch.float64)
    _require(bool(torch.isfinite(support).all()), "historical templates are nonfinite")
    return support


def _s_tau(templates: torch.Tensor) -> tuple[float, dict[str, Any]]:
    support = _image_template_support(templates)
    distances: list[torch.Tensor] = []
    for left in range(10):
        for right in range(left + 1, 10):
            distances.append((support[left] - support[right]).square().mean())
    values = torch.stack(distances)
    _require(bool(torch.isfinite(values).all()), "s_tau pair distances are nonfinite")
    value = float(values.median().item())
    _require(value > 0.0, "s_tau must be finite and strictly positive")
    return value, {
        "pair_count": int(values.numel()),
        "min": float(values.min()),
        "median": value,
        "max": float(values.max()),
        "distance": "mean squared pixel distance after clamp [0,1] and adaptive_avg_pool2d to 28x28",
    }


def _adapt_authorities(v7_authorities: Mapping[str, Any]) -> dict[str, Any]:
    authorities = copy.deepcopy(dict(v7_authorities))
    authorities["v3_control"]["initialization_role"] = (
        "V8 initial encoder; exact V3 CONTROL step-300 only; never V5/V6/V7 final"
    )
    authorities["v4_generator"]["generator_phase"]["role"] = (
        "exact frozen V4 step-300 generator for V8"
    )
    authorities["v5_prototypes"]["source_role"] = (
        "TRAIN-only frozen prototypes for latent cyclic shift; reused without recomputation"
    )
    authorities["v5_prototypes"]["recompute_during_v8"] = False
    authorities["historical_class_direction"]["semantic_authority"] = (
        "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED"
    )
    authorities["historical_class_direction"]["template_role"] = (
        "historical TRAIN templates tau_0...tau_9 for frozen absolute image anchors"
    )
    authorities["smooth_evaluator"]["role"] = (
        "frozen evaluation-only SmoothMNISTEvaluator v2.1; no gradient and not sole acceptance evidence"
    )
    return authorities


def build_contract(root: Path) -> dict[str, Any]:
    parent, v7, v7_payload = _validate_v7_parent(root)
    v6 = v7._load_v6_definition()
    base_authorities = v6._validate_authorities(root)
    authorities = _adapt_authorities(base_authorities)
    historical_ref = authorities["historical_class_direction"]["raw_templates"]
    historical_payload = safe_torch_load(
        _path(root, historical_ref["path"]),
        expected_sha256=str(historical_ref["sha256"]),
        map_location="cpu",
    )
    _require(isinstance(historical_payload, Mapping), "historical template artifact is not a mapping")
    templates = historical_payload.get("templates")
    s_tau, s_tau_summary = _s_tau(templates)
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
        "git": v7_payload["git"],
        "definition_entrypoint": {
            "path": _relative(root, definition_path),
            "sha256": file_sha256(definition_path),
        },
        "parent_contract": parent,
        "authorities": authorities,
        "objective": {
            "question": (
                "Can an absolute semantic image anchor plus the known latent transfer objective "
                "improve natural real-path semantic reconstruction?"
            ),
            "training_objective": "L_V8=L_anchor+lambda_transfer*L_transfer",
            "anchor": {
                "logits": "b_k(r)=-d_image(r,tau_k)/(s_tau+1e-12)",
                "loss": "cross_entropy(b(r), y)",
                "output": "natural raw CMOS output r=G(E(x))",
                "representation": "historical TRAIN templates tau_0...tau_9",
            },
            "transfer": {
                "loss": "mean(softplus(-Delta_template))",
                "delta_template": "M_shift-M_base",
                "margin": "M_template=d(raw,tau_y)-d(raw,tau_t)",
                "latent_shift": "z_shift=z-mu_y+mu_t",
                "target_rule": "t=(y+1) mod 10",
            },
            "image_distance": "mean squared pixel distance after clamp [0,1] and adaptive_avg_pool2d to 28x28",
            "s_tau": s_tau,
            "s_tau_summary": s_tau_summary,
            "lambda_rule": {
                "formula": "lambda_transfer=g_anchor/(g_transfer+1e-12)",
                "g_anchor": "L_anchor gradient norm on encoder.phase",
                "g_transfer": "L_transfer gradient norm on encoder.phase",
                "state": "exact initial V3 encoder and exact frozen first TRAIN batch",
                "recompute": False,
                "clipping_or_tuning": False,
            },
            "gradient_contributions": {
                "image_reconstruction": "ZERO; evaluation-only",
                "prototype_cross_entropy": "ZERO; prototypes are shift support only",
                "Smooth_evaluator": "ZERO; evaluation-only",
            },
        },
        "model_assumption": (
            "A frozen absolute image-space semantic anchor may align natural optical outputs "
            "with class identity while the retained transfer term preserves the known cyclic direction."
        ),
        "design_decisions": [
            "Use exact V3 CONTROL step-300 encoder as the only initial encoder.",
            "Use exact V4 step-300 generator fully frozen.",
            "Reuse exact frozen V5 TRAIN-only prototypes and historical TRAIN templates.",
            "Compute s_tau once before training from all 45 unordered template pairs and freeze it.",
            "Compute lambda_transfer once from the initial state and first deterministic TRAIN batch; never retune.",
            "Train only encoder.phase; use step-0 as the matched baseline.",
            "Use natural anchor plus cyclic transfer; do not use a second arm, sweep, or post-hoc gate.",
        ],
        "protocol": {
            "trainable": ["encoder.phase"],
            "frozen": [
                "generator.phase",
                "OLS",
                "T=I_100",
                "W_OLS",
                "V5 TRAIN prototypes",
                "historical TRAIN templates",
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
                "known_transfer_pass": "template pass and Smooth pass and broad positive pair coverage",
            },
            "descriptive_anchor": {
                "metric": "natural raw output nearest-template macro recall",
                "role": "descriptive objective diagnostic; not an additional post-hoc gate",
            },
            "secondary": [
                "TRAIN L_anchor/L_transfer/L_total at step 0/100/200/300",
                "TRAIN/DEV image reconstruction loss",
                "OLS prototype margin",
                "per-class Smooth recall",
                "template-anchor per-class recall and correct-template margin",
                "DEV collapse diagnostics at step 0/300",
                "ten cyclic-pair summaries",
                "initial and per-step encoder gradient diagnostics",
            ],
            "raw_validity": "finite, expected shape, structured, nonconstant at step 0/100/200/300 and cyclic base/shift",
            "stage_a_firewall": "R300 >= 0.937, independently reported and does not authorize Stage B",
        },
        "data_roles": {
            "TRAIN": {"rows": 55000, "used_for": "optimization and initial gradient rule"},
            "DEV": {"rows": 5000, "used_for": "frozen evaluation and balanced cyclic selection"},
            "protected": 0,
            "official_test": 0,
            "sealed": 0,
            "labels_as_model_input": False,
        },
        "engineering_preflight": {
            "required": [
                "resolve exact V3/V4/V5/historical/Smooth/runtime authorities",
                "verify exact s_tau from 45 unordered historical template pairs",
                "verify encoder.phase sole trainable parameter and generator/OLS firewall",
                "prove image, prototype, Smooth, and frozen/generator gradients are absent",
                "prove finite nonzero g_anchor and g_transfer and freeze lambda_transfer",
                "three-step CUDA smoke through the real entrypoint",
                "checkpoint/reload/resume identity with objective constants",
                "exact-core synthetic anchor/z-shift/raw/transfer/combined/bootstrap/sufficient/report/lineage reload",
                "negative-control gradient diagnostics",
            ],
            "scientific_meaning": "none for synthetic path",
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
    _require(target.is_relative_to(outputs), "V8 contract output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite V8 contract output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    payload = build_contract(root)
    path = target / CONTRACT_NAME
    atomic_write_new_json(path, payload, artifact_kind="V8 absolute-anchor semantic-warmup contract")
    result = {
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "outer_gate": payload["outer_gate"],
        "contract": {"path": _relative(root, path), "sha256": file_sha256(path)},
        "s_tau": payload["objective"]["s_tau"],
        "scientific_training_executed": False,
        "training_authorized": False,
        "stage_b_authorized": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
