"""CLI consumer for the generic, compute-aware experiment contract."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ...evidence import atomic_write_json, canonical_sha256
from ...experiment_contract import (
    ExperimentContract,
    load_experiment_contract,
    validate_preflight_report,
)


def _root(args: argparse.Namespace, action: str) -> Path:
    target = Path(args.run_root).resolve()
    target.mkdir(parents=True, exist_ok=True)
    root = target / f"experiment_{action}"
    root.mkdir(exist_ok=False)
    return root


def _provenance() -> tuple[str, str | None]:
    commit = subprocess.run(("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(("git", "status", "--porcelain", "--untracked-files=all"), check=True, capture_output=True, text=True).stdout
    return commit, canonical_sha256(status) if status else None


def _write(root: Path, contract: ExperimentContract, action: str, state: str, *, metrics: dict, permitted: list[str], reason: str | None = None) -> Path:
    commit, dirty_patch_hash = _provenance()
    payload = {
        "schema_version": 1,
        "evidence_kind": "experiment_contract",
        "action": action,
        "state": state,
        "contract_hash": contract.digest,
        "git_commit": commit,
        "dirty_patch_hash": dirty_patch_hash,
        "contract": contract.to_dict(),
        "metrics": metrics,
        "reason": reason,
        "formal_promotion_authorized": False,
        "permitted_next_actions": permitted,
    }
    payload["content_hash"] = canonical_sha256(payload)
    path = root / f"experiment_{action}_evidence.json"
    atomic_write_json(path, payload)
    print(path)
    return path


def validate(args: argparse.Namespace) -> int:
    contract = load_experiment_contract(args.contract)
    root = _root(args, "validate")
    _write(root, contract, "validate", "validated", metrics={"level": contract.level}, permitted=["preflight"])
    return 0


def _read_prior(path: Path, contract: ExperimentContract, *, state: str, action: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    supplied = payload.pop("content_hash", None)
    if supplied != canonical_sha256(payload):
        raise ValueError("experiment evidence content hash mismatch")
    if payload.get("contract_hash") != contract.digest or payload.get("state") != state or payload.get("action") != action:
        raise ValueError("experiment evidence cannot authorize this action")
    return payload


def preflight(args: argparse.Namespace) -> int:
    contract = load_experiment_contract(args.contract)
    prior = Path(args.prior_evidence)
    if not prior.is_absolute() or not prior.is_file():
        raise ValueError("preflight requires an absolute validate evidence path")
    prior_payload = _read_prior(prior, contract, state="validated", action="validate")
    if prior_payload.get("permitted_next_actions") != ["preflight"]:
        raise ValueError("validate evidence cannot authorize this preflight")
    report_path = Path(args.report)
    if not report_path.is_absolute() or not report_path.is_file():
        raise ValueError("preflight report must be an existing absolute path")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = validate_preflight_report(contract, report, elapsed_seconds=float(args.elapsed_seconds))
    root = _root(args, "preflight")
    _write(root, contract, "preflight", "preflight_passed", metrics=metrics | {"report_hash": canonical_sha256(report)}, permitted=["running"])
    return 0


def dispatch(args: argparse.Namespace) -> int:
    return {"validate": validate, "preflight": preflight}[args.research_action](args)
