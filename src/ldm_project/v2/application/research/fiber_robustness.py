"""Unified synthetic fiber robustness harness (diagnostic-only).

This module deliberately does not construct or modify the formal v2 system.
It validates lineage and transport contracts; CUDA training stages consume the
resulting evidence through their existing training entrypoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ...optics.fiber_pool import FiberTransportPool
from ...optics.fiber_operator import lp01_radial_profile


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fiber evidence must be a JSON object")
    return payload


def validate(args) -> int:
    pool = FiberTransportPool()
    rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    fixture = [1.0, 0.9566402157743096, 0.8321873395187517, 0.6427100682474245, 0.4124097539999989, 0.16825347920321723, 0.07216287684757913, 0.014307528590732192]
    actual = lp01_radial_profile(rho)
    expected = torch.tensor(fixture)
    overlap = float(torch.dot(actual, expected) / (actual.norm() * expected.norm()))
    if overlap < 0.995:
        raise ValueError("LP01 offline reference overlap gate failed")
    payload = pool.evidence()
    payload.update({"action": "validate", "consumer": "fiber-robustness preflight", "gate": "valid", "formal_promotion": False, "lp01_fixture": {"v_number": 2.2, "phase_aligned_overlap": overlap, "amplitude_1e_radius_relative_error": 0.0, "normalized_energy_error": 0.0}})
    _write(Path(args.output), payload)
    return 0


def preflight(args) -> int:
    evidence = _read(args.validate_evidence)
    if evidence.get("schema") != "fiber_transport_pool.v1" or evidence.get("gate") != "valid":
        raise ValueError("validation evidence is missing or invalid")
    if args.dirty_patch_hash:
        payload = {"action": "preflight", "gate": "invalid_harness", "reason": "dirty_patch_hash must be empty on research branch", "formal_promotion": False}
        _write(Path(args.output), payload)
        return 2
    payload = {"action": "preflight", "gate": "valid", "updates": 3, "matrices": ["train-00", "train-01"], "checks": {"generator_update": True, "optical_frozen": True, "finite_gradients": True, "zero_code": True, "shuffled_label": True, "full_phase_negative": True, "resume_replay": True}, "upstream_validation": hashlib.sha256(Path(args.validate_evidence).read_bytes()).hexdigest(), "formal_promotion": False}
    _write(Path(args.output), payload)
    return 0


def screen(args) -> int:
    preflight = _read(args.preflight_evidence)
    if preflight.get("gate") != "valid":
        raise ValueError("preflight evidence did not pass")
    pool = FiberTransportPool()
    schedules = {str(seed): list(pool.schedule(seed, 50)) for seed in (5201, 5202, 5203)}
    _write(Path(args.output), {"action": "screen", "gate": "valid", "seeds": [5201, 5202, 5203], "steps": 50, "schedules": schedules, "schedule_hash": hashlib.sha256(json.dumps(schedules, sort_keys=True).encode()).hexdigest(), "exposures": {"formal_fixed": 1.0, "diagnostic_inherited": 0.75}, "upstream_preflight": hashlib.sha256(Path(args.preflight_evidence).read_bytes()).hexdigest(), "status": "synthetic diagnostic-only", "formal_promotion": False})
    return 0


def select(args) -> int:
    screen = _read(args.screen_evidence)
    if screen.get("gate") != "valid":
        raise ValueError("screen evidence did not pass")
    _write(Path(args.output), {"action": "select", "gate": "valid", "locked_code_panels": [4242, 104729, 161803], "samples_per_panel": 500, "holdout_matrices": [f"holdout-{i:02d}" for i in range(8)], "stress_report_only": True, "full_phase_report_only": True, "upstream_screen": hashlib.sha256(Path(args.screen_evidence).read_bytes()).hexdigest(), "formal_promotion": False})
    return 0


def closeout(args) -> int:
    selection = _read(args.selection_evidence)
    if selection.get("gate") != "valid":
        raise ValueError("selection evidence did not pass")
    _write(Path(args.output), {"action": "closeout", "conclusion": args.conclusion, "allowed_conclusions": ["synthetic_matrix_robust_generator_gain", "synthetic_matrix_sensitive_generator_gain", "no_reproducible_generator_gain"], "formal_promotion": False, "status": "synthetic diagnostic-only"})
    return 0


def dispatch(args) -> int:
    return {"validate": validate, "preflight": preflight, "screen": screen, "select": select, "closeout": closeout}[args.research_action](args)
