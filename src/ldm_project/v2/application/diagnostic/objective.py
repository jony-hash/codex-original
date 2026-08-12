"""Pre-registered, short objective-alignment diagnostics.

These commands are diagnostic-only.  They use the training evaluator and a
fixed validation panel; no result from this module can authorize formal
training or acceptance.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ...evidence import GateResult, atomic_write_json, canonical_sha256, file_sha256
from ..current.runtime import _new_run, _require_identity


OBJECTIVE_DIAGNOSTIC_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_objective_alignment_v1",
        "candidates": ["A", "B", "C"],
        "screen_steps": 100,
        "confirm_steps": 400,
        "seeds": [1234, 1235, 1236],
        "panel": "17_samples_per_class",
        "weight_rule": "0.25*base_gradient_norm/auxiliary_gradient_norm",
        "camera_target": 0.30,
        "camera_temperature": 0.02,
        "coverage_target": 1.35,
        "coverage_temperature": 0.10,
    }
)


def calibrate_weights(base_gradient_norm: float, auxiliary_gradient_norm: float) -> float:
    if auxiliary_gradient_norm <= 1.0e-12 or not torch.isfinite(torch.tensor([base_gradient_norm, auxiliary_gradient_norm])).all():
        raise ValueError("objective gradient calibration encountered a zero or non-finite norm")
    weight = 0.25 * base_gradient_norm / auxiliary_gradient_norm
    if not 1.0e-3 <= weight <= 10.0:
        raise ValueError("objective gradient calibration produced a weight outside [1e-3,10]")
    return float(weight)


def _write_diagnostic(config, *, name: str, input_hashes: dict[str, str], gates: tuple[GateResult, ...], payload: dict[str, object]) -> int:
    run = _new_run(config)
    output = {"schema_version": 1, "contract": OBJECTIVE_DIAGNOSTIC_CONTRACT, "kind": name, "status": "passed" if all(gate.passed for gate in gates) else "failed", "input_hashes": input_hashes, "gates": [asdict(gate) for gate in gates], "payload": payload, "formal_promotion_authorized": False}
    atomic_write_json(run.root / f"{name}_evidence.json", output)
    print(run.root)
    return 0 if all(gate.passed for gate in gates) else 1


def conditional_objective_weight_calibration(args, config) -> int:
    _require_identity(config)
    base = float(args.base_gradient_norm)
    auxiliary = [float(value) for value in args.auxiliary_gradient_norm]
    weights = [calibrate_weights(base, value) for value in auxiliary]
    run = _new_run(config)
    artifact = run.root / "conditional_objective_weight_calibration.json"
    atomic_write_json(artifact, {"schema_version": 1, "contract": OBJECTIVE_DIAGNOSTIC_CONTRACT, "base_gradient_norm": base, "auxiliary_gradient_norm": auxiliary, "weights": weights, "frozen": True})
    print(run.root)
    return 0


def _parse_metric(payload: object, key: str) -> float:
    if not isinstance(payload, dict) or not isinstance(payload.get(key), (int, float)):
        raise ValueError(f"objective diagnostic payload lacks finite {key}")
    value = float(payload[key])
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"objective diagnostic {key} is non-finite")
    return value


def objective_alignment_screen(args, config) -> int:
    """Compare seed-matched A/B/C measurement payloads without selecting a formal objective."""
    _require_identity(config)
    baseline = __import__("json").loads(Path(args.baseline_measurement).read_text(encoding="utf-8"))
    candidates = {name: __import__("json").loads(Path(path).read_text(encoding="utf-8")) for name, path in zip(("A", "B", "C"), args.candidate_measurement, strict=True)}
    base_camera = _parse_metric(baseline, "camera_code_fraction")
    base_diversity = _parse_metric(baseline, "minimum_diversity_ratio")
    base_coverage = _parse_metric(baseline, "coverage_ratio")
    gates = []
    scores = {}
    for name, payload in candidates.items():
        camera = _parse_metric(payload, "camera_code_fraction")
        diversity = _parse_metric(payload, "minimum_diversity_ratio")
        coverage = _parse_metric(payload, "coverage_ratio")
        score = min((camera - base_camera) / 0.02, (diversity - base_diversity) / 0.05, (base_coverage - coverage) / 0.05)
        scores[name] = {"camera_code_fraction": camera, "minimum_diversity_ratio": diversity, "coverage_ratio": coverage, "normalized_minimum_improvement": score}
        if name != "A":
            gates.append(GateResult(f"{name}_screen_gate", camera >= base_camera + 0.02 and diversity >= base_diversity + 0.05 and coverage <= base_coverage - 0.05, scores[name]))
    gates.append(GateResult("finite_payload", True, {"candidates": 3.0}))
    return _write_diagnostic(config, name="conditional_objective_alignment_screen", input_hashes={"baseline": file_sha256(args.baseline_measurement), **{f"candidate_{name}": file_sha256(path) for name, path in zip(("A", "B", "C"), args.candidate_measurement, strict=True)}}, gates=tuple(gates), payload={"scores": scores, "selection": "diagnostic_only"})


def objective_alignment_confirm(args, config) -> int:
    _require_identity(config)
    import json

    payload = json.loads(Path(args.confirmation_measurement).read_text(encoding="utf-8"))
    rows = payload.get("seeds")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("objective confirmation requires exactly three seed rows")
    gates = (GateResult("three_seed_rows", True, {"count": 3.0}),)
    return _write_diagnostic(config, name="conditional_objective_alignment_confirm", input_hashes={"confirmation": file_sha256(args.confirmation_measurement)}, gates=gates, payload={"candidate": args.candidate, "seeds": rows, "formal_promotion_authorized": False})
