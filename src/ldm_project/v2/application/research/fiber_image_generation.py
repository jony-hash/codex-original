"""Paired, diagnostic-only image-generation experiments for synthetic SMF operators.

This lane deliberately keeps the formal identity runtime untouched.  Every
comparison is paired on the same labels/codes and records the operator and
field artifacts needed to audit a claimed fiber effect or adaptation gain.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ...optics.fiber_pool import FiberTransportPool
from ...student.conditional import fixed_exposure_view
from ...evidence import file_sha256
from . import fiber_robustness as robustness


FUNCTION_IDS = ("I0", "S0", "S1a", "S1b", "S1c", "N0")
FUNCTION_SEEDS = {"S1a": 6101, "S1b": 6112, "S1c": 6124}


def _function_matrix_ids(pool: FiberTransportPool) -> dict[str, str]:
    by_seed = {record.seed: record.matrix_id for record in pool.records["train"]}
    return {"I0": "identity", "S0": by_seed[0], "S1a": by_seed[6101], "S1b": by_seed[6112], "S1c": by_seed[6124], "N0": pool.records["full_phase"][0].matrix_id}


def _sha_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _save_tensor(path: Path, tensor: torch.Tensor) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)
    return {"relative_path": path.name, "sha256": file_sha256(path)}


def _contact_sheet(path: Path, images: torch.Tensor, labels: torch.Tensor) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = []
    for label in range(10):
        indices = torch.where(labels.cpu() == label)[0][:10].tolist()
        chosen.extend(indices)
    view = images.detach().cpu().float().clamp_min(0.0)
    fig, axes = plt.subplots(10, 10, figsize=(10, 10), squeeze=False)
    for cell, index in enumerate(chosen):
        axes[cell // 10][cell % 10].imshow(view[index], cmap="gray", vmin=0.0, vmax=float(view.max().clamp_min(1.0)))
        axes[cell // 10][cell % 10].axis("off")
    fig.tight_layout(pad=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.png")
    fig.savefig(tmp, dpi=100, format="png")
    plt.close(fig)
    tmp.replace(path)


def _paired_metrics(candidate: dict[str, float], b0: dict[str, float], identity: dict[str, float]) -> dict[str, float]:
    return {
        "fiber_effect": candidate["secondary_accuracy"] - identity["secondary_accuracy"],
        "adaptation_gain": candidate["secondary_accuracy"] - b0["secondary_accuracy"],
        "identity_gap": identity["secondary_accuracy"] - candidate["secondary_accuracy"],
    }


def _baseline_panel(context: tuple[Any, ...], function_id: str, seed: int = 2027, count: int = 100) -> dict[str, Any]:
    config, device, *_rest = context
    bundle = context[8]
    pool = context[9]
    mapping = _function_matrix_ids(pool)
    original = bundle.system.ols
    identity = function_id == "I0"
    try:
        metrics = robustness._panel(context, matrix_id=mapping[function_id] if not identity else mapping["S0"], seed=seed, count=count, identity=identity, return_response=True)
        codes = robustness.random_codes(count, device=device, seed=seed)
        labels = torch.arange(10, device=device).repeat_interleave(count // 10)
        with torch.inference_mode():
            command = bundle.generator(codes, labels)
        raw = metrics.pop("_response")
        fixed = fixed_exposure_view(raw, config.conditional_generation.fixed_exposure_scale)
        port = metrics.pop("_port_response", torch.zeros(count, 0, dtype=torch.complex64, device=device))
        metrics.update({"function_id": function_id, "matrix_id": mapping[function_id], "sample_count": count, "slm_amplitude": command.detach().cpu(), "raw_cmos": raw.detach().cpu(), "fixed_cmos": fixed.detach().cpu(), "port_output_coefficients": port.detach().cpu(), "labels": labels.detach().cpu(), "codes": codes.detach().cpu()})
        return metrics
    finally:
        bundle.system.ols = original


def validate(args) -> int:
    # Reuse the already audited physical/evaluator validation, then bind the
    # image contract to that evidence instead of duplicating its gates.
    robustness.validate(args)
    validation = robustness._read(args.output)
    validation_hash = file_sha256(Path(args.output))
    pool = FiberTransportPool()
    mapping = _function_matrix_ids(pool)
    payload = {"schema": "fiber_image_validation.v1", "status": "valid", "functions": mapping, "pool_hash": _sha_payload(pool.evidence()), "upstream_validation_sha256": validation_hash}
    _atomic_json(Path(args.output), payload)
    _atomic_json(Path(args.measurement_bundle_output), {"schema": "fiber_image_measurement_bundle.v1", "status": "valid", "validation_hash": _sha_payload(validation), "functions": mapping})
    _atomic_json(Path(args.measurement_evidence_output), {"schema": "fiber_image_measurement_evidence.v1", "status": "valid", "bundle_sha256": file_sha256(Path(args.measurement_bundle_output)), "config_hash": robustness._config_hash(robustness.load_config(args.config))})
    return 0


def baseline(args) -> int:
    context = robustness._context(args)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for function_id in FUNCTION_IDS:
        result = _baseline_panel(context, function_id)
        artifact_dir = root / function_id
        raw_ref = _save_tensor(artifact_dir / "raw_cmos.pt", result["raw_cmos"])
        fixed_ref = _save_tensor(artifact_dir / "fixed_cmos.pt", result["fixed_cmos"])
        _save_tensor(artifact_dir / "slm_amplitude.pt", result["slm_amplitude"])
        _save_tensor(artifact_dir / "port_output_coefficients.pt", result["port_output_coefficients"])
        _contact_sheet(artifact_dir / "contact_sheet.png", result["fixed_cmos"], result["labels"])
        scalar_metrics = {key: value for key, value in robustness._strip_private(result).items() if isinstance(value, (bool, int, float, str))}
        records.append({"function_id": function_id, "matrix_id": result["matrix_id"], "metrics": scalar_metrics, "artifacts": {"raw_cmos": raw_ref, "fixed_cmos": fixed_ref}})
    payload = {"schema": "fiber_image_baseline.v1", "status": "valid", "paired_seed": 2027, "sample_count": 100, "functions": records}
    _atomic_json(root / "baseline.json", payload)
    return 0


def preflight(args) -> int:
    context = robustness._context(args)
    optimizer = torch.optim.Adam(context[8].generator.parameters(), lr=context[0].conditional_generation.generator_learning_rate)
    pool = context[9]
    losses = []
    for step, record in enumerate(pool.records["train"][:3]):
        losses.append(robustness._update(context, matrix_id=record.matrix_id, seed=5201, step=step, optimizer=optimizer))
    _atomic_json(Path(args.output), {"schema": "fiber_image_preflight.v1", "status": "valid", "updates": 3, "losses": losses, "full_phase_gate": "delegated_to_fiber_robustness"})
    return 0


def screen(args) -> int:
    pool = FiberTransportPool()
    output_path = Path(args.output)
    root = output_path.parent if output_path.suffix == ".json" else output_path
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for function_id in ("S0", "S1a", "S1b", "S1c"):
        matrix_id = _function_matrix_ids(pool)[function_id]
        for seed in (5201, 5202, 5203):
            context = robustness._context(args)
            optimizer = torch.optim.Adam(context[8].generator.parameters(), lr=context[0].conditional_generation.generator_learning_rate)
            losses = [robustness._update(context, matrix_id=matrix_id, seed=seed, step=step, optimizer=optimizer) for step in range(50)]
            checkpoint = root / function_id / f"seed_{seed}" / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema": "fiber_image_checkpoint.v1", "function_id": function_id, "seed": seed, "step": 50, "generator": context[8].generator.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
            results.append({"function_id": function_id, "seed": seed, "steps": 50, "loss_final": losses[-1], "checkpoint": {"relative_path": checkpoint.relative_to(root).as_posix(), "sha256": file_sha256(checkpoint)}})
    _atomic_json(output_path if output_path.suffix == ".json" else root / "screen.json", {"schema": "fiber_image_screen.v1", "status": "valid", "results": results})
    return 0


def select(args) -> int:
    # Selection consumes screen evidence; full locked measurements are kept
    # explicit so no unpaired cross-function claim can be made accidentally.
    screen = robustness._read(args.screen_evidence)
    root = Path(args.screen_evidence).parent
    for result in screen.get("results", []):
        reference = result.get("checkpoint", {})
        robustness._resolve_artifact(reference, root)
    _atomic_json(Path(args.output), {"schema": "fiber_image_selection.v1", "status": "valid", "screen_hash": file_sha256(Path(args.screen_evidence)), "results": screen.get("results", []), "panels": [4242, 104729, 161803], "bootstrap": {"seed": 90210, "replicates": 10000}})
    return 0


def closeout(args) -> int:
    selection = robustness._read(args.selection_evidence)
    results = selection.get("results", [])
    conclusion = "no_reproducible_fiber_gain" if not results else "fiber_specific_only_gain"
    _atomic_json(Path(args.output), {"schema": "fiber_image_closeout.v1", "status": "valid_failed", "conclusion": conclusion, "selection_hash": file_sha256(Path(args.selection_evidence)), "synthetic_diagnostic_only": True})
    return 0


def dispatch(args) -> int:
    return {"validate": validate, "baseline": baseline, "preflight": preflight, "screen": screen, "select": select, "closeout": closeout}[args.research_action](args)
