"""No-update synthetic loss/exposure attribution for the fiber interface.

This lifecycle is deliberately separate from the modal-interface lifecycle. It
tests whether the measured S0 energy loss can be compensated by one global
detector exposure selected from an independent calibration panel. No optimizer
is created and no optical parameter is updated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ...evidence import atomic_torch_save, atomic_write_json, file_sha256, safe_torch_load
from ...student.conditional import fixed_exposure_view
from .._runtime_shared import random_codes
from . import fiber_interface_attribution as interface

CALIBRATION_PANEL = 271828
PANELS = interface.PANELS
SAMPLES_PER_PANEL = 500
SAMPLES_PER_LABEL = 50
BOOTSTRAP_REPLICATES = 10_000
STORAGE_LIMIT_BYTES = 2 * 1024**3
GAIN_GRID = (0.75, 1.0, 1.15, 1.50, 1.75, 2.0)
BASE_VARIANTS = (
    "I0-E0",
    "S0-E0",
    "S0-Ecal",
    "I0-Acal",
    "S0-field-gain-reference",
    "S0-per-sample-oracle",
)


def _write(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _require_file(args: Any, name: str) -> Path:
    value = getattr(args, name, None)
    if not value:
        raise ValueError(f"fiber loss/exposure attribution requires --{name.replace('_', '-')}")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"fiber loss/exposure attribution input does not exist: {path}")
    return path


def _git() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _root(output: str | Path, *, require_new: bool = False) -> Path:
    path = Path(output)
    root = path.parent if path.suffix == ".json" else path
    if require_new and root.exists() and any(root.iterdir()):
        raise ValueError(f"loss/exposure attribution output must be a new directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tensor_hash(value: torch.Tensor) -> str:
    return interface._tensor_hash(value)


def _ref(path: Path, root: Path, value: torch.Tensor | None = None) -> dict[str, str]:
    reference = {
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": file_sha256(path),
    }
    if value is not None:
        reference["tensor_sha256"] = _tensor_hash(value)
    return reference


def _resolve(reference: dict[str, Any], root: Path) -> Path:
    relative = reference.get("relative_path")
    expected = reference.get("sha256")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected, str)
    ):
        raise ValueError("invalid loss/exposure artifact reference")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file() or file_sha256(path) != expected:
        raise ValueError("missing or hash-mismatched loss/exposure artifact")
    return path


def _parameter_state(bundle: Any) -> dict[str, str]:
    values = {
        name: _tensor_hash(parameter.detach().cpu())
        for name, parameter in bundle.named_parameters()
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    values["__state_hash__"] = hashlib.sha256(encoded).hexdigest()
    return values


def _labels_codes(panel: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(10, device=device).repeat_interleave(SAMPLES_PER_LABEL)
    codes = random_codes(SAMPLES_PER_PANEL, device=device, seed=panel)
    return labels, codes


def _variant_names(gain: float) -> tuple[str, ...]:
    del gain
    grid = tuple(f"S0-gain-{int(round(value * 100)):03d}" for value in GAIN_GRID)
    return BASE_VARIANTS + grid


def _lineage_namespace(args: Any, root: Path) -> SimpleNamespace:
    values = vars(args).copy()
    values["measurement_bundle"] = str(root / "fiber_image_measurement_bundle.json")
    values["measurement_evidence"] = str(root / "fiber_image_measurement_evidence.json")
    values["validate_evidence"] = str(root / "fiber_image_validation.json")
    values["output"] = str(root / "fiber_image_validation.json")
    return SimpleNamespace(**values)


def _context(args: Any, root: Path) -> tuple[Any, ...]:
    return interface._context(args, root)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate_interface_inputs(args: Any) -> dict[str, Any]:
    validation_path = _require_file(args, "interface_validation")
    measurement_path = _require_file(args, "interface_measurement")
    closeout_path = _require_file(args, "interface_closeout")
    validation = _load_json(validation_path)
    if validation.get("schema") != "fiber_interface_validation.v1" or validation.get("gate") != "valid":
        raise ValueError("bound interface validation is not valid")
    if not all(validation.get("harness_gates", {}).values()):
        raise ValueError("bound interface validation contains a failed harness gate")
    _, measurement = interface._load_measurement(measurement_path, require_current=False)
    closeout = _load_json(closeout_path)
    if closeout.get("schema") != "fiber_interface_closeout.v1" or closeout.get("gate") != "valid":
        raise ValueError("bound interface closeout is not valid")
    if closeout.get("validation_sha256") != file_sha256(validation_path):
        raise ValueError("interface closeout does not bind interface validation")
    if closeout.get("measurement_sha256") != file_sha256(measurement_path):
        raise ValueError("interface closeout does not bind interface measurement")
    if measurement.get("git_commit") != validation.get("git_commit"):
        raise ValueError("interface validation and measurement commits differ")
    return {
        "validation": validation,
        "measurement": measurement,
        "closeout": closeout,
        "validation_sha256": file_sha256(validation_path),
        "measurement_sha256": file_sha256(measurement_path),
        "closeout_sha256": file_sha256(closeout_path),
    }


def _source_state_audit(args: Any, bundle: Any) -> dict[str, Any]:
    checkpoint_path = _require_file(args, "research_initialization_checkpoint")
    payload = safe_torch_load(
        checkpoint_path,
        expected_sha256=file_sha256(checkpoint_path),
        map_location="cpu",
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("research initialization checkpoint state_dict is missing")
    state = payload["state_dict"]
    source_keys = sorted(state)
    loaded = sorted(key for key in source_keys if key.startswith("generator."))
    ignored = sorted(key for key in source_keys if not key.startswith("generator."))
    generator_keys = sorted(f"generator.{key}" for key in bundle.generator.state_dict())
    if loaded != generator_keys:
        raise ValueError("checkpoint generator loading scope does not match runtime generator")
    source_encoder = state.get("system.encoder.phase")
    source_decoder = state.get("system.decoder.phase")
    if not isinstance(source_encoder, torch.Tensor) or not isinstance(source_decoder, torch.Tensor):
        raise ValueError("source checkpoint phase audit keys are missing")
    return {
        "source_state_key_count": len(source_keys),
        "loaded_generator_state_keys": loaded,
        "ignored_state_keys": ignored,
        "source_encoder_phase_norm": float(source_encoder.norm()),
        "source_decoder_phase_norm": float(source_decoder.norm()),
        "source_encoder_phase_max_abs": float(source_encoder.abs().max()),
        "source_decoder_phase_max_abs": float(source_decoder.abs().max()),
        "runtime_encoder_phase_zero": bool(torch.equal(bundle.system.encoder.phase, torch.zeros_like(bundle.system.encoder.phase))),
        "runtime_decoder_phase_zero": bool(torch.equal(bundle.system.decoder.phase, torch.zeros_like(bundle.system.decoder.phase))),
        "runtime_generator_trainable": [name for name, p in bundle.generator.named_parameters() if p.requires_grad],
        "runtime_optical_trainable": [name for name, p in bundle.system.named_parameters() if p.requires_grad],
    }


def _validation_payload(args: Any, root: Path) -> dict[str, Any]:
    bound = _validate_interface_inputs(args)
    upstream = interface._validate_upstream(args, root)
    historical = interface._validate_historical_inputs(args, upstream)
    context = _context(args, root)
    config, device, _, _, _, _, _, _, bundle, _, _, _ = context
    settings = config.conditional_generation
    if settings is None:
        raise ValueError("conditional generation settings are required")
    audit = _source_state_audit(args, bundle)
    interface_gates = bound["validation"].get("harness_gates", {})
    harness = {
        "device_cuda": device.type == "cuda",
        "interface_lineage": bool(interface_gates) and all(interface_gates.values()),
        "calibration_panel_disjoint": CALIBRATION_PANEL not in PANELS,
        "source_generator_scope": bool(audit["loaded_generator_state_keys"]),
        "source_system_keys_ignored": bool(audit["ignored_state_keys"]),
        "runtime_zero_phase": audit["runtime_encoder_phase_zero"] and audit["runtime_decoder_phase_zero"],
        "no_optical_trainable_parameters": not audit["runtime_optical_trainable"],
        "fixed_exposure_positive": settings.fixed_exposure_scale > 0.0,
        "no_optimizer": True,
        "input_mode_rank": interface_gates.get("input_mode_rank") is True,
        "output_mode_rank": interface_gates.get("output_mode_rank") is True,
        "plane_provenance": interface_gates.get("plane_provenance") is True,
    }
    return {
        "schema": "fiber_loss_exposure_validation.v1",
        "gate": "valid" if all(harness.values()) else "invalid_harness",
        "git_commit": _git(),
        "config_hash": bound["validation"].get("config_hash"),
        "dataset_split_manifest_hash": bound["validation"].get("dataset_split_manifest_hash"),
        "primary_evaluator_sha256": bound["validation"].get("primary_evaluator_sha256"),
        "secondary_evaluator_sha256": bound["validation"].get("secondary_evaluator_sha256"),
        "research_initialization_checkpoint_sha256": historical["research_initialization_checkpoint_sha256"],
        "source_training_evidence_sha256": historical["source_training_evidence_sha256"],
        "v3_closeout_sha256": historical["v3_closeout_sha256"],
        "attribution_measurement_sha256": historical["attribution_measurement_sha256"],
        "lp01_fixture_sha256": historical["lp01_fixture_sha256"],
        "interface_validation_sha256": bound["validation_sha256"],
        "interface_measurement_sha256": bound["measurement_sha256"],
        "interface_closeout_sha256": bound["closeout_sha256"],
        "mode_bank_spec_sha256": bound["validation"].get("mode_bank_spec_sha256"),
        "upstream_validation": upstream,
        "source_state_audit": audit,
        "panels": list(PANELS),
        "calibration_panel": CALIBRATION_PANEL,
        "sample_count_per_panel": SAMPLES_PER_PANEL,
        "variants": list(BASE_VARIANTS) + [f"S0-gain-{int(round(value * 100)):03d}" for value in GAIN_GRID],
        "fixed_exposure_scale": settings.fixed_exposure_scale,
        "harness_gates": harness,
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
    }


def validate(args: Any) -> int:
    output = Path(args.output)
    root = _root(output, require_new=True)
    if _dirty():
        _write(output, {"schema": "fiber_loss_exposure_validation.v1", "gate": "invalid_harness", "reason": "dirty worktree"})
        return 2
    try:
        payload = _validation_payload(args, root)
        _write(output, payload)
        return 0 if payload["gate"] == "valid" else 2
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        _write(output, {
            "schema": "fiber_loss_exposure_validation.v1",
            "gate": "invalid_harness",
            "reason": str(exc),
            "formal_promotion": False,
            "training_authorized": False,
            "synthetic_diagnostic_only": True,
        })
        return 2


def _energy(tensor: torch.Tensor, pixel_pitch: tuple[float, float]) -> torch.Tensor:
    area = pixel_pitch[0] * pixel_pitch[1]
    return tensor.abs().square().sum(dim=(-3, -2, -1)) * area


def _quantile_summary(value: torch.Tensor, *, maximum_samples: int = 1_000_000) -> tuple[float, float, float, int]:
    flattened = value.flatten()
    stride = max(1, (flattened.numel() + maximum_samples - 1) // maximum_samples)
    sampled = flattened[::stride]
    quantiles = torch.quantile(sampled, torch.tensor((0.01, 0.50, 0.99), device=sampled.device))
    return float(quantiles[0]), float(quantiles[1]), float(quantiles[2]), int(sampled.numel())


def _forward_raw(
    context: tuple[Any, ...],
    variant: str,
    labels: torch.Tensor,
    codes: torch.Tensor,
) -> dict[str, Any]:
    config, device, _, _, _, _, _, _, bundle, _, _, _ = context
    operator = interface._install_variant(bundle, "I0" if variant == "I0" else "S0", device)
    raw_parts: list[torch.Tensor] = []
    input_energy: list[torch.Tensor] = []
    output_energy: list[torch.Tensor] = []
    projection_fraction: list[torch.Tensor] = []
    residual_fraction: list[torch.Tensor] = []
    trace_parts: list[list[torch.Tensor]] = [[] for _ in interface.STAGES[:-1]]
    trace_metadata: list[dict[str, Any] | None] = [None for _ in interface.STAGES[:-1]]
    for start in range(0, SAMPLES_PER_PANEL, config.execution.evaluation_microbatch_size):
        code_batch = codes[start : start + config.execution.evaluation_microbatch_size]
        label_batch = labels[start : start + code_batch.shape[0]]
        with torch.inference_mode():
            command = bundle.generator(code_batch, label_batch)
            result = bundle.system.forward(bundle.system.command(command), return_optical_trace=True)
        raw_parts.append(result.raw_camera.tensor.detach().cpu())
        fields = result.optical_trace or ()
        if len(fields) != len(interface.STAGES) - 1:
            raise ValueError(f"unexpected optical trace length: {len(fields)}")
        for index, field in enumerate(fields):
            if start < 20:
                count = min(field.tensor.shape[0], 20 - start)
                trace_parts[index].append(field.tensor[:count].detach().cpu())
                if trace_metadata[index] is None:
                    trace_metadata[index] = {
                        "plane": asdict(field.plane),
                        "provenance": asdict(field.provenance),
                    }
        input_field = fields[3].tensor
        output_field = fields[4].tensor
        input_e = _energy(input_field, bundle.system.encoder_plane.pixel_pitch_m)
        output_e = _energy(output_field, bundle.system.encoder_plane.pixel_pitch_m)
        input_energy.append(input_e.cpu())
        output_energy.append(output_e.cpu())
        if variant == "S0":
            diagnostics = operator.last_diagnostics
            projection_fraction.append(diagnostics["projection_fraction"].cpu())
            residual_fraction.append(diagnostics["residual_fraction"].cpu())
    trace_payload = []
    for parts, metadata in zip(trace_parts, trace_metadata, strict=True):
        if not parts or metadata is None:
            raise ValueError("representative optical trace is incomplete")
        trace_payload.append({"tensor": torch.cat(parts), **metadata})
    return {
        "raw": torch.cat(raw_parts),
        "input_energy": torch.cat(input_energy),
        "output_energy": torch.cat(output_energy),
        "projection_fraction": torch.cat(projection_fraction) if projection_fraction else None,
        "residual_fraction": torch.cat(residual_fraction) if residual_fraction else None,
        "representative_trace": trace_payload,
    }


def _calibration_ratio(input_energy: torch.Tensor, projected_energy: torch.Tensor) -> tuple[float, float]:
    input_sum = input_energy.sum()
    projected_sum = projected_energy.sum()
    rho = projected_sum / input_sum.clamp_min(1.0e-30)
    if not torch.isfinite(rho) or not 0.0 < float(rho) <= 1.0 + 1.0e-5:
        raise ValueError("calibration energy ratio is not finite or passive")
    rho_value = float(rho)
    return rho_value, 1.0 / rho_value


def _view_metric(
    context: tuple[Any, ...],
    variant: str,
    panel: int,
    raw: torch.Tensor,
    exposure_scale: float,
    labels: torch.Tensor,
    codes: torch.Tensor,
    base: dict[str, Any],
    amplitude_scale: float | torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, _, _, _, reference_images, reference_labels, primary, secondary, _, _, _, _ = context
    settings = config.conditional_generation
    if settings is None:
        raise ValueError("conditional generation settings are required")
    view = fixed_exposure_view(raw, exposure_scale)
    metric, primary_prediction, secondary_prediction = interface._metric_summary(
        raw,
        view,
        labels,
        primary,
        secondary,
        reference_images,
        reference_labels,
        settings,
    )
    normalized = raw / exposure_scale
    normalized_p01, normalized_p50, normalized_p99, normalized_quantile_sample_count = _quantile_summary(normalized)
    metric.update({
        "variant": variant,
        "panel": panel,
        "codes_sha256": _tensor_hash(codes),
        "exposure_scale": exposure_scale,
        "amplitude_scale": "samplewise" if isinstance(amplitude_scale, torch.Tensor) else float(amplitude_scale),
        "saturation": float((raw > exposure_scale).float().mean()),
        "clipped_fraction": float((normalized >= 1.0).float().mean()),
        "normalized_raw_mean": float(normalized.mean()),
        "normalized_raw_p01": normalized_p01,
        "normalized_raw_p50": normalized_p50,
        "normalized_raw_p99": normalized_p99,
        "normalized_raw_quantile_sample_count": normalized_quantile_sample_count,
        "raw_finite": bool(torch.isfinite(raw).all()),
        "raw_nonnegative": bool((raw >= 0).all()),
        "passive_available": variant in {"S0-E0", "S0-Ecal", "I0-Acal"},
        "nonphysical_upper_bound": variant in {"S0-field-gain-reference", "S0-per-sample-oracle"},
        "energy_oracle": variant == "S0-per-sample-oracle",
        "labels_sha256": _tensor_hash(labels),
    })
    if base["projection_fraction"] is not None:
        metric["projection_fraction_mean"] = float(base["projection_fraction"].mean())
        metric["residual_fraction_mean"] = float(base["residual_fraction"].mean())
        metric["input_energy_mean"] = float(base["input_energy"].mean())
        metric["output_energy_mean"] = float((base["output_energy"] * torch.as_tensor(amplitude_scale).square()).mean())
        metric["energy_retention_mean"] = float(metric["output_energy_mean"] / max(metric["input_energy_mean"], 1.0e-30))
        metric["passive"] = bool(metric["energy_retention_mean"] <= 1.0 + 1.0e-5)
    else:
        scaled_output = base["output_energy"] * torch.as_tensor(amplitude_scale).square()
        metric["input_energy_mean"] = float(base["input_energy"].mean())
        metric["output_energy_mean"] = float(scaled_output.mean())
        metric["energy_retention_mean"] = float(scaled_output.mean() / base["input_energy"].mean().clamp_min(1.0e-30))
        metric["passive"] = bool(torch.all(scaled_output <= base["input_energy"] * (1.0 + 1.0e-5) + 1.0e-12))
    values = {
        "labels": labels.cpu(),
        "codes": codes,
        "fixed_cmos": view,
        "primary_prediction": primary_prediction,
        "secondary_prediction": secondary_prediction,
        "secondary_correctness": (secondary_prediction == labels.cpu()),
        "representative_trace": base["representative_trace"],
    }
    return metric, values


def _save_artifacts(root: Path, variant: str, panel: int, values: dict[str, Any]) -> dict[str, Any]:
    directory = root / "measurements" / variant / str(panel)
    directory.mkdir(parents=True, exist_ok=True)
    refs: dict[str, Any] = {}
    for name in (
        "labels",
        "codes",
        "fixed_cmos",
        "primary_prediction",
        "secondary_prediction",
        "secondary_correctness",
    ):
        path = directory / f"{name}.pt"
        atomic_torch_save(path, values[name].cpu())
        refs[name] = _ref(path, root, values[name].cpu())
    trace_path = directory / "representative_trace.pt"
    atomic_torch_save(trace_path, values["representative_trace"])
    refs["representative_trace"] = _ref(trace_path, root)
    return refs


def _bootstrap_delta(delta: torch.Tensor, generator: torch.Generator) -> list[float]:
    indices = torch.randint(
        delta.numel(),
        (BOOTSTRAP_REPLICATES, delta.numel()),
        generator=generator,
    )
    values = delta[indices].mean(1)
    return [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))]


def _paired_bootstrap(
    correctness: dict[str, dict[int, torch.Tensor]],
    baseline: str,
    variants: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(90210)
    result: dict[str, dict[str, Any]] = {}
    for variant in variants:
        deltas = [correctness[variant][panel] - correctness[baseline][panel] for panel in PANELS]
        aggregate = torch.cat(deltas)
        aggregate_indices = torch.randint(aggregate.numel(), (BOOTSTRAP_REPLICATES, aggregate.numel()), generator=generator)
        aggregate_values = aggregate[aggregate_indices].mean(1)
        class_ci = {
            str(panel): [_bootstrap_delta(delta.reshape(10, 50)[label], generator) for label in range(10)]
            for panel, delta in zip(PANELS, deltas, strict=True)
        }
        result[variant] = {
            "baseline": baseline,
            "observed_gain": float(aggregate.mean()),
            "ci95": [float(torch.quantile(aggregate_values, 0.025)), float(torch.quantile(aggregate_values, 0.975))],
            "panel_observed_gain": {str(panel): float(delta.mean()) for panel, delta in zip(PANELS, deltas, strict=True)},
            "panel_ci95": {str(panel): _bootstrap_delta(delta, generator) for panel, delta in zip(PANELS, deltas, strict=True)},
            "class_observed_gain": {str(panel): [float(x.mean()) for x in delta.reshape(10, 50)] for panel, delta in zip(PANELS, deltas, strict=True)},
            "class_ci95": class_ci,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": 90210,
        }
    return result


def _validate_artifacts(root: Path, payload: dict[str, Any]) -> None:
    names = (
        "labels",
        "codes",
        "fixed_cmos",
        "primary_prediction",
        "secondary_prediction",
        "secondary_correctness",
        "representative_trace",
    )
    for variant in payload["variants"]:
        for panel in payload["panels"]:
            refs = payload["rows"][variant][str(panel)]["artifacts"]
            for name in names:
                reference = refs.get(name)
                if not isinstance(reference, dict):
                    raise ValueError(f"missing artifact reference: {variant}/{panel}/{name}")
                path = _resolve(reference, root)
                if name != "representative_trace":
                    value = safe_torch_load(path, map_location="cpu")
                    if reference.get("tensor_sha256") != _tensor_hash(value):
                        raise ValueError(f"tensor hash mismatch: {variant}/{panel}/{name}")


def _load_measurement(path: Path, *, require_current: bool) -> tuple[Path, dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("schema") != "fiber_loss_exposure_measurement.v1" or payload.get("gate") != "valid":
        raise ValueError("invalid loss/exposure measurement")
    _validate_artifacts(path.parent, payload)
    if require_current and (_dirty() or payload.get("git_commit") != _git()):
        raise ValueError("loss/exposure measurement is stale or worktree is dirty")
    return path.parent, payload


def _classification(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    bootstrap = payload["bootstrap"]
    global_row = bootstrap["S0-Ecal"]
    oracle_row = bootstrap["S0-per-sample-oracle"]
    shape_row = bootstrap["I0-Acal"]
    panel_gains = [
        rows["S0-Ecal"][str(panel)]["secondary_accuracy"] - rows["S0-E0"][str(panel)]["secondary_accuracy"]
        for panel in PANELS
    ]
    recovered_panels = [
        panel
        for panel in PANELS
        if rows["S0-Ecal"][str(panel)]["secondary_min_recall"] > rows["S0-E0"][str(panel)]["secondary_min_recall"]
        and rows["S0-Ecal"][str(panel)]["secondary_min_recall"] > 0.0
    ]
    saturation_ok = max(rows["S0-Ecal"][str(panel)]["saturation"] for panel in PANELS) <= 0.08
    primary_ok = all(
        rows["S0-Ecal"][str(panel)]["primary_accuracy"] >= rows["S0-E0"][str(panel)]["primary_accuracy"] - 0.02
        for panel in PANELS
    )
    oracle_gain = oracle_row["observed_gain"]
    recovery_fraction = global_row["observed_gain"] / oracle_gain if oracle_gain > 0.0 else 0.0
    global_gate = (
        global_row["observed_gain"] >= 0.015
        and global_row["ci95"][0] > 0.0
        and sum(value > 0.0 for value in panel_gains) >= 2
        and saturation_ok
        and primary_ok
    )
    semantic_candidate = global_gate and len(recovered_panels) >= 2 and recovery_fraction >= 0.80
    oracle_only = global_gate and not semantic_candidate and recovery_fraction >= 0.80
    shape_gap = shape_row["observed_gain"]
    shape_candidate = (
        shape_gap >= 0.015
        and shape_row["ci95"][0] > 0.0
        and sum(value > 0.0 for value in shape_row["panel_observed_gain"].values()) >= 2
    ) or (global_gate and recovery_fraction < 0.50)
    if semantic_candidate:
        conclusion = "global_exposure_semantic_recovery_candidate"
    elif oracle_only:
        conclusion = "global_exposure_explains_oracle_only"
    elif shape_candidate:
        conclusion = "field_shape_dominant_after_power_match"
    else:
        conclusion = "loss_exposure_inconclusive"
    return {
        "conclusion": conclusion,
        "global_exposure_gate": global_gate,
        "semantic_recovery_candidate": semantic_candidate,
        "oracle_only": oracle_only,
        "panel_gains": {str(panel): gain for panel, gain in zip(PANELS, panel_gains, strict=True)},
        "minimum_recall_recovered_panels": recovered_panels,
        "saturation_ok": saturation_ok,
        "primary_ok": primary_ok,
        "oracle_gain": oracle_gain,
        "global_gain": global_row["observed_gain"],
        "oracle_recovery_fraction": recovery_fraction,
        "matched_power_shape_gap": shape_gap,
        "global_exposure_bootstrap": global_row,
        "oracle_bootstrap": oracle_row,
        "matched_power_bootstrap": shape_row,
    }


def run(args: Any) -> int:
    validation_path = _require_file(args, "validation_evidence")
    validation = _load_json(validation_path)
    if validation.get("schema") != "fiber_loss_exposure_validation.v1" or validation.get("gate") != "valid":
        raise ValueError("invalid loss/exposure validation")
    if _dirty() or validation.get("git_commit") != _git():
        raise ValueError("loss/exposure validation is stale or worktree is dirty")
    if validation.get("interface_measurement_sha256") != file_sha256(_require_file(args, "interface_measurement")):
        raise ValueError("loss/exposure validation lineage is stale")
    context = _context(args, validation_path.parent)
    config, device, _, _, _, _, _, _, bundle, _, _, _ = context
    settings = config.conditional_generation
    if settings is None:
        raise ValueError("conditional generation settings are required")
    root = _root(args.output, require_new=True)
    before_state = _parameter_state(bundle)

    calibration_labels, calibration_codes = _labels_codes(CALIBRATION_PANEL, device)
    calibration_raw = _forward_raw(context, "S0", calibration_labels, calibration_codes)
    rho, gain = _calibration_ratio(calibration_raw["input_energy"], calibration_raw["output_energy"])
    exposure = settings.fixed_exposure_scale * rho
    calibration = {
        "schema": "fiber_loss_exposure_calibration.v1",
        "gate": "valid",
        "git_commit": _git(),
        "validation_sha256": file_sha256(validation_path),
        "panel": CALIBRATION_PANEL,
        "sample_count": SAMPLES_PER_PANEL,
        "labels_sha256": _tensor_hash(calibration_labels.cpu()),
        "codes_sha256": _tensor_hash(calibration_codes.cpu()),
        "input_energy_sum": float(calibration_raw["input_energy"].sum()),
        "projected_energy_sum": float(calibration_raw["output_energy"].sum()),
        "rho_cal": rho,
        "global_gain": gain,
        "nominal_exposure_scale": settings.fixed_exposure_scale,
        "calibrated_exposure_scale": exposure,
        "evaluator_used": False,
        "optimizer_created": False,
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    _write(root / "calibration.json", calibration)
    calibration_sha256 = file_sha256(root / "calibration.json")

    all_variants = _variant_names(gain)
    rows: dict[str, dict[str, Any]] = {variant: {} for variant in all_variants}
    correctness: dict[str, dict[int, torch.Tensor]] = {variant: {} for variant in all_variants}
    pairing: dict[str, dict[str, dict[str, str]]] = {
        "labels": {variant: {} for variant in all_variants},
        "codes": {variant: {} for variant in all_variants},
    }
    for panel in PANELS:
        labels, codes = _labels_codes(panel, device)
        identity = _forward_raw(context, "I0", labels, codes)
        s0 = _forward_raw(context, "S0", labels, codes)
        oracle_factor = s0["input_energy"] / s0["output_energy"].clamp_min(1.0e-30)
        raw_by_variant: dict[str, tuple[torch.Tensor, float, float | torch.Tensor, dict[str, Any]]] = {
            "I0-E0": (identity["raw"], settings.fixed_exposure_scale, 1.0, identity),
            "S0-E0": (s0["raw"], settings.fixed_exposure_scale, 1.0, s0),
            "S0-Ecal": (s0["raw"], exposure, 1.0, s0),
            "I0-Acal": (identity["raw"] * rho, settings.fixed_exposure_scale, rho**0.5, identity),
            "S0-field-gain-reference": (s0["raw"] * gain, settings.fixed_exposure_scale, gain**0.5, s0),
            "S0-per-sample-oracle": (
                s0["raw"] * oracle_factor.reshape(-1, 1, 1, 1),
                settings.fixed_exposure_scale,
                oracle_factor.sqrt(),
                s0,
            ),
        }
        for grid_gain, variant in zip(GAIN_GRID, all_variants[len(BASE_VARIANTS):], strict=True):
            raw_by_variant[variant] = (s0["raw"], settings.fixed_exposure_scale / grid_gain, 1.0, s0)
        for variant in all_variants:
            raw, exposure_scale, amplitude_scale, base = raw_by_variant[variant]
            metric, values = _view_metric(context, variant, panel, raw, exposure_scale, labels, codes.cpu(), base, amplitude_scale)
            refs = _save_artifacts(root, variant, panel, values)
            metric["artifacts"] = refs
            rows[variant][str(panel)] = metric
            correctness[variant][panel] = values["secondary_correctness"].float()
            pairing["labels"][variant][str(panel)] = _tensor_hash(values["labels"])
            pairing["codes"][variant][str(panel)] = _tensor_hash(values["codes"])
    bootstrap_variants = ("S0-Ecal", "I0-Acal", "S0-per-sample-oracle")
    bootstrap = _paired_bootstrap(correctness, "S0-E0", bootstrap_variants)
    baseline_gate = True
    interface_payload = _load_json(_require_file(args, "interface_measurement"))
    for panel in PANELS:
        for new_variant, old_variant in (("I0-E0", "I0"), ("S0-E0", "S0")):
            old_ref = interface_payload["rows"][old_variant][str(panel)]["artifacts"]["fixed_cmos"]
            old_view = safe_torch_load(interface._resolve(old_ref, Path(args.interface_measurement).parent), map_location="cpu")
            new_view = safe_torch_load(_resolve(rows[new_variant][str(panel)]["artifacts"]["fixed_cmos"], root), map_location="cpu")
            baseline_gate = baseline_gate and interface._relative_l2(new_view, old_view) <= 1.0e-5
    after_state = _parameter_state(bundle)
    before_after_gate = before_state == after_state
    locked_rho = {
        str(panel): rows["S0-E0"][str(panel)]["projection_fraction_mean"]
        for panel in PANELS
    }
    rho_deviation = max(abs(value - rho) for value in locked_rho.values())
    pairing_labels = all(
        len({pairing["labels"][variant][str(panel)] for variant in all_variants}) == 1
        for panel in PANELS
    )
    pairing_codes = all(
        len({pairing["codes"][variant][str(panel)] for variant in all_variants}) == 1
        for panel in PANELS
    )
    harness = {
        "calibration": calibration["gate"] == "valid" and calibration["evaluator_used"] is False,
        "baseline_reproduction": baseline_gate,
        "pairing_labels": pairing_labels,
        "pairing_codes": pairing_codes,
        "field_gain_exposure_equivalence": all(
            interface._relative_l2(
                safe_torch_load(_resolve(rows["S0-field-gain-reference"][str(panel)]["artifacts"]["fixed_cmos"], root), map_location="cpu"),
                safe_torch_load(_resolve(rows["S0-Ecal"][str(panel)]["artifacts"]["fixed_cmos"], root), map_location="cpu"),
            ) <= 1.0e-5
            for panel in PANELS
        ),
        "passive_global_branches": all(
            rows[variant][str(panel)]["passive"] is True
            for variant in ("S0-E0", "S0-Ecal", "I0-Acal")
            for panel in PANELS
        ),
        "finite_nonnegative": all(
            rows[variant][str(panel)]["finite"] and rows[variant][str(panel)]["nonnegative"] and rows[variant][str(panel)]["raw_finite"] and rows[variant][str(panel)]["raw_nonnegative"]
            for variant in all_variants
            for panel in PANELS
        ),
        "candidate_saturation": max(rows["S0-Ecal"][str(panel)]["saturation"] for panel in PANELS) <= 0.08,
        "parameter_isolation": before_after_gate,
        "no_optimizer": True,
    }
    storage = sum(path.stat().st_size for path in root.parent.rglob("*") if path.is_file())
    harness["storage"] = storage <= STORAGE_LIMIT_BYTES
    payload = {
        "schema": "fiber_loss_exposure_measurement.v1",
        "gate": "valid" if all(harness.values()) else "invalid_harness",
        "git_commit": _git(),
        "validation_sha256": file_sha256(validation_path),
        "calibration_sha256": calibration_sha256,
        "interface_validation_sha256": validation["interface_validation_sha256"],
        "interface_measurement_sha256": validation["interface_measurement_sha256"],
        "interface_closeout_sha256": validation["interface_closeout_sha256"],
        "variants": list(all_variants),
        "panels": list(PANELS),
        "calibration_panel": CALIBRATION_PANEL,
        "rho_cal": rho,
        "global_gain": gain,
        "nominal_exposure_scale": settings.fixed_exposure_scale,
        "calibrated_exposure_scale": exposure,
        "locked_projection_fraction_mean": locked_rho,
        "projection_fraction_max_abs_deviation_from_calibration": rho_deviation,
        "parameter_state_before": before_state,
        "parameter_state_after": after_state,
        "rows": rows,
        "bootstrap": bootstrap,
        "pairing": pairing,
        "harness_gates": harness,
        "storage_bytes": storage,
        "storage_limit_bytes": STORAGE_LIMIT_BYTES,
        "status": "synthetic diagnostic-only",
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    _write(root / "measurement.json", payload)
    return 0 if payload["gate"] == "valid" else 2


def report(args: Any) -> int:
    measurement_path = _require_file(args, "measurement_evidence")
    _, payload = _load_measurement(measurement_path, require_current=False)
    output = _root(args.output, require_new=True)
    classification = _classification(payload)
    with (output / "loss_exposure_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "panel", "primary_accuracy", "secondary_accuracy", "secondary_min_recall", "exposure_scale", "saturation", "clipped_fraction", "camera_fraction", "minimum_diversity", "coverage_ratio"])
        for variant in payload["variants"]:
            for panel in payload["panels"]:
                row = payload["rows"][variant][str(panel)]
                writer.writerow([variant, panel, row["primary_accuracy"], row["secondary_accuracy"], row["secondary_min_recall"], row["exposure_scale"], row["saturation"], row["clipped_fraction"], row["camera_fraction"], row["minimum_diversity"], row["coverage_ratio"]])
    _write(output / "classification.json", classification)
    _write(output / "presentation.json", {
        "schema": "fiber_loss_exposure_presentation.v1",
        "gate": "valid",
        "measurement_sha256": file_sha256(measurement_path),
        "source_measurement_git_commit": payload["git_commit"],
        "report_git_commit": _git(),
        "classification": classification,
        "status": "synthetic diagnostic-only",
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
    })
    return 0


def closeout(args: Any) -> int:
    measurement_path = _require_file(args, "measurement_evidence")
    presentation_path = _require_file(args, "presentation_evidence")
    _, payload = _load_measurement(measurement_path, require_current=True)
    presentation = _load_json(presentation_path)
    if (
        presentation.get("schema") != "fiber_loss_exposure_presentation.v1"
        or presentation.get("measurement_sha256") != file_sha256(measurement_path)
        or presentation.get("source_measurement_git_commit") != payload["git_commit"]
    ):
        raise ValueError("presentation is not bound to loss/exposure measurement")
    classification = _classification(payload)
    actions = {
        "global_exposure_semantic_recovery_candidate": "register_synthetic_exposure_robustness_only",
        "global_exposure_explains_oracle_only": "register_zero_phase_decoder_controllability_attribution",
        "field_shape_dominant_after_power_match": "register_zero_phase_decoder_controllability_attribution",
        "loss_exposure_inconclusive": "close_current_fiber_route_no_e1",
    }
    storage_root = measurement_path.parent.parent if measurement_path.parent.name == "measurement" else measurement_path.parent
    storage = sum(path.stat().st_size for path in storage_root.rglob("*") if path.is_file())
    if storage > STORAGE_LIMIT_BYTES:
        raise ValueError("loss/exposure storage gate failed at closeout")
    _write(Path(args.output), {
        "schema": "fiber_loss_exposure_closeout.v1",
        "gate": "valid",
        "conclusion": classification["conclusion"],
        "next_action": actions[classification["conclusion"]],
        "measurement_sha256": file_sha256(measurement_path),
        "presentation_sha256": file_sha256(presentation_path),
        "validation_sha256": payload["validation_sha256"],
        "calibration_sha256": payload["calibration_sha256"],
        "interface_measurement_sha256": payload["interface_measurement_sha256"],
        "interface_closeout_sha256": payload["interface_closeout_sha256"],
        "rho_cal": payload["rho_cal"],
        "global_gain": payload["global_gain"],
        "storage_bytes": storage,
        "storage_limit_bytes": STORAGE_LIMIT_BYTES,
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
    })
    return 0


def dispatch(args: Any) -> int:
    return {"validate": validate, "run": run, "report": report, "closeout": closeout}[args.research_action](args)
