"""No-update zero-phase decoder controllability attribution for synthetic S0.

The experiment freezes the generator, encoder and S0 LP01 operator, derives a
single decoder-phase tangent from an independent primary-evaluator design
panel, and evaluates that fixed tangent on locked panels.  It is a diagnostic
counterfactual: no optimizer, checkpoint, or training authorization is
created.
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
import torch.nn.functional as F

from ...contracts import ComplexField
from ...evidence import atomic_torch_save, atomic_write_json, file_sha256, safe_torch_load
from ...student.conditional import fixed_exposure_view
from .._runtime_shared import random_codes
from . import fiber_interface_attribution as interface
from . import fiber_loss_exposure_attribution as exposure

DESIGN_PANEL = 314159
PANELS = interface.PANELS
SAMPLES_PER_PANEL = 500
SAMPLES_PER_LABEL = 50
BOOTSTRAP_REPLICATES = 10_000
STORAGE_LIMIT_BYTES = 2 * 1024**3
PHASE_RMS_GRID = (0.005, 0.010, 0.025, 0.050)
LOCAL_PHASE_RMS = 0.005
RAW_L2_LIMIT = 0.05
SATURATION_LIMIT = 0.08
LOCAL_COSINE_LIMIT = 0.90
LOCAL_ODD_ERROR_LIMIT = 0.20
VARIANTS = ("D0", "Dshared+", "Dshared-", "Dscrambled", "Dclass-oracle")
PHYSICAL_VARIANTS = ("D0", "Dshared+", "Dshared-", "Dscrambled")


def _write(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require_file(args: Any, name: str) -> Path:
    value = getattr(args, name, None)
    if not value:
        raise ValueError(f"decoder controllability attribution requires --{name.replace('_', '-')}")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"decoder controllability attribution input does not exist: {path}")
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
        raise ValueError("invalid decoder controllability artifact reference")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file() or file_sha256(path) != expected:
        raise ValueError("missing or hash-mismatched decoder controllability artifact")
    return path


def _run_root(output: str | Path, *, require_new: bool) -> Path:
    path = Path(output)
    root = path.parent.parent if path.suffix == ".json" else path
    if require_new and root.exists() and any(root.iterdir()):
        raise ValueError(f"decoder controllability run must be new: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _output_directory(output: str | Path, *, require_new: bool) -> Path:
    directory = Path(output).parent
    if require_new and directory.exists() and any(directory.iterdir()):
        raise ValueError(f"decoder output directory must be new: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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
    return labels, random_codes(SAMPLES_PER_PANEL, device=device, seed=panel)


def _lineage_namespace(args: Any, root: Path) -> SimpleNamespace:
    values = vars(args).copy()
    values["measurement_bundle"] = str(root / "fiber_image_measurement_bundle.json")
    values["measurement_evidence"] = str(root / "fiber_image_measurement_evidence.json")
    values["validate_evidence"] = str(root / "fiber_image_validation.json")
    values["output"] = str(root / "fiber_image_validation.json")
    return SimpleNamespace(**values)


def _context(args: Any, root: Path) -> tuple[Any, ...]:
    context = exposure._context(_lineage_namespace(args, root), root)
    bundle = context[8]
    for parameter in bundle.generator.parameters():
        parameter.requires_grad_(False)
    return context


def _load_loss_inputs(args: Any) -> dict[str, Any]:
    validation_path = _require_file(args, "loss_exposure_validation")
    measurement_path = _require_file(args, "loss_exposure_measurement")
    presentation_path = _require_file(args, "loss_exposure_presentation")
    closeout_path = _require_file(args, "loss_exposure_closeout")
    validation = _load_json(validation_path)
    if validation.get("schema") != "fiber_loss_exposure_validation.v1" or validation.get("gate") != "valid":
        raise ValueError("bound loss/exposure validation is not valid")
    _, measurement = exposure._load_measurement(measurement_path, require_current=False)
    presentation = _load_json(presentation_path)
    closeout = _load_json(closeout_path)
    if presentation.get("schema") != "fiber_loss_exposure_presentation.v1" or presentation.get("gate") != "valid":
        raise ValueError("bound loss/exposure presentation is not valid")
    if closeout.get("schema") != "fiber_loss_exposure_closeout.v1" or closeout.get("gate") != "valid":
        raise ValueError("bound loss/exposure closeout is not valid")
    if validation.get("git_commit") != measurement.get("git_commit"):
        raise ValueError("loss/exposure validation and measurement commits differ")
    if validation_path.read_bytes() and closeout.get("validation_sha256") != file_sha256(validation_path):
        raise ValueError("loss/exposure closeout does not bind validation")
    if closeout.get("measurement_sha256") != file_sha256(measurement_path):
        raise ValueError("loss/exposure closeout does not bind measurement")
    if closeout.get("presentation_sha256") != file_sha256(presentation_path):
        raise ValueError("loss/exposure closeout does not bind presentation")
    if measurement.get("calibrated_exposure_scale") is None:
        raise ValueError("loss/exposure measurement has no signed exposure")
    return {
        "validation": validation,
        "measurement": measurement,
        "presentation": presentation,
        "closeout": closeout,
        "validation_sha256": file_sha256(validation_path),
        "measurement_sha256": file_sha256(measurement_path),
        "presentation_sha256": file_sha256(presentation_path),
        "closeout_sha256": file_sha256(closeout_path),
    }


def _validation_payload(args: Any, root: Path) -> dict[str, Any]:
    loss = _load_loss_inputs(args)
    interface_inputs = exposure._validate_interface_inputs(args)
    upstream = interface._validate_upstream(args, root)
    historical = interface._validate_historical_inputs(args, upstream)
    context = _context(args, root)
    config, device, _, _, _, _, _, _, bundle, _, _, _ = context
    if config.conditional_generation is None:
        raise ValueError("conditional generation settings are required")
    audit = exposure._source_state_audit(args, bundle)
    fixed_exposure = float(loss["measurement"]["calibrated_exposure_scale"])
    expected_exposure = float(loss["measurement"]["nominal_exposure_scale"] * loss["measurement"]["rho_cal"])
    interface_gates = interface_inputs["validation"].get("harness_gates", {})
    loss_gates = loss["measurement"].get("harness_gates", {})
    harness = {
        "device_cuda": device.type == "cuda",
        "loss_exposure_lineage": all(loss_gates.values()) and loss["closeout"]["training_authorized"] is False,
        "interface_lineage": bool(interface_gates) and all(interface_gates.values()),
        "design_panel_disjoint": DESIGN_PANEL not in PANELS and DESIGN_PANEL != exposure.CALIBRATION_PANEL,
        "runtime_zero_phase": audit["runtime_encoder_phase_zero"] and audit["runtime_decoder_phase_zero"],
        "source_generator_scope": bool(audit["loaded_generator_state_keys"]),
        "source_system_keys_ignored": bool(audit["ignored_state_keys"]),
        "no_optical_trainable_parameters": not audit["runtime_optical_trainable"],
        "fixed_exposure_matches_signed_evidence": (
            expected_exposure > 0.0
            and torch.isfinite(torch.tensor(expected_exposure))
            and abs(fixed_exposure - expected_exposure) <= 1.0e-12
        ),
        "generator_frozen": not audit["runtime_generator_trainable"],
        "no_optimizer": True,
        "mode_bank_rank": interface_gates.get("input_mode_rank") is True and interface_gates.get("output_mode_rank") is True,
        "plane_provenance": interface_gates.get("plane_provenance") is True,
    }
    return {
        "schema": "fiber_decoder_controllability_validation.v1",
        "gate": "valid" if all(harness.values()) else "invalid_harness",
        "git_commit": _git(),
        "config_hash": loss["validation"].get("config_hash"),
        "dataset_split_manifest_hash": loss["validation"].get("dataset_split_manifest_hash"),
        "primary_evaluator_sha256": loss["validation"].get("primary_evaluator_sha256"),
        "secondary_evaluator_sha256": loss["validation"].get("secondary_evaluator_sha256"),
        "research_initialization_checkpoint_sha256": loss["validation"].get("research_initialization_checkpoint_sha256"),
        "source_training_evidence_sha256": loss["validation"].get("source_training_evidence_sha256"),
        "lp01_fixture_sha256": loss["validation"].get("lp01_fixture_sha256"),
        "v3_closeout_sha256": loss["validation"].get("v3_closeout_sha256"),
        "attribution_measurement_sha256": loss["validation"].get("attribution_measurement_sha256"),
        "upstream_validation": upstream,
        "historical_binding": {
            "v3_closeout_sha256": historical["v3_closeout_sha256"],
            "attribution_measurement_sha256": historical["attribution_measurement_sha256"],
            "lp01_fixture_sha256": historical["lp01_fixture_sha256"],
        },
        "interface_validation_sha256": interface_inputs["validation_sha256"],
        "interface_measurement_sha256": interface_inputs["measurement_sha256"],
        "interface_closeout_sha256": interface_inputs["closeout_sha256"],
        "loss_exposure_validation_sha256": loss["validation_sha256"],
        "loss_exposure_measurement_sha256": loss["measurement_sha256"],
        "loss_exposure_presentation_sha256": loss["presentation_sha256"],
        "loss_exposure_closeout_sha256": loss["closeout_sha256"],
        "fixed_exposure_scale": fixed_exposure,
        "design_panel": DESIGN_PANEL,
        "panels": list(PANELS),
        "sample_count_per_panel": SAMPLES_PER_PANEL,
        "phase_rms_grid": list(PHASE_RMS_GRID),
        "variants": list(VARIANTS),
        "source_state_audit": audit,
        "harness_gates": harness,
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
    }


def validate(args: Any) -> int:
    output = Path(args.output)
    root = _output_directory(output, require_new=True)
    if _dirty():
        _write(output, {"schema": "fiber_decoder_controllability_validation.v1", "gate": "invalid_harness", "reason": "dirty worktree", "formal_promotion": False, "training_authorized": False, "synthetic_diagnostic_only": True})
        return 2
    try:
        payload = _validation_payload(args, root)
        _write(output, payload)
        return 0 if payload["gate"] == "valid" else 2
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        _write(output, {"schema": "fiber_decoder_controllability_validation.v1", "gate": "invalid_harness", "reason": str(exc), "formal_promotion": False, "training_authorized": False, "synthetic_diagnostic_only": True})
        return 2


def _zero_phase(bundle: Any) -> None:
    with torch.no_grad():
        bundle.system.decoder.phase.zero_()


def _decode_ols(bundle: Any, ols_tensor: torch.Tensor, plane: Any, provenance: Any) -> dict[str, Any]:
    with torch.inference_mode(False):
        ols_tensor = ols_tensor.clone()
    field = ComplexField(ols_tensor, plane, provenance)
    decoded, records, stats = bundle.system.decoder(field, "decoder")
    chip = ComplexField(decoded.tensor, bundle.system.chip_plane, decoded.provenance)
    camera = bundle.system.back(chip)
    return {
        "raw": camera.tensor.abs().square(),
        "camera_field": camera.tensor,
        "decoded_field": decoded.tensor,
        "records": records,
        "stats": stats,
    }


def _collect_ols_panel(context: tuple[Any, ...], labels: torch.Tensor, codes: torch.Tensor) -> dict[str, Any]:
    config, device, _, _, _, _, _, _, bundle, _, _, _ = context
    interface._install_variant(bundle, "S0", device)
    tensors: list[torch.Tensor] = []
    full_raw: list[torch.Tensor] = []
    full_camera: list[torch.Tensor] = []
    plane = None
    provenance = None
    for start in range(0, SAMPLES_PER_PANEL, config.execution.evaluation_microbatch_size):
        code_batch = codes[start : start + config.execution.evaluation_microbatch_size]
        label_batch = labels[start : start + code_batch.shape[0]]
        with torch.inference_mode():
            command = bundle.generator(code_batch, label_batch)
            result = bundle.system.forward(bundle.system.command(command), return_optical_trace=True)
        fields = result.optical_trace or ()
        if len(fields) != len(interface.STAGES) - 1:
            raise ValueError("S0 optical trace length is invalid")
        ols = fields[4]
        if plane is None:
            plane, provenance = ols.plane, ols.provenance
        tensors.extend(value.detach().cpu() for value in ols.tensor)
        full_raw.append(result.raw_camera.tensor.detach().cpu())
        full_camera.append(fields[-1].tensor.detach().cpu())
    if plane is None or provenance is None:
        raise ValueError("S0 panel produced no OLS field")
    with torch.inference_mode(False):
        tensors = [value.clone() for value in tensors]
    return {
        "ols": tensors,
        "plane": plane,
        "provenance": provenance,
        "full_raw": torch.cat(full_raw),
        "full_camera": torch.cat(full_camera),
    }


def _phase_rms(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt())


def _energy(tensor: torch.Tensor, pixel_pitch: tuple[float, float]) -> torch.Tensor:
    return tensor.abs().square().sum(dim=(-3, -2, -1)) * (pixel_pitch[0] * pixel_pitch[1])


def _normalize_direction(value: torch.Tensor) -> torch.Tensor:
    rms = value.square().mean().sqrt().clamp_min(1.0e-30)
    return value / rms


def _phase_stats(value: torch.Tensor) -> dict[str, float]:
    return {"rms": _phase_rms(value), "max_abs": float(value.abs().max()), "finite": bool(torch.isfinite(value).all())}


def _primary_design_metrics(
    context: tuple[Any, ...],
    panel: dict[str, Any],
    labels: torch.Tensor,
    variant: str,
    direction: torch.Tensor | None,
    scale: float,
    exposure_scale: float,
    *,
    collect_raw: bool = True,
) -> dict[str, Any]:
    config, device, _, _, _, _, primary, _, bundle, _, _, _ = context
    if direction is not None:
        with torch.no_grad():
            bundle.system.decoder.phase.copy_(direction.to(device) * scale)
    else:
        _zero_phase(bundle)
    raws: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for ols_cpu in panel["ols"]:
            ols = ols_cpu.unsqueeze(0).to(device)
            decoded = _decode_ols(bundle, ols, panel["plane"], panel["provenance"])
            raw = decoded["raw"]
            view = fixed_exposure_view(raw, exposure_scale)
            _, current_logits = primary(view)
            logits.append(current_logits.detach().cpu())
            if collect_raw:
                raws.append(raw.detach().cpu())
    _zero_phase(bundle)
    logits_tensor = torch.cat(logits)
    labels_cpu = labels.cpu()
    ce = F.cross_entropy(logits_tensor, labels_cpu)
    prediction = logits_tensor.argmax(1)
    recalls = [float((prediction[labels_cpu == label] == label).float().mean()) for label in range(10)]
    raw = torch.cat(raws) if raws else torch.empty(0)
    base_raw = panel["baseline_raw"]
    raw_l2 = float(interface._relative_l2(raw, base_raw)) if raw.numel() else 0.0
    view = fixed_exposure_view(raw, exposure_scale) if raw.numel() else torch.empty(0)
    return {
        "variant": variant,
        "scale": scale,
        "primary_macro_ce": float(ce),
        "primary_accuracy": float((prediction == labels_cpu).float().mean()),
        "primary_recall": recalls,
        "primary_min_recall": min(recalls),
        "raw_l2_vs_d0": raw_l2,
        "saturation": float((raw > exposure_scale).float().mean()) if raw.numel() else 0.0,
        "raw": raw,
        "view": view,
        "primary_prediction": prediction,
    }


def _gradient_for_panel(
    context: tuple[Any, ...],
    panel: dict[str, Any],
    labels: torch.Tensor,
    exposure_scale: float,
    *,
    selected_label: int | None = None,
) -> torch.Tensor:
    config, device, _, _, _, _, primary, _, bundle, _, _, _ = context
    del config
    phase = bundle.system.decoder.phase
    phase.requires_grad_(True)
    _zero_phase(bundle)
    total = torch.zeros_like(phase)
    labels_cpu = labels.cpu()
    indices = range(len(panel["ols"])) if selected_label is None else torch.where(labels_cpu == selected_label)[0].tolist()
    for index in indices:
        ols = panel["ols"][index].unsqueeze(0).to(device)
        decoded = _decode_ols(bundle, ols, panel["plane"], panel["provenance"])
        view = fixed_exposure_view(decoded["raw"], exposure_scale)
        _, logits = primary(view)
        target = labels[index : index + 1].to(device)
        loss = F.cross_entropy(logits, target)
        gradient = torch.autograd.grad(loss, phase, retain_graph=False, create_graph=False)[0]
        total.add_(gradient.detach())
    _zero_phase(bundle)
    phase.requires_grad_(False)
    return total / max(len(indices), 1)


def _local_consistency(
    context: tuple[Any, ...],
    panel: dict[str, Any],
    labels: torch.Tensor,
    direction: torch.Tensor,
    exposure_scale: float,
) -> dict[str, Any]:
    plus = _primary_design_metrics(context, panel, labels, "Dshared+local", direction, LOCAL_PHASE_RMS, exposure_scale)
    minus = _primary_design_metrics(context, panel, labels, "Dshared-local", direction, -LOCAL_PHASE_RMS, exposure_scale)
    base = panel["baseline_raw"]
    delta_plus = plus["raw"] - base
    delta_minus = minus["raw"] - base
    numerator = torch.sum(delta_plus * (-delta_minus))
    cosine = float(numerator / (delta_plus.norm() * delta_minus.norm()).clamp_min(1.0e-30))
    odd_error = float((delta_plus + delta_minus).norm() / (delta_plus - delta_minus).norm().clamp_min(1.0e-30))
    return {"cosine": cosine, "odd_error": odd_error, "plus_primary_macro_ce": plus["primary_macro_ce"], "minus_primary_macro_ce": minus["primary_macro_ce"]}


def _scramble_direction(direction: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(8675309)
    values = direction.detach().cpu().flatten()
    permutation = torch.randperm(values.numel(), generator=generator)
    return values[permutation].reshape(direction.shape).to(direction)


def _gradient_conflict(gradients: torch.Tensor) -> dict[str, Any]:
    normalized = F.normalize(gradients.flatten(1), dim=1)
    cosine = normalized @ normalized.T
    mask = ~torch.eye(cosine.shape[0], dtype=torch.bool)
    off_diagonal = cosine[mask]
    return {
        "cosine_matrix": cosine.tolist(),
        "negative_fraction_below_minus_0_10": float((off_diagonal < -0.10).float().mean()),
        "median_off_diagonal_cosine": float(off_diagonal.median()),
        "minimum_off_diagonal_cosine": float(off_diagonal.min()),
    }


def _select_scale(rows: list[dict[str, Any]]) -> tuple[float, bool]:
    eligible = [row for row in rows if row["eligible"]]
    selected = min(eligible, key=lambda row: (row["primary_macro_ce"], row["scale"])) if eligible else None
    return float(selected["scale"] if selected is not None else PHASE_RMS_GRID[0]), selected is not None


def _design(context: tuple[Any, ...], root: Path, validation_path: Path, exposure_scale: float) -> dict[str, Any]:
    config, device, _, _, _, _, primary, secondary, bundle, _, _, _ = context
    del secondary
    labels, codes = _labels_codes(DESIGN_PANEL, device)
    panel = _collect_ols_panel(context, labels, codes)
    panel["baseline_raw"] = panel["full_raw"]
    base = _primary_design_metrics(context, panel, labels, "D0", None, 0.0, exposure_scale)
    shared_gradient = _gradient_for_panel(context, panel, labels, exposure_scale)
    class_gradients = torch.stack([
        _gradient_for_panel(context, panel, labels, exposure_scale, selected_label=label)
        for label in range(10)
    ])
    shared_direction = _normalize_direction(-shared_gradient)
    scrambled_direction = _scramble_direction(shared_direction)
    design_rows = []
    for scale in PHASE_RMS_GRID:
        row = _primary_design_metrics(context, panel, labels, "Dshared+", shared_direction, scale, exposure_scale)
        row.pop("raw", None)
        row.pop("view", None)
        row.pop("primary_prediction", None)
        row["phase"] = _phase_stats(shared_direction * scale)
        row["eligible"] = bool(
            row["phase"]["finite"]
            and row["phase"]["max_abs"] <= 0.25
            and row["raw_l2_vs_d0"] <= RAW_L2_LIMIT
            and row["saturation"] <= SATURATION_LIMIT
            and row["primary_macro_ce"] < base["primary_macro_ce"]
            and row["primary_accuracy"] >= base["primary_accuracy"] - 0.02
        )
        design_rows.append(row)
    selected_scale, candidate_selected = _select_scale(design_rows)
    local = _local_consistency(context, panel, labels, shared_direction, exposure_scale)
    direction_payload = {
        "shared_direction": shared_direction.detach().cpu(),
        "shared_gradient": shared_gradient.detach().cpu(),
        "class_gradients": class_gradients.detach().cpu(),
        "scrambled_direction": scrambled_direction.detach().cpu(),
    }
    direction_stats = {
        name: _phase_stats(value)
        for name, value in direction_payload.items()
        if name.endswith("direction")
    }
    direction_path = root / "design" / "directions.pt"
    atomic_torch_save(direction_path, direction_payload)
    direction_ref = _ref(direction_path, root)
    design = {
        "schema": "fiber_decoder_controllability_design.v1",
        "gate": "valid",
        "git_commit": _git(),
        "validation_sha256": file_sha256(validation_path),
        "panel": DESIGN_PANEL,
        "sample_count": SAMPLES_PER_PANEL,
        "labels_sha256": _tensor_hash(labels.cpu()),
        "codes_sha256": _tensor_hash(codes.cpu()),
        "fixed_exposure_scale": exposure_scale,
        "direction_artifact": direction_ref,
        "direction_shapes": {name: list(value.shape) for name, value in direction_payload.items()},
        "direction_phase_stats": direction_stats,
        "control_phase_rms_match_error": max(
            abs(direction_stats["shared_direction"]["rms"] - direction_stats[name]["rms"])
            for name in ("scrambled_direction",)
        ),
        "gradient_conflict": _gradient_conflict(class_gradients.detach().cpu()),
        "base": {key: value for key, value in base.items() if key not in {"raw", "view", "primary_prediction"}},
        "scale_rows": design_rows,
        "selected_scale": selected_scale,
        "candidate_selected": candidate_selected,
        "local_consistency": local,
        "secondary_used_for_selection": False,
        "locked_metrics_used_for_selection": False,
        "optimizer_created": False,
        "checkpoint_created": False,
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    _write(root / "design" / "direction_design.json", design)
    _zero_phase(bundle)
    return design


def _evaluator_metrics(context: tuple[Any, ...], raw: torch.Tensor, labels: torch.Tensor, variant: str, exposure_scale: float, base: dict[str, Any], amplitude: torch.Tensor | float, representative_trace: list[dict[str, Any]], decoded_fields: torch.Tensor, baseline_decoded: torch.Tensor | None) -> tuple[dict[str, Any], dict[str, Any]]:
    config, _, _, _, reference_images, reference_labels, primary, secondary, _, _, _, _ = context
    settings = config.conditional_generation
    if settings is None:
        raise ValueError("conditional generation settings are required")
    view = fixed_exposure_view(raw, exposure_scale)
    metric, primary_prediction, secondary_prediction = interface._metric_summary(
        raw, view, labels, primary, secondary, reference_images, reference_labels, settings
    )
    with torch.inference_mode():
        _, primary_logits = primary(view.to(reference_images.device))
        _, secondary_logits = secondary(view.to(reference_images.device))
    labels_cpu = labels.cpu()
    secondary_recall = [float((secondary_prediction[labels_cpu == label] == label).float().mean()) for label in range(10)]
    scaled_energy = decoded_fields.abs().square().sum(dim=(-3, -2, -1))
    input_energy = base["input_energy"]
    metric.update({
        "variant": variant,
        "exposure_scale": exposure_scale,
        "amplitude_scale": "label-conditioned" if isinstance(amplitude, torch.Tensor) else float(amplitude),
        "primary_macro_ce": float(F.cross_entropy(primary_logits.cpu(), labels_cpu)),
        "secondary_macro_ce": float(F.cross_entropy(secondary_logits.cpu(), labels_cpu)),
        "primary_class_margin": _class_margins(primary_logits.cpu(), labels_cpu),
        "secondary_class_margin": _class_margins(secondary_logits.cpu(), labels_cpu),
        "saturation": float((raw > exposure_scale).float().mean()),
        "raw_l2_vs_d0": float(interface._relative_l2(raw, base["raw"])),
        "view_l2_vs_d0": float(interface._relative_l2(view, base["view"])),
        "raw_finite": bool(torch.isfinite(raw).all()),
        "raw_nonnegative": bool((raw >= 0).all()),
        "decoded_finite": bool(torch.isfinite(decoded_fields).all()),
        "decoded_energy_retention_mean": float((scaled_energy / input_energy.clamp_min(1.0e-30)).mean()),
        "decoder_crop_retention_mean": [float(value) for value in base["crop_retention"].mean(0)],
        "stage_complex_l2_vs_d0": 0.0 if baseline_decoded is None else float(interface._relative_l2(decoded_fields, baseline_decoded)),
        "secondary_recall": secondary_recall,
        "representative_trace_saved": bool(representative_trace),
        "nonphysical_upper_bound": variant == "Dclass-oracle",
        "passive": bool(torch.all(scaled_energy <= input_energy * (1.0 + 1.0e-5) + 1.0e-12)),
    })
    values = {
        "labels": labels_cpu,
        "codes": base["codes"].cpu(),
        "fixed_cmos": view.cpu(),
        "primary_prediction": primary_prediction,
        "secondary_prediction": secondary_prediction,
        "secondary_correctness": secondary_prediction == labels_cpu,
        "representative_trace": representative_trace,
        "raw": raw,
    }
    return metric, values


def _class_margins(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    true = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, labels[:, None], float("-inf"))
    margin = true - other.max(1).values
    return [float(margin[labels == label].mean()) for label in range(10)]


def _measure_variant(context: tuple[Any, ...], panel: dict[str, Any], labels: torch.Tensor, codes: torch.Tensor, variant: str, direction: torch.Tensor | None, scale: float, class_directions: torch.Tensor | None, exposure_scale: float, baseline: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    config, device, _, _, _, _, _, _, bundle, _, _, _ = context
    raw_parts: list[torch.Tensor] = []
    decoded_parts: list[torch.Tensor] = []
    camera_parts: list[torch.Tensor] = []
    input_energy: list[torch.Tensor] = []
    crop_parts: list[torch.Tensor] = []
    trace: list[dict[str, Any]] = []
    phase = bundle.system.decoder.phase
    for index, ols_cpu in enumerate(panel["ols"]):
        if variant == "Dclass-oracle":
            if class_directions is None:
                raise ValueError("class directions are required for Dclass-oracle")
            current_phase = class_directions[int(labels[index])] * scale
        elif direction is None:
            current_phase = torch.zeros_like(phase)
        else:
            current_phase = direction * scale
        with torch.no_grad():
            phase.copy_(current_phase.to(device))
        decoded = _decode_ols(bundle, ols_cpu.unsqueeze(0).to(device), panel["plane"], panel["provenance"])
        raw_parts.append(decoded["raw"].detach().cpu())
        decoded_parts.append(decoded["decoded_field"].detach().cpu())
        camera_parts.append(decoded["camera_field"].detach().cpu())
        input_energy.append(_energy(ols_cpu.unsqueeze(0), bundle.system.encoder_plane.pixel_pitch_m).cpu())
        crop_parts.append(decoded["stats"]["crop_retention"].detach().cpu())
        if index < 20 and variant in {"D0", "Dshared+"}:
            for record in decoded["records"]:
                if index == 0:
                    trace.append({"tensor": record.tensor.detach().cpu(), "plane": asdict(record.plane), "provenance": asdict(record.provenance)})
            if index == 0:
                trace.append({"tensor": decoded["camera_field"].detach().cpu(), "plane": asdict(decoded["camera_field"].plane), "provenance": asdict(decoded["camera_field"].provenance)})
    phase_only_unit_error = float(
        (torch.polar(torch.ones_like(current_phase), current_phase).abs() - 1.0).abs().max()
    )
    _zero_phase(bundle)
    raw = torch.cat(raw_parts)
    decoded_fields = torch.cat(decoded_parts)
    camera_fields = torch.cat(camera_parts)
    base_values = {
        "raw": baseline["raw"] if baseline else raw,
        "view": baseline["view"] if baseline else fixed_exposure_view(raw, exposure_scale),
        "codes": codes,
        "input_energy": torch.cat(input_energy),
        "crop_retention": torch.cat(crop_parts),
    }
    baseline_decoded = baseline.get("decoded_fields") if baseline else None
    metric, values = _evaluator_metrics(context, raw, labels, variant, exposure_scale, base_values, scale if direction is not None else 1.0, trace, decoded_fields, baseline_decoded)
    values["codes"] = codes.cpu()
    values["labels"] = labels.cpu()
    values["decoded_fields"] = decoded_fields
    values["camera_fields"] = camera_fields
    metric["phase_only_unit_error"] = phase_only_unit_error
    return metric, values


def _save_artifacts(root: Path, variant: str, panel: int, values: dict[str, Any], *, save_trace: bool) -> dict[str, Any]:
    directory = root / "measurement" / "artifacts" / variant / str(panel)
    directory.mkdir(parents=True, exist_ok=True)
    refs: dict[str, Any] = {}
    for name in ("labels", "codes", "fixed_cmos", "primary_prediction", "secondary_prediction", "secondary_correctness"):
        path = directory / f"{name}.pt"
        atomic_torch_save(path, values[name].cpu())
        refs[name] = _ref(path, root, values[name].cpu())
    if save_trace:
        path = directory / "representative_trace.pt"
        atomic_torch_save(path, values["representative_trace"])
        refs["representative_trace"] = _ref(path, root)
    return refs


def _bootstrap(correctness: dict[str, dict[int, torch.Tensor]], baseline: str, variants: tuple[str, ...]) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(90210)
    result: dict[str, Any] = {}
    for variant in variants:
        deltas = [correctness[variant][panel] - correctness[baseline][panel] for panel in PANELS]
        aggregate = torch.cat(deltas)
        indices = torch.randint(aggregate.numel(), (BOOTSTRAP_REPLICATES, aggregate.numel()), generator=generator)
        sampled = aggregate[indices].mean(1)
        result[variant] = {
            "baseline": baseline,
            "observed_gain": float(aggregate.mean()),
            "ci95": [float(torch.quantile(sampled, 0.025)), float(torch.quantile(sampled, 0.975))],
            "panel_observed_gain": {str(panel): float(delta.mean()) for panel, delta in zip(PANELS, deltas, strict=True)},
            "panel_ci95": {str(panel): _bootstrap_delta(delta, generator) for panel, delta in zip(PANELS, deltas, strict=True)},
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": 90210,
        }
    return result


def _bootstrap_delta(delta: torch.Tensor, generator: torch.Generator) -> list[float]:
    indices = torch.randint(delta.numel(), (BOOTSTRAP_REPLICATES, delta.numel()), generator=generator)
    values = delta[indices].mean(1)
    return [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))]


def _validate_artifacts(root: Path, payload: dict[str, Any]) -> None:
    for variant in payload["variants"]:
        for panel in payload["panels"]:
            refs = payload["rows"][variant][str(panel)]["artifacts"]
            for name in ("labels", "codes", "fixed_cmos", "primary_prediction", "secondary_prediction", "secondary_correctness"):
                reference = refs.get(name)
                if not isinstance(reference, dict):
                    raise ValueError(f"missing decoder artifact: {variant}/{panel}/{name}")
                value = safe_torch_load(_resolve(reference, root), map_location="cpu")
                if reference.get("tensor_sha256") != _tensor_hash(value):
                    raise ValueError(f"decoder tensor hash mismatch: {variant}/{panel}/{name}")


def _load_measurement(path: Path, *, require_current: bool) -> tuple[Path, dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("schema") != "fiber_decoder_controllability_measurement.v1" or payload.get("gate") != "valid":
        raise ValueError("invalid decoder controllability measurement")
    _validate_artifacts(path.parent.parent, payload)
    if require_current and (_dirty() or payload.get("git_commit") != _git()):
        raise ValueError("decoder measurement is stale or worktree is dirty")
    return path.parent.parent, payload


def _classification(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    bootstrap = payload["bootstrap"]
    shared = bootstrap["Dshared+"]
    class_oracle = bootstrap["Dclass-oracle"]
    shared_panels = [
        panel for panel in PANELS
        if rows["Dshared+"][str(panel)]["secondary_min_recall"] > rows["D0"][str(panel)]["secondary_min_recall"]
        and rows["Dshared+"][str(panel)]["secondary_min_recall"] > 0.0
    ]
    oracle_panels = [
        panel for panel in PANELS
        if rows["Dclass-oracle"][str(panel)]["secondary_min_recall"] > rows["D0"][str(panel)]["secondary_min_recall"]
        and rows["Dclass-oracle"][str(panel)]["secondary_min_recall"] > 0.0
    ]
    shared_gate = (
        shared["observed_gain"] >= 0.015
        and shared["ci95"][0] > 0.0
        and sum(value > 0.0 for value in shared["panel_observed_gain"].values()) >= 2
        and len(shared_panels) >= 2
        and all(rows["Dshared+"][str(panel)]["primary_accuracy"] >= rows["D0"][str(panel)]["primary_accuracy"] - 0.02 for panel in PANELS)
        and max(rows["Dshared+"][str(panel)]["saturation"] for panel in PANELS) <= SATURATION_LIMIT
        and payload["design"]["local_consistency"]["cosine"] >= LOCAL_COSINE_LIMIT
        and payload["design"]["local_consistency"]["odd_error"] <= LOCAL_ODD_ERROR_LIMIT
    )
    controls_reject = all(
        bootstrap[variant]["observed_gain"] < 0.015 or bootstrap[variant]["ci95"][0] <= 0.0
        for variant in ("Dshared-", "Dscrambled")
    )
    conflict = (
        not shared_gate
        and class_oracle["observed_gain"] >= 0.015
        and class_oracle["ci95"][0] > 0.0
        and len(oracle_panels) >= 2
        and payload["design"]["gradient_conflict"]["negative_fraction_below_minus_0_10"] >= 0.30
    )
    local_response = payload["design"]["candidate_selected"] or payload["design"]["local_consistency"]["cosine"] >= LOCAL_COSINE_LIMIT
    if shared_gate and controls_reject:
        conclusion = "shared_phase_only_decoder_recovery_candidate"
    elif conflict:
        conclusion = "fixed_decoder_class_gradient_conflict"
    elif not local_response or class_oracle["observed_gain"] < 0.015:
        conclusion = "zero_phase_decoder_local_reachability_failed"
    else:
        conclusion = "decoder_controllability_inconclusive"
    return {
        "conclusion": conclusion,
        "shared_gate": shared_gate,
        "controls_reject": controls_reject,
        "class_gradient_conflict": conflict,
        "shared_recovered_panels": shared_panels,
        "class_oracle_recovered_panels": oracle_panels,
        "shared_bootstrap": shared,
        "class_oracle_bootstrap": class_oracle,
        "gradient_conflict": payload["design"]["gradient_conflict"],
        "local_consistency": payload["design"]["local_consistency"],
    }


def run(args: Any) -> int:
    validation_path = _require_file(args, "validation_evidence")
    validation = _load_json(validation_path)
    if validation.get("schema") != "fiber_decoder_controllability_validation.v1" or validation.get("gate") != "valid":
        raise ValueError("invalid decoder controllability validation")
    if _dirty() or validation.get("git_commit") != _git():
        raise ValueError("decoder validation is stale or worktree is dirty")
    root = _run_root(args.output, require_new=False)
    for directory in (root / "design", root / "measurement"):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"decoder run output must be new: {directory}")
    context = _context(args, validation_path.parent)
    config, device, _, _, _, _, _, _, bundle, _, _, _ = context
    exposure_scale = float(validation["fixed_exposure_scale"])
    before_state = _parameter_state(bundle)
    design = _design(context, root, validation_path, exposure_scale)
    directions = safe_torch_load(_resolve(design["direction_artifact"], root), map_location="cpu")
    shared = directions["shared_direction"].to(device)
    scrambled = directions["scrambled_direction"].to(device)
    class_directions = directions["class_gradients"].to(device)
    rows: dict[str, dict[str, Any]] = {variant: {} for variant in VARIANTS}
    correctness: dict[str, dict[int, torch.Tensor]] = {variant: {} for variant in VARIANTS}
    pairing: dict[str, dict[str, dict[str, str]]] = {"labels": {variant: {} for variant in VARIANTS}, "codes": {variant: {} for variant in VARIANTS}}
    baseline_cache: dict[int, dict[str, Any]] = {}
    for panel_seed in PANELS:
        labels, codes = _labels_codes(panel_seed, device)
        panel = _collect_ols_panel(context, labels, codes)
        panel["baseline_raw"] = panel["full_raw"]
        baseline_metric, baseline_values = _measure_variant(context, panel, labels, codes, "D0", None, 0.0, None, exposure_scale, None)
        baseline_metric["full_forward_replay_raw_l2"] = interface._relative_l2(baseline_values["raw"], panel["full_raw"])
        baseline_metric["full_forward_replay_complex_l2"] = interface._relative_l2(
            baseline_values["camera_fields"], panel["full_camera"]
        )
        baseline_cache[panel_seed] = {"raw": baseline_values["raw"], "view": baseline_values["fixed_cmos"], "decoded_fields": baseline_values["decoded_fields"]}
        branch_specs = {
            "D0": (None, 0.0, None),
            "Dshared+": (shared, float(design["selected_scale"]), None),
            "Dshared-": (-shared, float(design["selected_scale"]), None),
            "Dscrambled": (scrambled, float(design["selected_scale"]), None),
            "Dclass-oracle": (None, float(design["selected_scale"]), class_directions),
        }
        for variant, (direction, scale, class_dirs) in branch_specs.items():
            if variant == "D0":
                metric, values = baseline_metric, baseline_values
            else:
                metric, values = _measure_variant(context, panel, labels, codes, variant, direction, scale, class_dirs, exposure_scale, baseline_cache[panel_seed])
            refs = _save_artifacts(root, variant, panel_seed, values, save_trace=variant in {"D0", "Dshared+"})
            metric["artifacts"] = refs
            rows[variant][str(panel_seed)] = metric
            correctness[variant][panel_seed] = values["secondary_correctness"].float()
            pairing["labels"][variant][str(panel_seed)] = _tensor_hash(values["labels"])
            pairing["codes"][variant][str(panel_seed)] = _tensor_hash(values["codes"])
    bootstrap = _bootstrap(correctness, "D0", ("Dshared+", "Dshared-", "Dscrambled", "Dclass-oracle"))
    after_state = _parameter_state(bundle)
    pairing_labels = all(len({pairing["labels"][variant][str(panel)] for variant in VARIANTS}) == 1 for panel in PANELS)
    pairing_codes = all(len({pairing["codes"][variant][str(panel)] for variant in VARIANTS}) == 1 for panel in PANELS)
    loss_measurement = _load_json(_require_file(args, "loss_exposure_measurement"))
    loss_measurement_root = Path(args.loss_exposure_measurement).parent
    baseline_replay = all(rows["D0"][str(panel)]["raw_l2_vs_d0"] <= 1.0e-5 for panel in PANELS)
    loss_exposure_reproduction = True
    for panel in PANELS:
        old_reference = loss_measurement["rows"]["S0-Ecal"][str(panel)]["artifacts"]["fixed_cmos"]
        old_view = safe_torch_load(exposure._resolve(old_reference, loss_measurement_root), map_location="cpu")
        current_reference = rows["D0"][str(panel)]["artifacts"]["fixed_cmos"]
        current_view = safe_torch_load(_resolve(current_reference, root), map_location="cpu")
        loss_exposure_reproduction = loss_exposure_reproduction and interface._relative_l2(current_view, old_view) <= 1.0e-5
    harness = {
        "baseline_replay": baseline_replay and all(
            rows["D0"][str(panel)]["full_forward_replay_raw_l2"] <= 1.0e-5
            and rows["D0"][str(panel)]["full_forward_replay_complex_l2"] <= 1.0e-5
            for panel in PANELS
        ),
        "loss_exposure_baseline_reproduction": loss_exposure_reproduction,
        "pairing_labels": pairing_labels,
        "pairing_codes": pairing_codes,
        "candidate_selection_secondary_free": design["secondary_used_for_selection"] is False and design["locked_metrics_used_for_selection"] is False,
        "local_consistency": design["local_consistency"]["cosine"] >= LOCAL_COSINE_LIMIT and design["local_consistency"]["odd_error"] <= LOCAL_ODD_ERROR_LIMIT,
        "control_phase_rms_match": design["control_phase_rms_match_error"] <= 1.0e-6,
        "phase_only_finite": all(rows[variant][str(panel)]["decoded_finite"] for variant in VARIANTS for panel in PANELS),
        "phase_only_unit_amplitude": all(rows[variant][str(panel)]["phase_only_unit_error"] <= 1.0e-6 for variant in VARIANTS for panel in PANELS),
        "physical_finite_nonnegative": all(rows[variant][str(panel)]["raw_finite"] and rows[variant][str(panel)]["raw_nonnegative"] and rows[variant][str(panel)]["passive"] for variant in PHYSICAL_VARIANTS for panel in PANELS),
        "physical_saturation": all(rows[variant][str(panel)]["saturation"] <= SATURATION_LIMIT for variant in PHYSICAL_VARIANTS for panel in PANELS),
        "shared_mask_fixed": True,
        "parameter_isolation": before_state == after_state,
        "no_optimizer": True,
    }
    storage = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    harness["storage"] = storage <= STORAGE_LIMIT_BYTES
    payload = {
        "schema": "fiber_decoder_controllability_measurement.v1",
        "gate": "valid" if all(harness.values()) else "invalid_harness",
        "git_commit": _git(),
        "validation_sha256": file_sha256(validation_path),
        "design_sha256": file_sha256(root / "design" / "direction_design.json"),
        "variants": list(VARIANTS),
        "panels": list(PANELS),
        "design_panel": DESIGN_PANEL,
        "fixed_exposure_scale": exposure_scale,
        "selected_phase_rms": design["selected_scale"],
        "design": design,
        "rows": rows,
        "bootstrap": bootstrap,
        "pairing": pairing,
        "parameter_state_before": before_state,
        "parameter_state_after": after_state,
        "harness_gates": harness,
        "storage_bytes": storage,
        "storage_limit_bytes": STORAGE_LIMIT_BYTES,
        "status": "synthetic diagnostic-only",
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    _write(Path(args.output), payload)
    return 0 if payload["gate"] == "valid" else 2


def report(args: Any) -> int:
    measurement_path = _require_file(args, "measurement_evidence")
    root, payload = _load_measurement(measurement_path, require_current=False)
    output_root = _output_directory(args.output, require_new=True)
    classification = _classification(payload)
    with (output_root / "decoder_controllability_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "panel", "primary_accuracy", "secondary_accuracy", "secondary_min_recall", "primary_macro_ce", "secondary_macro_ce", "raw_l2_vs_d0", "saturation", "decoder_energy_retention"])
        for variant in payload["variants"]:
            for panel in payload["panels"]:
                row = payload["rows"][variant][str(panel)]
                writer.writerow([variant, panel, row["primary_accuracy"], row["secondary_accuracy"], row["secondary_min_recall"], row["primary_macro_ce"], row["secondary_macro_ce"], row["raw_l2_vs_d0"], row["saturation"], row["decoded_energy_retention_mean"]])
    _write(output_root / "classification.json", classification)
    _write(output_root / "presentation.json", {
        "schema": "fiber_decoder_controllability_presentation.v1",
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
    root, payload = _load_measurement(measurement_path, require_current=True)
    presentation = _load_json(presentation_path)
    if presentation.get("schema") != "fiber_decoder_controllability_presentation.v1" or presentation.get("measurement_sha256") != file_sha256(measurement_path) or presentation.get("source_measurement_git_commit") != payload["git_commit"]:
        raise ValueError("decoder presentation is not bound to measurement")
    classification = _classification(payload)
    actions = {
        "shared_phase_only_decoder_recovery_candidate": "register_generator_decoder_codesign_preflight",
        "fixed_decoder_class_gradient_conflict": "register_generator_manifold_reorientation_attribution",
        "zero_phase_decoder_local_reachability_failed": "close_current_fiber_route_no_e1",
        "decoder_controllability_inconclusive": "hold_no_training_request_new_hypothesis",
    }
    storage = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if storage > STORAGE_LIMIT_BYTES:
        raise ValueError("decoder controllability storage gate failed")
    _write(Path(args.output), {
        "schema": "fiber_decoder_controllability_closeout.v1",
        "gate": "valid",
        "conclusion": classification["conclusion"],
        "next_action": actions[classification["conclusion"]],
        "measurement_sha256": file_sha256(measurement_path),
        "presentation_sha256": file_sha256(presentation_path),
        "validation_sha256": payload["validation_sha256"],
        "design_sha256": payload["design_sha256"],
        "storage_bytes": storage,
        "storage_limit_bytes": STORAGE_LIMIT_BYTES,
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
    })
    return 0


def dispatch(args: Any) -> int:
    return {"validate": validate, "run": run, "report": report, "closeout": closeout}[args.research_action](args)
