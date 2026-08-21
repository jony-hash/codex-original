"""No-update attribution and compact presentation artifacts for synthetic fiber research."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ...evidence import atomic_torch_save, atomic_write_json, file_sha256, safe_torch_load
from ...optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from ...optics.fiber_pool import FiberTransportPool
from ...student.conditional import camera_code_fraction, fixed_exposure_view, relative_diversity_ratios
from ...student.conditional_acceptance import conditional_coverage_ratio
from .._runtime_shared import random_codes
from . import fiber_image_generation as image
from . import fiber_robustness as robust

VARIANTS = ("I0", "S0", "P1", "L1", "O1", "M1", "S0-D0", "M1-D0")
PANELS = (4242, 104729, 161803)
DIGIT_PIPELINE_STAGES = (
    ("slm_output", "SLM output"),
    ("encoder_input", "Encoder input / front 4f"),
    ("encoder_layer_1", "Encoder layer 1"),
    ("encoder_layer_2", "Encoder layer 2 / fiber input"),
    ("ols_output", "LP01 fiber output"),
    ("decoder_layer_1", "Decoder layer 1"),
    ("decoder_layer_2", "Decoder layer 2"),
    ("camera", "Camera incident field"),
    ("fixed_cmos", "Fixed-exposure CMOS"),
)


def _root(path: str | Path) -> Path:
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _require(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_file():
        raise ValueError(f"missing attribution input: {value}")
    return value


def _v3(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "fiber_image_closeout.v2" or payload.get("gate") != "valid_failed":
        raise ValueError("attribution requires structurally valid v3 closeout")
    if payload.get("conclusion") != "no_reproducible_fiber_gain" or payload.get("formal_promotion") is not False:
        raise ValueError("attribution is only authorized for the recorded negative v3 result")
    return payload


def _context_args(args: Any, root: Path) -> SimpleNamespace:
    values = vars(args).copy()
    values["validate_evidence"] = str(root / "fiber_image_validation.json")
    values["measurement_bundle"] = str(root / "measurement_bundle.json")
    values["measurement_evidence"] = str(root / "measurement_evidence.json")
    return SimpleNamespace(**values)


def validate(args: Any) -> int:
    root = _root(args.output)
    if image._dirty():
        _write(root / "attribution_validation.json", {"schema": "fiber_attribution_validation.v1", "gate": "invalid_harness", "reason": "dirty worktree"})
        return 2
    _v3(_require(args.v3_closeout))
    upstream = _context_args(args, root)
    upstream.output = str(root / "fiber_image_validation.json")
    upstream.measurement_bundle_output = str(root / "measurement_bundle.json")
    upstream.measurement_evidence_output = str(root / "measurement_evidence.json")
    image.validate(upstream)
    validation = json.loads((root / "fiber_image_validation.json").read_text(encoding="utf-8"))
    if validation.get("gate") != "valid":
        raise ValueError("fiber physical/evaluator validation failed")
    _write(root / "attribution_validation.json", {
        "schema": "fiber_attribution_validation.v1",
        "gate": "valid",
        "git_commit": image._git(),
        "v3_closeout_sha256": file_sha256(args.v3_closeout),
        "validation_sha256": file_sha256(root / "fiber_image_validation.json"),
        "measurement_bundle_sha256": file_sha256(root / "measurement_bundle.json"),
        "measurement_evidence_sha256": file_sha256(root / "measurement_evidence.json"),
        "config_hash": validation.get("config_hash"),
        "dataset_split_manifest_hash": validation.get("dataset_split_manifest_hash"),
        "pool_hash": validation.get("pool_hash"),
        "variants": list(VARIANTS),
        "panels": list(PANELS),
        "samples_per_panel": 500,
        "representative_samples_per_class": 2,
        "status": "synthetic diagnostic-only",
    })
    return 0


def _set_variant(bundle: Any, pool: FiberTransportPool, variant: str, device: torch.device) -> None:
    if variant == "I0":
        bundle.system.ols = robust._IdentityOLSModule()
        return
    record = pool.record("train-01")
    diagonal = torch.diagonal(pool.matrix_by_id(record.matrix_id)).to(device)
    if variant in {"S0", "S0-D0"}:
        diagonal, offset = torch.ones_like(diagonal), (0.0, 0.0)
    elif variant == "P1":
        diagonal, offset = torch.exp(1j * torch.angle(diagonal)), (0.0, 0.0)
    elif variant == "L1":
        diagonal, offset = diagonal.abs().to(torch.complex64), (0.0, 0.0)
    elif variant == "O1":
        diagonal, offset = torch.ones_like(diagonal), record.offset_m
    elif variant in {"M1", "M1-D0"}:
        offset = record.offset_m
    else:
        raise ValueError(f"unknown attribution variant: {variant}")
    bundle.system.ols = FiberArrayOLS(
        bundle.system.encoder_plane,
        FiberArraySpec(lateral_offset_m=offset),
        torch.diag(diagonal),
    ).to(device)


def _decoder_neutral(bundle: Any, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    saved = bundle.system.decoder.phase.detach().clone()
    with torch.no_grad():
        bundle.system.decoder.phase.zero_()
    return saved


def _restore_decoder(bundle: Any, saved: torch.Tensor | None) -> None:
    if saved is not None:
        with torch.no_grad():
            bundle.system.decoder.phase.copy_(saved)


def _trace(bundle: Any, generator: Any, codes: torch.Tensor, labels: torch.Tensor) -> list[dict[str, Any]]:
    with torch.inference_mode():
        command = generator(codes, labels)
        result = bundle.system.forward(bundle.system.command(command), return_optical_trace=True)
    rows = []
    for index in range(codes.shape[0]):
        rows.append({
            "command": command[index:index + 1].detach().cpu(),
            "fields": [
                {"tensor": field.tensor[index:index + 1].detach().cpu(), "plane": asdict(field.plane), "provenance": asdict(field.provenance)}
                for field in (result.optical_trace or ())
            ],
        })
    return rows


def _measure(context: tuple[Any, ...], variant: str, panel: int, root: Path) -> dict[str, Any]:
    config, device, _, _, reference_images, reference_labels, primary, secondary, bundle, pool, _, _ = context
    settings = config.conditional_generation
    labels = torch.arange(10, device=device).repeat_interleave(50)
    codes = random_codes(500, device=device, seed=panel)
    original_ols = bundle.system.ols
    _set_variant(bundle, pool, variant.replace("-D0", ""), device)
    saved_decoder = _decoder_neutral(bundle, variant.endswith("-D0"))
    raw_parts, view_parts, primary_parts, secondary_parts, feature_parts = [], [], [], [], []
    input_energy, output_energy, port_in, port_out = [], [], [], []
    try:
        for start in range(0, 500, config.execution.evaluation_microbatch_size):
            c = codes[start:start + config.execution.evaluation_microbatch_size]
            y = labels[start:start + c.shape[0]]
            with torch.inference_mode():
                command = bundle.generator(c, y)
                result = bundle.system.forward(bundle.system.command(command), return_optical_trace=False)
                raw = result.raw_camera.tensor
                view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
                _, pl = primary(view)
                sf, sl = secondary(view)
            raw_parts.append(raw.detach().cpu())
            view_parts.append(view.detach().cpu())
            primary_parts.append(pl.detach().cpu())
            secondary_parts.append(sl.detach().cpu())
            feature_parts.append(sf.detach().cpu())
            if variant != "I0":
                diagnostics = bundle.system.ols.last_diagnostics
                input_energy.append(diagnostics["input_energy"].detach().cpu())
                output_energy.append(diagnostics["output_energy"].detach().cpu())
                port_in.append(diagnostics["port_input_coefficients"].detach().cpu())
                port_out.append(diagnostics["port_output_coefficients"].detach().cpu())
    finally:
        _restore_decoder(bundle, saved_decoder)
        bundle.system.ols = original_ols
    raw = torch.cat(raw_parts)
    view = torch.cat(view_parts)
    primary_prediction = torch.cat(primary_parts).argmax(1)
    secondary_prediction = torch.cat(secondary_parts).argmax(1)
    features = torch.cat(feature_parts)
    reference_indices = torch.cat([torch.where(reference_labels == label)[0][:50] for label in range(10)])
    with torch.inference_mode():
        reference_features, _ = secondary(reference_images.index_select(0, reference_indices))
    metric = {
        "variant": variant,
        "panel": panel,
        "labels": labels.cpu(),
        "codes": codes.cpu(),
        "fixed_cmos": view,
        "primary_prediction": primary_prediction,
        "secondary_prediction": secondary_prediction,
        "secondary_correctness": secondary_prediction == labels.cpu(),
        "labels_sha256": image._tensor_content_hash(labels.cpu()),
        "codes_sha256": image._tensor_content_hash(codes.cpu()),
        "primary_accuracy": float((primary_prediction == labels.cpu()).float().mean()),
        "secondary_accuracy": float((secondary_prediction == labels.cpu()).float().mean()),
        "primary_min_recall": min(float((primary_prediction[labels.cpu() == label] == label).float().mean()) for label in range(10)),
        "secondary_min_recall": min(float((secondary_prediction[labels.cpu() == label] == label).float().mean()) for label in range(10)),
        "agreement": float((primary_prediction == secondary_prediction).float().mean()),
        "finite": bool(torch.isfinite(raw).all()),
        "nonnegative": bool((raw >= 0).all()),
        "saturation": float((raw > settings.fixed_exposure_scale).float().mean()),
        "camera_fraction": float(camera_code_fraction(raw, labels.cpu()).item()),
        "minimum_diversity": min(relative_diversity_ratios(features, reference_features.cpu(), labels.cpu())),
        "coverage_ratio": float(conditional_coverage_ratio(features.to(device), view.to(device), reference_features, reference_images.index_select(0, reference_indices), labels, epsilon=settings.sinkhorn_epsilon, iterations=settings.sinkhorn_iterations, seed=2027)),
        "passive_available": variant != "I0",
    }
    if variant != "I0":
        ie, oe = torch.cat(input_energy), torch.cat(output_energy)
        metric.update({
            "passive": bool(torch.all(oe <= ie * (1 + 1e-5) + 1e-12)),
            "input_energy_max": float(ie.max()),
            "output_energy_max": float(oe.max()),
            "port_input_coefficients": torch.cat(port_in),
            "port_output_coefficients": torch.cat(port_out),
        })
    representatives = torch.cat([torch.where(labels.cpu() == label)[0][:2] for label in range(10)])
    _set_variant(bundle, pool, variant.replace("-D0", ""), device)
    saved_decoder = _decoder_neutral(bundle, variant.endswith("-D0"))
    try:
        metric["representative_trace"] = _trace(
            bundle, bundle.generator, codes.index_select(0, representatives.to(device)), labels.index_select(0, representatives.to(device))
        )
    finally:
        _restore_decoder(bundle, saved_decoder)
        bundle.system.ols = original_ols
    return metric


def _save_metric(metric: dict[str, Any], root: Path) -> dict[str, Any]:
    directory = root / "measurements" / metric["variant"] / str(metric["panel"])
    directory.mkdir(parents=True, exist_ok=True)
    references = {}
    for name in ("labels", "codes", "fixed_cmos", "primary_prediction", "secondary_prediction", "secondary_correctness"):
        path = directory / f"{name}.pt"
        atomic_torch_save(path, metric[name])
        references[name] = {"relative_path": path.relative_to(root).as_posix(), "sha256": file_sha256(path), "tensor_sha256": image._tensor_content_hash(metric[name])}
    trace_path = directory / "representative_trace.pt"
    atomic_torch_save(trace_path, metric["representative_trace"])
    references["representative_trace"] = {"relative_path": trace_path.relative_to(root).as_posix(), "sha256": file_sha256(trace_path)}
    summary = {key: value for key, value in metric.items() if isinstance(value, (str, int, float, bool)) or value is None}
    summary["artifacts"] = references
    _write(directory / "measurement.json", summary)
    return summary


def _bootstrap(rows: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    pairs = {"coupling": ("I0", "S0"), "phase": ("P1", "S0"), "loss": ("L1", "S0"), "offset": ("O1", "S0"), "decoder": ("M1-D0", "M1")}
    generator = torch.Generator(device="cpu").manual_seed(90210)
    output = {}
    for stage, (variant, base) in pairs.items():
        deltas = [rows[variant][panel]["secondary_correctness"].float() - rows[base][panel]["secondary_correctness"].float() for panel in PANELS]
        observed = float(torch.cat(deltas).mean())
        samples = []
        for _ in range(10000):
            panel_means = []
            for delta in deltas:
                index = torch.randint(delta.numel(), (delta.numel(),), generator=generator)
                panel_means.append(delta[index].mean())
            samples.append(torch.stack(panel_means).mean())
        values = torch.stack(samples)
        output[stage] = {"observed_effect": observed, "ci95": [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))], "seed": 90210, "replicates": 10000}
    directions = {"coupling": 1.0, "phase": -1.0, "loss": -1.0, "offset": -1.0, "decoder": -1.0}
    ranked = sorted(((stage, max(0.0, directions[stage] * value["observed_effect"])) for stage, value in output.items()), key=lambda item: item[1], reverse=True)
    dominant = ranked[0][0] if len(ranked) > 1 and ranked[0][1] - ranked[1][1] >= 0.015 else None
    output["classification"] = {"conclusion": f"{dominant}_dominant_observed_drop" if dominant else "distributed_or_inconclusive", "ranked_effects": ranked}
    return output


def run(args: Any) -> int:
    root = _root(args.output)
    validation = json.loads(_require(args.validation_evidence).read_text(encoding="utf-8"))
    if validation.get("schema") != "fiber_attribution_validation.v1" or validation.get("gate") != "valid":
        raise ValueError("invalid attribution validation")
    _v3(_require(args.v3_closeout))
    context = image._context(_context_args(args, Path(args.validation_evidence).parent))
    rows: dict[str, dict[int, dict[str, Any]]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for panel in PANELS:
            rows[variant][panel] = _measure(context, variant, panel, root)
            _save_metric(rows[variant][panel], root)
    bootstrap = _bootstrap(rows)
    summary = {variant: {str(panel): _save_metric(rows[variant][panel], root) for panel in PANELS} for variant in VARIANTS}
    storage = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    payload = {
        "schema": "fiber_attribution_measurement.v1",
        "gate": "valid" if storage <= 2 * 1024**3 else "invalid_harness",
        "git_commit": image._git(),
        "validation_sha256": file_sha256(args.validation_evidence),
        "v3_closeout_sha256": file_sha256(args.v3_closeout),
        "variants": list(VARIANTS),
        "panels": list(PANELS),
        "rows": summary,
        "bootstrap": bootstrap,
        "storage_bytes": storage,
        "storage_limit_bytes": 2 * 1024**3,
        "status": "synthetic diagnostic-only",
    }
    _write(root / "measurement.json", payload)
    return 0 if payload["gate"] == "valid" else 2


def _load_measurement(args: Any, *, report_only: bool = False) -> tuple[Path, dict[str, Any]]:
    evidence = _require(args.measurement_evidence)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    if payload.get("schema") != "fiber_attribution_measurement.v1" or payload.get("gate") != "valid":
        raise ValueError("invalid attribution measurement")
    if image._dirty():
        raise ValueError("measurement is stale or worktree is dirty")
    if payload.get("git_commit") != image._git() and not report_only:
        raise ValueError("measurement is stale or worktree is dirty")
    return evidence.parent, payload


def _load_tensor(root: Path, reference: dict[str, Any]) -> torch.Tensor:
    path = (root / reference["relative_path"]).resolve()
    if root.resolve() not in path.parents or file_sha256(path) != reference["sha256"]:
        raise ValueError("attribution artifact hash mismatch")
    return safe_torch_load(path, expected_sha256=reference["sha256"], map_location="cpu")


def _trace_stage_rows(root: Path, row: dict[str, Any], variant: str) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    labels = _load_tensor(root, row["artifacts"]["labels"])
    codes = _load_tensor(root, row["artifacts"]["codes"])
    trace = safe_torch_load(
        (root / row["artifacts"]["representative_trace"]["relative_path"]).resolve(),
        expected_sha256=row["artifacts"]["representative_trace"]["sha256"],
        map_location="cpu",
    )
    if len(trace) != 20 or labels.numel() != 500 or codes.shape[0] != 500:
        raise ValueError("digit pipeline requires the complete 20-trace and 500-sample S0 panel")
    actual = [field["plane"]["plane_id"] for field in trace[0]["fields"]]
    expected = [name for name, _ in DIGIT_PIPELINE_STAGES[:-1]]
    if actual != expected:
        raise ValueError(f"unexpected optical trace stage order for {variant}: {actual}")
    rows = []
    for label in range(10):
        trace_index = label * 2
        sample_index = label * 50
        if int(labels[sample_index]) != label or int(labels[sample_index + 1]) != label:
            raise ValueError("digit trace is not aligned with the panel's fixed class ordering")
        rows.append({
            "label": label,
            "sample_index": sample_index,
            "trace_index": trace_index,
            "trace": trace[trace_index],
            "code_sha256": image._tensor_content_hash(codes[sample_index]),
        })
    return rows, labels, codes


def _save_figure_formats(fig: Any, output: Path, stem: str, *, dpi: int = 300) -> list[dict[str, Any]]:
    references = []
    for suffix in ("png", "svg", "pdf"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        references.append({"relative_path": path.name, "sha256": file_sha256(path)})
    return references


def _stage_display_title(stage_id: str, title: str, field: dict[str, Any] | None) -> str:
    if stage_id == "fixed_cmos":
        return f"{title}\n28×28 · fixed exposure"
    plane = field["plane"]
    shape = "×".join(str(value) for value in plane["shape"])
    pitch = "×".join(f"{value * 1e6:g}" for value in plane["pixel_pitch_m"])
    return f"{title}\n{shape} · {pitch} µm\n{plane['plane_id']}"


def _digit_pipeline_report(root: Path, output: Path, variant: str) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    measurement_payload = json.loads((root / "measurement.json").read_text(encoding="utf-8"))
    row = measurement_payload["rows"][variant][str(PANELS[0])]
    digit_rows, labels, codes = _trace_stage_rows(root, row, variant)
    fixed_cmos = _load_tensor(root, row["artifacts"]["fixed_cmos"])
    fields = [item["trace"]["fields"] for item in digit_rows]
    amplitudes = [field["tensor"][0, 0].abs().numpy() for digit in fields for field in digit]
    amplitudes.extend(fixed_cmos[digit["sample_index"], 0].numpy() for digit in digit_rows)
    amplitude_vmax = float(max(np.max(value) for value in amplitudes))
    if not np.isfinite(amplitude_vmax) or amplitude_vmax <= 0:
        raise ValueError("invalid global amplitude display scale")
    output.mkdir(parents=True, exist_ok=True)
    stage_titles = [title for _, title in DIGIT_PIPELINE_STAGES]
    manifest: dict[str, Any] = {
        "schema": "fiber_digit_pipeline_presentation.v1",
        "gate": "valid",
        "variant": variant,
        "panel": PANELS[0],
        "source_measurement_git_commit": measurement_payload.get("git_commit"),
        "report_git_commit": image._git(),
        "sample_policy": "first sample of each class in fixed panel ordering",
        "input_contract": "digit label 0-9 plus fixed 8D code; not a direct MNIST image input",
        "amplitude_vmax_global": amplitude_vmax,
        "phase_range": [-float(np.pi), float(np.pi)],
        "cmos_phase": "N/A — intensity-only detector",
        "digits": [],
    }

    for digit in digit_rows:
        label = digit["label"]
        trace_fields = digit["trace"]["fields"]
        values = [field["tensor"][0, 0] for field in trace_fields]
        values.append(fixed_cmos[digit["sample_index"], 0].to(torch.complex64))
        display_titles = [_stage_display_title(stage_id, stage_title, trace_fields[index] if index < len(trace_fields) else None) for index, (stage_id, stage_title) in enumerate(DIGIT_PIPELINE_STAGES)]
        for mode in ("amplitude", "phase"):
            fig, axes = plt.subplots(1, len(DIGIT_PIPELINE_STAGES), figsize=(18, 3.0), squeeze=False)
            for index, (stage_id, stage_title) in enumerate(DIGIT_PIPELINE_STAGES):
                axis = axes[0, index]
                if mode == "phase" and stage_id == "fixed_cmos":
                    axis.text(0.5, 0.5, "N/A\nintensity-only\ndetector", ha="center", va="center", fontsize=10)
                    axis.set_facecolor("#eeeeee")
                else:
                    image_value = values[index].abs().numpy() if mode == "amplitude" else values[index].angle().numpy()
                    axis.imshow(image_value, cmap="gray" if mode == "amplitude" else "twilight", vmin=0 if mode == "amplitude" else -np.pi, vmax=amplitude_vmax if mode == "amplitude" else np.pi)
                axis.set_title(display_titles[index], fontsize=7)
                axis.axis("off")
            fig.suptitle(f"Digit {label} · {mode} · Synthetic SMF-LP01 {variant}, V=2.2, unit transport, zero offset", fontsize=12)
            fig.text(0.5, 0.01, "Fixed G0; panel 4242; common scale across digits and stages; diagnostic-only synthetic proxy", ha="center", fontsize=8)
            names = _save_figure_formats(fig, output, f"digit_{label}_{mode}")
            plt.close(fig)
            digit.setdefault("files", {})[mode] = names
        manifest["digits"].append({
            "label": label,
            "panel_sample_index": digit["sample_index"],
            "trace_index": digit["trace_index"],
            "code_sha256": digit["code_sha256"],
            "trace_artifact_sha256": row["artifacts"]["representative_trace"]["sha256"],
            "stages": [{"stage_id": field_id, "title": title, "plane": trace_fields[index]["plane"], "provenance": trace_fields[index]["provenance"]} for index, (field_id, title) in enumerate(DIGIT_PIPELINE_STAGES[:-1])],
            "files": digit["files"],
        })

    display_titles = [_stage_display_title(stage_id, stage_title, fields[0][index] if index < len(fields[0]) else None) for index, (stage_id, stage_title) in enumerate(DIGIT_PIPELINE_STAGES)]

    def overview(mode: str, stem: str) -> list[dict[str, Any]]:
        fig, axes = plt.subplots(10, len(DIGIT_PIPELINE_STAGES), figsize=(20, 15), squeeze=False)
        for row_index, digit in enumerate(digit_rows):
            values = [field["tensor"][0, 0] for field in digit["trace"]["fields"]]
            values.append(fixed_cmos[digit["sample_index"], 0].to(torch.complex64))
            for col, value in enumerate(values):
                axis = axes[row_index, col]
                if mode == "phase" and col == len(values) - 1:
                    axis.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
                    axis.set_facecolor("#eeeeee")
                else:
                    image_value = value.abs().numpy() if mode == "amplitude" else value.angle().numpy()
                    axis.imshow(image_value, cmap="gray" if mode == "amplitude" else "twilight", vmin=0 if mode == "amplitude" else -np.pi, vmax=amplitude_vmax if mode == "amplitude" else np.pi)
                axis.axis("off")
                if row_index == 0:
                    axis.set_title(display_titles[col], fontsize=7)
            axes[row_index, 0].set_ylabel(str(digit["label"]), rotation=0, ha="right", va="center", fontsize=10)
        fig.suptitle(f"Digits 0–9 · {mode} through {variant} synthetic SMF-LP01 · V=2.2", fontsize=16)
        fig.text(0.5, 0.005, "Input: digit label + fixed 8D code; common scale; CMOS phase is unavailable", ha="center", fontsize=9)
        names = _save_figure_formats(fig, output, stem)
        plt.close(fig)
        return names

    manifest["overview_files"] = {
        "amplitude": overview("amplitude", "digits_0_9_amplitude_pipeline"),
        "phase": overview("phase", "digits_0_9_phase_pipeline"),
    }
    # Compact two-row-per-digit view, matching the requested PPT composition.
    fig, axes = plt.subplots(20, len(DIGIT_PIPELINE_STAGES), figsize=(20, 28), squeeze=False)
    for digit_index, digit in enumerate(digit_rows):
        values = [field["tensor"][0, 0] for field in digit["trace"]["fields"]]
        values.append(fixed_cmos[digit["sample_index"], 0].to(torch.complex64))
        for col, value in enumerate(values):
            for offset, mode in enumerate(("amplitude", "phase")):
                axis = axes[digit_index * 2 + offset, col]
                if mode == "phase" and col == len(values) - 1:
                    axis.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=7)
                    axis.set_facecolor("#eeeeee")
                else:
                    image_value = value.abs().numpy() if mode == "amplitude" else value.angle().numpy()
                    axis.imshow(image_value, cmap="gray" if mode == "amplitude" else "twilight", vmin=0 if mode == "amplitude" else -np.pi, vmax=amplitude_vmax if mode == "amplitude" else np.pi)
                axis.axis("off")
                if digit_index == 0 and offset == 0:
                    axis.set_title(stage_titles[col], fontsize=8)
        axes[digit_index * 2, 0].set_ylabel(f"{digit['label']}\namp", rotation=0, ha="right", va="center", fontsize=8)
        axes[digit_index * 2 + 1, 0].set_ylabel("phase", rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(f"Digits 0–9 · amplitude / phase full pipeline · {variant} synthetic SMF-LP01", fontsize=16)
    fig.text(0.5, 0.005, "Common amplitude scale; phase range [−π, π]; CMOS phase N/A", ha="center", fontsize=9)
    manifest["overview_files"]["compact"] = _save_figure_formats(fig, output, "digits_0_9_amplitude_phase_compact")
    plt.close(fig)
    _write(output / "digit_pipeline_manifest.json", manifest)
    return manifest


def report(args: Any) -> int:
    # Report-only consumers may render an immutable historical measurement after
    # a presentation-layer commit; the source Git commit remains in the output
    # manifest and the scientific measurement is never rewritten.
    root, payload = _load_measurement(args, report_only=True)
    out = _root(args.output)
    if args.digit_pipeline_only:
        _digit_pipeline_report(root, out, args.digit_pipeline_variant)
        _write(out / "presentation.json", {"schema": "fiber_digit_pipeline_presentation.v1", "gate": "valid", "measurement_sha256": file_sha256(args.measurement_evidence), "source_measurement_git_commit": payload.get("git_commit"), "report_git_commit": image._git(), "status": "synthetic diagnostic-only"})
        return 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = payload["rows"]
    fig, ax = plt.subplots(figsize=(16, 3.2))
    stages = [
        ("SLM", "200×200\n2.5 µm"),
        ("Encoder", "100×100\n2 µm"),
        ("LP01 coupling", "100 ports\nV=2.2"),
        ("Transport", "complex\ndiagonal"),
        ("Re-emission", "100×100\n2 µm"),
        ("Decoder", "finite crop\n2 layers"),
        ("CMOS", "200×200\nfixed exposure"),
    ]
    for index, (title, detail) in enumerate(stages):
        ax.text(index, 0.5, f"{title}\n{detail}", ha="center", va="center", fontsize=10,
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#d9eaf7" if index not in {2, 3} else "#fce4b2", "edgecolor": "#355c7d"})
        if index < len(stages) - 1:
            ax.annotate("", xy=(index + 0.72, 0.5), xytext=(index + 0.28, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_xlim(-0.6, len(stages) - 0.4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Synthetic SMF optical generation path; all coordinates are diagnostic proxies")
    fig.tight_layout()
    fig.savefig(out / "01_pipeline_overview.png", dpi=220)
    plt.close(fig)
    order = ("I0", "S0", "P1", "L1", "O1", "M1")
    fig, axes = plt.subplots(len(order), 10, figsize=(12, 7), squeeze=False)
    for row, variant in enumerate(order):
        data = rows[variant][str(PANELS[0])]
        view, labels = _load_tensor(root, data["artifacts"]["fixed_cmos"]), _load_tensor(root, data["artifacts"]["labels"])
        for col, label in enumerate(range(10)):
            index = int(torch.where(labels == label)[0][0])
            axes[row, col].imshow(view[index, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(variant, rotation=0, ha="right", va="center")
    fig.suptitle("Synthetic fiber attribution: fixed exposure CMOS")
    fig.tight_layout()
    fig.savefig(out / "03_fiber_function_comparison.png", dpi=220)
    plt.close(fig)
    trace_ref = rows["M1"][str(PANELS[0])]["artifacts"]["representative_trace"]
    trace_path = (root / trace_ref["relative_path"]).resolve()
    trace = safe_torch_load(trace_path, expected_sha256=trace_ref["sha256"], map_location="cpu")[0]
    fields = trace["fields"]
    fig, axes = plt.subplots(2, len(fields), figsize=(18, 5), squeeze=False)
    for index, field in enumerate(fields):
        tensor = field["tensor"][0, 0]
        axes[0, index].imshow(tensor.abs(), cmap="gray")
        axes[1, index].imshow(tensor.angle(), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[0, index].set_title(field["plane"]["plane_id"], fontsize=7)
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_ylabel("amplitude")
    axes[1, 0].set_ylabel("phase")
    fig.suptitle("Synthetic SMF optical trace")
    fig.tight_layout()
    fig.savefig(out / "02_stage_amplitude_phase.png", dpi=220)
    plt.close(fig)
    selection = json.loads(Path(args.v3_selection).read_text(encoding="utf-8"))
    funcs, gains = [], []
    for function in ("S0", "S1a", "S1b", "S1c"):
        funcs.append(function)
        gains.append(float(np.median([selection["bootstrap"][str(panel)][function]["median_gain"] for panel in PANELS])))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(funcs, gains, color="#4c78a8")
    ax.axhline(0.015, color="crimson", linestyle="--")
    ax.set_ylabel("secondary adaptation gain")
    fig.tight_layout()
    fig.savefig(out / "04_training_gain_ci.png", dpi=220)
    plt.close(fig)
    ranked = payload["bootstrap"]["classification"]["ranked_effects"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([x[0] for x in ranked], [x[1] for x in ranked], color="#f58518")
    ax.axhline(0.015, color="crimson", linestyle="--")
    ax.set_ylabel("observed secondary drop")
    fig.tight_layout()
    fig.savefig(out / "05_attribution_waterfall.png", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in (("S0", "#4c78a8"), ("S0-D0", "#e45756"), ("M1", "#72b7b2"), ("M1-D0", "#f58518")):
        ax.plot(PANELS, [rows[name][str(panel)]["secondary_accuracy"] for panel in PANELS], marker="o", label=name, color=color)
    ax.set_ylabel("secondary accuracy")
    ax.set_xlabel("panel seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "06_decoder_evaluator_comparison.png", dpi=220)
    plt.close(fig)
    with (out / "attribution_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "panel", "primary_accuracy", "secondary_accuracy", "secondary_min_recall", "camera_fraction", "minimum_diversity", "coverage_ratio"])
        for variant in VARIANTS:
            for panel in PANELS:
                row = rows[variant][str(panel)]
                writer.writerow([variant, panel, row["primary_accuracy"], row["secondary_accuracy"], row["secondary_min_recall"], row["camera_fraction"], row["minimum_diversity"], row["coverage_ratio"]])
    (out / "ppt_outline.md").write_text("# Synthetic fiber attribution PPT outline\n\n1. Question and contract — fixed G0, paired codes/labels, no-update attribution.\n2. Optical chain — SLM -> encoder -> LP01 coupling/transport -> decoder -> CMOS.\n3. Function comparison — I0/S0/P1/L1/O1/M1 under one exposure and one panel.\n4. Training result — v3 gains remain below 0.015; closeout is no reproducible gain.\n5. Attribution — waterfall and decoder-neutral comparison; synthetic observation only.\n6. Boundary — no calibrated NA/MFD/index/length or measured matrix; no hardware claim.\n", encoding="utf-8")
    _digit_pipeline_report(root, out / "digit_pipeline_s0", args.digit_pipeline_variant)
    _write(out / "presentation.json", {"schema": "fiber_attribution_presentation.v1", "gate": "valid", "measurement_sha256": file_sha256(args.measurement_evidence), "source_measurement_git_commit": payload.get("git_commit"), "report_git_commit": image._git(), "status": "synthetic diagnostic-only"})
    return 0


def closeout(args: Any) -> int:
    root, payload = _load_measurement(args)
    presentation = _require(args.presentation_evidence)
    _write(Path(args.output), {"schema": "fiber_attribution_closeout.v1", "gate": "valid", "conclusion": payload["bootstrap"]["classification"]["conclusion"], "measurement_sha256": file_sha256(args.measurement_evidence), "presentation_sha256": file_sha256(presentation), "formal_promotion": False, "synthetic_diagnostic_only": True})
    return 0


def dispatch(args: Any) -> int:
    return {"validate": validate, "run": run, "report": report, "closeout": closeout}[args.research_action](args)
