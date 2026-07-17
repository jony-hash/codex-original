"""Streaming, diagnostic-only preflight for the v68 transparent path.

The v67.4 audit retains every full optical field for several samples and is
therefore intentionally unsuitable for a 6 GB GPU.  This probe constructs one
grid at a time and one sample at a time, immediately moving only the central
artifact view and raw camera tensor to CPU.  It is a short admission check, not
the long gradient or A/B training job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

try:
    from tools.diagnose_v672_transparent_phase import compare_planes, correlation
    from tools.diagnose_v674_transparent_operator import PLANE_NAMES, _set_phase_baseline
except ModuleNotFoundError:  # direct ``python tools/<script>.py`` compatibility
    from diagnose_v672_transparent_phase import compare_planes, correlation
    from diagnose_v674_transparent_operator import PLANE_NAMES, _set_phase_baseline


REPORT_SCHEMA = "v68_transparent_admission_v2"


def _tensor_hash(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(_tensor_hash(value).encode())
    return digest.hexdigest()


def _build_prototype_bank(base: Any) -> torch.Tensor:
    """Use exactly the neutral MNIST prototype builder used by training."""
    from ldm_project.evaluation.prototypes import build_mnist_label_prototype_bank

    labels = tuple(base.distillation.class_labels) or tuple(
        range(int(base.one_shot.num_classes))
    )
    return build_mnist_label_prototype_bank(
        torch,
        class_labels=labels,
        image_size=int(base.first_stage.image_size),
        root=base.data.root,
        max_per_class=int(base.distillation.label_audit_prototypes_per_class),
    ).contiguous()


def _checkpoint_architecture(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    train_config = payload.get("train_config")
    one_shot = getattr(train_config, "one_shot", None)
    if one_shot is None and isinstance(train_config, dict):
        one_shot = train_config.get("one_shot", {})
    if one_shot is None:
        return None, None
    if isinstance(one_shot, dict):
        return one_shot.get("generator_e_arch"), one_shot.get(
            "generator_command_transform"
        )
    return getattr(one_shot, "generator_e_arch", None), getattr(
        one_shot, "generator_command_transform", None
    )


def _initialize_model(
    model: torch.nn.Module,
    *,
    base: Any,
    prototype_bank: torch.Tensor,
    device: torch.device,
    warm_start_mode: str,
    checkpoint_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if warm_start_mode not in {"prototype_init", "generator_subset"}:
        raise ValueError("warm_start_mode must be prototype_init or generator_subset")
    target_arch = base.one_shot.generator_e_arch
    target_transform = base.one_shot.generator_command_transform
    checkpoint_summary: dict[str, Any] | None = None
    if warm_start_mode == "generator_subset":
        if checkpoint_payload is None:
            raise ValueError("generator_subset mode requires --checkpoint")
        source_arch, source_transform = _checkpoint_architecture(checkpoint_payload)
        if (source_arch, source_transform) != (target_arch, target_transform):
            raise ValueError(
                "incompatible_generator_warmstart: checkpoint generator contract "
                f"({source_arch}, {source_transform}) does not match target "
                f"({target_arch}, {target_transform})"
            )
        from ldm_project import load_one_shot_checkpoint_subset

        checkpoint_summary = load_one_shot_checkpoint_subset(
            model,
            checkpoint_payload,
            include_prefixes=("generator_e.",),
            description="v68 generator-only warm-start",
        )
        init_source = "generator_subset_checkpoint"
    else:
        if not hasattr(model.generator_e, "initialize_label_templates"):
            raise ValueError(
                "prototype_init requires a label-template generator architecture"
            )
        model.generator_e.initialize_label_templates(prototype_bank.to(device))
        init_source = "mnist_prototype"
    return {
        "generator_init_source": init_source,
        "generator_state_hash": _module_hash(model.generator_e),
        "phase_initialization_hash": _module_hash(
            torch.nn.ModuleList([model.encoder, model.decoder])
        ),
        "generator_template_sha256": _tensor_hash(prototype_bank),
        "checkpoint_load_mode": warm_start_mode,
        "checkpoint_load_summary": checkpoint_summary,
        "warmstart_compatibility_status": "compatible"
        if checkpoint_summary is not None
        else "not-applicable",
    }


def _config_for_grid(base: Any, state_grid: int, fft_grid: int) -> Any:
    d2nn = replace(
        base.one_shot.d2nn,
        propagation_grid_size=state_grid,
        exterior_boundary="transparent_identity",
        transparent_propagation_method="padded_band_limited_asm",
        transparent_fft_grid_size=fft_grid,
        transparent_memory_mode="checkpointed_active_capture",
    )
    return replace(base.one_shot, physical_forward_contract=True, d2nn=d2nn)


def _stream_grid(
    *, base: Any, codes: torch.Tensor,
    labels: torch.Tensor, device: torch.device, state_grid: int, fft_grid: int,
    baseline: str, max_samples: int, prototype_bank: torch.Tensor,
    warm_start_mode: str, checkpoint_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    from ldm_project import OneShotOpticalGenerator

    physics = replace(base.physics, physical_forward_contract=True)
    config = _config_for_grid(base, state_grid, fft_grid)
    model = OneShotOpticalGenerator(physics, config, image_size=base.first_stage.image_size)
    model.to(device).eval()
    _set_phase_baseline(model, baseline)
    initialization = _initialize_model(
        model,
        base=base,
        prototype_bank=prototype_bank,
        device=device,
        warm_start_mode=warm_start_mode,
        checkpoint_payload=checkpoint_payload,
    )
    records: dict[str, list[torch.Tensor]] = {name: [] for name in PLANE_NAMES}
    cameras: list[torch.Tensor] = []
    energy_records: list[dict[str, float]] = []
    with torch.no_grad():
        for code, label in zip(codes[:max_samples], labels[:max_samples]):
            output = model(
                code[None].to(device), label[None].to(device),
                # Do not request the full optical-field bundle.  The regular
                # output contract already exposes active amplitude/phase views;
                # this is what makes the probe safe on a small GPU.
                return_optical_fields=False,
            )
            encoder_layers = torch.polar(
                output["diffractive_encoder_layer_amplitude"],
                output["diffractive_encoder_layer_phase"],
            )
            decoder_layers = torch.polar(
                output["diffractive_decoder_layer_amplitude"],
                output["diffractive_decoder_layer_phase"],
            )
            active_fields = {
                "encoder_layer_1": encoder_layers[:, 0],
                "encoder_layer_2": encoder_layers[:, 1],
                "encoder_output": torch.polar(
                    output["encoded_amplitude"], output["encoded_phase"]
                ),
                "decoder_layer_1": decoder_layers[:, 0],
                "decoder_layer_2": decoder_layers[:, 1],
                "chip": torch.complex(output["chip_real"], output["chip_imag"]),
            }
            for name in PLANE_NAMES:
                if name not in active_fields:
                    raise RuntimeError(f"missing active field record: {name}")
                records[name].append(active_fields[name].detach().cpu())
            cameras.append(output["physical_camera_intensity_raw"].detach().cpu())
            energy_records.append(
                {
                    key: float(output[key].detach().float().mean().cpu())
                    for key in output
                    if key.endswith("discarded_energy_ratio")
                    or key.endswith("guard_band_energy_ratio")
                    or key.endswith("propagation_energy_ratio")
                    or key.endswith("retained_energy")
                }
            )
            del output
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "state_grid": state_grid,
        "fft_grid": fft_grid,
        "baseline": baseline,
        "sample_count": len(cameras),
        "fields": {name: torch.cat(values) for name, values in records.items()},
        "camera": torch.cat(cameras),
        "energy_records": energy_records,
        "initialization": initialization,
    }


def _compare(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    planes = {
        name: compare_planes(candidate["fields"][name], reference["fields"][name])
        for name in PLANE_NAMES
    }
    candidate_energy = candidate["energy_records"]
    reference_energy = reference["energy_records"]
    energy_differences: dict[str, float] = {}
    for key in sorted(set().union(*(record.keys() for record in candidate_energy))):
        c = sum(record.get(key, 0.0) for record in candidate_energy) / len(candidate_energy)
        r = sum(record.get(key, 0.0) for record in reference_energy) / len(reference_energy)
        energy_differences[key] = abs(c - r) / max(abs(r), 1.0e-12)
    max_boundary_ratio = max(
        (
            value
            for record in candidate_energy
            for key, value in record.items()
            if key.endswith("guard_band_energy_ratio")
            or key.endswith("discarded_energy_ratio")
        ),
        default=0.0,
    )
    total_energy_pass = all(
        value <= 1.0e-3
        for key, value in energy_differences.items()
        if key.endswith("propagation_energy_ratio")
        or key.endswith("retained_energy")
    )
    boundary_pass = max_boundary_ratio <= 1.0e-4
    return {
        "planes": planes,
        "camera_intensity_corr": correlation(candidate["camera"], reference["camera"]),
        "candidate_initialization": candidate["initialization"],
        "reference_initialization": reference["initialization"],
        "first_failure_plane": next(
            (name for name in PLANE_NAMES
             if float(planes[name]["gauge_aligned_complex_field_relative_l2"]["mean"]) > 1.0e-3),
            None,
        ),
        "all_plane_gates_pass": all(
            float(metrics["gauge_aligned_complex_field_relative_l2"]["mean"]) <= 1.0e-3
            and float(metrics["amplitude_relative_l2"]["mean"]) <= 1.0e-3
            and float(metrics["weighted_phase_mae_rad"]["mean"]) <= 1.0e-3
            and float(metrics["intensity_corr"]["mean"]) >= 0.999
            for metrics in planes.values()
        ),
        "energy_relative_differences": energy_differences,
        "max_candidate_guard_or_discarded_ratio": max_boundary_ratio,
        "total_energy_gate_pass": total_energy_pass,
        "boundary_energy_gate_pass": boundary_pass,
    }


def run(
    *, output: str | Path, train_config: str | Path,
    checkpoint: str | Path | None, sample_tensors: str | Path,
    device: str = "cuda", max_samples: int = 4,
    warm_start_mode: str = "prototype_init",
) -> dict[str, Any]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v68 memory admission")
    from ldm_project import load_training_config

    base = load_training_config(train_config)
    payload = torch.load(sample_tensors, map_location="cpu", weights_only=False)
    checkpoint_payload = (
        torch.load(checkpoint, map_location=resolved, weights_only=False)
        if checkpoint is not None
        else None
    )
    codes, labels = payload["codes"], payload["class_labels"]
    prototype_bank = _build_prototype_bank(base)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "running",
        "warm_start_mode": warm_start_mode,
        "candidate": {"state_grid": 2048, "fft_grid": 3072},
        "reference": {"state_grid": 3072, "fft_grid": 4096},
        "streaming_contract": {
            "one_sample_at_a_time": True,
            "full_field_saved": False,
            "artifact_view": "central_active_view_only",
            "backend_receives_full_chip_field": True,
        },
    }
    try:
        results: dict[str, Any] = {}
        for baseline in ("zero", "small"):
            candidate = _stream_grid(
                base=base, codes=codes,
                labels=labels, device=resolved, state_grid=2048, fft_grid=3072,
                baseline=baseline, max_samples=max_samples,
                prototype_bank=prototype_bank, warm_start_mode=warm_start_mode,
                checkpoint_payload=checkpoint_payload,
            )
            reference = _stream_grid(
                base=base, codes=codes,
                labels=labels, device=resolved, state_grid=3072, fft_grid=4096,
                baseline=baseline, max_samples=max_samples,
                prototype_bank=prototype_bank, warm_start_mode=warm_start_mode,
                checkpoint_payload=checkpoint_payload,
            )
            results[baseline] = _compare(candidate, reference)
        report["initialization"] = {
            "prototype_sha256": _tensor_hash(prototype_bank),
            "candidate": results["small"].get("candidate_initialization"),
            "reference": results["small"].get("reference_initialization"),
        }
        report["baselines"] = results
        report["status"] = (
            "numerical-converged"
            if all(result["all_plane_gates_pass"] and result["camera_intensity_corr"] >= 0.999
                   for result in results.values())
            else "grid-not-converged"
        )
        report["training_admission_status"] = (
            "pending-gradient-and-memory-smoke"
            if report["status"] == "numerical-converged"
            else "stop-before-gradient-and-training"
        )
    except (RuntimeError, ValueError) as exc:
        if "incompatible_generator_warmstart" in str(exc):
            report["status"] = "incompatible_generator_warmstart"
            report["training_admission_status"] = "stop-before-gradient-and-training"
            report["error"] = str(exc)
        elif resolved.type == "cuda" and "out of memory" in str(exc).lower():
            report["status"] = "cuda-memory-blocked"
            report["training_admission_status"] = "reference-domain-memory-blocked"
            report["error"] = str(exc)
        else:
            raise
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-tensors", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", default=4, type=int)
    parser.add_argument(
        "--warm-start-mode",
        choices=("prototype_init", "generator_subset"),
        default="prototype_init",
    )
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
