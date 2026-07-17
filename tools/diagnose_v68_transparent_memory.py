"""Streaming, diagnostic-only preflight for the v68 transparent path.

The v67.4 audit retains every full optical field for several samples and is
therefore intentionally unsuitable for a 6 GB GPU.  This probe constructs one
grid at a time and one sample at a time, immediately moving only the central
artifact view and raw camera tensor to CPU.  It is a short admission check, not
the long gradient or A/B training job.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from tools.diagnose_v672_transparent_phase import compare_planes, correlation
from tools.diagnose_v674_transparent_operator import PLANE_NAMES, _set_phase_baseline


REPORT_SCHEMA = "v68_transparent_memory_gate_v1"


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
    *, base: Any, checkpoint_payload: dict[str, Any], codes: torch.Tensor,
    labels: torch.Tensor, device: torch.device, state_grid: int, fft_grid: int,
    baseline: str, max_samples: int,
) -> dict[str, Any]:
    from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint

    physics = replace(base.physics, physical_forward_contract=True)
    config = _config_for_grid(base, state_grid, fft_grid)
    model = OneShotOpticalGenerator(physics, config, image_size=base.first_stage.image_size)
    model.to(device).eval()
    load_one_shot_checkpoint(model, checkpoint_payload)
    _set_phase_baseline(model, baseline)
    records: dict[str, list[torch.Tensor]] = {name: [] for name in PLANE_NAMES}
    cameras: list[torch.Tensor] = []
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
    }


def _compare(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    planes = {
        name: compare_planes(candidate["fields"][name], reference["fields"][name])
        for name in PLANE_NAMES
    }
    return {
        "planes": planes,
        "camera_intensity_corr": correlation(candidate["camera"], reference["camera"]),
        "first_failure_plane": next(
            (name for name in PLANE_NAMES
             if float(planes[name]["gauge_aligned_complex_field_relative_l2"]["mean"]) > 1.0e-3),
            None,
        ),
    }


def run(
    *, output: str | Path, train_config: str | Path,
    checkpoint: str | Path, sample_tensors: str | Path,
    device: str = "cuda", max_samples: int = 4,
) -> dict[str, Any]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v68 memory admission")
    from ldm_project import load_training_config

    base = load_training_config(train_config)
    payload = torch.load(sample_tensors, map_location="cpu", weights_only=False)
    checkpoint_payload = torch.load(checkpoint, map_location=resolved, weights_only=False)
    codes, labels = payload["codes"], payload["class_labels"]
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "running",
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
                base=base, checkpoint_payload=checkpoint_payload, codes=codes,
                labels=labels, device=resolved, state_grid=2048, fft_grid=3072,
                baseline=baseline, max_samples=max_samples,
            )
            reference = _stream_grid(
                base=base, checkpoint_payload=checkpoint_payload, codes=codes,
                labels=labels, device=resolved, state_grid=3072, fft_grid=4096,
                baseline=baseline, max_samples=max_samples,
            )
            results[baseline] = _compare(candidate, reference)
        report["baselines"] = results
        report["status"] = "streaming-probe-complete"
        report["training_admission_status"] = "pending-gradient-and-memory-smoke"
    except RuntimeError as exc:
        if resolved.type == "cuda" and "out of memory" in str(exc).lower():
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
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sample-tensors", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", default=4, type=int)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
