"""v67.4 diagnostic audit for transparent full-domain propagation.

This tool isolates the repeated state-grid crop from the transparent operator.
It never changes the formal ``finite_crop`` path and never starts training.
The 1536/3072 path reproduces the v67.3 baseline; 3072/3072 is the proposed
full-domain candidate and 4096/4096 is a read-only reference.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

try:
    from tools.diagnose_v672_transparent_phase import compare_planes, correlation
except ModuleNotFoundError:  # direct ``python tools/<script>.py`` compatibility
    from diagnose_v672_transparent_phase import compare_planes, correlation


REPORT_SCHEMA = "v67_transparent_operator_audit_v1"
ACTIVE_SIZE = 100
PIXEL_SIZE = 2.0e-6
WAVELENGTH = 532.0e-9
LAYER_SPACING = 1.5e-3
BASELINE_NAMES = ("zero", "small")
PLANE_NAMES = (
    "encoder_layer_1",
    "encoder_layer_2",
    "encoder_output",
    "decoder_layer_1",
    "decoder_layer_2",
    "chip",
)


def _field(record: dict[str, Any]) -> torch.Tensor:
    return torch.complex(record["real"], record["imag"])


def _set_phase_baseline(model: torch.nn.Module, baseline: str) -> None:
    if baseline not in {"zero", "small"}:
        raise ValueError(f"unsupported baseline: {baseline}")
    for offset, stack in enumerate((model.encoder, model.decoder)):
        with torch.no_grad():
            if baseline == "zero":
                stack.phase.zero_()
                continue
            generator = torch.Generator(device=stack.phase.device).manual_seed(1234 + offset)
            values = torch.rand(
                stack.phase.shape,
                device=stack.phase.device,
                dtype=stack.phase.dtype,
                generator=generator,
            )
            stack.phase.copy_(values.mul_(0.10).add_(-0.05))


def _config_for_grid(base: Any, state_grid: int, fft_grid: int | None, *, method: str = "padded_band_limited_asm") -> Any:
    d2nn = replace(
        base.one_shot.d2nn,
        propagation_grid_size=state_grid,
        exterior_boundary="transparent_identity",
        transparent_propagation_method=method,
        transparent_fft_grid_size=fft_grid,
    )
    return replace(base.one_shot, physical_forward_contract=True, d2nn=d2nn)


def _capture_one_grid(
    *,
    base: Any,
    checkpoint_payload: dict[str, Any],
    codes: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    state_grid: int,
    fft_grid: int | None,
    baseline: str,
    method: str = "padded_band_limited_asm",
) -> dict[str, Any]:
    from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint

    physics = replace(base.physics, physical_forward_contract=True)
    config = _config_for_grid(base, state_grid, fft_grid, method=method)
    model = OneShotOpticalGenerator(physics, config, image_size=base.first_stage.image_size)
    model = model.to(device).eval()
    load_one_shot_checkpoint(model, checkpoint_payload)
    _set_phase_baseline(model, baseline)
    records: dict[str, list[torch.Tensor]] = {name: [] for name in PLANE_NAMES}
    cameras: list[torch.Tensor] = []
    with torch.no_grad():
        for code, label in zip(codes[:4], labels[:4]):
            output = model(
                code[None].to(device),
                label[None].to(device),
                return_optical_fields=True,
            )
            optical_fields = output.get("optical_fields")
            if not isinstance(optical_fields, dict):
                raise RuntimeError("transparent audit requires optical field records")
            for name in PLANE_NAMES:
                if name not in optical_fields:
                    raise RuntimeError(f"missing optical field record: {name}")
                records[name].append(_field(optical_fields[name]).detach().cpu())
            cameras.append(output["physical_camera_intensity_raw"].detach().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "state_grid": state_grid,
        "fft_grid": fft_grid,
        "method": method,
        "fields": {name: torch.cat(values) for name, values in records.items()},
        "camera": torch.cat(cameras),
    }


def _summarize_candidate(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    plane_metrics: dict[str, Any] = {}
    for name in PLANE_NAMES:
        plane_metrics[name] = compare_planes(
            candidate["fields"][name],
            reference["fields"][name],
        )
    result["planes"] = plane_metrics
    chip = plane_metrics["chip"]
    result.update(
        {
            "chip_gauge_aligned_complex_l2": chip["gauge_aligned_complex_field_relative_l2"],
            "camera_intensity_corr": correlation(candidate["camera"], reference["camera"]),
            "first_failure_plane": next(
                (
                    name
                    for name in PLANE_NAMES
                    if float(plane_metrics[name]["gauge_aligned_complex_field_relative_l2"]["mean"])
                    > 1.0e-3
                ),
                None,
            ),
        }
    )
    result["all_plane_gates_pass"] = all(
        float(metrics["gauge_aligned_complex_field_relative_l2"]["mean"]) <= 1.0e-3
        and float(metrics["amplitude_relative_l2"]["mean"]) <= 1.0e-3
        and float(metrics["weighted_phase_mae_rad"]["mean"]) <= 1.0e-3
        and float(metrics["intensity_corr"]["mean"]) >= 0.999
        and float(metrics["total_energy_relative_difference"]["mean"]) <= 1.0e-3
        and float(metrics["guard_band_ratio"]["mean"]) <= 1.0e-4
        for metrics in plane_metrics.values()
    )
    result["passed"] = result["all_plane_gates_pass"] and result["camera_intensity_corr"] >= 0.999
    return result


def _transfer_audit(
    *,
    candidate_grid: int = 3072,
    reference_grid: int = 4096,
    pixel_size: float = PIXEL_SIZE,
    wavelength: float = WAVELENGTH,
    distance: float = LAYER_SPACING,
) -> dict[str, Any]:
    """Compare analytic transfer values on exactly shared physical frequency bins."""
    def transfer(grid: int) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.fft.fftfreq(grid, d=pixel_size, dtype=torch.float64)
        fy, fx = torch.meshgrid(frequencies, frequencies, indexing="ij")
        radial = (wavelength * fx).square() + (wavelength * fy).square()
        propagating = radial <= 1.0
        root = (1.0 - radial).clamp_min(0.0).sqrt()
        phase = 2.0 * torch.pi * distance * root / wavelength
        # This is an audit-only analytic construction, not a second production
        # propagator; it verifies the physical transfer rule and grid mapping.
        return torch.exp(1j * phase), propagating

    candidate, candidate_mask = transfer(candidate_grid)
    reference, reference_mask = transfer(reference_grid)
    candidate_indices = torch.arange(candidate_grid)
    reference_indices = torch.arange(reference_grid)
    # n_candidate / candidate_grid == n_reference / reference_grid.
    common_candidate = candidate_indices[candidate_indices.remainder(3) == 0]
    common_reference = reference_indices[reference_indices.remainder(4) == 0]
    candidate_common = candidate[common_candidate[:, None], common_candidate[None, :]]
    reference_common = reference[common_reference[:, None], common_reference[None, :]]
    mask_common = candidate_mask[common_candidate[:, None], common_candidate[None, :]] & reference_mask[
        common_reference[:, None], common_reference[None, :]
    ]
    phase_error = torch.angle(candidate_common * reference_common.conj()).abs()
    return {
        "candidate_grid": candidate_grid,
        "reference_grid": reference_grid,
        "shared_frequency_bin_contract": "candidate index multiple of 3; reference index multiple of 4",
        "shared_bin_count": int(common_candidate.numel()),
        "propagating_mask_mismatch_fraction": float(
            (candidate_mask[common_candidate[:, None], common_candidate[None, :]]
             != reference_mask[common_reference[:, None], common_reference[None, :]]).float().mean()
        ),
        "transfer_phase_max_error_rad_on_shared_propagating_bins": float(phase_error[mask_common].max()),
        "transfer_phase_mean_error_rad_on_shared_propagating_bins": float(phase_error[mask_common].mean()),
    }


def run(
    *,
    output: str | Path,
    device: str = "cuda",
    train_config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    sample_tensors: str | Path | None = None,
) -> dict[str, Any]:
    resolved = torch.device(device)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "role": "periodic-full-domain-stress-test",
        "superseded_for_v68_admission": True,
        "status": "preflight-input-required",
        "device": str(resolved),
        "candidate_paths": {
            "legacy_per_layer_crop": {"state_grid": 1536, "fft_grid": 3072},
            "full_domain_candidate": {"state_grid": 3072, "fft_grid": 3072},
            "read_only_reference": {"state_grid": 4096, "fft_grid": 4096},
        },
        "physical_contract": {
            "pixel_pitch_m": PIXEL_SIZE,
            "wavelength_m": WAVELENGTH,
            "phase_plane_spacing_m": LAYER_SPACING,
            "active_window": [ACTIVE_SIZE, ACTIVE_SIZE],
            "exterior_transmission": "identity",
            "inter_layer_crop": False,
        },
        "gates": {
            "gauge_aligned_complex_field_relative_l2": 1.0e-3,
            "amplitude_relative_l2": 1.0e-3,
            "weighted_phase_mae_rad": 1.0e-3,
            "intensity_corr": 0.999,
            "total_energy_relative_difference": 1.0e-3,
            "guard_band_ratio": 1.0e-4,
        },
        "transfer_audit": {"status": "not-run"},
        "training_admission_status": "not-run",
        "decision": "stop-before-gradient-and-training",
    }
    if not (train_config and checkpoint and sample_tensors):
        report["reason"] = "train-config, checkpoint and sample-tensors are required"
    else:
        from ldm_project import load_training_config

        report["transfer_audit"] = _transfer_audit()
        base = load_training_config(train_config)
        payload = torch.load(sample_tensors, map_location="cpu", weights_only=False)
        checkpoint_payload = torch.load(checkpoint, map_location=resolved, weights_only=False)
        codes, labels = payload["codes"], payload["class_labels"]
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for v67.4 operator audit")
        captures: dict[str, dict[str, dict[str, Any]]] = {}
        try:
            for baseline in BASELINE_NAMES:
                captures[baseline] = {}
                captures[baseline]["legacy_per_layer_crop"] = _capture_one_grid(
                    base=base, checkpoint_payload=checkpoint_payload, codes=codes, labels=labels,
                    device=resolved, state_grid=1536, fft_grid=3072, baseline=baseline,
                )
                captures[baseline]["full_domain_candidate"] = _capture_one_grid(
                    base=base, checkpoint_payload=checkpoint_payload, codes=codes, labels=labels,
                    device=resolved, state_grid=3072, fft_grid=3072, baseline=baseline,
                )
                captures[baseline]["read_only_reference"] = _capture_one_grid(
                    base=base, checkpoint_payload=checkpoint_payload, codes=codes, labels=labels,
                    device=resolved, state_grid=4096, fft_grid=4096, baseline=baseline,
                )
        except RuntimeError as exc:
            if resolved.type != "cuda" or "out of memory" not in str(exc).lower():
                raise
            report["status"] = "cuda-memory-blocked"
            report["reference_domain_status"] = "reference-domain-memory-blocked"
            report["error"] = str(exc)
            report["decision"] = "stop-before-gradient-and-training"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report
        report["baselines"] = {}
        for baseline, paths in captures.items():
            report["baselines"][baseline] = {
                "legacy_vs_reference": _summarize_candidate(
                    paths["legacy_per_layer_crop"], paths["read_only_reference"]
                ),
                "full_domain_vs_reference": _summarize_candidate(
                    paths["full_domain_candidate"], paths["read_only_reference"]
                ),
            }
        small = report["baselines"]["small"]["full_domain_vs_reference"]
        report["first_failure_plane"] = small["first_failure_plane"]
        report["full_domain_candidate_converged"] = bool(
            all(report["baselines"][name]["full_domain_vs_reference"]["passed"] for name in BASELINE_NAMES)
        )
        report["per_layer_crop_is_causal"] = all(
            report["baselines"][name]["legacy_vs_reference"]["passed"] is False
            and report["baselines"][name]["full_domain_vs_reference"]["passed"] is True
            for name in BASELINE_NAMES
        )
        report["reference_domain_status"] = "relative_reference_only"
        report["status"] = (
            "numerical-converged" if report["full_domain_candidate_converged"] else "grid-not-converged"
        )
        if report["status"] == "numerical-converged":
            report["training_admission_status"] = "pending-user-gradient-and-memory-run"
            report["decision"] = "ready-for-gradient-and-memory-admission"
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-tensors", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
