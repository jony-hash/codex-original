# ruff: noqa: F401,F403,F405,F821,E402
"""Historical CLI implementation segment, loaded only for legacy commands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...config import V2Config
from .serialization import _trusted_legacy_torch_load
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    tensor_artifact_ref,
    tensor_sha256,
)
from ...student.checkpoints import CheckpointManifestV2, state_content_hash, state_signature
from ...student.clean_targets import CleanTargetManifestV2, build_clean_targets
from ...teacher import (
    TeacherPairManifestV2,
    TeacherTargetManifestV2,
    build_teacher_pair_manifest,
    build_teacher_target_manifest,
)
from ..runs import V2Run

class StudentBundle(nn.Module):
    def __init__(self, config: V2Config, basis) -> None:
        super().__init__()
        from ...optics import FormalAmplitudeFiniteCropIdentitySystem
        from ...student.generator import LabelPrototypeBasisGenerator

        self.system = FormalAmplitudeFiniteCropIdentitySystem(config.system)
        self.generator = LabelPrototypeBasisGenerator(
            code_dim=config.student_training.code_dim if config.student_training else 8,
            basis_artifact=basis,
        )


def _tensor(path: str, *, device: torch.device) -> torch.Tensor:
    value = _trusted_legacy_torch_load(path, map_location=device)
    if not torch.is_tensor(value):
        raise ValueError(f"{path} must contain one tensor")
    return value.to(device)


def _record(path: str) -> EvidenceRecordV2:
    return EvidenceRecordV2.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _pair_manifest(path: str) -> TeacherPairManifestV2:
    manifest = TeacherPairManifestV2.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    manifest.validate()
    return manifest


def _target_manifest(path: str) -> TeacherTargetManifestV2:
    manifest = TeacherTargetManifestV2.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    manifest.validate()
    return manifest


def _clean_target_manifest(path: str, config: V2Config) -> CleanTargetManifestV2:
    manifest = CleanTargetManifestV2.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    manifest.validate()
    if manifest.system_contract_hash is not None:
        if manifest.system_contract_hash != _system_signature(config):
            raise ValueError("clean-target manifest belongs to another physical system")
    elif manifest.config_hash != canonical_sha256(config.model_dump(mode="json")):
        raise ValueError("legacy clean-target manifest requires explicit migration")
    if manifest.shape != (10, 1, *config.system.slm_shape):
        raise ValueError(
            "clean-target manifest geometry does not match the formal SLM plane"
        )
    return manifest


def _validate_target_for_config(
    target: TeacherTargetManifestV2, config: V2Config
) -> None:
    if config.teacher is None:
        raise ValueError("v2 student stages require a teacher configuration")
    pair = build_teacher_pair_manifest(
        config.teacher.first_stage_checkpoint,
        config.teacher.diffusion_checkpoint,
        target_plane=config.teacher.target_plane,
        seed=config.teacher.seed,
        guidance_scale=config.teacher.guidance_scale,
        split=target.split,
        latent_geometry=config.teacher.latent_geometry,
    )
    if pair.manifest_hash != target.teacher_pair_manifest_hash:
        raise ValueError(
            "teacher target manifest is not paired with current teacher checkpoints/config"
        )
    if (
        target.system_contract_hash is not None
        and target.system_contract_hash != _system_signature(config)
    ):
        raise ValueError("teacher target manifest belongs to another physical system")
    if target.teacher_export_contract_hash is None:
        if not config.system.is_formal:
            target.validate(formal=False)
            return
        raise ValueError("teacher target manifest requires explicit scoped migration")
    expected_export_hash = _teacher_export_contract_hash(
        config, pair, target.legacy_export_config_hash or ""
    )
    if target.teacher_export_contract_hash != expected_export_hash:
        raise ValueError(
            "teacher target manifest belongs to another teacher export contract"
        )
    target.validate(formal=config.system.is_formal)


def _write_record(run: V2Run, name: str, record: EvidenceRecordV2) -> Path:
    record.validate()
    output = run.root / name
    atomic_write_json(output, record.to_dict())
    return output


def _system_signature(config: V2Config) -> str:
    return canonical_sha256(config.system.model_dump(mode="json"))


def _teacher_export_contract_hash(
    config: V2Config, pair: TeacherPairManifestV2, legacy_hash: str
) -> str:
    if config.teacher is None:
        raise ValueError("teacher export contract requires teacher configuration")
    return canonical_sha256(
        {
            "teacher_pair_manifest_hash": pair.manifest_hash,
            "legacy_export_config_hash": legacy_hash,
            "target_plane": config.teacher.target_plane,
            "latent_geometry": config.teacher.latent_geometry,
            "seed": config.teacher.seed,
            "guidance_scale": config.teacher.guidance_scale,
        }
    )


def _context(
    config: V2Config,
    *,
    teacher_manifest_hash: str | None = None,
    dataset_hash: str | None = None,
    calibration_hash: str | None = None,
    checkpoint_hash: str | None = None,
    basis_hash: str | None = None,
) -> EvidenceContextV2:
    return EvidenceContextV2(
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        teacher_manifest_hash,
        dataset_hash,
        calibration_hash,
        checkpoint_hash,
        basis_hash,
    )


def _require_clean_acceptance(config: V2Config) -> None:
    if config.run.evidence_tier != "acceptance":
        return
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("acceptance operations require a clean git commit")


def _legacy_export_contract(path: str, config: V2Config) -> str:
    from ..legacy_adapter import legacy_export_contract

    return legacy_export_contract(path, config)


def _new_run(config: V2Config) -> V2Run:
    _require_clean_acceptance(config)
    return V2Run.create(
        output_root=config.run.output_root,
        run_name=config.run.run_name,
        config_payload=config.model_dump(mode="json"),
    )


def _lightgen_contract(path: str) -> LightGenReferenceContractV2:
    from ...lightgen_contract import LightGenReferenceContractV2

    return LightGenReferenceContractV2.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _require_lightgen(config: V2Config) -> LightGenSystemConfig:
    from ...config import LightGenSystemConfig

    if not isinstance(config.system, LightGenSystemConfig):
        raise ValueError(
            "this audit requires diagnostic_lightgen_amplitude_fiber_ols_400"
        )
    if config.run.evidence_tier != "diagnostic":
        raise ValueError("LightGen architecture proxy is diagnostic-only")
    return config.system


def _lightgen_seeds(*, device: torch.device, shape: tuple[int, int]) -> torch.Tensor:
    """Fixed global nonlinear amplitude seeds; this is neither DCT nor a local-gain basis."""
    height, width = shape
    yy = torch.linspace(-1.0, 1.0, height, device=device).view(1, height, 1)
    xx = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, width)
    code = torch.eye(8, device=device).view(8, 8, 1, 1)
    frequencies = torch.arange(1, 9, device=device, dtype=yy.dtype).view(8, 1, 1)
    patterns = torch.sin(frequencies * 2.0 * torch.pi * xx) + torch.cos(
        (frequencies + 0.5) * 2.0 * torch.pi * yy
    )
    nonlinear = torch.tanh(code * patterns.unsqueeze(0)).sum(dim=1)
    # all directions modulate the entire active aperture before the amplitude mapping
    return (0.5 + 0.20 * nonlinear).sigmoid().unsqueeze(1).clamp(1.0e-4, 1.0 - 1.0e-4)


def _lightgen_contract_gap(args: argparse.Namespace, config: V2Config) -> int:
    system = _require_lightgen(config)
    reference = _lightgen_contract(args.reference_contract)
    run = _new_run(config)
    formal = {
        "resolution": [200, 200],
        "pixel_pitch_m": 2.5e-6,
        "encoder_layers": 2,
        "generator_layers": 2,
        "ols": "identity",
        "ols_dimension": None,
        "complex_ols": False,
        "spatial_bandwidth_product": 10000,
        "training": "local_8d_DCT_supported_basis",
    }
    lightgen = {
        "resolution": list(system.slm_shape),
        "pixel_pitch_m": system.slm_pixel_pitch_m,
        "encoder_layers": system.encoder_layers,
        "generator_layers": system.decoder_layers,
        "ols": system.ols_kind,
        "ols_dimension": 100,
        "complex_ols": True,
        "spatial_bandwidth_product": system.slm_shape[0] * system.slm_shape[1],
        "training": "global_nonlinear_amplitude_seed_architecture_proxy",
    }
    differences = {
        name: {
            "formal": formal[name],
            "lightgen_proxy": lightgen[name],
            "different": formal[name] != lightgen[name],
        }
        for name in formal
    }
    atomic_write_json(
        run.root / "lightgen_contract_gap.json",
        {
            "reference_contract_hash": reference.contract_hash,
            "reference_proxy_only": reference.architecture_proxy_only,
            "differences": differences,
        },
    )
    evidence = EvidenceRecordV2(
        2,
        "lightgen_contract_gap",
        "diagnostic",
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(config),
        {"reference_contract": reference.contract_hash},
        (
            GateResult(
                "isolated_from_formal_system", True, {"formal_system_changed": False}
            ),
        ),
        (),
        "architecture-proxy: reference contract retains unknown physical parameters",
    )
    _write_record(run, "lightgen_contract_gap_evidence.json", evidence)
    print(run.root)
    return 0


def _lightgen_ols_observability(args: argparse.Namespace, config: V2Config) -> int:
    from ...optics import (
        LightGenAmplitudeFiberOLSSystemV2,
        reference_project_single_mode_fibers,
    )

    _require_lightgen(config)
    reference = _lightgen_contract(args.reference_contract)
    run = _new_run(config)
    device = torch.device(config.run.device)
    system = LightGenAmplitudeFiberOLSSystemV2(config.system).to(device)
    before = state_content_hash(system.state_dict())
    seeds = _lightgen_seeds(device=device, shape=config.system.slm_shape)
    with torch.no_grad():
        result = system(system.command(seeds))
        reference_coefficients = reference_project_single_mode_fibers(
            result.optical_trace[3], result.ols.modes
        )
        if not torch.allclose(
            result.ols.coefficients, reference_coefficients, atol=2.0e-5, rtol=2.0e-5
        ):
            raise ValueError("fiber OLS production/reference disagreement")
        # The real-imaginary coupling response preserves the physical complex OLS interface.
        response = torch.cat(
            (result.ols.coefficients.real, result.ols.coefficients.imag), dim=1
        )
        singular = torch.linalg.svdvals(response.cpu())
        effective_rank = int(
            (singular / singular[0].clamp_min(1.0e-12) >= 1.0e-6).sum().item()
        )
        ratio = float((singular[7] / singular[0].clamp_min(1.0e-12)).item())
        gram = result.ols.modes.gram
        crosstalk = float(
            (gram - torch.eye(100, device=device, dtype=gram.dtype)).abs().max().item()
        )
        raw = result.raw_camera.tensor.detach().cpu()
    after = state_content_hash(system.state_dict())
    atomic_torch_save(run.root / "lightgen_observability_raw_cmos.pt", raw)
    atomic_torch_save(run.root / "lightgen_ols_singular_values.pt", singular)
    atomic_write_json(
        run.root / "lightgen_ols_observability_report.json",
        {
            "mode_count": 100,
            "effective_rank": effective_rank,
            "sigma8_over_sigma1": ratio,
            "max_mode_crosstalk": crosstalk,
            "parameters_unchanged": before == after,
            "optimizer_created": False,
            "physical_parameter_status": "unknown",
            "evidence_label": "architecture-proxy",
            "input_seed_hash": tensor_sha256(seeds),
            "raw_cmos_hash": tensor_sha256(raw),
            "optical_trace": [
                {
                    "plane": field.plane.to_dict(),
                    "provenance": field.provenance.to_dict(),
                }
                for field in result.optical_trace
            ],
        },
    )
    gates = (
        GateResult(
            "one_hundred_finite_complex_modes",
            bool(torch.isfinite(result.ols.coefficients).all()),
            {"mode_count": 100},
        ),
        GateResult(
            "effective_rank_at_least_eight",
            effective_rank >= 8,
            {"effective_rank": effective_rank},
        ),
        GateResult(
            "sigma8_over_sigma1", ratio >= 1.0e-3, {"sigma8_over_sigma1": ratio}
        ),
        GateResult(
            "parameters_unchanged",
            before == after,
            {"parameters_unchanged": before == after},
        ),
        GateResult(
            "complete_provenance",
            all(field.provenance.operation for field in result.optical_trace),
            {"planes": len(result.optical_trace)},
        ),
    )
    passed = all(gate.passed for gate in gates)
    evidence = EvidenceRecordV2(
        2,
        "lightgen_ols_observability",
        "diagnostic",
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        _context(config),
        {
            "reference_contract": reference.contract_hash,
            "raw_cmos": file_sha256(run.root / "lightgen_observability_raw_cmos.pt"),
            "singular_values": file_sha256(
                run.root / "lightgen_ols_singular_values.pt"
            ),
        },
        gates,
        (),
        "architecture-proxy: unknown wavelength, spacing, aperture, and fiber coupling prohibit physical-faithful/training authority"
        if passed
        else "architecture-proxy observability gate failed",
    )
    _write_record(run, "lightgen_ols_observability_evidence.json", evidence)
    print(run.root)
    return 0


def _zero_basis(shape: tuple[int, int]) -> CodeBasisArtifactV2:
    from ...student.capacity import CodeBasisArtifactV2

    return CodeBasisArtifactV2(
        torch.zeros(10, 8, *shape),
        torch.ones(10, 1, 1),
        8,
        1234,
        "prototype-calibration-zero-basis",
    )


def _bundle_manifest(
    config: V2Config,
    bundle: StudentBundle,
    *,
    dataset_hash: str,
    teacher_manifest_hash: str | None,
    calibration_hash: str,
    basis_hash: str,
) -> CheckpointManifestV2:
    state = bundle.state_dict()
    return CheckpointManifestV2(
        2,
        config.system.kind,
        config.system.boundary,
        _system_signature(config),
        canonical_sha256(config.model_dump(mode="json")),
        dataset_hash,
        teacher_manifest_hash,
        calibration_hash,
        basis_hash,
        state_signature(state),
        state_content_hash(state),
    )


def _memory_report(
    device: torch.device, *, baseline_reserved: int, baseline_allocated: int
) -> tuple[bool, dict[str, float | bool]]:
    if device.type != "cuda":
        return False, {"cuda_measured": False}
    properties = torch.cuda.get_device_properties(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    delta = max(0, peak_reserved - baseline_reserved)
    free_after_model = max(1, properties.total_memory - baseline_reserved)
    passed = (
        delta <= 0.85 * free_after_model
        and peak_reserved <= 0.90 * properties.total_memory
    )
    return passed, {
        "cuda_measured": True,
        "allocated_baseline": float(baseline_allocated),
        "reserved_baseline": float(baseline_reserved),
        "allocated_peak": float(peak_allocated),
        "reserved_peak": float(peak_reserved),
        "reserved_delta_peak": float(delta),
        "total_memory": float(properties.total_memory),
    }


def _preflight_stage(
    name: str,
    operation,
    *,
    device: torch.device,
    baseline_reserved: int,
    baseline_allocated: int,
) -> tuple[GateResult, dict[str, float | bool]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    memory_pass, memory = _memory_report(
        device,
        baseline_reserved=baseline_reserved,
        baseline_allocated=baseline_allocated,
    )
    memory["elapsed_seconds"] = time.perf_counter() - started
    return GateResult(
        f"preflight_{name}",
        memory_pass,
        memory,
        None if memory_pass else "CUDA memory gate was not measured or failed",
    ), memory


def _legacy_teacher(args: argparse.Namespace, config: V2Config, stage: str) -> int:
    command = [
        sys.executable,
        "train.py",
        "--train-config",
        args.legacy_train_config,
        "--stage",
        stage,
        "--device",
        config.run.device,
    ]
    subprocess.run(command, check=True)
    return 0


def train_first_stage(args: argparse.Namespace, config: V2Config) -> int:
    """Descriptor target for the historical first-stage command."""
    return _legacy_teacher(args, config, "first_stage")


def train_diffusion(args: argparse.Namespace, config: V2Config) -> int:
    """Descriptor target for the historical diffusion command."""
    return _legacy_teacher(args, config, "diffusion")


def export_targets(args: argparse.Namespace, config: V2Config) -> int:
    return _teacher_export(args, config)


def manifest(args: argparse.Namespace, config: V2Config) -> int:
    return _teacher_manifest(args, config)


def _teacher_manifest(args: argparse.Namespace, config: V2Config) -> int:
    if config.teacher is None:
        raise ValueError("teacher command requires teacher config")
    run = _new_run(config)
    manifest = build_teacher_pair_manifest(
        config.teacher.first_stage_checkpoint,
        config.teacher.diffusion_checkpoint,
        target_plane=config.teacher.target_plane,
        seed=config.teacher.seed,
        guidance_scale=config.teacher.guidance_scale,
        split=args.split,
        latent_geometry=config.teacher.latent_geometry,
    )
    atomic_write_json(
        run.root / "teacher_pair_manifest.json",
        manifest.to_dict() | {"manifest_hash": manifest.manifest_hash},
    )
    print(run.root / "teacher_pair_manifest.json")
    return 0


def _temporary_output(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as handle:
        return Path(handle.name)


def _teacher_export(args: argparse.Namespace, config: V2Config) -> int:
    if config.teacher is None:
        raise ValueError("teacher export requires teacher config")
    _require_clean_acceptance(config)
    legacy_hash = _legacy_export_contract(args.legacy_train_config, config)
    pair = _pair_manifest(args.teacher_manifest)
    if (
        pair.first_stage_path != config.teacher.first_stage_checkpoint
        or pair.diffusion_path != config.teacher.diffusion_checkpoint
        or pair.seed != config.teacher.seed
        or pair.guidance_scale != config.teacher.guidance_scale
    ):
        raise ValueError(
            "teacher pair manifest does not match v2 teacher configuration"
        )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("teacher target output already exists")
    temporary = _temporary_output(output)
    try:
        legacy_output_root = output.parent.parent
        command = [
            sys.executable,
            "train.py",
            "--train-config",
            args.legacy_train_config,
            "--stage",
            "teacher_distill_export",
            "--device",
            config.run.device,
            "--output-root",
            str(legacy_output_root),
            "--first-stage-checkpoint",
            pair.first_stage_path,
            "--diffusion-checkpoint",
            pair.diffusion_path,
            "--seed",
            str(pair.seed),
            "--guidance-scale",
            str(pair.guidance_scale),
            "--distill-output",
            str(temporary),
        ]
        subprocess.run(command, check=True)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        scoped_export_hash = _teacher_export_contract_hash(config, pair, legacy_hash)
        target = build_teacher_target_manifest(
            temporary,
            pair,
            formal=config.system.is_formal,
            generated_git_commit=commit,
            v2_config_hash=canonical_sha256(config.model_dump(mode="json")),
            legacy_export_config_hash=legacy_hash,
            system_contract_hash=_system_signature(config),
            teacher_export_contract_hash=scoped_export_hash,
        )
        os.replace(temporary, output)
        target = build_teacher_target_manifest(
            output,
            pair,
            formal=config.system.is_formal,
            generated_git_commit=commit,
            v2_config_hash=canonical_sha256(config.model_dump(mode="json")),
            legacy_export_config_hash=legacy_hash,
            system_contract_hash=_system_signature(config),
            teacher_export_contract_hash=scoped_export_hash,
        )
        target_path = Path(args.target_manifest)
        if target_path.exists():
            raise FileExistsError("teacher target manifest already exists")
        atomic_write_json(
            target_path, target.to_dict() | {"manifest_hash": target.manifest_hash}
        )
    finally:
        temporary.unlink(missing_ok=True)
    print(output)
    return 0


def _export_clean_targets(args: argparse.Namespace, config: V2Config) -> int:
    output, manifest_path = Path(args.output), Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise FileExistsError("clean-target output and manifest paths must be new")
    run = _new_run(config)
    targets = build_clean_targets(
        dataset_root=args.dataset_root, image_size=config.system.slm_shape[0]
    )
    if tuple(targets.shape) != (10, 1, *config.system.slm_shape):
        raise ValueError(
            "clean-target geometry does not match the configured SLM plane"
        )
    atomic_torch_save(output, targets)
    reference = tensor_artifact_ref(
        output, role="clean_mnist_targets", git_commit=run.git_commit
    )
    manifest = CleanTargetManifestV2(
        schema_version=2,
        dataset_root=str(Path(args.dataset_root).resolve()),
        dataset_split="train",
        samples_per_label=128,
        labels=tuple(range(10)),
        targets=reference,
        targets_hash=tensor_sha256(targets),
        shape=tuple(targets.shape),
        dtype=str(targets.dtype),
        config_hash=run.config_hash,
        git_commit=run.git_commit,
        system_contract_hash=_system_signature(config),
        producer_config_hash=run.config_hash,
    )
    manifest.validate()
    atomic_write_json(
        manifest_path, manifest.to_dict() | {"manifest_hash": manifest.manifest_hash}
    )
    print(output)
    return 0
