"""CLI implementation for the diagnostic-only capacity-conflict map.

This module deliberately makes unavailable hardware/upper-bound operators an
``invalid_harness`` result.  It never silently replaces them with a different
optical model, and its evidence is excluded from formal v2 acceptance.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.config import FormalSystemConfig
from ldm_project.v2.evidence import safe_torch_load, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256
from ldm_project.v2.optics.system import FormalAmplitudeFiniteCropIdentitySystem
from ldm_project.v2.student.conditional_generator import ConditionalClassConditionedModesGenerator
from ldm_project.v2.student.evaluator import SmoothMNISTEvaluator, load_frozen_evaluator, load_local_mnist, stratified_mnist_indices
from ldm_project.v2.student.identity_generator import IdentityAnchoredConditionalAmplitudeGenerator
from ldm_project.v2.application.research.shared.measurement import MEASUREMENT_CONTRACT, evaluator_features, exposure_sensitivity, measure_candidate, validation_reference
from ldm_project.v2.application.research.shared.models import ShapeMNISTEvaluator, SpatialSeedDecoder
from ldm_project.v2.application.research.lightgen.optics import ComplexControlFiniteCropStack, DiagnosticPhaseStackBypass
from ldm_project.v2.application.research.shared.spec import CapacityMapEvidence, CapacityMapSpec, ElectronicVariantSpec, OpticalVariantSpec, input_hashes, load_capacity_map_spec


def _root(spec: CapacityMapSpec, action: str) -> Path:
    day = datetime.now().strftime("%Y%m%d")
    root = Path(spec.output_root) / f"{day}_{spec.version}" / "runs"
    target = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_research_{action}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def _write(root: Path, name: str, evidence: CapacityMapEvidence) -> Path:
    target = root / name
    atomic_write_json(target, evidence.to_dict())
    return target


def _evidence(spec: CapacityMapSpec, action: str, state: str, args: argparse.Namespace, *, metrics: dict, reason: str | None = None, extra_inputs: dict[str, str] | None = None) -> CapacityMapEvidence:
    revision = int(getattr(args, "evidence_revision", 1))
    if revision < 1:
        raise ValueError("evidence revision must be positive")
    inputs = input_hashes(spec, args.config)
    inputs.update(extra_inputs or {})
    return CapacityMapEvidence(1, action, state, getattr(args, "variant", None), args.dataset, getattr(args, "seed", None), spec.config_hash, inputs, metrics, reason, evidence_revision=revision)


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu only for diagnostic execution-chain smoke")
    return torch.device(value)


def _make_electronic(item: ElectronicVariantSpec) -> nn.Module:
    if item.variant_id == "E0":
        return IdentityAnchoredConditionalAmplitudeGenerator()
    if item.variant_id == "E1":
        return ConditionalClassConditionedModesGenerator()
    return SpatialSeedDecoder(item.code_dim)


def _parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def _load_dataset(spec: CapacityMapSpec, dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset == "mnist":
        return load_local_mnist(spec.dataset_root, train=True)
    try:
        from torchvision.datasets import FashionMNIST
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("torchvision is required for Fashion-MNIST") from error
    if dataset == "fashion_mnist":
        root = Path(spec.dataset_root)
        if not root.exists():
            raise FileNotFoundError("local Fashion-MNIST root does not exist")
        data = FashionMNIST(root=str(root), train=True, download=False)
        return data.data.unsqueeze(1).float().div_(255.0), data.targets.long()
    if dataset != "controlled_shapes":
        raise ValueError(f"unknown capacity-map dataset {dataset}")
    generator = torch.Generator().manual_seed(1234)
    # Fixed, procedural ten-class data; individual factors are retained in the
    # manifest rather than inferred from generated pixels.
    per_class = 6000
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 28), torch.linspace(-1, 1, 28), indexing="ij")
    images, labels = [], []
    for label in range(10):
        values = torch.rand((per_class, 8), generator=generator)
        x = xx.unsqueeze(0) - (values[:, 0:1, None] - .5) * .7
        y = yy.unsqueeze(0) - (values[:, 1:2, None] - .5) * .7
        angle = values[:, 4:5, None] * torch.pi
        xr, yr = x * torch.cos(angle) - y * torch.sin(angle), x * torch.sin(angle) + y * torch.cos(angle)
        radius = .18 + .22 * values[:, 2:3, None]
        # A deterministic family of superellipse/stripe contours per label.
        power = 1.2 + label * .25
        shape = ((xr.abs() / radius).pow(power) + (yr.abs() / (radius * (.6 + values[:, 3:4, None]))).pow(power) < 1).float()
        images.append(shape * (.45 + .55 * values[:, 7:8, None]))
        labels.append(torch.full((per_class,), label, dtype=torch.long))
    return torch.cat(images).unsqueeze(1), torch.cat(labels)


def _load_evaluators(spec: CapacityMapSpec, device: torch.device) -> tuple[nn.Module, nn.Module]:
    if not all((spec.evaluator_checkpoint, spec.smooth_evaluator_checkpoint, spec.evaluator_bundle, spec.evaluator_qualification_evidence)):
        raise ValueError("capacity-map requires a qualified evaluator bundle")
    bundle = json.loads(Path(spec.evaluator_bundle).read_text(encoding="utf-8"))
    qualification = json.loads(Path(spec.evaluator_qualification_evidence).read_text(encoding="utf-8"))
    if qualification.get("status") != "passed":
        raise ValueError("capacity-map evaluator qualification did not pass")
    if bundle.get("primary_checkpoint") != file_sha256(spec.evaluator_checkpoint):
        raise ValueError("capacity-map primary evaluator hash disagrees with bundle")
    if bundle.get("smooth_checkpoint") != file_sha256(spec.smooth_evaluator_checkpoint):
        raise ValueError("capacity-map smooth evaluator hash disagrees with bundle")
    primary, primary_payload = load_frozen_evaluator(spec.evaluator_checkpoint, device=device)
    if bundle.get("split_manifest_hash") != primary_payload.get("split_manifest_hash"):
        raise ValueError("capacity-map evaluator split hash disagrees with bundle")
    payload = safe_torch_load(spec.smooth_evaluator_checkpoint)
    if not isinstance(payload, dict) or payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind or payload.get("split_manifest_hash") != bundle.get("split_manifest_hash"):
        raise ValueError("smooth evaluator checkpoint has an incompatible architecture")
    smooth = SmoothMNISTEvaluator().to(device)
    smooth.load_state_dict(payload["state_dict"], strict=True)
    return primary, smooth.freeze()


def _load_shape_evaluator(checkpoint: str, qualification: str, *, device: torch.device) -> nn.Module:
    evidence = json.loads(Path(qualification).read_text(encoding="utf-8"))
    if evidence.get("action") != "qualify-shape" or evidence.get("state") != "screen_passed":
        raise ValueError("shape evaluator qualification evidence did not pass")
    payload = safe_torch_load(checkpoint)
    if not isinstance(payload, dict) or payload.get("architecture_kind") != ShapeMNISTEvaluator.architecture_kind:
        raise ValueError("shape evaluator checkpoint has an incompatible architecture")
    if evidence.get("metrics", {}).get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("shape evaluator checkpoint does not match its qualification evidence")
    model = ShapeMNISTEvaluator().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.freeze()


def qualify_shape(args: argparse.Namespace) -> int:
    """Train and qualify the third evaluator without changing the v2.1 bundle."""
    spec = load_capacity_map_spec(args.config)
    root = _root(spec, "qualify_shape")
    try:
        if args.dataset != "mnist" or args.seed != 7319:
            raise ValueError("shape evaluator qualification is pre-registered for MNIST with seed 7319")
        device = _device(args.device)
        if device.type != "cuda":
            raise RuntimeError("shape evaluator qualification requires CUDA")
        primary, _ = _load_evaluators(spec, device)
        images, labels = _load_dataset(spec, "mnist")
        train_indices, validation_indices = stratified_mnist_indices(labels)
        model = ShapeMNISTEvaluator().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        for epoch in range(10):
            generator = torch.Generator(device="cpu").manual_seed(7319 + epoch)
            order = train_indices.index_select(0, torch.randperm(train_indices.numel(), generator=generator))
            model.train()
            for start in range(0, order.numel(), 256):
                index = order[start : start + 256]
                optimizer.zero_grad(set_to_none=True)
                _, logits = model(images.index_select(0, index).to(device))
                F.cross_entropy(logits, labels.index_select(0, index).to(device)).backward()
                optimizer.step()
        model.freeze()
        validation, targets = images.index_select(0, validation_indices).to(device), labels.index_select(0, validation_indices).to(device)
        with torch.no_grad():
            _, logits = model(validation)
            _, primary_logits = primary(validation)
        prediction, primary_prediction = logits.argmax(1), primary_logits.argmax(1)
        recall = [float((prediction[targets == label] == label).float().mean().item()) for label in range(10)]
        variants = {
            "intensity_minus_2pct": (validation * .98).clamp(0, 1),
            "intensity_plus_2pct": (validation * 1.02).clamp(0, 1),
            "translate_x_plus_1": torch.roll(validation, 1, dims=-1),
            "translate_y_plus_1": torch.roll(validation, 1, dims=-2),
        }
        with torch.no_grad():
            robustness = {name: float((model(value)[1].argmax(1) == prediction).float().mean().item()) for name, value in variants.items()}
        accuracy, agreement = float((prediction == targets).float().mean().item()), float((prediction == primary_prediction).float().mean().item())
        gates = {
            "validation": accuracy >= .985 and sum(recall) / 10 >= .985 and min(recall) >= .95,
            "robustness": all(value >= (.995 if "intensity" in name else .975) for name, value in robustness.items()),
            "agreement_primary": agreement >= .98,
        }
        checkpoint = root / "shape_mnist_evaluator_v22.pt"
        atomic_torch_save(checkpoint, {"architecture_kind": ShapeMNISTEvaluator.architecture_kind, "state_dict": model.state_dict(), "training_seed": 7319, "validation_accuracy": accuracy, "split_manifest_hash": json.loads(Path(spec.evaluator_bundle).read_text(encoding="utf-8"))["split_manifest_hash"], "contract": MEASUREMENT_CONTRACT})
        metrics = {"accuracy": accuracy, "recall": recall, "robustness": robustness, "agreement_primary": agreement, "gates": gates, "checkpoint_sha256": file_sha256(checkpoint), "checkpoint": str(checkpoint)}
        state, reason = ("screen_passed", None) if all(gates.values()) else ("screened_out", "shape_evaluator_qualification_gate_failed")
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(root, "capacity_map_shape_evaluator_qualification_evidence.json", _evidence(spec, "qualify-shape", state, args, metrics=metrics, reason=reason))
    print(path)
    return 0 if state == "screen_passed" else 1


def _build_optical(optical: OpticalVariantSpec, *, device: torch.device) -> tuple[nn.Module | None, str]:
    if optical.variant_id == "O0":
        return None, "electronic_oracle"
    if optical.variant_id in {"O1", "O2", "O3", "O4", "O5"}:
        if device.type != "cuda":
            raise RuntimeError(f"{optical.variant_id} requires CUDA full-grid propagation; CPU is not scientific evidence")
        kind = "diagnostic_amplitude_near_conjugate_relay_identity_phase_capacity_v7_2" if optical.variant_id == "O3" else "diagnostic_amplitude_finite_crop_identity"
        geometry = {}
        if optical.variant_id == "O3":
            geometry = {"encoder_layers": 4, "decoder_layers": 4}
        if optical.variant_id == "O4":
            # 200 px at 1 um retains the fixed 200 um active FOV. The relay
            # magnifications map the existing 500 um SLM/camera FOV to it.
            geometry = {"metasurface_shape": (200, 200), "metasurface_pixel_pitch_m": 1.0e-6}
        system = FormalAmplitudeFiniteCropIdentitySystem(FormalSystemConfig(kind=kind, propagation_grid_size=2048, **geometry)).to(device)
        if optical.variant_id == "O1":
            return DiagnosticPhaseStackBypass(system), "diagnostic_phase_stack_bypass_raw_cmos"
        if optical.variant_id == "O5":
            system.encoder = ComplexControlFiniteCropStack(layers=2, active_shape=(100, 100), propagation_shape=(2048, 2048), spacing_m=system.config.layer_spacing_m).to(device)
            system.decoder = ComplexControlFiniteCropStack(layers=2, active_shape=(100, 100), propagation_shape=(2048, 2048), spacing_m=system.config.layer_spacing_m).to(device)
        return system, "raw_cmos_area_readout"
    raise RuntimeError(f"{optical.variant_id} has no validated diagnostic operator yet")


def _optical_output(optical: OpticalVariantSpec, command: torch.Tensor, *, device: torch.device, system: nn.Module | None = None) -> tuple[torch.Tensor, str, dict[str, float]]:
    """Run only pre-registered operators, retaining their output provenance."""
    system, domain = _build_optical(optical, device=device) if system is None and optical.variant_id != "O0" else (system, "electronic_oracle" if optical.variant_id == "O0" else "raw_cmos_area_readout")
    if optical.variant_id == "O0":
        return F.interpolate(command, size=(28, 28), mode="area"), domain, {}
    if system is not None:
        result = system(system.command(command))
        raw = result.raw_camera.tensor
        return F.interpolate(raw, size=(28, 28), mode="area"), domain, {
            "encoder_crop_retention": float(result.crop_statistics["encoder_crop_retention"].mean().item()),
            "decoder_crop_retention": float(result.crop_statistics["decoder_crop_retention"].mean().item()),
            "raw_code_fraction": float(raw.std(dim=0, unbiased=False).mean().div(raw.mean().clamp_min(1e-8)).item()),
        }
    raise AssertionError("non-oracle optical variants require a constructed system")


def _metrics(images: torch.Tensor, labels: torch.Tensor, primary: nn.Module, smooth: nn.Module) -> dict[str, object]:
    with torch.no_grad():
        primary_features, primary_logits = primary(images)
        smooth_features, smooth_logits = smooth(images)
    prediction, smooth_prediction = primary_logits.argmax(1), smooth_logits.argmax(1)
    recall = [float((prediction[labels == label] == label).float().mean().item()) for label in range(10)]
    centered = primary_features - primary_features.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered) if centered.shape[0] > 1 else torch.zeros(1, device=images.device)
    return {
        "primary_accuracy": float((prediction == labels).float().mean().item()),
        "smooth_accuracy": float((smooth_prediction == labels).float().mean().item()),
        "macro_recall": float(sum(recall) / len(recall)),
        "minimum_per_class_recall": min(recall),
        "evaluator_agreement": float((prediction == smooth_prediction).float().mean().item()),
        "saturation": float((images >= 1).float().mean().item()),
        "feature_effective_rank": int((singular > singular.max().clamp_min(1e-8) * 1e-3).sum().item()),
        "feature_singular_values": [float(value) for value in singular[:16].cpu()],
    }


def validate(args: argparse.Namespace) -> int:
    spec = load_capacity_map_spec(args.config)
    root = _root(spec, "validate")
    errors: list[str] = []
    try:
        _device(args.device)
        if args.dataset in {"mnist", "fashion_mnist"} and not Path(spec.dataset_root).exists():
            errors.append("dataset_root_missing")
        if args.dataset == "mnist":
            _load_evaluators(spec, torch.device("cpu"))
        if args.variant:
            electronic, optical = spec.variant(args.variant)
            model = _make_electronic(electronic)
            if electronic.kind == "spatial_decoder" and _parameter_count(model) > electronic.parameter_budget:
                errors.append("electronic_parameter_budget_exceeded")
            optical.validate()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        errors.append(str(error))
    state = "invalid_harness" if errors else "screen_passed"
    evidence = _evidence(spec, "validate", state, args, metrics={"declared_matrix_size": len(spec.electronic) * len(spec.optical), "errors": errors}, reason="; ".join(errors) or None)
    atomic_write_json(root / "capacity_map_manifest.json", {
        "schema_version": 1,
        "config_hash": spec.config_hash,
        "config_inputs": input_hashes(spec, args.config),
        "dataset": args.dataset,
        "matrix": [f"{electronic.variant_id}:{optical.variant_id}" for electronic in spec.electronic for optical in spec.optical],
        "lifecycle": "diagnostic_research_only",
        "formal_promotion_authorized": False,
    })
    path = _write(root, "capacity_map_validate_evidence.json", evidence)
    print(path)
    return 1 if errors else 0


def screen(args: argparse.Namespace) -> int:
    spec = load_capacity_map_spec(args.config)
    root = _root(spec, "screen")
    electronic, optical = spec.variant(args.variant)
    if getattr(args, "stage", "audit") == "probe":
        try:
            _require_prior(args, "screen probe", spec)
            metrics = _smoke_train(spec, electronic, optical, dataset=args.dataset, seed=args.seed, device=_device(args.device), root=root, steps=spec.screening_probe_steps)
            baseline, final = metrics["baseline"], metrics["final"]
            improved = metrics["improvement"] > 0
            safe = (
                float(final["minimum_per_class_recall"]) >= float(baseline["minimum_per_class_recall"]) - 0.02
                and float(final["saturation"]) <= 0.08
            )
            state = "screen_passed" if improved and safe else "screened_out"
            reason = None if state == "screen_passed" else "probe_improvement_or_safety_gate_failed"
            metrics["probe_steps"] = spec.screening_probe_steps
            metrics["prior_evidence_sha256"] = file_sha256(args.prior_evidence)
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            metrics, state, reason = {}, "invalid_harness", str(error)
        path = _write(root, f"capacity_map_{args.variant.replace(':', '_')}_probe_evidence.json", _evidence(spec, "screen_probe", state, args, metrics=metrics, reason=reason))
        print(path)
        return 0 if state == "screen_passed" else 1
    try:
        device, model = _device(args.device), _make_electronic(electronic).to(_device(args.device))
        if args.preflight_only:
            if optical.variant_id == "O0":
                raise ValueError("O0 has no CUDA optical preflight; run the normal no-update screen")
            torch.cuda.reset_peak_memory_stats(device)
            labels = torch.zeros(1, dtype=torch.long, device=device)
            codes = torch.randn((1, electronic.code_dim), generator=torch.Generator(device=device).manual_seed(args.seed), device=device)
            system, domain = _build_optical(optical, device=device)
            with torch.no_grad():
                _, _, physical = _optical_output(optical, model(codes, labels), device=device, system=system)
            metrics = {"preflight_only": True, "output_domain": domain, "parameter_count": _parameter_count(model), "cuda_preflight_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), **physical}
            state, reason = "screen_passed", None
            path = _write(root, f"capacity_map_{args.variant.replace(':', '_')}_screen_evidence.json", _evidence(spec, "screen", state, args, metrics=metrics, reason=reason))
            print(path)
            return 0
        primary, smooth = _load_evaluators(spec, device)
        real_images, real_labels = _load_dataset(spec, args.dataset)
        labels = torch.arange(10).repeat_interleave(spec.batch_size // 10).to(device)
        codes = torch.randn((labels.numel(), electronic.code_dim), generator=torch.Generator(device=device).manual_seed(args.seed), device=device)
        command = model(codes, labels)
        if optical.variant_id == "O0":
            output, domain, physical = _optical_output(optical, command, device=device)
        else:
            # The single-sample call is the required CUDA memory preflight.
            torch.cuda.reset_peak_memory_stats(device)
            system, domain = _build_optical(optical, device=device)
            fragments = []
            physical = {}
            with torch.no_grad():
                for index in range(command.shape[0]):
                    value, _, statistics = _optical_output(optical, command[index : index + 1], device=device, system=system)
                    fragments.append(value)
                    if index == 0:
                        physical = statistics | {"cuda_preflight_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))}
            output = torch.cat(fragments, dim=0)
        metrics = _metrics(output.clamp(0, 1), labels, primary, smooth)
        # This is intentionally a no-update capacity audit. The separately
        # selected 50-step probe is a future action bound to this evidence;
        # running an optimizer here would invalidate the promised full 30-cell
        # screening matrix.
        metrics.update({"output_domain": domain, "parameter_count": _parameter_count(model), "real_sample_count": int(real_images.shape[0]), "dataset_label_hash": canonical_sha256(real_labels[:1000].tolist()), "optimizer_steps": 0, "screening_kind": "no_update_capacity_audit", **physical})
        state = "screen_passed"
        reason = None
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(root, f"capacity_map_{args.variant.replace(':', '_')}_screen_evidence.json", _evidence(spec, "screen", state, args, metrics=metrics, reason=reason))
    print(path)
    return 0 if state == "screen_passed" else 1


def _require_prior(args: argparse.Namespace, expected: str, spec: CapacityMapSpec | None = None) -> None:
    if not args.prior_evidence:
        raise ValueError(f"{expected} requires --prior-evidence from its preceding stage")
    payload = json.loads(Path(args.prior_evidence).read_text(encoding="utf-8"))
    if payload.get("state") not in {"screen_passed", "smoke_passed"}:
        raise ValueError("preceding evidence did not pass; research variants stop locally")
    if payload.get("variant") != args.variant or payload.get("dataset") != args.dataset:
        raise ValueError("preceding evidence variant/dataset mismatch")
    if spec is not None and payload.get("config_hash") != spec.config_hash:
        raise ValueError("preceding evidence belongs to another capacity-map revision")
    if payload.get("content_hash") != canonical_sha256({key: value for key, value in payload.items() if key != "content_hash"}):
        raise ValueError("preceding evidence content hash mismatch")


def _balanced_real(images: torch.Tensor, labels: torch.Tensor, *, batch_size: int, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    per_label = batch_size // 10
    selected = []
    for label in range(10):
        candidates = torch.nonzero(labels == label, as_tuple=False).flatten()
        selected.append(candidates.index_select(0, torch.randint(candidates.numel(), (per_label,), generator=generator)))
    indices = torch.cat(selected)
    return images.index_select(0, indices).to(device), labels.index_select(0, indices).to(device)


def _class_real(images: torch.Tensor, labels: torch.Tensor, *, label: int, count: int, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.nonzero(labels == label, as_tuple=False).flatten()
    indices = candidates.index_select(0, torch.randint(candidates.numel(), (count,), generator=generator))
    return images.index_select(0, indices).to(device), labels.index_select(0, indices).to(device)


def _evaluation_output(model: nn.Module, system: nn.Module | None, optical: OpticalVariantSpec, *, code_dim: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(10, device=device)
    codes = torch.randn((10, code_dim), generator=torch.Generator(device=device).manual_seed(seed + 9_999), device=device)
    commands = model(codes, labels)
    with torch.no_grad():
        if system is None:
            output, _, _ = _optical_output(optical, commands, device=device)
        else:
            fragments = [
                _optical_output(optical, commands[index : index + 1], device=device, system=system)[0]
                for index in range(commands.shape[0])
            ]
            output = torch.cat(fragments, dim=0)
    return output.clamp(0, 1), labels


def _distribution_loss(generated: torch.Tensor, real: torch.Tensor, labels: torch.Tensor, evaluator: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    generated_features, generated_logits = evaluator(generated)
    with torch.no_grad():
        real_features, _ = evaluator(real)
    semantic = F.cross_entropy(generated_logits, labels)
    means, variances = [], []
    for label in labels.unique(sorted=True).tolist():
        generated_label = generated_features[labels == label]
        real_label = real_features[labels == label]
        means.append(F.mse_loss(generated_label.mean(0), real_label.mean(0)))
        variances.append(F.mse_loss(generated_label.var(0, unbiased=False), real_label.var(0, unbiased=False)))
    distribution = torch.stack(means).mean() + torch.stack(variances).mean()
    saturation = (generated >= 1).float().mean()
    loss = semantic + distribution + 0.1 * saturation
    return loss, {"semantic_loss": float(semantic.detach().item()), "distribution_loss": float(distribution.detach().item()), "saturation_loss": float(saturation.detach().item())}


def _real_real_distribution_baseline(images: torch.Tensor, labels: torch.Tensor, evaluator: nn.Module, *, device: torch.device) -> torch.Tensor:
    """A fixed, evaluator-specific scale for the v2.2 dual-bundle loss."""
    reference, reference_labels = validation_reference(images, labels, device=device)
    features, _ = evaluator_features(evaluator, reference)
    terms = []
    for label in range(10):
        rows = features[reference_labels == label]
        first, second = rows[:250], rows[250:500]
        terms.append(F.mse_loss(first.mean(0), second.mean(0)) + F.mse_loss(first.var(0, unbiased=False), second.var(0, unbiased=False)))
    return torch.stack(terms).mean().detach().clamp_min(1.0e-8)


def _objective_loss(
    generated: torch.Tensor,
    real: torch.Tensor,
    labels: torch.Tensor,
    *,
    primary: nn.Module,
    smooth: nn.Module,
    objective_id: str,
    primary_baseline: torch.Tensor,
    smooth_baseline: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pre-registered primary-only or normalized dual-bundle objective."""
    primary_loss, primary_detail = _distribution_loss(generated, real, labels, primary)
    if objective_id == "primary_only":
        return primary_loss, {f"primary_{key}": value for key, value in primary_detail.items()}
    smooth_loss, smooth_detail = _distribution_loss(generated, real, labels, smooth)
    # Each helper contains CE + distribution + saturation.  Rebuild the
    # distribution-normalized scalar explicitly to keep the specified weights.
    primary_semantic = torch.as_tensor(primary_detail["semantic_loss"], device=generated.device)
    smooth_semantic = torch.as_tensor(smooth_detail["semantic_loss"], device=generated.device)
    # The detached summaries above cannot carry gradients, so recompute the
    # two differentiable components with their fixed real-real scales.
    def scaled(evaluator: nn.Module, baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        generated_features, logits = evaluator(generated)
        with torch.no_grad():
            real_features, _ = evaluator(real)
        means, variances = [], []
        for label in labels.unique(sorted=True).tolist():
            generated_label, real_label = generated_features[labels == label], real_features[labels == label]
            means.append(F.mse_loss(generated_label.mean(0), real_label.mean(0)))
            variances.append(F.mse_loss(generated_label.var(0, unbiased=False), real_label.var(0, unbiased=False)))
        return F.cross_entropy(logits, labels), (torch.stack(means).mean() + torch.stack(variances).mean()) / baseline
    p_semantic, p_distribution = scaled(primary, primary_baseline)
    s_semantic, s_distribution = scaled(smooth, smooth_baseline)
    saturation = (generated >= 1).float().mean()
    loss = .5 * (p_semantic + p_distribution) + .5 * (s_semantic + s_distribution) + .1 * saturation
    return loss, {
        "primary_semantic_loss": float(p_semantic.detach().item()),
        "primary_distribution_loss": float(p_distribution.detach().item()),
        "smooth_semantic_loss": float(s_semantic.detach().item()),
        "smooth_distribution_loss": float(s_distribution.detach().item()),
        "saturation_loss": float(saturation.detach().item()),
        "primary_raw_loss": float(primary_loss.detach().item()),
        "smooth_raw_loss": float(smooth_loss.detach().item()),
        "primary_summary_semantic": float(primary_semantic.item()),
        "smooth_summary_semantic": float(smooth_semantic.item()),
    }


def _panel_summary(measurement: dict[str, object]) -> dict[str, float]:
    panels = measurement["panels"]
    assert isinstance(panels, dict)
    primary = min(float(row["primary"]["accuracy"]) for row in panels.values())
    smooth = min(float(row["smooth"]["accuracy"]) for row in panels.values())
    minimum_recall = min(min(row[which]["recall"]) for row in panels.values() for which in ("primary", "smooth"))
    agreement = min(float(row["agreement"]) for row in panels.values())
    saturation = max(float(row["saturation"]) for row in panels.values())
    code_fraction = min(float(row["digital_output_code_fraction"]) for row in panels.values())
    margin = min((primary - .95) / .05, (smooth - .95) / .05, (minimum_recall - .85) / .15, (agreement - .80) / .20, (.08 - saturation) / .08, (code_fraction - .10) / .10)
    return {"primary_accuracy": primary, "smooth_accuracy": smooth, "minimum_per_class_recall": minimum_recall, "evaluator_agreement": agreement, "saturation": saturation, "code_fraction": code_fraction, "worst_normalized_margin": margin}


def _full_measurement_gates(measurement: dict[str, object]) -> dict[str, bool]:
    summary = _panel_summary(measurement)
    panels = measurement["panels"]
    assert isinstance(panels, dict)
    anti_template = all(
        all(value["feature"]["rank"] >= 8 and value["image"]["rank"] >= 8 and value["unique_quantized"] >= 8 for value in row["anti_template"][which].values())
        for row in panels.values()
        for which in ("primary", "smooth")
    )
    diversity = all(
        min(row[f"{which}_distribution"]["per_class_distance_ratio"]) >= .35
        and min(row[f"{which}_distribution"]["precision"]) >= .10
        and min(row[f"{which}_distribution"]["recall"]) >= .10
        for row in panels.values()
        for which in ("primary", "smooth")
    )
    return {
        "primary_semantics": summary["primary_accuracy"] >= .95,
        "smooth_semantics": summary["smooth_accuracy"] >= .95,
        "minimum_recall": summary["minimum_per_class_recall"] >= .85,
        "cross_evaluator_agreement": summary["evaluator_agreement"] >= .80,
        "distribution": diversity,
        "anti_template": anti_template,
        "code_response": summary["code_fraction"] >= .10,
        "saturation": summary["saturation"] <= .08,
    }


def _confirmation_gates(metrics: dict[str, object]) -> dict[str, bool]:
    """Strict final semantic/safety gates for research confirmation."""
    return {
        "primary_semantics": float(metrics["primary_accuracy"]) >= 0.95,
        "smooth_semantics": float(metrics["smooth_accuracy"]) >= 0.95,
        "minimum_recall": float(metrics["minimum_per_class_recall"]) >= 0.85,
        "cross_evaluator_agreement": float(metrics["evaluator_agreement"]) >= 0.80,
        "saturation": float(metrics["saturation"]) <= 0.08,
    }


def _smoke_train(spec: CapacityMapSpec, electronic: ElectronicVariantSpec, optical: OpticalVariantSpec, *, dataset: str, seed: int, device: torch.device, root: Path, steps: int | None = None, resume_checkpoint: str | None = None) -> dict[str, object]:
    if dataset != "mnist":
        raise RuntimeError("external datasets require their own qualified evaluator bundle before smoke training")
    if device.type != "cuda":
        raise RuntimeError("400-step smoke requires CUDA")
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    code_generator = torch.Generator(device=device).manual_seed(seed)
    model = _make_electronic(electronic).to(device)
    system, domain = _build_optical(optical, device=device)
    primary, smooth = _load_evaluators(spec, device)
    real_images, real_labels = _load_dataset(spec, dataset)
    full_measurement = spec.measurement_contract == MEASUREMENT_CONTRACT
    if full_measurement and (electronic.variant_id != "E2" or optical.variant_id != "O0"):
        raise RuntimeError("v2.2 full-panel reconciliation authorizes E2:O0 only; physical training remains stopped")
    primary_baseline = _real_real_distribution_baseline(real_images, real_labels, primary, device=device)
    smooth_baseline = _real_real_distribution_baseline(real_images, real_labels, smooth, device=device)
    parameters = list(model.parameters()) + ([] if system is None else [value for value in system.parameters() if value.requires_grad])
    optimizer = torch.optim.Adam(parameters, lr=2.0e-4)
    history: list[dict[str, float]] = []
    baseline: dict[str, object] | None = None
    optimizer_steps = spec.smoke_steps if steps is None else steps
    start_step = 0
    if resume_checkpoint:
        resumed = safe_torch_load(resume_checkpoint, map_location=device)
        if resumed.get("config_hash") != spec.config_hash or resumed.get("variant") != f"{electronic.variant_id}:{optical.variant_id}" or int(resumed.get("seed", -1)) != seed:
            raise ValueError("resume checkpoint does not match capacity-map variant/config/seed")
        model.load_state_dict(resumed["electronic"], strict=True)
        if system is not None:
            if resumed.get("optical") is None:
                raise ValueError("resume checkpoint lacks optical state")
            system.load_state_dict(resumed["optical"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        start_step = int(resumed["next_step"])
        history = list(resumed.get("history", []))
        baseline = resumed.get("baseline")
        generator.set_state(resumed["cpu_generator_state"])
        code_generator.set_state(resumed["code_generator_state"])
        if not 0 < start_step <= optimizer_steps:
            raise ValueError("resume checkpoint next_step is outside this run budget")

    def save_resume(next_step: int) -> None:
        atomic_torch_save(root / "capacity_map_resume.pt", {
            "schema_version": 1,
            "config_hash": spec.config_hash,
            "variant": f"{electronic.variant_id}:{optical.variant_id}",
            "seed": seed,
            "next_step": next_step,
            "electronic": model.state_dict(),
            "optical": None if system is None else system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "baseline": baseline,
            "cpu_generator_state": generator.get_state(),
            "code_generator_state": code_generator.get_state(),
        })

    physical = system is not None

    def render(codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        commands = model(codes, labels)
        if system is None:
            return _optical_output(optical, commands, device=device)[0]
        return torch.cat([_optical_output(optical, commands[index : index + 1], device=device, system=system)[0] for index in range(commands.shape[0])])

    def evaluate_full() -> dict[str, object]:
        measured = measure_candidate(model=model, render=render, real_images=real_images, real_labels=real_labels, primary=primary, smooth=smooth, device=device, output_domain=domain)
        return {"measurement": measured, "summary": _panel_summary(measured)}

    for step in range(start_step, optimizer_steps + 1):
        if physical:
            real, labels = _class_real(real_images, real_labels, label=step % 10, count=2, generator=generator, device=device)
        else:
            real, labels = _balanced_real(real_images, real_labels, batch_size=spec.batch_size, generator=generator, device=device)
        codes = torch.randn((labels.numel(), electronic.code_dim), generator=code_generator, device=device)
        commands = model(codes, labels)
        generated, _, _ = _optical_output(optical, commands, device=device, system=system)
        generated = generated.clamp(0, 1)
        if step == 0 and baseline is None:
            if full_measurement:
                baseline = evaluate_full()
            else:
                evaluated, evaluated_labels = _evaluation_output(model, system, optical, code_dim=electronic.code_dim, seed=seed, device=device)
                baseline = _metrics(evaluated, evaluated_labels, primary, smooth)
        if step == optimizer_steps:
            if full_measurement:
                final = evaluate_full()
            else:
                evaluated, evaluated_labels = _evaluation_output(model, system, optical, code_dim=electronic.code_dim, seed=seed, device=device)
                final = _metrics(evaluated, evaluated_labels, primary, smooth)
            break
        optimizer.zero_grad(set_to_none=True)
        loss, detail = _objective_loss(generated, real, labels, primary=primary, smooth=smooth, objective_id=spec.objective_id, primary_baseline=primary_baseline, smooth_baseline=smooth_baseline)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite teacher-free smoke loss")
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            history.append({"step": float(step), "loss": float(loss.detach().item()), **detail})
        if (step + 1) % 400 == 0:
            save_resume(step + 1)
    assert baseline is not None
    checkpoint = root / "capacity_map_smoke_checkpoint.pt"
    atomic_torch_save(checkpoint, {"electronic": model.state_dict(), "optical": None if system is None else system.state_dict(), "variant": f"{electronic.variant_id}:{optical.variant_id}", "seed": seed, "config_hash": spec.config_hash, "objective_id": spec.objective_id, "measurement_contract": spec.measurement_contract})
    if full_measurement:
        baseline_summary, final_summary = baseline["summary"], final["summary"]
        improvement = float(final_summary["worst_normalized_margin"]) - float(baseline_summary["worst_normalized_margin"])
    else:
        improvement = max(float(final["primary_accuracy"]) - float(baseline["primary_accuracy"]), float(final["feature_effective_rank"]) - float(baseline["feature_effective_rank"]))
    return {"baseline": baseline, "final": final, "history": history, "output_domain": domain, "objective_id": spec.objective_id, "measurement_contract": spec.measurement_contract, "real_real_distribution_baselines": {"primary": float(primary_baseline.item()), "smooth": float(smooth_baseline.item())}, "improvement": improvement, "checkpoint_sha256": file_sha256(checkpoint), "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "optimizer_steps": optimizer_steps}


def train_smoke(args: argparse.Namespace) -> int:
    spec = load_capacity_map_spec(args.config)
    root = _root(spec, "train_smoke")
    try:
        _require_prior(args, "train-smoke", spec)
        electronic, optical = spec.variant(args.variant)
        metrics = _smoke_train(spec, electronic, optical, dataset=args.dataset, seed=args.seed, device=_device(args.device), root=root, resume_checkpoint=args.resume_checkpoint)
        metrics["prior_evidence_sha256"] = file_sha256(args.prior_evidence)
        final_summary = metrics["final"].get("summary", metrics["final"])
        state = "smoke_passed" if metrics["improvement"] > 0 and float(final_summary["saturation"]) <= 0.08 else "smoke_failed"
        reason = None if state == "smoke_passed" else "pre_registered_smoke_improvement_or_saturation_gate_failed"
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(root, f"capacity_map_{args.variant.replace(':', '_')}_smoke_evidence.json", _evidence(spec, "train-smoke", state, args, metrics=metrics, reason=reason))
    print(path)
    return 0 if state == "smoke_passed" else 1


def confirm(args: argparse.Namespace) -> int:
    spec = load_capacity_map_spec(args.config)
    root = _root(spec, "confirm")
    try:
        _require_prior(args, "confirm", spec)
        electronic, optical = spec.variant(args.variant)
        metrics = _smoke_train(spec, electronic, optical, dataset=args.dataset, seed=args.seed, device=_device(args.device), root=root, steps=spec.confirm_epochs * spec.smoke_steps, resume_checkpoint=args.resume_checkpoint)
        metrics["epochs"] = spec.confirm_epochs
        metrics["prior_evidence_sha256"] = file_sha256(args.prior_evidence)
        if spec.measurement_contract == MEASUREMENT_CONTRACT:
            metrics["confirmation_gates"] = _full_measurement_gates(metrics["final"]["measurement"])
        else:
            metrics["confirmation_gates"] = _confirmation_gates(metrics["final"])
        state = "confirmed" if all(metrics["confirmation_gates"].values()) else "smoke_failed"
        reason = None if state == "confirmed" else "confirmation_final_gate_failed"
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(root, f"capacity_map_{args.variant.replace(':', '_')}_confirm_evidence.json", _evidence(spec, "confirm", state, args, metrics=metrics, reason=reason))
    print(path)
    return 0 if state == "confirmed" else 1


def diagnose(args: argparse.Namespace) -> int:
    """Replay a bound O0 checkpoint through the immutable v2.2 measurement."""
    spec = load_capacity_map_spec(args.config)
    root = _root(spec, "diagnose")
    try:
        if args.dataset != "mnist" or args.variant != "E2:O0":
            raise ValueError("v2.2 diagnose is pre-registered for the E2:O0 MNIST oracle")
        device = _device(args.device)
        source = json.loads(Path(args.source_evidence).read_text(encoding="utf-8"))
        if source.get("content_hash") != canonical_sha256({key: value for key, value in source.items() if key != "content_hash"}):
            raise ValueError("source evidence content hash mismatch")
        if source.get("action") != "confirm" or source.get("variant") != args.variant:
            raise ValueError("diagnose source must be an E2:O0 confirm evidence record")
        checkpoint = safe_torch_load(args.checkpoint, map_location=device)
        if not isinstance(checkpoint, dict) or checkpoint.get("variant") != args.variant:
            raise ValueError("diagnose checkpoint does not match E2:O0")
        if source.get("metrics", {}).get("checkpoint_sha256") != file_sha256(args.checkpoint):
            raise ValueError("diagnose checkpoint hash does not match source evidence")
        electronic, optical = spec.variant(args.variant)
        model = _make_electronic(electronic).to(device)
        model.load_state_dict(checkpoint["electronic"], strict=True)
        primary, smooth = _load_evaluators(spec, device)
        shape = None
        if bool(args.shape_evaluator_checkpoint) != bool(args.shape_qualification_evidence):
            raise ValueError("shape checkpoint and qualification evidence must be supplied together")
        if args.shape_evaluator_checkpoint:
            shape = _load_shape_evaluator(args.shape_evaluator_checkpoint, args.shape_qualification_evidence, device=device)
        real_images, real_labels = _load_dataset(spec, "mnist")
        metrics = measure_candidate(model=model, render=lambda codes, labels: _optical_output(optical, model(codes, labels), device=device)[0], real_images=real_images, real_labels=real_labels, primary=primary, smooth=smooth, shape=shape, device=device, output_domain="electronic_oracle")
        metrics["summary"] = _panel_summary(metrics)
        metrics["full_measurement_gates"] = _full_measurement_gates(metrics)
        metrics["exposure_sensitivity"] = exposure_sensitivity(render=lambda codes, labels: _optical_output(optical, model(codes, labels), device=device)[0], primary=primary, smooth=smooth, device=device)
        metrics["objective_id"] = checkpoint.get("objective_id", "historical_primary_only")
        state, reason = "screen_passed", None
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    extra = {"source_evidence": file_sha256(args.source_evidence), "source_checkpoint": file_sha256(args.checkpoint)}
    if args.shape_evaluator_checkpoint:
        extra["shape_evaluator"] = file_sha256(args.shape_evaluator_checkpoint)
        extra["shape_qualification"] = file_sha256(args.shape_qualification_evidence)
    path = _write(root, "capacity_map_E2_O0_diagnose_evidence.json", _evidence(spec, "diagnose", state, args, metrics=metrics, reason=reason, extra_inputs=extra))
    print(path)
    return 0 if state == "screen_passed" else 1


def _evidence_key(item: dict) -> tuple[object, ...]:
    return (item.get("config_hash"), item.get("dataset"), item.get("variant"), item.get("action"), item.get("seed"))


def _read_capacity_evidence(path: Path) -> dict | None:
    """Read only self-authenticating capacity evidence records."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "action" not in payload or "state" not in payload:
        return None
    if payload.get("content_hash") != canonical_sha256({key: value for key, value in payload.items() if key != "content_hash"}):
        return None
    return payload | {"path": str(path)}


def _latest_evidence(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Select a revision only when that revision has a unique content hash.

    An invalid record is intentionally not discarded here: it remains visible
    even when a later valid run shares the same lifecycle key.
    """
    grouped: dict[tuple[object, ...], list[dict]] = {}
    for item in records:
        grouped.setdefault(_evidence_key(item), []).append(item)
    selected, ambiguous = [], []
    for key, items in grouped.items():
        revision = max(int(item.get("evidence_revision", 1)) for item in items)
        candidates = [item for item in items if int(item.get("evidence_revision", 1)) == revision]
        hashes = sorted({str(item["content_hash"]) for item in candidates})
        if len(hashes) != 1:
            ambiguous.append({
                "key": list(key), "evidence_revision": revision,
                "content_hashes": hashes,
                "paths": sorted(str(item["path"]) for item in candidates),
                "classification": "ambiguous_revision",
            })
            continue
        selected.append(sorted(candidates, key=lambda item: str(item["path"]))[0])
    return selected, ambiguous


def _scan_evidence(source: Path, spec: CapacityMapSpec) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    records, legacy, preliminary = [], [], []
    for path in source.rglob("capacity_map_*.json"):
        payload = _read_capacity_evidence(path)
        if payload is None:
            continue
        is_preliminary = (
            int(payload.get("evidence_revision", 1)) >= 9
            and payload.get("action") in {"screen", "train-smoke", "confirm"}
            and payload.get("metrics", {}).get("measurement_contract") != MEASUREMENT_CONTRACT
        )
        if payload.get("config_hash") == spec.config_hash and not is_preliminary:
            records.append(payload)
        elif int(payload.get("evidence_revision", 1)) <= 8:
            legacy.append({"path": str(path), "config_hash": payload.get("config_hash"), "evidence_revision": payload.get("evidence_revision"), "classification": "legacy_evaluator_harness"})
        else:
            preliminary.append({"path": str(path), "config_hash": payload.get("config_hash"), "evidence_revision": payload.get("evidence_revision"), "classification": "preliminary_simplified_measurement"})
    latest, ambiguous = _latest_evidence(records)
    return latest, legacy, preliminary, ambiguous


def _invalid_revisions(records: list[dict]) -> list[dict]:
    return [
        {"path": item["path"], "key": list(_evidence_key(item)), "evidence_revision": item.get("evidence_revision", 1), "content_hash": item["content_hash"], "reason": item.get("reason")}
        for item in records if item.get("state") == "invalid_harness"
    ]


def summarize(args: argparse.Namespace) -> int:
    spec = load_capacity_map_spec(args.config)
    root = _root(spec, "summarize")
    source = Path(args.evidence_root or spec.output_root)
    all_records = [item for path in source.rglob("capacity_map_*.json") if (item := _read_capacity_evidence(path)) is not None and item.get("config_hash") == spec.config_hash]
    records, legacy, preliminary, ambiguous = _scan_evidence(source, spec)
    valid_records = [item for item in records if item["state"] != "invalid_harness" and not item.get("metrics", {}).get("preflight_only", False)]
    pareto_records = [item for item in valid_records if item.get("action") == "screen" and item.get("metrics", {}).get("screening_kind") == "no_update_capacity_audit"]
    combinations = {item.get("variant") for item in pareto_records if item.get("variant")}
    claim, attribution = "inconclusive", "no capacity attribution is permitted before the pre-registered optical bundles and multi-seed confirmations exist"
    confirmations = [item for item in records if item.get("action") == "confirm" and item.get("variant") == "E2:O0" and isinstance(item.get("metrics", {}).get("confirmation_gates"), dict)]
    if len(confirmations) >= 3:
        passed = sum(all(item["metrics"]["confirmation_gates"].values()) for item in confirmations)
        if passed < 2:
            claim = "training_objective_conflict"
            attribution = "the v2.2 E2:O0 dual-bundle three-seed confirmation did not reach the complete qualified dual-evaluator gate; physical capacity attribution and downstream branches remain stopped"
    for item in records:
        gates = item.get("metrics", {}).get("confirmation_gates")
        if claim == "inconclusive" and isinstance(gates, dict) and gates.get("primary_semantics") and not gates.get("smooth_semantics"):
            claim, attribution = "representation_readout_mismatch", "primary-only confirmation is excluded because the qualified smooth evaluator did not meet the final semantic gate"
            break
    summary = {"schema_version": 4, "config_hash": spec.config_hash, "records": records, "legacy_evaluator_harness": legacy, "preliminary_simplified_measurement": preliminary, "ambiguous_revisions": ambiguous, "invalid_revisions": _invalid_revisions(all_records), "pareto_eligible_count": len(pareto_records), "valid_variants": sorted(combinations), "invalid_harness_count": len(_invalid_revisions(all_records)), "confirmation_seed_count": len(confirmations), "confirmation_pass_count": sum(all(item["metrics"]["confirmation_gates"].values()) for item in confirmations), "claim": claim, "attribution": attribution, "rule": "only a unique latest revision is rankable; invalid revisions are retained, while invalid/preflight/legacy/preliminary/ambiguous/diagnose/probe/smoke records are excluded from Pareto ranking and causal attribution"}
    atomic_write_json(root / "capacity_map_summary.json", summary)
    report = "# Capacity-map report\n\nThis diagnostic report excludes invalid harness evidence from Pareto ranking. No formal promotion is authorized.\n\n| valid evidence | invalid harness | conclusion |\n| ---: | ---: | --- |\n| %d | %d | %s |\n\n## Attribution\n\n%s.\n" % (len(valid_records), len(records) - len(valid_records), claim, attribution)
    (root / "capacity_map_report.md").write_text(report, encoding="utf-8")
    print(root / "capacity_map_summary.json")
    return 0


def _summary_metrics(item: dict) -> dict:
    final = item.get("metrics", {}).get("final", {})
    return final.get("summary", final) if isinstance(final, dict) else {}


def _aggregate(values: list[float]) -> dict[str, float]:
    return {"median": float(statistics.median(values)), "variance": float(statistics.pvariance(values))}


def _closeout_records(records: list[dict], *, action: str, state: str | None = None) -> list[dict]:
    return [item for item in records if item.get("action") == action and (state is None or item.get("state") == state) and item.get("variant") == "E2:O0" and item.get("dataset") == "mnist"]


def closeout(args: argparse.Namespace) -> int:
    """Close the v2.2 O0 branch without constructing a model or CUDA object."""
    dual, primary = load_capacity_map_spec(args.config), load_capacity_map_spec(args.comparison_config)
    root = _root(dual, "closeout")
    if args.device != "cpu":
        raise ValueError("closeout is evidence-only and must use --device cpu")
    if (dual.objective_id, primary.objective_id) != ("dual_bundle", "primary_only") or dual.measurement_contract != MEASUREMENT_CONTRACT or primary.measurement_contract != MEASUREMENT_CONTRACT:
        raise ValueError("closeout requires the pre-registered v2.2 dual and primary configurations")
    dual_inputs, primary_inputs = input_hashes(dual, args.config), input_hashes(primary, args.comparison_config)
    shared_input_names = ("primary_evaluator", "smooth_evaluator", "evaluator_bundle", "evaluator_qualification_evidence")
    if any(dual_inputs[name] != primary_inputs[name] for name in shared_input_names):
        raise ValueError("closeout configurations do not bind the same qualified evaluator bundle")
    source = Path(args.evidence_root)
    dual_raw = [item for path in source.rglob("capacity_map_*.json") if (item := _read_capacity_evidence(path)) is not None and item.get("config_hash") == dual.config_hash]
    primary_raw = [item for path in source.rglob("capacity_map_*.json") if (item := _read_capacity_evidence(path)) is not None and item.get("config_hash") == primary.config_hash]
    dual_records, _, _, dual_ambiguous = _scan_evidence(source, dual)
    primary_records, _, _, primary_ambiguous = _scan_evidence(source, primary)
    seeds = list(dual.seeds)
    primary_smoke = {item.get("seed"): item for item in _closeout_records(primary_records, action="train-smoke")}
    dual_smoke = {item.get("seed"): item for item in _closeout_records(dual_records, action="train-smoke")}
    dual_confirm = {item.get("seed"): item for item in _closeout_records(dual_records, action="confirm")}
    missing = [name for name, collection in (("primary smoke", primary_smoke), ("dual smoke", dual_smoke), ("dual confirm", dual_confirm)) if set(collection) != set(seeds)]
    # The checkpoint replay and Shape qualification pre-date the dual-objective
    # configuration, so they are bound to the comparison configuration rather
    # than silently re-run under a new config hash.
    diagnostic_raw = [*dual_raw, *primary_raw]
    diagnoses = _closeout_records(diagnostic_raw, action="diagnose", state="screen_passed")
    shape_failures = [item for item in diagnostic_raw if item.get("action") == "qualify-shape" and item.get("state") == "screened_out"]
    physical = [item for item in dual_records if item.get("variant") and not str(item["variant"]).endswith(":O0") and item.get("action") in {"train-smoke", "confirm"}]
    if missing or len(diagnoses) != 1 or len(shape_failures) != 1 or physical:
        detail = "; ".join([*missing, "requires exactly one valid diagnose" if len(diagnoses) != 1 else "", "requires one Shape qualification failure" if len(shape_failures) != 1 else "", "physical downstream evidence is forbidden" if physical else ""])
        raise ValueError(f"closeout input contract failed: {detail}")
    metric_names = ("primary_accuracy", "smooth_accuracy", "evaluator_agreement", "minimum_per_class_recall", "worst_normalized_margin", "code_fraction", "saturation")
    differences: dict[str, list[float]] = {name: [] for name in metric_names}
    seed_rows = []
    for seed in seeds:
        p, d = _summary_metrics(primary_smoke[seed]), _summary_metrics(dual_smoke[seed])
        delta = {name: float(d[name]) - float(p[name]) for name in metric_names}
        for name, value in delta.items():
            differences[name].append(value)
        seed_rows.append({"seed": seed, "primary_smoke": p, "dual_smoke": d, "delta_dual_minus_primary": delta, "dual_confirm_gates": dual_confirm[seed].get("metrics", {}).get("confirmation_gates", {})})
    pass_count = sum(all(bool(value) for value in item.get("metrics", {}).get("confirmation_gates", {}).values()) for item in dual_confirm.values())
    inputs = sorted([*primary_smoke.values(), *dual_smoke.values(), *dual_confirm.values(), diagnoses[0], shape_failures[0]], key=lambda item: str(item["path"]))
    manifest = {"schema_version": 1, "status": "closed_conditional_negative" if pass_count == 0 else "closeout_gate_not_met", "dual_config_hash": dual.config_hash, "comparison_config_hash": primary.config_hash, "inputs": [{"path": item["path"], "content_hash": item["content_hash"], "action": item.get("action"), "state": item.get("state"), "seed": item.get("seed")} for item in inputs], "ambiguous_revisions": {"dual": dual_ambiguous, "primary": primary_ambiguous}, "invalid_revisions": {"dual": _invalid_revisions(dual_raw), "primary": _invalid_revisions(primary_raw)}}
    summary = {"schema_version": 1, "status": manifest["status"], "claim": "training_objective_conflict", "causal_scope": "E2:O0_declared_teacher_free_objectives", "causal_certainty": "not_uniquely_identified", "alternative_causes": ["electronic_generator_capacity", "optimization_dynamics", "evaluator_disagreement"], "downstream_status": "stopped_by_o0_gate", "confirmation_seed_count": len(dual_confirm), "confirmation_pass_count": pass_count, "seed_matched_comparison": {"rows": seed_rows, "dual_minus_primary": {name: _aggregate(values) for name, values in differences.items()}, "worst_dual_confirm_seed": min(seeds, key=lambda seed: float(_summary_metrics(dual_confirm[seed]).get("worst_normalized_margin", float("inf"))))}, "diagnostic_inputs": {"diagnose": diagnoses[0]["content_hash"], "shape_qualification_failure": shape_failures[0]["content_hash"]}, "input_hashes": {"dual_config": file_sha256(args.config), "comparison_config": file_sha256(args.comparison_config), **{name: dual_inputs[name] for name in shared_input_names}}, "evidence_governance": {"manifest_sha256": canonical_sha256(manifest), "ambiguous_revision_count": len(dual_ambiguous) + len(primary_ambiguous), "invalid_revision_count": len(_invalid_revisions(dual_raw)) + len(_invalid_revisions(primary_raw))}, "allowed_claim": "The current E2:O0 declared teacher-free objective family did not pass the qualified dual-evaluator gate.", "prohibited_claims": ["optical capacity failure", "phase-only capacity failure", "spatial-DOF failure", "universal semantic-diversity capacity conflict"], "unexecuted_branches": ["O1-O5 long training", "Teacher upper bound", "WGAN upper bound", "Controlled Shapes confirmation", "Fashion-MNIST confirmation"], "formal_promotion_authorized": False, "optimizer_created": False, "cuda_used": False}
    atomic_write_json(root / "capacity_map_closeout_manifest.json", manifest)
    atomic_write_json(root / "capacity_map_closeout_summary.json", summary)
    report = "# Capacity-map conditional-negative closeout\n\n**Status:** `%s`\n\n%s\n\n## Causal boundary\n\n- Scope: `%s`\n- Certainty: `%s`\n- Downstream: `%s`\n\nNo optical, phase-only, spatial-DOF, or universal-capacity conclusion is authorized.\n" % (summary["status"], summary["allowed_claim"], summary["causal_scope"], summary["causal_certainty"], summary["downstream_status"])
    (root / "capacity_map_closeout_report.md").write_text(report, encoding="utf-8")
    print(root / "capacity_map_closeout_summary.json")
    return 0 if summary["status"] == "closed_conditional_negative" else 1


def dispatch(args: argparse.Namespace) -> int:
    actions: dict[str, Callable[[argparse.Namespace], int]] = {"validate": validate, "screen": screen, "train-smoke": train_smoke, "confirm": confirm, "diagnose": diagnose, "qualify-shape": qualify_shape, "summarize": summarize, "closeout": closeout}
    return actions[args.research_action](args)
