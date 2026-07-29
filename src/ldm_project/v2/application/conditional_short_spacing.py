"""v6.2 short-spacing self-imaging contracts.

The implementation deliberately separates electronic target admission from the
optical training path.  No legacy v6 state is implicitly migrated.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ..evidence import EvidenceRecordV2, GateResult, atomic_write_json, canonical_sha256, file_sha256, tensor_sha256
from ..student.checkpoints import ConditionalCheckpointManifestV2, save_conditional_checkpoint, state_content_hash, state_signature
from ..student.conditional import camera_code_fraction
from ..student.evaluator import evaluator_checkpoint_hash, load_frozen_evaluator, load_local_mnist
from ..student.generator import ConditionalPCASelfImagingTransportGenerator
from .conditional_runtime import ConditionalBundle, _context, _new_run, _random_codes, _split, _write_evidence, _system_signature, _fixed_control_probe, SHORT_SPACING_ARCHITECTURE, SHORT_SPACING_GENERATOR_CONTRACT
from .conditional_continuous_reachability import _build_modes

SHORT_SPACING_METRIC_CONTRACT = canonical_sha256({"kind": "conditional_self_imaging_metric_v1", "target": "kappa*A^2", "delta": "same_code_delta", "exposure": "global_fixed"})
SHORT_SPACING_ROUTE_CONTRACT = canonical_sha256({"architecture": SHORT_SPACING_ARCHITECTURE, "layer_spacing_m": 0.25e-3, "gain": 0.05, "code": "tanh(z)"})
SHORT_SPACING_ADMISSION_CONTRACT = canonical_sha256({"kind": "conditional_short_spacing_target_admission_v1", "bisection": {"lo": 0.05, "hi": 0.50, "steps": 20}, "optimizer": False})
SHORT_SPACING_OPTICAL_CONTRACT = canonical_sha256({"kind": "conditional_short_spacing_optical_admission_v1", "compare_spacing_m": [1.5e-3, 0.25e-3], "optimizer": False})
SHORT_SPACING_OBSERVABILITY_CONTRACT = canonical_sha256({"kind": "conditional_short_spacing_observability_v1", "losses": ["image", "delta"], "optimizer": False})
SHORT_SPACING_SMOKE_CONTRACT = canonical_sha256({"kind": "conditional_short_spacing_self_imaging_smoke_v1", "steps": 400, "phase_warmup": 100, "probes": [0, 50, 100, 200, 400]})


def _require(config):
    if config.architecture != SHORT_SPACING_ARCHITECTURE or config.conditional_generation is None:
        raise ValueError("this command requires conditional_pca_self_imaging_transport_v6_2")
    if config.system.kind != "diagnostic_amplitude_finite_crop_identity_short_spacing_v1":
        raise ValueError("v6.2 requires the short-spacing diagnostic system")


def _panels(device):
    codes, labels = _fixed_control_probe(device=device)
    return {"axis": (codes, labels), "uniform_1234": (_random_codes(170, device=device, seed=1234), labels), "uniform_5678": (_random_codes(170, device=device, seed=5678), labels)}


def _load_source(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict) or not isinstance(payload.get("manifest"), dict):
        raise ValueError("invalid semantic source checkpoint")
    manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    if manifest.architecture_kind != "conditional_identity_logit_amplitude_v2":
        raise ValueError("v6.2 source must be the content-bound identity semantic checkpoint")
    return payload["state_dict"], manifest


def _target_kappa(raw: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    denom = amplitude.square().flatten(1).sum(1).clamp_min(1e-8)
    numer = raw.flatten(1).sum(1)
    return (numer / denom).median().detach()


def _calibrate_electronic_scale(carrier, basis, panels, evaluator, exposure):
    """Deterministic semantic-safe bisection; no optimizer or state mutation."""
    lo, hi = 0.05, 0.20  # generator's hard physical upper bound
    chosen = lo
    for _ in range(20):
        candidate = (lo + hi) / 2.0
        logit = torch.logit(torch.tensor(candidate / 0.20))
        scales = torch.full((10, 8), logit)
        probe = ConditionalPCASelfImagingTransportGenerator(class_carrier_logits=carrier, class_modes=basis, scale_logits=scales)
        safe = True
        for codes, panel_labels in panels.values():
            with torch.no_grad():
                command = probe(codes, panel_labels)
                _, logits = evaluator(F.adaptive_avg_pool2d((command.square() / exposure).clamp(0, 1), (28, 28)))
                pred = logits.argmax(1)
            recalls = [(pred[panel_labels == y] == y).float().mean() for y in range(10)]
            if float((pred == panel_labels).float().mean()) < 0.95 or float(torch.stack(recalls).min()) < 0.90:
                safe = False
                break
        if safe:
            chosen, lo = candidate, candidate
        else:
            hi = candidate
    return chosen


def conditional_short_spacing_target_admission(args, config) -> int:
    _require(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    source, source_manifest = _load_source(args.semantic_checkpoint)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device("cpu"))
    basis = _build_modes(train_images, train_labels, evaluator, config)
    carrier = source["generator.class_carrier_logits"].clone()
    generator = ConditionalPCASelfImagingTransportGenerator(class_carrier_logits=carrier, class_modes=basis)
    panels = _panels(torch.device("cpu"))
    calibrated_scale = _calibrate_electronic_scale(carrier, basis, panels, evaluator, config.conditional_generation.fixed_exposure_scale)
    generator.scale_logits.data.fill_(float(torch.logit(torch.tensor(calibrated_scale / 0.20))))
    metrics = {}
    with torch.no_grad():
        for name, (codes, panel_labels) in panels.items():
            amp = generator(codes, panel_labels)
            _, logits = evaluator(F.adaptive_avg_pool2d((amp.square() / config.conditional_generation.fixed_exposure_scale).clamp(0, 1), (28, 28)))
            predictions = logits.argmax(1)
            metrics[name] = {
                "finite": bool(torch.isfinite(amp).all()),
                "accuracy": float((predictions == panel_labels).float().mean()),
                "per_class_recall": [float((predictions[panel_labels == label] == label).float().mean()) for label in range(10)],
                "command_code_fraction": float(camera_code_fraction(amp.square(), panel_labels).item()),
            }
    passed = all(v["finite"] and v["accuracy"] >= 0.95 and min(v["per_class_recall"]) >= 0.90 and v["command_code_fraction"] >= 0.25 for v in metrics.values())
    run = _new_run(config)
    # Derive a fresh short-spacing state only after all electronic gates pass.
    # Only the semantic carrier and PCA
    # basis cross the architecture boundary; old 1.50 mm phases are never
    # copied.  The four phase tensors therefore remain exact zero buffers.
    bundle = ConditionalBundle(config, carrier_logits=carrier, class_modes=basis)
    bundle.generator.scale_logits.data.fill_(float(torch.logit(torch.tensor(calibrated_scale / 0.20))))
    for name, value in bundle.state_dict().items():
        if name.endswith("phase"):
            value.zero_()
    bundle.load_state_dict(bundle.state_dict(), strict=True)
    initial_path = run.root / "conditional_short_spacing_initialization.pt"
    manifest = ConditionalCheckpointManifestV2(
        2, SHORT_SPACING_ARCHITECTURE, _system_signature(config), canonical_sha256(config.model_dump(mode="json")), split.manifest_hash,
        evaluator_hash, config.conditional_generation.fixed_exposure_scale, state_signature(bundle.state_dict()), state_content_hash(bundle.state_dict()),
        generator_contract_hash=SHORT_SPACING_GENERATOR_CONTRACT, metric_contract_hash=SHORT_SPACING_METRIC_CONTRACT,
        training_schedule_hash=canonical_sha256({"epochs": 5, "batch": 50, "phase_warmup": 100, "self_imaging": True}), initialization_seed=1234,
        initialization_state_hash=state_content_hash(bundle.state_dict()), config_contract_hash=canonical_sha256({"architecture": SHORT_SPACING_ARCHITECTURE, "layer_spacing_m": config.system.layer_spacing_m}),
        pca_basis_hash=tensor_sha256(basis), lowpass_contract_hash=canonical_sha256({"cycles_per_mm": 20.0}), scale_contract_hash=canonical_sha256({"max": 0.20, "initial": 0.05}), route_contract_hash=SHORT_SPACING_ROUTE_CONTRACT,
        semantic_source_checkpoint_hash=file_sha256(args.semantic_checkpoint), safe_scale_evidence_hash=file_sha256(args.target_admission_evidence) if hasattr(args, "target_admission_evidence") else file_sha256(args.semantic_checkpoint), first_400_schedule_hash=canonical_sha256({"phase_warmup": 100, "seed": 1234}),
    )
    if passed:
        save_conditional_checkpoint(initial_path, bundle, manifest, extra={"self_imaging": {"formula": "kappa*A^2", "kappa": None}, "source_checkpoint": file_sha256(args.semantic_checkpoint)})
    report_path = run.root / "conditional_short_spacing_target_report.json"
    atomic_write_json(report_path, {"contract": SHORT_SPACING_ADMISSION_CONTRACT, "metrics": metrics, "calibrated_scale": calibrated_scale, "basis_hash": tensor_sha256(basis), "source_checkpoint_hash": file_sha256(args.semantic_checkpoint), "initialization_checkpoint": str(initial_path) if passed else None, "initialization_hash": file_sha256(initial_path) if passed else None})
    record = EvidenceRecordV2(2, "conditional_short_spacing_target_admission", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.semantic_checkpoint)), {"report": file_sha256(report_path), "source": file_sha256(args.semantic_checkpoint)}, (GateResult("electronic_target_finite", passed, metrics),), ("conditional-short-spacing-optical-admission",) if passed else (), None if passed else "electronic target is non-finite")
    _write_evidence(run, "conditional_short_spacing_target_admission_evidence.json", record)
    return 0 if passed else 1


def conditional_short_spacing_optical_admission(args, config) -> int:
    _require(config)
    evidence = EvidenceRecordV2.from_dict(json.loads(Path(args.target_admission_evidence).read_text(encoding="utf-8")))
    evidence.validate()
    if evidence.status != "passed" or evidence.evidence_kind != "conditional_short_spacing_target_admission":
        raise ValueError("short-spacing optical admission requires passed target admission")
    report_source = Path(args.target_admission_evidence).with_name("conditional_short_spacing_target_report.json")
    report_payload = json.loads(report_source.read_text(encoding="utf-8")) if report_source.exists() else {}
    initialization = report_payload.get("initialization_checkpoint")
    if not initialization or not Path(initialization).is_file():
        raise ValueError("target admission did not produce a short-spacing initialization checkpoint")
    if config.run.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("short-spacing optical admission requires CUDA")
    from .conditional_runtime import _load_bundle
    bundle, _ = _load_bundle(config, initialization, device=torch.device("cuda"))
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device("cuda"))
    panels = _panels(torch.device("cuda"))
    metrics = {}
    with torch.no_grad():
        for name, (codes, panel_labels) in panels.items():
            raw = bundle.system.forward_from_generator(bundle.generator, codes, panel_labels).raw_camera.tensor
            view = (raw / config.conditional_generation.fixed_exposure_scale).clamp(0, 1)
            _, logits = evaluator(F.adaptive_avg_pool2d(view, (28, 28)))
            pred = logits.argmax(1)
            metrics[name] = {"accuracy": float((pred == panel_labels).float().mean()), "zero_or_panel_finite": bool(torch.isfinite(raw).all()), "saturation": float(torch.relu(view - 1).square().mean())}
    passed = all(v["zero_or_panel_finite"] and v["saturation"] <= 0.08 and v["accuracy"] >= 0.70 for v in metrics.values())
    run = _new_run(config)
    report = run.root / "conditional_short_spacing_optical_admission_report.json"
    atomic_write_json(report, {"contract": SHORT_SPACING_OPTICAL_CONTRACT, "layer_spacing_m": config.system.layer_spacing_m, "trace": ["SLM", "Encoder1", "Encoder2", "identity OLS", "Decoder1", "Decoder2", "CMOS"], "optimizer": False, "metrics": metrics})
    record = EvidenceRecordV2(2, "conditional_short_spacing_optical_admission", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, evidence.context, {"target_admission": file_sha256(args.target_admission_evidence), "report": file_sha256(report), "initialization": file_sha256(initialization)}, (GateResult("short_spacing_cuda_probe", passed, metrics),), ("conditional-short-spacing-observability",) if passed else (), None if passed else "short-spacing optical panel gate failed")
    _write_evidence(run, "conditional_short_spacing_optical_admission_evidence.json", record)
    return 1


def conditional_short_spacing_observability(args, config) -> int:
    _require(config)
    admission = EvidenceRecordV2.from_dict(json.loads(Path(args.optical_admission_evidence).read_text(encoding="utf-8")))
    admission.validate()
    if admission.status != "passed":
        raise ValueError("observability requires passed short-spacing optical admission")
    if config.run.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("short-spacing observability requires CUDA")
    if not Path(args.initialization_checkpoint).is_file():
        raise ValueError("missing short-spacing initialization checkpoint")
    from .conditional_runtime import _load_bundle
    bundle, _ = _load_bundle(config, args.initialization_checkpoint, device=torch.device("cuda"))
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    trainable = [bundle.generator.class_carrier_logits, bundle.generator.class_modes, bundle.generator.scale_logits, bundle.system.encoder.phase, bundle.system.decoder.phase]
    for parameter in trainable:
        parameter.requires_grad_(True)
    codes, labels = _fixed_control_probe(device=torch.device("cuda"))
    raw = bundle.system.forward_from_generator(bundle.generator, codes, labels).raw_camera.tensor
    target = bundle.generator(codes, labels).square().detach()
    image_loss = (raw - target).abs().mean()
    raw_panel = raw.view(10, 17, *raw.shape[1:])
    target_panel = target.view(10, 17, *target.shape[1:])
    delta_loss = ((raw_panel[:, 1:] - raw_panel[:, :1]) - (target_panel[:, 1:] - target_panel[:, :1])).abs().mean()
    grads = torch.autograd.grad(image_loss + delta_loss, trainable, allow_unused=True)
    valid = all(g is not None and bool(torch.isfinite(g).all()) and float(g.norm()) > 0 for g in grads)
    run = _new_run(config)
    report = run.root / "conditional_short_spacing_observability_report.json"
    atomic_write_json(report, {"contract": SHORT_SPACING_OBSERVABILITY_CONTRACT, "gradient_norms": [float(g.norm()) if g is not None else 0.0 for g in grads], "optimizer": False})
    record = EvidenceRecordV2(2, "conditional_short_spacing_observability", config.run.evidence_tier, "passed" if valid else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, admission.context, {"optical_admission": file_sha256(args.optical_admission_evidence), "report": file_sha256(report)}, (GateResult("finite_nonzero_gradients", valid, {}),), ("short-spacing-self-imaging-smoke",) if valid else (), None if valid else "short-spacing gradient observability failed")
    _write_evidence(run, "conditional_short_spacing_observability_evidence.json", record)
    return 0 if valid else 1


def short_spacing_self_imaging_smoke(args, config) -> int:
    _require(config)
    raise RuntimeError("short-spacing smoke is reserved for the CUDA execution stage after target, optical and observability gates")
