"""v6.2 short-spacing self-imaging contracts.

The implementation deliberately separates electronic target admission from the
optical training path.  No legacy v6 state is implicitly migrated.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ...evidence import EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, tensor_sha256
from ...student.checkpoints import ConditionalCheckpointManifestV2, save_conditional_checkpoint, state_content_hash, state_signature
from ...student.conditional import camera_code_fraction, fixed_exposure_view, raw_from_generator_batched
from ...student.evaluator import evaluator_checkpoint_hash, load_frozen_evaluator, load_local_mnist
from ...student.generator import ConditionalPCASelfImagingTransportGenerator
from ...student.metrics import high_saturation_ratio
from ...optics.system import FormalAmplitudeFiniteCropIdentitySystem
from .support import ConditionalBundle, _context, _new_run, _random_codes, _split, _write_evidence, _system_signature, _fixed_control_probe, SHORT_SPACING_ARCHITECTURE, SHORT_SPACING_GENERATOR_CONTRACT
from .continuous_reachability import _build_modes

SHORT_SPACING_METRIC_CONTRACT = canonical_sha256({"kind": "conditional_self_imaging_metric_v2", "target": "kappa*A^2", "delta": "same_code_delta", "exposure": "global_fixed", "target_detached": True})
SHORT_SPACING_ROUTE_CONTRACT = canonical_sha256({"architecture": SHORT_SPACING_ARCHITECTURE, "layer_spacing_m": 0.25e-3, "gain": 0.05, "code": "tanh(z)"})
SHORT_SPACING_ADMISSION_CONTRACT = canonical_sha256({"kind": "conditional_short_spacing_target_admission_v2", "per_class_per_direction_bisection": {"lo": 0.05, "hi": 0.50, "steps": 20}, "uniform_safety_bisection": 20, "optimizer": False})
SHORT_SPACING_OPTICAL_CONTRACT = canonical_sha256({"kind": "conditional_short_spacing_optical_admission_v3", "compare_spacing_m": [1.5e-3, 0.25e-3], "kappa": "median_zero_code_energy_ratio", "saturation": "mean(raw/exposure>1)_before_clamp", "optimizer": False})
SHORT_SPACING_OBSERVABILITY_CONTRACT = canonical_sha256({"kind": "conditional_short_spacing_observability_v1", "losses": ["image", "delta"], "optimizer": False})
SHORT_SPACING_SMOKE_CONTRACT = canonical_sha256({"kind": "conditional_short_spacing_self_imaging_smoke_v2", "steps": 400, "phase_warmup": 100, "probes": [0, 50, 100, 200, 400]})


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


def _high_saturation_fraction(raw: torch.Tensor, exposure: float) -> float:
    """Report detector clipping before the fixed-exposure display clamp."""
    if raw.ndim != 4 or exposure <= 0.0:
        raise ValueError("raw must be [N,1,H,W] and exposure must be positive")
    return float(high_saturation_ratio(raw, exposure).item())


def _electronic_payload(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"architecture_kind", "carrier", "basis", "scale_logits", "scales"}
    if not isinstance(payload, dict) or not required.issubset(payload) or payload["architecture_kind"] != SHORT_SPACING_ARCHITECTURE:
        raise ValueError("invalid short-spacing electronic initialization")
    if tuple(payload["scales"].shape) != (10, 8) or not bool(((payload["scales"] >= 0.05) & (payload["scales"] <= 0.50)).all()):
        raise ValueError("electronic initialization violates v6.2.1 scale contract")
    return payload


def _encoder2_intensity_batched(system, generator, codes, labels):
    values = []
    with torch.no_grad():
        for index in range(codes.shape[0]):
            command = generator(codes[index : index + 1], labels[index : index + 1])
            trace = system.forward(system.command(command), return_optical_trace=True).optical_trace
            assert trace is not None
            values.append(trace[3].tensor.abs().square())
    return torch.cat(values, dim=0)


def _scale_logits(scales: torch.Tensor) -> torch.Tensor:
    fraction = ((scales - 0.05) / 0.45).clamp(1.0e-6, 1.0 - 1.0e-6)
    return torch.logit(fraction)


def _command_metrics(generator, codes, labels, evaluator, exposure):
    with torch.no_grad():
        amplitude = generator(codes, labels)
        view = fixed_exposure_view(amplitude.square(), exposure)
        features, logits = evaluator(view)
        predictions = logits.argmax(1)
    recalls = [float((predictions[labels == label] == label).float().mean()) for label in range(10)]
    if amplitude.shape[0] == 170:
        direction = amplitude.view(10, 17, *amplitude.shape[1:])
        rms = [float((direction[:, 1 + axis] - direction[:, 9 + axis]).square().mean().sqrt()) for axis in range(8)]
    else:
        rms = []
    return {
        "finite": bool(torch.isfinite(amplitude).all()),
        "accuracy": float((predictions == labels).float().mean()),
        "per_class_recall": recalls,
        "command_code_fraction": float(camera_code_fraction(amplitude.square(), labels)),
        "direction_rms": rms,
        "features": features.detach(),
    }


def _calibrate_scales(carrier, basis, panels, evaluator, exposure):
    """Calibrate [class,direction] axes then apply per-class uniform safety."""
    def semantic_ok(label, codes, values):
        logits = carrier[label : label + 1] + torch.einsum(
            "bk,khw->bhw", torch.tanh(codes) * values[None, :], basis[label]
        )
        command = torch.sigmoid(logits).unsqueeze(1)
        with torch.no_grad():
            _, output = evaluator(fixed_exposure_view(command.square(), exposure))
        return bool((output.argmax(1) == label).all())

    scales = torch.full((10, 8), 0.05)
    for label in range(10):
        for direction in range(8):
            lo, hi = 0.05, 0.50
            codes = torch.zeros(2, 8)
            codes[0, direction], codes[1, direction] = 1.0, -1.0
            for _ in range(20):
                candidate = (lo + hi) / 2.0
                proposal = scales.clone()
                proposal[label, direction] = candidate
                if semantic_ok(label, codes, proposal[label]):
                    lo = candidate
                else:
                    hi = candidate
            scales[label, direction] = lo
    # A common safety factor preserves independently selected axis directions
    # while preventing a uniform multi-axis code from crossing the semantic gate.
    for label in range(10):
        lo, hi = 0.0, 1.0
        for _ in range(20):
            beta = (lo + hi) / 2.0
            proposal = scales.clone()
            proposal[label] = 0.05 + beta * (proposal[label] - 0.05)
            safe = True
            for name in ("uniform_1234", "uniform_5678"):
                codes, labels_panel = panels[name]
                selected = labels_panel == label
                if not semantic_ok(label, codes[selected], proposal[label]):
                    safe = False
                    break
            if safe:
                lo = beta
            else:
                hi = beta
        scales[label] = 0.05 + lo * (scales[label] - 0.05)
    return scales


def conditional_short_spacing_target_admission(args, config) -> int:
    _require(config)
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    source, _ = _load_source(args.semantic_checkpoint)
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device("cpu"))
    basis = _build_modes(train_images, train_labels, evaluator, config)
    carrier = source["generator.class_carrier_logits"].clone()
    panels = _panels(torch.device("cpu"))
    transport_basis = ConditionalPCASelfImagingTransportGenerator(class_carrier_logits=carrier, class_modes=basis).lowpass_normalized_modes().detach()
    calibrated_scales = _calibrate_scales(carrier, transport_basis, panels, evaluator, config.conditional_generation.fixed_exposure_scale)
    generator = ConditionalPCASelfImagingTransportGenerator(class_carrier_logits=carrier, class_modes=basis, scale_logits=_scale_logits(calibrated_scales))
    metrics = {name: _command_metrics(generator, codes, panel_labels, evaluator, config.conditional_generation.fixed_exposure_scale) for name, (codes, panel_labels) in panels.items()}
    passed = all(
        value["finite"] and value["accuracy"] >= 0.95
        and min(value["per_class_recall"]) >= 0.90
        and value["command_code_fraction"] >= 0.25
        and min(value["direction_rms"]) >= 0.01
        for value in metrics.values()
    )
    run = _new_run(config)
    # Derive a fresh short-spacing state only after all electronic gates pass.
    # Only the semantic carrier and PCA
    # basis cross the architecture boundary; old 1.50 mm phases are never
    # copied.  The four phase tensors therefore remain exact zero buffers.
    initial_path = run.root / "conditional_short_spacing_electronic_initialization.pt"
    if passed:
        atomic_torch_save(initial_path, {"architecture_kind": SHORT_SPACING_ARCHITECTURE, "carrier": carrier, "basis": basis, "scale_logits": _scale_logits(calibrated_scales), "scales": calibrated_scales, "source_checkpoint_hash": file_sha256(args.semantic_checkpoint), "basis_hash": tensor_sha256(basis), "scale_hash": tensor_sha256(calibrated_scales)})
    report_path = run.root / "conditional_short_spacing_target_report.json"
    public_metrics = {name: {key: value for key, value in metric.items() if key != "features"} for name, metric in metrics.items()}
    atomic_write_json(report_path, {"contract": SHORT_SPACING_ADMISSION_CONTRACT, "metrics": public_metrics, "scales": calibrated_scales.tolist(), "scale_hash": tensor_sha256(calibrated_scales), "basis_hash": tensor_sha256(basis), "source_checkpoint_hash": file_sha256(args.semantic_checkpoint), "initialization_checkpoint": str(initial_path) if passed else None, "initialization_hash": file_sha256(initial_path) if passed else None})
    gates = (
        GateResult("semantic_accuracy", all(value["accuracy"] >= 0.95 and min(value["per_class_recall"]) >= 0.90 for value in metrics.values()), public_metrics),
        GateResult("electronic_code_fraction", all(value["command_code_fraction"] >= 0.25 for value in metrics.values()), public_metrics),
        GateResult("direction_response_rank", all(min(value["direction_rms"]) >= 0.01 for value in metrics.values()), public_metrics),
        GateResult("finite_command", all(value["finite"] for value in metrics.values()), public_metrics),
    )
    record = EvidenceRecordV2(2, "conditional_short_spacing_target_admission", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.semantic_checkpoint)), {"report": file_sha256(report_path), "source": file_sha256(args.semantic_checkpoint)}, gates, ("conditional-short-spacing-optical-admission",) if passed else (), None if passed else "short_spacing_electronic_target_gate_failed")
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
    electronic = report_payload.get("initialization_checkpoint")
    if not electronic or not Path(electronic).is_file():
        raise ValueError("target admission did not produce a short-spacing initialization checkpoint")
    if config.run.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("short-spacing optical admission requires CUDA")
    payload = _electronic_payload(electronic)
    device = torch.device("cuda")
    bundle = ConditionalBundle(config, carrier_logits=payload["carrier"], class_modes=payload["basis"], scale_logits=payload["scale_logits"]).to(device).eval()
    for parameter in (bundle.system.encoder.phase, bundle.system.decoder.phase):
        parameter.data.zero_()
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device("cuda"))
    panels = _panels(device)
    long_system_config = config.system.model_copy(update={"kind": "diagnostic_amplitude_finite_crop_identity", "layer_spacing_m": 1.50e-3})
    long_system = FormalAmplitudeFiniteCropIdentitySystem(long_system_config).to(device).eval()
    for parameter in (long_system.encoder.phase, long_system.decoder.phase):
        parameter.data.zero_()
    metrics = {}
    with torch.no_grad():
        for name, (codes, panel_labels) in panels.items():
            raw = raw_from_generator_batched(bundle.system, bundle.generator, codes, panel_labels, microbatch_size=1, activation_checkpoint=False)
            long_encoder2 = _encoder2_intensity_batched(long_system, bundle.generator, codes, panel_labels)
            short_encoder2 = _encoder2_intensity_batched(bundle.system, bundle.generator, codes, panel_labels)
            view = (raw / config.conditional_generation.fixed_exposure_scale).clamp(0, 1)
            _, logits = evaluator(F.adaptive_avg_pool2d(view, (28, 28)))
            pred = logits.argmax(1)
            long_fraction = float(camera_code_fraction(long_encoder2, panel_labels))
            short_fraction = float(camera_code_fraction(short_encoder2, panel_labels))
            metrics[name] = {"accuracy": float((pred == panel_labels).float().mean()), "zero_or_panel_finite": bool(torch.isfinite(raw).all()), "saturation": _high_saturation_fraction(raw, config.conditional_generation.fixed_exposure_scale), "encoder2_code_fraction_long": long_fraction, "encoder2_code_fraction_short": short_fraction, "encoder2_retention_improvement": short_fraction / max(long_fraction, 1.0e-12) - 1.0}
    zero_codes = torch.zeros(10, 8, device=device)
    zero_labels = torch.arange(10, device=device)
    with torch.no_grad():
        zero_command = bundle.generator(zero_codes, zero_labels)
        zero_raw = bundle.system.forward_from_generator(bundle.generator, zero_codes, zero_labels).raw_camera.tensor
        kappa = _target_kappa(zero_raw, zero_command)
    passed = all(v["zero_or_panel_finite"] and v["saturation"] <= 0.08 and v["accuracy"] >= 0.70 and v["encoder2_retention_improvement"] >= 0.25 for v in metrics.values()) and bool(torch.isfinite(kappa))
    run = _new_run(config)
    report = run.root / "conditional_short_spacing_optical_admission_report.json"
    initialization = run.root / "conditional_short_spacing_initialization.pt"
    manifest = ConditionalCheckpointManifestV2(2, SHORT_SPACING_ARCHITECTURE, _system_signature(config), canonical_sha256(config.model_dump(mode="json")), evidence.context.dataset_split_manifest_hash, evaluator_checkpoint_hash(args.evaluator_checkpoint), config.conditional_generation.fixed_exposure_scale, state_signature(bundle.state_dict()), state_content_hash(bundle.state_dict()), generator_contract_hash=SHORT_SPACING_GENERATOR_CONTRACT, metric_contract_hash=SHORT_SPACING_METRIC_CONTRACT, training_schedule_hash=canonical_sha256({"epochs": 5, "batch": 50, "phase_warmup": 100, "self_imaging": True}), initialization_seed=1234, initialization_state_hash=state_content_hash(bundle.state_dict()), config_contract_hash=canonical_sha256({"architecture": SHORT_SPACING_ARCHITECTURE, "layer_spacing_m": 0.25e-3}), pca_basis_hash=payload["basis_hash"], lowpass_contract_hash=canonical_sha256({"cycles_per_mm": 20.0}), scale_contract_hash=canonical_sha256({"parameterization": ".05+.45*sigmoid(g)", "scale_hash": tensor_sha256(payload["scales"])}), route_contract_hash=SHORT_SPACING_ROUTE_CONTRACT, semantic_source_checkpoint_hash=payload["source_checkpoint_hash"], safe_scale_evidence_hash=file_sha256(args.target_admission_evidence), first_400_schedule_hash=canonical_sha256({"phase_warmup": 100, "seed": 1234}))
    if passed:
        save_conditional_checkpoint(initialization, bundle, manifest, extra={"self_imaging": {"formula": "kappa*A^2", "kappa": float(kappa)}, "electronic_initialization_hash": file_sha256(electronic)})
    atomic_write_json(report, {"contract": SHORT_SPACING_OPTICAL_CONTRACT, "layer_spacing_m": config.system.layer_spacing_m, "trace": ["SLM", "Encoder1", "Encoder2", "identity OLS", "Decoder1", "Decoder2", "CMOS"], "optimizer": False, "metrics": metrics, "kappa": float(kappa), "initialization_checkpoint": str(initialization) if passed else None})
    record = EvidenceRecordV2(2, "conditional_short_spacing_optical_admission", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, evidence.context, {"target_admission": file_sha256(args.target_admission_evidence), "report": file_sha256(report), "electronic_initialization": file_sha256(electronic), "initialization": file_sha256(initialization) if passed else ""}, (GateResult("short_spacing_cuda_probe", passed, metrics | {"kappa": float(kappa)}),), ("conditional-short-spacing-observability",) if passed else (), None if passed else "short_spacing_optical_gate_failed")
    _write_evidence(run, "conditional_short_spacing_optical_admission_evidence.json", record)
    return 0 if passed else 1


def conditional_short_spacing_evidence_supersession(args, config) -> int:
    """Bind the pre-clamp saturation defect without mutating historical evidence."""
    _require(config)
    historical_path = Path(args.old_optical_evidence)
    historical = EvidenceRecordV2.from_dict(json.loads(historical_path.read_text(encoding="utf-8")))
    historical.validate()
    if historical.evidence_kind != "conditional_short_spacing_optical_admission":
        raise ValueError("supersession requires short-spacing optical-admission evidence")
    run = _new_run(config)
    report = run.root / "conditional_short_spacing_optical_supersession_report.json"
    atomic_write_json(report, {
        "superseded_evidence": str(historical_path.resolve()),
        "superseded_evidence_hash": file_sha256(historical_path),
        "classification": "invalid_saturation_harness",
        "reason": "the historical optical admission evaluated saturation after fixed-exposure clamp; v3 uses mean(raw/exposure>1) before clamp",
        "replacement_action": "rerun conditional-short-spacing-optical-admission with the v3 metric contract",
    })
    record = EvidenceRecordV2(
        2,
        "conditional_short_spacing_optical_supersession",
        config.run.evidence_tier,
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        historical.context,
        {"historical_optical_evidence": file_sha256(historical_path), "report": file_sha256(report)},
        (GateResult("historical_saturation_harness_invalid", True, {"historical_evidence_is_immutable": True}),),
        (),
        "historical short-spacing optical evidence cannot authorize downstream work because saturation was measured after clamp",
    )
    _write_evidence(run, "conditional_short_spacing_optical_supersession_evidence.json", record)
    return 0


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
    from .support import _load_bundle
    bundle, _ = _load_bundle(config, args.initialization_checkpoint, device=torch.device("cuda"))
    checkpoint_payload = torch.load(args.initialization_checkpoint, map_location="cpu", weights_only=False)
    kappa = float(checkpoint_payload.get("extra", {}).get("self_imaging", {}).get("kappa", float("nan")))
    if not torch.isfinite(torch.tensor(kappa)):
        raise ValueError("short-spacing initialization lacks frozen kappa")
    for parameter in bundle.parameters():
        parameter.requires_grad_(False)
    trainable = [bundle.generator.class_carrier_logits, bundle.generator.class_modes, bundle.generator.scale_logits, bundle.system.encoder.phase, bundle.system.decoder.phase]
    for parameter in trainable:
        parameter.requires_grad_(True)
    codes, labels = _fixed_control_probe(device=torch.device("cuda"))
    raw = bundle.system.forward_from_generator(bundle.generator, codes, labels).raw_camera.tensor
    command = bundle.generator(codes, labels)
    target = (kappa * command.square()).detach()
    image_loss = (raw / config.conditional_generation.fixed_exposure_scale - target / config.conditional_generation.fixed_exposure_scale).abs().mean()
    raw_panel = raw.view(10, 17, *raw.shape[1:])
    target_panel = target.view(10, 17, *target.shape[1:])
    delta_loss = ((raw_panel[:, 1:] - raw_panel[:, :1]) - (target_panel[:, 1:] - target_panel[:, :1])).abs().mean()
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device("cuda"))
    _, logits = evaluator(fixed_exposure_view(raw, config.conditional_generation.fixed_exposure_scale))
    class_loss = F.cross_entropy(logits, labels)
    grads = torch.autograd.grad(image_loss + delta_loss, trainable, allow_unused=True, retain_graph=True)
    class_grads = torch.autograd.grad(class_loss, trainable, allow_unused=True)
    cosine = [float(F.cosine_similarity(g.flatten(), c.flatten(), dim=0)) if g is not None and c is not None else float("nan") for g, c in zip(grads, class_grads)]
    valid = all(g is not None and bool(torch.isfinite(g).all()) and float(g.norm()) > 0 for g in grads) and all(value >= -0.10 for value in cosine if value == value)
    run = _new_run(config)
    report = run.root / "conditional_short_spacing_observability_report.json"
    atomic_write_json(report, {"contract": SHORT_SPACING_OBSERVABILITY_CONTRACT, "gradient_norms": [float(g.norm()) if g is not None else 0.0 for g in grads], "class_delta_cosine": cosine, "kappa": kappa, "optimizer": False})
    record = EvidenceRecordV2(2, "conditional_short_spacing_observability", config.run.evidence_tier, "passed" if valid else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, admission.context, {"optical_admission": file_sha256(args.optical_admission_evidence), "report": file_sha256(report)}, (GateResult("finite_nonzero_gradients", valid, {}),), ("short-spacing-self-imaging-smoke",) if valid else (), None if valid else "short-spacing gradient observability failed")
    _write_evidence(run, "conditional_short_spacing_observability_evidence.json", record)
    return 0 if valid else 1


def short_spacing_self_imaging_smoke(args, config) -> int:
    _require(config)
    raise RuntimeError("short-spacing smoke is reserved for the CUDA execution stage after target, optical and observability gates")
