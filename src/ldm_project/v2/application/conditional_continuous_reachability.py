"""Contract-corrected read-only reachability probe for continuous PCA modes."""
from __future__ import annotations

import json
from pathlib import Path
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from ..evidence import EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, tensor_sha256
from ..student.checkpoints import ConditionalCheckpointManifestV2, load_conditional_checkpoint, state_content_hash
from ..student.conditional import camera_code_fraction, fixed_exposure_view, raw_from_generator_batched, relative_diversity_ratios
from ..student.evaluator import MNISTSplitManifest, evaluator_checkpoint_hash, load_frozen_evaluator, load_local_mnist
from ..student.review_artifacts import _atomic_png
from .conditional_continuous_audit import _lowpass_modes, _pca, _real_probe_views
from .conditional_runtime import IDENTITY_ARCHITECTURE, ConditionalBundle, _context, _fixed_control_probe, _new_run, _random_codes, _split, _write_evidence


SAFE_AXIS_LOGIT_RMS = 0.05
SAFE_PROBE_ALPHA = SAFE_AXIS_LOGIT_RMS / float(torch.tanh(torch.tensor(1.0)).item())
EXPECTED_CONTINUOUS_BASIS_HASH = "de283cad951b1fddb54499c16c032fc44f6beb5239734d78795475effbe890bb"
CONTINUOUS_REACHABILITY_CONTRACT = canonical_sha256({
    "kind": "conditional_continuous_mode_reachability_v2",
    "basis_contract": "conditional_continuous_mode_capacity_v1",
    "safe_axis_logit_rms": SAFE_AXIS_LOGIT_RMS,
    "alpha": SAFE_PROBE_ALPHA,
    "panels": ["axis", "uniform_1234", "uniform_5678"],
    "margin_threshold": 0.5,
    "probe_raw_rms": 0.01,
    "logit_rms_limit": 0.05,
    "phase_rms_limit": 0.005,
    "optimizer": False,
    "checkpoint": False,
})


class _ContinuousProbeGenerator(nn.Module):
    """Diagnostic generator with an explicit, fixed safe logit gain."""

    def __init__(self, carrier: torch.Tensor, modes: torch.Tensor, *, gain: float = SAFE_PROBE_ALPHA) -> None:
        super().__init__()
        if carrier.shape != (10, 200, 200) or modes.shape != (10, 8, 200, 200):
            raise ValueError("continuous probe expects [10,200,200] carrier and [10,8,200,200] modes")
        self.class_carrier_logits = nn.Parameter(carrier.detach().clone())
        self.class_modes = nn.Parameter(modes.detach().clone())
        self.gain = float(gain)

    @staticmethod
    def normalized_modes(modes: torch.Tensor) -> torch.Tensor:
        centered = modes - modes.mean(dim=(-2, -1), keepdim=True)
        return centered / centered.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-12).sqrt()

    def logits(self, codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        coefficients = torch.tanh(codes)
        modes = self.normalized_modes(self.class_modes).index_select(0, labels)
        residual = torch.einsum("bk,bkhw->bhw", coefficients, modes)
        return self.class_carrier_logits.index_select(0, labels) + self.gain * residual

    def forward(self, codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(codes, labels)).unsqueeze(1)


def _build_modes(train_images: torch.Tensor, train_labels: torch.Tensor, evaluator: nn.Module, config) -> torch.Tensor:
    amplitude_modes = []
    amplitude = train_images.sqrt().clamp(1.0e-4, 1.0)
    for label in range(10):
        selected = train_labels == label
        _, _, vectors = _pca(amplitude[selected].flatten(1).to(torch.float64), components=8)
        amplitude_modes.append(vectors.T.reshape(8, 28, 28))
    modes = torch.stack([_lowpass_modes(item, shape=config.system.slm_shape, pixel_pitch_m=config.system.slm_pixel_pitch_m, max_cycles_per_mm=20.0) for item in amplitude_modes])
    return modes.to(torch.float32)


def _full_contact_sheet(path: Path, raw: torch.Tensor, labels: torch.Tensor, exposure: float) -> str:
    import matplotlib.pyplot as plt

    raw, labels = raw.detach().cpu(), labels.detach().cpu().long()
    if raw.shape[0] != 170 or labels.shape != (170,):
        raise ValueError("continuous reachability sheets require exactly 170 samples")
    figure, axes = plt.subplots(10, 17, figsize=(25, 15))
    for row in range(10):
        for col in range(17):
            axis = axes[row, col]
            axis.imshow((raw[row * 17 + col, 0] / exposure).clamp(0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            if row == 0:
                axis.set_title("z0" if col == 0 else (f"+e{col}" if col <= 8 else f"-e{col - 8}"), fontsize=6)
            if col == 0:
                axis.set_ylabel(str(int(labels[row * 17,].item())), fontsize=7)
    figure.tight_layout(pad=0.1)
    _atomic_png(path, figure)
    plt.close(figure)
    return file_sha256(path)


def _panel_metrics(raw: torch.Tensor, command: torch.Tensor, labels: torch.Tensor, evaluator: nn.Module, real_views: torch.Tensor, exposure: float) -> dict[str, object]:
    view = fixed_exposure_view(raw, exposure)
    features, logits = evaluator(view)
    predictions = logits.argmax(dim=1)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    margins = logits.gather(1, labels[:, None]).squeeze(1) - logits.masked_fill(F.one_hot(labels, 10).bool(), float("-inf")).max(dim=1).values
    recall = [float((predictions[labels == label] == label).float().mean().item()) for label in range(10)]
    return {
        "accuracy": float((predictions == labels).float().mean().item()),
        "recall": recall,
        "confusion": torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10).cpu().tolist(),
        "margins": margins.detach().cpu().tolist(),
        "camera_code_fraction": float(camera_code_fraction(raw, labels).item()),
        "physical_slm_code_fraction": float(camera_code_fraction(command.square(), labels).item()),
        "feature_distance_ratios": relative_diversity_ratios(features, real_features, labels),
        "saturation": float(torch.relu(raw / exposure - 1.0).square().mean().item()),
        "finite": bool(torch.isfinite(raw).all() and torch.isfinite(command).all()),
    }


def _margin_values(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the evaluator margin without changing evaluator parameters."""
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.masked_fill(F.one_hot(labels, 10).bool(), float("-inf"))
    return correct - competitors.max(dim=1).values


@contextmanager
def _temporary_delta(parameter: torch.Tensor, delta: torch.Tensor):
    """Apply one diagnostic delta and restore the exact original tensor."""
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(delta)
    try:
        yield
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def _safe_group_probe(
    *,
    group_name: str,
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    system: nn.Module,
    generator: _ContinuousProbeGenerator,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    baseline_raw: torch.Tensor,
    real_views: torch.Tensor,
    exposure: float,
    microbatch_size: int,
) -> dict[str, object]:
    """Run one reversible, one-shot tangent probe for a parameter group.

    This deliberately uses a single normalized direction and a reversible
    parameter view.  There is no optimizer, no state write-back, and no
    checkpoint.  The two safety limits are enforced on the actual generated
    command/phase perturbation, followed by the 1% raw-CMOS target.
    """
    if gradient is None or not torch.isfinite(gradient).all() or gradient.norm() <= 1.0e-12:
        return {"valid": False, "reason": "missing_or_zero_gradient", "group": group_name}
    direction = gradient.detach() / gradient.detach().norm().clamp_min(1.0e-12)
    base_view = fixed_exposure_view(baseline_raw, exposure)
    with torch.no_grad():
        _, base_logits = evaluator(base_view)
        base_margin = _margin_values(base_logits, labels)
        base_metrics = _panel_metrics(baseline_raw, generator(codes, labels), labels, evaluator, real_views, exposure)
    base_rms = baseline_raw.square().mean().sqrt().clamp_min(1.0e-12)

    # First evaluate an unscaled direction to obtain the physical response.
    unit_delta = direction
    with _temporary_delta(parameter, unit_delta):
        with torch.no_grad():
            unit_command = generator(codes, labels)
            unit_raw = raw_from_generator_batched(system, generator, codes, labels, microbatch_size=microbatch_size, activation_checkpoint=False)
    if group_name in {"class_carrier", "continuous_modes"}:
        with torch.no_grad():
            base_command = generator(codes, labels)
            logit_rms = float((torch.logit(unit_command.clamp(1.0e-6, 1.0 - 1.0e-6)) - torch.logit(base_command.clamp(1.0e-6, 1.0 - 1.0e-6))).square().mean().sqrt().item())
        scale = min(1.0, 0.05 / max(logit_rms, 1.0e-12))
        protection_value = logit_rms * scale
        protection_limit = 0.05
        protection_name = "slm_logit_rms"
    else:
        phase_rms = float(unit_delta.square().mean().sqrt().item())
        scale = min(1.0, 0.005 / max(phase_rms, 1.0e-12))
        protection_value = phase_rms * scale
        protection_limit = 0.005
        protection_name = "phase_rms"
    physical_rms = (unit_raw - baseline_raw).square().mean().sqrt().item() * scale
    raw_scale = min(1.0, 0.01 * float(base_rms.item()) / max(physical_rms, 1.0e-12))
    scale *= raw_scale
    zero_codes = torch.zeros(10, 8, device=codes.device, dtype=codes.dtype)
    zero_labels = torch.arange(10, device=codes.device)
    with torch.no_grad():
        zero_before = raw_from_generator_batched(system, generator, zero_codes, zero_labels, microbatch_size=microbatch_size, activation_checkpoint=False)
    with _temporary_delta(parameter, direction * scale):
        with torch.no_grad():
            perturbed_command = generator(codes, labels)
            perturbed_raw = raw_from_generator_batched(system, generator, codes, labels, microbatch_size=microbatch_size, activation_checkpoint=False)
            _, perturbed_logits = evaluator(fixed_exposure_view(perturbed_raw, exposure))
            perturbed_margin = _margin_values(perturbed_logits, labels)
            zero_after = raw_from_generator_batched(system, generator, zero_codes, zero_labels, microbatch_size=microbatch_size, activation_checkpoint=False)
    delta_margin = perturbed_margin - base_margin
    low = base_margin < 0.5
    with torch.no_grad():
        perturbed_metrics = _panel_metrics(perturbed_raw, perturbed_command, labels, evaluator, real_views, exposure)
    return {
        "valid": True,
        "group": group_name,
        "step_scale": float(scale),
        "raw_rms_fraction": float(((perturbed_raw - baseline_raw).square().mean().sqrt() / base_rms).item()),
        protection_name: float(protection_value),
        "protection_limit": protection_limit,
        "low_margin_count": int(low.sum().item()),
        "low_margin_positive_fraction": float((delta_margin[low] > 0).float().mean().item()) if low.any() else 1.0,
        "low_margin_median_delta": float(delta_margin[low].median().item()) if low.any() else 0.0,
        "correct_count_delta": int(((perturbed_logits.argmax(1) == labels).sum() - (base_logits.argmax(1) == labels).sum()).item()),
        "direction_derivative": (delta_margin / max(scale, 1.0e-12)).detach().cpu().tolist(),
        "camera_fraction_delta": float(perturbed_metrics["camera_code_fraction"]) - float(base_metrics["camera_code_fraction"]),
        "feature_ratio_delta": [float(a) - float(b) for a, b in zip(perturbed_metrics["feature_distance_ratios"], base_metrics["feature_distance_ratios"])],
        "saturation": float(perturbed_metrics["saturation"]),
        "zero_code_max_abs": float((zero_after - zero_before).abs().max().item()),
        "state_restored": True,
    }


def _streaming_panel_gradients(
    *,
    system: nn.Module,
    generator: _ContinuousProbeGenerator,
    evaluator: nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    real_views: torch.Tensor,
    groups: dict[str, tuple[torch.Tensor, ...]],
    exposure: float,
    microbatch_size: int,
) -> tuple[dict[str, dict[str, object]], dict[str, torch.Tensor]]:
    """Accumulate panel gradients one class at a time.

    A 170-sample physical backward graph does not fit on the project's 6 GB
    GPU.  The panel remains exactly the fixed 10x17 panel, but each class's
    graph is consumed immediately after its two VJPs.  This is equivalent to
    averaging the class-local objectives and avoids retaining 170 optical
    propagation graphs at once.
    """
    flat_parameters = tuple(parameter for values in groups.values() for parameter in values)
    margin_accumulator = [torch.zeros_like(parameter) for parameter in flat_parameters]
    distribution_accumulator = [torch.zeros_like(parameter) for parameter in flat_parameters]
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()
        if selected.numel() != 17:
            raise ValueError("continuous reachability requires exactly 17 samples per class")
        raw = raw_from_generator_batched(
            system,
            generator,
            codes.index_select(0, selected),
            labels.index_select(0, selected),
            microbatch_size=microbatch_size,
            activation_checkpoint=False,
        )
        generated_features, logits = evaluator(fixed_exposure_view(raw, exposure))
        margins = _margin_values(logits, labels.index_select(0, selected))
        low_margin = margins < 0.5
        margin_objective = margins[low_margin].mean() if low_margin.any() else margins.mean()
        real_selected = real_features.index_select(0, selected)
        feature_variation = torch.pdist(F.normalize(generated_features, dim=1), p=2).median()
        real_variation = torch.pdist(F.normalize(real_selected, dim=1), p=2).median().detach().clamp_min(1.0e-8)
        raw_variation = (raw - raw.mean(dim=0, keepdim=True)).square().mean() / raw.square().mean().detach().clamp_min(1.0e-12)
        saturation = torch.relu(raw / exposure - 1.0).square().mean()
        distribution_objective = feature_variation / real_variation + raw_variation - 0.1 * saturation
        margin_grads = torch.autograd.grad(margin_objective, flat_parameters, allow_unused=True, retain_graph=True)
        distribution_grads = torch.autograd.grad(distribution_objective, flat_parameters, allow_unused=True, retain_graph=False)
        for index, (margin_gradient, distribution_gradient) in enumerate(zip(margin_grads, distribution_grads)):
            if margin_gradient is not None:
                margin_accumulator[index].add_(margin_gradient.detach())
            if distribution_gradient is not None:
                distribution_accumulator[index].add_(distribution_gradient.detach())
        del raw, generated_features, logits, margins, margin_grads, distribution_grads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    offset = 0
    metrics: dict[str, dict[str, object]] = {}
    combined: dict[str, torch.Tensor] = {}
    for group_name, parameters in groups.items():
        count = len(parameters)
        margin_values = margin_accumulator[offset : offset + count]
        distribution_values = distribution_accumulator[offset : offset + count]
        offset += count
        margin_norm = torch.stack([value.norm() for value in margin_values]).sum()
        distribution_norm = torch.stack([value.norm() for value in distribution_values]).sum()
        valid = bool(
            torch.isfinite(margin_norm)
            and torch.isfinite(distribution_norm)
            and margin_norm > 1.0e-12
            and distribution_norm > 1.0e-12
        )
        metrics[group_name] = {
            "finite_nonzero": valid,
            "margin_norm": float(margin_norm.item()),
            "distribution_norm": float(distribution_norm.item()),
            "streamed_class_count": 10,
        }
        if valid:
            margin_vector = torch.stack(margin_values).sum(dim=0)
            distribution_vector = torch.stack(distribution_values).sum(dim=0)
            combined[group_name] = margin_vector / margin_vector.norm().clamp_min(1.0e-12) + 0.25 * distribution_vector / distribution_vector.norm().clamp_min(1.0e-12)
    return metrics, combined


def conditional_continuous_mode_reachability(args, config) -> int:
    if config.architecture != IDENTITY_ARCHITECTURE or config.conditional_generation is None:
        raise ValueError("continuous reachability requires identity v2 configuration")
    images, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels_all, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    checkpoint_hash = file_sha256(args.semantic_checkpoint)
    expected = _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=checkpoint_hash)
    old = EvidenceRecordV2.from_dict(json.loads(Path(args.capacity_evidence).read_text(encoding="utf-8")))
    old.validate()
    if old.evidence_kind != "conditional_continuous_mode_capacity" or old.status != "passed" or "conditional-continuous-mode-admission-design" not in old.permitted_next_actions or old.context.system_signature != expected.system_signature or old.context.evaluator_checkpoint_hash != expected.evaluator_checkpoint_hash or old.context.dataset_split_manifest_hash != expected.dataset_split_manifest_hash or old.context.checkpoint_hash != expected.checkpoint_hash:
        raise ValueError("capacity evidence is not bound to the current continuous reachability context")
    evaluator, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=torch.device("cpu"))
    if MNISTSplitManifest(**payload["split_manifest"]).manifest_hash != split.manifest_hash:
        raise ValueError("evaluator split mismatch")
    source_payload = torch.load(args.semantic_checkpoint, map_location="cpu", weights_only=False)
    source_manifest = ConditionalCheckpointManifestV2(**source_payload["manifest"])
    if source_manifest.architecture_kind != IDENTITY_ARCHITECTURE or source_manifest.system_signature != expected.system_signature or source_manifest.evaluator_checkpoint_hash != evaluator_hash or source_manifest.dataset_split_manifest_hash != split.manifest_hash:
        raise ValueError("semantic checkpoint is not the bound identity checkpoint")
    bundle = ConditionalBundle(config).cpu()
    load_conditional_checkpoint(args.semantic_checkpoint, bundle, source_manifest)
    before = state_content_hash(bundle.state_dict())
    device = torch.device(config.run.device)
    run = _new_run(config)
    report_path = run.root / "conditional_continuous_mode_reachability_report.json"
    basis = _build_modes(train_images, train_labels, evaluator, config)
    basis_hash = tensor_sha256(basis)
    panel_metrics: dict[str, object] = {}
    panel_hashes: dict[str, object] = {}
    checks: list[GateResult] = []
    checks.append(GateResult("basis_hash", basis_hash == EXPECTED_CONTINUOUS_BASIS_HASH, {"observed": basis_hash, "expected": EXPECTED_CONTINUOUS_BASIS_HASH}))
    if device.type != "cuda" or not torch.cuda.is_available():
        checks.append(GateResult("cuda_available", False, {"requested_device": str(device), "cuda_available": torch.cuda.is_available()}))
        classification = "harness_invalid"
    else:
        evaluator = evaluator.to(device).eval()
        bundle = bundle.to(device)
        probe = _ContinuousProbeGenerator(bundle.generator.class_carrier_logits.detach(), basis.to(device), gain=SAFE_PROBE_ALPHA).to(device)
        probe_state_before = state_content_hash(probe.state_dict())
        axis_codes, axis_labels = _fixed_control_probe(device=device)
        panels = {"axis": (axis_codes, axis_labels), "uniform_1234": (_random_codes(170, device=device, seed=1234), axis_labels), "uniform_5678": (_random_codes(170, device=device, seed=5678), axis_labels)}
        real_views = _real_probe_views(train_images, train_labels).to(device)
        baseline_raw: dict[str, torch.Tensor] = {}
        baseline_commands: dict[str, torch.Tensor] = {}
        for name, (codes, panel_labels) in panels.items():
            with torch.no_grad():
                command = probe(codes, panel_labels)
                raw = raw_from_generator_batched(bundle.system, probe, codes, panel_labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
            baseline_raw[name], baseline_commands[name] = raw, command
            metrics = _panel_metrics(raw, command, panel_labels, evaluator, real_views, config.conditional_generation.fixed_exposure_scale)
            panel_metrics[name] = metrics
            raw_path = run.root / f"continuous_reachability_{name}_raw_cmos.pt"
            atomic_torch_save(raw_path, raw.detach().cpu())
            panel_hashes[name] = {"raw": file_sha256(raw_path), "contact_sheet": _full_contact_sheet(run.root / f"continuous_reachability_{name}_10x17.png", raw, panel_labels, config.conditional_generation.fixed_exposure_scale)}
        zero_codes = torch.zeros(10, 8, device=device)
        zero_labels = torch.arange(10, device=device)
        with torch.no_grad():
            source_zero = raw_from_generator_batched(bundle.system, bundle.generator, zero_codes, zero_labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
            probe_zero = raw_from_generator_batched(bundle.system, probe, zero_codes, zero_labels, microbatch_size=config.execution.evaluation_microbatch_size, activation_checkpoint=False)
        axis_raw = baseline_raw["axis"]
        axis_commands = baseline_commands["axis"]
        direction_response = []
        logit_response = []
        for label in range(10):
            zero_raw = axis_raw[label * 17]
            zero_rms = zero_raw.square().mean().sqrt().clamp_min(1.0e-12)
            for direction in range(8):
                plus = axis_raw[label * 17 + 1 + direction]
                minus = axis_raw[label * 17 + 9 + direction]
                direction_response.append(float((plus - minus).square().mean().sqrt().div(zero_rms).item()))
                plus_logit = torch.logit(axis_commands[label * 17 + 1 + direction].clamp(1.0e-6, 1.0 - 1.0e-6))
                zero_logit = torch.logit(axis_commands[label * 17].clamp(1.0e-6, 1.0 - 1.0e-6))
                logit_response.append(float((plus_logit - zero_logit).square().mean().sqrt().item()))
        axis_logit_rms = sum(logit_response) / len(logit_response)
        checks.extend((
            GateResult("safe_axis_logit_rms", abs(axis_logit_rms - SAFE_AXIS_LOGIT_RMS) <= 0.002, {"observed": axis_logit_rms, "target": SAFE_AXIS_LOGIT_RMS}),
            GateResult("axis_semantic_accuracy", float(panel_metrics["axis"]["accuracy"]) >= 0.85, {"accuracy": panel_metrics["axis"]["accuracy"]}),
            GateResult("uniform_semantic_accuracy", all(float(panel_metrics[name]["accuracy"]) >= 0.80 for name in ("uniform_1234", "uniform_5678")), {name: panel_metrics[name]["accuracy"] for name in ("uniform_1234", "uniform_5678")}),
            GateResult("code_fraction", all(float(panel_metrics[name]["camera_code_fraction"]) >= 0.10 and float(panel_metrics[name]["physical_slm_code_fraction"]) >= 0.10 for name in panel_metrics), {}),
            GateResult("direction_response", min(direction_response) >= 0.01, {"minimum": min(direction_response)}),
            GateResult("saturation", all(float(panel_metrics[name]["saturation"]) <= 0.08 for name in panel_metrics), {}),
            GateResult("zero_code_unchanged", bool(torch.equal(source_zero, probe_zero)), {"max_abs": float((source_zero - probe_zero).abs().max().item())}),
            GateResult("finite", all(bool(panel_metrics[name]["finite"]) for name in panel_metrics), {}),
        ))
        # Read-only gradient observability.  The fixed 170-sample panel is
        # streamed one class at a time so a 6 GB GPU never retains 170 D2NN
        # propagation graphs.  No optimizer is constructed.
        probe.class_carrier_logits.requires_grad_(True)
        probe.class_modes.requires_grad_(True)
        groups = {"class_carrier": (probe.class_carrier_logits,), "continuous_modes": (probe.class_modes,), "encoder_phase": (bundle.system.encoder.phase,), "decoder_phase": (bundle.system.decoder.phase,)}
        gradient_metrics = {}
        tangent_metrics: dict[str, dict[str, object]] = {}
        for name, (codes, panel_labels) in panels.items():
            gradient_metrics[name], group_gradients = _streaming_panel_gradients(
                system=bundle.system,
                generator=probe,
                evaluator=evaluator,
                codes=codes,
                labels=panel_labels,
                real_views=real_views,
                groups=groups,
                exposure=config.conditional_generation.fixed_exposure_scale,
                microbatch_size=config.execution.training_microbatch_size,
            )
            tangent_metrics[name] = {}
            for group_name, parameters in groups.items():
                gradient = group_gradients.get(group_name)
                if gradient is None:
                    tangent_metrics[name][group_name] = {"valid": False, "reason": "gradient_gate_failed"}
                    continue
                tangent_metrics[name][group_name] = _safe_group_probe(
                    group_name=group_name,
                    parameter=parameters[0],
                    gradient=gradient,
                    system=bundle.system,
                    generator=probe,
                    evaluator=evaluator,
                    codes=codes,
                    labels=panel_labels,
                    baseline_raw=baseline_raw[name],
                    real_views=real_views,
                    exposure=config.conditional_generation.fixed_exposure_scale,
                    microbatch_size=config.execution.evaluation_microbatch_size,
                )
                derivatives = torch.tensor(tangent_metrics[name][group_name].get("direction_derivative", []), device=device)
                class_fractions = {
                    str(label): float((derivatives[panel_labels == label] > 0).float().mean().item())
                    for label in range(10)
                } if derivatives.numel() else {}
                tangent_metrics[name][group_name]["class_positive_fraction"] = class_fractions
        tangent_checks: list[GateResult] = []
        for group_name in groups:
            result_by_panel = {name: tangent_metrics[name].get(group_name, {}) for name in panels}
            if group_name == "continuous_modes":
                positive_limits = {"axis": 0.90, "uniform_1234": 0.80, "uniform_5678": 0.80}
                direction_ok = all(
                    bool(item.get("valid", False))
                    and float(item.get("low_margin_positive_fraction", 0.0)) >= positive_limits[name]
                    and float(item.get("class_positive_fraction", {}).get("5", 0.0)) >= 0.90
                    and float(item.get("class_positive_fraction", {}).get("7", 0.0)) >= 0.90
                    for name, item in result_by_panel.items()
                )
                semantic_probe_ok = all(
                    float(item.get("low_margin_median_delta", -1.0)) >= 0.01
                    and int(item.get("correct_count_delta", -1)) >= 0
                    and float(item.get("camera_fraction_delta", -1.0)) >= -0.01
                    and min(float(value) for value in item.get("feature_ratio_delta", [-1.0])) >= -0.03
                    and float(item.get("saturation", 1.0)) <= 0.08
                    and float(item.get("zero_code_max_abs", 1.0)) <= 1.0e-6
                    for item in result_by_panel.values()
                )
                tangent_checks.extend((
                    GateResult("continuous_modes_directional_semantics", direction_ok, {name: item for name, item in result_by_panel.items()}),
                    GateResult("continuous_modes_protected_probe", semantic_probe_ok, {name: item for name, item in result_by_panel.items()}),
                ))
            else:
                tangent_checks.append(GateResult(f"{group_name}_stateless_probe", all(bool(item.get("valid", False)) for item in result_by_panel.values()), result_by_panel))
        checks.append(GateResult("gradient_observability", all(item[group]["finite_nonzero"] for item in gradient_metrics.values() for group in item), gradient_metrics))
        checks.extend(tangent_checks)
        tangent_passed = all(item.passed for item in tangent_checks if item.name in {"continuous_modes_directional_semantics", "continuous_modes_protected_probe"})
        carrier_or_phase_passed = any(item.passed for item in tangent_checks if item.name in {"class_carrier_stateless_probe", "encoder_phase_stateless_probe", "decoder_phase_stateless_probe"})
        failed_names = {item.name for item in checks if not item.passed}
        harness_failed = bool(failed_names & {"basis_hash", "cuda_available", "zero_code_unchanged", "finite", "parameters_unchanged"})
        transport_failed = bool(failed_names & {"safe_axis_logit_rms", "code_fraction", "direction_response", "saturation"})
        if harness_failed:
            classification = "harness_invalid"
        elif transport_failed:
            classification = "safe_scale_transport_insufficient"
        elif tangent_passed:
            classification = "continuous_semantic_alignment_reachable"
        elif carrier_or_phase_passed:
            classification = "continuous_semantic_objective_conflict"
        else:
            classification = "continuous_semantic_objective_conflict"
        report_extra = {"direction_response_min": min(direction_response), "direction_response": direction_response, "axis_logit_rms": axis_logit_rms, "gradient_metrics": gradient_metrics, "tangent_metrics": tangent_metrics, "tangent_checks": [{"name": item.name, "passed": item.passed, "metrics": dict(item.metrics)} for item in tangent_checks], "panel_hashes": panel_hashes}
    after = state_content_hash(bundle.state_dict())
    probe_state_after = state_content_hash(probe.state_dict()) if "probe" in locals() else None
    parameters_unchanged = before == after and (probe_state_before == probe_state_after if "probe_state_before" in locals() else True)
    checks.append(GateResult("parameters_unchanged", parameters_unchanged, {"bundle_before": before, "bundle_after": after, "probe_before": probe_state_before if "probe_state_before" in locals() else None, "probe_after": probe_state_after}))
    if not parameters_unchanged:
        classification = "harness_invalid"
    report = {"contract": CONTINUOUS_REACHABILITY_CONTRACT, "classification": classification, "basis_hash": basis_hash, "safe_probe_alpha": SAFE_PROBE_ALPHA, "panels": panel_metrics, "artifacts": panel_hashes, "state_hash_before": before, "state_hash_after": after, "probe_state_hash_before": probe_state_before if "probe_state_before" in locals() else None, "probe_state_hash_after": probe_state_after, "optimizer_created": False, **locals().get("report_extra", {})}
    report["probe_mode"] = "reversible_single_delta_no_optimizer"
    atomic_write_json(report_path, report)
    basis_manifest = run.root / "continuous_mode_basis_manifest.json"
    atomic_write_json(basis_manifest, {"basis_hash": basis_hash, "shape": list(basis.shape), "contract": CONTINUOUS_REACHABILITY_CONTRACT, "checkpoint_created": False})
    passed = classification == "continuous_semantic_alignment_reachable" and all(item.passed for item in checks)
    record = EvidenceRecordV2(2, "conditional_continuous_mode_reachability", config.run.evidence_tier, "passed" if passed else "failed", run.git_commit, run.dirty_patch_hash, run.config_hash, expected, {"capacity_evidence": file_sha256(args.capacity_evidence), "report": file_sha256(report_path), "basis_manifest": file_sha256(basis_manifest), "semantic_checkpoint": checkpoint_hash, "basis_hash": basis_hash}, tuple(checks), ("conditional-continuous-mode-admission-design",) if passed else (), None if passed else classification)
    _write_evidence(run, "conditional_continuous_mode_reachability_evidence.json", record)
    print(run.root)
    return 0
