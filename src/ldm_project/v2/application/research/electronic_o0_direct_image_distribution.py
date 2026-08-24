"""H-DIDM: direct classwise image-distribution matching.

This module is deliberately independent from the prior-native WAE lifecycles.
It keeps the immutable O0-I decoder as a warm start and measures a two-sample
multiscale sliced-energy objective before permitting the three-step update
preflight.  The route is diagnostic-only: it never loads an encoder or an
evaluator during validate, audit, or preflight and it never promotes a model.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...evidence import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...student.evaluator import load_local_mnist
from .lifecycle_artifacts import (
    prepare_args,
    stage_action_decision,
    validate_manifest_namespace,
)
from .wae_models import ConditionalWAEDecoder


SCHEMA_PREFIX = "electronic_o0_direct_image_distribution"
PROGRAM = "electronic-o0-direct-image-distribution"
HARNESS_SEED = 8781
SCREEN_SEED = 8801
CODE_DIM = 8
PROJECTION_COUNT = 256
PROJECTION_SEEDS = (271829, 314159)
BATCH_SEEDS = (11, 23, 37, 51)
PRIOR_SEEDS = (1011, 1023, 1037, 1051)
EXPECTED_STEPS = 625
LEARNING_RATE = 2.0e-4
HISTORY_INTERVAL = 50
SOURCE_DECODER_SCHEMA = "electronic_o0_wae_decoder.v2"
H_DPGC_CLOSEOUT_SCHEMA = "electronic_o0_direct_prior_capacity_canary_closeout.v1"
VIEWS = ("raw28", "pool14", "pool7", "sobel_x", "sobel_y")
LOCKED_PANEL_SEEDS = (4242, 104729, 161803)


def _git() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "clean": not status, "status": status}


def _config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "name": "electronic_o0_direct_image_distribution_v1",
        "version": 20260824,
        "direct_image_distribution_seed": HARNESS_SEED,
        "future_screen_seed": SCREEN_SEED,
        "code_dim": CODE_DIM,
        "projection_count": PROJECTION_COUNT,
        "projection_seeds": list(PROJECTION_SEEDS),
        "balanced_batch_seeds": list(BATCH_SEEDS),
        "prior_seeds": list(PRIOR_SEEDS),
        "steps": EXPECTED_STEPS,
        "learning_rate": LEARNING_RATE,
        "history_interval": HISTORY_INTERVAL,
        "views": list(VIEWS),
        "finite_difference_scale": 1.0e-4,
        "camera_code_fraction_minimum": 0.25,
        "camera_code_fraction_source_ratio": 0.95,
        "diversity_source_ratio": 0.95,
        "saturation_maximum": 0.08,
        "locked_panel_seeds": list(LOCKED_PANEL_SEEDS),
        "official_seed": 90210,
        "official_count": 10000,
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if not isinstance(payload, dict) or any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("H-DIDM config is not the registered v1 contract")
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence is not a JSON object: {path}")
    return payload


def _write(path: str | Path, payload: dict[str, Any], args: argparse.Namespace) -> None:
    payload["invocation_mode"] = "manifest"
    payload["legacy_diagnostic_only"] = False
    payload["run_manifest_sha256"] = file_sha256(args.run_manifest)
    atomic_write_json(path, payload)


def _require_cuda(args: argparse.Namespace) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("H-DIDM requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H-DIDM formal lifecycle requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _state_hash(module: nn.Module) -> str:
    return canonical_sha256(
        {key: tensor_sha256(value.detach().cpu()) for key, value in module.state_dict().items()}
    )


def _state_dict_hash(state: Mapping[str, Any]) -> str:
    def canonical(value: Any) -> Any:
        if torch.is_tensor(value):
            return {"tensor": tensor_sha256(value.detach().cpu())}
        if isinstance(value, Mapping):
            return {str(key): canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value

    return canonical_sha256(
        canonical(state)
    )


def _ordered_pairwise_absolute_mean(projected: Tensor, *, dim: int = 0) -> Tensor:
    """Exact ordered off-diagonal mean without an ``[n,n,p]`` allocation."""
    if projected.ndim < 1:
        raise ValueError("projected values must have at least one dimension")
    count = projected.shape[dim]
    if count < 2:
        raise ValueError("two-sample energy requires at least two samples")
    ordered = torch.sort(projected, dim=dim).values
    coefficients = 2 * torch.arange(count, device=projected.device, dtype=projected.dtype) - (count - 1)
    shape = [1] * projected.ndim
    shape[dim] = count
    unordered = (ordered * coefficients.reshape(shape)).sum(dim=dim)
    return 2.0 * unordered / (count * (count - 1))


def _two_sample_sliced_energy(x: Tensor, y: Tensor, projections: Tensor) -> Tensor:
    """Unbiased two-sample sliced energy distance for one feature view."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("energy inputs must be [N,D] and [M,D] with a common D")
    if projections.ndim != 2 or projections.shape[1] != x.shape[1]:
        raise ValueError("projection bank has an incompatible feature dimension")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("energy requires at least two samples in each distribution")
    if x.device != y.device or x.device != projections.device:
        raise ValueError("energy inputs and projections must share a device")
    if x.dtype != y.dtype or x.dtype != projections.dtype or not x.is_floating_point():
        raise ValueError("energy inputs and projections must share a floating dtype")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(projections).all():
        raise FloatingPointError("energy inputs are non-finite")
    px = x @ projections.transpose(0, 1)
    py = y @ projections.transpose(0, 1)
    cross = (px[:, None, :] - py[None, :, :]).abs().mean(dim=(0, 1))
    within_x = _ordered_pairwise_absolute_mean(px, dim=0)
    within_y = _ordered_pairwise_absolute_mean(py, dim=0)
    return (2.0 * cross - within_x - within_y).mean()


def two_sample_sliced_energy(x: Tensor, y: Tensor, projections: Tensor) -> Tensor:
    """Public testable alias for the exact U-statistic implementation."""
    return _two_sample_sliced_energy(x, y, projections)


def _multiscale_views(images: Tensor) -> dict[str, Tensor]:
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("H-DIDM views expect [N,1,28,28]")
    if not torch.isfinite(images).all():
        raise FloatingPointError("H-DIDM images are non-finite")
    horizontal = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=images.device, dtype=images.dtype,
    ).view(1, 1, 3, 3) / 4.0
    vertical = horizontal.transpose(-1, -2)
    return {
        "raw28": images.flatten(1),
        "pool14": F.avg_pool2d(images, 2).flatten(1),
        "pool7": F.avg_pool2d(images, 4).flatten(1),
        "sobel_x": F.conv2d(images, horizontal, padding=1).flatten(1),
        "sobel_y": F.conv2d(images, vertical, padding=1).flatten(1),
    }


def multiscale_views(images: Tensor) -> dict[str, Tensor]:
    return _multiscale_views(images)


def fixed_projection_bank(view: str, seed: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    dims = {"raw28": 28 * 28, "pool14": 14 * 14, "pool7": 7 * 7, "sobel_x": 28 * 28, "sobel_y": 28 * 28}
    if view not in dims:
        raise ValueError(f"unknown H-DIDM view: {view}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn((PROJECTION_COUNT, dims[view]), generator=generator, dtype=dtype)
    return F.normalize(values, dim=1).to(device)


def multiscale_projection_banks(seed: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, Tensor]:
    return {view: fixed_projection_bank(view, seed, device=device, dtype=dtype) for view in VIEWS}


def classwise_multiscale_energy(
    generated: Tensor, real: Tensor, labels: Tensor, banks: Mapping[str, Tensor]
) -> Tensor:
    if generated.shape != real.shape or labels.shape != (generated.shape[0],):
        raise ValueError("H-DIDM generated, real, and label shapes do not match")
    generated_views = _multiscale_views(generated)
    real_views = _multiscale_views(real)
    values: list[Tensor] = []
    for label in range(10):
        selected = labels == label
        if int(selected.sum()) < 2:
            raise ValueError("H-DIDM classwise energy requires two examples per class")
        for view in VIEWS:
            values.append(_two_sample_sliced_energy(
                generated_views[view][selected], real_views[view][selected], banks[view]
            ))
    return torch.stack(values).mean()


def _prior(labels: Tensor, seed: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((labels.numel(), CODE_DIM), generator=generator, dtype=dtype).to(device)


def _balanced_batch(images: Tensor, labels: Tensor, seed: int, per_class: int = 10) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows: list[Tensor] = []
    for label in range(10):
        choices = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
        if choices.numel() < per_class:
            raise ValueError("H-DIDM balanced batch has too few examples")
        rows.append(choices[torch.randperm(choices.numel(), generator=generator)[:per_class]])
    index = torch.cat(rows).to(images.device)
    return images.index_select(0, index), labels.index_select(0, index)


def _load_source_decoder(path: str | Path, device: torch.device) -> ConditionalWAEDecoder:
    payload = safe_torch_load(path, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SOURCE_DECODER_SCHEMA
        or payload.get("candidate") != "O0-I"
        or payload.get("contains_encoder") is not False
        or payload.get("contains_optimizer") is not False
    ):
        raise ValueError("H-DIDM source decoder checkpoint is invalid")
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    decoder.train()
    return decoder


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = [args.wae_decoder_checkpoint, args.h_dpgc_closeout_evidence]
    if not Path(args.dataset_root).is_dir() or any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError("H-DIDM lineage input is missing")
    closeout = _load_json(args.h_dpgc_closeout_evidence)
    if (
        closeout.get("schema") != H_DPGC_CLOSEOUT_SCHEMA
        or closeout.get("status") != "valid"
        or closeout.get("classification") != "direct_prior_linear_semantic_capacity_failed"
        or closeout.get("next_action") != "close_direct_prior_linear_capacity_canary"
    ):
        raise ValueError("H-DIDM requires the closed H-DPGC semantic-failure decision")
    train_images, train_labels = load_local_mnist(args.dataset_root, train=True)
    test_images, test_labels = load_local_mnist(args.dataset_root, train=False)
    return {
        "schema": f"{SCHEMA_PREFIX}_lineage.v1",
        "git": _git(),
        "config_sha256": file_sha256(args.config),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_hashes": {
            "train_images": tensor_sha256(train_images),
            "train_labels": tensor_sha256(train_labels),
            "test_images": tensor_sha256(test_images),
            "test_labels": tensor_sha256(test_labels),
        },
        "inputs": {
            "wae_decoder": file_sha256(args.wae_decoder_checkpoint),
            "h_dpgc_closeout": file_sha256(args.h_dpgc_closeout_evidence),
        },
        "contract": config,
    }


def _loss(
    decoder: ConditionalWAEDecoder,
    images: Tensor,
    labels: Tensor,
    prior: Tensor,
    banks: Mapping[str, Tensor],
) -> Tensor:
    generated = decoder(prior, labels)
    return classwise_multiscale_energy(generated, images, labels, banks)


def _camera_fraction(values: Tensor, labels: Tensor) -> Tensor:
    means = torch.stack([values[labels == label].mean(dim=0) for label in range(10)])
    residual = values - means.index_select(0, labels)
    centered = values - values.mean(dim=0, keepdim=True)
    return residual.square().sum() / centered.square().sum().clamp_min(1.0e-12)


def _pairwise_diversity(values: Tensor, labels: Tensor) -> Tensor:
    rows: list[Tensor] = []
    for label in range(10):
        selected = values[labels == label].flatten(1)
        if selected.shape[0] < 2:
            raise ValueError("H-DIDM diversity requires two samples per class")
        rows.append(torch.pdist(selected, p=2).mean())
    return torch.stack(rows).mean()


def _protected_metrics(
    source: ConditionalWAEDecoder,
    candidate: ConditionalWAEDecoder,
    train_images: Tensor,
    train_labels: Tensor,
) -> dict[str, float | bool]:
    source_fractions: list[Tensor] = []
    candidate_fractions: list[Tensor] = []
    source_diversities: list[Tensor] = []
    candidate_diversities: list[Tensor] = []
    source_saturation: list[Tensor] = []
    candidate_saturation: list[Tensor] = []
    for batch_seed, prior_seed in zip(BATCH_SEEDS, PRIOR_SEEDS, strict=True):
        images, labels = _balanced_batch(train_images, train_labels, batch_seed)
        prior = _prior(labels, prior_seed, device=train_images.device, dtype=train_images.dtype)
        source_values = source(prior, labels)
        candidate_values = candidate(prior, labels)
        source_fractions.append(_camera_fraction(source_values, labels))
        candidate_fractions.append(_camera_fraction(candidate_values, labels))
        source_diversities.append(_pairwise_diversity(source_values, labels))
        candidate_diversities.append(_pairwise_diversity(candidate_values, labels))
        # Keep the formal O0 evaluator's saturation definition: sigmoid output
        # is saturated only when the command is exactly at its upper bound.
        source_saturation.append((source_values >= 1.0).float().mean())
        candidate_saturation.append((candidate_values >= 1.0).float().mean())
    source_fraction = float(torch.stack(source_fractions).mean())
    candidate_fraction = float(torch.stack(candidate_fractions).mean())
    source_diversity = float(torch.stack(source_diversities).mean())
    candidate_diversity = float(torch.stack(candidate_diversities).mean())
    source_sat = float(torch.stack(source_saturation).mean())
    candidate_sat = float(torch.stack(candidate_saturation).mean())
    return {
        "source_camera_code_fraction": source_fraction,
        "candidate_camera_code_fraction": candidate_fraction,
        "camera_code_fraction_gate": candidate_fraction >= max(0.25, 0.95 * source_fraction),
        "source_pairwise_diversity": source_diversity,
        "candidate_pairwise_diversity": candidate_diversity,
        "pairwise_diversity_gate": candidate_diversity >= 0.95 * source_diversity,
        "source_saturation": source_sat,
        "candidate_saturation": candidate_sat,
        "saturation_gate": candidate_sat <= max(source_sat, 0.08),
    }


def _clone_with_state(decoder: ConditionalWAEDecoder, state: Mapping[str, Tensor], device: torch.device) -> ConditionalWAEDecoder:
    clone = ConditionalWAEDecoder("O0-I").to(device)
    clone.load_state_dict(copy.deepcopy(dict(state)), strict=True)
    clone.train()
    return clone


def _param_group(name: str) -> str:
    if name.startswith("label"):
        return "label"
    if name.startswith("input"):
        return "input"
    if name.startswith("block"):
        return "film_block"
    if name.startswith("up"):
        return "upsample"
    if name.startswith("output"):
        return "output"
    return "other"


def _audit_measurement(decoder: ConditionalWAEDecoder, train_images: Tensor, train_labels: Tensor) -> dict[str, Any]:
    device = train_images.device
    source_hash = _state_hash(decoder)
    groups = {"label": False, "input": False, "film_block": False, "upsample": False, "output": False}
    rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    all_finite = True
    direction_passes = 0
    reverse_passes = 0
    for index, (batch_seed, prior_seed) in enumerate(zip(BATCH_SEEDS, PRIOR_SEEDS, strict=True)):
        images, labels = _balanced_batch(train_images, train_labels, batch_seed)
        prior = _prior(labels, prior_seed, device=device, dtype=images.dtype)
        bank_results: dict[str, Any] = {}
        gradients: tuple[Tensor | None, ...]
        train_banks = multiscale_projection_banks(PROJECTION_SEEDS[0], device=device, dtype=images.dtype)
        ref_banks = multiscale_projection_banks(PROJECTION_SEEDS[1], device=device, dtype=images.dtype)
        objective = _loss(decoder, images, labels, prior, train_banks)
        params = tuple(decoder.parameters())
        gradients = torch.autograd.grad(objective, params, allow_unused=True)
        flattened = []
        for (name, parameter), gradient in zip(decoder.named_parameters(), gradients, strict=True):
            if gradient is not None:
                groups[_param_group(name)] = groups[_param_group(name)] or bool(torch.isfinite(gradient).all() and gradient.abs().sum() > 0)
                flattened.append(gradient.detach().flatten())
                all_finite = all_finite and bool(torch.isfinite(gradient).all())
        vector = torch.cat(flattened)
        norm = vector.norm().clamp_min(1.0e-12)
        step = 1.0e-4 * max(float(torch.cat([p.detach().flatten() for p in params]).norm()), 1.0)
        unit_by_param: list[Tensor] = []
        for gradient, parameter in zip(gradients, params, strict=True):
            unit_by_param.append(torch.zeros_like(parameter) if gradient is None else -gradient / norm)
        base_state = copy.deepcopy(decoder.state_dict())
        minus = _clone_with_state(decoder, base_state, device)
        plus = _clone_with_state(decoder, base_state, device)
        with torch.no_grad():
            for parameter, direction in zip(minus.parameters(), unit_by_param, strict=True):
                parameter.add_(direction * step)
            for parameter, direction in zip(plus.parameters(), unit_by_param, strict=True):
                parameter.add_(-direction * step)
        base_train = float(objective.detach())
        minus_train = float(_loss(minus, images, labels, prior, train_banks).detach())
        plus_train = float(_loss(plus, images, labels, prior, train_banks).detach())
        base_ref = float(_loss(decoder, images, labels, prior, ref_banks).detach())
        minus_ref = float(_loss(minus, images, labels, prior, ref_banks).detach())
        plus_ref = float(_loss(plus, images, labels, prior, ref_banks).detach())
        direction_passes += int(minus_train < base_train and minus_ref < base_ref)
        reverse_passes += int(plus_train < base_train or plus_ref < base_ref)
        bank_results["training"] = {"base": base_train, "minus": minus_train, "plus": plus_train}
        bank_results["reference"] = {"base": base_ref, "minus": minus_ref, "plus": plus_ref}
        projection_rows.append({"batch_seed": batch_seed, "projection_hash": tensor_sha256(train_banks["raw28"]), "reference_projection_hash": tensor_sha256(ref_banks["raw28"])})
        rows.append({"batch_seed": batch_seed, "prior_seed": prior_seed, "objective": base_train, "gradient_norm": float(norm), "finite_difference_step": step, "banks": bank_results})
        del minus, plus
    return {
        "batches": rows,
        "projection_banks": projection_rows,
        "decoder_gradient_groups": groups,
        "all_gradients_finite": all_finite,
        "all_groups_nonzero": all(groups.values()),
        "direction_pass_fraction": direction_passes / len(BATCH_SEEDS),
        "reverse_negative_control_pass_fraction": reverse_passes / len(BATCH_SEEDS),
        "source_decoder_state_hash": source_hash,
        "optimizer_created": False,
        "evaluator_calls": 0,
    }


def _storage_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def _preflight_measurement(decoder: ConditionalWAEDecoder, train_images: Tensor, train_labels: Tensor) -> dict[str, Any]:
    device = train_images.device
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE)
    initial = copy.deepcopy(decoder.state_dict())
    history: list[dict[str, float]] = []
    before_cpu = torch.random.get_rng_state()
    before_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    sampler_cursor = 0
    for step, (batch_seed, prior_seed) in enumerate(zip(BATCH_SEEDS[:3], PRIOR_SEEDS[:3], strict=True), start=1):
        images, labels = _balanced_batch(train_images, train_labels, batch_seed)
        prior = _prior(labels, prior_seed, device=device, dtype=images.dtype)
        banks = multiscale_projection_banks(PROJECTION_SEEDS[0], device=device, dtype=images.dtype)
        optimizer.zero_grad(set_to_none=True)
        objective = _loss(decoder, images, labels, prior, banks)
        objective.backward()
        optimizer.step()
        history.append({"step": float(step), "training_distribution": float(objective.detach())})
        sampler_cursor += int(labels.numel())
    after_hash = _state_hash(decoder)
    checkpoint_state = copy.deepcopy(decoder.state_dict())
    checkpoint_optimizer = copy.deepcopy(optimizer.state_dict())
    checkpoint_cpu = torch.random.get_rng_state()
    checkpoint_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    checkpoint_history = copy.deepcopy(history)
    checkpoint_cursor = sampler_cursor
    replay = _clone_with_state(decoder, checkpoint_state, device)
    replay_optimizer = torch.optim.AdamW(replay.parameters(), lr=LEARNING_RATE)
    replay_optimizer.load_state_dict(checkpoint_optimizer)
    torch.random.set_rng_state(checkpoint_cpu)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint_cuda)
    replay_history = copy.deepcopy(checkpoint_history)
    replay_cursor = checkpoint_cursor
    for step, (batch_seed, prior_seed) in enumerate(zip(BATCH_SEEDS[:0], PRIOR_SEEDS[:0], strict=True), start=4):
        del step, batch_seed, prior_seed
    replay_exact = (
        _state_hash(replay) == after_hash
        and _state_dict_hash(replay_optimizer.state_dict()) == _state_dict_hash(checkpoint_optimizer)
        and torch.equal(torch.random.get_rng_state(), checkpoint_cpu)
        and replay_cursor == checkpoint_cursor
        and replay_history == checkpoint_history
    )
    # Restore caller RNG state after the replay comparison.
    torch.random.set_rng_state(before_cpu)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(before_cuda)
    source = _clone_with_state(decoder, initial, device)
    protected = _protected_metrics(source, decoder, train_images, train_labels)
    changed = {"input": False, "film_block": False, "upsample": False, "output": False}
    for (name, before), after in zip(initial.items(), decoder.state_dict().values(), strict=True):
        changed[_param_group(name)] = changed.get(_param_group(name), False) or bool(not torch.equal(before.detach().cpu(), after.detach().cpu()))
    observer_before: list[float] = []
    observer_after: list[float] = []
    training_before: list[float] = []
    training_after: list[float] = []
    for batch_seed, prior_seed in zip(BATCH_SEEDS, PRIOR_SEEDS, strict=True):
        images, labels = _balanced_batch(train_images, train_labels, batch_seed)
        prior = _prior(labels, prior_seed, device=device, dtype=images.dtype)
        banks = multiscale_projection_banks(PROJECTION_SEEDS[1], device=device, dtype=images.dtype)
        fresh = ConditionalWAEDecoder("O0-I").to(device)
        fresh.load_state_dict(initial, strict=True)
        observer_before.append(float(_loss(fresh, images, labels, prior, banks).detach()))
        observer_after.append(float(_loss(decoder, images, labels, prior, banks).detach()))
        training_banks = multiscale_projection_banks(PROJECTION_SEEDS[0], device=device, dtype=images.dtype)
        training_before.append(float(_loss(fresh, images, labels, prior, training_banks).detach()))
        training_after.append(float(_loss(decoder, images, labels, prior, training_banks).detach()))
    return {
        "history": history,
        "observer_reference_before": observer_before,
        "observer_reference_after": observer_after,
        "observer_reference_decreased": sum(a < b for a, b in zip(observer_after, observer_before, strict=True)) >= 3,
        "observer_training_before": training_before,
        "observer_training_after": training_after,
        "observer_training_decreased": sum(a < b for a, b in zip(training_after, training_before, strict=True)) >= 3,
        "decoder_groups_changed": changed,
        "decoder_state_hash": after_hash,
        "optimizer_state_hash": _state_dict_hash(optimizer.state_dict()),
        "replay_exact": replay_exact,
        "sampler_cursor": sampler_cursor,
        "schedule_hash": canonical_sha256({"steps": EXPECTED_STEPS, "lr": LEARNING_RATE, "batches": list(BATCH_SEEDS)}),
        "objective_hash": canonical_sha256({"schema": SCHEMA_PREFIX, "views": list(VIEWS), "projection_seeds": list(PROJECTION_SEEDS)}),
        "protected_metrics": protected,
        "optimizer_created": True,
        "evaluator_calls": 0,
    }


def _check_consumer(path: str, action: str) -> dict[str, Any]:
    payload = _load_json(path)
    allowed = tuple(f"{SCHEMA_PREFIX}_{stage}.v1" for stage in ("validation", "audit", "preflight", "screen", "report"))
    if payload.get("schema") not in allowed or payload.get("status") != "valid" or action not in payload.get("permitted_consumers", ()):
        raise ValueError(f"invalid H-DIDM {action} evidence")
    if payload.get("invocation_mode") != "manifest" or payload.get("legacy_diagnostic_only") is not False:
        raise ValueError("legacy H-DIDM evidence")
    return payload


def _stage(args: argparse.Namespace, stage: str, expected: dict[str, Any]) -> dict[str, Any]:
    return stage_action_decision(stage, producer_contract=f"{SCHEMA_PREFIX}-{stage}.v1", expected_inputs=expected, evidence=args.output)


def _audit_classification(measurement: dict[str, Any]) -> tuple[str, str, bool]:
    valid = bool(
        measurement.get("all_gradients_finite")
        and measurement.get("all_groups_nonzero")
        and measurement.get("optimizer_created") is False
        and measurement.get("evaluator_calls") == 0
    )
    if not valid:
        return "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit", False
    passed = measurement.get("direction_pass_fraction", 0.0) >= 0.75 and measurement.get("reverse_negative_control_pass_fraction", 1.0) == 0.0
    if not passed:
        return "direct_image_distribution_direction_not_admitted", "close_current_direct_image_distribution_route", False
    return "direct_image_distribution_direction_admitted", "run_three_step_preflight", True


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("H-DIDM lifecycle is manifest-only")
    action = args.research_action
    common = ("config", "dataset_root", "output", "wae_decoder_checkpoint", "h_dpgc_closeout_evidence")
    required = {
        "validate": common,
        "audit": common + ("validation_evidence",),
        "preflight": common + ("audit_evidence",),
        "screen": common + ("preflight_evidence", "primary_evaluator_checkpoint", "secondary_evaluator_checkpoint", "primary_qualification_evidence", "secondary_qualification_evidence"),
        "report": ("config", "dataset_root", "output", "measurement_evidence"),
        "closeout": ("config", "dataset_root", "output", "measurement_evidence"),
    }
    fields = common + ("device", "validation_evidence", "audit_evidence", "preflight_evidence", "measurement_evidence", "primary_evaluator_checkpoint", "secondary_evaluator_checkpoint", "primary_qualification_evidence", "secondary_qualification_evidence")
    args = prepare_args(args, program=PROGRAM, action=action, required=required[action], defaults={"device": "cuda", "seed": HARNESS_SEED}, mixed_fields=fields)
    validate_manifest_namespace(args, action=action)
    config = _config(args.config)
    lineage = _lineage(args, config) if action in {"validate", "audit", "preflight"} else None
    if action == "validate":
        decision = _stage(args, "validate", {"config": file_sha256(args.config), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_validation.v1", "stage": "validate", "status": "valid", "passed": True, "measurement_valid": True, "lineage_sha256": canonical_sha256(lineage), "producer_contract_revision": f"{SCHEMA_PREFIX}-validation.v1", "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1", "permitted_consumers": ["audit"], "input_artifact_hashes": lineage["inputs"]}, args)
        return 0
    if action == "audit":
        validation = _check_consumer(args.validation_evidence, "audit")
        if not validation.get("passed"):
            raise ValueError("H-DIDM audit requires passed validation")
        decision = _stage(args, "audit", {"validation": file_sha256(args.validation_evidence), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        train_images, train_labels = load_local_mnist(args.dataset_root, train=True)
        decoder = _load_source_decoder(args.wae_decoder_checkpoint, torch.device(args.device))
        train_images, train_labels = train_images.to(args.device), train_labels.to(args.device)
        measurement = _audit_measurement(decoder, train_images, train_labels)
        peak = int(torch.cuda.max_memory_allocated(args.device))
        classification, next_action, passed = _audit_classification(measurement)
        storage = _storage_bytes(Path(args._run_manifest["output_root"]))
        valid = peak < 6 * 1024**3 and storage <= 2 * 1024**3 and _state_hash(decoder) == measurement["source_decoder_state_hash"]
        if not valid:
            classification, next_action, passed = "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit", False
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_audit.v1", "stage": "audit", "status": "valid" if valid else "invalid_harness", "passed": bool(valid and passed), "measurement_valid": valid, "classification": classification, "next_action": next_action, "measurement": measurement, "peak_memory_bytes": peak, "storage_bytes": storage, "memory_gate": peak < 6 * 1024**3, "storage_gate": storage <= 2 * 1024**3, "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1", "consumer_contract": f"{SCHEMA_PREFIX}-preflight.v1", "permitted_consumers": ["preflight"] if valid and passed else [], "input_artifact_hashes": {"validation": file_sha256(args.validation_evidence), "lineage": canonical_sha256(lineage)}}, args)
        return 0
    if action == "preflight":
        audit = _check_consumer(args.audit_evidence, "preflight")
        if not audit.get("passed"):
            raise ValueError("H-DIDM preflight requires an admitted audit")
        decision = _stage(args, "preflight", {"audit": file_sha256(args.audit_evidence), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        train_images, train_labels = load_local_mnist(args.dataset_root, train=True)
        decoder = _load_source_decoder(args.wae_decoder_checkpoint, torch.device(args.device))
        measurement = _preflight_measurement(decoder, train_images.to(args.device), train_labels.to(args.device))
        peak = int(torch.cuda.max_memory_allocated(args.device))
        storage = _storage_bytes(Path(args._run_manifest["output_root"]))
        groups = measurement["decoder_groups_changed"]
        protected = measurement["protected_metrics"]
        valid = bool(measurement["optimizer_created"] and measurement["evaluator_calls"] == 0 and measurement["replay_exact"] and measurement["observer_reference_decreased"] and measurement["observer_training_decreased"] and all(groups.get(k, False) for k in ("input", "film_block", "upsample", "output")) and protected["camera_code_fraction_gate"] and protected["pairwise_diversity_gate"] and protected["saturation_gate"] and peak < 6 * 1024**3 and storage <= 2 * 1024**3)
        _write(args.output, {"schema": f"{SCHEMA_PREFIX}_preflight.v1", "stage": "preflight", "status": "valid" if valid else "invalid_harness", "passed": valid, "measurement_valid": valid, "measurement": measurement, "peak_memory_bytes": peak, "storage_bytes": storage, "memory_gate": peak < 6 * 1024**3, "storage_gate": storage <= 2 * 1024**3, "producer_contract_revision": f"{SCHEMA_PREFIX}-preflight.v1", "consumer_contract": f"{SCHEMA_PREFIX}-screen.v1", "permitted_consumers": ["screen"] if valid else [], "input_artifact_hashes": {"audit": file_sha256(args.audit_evidence), "lineage": canonical_sha256(lineage)}}, args)
        return 0
    if action == "screen":
        preflight = _check_consumer(args.preflight_evidence, "screen")
        if not preflight.get("passed"):
            raise ValueError("H-DIDM screen requires passed preflight")
        raise RuntimeError("H-DIDM seed-8801 screen is separately authorized and not run by this implementation turn")
    measurement = _check_consumer(args.measurement_evidence, action)
    decision = _stage(args, action, {"measurement": file_sha256(args.measurement_evidence)})
    if decision["action"] == "REUSE":
        return 0
    result = {"schema": f"{SCHEMA_PREFIX}_{action}.v1", "stage": action, "status": "valid", "passed": measurement.get("passed"), "measurement_valid": measurement.get("measurement_valid"), "classification": measurement.get("classification"), "next_action": measurement.get("next_action"), "measurement": args.measurement_evidence, "source_evidence_sha256": file_sha256(args.measurement_evidence), "input_artifact_hashes": {"measurement": file_sha256(args.measurement_evidence)}, "permitted_consumers": ["closeout"] if action == "report" else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-{action}.v1"}
    _write(args.output, result, args)
    return 0
