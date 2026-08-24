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
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...student.evaluator import load_local_mnist, stratified_mnist_indices
from .lifecycle_artifacts import (
    prepare_args,
    stage_action_decision,
    validate_manifest_namespace,
)
from .electronic_o0_wae import (
    _measure_checkpoint,
    balanced_epoch_schedule,
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
TRAINING_SCHEMA = "electronic_o0_direct_image_distribution_training.v1"
DECODER_SCHEMA = "electronic_o0_direct_image_distribution_decoder.v1"
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


def _two_sample_sliced_energy(
    x: Tensor, y: Tensor, projections: Tensor, *, chunk_size: int | None = None
) -> Tensor:
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
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("energy chunk_size must be positive")
    step = px.shape[0] if chunk_size is None else chunk_size
    cross_sum = px.new_zeros((px.shape[1],))
    for start in range(0, px.shape[0], step):
        left = px[start : start + step]
        for other_start in range(0, py.shape[0], step):
            right = py[other_start : other_start + step]
            cross_sum = cross_sum + (left[:, None, :] - right[None, :, :]).abs().sum(dim=(0, 1))
    cross = cross_sum / (px.shape[0] * py.shape[0])
    within_x = _ordered_pairwise_absolute_mean(px, dim=0)
    within_y = _ordered_pairwise_absolute_mean(py, dim=0)
    return (2.0 * cross - within_x - within_y).mean()


def two_sample_sliced_energy(
    x: Tensor, y: Tensor, projections: Tensor, *, chunk_size: int | None = None
) -> Tensor:
    """Public testable alias for the exact U-statistic implementation."""
    return _two_sample_sliced_energy(x, y, projections, chunk_size=chunk_size)


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
    generated: Tensor,
    real: Tensor,
    labels: Tensor,
    banks: Mapping[str, Tensor],
    *,
    real_labels: Tensor | None = None,
) -> Tensor:
    if generated.shape != real.shape or labels.shape != (generated.shape[0],):
        raise ValueError("H-DIDM generated, real, and label shapes do not match")
    if real_labels is None:
        real_labels = labels
    if real_labels.shape != (real.shape[0],):
        raise ValueError("H-DIDM real labels have an invalid shape")
    generated_views = _multiscale_views(generated)
    real_views = _multiscale_views(real)
    values: list[Tensor] = []
    for label in range(10):
        generated_selected = labels == label
        real_selected = real_labels == label
        if int(generated_selected.sum()) < 2 or int(real_selected.sum()) < 2:
            raise ValueError("H-DIDM classwise energy requires two examples per class")
        for view in VIEWS:
            values.append(_two_sample_sliced_energy(
                generated_views[view][generated_selected], real_views[view][real_selected], banks[view]
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


def _classwise_multiscale_values(
    generated: Tensor,
    real: Tensor,
    labels: Tensor,
    banks: Mapping[str, Tensor],
    *,
    real_labels: Tensor | None = None,
) -> list[Tensor]:
    if real_labels is None:
        real_labels = labels
    generated_views = _multiscale_views(generated)
    real_views = _multiscale_views(real)
    values: list[Tensor] = []
    for label in range(10):
        generated_selected = labels == label
        real_selected = real_labels == label
        if int(generated_selected.sum()) < 2 or int(real_selected.sum()) < 2:
            raise ValueError("H-DIDM classwise values require two examples per class")
        values.append(torch.stack([
            _two_sample_sliced_energy(
                generated_views[view][generated_selected],
                real_views[view][real_selected],
                banks[view],
            )
            for view in VIEWS
        ]).mean())
    return values


def _control_images(images: Tensor, labels: Tensor, *, seed: int) -> dict[str, tuple[Tensor, Tensor]]:
    """Construct fixed, evaluator-free controls for the image-distribution audit."""
    selected_images, _ = _balanced_batch(images, labels, seed, per_class=20)
    first_rows, second_rows = [], []
    for label in range(10):
        class_rows = selected_images[label * 20 : (label + 1) * 20]
        first_rows.append(class_rows[:10])
        second_rows.append(class_rows[10:])
    first_images = torch.cat(first_rows)
    second_images = torch.cat(second_rows)
    fixed_labels = torch.arange(10, device=images.device).repeat_interleave(10)
    class_means = torch.stack([
        second_images[fixed_labels == label].mean(dim=0) for label in range(10)
    ])
    collapse = class_means.index_select(0, fixed_labels)
    blur = F.avg_pool2d(second_images, kernel_size=3, stride=1, padding=1)
    generator = torch.Generator(device="cpu").manual_seed(seed + 991)
    shuffled = fixed_labels.cpu().index_select(
        0, torch.randperm(fixed_labels.numel(), generator=generator)
    ).to(images.device)
    return {
        "matched": (first_images, fixed_labels),
        "class_mean_collapse": (collapse, fixed_labels),
        "blur": (blur, fixed_labels),
        "label_shuffle": (second_images, shuffled),
    }


def _formula_and_chunk_controls(*, device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(HARNESS_SEED)
    x = torch.randn((6, 7), generator=generator).to(device).requires_grad_(True)
    y = torch.randn((5, 7), generator=generator).to(device).requires_grad_(True)
    projections = F.normalize(torch.randn((9, 7), generator=generator).to(device), dim=1)
    fast = _two_sample_sliced_energy(x, y, projections)
    chunked = _two_sample_sliced_energy(x, y, projections, chunk_size=3)
    px, py = x @ projections.T, y @ projections.T
    cross = (px[:, None, :] - py[None, :, :]).abs().mean((0, 1))
    naive = (2.0 * cross - _naive_pairwise(px) - _naive_pairwise(py)).mean()
    fast_grad = torch.autograd.grad(fast, (x, y), retain_graph=True)
    naive_grad = torch.autograd.grad(naive, (x, y), retain_graph=True)
    return {
        "value_matches_naive": bool(torch.allclose(fast, naive, atol=1.0e-7, rtol=1.0e-6)),
        "gradient_matches_naive": bool(all(torch.allclose(left, right, atol=1.0e-6, rtol=1.0e-5) for left, right in zip(fast_grad, naive_grad, strict=True))),
        "chunk_matches_full": bool(torch.allclose(fast, chunked, atol=1.0e-7, rtol=1.0e-6)),
        "finite": bool(torch.isfinite(fast) and torch.isfinite(chunked)),
    }


def _naive_pairwise(projected: Tensor) -> Tensor:
    count = projected.shape[0]
    if count < 2:
        raise ValueError("naive pairwise control requires two samples")
    values = (projected[:, None, :] - projected[None, :, :]).abs()
    return values.sum(dim=(0, 1)) / (count * (count - 1))


def _control_measurement(
    images: Tensor, labels: Tensor, banks_by_seed: Mapping[int, Mapping[str, Tensor]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    passed = True
    for bank_seed, banks in banks_by_seed.items():
        controls = _control_images(images, labels, seed=BATCH_SEEDS[0])
        values: dict[str, float] = {}
        for name, (control_images, control_labels) in controls.items():
            values[name] = float(classwise_multiscale_energy(
                controls["matched"][0], control_images, controls["matched"][1], banks,
                real_labels=control_labels,
            ).detach())
        bank_passed = values["matched"] < values["class_mean_collapse"] and values["matched"] < values["blur"] and values["matched"] < values["label_shuffle"]
        passed = passed and bank_passed
        rows.append({"projection_seed": bank_seed, "values": values, "passed": bank_passed})
    return {"banks": rows, "passed": passed}


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
    train_bank_for_controls = multiscale_projection_banks(PROJECTION_SEEDS[0], device=device, dtype=train_images.dtype)
    reference_bank_for_controls = multiscale_projection_banks(PROJECTION_SEEDS[1], device=device, dtype=train_images.dtype)
    controls = _control_measurement(
        train_images,
        train_labels,
        {PROJECTION_SEEDS[0]: train_bank_for_controls, PROJECTION_SEEDS[1]: reference_bank_for_controls},
    )
    formula_controls = _formula_and_chunk_controls(device=device)
    repeated_train = multiscale_projection_banks(PROJECTION_SEEDS[0], device=device, dtype=train_images.dtype)
    repeated_reference = multiscale_projection_banks(PROJECTION_SEEDS[1], device=device, dtype=train_images.dtype)
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
        "controls": controls,
        "formula_and_chunk_controls": formula_controls,
        "projection_determinism": {
            "training_repeat": all(tensor_sha256(train_bank_for_controls[view]) == tensor_sha256(repeated_train[view]) for view in VIEWS),
            "reference_repeat": all(tensor_sha256(reference_bank_for_controls[view]) == tensor_sha256(repeated_reference[view]) for view in VIEWS),
        },
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


def _schedule_hash(schedule: tuple[Tensor, ...]) -> str:
    return canonical_sha256({"steps": len(schedule), "schedule": [tensor_sha256(item.cpu()) for item in schedule]})


def _objective_hash(schedule_hash: str) -> str:
    return canonical_sha256({
        "schema": TRAINING_SCHEMA,
        "formula_revision": "two_sample_sliced_energy_u_v1",
        "projection_seeds": list(PROJECTION_SEEDS),
        "schedule_hash": schedule_hash,
        "learning_rate": LEARNING_RATE,
        "steps": EXPECTED_STEPS,
    })


def _training_data(dataset_root: str, *, seed: int = SCREEN_SEED) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
    all_images, all_labels = load_local_mnist(dataset_root, train=True)
    train_rows, _ = stratified_mnist_indices(all_labels)
    images = all_images.index_select(0, train_rows)
    labels = all_labels.index_select(0, train_rows)
    schedule = balanced_epoch_schedule(labels, seed=seed, per_class=10)
    if len(schedule) != EXPECTED_STEPS:
        raise ValueError("H-DIDM schedule does not contain the registered 625 steps")
    return images, labels, schedule


def _training_data_from_tensors(
    images: Tensor, labels: Tensor, *, seed: int = SCREEN_SEED
) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
    images_cpu = images.detach().cpu()
    labels_cpu = labels.detach().cpu()
    schedule = balanced_epoch_schedule(labels_cpu, seed=seed, per_class=10)
    if len(schedule) != EXPECTED_STEPS:
        raise ValueError("H-DIDM schedule does not contain the registered 625 steps")
    return images_cpu, labels_cpu, schedule


def _rng_state() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _update_step(
    decoder: ConditionalWAEDecoder,
    optimizer: torch.optim.Optimizer,
    images: Tensor,
    labels: Tensor,
    prior_seed: int,
    banks: Mapping[str, Tensor],
) -> dict[str, float]:
    prior = _prior(labels, prior_seed, device=images.device, dtype=images.dtype)
    optimizer.zero_grad(set_to_none=True)
    objective = _loss(decoder, images, labels, prior, banks)
    if not torch.isfinite(objective):
        raise FloatingPointError("H-DIDM objective became non-finite")
    objective.backward()
    optimizer.step()
    return {"distribution": float(objective.detach()), "gradient_norm": float(torch.sqrt(sum(parameter.grad.detach().square().sum() for parameter in decoder.parameters() if parameter.grad is not None)))}


def _training_checkpoint(
    decoder: ConditionalWAEDecoder,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    step: int,
    batch_cursor: int,
    schedule_hash: str,
    objective_hash: str,
    history: list[dict[str, Any]],
    history_window: list[dict[str, Any]],
    source_hash: str,
) -> dict[str, Any]:
    return {
        "schema": TRAINING_SCHEMA,
        "seed": seed,
        "epoch": 1 if step >= EXPECTED_STEPS else 0,
        "step": step,
        "batch_cursor": batch_cursor,
        "decoder": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "rng_state": _rng_state(),
        "sampler_state": {"seed": seed, "cursor": batch_cursor, "steps": EXPECTED_STEPS},
        "schedule_hash": schedule_hash,
        "objective_hash": objective_hash,
        "history": list(history),
        "history_window": list(history_window),
        "source_decoder_sha256": source_hash,
        "contains_encoder": False,
    }


def _preflight_measurement(
    decoder: ConditionalWAEDecoder,
    train_images: Tensor,
    train_labels: Tensor,
    *,
    source_hash: str,
) -> dict[str, Any]:
    device = train_images.device
    images_cpu, labels_cpu, schedule = _training_data_from_tensors(train_images, train_labels)
    schedule_hash = _schedule_hash(schedule)
    objective_hash = _objective_hash(schedule_hash)
    initial = copy.deepcopy(decoder.state_dict())
    initial_rng = _rng_state()
    banks = multiscale_projection_banks(PROJECTION_SEEDS[0], device=device, dtype=train_images.dtype)
    uninterrupted = _clone_with_state(decoder, initial, device)
    _restore_rng_state(initial_rng)
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=LEARNING_RATE)
    updates: list[dict[str, Any]] = []
    history_window: list[dict[str, Any]] = []
    for step in range(3):
        batch_images = images_cpu.index_select(0, schedule[step]).to(device)
        batch_labels = labels_cpu.index_select(0, schedule[step]).to(device)
        update = _update_step(uninterrupted, uninterrupted_optimizer, batch_images, batch_labels, PRIOR_SEEDS[step % len(PRIOR_SEEDS)], banks)
        update["step"] = step + 1
        updates.append(update)
        history_window.append(update)
    final_state = copy.deepcopy(uninterrupted.state_dict())
    final_optimizer = copy.deepcopy(uninterrupted_optimizer.state_dict())
    final_rng = _rng_state()
    interrupted = _clone_with_state(decoder, initial, device)
    _restore_rng_state(initial_rng)
    interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=LEARNING_RATE)
    interrupted_updates: list[dict[str, Any]] = []
    for step in range(2):
        batch_images = images_cpu.index_select(0, schedule[step]).to(device)
        batch_labels = labels_cpu.index_select(0, schedule[step]).to(device)
        update = _update_step(interrupted, interrupted_optimizer, batch_images, batch_labels, PRIOR_SEEDS[step % len(PRIOR_SEEDS)], banks)
        update["step"] = step + 1
        interrupted_updates.append(update)
    checkpoint = _training_checkpoint(interrupted, interrupted_optimizer, seed=SCREEN_SEED, step=2, batch_cursor=2, schedule_hash=schedule_hash, objective_hash=objective_hash, history=[], history_window=interrupted_updates, source_hash=source_hash)
    resumed = _clone_with_state(decoder, checkpoint["decoder"], device)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=LEARNING_RATE)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_rng_state(checkpoint["rng_state"])
    batch_images = images_cpu.index_select(0, schedule[2]).to(device)
    batch_labels = labels_cpu.index_select(0, schedule[2]).to(device)
    resumed_update = _update_step(resumed, resumed_optimizer, batch_images, batch_labels, PRIOR_SEEDS[2], banks)
    resumed_update["step"] = 3
    interrupted_updates.append(resumed_update)
    replay_exact = (
        all(torch.equal(final_state[key].detach().cpu(), resumed.state_dict()[key].detach().cpu()) for key in final_state)
        and _state_dict_hash(final_optimizer) == _state_dict_hash(resumed_optimizer.state_dict())
        and torch.equal(final_rng["cpu"], torch.get_rng_state())
        and all(torch.equal(left, right) for left, right in zip(final_rng["cuda"], torch.cuda.get_rng_state_all(), strict=True))
        and interrupted_updates == updates
        and checkpoint["sampler_state"]["cursor"] == 2
        and checkpoint["schedule_hash"] == schedule_hash
        and checkpoint["objective_hash"] == objective_hash
    )
    _restore_rng_state(initial_rng)
    source = _clone_with_state(decoder, initial, device)
    protected = _protected_metrics(source, uninterrupted, train_images, train_labels)
    changed = {"input": False, "film_block": False, "upsample": False, "output": False}
    for (name, before), after in zip(initial.items(), uninterrupted.state_dict().values(), strict=True):
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
        observer_after.append(float(_loss(uninterrupted, images, labels, prior, banks).detach()))
        training_banks = multiscale_projection_banks(PROJECTION_SEEDS[0], device=device, dtype=images.dtype)
        training_before.append(float(_loss(fresh, images, labels, prior, training_banks).detach()))
        training_after.append(float(_loss(uninterrupted, images, labels, prior, training_banks).detach()))
    return {
        "history": updates,
        "observer_reference_before": observer_before,
        "observer_reference_after": observer_after,
        "observer_reference_decreased": sum(a < b for a, b in zip(observer_after, observer_before, strict=True)) >= 3,
        "observer_training_before": training_before,
        "observer_training_after": training_after,
        "observer_training_decreased": sum(a < b for a, b in zip(training_after, training_before, strict=True)) >= 3,
        "decoder_groups_changed": changed,
        "decoder_state_hash": _state_hash(uninterrupted),
        "optimizer_state_hash": _state_dict_hash(uninterrupted_optimizer.state_dict()),
        "replay_exact": replay_exact,
        "sampler_cursor": 3,
        "schedule_hash": schedule_hash,
        "objective_hash": objective_hash,
        "protected_metrics": protected,
        "optimizer_created": True,
        "evaluator_calls": 0,
    }


def _observer_summary(
    decoder: ConditionalWAEDecoder,
    images: Tensor,
    labels: Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    decoder.eval()
    values: dict[str, list[float]] = {"training": [], "reference": []}
    for bank_seed, name in zip(PROJECTION_SEEDS, ("training", "reference"), strict=True):
        banks = multiscale_projection_banks(bank_seed, device=device, dtype=images.dtype)
        for batch_seed, prior_seed in zip(BATCH_SEEDS, PRIOR_SEEDS, strict=True):
            batch_images, batch_labels = _balanced_batch(images, labels, batch_seed)
            prior = _prior(batch_labels, prior_seed, device=device, dtype=images.dtype)
            values[name].append(float(_loss(decoder, batch_images, batch_labels, prior, banks).detach()))
    decoder.train()
    return {
        "training_distribution": sum(values["training"]) / len(values["training"]),
        "reference_distribution": sum(values["reference"]) / len(values["reference"]),
        "training_values": values["training"],
        "reference_values": values["reference"],
    }


def _distribution_summary(
    decoder: ConditionalWAEDecoder,
    images: Tensor,
    labels: Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    decoder.eval()
    result: dict[str, Any] = {}
    for bank_seed in PROJECTION_SEEDS:
        banks = multiscale_projection_banks(bank_seed, device=device, dtype=images.dtype)
        aggregate_rows: list[float] = []
        class_rows: list[list[float]] = []
        for batch_seed, prior_seed in zip(BATCH_SEEDS, PRIOR_SEEDS, strict=True):
            batch_images, batch_labels = _balanced_batch(images, labels, batch_seed)
            prior = _prior(batch_labels, prior_seed, device=device, dtype=images.dtype)
            generated = decoder(prior, batch_labels)
            values = _classwise_multiscale_values(generated, batch_images, batch_labels, banks)
            class_rows.append([float(item.detach()) for item in values])
            aggregate_rows.append(float(torch.stack(values).mean().detach()))
        result[str(bank_seed)] = {
            "aggregate": sum(aggregate_rows) / len(aggregate_rows),
            "classwise": [sum(row[index] for row in class_rows) / len(class_rows) for index in range(10)],
        }
    decoder.train()
    return result


def _save_screen_decoder(path: Path, decoder: ConditionalWAEDecoder, *, seed: int, source_checkpoint: Path) -> None:
    payload = {
        "schema": SOURCE_DECODER_SCHEMA,
        "route_schema": DECODER_SCHEMA,
        "candidate": "O0-I",
        "seed": seed,
        "epoch": 1,
        "decoder_state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        "decoder_content_hash": _state_hash(decoder),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "contains_encoder": False,
        "contains_optimizer": False,
    }
    atomic_torch_save(path, payload)


def _screen_train(
    args: argparse.Namespace,
    *,
    output: Path,
    resume: str | None,
) -> tuple[ConditionalWAEDecoder, dict[str, Any], Path, Path]:
    device = torch.device(args.device)
    source_hash = file_sha256(args.wae_decoder_checkpoint)
    images, labels, schedule = _training_data(args.dataset_root, seed=SCREEN_SEED)
    schedule_hash = _schedule_hash(schedule)
    objective_hash = _objective_hash(schedule_hash)
    torch.manual_seed(SCREEN_SEED)
    torch.cuda.manual_seed_all(SCREEN_SEED)
    decoder = _load_source_decoder(args.wae_decoder_checkpoint, device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE)
    history: list[dict[str, Any]] = []
    history_window: list[dict[str, Any]] = []
    step = 0
    if resume:
        payload = safe_torch_load(resume, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema") != TRAINING_SCHEMA:
            raise ValueError("H-DIDM resume checkpoint schema mismatch")
        if payload.get("seed") != SCREEN_SEED or payload.get("source_decoder_sha256") != source_hash:
            raise ValueError("H-DIDM resume checkpoint lineage mismatch")
        if payload.get("schedule_hash") != schedule_hash or payload.get("objective_hash") != objective_hash:
            raise ValueError("H-DIDM resume checkpoint schedule/objective mismatch")
        decoder.load_state_dict(payload["decoder"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng_state(payload["rng_state"])
        step = int(payload["step"])
        history = list(payload.get("history", []))
        history_window = list(payload.get("history_window", []))
    train_images = images.to(device)
    train_labels = labels.to(device)
    banks = multiscale_projection_banks(PROJECTION_SEEDS[0], device=device, dtype=train_images.dtype)
    progress_path = output.parent / f"seed_{SCREEN_SEED}_progress.pt"
    for cursor in range(step, EXPECTED_STEPS):
        batch_images = train_images.index_select(0, schedule[cursor].to(device))
        batch_labels = train_labels.index_select(0, schedule[cursor].to(device))
        update = _update_step(decoder, optimizer, batch_images, batch_labels, PRIOR_SEEDS[cursor % len(PRIOR_SEEDS)], banks)
        update["step"] = cursor + 1
        history_window.append(update)
        step = cursor + 1
        if step % HISTORY_INTERVAL == 0 or step == EXPECTED_STEPS:
            observer = _observer_summary(decoder, train_images, train_labels, device=device)
            history.append({"step": step, **observer})
            history_window = [] if step % HISTORY_INTERVAL == 0 else history_window
            atomic_torch_save(progress_path, _training_checkpoint(decoder, optimizer, seed=SCREEN_SEED, step=step, batch_cursor=step, schedule_hash=schedule_hash, objective_hash=objective_hash, history=history, history_window=history_window, source_hash=source_hash))
    training_path = output.parent / f"seed_{SCREEN_SEED}_training.pt"
    decoder_path = output.parent / f"seed_{SCREEN_SEED}_decoder.pt"
    atomic_torch_save(training_path, _training_checkpoint(decoder, optimizer, seed=SCREEN_SEED, step=step, batch_cursor=step, schedule_hash=schedule_hash, objective_hash=objective_hash, history=history, history_window=history_window, source_hash=source_hash))
    _save_screen_decoder(decoder_path, decoder, seed=SCREEN_SEED, source_checkpoint=training_path)
    return decoder, {"step": step, "history": history, "schedule_hash": schedule_hash, "objective_hash": objective_hash, "training_checkpoint": str(training_path), "decoder_checkpoint": str(decoder_path), "source_decoder_sha256": source_hash, "evaluator_calls_during_training": 0}, training_path, decoder_path


def _screen_lineage(args: argparse.Namespace, config: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    lineage = _lineage(args, config)
    qualification_paths = (args.primary_qualification_evidence, args.secondary_qualification_evidence)
    qualifications = [_load_json(path) for path in qualification_paths]
    if any(payload.get("status") != "passed" for payload in qualifications):
        raise ValueError("H-DIDM screen evaluator qualification is not passed")
    lineage["inputs"].update({
        "primary_evaluator": file_sha256(args.primary_evaluator_checkpoint),
        "secondary_evaluator": file_sha256(args.secondary_evaluator_checkpoint),
        "primary_qualification": file_sha256(args.primary_qualification_evidence),
        "secondary_qualification": file_sha256(args.secondary_qualification_evidence),
        "preflight": file_sha256(args.preflight_evidence),
    })
    lineage["preflight_sha256"] = file_sha256(args.preflight_evidence)
    return lineage


def _screen_measurement(args: argparse.Namespace, output: Path, preflight: dict[str, Any], lineage: dict[str, Any]) -> dict[str, Any]:
    _require_cuda(args)
    torch.cuda.reset_peak_memory_stats(args.device)
    decoder, training, training_path, decoder_path = _screen_train(args, output=output, resume=getattr(args, "resume_checkpoint", None))
    device = torch.device(args.device)
    images, labels, _ = _training_data(args.dataset_root, seed=SCREEN_SEED)
    source = _load_source_decoder(args.wae_decoder_checkpoint, device)
    source_distribution = _distribution_summary(source, images.to(device), labels.to(device), device=device)
    candidate_distribution = _distribution_summary(decoder, images.to(device), labels.to(device), device=device)
    distribution_gates = {}
    for bank_seed in PROJECTION_SEEDS:
        source_row = source_distribution[str(bank_seed)]
        candidate_row = candidate_distribution[str(bank_seed)]
        distribution_gates[str(bank_seed)] = {
            "aggregate_lower": candidate_row["aggregate"] < source_row["aggregate"],
            "all_classes_not_higher": all(candidate <= source + 1.0e-8 for candidate, source in zip(candidate_row["classwise"], source_row["classwise"], strict=True)),
        }
    evaluation = _measure_checkpoint(
        decoder_path,
        candidate="O0-I",
        device=device,
        dataset_root=args.dataset_root,
        primary_path=args.primary_evaluator_checkpoint,
        secondary_path=args.secondary_evaluator_checkpoint,
    )
    memory_pass = evaluation.get("memorization", {}).get("passed") is True
    jacobian_pass = evaluation.get("jacobian", {}).get("rank_1e-3_pass_fraction", 0.0) >= 0.80
    distribution_pass = all(row["aggregate_lower"] and row["all_classes_not_higher"] for row in distribution_gates.values())
    scientific_pass = bool(distribution_pass and evaluation.get("passed") and memory_pass and jacobian_pass)
    peak = int(torch.cuda.max_memory_allocated(args.device))
    storage = _storage_bytes(Path(args._run_manifest["output_root"]))
    finite = all(torch.isfinite(value).all() for value in decoder.state_dict().values())
    measurement_valid = bool(finite and training["evaluator_calls_during_training"] == 0 and peak < 6 * 1024**3 and storage <= 2 * 1024**3)
    classification, next_action = _screen_classification(measurement_valid, scientific_pass)
    return {
        "status": "valid" if measurement_valid else "invalid_harness",
        "measurement_valid": measurement_valid,
        "passed": bool(measurement_valid and scientific_pass),
        "classification": classification,
        "next_action": next_action,
        "training": training,
        "source_distribution": source_distribution,
        "candidate_distribution": candidate_distribution,
        "distribution_gates": distribution_gates,
        "evaluation": evaluation,
        "memorization_pass": memory_pass,
        "jacobian_pass": jacobian_pass,
        "evaluator_calls_during_training": 0,
        "peak_memory_bytes": peak,
        "storage_bytes": storage,
        "memory_gate": peak < 6 * 1024**3,
        "storage_gate": storage <= 2 * 1024**3,
        "training_checkpoint": str(training_path),
        "decoder_checkpoint": str(decoder_path),
        "training_checkpoint_sha256": file_sha256(training_path),
        "decoder_checkpoint_sha256": file_sha256(decoder_path),
        "lineage_sha256": canonical_sha256(lineage),
        "preflight_sha256": file_sha256(args.preflight_evidence),
        "preflight_passed": preflight.get("passed") is True,
    }


def _screen_classification(measurement_valid: bool, scientific_pass: bool) -> tuple[str, str]:
    if not measurement_valid:
        return "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit"
    if scientific_pass:
        return "direct_image_distribution_admitted", "register_direct_image_distribution_confirm"
    return "direct_image_distribution_failed", "close_current_direct_image_distribution_route"


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
        and measurement.get("controls", {}).get("passed")
        and measurement.get("formula_and_chunk_controls", {}).get("value_matches_naive")
        and measurement.get("formula_and_chunk_controls", {}).get("gradient_matches_naive")
        and measurement.get("formula_and_chunk_controls", {}).get("chunk_matches_full")
        and measurement.get("formula_and_chunk_controls", {}).get("finite")
        and measurement.get("projection_determinism", {}).get("training_repeat")
        and measurement.get("projection_determinism", {}).get("reference_repeat")
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
    fields = common + ("device", "validation_evidence", "audit_evidence", "preflight_evidence", "measurement_evidence", "resume_checkpoint", "primary_evaluator_checkpoint", "secondary_evaluator_checkpoint", "primary_qualification_evidence", "secondary_qualification_evidence")
    args = prepare_args(args, program=PROGRAM, action=action, required=required[action], defaults={"device": "cuda", "seed": HARNESS_SEED}, mixed_fields=fields)
    validate_manifest_namespace(args, action=action)
    config = _config(args.config)
    lineage = _lineage(args, config) if action in {"validate", "audit", "preflight"} else None
    if action == "validate":
        decision = _stage(args, "validate", {"config": file_sha256(args.config), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
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
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        train_images, train_labels, _ = _training_data(args.dataset_root, seed=SCREEN_SEED)
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
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        train_images, train_labels, _ = _training_data(args.dataset_root, seed=SCREEN_SEED)
        decoder = _load_source_decoder(args.wae_decoder_checkpoint, torch.device(args.device))
        measurement = _preflight_measurement(
            decoder,
            train_images.to(args.device),
            train_labels.to(args.device),
            source_hash=file_sha256(args.wae_decoder_checkpoint),
        )
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
        lineage = _screen_lineage(args, config, preflight)
        decision = _stage(args, "screen", {
            "preflight": file_sha256(args.preflight_evidence),
            "lineage_sha256": canonical_sha256(lineage),
            "resume": file_sha256(args.resume_checkpoint) if getattr(args, "resume_checkpoint", None) else None,
        })
        if decision["action"] == "REUSE":
            return 0
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        measurement = _screen_measurement(args, Path(args.output), preflight, lineage)
        _write(args.output, {
            "schema": f"{SCHEMA_PREFIX}_screen.v1",
            "stage": "screen",
            **measurement,
            "producer_contract_revision": f"{SCHEMA_PREFIX}-screen.v1",
            "consumer_contract": f"{SCHEMA_PREFIX}-report.v1",
            "permitted_consumers": ["report"] if measurement["measurement_valid"] else [],
            "input_artifact_hashes": {
                "preflight": file_sha256(args.preflight_evidence),
                "lineage": canonical_sha256(lineage),
            },
        }, args)
        return 0
    measurement = _check_consumer(args.measurement_evidence, action)
    decision = _stage(args, action, {"measurement": file_sha256(args.measurement_evidence)})
    if decision["action"] == "REUSE":
        return 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": f"{SCHEMA_PREFIX}_{action}.v1", "stage": action, "status": "valid", "passed": measurement.get("passed"), "measurement_valid": measurement.get("measurement_valid"), "classification": measurement.get("classification"), "next_action": measurement.get("next_action"), "measurement": args.measurement_evidence, "source_evidence_sha256": file_sha256(args.measurement_evidence), "input_artifact_hashes": {"measurement": file_sha256(args.measurement_evidence)}, "permitted_consumers": ["closeout"] if action == "report" else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-{action}.v1"}
    _write(args.output, result, args)
    return 0
