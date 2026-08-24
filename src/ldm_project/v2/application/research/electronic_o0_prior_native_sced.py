"""Prior-native electronic O0 lifecycle using a finite-batch-corrected objective.

This route is deliberately independent from the closed H-SWAE dispatcher.  It
keeps the validated WAE and the O0-I model boundary, but replaces the biased
fixed-quantile objective with a classwise sliced Gaussian energy U-statistic.
Evaluators are only loaded by confirm/locked-final consumers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn
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
from .electronic_o0_wae import (
    LOCKED_SEEDS,
    _jacobian_summary,
    _memorization_audit,
    _measure_checkpoint,
    _clone_state,
    _optimizer_state_equal,
    _states_equal,
    balanced_epoch_schedule,
    balanced_schedule_summary,
    panel_codes,
)
from .lifecycle_artifacts import (
    prepare_args,
    stage_action_decision,
    validate_manifest_namespace,
)
from .wae_models import (
    FIXED_EXPOSURE_SCALE,
    ConditionalWAEEncoder,
    ConditionalWAEDecoder,
    parameter_counts,
    render_view,
    sobel_l1,
)


SCHEMA_PREFIX = "electronic_o0_prior_native_sced"
PROGRAM = "electronic-o0-prior-native-sced"
TRAIN_SEEDS = (8601, 8602, 8603)
BATCH_SEEDS = (11, 23, 37, 51)
PROJECTION_SEED = 271829
REFERENCE_PROJECTION_SEED = 314159
BOOTSTRAP_SEED = 90210
LOCKED_PANEL_SEEDS = LOCKED_SEEDS
LEARNING_RATE = 2.0e-4
SCREEN_EPOCHS = 1
CONFIRM_EPOCHS = 3
CODE_DIM = 8
PROJECTION_COUNT = 256
AUDIT_REPLICATIONS = 512
SOURCE_TRAIN_SCHEMA = "electronic_o0_wae_training.v2"
SOURCE_DECODER_SCHEMA = "electronic_o0_wae_decoder.v2"
OBJECTIVE_REVISION = "gaussian_sliced_energy_u_v1"


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
    if not isinstance(payload, dict):
        raise ValueError("prior-native SCED config must be a mapping")
    required = {
        "schema_version": 1,
        "code_dim": CODE_DIM,
        "logical_batch_size": 100,
        "projection_count": PROJECTION_COUNT,
        "projection_seed": PROJECTION_SEED,
        "reference_projection_seed": REFERENCE_PROJECTION_SEED,
        "learning_rate": LEARNING_RATE,
        "screen_epochs": SCREEN_EPOCHS,
        "confirm_epochs": CONFIRM_EPOCHS,
        "locked_panel_seeds": list(LOCKED_PANEL_SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "harness_revision": 1,
        "objective_revision": OBJECTIVE_REVISION,
        "w_sced_gradient_ratio": 0.50,
        "w_cycle_gradient_ratio": 0.25,
        "screen_swd_reduction": 0.50,
        "screen_reconstruction_tolerance": 1.05,
        "parameter_budget": 8_000_000,
        "decoder_parameter_budget": 6_000_000,
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("prior-native SCED config is not the registered v1 contract")
    if tuple(payload.get("train_seeds", ())) != TRAIN_SEEDS:
        raise ValueError("prior-native SCED seeds are fixed to 8601/8602/8603")
    if payload.get("fixed_exposure_scale") != FIXED_EXPOSURE_SCALE:
        raise ValueError("prior-native SCED fixed exposure is invalid")
    return payload


def _require_cuda(args: argparse.Namespace) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("prior-native SCED requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("prior-native SCED formal lifecycle requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence is not a JSON object: {path}")
    return payload


def _write(path: str | Path, payload: dict[str, Any], args: argparse.Namespace) -> None:
    if payload.get("lineage_sha256"):
        payload.setdefault("input_artifact_hashes", {})["lineage"] = payload[
            "lineage_sha256"
        ]
    payload["invocation_mode"] = "manifest"
    payload["legacy_diagnostic_only"] = False
    payload["run_manifest_sha256"] = file_sha256(args.run_manifest)
    atomic_write_json(path, payload)


def _load_source_models(
    training_path: str | Path,
    decoder_path: str | Path,
    device: torch.device,
) -> tuple[ConditionalWAEEncoder, ConditionalWAEDecoder]:
    training = safe_torch_load(training_path, map_location="cpu")
    decoder_payload = safe_torch_load(decoder_path, map_location="cpu")
    if not isinstance(training, dict) or training.get("schema") != SOURCE_TRAIN_SCHEMA:
        raise ValueError("source WAE training checkpoint schema is invalid")
    if (
        not isinstance(decoder_payload, dict)
        or decoder_payload.get("schema") != SOURCE_DECODER_SCHEMA
        or decoder_payload.get("candidate") != "O0-I"
        or decoder_payload.get("contains_encoder") is not False
        or decoder_payload.get("contains_optimizer") is not False
    ):
        raise ValueError("source WAE decoder checkpoint is not inference-only")
    encoder = ConditionalWAEEncoder().to(device)
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    counts = parameter_counts(encoder, decoder)
    if counts["total"] > 8_000_000 or counts["decoder"] > 6_000_000:
        raise ValueError("WAE parameter budget exceeded")
    encoder.load_state_dict(training["encoder"], strict=True)
    decoder.load_state_dict(training["decoder"], strict=True)
    reference = ConditionalWAEDecoder("O0-I")
    reference.load_state_dict(decoder_payload["decoder_state_dict"], strict=True)
    source_hash = canonical_sha256(
        {key: tensor_sha256(value.detach().cpu()) for key, value in decoder.state_dict().items()}
    )
    reference_hash = canonical_sha256(
        {key: tensor_sha256(value.detach().cpu()) for key, value in reference.state_dict().items()}
    )
    if source_hash != reference_hash:
        raise ValueError("source training and decoder checkpoints disagree")
    return encoder, decoder


def _model_parameter_hash(
    encoder: nn.Module, decoder: nn.Module
) -> str:
    return canonical_sha256(
        {
            "encoder": {
                key: tensor_sha256(value.detach().cpu())
                for key, value in encoder.state_dict().items()
            },
            "decoder": {
                key: tensor_sha256(value.detach().cpu())
                for key, value in decoder.state_dict().items()
            },
        }
    )


def _source_contract(args: argparse.Namespace) -> dict[str, Any]:
    checks = (
        (args.wae_confirm_evidence, "electronic_o0_wae_confirm.v2", "wae_image_manifold_failed"),
        (args.latent_transport_confirm_evidence, "electronic_o0_latent_transport_confirm.v1", "latent_flow_semantic_failed"),
        (args.mixture_closeout, "electronic_o0_mixture_transport_closeout.v1", "mixture_latent_fit_failed"),
        (args.h_swae_closeout, "electronic_o0_prior_native_swae_closeout.v1", "prior_alignment_fit_failed"),
    )
    result: dict[str, Any] = {}
    for path, schema, classification in checks:
        payload = _load_json(path)
        if payload.get("schema") != schema or payload.get("status") != "valid":
            raise ValueError(f"upstream evidence schema/status is invalid: {path}")
        if payload.get("classification") != classification:
            raise ValueError(f"upstream evidence classification is invalid: {path}")
        result[Path(path).stem] = {
            "sha256": file_sha256(path),
            "classification": classification,
        }
    wae = _load_json(args.wae_confirm_evidence)
    measurement = wae.get("measurement", {})
    if measurement.get("checkpoint_sha256") != file_sha256(args.wae_decoder_checkpoint):
        raise ValueError("WAE confirm evidence does not bind the source decoder")
    result["source_training_sha256"] = file_sha256(args.wae_training_checkpoint)
    result["source_decoder_sha256"] = file_sha256(args.wae_decoder_checkpoint)
    return result


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    files = [
        args.config,
        args.wae_training_checkpoint,
        args.wae_decoder_checkpoint,
        args.wae_confirm_evidence,
        args.latent_transport_confirm_evidence,
        args.mixture_closeout,
        args.h_swae_closeout,
        args.primary_evaluator_checkpoint,
        args.secondary_evaluator_checkpoint,
        args.primary_qualification_evidence,
        args.secondary_qualification_evidence,
    ]
    missing = [str(path) for path in files if not Path(path).is_file()]
    if missing or not Path(args.dataset_root).is_dir():
        raise FileNotFoundError("SCED lineage input is missing: " + ", ".join(missing))
    git = _git()
    if not git["clean"]:
        raise RuntimeError("prior-native SCED evidence requires a clean worktree")
    return {
        "schema": f"{SCHEMA_PREFIX}_lineage.v1",
        "config_sha256": file_sha256(args.config),
        "config_content": config,
        "source": _source_contract(args),
        "primary_evaluator_sha256": file_sha256(args.primary_evaluator_checkpoint),
        "secondary_evaluator_sha256": file_sha256(args.secondary_evaluator_checkpoint),
        "primary_qualification_sha256": file_sha256(args.primary_qualification_evidence),
        "secondary_qualification_sha256": file_sha256(args.secondary_qualification_evidence),
        "git": git,
        "device": args.device,
        "seeds": {
            "train": list(TRAIN_SEEDS),
            "projection": PROJECTION_SEED,
            "reference_projection": REFERENCE_PROJECTION_SEED,
            "locked_panels": list(LOCKED_PANEL_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
        },
    }


def fixed_projections(
    *, seed: int = PROJECTION_SEED, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn((PROJECTION_COUNT, CODE_DIM), generator=generator, dtype=dtype)
    return F.normalize(values, dim=1).to(device)


def gaussian_absolute_moment(values: torch.Tensor) -> torch.Tensor:
    """Return E|x-Z| for Z~N(0,1), evaluated elementwise."""
    phi = torch.exp(-0.5 * values.square()) / math.sqrt(2.0 * math.pi)
    cdf = torch.special.ndtr(values)
    return 2.0 * phi + values * (2.0 * cdf - 1.0)


def _ordered_pairwise_absolute_mean(projected: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Exact off-diagonal mean without materializing a pairwise tensor."""
    count = projected.shape[dim]
    if count < 2:
        raise ValueError("SCED requires at least two samples per class")
    ordered = torch.sort(projected, dim=dim).values
    coefficients = 2 * torch.arange(
        count, device=projected.device, dtype=projected.dtype
    ) - (count - 1)
    shape = [1] * projected.ndim
    shape[dim] = count
    unordered_sum = (ordered * coefficients.reshape(shape)).sum(dim=dim)
    return 2.0 * unordered_sum / (count * (count - 1))


def _validate_codes(codes: torch.Tensor, labels: torch.Tensor, projections: torch.Tensor) -> None:
    if codes.ndim != 2 or codes.shape[1] != CODE_DIM:
        raise ValueError("SCED codes must have shape [N, 8]")
    if labels.shape != (codes.shape[0],) or labels.dtype not in (torch.int64, torch.int32):
        raise ValueError("SCED labels have an invalid shape or dtype")
    if projections.shape != (PROJECTION_COUNT, CODE_DIM):
        raise ValueError("SCED projection bank shape is invalid")
    if labels.device != codes.device or projections.device != codes.device:
        raise ValueError("SCED codes, labels, and projections must share a device")
    if projections.dtype != codes.dtype or not codes.is_floating_point():
        raise ValueError("SCED codes and projections must share a floating dtype")
    if labels.numel() == 0 or int(labels.min()) < 0 or int(labels.max()) > 9:
        raise ValueError("SCED labels must be the ten MNIST classes 0..9")
    if not torch.isfinite(codes).all() or not torch.isfinite(projections).all():
        raise FloatingPointError("SCED inputs are non-finite")


def _classwise_sced_values(
    codes: torch.Tensor, labels: torch.Tensor, *, projections: torch.Tensor
) -> list[torch.Tensor]:
    _validate_codes(codes, labels, projections)
    values: list[torch.Tensor] = []
    gaussian_constant = 2.0 / math.sqrt(math.pi)
    for label in range(10):
        selected = codes[labels == label]
        if selected.shape[0] < 2:
            raise ValueError("SCED requires at least two samples per class")
        projected = selected @ projections.t()
        cross = 2.0 * gaussian_absolute_moment(projected).mean(0)
        within = _ordered_pairwise_absolute_mean(projected, dim=0)
        values.append((cross - within - gaussian_constant).mean())
    return values


def classwise_sliced_energy_gaussian(
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    projections: torch.Tensor | None = None,
) -> torch.Tensor:
    if projections is None:
        projections = fixed_projections(device=codes.device, dtype=codes.dtype)
    return torch.stack(_classwise_sced_values(codes, labels, projections=projections)).mean()


def _gaussian_quantiles(count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if count < 2:
        raise ValueError("quantile reference requires at least two samples")
    probabilities = (torch.arange(count, device=device, dtype=dtype) + 0.5) / count
    return torch.special.ndtri(probabilities).reshape(count, 1)


def _reference_swd_values(
    codes: torch.Tensor, labels: torch.Tensor, *, projections: torch.Tensor
) -> list[torch.Tensor]:
    _validate_codes(codes, labels, projections)
    values = []
    for label in range(10):
        selected = codes[labels == label]
        if selected.shape[0] < 2:
            raise ValueError("reference SWD requires at least two samples per class")
        projected = torch.sort(selected @ projections.t(), dim=0).values
        target = _gaussian_quantiles(selected.shape[0], device=codes.device, dtype=codes.dtype)
        values.append((projected - target).square().mean())
    return values


def _reference_swd(codes: torch.Tensor, labels: torch.Tensor, projections: torch.Tensor) -> torch.Tensor:
    return torch.stack(_reference_swd_values(codes, labels, projections=projections)).mean()


def _synthetic_scale_losses(
    projected: torch.Tensor, scale: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SCED and n=10 quantile-SWD means over audit replications/classes."""
    if projected.ndim != 4 or projected.shape[1:3] != (10, 10):
        raise ValueError("synthetic audit projections must have shape [R, 10, 10, P]")
    scaled = projected * scale
    cross = 2.0 * gaussian_absolute_moment(scaled).mean(dim=2)
    within = _ordered_pairwise_absolute_mean(scaled, dim=2)
    sced = (cross - within - 2.0 / math.sqrt(math.pi)).mean()
    target = _gaussian_quantiles(
        10, device=scaled.device, dtype=scaled.dtype
    ).reshape(1, 1, 10, 1)
    swd = (torch.sort(scaled, dim=2).values - target).square().mean()
    return sced, swd


def _balanced_batch(
    images: torch.Tensor, labels: torch.Tensor, *, seed: int, per_class: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for label in range(10):
        choices = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
        if choices.numel() < per_class:
            raise ValueError("balanced batch does not contain enough examples")
        rows.append(choices[torch.randperm(choices.numel(), generator=generator)[:per_class]])
    index = torch.cat(rows).to(images.device)
    return images.index_select(0, index), labels.index_select(0, index)


def _terms(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor,
    *,
    w_sced: float,
    w_cycle: float,
    projections: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    encoded = encoder(images, labels)
    prediction = render_view(decoder(encoded, labels), "O0-I")
    reconstruction = F.binary_cross_entropy(prediction, images) + 0.25 * sobel_l1(prediction, images)
    sced = classwise_sliced_energy_gaussian(encoded, labels, projections=projections)
    cycle_prediction = render_view(decoder(prior, labels), "O0-I")
    cycle = F.smooth_l1_loss(encoder(cycle_prediction, labels), prior)
    total = reconstruction + float(w_sced) * sced + float(w_cycle) * cycle
    return total, {"reconstruction": reconstruction, "sced": sced, "cycle": cycle, "encoded": encoded}


def _calibrate_weights(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    params = tuple(encoder.parameters()) + tuple(decoder.parameters())
    projections = fixed_projections(device=images.device, dtype=images.dtype)
    rows = []
    for seed in BATCH_SEEDS:
        batch_images, batch_labels = _balanced_batch(images, labels, seed=seed)
        prior = torch.randn((batch_images.shape[0], CODE_DIM), generator=torch.Generator(device="cpu").manual_seed(seed + 1000)).to(images.device)
        encoded = encoder(batch_images, batch_labels)
        prediction = render_view(decoder(encoded, batch_labels), "O0-I")
        reconstruction = F.binary_cross_entropy(prediction, batch_images) + 0.25 * sobel_l1(prediction, batch_images)
        sced = classwise_sliced_energy_gaussian(encoded, batch_labels, projections=projections)
        cycle_prediction = render_view(decoder(prior, batch_labels), "O0-I")
        cycle = F.smooth_l1_loss(encoder(cycle_prediction, batch_labels), prior)
        norms = []
        for term in (reconstruction, sced, cycle):
            gradients = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            norms.append(float(torch.sqrt(sum((g.square().sum() for g in gradients if g is not None), torch.zeros((), device=images.device))).detach().cpu()))
        rows.append(norms)
    mean = torch.tensor(rows, dtype=torch.float64)
    if not torch.isfinite(mean).all() or float(mean[:, 0].mean()) <= 0:
        raise FloatingPointError("SCED gradient calibration is non-finite")
    reconstruction, sced, cycle = mean.mean(0)
    return {
        "batch_seeds": [11, 23, 37, 51],
        "w_sced": float(0.50 * reconstruction / sced.clamp_min(1.0e-12)),
        "w_cycle": float(0.25 * reconstruction / cycle.clamp_min(1.0e-12)),
        "gradient_norms": {"reconstruction": float(reconstruction), "sced": float(sced), "cycle": float(cycle)},
    }


def _objective_hash(
    weights: dict[str, Any], *, schedule_hash: str | None = None
) -> str:
    return canonical_sha256(
        {
            "weights": weights,
            "objective_revision": OBJECTIVE_REVISION,
            "projection_seed": PROJECTION_SEED,
            "projection_count": PROJECTION_COUNT,
            "reference_projection_seed": REFERENCE_PROJECTION_SEED,
            "cycle_ratio": 0.25,
            "reconstruction": "bce_plus_0.25_sobel_l1",
            "learning_rate": LEARNING_RATE,
            "schedule": {
                "batch_seeds": list(BATCH_SEEDS),
                "logical_batch_size": 100,
                "per_class": 10,
                "sampler": "balanced_epoch_schedule",
                "hash": schedule_hash,
            },
        }
    )


def _schedule_hash(schedule: tuple[torch.Tensor, ...]) -> str:
    return canonical_sha256([tensor_sha256(item.detach().cpu()) for item in schedule])


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _cuda_rng_states_equal(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> bool:
    return len(left) == len(right) and all(
        torch.equal(left_state, right_state)
        for left_state, right_state in zip(left, right, strict=True)
    )


def _optimizer_state_finite(state: dict[str, Any]) -> bool:
    return all(
        torch.isfinite(value).all()
        for values in state.get("state", {}).values()
        for value in values.values()
        if torch.is_tensor(value)
    )


def _cpu_clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_tree(item) for item in value)
    return value


def _optimizer(device: torch.device, encoder: nn.Module, decoder: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(tuple(encoder.parameters()) + tuple(decoder.parameters()), lr=LEARNING_RATE)


def _snapshot(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    epoch: int,
    batch_cursor: int,
    step: int,
    weights: dict[str, Any],
    schedule: tuple[torch.Tensor, ...],
    schedule_summary: dict[str, Any],
    history: list[dict[str, float]],
    history_window: list[dict[str, float]],
    source_hash: str,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_training.v1",
        "seed": seed,
        "epoch": epoch,
        "batch_cursor": batch_cursor,
        "step": step,
        "encoder": _cpu_clone_tree(encoder.state_dict()),
        "decoder": _cpu_clone_tree(decoder.state_dict()),
        "optimizer": _cpu_clone_tree(optimizer.state_dict()),
        "weights": weights,
        "schedule": dict(schedule_summary),
        "schedule_hash": _schedule_hash(schedule),
        "sampler_state": {"seed": seed, "per_class": 10, "cursor": batch_cursor},
        "history": list(history),
        "history_window": list(history_window),
        "objective_hash": _objective_hash(weights, schedule_hash=_schedule_hash(schedule)),
        "source_training_checkpoint_sha256": source_hash,
        "rng_state": {
            "cpu": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
        },
    }


def _restore(
    payload: dict[str, Any],
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    encoder.load_state_dict(payload["encoder"], strict=True)
    decoder.load_state_dict(payload["decoder"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    torch.set_rng_state(payload["rng_state"]["cpu"])
    torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])


def _load_training(path: str | Path, device: torch.device):
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != f"{SCHEMA_PREFIX}_training.v1":
        raise ValueError("SCED training checkpoint schema is invalid")
    encoder = ConditionalWAEEncoder().to(device)
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    optimizer = _optimizer(device, encoder, decoder)
    _restore(payload, encoder, decoder, optimizer, device)
    return encoder, decoder, optimizer, payload


def _update(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    weights: dict[str, Any],
    projections: torch.Tensor,
) -> dict[str, float]:
    prior = torch.randn((images.shape[0], CODE_DIM), device=images.device)
    optimizer.zero_grad(set_to_none=True)
    total, terms = _terms(encoder, decoder, images, labels, prior, w_sced=float(weights["w_sced"]), w_cycle=float(weights["w_cycle"]), projections=projections)
    if not torch.isfinite(total) or not all(torch.isfinite(value) for name, value in terms.items() if name != "encoded"):
        raise FloatingPointError("SCED update loss is non-finite")
    total.backward()
    optimizer.step()
    return {"total": float(total.detach().cpu()), **{name: float(value.detach().cpu()) for name, value in terms.items() if name != "encoded"}}


def _objective_snapshot(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor,
    weights: dict[str, Any],
    projections: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        _, terms = _terms(encoder, decoder, images, labels, prior, w_sced=float(weights["w_sced"]), w_cycle=float(weights["w_cycle"]), projections=projections)
    return {name: float(terms[name].detach().cpu()) for name in ("reconstruction", "sced", "cycle")}


def _save_decoder(path: Path, decoder: ConditionalWAEDecoder, *, seed: int, epoch: int, source_hash: str) -> None:
    payload = {
        "schema": SOURCE_DECODER_SCHEMA,
        "candidate": "O0-I",
        "seed": seed,
        "epoch": epoch,
        "decoder_state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        "decoder_content_hash": canonical_sha256({key: tensor_sha256(value.detach().cpu()) for key, value in decoder.state_dict().items()}),
        "source_checkpoint_sha256": source_hash,
        "contains_encoder": False,
        "contains_optimizer": False,
    }
    atomic_torch_save(path, payload)


def _train(
    args: argparse.Namespace,
    *,
    seed: int,
    target_epoch: int,
    output: Path,
    resume: str | None,
    source_hash: str,
    weights: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    _require_cuda(args)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    all_images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_rows, _ = stratified_mnist_indices(all_labels)
    train_images = all_images.index_select(0, train_rows)
    train_labels = all_labels.index_select(0, train_rows)
    schedule = balanced_epoch_schedule(train_labels, seed=seed, per_class=10)
    schedule_summary = balanced_schedule_summary(train_labels, schedule)
    schedule_hash = _schedule_hash(schedule)
    objective_hash = _objective_hash(weights, schedule_hash=schedule_hash)
    if resume:
        encoder, decoder, optimizer, payload = _load_training(resume, device)
        if payload.get("seed") != seed or payload.get("source_training_checkpoint_sha256") != source_hash:
            raise ValueError("SCED resume checkpoint lineage mismatch")
        if payload.get("schedule_hash") != schedule_hash or payload.get("objective_hash") != objective_hash:
            raise ValueError("SCED resume checkpoint objective/schedule mismatch")
        start_epoch = int(payload["epoch"])
        step = int(payload["step"])
        batch_cursor = int(payload.get("batch_cursor", 0))
        history = list(payload.get("history", []))
        window = list(payload.get("history_window", []))
    else:
        encoder, decoder = _load_source_models(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        optimizer = _optimizer(device, encoder, decoder)
        start_epoch = 0
        step = 0
        batch_cursor = 0
        history = []
        window = []
    projections = fixed_projections(device=device)
    losses: list[dict[str, float]] = []
    progress_path = output.with_name(f"seed_{seed}_progress.pt")
    for epoch in range(start_epoch, target_epoch):
        cursor = batch_cursor if epoch == start_epoch else 0
        for cursor in range(cursor, len(schedule)):
            indices = schedule[cursor]
            batch_images = train_images.index_select(0, indices).to(device)
            batch_labels = train_labels.index_select(0, indices).to(device)
            values = _update(encoder, decoder, optimizer, batch_images, batch_labels, weights, projections)
            losses.append(values)
            window.append(values)
            step += 1
            next_epoch, next_cursor = epoch, cursor + 1
            if next_cursor == len(schedule):
                next_epoch, next_cursor = epoch + 1, 0
            if step % 50 == 0 or next_epoch == target_epoch:
                history.append({name: sum(row[name] for row in window) / len(window) for name in ("total", "reconstruction", "sced", "cycle")})
                window = []
                payload = _snapshot(encoder, decoder, optimizer, seed=seed, epoch=next_epoch, batch_cursor=next_cursor, step=step, weights=weights, schedule=schedule, schedule_summary=schedule_summary, history=history, history_window=window, source_hash=source_hash)
                atomic_torch_save(progress_path, payload)
        batch_cursor = 0
    training_path = output.with_name(f"seed_{seed}_training.pt")
    decoder_path = output.with_name(f"seed_{seed}_decoder.pt")
    payload = _snapshot(encoder, decoder, optimizer, seed=seed, epoch=target_epoch, batch_cursor=0, step=step, weights=weights, schedule=schedule, schedule_summary=schedule_summary, history=history, history_window=window, source_hash=source_hash)
    atomic_torch_save(training_path, payload)
    _save_decoder(decoder_path, decoder, seed=seed, epoch=target_epoch, source_hash=file_sha256(training_path))
    return training_path, decoder_path, {"seed": seed, "epoch": target_epoch, "step": step, "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None, "history": history, "schedule": schedule_summary, "weights": weights}


def _audit(args: argparse.Namespace, lineage: dict[str, Any], output: Path) -> None:
    device = torch.device(args.device)
    _require_cuda(args)
    source_encoder, source_decoder = _load_source_models(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
    source_hash = _model_parameter_hash(source_encoder, source_decoder)
    generator = torch.Generator(device="cpu").manual_seed(271829)
    base = torch.randn(
        (AUDIT_REPLICATIONS, 10, 10, CODE_DIM), generator=generator
    ).to(device)
    projections = fixed_projections(device=device)
    projected = base @ projections.t()
    scales = (0.8, 0.8343666503, 0.9, 1.0, 1.1, 1.2)
    with torch.no_grad():
        losses = {scale: _synthetic_scale_losses(projected, scale) for scale in scales}
    sced_values = {
        str(scale): float(losses[scale][0].detach().cpu()) for scale in scales
    }
    swd_values = {
        str(scale): float(losses[scale][1].detach().cpu()) for scale in scales
    }
    scale = torch.tensor(0.8, device=device, requires_grad=True)
    low = _synthetic_scale_losses(projected, scale)[0]
    low_grad = float(torch.autograd.grad(low, scale)[0].detach().cpu())
    scale = torch.tensor(1.2, device=device, requires_grad=True)
    high = _synthetic_scale_losses(projected, scale)[0]
    high_grad = float(torch.autograd.grad(high, scale)[0].detach().cpu())
    positive = low_grad < 0.0 and high_grad > 0.0
    reverse_direction_control = low_grad > 0.0 and high_grad < 0.0
    source_hash_after = _model_parameter_hash(source_encoder, source_decoder)
    audit_passed = bool(
        positive
        and not reverse_direction_control
        and source_hash == source_hash_after
        and sced_values["1.0"] + 0.001 < sced_values["0.9"]
        and sced_values["1.0"] + 0.001 < sced_values["1.1"]
        and swd_values["0.8343666503"] + 0.02 < swd_values["1.0"]
    )
    payload = {
        "schema": f"{SCHEMA_PREFIX}_audit.v1",
        "stage": "audit",
        "status": "valid" if audit_passed else "invalid_harness",
        "passed": audit_passed,
        "classification": None if audit_passed else "invalid_harness",
        "replications": AUDIT_REPLICATIONS,
        "samples_per_class": 10,
        "scale_grid": {"sced": sced_values, "reference_swd": swd_values},
        "direction_probe": {
            "low_sigma_gradient": low_grad,
            "high_sigma_gradient": high_grad,
            "positive_passed": positive,
            "reverse_direction_control_passed": reverse_direction_control,
        },
        "optimizer_created": False,
        "evaluator_calls": 0,
        "source_parameter_hash_before": source_hash,
        "source_parameter_hash_after": source_hash_after,
        "lineage_sha256": canonical_sha256(lineage),
        "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1",
        "consumer_contract": f"{SCHEMA_PREFIX}-preflight.v1",
        "permitted_consumers": ["preflight"],
    }
    _write(output, payload, args)


def _preflight(args: argparse.Namespace, lineage: dict[str, Any], output: Path, audit: dict[str, Any]) -> None:
    _require_cuda(args)
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    source_hash = file_sha256(args.wae_training_checkpoint)
    all_images, all_labels = load_local_mnist(args.dataset_root, train=True)
    rows, _ = stratified_mnist_indices(all_labels)
    images = all_images.index_select(0, rows)
    labels = all_labels.index_select(0, rows)
    batches = tuple(
        tuple(item.to(device) for item in _balanced_batch(images, labels, seed=seed))
        for seed in BATCH_SEEDS
    )
    batch_images, batch_labels = batches[0]
    weights = _calibrate_weights(*_load_source_models(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device), batch_images, batch_labels)
    projections = fixed_projections(device=device)
    reference_projections = fixed_projections(seed=REFERENCE_PROJECTION_SEED, device=device)
    prior = torch.randn((batch_images.shape[0], CODE_DIM), generator=torch.Generator(device="cpu").manual_seed(1011)).to(device)
    first_encoder, first_decoder = _load_source_models(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
    initial_encoder_state = _clone_state(first_encoder)
    initial_decoder_state = _clone_state(first_decoder)
    first_optimizer = _optimizer(device, first_encoder, first_decoder)
    schedule = tuple(torch.tensor([seed], dtype=torch.int64) for seed in BATCH_SEEDS)
    schedule_summary = {
        "steps": len(schedule),
        "logical_samples": len(schedule),
        "unique_samples": len(schedule),
        "minimum_coverage": 1,
        "maximum_repeats": 1,
        "repeat_samples": 0,
    }
    torch.manual_seed(8601)
    torch.cuda.manual_seed_all(8601)
    initial = _objective_snapshot(first_encoder, first_decoder, batch_images, batch_labels, prior, weights, projections)
    with torch.no_grad():
        initial_reference_swd = float(
            _reference_swd(
                first_encoder(batch_images, batch_labels),
                batch_labels,
                reference_projections,
            ).detach().cpu()
        )
    updates = []
    checkpoint = None
    for step in range(3):
        update_images, update_labels = batches[step]
        updates.append(_update(first_encoder, first_decoder, first_optimizer, update_images, update_labels, weights, projections))
        if step == 1:
            checkpoint = _snapshot(
                first_encoder,
                first_decoder,
                first_optimizer,
                seed=8601,
                epoch=0,
                batch_cursor=2,
                step=2,
                weights=weights,
                schedule=schedule,
                schedule_summary=schedule_summary,
                history=[],
                history_window=list(updates),
                source_hash=source_hash,
            )
    after = _objective_snapshot(first_encoder, first_decoder, batch_images, batch_labels, prior, weights, projections)
    with torch.no_grad():
        after_reference_swd = float(
            _reference_swd(
                first_encoder(batch_images, batch_labels),
                batch_labels,
                reference_projections,
            ).detach().cpu()
        )
    first_cpu_rng_state = torch.get_rng_state().clone()
    first_cuda_rng_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    replay_encoder, replay_decoder = _load_source_models(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
    replay_optimizer = _optimizer(device, replay_encoder, replay_decoder)
    if checkpoint is None:
        raise RuntimeError("SCED preflight checkpoint was not captured")
    _restore(checkpoint, replay_encoder, replay_decoder, replay_optimizer, device)
    replay_update = _update(
        replay_encoder,
        replay_decoder,
        replay_optimizer,
        batches[2][0],
        batches[2][1],
        weights,
        projections,
    )
    replay_exact = _states_equal(_clone_state(first_encoder), _clone_state(replay_encoder)) and _states_equal(_clone_state(first_decoder), _clone_state(replay_decoder))
    optimizer_replay_exact = _optimizer_state_equal(first_optimizer.state_dict(), replay_optimizer.state_dict())
    cpu_rng_replay_exact = torch.equal(first_cpu_rng_state, torch.get_rng_state())
    cuda_rng_replay_exact = _cuda_rng_states_equal(
        first_cuda_rng_states, torch.cuda.get_rng_state_all()
    )
    history_replay_exact = (
        checkpoint.get("history", []) == []
        and [*checkpoint.get("history_window", []), replay_update] == updates
    )
    sampler_replay_exact = (
        checkpoint.get("schedule_hash") == _schedule_hash(schedule)
        and checkpoint.get("sampler_state", {}).get("cursor") == 2
        and checkpoint.get("sampler_state", {}).get("cursor") + 1 == 3
    )
    encoder_changed = not _states_equal(initial_encoder_state, _clone_state(first_encoder))
    decoder_changed = not _states_equal(initial_decoder_state, _clone_state(first_decoder))
    finite_updates = all(
        math.isfinite(value) for update in updates for value in update.values()
    )
    finite_parameters = all(
        torch.isfinite(value).all()
        for value in (*first_encoder.state_dict().values(), *first_decoder.state_dict().values())
    )
    finite_optimizer = _optimizer_state_finite(first_optimizer.state_dict())
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    storage_bytes = _tree_size(Path(args._run_manifest["output_root"]))
    preflight_passed = bool(
        audit.get("status") == "valid"
        and audit.get("passed")
        and finite_updates
        and finite_parameters
        and finite_optimizer
        and encoder_changed
        and decoder_changed
        and replay_exact
        and optimizer_replay_exact
        and cpu_rng_replay_exact
        and cuda_rng_replay_exact
        and history_replay_exact
        and sampler_replay_exact
        and after["sced"] < initial["sced"]
        and after_reference_swd <= initial_reference_swd
        and after["reconstruction"] <= initial["reconstruction"] * 1.05
        and after["cycle"] <= initial["cycle"]
        and peak_memory_bytes < 6 * 1024**3
        and storage_bytes <= 2 * 1024**3
    )
    payload = {
        "schema": f"{SCHEMA_PREFIX}_preflight.v1",
        "stage": "preflight",
        "status": "valid" if preflight_passed else "invalid_harness",
        "passed": preflight_passed,
        "classification": None if preflight_passed else "invalid_harness",
        "updates": updates,
        "weights": weights,
        "initial_objective": initial,
        "after_objective": after,
        "initial_reference_swd": initial_reference_swd,
        "after_reference_swd": after_reference_swd,
        "encoder_changed": encoder_changed,
        "decoder_changed": decoder_changed,
        "finite_updates": finite_updates,
        "finite_parameters": finite_parameters,
        "finite_optimizer": finite_optimizer,
        "interrupted_resumed_replay_exact": replay_exact,
        "optimizer_replay_exact": optimizer_replay_exact,
        "cpu_rng_replay_exact": cpu_rng_replay_exact,
        "cuda_rng_replay_exact": cuda_rng_replay_exact,
        "history_replay_exact": history_replay_exact,
        "sampler_replay_exact": sampler_replay_exact,
        "evaluator_calls": 0,
        "optimizer_created": True,
        "peak_memory_bytes": peak_memory_bytes,
        "storage_bytes": storage_bytes,
        "lineage_sha256": canonical_sha256(lineage),
        "producer_contract_revision": f"{SCHEMA_PREFIX}-preflight.v1",
        "consumer_contract": f"{SCHEMA_PREFIX}-screen.v1",
        "permitted_consumers": ["screen"],
    }
    _write(output, payload, args)


def _screen_metrics(args: argparse.Namespace, training_path: Path, decoder_path: Path, source_encoder: ConditionalWAEEncoder, source_decoder: ConditionalWAEDecoder, device: torch.device) -> dict[str, Any]:
    all_images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_rows, validation_rows = stratified_mnist_indices(all_labels)
    train_images, train_labels = all_images.index_select(0, train_rows), all_labels.index_select(0, train_rows)
    validation_images, validation_labels = all_images.index_select(0, validation_rows), all_labels.index_select(0, validation_rows)
    encoder, decoder, _, _ = _load_training(training_path, device)
    projections = fixed_projections(device=device)
    reference = fixed_projections(seed=REFERENCE_PROJECTION_SEED, device=device)
    images, labels = validation_images.to(device), validation_labels.to(device)
    with torch.no_grad():
        source_codes = source_encoder(images, labels)
        trained_codes = encoder(images, labels)
        prior = torch.randn((labels.shape[0], CODE_DIM), generator=torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)).to(device)
        source_views = render_view(source_decoder(source_codes, labels), "O0-I")
        trained_views = render_view(decoder(trained_codes, labels), "O0-I")
        prior_views = render_view(decoder(prior, labels), "O0-I")
    source_swd = [float(value.detach().cpu()) for value in _reference_swd_values(source_codes, labels, projections=projections)]
    trained_swd = [float(value.detach().cpu()) for value in _reference_swd_values(trained_codes, labels, projections=projections)]
    source_ref = [float(value.detach().cpu()) for value in _reference_swd_values(source_codes, labels, projections=reference)]
    trained_ref = [float(value.detach().cpu()) for value in _reference_swd_values(trained_codes, labels, projections=reference)]
    source_sced = [float(value.detach().cpu()) for value in _classwise_sced_values(source_codes, labels, projections=projections)]
    trained_sced = [float(value.detach().cpu()) for value in _classwise_sced_values(trained_codes, labels, projections=projections)]
    source_reconstruction = float((source_views - images).square().mean().cpu())
    reconstruction = float((trained_views - images).square().mean().cpu())
    source_terms = _objective_snapshot(source_encoder, source_decoder, images, labels, prior, {"w_sced": 0.0, "w_cycle": 0.0}, projections)
    trained_terms = _objective_snapshot(encoder, decoder, images, labels, prior, {"w_sced": 0.0, "w_cycle": 0.0}, projections)
    memory = _memorization_audit(prior_views, validation_labels, train_images, train_labels, validation_images, validation_labels, device=device)
    panel_codes_value, panel_labels = panel_codes(4242, device=device)
    jacobian = _jacobian_summary(decoder, torch.stack([panel_codes_value[label * 50 + offset] for label in range(10) for offset in range(3)]), torch.stack([panel_labels[label * 50 + offset] for label in range(10) for offset in range(3)]), candidate="O0-I")
    storage = sum(item.stat().st_size for item in Path(args._run_manifest["output_root"]).rglob("*") if item.is_file())
    swd = sum(trained_swd) / 10
    source_swd_mean = sum(source_swd) / 10
    ref = sum(trained_ref) / 10
    source_ref_mean = sum(source_ref) / 10
    details = {
        "source_swd": source_swd_mean,
        "swd": swd,
        "swd_relative_reduction": 1.0 - swd / max(source_swd_mean, 1.0e-12),
        "source_reference_swd": source_ref_mean,
        "reference_swd": ref,
        "reference_swd_relative_reduction": 1.0 - ref / max(source_ref_mean, 1.0e-12),
        "source_swd_by_class": source_swd,
        "swd_by_class": trained_swd,
        "source_reference_swd_by_class": source_ref,
        "reference_swd_by_class": trained_ref,
        "source_sced": sum(source_sced) / 10,
        "sced": sum(trained_sced) / 10,
        "source_reconstruction_mse": source_reconstruction,
        "reconstruction_mse": reconstruction,
        "source_reconstruction_objective": source_terms["reconstruction"],
        "reconstruction_objective": trained_terms["reconstruction"],
        "source_cycle": source_terms["cycle"],
        "cycle": trained_terms["cycle"],
        "prior_saturation": float((prior_views >= 1.0).float().mean().cpu()),
        "memorization": memory,
        "jacobian": jacobian,
        "checkpoint_sha256": file_sha256(decoder_path),
        "training_checkpoint_sha256": file_sha256(training_path),
        "storage_bytes": storage,
    }
    details["passed"] = bool(
        details["swd_relative_reduction"] >= 0.50
        and details["reference_swd_relative_reduction"] >= 0.50
        and all(candidate <= source for candidate, source in zip(trained_swd, source_swd, strict=True))
        and all(candidate <= source for candidate, source in zip(trained_ref, source_ref, strict=True))
        and details["sced"] < details["source_sced"]
        and trained_terms["reconstruction"] <= source_terms["reconstruction"] * 1.05
        and trained_terms["cycle"] <= source_terms["cycle"]
        and memory["exact_duplicate_count"] == 0
        and memory["pass_fraction"] >= 0.90
        and details["prior_saturation"] <= 0.08
        and jacobian["rank_1e-3_pass_fraction"] >= 0.80
        and storage <= 2 * 1024**3
    )
    return details


def _apply_complete_o0_gate(measurement: dict[str, Any]) -> dict[str, Any]:
    gates = dict(measurement.get("gates", {}))
    official = measurement.get("official_test", {})
    jacobian = measurement.get("jacobian", {})
    memorization = measurement.get("memorization", {})
    gates.update(
        {
            "official_test_count": official.get("count") == 10_000,
            "official_primary_accuracy": official.get("primary_accuracy", 0.0)
            >= 0.95,
            "official_secondary_accuracy": official.get("secondary_accuracy", 0.0)
            >= 0.95,
            "official_primary_minimum_recall": official.get(
                "primary_minimum_recall", 0.0
            )
            >= 0.90,
            "official_secondary_minimum_recall": official.get(
                "secondary_minimum_recall", 0.0
            )
            >= 0.90,
            "official_agreement": official.get("agreement", 0.0) >= 0.95,
            "official_camera_code_fraction": official.get(
                "camera_code_fraction", 0.0
            )
            >= 0.25,
            "official_saturation": official.get("saturation", 1.0) <= 0.08,
            "jacobian_rank": jacobian.get("rank_1e-3_pass_fraction", 0.0)
            >= 0.80,
            "memorization": memorization.get("pass_fraction", 0.0) >= 0.95
            and memorization.get("exact_duplicate_count") == 0,
        }
    )
    measurement["gates"] = gates
    measurement["passed"] = all(gates.values())
    return measurement


def _stage(args: argparse.Namespace, stage: str, contract: str, expected: dict[str, Any]) -> dict[str, Any]:
    expected = dict(expected)
    lineage_sha256 = getattr(args, "_lineage_sha256", None)
    if lineage_sha256 is not None:
        expected["lineage_sha256"] = lineage_sha256
    decision = stage_action_decision(stage, producer_contract=contract, expected_inputs=expected, evidence=args.output)
    args._stage_decision = decision
    return decision


def _check_consumer(path: str, *, action: str, schemas: tuple[str, ...]) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("status") != "valid" or payload.get("schema") not in schemas:
        raise ValueError(f"{path} is not valid {action} evidence")
    if action not in payload.get("permitted_consumers", ()):
        raise ValueError(f"{path} does not permit consumer {action}")
    if payload.get("invocation_mode") != "manifest" or payload.get("legacy_diagnostic_only") is not False:
        raise ValueError(f"{path} is not manifest evidence")
    return payload


def _anchor_confirm(args: argparse.Namespace, *, action: str) -> dict[str, Any] | None:
    path = getattr(args, "anchor_confirm_evidence", None)
    if args.seed == TRAIN_SEEDS[0]:
        if path:
            raise ValueError("seed-8601 must not declare an anchor confirmation")
        return None
    if not path:
        raise ValueError(f"seed-{args.seed} requires the passed seed-8601 anchor")
    anchor = _check_consumer(
        path, action=action, schemas=(f"{SCHEMA_PREFIX}_confirm.v1",)
    )
    if (
        anchor.get("seed") != TRAIN_SEEDS[0]
        or not anchor.get("passed")
        or anchor.get("classification") != "electronic_prior_native_sced_admitted"
    ):
        raise ValueError("SCED anchor is not an admitted seed-8601 confirmation")
    checkpoint = anchor.get("checkpoint")
    if (
        not checkpoint
        or not Path(checkpoint).is_file()
        or anchor.get("checkpoint_sha256") != file_sha256(checkpoint)
    ):
        raise ValueError("SCED anchor checkpoint is missing or stale")
    return anchor


def _locked_confirmations(
    confirm_path: str, peer_paths: list[str], checkpoint_path: str
) -> tuple[list[dict[str, Any]], list[int]]:
    confirmations = [
        _check_consumer(
            confirm_path,
            action="locked-final",
            schemas=(f"{SCHEMA_PREFIX}_confirm.v1",),
        ),
        *[
            _check_consumer(
                path,
                action="locked-final",
                schemas=(f"{SCHEMA_PREFIX}_confirm.v1",),
            )
            for path in peer_paths
        ],
    ]
    seeds = [item.get("seed") for item in confirmations]
    if (
        len(confirmations) != 3
        or set(seeds) != set(TRAIN_SEEDS)
        or not all(item.get("passed") for item in confirmations)
    ):
        raise ValueError("SCED locked-final requires three passed confirm evidences")
    for item in confirmations:
        checkpoint = item.get("checkpoint")
        if (
            item.get("classification") != "electronic_prior_native_sced_admitted"
            or not checkpoint
            or not Path(checkpoint).is_file()
            or item.get("checkpoint_sha256") != file_sha256(checkpoint)
        ):
            raise ValueError("SCED confirmation checkpoint is missing or stale")
    anchor = next(item for item in confirmations if item["seed"] == TRAIN_SEEDS[0])
    if Path(checkpoint_path).resolve() != Path(anchor["checkpoint"]).resolve():
        raise ValueError("locked-final checkpoint must be the seed-8601 anchor")
    return confirmations, [int(seed) for seed in seeds]


def _closeout(args: argparse.Namespace, measurement_path: str, output: Path) -> int:
    measurement = _load_json(measurement_path)
    passed = bool(measurement.get("status") == "valid" and measurement.get("passed"))
    classification = (
        "electronic_prior_native_sced_admitted"
        if passed
        else measurement.get("classification") or "prior_alignment_fit_failed"
    )
    payload = {
        "schema": f"{SCHEMA_PREFIX}_closeout.v1",
        "stage": "closeout",
        "status": "valid",
        "passed": passed,
        "classification": classification,
        "next_action": "prepare_separate_o0_g_review" if passed else "close_current_prior_native_sced_route",
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
        "measurement": str(measurement_path),
        "source_evidence_sha256": file_sha256(measurement_path),
        "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1",
        "permitted_consumers": [],
    }
    _write(output, payload, args)
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("prior-native SCED formal lifecycle is manifest-only")
    action = args.research_action
    common = (
        "config", "dataset_root", "output", "wae_training_checkpoint", "wae_decoder_checkpoint",
        "wae_confirm_evidence", "latent_transport_confirm_evidence", "mixture_closeout", "h_swae_closeout",
        "primary_evaluator_checkpoint", "secondary_evaluator_checkpoint", "primary_qualification_evidence", "secondary_qualification_evidence",
    )
    required: dict[str, tuple[str, ...]] = {
        "validate": common,
        "audit": common + ("validation_evidence",),
        "preflight": common + ("audit_evidence",),
        "screen": common + ("preflight_evidence",),
        "confirm": common + ("screen_evidence",),
        "locked-final": common + ("confirm_evidence", "checkpoint"),
        "report": ("config", "dataset_root", "output", "measurement_evidence"),
        "closeout": ("config", "dataset_root", "output", "measurement_evidence"),
    }
    fields = common + ("device", "seed", "validation_evidence", "audit_evidence", "preflight_evidence", "screen_evidence", "confirm_evidence", "measurement_evidence", "checkpoint", "resume_checkpoint", "anchor_confirm_evidence", "peer_confirm_evidence")
    defaults = {"device": "cuda", "seed": 8601}
    args = prepare_args(args, program=PROGRAM, action=action, required=required[action], defaults=defaults, mixed_fields=fields)
    validate_manifest_namespace(args, action=action)
    if action in {"validate", "audit", "preflight", "screen", "confirm", "locked-final"}:
        config = _config(args.config)
        lineage = _lineage(args, config)
        args._lineage_sha256 = canonical_sha256(lineage)
    else:
        config, lineage = None, None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if action == "validate":
        decision = _stage(args, "validate", f"{SCHEMA_PREFIX}-validation.v1", {"source_training": file_sha256(args.wae_training_checkpoint), "source_decoder": file_sha256(args.wae_decoder_checkpoint), "h_swae_closeout": file_sha256(args.h_swae_closeout)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        payload = {"schema": f"{SCHEMA_PREFIX}_validation.v1", "stage": "validate", "status": "valid", "passed": True, "lineage": lineage, "lineage_sha256": canonical_sha256(lineage), "producer_contract_revision": f"{SCHEMA_PREFIX}-validation.v1", "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1", "permitted_consumers": ["audit"], "input_artifact_hashes": {"source_training": file_sha256(args.wae_training_checkpoint), "source_decoder": file_sha256(args.wae_decoder_checkpoint), "h_swae_closeout": file_sha256(args.h_swae_closeout)}}
        _write(output, payload, args)
        return 0
    if action == "audit":
        _check_consumer(args.validation_evidence, action="audit", schemas=(f"{SCHEMA_PREFIX}_validation.v1",))
        decision = _stage(args, "audit", f"{SCHEMA_PREFIX}-audit.v1", {"validation": file_sha256(args.validation_evidence), "source_training": file_sha256(args.wae_training_checkpoint)})
        if decision["action"] == "REUSE":
            return 0
        _audit(args, lineage, output)
        return 0
    if action == "preflight":
        audit = _check_consumer(args.audit_evidence, action="preflight", schemas=(f"{SCHEMA_PREFIX}_audit.v1",))
        if not audit.get("passed"):
            raise ValueError("SCED preflight requires passed audit")
        decision = _stage(args, "preflight", f"{SCHEMA_PREFIX}-preflight.v1", {"audit": file_sha256(args.audit_evidence), "source_training": file_sha256(args.wae_training_checkpoint)})
        if decision["action"] == "REUSE":
            return 0
        _preflight(args, lineage, output, audit)
        return 0
    if action == "screen":
        preflight = _check_consumer(args.preflight_evidence, action="screen", schemas=(f"{SCHEMA_PREFIX}_preflight.v1",))
        if not preflight.get("passed"):
            raise ValueError("SCED screen requires passed preflight")
        anchor = _anchor_confirm(args, action="screen")
        anchor_hash = (
            file_sha256(args.anchor_confirm_evidence) if anchor is not None else None
        )
        decision = _stage(args, "screen", f"{SCHEMA_PREFIX}-screen.v1", {"preflight": file_sha256(args.preflight_evidence), "source_training": file_sha256(args.wae_training_checkpoint), "seed": args.seed, "anchor": anchor_hash})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        device = torch.device(args.device)
        source_encoder, source_decoder = _load_source_models(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        weights = preflight["weights"]
        train_path, decoder_path, training = _train(args, seed=args.seed, target_epoch=SCREEN_EPOCHS, output=output, resume=args.resume_checkpoint, source_hash=file_sha256(args.wae_training_checkpoint), weights=weights)
        details = _screen_metrics(args, train_path, decoder_path, source_encoder, source_decoder, device)
        payload = {"schema": f"{SCHEMA_PREFIX}_screen.v1", "stage": "screen", "status": "valid", "passed": details["passed"], "classification": None if details["passed"] else "prior_alignment_fit_failed", "next_action": None if details["passed"] else "close_current_prior_native_sced_route", "seed": args.seed, "checkpoint": str(decoder_path), "training_checkpoint": str(train_path), "checkpoint_sha256": file_sha256(decoder_path), "training_checkpoint_sha256": file_sha256(train_path), "training": training, "details": details, "evaluator_calls": 0, "lineage_sha256": canonical_sha256(lineage), "producer_contract_revision": f"{SCHEMA_PREFIX}-screen.v1", "consumer_contract": f"{SCHEMA_PREFIX}-confirm.v1" if details["passed"] else f"{SCHEMA_PREFIX}-report.v1", "permitted_consumers": ["confirm"] if details["passed"] else ["report", "closeout"], "input_artifact_hashes": {"preflight": file_sha256(args.preflight_evidence), "source_training": file_sha256(args.wae_training_checkpoint), "seed": args.seed, "anchor": anchor_hash}}
        _write(output, payload, args)
        return 0
    if action == "confirm":
        screen = _check_consumer(args.screen_evidence, action="confirm", schemas=(f"{SCHEMA_PREFIX}_screen.v1",))
        if (
            not screen.get("passed")
            or screen.get("seed") != args.seed
            or screen.get("checkpoint_sha256") != file_sha256(screen["checkpoint"])
        ):
            raise ValueError("SCED confirm requires passed screen")
        anchor = _anchor_confirm(args, action="confirm")
        anchor_hash = (
            file_sha256(args.anchor_confirm_evidence) if anchor is not None else None
        )
        decision = _stage(args, "confirm", f"{SCHEMA_PREFIX}-confirm.v1", {"screen": file_sha256(args.screen_evidence), "source_training": file_sha256(args.wae_training_checkpoint), "seed": args.seed, "checkpoint": file_sha256(screen["checkpoint"]), "anchor": anchor_hash})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        device = torch.device(args.device)
        train_path, decoder_path, training = _train(args, seed=args.seed, target_epoch=CONFIRM_EPOCHS, output=output, resume=args.resume_checkpoint or screen["training_checkpoint"], source_hash=file_sha256(args.wae_training_checkpoint), weights=screen["training"]["weights"])
        measurement = _apply_complete_o0_gate(
            _measure_checkpoint(decoder_path, candidate="O0-I", device=device, dataset_root=args.dataset_root, primary_path=args.primary_evaluator_checkpoint, secondary_path=args.secondary_evaluator_checkpoint)
        )
        passed = bool(measurement.get("passed"))
        classification = (
            "electronic_prior_native_sced_admitted"
            if passed
            else "prior_alignment_semantic_or_novelty_failed"
        )
        permitted_consumers = (
            ["locked-final", "screen", "confirm"]
            if passed and args.seed == TRAIN_SEEDS[0]
            else ["locked-final"]
            if passed
            else ["report", "closeout"]
        )
        payload = {"schema": f"{SCHEMA_PREFIX}_confirm.v1", "stage": "confirm", "status": "valid", "passed": passed, "classification": classification, "seed": args.seed, "checkpoint": str(decoder_path), "training_checkpoint": str(train_path), "training": training, "measurement": measurement, "checkpoint_sha256": file_sha256(decoder_path), "training_checkpoint_sha256": file_sha256(train_path), "evaluator_calls": 1, "lineage_sha256": canonical_sha256(lineage), "producer_contract_revision": f"{SCHEMA_PREFIX}-confirm.v1", "consumer_contract": f"{SCHEMA_PREFIX}-locked-final.v1" if passed else f"{SCHEMA_PREFIX}-report.v1", "permitted_consumers": permitted_consumers, "input_artifact_hashes": {"screen": file_sha256(args.screen_evidence), "source_training": file_sha256(args.wae_training_checkpoint), "seed": args.seed, "checkpoint": file_sha256(decoder_path), "anchor": anchor_hash}}
        _write(output, payload, args)
        return 0
    if action == "locked-final":
        _, seeds = _locked_confirmations(
            args.confirm_evidence,
            args.peer_confirm_evidence,
            args.checkpoint,
        )
        decision = _stage(args, "locked-final", f"{SCHEMA_PREFIX}-locked-final.v1", {"confirm": file_sha256(args.confirm_evidence), "checkpoint": file_sha256(args.checkpoint), "peer_confirm": canonical_sha256([file_sha256(path) for path in args.peer_confirm_evidence])})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        measurement = _apply_complete_o0_gate(
            _measure_checkpoint(
                args.checkpoint,
                candidate="O0-I",
                device=torch.device(args.device),
                dataset_root=args.dataset_root,
                primary_path=args.primary_evaluator_checkpoint,
                secondary_path=args.secondary_evaluator_checkpoint,
            )
        )
        storage_bytes = _tree_size(Path(args._run_manifest["output_root"]))
        if storage_bytes > 2 * 1024**3:
            raise RuntimeError("SCED output storage exceeds 2 GiB")
        passed = bool(measurement.get("passed"))
        payload = {"schema": f"{SCHEMA_PREFIX}_locked-final.v1", "stage": "locked-final", "status": "valid", "passed": passed, "classification": "electronic_prior_native_sced_admitted" if passed else "prior_alignment_semantic_or_novelty_failed", "seed_confirmations": seeds, "measurement": measurement, "storage_bytes": storage_bytes, "formal_promotion": False, "optical_training_authorized": False, "fiber_training_authorized": False, "synthetic_diagnostic_only": True, "checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint), "evaluator_calls": 1, "lineage_sha256": canonical_sha256(lineage), "producer_contract_revision": f"{SCHEMA_PREFIX}-locked-final.v1", "consumer_contract": f"{SCHEMA_PREFIX}-report.v1", "permitted_consumers": ["report", "closeout"], "input_artifact_hashes": {"confirm": file_sha256(args.confirm_evidence), "checkpoint": file_sha256(args.checkpoint), "peer_confirm": canonical_sha256([file_sha256(path) for path in args.peer_confirm_evidence])}}
        _write(output, payload, args)
        return 0
    if action == "report":
        measurement = _check_consumer(
            args.measurement_evidence,
            action="report",
            schemas=(
                f"{SCHEMA_PREFIX}_screen.v1",
                f"{SCHEMA_PREFIX}_confirm.v1",
                f"{SCHEMA_PREFIX}_locked-final.v1",
            ),
        )
        decision = _stage(
            args,
            "report",
            f"{SCHEMA_PREFIX}-report.v1",
            {"measurement": file_sha256(args.measurement_evidence)},
        )
        if decision["action"] == "REUSE":
            return 0
        payload = {"schema": f"{SCHEMA_PREFIX}_report.v1", "stage": "report", "status": "valid", "passed": bool(measurement.get("passed")), "classification": measurement.get("classification"), "measurement": args.measurement_evidence, "source_evidence_sha256": file_sha256(args.measurement_evidence), "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1", "permitted_consumers": ["closeout"]}
        _write(output, payload, args)
        return 0
    if action == "closeout":
        _check_consumer(
            args.measurement_evidence,
            action="closeout",
            schemas=(
                f"{SCHEMA_PREFIX}_screen.v1",
                f"{SCHEMA_PREFIX}_confirm.v1",
                f"{SCHEMA_PREFIX}_locked-final.v1",
                f"{SCHEMA_PREFIX}_report.v1",
            ),
        )
        decision = _stage(
            args,
            "closeout",
            f"{SCHEMA_PREFIX}-closeout.v1",
            {"measurement": file_sha256(args.measurement_evidence)},
        )
        if decision["action"] == "REUSE":
            return 0
        return _closeout(args, args.measurement_evidence, output)
    raise ValueError(f"unknown SCED action: {action}")
