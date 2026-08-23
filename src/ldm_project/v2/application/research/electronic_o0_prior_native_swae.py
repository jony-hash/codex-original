"""Prior-native electronic O0 lifecycle.

This research-only variant warm-starts the validated WAE image decoder and
changes exactly one scientific variable: MMD prior matching is replaced by a
deterministic classwise sliced-Wasserstein quantile objective.  The source
checkpoint is immutable, evaluators are never loaded before measurement, and
all formal invocations require an explicit research manifest.
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
    write_failure_record,
)
from .wae_models import (
    FIXED_EXPOSURE_SCALE,
    ConditionalWAEEncoder,
    ConditionalWAEDecoder,
    parameter_counts,
    render_view,
    sobel_l1,
)


SCHEMA_PREFIX = "electronic_o0_prior_native_swae"
TRAIN_SEEDS = (8601, 8602, 8603)
PROJECTION_SEED = 271829
BOOTSTRAP_SEED = 90210
LOCKED_PANEL_SEEDS = LOCKED_SEEDS
LEARNING_RATE = 2.0e-4
SCREEN_EPOCHS = 1
CONFIRM_EPOCHS = 3
CODE_DIM = 8
PROJECTION_COUNT = 256
SOURCE_TRAIN_SCHEMA = "electronic_o0_wae_training.v2"
SOURCE_DECODER_SCHEMA = "electronic_o0_wae_decoder.v2"


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
        raise ValueError("prior-native SWAE config must be a mapping")
    if payload.get("schema_version") != 1 or payload.get("code_dim") != CODE_DIM:
        raise ValueError("prior-native SWAE config schema/code dimension is invalid")
    if tuple(payload.get("train_seeds", ())) != TRAIN_SEEDS:
        raise ValueError("prior-native SWAE seeds are fixed to 8601/8602/8603")
    if payload.get("projection_seed") != PROJECTION_SEED:
        raise ValueError("prior-native SWAE projection seed is not registered")
    if payload.get("fixed_exposure_scale") != FIXED_EXPOSURE_SCALE:
        raise ValueError("prior-native SWAE fixed exposure is invalid")
    return payload


def _require_cuda(args: argparse.Namespace) -> None:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("prior-native SWAE formal lifecycle requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _evidence_invocation(args: argparse.Namespace) -> dict[str, Any]:
    """Return the explicit manifest invocation contract for evidence."""
    if not getattr(args, "_run_manifest", None):
        raise ValueError("prior-native SWAE evidence requires a run manifest")
    return {
        "invocation_mode": "manifest",
        "legacy_diagnostic_only": False,
        "run_manifest_sha256": file_sha256(args.run_manifest),
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence is not a JSON object: {path}")
    return payload


def _load_source_models(
    training_path: str | Path,
    decoder_path: str | Path,
    device: torch.device,
) -> tuple[ConditionalWAEEncoder, ConditionalWAEDecoder]:
    training = safe_torch_load(training_path, map_location=device)
    if not isinstance(training, dict) or training.get("schema") != SOURCE_TRAIN_SCHEMA:
        raise ValueError("source WAE training checkpoint schema is invalid")
    decoder_payload = safe_torch_load(decoder_path, map_location="cpu")
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
    decoder_reference = ConditionalWAEDecoder("O0-I")
    decoder_reference.load_state_dict(
        decoder_payload["decoder_state_dict"], strict=True
    )
    source_hash = canonical_sha256(
        {
            key: tensor_sha256(value.detach().cpu())
            for key, value in decoder.state_dict().items()
        }
    )
    reference_hash = canonical_sha256(
        {
            key: tensor_sha256(value.detach().cpu())
            for key, value in decoder_reference.state_dict().items()
        }
    )
    if source_hash != reference_hash:
        raise ValueError("source training and decoder checkpoints disagree")
    return encoder, decoder


def _source_contract(args: argparse.Namespace) -> dict[str, Any]:
    if not Path(args.wae_confirm_evidence).is_file():
        raise FileNotFoundError("WAE confirm evidence is missing")
    source_evidence = _load_json(args.wae_confirm_evidence)
    if (
        source_evidence.get("schema") != "electronic_o0_wae_confirm.v2"
        or source_evidence.get("status") != "valid"
        or source_evidence.get("classification") != "wae_image_manifold_failed"
    ):
        raise ValueError("new SWAE requires the valid failed WAE v2 confirm evidence")
    latent_evidence = _load_json(args.latent_transport_confirm_evidence)
    if (
        latent_evidence.get("schema") != "electronic_o0_latent_transport_confirm.v1"
        or latent_evidence.get("status") != "valid"
        or latent_evidence.get("classification") != "latent_flow_semantic_failed"
    ):
        raise ValueError(
            "new SWAE requires the valid failed latent-flow confirm evidence"
        )
    mixture_evidence = _load_json(args.mixture_closeout)
    if (
        mixture_evidence.get("schema") != "electronic_o0_mixture_transport_closeout.v1"
        or mixture_evidence.get("status") != "valid"
        or mixture_evidence.get("classification") != "mixture_latent_fit_failed"
    ):
        raise ValueError("new SWAE requires the valid failed mixture closeout")
    return {
        "training_checkpoint_sha256": file_sha256(args.wae_training_checkpoint),
        "decoder_checkpoint_sha256": file_sha256(args.wae_decoder_checkpoint),
        "confirm_evidence_sha256": file_sha256(args.wae_confirm_evidence),
        "confirm_classification": source_evidence["classification"],
        "latent_transport_confirm_sha256": file_sha256(
            args.latent_transport_confirm_evidence
        ),
        "latent_transport_classification": latent_evidence["classification"],
        "mixture_closeout_sha256": file_sha256(args.mixture_closeout),
        "mixture_classification": mixture_evidence["classification"],
    }


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    required_files = [
        args.config,
        args.wae_training_checkpoint,
        args.wae_decoder_checkpoint,
        args.wae_confirm_evidence,
        args.latent_transport_confirm_evidence,
        args.mixture_closeout,
        args.primary_evaluator_checkpoint,
        args.secondary_evaluator_checkpoint,
        args.primary_qualification_evidence,
        args.secondary_qualification_evidence,
    ]
    missing = [str(path) for path in required_files if not Path(path).is_file()]
    if missing or not Path(args.dataset_root).is_dir():
        raise FileNotFoundError(
            "prior-native SWAE lineage input is missing: " + ", ".join(missing)
        )
    git = _git()
    if not git["clean"]:
        raise RuntimeError("prior-native SWAE evidence requires a clean worktree")
    source = _source_contract(args)
    return {
        "schema": f"{SCHEMA_PREFIX}_lineage.v1",
        "config_sha256": file_sha256(args.config),
        "config_content": config,
        "source": source,
        "primary_evaluator_sha256": file_sha256(args.primary_evaluator_checkpoint),
        "secondary_evaluator_sha256": file_sha256(args.secondary_evaluator_checkpoint),
        "primary_qualification_sha256": file_sha256(
            args.primary_qualification_evidence
        ),
        "secondary_qualification_sha256": file_sha256(
            args.secondary_qualification_evidence
        ),
        "git": git,
        "device": args.device,
        "seeds": {
            "train": list(TRAIN_SEEDS),
            "projection": PROJECTION_SEED,
            "locked_panels": list(LOCKED_PANEL_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
        },
    }


def fixed_projections(
    *, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
    values = torch.randn((PROJECTION_COUNT, CODE_DIM), generator=generator, dtype=dtype)
    return F.normalize(values, dim=1).to(device)


def gaussian_quantiles(
    count: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if count < 2:
        raise ValueError("SWD quantiles require at least two samples")
    probabilities = (torch.arange(count, device=device, dtype=dtype) + 0.5) / count
    return torch.special.ndtri(probabilities).reshape(count, 1)


def classwise_sliced_wasserstein_quantile(
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    projections: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        codes.ndim != 2
        or codes.shape[1] != CODE_DIM
        or labels.shape != (codes.shape[0],)
    ):
        raise ValueError("SWD code/label shapes are invalid")
    if projections is None:
        projections = fixed_projections(device=codes.device, dtype=codes.dtype)
    if projections.shape != (PROJECTION_COUNT, CODE_DIM):
        raise ValueError("SWD projection bank shape is invalid")
    terms = []
    for label in range(10):
        selected = codes[labels == label]
        if selected.shape[0] < 2:
            raise ValueError("SWD requires at least two samples per class")
        projected = torch.sort(selected @ projections.t(), dim=0).values
        target = gaussian_quantiles(
            selected.shape[0], device=codes.device, dtype=codes.dtype
        )
        terms.append((projected - target).square().mean())
    return torch.stack(terms).mean()


def _prior_native_terms(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor,
    *,
    w_swd: float,
    w_cycle: float,
    projections: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    encoded = encoder(images, labels)
    prediction = render_view(decoder(encoded, labels), decoder.candidate)
    reconstruction = F.binary_cross_entropy(prediction, images) + 0.25 * sobel_l1(
        prediction, images
    )
    swd = classwise_sliced_wasserstein_quantile(
        encoded, labels, projections=projections
    )
    cycle_prediction = render_view(decoder(prior, labels), decoder.candidate)
    cycle = F.smooth_l1_loss(encoder(cycle_prediction, labels), prior)
    total = reconstruction + float(w_swd) * swd + float(w_cycle) * cycle
    return total, {
        "reconstruction": reconstruction,
        "swd": swd,
        "cycle": cycle,
        "encoded": encoded,
    }


def _balanced_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    per_class: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for label in range(10):
        choices = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
        rows.append(
            choices.index_select(
                0, torch.randperm(choices.numel(), generator=generator)
            )[:per_class]
        )
    index = torch.cat(rows)
    return images.index_select(0, index), labels.index_select(0, index)


def calibrate_swae_weights(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    parameters = tuple(encoder.parameters()) + tuple(decoder.parameters())
    projections = fixed_projections(device=images.device, dtype=images.dtype)
    values = []
    for seed in (11, 23, 37, 51):
        batch_images, batch_labels = _balanced_batch(images, labels, seed=seed)
        prior = torch.randn(
            (batch_images.shape[0], CODE_DIM),
            generator=torch.Generator(device="cpu").manual_seed(seed + 1000),
        ).to(images.device)
        encoded = encoder(batch_images, batch_labels)
        prediction = render_view(decoder(encoded, batch_labels), decoder.candidate)
        reconstruction = F.binary_cross_entropy(
            prediction, batch_images
        ) + 0.25 * sobel_l1(prediction, batch_images)
        swd = classwise_sliced_wasserstein_quantile(
            encoded, batch_labels, projections=projections
        )
        cycle_prediction = render_view(decoder(prior, batch_labels), decoder.candidate)
        cycle = F.smooth_l1_loss(encoder(cycle_prediction, batch_labels), prior)
        norms = []
        for term in (reconstruction, swd, cycle):
            gradients = torch.autograd.grad(
                term, parameters, retain_graph=True, allow_unused=True
            )
            norms.append(
                float(
                    torch.sqrt(
                        sum(
                            (g.square().sum() for g in gradients if g is not None),
                            torch.zeros((), device=images.device),
                        )
                    )
                    .detach()
                    .cpu()
                )
            )
        values.append(norms)
    mean = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(mean).all() or float(mean[:, 0].mean()) <= 0:
        raise FloatingPointError("SWAE gradient calibration is non-finite")
    reconstruction_norm, swd_norm, cycle_norm = mean.mean(0)
    return {
        "batch_seeds": [11, 23, 37, 51],
        "w_swd": float(0.50 * reconstruction_norm / swd_norm.clamp_min(1.0e-12)),
        "w_cycle": float(0.25 * reconstruction_norm / cycle_norm.clamp_min(1.0e-12)),
        "gradient_norms": {
            "reconstruction": float(reconstruction_norm),
            "swd": float(swd_norm),
            "cycle": float(cycle_norm),
        },
    }


def _model_hash(encoder: nn.Module, decoder: nn.Module) -> str:
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


def _optimizer(
    device: torch.device, encoder: nn.Module, decoder: nn.Module
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        tuple(encoder.parameters()) + tuple(decoder.parameters()), lr=LEARNING_RATE
    )


def _update(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    weights: dict[str, Any],
    projections: torch.Tensor,
) -> float:
    prior = torch.randn((images.shape[0], CODE_DIM), device=images.device)
    optimizer.zero_grad(set_to_none=True)
    total, _ = _prior_native_terms(
        encoder,
        decoder,
        images,
        labels,
        prior,
        w_swd=float(weights["w_swd"]),
        w_cycle=float(weights["w_cycle"]),
        projections=projections,
    )
    if not torch.isfinite(total):
        raise FloatingPointError("SWAE update loss is non-finite")
    total.backward()
    optimizer.step()
    return float(total.detach().cpu())


def _save_training(
    path: Path,
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    epoch: int,
    step: int,
    weights: dict[str, Any],
    schedule_summary: dict[str, Any],
    source_hash: str,
) -> None:
    payload = {
        "schema": f"{SCHEMA_PREFIX}_training.v1",
        "seed": seed,
        "epoch": epoch,
        "step": step,
        "encoder": {
            key: value.detach().cpu() for key, value in encoder.state_dict().items()
        },
        "decoder": {
            key: value.detach().cpu() for key, value in decoder.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "weights": weights,
        "schedule": schedule_summary,
        "source_training_checkpoint_sha256": source_hash,
        "rng_state": {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    atomic_torch_save(path, payload)


def _save_decoder(
    path: Path,
    decoder: ConditionalWAEDecoder,
    *,
    seed: int,
    epoch: int,
    source_hash: str,
) -> None:
    payload = {
        "schema": SOURCE_DECODER_SCHEMA,
        "candidate": "O0-I",
        "seed": seed,
        "epoch": epoch,
        "decoder_state_dict": {
            key: value.detach().cpu() for key, value in decoder.state_dict().items()
        },
        "decoder_content_hash": canonical_sha256(
            {
                key: tensor_sha256(value.detach().cpu())
                for key, value in decoder.state_dict().items()
            }
        ),
        "source_checkpoint_sha256": source_hash,
        "contains_encoder": False,
        "contains_optimizer": False,
    }
    atomic_torch_save(path, payload)


def _load_training(
    path: Path, device: torch.device
) -> tuple[
    ConditionalWAEEncoder, ConditionalWAEDecoder, torch.optim.Optimizer, dict[str, Any]
]:
    payload = safe_torch_load(path, map_location=device)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != f"{SCHEMA_PREFIX}_training.v1"
    ):
        raise ValueError("SWAE training checkpoint schema is invalid")
    encoder = ConditionalWAEEncoder().to(device)
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    optimizer = _optimizer(device, encoder, decoder)
    encoder.load_state_dict(payload["encoder"], strict=True)
    decoder.load_state_dict(payload["decoder"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    torch.set_rng_state(payload["rng_state"]["cpu"])
    torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])
    return encoder, decoder, optimizer, payload


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
    train_images, train_labels = load_local_mnist(args.dataset_root, train=True)
    train_rows, _ = stratified_mnist_indices(train_labels)
    train_images = train_images.index_select(0, train_rows)
    train_labels = train_labels.index_select(0, train_rows)
    if resume:
        encoder, decoder, optimizer, payload = _load_training(Path(resume), device)
        start_epoch = int(payload["epoch"])
        step = int(payload["step"])
        if payload.get("source_training_checkpoint_sha256") != source_hash:
            raise ValueError("SWAE resume checkpoint source hash mismatch")
    else:
        encoder, decoder = _load_source_models(
            args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
        )
        optimizer = _optimizer(device, encoder, decoder)
        start_epoch = 0
        step = 0
    projections = fixed_projections(device=device)
    losses = []
    schedule = balanced_epoch_schedule(train_labels, seed=seed, per_class=10)
    for epoch in range(start_epoch, target_epoch):
        for indices in schedule:
            batch_images = train_images.index_select(0, indices).to(device)
            batch_labels = train_labels.index_select(0, indices).to(device)
            losses.append(
                _update(
                    encoder,
                    decoder,
                    optimizer,
                    batch_images,
                    batch_labels,
                    weights,
                    projections,
                )
            )
            step += 1
    training_path = output.with_name(f"seed_{seed}_training.pt")
    decoder_path = output.with_name(f"seed_{seed}_decoder.pt")
    _save_training(
        training_path,
        encoder,
        decoder,
        optimizer,
        seed=seed,
        epoch=target_epoch,
        step=step,
        weights=weights,
        schedule_summary=balanced_schedule_summary(train_labels, schedule),
        source_hash=source_hash,
    )
    _save_decoder(
        decoder_path,
        decoder,
        seed=seed,
        epoch=target_epoch,
        source_hash=file_sha256(training_path),
    )
    return (
        training_path,
        decoder_path,
        {
            "seed": seed,
            "epoch": target_epoch,
            "step": step,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "schedule": balanced_schedule_summary(train_labels, schedule),
            "weights": weights,
        },
    )


def _screen_metrics(
    args: argparse.Namespace,
    decoder_path: Path,
    training_path: Path,
    *,
    source_encoder: ConditionalWAEEncoder,
    source_decoder: ConditionalWAEDecoder,
    device: torch.device,
) -> dict[str, Any]:
    train_all, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_rows, validation_rows = stratified_mnist_indices(labels_all)
    train_images, train_labels = (
        train_all.index_select(0, train_rows),
        labels_all.index_select(0, train_rows),
    )
    validation_images, validation_labels = (
        train_all.index_select(0, validation_rows),
        labels_all.index_select(0, validation_rows),
    )
    encoder, decoder, _, _ = _load_training(training_path, device)
    projections = fixed_projections(device=device)
    with torch.no_grad():
        source_encoded = source_encoder(
            validation_images.to(device), validation_labels.to(device)
        )
        encoded = encoder(validation_images.to(device), validation_labels.to(device))
        source_prediction = render_view(
            source_decoder(source_encoded, validation_labels.to(device)), "O0-I"
        )
        prediction = render_view(decoder(encoded, validation_labels.to(device)), "O0-I")
        prior = torch.randn(
            (validation_labels.numel(), CODE_DIM),
            generator=torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED),
        ).to(device)
        prior_views = render_view(decoder(prior, validation_labels.to(device)), "O0-I")
    source_swd = float(
        classwise_sliced_wasserstein_quantile(
            source_encoded, validation_labels.to(device), projections=projections
        ).cpu()
    )
    swd = float(
        classwise_sliced_wasserstein_quantile(
            encoded, validation_labels.to(device), projections=projections
        ).cpu()
    )
    source_reconstruction = float(
        (source_prediction - validation_images.to(device)).square().mean().cpu()
    )
    reconstruction = float(
        (prediction - validation_images.to(device)).square().mean().cpu()
    )
    memory = _memorization_audit(
        prior_views,
        validation_labels,
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        device=device,
    )
    codes, panel_labels = panel_codes(4242, device=device)
    jacobian = _jacobian_summary(
        decoder,
        torch.stack(
            [codes[label * 50 + offset] for label in range(10) for offset in range(3)]
        ),
        torch.stack(
            [
                panel_labels[label * 50 + offset]
                for label in range(10)
                for offset in range(3)
            ]
        ),
        candidate="O0-I",
    )
    details = {
        "source_swd": source_swd,
        "swd": swd,
        "swd_relative_reduction": 1.0 - swd / max(source_swd, 1.0e-12),
        "source_reconstruction_mse": source_reconstruction,
        "reconstruction_mse": reconstruction,
        "prior_saturation": float((prior_views >= 1.0).float().mean().cpu()),
        "memorization": memory,
        "jacobian": jacobian,
        "checkpoint_sha256": file_sha256(decoder_path),
        "training_checkpoint_sha256": file_sha256(training_path),
    }
    details["passed"] = bool(
        details["swd_relative_reduction"] >= 0.50
        and swd <= source_swd
        and reconstruction <= source_reconstruction * 1.05
        and memory["exact_duplicate_count"] == 0
        and memory["pass_fraction"] >= 0.90
        and details["prior_saturation"] <= 0.08
        and jacobian["rank_1e-3_pass_fraction"] >= 0.80
    )
    return details


def _apply_complete_o0_gate(measurement: dict[str, Any]) -> dict[str, Any]:
    """Add the O0 acceptance gates omitted by the shared panel helper."""
    gates = dict(measurement.get("gates", {}))
    official = measurement.get("official_test", {})
    jacobian = measurement.get("jacobian", {})
    memorization = measurement.get("memorization", {})
    gates.update(
        {
            "official_test_count": official.get("count") == 10_000,
            "official_primary_accuracy": official.get("primary_accuracy", 0.0) >= 0.95,
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
            "official_camera_code_fraction": official.get("camera_code_fraction", 0.0)
            >= 0.25,
            "official_saturation": official.get("saturation", 1.0) <= 0.08,
            "jacobian_rank": jacobian.get("rank_1e-3_pass_fraction", 0.0) >= 0.80,
            "memorization": memorization.get("pass_fraction", 0.0) >= 0.95
            and memorization.get("exact_duplicate_count") == 0,
        }
    )
    measurement["gates"] = gates
    measurement["passed"] = all(gates.values())
    return measurement


def _audit(
    args: argparse.Namespace, lineage: dict[str, Any], output: Path
) -> dict[str, Any]:
    _require_cuda(args)
    device = torch.device(args.device)
    encoder, decoder = _load_source_models(
        args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
    )
    before_hash = _model_hash(encoder, decoder)
    train_all, labels_all = load_local_mnist(args.dataset_root, train=True)
    _, validation_rows = stratified_mnist_indices(labels_all)
    validation_images = train_all.index_select(0, validation_rows)
    validation_labels = labels_all.index_select(0, validation_rows)
    batch_images, batch_labels = _balanced_batch(
        validation_images, validation_labels, seed=2718
    )
    weights = calibrate_swae_weights(
        encoder, decoder, batch_images.to(device), batch_labels.to(device)
    )
    projections = fixed_projections(device=device)
    prior = torch.randn(
        (100, CODE_DIM), generator=torch.Generator(device="cpu").manual_seed(2719)
    ).to(device)
    total, terms = _prior_native_terms(
        encoder,
        decoder,
        batch_images.to(device),
        batch_labels.to(device),
        prior,
        w_swd=weights["w_swd"],
        w_cycle=weights["w_cycle"],
        projections=projections,
    )
    gradients = torch.autograd.grad(
        terms["swd"], tuple(encoder.parameters()), allow_unused=True
    )
    gradient_norm = float(
        torch.sqrt(
            sum(
                (g.square().sum() for g in gradients if g is not None),
                torch.zeros((), device=device),
            )
        )
        .detach()
        .cpu()
    )
    base_swd = float(terms["swd"].detach().cpu())
    base_reconstruction = float(terms["reconstruction"].detach().cpu())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise FloatingPointError("SWAE audit gradient is invalid")

    def direction_probe(sign: float) -> dict[str, float]:
        probe_encoder = ConditionalWAEEncoder().to(device)
        probe_decoder = ConditionalWAEDecoder("O0-I").to(device)
        probe_encoder.load_state_dict(encoder.state_dict(), strict=True)
        probe_decoder.load_state_dict(decoder.state_dict(), strict=True)
        probe_terms = _prior_native_terms(
            probe_encoder,
            probe_decoder,
            batch_images.to(device),
            batch_labels.to(device),
            prior,
            w_swd=weights["w_swd"],
            w_cycle=weights["w_cycle"],
            projections=projections,
        )[1]
        probe_gradients = torch.autograd.grad(
            probe_terms["swd"], tuple(probe_encoder.parameters()), allow_unused=True
        )
        probe_norm = torch.sqrt(
            sum(
                (g.square().sum() for g in probe_gradients if g is not None),
                torch.zeros((), device=device),
            )
        ).clamp_min(1.0e-12)
        with torch.no_grad():
            for parameter, gradient in zip(
                probe_encoder.parameters(), probe_gradients, strict=True
            ):
                if gradient is not None:
                    parameter.add_(float(sign) * 1.0e-4 * gradient / probe_norm)
        with torch.no_grad():
            values = _prior_native_terms(
                probe_encoder,
                probe_decoder,
                batch_images.to(device),
                batch_labels.to(device),
                prior,
                w_swd=weights["w_swd"],
                w_cycle=weights["w_cycle"],
                projections=projections,
            )[1]
        return {
            "swd": float(values["swd"].cpu()),
            "reconstruction": float(values["reconstruction"].cpu()),
        }

    positive = direction_probe(1.0)
    negative = direction_probe(-1.0)
    after_hash = _model_hash(encoder, decoder)
    positive_passed = (
        positive["swd"] <= base_swd * 0.99
        and positive["reconstruction"] <= base_reconstruction * 1.02
    )
    negative_passed = (
        negative["swd"] <= base_swd * 0.99
        and negative["reconstruction"] <= base_reconstruction * 1.02
    )
    passed = bool(
        bool(torch.isfinite(total).item())
        and before_hash == after_hash
        and gradient_norm > 0
        and positive_passed
        and not negative_passed
    )
    payload = {
        "schema": f"{SCHEMA_PREFIX}_audit.v1",
        "stage": "audit",
        "status": "valid" if passed else "invalid_harness",
        "passed": passed,
        "lineage_sha256": canonical_sha256(lineage),
        "weights": weights,
        "swd_gradient_norm": gradient_norm,
        "swd_value": base_swd,
        "reconstruction_value": base_reconstruction,
        "direction_probe": {
            "positive": positive,
            "negative": negative,
            "positive_passed": positive_passed,
            "negative_passed": negative_passed,
        },
        "source_parameter_hash_before": before_hash,
        "source_parameter_hash_after": after_hash,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1",
        "consumer_contract": f"{SCHEMA_PREFIX}-preflight.v1",
        "permitted_consumers": ["preflight"],
        "input_artifact_hashes": {
            "source_training": file_sha256(args.wae_training_checkpoint),
            "source_decoder": file_sha256(args.wae_decoder_checkpoint),
        },
    }
    payload.update(_evidence_invocation(args))
    _write(output, payload)
    return payload


def _preflight(
    args: argparse.Namespace,
    lineage: dict[str, Any],
    output: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    _require_cuda(args)
    device = torch.device(args.device)
    train_all, labels_all = load_local_mnist(args.dataset_root, train=True)
    train_rows, _ = stratified_mnist_indices(labels_all)
    images = train_all.index_select(0, train_rows)
    labels = labels_all.index_select(0, train_rows)
    batch_images, batch_labels = _balanced_batch(images, labels, seed=8601)
    batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
    encoder, decoder = _load_source_models(
        args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
    )
    weights = audit["weights"]
    optimizer = _optimizer(device, encoder, decoder)
    projections = fixed_projections(device=device)
    initial_encoder = {
        key: value.detach().clone() for key, value in encoder.state_dict().items()
    }
    initial_decoder = {
        key: value.detach().clone() for key, value in decoder.state_dict().items()
    }
    initial_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
    updates = [
        _update(
            encoder,
            decoder,
            optimizer,
            batch_images,
            batch_labels,
            weights,
            projections,
        )
        for _ in range(3)
    ]
    uninterrupted_encoder = {
        key: value.detach().clone() for key, value in encoder.state_dict().items()
    }
    uninterrupted_decoder = {
        key: value.detach().clone() for key, value in decoder.state_dict().items()
    }
    uninterrupted_optimizer = optimizer.state_dict()
    uninterrupted_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
    interrupted_encoder, interrupted_decoder = _load_source_models(
        args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
    )
    interrupted_optimizer = _optimizer(device, interrupted_encoder, interrupted_decoder)
    torch.set_rng_state(initial_rng[0])
    torch.cuda.set_rng_state_all(initial_rng[1])
    _update(
        interrupted_encoder,
        interrupted_decoder,
        interrupted_optimizer,
        batch_images,
        batch_labels,
        weights,
        projections,
    )
    replay_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
    resumed_encoder = ConditionalWAEEncoder().to(device)
    resumed_decoder = ConditionalWAEDecoder("O0-I").to(device)
    resumed_encoder.load_state_dict(interrupted_encoder.state_dict(), strict=True)
    resumed_decoder.load_state_dict(interrupted_decoder.state_dict(), strict=True)
    resumed_optimizer = _optimizer(device, resumed_encoder, resumed_decoder)
    resumed_optimizer.load_state_dict(interrupted_optimizer.state_dict())
    for state in resumed_optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    torch.set_rng_state(replay_rng[0])
    torch.cuda.set_rng_state_all(replay_rng[1])
    for _ in range(2):
        _update(
            resumed_encoder,
            resumed_decoder,
            resumed_optimizer,
            batch_images,
            batch_labels,
            weights,
            projections,
        )
    replay_exact = (
        _states_equal(
            uninterrupted_encoder,
            {
                key: value.detach().clone()
                for key, value in resumed_encoder.state_dict().items()
            },
        )
        and _states_equal(
            uninterrupted_decoder,
            {
                key: value.detach().clone()
                for key, value in resumed_decoder.state_dict().items()
            },
        )
        and _optimizer_state_equal(
            uninterrupted_optimizer, resumed_optimizer.state_dict()
        )
        and torch.equal(uninterrupted_rng[0], torch.get_rng_state())
        and all(
            torch.equal(expected, actual)
            for expected, actual in zip(
                uninterrupted_rng[1], torch.cuda.get_rng_state_all(), strict=True
            )
        )
    )
    encoder_changed = not _states_equal(initial_encoder, uninterrupted_encoder)
    decoder_changed = not _states_equal(initial_decoder, uninterrupted_decoder)
    passed = bool(
        all(math.isfinite(value) for value in updates)
        and encoder_changed
        and decoder_changed
        and replay_exact
    )
    payload = {
        "schema": f"{SCHEMA_PREFIX}_preflight.v1",
        "stage": "preflight",
        "status": "valid" if passed else "invalid_harness",
        "passed": passed,
        "updates": updates,
        "weights": weights,
        "encoder_changed": encoder_changed,
        "decoder_changed": decoder_changed,
        "interrupted_resumed_replay_exact": replay_exact,
        "evaluator_calls": 0,
        "optimizer_created": True,
        "lineage_sha256": canonical_sha256(lineage),
        "producer_contract_revision": f"{SCHEMA_PREFIX}-preflight.v1",
        "consumer_contract": f"{SCHEMA_PREFIX}-screen.v1",
        "permitted_consumers": ["screen"],
        "input_artifact_hashes": {
            "audit": file_sha256(args.audit_evidence) if args.audit_evidence else None,
            "source_training": file_sha256(args.wae_training_checkpoint),
        },
    }
    payload.update(_evidence_invocation(args))
    _write(output, payload)
    return payload


def _stage_decision(
    args: argparse.Namespace, stage: str, producer: str, expected: dict[str, Any]
) -> dict[str, Any] | None:
    if not getattr(args, "_run_manifest", None):
        return None
    if Path(args.output).is_file():
        existing = _load_json(args.output)
        if existing.get("status") != "valid":
            raise ValueError(
                f"{stage} REJECT: existing evidence is not valid; "
                "use a new sibling run manifest"
            )
    decision = stage_action_decision(
        stage,
        producer_contract=producer,
        expected_inputs=expected,
        evidence=args.output,
    )
    args._stage_decision = decision
    return decision


def _closeout(args: argparse.Namespace, source_path: str, output: Path) -> int:
    source = _load_json(source_path)
    if source.get("status") != "valid":
        raise ValueError("SWAE closeout requires valid measurement evidence")
    classification = source.get("classification")
    if classification is None:
        classification = (
            "electronic_prior_native_swae_admitted"
            if source.get("passed")
            else "prior_alignment_semantic_or_novelty_failed"
        )
    payload = {
        "schema": f"{SCHEMA_PREFIX}_closeout.v1",
        "stage": "closeout",
        "status": "valid",
        "classification": classification,
        "next_action": "register_analytic_o0_g_command_bridge"
        if classification == "electronic_prior_native_swae_admitted"
        else "close_current_prior_native_swae_route",
        "source_evidence_sha256": file_sha256(source_path),
        "producer_contract_revision": f"{SCHEMA_PREFIX}-closeout.v1",
        "consumer_contract": "research-handoff.v1",
        "permitted_consumers": [],
        "input_artifact_hashes": {"measurement": file_sha256(source_path)},
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if getattr(args, "_run_manifest", None):
        payload.update(_evidence_invocation(args))
    _write(output, payload)
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("prior-native SWAE formal lifecycle is manifest-only")
    action = args.research_action
    required = {
        "validate": (
            "config",
            "dataset_root",
            "output",
            "wae_training_checkpoint",
            "wae_decoder_checkpoint",
            "wae_confirm_evidence",
            "latent_transport_confirm_evidence",
            "mixture_closeout",
            "primary_evaluator_checkpoint",
            "secondary_evaluator_checkpoint",
            "primary_qualification_evidence",
            "secondary_qualification_evidence",
        ),
        "audit": (
            "config",
            "dataset_root",
            "output",
            "wae_training_checkpoint",
            "wae_decoder_checkpoint",
            "wae_confirm_evidence",
            "latent_transport_confirm_evidence",
            "mixture_closeout",
            "primary_evaluator_checkpoint",
            "secondary_evaluator_checkpoint",
            "primary_qualification_evidence",
            "secondary_qualification_evidence",
        ),
        "preflight": (
            "config",
            "dataset_root",
            "output",
            "wae_training_checkpoint",
            "wae_decoder_checkpoint",
            "wae_confirm_evidence",
            "latent_transport_confirm_evidence",
            "mixture_closeout",
            "primary_evaluator_checkpoint",
            "secondary_evaluator_checkpoint",
            "primary_qualification_evidence",
            "secondary_qualification_evidence",
            "audit_evidence",
        ),
        "screen": (
            "config",
            "dataset_root",
            "output",
            "wae_training_checkpoint",
            "wae_decoder_checkpoint",
            "wae_confirm_evidence",
            "latent_transport_confirm_evidence",
            "mixture_closeout",
            "primary_evaluator_checkpoint",
            "secondary_evaluator_checkpoint",
            "primary_qualification_evidence",
            "secondary_qualification_evidence",
            "preflight_evidence",
        ),
        "confirm": (
            "config",
            "dataset_root",
            "output",
            "wae_training_checkpoint",
            "wae_decoder_checkpoint",
            "wae_confirm_evidence",
            "latent_transport_confirm_evidence",
            "mixture_closeout",
            "primary_evaluator_checkpoint",
            "secondary_evaluator_checkpoint",
            "primary_qualification_evidence",
            "secondary_qualification_evidence",
            "screen_evidence",
        ),
        "locked-final": (
            "config",
            "dataset_root",
            "output",
            "wae_training_checkpoint",
            "wae_decoder_checkpoint",
            "wae_confirm_evidence",
            "latent_transport_confirm_evidence",
            "mixture_closeout",
            "primary_evaluator_checkpoint",
            "secondary_evaluator_checkpoint",
            "primary_qualification_evidence",
            "secondary_qualification_evidence",
            "confirm_evidence",
            "checkpoint",
        ),
        "report": ("config", "dataset_root", "output", "measurement_evidence"),
        "closeout": ("config", "dataset_root", "output", "measurement_evidence"),
    }[action]
    fields = (
        "config",
        "dataset_root",
        "device",
        "output",
        "seed",
        "wae_training_checkpoint",
        "wae_decoder_checkpoint",
        "wae_confirm_evidence",
        "latent_transport_confirm_evidence",
        "mixture_closeout",
        "primary_evaluator_checkpoint",
        "secondary_evaluator_checkpoint",
        "primary_qualification_evidence",
        "secondary_qualification_evidence",
        "audit_evidence",
        "preflight_evidence",
        "screen_evidence",
        "confirm_evidence",
        "measurement_evidence",
        "checkpoint",
        "resume_checkpoint",
        "peer_confirm_evidence",
    )
    defaults = {"device": "cuda", "seed": 8601}
    try:
        args = prepare_args(
            args,
            program="electronic-o0-prior-native-swae",
            action=action,
            required=required,
            defaults=defaults,
            mixed_fields=fields,
        )
        validate_manifest_namespace(args, action=action)
        config = _config(args.config)
        lineage = (
            _lineage(args, config) if action not in {"report", "closeout"} else None
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if action == "validate":
            _stage_decision(
                args,
                "validate",
                f"{SCHEMA_PREFIX}-validation.v1",
                {
                    "source_training": file_sha256(args.wae_training_checkpoint),
                    "source_decoder": file_sha256(args.wae_decoder_checkpoint),
                },
            )
            if getattr(args, "_stage_decision", {}).get("decision") == "reuse":
                return 0
            _require_cuda(args)
            payload = {
                "schema": f"{SCHEMA_PREFIX}_validation.v1",
                "stage": "validate",
                "status": "valid",
                "passed": True,
                "lineage": lineage,
                "producer_contract_revision": f"{SCHEMA_PREFIX}-validation.v1",
                "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1",
                "permitted_consumers": ["audit"],
                "input_artifact_hashes": {
                    "source_training": file_sha256(args.wae_training_checkpoint),
                    "source_decoder": file_sha256(args.wae_decoder_checkpoint),
                },
            }
            payload.update(_evidence_invocation(args))
            _write(output, payload)
            return 0
        if action == "audit":
            _stage_decision(
                args,
                "audit",
                f"{SCHEMA_PREFIX}-audit.v1",
                {
                    "source_training": file_sha256(args.wae_training_checkpoint),
                    "source_decoder": file_sha256(args.wae_decoder_checkpoint),
                },
            )
            if getattr(args, "_stage_decision", {}).get("decision") == "reuse":
                return 0
            _audit(args, lineage, output)
            return 0
        if action == "preflight":
            audit = _load_json(args.audit_evidence)
            if audit.get("status") != "valid" or not audit.get("passed"):
                raise ValueError("SWAE preflight requires passed audit")
            _stage_decision(
                args,
                "preflight",
                f"{SCHEMA_PREFIX}-preflight.v1",
                {
                    "audit": file_sha256(args.audit_evidence),
                    "source_training": file_sha256(args.wae_training_checkpoint),
                },
            )
            if getattr(args, "_stage_decision", {}).get("decision") == "reuse":
                return 0
            _preflight(args, lineage, output, audit)
            return 0
        if action == "screen":
            preflight = _load_json(args.preflight_evidence)
            if preflight.get("status") != "valid" or not preflight.get("passed"):
                raise ValueError("SWAE screen requires passed preflight")
            _stage_decision(
                args,
                "screen",
                f"{SCHEMA_PREFIX}-screen.v1",
                {
                    "preflight": file_sha256(args.preflight_evidence),
                    "source_training": file_sha256(args.wae_training_checkpoint),
                    "seed": args.seed,
                },
            )
            if getattr(args, "_stage_decision", {}).get("decision") == "reuse":
                return 0
            _require_cuda(args)
            device = torch.device(args.device)
            source_encoder, source_decoder = _load_source_models(
                args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
            )
            weights = preflight["weights"]
            train_path, decoder_path, training = _train(
                args,
                seed=args.seed,
                target_epoch=SCREEN_EPOCHS,
                output=output,
                resume=args.resume_checkpoint,
                source_hash=file_sha256(args.wae_training_checkpoint),
                weights=weights,
            )
            details = _screen_metrics(
                args,
                decoder_path,
                train_path,
                source_encoder=source_encoder,
                source_decoder=source_decoder,
                device=device,
            )
            payload = {
                "schema": f"{SCHEMA_PREFIX}_screen.v1",
                "stage": "screen",
                "status": "valid",
                "passed": details["passed"],
                "classification": None,
                "checkpoint": str(decoder_path),
                "training_checkpoint": str(train_path),
                "training": training,
                "details": details,
                "evaluator_calls": 0,
                "lineage_sha256": canonical_sha256(lineage),
                "producer_contract_revision": f"{SCHEMA_PREFIX}-screen.v1",
                "consumer_contract": f"{SCHEMA_PREFIX}-confirm.v1",
                "permitted_consumers": ["confirm"],
                "input_artifact_hashes": {
                    "preflight": file_sha256(args.preflight_evidence),
                    "source_training": file_sha256(args.wae_training_checkpoint),
                    "seed": args.seed,
                },
            }
            payload.update(_evidence_invocation(args))
            _write(output, payload)
            return 0
        if action == "confirm":
            screen = _load_json(args.screen_evidence)
            if screen.get("status") != "valid" or not screen.get("passed"):
                raise ValueError("SWAE confirm requires passed screen")
            _stage_decision(
                args,
                "confirm",
                f"{SCHEMA_PREFIX}-confirm.v1",
                {
                    "screen": file_sha256(args.screen_evidence),
                    "source_training": file_sha256(args.wae_training_checkpoint),
                    "seed": args.seed,
                },
            )
            if getattr(args, "_stage_decision", {}).get("decision") == "reuse":
                return 0
            weights = screen["training"]["weights"]
            train_path, decoder_path, training = _train(
                args,
                seed=args.seed,
                target_epoch=CONFIRM_EPOCHS,
                output=output,
                resume=args.resume_checkpoint or screen["training_checkpoint"],
                source_hash=file_sha256(args.wae_training_checkpoint),
                weights=weights,
            )
            measurement = _apply_complete_o0_gate(
                _measure_checkpoint(
                    decoder_path,
                    candidate="O0-I",
                    device=torch.device(args.device),
                    dataset_root=args.dataset_root,
                    primary_path=args.primary_evaluator_checkpoint,
                    secondary_path=args.secondary_evaluator_checkpoint,
                )
            )
            classification = (
                "electronic_prior_native_swae_admitted"
                if measurement["passed"]
                else "prior_alignment_semantic_or_novelty_failed"
            )
            payload = {
                "schema": f"{SCHEMA_PREFIX}_confirm.v1",
                "stage": "confirm",
                "status": "valid",
                "passed": measurement["passed"],
                "classification": classification,
                "checkpoint": str(decoder_path),
                "training_checkpoint": str(train_path),
                "training": training,
                "measurement": measurement,
                "formal_promotion": False,
                "optical_training_authorized": False,
                "fiber_training_authorized": False,
                "synthetic_diagnostic_only": True,
                "lineage_sha256": canonical_sha256(lineage),
                "producer_contract_revision": f"{SCHEMA_PREFIX}-confirm.v1",
                "consumer_contract": f"{SCHEMA_PREFIX}-locked-final.v1",
                "permitted_consumers": ["locked-final"],
                "input_artifact_hashes": {
                    "screen": file_sha256(args.screen_evidence),
                    "source_training": file_sha256(args.wae_training_checkpoint),
                    "seed": args.seed,
                },
            }
            payload.update(_evidence_invocation(args))
            _write(output, payload)
            return 0
        if action == "locked-final":
            confirmations = [_load_json(args.confirm_evidence)] + [
                _load_json(path) for path in args.peer_confirm_evidence
            ]
            if (
                sum(
                    item.get("classification")
                    == "electronic_prior_native_swae_admitted"
                    for item in confirmations
                )
                < 2
            ):
                raise ValueError(
                    "SWAE locked-final requires at least two admitted confirmations"
                )
            _stage_decision(
                args,
                "locked-final",
                f"{SCHEMA_PREFIX}-locked-final.v1",
                {
                    "confirm": file_sha256(args.confirm_evidence),
                    "checkpoint": file_sha256(args.checkpoint),
                },
            )
            if getattr(args, "_stage_decision", {}).get("decision") == "reuse":
                return 0
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
            classification = (
                "electronic_prior_native_swae_admitted"
                if measurement["passed"]
                else "prior_alignment_semantic_or_novelty_failed"
            )
            payload = {
                "schema": f"{SCHEMA_PREFIX}_locked-final.v1",
                "stage": "locked-final",
                "status": "valid",
                "passed": measurement["passed"],
                "classification": classification,
                "measurement": measurement,
                "confirm_evidence_sha256": [
                    file_sha256(args.confirm_evidence),
                    *[file_sha256(path) for path in args.peer_confirm_evidence],
                ],
                "formal_promotion": False,
                "optical_training_authorized": False,
                "fiber_training_authorized": False,
                "synthetic_diagnostic_only": True,
                "lineage_sha256": canonical_sha256(lineage),
                "producer_contract_revision": f"{SCHEMA_PREFIX}-locked-final.v1",
                "consumer_contract": f"{SCHEMA_PREFIX}-report.v1",
                "permitted_consumers": ["report"],
                "input_artifact_hashes": {
                    "confirm": file_sha256(args.confirm_evidence),
                    "checkpoint": file_sha256(args.checkpoint),
                },
            }
            payload.update(_evidence_invocation(args))
            _write(output, payload)
            return 0
        if action == "report":
            source = _load_json(args.measurement_evidence)
            if source.get("status") != "valid":
                raise ValueError("SWAE report requires valid measurement evidence")
            _stage_decision(
                args,
                "report",
                f"{SCHEMA_PREFIX}-report.v1",
                {"measurement": file_sha256(args.measurement_evidence)},
            )
            if getattr(args, "_stage_decision", {}).get("decision") == "reuse":
                return 0
            payload = {
                "schema": f"{SCHEMA_PREFIX}_presentation.v1",
                "stage": "report",
                "status": "valid",
                "passed": source.get("passed", False),
                "source_evidence_sha256": file_sha256(args.measurement_evidence),
                "classification": source.get("classification"),
                "measurement": source.get("measurement", {}),
                "synthetic_diagnostic_only": True,
                "producer_contract_revision": f"{SCHEMA_PREFIX}-report.v1",
                "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1",
                "permitted_consumers": ["closeout"],
                "input_artifact_hashes": {
                    "measurement": file_sha256(args.measurement_evidence)
                },
            }
            payload.update(_evidence_invocation(args))
            _write(output, payload)
            return 0
        if action == "closeout":
            _stage_decision(
                args,
                "closeout",
                f"{SCHEMA_PREFIX}-closeout.v1",
                {"measurement": file_sha256(args.measurement_evidence)},
            )
            if getattr(args, "_stage_decision", {}).get("decision") == "reuse":
                return 0
            return _closeout(args, args.measurement_evidence, output)
        raise RuntimeError(f"unsupported SWAE action: {action}")
    except Exception as exc:
        manifest = getattr(args, "_run_manifest", None)
        if manifest is not None:
            write_failure_record(manifest, stage=action, exc=exc, completed={})
        raise
