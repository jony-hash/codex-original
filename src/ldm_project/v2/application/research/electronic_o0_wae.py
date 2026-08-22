"""Teacher-free electronic O0 WAE lifecycle.

This module is intentionally independent from the historical Objective-E O0
implementation.  It has one scientific purpose: decide whether the current
secondary failure is already present in the electronic image/command
manifold.  Evaluators are loaded only by the fixed measurement stages, never
by loss construction, auxiliary calibration, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...evidence import (
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...student.evaluator import (
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
    load_local_mnist,
    stratified_mnist_indices,
)
from ...student.conditional import camera_code_fraction, relative_diversity_ratios
from ...student.conditional_acceptance import conditional_coverage_ratio
from .wae_models import (
    FIXED_EXPOSURE_SCALE,
    ConditionalWAEEncoder,
    ConditionalWAEDecoder,
    Candidate,
    render_view,
    wae_loss,
    parameter_counts,
)

SCHEMA_PREFIX = "electronic_o0_wae"
TRAIN_SEEDS = (7301, 7302, 7303)
LOCKED_SEEDS = (4242, 104729, 161803)
BOOTSTRAP_SEED = 90210
OUTPUT_ROOT = Path("outputs/20260822_v2/runs/20260822_electronic_o0_wae_v2")


def _git() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "clean": not status, "status": status}


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 2
        or payload.get("harness_revision") != 2
        or payload.get("code_dim") != 8
        or payload.get("logical_batch_size") != 100
    ):
        raise ValueError(
            "WAE config must register code_dim=8 and logical_batch_size=100"
        )
    if payload.get("fixed_exposure_scale") != FIXED_EXPOSURE_SCALE:
        raise ValueError("WAE fixed exposure denominator is not the registered value")
    if tuple(payload.get("train_seeds", ())) != TRAIN_SEEDS:
        raise ValueError("WAE training seeds are fixed to 7301/7302/7303")
    return payload


def _require_files(paths: list[str | Path]) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("WAE lineage input is missing: " + ", ".join(missing))


def _require_lineage_files(paths: list[str | Path], dataset_root: str | Path) -> None:
    _require_files(paths)
    if not Path(dataset_root).is_dir():
        raise FileNotFoundError(f"MNIST dataset root is missing: {dataset_root}")


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _balanced_indices(
    labels: torch.Tensor, *, seed: int, count: int = 500
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for label in range(10):
        choices = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
        rows.append(
            choices.index_select(
                0, torch.randperm(choices.numel(), generator=generator)
            )[:count]
        )
    indices = torch.cat(rows)
    return indices, labels.index_select(0, indices.to(labels.device))


def panel_codes(
    seed: int, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randn((500, 8), generator=generator).to(device)
    labels = torch.arange(10, device=device).repeat_interleave(50)
    return codes, labels


def balanced_epoch_schedule(
    labels: torch.Tensor, *, seed: int, per_class: int = 10
) -> tuple[torch.Tensor, ...]:
    """Return a deterministic full-coverage class-balanced epoch schedule."""
    if labels.ndim != 1 or per_class < 1:
        raise ValueError("balanced schedule inputs are invalid")
    cpu_labels = labels.detach().cpu()
    class_rows: dict[int, torch.Tensor] = {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for label in range(10):
        choices = torch.nonzero(cpu_labels == label, as_tuple=False).flatten()
        if choices.numel() == 0:
            raise ValueError("balanced schedule requires every class")
        class_rows[label] = choices.index_select(
            0, torch.randperm(choices.numel(), generator=generator)
        )
    steps = max((rows.numel() + per_class - 1) // per_class for rows in class_rows.values())
    schedule = []
    for step in range(steps):
        rows = []
        for label in range(10):
            values = class_rows[label]
            start = step * per_class
            rows.append(values[(torch.arange(per_class) + start) % values.numel()])
        schedule.append(torch.cat(rows))
    return tuple(schedule)


def balanced_schedule_summary(
    labels: torch.Tensor, schedule: tuple[torch.Tensor, ...]
) -> dict[str, int | float]:
    flat = torch.cat(schedule)
    counts = torch.bincount(flat, minlength=labels.numel())
    return {
        "steps": len(schedule),
        "logical_samples": int(flat.numel()),
        "unique_samples": int((counts > 0).sum()),
        "minimum_coverage": int(counts[counts > 0].min()),
        "maximum_repeats": int(counts.max()),
        "repeat_samples": int((counts > 1).sum()),
    }


def calibrate_auxiliary_weights(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_seeds: tuple[int, ...] = (11, 23, 37, 51),
) -> dict[str, float]:
    """Calibrate auxiliary terms from four fixed batches, without evaluators."""
    if images.shape[0] < 100 or labels.shape[0] != images.shape[0]:
        raise ValueError("WAE calibration needs at least one logical balanced batch")
    values: list[tuple[float, float, float]] = []
    # MMD is intentionally an encoder-side regularizer.  Calibrating only
    # decoder parameters would report a zero norm and silently turn the
    # registered WAE objective into an unbounded weight.  The contract is a
    # full trainable-model gradient calibration; decoder-only state is still
    # the sole inference artifact.
    parameters = tuple(encoder.parameters()) + tuple(decoder.parameters())
    for seed in batch_seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows = []
        for label in range(10):
            choices = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
            rows.append(
                choices.index_select(
                    0, torch.randperm(choices.numel(), generator=generator)
                )[:10]
            )
        index = torch.cat(rows).to(images.device)
        batch_images, batch_labels = (
            images.index_select(0, index),
            labels.index_select(0, index),
        )
        codes = torch.randn(
            (100, 8), generator=torch.Generator(device="cpu").manual_seed(seed + 1000)
        ).to(images.device)
        # Recompute scalar terms so the gradient calibration is genuine and
        # does not rely on detached telemetry values.
        encoded = encoder(batch_images, batch_labels)
        commands = decoder(encoded, batch_labels)
        prediction = render_view(commands, decoder.candidate)
        rec = torch.nn.functional.binary_cross_entropy(prediction, batch_images)
        rec = rec + 0.25 * (prediction - batch_images).abs().mean()
        from .wae_models import classwise_mmd

        mmd = classwise_mmd(encoded, batch_labels)
        prior = torch.randn_like(codes)
        cycle = torch.nn.functional.smooth_l1_loss(
            encoder(
                render_view(decoder(prior, batch_labels), decoder.candidate),
                batch_labels,
            ),
            prior,
        )
        norms = []
        for term in (rec, mmd, cycle):
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
        values.append(tuple(norms))
    mean = torch.tensor(values, dtype=torch.float64).mean(0)
    if float(mean[0]) <= 0.0 or not torch.isfinite(mean).all():
        raise ValueError("WAE gradient calibration is non-finite")
    return {
        "w_mmd": float(0.25 * mean[0] / mean[1].clamp_min(1.0e-12)),
        "w_cycle": float(0.25 * mean[0] / mean[2].clamp_min(1.0e-12)),
        "batch_seeds": list(batch_seeds),
        "decoder_gradient_norms": {
            "reconstruction": float(mean[0]),
            "mmd": float(mean[1]),
            "cycle": float(mean[2]),
        },
    }


def _instantiate(
    candidate: Candidate, device: torch.device
) -> tuple[ConditionalWAEEncoder, ConditionalWAEDecoder]:
    encoder = ConditionalWAEEncoder().to(device)
    decoder = ConditionalWAEDecoder(candidate).to(device)
    counts = parameter_counts(encoder, decoder)
    if counts["total"] > 8_000_000 or counts["decoder"] > 6_000_000:
        raise ValueError("WAE parameter budget exceeded")
    return encoder, decoder


def _run_preflight_update(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    weights: dict[str, float],
) -> float:
    codes = torch.randn((images.shape[0], 8), device=images.device)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = wae_loss(
        decoder,
        encoder,
        images,
        labels,
        codes,
        w_mmd=weights["w_mmd"],
        w_cycle=weights["w_cycle"],
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("WAE preflight loss became non-finite")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _enable_deterministic_cuda() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _states_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _optimizer_state_equal(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    if left.keys() != right.keys() or left["param_groups"] != right["param_groups"]:
        return False
    left_state, right_state = left["state"], right["state"]
    if left_state.keys() != right_state.keys():
        return False
    for key in left_state:
        if left_state[key].keys() != right_state[key].keys():
            return False
        for name in left_state[key]:
            left_value, right_value = left_state[key][name], right_state[key][name]
            if isinstance(left_value, torch.Tensor):
                if not torch.equal(left_value, right_value):
                    return False
            elif left_value != right_value:
                return False
    return True


def _training_path(evidence_path: str | Path, seed: int) -> Path:
    path = Path(evidence_path)
    return path.with_name(f"seed_{seed}_training.pt")


def _inference_path(evidence_path: str | Path, seed: int) -> Path:
    path = Path(evidence_path)
    return path.with_name(f"seed_{seed}_decoder.pt")


def _checkpoint_payload(
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    weights: dict[str, float],
    seed: int,
    sampler_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "electronic_o0_wae_training.v2",
        "seed": seed,
        "epoch": epoch,
        "step": step,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "weights": weights,
        "rng_state": {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "sampler_state": sampler_state,
    }


def _save_inference(
    path: Path,
    decoder: ConditionalWAEDecoder,
    *,
    seed: int,
    epoch: int,
    source_checkpoint: Path,
) -> None:
    payload = {
        "schema": "electronic_o0_wae_decoder.v2",
        "candidate": decoder.candidate,
        "seed": seed,
        "epoch": epoch,
        "decoder_state_dict": decoder.state_dict(),
        "decoder_content_hash": canonical_sha256(
            {
                key: tensor_sha256(value.detach().cpu())
                for key, value in decoder.state_dict().items()
            }
        ),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "contains_encoder": False,
        "contains_optimizer": False,
    }
    atomic_torch_save(path, payload)


def _train(
    *,
    candidate: Candidate,
    dataset_root: str,
    device: torch.device,
    seed: int,
    epochs: int,
    evidence_path: str | Path,
    resume: str | None,
) -> tuple[Path, dict[str, Any]]:
    _enable_deterministic_cuda()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_images, train_labels = load_local_mnist(dataset_root, train=True)
    split_train, _ = stratified_mnist_indices(train_labels)
    train_images, train_labels = (
        train_images.index_select(0, split_train),
        train_labels.index_select(0, split_train),
    )
    encoder, decoder = _instantiate(candidate, device)
    optimizer = torch.optim.AdamW(
        tuple(encoder.parameters()) + tuple(decoder.parameters()), lr=2.0e-3
    )
    start_epoch, step = 0, 0
    calibration_index, _ = _balanced_indices(train_labels, seed=7701, count=100)
    weights = calibrate_auxiliary_weights(
        encoder,
        decoder,
        train_images.index_select(0, calibration_index).to(device),
        train_labels.index_select(0, calibration_index).to(device),
    )
    if resume:
        payload = safe_torch_load(resume)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "electronic_o0_wae_training.v2"
        ):
            raise ValueError("WAE resume checkpoint schema mismatch")
        encoder.load_state_dict(payload["encoder"], strict=True)
        decoder.load_state_dict(payload["decoder"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch, step, weights = (
            int(payload["epoch"]),
            int(payload["step"]),
            dict(payload["weights"]),
        )
        rng_state = payload.get("rng_state", {})
        if isinstance(rng_state, dict) and isinstance(rng_state.get("cpu"), torch.Tensor):
            torch.set_rng_state(rng_state["cpu"])
        if torch.cuda.is_available() and isinstance(rng_state, dict) and rng_state.get("cuda"):
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    for epoch in range(start_epoch, epochs):
        schedule = balanced_epoch_schedule(train_labels, seed=seed + epoch)
        for index in schedule:
            images = train_images.index_select(0, index).to(device, non_blocking=True)
            labels = train_labels.index_select(0, index).to(device, non_blocking=True)
            codes = torch.randn((100, 8), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = wae_loss(
                decoder,
                encoder,
                images,
                labels,
                codes,
                w_mmd=weights["w_mmd"],
                w_cycle=weights["w_cycle"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("WAE loss became non-finite")
            loss.backward()
            optimizer.step()
            step += 1
        training_path = _training_path(evidence_path, seed)
        atomic_torch_save(
            training_path,
            _checkpoint_payload(
                encoder,
                decoder,
                optimizer,
                epoch=epoch + 1,
                step=step,
                weights=weights,
                seed=seed,
                sampler_state={
                    "epoch_seed": seed + epoch,
                    "summary": balanced_schedule_summary(train_labels, schedule),
                    "cursor": len(schedule),
                },
            ),
        )
    inference = _inference_path(evidence_path, seed)
    _save_inference(
        inference,
        decoder,
        seed=seed,
        epoch=epochs,
        source_checkpoint=_training_path(evidence_path, seed),
    )
    return inference, {
        "seed": seed,
        "epoch": epochs,
        "steps": step,
        "weights": weights,
        "parameter_counts": parameter_counts(encoder, decoder),
        "training_checkpoint": str(_training_path(evidence_path, seed)),
        "inference_checkpoint": str(inference),
        "schedule": balanced_schedule_summary(train_labels, balanced_epoch_schedule(train_labels, seed=seed + epochs - 1)),
    }


def load_decoder_checkpoint(
    path: str | Path, *, candidate: Candidate, device: torch.device
) -> ConditionalWAEDecoder:
    payload = safe_torch_load(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "electronic_o0_wae_decoder.v2"
        or payload.get("candidate") != candidate
    ):
        raise ValueError("WAE decoder checkpoint is incompatible")
    if (
        any(key in payload for key in ("encoder", "optimizer"))
        or payload.get("contains_encoder")
        or payload.get("contains_optimizer")
    ):
        raise ValueError(
            "inference WAE checkpoint must not contain encoder or optimizer"
        )
    decoder = ConditionalWAEDecoder(candidate).to(device)
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    decoder.eval()
    return decoder


def _eval_model(
    model: nn.Module, images: torch.Tensor, batch_size: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    features, logits = [], []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            feature, output = model(images[start : start + batch_size])
            features.append(feature.cpu())
            logits.append(output.cpu())
    return torch.cat(features), torch.cat(logits)


def _recall(prediction: torch.Tensor, labels: torch.Tensor) -> list[float]:
    return [
        float((prediction[labels == label] == label).float().mean())
        for label in range(10)
    ]


def _bootstrap_ci(values: torch.Tensor, *, seed: int, samples: int = 10_000) -> list[float]:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    means = []
    for start in range(0, samples, 100):
        count = min(100, samples - start)
        indices = torch.randint(
            values.numel(), (count, values.numel()), generator=generator
        )
        means.append(values.cpu().index_select(0, indices.flatten()).reshape(count, -1).mean(1))
    result = torch.cat(means)
    return [float(torch.quantile(result, q)) for q in (0.025, 0.975)]


def _generate_views(
    decoder: ConditionalWAEDecoder,
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    candidate: Candidate,
    batch_size: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw, views = [], []
    with torch.no_grad():
        for start in range(0, codes.shape[0], batch_size):
            commands = decoder(
                codes[start : start + batch_size], labels[start : start + batch_size]
            )
            value = (
                commands
                if candidate == "O0-I"
                else torch.nn.functional.adaptive_avg_pool2d(
                    commands.square() / FIXED_EXPOSURE_SCALE, (28, 28)
                )
            )
            raw.append(value.cpu())
            views.append(value.clamp(0, 1).cpu())
    return torch.cat(raw), torch.cat(views)


def _balanced_reference(
    images: torch.Tensor, labels: torch.Tensor, *, seed: int, count: int = 50
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for label in range(10):
        choices = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
        if choices.numel() < count:
            raise ValueError("reference panel does not contain enough examples per class")
        rows.append(choices.index_select(0, torch.randperm(choices.numel(), generator=generator))[:count])
    index = torch.cat(rows)
    return images.index_select(0, index), labels.index_select(0, index)


def _jacobian_summary(
    decoder: ConditionalWAEDecoder,
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    candidate: Candidate,
) -> dict[str, Any]:
    rows = []
    for code, label in zip(codes, labels, strict=True):
        code = code.detach().requires_grad_(True)
        target = label.reshape(1)
        columns = []
        for direction in torch.eye(8, device=code.device):
            _, derivative = torch.autograd.functional.jvp(
                lambda value: render_view(
                    decoder(value.unsqueeze(0), target), candidate
                ).flatten(),
                code,
                direction,
                create_graph=False,
            )
            columns.append(derivative)
        singular = torch.linalg.svdvals(torch.stack(columns, dim=1))
        threshold_3 = singular[0].clamp_min(1.0e-12) * 1.0e-3
        threshold_8 = singular[0].clamp_min(1.0e-12) * 1.0e-8
        rows.append(
            {
                "rank_1e-3": int((singular >= threshold_3).sum()),
                "rank_1e-8": int((singular >= threshold_8).sum()),
                "condition": float((singular[0] / singular[-1].clamp_min(1.0e-12)).detach().cpu()),
                "singular_values": [float(value) for value in singular.detach().cpu()],
            }
        )
    return {
        "points": rows,
        "rank_1e-3_pass_fraction": float(sum(row["rank_1e-3"] >= 7 for row in rows) / len(rows)),
        "rank_1e-8_pass_fraction": float(sum(row["rank_1e-8"] >= 7 for row in rows) / len(rows)),
    }


def _memorization_audit(
    generated: torch.Tensor,
    generated_labels: torch.Tensor,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    generated = generated.flatten(1).to(device)
    train_images = train_images.flatten(1).to(device)
    validation_images = validation_images.flatten(1).to(device)
    thresholds, nearest_by_class, duplicates = [], {}, 0
    train_uint8 = (train_images.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    train_hashes = {tensor_sha256(row) for row in train_uint8}
    generated_uint8 = (generated.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    duplicates = sum(tensor_sha256(row) in train_hashes for row in generated_uint8)
    for label in range(10):
        train = train_images[train_labels.to(device) == label]
        heldout = validation_images[validation_labels.to(device) == label]
        query = generated[generated_labels.to(device) == label]
        heldout_distances = []
        for start in range(0, heldout.shape[0], 64):
            heldout_distances.append(torch.cdist(heldout[start : start + 64], train).square().min(1).values)
        threshold = float(torch.quantile(torch.cat(heldout_distances), 0.05).detach().cpu())
        generated_distances = []
        for start in range(0, query.shape[0], 64):
            generated_distances.append(torch.cdist(query[start : start + 64], train).square().min(1).values)
        nearest_by_class[label] = torch.cat(generated_distances).detach().cpu()
        thresholds.append(threshold)
    per_class_pass = []
    for label in range(10):
        nearest = nearest_by_class[label]
        per_class_pass.append(float((nearest >= thresholds[label]).float().mean()))
    return {
        "exact_duplicate_count": duplicates,
        "per_class_thresholds": thresholds,
        "per_class_pass_fraction": per_class_pass,
        "pass_fraction": float(sum(per_class_pass) / len(per_class_pass)),
        "passed": duplicates == 0 and min(per_class_pass) >= 0.95,
    }


def _measure_checkpoint(
    path: str | Path,
    *,
    candidate: Candidate,
    device: torch.device,
    dataset_root: str,
    primary_path: str,
    secondary_path: str,
) -> dict[str, Any]:
    decoder = load_decoder_checkpoint(path, candidate=candidate, device=device)
    primary, primary_payload = load_frozen_evaluator(primary_path, device=device)
    secondary_payload = safe_torch_load(secondary_path)
    if (
        not isinstance(secondary_payload, dict)
        or secondary_payload.get("architecture_kind")
        != SmoothMNISTEvaluator.architecture_kind
    ):
        raise ValueError("secondary evaluator qualification is invalid")
    secondary = SmoothMNISTEvaluator().to(device)
    secondary.load_state_dict(secondary_payload["state_dict"], strict=True)
    secondary.freeze()
    test_images, test_labels = load_local_mnist(dataset_root, train=False)
    train_images, train_labels = load_local_mnist(dataset_root, train=True)
    train_index, validation_index = stratified_mnist_indices(train_labels)
    all_images, all_labels = load_local_mnist(dataset_root, train=True)
    train_images, train_labels = all_images.index_select(0, train_index), all_labels.index_select(0, train_index)
    validation_images, validation_labels = all_images.index_select(0, validation_index), all_labels.index_select(0, validation_index)
    all_rows: dict[str, Any] = {}
    for panel_seed in LOCKED_SEEDS:
        codes, labels = panel_codes(panel_seed, device=device)
        raw_views, views = _generate_views(decoder, codes, labels, candidate=candidate)
        reference_images, reference_labels = _balanced_reference(test_images, test_labels, seed=panel_seed + 17)
        p_features, p_logits = _eval_model(primary, views.to(device))
        s_features, s_logits = _eval_model(secondary, views.to(device))
        p_ref_features, _ = _eval_model(primary, reference_images.to(device))
        s_ref_features, _ = _eval_model(secondary, reference_images.to(device))
        p_pred, s_pred = p_logits.argmax(1), s_logits.argmax(1)
        shuffle_generator = torch.Generator(device="cpu").manual_seed(8675309 + panel_seed)
        shuffled_labels = labels.cpu().index_select(0, torch.randperm(labels.numel(), generator=shuffle_generator))
        _, shuffled_views = _generate_views(decoder, codes, shuffled_labels.to(device), candidate=candidate)
        shuffled_prediction = _eval_model(primary, shuffled_views.to(device))[1].argmax(1)
        _, zero_views = _generate_views(
            decoder, torch.zeros_like(codes), labels, candidate=candidate
        )
        repeated_hash = tensor_sha256(_generate_views(decoder, codes, labels, candidate=candidate)[1]) == tensor_sha256(views)
        p_coverage = conditional_coverage_ratio(p_features, views, p_ref_features, reference_images, labels.cpu(), epsilon=0.05, iterations=20, seed=panel_seed)
        s_coverage = conditional_coverage_ratio(s_features, views, s_ref_features, reference_images, labels.cpu(), epsilon=0.05, iterations=20, seed=panel_seed)
        row = {
            "primary": {
                "accuracy": float((p_pred == labels.cpu()).float().mean()),
                "recall": _recall(p_pred, labels.cpu()),
                "minimum_recall": float(min(_recall(p_pred, labels.cpu()))),
            },
            "secondary": {
                "accuracy": float((s_pred == labels.cpu()).float().mean()),
                "recall": _recall(s_pred, labels.cpu()),
                "minimum_recall": float(min(_recall(s_pred, labels.cpu()))),
            },
            "agreement": float((p_pred == s_pred).float().mean()),
            "camera_code_fraction": float(camera_code_fraction(raw_views, labels.cpu())),
            "saturation": float((raw_views >= 1.0).float().mean()),
            "coverage": {"primary": p_coverage, "secondary": s_coverage},
            "diversity_ratio": {
                "primary": relative_diversity_ratios(p_features, p_ref_features, labels.cpu()),
                "secondary": relative_diversity_ratios(s_features, s_ref_features, labels.cpu()),
            },
            "label_shuffled_primary_accuracy": float((shuffled_prediction == labels.cpu()).float().mean()),
            "conditioning_label_shuffle_fraction": float((shuffled_labels != labels.cpu()).float().mean()),
            "prediction_histogram": {
                "primary": torch.bincount(p_pred, minlength=10).tolist(),
                "secondary": torch.bincount(s_pred, minlength=10).tolist(),
            },
            "view_hash": tensor_sha256(views),
            "zero_code_view_hash": tensor_sha256(zero_views),
            "no_update_replay_exact": repeated_hash,
            "bootstrap_accuracy_ci": {
                "primary": _bootstrap_ci((p_pred == labels.cpu()).float(), seed=BOOTSTRAP_SEED + panel_seed),
                "secondary": _bootstrap_ci((s_pred == labels.cpu()).float(), seed=BOOTSTRAP_SEED + panel_seed + 1),
            },
        }
        all_rows[str(panel_seed)] = row
    official_codes = torch.randn((test_labels.numel(), 8), generator=torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)).to(device)
    official_labels = test_labels.to(device)
    official_raw, official_views = _generate_views(decoder, official_codes, official_labels, candidate=candidate)
    official_p = _eval_model(primary, official_views.to(device))[1].argmax(1)
    official_s = _eval_model(secondary, official_views.to(device))[1].argmax(1)
    memory = _memorization_audit(
        official_views,
        official_labels.cpu(),
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        device=device,
    )
    official = {
        "count": int(test_labels.numel()),
        "primary_accuracy": float((official_p == official_labels.cpu()).float().mean()),
        "secondary_accuracy": float((official_s == official_labels.cpu()).float().mean()),
        "primary_minimum_recall": float(min(_recall(official_p, official_labels.cpu()))),
        "secondary_minimum_recall": float(min(_recall(official_s, official_labels.cpu()))),
        "agreement": float((official_p == official_s).float().mean()),
        "camera_code_fraction": float(camera_code_fraction(official_raw, official_labels.cpu())),
        "saturation": float((official_raw >= 1.0).float().mean()),
        "prediction_histogram": {"primary": torch.bincount(official_p, minlength=10).tolist(), "secondary": torch.bincount(official_s, minlength=10).tolist()},
    }
    jacobian_codes, jacobian_labels = [], []
    for panel_seed in LOCKED_SEEDS:
        codes, labels = panel_codes(panel_seed, device=device)
        for label in range(10):
            jacobian_codes.append(codes[label * 50])
            jacobian_labels.append(labels[label * 50])
    jacobian = _jacobian_summary(
        decoder,
        torch.stack(jacobian_codes),
        torch.stack(jacobian_labels),
        candidate=candidate,
    )
    secondary_gates = [
        row["secondary"]["accuracy"] >= 0.95
        and row["secondary"]["minimum_recall"] >= 0.90
        for row in all_rows.values()
    ]
    primary_gates = [
        row["primary"]["accuracy"] >= 0.95 and row["primary"]["minimum_recall"] >= 0.90
        for row in all_rows.values()
    ]
    gates = {
        "primary_accuracy": all(
            row["primary"]["accuracy"] >= 0.95 for row in all_rows.values()
        ),
        "secondary_accuracy": all(
            row["secondary"]["accuracy"] >= 0.95 for row in all_rows.values()
        ),
        "primary_minimum_recall": all(
            row["primary"]["minimum_recall"] >= 0.90 for row in all_rows.values()
        ),
        "secondary_minimum_recall": all(
            row["secondary"]["minimum_recall"] >= 0.90 for row in all_rows.values()
        ),
        "agreement": all(row["agreement"] >= 0.95 for row in all_rows.values()),
        "camera_code_fraction": all(
            row["camera_code_fraction"] >= 0.25 for row in all_rows.values()
        ),
        "saturation": all(row["saturation"] <= 0.08 for row in all_rows.values()),
        "label_shuffled": all(
            row["label_shuffled_primary_accuracy"] <= 0.20 for row in all_rows.values()
        ),
        "diversity": all(
            all(0.70 <= value <= 1.30 for value in row["diversity_ratio"][name])
            for row in all_rows.values()
            for name in ("primary", "secondary")
        ),
        "coverage": all(
            all(value <= 1.50 for value in row["coverage"].values())
            for row in all_rows.values()
        ),
    }
    return {
        "candidate": candidate,
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "evaluator_calls_during_training": 0,
        "primary_qualification": primary_payload.get("validation_accuracy"),
        "panels": all_rows,
        "gates": gates,
        "passed": all(gates.values()),
        "panel_pass_count": {
            "primary": sum(primary_gates),
            "secondary": sum(secondary_gates),
        },
        "official_test": official,
        "jacobian": jacobian,
        "memorization": memory,
        "test_set_count": int(test_labels.numel()),
        "test_set_bound": tensor_sha256(test_images.cpu()),
    }


def classify_o0(
    measurement: dict[str, Any], *, candidate: Candidate
) -> tuple[str, str]:
    if measurement.get("status") == "invalid_harness":
        return "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit"
    if candidate == "O0-I":
        return (
            ("electronic_image_manifold_admitted", "run_wae_o0_g")
            if measurement.get("passed")
            else ("wae_image_manifold_failed", "close_fiber_restart")
        )
    baseline = measurement.get("o0_i_baseline")
    if measurement.get("passed") and baseline is not None:
        return (
            "electronic_command_manifold_admitted",
            "register_smf_modal_density_attribution",
        )
    return "command_domain_bridge_failed", "stop_before_optics_or_fiber"


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    files = [
        args.config,
        args.primary_evaluator_checkpoint,
        args.secondary_evaluator_checkpoint,
        args.primary_qualification_evidence,
        args.secondary_qualification_evidence,
    ]
    _require_lineage_files(files, args.dataset_root)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("formal WAE lifecycle requires CUDA")
    return {
        "config_sha256": file_sha256(args.config),
        "primary_evaluator_sha256": file_sha256(args.primary_evaluator_checkpoint),
        "secondary_evaluator_sha256": file_sha256(args.secondary_evaluator_checkpoint),
        "primary_qualification_sha256": file_sha256(
            args.primary_qualification_evidence
        ),
        "secondary_qualification_sha256": file_sha256(
            args.secondary_qualification_evidence
        ),
        "config_content": config,
        "git": _git(),
        "candidate": args.candidate,
        "device": args.device,
        "seeds": {
            "train": list(TRAIN_SEEDS),
            "locked_panels": list(LOCKED_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
        },
    }


def dispatch(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    lineage = _lineage(args, config)
    if args.candidate == "O0-G":
        if not args.o0_i_closeout:
            raise ValueError("O0-G requires --o0-i-closeout")
        predecessor = json.loads(Path(args.o0_i_closeout).read_text(encoding="utf-8"))
        if (
            predecessor.get("schema") != f"{SCHEMA_PREFIX}_closeout.v2"
            or predecessor.get("status") != "valid"
            or predecessor.get("classification")
            != "electronic_image_manifold_admitted"
        ):
            raise ValueError("O0-G requires an admitted O0-I closeout")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.research_action == "validate":
        if not lineage["git"]["clean"]:
            raise RuntimeError("WAE validation requires a clean worktree")
        device = torch.device(args.device)
        encoder, decoder = _instantiate(args.candidate, device)
        payload = {
            "schema": f"{SCHEMA_PREFIX}_validation.v2",
            "status": "valid",
            "lineage": lineage,
            "parameter_counts": parameter_counts(encoder, decoder),
            "decoder_contract": {
                "candidate": args.candidate,
                "output": [28, 28] if args.candidate == "O0-I" else [200, 200],
                "fixed_exposure_scale": FIXED_EXPOSURE_SCALE,
            },
            "optimizer_created": False,
            "evaluator_calls": 0,
        }
        _write(output, payload)
        return 0
    if args.research_action == "preflight":
        validation = json.loads(
            Path(args.validation_evidence).read_text(encoding="utf-8")
        )
        if validation.get("status") != "valid":
            raise ValueError("WAE preflight requires valid validation evidence")
        device = torch.device(args.device)
        _enable_deterministic_cuda()
        train_images, train_labels = load_local_mnist(args.dataset_root, train=True)
        split, _ = stratified_mnist_indices(train_labels)
        train_images, train_labels = (
            train_images.index_select(0, split).to(device),
            train_labels.index_select(0, split).to(device),
        )
        calibration_index, _ = _balanced_indices(train_labels, seed=7701, count=100)
        calibration_index = calibration_index.to(device)
        train_images = train_images.index_select(0, calibration_index)
        train_labels = train_labels.index_select(0, calibration_index)
        encoder, decoder = _instantiate(args.candidate, device)
        weights = calibrate_auxiliary_weights(
            encoder, decoder, train_images, train_labels
        )
        optimizer = torch.optim.AdamW(
            tuple(encoder.parameters()) + tuple(decoder.parameters()), lr=2e-3
        )
        initial_encoder = _clone_state(encoder)
        initial_decoder = _clone_state(decoder)
        initial_cpu_rng = torch.get_rng_state()
        initial_cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        )
        before_encoder = {
            name: tensor_sha256(value.detach().cpu())
            for name, value in encoder.state_dict().items()
        }
        before_decoder = {
            name: tensor_sha256(value.detach().cpu())
            for name, value in decoder.state_dict().items()
        }
        updates = []
        rows = [
            torch.nonzero(train_labels == label, as_tuple=False).flatten()[:10]
            for label in range(10)
        ]
        index_rows = torch.cat(rows)
        images = train_images.index_select(0, index_rows)
        labels = torch.arange(10, device=device).repeat_interleave(10)
        for _ in range(3):
            updates.append(
                _run_preflight_update(encoder, decoder, optimizer, images, labels, weights)
            )
        after_encoder = {
            name: tensor_sha256(value.detach().cpu())
            for name, value in encoder.state_dict().items()
        }
        after_decoder = {
            name: tensor_sha256(value.detach().cpu())
            for name, value in decoder.state_dict().items()
        }
        true_images = train_images
        true_labels = train_labels
        shuffled_labels = true_labels.index_select(
            0,
            torch.randperm(
                true_labels.numel(),
                generator=torch.Generator(device="cpu").manual_seed(8675309),
            ).to(device),
        )
        with torch.no_grad():
            encoded = encoder(true_images, true_labels)
            true_prediction = render_view(
                decoder(encoded, true_labels), args.candidate
            )
            shuffled_prediction = render_view(
                decoder(encoder(true_images, shuffled_labels), shuffled_labels),
                args.candidate,
            )
            true_reconstruction = float(
                (true_prediction - true_images).square().mean().cpu()
            )
            shuffled_reconstruction = float(
                (shuffled_prediction - true_images).square().mean().cpu()
            )
        encoder_changed = before_encoder != after_encoder
        decoder_changed = before_decoder != after_decoder
        uninterrupted_encoder = _clone_state(encoder)
        uninterrupted_decoder = _clone_state(decoder)
        uninterrupted_optimizer = optimizer.state_dict()
        interrupted_encoder, interrupted_decoder = _instantiate(args.candidate, device)
        interrupted_encoder.load_state_dict(initial_encoder, strict=True)
        interrupted_decoder.load_state_dict(initial_decoder, strict=True)
        interrupted_optimizer = torch.optim.AdamW(
            tuple(interrupted_encoder.parameters())
            + tuple(interrupted_decoder.parameters()),
            lr=2e-3,
        )
        torch.set_rng_state(initial_cpu_rng)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(initial_cuda_rng)
        _run_preflight_update(
            interrupted_encoder,
            interrupted_decoder,
            interrupted_optimizer,
            images,
            labels,
            weights,
        )
        interrupted_rng = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        resumed_encoder, resumed_decoder = _instantiate(args.candidate, device)
        resumed_optimizer = torch.optim.AdamW(
            tuple(resumed_encoder.parameters()) + tuple(resumed_decoder.parameters()),
            lr=2e-3,
        )
        resumed_encoder.load_state_dict(_clone_state(interrupted_encoder), strict=True)
        resumed_decoder.load_state_dict(_clone_state(interrupted_decoder), strict=True)
        resumed_optimizer.load_state_dict(interrupted_optimizer.state_dict())
        torch.set_rng_state(interrupted_rng["cpu"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(interrupted_rng["cuda"])
        for _ in range(2):
            _run_preflight_update(
                resumed_encoder,
                resumed_decoder,
                resumed_optimizer,
                images,
                labels,
                weights,
            )
        replay_exact = (
            _states_equal(uninterrupted_encoder, _clone_state(resumed_encoder))
            and _states_equal(uninterrupted_decoder, _clone_state(resumed_decoder))
            and _optimizer_state_equal(
                uninterrupted_optimizer, resumed_optimizer.state_dict()
            )
        )
        payload = {
            "schema": f"{SCHEMA_PREFIX}_preflight.v2",
            "status": "valid"
            if all(torch.isfinite(torch.tensor(updates)))
            and encoder_changed
            and decoder_changed
            and true_reconstruction < shuffled_reconstruction
            and replay_exact
            else "invalid_harness",
            "updates": updates,
            "weights": weights,
            "encoder_changed": encoder_changed,
            "decoder_changed": decoder_changed,
            "true_reconstruction_mse": true_reconstruction,
            "shuffled_reconstruction_mse": shuffled_reconstruction,
            "interrupted_resumed_replay_exact": replay_exact,
            "evaluator_loaded": False,
            "evaluator_calls": 0,
            "lineage_sha256": canonical_sha256(lineage),
        }
        _write(output, payload)
        return 0
    if args.research_action == "screen":
        checkpoint, details = _train(
            candidate=args.candidate,
            dataset_root=args.dataset_root,
            device=torch.device(args.device),
            seed=args.seed,
            epochs=1,
            evidence_path=output,
            resume=args.resume_checkpoint,
        )
        payload = {
            "schema": f"{SCHEMA_PREFIX}_screen.v2",
            "status": "valid",
            "checkpoint": str(checkpoint),
            "details": details,
            "lineage_sha256": canonical_sha256(lineage),
            "evaluator_calls": 0,
        }
        _write(output, payload)
        return 0
    if args.research_action == "confirm":
        checkpoint, details = _train(
            candidate=args.candidate,
            dataset_root=args.dataset_root,
            device=torch.device(args.device),
            seed=args.seed,
            epochs=5,
            evidence_path=output,
            resume=args.resume_checkpoint,
        )
        measurement = _measure_checkpoint(
            checkpoint,
            candidate=args.candidate,
            device=torch.device(args.device),
            dataset_root=args.dataset_root,
            primary_path=args.primary_evaluator_checkpoint,
            secondary_path=args.secondary_evaluator_checkpoint,
        )
        classification, next_action = classify_o0(measurement, candidate=args.candidate)
        payload = {
            "schema": f"{SCHEMA_PREFIX}_confirm.v2",
            "status": "valid",
            "checkpoint": str(checkpoint),
            "training": details,
            "measurement": measurement,
            "classification": classification,
            "next_action": next_action,
            "lineage_sha256": canonical_sha256(lineage),
        }
        _write(output, payload)
        return 0
    if args.research_action == "locked-final":
        measurement = _measure_checkpoint(
            args.checkpoint,
            candidate=args.candidate,
            device=torch.device(args.device),
            dataset_root=args.dataset_root,
            primary_path=args.primary_evaluator_checkpoint,
            secondary_path=args.secondary_evaluator_checkpoint,
        )
        classification, next_action = classify_o0(measurement, candidate=args.candidate)
        payload = {
            "schema": f"{SCHEMA_PREFIX}_locked-final.v2",
            "status": "valid",
            "measurement": measurement,
            "classification": classification,
            "next_action": next_action,
            "lineage_sha256": canonical_sha256(lineage),
            "bootstrap": {"samples": 10000, "seed": BOOTSTRAP_SEED},
        }
        _write(output, payload)
        return 0
    if args.research_action == "closeout":
        locked = json.loads(Path(args.locked_evidence).read_text(encoding="utf-8"))
        classification = locked.get("classification", "invalid_harness")
        next_action = locked.get(
            "next_action", "repair_and_revalidate_on_a_new_clean_commit"
        )
        payload = {
            "schema": f"{SCHEMA_PREFIX}_closeout.v2",
            "status": locked.get("status", "invalid_harness"),
            "classification": classification,
            "next_action": next_action,
            "formal_promotion": False,
            "training_authorized": False,
            "synthetic_diagnostic_only": True,
            "locked_evidence_sha256": file_sha256(args.locked_evidence),
            "lineage_sha256": canonical_sha256(lineage),
        }
        _write(output, payload)
        return 0
    raise RuntimeError(f"unsupported WAE action {args.research_action}")
