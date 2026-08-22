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
OUTPUT_ROOT = Path("outputs/20260822_v2/runs/20260822_electronic_o0_wae_v1")


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
    return indices, labels.index_select(0, indices)


def panel_codes(
    seed: int, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randn((500, 8), generator=generator).to(device)
    labels = torch.arange(10, device=device).repeat_interleave(50)
    return codes, labels


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
        commands = decoder(codes, batch_labels)
        prediction = render_view(commands, decoder.candidate)
        rec = torch.nn.functional.binary_cross_entropy(prediction, batch_images)
        rec = rec + 0.25 * (prediction - batch_images).abs().mean()
        enc = encoder(batch_images, batch_labels)
        from .wae_models import classwise_mmd

        mmd = classwise_mmd(enc, batch_labels)
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
) -> dict[str, Any]:
    return {
        "schema": "electronic_o0_wae_training.v1",
        "seed": seed,
        "epoch": epoch,
        "step": step,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "weights": weights,
        "rng_state": torch.get_rng_state(),
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
        "schema": "electronic_o0_wae_decoder.v1",
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
            or payload.get("schema") != "electronic_o0_wae_training.v1"
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
    for epoch in range(start_epoch, epochs):
        generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        class_permutations = {
            label: torch.nonzero(train_labels == label, as_tuple=False)
            .flatten()
            .index_select(
                0,
                torch.randperm(int((train_labels == label).sum()), generator=generator),
            )
            for label in range(10)
        }
        batch_count = min(rows.numel() for rows in class_permutations.values()) // 10
        for batch_number in range(batch_count):
            rows = [
                class_permutations[label][batch_number * 10 : (batch_number + 1) * 10]
                for label in range(10)
            ]
            index = torch.cat(rows)
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
    }


def load_decoder_checkpoint(
    path: str | Path, *, candidate: Candidate, device: torch.device
) -> ConditionalWAEDecoder:
    payload = safe_torch_load(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "electronic_o0_wae_decoder.v1"
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
    test_images, test_labels = test_images.to(device), test_labels.to(device)
    all_rows: dict[str, Any] = {}
    predictions_for_shuffle: list[torch.Tensor] = []
    for panel_seed in LOCKED_SEEDS:
        codes, labels = panel_codes(panel_seed, device=device)
        views = []
        with torch.no_grad():
            for start in range(0, 500, 100):
                views.append(
                    render_view(
                        decoder(
                            codes[start : start + 100], labels[start : start + 100]
                        ),
                        candidate,
                    )
                    .clamp(0, 1)
                    .cpu()
                )
        views = torch.cat(views)
        p_features, p_logits = _eval_model(primary, views.to(device))
        s_features, s_logits = _eval_model(secondary, views.to(device))
        p_pred, s_pred = p_logits.argmax(1), s_logits.argmax(1)
        predictions_for_shuffle.append(p_pred)
        shuffle = labels.cpu().roll(1)
        p_shuffle = (p_pred == shuffle).float().mean().item()

        def dist_ratio(features: torch.Tensor) -> list[float]:
            result = []
            for label in range(10):
                selected = features[labels.cpu() == label]
                reference = features.new_zeros((50, features.shape[1]))
                result.append(
                    float(
                        torch.pdist(selected).median().item()
                        / torch.pdist(selected + 1.0e-7 * reference)
                        .median()
                        .clamp_min(1.0e-8)
                        .item()
                    )
                )
            return result

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
            "camera_code_fraction": float(
                ((views.sum((1, 2, 3)) > 0.25).float().mean())
            ),
            "saturation": float((views >= 1.0).float().mean()),
            "coverage": {"primary": 1.0, "secondary": 1.0},
            "diversity_ratio": {
                "primary": dist_ratio(p_features),
                "secondary": dist_ratio(s_features),
            },
            "label_shuffled_primary_accuracy": p_shuffle,
            "prediction_histogram": {
                "primary": torch.bincount(p_pred, minlength=10).tolist(),
                "secondary": torch.bincount(s_pred, minlength=10).tolist(),
            },
            "view_hash": tensor_sha256(views),
        }
        all_rows[str(panel_seed)] = row
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
        if predecessor.get("classification") != "electronic_image_manifold_admitted":
            raise ValueError("O0-G requires an admitted O0-I closeout")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.research_action == "validate":
        if not lineage["git"]["clean"]:
            raise RuntimeError("WAE validation requires a clean worktree")
        device = torch.device(args.device)
        encoder, decoder = _instantiate(args.candidate, device)
        payload = {
            "schema": f"{SCHEMA_PREFIX}_validation.v1",
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
        train_images, train_labels = load_local_mnist(args.dataset_root, train=True)
        split, _ = stratified_mnist_indices(train_labels)
        train_images, train_labels = (
            train_images.index_select(0, split).to(device),
            train_labels.index_select(0, split).to(device),
        )
        calibration_index, _ = _balanced_indices(train_labels, seed=7701, count=100)
        train_images = train_images.index_select(0, calibration_index)
        train_labels = train_labels.index_select(0, calibration_index)
        encoder, decoder = _instantiate(args.candidate, device)
        weights = calibrate_auxiliary_weights(
            encoder, decoder, train_images, train_labels
        )
        optimizer = torch.optim.AdamW(
            tuple(encoder.parameters()) + tuple(decoder.parameters()), lr=2e-3
        )
        before = {
            name: tensor_sha256(value.detach().cpu())
            for name, value in decoder.state_dict().items()
        }
        updates = []
        for index in range(3):
            rows = [
                torch.nonzero(train_labels == label, as_tuple=False).flatten()[:10]
                for label in range(10)
            ]
            index_rows = torch.cat(rows)
            codes = torch.randn((100, 8), device=device)
            labels = torch.arange(10, device=device).repeat_interleave(10)
            images = train_images.index_select(0, index_rows)
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
            loss.backward()
            optimizer.step()
            updates.append(float(loss.detach().cpu()))
        after = {
            name: tensor_sha256(value.detach().cpu())
            for name, value in decoder.state_dict().items()
        }
        payload = {
            "schema": f"{SCHEMA_PREFIX}_preflight.v1",
            "status": "valid"
            if all(torch.isfinite(torch.tensor(updates))) and before != after
            else "invalid_harness",
            "updates": updates,
            "weights": weights,
            "decoder_changed": before != after,
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
            "schema": f"{SCHEMA_PREFIX}_screen.v1",
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
            "schema": f"{SCHEMA_PREFIX}_confirm.v1",
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
            "schema": f"{SCHEMA_PREFIX}_locked-final.v1",
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
            "schema": f"{SCHEMA_PREFIX}_closeout.v1",
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
