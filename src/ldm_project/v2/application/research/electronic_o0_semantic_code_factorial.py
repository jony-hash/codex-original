"""H-SCF: no-update semantic code-source factorial for electronic O0.

This module only measures frozen WAE/H-DIDM decoders under Gaussian and
encoded-posterior code banks.  It deliberately creates no optimizer,
checkpoint, or candidate model.  Its artifacts are diagnostic evidence for
the H-SCF closeout and hypothesis ledger only.
    """

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ...evidence import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...student.conditional import camera_code_fraction, relative_diversity_ratios
from ...student.conditional_acceptance import conditional_coverage_ratio
from ...student.evaluator import (
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
    load_local_mnist,
    stratified_mnist_indices,
)
from .electronic_o0_wae import _balanced_reference, panel_codes
from .lifecycle_artifacts import (
    prepare_args,
    stage_action_decision,
    validate_manifest_namespace,
)
from .wae_models import ConditionalWAEEncoder, ConditionalWAEDecoder


SCHEMA_PREFIX = "electronic_o0_semantic_code_factorial"
PROGRAM = "electronic-o0-semantic-code-factorial"
CONFIG_NAME = "electronic_o0_semantic_code_factorial_v1"
VERSION = 20260824
HARNESS_SEED = 8811
CODE_DIM = 8
CLASS_COUNT = 10
GAUSSIAN_BANK_SEEDS = (223607, 244949)
POSTERIOR_SPLIT_SEED = 1234
HISTORICAL_PANEL_SEEDS = (4242, 104729, 161803)
SAMPLES_PER_CLASS = 250
PREFLIGHT_SAMPLES_PER_CLASS = 32
ACCURACY_MINIMUM = 0.95
RECALL_MINIMUM = 0.90
AGREEMENT_MINIMUM = 0.95
CAMERA_CODE_MINIMUM = 0.25
DIVERSITY_MINIMUM = 0.70
DIVERSITY_MAXIMUM = 1.30
COVERAGE_MAXIMUM = 1.50
SATURATION_MAXIMUM = 0.08
MEMORIZATION_MINIMUM = 0.95
LABEL_SHIFT_MAXIMUM = 0.20
MEMORY_LIMIT_BYTES = 6 * 1024**3
STORAGE_LIMIT_BYTES = 64 * 1024**2
PREFLIGHT_WALL_LIMIT_SECONDS = 5 * 60
SOURCE_DECODER_SCHEMA = "electronic_o0_wae_decoder.v2"
TRAINING_SCHEMA = "electronic_o0_wae_training.v2"
DIDM_DECODER_SCHEMA = "electronic_o0_direct_image_distribution_decoder.v1"
LATENT_ATTRIBUTION_SCHEMA = "electronic_o0_latent_transport_attribution.v1"
DIDM_SCREEN_SCHEMA = "electronic_o0_direct_image_distribution_screen.v1"
DIDM_REPORT_SCHEMA = "electronic_o0_direct_image_distribution_report.v1"
DIDM_CLOSEOUT_SCHEMA = "electronic_o0_direct_image_distribution_closeout.v1"

REGISTERED_WAE_CONFIRM_HASH = "a92dbf237e551cd0240ca379b9d7781cbf1da196a9a53f83dca22fa975ffb5c4"
REGISTERED_WAE_TRAINING_HASH = "dbfecfe57560483d0db421b6105b718cd55c007fa109e39c278dc699839050ef"
REGISTERED_WAE_DECODER_HASH = "cfe326154487aa35850ddb440ecb38ef03be7ea56e2ff5f61d431b008de0e0f4"
REGISTERED_LATENT_ATTRIBUTION_HASH = "21ba7833c0c44f8d76649ed9c6f568c39d4885f1420d29e5b3addf56226612ad"
REGISTERED_H_DIDM_DECODER_HASH = "c066349955608e2c170f7732a7ecc63be4fb97f6673271373016544b10b5b6f2"
REGISTERED_H_DIDM_SCREEN_HASH = "f2ccc0cb7ce72c8e6866bac5fa48be3d873cd636ecc52e47f2eb31374041d369"
REGISTERED_H_DIDM_REPORT_HASH = "7e1626c98a089f4db632d0170bcb9f430cdad6e4f549ec45083c9d3ce2c7afa9"
REGISTERED_H_DIDM_CLOSEOUT_HASH = "d65a36cdc7637f76799c5bd763bf2ea51f1973bea55cd55ed7f6845bbb07fab3"
REGISTERED_PRIMARY_EVALUATOR_HASH = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
REGISTERED_SECONDARY_EVALUATOR_HASH = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
REGISTERED_PRIMARY_QUALIFICATION_HASH = "26fea381bec5d6be2a2ed69806c190f7a7051b5f0944e5f90941a7d93f35695e"
REGISTERED_SECONDARY_QUALIFICATION_HASH = "93592543306b5a3da8c238d86e41805fc2de0080459d217dfbc3e7161d374d88"
REGISTERED_TRAIN_POSTERIOR_HASH = "415dbee24b72d9aef9cc1b50e1e559011c0531e8ef38330f14882f778980e190"
REGISTERED_VALIDATION_POSTERIOR_HASH = "b2553bf61cbac285a015d90a2f7af76540f6f4a9e99301c9d4e451dc7d66cb23"


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
        "name": CONFIG_NAME,
        "version": VERSION,
        "code_dim": CODE_DIM,
        "harness_seed": HARNESS_SEED,
        "gaussian_bank_seeds": list(GAUSSIAN_BANK_SEEDS),
        "samples_per_class_per_bank": SAMPLES_PER_CLASS,
        "posterior_split_seed": POSTERIOR_SPLIT_SEED,
        "historical_panel_seeds": list(HISTORICAL_PANEL_SEEDS),
        "evaluator_accuracy_minimum": ACCURACY_MINIMUM,
        "evaluator_minimum_recall": RECALL_MINIMUM,
        "evaluator_agreement_minimum": AGREEMENT_MINIMUM,
        "camera_code_fraction_minimum": CAMERA_CODE_MINIMUM,
        "diversity_ratio_minimum": DIVERSITY_MINIMUM,
        "diversity_ratio_maximum": DIVERSITY_MAXIMUM,
        "coverage_ratio_maximum": COVERAGE_MAXIMUM,
        "saturation_maximum": SATURATION_MAXIMUM,
        "memorization_pass_fraction_minimum": MEMORIZATION_MINIMUM,
        "label_shift_accuracy_maximum": LABEL_SHIFT_MAXIMUM,
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != set(expected)
        or any(payload.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("H-SCF config is not the registered v1 contract")
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence is not a JSON object: {path}")
    return payload


def _all_finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _write(path: str | Path, payload: dict[str, Any], args: argparse.Namespace) -> None:
    payload["invocation_mode"] = "manifest"
    payload["legacy_diagnostic_only"] = False
    payload["run_manifest_sha256"] = file_sha256(args.run_manifest)
    atomic_write_json(path, payload)


def _require_cuda(args: argparse.Namespace) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("H-SCF requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H-SCF formal lifecycle requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _state_hash(module: nn.Module) -> str:
    return canonical_sha256(
        {name: value.detach().cpu() for name, value in module.state_dict().items()}
    )


def _load_decoder(path: str | Path, device: torch.device, *, didm_final: bool) -> ConditionalWAEDecoder:
    payload = safe_torch_load(path, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SOURCE_DECODER_SCHEMA
        or payload.get("candidate") != "O0-I"
        or payload.get("contains_encoder") is not False
        or payload.get("contains_optimizer") is not False
    ):
        raise ValueError("H-SCF decoder checkpoint is incompatible")
    if didm_final and payload.get("route_schema") != DIDM_DECODER_SCHEMA:
        raise ValueError("H-SCF final decoder is not an H-DIDM decoder")
    if not didm_final and payload.get("route_schema") is not None:
        raise ValueError("H-SCF source decoder must be the unmodified WAE decoder")
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    decoder.eval()
    return decoder


def _load_encoder(path: str | Path, device: torch.device) -> ConditionalWAEEncoder:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != TRAINING_SCHEMA:
        raise ValueError("H-SCF WAE training checkpoint is incompatible")
    if "encoder" not in payload or "optimizer" not in payload:
        raise ValueError("H-SCF WAE training checkpoint lacks the frozen encoder contract")
    encoder = ConditionalWAEEncoder().to(device)
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()
    return encoder


def _load_secondary(path: str | Path, device: torch.device) -> SmoothMNISTEvaluator:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind:
        raise ValueError("H-SCF secondary evaluator is incompatible")
    model = SmoothMNISTEvaluator().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.freeze()


def _encode(encoder: nn.Module, images: Tensor, labels: Tensor, device: torch.device) -> Tensor:
    rows = []
    with torch.no_grad():
        for start in range(0, images.shape[0], 256):
            rows.append(
                encoder(
                    images[start : start + 256].to(device),
                    labels[start : start + 256].to(device),
                ).cpu()
            )
    return torch.cat(rows, dim=0)


def _balanced_posterior_banks(
    codes: Tensor,
    labels: Tensor,
    samples_per_class: int,
) -> dict[str, tuple[Tensor, Tensor]]:
    rows_a, rows_b = [], []
    for label in range(CLASS_COUNT):
        choices = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
        if choices.numel() < samples_per_class * 2:
            raise ValueError("H-SCF posterior bank has too few examples per class")
        rows_a.append(choices[:samples_per_class])
        rows_b.append(choices[samples_per_class : 2 * samples_per_class])
    indices_a, indices_b = torch.cat(rows_a), torch.cat(rows_b)
    if set(indices_a.tolist()).intersection(indices_b.tolist()):
        raise ValueError("H-SCF posterior banks overlap")
    return {
        "posterior_a": (codes.index_select(0, indices_a), labels.index_select(0, indices_a)),
        "posterior_b": (codes.index_select(0, indices_b), labels.index_select(0, indices_b)),
    }


def _gaussian_bank(seed: int, samples_per_class: int, device: torch.device) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.arange(CLASS_COUNT, device=device).repeat_interleave(samples_per_class)
    codes = torch.randn(
        (labels.numel(), CODE_DIM), generator=generator, dtype=torch.float32
    ).to(device)
    return codes, labels


def _decode(
    decoder: nn.Module,
    codes: Tensor,
    labels: Tensor,
    device: torch.device,
    *,
    batch_size: int = 128,
) -> Tensor:
    rows = []
    with torch.no_grad():
        for start in range(0, codes.shape[0], batch_size):
            rows.append(
                decoder(
                    codes[start : start + batch_size].to(device),
                    labels[start : start + batch_size].to(device),
                ).clamp(0.0, 1.0)
            )
    result = torch.cat(rows, dim=0)
    if result.ndim != 4 or result.shape[1:] != (1, 28, 28) or not torch.isfinite(result).all():
        raise ValueError("H-SCF decoder produced invalid images")
    return result


def _eval_model(model: nn.Module, images: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    features, logits = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, images.shape[0], 128):
            feature, output = model(images[start : start + 128].to(device))
            features.append(feature.cpu())
            logits.append(output.cpu())
    return torch.cat(features), torch.cat(logits)


def _counts(prediction: Tensor, labels: Tensor) -> dict[str, Any]:
    correct = prediction == labels
    recall_counts = [int(correct[labels == label].sum()) for label in range(CLASS_COUNT)]
    recall = [count / float((labels == label).sum()) for label, count in enumerate(recall_counts)]
    return {
        "correct_count": int(correct.sum()),
        "accuracy": float(correct.float().mean()),
        "recall_counts": recall_counts,
        "recall": recall,
        "minimum_recall": min(recall),
    }


def _quartiles(values: Tensor) -> dict[str, float]:
    values = values.detach().cpu().to(torch.float32)
    return {
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
    }


def _nearest_distances(rows: Tensor, references: Tensor, *, chunk_size: int = 64) -> Tensor:
    values = []
    for start in range(0, rows.shape[0], chunk_size):
        block = rows[start : start + chunk_size]
        for inner_start in range(0, block.shape[0], 8):
            inner = block[inner_start : inner_start + 8]
            values.append(
                (inner[:, None, :] - references[None, :, :])
                .square()
                .sum(dim=2)
                .min(dim=1)
                .values
            )
    return torch.cat(values)


def _prepare_memorization_context(
    train_images: Tensor,
    train_labels: Tensor,
    validation_images: Tensor,
    validation_labels: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    train_cpu = train_images.detach().cpu().clamp(0, 1)
    train_u8 = (train_cpu * 255).round().to(torch.uint8)
    train_hashes = {tensor_sha256(row) for row in train_u8}
    train_flat = train_images.to(device).flatten(1).clamp(0, 1)
    validation_flat = validation_images.to(device).flatten(1).clamp(0, 1)
    device_train_labels = train_labels.to(device)
    device_validation_labels = validation_labels.to(device)
    class_train = tuple(
        train_flat[device_train_labels == label] for label in range(CLASS_COUNT)
    )
    thresholds = []
    for label in range(CLASS_COUNT):
        heldout = validation_flat[device_validation_labels == label]
        distances = _nearest_distances(heldout, class_train[label], chunk_size=64)
        thresholds.append(float(torch.quantile(distances, 0.05).detach().cpu()))
    return {
        "device": device,
        "class_train": class_train,
        "thresholds": thresholds,
        "train_hashes": train_hashes,
    }


def _memorization(
    generated: Tensor,
    generated_labels: Tensor,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    generated_cpu = generated.detach().cpu().clamp(0, 1)
    generated_u8 = (generated_cpu * 255).round().to(torch.uint8)
    exact = sum(tensor_sha256(row) in prepared["train_hashes"] for row in generated_u8)
    generated_flat = generated.to(prepared["device"]).flatten(1).clamp(0, 1)
    device_labels = generated_labels.to(prepared["device"])
    thresholds = list(prepared["thresholds"])
    class_pass_counts, class_pass = [], []
    for label in range(CLASS_COUNT):
        generated_rows = generated_flat[device_labels == label]
        generated_distances = _nearest_distances(
            generated_rows, prepared["class_train"][label], chunk_size=64
        )
        pass_count = int((generated_distances >= thresholds[label]).sum())
        class_pass_counts.append(pass_count)
        class_pass.append(pass_count / float(generated_rows.shape[0]))
    return {
        "exact_duplicate_count": exact,
        "per_class_thresholds": thresholds,
        "per_class_pass_counts": class_pass_counts,
        "per_class_pass_fraction": class_pass,
        "pass_fraction": sum(class_pass) / CLASS_COUNT,
        "passed": exact == 0 and min(class_pass) >= MEMORIZATION_MINIMUM,
    }


def _semantic_row(primary_logits: Tensor, secondary_logits: Tensor, labels: Tensor) -> dict[str, Any]:
    primary_pred, secondary_pred = primary_logits.argmax(1), secondary_logits.argmax(1)
    primary = _counts(primary_pred, labels)
    secondary = _counts(secondary_pred, labels)
    agreement_count = int((primary_pred == secondary_pred).sum())
    target_margin = []
    for logits in (primary_logits, secondary_logits):
        target = logits.gather(1, labels[:, None]).squeeze(1)
        others = logits.clone()
        others.scatter_(1, labels[:, None], float("-inf"))
        target_margin.append(_quartiles(target - others.max(dim=1).values))
    samples_per_class = min(int((labels == label).sum()) for label in range(CLASS_COUNT))
    accuracy_min_count = math.ceil(ACCURACY_MINIMUM * labels.numel())
    recall_min_count = math.ceil(RECALL_MINIMUM * samples_per_class)
    agreement_min_count = math.ceil(AGREEMENT_MINIMUM * labels.numel())
    individual = bool(
        primary["correct_count"] >= accuracy_min_count
        and secondary["correct_count"] >= accuracy_min_count
        and min(primary["recall_counts"]) >= recall_min_count
        and min(secondary["recall_counts"]) >= recall_min_count
    )
    agreement = bool(agreement_count >= agreement_min_count)
    return {
        "primary": primary,
        "secondary": secondary,
        "agreement_count": agreement_count,
        "agreement": agreement_count / float(labels.numel()),
        "target_margin": {"primary": target_margin[0], "secondary": target_margin[1]},
        "individual_semantic_gate": individual,
        "agreement_gate": agreement,
        "semantic_gate": individual and agreement,
        "prediction_histogram": {
            "primary": torch.bincount(primary_pred, minlength=CLASS_COUNT).tolist(),
            "secondary": torch.bincount(secondary_pred, minlength=CLASS_COUNT).tolist(),
        },
    }


def _measure_arm(
    decoder: nn.Module,
    primary: nn.Module,
    secondary: nn.Module,
    codes: Tensor,
    labels: Tensor,
    memorization_context: Mapping[str, Any],
    validation_images: Tensor,
    test_images: Tensor,
    test_labels: Tensor,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    views = _decode(decoder, codes, labels, device)
    primary_features, primary_logits = _eval_model(primary, views, device)
    secondary_features, secondary_logits = _eval_model(secondary, views, device)
    semantic = _semantic_row(primary_logits, secondary_logits, labels.cpu())
    samples_per_class = int(labels.numel() // CLASS_COUNT)
    chunk_probe = views.flatten(1).cpu()[:257]
    chunk_reference = validation_images.flatten(1).cpu()[:257]
    chunk_64 = _nearest_distances(chunk_probe, chunk_reference, chunk_size=64)
    chunk_257 = _nearest_distances(chunk_probe, chunk_reference, chunk_size=257)
    reference, _ = _balanced_reference(test_images, test_labels, seed=seed + 17, count=samples_per_class)
    primary_reference, _ = _eval_model(primary, reference, device)
    secondary_reference, _ = _eval_model(secondary, reference, device)
    primary_coverage = conditional_coverage_ratio(
        primary_features, views.cpu(), primary_reference, reference,
        labels.cpu(), epsilon=0.05, iterations=20, seed=seed,
    )
    secondary_coverage = conditional_coverage_ratio(
        secondary_features, views.cpu(), secondary_reference, reference,
        labels.cpu(), epsilon=0.05, iterations=20, seed=seed,
    )
    diversity = {
        "primary": [float(value) for value in relative_diversity_ratios(primary_features, primary_reference, labels.cpu())],
        "secondary": [float(value) for value in relative_diversity_ratios(secondary_features, secondary_reference, labels.cpu())],
    }
    shifted_labels = torch.roll(labels, samples_per_class)
    shifted = _decode(decoder, codes, shifted_labels, device)
    _, shifted_logits = _eval_model(primary, shifted, device)
    _, shifted_secondary_logits = _eval_model(secondary, shifted, device)
    shifted_accuracy = float((shifted_logits.argmax(1) == labels.cpu()).float().mean())
    shifted_secondary_accuracy = float(
        (shifted_secondary_logits.argmax(1) == labels.cpu()).float().mean()
    )
    zero = _decode(decoder, torch.zeros_like(codes), labels, device)
    _, zero_primary_logits = _eval_model(primary, zero, device)
    _, zero_secondary_logits = _eval_model(secondary, zero, device)
    replay = _decode(decoder, codes, labels, device)
    camera_fraction = float(camera_code_fraction(views, labels.to(device)))
    safety = {
        "camera_code_fraction": camera_fraction,
        "camera_code_gate": camera_fraction >= CAMERA_CODE_MINIMUM,
        "diversity": diversity,
        "diversity_gate": all(DIVERSITY_MINIMUM <= value <= DIVERSITY_MAXIMUM for values in diversity.values() for value in values),
        "coverage": {"primary": float(primary_coverage), "secondary": float(secondary_coverage)},
        "coverage_gate": float(primary_coverage) <= COVERAGE_MAXIMUM and float(secondary_coverage) <= COVERAGE_MAXIMUM,
        "saturation": float((views >= 1.0).float().mean()),
        "saturation_gate": float((views >= 1.0).float().mean()) <= SATURATION_MAXIMUM,
        "label_shift_primary_accuracy": shifted_accuracy,
        "label_shift_secondary_accuracy": shifted_secondary_accuracy,
        "label_shift_gate": (
            shifted_accuracy <= LABEL_SHIFT_MAXIMUM
            and shifted_secondary_accuracy <= LABEL_SHIFT_MAXIMUM
        ),
        "memorization": _memorization(views, labels, memorization_context),
        "memorization_gate": False,
        "exact_duplicate_gate": False,
    }
    safety["memorization_gate"] = bool(safety["memorization"]["pass_fraction"] >= MEMORIZATION_MINIMUM)
    safety["exact_duplicate_gate"] = bool(safety["memorization"]["exact_duplicate_count"] == 0)
    safety["safety_gate"] = bool(
        safety["camera_code_gate"] and safety["diversity_gate"] and safety["coverage_gate"]
        and safety["saturation_gate"] and safety["label_shift_gate"]
        and safety["memorization_gate"] and safety["exact_duplicate_gate"]
    )
    semantic.update({
        "seed": seed,
        "count": int(labels.numel()),
        "evaluator_finite": all(
            bool(torch.isfinite(value).all())
            for value in (
                primary_features, primary_logits, secondary_features, secondary_logits,
                primary_reference, secondary_reference, shifted_logits,
                shifted_secondary_logits, zero_primary_logits, zero_secondary_logits,
            )
        ),
        "view_hash": tensor_sha256(views),
        "zero_code_view_hash": tensor_sha256(zero),
        "zero_code_control": _semantic_row(
            zero_primary_logits, zero_secondary_logits, labels.cpu()
        ),
        "no_update_replay_exact": tensor_sha256(views) == tensor_sha256(replay),
        "chunk_64_257_exact": tensor_sha256(chunk_64) == tensor_sha256(chunk_257),
        "safety": safety,
    })
    semantic["finite"] = _all_finite(semantic)
    return semantic


def _posterior_permutation(
    codes: Tensor, labels: Tensor, samples_per_class: int
) -> Tensor:
    rows = []
    for label in range(CLASS_COUNT):
        start = label * samples_per_class
        rows.append(torch.roll(codes[start : start + samples_per_class], 1, dims=0))
    return torch.cat(rows, dim=0)


def _measure_permutation(
    decoder: nn.Module,
    primary: nn.Module,
    secondary: nn.Module,
    codes: Tensor,
    labels: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    views = _decode(decoder, _posterior_permutation(codes, labels, int(labels.numel() // CLASS_COUNT)), labels, device)
    _, primary_logits = _eval_model(primary, views, device)
    _, secondary_logits = _eval_model(secondary, views, device)
    row = _semantic_row(primary_logits, secondary_logits, labels.cpu())
    row["evaluator_finite"] = bool(
        torch.isfinite(primary_logits).all() and torch.isfinite(secondary_logits).all()
    )
    row["view_hash"] = tensor_sha256(views)
    row["finite"] = _all_finite(row)
    return row


def _module_hashes(context: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: _state_hash(context[name])
        for name in ("encoder", "source_decoder", "didm_decoder", "primary", "secondary")
    }


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    all_images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_index, validation_index = stratified_mnist_indices(all_labels, seed=POSTERIOR_SPLIT_SEED)
    train_images = all_images.index_select(0, train_index)
    train_labels = all_labels.index_select(0, train_index)
    validation_images = all_images.index_select(0, validation_index)
    validation_labels = all_labels.index_select(0, validation_index)
    test_images, test_labels = load_local_mnist(args.dataset_root, train=False)
    encoder = _load_encoder(args.wae_training_checkpoint, device)
    source_decoder = _load_decoder(args.wae_decoder_checkpoint, device, didm_final=False)
    didm_decoder = _load_decoder(args.h_scf_final_decoder_checkpoint, device, didm_final=True)
    primary, primary_payload = load_frozen_evaluator(args.primary_evaluator_checkpoint, device=device)
    secondary = _load_secondary(args.secondary_evaluator_checkpoint, device)
    train_codes = _encode(encoder, train_images, train_labels, device)
    validation_codes = _encode(encoder, validation_images, validation_labels, device)
    memorization_context = _prepare_memorization_context(
        train_images, train_labels, validation_images, validation_labels, device
    )
    return {
        "device": device,
        "encoder": encoder,
        "source_decoder": source_decoder,
        "didm_decoder": didm_decoder,
        "primary": primary,
        "secondary": secondary,
        "primary_payload": primary_payload,
        "train_images": train_images,
        "train_labels": train_labels,
        "validation_images": validation_images,
        "validation_labels": validation_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "train_codes": train_codes,
        "validation_codes": validation_codes,
        "memorization_context": memorization_context,
    }


def _panel_metric_counts(panel: Mapping[str, Any], *, samples_per_class: int) -> dict[str, Any]:
    def expected_counts(section: Mapping[str, Any]) -> dict[str, Any]:
        recall = section.get("recall")
        if not isinstance(recall, list) or len(recall) != CLASS_COUNT:
            raise ValueError("historical panel is missing per-class recall")
        return {
            "correct_count": int(round(float(section["accuracy"]) * CLASS_COUNT * samples_per_class)),
            "recall_counts": [int(round(float(value) * samples_per_class)) for value in recall],
        }

    primary = panel.get("primary")
    secondary = panel.get("secondary")
    if not isinstance(primary, Mapping) or not isinstance(secondary, Mapping):
        raise ValueError("historical panel is missing evaluator metrics")
    return {
        "primary": expected_counts(primary),
        "secondary": expected_counts(secondary),
        "agreement_count": int(round(float(panel["agreement"]) * CLASS_COUNT * samples_per_class)),
    }


def _observed_panel_metric_counts(
    primary: nn.Module,
    secondary: nn.Module,
    images: Tensor,
    labels: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    _, primary_logits = _eval_model(primary, images, device)
    _, secondary_logits = _eval_model(secondary, images, device)
    row = _semantic_row(primary_logits, secondary_logits, labels.cpu())
    return {
        "primary": {
            "correct_count": row["primary"]["correct_count"],
            "recall_counts": row["primary"]["recall_counts"],
        },
        "secondary": {
            "correct_count": row["secondary"]["correct_count"],
            "recall_counts": row["secondary"]["recall_counts"],
        },
        "agreement_count": row["agreement_count"],
    }


def _historical_reproduction(args: argparse.Namespace, context: Mapping[str, Any]) -> dict[str, Any]:
    latent = _load_json(args.latent_transport_attribution)
    didm = _load_json(args.h_didm_screen_evidence)
    rows = {}
    device = context["device"]
    validation_codes, validation_labels = context["validation_codes"], context["validation_labels"]
    for seed in HISTORICAL_PANEL_SEEDS:
        codes, labels = panel_codes(seed, device=device)
        source_gaussian = _decode(
            context["source_decoder"], codes, labels, device, batch_size=100
        )
        generator = torch.Generator(device="cpu").manual_seed(seed + 2718)
        selected = []
        for label in range(CLASS_COUNT):
            choices = torch.nonzero(validation_labels.cpu() == label, as_tuple=False).flatten()
            selected.append(choices.index_select(0, torch.randperm(choices.numel(), generator=generator)[:50]))
        empirical = validation_codes.index_select(0, torch.cat(selected)).to(device)
        source_posterior = _decode(
            context["source_decoder"], empirical, labels, device, batch_size=100
        )
        didm_images = _decode(
            context["didm_decoder"], codes, labels, device, batch_size=100
        )
        expected_panels = {
            "source_gaussian": latent["branches"]["P0"]["panels"][str(seed)],
            "source_posterior": latent["branches"]["Rencoded"]["panels"][str(seed)],
            "didm_gaussian": didm["evaluation"]["panels"][str(seed)],
        }
        observed_images = {
            "source_gaussian": source_gaussian,
            "source_posterior": source_posterior,
            "didm_gaussian": didm_images,
        }
        panel_rows = {}
        for name, images in observed_images.items():
            observed_metrics = _observed_panel_metric_counts(
                context["primary"], context["secondary"], images, labels, device
            )
            expected_metrics = _panel_metric_counts(
                expected_panels[name], samples_per_class=50
            )
            view_hash = tensor_sha256(images)
            expected_hash = expected_panels[name]["view_hash"]
            panel_rows[name] = {
                "observed": view_hash,
                "expected": expected_hash,
                "view_hash_match": view_hash == expected_hash,
                "metrics": {
                    "observed": observed_metrics,
                    "expected": expected_metrics,
                    "match": observed_metrics == expected_metrics,
                },
                "match": view_hash == expected_hash and observed_metrics == expected_metrics,
            }
        rows[str(seed)] = {
            **panel_rows,
        }
    return {
        "panels": rows,
        "all_matches": all(value["match"] for row in rows.values() for value in row.values()),
        "posterior_hashes": {
            "train": tensor_sha256(context["train_codes"]),
            "validation": tensor_sha256(context["validation_codes"]),
        },
    }


def _build_measurement(args: argparse.Namespace, *, samples_per_class: int, include_history: bool) -> dict[str, Any]:
    context = _load_context(args)
    before = _module_hashes(context)
    posterior_banks = _balanced_posterior_banks(
        context["validation_codes"], context["validation_labels"], samples_per_class
    )
    banks = {
        f"gaussian_{index}": _gaussian_bank(seed, samples_per_class, context["device"])
        for index, seed in enumerate(GAUSSIAN_BANK_SEEDS, start=1)
    }
    banks.update(posterior_banks)
    arms = {}
    for arm_name, decoder_key in (("S-G", "source_decoder"), ("S-P", "source_decoder"), ("D-G", "didm_decoder"), ("D-P", "didm_decoder")):
        source_names = ("gaussian_1", "gaussian_2") if arm_name.endswith("G") else ("posterior_a", "posterior_b")
        rows = {}
        for bank_name in source_names:
            codes, labels = banks[bank_name]
            rows[bank_name] = _measure_arm(
                context[decoder_key], context["primary"], context["secondary"], codes, labels,
                context["memorization_context"], context["validation_images"],
                context["test_images"], context["test_labels"], context["device"],
                GAUSSIAN_BANK_SEEDS[0] if bank_name == "gaussian_1" else GAUSSIAN_BANK_SEEDS[1] if bank_name == "gaussian_2" else 8821 if bank_name == "posterior_a" else 8831,
            )
        arms[arm_name] = {"banks": rows, "semantic_pass": all(row["semantic_gate"] for row in rows.values()), "safety_pass": all(row["safety"]["safety_gate"] for row in rows.values())}
    permutation = {}
    for bank_name in ("posterior_a", "posterior_b"):
        codes, labels = banks[bank_name]
        permutation[bank_name] = _measure_permutation(
            context["didm_decoder"], context["primary"], context["secondary"], codes, labels, context["device"]
        )
    after = _module_hashes(context)
    historical = _historical_reproduction(args, context) if include_history else {"all_matches": True, "panels": {}}
    return {
        "arms": arms,
        "controls": {
            "D-P-within-class-permutation": permutation,
            "permutation_semantic_pass": all(row["semantic_gate"] for row in permutation.values()),
        },
        "historical_reproduction": historical,
        "posterior_hashes": historical["posterior_hashes"] if include_history else {"train": tensor_sha256(context["train_codes"]), "validation": tensor_sha256(context["validation_codes"])},
        "optimizer_created": False,
        "evaluator_calls": len(arms) * 2 * 8 + len(permutation) * 2,
        "parameter_hashes": {"before": before, "after": after, "unchanged": before == after},
        "chunk_invariance": all(
            row["chunk_64_257_exact"]
            for arm in arms.values()
            for row in arm["banks"].values()
        ),
        "finite": _all_finite({"arms": arms, "controls": {"D-P-within-class-permutation": permutation}}),
    }


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    input_paths = {
        "wae_training": args.wae_training_checkpoint,
        "wae_decoder": args.wae_decoder_checkpoint,
        "wae_confirm": args.wae_confirm_evidence,
        "latent_attribution": args.latent_transport_attribution,
        "h_scf_final_decoder": args.h_scf_final_decoder_checkpoint,
        "h_didm_screen": args.h_didm_screen_evidence,
        "h_didm_report": args.h_didm_report_evidence,
        "h_didm_closeout": args.h_didm_closeout_evidence,
        "primary_evaluator": args.primary_evaluator_checkpoint,
        "secondary_evaluator": args.secondary_evaluator_checkpoint,
        "primary_qualification": args.primary_qualification_evidence,
        "secondary_qualification": args.secondary_qualification_evidence,
    }
    paths = tuple(input_paths.values())
    if not Path(args.dataset_root).is_dir() or any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError("H-SCF lineage input is missing")
    input_hashes = {name: file_sha256(path) for name, path in input_paths.items()}
    expected_hashes = {
        "wae_training": REGISTERED_WAE_TRAINING_HASH,
        "wae_decoder": REGISTERED_WAE_DECODER_HASH,
        "wae_confirm": REGISTERED_WAE_CONFIRM_HASH,
        "latent_attribution": REGISTERED_LATENT_ATTRIBUTION_HASH,
        "h_scf_final_decoder": REGISTERED_H_DIDM_DECODER_HASH,
        "h_didm_screen": REGISTERED_H_DIDM_SCREEN_HASH,
        "h_didm_report": REGISTERED_H_DIDM_REPORT_HASH,
        "h_didm_closeout": REGISTERED_H_DIDM_CLOSEOUT_HASH,
        "primary_evaluator": REGISTERED_PRIMARY_EVALUATOR_HASH,
        "secondary_evaluator": REGISTERED_SECONDARY_EVALUATOR_HASH,
        "primary_qualification": REGISTERED_PRIMARY_QUALIFICATION_HASH,
        "secondary_qualification": REGISTERED_SECONDARY_QUALIFICATION_HASH,
    }
    if any(input_hashes[name] != expected_hash for name, expected_hash in expected_hashes.items()):
        raise ValueError("H-SCF upstream artifact hash is not the registered immutable input")
    primary_qualification = _load_json(args.primary_qualification_evidence)
    secondary_qualification = _load_json(args.secondary_qualification_evidence)
    if primary_qualification.get("status") != "passed" or secondary_qualification.get("status") != "passed":
        raise ValueError("H-SCF evaluator qualification is not passed")
    wae_confirm = _load_json(args.wae_confirm_evidence)
    if wae_confirm.get("schema") != "electronic_o0_wae_confirm.v2" or wae_confirm.get("status") != "valid":
        raise ValueError("H-SCF WAE confirm evidence is invalid")
    latent = _load_json(args.latent_transport_attribution)
    if latent.get("schema") != LATENT_ATTRIBUTION_SCHEMA or latent.get("status") != "valid":
        raise ValueError("H-SCF latent attribution evidence is invalid")
    if latent.get("posterior", {}).get("train_hash") != REGISTERED_TRAIN_POSTERIOR_HASH:
        raise ValueError("H-SCF train posterior hash does not match registered lineage")
    if latent.get("posterior", {}).get("validation_hash") != REGISTERED_VALIDATION_POSTERIOR_HASH:
        raise ValueError("H-SCF validation posterior hash does not match registered lineage")
    didm_screen = _load_json(args.h_didm_screen_evidence)
    if (
        didm_screen.get("schema") != DIDM_SCREEN_SCHEMA
        or didm_screen.get("status") != "valid"
        or didm_screen.get("classification") != "direct_image_distribution_failed"
        or didm_screen.get("next_action") != "close_current_direct_image_distribution_route"
        or didm_screen.get("passed") is not False
    ):
        raise ValueError("H-SCF H-DIDM screen is not the closed valid failure")
    didm_report = _load_json(args.h_didm_report_evidence)
    didm_closeout = _load_json(args.h_didm_closeout_evidence)
    if didm_report.get("schema") != DIDM_REPORT_SCHEMA or didm_report.get("status") != "valid":
        raise ValueError("H-SCF H-DIDM report is invalid")
    if (
        didm_closeout.get("schema") != DIDM_CLOSEOUT_SCHEMA
        or didm_closeout.get("status") != "valid"
        or didm_closeout.get("classification") != "direct_image_distribution_failed"
        or didm_closeout.get("next_action") != "close_current_direct_image_distribution_route"
    ):
        raise ValueError("H-SCF H-DIDM closeout is invalid")
    expected_decoder_hash = didm_screen.get("evaluation", {}).get("checkpoint_sha256")
    if expected_decoder_hash != file_sha256(args.h_scf_final_decoder_checkpoint):
        raise ValueError("H-SCF final decoder does not match H-DIDM screen")
    all_images, all_labels = load_local_mnist(args.dataset_root, train=True)
    train_index, validation_index = stratified_mnist_indices(
        all_labels, seed=POSTERIOR_SPLIT_SEED
    )
    if train_index.numel() != 55000 or validation_index.numel() != 5000:
        raise ValueError("H-SCF MNIST split has the wrong cardinality")
    if any(
        int((all_labels.index_select(0, validation_index) == label).sum()) != 500
        for label in range(CLASS_COUNT)
    ):
        raise ValueError("H-SCF MNIST validation split is not class-balanced")
    return {
        "git": _git(),
        "config_sha256": file_sha256(args.config),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_files": {
            "train": tensor_sha256(all_images),
            "test": tensor_sha256(load_local_mnist(args.dataset_root, train=False)[0]),
        },
        "split": {
            "seed": POSTERIOR_SPLIT_SEED,
            "train_count": int(train_index.numel()),
            "validation_count": int(validation_index.numel()),
            "train_index_hash": tensor_sha256(train_index),
            "validation_index_hash": tensor_sha256(validation_index),
        },
        "inputs": input_hashes,
        "registered_posterior_hashes": {"train": latent["posterior"]["train_hash"], "validation": latent["posterior"]["validation_hash"]},
        "contract": config,
    }


def _check_consumer(path: str, action: str) -> dict[str, Any]:
    payload = _load_json(path)
    allowed = tuple(f"{SCHEMA_PREFIX}_{stage}.v1" for stage in ("validation", "preflight", "audit", "report"))
    if payload.get("schema") not in allowed or payload.get("status") != "valid" or action not in payload.get("permitted_consumers", ()):
        raise ValueError(f"invalid H-SCF {action} evidence")
    if payload.get("invocation_mode") != "manifest" or payload.get("legacy_diagnostic_only") is not False:
        raise ValueError("legacy H-SCF evidence")
    return payload


def _stage(args: argparse.Namespace, stage: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    return stage_action_decision(
        stage,
        producer_contract=f"{SCHEMA_PREFIX}-{stage}.v1",
        expected_inputs=expected,
        evidence=args.output,
    )


def _measurement_structure_valid(measurement: Mapping[str, Any]) -> bool:
    arms = measurement.get("arms")
    controls = measurement.get("controls")
    if not isinstance(arms, Mapping) or set(arms) != {"S-G", "S-P", "D-G", "D-P"}:
        return False
    if not isinstance(controls, Mapping):
        return False
    permutation = controls.get("D-P-within-class-permutation")
    if (
        not isinstance(permutation, Mapping)
        or set(permutation) != {"posterior_a", "posterior_b"}
        or not isinstance(controls.get("permutation_semantic_pass"), bool)
    ):
        return False
    expected_banks = {
        "S-G": {"gaussian_1", "gaussian_2"},
        "D-G": {"gaussian_1", "gaussian_2"},
        "S-P": {"posterior_a", "posterior_b"},
        "D-P": {"posterior_a", "posterior_b"},
    }
    for arm_name, bank_names in expected_banks.items():
        arm = arms[arm_name]
        if (
            not isinstance(arm, Mapping)
            or set(arm.get("banks", {})) != bank_names
            or not isinstance(arm.get("semantic_pass"), bool)
            or not isinstance(arm.get("safety_pass"), bool)
        ):
            return False
        for row in arm["banks"].values():
            if not isinstance(row, Mapping) or not isinstance(row.get("safety"), Mapping):
                return False
            safety = row["safety"]
            if not all(
                isinstance(safety.get(key), bool)
                for key in (
                    "camera_code_gate", "diversity_gate", "coverage_gate",
                    "saturation_gate", "label_shift_gate", "memorization_gate",
                    "exact_duplicate_gate", "safety_gate",
                )
            ):
                return False
            if not all(
                isinstance(row.get(key), bool)
                for key in ("semantic_gate", "individual_semantic_gate", "agreement_gate", "no_update_replay_exact", "chunk_64_257_exact", "finite", "evaluator_finite")
            ):
                return False
    for row in permutation.values():
        if not isinstance(row, Mapping) or not all(
            isinstance(row.get(key), bool)
            for key in ("semantic_gate", "individual_semantic_gate", "agreement_gate", "finite", "evaluator_finite")
        ):
            return False
    return True


def _measurement_harness_valid(measurement: Mapping[str, Any]) -> bool:
    if not _measurement_structure_valid(measurement):
        return False
    parameters = measurement.get("parameter_hashes")
    historical = measurement.get("historical_reproduction")
    if not isinstance(parameters, Mapping) or not isinstance(historical, Mapping):
        return False
    if (
        measurement.get("optimizer_created") is not False
        or parameters.get("unchanged") is not True
        or historical.get("all_matches") is not True
        or measurement.get("posterior_hash_match") is not True
        or measurement.get("finite") is not True
        or measurement.get("chunk_invariance") is not True
    ):
        return False
    return all(
        row["no_update_replay_exact"] and row["chunk_64_257_exact"]
        for arm in measurement["arms"].values()
        for row in arm["banks"].values()
    ) and all(row["finite"] for row in measurement["controls"]["D-P-within-class-permutation"].values())


def _classify(measurement: dict[str, Any]) -> tuple[str, str]:
    if measurement.get("status") == "invalid_harness" or not measurement.get("measurement_valid", True):
        return "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit"
    if not _measurement_harness_valid(measurement):
        return "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit"
    arms = measurement.get("arms", {})
    controls = measurement.get("controls", {})
    if any(not bank.get("safety", {}).get("safety_gate", False) for arm in arms.values() for bank in arm.get("banks", {}).values()):
        return "semantic_counterfactual_safety_failed", "close_frozen_decoder_derived_semantic_route"
    d_p = arms.get("D-P", {})
    d_p_banks = list(d_p.get("banks", {}).values())
    if d_p_banks and all(row.get("individual_semantic_gate") for row in d_p_banks) and not all(row.get("agreement_gate") for row in d_p_banks):
        return "generated_evaluator_consensus_failed", "register_generated_ood_measurement_counterfactual"
    if not d_p.get("semantic_pass", False):
        return "didm_decoder_semantic_reachability_failed", "close_frozen_decoder_derived_semantic_route"
    if not controls.get("permutation_semantic_pass", False):
        return "posterior_instance_pairing_dependency", "close_frozen_decoder_derived_semantic_route"
    d_g = arms.get("D-G", {})
    if not d_g.get("semantic_pass", False):
        return "gaussian_semantic_mass_failed", "register_no_optimizer_nonlinear_direct_capacity_canary"
    return "gaussian_semantic_stability_failed", "close_frozen_decoder_derived_semantic_route"


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("H-SCF lifecycle is manifest-only")
    action = args.research_action
    common = (
        "config", "dataset_root", "output", "wae_training_checkpoint", "wae_decoder_checkpoint",
        "wae_confirm_evidence", "latent_transport_attribution", "h_scf_final_decoder_checkpoint",
        "h_didm_screen_evidence", "h_didm_report_evidence", "h_didm_closeout_evidence",
        "primary_evaluator_checkpoint", "secondary_evaluator_checkpoint",
        "primary_qualification_evidence", "secondary_qualification_evidence",
    )
    required = {
        "validate": common,
        "preflight": common + ("validation_evidence",),
        "audit": common + ("preflight_evidence",),
        "report": ("config", "dataset_root", "output", "measurement_evidence"),
        "closeout": ("config", "dataset_root", "output", "measurement_evidence"),
    }
    fields = common + ("device", "validation_evidence", "preflight_evidence", "measurement_evidence")
    args = prepare_args(args, program=PROGRAM, action=action, required=required[action], defaults={"device": "cuda", "seed": HARNESS_SEED}, mixed_fields=fields)
    validate_manifest_namespace(args, action=action)
    config = _config(args.config)
    lineage = _lineage(args, config) if action in {"validate", "preflight", "audit"} else None
    if action == "validate":
        decision = _stage(args, "validate", {"config": file_sha256(args.config), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _write(args.output, {
            "schema": f"{SCHEMA_PREFIX}_validation.v1", "stage": "validate", "status": "valid",
            "passed": True, "measurement_valid": True, "lineage_sha256": canonical_sha256(lineage),
            "consumer_contract": f"{SCHEMA_PREFIX}-preflight.v1", "permitted_consumers": ["preflight"],
            "producer_contract_revision": f"{SCHEMA_PREFIX}-validation.v1", "input_artifact_hashes": lineage["inputs"],
        }, args)
        return 0
    if action == "preflight":
        validation = _check_consumer(args.validation_evidence, "preflight")
        if not validation.get("passed"):
            raise ValueError("H-SCF preflight requires passed validation")
        decision = _stage(args, "preflight", {"validation": file_sha256(args.validation_evidence), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.monotonic()
        result = _build_measurement(args, samples_per_class=PREFLIGHT_SAMPLES_PER_CLASS, include_history=True)
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(args.device))
        root = Path(args._run_manifest["output_root"])
        storage = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        result["posterior_hash_match"] = result["posterior_hashes"] == {"train": REGISTERED_TRAIN_POSTERIOR_HASH, "validation": REGISTERED_VALIDATION_POSTERIOR_HASH}
        valid = bool(
            _measurement_harness_valid(result)
            and peak < MEMORY_LIMIT_BYTES
            and storage <= STORAGE_LIMIT_BYTES
            and elapsed <= PREFLIGHT_WALL_LIMIT_SECONDS
        )
        _write(args.output, {
            "schema": f"{SCHEMA_PREFIX}_preflight.v1", "stage": "preflight", "status": "valid" if valid else "invalid_harness",
            "passed": valid, "measurement_valid": valid, "result": result, "peak_memory_bytes": peak,
            "storage_bytes": storage, "memory_gate": peak < MEMORY_LIMIT_BYTES, "storage_gate": storage <= STORAGE_LIMIT_BYTES,
            "wall_time_seconds": elapsed, "wall_time_gate": elapsed <= PREFLIGHT_WALL_LIMIT_SECONDS,
            "optimizer_created": False, "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1",
            "permitted_consumers": ["audit"] if valid else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-preflight.v1",
            "input_artifact_hashes": {"validation": file_sha256(args.validation_evidence), "lineage": canonical_sha256(lineage)},
        }, args)
        return 0
    if action == "audit":
        preflight = _check_consumer(args.preflight_evidence, "audit")
        if not preflight.get("passed"):
            raise ValueError("H-SCF audit requires passed preflight")
        decision = _stage(args, "audit", {"preflight": file_sha256(args.preflight_evidence), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        torch.cuda.reset_peak_memory_stats(args.device)
        result = _build_measurement(args, samples_per_class=SAMPLES_PER_CLASS, include_history=False)
        result["posterior_hash_match"] = result["posterior_hashes"] == {"train": REGISTERED_TRAIN_POSTERIOR_HASH, "validation": REGISTERED_VALIDATION_POSTERIOR_HASH}
        peak = int(torch.cuda.max_memory_allocated(args.device))
        root = Path(args._run_manifest["output_root"])
        storage = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        valid = bool(
            _measurement_harness_valid(result)
            and peak < MEMORY_LIMIT_BYTES
            and storage <= STORAGE_LIMIT_BYTES
        )
        result["measurement_valid"] = valid
        classification, next_action = _classify({**result, "measurement_valid": valid})
        if not valid:
            classification, next_action = "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit"
        _write(args.output, {
            "schema": f"{SCHEMA_PREFIX}_audit.v1", "stage": "audit", "status": "valid" if valid else "invalid_harness",
            "passed": bool(valid and classification not in {"invalid_harness"}), "measurement_valid": valid,
            "classification": classification, "next_action": next_action, "result": result,
            "peak_memory_bytes": peak, "storage_bytes": storage, "memory_gate": peak < MEMORY_LIMIT_BYTES,
            "storage_gate": storage <= STORAGE_LIMIT_BYTES, "optimizer_created": False,
            "consumer_contract": f"{SCHEMA_PREFIX}-report.v1", "permitted_consumers": ["report"] if valid else [],
            "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1",
            "input_artifact_hashes": {"preflight": file_sha256(args.preflight_evidence), "lineage": canonical_sha256(lineage)},
        }, args)
        return 0
    measurement = _check_consumer(args.measurement_evidence, action)
    decision = _stage(args, action, {"measurement": file_sha256(args.measurement_evidence)})
    if decision["action"] == "REUSE":
        return 0
    if action == "report":
        _write(args.output, {
            "schema": f"{SCHEMA_PREFIX}_report.v1", "stage": "report", "status": "valid",
            "passed": measurement.get("passed"), "measurement_valid": measurement.get("measurement_valid"),
            "classification": measurement.get("classification"), "next_action": measurement.get("next_action"),
            "measurement": args.measurement_evidence, "source_evidence_sha256": file_sha256(args.measurement_evidence),
            "input_artifact_hashes": {"measurement": file_sha256(args.measurement_evidence)},
            "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1", "permitted_consumers": ["closeout"],
            "producer_contract_revision": f"{SCHEMA_PREFIX}-report.v1",
        }, args)
        return 0
    _write(args.output, {
        "schema": f"{SCHEMA_PREFIX}_closeout.v1", "stage": "closeout", "status": "valid",
        "passed": measurement.get("passed"), "measurement_valid": measurement.get("measurement_valid"),
        "classification": measurement.get("classification"), "next_action": measurement.get("next_action"),
        "measurement": args.measurement_evidence, "source_evidence_sha256": file_sha256(args.measurement_evidence),
        "input_artifact_hashes": {"measurement": file_sha256(args.measurement_evidence)},
        "permitted_consumers": [], "producer_contract_revision": f"{SCHEMA_PREFIX}-closeout.v1",
    }, args)
    return 0
