"""Read-only prior-gradient conflict attribution for electronic O0 routes.

H-PGC consumes the immutable WAE, H-SWAE, and H-SCED seed-8601 checkpoints.
It computes objective gradients and local constrained directions without an
optimizer or evaluator.  The lifecycle is diagnostic-only and cannot reopen a
closed prior-native route.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from ...evidence import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...student.evaluator import load_local_mnist, stratified_mnist_indices
from .electronic_o0_prior_native_sced import (
    _ordered_pairwise_absolute_mean,
    classwise_sliced_energy_gaussian,
    fixed_projections as sced_projections,
    gaussian_absolute_moment,
)
from .electronic_o0_prior_native_swae import (
    classwise_sliced_wasserstein_quantile,
    gaussian_quantiles,
)
from .lifecycle_artifacts import (
    prepare_args,
    stage_action_decision,
    validate_manifest_namespace,
)
from .wae_models import (
    ConditionalWAEEncoder,
    ConditionalWAEDecoder,
    parameter_counts,
    render_view,
    sobel_l1,
)


SCHEMA_PREFIX = "electronic_o0_prior_gradient_conflict"
PROGRAM = "electronic-o0-prior-gradient-conflict"
AUDIT_SEED = 8731
TRAIN_SEED = 8601
SOURCE_TRAIN_SCHEMA = "electronic_o0_wae_training.v2"
SOURCE_DECODER_SCHEMA = "electronic_o0_wae_decoder.v2"
SWAE_TRAIN_SCHEMA = "electronic_o0_prior_native_swae_training.v1"
SCED_TRAIN_SCHEMA = "electronic_o0_prior_native_sced_training.v1"
CODE_DIM = 8
BATCH_SEEDS = (11, 23, 37, 51)
PRIOR_SEEDS = (1011, 1023, 1037, 1051)
PROJECTION_SEEDS = (271829, 314159)
FD_RELATIVE_STEP = 1.0e-4
CYCLE_CONFLICT_COSINE = 0.05
PROJECTED_RETENTION_AVAILABLE = 0.50
STRUCTURAL_RETENTION_UPPER_BOUND = 0.10
REQUIRED_CONFLICT_BATCHES = 3
EXPECTED_STEP = 625


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _git() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "clean": not status, "status": status}


def _require_cuda(args: argparse.Namespace) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError(
            "prior-gradient conflict attribution requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("prior-gradient conflict attribution requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("H-PGC config must be a mapping")
    expected = {
        "schema_version": 1,
        "code_dim": CODE_DIM,
        "logical_batch_size": 100,
        "batch_seeds": list(BATCH_SEEDS),
        "prior_seeds": list(PRIOR_SEEDS),
        "projection_seeds": list(PROJECTION_SEEDS),
        "audit_seed": AUDIT_SEED,
        "finite_difference_relative_step": FD_RELATIVE_STEP,
        "cycle_conflict_cosine": CYCLE_CONFLICT_COSINE,
        "projected_retention_available": PROJECTED_RETENTION_AVAILABLE,
        "structural_retention_upper_bound": STRUCTURAL_RETENTION_UPPER_BOUND,
        "required_conflict_batches": REQUIRED_CONFLICT_BATCHES,
        "source_step": EXPECTED_STEP,
        "train_seed": TRAIN_SEED,
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("H-PGC config is not the registered v1 contract")
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
    if payload.get("lineage_sha256"):
        payload.setdefault("input_artifact_hashes", {})["lineage"] = payload[
            "lineage_sha256"
        ]
    atomic_write_json(path, payload)


def _config_lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    files = [
        args.config,
        args.wae_training_checkpoint,
        args.wae_decoder_checkpoint,
        args.swae_training_checkpoint,
        args.swae_screen_evidence,
        args.swae_closeout_evidence,
        args.sced_training_checkpoint,
        args.sced_screen_evidence,
        args.sced_closeout_evidence,
    ]
    missing = [str(path) for path in files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("H-PGC lineage input is missing: " + ", ".join(missing))
    if not Path(args.dataset_root).is_dir():
        raise FileNotFoundError(f"H-PGC dataset root is missing: {args.dataset_root}")
    git = _git()
    if not git["clean"]:
        raise RuntimeError("H-PGC evidence requires a clean worktree")
    return {
        "schema": f"{SCHEMA_PREFIX}_lineage.v1",
        "config_sha256": file_sha256(args.config),
        "config_content": config,
        "source_training_sha256": file_sha256(args.wae_training_checkpoint),
        "source_decoder_sha256": file_sha256(args.wae_decoder_checkpoint),
        "swae_training_sha256": file_sha256(args.swae_training_checkpoint),
        "swae_screen_sha256": file_sha256(args.swae_screen_evidence),
        "swae_closeout_sha256": file_sha256(args.swae_closeout_evidence),
        "sced_training_sha256": file_sha256(args.sced_training_checkpoint),
        "sced_screen_sha256": file_sha256(args.sced_screen_evidence),
        "sced_closeout_sha256": file_sha256(args.sced_closeout_evidence),
        "git": git,
        "device": args.device,
        "audit_seed": AUDIT_SEED,
        "batch_seeds": list(BATCH_SEEDS),
        "prior_seeds": list(PRIOR_SEEDS),
        "projection_seeds": list(PROJECTION_SEEDS),
    }


def _check_route_evidence(
    screen_path: str,
    closeout_path: str,
    training_path: str,
    *,
    screen_schema: str,
    closeout_schema: str,
    expected_classification: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    screen = _load_json(screen_path)
    closeout = _load_json(closeout_path)
    training = screen.get("training")
    screen_step = screen.get("step")
    screen_seed = screen.get("seed")
    if screen_step is None and isinstance(training, dict):
        screen_step = training.get("step")
    if screen_seed is None and isinstance(training, dict):
        screen_seed = training.get("seed")
    if (
        screen.get("schema") != screen_schema
        or screen.get("status") != "valid"
        or screen_seed != TRAIN_SEED
        or screen_step != EXPECTED_STEP
    ):
        raise ValueError(f"invalid paired screen evidence: {screen_path}")
    referenced_training = screen.get("training_checkpoint")
    if referenced_training is not None and Path(referenced_training).resolve() != Path(training_path).resolve():
        raise ValueError(f"paired screen checkpoint does not match manifest: {screen_path}")
    declared_training_hash = screen.get("training_checkpoint_sha256")
    if declared_training_hash is not None and declared_training_hash != file_sha256(training_path):
        raise ValueError(f"paired screen checkpoint hash mismatch: {screen_path}")
    if (
        closeout.get("schema") != closeout_schema
        or closeout.get("status") != "valid"
        or closeout.get("classification") != expected_classification
        or not str(closeout.get("next_action", "")).startswith("close_current_prior_native_")
    ):
        raise ValueError(f"invalid paired closeout evidence: {closeout_path}")
    return screen, closeout


def _state_hash(module: nn.Module) -> str:
    return canonical_sha256(
        {key: tensor_sha256(value.detach().cpu()) for key, value in module.state_dict().items()}
    )


def _load_source_models(
    training_path: str | Path,
    decoder_path: str | Path,
    device: torch.device,
) -> tuple[ConditionalWAEEncoder, ConditionalWAEDecoder]:
    training = safe_torch_load(training_path, map_location="cpu")
    decoder_payload = safe_torch_load(decoder_path, map_location="cpu")
    if not isinstance(training, dict) or training.get("schema") != SOURCE_TRAIN_SCHEMA:
        raise ValueError("H-PGC source training checkpoint schema is invalid")
    if (
        not isinstance(decoder_payload, dict)
        or decoder_payload.get("schema") != SOURCE_DECODER_SCHEMA
        or decoder_payload.get("candidate") != "O0-I"
        or decoder_payload.get("contains_encoder") is not False
        or decoder_payload.get("contains_optimizer") is not False
    ):
        raise ValueError("H-PGC source decoder checkpoint is invalid")
    encoder = ConditionalWAEEncoder().to(device)
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    if parameter_counts(encoder, decoder)["total"] > 8_000_000:
        raise ValueError("H-PGC source parameter budget exceeded")
    encoder.load_state_dict(training["encoder"], strict=True)
    decoder.load_state_dict(training["decoder"], strict=True)
    reference = ConditionalWAEDecoder("O0-I")
    reference.load_state_dict(decoder_payload["decoder_state_dict"], strict=True)
    if _state_hash(decoder) != _state_hash(reference):
        raise ValueError("H-PGC source training and decoder checkpoints disagree")
    return encoder, decoder


def _load_final(
    path: str | Path,
    *,
    schema: str,
    source_hash: str,
    device: torch.device,
) -> tuple[ConditionalWAEEncoder, ConditionalWAEDecoder, dict[str, Any]]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"H-PGC paired checkpoint schema is invalid: {path}")
    if (
        payload.get("seed") != TRAIN_SEED
        or payload.get("epoch") != 1
        or payload.get("step") != EXPECTED_STEP
        or payload.get("batch_cursor") != 0
        or payload.get("source_training_checkpoint_sha256") != source_hash
        or payload.get("schedule_hash") is None
        or payload.get("objective_hash") is None
    ):
        raise ValueError(f"H-PGC paired checkpoint binding is invalid: {path}")
    encoder = ConditionalWAEEncoder().to(device)
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    encoder.load_state_dict(payload["encoder"], strict=True)
    decoder.load_state_dict(payload["decoder"], strict=True)
    return encoder, decoder, payload


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
        if choices.numel() < per_class:
            raise ValueError("H-PGC balanced batch lacks a class")
        rows.append(choices[torch.randperm(choices.numel(), generator=generator)[:per_class]])
    index = torch.cat(rows).to(images.device)
    return images.index_select(0, index), labels.index_select(0, index)


def _parameter_groups(
    encoder: ConditionalWAEEncoder, decoder: ConditionalWAEDecoder
) -> dict[str, tuple[nn.Parameter, ...]]:
    named = dict(encoder.named_parameters()) | {
        f"decoder.{key}": value for key, value in decoder.named_parameters()
    }
    groups = {
        "encoder_features": tuple(
            value for key, value in named.items() if key.startswith("features.")
        ),
        "encoder_label_projection": tuple(
            value
            for key, value in named.items()
            if key.startswith("label.") or key.startswith("projection.")
        ),
        "decoder_input_label": tuple(
            value
            for key, value in named.items()
            if key.startswith("decoder.label.") or key.startswith("decoder.input.")
        ),
        "decoder_block": tuple(
            value for key, value in named.items() if key.startswith("decoder.block.")
        ),
        "decoder_upsample_output": tuple(
            value
            for key, value in named.items()
            if key.startswith(("decoder.up1.", "decoder.up2.", "decoder.up3.", "decoder.output."))
        ),
    }
    if any(not values for values in groups.values()):
        raise RuntimeError("H-PGC parameter grouping is incomplete")
    return groups


def _flat(values: Iterable[torch.Tensor | None], *, device: torch.device) -> torch.Tensor:
    parts = [value.reshape(-1) for value in values if value is not None]
    return torch.cat(parts) if parts else torch.zeros(0, device=device)


def _gradient_tree(
    gradients: tuple[torch.Tensor | None, ...],
    params: tuple[nn.Parameter, ...],
    groups: dict[str, tuple[nn.Parameter, ...]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    by_id = {id(parameter): gradient for parameter, gradient in zip(params, gradients, strict=True)}
    result = {
        name: _flat(
            (
                torch.zeros_like(parameter)
                if by_id.get(id(parameter)) is None
                else by_id[id(parameter)]
                for parameter in values
            ),
            device=device,
        )
        for name, values in groups.items()
    }
    result["encoder_aggregate"] = torch.cat(
        (result["encoder_features"], result["encoder_label_projection"])
    )
    result["decoder_aggregate"] = torch.cat(
        (result["decoder_input_label"], result["decoder_block"], result["decoder_upsample_output"])
    )
    result["full_joint"] = torch.cat((result["encoder_aggregate"], result["decoder_aggregate"]))
    return result


def _terms(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor,
    *,
    alignment: str,
    projections: torch.Tensor,
) -> dict[str, torch.Tensor]:
    encoded = encoder(images, labels)
    prediction = render_view(decoder(encoded, labels), "O0-I")
    reconstruction = F.binary_cross_entropy(prediction, images) + 0.25 * sobel_l1(
        prediction, images
    )
    if alignment == "sced":
        prior_alignment = classwise_sliced_energy_gaussian(
            encoded, labels, projections=projections
        )
    elif alignment == "swae":
        prior_alignment = classwise_sliced_wasserstein_quantile(
            encoded, labels, projections=projections
        )
    else:
        raise ValueError(f"unknown H-PGC alignment: {alignment}")
    cycle_prediction = render_view(decoder(prior, labels), "O0-I")
    cycle = F.smooth_l1_loss(encoder(cycle_prediction, labels), prior)
    return {"reconstruction": reconstruction, "alignment": prior_alignment, "cycle": cycle}


def _class_alignment_value(
    encoded: torch.Tensor,
    *,
    alignment: str,
    projections: torch.Tensor,
) -> torch.Tensor:
    projected = encoded @ projections.t()
    if alignment == "sced":
        cross = 2.0 * gaussian_absolute_moment(projected).mean(0)
        within = _ordered_pairwise_absolute_mean(projected, dim=0)
        return (cross - within - 2.0 / math.sqrt(math.pi)).mean()
    if alignment == "swae":
        ordered = torch.sort(projected, dim=0).values
        target = gaussian_quantiles(
            encoded.shape[0], device=encoded.device, dtype=encoded.dtype
        )
        return (ordered - target).square().mean()
    raise ValueError(f"unknown H-PGC alignment: {alignment}")


def _weights(payload: dict[str, Any], *, alignment: str) -> tuple[float, float]:
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("H-PGC checkpoint weights are missing")
    key = "w_sced" if alignment == "sced" else "w_swd"
    if key not in weights or "w_cycle" not in weights:
        raise ValueError("H-PGC checkpoint weights are incomplete")
    return float(weights[key]), float(weights["w_cycle"])


def _dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("H-PGC gradient shapes do not match")
    return torch.dot(left, right)


def _rho(gradient: torch.Tensor, direction: torch.Tensor) -> float:
    denominator = gradient.norm() * direction.norm()
    if float(denominator) <= 1.0e-20:
        return 0.0
    return float((_dot(gradient, direction) / denominator).detach().cpu())


def _project_halfspaces(
    target: torch.Tensor, first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project target onto first/second non-positive halfspaces analytically."""
    candidates: list[tuple[torch.Tensor, tuple[str, ...]]] = []
    first_norm = _dot(first, first)
    second_norm = _dot(second, second)
    if float(first_norm) <= 1.0e-20 or float(second_norm) <= 1.0e-20:
        raise ValueError("H-PGC halfspace constraint has zero gradient")

    def add(candidate: torch.Tensor, active: tuple[str, ...]) -> None:
        tolerance = 1.0e-6 * max(1.0, float(candidate.norm().detach().cpu()))
        if float(_dot(first, candidate)) <= tolerance and float(_dot(second, candidate)) <= tolerance:
            candidates.append((candidate, active))

    add(target, ())
    first_lambda = torch.clamp(_dot(first, target) / first_norm, min=0.0)
    add(target - first_lambda * first, ("reconstruction",))
    second_lambda = torch.clamp(_dot(second, target) / second_norm, min=0.0)
    add(target - second_lambda * second, ("cycle",))
    gram = torch.stack(
        (torch.stack((first_norm, _dot(first, second))), torch.stack((_dot(first, second), second_norm)))
    )
    rhs = torch.stack((_dot(first, target), _dot(second, target)))
    if float(torch.linalg.det(gram).abs()) > 1.0e-20:
        multipliers = torch.linalg.solve(gram, rhs)
        if bool(torch.all(multipliers >= 0)):
            add(target - multipliers[0] * first - multipliers[1] * second, ("reconstruction", "cycle"))
    if not candidates:
        raise FloatingPointError("H-PGC halfspace projection found no feasible candidate")
    candidate, active = min(candidates, key=lambda item: float((item[0] - target).square().sum().detach().cpu()))
    retention = float((_dot(target, candidate) / _dot(target, target)).detach().cpu())
    return candidate, {"active_constraints": list(active), "alignment_retention": retention}


def _loss_values(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor,
    *,
    alignment: str,
    projections: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        values = _terms(encoder, decoder, images, labels, prior, alignment=alignment, projections=projections)
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def _parameter_delta(
    params: tuple[nn.Parameter, ...], direction: torch.Tensor, *, relative_step: float
) -> list[torch.Tensor]:
    parameter_norm = torch.sqrt(sum((parameter.detach().square().sum() for parameter in params), torch.zeros((), device=direction.device)))
    direction_norm = direction.norm()
    if float(direction_norm) <= 1.0e-20:
        raise ValueError("H-PGC finite-difference direction has zero norm")
    scale = relative_step * max(1.0, float(parameter_norm.detach().cpu())) / direction_norm
    deltas: list[torch.Tensor] = []
    offset = 0
    for parameter in params:
        count = parameter.numel()
        deltas.append((scale * direction[offset : offset + count]).view_as(parameter))
        offset += count
    if offset != direction.numel():
        raise ValueError("H-PGC parameter delta does not cover all parameters")
    return deltas


def _finite_difference(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    params: tuple[nn.Parameter, ...],
    images: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor,
    *,
    alignment: str,
    projections: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, Any]:
    deltas = _parameter_delta(params, direction, relative_step=FD_RELATIVE_STEP)
    original = [parameter.detach().clone() for parameter in params]
    before = _loss_values(encoder, decoder, images, labels, prior, alignment=alignment, projections=projections)
    with torch.no_grad():
        for index, (parameter, delta) in enumerate(zip(params, deltas, strict=True)):
            parameter.copy_(original[index] + delta)
    plus = _loss_values(encoder, decoder, images, labels, prior, alignment=alignment, projections=projections)
    with torch.no_grad():
        for index, (parameter, delta) in enumerate(zip(params, deltas, strict=True)):
            parameter.copy_(original[index] - delta)
    minus = _loss_values(encoder, decoder, images, labels, prior, alignment=alignment, projections=projections)
    with torch.no_grad():
        for index, parameter in enumerate(params):
            parameter.copy_(original[index])
    signs = {}
    for name in ("reconstruction", "alignment", "cycle"):
        central = plus[name] - minus[name]
        signs[name] = {
            "before": before[name],
            "plus": plus[name],
            "minus": minus[name],
            "central_difference": central,
            "finite": all(math.isfinite(value) for value in (before[name], plus[name], minus[name], central)),
        }
    return signs


def _gradient_measurement(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor,
    *,
    alignment: str,
    projections: torch.Tensor,
    weights: tuple[float, float],
) -> dict[str, Any]:
    params = tuple(encoder.parameters()) + tuple(decoder.parameters())
    groups = _parameter_groups(encoder, decoder)
    values = _terms(encoder, decoder, images, labels, prior, alignment=alignment, projections=projections)
    gradients = {
        name: _gradient_tree(
            torch.autograd.grad(values[name], params, retain_graph=True, allow_unused=True),
            params,
            groups,
            device=images.device,
        )
        for name in ("reconstruction", "alignment", "cycle")
    }
    w_alignment, w_cycle = weights
    fixed: dict[str, torch.Tensor] = {}
    projected: dict[str, torch.Tensor] = {}
    group_records: dict[str, Any] = {}
    for group_name in gradients["reconstruction"]:
        g_reconstruction = gradients["reconstruction"][group_name]
        g_alignment = gradients["alignment"][group_name]
        g_cycle = gradients["cycle"][group_name]
        fixed[group_name] = -(
            g_reconstruction + w_alignment * g_alignment + w_cycle * g_cycle
        )
        if float(g_alignment.norm()) <= 1.0e-20:
            projected[group_name] = torch.zeros_like(g_alignment)
            projection_record = {
                "active_constraints": [],
                "alignment_retention": 0.0,
                "alignment_zero": True,
            }
        else:
            projected[group_name], projection_record = _project_halfspaces(
                -g_alignment, g_reconstruction, g_cycle
            )
        group_records[group_name] = {
            "gradient_norms": {
                name: float(gradients[name][group_name].norm().detach().cpu())
                for name in ("reconstruction", "alignment", "cycle")
            },
            "fixed_rho": {
                name: _rho(gradients[name][group_name], fixed[group_name])
                for name in ("reconstruction", "alignment", "cycle")
            },
            "projected": projection_record,
        }
    params_flat = torch.cat([parameter.detach().reshape(-1) for parameter in params])
    finite = bool(all(torch.isfinite(value).all() for term in gradients.values() for value in term.values()))
    finite = finite and bool(torch.isfinite(params_flat).all())
    if not finite:
        raise FloatingPointError("H-PGC gradient measurement is non-finite")
    alignment_decoder_norm = float(gradients["alignment"]["decoder_aggregate"].norm().detach().cpu())
    finite_difference = {}
    for direction_name, direction in (
        ("reconstruction_negative", -gradients["reconstruction"]["full_joint"]),
        ("reconstruction_positive", gradients["reconstruction"]["full_joint"]),
        ("alignment_negative", -gradients["alignment"]["full_joint"]),
        ("alignment_positive", gradients["alignment"]["full_joint"]),
        ("cycle_negative", -gradients["cycle"]["full_joint"]),
        ("cycle_positive", gradients["cycle"]["full_joint"]),
        ("projected", projected["full_joint"]),
    ):
        finite_difference[direction_name] = _finite_difference(
            encoder,
            decoder,
            params,
            images,
            labels,
            prior,
            alignment=alignment,
            projections=projections,
            direction=direction,
        )
    classwise: list[dict[str, Any]] = []
    for class_id in range(10):
        mask = labels == class_id
        class_images = images[mask]
        class_labels = labels[mask]
        class_prior = prior[mask]
        class_encoded = encoder(class_images, class_labels)
        class_prediction = render_view(decoder(class_encoded, class_labels), "O0-I")
        class_reconstruction_value = F.binary_cross_entropy(
            class_prediction, class_images
        ) + 0.25 * sobel_l1(class_prediction, class_images)
        class_cycle_prediction = render_view(
            decoder(class_prior, class_labels), "O0-I"
        )
        class_terms = {
            "reconstruction": class_reconstruction_value,
            "alignment": _class_alignment_value(
                class_encoded,
                alignment=alignment,
                projections=projections,
            ),
            "cycle": F.smooth_l1_loss(
                encoder(class_cycle_prediction, class_labels), class_prior
            ),
        }
        class_gradients = {
            name: _gradient_tree(
                torch.autograd.grad(
                    class_terms[name],
                    params,
                    retain_graph=True,
                    allow_unused=True,
                ),
                params,
                groups,
                device=images.device,
            )
            for name in ("reconstruction", "alignment", "cycle")
        }
        class_alignment = class_gradients["alignment"]["full_joint"]
        class_reconstruction = class_gradients["reconstruction"]["full_joint"]
        class_cycle = class_gradients["cycle"]["full_joint"]
        if float(class_alignment.norm()) <= 1.0e-20:
            class_projected = torch.zeros_like(class_alignment)
            class_projection = {
                "active_constraints": [],
                "alignment_retention": 0.0,
                "alignment_zero": True,
            }
        else:
            class_projected, class_projection = _project_halfspaces(
                -class_alignment,
                class_reconstruction,
                class_cycle,
            )
        classwise.append(
            {
                "class": class_id,
                "alignment_gradient_norm": float(class_alignment.norm().detach().cpu()),
                "fixed_directional_derivatives": {
                    name: float(_dot(class_gradients[name]["full_joint"], fixed["full_joint"]).detach().cpu())
                    for name in ("reconstruction", "alignment", "cycle")
                },
                "projected_directional_derivatives": {
                    name: float(_dot(class_gradients[name]["full_joint"], class_projected).detach().cpu())
                    for name in ("reconstruction", "alignment", "cycle")
                },
                "projected_retention": class_projection["alignment_retention"],
            }
        )
    return {
        "alignment": alignment,
        "weights": {"alignment": w_alignment, "cycle": w_cycle},
        "losses": {name: float(value.detach().cpu()) for name, value in values.items()},
        "alignment_decoder_gradient_norm": alignment_decoder_norm,
        "groups": group_records,
        "classwise": classwise,
        "finite": finite,
        "finite_difference": finite_difference,
    }


def _gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1.0e-20:
        raise ValueError("H-PGC gradient cosine has zero norm")
    return float((_dot(left, right) / denominator).detach().cpu())


def _measurement_valid(records: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for index, record in enumerate(records):
        if not record["finite"]:
            failures.append(f"measurement_{index}_nonfinite")
        if record["alignment_decoder_gradient_norm"] > 1.0e-10:
            failures.append(f"measurement_{index}_alignment_decoder_gradient")
        groups = record["groups"]
        for group_name in ("encoder_aggregate", "decoder_aggregate"):
            for objective in ("reconstruction", "cycle"):
                if groups[group_name]["gradient_norms"][objective] <= 1.0e-12:
                    failures.append(
                        f"measurement_{index}_{group_name}_{objective}_zero"
                    )
        if groups["encoder_aggregate"]["gradient_norms"]["alignment"] <= 1.0e-12:
            failures.append(f"measurement_{index}_alignment_encoder_zero")
        for item in record.get("classwise", ()):
            for value in (
                item["alignment_gradient_norm"],
                item["projected_retention"],
                *item["fixed_directional_derivatives"].values(),
                *item["projected_directional_derivatives"].values(),
            ):
                if not math.isfinite(value):
                    failures.append(f"measurement_{index}_classwise_nonfinite")
        for name, direction in record["finite_difference"].items():
            for term, values in direction.items():
                if isinstance(values, dict) and not values.get("finite", False):
                    failures.append(f"measurement_{index}_{name}_{term}_nonfinite")
    return not failures, failures


def _directional_controls(records: list[dict[str, Any]]) -> bool:
    for record in records:
        fd = record["finite_difference"]
        for term, negative, positive in (
            ("reconstruction", fd["reconstruction_negative"], fd["reconstruction_positive"]),
            ("alignment", fd["alignment_negative"], fd["alignment_positive"]),
            ("cycle", fd["cycle_negative"], fd["cycle_positive"]),
        ):
            if not (
                negative[term]["plus"] < negative[term]["before"]
                and positive[term]["plus"] > positive[term]["before"]
            ):
                return False
    return True


def _classify(
    sced_records: list[dict[str, Any]],
    swae_records: list[dict[str, Any]],
    sced_cosines: list[float],
    *,
    controls_passed: bool,
) -> tuple[str, str, dict[str, Any]]:
    if not controls_passed:
        return "invalid_harness", "repair_hpgc_finite_difference_controls", {}
    sced_cycle_rho = [
        record["groups"]["full_joint"]["fixed_rho"]["cycle"]
        for record in sced_records
    ]
    swae_cycle_rho = [
        record["groups"]["full_joint"]["fixed_rho"]["cycle"]
        for record in swae_records
    ]
    sced_rho_by_bank = {
        seed: [
            record["groups"]["full_joint"]["fixed_rho"]["cycle"]
            for record in sced_records
            if int(record["projection_seed"]) == seed
        ]
        for seed in sorted({int(record["projection_seed"]) for record in sced_records})
    }
    swae_rho_by_bank = {
        seed: [
            record["groups"]["full_joint"]["fixed_rho"]["cycle"]
            for record in swae_records
            if int(record["projection_seed"]) == seed
        ]
        for seed in sorted({int(record["projection_seed"]) for record in swae_records})
    }
    sced_conflict = bool(sced_rho_by_bank) and all(
        sum(value >= CYCLE_CONFLICT_COSINE for value in values)
        >= REQUIRED_CONFLICT_BATCHES
        for values in sced_rho_by_bank.values()
    )
    retention = [
        record["groups"]["full_joint"]["projected"]["alignment_retention"]
        for record in sced_records
    ]
    retention_by_bank = {
        seed: [
            record["groups"]["full_joint"]["projected"]["alignment_retention"]
            for record in sced_records
            if int(record["projection_seed"]) == seed
        ]
        for seed in sorted({int(record["projection_seed"]) for record in sced_records})
    }
    projected_fd = [
        record["finite_difference"]["projected"]
        for record in sced_records
    ]
    projected_feasible_by_bank = {
        seed: sum(
            item["alignment"]["plus"] < item["alignment"]["before"]
            and item["reconstruction"]["plus"] <= item["reconstruction"]["before"]
            and item["cycle"]["plus"] <= item["cycle"]["before"]
            for item, record in zip(
                projected_fd,
                sced_records,
                strict=True,
            )
            if int(record["projection_seed"]) == seed
        )
        >= REQUIRED_CONFLICT_BATCHES
        for seed in sorted({int(record["projection_seed"]) for record in sced_records})
    }
    projected_feasible = bool(projected_feasible_by_bank) and all(
        projected_feasible_by_bank.values()
    )
    estimator_independent = bool(swae_rho_by_bank) and all(
        sum(value >= CYCLE_CONFLICT_COSINE for value in values)
        >= REQUIRED_CONFLICT_BATCHES
        for values in swae_rho_by_bank.values()
    )
    shared = {
        "sced_cycle_rho": sced_cycle_rho,
        "swae_cycle_rho": swae_cycle_rho,
        "sced_cycle_rho_by_bank": sced_rho_by_bank,
        "swae_cycle_rho_by_bank": swae_rho_by_bank,
        "alignment_retention": retention,
        "alignment_retention_by_bank": retention_by_bank,
        "projected_feasible": projected_feasible,
        "projected_feasible_by_bank": projected_feasible_by_bank,
        "alignment_estimator_independent": estimator_independent,
        "sced_specific": not estimator_independent,
    }
    if not sced_conflict:
        return "gradient_conflict_not_supported", "register_trajectory_or_curvature_attribution", shared
    if min(sced_cosines) < 0.90:
        return "invalid_harness", "repair_projection_gradient_stability", {
            "sced_cycle_rho": sced_cycle_rho,
            "alignment_retention": retention,
            "projected_feasible": projected_feasible,
        }
    bank_retention_medians = [
        statistics_median(values) for values in retention_by_bank.values()
    ]
    if projected_feasible and min(bank_retention_medians) >= PROJECTED_RETENTION_AVAILABLE:
        classification = "fixed_weight_conflict_projected_descent_available"
        next_action = "register_constraint_projected_gradient_hypothesis"
    elif max(bank_retention_medians) <= STRUCTURAL_RETENTION_UPPER_BOUND:
        classification = "local_structural_prior_cycle_incompatibility"
        next_action = "register_structural_prior_consistency_hypothesis"
    else:
        classification = "conflict_confirmed_resolution_underdetermined"
        next_action = "register_local_curvature_trust_region_attribution"
    return classification, next_action, shared


def statistics_median(values: list[float]) -> float:
    if not values:
        raise ValueError("H-PGC median requires values")
    return float(torch.tensor(values, dtype=torch.float64).median())


def _audit(args: argparse.Namespace, lineage: dict[str, Any], output: Path) -> None:
    _require_cuda(args)
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    source_hash = file_sha256(args.wae_training_checkpoint)
    _check_route_evidence(
        args.swae_screen_evidence,
        args.swae_closeout_evidence,
        args.swae_training_checkpoint,
        screen_schema="electronic_o0_prior_native_swae_screen.v1",
        closeout_schema="electronic_o0_prior_native_swae_closeout.v1",
        expected_classification="prior_alignment_fit_failed",
    )
    _check_route_evidence(
        args.sced_screen_evidence,
        args.sced_closeout_evidence,
        args.sced_training_checkpoint,
        screen_schema="electronic_o0_prior_native_sced_screen.v1",
        closeout_schema="electronic_o0_prior_native_sced_closeout.v1",
        expected_classification="prior_alignment_fit_failed",
    )
    source_encoder, source_decoder = _load_source_models(
        args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
    )
    swae_encoder, swae_decoder, swae_payload = _load_final(
        args.swae_training_checkpoint,
        schema=SWAE_TRAIN_SCHEMA,
        source_hash=source_hash,
        device=device,
    )
    sced_encoder, sced_decoder, sced_payload = _load_final(
        args.sced_training_checkpoint,
        schema=SCED_TRAIN_SCHEMA,
        source_hash=source_hash,
        device=device,
    )
    if (
        swae_payload.get("source_training_checkpoint_sha256")
        != sced_payload.get("source_training_checkpoint_sha256")
        or swae_payload.get("schedule_hash") != sced_payload.get("schedule_hash")
        or swae_payload.get("step") != sced_payload.get("step")
        or swae_payload.get("seed") != sced_payload.get("seed")
    ):
        raise ValueError("H-PGC paired final checkpoints do not share source/schedule contract")
    all_images, all_labels = load_local_mnist(args.dataset_root, train=True)
    rows, _ = stratified_mnist_indices(all_labels)
    images = all_images.index_select(0, rows)
    labels = all_labels.index_select(0, rows)
    model_specs = {
        "source": (source_encoder, source_decoder, "sced", sced_payload),
        "swae_final": (swae_encoder, swae_decoder, "swae", swae_payload),
        "sced_final": (sced_encoder, sced_decoder, "sced", sced_payload),
    }
    hashes_before = {
        name: {"encoder": _state_hash(encoder), "decoder": _state_hash(decoder)}
        for name, (encoder, decoder, _, _) in model_specs.items()
    }
    sced_records: list[dict[str, Any]] = []
    swae_records: list[dict[str, Any]] = []
    sced_cosines: list[float] = []
    all_records: list[dict[str, Any]] = []
    for model_name, (encoder, decoder, alignment, payload) in model_specs.items():
        weights = _weights(payload, alignment=alignment)
        for batch_seed, prior_seed in zip(BATCH_SEEDS, PRIOR_SEEDS, strict=True):
            batch_images, batch_labels = _balanced_batch(images, labels, seed=batch_seed)
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            prior = torch.randn(
                (batch_images.shape[0], CODE_DIM),
                generator=torch.Generator(device="cpu").manual_seed(prior_seed),
            ).to(device)
            banks = [sced_projections(seed=seed, device=device) for seed in PROJECTION_SEEDS]
            bank_records = []
            for bank_seed, projections in zip(PROJECTION_SEEDS, banks, strict=True):
                record = _gradient_measurement(
                    encoder,
                    decoder,
                    batch_images,
                    batch_labels,
                    prior,
                    alignment=alignment,
                    projections=projections,
                    weights=weights,
                )
                record["model"] = model_name
                record["batch_seed"] = batch_seed
                record["prior_seed"] = prior_seed
                record["projection_seed"] = bank_seed
                bank_records.append(record)
                all_records.append(record)
            target = sced_records if model_name == "sced_final" else swae_records if model_name == "swae_final" else None
            if target is not None:
                target.extend(bank_records)
    valid, validity_failures = _measurement_valid(all_records)
    controls_passed = _directional_controls(all_records)
    # The bank-cosine records use the complete alignment-gradient vectors from
    # the two measurements; reconstruct them from the stored directional data
    # is impossible, so the audit keeps a conservative scalar consistency gate.
    # The full gradient cosine is computed in _bank_alignment_cosine below.
    sced_cosines = _bank_alignment_cosines(
        sced_encoder,
        sced_decoder,
        images,
        labels,
        device=device,
        payload=sced_payload,
    )
    if min(sced_cosines) < 0.90:
        valid = False
        validity_failures.append("sced_projection_gradient_cosine")
    hashes_after = {
        name: {"encoder": _state_hash(encoder), "decoder": _state_hash(decoder)}
        for name, (encoder, decoder, _, _) in model_specs.items()
    }
    hashes_unchanged = hashes_before == hashes_after
    if not hashes_unchanged:
        valid = False
        validity_failures.append("parameter_hash_changed")
    peak = int(torch.cuda.max_memory_allocated(device))
    storage = _tree_size(Path(args._run_manifest["output_root"]))
    memory_gate = peak < 6 * 1024**3
    storage_gate = storage <= 2 * 1024**3
    if not memory_gate:
        valid = False
        validity_failures.append("cuda_peak_memory_gate")
    if not storage_gate:
        valid = False
        validity_failures.append("run_storage_gate")
    if not controls_passed:
        valid = False
        validity_failures.append("finite_difference_directional_control")
    classification, next_action, decision = _classify(
        sced_records,
        swae_records,
        sced_cosines,
        controls_passed=controls_passed,
    )
    if not valid:
        classification = "invalid_harness"
        next_action = "repair_hpgc_harness_and_reissue_lineage"
    payload = {
        "schema": f"{SCHEMA_PREFIX}_audit.v1",
        "stage": "audit",
        "status": "valid" if valid else "invalid_harness",
        "passed": valid,
        "classification": classification,
        "next_action": next_action,
        "measurement_valid": valid,
        "validity_failures": validity_failures,
        "controls_passed": controls_passed,
        "decision": decision,
        "paired_models": {
            "source": {"schema": SOURCE_TRAIN_SCHEMA, "step": 0},
            "swae_final": {"schema": SWAE_TRAIN_SCHEMA, "step": swae_payload["step"], "objective_hash": swae_payload["objective_hash"]},
            "sced_final": {"schema": SCED_TRAIN_SCHEMA, "step": sced_payload["step"], "objective_hash": sced_payload["objective_hash"]},
        },
        "sced_projection_gradient_cosines": sced_cosines,
        "records": all_records,
        "parameter_hashes_before": hashes_before,
        "parameter_hashes_after": hashes_after,
        "parameters_unchanged": hashes_unchanged,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "peak_memory_bytes": peak,
        "storage_bytes": storage,
        "memory_gate": memory_gate,
        "storage_gate": storage_gate,
        "input_artifact_hashes": {
            "validation": file_sha256(args.validation_evidence),
            "source_training": file_sha256(args.wae_training_checkpoint),
            "source_decoder": file_sha256(args.wae_decoder_checkpoint),
            "swae_training": file_sha256(args.swae_training_checkpoint),
            "swae_screen": file_sha256(args.swae_screen_evidence),
            "swae_closeout": file_sha256(args.swae_closeout_evidence),
            "sced_training": file_sha256(args.sced_training_checkpoint),
            "sced_screen": file_sha256(args.sced_screen_evidence),
            "sced_closeout": file_sha256(args.sced_closeout_evidence),
        },
        "consumer_contract": f"{SCHEMA_PREFIX}-report.v1",
        "permitted_consumers": ["report", "closeout"],
        "lineage_sha256": canonical_sha256(lineage),
        "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1",
    }
    _write(output, payload, args)


def _bank_alignment_cosines(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    payload: dict[str, Any],
) -> list[float]:
    alignment = "sced"
    params = tuple(encoder.parameters()) + tuple(decoder.parameters())
    results: list[float] = []
    for batch_seed, prior_seed in zip(BATCH_SEEDS, PRIOR_SEEDS, strict=True):
        batch_images, batch_labels = _balanced_batch(images, labels, seed=batch_seed)
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        prior = torch.randn((batch_images.shape[0], CODE_DIM), generator=torch.Generator(device="cpu").manual_seed(prior_seed)).to(device)
        vectors = []
        for seed in PROJECTION_SEEDS:
            terms = _terms(encoder, decoder, batch_images, batch_labels, prior, alignment=alignment, projections=sced_projections(seed=seed, device=device))
            gradients = torch.autograd.grad(terms["alignment"], params, retain_graph=False, allow_unused=True)
            vectors.append(torch.cat([torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1) for parameter, gradient in zip(params, gradients, strict=True)]))
        results.append(_gradient_cosine(vectors[0], vectors[1]))
    return results


def _stage(args: argparse.Namespace, stage: str, contract: str, expected: dict[str, Any]) -> dict[str, Any]:
    merged = dict(expected)
    if getattr(args, "_lineage_sha256", None) is not None:
        merged["lineage_sha256"] = args._lineage_sha256
    args._stage_decision = stage_action_decision(
        stage,
        producer_contract=contract,
        expected_inputs=merged,
        evidence=args.output,
    )
    return args._stage_decision


def _check_consumer(path: str, *, action: str, schemas: tuple[str, ...]) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("status") != "valid" or payload.get("schema") not in schemas:
        raise ValueError(f"{path} is not valid H-PGC {action} evidence")
    if action not in payload.get("permitted_consumers", ()):
        raise ValueError(f"{path} does not permit H-PGC consumer {action}")
    if payload.get("invocation_mode") != "manifest" or payload.get("legacy_diagnostic_only") is not False:
        raise ValueError(f"{path} is not manifest H-PGC evidence")
    return payload


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("H-PGC lifecycle is manifest-only")
    action = args.research_action
    common = (
        "config",
        "dataset_root",
        "output",
        "wae_training_checkpoint",
        "wae_decoder_checkpoint",
        "swae_training_checkpoint",
        "swae_screen_evidence",
        "swae_closeout_evidence",
        "sced_training_checkpoint",
        "sced_screen_evidence",
        "sced_closeout_evidence",
    )
    required: dict[str, tuple[str, ...]] = {
        "validate": common,
        "audit": common + ("validation_evidence",),
        "report": ("config", "dataset_root", "output", "measurement_evidence"),
        "closeout": ("config", "dataset_root", "output", "measurement_evidence"),
    }
    fields = common + ("device", "validation_evidence", "measurement_evidence")
    defaults = {"device": "cuda", "seed": AUDIT_SEED}
    args = prepare_args(
        args,
        program=PROGRAM,
        action=action,
        required=required[action],
        defaults=defaults,
        mixed_fields=fields,
    )
    validate_manifest_namespace(args, action=action)
    if action in {"validate", "audit"}:
        config = _config(args.config)
        lineage = _config_lineage(args, config)
        args._lineage_sha256 = canonical_sha256(lineage)
    else:
        config = None
        lineage = None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if action == "validate":
        decision = _stage(
            args,
            "validate",
            f"{SCHEMA_PREFIX}-validation.v1",
            {"source_training": file_sha256(args.wae_training_checkpoint)},
        )
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        payload = {
            "schema": f"{SCHEMA_PREFIX}_validation.v1",
            "stage": "validate",
            "status": "valid",
            "passed": True,
            "lineage_sha256": canonical_sha256(lineage),
            "producer_contract_revision": f"{SCHEMA_PREFIX}-validation.v1",
            "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1",
            "permitted_consumers": ["audit"],
            "input_artifact_hashes": {"source_training": file_sha256(args.wae_training_checkpoint)},
        }
        _write(output, payload, args)
        return 0
    if action == "audit":
        validation = _check_consumer(
            args.validation_evidence,
            action="audit",
            schemas=(f"{SCHEMA_PREFIX}_validation.v1",),
        )
        if not validation.get("passed"):
            raise ValueError("H-PGC audit requires passed validation")
        decision = _stage(
            args,
            "audit",
            f"{SCHEMA_PREFIX}-audit.v1",
            {
                "validation": file_sha256(args.validation_evidence),
                "source_training": file_sha256(args.wae_training_checkpoint),
            },
        )
        if decision["action"] == "REUSE":
            return 0
        _audit(args, lineage, output)
        return 0
    measurement = _check_consumer(
        args.measurement_evidence,
        action=action,
        schemas=(f"{SCHEMA_PREFIX}_audit.v1", f"{SCHEMA_PREFIX}_report.v1")
        if action == "closeout"
        else (f"{SCHEMA_PREFIX}_audit.v1",),
    )
    decision = _stage(
        args,
        action,
        f"{SCHEMA_PREFIX}-{action}.v1",
        {"measurement": file_sha256(args.measurement_evidence)},
    )
    if decision["action"] == "REUSE":
        return 0
    if action == "report":
        payload = {
            "schema": f"{SCHEMA_PREFIX}_report.v1",
            "stage": "report",
            "status": "valid",
            "passed": bool(measurement.get("passed")),
            "classification": measurement.get("classification"),
            "next_action": measurement.get("next_action"),
            "measurement_valid": measurement.get("measurement_valid"),
            "measurement": args.measurement_evidence,
            "source_evidence_sha256": file_sha256(args.measurement_evidence),
            "input_artifact_hashes": {"measurement": file_sha256(args.measurement_evidence)},
            "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1",
            "permitted_consumers": ["closeout"],
        }
        _write(output, payload, args)
        return 0
    payload = {
        "schema": f"{SCHEMA_PREFIX}_closeout.v1",
        "stage": "closeout",
        "status": "valid",
        "passed": bool(measurement.get("passed")),
        "classification": measurement.get("classification"),
        "next_action": measurement.get("next_action"),
        "measurement_valid": measurement.get("measurement_valid"),
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
        "measurement": args.measurement_evidence,
        "source_evidence_sha256": file_sha256(args.measurement_evidence),
        "input_artifact_hashes": {"measurement": file_sha256(args.measurement_evidence)},
        "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1",
        "permitted_consumers": [],
    }
    _write(output, payload, args)
    return 0
