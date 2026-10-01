"""Deterministic paired trajectory/curvature attribution for electronic O0.

H-TCA replays the already closed H-SWAE and H-SCED schedules on temporary
model copies.  It never exports a candidate model and never loads an
evaluator.  The final state of each replay is compared to the immutable
historical checkpoint before any attribution is accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.evidence import atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.student.evaluator import load_local_mnist, stratified_mnist_indices
from ldm_project.v2.application.research.capacity.electronic_o0_prior_native_sced import _ordered_pairwise_absolute_mean, classwise_sliced_energy_gaussian, fixed_projections as sced_projections, gaussian_absolute_moment
from ldm_project.v2.application.research.capacity.electronic_o0_prior_native_swae import classwise_sliced_wasserstein_quantile, gaussian_quantiles
from ldm_project.v2.application.research.capacity.electronic_o0_prior_native_sced import balanced_epoch_schedule
from ldm_project.v2.application.research.shared.lifecycle_artifacts import prepare_args, stage_action_decision, validate_manifest_namespace
from ldm_project.v2.application.research.capacity.wae_models import ConditionalWAEEncoder, ConditionalWAEDecoder, parameter_counts, render_view, sobel_l1


SCHEMA_PREFIX = "electronic_o0_prior_trajectory_curvature"
PROGRAM = "electronic-o0-prior-trajectory-curvature"
HARNESS_SEED = 8741
TRAIN_SEED = 8601
EXPECTED_STEP = 625
CODE_DIM = 8
BATCH_SEEDS = (11, 23, 37, 51)
PRIOR_SEEDS = (1011, 1023, 1037, 1051)
PROJECTION_SEEDS = (271829, 314159)
PROJECTION_COUNT = 256
LANDMARK_STEPS = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 625)
LEARNING_RATE = 2.0e-4
FD_RELATIVE_STEP = 1.0e-4
REQUIRED_SUPPORT_LANDMARKS = 2
REQUIRED_OBSERVER_BATCHES = 3
CYCLE_CONFLICT_COSINE = 0.05
CURVATURE_REMAINDER_FRACTION = 0.50
SCED_SCHEMA = "electronic_o0_prior_native_sced_training.v1"
SWAE_SCHEMA = "electronic_o0_prior_native_swae_training.v1"
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


def _require_cuda(args: argparse.Namespace) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("H-TCA requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H-TCA requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("H-TCA config must be a mapping")
    expected = {
        "schema_version": 1,
        "name": "electronic_o0_prior_trajectory_curvature_v1",
        "version": 20260824,
        "code_dim": CODE_DIM,
        "logical_batch_size": 100,
        "per_class": 10,
        "trajectory_seed": HARNESS_SEED,
        "historical_seed": TRAIN_SEED,
        "historical_steps": EXPECTED_STEP,
        "landmark_steps": list(LANDMARK_STEPS),
        "batch_seeds": list(BATCH_SEEDS),
        "prior_seeds": list(PRIOR_SEEDS),
        "projection_seeds": list(PROJECTION_SEEDS),
        "finite_difference_relative_step": FD_RELATIVE_STEP,
        "cycle_conflict_cosine": CYCLE_CONFLICT_COSINE,
        "required_support_landmarks": REQUIRED_SUPPORT_LANDMARKS,
        "required_observer_batches": REQUIRED_OBSERVER_BATCHES,
        "curvature_remainder_fraction": 0.50,
        "cycle_delta_absolute_tolerance": 1.0e-7,
        "cycle_delta_relative_tolerance": 1.0e-4,
        "parameter_budget": 8_000_000,
        "decoder_parameter_budget": 6_000_000,
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("H-TCA config is not the registered v1 contract")
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


def _state_hash(module: nn.Module) -> str:
    return canonical_sha256(
        {key: tensor_sha256(value.detach().cpu()) for key, value in module.state_dict().items()}
    )


def _state_dict_hash(state: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            key: tensor_sha256(value.detach().cpu())
            for key, value in state.items()
            if torch.is_tensor(value)
        }
    )


def _canonical_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"tensor_sha256": tensor_sha256(value.detach().cpu())}
    if isinstance(value, dict):
        return {str(key): _canonical_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_tree(item) for item in value]
    return value


def _tree_hash(value: Any) -> str:
    return canonical_sha256(_canonical_tree(value))


def _rng_hash() -> dict[str, Any]:
    return {
        "cpu": tensor_sha256(torch.get_rng_state().cpu()),
        "cuda": [tensor_sha256(item.cpu()) for item in torch.cuda.get_rng_state_all()],
    }


def _load_source_models(
    training_path: str | Path, decoder_path: str | Path, device: torch.device
) -> tuple[ConditionalWAEEncoder, ConditionalWAEDecoder]:
    training = safe_torch_load(training_path, map_location="cpu")
    decoder_payload = safe_torch_load(decoder_path, map_location="cpu")
    if not isinstance(training, dict) or training.get("schema") != SOURCE_TRAIN_SCHEMA:
        raise ValueError("H-TCA source training schema is invalid")
    if (
        not isinstance(decoder_payload, dict)
        or decoder_payload.get("schema") != SOURCE_DECODER_SCHEMA
        or decoder_payload.get("candidate") != "O0-I"
        or decoder_payload.get("contains_encoder") is not False
        or decoder_payload.get("contains_optimizer") is not False
    ):
        raise ValueError("H-TCA source decoder checkpoint is invalid")
    encoder = ConditionalWAEEncoder().to(device)
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    counts = parameter_counts(encoder, decoder)
    if counts["total"] > 8_000_000 or counts["decoder"] > 6_000_000:
        raise ValueError("H-TCA parameter budget exceeded")
    encoder.load_state_dict(training["encoder"], strict=True)
    decoder.load_state_dict(training["decoder"], strict=True)
    reference = ConditionalWAEDecoder("O0-I")
    reference.load_state_dict(decoder_payload["decoder_state_dict"], strict=True)
    if _state_hash(decoder) != _state_hash(reference):
        raise ValueError("H-TCA source training and decoder checkpoints disagree")
    return encoder, decoder


def _route_payload(path: str | Path, schema: str) -> dict[str, Any]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"H-TCA route checkpoint schema is invalid: {path}")
    if (
        payload.get("seed") != TRAIN_SEED
        or payload.get("epoch") != 1
        or payload.get("step") != EXPECTED_STEP
        or payload.get("batch_cursor") != 0
        or not payload.get("schedule_hash")
        or not payload.get("objective_hash")
    ):
        raise ValueError(f"H-TCA route checkpoint binding is invalid: {path}")
    return payload


def _check_route_evidence(
    screen_path: str,
    closeout_path: str,
    training_path: str,
    *,
    screen_schema: str,
    closeout_schema: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    screen = _load_json(screen_path)
    closeout = _load_json(closeout_path)
    if screen.get("schema") != screen_schema or screen.get("status") != "valid":
        raise ValueError(f"invalid paired screen evidence: {screen_path}")
    screen_training_hash = screen.get("training_checkpoint_sha256")
    if screen_training_hash is not None and screen_training_hash != file_sha256(training_path):
        raise ValueError(f"paired screen checkpoint hash mismatch: {screen_path}")
    if (
        closeout.get("schema") != closeout_schema
        or closeout.get("status") != "valid"
        or closeout.get("classification") != "prior_alignment_fit_failed"
    ):
        raise ValueError(f"invalid paired closeout evidence: {closeout_path}")
    return screen, closeout


def _check_pgc(path: str, schema: str, consumer: str) -> dict[str, Any]:
    payload = _load_json(path)
    if (
        payload.get("schema") != schema
        or payload.get("status") != "valid"
        or (consumer and consumer not in payload.get("permitted_consumers", ()))
        or payload.get("invocation_mode") != "manifest"
        or payload.get("legacy_diagnostic_only") is not False
    ):
        raise ValueError(f"invalid H-PGC upstream evidence: {path}")
    return payload


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = [
        args.config,
        args.wae_training_checkpoint,
        args.wae_decoder_checkpoint,
        args.swae_training_checkpoint,
        args.swae_screen_evidence,
        args.swae_closeout_evidence,
        args.sced_training_checkpoint,
        args.sced_screen_evidence,
        args.sced_closeout_evidence,
        args.h_pgc_audit_evidence,
        args.h_pgc_report_evidence,
        args.h_pgc_closeout_evidence,
    ]
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing or not Path(args.dataset_root).is_dir():
        raise FileNotFoundError("H-TCA lineage input is missing: " + ", ".join(missing))
    git = _git()
    if not git["clean"]:
        raise RuntimeError("H-TCA evidence requires a clean worktree")
    _check_route_evidence(
        args.swae_screen_evidence,
        args.swae_closeout_evidence,
        args.swae_training_checkpoint,
        screen_schema="electronic_o0_prior_native_swae_screen.v1",
        closeout_schema="electronic_o0_prior_native_swae_closeout.v1",
    )
    _check_route_evidence(
        args.sced_screen_evidence,
        args.sced_closeout_evidence,
        args.sced_training_checkpoint,
        screen_schema="electronic_o0_prior_native_sced_screen.v1",
        closeout_schema="electronic_o0_prior_native_sced_closeout.v1",
    )
    _check_pgc(
        args.h_pgc_audit_evidence,
        "electronic_o0_prior_gradient_conflict_audit.v1",
        "report",
    )
    _check_pgc(
        args.h_pgc_report_evidence,
        "electronic_o0_prior_gradient_conflict_report.v1",
        "closeout",
    )
    _check_pgc(
        args.h_pgc_closeout_evidence,
        "electronic_o0_prior_gradient_conflict_closeout.v1",
        "",
    )
    source_hash = file_sha256(args.wae_training_checkpoint)
    swae = _route_payload(args.swae_training_checkpoint, SWAE_SCHEMA)
    sced = _route_payload(args.sced_training_checkpoint, SCED_SCHEMA)
    if (
        swae.get("source_training_checkpoint_sha256") != source_hash
        or sced.get("source_training_checkpoint_sha256") != source_hash
        or swae.get("seed") != sced.get("seed")
        or swae.get("step") != sced.get("step")
        or swae.get("schedule_hash") != sced.get("schedule_hash")
    ):
        raise ValueError("H-TCA paired final checkpoints do not share source/schedule contract")
    return {
        "schema": f"{SCHEMA_PREFIX}_lineage.v1",
        "config_sha256": file_sha256(args.config),
        "config_content": config,
        "source_training_sha256": source_hash,
        "source_decoder_sha256": file_sha256(args.wae_decoder_checkpoint),
        "swae_training_sha256": file_sha256(args.swae_training_checkpoint),
        "sced_training_sha256": file_sha256(args.sced_training_checkpoint),
        "swae_screen_sha256": file_sha256(args.swae_screen_evidence),
        "swae_closeout_sha256": file_sha256(args.swae_closeout_evidence),
        "sced_screen_sha256": file_sha256(args.sced_screen_evidence),
        "sced_closeout_sha256": file_sha256(args.sced_closeout_evidence),
        "h_pgc_audit_sha256": file_sha256(args.h_pgc_audit_evidence),
        "h_pgc_report_sha256": file_sha256(args.h_pgc_report_evidence),
        "h_pgc_closeout_sha256": file_sha256(args.h_pgc_closeout_evidence),
        "git": git,
        "device": args.device,
        "historical_seed": TRAIN_SEED,
        "harness_seed": HARNESS_SEED,
        "batch_seeds": list(BATCH_SEEDS),
        "prior_seeds": list(PRIOR_SEEDS),
        "projection_seeds": list(PROJECTION_SEEDS),
    }


def _balanced_batch(
    images: torch.Tensor, labels: torch.Tensor, *, seed: int, per_class: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for label in range(10):
        choices = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()
        if choices.numel() < per_class:
            raise ValueError("H-TCA balanced batch lacks a class")
        rows.append(choices[torch.randperm(choices.numel(), generator=generator)[:per_class]])
    index = torch.cat(rows).to(images.device)
    return images.index_select(0, index), labels.index_select(0, index)


def _optimizer(device: torch.device, encoder: nn.Module, decoder: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(tuple(encoder.parameters()) + tuple(decoder.parameters()), lr=LEARNING_RATE)


def _alignment(
    encoded: torch.Tensor,
    labels: torch.Tensor,
    *,
    route: str,
    projections: torch.Tensor,
) -> torch.Tensor:
    if route == "sced":
        return classwise_sliced_energy_gaussian(encoded, labels, projections=projections)
    if route == "swae":
        return classwise_sliced_wasserstein_quantile(encoded, labels, projections=projections)
    raise ValueError(f"unknown H-TCA route: {route}")


def _class_alignment(
    encoded: torch.Tensor,
    *,
    route: str,
    projections: torch.Tensor,
) -> torch.Tensor:
    projected = encoded @ projections.t()
    if encoded.shape[0] < 2:
        raise ValueError("H-TCA classwise alignment requires at least two samples")
    if route == "sced":
        cross = 2.0 * gaussian_absolute_moment(projected).mean(0)
        within = _ordered_pairwise_absolute_mean(projected, dim=0)
        return (cross - within - 2.0 / math.sqrt(math.pi)).mean()
    if route == "swae":
        ordered = torch.sort(projected, dim=0).values
        target = gaussian_quantiles(encoded.shape[0], device=encoded.device, dtype=encoded.dtype)
        return (ordered - target).square().mean()
    raise ValueError(f"unknown H-TCA route: {route}")


def _projection_bank(*, route: str, seed: int, device: torch.device) -> torch.Tensor:
    if route == "sced":
        return sced_projections(seed=seed, device=device)
    if route == "swae":
        # The historical SWAE helper fixes the training seed internally.  H-TCA
        # needs the second locked bank as well, so reproduce its exact CPU
        # generator/normalization contract with an explicit seed.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = torch.randn((PROJECTION_COUNT, CODE_DIM), generator=generator, dtype=torch.float32)
        return F.normalize(values, dim=1).to(device)
    raise ValueError(f"unknown H-TCA route: {route}")


def _terms(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor,
    *,
    route: str,
    weights: dict[str, Any],
    projections: torch.Tensor,
) -> dict[str, torch.Tensor]:
    encoded = encoder(images, labels)
    prediction = render_view(decoder(encoded, labels), "O0-I")
    reconstruction = F.binary_cross_entropy(prediction, images) + 0.25 * sobel_l1(prediction, images)
    alignment = _alignment(encoded, labels, route=route, projections=projections)
    cycle_prediction = render_view(decoder(prior, labels), "O0-I")
    cycle = F.smooth_l1_loss(encoder(cycle_prediction, labels), prior)
    weight_key = "w_sced" if route == "sced" else "w_swd"
    total = reconstruction + float(weights[weight_key]) * alignment + float(weights["w_cycle"]) * cycle
    return {"total": total, "reconstruction": reconstruction, "alignment": alignment, "cycle": cycle}


def _flat(values: Iterable[torch.Tensor | None], *, device: torch.device) -> torch.Tensor:
    parts = [value.reshape(-1) for value in values if value is not None]
    return torch.cat(parts) if parts else torch.zeros(0, device=device)


def _group_vectors(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    gradients: tuple[torch.Tensor | None, ...],
    params: tuple[nn.Parameter, ...],
) -> dict[str, torch.Tensor]:
    by_id = {id(parameter): gradient for parameter, gradient in zip(params, gradients, strict=True)}
    named = list(encoder.named_parameters()) + [
        (f"decoder.{name}", value) for name, value in decoder.named_parameters()
    ]
    def selected(predicate):
        return _flat(
            (
                torch.zeros_like(value)
                if by_id[id(value)] is None
                else by_id[id(value)]
                for name, value in named
                if predicate(name)
            ),
            device=params[0].device,
        )
    result = {
        "encoder_features": selected(lambda name: name.startswith("features.")),
        "encoder_label_projection": selected(lambda name: name.startswith("label.") or name.startswith("projection.")),
        "decoder_input_label": selected(lambda name: name.startswith("decoder.label.") or name.startswith("decoder.input.")),
        "decoder_block": selected(lambda name: name.startswith("decoder.block.")),
        "decoder_upsample_output": selected(lambda name: name.startswith(("decoder.up1.", "decoder.up2.", "decoder.up3.", "decoder.output."))),
    }
    result["encoder_aggregate"] = torch.cat((result["encoder_features"], result["encoder_label_projection"]))
    result["decoder_aggregate"] = torch.cat((result["decoder_input_label"], result["decoder_block"], result["decoder_upsample_output"]))
    result["full_joint"] = torch.cat((result["encoder_aggregate"], result["decoder_aggregate"]))
    return result


def _rho(gradient: torch.Tensor, direction: torch.Tensor) -> float:
    denominator = gradient.norm() * direction.norm()
    if float(denominator.detach().cpu()) <= 1.0e-20:
        return 0.0
    return float(torch.dot(gradient, direction).div(denominator).detach().cpu())


def _loss_values(
    encoder, decoder, images, labels, prior, *, route, weights, projections
) -> dict[str, float]:
    with torch.no_grad():
        values = _terms(encoder, decoder, images, labels, prior, route=route, weights=weights, projections=projections)
    return {name: float(values[name].detach().cpu()) for name in ("reconstruction", "alignment", "cycle")}


def _observer_measurement(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    route: str,
    weights: dict[str, Any],
    actual_delta: torch.Tensor | None = None,
    include_classwise: bool = True,
) -> list[dict[str, Any]]:
    params = tuple(encoder.parameters()) + tuple(decoder.parameters())
    records: list[dict[str, Any]] = []
    weight_key = "w_sced" if route == "sced" else "w_swd"
    for batch_seed, prior_seed in zip(BATCH_SEEDS, PRIOR_SEEDS, strict=True):
        batch_images, batch_labels = _balanced_batch(images, labels, seed=batch_seed)
        prior = torch.randn(
            (batch_images.shape[0], CODE_DIM),
            generator=torch.Generator(device="cpu").manual_seed(prior_seed),
        ).to(images.device)
        for projection_seed in PROJECTION_SEEDS:
            projections = _projection_bank(route=route, seed=projection_seed, device=images.device)
            values = _terms(encoder, decoder, batch_images, batch_labels, prior, route=route, weights=weights, projections=projections)
            gradients = {
                name: _group_vectors(
                    encoder,
                    decoder,
                    torch.autograd.grad(values[name], params, retain_graph=True, allow_unused=True),
                    params,
                )
                for name in ("reconstruction", "alignment", "cycle")
            }
            g_reconstruction = gradients["reconstruction"]["full_joint"]
            g_alignment = gradients["alignment"]["full_joint"]
            g_cycle = gradients["cycle"]["full_joint"]
            d_raw = -(g_reconstruction + float(weights[weight_key]) * g_alignment + float(weights["w_cycle"]) * g_cycle)
            classwise = []
            if include_classwise:
                for class_id in range(10):
                    mask = batch_labels == class_id
                    class_encoded = encoder(batch_images[mask], batch_labels[mask])
                    class_value = _class_alignment(class_encoded, route=route, projections=projections)
                    class_gradients = torch.autograd.grad(class_value, params, allow_unused=True)
                    class_vector = _group_vectors(encoder, decoder, class_gradients, params)["full_joint"]
                    classwise.append({"class": class_id, "alignment_gradient_norm": float(class_vector.norm().detach().cpu())})
            record: dict[str, Any] = {
                "batch_seed": batch_seed,
                "prior_seed": prior_seed,
                "projection_seed": projection_seed,
                "losses": {name: float(values[name].detach().cpu()) for name in ("reconstruction", "alignment", "cycle")},
                "gradient_norms": {
                    objective: {group: float(vector.norm().detach().cpu()) for group, vector in gradients[objective].items()}
                    for objective in ("reconstruction", "alignment", "cycle")
                },
                "fixed_rho": {
                    "reconstruction": _rho(gradients["reconstruction"]["full_joint"], d_raw),
                    "alignment": _rho(gradients["alignment"]["full_joint"], d_raw),
                    "cycle": _rho(gradients["cycle"]["full_joint"], d_raw),
                },
                "training_batch": False,
                "classwise": classwise,
                "finite": bool(all(torch.isfinite(vector).all() for objective in gradients.values() for vector in objective.values()) and torch.isfinite(d_raw).all()),
                "_cycle_gradient": g_cycle.detach(),
                "_d_raw": d_raw.detach(),
                "_images": batch_images,
                "_labels": batch_labels,
                "_prior": prior,
                "_projections": projections,
            }
            if actual_delta is not None:
                record["adam_rho"] = _rho(g_cycle, actual_delta)
                record["adam_directional_derivatives"] = {
                    "reconstruction": float(torch.dot(gradients["reconstruction"]["full_joint"], actual_delta).detach().cpu()),
                    "alignment": float(torch.dot(g_alignment, actual_delta).detach().cpu()),
                    "cycle": float(torch.dot(g_cycle, actual_delta).detach().cpu()),
                }
            records.append(record)
    return records


def _strip(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _parameter_vector(encoder: nn.Module, decoder: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in tuple(encoder.parameters()) + tuple(decoder.parameters())])


def _restore_parameters(encoder, decoder, values: list[torch.Tensor]) -> None:
    for parameter, value in zip(tuple(encoder.parameters()) + tuple(decoder.parameters()), values, strict=True):
        parameter.data.copy_(value)


def _direction_delta(params: tuple[nn.Parameter, ...], direction: torch.Tensor) -> list[torch.Tensor]:
    parameter_norm = torch.sqrt(sum((parameter.detach().square().sum() for parameter in params), torch.zeros((), device=direction.device)))
    norm = direction.norm()
    if float(norm.detach().cpu()) <= 1.0e-20:
        raise ValueError("H-TCA direction has zero norm")
    scale = FD_RELATIVE_STEP * max(1.0, float(parameter_norm.detach().cpu())) / norm
    values: list[torch.Tensor] = []
    offset = 0
    for parameter in params:
        count = parameter.numel()
        values.append((direction[offset : offset + count] * scale).view_as(parameter))
        offset += count
    if offset != direction.numel():
        raise ValueError("H-TCA direction does not cover all parameters")
    return values


def _curvature(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    record: dict[str, Any],
    *,
    route: str,
    weights: dict[str, Any],
    direction_name: str,
    direction: torch.Tensor,
) -> dict[str, Any]:
    params = tuple(encoder.parameters()) + tuple(decoder.parameters())
    unit = direction / direction.norm().clamp_min(1.0e-20)
    values = _terms(encoder, decoder, record["_images"], record["_labels"], record["_prior"], route=route, weights=weights, projections=record["_projections"])
    cycle_grads = torch.autograd.grad(values["cycle"], params, create_graph=True, allow_unused=True)
    flat_cycle = _flat(cycle_grads, device=unit.device)
    directional = torch.dot(flat_cycle, unit)
    hvp = torch.autograd.grad(directional, params, allow_unused=True)
    flat_hvp = _flat(hvp, device=unit.device)
    curvature = float(torch.dot(flat_hvp, unit).detach().cpu())
    original = [parameter.detach().clone() for parameter in params]
    deltas = _direction_delta(params, direction)
    with torch.no_grad():
        for parameter, delta in zip(params, deltas, strict=True):
            parameter.add_(delta)
    plus = _loss_values(encoder, decoder, record["_images"], record["_labels"], record["_prior"], route=route, weights=weights, projections=record["_projections"])["cycle"]
    with torch.no_grad():
        for parameter, delta in zip(params, deltas, strict=True):
            parameter.sub_(2.0 * delta)
    minus = _loss_values(encoder, decoder, record["_images"], record["_labels"], record["_prior"], route=route, weights=weights, projections=record["_projections"])["cycle"]
    _restore_parameters(encoder, decoder, original)
    before = record["losses"]["cycle"]
    central = plus - minus
    return {
        "direction": direction_name,
        "batch_seed": record["batch_seed"],
        "prior_seed": record["prior_seed"],
        "projection_seed": record["projection_seed"],
        "directional_derivative": float(directional.detach().cpu()),
        "hvp_quadratic": curvature,
        "finite_difference_central": central,
        "finite_difference_plus": plus,
        "finite_difference_minus": minus,
        "finite_difference_forward": plus - before,
        "baseline_cycle": before,
        "finite": all(
            math.isfinite(value)
            for value in (curvature, central, plus, minus, plus - before, before)
        ),
    }


def _update(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    route: str,
    weights: dict[str, Any],
    projections: torch.Tensor,
) -> dict[str, float]:
    prior = torch.randn((images.shape[0], CODE_DIM), device=images.device)
    optimizer.zero_grad(set_to_none=True)
    values = _terms(encoder, decoder, images, labels, prior, route=route, weights=weights, projections=projections)
    if not all(torch.isfinite(value) for value in values.values()):
        raise FloatingPointError("H-TCA replay update is non-finite")
    values["total"].backward()
    optimizer.step()
    return {name: float(values[name].detach().cpu()) for name in ("total", "reconstruction", "alignment", "cycle")}


def _post_observer_values(
    encoder, decoder, record: dict[str, Any], *, route: str, weights: dict[str, Any]
) -> dict[str, float]:
    return _loss_values(encoder, decoder, record["_images"], record["_labels"], record["_prior"], route=route, weights=weights, projections=record["_projections"])


def _final_match(
    encoder,
    decoder,
    optimizer,
    payload: dict[str, Any],
    *,
    history: list[dict[str, float]],
    route: str,
    step: int,
    schedule_hash: str,
) -> dict[str, Any]:
    observed = {
        "encoder": _state_hash(encoder),
        "decoder": _state_hash(decoder),
        "optimizer": _tree_hash(optimizer.state_dict()),
        "rng": _rng_hash(),
        "schedule_hash": schedule_hash,
        "step": step,
        "cursor": 0,
    }
    expected = {
        "encoder": _state_dict_hash(payload["encoder"]),
        "decoder": _state_dict_hash(payload["decoder"]),
        "optimizer": _tree_hash(payload["optimizer"]),
        "rng": {
            "cpu": tensor_sha256(payload["rng_state"]["cpu"]),
            "cuda": [tensor_sha256(item) for item in payload["rng_state"]["cuda"]],
        },
        "schedule_hash": payload.get("schedule_hash"),
        "step": payload.get("step"),
        "cursor": payload.get("batch_cursor"),
    }
    if route == "sced":
        observed["history"] = history
        expected["history"] = payload.get("history", [])
    else:
        observed["history"] = []
        expected["history"] = []
    equal = observed == expected
    return {"equal": equal, "observed": observed, "expected": expected}


def _save_progress(
    path: Path,
    *,
    route: str,
    step: int,
    epoch: int,
    cursor: int,
    encoder,
    decoder,
    optimizer,
    history: list[dict[str, float]],
    history_window: list[dict[str, float]],
    landmarks: list[dict[str, Any]],
) -> None:
    payload = {
        "schema": f"{SCHEMA_PREFIX}_replay.v1",
        "route": route,
        "step": step,
        "epoch": epoch,
        "batch_cursor": cursor,
        "encoder": {key: value.detach().cpu() for key, value in encoder.state_dict().items()},
        "decoder": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        "optimizer": _canonical_tree(optimizer.state_dict()),
        "history": list(history),
        "history_window": list(history_window),
        "landmarks": landmarks,
        "rng_hash": _rng_hash(),
    }
    atomic_torch_save(path, payload)


def _prepare_data(dataset_root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    all_images, all_labels = load_local_mnist(dataset_root, train=True)
    rows, _ = stratified_mnist_indices(all_labels)
    return all_images.index_select(0, rows), all_labels.index_select(0, rows)


def _replay_arm(
    args: argparse.Namespace,
    *,
    route: str,
    final_payload: dict[str, Any],
    source_hash: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    output_root: Path,
    measure_curvature: bool,
) -> dict[str, Any]:
    device = torch.device(args.device)
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    encoder, decoder = _load_source_models(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
    optimizer = _optimizer(device, encoder, decoder)
    weights = dict(final_payload["weights"])
    schedule = balanced_epoch_schedule(labels, seed=TRAIN_SEED, per_class=10)
    schedule_hash = canonical_sha256([tensor_sha256(item.detach().cpu()) for item in schedule])
    if schedule_hash != final_payload.get("schedule_hash"):
        raise ValueError(f"{route} schedule hash does not reproduce historical schedule")
    train_projection = _projection_bank(route=route, seed=PROJECTION_SEEDS[0], device=device)
    history: list[dict[str, float]] = []
    history_window: list[dict[str, float]] = []
    landmarks: list[dict[str, Any]] = []
    cycle_medians: list[float] = []
    progress = output_root / f"{route}_replay_progress.pt"
    step = 0
    cursor = 0
    curvature_points = {0, 200, 400, 625}
    for epoch in range(1):
        while cursor < len(schedule):
            if step in LANDMARK_STEPS:
                records = _observer_measurement(
                    encoder,
                    decoder,
                    images.to(device),
                    labels.to(device),
                    route=route,
                    weights=weights,
                    include_classwise=True,
                )
                cycle_median = float(torch.tensor([item["losses"]["cycle"] for item in records], dtype=torch.float64).median())
                cycle_medians.append(cycle_median)
                if len(cycle_medians) >= 3 and cycle_medians[-1] > cycle_medians[-2] > cycle_medians[-3]:
                    curvature_points.add(step)
                curvature_records: list[dict[str, Any]] = []
                pre_parameters = [parameter.detach().clone() for parameter in tuple(encoder.parameters()) + tuple(decoder.parameters())] if measure_curvature and step in curvature_points else None
                if pre_parameters is not None:
                    for item in records:
                        curvature_records.append(_curvature(encoder, decoder, item, route=route, weights=weights, direction_name="raw", direction=item["_d_raw"]))
                landmark = {"step": step, "cycle_median": cycle_median, "records": records, "curvature": curvature_records}
                if step == EXPECTED_STEP:
                    landmarks.append(landmark)
                    break
            indices = schedule[cursor]
            batch_images = images.index_select(0, indices).to(device)
            batch_labels = labels.index_select(0, indices).to(device)
            training_prior = None
            training_before = None
            if "landmark" in locals() and landmark.get("step") == step:
                rng_before = torch.cuda.get_rng_state()
                training_prior = torch.randn((batch_images.shape[0], CODE_DIM), device=device)
                torch.cuda.set_rng_state(rng_before)
                training_before = _loss_values(
                    encoder,
                    decoder,
                    batch_images,
                    batch_labels,
                    training_prior,
                    route=route,
                    weights=weights,
                    projections=train_projection,
                )
            before = _parameter_vector(encoder, decoder)
            values = _update(encoder, decoder, optimizer, batch_images, batch_labels, route=route, weights=weights, projections=train_projection)
            after = _parameter_vector(encoder, decoder)
            actual_delta = after - before
            if 'landmark' in locals() and landmark.get("step") == step:
                for item in landmark["records"]:
                    item["adam_rho"] = _rho(item["_cycle_gradient"], actual_delta)
                    item["adam_directional_derivatives"] = {
                        "cycle": float(torch.dot(item["_cycle_gradient"], actual_delta).detach().cpu()),
                    }
                    post = _post_observer_values(encoder, decoder, item, route=route, weights=weights)
                    item["post_losses"] = post
                    item["actual_delta"] = {name: post[name] - item["losses"][name] for name in ("reconstruction", "alignment", "cycle")}
                    item["taylor_remainder_cycle"] = item["actual_delta"]["cycle"] - item["adam_directional_derivatives"]["cycle"]
                training_after = _loss_values(
                    encoder,
                    decoder,
                    batch_images,
                    batch_labels,
                    training_prior,
                    route=route,
                    weights=weights,
                    projections=train_projection,
                )
                landmark["training_losses_before"] = training_before
                landmark["training_losses_after"] = training_after
                landmark["training_cycle_delta"] = training_after["cycle"] - training_before["cycle"]
                if pre_parameters is not None:
                    post_parameters = [parameter.detach().clone() for parameter in tuple(encoder.parameters()) + tuple(decoder.parameters())]
                    _restore_parameters(encoder, decoder, pre_parameters)
                    for item in landmark["records"]:
                        if item["projection_seed"] == PROJECTION_SEEDS[0]:
                            landmark["curvature"].append(_curvature(encoder, decoder, item, route=route, weights=weights, direction_name="adam", direction=actual_delta))
                            landmark["curvature"][-1]["first_order_cycle"] = item["adam_directional_derivatives"]["cycle"]
                            landmark["curvature"][-1]["actual_cycle_delta"] = item["actual_delta"]["cycle"]
                            landmark["curvature"][-1]["taylor_remainder_cycle"] = item["taylor_remainder_cycle"]
                    _restore_parameters(encoder, decoder, post_parameters)
                landmarks.append(landmark)
                del landmark
            history_window.append(values)
            if route == "sced":
                history_window[-1] = {
                    "total": values["total"],
                    "reconstruction": values["reconstruction"],
                    "sced": values["alignment"],
                    "cycle": values["cycle"],
                }
            step += 1
            cursor += 1
            if route == "sced" and (step % 50 == 0 or step == EXPECTED_STEP):
                history.append({name: sum(row[name] for row in history_window) / len(history_window) for name in ("total", "reconstruction", "sced", "cycle")})
                history_window = []
            if step % 50 == 0 or step == EXPECTED_STEP:
                _save_progress(progress, route=route, step=step, epoch=1 if step == EXPECTED_STEP else 0, cursor=cursor, encoder=encoder, decoder=decoder, optimizer=optimizer, history=history, history_window=history_window, landmarks=[{key: value for key, value in item.items() if key != "records"} for item in landmarks])
    if step == EXPECTED_STEP and (not landmarks or landmarks[-1]["step"] != EXPECTED_STEP):
        final_records = _observer_measurement(
            encoder,
            decoder,
            images.to(device),
            labels.to(device),
            route=route,
            weights=weights,
            include_classwise=True,
        )
        final_cycle_median = float(
            torch.tensor(
                [item["losses"]["cycle"] for item in final_records], dtype=torch.float64
            ).median()
        )
        final_curvature = []
        if measure_curvature:
            for item in final_records:
                final_curvature.append(
                    _curvature(
                        encoder,
                        decoder,
                        item,
                        route=route,
                        weights=weights,
                        direction_name="raw",
                        direction=item["_d_raw"],
                    )
                )
        landmarks.append(
            {
                "step": EXPECTED_STEP,
                "cycle_median": final_cycle_median,
                "records": final_records,
                "curvature": final_curvature,
            }
        )
    if step != EXPECTED_STEP:
        raise RuntimeError(f"{route} replay stopped at step {step}")
    final = _final_match(encoder, decoder, optimizer, final_payload, history=history, route=route, step=step, schedule_hash=schedule_hash)
    serializable_landmarks = []
    for landmark in landmarks:
        serializable = {key: value for key, value in landmark.items() if key != "records"}
        serializable["records"] = [_strip(record) for record in landmark["records"]]
        serializable_landmarks.append(serializable)
    return {
        "route": route,
        "step": step,
        "schedule_hash": schedule_hash,
        "history": history,
        "final_match": final,
        "landmarks": serializable_landmarks,
        "optimizer_created": True,
        "evaluator_calls": 0,
    }


def _short_replay(
    args: argparse.Namespace,
    *,
    route: str,
    weights: dict[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device(args.device)
    schedule = balanced_epoch_schedule(labels, seed=TRAIN_SEED, per_class=10)
    schedule_hash = canonical_sha256([tensor_sha256(item.detach().cpu()) for item in schedule])
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    encoder, decoder = _load_source_models(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
    optimizer = _optimizer(device, encoder, decoder)
    projection = _projection_bank(route=route, seed=PROJECTION_SEEDS[0], device=device)
    uninterrupted = []
    uninterrupted_history_window: list[dict[str, float]] = []
    last_delta: torch.Tensor | None = None
    for index in range(3):
        batch_images = images.index_select(0, schedule[index]).to(device)
        batch_labels = labels.index_select(0, schedule[index]).to(device)
        before = _parameter_vector(encoder, decoder)
        values = _update(encoder, decoder, optimizer, batch_images, batch_labels, route=route, weights=weights, projections=projection)
        last_delta = _parameter_vector(encoder, decoder) - before
        uninterrupted.append(values)
        if route == "sced":
            uninterrupted_history_window.append(
                {
                    "total": values["total"],
                    "reconstruction": values["reconstruction"],
                    "sced": values["alignment"],
                    "cycle": values["cycle"],
                }
            )

    def state_digest(
        model_encoder,
        model_decoder,
        model_optimizer,
        *,
        step: int,
        cursor: int,
        history_window: list[dict[str, float]],
    ) -> dict[str, Any]:
        return {
            "encoder": _state_hash(model_encoder),
            "decoder": _state_hash(model_decoder),
            "optimizer": _tree_hash(model_optimizer.state_dict()),
            "rng": _rng_hash(),
            "schedule_hash": schedule_hash,
            "step": step,
            "cursor": cursor,
            "history": [],
            "history_window": history_window if route == "sced" else [],
        }

    uninterrupted_digest = state_digest(
        encoder,
        decoder,
        optimizer,
        step=3,
        cursor=3,
        history_window=uninterrupted_history_window,
    )

    observer_records = _observer_measurement(
        encoder,
        decoder,
        images.to(device),
        labels.to(device),
        route=route,
        weights=weights,
        actual_delta=last_delta,
        include_classwise=False,
    )
    observer_finite = all(
        record.get("finite")
        and all(math.isfinite(float(value)) for value in record.get("adam_directional_derivatives", {}).values())
        for record in observer_records
    )
    curvature_finite = False
    if observer_records and last_delta is not None:
        curvature_records = [
            _curvature(
                encoder,
                decoder,
                observer_records[0],
                route=route,
                weights=weights,
                direction_name="raw",
                direction=observer_records[0]["_d_raw"],
            ),
            _curvature(
                encoder,
                decoder,
                observer_records[0],
                route=route,
                weights=weights,
                direction_name="adam",
                direction=last_delta,
            ),
        ]
        curvature_finite = all(item.get("finite") for item in curvature_records)

    with tempfile.TemporaryDirectory(prefix="h_tca_replay_") as directory:
        torch.manual_seed(TRAIN_SEED)
        torch.cuda.manual_seed_all(TRAIN_SEED)
        interrupted_encoder, interrupted_decoder = _load_source_models(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        interrupted_optimizer = _optimizer(device, interrupted_encoder, interrupted_decoder)
        first = _update(interrupted_encoder, interrupted_decoder, interrupted_optimizer, images.index_select(0, schedule[0]).to(device), labels.index_select(0, schedule[0]).to(device), route=route, weights=weights, projections=projection)
        resumed_history_window = []
        if route == "sced":
            resumed_history_window.append(
                {
                    "total": first["total"],
                    "reconstruction": first["reconstruction"],
                    "sced": first["alignment"],
                    "cycle": first["cycle"],
                }
            )
        checkpoint = Path(directory) / "replay.pt"
        atomic_torch_save(
            checkpoint,
            {
                "schema": f"{SCHEMA_PREFIX}_replay.v1",
                "encoder": {key: value.detach().cpu() for key, value in interrupted_encoder.state_dict().items()},
                "decoder": {key: value.detach().cpu() for key, value in interrupted_decoder.state_dict().items()},
                "optimizer": _canonical_tree(interrupted_optimizer.state_dict()),
                "optimizer_state": interrupted_optimizer.state_dict(),
                "rng_state": {"cpu": torch.get_rng_state().clone(), "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]},
                "rng": _rng_hash(),
                "schedule_hash": schedule_hash,
                "step": 1,
                "batch_cursor": 1,
                "history": [],
                "history_window": resumed_history_window,
            },
        )
        resumed_encoder, resumed_decoder = _load_source_models(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        resumed_optimizer = _optimizer(device, resumed_encoder, resumed_decoder)
        payload = safe_torch_load(checkpoint, map_location="cpu")
        resumed_encoder.load_state_dict(payload["encoder"], strict=True)
        resumed_decoder.load_state_dict(payload["decoder"], strict=True)
        resumed_optimizer.load_state_dict(payload["optimizer_state"])
        for state in resumed_optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        torch.set_rng_state(payload["rng_state"]["cpu"])
        torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])
        for index in (1, 2):
            values = _update(resumed_encoder, resumed_decoder, resumed_optimizer, images.index_select(0, schedule[index]).to(device), labels.index_select(0, schedule[index]).to(device), route=route, weights=weights, projections=projection)
            if route == "sced":
                resumed_history_window.append(
                    {
                        "total": values["total"],
                        "reconstruction": values["reconstruction"],
                        "sced": values["alignment"],
                        "cycle": values["cycle"],
                    }
                )
        resumed_digest = state_digest(
            resumed_encoder,
            resumed_decoder,
            resumed_optimizer,
            step=3,
            cursor=3,
            history_window=resumed_history_window,
        )
    interrupted_resumed_exact = uninterrupted_digest == resumed_digest
    return {
        "route": route,
        "finite": bool(all(math.isfinite(float(value)) for row in uninterrupted for value in row.values())),
        "interrupted_resumed_exact": interrupted_resumed_exact,
        "schedule_exact": uninterrupted_digest["schedule_hash"] == resumed_digest["schedule_hash"],
        "sampler_cursor_exact": uninterrupted_digest["cursor"] == resumed_digest["cursor"],
        "partial_history_exact": uninterrupted_digest["history_window"] == resumed_digest["history_window"],
        "observer_finite": observer_finite,
        "directional_derivatives_finite": observer_finite,
        "curvature_finite": curvature_finite,
        "uninterrupted_updates": uninterrupted,
    }


def _classify(arms: dict[str, dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    support: dict[str, dict[str, bool]] = {}
    for route, result in arms.items():
        route_support: dict[str, bool] = {}
        landmarks = result["landmarks"]
        for mechanism in ("trajectory_gradient_rotation", "optimizer_preconditioning_cycle_conflict", "finite_step_curvature_cycle_drift", "observer_distribution_mismatch"):
            route_support[mechanism] = False
        for index in range(2, len(landmarks)):
            window = [landmarks[index - 2], landmarks[index - 1], landmarks[index]]
            for mechanism in route_support:
                hits = 0
                for batch_seed in BATCH_SEEDS:
                    batch_records = [r for landmark in window for r in landmark["records"] if r["batch_seed"] == batch_seed]
                    def bank_hits(predicate):
                        return all(
                            sum(predicate(record) for record in batch_records if record["projection_seed"] == projection_seed)
                            >= 2
                            for projection_seed in PROJECTION_SEEDS
                        )

                    if mechanism == "trajectory_gradient_rotation" and bank_hits(
                        lambda record: record["fixed_rho"]["cycle"] >= CYCLE_CONFLICT_COSINE
                    ):
                        hits += 1
                    elif mechanism == "optimizer_preconditioning_cycle_conflict" and bank_hits(
                        lambda record: record.get("fixed_rho", {}).get("cycle", 0.0) < CYCLE_CONFLICT_COSINE
                        and record.get("adam_rho", 0.0) >= CYCLE_CONFLICT_COSINE
                    ):
                        hits += 1
                    elif mechanism == "observer_distribution_mismatch":
                        # A mismatch is only supported when the actual training
                        # batch is non-increasing while each fixed observer bank
                        # is increasing over the same consecutive landmark
                        # window. Observer-only drift is insufficient to
                        # distinguish a distribution mismatch from ordinary
                        # trajectory drift.
                        training_cycle_deltas = [
                            landmark.get("training_cycle_delta")
                            for landmark in window
                        ]
                        training_non_increasing = all(
                            delta is not None and delta <= 1.0e-7
                            for delta in training_cycle_deltas
                        )
                        if training_non_increasing and bank_hits(
                            lambda record: record.get("actual_delta", {}).get("cycle", 0.0) > 1.0e-7
                        ):
                            hits += 1
                if hits >= REQUIRED_OBSERVER_BATCHES:
                    route_support[mechanism] = True
        curvature_items = [item for landmark in landmarks for item in landmark.get("curvature", []) if item["direction"] == "adam"]
        if curvature_items and sum(
            item["finite"]
            and item["hvp_quadratic"] > 0.0
            and item["finite_difference_forward"] > 1.0e-7
            and item.get("first_order_cycle", 1.0) <= 1.0e-12
            and item.get("actual_cycle_delta", 0.0) > 1.0e-7
            and item.get("taylor_remainder_cycle", 0.0)
            >= CURVATURE_REMAINDER_FRACTION * item.get("actual_cycle_delta", 0.0)
            for item in curvature_items
        ) >= REQUIRED_SUPPORT_LANDMARKS:
            route_support["finite_step_curvature_cycle_drift"] = True
        support[route] = route_support
    supported = [name for name in ("trajectory_gradient_rotation", "optimizer_preconditioning_cycle_conflict", "finite_step_curvature_cycle_drift", "observer_distribution_mismatch") if all(support[route].get(name, False) for route in arms)]
    any_supported = [name for name in ("trajectory_gradient_rotation", "optimizer_preconditioning_cycle_conflict", "finite_step_curvature_cycle_drift", "observer_distribution_mismatch") if any(support[route].get(name, False) for route in arms)]
    if len(supported) > 1 or len(any_supported) > 1:
        classification = "mixed_trajectory_path_dependence"
        next_action = "register_local_path_mechanism_hypothesis"
    elif len(supported) == 1:
        classification = supported[0]
        next_action = "register_trajectory_mechanism_hypothesis"
    else:
        classification = "trajectory_attribution_underdetermined"
        next_action = "register_denser_local_trajectory_attribution"
    return classification, next_action, {"support_by_route": support, "shared_supported_mechanisms": supported, "any_supported_mechanisms": any_supported, "alignment_estimator_independent": bool(supported)}


def _audit(args: argparse.Namespace, lineage: dict[str, Any], output: Path) -> None:
    _require_cuda(args)
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    source_hash = file_sha256(args.wae_training_checkpoint)
    swae_payload = _route_payload(args.swae_training_checkpoint, SWAE_SCHEMA)
    sced_payload = _route_payload(args.sced_training_checkpoint, SCED_SCHEMA)
    images, labels = _prepare_data(args.dataset_root)
    arms = {
        "swae": _replay_arm(args, route="swae", final_payload=swae_payload, source_hash=source_hash, images=images, labels=labels, output_root=Path(args._run_manifest["output_root"]), measure_curvature=True),
        "sced": _replay_arm(args, route="sced", final_payload=sced_payload, source_hash=source_hash, images=images, labels=labels, output_root=Path(args._run_manifest["output_root"]), measure_curvature=True),
    }
    failures = []
    for route, result in arms.items():
        if not result["final_match"]["equal"]:
            failures.append(f"{route}_historical_replay_mismatch")
        for landmark in result["landmarks"]:
            for record in landmark["records"]:
                if not record.get("finite") or not all(math.isfinite(float(value)) for value in record.get("actual_delta", {}).values()):
                    failures.append(f"{route}_nonfinite_landmark")
            for item in landmark.get("curvature", []):
                if not item.get("finite"):
                    failures.append(f"{route}_nonfinite_curvature")
    classification, next_action, decision = _classify(arms)
    peak = int(torch.cuda.max_memory_allocated(device))
    storage = sum(item.stat().st_size for item in Path(args._run_manifest["output_root"]).rglob("*") if item.is_file())
    if peak >= 6 * 1024**3:
        failures.append("cuda_peak_memory_gate")
    if storage > 2 * 1024**3:
        failures.append("run_storage_gate")
    valid = not failures
    if not valid:
        classification = "invalid_harness"
        next_action = "repair_h_tca_harness_and_reissue_lineage"
    payload = {
        "schema": f"{SCHEMA_PREFIX}_audit.v1",
        "stage": "audit",
        "status": "valid" if valid else "invalid_harness",
        "passed": valid,
        "measurement_valid": valid,
        "classification": classification,
        "next_action": next_action,
        "validity_failures": failures,
        "decision": decision,
        "arms": arms,
        "optimizer_created": True,
        "evaluator_calls": 0,
        "source_parameters_unchanged": True,
        "peak_memory_bytes": peak,
        "storage_bytes": storage,
        "memory_gate": peak < 6 * 1024**3,
        "storage_gate": storage <= 2 * 1024**3,
        "input_artifact_hashes": {key: file_sha256(value) for key, value in {
            "source_training": args.wae_training_checkpoint,
            "source_decoder": args.wae_decoder_checkpoint,
            "swae_training": args.swae_training_checkpoint,
            "swae_screen": args.swae_screen_evidence,
            "swae_closeout": args.swae_closeout_evidence,
            "sced_training": args.sced_training_checkpoint,
            "sced_screen": args.sced_screen_evidence,
            "sced_closeout": args.sced_closeout_evidence,
            "h_pgc_audit": args.h_pgc_audit_evidence,
            "h_pgc_report": args.h_pgc_report_evidence,
            "h_pgc_closeout": args.h_pgc_closeout_evidence,
        }.items()},
        "lineage_sha256": canonical_sha256(lineage),
        "consumer_contract": f"{SCHEMA_PREFIX}-report.v1",
        "permitted_consumers": ["report"],
        "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1",
    }
    _write(output, payload, args)


def _check_consumer(path: str, *, action: str, schemas: tuple[str, ...]) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema") not in schemas or payload.get("status") != "valid":
        raise ValueError(f"invalid H-TCA {action} evidence: {path}")
    if action not in payload.get("permitted_consumers", ()):
        raise ValueError(f"{path} does not permit H-TCA consumer {action}")
    if payload.get("invocation_mode") != "manifest" or payload.get("legacy_diagnostic_only") is not False:
        raise ValueError(f"{path} is legacy H-TCA evidence")
    return payload


def _stage(args: argparse.Namespace, stage: str, expected: dict[str, Any]) -> dict[str, Any]:
    args._stage_decision = stage_action_decision(
        stage,
        producer_contract=f"{SCHEMA_PREFIX}-{stage}.v1",
        expected_inputs=expected,
        evidence=args.output,
    )
    return args._stage_decision


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("H-TCA lifecycle is manifest-only")
    action = args.research_action
    common = (
        "config", "dataset_root", "output", "wae_training_checkpoint", "wae_decoder_checkpoint",
        "swae_training_checkpoint", "swae_screen_evidence", "swae_closeout_evidence",
        "sced_training_checkpoint", "sced_screen_evidence", "sced_closeout_evidence",
        "h_pgc_audit_evidence", "h_pgc_report_evidence", "h_pgc_closeout_evidence",
    )
    required = {
        "validate": common,
        "preflight": common + ("validation_evidence",),
        "audit": common + ("preflight_evidence",),
        "report": ("config", "dataset_root", "output", "measurement_evidence"),
        "closeout": ("config", "dataset_root", "output", "measurement_evidence"),
    }
    fields = common + ("device", "validation_evidence", "preflight_evidence", "measurement_evidence", "resume_checkpoint")
    args = prepare_args(args, program=PROGRAM, action=action, required=required[action], defaults={"device": "cuda", "seed": HARNESS_SEED}, mixed_fields=fields)
    validate_manifest_namespace(args, action=action)
    output = Path(args.output)
    if action in {"validate", "preflight", "audit"}:
        config = _config(args.config)
        lineage = _lineage(args, config)
    else:
        config = _config(args.config)
        lineage = None
    if action == "validate":
        decision = _stage(args, "validate", {"config": file_sha256(args.config), "source_training": file_sha256(args.wae_training_checkpoint), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        _write(output, {"schema": f"{SCHEMA_PREFIX}_validation.v1", "stage": "validate", "status": "valid", "passed": True, "lineage_sha256": canonical_sha256(lineage), "producer_contract_revision": f"{SCHEMA_PREFIX}-validation.v1", "consumer_contract": f"{SCHEMA_PREFIX}-preflight.v1", "permitted_consumers": ["preflight"], "input_artifact_hashes": {"source_training": file_sha256(args.wae_training_checkpoint)}}, args)
        return 0
    if action == "preflight":
        validation = _check_consumer(args.validation_evidence, action="preflight", schemas=(f"{SCHEMA_PREFIX}_validation.v1",))
        if not validation.get("passed"):
            raise ValueError("H-TCA preflight requires passed validation")
        decision = _stage(args, "preflight", {"validation": file_sha256(args.validation_evidence), "source_training": file_sha256(args.wae_training_checkpoint), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        device = torch.device(args.device)
        torch.cuda.reset_peak_memory_stats(device)
        images, labels = _prepare_data(args.dataset_root)
        results = {
            route: _short_replay(args, route=route, weights=dict(payload["weights"]), images=images, labels=labels)
            for route, payload in (("swae", _route_payload(args.swae_training_checkpoint, SWAE_SCHEMA)), ("sced", _route_payload(args.sced_training_checkpoint, SCED_SCHEMA)))
        }
        storage = sum(item.stat().st_size for item in Path(args._run_manifest["output_root"]).rglob("*") if item.is_file())
        passed = all(
            item["finite"]
            and item["interrupted_resumed_exact"]
            and item["schedule_exact"]
            and item["sampler_cursor_exact"]
            and item["partial_history_exact"]
            and item["observer_finite"]
            and item["directional_derivatives_finite"]
            and item["curvature_finite"]
            for item in results.values()
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        if peak >= 6 * 1024**3:
            passed = False
        if storage > 2 * 1024**3:
            passed = False
        _write(output, {"schema": f"{SCHEMA_PREFIX}_preflight.v1", "stage": "preflight", "status": "valid" if passed else "invalid_harness", "passed": passed, "measurement_valid": passed, "results": results, "optimizer_created": True, "evaluator_calls": 0, "peak_memory_bytes": peak, "memory_gate": peak < 6 * 1024**3, "storage_bytes": storage, "storage_gate": storage <= 2 * 1024**3, "input_artifact_hashes": {"validation": file_sha256(args.validation_evidence)}, "lineage_sha256": canonical_sha256(lineage), "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1", "permitted_consumers": ["audit"] if passed else [], "producer_contract_revision": f"{SCHEMA_PREFIX}-preflight.v1"}, args)
        return 0
    if action == "audit":
        preflight = _check_consumer(args.preflight_evidence, action="audit", schemas=(f"{SCHEMA_PREFIX}_preflight.v1",))
        if not preflight.get("passed"):
            raise ValueError("H-TCA audit requires passed preflight")
        decision = _stage(args, "audit", {"preflight": file_sha256(args.preflight_evidence), "source_training": file_sha256(args.wae_training_checkpoint), "lineage_sha256": canonical_sha256(lineage)})
        if decision["action"] == "REUSE":
            return 0
        _audit(args, lineage, output)
        return 0
    measurement = _check_consumer(args.measurement_evidence, action=action, schemas=(f"{SCHEMA_PREFIX}_audit.v1", f"{SCHEMA_PREFIX}_report.v1") if action == "closeout" else (f"{SCHEMA_PREFIX}_audit.v1",))
    decision = _stage(args, action, {"measurement": file_sha256(args.measurement_evidence)})
    if decision["action"] == "REUSE":
        return 0
    if action == "report":
        _write(output, {"schema": f"{SCHEMA_PREFIX}_report.v1", "stage": "report", "status": "valid", "passed": bool(measurement.get("passed")), "measurement_valid": measurement.get("measurement_valid"), "classification": measurement.get("classification"), "next_action": measurement.get("next_action"), "measurement": args.measurement_evidence, "source_evidence_sha256": file_sha256(args.measurement_evidence), "input_artifact_hashes": {"measurement": file_sha256(args.measurement_evidence)}, "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1", "permitted_consumers": ["closeout"], "producer_contract_revision": f"{SCHEMA_PREFIX}-report.v1"}, args)
        return 0
    _write(output, {"schema": f"{SCHEMA_PREFIX}_closeout.v1", "stage": "closeout", "status": "valid", "passed": bool(measurement.get("passed")), "measurement_valid": measurement.get("measurement_valid"), "classification": measurement.get("classification"), "next_action": measurement.get("next_action"), "measurement": args.measurement_evidence, "source_evidence_sha256": file_sha256(args.measurement_evidence), "formal_promotion": False, "optical_training_authorized": False, "fiber_training_authorized": False, "synthetic_diagnostic_only": True, "input_artifact_hashes": {"measurement": file_sha256(args.measurement_evidence)}, "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1", "permitted_consumers": [], "producer_contract_revision": f"{SCHEMA_PREFIX}-closeout.v1"}, args)
    return 0
