"""Dense local, diagnostic-only trajectory attribution for electronic O0.

This module replays the immutable H-SWAE/H-SCED histories on temporary model
copies.  It does not train a candidate, load an evaluator, or modify prior
route artifacts.  The only new variable is temporal diagnostic resolution and
the exact decomposition of the historical AdamW displacement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
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
from . import electronic_o0_prior_trajectory_curvature as tca
from .lifecycle_artifacts import (
    prepare_args,
    stage_action_decision,
    validate_manifest_namespace,
)


SCHEMA_PREFIX = "electronic_o0_prior_dense_local_attribution"
PROGRAM = "electronic-o0-prior-dense-local-attribution"
HARNESS_SEED = 8751
TRAIN_SEED = 8601
EXPECTED_STEP = 625
CODE_DIM = 8
BATCH_SEEDS = (11, 23, 37, 51)
PRIOR_SEEDS = (1011, 1023, 1037, 1051)
PROJECTION_SEEDS = (271829, 314159)
PROJECTION_COUNT = 256
LANDMARK_STEPS = (
    tuple(range(0, 21, 2))
    + tuple(range(230, 271, 5))
    + tuple(range(330, 371, 5))
    + tuple(range(480, 521, 5))
    + (625,)
)
CLASSWISE_STEPS = {0, 250, 350, 500, 625}
CURVATURE_ALPHAS = (0.25, 0.5, 1.0)
LEARNING_RATE = 2.0e-4
BETAS = (0.9, 0.999)
EPS = 1.0e-8
WEIGHT_DECAY = 0.01
CYCLE_CONFLICT_COSINE = 0.05
REQUIRED_SUPPORT_LANDMARKS = 3
REQUIRED_OBSERVER_BATCHES = 3
REMAINDER_FRACTION = 0.50
ABS_CYCLE_TOLERANCE = 1.0e-7
REL_CYCLE_TOLERANCE = 1.0e-4
SCED_SCHEMA = "electronic_o0_prior_native_sced_training.v1"
SWAE_SCHEMA = "electronic_o0_prior_native_swae_training.v1"


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
        raise RuntimeError("H-DLTA requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H-DLTA requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("H-DLTA config must be a mapping")
    expected = {
        "schema_version": 1,
        "name": "electronic_o0_prior_dense_local_attribution_v1",
        "version": 20260824,
        "code_dim": CODE_DIM,
        "logical_batch_size": 100,
        "per_class": 10,
        "dense_local_attribution_seed": HARNESS_SEED,
        "historical_seed": TRAIN_SEED,
        "historical_steps": EXPECTED_STEP,
        "landmark_steps": list(LANDMARK_STEPS),
        "batch_seeds": list(BATCH_SEEDS),
        "prior_seeds": list(PRIOR_SEEDS),
        "projection_seeds": list(PROJECTION_SEEDS),
        "curvature_alphas": list(CURVATURE_ALPHAS),
        "cycle_conflict_cosine": CYCLE_CONFLICT_COSINE,
        "required_support_landmarks": REQUIRED_SUPPORT_LANDMARKS,
        "required_observer_batches": REQUIRED_OBSERVER_BATCHES,
        "adamw_lr": LEARNING_RATE,
        "adamw_betas": list(BETAS),
        "adamw_eps": EPS,
        "adamw_weight_decay": WEIGHT_DECAY,
        "cycle_delta_absolute_tolerance": ABS_CYCLE_TOLERANCE,
        "cycle_delta_relative_tolerance": REL_CYCLE_TOLERANCE,
        "parameter_budget": 8_000_000,
        "decoder_parameter_budget": 6_000_000,
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("H-DLTA config is not the registered v1 contract")
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
        {
            key: tensor_sha256(value.detach().cpu())
            for key, value in module.state_dict().items()
        }
    )


def _tree_hash(value: Any) -> str:
    return canonical_sha256(tca._canonical_tree(value))


def _rng_hash() -> dict[str, Any]:
    return {
        "cpu": tensor_sha256(torch.get_rng_state().cpu()),
        "cuda": [tensor_sha256(item.cpu()) for item in torch.cuda.get_rng_state_all()],
    }


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = [
        args.config,
        args.dataset_root,
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
        args.h_tca_audit_evidence,
        args.h_tca_report_evidence,
        args.h_tca_closeout_evidence,
    ]
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "H-DLTA lineage input is missing: " + ", ".join(missing)
        )
    git = _git()
    if not git["clean"]:
        raise RuntimeError("H-DLTA evidence requires a clean worktree")
    tca._check_route_evidence(
        args.swae_screen_evidence,
        args.swae_closeout_evidence,
        args.swae_training_checkpoint,
        screen_schema="electronic_o0_prior_native_swae_screen.v1",
        closeout_schema="electronic_o0_prior_native_swae_closeout.v1",
    )
    tca._check_route_evidence(
        args.sced_screen_evidence,
        args.sced_closeout_evidence,
        args.sced_training_checkpoint,
        screen_schema="electronic_o0_prior_native_sced_screen.v1",
        closeout_schema="electronic_o0_prior_native_sced_closeout.v1",
    )
    tca._check_pgc(
        args.h_pgc_audit_evidence,
        "electronic_o0_prior_gradient_conflict_audit.v1",
        "report",
    )
    pgc_report = tca._check_pgc(
        args.h_pgc_report_evidence,
        "electronic_o0_prior_gradient_conflict_report.v1",
        "closeout",
    )
    pgc_closeout = tca._check_pgc(
        args.h_pgc_closeout_evidence,
        "electronic_o0_prior_gradient_conflict_closeout.v1",
        "",
    )
    if pgc_report.get("source_evidence_sha256") != file_sha256(
        args.h_pgc_audit_evidence
    ) or pgc_closeout.get("source_evidence_sha256") != file_sha256(
        args.h_pgc_report_evidence
    ):
        raise ValueError("H-PGC evidence chain hash mismatch")
    tca_audit = _load_json(args.h_tca_audit_evidence)
    tca_report = _load_json(args.h_tca_report_evidence)
    tca_closeout = _load_json(args.h_tca_closeout_evidence)
    if (
        tca_audit.get("schema") != "electronic_o0_prior_trajectory_curvature_audit.v1"
        or tca_audit.get("status") != "valid"
        or not tca_audit.get("passed")
    ):
        raise ValueError("H-DLTA requires valid H-TCA audit")
    if (
        tca_report.get("schema") != "electronic_o0_prior_trajectory_curvature_report.v1"
        or tca_report.get("status") != "valid"
    ):
        raise ValueError("H-DLTA requires valid H-TCA report")
    if (
        tca_closeout.get("schema")
        != "electronic_o0_prior_trajectory_curvature_closeout.v1"
        or tca_closeout.get("status") != "valid"
        or tca_closeout.get("classification")
        != "trajectory_attribution_underdetermined"
    ):
        raise ValueError("H-DLTA requires H-TCA underdetermined closeout")
    if (
        tca_closeout.get("next_action")
        != "register_denser_local_trajectory_attribution"
    ):
        raise ValueError("H-TCA closeout does not authorize H-DLTA")
    if tca_report.get("source_evidence_sha256") != file_sha256(
        args.h_tca_audit_evidence
    ) or tca_closeout.get("source_evidence_sha256") != file_sha256(
        args.h_tca_report_evidence
    ):
        raise ValueError("H-TCA evidence chain hash mismatch")
    source_hash = file_sha256(args.wae_training_checkpoint)
    swae = tca._route_payload(args.swae_training_checkpoint, SWAE_SCHEMA)
    sced = tca._route_payload(args.sced_training_checkpoint, SCED_SCHEMA)
    if (
        swae.get("source_training_checkpoint_sha256") != source_hash
        or sced.get("source_training_checkpoint_sha256") != source_hash
        or swae.get("seed") != sced.get("seed")
        or swae.get("step") != sced.get("step")
        or swae.get("schedule_hash") != sced.get("schedule_hash")
    ):
        raise ValueError(
            "H-DLTA paired checkpoints do not share source/schedule contract"
        )
    expected_tca_inputs = {
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
    }
    tca_inputs = tca_audit.get("input_artifact_hashes", {})
    if any(
        tca_inputs.get(key) != file_sha256(path)
        for key, path in expected_tca_inputs.items()
    ):
        raise ValueError("H-TCA audit is not bound to current paired artifacts")
    resume_path = getattr(args, "resume_checkpoint", None)
    resume_hash = None
    if resume_path:
        if not Path(resume_path).is_file():
            raise FileNotFoundError(
                f"H-DLTA resume checkpoint is missing: {resume_path}"
            )
        resume_payload = safe_torch_load(resume_path, map_location="cpu")
        if (
            not isinstance(resume_payload, dict)
            or resume_payload.get("schema") != f"{SCHEMA_PREFIX}_replay.v1"
        ):
            raise ValueError("H-DLTA resume checkpoint schema is invalid")
        resume_hash = file_sha256(resume_path)
    return {
        "schema": f"{SCHEMA_PREFIX}_lineage.v1",
        "config_sha256": file_sha256(args.config),
        "config_content": config,
        "source_training_sha256": source_hash,
        "source_decoder_sha256": file_sha256(args.wae_decoder_checkpoint),
        "swae_training_sha256": file_sha256(args.swae_training_checkpoint),
        "sced_training_sha256": file_sha256(args.sced_training_checkpoint),
        "h_tca_audit_sha256": file_sha256(args.h_tca_audit_evidence),
        "h_tca_report_sha256": file_sha256(args.h_tca_report_evidence),
        "h_tca_closeout_sha256": file_sha256(args.h_tca_closeout_evidence),
        "swae_screen_sha256": file_sha256(args.swae_screen_evidence),
        "swae_closeout_sha256": file_sha256(args.swae_closeout_evidence),
        "sced_screen_sha256": file_sha256(args.sced_screen_evidence),
        "sced_closeout_sha256": file_sha256(args.sced_closeout_evidence),
        "h_pgc_audit_sha256": file_sha256(args.h_pgc_audit_evidence),
        "h_pgc_report_sha256": file_sha256(args.h_pgc_report_evidence),
        "h_pgc_closeout_sha256": file_sha256(args.h_pgc_closeout_evidence),
        "git": git,
        "device": args.device,
        "harness_seed": HARNESS_SEED,
        "historical_seed": TRAIN_SEED,
        "landmark_steps": list(LANDMARK_STEPS),
        "batch_seeds": list(BATCH_SEEDS),
        "prior_seeds": list(PRIOR_SEEDS),
        "projection_seeds": list(PROJECTION_SEEDS),
        "resume_checkpoint_sha256": resume_hash,
    }


def _flat(values: list[torch.Tensor | None], device: torch.device) -> torch.Tensor:
    parts = [value.reshape(-1) for value in values if value is not None]
    return torch.cat(parts) if parts else torch.zeros(0, device=device)


def _rho(gradient: torch.Tensor, direction: torch.Tensor) -> float:
    denominator = gradient.norm() * direction.norm()
    if float(denominator.detach().cpu()) <= 1.0e-20:
        return 0.0
    return float(torch.dot(gradient, direction).div(denominator).detach().cpu())


def _directional(gradient: torch.Tensor, direction: torch.Tensor) -> float:
    value = torch.dot(gradient, direction).detach().cpu()
    return float(value)


def _decompose_adamw(
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, torch.Tensor]:
    params = tuple(encoder.parameters()) + tuple(decoder.parameters())
    if len(optimizer.param_groups) != 1:
        raise ValueError("H-DLTA requires one AdamW parameter group")
    group = optimizer.param_groups[0]
    lr = float(group["lr"])
    beta1, beta2 = tuple(float(value) for value in group["betas"])
    eps = float(group["eps"])
    decay = float(group["weight_decay"])
    components = {
        name: [] for name in ("gradient", "momentum", "adaptive", "decay", "predicted")
    }
    for parameter, gradient in zip(params, gradients, strict=True):
        grad = torch.zeros_like(parameter) if gradient is None else gradient
        state = optimizer.state.get(parameter, {})
        old_m = state.get("exp_avg", torch.zeros_like(parameter))
        old_v = state.get("exp_avg_sq", torch.zeros_like(parameter))
        old_step = state.get("step", 0)
        old_step = int(old_step.item()) if torch.is_tensor(old_step) else int(old_step)
        step = old_step + 1
        m = old_m.clone()
        m.lerp_(grad, 1.0 - beta1)
        v = old_v.clone()
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        gradient_direction = -lr * grad
        momentum_direction = -(lr / bias1) * m
        denominator = v.sqrt().div(math.sqrt(bias2)).add_(eps)
        adaptive_direction = torch.zeros_like(parameter)
        adaptive_direction.addcdiv_(m, denominator, value=-(lr / bias1))
        decayed = parameter.detach().clone()
        decayed.mul_(1.0 - lr * decay)
        decay_direction = decayed - parameter.detach()
        predicted_parameter = parameter.detach().clone()
        predicted_parameter.mul_(1.0 - lr * decay)
        predicted_parameter.addcdiv_(m, denominator, value=-(lr / bias1))
        predicted = predicted_parameter - parameter.detach()
        for name, value in (
            ("gradient", gradient_direction),
            ("momentum", momentum_direction),
            ("adaptive", adaptive_direction),
            ("decay", decay_direction),
            ("predicted", predicted),
        ):
            components[name].append(value.reshape(-1))
    result = {name: torch.cat(values) for name, values in components.items()}
    # Keep the contract names explicit while retaining the short internal names
    # used by the attribution predicates.
    result.update({f"d_{name}": value for name, value in result.items()})
    return result


def _component_gate(predicted: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    difference = predicted - actual
    actual_norm = float(actual.norm().detach().cpu())
    difference_norm = float(difference.norm().detach().cpu())
    relative = difference_norm / max(actual_norm, 1.0e-20)
    max_abs = float(difference.abs().max().detach().cpu())
    return {
        "cosine": _rho(predicted, actual),
        "relative_l2_error": relative,
        "max_abs_error": max_abs,
        "finite": bool(
            torch.isfinite(predicted).all()
            and torch.isfinite(actual).all()
            and torch.isfinite(difference).all()
        ),
        "passed": bool(
            _rho(predicted, actual) >= 0.999999
            and relative <= 1.0e-5
            and max_abs <= 1.0e-7
        ),
    }


def _loss_values(
    encoder, decoder, images, labels, prior, *, route, weights, projections
):
    return tca._loss_values(
        encoder,
        decoder,
        images,
        labels,
        prior,
        route=route,
        weights=weights,
        projections=projections,
    )


def _curvature(
    encoder,
    decoder,
    record: dict[str, Any],
    direction: torch.Tensor,
    pre_parameters: list[torch.Tensor],
    post_parameters: list[torch.Tensor],
    *,
    route: str,
    weights: dict[str, Any],
) -> dict[str, Any]:
    params = tuple(encoder.parameters()) + tuple(decoder.parameters())
    tca._restore_parameters(encoder, decoder, pre_parameters)
    values = tca._terms(
        encoder,
        decoder,
        record["_images"],
        record["_labels"],
        record["_prior"],
        route=route,
        weights=weights,
        projections=record["_projections"],
    )
    gradients = torch.autograd.grad(
        values["cycle"], params, create_graph=True, allow_unused=True
    )
    flat_gradient = _flat(list(gradients), direction.device)
    directional = torch.dot(flat_gradient, direction)
    hvp = torch.autograd.grad(directional, params, allow_unused=True)
    quadratic = float(
        torch.dot(_flat(list(hvp), direction.device), direction).detach().cpu()
    )
    with torch.no_grad():
        for parameter, value in zip(params, post_parameters, strict=True):
            parameter.copy_(value)
    baseline = record["losses"]["cycle"]
    finite = {}
    for alpha in CURVATURE_ALPHAS:
        tca._restore_parameters(encoder, decoder, pre_parameters)
        with torch.no_grad():
            for parameter, delta in zip(
                params, _split_direction(direction * float(alpha), params), strict=True
            ):
                parameter.add_(delta)
        finite[str(alpha)] = _loss_values(
            encoder,
            decoder,
            record["_images"],
            record["_labels"],
            record["_prior"],
            route=route,
            weights=weights,
            projections=record["_projections"],
        )["cycle"]
    tca._restore_parameters(encoder, decoder, post_parameters)
    actual = finite["1.0"] - baseline
    first_order = _directional(flat_gradient, direction)
    second_order = first_order + 0.5 * quadratic
    return {
        "direction": "actual_adamw",
        "batch_seed": record["batch_seed"],
        "prior_seed": record["prior_seed"],
        "projection_seed": record["projection_seed"],
        "baseline_cycle": baseline,
        "first_order_cycle": first_order,
        "hvp_quadratic": quadratic,
        "second_order_cycle": second_order,
        "actual_cycle_delta": actual,
        "taylor_remainder_cycle": actual - first_order,
        "finite_difference": finite,
        "finite": all(
            math.isfinite(float(value))
            for value in (first_order, quadratic, second_order, actual)
            + tuple(finite.values())
        ),
    }


def _observer_records(
    encoder, decoder, images, labels, *, route, weights, include_classwise
):
    return tca._observer_measurement(
        encoder,
        decoder,
        images,
        labels,
        route=route,
        weights=weights,
        include_classwise=include_classwise,
    )


def _update(
    encoder, decoder, optimizer, images, labels, *, route, weights, projections
):
    return tca._update(
        encoder,
        decoder,
        optimizer,
        images,
        labels,
        route=route,
        weights=weights,
        projections=projections,
    )


def _save_progress(
    path: Path,
    *,
    route: str,
    step: int,
    cursor: int,
    encoder,
    decoder,
    optimizer,
    history,
    history_window,
    landmarks,
    schedule_hash: str,
    objective_hash: str,
    source_training_sha256: str,
) -> None:
    atomic_torch_save(
        path,
        {
            "schema": f"{SCHEMA_PREFIX}_replay.v1",
            "route": route,
            "step": step,
            "batch_cursor": cursor,
            "schedule_hash": schedule_hash,
            "objective_hash": objective_hash,
            "source_training_sha256": source_training_sha256,
            "encoder": {
                key: value.detach().cpu() for key, value in encoder.state_dict().items()
            },
            "decoder": {
                key: value.detach().cpu() for key, value in decoder.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "rng_state": {
                "cpu": torch.get_rng_state().clone(),
                "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
            },
            "history": list(history),
            "history_window": list(history_window),
            "landmarks": list(landmarks),
        },
    )


def _split_direction(
    direction: torch.Tensor, params: tuple[nn.Parameter, ...]
) -> list[torch.Tensor]:
    values = []
    offset = 0
    for parameter in params:
        count = parameter.numel()
        values.append(direction[offset : offset + count].view_as(parameter))
        offset += count
    if offset != direction.numel():
        raise ValueError("H-DLTA direction does not cover model parameters")
    return values


def _finite_scalar_mapping(values: dict[str, Any]) -> bool:
    for value in values.values():
        if isinstance(value, dict):
            if not _finite_scalar_mapping(value):
                return False
        elif isinstance(value, (float, int)) and not math.isfinite(float(value)):
            return False
    return True


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _serializable(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_serializable(item) for item in value]
    if torch.is_tensor(value):
        return {"tensor_sha256": tensor_sha256(value.detach().cpu())}
    return value


def _training_measurement(
    encoder, decoder, optimizer, images, labels, prior, *, route, weights, projections
):
    params = tuple(encoder.parameters()) + tuple(decoder.parameters())
    values = tca._terms(
        encoder,
        decoder,
        images,
        labels,
        prior,
        route=route,
        weights=weights,
        projections=projections,
    )
    gradients = {
        name: _flat(
            list(
                torch.autograd.grad(
                    values[name], params, retain_graph=True, allow_unused=True
                )
            ),
            images.device,
        )
        for name in ("reconstruction", "alignment", "cycle", "total")
    }
    return values, gradients


def _attach_components(
    records: list[dict[str, Any]],
    components: dict[str, torch.Tensor],
    actual_delta: torch.Tensor,
) -> None:
    names = ("gradient", "momentum", "adaptive", "decay", "predicted")
    for record in records:
        cycle_gradient = record["_cycle_gradient"]
        record["component_rho"] = {
            name: _rho(cycle_gradient, direction)
            for name in names
            for direction in (components[name],)
        }
        record["component_directional_derivatives"] = {
            name: _directional(cycle_gradient, components[name]) for name in names
        }
        record["displacement_norms"] = {
            f"d_{name}": float(components[name].norm().detach().cpu())
            for name in names
        }
        record["displacement_sha256"] = {
            f"d_{name}": tensor_sha256(components[name].detach().cpu())
            for name in names
        }
        record["actual_rho"] = _rho(cycle_gradient, actual_delta)
        record["d_adam_actual_norm"] = float(actual_delta.norm().detach().cpu())
        record["d_adam_actual_sha256"] = tensor_sha256(actual_delta.detach().cpu())
        record["actual_directional_derivative"] = _directional(
            cycle_gradient, actual_delta
        )
        record["reverse_directional_derivative"] = _directional(
            cycle_gradient, -actual_delta
        )


def _replay_arm(
    args: argparse.Namespace,
    *,
    route: str,
    final_payload: dict[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    output_root: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    encoder, decoder = tca._load_source_models(
        args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
    )
    optimizer = tca._optimizer(device, encoder, decoder)
    weights = dict(final_payload["weights"])
    schedule = tca.balanced_epoch_schedule(labels, seed=TRAIN_SEED, per_class=10)
    schedule_hash = canonical_sha256(
        [tensor_sha256(item.detach().cpu()) for item in schedule]
    )
    if schedule_hash != final_payload.get("schedule_hash"):
        raise ValueError(
            f"{route} H-DLTA schedule hash does not reproduce historical schedule"
        )
    train_projection = tca._projection_bank(
        route=route, seed=PROJECTION_SEEDS[0], device=device
    )
    history: list[dict[str, float]] = []
    history_window: list[dict[str, float]] = []
    landmarks: list[dict[str, Any]] = []
    step = 0
    cursor = 0
    resume_path = getattr(args, "resume_checkpoint", None)
    if resume_path:
        resume_payload = safe_torch_load(resume_path, map_location="cpu")
        if not isinstance(resume_payload, dict):
            raise ValueError("H-DLTA resume checkpoint is not a mapping")
        checkpoint_route = resume_payload.get("route")
        if checkpoint_route == route:
            if resume_payload.get("schema") != f"{SCHEMA_PREFIX}_replay.v1":
                raise ValueError("H-DLTA resume checkpoint schema is invalid")
            if resume_payload.get("schedule_hash") != schedule_hash:
                raise ValueError("H-DLTA resume schedule hash mismatch")
            if resume_payload.get("objective_hash") != final_payload.get(
                "objective_hash"
            ):
                raise ValueError("H-DLTA resume objective hash mismatch")
            if resume_payload.get("source_training_sha256") != file_sha256(
                args.wae_training_checkpoint
            ):
                raise ValueError("H-DLTA resume source hash mismatch")
            step = int(resume_payload.get("step", -1))
            cursor = int(resume_payload.get("batch_cursor", -1))
            if not 0 <= step <= EXPECTED_STEP or cursor != step:
                raise ValueError("H-DLTA resume cursor is invalid")
            encoder.load_state_dict(resume_payload["encoder"], strict=True)
            decoder.load_state_dict(resume_payload["decoder"], strict=True)
            optimizer.load_state_dict(resume_payload["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            history = list(resume_payload.get("history", []))
            history_window = list(resume_payload.get("history_window", []))
            landmarks = list(resume_payload.get("landmarks", []))
            torch.set_rng_state(resume_payload["rng_state"]["cpu"])
            torch.cuda.set_rng_state_all(resume_payload["rng_state"]["cuda"])
        elif checkpoint_route not in (None, "swae", "sced"):
            raise ValueError("H-DLTA resume checkpoint route is invalid")
    progress = output_root / f"{route}_dense_local_replay_progress.pt"
    landmark_set = set(LANDMARK_STEPS)
    for cursor, indices in enumerate(schedule[cursor:], start=cursor):
        step = cursor
        batch_images = images.index_select(0, indices).to(device)
        batch_labels = labels.index_select(0, indices).to(device)
        if step in landmark_set:
            pre_parameters = [
                parameter.detach().clone()
                for parameter in tuple(encoder.parameters())
                + tuple(decoder.parameters())
            ]
            rng_before = torch.cuda.get_rng_state()
            prior = torch.randn((batch_images.shape[0], CODE_DIM), device=device)
            torch.cuda.set_rng_state(rng_before)
            values, gradients = _training_measurement(
                encoder,
                decoder,
                optimizer,
                batch_images,
                batch_labels,
                prior,
                route=route,
                weights=weights,
                projections=train_projection,
            )
            identity_before = _loss_values(
                encoder,
                decoder,
                batch_images,
                batch_labels,
                prior,
                route=route,
                weights=weights,
                projections=train_projection,
            )
            components = _decompose_adamw(
                encoder,
                decoder,
                optimizer,
                tuple(
                    torch.autograd.grad(
                        values["total"],
                        tuple(encoder.parameters()) + tuple(decoder.parameters()),
                        retain_graph=True,
                        allow_unused=True,
                    )
                ),
            )
            observer_records = _observer_records(
                encoder,
                decoder,
                images.to(device),
                labels.to(device),
                route=route,
                weights=weights,
                include_classwise=step in CLASSWISE_STEPS,
            )
        before = tca._parameter_vector(encoder, decoder)
        update_values = _update(
            encoder,
            decoder,
            optimizer,
            batch_images,
            batch_labels,
            route=route,
            weights=weights,
            projections=train_projection,
        )
        after = tca._parameter_vector(encoder, decoder)
        actual_delta = after - before
        if step in landmark_set:
            post_parameters = [
                parameter.detach().clone()
                for parameter in tuple(encoder.parameters())
                + tuple(decoder.parameters())
            ]
            component_gate = _component_gate(components["predicted"], actual_delta)
            training_before = {
                name: float(values[name].detach().cpu())
                for name in ("reconstruction", "alignment", "cycle")
            }
            training_after = _loss_values(
                encoder,
                decoder,
                batch_images,
                batch_labels,
                prior,
                route=route,
                weights=weights,
                projections=train_projection,
            )
            training_identity = all(
                abs(training_before[name] - identity_before[name]) <= 1.0e-12
                for name in training_before
            )
            for record in observer_records:
                record["post_losses"] = tca._post_observer_values(
                    encoder, decoder, record, route=route, weights=weights
                )
                record["actual_delta"] = {
                    name: record["post_losses"][name] - record["losses"][name]
                    for name in ("reconstruction", "alignment", "cycle")
                }
            _attach_components(observer_records, components, actual_delta)
            curvature = []
            for record in observer_records:
                if record["projection_seed"] == PROJECTION_SEEDS[0]:
                    curvature.append(
                        _curvature(
                            encoder,
                            decoder,
                            record,
                            actual_delta,
                            pre_parameters,
                            post_parameters,
                            route=route,
                            weights=weights,
                        )
                    )
            reverse_ok = all(
                abs(
                    record["reverse_directional_derivative"]
                    + record["actual_directional_derivative"]
                )
                <= max(1.0e-8, abs(record["actual_directional_derivative"]) * 1.0e-5)
                for record in observer_records
            )
            landmark = {
                "step": step,
                "training_losses_before": training_before,
                "training_losses_after": training_after,
                "training_cycle_delta": training_after["cycle"]
                - training_before["cycle"],
                "training_grad_norms": {
                    name: float(value.norm().detach().cpu())
                    for name, value in gradients.items()
                },
                "component_gate": component_gate,
                "training_identity": training_identity,
                "reverse_control": reverse_ok,
                "records": observer_records,
                "curvature": curvature,
            }
            landmarks.append(landmark)
        if route == "sced":
            history_window.append(
                {
                    "total": update_values["total"],
                    "reconstruction": update_values["reconstruction"],
                    "sced": update_values["alignment"],
                    "cycle": update_values["cycle"],
                }
            )
        if (step + 1) % 50 == 0:
            if route == "sced":
                history.append(
                    {
                        name: sum(row[name] for row in history_window)
                        / len(history_window)
                        for name in ("total", "reconstruction", "sced", "cycle")
                    }
                )
                history_window = []
            _save_progress(
                progress,
                route=route,
                step=step + 1,
                cursor=step + 1,
                encoder=encoder,
                decoder=decoder,
                optimizer=optimizer,
                history=history,
                history_window=history_window,
                landmarks=[_serializable(item) for item in landmarks],
                schedule_hash=schedule_hash,
                objective_hash=str(final_payload["objective_hash"]),
                source_training_sha256=file_sha256(args.wae_training_checkpoint),
            )
    if not landmarks or landmarks[-1]["step"] != EXPECTED_STEP:
        final_records = _observer_records(
            encoder,
            decoder,
            images.to(device),
            labels.to(device),
            route=route,
            weights=weights,
            include_classwise=True,
        )
        landmarks.append(
            {
                "step": EXPECTED_STEP,
                "training_identity": True,
                "reverse_control": True,
                "records": final_records,
                "curvature": [],
            }
        )
    final_match = tca._final_match(
        encoder,
        decoder,
        optimizer,
        final_payload,
        history=history,
        route=route,
        step=EXPECTED_STEP,
        schedule_hash=schedule_hash,
    )
    return {
        "route": route,
        "step": EXPECTED_STEP,
        "schedule_hash": schedule_hash,
        "history": history,
        "final_match": final_match,
        "landmarks": [_serializable(item) for item in landmarks],
        "optimizer_created": True,
        "evaluator_calls": 0,
    }


def _cycle_threshold(baseline: float) -> float:
    return max(ABS_CYCLE_TOLERANCE, REL_CYCLE_TOLERANCE * abs(baseline))


def _window_name(step: int) -> str:
    if step <= 20:
        return "early"
    if 230 <= step <= 270:
        return "w250"
    if 330 <= step <= 370:
        return "w350"
    if 480 <= step <= 520:
        return "w500"
    return "anchor"


def _mechanism_predicate(name: str, record: dict[str, Any]) -> bool:
    rho = record.get("component_rho", {})
    if name == "raw_gradient_rotation":
        return rho.get("gradient", 0.0) >= CYCLE_CONFLICT_COSINE
    if name == "momentum_induced_cycle_conflict":
        return (
            rho.get("gradient", 0.0) < CYCLE_CONFLICT_COSINE
            and rho.get("momentum", 0.0) >= CYCLE_CONFLICT_COSINE
        )
    if name == "adaptive_preconditioning_cycle_conflict":
        return (
            rho.get("momentum", 0.0) < CYCLE_CONFLICT_COSINE
            and rho.get("adaptive", 0.0) >= CYCLE_CONFLICT_COSINE
        )
    if name == "weight_decay_cycle_conflict":
        predicted = record.get("component_directional_derivatives", {}).get(
            "predicted", 0.0
        )
        decay = record.get("component_directional_derivatives", {}).get("decay", 0.0)
        return (
            rho.get("adaptive", 0.0) < CYCLE_CONFLICT_COSINE
            and rho.get("predicted", 0.0) >= CYCLE_CONFLICT_COSINE
            and decay >= 0.5 * max(predicted, 0.0)
        )
    if name == "observer_sample_distribution_mismatch":
        return record.get("actual_delta", {}).get("cycle", 0.0) > _cycle_threshold(
            record.get("losses", {}).get("cycle", 0.0)
        )
    return False


def _curvature_supported(item: dict[str, Any]) -> bool:
    baseline = abs(float(item.get("baseline_cycle", 0.0)))
    actual = float(item.get("actual_cycle_delta", 0.0))
    finite_difference = item.get("finite_difference", {})
    values = [
        float(finite_difference.get(str(alpha), float("nan")))
        for alpha in CURVATURE_ALPHAS
    ]
    trend = (
        values[0] > float(item.get("baseline_cycle", 0.0))
        and values[0] <= values[1] <= values[2]
    )
    return bool(
        item.get("finite")
        and item.get("first_order_cycle", 1.0) <= 1.0e-12
        and actual > _cycle_threshold(baseline)
        and item.get("hvp_quadratic", 0.0) > 0.0
        and item.get("taylor_remainder_cycle", 0.0) >= REMAINDER_FRACTION * actual
        and trend
    )


def _batch_bank_support(
    window: list[dict[str, Any]], name: str, batch_seed: int
) -> bool:
    records = [
        record
        for landmark in window
        for record in landmark.get("records", [])
        if record.get("batch_seed") == batch_seed
    ]
    return all(
        sum(
            _mechanism_predicate(name, record)
            for record in records
            if record.get("projection_seed") == seed
        )
        >= 3
        for seed in PROJECTION_SEEDS
    )


def _mismatch_batch_support(window: list[dict[str, Any]], batch_seed: int) -> bool:
    records = [
        record
        for landmark in window
        for record in landmark.get("records", [])
        if record.get("batch_seed") == batch_seed
    ]
    training_ok = all(
        landmark.get("training_cycle_delta") is not None
        and landmark["training_cycle_delta"]
        <= _cycle_threshold(
            landmark.get("training_losses_before", {}).get("cycle", 0.0)
        )
        for landmark in window
    )
    return training_ok and all(
        sum(
            _mechanism_predicate("observer_sample_distribution_mismatch", record)
            for record in records
            if record.get("projection_seed") == seed
        )
        >= 3
        for seed in PROJECTION_SEEDS
    )


def _classify(arms: dict[str, dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    mechanisms = (
        "raw_gradient_rotation",
        "momentum_induced_cycle_conflict",
        "adaptive_preconditioning_cycle_conflict",
        "weight_decay_cycle_conflict",
        "finite_step_curvature_cycle_drift",
        "observer_sample_distribution_mismatch",
    )
    support: dict[str, dict[str, dict[str, bool]]] = {}
    for route, result in arms.items():
        by_window = {
            name: {mechanism: False for mechanism in mechanisms}
            for name in ("early", "w250", "w350", "w500")
        }
        landmarks = result.get("landmarks", [])
        for start in range(len(landmarks) - 2):
            window = landmarks[start : start + 3]
            name = _window_name(window[-1].get("step", -1))
            if name not in by_window:
                continue
            if any(
                _window_name(landmark.get("step", -1)) != name for landmark in window
            ):
                continue
            for mechanism in mechanisms:
                if mechanism == "finite_step_curvature_cycle_drift":
                    hits = sum(
                        all(
                            any(
                                _curvature_supported(item)
                                for item in landmark.get("curvature", [])
                                if item.get("batch_seed") == batch
                            )
                            for landmark in window
                        )
                        for batch in BATCH_SEEDS
                    )
                    by_window[name][mechanism] = hits >= REQUIRED_OBSERVER_BATCHES
                elif mechanism == "observer_sample_distribution_mismatch":
                    by_window[name][mechanism] = (
                        sum(
                            _mismatch_batch_support(window, batch)
                            for batch in BATCH_SEEDS
                        )
                        >= REQUIRED_OBSERVER_BATCHES
                    )
                else:
                    by_window[name][mechanism] = (
                        sum(
                            _batch_bank_support(window, mechanism, batch)
                            for batch in BATCH_SEEDS
                        )
                        >= REQUIRED_OBSERVER_BATCHES
                    )
        support[route] = by_window
    later_names = ("w250", "w350", "w500")
    shared_later = [
        mechanism
        for mechanism in mechanisms
        if any(
            all(support[route][window][mechanism] for route in arms)
            for window in later_names
        )
    ]
    any_later = [
        mechanism
        for mechanism in mechanisms
        if any(
            support[route][window][mechanism]
            for route in arms
            for window in later_names
        )
    ]
    early_only = [
        mechanism
        for mechanism in mechanisms
        if any(support[route]["early"][mechanism] for route in arms)
        and mechanism not in any_later
    ]
    if not any_later and early_only:
        classification = "initialization_transient_path_dependence"
        next_action = "register_initialization_transient_counterfactual"
    elif len(any_later) == 1:
        classification = any_later[0]
        next_action = "register_dense_local_counterfactual"
    elif len(any_later) > 1:
        classification = "mixed_dense_local_path_dependence"
        next_action = "register_local_counterfactual_attribution"
    else:
        classification = "dense_local_attribution_underdetermined"
        next_action = "close_optimizer_trajectory_attribution_family"
    route_specific = {
        route: [
            mechanism
            for mechanism in mechanisms
            if any(support[route][window][mechanism] for window in later_names)
        ]
        for route in arms
    }
    return (
        classification,
        next_action,
        {
            "support_by_route_window": support,
            "shared_supported_mechanisms": shared_later,
            "any_supported_mechanisms": any_later,
            "route_specific": route_specific,
            "alignment_estimator_independent": bool(shared_later),
            "early_only_mechanisms": early_only,
        },
    )


def _state_digest(
    encoder,
    decoder,
    optimizer,
    *,
    schedule_hash: str,
    step: int,
    cursor: int,
    history: list[dict[str, float]],
    history_window: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "encoder": _state_hash(encoder),
        "decoder": _state_hash(decoder),
        "optimizer": _tree_hash(optimizer.state_dict()),
        "rng": _rng_hash(),
        "schedule_hash": schedule_hash,
        "step": step,
        "cursor": cursor,
        "history": history,
        "history_window": history_window,
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
    schedule = tca.balanced_epoch_schedule(labels, seed=TRAIN_SEED, per_class=10)
    schedule_hash = canonical_sha256(
        [tensor_sha256(item.detach().cpu()) for item in schedule]
    )
    projection = tca._projection_bank(
        route=route, seed=PROJECTION_SEEDS[0], device=device
    )
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    encoder, decoder = tca._load_source_models(
        args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
    )
    optimizer = tca._optimizer(device, encoder, decoder)
    updates = []
    history_window = []
    component_gate = None
    observer_finite = False
    curvature_finite = False
    reverse_control = False
    for index in range(3):
        batch_images = images.index_select(0, schedule[index]).to(device)
        batch_labels = labels.index_select(0, schedule[index]).to(device)
        if index == 0:
            pre_parameters = [
                parameter.detach().clone()
                for parameter in tuple(encoder.parameters())
                + tuple(decoder.parameters())
            ]
            rng_before = torch.cuda.get_rng_state()
            prior = torch.randn((batch_images.shape[0], CODE_DIM), device=device)
            torch.cuda.set_rng_state(rng_before)
            values, _ = _training_measurement(
                encoder,
                decoder,
                optimizer,
                batch_images,
                batch_labels,
                prior,
                route=route,
                weights=weights,
                projections=projection,
            )
            gradients = tuple(
                torch.autograd.grad(
                    values["total"],
                    tuple(encoder.parameters()) + tuple(decoder.parameters()),
                    retain_graph=True,
                    allow_unused=True,
                )
            )
            components = _decompose_adamw(encoder, decoder, optimizer, gradients)
            observers = _observer_records(
                encoder,
                decoder,
                images.to(device),
                labels.to(device),
                route=route,
                weights=weights,
                include_classwise=False,
            )
        before = tca._parameter_vector(encoder, decoder)
        result = _update(
            encoder,
            decoder,
            optimizer,
            batch_images,
            batch_labels,
            route=route,
            weights=weights,
            projections=projection,
        )
        actual = tca._parameter_vector(encoder, decoder) - before
        updates.append(result)
        if index == 0:
            post_parameters = [
                parameter.detach().clone()
                for parameter in tuple(encoder.parameters())
                + tuple(decoder.parameters())
            ]
            component_gate = _component_gate(components["predicted"], actual)
            _attach_components(observers, components, actual)
            observer_finite = all(
                record.get("finite")
                and _finite_scalar_mapping(
                    record.get("component_directional_derivatives", {})
                )
                for record in observers
            )
            curvature = _curvature(
                encoder,
                decoder,
                observers[0],
                actual,
                pre_parameters,
                post_parameters,
                route=route,
                weights=weights,
            )
            curvature_finite = bool(curvature.get("finite"))
            reverse_control = all(
                abs(
                    record["reverse_directional_derivative"]
                    + record["actual_directional_derivative"]
                )
                <= max(1.0e-8, abs(record["actual_directional_derivative"]) * 1.0e-5)
                for record in observers
            )
        if route == "sced":
            history_window.append(
                {
                    "total": result["total"],
                    "reconstruction": result["reconstruction"],
                    "sced": result["alignment"],
                    "cycle": result["cycle"],
                }
            )
    uninterrupted = _state_digest(
        encoder,
        decoder,
        optimizer,
        schedule_hash=schedule_hash,
        step=3,
        cursor=3,
        history=[],
        history_window=history_window,
    )
    with tempfile.TemporaryDirectory(prefix="h_dlta_replay_") as directory:
        torch.manual_seed(TRAIN_SEED)
        torch.cuda.manual_seed_all(TRAIN_SEED)
        int_encoder, int_decoder = tca._load_source_models(
            args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
        )
        int_optimizer = tca._optimizer(device, int_encoder, int_decoder)
        first = _update(
            int_encoder,
            int_decoder,
            int_optimizer,
            images.index_select(0, schedule[0]).to(device),
            labels.index_select(0, schedule[0]).to(device),
            route=route,
            weights=weights,
            projections=projection,
        )
        int_history = (
            []
            if route != "sced"
            else [
                {
                    "total": first["total"],
                    "reconstruction": first["reconstruction"],
                    "sced": first["alignment"],
                    "cycle": first["cycle"],
                }
            ]
        )
        checkpoint = Path(directory) / "partial.pt"
        atomic_torch_save(
            checkpoint,
            {
                "encoder": {
                    key: value.detach().cpu()
                    for key, value in int_encoder.state_dict().items()
                },
                "decoder": {
                    key: value.detach().cpu()
                    for key, value in int_decoder.state_dict().items()
                },
                "optimizer": int_optimizer.state_dict(),
                "rng_state": {
                    "cpu": torch.get_rng_state().clone(),
                    "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
                },
                "schedule_hash": schedule_hash,
                "step": 1,
                "cursor": 1,
                "history": [],
                "history_window": int_history,
            },
        )
        res_encoder, res_decoder = tca._load_source_models(
            args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
        )
        res_optimizer = tca._optimizer(device, res_encoder, res_decoder)
        payload = safe_torch_load(checkpoint, map_location="cpu")
        res_encoder.load_state_dict(payload["encoder"], strict=True)
        res_decoder.load_state_dict(payload["decoder"], strict=True)
        res_optimizer.load_state_dict(payload["optimizer"])
        for state in res_optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        torch.set_rng_state(payload["rng_state"]["cpu"])
        torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])
        for index in (1, 2):
            result = _update(
                res_encoder,
                res_decoder,
                res_optimizer,
                images.index_select(0, schedule[index]).to(device),
                labels.index_select(0, schedule[index]).to(device),
                route=route,
                weights=weights,
                projections=projection,
            )
            if route == "sced":
                int_history.append(
                    {
                        "total": result["total"],
                        "reconstruction": result["reconstruction"],
                        "sced": result["alignment"],
                        "cycle": result["cycle"],
                    }
                )
        resumed = _state_digest(
            res_encoder,
            res_decoder,
            res_optimizer,
            schedule_hash=schedule_hash,
            step=3,
            cursor=3,
            history=[],
            history_window=int_history,
        )
    return {
        "route": route,
        "finite": all(_finite_scalar_mapping(row) for row in updates),
        "interrupted_resumed_exact": uninterrupted == resumed,
        "schedule_exact": uninterrupted["schedule_hash"] == resumed["schedule_hash"],
        "sampler_cursor_exact": uninterrupted["cursor"] == resumed["cursor"],
        "partial_history_exact": uninterrupted["history_window"]
        == resumed["history_window"],
        "component_gate": component_gate,
        "observer_finite": observer_finite,
        "directional_derivatives_finite": observer_finite,
        "curvature_finite": curvature_finite,
        "reverse_control": reverse_control,
        "uninterrupted_updates": updates,
    }


def _check_consumer(
    path: str, *, action: str, schemas: tuple[str, ...]
) -> dict[str, Any]:
    payload = _load_json(path)
    if (
        payload.get("schema") not in schemas
        or payload.get("status") != "valid"
        or action not in payload.get("permitted_consumers", ())
        or payload.get("invocation_mode") != "manifest"
        or payload.get("legacy_diagnostic_only") is not False
    ):
        raise ValueError(f"invalid H-DLTA {action} evidence: {path}")
    return payload


def _stage(
    args: argparse.Namespace, stage: str, expected: dict[str, Any]
) -> dict[str, Any]:
    args._stage_decision = stage_action_decision(
        stage,
        producer_contract=f"{SCHEMA_PREFIX}-{stage}.v1",
        expected_inputs=expected,
        evidence=args.output,
    )
    return args._stage_decision


def _audit(args: argparse.Namespace, lineage: dict[str, Any], output: Path) -> None:
    _require_cuda(args)
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    images, labels = tca._prepare_data(args.dataset_root)
    arms = {
        route: _replay_arm(
            args,
            route=route,
            final_payload=tca._route_payload(path, schema),
            images=images,
            labels=labels,
            output_root=Path(args._run_manifest["output_root"]),
        )
        for route, path, schema in (
            ("swae", args.swae_training_checkpoint, SWAE_SCHEMA),
            ("sced", args.sced_training_checkpoint, SCED_SCHEMA),
        )
    }
    failures = []
    for route, result in arms.items():
        if not result["final_match"]["equal"]:
            failures.append(f"{route}_historical_replay_mismatch")
        for landmark in result["landmarks"]:
            if not landmark.get("training_identity", True) or not landmark.get(
                "reverse_control", True
            ):
                failures.append(f"{route}_negative_control")
            gate = landmark.get("component_gate", {"passed": True, "finite": True})
            if not gate.get("passed") or not gate.get("finite"):
                failures.append(f"{route}_adamw_decomposition")
            for record in landmark.get("records", []):
                if not record.get("finite") or not _finite_scalar_mapping(
                    record.get("actual_delta", {})
                ):
                    failures.append(f"{route}_nonfinite_landmark")
            for item in landmark.get("curvature", []):
                if not item.get("finite"):
                    failures.append(f"{route}_nonfinite_curvature")
    classification, next_action, decision = _classify(arms)
    peak = int(torch.cuda.max_memory_allocated(device))
    storage = sum(
        item.stat().st_size
        for item in Path(args._run_manifest["output_root"]).rglob("*")
        if item.is_file()
    )
    if peak >= 6 * 1024**3:
        failures.append("cuda_peak_memory_gate")
    if storage > 2 * 1024**3:
        failures.append("run_storage_gate")
    valid = not failures
    if not valid:
        classification = "invalid_harness"
        next_action = "repair_h_dlta_harness_and_reissue_lineage"
    inputs = {
        key: file_sha256(value)
        for key, value in {
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
            "h_tca_audit": args.h_tca_audit_evidence,
            "h_tca_report": args.h_tca_report_evidence,
            "h_tca_closeout": args.h_tca_closeout_evidence,
        }.items()
    }
    _write(
        output,
        {
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
            "input_artifact_hashes": inputs,
            "lineage_sha256": canonical_sha256(lineage),
            "consumer_contract": f"{SCHEMA_PREFIX}-report.v1",
            "permitted_consumers": ["report"] if valid else [],
            "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1",
        },
        args,
    )


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("H-DLTA lifecycle is manifest-only")
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
        "h_pgc_audit_evidence",
        "h_pgc_report_evidence",
        "h_pgc_closeout_evidence",
        "h_tca_audit_evidence",
        "h_tca_report_evidence",
        "h_tca_closeout_evidence",
    )
    required = {
        "validate": common,
        "preflight": common + ("validation_evidence",),
        "audit": common + ("preflight_evidence",),
        "report": ("config", "dataset_root", "output", "measurement_evidence"),
        "closeout": ("config", "dataset_root", "output", "measurement_evidence"),
    }
    fields = common + (
        "device",
        "validation_evidence",
        "preflight_evidence",
        "measurement_evidence",
        "resume_checkpoint",
    )
    args = prepare_args(
        args,
        program=PROGRAM,
        action=action,
        required=required[action],
        defaults={"device": "cuda", "seed": HARNESS_SEED},
        mixed_fields=fields,
    )
    validate_manifest_namespace(args, action=action)
    output = Path(args.output)
    config = _config(args.config)
    lineage = (
        _lineage(args, config) if action in {"validate", "preflight", "audit"} else None
    )
    if action == "validate":
        decision = _stage(
            args,
            "validate",
            {
                "config": file_sha256(args.config),
                "source_training": file_sha256(args.wae_training_checkpoint),
                "lineage_sha256": canonical_sha256(lineage),
            },
        )
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        _write(
            output,
            {
                "schema": f"{SCHEMA_PREFIX}_validation.v1",
                "stage": "validate",
                "status": "valid",
                "passed": True,
                "lineage_sha256": canonical_sha256(lineage),
                "producer_contract_revision": f"{SCHEMA_PREFIX}-validation.v1",
                "consumer_contract": f"{SCHEMA_PREFIX}-preflight.v1",
                "permitted_consumers": ["preflight"],
                "input_artifact_hashes": {
                    "source_training": file_sha256(args.wae_training_checkpoint)
                },
            },
            args,
        )
        return 0
    if action == "preflight":
        validation = _check_consumer(
            args.validation_evidence,
            action="preflight",
            schemas=(f"{SCHEMA_PREFIX}_validation.v1",),
        )
        if not validation.get("passed"):
            raise ValueError("H-DLTA preflight requires passed validation")
        decision = _stage(
            args,
            "preflight",
            {
                "validation": file_sha256(args.validation_evidence),
                "source_training": file_sha256(args.wae_training_checkpoint),
                "lineage_sha256": canonical_sha256(lineage),
            },
        )
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        device = torch.device(args.device)
        torch.cuda.reset_peak_memory_stats(device)
        images, labels = tca._prepare_data(args.dataset_root)
        results = {
            route: _short_replay(
                args,
                route=route,
                weights=dict(tca._route_payload(path, schema)["weights"]),
                images=images,
                labels=labels,
            )
            for route, path, schema in (
                ("swae", args.swae_training_checkpoint, SWAE_SCHEMA),
                ("sced", args.sced_training_checkpoint, SCED_SCHEMA),
            )
        }
        storage = sum(
            item.stat().st_size
            for item in Path(args._run_manifest["output_root"]).rglob("*")
            if item.is_file()
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        passed = (
            all(
                item["finite"]
                and item["interrupted_resumed_exact"]
                and item["schedule_exact"]
                and item["sampler_cursor_exact"]
                and item["partial_history_exact"]
                and item["component_gate"]["passed"]
                and item["observer_finite"]
                and item["curvature_finite"]
                and item["reverse_control"]
                for item in results.values()
            )
            and peak < 6 * 1024**3
            and storage <= 2 * 1024**3
        )
        _write(
            output,
            {
                "schema": f"{SCHEMA_PREFIX}_preflight.v1",
                "stage": "preflight",
                "status": "valid" if passed else "invalid_harness",
                "passed": passed,
                "measurement_valid": passed,
                "results": results,
                "optimizer_created": True,
                "evaluator_calls": 0,
                "peak_memory_bytes": peak,
                "storage_bytes": storage,
                "memory_gate": peak < 6 * 1024**3,
                "storage_gate": storage <= 2 * 1024**3,
                "input_artifact_hashes": {
                    "validation": file_sha256(args.validation_evidence)
                },
                "lineage_sha256": canonical_sha256(lineage),
                "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1",
                "permitted_consumers": ["audit"] if passed else [],
                "producer_contract_revision": f"{SCHEMA_PREFIX}-preflight.v1",
            },
            args,
        )
        return 0
    if action == "audit":
        preflight = _check_consumer(
            args.preflight_evidence,
            action="audit",
            schemas=(f"{SCHEMA_PREFIX}_preflight.v1",),
        )
        if not preflight.get("passed"):
            raise ValueError("H-DLTA audit requires passed preflight")
        decision = _stage(
            args,
            "audit",
            {
                "preflight": file_sha256(args.preflight_evidence),
                "source_training": file_sha256(args.wae_training_checkpoint),
                "lineage_sha256": canonical_sha256(lineage),
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
        args, action, {"measurement": file_sha256(args.measurement_evidence)}
    )
    if decision["action"] == "REUSE":
        return 0
    if action == "report":
        _write(
            output,
            {
                "schema": f"{SCHEMA_PREFIX}_report.v1",
                "stage": "report",
                "status": "valid",
                "passed": bool(measurement.get("passed")),
                "measurement_valid": measurement.get("measurement_valid"),
                "classification": measurement.get("classification"),
                "next_action": measurement.get("next_action"),
                "measurement": args.measurement_evidence,
                "source_evidence_sha256": file_sha256(args.measurement_evidence),
                "input_artifact_hashes": {
                    "measurement": file_sha256(args.measurement_evidence)
                },
                "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1",
                "permitted_consumers": ["closeout"],
                "producer_contract_revision": f"{SCHEMA_PREFIX}-report.v1",
            },
            args,
        )
        return 0
    _write(
        output,
        {
            "schema": f"{SCHEMA_PREFIX}_closeout.v1",
            "stage": "closeout",
            "status": "valid",
            "passed": bool(measurement.get("passed")),
            "measurement_valid": measurement.get("measurement_valid"),
            "classification": measurement.get("classification"),
            "next_action": measurement.get("next_action"),
            "measurement": args.measurement_evidence,
            "source_evidence_sha256": file_sha256(args.measurement_evidence),
            "formal_promotion": False,
            "optical_training_authorized": False,
            "fiber_training_authorized": False,
            "synthetic_diagnostic_only": True,
            "input_artifact_hashes": {
                "measurement": file_sha256(args.measurement_evidence)
            },
            "consumer_contract": f"{SCHEMA_PREFIX}-closeout.v1",
            "permitted_consumers": [],
            "producer_contract_revision": f"{SCHEMA_PREFIX}-closeout.v1",
        },
        args,
    )
    return 0
