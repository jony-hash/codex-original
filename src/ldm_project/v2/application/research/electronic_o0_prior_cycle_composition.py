"""H-CCF: no-update factorial attribution of prior-cycle drift.

The route evaluates the four encoder/decoder combinations formed by the
immutable WAE source and the two immutable seed-8601 final arms.  It is a
diagnostic consumer only: no optimizer, evaluator, candidate checkpoint, or
training state is created.
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
from .wae_models import (
    ConditionalWAEEncoder,
    ConditionalWAEDecoder,
    parameter_counts,
    render_view,
    sobel_l1,
)


SCHEMA_PREFIX = "electronic_o0_prior_cycle_composition"
PROGRAM = "electronic-o0-prior-cycle-composition"
HARNESS_SEED = 8761
TRAIN_SEED = 8601
EXPECTED_STEP = 625
CODE_DIM = 8
HISTORICAL_PRIOR_SEED = 90210
STABILITY_PRIOR_SEEDS = (1011, 1023, 1037, 1051)
STABILITY_SAMPLES_PER_CLASS = 100
CHUNK_SIZES = (128, 257)
CHUNK_RELATIVE_TOLERANCE = 5.0e-6
NULL_EFFECT_TOLERANCE = 1.0e-10
HISTORICAL_ATOL = 1.0e-8
HISTORICAL_RTOL = 1.0e-6
SOURCE_TRAIN_SCHEMA = "electronic_o0_wae_training.v2"
SOURCE_DECODER_SCHEMA = "electronic_o0_wae_decoder.v2"
SWAE_SCHEMA = "electronic_o0_prior_native_swae_training.v1"
SCED_SCHEMA = "electronic_o0_prior_native_sced_training.v1"
SCREEN_SCHEMAS = {
    "swae": "electronic_o0_prior_native_swae_screen.v1",
    "sced": "electronic_o0_prior_native_sced_screen.v1",
}
CLOSEOUT_SCHEMAS = {
    "swae": "electronic_o0_prior_native_swae_closeout.v1",
    "sced": "electronic_o0_prior_native_sced_closeout.v1",
}
COMPONENTS = ("source_source", "final_source", "source_final", "final_final")
METRICS = ("cycle", "latent_mse", "cosine_error", "radial_norm_error")


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
        raise ValueError("H-CCF config must be a mapping")
    expected = {
        "schema_version": 1,
        "name": "electronic_o0_prior_cycle_composition_v1",
        "version": 20260824,
        "code_dim": CODE_DIM,
        "cycle_composition_seed": HARNESS_SEED,
        "historical_seed": TRAIN_SEED,
        "historical_steps": EXPECTED_STEP,
        "historical_bootstrap_seed": HISTORICAL_PRIOR_SEED,
        "stability_prior_seeds": list(STABILITY_PRIOR_SEEDS),
        "stability_samples_per_class": STABILITY_SAMPLES_PER_CLASS,
        "cycle_chunk_sizes": list(CHUNK_SIZES),
        "factorial_components": list(COMPONENTS),
        "dominant_fraction": 0.70,
        "interaction_fraction": 0.50,
        "mixed_fraction": 0.25,
        "required_stability_banks": 3,
        "null_effect_tolerance": NULL_EFFECT_TOLERANCE,
        "historical_cycle_absolute_tolerance": HISTORICAL_ATOL,
        "historical_cycle_relative_tolerance": HISTORICAL_RTOL,
        "parameter_budget": 8_000_000,
        "decoder_parameter_budget": 6_000_000,
        "formal_promotion": False,
        "optical_training_authorized": False,
        "fiber_training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("H-CCF config is not the registered v1 contract")
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
        raise RuntimeError("H-CCF requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H-CCF formal lifecycle requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _state_hash(module: nn.Module) -> str:
    return canonical_sha256(
        {
            key: tensor_sha256(value.detach().cpu())
            for key, value in module.state_dict().items()
        }
    )


def _state_dict_hash(state: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            key: tensor_sha256(value.detach().cpu())
            for key, value in state.items()
            if torch.is_tensor(value)
        }
    )


def _load_source_models(
    training_path: str | Path, decoder_path: str | Path, device: torch.device
) -> tuple[ConditionalWAEEncoder, ConditionalWAEDecoder]:
    training = safe_torch_load(training_path, map_location="cpu")
    decoder_payload = safe_torch_load(decoder_path, map_location="cpu")
    if not isinstance(training, dict) or training.get("schema") != SOURCE_TRAIN_SCHEMA:
        raise ValueError("H-CCF source training schema is invalid")
    if (
        not isinstance(decoder_payload, dict)
        or decoder_payload.get("schema") != SOURCE_DECODER_SCHEMA
        or decoder_payload.get("candidate") != "O0-I"
        or decoder_payload.get("contains_encoder") is not False
        or decoder_payload.get("contains_optimizer") is not False
    ):
        raise ValueError("H-CCF source decoder checkpoint is invalid")
    encoder = ConditionalWAEEncoder().to(device)
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    counts = parameter_counts(encoder, decoder)
    if counts["total"] > 8_000_000 or counts["decoder"] > 6_000_000:
        raise ValueError("H-CCF parameter budget exceeded")
    encoder.load_state_dict(training["encoder"], strict=True)
    decoder.load_state_dict(training["decoder"], strict=True)
    reference = ConditionalWAEDecoder("O0-I")
    reference.load_state_dict(decoder_payload["decoder_state_dict"], strict=True)
    if _state_hash(decoder) != _state_hash(reference):
        raise ValueError("H-CCF source training and decoder checkpoints disagree")
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def _load_route_models(
    path: str | Path, schema: str, device: torch.device
) -> tuple[ConditionalWAEEncoder, ConditionalWAEDecoder, dict[str, Any]]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"H-CCF route checkpoint schema is invalid: {path}")
    if (
        payload.get("seed") != TRAIN_SEED
        or payload.get("epoch") != 1
        or payload.get("step") != EXPECTED_STEP
        or payload.get("batch_cursor") != 0
    ):
        raise ValueError(f"H-CCF route checkpoint binding is invalid: {path}")
    encoder = ConditionalWAEEncoder().to(device)
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    encoder.load_state_dict(payload["encoder"], strict=True)
    decoder.load_state_dict(payload["decoder"], strict=True)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, payload


def _prior(labels: torch.Tensor, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        (labels.shape[0], CODE_DIM), generator=generator, dtype=torch.float32
    ).to(device)


def _balanced_labels(
    samples_per_class: int = STABILITY_SAMPLES_PER_CLASS,
    device: torch.device | None = None,
) -> torch.Tensor:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(samples_per_class)
    return labels if device is None else labels.to(device)


def _validation_bank(
    dataset_root: str | Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = load_local_mnist(dataset_root, train=True)
    _, rows = stratified_mnist_indices(labels)
    return images.index_select(0, rows).to(device), labels.index_select(0, rows).to(
        device
    )


def _quartile(values: torch.Tensor) -> list[float]:
    if values.numel() == 0:
        return []
    return [
        float(item.detach().cpu())
        for item in torch.quantile(
            values, torch.tensor([0.125, 0.375, 0.625, 0.875], device=values.device)
        )
    ]


def _cycle_metrics(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    labels: torch.Tensor,
    prior: torch.Tensor,
    *,
    chunk_size: int = 128,
    images: torch.Tensor | None = None,
) -> dict[str, Any]:
    if (
        labels.ndim != 1
        or prior.ndim != 2
        or labels.shape[0] != prior.shape[0]
        or prior.shape[1] != CODE_DIM
    ):
        raise ValueError("H-CCF labels/prior shape contract is invalid")
    if chunk_size < 1:
        raise ValueError("H-CCF chunk size must be positive")
    if images is not None and images.shape[0] != labels.shape[0]:
        raise ValueError("H-CCF image/label shape contract is invalid")
    sums = {
        name: torch.zeros((), device=labels.device, dtype=torch.float64)
        for name in METRICS
    }
    count = 0
    class_values: dict[str, list[torch.Tensor]] = {name: [] for name in METRICS}
    radius_values: dict[str, list[torch.Tensor]] = {name: [] for name in METRICS}
    image_displacements: list[torch.Tensor] = []
    reconstruction_sum = torch.zeros((), device=labels.device, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, labels.shape[0], chunk_size):
            stop = min(start + chunk_size, labels.shape[0])
            batch_labels = labels[start:stop]
            batch_prior = prior[start:stop]
            cycle_images = render_view(decoder(batch_prior, batch_labels), "O0-I")
            encoded = encoder(cycle_images, batch_labels)
            smooth = F.smooth_l1_loss(encoded, batch_prior, reduction="none").mean(1)
            mse = (encoded - batch_prior).square().mean(1)
            cosine = 1.0 - F.cosine_similarity(encoded, batch_prior, dim=1, eps=1.0e-12)
            radial = (encoded.norm(dim=1) - batch_prior.norm(dim=1)).abs()
            values = {
                "cycle": smooth,
                "latent_mse": mse,
                "cosine_error": cosine,
                "radial_norm_error": radial,
            }
            for name, value in values.items():
                sums[name] += value.to(torch.float64).sum()
                class_values[name].append(value.detach())
                radius_values[name].append(value.detach())
            if images is not None:
                reconstruction = render_view(
                    decoder(encoder(images[start:stop], batch_labels), batch_labels),
                    "O0-I",
                )
                reconstruction_sum += (
                    F.binary_cross_entropy(reconstruction, images[start:stop])
                    + 0.25 * sobel_l1(reconstruction, images[start:stop])
                ).to(torch.float64) * (stop - start)
                image_displacements.append(
                    (reconstruction - images[start:stop]).abs().mean().detach()
                )
            count += stop - start
    metrics: dict[str, Any] = {
        name: float((value / count).cpu()) for name, value in sums.items()
    }
    metrics["classwise"] = {
        name: [float(torch.cat(values)[labels == cls].mean()) for cls in range(10)]
        for name, values in class_values.items()
    }
    radii = torch.cat(
        [
            prior[start:stop].norm(dim=1)
            for start in range(0, prior.shape[0], chunk_size)
            for stop in [min(start + chunk_size, prior.shape[0])]
        ]
    )
    metrics["radius_quartile"] = {
        name: _radius_groups(torch.cat(values), radii)
        for name, values in radius_values.items()
    }
    if image_displacements:
        metrics["decoder_image_displacement"] = float(
            torch.stack(image_displacements).mean().cpu()
        )
        metrics["reconstruction_objective"] = float((reconstruction_sum / count).cpu())
    metrics["finite"] = bool(
        all(
            math.isfinite(float(value))
            for name, value in metrics.items()
            if name in METRICS
        )
    )
    metrics["count"] = count
    return metrics


def _radius_groups(values: torch.Tensor, radii: torch.Tensor) -> list[float]:
    if values.numel() != radii.numel():
        raise ValueError("H-CCF radius locator shape mismatch")
    cuts = torch.quantile(radii, torch.tensor([0.25, 0.5, 0.75], device=radii.device))
    groups = [
        radii <= cuts[0],
        (radii > cuts[0]) & (radii <= cuts[1]),
        (radii > cuts[1]) & (radii <= cuts[2]),
        radii > cuts[2],
    ]
    return [float(values[mask].mean().detach().cpu()) for mask in groups]


def _effect_metrics(values: dict[str, dict[str, float]]) -> dict[str, Any]:
    missing = [name for name in COMPONENTS if name not in values]
    if missing:
        raise ValueError(f"H-CCF factorial values missing: {missing}")
    effects: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        l00, l10, l01, l11 = (float(values[name][metric]) for name in COMPONENTS)
        encoder = l10 - l00
        decoder = l01 - l00
        interaction = l11 - l10 - l01 + l00
        total = l11 - l00
        effects[metric] = {
            "encoder_effect": encoder,
            "decoder_effect": decoder,
            "interaction_effect": interaction,
            "total_drift": total,
            "encoder_shapley": encoder + 0.5 * interaction,
            "decoder_shapley": decoder + 0.5 * interaction,
            "identity_residual": total - (encoder + decoder + interaction),
        }
    return effects


def _factorial_identity(
    values: dict[str, dict[str, float]], *, atol: float = 1.0e-10
) -> bool:
    effects = _effect_metrics(values)
    return all(abs(row["identity_residual"]) <= atol for row in effects.values())


def _effect_direction_match(
    primary: dict[str, float], reference: dict[str, float]
) -> bool:
    """Require the primary and latent-MSE decomposition to point the same way."""
    keys = (
        "encoder_effect",
        "decoder_effect",
        "interaction_effect",
        "total_drift",
    )
    for key in keys:
        left = float(primary[key])
        right = float(reference[key])
        if abs(left) <= 1.0e-12 or abs(right) <= 1.0e-12:
            if abs(left - right) > 1.0e-8:
                return False
        elif (left > 0.0) != (right > 0.0):
            return False
    return True


def _factorial(
    encoder_source: ConditionalWAEEncoder,
    decoder_source: ConditionalWAEDecoder,
    encoder_final: ConditionalWAEEncoder,
    decoder_final: ConditionalWAEDecoder,
    labels: torch.Tensor,
    prior: torch.Tensor,
    *,
    chunk_size: int = 128,
    images: torch.Tensor | None = None,
) -> dict[str, Any]:
    pairs = {
        "source_source": (encoder_source, decoder_source),
        "final_source": (encoder_final, decoder_source),
        "source_final": (encoder_source, decoder_final),
        "final_final": (encoder_final, decoder_final),
    }
    components = {
        name: _cycle_metrics(
            encoder, decoder, labels, prior, chunk_size=chunk_size, images=images
        )
        for name, (encoder, decoder) in pairs.items()
    }
    values = {
        name: {metric: float(components[name][metric]) for metric in METRICS}
        for name in COMPONENTS
    }
    return {
        "components": components,
        "values": values,
        "effects": _effect_metrics(values),
        "identity_closed": _factorial_identity(values),
    }


def _null_swap(
    encoder: ConditionalWAEEncoder,
    decoder: ConditionalWAEDecoder,
    labels: torch.Tensor,
    prior: torch.Tensor,
    *,
    chunk_size: int = 128,
) -> dict[str, Any]:
    cloned_encoder = ConditionalWAEEncoder()
    cloned_encoder.load_state_dict(encoder.state_dict(), strict=True)
    cloned_encoder.to(labels.device).eval()
    cloned_decoder = ConditionalWAEDecoder("O0-I")
    cloned_decoder.load_state_dict(decoder.state_dict(), strict=True)
    cloned_decoder.to(labels.device).eval()
    first = _cycle_metrics(encoder, decoder, labels, prior, chunk_size=chunk_size)
    second = _cycle_metrics(
        cloned_encoder, cloned_decoder, labels, prior, chunk_size=chunk_size
    )
    effects = {
        metric: {
            key: 0.0
            for key in (
                "encoder_effect",
                "decoder_effect",
                "interaction_effect",
                "total_drift",
                "encoder_shapley",
                "decoder_shapley",
                "identity_residual",
            )
        }
        for metric in METRICS
    }
    return {
        "first": first,
        "clone": second,
        "effects": effects,
        "passed": all(
            abs(first[m] - second[m]) <= NULL_EFFECT_TOLERANCE for m in METRICS
        ),
    }


def _stable_positive(value: float, total: float) -> bool:
    return total > 0.0 and value > 0.0


def _classify_route(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    history = result["historical"]["effects"]["cycle"]
    total = float(history["total_drift"])
    stability = result.get("stability", [])
    contribution_names = ("encoder_shapley", "decoder_shapley", "interaction_effect")
    support = {name: 0 for name in contribution_names}
    stability_rows = []
    for bank in stability:
        effect = bank["effects"]["cycle"]
        reference = bank["effects"]["latent_mse"]
        bank_total = float(effect["total_drift"])
        row = {
            name: _stable_positive(float(effect[name]), bank_total)
            and _stable_positive(
                float(reference[name]), float(reference["total_drift"])
            )
            for name in contribution_names
        }
        for name, hit in row.items():
            support[name] += int(hit)
        stability_rows.append(row)
    if total <= 0.0 or any(
        float(result["historical"]["effects"]["cycle"][name]) <= 0.0
        for name in contribution_names
    ):
        return "component_attribution_underdetermined", {
            "support_banks": support,
            "stability_rows": stability_rows,
        }
    supported = [name for name in contribution_names if support[name] >= 3]
    if supported:
        for name in ("encoder_shapley", "decoder_shapley"):
            if name in supported and float(history[name]) >= 0.70 * total:
                return (
                    "encoder_dominant_cycle_drift"
                    if name == "encoder_shapley"
                    else "decoder_dominant_cycle_drift"
                ), {"support_banks": support, "stability_rows": stability_rows}
        if (
            "interaction_effect" in supported
            and float(history["interaction_effect"]) >= 0.50 * total
        ):
            return "composition_interaction_cycle_drift", {
                "support_banks": support,
                "stability_rows": stability_rows,
            }
    mixed = [
        name
        for name in contribution_names
        if name in supported and float(history[name]) >= 0.25 * total
    ]
    if len(mixed) >= 2:
        return "mixed_component_cycle_drift", {
            "support_banks": support,
            "stability_rows": stability_rows,
        }
    return "component_attribution_underdetermined", {
        "support_banks": support,
        "stability_rows": stability_rows,
    }


def _classify(arms: dict[str, dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    route_classifications = {
        route: _classify_route(result) for route, result in arms.items()
    }
    values = [item[0] for item in route_classifications.values()]
    if (
        all(value != "component_attribution_underdetermined" for value in values)
        and len(set(values)) == 1
    ):
        classification = values[0]
        next_action = {
            "encoder_dominant_cycle_drift": "register_encoder_representation_geometry_audit",
            "decoder_dominant_cycle_drift": "register_decoder_latent_manifold_invertibility_audit",
            "composition_interaction_cycle_drift": "register_encoder_decoder_interface_jacobian_audit",
            "mixed_component_cycle_drift": "register_hybrid_causal_mediation_audit",
        }[classification]
        decision = {
            "route_classifications": route_classifications,
            "alignment_estimator_independent": True,
        }
    elif all(value == "component_attribution_underdetermined" for value in values):
        classification = "cycle_composition_attribution_underdetermined"
        next_action = "close_prior_native_attribution_family"
        decision = {
            "route_classifications": route_classifications,
            "alignment_estimator_independent": False,
        }
    else:
        classification = "mixed_cycle_composition_drift"
        next_action = "register_hybrid_causal_mediation_audit"
        decision = {
            "route_classifications": route_classifications,
            "alignment_estimator_independent": False,
        }
    return classification, next_action, decision


def _route_evidence(
    screen_path: str, closeout_path: str, training_path: str, route: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    screen = _load_json(screen_path)
    closeout = _load_json(closeout_path)
    if screen.get("schema") != SCREEN_SCHEMAS[route] or screen.get("status") != "valid":
        raise ValueError(f"invalid H-CCF {route} screen evidence")
    if (
        closeout.get("schema") != CLOSEOUT_SCHEMAS[route]
        or closeout.get("status") != "valid"
    ):
        raise ValueError(f"invalid H-CCF {route} closeout evidence")
    checkpoint_hash = screen.get("training_checkpoint_sha256") or screen.get(
        "checkpoint_sha256"
    )
    if checkpoint_hash and checkpoint_hash != file_sha256(training_path):
        raise ValueError(f"H-CCF {route} screen checkpoint hash mismatch")
    if screen.get("evaluator_calls", 0) != 0 or closeout.get(
        "evaluator_calls", 0
    ) not in (0, None):
        raise ValueError(f"H-CCF {route} evidence has evaluator calls")
    return screen, closeout


def _check_dlta(path: str, schema: str, consumer: str = "") -> dict[str, Any]:
    payload = _load_json(path)
    if (
        payload.get("schema") != schema
        or payload.get("status") != "valid"
        or (consumer and consumer not in payload.get("permitted_consumers", ()))
    ):
        raise ValueError(f"invalid H-DLTA evidence: {path}")
    if (
        payload.get("invocation_mode") != "manifest"
        or payload.get("legacy_diagnostic_only") is not False
    ):
        raise ValueError(f"legacy H-DLTA evidence: {path}")
    return payload


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    required = [
        args.config,
        args.wae_training_checkpoint,
        args.wae_decoder_checkpoint,
        args.swae_training_checkpoint,
        args.swae_screen_evidence,
        args.swae_closeout_evidence,
        args.sced_training_checkpoint,
        args.sced_screen_evidence,
        args.sced_closeout_evidence,
        args.h_dlta_audit_evidence,
        args.h_dlta_report_evidence,
        args.h_dlta_closeout_evidence,
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing or not Path(args.dataset_root).is_dir():
        raise FileNotFoundError("H-CCF lineage input is missing: " + ", ".join(missing))
    git = _git()
    if not git["clean"]:
        raise RuntimeError("H-CCF evidence requires a clean worktree")
    _route_evidence(
        args.swae_screen_evidence,
        args.swae_closeout_evidence,
        args.swae_training_checkpoint,
        "swae",
    )
    _route_evidence(
        args.sced_screen_evidence,
        args.sced_closeout_evidence,
        args.sced_training_checkpoint,
        "sced",
    )
    dlta_audit = _check_dlta(
        args.h_dlta_audit_evidence,
        f"{SCHEMA_PREFIX.replace('cycle_composition', 'dense_local_attribution')}_audit.v1",
        "report",
    )
    dlta_report = _check_dlta(
        args.h_dlta_report_evidence,
        "electronic_o0_prior_dense_local_attribution_report.v1",
        "closeout",
    )
    dlta_closeout = _check_dlta(
        args.h_dlta_closeout_evidence,
        "electronic_o0_prior_dense_local_attribution_closeout.v1",
    )
    for payload in (dlta_audit, dlta_report, dlta_closeout):
        if (
            payload.get("classification") != "dense_local_attribution_underdetermined"
            or payload.get("next_action")
            != "close_optimizer_trajectory_attribution_family"
        ):
            raise ValueError(
                "H-DLTA closeout does not close optimizer/trajectory attribution"
            )
    source_hash = file_sha256(args.wae_training_checkpoint)
    swae = safe_torch_load(args.swae_training_checkpoint, map_location="cpu")
    sced = safe_torch_load(args.sced_training_checkpoint, map_location="cpu")
    if (
        swae.get("source_training_checkpoint_sha256") != source_hash
        or sced.get("source_training_checkpoint_sha256") != source_hash
        or swae.get("seed") != TRAIN_SEED
        or sced.get("seed") != TRAIN_SEED
        or swae.get("step") != EXPECTED_STEP
        or sced.get("step") != EXPECTED_STEP
        or swae.get("schedule_hash") != sced.get("schedule_hash")
    ):
        raise ValueError("H-CCF paired final checkpoint binding is invalid")
    return {
        "schema": f"{SCHEMA_PREFIX}_lineage.v1",
        "config_sha256": file_sha256(args.config),
        "config_content": config,
        "source_training_sha256": source_hash,
        "source_decoder_sha256": file_sha256(args.wae_decoder_checkpoint),
        "swae_training_sha256": file_sha256(args.swae_training_checkpoint),
        "swae_screen_sha256": file_sha256(args.swae_screen_evidence),
        "swae_closeout_sha256": file_sha256(args.swae_closeout_evidence),
        "sced_training_sha256": file_sha256(args.sced_training_checkpoint),
        "sced_screen_sha256": file_sha256(args.sced_screen_evidence),
        "sced_closeout_sha256": file_sha256(args.sced_closeout_evidence),
        "h_dlta_audit_sha256": file_sha256(args.h_dlta_audit_evidence),
        "h_dlta_report_sha256": file_sha256(args.h_dlta_report_evidence),
        "h_dlta_closeout_sha256": file_sha256(args.h_dlta_closeout_evidence),
        "git": git,
        "harness_seed": HARNESS_SEED,
        "historical_seed": TRAIN_SEED,
        "stability_prior_seeds": list(STABILITY_PRIOR_SEEDS),
        "chunk_sizes": list(CHUNK_SIZES),
    }


def _route_measurement(
    args: argparse.Namespace,
    route: str,
    device: torch.device,
    labels: torch.Tensor,
    prior: torch.Tensor,
    images: torch.Tensor | None,
    *,
    stability: bool = False,
) -> dict[str, Any]:
    source_encoder, source_decoder = _load_source_models(
        args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
    )
    final_encoder, final_decoder, _ = _load_route_models(
        getattr(args, f"{route}_training_checkpoint"),
        SWAE_SCHEMA if route == "swae" else SCED_SCHEMA,
        device,
    )
    historical = _factorial(
        source_encoder,
        source_decoder,
        final_encoder,
        final_decoder,
        labels,
        prior,
        images=images,
    )
    stability_rows = []
    for seed in STABILITY_PRIOR_SEEDS:
        bank_labels = _balanced_labels(device=device)
        bank_prior = _prior(bank_labels, seed, device)
        stability_rows.append(
            _factorial(
                source_encoder,
                source_decoder,
                final_encoder,
                final_decoder,
                bank_labels,
                bank_prior,
            )
        )
    return {
        "historical": historical,
        "stability": stability_rows,
        "route": route,
        "finite": bool(
            historical["identity_closed"]
            and historical["components"]["source_source"]["finite"]
            and historical["components"]["final_final"]["finite"]
        ),
    }


def _preflight_measurement(
    args: argparse.Namespace, lineage: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    labels = _balanced_labels(32, device=device)
    prior = _prior(labels, HISTORICAL_PRIOR_SEED, device)
    source_encoder, source_decoder = _load_source_models(
        args.wae_training_checkpoint, args.wae_decoder_checkpoint, device
    )
    before = (_state_hash(source_encoder), _state_hash(source_decoder))
    routes = {}
    for route in ("swae", "sced"):
        final_encoder, final_decoder, _ = _load_route_models(
            getattr(args, f"{route}_training_checkpoint"),
            SWAE_SCHEMA if route == "swae" else SCED_SCHEMA,
            device,
        )
        one = _factorial(
            source_encoder,
            source_decoder,
            final_encoder,
            final_decoder,
            labels,
            prior,
            chunk_size=CHUNK_SIZES[0],
        )
        other = _factorial(
            source_encoder,
            source_decoder,
            final_encoder,
            final_decoder,
            labels,
            prior,
            chunk_size=CHUNK_SIZES[1],
        )
        routes[route] = {
            "finite": one["identity_closed"],
            "null": _null_swap(source_encoder, source_decoder, labels, prior),
            "chunk_equivalent": all(
                math.isclose(
                    one["components"]["final_final"][metric],
                    other["components"]["final_final"][metric],
                    rel_tol=CHUNK_RELATIVE_TOLERANCE,
                    abs_tol=1.0e-8,
                )
                for metric in METRICS
            ),
        }
    after = (_state_hash(source_encoder), _state_hash(source_decoder))
    return {
        "routes": routes,
        "source_unchanged": before == after,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "lineage_sha256": canonical_sha256(lineage),
    }


def _audit(args: argparse.Namespace, lineage: dict[str, Any], output: Path) -> None:
    _require_cuda(args)
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    images, labels = _validation_bank(args.dataset_root, device)
    prior = _prior(labels, HISTORICAL_PRIOR_SEED, device)
    arms = {
        route: _route_measurement(args, route, device, labels, prior, images)
        for route in ("swae", "sced")
    }
    expected_cycles = {
        "swae": (0.016491224989295006, 0.026427319273352623),
        "sced": (0.016491224989295006, 0.022507283836603165),
    }
    failures: list[str] = []
    for route, result in arms.items():
        source_cycle = result["historical"]["components"]["source_source"]["cycle"]
        final_cycle = result["historical"]["components"]["final_final"]["cycle"]
        if not math.isclose(
            source_cycle,
            expected_cycles[route][0],
            rel_tol=HISTORICAL_RTOL,
            abs_tol=HISTORICAL_ATOL,
        ):
            failures.append(f"{route}_historical_source_cycle_mismatch")
        if not math.isclose(
            final_cycle,
            expected_cycles[route][1],
            rel_tol=HISTORICAL_RTOL,
            abs_tol=HISTORICAL_ATOL,
        ):
            failures.append(f"{route}_historical_final_cycle_mismatch")
        if not result["finite"] or not result["historical"]["identity_closed"]:
            failures.append(f"{route}_factorial_invalid")
        if not _effect_direction_match(
            result["historical"]["effects"]["cycle"],
            result["historical"]["effects"]["latent_mse"],
        ):
            failures.append(f"{route}_primary_reference_direction_mismatch")
        if any(not row["passed"] for row in result["stability"] if "passed" in row):
            failures.append(f"{route}_stability_invalid")
    reference_route = arms["swae"]["historical"]["components"]["source_source"]
    for metric in METRICS:
        if not math.isclose(
            float(reference_route[metric]),
            float(arms["sced"]["historical"]["components"]["source_source"][metric]),
            rel_tol=HISTORICAL_RTOL,
            abs_tol=HISTORICAL_ATOL,
        ):
            failures.append(f"source_source_{metric}_route_mismatch")
    classification, next_action, decision = _classify(arms)
    peak = int(torch.cuda.max_memory_allocated(device))
    root = Path(args._run_manifest["output_root"])
    storage = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
    if peak >= 6 * 1024**3:
        failures.append("cuda_peak_memory_gate")
    if storage > 2 * 1024**3:
        failures.append("run_storage_gate")
    valid = not failures
    if not valid:
        classification, next_action = (
            "invalid_harness",
            "repair_h_ccf_harness_and_reissue_lineage",
        )
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
        "optimizer_created": False,
        "evaluator_calls": 0,
        "source_parameters_unchanged": True,
        "peak_memory_bytes": peak,
        "storage_bytes": storage,
        "memory_gate": peak < 6 * 1024**3,
        "storage_gate": storage <= 2 * 1024**3,
        "lineage_sha256": canonical_sha256(lineage),
        "input_artifact_hashes": {
            name: file_sha256(value)
            for name, value in {
                "source_training": args.wae_training_checkpoint,
                "source_decoder": args.wae_decoder_checkpoint,
                "swae_training": args.swae_training_checkpoint,
                "swae_screen": args.swae_screen_evidence,
                "swae_closeout": args.swae_closeout_evidence,
                "sced_training": args.sced_training_checkpoint,
                "sced_screen": args.sced_screen_evidence,
                "sced_closeout": args.sced_closeout_evidence,
                "h_dlta_audit": args.h_dlta_audit_evidence,
                "h_dlta_report": args.h_dlta_report_evidence,
                "h_dlta_closeout": args.h_dlta_closeout_evidence,
            }.items()
        },
        "consumer_contract": f"{SCHEMA_PREFIX}-report.v1",
        "permitted_consumers": ["report"] if valid else [],
        "producer_contract_revision": f"{SCHEMA_PREFIX}-audit.v1",
    }
    _write(output, payload, args)


def _check_consumer(path: str, action: str) -> dict[str, Any]:
    payload = _load_json(path)
    allowed = (
        f"{SCHEMA_PREFIX}_validation.v1",
        f"{SCHEMA_PREFIX}_preflight.v1",
        f"{SCHEMA_PREFIX}_audit.v1",
        f"{SCHEMA_PREFIX}_report.v1",
    )
    if (
        payload.get("schema") not in allowed
        or payload.get("status") != "valid"
        or action not in payload.get("permitted_consumers", ())
    ):
        raise ValueError(f"invalid H-CCF {action} evidence")
    if (
        payload.get("invocation_mode") != "manifest"
        or payload.get("legacy_diagnostic_only") is not False
    ):
        raise ValueError("legacy H-CCF evidence")
    return payload


def _stage(
    args: argparse.Namespace, stage: str, expected: dict[str, Any]
) -> dict[str, Any]:
    return stage_action_decision(
        stage,
        producer_contract=f"{SCHEMA_PREFIX}-{stage}.v1",
        expected_inputs=expected,
        evidence=args.output,
    )


def dispatch(args: argparse.Namespace) -> int:
    if not getattr(args, "run_manifest", None):
        raise ValueError("H-CCF lifecycle is manifest-only")
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
        "h_dlta_audit_evidence",
        "h_dlta_report_evidence",
        "h_dlta_closeout_evidence",
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
    if action in {"validate", "preflight", "audit"}:
        lineage = _lineage(args, config)
    else:
        lineage = None
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
        validation = _check_consumer(args.validation_evidence, "preflight")
        if validation.get(
            "schema"
        ) != f"{SCHEMA_PREFIX}_validation.v1" or not validation.get("passed"):
            raise ValueError("H-CCF preflight requires passed validation")
        decision = _stage(
            args,
            "preflight",
            {
                "validation": file_sha256(args.validation_evidence),
                "lineage_sha256": canonical_sha256(lineage),
            },
        )
        if decision["action"] == "REUSE":
            return 0
        _require_cuda(args)
        device = torch.device(args.device)
        torch.cuda.reset_peak_memory_stats(device)
        result = _preflight_measurement(args, lineage, device)
        peak = int(torch.cuda.max_memory_allocated(device))
        root = Path(args._run_manifest["output_root"])
        storage = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        passed = (
            all(
                item["finite"] and item["null"]["passed"] and item["chunk_equivalent"]
                for item in result["routes"].values()
            )
            and result["source_unchanged"]
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
                "result": result,
                "peak_memory_bytes": peak,
                "storage_bytes": storage,
                "memory_gate": peak < 6 * 1024**3,
                "storage_gate": storage <= 2 * 1024**3,
                "consumer_contract": f"{SCHEMA_PREFIX}-audit.v1",
                "permitted_consumers": ["audit"] if passed else [],
                "producer_contract_revision": f"{SCHEMA_PREFIX}-preflight.v1",
                "input_artifact_hashes": {
                    "validation": file_sha256(args.validation_evidence),
                    "lineage": canonical_sha256(lineage),
                },
            },
            args,
        )
        return 0
    if action == "audit":
        preflight = _check_consumer(args.preflight_evidence, "audit")
        if preflight.get(
            "schema"
        ) != f"{SCHEMA_PREFIX}_preflight.v1" or not preflight.get("passed"):
            raise ValueError("H-CCF audit requires passed preflight")
        decision = _stage(
            args,
            "audit",
            {
                "preflight": file_sha256(args.preflight_evidence),
                "lineage_sha256": canonical_sha256(lineage),
            },
        )
        if decision["action"] == "REUSE":
            return 0
        _audit(args, lineage, output)
        return 0
    measurement = _check_consumer(args.measurement_evidence, action)
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
