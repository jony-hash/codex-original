"""Fail-closed paired image-generation research lane for synthetic SMF operators."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from ...config import load_config
from ...evidence import (
    atomic_torch_save,
    atomic_write_json,
    file_sha256,
    safe_torch_load,
)
from ...optics.fiber_operator import (
    FiberTransportEnsembleOLS,
    lp01_finite_difference_eigenpair,
    lp01_radial_profile,
    lp01_reference_metrics,
)
from ...optics.fiber_pool import FiberTransportPool
from ...student.conditional import (
    camera_code_fraction,
    diversity_hinge,
    fixed_exposure_view,
    log_sinkhorn,
    phase_total_variation,
    raw_from_generator_batched,
    relative_diversity_ratios,
)
from ...student.conditional_acceptance import conditional_coverage_ratio
from ...student.evaluator import (
    SmoothMNISTEvaluator,
    balanced_batches,
    load_frozen_evaluator,
    load_local_mnist,
)
from .._runtime_shared import random_codes, split_mnist
from ..current.runtime import _seeded_conditional_bundle, _system_signature
from . import fiber_robustness as robust

FUNCTION_IDS = ("I0", "S0", "S1a", "S1b", "S1c", "N0")
TRAIN_SEEDS = (5201, 5202, 5203)
PANEL_SEEDS = (4242, 104729, 161803)


def _adaptive_average_weights(input_size: int, output_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the exact linear weights used by adaptive average pooling."""
    if input_size < 1 or output_size < 1:
        raise ValueError("pool dimensions must be positive")
    weights = torch.zeros(output_size, input_size, device=device, dtype=dtype)
    for index in range(output_size):
        start = (index * input_size) // output_size
        end = ((index + 1) * input_size + output_size - 1) // output_size
        weights[index, start:end] = 1.0 / float(end - start)
    return weights


def _deterministic_adaptive_average(values: torch.Tensor, output_size: int | tuple[int, int]) -> torch.Tensor:
    """Adaptive average pooling as two fixed matrix multiplies.

    The research lane uses this replacement because CUDA's native adaptive
    pooling backward is not deterministic for the 200->28 and 28->14 paths.
    """
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("deterministic pooling expects [N,1,H,W]")
    target_h, target_w = (output_size, output_size) if isinstance(output_size, int) else output_size
    height, width = values.shape[-2:]
    wy = _adaptive_average_weights(height, target_h, device=values.device, dtype=values.dtype)
    wx = _adaptive_average_weights(width, target_w, device=values.device, dtype=values.dtype)
    image = values[:, 0]
    return torch.matmul(torch.matmul(wy, image), wx.t()).unsqueeze(1)


def _deterministic_fixed_exposure_view(raw_cmos: torch.Tensor, exposure_scale: float, *, output_size: int = 28) -> torch.Tensor:
    if raw_cmos.ndim != 4 or raw_cmos.shape[1] != 1 or exposure_scale <= 0.0:
        raise ValueError("raw CMOS must be [batch, 1, H, W] with a positive fixed exposure")
    return _deterministic_adaptive_average((raw_cmos / exposure_scale).clamp(0.0, 1.0), output_size)


def _deterministic_sinkhorn_cost_view(view: torch.Tensor, *, output_size: int = 14) -> torch.Tensor:
    if view.ndim != 4 or view.shape[1] != 1:
        raise ValueError("view must be [batch, 1, H, W]")
    return _deterministic_adaptive_average(view, output_size)


def _research_sinkhorn_loss(generated_features: torch.Tensor, generated_view: torch.Tensor, real_features: torch.Tensor, real_view: torch.Tensor, labels: torch.Tensor, *, epsilon: float, iterations: int) -> torch.Tensor:
    if generated_features.shape != real_features.shape or generated_view.shape != real_view.shape:
        raise ValueError("generated and real batches must have matching shapes")
    terms = []
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()
        if selected.numel() < 2:
            raise ValueError("each conditional Sinkhorn batch needs at least two samples per label")
        generated_f = torch.nn.functional.normalize(generated_features.index_select(0, selected), dim=1)
        real_f = torch.nn.functional.normalize(real_features.index_select(0, selected), dim=1)
        feature_cost = torch.cdist(generated_f, real_f).square()
        generated_pixels = _deterministic_sinkhorn_cost_view(generated_view.index_select(0, selected)).flatten(1)
        real_pixels = _deterministic_sinkhorn_cost_view(real_view.index_select(0, selected)).flatten(1)
        pixel_cost = torch.cdist(generated_pixels, real_pixels, p=1) / generated_pixels.shape[1]
        terms.append(log_sinkhorn(feature_cost + 0.25 * pixel_cost, epsilon=epsilon, iterations=iterations))
    return torch.stack(terms).mean()


def _research_conditional_loss(*, system: Any, generator: Any, evaluator: Any, codes: torch.Tensor, labels: torch.Tensor, real_views: torch.Tensor, exposure_scale: float, sinkhorn_epsilon: float, sinkhorn_iterations: int, microbatch_size: int) -> torch.Tensor:
    raw = raw_from_generator_batched(system, generator, codes, labels, microbatch_size=microbatch_size, activation_checkpoint=False)
    generated_view = _deterministic_fixed_exposure_view(raw, exposure_scale)
    generated_features, logits = evaluator(generated_view)
    with torch.no_grad():
        real_features, _ = evaluator(real_views)
    sinkhorn = _research_sinkhorn_loss(generated_features, generated_view, real_features, real_views, labels, epsilon=sinkhorn_epsilon, iterations=sinkhorn_iterations)
    total = sinkhorn + 0.5 * torch.nn.functional.cross_entropy(logits, labels)
    total = total + 0.1 * diversity_hinge(generated_features, codes, labels)
    total = total + 0.1 * torch.relu(raw / exposure_scale - 1.0).square().mean()
    total = total + 1.0e-3 * generator.mode_orthogonality_penalty()
    total = total + 1.0e-5 * phase_total_variation(system)
    return total


def _update(context: tuple[Any, ...], *, matrix_id: str, seed: int, step: int, optimizer: torch.optim.Optimizer) -> float:
    config, device, images, labels, _, _, _, _, bundle, _, _, _ = context
    settings = config.conditional_generation
    if settings is None or not isinstance(bundle.system.ols, FiberTransportEnsembleOLS):
        raise ValueError("fiber image update requires a fiber system")
    bundle.system.ols.select(matrix_id)
    indices = next(iter(balanced_batches(labels.detach().cpu(), batch_size=settings.logical_batch_size, seed=seed + step)))
    batch_labels = labels.index_select(0, indices.to(device))
    batch_images = images.index_select(0, indices.to(device))
    codes = random_codes(batch_labels.numel(), device=device, seed=(seed + 1) * 100000 + step)
    optimizer.zero_grad(set_to_none=True)
    loss = _research_conditional_loss(system=bundle.system, generator=bundle.generator, evaluator=context[6], codes=codes, labels=batch_labels, real_views=batch_images, exposure_scale=settings.fixed_exposure_scale, sinkhorn_epsilon=settings.sinkhorn_epsilon, sinkhorn_iterations=settings.sinkhorn_iterations, microbatch_size=config.execution.training_microbatch_size)
    if not torch.isfinite(loss):
        raise ValueError("fiber image update loss is non-finite")
    loss.backward()
    gradients = [parameter.grad for parameter in bundle.generator.parameters()]
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError("fiber image generator gradient is missing/non-finite")
    optimizer.step()
    return float(loss.detach().cpu())


@dataclass(frozen=True)
class FiberImageResearchSpec:
    schema: str = "fiber_image_spec.v1"
    baseline_seed: int = 2027
    baseline_samples: int = 100
    locked_samples: int = 500
    steps: int = 50
    bootstrap_seed: int = 90210
    bootstrap_replicates: int = 10000


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("fiber image evidence must be an object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    atomic_write_json(path, value)


def _tensor_content_hash(value: torch.Tensor) -> str:
    """Hash tensor content, independent of the PyTorch container format."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _spec_hash(pool: FiberTransportPool) -> str:
    return hashlib.sha256(
        json.dumps(
            {"spec": asdict(FiberImageResearchSpec()), "pool": pool.evidence()},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def _dataset_manifest(dataset_root: Path) -> dict[str, Any]:
    images, labels = load_local_mnist(str(dataset_root), train=True)
    _, _, _, _, split = split_mnist(images, labels, str(dataset_root))
    return {"manifest_hash": split.manifest_hash, "manifest": asdict(split)}


def _assert_validation_current(args: Any, *, validation: dict[str, Any], baseline: dict[str, Any] | None = None) -> None:
    if validation.get("schema") != "fiber_image_validation.v2" or validation.get("gate") != "valid":
        raise ValueError("validation evidence is not a valid v2 artifact")
    if validation.get("git_commit") != _git() or _dirty():
        raise ValueError("validation evidence is stale or worktree is dirty")
    config = load_config(str(_required(args, "config")))
    if validation.get("config_hash") != robust._config_hash(config):
        raise ValueError("validation config hash mismatch")
    dataset = _dataset_manifest(_required(args, "dataset_root", directory=True))
    if validation.get("dataset_split_manifest_hash") != dataset["manifest_hash"]:
        raise ValueError("validation dataset split mismatch")
    primary = _required(args, "primary_evaluator_checkpoint")
    secondary = _required(args, "secondary_evaluator_checkpoint")
    if validation.get("primary_evaluator_sha256") != file_sha256(primary) or validation.get("secondary_evaluator_sha256") != file_sha256(secondary):
        raise ValueError("validation evaluator binding mismatch")
    pool = FiberTransportPool()
    if validation.get("pool_hash") != _spec_hash(pool):
        raise ValueError("validation pool/spec hash mismatch")
    if baseline is not None:
        if baseline.get("schema") != "fiber_image_baseline.v2" or baseline.get("gate") != "valid":
            raise ValueError("baseline evidence is not a valid v2 artifact")
        if baseline.get("validation_sha256") != file_sha256(_required(args, "validate_evidence")):
            raise ValueError("baseline validation binding mismatch")
        if baseline.get("git_commit") != _git():
            raise ValueError("baseline Git binding mismatch")


def _git() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _required(args: Any, name: str, *, directory: bool = False) -> Path:
    value = getattr(args, name, None)
    if not value:
        raise ValueError(f"fiber image requires --{name.replace('_', '-')}")
    path = Path(value)
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"fiber image input does not exist: {path}")
    return path


def _ref(path: Path, root: Path) -> dict[str, str]:
    relative = path.resolve().relative_to(root.resolve())
    if ".." in relative.parts:
        raise ValueError("artifact escaped run directory")
    return {"relative_path": relative.as_posix(), "sha256": file_sha256(path)}


def _resolve(reference: dict[str, Any], root: Path) -> Path:
    relative, expected = reference.get("relative_path"), reference.get("sha256")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected, str)
    ):
        raise ValueError("invalid artifact reference")
    path = (root / relative).resolve()
    if (
        root.resolve() not in path.parents
        or not path.is_file()
        or file_sha256(path) != expected
    ):
        raise ValueError("missing or hash-mismatched artifact")
    return path


def _tensor_ref(path: Path, value: torch.Tensor, root: Path) -> dict[str, str]:
    reference = _ref(path, root)
    reference["tensor_sha256"] = _tensor_content_hash(value)
    return reference


def _mapping(pool: FiberTransportPool) -> dict[str, str]:
    by_seed = {record.seed: record.matrix_id for record in pool.records["train"]}
    return {
        "I0": "identity",
        "S0": by_seed[0],
        "S1a": by_seed[6101],
        "S1b": by_seed[6112],
        "S1c": by_seed[6124],
        "N0": pool.records["full_phase"][0].matrix_id,
    }


def _lineage(args: Any, *, initialization: bool) -> dict[str, Any]:
    config_path, dataset = (
        _required(args, "config"),
        _required(args, "dataset_root", directory=True),
    )
    primary, secondary = (
        _required(args, "primary_evaluator_checkpoint"),
        _required(args, "secondary_evaluator_checkpoint"),
    )
    primary_q = robust._validate_historical_evidence(
        _required(args, "primary_qualification_evidence")
    )
    secondary_q = robust._validate_historical_evidence(
        _required(args, "secondary_qualification_evidence")
    )
    config = load_config(str(config_path))
    result = {
        "config": config,
        "config_hash": robust._config_hash(config),
        "dataset_root": dataset,
        "primary": primary,
        "secondary": secondary,
        "primary_hash": file_sha256(primary),
        "secondary_hash": file_sha256(secondary),
        "qualification": {"primary": primary_q, "secondary": secondary_q},
        "git_commit": _git(),
    }
    if initialization:
        checkpoint, source = (
            _required(args, "research_initialization_checkpoint"),
            _required(args, "source_training_evidence"),
        )
        source_payload = robust._validate_historical_evidence(
            source, expected_kind="conditional_training"
        )
        if source_payload.get("status") != "passed" or source_payload.get("dirty_patch_hash") not in {None, ""}:
            raise ValueError("source training evidence is not a clean passed artifact")
        recorded = (
            source_payload.get("input_hashes", {}).get("checkpoint")
            or source_payload.get("context", {}).get("checkpoint_hash")
            or source_payload.get("checkpoint_hash")
        )
        if recorded != file_sha256(checkpoint):
            raise ValueError(
                "initialization checkpoint is not bound to source evidence"
            )
        bundle, evidence = (
            _required(args, "measurement_bundle"),
            _required(args, "measurement_evidence"),
        )
        bundle_payload, evidence_payload = _read(bundle), _read(evidence)
        if (
            bundle_payload.get("schema") != "fiber_image_measurement_bundle.v2"
            or evidence_payload.get("schema") != "fiber_image_measurement_evidence.v2"
            or evidence_payload.get("bundle_sha256") != file_sha256(bundle)
            or evidence_payload.get("config_hash") != result["config_hash"]
            or bundle_payload.get("config_hash") != result["config_hash"]
            or bundle_payload.get("primary_evaluator_sha256") != result["primary_hash"]
            or bundle_payload.get("secondary_evaluator_sha256") != result["secondary_hash"]
            or evidence_payload.get("git_commit") != _git()
            or evidence_payload.get("dataset_split_manifest_hash") != bundle_payload.get("dataset_split_manifest_hash")
            or bundle_payload.get("pool_hash") != bundle_payload.get("spec_hash")
        ):
            raise ValueError("measurement lineage mismatch")
        result.update(
            {
                "checkpoint": checkpoint,
                "source_evidence": source,
                "measurement_bundle": bundle,
                "measurement_evidence": evidence,
            }
        )
    return result


def _context(args: Any) -> tuple[Any, ...]:
    lineage = _lineage(args, initialization=True)
    config, device = lineage["config"], torch.device(args.device)
    images, labels = load_local_mnist(str(lineage["dataset_root"]), train=True)
    train_images, train_labels, validation_images, validation_labels, split = (
        split_mnist(images, labels, str(lineage["dataset_root"]))
    )
    primary, _ = load_frozen_evaluator(str(lineage["primary"]), device=device)
    secondary_payload = safe_torch_load(
        lineage["secondary"],
        expected_sha256=lineage["secondary_hash"],
        map_location=device,
    )
    if (
        not isinstance(secondary_payload, dict)
        or secondary_payload.get("architecture_kind")
        != SmoothMNISTEvaluator.architecture_kind
    ):
        raise ValueError("secondary evaluator architecture mismatch")
    secondary = SmoothMNISTEvaluator().to(device)
    secondary.load_state_dict(secondary_payload["state_dict"], strict=True)
    secondary.freeze()
    payload = safe_torch_load(
        lineage["checkpoint"],
        expected_sha256=file_sha256(lineage["checkpoint"]),
        map_location="cpu",
    )
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("manifest"), dict)
        or payload["manifest"].get("system_signature") != _system_signature(config)
    ):
        raise ValueError("initialization checkpoint contract mismatch")
    generator_state = {
        key.removeprefix("generator."): value
        for key, value in payload["state_dict"].items()
        if key.startswith("generator.")
    }
    bundle = _seeded_conditional_bundle(config)
    bundle.generator.load_state_dict(generator_state, strict=True)
    bundle = bundle.to(device)
    pool = FiberTransportPool()
    bundle.system.ols = FiberTransportEnsembleOLS(bundle.system.encoder_plane, pool).to(
        device
    )
    for parameter in bundle.system.parameters():
        parameter.requires_grad_(False)
    for parameter in bundle.generator.parameters():
        parameter.requires_grad_(True)
    return (
        config,
        device,
        train_images.to(device),
        train_labels.to(device),
        validation_images.to(device),
        validation_labels.to(device),
        primary,
        secondary,
        bundle,
        pool,
        split,
        lineage,
    )


def _select(bundle: Any, pool: FiberTransportPool, function_id: str) -> str:
    if function_id == "I0":
        bundle.system.ols = robust._IdentityOLSModule()
        return "identity"
    if not isinstance(bundle.system.ols, FiberTransportEnsembleOLS):
        raise ValueError("fiber operator was not injected")
    matrix_id = _mapping(pool)[function_id]
    bundle.system.ols.select(matrix_id)
    return matrix_id


def _panel(
    context: tuple[Any, ...],
    function_id: str,
    seed: int,
    count: int,
    *,
    trace: bool = False,
    zero_code: bool = False,
    shuffled_labels: bool = False,
) -> dict[str, Any]:
    config, device, _, _, _, _, primary, secondary, bundle, pool, _, _ = context
    settings = config.conditional_generation
    if settings is None:
        raise ValueError("conditional generation settings are required")
    original_ols = bundle.system.ols
    _select(bundle, pool, function_id)
    codes = (
        torch.zeros(count, 8, device=device)
        if zero_code
        else random_codes(count, device=device, seed=seed)
    )
    labels = torch.arange(10, device=device).repeat_interleave(count // 10)
    input_labels = labels.roll(1) if shuffled_labels else labels
    (
        rows,
        views,
        primary_logits,
        secondary_logits,
        secondary_features,
        traces,
        port_inputs,
        port_outputs,
        input_energies,
        output_energies,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    try:
        for start in range(0, count, config.execution.evaluation_microbatch_size):
            c = codes[start : start + config.execution.evaluation_microbatch_size]
            with torch.inference_mode():
                command = bundle.generator(c, input_labels[start : start + c.shape[0]])
                result = bundle.system.forward(
                    bundle.system.command(command), return_optical_trace=trace
                )
                raw = result.raw_camera.tensor
                view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
                _, p = primary(view)
                sf, s = secondary(view)
            rows.append(raw.detach())
            views.append(view.detach())
            primary_logits.append(p.detach())
            secondary_logits.append(s.detach())
            secondary_features.append(sf.detach())
            if trace:
                fields = []
                for field in result.optical_trace or ():
                    if hasattr(field, "tensor"):
                        fields.append(
                            {
                                "tensor": field.tensor.detach().cpu(),
                                "plane": asdict(field.plane),
                                "provenance": asdict(field.provenance),
                            }
                        )
                traces.append({"command": command.detach().cpu(), "fields": fields})
            if function_id != "I0":
                port_inputs.append(
                    bundle.system.ols.last_diagnostics["port_input_coefficients"]
                    .detach()
                    .cpu()
                )
                port_outputs.append(
                    bundle.system.ols.last_diagnostics["port_output_coefficients"]
                    .detach()
                    .cpu()
                )
                input_energy = bundle.system.ols.last_diagnostics["input_energy"].detach().cpu()
                output_energy = bundle.system.ols.last_diagnostics["output_energy"].detach().cpu()
                input_energies.append(input_energy)
                output_energies.append(output_energy)
    finally:
        bundle.system.ols = original_ols
    raw, view = torch.cat(rows), torch.cat(views)
    p, s = torch.cat(primary_logits).argmax(1), torch.cat(secondary_logits).argmax(1)
    generated_features = torch.cat(secondary_features)
    reference_images, reference_labels = context[4], context[5]
    diversity_indices = torch.cat(
        [
            torch.where(reference_labels == label)[0][: count // 10]
            for label in range(10)
        ]
    )
    reference_count = max(count // 10, 4)
    reference_indices = torch.cat(
        [
            torch.where(reference_labels == label)[0][:reference_count]
            for label in range(10)
        ]
    )
    with torch.inference_mode():
        diversity_features, _ = secondary(
            reference_images.index_select(0, diversity_indices)
        )
        reference_features, _ = secondary(
            reference_images.index_select(0, reference_indices)
        )
    coverage_available = count >= 40
    metric = {
        "primary_accuracy": float((p == labels).float().mean()),
        "secondary_accuracy": float((s == labels).float().mean()),
        "primary_min_recall": min(
            float((p[labels == label] == label).float().mean()) for label in range(10)
        ),
        "secondary_min_recall": min(
            float((s[labels == label] == label).float().mean()) for label in range(10)
        ),
        "agreement": float((p == s).float().mean()),
        "finite": bool(torch.isfinite(raw).all()),
        "nonnegative": bool((raw >= 0).all()),
        "saturation": float((raw > settings.fixed_exposure_scale).float().mean()),
        "camera_fraction": float(camera_code_fraction(raw, labels).item()),
        "minimum_diversity": min(
            relative_diversity_ratios(generated_features, diversity_features, labels)
        ),
        "coverage_ratio_available": coverage_available,
        "coverage_ratio": conditional_coverage_ratio(
            generated_features,
            view,
            reference_features,
            reference_images.index_select(0, reference_indices),
            labels,
            epsilon=settings.sinkhorn_epsilon,
            iterations=settings.sinkhorn_iterations,
            seed=2027,
        ) if coverage_available else None,
        "labels": labels.cpu(),
        "codes": codes.cpu(),
        "raw_cmos": raw.cpu(),
        "fixed_cmos": view.cpu(),
        "primary_prediction": p.cpu(),
        "secondary_prediction": s.cpu(),
    }
    if trace:
        metric["trace_rows"] = traces
    if function_id == "I0":
        metric["port_available"] = False
        metric["passive_available"] = False
    else:
        metric["port_available"] = True
        input_energy = torch.cat(input_energies)
        output_energy = torch.cat(output_energies)
        metric["passive_available"] = True
        metric["passive"] = bool(torch.all(output_energy <= input_energy * (1.0 + 1.0e-5) + 1.0e-12))
        metric["input_energy_max"] = float(input_energy.max())
        metric["output_energy_max"] = float(output_energy.max())
        metric["port_input_coefficients"] = (
            torch.cat(port_inputs) if port_inputs else None
        )
        metric["port_output_coefficients"] = (
            torch.cat(port_outputs) if port_outputs else None
        )
    if trace:
        metric["trace_fields"] = traces
    return metric


def _sheet(
    path: Path,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    diff: torch.Tensor | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        index
        for label in range(10)
        for index in torch.where(labels == label)[0][:10].tolist()
    ]
    values = diff if diff is not None else images
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    vmax = float(values.abs().max().clamp_min(1e-12))
    fig, axes = plt.subplots(10, 10, figsize=(10, 10), squeeze=False)
    for cell, index in enumerate(selected):
        axes[cell // 10][cell % 10].imshow(
            values[index],
            cmap="coolwarm" if diff is not None else "gray",
            vmin=-vmax if diff is not None else 0,
            vmax=vmax,
        )
        axes[cell // 10][cell % 10].axis("off")
    fig.tight_layout(pad=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fig.savefig(temporary, dpi=100, format="png")
    plt.close(fig)
    temporary.replace(path)


def _bundle(
    root: Path,
    function_id: str,
    result: dict[str, Any],
    identity: torch.Tensor | None = None,
) -> dict[str, Any]:
    directory = root / function_id
    directory.mkdir(parents=True, exist_ok=True)
    refs = {}
    for name in (
        "labels",
        "codes",
        "raw_cmos",
        "fixed_cmos",
        "primary_prediction",
        "secondary_prediction",
    ):
        path = directory / f"{name}.pt"
        atomic_torch_save(path, result[name])
        refs[name] = _tensor_ref(path, result[name], root)
    _sheet(directory / "contact_sheet.png", result["fixed_cmos"], result["labels"])
    refs["contact_sheet"] = _ref(directory / "contact_sheet.png", root)
    if "trace_rows" in result:
        path = directory / "optical_trace.pt"
        atomic_torch_save(path, result["trace_rows"])
        refs["optical_trace"] = _ref(path, root)
        command_path = directory / "slm_amplitude.pt"
        atomic_torch_save(
            command_path,
            torch.cat([row["command"] for row in result["trace_rows"]], dim=0),
        )
        refs["slm_amplitude"] = _tensor_ref(
            command_path,
            torch.cat([row["command"] for row in result["trace_rows"]], dim=0),
            root,
        )
        # Trace fields are keyed by their declared plane/provenance, never by
        # tuple position.  The field immediately upstream of ``ols_output``
        # is the encoder-side fiber input plane in the formal trace contract.
        fields = result["trace_rows"][0]["fields"] if result["trace_rows"] else []
        output_index = next(
            (index for index, field in enumerate(fields) if field["plane"].get("plane_id") == "ols_output"),
            None,
        )
        if output_index is not None and output_index > 0:
            input_field = fields[output_index - 1]
            output_field = fields[output_index]
            for prefix in ("fiber_input", "fiber_output"):
                amplitude_path = directory / f"{prefix}_amplitude.pt"
                phase_path = directory / f"{prefix}_phase.pt"
                atomic_torch_save(amplitude_path, torch.stack([row["fields"][output_index - 1 if prefix == "fiber_input" else output_index]["tensor"] for row in result["trace_rows"]]).abs())
                atomic_torch_save(phase_path, torch.stack([row["fields"][output_index - 1 if prefix == "fiber_input" else output_index]["tensor"] for row in result["trace_rows"]]).angle())
                refs[f"{prefix}_amplitude"] = _tensor_ref(
                    amplitude_path,
                    torch.stack([row["fields"][output_index - 1 if prefix == "fiber_input" else output_index]["tensor"] for row in result["trace_rows"]]).abs(),
                    root,
                )
                refs[f"{prefix}_phase"] = _tensor_ref(
                    phase_path,
                    torch.stack([row["fields"][output_index - 1 if prefix == "fiber_input" else output_index]["tensor"] for row in result["trace_rows"]]).angle(),
                    root,
                )
            provenance_path = directory / "fiber_field_provenance.json"
            _write(provenance_path, {"input": input_field["provenance"], "output": output_field["provenance"], "input_plane": input_field["plane"], "output_plane": output_field["plane"]})
            refs["fiber_field_provenance"] = _ref(provenance_path, root)
    if result.get("port_available"):
        for name in ("port_input_coefficients", "port_output_coefficients"):
            path = directory / f"{name}.pt"
            atomic_torch_save(path, result[name])
            refs[name] = _tensor_ref(path, result[name], root)
    if identity is not None:
        difference = result["fixed_cmos"] - identity
        _sheet(
            directory / "difference_identity.png",
            difference,
            result["labels"],
            diff=difference,
        )
        refs["difference_identity"] = _ref(directory / "difference_identity.png", root)
    return refs


def validate(args) -> int:
    if _dirty():
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_validation.v2",
                "gate": "invalid_harness",
                "reason": "dirty worktree",
            },
        )
        return 2
    fixture, config_path = _required(args, "lp01_fixture"), _required(args, "config")
    config = load_config(str(config_path))
    pool = FiberTransportPool()
    dataset_root = _required(args, "dataset_root", directory=True)
    primary_path = _required(args, "primary_evaluator_checkpoint")
    secondary_path = _required(args, "secondary_evaluator_checkpoint")
    images, labels = load_local_mnist(str(dataset_root), train=True)
    _, _, validation_images, validation_labels, split = split_mnist(images, labels, str(dataset_root))
    primary, _ = load_frozen_evaluator(str(primary_path), device=torch.device("cpu"))
    secondary_payload = safe_torch_load(
        secondary_path,
        expected_sha256=file_sha256(secondary_path),
        map_location="cpu",
    )
    if not isinstance(secondary_payload, dict) or secondary_payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind:
        raise ValueError("secondary evaluator architecture mismatch")
    secondary_model = SmoothMNISTEvaluator()
    secondary_model.load_state_dict(secondary_payload["state_dict"], strict=True)
    secondary_model.freeze()
    primary_predictions, secondary_predictions = [], []
    with torch.inference_mode():
        for start in range(0, validation_images.shape[0], 256):
            batch = validation_images[start : start + 256]
            _, primary_logits = primary(batch)
            _, secondary_logits = secondary_model(batch)
            primary_predictions.append(primary_logits.argmax(1))
            secondary_predictions.append(secondary_logits.argmax(1))
    primary_prediction = torch.cat(primary_predictions)
    secondary_prediction = torch.cat(secondary_predictions)
    qualification = {
        "sample_count": int(validation_labels.numel()),
        "primary_accuracy": float((primary_prediction == validation_labels).float().mean()),
        "secondary_accuracy": float((secondary_prediction == validation_labels).float().mean()),
        "agreement": float((primary_prediction == secondary_prediction).float().mean()),
        "primary_min_recall": min(float((primary_prediction[validation_labels == label] == label).float().mean()) for label in range(10)),
        "secondary_min_recall": min(float((secondary_prediction[validation_labels == label] == label).float().mean()) for label in range(10)),
        "finite": bool(torch.isfinite(primary_prediction.float()).all() and torch.isfinite(secondary_prediction.float()).all()),
    }
    if not qualification["finite"] or qualification["primary_accuracy"] < 0.90 or qualification["secondary_accuracy"] < 0.90 or qualification["agreement"] < 0.90:
        raise ValueError("current evaluator qualification failed")
    fixture_payload = _read(fixture)
    rho = torch.tensor(fixture_payload["rho"], dtype=torch.float64)
    fixture_field = torch.tensor(fixture_payload["normalized_amplitude"], dtype=torch.float64)
    v_number = float(fixture_payload["v_number"])
    reference = lp01_reference_metrics(v_number)
    finite_difference = lp01_finite_difference_eigenpair(v_number)
    actual_field = lp01_radial_profile(rho, v_number).to(torch.float64)
    overlap = float(torch.dot(actual_field, fixture_field) / (actual_field.norm() * fixture_field.norm()))
    expected_energy = torch.trapezoid(fixture_field.square() * rho, rho)
    actual_energy = torch.trapezoid(actual_field.square() * rho, rho)
    energy_error = float((actual_energy - expected_energy).abs() / expected_energy.clamp_min(1.0e-30))
    radius_error = abs(reference["amplitude_1e_radius_rho"] - float(fixture_payload["amplitude_1e_radius_rho"])) / float(fixture_payload["amplitude_1e_radius_rho"])
    lp01_gate = (
        reference["closure_error"] <= 1.0e-12
        and reference["characteristic_residual"] <= 1.0e-10
        and reference["boundary_derivative_relative_error"] <= 1.0e-6
        and overlap >= 0.995
        and radius_error <= 0.01
        and energy_error <= 1.0e-4
        and abs(finite_difference[0] - reference["u"]) / reference["u"] <= 0.005
        and abs(finite_difference[1] - reference["w"]) / reference["w"] <= 0.005
    )
    if not lp01_gate:
        raise ValueError("LP01 reverse-validation gate failed")
    probe = torch.linspace(0.0, 1.0, 2 * 200 * 200, dtype=torch.float64).reshape(2, 1, 200, 200)
    native_28 = torch.nn.functional.adaptive_avg_pool2d(probe, (28, 28))
    research_28 = _deterministic_adaptive_average(probe, 28)
    native_14 = torch.nn.functional.adaptive_avg_pool2d(native_28, (14, 14))
    research_14 = _deterministic_adaptive_average(research_28, 14)
    pooling_forward_error = max(float((native_28 - research_28).abs().max()), float((native_14 - research_14).abs().max()))
    pooling_gate = pooling_forward_error <= 1.0e-6
    if not pooling_gate:
        raise ValueError("deterministic research pooling does not match native forward")
    spec = asdict(FiberImageResearchSpec())
    payload = {
        "schema": "fiber_image_validation.v2",
        "gate": "valid",
        "git_commit": _git(),
        "config_hash": robust._config_hash(config),
        "fixture_sha256": file_sha256(fixture),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_split_manifest_hash": split.manifest_hash,
        "primary_evaluator_sha256": file_sha256(primary_path),
        "secondary_evaluator_sha256": file_sha256(secondary_path),
        "qualification": qualification,
        "spec": spec,
        "functions": _mapping(pool),
        "pool_hash": _spec_hash(pool),
        "deterministic_pooling": {
            "forward_max_abs_error": pooling_forward_error,
            "gate": "valid" if pooling_gate else "invalid_harness",
            "paths": ["200->28", "28->14"],
        },
        "lp01_fixture": {
            **reference,
            "phase_aligned_overlap": overlap,
            "amplitude_1e_radius_relative_error": radius_error,
            "normalized_energy_error": energy_error,
            "finite_difference_u": finite_difference[0],
            "finite_difference_w": finite_difference[1],
            "gate": "valid" if lp01_gate else "invalid_harness",
        },
    }
    _write(Path(args.output), payload)
    _write(
        Path(args.measurement_bundle_output),
        {
            "schema": "fiber_image_measurement_bundle.v2",
            "gate": "valid",
            "validation_sha256": file_sha256(args.output),
            "config_hash": payload["config_hash"],
            "fixture_sha256": payload["fixture_sha256"],
            "dataset_split_manifest_hash": split.manifest_hash,
            "primary_evaluator_sha256": payload["primary_evaluator_sha256"],
            "secondary_evaluator_sha256": payload["secondary_evaluator_sha256"],
            "pool_hash": payload["pool_hash"],
            "spec_hash": payload["pool_hash"],
        },
    )
    _write(
        Path(args.measurement_evidence_output),
        {
            "schema": "fiber_image_measurement_evidence.v2",
            "gate": "valid",
            "bundle_sha256": file_sha256(args.measurement_bundle_output),
            "config_hash": payload["config_hash"],
            "git_commit": _git(),
            "dataset_split_manifest_hash": split.manifest_hash,
            "primary_evaluator_sha256": payload["primary_evaluator_sha256"],
            "secondary_evaluator_sha256": payload["secondary_evaluator_sha256"],
            "pool_hash": payload["pool_hash"],
            "consumer": "fiber-image baseline and preflight",
        },
    )
    return 0


def baseline(args) -> int:
    validation = _read(_required(args, "validate_evidence"))
    _assert_validation_current(args, validation=validation)
    context = _context(args)
    output_path = Path(args.output)
    root = output_path.parent if output_path.suffix == ".json" else output_path
    root.mkdir(parents=True, exist_ok=True)
    identity = _panel(context, "I0", 2027, 100, trace=True)
    outputs = {}
    for function_id in FUNCTION_IDS:
        result = (
            identity
            if function_id == "I0"
            else _panel(context, function_id, 2027, 100, trace=True)
        )
        outputs[function_id] = {
            "matrix_id": _mapping(context[9])[function_id],
            "metrics": {
                key: value
                for key, value in result.items()
                if isinstance(value, (bool, int, float, str))
            },
            "artifacts": _bundle(
                root,
                function_id,
                result,
                None if function_id == "I0" else identity["fixed_cmos"],
            ),
        }
    pair_hashes = {"labels": {}, "codes": {}}
    for field in pair_hashes:
        for function_id in FUNCTION_IDS:
            path = _resolve(outputs[function_id]["artifacts"][field], root)
            value = safe_torch_load(path, map_location="cpu")
            pair_hashes[field][function_id] = _tensor_content_hash(value)
    paired = {
        field: len(set(values.values())) == 1
        for field, values in pair_hashes.items()
    }
    nominal_result = outputs["S0"]
    full_phase_result = outputs["N0"]
    # Use the retained tensors for the observability gate; metrics intentionally
    # remain scalar-only in the manifest.
    nominal_raw = safe_torch_load(_resolve(nominal_result["artifacts"]["raw_cmos"], root), map_location="cpu")
    full_phase_raw = safe_torch_load(_resolve(full_phase_result["artifacts"]["raw_cmos"], root), map_location="cpu")
    cmos_observability = float((full_phase_raw - nominal_raw).norm() / nominal_raw.norm().clamp_min(1.0e-12))
    nominal_ports = safe_torch_load(_resolve(nominal_result["artifacts"]["port_output_coefficients"], root), map_location="cpu")
    full_phase_ports = safe_torch_load(_resolve(full_phase_result["artifacts"]["port_output_coefficients"], root), map_location="cpu")
    port_observability = float((full_phase_ports - nominal_ports).norm() / nominal_ports.norm().clamp_min(1.0e-12))
    if not all(paired.values()) or cmos_observability < 0.1 or port_observability < 0.1:
        raise ValueError("paired baseline or full-phase observability gate failed")
    _write(
        output_path if output_path.suffix == ".json" else root / "baseline.json",
        {
            "schema": "fiber_image_baseline.v2",
            "gate": "valid",
            "paired_seed": 2027,
            "sample_count": 100,
            "outputs": outputs,
            "validation_sha256": file_sha256(args.validate_evidence),
            "git_commit": _git(),
            "pairing": {"hashes": pair_hashes, "gates": paired},
            "observability": {"cmos_normalized_l2": cmos_observability, "port_normalized_l2": port_observability},
        },
    )
    return 0


def _resume_replay(args: Any, matrix_id: str) -> dict[str, Any]:
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    cudnn_before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    tf32_before = torch.backends.cuda.matmul.allow_tf32
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    def run(
        context: tuple[Any, ...], optimizer: torch.optim.Optimizer, steps: range
    ) -> None:
        for step in steps:
            _update(
                context, matrix_id=matrix_id, seed=5201, step=step, optimizer=optimizer
            )

    try:
        continuous = _context(args)
        opt_a = torch.optim.Adam(
            continuous[8].generator.parameters(),
            lr=continuous[0].conditional_generation.generator_learning_rate,
        )
        run(continuous, opt_a, range(3))
        partial = _context(args)
        opt_b = torch.optim.Adam(
            partial[8].generator.parameters(),
            lr=partial[0].conditional_generation.generator_learning_rate,
        )
        run(partial, opt_b, range(1))
        saved_generator = {
            key: value.detach().clone()
            for key, value in partial[8].generator.state_dict().items()
        }
        saved_optimizer = opt_b.state_dict()
        resumed = _context(args)
        resumed[8].generator.load_state_dict(saved_generator, strict=True)
        opt_c = torch.optim.Adam(
            resumed[8].generator.parameters(),
            lr=resumed[0].conditional_generation.generator_learning_rate,
        )
        opt_c.load_state_dict(saved_optimizer)
        run(resumed, opt_c, range(1, 3))
        generator_error = max(
            float(
                (
                    continuous[8].generator.state_dict()[key]
                    - resumed[8].generator.state_dict()[key]
                )
                .abs()
                .max()
            )
            for key in continuous[8].generator.state_dict()
        )
        optimizer_equal = robust._nested_equal(opt_a.state_dict(), opt_c.state_dict())
        return {
            "passed": generator_error <= 1.0e-5 and optimizer_equal,
            "generator_max_abs_error": generator_error,
            "optimizer_equal": optimizer_equal,
            "rtol": 1.0e-5,
            "atol": 1.0e-5,
        }
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = (
            cudnn_before
        )
        torch.backends.cuda.matmul.allow_tf32 = tf32_before


def preflight(args) -> int:
    if _dirty():
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_preflight.v2",
                "gate": "invalid_harness",
                "reason": "dirty worktree",
            },
        )
        return 2
    validation_path = _required(args, "validate_evidence")
    validation = _read(validation_path)
    baseline_path = _required(args, "baseline_evidence")
    baseline = _read(baseline_path)
    _assert_validation_current(args, validation=validation, baseline=baseline)
    context = _context(args)
    config, bundle, pool = context[0], context[8], context[9]
    matrix_id = _mapping(pool)[args.function_id]
    _select(bundle, pool, args.function_id)
    optimizer = torch.optim.Adam(
        bundle.generator.parameters(),
        lr=config.conditional_generation.generator_learning_rate,
    )
    before = {
        key: value.detach().clone()
        for key, value in bundle.generator.state_dict().items()
    }
    frozen_before = {
        key: value.detach().clone()
        for key, value in bundle.system.state_dict().items()
        if not key.startswith("ols.")
    }
    losses = [
        _update(
            context, matrix_id=matrix_id, seed=5201, step=step, optimizer=optimizer
        )
        for step in range(3)
    ]
    normal = _panel(context, args.function_id, 2027, 20, trace=True)
    zero = _panel(context, args.function_id, 2027, 20, zero_code=True)
    labelled = _panel(context, args.function_id, 2028, 20)
    shuffled = _panel(context, args.function_id, 2028, 20, shuffled_labels=True)
    nominal = _panel(context, "S0", 2029, 20, trace=True)
    full_phase = _panel(context, "N0", 2029, 20, trace=True)

    def relative(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left - right).norm() / right.norm().clamp_min(1.0e-12))

    resume = _resume_replay(args, matrix_id)
    checks = {
        "generator_update": any(
            not torch.equal(before[key], bundle.generator.state_dict()[key])
            for key in before
        ),
        "finite": bool(torch.isfinite(torch.tensor(losses)).all()),
        "optical_frozen": all(
            torch.equal(value, bundle.system.state_dict()[key])
            for key, value in frozen_before.items()
        ),
        "zero_code": relative(zero["raw_cmos"], normal["raw_cmos"]) >= 1.0e-4,
        "shuffled_label": relative(shuffled["raw_cmos"], labelled["raw_cmos"])
        >= 1.0e-4,
        "full_phase_port": relative(
            full_phase["port_output_coefficients"], nominal["port_output_coefficients"]
        )
        >= 0.1,
        "full_phase_cmos": relative(full_phase["raw_cmos"], nominal["raw_cmos"]) >= 0.1,
        "resume_replay": resume["passed"],
        "peak_memory_budget": not torch.cuda.is_available()
        or torch.cuda.max_memory_allocated(context[1])
        <= int(torch.cuda.get_device_properties(context[1]).total_memory * 0.95),
    }
    gate = all(checks.values()) and bool(torch.isfinite(torch.tensor(losses)).all())
    _write(
        Path(args.output),
        {
            "schema": "fiber_image_preflight.v2",
            "gate": "valid" if gate else "invalid_harness",
            "function_id": args.function_id,
            "updates": 3,
            "losses": losses,
            "checks": checks,
            "resume_replay": resume,
            "responses": {
                "zero_code": relative(zero["raw_cmos"], normal["raw_cmos"]),
                "shuffled_label": relative(shuffled["raw_cmos"], labelled["raw_cmos"]),
                "full_phase_port": relative(
                    full_phase["port_output_coefficients"],
                    nominal["port_output_coefficients"],
                ),
                "full_phase_cmos": relative(
                    full_phase["raw_cmos"], nominal["raw_cmos"]
                ),
            },
            "git_commit": _git(),
            "config_hash": robust._config_hash(config),
            "validation_sha256": file_sha256(validation_path),
            "baseline_sha256": file_sha256(baseline_path),
            "measurement_bundle_sha256": file_sha256(_required(args, "measurement_bundle")),
            "measurement_evidence_sha256": file_sha256(_required(args, "measurement_evidence")),
            "initialization_checkpoint_sha256": file_sha256(_required(args, "research_initialization_checkpoint")),
        },
    )
    return 0 if gate else 2


def screen(args) -> int:
    evidence = _read(_required(args, "preflight_evidence"))
    if (
        evidence.get("schema") != "fiber_image_preflight.v2"
        or evidence.get("gate") != "valid"
    ):
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_screen.v2",
                "gate": "invalid_harness",
                "reason": "preflight did not pass",
            },
        )
        return 2
    if evidence.get("function_id") != args.function_id or evidence.get("git_commit") != _git() or _dirty():
        raise ValueError("preflight function or clean worktree mismatch")
    output_path = Path(args.output)
    root = output_path.parent if output_path.suffix == ".json" else output_path
    root.mkdir(parents=True, exist_ok=True)
    context = _context(args)
    config, bundle, pool = context[0], context[8], context[9]
    matrix_id = _mapping(pool)[args.function_id]
    _select(bundle, pool, args.function_id)
    optimizer = torch.optim.Adam(
        bundle.generator.parameters(),
        lr=config.conditional_generation.generator_learning_rate,
    )
    start_step = 0
    schedule = [matrix_id] * 50
    if args.resume_checkpoint:
        checkpoint_path = _required(
            SimpleNamespace(resume_checkpoint=args.resume_checkpoint),
            "resume_checkpoint",
        )
        payload = safe_torch_load(
            checkpoint_path,
            expected_sha256=file_sha256(checkpoint_path),
            map_location=context[1],
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "fiber_image_checkpoint.v2"
            or payload.get("function_id") != args.function_id
            or payload.get("seed") != args.seed
            or payload.get("git_commit") != _git()
            or payload.get("preflight_sha256") != file_sha256(args.preflight_evidence)
            or payload.get("config_hash") != robust._config_hash(config)
            or payload.get("measurement_bundle_sha256") != file_sha256(_required(args, "measurement_bundle"))
            or payload.get("measurement_evidence_sha256") != file_sha256(_required(args, "measurement_evidence"))
            or payload.get("initialization_checkpoint_sha256") != file_sha256(_required(args, "research_initialization_checkpoint"))
        ):
            raise ValueError("resume checkpoint lineage mismatch")
        bundle.generator.load_state_dict(payload["generator_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload["step"])
        if int(payload.get("cursor", -1)) != start_step or not 0 <= start_step <= 50:
            raise ValueError("resume checkpoint cursor mismatch")
        if tuple(payload.get("schedule", ())) != tuple(schedule):
            raise ValueError("resume matrix schedule mismatch")
    checkpoints = []
    for step in range(start_step, 50):
        loss = _update(
            context, matrix_id=matrix_id, seed=args.seed, step=step, optimizer=optimizer
        )
        current_step = step + 1
        if current_step % 10 == 0:
            measurement = _panel(context, args.function_id, 2027, 170)
            validation = {
                "fixed_function": {
                    key: value
                    for key, value in measurement.items()
                    if isinstance(value, (bool, int, float, str))
                }
            }
            checkpoint = (
                root
                / args.function_id
                / f"seed_{args.seed}"
                / f"step_{current_step}.pt"
            )
            atomic_torch_save(
                checkpoint,
                {
                    "schema": "fiber_image_checkpoint.v2",
                    "function_id": args.function_id,
                    "seed": args.seed,
                    "step": current_step,
                    "schedule": schedule,
                    "cursor": current_step,
                    "generator_state": {
                        key: value.detach().cpu()
                        for key, value in bundle.generator.state_dict().items()
                    },
                    "optimizer_state": optimizer.state_dict(),
                    "validation": validation,
                    "config_hash": robust._config_hash(config),
                    "git_commit": _git(),
                    "preflight_sha256": file_sha256(args.preflight_evidence),
                    "measurement_bundle_sha256": file_sha256(_required(args, "measurement_bundle")),
                    "measurement_evidence_sha256": file_sha256(_required(args, "measurement_evidence")),
                    "initialization_checkpoint_sha256": file_sha256(_required(args, "research_initialization_checkpoint")),
                },
            )
            checkpoints.append(
                {
                    "step": current_step,
                    "artifact": _ref(checkpoint, root),
                    "validation": validation,
                    "loss": loss,
                }
            )
    gate = bool(checkpoints) and all(
        item["metrics"]["finite"]
        if "metrics" in item
        else all(value["finite"] for value in item["validation"].values())
        for item in checkpoints
    )
    payload = {
        "schema": "fiber_image_screen.v2",
        "gate": "valid" if gate else "invalid_harness",
        "function_id": args.function_id,
        "seed": args.seed,
        "steps": 50,
        "schedule": schedule,
        "checkpoints": checkpoints,
        "git_commit": _git(),
        "preflight_sha256": file_sha256(args.preflight_evidence),
        "config_hash": robust._config_hash(config),
        "measurement_bundle_sha256": file_sha256(_required(args, "measurement_bundle")),
        "measurement_evidence_sha256": file_sha256(_required(args, "measurement_evidence")),
        "initialization_checkpoint_sha256": file_sha256(_required(args, "research_initialization_checkpoint")),
    }
    _write(
        output_path if output_path.suffix == ".json" else root / "screen.json", payload
    )
    return 0 if gate else 2


def select(args) -> int:
    evidence = [_read(path) for path in args.screen_evidence]
    if len(evidence) != 12 or len(set(args.screen_evidence)) != 12 or any(
        item.get("schema") != "fiber_image_screen.v2" or item.get("gate") != "valid"
        for item in evidence
    ):
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_selection.v2",
                "gate": "invalid_harness",
                "reason": "twelve valid screen evidences are required",
            },
        )
        return 2
    keys = {(item.get("function_id"), item.get("seed")) for item in evidence}
    expected = {
        (function_id, seed)
        for function_id in ("S0", "S1a", "S1b", "S1c")
        for seed in TRAIN_SEEDS
    }
    if keys != expected or any(
        len(item.get("checkpoints", ())) != 5 for item in evidence
    ):
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_selection.v2",
                "gate": "invalid_harness",
                "reason": "screen evidence must cover four functions and three seeds",
            },
        )
        return 2
    if _dirty() or any(item.get("git_commit") != _git() for item in evidence):
        raise ValueError("screen evidence is stale or worktree is dirty")
    config_hashes = {item.get("config_hash") for item in evidence}
    if len(config_hashes) != 1 or None in config_hashes:
        raise ValueError("screen configuration lineage is inconsistent")
    output_path = Path(args.output)
    selection_root = output_path.parent if output_path.suffix == ".json" else output_path
    selection_root.mkdir(parents=True, exist_ok=True)
    context_template = _context(args)
    rows = []
    for screen, screen_path in zip(evidence, args.screen_evidence, strict=True):
        selected = max(
            screen["checkpoints"],
            key=lambda item: (
                sum(
                    (m["primary_accuracy"] + m["secondary_accuracy"]) / 2
                    for m in item["validation"].values()
                )
                / len(item["validation"]),
                sum(m["secondary_accuracy"] for m in item["validation"].values())
                / len(item["validation"]),
                -int(item["step"]),
            ),
        )
        checkpoint_path = _resolve(selected["artifact"], Path(screen_path).parent)
        checkpoint = safe_torch_load(
            checkpoint_path,
            expected_sha256=selected["artifact"]["sha256"],
            map_location=context_template[1],
        )
        candidate = _context(args)
        candidate[8].generator.load_state_dict(
            checkpoint["generator_state"], strict=True
        )
        seed_rows = []
        function_id, train_seed = screen["function_id"], screen["seed"]
        for panel_seed in PANEL_SEEDS:
            candidate_metrics = _panel(candidate, function_id, panel_seed, 500, trace=True)
            b0_context = _context(args)
            b0_metrics = _panel(b0_context, function_id, panel_seed, 500, trace=True)
            identity_metrics = _panel(candidate, "I0", panel_seed, 500, trace=True)
            candidate_correct = (
                candidate_metrics["secondary_prediction"] == candidate_metrics["labels"]
            ).to(torch.float32)
            b0_correct = (
                b0_metrics["secondary_prediction"] == b0_metrics["labels"]
            ).to(torch.float32)
            seed_rows.append(
                {
                    "panel_seed": panel_seed,
                    "candidate": {
                        key: value
                        for key, value in candidate_metrics.items()
                        if isinstance(value, (bool, int, float, str))
                    },
                    "b0": {
                        key: value
                        for key, value in b0_metrics.items()
                        if isinstance(value, (bool, int, float, str))
                    },
                    "identity": {
                        key: value
                        for key, value in identity_metrics.items()
                        if isinstance(value, (bool, int, float, str))
                    },
                    "gain_vector": (candidate_correct - b0_correct).tolist(),
                    "identity_gap": identity_metrics["secondary_accuracy"]
                    - candidate_metrics["secondary_accuracy"],
                }
            )
            artifact_base = selection_root / "locked_artifacts" / f"{function_id}_{train_seed}_{panel_seed}"
            artifact_refs = {
                "candidate_fiber": _bundle(artifact_base, "candidate_fiber", candidate_metrics, identity_metrics["fixed_cmos"]),
                "same_fiber_b0": _bundle(artifact_base, "same_fiber_b0", b0_metrics, identity_metrics["fixed_cmos"]),
                "candidate_identity": _bundle(artifact_base, "candidate_identity", identity_metrics),
            }
            panel_record = seed_rows[-1]
            panel_record["artifact_root"] = artifact_base.relative_to(selection_root).as_posix()
            panel_record["artifacts"] = artifact_refs
            correctness_path = artifact_base / "correctness.pt"
            correctness_path.parent.mkdir(parents=True, exist_ok=True)
            correctness = torch.stack([candidate_correct.cpu(), b0_correct.cpu()])
            atomic_torch_save(correctness_path, correctness)
            panel_record["correctness_artifact"] = _tensor_ref(correctness_path, correctness, artifact_base)
        rows.append(
            {
                "function_id": function_id,
                "seed": train_seed,
                "selected_step": selected["step"],
                "selected_checkpoint": selected["artifact"],
                "panels": seed_rows,
            }
        )
    bootstrap = _bootstrap_rows(rows)
    _write(
        output_path,
        {
            "schema": "fiber_image_selection.v2",
            "gate": "valid",
            "rows": rows,
            "bootstrap": bootstrap,
            "git_commit": _git(),
            "screen_hashes": [file_sha256(path) for path in args.screen_evidence],
        },
    )
    return 0


def _bootstrap_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(90210)
    result = {}
    for panel_seed in PANEL_SEEDS:
        by_function = {}
        sampled_by_function: dict[str, torch.Tensor] = {}
        indices_by_class = torch.stack(
            [torch.randint(start, start + 50, (10000, 50), generator=generator) for start in range(0, 500, 50)]
        )
        for function_id in ("S0", "S1a", "S1b", "S1c"):
            selected = [row for row in rows if row["function_id"] == function_id]
            if len(selected) != 3:
                raise ValueError("bootstrap requires exactly three training seeds per function")
            vectors = torch.tensor(
                [
                    next(
                        panel["gain_vector"]
                        for panel in row["panels"]
                        if panel["panel_seed"] == panel_seed
                    )
                    for row in selected
                ],
                dtype=torch.float32,
            )
            class_gains = torch.stack(
                [vectors[:, indices_by_class[class_index]].mean(dim=2) for class_index in range(10)], dim=1
            )
            gains_by_seed = class_gains.mean(dim=1)
            gains = gains_by_seed.median(dim=0).values
            sampled_by_function[function_id] = gains
            by_function[function_id] = {
                "median_gain": float(vectors.mean(dim=1).median()),
                "ci95": [float(gains.quantile(0.025)), float(gains.quantile(0.975))],
                "seed_median_gains": [float(value) for value in gains_by_seed.mean(dim=1)],
                "seed": 90210,
                "replicates": 10000,
            }
        observed = {function_id: by_function[function_id]["median_gain"] for function_id in by_function}
        maximum = max(observed, key=observed.get)
        minimum = min(observed, key=observed.get)
        difference = sampled_by_function[maximum] - sampled_by_function[minimum]
        by_function["cross_function"] = {
            "maximum_function": maximum,
            "minimum_function": minimum,
            "observed_difference": float(observed[maximum] - observed[minimum]),
            "ci95": [float(difference.quantile(0.025)), float(difference.quantile(0.975))],
            "seed": 90210,
            "replicates": 10000,
        }
        result[str(panel_seed)] = by_function
    return result


def closeout(args) -> int:
    selection = _read(_required(args, "selection_evidence"))
    if (
        selection.get("schema") != "fiber_image_selection.v2"
        or selection.get("gate") != "valid"
    ):
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_closeout.v2",
                "gate": "invalid_harness",
                "conclusion": "invalid_harness",
                "structural_validity": False,
                "measurement_validity": False,
            },
        )
        return 2
    rows = selection.get("rows", [])
    if len(rows) != 12 or set(selection.get("bootstrap", {})) != {
        str(seed) for seed in PANEL_SEEDS
    }:
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_closeout.v2",
                "gate": "invalid_harness",
                "conclusion": "invalid_harness",
                "structural_validity": False,
                "measurement_validity": False,
            },
        )
        return 2
    selection_root = Path(args.selection_evidence).resolve().parent
    required_roles = {"candidate_fiber", "same_fiber_b0", "candidate_identity"}
    try:
        for row in rows:
            for panel in row["panels"]:
                artifact_root = (selection_root / panel["artifact_root"]).resolve()
                if selection_root not in artifact_root.parents:
                    raise ValueError("locked artifact root escaped selection run")
                if set(panel.get("artifacts", {})) != required_roles:
                    raise ValueError("locked artifact roles are incomplete")
                for role in required_roles:
                    refs = panel["artifacts"][role]
                    for name in ("labels", "codes", "raw_cmos", "fixed_cmos", "primary_prediction", "secondary_prediction"):
                        path = _resolve(refs[name], artifact_root)
                        value = safe_torch_load(path, map_location="cpu")
                        if refs[name].get("tensor_sha256") != _tensor_content_hash(value):
                            raise ValueError("locked tensor content hash mismatch")
                    if role != "candidate_identity":
                        for name in ("port_input_coefficients", "port_output_coefficients", "fiber_input_amplitude", "fiber_output_amplitude", "fiber_input_phase", "fiber_output_phase", "fiber_field_provenance"):
                            if name not in refs:
                                raise ValueError("fiber locked field artifact is missing")
                _resolve(panel["correctness_artifact"], artifact_root)
    except (KeyError, TypeError, ValueError):
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_closeout.v2",
                "gate": "invalid_harness",
                "conclusion": "invalid_harness",
                "structural_validity": False,
                "measurement_validity": False,
                "reason": "locked artifacts are incomplete or tampered",
            },
        )
        return 2
    functions = {}
    for function_id in ("S0", "S1a", "S1b", "S1c"):
        function_rows = [row for row in rows if row["function_id"] == function_id]
        panel_gates = []
        identity_gaps = []
        for panel in PANEL_SEEDS:
            measurements = [
                next(item for item in row["panels"] if item["panel_seed"] == panel)
                for row in function_rows
            ]
            gains = [
                float(
                    item["candidate"]["secondary_accuracy"]
                    - item["b0"]["secondary_accuracy"]
                )
                for item in measurements
            ]
            panel_gates.append(
                float(selection["bootstrap"][str(panel)][function_id]["median_gain"])
                >= 0.015
                and float(selection["bootstrap"][str(panel)][function_id]["ci95"][0])
                > 0
                and sum(gain > 0 for gain in gains) >= 2
            )
            identity_gaps.extend(float(item["identity_gap"]) for item in measurements)
        candidates = [
            item["candidate"] for row in function_rows for item in row["panels"]
        ]
        quality = all(
            item.get("finite")
            and item.get("nonnegative")
            and item.get("passive_available")
            and item.get("passive")
            and item.get("saturation", 1.0) <= 0.08
            and item.get("primary_min_recall", 0.0) > 0
            and item.get("camera_fraction", 0.0) >= 0.15
            and item.get("minimum_diversity", 0.0) >= 0.35
            and item.get("coverage_ratio", 99.0) <= 1.35
            for item in candidates
        )
        primary_quality = all(
            item["candidate"].get("primary_accuracy", -1.0)
            >= item["b0"].get("primary_accuracy", 99.0) - 0.03
            for row in function_rows
            for item in row["panels"]
        )
        functions[function_id] = {
            "panel_gain": all(panel_gates),
            "quality": quality and primary_quality,
            "primary_quality": primary_quality,
            "identity_gap": max(identity_gaps, default=99.0),
            "passed": all(panel_gates)
            and quality
            and max(identity_gaps, default=99.0) <= 0.05,
        }
    passed = [
        function_id for function_id, result in functions.items() if result["passed"]
    ]
    measurement_validity = bool(rows) and all(
        item["candidate"].get("finite")
        and item["b0"].get("finite")
        and item["identity"].get("finite")
        for row in rows
        for item in row["panels"]
    )
    cross = {
        str(panel): selection["bootstrap"][str(panel)].get("cross_function")
        for panel in PANEL_SEEDS
    }
    cross_difference_valid = all(
        item is not None and item.get("ci95", [0.0, 0.0])[0] > 0.0
        for item in cross.values()
    )
    if not measurement_validity:
        conclusion, gate = "invalid_harness", "invalid_harness"
    elif len(passed) == 0:
        conclusion, gate = "no_reproducible_fiber_gain", "valid_failed"
    elif len(passed) == 1:
        conclusion, gate = "fiber_specific_only_gain", "valid_failed"
    elif cross_difference_valid and all(
        float(item["observed_difference"]) >= 0.015 for item in cross.values()
    ):
        conclusion, gate = "fiber_function_dependent_generator_gain", "valid_failed"
    else:
        conclusion, gate = "common_fiber_generator_gain", "valid_failed"
    _write(
        Path(args.output),
        {
            "schema": "fiber_image_closeout.v2",
            "gate": gate,
            "conclusion": conclusion,
            "structural_validity": True,
            "measurement_validity": measurement_validity,
            "scientific_gates": {
                "functions": functions,
                "passed_functions": passed,
                "cross_function_bootstrap": cross,
            },
            "formal_promotion": False,
            "synthetic_diagnostic_only": True,
            "selection_sha256": file_sha256(args.selection_evidence),
        },
    )
    return 0 if gate == "valid_failed" else 2


def dispatch(args) -> int:
    return {
        "validate": validate,
        "baseline": baseline,
        "preflight": preflight,
        "screen": screen,
        "select": select,
        "closeout": closeout,
    }[args.research_action](args)
