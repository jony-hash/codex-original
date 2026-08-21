"""Fail-closed paired image-generation research lane for synthetic SMF operators."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ...config import load_config
from ...evidence import (
    atomic_torch_save,
    atomic_write_json,
    file_sha256,
    safe_torch_load,
)
from ...optics.fiber_operator import FiberTransportEnsembleOLS
from ...optics.fiber_pool import FiberTransportPool
from ...student.conditional import camera_code_fraction, fixed_exposure_view
from ...student.evaluator import (
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
    load_local_mnist,
)
from .._runtime_shared import random_codes, split_mnist
from ..current.runtime import _seeded_conditional_bundle, _system_signature
from . import fiber_robustness as robust

FUNCTION_IDS = ("I0", "S0", "S1a", "S1b", "S1c", "N0")
TRAIN_SEEDS = (5201, 5202, 5203)
PANEL_SEEDS = (4242, 104729, 161803)


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
            bundle_payload.get("schema") != "fiber_image_measurement_bundle.v1"
            or evidence_payload.get("schema") != "fiber_image_measurement_evidence.v1"
            or evidence_payload.get("bundle_sha256") != file_sha256(bundle)
            or evidence_payload.get("config_hash") != result["config_hash"]
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
) -> dict[str, Any]:
    config, device, _, _, _, _, primary, secondary, bundle, pool, _, _ = context
    settings = config.conditional_generation
    if settings is None:
        raise ValueError("conditional generation settings are required")
    original_ols = bundle.system.ols
    _select(bundle, pool, function_id)
    codes = random_codes(count, device=device, seed=seed)
    labels = torch.arange(10, device=device).repeat_interleave(count // 10)
    rows, views, primary_logits, secondary_logits, traces, port_inputs, port_outputs = (
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
            c, y = (
                codes[start : start + config.execution.evaluation_microbatch_size],
                labels[start : start + config.execution.evaluation_microbatch_size],
            )
            with torch.inference_mode():
                command = bundle.generator(c, y)
                result = bundle.system.forward(
                    bundle.system.command(command), return_optical_trace=trace
                )
                raw = result.raw_camera.tensor
                view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
                _, p = primary(view)
                _, s = secondary(view)
            rows.append(raw.detach())
            views.append(view.detach())
            primary_logits.append(p.detach())
            secondary_logits.append(s.detach())
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
    finally:
        bundle.system.ols = original_ols
    raw, view = torch.cat(rows), torch.cat(views)
    p, s = torch.cat(primary_logits).argmax(1), torch.cat(secondary_logits).argmax(1)
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
    else:
        metric["port_available"] = True
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
        refs[name] = _ref(path, root)
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
        refs["slm_amplitude"] = _ref(command_path, root)
    if result.get("port_available"):
        for name in ("port_input_coefficients", "port_output_coefficients"):
            path = directory / f"{name}.pt"
            atomic_torch_save(path, result[name])
            refs[name] = _ref(path, root)
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
                "schema": "fiber_image_validation.v1",
                "gate": "invalid_harness",
                "reason": "dirty worktree",
            },
        )
        return 2
    fixture, config_path = _required(args, "lp01_fixture"), _required(args, "config")
    config = load_config(str(config_path))
    pool = FiberTransportPool()
    spec = asdict(FiberImageResearchSpec())
    payload = {
        "schema": "fiber_image_validation.v1",
        "gate": "valid",
        "git_commit": _git(),
        "config_hash": robust._config_hash(config),
        "fixture_sha256": file_sha256(fixture),
        "spec": spec,
        "functions": _mapping(pool),
        "pool_hash": hashlib.sha256(
            json.dumps(pool.evidence(), sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    _write(Path(args.output), payload)
    _write(
        Path(args.measurement_bundle_output),
        {
            "schema": "fiber_image_measurement_bundle.v1",
            "gate": "valid",
            "validation_sha256": file_sha256(args.output),
            "config_hash": payload["config_hash"],
        },
    )
    _write(
        Path(args.measurement_evidence_output),
        {
            "schema": "fiber_image_measurement_evidence.v1",
            "gate": "valid",
            "bundle_sha256": file_sha256(args.measurement_bundle_output),
            "config_hash": payload["config_hash"],
            "git_commit": _git(),
        },
    )
    return 0


def baseline(args) -> int:
    validation = _read(_required(args, "validate_evidence"))
    if (
        validation.get("schema") != "fiber_image_validation.v1"
        or validation.get("gate") != "valid"
        or validation.get("git_commit") != _git()
        or _dirty()
    ):
        raise ValueError("validation is stale or worktree is dirty")
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
    _write(
        output_path if output_path.suffix == ".json" else root / "baseline.json",
        {
            "schema": "fiber_image_baseline.v1",
            "gate": "valid",
            "paired_seed": 2027,
            "sample_count": 100,
            "outputs": outputs,
            "validation_sha256": file_sha256(args.validate_evidence),
        },
    )
    return 0


def preflight(args) -> int:
    if _dirty():
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_preflight.v1",
                "gate": "invalid_harness",
                "reason": "dirty worktree",
            },
        )
        return 2
    validation = _read(_required(args, "validate_evidence"))
    if (
        validation.get("schema") != "fiber_image_validation.v1"
        or validation.get("gate") != "valid"
    ):
        raise ValueError("validation evidence did not pass")
    context = _context(args)
    config, bundle, pool = context[0], context[8], context[9]
    _select(bundle, pool, args.function_id)
    optimizer = torch.optim.Adam(
        bundle.generator.parameters(),
        lr=config.conditional_generation.generator_learning_rate,
    )
    before = {
        key: value.detach().clone()
        for key, value in bundle.generator.state_dict().items()
    }
    losses = []
    for step in range(3):
        losses.append(
            robust._update(
                context,
                matrix_id=_mapping(pool)[args.function_id],
                seed=5201,
                step=step,
                optimizer=optimizer,
            )
        )
    checks = {
        "generator_update": any(
            not torch.equal(before[key], bundle.generator.state_dict()[key])
            for key in before
        ),
        "finite": all(torch.isfinite(torch.tensor(losses))),
        "full_phase_port": False,
        "full_phase_cmos": False,
        "resume_replay": False,
    }
    _write(
        Path(args.output),
        {
            "schema": "fiber_image_preflight.v1",
            "gate": "invalid_harness",
            "function_id": args.function_id,
            "updates": 3,
            "losses": losses,
            "checks": checks,
            "reason": "negative-control and resume gates are mandatory and not yet passed",
        },
    )
    return 2


def screen(args) -> int:
    evidence = _read(_required(args, "preflight_evidence"))
    if (
        evidence.get("schema") != "fiber_image_preflight.v1"
        or evidence.get("gate") != "valid"
    ):
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_screen.v1",
                "gate": "invalid_harness",
                "reason": "preflight did not pass",
            },
        )
        return 2
    raise ValueError("screen is blocked until valid image preflight exists")


def select(args) -> int:
    evidence = [_read(path) for path in args.screen_evidence]
    if len(evidence) != 12 or any(
        item.get("schema") != "fiber_image_screen.v1" or item.get("gate") != "valid"
        for item in evidence
    ):
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_selection.v1",
                "gate": "invalid_harness",
                "reason": "twelve valid screen evidences are required",
            },
        )
        return 2
    raise ValueError("selection is blocked until real screens exist")


def closeout(args) -> int:
    selection = _read(_required(args, "selection_evidence"))
    if (
        selection.get("schema") != "fiber_image_selection.v1"
        or selection.get("gate") != "valid"
    ):
        _write(
            Path(args.output),
            {
                "schema": "fiber_image_closeout.v1",
                "gate": "invalid_harness",
                "conclusion": "invalid_harness",
                "structural_validity": False,
                "measurement_validity": False,
            },
        )
        return 2
    raise ValueError("closeout requires complete paired selection evidence")


def dispatch(args) -> int:
    return {
        "validate": validate,
        "baseline": baseline,
        "preflight": preflight,
        "screen": screen,
        "select": select,
        "closeout": closeout,
    }[args.research_action](args)
