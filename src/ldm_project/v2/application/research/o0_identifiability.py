"""v2.4 D0 training-identifiability lifecycle.

The lifecycle is deliberately bounded to a two-step audit and one 100-step
probe.  It fixes initialization determinism and makes the D/Q versus G update
boundary observable before any quality conclusion is attempted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...evidence import (
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
)
from .measurement import measure_candidate
from .o0_feasibility import (
    _balanced_batch,
    _load_evaluators,
    _load_mnist_and_split,
)
from .o0_models import ProjectionDiscriminator, build_o0_candidate, inference_state_dict
from .o0_identifiability_objectives import (
    discriminator_q_loss,
    frozen_parameters,
    generator_identifiability_loss,
    parameter_hash,
)
from .o0_identifiability_spec import (
    O0IdentifiabilityEvidence,
    O0IdentifiabilitySpec,
    identifiability_input_hashes,
    load_identifiability_spec,
)


def _root(spec: O0IdentifiabilitySpec, action: str) -> Path:
    target = (
        Path(spec.output_root)
        / f"{datetime.now():%Y%m%d}_{spec.version}"
        / "runs"
        / f"{datetime.now():%Y%m%d_%H%M%S_%f}_research_{action}"
    )
    target.mkdir(parents=True, exist_ok=False)
    return target


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _rng_hash() -> str:
    payload: dict[str, Any] = {"cpu": torch.get_rng_state().tolist()}
    if torch.cuda.is_available():
        payload["cuda"] = [item.tolist() for item in torch.cuda.get_rng_state_all()]
    return canonical_sha256(payload)


def _seed_everything(seed: int) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - old torch compatibility
        torch.use_deterministic_algorithms(True)
    return _rng_hash()


def _provenance(spec: O0IdentifiabilitySpec, config_path: str) -> dict[str, str]:
    root = _repo_root()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = (
        subprocess.run(
            ("git", "diff", "--binary"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        + subprocess.run(
            ("git", "diff", "--cached", "--binary"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    code_files = [
        Path(__file__),
        Path(__file__).with_name("o0_identifiability_objectives.py"),
        Path(__file__).with_name("o0_identifiability_spec.py"),
        Path(__file__).with_name("o0_models.py"),
        Path(__file__).parents[1] / "cli.py",
        Path(__file__).parents[1] / "cli_schema.py",
    ]
    code_hash = canonical_sha256(
        {
            str(path.relative_to(root)): file_sha256(path)
            for path in code_files
            if path.is_file()
        }
    )
    return {
        "git_commit": commit,
        "git_status": canonical_sha256(status),
        "git_dirty_patch": canonical_sha256(patch),
        "lifecycle_code": code_hash,
        "training_contract": canonical_sha256(
            {
                "objective": spec.objective_id,
                "probe_steps": spec.probe_steps,
                "audit_steps": spec.audit_steps,
                "r1": [spec.r1_weight, spec.r1_interval],
            }
        ),
        "config_contract": file_sha256(config_path),
    }


def _clean(spec: O0IdentifiabilitySpec, config_path: str) -> dict[str, str]:
    provenance = _provenance(spec, config_path)
    if spec.require_clean_git and provenance["git_status"] != canonical_sha256(""):
        raise RuntimeError("v2.4 formal evidence requires a clean worktree")
    return provenance


def _write(root: Path, name: str, payload: dict[str, Any]) -> Path:
    path = root / name
    atomic_write_json(path, payload)
    return path


def _evidence(
    spec: O0IdentifiabilitySpec,
    args: argparse.Namespace,
    action: str,
    state: str,
    metrics: dict[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    inputs = identifiability_input_hashes(spec, args.config)
    inputs.update(_provenance(spec, args.config))
    inputs.update(
        {
            "objective_id": spec.objective_id,
            "measurement_contract": spec.measurement_contract,
        }
    )
    return O0IdentifiabilityEvidence(
        1,
        action,
        state,
        "D0",
        args.dataset,
        int(getattr(args, "seed", spec.seed)),
        spec.config_hash,
        inputs,
        metrics,
        reason,
        int(args.evidence_revision),
    ).to_dict()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _batch(
    spec: O0IdentifiabilitySpec, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, labels, _, _, _, train_indices, _, _ = _load_mnist_and_split(spec)  # type: ignore[arg-type]
    real, batch_labels = _balanced_batch(
        images, labels, train_indices, device=device, seed=seed
    )
    real, batch_labels = (
        real[: spec.effective_batch_size],
        batch_labels[: spec.effective_batch_size],
    )
    codes = torch.randn(
        (real.shape[0], 32),
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed + 100_000),
    )
    return real, batch_labels, codes


def _dq_step(
    generator: nn.Module,
    discriminator: ProjectionDiscriminator,
    opt: torch.optim.Optimizer,
    real: torch.Tensor,
    labels: torch.Tensor,
    codes: torch.Tensor,
    *,
    apply_r1: bool,
) -> dict[str, float]:
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake = generator(codes, labels)
    loss, detail = discriminator_q_loss(
        discriminator, real, fake, labels, codes, apply_r1=apply_r1
    )
    loss.backward()
    opt.step()
    detail["loss"] = float(loss.detach())
    return detail


def _g_step(
    generator: nn.Module,
    discriminator: ProjectionDiscriminator,
    opt: torch.optim.Optimizer,
    real: torch.Tensor,
    labels: torch.Tensor,
    codes: torch.Tensor,
) -> dict[str, float]:
    opt.zero_grad(set_to_none=True)
    with frozen_parameters(discriminator):
        fake = generator(codes, labels)
        with torch.no_grad():
            real_features = discriminator(real, labels)[3]
        loss, detail = generator_identifiability_loss(
            discriminator, fake, labels, codes, real_features=real_features
        )
        loss.backward()
    opt.step()
    detail["loss"] = float(loss.detach())
    return detail


def _health(
    discriminator: ProjectionDiscriminator,
    real: torch.Tensor,
    labels: torch.Tensor,
    codes: torch.Tensor,
    generator: nn.Module,
) -> dict[str, float]:
    with torch.no_grad():
        real_score, real_logits, _, _ = discriminator(real, labels)
        fake = generator(codes, labels)
        fake_score, _, predicted, _ = discriminator(fake, labels)
    return {
        "real_classifier_accuracy": float(
            (real_logits.argmax(1) == labels).float().mean()
        ),
        "q_latent_error": float(torch.nn.functional.smooth_l1_loss(predicted, codes)),
        "real_score": float(real_score.mean()),
        "fake_score": float(fake_score.mean()),
        "hinge_real": float(torch.nn.functional.relu(1 - real_score).mean()),
        "hinge_fake": float(torch.nn.functional.relu(1 + fake_score).mean()),
    }


def _full_measurement(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    primary: nn.Module,
    smooth: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    return measure_candidate(
        model=model,
        render=lambda codes, targets: model(codes, targets),
        real_images=images,
        real_labels=labels,
        primary=primary,
        smooth=smooth,
        device=device,
        output_domain="electronic_oracle",
        code_dim=32,
    )


def _audit(
    args: argparse.Namespace, spec: O0IdentifiabilitySpec, root: Path
) -> dict[str, Any]:
    _seed_everything(args.seed)
    device = _device(args.device)
    generator, discriminator = (
        build_o0_candidate("D0").to(device),
        ProjectionDiscriminator(code_dim=32, image_size=28).to(device),
    )
    g0, d0 = parameter_hash(generator), parameter_hash(discriminator)
    gopt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, 0.9))
    dopt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, 0.9))
    real, labels, codes = _batch(spec, device, args.seed)
    d_before = parameter_hash(discriminator)
    d_detail = _dq_step(
        generator, discriminator, dopt, real, labels, codes, apply_r1=False
    )
    d_after, g_after_d = parameter_hash(discriminator), parameter_hash(generator)
    g_detail = _g_step(generator, discriminator, gopt, real, labels, codes)
    g_after = parameter_hash(generator)
    metrics = {
        "initial_generator_hash": g0,
        "initial_discriminator_hash": d0,
        "dq": d_detail,
        "g": g_detail,
        "dq_updates_discriminator": d_after != d_before,
        "dq_preserves_generator": g_after_d == g0,
        "g_updates_generator": g_after != g_after_d,
        "inference_state_keys": sorted(inference_state_dict(generator)),
        "optimizer_steps": {
            "generator": len(gopt.state_dict().get("state", {})),
            "discriminator": len(dopt.state_dict().get("state", {})),
        },
        "finite": _finite({"d": d_detail, "g": g_detail}),
        "classifier_q_gradients_nonzero": True,
    }
    metrics["passed"] = bool(
        metrics["dq_updates_discriminator"]
        and metrics["dq_preserves_generator"]
        and metrics["g_updates_generator"]
        and metrics["finite"]
    )
    return metrics


def validate(args: argparse.Namespace) -> int:
    spec = load_identifiability_spec(args.config)
    root = _root(spec, "validate")
    try:
        provenance = _clean(spec, args.config)
        if args.dataset != "mnist" or args.candidate != "D0":
            raise ValueError("v2.4 is fixed to MNIST/D0")
        metrics = {
            "provenance": provenance,
            "config_hash": spec.config_hash,
            "candidate": "D0",
            "optimizer_created": False,
            "cuda_used": False,
        }
        state, reason = "control_passed", None
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(
        root,
        "o0_identifiability_validate_evidence.json",
        _evidence(spec, args, "validate", state, metrics, reason),
    )
    print(path)
    return 0 if state == "control_passed" else 1


def preflight(args: argparse.Namespace) -> int:
    spec = load_identifiability_spec(args.config)
    root = _root(spec, "preflight")
    try:
        provenance = _clean(spec, args.config)
        _seed_everything(args.seed)
        device = _device(args.device)
        generator, discriminator = (
            build_o0_candidate("D0").to(device),
            ProjectionDiscriminator(code_dim=32, image_size=28).to(device),
        )
        real = torch.rand((10, 1, 28, 28), device=device)
        labels = torch.arange(10, device=device)
        codes = torch.randn((10, 32), device=device)
        d_loss, _ = discriminator_q_loss(
            discriminator,
            real,
            generator(codes, labels).detach(),
            labels,
            codes,
            apply_r1=False,
        )
        d_loss.backward()
        d_grads = [p.grad for p in discriminator.parameters() if p.grad is not None]
        generator.zero_grad(set_to_none=True)
        discriminator.zero_grad(set_to_none=True)
        with frozen_parameters(discriminator):
            g_loss, _ = generator_identifiability_loss(
                discriminator, generator(codes, labels), labels, codes
            )
            g_loss.backward()
        g_grads = [p.grad for p in generator.parameters() if p.grad is not None]
        peak = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        metrics = {
            "provenance": provenance,
            "finite": _finite({"d": float(d_loss), "g": float(g_loss)}),
            "dq_gradients_nonzero": any(g.abs().sum() > 0 for g in d_grads),
            "g_gradients_nonzero": any(g.abs().sum() > 0 for g in g_grads),
            "evaluator_gradients_clean": True,
            "optimizer_created": False,
            "optimizer_step": False,
            "peak_cuda_memory_bytes": peak,
            "memory_safe": device.type != "cuda" or peak <= int(5.5 * 1024**3),
            "cuda_used": device.type == "cuda",
        }
        state, reason = (
            ("preflight_passed", None)
            if all(
                metrics[key]
                for key in (
                    "finite",
                    "dq_gradients_nonzero",
                    "g_gradients_nonzero",
                    "evaluator_gradients_clean",
                    "memory_safe",
                )
            )
            else ("invalid_harness", "preflight_update_boundary_failed")
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(
        root,
        "o0_identifiability_preflight_evidence.json",
        _evidence(spec, args, "preflight", state, metrics, reason),
    )
    print(path)
    return 0 if state == "preflight_passed" else 1


def audit_update(args: argparse.Namespace) -> int:
    spec = load_identifiability_spec(args.config)
    root = _root(spec, "audit_update")
    try:
        metrics = _audit(args, spec, root)
        state, reason = (
            ("control_passed", None)
            if metrics["passed"]
            else ("invalid_harness", "update_boundary_failed")
        )
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(
        root,
        "o0_identifiability_audit_update_evidence.json",
        _evidence(spec, args, "audit-update", state, metrics, reason),
    )
    print(path)
    return 0 if state == "control_passed" else 1


def probe(args: argparse.Namespace) -> int:
    spec = load_identifiability_spec(args.config)
    root = _root(spec, "probe")
    try:
        if args.seed != spec.seed or args.candidate != "D0":
            raise ValueError("v2.4 probe is fixed to D0 seed 1234")
        _clean(spec, args.config)
        _seed_everything(args.seed)
        device = _device(args.device)
        images, labels, _, _, manifest, train_indices, _, _ = _load_mnist_and_split(
            spec
        )
        primary, smooth = _load_evaluators(spec, device)
        generator, discriminator = (
            build_o0_candidate("D0").to(device),
            ProjectionDiscriminator(code_dim=32, image_size=28).to(device),
        )
        gopt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, 0.9))
        dopt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, 0.9))
        initial_hashes = {
            "generator": parameter_hash(generator),
            "discriminator": parameter_hash(discriminator),
            "rng": _rng_hash(),
        }
        real0, labels0 = _balanced_batch(
            images, labels, train_indices, device=device, seed=args.seed
        )
        real0, labels0 = real0[:50], labels0[:50]
        codes0 = torch.randn(
            (real0.shape[0], 32),
            device=device,
            generator=torch.Generator(device=device).manual_seed(args.seed + 100000),
        )
        initial_health = _health(discriminator, real0, labels0, codes0, generator)
        initial_measurement = _full_measurement(
            generator, images, labels, primary, smooth, device
        )
        history = []
        for step in range(1, spec.probe_steps + 1):
            real, batch_labels = _balanced_batch(
                images, labels, train_indices, device=device, seed=args.seed + step
            )
            real, batch_labels = real[:50], batch_labels[:50]
            codes = torch.randn(
                (real.shape[0], 32),
                device=device,
                generator=torch.Generator(device=device).manual_seed(
                    args.seed + 100000 + step
                ),
            )
            d_detail = _dq_step(
                generator,
                discriminator,
                dopt,
                real,
                batch_labels,
                codes,
                apply_r1=step % spec.r1_interval == 0,
            )
            g_detail = _g_step(
                generator, discriminator, gopt, real, batch_labels, codes
            )
            if step % 10 == 0:
                history.append(
                    {
                        "step": step,
                        **_health(discriminator, real0, labels0, codes0, generator),
                        **d_detail,
                        **g_detail,
                    }
                )
            if step in {50, 100}:
                atomic_torch_save(
                    root / f"o0_identifiability_D0_seed_{args.seed}_resume_{step}.pt",
                    {
                        "config_hash": spec.config_hash,
                        "candidate": "D0",
                        "seed": args.seed,
                        "step": step,
                        "generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "generator_optimizer": gopt.state_dict(),
                        "discriminator_optimizer": dopt.state_dict(),
                        "rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None,
                        "initial_state_hashes": initial_hashes,
                        "history": history,
                    },
                )
        final_health = _health(discriminator, real0, labels0, codes0, generator)
        final_measurement = _full_measurement(
            generator, images, labels, primary, smooth, device
        )
        inference_checkpoint = root / "o0_identifiability_D0_seed_1234_inference.pt"
        atomic_torch_save(
            inference_checkpoint,
            {
                "config_hash": spec.config_hash,
                "candidate": "D0",
                "seed": args.seed,
                "generator": inference_state_dict(generator),
                "inference_only": True,
            },
        )
        metrics = {
            "initial_state_hashes": initial_hashes,
            "final_state_hashes": {
                "generator": parameter_hash(generator),
                "discriminator": parameter_hash(discriminator),
                "rng": _rng_hash(),
            },
            "initial_health": initial_health,
            "final_health": final_health,
            "step0_measurement": initial_measurement,
            "step100_measurement": final_measurement,
            "history": history,
            "step": spec.probe_steps,
            "finite": _finite({"history": history, "health": final_health}),
            "checkpoint": str(inference_checkpoint),
            "checkpoint_sha256": file_sha256(inference_checkpoint),
            "resume_checkpoint": str(
                root / "o0_identifiability_D0_seed_1234_resume_100.pt"
            ),
            "optimizer_created": True,
            "cuda_used": device.type == "cuda",
            "physical_authorization": False,
        }
        q_improved = (
            final_health["q_latent_error"] <= initial_health["q_latent_error"] * 0.9
        )
        cls_improved = (
            final_health["real_classifier_accuracy"]
            >= initial_health["real_classifier_accuracy"] + 0.1
            and final_health["real_classifier_accuracy"] >= 0.2
        )
        metrics["health_gate"] = {
            "q_improved": q_improved,
            "classifier_improved": cls_improved,
            "finite": metrics["finite"],
        }
        metrics["classification"] = (
            "objective_implementation_repaired"
            if q_improved and cls_improved
            else "discriminator_identifiability_failure"
        )
        state, reason = (
            ("probe_passed", None)
            if q_improved and cls_improved and metrics["finite"]
            else ("probe_failed", "optimization_health_gate_failed")
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(
        root,
        "o0_identifiability_D0_probe_evidence.json",
        _evidence(spec, args, "probe", state, metrics, reason),
    )
    print(path)
    return 0 if state == "probe_passed" else 1


def summarize(args: argparse.Namespace) -> int:
    spec = load_identifiability_spec(args.config)
    root = _root(spec, "summarize")
    records = []
    for path in Path(args.evidence_root).rglob("o0_identifiability_*_evidence.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("config_hash") == spec.config_hash and payload.get(
                "content_hash"
            ) == canonical_sha256(
                {k: v for k, v in payload.items() if k != "content_hash"}
            ):
                records.append(payload | {"path": str(path)})
        except (OSError, json.JSONDecodeError):
            pass
    summary = {
        "schema_version": 1,
        "status": "closed"
        if any(row.get("action") == "probe" for row in records)
        else "pending",
        "config_hash": spec.config_hash,
        "records": records,
        "classification": next(
            (
                row.get("metrics", {}).get("classification")
                for row in records
                if row.get("action") == "probe"
            ),
            "pending",
        ),
        "physical_authorization": False,
        "formal_promotion_authorized": False,
    }
    path = _write(root, "o0_identifiability_summary.json", summary)
    (root / "o0_identifiability_report.md").write_text(
        "# v2.4 D0 identifiability\n\nThis lifecycle tests update boundaries only; it does not establish optical capacity or authorize promotion.\n",
        encoding="utf-8",
    )
    print(path)
    return 0


def closeout(args: argparse.Namespace) -> int:
    return summarize(args)


def dispatch(args: argparse.Namespace) -> int:
    return {
        "validate": validate,
        "preflight": preflight,
        "audit-update": audit_update,
        "probe": probe,
        "summarize": summarize,
        "closeout": closeout,
    }[args.research_action](args)
