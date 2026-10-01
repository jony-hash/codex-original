"""v2.3 teacher-free electronic-oracle feasibility lifecycle.

This module is deliberately separate from the closed v2.2 capacity map.  It
does not construct an optical operator, import the formal runtime, or authorize
any physical candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.evidence import safe_torch_load, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, tensor_sha256
from ldm_project.v2.student.evaluator import SmoothMNISTEvaluator, load_frozen_evaluator, load_local_mnist, stratified_mnist_indices
from ldm_project.v2.application.research.shared.measurement import measure_candidate
from ldm_project.v2.application.research.capacity.o0_models import ProjectionDiscriminator, build_o0_candidate, inference_state_dict
from ldm_project.v2.application.research.capacity.o0_objectives import discriminator_loss, generator_loss, soft_anchor_loss
from ldm_project.v2.application.research.capacity.o0_spec import O0FeasibilityEvidence, O0FeasibilitySpec, O0SplitManifest, load_o0_spec, o0_input_hashes


def _root(spec: O0FeasibilitySpec, action: str) -> Path:
    target = Path(spec.output_root) / f"{datetime.now():%Y%m%d}_{spec.version}" / "runs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_research_{action}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _provenance(spec: O0FeasibilitySpec, config_path: str) -> dict[str, str]:
    root = _repo_root()
    def run(*command: str) -> str:
        result = subprocess.run(("git", *command), cwd=root, check=True, capture_output=True, text=True)
        return result.stdout
    status = run("status", "--porcelain", "--untracked-files=all")
    commit = run("rev-parse", "HEAD").strip()
    patch = run("diff", "--binary") + run("diff", "--cached", "--binary")
    lifecycle = [
        root / "src/ldm_project/v2/application/research/capacity/o0_feasibility.py",
        root / "src/ldm_project/v2/application/research/capacity/o0_models.py",
        root / "src/ldm_project/v2/application/research/capacity/o0_objectives.py",
        root / "src/ldm_project/v2/application/research/capacity/o0_spec.py",
        root / "src/ldm_project/v2/application/research/shared/measurement.py",
        root / "src/ldm_project/v2/application/cli.py",
        root / "src/ldm_project/v2/application/cli_schema.py",
    ]
    code_hash = canonical_sha256({str(path.relative_to(root)): file_sha256(path) for path in lifecycle if path.is_file()})
    return {
        "git_commit": commit,
        "git_dirty_patch": canonical_sha256(patch),
        "git_status": canonical_sha256(status),
        "lifecycle_code": code_hash,
        "candidate_contract": canonical_sha256({item.candidate_id: item.__dict__ for item in spec.candidates}),
        "training_contract": canonical_sha256({"objective": spec.objective_id, "microbatch": spec.microbatch_size, "effective_batch": spec.effective_batch_size, "probe": spec.probe_steps, "smoke": spec.smoke_steps, "confirm_epochs": spec.confirm_epochs}),
        "config_contract": file_sha256(config_path),
    }


def _assert_clean_provenance(spec: O0FeasibilitySpec, config_path: str) -> dict[str, str]:
    provenance = _provenance(spec, config_path)
    if spec.require_clean_git and provenance["git_status"] != canonical_sha256(""):
        raise RuntimeError("v2.3-r2 requires a clean git worktree before formal evidence")
    return provenance


def _load_evaluators(spec: O0FeasibilitySpec, device: torch.device) -> tuple[nn.Module, nn.Module]:
    if not all(Path(value).is_file() for value in (spec.evaluator_checkpoint, spec.smooth_evaluator_checkpoint, spec.evaluator_bundle, spec.evaluator_qualification_evidence)):
        raise FileNotFoundError("v2.3 requires the qualified v2.1 evaluator bundle")
    bundle = json.loads(Path(spec.evaluator_bundle).read_text(encoding="utf-8"))
    qualification = json.loads(Path(spec.evaluator_qualification_evidence).read_text(encoding="utf-8"))
    if qualification.get("status") != "passed":
        raise ValueError("qualified evaluator evidence is not passed")
    if bundle.get("primary_checkpoint") != file_sha256(spec.evaluator_checkpoint) or bundle.get("smooth_checkpoint") != file_sha256(spec.smooth_evaluator_checkpoint):
        raise ValueError("v2.3 evaluator bundle checkpoint hash mismatch")
    primary, payload = load_frozen_evaluator(spec.evaluator_checkpoint, device=device)
    smooth_payload = safe_torch_load(spec.smooth_evaluator_checkpoint)
    if not isinstance(smooth_payload, dict) or smooth_payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind:
        raise ValueError("v2.3 smooth evaluator architecture mismatch")
    if smooth_payload.get("split_manifest_hash") != payload.get("split_manifest_hash"):
        raise ValueError("v2.3 evaluators use different split manifests")
    smooth = SmoothMNISTEvaluator().to(device)
    smooth.load_state_dict(smooth_payload["state_dict"], strict=True)
    return primary.freeze(), smooth.freeze()


def _locked_indices(labels: torch.Tensor) -> torch.Tensor:
    selected = []
    for label in range(10):
        rows = torch.nonzero(labels == label, as_tuple=False).flatten()
        if rows.numel() < 500:
            raise ValueError("locked MNIST test split requires at least 500 examples per class")
        selected.append(rows[:500])
    return torch.cat(selected)


def _split_manifest(spec: O0FeasibilitySpec, train_labels: torch.Tensor, test_labels: torch.Tensor) -> tuple[O0SplitManifest, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_indices, dev_indices = stratified_mnist_indices(train_labels, seed=spec.split_seed)
    locked_indices = _locked_indices(test_labels)
    manifest = O0SplitManifest(
        schema_version=1,
        dataset_root=spec.dataset_root,
        seed=spec.split_seed,
        train_indices_hash=canonical_sha256(train_indices.tolist()),
        dev_indices_hash=canonical_sha256(dev_indices.tolist()),
        locked_test_indices_hash=canonical_sha256(locked_indices.tolist()),
        dataset_content_hash=canonical_sha256({"train_labels": tensor_sha256(train_labels), "test_labels": tensor_sha256(test_labels)}),
    )
    manifest.validate()
    return manifest, train_indices, dev_indices, locked_indices


def _evidence(spec: O0FeasibilitySpec, args: argparse.Namespace, action: str, state: str, *, metrics: dict, reason: str | None = None, split_hash: str | None = None, extra: dict[str, str] | None = None) -> dict:
    inputs = o0_input_hashes(spec, args.config)
    inputs.update(_provenance(spec, args.config))
    inputs.update({"objective_id": spec.objective_id, "measurement_contract": spec.measurement_contract})
    if split_hash:
        inputs["split_manifest"] = split_hash
    inputs.update(extra or {})
    return O0FeasibilityEvidence(1, action, state, getattr(args, "candidate", None), args.dataset, getattr(args, "seed", None), spec.config_hash, split_hash, inputs, metrics, reason, int(args.evidence_revision)).to_dict()


def _write(root: Path, name: str, payload: dict) -> Path:
    target = root / name
    atomic_write_json(target, payload)
    return target


def _load_mnist_and_split(spec: O0FeasibilitySpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, O0SplitManifest, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_images, train_labels = load_local_mnist(spec.dataset_root, train=True)
    test_images, test_labels = load_local_mnist(spec.dataset_root, train=False)
    manifest, train_indices, dev_indices, locked_indices = _split_manifest(spec, train_labels, test_labels)
    return train_images, train_labels, test_images, test_labels, manifest, train_indices, dev_indices, locked_indices


def validate(args: argparse.Namespace) -> int:
    spec = load_o0_spec(args.config)
    root = _root(spec, "validate")
    try:
        provenance = _assert_clean_provenance(spec, args.config)
        if args.dataset != "mnist":
            raise ValueError("v2.3 is pre-registered for MNIST only")
        candidate = spec.candidates[[item.candidate_id for item in spec.candidates].index(args.candidate)] if args.candidate else None
        if candidate:
            candidate.validate()
        train_images, train_labels, test_images, test_labels, manifest, *_ = _load_mnist_and_split(spec)
        _write(root, "o0_split_manifest.json", manifest.__dict__ | {"manifest_hash": manifest.manifest_hash})
        metrics = {"split_manifest": manifest.__dict__, "split_manifest_hash": manifest.manifest_hash, "train_shape": list(train_images.shape), "test_shape": list(test_images.shape), "candidate": None if candidate is None else candidate.__dict__, "provenance": provenance, "optimizer_created": False, "cuda_used": False}
        state, reason = "control_passed", None
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(root, "o0_feasibility_validate_evidence.json", _evidence(spec, args, "validate", state, metrics=metrics, reason=reason, split_hash=metrics.get("split_manifest_hash")))
    print(path)
    return 0 if state == "control_passed" else 1


def preflight(args: argparse.Namespace) -> int:
    spec = load_o0_spec(args.config)
    root = _root(spec, "preflight")
    try:
        if args.candidate != "D0" or args.seed != 1234:
            raise ValueError("v2.3-r2 preflight is fixed to D0 seed 1234")
        provenance = _assert_clean_provenance(spec, args.config)
        device = _device(args.device)
        images, labels, _, _, manifest, train_indices, _, _ = _load_mnist_and_split(spec)
        primary, smooth = _load_evaluators(spec, device)
        candidate = build_o0_candidate("D0").to(device)
        discriminator = ProjectionDiscriminator(code_dim=32, image_size=28).to(device)
        real, batch_labels = _balanced_batch(images, labels, train_indices, device=device, seed=args.seed)
        code = torch.randn((spec.microbatch_size, 32), device=device, generator=torch.Generator(device=device).manual_seed(args.seed + 100_000))
        real, batch_labels = real[: spec.microbatch_size], batch_labels[: spec.microbatch_size]
        fake = _render(candidate, "D0", code, batch_labels)
        d_loss, _ = discriminator_loss(discriminator, real, fake, batch_labels, apply_r1=False)
        d_loss.backward()
        real_features = discriminator(real, batch_labels)[3].detach()
        fake = _render(candidate, "D0", code, batch_labels)
        g_loss, _ = generator_loss(discriminator, fake, batch_labels, code, real_features=real_features)
        g_loss.backward()
        grad_values = [parameter.grad for parameter in (*candidate.parameters(), *discriminator.parameters()) if parameter.grad is not None]
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        primary_grads_clean = all(parameter.grad is None for parameter in primary.parameters())
        smooth_grads_clean = all(parameter.grad is None for parameter in smooth.parameters())
        finite = _finite_tree({"d_loss": float(d_loss.detach()), "g_loss": float(g_loss.detach()), "grad_norms": [float(grad.detach().norm()) for grad in grad_values]})
        safe_memory = device.type != "cuda" or peak <= int(5.5 * 1024**3)
        metrics = {"split_manifest": manifest.__dict__, "split_manifest_hash": manifest.manifest_hash, "provenance": provenance, "d_loss": float(d_loss.detach()), "g_loss": float(g_loss.detach()), "generator_gradients": bool(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in candidate.parameters())), "discriminator_gradients": bool(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in discriminator.parameters())), "primary_evaluator_gradients_clean": primary_grads_clean, "smooth_evaluator_gradients_clean": smooth_grads_clean, "finite": finite, "peak_cuda_memory_bytes": peak, "memory_safe": safe_memory, "optimizer_created": False, "optimizer_step": False, "cuda_used": device.type == "cuda"}
        passed = finite and metrics["generator_gradients"] and metrics["discriminator_gradients"] and primary_grads_clean and smooth_grads_clean and safe_memory
        state, reason = ("preflight_passed", None) if passed else ("preflight_failed", "preflight_gradient_or_memory_contract_failed")
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(root, "o0_feasibility_preflight_evidence.json", _evidence(spec, args, "preflight", state, metrics=metrics, reason=reason, split_hash=metrics.get("split_manifest_hash")))
    print(path)
    return 0 if state == "preflight_passed" else 1


def _classification_metrics(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, object]:
    with torch.no_grad():
        _, logits = model(images.to(device))
    prediction = logits.argmax(1).cpu()
    target = labels.cpu()
    recall = [float((prediction[target == label] == label).float().mean()) for label in range(10)]
    return {"accuracy": float((prediction == target).float().mean()), "recall": recall, "minimum_recall": min(recall), "prediction": prediction}


def controls(args: argparse.Namespace) -> int:
    spec = load_o0_spec(args.config)
    root = _root(spec, "controls")
    try:
        provenance = _assert_clean_provenance(spec, args.config)
        device = _device(args.device)
        images, labels, _, _, manifest, _, dev_indices, _ = _load_mnist_and_split(spec)
        primary, smooth = _load_evaluators(spec, device)
        dev_images, dev_labels = images.index_select(0, dev_indices), labels.index_select(0, dev_indices)
        p, s = _classification_metrics(primary, dev_images, dev_labels, device), _classification_metrics(smooth, dev_images, dev_labels, device)
        # A class permutation, rather than a one-position roll, is required
        # because the immutable stratified split is concatenated by class.
        shuffled = (dev_labels + 1) % 10
        shuffled_p = _classification_metrics(primary, dev_images, shuffled, device)
        shuffled_s = _classification_metrics(smooth, dev_images, shuffled, device)
        agreement = float((p["prediction"] == s["prediction"]).float().mean())
        real_pass = min(float(p["accuracy"]), float(s["accuracy"])) >= .95 and min(float(p["minimum_recall"]), float(s["minimum_recall"])) >= .85 and agreement >= .80
        shuffled_fail = max(float(shuffled_p["accuracy"]), float(shuffled_s["accuracy"])) < .95
        template_images = torch.cat([dev_images[dev_labels == label][:1].repeat(17, 1, 1, 1) for label in range(10)])
        template_labels = torch.arange(10).repeat_interleave(17)
        template_primary = _classification_metrics(primary, template_images, template_labels, device)
        template_smooth = _classification_metrics(smooth, template_images, template_labels, device)
        template_unique = int(torch.unique(template_images.flatten(1), dim=0).shape[0])
        template_fails = template_unique < 80
        metrics = {"split_manifest": manifest.__dict__, "split_manifest_hash": manifest.manifest_hash, "real": {"primary": {key: value for key, value in p.items() if key != "prediction"}, "smooth": {key: value for key, value in s.items() if key != "prediction"}, "agreement": agreement}, "label_shuffle": {"primary_accuracy": float(shuffled_p["accuracy"]), "smooth_accuracy": float(shuffled_s["accuracy"])}, "template": {"primary_accuracy": float(template_primary["accuracy"]), "smooth_accuracy": float(template_smooth["accuracy"]), "unique_images": template_unique}, "controls": {"real_pass": real_pass, "label_shuffle_fails": shuffled_fail, "template_control_fails": template_fails}, "provenance": provenance, "optimizer_created": False, "cuda_used": device.type == "cuda"}
        state, reason = ("control_passed", None) if real_pass and shuffled_fail and template_fails else ("control_failed", "real_or_label_shuffle_or_template_control_failed")
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(root, "o0_feasibility_controls_evidence.json", _evidence(spec, args, "controls", state, metrics=metrics, reason=reason, split_hash=metrics.get("split_manifest_hash")))
    print(path)
    return 0 if state == "control_passed" else 1


def _render(model: nn.Module, candidate: str, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    output = model(code, labels)
    return output if candidate == "D0" else F.interpolate(output, size=(28, 28), mode="area")


def _balanced_batch(images: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, *, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for label in range(10):
        choices = indices[labels.index_select(0, indices) == label]
        rows.append(choices.index_select(0, torch.randint(choices.numel(), (5,), generator=generator)))
    selected = torch.cat(rows)
    return images.index_select(0, selected).to(device), labels.index_select(0, selected).to(device)


def _measure(model: nn.Module, candidate: str, images: torch.Tensor, labels: torch.Tensor, primary: nn.Module, smooth: nn.Module, device: torch.device, *, reference: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict[str, object]:
    code_dim = int(getattr(model, "code_dim"))
    return measure_candidate(model=model, render=lambda code, target: _render(model, candidate, code, target), real_images=images, real_labels=labels, primary=primary, smooth=smooth, device=device, output_domain="electronic_oracle", code_dim=code_dim, reference_override=reference)


def _panel_gate(measurement: dict[str, object]) -> dict[str, bool]:
    panels = measurement["panels"]
    assert isinstance(panels, dict)
    primary = min(float(row["primary"]["accuracy"]) for row in panels.values())
    smooth = min(float(row["smooth"]["accuracy"]) for row in panels.values())
    recall = min(min(row[which]["recall"]) for row in panels.values() for which in ("primary", "smooth"))
    agreement = min(float(row["agreement"]) for row in panels.values())
    saturation = max(float(row["saturation"]) for row in panels.values())
    code_fraction = min(float(row["digital_output_code_fraction"]) for row in panels.values())
    diversity = all(min(row[f"{which}_distribution"]["per_class_distance_ratio"]) >= .35 and min(row[f"{which}_distribution"]["precision"]) >= .10 and min(row[f"{which}_distribution"]["recall"]) >= .10 for row in panels.values() for which in ("primary", "smooth"))
    anti_template = all(all(value["feature"]["rank"] >= 8 and value["image"]["rank"] >= 8 and value["unique_quantized"] >= 8 for value in row["anti_template"][which].values()) for row in panels.values() for which in ("primary", "smooth"))
    return {"primary_semantics": primary >= .95, "smooth_semantics": smooth >= .95, "minimum_recall": recall >= .85, "cross_evaluator_agreement": agreement >= .80, "distribution": diversity, "anti_template": anti_template, "code_response": code_fraction >= .10, "saturation": saturation <= .08}


def _finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _worst_margin(measurement: dict[str, object]) -> float:
    panels = measurement["panels"]
    assert isinstance(panels, dict)
    margins: list[float] = []
    for row in panels.values():
        assert isinstance(row, dict)
        for evaluator in ("primary", "smooth"):
            margins.append(float(row[evaluator]["accuracy"]) / .95 - 1.0)
            margins.append(min(row[evaluator]["recall"]) / .85 - 1.0)
            distribution = row[f"{evaluator}_distribution"]
            margins.extend((min(distribution["per_class_distance_ratio"]) / .35 - 1.0, min(distribution["precision"]) / .10 - 1.0, min(distribution["recall"]) / .10 - 1.0))
        margins.append(float(row["agreement"]) / .80 - 1.0)
        margins.append(float(row["digital_output_code_fraction"]) / .10 - 1.0)
        margins.append((.08 - float(row["saturation"])) / .08)
    return min(margins)


def _stage_gate(before: dict[str, object], after: dict[str, object], spec: O0FeasibilitySpec, *, action: str) -> dict[str, object]:
    before_measurement, after_measurement = before["measurement"], after["measurement"]
    before_margin, after_margin = _worst_margin(before_measurement), _worst_margin(after_measurement)
    before_fraction = min(float(row["digital_output_code_fraction"]) for row in before_measurement["panels"].values())
    after_fraction = min(float(row["digital_output_code_fraction"]) for row in after_measurement["panels"].values())
    fraction_ok = after_fraction >= before_fraction * .90 if before_fraction > 1.0e-8 else after_fraction >= before_fraction
    saturation_ok = all(float(row["saturation"]) <= .08 for row in after_measurement["panels"].values())
    safety_ok = bool(after.get("finite", True)) and all(after["gates"].values()) if action == "confirm" else bool(after.get("finite", True)) and after["gates"].get("anti_template", False) and saturation_ok
    return {"before_worst_margin": before_margin, "after_worst_margin": after_margin, "margin_delta": after_margin - before_margin, "code_fraction_before": before_fraction, "code_fraction_after": after_fraction, "code_fraction_ok": fraction_ok, "saturation_ok": saturation_ok, "finite_ok": bool(after.get("finite", True)), "safety_ok": safety_ok, "pass": after_margin > before_margin + spec.probe_margin_epsilon and fraction_ok and saturation_ok and safety_ok}


def _train(spec: O0FeasibilitySpec, args: argparse.Namespace, *, steps: int, root: Path) -> dict[str, object]:
    _assert_clean_provenance(spec, args.config)
    device = _device(args.device)
    images, labels, _, _, manifest, train_indices, _, _ = _load_mnist_and_split(spec)
    candidate = build_o0_candidate(args.candidate).to(device)
    code_dim = int(getattr(candidate, "code_dim"))
    discriminator = ProjectionDiscriminator(code_dim=code_dim, image_size=28).to(device)
    generator_optimizer = torch.optim.Adam(candidate.parameters(), lr=2.0e-4, betas=(0.0, .9))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2.0e-4, betas=(0.0, .9))
    start_step = 0
    if args.resume_checkpoint:
        payload = safe_torch_load(args.resume_checkpoint, map_location=device)
        if not isinstance(payload, dict) or payload.get("config_hash") != spec.config_hash or payload.get("candidate") != args.candidate or payload.get("seed") != args.seed:
            raise ValueError("resume checkpoint context does not match v2.3 request")
        candidate.load_state_dict(payload["generator"], strict=True)
        discriminator.load_state_dict(payload["discriminator"], strict=True)
        generator_optimizer.load_state_dict(payload["generator_optimizer"])
        discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
        start_step = int(payload["step"])
    initial_generator = inference_state_dict(candidate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    for step in range(start_step, steps):
        real, batch_labels = _balanced_batch(images, labels, train_indices, device=device, seed=args.seed + step)
        code = torch.randn((real.shape[0], code_dim), device=device, generator=torch.Generator(device=device).manual_seed(args.seed + 100_000 + step))
        chunks = tuple(slice(start, start + spec.microbatch_size) for start in range(0, real.shape[0], spec.microbatch_size))
        discriminator_optimizer.zero_grad(set_to_none=True)
        d_detail = {}
        for chunk in chunks:
            fake = _render(candidate, args.candidate, code[chunk], batch_labels[chunk])
            d_loss, d_detail = discriminator_loss(discriminator, real[chunk], fake, batch_labels[chunk], apply_r1=(step + 1) % 16 == 0)
            (d_loss / len(chunks)).backward()
        discriminator_optimizer.step()
        generator_optimizer.zero_grad(set_to_none=True)
        g_detail = {}
        for chunk in chunks:
            fake = _render(candidate, args.candidate, code[chunk], batch_labels[chunk])
            real_features = discriminator(real[chunk], batch_labels[chunk])[3].detach()
            anchor = soft_anchor_loss(candidate, batch_labels[chunk], device=device) if args.candidate in {"C2", "C3"} else None
            g_loss, g_detail = generator_loss(discriminator, fake, batch_labels[chunk], code[chunk], real_features=real_features, anchor_value=anchor)
            (g_loss / len(chunks)).backward()
        generator_optimizer.step()
        if (step + 1) % 50 == 0:
            history.append({"step": step + 1, "d_loss": float(d_loss.detach()), "g_loss": float(g_loss.detach()), **d_detail, **g_detail})
        if (step + 1) % 400 == 0:
            _save_resume(root, args, step + 1, candidate, discriminator, generator_optimizer, discriminator_optimizer, history)
    if spec.stage_endpoint_checkpoint:
        _save_resume(root, args, steps, candidate, discriminator, generator_optimizer, discriminator_optimizer, history)
    inference_checkpoint = root / f"o0_{args.candidate}_seed_{args.seed}_inference.pt"
    atomic_torch_save(inference_checkpoint, {"config_hash": spec.config_hash, "candidate": args.candidate, "seed": args.seed, "code_dim": code_dim, "generator": inference_state_dict(candidate), "inference_only": True})
    resume_path = root / f"o0_{args.candidate}_seed_{args.seed}_resume_{steps}.pt"
    return {"split_manifest": manifest.__dict__, "split_manifest_hash": manifest.manifest_hash, "history": history, "step": steps, "checkpoint": str(inference_checkpoint), "checkpoint_sha256": file_sha256(inference_checkpoint), "optimizer_steps": steps, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0, "resume_checkpoint": str(resume_path), "resume_checkpoint_sha256": file_sha256(resume_path), "_initial_generator": initial_generator}


def _save_resume(root: Path, args: argparse.Namespace, step: int, candidate: nn.Module, discriminator: nn.Module, generator_optimizer: torch.optim.Optimizer, discriminator_optimizer: torch.optim.Optimizer, history: list[dict]) -> None:
    source_hash = file_sha256(args.prior_evidence) if getattr(args, "prior_evidence", None) else None
    atomic_torch_save(root / f"o0_{args.candidate}_seed_{args.seed}_resume_{step}.pt", {"config_hash": load_o0_spec(args.config).config_hash, "candidate": args.candidate, "seed": args.seed, "step": step, "source_evidence_sha256": source_hash, "generator": candidate.state_dict(), "discriminator": discriminator.state_dict(), "generator_optimizer": generator_optimizer.state_dict(), "discriminator_optimizer": discriminator_optimizer.state_dict(), "history": history})


def _require_prior(args: argparse.Namespace, spec: O0FeasibilitySpec, allowed: set[str]) -> dict:
    if not args.prior_evidence:
        raise ValueError("this v2.3 action requires --prior-evidence")
    payload = json.loads(Path(args.prior_evidence).read_text(encoding="utf-8"))
    if payload.get("content_hash") != canonical_sha256({key: value for key, value in payload.items() if key != "content_hash"}):
        raise ValueError("prior evidence content hash mismatch")
    candidate_matches = payload.get("candidate") == args.candidate or payload.get("state") == "control_passed"
    if payload.get("config_hash") != spec.config_hash or not candidate_matches or payload.get("state") not in allowed:
        raise ValueError("prior evidence does not authorize this v2.3 action")
    return payload


def _candidate_sequence_guard(args: argparse.Namespace, spec: O0FeasibilitySpec) -> None:
    """Enforce the pre-registered stop/advance order without global promotion."""
    if args.candidate == "D0":
        return
    records = _selected_records(_scan(Path(spec.output_root), spec.config_hash))
    passed = {item.get("candidate") for item in records if item.get("state") in {"final_passed", "confirm_passed"}}
    if "D0" not in passed:
        raise ValueError("D0 must pass the electronic upper-bound gate before command candidates")
    predecessor = {"C0": None, "C1": "C0", "C2": "C1", "C3": "C2"}[args.candidate]
    if predecessor is not None and not any(item.get("candidate") == predecessor and item.get("state") == "final_failed" for item in records):
        raise ValueError(f"candidate {args.candidate} is not authorized before predecessor {predecessor} final failure")


def _selected_records(records: list[dict]) -> list[dict]:
    """Select only latest unambiguous revisions; retain invalid records for reports."""
    by_revision: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record.get("config_hash"), record.get("dataset"), record.get("candidate"), record.get("action"), record.get("seed"), record.get("evidence_revision"))
        by_revision.setdefault(key, []).append(record)
    ambiguous = {key for key, values in by_revision.items() if len({value.get("content_hash") for value in values}) > 1}
    valid = [record for key, values in by_revision.items() if key not in ambiguous for record in values]
    latest: dict[tuple, dict] = {}
    for record in valid:
        key = (record.get("config_hash"), record.get("dataset"), record.get("candidate"), record.get("action"), record.get("seed"))
        if record.get("evidence_revision", 1) >= latest.get(key, {}).get("evidence_revision", -1):
            latest[key] = record
    return list(latest.values())


def _smoke_aggregate_passed(spec: O0FeasibilitySpec) -> bool:
    records = _selected_records(_scan(Path(spec.output_root), spec.config_hash))
    smoke = [record for record in records if record.get("action") == "smoke" and record.get("candidate") == "D0" and record.get("state") in {"smoke_passed", "smoke_failed"} and record.get("seed") in spec.seeds]
    if len({record.get("seed") for record in smoke}) != 3:
        return False
    stage_gates = [record.get("metrics", {}).get("stage_gate", {}) for record in smoke]
    return sum(bool(gate.get("pass")) for gate in stage_gates) >= 2 and statistics.median(float(gate.get("margin_delta", -float("inf"))) for gate in stage_gates) > spec.probe_margin_epsilon


def _evaluate_model(spec: O0FeasibilitySpec, args: argparse.Namespace, state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
    device = _device(args.device)
    images, labels, _, _, manifest, _, _, _ = _load_mnist_and_split(spec)
    primary, smooth = _load_evaluators(spec, device)
    model = build_o0_candidate(args.candidate).to(device)
    model.load_state_dict(state_dict, strict=True)
    measurement = _measure(model, args.candidate, images, labels, primary, smooth, device)
    gates = _panel_gate(measurement)
    return {"split_manifest": manifest.__dict__, "split_manifest_hash": manifest.manifest_hash, "measurement": measurement, "gates": gates, "all_gates_pass": all(gates.values()), "finite": _finite_tree(measurement), "optimizer_created": False, "cuda_used": device.type == "cuda"}


def _evaluate_checkpoint(spec: O0FeasibilitySpec, args: argparse.Namespace, checkpoint: str) -> dict[str, object]:
    """Replay a generator checkpoint with the complete four-panel contract."""
    payload = safe_torch_load(checkpoint)
    if not isinstance(payload, dict) or payload.get("inference_only") is not True:
        raise ValueError("training did not produce an inference-only checkpoint")
    if payload.get("config_hash") != spec.config_hash or payload.get("candidate") != args.candidate or payload.get("seed") != args.seed:
        raise ValueError("checkpoint context does not match v2.3 evidence context")
    result = _evaluate_model(spec, args, payload["generator"])
    result.update({"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)})
    return result


def _run_train_action(args: argparse.Namespace, action: str, steps: int) -> int:
    spec = load_o0_spec(args.config)
    root = _root(spec, action.replace("-", "_"))
    try:
        if args.candidate is None:
            raise ValueError(f"{action} requires --candidate")
        if action == "probe" and (args.candidate != "D0" or args.seed != 1234):
            raise ValueError("v2.3-r2 probe is fixed to D0 seed 1234")
        if action == "smoke" and args.seed == 1234 and not args.resume_checkpoint:
            raise ValueError("D0 seed 1234 smoke must resume the probe endpoint")
        if action == "confirm" and not args.resume_checkpoint:
            raise ValueError("confirm must resume its seed-matched smoke endpoint")
        _candidate_sequence_guard(args, spec)
        allowed = {"control_passed"} if action == "probe" else {"probe_passed"} if action == "smoke" else {"smoke_passed"}
        prior = _require_prior(args, spec, allowed)
        if action == "confirm" and not _smoke_aggregate_passed(spec):
            raise ValueError("three-seed smoke aggregate has not passed")
        if args.resume_checkpoint and prior.get("metrics", {}).get("resume_checkpoint_sha256") != file_sha256(args.resume_checkpoint):
            raise ValueError("resume checkpoint is not bound to the supplied prior evidence")
        if action == "confirm" and prior.get("seed") != args.seed:
            raise ValueError("confirm prior evidence must match the requested seed")
        if action == "smoke" and args.seed == 1234 and prior.get("seed") != args.seed:
            raise ValueError("seed 1234 smoke prior must be the seed-matched probe")
        if args.candidate == "C3" and action == "confirm":
            raise ValueError("C3 is diagnostic-only and cannot receive formal confirm authorization")
        metrics = _train(spec, args, steps=steps, root=root)
        initial_state = metrics.pop("_initial_generator")
        baseline = _evaluate_model(spec, args, initial_state)
        replay = _evaluate_checkpoint(spec, args, str(metrics["checkpoint"]))
        stage_gate = _stage_gate(baseline, replay, spec, action=action)
        metrics["step0_measurement"] = baseline["measurement"]
        metrics["measurement"] = replay["measurement"]
        metrics["gates"] = replay["gates"]
        metrics["all_gates_pass"] = replay["all_gates_pass"]
        metrics["finite"] = replay["finite"]
        metrics["stage_gate"] = stage_gate
        required_state = {"probe": "probe_passed", "smoke": "smoke_passed", "confirm": "confirm_passed"}[action]
        failed_state = {"probe": "probe_failed", "smoke": "smoke_failed", "confirm": "confirm_failed"}[action]
        passed = replay["all_gates_pass"] if action == "confirm" else bool(stage_gate["pass"])
        state, reason = (required_state, None) if passed else (failed_state, "stage_gate_failed")
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    extra = {"prior_evidence": file_sha256(args.prior_evidence)} if getattr(args, "prior_evidence", None) else None
    path = _write(root, f"o0_feasibility_{args.candidate}_{action}_evidence.json", _evidence(spec, args, action, state, metrics=metrics, reason=reason, split_hash=metrics.get("split_manifest_hash"), extra=extra))
    print(path)
    return 0 if state in {"probe_passed", "smoke_passed", "confirm_passed"} else 1


def probe(args: argparse.Namespace) -> int:
    spec = load_o0_spec(args.config)
    return _run_train_action(args, "probe", spec.probe_steps)


def smoke(args: argparse.Namespace) -> int:
    spec = load_o0_spec(args.config)
    return _run_train_action(args, "smoke", spec.smoke_steps)


def confirm(args: argparse.Namespace) -> int:
    spec = load_o0_spec(args.config)
    return _run_train_action(args, "confirm", spec.confirm_epochs * 1_100)


def final(args: argparse.Namespace) -> int:
    spec = load_o0_spec(args.config)
    root = _root(spec, "final")
    requested_set = canonical_sha256(sorted(args.checkpoint))
    try:
        _assert_clean_provenance(spec, args.config)
        _candidate_sequence_guard(args, spec)
        if args.candidate == "C3":
            raise ValueError("C3 is diagnostic-only and cannot run locked-final")
        if len(args.checkpoint) != len(args.source_evidence) or len(args.checkpoint) != 3:
            raise ValueError("locked-final requires exactly three checkpoint/evidence pairs")
        for prior_final in _scan(Path(args.evidence_root) if getattr(args, "evidence_root", None) else Path(spec.output_root), spec.config_hash):
            if prior_final.get("action") == "final" and prior_final.get("candidate") == args.candidate and prior_final.get("inputs", {}).get("checkpoint_set") == requested_set:
                raise ValueError("locked-final is one-shot for a candidate and checkpoint set")
        device = _device(args.device)
        train_images, train_labels, test_images, test_labels, manifest, _, _, locked = _load_mnist_and_split(spec)
        primary, smooth = _load_evaluators(spec, device)
        rows = []
        source_seeds = []
        dev_pass_count = 0
        for checkpoint_path, evidence_path in zip(args.checkpoint, args.source_evidence):
            source = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
            if source.get("content_hash") != canonical_sha256({key: value for key, value in source.items() if key != "content_hash"}):
                raise ValueError("locked-final source evidence content hash mismatch")
            if source.get("state") not in {"confirm_passed", "confirm_failed"} or source.get("config_hash") != spec.config_hash or source.get("candidate") != args.candidate:
                raise ValueError("locked-final source evidence is not a matching confirm result")
            source_seed = source.get("seed")
            if source_seed in source_seeds or source_seed not in spec.seeds:
                raise ValueError("locked-final requires distinct pre-registered confirm seeds")
            source_seeds.append(source_seed)
            payload = safe_torch_load(checkpoint_path, map_location=device)
            if not isinstance(payload, dict) or payload.get("inference_only") is not True or payload.get("config_hash") != spec.config_hash or payload.get("candidate") != args.candidate:
                raise ValueError("final checkpoint is not an inference-only v2.3 checkpoint")
            if source.get("metrics", {}).get("checkpoint_sha256") != file_sha256(checkpoint_path):
                raise ValueError("source evidence is not bound to the supplied checkpoint")
            if source.get("state") == "confirm_failed":
                rows.append({"checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path), "source_evidence_sha256": file_sha256(evidence_path), "dev_pass": False, "final_pass": False, "reason": "dev_confirm_failed"})
                continue
            model = build_o0_candidate(args.candidate).to(device)
            model.load_state_dict(payload["generator"], strict=True)
            reference = (test_images.index_select(0, locked), test_labels.index_select(0, locked))
            measurement = _measure(model, args.candidate, test_images, test_labels, primary, smooth, device, reference=reference)
            gates = _panel_gate(measurement)
            dev_pass_count += 1
            rows.append({"checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path), "source_evidence_sha256": file_sha256(evidence_path), "dev_pass": True, "final_pass": all(gates.values()), "gates": gates, "measurement": measurement})
        if dev_pass_count < 2:
            raise ValueError("locked-final requires at least two dev-confirm-passed seeds")
        passed = sum(bool(row.get("final_pass", False)) for row in rows)
        metrics = {"locked_test_split_manifest_hash": manifest.manifest_hash, "rows": rows, "dev_pass_count": dev_pass_count, "pass_count": passed, "required_pass_count": 2, "optimizer_created": False, "cuda_used": device.type == "cuda"}
        state, reason = ("final_passed", None) if passed >= 2 else ("final_failed", "locked_final_gate_failed")
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as error:
        metrics, state, reason = {}, "invalid_harness", str(error)
    path = _write(root, f"o0_feasibility_{args.candidate}_final_evidence.json", _evidence(spec, args, "final", state, metrics=metrics, reason=reason, split_hash=metrics.get("locked_test_split_manifest_hash"), extra={"checkpoint_set": requested_set}))
    print(path)
    return 0 if state == "final_passed" else 1


def _scan(source: Path, config_hash: str) -> list[dict]:
    records = []
    for path in source.rglob("o0_feasibility_*_evidence.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("config_hash") == config_hash and payload.get("content_hash") == canonical_sha256({key: value for key, value in payload.items() if key != "content_hash"}):
                records.append(payload | {"path": str(path)})
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _classification(records: list[dict]) -> str:
    final_states = {(item.get("candidate"), item.get("state")) for item in records if item.get("action") == "final"}
    if any(item.get("candidate") == "D0" and item.get("state") in {"probe_failed", "smoke_failed", "confirm_failed", "final_failed"} for item in records):
        return "training_objective_or_evaluator_inconclusive"
    if ("C0", "final_passed") in final_states:
        return "electronic_oracle_reachable"
    if ("C1", "final_passed") in final_states:
        return "electronic_generator_capacity"
    if ("C2", "final_passed") in final_states:
        return "exact_anchor_bottleneck"
    if ("C3", "final_passed") in final_states:
        return "input_manifold_bottleneck"
    if ("D0", "final_passed") in final_states:
        return "d0_image_oracle_reachable"
    if ("D0", "final_failed") in final_states:
        return "training_objective_or_evaluator_inconclusive"
    if ("C3", "final_failed") in final_states:
        return "command_parameterization_inconclusive"
    return "pending"


def summarize(args: argparse.Namespace) -> int:
    spec = load_o0_spec(args.config)
    root = _root(spec, "summarize")
    records = _scan(Path(args.evidence_root), spec.config_hash)
    finals = [item for item in records if item.get("action") == "final"]
    summary = {"schema_version": 1, "config_hash": spec.config_hash, "records": records, "classification": _classification(records), "final_pass_count": sum(item.get("state") == "final_passed" for item in finals), "candidate_states": {candidate: sorted({item.get("state") for item in records if item.get("candidate") == candidate}) for candidate in ("D0", "C0", "C1", "C2", "C3")}, "formal_promotion_authorized": False, "physical_authorization": False, "permitted_next_actions": ["write an independent C0 command-generator plan only after d0_image_oracle_reachable"]}
    atomic_write_json(root / "o0_feasibility_summary.json", summary)
    (root / "o0_feasibility_report.md").write_text("# v2.3 O0 feasibility summary\n\nThis teacher-free electronic lifecycle does not authorize optical training, external-data experiments, or promotion. Any negative result is scoped to the declared electronic candidates and evaluator contract.\n", encoding="utf-8")
    print(root / "o0_feasibility_summary.json")
    return 0


def closeout(args: argparse.Namespace) -> int:
    spec = load_o0_spec(args.config)
    root = _root(spec, "closeout")
    records = _scan(Path(args.evidence_root), spec.config_hash)
    final_pass_count = sum(item.get("action") == "final" and item.get("state") == "final_passed" for item in records)
    classification = _classification(records)
    status = "closed" if classification != "pending" else "incomplete"
    summary = {"schema_version": 1, "status": status, "classification": classification, "config_hash": spec.config_hash, "records": records, "final_pass_count": final_pass_count, "physical_authorization": False, "formal_promotion_authorized": False, "permitted_next_actions": ["write an independent C0 command-generator plan only after d0_image_oracle_reachable"] if classification == "d0_image_oracle_reachable" else [], "prohibited_next_actions": ["C0-C3", "O1-O5", "Teacher", "diffusion", "external datasets"] if classification != "d0_image_oracle_reachable" else ["automatic physical promotion"]}
    summary_path = root / "o0_feasibility_closeout_summary.json"
    atomic_write_json(summary_path, summary)
    manifest = {"schema_version": 1, "status": status, "classification": classification, "config_hash": spec.config_hash, "summary_sha256": file_sha256(summary_path), "record_count": len(records), "final_pass_count": final_pass_count, "cuda_used": False, "optimizer_created": False, "physical_authorization": False, "formal_promotion_authorized": False}
    atomic_write_json(root / "o0_feasibility_closeout_manifest.json", manifest)
    (root / "o0_feasibility_closeout_report.md").write_text("# v2.3 O0 feasibility closeout\n\nStatus: ``" + status + "``.\n\nClassification: ``" + classification + "``.\n\nThe D0 probe stopped the branch before smoke, confirm, or locked-final. This report is limited to the teacher-free electronic image-domain objective. It neither establishes nor tests optical capacity, phase-only capacity, spatial degrees of freedom, or cross-dataset generality.\n", encoding="utf-8")
    print(summary_path)
    return 0


def dispatch(args: argparse.Namespace) -> int:
    actions: dict[str, Callable[[argparse.Namespace], int]] = {"validate": validate, "preflight": preflight, "controls": controls, "probe": probe, "smoke": smoke, "confirm": confirm, "final": final, "summarize": summarize, "closeout": closeout}
    return actions[args.research_action](args)
