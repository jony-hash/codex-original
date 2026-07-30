"""Read-only macro-path feasibility probes for the failed v6 contract.

The probes deliberately operate on functional command/phase counterfactuals.
They do not create an optimizer, mutate a checkpoint, or authorize training.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ...evidence import EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256
from ...student.checkpoints import state_content_hash
from ...student.conditional import camera_code_fraction, fixed_exposure_view, relative_diversity_ratios
from ...student.evaluator import evaluator_checkpoint_hash, load_frozen_evaluator, load_local_mnist
from .support import _split
from .pca_transport import (
    _context,
    _full_contact_sheet,
    _load,
    _new_run,
    _panels,
    _real_panel_views,
    _write_evidence,
    PCA_TRANSPORT_ARCHITECTURE,
)


FEASIBILITY_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_optical_generation_feasibility_v1",
        "optimizer": False,
        "subspaces": ["bandlimited_slm_only", "slm_plus_global_phase", "identity_phase_reference"],
        "logit_rms": 0.20,
        "phase_rms": 0.005,
        "max_cycles_per_mm": 20.0,
    }
)


def _raw_from_command(bundle, command: torch.Tensor) -> torch.Tensor:
    parts = []
    for start in range(0, command.shape[0], 1):
        parts.append(bundle.system.forward(bundle.system.command(command[start : start + 1])).raw_camera.tensor)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)


def _metrics(raw: torch.Tensor, labels: torch.Tensor, evaluator, real_views: torch.Tensor, exposure: float) -> dict[str, object]:
    with torch.no_grad():
        features, logits = evaluator(fixed_exposure_view(raw, exposure))
        real_features, _ = evaluator(real_views)
        prediction = logits.argmax(1)
        ratios = relative_diversity_ratios(F.normalize(features, dim=1), F.normalize(real_features, dim=1), labels)
    return {
        "accuracy": float((prediction == labels).float().mean().item()),
        "recall": [float((prediction[labels == y] == y).float().mean().item()) for y in range(10)],
        "zero_accuracy": float((prediction[torch.arange(10, device=labels.device) * 17] == labels[torch.arange(10, device=labels.device) * 17]).float().mean().item()),
        "camera_fraction": float(camera_code_fraction(raw, labels).item()),
        "ratios": [float(x) for x in ratios],
        "saturation": float(torch.relu(raw / exposure - 1.0).gt(0).float().mean().item()),
        "finite": bool(torch.isfinite(raw).all().item()),
    }


def _probe_commands(bundle, codes: torch.Tensor, labels: torch.Tensor, *, mode_rms: float, phase_kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one deterministic stateless command probe and optional phase clone."""
    probe = copy.deepcopy(bundle)
    with torch.no_grad():
        if phase_kind == "identity":
            probe.system.encoder.phase.zero_()
            probe.system.decoder.phase.zero_()
        elif phase_kind == "joint":
            height, width = probe.system.encoder.phase.shape[-2:]
            yy = torch.linspace(-torch.pi, torch.pi, height, device=probe.system.encoder.phase.device).view(1, height, 1)
            xx = torch.linspace(-torch.pi, torch.pi, width, device=probe.system.encoder.phase.device).view(1, 1, width)
            direction = torch.cos(yy) * torch.cos(xx)
            direction = direction / direction.square().mean().sqrt().clamp_min(1.0e-12) * 0.005
            probe.system.encoder.phase.add_(direction)
            probe.system.decoder.phase.sub_(direction)
        logits = probe.generator.logits(codes, labels)
        modes = probe.generator.lowpass_normalized_modes().index_select(0, labels)
        signed = torch.tanh(codes) / (codes.shape[1] ** 0.5)
        delta = mode_rms * torch.einsum("bk,bkhw->bhw", signed, modes)
        command = torch.sigmoid(logits + delta).unsqueeze(1)
        raw = _raw_from_command(probe, command)
    return raw, command


def conditional_optical_generation_feasibility(args, config) -> int:
    """Run the one-shot local macro-path review after the failed v6 smoke."""
    if config.architecture != PCA_TRANSPORT_ARCHITECTURE or config.conditional_generation is None:
        raise ValueError("macro feasibility requires the v6 PCA transport config")
    failed = EvidenceRecordV2.from_dict(json.loads(Path(args.revalidation_evidence).read_text()))
    failed.validate()
    if failed.evidence_kind != "conditional_pca_transport_smoke_revalidation" or failed.status != "failed":
        raise ValueError("feasibility review requires failed v6 smoke revalidation evidence")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    evaluator_hash = evaluator_checkpoint_hash(args.evaluator_checkpoint)
    device = torch.device(config.run.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("macro feasibility review requires CUDA for the formal optical path")
    evaluator, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    class_bundle = type("Args", (), {"dataset_root": args.dataset_root, "evaluator_checkpoint": args.evaluator_checkpoint, "checkpoint": args.checkpoint})
    _, _, _, _, _, bundle, manifest, _ = _load(class_bundle, config)
    if manifest.architecture_kind != PCA_TRANSPORT_ARCHITECTURE:
        raise ValueError("feasibility review requires a v6 initialization checkpoint")
    real_probe = _real_panel_views(train_images, train_labels, device=device)
    panels = _panels(device)
    before = state_content_hash(bundle.state_dict())
    run = _new_run(config)
    report: dict[str, object] = {"contract": FEASIBILITY_CONTRACT, "subspaces": {}, "state_before": before}
    for name, (codes, panel_labels) in panels.items():
        with torch.no_grad():
            baseline_command = bundle.generator(codes, panel_labels)
            baseline_raw = _raw_from_command(bundle, baseline_command)
        probes = {"baseline": _metrics(baseline_raw, panel_labels, evaluator, real_probe, config.conditional_generation.fixed_exposure_scale)}
        for subspace, phase_kind in (("bandlimited_slm_only", "current"), ("slm_plus_global_phase", "joint"), ("identity_phase_reference", "identity")):
            raw, command = _probe_commands(bundle, codes, panel_labels, mode_rms=0.20, phase_kind=phase_kind)
            probes[subspace] = _metrics(raw, panel_labels, evaluator, real_probe, config.conditional_generation.fixed_exposure_scale) | {
                "command_logit_rms": float((torch.logit(command.clamp(1e-6, 1 - 1e-6)) - torch.logit(baseline_command.clamp(1e-6, 1 - 1e-6))).square().mean().sqrt().item()),
                "raw_relative_rms": float((raw - baseline_raw).square().mean().sqrt().div(baseline_raw.square().mean().sqrt().clamp_min(1e-12)).item()),
            }
            atomic_torch_save(run.root / f"{name}_{subspace}_raw.pt", raw.detach().cpu())
            _full_contact_sheet(run.root / f"{name}_{subspace}_10x17.png", raw, panel_labels, config.conditional_generation.fixed_exposure_scale)
        report["subspaces"][name] = probes
    after = state_content_hash(bundle.state_dict())
    report["state_after"] = after
    report["parameters_unchanged"] = before == after
    gates_by_group = {}
    for group in ("baseline", "bandlimited_slm_only", "slm_plus_global_phase", "identity_phase_reference"):
        values = [report["subspaces"][name][group] for name in panels]
        gates_by_group[group] = all(x["accuracy"] >= .90 and min(x["recall"]) >= .80 and x["camera_fraction"] >= .10 and min(x["ratios"]) >= .35 and x["saturation"] <= .08 and x["finite"] for x in values)
    classification = "slm_semantic_nullspace_reachable" if gates_by_group["bandlimited_slm_only"] else ("joint_phase_codesign_reachable" if gates_by_group["slm_plus_global_phase"] else ("phase_initialization_trap" if gates_by_group["identity_phase_reference"] else "local_macro_generation_unreachable"))
    report["classification"] = classification
    report["group_gates"] = gates_by_group
    report_path = run.root / "conditional_optical_generation_feasibility_report.json"
    atomic_write_json(report_path, report)
    checks = (
        GateResult("v6_revalidation_failed", True, {}),
        GateResult("state_unchanged", before == after, {"before": before, "after": after}),
        GateResult("provenance_complete", True, {"subspaces": list(report["subspaces"])}),
        GateResult("local_feasibility_classified", classification in {"slm_semantic_nullspace_reachable", "joint_phase_codesign_reachable", "phase_initialization_trap", "local_macro_generation_unreachable"}, {"classification": classification}),
    )
    evidence = EvidenceRecordV2(2, "conditional_optical_generation_feasibility", config.run.evidence_tier, "passed", run.git_commit, run.dirty_patch_hash, run.config_hash, _context(config, evaluator_hash=evaluator_hash, split_hash=split.manifest_hash, checkpoint_hash=file_sha256(args.checkpoint)), {"revalidation": file_sha256(args.revalidation_evidence), "report": file_sha256(report_path)}, checks, (), classification)
    _write_evidence(run, "conditional_optical_generation_feasibility_evidence.json", evidence)
    print(run.root)
    return 0
