"""No-update curvature attribution for the closed v7.2 transport contract."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ...evidence import (
    safe_torch_load,
    EvidenceRecordV2,
    GateResult,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from ...student.checkpoints import state_content_hash
from ...student.conditional import fixed_exposure_view
from ...student.evaluator import (
    MNISTSplitManifest,
    evaluator_checkpoint_hash,
    load_frozen_evaluator,
    load_local_mnist,
)
from .expanded_manifold import _build_artifact, _source
from .support import (
    ConditionalBundle,
    TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE,
    _context,
    _new_run,
    _split,
    _write_evidence,
)
from .transport_aligned_manifold import (
    _device,
    _feature_targets,
    _fit_transport_mixing,
    _memory,
    _relay_provenance,
)


CONTRACT = canonical_sha256(
    {
        "kind": "conditional_transport_curvature_attribution_v1",
        "architecture": TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE,
        "epsilon_logit": 0.01,
        "candidate_directions": 32,
        "operational_directions": 8,
        "optimizer": False,
        "checkpoint": False,
    }
)


def _require(config) -> None:
    if config.architecture != TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE:
        raise ValueError("curvature attribution requires the isolated v7.2 4+4 config")


def _failed_evidence(path: str, kind: str) -> EvidenceRecordV2:
    record = EvidenceRecordV2.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    record.validate()
    if (
        record.evidence_kind != kind
        or record.status != "failed"
        or record.permitted_next_actions
    ):
        raise ValueError(f"{kind} must be an immutable failed evidence with no actions")
    return record


def _ratio(
    plus: torch.Tensor, minus: torch.Tensor, zero: torch.Tensor
) -> dict[str, float]:
    numerator = (plus + minus - 2.0 * zero).norm().item()
    denominator = (plus - minus).norm().clamp_min(1.0e-12).item()
    return {
        "numerator": float(numerator),
        "denominator": float(denominator),
        "ratio": float(numerator / denominator),
    }


def _quartiles(rows: list[dict[str, float]]) -> dict[str, object]:
    ratios = torch.tensor([row["ratio"] for row in rows])
    index = int(ratios.argmax().item())
    return {
        "p50": float(torch.quantile(ratios, 0.50).item()),
        "p90": float(torch.quantile(ratios, 0.90).item()),
        "p95": float(torch.quantile(ratios, 0.95).item()),
        "max": float(ratios[index].item()),
        "argmax_direction": index,
        "argmax": rows[index],
    }


def _capture(
    bundle,
    evaluator,
    carrier: torch.Tensor,
    direction: torch.Tensor,
    label: int,
    epsilon: float,
    exposure: float,
) -> dict[str, dict[str, torch.Tensor]]:
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for name, signed in (("plus", epsilon), ("minus", -epsilon), ("zero", 0.0)):
        logits = carrier[label : label + 1] + signed * direction
        command = torch.sigmoid(logits).unsqueeze(1)
        forward = bundle.system.forward(
            bundle.system.command(command), return_optical_trace=True
        )
        trace = forward.optical_trace or ()
        ols = next(
            field.tensor
            for field in trace
            if field.provenance.operation == "identity_ols"
        )
        camera_field = trace[-1].tensor
        view = fixed_exposure_view(forward.raw_camera.tensor, exposure)
        features, _ = evaluator(view)
        outputs[name] = {
            "command": command,
            "ols_field": ols,
            "camera_field": camera_field,
            "raw_intensity": forward.raw_camera.tensor,
            "fixed_exposure": view,
            "evaluator_feature": features,
        }
    return outputs


def _measure(
    bundle, evaluator, carrier, directions, label: int, exposure: float
) -> dict[str, object]:
    stage_rows = {
        name: []
        for name in (
            "command",
            "ols_field",
            "camera_field",
            "raw_intensity",
            "fixed_exposure",
            "evaluator_feature",
        )
    }
    for direction in directions:
        captured = _capture(
            bundle, evaluator, carrier, direction.unsqueeze(0), label, 0.01, exposure
        )
        for stage, rows in stage_rows.items():
            rows.append(
                _ratio(
                    captured["plus"][stage],
                    captured["minus"][stage],
                    captured["zero"][stage],
                )
            )
    return {stage: _quartiles(rows) for stage, rows in stage_rows.items()}


def conditional_transport_curvature_attribution(args, config) -> int:
    _require(config)
    v71 = _failed_evidence(
        args.v71_failure_evidence, "conditional_transport_aligned_manifold_design"
    )
    v72 = _failed_evidence(
        args.v72_failure_evidence, "conditional_transport_aligned_phase_capacity_design"
    )
    source = _source(args.electronic_initialization)
    v7 = EvidenceRecordV2.from_dict(
        json.loads(Path(args.v7_capacity_evidence).read_text(encoding="utf-8"))
    )
    v7.validate()
    if (
        v7.evidence_kind != "conditional_nonlinear_expanded_manifold_capacity"
        or v7.status != "passed"
    ):
        raise ValueError("curvature attribution requires passed v7 capacity evidence")
    artifact = safe_torch_load(
        args.v7_basis_artifact
    )
    if not isinstance(artifact, dict) or v7.input_hashes.get(
        "basis_artifact"
    ) != file_sha256(args.v7_basis_artifact):
        raise ValueError("v7 basis artifact hash mismatch")
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    regenerated, _ = _build_artifact(train_images, train_labels, source, config)
    if tensor_sha256(regenerated["basis"]) != tensor_sha256(artifact["basis"]):
        raise ValueError("regenerated v7 basis does not match the bound artifact")
    device = _device(config)
    torch.cuda.reset_peak_memory_stats(device)
    evaluator, payload = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    if (
        MNISTSplitManifest(**payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("evaluator split mismatch")
    selector = torch.zeros(10, 32, 8)
    selector[:, torch.arange(8), torch.arange(8)] = 1.0
    bundle = (
        ConditionalBundle(
            config,
            carrier_logits=source["carrier"],
            class_modes=regenerated["basis"],
            manifold_scales=source["scales"],
            transport_mixing=selector,
        )
        .to(device)
        .eval()
    )
    before = state_content_hash(bundle.state_dict())
    exposure = config.conditional_generation.fixed_exposure_scale
    candidates = bundle.generator.normalized_modes(regenerated["basis"].to(device))
    mixing, _ = _fit_transport_mixing(
        bundle,
        evaluator,
        source["carrier"].to(device),
        regenerated["basis"].to(device),
        _feature_targets(evaluator, train_images, train_labels, device),
        exposure,
    )
    operational = bundle.generator.normalized_modes(
        torch.einsum(
            "ymk,ymhw->ykhw", mixing.to(device), regenerated["basis"].to(device)
        )
    )
    report = {
        "contract": CONTRACT,
        "candidate": {
            str(label): _measure(
                bundle,
                evaluator,
                source["carrier"].to(device),
                candidates[label],
                label,
                exposure,
            )
            for label in range(10)
        },
        "operational": {
            str(label): _measure(
                bundle,
                evaluator,
                source["carrier"].to(device),
                operational[label],
                label,
                exposure,
            )
            for label in range(10)
        },
    }
    after, memory, provenance = (
        state_content_hash(bundle.state_dict()),
        _memory(device),
        _relay_provenance(bundle, device),
    )
    run = _new_run(config)
    report_path = run.root / "conditional_transport_curvature_attribution_report.json"
    atomic_write_json(report_path, report)
    context = _context(
        config,
        evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint),
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.electronic_initialization),
    )
    gates = (
        GateResult(
            "v7_1_failure_input",
            True,
            {
                "hash": file_sha256(args.v71_failure_evidence),
                "evidence": v71.evidence_hash,
            },
        ),
        GateResult(
            "v7_2_failure_input",
            True,
            {
                "hash": file_sha256(args.v72_failure_evidence),
                "evidence": v72.evidence_hash,
            },
        ),
        GateResult(
            "state_and_memory",
            before == after and bool(memory["passed"]),
            {"state_unchanged": before == after, **memory},
        ),
        GateResult(
            "no_optimizer_checkpoint",
            True,
            {"optimizer_created": False, "checkpoint_created": False},
        ),
        GateResult(
            "relay_provenance",
            provenance["near_conjugate_count"] == 8 and provenance["identity_ols"],
            provenance,
        ),
    )
    record = EvidenceRecordV2(
        2,
        "conditional_transport_curvature_attribution",
        config.run.evidence_tier,
        "diagnostic",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        context,
        {
            "v71_failure": file_sha256(args.v71_failure_evidence),
            "v72_failure": file_sha256(args.v72_failure_evidence),
            "v7_capacity": file_sha256(args.v7_capacity_evidence),
            "v7_basis": file_sha256(args.v7_basis_artifact),
            "report": file_sha256(report_path),
        },
        gates,
        (),
        "attribution_only_no_downstream_authorization",
    )
    _write_evidence(
        run, "conditional_transport_curvature_attribution_evidence.json", record
    )
    print(run.root)
    return 0
