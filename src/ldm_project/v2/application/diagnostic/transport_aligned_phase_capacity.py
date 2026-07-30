"""Isolated v7.2 4+4 phase-capacity design gate.

The v7.1 failure is an immutable input fact, never an authorization token.  A
failed design deliberately has no artifact and therefore cannot reach the
admission, tangent, or smoke commands.
"""

from __future__ import annotations

import json

import torch

from ...evidence import (
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from ...student.checkpoints import state_content_hash
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
    _calibrate_scales,
    _device,
    _feature_targets,
    _fit_transport_mixing,
    _memory,
    _relay_provenance,
)


DESIGN_CONTRACT = canonical_sha256(
    {
        "kind": "conditional_transport_aligned_phase_capacity_design_v7_2",
        "architecture": TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "mechanical_distance_m": 0.798,
        "epsilon_logit": 0.01,
        "nonlinearity_max": 0.10,
        "components": 32,
        "directions": 8,
        "scale_bisection_steps": 20,
        "optimizer": False,
        "checkpoint": False,
    }
)


def _require(config) -> None:
    if (
        config.architecture != TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE
        or config.conditional_generation is None
    ):
        raise ValueError(
            "v7.2 requires conditional_transport_aligned_phase_capacity_v7_2"
        )
    if (config.system.encoder_layers, config.system.decoder_layers) != (4, 4):
        raise ValueError("v7.2 design requires the locked 4+4 system")


def conditional_transport_aligned_phase_capacity_design(args, config) -> int:
    _require(config)
    v71 = EvidenceRecordV2.from_dict(
        json.loads(open(args.v71_failure_evidence, encoding="utf-8").read())
    )
    v71.validate()
    if (
        v71.evidence_kind != "conditional_transport_aligned_manifold_design"
        or v71.status != "failed"
        or v71.permitted_next_actions
    ):
        raise ValueError(
            "v7.2 requires the immutable, non-authorizing v7.1 failed-design evidence"
        )
    source = _source(args.electronic_initialization)
    v7 = EvidenceRecordV2.from_dict(
        json.loads(open(args.v7_capacity_evidence, encoding="utf-8").read())
    )
    v7.validate()
    if (
        v7.evidence_kind != "conditional_nonlinear_expanded_manifold_capacity"
        or v7.status != "passed"
    ):
        raise ValueError("v7 capacity evidence must be passed")
    v7_artifact = torch.load(
        args.v7_basis_artifact, map_location="cpu", weights_only=False
    )
    if not isinstance(v7_artifact, dict) or v7.input_hashes.get(
        "basis_artifact"
    ) != file_sha256(args.v7_basis_artifact):
        raise ValueError("v7 basis artifact is not content-bound")
    if (
        v7.context.checkpoint_hash != file_sha256(args.electronic_initialization)
        or v7_artifact.get("source_basis_hash") != source["basis_hash"]
        or v7_artifact.get("source_electronic_hash") != source["source_checkpoint_hash"]
    ):
        raise ValueError(
            "v7 source does not match the frozen electronic initialization"
        )
    images, labels = load_local_mnist(args.dataset_root, train=True)
    train_images, train_labels, _, _, split = _split(images, labels, args.dataset_root)
    regenerated, _ = _build_artifact(train_images, train_labels, source, config)
    if tensor_sha256(regenerated["basis"]) != tensor_sha256(v7_artifact["basis"]):
        raise ValueError("train-only v7 candidate pool regeneration mismatch")
    device = _device(config)
    torch.cuda.reset_peak_memory_stats(device)
    evaluator, evaluator_payload = load_frozen_evaluator(
        args.evaluator_checkpoint, device=device
    )
    if (
        MNISTSplitManifest(**evaluator_payload["split_manifest"]).manifest_hash
        != split.manifest_hash
    ):
        raise ValueError("frozen evaluator split mismatch")
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
    mixing, fit = _fit_transport_mixing(
        bundle,
        evaluator,
        source["carrier"].to(device),
        regenerated["basis"].to(device),
        _feature_targets(evaluator, train_images, train_labels, device),
        exposure,
    )
    scales, calibration = _calibrate_scales(
        bundle,
        evaluator,
        source["carrier"].to(device),
        regenerated["basis"].to(device),
        mixing,
        exposure,
    )
    provenance, memory = _relay_provenance(bundle, device), _memory(device)
    expected = _context(
        config,
        evaluator_hash=evaluator_checkpoint_hash(args.evaluator_checkpoint),
        split_hash=split.manifest_hash,
        checkpoint_hash=file_sha256(args.electronic_initialization),
    )
    zero_labels = torch.arange(10, device=device)
    zero_equal = bool(
        torch.equal(
            bundle.generator(torch.zeros(10, 8, device=device), zero_labels),
            bundle.generator.class_carrier_command(zero_labels),
        )
    )
    after = state_content_hash(bundle.state_dict())
    gates = (
        GateResult(
            "v7_1_failure_input_fact",
            True,
            {
                "evidence_hash": file_sha256(args.v71_failure_evidence),
                "permitted_next_actions": [],
            },
        ),
        GateResult(
            "v7_capacity_input",
            True,
            {"evidence_hash": file_sha256(args.v7_capacity_evidence)},
        ),
        GateResult(
            "response_rank_8",
            all(
                item["rank"]["rank"] == 8 and item["rank"]["sigma8_over_sigma1"] >= 0.05
                for item in fit.values()
            ),
            fit,
        ),
        GateResult(
            "central_difference_linearity",
            all(item["nonlinearity"] <= 0.10 for item in fit.values()),
            fit,
        ),
        GateResult(
            "positive_negative_scale",
            bool(((scales >= 0.05) & (scales <= 0.50)).all()),
            {"steps": 20, "calibration": calibration},
        ),
        GateResult("zero_carrier", zero_equal, {}),
        GateResult(
            "relay_provenance",
            bool(
                provenance["identity_ols"]
                and provenance["near_conjugate_count"] == 8
                and provenance["operations"].count("finite_crop_hard_aperture") == 8
                and any(
                    row.get("intensity_transmission") == 0.80
                    and row.get("per_sample") is False
                    for row in provenance["attenuation"]
                )
            ),
            provenance,
        ),
        GateResult(
            "execution_integrity",
            before == after and bool(memory["passed"]),
            {"state_unchanged": before == after, **memory},
        ),
        GateResult(
            "no_optimizer_checkpoint",
            True,
            {"optimizer_created": False, "checkpoint_created": False},
        ),
    )
    passed = all(gate.passed for gate in gates)
    run = _new_run(config)
    report = (
        run.root / "conditional_transport_aligned_phase_capacity_design_report.json"
    )
    atomic_write_json(
        report,
        {
            "contract": DESIGN_CONTRACT,
            "fit": fit,
            "calibration": calibration,
            "provenance": provenance,
            "memory": memory,
            "state_hash_before": before,
            "state_hash_after": after,
            "artifact_published": passed,
        },
    )
    inputs = {
        "v7_1_failure_evidence": file_sha256(args.v71_failure_evidence),
        "electronic_initialization": file_sha256(args.electronic_initialization),
        "v7_capacity_evidence": file_sha256(args.v7_capacity_evidence),
        "v7_basis_artifact": file_sha256(args.v7_basis_artifact),
        "report": file_sha256(report),
    }
    if passed:
        artifact = {
            "schema_version": 1,
            "contract": DESIGN_CONTRACT,
            "architecture": TRANSPORT_ALIGNED_PHASE_CAPACITY_ARCHITECTURE,
            "carrier": source["carrier"].float(),
            "candidate_modes": regenerated["basis"].float(),
            "transport_mixing": mixing.cpu(),
            "manifold_scales": scales.cpu(),
            "carrier_hash": tensor_sha256(source["carrier"].float()),
            "candidate_basis_hash": tensor_sha256(regenerated["basis"]),
            "mixing_hash": tensor_sha256(mixing.cpu()),
            "scale_hash": tensor_sha256(scales.cpu()),
            "v7_1_failure_evidence_hash": file_sha256(args.v71_failure_evidence),
            "v7_capacity_evidence_hash": file_sha256(args.v7_capacity_evidence),
            "v7_basis_artifact_hash": file_sha256(args.v7_basis_artifact),
        }
        artifact_path = (
            run.root / "conditional_transport_aligned_phase_capacity_design.pt"
        )
        atomic_torch_save(artifact_path, artifact)
        inputs["artifact"] = file_sha256(artifact_path)
    record = EvidenceRecordV2(
        2,
        "conditional_transport_aligned_phase_capacity_design",
        config.run.evidence_tier,
        "passed" if passed else "failed",
        run.git_commit,
        run.dirty_patch_hash,
        run.config_hash,
        expected,
        inputs,
        gates,
        ("conditional-transport-aligned-phase-capacity-admission",) if passed else (),
        None if passed else "phase_capacity_design_gate_failed",
    )
    _write_evidence(
        run, "conditional_transport_aligned_phase_capacity_design_evidence.json", record
    )
    print(run.root)
    return 0 if passed else 1


def _unreachable(stage: str):
    def runner(args, config) -> int:
        _require(config)
        raise RuntimeError(
            f"v7.2 {stage} requires a passed v7.2 design artifact; no execution is permitted before that gate"
        )

    return runner


conditional_transport_aligned_phase_capacity_admission = _unreachable("admission")
conditional_transport_aligned_phase_capacity_tangent = _unreachable("tangent")
transport_aligned_phase_capacity_smoke = _unreachable("smoke")
