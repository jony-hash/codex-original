from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.application.current import dataset as current_dataset
from ldm_project.v2.application.current import evaluation as current_evaluation
from ldm_project.v2.application.current import training as current_training
from ldm_project.v2.application.current import runtime as current_runtime
from ldm_project.v2.application.current.runtime import (
    CONDITIONAL_CONTROL_ADMISSION_CONTRACT,
    BALANCED_TRAINING_SCHEDULE_CONTRACT,
    IDENTITY_ARCHITECTURE,
    _context,
    balanced_schedule_steps,
    _system_signature,
    _validate_balanced_training_payload,
    _validate_current_evidence_source,
    require_clean_acceptance_worktree,
)
from ldm_project.v2.config import load_config
from ldm_project.v2.evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    canonical_sha256,
    file_sha256,
)
from ldm_project.v2.student.checkpoints import (
    state_content_hash,
)


CONFIG_PATH = "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
CHECKPOINT_HASH = "c" * 64
INITIALIZATION_HASH = "i" * 64
STATE_HASH = "s" * 64
EVALUATOR_HASH = "e" * 64
SPLIT_HASH = "p" * 64


def _config():
    return load_config(CONFIG_PATH)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_git_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "unit@example.invalid")
    _git(tmp_path, "config", "user.name", "Unit Test")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-qm", "initial")
    return tracked


def test_acceptance_preflight_rejects_dirty_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    monkeypatch.setattr(
        current_runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M src/changed.py\n"),
    )
    with pytest.raises(ValueError, match="clean git commit"):
        require_clean_acceptance_worktree(config)


def _record(
    config,
    kind: str,
    context,
    inputs: dict[str, str],
    actions: tuple[str, ...],
    *,
    status: str = "passed",
) -> EvidenceRecordV2:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return EvidenceRecordV2(
        2,
        kind,
        "acceptance",
        status,
        commit,
        None,
        context.config_hash,
        context,
        inputs,
        (GateResult("test_gate", True, {}),),
        actions,
        None,
    )


def _write_record(path: Path, record: EvidenceRecordV2) -> None:
    path.write_text(json.dumps(record.to_dict()), encoding="utf-8")


def _manifest(config, *, state_hash: str = STATE_HASH):
    return SimpleNamespace(
        architecture_kind=IDENTITY_ARCHITECTURE,
        system_signature=_system_signature(config),
        config_hash=canonical_sha256(config.model_dump(mode="json")),
        initialization_seed=config.conditional_generation.initialization_seed,
        initialization_state_hash=state_hash,
        evaluator_checkpoint_hash=EVALUATOR_HASH,
        dataset_split_manifest_hash=SPLIT_HASH,
        training_schedule_hash=BALANCED_TRAINING_SCHEDULE_CONTRACT,
    )


def _lineage_context(config, *, checkpoint_hash: str = CHECKPOINT_HASH):
    return _context(
        config,
        evaluator_hash=EVALUATOR_HASH,
        split_hash=SPLIT_HASH,
        checkpoint_hash=checkpoint_hash,
    )


def test_current_evidence_source_accepts_clean_head() -> None:
    config = _config()
    context = _lineage_context(config)
    current = _record(config, "conditional_observability", context, {}, ("conditional_control_isolation",))
    _validate_current_evidence_source(current)


def test_current_evidence_source_allows_docs_and_tests_descendants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_git_repo(tmp_path)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    config = _config()
    monkeypatch.chdir(tmp_path)
    record = replace(_record(config, "conditional_observability", _lineage_context(config), {}, ("conditional_control_isolation",)), git_commit=base)
    for relative in ("docs/note.md", "tests/note.txt"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("follow-up\n", encoding="utf-8")
        _git(tmp_path, "add", relative)
        _git(tmp_path, "commit", "-qm", relative)
        _validate_current_evidence_source(record)


@pytest.mark.parametrize("relative", ["src/ldm_project/v2/runtime.py", "configs/v2/current/test.yaml", "pyproject.toml"])
def test_current_evidence_source_rejects_scoped_descendant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str) -> None:
    _init_git_repo(tmp_path)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    config = _config()
    monkeypatch.chdir(tmp_path)
    record = replace(_record(config, "conditional_observability", _lineage_context(config), {}, ("conditional_control_isolation",)), git_commit=base)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("changed\n", encoding="utf-8")
    _git(tmp_path, "add", relative)
    _git(tmp_path, "commit", "-qm", relative)
    with pytest.raises(ValueError, match="runtime tree changed"):
        _validate_current_evidence_source(record)


def test_current_evidence_source_rejects_non_ancestor_and_non_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_git_repo(tmp_path)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(tmp_path), "branch", "--show-current"], check=True, capture_output=True, text=True).stdout.strip()
    _git(tmp_path, "checkout", "-qb", "side")
    (tmp_path / "side.py").write_text("side\n", encoding="utf-8")
    _git(tmp_path, "add", "side.py")
    _git(tmp_path, "commit", "-qm", "side")
    side = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    _git(tmp_path, "checkout", "-q", branch)
    config = _config()
    monkeypatch.chdir(tmp_path)
    record = replace(_record(config, "conditional_observability", _lineage_context(config), {}, ("conditional_control_isolation",)), git_commit=side)
    with pytest.raises(ValueError, match="not an ancestor"):
        _validate_current_evidence_source(record)
    monkeypatch.chdir(tmp_path.parent)
    with pytest.raises(ValueError, match="unavailable"):
        _validate_current_evidence_source(replace(record, git_commit=base))


def test_current_evidence_source_rejects_dirty_provenance() -> None:
    config = _config()
    context = _lineage_context(config)
    record = _record(config, "conditional_observability", context, {}, ("conditional_control_isolation",))
    with pytest.raises(ValueError, match="current evidence"):
        _validate_current_evidence_source(replace(record, dirty_patch_hash="d" * 64))


def _stub_mnist(monkeypatch, module) -> None:
    images = torch.zeros(20, 1, 28, 28)
    labels = torch.arange(20, dtype=torch.long) % 10
    split = SimpleNamespace(manifest_hash=SPLIT_HASH)
    monkeypatch.setattr(module, "load_local_mnist", lambda *args, **kwargs: (images, labels))
    monkeypatch.setattr(
        module,
        "_split",
        lambda *args, **kwargs: (images, labels, images, labels, split),
    )
    monkeypatch.setattr(module, "evaluator_checkpoint_hash", lambda _: EVALUATOR_HASH)
    monkeypatch.setattr(module, "load_frozen_evaluator", lambda *args, **kwargs: (object(), None))


def test_balanced_schedule_contract_is_split_derived_and_stable() -> None:
    labels = torch.arange(500, dtype=torch.long) % 10
    assert balanced_schedule_steps(labels, batch_size=50, epochs=5) == (10, 50)
    changed = torch.cat([labels, torch.zeros(50, dtype=torch.long)])
    assert balanced_schedule_steps(changed, batch_size=50, epochs=5) == (10, 50)
    reduced = labels[labels != 9]
    with pytest.raises(ValueError, match="no batches"):
        balanced_schedule_steps(reduced, batch_size=50, epochs=5)


def test_conditional_evaluate_rejects_missing_independent_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    all_images = torch.zeros(600, 1, 28, 28)
    all_labels = torch.arange(600, dtype=torch.long) % 10
    train_images = all_images[:500]
    train_labels = all_labels[:500]
    validation_images = all_images[500:]
    validation_labels = all_labels[500:]
    split = SimpleNamespace(manifest_hash=SPLIT_HASH)
    monkeypatch.setattr(
        current_evaluation,
        "load_local_mnist",
        lambda *args, **kwargs: (all_images, all_labels),
    )
    monkeypatch.setattr(
        current_evaluation,
        "_split",
        lambda *args, **kwargs: (
            train_images,
            train_labels,
            validation_images,
            validation_labels,
            split,
        ),
    )
    monkeypatch.setattr(current_evaluation, "evaluator_checkpoint_hash", lambda _: EVALUATOR_HASH)
    monkeypatch.setattr(current_evaluation, "load_frozen_evaluator", lambda *args, **kwargs: (object(), None))
    context = _lineage_context(config)
    record = _record(
        config,
        "conditional_training",
        context,
        {
            "checkpoint": CHECKPOINT_HASH,
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": STATE_HASH,
            "schedule_contract": BALANCED_TRAINING_SCHEDULE_CONTRACT,
            "schedule_steps_per_epoch": "10",
            "schedule_total_steps": "50",
        },
        ("evaluate",),
    )
    evidence_path = tmp_path / "training.json"
    _write_record(evidence_path, record)
    monkeypatch.setattr(
        current_evaluation,
        "_load_bundle_with_payload",
        lambda *args, **kwargs: (
            SimpleNamespace(system=object(), generator=object()),
            _manifest(config),
            {
                "extra": {
                    "history": [{"epoch": epoch} for epoch in range(1, 6) for _ in range(10)],
                    "schedule_steps_per_epoch": 10,
                    "schedule_total_steps": 50,
                }
            },
        ),
    )
    monkeypatch.setattr(current_evaluation, "_validate_conditional_lineage", lambda *args, **kwargs: None)
    args = SimpleNamespace(
        training_evidence=str(evidence_path),
        dataset_root=str(tmp_path),
        evaluator_checkpoint=str(tmp_path / "evaluator.pt"),
        checkpoint=str(tmp_path / "checkpoint.pt"),
        acceptance_evaluator_checkpoint=None,
        acceptance_measurement_bundle=None,
        acceptance_measurement_evidence=None,
    )
    with pytest.raises(ValueError, match="independent acceptance measurement"):
        current_evaluation.conditional_evaluate(args, config)


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"schedule_steps_per_epoch": 984, "schedule_total_steps": 5500}, id="old_total"),
        pytest.param({"schedule_steps_per_epoch": 5500, "schedule_total_steps": 4920}, id="old_epoch"),
        pytest.param({"schedule_steps_per_epoch": 984, "schedule_total_steps": 4920, "history": []}, id="missing_history"),
        pytest.param(
            {
                "schedule_steps_per_epoch": 984,
                "schedule_total_steps": 4920,
                "history": [{"epoch": 1}] * 4920,
            },
            id="wrong_epoch_counts",
        ),
    ],
)
def test_balanced_schedule_payload_rejects_legacy_or_malformed_history(extra) -> None:
    with pytest.raises(ValueError, match="balanced schedule|step count|epoch"):
        _validate_balanced_training_payload(
            {"extra": extra}, steps_per_epoch=984, total_steps=4920
        )


def test_balanced_schedule_payload_accepts_complete_epoch_history() -> None:
    history = [{"epoch": epoch} for epoch in range(1, 6) for _ in range(984)]
    _validate_balanced_training_payload(
        {
            "extra": {
                "schedule_steps_per_epoch": 984,
                "schedule_total_steps": 4920,
                "history": history,
            }
        },
        steps_per_epoch=984,
        total_steps=4920,
    )


def test_conditional_train_consumes_injection_action_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config = config.model_copy(update={"run": config.run.model_copy(update={"evidence_tier": "diagnostic"})})
    _stub_mnist(monkeypatch, current_training)
    context = _lineage_context(config, checkpoint_hash=INITIALIZATION_HASH)
    injection = _record(
        config,
        "conditional_injection_admission",
        context,
        {
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": STATE_HASH,
        },
        (),
    )
    observability = _record(
        config,
        "conditional_observability",
        context,
        {},
        ("conditional_control_isolation",),
    )
    recovery = _record(
        config,
        "conditional_semantic_recovery_smoke",
        context,
        {},
        ("train",),
    )
    injection_path = tmp_path / "injection.json"
    observability_path = tmp_path / "observability.json"
    recovery_path = tmp_path / "recovery.json"
    for path, record in (
        (injection_path, injection),
        (observability_path, observability),
        (recovery_path, recovery),
    ):
        _write_record(path, record)
    args = SimpleNamespace(
        dataset_root=str(tmp_path),
        evaluator_checkpoint=str(tmp_path / "evaluator.pt"),
        injection_evidence=str(injection_path),
        observability_evidence=str(observability_path),
        semantic_recovery_evidence=str(recovery_path),
        controllability_evidence=None,
        initialization_checkpoint=str(tmp_path / "initialization.pt"),
    )
    monkeypatch.setattr(
        current_training,
        "_load_bundle",
        lambda *args, **kwargs: pytest.fail("checkpoint loaded before action rejection"),
    )
    with pytest.raises(ValueError, match="conditional_observability"):
        current_training.conditional_train(args, config)


def test_conditional_train_accepts_controllability_authorization_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config = config.model_copy(update={"run": config.run.model_copy(update={"evidence_tier": "diagnostic"})})
    _stub_mnist(monkeypatch, current_training)
    context = _lineage_context(config, checkpoint_hash=INITIALIZATION_HASH)
    injection = _record(
        config,
        "conditional_injection_admission",
        context,
        {
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": STATE_HASH,
        },
        ("conditional_observability",),
    )
    observability = _record(
        config,
        "conditional_observability",
        context,
        {},
        ("conditional_control_isolation",),
    )
    control = _record(
        config,
        "controllability_smoke",
        context,
        {
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": STATE_HASH,
            "metric_contract": canonical_sha256(CONDITIONAL_CONTROL_ADMISSION_CONTRACT),
            "first_100_schedule_hash": "unused-in-stub",
        },
        ("train",),
    )
    paths = {
        name: tmp_path / f"{name}.json"
        for name in ("injection", "observability", "control")
    }
    for name, record in (("injection", injection), ("observability", observability), ("control", control)):
        _write_record(paths[name], record)
    args = SimpleNamespace(
        dataset_root=str(tmp_path),
        evaluator_checkpoint=str(tmp_path / "evaluator.pt"),
        injection_evidence=str(paths["injection"]),
        observability_evidence=str(paths["observability"]),
        controllability_evidence=str(paths["control"]),
        semantic_recovery_evidence=None,
        initialization_checkpoint=str(tmp_path / "initialization.pt"),
    )
    monkeypatch.setattr(
        current_training,
        "_first_hundred_schedule_hash",
        lambda labels: "unused-in-stub",
    )
    monkeypatch.setattr(
        current_training,
        "_load_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("authorization unexpectedly loaded checkpoint")
        ),
    )
    with pytest.raises(RuntimeError, match="authorization unexpectedly"):
        current_training.conditional_train(args, config)


def test_conditional_train_rejects_both_training_authorizations_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config = config.model_copy(update={"run": config.run.model_copy(update={"evidence_tier": "diagnostic"})})
    _stub_mnist(monkeypatch, current_training)
    args = SimpleNamespace(
        dataset_root=str(tmp_path),
        evaluator_checkpoint=str(tmp_path / "evaluator.pt"),
        injection_evidence=str(tmp_path / "missing-injection.json"),
        observability_evidence=str(tmp_path / "missing-observability.json"),
        controllability_evidence=str(tmp_path / "control.json"),
        semantic_recovery_evidence=str(tmp_path / "recovery.json"),
        initialization_checkpoint=str(tmp_path / "initialization.pt"),
    )
    with pytest.raises(ValueError, match="exactly one"):
        current_training.conditional_train(args, config)


def test_conditional_evaluate_requires_action_and_rejects_state_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    _stub_mnist(monkeypatch, current_evaluation)
    context = _lineage_context(config)
    record = _record(
        config,
        "conditional_training",
        context,
        {
            "checkpoint": CHECKPOINT_HASH,
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": STATE_HASH,
        },
        (),
    )
    evidence_path = tmp_path / "training.json"
    _write_record(evidence_path, record)
    args = SimpleNamespace(
        training_evidence=str(evidence_path),
        dataset_root=str(tmp_path),
        evaluator_checkpoint=str(tmp_path / "evaluator.pt"),
        checkpoint=str(tmp_path / "checkpoint.pt"),
    )
    monkeypatch.setattr(
        current_evaluation,
        "_load_bundle",
        lambda *args, **kwargs: pytest.fail("checkpoint loaded before action rejection"),
    )
    with pytest.raises(ValueError, match="evaluate"):
        current_evaluation.conditional_evaluate(args, config)

    valid = _record(
        config,
        "conditional_training",
        context,
        {
            "checkpoint": CHECKPOINT_HASH,
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": "x" * 64,
            "schedule_contract": BALANCED_TRAINING_SCHEDULE_CONTRACT,
            "schedule_steps_per_epoch": "984",
            "schedule_total_steps": "4920",
        },
        ("evaluate",),
    )
    _write_record(evidence_path, valid)
    monkeypatch.setattr(
        current_evaluation,
        "_load_bundle_with_payload",
        lambda *args, **kwargs: (
            object(),
            _manifest(config),
            {"extra": {"history": [{"epoch": 1}] * 5, "schedule_steps_per_epoch": 1, "schedule_total_steps": 5}},
        ),
    )
    with pytest.raises(ValueError, match="initialization_state_hash"):
        current_evaluation.conditional_evaluate(args, config)


def test_export_rejects_missing_export_action_before_loading_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    context = _lineage_context(config)
    record = _record(
        config,
        "conditional_generation_admission",
        context,
        {
            "checkpoint": CHECKPOINT_HASH,
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": STATE_HASH,
        },
        (),
    )
    evidence_path = tmp_path / "admission.json"
    _write_record(evidence_path, record)
    args = SimpleNamespace(
        acceptance_evidence=str(evidence_path),
        checkpoint=str(tmp_path / "checkpoint.pt"),
        samples_per_label=None,
    )
    monkeypatch.setattr(
        current_dataset,
        "_load_bundle",
        lambda *args, **kwargs: pytest.fail("checkpoint loaded before export rejection"),
    )
    with pytest.raises(ValueError, match="export"):
        current_dataset.export_conditional_dataset(args, config)


def test_manual_admission_rejects_context_and_artifact_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    context = _lineage_context(config)
    record = _record(
        config,
        "conditional_acceptance",
        context,
        {
            "checkpoint": CHECKPOINT_HASH,
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": STATE_HASH,
            "raw_cmos": "r" * 64,
            "labels": "l" * 64,
            "review_preview": "v" * 64,
        },
        ("admit_conditional_evaluation",),
        status="pending_review",
    )
    evidence_path = tmp_path / "evaluation.json"
    _write_record(evidence_path, record)
    args = SimpleNamespace(
        evaluation_evidence=str(evidence_path),
        evaluation_raw_cmos=str(tmp_path / "raw.pt"),
        evaluation_labels=str(tmp_path / "labels.pt"),
        evaluation_preview=str(tmp_path / "preview.png"),
        manual_review=str(tmp_path / "review.json"),
    )
    monkeypatch.setattr(
        current_evaluation,
        "safe_torch_load",
        lambda *args, **kwargs: pytest.fail("artifacts loaded after context rejection"),
    )
    tampered_context = EvidenceContextV2(
        _system_signature(config),
        "x" * 64,
        checkpoint_hash=CHECKPOINT_HASH,
        evaluator_checkpoint_hash=EVALUATOR_HASH,
        dataset_split_manifest_hash=SPLIT_HASH,
    )
    tampered = _record(
        config,
        "conditional_acceptance",
        tampered_context,
        record.input_hashes,
        ("admit_conditional_evaluation",),
        status="pending_review",
    )
    _write_record(evidence_path, tampered)
    with pytest.raises(ValueError, match="context mismatch"):
        current_evaluation.admit_conditional_evaluation(args, config)


def _candidate_fixture(tmp_path: Path, config):
    root = tmp_path / "candidate"
    root.mkdir()
    shard = root / "part_000.pt"
    torch.save(
        {
            "labels": torch.zeros(1, dtype=torch.long),
            "mnist_view_uint8": torch.zeros(1, 1, 28, 28, dtype=torch.uint8),
        },
        shard,
    )
    shard_hash = file_sha256(shard)
    payload = {
        "schema_version": 1,
        "status": "candidate",
        "checkpoint_hash": CHECKPOINT_HASH,
        "evaluator_checkpoint_hash": EVALUATOR_HASH,
        "dataset_split_manifest_hash": SPLIT_HASH,
        "initialization_checkpoint_hash": INITIALIZATION_HASH,
        "initialization_state_hash": STATE_HASH,
        "shards": [{"path": shard.name, "sha256": shard_hash, "count": 1}],
    }
    payload["manifest_hash"] = canonical_sha256(payload)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    context = _context(
        config,
        evaluator_hash=EVALUATOR_HASH,
        split_hash=SPLIT_HASH,
        checkpoint_hash=CHECKPOINT_HASH,
        candidate_manifest_hash=canonical_sha256(payload),
    )
    return root, payload, context


def test_admit_dataset_rejects_generation_evaluator_split_mismatch_before_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    root, payload, candidate_context = _candidate_fixture(tmp_path, config)
    admission_context = _context(
        config,
        evaluator_hash="z" * 64,
        split_hash=SPLIT_HASH,
        checkpoint_hash=CHECKPOINT_HASH,
    )
    admission = _record(
        config,
        "conditional_generation_admission",
        admission_context,
        {
            "checkpoint": CHECKPOINT_HASH,
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": STATE_HASH,
        },
        ("export",),
    )
    memorization = _record(
        config,
        "conditional_memorization_audit",
        candidate_context,
        {},
        ("admit_dataset",),
    )
    utility = _record(
        config,
        "conditional_utility",
        candidate_context,
        {},
        ("admit_dataset",),
    )
    admission_path = tmp_path / "admission.json"
    memorization_path = tmp_path / "memorization.json"
    utility_path = tmp_path / "utility.json"
    _write_record(admission_path, admission)
    _write_record(memorization_path, memorization)
    _write_record(utility_path, utility)
    args = SimpleNamespace(
        generation_admission_evidence=str(admission_path),
        memorization_evidence=str(memorization_path),
        utility_evidence=str(utility_path),
        candidate_dataset=str(root),
    )
    monkeypatch.setattr(
        current_dataset,
        "_new_run",
        lambda *args, **kwargs: pytest.fail("accepted manifest created after mismatch"),
    )
    with pytest.raises(ValueError, match="evidence context mismatch"):
        current_dataset.admit_dataset(args, config)


@pytest.mark.parametrize(
    ("action", "message"),
    [
        ("audit_memorization", "audit_memorization"),
        ("evaluate_utility", "evaluate_utility"),
    ],
)
def test_candidate_consumers_require_their_declared_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    message: str,
) -> None:
    config = _config()
    root, payload, context = _candidate_fixture(tmp_path, config)
    candidate = _record(
        config,
        "conditional_dataset_candidate",
        context,
        {},
        (),
    )
    candidate_path = tmp_path / "candidate.json"
    _write_record(candidate_path, candidate)
    args = SimpleNamespace(
        candidate_evidence=str(candidate_path),
        candidate_dataset=str(root),
        evaluator_checkpoint=str(tmp_path / "evaluator.pt"),
        dataset_root=str(tmp_path),
    )
    target = (
        current_dataset.audit_memorization
        if action == "audit_memorization"
        else current_dataset.evaluate_utility
    )
    monkeypatch.setattr(
        current_dataset,
        "_new_run",
        lambda *args, **kwargs: pytest.fail("report created after action rejection"),
    )
    with pytest.raises(ValueError, match=message):
        target(args, config)


def test_training_completion_binds_corrected_checkpoint_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_config = _config()
    config = base_config.model_copy(
        update={
            "run": base_config.run.model_copy(
                update={"device": "cpu", "output_root": str(tmp_path / "outputs")}
            )
        }
    )
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    from ldm_project.v2.evidence import file_sha256 as original_file_sha256

    source_hash = original_file_sha256(source)
    context = _context(
        config,
        evaluator_hash=EVALUATOR_HASH,
        split_hash=SPLIT_HASH,
        checkpoint_hash=source_hash,
    )
    training = _record(
        config,
        "conditional_training",
        context,
        {"checkpoint": source_hash},
        (),
    )
    training_path = tmp_path / "training.json"
    _write_record(training_path, training)
    images = torch.zeros(20, 1, 28, 28)
    labels = torch.arange(20, dtype=torch.long) % 10
    split = SimpleNamespace(manifest_hash=SPLIT_HASH)
    monkeypatch.setattr(current_training, "load_local_mnist", lambda *args, **kwargs: (images, labels))
    monkeypatch.setattr(
        current_training,
        "_split",
        lambda *args, **kwargs: (images, labels, images, labels, split),
    )
    monkeypatch.setattr(current_training, "evaluator_checkpoint_hash", lambda _: EVALUATOR_HASH)
    monkeypatch.setattr(current_training, "_balanced_epoch_steps", lambda *args: 1)
    bundle = torch.nn.Linear(2, 2)
    old_manifest = SimpleNamespace(
        system_signature=_system_signature(config),
        config_hash=canonical_sha256(config.model_dump(mode="json")),
        dataset_split_manifest_hash=SPLIT_HASH,
        evaluator_checkpoint_hash=EVALUATOR_HASH,
        exposure_scale=config.conditional_generation.fixed_exposure_scale,
        state_content_hash=state_content_hash(bundle.state_dict()),
        initialization_seed=config.conditional_generation.initialization_seed,
        initialization_state_hash=STATE_HASH,
    )
    history = [{"total": 1.0} for _ in range(config.conditional_generation.epochs)]
    monkeypatch.setattr(
        current_training,
        "_load_bundle_with_payload",
        lambda *args, **kwargs: (bundle, old_manifest, {"extra": {"history": history}}),
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(
        current_training,
        "_new_run",
        lambda *args, **kwargs: SimpleNamespace(
            root=run_root,
            git_commit=subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            dirty_patch_hash=None,
            config_hash=canonical_sha256(config.model_dump(mode="json")),
        ),
    )
    monkeypatch.setattr(
        current_training,
        "save_conditional_checkpoint",
        lambda path, *args, **kwargs: Path(path).write_bytes(b"corrected"),
    )
    captured: list[EvidenceRecordV2] = []
    monkeypatch.setattr(
        current_training,
        "_write_evidence",
        lambda run, name, record: captured.append(record),
    )
    args = SimpleNamespace(
        dataset_root=str(tmp_path),
        evaluator_checkpoint=str(tmp_path / "evaluator.pt"),
        training_evidence=str(training_path),
        checkpoint=str(source),
    )
    current_training.conditional_training_completion(args, config)
    assert len(captured) == 1
    assert captured[0].context.checkpoint_hash == captured[0].input_hashes["checkpoint"]
