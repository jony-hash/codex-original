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
    TRAINING_AUTHORIZATION_EVIDENCE_HASH_KEY,
    TRAINING_AUTHORIZATION_KIND_KEY,
    training_authorization_binding,
    validate_training_authorization_binding,
    CURRENT_GATE_CONTRACTS,
    validate_current_gate_record,
    require_current_gate_action,
    require_clean_acceptance_worktree,
)
from ldm_project.v2.application.current.measurement import ACCEPTANCE_MEASUREMENT_CONTRACT
from ldm_project.v2.application.current.exposure import EXPOSURE_CONTRACT, require_exposure_predecessor
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


@pytest.mark.parametrize(
    "kind", ["controllability_smoke", "conditional_semantic_recovery_smoke"]
)
def test_training_authorization_binding_round_trips_for_formal_sources(kind: str) -> None:
    config = _config()
    source = _record(config, kind, _lineage_context(config), {}, ("train",))
    binding = training_authorization_binding(source)
    bound = replace(source, input_hashes=binding)
    validate_training_authorization_binding(bound, binding)
    assert binding[TRAINING_AUTHORIZATION_KIND_KEY] == kind
    assert len(binding[TRAINING_AUTHORIZATION_EVIDENCE_HASH_KEY]) == 64


@pytest.mark.parametrize(
    "input_hashes, extra",
    [
        ({}, {}),
        ({TRAINING_AUTHORIZATION_KIND_KEY: "unknown", TRAINING_AUTHORIZATION_EVIDENCE_HASH_KEY: "a" * 64}, {}),
        ({TRAINING_AUTHORIZATION_KIND_KEY: "controllability_smoke", TRAINING_AUTHORIZATION_EVIDENCE_HASH_KEY: "a" * 63}, {}),
    ],
)
def test_training_authorization_binding_rejects_invalid_or_missing_values(
    input_hashes: dict[str, str], extra: dict[str, str]
) -> None:
    config = _config()
    record = _record(config, "conditional_training", _lineage_context(config), input_hashes, ("evaluate",))
    with pytest.raises(ValueError, match="authorization"):
        validate_training_authorization_binding(record, extra)


def test_training_authorization_binding_rejects_checkpoint_mismatch() -> None:
    config = _config()
    source = _record(config, "controllability_smoke", _lineage_context(config), {}, ("train",))
    binding = training_authorization_binding(source)
    record = replace(source, input_hashes=binding)
    with pytest.raises(ValueError, match="disagrees"):
        validate_training_authorization_binding(
            record,
            {TRAINING_AUTHORIZATION_KIND_KEY: binding[TRAINING_AUTHORIZATION_KIND_KEY], TRAINING_AUTHORIZATION_EVIDENCE_HASH_KEY: "b" * 64},
        )


def test_current_gate_contract_registers_formal_transitions() -> None:
    assert {
        "evaluator_qualification",
        "acceptance_evaluator_training",
        "conditional_acceptance_measurement",
        "conditional_injection_admission",
        "conditional_observability",
        "conditional_control_isolation",
        "conditional_semantic_localization",
        "controllability_smoke",
        "conditional_semantic_recovery_smoke",
        "conditional_training",
        "conditional_acceptance",
        "conditional_generation_admission",
        "conditional_dataset_candidate",
        "conditional_memorization_audit",
        "conditional_utility",
        "conditional_dataset_accepted",
        "formal_evaluation_exposure",
    } <= set(CURRENT_GATE_CONTRACTS)


@pytest.mark.parametrize("status", ["failed", "diagnostic"])
def test_current_formal_gate_rejects_ambiguous_or_diagnostic_authorization(status: str) -> None:
    config = _config()
    record = _record(
        config,
        "controllability_smoke",
        _lineage_context(config),
        {},
        (),
        status=status,
    )
    with pytest.raises(ValueError):
        validate_current_gate_record(record)


def test_valid_failed_smoke_is_the_only_recovery_transition() -> None:
    config = _config()
    failed = _record(
        config,
        "controllability_smoke",
        _lineage_context(config),
        {},
        ("semantic_recovery_smoke",),
        status="valid_failed",
    )
    require_current_gate_action(
        failed,
        kind="controllability_smoke",
        action="semantic_recovery_smoke",
        expected_context=_lineage_context(config),
    )
    invalid = replace(failed, status="invalid_harness", permitted_next_actions=())
    with pytest.raises(ValueError):
        require_current_gate_action(
            invalid,
            kind="controllability_smoke",
            action="semantic_recovery_smoke",
            expected_context=_lineage_context(config),
        )


def test_exposure_predecessor_binds_chain_and_rejects_cross_measurement() -> None:
    config = _config()
    context = _context(config, evaluator_hash=EVALUATOR_HASH, split_hash=SPLIT_HASH)
    predecessor = _record(
        config,
        "formal_evaluation_exposure",
        context,
        {
            "exposure_chain_id": "h" * 64,
            "exposure_contract": EXPOSURE_CONTRACT,
            "exposure_sequence": "0",
            "primary_evaluator": EVALUATOR_HASH,
            "secondary_evaluator": "s" * 64,
            "test_manifest": "t" * 64,
            "measurement_evidence": "genesis",
            "formal_evaluation_id": "i" * 64,
        },
        ("record_next_official_test_exposure",),
    )
    assert require_exposure_predecessor(
        predecessor,
        config=config,
        primary_evaluator=EVALUATOR_HASH,
        secondary_evaluator="s" * 64,
        test_manifest="t" * 64,
        measurement_evidence="m" * 64,
        chain_id="h" * 64,
        split_hash=SPLIT_HASH,
    ) == 0
    with pytest.raises(ValueError, match="chain"):
        require_exposure_predecessor(
            predecessor,
            config=config,
            primary_evaluator=EVALUATOR_HASH,
            secondary_evaluator="s" * 64,
            test_manifest="t" * 64,
            measurement_evidence="m" * 64,
            chain_id="x" * 64,
            split_hash=SPLIT_HASH,
        )


def _measurement_fixture(tmp_path: Path, config, *, context=None, actions=("evaluate",)) -> Path:
    secondary = tmp_path / "secondary.pt"
    secondary.write_bytes(b"secondary")
    test_manifest = tmp_path / "mnist_test_manifest.json"
    test_manifest.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "measurement_bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "contract": ACCEPTANCE_MEASUREMENT_CONTRACT,
                "primary_checkpoint": EVALUATOR_HASH,
                "secondary_checkpoint": file_sha256(secondary),
                "test_manifest_filename": test_manifest.name,
                "test_manifest_hash": file_sha256(test_manifest),
            }
        ),
        encoding="utf-8",
    )
    record = _record(
        config,
        "conditional_acceptance_measurement",
        context or _context(config, evaluator_hash=EVALUATOR_HASH, split_hash=SPLIT_HASH),
        {
            "primary_evaluator": EVALUATOR_HASH,
            "secondary_evaluator": file_sha256(secondary),
            "measurement_bundle": file_sha256(bundle),
            "test_manifest": file_sha256(test_manifest),
            "exposure_genesis": "g" * 64,
            "exposure_chain_id": "h" * 64,
        },
        actions,
    )
    path = tmp_path / "measurement.json"
    _write_record(path, record)
    return path


def _exposure_fixture(tmp_path: Path, config, measurement_path: Path) -> Path:
    measurement = EvidenceRecordV2.from_dict(
        json.loads(measurement_path.read_text(encoding="utf-8"))
    )
    record = _record(
        config,
        "formal_evaluation_exposure",
        measurement.context,
        {
            "exposure_chain_id": "h" * 64,
            "exposure_sequence": "0",
            "exposure_contract": EXPOSURE_CONTRACT,
            "primary_evaluator": EVALUATOR_HASH,
            "secondary_evaluator": measurement.input_hashes["secondary_evaluator"],
            "test_manifest": measurement.input_hashes["test_manifest"],
            "measurement_evidence": "genesis",
            "formal_evaluation_id": "h" * 64,
        },
        ("record_next_official_test_exposure",),
    )
    path = tmp_path / "exposure.json"
    _write_record(path, record)
    measurement = replace(
        measurement,
        input_hashes={**measurement.input_hashes, "exposure_genesis": record.evidence_hash},
    )
    _write_record(measurement_path, measurement)
    return path


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


@pytest.mark.parametrize("relative", ["src/ldm_project/v2/application/current/runtime.py", "configs/v2/current/test.yaml", "pyproject.toml"])
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


def test_conditional_evaluate_rejects_measurement_without_evaluate_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    _stub_mnist(monkeypatch, current_evaluation)
    context = _lineage_context(config)
    training = _record(
        config,
        "conditional_training",
        context,
        {"checkpoint": CHECKPOINT_HASH},
        ("evaluate",),
    )
    training_path = tmp_path / "training.json"
    _write_record(training_path, training)
    measurement_path = _measurement_fixture(tmp_path, config, actions=())
    args = SimpleNamespace(
        training_evidence=str(training_path),
        dataset_root=str(tmp_path),
        evaluator_checkpoint=str(tmp_path / "evaluator.pt"),
        checkpoint=str(tmp_path / "checkpoint.pt"),
        acceptance_evaluator_checkpoint=str(tmp_path / "secondary.pt"),
        acceptance_measurement_bundle=str(tmp_path / "measurement_bundle.json"),
        acceptance_measurement_evidence=str(measurement_path),
    )
    monkeypatch.setattr(
        current_evaluation,
        "load_frozen_evaluator",
        lambda *args, **kwargs: pytest.fail("evaluator loaded before measurement action rejection"),
    )
    monkeypatch.setattr(
        current_evaluation,
        "_load_bundle_with_payload",
        lambda *args, **kwargs: pytest.fail("checkpoint loaded before measurement action rejection"),
    )
    with pytest.raises(ValueError, match="evaluate"):
        current_evaluation.conditional_evaluate(args, config)


@pytest.mark.parametrize("field", ["evaluator", "split"])
def test_conditional_evaluate_rejects_measurement_context_mismatch_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    config = _config()
    _stub_mnist(monkeypatch, current_evaluation)
    context = _lineage_context(config)
    training = _record(
        config,
        "conditional_training",
        context,
        {"checkpoint": CHECKPOINT_HASH},
        ("evaluate",),
    )
    training_path = tmp_path / "training.json"
    _write_record(training_path, training)
    mismatch = _context(
        config,
        evaluator_hash=("z" * 64 if field == "evaluator" else EVALUATOR_HASH),
        split_hash=("z" * 64 if field == "split" else SPLIT_HASH),
    )
    measurement_path = _measurement_fixture(tmp_path, config, context=mismatch)
    args = SimpleNamespace(
        training_evidence=str(training_path),
        dataset_root=str(tmp_path),
        evaluator_checkpoint=str(tmp_path / "evaluator.pt"),
        checkpoint=str(tmp_path / "checkpoint.pt"),
        acceptance_evaluator_checkpoint=str(tmp_path / "secondary.pt"),
        acceptance_measurement_bundle=str(tmp_path / "measurement_bundle.json"),
        acceptance_measurement_evidence=str(measurement_path),
    )
    monkeypatch.setattr(
        current_evaluation,
        "load_frozen_evaluator",
        lambda *args, **kwargs: pytest.fail("evaluator loaded before measurement context rejection"),
    )
    monkeypatch.setattr(
        current_evaluation,
        "_load_bundle_with_payload",
        lambda *args, **kwargs: pytest.fail("checkpoint loaded before measurement context rejection"),
    )
    with pytest.raises(ValueError, match="context mismatch"):
        current_evaluation.conditional_evaluate(args, config)


def test_conditional_evaluate_accepts_measurement_authorization_before_checkpoint_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    _stub_mnist(monkeypatch, current_evaluation)
    context = _lineage_context(config)
    training = _record(
        config,
        "conditional_training",
        context,
        {"checkpoint": CHECKPOINT_HASH},
        ("evaluate",),
    )
    training_path = tmp_path / "training.json"
    _write_record(training_path, training)
    measurement_path = _measurement_fixture(tmp_path, config)
    exposure_path = _exposure_fixture(tmp_path, config, measurement_path)
    args = SimpleNamespace(
        training_evidence=str(training_path),
        dataset_root=str(tmp_path),
        evaluator_checkpoint=str(tmp_path / "evaluator.pt"),
        checkpoint=str(tmp_path / "checkpoint.pt"),
        acceptance_evaluator_checkpoint=str(tmp_path / "secondary.pt"),
        acceptance_measurement_bundle=str(tmp_path / "measurement_bundle.json"),
        acceptance_measurement_evidence=str(measurement_path),
        official_test_exposure_evidence=str(exposure_path),
    )
    monkeypatch.setattr(
        current_evaluation,
        "_load_bundle_with_payload",
        lambda *args, **kwargs: pytest.fail("valid measurement did not reach checkpoint boundary"),
    )
    with pytest.raises(pytest.fail.Exception, match="checkpoint boundary"):
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
        acceptance_evaluator_checkpoint=None,
        acceptance_measurement_bundle=None,
        acceptance_measurement_evidence=None,
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
    measurement_path = _measurement_fixture(tmp_path, config)
    exposure_path = _exposure_fixture(tmp_path, config, measurement_path)
    args.acceptance_evaluator_checkpoint = str(tmp_path / "secondary.pt")
    args.acceptance_measurement_bundle = str(tmp_path / "measurement_bundle.json")
    args.acceptance_measurement_evidence = str(measurement_path)
    args.official_test_exposure_evidence = str(exposure_path)
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


def test_admit_dataset_does_not_publish_manifest_if_run_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    root, payload, candidate_context = _candidate_fixture(tmp_path, config)
    measurement_inputs = {
        "secondary_evaluator": "s" * 64,
        "measurement_bundle": "b" * 64,
        "measurement_evidence": "m" * 64,
        "test_manifest": "t" * 64,
        "test_content_manifest": "c" * 64,
        "exposure_genesis": "g" * 64,
        "exposure_chain_id": "h" * 64,
        "formal_evaluation_exposure": "x" * 64,
        "formal_evaluation_id": "y" * 64,
        "exposure_sequence": "1",
    }
    admission = _record(
        config,
        "conditional_generation_admission",
        _context(config, evaluator_hash=EVALUATOR_HASH, split_hash=SPLIT_HASH, checkpoint_hash=CHECKPOINT_HASH),
        {
            "checkpoint": CHECKPOINT_HASH,
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": STATE_HASH,
            **measurement_inputs,
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
    paths = {}
    for name, record in (("admission", admission), ("memorization", memorization), ("utility", utility)):
        path = tmp_path / f"{name}.json"
        _write_record(path, record)
        paths[name] = path
    args = SimpleNamespace(
        generation_admission_evidence=str(paths["admission"]),
        memorization_evidence=str(paths["memorization"]),
        utility_evidence=str(paths["utility"]),
        candidate_dataset=str(root),
    )
    monkeypatch.setattr(
        current_dataset,
        "_new_run",
        lambda *args, **kwargs: pytest.fail("run creation failure"),
    )
    with pytest.raises(pytest.fail.Exception, match="run creation failure"):
        current_dataset.admit_dataset(args, config)
    assert not (root / "accepted_manifest.json").exists()


def test_admit_dataset_publishes_manifest_and_evidence_after_valid_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    root, payload, candidate_context = _candidate_fixture(tmp_path, config)
    measurement_inputs = {
        "secondary_evaluator": "s" * 64,
        "measurement_bundle": "b" * 64,
        "measurement_evidence": "m" * 64,
        "test_manifest": "t" * 64,
        "test_content_manifest": "c" * 64,
        "exposure_genesis": "g" * 64,
        "exposure_chain_id": "h" * 64,
        "formal_evaluation_exposure": "x" * 64,
        "formal_evaluation_id": "y" * 64,
        "exposure_sequence": "1",
    }
    admission = _record(
        config,
        "conditional_generation_admission",
        _context(config, evaluator_hash=EVALUATOR_HASH, split_hash=SPLIT_HASH, checkpoint_hash=CHECKPOINT_HASH),
        {
            "checkpoint": CHECKPOINT_HASH,
            "initialization_checkpoint": INITIALIZATION_HASH,
            "initialization_state_hash": STATE_HASH,
            **measurement_inputs,
        },
        ("export",),
    )
    memorization = _record(config, "conditional_memorization_audit", candidate_context, {}, ("admit_dataset",))
    utility = _record(config, "conditional_utility", candidate_context, {}, ("admit_dataset",))
    paths = {}
    for name, record in (("admission", admission), ("memorization", memorization), ("utility", utility)):
        path = tmp_path / f"{name}.json"
        _write_record(path, record)
        paths[name] = path
    run_root = tmp_path / "run"
    run_root.mkdir()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(
        current_dataset,
        "_new_run",
        lambda *args, **kwargs: SimpleNamespace(
            root=run_root,
            git_commit=commit,
            dirty_patch_hash=None,
            config_hash=canonical_sha256(config.model_dump(mode="json")),
        ),
    )
    args = SimpleNamespace(
        generation_admission_evidence=str(paths["admission"]),
        memorization_evidence=str(paths["memorization"]),
        utility_evidence=str(paths["utility"]),
        candidate_dataset=str(root),
    )
    assert current_dataset.admit_dataset(args, config) == 0
    assert (root / "accepted_manifest.json").exists()
    assert (run_root / "conditional_dataset_accepted_evidence.json").exists()


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

    history[0] = "invalid"
    monkeypatch.setattr(
        current_training,
        "_new_run",
        lambda *args, **kwargs: pytest.fail("malformed history reached run creation"),
    )
    with pytest.raises(ValueError, match="invalid history row"):
        current_training.conditional_training_completion(args, config)
