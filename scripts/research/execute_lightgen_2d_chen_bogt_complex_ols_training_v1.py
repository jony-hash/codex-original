"""Chen BOGT complex-OLS consumer, preflight, and bounded execution entrypoint.

The scientific arm is explicit: ``--execute`` is required.  ``--dry-run``
only prints the frozen execution graph and ``--preflight`` remains the
synthetic/CPU engineering check used before scientific authorization.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen_2d_chen_bogt_complex_ols_training import (
    ALLOWED_TRAINABLE_PARAMETER_NAMES,
    ADAM_BETAS,
    ADAM_EPSILON,
    AMSGRAD,
    BogtLossWeights,
    CHECKPOINT_SCHEMA,
    CHECKPOINT_STEPS,
    DataReadLedger,
    DEV_ROLE,
    DeterministicTrainRoleSampler,
    FrozenBogtReferences,
    LightGen2DChenBOGTComplexOLSTrainingConsumer,
    SCIENTIFIC_UPDATES,
    SYNTHETIC_ROLE,
    TRAIN_ROLE,
    PRODUCTION_PROPAGATION_GRID,
    SCHEDULE_SEED,
    TrainingContractError,
    WEIGHT_DECAY,
    calibrate_initial_gradient_weights,
    checkpoint_payload,
    collect_bogt_train_references,
    construct_sampled_ols,
    contracted_plane_powers,
    fit_wrapped_conditional_phase_model,
    fixed_prior,
    frozen_contract_identity,
    git_identity,
    load_checkpoint,
    save_checkpoint,
    statistical_variable,
    WrappedConditionalPhaseModel,
)
from ldm_project.v2.application.research.research_run_control import (
    HEARTBEAT_INTERVAL_SECONDS,
    PREFLIGHT_BLOCKED,
    PREFLIGHT_READY,
    RECOVERY_INTERVAL_SECONDS,
    RECOVERY_INTERVAL_STEPS,
    RecoveryCheckpointManager,
    ResearchRunControl,
    ResearchRunIdentity,
    RunControlError,
    HeartbeatWriter,
    InterruptController,
    RUN_COMPLETED,
    RUN_INTERRUPTED,
    RUNNING,
    RUN_PREPARING,
    capture_execution_source_manifest,
    recovery_due,
)
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import (
    FrozenMNISTEvaluator,
    load_frozen_evaluator,
    load_local_mnist,
    stratified_mnist_indices,
)


NEXT_TASK = "EXECUTE_CHEN_BOGT_ON_COMPLEX_OLS_BOUNDED_TRAINING_V1"
GRAPH_SCHEMA = "chen_bogt_complex_ols_scientific_execution_graph.v1"
FAST_PATH_OUTER_GATE = "RESEARCH_FAST_PATH_AND_LONG_RUN_RESILIENCE_V1_ESTABLISHED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
SCIENTIFIC_RUN_ID = "chen-bogt-complex-ols-v1"
SCIENTIFIC_TRAJECTORY_ID = "chen-bogt-complex-ols-trajectory-v1"
DEFAULT_OUTPUT_DIR = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01")
TRAINING_SOURCE_RELATIVE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
EVALUATOR_IMPLEMENTATION_RELATIVE_PATH = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_CHECKPOINT_RELATIVE_PATH = Path(
    "outputs/20260724_v2/runs/"
    "20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "frozen_mnist_evaluator.pt"
)
EVALUATOR_CHECKPOINT_SHA256 = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
SEMANTIC_DIAGNOSTIC_THRESHOLD = 0.937
STATISTICS_RECOVERY_KIND = "STATISTICS_RECOVERY_V1"
COLLECTION_RECOVERY_KIND = "POST_TRAIN_COLLECTION_RECOVERY_V1"
FINAL_COLLECTION_SCHEMA = "chen_bogt_final_train_ols_collection.v1"
PHASE_MODEL_ARTIFACT_SCHEMA = "chen_bogt_post_train_phase_model.v1"
GENERATION_BUNDLE_SCHEMA = "chen_bogt_generation_bundle.v1"
REPORT_SCHEMA = "chen_bogt_complex_ols_bounded_training_report.v1"
OUTER_GATE = "CHEN_BOGT_ON_COMPLEX_OLS_BOUNDED_TRAINING_V1_RESOLVED"
RUN_MANIFEST_SCHEMA = "chen_bogt_scientific_execution_manifest.v1"
PHASE0_SCHEMA = "chen_bogt_phase0_lightweight_runtime_checks.v1"
DEV_DIAGNOSTIC_SCHEMA = "chen_bogt_dev_diagnostic.v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fixture_images() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 10 * 28 * 28, dtype=torch.float32).reshape(10, 1, 28, 28)


def _execution_source_paths(root: Path) -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        root / "src/ldm_project/v2/application/research/lightgen_2d_chen_bogt_complex_ols_training.py",
        root / "src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py",
        root / "src/ldm_project/v2/application/research/research_run_control.py",
        root / EVALUATOR_IMPLEMENTATION_RELATIVE_PATH,
    )


def resolve_declared_input_authorities(
    available: Mapping[str, Any],
    declared: Iterable[str],
) -> dict[str, Any]:
    """Consume only the authority names explicitly declared by the task."""
    names = tuple(declared)
    if not names or len(set(names)) != len(names):
        raise RunControlError("input authority list must be non-empty and unique")
    missing = [name for name in names if name not in available]
    if missing:
        raise RunControlError(f"declared input authority is missing: {missing}")
    return {name: available[name] for name in names}


def build_scientific_execution_graph(
    *,
    repo_root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the future graph without constructing data, optimizer, or output."""
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = (
        root / "outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01"
        if output_dir is None
        else (root / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    )
    if target.exists():
        raise TrainingContractError(f"write-once output target already exists: {target}")
    outputs_root = (root / "outputs").resolve()
    if outputs_root not in target.parents:
        raise TrainingContractError("scientific output must remain under outputs/")
    if "final_pt" in target.parts or "_wsl_restore_metadata_v4" in target.parts:
        raise TrainingContractError("restored/final output namespace is forbidden")
    # A new run must have a distinct child below outputs; the literal 'runs'
    # component is expected only as the standard run namespace, not as a target
    # root by itself.
    if target == outputs_root or target == outputs_root / "runs":
        raise TrainingContractError("output target is too broad")
    return {
        "schema": GRAPH_SCHEMA,
        "task_id": NEXT_TASK,
        "outer_gate": FAST_PATH_OUTER_GATE,
        "authorization_required": True,
        "authorization_model": {
            "one_authorization_id": "same authorization passes READY preflight into execution",
            "second_authorization": False,
            "internal_checks_are_not_gates": True,
        },
        "scientific_execution_started": False,
        "output_dir": str(target.relative_to(root).as_posix()),
        "trainable_parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
        "updates": SCIENTIFIC_UPDATES,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "selection": "fixed final update step 300",
        "forbidden_roles": ["dev", "protected", "test", "sealed"],
        "steps": [
            "verify authority and capture the execution source manifest",
            "collect full TRAIN s_j and r_p at theta0",
            "calibrate alpha/beta/gamma from first TRAIN batch",
            "create exact Adam optimizer",
            "execute 300 BOGT updates",
            "write checkpoints 0/100/200/300",
            "freeze and select step 300",
            "collect final TRAIN real OLS",
            "fit post-train phase model once",
            "publish write-once generation bundle",
        ],
        "internal_checks": [
            "CHECK_AUTHORITY_RUNTIME_DATA_ROLES",
            "CHECK_TINY_FORWARD_LOSS_GRADIENT_OPTIMIZER_UPDATE",
            "CHECK_CHECKPOINT_SAVE_RELOAD",
        ],
        "hard_blockers": [
            "required input/authority identity unresolved",
            "data leakage or invalid scientific comparison",
            "missing authorization for a controlled action",
            "conflict with an authoritative contract",
            "mathematically/technically undefined computation",
        ],
        "hash_policy": {
            "semantic_identity": "semantic ID plus run/task identity; exact hashes only at named machine boundaries",
            "cross_domain_equality": False,
            "dirty_worktree": "allowed when the source manifest captures relevant execution semantics",
        },
        "heartbeat": {
            "interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
            "stale_warning_seconds": 300.0,
            "stale_action": "warn_only_no_kill_or_restart",
            "progress_log": "progress.jsonl_append_flush",
            "latest_status": "latest_status.json_atomic_replace",
        },
        "recovery": {
            "interval_seconds": RECOVERY_INTERVAL_SECONDS,
            "interval_steps": RECOVERY_INTERVAL_STEPS,
            "rotation_keep": 3,
            "artifact_scope": "operational_only",
            "scientific_checkpoint_steps": list(CHECKPOINT_STEPS),
            "resume_identity": "same task/run/trajectory/scientific-attempt and execution semantics",
        },
        "resume_argument": "--resume <run_dir|checkpoint>",
        "input_authorities": ["M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION", "CHEN_BOGT_ON_COMPLEX_OLS_TRAINING_CONSUMER_V1"],
        "operational_artifacts": ["progress.jsonl", "latest_status.json", "recovery_*.pt"],
        "scientific_artifacts": ["scientific_step_000000.pt", "scientific_step_000100.pt", "scientific_step_000200.pt", "scientific_step_000300.pt"],
        "implementation_only": False,
        "cli_execution": "--execute",
    }


def _synthetic_phase_bank(references: FrozenBogtReferences) -> torch.Tensor:
    rows = 24
    u = torch.full((rows, 100), 0.5, dtype=torch.float64)
    u[:, 0] = torch.linspace(1.0, 2.5, rows, dtype=torch.float64)
    u[:, 1] = torch.linspace(0.7, 1.7, rows, dtype=torch.float64)
    phase = torch.zeros((rows, 100), dtype=torch.float64)
    phase[:, 1] = 0.1 * torch.log(u[:, 0])
    phase[:, 2] = -0.2 * torch.log(u[:, 1])
    phase[:, 3] = 0.05 * torch.log(u[:, 0] * u[:, 1])
    amplitude = ((references.scale + 1.0e-12) * u).sqrt()
    return torch.complex(amplitude * phase.cos(), amplitude * phase.sin()).to(torch.complex64)


def run_engineering_preflight(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Run only synthetic/three-step CPU engineering checks."""
    root = _repo_root() if repo_root is None else repo_root.resolve()
    ledger = DataReadLedger()
    run_temp = tempfile.TemporaryDirectory(prefix="chen-bogt-fast-path-")
    try:
        source_manifest = capture_execution_source_manifest(
            root,
            task_id=NEXT_TASK,
            semantic_authority_id=SEMANTIC_AUTHORITY_ID,
            relevant_paths=_execution_source_paths(root),
            config_identity={"propagation_grid_size": 128, "engineering_updates": 3},
        )
        run_identity = ResearchRunIdentity(
            task_id=NEXT_TASK,
            run_id=f"{SCIENTIFIC_RUN_ID}:engineering-preflight",
            trajectory_id=f"{SCIENTIFIC_TRAJECTORY_ID}:engineering-preflight",
            semantic_authority_id=SEMANTIC_AUTHORITY_ID,
            data_role=SYNTHETIC_ROLE,
            config_identity={"propagation_grid_size": 128, "engineering_updates": 3},
            model_identity="LIGHTGEN_2D_CHEN_BOGT_COMPLEX_OLS_CONSUMER_V1",
            optimizer_identity={"name": "Adam", "learning_rate": 1.0e-3},
            sampler_identity={"seed": 1234, "steps": 3, "role": SYNTHETIC_ROLE},
            execution_source_semantics_id=source_manifest.execution_source_semantics_id,
            source_manifest_id=source_manifest.source_manifest_id,
        )
        control = ResearchRunControl(
            identity=run_identity,
            authorization_id=f"{NEXT_TASK}:authorization",
        )
        heartbeat = HeartbeatWriter(
            run_temp.name,
            run_identity.run_id,
            total_steps=3,
            interval_seconds=0.0,
            clock=time.time,
        )
        recovery = RecoveryCheckpointManager(
            run_temp.name,
            run_identity,
            source_manifest_id=source_manifest.source_manifest_id,
            authorization_id=control.authorization_id,
        )
        heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=0, force=True)
        images = _fixture_images()
        consumer_a = LightGen2DChenBOGTComplexOLSTrainingConsumer(
            device="cpu", propagation_grid_size=128
        )
        references = collect_bogt_train_references(
            consumer_a,
            images,
            data_role=SYNTHETIC_ROLE,
            engineering_only=True,
            source_indices=torch.arange(images.shape[0], dtype=torch.long),
            data_identity={"source": "in-memory synthetic fixture"},
            ledger=ledger,
        )
        consumer_a.bind_references(references)
        weights = calibrate_initial_gradient_weights(
            consumer_a,
            images,
            references,
            data_role=SYNTHETIC_ROLE,
            engineering_only=True,
            calibration_batch_identity={"source": "synthetic fixture", "seed": 1234, "batch_size": 10, "class_balanced": True},
            ledger=ledger,
        )
        consumer_a.bind_weights(weights)

        expected = [parameter for _, parameter in consumer_a._named_runtime_parameters()]
        extra = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        try:
            consumer_a.create_optimizer([*expected, extra])
        except TrainingContractError:
            negative_control = True
        else:
            negative_control = False
        if not negative_control:
            raise TrainingContractError("frozen optimizer candidate negative control was accepted")

        control.mark_preflight_ready()
        control.start_execution()
        heartbeat.update(state=RUNNING, phase="TRAINING", step=0, force=True)
        optimizer_a = consumer_a.create_optimizer()
        sampler_a = DeterministicTrainRoleSampler(
            torch.arange(55_000, dtype=torch.long),
            torch.arange(60_000, dtype=torch.long).remainder(10),
        )
        trace_a: list[dict[str, Any]] = []
        for step in range(3):
            sampler_a.next_batch()
            update = consumer_a.bogt_training_step(
                images,
                optimizer_a,
                global_step=step,
                data_role=SYNTHETIC_ROLE,
                ledger=ledger,
            )
            trace_a.append(update.metrics())
            heartbeat.update(
                state=RUNNING,
                phase="TRAINING",
                step=step + 1,
                latest_losses=update.metrics().get("loss"),
                force=True,
            )

        consumer_b = LightGen2DChenBOGTComplexOLSTrainingConsumer(
            device="cpu", propagation_grid_size=128
        )
        consumer_b.bind_references(references)
        consumer_b.bind_weights(weights)
        optimizer_b = consumer_b.create_optimizer()
        sampler_b = DeterministicTrainRoleSampler(
            torch.arange(55_000, dtype=torch.long),
            torch.arange(60_000, dtype=torch.long).remainder(10),
        )
        sampler_b.next_batch()
        first_update = consumer_b.bogt_training_step(
            images,
            optimizer_b,
            global_step=0,
            data_role=SYNTHETIC_ROLE,
            ledger=ledger,
        )
        with tempfile.TemporaryDirectory(prefix="chen-bogt-preflight-") as temporary:
            checkpoint = Path(temporary) / "step_000001.engineering.pt"
            save_checkpoint(
                checkpoint,
                checkpoint_payload(
                    consumer_b,
                    optimizer_b,
                    completed_updates=1,
                    sampler_state=sampler_b.state_dict(),
                    training_history=[first_update.metrics()],
                    engineering_only=True,
                    git=git_identity(root),
                    authorization_id=control.authorization_id,
                ),
            )
            consumer_c = LightGen2DChenBOGTComplexOLSTrainingConsumer(
                device="cpu", propagation_grid_size=128
            )
            optimizer_c = consumer_c.create_optimizer()
            loaded = load_checkpoint(
                checkpoint,
                consumer_c,
                optimizer_c,
                expected_engineering_only=True,
            )
            sampler_c = DeterministicTrainRoleSampler(
                torch.arange(55_000, dtype=torch.long),
                torch.arange(60_000, dtype=torch.long).remainder(10),
            )
            sampler_c.load_state_dict(loaded.sampler_state)
            trace_c = [first_update.metrics()]
            for step in range(1, 3):
                sampler_c.next_batch()
                trace_c.append(
                    consumer_c.bogt_training_step(
                        images,
                        optimizer_c,
                        global_step=step,
                        data_role=SYNTHETIC_ROLE,
                        ledger=ledger,
                    ).metrics()
                )
        for name, value in consumer_a.trainable_state_dict().items():
            if not torch.allclose(value, consumer_c.trainable_state_dict()[name], rtol=1.0e-5, atol=1.0e-6):
                raise TrainingContractError(f"checkpoint resume state mismatch: {name}")
        if [record["loss"] for record in trace_a] != [record["loss"] for record in trace_c]:
            for left, right in zip(trace_a, trace_c, strict=True):
                for key in left["loss"]:
                    if abs(left["loss"][key] - right["loss"][key]) > 1.0e-6:
                        raise TrainingContractError("checkpoint resume loss trace mismatch")
        if sampler_a.state_dict()["cursor"] != sampler_c.state_dict()["cursor"]:
            raise TrainingContractError("checkpoint resume sampler cursor mismatch")

        try:
            collect_bogt_train_references(
                consumer_a,
                images,
                data_role=DEV_ROLE,
                engineering_only=True,
                ledger=ledger,
            )
        except TrainingContractError:
            pass
        else:
            raise TrainingContractError("DEV data-role negative control was accepted")
        ledger.assert_forbidden_zero()

        phase_references = FrozenBogtReferences(
            torch.ones(100, dtype=torch.float64),
            torch.ones(5, dtype=torch.float64),
            SYNTHETIC_ROLE,
            24,
            True,
            consumer_a.initialization_fingerprint(),
            {"source": "synthetic phase bank"},
        )
        phase_z = _synthetic_phase_bank(phase_references)
        phase_model = fit_wrapped_conditional_phase_model(
            phase_z,
            phase_references,
            data_role=SYNTHETIC_ROLE,
            engineering_only=True,
        )
        sampled_u = statistical_variable(phase_z[:3], phase_references)
        sampled = construct_sampled_ols(
            sampled_u,
            phase_references,
            phase_model,
            generator=torch.Generator(device="cpu").manual_seed(7),
        )
        recovered_u = statistical_variable(sampled.coefficients, phase_references)
        if not torch.allclose(recovered_u, sampled_u, rtol=2.0e-5, atol=2.0e-6):
            raise TrainingContractError("sample construction did not preserve u")
        real_result = consumer_a.runtime.forward_from_image(images)
        generated_result = consumer_a.runtime.forward_from_ols(real_result.z_ols)
        if not torch.allclose(
            real_result.generator_input_field.tensor,
            generated_result.generator_input_field.tensor,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise TrainingContractError("real/generated W_OLS parity failed")

        recovery_checkpoint = recovery.save({"preflight": True, "completed_updates": 3}, step=3)
        control.complete()
        heartbeat.update(
            state=RUN_COMPLETED,
            phase="COMPLETED",
            step=3,
            last_recovery_checkpoint=recovery_checkpoint.name,
            force=True,
        )

        return {
            "outer_gate": FAST_PATH_OUTER_GATE,
            "preflight_status": control.preflight_status,
            "engineering_only": True,
            "authorization_id": control.authorization_id,
            "run_state": control.state,
            "run_identity": run_identity.to_dict(),
            "source_manifest": source_manifest.to_dict(),
            "scientific_updates": 0,
            "full_train_statistics": False,
            "full_dev_evaluation": False,
            "full_phase_fit": False,
            "generation_evaluation": False,
            "negative_controls": {"optimizer_frozen_component": True, "dev_role": True},
            "data_reads": ledger.to_dict(),
            "protected_reads": ledger.counts["protected"],
            "test_reads": ledger.counts["test"],
            "sealed_reads": ledger.counts["sealed"],
        }
    except Exception as error:
        return {
            "outer_gate": FAST_PATH_OUTER_GATE,
            "preflight_status": PREFLIGHT_BLOCKED,
            "engineering_only": True,
            "scientific_updates": 0,
            "error_type": type(error).__name__,
            "error": str(error),
            "data_reads": ledger.to_dict(),
            "protected_reads": ledger.counts["protected"],
            "test_reads": ledger.counts["test"],
            "sealed_reads": ledger.counts["sealed"],
        }
    finally:
        run_temp.cleanup()


def run_bogt_scientific_trajectory(
    *,
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    references: FrozenBogtReferences,
    weights: BogtLossWeights,
    sampler: DeterministicTrainRoleSampler,
    batch_loader: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    run_dir: str | Path,
    run_identity: ResearchRunIdentity,
    source_manifest: Mapping[str, Any],
    authorization_id: str,
    preflight_status: str,
    resume_checkpoint: str | Path | None = None,
    repo_root: Path | None = None,
    ledger: DataReadLedger | None = None,
    on_checkpoint: Callable[
        [int, LightGen2DChenBOGTComplexOLSTrainingConsumer, Iterable[Mapping[str, Any]]],
        Mapping[str, Any] | None,
    ] | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Run the explicit 300-update arm with resumable operational state."""
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = Path(run_dir).resolve()
    outputs_root = (root / "outputs").resolve()
    if outputs_root not in target.parents or "final_pt" in target.parts or "_wsl_restore_metadata_v4" in target.parts:
        raise RunControlError("scientific run directory is outside the allowed outputs namespace")
    if target in {outputs_root, outputs_root / "runs"}:
        raise RunControlError("scientific run directory is too broad")
    resuming = resume_checkpoint is not None
    if not resuming and target.exists():
        existing_scientific = tuple(target.glob("scientific_step_*.pt"))
        if existing_scientific:
            raise RunControlError(f"write-once scientific output target already contains checkpoints: {target}")
    if not isinstance(source_manifest, Mapping):
        raise RunControlError("scientific execution requires an explicit source manifest")
    if source_manifest.get("task_id") != run_identity.task_id or source_manifest.get("semantic_authority_id") != run_identity.semantic_authority_id:
        raise RunControlError("source manifest authority identity differs from the run")
    if source_manifest.get("execution_source_semantics_id") != run_identity.execution_source_semantics_id:
        raise RunControlError("execution source semantics differ from the run identity")
    if run_identity.source_manifest_id is None or source_manifest.get("source_manifest_id") != run_identity.source_manifest_id:
        raise RunControlError("run identity requires the matching source manifest provenance")
    if run_identity.data_role != TRAIN_ROLE:
        raise RunControlError("scientific trajectory requires the TRAIN data role")
    if preflight_status != PREFLIGHT_READY:
        raise RunControlError("scientific trajectory requires the same READY preflight")
    if not authorization_id:
        raise RunControlError("scientific trajectory requires one explicit authorization")
    if resuming and run_identity.scientific_attempt_id is None:
        raise RunControlError("resume requires the existing scientific attempt identity")

    control = ResearchRunControl(run_identity, authorization_id)
    control.mark_preflight_ready()
    control.start_execution()
    consumer.bind_references(references, for_scientific=True)
    consumer.bind_weights(weights, for_scientific=True)
    optimizer = consumer.create_optimizer()
    # The scientific attempt is created only after the TRAIN consumer and its
    # exact optimizer are ready, immediately before the first scientific
    # checkpoint/update.  A resume carries this same attempt identity.
    effective_identity = control.begin_scientific_attempt(run_identity.scientific_attempt_id)
    target.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatWriter(
        target,
        effective_identity.run_id,
        total_steps=SCIENTIFIC_UPDATES,
        clock=clock,
    )
    recovery = RecoveryCheckpointManager(
        target,
        effective_identity,
        source_manifest_id=effective_identity.source_manifest_id,
        authorization_id=authorization_id,
    )
    history: list[Mapping[str, Any]] = []
    completed = 0
    last_recovery_step = 0
    last_recovery_time = float(clock())
    last_recovery_checkpoint: Path | None = None
    latest_metrics: Mapping[str, Any] = {}
    checkpoint_diagnostics: dict[str, Mapping[str, Any]] = {}

    def checkpoint_payload_for(step: int, *, operational_recovery: bool) -> Mapping[str, Any]:
        return checkpoint_payload(
            consumer,
            optimizer,
            completed_updates=step,
            sampler_state=sampler.state_dict(),
            training_history=history,
            engineering_only=False,
            git=git_identity(root),
            recovery=operational_recovery,
            run_identity=effective_identity,
            source_manifest=source_manifest,
            authorization_id=authorization_id,
        )

    def save_scientific_checkpoint(step: int) -> Path:
        checkpoint = target / f"scientific_step_{step:06d}.pt"
        save_checkpoint(checkpoint, checkpoint_payload_for(step, operational_recovery=False))
        return checkpoint

    resume_path: Path | None = None
    if resume_checkpoint is not None:
        requested = Path(resume_checkpoint).resolve()
        if requested.is_dir():
            if requested != target:
                raise RunControlError("resume directory must be the named run directory")
            resume_path = recovery.latest()
            if resume_path is None:
                raise RunControlError("the named run directory has no operational recovery checkpoint")
        else:
            resume_path = requested
        if resume_path.parent != target:
            raise RunControlError("resume checkpoint must be inside the named run directory")
        loaded = load_checkpoint(
            resume_path,
            consumer,
            optimizer,
            expected_engineering_only=False,
            expected_run_identity=effective_identity,
            expected_authorization_id=authorization_id,
        )
        sampler.load_state_dict(loaded.sampler_state)
        completed = loaded.completed_updates
        history = list(loaded.training_history)
        if sampler.cursor != completed:
            raise RunControlError("resume sampler cursor does not equal completed updates")
        heartbeat.update(
            state=RUNNING,
            phase="TRAINING",
            step=completed,
            latest_losses=history[-1].get("loss", {}) if history else {},
            force=True,
        )
        last_recovery_step = completed if loaded.recovery else max(0, completed - 1)
        if on_checkpoint is not None and completed in CHECKPOINT_STEPS:
            diagnostic = on_checkpoint(completed, consumer, history)
            if diagnostic is not None:
                checkpoint_diagnostics[str(completed)] = dict(diagnostic)
    else:
        if sampler.cursor != 0:
            raise RunControlError("a fresh scientific run requires a zeroed sampler")
        save_scientific_checkpoint(0)
        heartbeat.update(state=RUNNING, phase="TRAINING", step=0, force=True)
        if on_checkpoint is not None:
            diagnostic = on_checkpoint(0, consumer, history)
            heartbeat.update(state=RUNNING, phase="TRAINING", step=0, force=True)
            if diagnostic is not None:
                checkpoint_diagnostics["0"] = dict(diagnostic)

    def recovery_payload_for(step: int) -> Mapping[str, Any]:
        return checkpoint_payload_for(step, operational_recovery=True)

    controller = InterruptController(
        heartbeat=heartbeat,
        recovery=recovery,
        payload_factory=recovery_payload_for,
        phase="TRAINING",
        total_steps=SCIENTIFIC_UPDATES,
    )
    with controller:
        while completed < SCIENTIFIC_UPDATES:
            indices, labels = sampler.next_batch()
            batch = batch_loader(indices, labels)
            update = consumer.bogt_training_step(
                batch,
                optimizer,
                global_step=completed,
                data_role=TRAIN_ROLE,
                ledger=ledger,
            )
            completed += 1
            latest_metrics = update.metrics()
            history.append(latest_metrics)
            now = float(clock())
            heartbeat.update(
                state=RUNNING,
                phase="TRAINING",
                step=completed,
                latest_losses=latest_metrics.get("loss", {}),
                last_recovery_checkpoint=None if last_recovery_checkpoint is None else last_recovery_checkpoint.name,
            )
            if completed in CHECKPOINT_STEPS[1:]:
                save_scientific_checkpoint(completed)
                if on_checkpoint is not None:
                    diagnostic = on_checkpoint(completed, consumer, history)
                    heartbeat.update(
                        state=RUNNING,
                        phase="TRAINING",
                        step=completed,
                        latest_losses=latest_metrics.get("loss", {}),
                        force=True,
                    )
                    if diagnostic is not None:
                        checkpoint_diagnostics[str(completed)] = dict(diagnostic)
            if recovery_due(
                step=completed,
                last_recovery_step=last_recovery_step,
                now=now,
                last_recovery_time=last_recovery_time,
            ):
                last_recovery_checkpoint = recovery.save(recovery_payload_for(completed), step=completed)
                last_recovery_step = completed
                last_recovery_time = now
                heartbeat.update(
                    state=RUNNING,
                    phase="TRAINING",
                    step=completed,
                    latest_losses=latest_metrics.get("loss", {}),
                    last_recovery_checkpoint=last_recovery_checkpoint.name,
                    force=True,
                )
            interrupted_checkpoint = controller.handle_safe_boundary(
                step=completed,
                latest_losses=latest_metrics.get("loss", {}),
                last_recovery_checkpoint=None if last_recovery_checkpoint is None else last_recovery_checkpoint.name,
            )
            if interrupted_checkpoint is not None:
                control.interrupted()
                return {
                    "state": RUN_INTERRUPTED,
                    "completed_updates": completed,
                    "scientific_attempt_id": effective_identity.scientific_attempt_id,
                    "recovery_checkpoint": interrupted_checkpoint.name,
                    "history": history,
                    "checkpoint_diagnostics": checkpoint_diagnostics,
                    "optimizer_manifest": consumer.optimizer_manifest(optimizer),
                    "sampler_state": sampler.state_dict(),
                    "run_identity": effective_identity.to_dict(),
                }
    control.complete()
    heartbeat.update(
        state=RUN_COMPLETED,
        phase="COMPLETED",
        step=completed,
        latest_losses=latest_metrics.get("loss", {}) if latest_metrics else {},
        last_recovery_checkpoint=None if last_recovery_checkpoint is None else last_recovery_checkpoint.name,
        force=True,
    )
    return {
        "state": RUN_COMPLETED,
        "completed_updates": completed,
        "selected_step": 300,
        "scientific_attempt_id": effective_identity.scientific_attempt_id,
        "scientific_checkpoint": "scientific_step_000300.pt",
        "operational_recovery": False,
        "history": history,
        "checkpoint_diagnostics": checkpoint_diagnostics,
        "optimizer_manifest": consumer.optimizer_manifest(optimizer),
        "sampler_state": sampler.state_dict(),
        "run_identity": effective_identity.to_dict(),
    }


def _resolve_under_root(root: Path, value: str | Path) -> Path:
    target = Path(value)
    return (root / target if not target.is_absolute() else target).resolve()


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(torch.float64)
    finite = bool(torch.isfinite(tensor).all().item())
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return {"finite": finite, "count": 0}
    return {
        "finite": finite,
        "count": int(flat.numel()),
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "min": float(flat.min().cpu()),
        "max": float(flat.max().cpu()),
    }


def _phase_change(initial: Mapping[str, torch.Tensor], current: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float((current[name].to(torch.float64) - value.to(torch.float64)).square().sum().sqrt().cpu())
        for name, value in initial.items()
    }


def _load_scientific_dataset(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = root / TRAINING_SOURCE_RELATIVE_PATH
    if not source.is_file():
        raise RunControlError(f"required TRAIN source is missing: {source}")
    actual_sha = file_sha256(source)
    if actual_sha != TRAINING_SOURCE_SHA256:
        raise RunControlError(
            "required TRAIN source identity changed: "
            f"expected {TRAINING_SOURCE_SHA256}, got {actual_sha}"
        )
    images, labels = load_local_mnist(root / "data", train=True)
    if tuple(images.shape) != (60_000, 1, 28, 28) or tuple(labels.shape) != (60_000,):
        raise RunControlError("local MNIST train source shape changed")
    if images.dtype != torch.float32 or not torch.isfinite(images).all() or not bool(((images >= 0.0) & (images <= 1.0)).all().item()):
        raise RunControlError("local MNIST train images are not finite float32 values in [0,1]")
    if labels.dtype != torch.long or not torch.equal(torch.unique(labels), torch.arange(10, dtype=torch.long)):
        raise RunControlError("local MNIST train labels are not the registered ten-class source")
    train_indices, development_indices = stratified_mnist_indices(
        labels,
        seed=SCHEDULE_SEED,
        validation_per_label=500,
    )
    train_indices = torch.sort(train_indices).values.contiguous()
    development_indices = torch.sort(development_indices).values.contiguous()
    if train_indices.numel() != 55_000 or development_indices.numel() != 5_000:
        raise RunControlError("registered MNIST split counts changed")
    combined = torch.cat((train_indices, development_indices))
    if torch.unique(combined).numel() != 60_000:
        raise RunControlError("TRAIN and DEV roles overlap")
    train_counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
    development_counts = torch.bincount(labels.index_select(0, development_indices), minlength=10)
    if not torch.equal(development_counts, torch.full((10,), 500, dtype=torch.long)):
        raise RunControlError("DEV role is not exactly 500 rows per class")
    if int(train_counts.min()) < SCIENTIFIC_UPDATES:
        raise RunControlError("TRAIN role cannot support the fixed one-row-per-class schedule")
    data_identity = {
        "source_path": TRAINING_SOURCE_RELATIVE_PATH.as_posix(),
        "source_sha256": actual_sha,
        "loader": "ldm_project.v2.student.evaluator.load_local_mnist",
        "split": "stratified_mnist_indices(seed=1234, validation_per_label=500), then source-row sort",
        "seed": SCHEDULE_SEED,
        "train_count": int(train_indices.numel()),
        "development_count": int(development_indices.numel()),
        "train_counts_by_class": [int(value) for value in train_counts.tolist()],
        "development_counts_by_class": [int(value) for value in development_counts.tolist()],
        "labels_tensor_sha256": tensor_sha256(labels),
        "train_indices_sha256": tensor_sha256(train_indices),
        "development_indices_sha256": tensor_sha256(development_indices),
        "train_order": "increasing canonical source row order for statistics and collection",
        "development_use": "diagnostics only",
        "protected_test_read": False,
        "sealed_read": False,
    }
    return images, labels, train_indices, development_indices, data_identity


def _load_evaluator(root: Path, device: torch.device) -> tuple[FrozenMNISTEvaluator, dict[str, Any]]:
    implementation = root / EVALUATOR_IMPLEMENTATION_RELATIVE_PATH
    checkpoint = root / EVALUATOR_CHECKPOINT_RELATIVE_PATH
    if not implementation.is_file() or not checkpoint.is_file():
        raise RunControlError("explicit FrozenMNISTEvaluator authority is unavailable")
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != EVALUATOR_CHECKPOINT_SHA256:
        raise RunControlError("FrozenMNISTEvaluator checkpoint identity changed")
    evaluator, payload = load_frozen_evaluator(checkpoint, device=device)
    return evaluator, {
        "implementation": EVALUATOR_IMPLEMENTATION_RELATIVE_PATH.as_posix(),
        "implementation_sha256": file_sha256(implementation),
        "checkpoint": EVALUATOR_CHECKPOINT_RELATIVE_PATH.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "architecture": "FrozenMNISTEvaluator v1",
        "validation_accuracy": float(payload.get("validation_accuracy", 0.0)),
        "view": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28",
        "role": "frozen diagnostic only; no training gradient",
    }


def _make_run_config(data_identity: Mapping[str, Any], evaluator_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "propagation_grid_size": PRODUCTION_PROPAGATION_GRID,
        "encoder_layers": 2,
        "generator_layers": 2,
        "phase_initialization": "exact zeros",
        "ols": "10x10 complex, 100 ports, row-major, T=I_100",
        "train_count": 55_000,
        "development_count": 5_000,
        "logical_batch_size": 10,
        "schedule_seed": SCHEDULE_SEED,
        "updates": SCIENTIFIC_UPDATES,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "selection": "fixed final update step 300",
        "data_source": {
            "path": data_identity["source_path"],
            "sha256": data_identity["source_sha256"],
            "train_indices_sha256": data_identity["train_indices_sha256"],
            "development_indices_sha256": data_identity["development_indices_sha256"],
        },
        "evaluator": {
            "implementation": evaluator_identity["implementation"],
            "implementation_sha256": evaluator_identity["implementation_sha256"],
            "checkpoint": evaluator_identity["checkpoint"],
            "checkpoint_sha256": evaluator_identity["checkpoint_sha256"],
        },
    }


def _statistics_recovery_payload(state: Mapping[str, Any], *, data_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": STATISTICS_RECOVERY_KIND,
        "phase": "STATISTICS",
        "completed_rows": int(state["completed_rows"]),
        "scale_sum": state["scale_sum"].detach().cpu().clone(),
        "power_sum": state["power_sum"].detach().cpu().clone(),
        "data_identity": dict(data_identity),
        "contract": "one float64 TRAIN theta0 pass; s_j and r_p frozen before optimizer",
    }


def _collect_statistics_with_resilience(
    *,
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    images: torch.Tensor,
    train_indices: torch.Tensor,
    data_identity: Mapping[str, Any],
    target: Path,
    run_identity: ResearchRunIdentity,
    recovery: RecoveryCheckpointManager,
    ledger: DataReadLedger,
    resume_payload: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    consumer.assert_theta0()
    state: dict[str, Any] = {
        "completed_rows": 0,
        "scale_sum": torch.zeros(100, dtype=torch.float64),
        "power_sum": torch.zeros(5, dtype=torch.float64),
    }
    if resume_payload is not None:
        if resume_payload.get("kind") != STATISTICS_RECOVERY_KIND:
            raise RunControlError("statistics resume payload has the wrong phase")
        completed_rows = resume_payload.get("completed_rows")
        scale_sum = resume_payload.get("scale_sum")
        power_sum = resume_payload.get("power_sum")
        if (
            not isinstance(completed_rows, int)
            or not 0 <= completed_rows <= 55_000
            or not isinstance(scale_sum, torch.Tensor)
            or scale_sum.shape != (100,)
            or scale_sum.dtype != torch.float64
            or not isinstance(power_sum, torch.Tensor)
            or power_sum.shape != (5,)
            or power_sum.dtype != torch.float64
        ):
            raise RunControlError("statistics recovery payload is malformed")
        if resume_payload.get("data_identity") != dict(data_identity):
            raise RunControlError("statistics recovery data identity changed")
        state.update({"completed_rows": completed_rows, "scale_sum": scale_sum.clone(), "power_sum": power_sum.clone()})

    heartbeat = HeartbeatWriter(target, run_identity.run_id, total_steps=55_000, clock=clock)
    heartbeat.update(state=RUNNING, phase="STATISTICS", step=state["completed_rows"], force=True)
    last_recovery_step = int(state["completed_rows"])
    last_recovery_time = float(clock())
    last_recovery_checkpoint: Path | None = None

    def recovery_payload(_: int) -> Mapping[str, Any]:
        return _statistics_recovery_payload(state, data_identity=data_identity)

    controller = InterruptController(
        heartbeat=heartbeat,
        recovery=recovery,
        payload_factory=recovery_payload,
        phase="STATISTICS",
        total_steps=55_000,
    )
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    try:
        with controller:
            while state["completed_rows"] < 55_000:
                start = int(state["completed_rows"])
                row_indices = train_indices[start : start + 10]
                batch = images.index_select(0, row_indices).to(consumer.device)
                with torch.no_grad():
                    result = consumer.runtime.forward_from_image(batch)
                    z = result.z_ols.coefficients
                    scale = z.real.to(torch.float64).square() + z.imag.to(torch.float64).square()
                    powers = contracted_plane_powers(result)
                if not torch.isfinite(scale).all() or not torch.isfinite(powers).all():
                    raise TrainingContractError("nonfinite theta0 statistics")
                state["scale_sum"] += scale.sum(dim=0).cpu()
                state["power_sum"] += powers.to(torch.float64).sum(dim=0).cpu()
                state["completed_rows"] = start + int(row_indices.numel())
                ledger.record(TRAIN_ROLE, int(row_indices.numel()))
                now = float(clock())
                heartbeat.update(
                    state=RUNNING,
                    phase="STATISTICS",
                    step=state["completed_rows"],
                    last_recovery_checkpoint=None if last_recovery_checkpoint is None else last_recovery_checkpoint.name,
                )
                if recovery_due(
                    step=state["completed_rows"],
                    last_recovery_step=last_recovery_step,
                    now=now,
                    last_recovery_time=last_recovery_time,
                    interval_steps=250,
                ):
                    last_recovery_checkpoint = recovery.save(recovery_payload(state["completed_rows"]), step=state["completed_rows"])
                    last_recovery_step = state["completed_rows"]
                    last_recovery_time = now
                    heartbeat.update(
                        state=RUNNING,
                        phase="STATISTICS",
                        step=state["completed_rows"],
                        last_recovery_checkpoint=last_recovery_checkpoint.name,
                        force=True,
                    )
                interrupted = controller.handle_safe_boundary(
                    step=state["completed_rows"],
                    last_recovery_checkpoint=None if last_recovery_checkpoint is None else last_recovery_checkpoint.name,
                )
                if interrupted is not None:
                    return {
                        "interrupted": True,
                        "completed_rows": state["completed_rows"],
                        "recovery_checkpoint": interrupted.name,
                    }
    finally:
        consumer.runtime.train(was_training)

    scale = state["scale_sum"] / 55_000.0
    power_reference = state["power_sum"] / 55_000.0
    if not torch.isfinite(scale).all() or not bool((scale > 1.0e-12).all().item()):
        raise TrainingContractError("scientific s_j is not finite and strictly positive")
    if not torch.isfinite(power_reference).all() or not bool((power_reference > 0.0).all().item()):
        raise TrainingContractError("scientific r_p is not finite and strictly positive")
    references = FrozenBogtReferences(
        scale,
        power_reference,
        TRAIN_ROLE,
        55_000,
        False,
        consumer.initialization_fingerprint(),
        dict(data_identity),
    )
    return {
        "interrupted": False,
        "references": references,
        "s_summary": _tensor_summary(scale),
        "r_power_summary": _tensor_summary(power_reference),
        "last_recovery_checkpoint": None if last_recovery_checkpoint is None else last_recovery_checkpoint.name,
        "completed_rows": 55_000,
    }


def _evaluate_dev_checkpoint(
    *,
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    evaluator: FrozenMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    development_indices: torch.Tensor,
    references: FrozenBogtReferences,
    ledger: DataReadLedger,
) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    evaluator.eval()
    correct = torch.zeros(10, dtype=torch.long)
    totals = torch.zeros(10, dtype=torch.long)
    q_values: list[torch.Tensor] = []
    powers: list[torch.Tensor] = []
    reconstruction_sum = 0.0
    raw_count = 0
    raw_sum = 0.0
    raw_square_sum = 0.0
    raw_min = float("inf")
    raw_max = float("-inf")
    raw_nonconstant = True
    try:
        with torch.no_grad():
            for start in range(0, int(development_indices.numel()), 10):
                row_indices = development_indices[start : start + 10]
                batch = images.index_select(0, row_indices).to(consumer.device)
                result = consumer.runtime.forward_from_image(batch)
                raw = result.raw_intensity
                if not torch.isfinite(raw).all():
                    raise TrainingContractError("DEV raw CMOS became nonfinite")
                raw_cpu = raw.detach().cpu().to(torch.float64)
                raw_count += int(raw_cpu.numel())
                raw_sum += float(raw_cpu.sum())
                raw_square_sum += float(raw_cpu.square().sum())
                raw_min = min(raw_min, float(raw_cpu.min()))
                raw_max = max(raw_max, float(raw_cpu.max()))
                raw_nonconstant = raw_nonconstant and bool((raw_cpu.reshape(raw_cpu.shape[0], -1).std(dim=1, unbiased=False) > 0.0).all().item())
                target = result.image_input.slm_intensity if result.image_input is not None else None
                if target is None:
                    raise TrainingContractError("DEV real path did not expose the SLM target")
                reconstruction_sum += float((raw - target).square().sum().cpu())
                q_values.append(statistical_variable(result.z_ols.coefficients, references).cpu())
                powers.append(contracted_plane_powers(result).to(torch.float64).cpu())
                view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))
                _, logits = evaluator(view)
                prediction = logits.argmax(dim=1).cpu()
                target_labels = labels.index_select(0, row_indices)
                for digit in range(10):
                    mask = target_labels == digit
                    totals[digit] += int(mask.sum())
                    correct[digit] += int((prediction[mask] == digit).sum())
                ledger.record(DEV_ROLE, int(row_indices.numel()))
    finally:
        consumer.runtime.train(was_training)
    q_tensor = torch.cat(q_values, dim=0)
    power_tensor = torch.cat(powers, dim=0)
    per_class = {
        str(digit): float(correct[digit].item() / totals[digit].item())
        for digit in range(10)
    }
    recalls = torch.tensor(list(per_class.values()), dtype=torch.float64)
    return {
        "semantic": {
            "evaluator": "FrozenMNISTEvaluator v1",
            "macro_recall": float(recalls.mean()),
            "per_class_recall": per_class,
            "diagnostic_threshold": SEMANTIC_DIAGNOSTIC_THRESHOLD,
            "threshold_reached": bool((recalls >= SEMANTIC_DIAGNOSTIC_THRESHOLD).all().item()),
        },
        "reconstruction": {
            "L_rec": reconstruction_sum / float(raw_count),
        },
        "latent_summary": _tensor_summary(q_tensor),
        "optical_power_summary": {
            "mean": [float(value) for value in power_tensor.mean(dim=0).tolist()],
            "min": [float(value) for value in power_tensor.min(dim=0).values.tolist()],
            "max": [float(value) for value in power_tensor.max(dim=0).values.tolist()],
        },
        "raw_cmos": {
            "finite": True,
            "nonconstant": raw_nonconstant,
            "min": raw_min,
            "max": raw_max,
            "std": max(0.0, raw_square_sum / float(raw_count) - (raw_sum / float(raw_count)) ** 2) ** 0.5,
        },
        "rows": int(development_indices.numel()),
        "role": DEV_ROLE,
    }


def _save_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_new_json(path, dict(payload))


def _save_new_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


def _load_references(path: Path, *, expected_data_identity: Mapping[str, Any]) -> FrozenBogtReferences:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("kind") != "FROZEN_BOGT_REFERENCES_V1":
        raise RunControlError("frozen references artifact schema is invalid")
    expected_reference_identity = dict(expected_data_identity)
    enriched_reference_identity = dict(expected_data_identity)
    enriched_reference_identity.update(
        {
            "role": TRAIN_ROLE,
            "count": 55_000,
            "canonical_order": "source_indices_in_input_order",
            "engineering_only": False,
        }
    )
    if payload.get("data_identity") not in (expected_reference_identity, enriched_reference_identity):
        raise RunControlError("frozen references data identity changed")
    references_payload = payload.get("references")
    if not isinstance(references_payload, Mapping):
        raise RunControlError("frozen references payload is missing")
    return FrozenBogtReferences.from_payload(references_payload)


def _load_weights(path: Path, *, expected_reference_identity: str) -> BogtLossWeights:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("kind") != "FROZEN_BOGT_WEIGHTS_V1":
        raise RunControlError("frozen weight artifact schema is invalid")
    weights_payload = payload.get("weights")
    if not isinstance(weights_payload, Mapping):
        raise RunControlError("frozen weight artifact is missing weights")
    weights = BogtLossWeights.from_payload(weights_payload)
    if weights.references_identity_sha256 != expected_reference_identity:
        raise RunControlError("frozen weight artifact is bound to another references identity")
    return weights


def _write_frozen_references(
    path: Path,
    references: FrozenBogtReferences,
    *,
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
) -> None:
    _save_new_torch(
        path,
        {
            "kind": "FROZEN_BOGT_REFERENCES_V1",
            "references": references.to_payload(),
            "contract": "s_j and r_p collected once from 55,000 canonical TRAIN rows at theta0",
            "source_manifest_id": source_manifest["source_manifest_id"],
            "run_identity": run_identity.to_dict(),
            "data_identity": dict(references.data_identity),
        },
        kind="frozen Chen BOGT references",
    )


def _write_frozen_weights(
    path: Path,
    weights: BogtLossWeights,
    *,
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
) -> None:
    _save_new_torch(
        path,
        {
            "kind": "FROZEN_BOGT_WEIGHTS_V1",
            "weights": weights.to_payload(),
            "contract": "one-shot inverse initial-gradient balancing on the first seed-1234 TRAIN batch",
            "source_manifest_id": source_manifest["source_manifest_id"],
            "run_identity": run_identity.to_dict(),
        },
        kind="frozen Chen BOGT weights",
    )


def _load_collection_chunks(target: Path, *, expected_rows: int) -> tuple[list[torch.Tensor], int]:
    chunks = sorted(target.glob("final_train_ols_chunk_*.pt"))
    values: list[torch.Tensor] = []
    completed = 0
    for path in chunks:
        payload = safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, Mapping) or payload.get("schema") != "chen_bogt_final_train_ols_chunk.v1":
            raise RunControlError(f"final OLS chunk schema is invalid: {path.name}")
        start = payload.get("start_row")
        z = payload.get("z")
        if start != completed or not isinstance(z, torch.Tensor) or not torch.is_complex(z) or z.ndim != 2 or z.shape[1] != 100:
            raise RunControlError("final OLS chunks are not contiguous canonical rows")
        if not torch.isfinite(z).all():
            raise TrainingContractError("final TRAIN OLS chunk is nonfinite")
        values.append(z.to(torch.complex64).contiguous())
        completed += int(z.shape[0])
    if completed > expected_rows:
        raise RunControlError("final TRAIN OLS chunks exceed the registered TRAIN role")
    return values, completed


def _collection_recovery_payload(state: Mapping[str, Any], *, data_identity: Mapping[str, Any]) -> dict[str, Any]:
    pending = state.get("pending_z")
    return {
        "kind": COLLECTION_RECOVERY_KIND,
        "phase": "POST_TRAIN_COLLECTION",
        "persisted_rows": int(state["persisted_rows"]),
        "next_row": int(state["next_row"]),
        "pending_start": int(state["pending_start"]),
        "pending_z": pending.detach().cpu().clone() if isinstance(pending, torch.Tensor) else torch.empty((0, 100), dtype=torch.complex64),
        "data_identity": dict(data_identity),
        "selected_step": 300,
    }


def _collect_final_train_ols_with_resilience(
    *,
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    images: torch.Tensor,
    train_indices: torch.Tensor,
    data_identity: Mapping[str, Any],
    target: Path,
    run_identity: ResearchRunIdentity,
    recovery: RecoveryCheckpointManager,
    ledger: DataReadLedger,
    resume_payload: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    chunk_values, persisted_rows = _load_collection_chunks(target, expected_rows=55_000)
    state: dict[str, Any] = {
        "persisted_rows": persisted_rows,
        "next_row": persisted_rows,
        "pending_start": persisted_rows,
        "pending_z": torch.empty((0, 100), dtype=torch.complex64),
    }
    if resume_payload is not None:
        if resume_payload.get("kind") != COLLECTION_RECOVERY_KIND or resume_payload.get("data_identity") != dict(data_identity):
            raise RunControlError("final OLS collection recovery identity changed")
        for key in ("persisted_rows", "next_row", "pending_start"):
            if not isinstance(resume_payload.get(key), int):
                raise RunControlError("final OLS collection recovery cursor is invalid")
        pending = resume_payload.get("pending_z")
        if not isinstance(pending, torch.Tensor) or pending.dtype != torch.complex64 or pending.ndim != 2 or pending.shape[1] != 100:
            raise RunControlError("final OLS collection recovery pending state is invalid")
        if int(resume_payload["persisted_rows"]) != persisted_rows:
            raise RunControlError("final OLS collection recovery disagrees with persisted chunks")
        state.update(
            {
                "persisted_rows": int(resume_payload["persisted_rows"]),
                "next_row": int(resume_payload["next_row"]),
                "pending_start": int(resume_payload["pending_start"]),
                "pending_z": pending.clone(),
            }
        )
        if state["next_row"] != state["pending_start"] + int(state["pending_z"].shape[0]):
            raise RunControlError("final OLS collection pending cursor is inconsistent")

    heartbeat = HeartbeatWriter(target, run_identity.run_id, total_steps=55_000, clock=clock)
    heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=state["next_row"], force=True)
    last_recovery_step = int(state["persisted_rows"])
    last_recovery_time = float(clock())
    last_recovery_checkpoint: Path | None = None

    def recovery_payload(_: int) -> Mapping[str, Any]:
        return _collection_recovery_payload(state, data_identity=data_identity)

    controller = InterruptController(
        heartbeat=heartbeat,
        recovery=recovery,
        payload_factory=recovery_payload,
        phase="POST_TRAIN_COLLECTION",
        total_steps=55_000,
    )
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    try:
        with controller:
            while state["next_row"] < 55_000:
                start = int(state["next_row"])
                row_indices = train_indices[start : start + 10]
                batch = images.index_select(0, row_indices).to(consumer.device)
                with torch.no_grad():
                    result = consumer.runtime.forward_from_image(batch)
                    z = result.z_ols.coefficients.detach().to(torch.complex64).cpu()
                if not torch.isfinite(z).all():
                    raise TrainingContractError("final TRAIN OLS became nonfinite")
                if state["pending_z"].numel() == 0:
                    state["pending_start"] = start
                    state["pending_z"] = z
                else:
                    state["pending_z"] = torch.cat((state["pending_z"], z), dim=0)
                state["next_row"] = start + int(row_indices.numel())
                ledger.record(TRAIN_ROLE, int(row_indices.numel()))
                if int(state["pending_z"].shape[0]) >= 1_000 or state["next_row"] == 55_000:
                    chunk_path = target / f"final_train_ols_chunk_{state['pending_start']:06d}.pt"
                    if not chunk_path.exists():
                        _save_new_torch(
                            chunk_path,
                            {
                                "schema": "chen_bogt_final_train_ols_chunk.v1",
                                "start_row": int(state["pending_start"]),
                                "z": state["pending_z"].clone(),
                                "data_identity": dict(data_identity),
                            },
                            kind="final TRAIN OLS collection chunk",
                        )
                    chunk_values.append(state["pending_z"].clone())
                    state["persisted_rows"] = state["next_row"]
                    state["pending_start"] = state["next_row"]
                    state["pending_z"] = torch.empty((0, 100), dtype=torch.complex64)
                now = float(clock())
                heartbeat.update(
                    state=RUNNING,
                    phase="POST_TRAIN_COLLECTION",
                    step=state["next_row"],
                    last_recovery_checkpoint=None if last_recovery_checkpoint is None else last_recovery_checkpoint.name,
                )
                if recovery_due(
                    step=state["next_row"],
                    last_recovery_step=last_recovery_step,
                    now=now,
                    last_recovery_time=last_recovery_time,
                    interval_steps=1_000,
                ):
                    last_recovery_checkpoint = recovery.save(recovery_payload(state["next_row"]), step=state["next_row"])
                    last_recovery_step = state["next_row"]
                    last_recovery_time = now
                    heartbeat.update(
                        state=RUNNING,
                        phase="POST_TRAIN_COLLECTION",
                        step=state["next_row"],
                        last_recovery_checkpoint=last_recovery_checkpoint.name,
                        force=True,
                    )
                interrupted = controller.handle_safe_boundary(
                    step=state["next_row"],
                    last_recovery_checkpoint=None if last_recovery_checkpoint is None else last_recovery_checkpoint.name,
                )
                if interrupted is not None:
                    return {
                        "interrupted": True,
                        "completed_rows": state["next_row"],
                        "recovery_checkpoint": interrupted.name,
                    }
    finally:
        consumer.runtime.train(was_training)
    z_final = torch.cat(chunk_values, dim=0)
    if z_final.shape != (55_000, 100):
        raise RunControlError("final TRAIN OLS collection shape is invalid")
    u_final = statistical_variable(z_final, consumer.references)
    if not torch.isfinite(u_final).all() or not bool((u_final >= 0.0).all().item()):
        raise TrainingContractError("final TRAIN OLS statistical variable is invalid")
    final_payload = {
        "schema": FINAL_COLLECTION_SCHEMA,
        "selected_step": 300,
        "z_final": z_final,
        "u_final": u_final,
        "source_indices": train_indices.clone(),
        "references_identity": consumer.references.identity() if consumer.references is not None else {},
        "data_identity": dict(data_identity),
        "canonical_order": "increasing source row order",
    }
    final_path = target / "final_train_ols.pt"
    if not final_path.exists():
        _save_new_torch(final_path, final_payload, kind="final TRAIN OLS collection")
    return {
        "interrupted": False,
        "path": final_path,
        "z_final": z_final,
        "u_final": u_final,
        "completed_rows": 55_000,
        "last_recovery_checkpoint": None if last_recovery_checkpoint is None else last_recovery_checkpoint.name,
    }


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunControlError(f"JSON artifact is unreadable: {path}") from error
    if not isinstance(payload, Mapping):
        raise RunControlError(f"JSON artifact is not a mapping: {path}")
    return dict(payload)


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path.resolve()
    try:
        relative = target.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RunControlError(f"artifact is outside repository: {target}") from error
    if not target.is_file():
        raise RunControlError(f"artifact is missing: {target}")
    return {"path": relative, "sha256": file_sha256(target), "role": role}


def _latest_recovery_payload(
    manager: RecoveryCheckpointManager,
    *,
    kind: str,
) -> tuple[Mapping[str, Any] | None, Path | None]:
    paths = sorted(
        manager.run_dir.glob("recovery_*.pt"),
        key=lambda path: path.name,
        reverse=True,
    )
    for path in paths:
        wrapper = manager.load(path)
        payload = wrapper.get("payload")
        if isinstance(payload, Mapping) and payload.get("kind") == kind:
            return payload, path
    return None, None


def _latest_training_resume_path(
    manager: RecoveryCheckpointManager,
) -> Path | None:
    paths = sorted(
        manager.run_dir.glob("recovery_*.pt"),
        key=lambda path: path.name,
        reverse=True,
    )
    for path in paths:
        wrapper = safe_torch_load(path, map_location="cpu")
        if not isinstance(wrapper, Mapping):
            raise RunControlError(f"recovery checkpoint wrapper is invalid: {path.name}")
        payload = wrapper.get("payload")
        if (
            isinstance(payload, Mapping)
            and payload.get("schema") == CHECKPOINT_SCHEMA
            and payload.get("operational_artifact") is True
            and payload.get("engineering_only") is False
        ):
            manager.load(path)
            return path
    checkpoints = sorted(manager.run_dir.glob("scientific_step_*.pt"), reverse=True)
    return checkpoints[0] if checkpoints else None


def _load_final_collection(
    path: Path,
    *,
    train_indices: torch.Tensor,
    data_identity: Mapping[str, Any],
    references: FrozenBogtReferences,
) -> dict[str, Any]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != FINAL_COLLECTION_SCHEMA:
        raise RunControlError("final TRAIN OLS artifact schema is invalid")
    z_final = payload.get("z_final")
    u_final = payload.get("u_final")
    source_indices = payload.get("source_indices")
    if (
        not isinstance(z_final, torch.Tensor)
        or z_final.dtype != torch.complex64
        or z_final.shape != (55_000, 100)
        or not isinstance(u_final, torch.Tensor)
        or u_final.dtype != torch.float64
        or u_final.shape != (55_000, 100)
        or not isinstance(source_indices, torch.Tensor)
        or source_indices.dtype != torch.long
        or not torch.equal(source_indices, train_indices)
    ):
        raise RunControlError("final TRAIN OLS artifact shape/order is invalid")
    if payload.get("data_identity") != dict(data_identity):
        raise RunControlError("final TRAIN OLS data identity changed")
    if payload.get("references_identity") != references.identity():
        raise RunControlError("final TRAIN OLS references identity changed")
    if not torch.isfinite(z_final).all() or not torch.isfinite(u_final).all() or not bool((u_final >= 0.0).all().item()):
        raise TrainingContractError("final TRAIN OLS artifact contains invalid values")
    return {"path": path, "z_final": z_final, "u_final": u_final, "completed_rows": 55_000}


def _load_phase_model(path: Path, *, references: FrozenBogtReferences) -> WrappedConditionalPhaseModel:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != PHASE_MODEL_ARTIFACT_SCHEMA:
        raise RunControlError("phase model artifact schema is invalid")
    phase_payload = payload.get("phase_model")
    if not isinstance(phase_payload, Mapping):
        raise RunControlError("phase model artifact is missing its model payload")
    required = ("coefficients", "residual_mean", "covariance", "cholesky", "scale_sha256", "fit_data_role", "fit_count", "engineering_only", "data_identity")
    if any(key not in phase_payload for key in required):
        raise RunControlError("phase model artifact fields are incomplete")
    model = WrappedConditionalPhaseModel(
        phase_payload["coefficients"],
        phase_payload["residual_mean"],
        phase_payload["covariance"],
        phase_payload["cholesky"],
        str(phase_payload["scale_sha256"]),
        str(phase_payload["fit_data_role"]),
        int(phase_payload["fit_count"]),
        bool(phase_payload["engineering_only"]),
        dict(phase_payload["data_identity"]),
        float(phase_payload.get("log_epsilon", 1.0e-12)),
        float(phase_payload.get("covariance_epsilon", 1.0e-8)),
    )
    if model.scale_sha256 != references.scale_sha256 or model.fit_data_role != TRAIN_ROLE or model.fit_count != 55_000 or model.engineering_only:
        raise RunControlError("phase model is not bound to the final TRAIN references")
    if not torch.allclose(model.cholesky, torch.tril(model.cholesky), rtol=0.0, atol=0.0) or not bool((torch.diag(model.cholesky) > 0.0).all().item()):
        raise RunControlError("phase model Cholesky factor is invalid")
    return model


def _phase_model_summary(model: WrappedConditionalPhaseModel) -> dict[str, Any]:
    reconstruction = model.cholesky @ model.cholesky.T
    covariance_error = float((reconstruction - model.covariance).abs().max().cpu())
    return {
        "schema": PHASE_MODEL_ARTIFACT_SCHEMA,
        "fit_data_role": model.fit_data_role,
        "fit_count": model.fit_count,
        "engineering_only": model.engineering_only,
        "scale_sha256": model.scale_sha256,
        "coefficients_sha256": tensor_sha256(model.coefficients),
        "residual_mean_sha256": tensor_sha256(model.residual_mean),
        "covariance_sha256": tensor_sha256(model.covariance),
        "cholesky_sha256": tensor_sha256(model.cholesky),
        "covariance_cholesky_max_abs_error": covariance_error,
        "finite": bool(
            torch.isfinite(model.coefficients).all()
            and torch.isfinite(model.residual_mean).all()
            and torch.isfinite(model.covariance).all()
            and torch.isfinite(model.cholesky).all()
        ),
        "deterministic_fit": True,
        "optimizer_used": False,
        "dev_used": False,
        "phase_family_retry": False,
    }


def _write_phase_model(
    path: Path,
    model: WrappedConditionalPhaseModel,
    *,
    root: Path,
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    final_collection: Path,
) -> None:
    _save_new_torch(
        path,
        {
            "schema": PHASE_MODEL_ARTIFACT_SCHEMA,
            "phase_model": model.to_payload(),
            "source_manifest_id": source_manifest["source_manifest_id"],
            "run_identity": run_identity.to_dict(),
            "final_collection": _artifact_ref(root, final_collection, role="phase-fit TRAIN source"),
        },
        kind="post-training wrapped conditional phase model",
    )


def _prior_payload() -> dict[str, Any]:
    prior = fixed_prior(device=torch.device("cpu"), dtype=torch.float64)
    return {
        "name": "POSITIVE_TRUNCATED_NORMAL_PRIOR_V1",
        "loc": prior.loc.clone(),
        "variance": prior.variance.clone(),
        "dtype": str(prior.loc.dtype),
    }


def _write_generation_bundle(
    *,
    path: Path,
    root: Path,
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    references: FrozenBogtReferences,
    weights: BogtLossWeights,
    phase_model: WrappedConditionalPhaseModel,
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    run_config: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    train_indices: torch.Tensor,
    final_collection_path: Path,
    sampler: DeterministicTrainRoleSampler,
) -> dict[str, Any]:
    transport = consumer.runtime.ols.transport.detach().cpu().clone()
    if not torch.equal(transport, torch.eye(100, dtype=torch.complex64)):
        raise TrainingContractError("bundle OLS transport is not T=I_100")
    trainable = consumer.trainable_state_dict()
    bundle = {
        "schema": GENERATION_BUNDLE_SCHEMA,
        "bundle_role": "GENERATION_INPUTS_COMPLETE",
        "generation_inputs_complete": True,
        "generation_executed": False,
        "generation_success": False,
        "novel_sample_generated": False,
        "promotion_authorized": False,
        "selected_step": 300,
        "contracts": frozen_contract_identity(),
        "source_manifest": dict(source_manifest),
        "run_identity": run_identity.to_dict(),
        "run_config": dict(run_config),
        "theta0_initialization": consumer.initialization_fingerprint(),
        "final_encoder_generator": {
            "state": {name: value.clone() for name, value in trainable.items()},
            "parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
            "selection": "scientific_step_000300",
            "frozen_after_selection": True,
        },
        "ols": {
            "interface": consumer.runtime_identity(),
            "transport": transport,
            "transport_sha256": tensor_sha256(transport),
            "transport_contract": "T=I_100",
            "writer": "W_OLS(z)=sum_j (Tz)_j v_j",
            "writer_operation": "lightgen_ols_coefficient_transmission_and_modal_reemission",
        },
        "frozen_references": {
            "identity": references.identity(),
            "scale": references.scale.clone(),
            "power_reference": references.power_reference.clone(),
            "s_definition": "mean_TRAIN |z_j(theta0)|^2 in canonical source-row order",
            "r_definition": "mean_TRAIN five contracted physical-plane powers at theta0",
        },
        "frozen_weights": weights.to_payload(),
        "prior": _prior_payload(),
        "phase_model": phase_model.to_payload(),
        "sampler": {
            "identity": {
                "seed": SCHEDULE_SEED,
                "batch_size": 10,
                "class_balanced": True,
                "rows_per_class": 1,
                "updates": SCIENTIFIC_UPDATES,
                "train_order": "sampler permutations over canonical TRAIN indices",
            },
            "final_state": sampler.state_dict(),
            "rng_semantics": {
                "training_checkpoint_rng": "scientific_step_000300.pt",
                "generation_requires_explicit_sampler_seed": True,
                "no_generation_rng_consumed": True,
            },
        },
        "train_split": {
            "identity": dict(data_identity),
            "source_indices": train_indices.clone(),
            "role": TRAIN_ROLE,
            "protected_test_read": False,
            "sealed_read": False,
        },
        "final_collection": _artifact_ref(root, final_collection_path, role="final TRAIN real OLS collection"),
        "no_generation_claim": "bundle completeness is not generation success",
    }
    _save_new_torch(path, bundle, kind="generation-ready input bundle")
    return {
        "path": path,
        "schema": GENERATION_BUNDLE_SCHEMA,
        "generation_inputs_complete": True,
        "generation_executed": False,
        "generation_success": False,
    }


def _training_summary(history: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = [dict(record) for record in history]
    if not records:
        return {"updates": 0, "trace": []}
    losses = [record["loss"] for record in records]
    return {
        "updates": len(records),
        "first": records[0],
        "final": records[-1],
        "minimum_losses": {
            name: min(float(loss[name]) for loss in losses)
            for name in ("L_dist", "L_rec", "L_op", "L_total")
        },
        "trace": records,
    }


def _sampler_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    permutations = state.get("permutations")
    if not isinstance(permutations, list) or not all(isinstance(value, torch.Tensor) for value in permutations):
        raise RunControlError("sampler state is not serializable for the report")
    return {
        "schema": state.get("schema"),
        "seed": state.get("seed"),
        "steps": state.get("steps"),
        "cursor": state.get("cursor"),
        "class_count": state.get("class_count"),
        "rows_per_class": state.get("rows_per_class"),
        "permutation_sha256": [tensor_sha256(value) for value in permutations],
    }


def _checkpoint_summary(target: Path, *, root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for step in CHECKPOINT_STEPS:
        path = target / f"scientific_step_{step:06d}.pt"
        if not path.is_file():
            raise RunControlError(f"required scientific checkpoint is missing: {path.name}")
        payload = safe_torch_load(path, map_location="cpu")
        if not isinstance(payload, Mapping) or payload.get("completed_updates") != step or payload.get("scientific_artifact") is not True:
            raise RunControlError(f"scientific checkpoint {path.name} is invalid")
        result.append({
            "step": step,
            "artifact": _artifact_ref(root, path, role="scientific checkpoint"),
            "completed_updates": step,
        })
    return result


def _read_dev_diagnostics(target: Path, *, root: Path) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for step in CHECKPOINT_STEPS:
        path = target / f"dev_diagnostics_step_{step:06d}.json"
        if not path.is_file():
            raise RunControlError(f"DEV diagnostic is missing: {path.name}")
        payload = _read_json_mapping(path)
        if payload.get("schema") != DEV_DIAGNOSTIC_SCHEMA or payload.get("step") != step:
            raise RunControlError(f"DEV diagnostic {path.name} is invalid")
        diagnostics[str(step)] = {
            "artifact": _artifact_ref(root, path, role="DEV diagnostic only"),
            "diagnostic": payload.get("diagnostic"),
        }
    return diagnostics


def _base_report(
    *,
    root: Path,
    target: Path,
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    authorization_id: str,
    run_config: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    ledger: DataReadLedger,
    git_before: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "task_id": NEXT_TASK,
        "outer_gate": OUTER_GATE,
        "outer_classification": None,
        "git": {
            "branch": git_before.get("branch"),
            "head": git_before.get("head"),
            "dirty_before": bool(git_before.get("worktree_dirty")),
        },
        "source_manifest_id": source_manifest.get("source_manifest_id"),
        "source_manifest": dict(source_manifest),
        "authorization_id": authorization_id,
        "run_identity": run_identity.to_dict(),
        "run_config": dict(run_config),
        "data_identity": dict(data_identity),
        "output_dir": str(target.relative_to(root).as_posix()),
        "project_mainline": "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION",
        "current_authorized_next_action": "NONE",
        "input_authorities": [
            "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION",
            "CHEN_BOGT_ON_COMPLEX_OLS_MODEL_CONTRACT_V1",
            "CHEN_BOGT_ON_COMPLEX_OLS_TRAINING_CONSUMER_CONTRACT_V1",
            "RESEARCH_FAST_PATH_AND_LONG_RUN_RESILIENCE_V1",
        ],
        "data_reads": ledger.to_dict(),
        "forbidden_data_reads": {
            "dev": ledger.counts.get(DEV_ROLE, 0),
            "protected": ledger.counts.get("protected", 0),
            "test": ledger.counts.get("test", 0),
            "sealed": ledger.counts.get("sealed", 0),
        },
        "scientific_updates": {
            "planned": SCIENTIFIC_UPDATES,
            "executed": 0,
            "training_role_only": True,
        },
        "training_contract": {
            "consumer": "LightGen2DChenBOGTComplexOLSTrainingConsumer",
            "trainable_parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
            "optimizer": {
                "name": "Adam",
                "learning_rate": 1.0e-3,
                "betas": list(ADAM_BETAS),
                "eps": ADAM_EPSILON,
                "weight_decay": WEIGHT_DECAY,
                "amsgrad": AMSGRAD,
                "autocast": False,
                "clipping": None,
                "scheduler": None,
                "dynamic_reweighting": False,
            },
        },
        "reads": {
            "development": "diagnostic only",
            "protected": "not read",
            "test": "not read",
            "sealed": "not read",
        },
        "scientific_facts_modified": False,
        "generation_claim": "no sampled generation and no promotion were executed",
    }


def _write_report(target: Path, report: Mapping[str, Any]) -> Path:
    path = target / "report.json"
    if not path.exists():
        _save_new_json(path, report)
    return path


def run_scientific_execution(
    *,
    repo_root: Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Execute or resume the one explicit CUDA Chen-BOGT bounded trajectory."""
    root = _repo_root() if repo_root is None else repo_root.resolve()
    fresh = resume is None
    requested_output = _resolve_under_root(root, output_dir)
    if fresh:
        target = requested_output
        if target.exists():
            raise RunControlError(f"fresh scientific output target already exists: {target}")
    else:
        resume_target = _resolve_under_root(root, resume)
        target = resume_target if resume_target.is_dir() else resume_target.parent
        if not target.is_dir():
            raise RunControlError(f"resume run directory is missing: {target}")
    outputs_root = (root / "outputs").resolve()
    if outputs_root not in target.parents or target in {outputs_root, outputs_root / "runs"}:
        raise RunControlError("scientific output must remain below a specific outputs run directory")
    if "final_pt" in target.parts or "_wsl_restore_metadata_v4" in target.parts:
        raise RunControlError("restored/final output namespace is forbidden")

    if not torch.cuda.is_available():
        raise RunControlError("the formal bounded run requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    ledger = DataReadLedger()
    images, labels, train_indices, development_indices, data_identity = _load_scientific_dataset(root)
    evaluator, evaluator_identity = _load_evaluator(root, device)
    run_config = _make_run_config(data_identity, evaluator_identity)
    source_repair: dict[str, Any] | None = None
    source_repair_pinned = False

    if fresh:
        git_before = git_identity(root)
        source_manifest_object = capture_execution_source_manifest(
            root,
            task_id=NEXT_TASK,
            semantic_authority_id=SEMANTIC_AUTHORITY_ID,
            relevant_paths=_execution_source_paths(root),
            config_identity=run_config,
        )
        source_manifest = source_manifest_object.to_dict()
        target.mkdir(parents=True, exist_ok=False)
        identity = ResearchRunIdentity(
            task_id=NEXT_TASK,
            run_id=SCIENTIFIC_RUN_ID,
            trajectory_id=SCIENTIFIC_TRAJECTORY_ID,
            semantic_authority_id=SEMANTIC_AUTHORITY_ID,
            data_role=TRAIN_ROLE,
            config_identity=run_config,
            model_identity="LIGHTGEN_2D_CHEN_BOGT_COMPLEX_OLS_CONSUMER_V1",
            optimizer_identity={
                "name": "Adam",
                "learning_rate": 1.0e-3,
                "betas": list(ADAM_BETAS),
                "eps": ADAM_EPSILON,
                "weight_decay": WEIGHT_DECAY,
                "amsgrad": AMSGRAD,
            },
            sampler_identity={
                "seed": SCHEDULE_SEED,
                "steps": SCIENTIFIC_UPDATES,
                "batch_size": 10,
                "class_balanced": True,
                "train_indices_sha256": data_identity["train_indices_sha256"],
            },
            execution_source_semantics_id=source_manifest["execution_source_semantics_id"],
            source_manifest_id=source_manifest["source_manifest_id"],
            scientific_attempt_id=f"{SCIENTIFIC_RUN_ID}:attempt_01",
        )
        authorization_id = f"{NEXT_TASK}:authorization:01"
        _save_new_json(target / "source_manifest.json", source_manifest)
        _save_new_json(
            target / "execution_manifest.json",
            {
                "schema": RUN_MANIFEST_SCHEMA,
                "task_id": NEXT_TASK,
                "outer_gate": OUTER_GATE,
                "input_authorities": [
                    "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION",
                    "CHEN_BOGT_ON_COMPLEX_OLS_MODEL_CONTRACT_V1",
                    "CHEN_BOGT_ON_COMPLEX_OLS_TRAINING_CONSUMER_CONTRACT_V1",
                    "RESEARCH_FAST_PATH_AND_LONG_RUN_RESILIENCE_V1",
                ],
                "authorization_id": authorization_id,
                "source_manifest": source_manifest,
                "run_identity": identity.to_dict(),
                "run_config": run_config,
                "data_identity": data_identity,
                "evaluator_identity": evaluator_identity,
                "git_before": git_before,
                "scope": {
                    "bounded_updates": SCIENTIFIC_UPDATES,
                    "generation_executed": False,
                    "promotion_authorized": False,
                },
            },
        )
    else:
        repaired_manifests = sorted(target.glob("execution_manifest_repaired*.json"))
        manifest_path = repaired_manifests[-1] if repaired_manifests else target / "execution_manifest.json"
        manifest_payload = _read_json_mapping(manifest_path)
        if manifest_payload.get("schema") != RUN_MANIFEST_SCHEMA:
            raise RunControlError("execution manifest schema is invalid")
        stored_repair = manifest_payload.get("engineering_repair")
        if isinstance(stored_repair, Mapping):
            source_repair = dict(stored_repair)
        source_manifest = manifest_payload.get("source_manifest")
        if not isinstance(source_manifest, Mapping):
            raise RunControlError("execution manifest source manifest is missing")
        source_manifest = dict(source_manifest)
        identity_payload = manifest_payload.get("run_identity")
        if not isinstance(identity_payload, Mapping):
            raise RunControlError("execution manifest run identity is missing")
        identity = ResearchRunIdentity.from_mapping(identity_payload)
        authorization_id = manifest_payload.get("authorization_id")
        if not isinstance(authorization_id, str) or not authorization_id:
            raise RunControlError("execution manifest authorization is missing")
        stored_data_identity = manifest_payload.get("data_identity")
        stored_evaluator_identity = manifest_payload.get("evaluator_identity")
        stored_run_config = manifest_payload.get("run_config")
        if not isinstance(stored_data_identity, Mapping) or not isinstance(stored_evaluator_identity, Mapping) or not isinstance(stored_run_config, Mapping):
            raise RunControlError("execution manifest input identities are incomplete")
        if dict(stored_data_identity) != data_identity or dict(stored_evaluator_identity) != evaluator_identity or dict(stored_run_config) != run_config:
            raise RunControlError("resume input identity changed")
        current_source = capture_execution_source_manifest(
            root,
            task_id=NEXT_TASK,
            semantic_authority_id=SEMANTIC_AUTHORITY_ID,
            relevant_paths=_execution_source_paths(root),
            config_identity=run_config,
        )
        if current_source.execution_source_semantics_id != source_manifest.get("execution_source_semantics_id"):
            pretraining_repair_eligible = (
                (target / "frozen_references.pt").is_file()
                and not (target / "frozen_weights.pt").exists()
                and not tuple(target.glob("scientific_step_*.pt"))
            )
            post_training_repair_eligible = (
                (target / "scientific_step_000300.pt").is_file()
                and not (target / "final_train_ols.pt").exists()
                and not (target / "phase_model.pt").exists()
                and not (target / "generation_ready_bundle.pt").exists()
            )
            if not pretraining_repair_eligible and not post_training_repair_eligible:
                raise RunControlError("resume execution source semantics changed")
            old_source_manifest = source_manifest
            if pretraining_repair_eligible:
                source_manifest = current_source.to_dict()
                identity = replace(
                    identity,
                    execution_source_semantics_id=source_manifest["execution_source_semantics_id"],
                    source_manifest_id=source_manifest["source_manifest_id"],
                )
            else:
                source_repair_pinned = True
            repair_files = sorted(target.glob("execution_source_repair*.json"))
            repair_sequence = len(repair_files) + 1
            source_repair = {
                "schema": "chen_bogt_pretraining_engineering_repair.v1",
                "sequence": repair_sequence,
                "reason": (
                    "register POST_TRAIN_COLLECTION as a valid heartbeat phase after step-300 training"
                    if post_training_repair_eligible
                    else "normalize CUDA device identity from cuda to cuda:0 before calibration/training"
                ),
                "scientific_updates_before_repair": 300 if post_training_repair_eligible else 0,
                "statistics_artifact_reused": True,
                "checkpoint_identity_pinned": post_training_repair_eligible,
                "old_source_manifest_id": old_source_manifest.get("source_manifest_id"),
                "old_execution_source_semantics_id": old_source_manifest.get("execution_source_semantics_id"),
                "new_source_manifest_id": current_source.source_manifest_id,
                "new_execution_source_semantics_id": current_source.execution_source_semantics_id,
                "old_source_manifest": old_source_manifest,
                "new_source_manifest": current_source.to_dict(),
            }
            repair_path = target / ("execution_source_repair.json" if repair_sequence == 1 else f"execution_source_repair_v{repair_sequence:02d}.json")
            _save_new_json(repair_path, source_repair)
            repaired_manifest = dict(manifest_payload)
            if pretraining_repair_eligible:
                repaired_manifest["source_manifest"] = source_manifest
                repaired_manifest["run_identity"] = identity.to_dict()
            repaired_manifest["engineering_repair"] = source_repair
            repaired_manifest_path = target / ("execution_manifest_repaired.json" if repair_sequence == 1 else f"execution_manifest_repaired_v{repair_sequence:02d}.json")
            _save_new_json(repaired_manifest_path, repaired_manifest)
        git_before = manifest_payload.get("git_before")
        if not isinstance(git_before, Mapping):
            git_before = {
                "branch": source_manifest.get("branch"),
                "head": source_manifest.get("head"),
                "worktree_dirty": bool(source_manifest.get("dirty")),
            }

    if identity.task_id != NEXT_TASK or identity.semantic_authority_id != SEMANTIC_AUTHORITY_ID or identity.data_role != TRAIN_ROLE:
        raise RunControlError("execution identity does not match the declared TRAIN task")
    if identity.source_manifest_id != source_manifest.get("source_manifest_id"):
        raise RunControlError("execution identity and source manifest are not bound")
    if identity.scientific_attempt_id is None:
        raise RunControlError("scientific attempt identity is missing")

    consumer = LightGen2DChenBOGTComplexOLSTrainingConsumer(device=device)
    consumer.assert_theta0()
    parameter_manifest = consumer.parameter_manifest()
    if parameter_manifest["optimizer_parameter_names"] != list(ALLOWED_TRAINABLE_PARAMETER_NAMES):
        raise RunControlError("runtime optimizer allowlist changed")
    if consumer.runtime.config.propagation_grid_size != PRODUCTION_PROPAGATION_GRID:
        raise RunControlError("formal execution did not construct the production 2048 grid")

    phase0_checks = {
        "schema": PHASE0_SCHEMA,
        "upstream_preflight_status": PREFLIGHT_READY,
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_device_index": torch.cuda.current_device(),
        "theta0": True,
        "trainable_parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
        "parameter_manifest": parameter_manifest,
        "data_role": TRAIN_ROLE,
        "protected_test_read": False,
        "sealed_read": False,
        "source_manifest_id": source_manifest["source_manifest_id"],
        "output_dir": str(target.relative_to(root).as_posix()),
        "output_writeable": True,
        "large_preflight_repeated": False,
        "scientific_tests_repeated": False,
    }
    phase0_path = target / "phase0_checks.json"
    if not phase0_path.exists():
        _save_new_json(phase0_path, phase0_checks)
    elif _read_json_mapping(phase0_path) != phase0_checks:
        existing_phase0 = _read_json_mapping(phase0_path)
        repair_phase0 = dict(phase0_checks)
        if source_repair is not None:
            repair_phase0["source_manifest_id"] = existing_phase0.get("source_manifest_id")
        if existing_phase0 != repair_phase0:
            raise RunControlError("phase-0 lightweight checks changed across resume")

    control = ResearchRunControl(identity, authorization_id)
    control.mark_preflight_ready()
    control.start_execution()
    effective_identity = control.begin_scientific_attempt(identity.scientific_attempt_id)
    recovery = RecoveryCheckpointManager(
        target,
        effective_identity,
        source_manifest_id=source_manifest["source_manifest_id"],
        authorization_id=authorization_id,
    )

    references_path = target / "frozen_references.pt"
    if references_path.exists():
        references = _load_references(references_path, expected_data_identity=data_identity)
        reference_phase = {"resumed_from_artifact": True, "artifact": _artifact_ref(root, references_path, role="frozen TRAIN statistics")}
    else:
        statistics_resume, statistics_recovery_path = _latest_recovery_payload(recovery, kind=STATISTICS_RECOVERY_KIND)
        statistics = _collect_statistics_with_resilience(
            consumer=consumer,
            images=images,
            train_indices=train_indices,
            data_identity=data_identity,
            target=target,
            run_identity=effective_identity,
            recovery=recovery,
            ledger=ledger,
            resume_payload=statistics_resume,
        )
        if statistics.get("interrupted"):
            return {
                "result": "INTERRUPTED",
                "phase": "STATISTICS",
                "completed_rows": statistics["completed_rows"],
                "recovery_checkpoint": statistics["recovery_checkpoint"],
                "run_id": identity.run_id,
                "trajectory_id": identity.trajectory_id,
                "scientific_attempt_id": identity.scientific_attempt_id,
                "output_dir": str(target.relative_to(root).as_posix()),
                "resume_command": f"--execute --resume {target.relative_to(root).as_posix()}",
            }
        references = statistics["references"]
        _write_frozen_references(
            references_path,
            references,
            source_manifest=source_manifest,
            run_identity=effective_identity,
        )
        reference_phase = {
            "resumed_from_artifact": False,
            "recovery_checkpoint": None if statistics_recovery_path is None else statistics_recovery_path.name,
            "artifact": _artifact_ref(root, references_path, role="frozen TRAIN statistics"),
            "s_summary": statistics["s_summary"],
            "r_power_summary": statistics["r_power_summary"],
        }
    consumer.bind_references(references, for_scientific=True)

    weights_path = target / "frozen_weights.pt"
    if weights_path.exists():
        weights = _load_weights(
            weights_path,
            expected_reference_identity=canonical_sha256(references.identity()),
        )
    else:
        calibration_sampler = DeterministicTrainRoleSampler(train_indices, labels)
        calibration_indices, calibration_labels = calibration_sampler.next_batch()
        calibration_identity = {
            "source": data_identity["source_path"],
            "seed": SCHEDULE_SEED,
            "batch_size": 10,
            "class_balanced": True,
            "rows_per_class": 1,
            "labels": [int(value) for value in calibration_labels.tolist()],
            "source_indices": [int(value) for value in calibration_indices.tolist()],
            "source_indices_sha256": tensor_sha256(calibration_indices),
            "train_indices_sha256": data_identity["train_indices_sha256"],
        }
        calibration_heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=300)
        calibration_heartbeat.update(state=RUNNING, phase="CALIBRATION", step=0, force=True)
        calibration_batch = images.index_select(0, calibration_indices).to(device)
        weights = calibrate_initial_gradient_weights(
            consumer,
            calibration_batch,
            references,
            data_role=TRAIN_ROLE,
            calibration_batch_identity=calibration_identity,
            engineering_only=False,
            ledger=ledger,
        )
        consumer.bind_weights(weights, for_scientific=True)
        _write_frozen_weights(
            weights_path,
            weights,
            source_manifest=source_manifest,
            run_identity=effective_identity,
        )
        calibration_heartbeat.update(
            state=RUNNING,
            phase="CALIBRATION",
            step=0,
            latest_losses={"gradient_norms": weights.gradient_norms},
            force=True,
        )
    consumer.bind_weights(weights, for_scientific=True)

    def batch_loader(indices: torch.Tensor, batch_labels: torch.Tensor) -> torch.Tensor:
        if not torch.equal(batch_labels, torch.arange(10, dtype=torch.long)):
            raise RunControlError("scientific batch labels are not the fixed class-balanced order")
        return images.index_select(0, indices).to(device)

    def on_checkpoint(
        step: int,
        checkpoint_consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
        history: Iterable[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        del history
        diagnostic_path = target / f"dev_diagnostics_step_{step:06d}.json"
        if diagnostic_path.exists():
            existing = _read_json_mapping(diagnostic_path)
            diagnostic = existing.get("diagnostic")
            if not isinstance(diagnostic, Mapping):
                raise RunControlError(f"existing DEV diagnostic is malformed: {diagnostic_path.name}")
            return dict(diagnostic)
        diagnostic = _evaluate_dev_checkpoint(
            consumer=checkpoint_consumer,
            evaluator=evaluator,
            images=images,
            labels=labels,
            development_indices=development_indices,
            references=references,
            ledger=ledger,
        )
        _save_new_json(
            diagnostic_path,
            {
                "schema": DEV_DIAGNOSTIC_SCHEMA,
                "step": step,
                "role": DEV_ROLE,
                "diagnostic": diagnostic,
                "threshold_is_diagnostic_only": True,
            },
        )
        return diagnostic

    sampler = DeterministicTrainRoleSampler(train_indices, labels)
    training_resume: Path | None = None
    if not fresh:
        selected_checkpoint = target / "scientific_step_000300.pt"
        training_resume = selected_checkpoint if selected_checkpoint.exists() else _latest_training_resume_path(recovery)
        if training_resume is None and tuple(target.glob("scientific_step_*.pt")):
            raise RunControlError("resume requested but no training checkpoint/recovery is available")
    theta0_trainable = consumer.trainable_state_dict()
    trajectory = run_bogt_scientific_trajectory(
        consumer=consumer,
        references=references,
        weights=weights,
        sampler=sampler,
        batch_loader=batch_loader,
        run_dir=target,
        run_identity=effective_identity,
        source_manifest=source_manifest,
        authorization_id=authorization_id,
        preflight_status=PREFLIGHT_READY,
        resume_checkpoint=training_resume,
        repo_root=root,
        ledger=ledger,
        on_checkpoint=on_checkpoint,
    )
    if trajectory.get("state") != RUN_COMPLETED:
        return {
            "result": "INTERRUPTED",
            "phase": "TRAINING",
            "completed_updates": trajectory.get("completed_updates", 0),
            "recovery_checkpoint": trajectory.get("recovery_checkpoint"),
            "run_id": identity.run_id,
            "trajectory_id": identity.trajectory_id,
            "scientific_attempt_id": identity.scientific_attempt_id,
            "output_dir": str(target.relative_to(root).as_posix()),
            "resume_command": f"--execute --resume {target.relative_to(root).as_posix()}",
        }
    history = list(trajectory["history"])
    if len(history) != SCIENTIFIC_UPDATES or trajectory.get("selected_step") != 300:
        raise TrainingContractError("scientific trajectory did not complete exactly 300 updates")
    training_phase_change = _phase_change(theta0_trainable, consumer.trainable_state_dict())

    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
    consumer.runtime.eval()
    final_collection_path = target / "final_train_ols.pt"
    if final_collection_path.exists():
        collection = _load_final_collection(
            final_collection_path,
            train_indices=train_indices,
            data_identity=data_identity,
            references=references,
        )
        collection_phase = {"resumed_from_artifact": True, "artifact": _artifact_ref(root, final_collection_path, role="final TRAIN real OLS collection")}
    else:
        collection_resume, collection_recovery_path = _latest_recovery_payload(recovery, kind=COLLECTION_RECOVERY_KIND)
        collection = _collect_final_train_ols_with_resilience(
            consumer=consumer,
            images=images,
            train_indices=train_indices,
            data_identity=data_identity,
            target=target,
            run_identity=effective_identity,
            recovery=recovery,
            ledger=ledger,
            resume_payload=collection_resume,
        )
        if collection.get("interrupted"):
            return {
                "result": "INTERRUPTED",
                "phase": "POST_TRAIN_COLLECTION",
                "completed_rows": collection["completed_rows"],
                "recovery_checkpoint": collection["recovery_checkpoint"],
                "run_id": identity.run_id,
                "trajectory_id": identity.trajectory_id,
                "scientific_attempt_id": identity.scientific_attempt_id,
                "output_dir": str(target.relative_to(root).as_posix()),
                "resume_command": f"--execute --resume {target.relative_to(root).as_posix()}",
            }
        collection_phase = {
            "resumed_from_artifact": False,
            "recovery_checkpoint": None if collection_recovery_path is None else collection_recovery_path.name,
            "artifact": _artifact_ref(root, final_collection_path, role="final TRAIN real OLS collection"),
        }
    z_final = collection["z_final"]
    u_final = collection["u_final"]
    final_u_summary = _tensor_summary(u_final)

    fit_heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=300)
    fit_heartbeat.update(state=RUNNING, phase="POST_TRAIN_FIT", step=300, force=True)
    phase_model_path = target / "phase_model.pt"
    phase_fit_error: str | None = None
    if phase_model_path.exists():
        phase_model = _load_phase_model(phase_model_path, references=references)
    else:
        phase_data_identity = {
            "role": TRAIN_ROLE,
            "count": 55_000,
            "source_artifact": "final_train_ols.pt",
            "source_artifact_sha256": file_sha256(final_collection_path),
            "references_identity": references.identity(),
            "gauge": "max-q with lowest-index ties",
            "phase": "wrapped angle; h=log(q+1e-12); X=[1,h]",
        }
        try:
            phase_model = fit_wrapped_conditional_phase_model(
                z_final,
                references,
                data_role=TRAIN_ROLE,
                engineering_only=False,
                data_identity=phase_data_identity,
            )
        except TrainingContractError as error:
            phase_fit_error = str(error)
            phase_model = None
        if phase_model is not None:
            _write_phase_model(
                phase_model_path,
                phase_model,
                root=root,
                source_manifest=source_manifest,
                run_identity=effective_identity,
                final_collection=final_collection_path,
            )

    report = _base_report(
        root=root,
        target=target,
        source_manifest=source_manifest,
        run_identity=effective_identity,
        authorization_id=authorization_id,
        run_config=run_config,
        data_identity=data_identity,
        ledger=ledger,
        git_before=git_before,
    )
    if source_repair is not None:
        report["engineering_repair"] = {
            "artifacts": [
                _artifact_ref(root, path, role="pre-training engineering repair record")
                for path in sorted(target.glob("execution_source_repair*.json"))
            ],
            "reason": source_repair["reason"],
            "statistics_artifact_reused": True,
            "scientific_updates_before_repair": source_repair["scientific_updates_before_repair"],
            "checkpoint_identity_pinned": source_repair_pinned,
            "old_source_manifest_id": source_repair["old_source_manifest_id"],
            "new_source_manifest_id": source_repair["new_source_manifest_id"],
        }
    report["scientific_updates"] = {
        "planned": SCIENTIFIC_UPDATES,
        "executed": len(history),
        "training_role_only": True,
        "selected_step": 300,
    }
    report["statistics"] = {
        "phase": "STATISTICS",
        "rows": references.count,
        "data_role": references.data_role,
        "engineering_only": references.engineering_only,
        "s_summary": _tensor_summary(references.scale),
        "r_power_summary": _tensor_summary(references.power_reference),
        "reference_identity": references.identity(),
        "artifact": _artifact_ref(root, references_path, role="frozen TRAIN statistics"),
        "phase_control": reference_phase,
    }
    report["weight_calibration"] = {
        "phase": "CALIBRATION",
        "weights": weights.identity(),
        "artifact": _artifact_ref(root, weights_path, role="frozen initial BOGT weights"),
        "weights_frozen_after_calibration": True,
        "dynamic_reweighting": False,
        "manual_fallback": False,
    }
    report["training"] = {
        "state": trajectory["state"],
        "completed_updates": trajectory["completed_updates"],
        "selected_step": trajectory["selected_step"],
        "scientific_attempt_id": trajectory["scientific_attempt_id"],
        "history": _training_summary(history),
        "phase_update_norms_from_theta0": training_phase_change,
        "optimizer": trajectory["optimizer_manifest"],
        "sampler_final_state": _sampler_summary(trajectory["sampler_state"]),
        "frozen_runtime_state_unchanged": consumer.frozen_state_sha256 == consumer._theta0_identity["frozen_runtime_state_sha256"],
        "trainable_parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
    }
    report["checkpoints"] = _checkpoint_summary(target, root=root)
    report["development_diagnostics"] = _read_dev_diagnostics(target, root=root)
    final_metrics = history[-1]
    report["distribution"] = {
        "L_dist_final_step_300": final_metrics["loss"]["L_dist"],
        "q_moments_final": final_metrics["q_moment_summary"],
        "prior": "fixed PositiveTruncatedNormal(loc=1, variance=0.25) across 100 ports",
    }
    report["reconstruction"] = {"L_rec_final_step_300": final_metrics["loss"]["L_rec"]}
    report["optical"] = {
        "L_op_final_step_300": final_metrics["loss"]["L_op"],
        "power_summary_final": final_metrics["power_summary"],
        "raw_cmos_diagnostic": "reported only at DEV checkpoints",
    }
    report["semantic"] = {
        "evaluator": evaluator_identity,
        "diagnostic_only": True,
        "threshold": SEMANTIC_DIAGNOSTIC_THRESHOLD,
        "per_checkpoint": report["development_diagnostics"],
    }
    report["interruptions_recovery"] = {
        "interrupted": False,
        "same_run_resume_supported": True,
        "recovery_interval_seconds": RECOVERY_INTERVAL_SECONDS,
        "recovery_interval_steps": RECOVERY_INTERVAL_STEPS,
        "rotation_keep": 3,
        "artifacts": [
            _artifact_ref(root, path, role="operational recovery checkpoint")
            for path in sorted(target.glob("recovery_*.pt"))
        ],
        "latest": None if recovery.latest() is None else recovery.latest().name,
    }
    report["final_collection"] = {
        "phase": "POST_TRAIN_COLLECTION",
        "selected_step": 300,
        "z_final_shape": list(z_final.shape),
        "z_final_dtype": str(z_final.dtype),
        "u_final_shape": list(u_final.shape),
        "u_final_dtype": str(u_final.dtype),
        "z_final_finite": bool(torch.isfinite(z_final).all().item()),
        "u_final_finite": bool(torch.isfinite(u_final).all().item()),
        "u_final_summary": final_u_summary,
        "canonical_order": "increasing source row order",
        "artifact": _artifact_ref(root, final_collection_path, role="final TRAIN real OLS collection"),
        "phase_control": collection_phase,
    }

    if phase_model is None:
        report["result"] = "VALID_COMPLETED_PHASE_MODEL_NOT_SUPPORTED"
        report["outer_classification"] = "B"
        report["phase_fit"] = {
            "supported": False,
            "error": phase_fit_error,
            "retry": False,
            "scientific_result_classification": "B",
        }
        report["bundle"] = {
            "generation_inputs_complete": False,
            "generation_executed": False,
            "reason": "phase model assumption was not supported; no phase-family retry",
        }
    else:
        report["phase_fit"] = {
            "supported": True,
            "model": _phase_model_summary(phase_model),
            "artifact": _artifact_ref(root, phase_model_path, role="post-training phase model"),
        }
        final_bundle_path = target / "generation_ready_bundle.pt"
        if final_bundle_path.exists():
            bundle_payload = safe_torch_load(final_bundle_path, map_location="cpu")
            if not isinstance(bundle_payload, Mapping) or bundle_payload.get("schema") != GENERATION_BUNDLE_SCHEMA or bundle_payload.get("generation_inputs_complete") is not True:
                raise RunControlError("existing generation bundle is invalid")
            bundle = {
                "path": final_bundle_path,
                "schema": GENERATION_BUNDLE_SCHEMA,
                "generation_inputs_complete": True,
                "generation_executed": False,
                "generation_success": False,
            }
        else:
            bundle = _write_generation_bundle(
                path=final_bundle_path,
                root=root,
                consumer=consumer,
                references=references,
                weights=weights,
                phase_model=phase_model,
                source_manifest=source_manifest,
                run_identity=effective_identity,
                run_config=run_config,
                data_identity=data_identity,
                train_indices=train_indices,
                final_collection_path=final_collection_path,
                sampler=sampler,
            )
        report["result"] = "VALID_COMPLETED_GENERATION_BUNDLE_READY"
        report["outer_classification"] = "A"
        report["bundle"] = {
            **{key: value for key, value in bundle.items() if key != "path"},
            "artifact": _artifact_ref(root, final_bundle_path, role="generation-ready input bundle"),
            "generation_inputs_complete": True,
            "generation_executed": False,
            "generation_success": False,
            "promotion_authorized": False,
        }
    report["next_future_task"] = "An explicitly authorized generation-consumer task may consume the bundle; this task executed no sampling."
    report["next_action_executed"] = False
    report["git"]["dirty_after"] = bool(git_identity(root)["worktree_dirty"])
    report_path = _write_report(target, report)
    final_heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=300)
    final_heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=300, force=True)
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the future graph without executing it")
    parser.add_argument("--preflight", action="store_true", help="run bounded synthetic/CPU engineering preflight")
    parser.add_argument("--execute", action="store_true", help="execute the one explicit CUDA scientific trajectory")
    parser.add_argument("--resume", help="resume the explicitly named run directory or checkpoint")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="new run directory below outputs/")
    args = parser.parse_args()
    selected_modes = sum(bool(value) for value in (args.dry_run, args.preflight, args.execute))
    if selected_modes != 1:
        parser.error("select exactly one of --dry-run, --preflight, or --execute")
    if args.resume is not None and not args.execute:
        parser.error("--resume is valid only with --execute")
    if args.dry_run:
        print(json.dumps(build_scientific_execution_graph(output_dir=args.output_dir), indent=2, sort_keys=True))
        return 0
    if args.preflight:
        result = run_engineering_preflight()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["preflight_status"] == PREFLIGHT_READY else 1
    result = run_scientific_execution(output_dir=args.output_dir, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("result") == "INTERRUPTED":
        return 2
    return 0 if result.get("outer_classification") in {"A", "B", "C"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
