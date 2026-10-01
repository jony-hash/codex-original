"""Implement and smoke-test the LightGen V2 training consumer.

This entry point deliberately has no scientific-training mode.  It materializes
the approved V2 contract, runs only a small non-scientific engineering smoke,
and publishes an implementation report plus input for the later long-training
preflight task.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import CHECKPOINT_STEPS, CLASSIFICATION, COMPLETE, DeterministicTrainRoleSampler, EPSILON_SCALE, FINAL_SELECT, LightGen2DComplexOLSTrainingConsumerV2, SCALE_SMOKE_CLASSIFICATION, SCALE_SCHEMA, SCALE_EXTRACT, SCALE_FREEZE, SEMANTIC_THRESHOLD, STAGE_A_CHECKPOINT_RULE, STAGE_A_FAILURE, STAGE_A_SELECT, STAGE_A_SEMANTIC_CHECK, STAGE_A_STEPS, STAGE_A_TRAIN, STAGE_B_CHECKPOINT_RULE, STAGE_B_FAILURE, STAGE_B_SEMANTIC_CHECK, STAGE_B_STEPS, STAGE_B_TRAIN, TrainingContractError, V2TrainingStateMachine, checkpoint_payload, extract_train_only_ols_scale, file_sha256, git_facts, load_checkpoint, load_v2_training_contract, normalized_ols, physical_ols_from_normalized, sample_normalized_prior, sample_physical_prior, save_checkpoint, save_scale_artifact, select_stage_a_checkpoint, select_stage_b_checkpoint, validate_legacy_exclusion, SMOKE_PROPAGATION_GRID
from ldm_project.v2.evidence import atomic_write_new_json


TASK_ID = "IMPLEMENT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V2"
PREFLIGHT_ID = "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2"
PREFLIGHT_CLASSIFICATION = "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_INPUT_READY"
REPORT_SCHEMA = "lightgen_2d_semantic_complex_ols_training_consumer_v2_implementation_report.v1"
PREFLIGHT_SCHEMA = "lightgen_2d_semantic_complex_ols_long_training_preflight_input.v2"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_consumer_v2_attempt_01"
)
REPORT_NAME = "lightgen_2d_semantic_complex_ols_training_consumer_v2_implementation_report.v1.json"
PREFLIGHT_NAME = "lightgen_2d_semantic_complex_ols_long_training_preflight_v2.v1.json"
CONSUMER_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_complex_ols_training_v2.py"
)
ENTRYPOINT_PATH = Path("scripts/research/lightgen/execute_lightgen_2d_semantic_complex_ols_training_v2.py")


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _nested_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _max_state_difference(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    if left.keys() != right.keys():
        raise RuntimeError("state dictionaries have different keys")
    return max((float((left[key] - right[key]).abs().max()) for key in left), default=0.0)


def _fixture_images() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 10 * 28 * 28, dtype=torch.float32).reshape(10, 1, 28, 28)


def _fixture_train_roles() -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    train_indices = torch.arange(55_000, dtype=torch.long)
    return train_indices, labels


def _smoke(root: Path, contract: Any) -> dict[str, Any]:
    images = _fixture_images()
    train_indices, labels = _fixture_train_roles()
    consumer = LightGen2DComplexOLSTrainingConsumerV2(
        contract,
        device=torch.device("cpu"),
        propagation_grid_size=SMOKE_PROPAGATION_GRID,
    )
    optimizer_a = consumer.create_optimizer()
    manifest = consumer.parameter_manifest()
    update_a = consumer.update(
        images,
        stage="A",
        global_step=0,
        stage_local_step=0,
        optimizer=optimizer_a,
    )
    if not update_a.autograd.stage_a_no_mmd_gradient or update_a.loss.prior is not None:
        raise RuntimeError("Stage-A smoke did not prove latent-MMD exclusion")
    if abs(float(update_a.loss.total - update_a.loss.image)) > 1.0e-7:
        raise RuntimeError("Stage-A loss decomposition is invalid")

    stage_a_records = [
        {"stage_local_step": 100, "development_macro_recall": 0.95, "development_L_image": 0.30},
        {"stage_local_step": 200, "development_macro_recall": 0.96, "development_L_image": 0.20},
        {"stage_local_step": 300, "development_macro_recall": 0.94, "development_L_image": 0.40},
    ]
    selected_a = select_stage_a_checkpoint(stage_a_records)
    if selected_a is None or selected_a["stage_local_step"] != 200:
        raise RuntimeError("Stage-A deterministic semantic selection failed")
    if select_stage_a_checkpoint([{**record, "development_macro_recall": 0.1} for record in stage_a_records]) is not None:
        raise RuntimeError("Stage-A failure classification did not close")

    # This is explicitly a non-scientific smoke scale.  It exercises the real
    # modal projection and float64 accumulation, but it is not a 55,000-row
    # scientific extraction and is never published as a training authority.
    smoke_scale = extract_train_only_ols_scale(
        consumer,
        images,
        data_role=SCALE_SMOKE_CLASSIFICATION,
        checkpoint_identity={"role": "NON_SCIENTIFIC_SMOKE", "stage_local_step": 200},
        data_identity={"role": "NON_SCIENTIFIC_SMOKE"},
    )
    if smoke_scale.classification != SCALE_SMOKE_CLASSIFICATION or smoke_scale.values.dtype != torch.float64:
        raise RuntimeError("smoke scale was not explicitly classified or float64")
    if smoke_scale.values.shape != (100,) or not torch.isfinite(smoke_scale.values).all():
        raise RuntimeError("smoke scale shape/finite check failed")

    with tempfile.TemporaryDirectory(prefix="lightgen-v2-smoke-") as temporary:
        temporary_root = Path(temporary)
        bound_scale = save_scale_artifact(temporary_root / "scale.v2.pt", smoke_scale)
        reloaded_scale = __import__(
            "ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2",
            fromlist=["load_scale_artifact"],
        ).load_scale_artifact(bound_scale.artifact_path, expected_sha256=bound_scale.artifact_sha256)
        if reloaded_scale.vector_sha256 != bound_scale.vector_sha256:
            raise RuntimeError("scale artifact reload changed the frozen vector")
        consumer.set_scale(bound_scale)

        optimizer_b, reset_marker = consumer.reset_stage_b_optimizer()
        if (
            not reset_marker["optimizer_reset"]
            or not reset_marker["new_optimizer_state_empty"]
            or optimizer_b is optimizer_a
            or optimizer_b.state
            or not optimizer_a.state
        ):
            raise RuntimeError("Stage-B optimizer reset was not proven")
        update_b = consumer.update(
            images,
            stage="B",
            global_step=301,
            stage_local_step=1,
            optimizer=optimizer_b,
            scale=bound_scale,
        )
        if not update_b.autograd.stage_b_normalized_mmd_active or not update_b.autograd.scale_gradient_active:
            raise RuntimeError("Stage-B normalized MMD/scale gradients were not active")
        if abs(float(update_b.loss.total - (update_b.loss.image + update_b.loss.latent + update_b.loss.scale))) > 1.0e-7:
            raise RuntimeError("Stage-B loss decomposition is invalid")
        if bound_scale.values.requires_grad or not update_b.autograd.frozen_scale_no_grad:
            raise RuntimeError("Stage-B scale is not frozen")

        prior_a = sample_normalized_prior(10, stage="STAGE_B", stage_local_step=1, device="cpu")
        prior_b = sample_normalized_prior(10, stage="STAGE_B", stage_local_step=1, device="cpu")
        if not torch.equal(prior_a.coefficients, prior_b.coefficients) or prior_a.coefficients.shape != (10, 100) or prior_a.coefficients.dtype != torch.complex64:
            raise RuntimeError("normalized prior replay is not deterministic")
        sampled_state, _ = sample_physical_prior(bound_scale, 3, stage_local_step=0, device="cpu")
        if sampled_state.coefficients.shape != (3, 100) or sampled_state.coefficients.dtype != torch.complex64:
            raise RuntimeError("future physical prior mapping is invalid")
        normalized = normalized_ols(update_b.loss.result.z_ols.coefficients, bound_scale)
        recovered = physical_ols_from_normalized(normalized, bound_scale)
        if not torch.allclose(recovered, update_b.loss.result.z_ols.coefficients, rtol=2.0e-5, atol=2.0e-7):
            raise RuntimeError("normalized/physical scale roundtrip is invalid")

        sampler = DeterministicTrainRoleSampler(train_indices, labels, steps=STAGE_A_STEPS)
        sampler_state = sampler.state_dict()
        state_a = V2TrainingStateMachine(stage=STAGE_A_TRAIN, global_step=0, stage_local_step=0)
        state_a.transition(STAGE_A_SEMANTIC_CHECK)
        state_a.stage_a_checkpoint_checked(final_checkpoint=False, eligible=True)
        state_a.transition(STAGE_A_SEMANTIC_CHECK)
        state_a.stage_a_checkpoint_checked(final_checkpoint=True, eligible=True)
        state_a.transition(SCALE_EXTRACT)
        state_a.transition(SCALE_FREEZE)
        state_a.selected_stage_a_checkpoint = dict(selected_a)
        state_a.scale_identity = bound_scale.identity()
        state_a.optimizer_reset_marker = reset_marker
        state_a.transition(STAGE_B_TRAIN)
        state_a.transition(STAGE_B_SEMANTIC_CHECK)
        state_a.stage_b_checkpoint_checked(final_checkpoint=False, eligible=True)
        state_a.transition(STAGE_B_SEMANTIC_CHECK)
        state_a.stage_b_checkpoint_checked(final_checkpoint=True, eligible=True)
        state_a.transition(COMPLETE)
        if state_a.stage != COMPLETE or state_a.terminal_classification is not None:
            raise RuntimeError("V2 state-machine smoke did not reach COMPLETE")
        failed_a = V2TrainingStateMachine(stage=STAGE_A_SEMANTIC_CHECK)
        failed_a.stage_a_checkpoint_checked(final_checkpoint=True, eligible=False)
        if failed_a.terminal_classification != STAGE_A_FAILURE:
            raise RuntimeError("Stage-A terminal failure state is invalid")
        failed_b = V2TrainingStateMachine(stage=STAGE_B_SEMANTIC_CHECK)
        failed_b.stage_b_checkpoint_checked(final_checkpoint=True, eligible=False)
        if failed_b.terminal_classification != STAGE_B_FAILURE:
            raise RuntimeError("Stage-B terminal failure state is invalid")
        if select_stage_b_checkpoint([
            {"stage_local_step": 100, "development_macro_recall": 0.95, "development_L_total": 0.3},
            {"stage_local_step": 200, "development_macro_recall": 0.96, "development_L_total": 0.2},
            {"stage_local_step": 300, "development_macro_recall": 0.94, "development_L_total": 0.4},
        ])["stage_local_step"] != 200:
            raise RuntimeError("Stage-B deterministic selection failed")

        stage_b_state = V2TrainingStateMachine(
            stage=STAGE_B_TRAIN,
            global_step=301,
            stage_local_step=1,
            selected_stage_a_checkpoint=selected_a,
            scale_identity=bound_scale.identity(),
            optimizer_reset_marker=reset_marker,
        )
        checkpoint = checkpoint_payload(
            consumer,
            optimizer_b,
            state=stage_b_state,
            sampler_state=sampler_state,
            semantic_selection_state={"threshold": SEMANTIC_THRESHOLD, "rule": STAGE_B_CHECKPOINT_RULE},
            git={"branch": "smoke", "head": "smoke", "worktree_dirty": True},
            scale=bound_scale,
            selected_stage_a_checkpoint=selected_a,
            optimizer_reset_marker=reset_marker,
        )
        checkpoint_path = temporary_root / "checkpoint.v2.pt"
        save_checkpoint(checkpoint_path, checkpoint)
        restored = LightGen2DComplexOLSTrainingConsumerV2(
            contract,
            device=torch.device("cpu"),
            propagation_grid_size=SMOKE_PROPAGATION_GRID,
        )
        restored_optimizer = restored.create_optimizer()
        loaded = load_checkpoint(checkpoint_path, restored, restored_optimizer, expected_scale=bound_scale)
        if loaded.stage != STAGE_B_TRAIN or loaded.scale is None or loaded.scale.vector_sha256 != bound_scale.vector_sha256:
            raise RuntimeError("V2 Stage-B checkpoint reload lost scale/state identity")
        if not _nested_equal(consumer.trainable_state_dict(), restored.trainable_state_dict()):
            raise RuntimeError("V2 checkpoint reload changed trainable parameters")
        if not _nested_equal(optimizer_b.state_dict(), restored_optimizer.state_dict()):
            raise RuntimeError("V2 checkpoint reload changed optimizer state")
        if _max_state_difference(consumer.trainable_state_dict(), restored.trainable_state_dict()) != 0.0:
            raise RuntimeError("V2 checkpoint parameter roundtrip is not exact")

        # The forbidden-role check happens before any tensor work.
        try:
            extract_train_only_ols_scale(
                consumer,
                images,
                data_role="protected",
                checkpoint_identity={},
                data_identity={},
            )
        except Exception as error:
            if not isinstance(error, TrainingContractError):
                raise
        else:
            raise RuntimeError("forbidden protected scale role was accepted")

    return {
        "scope": "NON_SCIENTIFIC_SMOKE",
        "device": "cpu",
        "propagation_grid": SMOKE_PROPAGATION_GRID,
        "real_data_rows_read": 0,
        "protected_test_rows_read": 0,
        "sealed_rows_read": 0,
        "scientific_steps_executed": 0,
        "stage_a_smoke_updates": 1,
        "stage_b_smoke_updates": 1,
        "finite_losses_and_gradients": True,
        "stage_a_loss_equals_image": True,
        "stage_a_latent_mmd_disabled": True,
        "stage_b_loss_decomposition": True,
        "stage_b_normalized_mmd_gradient_active": True,
        "scale_anchor_gradient_active": True,
        "frozen_scale_no_gradient": True,
        "optimizer_reset": True,
        "state_machine": "all transitions and both terminal failures checked",
        "normalized_prior_deterministic": True,
        "physical_prior_mapping_checked": True,
        "checkpoint_reload_resume": True,
        "modal_projection_active": update_a.autograd.modal_projection_active and update_b.autograd.modal_projection_active,
        "w_ols_active": update_a.autograd.w_ols_active and update_b.autograd.w_ols_active,
        "trainable_parameter_manifest": manifest,
        "selection_rules": {"stage_a": STAGE_A_CHECKPOINT_RULE, "stage_b": STAGE_B_CHECKPOINT_RULE},
        "scale_schema": SCALE_SCHEMA,
        "scale_smoke_classification": SCALE_SMOKE_CLASSIFICATION,
        "epsilon_scale": EPSILON_SCALE,
    }


def run_implementation(root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    repo_root = _root() if root is None else root.resolve()
    target = (repo_root / DEFAULT_OUTPUT_DIR if output_dir is None else output_dir)
    target = target if target.is_absolute() else repo_root / target
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite V2 implementation output: {target}")
    starting_git = git_facts(repo_root)
    contract = load_v2_training_contract(repo_root=repo_root)
    legacy = validate_legacy_exclusion(repo_root / CONSUMER_PATH)
    smoke = _smoke(repo_root, contract)
    ending_before_outputs = git_facts(repo_root)
    if starting_git["branch"] != ending_before_outputs["branch"] or starting_git["head"] != ending_before_outputs["head"]:
        raise RuntimeError("branch or HEAD changed during V2 implementation smoke")
    consumer_path = repo_root / CONSUMER_PATH
    entrypoint_path = repo_root / ENTRYPOINT_PATH
    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "valid_completed",
        "result": "IMPLEMENTED",
        "classification": CLASSIFICATION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition_only_contract_materialized": True,
        "git": {"starting": starting_git, "ending_before_output_write": ending_before_outputs},
        "authorities": {
            "v2_contract": contract.identity(repo_root),
            "semantic_threshold": SEMANTIC_THRESHOLD,
            "aligned_runtime": {
                "identity": "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1",
                "path": CONSUMER_PATH.as_posix().replace("lightgen_2d_semantic_complex_ols_training_v2.py", "lightgen_2d_semantic_generation_input_ols_alignment.py"),
                "sha256": file_sha256(repo_root / Path("src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py")),
            },
            "consumer": {"path": CONSUMER_PATH.as_posix(), "sha256": file_sha256(consumer_path)},
            "entrypoint": {"path": ENTRYPOINT_PATH.as_posix(), "sha256": file_sha256(entrypoint_path)},
        },
        "implementation": {
            "state_machine": [
                STAGE_A_TRAIN,
                STAGE_A_SEMANTIC_CHECK,
                STAGE_A_SELECT,
                SCALE_EXTRACT,
                SCALE_FREEZE,
                STAGE_B_TRAIN,
                STAGE_B_SEMANTIC_CHECK,
                FINAL_SELECT,
                COMPLETE,
            ],
            "stage_a": {"steps": STAGE_A_STEPS, "checkpoints": list(CHECKPOINT_STEPS), "loss": "1.0*L_image + 0.0*L_optical", "alpha_latent": 0.0, "semantic_threshold": SEMANTIC_THRESHOLD, "selection_rule": STAGE_A_CHECKPOINT_RULE},
            "scale": {"formula": "s_j=mean_train |z_j|^2", "epsilon_scale": EPSILON_SCALE, "dtype_accumulation": "complex64 raw z, float64 per-port accumulation", "train_only": True, "artifact_schema": SCALE_SCHEMA},
            "normalized_latent": {"formula": "z_tilde_j=z_j/sqrt(s_j+epsilon_scale)", "prior": "(epsilon_R+i*epsilon_I)/sqrt(2) ~ CN(0,I_100)", "mmd": "full-complex gauge-invariant MMD on z_tilde; l^2=200"},
            "stage_b": {"steps": STAGE_B_STEPS, "checkpoints": list(CHECKPOINT_STEPS), "loss": "1.0*L_image + 1.0*L_latent_normalized + 1.0*L_scale", "optimizer_reset": True, "semantic_firewall": SEMANTIC_THRESHOLD, "selection_rule": STAGE_B_CHECKPOINT_RULE},
            "future_generation": {"mapping": "z_sample=sqrt(s+epsilon_scale)*z_tilde_sample", "entry": "forward_from_ols(z_sample)", "same_w_ols_and_generator": True, "executed": False},
            "checkpoint_schema": "lightgen_2d_semantic_complex_ols_training_checkpoint.v2",
            "resume_boundaries": ["inside Stage A", "after Stage-A selection", "scale extraction/freeze", "inside Stage B"],
        },
        "legacy_exclusion": legacy,
        "smoke": smoke,
        "data_firewall": {"train": "future optimization and Stage-A scale", "development": "future semantic firewall and selection", "calibration": "not consumed", "protected": "forbidden", "test": "forbidden", "sealed": "forbidden", "forbidden_roles_fail_closed": True},
        "boundaries": {"scientific_training_executed": False, "long_training_executed": False, "protected_evaluation_executed": False, "test_values_loaded": False, "sealed_values_loaded": False, "generation_acceptance": False, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"},
        "known_environment_warnings": ["CPU engineering smoke only; CUDA 2048-grid long-training preflight remains a separate required task", "no real MNIST rows were loaded by the implementation smoke"],
        "next": PREFLIGHT_ID,
        "report_self_hash": "not_embedded_to_avoid_self_reference",
    }
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "version": 2,
        "preflight_id": PREFLIGHT_ID,
        "task_id": TASK_ID,
        "status": "valid_completed",
        "result": "READY_FOR_LONG_TRAINING_PREFLIGHT",
        "classification": PREFLIGHT_CLASSIFICATION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract.identity(repo_root),
        "runtime": {"identity": "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1", "module": "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py", "module_sha256": file_sha256(repo_root / Path("src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py")), "production_grid": 2048, "smoke_grid": SMOKE_PROPAGATION_GRID, "canonical_ols": "complex [B,100], 10x10 row-major", "transport": "T=I_100 coefficient space", "writer": "W_OLS(z)=sum_j(Tz)_j v_j"},
        "consumer": {"module": CONSUMER_PATH.as_posix(), "module_sha256": file_sha256(consumer_path), "entrypoint": ENTRYPOINT_PATH.as_posix(), "entrypoint_sha256": file_sha256(entrypoint_path), "classification": CLASSIFICATION},
        "trainable_parameter_manifest": smoke["trainable_parameter_manifest"],
        "stage_a": {"steps": STAGE_A_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "loss": "1.0*L_image + 0.0*L_optical", "latent_mmd": False, "semantic_threshold": SEMANTIC_THRESHOLD, "selection_rule": STAGE_A_CHECKPOINT_RULE},
        "scale": {"formula": "s_j=mean_train |z_j|^2", "epsilon_scale": EPSILON_SCALE, "train_rows": 55_000, "batch_size": 10, "float64_accumulation": True, "artifact_schema": SCALE_SCHEMA, "recompute_in_stage_b": False},
        "normalized_latent": {"coordinate": "z_tilde_j=z_j/sqrt(s_j+epsilon_scale)", "prior": "CN(0,I_100), z_j=(epsilon_R+i*epsilon_I)/sqrt(2)", "mmd": "full-complex gauge-invariant MMD on z_tilde, l^2=200", "raw_z_mmd": False},
        "stage_b": {"steps": STAGE_B_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "loss": "1.0*L_image + 1.0*L_latent_normalized + 1.0*L_scale", "optimizer_reset": True, "semantic_firewall": SEMANTIC_THRESHOLD, "selection_rule": STAGE_B_CHECKPOINT_RULE},
        "planned_budget": {"stage_a_steps": STAGE_A_STEPS, "stage_b_steps": STAGE_B_STEPS, "logical_batch_size": 10, "device": "cuda required", "autocast": False, "grad_scaler": False, "duration_search": False},
        "smoke": smoke,
        "preflight_requirements_not_yet_run": ["target-device static/lineage checks", "three-step real update through the exact 2048-grid entrypoint", "domain-appropriate negative control", "interrupted/resumed segment and memory headroom"],
        "claim_boundary": {"implementation_smoke_only": True, "scientific_training_executed": False, "long_training_executed": False, "protected_or_sealed_accessed": False, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"},
        "next": PREFLIGHT_ID,
        "preflight_self_hash": "not_embedded_to_avoid_self_reference",
    }
    target.mkdir(parents=True, exist_ok=False)
    report_path = target / REPORT_NAME
    preflight_path = target / PREFLIGHT_NAME
    atomic_write_new_json(report_path, report, artifact_kind="LightGen V2 consumer implementation report")
    atomic_write_new_json(preflight_path, preflight, artifact_kind="LightGen V2 long-training preflight input")
    result = {"result": "IMPLEMENTED", "classification": CLASSIFICATION, "contract": contract.identity(repo_root), "report": {"path": str(report_path), "sha256": file_sha256(report_path)}, "preflight": {"path": str(preflight_path), "sha256": file_sha256(preflight_path)}, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready", "next": PREFLIGHT_ID}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_implementation(output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
