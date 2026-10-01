"""Implement and preflight the LightGen complex-OLS training consumer.

The entry point intentionally runs only a short CPU engineering smoke.  It has
no long-training mode, no scientific evaluation mode, and no promotion path.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training import ALIGNMENT_RUNTIME_RELATIVE_PATH, ALIGNMENT_RUNTIME_SHA256, CLASSIFICATION, DatasetRoleMetadata, LightGen2DComplexOLSTrainingConsumer, PREFLIGHT_ID, SMOKE_PROPAGATION_GRID, TRAINING_STEPS, BalancedTrainRoleSampler, checkpoint_payload, cuda_capability_status, git_facts, load_frozen_training_contract, load_smoke_checkpoint, save_smoke_checkpoint, validate_legacy_exclusion
from ldm_project.v2.evidence import atomic_write_new_json, file_sha256, tensor_sha256
from ldm_project.v2.student.evaluator import load_local_mnist, stratified_mnist_indices


TASK_ID = "IMPLEMENT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V1"
PREFLIGHT_ARTIFACT_NAME = "lightgen_2d_semantic_complex_ols_training_preflight_input.v1.json"
SMOKE_CHECKPOINT_NAME = "lightgen_2d_semantic_complex_ols_training_smoke_checkpoint.v1.pt"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_consumer_v1_attempt_01"
)


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
    differences = [float((left[key] - right[key]).abs().max()) for key in left]
    return max(differences, default=0.0)


def _load_train_role(
    repo_root: Path,
    contract_payload: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    data_contract = contract_payload["data_contract"]
    if not isinstance(data_contract, Mapping):
        raise RuntimeError("training data contract is malformed")
    source_artifacts = data_contract["source_artifacts"]
    if not isinstance(source_artifacts, Mapping):
        raise RuntimeError("training source artifact manifest is malformed")
    training_ref = source_artifacts["data/MNIST/processed/training.pt"]
    if not isinstance(training_ref, Mapping):
        raise RuntimeError("training source artifact reference is malformed")
    training_path = repo_root / Path(str(training_ref["path"]))
    actual_training_sha = file_sha256(training_path)
    if actual_training_sha != training_ref["sha256"]:
        raise RuntimeError("local MNIST training source no longer matches the frozen contract")
    dataset_root = repo_root / Path(str(data_contract["root"]))
    loader_root = dataset_root.parent if dataset_root.name == "MNIST" else dataset_root
    if loader_root / "MNIST" != dataset_root:
        raise RuntimeError("the frozen MNIST root cannot be bound to the canonical loader")
    images, labels = load_local_mnist(loader_root, train=True)
    train_indices, development_indices = stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    if tuple(images.shape) != (60_000, 1, 28, 28) or labels.shape != (60_000,):
        raise RuntimeError("local MNIST source shape is outside the frozen contract")
    if train_indices.numel() != 55_000 or development_indices.numel() != 5_000:
        raise RuntimeError("local MNIST split counts are outside the frozen contract")
    if torch.unique(torch.cat((train_indices, development_indices))).numel() != 60_000:
        raise RuntimeError("train and development roles overlap or omit source rows")
    train_counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
    development_counts = torch.bincount(labels.index_select(0, development_indices), minlength=10)
    if not torch.equal(development_counts, torch.full((10,), 500, dtype=torch.long)) or int(train_counts.min()) < 100:
        raise RuntimeError("local MNIST split cannot support the frozen 100-step one-per-class schedule")
    base_roles = DatasetRoleMetadata().to_dict()
    dataset = {
        **base_roles,
        "loader": "ldm_project.v2.student.evaluator.load_local_mnist",
        "source_artifacts": {
            "training": {
                "path": str(training_ref["path"]),
                "expected_sha256": str(training_ref["sha256"]),
                "actual_sha256": actual_training_sha,
                "read": True,
            },
            "official_test": {
                "path": str(source_artifacts["data/MNIST/processed/test.pt"]["path"]),
                "contract_sha256": str(source_artifacts["data/MNIST/processed/test.pt"]["sha256"]),
                "read": False,
                "role": "protected heldout; not consumed",
            },
        },
        "split": {
            "algorithm": "stratified_mnist_indices(labels, seed=1234, validation_per_label=500)",
            "seed": 1234,
            "train_count": int(train_indices.numel()),
            "development_count": int(development_indices.numel()),
            "train_counts_by_class": [int(value) for value in train_counts.tolist()],
            "development_counts_by_class": [int(value) for value in development_counts.tolist()],
            "train_indices_sha256": tensor_sha256(train_indices),
            "development_indices_sha256": tensor_sha256(development_indices),
        },
        "heldout_consumed": False,
        "sealed_consumed": False,
    }
    return images, labels, train_indices, development_indices, dataset


def _smoke(
    *,
    output_dir: Path,
    contract: Any,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    dataset_roles: Mapping[str, Any],
    starting_git: Mapping[str, Any],
) -> dict[str, Any]:
    sampler = BalancedTrainRoleSampler(train_indices, labels)
    consumer = LightGen2DComplexOLSTrainingConsumer(
        contract,
        device=torch.device("cpu"),
        propagation_grid_size=SMOKE_PROPAGATION_GRID,
    )
    optimizer = consumer.create_optimizer()
    manifest = consumer.parameter_manifest()
    updates = []
    checkpoint_payload_value: Mapping[str, Any] | None = None
    checkpoint_path = output_dir / SMOKE_CHECKPOINT_NAME
    for global_step in range(3):
        indices, sampler_labels = sampler.next_batch()
        if not torch.equal(sampler_labels, torch.arange(10, dtype=torch.long)):
            raise RuntimeError("sampler label order is not the frozen 0..9 metadata order")
        batch = images.index_select(0, indices)
        update = consumer.update(batch, global_step=global_step, optimizer=optimizer)
        updates.append(update)
        if global_step == 1:
            checkpoint_payload_value = checkpoint_payload(
                consumer,
                optimizer,
                global_step=global_step,
                sampler_state=sampler.state_dict(),
                prior=update.loss.prior,
                dataset_roles=dataset_roles,
                git=starting_git,
            )
            save_smoke_checkpoint(checkpoint_path, checkpoint_payload_value)
    if checkpoint_payload_value is None:
        raise RuntimeError("smoke checkpoint was not created")

    resumed = LightGen2DComplexOLSTrainingConsumer(
        contract,
        device=torch.device("cpu"),
        propagation_grid_size=SMOKE_PROPAGATION_GRID,
    )
    resumed_optimizer = resumed.create_optimizer()
    loaded = load_smoke_checkpoint(checkpoint_path, resumed, resumed_optimizer)
    loaded_parameter_state = resumed.trainable_state_dict()
    saved_parameter_state = checkpoint_payload_value["trainable_parameter_state"]
    if not isinstance(saved_parameter_state, Mapping) or not _nested_equal(saved_parameter_state, loaded_parameter_state):
        raise RuntimeError("checkpoint reload did not preserve trainable parameter tensors")
    if not _nested_equal(checkpoint_payload_value["optimizer_state"], resumed_optimizer.state_dict()):
        raise RuntimeError("checkpoint reload did not preserve optimizer state")
    if not torch.equal(checkpoint_payload_value["rng_state"]["cpu"], torch.random.get_rng_state()):
        raise RuntimeError("checkpoint reload did not restore CPU RNG state")
    if not _nested_equal(checkpoint_payload_value["prior_rng"], loaded.prior_rng):
        raise RuntimeError("checkpoint reload did not preserve prior RNG state")

    resumed_sampler = BalancedTrainRoleSampler(train_indices, labels)
    resumed_sampler.load_state_dict(loaded.sampler_state)
    next_indices, next_labels = resumed_sampler.next_batch()
    # The original sampler has already consumed global_step=2.  Its third row
    # is retained in the update history by comparing with the deterministic
    # sampler restored from the checkpoint.
    reference_sampler = BalancedTrainRoleSampler(train_indices, labels)
    reference_sampler.load_state_dict(checkpoint_payload_value["sampler_state"])
    reference_indices, reference_labels = reference_sampler.next_batch()
    if not torch.equal(next_indices, reference_indices) or not torch.equal(next_labels, reference_labels):
        raise RuntimeError("checkpoint reload did not preserve sampler continuation")
    resumed_update = resumed.update(
        images.index_select(0, next_indices),
        global_step=2,
        optimizer=resumed_optimizer,
    )
    reference_update = updates[2]
    loss_differences = {
        name: abs(reference_update.loss.metrics()[name] - resumed_update.loss.metrics()[name])
        for name in reference_update.loss.metrics()
    }
    if any(value > 1.0e-5 for value in loss_differences.values()):
        raise RuntimeError(f"checkpoint resume loss mismatch: {loss_differences}")
    parameter_difference = _max_state_difference(consumer.trainable_state_dict(), resumed.trainable_state_dict())
    if parameter_difference > 1.0e-5:
        raise RuntimeError(f"checkpoint resume parameter mismatch: {parameter_difference}")

    checkpoint_payload_value = dict(checkpoint_payload_value)
    checkpoint_payload_value["reload_verified"] = True
    return {
        "propagation_grid": SMOKE_PROPAGATION_GRID,
        "updates_executed": len(updates),
        "global_steps": [update.global_step for update in updates],
        "batch_size": 10,
        "labels_used_as_model_input": False,
        "all_component_losses_finite": all(
            all(torch.isfinite(value) for value in (update.loss.image, update.loss.latent, update.loss.optical, update.loss.total))
            for update in updates
        ),
        "losses": [update.loss.metrics() for update in updates],
        "autograd": [update.autograd.to_dict() for update in updates],
        "parameter_updates": [dict(update.parameter_update_norms) for update in updates],
        "trainable_parameters_changed": all(
            any(value > 0.0 for value in update.parameter_update_norms.values()) for update in updates
        ),
        "frozen_runtime_state_unchanged": all(update.frozen_state_unchanged for update in updates),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "schema": "lightgen_2d_semantic_complex_ols_training_smoke_checkpoint.v1",
            "accepted_training_checkpoint": False,
            "safe_reload": True,
            "parameter_roundtrip": True,
            "optimizer_roundtrip": True,
            "rng_roundtrip": True,
            "prior_rng_roundtrip": True,
            "sampler_roundtrip": True,
            "next_step_loss_max_abs_difference": max(loss_differences.values(), default=0.0),
            "next_step_parameter_max_abs_difference": parameter_difference,
        },
        "parameter_manifest": manifest,
    }


def run_preflight(repo_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    root = repo_root.resolve()
    contract = load_frozen_training_contract(repo_root=root)
    runtime_path = root / ALIGNMENT_RUNTIME_RELATIVE_PATH
    if not runtime_path.is_file() or file_sha256(runtime_path) != ALIGNMENT_RUNTIME_SHA256:
        raise RuntimeError("aligned runtime source identity is unavailable or changed")
    if output_dir is None:
        target_dir = root / Path("outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_consumer_v1_attempt_01")
    else:
        target_dir = output_dir if output_dir.is_absolute() else root / output_dir
    target_dir = target_dir.resolve()
    if target_dir.exists():
        raise FileExistsError(f"refusing to overwrite preflight output directory: {target_dir}")
    starting_git = git_facts(root)
    legacy = validate_legacy_exclusion()
    images, labels, train_indices, _development_indices, dataset_roles = _load_train_role(root, contract.payload)
    target_dir.mkdir(parents=True, exist_ok=False)
    smoke = _smoke(
        output_dir=target_dir,
        contract=contract,
        images=images,
        labels=labels,
        train_indices=train_indices,
        dataset_roles=dataset_roles,
        starting_git=starting_git,
    )
    ending_git = git_facts(root)
    if starting_git["head"] != ending_git["head"] or starting_git["branch"] != ending_git["branch"]:
        raise RuntimeError("repository branch or HEAD changed during preflight")
    preflight_path = target_dir / PREFLIGHT_ARTIFACT_NAME
    payload = {
        "schema": "lightgen_2d_semantic_complex_ols_training_preflight_input.v1",
        "version": 1,
        "task_id": TASK_ID,
        "preflight_id": PREFLIGHT_ID,
        "status": "valid_completed",
        "result": "IMPLEMENTED",
        "classification": CLASSIFICATION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract.identity(root),
        "consumer": {
            "module": "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_complex_ols_training.py",
            "module_sha256": file_sha256(Path(__file__).parents[3] / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_complex_ols_training.py"),
            "entrypoint": "scripts/research/lightgen/execute_lightgen_2d_semantic_complex_ols_training_v1.py",
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "identity": TASK_ID,
        },
        "runtime": {
            "identity": "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1",
            "module": ALIGNMENT_RUNTIME_RELATIVE_PATH.as_posix(),
            "module_sha256": ALIGNMENT_RUNTIME_SHA256,
            "training_runtime_production_grid": 2048,
            "smoke_runtime_grid": SMOKE_PROPAGATION_GRID,
            "canonical_ols": "complex [B,100], 10x10, row-major j=10*r+c",
            "modal_projection": "z_j=<E_encoder,u_j>",
            "transport": "T=I_100 coefficient space",
            "writer": "W_OLS(z)=sum_j (Tz)_j v_j",
            "raw_output": "raw CMOS intensity |E_out|^2",
        },
        "git": {"starting": starting_git, "ending": ending_git},
        "trainable_parameter_manifest": smoke["parameter_manifest"],
        "optimizer": {
            "identity": "Adam",
            "learning_rate": 1.0e-3,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "gradient_clipping": None,
            "scheduler": None,
            "parameter_names": ["encoder.phase", "generator.phase"],
        },
        "dataset": dataset_roles,
        "prior": {
            "name": "PROJECT_COMPLEX_ISOTROPIC_GAUSSIAN_NATIVE_OLS_V1",
            "formula": "z_j=sigma_z/sqrt(2)*(epsilon_j_real+i*epsilon_j_imag)",
            "sigma_z": 1.0e-3,
            "shape": [10, 100],
            "dtype": "torch.complex64",
            "namespace": "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1_PRIOR",
            "seed_derivation": "SHA256(namespace|role|global_step|sample_count), first 8 bytes reduced to signed 63-bit seed",
            "sampling_path": "direct CPU generator -> complex tensor -> ComplexOLSState",
            "q_or_phase_dependency": False,
        },
        "loss": {
            "image": "mean((I_cmos_raw-I_target)^2), target=aligned 200x200 bilinear/clamp image adapter intensity",
            "latent": "biased B^2 V-stat full-complex Gaussian-kernel MMD on S(z) in R^200, l^2=200",
            "optical": {"included": False, "value": 0.0, "reason": "explicitly omitted by contract"},
            "total": "1.0*L_image + 1.0*L_latent + 0.0*L_optical",
            "adaptive_weighting": False,
        },
        "schedule": {
            "logical_batch_size": 10,
            "planned_steps": TRAINING_STEPS,
            "planned_epochs": None,
            "schedule_seed": 1234,
            "device_requirement": "cuda required for future long training",
            "smoke_device": "cpu",
            "checkpoint_steps": [50, 100],
        },
        "smoke": smoke,
        "legacy_exclusion": legacy,
        "environment": {
            "cuda": cuda_capability_status(),
            "limitations": [
                "only CPU engineering smoke was executed",
                "production 2048-grid future training was not executed",
                "CUDA long training remains unverified",
                "no heldout/protected evaluation was consumed",
                "consumed N3 sealed evidence was not read or reused",
            ],
        },
        "claim_boundary": {
            "engineering_smoke_only": True,
            "scientific_training_executed": False,
            "long_training_executed": False,
            "heldout_evaluation_executed": False,
            "generation_acceptance": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "sealed_reused": False,
        },
        "next": "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V1",
        "preflight_artifact_self_hash": "not_embedded_to_avoid_self_reference",
    }
    atomic_write_new_json(preflight_path, payload, artifact_kind="LightGen complex-OLS preflight input")
    result = {
        "result": "IMPLEMENTED",
        "classification": CLASSIFICATION,
        "preflight_id": PREFLIGHT_ID,
        "artifact": str(preflight_path),
        "artifact_sha256": file_sha256(preflight_path),
        "checkpoint": str(target_dir / SMOKE_CHECKPOINT_NAME),
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "next": payload["next"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="new preflight output directory")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_preflight(Path(__file__).resolve().parents[3], args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
