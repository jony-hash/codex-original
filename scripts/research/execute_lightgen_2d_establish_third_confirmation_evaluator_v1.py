"""Establish one preregistered, independent canonical-digit evaluator.

This consumer trains exactly one fixed evaluator on a deterministic split of the
canonical MNIST training role.  It never loads a Generator, optical model,
generated CMOS, DEV output, P2/Q2 bank, official test split, or sealed data.
The final epoch is the only checkpoint candidate and qualification is measured
once on the frozen canonical qualification split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

# PyTorch requires this process-level setting for deterministic CUDA matmul
# operations when deterministic algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.application.research.research_run_control import (
    capture_execution_source_manifest,
)
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import stratified_mnist_indices


TASK_ID = "ESTABLISH_THIRD_CONFIRMATION_EVALUATOR_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_ESTABLISH_THIRD_CONFIRMATION_EVALUATOR_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "THIRD_CONFIRMATION_EVALUATOR_AUTHORITY_ESTABLISHED"
CLASSIFICATION_SUPPORTED = "THIRD_CONFIRMATION_EVALUATOR_AUTHORITY_ESTABLISHED"
CLASSIFICATION_QUALIFICATION_FAILED = "THIRD_CONFIRMATION_EVALUATOR_QUALIFICATION_FAILED"
CLASSIFICATION_INVALID_HARNESS = "INVALID_HARNESS"
SEMANTIC_AUTHORITY_ID = "THIRD_CONFIRMATION_EVALUATOR_V1"
RUN_ID = "third-confirmation-evaluator-v1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260919_third_confirmation_evaluator_v1/runs/attempt_01"
)
DEFAULT_DATASET_ROOT = Path("data/MNIST")
TRAINING_SOURCE_RELATIVE = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"

PARENT_SPLIT_SEED = 1234
PARENT_VALIDATION_PER_LABEL = 500
QUALIFICATION_SPLIT_SEED = 2026091902
QUALIFICATION_PER_CLASS = 550
TRAINING_SEED = 2026091903
TRAINING_EPOCHS = 20
TRAINING_BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0

# The current project acceptance evaluator contract is applicable to this
# canonical digit confirmation role.  The fallback thresholds in the prepared
# plan are deliberately not used because they are weaker than this authority.
QUALIFICATION_POLICY_ID = "current_acceptance_evaluator_secondary_validation_v1"
QUALIFICATION_MIN_ACCURACY = 0.985
QUALIFICATION_MIN_MACRO_RECALL = 0.985
QUALIFICATION_MIN_CLASS_RECALL = 0.95

ARCHITECTURE_SCHEMA = "third_confirmation_evaluator_architecture.v1"
PREPROCESSING_SCHEMA = "third_confirmation_evaluator_preprocessing.v1"
SPLIT_SCHEMA = "third_confirmation_evaluator_split.v1"
TRAINING_SCHEMA = "third_confirmation_evaluator_training_config.v1"
PREFLIGHT_SCHEMA = "third_confirmation_evaluator_preflight.v1"
CHECKPOINT_SCHEMA = "third_confirmation_evaluator_checkpoint.v1"
QUALIFICATION_SCHEMA = "third_confirmation_evaluator_qualification.v1"
PROVENANCE_SCHEMA = "third_confirmation_evaluator_provenance.v1"
REPORT_SCHEMA = "third_confirmation_evaluator_report.v1"


class ThirdConfirmationEvaluator(nn.Module):
    """The fixed fallback E_C architecture from the prepared contract."""

    architecture_kind = "third_confirmation_canonical_cnn_v1"

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("E_C input must have shape [B,1,28,28]")
        features = self.embedding(self.features(images))
        return features, self.classifier(features)

    def freeze(self) -> "ThirdConfirmationEvaluator":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


class EvaluatorExecutionError(RuntimeError):
    """Raised when the prepared evaluator consumer cannot proceed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluatorExecutionError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _new_model(*, device: torch.device, seed: int) -> ThirdConfirmationEvaluator:
    _seed_everything(seed)
    return ThirdConfirmationEvaluator().to(device)


def _new_optimizer(model: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )


def _state_hash(model: nn.Module) -> str:
    entries = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value.detach().cpu()),
        }
        for name, value in sorted(model.state_dict().items())
    ]
    return canonical_sha256({"state_dict": entries})


def _architecture_payload() -> dict[str, Any]:
    return {
        "schema": ARCHITECTURE_SCHEMA,
        "architecture_kind": ThirdConfirmationEvaluator.architecture_kind,
        "input": {"shape": [1, 28, 28], "range": "[0,1]"},
        "layers": [
            {"type": "Conv2d", "in_channels": 1, "out_channels": 32, "kernel": 5, "padding": 2},
            {"type": "ReLU"},
            {"type": "MaxPool2d", "kernel": 2},
            {"type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel": 5, "padding": 2},
            {"type": "ReLU"},
            {"type": "MaxPool2d", "kernel": 2},
            {"type": "Flatten"},
            {"type": "Linear", "in_features": 64 * 7 * 7, "out_features": 128},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 128, "out_features": 10},
        ],
        "forbidden": ["dropout", "batch_norm", "residual_blocks", "pretrained_weights"],
        "class_order": list(range(10)),
        "prediction_rule": "argmax(logits,dim=1)",
    }


def _preprocessing_payload() -> dict[str, Any]:
    return {
        "schema": PREPROCESSING_SCHEMA,
        "input_domain": "canonical MNIST train images",
        "value_contract": "float32 values in [0,1] from uint8/255",
        "shape": [1, 28, 28],
        "normalization": "none beyond canonical uint8/255 conversion",
        "augmentation": "none",
        "synthetic_cmos": False,
        "optical_model": False,
    }


def _load_canonical_train_only(dataset_root: Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    source = dataset_root / "processed" / "training.pt"
    _require(source.is_file(), f"canonical TRAIN source is missing: {source}")
    source_sha256 = file_sha256(source)
    _require(source_sha256 == TRAINING_SOURCE_SHA256, "canonical TRAIN source SHA256 changed")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise EvaluatorExecutionError("canonical TRAIN source cannot be read") from error
    _require(isinstance(payload, (tuple, list)) and len(payload) == 2, "canonical TRAIN source schema changed")
    images, labels = payload
    _require(isinstance(images, torch.Tensor) and isinstance(labels, torch.Tensor), "canonical TRAIN source tensors missing")
    _require(images.ndim == 3 and tuple(images.shape[1:]) == (28, 28), "canonical TRAIN images have unexpected shape")
    _require(images.dtype == torch.uint8 and tuple(labels.shape) == (60_000,) and labels.dtype == torch.long, "canonical TRAIN dtypes or count changed")
    _require(images.shape[0] == 60_000, "canonical TRAIN row count changed")
    _require(int(images.min()) >= 0 and int(images.max()) <= 255, "canonical TRAIN image range changed")
    _require(int(labels.min()) >= 0 and int(labels.max()) <= 9, "canonical TRAIN labels changed")
    return images.unsqueeze(1).to(torch.float32).div_(255.0), labels, source_sha256


def _parent_and_ec_split(
    dataset_root: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    images, labels, source_sha256 = _load_canonical_train_only(dataset_root)
    _require(images.ndim == 4 and tuple(images.shape[1:]) == (1, 28, 28), "canonical images have unexpected shape")
    _require(labels.ndim == 1 and images.shape[0] == labels.numel() == 60_000, "canonical MNIST train role has unexpected size")
    _require(float(images.min()) >= 0.0 and float(images.max()) <= 1.0, "canonical images are outside [0,1]")

    parent_train, parent_source_validation = stratified_mnist_indices(
        labels, seed=PARENT_SPLIT_SEED, validation_per_label=PARENT_VALIDATION_PER_LABEL
    )
    _require(parent_train.numel() == 55_000 and parent_source_validation.numel() == 5_000, "canonical parent split changed")

    parent_labels = labels.index_select(0, parent_train)
    split_generator = torch.Generator(device="cpu").manual_seed(QUALIFICATION_SPLIT_SEED)
    fit_parts: list[torch.Tensor] = []
    qualification_parts: list[torch.Tensor] = []
    for class_id in range(10):
        matches = torch.nonzero(parent_labels == class_id, as_tuple=False).flatten()
        _require(matches.numel() > QUALIFICATION_PER_CLASS, f"class {class_id} lacks E_C qualification rows")
        shuffled = matches.index_select(0, torch.randperm(matches.numel(), generator=split_generator))
        qualification_parts.append(shuffled[:QUALIFICATION_PER_CLASS])
        fit_parts.append(shuffled[QUALIFICATION_PER_CLASS:])

    fit_relative = torch.cat(fit_parts).sort().values
    qualification_relative = torch.cat(qualification_parts).sort().values
    fit_source = parent_train.index_select(0, fit_relative).sort().values
    qualification_source = parent_train.index_select(0, qualification_relative).sort().values
    _require(fit_source.numel() == 49_500 and qualification_source.numel() == 5_500, "E_C split counts changed")
    _require(not bool(torch.isin(fit_source, qualification_source).any().item()), "E_C fit/qualification overlap")

    fit_images = images.index_select(0, fit_source)
    fit_labels = labels.index_select(0, fit_source)
    qualification_images = images.index_select(0, qualification_source)
    qualification_labels = labels.index_select(0, qualification_source)
    qualification_counts = torch.bincount(qualification_labels, minlength=10)
    _require(qualification_counts.tolist() == [QUALIFICATION_PER_CLASS] * 10, "E_C qualification is not class-balanced")

    parent_payload = {
        "seed": PARENT_SPLIT_SEED,
        "validation_per_label": PARENT_VALIDATION_PER_LABEL,
        "train_count": int(parent_train.numel()),
        "source_validation_count": int(parent_source_validation.numel()),
        "train_indices_sha256": tensor_sha256(parent_train),
        "source_validation_indices_sha256": tensor_sha256(parent_source_validation),
        "dataset_content_sha256": canonical_sha256(
            {"images": tensor_sha256(images), "labels": tensor_sha256(labels)}
        ),
        "source_file": {
            "path": TRAINING_SOURCE_RELATIVE.as_posix(),
            "sha256": source_sha256,
            "schema": "60000x28x28 uint8 images plus 60000 int64 labels",
        },
    }
    manifest = {
        "schema": SPLIT_SCHEMA,
        "role": "canonical_train_only_evaluator_fit_qualification",
        "parent": parent_payload,
        "selection": {
            "seed": QUALIFICATION_SPLIT_SEED,
            "algorithm": "per-class deterministic shuffle; 550 qualification rows/class; remaining parent rows fit; source-index sort",
            "qualification_per_class": QUALIFICATION_PER_CLASS,
            "fit_count": int(fit_source.numel()),
            "qualification_count": int(qualification_source.numel()),
            "fit_counts_by_class": torch.bincount(fit_labels, minlength=10).tolist(),
            "qualification_counts_by_class": qualification_counts.tolist(),
            "fit_source_indices_sha256": tensor_sha256(fit_source),
            "qualification_source_indices_sha256": tensor_sha256(qualification_source),
        },
        "forbidden_roles": {
            "dev_generated_cmos": 0,
            "optical_outputs": 0,
            "p2_q2": 0,
            "official_test": 0,
            "protected": 0,
            "sealed": 0,
        },
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return fit_images, fit_labels, qualification_images, qualification_labels, {
        "manifest": manifest,
        "fit_source_indices": fit_source,
        "qualification_source_indices": qualification_source,
        "parent_train_indices": parent_train,
        "source_validation_indices": parent_source_validation,
    }


def _epoch_order(count: int, *, seed: int, epoch: int) -> torch.Tensor:
    return torch.randperm(count, generator=torch.Generator(device="cpu").manual_seed(seed + epoch))


def _run_updates(
    model: ThirdConfirmationEvaluator,
    optimizer: torch.optim.Adam,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    seed: int,
    max_updates: int | None = None,
    capture_update_trace: bool = False,
) -> dict[str, Any]:
    model.train()
    update_count = 0
    epoch_trace: list[dict[str, Any]] = []
    update_trace: list[dict[str, Any]] = []
    for epoch in range(epochs):
        order = _epoch_order(images.shape[0], seed=seed, epoch=epoch)
        epoch_loss = 0.0
        epoch_batches = 0
        for start in range(0, order.numel(), TRAINING_BATCH_SIZE):
            if max_updates is not None and update_count >= max_updates:
                return {"updates": update_count, "epochs": epoch_trace, "update_trace": update_trace}
            batch_indices = order[start : start + TRAINING_BATCH_SIZE]
            batch_images = images.index_select(0, batch_indices).to(device)
            batch_labels = labels.index_select(0, batch_indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(batch_images)
            loss = F.cross_entropy(logits, batch_labels)
            _finite_tensor(loss, "E_C training loss")
            loss.backward()
            gradient_norm = math.sqrt(
                sum(
                    float(parameter.grad.detach().square().sum().item())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
            _require(math.isfinite(gradient_norm) and gradient_norm > 0.0, "E_C gradient is zero or nonfinite")
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            epoch_batches += 1
            update_count += 1
            if capture_update_trace:
                update_trace.append(
                    {
                        "update": update_count,
                        "epoch": epoch,
                        "loss": float(loss.detach().item()),
                        "gradient_l2": gradient_norm,
                    }
                )
        epoch_trace.append(
            {
                "epoch": epoch + 1,
                "updates": update_count,
                "mean_fit_loss": epoch_loss / max(epoch_batches, 1),
                "batches": epoch_batches,
            }
        )
    return {"updates": update_count, "epochs": epoch_trace, "update_trace": update_trace}


def _preflight(
    fit_images: torch.Tensor,
    fit_labels: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    _require(device.type == "cuda", "formal E_C execution requires CUDA")
    _require(torch.cuda.is_available(), "CUDA is unavailable for formal E_C execution")
    valid_model = _new_model(device=device, seed=TRAINING_SEED)
    valid_optimizer = _new_optimizer(valid_model)
    valid_before = _state_hash(valid_model)
    valid_run = _run_updates(
        valid_model,
        valid_optimizer,
        fit_images,
        fit_labels,
        device=device,
        epochs=1,
        seed=TRAINING_SEED,
        max_updates=3,
        capture_update_trace=True,
    )
    valid_after = _state_hash(valid_model)
    valid_changed = valid_before != valid_after

    negative_model = _new_model(device=device, seed=TRAINING_SEED)
    negative_optimizer = _new_optimizer(negative_model)
    negative_labels = fit_labels.add(1).remainder(10)
    negative_run = _run_updates(
        negative_model,
        negative_optimizer,
        fit_images,
        negative_labels,
        device=device,
        epochs=1,
        seed=TRAINING_SEED,
        max_updates=3,
        capture_update_trace=True,
    )
    negative_state_changed = valid_after != _state_hash(negative_model)
    valid_last_loss = float(valid_run["update_trace"][-1]["loss"])
    negative_last_loss = float(negative_run["update_trace"][-1]["loss"])
    negative_control_passed = bool(negative_state_changed and abs(valid_last_loss - negative_last_loss) > 1.0e-9)

    status = bool(valid_run["updates"] == 3 and valid_changed and negative_control_passed)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY" if status else "INVALID_HARNESS",
        "device": str(device),
        "static_contract": {
            "architecture": ThirdConfirmationEvaluator.architecture_kind,
            "trainable_groups": ["E_C.parameters"],
            "frozen_upstream_groups": [],
            "generated_cmos_reads": 0,
            "optical_forward_calls": 0,
            "protected_test_sealed_reads": 0,
        },
        "three_step_real_update": {
            "updates": valid_run["updates"],
            "parameters_departed": valid_changed,
            "trace": valid_run["update_trace"],
        },
        "negative_control": {
            "kind": "deterministic_cyclic_label_shift",
            "updates": negative_run["updates"],
            "state_differs_from_valid_label_run": negative_state_changed,
            "valid_last_loss": valid_last_loss,
            "shifted_label_last_loss": negative_last_loss,
            "passed": negative_control_passed,
        },
        "no_scientific_20_epoch_training": True,
    }


def _qualification_metrics(
    model: ThirdConfirmationEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], TRAINING_BATCH_SIZE):
            _, logits = model(images[start : start + TRAINING_BATCH_SIZE].to(device))
            _finite_tensor(logits, "E_C qualification logits")
            predictions.append(logits.argmax(dim=1).cpu())
    predicted = torch.cat(predictions)
    confusion = torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
    denominators = confusion.sum(dim=1)
    per_class_recall = [float(confusion[i, i].item() / denominators[i].item()) for i in range(10)]
    accuracy = float((predicted == labels).float().mean().item())
    macro_recall = float(sum(per_class_recall) / 10.0)
    predicted_histogram = torch.bincount(predicted, minlength=10).tolist()
    passed = bool(
        accuracy >= QUALIFICATION_MIN_ACCURACY
        and macro_recall >= QUALIFICATION_MIN_MACRO_RECALL
        and min(per_class_recall) >= QUALIFICATION_MIN_CLASS_RECALL
    )
    return {
        "schema": QUALIFICATION_SCHEMA,
        "policy": {
            "id": QUALIFICATION_POLICY_ID,
            "source": "src/ldm_project/v2/application/current/measurement.py::secondary_validation",
            "min_accuracy": QUALIFICATION_MIN_ACCURACY,
            "min_macro_recall": QUALIFICATION_MIN_MACRO_RECALL,
            "min_class_recall": QUALIFICATION_MIN_CLASS_RECALL,
        },
        "split_role": "E_C_QUALIFICATION",
        "evaluation_calls": 1,
        "count": int(labels.numel()),
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion.tolist(),
        "predicted_class_histogram": predicted_histogram,
        "passed": passed,
    }


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    return {"path": _relative(path, root), "role": role, "sha256": file_sha256(path)}


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_write_new_json(path, dict(payload), artifact_kind=kind)


def _write_split_artifact(target: Path, split_data: Mapping[str, Any]) -> dict[str, Any]:
    split_indices_path = target / "split_indices.pt"
    atomic_torch_save_new(
        split_indices_path,
        {
            "schema": "third_confirmation_evaluator_split_indices.v1",
            "parent_train_indices": split_data["parent_train_indices"].cpu(),
            "source_validation_indices": split_data["source_validation_indices"].cpu(),
            "fit_source_indices": split_data["fit_source_indices"].cpu(),
            "qualification_source_indices": split_data["qualification_source_indices"].cpu(),
        },
        artifact_kind="E_C split indices",
    )
    manifest = dict(split_data["manifest"])
    manifest["split_indices_artifact"] = {
        "path": split_indices_path.name,
        "sha256": file_sha256(split_indices_path),
    }
    _write_json(target / "split_manifest.json", manifest, kind="E_C split manifest")
    return manifest


def _checkpoint_payload(
    model: ThirdConfirmationEvaluator,
    *,
    split_manifest: Mapping[str, Any],
    training_config: Mapping[str, Any],
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
        "architecture_kind": ThirdConfirmationEvaluator.architecture_kind,
        "architecture": _architecture_payload(),
        "preprocessing": _preprocessing_payload(),
        "class_order": list(range(10)),
        "prediction_rule": "argmax(logits,dim=1)",
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "split_manifest_hash": split_manifest["manifest_hash"],
        "training_config": dict(training_config),
        "qualification": dict(qualification),
        "independence": {
            "role": "INDEPENDENT_CONFIRMATION_ONLY",
            "used_in_generator_loss": False,
            "used_in_gradient_attribution": False,
            "used_in_hyperparameter_selection": False,
            "used_on_generated_cmos_in_this_task": False,
            "frozen_before_future_optical_training": True,
        },
    }


def _training_config_payload() -> dict[str, Any]:
    return {
        "schema": TRAINING_SCHEMA,
        "epochs": TRAINING_EPOCHS,
        "batch_size": TRAINING_BATCH_SIZE,
        "seed": TRAINING_SEED,
        "order": "torch.randperm(fit_count, cpu_generator.manual_seed(seed+epoch))",
        "optimizer": {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPSILON,
            "weight_decay": WEIGHT_DECAY,
        },
        "objective": "cross_entropy(logits,canonical_digit_label)",
        "checkpoint_selection": "final_epoch_only",
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "architecture_selection": "none",
        "seed_sweep": False,
        "optimizer_sweep": False,
        "lr_sweep": False,
        "future_cmos_selection": False,
    }


def _prepare_output(target: Path) -> None:
    _require(not target.exists(), f"refusing to reuse existing E_C attempt directory: {target}")
    target.mkdir(parents=True, exist_ok=False)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    _require(args.authorization_token == AUTHORIZATION_TOKEN, "authorization token mismatch")
    _require(args.device == "cuda", "formal E_C execution requires --device cuda")
    _require(torch.cuda.is_available(), "formal E_C execution requires CUDA")
    _require(args.epochs == TRAINING_EPOCHS, "E_C training horizon is fixed at 20 epochs")
    _require(args.batch_size == TRAINING_BATCH_SIZE, "E_C batch size is fixed at 256")
    _require(args.qualification_per_class == QUALIFICATION_PER_CLASS, "E_C qualification split is fixed at 550/class")

    target = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    _prepare_output(target)
    device = torch.device(args.device)
    dataset_root = (root / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root).resolve()

    training_config = _training_config_payload()
    source_manifest = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=[
            Path(__file__),
            root / "src/ldm_project/v2/student/evaluator.py",
            root / "src/ldm_project/v2/application/current/measurement.py",
            root / "src/ldm_project/v2/application/_runtime_shared.py",
            root / "src/ldm_project/v2/evidence/io.py",
            root / TRAINING_SOURCE_RELATIVE,
        ],
        config_identity={
            "architecture": ThirdConfirmationEvaluator.architecture_kind,
            "split_seed": QUALIFICATION_SPLIT_SEED,
            "training": training_config,
            "qualification_policy": QUALIFICATION_POLICY_ID,
        },
    )
    _write_json(target / "source_manifest.json", source_manifest.to_dict(), kind="E_C source manifest")
    _write_json(target / "architecture.json", _architecture_payload(), kind="E_C architecture schema")
    _write_json(target / "preprocessing.json", _preprocessing_payload(), kind="E_C preprocessing contract")
    _write_json(target / "training_config.json", training_config, kind="E_C training configuration")

    fit_images, fit_labels, qualification_images, qualification_labels, split_data = _parent_and_ec_split(dataset_root)
    split_manifest = _write_split_artifact(target, split_data)
    preflight = _preflight(fit_images, fit_labels, device=device)
    _write_json(target / "preflight.json", preflight, kind="E_C preflight")
    _require(preflight["status"] == "READY", "E_C preflight did not reach READY")

    model = _new_model(device=device, seed=TRAINING_SEED)
    optimizer = _new_optimizer(model)
    training = _run_updates(
        model,
        optimizer,
        fit_images,
        fit_labels,
        device=device,
        epochs=TRAINING_EPOCHS,
        seed=TRAINING_SEED,
    )
    _require(training["updates"] > 0 and len(training["epochs"]) == TRAINING_EPOCHS, "E_C training did not complete 20 epochs")
    _write_json(target / "training_trace.json", training, kind="E_C training trace")

    qualification = _qualification_metrics(
        model,
        qualification_images,
        qualification_labels,
        device=device,
    )
    _write_json(target / "qualification_metrics.json", qualification, kind="E_C qualification metrics")

    checkpoint_path: Path | None = None
    if qualification["passed"]:
        checkpoint_path = target / "evaluator_checkpoint.pt"
        atomic_torch_save_new(
            checkpoint_path,
            _checkpoint_payload(
                model,
                split_manifest=split_manifest,
                training_config=training_config,
                qualification=qualification,
            ),
            artifact_kind="E_C frozen evaluator checkpoint",
        )

    source_manifest_ref = _artifact_ref(root, target / "source_manifest.json", role="execution source manifest")
    artifact_refs = {
        "architecture": _artifact_ref(root, target / "architecture.json", role="fixed E_C architecture schema"),
        "preprocessing": _artifact_ref(root, target / "preprocessing.json", role="fixed E_C preprocessing contract"),
        "training_config": _artifact_ref(root, target / "training_config.json", role="fixed E_C training contract"),
        "split_manifest": _artifact_ref(root, target / "split_manifest.json", role="E_C FIT/QUALIFICATION split identity"),
        "split_indices": _artifact_ref(root, target / "split_indices.pt", role="E_C FIT/QUALIFICATION indices"),
        "preflight": _artifact_ref(root, target / "preflight.json", role="E_C preflight"),
        "training_trace": _artifact_ref(root, target / "training_trace.json", role="E_C training trace"),
        "qualification": _artifact_ref(root, target / "qualification_metrics.json", role="canonical E_C qualification"),
        "source_manifest": source_manifest_ref,
    }
    if checkpoint_path is not None:
        artifact_refs["checkpoint"] = _artifact_ref(root, checkpoint_path, role="frozen E_C confirmation evaluator")

    classification = CLASSIFICATION_SUPPORTED if qualification["passed"] else CLASSIFICATION_QUALIFICATION_FAILED
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": bool(qualification["passed"]),
        "classification": classification,
        "authorization_consumed": {
            "token": AUTHORIZATION_TOKEN,
            "id": AUTHORIZATION_ID,
            "one_authorization": True,
        },
        "local_state": {
            "branch": source_manifest.branch,
            "head": source_manifest.head,
            "dirty_before": source_manifest.dirty,
            "worktree_preserved": True,
        },
        "evaluator_role": "INDEPENDENT_CONFIRMATION_ONLY",
        "architecture_authority": {
            "kind": ThirdConfirmationEvaluator.architecture_kind,
            "source": "prepared fallback contract; no unused project-authoritative family existed",
            "structurally_distinct_from": {
                "SmoothMNISTEvaluator_v2": "kernel=5/ReLU/MaxPool versus kernel=3/SiLU/AvgPool",
                "FrozenMNISTEvaluator_v1": "kernel=5 versus kernel=3",
                "previous_teacher": "checkpoint identity and architecture family excluded before training",
            },
        },
        "independence_check": {
            "smooth_training": False,
            "independent_evaluator_2": False,
            "previous_teacher_objective": False,
            "cross_evaluator_attribution": False,
            "joint_gradient_construction": False,
            "hyperparameter_selection": False,
            "generated_cmos_in_this_task": False,
            "future_optical_training_seen": False,
        },
        "data_contract": {
            "dataset_root": _relative(dataset_root, root),
            "training_source": {
                "path": TRAINING_SOURCE_RELATIVE.as_posix(),
                "sha256": TRAINING_SOURCE_SHA256,
                "role": "canonical TRAIN source; only this processed TRAIN file was read",
            },
            "canonical_source_train_rows_loaded": 60_000,
            "canonical_parent_train_rows": 55_000,
            "fit_rows": int(fit_labels.numel()),
            "qualification_rows": int(qualification_labels.numel()),
            "dev_generated_cmos_reads": 0,
            "optical_output_reads": 0,
            "p2_q2_reads": 0,
            "official_test_reads": 0,
            "protected_reads": 0,
            "sealed_reads": 0,
        },
        "training_contract": training_config,
        "qualification_contract": qualification["policy"],
        "qualification_result": qualification,
        "artifacts": artifact_refs,
        "claim_boundary": {
            "third_confirmation_authority_established": bool(qualification["passed"]),
            "generator_semantic_recovery": False,
            "generation_quality": False,
            "optical_system_validity": False,
            "formal_promotion": False,
        },
        "unresolved": [] if qualification["passed"] else ["E_C did not satisfy the frozen canonical-image qualification policy."],
        "next_action": "AUTHORIZED_EXECUTE_EVALUATOR_ROBUST_SEMANTIC_TRAINING_V1" if qualification["passed"] else "NONE",
        "output_dir": _relative(target, root),
    }
    _write_json(target / "provenance_manifest.json", {
        "schema": PROVENANCE_SCHEMA,
        "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
        "task_id": TASK_ID,
        "source_manifest": source_manifest_ref,
        "artifact_refs": artifact_refs,
        "checkpoint_frozen_before_future_optical_training": bool(checkpoint_path is not None),
    }, kind="E_C provenance manifest")
    report["artifacts"]["provenance"] = _artifact_ref(root, target / "provenance_manifest.json", role="E_C provenance manifest")
    _write_json(target / "report.json", report, kind="E_C report")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-token", required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAINING_BATCH_SIZE)
    parser.add_argument("--qualification-per-class", type=int, default=QUALIFICATION_PER_CLASS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = execute(args)
    print(json.dumps({
        "task_id": report["task_id"],
        "result": report["result"],
        "classification": report["classification"],
        "outer_gate_passed": report["outer_gate_passed"],
        "output_dir": report["output_dir"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["result"] == "VALID_COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
