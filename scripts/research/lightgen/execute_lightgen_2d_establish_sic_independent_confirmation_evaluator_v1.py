"""Establish one independent canonical-digit confirmation evaluator.

The consumer trains exactly one preregistered PixelMLP evaluator on the
canonical MNIST TRAIN role.  It never loads a Generator, optical output,
generated CMOS, sampler bank, official test split, protected data, or sealed
data.  The final epoch is the only candidate and qualification is measured
exactly once on the frozen class-stratified qualification split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.application.research.shared.research_run_control import capture_execution_source_manifest
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import PixelMLPConfirmationEvaluatorV1
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import stratified_mnist_indices


TASK_ID = "ESTABLISH_SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_ESTABLISH_SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_ESTABLISHED"
CLASSIFICATION_SUPPORTED = "SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_ESTABLISHED"
CLASSIFICATION_QUALIFICATION_FAILED = "SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_QUALIFICATION_FAILED"
CLASSIFICATION_INVALID_HARNESS = "INVALID_HARNESS"
SEMANTIC_AUTHORITY_ID = "SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_D_V1"
RUN_ID = "sic-independent-confirmation-evaluator-v1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01"
)
DEFAULT_DATASET_ROOT = Path("data/MNIST")
TRAINING_SOURCE_RELATIVE = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"

PARENT_SPLIT_SEED = 1234
PARENT_VALIDATION_PER_LABEL = 500
ED_SPLIT_SEED = 2026092003
QUALIFICATION_PER_CLASS = 550
TRAINING_SEED = 2026092004
TRAINING_EPOCHS = 20
TRAINING_BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False

# This is the task-specific, frozen E_D confirmation rule.  It is not used
# to requalify any previously consumed evaluator.
QUALIFICATION_POLICY_ID = "sic_independent_confirmation_canonical_digit_v1"
QUALIFICATION_MIN_ACCURACY = 0.97
QUALIFICATION_MIN_MACRO_RECALL = 0.97
QUALIFICATION_MIN_CLASS_RECALL = 0.94

ES_BUNDLE = Path(
    "outputs/20260730_v2/runs/20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/conditional_evaluator_bundle_v21.json"
)
ES_BUNDLE_SHA256 = "970dd6ea39cbda8aa23d3ca8338d9b0140a21cde45e4199137ef303f9d73ea31"
ES_CHECKPOINT = Path(
    "outputs/20260730_v2/runs/20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/smooth_mnist_evaluator_v21.pt"
)
ES_CHECKPOINT_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
EI_QUALIFICATION = Path(
    "outputs/20260724_v2/runs/20260724_114819_935841_formal_v2_amplitude_finite_crop_identity/evaluator_qualification_evidence.json"
)
EI_QUALIFICATION_SHA256 = "e5ab7ea262756bb50badbd810ce6aff165b8b9ca84f0c8e3a04424474c78aa29"
EI_CHECKPOINT = Path(
    "outputs/20260724_v2/runs/20260724_114819_935841_formal_v2_amplitude_finite_crop_identity/frozen_mnist_evaluator.pt"
)
EI_CHECKPOINT_SHA256 = "16a42fad849ed9af249fa65bed9567a8fabc011fcc9d107061f69cdb27f21684"
EC_ARCHITECTURE = Path(
    "outputs/20260919_third_confirmation_evaluator_v1/runs/attempt_03/architecture.json"
)
EC_ARCHITECTURE_SHA256 = "df091a7f177d4cb427d6b8b9df4b51f5baf057207c4f2d565969045dd0cb56dc"
EC_CHECKPOINT = Path(
    "outputs/20260919_third_confirmation_evaluator_v1/runs/attempt_03/evaluator_checkpoint.pt"
)
EC_CHECKPOINT_SHA256 = "e5688c68a6fba506a22530770efd3f6a09d5c72097437c28c535ddec26a50f95"
SHAPE_QUALIFICATION = Path(
    "outputs/20260804_capacity_map_v2_2/runs/20260804_010321_583684_research_qualify_shape/capacity_map_shape_evaluator_qualification_evidence.json"
)
SHAPE_QUALIFICATION_SHA256 = "a1f7ed72192e1704e07b24d55245bf5430dc5e1276a66e5a0ece8ba2430461b0"
SHAPE_CHECKPOINT = Path(
    "outputs/20260804_capacity_map_v2_2/runs/20260804_010321_583684_research_qualify_shape/shape_mnist_evaluator_v22.pt"
)
SHAPE_CHECKPOINT_SHA256 = "519249d8936c0cedcd9c0a181c791472cd88f9efca3d23b4aa52fad1d4675019"

ARCHITECTURE_SCHEMA = "sic_confirmation_evaluator_d_architecture.v1"
PREPROCESSING_SCHEMA = "sic_confirmation_evaluator_d_preprocessing.v1"
SPLIT_SCHEMA = "sic_confirmation_evaluator_d_split.v1"
TRAINING_SCHEMA = "sic_confirmation_evaluator_d_training_config.v1"
PREFLIGHT_SCHEMA = "sic_confirmation_evaluator_d_preflight.v1"
CHECKPOINT_SCHEMA = "sic_confirmation_evaluator_d_checkpoint.v1"
QUALIFICATION_SCHEMA = "sic_confirmation_evaluator_d_qualification.v1"
PROVENANCE_SCHEMA = "sic_confirmation_evaluator_d_provenance.v1"
REPORT_SCHEMA = "sic_confirmation_evaluator_d_report.v1"


class EvaluatorExecutionError(RuntimeError):
    """Raised when the E_D consumer cannot proceed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluatorExecutionError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _new_model(*, device: torch.device, seed: int) -> PixelMLPConfirmationEvaluatorV1:
    _seed_everything(seed)
    return PixelMLPConfirmationEvaluatorV1().to(device)


def _new_optimizer(model: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
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
        "architecture_kind": PixelMLPConfirmationEvaluatorV1.architecture_kind,
        "input": {"shape": [1, 28, 28], "range": "[0,1]"},
        "layers": [
            {"type": "Flatten", "in_features": 784},
            {"type": "Linear", "in_features": 784, "out_features": 512},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 512, "out_features": 256},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 256, "out_features": 10},
        ],
        "forbidden": [
            "convolution",
            "pooling",
            "batch_norm",
            "dropout",
            "residual_blocks",
            "pretrained_weights",
            "augmentation_specific_modules",
        ],
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


def _load_json(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    _require(path.is_file(), f"authority is missing: {path}")
    _require(file_sha256(path) == expected_sha256, f"authority changed: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluatorExecutionError(f"authority cannot be parsed: {path}") from error
    _require(isinstance(payload, dict), f"authority is not an object: {path.name}")
    return payload


def _architecture_independence_audit(root: Path) -> dict[str, Any]:
    es_bundle = _load_json(root / ES_BUNDLE, expected_sha256=ES_BUNDLE_SHA256)
    ei_qualification = _load_json(root / EI_QUALIFICATION, expected_sha256=EI_QUALIFICATION_SHA256)
    ec_architecture = _load_json(root / EC_ARCHITECTURE, expected_sha256=EC_ARCHITECTURE_SHA256)
    shape_qualification = _load_json(root / SHAPE_QUALIFICATION, expected_sha256=SHAPE_QUALIFICATION_SHA256)
    for path, expected in (
        (root / ES_CHECKPOINT, ES_CHECKPOINT_SHA256),
        (root / EI_CHECKPOINT, EI_CHECKPOINT_SHA256),
        (root / EC_CHECKPOINT, EC_CHECKPOINT_SHA256),
        (root / SHAPE_CHECKPOINT, SHAPE_CHECKPOINT_SHA256),
    ):
        _require(path.is_file() and file_sha256(path) == expected, f"authority changed: {path.name}")

    _require(es_bundle.get("smooth_checkpoint") == ES_CHECKPOINT_SHA256, "E_S bundle checkpoint binding changed")
    _require(ei_qualification.get("status") == "passed", "E_I qualification authority is not passed")
    _require(
        ei_qualification.get("context", {}).get("evaluator_checkpoint_hash") == EI_CHECKPOINT_SHA256,
        "E_I qualification checkpoint binding changed",
    )
    _require(
        ec_architecture.get("architecture_kind") == "third_confirmation_canonical_cnn_v1",
        "E_C architecture authority changed",
    )
    _require(shape_qualification.get("state") == "screened_out", "Shape evaluator exclusion authority changed")
    _require(
        shape_qualification.get("metrics", {}).get("checkpoint_sha256") == SHAPE_CHECKPOINT_SHA256,
        "Shape evaluator checkpoint binding changed",
    )

    existing = {
        "E_S": {
            "architecture_kind": "smooth_mnist_evaluator_v2",
            "family": "convolutional",
            "signature": "Conv2d(3)-SiLU-AvgPool2d-Conv2d(3)-SiLU-AvgPool2d-Flatten-Linear(3136,128)-SiLU-Linear(128,10)",
            "checkpoint": ES_CHECKPOINT.as_posix(),
        },
        "E_I": {
            "architecture_kind": "frozen_mnist_evaluator_v1",
            "family": "convolutional",
            "signature": "Conv2d(3)-ReLU-MaxPool2d-Conv2d(3)-ReLU-MaxPool2d-Flatten-Linear(3136,128)-ReLU-Linear(128,10)",
            "checkpoint": EI_CHECKPOINT.as_posix(),
        },
        "E_C": {
            "architecture_kind": ec_architecture["architecture_kind"],
            "family": "convolutional",
            "signature": "Conv2d(5)-ReLU-MaxPool2d-Conv2d(5)-ReLU-MaxPool2d-Flatten-Linear(3136,128)-ReLU-Linear(128,10)",
            "checkpoint": EC_CHECKPOINT.as_posix(),
        },
        "Shape": {
            "architecture_kind": "shape_mnist_evaluator_v22",
            "family": "convolutional",
            "signature": "Conv2d(3)-GELU-Conv2d(3,stride=2)-GELU-Conv2d(3,stride=2)-GELU-AdaptiveAvgPool2d-Flatten-Linear(96,128)-GELU-Linear(128,10)",
            "checkpoint": SHAPE_CHECKPOINT.as_posix(),
            "qualification_state": shape_qualification.get("state"),
            "qualification_accuracy": shape_qualification.get("metrics", {}).get("accuracy"),
        },
    }
    ed = {
        "architecture_kind": PixelMLPConfirmationEvaluatorV1.architecture_kind,
        "family": "fully_connected_mlp",
        "signature": "Flatten-Linear(784,512)-ReLU-Linear(512,256)-ReLU-Linear(256,10)",
    }
    material_matches = [
        name
        for name, authority in existing.items()
        if authority["family"] == ed["family"] or authority["signature"] == ed["signature"]
    ]
    return {
        "status": "PASS" if not material_matches else "BLOCKED_ED_ARCHITECTURE_INDEPENDENCE_UNRESOLVED",
        "candidate": ed,
        "already_consumed": existing,
        "material_matches": material_matches,
        "decision": "PixelMLP is a distinct fully-connected family from all consumed convolutional evaluators.",
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
    _require(isinstance(images, torch.Tensor) and isinstance(labels, torch.Tensor), "canonical TRAIN tensors missing")
    _require(images.shape == (60_000, 28, 28), "canonical TRAIN image shape changed")
    _require(images.dtype == torch.uint8 and labels.shape == (60_000,) and labels.dtype == torch.long, "canonical TRAIN schema changed")
    _require(int(images.min()) >= 0 and int(images.max()) <= 255, "canonical TRAIN image range changed")
    _require(int(labels.min()) >= 0 and int(labels.max()) <= 9, "canonical TRAIN labels changed")
    return images.unsqueeze(1).to(torch.float32).div_(255.0), labels, source_sha256


def _ed_split(
    dataset_root: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    images, labels, source_sha256 = _load_canonical_train_only(dataset_root)
    parent_train, source_validation = stratified_mnist_indices(
        labels, seed=PARENT_SPLIT_SEED, validation_per_label=PARENT_VALIDATION_PER_LABEL
    )
    _require(parent_train.numel() == 55_000 and source_validation.numel() == 5_000, "canonical parent split changed")
    parent_labels = labels.index_select(0, parent_train)
    generator = torch.Generator(device="cpu").manual_seed(ED_SPLIT_SEED)
    fit_parts: list[torch.Tensor] = []
    qualification_parts: list[torch.Tensor] = []
    for class_id in range(10):
        matches = torch.nonzero(parent_labels == class_id, as_tuple=False).flatten()
        _require(matches.numel() > QUALIFICATION_PER_CLASS, f"class {class_id} lacks E_D rows")
        shuffled = matches.index_select(0, torch.randperm(matches.numel(), generator=generator))
        qualification_parts.append(shuffled[:QUALIFICATION_PER_CLASS])
        fit_parts.append(shuffled[QUALIFICATION_PER_CLASS:])
    fit_source = parent_train.index_select(0, torch.cat(fit_parts).sort().values).sort().values
    qualification_source = parent_train.index_select(0, torch.cat(qualification_parts).sort().values).sort().values
    _require(fit_source.numel() == 49_500 and qualification_source.numel() == 5_500, "E_D split counts changed")
    _require(not bool(torch.isin(fit_source, qualification_source).any().item()), "E_D split leakage")
    fit_labels = labels.index_select(0, fit_source)
    qualification_labels = labels.index_select(0, qualification_source)
    _require(torch.bincount(qualification_labels, minlength=10).tolist() == [550] * 10, "E_D qualification is not balanced")
    manifest = {
        "schema": SPLIT_SCHEMA,
        "role": "canonical_train_only_evaluator_fit_qualification",
        "parent": {
            "seed": PARENT_SPLIT_SEED,
            "validation_per_label": PARENT_VALIDATION_PER_LABEL,
            "train_count": int(parent_train.numel()),
            "source_validation_count": int(source_validation.numel()),
            "train_indices_sha256": tensor_sha256(parent_train),
            "source_validation_indices_sha256": tensor_sha256(source_validation),
            "dataset_content_sha256": canonical_sha256({"images": tensor_sha256(images), "labels": tensor_sha256(labels)}),
            "source_file": {
                "path": TRAINING_SOURCE_RELATIVE.as_posix(),
                "sha256": source_sha256,
                "schema": "60000x28x28 uint8 images plus 60000 int64 labels",
            },
        },
        "selection": {
            "seed": ED_SPLIT_SEED,
            "algorithm": "per-class deterministic shuffle; 550 qualification rows/class; remaining parent rows fit; source-index sort",
            "qualification_per_class": QUALIFICATION_PER_CLASS,
            "fit_count": int(fit_source.numel()),
            "qualification_count": int(qualification_source.numel()),
            "fit_counts_by_class": torch.bincount(fit_labels, minlength=10).tolist(),
            "qualification_counts_by_class": torch.bincount(qualification_labels, minlength=10).tolist(),
            "fit_source_indices_sha256": tensor_sha256(fit_source),
            "qualification_source_indices_sha256": tensor_sha256(qualification_source),
        },
        "forbidden_roles": {
            "optical_dev": 0,
            "dev_generated_cmos": 0,
            "optical_outputs": 0,
            "p2_q2": 0,
            "s1": 0,
            "official_test": 0,
            "protected": 0,
            "sealed": 0,
        },
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return (
        images.index_select(0, fit_source),
        fit_labels,
        images.index_select(0, qualification_source),
        qualification_labels,
        {
            "manifest": manifest,
            "fit_source_indices": fit_source,
            "qualification_source_indices": qualification_source,
            "parent_train_indices": parent_train,
            "source_validation_indices": source_validation,
            "all_labels": labels,
        },
    )


def _epoch_order(count: int, *, seed: int, epoch: int) -> torch.Tensor:
    return torch.randperm(count, generator=torch.Generator(device="cpu").manual_seed(seed + epoch))


def _run_updates(
    model: PixelMLPConfirmationEvaluatorV1,
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
            logits = model(batch_images)
            loss = F.cross_entropy(logits, batch_labels)
            _finite_tensor(loss, "E_D training loss")
            loss.backward()
            gradient_norm = math.sqrt(
                sum(
                    float(parameter.grad.detach().square().sum().item())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
            _require(math.isfinite(gradient_norm) and gradient_norm > 0.0, "E_D gradient is zero or nonfinite")
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            epoch_batches += 1
            update_count += 1
            if capture_update_trace:
                update_trace.append(
                    {"update": update_count, "epoch": epoch, "loss": float(loss.detach().item()), "gradient_l2": gradient_norm}
                )
        epoch_trace.append(
            {"epoch": epoch + 1, "updates": update_count, "mean_fit_loss": epoch_loss / max(epoch_batches, 1), "batches": epoch_batches}
        )
    return {"updates": update_count, "epochs": epoch_trace, "update_trace": update_trace}


def _preflight(
    fit_images: torch.Tensor,
    fit_labels: torch.Tensor,
    qualification_labels: torch.Tensor,
    split_data: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    _require(device.type == "cuda" and torch.cuda.is_available(), "formal E_D execution requires CUDA")
    fit_source = split_data["fit_source_indices"]
    qualification_source = split_data["qualification_source_indices"]
    all_labels = split_data["all_labels"]
    alignment_passed = bool(torch.equal(fit_labels, all_labels.index_select(0, fit_source)))
    disjoint_passed = not bool(torch.isin(fit_source, qualification_source).any().item())
    qualification_count_passed = qualification_labels.numel() == 5_500
    _require(alignment_passed and disjoint_passed and qualification_count_passed, "E_D split alignment preflight failed")

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
    parameters_departed = valid_before != valid_after

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
    negative_state_differs = valid_after != _state_hash(negative_model)
    valid_last_loss = float(valid_run["update_trace"][-1]["loss"])
    negative_last_loss = float(negative_run["update_trace"][-1]["loss"])
    negative_control_passed = bool(negative_state_differs and abs(valid_last_loss - negative_last_loss) > 1.0e-9)
    status = bool(
        valid_run["updates"] == 3
        and parameters_departed
        and alignment_passed
        and disjoint_passed
        and negative_control_passed
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY" if status else "INVALID_HARNESS",
        "device": str(device),
        "static_contract": {
            "architecture": PixelMLPConfirmationEvaluatorV1.architecture_kind,
            "trainable_groups": ["E_D.parameters"],
            "frozen_upstream_groups": [],
            "generated_cmos_reads": 0,
            "optical_forward_calls": 0,
            "protected_test_sealed_reads": 0,
        },
        "split_alignment": {
            "fit_labels_bound_to_source_indices": alignment_passed,
            "fit_qualification_disjoint": disjoint_passed,
            "qualification_count": int(qualification_labels.numel()),
            "qualification_rows_in_optimizer_batches": False,
        },
        "three_step_real_update": {
            "updates": valid_run["updates"],
            "parameters_departed": parameters_departed,
            "trace": valid_run["update_trace"],
        },
        "negative_control": {
            "kind": "deterministic_cyclic_label_shift",
            "updates": negative_run["updates"],
            "state_differs_from_valid_label_run": negative_state_differs,
            "valid_last_loss": valid_last_loss,
            "shifted_label_last_loss": negative_last_loss,
            "passed": negative_control_passed,
        },
        "no_scientific_training": True,
    }


def _qualification_metrics(
    model: PixelMLPConfirmationEvaluatorV1,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], TRAINING_BATCH_SIZE):
            logits = model(images[start : start + TRAINING_BATCH_SIZE].to(device))
            _finite_tensor(logits, "E_D qualification logits")
            predictions.append(logits.argmax(dim=1).cpu())
    predicted = torch.cat(predictions)
    confusion = torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
    denominators = confusion.sum(dim=1)
    per_class_recall = [float(confusion[i, i].item() / denominators[i].item()) for i in range(10)]
    accuracy = float((predicted == labels).float().mean().item())
    macro_recall = float(sum(per_class_recall) / 10.0)
    passed = bool(
        accuracy >= QUALIFICATION_MIN_ACCURACY
        and macro_recall >= QUALIFICATION_MIN_MACRO_RECALL
        and min(per_class_recall) >= QUALIFICATION_MIN_CLASS_RECALL
    )
    return {
        "schema": QUALIFICATION_SCHEMA,
        "policy": {
            "id": QUALIFICATION_POLICY_ID,
            "min_accuracy": QUALIFICATION_MIN_ACCURACY,
            "min_macro_recall": QUALIFICATION_MIN_MACRO_RECALL,
            "min_class_recall": QUALIFICATION_MIN_CLASS_RECALL,
            "frozen_before_training": True,
        },
        "split_role": "E_D_QUALIFICATION",
        "evaluation_calls": 1,
        "count": int(labels.numel()),
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion.tolist(),
        "predicted_class_histogram": torch.bincount(predicted, minlength=10).tolist(),
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
            "schema": "sic_confirmation_evaluator_d_split_indices.v1",
            "parent_train_indices": split_data["parent_train_indices"].cpu(),
            "source_validation_indices": split_data["source_validation_indices"].cpu(),
            "fit_source_indices": split_data["fit_source_indices"].cpu(),
            "qualification_source_indices": split_data["qualification_source_indices"].cpu(),
        },
        artifact_kind="E_D split indices",
    )
    manifest = dict(split_data["manifest"])
    manifest["split_indices_artifact"] = {"path": split_indices_path.name, "sha256": file_sha256(split_indices_path)}
    _write_json(target / "split_manifest.json", manifest, kind="E_D split manifest")
    return manifest


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
            "amsgrad": AMSGRAD,
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
        "hidden_width_search": False,
        "early_stopping": False,
        "future_cmos_selection": False,
    }


def _checkpoint_payload(
    model: PixelMLPConfirmationEvaluatorV1,
    *,
    split_manifest: Mapping[str, Any],
    training_config: Mapping[str, Any],
    qualification: Mapping[str, Any],
    authority_status: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
        "architecture_kind": PixelMLPConfirmationEvaluatorV1.architecture_kind,
        "architecture": _architecture_payload(),
        "preprocessing": _preprocessing_payload(),
        "class_order": list(range(10)),
        "prediction_rule": "argmax(logits,dim=1)",
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "split_manifest_hash": split_manifest["manifest_hash"],
        "training_config": dict(training_config),
        "qualification": dict(qualification),
        "authority_status": authority_status,
        "independence": {
            "role": "INDEPENDENT_CONFIRMATION_ONLY",
            "used_in_ES_EI_EC_training": False,
            "used_in_SI_training": False,
            "used_in_SIC_attribution": False,
            "used_in_sampler_selection": False,
            "used_in_generator_architecture_decisions": False,
            "used_in_hyperparameter_selection": False,
            "used_on_generated_cmos_in_this_task": False,
            "frozen_before_future_sic_training": authority_status == "QUALIFIED",
        },
    }


def _prepare_output(target: Path) -> None:
    _require(not target.exists(), f"refusing to reuse existing E_D attempt directory: {target}")
    target.mkdir(parents=True, exist_ok=False)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    _require(args.authorization_token == AUTHORIZATION_TOKEN, "authorization token mismatch")
    _require(args.device == "cuda", "formal E_D execution requires --device cuda")
    _require(args.epochs == TRAINING_EPOCHS, "E_D training horizon is fixed at 20 epochs")
    _require(args.batch_size == TRAINING_BATCH_SIZE, "E_D batch size is fixed at 256")
    _require(args.qualification_per_class == QUALIFICATION_PER_CLASS, "E_D qualification split is fixed at 550/class")
    _require(torch.cuda.is_available(), "formal E_D execution requires CUDA")

    target = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    _prepare_output(target)
    device = torch.device(args.device)
    dataset_root = (root / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root).resolve()
    architecture_audit = _architecture_independence_audit(root)
    _require(architecture_audit["status"] == "PASS", "E_D architecture independence audit failed")
    training_config = _training_config_payload()

    source_manifest = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=[
            Path(__file__),
            root / "src/ldm_project/v2/application/research/lightgen/confirmation_evaluators.py",
            root / "src/ldm_project/v2/student/evaluator.py",
            root / "src/ldm_project/v2/application/research/shared/models.py",
            root / "src/ldm_project/v2/application/current/measurement.py",
            root / "src/ldm_project/v2/application/research/shared/research_run_control.py",
            root / "src/ldm_project/v2/evidence/io.py",
            root / TRAINING_SOURCE_RELATIVE,
            root / ES_BUNDLE,
            root / ES_CHECKPOINT,
            root / EI_QUALIFICATION,
            root / EI_CHECKPOINT,
            root / EC_ARCHITECTURE,
            root / EC_CHECKPOINT,
            root / SHAPE_QUALIFICATION,
            root / SHAPE_CHECKPOINT,
        ],
        config_identity={
            "architecture": PixelMLPConfirmationEvaluatorV1.architecture_kind,
            "split_seed": ED_SPLIT_SEED,
            "training": training_config,
            "qualification_policy": {
                "id": QUALIFICATION_POLICY_ID,
                "min_accuracy": QUALIFICATION_MIN_ACCURACY,
                "min_macro_recall": QUALIFICATION_MIN_MACRO_RECALL,
                "min_class_recall": QUALIFICATION_MIN_CLASS_RECALL,
            },
            "independence_audit": architecture_audit["status"],
        },
    )
    _write_json(target / "source_manifest.json", source_manifest.to_dict(), kind="E_D source manifest")
    _write_json(target / "architecture.json", _architecture_payload(), kind="E_D architecture schema")
    _write_json(target / "preprocessing.json", _preprocessing_payload(), kind="E_D preprocessing contract")
    _write_json(target / "training_config.json", training_config, kind="E_D training configuration")
    _write_json(target / "architecture_independence_audit.json", architecture_audit, kind="E_D architecture audit")

    fit_images, fit_labels, qualification_images, qualification_labels, split_data = _ed_split(dataset_root)
    split_manifest = _write_split_artifact(target, split_data)
    preflight = _preflight(fit_images, fit_labels, qualification_labels, split_data, device=device)
    _write_json(target / "preflight.json", preflight, kind="E_D preflight")
    _require(preflight["status"] == "READY", "E_D preflight did not reach READY")

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
    _require(training["updates"] > 0 and len(training["epochs"]) == TRAINING_EPOCHS, "E_D training did not complete 20 epochs")
    _write_json(target / "training_trace.json", training, kind="E_D training trace")

    model.freeze()
    qualification = _qualification_metrics(model, qualification_images, qualification_labels, device=device)
    _write_json(target / "qualification_metrics.json", qualification, kind="E_D qualification metrics")
    authority_status = "QUALIFIED" if qualification["passed"] else "REJECTED"
    candidate_path = target / "final_epoch_candidate.pt"
    atomic_torch_save_new(
        candidate_path,
        _checkpoint_payload(
            model,
            split_manifest=split_manifest,
            training_config=training_config,
            qualification=qualification,
            authority_status=authority_status,
        ),
        artifact_kind="E_D final epoch candidate",
    )

    source_manifest_ref = _artifact_ref(root, target / "source_manifest.json", role="execution source manifest")
    artifact_refs = {
        "architecture": _artifact_ref(root, target / "architecture.json", role="fixed E_D architecture schema"),
        "preprocessing": _artifact_ref(root, target / "preprocessing.json", role="fixed E_D preprocessing contract"),
        "architecture_independence_audit": _artifact_ref(root, target / "architecture_independence_audit.json", role="E_D independence audit"),
        "training_config": _artifact_ref(root, target / "training_config.json", role="fixed E_D training contract"),
        "split_manifest": _artifact_ref(root, target / "split_manifest.json", role="E_D FIT/QUALIFICATION split identity"),
        "split_indices": _artifact_ref(root, target / "split_indices.pt", role="E_D FIT/QUALIFICATION indices"),
        "preflight": _artifact_ref(root, target / "preflight.json", role="E_D preflight"),
        "training_trace": _artifact_ref(root, target / "training_trace.json", role="E_D training trace"),
        "qualification": _artifact_ref(root, target / "qualification_metrics.json", role="E_D canonical qualification"),
        "final_epoch_candidate": _artifact_ref(root, candidate_path, role="E_D final epoch candidate"),
        "source_manifest": source_manifest_ref,
    }
    if qualification["passed"]:
        artifact_refs["checkpoint"] = _artifact_ref(root, candidate_path, role="frozen E_D confirmation evaluator")

    classification = CLASSIFICATION_SUPPORTED if qualification["passed"] else CLASSIFICATION_QUALIFICATION_FAILED
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": bool(qualification["passed"]),
        "classification": classification,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "local_state": {"branch": source_manifest.branch, "head": source_manifest.head, "dirty_before": source_manifest.dirty, "worktree_preserved": True},
        "evaluator_role": "INDEPENDENT_CONFIRMATION_ONLY",
        "architecture": _architecture_payload(),
        "architecture_independence_audit": architecture_audit,
        "data_contract": {
            "dataset_root": _relative(dataset_root, root),
            "training_source": {"path": TRAINING_SOURCE_RELATIVE.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "role": "canonical TRAIN source only"},
            "canonical_source_train_rows_loaded": 60_000,
            "canonical_parent_train_rows": 55_000,
            "fit_rows": int(fit_labels.numel()),
            "qualification_rows": int(qualification_labels.numel()),
            "dev_generated_cmos_reads": 0,
            "optical_output_reads": 0,
            "p2_q2_reads": 0,
            "s1_reads": 0,
            "official_test_reads": 0,
            "protected_reads": 0,
            "sealed_reads": 0,
        },
        "preprocessing": _preprocessing_payload(),
        "class_order": list(range(10)),
        "training_contract": training_config,
        "qualification_contract": qualification["policy"],
        "qualification_result": qualification,
        "independence_audit": {
            "E_S_training": False,
            "E_I_training": False,
            "E_C_training": False,
            "SI_training": False,
            "S_I_C_gradient_attribution": False,
            "sampler_selection": False,
            "generator_architecture_decisions": False,
            "optical_hyperparameter_selection": False,
            "generated_cmos_reads": 0,
            "optical_output_reads": 0,
        },
        "sic_binding": {
            "bound_only_if_qualified": True,
            "semantic_authority_id": SEMANTIC_AUTHORITY_ID if qualification["passed"] else None,
            "prior_unresolved_E_D_field_resolved": bool(qualification["passed"]),
            "sic_gradient_reopened": False,
            "delta_sic_reopened": False,
            "optimizer_reopened": False,
            "step3000_trajectory_consumed": False,
        },
        "anti_evaluator_chase": {
            "final_new_classifier_endpoint_for_current_sic_branch": True,
            "create_E_E_if_future_confirmation_fails": False,
            "promote_E_D_to_training_teacher": False,
        },
        "artifacts": artifact_refs,
        "claim_boundary": {
            "independent_evaluator_established": bool(qualification["passed"]),
            "sic_training_success": False,
            "generation_quality": False,
            "human_perceptual_equivalence": False,
            "hardware_readiness": False,
            "formal_promotion": False,
        },
        "unresolved": [] if qualification["passed"] else ["E_D did not satisfy the frozen canonical-digit qualification policy."],
        "next_action": "AUTHORIZED_EXECUTE_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1" if qualification["passed"] else "NONE",
        "output_dir": _relative(target, root),
    }
    _write_json(
        target / "provenance_manifest.json",
        {
            "schema": PROVENANCE_SCHEMA,
            "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
            "task_id": TASK_ID,
            "source_manifest": source_manifest_ref,
            "artifact_refs": artifact_refs,
            "checkpoint_frozen_before_future_sic_training": bool(qualification["passed"]),
        },
        kind="E_D provenance manifest",
    )
    report["artifacts"]["provenance"] = _artifact_ref(root, target / "provenance_manifest.json", role="E_D provenance manifest")
    _write_json(target / "report.json", report, kind="E_D report")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
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
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN}, ensure_ascii=False, indent=2))
        return 2
    try:
        report = execute(args)
    except (EvaluatorExecutionError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"task_id": report["task_id"], "result": report["result"], "classification": report["classification"], "outer_gate_passed": report["outer_gate_passed"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
