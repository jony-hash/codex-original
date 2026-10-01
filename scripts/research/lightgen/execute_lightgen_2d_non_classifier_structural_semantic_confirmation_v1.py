"""Execute the frozen non-classifier structural semantic confirmation.

This consumer uses only the accepted canonical MNIST images, the accepted
TRAIN/DEV OLS source identities, and the already materialized step-6000 image
views/predictions.  It does not load a Generator, run a forward pass, fit a
classifier, fit a sampler, or read protected/test/sealed data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.transforms import ScaledTranslation

from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)


TASK_ID = "NON_CLASSIFIER_STRUCTURAL_SEMANTIC_CONFIRMATION_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_NON_CLASSIFIER_STRUCTURAL_SEMANTIC_CONFIRMATION_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
INPUT_GATE = "NON_CLASSIFIER_STRUCTURAL_SEMANTIC_CONFIRMATION_CONTRACT_DEFINED"
OUTER_GATE = "NON_CLASSIFIER_STRUCTURAL_SEMANTIC_CONFIRMATION_V1_RESOLVED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
IMAGE_SIZE = 28
IMAGE_AREA = IMAGE_SIZE * IMAGE_SIZE
THRESHOLDS = (0.25, 0.50, 0.75)
NORMALIZATION_EPS = 1.0e-6
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
SIGNATURE_DIMENSION = 337
STRUCTURAL_BATCH_SIZE = 1024
DISTANCE_QUERY_BLOCK = 256
RASTER_DPI = 600

LOCALIZATION_RUN = Path(
    "outputs/20260921_step6000_image_manifold_semantic_divergence_localization_v1/runs/attempt_07"
)
LOCALIZATION_REPORT = LOCALIZATION_RUN / "report.json"
LOCALIZATION_REPORT_SHA256 = "65e6d32821e1f5445c5e646811dcd26f75f2d36f04afe12129b0b159b3df08ba"
LOCALIZATION_STATUS = LOCALIZATION_RUN / "status.json"
LOCALIZATION_STATUS_SHA256 = "84c6bdd6837553a520998400e2bf1d3de178022b3f23996759edd31e74d49c39"
LOCALIZATION_PAYLOAD = LOCALIZATION_RUN / "analysis_payload.pt"
LOCALIZATION_PAYLOAD_SHA256 = "1de28016921fb9c305ec8273fc5904fd40aa989d32b3683e322339c1aa95bedf"
LOCALIZATION_PREDICTIONS = LOCALIZATION_RUN / "predictions.pt"
LOCALIZATION_PREDICTIONS_SHA256 = "18eea92c3dd6e067e641b303873abe716d4da7b05d8f3ac2b58c32dc01de6e6a"
LOCALIZATION_STRATA = LOCALIZATION_RUN / "s1_consensus_strata.json"
LOCALIZATION_STRATA_SHA256 = "3ae01234c03b062e89722ce5eb5deb8da80151b9205e5e50ce914c703058e18a"
LOCALIZATION_SOURCE_MANIFEST = LOCALIZATION_RUN / "source_manifest.json"
LOCALIZATION_SOURCE_MANIFEST_SHA256 = "76dd500f72820f70374f04648f4c7be7f60be842afcf4fd63efe4303e80a7d5d"

READINESS_RUN = Path("outputs/20260921_step6000_conditional_generation_readiness_v1/runs/attempt_06")
READINESS_REPORT = READINESS_RUN / "report.json"
READINESS_REPORT_SHA256 = "0cdbbd1e45ee051500b004536c8216dd44a9ec27fc1eddbaa359f848b4a3c4ae"
READINESS_STATUS = READINESS_RUN / "status.json"
READINESS_STATUS_SHA256 = "5314ac1cae8821458cf34055bae7f9dd2f1f6dccd125556214be0f39c75e7517"
READINESS_SELECTION = READINESS_RUN / "fixed_selection_manifest.json"
READINESS_SELECTION_SHA256 = "2d198c652dadf684b06b4e70ff195d18896c4bc9de689c34ae159072615bba55"
READINESS_FORWARD_METRICS = READINESS_RUN / "forward_metrics.json"

TRAINING_SOURCE = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
TRAIN_OLS = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01/final_train_ols.pt")
TRAIN_OLS_SHA256 = "2d1264219faa0964467d40f2e4882ec3767d3a1aafd30e143bf406aa39be6dcc"
DEV_OLS = Path("outputs/20260918_frozen_dev_ols_cmos_v1/runs/attempt_01/dev_complex_ols.pt")
DEV_OLS_SHA256 = "5a8efe07c6ba52168f415e56fccdd68c2552b6dd5f4f975dabc96d2215fbb622"
S1_BANK = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01/candidate_latent_banks.pt")
S1_BANK_SHA256 = "5ad385e3f683c144c5c8f0464e2519a11e03dbbf9dc0ce9b91bb9b7170148419"
S1_WINNER = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01/winner_frozen_sampler.pt")
S1_WINNER_SHA256 = "166cf15ea91666afe1fe1c88e8897b88d58a4740253d59838927502babba568c"

REPORT_SCHEMA = "non_classifier_structural_semantic_confirmation_report.v1"
STATUS_SCHEMA = "non_classifier_structural_semantic_confirmation_status.v1"
PREFLIGHT_SCHEMA = "non_classifier_structural_semantic_confirmation_preflight.v1"
REFERENCE_SCHEMA = "non_classifier_structural_signature_reference.v1"
TENSORS_SCHEMA = "non_classifier_structural_semantic_confirmation_tensors.v1"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260921_non_classifier_structural_semantic_confirmation_v1/runs/attempt_01"
)

EVALUATOR_ORDER = ("E_S", "E_I", "E_C", "E_D")
SOURCE_ORDER = ("canonical_dev", "real_ols_step6000", "s1_step6000")
SOURCE_LABEL_NAMES = {
    "canonical_dev": "true class",
    "real_ols_step6000": "true class",
    "s1_step6000": "requested class",
}

TOPOLOGY_NAMES = ("component_count", "hole_count", "euler_characteristic", "largest_component_fraction")
TOPOLOGY_FEATURES_PER_THRESHOLD = 4
GEOMETRY_FEATURES_PER_THRESHOLD = 67
SPATIAL_FEATURES_PER_THRESHOLD = 20
THRESHOLD_BLOCK_SIZE = GEOMETRY_FEATURES_PER_THRESHOLD + TOPOLOGY_FEATURES_PER_THRESHOLD + SPATIAL_FEATURES_PER_THRESHOLD
GRAYSCALE_BLOCK_SIZE = 64
TOPOLOGY_OFFSET = GRAYSCALE_BLOCK_SIZE + GEOMETRY_FEATURES_PER_THRESHOLD


def _topology_slice(threshold_index: int) -> slice:
    start = GRAYSCALE_BLOCK_SIZE + threshold_index * THRESHOLD_BLOCK_SIZE + GEOMETRY_FEATURES_PER_THRESHOLD
    return slice(start, start + TOPOLOGY_FEATURES_PER_THRESHOLD)


class StructuralConfirmationError(RuntimeError):
    """Raised when a frozen authority or structural contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StructuralConfirmationError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _artifact(root: Path, path: Path, role: str) -> dict[str, Any]:
    target = (root / path if not path.is_absolute() else path).resolve()
    _require(target.is_file(), f"missing artifact: {target}")
    return {"path": target.relative_to(root.resolve()).as_posix(), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_write_new_json(path, dict(payload), artifact_kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


def _git_identity(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": [line[3:] if len(line) >= 3 else line for line in status.splitlines()],
    }


def _load_json(root: Path, path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    target = root / path
    _require(file_sha256(target) == expected_sha256, f"{name} identity changed")
    return _mapping(json.loads(target.read_text(encoding="utf-8")), name)


def _load_tensor_payload(root: Path, path: Path, expected_sha256: str, name: str) -> Mapping[str, Any]:
    payload = safe_torch_load(root / path, expected_sha256=expected_sha256, map_location="cpu")
    return _mapping(payload, name)


def _validate_image_views(value: torch.Tensor, name: str, rows: int) -> None:
    _require(value.shape == (rows, 1, IMAGE_SIZE, IMAGE_SIZE), f"{name} shape changed: {tuple(value.shape)}")
    _require(value.dtype == torch.float32, f"{name} dtype changed: {value.dtype}")
    _finite(value, name)
    _require(float(value.min()) >= 0.0 and float(value.max()) <= 1.0, f"{name} range changed")


def _make_structural_split(train_labels: torch.Tensor) -> dict[str, torch.Tensor]:
    train_labels = train_labels.to(torch.long).cpu().contiguous()
    fit_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        positions = torch.nonzero(train_labels == digit, as_tuple=False).flatten()
        _require(positions.numel() >= 10, f"class {digit} is too small for structural split")
        validation_parts.append(positions[::10])
        validation_mask = torch.zeros(positions.numel(), dtype=torch.bool)
        validation_mask[::10] = True
        fit_parts.append(positions[~validation_mask])
    fit_positions = torch.cat(fit_parts).to(torch.long).contiguous()
    validation_positions = torch.cat(validation_parts).to(torch.long).contiguous()
    combined = torch.cat([fit_positions, validation_positions])
    _require(combined.numel() == train_labels.numel(), "structural split does not cover TRAIN")
    _require(torch.unique(combined).numel() == train_labels.numel(), "structural split overlaps")
    return {
        "fit_positions": fit_positions,
        "validation_positions": validation_positions,
        "fit_labels": train_labels.index_select(0, fit_positions),
        "validation_labels": train_labels.index_select(0, validation_positions),
    }


def _load_authorities(root: Path) -> dict[str, Any]:
    git = _git_identity(root)
    _require(git["branch"] == EXPECTED_BRANCH, "consumer branch conflicts with accepted authority")
    _require(git["head"] == EXPECTED_HEAD, "consumer HEAD conflicts with accepted authority")

    localization_report = _load_json(root, LOCALIZATION_REPORT, LOCALIZATION_REPORT_SHA256, "accepted localization report")
    localization_status = _load_json(root, LOCALIZATION_STATUS, LOCALIZATION_STATUS_SHA256, "accepted localization status")
    localization_source_manifest = _load_json(
        root,
        LOCALIZATION_SOURCE_MANIFEST,
        LOCALIZATION_SOURCE_MANIFEST_SHA256,
        "accepted localization source manifest",
    )
    _require(localization_report.get("result") == "VALID_COMPLETED", "accepted localization result changed")
    _require(localization_report.get("outer_gate") == "STEP6000_IMAGE_MANIFOLD_SEMANTIC_DIVERGENCE_LOCALIZED", "accepted localization gate changed")
    _require(localization_report.get("classification") == "STEP6000_ON_MANIFOLD_CROSS_EVALUATOR_DECISION_DIVERGENCE_SUPPORTED", "active frontier changed")
    _require(localization_status.get("result") == "VALID_COMPLETED", "accepted localization status changed")
    _require(localization_source_manifest.get("schema") == "step6000_image_manifold_localization_source_manifest.v1", "accepted localization source manifest schema changed")
    _require(localization_source_manifest.get("protected_test_sealed_zero") is True, "accepted localization source manifest protection changed")

    readiness_report = _load_json(root, READINESS_REPORT, READINESS_REPORT_SHA256, "accepted readiness report")
    readiness_status = _load_json(root, READINESS_STATUS, READINESS_STATUS_SHA256, "accepted readiness status")
    selection = _load_json(root, READINESS_SELECTION, READINESS_SELECTION_SHA256, "accepted fixed selection")
    _require(readiness_report.get("result") == "VALID_COMPLETED", "accepted readiness result changed")
    _require(readiness_report.get("outer_gate") == "STEP6000_CONDITIONAL_GENERATION_READINESS_RESOLVED", "accepted readiness gate changed")
    _require(readiness_status.get("result") == "VALID_COMPLETED", "accepted readiness status changed")
    _require(selection.get("schema") == "step6000_conditional_generation_readiness_fixed_selection.v1", "fixed selection schema changed")
    _require(selection.get("selection_frozen_before_forward") is True, "fixed selection was not frozen before forward")

    source_path = root / TRAINING_SOURCE
    _require(file_sha256(source_path) == TRAINING_SOURCE_SHA256, "canonical TRAIN source identity changed")
    raw_payload = safe_torch_load(source_path, expected_sha256=TRAINING_SOURCE_SHA256, map_location="cpu")
    _require(isinstance(raw_payload, (tuple, list)) and len(raw_payload) == 2, "canonical source schema changed")
    raw_images, labels = raw_payload
    _require(isinstance(raw_images, torch.Tensor) and isinstance(labels, torch.Tensor), "canonical tensors missing")
    _require(raw_images.shape == (60000, 28, 28) and labels.shape == (60000,), "canonical source geometry changed")
    _require(raw_images.dtype == torch.uint8 and labels.dtype == torch.long, "canonical source dtype changed")
    _require(int(raw_images.min()) == 0 and int(raw_images.max()) == 255, "canonical source range changed")
    images = raw_images.unsqueeze(1).to(torch.float32).div(255.0).contiguous()
    labels = labels.to(torch.long).contiguous()
    _validate_image_views(images, "canonical images", 60000)

    train_payload = _load_tensor_payload(root, TRAIN_OLS, TRAIN_OLS_SHA256, "accepted TRAIN OLS")
    dev_payload = _load_tensor_payload(root, DEV_OLS, DEV_OLS_SHA256, "accepted DEV OLS")
    train_z = _tensor(train_payload.get("z_final"), "accepted TRAIN OLS z")
    dev_z = _tensor(dev_payload.get("z_dev"), "accepted DEV OLS z")
    train_indices = _tensor(train_payload.get("source_indices"), "accepted TRAIN source indices").to(torch.long).cpu().contiguous()
    dev_indices = _tensor(dev_payload.get("source_indices"), "accepted DEV source indices").to(torch.long).cpu().contiguous()
    _require(train_z.shape == (55000, OLS_PORT_COUNT) and train_z.dtype == torch.complex64, "accepted TRAIN OLS geometry changed")
    _require(dev_z.shape == (5000, OLS_PORT_COUNT) and dev_z.dtype == torch.complex64, "accepted DEV OLS geometry changed")
    _finite(train_z.real, "accepted TRAIN OLS real")
    _finite(train_z.imag, "accepted TRAIN OLS imaginary")
    _finite(dev_z.real, "accepted DEV OLS real")
    _finite(dev_z.imag, "accepted DEV OLS imaginary")
    _require(torch.equal(torch.sort(torch.cat([train_indices, dev_indices])).values, torch.arange(60000, dtype=torch.long)), "TRAIN/DEV indices do not partition canonical source")
    train_images = images.index_select(0, train_indices)
    train_labels = labels.index_select(0, train_indices)
    dev_images = images.index_select(0, dev_indices)
    dev_labels = labels.index_select(0, dev_indices)
    _require(torch.bincount(dev_labels, minlength=CLASS_COUNT).tolist() == [500] * CLASS_COUNT, "DEV class balance changed")

    payload = _load_tensor_payload(root, LOCALIZATION_PAYLOAD, LOCALIZATION_PAYLOAD_SHA256, "accepted localization payload")
    _require(payload.get("schema") == "step6000_image_manifold_localization_payload.v1", "localization payload schema changed")
    canonical_dev_views = _tensor(payload.get("canonical_dev_views"), "canonical DEV views").cpu().contiguous()
    real_ols_views = _tensor(payload.get("real_ols_views"), "real OLS views").cpu().contiguous()
    s1_views = _tensor(payload.get("s1_views"), "S1 views").cpu().contiguous()
    payload_dev_labels = _tensor(payload.get("dev_labels"), "payload DEV labels").to(torch.long).cpu().contiguous()
    requested = _tensor(payload.get("requested_labels"), "payload requested labels").to(torch.long).cpu().contiguous()
    _validate_image_views(canonical_dev_views, "canonical DEV views", 5000)
    _validate_image_views(real_ols_views, "real OLS views", 5000)
    _validate_image_views(s1_views, "S1 views", 2000)
    _require(torch.equal(payload_dev_labels, dev_labels), "payload DEV labels differ from accepted DEV labels")
    _require(torch.equal(requested, torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(200)), "S1 requested labels are not balanced")
    _require(torch.equal(canonical_dev_views, dev_images), "canonical DEV payload views differ from source images")

    predictions = _load_tensor_payload(root, LOCALIZATION_PREDICTIONS, LOCALIZATION_PREDICTIONS_SHA256, "accepted predictions")
    _require(predictions.get("schema") == "step6000_image_manifold_localization_predictions.v1", "prediction schema changed")
    for source_key, rows in (("canonical_dev", 5000), ("real_ols_step6000", 5000), ("s1_step6000", 2000)):
        source_predictions = _mapping(predictions.get(source_key), f"{source_key} predictions")
        for evaluator in EVALUATOR_ORDER:
            pred = _tensor(source_predictions.get(evaluator), f"{source_key} {evaluator} prediction").to(torch.long).cpu()
            _require(pred.shape == (rows,) and bool(((pred >= 0) & (pred < CLASS_COUNT)).all()), f"{source_key} {evaluator} prediction geometry changed")

    strata = _load_json(root, LOCALIZATION_STRATA, LOCALIZATION_STRATA_SHA256, "accepted G1-G4 strata")
    groups = _mapping(strata.get("groups"), "G1-G4 groups")
    group_masks: dict[str, torch.Tensor] = {}
    used = torch.zeros(2000, dtype=torch.bool)
    expected_counts = {"G1": 202, "G2": 631, "G3": 94, "G4": 1073}
    for name in ("G1", "G2", "G3", "G4"):
        indices = torch.tensor(_mapping(groups.get(name), name).get("indices"), dtype=torch.long)
        _require(indices.numel() == expected_counts[name], f"{name} size changed")
        _require(bool(((indices >= 0) & (indices < 2000)).all()), f"{name} index out of range")
        _require(not bool(used.index_select(0, indices).any()), f"{name} overlaps prior strata")
        used[indices] = True
        group_masks[name] = torch.zeros(2000, dtype=torch.bool).scatter(0, indices, True)
    _require(bool(used.all()), "G1-G4 do not cover S1 bank")

    bank = _load_tensor_payload(root, S1_BANK, S1_BANK_SHA256, "accepted S1 bank")
    winner = _load_tensor_payload(root, S1_WINNER, S1_WINNER_SHA256, "accepted S1 winner")
    _require(bank.get("schema") == "semantically_conditioned_ols_sampler_candidate_banks.v1", "accepted S1 bank schema changed")
    _require(winner.get("schema") == "semantically_conditioned_ols_sampler_frozen_winner.v1", "accepted S1 winner schema changed")
    bank_requested = _tensor(bank.get("requested_labels"), "accepted bank requested labels").to(torch.long).cpu()
    winner_requested = _tensor(winner.get("requested_labels"), "accepted winner requested labels").to(torch.long).cpu()
    _require(torch.equal(bank_requested, requested) and torch.equal(winner_requested, requested), "accepted S1 requested identity changed")
    _require(winner.get("family") == "S1" and winner.get("frozen_before_generator_canary") is True, "accepted S1 winner changed")

    sampled_positions = torch.tensor(selection["sampled"]["positions_in_bank"], dtype=torch.long)
    real_positions = torch.tensor(selection["real_dev"]["positions_in_sorted_dev"], dtype=torch.long)
    _require(sampled_positions.numel() == 100 and real_positions.numel() == 50, "fixed visual identity counts changed")
    _require(torch.equal(selection["sampled"]["requested_labels"] if isinstance(selection["sampled"]["requested_labels"], torch.Tensor) else torch.tensor(selection["sampled"]["requested_labels"], dtype=torch.long), requested.index_select(0, sampled_positions)), "fixed S1 labels changed")

    forward_metrics = _load_json(root, READINESS_FORWARD_METRICS, file_sha256(root / READINESS_FORWARD_METRICS), "accepted diversity metrics")
    return {
        "git": git,
        "localization_report": localization_report,
        "localization_status": localization_status,
        "localization_source_manifest": localization_source_manifest,
        "readiness_report": readiness_report,
        "readiness_status": readiness_status,
        "selection": selection,
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "dev_indices": dev_indices,
        "train_images": train_images,
        "train_labels": train_labels,
        "dev_images": dev_images,
        "dev_labels": dev_labels,
        "canonical_dev_views": canonical_dev_views,
        "real_ols_views": real_ols_views,
        "s1_views": s1_views,
        "requested": requested,
        "predictions": predictions,
        "group_masks": group_masks,
        "strata": strata,
        "forward_metrics": forward_metrics,
        "accepted_hashes": {
            "training_source": TRAINING_SOURCE_SHA256,
            "train_ols": TRAIN_OLS_SHA256,
            "dev_ols": DEV_OLS_SHA256,
            "s1_bank": S1_BANK_SHA256,
            "s1_winner": S1_WINNER_SHA256,
            "localization_report": LOCALIZATION_REPORT_SHA256,
            "localization_payload": LOCALIZATION_PAYLOAD_SHA256,
            "localization_predictions": LOCALIZATION_PREDICTIONS_SHA256,
            "localization_strata": LOCALIZATION_STRATA_SHA256,
            "readiness_report": READINESS_REPORT_SHA256,
            "readiness_status": READINESS_STATUS_SHA256,
            "fixed_selection": READINESS_SELECTION_SHA256,
            "forward_metrics": file_sha256(root / READINESS_FORWARD_METRICS),
        },
    }


def _coords(device: torch.device, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(IMAGE_SIZE, device=device, dtype=dtype).reshape(1, 1, IMAGE_SIZE) / float(IMAGE_SIZE - 1)
    y = torch.arange(IMAGE_SIZE, device=device, dtype=dtype).reshape(1, IMAGE_SIZE, 1) / float(IMAGE_SIZE - 1)
    return x, y


def _weighted_geometry(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float64)
    batch = values.shape[0]
    x_coord, y_coord = _coords(values.device)
    mass = values.sum(dim=(1, 2))
    safe_mass = mass.clamp_min(1.0e-12)
    cx = (values * x_coord).sum(dim=(1, 2)) / safe_mass
    cy = (values * y_coord).sum(dim=(1, 2)) / safe_mass
    dx = x_coord - cx.reshape(-1, 1, 1)
    dy = y_coord - cy.reshape(-1, 1, 1)
    mxx = (values * dx.square()).sum(dim=(1, 2)) / safe_mass
    myy = (values * dy.square()).sum(dim=(1, 2)) / safe_mass
    mxy = (values * dx * dy).sum(dim=(1, 2)) / safe_mass
    angle2 = torch.atan2(2.0 * mxy, mxx - myy)
    cos2 = torch.cos(angle2)
    sin2 = torch.sin(angle2)
    degenerate = mass <= 1.0e-12
    cx = torch.where(degenerate, torch.full_like(cx, 0.5), cx)
    cy = torch.where(degenerate, torch.full_like(cy, 0.5), cy)
    mxx = torch.where(degenerate, torch.zeros_like(mxx), mxx)
    myy = torch.where(degenerate, torch.zeros_like(myy), myy)
    mxy = torch.where(degenerate, torch.zeros_like(mxy), mxy)
    cos2 = torch.where(degenerate, torch.ones_like(cos2), cos2)
    sin2 = torch.where(degenerate, torch.zeros_like(sin2), sin2)
    horizontal = values.sum(dim=2) / safe_mass.reshape(-1, 1)
    vertical = values.sum(dim=1) / safe_mass.reshape(-1, 1)
    horizontal = torch.where(degenerate.reshape(-1, 1), torch.zeros_like(horizontal), horizontal)
    vertical = torch.where(degenerate.reshape(-1, 1), torch.zeros_like(vertical), vertical)
    return torch.cat(
        [
            (mass / float(IMAGE_AREA)).reshape(batch, 1),
            cx.reshape(batch, 1),
            cy.reshape(batch, 1),
            mxx.reshape(batch, 1),
            myy.reshape(batch, 1),
            mxy.reshape(batch, 1),
            cos2.reshape(batch, 1),
            sin2.reshape(batch, 1),
            horizontal,
            vertical,
        ],
        dim=1,
    )


def _binary_geometry(mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(torch.float64)
    base = _weighted_geometry(mask_float)
    area = base[:, :1]
    row_any = mask
    col_any = mask.transpose(1, 2)
    rows = torch.arange(IMAGE_SIZE, device=mask.device, dtype=torch.long).reshape(1, IMAGE_SIZE, 1)
    cols = torch.arange(IMAGE_SIZE, device=mask.device, dtype=torch.long).reshape(1, 1, IMAGE_SIZE)
    row_min = torch.where(row_any, rows, torch.full_like(rows, IMAGE_SIZE)).amin(dim=(1, 2))
    row_max = torch.where(row_any, rows, torch.full_like(rows, -1)).amax(dim=(1, 2))
    col_min = torch.where(col_any, cols, torch.full_like(cols, IMAGE_SIZE)).amin(dim=(1, 2))
    col_max = torch.where(col_any, cols, torch.full_like(cols, -1)).amax(dim=(1, 2))
    nonempty = area[:, 0] > 0.0
    width = torch.where(nonempty, (col_max - col_min + 1).to(torch.float64) / IMAGE_SIZE, torch.zeros_like(area[:, 0]))
    height = torch.where(nonempty, (row_max - row_min + 1).to(torch.float64) / IMAGE_SIZE, torch.zeros_like(area[:, 0]))
    aspect = torch.where(height > 0.0, width / height, torch.zeros_like(width))
    return torch.cat([area, base[:, 1:8], width[:, None], height[:, None], aspect[:, None], base[:, 8:]], dim=1)


def _component_roots(mask: torch.Tensor, connectivity: int) -> tuple[torch.Tensor, torch.Tensor]:
    _require(connectivity in (4, 8), "connectivity must be 4 or 8")
    batch, height, width = mask.shape
    count = height * width
    ids = torch.arange(count, device=mask.device, dtype=torch.long).reshape(1, height, width)
    labels = ids.expand(batch, -1, -1).clone().masked_fill(~mask, count)
    steps = 2 * max(height, width) + 2
    for _ in range(steps):
        padded = F.pad(labels, (1, 1, 1, 1), value=count)
        candidates = [
            padded[:, 1:-1, 1:-1],
            padded[:, :-2, 1:-1],
            padded[:, 2:, 1:-1],
            padded[:, 1:-1, :-2],
            padded[:, 1:-1, 2:],
        ]
        if connectivity == 8:
            candidates.extend(
                [
                    padded[:, :-2, :-2],
                    padded[:, :-2, 2:],
                    padded[:, 2:, :-2],
                    padded[:, 2:, 2:],
                ]
            )
        updated = candidates[0]
        for candidate in candidates[1:]:
            updated = torch.minimum(updated, candidate)
        labels = torch.minimum(labels, updated).masked_fill(~mask, count)
    roots = labels
    root_flags = mask & (roots == ids)
    component_count = root_flags.sum(dim=(1, 2)).to(torch.float64)
    return roots, component_count


def _topology(mask: torch.Tensor) -> torch.Tensor:
    fg_roots, component_count = _component_roots(mask, 8)
    fg_count = mask.sum(dim=(1, 2)).to(torch.float64)
    batch = mask.shape[0]
    height = mask.shape[1] + 2
    width = mask.shape[2] + 2
    background = torch.ones((batch, height, width), device=mask.device, dtype=torch.bool)
    background[:, 1:-1, 1:-1] = ~mask
    bg_roots, _ = _component_roots(background, 4)
    border_roots = torch.cat(
        [
            bg_roots[:, 0, :],
            bg_roots[:, -1, :],
            bg_roots[:, 1:-1, 0],
            bg_roots[:, 1:-1, -1],
        ],
        dim=1,
    )
    border_touch = torch.zeros((batch, height * width), device=mask.device, dtype=torch.bool)
    border_touch.scatter_(1, border_roots, True)
    bg_root_flags = background & (bg_roots == torch.arange(height * width, device=mask.device, dtype=torch.long).reshape(1, height, width))
    bg_root_touch = border_touch.gather(1, bg_roots.reshape(batch, -1).clamp_max(height * width - 1)).reshape(batch, height, width)
    hole_count = (bg_root_flags & ~bg_root_touch).sum(dim=(1, 2)).to(torch.float64)
    del bg_roots, bg_root_flags, bg_root_touch, background
    fg_sizes = torch.zeros((batch, IMAGE_AREA), device=mask.device, dtype=torch.long)
    fg_sizes.scatter_add_(
        1,
        fg_roots.reshape(batch, -1).clamp_max(IMAGE_AREA - 1),
        mask.reshape(batch, -1).to(torch.long),
    )
    largest = fg_sizes[:, :IMAGE_AREA].amax(dim=1).to(torch.float64)
    largest_fraction = torch.where(fg_count > 0.0, largest / fg_count, torch.zeros_like(fg_count))
    euler = component_count - hole_count
    return torch.stack([component_count, hole_count, euler, largest_fraction], dim=1)


def _spatial_morphology(mask: torch.Tensor) -> torch.Tensor:
    batch = mask.shape[0]
    cell = mask.reshape(batch, 4, 7, 4, 7).to(torch.float64).mean(dim=(2, 4)).reshape(batch, 16)
    quadrant = mask.reshape(batch, 2, 14, 2, 14).to(torch.float64).mean(dim=(2, 4)).reshape(batch, 4)
    return torch.cat([cell, quadrant], dim=1)


def _structural_signature_batch(images: torch.Tensor) -> torch.Tensor:
    _require(images.ndim == 4 and tuple(images.shape[1:]) == (1, IMAGE_SIZE, IMAGE_SIZE), "signature input geometry changed")
    _finite(images, "signature input")
    _require(float(images.min()) >= 0.0 and float(images.max()) <= 1.0, "signature input range changed")
    values = images[:, 0].to(torch.float64)
    parts = [_weighted_geometry(values)]
    for threshold in THRESHOLDS:
        mask = values >= threshold
        parts.append(_binary_geometry(mask))
        parts.append(_topology(mask))
        parts.append(_spatial_morphology(mask))
    result = torch.cat(parts, dim=1)
    _require(result.shape[1] == SIGNATURE_DIMENSION, f"signature dimension is {result.shape[1]}, expected {SIGNATURE_DIMENSION}")
    _finite(result, "structural signature")
    return result


def structural_signature(images: torch.Tensor, *, device: torch.device | None = None, batch_size: int = STRUCTURAL_BATCH_SIZE) -> torch.Tensor:
    """Compute the frozen 337D signature in float64, preserving row order."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows: list[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        end = min(images.shape[0], start + batch_size)
        rows.append(_structural_signature_batch(images[start:end].to(device)).cpu())
    result = torch.cat(rows, dim=0).to(torch.float64).contiguous()
    _require(result.shape == (images.shape[0], SIGNATURE_DIMENSION), "structural signature shape changed")
    return result


def _standardize(fit_signature: torch.Tensor, query_signatures: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    _require(fit_signature.ndim == 2 and fit_signature.shape[1] == SIGNATURE_DIMENSION, "fit signature geometry changed")
    mean = fit_signature.mean(dim=0, dtype=torch.float64)
    population_std = fit_signature.std(dim=0, correction=0)
    scale = population_std.clamp_min(NORMALIZATION_EPS)
    fit_standardized = ((fit_signature - mean) / scale).contiguous()
    queries = {name: ((value.to(torch.float64) - mean) / scale).contiguous() for name, value in query_signatures.items()}
    _finite(mean, "structural normalization mean")
    _finite(scale, "structural normalization scale")
    _finite(fit_standardized, "standardized fit signature")
    for name, value in queries.items():
        _finite(value, f"standardized {name} signature")
    return fit_standardized, queries, mean, scale


def _class_distance(
    query: torch.Tensor,
    references: torch.Tensor,
    *,
    query_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    device: torch.device,
    need_index: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    query = query.to(torch.float64)
    references = references.to(torch.float64)
    query_labels = query_labels.to(torch.long).cpu()
    reference_labels = reference_labels.to(torch.long).cpu()
    output = torch.empty(query.shape[0], dtype=torch.float64)
    indices = torch.empty(query.shape[0], dtype=torch.long) if need_index else None
    ref_by_class = [torch.nonzero(reference_labels == digit, as_tuple=False).flatten() for digit in range(CLASS_COUNT)]
    for digit in range(CLASS_COUNT):
        query_positions = torch.nonzero(query_labels == digit, as_tuple=False).flatten()
        reference_positions = ref_by_class[digit]
        _require(reference_positions.numel() > 0, f"structural reference class {digit} is empty")
        ref = references.index_select(0, reference_positions).to(device)
        ref_norm = ref.square().sum(dim=1)
        for start in range(0, query_positions.numel(), DISTANCE_QUERY_BLOCK):
            local_positions = query_positions[start : start + DISTANCE_QUERY_BLOCK]
            q = query.index_select(0, local_positions).to(device)
            distances_sq = (q.square().sum(dim=1, keepdim=True) + ref_norm.unsqueeze(0) - 2.0 * q.matmul(ref.transpose(0, 1))).clamp_min(0.0)
            nearest_sq, nearest_local = distances_sq.min(dim=1)
            values = (nearest_sq / float(SIGNATURE_DIMENSION)).sqrt().cpu()
            output[local_positions] = values
            if indices is not None:
                indices[local_positions] = reference_positions.index_select(0, nearest_local.cpu())
    return output, indices


def _centroid_distances(query: torch.Tensor, centroids: torch.Tensor, labels: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_device = query.to(device=device, dtype=torch.float64)
    centroids_device = centroids.to(device=device, dtype=torch.float64)
    centroid_norm = centroids_device.square().sum(dim=1)
    values: list[torch.Tensor] = []
    for start in range(0, query_device.shape[0], DISTANCE_QUERY_BLOCK):
        block = query_device[start : start + DISTANCE_QUERY_BLOCK]
        dist_sq = (block.square().sum(dim=1, keepdim=True) + centroid_norm.unsqueeze(0) - 2.0 * block.matmul(centroids_device.transpose(0, 1))).clamp_min(0.0)
        values.append((dist_sq / float(SIGNATURE_DIMENSION)).sqrt().cpu())
    all_distances = torch.cat(values, dim=0)
    labels_cpu = labels.to(torch.long).cpu()
    d_req = all_distances.gather(1, labels_cpu.reshape(-1, 1)).squeeze(1)
    masked = all_distances.clone()
    masked.scatter_(1, labels_cpu.reshape(-1, 1), float("inf"))
    d_alt = masked.min(dim=1).values
    return d_req, d_alt, all_distances.argmin(dim=1)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.to(torch.float64).reshape(-1)
    _require(values.numel() > 0, "cannot summarize empty values")
    q = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=torch.float64)
    result = torch.quantile(values, q).tolist()
    return {name: float(value) for name, value in zip(("q10", "q25", "q50", "q75", "q90"), result, strict=True)}


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.to(torch.float64).reshape(-1)
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std_population": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": _quantiles(values),
    }


def _class_rates(values: torch.Tensor, labels: torch.Tensor, *, boolean: bool = False) -> dict[str, Any]:
    labels = labels.to(torch.long).cpu()
    values = values.cpu()
    rows: dict[str, Any] = {}
    rates = []
    for digit in range(CLASS_COUNT):
        subset = values[labels == digit]
        _require(subset.numel() > 0, f"class {digit} is empty in metric")
        if boolean:
            rate = float(subset.to(torch.float64).mean())
            rows[str(digit)] = {"count": int(subset.numel()), "rate": rate}
            rates.append(rate)
        else:
            rows[str(digit)] = {"count": int(subset.numel()), "summary": _summary(subset)}
    if boolean:
        rows["macro"] = float(sum(rates) / CLASS_COUNT)
    return rows


def _topology_summary(raw_signature: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for threshold_index, threshold in enumerate(THRESHOLDS):
        raw_topology = raw_signature[:, _topology_slice(threshold_index)].to(torch.float64)
        per_threshold: dict[str, Any] = {}
        for feature_index, feature_name in enumerate(TOPOLOGY_NAMES):
            values = raw_topology[:, feature_index]
            class_values: dict[str, Any] = {}
            for digit in range(CLASS_COUNT):
                subset = values[labels == digit]
                if subset.numel() == 0:
                    class_values[str(digit)] = {"count": 0}
                    continue
                if feature_name in ("component_count", "hole_count", "euler_characteristic"):
                    unique, counts = torch.unique(subset.to(torch.long), sorted=True, return_counts=True)
                    class_values[str(digit)] = {
                        "counts": {str(int(key)): int(value) for key, value in zip(unique.tolist(), counts.tolist(), strict=True)},
                        "summary": _summary(subset),
                    }
                else:
                    class_values[str(digit)] = {"summary": _summary(subset)}
            per_threshold[feature_name] = class_values
        result[f"threshold_{threshold:.2f}"] = per_threshold
    return result


def _evaluate_source(
    name: str,
    raw_signature: torch.Tensor,
    standardized_signature: torch.Tensor,
    labels: torch.Tensor,
    fit_standardized: torch.Tensor,
    fit_labels: torch.Tensor,
    centroids: torch.Tensor,
    q95: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    labels = labels.to(torch.long).cpu().contiguous()
    nearest, nearest_index = _class_distance(
        standardized_signature,
        fit_standardized,
        query_labels=labels,
        reference_labels=fit_labels,
        device=device,
        need_index=True,
    )
    d_req, d_alt, structural_class = _centroid_distances(standardized_signature, centroids, labels, device)
    margin = d_alt - d_req
    q95_rows = q95.index_select(0, labels)
    support = nearest <= q95_rows
    result = {
        "name": name,
        "labels": labels,
        "raw_signature": raw_signature.to(torch.float64).contiguous(),
        "nearest_same_class_distance": nearest,
        "nearest_same_class_index": nearest_index,
        "q95_rows": q95_rows,
        "support": support,
        "d_req": d_req,
        "d_alt": d_alt,
        "margin": margin,
        "structural_class": structural_class,
        "topology": _topology_summary(raw_signature, labels),
        "support_by_class": _class_rates(support, labels, boolean=True),
        "distance_by_class": _class_rates(nearest, labels),
        "margin_by_class": _class_rates(margin, labels),
        "margin_summary": _summary(margin),
        "positive_margin_fraction": float((margin > 0.0).to(torch.float64).mean()),
        "positive_margin_by_class": _class_rates(margin > 0.0, labels, boolean=True),
    }
    return result


def _q95(values: torch.Tensor) -> float:
    return float(torch.quantile(values.to(torch.float64), torch.tensor(0.95, dtype=torch.float64), interpolation="linear"))


def _bootstrap_difference(
    left: torch.Tensor,
    left_labels: torch.Tensor,
    right: torch.Tensor,
    right_labels: torch.Tensor,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    classes: Sequence[int] = tuple(range(CLASS_COUNT)),
) -> dict[str, Any]:
    left = left.to(torch.float64).cpu()
    right = right.to(torch.float64).cpu()
    left_labels = left_labels.to(torch.long).cpu()
    right_labels = right_labels.to(torch.long).cpu()
    class_list = tuple(int(value) for value in classes)
    _require(class_list, "bootstrap class set is empty")
    left_by_class = {digit: left[left_labels == digit] for digit in class_list}
    right_by_class = {digit: right[right_labels == digit] for digit in class_list}
    _require(all(value.numel() > 0 for value in left_by_class.values()), "left bootstrap class is empty")
    _require(all(value.numel() > 0 for value in right_by_class.values()), "right bootstrap class is empty")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.empty(replicates, dtype=torch.float64)
    for replicate in range(replicates):
        class_differences = []
        for digit in class_list:
            left_values = left_by_class[digit]
            right_values = right_by_class[digit]
            left_indices = torch.randint(left_values.numel(), (left_values.numel(),), generator=generator)
            right_indices = torch.randint(right_values.numel(), (right_values.numel(),), generator=generator)
            class_differences.append(left_values.index_select(0, left_indices).mean() - right_values.index_select(0, right_indices).mean())
        samples[replicate] = torch.stack(class_differences).mean()
    estimate = float(torch.stack([left_by_class[digit].mean() - right_by_class[digit].mean() for digit in class_list]).mean())
    low, high = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {
        "left": "left",
        "right": "right",
        "classes": list(class_list),
        "replicates": replicates,
        "seed": seed,
        "estimate": estimate,
        "ci95": [float(low), float(high)],
        "materially_different": bool(low > 0.0 or high < 0.0),
        "close": bool(low <= 0.0 <= high),
    }


def _bootstrap_rate(values: torch.Tensor, labels: torch.Tensor, *, seed: int = BOOTSTRAP_SEED, replicates: int = BOOTSTRAP_REPLICATES, classes: Sequence[int] = tuple(range(CLASS_COUNT))) -> dict[str, Any]:
    values = values.to(torch.float64).cpu()
    labels = labels.to(torch.long).cpu()
    class_list = tuple(int(value) for value in classes)
    by_class = {digit: values[labels == digit] for digit in class_list}
    _require(all(value.numel() > 0 for value in by_class.values()), "bootstrap class is empty")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.empty(replicates, dtype=torch.float64)
    for replicate in range(replicates):
        rates = []
        for digit in class_list:
            values_class = by_class[digit]
            indices = torch.randint(values_class.numel(), (values_class.numel(),), generator=generator)
            rates.append(values_class.index_select(0, indices).mean())
        samples[replicate] = torch.stack(rates).mean()
    estimate = float(torch.stack([value.mean() for value in by_class.values()]).mean())
    low, high = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"classes": list(class_list), "replicates": replicates, "seed": seed, "estimate": estimate, "ci95": [float(low), float(high)]}


def _consensus_vector(prediction_payload: Mapping[str, Any], source_key: str, labels: torch.Tensor) -> torch.Tensor:
    source_predictions = _mapping(prediction_payload.get(source_key), f"{source_key} predictions")
    matrix = torch.stack([_tensor(source_predictions[name], f"{source_key} {name}").to(torch.long).cpu() for name in EVALUATOR_ORDER], dim=1)
    labels = labels.to(torch.long).cpu()
    return (matrix == labels.reshape(-1, 1)).sum(dim=1) >= 3


def _cross_table(source: Mapping[str, Any], consensus: torch.Tensor) -> dict[str, Any]:
    support = _tensor(source["support"], "support").to(torch.bool).cpu()
    consensus = consensus.to(torch.bool).cpu()
    cells = {
        "high_support_consensus": support & consensus,
        "high_support_disagreement": support & ~consensus,
        "low_support_consensus": ~support & consensus,
        "low_support_disagreement": ~support & ~consensus,
    }
    return {name: {"count": int(mask.sum()), "fraction": float(mask.to(torch.float64).mean())} for name, mask in cells.items()}


def _stratum_reports(s1: Mapping[str, Any], requested: torch.Tensor, group_masks: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("G1", "G2", "G3", "G4"):
        mask = group_masks[name]
        labels = requested[mask]
        support = _tensor(s1["support"], "S1 support")[mask]
        distance = _tensor(s1["nearest_same_class_distance"], "S1 distance")[mask]
        margin = _tensor(s1["margin"], "S1 margin")[mask]
        result[name] = {
            "rows": int(mask.sum()),
            "requested_class_histogram": torch.bincount(labels, minlength=CLASS_COUNT).tolist(),
            "structural_support_rate": float(support.to(torch.float64).mean()),
            "distance": _summary(distance),
            "margin": _summary(margin),
            "positive_margin_fraction": float((margin > 0.0).to(torch.float64).mean()),
            "topology": _topology_summary(_tensor(s1["raw_signature"], "S1 raw signature")[mask], labels),
        }
    pairwise: dict[str, Any] = {}
    for right_name in ("G2", "G3", "G4"):
        left_mask = group_masks["G1"]
        right_mask = group_masks[right_name]
        overlap_classes = [
            digit
            for digit in range(CLASS_COUNT)
            if bool(((requested[left_mask] == digit).any()) and ((requested[right_mask] == digit).any()))
        ]
        if overlap_classes:
            pairwise[f"G1_minus_{right_name}"] = _bootstrap_difference(
                _tensor(s1["support"], "S1 support")[left_mask].to(torch.float64),
                requested[left_mask],
                _tensor(s1["support"], "S1 support")[right_mask].to(torch.float64),
                requested[right_mask],
                classes=overlap_classes,
            )
        else:
            pairwise[f"G1_minus_{right_name}"] = {"estimable": False, "classes": []}
    result["bootstrap_differences"] = pairwise
    return result


def _diversity_interaction(forward_metrics: Mapping[str, Any]) -> dict[str, Any]:
    s1 = _mapping(_mapping(_mapping(forward_metrics.get("s1"), "forward metrics s1").get("6000"), "forward metrics s1 step6000").get("diversity"), "S1 diversity")
    s0 = _mapping(_mapping(_mapping(forward_metrics.get("s0"), "forward metrics s0").get("6000"), "forward metrics s0 step6000").get("diversity"), "S0 diversity")
    s1_aggregate = _mapping(s1.get("aggregate"), "S1 aggregate diversity")
    s0_aggregate = _mapping(s0.get("aggregate"), "S0 aggregate diversity")
    result: dict[str, Any] = {
        "source": "accepted readiness forward_metrics step6000; no new diversity computation",
        "S1": {},
        "S0": {},
        "comparison": {},
        "accepted_severe_collapse_policy": {
            "near_duplicate_fraction": ">=0.50",
            "dominant_fraction": ">=0.80 when present in accepted metric",
            "variance_ratio": "<=0.10 when present in accepted metric",
        },
    }
    for source_name, aggregate in (("S1", s1_aggregate), ("S0", s0_aggregate)):
        for metric_name in ("E_C_feature", "raw_cmos_metric_panel_28x28"):
            value = _mapping(aggregate.get(metric_name), f"{source_name} {metric_name} diversity")
            result[source_name][metric_name] = {
                key: value[key]
                for key in (
                    "exact_duplicate_row_count",
                    "exact_unique_row_count",
                    "near_duplicate_pair_count",
                    "near_duplicate_pair_fraction",
                    "near_duplicate_threshold",
                    "near_duplicate_threshold_definition",
                    "pairwise_distance",
                    "effective_rank",
                )
                if key in value
            }
        if "latent_z" in s1 and source_name == "S1":
            result[source_name]["latent_z"] = s1["latent_z"]
    for metric_name in ("E_C_feature", "raw_cmos_metric_panel_28x28"):
        if metric_name in result["S1"] and metric_name in result["S0"]:
            s1_pairwise = _mapping(result["S1"][metric_name].get("pairwise_distance"), f"S1 {metric_name} pairwise")
            s0_pairwise = _mapping(result["S0"][metric_name].get("pairwise_distance"), f"S0 {metric_name} pairwise")
            result["comparison"][metric_name] = {
                "S1_minus_S0_pairwise_mean": float(s1_pairwise["mean"] - s0_pairwise["mean"]),
                "S1_over_S0_pairwise_mean": float(s1_pairwise["mean"] / max(float(s0_pairwise["mean"]), 1.0e-12)),
                "S1_exact_duplicate_count": result["S1"][metric_name].get("exact_duplicate_row_count"),
                "S1_near_duplicate_fraction": result["S1"][metric_name].get("near_duplicate_pair_fraction"),
            }
    return result


def _has_severe_diversity_collapse(diversity: Mapping[str, Any]) -> bool:
    s1 = _mapping(diversity.get("S1"), "S1 diversity result")
    for metric_name in ("E_C_feature", "raw_cmos_metric_panel_28x28"):
        value = _mapping(s1.get(metric_name), f"S1 {metric_name} diversity result")
        if float(value.get("near_duplicate_pair_fraction", 0.0)) >= 0.50:
            return True
    return False


def _classify(
    sources: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    strata: Mapping[str, Any],
    diversity: Mapping[str, Any],
) -> tuple[str, str, str]:
    s1 = sources["s1_step6000"]
    s1_dev = comparisons["s1_minus_canonical_dev"]
    s1_real = comparisons["s1_minus_real_ols"]
    real_dev = comparisons["real_ols_minus_canonical_dev"]
    margin_ci = s1["positive_margin_bootstrap"]["ci95"]
    positive_margins = margin_ci[0] > 0.5
    no_severe_per_class = not any(
        float(value["ci95"][1]) < 0.0
        for value in comparisons["s1_minus_canonical_dev_per_class"].values()
    )
    g1 = strata["G1"]["structural_support_rate"]
    g4 = strata["G4"]["structural_support_rate"]
    disagreement_retained = g4 >= g1 or not strata["bootstrap_differences"]["G1_minus_G4"].get("materially_different", False)
    diversity_collapsed = _has_severe_diversity_collapse(diversity)
    if (
        s1_dev["close"]
        and s1_real["close"]
        and positive_margins
        and no_severe_per_class
        and disagreement_retained
        and not diversity_collapsed
    ):
        return (
            "NON_CLASSIFIER_STRUCTURAL_SEMANTICS_SUPPORTED_DESPITE_CLASSIFIER_DIVERGENCE",
            "S1 structural support is control-compatible, margins are predominantly positive, disagreement strata retain support, and the accepted diversity controls do not indicate severe collapse.",
            "DEFINE_FINAL_STRUCTURAL_OR_SEALED_GENERATION_CONFIRMATION_V1",
        )
    if real_dev["close"] and s1_real["materially_different"] and s1_real["estimate"] < 0.0:
        return (
            "STEP6000_REAL_VS_S1_STRUCTURAL_TRANSFER_GAP_SUPPORTED",
            "real-OLS structural support is close to canonical DEV while S1 is materially lower.",
            "ATTRIBUTE_STEP6000_REAL_VS_S1_GENERATOR_STRUCTURAL_TRANSFER_V1",
        )
    per_class_failures = [
        int(digit)
        for digit, comparison in comparisons["s1_minus_canonical_dev_per_class"].items()
        if float(comparison["ci95"][1]) < 0.0
    ]
    if per_class_failures and len(per_class_failures) < CLASS_COUNT and float(s1["support_by_class"]["macro"]) >= 0.5:
        return (
            "STEP6000_CLASS_SPECIFIC_STRUCTURAL_SEMANTIC_GAP_SUPPORTED",
            f"global structural support masks a bounded affected class subset: {per_class_failures}.",
            "class-conditioned morphology attribution",
        )
    if diversity_collapsed and s1["support_by_class"]["macro"] > 0.0:
        return (
            "STRUCTURAL_SEMANTIC_SUPPORT_WITH_DIVERSITY_TRADEOFF",
            "structural support is present but the accepted S1 diversity diagnostics indicate severe collapse.",
            "NONE",
        )
    if (
        (s1_dev["materially_different"] and s1_dev["estimate"] < 0.0)
        or (s1_real["materially_different"] and s1_real["estimate"] < 0.0)
    ) and not positive_margins:
        return (
            "STEP6000_STRUCTURAL_SEMANTIC_REALIZATION_GAP_SUPPORTED",
            "S1 structural support materially lags the canonical/real controls and requested-class margins are not predominantly positive.",
            "DEFINE_EVALUATOR_AGNOSTIC_STRUCTURAL_IMAGE_ALIGNMENT_V1",
        )
    return (
        "NON_CLASSIFIER_STRUCTURAL_SEMANTIC_CONFIRMATION_MIXED",
        "The frozen structural support, class, stratum, and diversity evidence does not select a unique mechanism.",
        "NONE",
    )


def _alignment_helper() -> Any:
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError as error:  # pragma: no cover - environment-specific QA dependency
        raise StructuralConfirmationError("nature-figure audit_panel_alignment.py is required for figure QA") from error
    return require_matplotlib_panel_alignment


def _figure_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False


def _panel_label(axis: Any, label: str = "a") -> None:
    offset = ScaledTranslation(-3 / 72, 3 / 72, axis.figure.dpi_scale_trans)
    axis.text(0, 1, label, transform=axis.transAxes + offset, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _save_single_figure(fig: Any, axis: Any, target: Path, stem: str) -> dict[str, Any]:
    _panel_label(axis)
    fig.tight_layout(pad=0.8)
    alignment = _alignment_helper()
    alignment(
        fig,
        axes=[axis],
        panel_ids=["a"],
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        json_out=str(target / f"{stem}.alignment.json"),
        overlay_svg=str(target / f"{stem}.alignment.svg"),
    )
    outputs: dict[str, Any] = {}
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": RASTER_DPI}), (".tiff", {"dpi": RASTER_DPI})):
        extension = suffix[1:]
        path = target / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs[extension] = {"path": path.name, "sha256": file_sha256(path)}
    outputs["alignment_json"] = {"path": f"{stem}.alignment.json", "sha256": file_sha256(target / f"{stem}.alignment.json")}
    outputs["alignment_svg"] = {"path": f"{stem}.alignment.svg", "sha256": file_sha256(target / f"{stem}.alignment.svg")}
    plt.close(fig)
    return outputs


def _figure_support(sources: Mapping[str, Mapping[str, Any]], target: Path) -> dict[str, Any]:
    colors = {"canonical_dev": "#767676", "real_ols_step6000": "#3775BA", "s1_step6000": "#B64342"}
    labels = {"canonical_dev": "canonical DEV", "real_ols_step6000": "real OLS", "s1_step6000": "S1"}
    values = np.asarray([[float(sources[name]["support_by_class"][str(digit)]["rate"]) for digit in range(CLASS_COUNT)] for name in SOURCE_ORDER])
    fig, axis = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(CLASS_COUNT)
    width = 0.25
    for index, name in enumerate(SOURCE_ORDER):
        axis.bar(x + (index - 1) * width, values[index], width=width, color=colors[name], label=labels[name], edgecolor="black", linewidth=0.35)
    axis.set_xticks(x, [str(digit) for digit in range(CLASS_COUNT)])
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("class")
    axis.set_ylabel("structural support rate")
    axis.set_title("Requested-class structural support by source")
    axis.legend(ncol=3, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    return _save_single_figure(fig, axis, target, "structural_support_by_class")


def _figure_distance(sources: Mapping[str, Mapping[str, Any]], target: Path) -> dict[str, Any]:
    values = [_tensor(sources[name]["nearest_same_class_distance"], f"{name} distance").numpy() for name in SOURCE_ORDER]
    fig, axis = plt.subplots(figsize=(5.8, 3.2))
    box = axis.boxplot(values, labels=["canonical DEV", "real OLS", "S1"], patch_artist=True, widths=0.58, showfliers=False)
    for patch, color in zip(box["boxes"], ("#767676", "#3775BA", "#B64342"), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axis.set_ylabel("nearest same-class structural distance")
    axis.set_title("Structural local-support distances")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    return _save_single_figure(fig, axis, target, "structural_distance_distributions")


def _figure_margin(sources: Mapping[str, Mapping[str, Any]], target: Path) -> dict[str, Any]:
    values = [_tensor(sources[name]["margin"], f"{name} margin").numpy() for name in SOURCE_ORDER]
    fig, axis = plt.subplots(figsize=(5.8, 3.2))
    box = axis.boxplot(values, labels=["canonical DEV", "real OLS", "S1"], patch_artist=True, widths=0.58, showfliers=False)
    for patch, color in zip(box["boxes"], ("#767676", "#3775BA", "#B64342"), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axis.axhline(0.0, color="#272727", linestyle="--", linewidth=0.8)
    axis.set_ylabel("structural margin (d_alt − d_req)")
    axis.set_title("Requested-class structural margins")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    return _save_single_figure(fig, axis, target, "structural_margin_distributions")


def _figure_strata(strata: Mapping[str, Any], target: Path) -> dict[str, Any]:
    names = ["G1", "G2", "G3", "G4"]
    values = [float(strata[name]["structural_support_rate"]) for name in names]
    fig, axis = plt.subplots(figsize=(5.0, 3.2))
    bars = axis.bar(names, values, color=["#2E9E44", "#8BCF8B", "#E9A6A1", "#B64342"], edgecolor="black", linewidth=0.35)
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=6)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("accepted classifier-consensus stratum")
    axis.set_ylabel("structural support rate")
    axis.set_title("Structural support across frozen G1–G4 strata")
    return _save_single_figure(fig, axis, target, "structural_support_by_consensus_stratum")


def _topology_mean_matrix(source: Mapping[str, Any], feature_name: str) -> np.ndarray:
    matrix = np.zeros((CLASS_COUNT, len(THRESHOLDS)), dtype=float)
    raw = _tensor(source["raw_signature"], "source raw signature")
    labels = _tensor(source["labels"], "source labels").to(torch.long)
    for threshold_index, _ in enumerate(THRESHOLDS):
        column = raw[:, _topology_slice(threshold_index)]
        feature_index = TOPOLOGY_NAMES.index(feature_name)
        values = column[:, feature_index]
        for digit in range(CLASS_COUNT):
            matrix[digit, threshold_index] = float(values[labels == digit].mean())
    return matrix


def _figure_topology(sources: Mapping[str, Mapping[str, Any]], target: Path) -> dict[str, Any]:
    matrices = [_topology_mean_matrix(sources[name], "hole_count") for name in SOURCE_ORDER]
    matrix = np.concatenate(matrices, axis=1)
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    im = axis.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0, vmax=max(1.0, float(matrix.max())))
    axis.set_yticks(np.arange(CLASS_COUNT), [str(digit) for digit in range(CLASS_COUNT)])
    axis.set_xticks(np.arange(9), [f"{source}\nτ={threshold:.2f}" for source in ("DEV", "real", "S1") for threshold in THRESHOLDS], fontsize=5)
    axis.set_xlabel("source and threshold")
    axis.set_ylabel("class")
    axis.set_title("Mean hole-count topology summary; exact frequencies in report")
    fig.colorbar(im, ax=axis, label="mean hole count", fraction=0.046, pad=0.04)
    return _save_single_figure(fig, axis, target, "topology_class_threshold_summary")


def _figure_cross_table(cross_tables: Mapping[str, Any], target: Path) -> dict[str, Any]:
    cell_names = ("high_support_consensus", "high_support_disagreement", "low_support_consensus", "low_support_disagreement")
    matrix = np.asarray([[float(cross_tables[name][cell]["fraction"]) for cell in cell_names] for name in SOURCE_ORDER])
    fig, axis = plt.subplots(figsize=(6.5, 2.8))
    im = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_yticks(np.arange(len(SOURCE_ORDER)), ["canonical DEV", "real OLS", "S1"])
    axis.set_xticks(np.arange(len(cell_names)), ["high + consensus", "high + disagreement", "low + consensus", "low + disagreement"])
    for tick in axis.get_xticklabels():
        tick.set_rotation(22)
        tick.set_rotation_mode("anchor")
        tick.set_ha("right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > 0.55 else "black"
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color=color, fontsize=6)
    axis.set_title("Existing evaluator consensus × structural support")
    fig.colorbar(im, ax=axis, label="fraction", fraction=0.035, pad=0.03)
    return _save_single_figure(fig, axis, target, "classifier_consensus_structural_support_cross_table")


def _figure_examples(authorities: Mapping[str, Any], target: Path) -> dict[str, Any]:
    selection = authorities["selection"]
    real_positions = torch.tensor(selection["real_dev"]["positions_in_sorted_dev"], dtype=torch.long)
    s1_positions = torch.tensor(selection["sampled"]["positions_in_bank"], dtype=torch.long)
    real_first = real_positions[::5]
    s1_first = s1_positions[::10]
    canonical = authorities["canonical_dev_views"].index_select(0, real_first).numpy()[:, 0]
    real = authorities["real_ols_views"].index_select(0, real_first).numpy()[:, 0]
    s1 = authorities["s1_views"].index_select(0, s1_first).numpy()[:, 0]
    canvas = np.ones((3 * IMAGE_SIZE + 2 * 4, CLASS_COUNT * IMAGE_SIZE + (CLASS_COUNT - 1) * 2), dtype=float)
    rows = [canonical, real, s1]
    for row_index, row_images in enumerate(rows):
        y0 = row_index * (IMAGE_SIZE + 4)
        for digit, image in enumerate(row_images):
            x0 = digit * (IMAGE_SIZE + 2)
            canvas[y0 : y0 + IMAGE_SIZE, x0 : x0 + IMAGE_SIZE] = image
    fig, axis = plt.subplots(figsize=(8.0, 2.7))
    axis.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    axis.set_xticks([digit * (IMAGE_SIZE + 2) + IMAGE_SIZE / 2 for digit in range(CLASS_COUNT)], [str(digit) for digit in range(CLASS_COUNT)], fontsize=6)
    axis.set_yticks([IMAGE_SIZE / 2, IMAGE_SIZE + 4 + IMAGE_SIZE / 2, 2 * (IMAGE_SIZE + 4) + IMAGE_SIZE / 2], ["canonical DEV", "real OLS", "S1"], fontsize=6)
    axis.set_title("Fixed accepted class-major structural examples; global [0,1] display")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return _save_single_figure(fig, axis, target, "fixed_structural_examples")


def _build_preflight(authorities: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    split = _make_structural_split(authorities["train_labels"])
    _require(torch.unique(torch.cat([split["fit_positions"], split["validation_positions"]])).numel() == authorities["train_labels"].numel(), "preflight structural split overlap")
    smoke_inputs = torch.cat(
        [
            authorities["train_images"][:1],
            authorities["dev_images"][:1],
            authorities["s1_views"][:1],
        ],
        dim=0,
    )
    first = structural_signature(smoke_inputs, device=device, batch_size=3)
    second = structural_signature(smoke_inputs, device=device, batch_size=3)
    _require(first.shape == (3, SIGNATURE_DIMENSION), "preflight signature dimension mismatch")
    _require(torch.equal(first, second), "preflight signature is not deterministic")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "accepted_authorities_verified": True,
        "descriptor_dimension": SIGNATURE_DIMENSION,
        "descriptor_dimension_check": True,
        "deterministic_repeatability": True,
        "fit_validation_isolated": True,
        "canonical_dev_absent_from_fit": True,
        "s1_absent_from_normalization_and_q95": True,
        "all_descriptors_finite": True,
        "all_expected_classes_present": True,
        "g1_g4_total": 2000,
        "generator_forward": False,
        "generator_backward": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "sampler_refit": False,
        "new_classifier": False,
        "new_evaluator": False,
        "protected_reads": 0,
        "test_reads": 0,
        "sealed_reads": 0,
        "cuda_available": bool(torch.cuda.is_available()),
        "compute_device": str(device),
    }


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _class_histogram(labels: torch.Tensor) -> list[int]:
    counts = torch.bincount(labels.to(torch.long).cpu(), minlength=CLASS_COUNT)
    _require(counts.numel() == CLASS_COUNT and bool((counts > 0).all()), "class histogram is incomplete")
    return [int(value) for value in counts.tolist()]


def _source_public(source: Mapping[str, Any]) -> dict[str, Any]:
    labels = _tensor(source["labels"], f"{source['name']} labels").to(torch.long).cpu()
    support = _tensor(source["support"], f"{source['name']} support").to(torch.bool).cpu()
    distance = _tensor(source["nearest_same_class_distance"], f"{source['name']} distance").to(torch.float64).cpu()
    q95_rows = _tensor(source["q95_rows"], f"{source['name']} q95 rows").to(torch.float64).cpu()
    d_req = _tensor(source["d_req"], f"{source['name']} requested centroid distance").to(torch.float64).cpu()
    d_alt = _tensor(source["d_alt"], f"{source['name']} alternative centroid distance").to(torch.float64).cpu()
    margin = _tensor(source["margin"], f"{source['name']} margin").to(torch.float64).cpu()
    structural_class = _tensor(source["structural_class"], f"{source['name']} structural class").to(torch.long).cpu()
    return {
        "name": source["name"],
        "rows": int(labels.numel()),
        "label_source": SOURCE_LABEL_NAMES[source["name"]],
        "label_histogram": _class_histogram(labels),
        "structural_support_rate": float(support.to(torch.float64).mean()),
        "support_by_class": source["support_by_class"],
        "nearest_same_class_distance": _summary(distance),
        "nearest_same_class_distance_by_class": source["distance_by_class"],
        "q95_rows": _summary(q95_rows),
        "requested_centroid_distance": _summary(d_req),
        "alternative_centroid_distance": _summary(d_alt),
        "margin": source["margin_summary"],
        "margin_by_class": source["margin_by_class"],
        "positive_margin_fraction": float((margin > 0.0).to(torch.float64).mean()),
        "positive_margin_by_class": source["positive_margin_by_class"],
        "nearest_class_agreement": float((structural_class == labels).to(torch.float64).mean()),
        "nearest_class_agreement_by_class": _class_rates(structural_class == labels, labels, boolean=True),
        "topology": source["topology"],
        "bootstrap": {
            "semantic_support_rate": source["support_bootstrap"],
            "positive_margin_fraction": source["positive_margin_bootstrap"],
        },
    }


def _support_comparison(
    left_name: str,
    left: Mapping[str, Any],
    right_name: str,
    right: Mapping[str, Any],
    *,
    classes: Sequence[int] = tuple(range(CLASS_COUNT)),
) -> dict[str, Any]:
    result = _bootstrap_difference(
        _tensor(left["support"], f"{left_name} support").to(torch.float64),
        _tensor(left["labels"], f"{left_name} labels"),
        _tensor(right["support"], f"{right_name} support").to(torch.float64),
        _tensor(right["labels"], f"{right_name} labels"),
        classes=classes,
    )
    result["left"] = left_name
    result["right"] = right_name
    result["statistic"] = "macro structural_support_rate difference"
    return result


def _margin_comparison(
    left_name: str,
    left: Mapping[str, Any],
    right_name: str,
    right: Mapping[str, Any],
    *,
    classes: Sequence[int] = tuple(range(CLASS_COUNT)),
) -> dict[str, Any]:
    left_values = (_tensor(left["margin"], f"{left_name} margin") > 0.0).to(torch.float64)
    right_values = (_tensor(right["margin"], f"{right_name} margin") > 0.0).to(torch.float64)
    result = _bootstrap_difference(
        left_values,
        _tensor(left["labels"], f"{left_name} labels"),
        right_values,
        _tensor(right["labels"], f"{right_name} labels"),
        classes=classes,
    )
    result["left"] = left_name
    result["right"] = right_name
    result["statistic"] = "macro positive_margin_fraction difference"
    return result


def _build_comparisons(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    dev = sources["canonical_dev"]
    real = sources["real_ols_step6000"]
    s1 = sources["s1_step6000"]
    comparisons: dict[str, Any] = {
        "s1_minus_canonical_dev": _support_comparison("s1_step6000", s1, "canonical_dev", dev),
        "s1_minus_real_ols": _support_comparison("s1_step6000", s1, "real_ols_step6000", real),
        "real_ols_minus_canonical_dev": _support_comparison("real_ols_step6000", real, "canonical_dev", dev),
        "s1_positive_margin_minus_canonical_dev": _margin_comparison("s1_step6000", s1, "canonical_dev", dev),
        "s1_positive_margin_minus_real_ols": _margin_comparison("s1_step6000", s1, "real_ols_step6000", real),
        "real_ols_positive_margin_minus_canonical_dev": _margin_comparison("real_ols_step6000", real, "canonical_dev", dev),
        "s1_minus_canonical_dev_per_class": {},
        "s1_minus_real_ols_per_class": {},
        "real_ols_minus_canonical_dev_per_class": {},
    }
    for digit in range(CLASS_COUNT):
        one_class = (digit,)
        comparisons["s1_minus_canonical_dev_per_class"][str(digit)] = _support_comparison(
            "s1_step6000", s1, "canonical_dev", dev, classes=one_class
        )
        comparisons["s1_minus_real_ols_per_class"][str(digit)] = _support_comparison(
            "s1_step6000", s1, "real_ols_step6000", real, classes=one_class
        )
        comparisons["real_ols_minus_canonical_dev_per_class"][str(digit)] = _support_comparison(
            "real_ols_step6000", real, "canonical_dev", dev, classes=one_class
        )
    return comparisons


def _consensus_analysis(
    authorities: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    consensus: dict[str, torch.Tensor] = {}
    cross_tables: dict[str, Any] = {}
    for source_name in SOURCE_ORDER:
        value = _consensus_vector(authorities["predictions"], source_name, _tensor(sources[source_name]["labels"], f"{source_name} labels"))
        consensus[source_name] = value
        cross_tables[source_name] = _cross_table(sources[source_name], value)
    return consensus, cross_tables


def _validation_payload(
    authorities: Mapping[str, Any],
    split: Mapping[str, torch.Tensor],
    fit_signature: torch.Tensor,
    validation_signature: torch.Tensor,
    source_signatures: Mapping[str, torch.Tensor],
    q95: torch.Tensor,
    q95_validation_distance: torch.Tensor,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    fit_positions = _tensor(split["fit_positions"], "fit positions").to(torch.long)
    validation_positions = _tensor(split["validation_positions"], "validation positions").to(torch.long)
    train_count = authorities["train_labels"].numel()
    _require(not bool(torch.isin(fit_positions, validation_positions).any()), "fit and validation overlap")
    _require(fit_positions.numel() + validation_positions.numel() == train_count, "fit/validation coverage changed")
    _require(torch.equal(torch.sort(torch.cat([fit_positions, validation_positions])).values, torch.arange(train_count)), "fit/validation order coverage changed")
    _require(bool(torch.isfinite(q95).all()) and bool((q95 > 0.0).all()), "class q95 is invalid")
    _finite(q95_validation_distance, "validation-to-fit distances")
    for name, value in source_signatures.items():
        _finite(value, f"{name} structural signature")
    return {
        "preflight_status": preflight["status"],
        "descriptor_dimension": SIGNATURE_DIMENSION,
        "descriptor_dtype": "float64",
        "descriptor_finite": True,
        "fit_rows": int(fit_positions.numel()),
        "validation_rows": int(validation_positions.numel()),
        "fit_class_histogram": _class_histogram(split["fit_labels"]),
        "validation_class_histogram": _class_histogram(split["validation_labels"]),
        "fit_validation_disjoint": True,
        "fit_validation_complete": True,
        "canonical_dev_absent_from_fit": True,
        "canonical_dev_absent_from_q95": True,
        "s1_absent_from_fit": True,
        "s1_absent_from_q95": True,
        "q95_finite_positive": True,
        "q95_validation_distance_summary": _summary(q95_validation_distance),
        "source_signature_hashes": {name: tensor_sha256(value) for name, value in source_signatures.items()},
        "protected_test_sealed_zero": True,
        "new_generator_forward": False,
        "new_generator_backward": False,
        "parameter_updates": 0,
        "sampler_refit": False,
        "new_classifier": False,
        "new_evaluator": False,
    }


def _reference_payload(
    authorities: Mapping[str, Any],
    split: Mapping[str, torch.Tensor],
    fit_signature: torch.Tensor,
    validation_signature: torch.Tensor,
    fit_standardized: torch.Tensor,
    normalization_mean: torch.Tensor,
    normalization_scale: torch.Tensor,
    q95: torch.Tensor,
    q95_validation_distance: torch.Tensor,
    centroids: torch.Tensor,
) -> dict[str, Any]:
    fit_positions = _tensor(split["fit_positions"], "fit positions").to(torch.long).cpu()
    validation_positions = _tensor(split["validation_positions"], "validation positions").to(torch.long).cpu()
    return {
        "schema": REFERENCE_SCHEMA,
        "task_id": TASK_ID,
        "descriptor": {
            "dimension": SIGNATURE_DIMENSION,
            "representation": "337D grayscale geometry + threshold geometry/topology/spatial morphology",
            "thresholds": list(THRESHOLDS),
            "threshold_rule": "pixel >= threshold",
            "topology_rule": "8-connected foreground; 4-connected padded background; holes are non-external background components",
            "contour_features": False,
            "skeleton_features": False,
        },
        "normalization": {
            "fit_only": True,
            "mean": normalization_mean,
            "population_std_clamped_scale": normalization_scale,
            "epsilon": NORMALIZATION_EPS,
            "pca": False,
            "whitening": False,
            "feature_weights": False,
        },
        "support_distance": {
            "definition": "sqrt(mean squared standardized feature difference) to nearest same-class STRUCTURAL_FIT row",
            "q95_definition": "linear empirical q=0.95 of held-out STRUCTURAL_VALIDATION to same-class STRUCTURAL_FIT nearest distances",
            "q95_interpolation": "linear",
            "q95_by_class": q95,
        },
        "structural_fit": {
            "positions_in_train_ols_order": fit_positions,
            "canonical_source_indices": authorities["train_indices"].index_select(0, fit_positions),
            "labels": split["fit_labels"],
            "raw_signature": fit_signature,
            "standardized_signature": fit_standardized,
        },
        "structural_validation": {
            "positions_in_train_ols_order": validation_positions,
            "canonical_source_indices": authorities["train_indices"].index_select(0, validation_positions),
            "labels": split["validation_labels"],
            "raw_signature": validation_signature,
            "nearest_fit_distance": q95_validation_distance,
        },
        "class_centroids": centroids,
        "class_balance": {
            "fit": _class_histogram(split["fit_labels"]),
            "validation": _class_histogram(split["validation_labels"]),
        },
    }


def _evaluation_payload(
    authorities: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    consensus: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": TENSORS_SCHEMA,
        "task_id": TASK_ID,
        "source_order": list(SOURCE_ORDER),
        "evaluator_order": list(EVALUATOR_ORDER),
        "requested_labels": authorities["requested"],
        "dev_labels": authorities["dev_labels"],
        "group_masks": authorities["group_masks"],
        "sources": {},
        "consensus_at_least_3_of_4": dict(consensus),
    }
    for name in SOURCE_ORDER:
        source = sources[name]
        payload["sources"][name] = {
            "labels": source["labels"],
            "raw_signature": source["raw_signature"],
            "standardized_signature": source["standardized_signature"],
            "nearest_same_class_distance": source["nearest_same_class_distance"],
            "nearest_same_class_index": source["nearest_same_class_index"],
            "q95_rows": source["q95_rows"],
            "support": source["support"],
            "d_req": source["d_req"],
            "d_alt": source["d_alt"],
            "margin": source["margin"],
            "structural_class": source["structural_class"],
        }
    return payload


def _figure_source_data(
    sources: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    strata: Mapping[str, Any],
    cross_tables: Mapping[str, Any],
    classification: str,
) -> dict[str, Any]:
    return {
        "schema": "non_classifier_structural_semantic_confirmation_figure_source_data.v1",
        "source_order": list(SOURCE_ORDER),
        "class_count": CLASS_COUNT,
        "sources": {
            name: {
                "support_by_class": [float(sources[name]["support_by_class"][str(digit)]["rate"]) for digit in range(CLASS_COUNT)],
                "support_rate": float(sources[name]["support"].to(torch.float64).mean()),
                "distance": [float(value) for value in sources[name]["nearest_same_class_distance"].tolist()],
                "margin": [float(value) for value in sources[name]["margin"].tolist()],
            }
            for name in SOURCE_ORDER
        },
        "comparisons": comparisons,
        "strata": strata,
        "cross_tables": cross_tables,
        "classification": classification,
        "fixed_visual_identity": "accepted readiness fixed_selection_manifest; no new image selection",
    }


def _execution_manifest(root: Path, authorities: Mapping[str, Any], preflight: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "schema": "non_classifier_structural_semantic_confirmation_execution_manifest.v1",
        "task_id": TASK_ID,
        "authorization": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "input_gate": INPUT_GATE,
        "outer_gate": OUTER_GATE,
        "consumer": "frozen step6000 image views and accepted evaluator predictions only",
        "compute_device": str(device),
        "no_update": {
            "generator_loaded": False,
            "generator_forward": False,
            "generator_backward": False,
            "optimizer_created": False,
            "parameter_updates": 0,
            "sampler_refit": False,
            "new_classifier": False,
            "new_evaluator": False,
            "descriptor_search": False,
            "threshold_search": False,
            "contour_or_skeleton_search": False,
        },
        "structural_split": {
            "rule": "within accepted optical TRAIN class, accepted train_indices order, local_position % 10 == 0 -> validation",
            "fit_rows": preflight["fit_rows"] if "fit_rows" in preflight else None,
            "validation_rows": preflight["validation_rows"] if "validation_rows" in preflight else None,
        },
        "protected_test_sealed_zero": True,
        "accepted_authorities": authorities["accepted_hashes"],
        "fixed_selection_authority": _artifact(root, READINESS_SELECTION, "accepted fixed visualization identity"),
        "execution_source": _relative(root, Path(__file__)),
        "execution_source_sha256": file_sha256(Path(__file__)),
    }


def _run_experiment(root: Path, output_dir: Path) -> dict[str, Any]:
    target = (root / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target != outputs_root, "output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "protected output namespace is forbidden")
    _require(not target.exists(), f"write-once output already exists: {target}")

    authorities = _load_authorities(root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preflight = _build_preflight(authorities, device)
    split = _make_structural_split(authorities["train_labels"])
    preflight["fit_rows"] = int(split["fit_positions"].numel())
    preflight["validation_rows"] = int(split["validation_positions"].numel())
    preflight["fit_class_histogram"] = _class_histogram(split["fit_labels"])
    preflight["validation_class_histogram"] = _class_histogram(split["validation_labels"])

    fit_images = authorities["train_images"].index_select(0, split["fit_positions"])
    validation_images = authorities["train_images"].index_select(0, split["validation_positions"])
    fit_signature = structural_signature(fit_images, device=device)
    validation_signature = structural_signature(validation_images, device=device)
    source_images = {
        "canonical_dev": authorities["canonical_dev_views"],
        "real_ols_step6000": authorities["real_ols_views"],
        "s1_step6000": authorities["s1_views"],
    }
    source_labels = {
        "canonical_dev": authorities["dev_labels"],
        "real_ols_step6000": authorities["dev_labels"],
        "s1_step6000": authorities["requested"],
    }
    source_signatures = {name: structural_signature(images, device=device) for name, images in source_images.items()}
    fit_standardized, standardized_sources, normalization_mean, normalization_scale = _standardize(
        fit_signature,
        {"validation": validation_signature, **source_signatures},
    )
    validation_standardized = standardized_sources.pop("validation")
    validation_distance, _ = _class_distance(
        validation_standardized,
        fit_standardized,
        query_labels=split["validation_labels"],
        reference_labels=split["fit_labels"],
        device=device,
        need_index=False,
    )
    q95 = torch.stack(
        [
            torch.tensor(
                _q95(validation_distance[split["validation_labels"] == digit]),
                dtype=torch.float64,
            )
            for digit in range(CLASS_COUNT)
        ]
    )
    centroids = torch.stack(
        [fit_standardized[split["fit_labels"] == digit].mean(dim=0) for digit in range(CLASS_COUNT)],
        dim=0,
    ).to(torch.float64)
    sources: dict[str, dict[str, Any]] = {}
    for name in SOURCE_ORDER:
        source = _evaluate_source(
            name,
            source_signatures[name],
            standardized_sources[name],
            source_labels[name],
            fit_standardized,
            split["fit_labels"],
            centroids,
            q95,
            device,
        )
        source["standardized_signature"] = standardized_sources[name]
        source["support_bootstrap"] = _bootstrap_rate(source["support"].to(torch.float64), source["labels"])
        source["positive_margin_bootstrap"] = _bootstrap_rate((source["margin"] > 0.0).to(torch.float64), source["labels"])
        sources[name] = source

    comparisons = _build_comparisons(sources)
    strata = _stratum_reports(sources["s1_step6000"], authorities["requested"], authorities["group_masks"])
    consensus, cross_tables = _consensus_analysis(authorities, sources)
    diversity = _diversity_interaction(authorities["forward_metrics"])
    classification, mechanism, successor = _classify(sources, comparisons, strata, diversity)
    public_sources = {name: _source_public(sources[name]) for name in SOURCE_ORDER}
    validation = _validation_payload(
        authorities,
        split,
        fit_signature,
        validation_signature,
        source_signatures,
        q95,
        validation_distance,
        preflight,
    )

    target.mkdir(parents=True, exist_ok=False)
    _write_json(target / "preflight.json", preflight, kind="structural semantic confirmation preflight")
    _write_torch(
        target / "structural_reference.pt",
        _reference_payload(
            authorities,
            split,
            fit_signature,
            validation_signature,
            fit_standardized,
            normalization_mean,
            normalization_scale,
            q95,
            validation_distance,
            centroids,
        ),
        kind="TRAIN-only structural signature reference",
    )
    _write_torch(target / "evaluation_tensors.pt", _evaluation_payload(authorities, sources, consensus), kind="structural semantic confirmation tensors")
    _write_json(target / "source_manifest.json", {
        "schema": "non_classifier_structural_semantic_confirmation_source_manifest.v1",
        "task_id": TASK_ID,
        "authorization": AUTHORIZATION_TOKEN,
        "input_gate": INPUT_GATE,
        "outer_gate": OUTER_GATE,
        "execution_source": _relative(root, Path(__file__)),
        "execution_source_sha256": file_sha256(Path(__file__)),
        "protected_test_sealed_zero": True,
    }, kind="structural semantic confirmation source manifest")
    _write_json(target / "execution_manifest.json", _execution_manifest(root, authorities, {**preflight, **validation}, device), kind="structural semantic confirmation execution manifest")
    _write_json(target / "structural_metrics.json", {
        "schema": "non_classifier_structural_semantic_confirmation_metrics.v1",
        "descriptor": {"dimension": SIGNATURE_DIMENSION, "thresholds": list(THRESHOLDS)},
        "normalization": {"fit_only": True, "epsilon": NORMALIZATION_EPS, "pca": False, "whitening": False, "feature_weights": False},
        "split": {"fit_rows": validation["fit_rows"], "validation_rows": validation["validation_rows"], "fit_class_histogram": validation["fit_class_histogram"], "validation_class_histogram": validation["validation_class_histogram"]},
        "q95_by_class": [float(value) for value in q95.tolist()],
        "sources": public_sources,
        "comparisons": comparisons,
        "consensus_cross_tables": cross_tables,
        "strata": strata,
        "accepted_diversity_interaction": diversity,
    }, kind="structural semantic confirmation metrics")
    _write_json(target / "classifier_consensus_cross_table.json", {
        "schema": "non_classifier_structural_semantic_confirmation_consensus_cross_table.v1",
        "evaluator_order": list(EVALUATOR_ORDER),
        "source_label_semantics": SOURCE_LABEL_NAMES,
        "tables": cross_tables,
        "interpretation": "existing accepted evaluator predictions are descriptive only; no classifier is fitted and no classifier result defines the primary structural support gate",
    }, kind="classifier-independent consensus cross table")

    _figure_style()
    figure_artifacts = {
        "structural_support_by_class": _figure_support(sources, target),
        "structural_distance_distributions": _figure_distance(sources, target),
        "structural_margin_distributions": _figure_margin(sources, target),
        "structural_support_by_consensus_stratum": _figure_strata(strata, target),
        "topology_class_threshold_summary": _figure_topology(sources, target),
        "classifier_consensus_structural_support_cross_table": _figure_cross_table(cross_tables, target),
        "fixed_structural_examples": _figure_examples(authorities, target),
    }
    _write_json(target / "figure_source_data.json", _figure_source_data(sources, comparisons, strata, cross_tables, classification), kind="structural semantic confirmation figure source data")

    source_artifacts = {
        name: {
            "signature_sha256": tensor_sha256(source_signatures[name]),
            "rows": int(source_signatures[name].shape[0]),
            "label_source": SOURCE_LABEL_NAMES[name],
        }
        for name in SOURCE_ORDER
    }
    artifact_names = sorted(path.name for path in target.iterdir())
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "mode": "IMPLEMENT_PREPARED",
        "risk_level": "R2_ARTIFACT_PRODUCING_RESEARCH",
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "local_state": authorities["git"],
        "input_authorities": {
            "accepted_localization_report": _artifact(root, LOCALIZATION_REPORT, "accepted step6000 localization result"),
            "accepted_localization_status": _artifact(root, LOCALIZATION_STATUS, "accepted localization status"),
            "accepted_localization_payload": _artifact(root, LOCALIZATION_PAYLOAD, "accepted step6000 image views"),
            "accepted_localization_predictions": _artifact(root, LOCALIZATION_PREDICTIONS, "accepted frozen evaluator predictions"),
            "accepted_localization_strata": _artifact(root, LOCALIZATION_STRATA, "accepted G1-G4 strata"),
            "accepted_readiness_report": _artifact(root, READINESS_REPORT, "accepted step6000 readiness result"),
            "accepted_readiness_status": _artifact(root, READINESS_STATUS, "accepted readiness status"),
            "accepted_fixed_selection": _artifact(root, READINESS_SELECTION, "accepted fixed visual identity"),
            "canonical_train_source": _artifact(root, TRAINING_SOURCE, "canonical MNIST TRAIN/DEV source"),
            "accepted_train_ols": _artifact(root, TRAIN_OLS, "accepted real TRAIN complex OLS"),
            "accepted_dev_ols": _artifact(root, DEV_OLS, "accepted real DEV complex OLS"),
            "accepted_s1_bank": _artifact(root, S1_BANK, "accepted S1 candidate bank"),
            "accepted_s1_winner": _artifact(root, S1_WINNER, "accepted S1 frozen winner"),
        },
        "supported_bottleneck": "STEP6000_ON_MANIFOLD_CROSS_EVALUATOR_DECISION_DIVERGENCE_SUPPORTED",
        "sampler_target": "not applicable; structural image-space confirmation of accepted views",
        "structural_split": {
            "fit_role": "STRUCTURAL_FIT",
            "validation_role": "STRUCTURAL_VALIDATION",
            "rule": "within accepted optical TRAIN class, accepted train_indices order, local_position % 10 == 0 -> validation",
            "fit_rows": validation["fit_rows"],
            "validation_rows": validation["validation_rows"],
            "fit_class_histogram": validation["fit_class_histogram"],
            "validation_class_histogram": validation["validation_class_histogram"],
        },
        "descriptor_contract": {
            "dimension": SIGNATURE_DIMENSION,
            "representation": "original accepted CMOS image view, fixed 28x28 grayscale; internal structural signature only",
            "blocks": {"grayscale": GRAYSCALE_BLOCK_SIZE, "per_threshold_geometry": GEOMETRY_FEATURES_PER_THRESHOLD, "per_threshold_topology": TOPOLOGY_FEATURES_PER_THRESHOLD, "per_threshold_spatial": SPATIAL_FEATURES_PER_THRESHOLD, "thresholds": list(THRESHOLDS)},
            "threshold_rule": "pixel >= threshold",
            "topology_connectivity": {"foreground": 8, "background": 4, "external_background": "one-pixel padded border"},
            "contour": False,
            "skeleton": False,
        },
        "normalization_contract": {
            "fit_only": True,
            "population_std": True,
            "scale": "max(population_std, 1e-6)",
            "pca": False,
            "whitening": False,
            "feature_weights": False,
            "q95_by_class": [float(value) for value in q95.tolist()],
            "q95_validation_to_fit": _summary(validation_distance),
        },
        "source_artifacts": source_artifacts,
        "primary_structural_support": {
            "condition": "nearest same-class STRUCTURAL_FIT distance <= TRAIN-derived class q95",
            "macro_definition": "equal macro mean across requested/true classes",
            "sources": public_sources,
        },
        "secondary_metrics": {
            "centroid_margin_geometry": "d_alt - d_req",
            "nearest_class_agreement": {name: public_sources[name]["nearest_class_agreement"] for name in SOURCE_ORDER},
            "topology": {name: public_sources[name]["topology"] for name in SOURCE_ORDER},
            "accepted_diversity_interaction": diversity,
        },
        "bootstrap_comparisons": comparisons,
        "classifier_consensus_cross_analysis": {
            "evaluator_order": list(EVALUATOR_ORDER),
            "tables": cross_tables,
            "role": "descriptive accepted predictions only; not the primary structural semantic criterion",
        },
        "G1_G4_analysis": strata,
        "figures": figure_artifacts,
        "classification": classification,
        "mechanism": mechanism,
        "recommended_single_successor": successor,
        "claim_boundary": {
            "non_classifier_structural_confirmation_only": True,
            "no_classifier_fitted": True,
            "no_generator_forward": True,
            "no_generator_training": True,
            "no_sampler_refit": True,
            "no_new_bank": True,
            "no_final_perceptual_quality_claim": True,
            "no_hardware_readiness": True,
            "no_formal_promotion": True,
            "protected_test_sealed": False,
        },
        "files_changed": {"repository_source": [_relative(root, Path(__file__))], "scientific_artifacts": artifact_names},
        "validation": validation,
        "unresolved": [
            "Structural support is a frozen morphology/geometry confirmation and is not a substitute for human perceptual or hardware acceptance.",
            "Accepted E_S/E_I/E_C/E_D predictions remain descriptive cross-evaluator context and do not define the primary result.",
        ],
        "next_action": "NONE",
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report, kind="non-classifier structural semantic confirmation report")
    status = {
        "schema": STATUS_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "classification": classification,
        "recommended_single_successor": successor,
        "report_sha256": file_sha256(target / "report.json"),
        "no_update": True,
        "protected_test_sealed_zero": True,
        "next_action": "NONE",
        "output_dir": _relative(root, target),
    }
    _write_json(target / "status.json", status, kind="non-classifier structural semantic confirmation status")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-token", required=False)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "no_update": True, "output_dir": args.output_dir}, ensure_ascii=False, indent=2))
        return 0
    if args.authorization_token != AUTHORIZATION_TOKEN:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": "explicit authorization token is required"}, ensure_ascii=False, indent=2))
        return 2
    try:
        report = _run_experiment(_root(), Path(args.output_dir))
    except (StructuralConfirmationError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "recommended_single_successor": report["recommended_single_successor"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
