"""Materialize matched frozen DEV OLS and raw CMOS authorities.

This is a read-only representation consumer.  It never creates an optimizer,
calls backward, or evaluates a semantic probe.  ``--execute`` is required so
that importing or inspecting the module cannot publish a DEV artifact.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import LightGen2DChenBOGTComplexOLSTrainingConsumer, PRODUCTION_PROPAGATION_GRID, git_identity
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import (
    load_local_mnist,
    stratified_mnist_indices,
)


TASK_ID = "MATERIALIZE_FROZEN_DEV_OLS_CMOS_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_MATERIALIZE_FROZEN_DEV_OLS_CMOS_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "FROZEN_DEV_OLS_CMOS_INPUTS_MATERIALIZED"
CLASSIFICATION = "INPUT_AUTHORITY_MATERIALIZATION_ONLY"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

TRAINING_SOURCE = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
SPLIT_SEED = 1234
DEV_COUNT = 5_000
TRAIN_COUNT = 55_000
DEV_BATCH_SIZE = 10

PARENT_REPORT = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01/report.json")
PARENT_REPORT_SHA256 = "e2e87a9aece563616f9e64b329411a4bcd9a0a3f6dcbcab7f220e690bee15dfc"
PARENT_CHECKPOINT = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01/scientific_step_000300.pt")
PARENT_CHECKPOINT_SHA256 = "574440ac9344b1c8fb407c45debf3514fea2c9ab6526873c3286852077f3a0c4"
FINAL_TRAIN_OLS = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01/final_train_ols.pt")
FINAL_TRAIN_OLS_SHA256 = "2d1264219faa0964467d40f2e4882ec3767d3a1aafd30e143bf406aa39be6dcc"

CLEAN_REPORT = Path("outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/report.json")
CLEAN_REPORT_SHA256 = "dc05742fae2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904"
CLEAN_CHECKPOINT = Path("outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/scientific_step_000000.pt")
CLEAN_CHECKPOINT_SHA256 = "a61dd84728aa220aa567ab6cd5f9daa788b21e76be3c506054c2b7c0a23dcd34"

DEFAULT_OUTPUT_DIR = Path("outputs/20260918_frozen_dev_ols_cmos_v1/runs/attempt_01")
OLS_ARTIFACT_SCHEMA = "frozen_dev_complex_ols.v1"
CMOS_ARTIFACT_SCHEMA = "frozen_dev_raw_cmos.v1"
REPORT_SCHEMA = "frozen_dev_ols_cmos_materialization_report.v1"
SOURCE_MANIFEST_SCHEMA = "frozen_dev_ols_cmos_source_manifest.v1"
OLS_STAGE = "result.z_ols.coefficients: real_image_encoder_to_ols_projection_before_W_OLS"
CMOS_STAGE = "result.raw_camera.tensor: raw_cmos_intensity_before_evaluator_processing"


class MaterializationError(RuntimeError):
    """Raised when the frozen input-authority contract cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _read_json(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    _require(path.is_file(), f"JSON authority is missing: {path}")
    _require(file_sha256(path) == expected_sha256, f"JSON authority identity changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError(f"JSON authority is unreadable: {path}") from error
    return dict(_mapping(value, str(path)))


def _state_snapshot(runtime: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in runtime.state_dict().items()
    }


def _states_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _rng_snapshot(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()


def _prepare_cuda_runtime_dll_search_path() -> dict[str, Any]:
    cuda_build = torch.version.cuda
    _require(isinstance(cuda_build, str) and len(cuda_build.split(".")) == 2, "CUDA build cannot identify the NVRTC DLL")
    major, minor = cuda_build.split(".")
    environment_bin = (Path(sys.prefix) / "bin").resolve()
    dll_path = environment_bin / f"nvrtc-builtins64_{major}{minor}.dll"
    _require(environment_bin.is_dir() and dll_path.is_file(), f"required CUDA runtime DLL is unavailable: {dll_path}")
    current_path = os.environ.get("PATH", "")
    normalized_bin = os.path.normcase(os.path.normpath(str(environment_bin)))
    entries = [entry for entry in current_path.split(os.pathsep) if entry]
    present = any(os.path.normcase(os.path.normpath(entry)) == normalized_bin for entry in entries)
    if not present:
        os.environ["PATH"] = str(environment_bin) + (os.pathsep + current_path if current_path else "")
    return {
        "status": "already_present" if present else "prepended",
        "scope": "current_python_process_only",
        "cuda_build": cuda_build,
        "environment_bin": str(environment_bin),
        "expected_dll": dll_path.name,
        "expected_dll_exists": True,
        "path_changed": not present,
    }


def _validate_target(root: Path, output_dir: Path) -> Path:
    target = (root / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(not target.exists(), f"write-once output target already exists: {target}")
    _require(outputs_root in target.parents, "output target must remain below outputs/")
    _require("final_pt" not in target.parts, "final_pt namespace is forbidden")
    _require("_wsl_restore_metadata_v4" not in target.parts, "restored namespace is forbidden")
    _require(target != outputs_root and target != outputs_root / "runs", "output target is too broad")
    return target


def build_execution_plan(*, repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    """Return the explicit execution contract without reading data or models."""
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = _validate_target(root, DEFAULT_OUTPUT_DIR if output_dir is None else output_dir)
    return {
        "schema": "frozen_dev_ols_cmos_execution_plan.v1",
        "task_id": TASK_ID,
        "authorization_required": True,
        "outer_gate": OUTER_GATE,
        "output_dir": _relative(root, target),
        "device": "cuda:0",
        "batch_size": DEV_BATCH_SIZE,
        "trainable_parameters": [],
        "optimizer": False,
        "backward": False,
        "probe_fitting": False,
        "protected_test_sealed": False,
        "stages": {"ols": OLS_STAGE, "cmos": CMOS_STAGE},
    }


def _load_authorities(root: Path) -> dict[str, Any]:
    parent_report = _read_json(root / PARENT_REPORT, expected_sha256=PARENT_REPORT_SHA256)
    _require(parent_report.get("result") == "VALID_COMPLETED_GENERATION_BUNDLE_READY", "accepted Chen BOGT parent result changed")
    _require(parent_report.get("outer_gate") == "CHEN_BOGT_ON_COMPLEX_OLS_BOUNDED_TRAINING_V1_RESOLVED", "accepted Chen BOGT parent gate changed")

    clean_report = _read_json(root / CLEAN_REPORT, expected_sha256=CLEAN_REPORT_SHA256)
    _require(clean_report.get("result") == "VALID_COMPLETED", "clean 2L step-0 report is not valid completed")
    _require(clean_report.get("outer_gate") == "CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1_RESOLVED", "clean 2L step-0 gate changed")

    parent = safe_torch_load(root / PARENT_CHECKPOINT, expected_sha256=PARENT_CHECKPOINT_SHA256, map_location="cpu")
    clean = safe_torch_load(root / CLEAN_CHECKPOINT, expected_sha256=CLEAN_CHECKPOINT_SHA256, map_location="cpu")
    final = safe_torch_load(root / FINAL_TRAIN_OLS, expected_sha256=FINAL_TRAIN_OLS_SHA256, map_location="cpu")
    _require(isinstance(parent, Mapping), "parent checkpoint is malformed")
    _require(isinstance(clean, Mapping), "clean step-0 checkpoint is malformed")
    _require(isinstance(final, Mapping), "final TRAIN OLS artifact is malformed")
    _require(parent.get("completed_updates") == 300 and parent.get("scientific_artifact") is True and parent.get("engineering_only") is False, "parent checkpoint is not accepted scientific step 300")
    _require(clean.get("completed_updates") == 0 and clean.get("scientific_artifact") is True and clean.get("operational_artifact") is False, "clean step-0 checkpoint role changed")
    _require(clean.get("trainable_parameter_names") == ["generator.phase"], "clean step-0 trainable boundary changed")
    _require("encoder.phase" in clean.get("frozen_parameter_names", []), "clean step-0 encoder freeze is missing")

    parent_state = _mapping(parent.get("trainable_parameter_state"), "parent trainable state")
    clean_encoder = _tensor(clean.get("encoder_state"), "clean encoder state")
    clean_generator = _tensor(clean.get("generator_state"), "clean generator state")
    _require(set(parent_state) == {"encoder.phase", "generator.phase"}, "parent trainable state surface changed")
    _require(torch.equal(clean_encoder, _tensor(parent_state.get("encoder.phase"), "parent encoder state")), "clean encoder is not the accepted parent step-300 encoder")
    _require(torch.equal(clean_generator, _tensor(parent_state.get("generator.phase"), "parent generator state")), "clean generator is not the accepted parent step-300 generator")
    _require(clean_encoder.shape == (2, 1, 100, 100) and clean_encoder.dtype == torch.float32, "clean encoder shape/dtype changed")
    _require(clean_generator.shape == (2, 1, 100, 100) and clean_generator.dtype == torch.float32, "clean generator shape/dtype changed")
    _finite(clean_encoder, "clean encoder state")
    _finite(clean_generator, "clean generator state")

    _require(final.get("schema") == "chen_bogt_final_train_ols_collection.v1", "final TRAIN OLS schema changed")
    _require(final.get("selected_step") == 300, "final TRAIN OLS selected step changed")
    z_final = _tensor(final.get("z_final"), "final TRAIN OLS z_final")
    _require(z_final.shape == (TRAIN_COUNT, 100) and z_final.dtype == torch.complex64, "final TRAIN OLS shape/dtype changed")
    _finite(z_final.real, "final TRAIN OLS real part")
    _finite(z_final.imag, "final TRAIN OLS imaginary part")
    _require(final.get("canonical_order") == "increasing source row order", "final TRAIN OLS order contract changed")

    runtime_authority = _mapping(parent.get("runtime"), "parent runtime authority")
    _require(runtime_authority.get("ols_interface") == "lightgen_10x10_complex_ols_v1", "parent OLS interface changed")
    _require(runtime_authority.get("transport") == "T=I_100 coefficient space", "parent OLS transport changed")
    _require(runtime_authority.get("writer") == "W_OLS(z)=sum_j (Tz)_j v_j", "parent OLS writer changed")
    _require(runtime_authority.get("config", {}).get("propagation_grid_size") == PRODUCTION_PROPAGATION_GRID, "parent propagation grid changed")

    parent_identity = _mapping(parent_report.get("data_identity"), "parent data identity")
    final_identity = _mapping(final.get("data_identity"), "final TRAIN OLS data identity")
    _require(dict(final_identity) == dict(parent_identity), "final TRAIN OLS data identity differs from parent report")

    return {
        "parent_report": parent_report,
        "parent": dict(parent),
        "clean_report": clean_report,
        "clean": dict(clean),
        "final": dict(final),
        "data_identity": dict(parent_identity),
        "authority_refs": {
            "parent_report": _artifact_ref(root, root / PARENT_REPORT, role="accepted Chen BOGT report"),
            "parent_checkpoint_step_300": _artifact_ref(root, root / PARENT_CHECKPOINT, role="accepted clean 2L parent state"),
            "clean_2l_step_0": _artifact_ref(root, root / CLEAN_CHECKPOINT, role="accepted clean 2L step-0 state consumed"),
            "clean_2l_report": _artifact_ref(root, root / CLEAN_REPORT, role="clean 2L starting-authority report"),
            "final_train_ols": _artifact_ref(root, root / FINAL_TRAIN_OLS, role="accepted TRAIN OLS stage authority"),
        },
    }


def _load_dev_data(root: Path, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    source = root / TRAINING_SOURCE
    _require(source.is_file(), f"registered MNIST source is missing: {source}")
    _require(file_sha256(source) == TRAINING_SOURCE_SHA256, "registered MNIST source identity changed")
    images, labels = load_local_mnist(root / "data", train=True)
    _require(images.shape == (60_000, 1, 28, 28) and images.dtype == torch.float32, "MNIST image source shape/dtype changed")
    _require(labels.shape == (60_000,) and labels.dtype == torch.long, "MNIST label source shape/dtype changed")
    _require(bool(torch.isfinite(images).all().item()) and bool(((images >= 0.0) & (images <= 1.0)).all().item()), "MNIST images are not finite in [0,1]")
    _require(torch.equal(torch.unique(labels), torch.arange(10, dtype=torch.long)), "MNIST label source classes changed")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=SPLIT_SEED, validation_per_label=500)
    train_indices = torch.sort(train_indices).values.contiguous()
    development_indices = torch.sort(development_indices).values.contiguous()
    _require(train_indices.numel() == TRAIN_COUNT and development_indices.numel() == DEV_COUNT, "registered split counts changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN and DEV roles overlap")
    dev_labels = labels.index_select(0, development_indices).contiguous()
    dev_images = images.index_select(0, development_indices).contiguous()
    train_counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
    dev_counts = torch.bincount(dev_labels, minlength=10)
    _require(torch.equal(dev_counts, torch.full((10,), 500, dtype=torch.long)), "DEV role is not exactly 500 rows per class")
    identity = {
        "source_path": TRAINING_SOURCE.as_posix(),
        "source_sha256": TRAINING_SOURCE_SHA256,
        "loader": "ldm_project.v2.student.evaluator.load_local_mnist",
        "split": "stratified_mnist_indices(seed=1234, validation_per_label=500), then source-row sort",
        "seed": SPLIT_SEED,
        "train_count": TRAIN_COUNT,
        "development_count": DEV_COUNT,
        "train_counts_by_class": [int(value) for value in train_counts.tolist()],
        "development_counts_by_class": [int(value) for value in dev_counts.tolist()],
        "labels_tensor_sha256": tensor_sha256(labels),
        "train_indices_sha256": tensor_sha256(train_indices),
        "development_indices_sha256": tensor_sha256(development_indices),
        "train_order": "increasing canonical source row order for statistics and collection",
        "development_use": "diagnostics only",
        "protected_test_read": False,
        "sealed_read": False,
    }
    _require(identity == dict(expected_identity), "reconstructed DEV split identity differs from accepted authority")
    final = _mapping(expected_identity, "expected data identity")
    _require(final.get("development_indices_sha256") == "9489a7f1b054a7f88163710030c43f1582e8e14f2c7013504ed3be10300f2c9a", "DEV ordering hash is not the accepted split")
    del images
    gc.collect()
    return {
        "images": dev_images,
        "labels": dev_labels,
        "development_indices": development_indices,
        "train_indices": train_indices,
        "identity": identity,
    }


def _load_frozen_model(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[LightGen2DChenBOGTComplexOLSTrainingConsumer, dict[str, Any]]:
    clean = _mapping(authorities.get("clean"), "clean model authority")
    parent = _mapping(authorities.get("parent"), "parent model authority")
    consumer = LightGen2DChenBOGTComplexOLSTrainingConsumer(device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    parent_runtime = _mapping(parent.get("runtime"), "parent runtime")
    runtime_identity = consumer.runtime_identity()
    _require(runtime_identity == dict(parent_runtime), "current runtime differs from accepted clean OLS/GPU runtime authority")
    consumer.load_trainable_state_dict({
        "encoder.phase": _tensor(clean.get("encoder_state"), "clean encoder state"),
        "generator.phase": _tensor(clean.get("generator_state"), "clean generator state"),
    })
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
    consumer.runtime.eval()
    _require(all(not parameter.requires_grad for parameter in consumer.runtime.parameters()), "frozen model parameter firewall is not closed")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "frozen model has a stale gradient")
    return consumer, {
        "runtime_identity": runtime_identity,
        "loaded_state": {
            "encoder_phase_sha256": tensor_sha256(_tensor(clean.get("encoder_state"), "clean encoder state")),
            "generator_phase_sha256": tensor_sha256(_tensor(clean.get("generator_state"), "clean generator state")),
        },
        "trainable_parameters": [],
        "requires_grad_all_false": True,
        "optimizer_created": False,
        "backward_called": False,
    }


def _validate_stage(result: Any) -> None:
    _require(result.image_input is not None and result.ols_input_field is not None, "real image forward did not expose the OLS stage")
    _require(result.z_ols.source == "real_encoded", "captured OLS is not the real encoded state")
    _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "OLS projection boundary marker is missing")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "forward did not use the accepted W_OLS writer")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "forward did not use T=I_100")
    _require(result.raw_camera.plane.plane_id == "lightgen_raw_cmos", "raw CMOS plane identity changed")


def _forward_batch(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = consumer.runtime.forward_from_image(batch.to(consumer.device))
    _validate_stage(result)
    z = result.z_ols.coefficients
    raw = result.raw_intensity
    _require(z.shape == (batch.shape[0], 100) and z.dtype == torch.complex64, "DEV OLS batch shape/dtype changed")
    _require(raw.shape == (batch.shape[0], 1, 200, 200) and raw.dtype == torch.float32, "DEV raw CMOS batch shape/dtype changed")
    _finite(z.real, "DEV OLS batch real part")
    _finite(z.imag, "DEV OLS batch imaginary part")
    _finite(raw, "DEV raw CMOS batch")
    return z.detach().cpu(), raw.detach().cpu()


def _validate_materialized_tensors(
    z: torch.Tensor,
    raw: torch.Tensor,
    development_indices: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    _require(z.shape == (DEV_COUNT, 100) and z.dtype == torch.complex64, "DEV OLS materialization shape/dtype is invalid")
    _require(raw.shape == (DEV_COUNT, 1, 200, 200) and raw.dtype == torch.float32, "DEV CMOS materialization shape/dtype is invalid")
    _finite(z.real, "DEV OLS real part")
    _finite(z.imag, "DEV OLS imaginary part")
    _finite(raw, "DEV raw CMOS")
    _require(bool((raw >= 0.0).all().item()), "DEV raw CMOS contains negative intensity")
    flat = raw.reshape(DEV_COUNT, -1)
    row_max = flat.amax(dim=1)
    row_min = flat.amin(dim=1)
    zero_image_count = int((row_max == 0.0).sum().item())
    constant_image_count = int((row_max == row_min).sum().item())
    _require(zero_image_count == 0, "DEV raw CMOS contains zero images")
    _require(constant_image_count == 0, "DEV raw CMOS contains constant images")
    _require(development_indices.shape == (DEV_COUNT,) and labels.shape == (DEV_COUNT,), "DEV identity vectors have invalid shape")
    return {
        "ols": {
            "shape": list(z.shape),
            "dtype": str(z.dtype),
            "real_finite": True,
            "imag_finite": True,
            "tensor_sha256": tensor_sha256(z),
        },
        "cmos": {
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "finite_fraction": float(torch.isfinite(raw).to(torch.float64).mean().item()),
            "nonconstant": True,
            "zero_image_count": zero_image_count,
            "constant_image_count": constant_image_count,
            "min": float(raw.min().item()),
            "max": float(raw.max().item()),
            "tensor_sha256": tensor_sha256(raw),
        },
        "identity": {
            "row_count": DEV_COUNT,
            "indices_sha256": tensor_sha256(development_indices),
            "labels_sha256": tensor_sha256(labels),
            "indices_strictly_increasing": bool((development_indices[1:] > development_indices[:-1]).all().item()),
            "labels_per_class": [int(value) for value in torch.bincount(labels, minlength=10).tolist()],
        },
    }


def _materialize(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    dev_images: torch.Tensor,
    dev_indices: torch.Tensor,
    dev_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    _require(dev_images.shape[0] == DEV_COUNT, "DEV image count changed before forward")
    before_state = _state_snapshot(consumer.runtime)
    before_rng = _rng_snapshot(consumer.device)
    with torch.inference_mode():
        smoke_z, smoke_raw = _forward_batch(consumer, dev_images[:DEV_BATCH_SIZE])
        _require(smoke_z.shape == (DEV_BATCH_SIZE, 100) and smoke_raw.shape[0] == DEV_BATCH_SIZE, "bounded DEV smoke forward failed")
        z = torch.empty((DEV_COUNT, 100), dtype=torch.complex64)
        raw = torch.empty((DEV_COUNT, 1, 200, 200), dtype=torch.float32)
        batch_count = 0
        for start in range(0, DEV_COUNT, DEV_BATCH_SIZE):
            stop = min(start + DEV_BATCH_SIZE, DEV_COUNT)
            batch_z, batch_raw = _forward_batch(consumer, dev_images[start:stop])
            z[start:stop].copy_(batch_z)
            raw[start:stop].copy_(batch_raw)
            batch_count += 1
    torch.cuda.synchronize(consumer.device)
    after_state = _state_snapshot(consumer.runtime)
    after_rng = _rng_snapshot(consumer.device)
    _require(_states_equal(before_state, after_state), "frozen model state changed during DEV forward")
    _require(torch.equal(before_rng[0], after_rng[0]) and torch.equal(before_rng[1], after_rng[1]), "DEV forward consumed a random-number state")
    _require(all(not parameter.requires_grad for parameter in consumer.runtime.parameters()), "parameter freeze was not preserved")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "DEV forward produced a parameter gradient")
    diagnostics = _validate_materialized_tensors(z, raw, dev_indices, dev_labels)
    diagnostics["forward"] = {
        "device": str(consumer.device),
        "batch_size": DEV_BATCH_SIZE,
        "batch_count": batch_count,
        "smoke_rows": DEV_BATCH_SIZE,
        "materialized_rows": DEV_COUNT,
        "row_order": "contiguous slices from source-row-sorted DEV input; concatenated without reorder",
        "labels_used_for_optical_computation": False,
        "optimizer_created": False,
        "backward_called": False,
        "cuda_synchronized": True,
        "runtime_state_unchanged": True,
        "rng_state_unchanged": True,
    }
    return z, raw, diagnostics


def _payload_identity(data: Mapping[str, Any], indices: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return {
        "data_identity": dict(data),
        "source_indices": indices.clone(),
        "labels": labels.clone(),
        "canonical_order": "increasing source row order",
    }


def _load_and_validate_saved(path: Path, *, expected_sha256: str, schema: str, tensor_key: str, expected_tensor_sha256: str, indices: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    payload = safe_torch_load(path, expected_sha256=expected_sha256, map_location="cpu")
    mapping = _mapping(payload, str(path))
    _require(mapping.get("schema") == schema, f"saved artifact schema changed: {path.name}")
    value = _tensor(mapping.get(tensor_key), f"saved {tensor_key}")
    _require(tensor_sha256(value) == expected_tensor_sha256, f"saved tensor differs from materialized tensor: {path.name}")
    _require(torch.equal(_tensor(mapping.get("source_indices"), "saved source indices"), indices), f"saved source indices changed: {path.name}")
    _require(torch.equal(_tensor(mapping.get("labels"), "saved labels"), labels), f"saved labels changed: {path.name}")
    return {"path": path, "sha256": expected_sha256, "tensor_sha256": tensor_sha256(value)}


def _source_manifest(root: Path, start_git: Mapping[str, Any], authorities: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    relevant = (
        Path(__file__).resolve(),
        root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
        root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
        root / "src/ldm_project/v2/student/evaluator.py",
        root / "scripts/research/lightgen/execute_lightgen_2d_chen_bogt_complex_ols_training_v1.py",
    )
    paths = [_relative(root, path) for path in relevant]
    hashes = {path: file_sha256(root / path) for path in paths}
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "authorization_token": AUTHORIZATION_TOKEN,
        "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
        "branch": start_git.get("branch"),
        "head": start_git.get("head"),
        "dirty_before": bool(start_git.get("worktree_dirty")),
        "status_sha256_before": start_git.get("status_sha256"),
        "relevant_source_paths": paths,
        "relevant_source_sha256": hashes,
        "input_authority_sha256": {name: ref["sha256"] for name, ref in _mapping(authorities.get("authority_refs"), "authority refs").items()},
        "output_dir": _relative(root, output_dir),
        "scope": {
            "frozen_dev_rows": DEV_COUNT,
            "train_forward_rows": 0,
            "protected_test_sealed_reads": 0,
            "optimizer": False,
            "backward": False,
            "probe": False,
        },
    }


def execute(*, repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = _validate_target(root, DEFAULT_OUTPUT_DIR if output_dir is None else output_dir)
    start_git = git_identity(root)
    _require(start_git.get("branch") == EXPECTED_BRANCH, "current branch differs from accepted authority")
    _require(start_git.get("head") == EXPECTED_HEAD, "current HEAD differs from accepted authority")

    authorities = _load_authorities(root)
    data = _load_dev_data(root, authorities["data_identity"])
    final = _mapping(authorities.get("final"), "final TRAIN OLS")
    _require(torch.equal(_tensor(final.get("source_indices"), "final TRAIN OLS source indices"), data["train_indices"]), "final TRAIN OLS source ordering differs from reconstructed split")
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "frozen DEV materialization requires CUDA")
    cuda_runtime = _prepare_cuda_runtime_dll_search_path()
    consumer, model_authority = _load_frozen_model(root, authorities, torch.device("cuda:0"))

    z, raw, diagnostics = _materialize(consumer, data["images"], data["development_indices"], data["labels"])
    del data["images"]
    gc.collect()

    ols_path = target / "dev_complex_ols.pt"
    cmos_path = target / "dev_raw_cmos.pt"
    atomic_torch_save_new(
        ols_path,
        {
            "schema": OLS_ARTIFACT_SCHEMA,
            "representation": "real_DEV_complex_OLS",
            "stage": OLS_STAGE,
            "z_dev": z,
            **_payload_identity(authorities["data_identity"], data["development_indices"], data["labels"]),
        },
        artifact_kind="frozen DEV complex OLS",
    )
    ols_sha = file_sha256(ols_path)
    del z
    gc.collect()
    atomic_torch_save_new(
        cmos_path,
        {
            "schema": CMOS_ARTIFACT_SCHEMA,
            "representation": "real_DEV_raw_CMOS",
            "stage": CMOS_STAGE,
            "raw_cmos_intensity": raw,
            **_payload_identity(authorities["data_identity"], data["development_indices"], data["labels"]),
        },
        artifact_kind="frozen DEV raw CMOS",
    )
    cmos_sha = file_sha256(cmos_path)
    del raw
    gc.collect()

    saved_ols = _load_and_validate_saved(ols_path, expected_sha256=ols_sha, schema=OLS_ARTIFACT_SCHEMA, tensor_key="z_dev", expected_tensor_sha256=diagnostics["ols"]["tensor_sha256"], indices=data["development_indices"], labels=data["labels"])
    del saved_ols
    gc.collect()
    saved_cmos_payload = safe_torch_load(cmos_path, expected_sha256=cmos_sha, map_location="cpu")
    saved_cmos = _mapping(saved_cmos_payload, "saved DEV raw CMOS")
    saved_raw = _tensor(saved_cmos.get("raw_cmos_intensity"), "saved DEV raw CMOS tensor")
    _require(saved_raw.shape == (DEV_COUNT, 1, 200, 200) and saved_raw.dtype == torch.float32, "saved DEV raw CMOS shape/dtype changed")
    _finite(saved_raw, "saved DEV raw CMOS")
    _require(tensor_sha256(saved_raw) == diagnostics["cmos"]["tensor_sha256"], "saved DEV raw CMOS differs from materialized tensor")
    _require(torch.equal(_tensor(saved_cmos.get("source_indices"), "saved CMOS source indices"), data["development_indices"]), "saved CMOS source indices changed")
    _require(torch.equal(_tensor(saved_cmos.get("labels"), "saved CMOS labels"), data["labels"]), "saved CMOS labels changed")
    del saved_raw, saved_cmos_payload
    gc.collect()

    source_manifest = _source_manifest(root, start_git, authorities, target)
    source_manifest_path = target / "source_manifest.json"
    atomic_write_new_json(source_manifest_path, source_manifest, artifact_kind="frozen DEV source manifest")
    source_manifest_sha = file_sha256(source_manifest_path)
    end_git = git_identity(root)
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "result": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "classification": CLASSIFICATION,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "objective": "Create exactly matched frozen real-DEV OLS and raw CMOS input authorities for the subsequent semantic-localization consumer.",
        "local_state": {"branch": start_git["branch"], "head": start_git["head"], "dirty_before": bool(start_git["worktree_dirty"]), "git_end": end_git},
        "input_authorities": authorities["authority_refs"],
        "clean_model_authority": {
            "checkpoint": authorities["authority_refs"]["clean_2l_step_0"],
            "role": "accepted clean 2L conflict-aware scientific step-0 state",
            "successor_step_300_consumed": False,
            "encoder_state_sha256": model_authority["loaded_state"]["encoder_phase_sha256"],
            "generator_state_sha256": model_authority["loaded_state"]["generator_phase_sha256"],
            "runtime": model_authority["runtime_identity"],
        },
        "dev_split_identity": {"rows": DEV_COUNT, "labels": {"tensor_sha256": tensor_sha256(data["labels"]), "per_class": [int(value) for value in torch.bincount(data["labels"], minlength=10).tolist()]}, "ordering": {"indices_tensor_sha256": tensor_sha256(data["development_indices"]), "canonical_order": "increasing source row order", "source_identity": authorities["data_identity"]}},
        "forward_contract": {"path": "DEV image -> frozen Encoder -> result.z_ols.coefficients -> same W_OLS -> frozen clean 2L Generator -> raw CMOS intensity", "ols_stage": OLS_STAGE, "cmos_stage": CMOS_STAGE, "batch_size": DEV_BATCH_SIZE, "propagation_grid_size": PRODUCTION_PROPAGATION_GRID, "labels_in_optical_computation": False},
        "dev_ols_artifact": {"path": _relative(root, ols_path), "sha256": ols_sha, "shape": [DEV_COUNT, 100], "dtype": "torch.complex64", "finite": True, "semantic_stage": OLS_STAGE, "tensor_sha256": diagnostics["ols"]["tensor_sha256"]},
        "dev_cmos_artifact": {"path": _relative(root, cmos_path), "sha256": cmos_sha, **diagnostics["cmos"], "semantic_stage": CMOS_STAGE},
        "row_label_alignment": {"proven": True, "row_count": DEV_COUNT, "ols_source_indices_equal": True, "cmos_source_indices_equal": True, "ols_labels_equal": True, "cmos_labels_equal": True, "same_order": True, "identity": diagnostics["identity"]},
        "parameter_freeze_check": model_authority | {"runtime_state_unchanged": diagnostics["forward"]["runtime_state_unchanged"], "rng_state_unchanged": diagnostics["forward"]["rng_state_unchanged"]},
        "data_role_check": {"train_forward_rows": 0, "dev_materialized_rows": DEV_COUNT, "dev_smoke_rows": DEV_BATCH_SIZE, "protected_reads": 0, "test_reads": 0, "sealed_reads": 0, "labels_used_only_for_identity": True},
        "validation": diagnostics | {"source_manifest_sha256": source_manifest_sha, "cuda_runtime": cuda_runtime},
        "source_manifest": {"path": _relative(root, source_manifest_path), "sha256": source_manifest_sha},
        "output_dir": _relative(root, target),
        "files_changed": {"repository_source": [_relative(root, Path(__file__).resolve())], "artifacts": [_relative(root, ols_path), _relative(root, cmos_path), _relative(root, source_manifest_path), _relative(root, target / "report.json")]},
        "unresolved": [],
        "next_action": "RESUME_LOCALIZE_OLS_TO_CMOS_SEMANTIC_INFORMATION_V1",
        "claim_boundary": "Only matched frozen DEV OLS and raw CMOS representations were materialized and validated; no semantic bottleneck classification was performed.",
    }
    atomic_write_new_json(target / "report.json", report, artifact_kind="frozen DEV materialization report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and materialize the frozen DEV artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(build_execution_plan(output_dir=args.output_dir), indent=2, sort_keys=True))
        return
    report = execute(output_dir=args.output_dir)
    print(json.dumps({"result": report["result"], "outer_gate": report["outer_gate"], "output_dir": report["output_dir"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
