"""Localize class information across the frozen OLS-to-CMOS chain.

The consumer materializes one clean TRAIN raw-CMOS authority when no named
authority exists, fits one fixed TRAIN-only multinomial logistic probe on the
accepted complex OLS and raw-CMOS representations, and measures both probes
once on the accepted matched DEV identity.  It never updates optical state.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.lightgen import materialize_lightgen_2d_frozen_dev_ols_cmos_v1 as dev_materializer
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import LightGen2DChenBOGTComplexOLSTrainingConsumer, git_identity
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import (
    SmoothMNISTEvaluator,
    load_local_mnist,
    stratified_mnist_indices,
)


TASK_ID = "RESUME_LOCALIZE_OLS_TO_CMOS_SEMANTIC_INFORMATION_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_RESUME_LOCALIZE_OLS_TO_CMOS_SEMANTIC_INFORMATION_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "OLS_TO_CMOS_SEMANTIC_INFORMATION_LOCALIZED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

TRAINING_SOURCE = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
SPLIT_SEED = 1234
TRAIN_COUNT = 55_000
DEV_COUNT = 5_000
CLASS_COUNT = 10

DEV_RUN = Path("outputs/20260918_frozen_dev_ols_cmos_v1/runs/attempt_01")
DEV_REPORT = DEV_RUN / "report.json"
DEV_REPORT_SHA256 = "7043a060f0833ef7dde048fff09928fb987de4b0956d4deab474e0a980c0e126"
DEV_OLS = DEV_RUN / "dev_complex_ols.pt"
DEV_OLS_SHA256 = "5a8efe07c6ba52168f415e56fccdd68c2552b6dd5f4f975dabc96d2215fbb622"
DEV_OLS_TENSOR_SHA256 = "49150b0f5185171f676923b661bc4f09e3a235b89d339937b9a4cd9aed936c82"
DEV_CMOS = DEV_RUN / "dev_raw_cmos.pt"
DEV_CMOS_SHA256 = "a018840700e22f5b1ef926b842041db0dba0d4f342abb5d0df2110f9c619d4c2"
DEV_CMOS_TENSOR_SHA256 = "14cdc2b393776416b747483b36494c0dd7866982a9f08b7d115aa03437ad3fd5"

SMOOTH_CHECKPOINT = Path(
    "outputs/20260730_v2/runs/20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/smooth_mnist_evaluator_v21.pt"
)
SMOOTH_CHECKPOINT_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01"
)
TRAIN_CMOS_SCHEMA = "frozen_train_raw_cmos.v1"
PROBE_SCHEMA = "ols_to_cmos_semantic_localization_probes.v1"
REPORT_SCHEMA = "ols_to_cmos_semantic_information_localization_report.v1"
SOURCE_MANIFEST_SCHEMA = "ols_to_cmos_semantic_information_localization_source_manifest.v1"
TRAIN_CMOS_STAGE = "result.raw_camera.tensor: raw_cmos_intensity_before_evaluator_processing"
TRAIN_FORWARD_BATCH_SIZE = 10
SMOOTH_BATCH_SIZE = 256
CMOS_POOL_SHAPE = (28, 28)
PROBE_C = 1.0
PROBE_MAX_ITER = 75
PROBE_HISTORY_SIZE = 10
PROBE_TOLERANCE_GRAD = 1.0e-9
PROBE_TOLERANCE_CHANGE = 1.0e-12
NEGATIVE_CONTROL_SEED = 20260919


class LocalizationError(RuntimeError):
    """Raised when a localization authority or boundary cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalizationError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _tensor_sha256_streaming(value: torch.Tensor, *, rows_per_chunk: int = 32) -> str:
    """Hash a large CPU tensor without materializing its full byte buffer."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    if tensor.ndim == 0:
        digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()
    for start in range(0, tensor.shape[0], rows_per_chunk):
        digest.update(tensor[start : start + rows_per_chunk].numpy().tobytes())
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    _require(path.is_file(), f"JSON authority is missing: {path}")
    if expected_sha256 is not None:
        _require(file_sha256(path) == expected_sha256, f"JSON authority identity changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LocalizationError(f"JSON authority is unreadable: {path}") from error
    return dict(_mapping(value, str(path)))


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _validate_target(root: Path, output_dir: Path) -> Path:
    target = (root / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    _validate_target_location(root, target)
    _require(not target.exists(), f"write-once output target already exists: {target}")
    return target


def _validate_target_location(root: Path, target: Path) -> None:
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents, "output target must remain below outputs/")
    _require("final_pt" not in target.parts, "final_pt namespace is forbidden")
    _require("_wsl_restore_metadata_v4" not in target.parts, "restored namespace is forbidden")
    _require(target not in {outputs_root, outputs_root / "runs"}, "output target is too broad")


def _validate_partial_target(root: Path, output_dir: Path) -> Path:
    target = (root / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    _validate_target_location(root, target)
    _require(target.is_dir(), f"partial output target is not a directory: {target}")
    _require((target / "source_manifest.json").is_file(), "partial output is missing its source manifest")
    _require((target / "train_raw_cmos.pt").is_file(), "partial output is missing its TRAIN CMOS authority")
    for name in ("report.json", "probe_weights.pt", "probe_metrics.json"):
        _require(not (target / name).exists(), f"partial output already contains final artifact: {name}")
    return target


def build_execution_plan(*, repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    """Return the frozen execution contract without reading scientific inputs."""
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = _validate_target(root, DEFAULT_OUTPUT_DIR if output_dir is None else output_dir)
    return {
        "schema": "ols_to_cmos_semantic_information_localization_execution_plan.v1",
        "task_id": TASK_ID,
        "authorization_required": True,
        "authorization_token": AUTHORIZATION_TOKEN,
        "outer_gate": OUTER_GATE,
        "output_dir": _relative(root, target),
        "train_cmos_resolution": "consume_named_valid_authority_else_materialize_exactly_one_clean_frozen_train_artifact",
        "train_forward_rows": TRAIN_COUNT,
        "protected_test_sealed_reads": 0,
        "optical_updates": False,
        "optimizer_or_backward_through_optics": False,
        "probe": {
            "family": "multinomial_logistic_regression",
            "regularization": "L2_on_weights_only",
            "C": PROBE_C,
            "solver": "torch_LBFGS_strong_wolfe",
            "max_iter": PROBE_MAX_ITER,
            "history_size": PROBE_HISTORY_SIZE,
            "dev_tuning": False,
            "family_sweep": False,
        },
        "representations": {
            "ols": "concat(real(z), imag(z)) in R^200; TRAIN-only standardization",
            "cmos": "clamp[0,1] -> adaptive_avg_pool2d(28,28) -> flatten; TRAIN-only standardization",
        },
    }


def _load_authorities(root: Path) -> dict[str, Any]:
    """Load and validate all named inputs without automatic-latest resolution."""
    start_git = git_identity(root)
    _require(start_git.get("branch") == EXPECTED_BRANCH, "current branch differs from accepted authority")
    _require(start_git.get("head") == EXPECTED_HEAD, "current HEAD differs from accepted authority")

    base = dev_materializer._load_authorities(root)
    dev_report = _read_json(root / DEV_REPORT, expected_sha256=DEV_REPORT_SHA256)
    _require(dev_report.get("result") == "VALID_COMPLETED", "accepted DEV materialization is not VALID_COMPLETED")
    _require(dev_report.get("outer_gate") == "FROZEN_DEV_OLS_CMOS_INPUTS_MATERIALIZED", "accepted DEV materialization gate changed")
    _require(dev_report.get("row_label_alignment", {}).get("proven") is True, "accepted DEV row-label alignment is not proven")

    ols_payload = safe_torch_load(root / DEV_OLS, expected_sha256=DEV_OLS_SHA256, map_location="cpu")
    cmos_payload = safe_torch_load(root / DEV_CMOS, expected_sha256=DEV_CMOS_SHA256, map_location="cpu")
    ols_map = _mapping(ols_payload, "accepted DEV OLS payload")
    cmos_map = _mapping(cmos_payload, "accepted DEV CMOS payload")
    _require(ols_map.get("schema") == "frozen_dev_complex_ols.v1", "accepted DEV OLS schema changed")
    _require(cmos_map.get("schema") == "frozen_dev_raw_cmos.v1", "accepted DEV CMOS schema changed")
    dev_z = _tensor(ols_map.get("z_dev"), "accepted DEV OLS tensor")
    dev_raw = _tensor(cmos_map.get("raw_cmos_intensity"), "accepted DEV raw CMOS tensor")
    dev_indices = _tensor(ols_map.get("source_indices"), "accepted DEV OLS indices")
    dev_labels = _tensor(ols_map.get("labels"), "accepted DEV OLS labels")
    _require(dev_z.shape == (DEV_COUNT, 100) and dev_z.dtype == torch.complex64, "accepted DEV OLS shape/dtype changed")
    _require(dev_raw.shape == (DEV_COUNT, 1, 200, 200) and dev_raw.dtype == torch.float32, "accepted DEV CMOS shape/dtype changed")
    _finite(dev_z.real, "accepted DEV OLS real part")
    _finite(dev_z.imag, "accepted DEV OLS imaginary part")
    _finite(dev_raw, "accepted DEV raw CMOS")
    _require(tensor_sha256(dev_z) == DEV_OLS_TENSOR_SHA256, "accepted DEV OLS tensor identity changed")
    _require(tensor_sha256(dev_raw) == DEV_CMOS_TENSOR_SHA256, "accepted DEV CMOS tensor identity changed")
    _require(torch.equal(dev_indices, _tensor(cmos_map.get("source_indices"), "accepted DEV CMOS indices")), "DEV OLS/CMOS indices differ")
    _require(torch.equal(dev_labels, _tensor(cmos_map.get("labels"), "accepted DEV CMOS labels")), "DEV OLS/CMOS labels differ")
    _require(dict(ols_map.get("data_identity", {})) == dict(base["data_identity"]), "accepted DEV OLS data identity changed")
    _require(dict(cmos_map.get("data_identity", {})) == dict(base["data_identity"]), "accepted DEV CMOS data identity changed")
    _require(torch.equal(dev_indices, torch.sort(dev_indices).values), "accepted DEV indices are not canonical source order")
    _require(torch.bincount(dev_labels, minlength=CLASS_COUNT).tolist() == [500] * CLASS_COUNT, "accepted DEV labels are not balanced")

    clean_report = _mapping(base.get("clean_report"), "accepted clean report")
    smooth_report = _mapping(clean_report.get("dev_smooth"), "accepted clean DEV Smooth result")
    smooth_primary = _mapping(smooth_report.get("primary_evaluator"), "accepted Smooth evaluator")
    _require(smooth_primary.get("checkpoint_sha256") == SMOOTH_CHECKPOINT_SHA256, "accepted Smooth checkpoint identity changed")
    _require(smooth_primary.get("view") == "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28", "accepted Smooth preprocessing changed")
    smooth_path = root / SMOOTH_CHECKPOINT
    _require(smooth_path.is_file() and file_sha256(smooth_path) == SMOOTH_CHECKPOINT_SHA256, "accepted Smooth checkpoint is unavailable or changed")

    final = _mapping(base.get("final"), "accepted TRAIN OLS")
    train_z = _tensor(final.get("z_final"), "accepted TRAIN OLS tensor")
    train_indices = _tensor(final.get("source_indices"), "accepted TRAIN OLS indices")
    _require(train_z.shape == (TRAIN_COUNT, 100) and train_z.dtype == torch.complex64, "accepted TRAIN OLS shape/dtype changed")
    _finite(train_z.real, "accepted TRAIN OLS real part")
    _finite(train_z.imag, "accepted TRAIN OLS imaginary part")
    _require(final.get("canonical_order") == "increasing source row order", "accepted TRAIN OLS ordering contract changed")
    _require(torch.equal(train_indices, torch.sort(train_indices).values), "accepted TRAIN OLS indices are not canonical source order")

    return {
        "git": start_git,
        "base": base,
        "dev_report": dev_report,
        "dev_z": dev_z,
        "dev_raw": dev_raw,
        "dev_indices": dev_indices,
        "dev_labels": dev_labels,
        "train_z": train_z,
        "train_indices": train_indices,
        "smooth_report": smooth_report,
        "smooth_primary": smooth_primary,
        "smooth_path": smooth_path,
    }


def _load_split_labels(root: Path, expected_identity: Mapping[str, Any]) -> dict[str, torch.Tensor | dict[str, Any]]:
    source = root / TRAINING_SOURCE
    _require(source.is_file(), f"registered MNIST source is missing: {source}")
    _require(file_sha256(source) == TRAINING_SOURCE_SHA256, "registered MNIST source identity changed")
    images, labels = load_local_mnist(root / "data", train=True)
    del images
    gc.collect()
    _require(labels.shape == (60_000,) and labels.dtype == torch.long, "MNIST labels shape/dtype changed")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=SPLIT_SEED, validation_per_label=500)
    train_indices = torch.sort(train_indices).values.contiguous()
    development_indices = torch.sort(development_indices).values.contiguous()
    data_identity = dict(expected_identity)
    _require(tensor_sha256(labels) == data_identity.get("labels_tensor_sha256"), "MNIST label identity changed")
    _require(tensor_sha256(train_indices) == data_identity.get("train_indices_sha256"), "TRAIN ordering identity changed")
    _require(tensor_sha256(development_indices) == data_identity.get("development_indices_sha256"), "DEV ordering identity changed")
    _require(train_indices.numel() == TRAIN_COUNT and development_indices.numel() == DEV_COUNT, "split counts changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN and DEV roles overlap")
    return {
        "labels": labels,
        "train_indices": train_indices,
        "development_indices": development_indices,
        "train_labels": labels.index_select(0, train_indices).contiguous(),
        "development_labels": labels.index_select(0, development_indices).contiguous(),
        "data_identity": data_identity,
    }


def _state_snapshot(runtime: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in runtime.state_dict().items()}


def _states_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _rng_snapshot(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()


def _load_clean_model(root: Path, authorities: Mapping[str, Any]) -> tuple[LightGen2DChenBOGTComplexOLSTrainingConsumer, dict[str, Any]]:
    cuda_runtime = dev_materializer._prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "clean TRAIN CMOS materialization requires CUDA")
    consumer, model_identity = dev_materializer._load_frozen_model(root, authorities["base"], torch.device("cuda:0"))
    _require(all(not parameter.requires_grad for parameter in consumer.runtime.parameters()), "clean optical model is not frozen")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "clean optical model has a stale gradient")
    model_identity = dict(model_identity)
    model_identity["cuda_runtime"] = cuda_runtime
    return consumer, model_identity


def _validate_external_result(result: Any, expected_rows: int) -> torch.Tensor:
    _require(result.image_input is None and result.ols_input_field is None, "TRAIN CMOS path unexpectedly executed an image/encoder stage")
    _require(result.z_ols.source == "external_generated", "TRAIN OLS source provenance changed")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "TRAIN CMOS path did not use W_OLS")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "TRAIN CMOS transport changed")
    _require(result.raw_camera.plane.plane_id == "lightgen_raw_cmos", "TRAIN CMOS plane identity changed")
    raw = result.raw_intensity
    _require(raw.shape == (expected_rows, 1, 200, 200) and raw.dtype == torch.float32, "TRAIN raw CMOS shape/dtype changed")
    _finite(raw, "TRAIN raw CMOS batch")
    _require(bool((raw >= 0.0).all().item()), "TRAIN raw CMOS contains negative intensity")
    return raw.detach().cpu()


def _materialize_train_cmos(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    train_z: torch.Tensor,
    train_indices: torch.Tensor,
    train_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _require(train_z.shape == (TRAIN_COUNT, 100) and train_z.dtype == torch.complex64, "TRAIN OLS input is invalid")
    _require(train_indices.shape == (TRAIN_COUNT,) and train_labels.shape == (TRAIN_COUNT,), "TRAIN identity vectors are invalid")
    before_state = _state_snapshot(consumer.runtime)
    before_rng = _rng_snapshot(consumer.device)
    raw = torch.empty((TRAIN_COUNT, 1, 200, 200), dtype=torch.float32)
    batch_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, TRAIN_COUNT, TRAIN_FORWARD_BATCH_SIZE):
            stop = min(start + TRAIN_FORWARD_BATCH_SIZE, TRAIN_COUNT)
            result = consumer.runtime.forward_from_ols(train_z[start:stop].to(consumer.device))
            raw[start:stop].copy_(_validate_external_result(result, stop - start))
            batch_count += 1
            if batch_count == 1 or batch_count % 250 == 0 or stop == TRAIN_COUNT:
                print(
                    json.dumps(
                        {
                            "phase": "TRAIN_CMOS_FORWARD",
                            "completed_rows": stop,
                            "total_rows": TRAIN_COUNT,
                            "batch_count": batch_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            del result
    torch.cuda.synchronize(consumer.device)
    after_state = _state_snapshot(consumer.runtime)
    after_rng = _rng_snapshot(consumer.device)
    _require(_states_equal(before_state, after_state), "clean optical model state changed during TRAIN CMOS forward")
    _require(torch.equal(before_rng[0], after_rng[0]) and torch.equal(before_rng[1], after_rng[1]), "TRAIN CMOS forward consumed RNG state")
    _require(all(not parameter.requires_grad for parameter in consumer.runtime.parameters()), "clean optical model freeze was not preserved")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "TRAIN CMOS forward produced a parameter gradient")
    flat = raw.reshape(TRAIN_COUNT, -1)
    row_max = flat.amax(dim=1)
    row_min = flat.amin(dim=1)
    zero_count = int((row_max == 0.0).sum().item())
    constant_count = int((row_max == row_min).sum().item())
    _require(zero_count == 0 and constant_count == 0, "TRAIN raw CMOS contains zero or constant images")
    return raw, {
        "rows": TRAIN_COUNT,
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "finite_fraction": 1.0,
        "min": float(raw.min().item()),
        "max": float(raw.max().item()),
        "zero_image_count": zero_count,
        "constant_image_count": constant_count,
        "source_indices_sha256": tensor_sha256(train_indices),
        "labels_sha256": tensor_sha256(train_labels),
        "forward": {
            "device": str(consumer.device),
            "batch_size": TRAIN_FORWARD_BATCH_SIZE,
            "batch_count": batch_count,
            "optical_input": "accepted TRAIN complex OLS only",
            "image_encoder_forward": False,
            "optimizer_created": False,
            "backward_through_optics": False,
            "runtime_state_unchanged": True,
            "rng_state_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def _validate_existing_train_cmos(
    root: Path,
    path: Path,
    authorities: Mapping[str, Any],
    train_indices: torch.Tensor,
    train_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any], str, str]:
    """Resume only from the exact partial artifact created by this task."""
    file_digest = file_sha256(path)
    payload = safe_torch_load(path, expected_sha256=file_digest, map_location="cpu")
    payload_map = _mapping(payload, "partial TRAIN CMOS payload")
    _require(payload_map.get("schema") == TRAIN_CMOS_SCHEMA, "partial TRAIN CMOS schema changed")
    _require(payload_map.get("stage") == TRAIN_CMOS_STAGE, "partial TRAIN CMOS stage changed")
    raw = _tensor(payload_map.get("raw_cmos_intensity"), "partial TRAIN raw CMOS")
    saved_indices = _tensor(payload_map.get("source_indices"), "partial TRAIN indices")
    saved_labels = _tensor(payload_map.get("labels"), "partial TRAIN labels")
    _require(raw.shape == (TRAIN_COUNT, 1, 200, 200) and raw.dtype == torch.float32, "partial TRAIN CMOS shape/dtype changed")
    _require(saved_indices.shape == (TRAIN_COUNT,) and saved_labels.shape == (TRAIN_COUNT,), "partial TRAIN identity shape changed")
    _require(torch.equal(saved_indices, train_indices), "partial TRAIN CMOS ordering does not bind to accepted TRAIN OLS")
    _require(torch.equal(saved_labels, train_labels), "partial TRAIN CMOS labels do not bind to accepted TRAIN ordering")
    _require(dict(payload_map.get("data_identity", {})) == dict(authorities["base"]["data_identity"]), "partial TRAIN CMOS data identity changed")
    clean_authority = _mapping(payload_map.get("clean_model_authority"), "partial TRAIN clean authority")
    expected_clean = _artifact_ref(root, root / dev_materializer.CLEAN_CHECKPOINT, role="accepted clean 2L step-0 model")
    saved_checkpoint = _mapping(clean_authority.get("checkpoint"), "partial TRAIN clean checkpoint")
    _require(saved_checkpoint.get("path") == expected_clean["path"] and saved_checkpoint.get("sha256") == expected_clean["sha256"], "partial TRAIN CMOS clean checkpoint authority changed")
    expected_clean_state = authorities["base"]["clean"]
    _require(clean_authority.get("encoder_phase_sha256") == tensor_sha256(_tensor(expected_clean_state.get("encoder_state"), "accepted clean encoder")), "partial TRAIN encoder authority changed")
    _require(clean_authority.get("generator_phase_sha256") == tensor_sha256(_tensor(expected_clean_state.get("generator_state"), "accepted clean generator")), "partial TRAIN generator authority changed")
    expected_runtime = _mapping(authorities["base"]["parent"].get("runtime"), "accepted clean runtime")
    _require(dict(clean_authority.get("runtime", {})) == dict(expected_runtime), "partial TRAIN runtime authority changed")

    flat = raw.reshape(TRAIN_COUNT, -1)
    zero_count = 0
    constant_count = 0
    for start in range(0, TRAIN_COUNT, 256):
        stop = min(start + 256, TRAIN_COUNT)
        chunk = raw[start:stop]
        _finite(chunk, "partial TRAIN raw CMOS")
        _require(bool((chunk >= 0.0).all().item()), "partial TRAIN raw CMOS contains negative intensity")
        chunk_flat = chunk.reshape(stop - start, -1)
        row_max = chunk_flat.amax(dim=1)
        row_min = chunk_flat.amin(dim=1)
        zero_count += int((row_max == 0.0).sum().item())
        constant_count += int((row_max == row_min).sum().item())
    _require(zero_count == 0 and constant_count == 0, "partial TRAIN raw CMOS contains zero or constant images")
    tensor_digest = _tensor_sha256_streaming(raw)
    diagnostics = {
        "rows": TRAIN_COUNT,
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "finite_fraction": 1.0,
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "zero_image_count": zero_count,
        "constant_image_count": constant_count,
        "source_indices_sha256": tensor_sha256(train_indices),
        "labels_sha256": tensor_sha256(train_labels),
        "forward": {
            "device": "not_repeated; reused published clean frozen authority",
            "batch_size": TRAIN_FORWARD_BATCH_SIZE,
            "batch_count": 0,
            "optical_input": "accepted TRAIN complex OLS only",
            "image_encoder_forward": False,
            "optimizer_created": False,
            "backward_through_optics": False,
            "runtime_state_unchanged": True,
            "rng_state_unchanged": True,
            "resumed_partial_task": True,
        },
    }
    return raw, diagnostics, file_digest, tensor_digest


def _cmos_view(raw: torch.Tensor) -> torch.Tensor:
    _require(raw.ndim == 4 and raw.shape[1:] == (1, 200, 200), "raw CMOS input geometry changed")
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), CMOS_POOL_SHAPE).flatten(1)


def _cmos_image_view(raw: torch.Tensor) -> torch.Tensor:
    _require(raw.ndim == 4 and raw.shape[1:] == (1, 200, 200), "raw CMOS input geometry changed")
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), CMOS_POOL_SHAPE)


def _ols_features(z: torch.Tensor) -> torch.Tensor:
    _require(z.ndim == 2 and z.shape[1] == 100 and torch.is_complex(z), "OLS representation must be complex [N,100]")
    features = torch.cat((z.real, z.imag), dim=1).to(torch.float64).contiguous()
    _finite(features, "OLS real Cartesian features")
    return features


def _cmos_features(raw: torch.Tensor) -> torch.Tensor:
    features = _cmos_view(raw).to(torch.float64).contiguous()
    _finite(features, "CMOS pooled features")
    return features


def _standardize_train_only(train: torch.Tensor, dev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    _require(train.ndim == 2 and dev.ndim == 2 and train.shape[1] == dev.shape[1], "feature dimensions do not match")
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False)
    safe_std = torch.where(std > 0.0, std, torch.ones_like(std))
    train_z = ((train - mean) / safe_std).contiguous()
    dev_z = ((dev - mean) / safe_std).contiguous()
    _finite(train_z, "standardized TRAIN features")
    _finite(dev_z, "standardized DEV features")
    return train_z, dev_z, {
        "fit_role": "TRAIN only",
        "feature_count": train.shape[1],
        "mean_tensor_sha256": tensor_sha256(mean),
        "std_tensor_sha256": tensor_sha256(safe_std),
        "constant_feature_count": int((std == 0.0).sum().item()),
    }


def _fit_logistic(features: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    _require(features.ndim == 2 and labels.ndim == 1 and features.shape[0] == labels.shape[0], "logistic fit shapes do not match")
    _require(features.dtype == torch.float64 and labels.dtype == torch.long, "logistic fit dtype contract changed")
    _finite(features, "logistic TRAIN features")
    _require(torch.equal(torch.unique(labels), torch.arange(CLASS_COUNT, dtype=torch.long)), "logistic TRAIN labels do not cover classes 0-9")
    n_rows, feature_count = features.shape
    weights = nn.Parameter(torch.zeros((CLASS_COUNT, feature_count), dtype=torch.float64))
    bias = nn.Parameter(torch.zeros((CLASS_COUNT,), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [weights, bias],
        lr=1.0,
        max_iter=PROBE_MAX_ITER,
        history_size=PROBE_HISTORY_SIZE,
        tolerance_grad=PROBE_TOLERANCE_GRAD,
        tolerance_change=PROBE_TOLERANCE_CHANGE,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        optimizer.zero_grad(set_to_none=True)
        logits = features @ weights.T + bias
        loss = F.cross_entropy(logits, labels, reduction="mean") + weights.square().sum() / (2.0 * PROBE_C * n_rows)
        _finite(loss, "logistic objective")
        loss.backward()
        closure_calls += 1
        return loss

    optimizer.step(closure)
    _finite(weights, "logistic weights")
    _finite(bias, "logistic bias")
    return {"weights": weights.detach().clone(), "bias": bias.detach().clone()}, {
        "family": "multinomial_logistic_regression",
        "regularization": "L2_on_weights_only",
        "C": PROBE_C,
        "objective": "mean_cross_entropy + ||W||_F^2/(2*C*N); bias unregularized",
        "solver": "torch.optim.LBFGS(strong_wolfe)",
        "max_iter": PROBE_MAX_ITER,
        "history_size": PROBE_HISTORY_SIZE,
        "tolerance_grad": PROBE_TOLERANCE_GRAD,
        "tolerance_change": PROBE_TOLERANCE_CHANGE,
        "closure_calls": closure_calls,
        "fit_rows": n_rows,
        "feature_count": feature_count,
        "backward_scope": "probe parameters only; no optical tensor requires_grad",
    }


def _predict(model: Mapping[str, torch.Tensor], features: torch.Tensor) -> torch.Tensor:
    logits = features @ _tensor(model.get("weights"), "probe weights").T + _tensor(model.get("bias"), "probe bias")
    _finite(logits, "probe logits")
    return logits.argmax(dim=1).to(torch.long)


def _metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    _require(predictions.shape == labels.shape and predictions.ndim == 1, "prediction/label shapes do not match")
    confusion = torch.zeros((CLASS_COUNT, CLASS_COUNT), dtype=torch.long)
    for actual in range(CLASS_COUNT):
        actual_mask = labels == actual
        _require(bool(actual_mask.any().item()), f"class {actual} has no metric rows")
        for predicted in range(CLASS_COUNT):
            confusion[actual, predicted] = int(((predictions == predicted) & actual_mask).sum().item())
    row_counts = confusion.sum(dim=1)
    per_class = torch.diag(confusion).to(torch.float64) / row_counts.to(torch.float64)
    macro = float(per_class.mean().item())
    accuracy = float((predictions == labels).to(torch.float64).mean().item())
    _require(math.isfinite(macro) and math.isfinite(accuracy), "probe metrics are nonfinite")
    return {
        "macro_recall": macro,
        "accuracy": accuracy,
        "per_class_recall": [float(value) for value in per_class.tolist()],
        "confusion_matrix_actual_rows_predicted_columns": confusion.tolist(),
        "rows": int(labels.numel()),
        "prediction_tensor_sha256": tensor_sha256(predictions),
    }


def _evaluate_probe(
    model: Mapping[str, torch.Tensor],
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    dev_features: torch.Tensor,
    dev_labels: torch.Tensor,
) -> dict[str, Any]:
    train_predictions = _predict(model, train_features)
    dev_predictions = _predict(model, dev_features)
    return {
        "train": _metrics(train_predictions, train_labels),
        "dev": _metrics(dev_predictions, dev_labels),
        "_train_predictions": train_predictions,
        "_dev_predictions": dev_predictions,
    }


def _load_smooth(root: Path) -> tuple[SmoothMNISTEvaluator, dict[str, Any]]:
    payload = safe_torch_load(root / SMOOTH_CHECKPOINT, expected_sha256=SMOOTH_CHECKPOINT_SHA256, map_location="cpu")
    payload_map = _mapping(payload, "Smooth evaluator payload")
    _require(payload_map.get("architecture_kind") == "smooth_mnist_evaluator_v2", "Smooth evaluator architecture changed")
    state = _mapping(payload_map.get("state_dict"), "Smooth evaluator state")
    evaluator = SmoothMNISTEvaluator()
    evaluator.load_state_dict(state, strict=True)
    evaluator.freeze()
    _require(all(not parameter.requires_grad for parameter in evaluator.parameters()), "Smooth evaluator is not frozen")
    return evaluator, {
        "path": _relative(root, root / SMOOTH_CHECKPOINT),
        "sha256": SMOOTH_CHECKPOINT_SHA256,
        "architecture": "SmoothMNISTEvaluator v2.1",
        "implementation": "src/ldm_project/v2/student/evaluator.py",
        "implementation_sha256": file_sha256(root / "src/ldm_project/v2/student/evaluator.py"),
        "view": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28",
        "role": "independent DEV primary; frozen evaluation only; no training gradient",
    }


def _smooth_metrics(evaluator: SmoothMNISTEvaluator, raw: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
    predictions = torch.empty((raw.shape[0],), dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, raw.shape[0], SMOOTH_BATCH_SIZE):
            stop = min(start + SMOOTH_BATCH_SIZE, raw.shape[0])
            _, logits = evaluator(_cmos_image_view(raw[start:stop]))
            predictions[start:stop].copy_(logits.argmax(dim=1).cpu())
    return _metrics(predictions, labels), predictions


def _matched_transitions(
    labels: torch.Tensor,
    ols_predictions: torch.Tensor,
    cmos_predictions: torch.Tensor,
    smooth_predictions: torch.Tensor,
) -> dict[str, Any]:
    ols_correct = ols_predictions == labels
    cmos_correct = cmos_predictions == labels
    smooth_correct = smooth_predictions == labels
    matrix = torch.zeros((2, 2), dtype=torch.long)
    for ols_state in range(2):
        for cmos_state in range(2):
            matrix[ols_state, cmos_state] = int(((ols_correct == bool(ols_state)) & (cmos_correct == bool(cmos_state))).sum().item())
    triple: dict[str, int] = {}
    for ols_state in (False, True):
        for cmos_state in (False, True):
            for smooth_state in (False, True):
                key = f"OLS_{'correct' if ols_state else 'wrong'}__CMOS_{'correct' if cmos_state else 'wrong'}__Smooth_{'correct' if smooth_state else 'wrong'}"
                triple[key] = int(((ols_correct == ols_state) & (cmos_correct == cmos_state) & (smooth_correct == smooth_state)).sum().item())
    per_class = []
    for label in range(CLASS_COUNT):
        mask = labels == label
        per_class.append(
            {
                "class": label,
                "rows": int(mask.sum().item()),
                "ols_recall": float(ols_correct[mask].to(torch.float64).mean().item()),
                "cmos_recall": float(cmos_correct[mask].to(torch.float64).mean().item()),
                "smooth_recall": float(smooth_correct[mask].to(torch.float64).mean().item()),
            }
        )
    return {
        "ols_correct_cmos_correct": int((ols_correct & cmos_correct).sum().item()),
        "ols_correct_cmos_wrong": int((ols_correct & ~cmos_correct).sum().item()),
        "ols_wrong_cmos_correct": int((~ols_correct & cmos_correct).sum().item()),
        "ols_wrong_cmos_wrong": int((~ols_correct & ~cmos_correct).sum().item()),
        "ols_to_cmos_matrix": matrix.tolist(),
        "ols_to_cmos_matrix_axes": {"rows": ["OLS_wrong", "OLS_correct"], "columns": ["CMOS_wrong", "CMOS_correct"]},
        "ols_to_cmos_to_smooth_triple_counts": triple,
        "per_class": per_class,
    }


def _negative_control(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    dev_features: torch.Tensor,
    dev_labels: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(NEGATIVE_CONTROL_SEED)
    permutation = torch.randperm(train_labels.numel(), generator=generator)
    permuted_labels = train_labels.index_select(0, permutation)
    model, solver = _fit_logistic(train_features, permuted_labels)
    evaluation = _evaluate_probe(model, train_features, permuted_labels, dev_features, dev_labels)
    report = {
        "kind": "one_deterministic_train_label_permutation",
        "seed": NEGATIVE_CONTROL_SEED,
        "permutation_tensor_sha256": tensor_sha256(permutation),
        "permuted_train_labels_sha256": tensor_sha256(permuted_labels),
        "solver": solver,
        "train_permuted_labels": evaluation["train"],
        "dev_real_labels": evaluation["dev"],
        "purpose": "detect pipeline leakage or sample-label identity errors",
    }
    return report, model


def _classify(
    ols: Mapping[str, Any],
    cmos: Mapping[str, Any],
    smooth: Mapping[str, Any],
    negative: Mapping[str, Any],
    transitions: Mapping[str, Any],
) -> dict[str, Any]:
    chance = 1.0 / CLASS_COUNT
    ols_train = float(_mapping(ols.get("train"), "OLS train metrics")["macro_recall"])
    ols_dev = float(_mapping(ols.get("dev"), "OLS DEV metrics")["macro_recall"])
    cmos_train = float(_mapping(cmos.get("train"), "CMOS train metrics")["macro_recall"])
    cmos_dev = float(_mapping(cmos.get("dev"), "CMOS DEV metrics")["macro_recall"])
    smooth_dev = float(_mapping(smooth, "Smooth DEV metrics")["macro_recall"])
    negative_dev = float(_mapping(negative.get("dev_real_labels"), "negative-control DEV metrics")["macro_recall"])
    ols_transfer = ols_train > chance and ols_dev > chance and ols_dev > negative_dev
    cmos_transfer = cmos_train > chance and cmos_dev > chance
    coupling_loss = ols_dev > cmos_dev and int(transitions["ols_correct_cmos_wrong"]) > int(transitions["ols_wrong_cmos_correct"])
    smooth_weak = smooth_dev <= chance
    if ols_transfer and coupling_loss:
        classification = "OLS_TO_GENERATOR_SEMANTIC_COUPLING_BOTTLENECK_SUPPORTED"
        mechanism = "transferable linearly decodable class information is present in OLS and matched correct OLS rows are lost at CMOS more often than recovered"
    elif not ols_transfer and ols_dev <= chance:
        classification = "ENCODER_OLS_SEMANTIC_REPRESENTATION_BOTTLENECK_SUPPORTED"
        mechanism = "the primary OLS probe does not transfer above the ten-class chance context"
    elif ols_transfer and cmos_transfer and smooth_weak:
        classification = "EVALUATOR_OR_OUTPUT_DECISION_MISMATCH_SUPPORTED"
        mechanism = "class information remains linearly decodable through CMOS while the frozen Smooth decision path remains at or below chance context"
    else:
        classification = "SEMANTIC_INFORMATION_LOCALIZATION_MIXED"
        mechanism = "the observed train-to-DEV, OLS-to-CMOS, CMOS-to-Smooth, or per-class transitions do not isolate one bottleneck under the preregistered comparisons"
    return {
        "classification": classification,
        "mechanism": mechanism,
        "chance_macro_recall_context": chance,
        "evidence_flags": {
            "ols_train_above_chance": ols_train > chance,
            "ols_dev_above_chance": ols_dev > chance,
            "ols_dev_above_permuted_control": ols_dev > negative_dev,
            "ols_transferable_by_preregistered_comparison": ols_transfer,
            "cmos_train_above_chance": cmos_train > chance,
            "cmos_dev_above_chance": cmos_dev > chance,
            "cmos_transferable_by_chance_context": cmos_transfer,
            "smooth_at_or_below_chance": smooth_weak,
            "matched_ols_correct_to_cmos_wrong_exceeds_reverse": coupling_loss,
        },
        "decision_rule": "chance context plus TRAIN-to-DEV comparison, negative-control comparison, strict OLS-to-CMOS ordering, and matched transition direction; no arbitrary accuracy cutoff",
    }


def _source_manifest(root: Path, start_git: Mapping[str, Any], authorities: Mapping[str, Any], target: Path) -> dict[str, Any]:
    relevant_paths = [
        Path(__file__).resolve(),
        root / "scripts/research/lightgen/materialize_lightgen_2d_frozen_dev_ols_cmos_v1.py",
        root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
        root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
        root / "src/ldm_project/v2/student/evaluator.py",
    ]
    relative_paths = [_relative(root, path) for path in relevant_paths]
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "authorization_token": AUTHORIZATION_TOKEN,
        "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
        "branch": start_git.get("branch"),
        "head": start_git.get("head"),
        "dirty_before": bool(start_git.get("worktree_dirty")),
        "status_sha256_before": start_git.get("status_sha256"),
        "relevant_source_paths": relative_paths,
        "relevant_source_sha256": {path: file_sha256(root / path) for path in relative_paths},
        "input_authorities": {
            "accepted_dev_report": _artifact_ref(root, root / DEV_REPORT, role="accepted DEV OLS/CMOS materialization report"),
            "accepted_dev_ols": _artifact_ref(root, root / DEV_OLS, role="accepted DEV complex OLS"),
            "accepted_dev_cmos": _artifact_ref(root, root / DEV_CMOS, role="accepted DEV raw CMOS"),
            "accepted_train_ols": _artifact_ref(root, root / dev_materializer.FINAL_TRAIN_OLS, role="accepted TRAIN complex OLS"),
            "accepted_clean_step0": _artifact_ref(root, root / dev_materializer.CLEAN_CHECKPOINT, role="accepted clean 2L step-0 model"),
            "accepted_smooth": _artifact_ref(root, root / SMOOTH_CHECKPOINT, role="accepted frozen Smooth evaluator"),
            "training_source": _artifact_ref(root, root / TRAINING_SOURCE, role="accepted MNIST TRAIN source"),
        },
        "output_dir": _relative(root, target),
        "scope": {
            "train_forward_rows": TRAIN_COUNT,
            "dev_measurement_rows": DEV_COUNT,
            "protected_rows": 0,
            "test_rows": 0,
            "sealed_rows": 0,
            "optical_updates": False,
            "optical_optimizer": False,
            "optical_backward": False,
            "probe_family_sweep": False,
            "dev_tuning": False,
        },
    }


def _clean_public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clean_public(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_clean_public(item) for item in value]
    return value


def execute(*, repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    requested_output = DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    requested_target = (root / requested_output if not requested_output.is_absolute() else requested_output).resolve()
    resumed_partial = requested_target.exists()
    target = _validate_partial_target(root, requested_output) if resumed_partial else _validate_target(root, requested_output)
    authorities = _load_authorities(root)
    split = _load_split_labels(root, authorities["base"]["data_identity"])
    _require(torch.equal(authorities["train_indices"], split["train_indices"]), "accepted TRAIN OLS rows do not bind to reconstructed TRAIN ordering")
    _require(torch.equal(authorities["dev_indices"], split["development_indices"]), "accepted DEV rows do not bind to reconstructed DEV ordering")
    _require(torch.equal(authorities["dev_labels"], split["development_labels"]), "accepted DEV labels do not bind to reconstructed DEV ordering")
    train_labels = _tensor(split["train_labels"], "TRAIN labels")
    dev_labels = _tensor(split["development_labels"], "DEV labels")

    if resumed_partial:
        train_cmos_path = target / "train_raw_cmos.pt"
        train_raw, train_cmos_diagnostics, train_cmos_sha256, train_cmos_tensor_sha256 = _validate_existing_train_cmos(
            root,
            train_cmos_path,
            authorities,
            authorities["train_indices"],
            train_labels,
        )
        print(json.dumps({"phase": "TRAIN_CMOS_AUTHORITY_REUSED", "path": _relative(root, train_cmos_path), "sha256": train_cmos_sha256}, sort_keys=True), flush=True)
    else:
        consumer, model_identity = _load_clean_model(root, authorities)
        train_raw, train_cmos_diagnostics = _materialize_train_cmos(consumer, authorities["train_z"], authorities["train_indices"], train_labels)
        del consumer
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=False, exist_ok=False)
        train_cmos_path = target / "train_raw_cmos.pt"

    start_git = authorities["git"]
    source_manifest = _source_manifest(root, start_git, authorities, target)
    if resumed_partial:
        source_manifest_path = target / "resume_source_manifest.json"
        atomic_write_new_json(source_manifest_path, source_manifest, artifact_kind="OLS-to-CMOS resume source manifest")
        previous_source_manifest_path = target / "source_manifest.json"
        previous_source_manifest_sha256 = file_sha256(previous_source_manifest_path)
    else:
        source_manifest_path = target / "source_manifest.json"
        atomic_write_new_json(source_manifest_path, source_manifest, artifact_kind="OLS-to-CMOS source manifest")
        previous_source_manifest_path = None
        previous_source_manifest_sha256 = None

    if not resumed_partial:
        atomic_torch_save_new(
            train_cmos_path,
            {
                "schema": TRAIN_CMOS_SCHEMA,
                "representation": "real_TRAIN_raw_CMOS",
                "stage": TRAIN_CMOS_STAGE,
                "raw_cmos_intensity": train_raw,
                "source_indices": authorities["train_indices"].clone(),
                "labels": train_labels.clone(),
                "data_identity": authorities["base"]["data_identity"],
                "canonical_order": "increasing source row order",
                "clean_model_authority": {
                    "checkpoint": _artifact_ref(root, root / dev_materializer.CLEAN_CHECKPOINT, role="accepted clean 2L step-0 model"),
                    "runtime": model_identity["runtime_identity"],
                    "encoder_phase_sha256": model_identity["loaded_state"]["encoder_phase_sha256"],
                    "generator_phase_sha256": model_identity["loaded_state"]["generator_phase_sha256"],
                },
            },
            artifact_kind="frozen TRAIN raw CMOS",
        )
        train_cmos_sha256 = file_sha256(train_cmos_path)
        train_cmos_tensor_sha256 = _tensor_sha256_streaming(train_raw)
        print(json.dumps({"phase": "TRAIN_CMOS_AUTHORITY_PUBLISHED", "path": _relative(root, train_cmos_path), "sha256": train_cmos_sha256}, sort_keys=True), flush=True)

    train_ols_features = _ols_features(authorities["train_z"])
    dev_ols_features = _ols_features(authorities["dev_z"])
    train_cmos_features = _cmos_features(train_raw)
    dev_cmos_features = _cmos_features(authorities["dev_raw"])
    del train_raw
    gc.collect()

    train_ols_std, dev_ols_std, ols_standardization = _standardize_train_only(train_ols_features, dev_ols_features)
    train_cmos_std, dev_cmos_std, cmos_standardization = _standardize_train_only(train_cmos_features, dev_cmos_features)
    del train_ols_features, dev_ols_features, train_cmos_features, dev_cmos_features
    gc.collect()

    ols_model, ols_solver = _fit_logistic(train_ols_std, train_labels)
    ols_evaluation = _evaluate_probe(ols_model, train_ols_std, train_labels, dev_ols_std, dev_labels)
    print(json.dumps({"phase": "OLS_PROBE_COMPLETE", "dev_macro_recall": ols_evaluation["dev"]["macro_recall"]}, sort_keys=True), flush=True)
    cmos_model, cmos_solver = _fit_logistic(train_cmos_std, train_labels)
    cmos_evaluation = _evaluate_probe(cmos_model, train_cmos_std, train_labels, dev_cmos_std, dev_labels)
    print(json.dumps({"phase": "CMOS_PROBE_COMPLETE", "dev_macro_recall": cmos_evaluation["dev"]["macro_recall"]}, sort_keys=True), flush=True)
    negative_report, negative_model = _negative_control(train_ols_std, train_labels, dev_ols_std, dev_labels)

    smooth, smooth_identity = _load_smooth(root)
    smooth_metrics, smooth_predictions = _smooth_metrics(smooth, authorities["dev_raw"], dev_labels)
    expected_smooth = authorities["smooth_report"]
    _require(math.isclose(smooth_metrics["macro_recall"], float(expected_smooth["step_0_macro_recall"]), rel_tol=0.0, abs_tol=1.0e-7), "recomputed DEV Smooth macro recall does not reproduce accepted clean step-0 result")
    _require(smooth_metrics["per_class_recall"] == [float(value) for value in expected_smooth["step_0_per_class_recall"]], "recomputed DEV Smooth per-class recall does not reproduce accepted clean step-0 result")

    transitions = _matched_transitions(
        dev_labels,
        ols_evaluation["_dev_predictions"],
        cmos_evaluation["_dev_predictions"],
        smooth_predictions,
    )
    classification = _classify(ols_evaluation, cmos_evaluation, smooth_metrics, negative_report, transitions)
    ols_dev_recall = float(ols_evaluation["dev"]["macro_recall"])
    cmos_dev_recall = float(cmos_evaluation["dev"]["macro_recall"])
    smooth_dev_recall = float(smooth_metrics["macro_recall"])
    retention = {
        "OLS_to_CMOS": None if ols_dev_recall <= 0.0 else cmos_dev_recall / ols_dev_recall,
        "CMOS_to_Smooth": None if cmos_dev_recall <= 0.0 else smooth_dev_recall / cmos_dev_recall,
        "interpretation": "descriptive ratios only; not mutual information",
    }

    probe_weights_path = target / "probe_weights.pt"
    atomic_torch_save_new(
        probe_weights_path,
        {
            "schema": PROBE_SCHEMA,
            "task_id": TASK_ID,
            "primary_features": {
                "ols": "concat(real(z), imag(z)) in R^200",
                "cmos": "clamp[0,1] -> adaptive_avg_pool2d(28,28) -> flatten",
            },
            "standardization": {"ols": ols_standardization, "cmos": cmos_standardization},
            "solver": {"ols": ols_solver, "cmos": cmos_solver, "negative_control": negative_report["solver"]},
            "ols": ols_model,
            "cmos": cmos_model,
            "negative_control_ols": negative_model,
        },
        artifact_kind="OLS-to-CMOS probe weights",
    )
    probe_weights_sha256 = file_sha256(probe_weights_path)

    metrics_path = target / "probe_metrics.json"
    metrics_payload = {
        "schema": "ols_to_cmos_semantic_localization_probe_metrics.v1",
        "task_id": TASK_ID,
        "ols": _clean_public(ols_evaluation),
        "cmos": _clean_public(cmos_evaluation),
        "smooth": smooth_metrics,
        "negative_control": negative_report,
        "matched_transitions": transitions,
        "semantic_retention": retention,
        "classification": classification,
    }
    atomic_write_new_json(metrics_path, metrics_payload, artifact_kind="OLS-to-CMOS probe metrics")
    metrics_sha256 = file_sha256(metrics_path)

    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "result": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "execution_recovery": {
            "partial_task_reused": resumed_partial,
            "optical_forward_repeated": False if resumed_partial else True,
            "previous_source_manifest": None
            if previous_source_manifest_path is None
            else {"path": _relative(root, previous_source_manifest_path), "sha256": previous_source_manifest_sha256},
        },
        "local_state": {"branch": start_git["branch"], "head": start_git["head"], "dirty_before": bool(start_git.get("worktree_dirty")), "git_end": git_identity(root)},
        "objective": "Determine where class-conditional semantic information degrades across image -> complex OLS -> raw CMOS -> Smooth output.",
        "train_cmos_authority": {
            "resolution": "reused_exactly_one_clean_frozen_train_artifact_from_partial_task" if resumed_partial else "materialized_exactly_one_clean_frozen_train_artifact",
            "path": _relative(root, train_cmos_path),
            "sha256": train_cmos_sha256,
            "tensor_sha256": train_cmos_tensor_sha256,
            "shape": [TRAIN_COUNT, 1, 200, 200],
            "dtype": "torch.float32",
            "row_alignment": {
                "proven": True,
                "source_indices_sha256": tensor_sha256(authorities["train_indices"]),
                "labels_sha256": tensor_sha256(train_labels),
                "canonical_order": "increasing source row order",
                "same_clean_2l_authority": True,
                "intervention_checkpoint": False,
            },
            "diagnostics": train_cmos_diagnostics,
        },
        "input_authorities": {
            "accepted_dev_ols": _artifact_ref(root, root / DEV_OLS, role="accepted DEV complex OLS"),
            "accepted_dev_cmos": _artifact_ref(root, root / DEV_CMOS, role="accepted DEV raw CMOS"),
            "accepted_train_ols": _artifact_ref(root, root / dev_materializer.FINAL_TRAIN_OLS, role="accepted TRAIN complex OLS"),
            "accepted_clean_2l": _artifact_ref(root, root / dev_materializer.CLEAN_CHECKPOINT, role="accepted clean 2L step-0 model"),
            "accepted_smooth": smooth_identity,
            "accepted_data_identity": authorities["base"]["data_identity"],
        },
        "probe_contract": {
            "fit_role": "TRAIN",
            "measurement_role": "DEV",
            "ols_representation": "concat(Re(z), Im(z)) in R^200",
            "cmos_representation": "raw CMOS clamp[0,1] -> adaptive_avg_pool2d(28,28) -> flatten",
            "standardization": "mean/std fit on TRAIN only and applied unchanged to DEV",
            "family": "multinomial logistic regression + L2",
            "C": PROBE_C,
            "solver": "torch.optim.LBFGS(strong_wolfe)",
            "regularization": "weights only; bias unregularized",
            "sweep": False,
            "dev_tuning": False,
            "nonlinear_feature_extractor": False,
        },
        "ols_probe": _clean_public(ols_evaluation) | {"solver": ols_solver, "standardization": ols_standardization},
        "ols_negative_control": negative_report,
        "cmos_probe": _clean_public(cmos_evaluation) | {"solver": cmos_solver, "standardization": cmos_standardization},
        "smooth_reference": {
            "metrics": smooth_metrics,
            "authority": smooth_identity,
            "accepted_clean_step0_macro_recall": expected_smooth["step_0_macro_recall"],
            "accepted_clean_step0_per_class_recall": expected_smooth["step_0_per_class_recall"],
            "optical_rerun": False,
            "recomputed_from_materialized_dev_cmos": True,
        },
        "semantic_retention": retention,
        "matched_dev_transitions": transitions,
        "per_class_information_flow": transitions["per_class"],
        "optional_secondary_ols": {"executed": False, "reason": "primary preregistered Cartesian complex-OLS probe was sufficient"},
        "data_role_integrity": {
            "train_ols_rows": TRAIN_COUNT,
            "train_cmos_forward_rows": TRAIN_COUNT,
            "dev_ols_rows": DEV_COUNT,
            "dev_cmos_rows": DEV_COUNT,
            "dev_labels_used_only_for_measurement": True,
            "protected_reads": 0,
            "test_reads": 0,
            "sealed_reads": 0,
            "optical_training_updates": 0,
            "optical_optimizer_created": False,
            "optical_backward_executed": False,
        },
        "classification": classification["classification"],
        "mechanism": classification["mechanism"],
        "classification_evidence": classification,
        "claim_boundary": "The probes establish readily linearly decodable class information under the declared representations only; they do not establish exact mutual information, nonlinear information absence, optical inference readiness, or promotion.",
        "artifacts": {
            "source_manifest": {"path": _relative(root, source_manifest_path), "sha256": file_sha256(source_manifest_path)},
            "train_raw_cmos": {"path": _relative(root, train_cmos_path), "sha256": train_cmos_sha256, "tensor_sha256": train_cmos_tensor_sha256},
            "probe_weights": {"path": _relative(root, probe_weights_path), "sha256": probe_weights_sha256},
            "probe_metrics": {"path": _relative(root, metrics_path), "sha256": metrics_sha256},
            "report": {"path": _relative(root, target / "report.json")},
        },
        "files_changed": {
            "repository_source": [_relative(root, Path(__file__).resolve())],
            "artifacts": [
                _relative(root, source_manifest_path),
                _relative(root, train_cmos_path),
                _relative(root, probe_weights_path),
                _relative(root, metrics_path),
                _relative(root, target / "report.json"),
            ],
        },
        "validation": {
            "accepted_dev_row_alignment_reverified": True,
            "accepted_train_ols_row_alignment_reverified": True,
            "clean_model_state_unchanged": True,
            "clean_model_rng_unchanged": True,
            "smooth_reference_reproduced": True,
            "train_cmos_serialization": {"file_sha256": train_cmos_sha256, "tensor_sha256": train_cmos_tensor_sha256},
            "probe_weights_sha256": probe_weights_sha256,
            "probe_metrics_sha256": metrics_sha256,
        },
        "unresolved": [
            "Probe results are diagnostic and do not establish nonlinear semantic absence or mutual information.",
            "No optical training, coupling repair, evaluator replacement, generation, or promotion was executed.",
        ],
        "next_action": "NONE",
    }
    atomic_write_new_json(target / "report.json", report, artifact_kind="OLS-to-CMOS semantic localization report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and execute the frozen localization consumer")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(build_execution_plan(output_dir=args.output_dir), indent=2, sort_keys=True))
        return
    report = execute(output_dir=args.output_dir)
    print(json.dumps({"result": report["result"], "outer_gate": report["outer_gate"], "classification": report["classification"], "output_dir": report["artifacts"]["report"]["path"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
