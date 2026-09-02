"""Characterize the frozen S9 MPCDAR rank-16 float32 readout runtime.

This module is an engineering-only consumer of the immutable S9 artifact.  It
does not load the optical model or dataset, refit a readout, or alter the
frozen runtime.  The required measurement is a deterministic CPU readout
benchmark; GPU, full-W runtime, and end-to-end optical timing are explicitly
outside this qualification.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from ..current import frozen_mpcdar_rank16_float32_readout as runtime
from . import optical_manifold as base


CONFIG_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_runtime_performance_qualification_v1"
)
REVIEW_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_runtime_performance_qualification_review.v1"
)
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_runtime_performance_qualification_validation.v1"
)
AUDIT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_runtime_performance_qualification.v1"
)
COMPACT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_runtime_performance_qualification_compact_report.v1"
)

HYPOTHESIS = "H-LOMA-CBOGT-S10-MPCDAR-R16-RUNTIME-PQ"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
S10_STARTING_HEAD = "8680c12be955df72906b955433e6daeabfdfec4a"
S9V_MANIFEST = (
    "outputs/20260902_v2/visualizations/"
    "scaled_chen_full_path_reconstruction_v1_attempt_01/visualization_manifest.json"
)
S9V_MANIFEST_SHA256 = (
    "ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749"
)
S9_ARTIFACT_DIR = (
    "outputs/20260902_v2/runs/"
    "lightgen_scaled_chen_mpcdar_rank16_float32_numerical_qualification_v1_attempt_02/artifact"
)
S9_ARTIFACT_MANIFEST_SHA256 = (
    "1ad6d9edfb1037564c3aec533904ed8d48bc4614a7bbcb221a56c9a244d802c7"
)
S9_ARTIFACT_PAYLOAD_SHA256 = (
    "480c57bff42e28cdfdf5ba260978234ee98f3d1169407c2523beeba76d39e834"
)
S9_VALIDATION = (
    "outputs/20260902_v2/runs/"
    "lightgen_scaled_chen_mpcdar_rank16_float32_numerical_qualification_v1_attempt_02/validation.json"
)
S9_VALIDATION_SHA256 = (
    "5728fb9b50212d5dc7359975b06fe6d3f51be3645a23410683334abab7bb2137"
)
S9_AUDIT = (
    "outputs/20260902_v2/runs/"
    "lightgen_scaled_chen_mpcdar_rank16_float32_numerical_qualification_v1_attempt_02/audit.json"
)
S9_AUDIT_SHA256 = (
    "0ff4aeaddd2035f0ced865a5e9f76331db53a2a8a213ea18064bd8e1d7368bde"
)
S9_COMPACT = (
    "outputs/20260902_v2/runs/"
    "lightgen_scaled_chen_mpcdar_rank16_float32_numerical_qualification_v1_attempt_02/compact_report.json"
)
S9_COMPACT_SHA256 = (
    "5b45f1401489fab63382366045a95f52984505038a379ca8bd8d0a780424c940"
)

RANK = 16
FEATURE_DIM = 256
OUTPUT_DIM = 40000
RAW_FLOAT32_TENSOR_BYTES = 2737408
SERIALIZED_BYTES = 2739448
BATCH_SIZES = (1, 8, 32)
WARMUP_COUNT = 50
MEASURED_REPETITIONS = 200
FIXTURE_SEED = 20260902
TIMER = "time.perf_counter_ns"
QUANTILE_METHOD = "linear: index=(n-1)*q"
FULL_W_MACS_PER_SAMPLE = 10240000
FACTORIZED_MACS_PER_SAMPLE = 644096
MAC_RATIO = FULL_W_MACS_PER_SAMPLE / FACTORIZED_MACS_PER_SAMPLE
CLASSIFICATION = "frozen_rank16_float32_runtime_performance_characterized"
NEXT_ACTION_RESULT_REVIEW = "manual_review_MPCDAR_R16_RUNTIME_PQ_result"
PYTHON_EXECUTABLE = r"D:\ANACONDA\envs\optical-generative-models-win\python.exe"
EXPECTED_PYTHON_VERSION = "3.11.16"
EXPECTED_TORCH_VERSION = "2.5.1"
EXPECTED_WINDOWS_VERSION = "10.0.26200.0"
EXPECTED_WINDOWS_VERSION_PREFIX = "10.0.26200"


class MPCDARRuntimeQualificationHarnessFailure(RuntimeError):
    """A frozen-input or runtime characterization contract failed closed."""


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


def _repo_root() -> Path:
    return base._repo_root()


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    root_candidate = _repo_root() / candidate
    if root_candidate.exists() or base_path is None:
        return root_candidate.resolve()
    return (base_path / candidate).resolve()


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_registered_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise MPCDARRuntimeQualificationHarnessFailure(
            f"S10 requires clean registered branch {BRANCH}"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", S10_STARTING_HEAD, str(state["commit"])],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise MPCDARRuntimeQualificationHarnessFailure(
            "current S10 commit is not descended from the registered S9V closeout"
        )


def _config_payload() -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "branch": BRANCH,
        "starting_head": S10_STARTING_HEAD,
        "parent": {
            "s9v_manifest": S9V_MANIFEST,
            "s9v_manifest_sha256": S9V_MANIFEST_SHA256,
            "s9v_status": "completed",
            "s9v_classification": "descriptive_visualization_completed",
            "s9_artifact_dir": S9_ARTIFACT_DIR,
            "s9_manifest_sha256": S9_ARTIFACT_MANIFEST_SHA256,
            "s9_payload_sha256": S9_ARTIFACT_PAYLOAD_SHA256,
            "s9_validation": S9_VALIDATION,
            "s9_validation_sha256": S9_VALIDATION_SHA256,
            "s9_audit": S9_AUDIT,
            "s9_audit_sha256": S9_AUDIT_SHA256,
            "s9_compact": S9_COMPACT,
            "s9_compact_sha256": S9_COMPACT_SHA256,
        },
        "consumer": {
            "runtime_module": "src/ldm_project/v2/application/current/frozen_mpcdar_rank16_float32_readout.py",
            "repair_id": runtime.REPAIR_ID,
            "rank": RANK,
            "dtype": "torch.float32",
            "feature_dim": FEATURE_DIM,
            "output_dim": OUTPUT_DIM,
            "formula": "centered = z - mu_fit; h = centered @ T16; prediction = c_mean + h @ V16.T",
            "tensor_shapes": {
                "mu_fit": [256],
                "c_mean": [40000],
                "T16": [256, 16],
                "V16": [40000, 16],
            },
            "raw_tensor_bytes": RAW_FLOAT32_TENSOR_BYTES,
            "serialized_bytes": SERIALIZED_BYTES,
        },
        "benchmark": {
            "primary": "cpu_readout_only",
            "device": "cpu",
            "dtype": "torch.float32",
            "fixture_seed": FIXTURE_SEED,
            "fixture_shape": [32, 256],
            "batch_sizes": list(BATCH_SIZES),
            "warmup": WARMUP_COUNT,
            "repetitions": MEASURED_REPETITIONS,
            "timer": TIMER,
            "synchronization": "CPU eager synchronous; no CUDA synchronization",
            "quantile_method": QUANTILE_METHOD,
            "prediction_bank_saved": False,
            "steady_state_excludes": [
                "first_import",
                "artifact_load",
                "fixture_generation",
                "statistics",
                "json_writes",
            ],
            "fresh_process_initialization": True,
        },
        "optional": {
            "end_to_end_software_pipeline": "not_evaluated",
            "full_w_runtime": "not_evaluated",
            "gpu": "not_evaluated",
            "analytical_factorized_macs_per_sample": FACTORIZED_MACS_PER_SAMPLE,
            "analytical_full_w_macs_per_sample": FULL_W_MACS_PER_SAMPLE,
            "analytical_mac_ratio": MAC_RATIO,
        },
        "memory": {
            "process_memory_delta": "not_recorded",
            "peak_memory": "not_recorded",
            "rss_helper": "not_available_in_v2_contract",
        },
        "authorization": {
            "no_model_load": True,
            "no_cuda": True,
            "no_training": True,
            "no_refit": True,
            "no_rank_change": True,
            "no_precision_change": True,
            "no_optimizer": True,
            "no_backward": True,
            "no_quantization": True,
            "no_compile_or_jit": True,
            "no_system_tuning": True,
            "training_authorized": False,
            "formal_promotion": False,
            "generation": False,
        },
    }


def _review_payload() -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_cpu_only_frozen_mpcdar_rank16_runtime_characterization",
        "approved_contract": _config_payload(),
        "no_update_diagnostic_authorized": True,
        "status_success_semantics": "completed_when_required_cpu_telemetry_and_correctness_are_valid",
        "no_arbitrary_latency_threshold": True,
        "classification": CLASSIFICATION,
        "gpu_required_for_success": False,
        "full_w_runtime_required": False,
        "end_to_end_pipeline_required": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation": False,
        "deployment_ready": False,
    }


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MPCDARRuntimeQualificationHarnessFailure(
            f"S10 config is unreadable: {target}"
        ) from exc
    if payload != _config_payload():
        raise MPCDARRuntimeQualificationHarnessFailure("S10 config contract is not locked")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    if spec.payload != _config_payload():
        raise MPCDARRuntimeQualificationHarnessFailure("S10 config contract is not locked")


def _load_review(path: str | Path, spec: Spec) -> Path:
    del spec
    target = _resolve(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MPCDARRuntimeQualificationHarnessFailure(
            f"S10 review contract is unreadable: {target}"
        ) from exc
    if payload != _review_payload():
        raise MPCDARRuntimeQualificationHarnessFailure("S10 review contract is not locked")
    return target


def _read_json_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected:
        raise MPCDARRuntimeQualificationHarnessFailure(
            f"{label} is missing, stale, or tampered"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MPCDARRuntimeQualificationHarnessFailure(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise MPCDARRuntimeQualificationHarnessFailure(f"{label} is not a mapping")
    return payload


def _verify_parent_evidence() -> dict[str, Any]:
    s9v_path = _resolve(S9V_MANIFEST)
    s9v = _read_json_hash(s9v_path, S9V_MANIFEST_SHA256, "S9V manifest")
    if (
        s9v.get("task_id") != "H-LOMA-CBOGT-S9V-FULL-PATH-VIS"
        or s9v.get("status") != "completed"
        or s9v.get("classification") != "descriptive_visualization_completed"
        or s9v.get("forward_consistency_max_abs") != 0
    ):
        raise MPCDARRuntimeQualificationHarnessFailure("S9V terminal identity changed")

    evidence = {}
    for name, relative, digest in (
        ("validation", S9_VALIDATION, S9_VALIDATION_SHA256),
        ("audit", S9_AUDIT, S9_AUDIT_SHA256),
        ("compact", S9_COMPACT, S9_COMPACT_SHA256),
    ):
        evidence[name] = _read_json_hash(_resolve(relative), digest, f"S9 {name}")
    if any(item.get("status") != "passed" for item in evidence.values()):
        raise MPCDARRuntimeQualificationHarnessFailure("S9 evidence is not passed")
    if evidence["compact"].get("classification") != "frozen_mean_preserving_data_aware_rank16_float32_numerically_qualified":
        raise MPCDARRuntimeQualificationHarnessFailure("S9 compact classification changed")

    artifact_dir = _resolve(S9_ARTIFACT_DIR)
    manifest_path = artifact_dir / "manifest.json"
    payload_path = artifact_dir / "payload.pt"
    if file_sha256(manifest_path) != S9_ARTIFACT_MANIFEST_SHA256:
        raise MPCDARRuntimeQualificationHarnessFailure("S9 artifact manifest identity changed")
    if file_sha256(payload_path) != S9_ARTIFACT_PAYLOAD_SHA256:
        raise MPCDARRuntimeQualificationHarnessFailure("S9 artifact payload identity changed")
    try:
        artifact = runtime.load_frozen_mpcdar_rank16_float32_readout(manifest_path)
    except (OSError, ValueError) as exc:
        raise MPCDARRuntimeQualificationHarnessFailure(
            "S9 float32 artifact cannot be loaded"
        ) from exc
    manifest = artifact.manifest
    if (
        manifest.get("repair_id") != runtime.REPAIR_ID
        or manifest.get("rank") != RANK
        or manifest.get("dtype") != "torch.float32"
        or manifest.get("feature_dim") != FEATURE_DIM
        or manifest.get("output_dim") != OUTPUT_DIM
        or manifest.get("raw_float32_tensor_bytes") != RAW_FLOAT32_TENSOR_BYTES
        or manifest.get("serialized_bytes") != SERIALIZED_BYTES
        or manifest.get("formula_version") != runtime.FORMULA_VERSION
    ):
        raise MPCDARRuntimeQualificationHarnessFailure("S9 artifact scalar contract changed")
    tensors = {
        "mu_fit": artifact.mu_fit,
        "c_mean": artifact.c_mean,
        "T16": artifact.T16,
        "V16": artifact.V16,
    }
    expected_shapes = {
        "mu_fit": (256,),
        "c_mean": (40000,),
        "T16": (256, 16),
        "V16": (40000, 16),
    }
    for name, value in tensors.items():
        if (
            value.dtype != torch.float32
            or value.device.type != "cpu"
            or tuple(value.shape) != expected_shapes[name]
            or not bool(torch.isfinite(value).all())
            or tensor_sha256(value) != manifest["tensors"][name]["tensor_sha256"]
        ):
            raise MPCDARRuntimeQualificationHarnessFailure(
                f"S9 tensor identity is invalid for {name}"
            )
    return {
        "s9v_manifest_path": s9v_path,
        "s9v_manifest": s9v,
        "s9_evidence": evidence,
        "artifact_dir": artifact_dir,
        "manifest_path": manifest_path,
        "payload_path": payload_path,
        "artifact": artifact,
    }


def _runtime_source_contract() -> dict[str, Any]:
    source = inspect.getsource(runtime)
    forbidden = {
        "float64_upcast": (".to(torch.float64)", ".double("),
        "research_dependency": ("application.research", "from ..research"),
        "factorization_or_search": ("torch.linalg", "torch.svd", "torch.eigh"),
        "training_or_optimizer": ("torch.optim", ".backward(", "optimizer"),
        "full_w_or_refit_dependency": ("C_fit", "W64", "load W", "probes.pt"),
    }
    found = {
        name: [token for token in tokens if token in source]
        for name, tokens in forbidden.items()
    }
    formula_tokens = (
        "centered = z32 - mu_fit",
        "hidden = centered.mm(T16)",
        "prediction = c_mean + hidden.mm(V16.T)",
    )
    return {
        "source_path": "src/ldm_project/v2/application/current/frozen_mpcdar_rank16_float32_readout.py",
        "formula_tokens_present": all(token in source for token in formula_tokens),
        "forbidden_tokens_found": found,
        "pure_float32_frozen_runtime": all(not values for values in found.values()),
    }


def _environment() -> dict[str, Any]:
    python_version = platform.python_version()
    torch_version = torch.__version__.split("+", 1)[0]
    os_version = platform.version()
    processor = platform.processor().strip() or "not_recorded"
    logical = os.cpu_count()
    windows_version_match = os_version.startswith(EXPECTED_WINDOWS_VERSION_PREFIX)
    registered_runtime_match = bool(
        Path(sys.executable).resolve() == Path(PYTHON_EXECUTABLE).resolve()
        and python_version == EXPECTED_PYTHON_VERSION
        and torch_version == EXPECTED_TORCH_VERSION
        and sys.platform == "win32"
        and torch.get_num_threads() == 8
        and torch.get_num_interop_threads() == 8
    )
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": python_version,
        "os": platform.platform(),
        "os_version": os_version,
        "cpu_model": processor,
        "logical_cores": logical if logical is not None else "not_recorded",
        "physical_cores": "not_recorded",
        "ram_bytes": "not_recorded",
        "pytorch_version": torch_version,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "expected_windows_version": EXPECTED_WINDOWS_VERSION,
        "observed_windows_version": os_version,
        "windows_version_match": windows_version_match,
        "registered_runtime_match": registered_runtime_match,
        "gpu": "not_evaluated",
        "cuda_benchmarked": False,
        "registered_match": bool(
            registered_runtime_match and windows_version_match
        ),
    }


def _make_fixture(seed: int = FIXTURE_SEED) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fixture = torch.randn(
        (32, FEATURE_DIM), generator=generator, dtype=torch.float32, device="cpu"
    )
    if (
        fixture.shape != (32, FEATURE_DIM)
        or fixture.dtype != torch.float32
        or fixture.device.type != "cpu"
        or not bool(torch.isfinite(fixture).all())
        or int(torch.count_nonzero(fixture)) != fixture.numel()
        or bool(
            ((fixture != 0) & (fixture.abs() < torch.finfo(torch.float32).tiny)).any()
        )
    ):
        raise MPCDARRuntimeQualificationHarnessFailure(
            "deterministic benchmark fixture is not dense finite float32"
        )
    return fixture.contiguous()


def _fixture_summary(fixture: torch.Tensor) -> dict[str, Any]:
    values = fixture.detach().cpu().reshape(-1).to(torch.float64)
    return {
        "seed": FIXTURE_SEED,
        "shape": list(fixture.shape),
        "dtype": str(fixture.dtype),
        "device": fixture.device.type,
        "sha256": tensor_sha256(fixture),
        "finite": bool(torch.isfinite(fixture).all()),
        "dense": int(torch.count_nonzero(fixture)) == fixture.numel(),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
    }


def _direct_formula(reader: runtime.FrozenMPCDARRank16Float32Readout, z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        centered = z - reader.mu_fit
        hidden = centered.mm(reader.T16)
        return reader.c_mean + hidden.mm(reader.V16.T)


def _correctness(reader: runtime.FrozenMPCDARRank16Float32Readout, fixture: torch.Tensor) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for batch_size in BATCH_SIZES:
        z = fixture[:batch_size]
        actual = reader.predict(z)
        expected = _direct_formula(reader, z)
        repeat = reader.predict(z)
        difference = (actual - expected).abs()
        denominator = max(float(expected.norm()), 1.0e-30)
        normalized_l2 = float((actual - expected).norm() / denominator)
        passed = bool(
            actual.shape == (batch_size, OUTPUT_DIM)
            and actual.dtype == torch.float32
            and bool(torch.isfinite(actual).all())
            and torch.allclose(
                actual,
                expected,
                atol=runtime.FLOAT32_PREDICTION_ATOL,
                rtol=runtime.FLOAT32_PREDICTION_RTOL,
            )
            and torch.equal(actual, repeat)
        )
        if not passed:
            raise MPCDARRuntimeQualificationHarnessFailure(
                f"S9 runtime correctness failed for batch {batch_size}"
            )
        rows[str(batch_size)] = {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "max_abs_difference": float(difference.max()),
            "normalized_l2_difference": normalized_l2,
            "direct_formula_match": True,
            "repeated_call_deterministic": True,
        }
    return {"passed": True, "batches": rows}


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise MPCDARRuntimeQualificationHarnessFailure("latency samples are empty or non-finite")
    numeric = [float(value) for value in values]
    return {
        "sample_count": len(numeric),
        "mean_ms": sum(numeric) / len(numeric),
        "median_ms": _linear_quantile(numeric, 0.50),
        "p95_ms": _linear_quantile(numeric, 0.95),
        "p99_ms": _linear_quantile(numeric, 0.99),
        "min_ms": min(numeric),
        "max_ms": max(numeric),
        "quantile_method": QUANTILE_METHOD,
    }


def _benchmark_batch(
    reader: runtime.FrozenMPCDARRank16Float32Readout,
    fixture: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    inputs = fixture[:batch_size]
    sink = 0.0
    for _ in range(WARMUP_COUNT):
        output = reader.predict(inputs)
        sink += float(output.reshape(-1)[0].item())
        del output
    samples: list[float] = []
    for _ in range(MEASURED_REPETITIONS):
        start = time.perf_counter_ns()
        output = reader.predict(inputs)
        stop = time.perf_counter_ns()
        samples.append((stop - start) / 1.0e6)
        sink += float(output.reshape(-1)[0].item())
        del output
    batch_stats = _summary(samples)
    per_sample = [value / batch_size for value in samples]
    throughput = [batch_size / (value / 1000.0) for value in samples]
    return {
        "batch_size": batch_size,
        "input_shape": list(inputs.shape),
        "batch_latency_ms": batch_stats,
        "per_sample_latency_ms": _summary(per_sample),
        "throughput_samples_per_second": _summary(throughput),
        "latency_samples_saved": False,
        "prediction_bank_saved": False,
        "sink_finite": math.isfinite(sink),
        "timer": TIMER,
        "synchronization": "CPU eager synchronous; no CUDA synchronization",
        "warmup": WARMUP_COUNT,
        "repetitions": MEASURED_REPETITIONS,
    }


def _fresh_process_initialization(manifest_path: Path) -> dict[str, Any]:
    manifest_literal = repr(str(manifest_path.resolve()))
    script = textwrap.dedent(
        f"""
        import json
        import time
        from importlib import import_module
        from pathlib import Path

        safe_torch_load = import_module("ldm_project.v2.evidence").safe_torch_load
        load_frozen_mpcdar_rank16_float32_readout = import_module(
            "ldm_project.v2.application.current.frozen_mpcdar_rank16_float32_readout"
        ).load_frozen_mpcdar_rank16_float32_readout

        manifest_path = Path({manifest_literal})
        total_start = time.perf_counter_ns()
        manifest_start = time.perf_counter_ns()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_stop = time.perf_counter_ns()
        payload_path = manifest_path.parent / manifest["payload_file"]
        payload_start = time.perf_counter_ns()
        safe_torch_load(payload_path, expected_sha256=manifest["payload_file_sha256"], map_location="cpu")
        payload_stop = time.perf_counter_ns()
        consumer_start = time.perf_counter_ns()
        load_frozen_mpcdar_rank16_float32_readout(manifest_path)
        consumer_stop = time.perf_counter_ns()
        total_stop = time.perf_counter_ns()
        def milliseconds(start, stop):
            return (stop - start) / 1.0e6
        print(json.dumps({{
            "manifest_parse_load_ms": milliseconds(manifest_start, manifest_stop),
            "payload_load_ms": milliseconds(payload_start, payload_stop),
            "consumer_initialization_ms": milliseconds(consumer_start, consumer_stop),
            "total_artifact_initialization_ms": milliseconds(total_start, total_stop),
        }}))
        """
    )
    started = time.perf_counter_ns()
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
    if result.returncode != 0:
        raise MPCDARRuntimeQualificationHarnessFailure(
            f"fresh-process initialization failed: {result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MPCDARRuntimeQualificationHarnessFailure(
            "fresh-process initialization did not return JSON telemetry"
        ) from exc
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(key), (int, float))
        for key in (
            "manifest_parse_load_ms",
            "payload_load_ms",
            "consumer_initialization_ms",
            "total_artifact_initialization_ms",
        )
    ):
        raise MPCDARRuntimeQualificationHarnessFailure(
            "fresh-process initialization telemetry is incomplete"
        )
    payload["fresh_process_wall_ms"] = elapsed_ms
    payload["terminology"] = "fresh_process_initialization"
    payload["filesystem_cache_flush"] = False
    payload["true_cold_disk_latency_claim"] = False
    payload["stage_timers_may_overlap"] = True
    return payload


def _storage(reader: runtime.FrozenMPCDARRank16Float32Readout, manifest_path: Path, payload_path: Path) -> dict[str, Any]:
    tensor_bytes = sum(
        value.numel() * value.element_size()
        for value in (reader.mu_fit, reader.c_mean, reader.T16, reader.V16)
    )
    serialized = payload_path.stat().st_size
    if serialized != SERIALIZED_BYTES or tensor_bytes != RAW_FLOAT32_TENSOR_BYTES:
        raise MPCDARRuntimeQualificationHarnessFailure("S9 storage telemetry changed")
    return {
        "manifest_bytes": manifest_path.stat().st_size,
        "serialized_artifact_bytes": serialized,
        "raw_tensor_bytes": RAW_FLOAT32_TENSOR_BYTES,
        "runtime_tensor_bytes": tensor_bytes,
        "process_memory_delta": "not_recorded",
        "peak_memory": "not_recorded",
        "rss_claim": False,
    }


def _artifact_snapshot(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_sha256": file_sha256(parent["manifest_path"]),
        "payload_sha256": file_sha256(parent["payload_path"]),
        "tensor_sha256": {
            name: tensor_sha256(getattr(parent["artifact"], name))
            for name in runtime.PAYLOAD_KEYS
        },
    }


def _context(spec: Spec) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "config": spec.payload,
            "review": _review_payload(),
            "artifact_contract": runtime.MANIFEST_SCHEMA,
        }
    )
    return EvidenceContextV2(signature, spec.config_hash)


def _input_hashes(spec: Spec, review_path: Path, parent: Mapping[str, Any], fixture: torch.Tensor | None = None) -> dict[str, str]:
    hashes = {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "s9v_manifest": file_sha256(parent["s9v_manifest_path"]),
        "s9_manifest": file_sha256(parent["manifest_path"]),
        "s9_payload": file_sha256(parent["payload_path"]),
    }
    if fixture is not None:
        hashes["deterministic_fixture"] = tensor_sha256(fixture)
    return hashes


def _evidence(
    schema: str,
    spec: Spec,
    state: Mapping[str, Any],
    review_path: Path,
    parent: Mapping[str, Any],
    gates: Sequence[GateResult],
    contract: Mapping[str, Any],
    classification: str | None,
    extra_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    hashes = _input_hashes(spec, review_path, parent)
    if extra_hashes:
        hashes.update(extra_hashes)
    record = EvidenceRecordV2(
        2,
        schema,
        "diagnostic",
        "passed",
        str(state["commit"]),
        None,
        spec.config_hash,
        _context(spec),
        hashes,
        tuple(gates),
        (NEXT_ACTION_RESULT_REVIEW,),
        None,
    )
    record.validate()
    return {
        "hypothesis_id": HYPOTHESIS,
        "schema": schema,
        "hypothesis": HYPOTHESIS,
        "status": "passed",
        "classification": classification,
        "diagnostic_only": True,
        "runtime_performance_characterization_only": True,
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
        "generation": False,
        "optimizer_created": False,
        "backward_calls": 0,
        "probe_refit": False,
        "bias_refit": False,
        "decoder_created": False,
        "checkpoint_mutation": False,
        "model_loaded": False,
        "cuda_used": False,
        "physical_forward_called": False,
        "prediction_bank_saved": False,
        "large_trace_saved": False,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _gate(name: str, passed: bool, **metrics: float | bool) -> GateResult:
    return GateResult(name, bool(passed), metrics)


def _static_check(
    spec: Spec,
    review_path: Path,
    state: Mapping[str, Any],
    parent: Mapping[str, Any],
    artifact_dir: Path,
    output_path: Path,
) -> tuple[tuple[GateResult, ...], dict[str, Any], torch.Tensor]:
    environment = _environment()
    source_contract = _runtime_source_contract()
    fixture = _make_fixture()
    artifact_path_ok = artifact_dir.resolve() == parent["artifact_dir"].resolve()
    output_fresh = not output_path.exists()
    gates = (
        _gate("clean_registered_branch", state.get("branch") == BRANCH and state.get("clean") is True),
        _gate("s10_starting_head_ancestor", True),
        _gate("config_and_review_contract", spec.payload == _config_payload() and review_path.is_file()),
        _gate("s9v_completed_identity", parent["s9v_manifest"].get("status") == "completed"),
        _gate("s9_artifact_identity", artifact_path_ok and file_sha256(parent["manifest_path"]) == S9_ARTIFACT_MANIFEST_SHA256 and file_sha256(parent["payload_path"]) == S9_ARTIFACT_PAYLOAD_SHA256),
        _gate("s9_rank16_float32_contract", parent["artifact"].manifest.get("rank") == RANK and parent["artifact"].manifest.get("dtype") == "torch.float32"),
        _gate("s9_runtime_formula_contract", source_contract["formula_tokens_present"] and source_contract["pure_float32_frozen_runtime"]),
        _gate("registered_environment", environment["registered_runtime_match"]),
        _gate("deterministic_dense_fixture", fixture.shape == (32, FEATURE_DIM) and fixture.dtype == torch.float32 and bool(torch.isfinite(fixture).all()) and int(torch.count_nonzero(fixture)) == fixture.numel()),
        _gate("fixed_cpu_benchmark_contract", tuple(BATCH_SIZES) == (1, 8, 32) and WARMUP_COUNT == 50 and MEASURED_REPETITIONS == 200 and TIMER == "time.perf_counter_ns"),
        _gate("fresh_output_slot", output_fresh),
        _gate("no_update_scope", True),
    )
    details = {
        "environment": environment,
        "source_contract": source_contract,
        "fixture": _fixture_summary(fixture),
        "output_path": str(output_path),
        "artifact_path_locked": artifact_path_ok,
    }
    return gates, details, fixture


def _write_new(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=kind)


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_registered_branch(state)
    parent = _verify_parent_evidence()
    artifact_dir = _resolve(args.artifact_dir)
    output_path = _resolve(args.output)
    if output_path.name != "validation.json":
        raise MPCDARRuntimeQualificationHarnessFailure("S10 validation output must be validation.json")
    gates, details, fixture = _static_check(spec, review_path, state, parent, artifact_dir, output_path)
    if not all(gate.passed for gate in gates):
        raise MPCDARRuntimeQualificationHarnessFailure("S10 static validation failed")
    payload = _evidence(
        VALIDATION_SCHEMA,
        spec,
        state,
        review_path,
        parent,
        gates,
        {"phase": "static_contract_validation", "details": details},
        "frozen_rank16_float32_runtime_performance_contract_validated",
        {"deterministic_fixture": tensor_sha256(fixture)},
    )
    payload["git"] = dict(state)
    payload["starting_head"] = S10_STARTING_HEAD
    payload["parent"] = {
        "s9v_manifest_sha256": S9V_MANIFEST_SHA256,
        "s9_manifest_sha256": S9_ARTIFACT_MANIFEST_SHA256,
        "s9_payload_sha256": S9_ARTIFACT_PAYLOAD_SHA256,
    }
    _write_new(output_path, payload, "S10 validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _require_validation(path: Path, spec: Spec, state: Mapping[str, Any]) -> None:
    validation = _read_json_hash(path, file_sha256(path), "S10 validation evidence")
    if (
        validation.get("status") != "passed"
        or validation.get("classification") != "frozen_rank16_float32_runtime_performance_contract_validated"
        or validation.get("evidence", {}).get("git_commit") != state.get("commit")
        or validation.get("evidence", {}).get("config_hash") != spec.config_hash
    ):
        raise MPCDARRuntimeQualificationHarnessFailure(
            "S10 run requires passed validation from the current commit and config"
        )


def run(args: argparse.Namespace) -> int:
    if args.device != "cpu":
        raise MPCDARRuntimeQualificationHarnessFailure("S10 primary benchmark is CPU-only")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_registered_branch(state)
    parent = _verify_parent_evidence()
    artifact_dir = _resolve(args.artifact_dir)
    output_path = _resolve(args.output)
    validation_path = _resolve(args.validation_evidence)
    if output_path.name != "audit.json" or output_path.exists():
        raise MPCDARRuntimeQualificationHarnessFailure("S10 audit output slot is not fresh")
    compact_path = output_path.parent / "compact_report.json"
    if compact_path.exists():
        raise MPCDARRuntimeQualificationHarnessFailure("S10 compact output slot is not fresh")
    _require_validation(validation_path, spec, state)
    static_gates, static_details, fixture = _static_check(
        spec, review_path, state, parent, artifact_dir, output_path
    )
    if not all(gate.passed for gate in static_gates):
        raise MPCDARRuntimeQualificationHarnessFailure("S10 static validation failed before timing")
    reader = parent["artifact"]
    snapshot_before = _artifact_snapshot(parent)
    correctness = _correctness(reader, fixture)
    fresh_initialization = _fresh_process_initialization(parent["manifest_path"])
    batches = {
        str(batch_size): _benchmark_batch(reader, fixture, batch_size)
        for batch_size in BATCH_SIZES
    }
    snapshot_after = _artifact_snapshot(parent)
    if snapshot_before != snapshot_after:
        raise MPCDARRuntimeQualificationHarnessFailure("S9 artifact changed during benchmark")
    storage = _storage(reader, parent["manifest_path"], parent["payload_path"])
    runtime_gates = (
        _gate("correctness_before_timing", correctness["passed"]),
        _gate("fresh_process_initialization_recorded", all(key in fresh_initialization for key in ("manifest_parse_load_ms", "payload_load_ms", "consumer_initialization_ms", "total_artifact_initialization_ms"))),
        _gate("cpu_primary_required", len(batches) == len(BATCH_SIZES)),
        _gate("fixed_batch_set", tuple(int(key) for key in batches) == BATCH_SIZES),
        _gate("warmup_and_repetition_contract", all(row["warmup"] == WARMUP_COUNT and row["repetitions"] == MEASURED_REPETITIONS for row in batches.values())),
        _gate("latency_statistics_complete", all(row["batch_latency_ms"]["sample_count"] == MEASURED_REPETITIONS and row["per_sample_latency_ms"]["sample_count"] == MEASURED_REPETITIONS and row["throughput_samples_per_second"]["sample_count"] == MEASURED_REPETITIONS for row in batches.values())),
        _gate("no_best_only_reporting", all("median_ms" in row["batch_latency_ms"] and "p95_ms" in row["batch_latency_ms"] and "p99_ms" in row["batch_latency_ms"] for row in batches.values())),
        _gate("throughput_definition", all(row["throughput_samples_per_second"]["sample_count"] == MEASURED_REPETITIONS for row in batches.values())),
        _gate("no_prediction_bank", all(row["prediction_bank_saved"] is False and row["latency_samples_saved"] is False for row in batches.values())),
        _gate("storage_telemetry", storage["serialized_artifact_bytes"] == SERIALIZED_BYTES and storage["raw_tensor_bytes"] == RAW_FLOAT32_TENSOR_BYTES and storage["runtime_tensor_bytes"] == RAW_FLOAT32_TENSOR_BYTES),
        _gate("no_unsupported_rss_claim", storage["rss_claim"] is False),
        _gate("artifact_immutable", snapshot_before == snapshot_after),
        _gate("optional_paths_not_evaluated", True),
        _gate("state_and_readiness_unchanged", True),
    )
    all_gates = tuple(static_gates) + tuple(runtime_gates)
    if not all(gate.passed for gate in all_gates):
        raise MPCDARRuntimeQualificationHarnessFailure("S10 runtime characterization gate failed")
    contract = {
        "config": spec.payload,
        "review": _review_payload(),
        "starting_head": S10_STARTING_HEAD,
        "git": dict(state),
        "parent": {
            "s9v_manifest_sha256": S9V_MANIFEST_SHA256,
            "s9_manifest_sha256": S9_ARTIFACT_MANIFEST_SHA256,
            "s9_payload_sha256": S9_ARTIFACT_PAYLOAD_SHA256,
        },
        "static": static_details,
        "environment": static_details["environment"],
        "fixture": _fixture_summary(fixture),
        "correctness": correctness,
        "fresh_process_initialization": fresh_initialization,
        "steady_state": {
            "batch_sizes": list(BATCH_SIZES),
            "warmup": WARMUP_COUNT,
            "repetitions": MEASURED_REPETITIONS,
            "timer": TIMER,
            "quantile_method": QUANTILE_METHOD,
            "batches": batches,
        },
        "storage": storage,
        "analytical_mac_comparator": {
            "kind": "analytical_operation_count_comparator",
            "factorized_rank16_macs_per_sample": FACTORIZED_MACS_PER_SAMPLE,
            "full_w_macs_per_sample": FULL_W_MACS_PER_SAMPLE,
            "ratio": MAC_RATIO,
            "measured_speedup_claim": False,
        },
        "full_w_runtime": "not_evaluated",
        "gpu": "not_evaluated",
        "end_to_end_software_pipeline": "not_evaluated",
        "prediction_bank_saved": False,
        "large_trace_saved": False,
    }
    hashes = _input_hashes(spec, review_path, parent, fixture)
    hashes["validation_evidence"] = file_sha256(validation_path)
    audit = _evidence(
        AUDIT_SCHEMA,
        spec,
        state,
        review_path,
        parent,
        all_gates,
        contract,
        CLASSIFICATION,
        hashes,
    )
    audit["git"] = dict(state)
    audit["starting_head"] = S10_STARTING_HEAD
    audit["audit_gate_count"] = len(all_gates)
    audit["audit_gate_passed"] = sum(1 for gate in all_gates if gate.passed)
    _write_new(output_path, audit, "S10 audit evidence")
    compact = {
        "schema": COMPACT_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "status": "passed",
        "classification": CLASSIFICATION,
        "starting_head": S10_STARTING_HEAD,
        "git": dict(state),
        "validation_sha256": file_sha256(validation_path),
        "audit_sha256": file_sha256(output_path),
        "parent": contract["parent"],
        "environment": contract["environment"],
        "fixture": contract["fixture"],
        "correctness": correctness,
        "fresh_process_initialization": fresh_initialization,
        "steady_state": contract["steady_state"],
        "storage": storage,
        "analytical_mac_comparator": contract["analytical_mac_comparator"],
        "full_w_runtime": "not_evaluated",
        "gpu": "not_evaluated",
        "end_to_end_software_pipeline": "not_evaluated",
        "process_memory_delta": "not_recorded",
        "peak_memory": "not_recorded",
        "prediction_bank_saved": False,
        "large_trace_saved": False,
        "audit_gates": {"passed": len(all_gates), "total": len(all_gates)},
        "next_action": NEXT_ACTION_RESULT_REVIEW,
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }
    _write_new(compact_path, compact, "S10 compact report")
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-mpcdar-rank16-runtime-performance-qualification-validate":
        return validate(args)
    if args.research_action == "scaled-chen-mpcdar-rank16-runtime-performance-qualification-run":
        return run(args)
    raise ValueError(f"unsupported S10 action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "BATCH_SIZES",
    "CLASSIFICATION",
    "COMPACT_SCHEMA",
    "CONFIG_SCHEMA",
    "FACTORIZED_MACS_PER_SAMPLE",
    "FULL_W_MACS_PER_SAMPLE",
    "HYPOTHESIS",
    "MEASURED_REPETITIONS",
    "MPCDARRuntimeQualificationHarnessFailure",
    "NEXT_ACTION_RESULT_REVIEW",
    "RAW_FLOAT32_TENSOR_BYTES",
    "RANK",
    "REVIEW_SCHEMA",
    "S10_STARTING_HEAD",
    "SERIALIZED_BYTES",
    "VALIDATION_SCHEMA",
    "WARMUP_COUNT",
    "_config_payload",
    "_correctness",
    "_environment",
    "_fresh_process_initialization",
    "_linear_quantile",
    "_load_review",
    "_load_spec",
    "_make_fixture",
    "_review_payload",
    "_runtime_source_contract",
    "_summary",
    "dispatch",
    "run",
    "validate",
]
