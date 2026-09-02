"""Self-contained frozen MPCDAR rank-16 float32 readout.

This is the deliberately narrow S9 consumer.  It accepts an existing float32
CMOS feature tensor and a separately published float32 artifact.  It does not
construct, fit, or discover a readout and it keeps the S8 float64 consumer
unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ...evidence import file_sha256, safe_torch_load, tensor_sha256


REPAIR_ID = "H-LOMA-CBOGT-S9-MPCDAR-R16-F32-NQ"
PARENT_REPAIR_ID = "H-LOMA-CBOGT-S8-MPCDAR-R16-IQ"
MANIFEST_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_float32_numerical_qualification_artifact.v1"
)
PAYLOAD_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_float32_frozen_payload.v1"
)
FORMULA_VERSION = "centered_mean_preserving_data_aware_rank16_factorized_v1"
CAST_RULE = "source_tensor.to(torch.float32) on CPU, one cast per tensor"
FEATURE_CONTRACT = "output.camera -> mra._feature_projection(real, complex_value=False)"
FEATURE_CONTRACT_VERSION = "mra._feature_projection.real.adaptive_avg_pool_16x16.v1"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
RANK = 16
FEATURE_DIM = 256
OUTPUT_DIM = 40000
TARGET_SHAPE = (200, 200)
SCALAR_COUNT_TOTAL = 684352
RAW_FLOAT32_TENSOR_BYTES = 2737408
PAYLOAD_KEYS = frozenset({"mu_fit", "c_mean", "T16", "V16"})
TENSOR_SHAPES = {
    "mu_fit": (FEATURE_DIM,),
    "c_mean": (OUTPUT_DIM,),
    "T16": (FEATURE_DIM, RANK),
    "V16": (OUTPUT_DIM, RANK),
}

PARENT_MANIFEST_SHA256 = (
    "e580dd59f3e277055a9c0cff6d9f076e06346b5761a8b9f228608e88b3a23c45"
)
PARENT_PAYLOAD_SHA256 = (
    "e839daaff8d718347d1cd4194651ac0773f842ffa39fe4efaa3e0218ea2e1d4e"
)
PARENT_CREATED_COMMIT = "c0140577fddb56af79d72a6d473cc49ef04f5ffc"
PARENT_MANIFEST_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification_artifact.v1"
)
PARENT_PAYLOAD_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_frozen_payload.v1"
)
PARENT_TENSOR_SHA256 = {
    "mu_fit": "187bb9a64c2142dcb7eb79325f3720bc8d86a6c854c9ddcc4725add4c4069496",
    "c_mean": "56c28fe8e94167c045c0d68d67ed484c4350c89426f93becd960b1cf75c054bc",
    "T16": "31c71923c1580003e5db8ef0df2241452737baac1a5f37b7e419094a83cb4c29",
    "V16": "8fda4bbd4dffaa676806bb0f32717e44b42e3131f6f5cb2af41c50f6ea63a30e",
}

FLOAT32_PREDICTION_ATOL = 2.0e-4
FLOAT32_PREDICTION_RTOL = 2.0e-4
FLOAT32_NORMALIZED_L2_MAX = 2.0e-4
FLOAT32_PROBE_MSE_ABSOLUTE_DRIFT_MAX = 5.0e-6
FLOAT32_PROBE_MSE_RELATIVE_DRIFT_MAX = 5.0e-4


class FrozenMPCDARRank16Float32ArtifactError(ValueError):
    """The S9 artifact is missing, malformed, or no longer immutable."""


def _invalid(message: str) -> FrozenMPCDARRank16Float32ArtifactError:
    return FrozenMPCDARRank16Float32ArtifactError(message)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise _invalid(f"{label} must be a SHA-256 hex string")
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise _invalid(f"{label} must be a SHA-256 hex string")
    return lowered


def _git_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise _invalid(f"{label} must be a Git commit hex string")
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise _invalid(f"{label} must be a Git commit hex string")
    return lowered


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _invalid(f"{label} must be a mapping")
    return value


def _parent_contract() -> dict[str, Any]:
    return {
        "repair_id": PARENT_REPAIR_ID,
        "manifest_schema": PARENT_MANIFEST_SCHEMA,
        "payload_schema": PARENT_PAYLOAD_SCHEMA,
        "dtype": "torch.float64",
        "rank": RANK,
        "feature_dim": FEATURE_DIM,
        "output_dim": OUTPUT_DIM,
        "scalar_count_total": SCALAR_COUNT_TOTAL,
        "raw_float64_tensor_bytes": 5474816,
        "manifest_sha256": PARENT_MANIFEST_SHA256,
        "payload_sha256": PARENT_PAYLOAD_SHA256,
        "created_commit": PARENT_CREATED_COMMIT,
        "tensor_sha256": dict(PARENT_TENSOR_SHA256),
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "payload_schema",
        "repair_id",
        "parent",
        "rank",
        "feature_dim",
        "output_dim",
        "target_shape",
        "formula_version",
        "dtype",
        "device_storage",
        "feature_contract",
        "feature_contract_version",
        "cast_rule",
        "source_dtype",
        "target_dtype",
        "parent_artifact",
        "tensor_count",
        "scalar_count_total",
        "raw_float32_tensor_bytes",
        "tensors",
        "payload_file",
        "payload_file_sha256",
        "serialized_bytes",
        "created_git",
    }
    if set(manifest) != required:
        raise _invalid("float32 artifact manifest fields are invalid")
    expected_scalars = {
        "schema_version": MANIFEST_SCHEMA,
        "payload_schema": PAYLOAD_SCHEMA,
        "repair_id": REPAIR_ID,
        "parent": PARENT_REPAIR_ID,
        "rank": RANK,
        "feature_dim": FEATURE_DIM,
        "output_dim": OUTPUT_DIM,
        "target_shape": list(TARGET_SHAPE),
        "formula_version": FORMULA_VERSION,
        "dtype": "torch.float32",
        "device_storage": "cpu",
        "feature_contract": FEATURE_CONTRACT,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "cast_rule": CAST_RULE,
        "source_dtype": "torch.float64",
        "target_dtype": "torch.float32",
        "tensor_count": len(PAYLOAD_KEYS),
        "scalar_count_total": SCALAR_COUNT_TOTAL,
        "raw_float32_tensor_bytes": RAW_FLOAT32_TENSOR_BYTES,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise _invalid(f"float32 artifact manifest field {key} is mismatched")
    if dict(_require_mapping(manifest["parent_artifact"], "parent_artifact")) != _parent_contract():
        raise _invalid("S8 parent artifact identity is mismatched")
    parent_artifact = _require_mapping(manifest["parent_artifact"], "parent_artifact")
    _sha256(parent_artifact["manifest_sha256"], "parent_artifact.manifest_sha256")
    _sha256(parent_artifact["payload_sha256"], "parent_artifact.payload_sha256")
    for name, value in PARENT_TENSOR_SHA256.items():
        _sha256(parent_artifact["tensor_sha256"][name], f"parent_artifact.tensor_sha256.{name}")
        if parent_artifact["tensor_sha256"][name] != value:
            raise _invalid(f"parent artifact tensor hash is mismatched for {name}")
    tensors = _require_mapping(manifest["tensors"], "tensors")
    if set(tensors) != PAYLOAD_KEYS:
        raise _invalid("float32 artifact tensor entries are invalid")
    for name in PAYLOAD_KEYS:
        entry = _require_mapping(tensors[name], f"tensors.{name}")
        if set(entry) != {"shape", "dtype", "tensor_sha256"}:
            raise _invalid(f"float32 artifact tensor entry {name} is invalid")
        if entry["shape"] != list(TENSOR_SHAPES[name]) or entry["dtype"] != "torch.float32":
            raise _invalid(f"float32 artifact tensor contract for {name} is mismatched")
        _sha256(entry["tensor_sha256"], f"tensors.{name}.tensor_sha256")
    _sha256(manifest["payload_file_sha256"], "payload_file_sha256")
    if not isinstance(manifest["serialized_bytes"], int) or isinstance(manifest["serialized_bytes"], bool):
        raise _invalid("serialized_bytes must be an integer")
    created_git = _require_mapping(manifest["created_git"], "created_git")
    if set(created_git) != {"commit", "branch", "clean"}:
        raise _invalid("created_git fields are invalid")
    _git_commit(created_git["commit"], "created_git.commit")
    if created_git["branch"] != BRANCH or created_git["clean"] is not True:
        raise _invalid("float32 artifact was not created on the clean registered branch")


def _payload_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    value = manifest["payload_file"]
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise _invalid("payload_file must be a relative filename")
    relative = Path(value)
    if relative.name != value or ".." in relative.parts:
        raise _invalid("payload_file must remain beside manifest.json")
    return manifest_path.parent / relative


def _load_payload(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    payload_path = _payload_path(manifest_path, manifest)
    if not payload_path.is_file():
        raise _invalid(f"float32 artifact payload is missing: {payload_path}")
    expected_file_hash = _sha256(manifest["payload_file_sha256"], "payload_file_sha256")
    if file_sha256(payload_path) != expected_file_hash:
        raise _invalid("float32 artifact payload file hash mismatch")
    if payload_path.stat().st_size != manifest["serialized_bytes"]:
        raise _invalid("float32 artifact serialized byte count mismatch")
    try:
        payload = safe_torch_load(payload_path, expected_sha256=expected_file_hash, map_location="cpu")
    except (OSError, ValueError) as exc:
        raise _invalid("float32 artifact payload is not safely loadable") from exc
    if not isinstance(payload, Mapping) or set(payload) != PAYLOAD_KEYS:
        raise _invalid("float32 artifact payload keys are invalid")
    entries = _require_mapping(manifest["tensors"], "tensors")
    loaded: dict[str, torch.Tensor] = {}
    for name in PAYLOAD_KEYS:
        value = payload[name]
        if not torch.is_tensor(value):
            raise _invalid(f"float32 artifact payload entry {name} is not a tensor")
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise _invalid(f"float32 artifact payload entry {name} dtype/device is invalid")
        if tuple(value.shape) != TENSOR_SHAPES[name] or not bool(torch.isfinite(value).all()):
            raise _invalid(f"float32 artifact payload entry {name} shape or finite contract is invalid")
        if tensor_sha256(value) != entries[name]["tensor_sha256"]:
            raise _invalid(f"float32 artifact tensor hash mismatch for {name}")
        loaded[name] = value.detach().contiguous()
    return loaded


@dataclass(frozen=True)
class FrozenMPCDARRank16Float32Readout:
    """A read-only rank-16 float32 readout loaded from a validated artifact."""

    manifest_path: Path
    mu_fit: torch.Tensor
    c_mean: torch.Tensor
    T16: torch.Tensor
    V16: torch.Tensor
    manifest: Mapping[str, Any]

    @classmethod
    def from_manifest(cls, path: str | Path) -> "FrozenMPCDARRank16Float32Readout":
        manifest_path = Path(path).resolve()
        if not manifest_path.is_file():
            raise _invalid(f"float32 artifact manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise _invalid("float32 artifact manifest is unreadable") from exc
        manifest_mapping = _require_mapping(manifest, "manifest")
        _validate_manifest(manifest_mapping)
        payload = _load_payload(manifest_path, manifest_mapping)
        return cls(
            manifest_path,
            payload["mu_fit"],
            payload["c_mean"],
            payload["T16"],
            payload["V16"],
            manifest_mapping,
        )

    load = from_manifest

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the only registered pure-float32 prediction formula."""
        if not torch.is_tensor(z) or z.ndim < 1 or z.shape[-1] != FEATURE_DIM:
            raise ValueError("float32 MPCDAR readout input must have shape [...,256]")
        if z.dtype != torch.float32 or not bool(torch.isfinite(z).all()):
            raise ValueError("float32 MPCDAR readout input must be finite torch.float32 features")
        leading_shape = tuple(z.shape[:-1])
        z32 = z.reshape(-1, FEATURE_DIM)
        device = z32.device
        mu_fit = self.mu_fit.to(device=device)
        c_mean = self.c_mean.to(device=device)
        T16 = self.T16.to(device=device)
        V16 = self.V16.to(device=device)
        with torch.no_grad():
            centered = z32 - mu_fit
            hidden = centered.mm(T16)
            prediction = c_mean + hidden.mm(V16.T)
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("float32 MPCDAR readout produced non-finite output")
        return prediction.reshape((*leading_shape, OUTPUT_DIM))

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.predict(z)


def load_frozen_mpcdar_rank16_float32_readout(
    path: str | Path,
) -> FrozenMPCDARRank16Float32Readout:
    """Load and validate an explicit frozen S9 float32 manifest."""
    return FrozenMPCDARRank16Float32Readout.from_manifest(path)


__all__ = [
    "BRANCH",
    "CAST_RULE",
    "FEATURE_CONTRACT",
    "FEATURE_CONTRACT_VERSION",
    "FLOAT32_NORMALIZED_L2_MAX",
    "FLOAT32_PREDICTION_ATOL",
    "FLOAT32_PREDICTION_RTOL",
    "FLOAT32_PROBE_MSE_ABSOLUTE_DRIFT_MAX",
    "FLOAT32_PROBE_MSE_RELATIVE_DRIFT_MAX",
    "FORMULA_VERSION",
    "FrozenMPCDARRank16Float32ArtifactError",
    "FrozenMPCDARRank16Float32Readout",
    "MANIFEST_SCHEMA",
    "OUTPUT_DIM",
    "PARENT_MANIFEST_SHA256",
    "PARENT_PAYLOAD_SHA256",
    "PARENT_REPAIR_ID",
    "PAYLOAD_KEYS",
    "PAYLOAD_SCHEMA",
    "RANK",
    "RAW_FLOAT32_TENSOR_BYTES",
    "REPAIR_ID",
    "SCALAR_COUNT_TOTAL",
    "TENSOR_SHAPES",
    "load_frozen_mpcdar_rank16_float32_readout",
]
