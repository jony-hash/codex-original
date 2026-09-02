"""Self-contained frozen MPCDAR rank-16 post-detector readout.

The module deliberately has no dependency on research producers, datasets, a
probe, or the construction-time covariance/basis.  The artifact manifest and
payload are the only external inputs to the loader; prediction consumes an
already projected detector feature tensor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ...evidence import file_sha256, safe_torch_load, tensor_sha256


REPAIR_ID = "H-LOMA-CBOGT-S8-MPCDAR-R16-IQ"
PARENT_ID = "H-LOMA-CBOGT-S7-MPCDAR_attempt-03"
MANIFEST_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification_artifact.v1"
)
PAYLOAD_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_frozen_payload.v1"
)
FORMULA_VERSION = "centered_mean_preserving_data_aware_rank16_factorized_v1"
FEATURE_CONTRACT = (
    "output.camera -> mra._feature_projection(real, complex_value=False)"
)
FEATURE_CONTRACT_VERSION = "mra._feature_projection.real.adaptive_avg_pool_16x16.v1"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
RANK = 16
FEATURE_DIM = 256
OUTPUT_DIM = 40000
TARGET_SHAPE = (200, 200)
SCALAR_COUNT_TOTAL = 684352
RAW_FLOAT64_BYTES = 5474816
PAYLOAD_KEYS = frozenset({"mu_fit", "c_mean", "T16", "V16"})
TENSOR_SHAPES = {
    "mu_fit": (FEATURE_DIM,),
    "c_mean": (OUTPUT_DIM,),
    "T16": (FEATURE_DIM, RANK),
    "V16": (OUTPUT_DIM, RANK),
}

PARENT_PRODUCER_COMMIT = "b59172750d0c88bbc9f7426ab7a21a1d63639f57"
PARENT_PRODUCER_MODULE_SHA256 = (
    "aaf2892651b6c22d9ee78d7deecb10fd47af1a18a481f7a3699735631afb7337"
)
PARENT_CONFIG_SHA256 = (
    "eee1cb9c26369e7226ae6b8588943ed3e83fa18a71013afa29fbfa0e89e5d4c3"
)
PARENT_REVIEW_SHA256 = (
    "108721c4db7c80dd56b7166bb32ef5b4ca2129f91bd76b60682c39e26a5cfcbd"
)
PROBE_SHA256 = "3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed"
MU_FIT_SHA256 = "187bb9a64c2142dcb7eb79325f3720bc8d86a6c854c9ddcc4725add4c4069496"
C_FIT_SHA256 = "be2ac0c63a6bcf290ee04fde074508d823c462368c9360078f99b9556b2b4d52"
FIT_PANEL_HASH = "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d"
PARENT_EVIDENCE_HASHES = {
    "validation": "b538b3ab242e09a0042699b1b1d0779d504f992498b1fcd75c23bae9937512d2",
    "audit": "43b4cef14202c19cc6aca22ec2c49cf5a65ed1ac05b7b69f7d00390de4b95785",
    "compact": "9e321b457e9e7111e1bf0d800e1d8e724f9753cf34f92e13d1eb2e081034a4e9",
}


class FrozenMPCDARRank16ArtifactError(ValueError):
    """The frozen artifact is missing, malformed, or no longer immutable."""


def _invalid(message: str) -> FrozenMPCDARRank16ArtifactError:
    return FrozenMPCDARRank16ArtifactError(message)


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
        "producer_commit",
        "producer_module_sha256",
        "producer_config_sha256",
        "producer_review_sha256",
        "parent_evidence_hashes",
        "probe_sha256",
        "fit_panel_identity",
        "fit_panel_hash",
        "mu_fit_parent_hash",
        "C_fit_parent_hash",
        "tensor_count",
        "scalar_count_total",
        "raw_float64_tensor_bytes",
        "tensors",
        "payload_file",
        "payload_file_sha256",
        "serialized_bytes",
        "created_git",
    }
    if set(manifest) != required:
        raise _invalid("frozen artifact manifest fields are invalid")
    expected_scalars = {
        "schema_version": MANIFEST_SCHEMA,
        "payload_schema": PAYLOAD_SCHEMA,
        "repair_id": REPAIR_ID,
        "parent": PARENT_ID,
        "rank": RANK,
        "feature_dim": FEATURE_DIM,
        "output_dim": OUTPUT_DIM,
        "target_shape": list(TARGET_SHAPE),
        "formula_version": FORMULA_VERSION,
        "dtype": "torch.float64",
        "device_storage": "cpu",
        "feature_contract": FEATURE_CONTRACT,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "producer_commit": PARENT_PRODUCER_COMMIT,
        "producer_module_sha256": PARENT_PRODUCER_MODULE_SHA256,
        "producer_config_sha256": PARENT_CONFIG_SHA256,
        "producer_review_sha256": PARENT_REVIEW_SHA256,
        "probe_sha256": PROBE_SHA256,
        "fit_panel_identity": "probe_fit_origin_reference|mnist_train_indices_0_54999|N=1000|per_label=100",
        "fit_panel_hash": FIT_PANEL_HASH,
        "mu_fit_parent_hash": MU_FIT_SHA256,
        "C_fit_parent_hash": C_FIT_SHA256,
        "tensor_count": len(PAYLOAD_KEYS),
        "scalar_count_total": SCALAR_COUNT_TOTAL,
        "raw_float64_tensor_bytes": RAW_FLOAT64_BYTES,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise _invalid(f"frozen artifact manifest field {key} is mismatched")
    parent_hashes = _require_mapping(manifest["parent_evidence_hashes"], "parent_evidence_hashes")
    if dict(parent_hashes) != PARENT_EVIDENCE_HASHES:
        raise _invalid("parent evidence hashes are mismatched")
    for label, value in PARENT_EVIDENCE_HASHES.items():
        _sha256(value, f"parent_evidence_hashes.{label}")
    tensors = _require_mapping(manifest["tensors"], "tensors")
    if set(tensors) != PAYLOAD_KEYS:
        raise _invalid("frozen artifact tensor entries are invalid")
    for name in PAYLOAD_KEYS:
        entry = _require_mapping(tensors[name], f"tensors.{name}")
        if set(entry) != {"shape", "dtype", "tensor_sha256"}:
            raise _invalid(f"frozen artifact tensor entry {name} is invalid")
        if entry["shape"] != list(TENSOR_SHAPES[name]) or entry["dtype"] != "torch.float64":
            raise _invalid(f"frozen artifact tensor contract for {name} is mismatched")
        _sha256(entry["tensor_sha256"], f"tensors.{name}.tensor_sha256")
    _sha256(manifest["payload_file_sha256"], "payload_file_sha256")
    if not isinstance(manifest["serialized_bytes"], int) or isinstance(manifest["serialized_bytes"], bool):
        raise _invalid("serialized_bytes must be an integer")
    created_git = _require_mapping(manifest["created_git"], "created_git")
    if set(created_git) != {"commit", "branch", "clean"}:
        raise _invalid("created_git fields are invalid")
    _git_commit(created_git["commit"], "created_git.commit")
    if created_git["branch"] != BRANCH or created_git["clean"] is not True:
        raise _invalid("artifact was not created on the clean registered branch")


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
        raise _invalid(f"frozen artifact payload is missing: {payload_path}")
    expected_file_hash = _sha256(manifest["payload_file_sha256"], "payload_file_sha256")
    if file_sha256(payload_path) != expected_file_hash:
        raise _invalid("frozen artifact payload file hash mismatch")
    if payload_path.stat().st_size != manifest["serialized_bytes"]:
        raise _invalid("frozen artifact serialized byte count mismatch")
    try:
        payload = safe_torch_load(payload_path, expected_sha256=expected_file_hash, map_location="cpu")
    except (OSError, ValueError) as exc:
        raise _invalid("frozen artifact payload is not safely loadable") from exc
    if not isinstance(payload, Mapping) or set(payload) != PAYLOAD_KEYS:
        raise _invalid("frozen artifact payload keys are invalid")
    entries = _require_mapping(manifest["tensors"], "tensors")
    loaded: dict[str, torch.Tensor] = {}
    for name in PAYLOAD_KEYS:
        value = payload[name]
        if not torch.is_tensor(value):
            raise _invalid(f"frozen artifact payload entry {name} is not a tensor")
        if value.device.type != "cpu" or value.dtype != torch.float64:
            raise _invalid(f"frozen artifact payload entry {name} dtype/device is invalid")
        if tuple(value.shape) != TENSOR_SHAPES[name] or not bool(torch.isfinite(value).all()):
            raise _invalid(f"frozen artifact payload entry {name} shape or finite contract is invalid")
        if tensor_sha256(value) != entries[name]["tensor_sha256"]:
            raise _invalid(f"frozen artifact tensor hash mismatch for {name}")
        loaded[name] = value.detach().contiguous()
    return loaded


@dataclass(frozen=True)
class FrozenMPCDARRank16Readout:
    """A read-only rank-16 readout loaded from a validated artifact."""

    manifest_path: Path
    mu_fit: torch.Tensor
    c_mean: torch.Tensor
    T16: torch.Tensor
    V16: torch.Tensor
    manifest: Mapping[str, Any]

    @classmethod
    def from_manifest(cls, path: str | Path) -> "FrozenMPCDARRank16Readout":
        manifest_path = Path(path).resolve()
        if not manifest_path.is_file():
            raise _invalid(f"frozen artifact manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise _invalid("frozen artifact manifest is unreadable") from exc
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
        """Apply the only registered frozen rank-16 prediction formula."""
        if not torch.is_tensor(z) or z.ndim < 1 or z.shape[-1] != FEATURE_DIM:
            raise ValueError("frozen MPCDAR readout input must have shape [...,256]")
        if not z.is_floating_point() or not bool(torch.isfinite(z).all()):
            raise ValueError("frozen MPCDAR readout input must be finite real-valued features")
        leading_shape = tuple(z.shape[:-1])
        z64 = z.reshape(-1, FEATURE_DIM).to(torch.float64)
        device = z64.device
        mu_fit = self.mu_fit.to(device=device)
        c_mean = self.c_mean.to(device=device)
        T16 = self.T16.to(device=device)
        V16 = self.V16.to(device=device)
        with torch.no_grad():
            z_centered = z64 - mu_fit
            hidden = z_centered.mm(T16)
            prediction = c_mean + hidden.mm(V16.T)
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("frozen MPCDAR readout produced non-finite output")
        return prediction.reshape((*leading_shape, OUTPUT_DIM))

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.predict(z)


def load_frozen_mpcdar_rank16_readout(path: str | Path) -> FrozenMPCDARRank16Readout:
    """Load and validate an explicit frozen MPCDAR rank-16 manifest."""
    return FrozenMPCDARRank16Readout.from_manifest(path)


__all__ = [
    "C_FIT_SHA256",
    "FEATURE_CONTRACT",
    "FEATURE_CONTRACT_VERSION",
    "FORMULA_VERSION",
    "FrozenMPCDARRank16ArtifactError",
    "FrozenMPCDARRank16Readout",
    "MANIFEST_SCHEMA",
    "MU_FIT_SHA256",
    "OUTPUT_DIM",
    "PARENT_EVIDENCE_HASHES",
    "PARENT_ID",
    "PAYLOAD_KEYS",
    "PAYLOAD_SCHEMA",
    "RANK",
    "REPAIR_ID",
    "SCALAR_COUNT_TOTAL",
    "TENSOR_SHAPES",
    "load_frozen_mpcdar_rank16_readout",
]
