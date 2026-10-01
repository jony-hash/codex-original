"""LightGen-like Identity optical-manifold research lifecycle.

This module is independent from the current conditional generator and from
historical fiber/evaluator research routes. It owns validation and a bounded
three-step CUDA preflight; it deliberately does not expose formal training.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ldm_project.v2.config import FormalSystemConfig, load_config
from ldm_project.v2.contracts import ExposureCalibration
from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.optics.system import FormalAmplitudeFiniteCropIdentitySystem


BRANCH = "research/v2-lightgen-optical-manifold-v1"
CONFIG_SCHEMA = "lightgen_optical_manifold_identity_v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_identity_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_identity_preflight.v1"
RESUME_SCHEMA = "lightgen_optical_manifold_identity_resume.v1"
HYPOTHESIS = "H-LOMA-I0"
NEXT_ACTION = "manual_review_identity_reconstruction_training_contract"
EXPECTED_FORMAL_KIND = "formal_amplitude_finite_crop_identity"
EXPECTED_TRAINABLE = ("system.encoder.phase", "system.decoder.phase")
EXPOSURE_ID = "lightgen-optical-manifold-fixed-global-v1"
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_WALL_SECONDS = 300.0
MAX_STORAGE_BYTES = 2 * 1024**3
PREFLIGHT_STEPS = 3

EXPECTED_SYSTEM = {
    "wavelength_m": 532.0e-9,
    "slm_shape": (200, 200),
    "slm_pixel_pitch_m": 2.5e-6,
    "metasurface_shape": (100, 100),
    "metasurface_pixel_pitch_m": 2.0e-6,
    "propagation_grid_size": 2048,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "layer_spacing_m": 1.50e-3,
    "front_magnification": -0.4,
    "back_magnification": -2.5,
    "ols_kind": "identity",
    "boundary": "finite_crop",
}


@dataclass(frozen=True)
class OpticalManifoldSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class RenderedBatch:
    intensity: torch.Tensor
    amplitude: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class OpticalManifoldOutput:
    raw_camera: torch.Tensor
    reconstruction: torch.Tensor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    root_candidate = _repo_root() / candidate
    if root_candidate.exists():
        return root_candidate
    return (base or Path.cwd()) / candidate


def _git_state() -> dict[str, Any]:
    root = _repo_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "branch": branch, "clean": not status, "status": status}


def _load_spec(path: str | Path) -> OpticalManifoldSpec:
    target = _resolve_path(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("optical-manifold config must be a YAML mapping")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("unsupported optical-manifold config")
    required = {
        "schema", "hypothesis", "formal_system_config", "formal_system", "dataset",
        "input_encoding", "target", "exposure", "loss", "optimizer",
        "initialization", "execution", "research",
    }
    if set(payload) != required:
        raise ValueError("optical-manifold config has unexpected or missing fields")
    return OpticalManifoldSpec(target, payload, canonical_sha256(payload))


def _assert_close(actual: float, expected: float, name: str) -> None:
    if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
        raise ValueError(f"optical-manifold config locks {name}={expected!r}")


def _load_formal_config(spec: OpticalManifoldSpec) -> tuple[FormalSystemConfig, Path]:
    formal_path = _resolve_path(spec.payload["formal_system_config"], base=spec.path.parent)
    formal = load_config(formal_path)
    if formal.architecture != "conditional_identity_logit_amplitude_v2":
        raise ValueError("Identity optical-manifold requires the current identity lineage")
    if formal.system.kind != EXPECTED_FORMAL_KIND or not formal.system.is_formal:
        raise ValueError("optical-manifold requires the formal finite-crop Identity system")
    for name, expected in EXPECTED_SYSTEM.items():
        actual = getattr(formal.system, name)
        if isinstance(expected, float):
            _assert_close(actual, expected, name)
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    declared = spec.payload["formal_system"]
    expected_declared = {
        "wavelength_m": 532.0e-9,
        "slm_shape": [200, 200],
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "boundary": "finite_crop",
        "ols": "identity",
    }
    if not isinstance(declared, Mapping) or dict(declared) != expected_declared:
        raise ValueError("research config does not lock the required formal geometry")
    return formal.system, formal_path


def _assert_research_contract(spec: OpticalManifoldSpec) -> None:
    payload = spec.payload
    if payload["dataset"] != {
        "train": True,
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
        "representative_selection": "first_occurrence_per_label",
    }:
        raise ValueError("optical-manifold dataset contract is not locked")
    if payload["input_encoding"] != "intensity_sqrt_amplitude":
        raise ValueError("optical-manifold requires sqrt intensity amplitude encoding")
    if payload["target"] != "camera_intensity_same_fov" or payload["loss"] != "mse":
        raise ValueError("optical-manifold target/loss contract is not locked")
    if payload["exposure"] != {"scale": 1.0, "gamma": 1.0}:
        raise ValueError("optical-manifold exposure is not fixed at scale 1 and gamma 1")
    if payload["optimizer"] != {
        "name": "Adam", "learning_rate": 1.0e-3,
        "logical_batch_size": 10, "microbatch_size": 1,
    }:
        raise ValueError("optical-manifold optimizer contract is not locked")
    if payload["initialization"] != {"phase": "zero", "seed": 1234}:
        raise ValueError("optical-manifold initialization contract is not locked")
    if payload["execution"] != {
        "propagation_grid_size": 2048, "activation_checkpoint": True, "device": "cuda",
    }:
        raise ValueError("optical-manifold execution contract is not locked")
    if payload["research"] != {
        "diagnostic_only": True, "formal_promotion": False, "training_authorized": False,
        "evaluator": "none", "software_decoder": False, "smf": False, "fmf": False,
        "next_action": NEXT_ACTION,
    }:
        raise ValueError("optical-manifold research boundary is not locked")


def load_local_mnist(dataset_root: str | Path, *, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Load local MNIST without importing the evaluator runtime."""
    from torchvision.datasets import MNIST

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError("local MNIST root does not exist")
    dataset = MNIST(root=str(root), train=train, download=False)
    return dataset.data.unsqueeze(1).to(torch.float32).div_(255.0), dataset.targets.to(torch.long)


def render_intensity_batch(
    intensity: torch.Tensor, *, output_shape: tuple[int, int] = (200, 200)
) -> RenderedBatch:
    """Resize real intensity, then derive the real amplitude SLM command."""
    if intensity.ndim != 4 or tuple(intensity.shape[1:]) != (1, 28, 28):
        raise ValueError("MNIST intensity must have shape [N, 1, 28, 28]")
    if not torch.is_floating_point(intensity) or torch.is_complex(intensity):
        raise TypeError("MNIST intensity must be a real floating tensor")
    if not torch.isfinite(intensity).all() or intensity.min() < 0.0 or intensity.max() > 1.0:
        raise ValueError("MNIST intensity must be finite and in [0, 1]")
    rendered = F.interpolate(
        intensity, size=output_shape, mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0)
    return RenderedBatch(rendered, rendered.sqrt(), rendered.clone())


class IdentityOpticalManifoldAutoencoder(nn.Module):
    """Formal relay -> 2+2 phase-only -> Identity OLS -> CMOS autoencoder."""

    def __init__(self, config: FormalSystemConfig) -> None:
        super().__init__()
        self.system = FormalAmplitudeFiniteCropIdentitySystem(config)
        self.exposure = ExposureCalibration(EXPOSURE_ID, 1.0, 1.0)
        self._set_trainable_contract()

    def _set_trainable_contract(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in EXPECTED_TRAINABLE)
        actual = tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )
        if actual != EXPECTED_TRAINABLE:
            raise ValueError(f"unexpected trainable parameters: {actual}")

    def forward(
        self,
        amplitude: torch.Tensor,
        *,
        activation_checkpoint: bool = False,
        return_optical_trace: bool = False,
    ) -> OpticalManifoldOutput:
        if amplitude.ndim != 4 or tuple(amplitude.shape[1:]) != (1, 200, 200):
            raise ValueError("SLM amplitude must have shape [N, 1, 200, 200]")
        if torch.is_complex(amplitude) or not torch.isfinite(amplitude).all():
            raise ValueError("SLM amplitude must be finite and real")
        if amplitude.min() < 0.0 or amplitude.max() > 1.0:
            raise ValueError("SLM amplitude must be in [0, 1]")

        def run(command: torch.Tensor) -> torch.Tensor:
            return self.system(
                self.system.command(command),
                exposure=self.exposure,
                return_optical_trace=return_optical_trace,
            ).raw_camera.tensor

        if activation_checkpoint and torch.is_grad_enabled() and not return_optical_trace:
            raw = checkpoint(run, amplitude, use_reentrant=False)
        else:
            raw = run(amplitude)
        return OpticalManifoldOutput(raw, (raw / self.exposure.scale).clamp(0.0, 1.0))


def _dataset_manifest(
    images: torch.Tensor, labels: torch.Tensor, dataset_root: str | Path
) -> dict[str, Any]:
    if tuple(images.shape) != (60000, 1, 28, 28) or tuple(labels.shape) != (60000,):
        raise ValueError("optical-manifold requires the complete MNIST train split")
    if not torch.isfinite(images).all() or images.min() < 0.0 or images.max() > 1.0:
        raise ValueError("MNIST images must be finite and normalized to [0, 1]")
    representatives = []
    for label in range(10):
        matches = torch.nonzero(labels == label, as_tuple=False).flatten()
        if matches.numel() == 0:
            raise ValueError(f"MNIST manifest has no examples for label {label}")
        representatives.append(int(matches[0]))
    manifest = {
        "schema": "lightgen_optical_manifold_mnist_manifest.v1",
        "dataset_root": str(Path(dataset_root).resolve()),
        "train": True,
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "images_sha256": tensor_sha256(images),
        "labels_sha256": tensor_sha256(labels),
        "representative_indices": representatives,
        "representative_labels": list(range(10)),
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return manifest


def _selected_batch(
    images: torch.Tensor, labels: torch.Tensor, manifest: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, RenderedBatch]:
    indices = torch.as_tensor(manifest["representative_indices"], dtype=torch.long)
    selected_labels = labels.index_select(0, indices)
    if not torch.equal(selected_labels, torch.arange(10, dtype=torch.long)):
        raise ValueError("MNIST representative manifest is not one-per-class")
    selected_images = images.index_select(0, indices)
    return selected_images, selected_labels, render_intensity_batch(selected_images)


def _parameter_contract(model: IdentityOpticalManifoldAutoencoder) -> dict[str, Any]:
    names = tuple(name for name, _ in model.named_parameters())
    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if trainable != EXPECTED_TRAINABLE:
        raise ValueError("trainable parameter names are not exactly encoder/decoder phase")
    if any(parameter.requires_grad != (name in EXPECTED_TRAINABLE) for name, parameter in model.named_parameters()):
        raise ValueError("parameter freeze contract failed")
    return {
        "all_parameter_names": list(names),
        "trainable_parameter_names": list(trainable),
        "optimizer_parameter_names": list(EXPECTED_TRAINABLE),
        "phase_shapes": {
            name: list(parameter.shape)
            for name, parameter in model.named_parameters()
            if name in EXPECTED_TRAINABLE
        },
    }


def _load_data_manifest(dataset_root: str | Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    images, labels = load_local_mnist(dataset_root, train=True)
    return images, labels, _dataset_manifest(images, labels, dataset_root)


def _record_payload(
    *,
    schema: str,
    record: EvidenceRecordV2,
    contract: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": schema,
        "evidence": record.to_dict(),
        "hypothesis": HYPOTHESIS,
        "training_authorized": False,
        "formal_promotion": False,
        "contract": dict(contract),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_new_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {target}")
    atomic_write_json(target, payload)


def _context(
    spec: OpticalManifoldSpec,
    formal: FormalSystemConfig,
    manifest: Mapping[str, Any],
) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=str(manifest["manifest_hash"]),
        calibration_hash=canonical_sha256(
            {"id": EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}
        ),
        exposure_scale=1.0,
    )


def _validation_record(
    state: Mapping[str, Any],
    spec: OpticalManifoldSpec,
    formal: FormalSystemConfig,
    formal_path: Path,
    manifest: Mapping[str, Any],
    parameter_contract: Mapping[str, Any],
) -> EvidenceRecordV2:
    gates = tuple(
        GateResult(name, True, metrics)
        for name, metrics in (
            ("clean_lineage", {"clean": True}),
            ("independent_branch", {"branch_matches": True}),
            ("formal_geometry", {"propagation_grid_size": 2048}),
            ("identity_ols", {"identity": True}),
            ("mnist_manifest", {"representative_count": 10}),
            ("input_target_coordinates", {"same_fov": True}),
            ("fixed_exposure", {"scale": 1.0, "gamma": 1.0}),
            ("trainable_parameter_set", {"count": len(parameter_contract["trainable_parameter_names"])}),
        )
    )
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=VALIDATION_SCHEMA,
        tier="diagnostic",
        status="passed",
        git_commit=str(state["commit"]),
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, manifest),
        input_hashes={
            "research_config": file_sha256(spec.path),
            "formal_config": file_sha256(formal_path),
            "mnist_manifest": str(manifest["manifest_hash"]),
        },
        gates=gates,
        permitted_next_actions=(NEXT_ACTION,),
    )
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_research_contract(spec)
    state = _git_state()
    if not state["clean"]:
        raise ValueError("optical-manifold validation requires a clean research worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"optical-manifold validation requires branch {BRANCH}")
    formal, formal_path = _load_formal_config(spec)
    images, labels, manifest = _load_data_manifest(args.dataset_root)
    model = IdentityOpticalManifoldAutoencoder(formal)
    parameter_contract = _parameter_contract(model)
    _, selected_labels, rendered = _selected_batch(images, labels, manifest)
    if tuple(rendered.amplitude.shape) != (10, 1, 200, 200):
        raise ValueError("rendered SLM batch shape contract failed")
    if not torch.allclose(
        rendered.amplitude.square(), rendered.target, atol=1.0e-7, rtol=0.0
    ):
        raise ValueError("sqrt amplitude and camera target are not paired")
    if not torch.equal(selected_labels, torch.arange(10)):
        raise ValueError("representative labels are not deterministic one-per-class")
    record = _validation_record(
        state, spec, formal, formal_path, manifest, parameter_contract
    )
    payload = _record_payload(
        schema=VALIDATION_SCHEMA,
        record=record,
        contract={
            "formal_config": str(formal_path.resolve()),
            "system_kind": EXPECTED_FORMAL_KIND,
            "input_encoding": "intensity_sqrt_amplitude",
            "target": "camera_intensity_same_fov",
            "parameter_contract": parameter_contract,
            "dataset_manifest": manifest,
            "evaluator_calls": 0,
        },
    )
    _write_new_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(
    path: str | Path, spec: OpticalManifoldSpec, manifest: Mapping[str, Any]
) -> EvidenceRecordV2:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != VALIDATION_SCHEMA:
        raise ValueError("preflight requires optical-manifold validation evidence")
    if payload.get("hypothesis") != HYPOTHESIS or payload.get("training_authorized") is not False:
        raise ValueError("validation evidence has an invalid research boundary")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed":
        raise ValueError("validation evidence must be passed")
    state = _git_state()
    if record.git_commit != state["commit"] or state["branch"] != BRANCH or not state["clean"]:
        raise ValueError("validation evidence is stale or lineage is not clean")
    if record.config_hash != spec.config_hash:
        raise ValueError("validation evidence config hash mismatch")
    if record.input_hashes.get("research_config") != file_sha256(spec.path):
        raise ValueError("validation evidence research config binding mismatch")
    if record.input_hashes.get("mnist_manifest") != manifest["manifest_hash"]:
        raise ValueError("validation evidence MNIST manifest mismatch")
    return record


def _clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _rng_state() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _named_optimizer_parameters(
    model: IdentityOpticalManifoldAutoencoder, optimizer: torch.optim.Optimizer
) -> tuple[str, ...]:
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    return tuple(
        by_id[id(parameter)]
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def _make_optimizer(
    model: IdentityOpticalManifoldAutoencoder,
) -> torch.optim.Adam:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name in EXPECTED_TRAINABLE
    ]
    optimizer = torch.optim.Adam(parameters, lr=1.0e-3)
    if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]["lr"] != 1.0e-3:
        raise ValueError("optical-manifold optimizer group contract failed")
    if _named_optimizer_parameters(model, optimizer) != EXPECTED_TRAINABLE:
        raise ValueError("optical-manifold optimizer parameter contract failed")
    return optimizer


def _step(
    model: IdentityOpticalManifoldAutoencoder,
    optimizer: torch.optim.Optimizer,
    batch: RenderedBatch,
    *,
    activation_checkpoint: bool,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    losses: list[torch.Tensor] = []
    cameras: list[torch.Tensor] = []
    for start in range(0, batch.amplitude.shape[0], 1):
        output = model(
            batch.amplitude[start : start + 1],
            activation_checkpoint=activation_checkpoint,
        )
        loss = F.mse_loss(
            output.reconstruction, batch.target[start : start + 1]
        )
        losses.append(loss.detach())
        cameras.append(output.raw_camera.detach())
        (loss / batch.amplitude.shape[0]).backward()
    gradients = {
        name: float(parameter.grad.detach().norm().cpu())
        for name, parameter in model.named_parameters()
        if name in EXPECTED_TRAINABLE and parameter.grad is not None
    }
    optimizer.step()
    cameras_tensor = torch.cat(cameras, dim=0)
    mean_loss = torch.stack(losses).mean()
    return {
        "loss": float(mean_loss.cpu()),
        "gradient_norms": gradients,
        "camera_finite": bool(torch.isfinite(cameras_tensor).all()),
        "camera_min": float(cameras_tensor.min().cpu()),
        "camera_max": float(cameras_tensor.max().cpu()),
    }


def _resume_payload(
    model: IdentityOpticalManifoldAutoencoder,
    optimizer: torch.optim.Optimizer,
    rng: Mapping[str, Any],
    *,
    step: int,
    spec: OpticalManifoldSpec,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state": _clone_tree(optimizer.state_dict()),
        "rng_state": _clone_tree(rng),
        "step": step,
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": manifest["manifest_hash"],
        "data_cursor": {
            "logical_batch_size": 10,
            "microbatch_size": 1,
            "next_step": step + 1,
        },
    }


def _load_resume(
    path: str | Path,
    model: IdentityOpticalManifoldAutoencoder,
    optimizer: torch.optim.Optimizer,
    *,
    spec: OpticalManifoldSpec,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != RESUME_SCHEMA:
        raise ValueError("invalid optical-manifold resume artifact")
    if (
        payload.get("config_hash") != spec.config_hash
        or payload.get("dataset_manifest_hash") != manifest["manifest_hash"]
    ):
        raise ValueError("resume artifact lineage mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng(payload["rng_state"])
    return payload


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _invalid_resource_record(
    state: Mapping[str, Any],
    spec: OpticalManifoldSpec,
    formal: FormalSystemConfig,
    manifest: Mapping[str, Any],
    reason: str,
) -> EvidenceRecordV2:
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=PREFLIGHT_SCHEMA,
        tier="diagnostic",
        status="invalid_harness",
        git_commit=str(state["commit"]),
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, manifest),
        input_hashes={
            "validation_evidence": "not-created",
            "mnist_manifest": manifest["manifest_hash"],
        },
        gates=(GateResult("resource_preflight", False, {}, reason),),
        permitted_next_actions=(),
        invalidation_reason=reason,
    )
    record.validate()
    return record


def preflight(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("optical-manifold preflight requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    spec = _load_spec(args.config)
    _assert_research_contract(spec)
    state = _git_state()
    if not state["clean"] or state["branch"] != BRANCH:
        raise ValueError("optical-manifold preflight requires the clean research branch")
    formal, _ = _load_formal_config(spec)
    images, labels, manifest = _load_data_manifest(args.dataset_root)
    validation = _load_validation(args.validate_evidence, spec, manifest)
    del validation
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing preflight evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    validation_hash = file_sha256(args.validate_evidence)
    if not torch.cuda.is_available():
        record = _invalid_resource_record(
            state, spec, formal, manifest, "CUDA is unavailable"
        )
        payload = _record_payload(
            schema=PREFLIGHT_SCHEMA,
            record=record,
            contract={"validation_evidence_hash": validation_hash},
        )
        _write_new_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        record = _invalid_resource_record(
            state,
            spec,
            formal,
            manifest,
            f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}",
        )
        payload = _record_payload(
            schema=PREFLIGHT_SCHEMA,
            record=record,
            contract={
                "validation_evidence_hash": validation_hash,
                "free_cuda_bytes": int(free_bytes),
                "total_cuda_bytes": int(total_bytes),
            },
        )
        _write_new_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = IdentityOpticalManifoldAutoencoder(formal).to(device)
    _, _, rendered = _selected_batch(images, labels, manifest)
    batch = RenderedBatch(
        rendered.intensity.to(device),
        rendered.amplitude.to(device),
        rendered.target.to(device),
    )
    optimizer = _make_optimizer(model)
    initial_state = _clone_tree(model.state_dict())
    initial_rng = _rng_state()
    initial_optimizer = _clone_tree(optimizer.state_dict())
    steps = [
        _step(model, optimizer, batch, activation_checkpoint=True)
        for _ in range(PREFLIGHT_STEPS)
    ]
    uninterrupted_state = _clone_tree(model.state_dict())
    uninterrupted_optimizer = _clone_tree(optimizer.state_dict())
    uninterrupted_rng = _rng_state()

    interrupted = IdentityOpticalManifoldAutoencoder(formal).to(device)
    interrupted.load_state_dict(initial_state, strict=True)
    interrupted_optimizer = _make_optimizer(interrupted)
    interrupted_optimizer.load_state_dict(initial_optimizer)
    _restore_rng(initial_rng)
    _step(interrupted, interrupted_optimizer, batch, activation_checkpoint=True)
    _step(interrupted, interrupted_optimizer, batch, activation_checkpoint=True)
    resume_rng = _rng_state()
    resume_path = output.with_name(output.stem + ".resume.pt")
    resume = _resume_payload(
        interrupted,
        interrupted_optimizer,
        resume_rng,
        step=2,
        spec=spec,
        manifest=manifest,
    )
    atomic_torch_save(resume_path, resume)
    del interrupted, interrupted_optimizer
    torch.cuda.empty_cache()

    resumed = IdentityOpticalManifoldAutoencoder(formal).to(device)
    resumed_optimizer = _make_optimizer(resumed)
    _load_resume(
        resume_path,
        resumed,
        resumed_optimizer,
        spec=spec,
        manifest=manifest,
    )
    resumed_step = _step(resumed, resumed_optimizer, batch, activation_checkpoint=True)
    replay_exact = (
        _tree_equal(uninterrupted_state, resumed.state_dict())
        and _tree_equal(uninterrupted_optimizer, resumed_optimizer.state_dict())
        and _tree_equal(uninterrupted_rng, _rng_state())
        and steps[2] == resumed_step
    )
    elapsed = time.monotonic() - started
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    storage_bytes = _directory_size(output.parent)
    parameter_updates = {
        name: float(
            (model.state_dict()[name] - initial_state[name].to(device)).abs().sum().cpu()
        )
        for name in EXPECTED_TRAINABLE
    }
    frozen_unchanged = all(
        torch.equal(model.state_dict()[name], initial_state[name].to(device))
        for name in model.state_dict()
        if name not in EXPECTED_TRAINABLE
    )
    finite = all(
        torch.isfinite(torch.tensor(item["loss"]))
        and item["camera_finite"]
        and torch.isfinite(torch.tensor(item["camera_min"]))
        and torch.isfinite(torch.tensor(item["camera_max"]))
        and all(torch.isfinite(torch.tensor(value)) for value in item["gradient_norms"].values())
        for item in steps
    )
    gradients_nonzero = all(
        item["gradient_norms"].get(name, 0.0) > 0.0
        for item in steps
        for name in EXPECTED_TRAINABLE
    )
    updates_nonzero = all(value > 0.0 for value in parameter_updates.values())
    gates = (
        GateResult("finite_values", bool(finite), {"steps": PREFLIGHT_STEPS}),
        GateResult("phase_gradients_nonzero", gradients_nonzero, {"parameter_count": 4}),
        GateResult("phase_parameters_updated", updates_nonzero, parameter_updates),
        GateResult("frozen_state_unchanged", frozen_unchanged, {}),
        GateResult(
            "loss_decreased",
            steps[-1]["loss"] < steps[0]["loss"],
            {"initial": steps[0]["loss"], "step_3": steps[-1]["loss"]},
        ),
        GateResult(
            "identity_ols_and_formal_relay",
            True,
            {"identity_ols": True, "propagation_grid_size": 2048},
        ),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("resume_exact", replay_exact, {}),
        GateResult("cuda_peak_memory", peak_bytes <= MAX_PEAK_BYTES, {"bytes": peak_bytes}),
        GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage_bytes <= MAX_STORAGE_BYTES, {"bytes": storage_bytes}),
    )
    passed = all(gate.passed for gate in gates)
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=PREFLIGHT_SCHEMA,
        tier="diagnostic",
        status="passed" if passed else "invalid_harness",
        git_commit=state["commit"],
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, manifest),
        input_hashes={
            "validation_evidence": validation_hash,
            "mnist_manifest": manifest["manifest_hash"],
            "resume_artifact": file_sha256(resume_path),
        },
        gates=gates,
        permitted_next_actions=(NEXT_ACTION,) if passed else (),
        invalidation_reason=None if passed else "one or more preflight execution gates failed",
    )
    record.validate()
    payload = _record_payload(
        schema=PREFLIGHT_SCHEMA,
        record=record,
        contract={
            "validation_evidence_hash": validation_hash,
            "formal_system_kind": EXPECTED_FORMAL_KIND,
            "input_encoding": "intensity_sqrt_amplitude",
            "target": "camera_intensity_same_fov",
            "optimizer": {"name": "Adam", "learning_rate": 1.0e-3},
            "logical_batch_size": 10,
            "microbatch_size": 1,
            "activation_checkpoint": True,
            "trainable_parameter_names": list(EXPECTED_TRAINABLE),
            "resume_artifact": str(resume_path.resolve()),
            "camera_shape": [10, 1, 200, 200],
            "evaluator_calls": 0,
            "steps": steps,
            "resumed_step": resumed_step,
            "free_cuda_bytes_before": int(free_bytes),
            "total_cuda_bytes": int(total_bytes),
            "peak_cuda_bytes": peak_bytes,
            "wall_seconds": elapsed,
            "run_storage_bytes": storage_bytes,
        },
    )
    _write_new_json(output, payload)
    torch.use_deterministic_algorithms(deterministic_before)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action in {
        "train-validate", "train-preflight", "train",
        "full-epoch-validate", "full-epoch-preflight", "full-epoch-train",
        "reconstruction-audit-validate", "reconstruction-audit-run",
        "latent-observer-validate", "latent-observer-run",
        "manifold-regularization-audit-validate",
        "manifold-regularization-audit-run",
        "manifold-regularization-pilot-validate",
        "manifold-regularization-pilot-preflight",
        "manifold-regularization-pilot-train",
        "manifold-regularization-objective-audit-validate",
        "manifold-regularization-objective-audit-run",
        "smf-integration-canary-validate",
        "smf-integration-canary-preflight",
        "smf-integration-canary-train",
    }:
        if args.research_action.startswith("full-epoch"):
            from ldm_project.v2.application.research.optical_manifold.optical_manifold_full_epoch import dispatch as training_dispatch
        elif args.research_action.startswith("reconstruction-audit"):
            from ldm_project.v2.application.research.optical_manifold.optical_manifold_reconstruction_audit import dispatch as audit_dispatch

            return int(audit_dispatch(args) or 0)
        elif args.research_action.startswith("latent-observer"):
            from ldm_project.v2.application.research.optical_manifold.optical_manifold_latent_observer import dispatch as observer_dispatch

            return int(observer_dispatch(args) or 0)
        elif args.research_action.startswith("manifold-regularization-audit"):
            from ldm_project.v2.application.research.optical_manifold.optical_manifold_regularization_audit import dispatch as regularization_dispatch

            return int(regularization_dispatch(args) or 0)
        elif args.research_action.startswith("manifold-regularization-pilot"):
            from ldm_project.v2.application.research.optical_manifold.optical_manifold_regularization_pilot import dispatch as pilot_dispatch

            return int(pilot_dispatch(args) or 0)
        elif args.research_action.startswith("manifold-regularization-objective-audit"):
            from ldm_project.v2.application.research.optical_manifold.optical_manifold_regularization_objective_audit import dispatch as objective_dispatch

            return int(objective_dispatch(args) or 0)
        else:
            from ldm_project.v2.application.research.optical_manifold.optical_manifold_training import dispatch as training_dispatch

        return int(training_dispatch(args) or 0)
    if args.research_action == "validate":
        return validate(args)
    if args.research_action == "preflight":
        return preflight(args)
    raise ValueError(f"unsupported optical-manifold action: {args.research_action}")
