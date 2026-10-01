"""Run the matched long ideal output-facet ReLU experiment.

The run is deliberately a bounded research consumer of the frozen N5-A
endpoint.  It does not alter the formal v2 runtime, add optical depth, read
protected data, or claim physical Te validation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load research authority: {path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load research authority: {path}")
    spec.loader.exec_module(module)
    return module


# Existing N5/N4/V10/V11 files are code authorities only.  No historical N6
# observation is imported as a scientific observation by this consumer.
N6 = _load_module(
    Path(__file__).resolve().parents[1]
    / "optical_manifold"
    / "execute_lightgen_2d_nonlinear_fiber_facet_activation_canary_v1.py",
    "lightgen_long_relu_n6_code_authority",
)
N5 = N6.N5
N4 = N6.N4
V10 = N6.V10
V11 = N6.V11


TASK_ID = "EXECUTE_LONG_IDEAL_FACET_RELU_PRIMARY_GAIN_V1"
DESIGN_AUTHORITY = "DEFINE_LONG_IDEAL_FACET_RELU_PRIMARY_GAIN_V1"
OUTER_GATE = "LONG_IDEAL_FACET_RELU90_PRIMARY_GAIN_V1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

RUN_ROOT = Path("outputs/20260915_ideal_facet_relu_long/runs")
CONTRACT_DIR = RUN_ROOT / "lightgen_2d_ideal_facet_relu_long_contract_v1_attempt_06_minimal_ondemand"
CONTRACT_PATH = CONTRACT_DIR / "lightgen_2d_ideal_facet_relu_long_contract.v1.json"
PREFLIGHT_DIR = RUN_ROOT / "lightgen_2d_ideal_facet_relu_long_preflight_v1_attempt_06_minimal_ondemand"
PREFLIGHT_REPORT = PREFLIGHT_DIR / "lightgen_2d_ideal_facet_relu_long_preflight_report.v1.json"
PREFLIGHT_LINEAGE = PREFLIGHT_DIR / "lightgen_2d_ideal_facet_relu_long_preflight_lineage.v1.json"
PREFLIGHT_INPUT = PREFLIGHT_DIR / "lightgen_2d_ideal_facet_relu_long_preflight_input.v1.json"
CACHE_DIR = PREFLIGHT_DIR / "cache"
CACHE_PATH = CACHE_DIR / "lightgen_2d_ideal_facet_relu_long_frozen_ols_cache.v1.pt"
ARM_DIRS = {
    "A_LINEAR": RUN_ROOT / "lightgen_2d_ideal_facet_relu_long_a_linear_attempt_05_minimal_ondemand",
    "B_RELU90": RUN_ROOT / "lightgen_2d_ideal_facet_relu_long_b_relu90_attempt_05_minimal_ondemand",
}
AGGREGATE_DIR = RUN_ROOT / "lightgen_2d_ideal_facet_relu_long_aggregate_v1_attempt_05_minimal_ondemand"

N5_A_REPORT_PATH = N6.N5_A_REPORT_PATH
N5_A_CHECKPOINT_PATH = N6.N5_A_CHECKPOINT_PATH
N5_A_REPORT_SHA256 = N6.N5_A_REPORT_SHA256
N5_A_CHECKPOINT_SHA256 = N6.N5_A_CHECKPOINT_SHA256
N5_CONTRACT_SHA256 = N6.N5_CONTRACT_SHA256

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
ADAPTER_RANK = 9
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = N6.PRODUCTION_GRID
PRECHECK_STEPS = 3
FAST_DIRECT_PREFLIGHT = True
FAST_DIRECT_K = 2000
FAST_DIRECT_REFINEMENT_ITERS = 2
CALIBRATION_SAMPLES_PER_CLASS = 1
CALIBRATION_SEED = 1931
SAMPLER_SEED = 1234
K_MIN = 2000
K_MAX = 3000
ACTIVE_GPU_BUDGET_SECONDS = 18.0 * 3600.0
BLOCK_SIZE = 200
CALIBRATION_PIXEL_STRIDE = 1024
CALIBRATION_REFINEMENT_ITERS = 20
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
STAGE_A_FIREWALL = 0.937
MATERIAL_DELTA = 0.05
RECONSTRUCTION_MAX_DELTA = 0.005
LAMBDA_TRANSFER = 28.48464149920509
LAMBDA_IMAGE = 13.353282014834436
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
INITIAL_LR = 1.0e-3
MID_LR = 2.0e-4
FINAL_LR = 5.0e-5
MATERIAL_EPSILON = N6.MATERIAL_EPSILON
MATERIAL_SLOPE = 1.0
MATERIAL_GAIN = 1.0
RETENTION_TOLERANCE = 1.0e-5

TRAINABLE_PARAMETER_NAMES = [
    "adapter.u_real",
    "adapter.u_imag",
    "adapter.v_real",
    "adapter.v_imag",
    "generator.phase",
]
DEV_CHECKPOINT_FRACTIONS = (0.0, 1.0)


class InvalidLongRun(RuntimeError):
    """Invalid implementation, input, or lineage boundary."""


class HardBlockedLongRun(RuntimeError):
    """A valid boundary prevents the authorized execution."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidLongRun(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(root: Path, value: Path) -> str:
    return value.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    _require(_sha256(path) == expected, f"{name} file-byte SHA-256 mismatch")


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return list(left) == list(right) and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _digest_tree(value: Any) -> str:
    return N4._digest_tree(_cpu_tree(value))


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _status_lines(root: Path) -> list[str]:
    return V10.V6._status_lines(root)


class BalancedMultiEpochSampler:
    """One example per class per logical batch, with exact CPU RNG state."""

    SCHEMA = "lightgen_long_balanced_multi_epoch_sampler.v1"

    def __init__(self, train_indices: torch.Tensor, labels: torch.Tensor, *, seed: int, steps: int) -> None:
        _require(train_indices.dtype == torch.long and labels.dtype == torch.long, "long sampler roles must be long tensors")
        _require(train_indices.ndim == 1 and labels.ndim == 1, "long sampler role tensors must be one-dimensional")
        self.seed = int(seed)
        self.steps = int(steps)
        self.cursor = 0
        self.epoch = 0
        self.position = 0
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)
        role_labels = labels.index_select(0, train_indices).cpu()
        self._candidates: list[torch.Tensor] = []
        for digit in range(CLASS_COUNT):
            values = train_indices[role_labels == digit].cpu()
            _require(values.numel() > 0, f"long sampler has no TRAIN samples for class {digit}")
            self._candidates.append(values.clone())
        self._per_class = min(int(values.numel()) for values in self._candidates)
        self._permutations = [self._new_permutation(values) for values in self._candidates]

    def _new_permutation(self, values: torch.Tensor) -> torch.Tensor:
        return values.index_select(0, torch.randperm(values.numel(), generator=self._generator)).clone()

    def _advance_epoch(self) -> None:
        self.epoch += 1
        self.position = 0
        self._permutations = [self._new_permutation(values) for values in self._candidates]

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cursor >= self.steps:
            raise StopIteration("long balanced sampler is exhausted")
        indices = torch.stack([values[self.position] for values in self._permutations])
        labels = torch.arange(CLASS_COUNT, dtype=torch.long)
        self.cursor += 1
        self.position += 1
        if self.position == self._per_class:
            self._advance_epoch()
        return indices, labels

    def peek_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cursor >= self.steps:
            raise StopIteration("long balanced sampler is exhausted")
        return torch.stack([values[self.position] for values in self._permutations]), torch.arange(CLASS_COUNT, dtype=torch.long)

    def state_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "seed": self.seed, "steps": self.steps, "cursor": self.cursor, "epoch": self.epoch, "position": self.position, "per_class": self._per_class, "generator_state": self._generator.get_state().clone(), "permutations": [value.clone() for value in self._permutations]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _require(state.get("schema") == self.SCHEMA, "long sampler schema changed")
        _require(int(state.get("seed", -1)) == self.seed and int(state.get("steps", -1)) == self.steps, "long sampler identity changed")
        cursor, epoch, position = state.get("cursor"), state.get("epoch"), state.get("position")
        _require(isinstance(cursor, int) and 0 <= cursor <= self.steps, "long sampler cursor is invalid")
        _require(isinstance(epoch, int) and isinstance(position, int), "long sampler epoch state is invalid")
        _require(int(state.get("per_class", -1)) == self._per_class, "long sampler class-window size changed")
        _require(epoch >= 0 and 0 <= position < self._per_class, "long sampler epoch position is invalid")
        _require(cursor == epoch * self._per_class + position, "long sampler cursor/epoch relation changed")
        generator_state = state.get("generator_state")
        _require(isinstance(generator_state, torch.Tensor), "long sampler generator state is missing")
        permutations = state.get("permutations")
        _require(isinstance(permutations, list) and len(permutations) == CLASS_COUNT, "long sampler permutations are invalid")
        for candidates, actual in zip(self._candidates, permutations, strict=True):
            _require(torch.is_tensor(actual) and actual.dtype == torch.long and actual.shape == candidates.shape, "long sampler permutation shape changed")
            _require(torch.equal(torch.sort(actual).values, torch.sort(candidates).values), "long sampler permutation members changed")
        self.cursor = cursor
        self.epoch = epoch
        self.position = position
        self._generator.set_state(generator_state.cpu())
        self._permutations = [value.cpu().clone() for value in permutations]


def _train_position_map(train_indices: torch.Tensor) -> dict[int, int]:
    values = [int(value) for value in train_indices.tolist()]
    _require(len(values) == len(set(values)), "TRAIN indices are not unique")
    return {sample_id: position for position, sample_id in enumerate(values)}


def _cache_positions(position_map: Mapping[int, int], sample_ids: torch.Tensor) -> torch.Tensor:
    try:
        values = [position_map[int(value)] for value in sample_ids.tolist()]
    except KeyError as error:
        raise InvalidLongRun(f"sample ID is outside the cached TRAIN role: {error}") from error
    return torch.tensor(values, dtype=torch.long)


class LongFacetProbe:
    """Fixed local intensity ReLU inserted at the existing decoder entrance."""

    def __init__(self, *, mode: str, tau: float = 0.0, capture: bool = False) -> None:
        _require(mode in {"linear", "relu"}, f"unknown facet mode: {mode}")
        self.mode = mode
        self.tau = float(tau)
        self.capture = bool(capture)
        self.capture_samples = False
        self.records: list[dict[str, torch.Tensor]] = []
        self.input_samples: list[torch.Tensor] = []
        self.placements: list[dict[str, Any]] = []
        self.input_max = 0.0

    @property
    def nonlinear(self) -> bool:
        return self.mode == "relu"

    def clear(self) -> None:
        self.records.clear()
        self.input_samples.clear()
        self.placements.clear()
        self.input_max = 0.0

    def apply(self, field: Any) -> Any:
        tensor = field.tensor
        intensity = tensor.abs().square()
        flat = intensity.flatten(1)
        self.input_max = max(self.input_max, float(intensity.detach().max().cpu()))
        if self.capture_samples:
            self.input_samples.append(flat[:, ::CALIBRATION_PIXEL_STRIDE].detach().cpu())
        output_intensity = torch.relu(intensity - self.tau) if self.nonlinear else intensity
        if self.nonlinear:
            ratio = (output_intensity + MATERIAL_EPSILON) / (intensity + MATERIAL_EPSILON)
            output_tensor = tensor * ratio.sqrt()
        else:
            output_tensor = tensor
        _finite(output_tensor.real, "long facet output real")
        _finite(output_tensor.imag, "long facet output imag")
        input_power = flat.sum(dim=1)
        output_power = output_intensity.flatten(1).sum(dim=1)
        summaries = {
            "input_mean": flat.mean(dim=1),
            "output_mean": output_intensity.flatten(1).mean(dim=1),
            "input_power": input_power,
            "output_power": output_power,
            "power_ratio": output_power / input_power.clamp_min(MATERIAL_EPSILON),
            "active_fraction": (intensity > self.tau).flatten(1).to(torch.float32).mean(dim=1),
            "suppressed_fraction": (output_intensity <= MATERIAL_EPSILON).flatten(1).to(torch.float32).mean(dim=1),
            "transition_fraction": ((intensity - self.tau).abs() <= 0.01).flatten(1).to(torch.float32).mean(dim=1),
            "local_slope": (intensity > self.tau).flatten(1).to(torch.float32).mean(dim=1),
        }
        for name, value in summaries.items():
            _finite(value, f"long facet statistic {name}")
        if self.capture:
            self.records.append({name: value.detach().cpu().to(torch.float64) for name, value in summaries.items()})
        self.placements.append({"input_plane_id": field.plane.plane_id, "output_plane_id": field.plane.plane_id, "input_shape": list(field.plane.shape), "output_shape": list(field.plane.shape), "input_coordinate_frame": field.plane.coordinate_frame, "input_provenance_operation": field.provenance.operation, "input_provenance_parameters": dict(field.provenance.parameters), "crop": False, "pad": False, "resample": False, "new_diffraction_plane": False, "material_applied": self.nonlinear, "tau": self.tau})
        if not self.nonlinear:
            return field
        provenance_type = type(field.provenance)
        provenance = provenance_type("long_fixed_ideal_facet_relu", field.plane.plane_id, None, {"intensity_threshold": self.tau, "intensity_slope": MATERIAL_SLOPE, "intensity_gain": MATERIAL_GAIN, "phase_shift": 0.0, "algorithmic_surrogate": True})
        return type(field)(output_tensor, field.plane, provenance)


def _install_probe(consumer: Any, *, mode: str, tau: float = 0.0, capture: bool = False) -> LongFacetProbe:
    generator = consumer.runtime.generator
    original_forward = generator.forward
    probe = LongFacetProbe(mode=mode, tau=tau, capture=capture)

    def forward_with_facet(field: Any, prefix: str) -> Any:
        return original_forward(probe.apply(field), prefix)

    generator.forward = forward_with_facet
    return probe


def _load_source(root: Path) -> dict[str, Any]:
    report_path = _path(root, N5_A_REPORT_PATH)
    checkpoint_path = _path(root, N5_A_CHECKPOINT_PATH)
    _validate_sha(report_path, N5_A_REPORT_SHA256, "N5-A report")
    _validate_sha(checkpoint_path, N5_A_CHECKPOINT_SHA256, "N5-A checkpoint")
    report = _read_json(report_path, "N5-A report")
    _require(report.get("result") == "VALID_COMPLETED" and report.get("status") == "valid_completed", "N5-A report is not valid")
    _require(report.get("candidate") == "A_CONTINUED_C1" and report.get("scientific_updates") == 300, "N5-A report identity changed")
    endpoint = _mapping(report.get("checkpoints", {}).get("300"), "N5-A endpoint")
    _require(endpoint.get("path") == N5_A_CHECKPOINT_PATH.as_posix() and endpoint.get("sha256") == N5_A_CHECKPOINT_SHA256, "N5-A endpoint reference changed")
    checkpoint = _mapping(N4.safe_torch_load(checkpoint_path, expected_sha256=N5_A_CHECKPOINT_SHA256, map_location="cpu"), "N5-A checkpoint")
    _require(checkpoint.get("candidate") == "A_CONTINUED_C1" and checkpoint.get("checkpoint_step") == 300 and checkpoint.get("global_step") == 600, "N5-A checkpoint step changed")
    _require(checkpoint.get("rank") == ADAPTER_RANK and checkpoint.get("trainable_parameter_names") == TRAINABLE_PARAMETER_NAMES, "N5-A trainable surface changed")
    _require(_digest_tree(checkpoint["runtime_state"]) == endpoint.get("runtime_state_digest"), "N5-A runtime state digest changed")
    _require(_digest_tree(checkpoint["adapter_state"]) == endpoint.get("adapter_state_digest"), "N5-A adapter state digest changed")
    return {"report": report, "checkpoint": checkpoint, "report_ref": {"path": _relative(root, report_path), "sha256": N5_A_REPORT_SHA256}, "checkpoint_ref": {"path": _relative(root, checkpoint_path), "sha256": N5_A_CHECKPOINT_SHA256}}


def _resolve_bundle(root: Path) -> dict[str, Any]:
    n5_contract, _c1, _baseline, authority_context, _extra = N5._validate_n5_authorities_v2(root)
    bundle = N4._validate_v11_authorities(root)
    return {"n5_contract": n5_contract, "bundle": bundle, "authority_context": authority_context}


def _new_candidate(bundle: Mapping[str, Any], source: Mapping[str, Any], device: torch.device, *, mode: str, tau: float, trainable: bool = True) -> tuple[Any, Any, LongFacetProbe]:
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V10.V8._load_runtime_state(consumer, dict(source["checkpoint"]["runtime_state"]))
    adapter = V11.LowRankAdapter(device=device)
    adapter.load_state_dict(dict(source["checkpoint"]["adapter_state"]), strict=True)
    N5._configure_candidate(consumer, adapter, "A_CONTINUED_C1", trainable=trainable)
    probe = _install_probe(consumer, mode=mode, tau=tau)
    return consumer, adapter, probe


def _optimizer(consumer: Any, adapter: Any) -> torch.optim.Adam:
    optimizer = N5._optimizer(consumer, adapter, "A_CONTINUED_C1")
    _require(optimizer.param_groups[0]["params"] == list(N5._candidate_named(consumer, adapter, "A_CONTINUED_C1").values()), "long optimizer parameter order changed")
    return optimizer


def _restore_optimizer(optimizer: torch.optim.Adam, state: Mapping[str, Any]) -> None:
    optimizer.load_state_dict(copy.deepcopy(dict(state)))
    _require(len(optimizer.state) == len(optimizer.param_groups[0]["params"]), "long optimizer state restoration is incomplete")


def _set_stage(consumer: Any, adapter: Any, *, coadapt: bool) -> str:
    named = N5._candidate_named(consumer, adapter, "A_CONTINUED_C1")
    for name, parameter in named.items():
        parameter.requires_grad_(name == "generator.phase" or coadapt)
        parameter.grad = None
    consumer.runtime.train()
    return "adapter_decoder_coadaptation" if coadapt else "decoder_only_refinement"


def _schedule_lr(step: int, total_steps: int) -> tuple[float, str]:
    switch = int(total_steps * 0.8)
    if step <= switch:
        fraction = step / float(max(switch, 1))
        return INITIAL_LR + fraction * (MID_LR - INITIAL_LR), "adapter_decoder_coadaptation"
    fraction = (step - switch) / float(max(total_steps - switch, 1))
    return MID_LR + fraction * (FINAL_LR - MID_LR), "decoder_only_refinement"


def _set_lr(optimizer: torch.optim.Adam, step: int, total_steps: int) -> tuple[float, str]:
    lr, stage = _schedule_lr(step, total_steps)
    optimizer.param_groups[0]["lr"] = lr
    return lr, stage


def _validate_adapted_result(result: Any, batch_size: int, *, require_grad: bool) -> torch.Tensor:
    return V10._validate_adapted_result(result, batch_size, require_grad=require_grad)


def _target(runtime: Any, batch: torch.Tensor) -> torch.Tensor:
    value = runtime.input_adapter(batch).slm_intensity
    _finite(value, "cached target")
    return value


def _cached_components(consumer: Any, adapter: Any, z: torch.Tensor, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, *, require_grad: bool) -> dict[str, Any]:
    _require(z.dtype == torch.complex64 and z.ndim == 2 and z.shape[1] == OLS_PORT_COUNT, "cached OLS representation shape/dtype changed")
    z_adapted = adapter(z)
    if require_grad and not any(parameter.requires_grad for parameter in adapter.parameters()):
        z_adapted = z_adapted.detach().requires_grad_(True)
    base_result = consumer.runtime.forward_from_ols(z_adapted)
    raw_base = _validate_adapted_result(base_result, int(batch.shape[0]), require_grad=require_grad)
    targets = (labels + 1) % CLASS_COUNT
    support = prototypes.to(device=z.device)
    z_shift = z - support.index_select(0, labels) + support.index_select(0, targets)
    z_shift_adapted = adapter(z_shift)
    if require_grad and not any(parameter.requires_grad for parameter in adapter.parameters()):
        z_shift_adapted = z_shift_adapted.detach().requires_grad_(True)
    shifted_result = consumer.runtime.forward_from_ols(z_shift_adapted)
    raw_shift = _validate_adapted_result(shifted_result, int(batch.shape[0]), require_grad=require_grad)
    distances = V10.V8._image_distances(raw_base, templates)
    anchor_logits = -distances / (N6.S_TAU + 1.0e-12)
    anchor_loss = F.cross_entropy(anchor_logits, labels)
    base_margin, shifted_margin = V10.V7._template_margins_differentiable(raw_base, raw_shift, templates, labels, targets)
    delta = shifted_margin - base_margin
    transfer_loss = F.softplus(-delta).mean()
    target = _target(consumer.runtime, batch)
    image_loss = (raw_base - target).square().mean()
    semantic_loss = anchor_loss + LAMBDA_TRANSFER * transfer_loss
    total_loss = semantic_loss + LAMBDA_IMAGE * image_loss
    for name, value in (("anchor_loss", anchor_loss), ("transfer_loss", transfer_loss), ("image_loss", image_loss), ("semantic_loss", semantic_loss), ("total_loss", total_loss), ("delta", delta), ("raw_shift", raw_shift)):
        _finite(value, f"cached {name}")
    return {"z": z, "z_adapted": z_adapted, "raw_base": raw_base, "raw_shift": raw_shift, "target": target, "targets": targets, "distances": distances, "anchor_logits": anchor_logits, "anchor_loss": anchor_loss, "transfer_loss": transfer_loss, "image_loss": image_loss, "semantic_loss": semantic_loss, "total_loss": total_loss, "delta": delta}


def _components(consumer: Any, adapter: Any, z: torch.Tensor, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, *, use_cache: bool, require_grad: bool) -> dict[str, Any]:
    if use_cache:
        return _cached_components(consumer, adapter, z, batch, labels, prototypes, templates, require_grad=require_grad)
    with torch.no_grad():
        direct = consumer.runtime.forward_from_image(batch)
        z_direct = direct.z_ols.coefficients.detach()
    return _cached_components(consumer, adapter, z_direct, batch, labels, prototypes, templates, require_grad=require_grad)


def _train_metric(consumer: Any, adapter: Any, z: torch.Tensor, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, *, use_cache: bool) -> dict[str, float]:
    with torch.no_grad():
        components = _components(consumer, adapter, z, batch, labels, prototypes, templates, use_cache=use_cache, require_grad=False)
    return {name: float(components[key].detach().cpu()) for name, key in (("L_anchor", "anchor_loss"), ("L_transfer", "transfer_loss"), ("L_image", "image_loss"), ("L_sem", "semantic_loss"), ("L_total", "total_loss"))}


def _norm(value: torch.Tensor, name: str) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result), f"{name} norm is nonfinite")
    return result


def _train_step(consumer: Any, adapter: Any, optimizer: torch.optim.Adam, z: torch.Tensor, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, *, step: int, total_steps: int, use_cache: bool) -> dict[str, Any]:
    lr, stage = _set_lr(optimizer, step, total_steps)
    _set_stage(consumer, adapter, coadapt=stage == "adapter_decoder_coadaptation")
    named = N5._candidate_named(consumer, adapter, "A_CONTINUED_C1")
    before = {name: parameter.detach().cpu().clone() for name, parameter in named.items()}
    frozen_runtime_before = N5._frozen_runtime_digest(consumer, "A_CONTINUED_C1")
    optimizer.zero_grad(set_to_none=True)
    components = _components(consumer, adapter, z, batch, labels, prototypes, templates, use_cache=use_cache, require_grad=True)
    total_loss = components["total_loss"]
    _require(total_loss.requires_grad and components["semantic_loss"].requires_grad and components["image_loss"].requires_grad, "long objective path is detached")
    total_loss.backward()
    gradients: dict[str, float | None] = {}
    for name, parameter in named.items():
        if parameter.requires_grad:
            _require(parameter.grad is not None, f"long {name} gradient is missing")
            gradients[name] = _norm(parameter.grad, f"long {name} gradient")
        else:
            _require(parameter.grad is None, f"long frozen {name} received a gradient")
            gradients[name] = None
    _require(any(value is not None and value > 0.0 for value in gradients.values()), "long combined gradient is zero")
    _require(all(parameter.grad is None for name, parameter in consumer.runtime.named_parameters() if name != "generator.phase"), "long frozen runtime received a gradient")
    optimizer.step()
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    _require(frozen_runtime_before == N5._frozen_runtime_digest(consumer, "A_CONTINUED_C1"), "long frozen runtime state changed")
    updates = {name: float((parameter.detach().cpu() - before[name]).to(torch.float64).square().sum().sqrt()) for name, parameter in named.items()}
    for name, parameter in named.items():
        if parameter.requires_grad:
            _require(math.isfinite(updates[name]) and updates[name] > 0.0, f"long trainable group did not update: {name}")
        else:
            _require(updates[name] == 0.0, f"long frozen group updated: {name}")
    return {"step": step, "global_step": 600 + step, "stage": stage, "learning_rate": lr, "L_anchor": float(components["anchor_loss"].detach().cpu()), "L_transfer": float(components["transfer_loss"].detach().cpu()), "L_image": float(components["image_loss"].detach().cpu()), "L_sem": float(components["semantic_loss"].detach().cpu()), "L_total": float(total_loss.detach().cpu()), "gradient_norms": gradients, "update_norms": updates, "trainable_parameter_names": [name for name, parameter in named.items() if parameter.requires_grad], "adapter_updated": stage == "adapter_decoder_coadaptation", "generator_updated": True, "encoder_updated": False, "ols_updated": False}


def _build_cache(root: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, dev_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=False)
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, _adapter, _probe = _new_candidate(bundle, source, device=device, mode="linear", tau=0.0, trainable=False)
    ordered_ids = torch.cat([train_indices.cpu(), dev_indices.cpu()])
    chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, ordered_ids.numel(), LOGICAL_BATCH_SIZE):
            ids = ordered_ids[start : start + LOGICAL_BATCH_SIZE]
            result = consumer.runtime.forward_from_image(images.index_select(0, ids).to(device))
            z = result.z_ols.coefficients.detach().cpu()
            _require(z.dtype == torch.complex64 and z.shape == (ids.numel(), OLS_PORT_COUNT), "cache z shape/dtype changed")
            _finite(z.real, "cache z real")
            _finite(z.imag, "cache z imag")
            chunks.append(z)
            if start == 0 or (start // LOGICAL_BATCH_SIZE) % 500 == 0:
                print(json.dumps({"event": "cache_progress", "samples": int(min(start + LOGICAL_BATCH_SIZE, ordered_ids.numel())), "total": int(ordered_ids.numel())}, ensure_ascii=False), flush=True)
    z_all = torch.cat(chunks, dim=0).contiguous()
    payload = {"schema": "lightgen_2d_ideal_facet_relu_long_frozen_ols_cache.v1", "version": 1, "task_id": TASK_ID, "sample_ids": ordered_ids, "train_sample_ids": train_indices.cpu(), "dev_sample_ids": dev_indices.cpu(), "labels": labels.index_select(0, ordered_ids).cpu(), "train_count": int(train_indices.numel()), "dev_count": int(dev_indices.numel()), "z_ols": z_all, "source_checkpoint": source["checkpoint_ref"], "encoder_frozen": True, "elapsed_seconds": time.perf_counter() - started}
    N4.atomic_torch_save_new(CACHE_PATH, payload, artifact_kind="long frozen OLS cache")
    digest = _sha256(CACHE_PATH)
    loaded = _mapping(N4.safe_torch_load(CACHE_PATH, expected_sha256=digest, map_location="cpu"), "long frozen OLS cache reload")
    _require(torch.equal(loaded["sample_ids"], ordered_ids) and torch.equal(loaded["labels"], payload["labels"]) and torch.equal(loaded["z_ols"], z_all), "cache reload changed sample identity or z")
    return {"path": _relative(root, CACHE_PATH), "sha256": digest, "sample_count": int(ordered_ids.numel()), "shape": list(z_all.shape), "elapsed_seconds": payload["elapsed_seconds"], "z_digest": _digest_tree(z_all), "train_sample_ids_sha256": V10.V6.tensor_sha256(train_indices.cpu()), "dev_sample_ids_sha256": V10.V6.tensor_sha256(dev_indices.cpu())}


def _cache_views(cache: Mapping[str, Any], train_indices: torch.Tensor, dev_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _require(torch.equal(cache["train_sample_ids"], train_indices.cpu()) and torch.equal(cache["dev_sample_ids"], dev_indices.cpu()), "cache TRAIN/DEV role order changed")
    count = int(cache["train_count"])
    _require(count == int(train_indices.numel()) and int(cache["dev_count"]) == int(dev_indices.numel()), "cache role counts changed")
    return cache["z_ols"][:count], cache["z_ols"][count:]


def _direct_placeholder_cache(train_indices: torch.Tensor, dev_indices: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """Keep the production interfaces shaped while deliberately disabling OLS caching."""
    sample_ids = torch.cat([train_indices.cpu(), dev_indices.cpu()])
    return {
        "schema": "lightgen_2d_ideal_facet_relu_long_direct_path_placeholder.v1",
        "sample_ids": sample_ids,
        "train_sample_ids": train_indices.cpu(),
        "dev_sample_ids": dev_indices.cpu(),
        "labels": labels.index_select(0, sample_ids).cpu(),
        "train_count": int(train_indices.numel()),
        "dev_count": int(dev_indices.numel()),
        "z_ols": torch.zeros((sample_ids.numel(), OLS_PORT_COUNT), dtype=torch.complex64),
    }


def _calibration_indices(train_indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(CALIBRATION_SEED)
    parts = []
    train_labels = labels.index_select(0, train_indices)
    for digit in range(CLASS_COUNT):
        candidates = train_indices[train_labels == digit].cpu()
        _require(candidates.numel() >= CALIBRATION_SAMPLES_PER_CLASS, f"calibration class {digit} is too small")
        order = torch.randperm(candidates.numel(), generator=generator)[:CALIBRATION_SAMPLES_PER_CLASS]
        parts.append(candidates.index_select(0, order))
    result = torch.cat(parts)
    _require(result.numel() == CLASS_COUNT * CALIBRATION_SAMPLES_PER_CLASS, "calibration sample count changed")
    return result


def _tau_for_retention(values: torch.Tensor, target: float) -> tuple[float, float]:
    values = values.to(torch.float64).flatten()
    _finite(values, "calibration intensity")
    total = float(values.sum())
    _require(total > 0.0, "calibration intensity power is zero")
    lower, upper = 0.0, float(values.max())
    for _ in range(80):
        middle = (lower + upper) / 2.0
        retention = float(torch.relu(values - middle).sum()) / total
        if retention > target:
            lower = middle
        else:
            upper = middle
    tau = (lower + upper) / 2.0
    return tau, float(torch.relu(values - tau).sum()) / total


def _forward_cached_batches(consumer: Any, adapter: Any, z_train: torch.Tensor, positions: torch.Tensor, device: torch.device) -> None:
    with torch.no_grad():
        for start in range(0, positions.numel(), LOGICAL_BATCH_SIZE):
            z = z_train.index_select(0, positions[start : start + LOGICAL_BATCH_SIZE]).to(device)
            consumer.runtime.forward_from_ols(adapter(z))


def _measure_retention(consumer: Any, adapter: Any, probe: LongFacetProbe, z_train: torch.Tensor, positions: torch.Tensor, device: torch.device, tau: float) -> tuple[float, float, float]:
    probe.tau = float(tau)
    probe.clear()
    probe.capture = True
    _forward_cached_batches(consumer, adapter, z_train, positions, device)
    _require(len(probe.records) == math.ceil(positions.numel() / LOGICAL_BATCH_SIZE), "calibration facet record count changed")
    input_power = torch.cat([record["input_power"] for record in probe.records]).to(torch.float64)
    output_power = torch.cat([record["output_power"] for record in probe.records]).to(torch.float64)
    total_input = float(input_power.sum())
    total_output = float(output_power.sum())
    _require(total_input > 0.0, "calibration input power is zero")
    return total_output / total_input, total_input, float(probe.input_max)


def _solve_exact_retention(consumer: Any, adapter: Any, probe: LongFacetProbe, z_train: torch.Tensor, positions: torch.Tensor, device: torch.device, *, target: float, guess: float, upper: float) -> tuple[float, float]:
    guess_retention, _total_input, _max_input = _measure_retention(consumer, adapter, probe, z_train, positions, device, guess)
    lower = 0.0 if guess_retention <= target else guess
    upper_bound = guess if guess_retention <= target else upper
    for _ in range(CALIBRATION_REFINEMENT_ITERS):
        middle = (lower + upper_bound) / 2.0
        retention, _total_input, _max_input = _measure_retention(consumer, adapter, probe, z_train, positions, device, middle)
        if retention > target:
            lower = middle
        else:
            upper_bound = middle
    tau = (lower + upper_bound) / 2.0
    retention, _total_input, _max_input = _measure_retention(consumer, adapter, probe, z_train, positions, device, tau)
    _require(abs(retention - target) <= RETENTION_TOLERANCE, f"calibration retention target not met: {retention} vs {target}")
    return tau, retention


def _calibrate_thresholds(root: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], labels: torch.Tensor, train_indices: torch.Tensor, z_train: torch.Tensor, device: torch.device) -> dict[str, Any]:
    calibration_ids = _calibration_indices(train_indices, labels)
    positions = _cache_positions(_train_position_map(train_indices), calibration_ids)
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, adapter, probe = _new_candidate(bundle, source, device=device, mode="relu", tau=0.0, trainable=False)
    probe.capture = True
    probe.capture_samples = True
    _forward_cached_batches(consumer, adapter, z_train, positions, device)
    _require(len(probe.input_samples) == math.ceil(calibration_ids.numel() / LOGICAL_BATCH_SIZE), "calibration sample capture count changed")
    sampled_values = torch.cat(probe.input_samples, dim=0).to(torch.float64)
    upper = float(probe.input_max)
    _require(upper > 0.0, "calibration input intensity maximum is zero")
    tau90_guess, _ = _tau_for_retention(sampled_values, 0.90)
    tau90, retention90 = _solve_exact_retention(consumer, adapter, probe, z_train, positions, device, target=0.90, guess=tau90_guess, upper=upper)
    return {"seed": CALIBRATION_SEED, "samples_per_class": CALIBRATION_SAMPLES_PER_CLASS, "sample_count": int(calibration_ids.numel()), "sample_ids": [int(value) for value in calibration_ids.tolist()], "sample_ids_sha256": V10.V6.tensor_sha256(calibration_ids), "source": "TRAIN-only; balanced; frozen N5-A endpoint; no DEV score selection", "spatial_sampling": {"stride": CALIBRATION_PIXEL_STRIDE, "purpose": "bounded initial threshold estimate only; final retention is measured on all facet pixels"}, "tau_90": tau90, "initial_power_retention_tau_90": retention90, "input_power": float(_measure_retention(consumer, adapter, probe, z_train, positions, device, 0.0)[1]), "input_intensity_quantiles": {key: float(value) for key, value in zip(("min", "q25", "median", "q75", "max"), torch.quantile(sampled_values.flatten(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)).tolist(), strict=True)}, "upper_intensity_observed": upper}


def _forward_image_batches(consumer: Any, images: torch.Tensor, sample_ids: torch.Tensor, device: torch.device) -> None:
    with torch.no_grad():
        for start in range(0, sample_ids.numel(), LOGICAL_BATCH_SIZE):
            ids = sample_ids[start : start + LOGICAL_BATCH_SIZE]
            consumer.runtime.forward_from_image(images.index_select(0, ids).to(device))


def _measure_image_retention(consumer: Any, probe: LongFacetProbe, images: torch.Tensor, sample_ids: torch.Tensor, device: torch.device, tau: float) -> tuple[float, float, float]:
    probe.tau = float(tau)
    probe.clear()
    probe.capture = True
    _forward_image_batches(consumer, images, sample_ids, device)
    expected = math.ceil(sample_ids.numel() / LOGICAL_BATCH_SIZE)
    _require(len(probe.records) == expected, "direct calibration facet record count changed")
    input_power = torch.cat([record["input_power"] for record in probe.records]).to(torch.float64)
    output_power = torch.cat([record["output_power"] for record in probe.records]).to(torch.float64)
    total_input = float(input_power.sum())
    _require(total_input > 0.0, "direct calibration input power is zero")
    return float(output_power.sum()) / total_input, total_input, float(probe.input_max)


def _calibrate_thresholds_direct(root: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Fast TRAIN-only threshold calibration without materializing a 60k-sample cache."""
    calibration_ids = _calibration_indices(train_indices, labels)
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, _adapter, probe = _new_candidate(bundle, source, device=device, mode="relu", tau=0.0, trainable=False)
    probe.capture_samples = True
    _forward_image_batches(consumer, images, calibration_ids, device)
    _require(len(probe.input_samples) == math.ceil(calibration_ids.numel() / LOGICAL_BATCH_SIZE), "direct calibration sample capture count changed")
    sampled_values = torch.cat(probe.input_samples, dim=0).to(torch.float64)
    upper = float(probe.input_max)
    _require(upper > 0.0, "direct calibration input intensity maximum is zero")
    tau_guess, _sampled_retention = _tau_for_retention(sampled_values, 0.90)
    retention_guess, input_power, _ = _measure_image_retention(consumer, probe, images, calibration_ids, device, tau_guess)
    lower, upper_bound = (0.0, tau_guess) if retention_guess <= 0.90 else (tau_guess, upper)
    for _ in range(FAST_DIRECT_REFINEMENT_ITERS):
        middle = (lower + upper_bound) / 2.0
        retention_middle, input_power, _ = _measure_image_retention(consumer, probe, images, calibration_ids, device, middle)
        if retention_middle > 0.90:
            lower = middle
        else:
            upper_bound = middle
    tau90 = (lower + upper_bound) / 2.0
    retention90, input_power, _ = _measure_image_retention(consumer, probe, images, calibration_ids, device, tau90)
    _require(math.isfinite(retention90) and 0.0 <= retention90 <= 1.0, "direct calibration retention is invalid")
    return {"seed": CALIBRATION_SEED, "samples_per_class": CALIBRATION_SAMPLES_PER_CLASS, "sample_count": int(calibration_ids.numel()), "sample_ids": [int(value) for value in calibration_ids.tolist()], "sample_ids_sha256": V10.V6.tensor_sha256(calibration_ids), "source": "TRAIN-only; balanced; fast direct-path calibration; no DEV score selection", "spatial_sampling": {"stride": CALIBRATION_PIXEL_STRIDE, "purpose": "fast deterministic threshold estimate; direct path; no full OLS cache"}, "tau_90": float(tau90), "initial_power_retention_tau_90": float(retention90), "input_power": float(input_power), "input_intensity_quantiles": {key: float(value) for key, value in zip(("min", "q25", "median", "q75", "max"), torch.quantile(sampled_values.flatten(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)).tolist(), strict=True)}, "upper_intensity_observed": upper}


def _cache_equivalence(bundle: Mapping[str, Any], source: Mapping[str, Any], cache: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, dev_indices: torch.Tensor, calibration: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    z_train, _ = _cache_views(cache, train_indices, dev_indices)
    position_map = _train_position_map(train_indices)
    sample_ids = torch.tensor([int(value) for value in calibration["sample_ids"][:LOGICAL_BATCH_SIZE]], dtype=torch.long)
    positions = _cache_positions(position_map, sample_ids)
    batch = images.index_select(0, sample_ids).to(device)
    batch_labels = labels.index_select(0, sample_ids).to(device)
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    checks: dict[str, Any] = {}
    for mode, tau, label in (("linear", 0.0, "A_LINEAR"), ("relu", float(calibration["tau_90"]), "B_RELU90")):
        V10.V6._restore_rng(source["checkpoint"]["rng_state"])
        full_consumer, full_adapter, _ = _new_candidate(bundle, source, device=device, mode=mode, tau=tau, trainable=True)
        V10.V6._restore_rng(source["checkpoint"]["rng_state"])
        cache_consumer, cache_adapter, _ = _new_candidate(bundle, source, device=device, mode=mode, tau=tau, trainable=True)
        with torch.no_grad():
            full_result = full_consumer.runtime.forward_from_image(batch)
            z_full = full_result.z_ols.coefficients.detach()
        z_cached = z_train.index_select(0, positions).to(device)
        _require(torch.equal(z_full, z_cached), f"{label} cache z differs from full path")
        full_components = N5._objective_components(full_consumer, full_adapter, batch, batch_labels, prototypes, templates, require_grad=True)
        cache_components = _cached_components(cache_consumer, cache_adapter, z_cached, batch, batch_labels, prototypes, templates, require_grad=True)
        for key in ("raw_base", "raw_shift", "anchor_loss", "transfer_loss", "image_loss", "total_loss"):
            _require(torch.allclose(full_components[key].detach(), cache_components[key].detach(), rtol=1.0e-5, atol=1.0e-6), f"{label} cache {key} differs from full path")
        full_grad = torch.autograd.grad(full_components["total_loss"], tuple(N5._candidate_named(full_consumer, full_adapter, "A_CONTINUED_C1").values()), retain_graph=False, allow_unused=False)
        cache_grad = torch.autograd.grad(cache_components["total_loss"], tuple(N5._candidate_named(cache_consumer, cache_adapter, "A_CONTINUED_C1").values()), retain_graph=False, allow_unused=False)
        gradient_deltas = [float((left - right).detach().to(torch.float64).abs().max().cpu()) for left, right in zip(full_grad, cache_grad, strict=True)]
        _require(all(delta <= 1.0e-5 for delta in gradient_deltas), f"{label} cache gradient differs from full path")
        full_optimizer, cache_optimizer = _optimizer(full_consumer, full_adapter), _optimizer(cache_consumer, cache_adapter)
        _restore_optimizer(full_optimizer, source["checkpoint"]["optimizer_state"])
        _restore_optimizer(cache_optimizer, source["checkpoint"]["optimizer_state"])
        _set_stage(full_consumer, full_adapter, coadapt=True)
        _set_stage(cache_consumer, cache_adapter, coadapt=True)
        _set_lr(full_optimizer, 1, K_MAX)
        _set_lr(cache_optimizer, 1, K_MAX)
        full_optimizer.zero_grad(set_to_none=True)
        cache_optimizer.zero_grad(set_to_none=True)
        full_components = N5._objective_components(full_consumer, full_adapter, batch, batch_labels, prototypes, templates, require_grad=True)
        cache_components = _cached_components(cache_consumer, cache_adapter, z_cached, batch, batch_labels, prototypes, templates, require_grad=True)
        full_components["total_loss"].backward()
        cache_components["total_loss"].backward()
        full_optimizer.step()
        cache_optimizer.step()
        full_state = _cpu_tree({**dict(full_consumer.runtime.state_dict()), **{f"adapter.{key}": value for key, value in full_adapter.state_dict().items()}})
        cache_state = _cpu_tree({**dict(cache_consumer.runtime.state_dict()), **{f"adapter.{key}": value for key, value in cache_adapter.state_dict().items()}})
        _require(_tree_equal(full_state, cache_state), f"{label} cache one-step parameter update differs from full path")
        checks[label] = {"z_equal": True, "outputs_losses_equal": True, "gradient_max_abs_deltas": gradient_deltas, "one_step_update_equal": True}
        del full_consumer, full_adapter, cache_consumer, cache_adapter
        gc.collect()
        torch.cuda.empty_cache()
    return {"status": "PASS", "checks": checks, "sample_ids": [int(value) for value in sample_ids.tolist()]}


def _preflight_smoke(bundle: Mapping[str, Any], source: Mapping[str, Any], z_train: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, *, mode: str, tau: float, steps: int, use_cache: bool) -> dict[str, Any]:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, adapter, _ = _new_candidate(bundle, source, device=device, mode=mode, tau=tau, trainable=True)
    optimizer = _optimizer(consumer, adapter)
    _restore_optimizer(optimizer, source["checkpoint"]["optimizer_state"])
    sampler = BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=max(steps, PRECHECK_STEPS))
    position_map = _train_position_map(train_indices)
    traces = []
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    peek_indices, peek_labels = sampler.peek_batch()
    peek_positions = _cache_positions(position_map, peek_indices)
    peek_batch = images.index_select(0, peek_indices).to(device)
    peek_z = z_train.index_select(0, peek_positions).to(device)
    negative_components = _components(consumer, adapter, peek_z, peek_batch, peek_labels.to(device), prototypes, templates, use_cache=use_cache, require_grad=True)
    wrong_anchor = F.cross_entropy(negative_components["anchor_logits"], (peek_labels.to(device) + 1) % CLASS_COUNT) + LAMBDA_IMAGE * negative_components["image_loss"]
    wrong_transfer = LAMBDA_TRANSFER * F.softplus(negative_components["delta"]).mean() + LAMBDA_IMAGE * negative_components["image_loss"]
    negative_controls = {}
    named = N5._candidate_named(consumer, adapter, "A_CONTINUED_C1")
    for name, loss in (("wrong_label_anchor", wrong_anchor), ("wrong_direction_transfer", wrong_transfer)):
        gradients = torch.autograd.grad(loss, tuple(named.values()), retain_graph=name == "wrong_label_anchor", allow_unused=False)
        norms = [_norm(gradient, f"long {mode} {name} {parameter_name}") for parameter_name, gradient in zip(named, gradients, strict=True)]
        _require(any(value > 0.0 for value in norms), f"long {mode} negative control gradient is zero")
        negative_controls[name] = {parameter_name: value for parameter_name, value in zip(named, norms, strict=True)}
    started = time.perf_counter()
    for step in range(1, PRECHECK_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        positions = _cache_positions(position_map, indices)
        traces.append(_train_step(consumer, adapter, optimizer, z_train.index_select(0, positions).to(device), images.index_select(0, indices).to(device), batch_labels.to(device), prototypes, templates, step=step, total_steps=max(steps, K_MAX), use_cache=use_cache))
    elapsed = time.perf_counter() - started
    return {"mode": mode, "tau": tau, "steps": PRECHECK_STEPS, "traces": traces, "negative_controls": negative_controls, "elapsed_seconds": elapsed, "seconds_per_update": elapsed / PRECHECK_STEPS, "sampler_cursor": sampler.cursor, "scientific_updates": 0}


def _sampler_boundary_probe(train_indices: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    sampler = BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=10_000)
    per_class = sampler._per_class
    for _ in range(per_class + 1):
        sampler.next_batch()
    _require(sampler.epoch == 1 and sampler.position == 1 and sampler.cursor == per_class + 1, "sampler epoch boundary is invalid")
    state = _cpu_tree(sampler.state_dict())
    resumed = BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=10_000)
    resumed.load_state_dict(state)
    left, left_labels = sampler.next_batch()
    right, right_labels = resumed.next_batch()
    _require(torch.equal(left, right) and torch.equal(left_labels, right_labels), "sampler epoch resume order changed")
    return {"status": "PASS", "per_class_samples": per_class, "boundary_cursor": per_class, "post_boundary_cursor": sampler.cursor, "epoch": sampler.epoch, "resumed_batch_equal": True}


def _resume_equivalence(bundle: Mapping[str, Any], source: Mapping[str, Any], z_train: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, *, mode: str, tau: float, total_steps: int, use_cache: bool) -> dict[str, Any]:
    split = min(7, total_steps - 1)
    position_map = _train_position_map(train_indices)
    source_rng = _cpu_tree(source["checkpoint"]["rng_state"])
    V10.V6._restore_rng(source_rng)
    sampler_a = BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=total_steps)
    consumer_a, adapter_a, _ = _new_candidate(bundle, source, device=device, mode=mode, tau=tau, trainable=True)
    optimizer_a = _optimizer(consumer_a, adapter_a)
    _restore_optimizer(optimizer_a, source["checkpoint"]["optimizer_state"])
    V10.V6._restore_rng(source_rng)
    sampler_b = BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=total_steps)
    consumer_b, adapter_b, _ = _new_candidate(bundle, source, device=device, mode=mode, tau=tau, trainable=True)
    optimizer_b = _optimizer(consumer_b, adapter_b)
    _restore_optimizer(optimizer_b, source["checkpoint"]["optimizer_state"])
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    for step in range(1, split + 1):
        indices_a, labels_a = sampler_a.next_batch()
        indices_b, labels_b = sampler_b.next_batch()
        _require(torch.equal(indices_a, indices_b) and torch.equal(labels_a, labels_b), "sampler uninterrupted paths diverged before checkpoint")
        positions = _cache_positions(position_map, indices_a)
        batch_a = images.index_select(0, indices_a).to(device)
        batch_b = images.index_select(0, indices_b).to(device)
        z = z_train.index_select(0, positions).to(device)
        _train_step(consumer_a, adapter_a, optimizer_a, z, batch_a, labels_a.to(device), prototypes, templates, step=step, total_steps=total_steps, use_cache=use_cache)
        _train_step(consumer_b, adapter_b, optimizer_b, z, batch_b, labels_b.to(device), prototypes, templates, step=step, total_steps=total_steps, use_cache=use_cache)
    checkpoint = {"runtime": _cpu_tree(consumer_a.runtime.state_dict()), "adapter": _cpu_tree(adapter_a.state_dict()), "optimizer": _cpu_tree(optimizer_a.state_dict()), "sampler": _cpu_tree(sampler_a.state_dict()), "rng": _cpu_tree(V10.V6._capture_rng())}
    V10.V6._restore_rng(source_rng)
    resumed_consumer, resumed_adapter, _ = _new_candidate(bundle, source, device=device, mode=mode, tau=tau, trainable=True)
    resumed_consumer.runtime.load_state_dict(checkpoint["runtime"], strict=True)
    resumed_adapter.load_state_dict(checkpoint["adapter"], strict=True)
    resumed_optimizer = _optimizer(resumed_consumer, resumed_adapter)
    _restore_optimizer(resumed_optimizer, checkpoint["optimizer"])
    resumed_sampler = BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=total_steps)
    resumed_sampler.load_state_dict(checkpoint["sampler"])
    V10.V6._restore_rng(checkpoint["rng"])
    for step in range(split + 1, total_steps + 1):
        indices, batch_labels = sampler_a.next_batch()
        positions = _cache_positions(position_map, indices)
        batch = images.index_select(0, indices).to(device)
        _train_step(consumer_a, adapter_a, optimizer_a, z_train.index_select(0, positions).to(device), batch, batch_labels.to(device), prototypes, templates, step=step, total_steps=total_steps, use_cache=use_cache)
    rng_after_uninterrupted = _cpu_tree(V10.V6._capture_rng())
    V10.V6._restore_rng(checkpoint["rng"])
    for step in range(split + 1, total_steps + 1):
        indices, batch_labels = resumed_sampler.next_batch()
        positions = _cache_positions(position_map, indices)
        batch = images.index_select(0, indices).to(device)
        _train_step(resumed_consumer, resumed_adapter, resumed_optimizer, z_train.index_select(0, positions).to(device), batch, batch_labels.to(device), prototypes, templates, step=step, total_steps=total_steps, use_cache=use_cache)
    rng_after_resumed = _cpu_tree(V10.V6._capture_rng())
    _require(_tree_equal(consumer_a.runtime.state_dict(), resumed_consumer.runtime.state_dict()), "resumed runtime state differs")
    _require(_tree_equal(adapter_a.state_dict(), resumed_adapter.state_dict()), "resumed adapter state differs")
    _require(_tree_equal(optimizer_a.state_dict(), resumed_optimizer.state_dict()), "resumed optimizer state differs")
    _require(_tree_equal(sampler_a.state_dict(), resumed_sampler.state_dict()), "resumed sampler state differs")
    _require(_tree_equal(rng_after_uninterrupted, rng_after_resumed), "resumed RNG continuation differs")
    return {"status": "PASS", "split_step": split, "terminal_step": total_steps, "sample_order_equal": True, "model_equal": True, "optimizer_equal": True, "sampler_equal": True, "rng_equal": True}


def _save_json(path: Path, payload: Mapping[str, Any], name: str) -> dict[str, str]:
    N4.atomic_write_new_json(path, dict(payload), artifact_kind=name)
    return {"path": path.as_posix(), "sha256": _sha256(path)}


def _checkpoint_payload(root: Path, contract_ref: Mapping[str, Any], source: Mapping[str, Any], cache_ref: Mapping[str, Any], consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: BalancedMultiEpochSampler, *, mode: str, tau: float, step: int, total_steps: int, use_cache: bool, block_size: int, initial_phase: torch.Tensor) -> dict[str, Any]:
    lr, stage = _schedule_lr(step, total_steps)
    return {"schema": "lightgen_2d_ideal_facet_relu_long_checkpoint.v1", "version": 1, "task_id": TASK_ID, "candidate": mode, "contract": dict(contract_ref), "source_checkpoint": source["checkpoint_ref"], "cache": dict(cache_ref), "checkpoint_step": step, "global_step": 600 + step, "total_steps": total_steps, "runtime_state": _cpu_tree(consumer.runtime.state_dict()), "adapter_state": _cpu_tree(adapter.state_dict()), "optimizer_state": _cpu_tree(optimizer.state_dict()), "sampler_state": _cpu_tree(sampler.state_dict()), "rng_state": _cpu_tree(V10.V6._capture_rng()), "schedule_state": {"last_update": step, "stage": stage, "learning_rate": lr, "switch_step": int(total_steps * 0.8)}, "trainable_parameter_names": [name for name, parameter in N5._candidate_named(consumer, adapter, "A_CONTINUED_C1").items() if parameter.requires_grad], "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "facet", "Smooth evaluator", "templates", "prototypes"], "initial_generator_phase": _cpu_tree(initial_phase), "use_cache": use_cache, "block_size": block_size, "terminal_sampler_rule": "checkpoint step s saves sampler cursor=s and does not request s+1"}


def _save_checkpoint(root: Path, path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    N4.atomic_torch_save_new(path, dict(payload), artifact_kind=f"long {payload['candidate']} checkpoint step {payload['checkpoint_step']}")
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "long checkpoint reload")
    _require(loaded.get("schema") == payload.get("schema") and loaded.get("candidate") == payload.get("candidate") and loaded.get("checkpoint_step") == payload.get("checkpoint_step"), "long checkpoint identity changed on reload")
    for key in ("runtime_state", "adapter_state", "optimizer_state", "sampler_state", "rng_state", "schedule_state"):
        _require(_tree_equal(loaded[key], payload[key]), f"long checkpoint {key} changed on reload")
    sampler_state = _mapping(loaded["sampler_state"], "long checkpoint sampler")
    _require(int(sampler_state["cursor"]) == int(payload["checkpoint_step"]), "checkpoint sampler cursor is not terminal for its update index")
    return {"path": _relative(root, path), "sha256": digest, "step": int(payload["checkpoint_step"]), "global_step": int(payload["global_step"]), "reloaded": True}


def _save_tensor(root: Path, path: Path, payload: Mapping[str, Any], name: str) -> dict[str, str]:
    N4.atomic_torch_save_new(path, dict(payload), artifact_kind=name)
    digest = _sha256(path)
    loaded = N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu")
    _require(isinstance(loaded, Mapping) and loaded.get("schema") == payload.get("schema"), f"{name} reload schema changed")
    return {"path": _relative(root, path), "sha256": digest}


def _benchmark_dev(root: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, dev_indices: torch.Tensor, device: torch.device, *, mode: str, tau: float) -> dict[str, Any]:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, adapter, probe = _new_candidate(bundle, source, device=device, mode=mode, tau=tau, trainable=False)
    dev_images = images.index_select(0, dev_indices).contiguous()
    dev_labels = labels.index_select(0, dev_indices).contiguous()
    selection_images, selection_labels, selection_indices = V10.V6._load_selection(root, images, labels, dev_indices)
    probe.clear()
    started = time.perf_counter()
    with torch.no_grad():
        evaluation = N4._evaluate(consumer, adapter, bundle["authorities"]["smooth"], dev_images, dev_labels, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"], step=0, train_trace=None)
        cyclic = V10._cyclic_transfer_v10(consumer, adapter, bundle["authorities"]["smooth"], selection_images, selection_labels, selection_indices, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"])
    elapsed = time.perf_counter() - started
    _require(math.isfinite(elapsed) and elapsed > 0.0, "DEV benchmark time is invalid")
    return {"mode": mode, "tau": tau, "seconds": elapsed, "dev_rows": int(dev_images.shape[0]), "selection_rows": int(selection_images.shape[0]), "raw_validity": evaluation["raw_validity"], "facet_placements": len(probe.placements), "facet_placement": probe.placements[0] if probe.placements else {}}


def _benchmark_checkpoint_overhead(root: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], cache_ref: Mapping[str, Any], train_indices: torch.Tensor, labels: torch.Tensor, device: torch.device, *, mode: str, tau: float) -> dict[str, Any]:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, adapter, _probe = _new_candidate(bundle, source, device=device, mode="linear" if mode == "A_LINEAR" else "relu", tau=tau, trainable=True)
    optimizer = _optimizer(consumer, adapter)
    _restore_optimizer(optimizer, source["checkpoint"]["optimizer_state"])
    sampler = BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=K_MAX)
    _set_stage(consumer, adapter, coadapt=True)
    _set_lr(optimizer, 0, K_MAX)
    payload = _checkpoint_payload(root, {"path": "preflight_pending_contract", "sha256": "preflight_pending"}, source, cache_ref, consumer, adapter, optimizer, sampler, mode=mode, tau=tau, step=0, total_steps=K_MAX, use_cache=True, block_size=BLOCK_SIZE, initial_phase=source["checkpoint"]["runtime_state"]["generator.phase"])
    path = PREFLIGHT_DIR / f"benchmark_{mode.lower()}_checkpoint.v1.pt"
    started = time.perf_counter()
    reference = _save_checkpoint(root, path, payload)
    elapsed = time.perf_counter() - started
    return {"mode": mode, "seconds": elapsed, "reference": reference}


def _make_contract(root: Path, started: Mapping[str, Any], k: int, calibration: Mapping[str, Any], benchmark: Mapping[str, Any], cache_ref: Mapping[str, Any], *, cache_used: bool) -> dict[str, Any]:
    payload = {"schema": "lightgen_2d_ideal_facet_relu_long_contract.v1", "version": 1, "task_id": TASK_ID, "design_authority": DESIGN_AUTHORITY, "status": "READY_FOR_BOUNDED_LONG_SCIENTIFIC_EXECUTION", "outer_gate": OUTER_GATE, "git_boundary": {"branch": started["branch"], "head": started["head"]}, "source_n5_a_checkpoint": {"path": N5_A_CHECKPOINT_PATH.as_posix(), "sha256": N5_A_CHECKPOINT_SHA256, "global_step": 600}, "geometry": {"encoder_layers": 2, "decoder_layers": 2, "layer_spacing_m": 1.50e-3, "ols_grid": [10, 10], "ols_ports": 100, "ols_pixel_pitch_m": 2.0e-6, "wavelength_m": 532.0e-9, "propagation_grid": PRODUCTION_GRID, "new_diffraction_plane": False}, "path": "image -> amplitude SLM -> frozen optical encoder -> complex 100-port OLS -> rank-9 adapter T=I+UV^H -> W_OLS modal re-emission -> output-facet operator -> existing decoder -> raw CMOS", "facet_placement": "after lightgen_ols_coefficient_transmission_and_modal_reemission and before existing decoder", "objective": {"formula": "L=L_anchor+lambda_transfer*L_transfer+lambda_image*L_image", "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE}, "arms": {"executed": ["A_LINEAR", "B_RELU90"], "primary": "B_RELU90_vs_A_LINEAR"}, "facet": {"kind": "IDEAL_INTENSITY_RELU_SURROGATE", "formula": "E_out=E_in*sqrt((relu(|E_in|^2-tau)+eps)/(|E_in|^2+eps))", "gain": MATERIAL_GAIN, "slope": MATERIAL_SLOPE, "phase_shift": 0.0, "trainable": False, "physical_te_validation": False, "additional_propagation": False, "tau_90": calibration["tau_90"]}, "calibration": dict(calibration), "training": {"K": k, "batch": LOGICAL_BATCH_SIZE, "stage_switch": int(k * 0.8), "stage_schedule": "first 80% adapter+decoder; last 20% decoder only", "optimizer": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "lr_start": INITIAL_LR, "lr_switch": MID_LR, "lr_end": FINAL_LR, "sampler": BalancedMultiEpochSampler.SCHEMA, "sampler_seed": SAMPLER_SEED, "block_size": BLOCK_SIZE, "evaluation_fractions": list(DEV_CHECKPOINT_FRACTIONS)}, "budget": dict(benchmark), "cache": {**dict(cache_ref), "used": cache_used}, "data_roles": {"TRAIN": 55000, "DEV": 5000, "calibration": "TRAIN subset only", "protected": 0, "official_test": 0, "sealed": 0}, "claim_boundary": {"ideal_simulation_only": True, "physical_te_validation": False, "stage_b": False, "latent_prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "pause_resume": {"checkpoint_state": ["model", "optimizer", "sampler", "RNG", "schedule", "stage", "update_index"], "explicit_resume_required": True}, "no_retry": True}
    CONTRACT_DIR.mkdir(parents=True, exist_ok=False)
    N4.atomic_write_new_json(CONTRACT_PATH, payload, artifact_kind="long ideal facet ReLU contract")
    return {"payload": payload, "ref": {"path": _relative(root, CONTRACT_PATH), "sha256": _sha256(CONTRACT_PATH)}}


def _preflight_payload(started: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "lightgen_2d_ideal_facet_relu_long_preflight_report.v1", "version": 1, "task_id": TASK_ID, "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_LONG_IDEAL_FACET_RELU_PREFLIGHT", "created_at_utc": datetime.now(timezone.utc).isoformat(), "outer_gate": OUTER_GATE, "provenance": {"git_start": dict(started)}, "claim_boundary": {"engineering_preflight_executed": False, "scientific_training_executed": False, "scientific_updates": 0, "scientific_forward_evaluations": 0, "stage_b": False, "latent_prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}}


def run_preflight(root: Path) -> dict[str, Any]:
    if PREFLIGHT_DIR.exists():
        _require(not any(PREFLIGHT_DIR.iterdir()), f"preflight output already exists: {_path(root, PREFLIGHT_DIR)}")
    for path in (*ARM_DIRS.values(), AGGREGATE_DIR, CONTRACT_DIR):
        _require(not path.exists(), f"long run output already exists: {_path(root, path)}")
    if not PREFLIGHT_DIR.exists():
        PREFLIGHT_DIR.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = _status_lines(root)
    payload = _preflight_payload(started)
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from authorized plan")
        source = _load_source(root)
        authority = _resolve_bundle(root)
        bundle = authority["bundle"]
        V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        if device.type != "cuda":
            raise HardBlockedLongRun("BLOCKED_CUDA_UNAVAILABLE", "CUDA is required for this scientific run", device=device_facts)
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, dev_indices, dataset = V10.V6._load_roles(root)
        _require(int(train_indices.numel()) == 55_000 and int(dev_indices.numel()) == 5_000, "TRAIN/DEV role counts changed")
        _require(dataset.get("protected", 0) in (0, [], None) and dataset.get("sealed", 0) in (0, [], None), "protected or sealed role is nonempty")
        position_map = _train_position_map(train_indices)
        V10.V6._restore_rng(source["checkpoint"]["rng_state"])
        start_a, adapter_a, _ = _new_candidate(bundle, source, device=device, mode="linear", tau=0.0, trainable=True)
        opt_a = _optimizer(start_a, adapter_a)
        _restore_optimizer(opt_a, source["checkpoint"]["optimizer_state"])
        digest_a = _digest_tree({"runtime": start_a.runtime.state_dict(), "adapter": adapter_a.state_dict(), "optimizer": opt_a.state_dict()})
        V10.V6._restore_rng(source["checkpoint"]["rng_state"])
        start_b, adapter_b, _ = _new_candidate(bundle, source, device=device, mode="relu", tau=0.0, trainable=True)
        opt_b = _optimizer(start_b, adapter_b)
        _restore_optimizer(opt_b, source["checkpoint"]["optimizer_state"])
        digest_b = _digest_tree({"runtime": start_b.runtime.state_dict(), "adapter": adapter_b.state_dict(), "optimizer": opt_b.state_dict()})
        _require(digest_a == digest_b, "A/B common starting model or optimizer state differs")
        del start_a, adapter_a, opt_a, start_b, adapter_b, opt_b
        gc.collect()
        torch.cuda.empty_cache()
        if FAST_DIRECT_PREFLIGHT:
            cache_ref = {"path": "none", "sha256": "none", "sample_count": 0, "shape": [0, OLS_PORT_COUNT], "z_digest": None, "used": False, "mode": "direct_full_path"}
            cache = _direct_placeholder_cache(train_indices, dev_indices, labels)
            z_train, _z_dev = _cache_views(cache, train_indices, dev_indices)
            calibration = _calibrate_thresholds_direct(root, bundle, source, images, labels, train_indices, device)
            cache_equivalence = {"status": "SKIPPED_FAST_DIRECT", "reason": "full frozen-OLS cache and cache-equivalence replay disabled to reserve resources for A/B updates", "cache_used": False}
            use_cache = False
        else:
            cache_ref = _build_cache(root, bundle, source, images, labels, train_indices, dev_indices, device)
            cache = _mapping(N4.safe_torch_load(_path(root, cache_ref["path"]), expected_sha256=cache_ref["sha256"], map_location="cpu"), "long cache")
            z_train, _z_dev = _cache_views(cache, train_indices, dev_indices)
            calibration = _calibrate_thresholds(root, bundle, source, labels, train_indices, z_train, device)
            _require(abs(float(calibration["initial_power_retention_tau_90"]) - 0.90) <= RETENTION_TOLERANCE, "calibration retention target not met")
            use_cache = True
            try:
                cache_equivalence = _cache_equivalence(bundle, source, cache, images, labels, train_indices, dev_indices, calibration, device)
            except InvalidLongRun as error:
                use_cache = False
                cache_equivalence = {"status": "FAIL_DISABLED", "reason": str(error), "cache_used": False}
        benchmark: dict[str, Any] = {"cache": {"used": use_cache, "reference": cache_ref, "equivalence": cache_equivalence}}
        benchmark["smoke"] = {}
        for mode, tau, name in (("linear", 0.0, "A_LINEAR"), ("relu", float(calibration["tau_90"]), "B_RELU90")):
            benchmark["smoke"][name] = _preflight_smoke(bundle, source, z_train, images, labels, train_indices, device, mode=mode, tau=tau, steps=K_MAX, use_cache=use_cache)
        if FAST_DIRECT_PREFLIGHT:
            benchmark["dev"] = {"status": "SKIPPED_FAST_DIRECT", "seconds": 0.0, "reason": "DEV is reserved for step-0/final scientific endpoints"}
            benchmark["checkpoint"] = {"status": "SKIPPED_FAST_DIRECT", "seconds": 0.0, "reason": "checkpoint overhead is not measured before A/B"}
        else:
            benchmark["dev"] = {}
            for mode, tau, name in (("linear", 0.0, "A_LINEAR"), ("relu", float(calibration["tau_90"]), "B_RELU90")):
                benchmark["dev"][name] = _benchmark_dev(root, bundle, source, images, labels, dev_indices, device, mode=mode, tau=tau)
            benchmark["checkpoint"] = {}
            for mode, tau, name in (("linear", 0.0, "A_LINEAR"), ("relu", float(calibration["tau_90"]), "B_RELU90")):
                benchmark["checkpoint"][name] = _benchmark_checkpoint_overhead(root, bundle, source, cache_ref, train_indices, labels, device, mode=name, tau=tau)
        sampler_probe = _sampler_boundary_probe(train_indices, labels)
        resume = _resume_equivalence(bundle, source, z_train, images, labels, train_indices, device, mode="relu", tau=float(calibration["tau_90"]), total_steps=1 if FAST_DIRECT_PREFLIGHT else 20, use_cache=use_cache)
        step_seconds = max(float(value["seconds_per_update"]) for value in benchmark["smoke"].values()) * 1.25
        dev_seconds = 0.0 if FAST_DIRECT_PREFLIGHT else max(float(value["seconds"]) for value in benchmark["dev"].values())
        checkpoint_seconds = 0.0 if FAST_DIRECT_PREFLIGHT else max(float(value["seconds"]) for value in benchmark["checkpoint"].values())
        def estimated_active_seconds(k: int) -> float:
            block_count = math.ceil(k / BLOCK_SIZE)
            return 2.0 * (k * step_seconds + block_count * checkpoint_seconds + dev_seconds)
        estimate_3000 = estimated_active_seconds(K_MAX)
        estimate_2000 = estimated_active_seconds(K_MIN)
        if FAST_DIRECT_PREFLIGHT:
            k = FAST_DIRECT_K
        elif estimate_3000 <= ACTIVE_GPU_BUDGET_SECONDS:
            k = K_MAX
        elif estimate_2000 <= ACTIVE_GPU_BUDGET_SECONDS:
            k = K_MIN
        else:
            raise HardBlockedLongRun("BLOCKED_RESOURCE_BUDGET", "A/B each 2000 updates exceeds the bounded personal-PC active-GPU budget", measured_seconds_per_update=step_seconds, estimated_active_seconds_3000=estimate_3000, estimated_active_seconds_2000=estimate_2000, dev_seconds=dev_seconds, achievable_k=math.floor((ACTIVE_GPU_BUDGET_SECONDS / 2.0 - dev_seconds) / max(step_seconds, 1.0e-12)))
        benchmark["budget"] = {"active_gpu_budget_seconds": ACTIVE_GPU_BUDGET_SECONDS, "conservative_seconds_per_update": step_seconds, "dev_seconds_per_arm": dev_seconds, "checkpoint_seconds_per_block_per_arm": checkpoint_seconds, "estimated_active_seconds_3000": estimate_3000, "estimated_active_seconds_2000": estimate_2000, "selected_K": k, "selection_rule": "fast direct path fixes K=2000 and reserves compute for A/B updates" if FAST_DIRECT_PREFLIGHT else "prefer 3000; use 2000 only when 3000 exceeds bounded personal-PC budget"}
        contract = _make_contract(root, started, k, calibration, benchmark, cache_ref, cache_used=use_cache)
        contract_ref = contract["ref"]
        input_payload = {"schema": "lightgen_2d_ideal_facet_relu_long_preflight_input.v1", "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "contract": contract_ref, "source_n5_a": source["checkpoint_ref"], "arms": ["A_LINEAR", "B_RELU90"], "K": k, "calibration": calibration, "cache": cache_ref, "cache_used": use_cache, "cache_equivalence": cache_equivalence, "sampler": sampler_probe, "resume_equivalence": resume, "benchmark": benchmark, "data_roles": {"TRAIN": 55000, "DEV": 5000, "protected": 0, "official_test": 0, "sealed": 0}, "scientific_updates": 0}
        N4.atomic_write_new_json(PREFLIGHT_INPUT, input_payload, artifact_kind="long ideal facet ReLU preflight input")
        payload.update({"result": "AUTHORIZABLE", "status": "valid_completed", "classification": "LONG_IDEAL_FACET_RELU_PREFLIGHT_AUTHORIZABLE", "contract": contract_ref, "authority": {"n5_a_report": source["report_ref"], "n5_a_checkpoint": source["checkpoint_ref"], "n5_contract_sha256": N5_CONTRACT_SHA256}, "runtime": {"device": device_facts, "production_grid": PRODUCTION_GRID}, "dataset": N6._public(dataset), "arms": ["A_LINEAR", "B_RELU90"], "K": k, "calibration": calibration, "cache": cache_ref, "cache_used": use_cache, "cache_equivalence": cache_equivalence, "benchmark": benchmark, "sampler": sampler_probe, "resume_equivalence": resume, "preflight_checks": {"authorities": True, "exact_common_start": True, "train_dev_roles": True, "facet_placement_benchmark": True, "A_identity_and_B_threshold_forward": True, "calibration_train_only": True, "cache_equivalence": cache_equivalence["status"], "three_step_real_updates": True, "sampler_epoch_boundary": True, "sampler_resume": True, "RNG_resume": True, "checkpoint_reload": True, "terminal_sampler_rule": True, "device": "cuda", "scientific_updates": 0}, "input": {"path": _relative(root, PREFLIGHT_INPUT), "sha256": _sha256(PREFLIGHT_INPUT)}, "claim_boundary": {"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_updates": 0, "scientific_forward_evaluations": 0, "stage_b": False, "latent_prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}})
    except HardBlockedLongRun as error:
        payload.update({"result": "BLOCKED", "status": "blocked", "classification": error.code, "hard_blocker": {"message": str(error), **error.details}, "next": "stop before scientific update"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_LONG_IDEAL_FACET_RELU_PREFLIGHT_UNEXPECTED", "invalid_input": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "repair harness and create a fresh preflight attempt; no scientific update"})
    finally:
        ending = V10.V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == _status_lines(root)})
    N4.atomic_write_new_json(PREFLIGHT_REPORT, payload, artifact_kind="long ideal facet ReLU preflight report")
    report_ref = {"path": _relative(root, PREFLIGHT_REPORT), "sha256": _sha256(PREFLIGHT_REPORT)}
    lineage = {"schema": "lightgen_2d_ideal_facet_relu_long_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "contract": payload.get("contract"), "provenance": payload["provenance"], "claim_boundary": payload["claim_boundary"]}
    N4.atomic_write_new_json(PREFLIGHT_LINEAGE, lineage, artifact_kind="long ideal facet ReLU preflight lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": _relative(root, PREFLIGHT_LINEAGE), "sha256": _sha256(PREFLIGHT_LINEAGE)}, "contract": payload.get("contract"), "K": payload.get("K"), "arms": payload.get("arms")}


def _load_preflight(root: Path) -> dict[str, Any]:
    report = _read_json(_path(root, PREFLIGHT_REPORT), "long preflight report")
    lineage = _read_json(_path(root, PREFLIGHT_LINEAGE), "long preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed", "long preflight is not authorizable")
    report_path = _path(root, PREFLIGHT_REPORT)
    report_hash = _sha256(report_path)
    _require(lineage.get("report", {}).get("sha256") == report_hash, "long preflight lineage binding changed")
    contract_ref = _mapping(report.get("contract"), "long contract reference")
    contract_path = _path(root, contract_ref["path"])
    _validate_sha(contract_path, str(contract_ref["sha256"]), "long contract")
    contract = _read_json(contract_path, "long contract")
    _require(contract.get("task_id") == TASK_ID and contract.get("outer_gate") == OUTER_GATE, "long contract identity changed")
    _require(contract.get("arms", {}).get("executed") == ["A_LINEAR", "B_RELU90"], "long arm set changed")
    k = int(contract.get("training", {}).get("K", -1))
    _require(K_MIN <= k <= K_MAX, "long K is outside authorized range")
    _require(report.get("claim_boundary", {}).get("scientific_updates") == 0, "preflight scientific boundary changed")
    return {"report": report, "contract": contract, "contract_ref": dict(contract_ref), "report_ref": {"path": _relative(root, report_path), "sha256": report_hash}, "lineage_ref": {"path": _relative(root, _path(root, PREFLIGHT_LINEAGE)), "sha256": _sha256(_path(root, PREFLIGHT_LINEAGE))}}


def _facet_observations(probe: LongFacetProbe, rows: int) -> dict[str, Any]:
    batches = math.ceil(rows / LOGICAL_BATCH_SIZE)
    _require(len(probe.records) >= 3 * batches, f"long facet observation records incomplete: {len(probe.records)}")
    base_records = [probe.records[3 * index + 1] for index in range(batches)]
    shifted_records = [probe.records[3 * index + 2] for index in range(batches)]
    payload: dict[str, Any] = {"base": {}, "shifted": {}, "placement": probe.placements[1] if len(probe.placements) >= 2 else (probe.placements[0] if probe.placements else {})}
    for target, records in ((payload["base"], base_records), (payload["shifted"], shifted_records)):
        _require(bool(records), "long facet records are empty")
        for name in records[0]:
            target[name] = torch.cat([record[name] for record in records], dim=0)[:rows]
    _require(payload["base"]["input_mean"].shape == (rows,) and payload["shifted"]["input_mean"].shape == (rows,), "long facet observation rows changed")
    return payload


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    values = value.to(torch.float64).flatten()
    _finite(values, "long facet quantile input")
    q = torch.quantile(values, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
    return {key: float(item) for key, item in zip(("min", "q25", "median", "q75", "max"), q.tolist(), strict=True)} | {"mean": float(values.mean())}


def _facet_diagnostics(observations: Mapping[str, Any], *, mode: str, tau: float) -> dict[str, Any]:
    result = {"mode": mode, "tau": tau, "placement": dict(observations.get("placement", {}))}
    for name in ("base", "shifted"):
        values = observations[name]
        result[name] = {key: _quantiles(values[key]) for key in ("input_mean", "output_mean", "input_power", "output_power", "power_ratio", "active_fraction", "suppressed_fraction", "transition_fraction", "local_slope")}
    result["power_retention_mean"] = result["base"]["power_ratio"]["mean"]
    result["activation_mean"] = result["base"]["active_fraction"]["mean"]
    result["suppression_mean"] = result["base"]["suppressed_fraction"]["mean"]
    return result


def _direct_diversity(raw: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    flat = raw.detach().to(torch.float64).flatten(1)
    _finite(flat, "direct diversity raw")
    overall = flat.var(dim=0, unbiased=False)
    class_records = []
    class_means = []
    for digit in range(CLASS_COUNT):
        values = flat[labels == digit]
        _require(values.shape[0] > 0, f"direct diversity class {digit} absent")
        class_means.append(values.mean(dim=0))
        class_records.append({"class": digit, "count": int(values.shape[0]), "mean": float(values.mean()), "std": float(values.var(dim=0, unbiased=False).mean().sqrt()), "pixel_variance_mean": float(values.var(dim=0, unbiased=False).mean()), "pixel_variance_max": float(values.var(dim=0, unbiased=False).max())})
    distances = torch.pdist(torch.stack(class_means))
    return {"overall": {"count": int(flat.shape[0]), "mean": float(flat.mean()), "std": float(overall.mean().sqrt()), "pixel_variance_mean": float(overall.mean()), "pixel_variance_max": float(overall.max())}, "within_class": class_records, "between_class_mean_separation": {"mean": float(distances.mean()), "median": float(distances.median()), "min": float(distances.min()), "max": float(distances.max()), "pair_count": int(distances.numel())}}


def _evaluate_endpoint(root: Path, output_dir: Path, consumer: Any, adapter: Any, probe: LongFacetProbe, bundle: Mapping[str, Any], dev_images: torch.Tensor, dev_labels: torch.Tensor, selection_images: torch.Tensor, selection_labels: torch.Tensor, selection_indices: torch.Tensor, *, step: int, total_steps: int, train_metric: Mapping[str, Any], contract_ref: Mapping[str, Any], mode: str, tau: float, initial_phase: torch.Tensor, counters: dict[str, int]) -> dict[str, Any]:
    runtime_before = _digest_tree(consumer.runtime.state_dict())
    adapter_before = _digest_tree(adapter.state_dict())
    probe.clear()
    probe.capture = True
    with torch.no_grad():
        counters["scientific_forward_evaluations"] += 1
        evaluation = N4._evaluate(consumer, adapter, bundle["authorities"]["smooth"], dev_images, dev_labels, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"], step=step, train_trace=None)
        counters["scientific_forward_evaluations"] += 1
        cyclic = V10._cyclic_transfer_v10(consumer, adapter, bundle["authorities"]["smooth"], selection_images, selection_labels, selection_indices, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"])
    _require(runtime_before == _digest_tree(consumer.runtime.state_dict()) and adapter_before == _digest_tree(adapter.state_dict()), f"{mode} evaluation changed model state")
    facet = _facet_observations(probe, int(dev_images.shape[0]))
    sufficient = N5._canonical_sufficient_payload(contract_ref, mode, step, evaluation, cyclic, dev_labels, train_metric)
    sufficient.update({"schema": "lightgen_2d_ideal_facet_relu_long_sufficient_observations.v1", "task_id": TASK_ID, "candidate": mode, "global_step": 600 + step, "facet_observations": facet, "facet_material": {"kind": "IDEAL_INTENSITY_RELU_SURROGATE", "mode": mode, "tau": tau, "gain": MATERIAL_GAIN, "slope": MATERIAL_SLOPE, "phase_shift": 0.0, "trainable": False}, "forward_mode": "scientific evaluation; sufficient observations persisted before optional diagnostics"})
    sufficient_path = output_dir / f"lightgen_2d_ideal_facet_relu_long_{mode.lower()}_step_{step:05d}_sufficient_observations.v1.pt"
    sufficient_ref = _save_tensor(root, sufficient_path, sufficient, f"long {mode} sufficient observations step {step}")
    loaded = _mapping(N4.safe_torch_load(_path(root, sufficient_ref["path"]), expected_sha256=sufficient_ref["sha256"], map_location="cpu"), f"long {mode} sufficient reload")
    raw_base = cyclic["_observations"]["raw_base"]
    raw_labels = cyclic["_observations"]["labels"]
    diagnostics = {"facet": _facet_diagnostics(loaded["facet_observations"], mode=mode, tau=tau), "adapter": N5._adapter_diagnostics(adapter), "generator": N5._generator_diagnostics(consumer, "A_CONTINUED_C1", initial_phase), "cyclic": N5._cyclic_diagnostics_from_observations(loaded), "direct_diversity": _direct_diversity(raw_base, raw_labels)}
    diagnostic_path = output_dir / f"lightgen_2d_ideal_facet_relu_long_{mode.lower()}_step_{step:05d}_diagnostics.v1.json"
    diagnostic_ref = _save_json(diagnostic_path, {"schema": "lightgen_2d_ideal_facet_relu_long_diagnostics.v1", "task_id": TASK_ID, "candidate": mode, "step": step, "sufficient_observations": sufficient_ref, "diagnostics": diagnostics}, f"long {mode} endpoint diagnostics")
    raw_ref = None
    if step == total_steps:
        raw_payload = {"schema": "lightgen_2d_ideal_facet_relu_long_cyclic_raw_outputs.v1", "task_id": TASK_ID, "candidate": mode, "contract": dict(contract_ref), "selection_indices": cyclic["_observations"]["selection_indices"], "labels": cyclic["_observations"]["labels"], "targets": cyclic["_observations"]["targets"], "raw_base": cyclic["_observations"]["raw_base"], "raw_shifted": cyclic["_observations"]["raw_shifted"]}
        raw_path = output_dir / f"lightgen_2d_ideal_facet_relu_long_{mode.lower()}_step_{step:05d}_cyclic_raw_outputs.v1.pt"
        raw_ref = _save_tensor(root, raw_path, raw_payload, f"long {mode} endpoint raw outputs")
    return {"step": step, "mode": mode, "tau": tau, "global_step": 600 + step, "dev_labels": loaded["dev_labels"], "dev_correct": loaded["dev_smooth_correct"], "evaluation_public": loaded["evaluation_public"], "cyclic_public": loaded["cyclic_public"], "dev_image_loss": float(loaded["dev_image_loss"]), "train_metric": dict(train_metric), "sufficient_ref": sufficient_ref, "diagnostic_ref": diagnostic_ref, "facet_diagnostics": diagnostics["facet"], "raw_ref": raw_ref}


def _checkpoint_path(output_dir: Path, mode: str, step: int) -> Path:
    return output_dir / f"{mode.lower()}_step_{step:05d}_checkpoint.v1.pt"


def _sufficient_path(output_dir: Path, mode: str, step: int) -> Path:
    return output_dir / f"lightgen_2d_ideal_facet_relu_long_{mode.lower()}_step_{step:05d}_sufficient_observations.v1.pt"


def _diagnostic_path(output_dir: Path, mode: str, step: int) -> Path:
    return output_dir / f"lightgen_2d_ideal_facet_relu_long_{mode.lower()}_step_{step:05d}_diagnostics.v1.json"


def _raw_path(output_dir: Path, mode: str, step: int) -> Path:
    return output_dir / f"lightgen_2d_ideal_facet_relu_long_{mode.lower()}_step_{step:05d}_cyclic_raw_outputs.v1.pt"


def _load_checkpoint_for_step(root: Path, output_dir: Path, mode: str, step: int, contract_ref: Mapping[str, Any], total_steps: int) -> dict[str, Any]:
    path = _checkpoint_path(output_dir, mode, step)
    _require(path.is_file(), f"explicit {mode} checkpoint step {step} is unavailable: {path}")
    digest = _sha256(path)
    checkpoint = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"{mode} checkpoint {step}")
    _require(checkpoint.get("schema") == "lightgen_2d_ideal_facet_relu_long_checkpoint.v1" and checkpoint.get("candidate") == mode and checkpoint.get("checkpoint_step") == step and checkpoint.get("total_steps") == total_steps, f"{mode} checkpoint identity changed")
    _require(checkpoint.get("contract") == dict(contract_ref), f"{mode} checkpoint contract binding changed")
    _require(int(_mapping(checkpoint.get("sampler_state"), f"{mode} sampler state")["cursor"]) == step, f"{mode} checkpoint sampler cursor changed")
    checkpoint["_ref"] = {"path": _relative(root, path), "sha256": digest, "step": step, "global_step": 600 + step, "reloaded": True}
    return dict(checkpoint)


def _load_arm_context(root: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], cache_ref: Mapping[str, Any], z_train: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, *, mode: str, tau: float, step: int, total_steps: int, contract_ref: Mapping[str, Any], use_cache: bool) -> dict[str, Any]:
    output_dir = _path(root, ARM_DIRS[mode])
    if step == 0:
        checkpoint_path = _checkpoint_path(output_dir, mode, 0)
        if checkpoint_path.is_file():
            checkpoint = _load_checkpoint_for_step(root, output_dir, mode, 0, contract_ref, total_steps)
            source_for_model = {"checkpoint": checkpoint, "checkpoint_ref": source["checkpoint_ref"]}
        else:
            checkpoint = source["checkpoint"]
            source_for_model = source
    else:
        checkpoint = _load_checkpoint_for_step(root, output_dir, mode, step, contract_ref, total_steps)
        source_for_model = {"checkpoint": checkpoint, "checkpoint_ref": source["checkpoint_ref"]}
    rng_state = _cpu_tree(checkpoint["rng_state"])
    V10.V6._restore_rng(rng_state)
    consumer, adapter, probe = _new_candidate(bundle, source_for_model, device=device, mode="linear" if mode == "A_LINEAR" else "relu", tau=tau, trainable=True)
    optimizer = _optimizer(consumer, adapter)
    _restore_optimizer(optimizer, checkpoint["optimizer_state"])
    V10.V6._restore_rng(rng_state)
    sampler = BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=total_steps)
    if step > 0:
        sampler.load_state_dict(_mapping(checkpoint["sampler_state"], f"{mode} sampler state"))
    _require(sampler.cursor == step, f"{mode} context sampler cursor does not match update index")
    expected_lr, expected_stage = _schedule_lr(step, total_steps)
    if step > 0:
        schedule = _mapping(checkpoint.get("schedule_state"), f"{mode} schedule state")
        _require(schedule.get("stage") == expected_stage and math.isclose(float(schedule.get("learning_rate")), expected_lr, rel_tol=0.0, abs_tol=1.0e-15), f"{mode} checkpoint schedule changed")
    _set_stage(consumer, adapter, coadapt=expected_stage == "adapter_decoder_coadaptation")
    _set_lr(optimizer, step, total_steps)
    return {"mode": mode, "tau": tau, "output_dir": output_dir, "consumer": consumer, "adapter": adapter, "probe": probe, "optimizer": optimizer, "sampler": sampler, "checkpoint": checkpoint, "initial_phase": source_for_model["checkpoint"]["runtime_state"]["generator.phase"].detach().cpu(), "position_map": _train_position_map(train_indices), "z_train": z_train, "images": images, "labels": labels, "train_indices": train_indices, "bundle": bundle, "source": source, "cache_ref": cache_ref, "contract_ref": contract_ref, "device": device, "total_steps": total_steps, "use_cache": use_cache}


def _fixed_monitor_batch(context: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sampler = BalancedMultiEpochSampler(context["train_indices"], context["labels"], seed=SAMPLER_SEED, steps=context["total_steps"])
    indices, labels = sampler.peek_batch()
    positions = _cache_positions(context["position_map"], indices)
    return positions, context["images"].index_select(0, indices).to(context["device"]), labels.to(context["device"])


def _initialize_arm(root: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], cache_ref: Mapping[str, Any], z_train: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, dev_indices: torch.Tensor, device: torch.device, *, mode: str, tau: float, total_steps: int, contract_ref: Mapping[str, Any], use_cache: bool) -> dict[str, Any]:
    output_dir = _path(root, ARM_DIRS[mode])
    if output_dir.exists():
        _require(not any(output_dir.iterdir()), f"arm output is not empty before initialization: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    context = _load_arm_context(root, bundle, source, cache_ref, z_train, images, labels, train_indices, device, mode=mode, tau=tau, step=0, total_steps=total_steps, contract_ref=contract_ref, use_cache=use_cache)
    dev_images = images.index_select(0, dev_indices).contiguous()
    dev_labels = labels.index_select(0, dev_indices).contiguous()
    selection_images, selection_labels, selection_indices = V10.V6._load_selection(root, images, labels, dev_indices)
    positions, monitor_batch, monitor_labels = _fixed_monitor_batch(context)
    train_metric = _train_metric(context["consumer"], context["adapter"], context["z_train"].index_select(0, positions).to(device), monitor_batch, monitor_labels, context["bundle"]["authorities"]["prototypes"].to(device), context["bundle"]["authorities"]["templates"].to(device), use_cache=use_cache)
    counters = {"scientific_forward_evaluations": 0}
    record = _evaluate_endpoint(root, output_dir, context["consumer"], context["adapter"], context["probe"], bundle, dev_images, dev_labels, selection_images, selection_labels, selection_indices, step=0, total_steps=total_steps, train_metric=train_metric, contract_ref=contract_ref, mode=mode, tau=tau, initial_phase=context["initial_phase"], counters=counters)
    checkpoint_payload = _checkpoint_payload(root, contract_ref, source, cache_ref, context["consumer"], context["adapter"], context["optimizer"], context["sampler"], mode=mode, tau=tau, step=0, total_steps=total_steps, use_cache=use_cache, block_size=BLOCK_SIZE, initial_phase=context["initial_phase"])
    checkpoint_ref = _save_checkpoint(root, _checkpoint_path(output_dir, mode, 0), checkpoint_payload)
    del context, dev_images, dev_labels, selection_images, selection_labels, selection_indices, monitor_batch
    gc.collect()
    torch.cuda.empty_cache()
    return {"mode": mode, "record": record, "checkpoint": checkpoint_ref, "scientific_forward_evaluations": counters["scientific_forward_evaluations"]}


def _run_arm_block(root: Path, preflight: Mapping[str, Any], source: Mapping[str, Any], bundle: Mapping[str, Any], cache_ref: Mapping[str, Any], cache: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, dev_indices: torch.Tensor, device: torch.device, *, mode: str, tau: float, current_step: int, target_step: int) -> dict[str, Any]:
    total_steps = int(preflight["contract"]["training"]["K"])
    use_cache = bool(preflight["contract"]["cache"].get("used", False))
    z_train, _ = _cache_views(cache, train_indices, dev_indices)
    context = _load_arm_context(root, bundle, source, cache_ref, z_train, images, labels, train_indices, device, mode=mode, tau=tau, step=current_step, total_steps=total_steps, contract_ref=preflight["contract_ref"], use_cache=use_cache)
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    training_started = time.perf_counter()
    for step in range(current_step + 1, target_step + 1):
        indices, batch_labels = context["sampler"].next_batch()
        positions = _cache_positions(context["position_map"], indices)
        batch = images.index_select(0, indices).to(device)
        z = z_train.index_select(0, positions).to(device)
        _train_step(context["consumer"], context["adapter"], context["optimizer"], z, batch, batch_labels.to(device), prototypes, templates, step=step, total_steps=total_steps, use_cache=use_cache)
        del batch, z
    _require(context["sampler"].cursor == target_step, f"{mode} block terminal sampler cursor is invalid")
    checkpoint_ref = None
    final_record = None
    if target_step == total_steps:
        dev_images = images.index_select(0, dev_indices).contiguous()
        dev_labels = labels.index_select(0, dev_indices).contiguous()
        selection_images, selection_labels, selection_indices = V10.V6._load_selection(root, images, labels, dev_indices)
        positions, monitor_batch, monitor_labels = _fixed_monitor_batch(context)
        train_metric = _train_metric(context["consumer"], context["adapter"], z_train.index_select(0, positions).to(device), monitor_batch, monitor_labels, prototypes, templates, use_cache=use_cache)
        counters = {"scientific_forward_evaluations": 0}
        final_record = _evaluate_endpoint(root, context["output_dir"], context["consumer"], context["adapter"], context["probe"], bundle, dev_images, dev_labels, selection_images, selection_labels, selection_indices, step=target_step, total_steps=total_steps, train_metric=train_metric, contract_ref=preflight["contract_ref"], mode=mode, tau=tau, initial_phase=context["initial_phase"], counters=counters)
        del dev_images, dev_labels, selection_images, selection_labels, selection_indices, monitor_batch
    checkpoint_payload = _checkpoint_payload(root, preflight["contract_ref"], source, cache_ref, context["consumer"], context["adapter"], context["optimizer"], context["sampler"], mode=mode, tau=tau, step=target_step, total_steps=total_steps, use_cache=use_cache, block_size=BLOCK_SIZE, initial_phase=context["initial_phase"])
    checkpoint_ref = _save_checkpoint(root, _checkpoint_path(context["output_dir"], mode, target_step), checkpoint_payload)
    del context
    gc.collect()
    torch.cuda.empty_cache()
    return {"mode": mode, "current_step": current_step, "target_step": target_step, "checkpoint": checkpoint_ref, "final_record": final_record, "training_seconds": time.perf_counter() - training_started}


def _load_endpoint(root: Path, output_dir: Path, mode: str, step: int, total_steps: int) -> dict[str, Any]:
    sufficient_path = _sufficient_path(output_dir, mode, step)
    _require(sufficient_path.is_file(), f"{mode} sufficient endpoint {step} is unavailable")
    sufficient_digest = _sha256(sufficient_path)
    sufficient = _mapping(N4.safe_torch_load(sufficient_path, expected_sha256=sufficient_digest, map_location="cpu"), f"{mode} sufficient endpoint {step}")
    _require(sufficient.get("schema") == "lightgen_2d_ideal_facet_relu_long_sufficient_observations.v1" and sufficient.get("candidate") == mode and sufficient.get("global_step") == 600 + step, f"{mode} sufficient endpoint identity changed")
    diagnostic_path = _diagnostic_path(output_dir, mode, step)
    diagnostic = _read_json(diagnostic_path, f"{mode} diagnostic endpoint {step}")
    raw_ref = None
    if step == total_steps:
        raw_path = _raw_path(output_dir, mode, step)
        _require(raw_path.is_file(), f"{mode} endpoint raw outputs are unavailable")
        raw_ref = {"path": _relative(root, raw_path), "sha256": _sha256(raw_path)}
    checkpoint_path = _checkpoint_path(output_dir, mode, step)
    _require(checkpoint_path.is_file(), f"{mode} endpoint checkpoint {step} is unavailable")
    return {"step": step, "mode": mode, "sufficient": sufficient, "sufficient_ref": {"path": _relative(root, sufficient_path), "sha256": sufficient_digest}, "diagnostic": diagnostic, "diagnostic_ref": {"path": _relative(root, diagnostic_path), "sha256": _sha256(diagnostic_path)}, "checkpoint_ref": {"path": _relative(root, checkpoint_path), "sha256": _sha256(checkpoint_path), "step": step, "global_step": 600 + step}, "raw_ref": raw_ref, "dev_labels": sufficient["dev_labels"], "dev_correct": sufficient["dev_smooth_correct"], "evaluation_public": sufficient["evaluation_public"], "cyclic_public": sufficient["cyclic_public"], "dev_image_loss": float(sufficient["dev_image_loss"]), "train_metric": sufficient.get("train_metric", {})}


def _checkpoint_steps(total_steps: int) -> list[int]:
    values = [0, *range(BLOCK_SIZE, total_steps + 1, BLOCK_SIZE)]
    if values[-1] != total_steps:
        values.append(total_steps)
    return values


def _arm_report(root: Path, preflight: Mapping[str, Any], source: Mapping[str, Any], mode: str, total_steps: int, *, actual_training_seconds: float, resume_sessions: int) -> dict[str, Any]:
    output_dir = _path(root, ARM_DIRS[mode])
    initial = _load_endpoint(root, output_dir, mode, 0, total_steps)
    final = _load_endpoint(root, output_dir, mode, total_steps, total_steps)
    labels = final["dev_labels"].to(torch.long)
    training_delta = final["dev_correct"].to(torch.float64) - initial["dev_correct"].to(torch.float64)
    training_ci = V10.V6._paired_bootstrap(training_delta, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    public_evaluations = {"0": initial["evaluation_public"], str(total_steps): final["evaluation_public"]}
    checkpoint_refs = {str(step): {"path": _relative(root, _checkpoint_path(output_dir, mode, step)), "sha256": _sha256(_checkpoint_path(output_dir, mode, step)), "step": step, "global_step": 600 + step} for step in _checkpoint_steps(total_steps)}
    raw_validity = bool(final["evaluation_public"]["raw_validity"]["finite"] and final["evaluation_public"]["raw_validity"]["structured"] and final["evaluation_public"]["raw_validity"]["nonconstant"] and final["cyclic_public"]["raw_validity"]["base"]["finite"] and final["cyclic_public"]["raw_validity"]["shifted"]["finite"])
    payload = {"schema": "lightgen_2d_ideal_facet_relu_long_scientific_report.v1", "version": 1, "task_id": TASK_ID, "candidate": mode, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "VALID_COMPLETED", "status": "valid_completed", "classification": f"LONG_{mode}_SCIENTIFIC_COMPLETED", "outer_gate": OUTER_GATE, "contract": preflight["contract_ref"], "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"]}, "source_n5_a": source["checkpoint_ref"], "tau": float(preflight["contract"]["facet"]["tau_90"]) if mode == "B_RELU90" else 0.0, "evaluations": public_evaluations, "train_metrics": {"0": initial["train_metric"], str(total_steps): final["train_metric"]}, "sufficient_observations": {"0": initial["sufficient_ref"], str(total_steps): final["sufficient_ref"]}, "diagnostics": {"0": initial["diagnostic_ref"], str(total_steps): final["diagnostic_ref"]}, "checkpoints": checkpoint_refs, "endpoint_raw_outputs": final["raw_ref"], "training_delta": {"R0": float(initial["evaluation_public"]["smooth"]["macro_recall"]), "RK": float(final["evaluation_public"]["smooth"]["macro_recall"]), "Delta_R": float(training_delta.mean()), "paired_bootstrap": {key: value for key, value in training_ci.items() if key != "delta_records"}}, "reconstruction": {"DEV_L_image_R0": float(initial["dev_image_loss"]), "DEV_L_image_RK": float(final["dev_image_loss"]), "Delta_L_image_within_arm": float(final["dev_image_loss"] - initial["dev_image_loss"])}, "cyclic_transfer": final["cyclic_public"], "facet_diagnostics": {"0": initial["diagnostic"].get("diagnostics", {}).get("facet", {}), str(total_steps): final["diagnostic"].get("diagnostics", {}).get("facet", {})}, "raw_validity_pass": raw_validity, "timing": {"training_seconds_current_execution": actual_training_seconds, "pause_resume_sessions": resume_sessions}, "claim_boundary": {"scientific_training_executed": True, "scientific_data_consumed": True, "scientific_updates": total_steps, "scientific_forward_evaluations": 4, "stage_b": False, "latent_prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {"rank": ADAPTER_RANK, "encoder_update": False, "ols_update": False, "w_ols_update": False, "material_trainable": False, "cache_used": bool(preflight["contract"]["cache"].get("used", False)), "sampler": BalancedMultiEpochSampler.SCHEMA, "block_size": BLOCK_SIZE, "no_retry": True}}
    report_path = output_dir / f"lightgen_2d_ideal_facet_relu_long_{mode.lower()}_scientific_report.v1.json"
    N4.atomic_write_new_json(report_path, payload, artifact_kind=f"long {mode} scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = output_dir / f"lightgen_2d_ideal_facet_relu_long_{mode.lower()}_scientific_lineage.v1.json"
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_ideal_facet_relu_long_scientific_lineage.v1", "version": 1, "task_id": TASK_ID, "candidate": mode, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "contract": payload["contract"], "preflight": payload["preflight"], "claim_boundary": payload["claim_boundary"]}, artifact_kind=f"long {mode} scientific lineage")
    return {"result": "VALID_COMPLETED", "candidate": mode, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "scientific_updates": total_steps}


def _cyclic_pass(cyclic: Mapping[str, Any]) -> bool:
    return bool(cyclic.get("template", {}).get("pass") and cyclic.get("smooth", {}).get("pass") and cyclic.get("broad") and not cyclic.get("pair_dominated"))


def _arm_report_path(preflight: Mapping[str, Any], mode: str) -> Path:
    del preflight
    return ARM_DIRS[mode] / f"lightgen_2d_ideal_facet_relu_long_{mode.lower()}_scientific_report.v1.json"


def _aggregate(root: Path, preflight: Mapping[str, Any], source: Mapping[str, Any], *, training_seconds: float, resume_sessions: int) -> dict[str, Any]:
    _require(not AGGREGATE_DIR.exists(), f"aggregate output already exists: {_path(root, AGGREGATE_DIR)}")
    total_steps = int(preflight["contract"]["training"]["K"])
    arm_reports = {mode: _read_json(_path(root, _arm_report_path(preflight, mode)), f"{mode} scientific report") for mode in ("A_LINEAR", "B_RELU90")}
    endpoint_a = _load_endpoint(root, _path(root, ARM_DIRS["A_LINEAR"]), "A_LINEAR", total_steps, total_steps)
    endpoint_b = _load_endpoint(root, _path(root, ARM_DIRS["B_RELU90"]), "B_RELU90", total_steps, total_steps)
    labels = endpoint_a["dev_labels"].to(torch.long)
    _require(torch.equal(labels, endpoint_b["dev_labels"].to(torch.long)), "A/B endpoint DEV labels are not matched")
    delta = endpoint_b["dev_correct"].to(torch.float64) - endpoint_a["dev_correct"].to(torch.float64)
    ci = V10.V6._paired_bootstrap(delta, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    image_delta = float(endpoint_b["dev_image_loss"] - endpoint_a["dev_image_loss"])
    transfer_a, transfer_b = _cyclic_pass(endpoint_a["cyclic_public"]), _cyclic_pass(endpoint_b["cyclic_public"])
    raw = bool(arm_reports["A_LINEAR"].get("raw_validity_pass") and arm_reports["B_RELU90"].get("raw_validity_pass"))
    supported = bool(float(delta.mean()) >= MATERIAL_DELTA and float(ci["ci_lower"]) > 0.0 and image_delta <= RECONSTRUCTION_MAX_DELTA and transfer_a and transfer_b and raw)
    stage_a = {mode: {"RK": float(report["training_delta"]["RK"]), "threshold": STAGE_A_FIREWALL, "status": "STAGE_A_FIREWALL_PASSED" if float(report["training_delta"]["RK"]) >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False} for mode, report in arm_reports.items()}
    per_class = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        per_class.append({"class": digit, "A_recall": float((endpoint_a["dev_correct"][mask]).to(torch.float64).mean()), "B_recall": float((endpoint_b["dev_correct"][mask]).to(torch.float64).mean()), "B_minus_A": float((delta[mask]).mean())})
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=False)
    payload = {"schema": "lightgen_2d_ideal_facet_relu_long_aggregate.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "VALID_COMPLETED", "status": "valid_completed", "classification": "LONG_IDEAL_FACET_RELU90_SUPPORTED" if supported else "LONG_IDEAL_FACET_RELU90_NOT_SUPPORTED", "outer_gate": OUTER_GATE, "contract": preflight["contract_ref"], "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"]}, "candidates": {mode: {"report": {"path": _relative(root, _path(root, _arm_report_path(preflight, mode))), "sha256": _sha256(_path(root, _arm_report_path(preflight, mode)))}, "RK": float(report["training_delta"]["RK"]), "checkpoint": report["checkpoints"][str(total_steps)]} for mode, report in arm_reports.items()}, "primary": {"A": "A_LINEAR", "B": "B_RELU90", "Delta_R_B_minus_A": float(delta.mean()), "paired_bootstrap": {key: value for key, value in ci.items() if key != "delta_records"}, "image_delta_B_minus_A": image_delta, "reconstruction_max_delta": RECONSTRUCTION_MAX_DELTA, "cyclic_transfer_A": transfer_a, "cyclic_transfer_B": transfer_b, "raw_validity": raw, "supported": supported}, "per_class": per_class, "trajectory": {mode: {"R0": float(report["training_delta"]["R0"]), "RK": float(report["training_delta"]["RK"]), "Delta_R": float(report["training_delta"]["Delta_R"])} for mode, report in arm_reports.items()}, "facet_diagnostics": {mode: report.get("facet_diagnostics", {}) for mode, report in arm_reports.items()}, "timing": {"actual_training_seconds_current_execution": training_seconds, "pause_resume_sessions": resume_sessions, "predicted_active_gpu_seconds": preflight["contract"].get("budget", {}).get("budget", {}).get("estimated_active_seconds_" + str(total_steps), None)}, "stage_a": stage_a, "claim_boundary": {"training_updates": 2 * total_steps, "per_arm_updates": total_steps, "ideal_simulation_only": True, "physical_te_validation": False, "stage_b": False, "latent_prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {"cache_used": bool(preflight["contract"]["cache"].get("used", False)), "sampler": BalancedMultiEpochSampler.SCHEMA, "block_size": BLOCK_SIZE, "no_retry": True, "no_new_geometry": True, "no_tau_sweep": True}, "next": "visualization may consume this aggregate; Stage B, latent prior, and sampled generation remain closed"}
    aggregate_path = AGGREGATE_DIR / "lightgen_2d_ideal_facet_relu_long_aggregate_report.v1.json"
    N4.atomic_write_new_json(aggregate_path, payload, artifact_kind="long ideal facet ReLU aggregate report")
    report_ref = {"path": _relative(root, aggregate_path), "sha256": _sha256(aggregate_path)}
    lineage_path = AGGREGATE_DIR / "lightgen_2d_ideal_facet_relu_long_aggregate_lineage.v1.json"
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_ideal_facet_relu_long_aggregate_lineage.v1", "version": 1, "task_id": TASK_ID, "result": "VALID_COMPLETED", "classification": payload["classification"], "report": report_ref, "contract": payload["contract"], "preflight": payload["preflight"], "claim_boundary": payload["claim_boundary"]}, artifact_kind="long ideal facet ReLU aggregate lineage")
    return {"result": "VALID_COMPLETED", "classification": payload["classification"], "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "primary": payload["primary"], "stage_a": stage_a}


def run_scientific(root: Path, *, resume_a_step: int | None = None, resume_b_step: int | None = None) -> dict[str, Any]:
    preflight = _load_preflight(root)
    _require((resume_a_step is None) == (resume_b_step is None), "resume requires explicit A and B checkpoint steps")
    source = _load_source(root)
    authority = _resolve_bundle(root)
    bundle = authority["bundle"]
    V10.V6._prepare_cuda_runtime_dll_search_path()
    device, device_facts = V10.V6._cuda_facts()
    _require(device.type == "cuda", "CUDA is required for long scientific execution")
    bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
    images, labels, train_indices, dev_indices, dataset = V10.V6._load_roles(root)
    _require(dataset.get("protected", 0) in (0, [], None) and dataset.get("sealed", 0) in (0, [], None), "protected or sealed role is nonempty")
    cache_ref = _mapping(preflight["report"].get("cache"), "long cache reference")
    if bool(preflight["contract"]["cache"].get("used", False)):
        cache = _mapping(N4.safe_torch_load(_path(root, cache_ref["path"]), expected_sha256=str(cache_ref["sha256"]), map_location="cpu"), "long cache")
    else:
        cache = _direct_placeholder_cache(train_indices, dev_indices, labels)
    z_train, _ = _cache_views(cache, train_indices, dev_indices)
    total_steps = int(preflight["contract"]["training"]["K"])
    tau_by_mode = {"A_LINEAR": 0.0, "B_RELU90": float(preflight["contract"]["facet"]["tau_90"])}
    started = time.perf_counter()
    resume_sessions = 1 if resume_a_step is not None else 0
    current = {"A_LINEAR": 0, "B_RELU90": 0}
    if resume_a_step is None:
        for mode in ("A_LINEAR", "B_RELU90"):
            output_dir = _path(root, ARM_DIRS[mode])
            output_dir.mkdir(parents=True, exist_ok=False)
        for mode in ("A_LINEAR", "B_RELU90"):
            _initialize_arm(root, bundle, source, cache_ref, z_train, images, labels, train_indices, dev_indices, device, mode=mode, tau=tau_by_mode[mode], total_steps=total_steps, contract_ref=preflight["contract_ref"], use_cache=bool(preflight["contract"]["cache"].get("used", False)))
    else:
        _require(isinstance(resume_a_step, int) and isinstance(resume_b_step, int), "resume steps must be integers")
        _require(0 <= resume_a_step <= total_steps and 0 <= resume_b_step <= total_steps, "resume step is outside K")
        for mode, step in (("A_LINEAR", resume_a_step), ("B_RELU90", resume_b_step)):
            output_dir = _path(root, ARM_DIRS[mode])
            _require(output_dir.is_dir(), f"explicit resume arm directory is unavailable: {output_dir}")
            if not _checkpoint_path(output_dir, mode, step).is_file():
                _require(step == 0, f"explicit resume checkpoint is unavailable for {mode} at step {step}")
                _initialize_arm(root, bundle, source, cache_ref, z_train, images, labels, train_indices, dev_indices, device, mode=mode, tau=tau_by_mode[mode], total_steps=total_steps, contract_ref=preflight["contract_ref"], use_cache=bool(preflight["contract"]["cache"].get("used", False)))
        current = {"A_LINEAR": resume_a_step, "B_RELU90": resume_b_step}
    training_seconds = 0.0
    print(json.dumps({"event": "scientific_start", "arms": ["A_LINEAR", "B_RELU90"], "K": total_steps, "resume": resume_sessions > 0, "device": device_facts}, ensure_ascii=False), flush=True)
    while current["A_LINEAR"] < total_steps or current["B_RELU90"] < total_steps:
        target = min(total_steps, max(current.values()) + BLOCK_SIZE)
        for mode in ("A_LINEAR", "B_RELU90"):
            if current[mode] < target:
                block = _run_arm_block(root, preflight, source, bundle, cache_ref, cache, images, labels, train_indices, dev_indices, device, mode=mode, tau=tau_by_mode[mode], current_step=current[mode], target_step=target)
                current[mode] = target
                training_seconds += float(block["training_seconds"])
                print(json.dumps({"event": "long_progress", "candidate": mode, "step": target, "K": total_steps, "matched_steps": current, "training_seconds_current_execution": training_seconds}, ensure_ascii=False), flush=True)
    arm_results = {mode: _arm_report(root, preflight, source, mode, total_steps, actual_training_seconds=training_seconds, resume_sessions=resume_sessions) for mode in ("A_LINEAR", "B_RELU90")}
    result = _aggregate(root, preflight, source, training_seconds=training_seconds, resume_sessions=resume_sessions)
    result["arms"] = arm_results
    result["runtime"] = {"device": device_facts, "scientific_wall_seconds_current_execution": time.perf_counter() - started}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific", "all"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--resume-a-step", type=int, default=None, help="explicit last A checkpoint step; must be paired with --resume-b-step")
    parser.add_argument("--resume-b-step", type=int, default=None, help="explicit last B checkpoint step; must be paired with --resume-a-step")
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    if args.action == "preflight":
        result = run_preflight(root)
    elif args.action == "scientific":
        result = run_scientific(root, resume_a_step=args.resume_a_step, resume_b_step=args.resume_b_step)
    else:
        _require(args.resume_a_step is None and args.resume_b_step is None, "--action all does not accept resume steps")
        result = run_preflight(root)
        if result.get("result") != "AUTHORIZABLE":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2
        result = run_scientific(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("result") in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
