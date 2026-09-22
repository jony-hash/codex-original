"""Execute the one-shot N6 nonlinear fiber-output facet canary.

The entrypoint has two explicit phases.  ``preflight`` is bounded engineering
work with zero scientific updates.  ``scientific`` consumes the authorizable
preflight once and runs exactly one A/B continuation from the frozen N5-A
step-300 endpoint.  The fixed material is installed by wrapping the existing
generator call; no new optical plane or trainable material is introduced.
"""

from __future__ import annotations

import argparse
import copy
import gc
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
    spec.loader.exec_module(module)
    return module


# Existing N5/V11/V10 modules are code authorities only.  Scientific
# observations are resolved explicitly below from the N5-A checkpoint.
N5 = _load_module(
    Path(__file__).with_name("execute_lightgen_2d_true_optical_depth_comparison_v2.py"),
    "lightgen_n6_n5_code_authority",
)
N4 = N5.N4
V10 = N5.V10
V11 = N5.V11


TASK_ID = "EXECUTE_N6_NONLINEAR_FIBER_FACET_ACTIVATION_CANARY_V1"
DESIGN_AUTHORITY = "DEFINE_N6_NONLINEAR_FIBER_FACET_ACTIVATION_CANARY_V1"
OUTER_GATE = "N6_NONLINEAR_FIBER_FACET_ACTIVATION_SUPPORTED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

BASE_OUTPUT = Path("outputs/20260914_n6/runs")
CONTRACT_PATH = BASE_OUTPUT / (
    "lightgen_2d_nonlinear_fiber_facet_activation_canary_contract_v1_attempt_02/"
    "lightgen_2d_nonlinear_fiber_facet_activation_canary_contract.v1.json"
)
CONTRACT_SHA256 = "1943c9327076e57451923ef3d0bc6815b9f336a5e7fc8efc2b6348702192bfd5"
PREFLIGHT_DIR = BASE_OUTPUT / "lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_v1_attempt_04"
CANDIDATE_A_DIR = BASE_OUTPUT / "lightgen_2d_nonlinear_fiber_facet_activation_canary_a_attempt_01"
CANDIDATE_B_DIR = BASE_OUTPUT / "lightgen_2d_nonlinear_fiber_facet_activation_canary_b_attempt_01"
AGGREGATE_DIR = BASE_OUTPUT / "lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_v1_attempt_01"

N5_A_REPORT_PATH = Path(
    "outputs/20260914_n5_v2/runs/"
    "lightgen_2d_true_optical_depth_comparison_v2_candidate_a_attempt_01/"
    "lightgen_2d_true_optical_depth_comparison_v2_a_continued_c1_scientific_report.v1.json"
)
N5_A_REPORT_SHA256 = "ed796d14971aa8cf44f708ba3dc274601f38db46b82559dc4572bcd202d14fca"
N5_A_CHECKPOINT_PATH = Path(
    "outputs/20260914_n5_v2/runs/"
    "lightgen_2d_true_optical_depth_comparison_v2_candidate_a_attempt_01/"
    "lightgen_2d_true_optical_depth_comparison_v2_a_continued_c1_step_300_checkpoint.v1.pt"
)
N5_A_CHECKPOINT_SHA256 = "5ac782bde83e7f33c34df872c2b907005f0b27200ff8c3b06386019a7294be6e"
N5_CONTRACT_SHA256 = "f7928d640093d8133380233c0ee451221b93f1ee03236b381227228a5c1df9fb"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
ADAPTER_RANK = 9
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = 2048
PRECHECK_STEPS = 3
SCIENTIFIC_STEPS = 300
CHECKPOINT_STEPS = (0, 100, 200, 300)
N5_GLOBAL_OFFSET = 600
N6_GLOBAL_TERMINAL = 900
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
SPLIT_SEED = 1234
STAGE_A_FIREWALL = 0.937
MATERIAL_DELTA = 0.05
RECONSTRUCTION_MAX_DELTA = 0.005
S_TAU = 0.02919053753794475
LAMBDA_TRANSFER = 28.48464149920509
LAMBDA_IMAGE = 13.353282014834436
MATERIAL_THRESHOLD = 0.05
MATERIAL_SLOPE = 1.0
MATERIAL_GAIN = 1.0
MATERIAL_EPSILON = 1.0e-12
MATERIAL_TRANSITION_HALF_WIDTH = 0.01

ADAPTER_PARAMETER_NAMES = [
    "adapter.u_real",
    "adapter.u_imag",
    "adapter.v_real",
    "adapter.v_imag",
]
TRAINABLE_PARAMETER_NAMES = ADAPTER_PARAMETER_NAMES + ["generator.phase"]

PREFLIGHT_REPORT_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_lineage.v1.json"
PREFLIGHT_INPUT_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_input.v1.json"
PREFLIGHT_CHECKPOINT_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_{candidate}_smoke_checkpoint.v1.pt"
PREFLIGHT_CHAIN_OBS_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_synthetic_chain_sufficient.v1.pt"
PREFLIGHT_CHAIN_REPORT_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_synthetic_chain_report.v1.json"
PREFLIGHT_CHAIN_LINEAGE_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_synthetic_chain_lineage.v1.json"

SCIENTIFIC_INPUT_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_{candidate}_scientific_input.v1.json"
SCIENTIFIC_REPORT_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_{candidate}_scientific_report.v1.json"
SCIENTIFIC_LINEAGE_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_{candidate}_scientific_lineage.v1.json"
SUFFICIENT_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_{candidate}_step_{step}_sufficient_observations.v1.pt"
DIAGNOSTICS_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_{candidate}_step_{step}_diagnostics.v1.json"
RAW_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_{candidate}_step_300_cyclic_raw_outputs.v1.pt"
CHECKPOINT_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_{candidate}_step_{step}_checkpoint.v1.pt"
AGGREGATE_REPORT_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_report.v1.json"
AGGREGATE_LINEAGE_NAME = "lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_lineage.v1.json"


class InvalidN6(RuntimeError):
    """Invalid harness or missing required evidence."""


class HardBlockedN6(RuntimeError):
    """A valid boundary prevents scientific continuation."""

    def __init__(self, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidN6(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, value: str | Path) -> Path:
    relative = Path(value)
    _require(not relative.is_absolute(), f"absolute artifact path is not allowed: {relative}")
    return (root / relative).resolve()


def _relative(root: Path, value: Path) -> str:
    return value.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidN6(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _sha256(path: Path) -> str:
    return N4._file_sha256(path)


def _validate_sha(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = _sha256(path)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")


def _cpu_tree(value: Any) -> Any:
    return N4._cpu_tree(value)


def _tree_equal(left: Any, right: Any) -> bool:
    return N4._tree_equal(left, right)


def _digest_tree(value: Any) -> str:
    return N4._digest_tree(value)


def _public(value: Any) -> Any:
    return N5._public(value)


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _contract(root: Path) -> dict[str, Any]:
    path = _path(root, CONTRACT_PATH)
    _validate_sha(path, CONTRACT_SHA256, "N6 contract")
    contract = _read_json(path, "N6 contract")
    _require(contract.get("schema") == "lightgen_2d_nonlinear_fiber_facet_activation_canary_contract.v1", "N6 contract schema changed")
    _require(contract.get("status") == "READY_FOR_BOUNDED_PREFLIGHT", "N6 contract is not ready")
    _require(contract.get("outer_gate") == OUTER_GATE, "N6 outer gate changed")
    _require(contract.get("git_boundary", {}).get("branch") == EXPECTED_BRANCH, "N6 branch boundary changed")
    _require(contract.get("git_boundary", {}).get("head") == EXPECTED_HEAD, "N6 HEAD boundary changed")
    material = _mapping(contract.get("material"), "N6 material")
    transfer = _mapping(material.get("fixed_transfer"), "N6 fixed transfer")
    _require(material.get("kind") == "ALGORITHMIC_NONLINEAR_FACET_CANARY", "N6 material kind changed")
    _require(material.get("measured_transfer_curve") is False and material.get("trainable") is False and material.get("tuning") is False, "N6 material authority changed")
    _require(float(transfer.get("threshold")) == MATERIAL_THRESHOLD, "N6 threshold changed")
    _require(float(transfer.get("slope")) == MATERIAL_SLOPE and float(transfer.get("gain")) == MATERIAL_GAIN, "N6 transfer coefficients changed")
    _require("I_out+epsilon" in str(transfer.get("field")), "N6 field differentiability stabilizer changed")
    objective = _mapping(contract.get("common_protocol"), "N6 common protocol")
    _require(float(objective.get("lambda_transfer")) == LAMBDA_TRANSFER and float(objective.get("lambda_image")) == LAMBDA_IMAGE, "N6 objective weights changed")
    _require(objective.get("additional_updates") == SCIENTIFIC_STEPS and objective.get("global_offset") == N5_GLOBAL_OFFSET, "N6 continuation budget changed")
    _require(contract.get("stage_a", {}).get("firewall") == "R300>=0.937", "N6 Stage-A firewall changed")
    _require(contract.get("data_boundary", {}).get("protected") is False and contract.get("data_boundary", {}).get("sealed") is False, "N6 data boundary changed")
    return contract


def _load_n5_a(root: Path) -> dict[str, Any]:
    report_path = _path(root, N5_A_REPORT_PATH)
    checkpoint_path = _path(root, N5_A_CHECKPOINT_PATH)
    _validate_sha(report_path, N5_A_REPORT_SHA256, "N5-A scientific report")
    _validate_sha(checkpoint_path, N5_A_CHECKPOINT_SHA256, "N5-A step300 checkpoint")
    report = _read_json(report_path, "N5-A scientific report")
    _require(report.get("result") == "VALID_COMPLETED" and report.get("status") == "valid_completed", "N5-A report is not VALID_COMPLETED")
    _require(report.get("candidate") == "A_CONTINUED_C1" and report.get("scientific_updates") == 300, "N5-A report identity changed")
    checkpoint_ref = _mapping(report.get("checkpoints", {}).get("300"), "N5-A step300 checkpoint reference")
    _require(checkpoint_ref.get("sha256") == N5_A_CHECKPOINT_SHA256 and checkpoint_ref.get("path") == N5_A_CHECKPOINT_PATH.as_posix(), "N5-A endpoint reference changed")
    checkpoint = _mapping(N4.safe_torch_load(checkpoint_path, expected_sha256=N5_A_CHECKPOINT_SHA256, map_location="cpu"), "N5-A step300 checkpoint")
    _require(checkpoint.get("candidate") == "A_CONTINUED_C1" and checkpoint.get("checkpoint_step") == 300 and checkpoint.get("global_step") == N5_GLOBAL_OFFSET, "N5-A checkpoint step identity changed")
    _require(checkpoint.get("rank") == ADAPTER_RANK, "N5-A rank changed")
    _require(checkpoint.get("trainable_parameter_names") == TRAINABLE_PARAMETER_NAMES, "N5-A trainable surface changed")
    _require(_digest_tree(_cpu_tree(checkpoint["runtime_state"])) == checkpoint_ref.get("runtime_state_digest"), "N5-A runtime state digest changed")
    _require(_digest_tree(_cpu_tree(checkpoint["adapter_state"])) == checkpoint_ref.get("adapter_state_digest"), "N5-A adapter state digest changed")
    return {
        "report": report,
        "checkpoint": checkpoint,
        "report_ref": {"path": _relative(root, report_path), "sha256": N5_A_REPORT_SHA256},
        "checkpoint_ref": {"path": _relative(root, checkpoint_path), "sha256": N5_A_CHECKPOINT_SHA256},
    }


def _resolve_authorities(root: Path, device: torch.device | None = None) -> dict[str, Any]:
    # This validates the existing V11/N3/N5 authority chain without consuming
    # any new scientific observation.
    n5_contract, _c1, _baseline, authority_context, _extra = N5._validate_n5_authorities_v2(root)
    bundle = N4._validate_v11_authorities(root)
    if device is not None:
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
    return {"n5_contract": n5_contract, "bundle": bundle, "authority_context": authority_context}


class N6ContinuationSampler:
    """The seed-1234 per-class schedule continued from N5 global step 600."""

    SCHEMA = "lightgen_n6_continuation_sampler.v1"

    def __init__(self, train_indices: torch.Tensor, labels: torch.Tensor) -> None:
        _require(train_indices.dtype == torch.long and labels.dtype == torch.long, "N6 sampler roles must be long tensors")
        _require(train_indices.ndim == 1 and labels.ndim == 1 and train_indices.numel() == 55_000 and labels.numel() == 60_000, "N6 sampler role sizes changed")
        self.seed = SPLIT_SEED
        self.global_offset = N5_GLOBAL_OFFSET
        self.steps = N6_GLOBAL_TERMINAL
        self._cursor = N5_GLOBAL_OFFSET
        generator = torch.Generator(device="cpu").manual_seed(SPLIT_SEED)
        role_labels = labels.index_select(0, train_indices).cpu()
        self._permutations: list[torch.Tensor] = []
        for digit in range(CLASS_COUNT):
            candidates = train_indices[role_labels == digit].cpu()
            _require(candidates.numel() >= self.steps, f"N6 TRAIN role cannot support class {digit}")
            self._permutations.append(candidates.index_select(0, torch.randperm(candidates.numel(), generator=generator)))

    @property
    def cursor(self) -> int:
        return self._cursor

    def batch_at(self, global_step: int) -> tuple[torch.Tensor, torch.Tensor]:
        _require(self.global_offset <= global_step < self.steps, "N6 continuation batch is outside the frozen window")
        return torch.stack([permutation[global_step] for permutation in self._permutations]), torch.arange(CLASS_COUNT, dtype=torch.long)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cursor >= self.steps:
            raise StopIteration("N6 continuation sampler is exhausted")
        result = self.batch_at(self._cursor)
        self._cursor += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "seed": self.seed, "global_offset": self.global_offset, "steps": self.steps, "cursor": self._cursor, "class_count": CLASS_COUNT, "rows_per_class": [int(value.numel()) for value in self._permutations], "permutations": [value.clone() for value in self._permutations]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _require(state.get("schema") == self.SCHEMA and state.get("seed") == self.seed, "N6 sampler schema/seed changed")
        _require(state.get("global_offset") == self.global_offset and state.get("steps") == self.steps, "N6 sampler window changed")
        cursor = state.get("cursor")
        _require(isinstance(cursor, int) and self.global_offset <= cursor <= self.steps, "N6 sampler cursor is invalid")
        permutations = state.get("permutations")
        _require(isinstance(permutations, list) and len(permutations) == CLASS_COUNT, "N6 sampler permutations are invalid")
        for index, value in enumerate(permutations):
            _require(isinstance(value, torch.Tensor) and value.dtype == torch.long and value.shape == self._permutations[index].shape, "N6 sampler permutation shape changed")
        self._permutations = [value.cpu().clone() for value in permutations]
        self._cursor = cursor


class FacetMaterialProbe:
    """Fixed facet transfer with optional observation/gradient capture."""

    def __init__(self, *, nonlinear: bool, capture: bool = False) -> None:
        self.nonlinear = nonlinear
        self.capture = capture
        self.capture_gradient = False
        self.records: list[dict[str, Any]] = []
        self.gradient_inputs: list[torch.Tensor] = []
        self.placements: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.records.clear()
        self.gradient_inputs.clear()
        self.placements.clear()

    def apply(self, field: Any) -> Any:
        tensor = field.tensor
        intensity = tensor.abs().square()
        flat = intensity.flatten(1)
        active = intensity > MATERIAL_THRESHOLD
        transition = (intensity - MATERIAL_THRESHOLD).abs() <= MATERIAL_TRANSITION_HALF_WIDTH
        if self.nonlinear:
            transfer_intensity = MATERIAL_GAIN * torch.relu(MATERIAL_SLOPE * (intensity - MATERIAL_THRESHOLD))
            # The numerator epsilon is the preregistered differentiability
            # stabilizer in the frozen contract; it avoids 0*infinity at the
            # ReLU-suppressed branch while leaving the transfer effectively
            # zero at float32 scale.
            ratio = (transfer_intensity + MATERIAL_EPSILON) / (intensity + MATERIAL_EPSILON)
            output_tensor = tensor * ratio.sqrt()
            output_intensity = output_tensor.abs().square()
        else:
            transfer_intensity = intensity
            output_intensity = intensity
            output_tensor = tensor
        _finite(output_tensor.real, "N6 facet output real")
        _finite(output_tensor.imag, "N6 facet output imag")
        input_power = flat.sum(dim=1)
        output_power = output_intensity.flatten(1).sum(dim=1)
        power_ratio = output_power / input_power.clamp_min(MATERIAL_EPSILON)
        summaries = {
            "input_mean": flat.mean(dim=1),
            "output_mean": output_intensity.flatten(1).mean(dim=1),
            "input_power": input_power,
            "output_power": output_power,
            "power_ratio": power_ratio,
            "active_fraction": active.flatten(1).to(torch.float32).mean(dim=1),
            "suppressed_fraction": (transfer_intensity <= 0.0).flatten(1).to(torch.float32).mean(dim=1),
            "transition_fraction": transition.flatten(1).to(torch.float32).mean(dim=1),
            "local_slope": torch.where(active, torch.full_like(intensity, MATERIAL_SLOPE), torch.zeros_like(intensity)).flatten(1).mean(dim=1),
        }
        for name, value in summaries.items():
            _finite(value, f"N6 facet statistic {name}")
        if self.capture_gradient and tensor.requires_grad:
            self.gradient_inputs.append(tensor)
        placement = {
            "input_plane_id": field.plane.plane_id,
            "input_shape": list(field.plane.shape),
            "input_coordinate_frame": field.plane.coordinate_frame,
            "input_provenance_operation": field.provenance.operation,
            "input_provenance_parameters": dict(field.provenance.parameters),
            "output_plane_id": field.plane.plane_id,
            "output_shape": list(field.plane.shape),
            "crop": False,
            "pad": False,
            "resample": False,
            "new_diffraction_plane": False,
            "material_applied": self.nonlinear,
        }
        self.placements.append(placement)
        if self.capture:
            self.records.append({name: value.detach().cpu().to(torch.float64) for name, value in summaries.items()})
        if not self.nonlinear:
            return field
        provenance_type = type(field.provenance)
        output_provenance = provenance_type(
            "n6_fixed_nonlinear_fiber_facet_activation",
            field.plane.plane_id,
            None,
            {
                "intensity_threshold": MATERIAL_THRESHOLD,
                "intensity_slope": MATERIAL_SLOPE,
                "intensity_gain": MATERIAL_GAIN,
                "phase_shift": 0.0,
                "algorithmic_surrogate": True,
            },
        )
        return type(field)(output_tensor, field.plane, output_provenance)


def _install_facet_probe(consumer: Any, *, nonlinear: bool, capture: bool = False) -> FacetMaterialProbe:
    generator = consumer.runtime.generator
    original_forward = generator.forward
    probe = FacetMaterialProbe(nonlinear=nonlinear, capture=capture)

    def forward_with_facet(field: Any, prefix: str) -> Any:
        return original_forward(probe.apply(field), prefix)

    # Store only the probe and the original bound method outside state_dict;
    # the runtime parameter surface remains exactly the canonical two-layer
    # generator surface.
    generator.forward = forward_with_facet
    return probe


def _new_candidate(bundle: Mapping[str, Any], source: Mapping[str, Any], device: torch.device, *, nonlinear: bool, trainable: bool) -> tuple[Any, Any, FacetMaterialProbe]:
    checkpoint = source["checkpoint"]
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V10.V8._load_runtime_state(consumer, dict(checkpoint["runtime_state"]))
    adapter = V11.LowRankAdapter(device=device)
    adapter.load_state_dict(dict(checkpoint["adapter_state"]), strict=True)
    firewall = N5._configure_candidate(consumer, adapter, "A_CONTINUED_C1", trainable=trainable)
    _require(firewall["trainable"] == TRAINABLE_PARAMETER_NAMES if trainable else firewall["trainable"] == [], "N6 trainable firewall surface changed")
    probe = _install_facet_probe(consumer, nonlinear=nonlinear)
    return consumer, adapter, probe


def _optimizer(consumer: Any, adapter: Any) -> torch.optim.Adam:
    return N5._optimizer(consumer, adapter, "A_CONTINUED_C1")


def _restore_optimizer(optimizer: torch.optim.Adam, saved: Mapping[str, Any]) -> None:
    optimizer.load_state_dict(copy.deepcopy(dict(saved)))
    _require(len(optimizer.state) == len(optimizer.param_groups[0]["params"]), "N6 optimizer state restoration is incomplete")


def _train_step(consumer: Any, adapter: Any, optimizer: torch.optim.Adam, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, step: int) -> dict[str, Any]:
    # N5's production helper is reused for the frozen V11 objective and
    # trainable/frozen checks; the only changed boundary is the wrapped facet.
    trace = N5._train_step(consumer, adapter, optimizer, "A_CONTINUED_C1", batch, labels, prototypes, templates, step)
    trace = dict(trace)
    trace["global_step"] = N5_GLOBAL_OFFSET + step
    trace["next"] = "continue within the single N6 A/B comparison"
    return trace


def _train_metric(consumer: Any, adapter: Any, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor) -> dict[str, Any]:
    return N5._measure_train_batch(consumer, adapter, batch, labels, prototypes, templates)


def _gradient_probe(consumer: Any, adapter: Any, probe: FacetMaterialProbe, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, candidate: str) -> dict[str, Any]:
    probe.clear()
    probe.capture_gradient = True
    components = N5._objective_components(consumer, adapter, batch, labels, prototypes, templates, require_grad=True)
    named = N5._candidate_named(consumer, adapter, "A_CONTINUED_C1")
    _require(len(probe.gradient_inputs) >= 2, f"{candidate} facet gradient capture did not see adapted-base and shifted paths")
    records: dict[str, Any] = {}
    targets = tuple(named.values()) + tuple(probe.gradient_inputs[-2:])
    for objective_name, loss in (("semantic", components["semantic_loss"]), ("image", components["image_loss"]), ("combined", components["total_loss"])):
        gradients = torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
        per_parameter: dict[str, float] = {}
        for name, gradient in zip(named, gradients[: len(named)], strict=True):
            _require(gradient is not None, f"{candidate} {objective_name} gradient missing: {name}")
            _finite(gradient, f"{candidate} {objective_name} {name} gradient")
            per_parameter[name] = float(gradient.detach().to(torch.float64).square().sum().sqrt().cpu())
        field_norms: list[float | None] = []
        for field_index, gradient in enumerate(gradients[len(named) :], start=1):
            if gradient is None:
                field_norms.append(None)
                continue
            _finite(gradient.real, f"{candidate} {objective_name} facet field {field_index} real gradient")
            _finite(gradient.imag, f"{candidate} {objective_name} facet field {field_index} imag gradient")
            field_norms.append(float(gradient.detach().abs().to(torch.float64).square().sum().sqrt().cpu()))
        total = math.sqrt(sum(value * value for value in per_parameter.values()))
        _require(math.isfinite(total) and total > 0.0, f"{candidate} {objective_name} trainable gradient is unusable")
        records[objective_name] = {"per_parameter": per_parameter, "total": total, "facet_input_field_gradient_norms": field_norms, "facet_gradient_nonzero": any(value is not None and value > 0.0 for value in field_norms)}
        if objective_name == "combined":
            _require(records[objective_name]["facet_gradient_nonzero"], f"{candidate} combined gradient does not reach the facet")
    wrong_anchor = F.cross_entropy(components["anchor_logits"], (labels + 1) % CLASS_COUNT) + LAMBDA_IMAGE * components["image_loss"]
    wrong_transfer = LAMBDA_TRANSFER * F.softplus(components["delta"]).mean() + LAMBDA_IMAGE * components["image_loss"]
    negatives: dict[str, Any] = {}
    for name, loss in (("wrong_label_anchor", wrong_anchor), ("wrong_direction_transfer", wrong_transfer)):
        gradients = torch.autograd.grad(loss, tuple(named.values()), retain_graph=True, allow_unused=False)
        values = {parameter_name: float(gradient.detach().to(torch.float64).square().sum().sqrt().cpu()) for parameter_name, gradient in zip(named, gradients, strict=True)}
        _require(all(math.isfinite(value) and value > 0.0 for value in values.values()), f"{candidate} negative control gradient is unusable")
        negatives[name] = {"per_parameter": values, "total": math.sqrt(sum(value * value for value in values.values())), "scientific_meaning": "none"}
    probe.capture_gradient = False
    return {
        "candidate": candidate,
        "semantic_image_gradient_defined": True,
        "gradients": records,
        "negative_controls": negatives,
        "facet_records_seen": len(probe.gradient_inputs),
    }


def _facet_observations(probe: FacetMaterialProbe, labels: torch.Tensor) -> dict[str, Any]:
    rows = int(labels.shape[0])
    batches = (rows + LOGICAL_BATCH_SIZE - 1) // LOGICAL_BATCH_SIZE
    _require(len(probe.records) >= 3 * batches, f"N6 facet observation record count is incomplete: {len(probe.records)} < {3 * batches}")
    base_records = [probe.records[3 * index + 1] for index in range(batches)]
    shifted_records = [probe.records[3 * index + 2] for index in range(batches)]
    payload: dict[str, Any] = {"base": {}, "shifted": {}, "dev_labels": labels.detach().cpu(), "placement": probe.placements[1] if len(probe.placements) >= 2 else (probe.placements[0] if probe.placements else {})}
    for target, records in ((payload["base"], base_records), (payload["shifted"], shifted_records)):
        for name in records[0]:
            target[name] = torch.cat([record[name] for record in records], dim=0)[:rows]
    _require(payload["base"]["input_mean"].shape == (rows,), "N6 base facet observation rows changed")
    _require(payload["shifted"]["input_mean"].shape == (rows,), "N6 shifted facet observation rows changed")
    return payload


def _quantiles(value: torch.Tensor) -> dict[str, Any]:
    values = value.detach().to(torch.float64).flatten()
    _finite(values, "N6 facet quantile input")
    quantile = torch.quantile(values, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
    return {"min": float(quantile[0]), "q25": float(quantile[1]), "median": float(quantile[2]), "q75": float(quantile[3]), "max": float(quantile[4]), "mean": float(values.mean())}


def _facet_diagnostics(observations: Mapping[str, Any], *, nonlinear: bool) -> dict[str, Any]:
    base = _mapping(observations.get("base"), "N6 facet base observations")
    labels = observations["dev_labels"].to(torch.long)
    result: dict[str, Any] = {
        "material_kind": "ALGORITHMIC_NONLINEAR_FACET_CANARY" if nonlinear else "LINEAR_IDENTITY_FACET_CONTROL",
        "material_applied": nonlinear,
        "input_intensity_distribution": _quantiles(base["input_mean"]),
        "output_intensity_distribution": _quantiles(base["output_mean"]),
        "input_power_distribution": _quantiles(base["input_power"]),
        "output_power_distribution": _quantiles(base["output_power"]),
        "power_ratio_distribution": _quantiles(base["power_ratio"]),
        "transition_fraction": float(base["transition_fraction"].to(torch.float64).mean()),
        "suppressed_fraction": float(base["suppressed_fraction"].to(torch.float64).mean()),
        "activation_fraction": float(base["active_fraction"].to(torch.float64).mean()),
        "local_slope_distribution": _quantiles(base["local_slope"]),
        "material_derivative_nonzero_fraction": float(base["active_fraction"].to(torch.float64).mean()),
        "per_class_activation_fraction": [float(base["active_fraction"][labels == digit].to(torch.float64).mean()) for digit in range(CLASS_COUNT)],
        "placement": dict(observations.get("placement", {})),
        "computed_without_forward_from_persisted_observations": True,
    }
    return result


def _checkpoint_payload(root: Path, contract_ref: Mapping[str, Any], source: Mapping[str, Any], candidate: str, consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: N6ContinuationSampler, step: int) -> dict[str, Any]:
    return {
        "schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_checkpoint.v1",
        "version": 1,
        "task_id": TASK_ID,
        "candidate": candidate,
        "contract": dict(contract_ref),
        "checkpoint_step": step,
        "global_step": N5_GLOBAL_OFFSET + step,
        "runtime_state": _cpu_tree(consumer.runtime.state_dict()),
        "adapter_state": _cpu_tree(adapter.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "rng_state": _cpu_tree(V10.V6._capture_rng()),
        "source_n5_a_checkpoint": {"path": N5_A_CHECKPOINT_PATH.as_posix(), "sha256": N5_A_CHECKPOINT_SHA256, "checkpoint_step": 300, "global_step": 600},
        "material": {"kind": "ALGORITHMIC_NONLINEAR_FACET_CANARY", "nonlinear": candidate == "B_NONLINEAR_FACET", "threshold": MATERIAL_THRESHOLD, "slope": MATERIAL_SLOPE, "gain": MATERIAL_GAIN, "phase_shift": 0.0, "trainable": False},
        "objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image",
        "rank": ADAPTER_RANK,
        "trainable_parameter_names": TRAINABLE_PARAMETER_NAMES,
        "data_roles": {"TRAIN": True, "DEV": True, "protected": False, "official_test": False, "sealed": False},
        "stage_b_executed": False,
        "generation_executed": False,
        "formal_promotion": False,
    }


def _new_from_payload(bundle: Mapping[str, Any], payload: Mapping[str, Any], device: torch.device) -> tuple[Any, Any, FacetMaterialProbe]:
    source = {"checkpoint": payload}
    return _new_candidate(bundle, source, device, nonlinear=bool(payload.get("material", {}).get("nonlinear", False)), trainable=True)


def _save_checkpoint(root: Path, output_dir: Path, bundle: Mapping[str, Any], contract_ref: Mapping[str, Any], source: Mapping[str, Any], candidate: str, consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: N6ContinuationSampler, step: int) -> dict[str, Any]:
    payload = _checkpoint_payload(root, contract_ref, source, candidate, consumer, adapter, optimizer, sampler, step)
    path = output_dir / CHECKPOINT_NAME.format(candidate=candidate.lower(), step=f"{step:03d}")
    N4.atomic_torch_save_new(path, payload, artifact_kind=f"N6 {candidate} checkpoint")
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"N6 {candidate} checkpoint reload")
    _require(loaded.get("schema") == payload["schema"] and loaded.get("checkpoint_step") == step and loaded.get("candidate") == candidate, f"N6 {candidate} checkpoint reload identity changed")
    clone, clone_adapter, _probe = _new_from_payload(bundle, loaded, consumer.device)
    _require(_digest_tree(_cpu_tree(clone.runtime.state_dict())) == _digest_tree(_cpu_tree(loaded["runtime_state"])), f"N6 {candidate} runtime reload changed")
    _require(_digest_tree(_cpu_tree(clone_adapter.state_dict())) == _digest_tree(_cpu_tree(loaded["adapter_state"])), f"N6 {candidate} adapter reload changed")
    clone_optimizer = _optimizer(clone, clone_adapter)
    _restore_optimizer(clone_optimizer, loaded["optimizer_state"])
    _require(_tree_equal(_cpu_tree(clone_optimizer.state_dict()), loaded["optimizer_state"]), f"N6 {candidate} optimizer reload changed")
    sampler_clone = N6ContinuationSampler(bundle["train_indices"], bundle["labels"])
    sampler_clone.load_state_dict(loaded["sampler_state"])
    current_rng = V10.V6._capture_rng()
    V10.V6._restore_rng(loaded["rng_state"])
    reloaded_rng = V10.V6._capture_rng()
    V10.V6._restore_rng(current_rng)
    _require(_tree_equal(reloaded_rng, loaded["rng_state"]), f"N6 {candidate} RNG reload changed")
    del clone, clone_adapter, clone_optimizer, sampler_clone
    return {"path": _relative(root, path), "sha256": digest, "step": step, "global_step": N5_GLOBAL_OFFSET + step, "reloaded": True, "resume_identity": True, "runtime_state_digest": _digest_tree(_cpu_tree(payload["runtime_state"])), "adapter_state_digest": _digest_tree(_cpu_tree(payload["adapter_state"]))}


def _save_tensor_artifact(root: Path, path: Path, payload: Mapping[str, Any], name: str) -> dict[str, str]:
    N4.atomic_torch_save_new(path, dict(payload), artifact_kind=name)
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"{name} reload")
    _require(loaded.get("schema") == payload.get("schema"), f"{name} schema reload changed")
    return {"path": _relative(root, path), "sha256": digest}


def _canonical_sufficient(contract_ref: Mapping[str, Any], candidate: str, step: int, evaluation: Mapping[str, Any], cyclic: Mapping[str, Any], labels: torch.Tensor, train_metric: Mapping[str, Any], facet: Mapping[str, Any]) -> dict[str, Any]:
    payload = N5._canonical_sufficient_payload(contract_ref, candidate, step, evaluation, cyclic, labels, train_metric)
    payload.update({
        "schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_sufficient_observations.v1",
        "task_id": TASK_ID,
        "candidate": candidate,
        "global_step": N5_GLOBAL_OFFSET + step,
        "facet_observations": facet,
        "facet_material": {"kind": "ALGORITHMIC_NONLINEAR_FACET_CANARY", "threshold": MATERIAL_THRESHOLD, "slope": MATERIAL_SLOPE, "gain": MATERIAL_GAIN, "phase_shift": 0.0, "trainable": False},
        "forward_mode": "scientific evaluation; sufficient observations persisted before facet diagnostics",
    })
    return payload


def _save_sufficient(root: Path, output_dir: Path, contract_ref: Mapping[str, Any], candidate: str, step: int, payload: Mapping[str, Any]) -> dict[str, str]:
    path = output_dir / SUFFICIENT_NAME.format(candidate=candidate.lower(), step=f"{step:03d}")
    return _save_tensor_artifact(root, path, payload, f"N6 {candidate} sufficient observations step {step}")


def _save_diagnostics(root: Path, output_dir: Path, contract_ref: Mapping[str, Any], candidate: str, step: int, sufficient_ref: Mapping[str, Any], material_diag: Mapping[str, Any], adapter_diag: Mapping[str, Any], generator_diag: Mapping[str, Any], cyclic_diag: Mapping[str, Any], runtime_digest: str, adapter_digest: str) -> dict[str, str]:
    path = output_dir / DIAGNOSTICS_NAME.format(candidate=candidate.lower(), step=f"{step:03d}")
    payload = {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_diagnostics.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "relative_step": step, "contract": dict(contract_ref), "sufficient_observations": dict(sufficient_ref), "material": dict(material_diag), "adapter": dict(adapter_diag), "generator": dict(generator_diag), "cyclic": dict(cyclic_diag), "runtime_state_digest": runtime_digest, "adapter_state_digest": adapter_digest, "computed_after_sufficient_reload": True}
    N4.atomic_write_new_json(path, payload, artifact_kind=f"N6 {candidate} diagnostics")
    return {"path": _relative(root, path), "sha256": _sha256(path), "reloaded": True}


def _save_raw(root: Path, output_dir: Path, contract_ref: Mapping[str, Any], candidate: str, observations: Mapping[str, Any]) -> dict[str, str]:
    payload = {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_cyclic_raw_outputs.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "contract": dict(contract_ref), "selection_indices": observations["selection_indices"], "labels": observations["labels"], "targets": observations["targets"], "raw_base": observations["raw_base"], "raw_shifted": observations["raw_shifted"]}
    return _save_tensor_artifact(root, output_dir / RAW_NAME.format(candidate=candidate.lower()), payload, f"N6 {candidate} cyclic raw outputs")


def _evaluate_persist_first(consumer: Any, adapter: Any, probe: FacetMaterialProbe, bundle: Mapping[str, Any], dev_images: torch.Tensor, dev_labels: torch.Tensor, selection_images: torch.Tensor, selection_labels: torch.Tensor, selection_indices: torch.Tensor, step: int, train_metric: Mapping[str, Any], train_trace: Mapping[str, Any] | None, candidate: str, initial_phase: torch.Tensor, contract_ref: Mapping[str, Any], output_dir: Path, root: Path, counters: dict[str, int]) -> dict[str, Any]:
    runtime_before = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    adapter_before = _digest_tree(_cpu_tree(adapter.state_dict()))
    probe.clear()
    probe.capture = True
    counters["scientific_forward_evaluations"] += 1
    with torch.no_grad():
        evaluation = N4._evaluate(consumer, adapter, bundle["authorities"]["smooth"], dev_images, dev_labels, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"], step=step, train_trace=train_trace)
    counters["scientific_forward_evaluations"] += 1
    with torch.no_grad():
        cyclic = V10._cyclic_transfer_v10(consumer, adapter, bundle["authorities"]["smooth"], selection_images, selection_labels, selection_indices, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"])
    probe.capture = False
    _require(runtime_before == _digest_tree(_cpu_tree(consumer.runtime.state_dict())), f"{candidate} evaluation changed runtime state")
    _require(adapter_before == _digest_tree(_cpu_tree(adapter.state_dict())), f"{candidate} evaluation changed adapter state")
    facet = _facet_observations(probe, dev_labels)
    canonical = _canonical_sufficient(contract_ref, candidate, step, evaluation, cyclic, dev_labels, train_metric, facet)
    sufficient_ref = _save_sufficient(root, output_dir, contract_ref, candidate, step, canonical)
    loaded_sufficient = _mapping(N4.safe_torch_load(_path(root, sufficient_ref["path"]), expected_sha256=sufficient_ref["sha256"], map_location="cpu"), f"{candidate} sufficient reload")
    facet_diag = _facet_diagnostics(loaded_sufficient["facet_observations"], nonlinear=candidate == "B_NONLINEAR_FACET")
    adapter_diag = N5._adapter_diagnostics(adapter)
    generator_diag = N5._generator_diagnostics(consumer, "A_CONTINUED_C1", initial_phase)
    cyclic_diag = N5._cyclic_diagnostics_from_observations(loaded_sufficient)
    diagnostics_ref = _save_diagnostics(root, output_dir, contract_ref, candidate, step, sufficient_ref, facet_diag, adapter_diag, generator_diag, cyclic_diag, runtime_before, adapter_before)
    raw_ref = _save_raw(root, output_dir, contract_ref, candidate, {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in cyclic["_observations"].items()}) if step == 300 else None
    consumer.runtime.train()
    return {
        "candidate": candidate,
        "relative_step": step,
        "global_step": N5_GLOBAL_OFFSET + step,
        "dev_labels": loaded_sufficient["dev_labels"],
        "dev_correct": loaded_sufficient["dev_smooth_correct"],
        "evaluation_public": loaded_sufficient["evaluation_public"],
        "cyclic_public": loaded_sufficient["cyclic_public"],
        "dev_image_loss": float(loaded_sufficient["dev_image_loss"]),
        "train_metric": dict(train_metric),
        "sufficient_ref": sufficient_ref,
        "diagnostics_ref": diagnostics_ref,
        "facet_diagnostics": facet_diag,
        "adapter_diagnostics": adapter_diag,
        "generator_diagnostics": generator_diag,
        "cyclic_diagnostics": cyclic_diag,
        "cyclic_observations": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in cyclic["_observations"].items()},
        "raw_ref": raw_ref,
    }


def _scientific_payload(candidate: str, started: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_scientific_report.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N6_SCIENTIFIC_HARNESS", "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "provenance": {"git_start": dict(started)}, "claim_boundary": {"scientific_training_executed": False, "scientific_data_consumed": False, "scientific_forward_evaluations": 0, "scientific_updates": 0, "new_optical_depth": False, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {"candidate": candidate, "rank": ADAPTER_RANK, "new_optical_depth": False, "material_trainable": False, "retry": False, "second_arm": False, "rank_change": False, "loss_change": False, "lambda_change": False, "lr_change": False, "encoder_update": False, "generator_depth_change": False, "stage_b": False, "generation": False, "promotion": False}}


def _preflight_smoke(root: Path, output_dir: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], contract_ref: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, device: torch.device, *, nonlinear: bool, candidate: str) -> dict[str, Any]:
    V10.V6._restore_rng(source["checkpoint"]["rng_state"])
    consumer, adapter, probe = _new_candidate(bundle, source, device, nonlinear=nonlinear, trainable=True)
    optimizer = _optimizer(consumer, adapter)
    _restore_optimizer(optimizer, source["checkpoint"]["optimizer_state"])
    sampler = N6ContinuationSampler(bundle["train_indices"], bundle["labels"])
    traces = []
    probe.capture = False
    for step in range(1, PRECHECK_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        batch = images.index_select(0, indices).to(device)
        traces.append(_train_step(consumer, adapter, optimizer, batch, batch_labels.to(device), bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), step))
    checkpoint = _checkpoint_payload(root, contract_ref, source, candidate, consumer, adapter, optimizer, sampler, PRECHECK_STEPS)
    path = output_dir / PREFLIGHT_CHECKPOINT_NAME.format(candidate=candidate.lower())
    N4.atomic_torch_save_new(path, checkpoint, artifact_kind=f"N6 {candidate} disposable preflight checkpoint")
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"N6 {candidate} disposable checkpoint reload")
    sampler_clone = N6ContinuationSampler(bundle["train_indices"], bundle["labels"])
    sampler_clone.load_state_dict(loaded["sampler_state"])
    _require(sampler_clone.cursor == N5_GLOBAL_OFFSET + PRECHECK_STEPS, "N6 preflight sampler resume cursor changed")
    return {"candidate": candidate, "firewall": {"trainable": TRAINABLE_PARAMETER_NAMES, "encoder_updated": False, "ols_updated": False, "material_trainable": False}, "traces": traces, "disposable_updates": PRECHECK_STEPS, "scientific_updates": 0, "checkpoint_reload": {"path": _relative(root, path), "sha256": digest, "reloaded": True, "resume_identity": True}, "sampler_cursor_after_smoke": sampler.cursor}


def _synthetic_chain(root: Path, output_dir: Path, bundle: Mapping[str, Any], source: Mapping[str, Any], contract_ref: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    consumer, adapter, probe = _new_candidate(bundle, source, device, nonlinear=True, trainable=False)
    ramp = torch.linspace(0.0, 1.0, 10 * 28 * 28, dtype=torch.float32).reshape(10, 1, 28, 28)
    labels = torch.arange(CLASS_COUNT, dtype=torch.long)
    probe.clear()
    probe.capture = True
    with torch.no_grad():
        evaluation = N4._evaluate(consumer, adapter, bundle["authorities"]["smooth"], ramp.to(device), labels, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"], step=0, train_trace=None)
        cyclic = V10._cyclic_transfer_v10(consumer, adapter, bundle["authorities"]["smooth"], ramp.to(device), labels, torch.arange(CLASS_COUNT), bundle["authorities"]["prototypes"], bundle["authorities"]["templates"])
    probe.capture = False
    facet = _facet_observations(probe, labels)
    canonical = _canonical_sufficient(contract_ref, "B_NONLINEAR_FACET", 0, evaluation, cyclic, labels, {"scientific_meaning": "none"}, facet)
    obs_path = output_dir / PREFLIGHT_CHAIN_OBS_NAME
    obs_ref = _save_tensor_artifact(root, obs_path, canonical, "N6 synthetic production-chain sufficient observations")
    loaded = _mapping(N4.safe_torch_load(obs_path, expected_sha256=obs_ref["sha256"], map_location="cpu"), "N6 synthetic sufficient reload")
    material_diag = _facet_diagnostics(loaded["facet_observations"], nonlinear=True)
    report_payload = {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_synthetic_chain_report.v1", "version": 1, "task_id": TASK_ID, "result": "VALID_COMPLETED", "scientific_meaning": "none", "sufficient_observations": obs_ref, "material": material_diag, "chain": ["forward-result", "canonical sufficient builder", "write-once persistence", "reload/hash/shape verification", "diagnostics from persisted observations", "report", "lineage"], "scientific_forward": False, "scientific_updates": 0}
    report_path = output_dir / PREFLIGHT_CHAIN_REPORT_NAME
    N4.atomic_write_new_json(report_path, report_payload, artifact_kind="N6 synthetic production-chain report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = output_dir / PREFLIGHT_CHAIN_LINEAGE_NAME
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_synthetic_chain_lineage.v1", "version": 1, "task_id": TASK_ID, "result": "VALID_COMPLETED", "scientific_meaning": "none", "report": report_ref, "scientific_forward": False, "scientific_updates": 0}, artifact_kind="N6 synthetic production-chain lineage")
    return {"status": "PASS", "observation": obs_ref, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "material": material_diag, "scientific_forward": False, "scientific_updates": 0}


def _preflight_report_payload(started: Mapping[str, Any], contract_ref: Mapping[str, Any], device_facts: Mapping[str, Any], source: Mapping[str, Any], placement: Mapping[str, Any], equivalence: Mapping[str, Any], gradients: Mapping[str, Any], smokes: Mapping[str, Any], chain: Mapping[str, Any], dataset: Mapping[str, Any], first_indices: torch.Tensor, first_labels: torch.Tensor) -> dict[str, Any]:
    return {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "AUTHORIZABLE", "status": "valid_completed", "classification": "LIGHTGEN_2D_N6_NONLINEAR_FIBER_FACET_CANARY_AUTHORIZABLE", "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "contract": dict(contract_ref), "authority": {"n5_a_report": source["report_ref"], "n5_a_checkpoint": source["checkpoint_ref"], "n5_contract_sha256": N5_CONTRACT_SHA256}, "runtime": {"device": dict(device_facts), "production_grid": PRODUCTION_GRID, "canonical_generator_layers": 2}, "starting_state": dict(equivalence), "facet_placement": dict(placement), "gradient_probe": gradients, "smoke": smokes, "synthetic_production_chain": chain, "material": {"kind": "ALGORITHMIC_NONLINEAR_FACET_CANARY", "measured_transfer_curve": False, "threshold": MATERIAL_THRESHOLD, "slope": MATERIAL_SLOPE, "gain": MATERIAL_GAIN, "phase_shift": 0.0, "trainable": False}, "sampler": {"schema": N6ContinuationSampler.SCHEMA, "seed": SPLIT_SEED, "global_offset": N5_GLOBAL_OFFSET, "terminal_global_step": N6_GLOBAL_TERMINAL, "first_batch_indices_sha256": V10.V6.tensor_sha256(first_indices), "first_batch_labels_sha256": V10.V6.tensor_sha256(first_labels)}, "data_roles": _public(dataset), "preflight_checks": {"authorities": True, "n5_terminal_closure": True, "n5_a_endpoint_resolved": True, "a_b_starting_state_exact": True, "physical_facet_placement": True, "no_crop_pad_resample": True, "finite_material_output": True, "usable_adapter_and_generator_gradients": True, "frozen_encoder_ols_material": True, "three_step_real_updates": True, "checkpoint_reload_resume": True, "exact_core_synthetic_persistence": True}, "claim_boundary": {"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_updates": 0, "scientific_forward_evaluations": 0, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}, "next": "proceed to exactly one N6 A/B 300-update scientific comparison"}


def run_preflight(root: Path) -> dict[str, Any]:
    _require(not PREFLIGHT_DIR.exists(), f"refusing to reuse N6 preflight directory: {_path(root, PREFLIGHT_DIR)}")
    for directory in (CANDIDATE_A_DIR, CANDIDATE_B_DIR, AGGREGATE_DIR):
        _require(not directory.exists(), f"N6 one-shot output boundary is not fresh: {_path(root, directory)}")
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = N5._status_lines(root)
    payload: dict[str, Any] = {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N6_PREFLIGHT_HARNESS", "provenance": {"git_start": started}, "claim_boundary": {"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_updates": 0, "scientific_forward_evaluations": 0}}
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from frozen N6 authorization")
        _contract(root)
        source = _load_n5_a(root)
        authorities = _resolve_authorities(root)
        V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        _require(device.type == "cuda", "CUDA is required for N6 scientific execution")
        bundle = authorities["bundle"]
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V10.V6._load_roles(root)
        bundle.update({"images": images, "labels": labels, "train_indices": train_indices, "development_indices": development_indices})
        _require(dataset.get("protected", 0) in (0, [], None) and dataset.get("sealed", 0) in (0, [], None), "N6 protected/sealed data role is not empty")
        sampler = N6ContinuationSampler(train_indices, labels)
        first_indices, first_labels = sampler.batch_at(N5_GLOBAL_OFFSET)
        first_batch = images.index_select(0, first_indices).to(device)
        contract_ref = {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256}
        V10.V6._restore_rng(source["checkpoint"]["rng_state"])
        consumer_a, adapter_a, probe_a = _new_candidate(bundle, source, device, nonlinear=False, trainable=True)
        V10.V6._restore_rng(source["checkpoint"]["rng_state"])
        consumer_b, adapter_b, probe_b = _new_candidate(bundle, source, device, nonlinear=True, trainable=True)
        optimizer_a = _optimizer(consumer_a, adapter_a)
        optimizer_b = _optimizer(consumer_b, adapter_b)
        _restore_optimizer(optimizer_a, source["checkpoint"]["optimizer_state"])
        _restore_optimizer(optimizer_b, source["checkpoint"]["optimizer_state"])
        runtime_a = _cpu_tree(consumer_a.runtime.state_dict())
        runtime_b = _cpu_tree(consumer_b.runtime.state_dict())
        adapter_state_a = _cpu_tree(adapter_a.state_dict())
        adapter_state_b = _cpu_tree(adapter_b.state_dict())
        _require(_tree_equal(runtime_a, runtime_b) and _tree_equal(adapter_state_a, adapter_state_b), "N6 A/B starting states are not exact")
        equivalence = {"runtime_digest": _digest_tree(runtime_a), "adapter_digest": _digest_tree(adapter_state_a), "optimizer_digest": _digest_tree(_cpu_tree(optimizer_a.state_dict())), "runtime_exact": True, "adapter_exact": True, "optimizer_exact": True, "starting_global_step": N5_GLOBAL_OFFSET, "source_checkpoint": source["checkpoint_ref"]}
        probe_b.capture = True
        with torch.no_grad():
            forward = V10._forward_adapter(consumer_b, adapter_b, first_batch, require_grad=False)
        probe_b.capture = False
        _require(len(probe_b.placements) >= 2, "N6 physical facet placement was not observed")
        placement = dict(probe_b.placements[1])
        _require(placement["input_provenance_operation"] == "lightgen_ols_coefficient_transmission_and_modal_reemission", "N6 material was not placed after canonical W_OLS re-emission")
        _require(placement["input_provenance_parameters"].get("transport") == "T=I_100_coefficient_space", "N6 material placement changed coefficient-space transport")
        _require(tuple(placement["input_shape"]) == (100, 100) and placement["crop"] is False and placement["pad"] is False and placement["resample"] is False and placement["new_diffraction_plane"] is False, "N6 material changed facet geometry")
        _finite(forward["raw_base"], "N6 preflight raw output")
        gradients: dict[str, Any] = {}
        for candidate, consumer, adapter, probe in (("A_LINEAR_CONTINUATION_CONTROL", consumer_a, adapter_a, probe_a), ("B_NONLINEAR_FACET", consumer_b, adapter_b, probe_b)):
            V10.V6._restore_rng(source["checkpoint"]["rng_state"])
            gradients[candidate] = _gradient_probe(consumer, adapter, probe, first_batch, first_labels.to(device), bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), candidate)
        smokes = {
            "A_LINEAR_CONTINUATION_CONTROL": _preflight_smoke(root, PREFLIGHT_DIR, bundle, source, contract_ref, images, labels, device, nonlinear=False, candidate="A_LINEAR_CONTINUATION_CONTROL"),
            "B_NONLINEAR_FACET": _preflight_smoke(root, PREFLIGHT_DIR, bundle, source, contract_ref, images, labels, device, nonlinear=True, candidate="B_NONLINEAR_FACET"),
        }
        chain = _synthetic_chain(root, PREFLIGHT_DIR, bundle, source, contract_ref, device)
        input_payload = {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_input.v1", "version": 1, "task_id": TASK_ID, "design_authority": DESIGN_AUTHORITY, "outer_gate": OUTER_GATE, "contract": contract_ref, "n5_a_endpoint": source["checkpoint_ref"], "candidates": ["A_LINEAR_CONTINUATION_CONTROL", "B_NONLINEAR_FACET"], "material": {"kind": "ALGORITHMIC_NONLINEAR_FACET_CANARY", "threshold": MATERIAL_THRESHOLD, "slope": MATERIAL_SLOPE, "gain": MATERIAL_GAIN, "phase_shift": 0.0, "trainable": False}, "objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "sampler": {"schema": N6ContinuationSampler.SCHEMA, "global_offset": N5_GLOBAL_OFFSET, "terminal_global_step": N6_GLOBAL_TERMINAL}, "data_roles": {"TRAIN": 55000, "DEV": 5000, "protected": 0, "official_test": 0, "sealed": 0}, "scientific_updates": 0}
        input_path = PREFLIGHT_DIR / PREFLIGHT_INPUT_NAME
        N4.atomic_write_new_json(input_path, input_payload, artifact_kind="N6 preflight input")
        payload = _preflight_report_payload(started, contract_ref, device_facts, source, placement, equivalence, gradients, smokes, chain, dataset, first_indices, first_labels)
        payload["input"] = {"path": _relative(root, input_path), "sha256": _sha256(input_path)}
        payload["provenance"] = {"git_start": started, "git_end": V10.V6.git_facts(root)}
    except HardBlockedN6 as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": "N6_PREFLIGHT_HARD_BLOCKED", "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific execution"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N6_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR", "invalid_input": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "stop; no scientific execution"})
    finally:
        ending = V10.V6.git_facts(root)
        payload.setdefault("provenance", {}).update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == N5._status_lines(root)})
    report_path = PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME
    N4.atomic_write_new_json(report_path, payload, artifact_kind="N6 preflight report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "report": report_ref, "provenance": payload.get("provenance"), "claim_boundary": payload["claim_boundary"]}, artifact_kind="N6 preflight lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report_ref": report_ref, "lineage_ref": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "scientific_updates": 0}


def _load_preflight(root: Path) -> dict[str, Any]:
    report_path = _path(root, PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = _path(root, PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    _require(report_path.is_file() and lineage_path.is_file(), "N6 preflight authority is unavailable")
    report = _read_json(report_path, "N6 preflight report")
    lineage = _read_json(lineage_path, "N6 preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed" and report.get("classification") == "LIGHTGEN_2D_N6_NONLINEAR_FIBER_FACET_CANARY_AUTHORIZABLE", "N6 preflight is not authorizable")
    _require(report.get("claim_boundary", {}).get("scientific_updates") == 0 and report.get("claim_boundary", {}).get("scientific_training_executed") is False, "N6 preflight scientific boundary changed")
    report_hash = _sha256(report_path)
    _require(lineage.get("report", {}).get("sha256") == report_hash, "N6 preflight lineage binding changed")
    return {"report": report, "report_ref": {"path": _relative(root, report_path), "sha256": report_hash}, "lineage_ref": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}}


def run_candidate(root: Path, candidate: str, *, nonlinear: bool) -> dict[str, Any]:
    output_dir = CANDIDATE_B_DIR if nonlinear else CANDIDATE_A_DIR
    _require(not output_dir.exists(), f"refusing to reuse N6 {candidate} output directory: {_path(root, output_dir)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = N5._status_lines(root)
    payload = _scientific_payload(candidate, started)
    scientific_started = False
    updates_completed = 0
    counters = {"scientific_forward_evaluations": 0}
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from frozen N6 authorization")
        _contract(root)
        preflight = _load_preflight(root)
        source = _load_n5_a(root)
        authorities = _resolve_authorities(root)
        V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        _require(device.type == "cuda", "CUDA is required for N6 scientific execution")
        bundle = authorities["bundle"]
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V10.V6._load_roles(root)
        bundle.update({"images": images, "labels": labels, "train_indices": train_indices, "development_indices": development_indices})
        dev_images = images.index_select(0, development_indices).contiguous()
        dev_labels = labels.index_select(0, development_indices).contiguous()
        selection_images, selection_labels, selection_indices = V10.V6._load_selection(root, images, labels, development_indices)
        continuation = N6ContinuationSampler(train_indices, labels)
        V10.V6._restore_rng(source["checkpoint"]["rng_state"])
        consumer, adapter, probe = _new_candidate(bundle, source, device, nonlinear=nonlinear, trainable=True)
        optimizer = _optimizer(consumer, adapter)
        _restore_optimizer(optimizer, source["checkpoint"]["optimizer_state"])
        initial_phase = source["checkpoint"]["runtime_state"]["generator.phase"].detach().cpu()
        first_indices, first_labels = continuation.batch_at(N5_GLOBAL_OFFSET)
        first_batch = images.index_select(0, first_indices).to(device)
        contract_ref = {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256}
        preflight_ref = {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"], "result": "AUTHORIZABLE"}
        input_payload = {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_scientific_input.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "design_authority": DESIGN_AUTHORITY, "outer_gate": OUTER_GATE, "contract": contract_ref, "preflight": preflight_ref, "source_n5_a_checkpoint": source["checkpoint_ref"], "protocol": {"objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "rank": ADAPTER_RANK, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "batch": LOGICAL_BATCH_SIZE, "additional_updates": SCIENTIFIC_STEPS, "relative_checkpoints": list(CHECKPOINT_STEPS), "global_offset": N5_GLOBAL_OFFSET, "global_terminal_step": N6_GLOBAL_TERMINAL, "trainable": TRAINABLE_PARAMETER_NAMES}, "facet_material": {"kind": "ALGORITHMIC_NONLINEAR_FACET_CANARY", "nonlinear": nonlinear, "threshold": MATERIAL_THRESHOLD, "slope": MATERIAL_SLOPE, "gain": MATERIAL_GAIN, "phase_shift": 0.0, "trainable": False}, "data_roles": {"TRAIN": 55000, "DEV": 5000, "protected": 0, "official_test": 0, "sealed": 0}, "one_shot": {"candidate_attempt": 1, "retry": False, "extra_candidate": False}, "claim_boundary": payload["claim_boundary"]}
        input_path = output_dir / SCIENTIFIC_INPUT_NAME.format(candidate=candidate.lower())
        N4.atomic_write_new_json(input_path, input_payload, artifact_kind=f"N6 {candidate} scientific input")
        scientific_started = True
        probe.capture = False
        with torch.no_grad():
            initial_train_metric = _train_metric(consumer, adapter, first_batch, first_labels.to(device), bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device))
        payload.update({"contract": contract_ref, "preflight": preflight_ref, "source_n5_a_checkpoint": source["checkpoint_ref"], "runtime": {"device": device_facts, "production_grid": PRODUCTION_GRID, "candidate": candidate}, "authorities": {"n5_a_report": source["report_ref"], "n5_a_checkpoint": source["checkpoint_ref"], "n5_contract_sha256": N5_CONTRACT_SHA256}, "dataset": _public(dataset), "firewall": {"trainable": TRAINABLE_PARAMETER_NAMES, "encoder_updated": False, "ols_updated": False, "material_trainable": False}, "objective": input_payload["protocol"], "facet_material": input_payload["facet_material"], "initial_state": {"source_runtime_digest": _digest_tree(_cpu_tree(source["checkpoint"]["runtime_state"])), "source_adapter_digest": _digest_tree(_cpu_tree(source["checkpoint"]["adapter_state"])), "candidate_runtime_digest": _digest_tree(_cpu_tree(consumer.runtime.state_dict())), "candidate_adapter_digest": _digest_tree(_cpu_tree(adapter.state_dict())), "starting_global_step": N5_GLOBAL_OFFSET, "input": {"path": _relative(root, input_path), "sha256": _sha256(input_path)}}, "sampler": {"schema": N6ContinuationSampler.SCHEMA, "seed": SPLIT_SEED, "global_offset": N5_GLOBAL_OFFSET, "terminal_global_step": N6_GLOBAL_TERMINAL, "first_batch_indices_sha256": V10.V6.tensor_sha256(first_indices), "first_batch_labels_sha256": V10.V6.tensor_sha256(first_labels)}, "data_roles": input_payload["data_roles"], "claim_boundary": {**payload["claim_boundary"], "scientific_training_executed": True, "scientific_data_consumed": True}})
        records: dict[str, Any] = {}
        train_metrics: dict[str, Any] = {"0": initial_train_metric}
        sufficient_refs: dict[str, Any] = {}
        diagnostics_refs: dict[str, Any] = {}
        checkpoint_refs: dict[str, Any] = {}
        training_start = time.perf_counter()
        record = _evaluate_persist_first(consumer, adapter, probe, bundle, dev_images, dev_labels, selection_images, selection_labels, selection_indices, 0, initial_train_metric, None, candidate, initial_phase, contract_ref, output_dir, root, counters)
        records["0"] = record
        sufficient_refs["0"] = record["sufficient_ref"]
        diagnostics_refs["0"] = record["diagnostics_ref"]
        checkpoint_refs["0"] = _save_checkpoint(root, output_dir, bundle, contract_ref, source, candidate, consumer, adapter, optimizer, continuation, 0)
        for step in range(1, SCIENTIFIC_STEPS + 1):
            indices, batch_labels = continuation.next_batch()
            batch = images.index_select(0, indices).to(device)
            trace = _train_step(consumer, adapter, optimizer, batch, batch_labels.to(device), bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), step)
            updates_completed = step
            del batch
            if step in CHECKPOINT_STEPS[1:]:
                with torch.no_grad():
                    train_metric = _train_metric(consumer, adapter, first_batch, first_labels.to(device), bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device))
                train_metrics[str(step)] = train_metric
                record = _evaluate_persist_first(consumer, adapter, probe, bundle, dev_images, dev_labels, selection_images, selection_labels, selection_indices, step, train_metric, trace, candidate, initial_phase, contract_ref, output_dir, root, counters)
                records[str(step)] = record
                sufficient_refs[str(step)] = record["sufficient_ref"]
                diagnostics_refs[str(step)] = record["diagnostics_ref"]
                checkpoint_refs[str(step)] = _save_checkpoint(root, output_dir, bundle, contract_ref, source, candidate, consumer, adapter, optimizer, continuation, step)
                print(json.dumps({"event": "N6_scientific_progress", "candidate": candidate, "relative_step": step, "global_step": N5_GLOBAL_OFFSET + step, "elapsed_seconds": time.perf_counter() - training_start}, ensure_ascii=False), flush=True)
        _require(set(records) == {"0", "100", "200", "300"} and set(sufficient_refs) == {"0", "100", "200", "300"} and set(checkpoint_refs) == {"0", "100", "200", "300"}, f"{candidate} endpoint set is incomplete")
        final_record = records["300"]
        step0_correct = records["0"]["dev_correct"].to(torch.float64)
        final_correct = final_record["dev_correct"].to(torch.float64)
        training_ci = V10.V6._paired_bootstrap(final_correct - step0_correct, dev_labels.cpu(), replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        reconstruction_delta = float(final_record["dev_image_loss"] - records["0"]["dev_image_loss"])
        raw_valid = bool(all(record["evaluation_public"]["raw_validity"]["finite"] and record["evaluation_public"]["raw_validity"]["structured"] and record["evaluation_public"]["raw_validity"]["nonconstant"] for record in records.values()) and final_record["cyclic_public"]["raw_validity"]["base"]["finite"] and final_record["cyclic_public"]["raw_validity"]["shifted"]["finite"])
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": f"N6_{'NONLINEAR_FACET' if nonlinear else 'LINEAR_CONTINUATION'}_SCIENTIFIC_COMPLETED", "scientific_updates": SCIENTIFIC_STEPS, "scientific_forward_evaluations": counters["scientific_forward_evaluations"], "evaluations": {step: record["evaluation_public"] for step, record in records.items()}, "train_metrics": train_metrics, "sufficient_observations": sufficient_refs, "diagnostics_artifacts": diagnostics_refs, "checkpoints": checkpoint_refs, "cyclic_raw_outputs": final_record["raw_ref"], "training_delta": {"R0": float(records["0"]["evaluation_public"]["smooth"]["macro_recall"]), "R100": float(records["100"]["evaluation_public"]["smooth"]["macro_recall"]), "R200": float(records["200"]["evaluation_public"]["smooth"]["macro_recall"]), "R300": float(final_record["evaluation_public"]["smooth"]["macro_recall"]), "Delta_R": float(final_record["evaluation_public"]["smooth"]["macro_recall"] - records["0"]["evaluation_public"]["smooth"]["macro_recall"]), "paired_bootstrap": {key: value for key, value in training_ci.items() if key != "delta_records"}}, "reconstruction": {"DEV_L_image_step0": float(records["0"]["dev_image_loss"]), "DEV_L_image_step300": float(final_record["dev_image_loss"]), "Delta_L_image": reconstruction_delta, "max_allowed_delta": RECONSTRUCTION_MAX_DELTA}, "cyclic_transfer": final_record["cyclic_public"], "raw_diversity": {"step300": {"dev": final_record["evaluation_public"]["raw_validity"], "collapse": final_record["evaluation_public"]["collapse"], "cyclic": final_record["cyclic_public"]["raw_validity"]}}, "facet_diagnostics": {step: record["facet_diagnostics"] for step, record in records.items()}, "diagnostics": {"step0": {"adapter": records["0"]["adapter_diagnostics"], "generator": records["0"]["generator_diagnostics"], "material": records["0"]["facet_diagnostics"]}, "step300": {"adapter": final_record["adapter_diagnostics"], "generator": final_record["generator_diagnostics"], "material": final_record["facet_diagnostics"]}}, "raw_validity_pass": raw_valid, "claim_boundary": {**payload["claim_boundary"], "scientific_training_executed": True, "scientific_data_consumed": True, "scientific_forward_evaluations": counters["scientific_forward_evaluations"], "scientific_updates": SCIENTIFIC_STEPS}, "governance": {**payload["governance"], "scientific_attempt": 1, "adapter_update": True, "generator_phase_update": True, "material_trainable": False, "new_optical_depth": False, "retries": False}, "next": "candidate result is bounded input to the frozen N6 A/B outer gate; Stage B and generation remain closed"})
    except HardBlockedN6 as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": "N6_SCIENTIFIC_HARD_BLOCKED", "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific retry"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N6_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "invalid_input": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "stop; no scientific retry; persisted observations remain immutable"})
    finally:
        ending = V10.V6.git_facts(root)
        payload.setdefault("provenance", {}).update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == N5._status_lines(root), "scientific_started": scientific_started})
        payload["claim_boundary"]["scientific_training_executed"] = bool(scientific_started)
        payload["claim_boundary"]["scientific_data_consumed"] = bool(scientific_started)
        payload["claim_boundary"]["scientific_updates"] = updates_completed
        payload["claim_boundary"]["scientific_forward_evaluations"] = counters["scientific_forward_evaluations"]
    report_path = output_dir / SCIENTIFIC_REPORT_NAME.format(candidate=candidate.lower())
    N4.atomic_write_new_json(report_path, payload, artifact_kind=f"N6 {candidate} scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = output_dir / SCIENTIFIC_LINEAGE_NAME.format(candidate=candidate.lower())
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_scientific_lineage.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "report": report_ref, "contract": payload.get("contract"), "preflight": payload.get("preflight"), "provenance": payload["provenance"], "claim_boundary": payload["claim_boundary"], "scientific_observations_new": bool(scientific_started), "retry": False}, artifact_kind=f"N6 {candidate} scientific lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "candidate": candidate, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "scientific_updates": updates_completed, "scientific_forward_evaluations": counters["scientific_forward_evaluations"], "smooth_R300": payload.get("evaluations", {}).get("300", {}).get("smooth", {}).get("macro_recall"), "next": payload.get("next")}


def _load_step_sufficient(root: Path, report: Mapping[str, Any], candidate: str, step: int) -> Mapping[str, Any]:
    ref = _mapping(report.get("sufficient_observations", {}).get(str(step)), f"N6 {candidate} sufficient step {step}")
    path = _path(root, ref["path"])
    _validate_sha(path, str(ref["sha256"]), f"N6 {candidate} sufficient step {step}")
    return _mapping(N4.safe_torch_load(path, expected_sha256=str(ref["sha256"]), map_location="cpu"), f"N6 {candidate} sufficient step {step}")


def _aggregate(root: Path, contract: Mapping[str, Any], preflight: Mapping[str, Any], a_result: Mapping[str, Any], b_result: Mapping[str, Any], a_report: Mapping[str, Any], b_report: Mapping[str, Any], a_steps: Mapping[str, Mapping[str, Any]], b_steps: Mapping[str, Mapping[str, Any]], source: Mapping[str, Any]) -> dict[str, Any]:
    _require(a_result["result"] == "VALID_COMPLETED" and b_result["result"] == "VALID_COMPLETED", "N6 aggregate requires both valid candidates")
    _require(not AGGREGATE_DIR.exists(), f"refusing to reuse N6 aggregate directory: {_path(root, AGGREGATE_DIR)}")
    output_dir = _path(root, AGGREGATE_DIR)
    output_dir.mkdir(parents=True, exist_ok=False)
    labels = a_steps["300"]["dev_labels"]
    _require(torch.equal(labels, b_steps["300"]["dev_labels"]), "N6 A/B DEV labels are not symmetric")
    a300 = a_steps["300"]["dev_smooth_correct"].to(torch.float64)
    b300 = b_steps["300"]["dev_smooth_correct"].to(torch.float64)
    delta_nl = b300 - a300
    ci = V10.V6._paired_bootstrap(delta_nl, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    b_transfer = bool(b_report["cyclic_transfer"]["template"]["pass"] and b_report["cyclic_transfer"]["smooth"]["pass"] and b_report["cyclic_transfer"]["broad"] and not b_report["cyclic_transfer"]["pair_dominated"])
    a_transfer = bool(a_report["cyclic_transfer"]["template"]["pass"] and a_report["cyclic_transfer"]["smooth"]["pass"] and a_report["cyclic_transfer"]["broad"] and not a_report["cyclic_transfer"]["pair_dominated"])
    raw = bool(a_report["raw_validity_pass"] and b_report["raw_validity_pass"])
    reconstruction = float(b_report["reconstruction"]["Delta_L_image"])
    supported = bool(float(delta_nl.mean()) >= MATERIAL_DELTA and float(ci["ci_lower"]) > 0.0 and reconstruction <= RECONSTRUCTION_MAX_DELTA and a_transfer and b_transfer and raw)
    classification = OUTER_GATE if supported else "N6_NONLINEAR_FIBER_FACET_ACTIVATION_NOT_SUPPORTED"
    stage_a = {}
    for name, report in (("A_LINEAR_CONTINUATION_CONTROL", a_report), ("B_NONLINEAR_FACET", b_report)):
        value = float(report["evaluations"]["300"]["smooth"]["macro_recall"])
        stage_a[name] = {"R300": value, "threshold": STAGE_A_FIREWALL, "status": "STAGE_A_FIREWALL_PASSED" if value >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False}
    contract_ref = {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256}
    pairwise = {"Delta_NL_R_B300_minus_A300": {"observed_delta": float(delta_nl.mean()), "paired_bootstrap": {key: value for key, value in ci.items() if key != "delta_records"}}, "R_A300_minus_N5A": {"observed_delta": float(a_report["evaluations"]["300"]["smooth"]["macro_recall"] - 0.5526)}, "R_B300_minus_N5A": {"observed_delta": float(b_report["evaluations"]["300"]["smooth"]["macro_recall"] - 0.5526)}}
    payload = {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "VALID_COMPLETED", "status": "valid_completed", "classification": classification, "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "contract": contract_ref, "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"]}, "n5_closure": {"terminal": True, "classification": "OPTIMIZATION_HORIZON_DOMINANT_TRUE_DEPTH_NOT_SUPPORTED", "best_linear_endpoint": source["checkpoint_ref"], "no_more_linear_optical_depth": True}, "candidates": {"A_LINEAR_CONTINUATION_CONTROL": {"result": a_result, "Smooth_R300": float(a_report["evaluations"]["300"]["smooth"]["macro_recall"]), "DEV_L_image_step300": float(a_report["reconstruction"]["DEV_L_image_step300"]), "report": a_result["report"], "checkpoint": a_report["checkpoints"]["300"]}, "B_NONLINEAR_FACET": {"result": b_result, "Smooth_R300": float(b_report["evaluations"]["300"]["smooth"]["macro_recall"]), "DEV_L_image_step300": float(b_report["reconstruction"]["DEV_L_image_step300"]), "report": b_result["report"], "checkpoint": b_report["checkpoints"]["300"]}}, "comparison": {"protocol": "symmetric N5-A continuation; fixed two-layer generator; only B has the fixed facet surrogate", "pairwise": pairwise, "primary_gate": {"name": OUTER_GATE, "supported": supported, "threshold_delta": MATERIAL_DELTA, "ci_lower_threshold": 0.0, "reconstruction_max_delta": RECONSTRUCTION_MAX_DELTA, "observed_reconstruction_delta_B": reconstruction, "cyclic_transfer_A": a_transfer, "cyclic_transfer_B": b_transfer, "raw_validity": raw}}, "reconstruction": {"A": a_report["reconstruction"], "B": b_report["reconstruction"]}, "cyclic_transfer": {"A": a_report["cyclic_transfer"], "B": b_report["cyclic_transfer"]}, "raw_diversity": {"A": a_report["raw_diversity"], "B": b_report["raw_diversity"]}, "facet_diagnostics": {"A": a_report["facet_diagnostics"], "B": b_report["facet_diagnostics"]}, "stage_a": stage_a, "claim_boundary": {"training_updates": 600, "optimizer_updates": 600, "new_optical_depth": False, "rank": ADAPTER_RANK, "encoder_update": False, "material_trainable": False, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {"rank": ADAPTER_RANK, "new_optical_depth": False, "adapter_update": True, "generator_phase_update": True, "generator_depth_change": False, "material_trainable": False, "retry": False, "extra_candidate": False, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "promotion": False}, "original_n5_v2": "CLOSED_AS_TERMINAL_LINEAR_BRANCH", "next": "N6 result does not authorize Stage B or generation"}
    report_path = output_dir / AGGREGATE_REPORT_NAME
    N4.atomic_write_new_json(report_path, payload, artifact_kind="N6 aggregate report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = output_dir / AGGREGATE_LINEAGE_NAME
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "report": report_ref, "claim_boundary": payload["claim_boundary"], "n5_closure": payload["n5_closure"]}, artifact_kind="N6 aggregate lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": classification, "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "pairwise": pairwise, "primary_gate": payload["comparison"]["primary_gate"], "stage_a": stage_a, "n5_closure": payload["n5_closure"]}


def run_scientific(root: Path) -> dict[str, Any]:
    preflight = _load_preflight(root)
    _require(preflight["report"].get("result") == "AUTHORIZABLE", "N6 preflight is not authorizable")
    a_result = run_candidate(root, "A_LINEAR_CONTINUATION_CONTROL", nonlinear=False)
    if a_result["result"] != "VALID_COMPLETED":
        return {"result": a_result["result"], "status": a_result["status"], "classification": a_result["classification"], "candidate_A": a_result, "candidate_B": "not executed because symmetric comparison could not be completed", "outer_gate": OUTER_GATE}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    b_result = run_candidate(root, "B_NONLINEAR_FACET", nonlinear=True)
    if b_result["result"] != "VALID_COMPLETED":
        return {"result": b_result["result"], "status": b_result["status"], "classification": b_result["classification"], "candidate_A": a_result, "candidate_B": b_result, "outer_gate": OUTER_GATE}
    a_report = _read_json(_path(root, a_result["report"]["path"]), "N6 candidate A report")
    b_report = _read_json(_path(root, b_result["report"]["path"]), "N6 candidate B report")
    a_steps = {str(step): _load_step_sufficient(root, a_report, "A_LINEAR_CONTINUATION_CONTROL", step) for step in CHECKPOINT_STEPS}
    b_steps = {str(step): _load_step_sufficient(root, b_report, "B_NONLINEAR_FACET", step) for step in CHECKPOINT_STEPS}
    source = _load_n5_a(root)
    contract = _contract(root)
    return _aggregate(root, contract, preflight, a_result, b_result, a_report, b_report, a_steps, b_steps, source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific", "all"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    if args.action == "preflight":
        result = run_preflight(root)
    elif args.action == "scientific":
        result = run_scientific(root)
    else:
        result = run_preflight(root)
        if result["result"] != "AUTHORIZABLE":
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 2
        result = run_scientific(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("result") in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
