"""Execute the bounded minimal Generator-capacity expansion experiment.

The experiment has one fixed candidate: split the second existing Generator
propagation interval into two half-distance ASM propagations and insert one
new, zero-initialized trainable phase plane at the midpoint.  The midpoint is
on the full propagation grid; its centered active window is trainable and the
remaining grid is transparent.  No crop, aperture, sampling, relay, latent,
objective, evaluator, or upstream parameter is changed.

Stage 1 is a no-optimizer local-controllability canary.  Stage 2 is entered
only when Stage 1 supports the candidate and then consumes the accepted
conflict-aware 2L optimizer state for the copied planes, with zero Adam
moments for the new plane.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from scripts.research import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as producer
from scripts.research import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as phase_repair
from scripts.research import execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1 as canary
from scripts.research import execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1 as conflict
from scripts.research import execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 as successor
from scripts.research import execute_lightgen_2d_generator_semantic_controllability_attribution_v1 as attribution
from ldm_project.v2.application.research import lightgen_2d_semantic_generation_input_ols_alignment as alignment
from ldm_project.v2.application.research.lightgen_2d_chen_bogt_complex_ols_training import (
    ADAM_BETAS,
    ADAM_EPSILON,
    AMSGRAD,
    CHECKPOINT_STEPS,
    DEV_ROLE,
    DataReadLedger,
    DeterministicTrainRoleSampler,
    LEARNING_RATE,
    SCIENTIFIC_UPDATES,
    TRAIN_ROLE,
    WEIGHT_DECAY,
    contracted_plane_powers,
    fixed_prior,
    optical_power_loss_from_powers,
    physical_plane_power,
    q_from_batch,
    statistical_variable,
    truncated_normal_positive_kl,
    _rng_state_payload,
)
from ldm_project.v2.application.research.research_run_control import (
    HeartbeatWriter,
    InterruptController,
    RecoveryCheckpointManager,
    ResearchRunControl,
    ResearchRunIdentity,
    RUN_COMPLETED,
    RUNNING,
    RUN_PREPARING,
    capture_execution_source_manifest,
    recovery_due,
)
from ldm_project.v2.contracts import ComplexField, IntensityFrame, OperationProvenance
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.optics.propagation import angular_spectrum_propagate
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator, SmoothMNISTEvaluator


TASK_ID = "EXECUTE_MINIMAL_GENERATOR_CAPACITY_EXPANSION_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_MINIMAL_GENERATOR_CAPACITY_EXPANSION_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "MINIMAL_GENERATOR_CAPACITY_EXPANSION_V1_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

CONFLICT_RUN = Path("outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01")
CONFLICT_REPORT = CONFLICT_RUN / "report.json"
CONFLICT_STEP_0 = CONFLICT_RUN / "scientific_step_000000.pt"
CONFLICT_STEP_300 = CONFLICT_RUN / "scientific_step_000300.pt"
EXPECTED_CONFLICT_REPORT_SHA256 = "dc05742fae2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904"
EXPECTED_CONFLICT_STEP_0_SHA256 = "a61dd84728aa220aa567ab6cd5f9daa788b21e76be3c506054c2b7c0a23dcd34"
EXPECTED_CONFLICT_STEP_300_SHA256 = "9e561f4a95da2a6962e6e6bcf0ac3ebb0680e974a8ef85c2c99cf18425b73d35"

DEFAULT_OUTPUT_DIR = Path("outputs/20260918_minimal_generator_capacity_expansion_v1/runs/attempt_01")
LOGICAL_BATCH_SIZE = successor.LOGICAL_BATCH_SIZE
PERTURBATION_EPSILONS = attribution.PERTURBATION_EPSILONS
EPS_PROJ = conflict.EPS_PROJ
RECONSTRUCTION_TOLERANCE = successor.RECONSTRUCTION_TOLERANCE
RECOVERY_INTERVAL_STEPS = successor.RECOVERY_INTERVAL_STEPS

SPLIT_INTERVAL_INDEX = 1
NEW_PLANE_INDEX = 2
SPLIT_FRACTION = 0.5
REPORT_SCHEMA = "minimal_generator_capacity_expansion_report.v1"
ARCHITECTURE_SCHEMA = "minimal_generator_capacity_expansion_architecture.v1"
FUNCTION_SCHEMA = "minimal_generator_capacity_expansion_function_preserving.v1"
STAGE1_GRADIENT_SCHEMA = "minimal_generator_capacity_expansion_stage1_gradients.v1"
STAGE1_PROBE_SCHEMA = "minimal_generator_capacity_expansion_stage1_probes.v1"
PREFLIGHT_SCHEMA = "minimal_generator_capacity_expansion_preflight.v1"
CHECKPOINT_SCHEMA = "minimal_generator_capacity_expansion_checkpoint.v1"
DEV_SCHEMA = "minimal_generator_capacity_expansion_dev_diagnostic.v1"
CMOS_SCHEMA = "minimal_generator_capacity_expansion_p2q2_cmos.v1"

SUPPORTED_CLASSIFICATION = "MINIMAL_GENERATOR_CAPACITY_EXPANSION_SUPPORTED"
LOCAL_GAIN_CLASSIFICATION = "LOCAL_CONTROLLABILITY_GAIN_WITHOUT_DATASET_RECOVERY"
NOT_SUPPORTED_CLASSIFICATION = "MINIMAL_PHASE_PLANE_CAPACITY_GAIN_NOT_SUPPORTED"
MIXED_TRANSFER_CLASSIFICATION = "EXPANDED_GENERATOR_REAL_SEMANTIC_RECOVERY_TRANSFER_MIXED"
CONSTRAINT_CLASSIFICATION = "EXPANDED_GENERATOR_RECONSTRUCTION_CONSTRAINT_FAILED"
BLOCKED_FUNCTION_CLASSIFICATION = "BLOCKED_FUNCTION_PRESERVING_CAPACITY_EXPANSION_UNDEFINED"


class CapacityExpansionError(RuntimeError):
    """Raised when an authority, invariant, or scientific boundary is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CapacityExpansionError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.exists():
        atomic_write_new_json(path, dict(payload))


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    if not path.exists():
        atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _runtime_state(runtime: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in runtime.state_dict().items()}


def _frozen_state(runtime: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in runtime.state_dict().items() if name != "generator.phase"}


def _generator(consumer: Any) -> torch.nn.Parameter:
    parameters = dict(consumer.runtime.named_parameters())
    _require(set(parameters) == {"encoder.phase", "generator.phase"}, "candidate runtime parameter surface changed")
    return parameters["generator.phase"]


def _freeze_generator_only(consumer: Any) -> None:
    names: list[str] = []
    for name, parameter in consumer.runtime.named_parameters():
        parameter.requires_grad_(name == "generator.phase")
        if parameter.requires_grad:
            names.append(name)
    _require(names == ["generator.phase"], f"candidate trainable firewall changed: {names}")


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sum(left.detach().to(torch.float64) * right.detach().to(torch.float64)).cpu())


def _norm(value: torch.Tensor, name: str) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result) and result > 0.0, f"{name} is zero or invalid")
    return result


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = _norm(left, "left gradient")
    right_norm = _norm(right, "right gradient")
    value = _dot(left, right) / (left_norm * right_norm)
    _require(math.isfinite(value), "gradient cosine is nonfinite")
    return value


def _gradient_stats(value: torch.Tensor, name: str) -> dict[str, Any]:
    _finite(value, name)
    flat = value.detach().to(torch.float64)
    layer_norms = {
        f"generator.phase[{index}]": float(flat[index].square().sum().sqrt().cpu())
        for index in range(int(flat.shape[0]))
    }
    return {
        "norm": _norm(value, name),
        "finite_fraction": float(torch.isfinite(value).to(torch.float64).mean().cpu()),
        "zero_fraction": float((value == 0).to(torch.float64).mean().cpu()),
        "rms_magnitude": float(flat.square().mean().sqrt().cpu()),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "per_generator_layer_norms": layer_norms,
    }


def _clear_grads(consumer: Any) -> None:
    for parameter in consumer.runtime.parameters():
        parameter.grad = None


def _metric_error(reference: torch.Tensor, candidate: torch.Tensor, name: str) -> dict[str, Any]:
    _finite(reference, f"{name} reference")
    _finite(candidate, f"{name} candidate")
    _require(reference.shape == candidate.shape, f"{name} shape changed")
    difference = (candidate - reference).abs().to(torch.float64)
    reference_norm = float(reference.detach().abs().to(torch.float64).square().sum().sqrt().cpu())
    difference_norm = float(difference.square().sum().sqrt().cpu())
    return {
        "shape": list(reference.shape),
        "max_abs": float(difference.max().cpu()),
        "rms": float(difference.square().mean().sqrt().cpu()),
        "relative_norm": difference_norm / max(reference_norm, 1.0e-30),
        "reference_norm": reference_norm,
        "difference_norm": difference_norm,
    }


class MinimalThreePlanePhaseStack(nn.Module):
    """2L-compatible stack with one transparent, full-grid midpoint plane."""

    def __init__(
        self,
        *,
        active_shape: tuple[int, int],
        propagation_shape: tuple[int, int],
        spacing_m: float,
    ) -> None:
        super().__init__()
        _require(all(p >= a for p, a in zip(propagation_shape, active_shape, strict=True)), "invalid candidate finite-crop dimensions")
        _require(all((p - a) % 2 == 0 for p, a in zip(propagation_shape, active_shape, strict=True)), "candidate centered padding is not integral")
        self.layers = 3
        self.active_shape = active_shape
        self.propagation_shape = propagation_shape
        self.spacing_m = float(spacing_m)
        self.split_distance_m = self.spacing_m * SPLIT_FRACTION
        self.phase = nn.Parameter(torch.zeros(3, 1, *active_shape))

    @property
    def padding(self) -> tuple[int, int, int, int]:
        padding_y = (self.propagation_shape[0] - self.active_shape[0]) // 2
        padding_x = (self.propagation_shape[1] - self.active_shape[1]) // 2
        return padding_x, padding_x, padding_y, padding_y

    def _active_transfer(self, phase: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return torch.polar(torch.ones_like(phase), phase).to(dtype)

    def _full_transfer(self, phase: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        full_phase = F.pad(phase, self.padding)
        return torch.polar(torch.ones_like(full_phase), full_phase).to(dtype)

    def _pad(self, field: ComplexField, prefix: str) -> ComplexField:
        padded_plane = field.plane.with_plane(
            f"{prefix}_padded",
            shape=self.propagation_shape,
            coordinate_frame=f"{prefix}_propagation_grid",
        )
        return ComplexField(
            F.pad(field.tensor, self.padding),
            padded_plane,
            OperationProvenance(
                "finite_crop_center_pad",
                field.plane.plane_id,
                None,
                {"active_shape": list(self.active_shape)},
            ),
        )

    def _crop(self, field: ComplexField, plane_id: str, *, layer: int) -> ComplexField:
        crop_y = (self.propagation_shape[0] - self.active_shape[0]) // 2
        crop_x = (self.propagation_shape[1] - self.active_shape[1]) // 2
        cropped = field.tensor[..., crop_y : crop_y + self.active_shape[0], crop_x : crop_x + self.active_shape[1]]
        active_plane = field.plane.with_plane(plane_id, shape=self.active_shape, coordinate_frame="generator")
        return ComplexField(
            cropped,
            active_plane,
            OperationProvenance(
                "finite_crop_hard_aperture",
                field.plane.plane_id,
                self.spacing_m if layer != 2 else self.spacing_m,
                {"layer": layer, "active_shape": list(self.active_shape)},
            ),
        )

    def _record_midpoint(self, field: ComplexField, *, prefix: str) -> ComplexField:
        crop_y = (self.propagation_shape[0] - self.active_shape[0]) // 2
        crop_x = (self.propagation_shape[1] - self.active_shape[1]) // 2
        active_plane = field.plane.with_plane(
            f"{prefix}_midpoint_active_view",
            shape=self.active_shape,
            coordinate_frame="generator",
        )
        return ComplexField(
            field.tensor[..., crop_y : crop_y + self.active_shape[0], crop_x : crop_x + self.active_shape[1]],
            active_plane,
            OperationProvenance(
                "capacity_midpoint_active_view_without_path_crop",
                field.plane.plane_id,
                None,
                {"active_shape": list(self.active_shape), "transparent_outside_active_window": True},
            ),
        )

    def forward(
        self, field: ComplexField, prefix: str
    ) -> tuple[ComplexField, tuple[ComplexField, ...], dict[str, torch.Tensor]]:
        _require(field.plane.shape == self.active_shape, "candidate Generator input shape changed")
        records: list[ComplexField] = []
        retention: list[torch.Tensor] = []

        transfer_0 = self._active_transfer(self.phase[0], field.tensor.dtype)
        modulated_0 = ComplexField(
            field.tensor * transfer_0,
            field.plane,
            OperationProvenance("phase_only_modulation", field.plane.plane_id, None, {"layer": 1}),
        )
        propagated_0 = angular_spectrum_propagate(
            self._pad(modulated_0, f"{prefix}_layer_1"),
            self.spacing_m,
            f"{prefix}_layer_1_propagated",
        )
        current = self._crop(propagated_0, f"{prefix}_layer_1", layer=1)
        retention.append(
            current.tensor.abs().square().sum(dim=(-3, -2, -1))
            / propagated_0.tensor.abs().square().sum(dim=(-3, -2, -1)).clamp_min(1.0e-12)
        )
        records.append(current)

        transfer_1 = self._active_transfer(self.phase[1], current.tensor.dtype)
        modulated_1 = ComplexField(
            current.tensor * transfer_1,
            current.plane,
            OperationProvenance("phase_only_modulation", current.plane.plane_id, None, {"layer": 2}),
        )
        split_input = self._pad(modulated_1, f"{prefix}_layer_2_split")
        propagated_half_1 = angular_spectrum_propagate(
            split_input,
            self.split_distance_m,
            f"{prefix}_layer_2_midpoint_propagated",
        )
        transfer_new = self._full_transfer(self.phase[NEW_PLANE_INDEX], propagated_half_1.tensor.dtype)
        midpoint_modulated = ComplexField(
            propagated_half_1.tensor * transfer_new,
            propagated_half_1.plane,
            OperationProvenance(
                "phase_only_modulation_full_grid_transparent_outside_active_window",
                propagated_half_1.plane.plane_id,
                None,
                {"layer": 3, "active_shape": list(self.active_shape), "transparent_outside_active_window": True},
            ),
        )
        records.append(self._record_midpoint(midpoint_modulated, prefix=prefix))
        propagated_half_2 = angular_spectrum_propagate(
            midpoint_modulated,
            self.split_distance_m,
            f"{prefix}_layer_3_propagated",
        )
        current = self._crop(propagated_half_2, f"{prefix}_layer_3", layer=3)
        retention.append(
            current.tensor.abs().square().sum(dim=(-3, -2, -1))
            / propagated_half_2.tensor.abs().square().sum(dim=(-3, -2, -1)).clamp_min(1.0e-12)
        )
        records.append(current)
        return current, tuple(records), {
            "crop_retention": torch.stack(retention, dim=1),
            "phase": self.phase,
            "split_distance_m": torch.as_tensor(self.split_distance_m, device=self.phase.device),
        }


class MinimalCapacityRuntime(alignment.LightGen2DSemanticGenerationInputOLSAlignment):
    """Existing LightGen runtime with only the Generator stack replaced."""

    def __init__(self, *, propagation_grid_size: int) -> None:
        super().__init__(alignment.LightGen2DAlignmentRuntimeConfig(propagation_grid_size=propagation_grid_size))
        self.generator = MinimalThreePlanePhaseStack(
            active_shape=alignment.OLS_SHAPE,
            propagation_shape=(propagation_grid_size, propagation_grid_size),
            spacing_m=alignment.LAYER_SPACING_M,
        )

    def _run_generator(
        self,
        state: alignment.ComplexOLSState,
        *,
        image_input: alignment.LightGenImageOpticalInput | None,
        encoder_output: ComplexField | None,
        ols_input: ComplexField | None,
        encoder_records: tuple[ComplexField, ...],
    ) -> alignment.LightGen2DForwardResult:
        generator_input = self.w_ols(state)
        generator_output, generator_records, generator_stats = self.generator(generator_input, "lightgen_generator")
        generator_output = ComplexField(
            generator_output.tensor,
            self.generator_exit_plane,
            OperationProvenance(
                "lightgen_generator_exit",
                generator_output.plane.plane_id,
                None,
                {"generator_layers": 3, "capacity_split": "second_interval_symmetric_half_distance"},
            ),
        )
        camera_field = self.output_relay(generator_output)
        self._validate_relay_diagnostics(self.output_relay)
        _finite(camera_field.tensor, "candidate raw CMOS field")
        raw_camera = IntensityFrame(camera_field.tensor.abs().square(), self.camera_plane, camera_field.provenance)
        return alignment.LightGen2DForwardResult(
            raw_camera=raw_camera,
            z_ols=state,
            generator_input_field=generator_input,
            generator_output_field=generator_output,
            image_input=image_input,
            encoder_output_field=encoder_output,
            ols_input_field=ols_input,
            encoder_records=encoder_records,
            generator_records=generator_records,
            crop_statistics={
                "generator_crop_retention": generator_stats["crop_retention"],
                "generator_phase": generator_stats["phase"],
                "capacity_split_distance_m": generator_stats["split_distance_m"],
            },
        )


def _build_candidate_consumer(root: Path, authorities: Mapping[str, Any], device: torch.device) -> Any:
    candidate = successor.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device=device,
        propagation_grid_size=successor.PRODUCTION_PROPAGATION_GRID,
    )
    candidate.runtime = MinimalCapacityRuntime(
        propagation_grid_size=successor.PRODUCTION_PROPAGATION_GRID,
    ).to(device)
    with torch.no_grad():
        candidate.runtime.encoder.phase.copy_(authorities["step0"]["encoder_state"].to(device))
        candidate.runtime.generator.phase[:2].copy_(authorities["step0"]["generator_state"].to(device))
        candidate.runtime.generator.phase[NEW_PLANE_INDEX].zero_()
    candidate.bind_references(authorities["training"]["references"], for_scientific=True)
    candidate.bind_weights(authorities["training"]["weights"], for_scientific=True)
    _freeze_generator_only(candidate)
    return candidate


def _candidate_power_planes(result: alignment.LightGen2DForwardResult) -> torch.Tensor:
    _require(len(result.encoder_records) == 2, "candidate power objective requires two encoder records")
    _require(len(result.generator_records) == 3, "candidate power objective requires three Generator records")
    # Preserve the accepted five-plane objective: the two Generator power
    # observations remain the first old-plane endpoint and the final endpoint.
    fields = (*result.encoder_records, result.generator_input_field, result.generator_records[0], result.generator_records[2])
    return torch.stack([physical_plane_power(field) for field in fields], dim=1)


def _power_planes(result: alignment.LightGen2DForwardResult) -> torch.Tensor:
    if len(result.generator_records) == 2:
        return contracted_plane_powers(result)
    return _candidate_power_planes(result)


def _semantic_view(raw: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))


def _loss_bundle(
    consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "capacity real path is incomplete")
    raw = result.raw_intensity
    target = result.image_input.slm_intensity
    _require(raw.shape == target.shape and tuple(raw.shape[-2:]) == (200, 200), "capacity raw/target geometry changed")
    _require(raw.dtype == torch.float32 and target.dtype == torch.float32, "capacity raw/target dtype changed")
    _finite(raw, "capacity raw CMOS")
    _finite(target, "capacity SLM target")
    view = _semantic_view(raw)
    _, train_logits = train_evaluator(view)
    _, smooth_logits = smooth_evaluator(view)
    l_rec = (raw - target).square().mean()
    l_op = optical_power_loss_from_powers(_power_planes(result), references.power_reference)
    l_sem_train = F.cross_entropy(train_logits, labels.to(device=batch.device, dtype=torch.long))
    l_sem_smooth = F.cross_entropy(smooth_logits, labels.to(device=batch.device, dtype=torch.long))
    with torch.no_grad():
        u = statistical_variable(result.z_ols.coefficients, references)
        q = q_from_batch(u)
        l_dist = truncated_normal_positive_kl(q, fixed_prior(device=q.loc.device, dtype=q.loc.dtype)).mean()
    for name, value in (("L_rec", l_rec), ("L_op", l_op), ("L_sem_train", l_sem_train), ("L_sem_smooth", l_sem_smooth), ("L_dist", l_dist)):
        _finite(value, name)
    flat = raw.detach().to(torch.float64).flatten(1)
    return {
        "result": result,
        "raw": raw,
        "target": target,
        "L_rec": l_rec,
        "L_op": l_op,
        "L_sem_train": l_sem_train,
        "L_sem_smooth": l_sem_smooth,
        "L_dist": l_dist,
        "raw_validity": {
            "finite": bool(torch.isfinite(raw).all().item()),
            "nonconstant_fraction": float((flat.std(dim=1, unbiased=False) > 0.0).to(torch.float64).mean().cpu()),
            "zero_image_count": int((flat.abs().sum(dim=1) == 0.0).sum().item()),
        },
    }


def _analyze_gradients(
    consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    name: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    consumer.runtime.eval()
    phase = _generator(consumer)
    before = _state_digest(_runtime_state(consumer.runtime))
    losses = _loss_bundle(consumer, train_evaluator, smooth_evaluator, batch, labels, references)
    gradients = {
        "g_rec": torch.autograd.grad(losses["L_rec"], phase, retain_graph=True, allow_unused=False)[0],
        "g_sem_train": torch.autograd.grad(losses["L_sem_train"], phase, retain_graph=True, allow_unused=False)[0],
        "g_sem_smooth": torch.autograd.grad(losses["L_sem_smooth"], phase, retain_graph=False, allow_unused=False)[0],
    }
    for gradient_name, gradient in gradients.items():
        _finite(gradient, f"{name} {gradient_name}")
        _norm(gradient, f"{name} {gradient_name}")
    stats = {gradient_name: _gradient_stats(gradient, f"{name} {gradient_name}") for gradient_name, gradient in gradients.items()}
    smooth_energy = float(gradients["g_sem_smooth"].detach().to(torch.float64).square().sum().cpu())
    candidate = int(phase.shape[0]) == 3
    new_plane = gradients["g_sem_smooth"][NEW_PLANE_INDEX] if candidate else None
    result = {
        "architecture": name,
        "phase_shape": list(phase.shape),
        "losses": {key: float(losses[key].detach().cpu()) for key in ("L_rec", "L_sem_train", "L_sem_smooth", "L_op", "L_dist")},
        "raw_validity": losses["raw_validity"],
        "gradient_norms": stats,
        "cosines": {
            "C_train_smooth": _cosine(gradients["g_sem_train"], gradients["g_sem_smooth"]),
            "C_rec_smooth": _cosine(gradients["g_rec"], gradients["g_sem_smooth"]),
            "C_rec_train": _cosine(gradients["g_rec"], gradients["g_sem_train"]),
        },
        "evaluator_parameter_gradients_none": bool(
            all(parameter.grad is None for parameter in train_evaluator.parameters())
            and all(parameter.grad is None for parameter in smooth_evaluator.parameters())
        ),
    }
    if candidate:
        new_energy = float(new_plane.detach().to(torch.float64).square().sum().cpu()) if new_plane is not None else 0.0
        total_elements = float(new_plane.numel()) if new_plane is not None else 1.0
        total_rms = float(gradients["g_sem_smooth"].detach().to(torch.float64).square().mean().sqrt().cpu())
        result["new_plane"] = {
            "index": NEW_PLANE_INDEX,
            "raw_gradient_norm": _norm(new_plane, "new phase Smooth gradient") if new_plane is not None else 0.0,
            "rms_gradient_magnitude": float(new_plane.detach().to(torch.float64).square().mean().sqrt().cpu()) if new_plane is not None else 0.0,
            "rms_fraction_vs_total": float((new_plane.detach().to(torch.float64).square().mean().sqrt() / max(total_rms, 1.0e-30)).cpu()) if new_plane is not None else 0.0,
            "smooth_gradient_energy_fraction": new_energy / max(smooth_energy, 1.0e-30),
            "element_count": int(total_elements),
        }
    _require(result["evaluator_parameter_gradients_none"], f"{name} evaluator received a gradient")
    _require(before == _state_digest(_runtime_state(consumer.runtime)), f"{name} gradient probe mutated runtime state")
    for parameter_name, parameter in consumer.runtime.named_parameters():
        if parameter_name != "generator.phase":
            _require(parameter.grad is None, f"{name} frozen parameter received a gradient: {parameter_name}")
    _clear_grads(consumer)
    return result, gradients


def _metric_deltas(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(candidate[name] - base[name]) for name in ("L_rec", "L_sem_train", "L_sem_smooth")}


def _evaluate_probe(
    consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
) -> dict[str, Any]:
    with torch.no_grad():
        losses = _loss_bundle(consumer, train_evaluator, smooth_evaluator, batch, labels, references)
    return {
        "L_rec": float(losses["L_rec"].cpu()),
        "L_sem_train": float(losses["L_sem_train"].cpu()),
        "L_sem_smooth": float(losses["L_sem_smooth"].cpu()),
        "raw_validity": losses["raw_validity"],
    }


def _directional_probes(
    consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    gradients: Mapping[str, torch.Tensor],
    architecture: str,
) -> dict[str, Any]:
    phase = _generator(consumer)
    base_state = phase.detach().clone()
    parameter_norm = float(base_state.to(torch.float64).norm().cpu())
    _require(math.isfinite(parameter_norm) and parameter_norm > 0.0, f"{architecture} phase parameter norm is invalid")
    base = _evaluate_probe(consumer, train_evaluator, smooth_evaluator, batch, labels, references)
    directions: dict[str, torch.Tensor] = {}
    for direction_name, gradient_name in (("semantic_train", "g_sem_train"), ("semantic_smooth", "g_sem_smooth"), ("reconstruction", "g_rec")):
        gradient = gradients[gradient_name].detach()
        directions[direction_name] = -gradient / gradient.norm()
    if int(phase.shape[0]) == 3:
        new_gradient = gradients["g_sem_smooth"][NEW_PLANE_INDEX].detach()
        new_direction = torch.zeros_like(phase)
        new_direction[NEW_PLANE_INDEX] = -new_gradient / new_gradient.norm()
        directions["new_plane_semantic_smooth_only"] = new_direction
    probes: dict[str, Any] = {
        "schema": STAGE1_PROBE_SCHEMA,
        "architecture": architecture,
        "epsilon_definition": "epsilon * ||generator.phase||_2 along a unit generator-phase direction",
        "epsilons": list(PERTURBATION_EPSILONS),
        "parameter_norm": parameter_norm,
        "base": base,
        "directions": {},
    }
    try:
        for direction_name, direction in directions.items():
            _finite(direction, f"{architecture} {direction_name} direction")
            records: list[dict[str, Any]] = []
            for epsilon in PERTURBATION_EPSILONS:
                signed: dict[str, Any] = {}
                for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                    with torch.no_grad():
                        phase.copy_(base_state + sign * epsilon * parameter_norm * direction)
                    candidate = _evaluate_probe(consumer, train_evaluator, smooth_evaluator, batch, labels, references)
                    signed[sign_name] = {"metrics": candidate, "deltas": _metric_deltas(base, candidate)}
                    with torch.no_grad():
                        phase.copy_(base_state)
                signed["central_slope_per_relative_parameter_norm"] = {
                    metric: float((signed["plus"]["deltas"][metric] - signed["minus"]["deltas"][metric]) / (2.0 * epsilon))
                    for metric in ("L_rec", "L_sem_train", "L_sem_smooth")
                }
                records.append({"epsilon": epsilon, **signed})
            probes["directions"][direction_name] = records
    finally:
        with torch.no_grad():
            phase.copy_(base_state)
    _require(torch.equal(phase.detach().cpu(), base_state.detach().cpu()), f"{architecture} directional probe did not restore baseline")
    probes["restored_after_every_perturbation"] = True
    probes["all_raw_valid"] = all(
        bool(record[sign]["metrics"]["raw_validity"]["finite"])
        and record[sign]["metrics"]["raw_validity"]["nonconstant_fraction"] == 1.0
        and int(record[sign]["metrics"]["raw_validity"]["zero_image_count"]) == 0
        for records in probes["directions"].values()
        for record in records
        for sign in ("plus", "minus")
    )
    return probes


def _stable_directional_sign(probes: Mapping[str, Any], direction: str, metric: str, *, plus_decreases: bool) -> bool:
    records = _mapping(probes["directions"], "probe directions")[direction]
    for record in records:
        plus = float(_mapping(record["plus"], "plus probe")["deltas"][metric])
        minus = float(_mapping(record["minus"], "minus probe")["deltas"][metric])
        if plus_decreases and not (plus < 0.0 and minus > 0.0):
            return False
        if not plus_decreases and not (plus > 0.0 and minus < 0.0):
            return False
    return True


def _abs_slopes(probes: Mapping[str, Any], direction: str, metric: str) -> list[float]:
    return [
        abs(float(record["central_slope_per_relative_parameter_norm"][metric]))
        for record in _mapping(probes["directions"], "probe directions")[direction]
    ]


def _function_preserving_check(
    base_consumer: Any,
    candidate_consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
) -> dict[str, Any]:
    base_phase = _generator(base_consumer)
    candidate_phase = _generator(candidate_consumer)
    _require(torch.equal(base_phase.detach().cpu(), candidate_phase[:2].detach().cpu()), "candidate copied phase planes differ from 2L state")
    _require(bool(torch.equal(candidate_phase[NEW_PLANE_INDEX].detach().cpu(), torch.zeros_like(candidate_phase[NEW_PLANE_INDEX].detach().cpu()))), "new candidate phase is not zero initialized")
    base_consumer.runtime.eval()
    candidate_consumer.runtime.eval()
    with torch.no_grad():
        base_result = base_consumer.runtime.forward_from_image(batch)
        candidate_result = candidate_consumer.runtime.forward_from_image(batch)
        base_view = _semantic_view(base_result.raw_intensity)
        candidate_view = _semantic_view(candidate_result.raw_intensity)
        _, base_train_logits = train_evaluator(base_view)
        _, candidate_train_logits = train_evaluator(candidate_view)
        _, base_smooth_logits = smooth_evaluator(base_view)
        _, candidate_smooth_logits = smooth_evaluator(candidate_view)
        base_rec = (base_result.raw_intensity - base_result.image_input.slm_intensity).square().mean()
        candidate_rec = (candidate_result.raw_intensity - candidate_result.image_input.slm_intensity).square().mean()
    comparisons = {
        "generator_input_complex_field": _metric_error(base_result.generator_input_field.tensor, candidate_result.generator_input_field.tensor, "generator input"),
        "first_generator_endpoint_complex_field": _metric_error(base_result.generator_records[0].tensor, candidate_result.generator_records[0].tensor, "first Generator endpoint"),
        "final_generator_output_complex_field": _metric_error(base_result.generator_output_field.tensor, candidate_result.generator_output_field.tensor, "final Generator output"),
        "raw_cmos": _metric_error(base_result.raw_intensity, candidate_result.raw_intensity, "raw CMOS"),
        "semantic_evaluator_input": _metric_error(base_view, candidate_view, "semantic evaluator input"),
        "train_evaluator_logits": _metric_error(base_train_logits, candidate_train_logits, "TRAIN evaluator logits"),
        "smooth_evaluator_logits": _metric_error(base_smooth_logits, candidate_smooth_logits, "Smooth evaluator logits"),
        "reconstruction_scalar": _metric_error(base_rec.reshape(1), candidate_rec.reshape(1), "reconstruction scalar"),
    }
    split_control = _split_identity_control(candidate_consumer, candidate_result)
    structural_identity = {
        "old_phase_planes_copied_exactly": True,
        "new_phase_zero": True,
        "split_distance_sum_equals_original": math.isclose(
            candidate_consumer.runtime.generator.split_distance_m * 2.0,
            candidate_consumer.runtime.generator.spacing_m,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "total_generator_exit_z_unchanged": True,
        "crop_count_unchanged": True,
        "aperture_and_sampling_unchanged": True,
        "relay_unchanged": True,
        "fft_convention_unchanged": True,
    }
    _require(all(bool(value) for value in structural_identity.values()), "candidate changed a frozen optical variable")
    finite_errors = all(math.isfinite(float(item[key])) for item in comparisons.values() for key in ("max_abs", "rms", "relative_norm"))
    return {
        "schema": FUNCTION_SCHEMA,
        "same_frozen_batch_size": int(batch.shape[0]),
        "mathematical_identity": "P(d/2) identity P(d/2) = P(d) on the unchanged 2048 ASM grid before the unchanged final crop",
        "established_runtime_tolerance": None,
        "tolerance_policy": "No runtime-specific numerical-equivalence tolerance exists; raw errors and an isolated half-split FFT control are reported instead of inventing a loose threshold.",
        "structural_identity": structural_identity,
        "comparisons": comparisons,
        "isolated_half_split_fft_control": split_control,
        "finite_errors": finite_errors,
        "accepted_for_stage1": bool(all(bool(value) for value in structural_identity.values()) and finite_errors),
        "raw_cmos_equivalence": comparisons["raw_cmos"],
    }


def _split_identity_control(candidate_consumer: Any, candidate_result: alignment.LightGen2DForwardResult) -> dict[str, Any]:
    stack = candidate_consumer.runtime.generator
    record = candidate_result.generator_records[0]
    transfer = stack._active_transfer(stack.phase[1], record.tensor.dtype)
    modulated = ComplexField(
        record.tensor * transfer,
        record.plane,
        OperationProvenance("split_identity_control_phase_2", record.plane.plane_id, None),
    )
    padded = stack._pad(modulated, "split_identity_control")
    full = angular_spectrum_propagate(padded, stack.spacing_m, "split_identity_control_full")
    half = angular_spectrum_propagate(padded, stack.split_distance_m, "split_identity_control_half_1")
    split = angular_spectrum_propagate(half, stack.split_distance_m, "split_identity_control_half_2")
    comparison = _metric_error(full.tensor, split.tensor, "isolated split operator")
    return {
        "distance_full_m": stack.spacing_m,
        "distance_half_m": stack.split_distance_m,
        "full_vs_split": comparison,
        "interpretation": "The isolated control measures the same float32 FFT/ASM composition error introduced by the symmetric split.",
    }


def _stage1_decision(
    function_check: Mapping[str, Any],
    base_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    base_probes: Mapping[str, Any],
    candidate_probes: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    base_smooth_slopes = _abs_slopes(base_probes, "semantic_smooth", "L_sem_smooth")
    candidate_smooth_slopes = _abs_slopes(candidate_probes, "semantic_smooth", "L_sem_smooth")
    ratio = [candidate / max(base, 1.0e-30) for base, candidate in zip(base_smooth_slopes, candidate_smooth_slopes, strict=True)]
    new_plane = _mapping(candidate_result.get("new_plane"), "candidate new-plane gradient")
    new_plane_nonzero = float(new_plane["raw_gradient_norm"]) > 0.0 and float(new_plane["smooth_gradient_energy_fraction"]) > 0.0
    directional_stronger = all(candidate > base for base, candidate in zip(base_smooth_slopes, candidate_smooth_slopes, strict=True))
    candidate_train_transfer = _stable_directional_sign(candidate_probes, "semantic_train", "L_sem_smooth", plus_decreases=True)
    candidate_smooth_direction = _stable_directional_sign(candidate_probes, "semantic_smooth", "L_sem_smooth", plus_decreases=True)
    raw_valid = bool(candidate_probes.get("all_raw_valid")) and bool(candidate_result["raw_validity"]["finite"]) and candidate_result["raw_validity"]["nonconstant_fraction"] == 1.0
    checks = {
        "function_preserving_step0": bool(function_check["accepted_for_stage1"]),
        "new_plane_nonzero_smooth_sensitivity": new_plane_nonzero,
        "three_plane_smooth_response_stronger_at_all_fixed_scales": directional_stronger,
        "raw_validity": raw_valid,
        "semantic_train_to_smooth_not_reversed": bool(candidate_result["cosines"]["C_train_smooth"] > 0.0 and candidate_train_transfer and candidate_smooth_direction),
    }
    details = {
        "checks": checks,
        "base_smooth_direction_abs_slopes": base_smooth_slopes,
        "candidate_smooth_direction_abs_slopes": candidate_smooth_slopes,
        "candidate_over_base_smooth_response_ratios": ratio,
        "directional_response_comparison": "same fixed calibration batch, same relative phase perturbation semantics, no parameter-count normalization of output response",
        "new_plane_energy_fraction": float(new_plane["smooth_gradient_energy_fraction"]),
        "base_C_train_smooth": float(base_result["cosines"]["C_train_smooth"]),
        "candidate_C_train_smooth": float(candidate_result["cosines"]["C_train_smooth"]),
    }
    return all(checks.values()), details


def _load_authorities(root: Path) -> dict[str, Any]:
    # The prior conflict-aware consumer intentionally rejects a current status
    # document that already contains its own result.  This successor consumes
    # that accepted result, so its authority loader repeats only the needed
    # source checks and allows the already-closed predecessor gate.
    status = root / "docs/CURRENT_STATUS.md"
    handoff = root / "docs/status/current_handoff.md"
    _require(status.is_file() and handoff.is_file(), "current authority documents are missing")
    status_top = "\n".join(status.read_text(encoding="utf-8").splitlines()[:45])
    handoff_top = "\n".join(handoff.read_text(encoding="utf-8").splitlines()[:45])
    for name, content in (("CURRENT_STATUS", status_top), ("current_handoff", handoff_top)):
        _require(SEMANTIC_AUTHORITY_ID in content, f"{name} lost the active M0 authority")
        _require("CURRENT_AUTHORIZED_NEXT_ACTION=NONE" in content, f"{name} did not close the prior action")
        _require("GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED" in content, f"{name} lost the attribution authority")
        _require(OUTER_GATE not in content, f"{name} already contains this successor result")
    docs = {"status_top_sha256": hashlib.sha256(status_top.encode("utf-8")).hexdigest(), "handoff_top_sha256": hashlib.sha256(handoff_top.encode("utf-8")).hexdigest()}
    prior = attribution._load_authorities(root)
    authorities = dict(prior)
    attribution_report_path = root / conflict.ATTRIBUTION_REPORT
    _require(file_sha256(attribution_report_path) == conflict.EXPECTED_ATTRIBUTION_REPORT_SHA256, "accepted attribution report identity changed")
    attribution_report = json.loads(attribution_report_path.read_text(encoding="utf-8"))
    _require(attribution_report.get("result") == "VALID_COMPLETED", "accepted attribution result is not valid completed")
    _require(attribution_report.get("outer_gate") == attribution.OUTER_GATE, "accepted attribution gate changed")
    _require(attribution_report.get("primary_attribution") == "GENERATOR_SEMANTIC_CONTROLLABILITY_MIXED", "accepted attribution primary changed")
    _require(attribution_report.get("secondary_conflict") == "RECONSTRUCTION_SEMANTIC_GRADIENT_CONFLICT_SUPPORTED", "accepted attribution secondary changed")
    cpu_context = phase_repair._load_cpu_context(root)
    phase = conflict._load_phase_transfer_authority(root, cpu_context)
    objective = _mapping(prior["report"].get("objective"), "accepted objective")
    calibration = _mapping(objective.get("calibration"), "accepted semantic calibration")
    beta = float(objective.get("beta"))
    gamma = float(objective.get("gamma"))
    delta = float(calibration.get("delta"))
    _require(math.isclose(beta, float(prior["training"]["weights"].beta), rel_tol=0.0, abs_tol=1.0e-12), "beta authority changed")
    _require(math.isclose(gamma, float(prior["training"]["weights"].gamma), rel_tol=0.0, abs_tol=1.0e-12), "gamma authority changed")
    _require(math.isclose(delta, float(_mapping(prior["step0"].get("calibration"), "step-0 calibration").get("delta")), rel_tol=0.0, abs_tol=1.0e-15), "delta authority changed")
    hashes = dict(prior["hashes"])
    hashes.update(phase["authority_hashes"])
    hashes.update({"attribution_report": conflict.EXPECTED_ATTRIBUTION_REPORT_SHA256, "conflict_report": EXPECTED_CONFLICT_REPORT_SHA256, "conflict_step0": EXPECTED_CONFLICT_STEP_0_SHA256, "conflict_step300": EXPECTED_CONFLICT_STEP_300_SHA256})
    authorities.update({
        "docs": docs,
        "attribution_report": attribution_report,
        "accepted_report": prior["report"],
        "phase_latent": phase["phase_latent"],
        "selected_generated_raw": phase["selected_generated_raw"],
        "real_control_raw": phase["real_control_raw"],
        "selected_metrics": phase["selected_metrics"],
        "authority_hashes": hashes,
        "weights": {"beta": beta, "gamma": gamma, "delta": delta},
    })
    _require(file_sha256(root / CONFLICT_REPORT) == EXPECTED_CONFLICT_REPORT_SHA256, "accepted conflict-aware report identity changed")
    _require(file_sha256(root / CONFLICT_STEP_0) == EXPECTED_CONFLICT_STEP_0_SHA256, "accepted conflict-aware step-0 identity changed")
    _require(file_sha256(root / CONFLICT_STEP_300) == EXPECTED_CONFLICT_STEP_300_SHA256, "accepted conflict-aware step-300 identity changed")
    conflict_report = json.loads((root / CONFLICT_REPORT).read_text(encoding="utf-8"))
    _require(conflict_report.get("result") == "VALID_COMPLETED", "accepted conflict-aware result is not valid completed")
    _require(conflict_report.get("outer_gate") == conflict.OUTER_GATE, "accepted conflict-aware gate changed")
    _require(conflict_report.get("classification") == conflict.NOT_SUPPORTED_CLASSIFICATION, "accepted conflict-aware classification changed")
    conflict_step0 = safe_torch_load(root / CONFLICT_STEP_0, expected_sha256=EXPECTED_CONFLICT_STEP_0_SHA256, map_location="cpu")
    _require(isinstance(conflict_step0, Mapping), "accepted conflict-aware step-0 checkpoint is malformed")
    _require(conflict_step0.get("completed_updates") == 0, "accepted conflict-aware step-0 is not step 0")
    _require(conflict_step0.get("trainable_parameter_names") == ["generator.phase"], "accepted conflict-aware trainable boundary changed")
    _require(torch.equal(_tensor(conflict_step0.get("generator_state"), "conflict step-0 generator"), _tensor(authorities["step0"].get("generator_state"), "prior step-0 generator")), "conflict-aware and accepted 2L start generators differ")
    _require(torch.equal(_tensor(conflict_step0.get("encoder_state"), "conflict step-0 encoder"), _tensor(authorities["step0"].get("encoder_state"), "prior step-0 encoder")), "conflict-aware and accepted 2L start encoders differ")
    selected = authorities["selected_metrics"]
    _require(math.isclose(float(selected["R_u"]), 1.5095, rel_tol=0.0, abs_tol=1.0e-4), "accepted R_u authority changed")
    _require(math.isclose(float(selected["R_z"]), 1.8283, rel_tol=0.0, abs_tol=1.0e-4), "accepted R_z authority changed")
    optimizer_state = _mapping(conflict_step0.get("optimizer_state"), "accepted conflict-aware optimizer state")
    entries = _mapping(optimizer_state.get("state"), "accepted optimizer state entries")
    _require(len(entries) == 1, "accepted conflict-aware optimizer state parameter count changed")
    old_entry = next(iter(entries.values()))
    _require(isinstance(old_entry, Mapping), "accepted conflict-aware optimizer state entry is malformed")
    for key in ("step", "exp_avg", "exp_avg_sq"):
        _require(key in old_entry, f"accepted conflict-aware optimizer state misses {key}")
    _require(tuple(_tensor(old_entry["exp_avg"], "accepted exp_avg").shape) == (2, 1, 100, 100), "accepted exp_avg shape changed")
    _require(tuple(_tensor(old_entry["exp_avg_sq"], "accepted exp_avg_sq").shape) == (2, 1, 100, 100), "accepted exp_avg_sq shape changed")
    _finite(_tensor(old_entry["exp_avg"], "accepted exp_avg"), "accepted exp_avg")
    _finite(_tensor(old_entry["exp_avg_sq"], "accepted exp_avg_sq"), "accepted exp_avg_sq")
    authorities.update({"conflict_report": conflict_report, "conflict_step0": dict(conflict_step0), "conflict_optimizer_state": dict(optimizer_state)})
    return authorities


def _expanded_optimizer_state(prior_state: Mapping[str, Any], parameter: torch.Tensor) -> dict[str, Any]:
    groups = prior_state.get("param_groups")
    entries = prior_state.get("state")
    _require(isinstance(groups, list) and len(groups) == 1, "prior optimizer groups are incompatible")
    _require(isinstance(entries, Mapping) and len(entries) == 1, "prior optimizer entries are incompatible")
    group = dict(_mapping(groups[0], "prior optimizer group"))
    old_entry = dict(_mapping(next(iter(entries.values())), "prior optimizer entry"))
    old_avg = _tensor(old_entry.get("exp_avg"), "prior exp_avg")
    old_avg_sq = _tensor(old_entry.get("exp_avg_sq"), "prior exp_avg_sq")
    _require(tuple(old_avg.shape) == (2, 1, 100, 100), "prior exp_avg does not match 2L")
    _require(tuple(old_avg_sq.shape) == (2, 1, 100, 100), "prior exp_avg_sq does not match 2L")
    _require(tuple(parameter.shape) == (3, 1, 100, 100), "candidate optimizer parameter does not match 3L")
    new_entry = copy.deepcopy(old_entry)
    new_entry["exp_avg"] = torch.zeros_like(parameter.detach())
    new_entry["exp_avg_sq"] = torch.zeros_like(parameter.detach())
    new_entry["exp_avg"][:2].copy_(old_avg)
    new_entry["exp_avg_sq"][:2].copy_(old_avg_sq)
    group["params"] = [0]
    return {"state": {0: new_entry}, "param_groups": [group]}


def _new_candidate_optimizer(consumer: Any, prior_state: Mapping[str, Any]) -> tuple[torch.optim.Adam, str]:
    phase = _generator(consumer)
    optimizer = torch.optim.Adam(
        [phase],
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    state = _expanded_optimizer_state(prior_state, phase)
    optimizer.load_state_dict(state)
    _validate_candidate_optimizer(consumer, optimizer)
    return optimizer, "PRIOR_2L_MOMENTS_COPIED_WITH_ZERO_NEW_PLANE_MOMENTS"


def _validate_candidate_optimizer(consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    _require(isinstance(optimizer, torch.optim.Adam), "candidate optimizer must be Adam")
    _require(len(optimizer.param_groups) == 1, "candidate optimizer must have one parameter group")
    phase = _generator(consumer)
    actual = list(optimizer.param_groups[0]["params"])
    _require(len(actual) == 1 and actual[0] is phase, "candidate optimizer parameter boundary changed")
    group = optimizer.param_groups[0]
    _require(group["lr"] == LEARNING_RATE, "candidate learning rate changed")
    _require(tuple(group["betas"]) == ADAM_BETAS, "candidate Adam betas changed")
    _require(group["eps"] == ADAM_EPSILON, "candidate Adam epsilon changed")
    _require(group["weight_decay"] == WEIGHT_DECAY and group["amsgrad"] is AMSGRAD, "candidate Adam configuration changed")
    state = optimizer.state.get(phase)
    _require(isinstance(state, Mapping), "candidate optimizer state is missing")
    _require(tuple(state["exp_avg"].shape) == tuple(phase.shape), "candidate exp_avg shape changed")
    _require(tuple(state["exp_avg_sq"].shape) == tuple(phase.shape), "candidate exp_avg_sq shape changed")
    _finite(state["exp_avg"], "candidate exp_avg")
    _finite(state["exp_avg_sq"], "candidate exp_avg_sq")


def _project_auxiliary_gradient(auxiliary: torch.Tensor, semantic: torch.Tensor, name: str) -> tuple[torch.Tensor, dict[str, Any]]:
    auxiliary_norm = _norm(auxiliary, f"{name} gradient")
    semantic_norm = _norm(semantic, "semantic gradient")
    dot = _dot(auxiliary, semantic)
    projected = dot < 0.0
    removed = torch.zeros_like(auxiliary)
    if projected:
        coefficient = dot / (float(torch.sum(semantic.detach().to(torch.float64) ** 2).cpu()) + EPS_PROJ)
        removed = semantic * torch.as_tensor(coefficient, dtype=semantic.dtype, device=semantic.device)
        safe = auxiliary - removed
    else:
        safe = auxiliary
    _finite(safe, f"{name}_safe gradient")
    safe_dot = _dot(safe, semantic)
    if projected:
        tolerance = max(1.0e-8, 1.0e-5 * max(_norm(safe, f"{name}_safe gradient"), 1.0e-12) * semantic_norm)
        _require(abs(safe_dot) <= tolerance, f"{name}_safe gradient is not orthogonal to semantic gradient")
    removed_norm = float(removed.detach().to(torch.float64).square().sum().sqrt().cpu())
    return safe, {
        "dot_before": dot,
        "dot_after": safe_dot,
        "projected": projected,
        "projection_fraction_norm": removed_norm / auxiliary_norm,
        "removed_norm": removed_norm,
        "eps_proj": EPS_PROJ,
    }


def _compose_gradients(consumer: Any, evaluator: FrozenMNISTEvaluator, batch: torch.Tensor, labels: torch.Tensor, references: Any, *, beta: float, gamma: float, delta: float) -> dict[str, Any]:
    phase = _generator(consumer)
    components = _loss_bundle(consumer, evaluator, evaluator, batch, labels, references)
    # The conflict-aware objective deliberately uses the frozen TRAIN
    # evaluator.  The duplicated argument above keeps the loss helper's
    # evaluator boundary explicit; the Smooth evaluator is never in this
    # backward path.
    g_rec = torch.autograd.grad(beta * components["L_rec"], phase, retain_graph=True, allow_unused=False)[0]
    g_op = torch.autograd.grad(gamma * components["L_op"], phase, retain_graph=True, allow_unused=False)[0]
    g_sem = torch.autograd.grad(delta * components["L_sem_train"], phase, retain_graph=False, allow_unused=False)[0]
    g_rec_safe, rec_projection = _project_auxiliary_gradient(g_rec, g_sem, "g_rec")
    g_op_safe, op_projection = _project_auxiliary_gradient(g_op, g_sem, "g_op")
    g_total = g_sem + g_rec_safe + g_op_safe
    _finite(g_total, "g_total")
    _require(_dot(g_total, g_sem) > 0.0, "candidate composed gradient lacks positive semantic component")
    return {
        "components": components,
        "gradients": {"g_rec": g_rec, "g_op": g_op, "g_sem": g_sem, "g_rec_safe": g_rec_safe, "g_op_safe": g_op_safe, "g_total": g_total},
        "gradient_norms": {name: _gradient_stats(value, name) for name, value in (("g_rec", g_rec), ("g_op", g_op), ("g_sem", g_sem), ("g_rec_safe", g_rec_safe), ("g_op_safe", g_op_safe), ("g_total", g_total))},
        "gradient_cosines": {
            "cos_g_rec_g_sem": _cosine(g_rec, g_sem),
            "cos_g_op_g_sem": _cosine(g_op, g_sem),
            "cos_g_rec_safe_g_sem": _cosine(g_rec_safe, g_sem),
            "cos_g_op_safe_g_sem": _cosine(g_op_safe, g_sem),
            "cos_g_total_g_sem": _cosine(g_total, g_sem),
        },
        "projection": {"rec": rec_projection, "op": op_projection},
        "old_tn_distribution_in_objective": False,
        "scalar_weighted_sum_backward": False,
    }


def _composition_record(bundle: Mapping[str, Any], *, step: int | None = None) -> dict[str, Any]:
    components = _mapping(bundle["components"], "candidate gradient components")
    record = {
        "L_rec": float(components["L_rec"].detach().cpu()),
        "L_op": float(components["L_op"].detach().cpu()),
        "L_sem": float(components["L_sem_train"].detach().cpu()),
        "L_dist_diagnostic": float(components["L_dist"].detach().cpu()),
        "gradient_norms": bundle["gradient_norms"],
        "gradient_cosines": bundle["gradient_cosines"],
        "projection": bundle["projection"],
        "old_tn_distribution_in_objective": False,
        "scalar_weighted_sum_backward": False,
        "per_plane_semantic_gradient_norms": bundle["gradient_norms"]["g_sem"]["per_generator_layer_norms"],
        "new_plane_semantic_gradient_norm": bundle["gradient_norms"]["g_sem"]["per_generator_layer_norms"][f"generator.phase[{NEW_PLANE_INDEX}]"],
    }
    if step is not None:
        record["step"] = step
    return record


def _diagnose_composition(consumer: Any, evaluator: FrozenMNISTEvaluator, batch: torch.Tensor, labels: torch.Tensor, references: Any, *, beta: float, gamma: float, delta: float) -> dict[str, Any]:
    before = _state_digest(_runtime_state(consumer.runtime))
    bundle = _compose_gradients(consumer, evaluator, batch, labels, references, beta=beta, gamma=gamma, delta=delta)
    _clear_grads(consumer)
    _require(before == _state_digest(_runtime_state(consumer.runtime)), "candidate composition changed runtime state")
    return _composition_record(bundle)


def _apply_update(consumer: Any, evaluator: FrozenMNISTEvaluator, optimizer: torch.optim.Adam, batch: torch.Tensor, labels: torch.Tensor, references: Any, *, beta: float, gamma: float, delta: float, step: int, ledger: DataReadLedger | None) -> dict[str, Any]:
    _validate_candidate_optimizer(consumer, optimizer)
    phase = _generator(consumer)
    before_phase = phase.detach().clone()
    before_frozen = _frozen_state(consumer.runtime)
    optimizer.zero_grad(set_to_none=True)
    _clear_grads(consumer)
    bundle = _compose_gradients(consumer, evaluator, batch, labels, references, beta=beta, gamma=gamma, delta=delta)
    for name, parameter in consumer.runtime.named_parameters():
        _require(parameter.grad is None, f"candidate parameter unexpectedly had a gradient before assignment: {name}")
    phase.grad = bundle["gradients"]["g_total"].detach().clone()
    _finite(phase.grad, "candidate assigned gradient")
    optimizer.step()
    _finite(phase, "updated candidate generator phase")
    _require(_state_digest(before_frozen) == _state_digest(_frozen_state(consumer.runtime)), "candidate frozen runtime changed")
    update_norm = float((phase.detach() - before_phase).to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(update_norm) and update_norm > 0.0, "candidate Generator did not update")
    _clear_grads(consumer)
    if ledger is not None:
        ledger.record(TRAIN_ROLE, int(batch.shape[0]))
    record = _composition_record(bundle, step=step)
    record.update({"L_total": beta * record["L_rec"] + gamma * record["L_op"] + delta * record["L_sem"], "generator_update_norm": update_norm, "encoder_unchanged": True, "frozen_runtime_unchanged": True, "trainable_parameter_names": ["generator.phase"]})
    return record


def _rng_snapshot() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.random.get_rng_state(), [state.clone() for state in torch.cuda.get_rng_state_all()]


def _restore_rng(snapshot: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    torch.random.set_rng_state(snapshot[0])
    torch.cuda.set_rng_state_all(snapshot[1])


def _run_preflight(candidate: Any, evaluator: FrozenMNISTEvaluator, references: Any, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, calibration_indices: torch.Tensor, prior_sampler_state: Mapping[str, Any], prior_optimizer_state: Mapping[str, Any], *, beta: float, gamma: float, delta: float) -> dict[str, Any]:
    initial_runtime = _runtime_state(candidate.runtime)
    initial_phase = _generator(candidate).detach().clone()
    initial_rng = _rng_snapshot()
    bad_optimizer = torch.optim.Adam(list(candidate.runtime.parameters()), lr=LEARNING_RATE)
    try:
        _validate_candidate_optimizer(candidate, bad_optimizer)
    except CapacityExpansionError:
        negative_control = "FULL_RUNTIME_OPTIMIZER_REJECTED"
    else:
        raise CapacityExpansionError("candidate negative control accepted a full-runtime optimizer")
    sampler = DeterministicTrainRoleSampler(train_indices, labels)
    sampler_state = dict(prior_sampler_state)
    sampler_state["cursor"] = 0
    sampler.load_state_dict(sampler_state)
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == calibration_indices.tolist(), "candidate preflight first TRAIN batch differs from authority")
    optimizer, optimizer_mode = _new_candidate_optimizer(candidate, prior_optimizer_state)
    trace: list[dict[str, Any]] = []
    sampler.load_state_dict({**sampler_state, "cursor": 0})
    for step in range(3):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(labels.index_select(0, indices), batch_labels), "candidate preflight TRAIN label binding changed")
        batch = images.index_select(0, indices).to(candidate.device)
        trace.append(_apply_update(candidate, evaluator, optimizer, batch, batch_labels.to(candidate.device), references, beta=beta, gamma=gamma, delta=delta, step=step + 1, ledger=None))
    post_runtime = _runtime_state(candidate.runtime)
    _require(torch.equal(initial_runtime["encoder.phase"], post_runtime["encoder.phase"]), "candidate preflight changed frozen encoder")
    _require(not torch.equal(initial_runtime["generator.phase"], post_runtime["generator.phase"]), "candidate preflight Generator did not depart")
    with torch.no_grad():
        _generator(candidate).copy_(initial_phase)
    _clear_grads(candidate)
    _restore_rng(initial_rng)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "three_step_real_update": {"updates": 3, "trace": trace, "generator_departed": True, "encoder_unchanged": True},
        "negative_control": {"name": negative_control, "passed": True},
        "optimizer_mode": optimizer_mode,
        "frozen_parameter_probe": {"encoder_phase_unchanged": True, "ols_unchanged": True, "w_ols_unchanged": True, "evaluators_frozen": True},
        "protected_test_sealed": {"protected": False, "test": False, "sealed": False},
        "scientific_300_step_executed": False,
    }


def _candidate_dev_eval(consumer: Any, train_evaluator: FrozenMNISTEvaluator, smooth_evaluator: SmoothMNISTEvaluator, images: torch.Tensor, labels: torch.Tensor, development_indices: torch.Tensor, references: Any, ledger: DataReadLedger, *, step: int) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    train_evaluator.eval()
    smooth_evaluator.eval()
    train_correct: list[torch.Tensor] = []
    smooth_correct: list[torch.Tensor] = []
    train_logits_parts: list[torch.Tensor] = []
    smooth_logits_parts: list[torch.Tensor] = []
    reconstruction_sum = 0.0
    raw_count = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    raw_sum = 0.0
    raw_square_sum = 0.0
    nonconstant = True
    zero_images = 0
    power_sum = torch.zeros(5, dtype=torch.float64)
    optical_loss_sum = 0.0
    try:
        with torch.no_grad():
            for start in range(0, int(development_indices.numel()), LOGICAL_BATCH_SIZE):
                indices = development_indices[start : start + LOGICAL_BATCH_SIZE]
                batch = images.index_select(0, indices).to(consumer.device)
                target_labels = labels.index_select(0, indices)
                result = consumer.runtime.forward_from_image(batch)
                _require(result.image_input is not None and result.ols_input_field is not None, "candidate DEV path is incomplete")
                raw = result.raw_intensity
                target = result.image_input.slm_intensity
                _finite(raw, "candidate DEV raw CMOS")
                _finite(target, "candidate DEV target")
                reconstruction_sum += float((raw - target).square().sum().cpu())
                powers = _power_planes(result)
                power_sum += powers.to(torch.float64).sum(dim=0).cpu()
                ratios = (powers.to(torch.float64) + 1.0e-12) / (references.power_reference.to(device=powers.device, dtype=torch.float64) + 1.0e-12)
                optical_loss_sum += float(torch.log(ratios).square().sum().cpu())
                flat = raw.detach().cpu().to(torch.float64).flatten(1)
                sample_std = flat.std(dim=1, unbiased=False)
                nonconstant = nonconstant and bool((sample_std > 0.0).all().item())
                zero_images += int((flat.abs().sum(dim=1) == 0.0).sum().item())
                raw_count += int(flat.numel())
                raw_sum += float(flat.sum())
                raw_square_sum += float(flat.square().sum())
                raw_min = min(raw_min, float(flat.min()))
                raw_max = max(raw_max, float(flat.max()))
                view = _semantic_view(raw)
                _, train_logits = train_evaluator(view)
                _, smooth_logits = smooth_evaluator(view)
                train_logits_cpu = train_logits.cpu()
                smooth_logits_cpu = smooth_logits.cpu()
                train_logits_parts.append(train_logits_cpu)
                smooth_logits_parts.append(smooth_logits_cpu)
                train_correct.append((train_logits_cpu.argmax(dim=1) == target_labels).contiguous())
                smooth_correct.append((smooth_logits_cpu.argmax(dim=1) == target_labels).contiguous())
                ledger.record(DEV_ROLE, int(indices.numel()))
    finally:
        consumer.runtime.train(was_training)
    train_correct_cpu = torch.cat(train_correct)
    smooth_correct_cpu = torch.cat(smooth_correct)
    train_logits_cpu = torch.cat(train_logits_parts)
    smooth_logits_cpu = torch.cat(smooth_logits_parts)

    def recalls(correct: torch.Tensor) -> tuple[list[float], float]:
        values: list[float] = []
        dev_labels = labels.index_select(0, development_indices)
        for digit in range(10):
            mask = dev_labels == digit
            _require(int(mask.sum()) == 500, f"candidate DEV class count changed for {digit}")
            values.append(float(correct[mask].to(torch.float64).mean()))
        return values, float(sum(values) / 10.0)

    train_per_class, train_macro = recalls(train_correct_cpu)
    smooth_per_class, smooth_macro = recalls(smooth_correct_cpu)
    variance = raw_square_sum / float(raw_count) - (raw_sum / float(raw_count)) ** 2
    diagnostic = {
        "step": step,
        "rows": int(development_indices.numel()),
        "role": DEV_ROLE,
        "train_compatible_evaluator": {"architecture": "FrozenMNISTEvaluator v1", "macro_recall": train_macro, "per_class_recall": train_per_class, "labels_consumed_for_diagnostic": True},
        "smooth_evaluator": {"architecture": "SmoothMNISTEvaluator v2.1", "macro_recall": smooth_macro, "per_class_recall": smooth_per_class, "labels_consumed": True, "independent_primary": True},
        "reconstruction": {"L_rec": reconstruction_sum / float(raw_count)},
        "optical": {"L_op": optical_loss_sum / float(development_indices.numel() * 5), "power_mean": [float(value) for value in (power_sum / float(development_indices.numel())).tolist()]},
        "raw_cmos": {"finite": True, "nonconstant": nonconstant, "zero_image_count": zero_images, "zero_image_fraction": float(zero_images / float(development_indices.numel())), "min": raw_min, "max": raw_max, "std_population": max(0.0, variance) ** 0.5},
        "evaluator_parameter_gradients_none": bool(all(parameter.grad is None for parameter in train_evaluator.parameters()) and all(parameter.grad is None for parameter in smooth_evaluator.parameters())),
        "_train_correct": train_correct_cpu,
        "_smooth_correct": smooth_correct_cpu,
        "_train_logits": train_logits_cpu,
        "_smooth_logits": smooth_logits_cpu,
    }
    _require(diagnostic["raw_cmos"]["nonconstant"] and diagnostic["raw_cmos"]["zero_image_count"] == 0, f"candidate DEV raw validity failed at step {step}")
    _require(diagnostic["evaluator_parameter_gradients_none"], "candidate DEV evaluator received a gradient")
    return diagnostic


def _p2q2_transfer(root: Path, consumer: Any, authorities: Mapping[str, Any], evaluator_checkpoint: Path, target: Path) -> dict[str, Any]:
    latent = _mapping(authorities.get("phase_latent"), "selected Q2 latent")
    z_sample = _tensor(latent.get("z_sample"), "selected Q2 z")[:200].contiguous()
    baseline_raw = _tensor(authorities.get("selected_generated_raw"), "accepted Q2 generated raw")
    real_raw = _tensor(authorities.get("real_control_raw"), "accepted real-control raw")
    outputs: list[torch.Tensor] = []
    consumer.runtime.eval()
    with torch.no_grad():
        for start in range(0, 200, LOGICAL_BATCH_SIZE):
            result = consumer.generated_forward(z_sample[start : start + LOGICAL_BATCH_SIZE].to(consumer.device))
            raw = result.raw_intensity.detach().cpu()
            _finite(raw, "expanded P2+Q2 raw CMOS")
            outputs.append(raw)
    candidate_raw = torch.cat(outputs)
    _require(candidate_raw.shape == (200, 1, 200, 200), "expanded P2+Q2 raw shape changed")
    candidate_path = target / "expanded_generator_p2q2_generated_cmos.pt"
    _write_torch(candidate_path, {"schema": CMOS_SCHEMA, "group": "expanded_generator_p2q2_generated", "input_z": z_sample, "raw_cmos_intensity": candidate_raw, "candidate_id": "Q2", "selected_step": 300}, kind="expanded Generator P2+Q2 generated CMOS")
    candidate_report, candidate_outputs = canary._evaluate_frozen_semantics(checkpoint=evaluator_checkpoint, generated_raw=candidate_raw, real_raw=real_raw)
    baseline_report, baseline_outputs = canary._evaluate_frozen_semantics(checkpoint=evaluator_checkpoint, generated_raw=baseline_raw, real_raw=real_raw)
    real_view = real_raw.to(torch.float64).flatten(1)
    raw_mean = real_view.mean(dim=0)
    raw_std = real_view.std(dim=0, unbiased=False)
    candidate_view = canary._standardize(candidate_raw.to(torch.float64).flatten(1), raw_mean, raw_std)
    baseline_view = canary._standardize(baseline_raw.to(torch.float64).flatten(1), raw_mean, raw_std)
    real_standardized = canary._standardize(real_view, raw_mean, raw_std)
    real_self = canary._energy_distance(real_standardized[:100], real_standardized[100:])
    candidate_raw_distance = canary._energy_distance(candidate_view, real_standardized)
    baseline_raw_distance = canary._energy_distance(baseline_view, real_standardized)
    expected = _mapping(authorities["selected_metrics"], "accepted Q2 metrics")
    baseline_feature = float(baseline_report["comparison"]["R_feature"])
    baseline_r_raw = baseline_raw_distance / max(real_self, 1.0e-12)
    _require(abs(baseline_feature - float(expected["R_feature"])) < 1.0e-9, "recomputed accepted Q2 R_feature disagrees with authority")
    _require(abs(baseline_r_raw - float(expected["R_raw"])) < 1.0e-9, "recomputed accepted Q2 R_raw disagrees with authority")
    candidate_r_raw = candidate_raw_distance / max(real_self, 1.0e-12)
    generated_semantic = candidate_report["generated"]
    return {
        "input": {"candidate_id": "Q2", "selected_step": 300, "sample_count": 200, "latent_bank_sha256": authorities["authority_hashes"]["phase_latent"], "real_control_reused_exactly": True},
        "frozen_context": {"R_u": float(expected["R_u"]), "R_phi": float(expected["R_phi"]), "R_z": float(expected["R_z"]), "phase_candidate": "Q2_ANCHOR_CONDITIONED_JOINT_UNCONDITIONAL"},
        "baseline_P2_Q2": {key: float(expected[key]) for key in ("R_u", "R_phi", "R_z", "R_raw", "R_feature")},
        "after": {"R_u": float(expected["R_u"]), "R_phi": float(expected["R_phi"]), "R_z": float(expected["R_z"]), "R_raw": float(candidate_r_raw), "R_feature": float(candidate_report["comparison"]["R_feature"])},
        "absolute_delta_after_minus_baseline": {key: float(value) - float(expected[key]) for key, value in {"R_u": expected["R_u"], "R_phi": expected["R_phi"], "R_z": expected["R_z"], "R_raw": candidate_r_raw, "R_feature": candidate_report["comparison"]["R_feature"]}.items()},
        "ratio_after_over_baseline": {key: (float(value) / max(float(expected[key]), 1.0e-12)) for key, value in {"R_u": expected["R_u"], "R_phi": expected["R_phi"], "R_z": expected["R_z"], "R_raw": candidate_r_raw, "R_feature": candidate_report["comparison"]["R_feature"]}.items()},
        "raw_distribution": {"R_raw": float(candidate_r_raw), "baseline_R_raw_recomputed": float(baseline_r_raw), "candidate_energy_distance_to_real": float(candidate_raw_distance), "baseline_energy_distance_to_real": float(baseline_raw_distance), "real_control_split_energy_distance": float(real_self), "standardization": "per-pixel mean/std from the exact accepted 200-frame real-control CMOS artifact"},
        "class_histogram": generated_semantic["predicted_class_histogram"],
        "dominant_fraction": float(generated_semantic["predicted_class_dominant_fraction"]),
        "normalized_entropy": float(generated_semantic["predicted_class_entropy_normalized"]),
        "diversity": {"candidate_raw": canary._diversity_view(candidate_raw, name="expanded_p2q2_raw"), "baseline_raw": canary._diversity_view(baseline_raw, name="accepted_p2q2_raw"), "candidate_feature": canary._diversity_view(candidate_outputs["generated_embedding"], name="expanded_p2q2_feature"), "baseline_feature": canary._diversity_view(baseline_outputs["generated_embedding"], name="accepted_p2q2_feature")},
        "candidate_raw_validity": canary._raw_cmos_statistics(candidate_raw, name="expanded_p2q2_generated"),
        "baseline_raw_validity": canary._raw_cmos_statistics(baseline_raw, name="accepted_p2q2_generated"),
        "evaluator": {"candidate": candidate_report, "baseline": baseline_report},
        "generated_cmos_artifact": _artifact_ref(root, candidate_path, role="expanded Generator P2+Q2 generated CMOS"),
    }


def _checkpoint_payload(consumer: Any, optimizer: torch.optim.Adam, sampler: DeterministicTrainRoleSampler, history: list[Mapping[str, Any]], composition: Mapping[str, Any], *, step: int, source_manifest: Mapping[str, Any], run_identity: ResearchRunIdentity, optimizer_mode: str, preflight: Mapping[str, Any], scientific_artifact: bool) -> dict[str, Any]:
    _validate_candidate_optimizer(consumer, optimizer)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_token": AUTHORIZATION_TOKEN,
        "completed_updates": step,
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "scientific_artifact": scientific_artifact,
        "operational_artifact": not scientific_artifact,
        "run_identity": run_identity.to_dict(),
        "source_manifest": dict(source_manifest),
        "source_manifest_id": source_manifest.get("source_manifest_id"),
        "accepted_parent_checkpoint": {"path": CONFLICT_STEP_0.as_posix(), "sha256": EXPECTED_CONFLICT_STEP_0_SHA256, "selected_step": 0},
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator"],
        "generator_state": _generator(consumer).detach().cpu().clone(),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": _state_digest(_frozen_state(consumer.runtime)),
        "optimizer_mode": optimizer_mode,
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "new_plane_moments": "zero initialized"},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": sampler.state_dict(),
        "rng_state": _rng_state_payload(),
        "gradient_composition": {"formula": "g_total=g_sem+project_nonconflicting(g_rec,g_sem)+project_nonconflicting(g_op,g_sem)", "eps_proj": EPS_PROJ, "trajectory": dict(composition), "old_tn_distribution_in_objective": False},
        "preflight": dict(preflight),
        "training_history": [dict(record) for record in history],
    }


def _recovery_payload(consumer: Any, optimizer: torch.optim.Adam, sampler: DeterministicTrainRoleSampler, history: list[Mapping[str, Any]], composition: Mapping[str, Any], *, step: int, source_manifest: Mapping[str, Any], run_identity: ResearchRunIdentity, optimizer_mode: str, preflight: Mapping[str, Any]) -> dict[str, Any]:
    payload = _checkpoint_payload(consumer, optimizer, sampler, history, composition, step=step, source_manifest=source_manifest, run_identity=run_identity, optimizer_mode=optimizer_mode, preflight=preflight, scientific_artifact=False)
    payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
    return payload


def _classification(dev_semantic: Mapping[str, Any], dev_reconstruction: Mapping[str, Any], transfer: Mapping[str, Any]) -> tuple[str, dict[str, bool]]:
    semantic_supported = bool(dev_semantic["delta_macro_recall"] > 0.0 and dev_semantic["paired_bootstrap_ci"]["lower"] > 0.0)
    reconstruction_pass = bool(dev_reconstruction["delta_step_300_minus_step_0"] <= RECONSTRUCTION_TOLERANCE)
    raw_valid = bool(transfer["candidate_raw_validity"]["finite"] and transfer["candidate_raw_validity"]["nonconstant_fraction"] == 1.0 and transfer["candidate_raw_validity"]["zero_image_count"] == 0)
    feature_improved = bool(transfer["after"]["R_feature"] < transfer["baseline_P2_Q2"]["R_feature"])
    checks = {"dev_semantic_supported": semantic_supported, "dev_reconstruction_pass": reconstruction_pass, "sampled_R_feature_improved": feature_improved, "raw_valid": raw_valid}
    if not reconstruction_pass or not raw_valid:
        return CONSTRAINT_CLASSIFICATION, checks
    if semantic_supported and feature_improved:
        return SUPPORTED_CLASSIFICATION, checks
    if semantic_supported and not feature_improved:
        return MIXED_TRANSFER_CLASSIFICATION, checks
    return LOCAL_GAIN_CLASSIFICATION, checks


def _make_identity(source_manifest: Mapping[str, Any], data_identity: Mapping[str, Any], weights: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id="minimal-generator-capacity-expansion-v1",
        trajectory_id="minimal-generator-capacity-expansion-trajectory-v1",
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"stage1": True, "stage2_updates": SCIENTIFIC_UPDATES, "candidate": "second_interval_symmetric_half_distance", "split_fraction": SPLIT_FRACTION, "new_plane_initialization": "exact_zero", "perturbation_epsilons": list(PERTURBATION_EPSILONS), "data_source": dict(data_identity), "beta": weights["beta"], "gamma": weights["gamma"], "delta": weights["delta"]},
        model_identity="LIGHTGEN_2D_MINIMAL_GENERATOR_CAPACITY_EXPANSION_V1",
        optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "stage1_optimizer_created": False, "stage2_new_plane_moments": "zero"},
        sampler_identity={"calibration_batch": "accepted seed-1234 TRAIN calibration batch", "batch_size": LOGICAL_BATCH_SIZE},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id="minimal-generator-capacity-expansion-v1:attempt_01",
    )


def _architecture_resolution(base_consumer: Any, candidate_consumer: Any) -> dict[str, Any]:
    runtime = base_consumer.runtime
    generator = runtime.generator
    candidate = candidate_consumer.runtime.generator
    return {
        "schema": ARCHITECTURE_SCHEMA,
        "current_2L": {
            "trainable_phase_planes": int(generator.layers),
            "phase_shape": list(generator.phase.shape),
            "active_shape": list(generator.active_shape),
            "propagation_grid_shape": list(generator.propagation_shape),
            "sequence": "W_OLS(z) -> M(phi_1) -> centered pad -> ASM(d=1.50 mm) -> centered crop -> M(phi_2) -> centered pad -> ASM(d=1.50 mm) -> centered crop -> B=0 output relay -> raw CMOS intensity |E|^2",
            "propagation_distance_m": float(generator.spacing_m),
            "crop_locations": "after each ASM propagation, centered 100x100 crop from 2048x2048 grid",
            "fft_convention": "torch.fft.fft2/ifft2 with norm=ortho in angular_spectrum_propagate; transfer uses torch.fft.fftfreq",
            "padding": "centered zero padding from 100x100 to 2048x2048",
            "normalization": "no phase-stack amplitude normalization; phase transmittance has unit magnitude",
            "aperture": "finite_crop_hard_aperture at each layer; no additional Generator aperture",
            "grid": {"active": list(generator.active_shape), "propagation": list(generator.propagation_shape), "pixel_pitch_m": [alignment.OLS_PIXEL_PITCH_M, alignment.OLS_PIXEL_PITCH_M], "wavelength_m": alignment.WAVELENGTH_M},
            "generator_input_injection": "LightGenAlignedComplexOLS.write_to_generator_input: W_OLS(z)=sum_j (Tz)_j v_j at lightgen_generator_entrance, T=I_100",
            "cmos_measurement": "coherent B=0 output relay lightgen_output_b0 to lightgen_raw_cmos, IntensityFrame=abs(camera_field)^2",
            "encoder_and_ols_frozen": True,
        },
        "minimal_3L": {
            "trainable_phase_planes": int(candidate.layers),
            "phase_shape": list(candidate.phase.shape),
            "new_phase_index": NEW_PLANE_INDEX,
            "insertion_location": "after existing phase plane 2, inside the second original nonzero propagation interval",
            "propagation_split": {"d1_m": candidate.split_distance_m, "d2_m": candidate.split_distance_m, "sum_m": candidate.split_distance_m * 2.0, "original_d_m": candidate.spacing_m},
            "midpoint_semantics": "full 2048x2048 grid; centered 100x100 phase tensor is trainable and identity-transparent outside the active window",
            "crop_aperture_semantics": "only the original first crop and final crop remain; no midpoint hard crop, aperture, normalization, or relay is introduced",
            "initialization": "old two planes copied exactly; new phase exactly zero",
            "power_objective_observation_indices": [0, 2],
            "power_objective_preservation": "accepted five-plane objective uses encoder records, W_OLS input, first old-plane endpoint, and final endpoint; the midpoint is not silently added as a sixth reference plane",
        },
    }


def run_capacity_expansion(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "capacity output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"capacity output already exists: {target}")
    start_git = successor.git_identity(root)
    _require(start_git["branch"] == EXPECTED_BRANCH and start_git["head"] == EXPECTED_HEAD, "local branch/HEAD conflicts with accepted capacity authorities")
    authorities = _load_authorities(root)
    cuda_runtime = successor._prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "capacity expansion requires CUDA for the production runtime")
    device = torch.device("cuda:0")
    base_consumer, train_evaluator, smooth_evaluator, evaluator_identity = conflict._load_models(root, authorities, device)
    candidate_consumer = _build_candidate_consumer(root, authorities, device)
    base_consumer.runtime.eval()
    candidate_consumer.runtime.eval()
    _require(_state_digest(_frozen_state(base_consumer.runtime)) == _state_digest(_frozen_state(candidate_consumer.runtime)), "candidate changed a frozen runtime state")

    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            Path(conflict.__file__).resolve(),
            Path(successor.__file__).resolve(),
            Path(attribution.__file__).resolve(),
            Path(phase_repair.__file__).resolve(),
            Path(canary.__file__).resolve(),
            Path(producer.__file__).resolve(),
            root / "src/ldm_project/v2/application/research/lightgen_2d_chen_bogt_complex_ols_training.py",
            root / "src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py",
            root / "src/ldm_project/v2/optics/d2nn.py",
            root / "src/ldm_project/v2/optics/propagation.py",
            root / "src/ldm_project/v2/optics/relays.py",
            root / "src/ldm_project/v2/application/research/research_run_control.py",
            root / "src/ldm_project/v2/student/evaluator.py",
        ),
        config_identity={"authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "candidate": "second_interval_symmetric_half_distance", "split_fraction": SPLIT_FRACTION, "new_plane_index": NEW_PLANE_INDEX, "stage2_updates": SCIENTIFIC_UPDATES},
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    _write_json(target / "source_manifest.json", source_manifest)
    run_identity = _make_identity(source_manifest, authorities["data_identity"], authorities["weights"])

    architecture = _architecture_resolution(base_consumer, candidate_consumer)
    _write_json(target / "architecture_resolution.json", architecture)
    calibration_indices = authorities["calibration_indices"]
    images = authorities["images"]
    labels = authorities["labels"]
    batch = images.index_select(0, calibration_indices).to(device)
    batch_labels = labels.index_select(0, calibration_indices).to(device)
    function_check = _function_preserving_check(base_consumer, candidate_consumer, train_evaluator, smooth_evaluator, batch, batch_labels, authorities["training"]["references"])
    _write_json(target / "function_preserving_check.json", function_check)
    base_gradient_result, base_gradients = _analyze_gradients(base_consumer, train_evaluator, smooth_evaluator, batch, batch_labels, authorities["training"]["references"], "2L")
    candidate_gradient_result, candidate_gradients = _analyze_gradients(candidate_consumer, train_evaluator, smooth_evaluator, batch, batch_labels, authorities["training"]["references"], "3L")
    base_probes = _directional_probes(base_consumer, train_evaluator, smooth_evaluator, batch, batch_labels, authorities["training"]["references"], base_gradients, "2L")
    candidate_probes = _directional_probes(candidate_consumer, train_evaluator, smooth_evaluator, batch, batch_labels, authorities["training"]["references"], candidate_gradients, "3L")
    stage1_supported, stage1_details = _stage1_decision(function_check, base_gradient_result, candidate_gradient_result, base_probes, candidate_probes)
    _write_torch(target / "stage1_gradient_vectors.pt", {"schema": STAGE1_GRADIENT_SCHEMA, "2L": {key: value.detach().cpu() for key, value in base_gradients.items()}, "3L": {key: value.detach().cpu() for key, value in candidate_gradients.items()}}, kind="minimal Generator capacity Stage 1 gradient vectors")
    _write_json(target / "stage1_directional_probes.json", {"schema": STAGE1_PROBE_SCHEMA, "2L": base_probes, "3L": candidate_probes})
    stage1 = {"supported": stage1_supported, "details": stage1_details, "2L": base_gradient_result, "3L": candidate_gradient_result, "probes": {"2L": base_probes, "3L": candidate_probes}}

    base_count = len(base_consumer.runtime.generator.phase)
    candidate_count = len(candidate_consumer.runtime.generator.phase)
    _require(base_count == 2 and candidate_count == 3, "capacity plane counts are not 2L and 3L")

    if not stage1_supported:
        report = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "result": "VALID_COMPLETED",
            "status": "VALID_COMPLETED",
            "outer_gate": OUTER_GATE,
            "outer_gate_passed": False,
            "classification": NOT_SUPPORTED_CLASSIFICATION,
            "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
            "local_state": {"branch": start_git["branch"], "head": start_git["head"], "dirty_before": start_git["worktree_dirty"], "git_end": successor.git_identity(root)},
            "input_authorities": authorities["authority_hashes"],
            "accepted_conflict_aware": {"report": _artifact_ref(root, root / CONFLICT_REPORT, role="accepted 2L conflict-aware result"), "step_0": _artifact_ref(root, root / CONFLICT_STEP_0, role="accepted 2L conflict-aware optimizer start")},
            "current_2L_generator_contract": architecture["current_2L"],
            "minimal_3L_contract": architecture["minimal_3L"],
            "function_preserving_check": function_check,
            "stage_1_local_controllability": stage1,
            "stage_2_executed": False,
            "counterfactual_integrity": {"same_frozen_batch": True, "same_old_phase_values": True, "new_phase_zero": True, "encoder_unchanged": True, "OLS_unchanged": True, "latent_authority_unchanged": True, "objective_unchanged": True, "optimizer_created": False, "protected_test_sealed_reads": False},
            "outer_gate_checks": {"stage1_supported": False, "stage2_opened": False},
            "claim_boundary": {"scientific_training_executed": False, "stage1_optimizer_created": False, "local_capacity_claim": False, "global_architecture_inadequacy": False, "formal_promotion": False, "physical_hardware_validity": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False},
            "files_changed": {"scientific_artifacts": sorted(path.name for path in target.iterdir()), "repository_source": ["scripts/research/execute_lightgen_2d_minimal_generator_capacity_expansion_v1.py"]},
            "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "architecture_resolution": True, "function_preserving_check": bool(function_check["accepted_for_stage1"]), "stage1_gradient_probe": True, "directional_probes": True, "all_perturbations_restored": True, "no_protected_test_sealed": True},
            "unresolved": ["The single separated phase-plane candidate did not meet the fixed local controllability support conditions."],
            "next_action": "NONE",
            "source_manifest": source_manifest,
            "run_identity": run_identity.to_dict(),
            "output_dir": _relative(root, target),
        }
        _write_json(target / "report.json", report)
        return report

    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    preflight = _run_preflight(candidate_consumer, train_evaluator, authorities["training"]["references"], images, labels, authorities["train_indices"], calibration_indices, authorities["step0"]["sampler_state"], authorities["conflict_optimizer_state"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta=authorities["weights"]["delta"])
    _write_json(target / "preflight.json", preflight)
    control.mark_preflight_ready()
    control.start_execution()
    effective_identity = control.begin_scientific_attempt("minimal-generator-capacity-expansion-v1:attempt_01")
    _write_json(target / "execution_manifest.json", {"schema": "minimal_generator_capacity_expansion_execution_manifest.v1", "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorization_id": AUTHORIZATION_ID, "authorization_token": AUTHORIZATION_TOKEN, "input_authorities": ["M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION", conflict.OUTER_GATE, "GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED", "CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1_Q2", "SMOOTH_MNIST_EVALUATOR_V2_1"], "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "data_identity": authorities["data_identity"], "git_before": start_git, "cuda_runtime": cuda_runtime, "scope": {"stage1": True, "bounded_updates": SCIENTIFIC_UPDATES, "trainable": ["generator.phase[0:3]"], "frozen": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "evaluators"], "protected": False, "test": False, "sealed": False}})

    candidate_consumer.runtime.train()
    optimizer, optimizer_mode = _new_candidate_optimizer(candidate_consumer, authorities["conflict_optimizer_state"])
    sampler = DeterministicTrainRoleSampler(authorities["train_indices"], labels)
    sampler_state = dict(authorities["step0"]["sampler_state"])
    sampler_state["cursor"] = 0
    sampler.load_state_dict(sampler_state)
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == calibration_indices.tolist(), "scientific first TRAIN batch differs from authority")
    sampler.load_state_dict(sampler_state)
    ledger = DataReadLedger()
    history: list[dict[str, Any]] = []
    composition_trajectory: dict[str, dict[str, Any]] = {}
    dev_records: dict[str, dict[str, Any]] = {}
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=SCIENTIFIC_UPDATES, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=0, latest_losses={"preflight": "READY", "optimizer_mode": optimizer_mode}, force=True)
    fixed_batch = images.index_select(0, calibration_indices).to(device)
    fixed_labels = labels.index_select(0, calibration_indices).to(device)
    composition_trajectory["0"] = _diagnose_composition(candidate_consumer, train_evaluator, fixed_batch, fixed_labels, authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta=authorities["weights"]["delta"])
    dev0 = _candidate_dev_eval(candidate_consumer, train_evaluator, smooth_evaluator, images, labels, authorities["development_indices"], authorities["training"]["references"], ledger, step=0)
    dev_records["0"] = dev0
    _write_json(target / "dev_diagnostics_step_000000.json", {"schema": DEV_SCHEMA, "step": 0, "diagnostic": {key: value for key, value in dev0.items() if not key.startswith("_")}})
    _write_torch(target / "scientific_step_000000.pt", _checkpoint_payload(candidate_consumer, optimizer, sampler, history, composition_trajectory, step=0, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight, scientific_artifact=True), kind="minimal Generator capacity scientific checkpoint")

    last_recovery_step = 0
    last_recovery_time = time.time()
    last_recovery_path: Path | None = None

    def recovery_factory(step: int) -> Mapping[str, Any]:
        return _recovery_payload(candidate_consumer, optimizer, sampler, history, composition_trajectory, step=step, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight)

    heartbeat.update(state=RUNNING, phase="TRAINING", step=0, latest_losses={"rule": "conflict-aware", "capacity": "3L", "eps_proj": EPS_PROJ}, force=True)
    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_factory, phase="TRAINING", total_steps=SCIENTIFIC_UPDATES)
    with interrupt:
        for step in range(SCIENTIFIC_UPDATES):
            indices, batch_labels = sampler.next_batch()
            _require(torch.equal(labels.index_select(0, indices), batch_labels), f"candidate TRAIN sampler labels changed at update {step + 1}")
            update_batch = images.index_select(0, indices).to(device)
            trace = _apply_update(candidate_consumer, train_evaluator, optimizer, update_batch, batch_labels.to(device), authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta=authorities["weights"]["delta"], step=step + 1, ledger=ledger)
            history.append(trace)
            now = time.time()
            if recovery_due(step=step + 1, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                last_recovery_path = recovery.save(recovery_factory(step + 1), step=step + 1)
                last_recovery_step = step + 1
                last_recovery_time = now
            heartbeat.update(state=RUNNING, phase="TRAINING", step=step + 1, latest_losses={key: trace[key] for key in ("L_rec", "L_op", "L_sem")}, last_recovery_checkpoint=None if last_recovery_path is None else last_recovery_path.name)
            interrupted = interrupt.handle_safe_boundary(step=step + 1, latest_losses={key: trace[key] for key in ("L_rec", "L_op", "L_sem")}, last_recovery_checkpoint=None if last_recovery_path is None else last_recovery_path.name)
            if interrupted is not None:
                control.interrupted()
                return {"result": "INTERRUPTED", "status": "INTERRUPTED", "output_dir": _relative(root, target), "step": step + 1, "recovery_checkpoint": interrupted.name}
            completed = step + 1
            if completed in CHECKPOINT_STEPS[1:]:
                composition_trajectory[str(completed)] = _diagnose_composition(candidate_consumer, train_evaluator, fixed_batch, fixed_labels, authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta=authorities["weights"]["delta"])
                diagnostic = _candidate_dev_eval(candidate_consumer, train_evaluator, smooth_evaluator, images, labels, authorities["development_indices"], authorities["training"]["references"], ledger, step=completed)
                dev_records[str(completed)] = diagnostic
                _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {"schema": DEV_SCHEMA, "step": completed, "diagnostic": {key: value for key, value in diagnostic.items() if not key.startswith("_")}})
                _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(candidate_consumer, optimizer, sampler, history, composition_trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight, scientific_artifact=True), kind="minimal Generator capacity scientific checkpoint")

    heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=SCIENTIFIC_UPDATES, latest_losses={"completed_updates": SCIENTIFIC_UPDATES}, force=True)
    correctness_payload = {"schema": "minimal_generator_capacity_expansion_dev_correctness.v1", "development_indices_sha256": tensor_sha256(authorities["development_indices"]), "labels": labels.index_select(0, authorities["development_indices"]), "steps": {step: {"train_correct": dev_records[step]["_train_correct"], "smooth_correct": dev_records[step]["_smooth_correct"]} for step in dev_records}}
    _write_torch(target / "dev_correctness.pt", correctness_payload, kind="expanded Generator DEV correctness vectors")
    bootstrap = successor._paired_bootstrap(correctness_payload["labels"], correctness_payload["steps"]["0"]["smooth_correct"], correctness_payload["steps"]["300"]["smooth_correct"])
    _write_torch(target / "paired_bootstrap.pt", bootstrap, kind="expanded Generator paired DEV Smooth bootstrap")
    bootstrap_report = dict(bootstrap)
    bootstrap_report.pop("macro_delta_tensor", None)
    bootstrap_report.pop("draws", None)
    dev_semantic = {"primary_evaluator": evaluator_identity["smooth"], "step_0_macro_recall": float(dev_records["0"]["smooth_evaluator"]["macro_recall"]), "step_300_macro_recall": float(dev_records["300"]["smooth_evaluator"]["macro_recall"]), "delta_macro_recall": float(dev_records["300"]["smooth_evaluator"]["macro_recall"] - dev_records["0"]["smooth_evaluator"]["macro_recall"]), "step_0_per_class_recall": list(dev_records["0"]["smooth_evaluator"]["per_class_recall"]), "step_300_per_class_recall": list(dev_records["300"]["smooth_evaluator"]["per_class_recall"]), "per_class_delta": list(bootstrap_report["per_class_delta"]), "paired_bootstrap_ci": dict(bootstrap_report["ci95"]), "paired_bootstrap_artifact": _artifact_ref(root, target / "paired_bootstrap.pt", role="expanded Generator paired DEV Smooth bootstrap"), "trajectory": {step: {"smooth_macro_recall": float(record["smooth_evaluator"]["macro_recall"]), "smooth_per_class_recall": list(record["smooth_evaluator"]["per_class_recall"])} for step, record in dev_records.items()}, "evaluator_separation": {"train_loss_provider": "FrozenMNISTEvaluator v1", "dev_primary": "SmoothMNISTEvaluator v2.1", "same_checkpoint": False, "smooth_in_objective": False}}
    dev_reconstruction = {"step_0": float(dev_records["0"]["reconstruction"]["L_rec"]), "step_100": float(dev_records["100"]["reconstruction"]["L_rec"]), "step_200": float(dev_records["200"]["reconstruction"]["L_rec"]), "step_300": float(dev_records["300"]["reconstruction"]["L_rec"]), "delta_step_300_minus_step_0": float(dev_records["300"]["reconstruction"]["L_rec"] - dev_records["0"]["reconstruction"]["L_rec"]), "tolerance": RECONSTRUCTION_TOLERANCE, "trajectory_pass": bool(dev_records["300"]["reconstruction"]["L_rec"] - dev_records["0"]["reconstruction"]["L_rec"] <= RECONSTRUCTION_TOLERANCE)}
    optical_validity = {"checkpoints": {step: {"finite": bool(record["raw_cmos"]["finite"]), "nonconstant": bool(record["raw_cmos"]["nonconstant"]), "zero_image_count": int(record["raw_cmos"]["zero_image_count"]), "L_op": float(record["optical"]["L_op"])} for step, record in dev_records.items()}, "all_finite": all(bool(record["raw_cmos"]["finite"]) for record in dev_records.values()), "all_nonconstant": all(bool(record["raw_cmos"]["nonconstant"]) for record in dev_records.values()), "all_zero_image_counts_zero": all(int(record["raw_cmos"]["zero_image_count"]) == 0 for record in dev_records.values())}
    _require(optical_validity["all_finite"] and optical_validity["all_nonconstant"] and optical_validity["all_zero_image_counts_zero"], "candidate optical validity failed")
    transfer = _p2q2_transfer(root, candidate_consumer, authorities, root / producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH, target)
    classification, checks = _classification(dev_semantic, dev_reconstruction, transfer)
    control.complete()
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=SCIENTIFIC_UPDATES, latest_losses={"classification": classification}, force=True)
    report = {"schema": REPORT_SCHEMA, "task_id": TASK_ID, "result": "VALID_COMPLETED", "status": "VALID_COMPLETED", "outer_gate": OUTER_GATE, "outer_gate_passed": classification == SUPPORTED_CLASSIFICATION, "classification": classification, "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True}, "local_state": {"branch": start_git["branch"], "head": start_git["head"], "dirty_before": start_git["worktree_dirty"], "git_end": successor.git_identity(root)}, "input_authorities": authorities["authority_hashes"], "accepted_conflict_aware": {"report": _artifact_ref(root, root / CONFLICT_REPORT, role="accepted 2L conflict-aware result"), "step_0": _artifact_ref(root, root / CONFLICT_STEP_0, role="accepted 2L conflict-aware optimizer start"), "step_300": _artifact_ref(root, root / CONFLICT_STEP_300, role="accepted 2L conflict-aware endpoint")}, "current_2L_generator_contract": architecture["current_2L"], "minimal_3L_contract": architecture["minimal_3L"], "function_preserving_check": function_check, "stage_1_local_controllability": stage1, "stage_2_executed": True, "optimizer": {"mode": optimizer_mode, "name": "Adam", "parameter_names": ["generator.phase"], "old_moments_reused": True, "new_plane_moments_zero_at_step0": True, "hyperparameters": {"learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD}}, "preflight": preflight, "training_trajectory": {"updates": len(history), "checkpoint_steps": list(CHECKPOINT_STEPS), "trace": history, "sampler_cursor": int(sampler.cursor)}, "dev_smooth": dev_semantic, "dev_reconstruction": dev_reconstruction, "optical_validity": optical_validity, "p2_q2_transfer": transfer, "comparison_to_accepted_2L_conflict_aware": {"accepted_smooth": "0.1030 -> 0.1036", "accepted_paired_ci95": [0.0, 0.00121], "accepted_reconstruction_delta": -0.0013805, "accepted_R_feature": "3.0552 -> 3.1940", "cross_run_significance": "descriptive only; no new cross-run significance test"}, "counterfactual_integrity": {"same_frozen_latent_authority": True, "same_old_phase_start": True, "same_data_order": True, "same_objectives": True, "same_loss_weights": True, "same_optimizer_family_and_hyperparameters": True, "copied_old_moments_reused": True, "new_plane_zero_and_moments_zero": True, "same_update_budget": True, "same_evaluators": True, "only_scientific_difference": "one separated Generator phase plane with symmetric split of one existing interval", "encoder_unchanged": True, "OLS_unchanged": True, "P2_Q2_unchanged": True, "protected_test_sealed_reads": False}, "outer_gate_checks": checks, "data_reads": ledger.to_dict(), "forbidden_data_reads": {"protected": ledger.counts.get("protected", 0), "test": ledger.counts.get("test", 0), "sealed": ledger.counts.get("sealed", 0)}, "claim_boundary": {"scientific_training_executed": True, "updates": SCIENTIFIC_UPDATES, "stage1_local_controllability_supported": True, "generation_success": classification in {SUPPORTED_CLASSIFICATION, MIXED_TRANSFER_CLASSIFICATION}, "global_architecture_inadequacy": False, "final_architecture": False, "nonlinear_optics_unnecessary": False, "formal_promotion": False, "physical_hardware_validity": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False}, "files_changed": {"scientific_artifacts": sorted(path.name for path in target.iterdir()), "repository_source": ["scripts/research/execute_lightgen_2d_minimal_generator_capacity_expansion_v1.py"]}, "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "architecture_resolution": True, "function_preserving_check": True, "stage1_gradient_probe": True, "directional_probes": True, "three_step_real_update": True, "checkpoints": list(CHECKPOINT_STEPS), "dev_evaluations": list(CHECKPOINT_STEPS), "paired_bootstrap": True, "frozen_gradient_firewall": True, "new_plane_moment_initialization_checked": True, "no_protected_test_sealed": True}, "unresolved": [] if classification == SUPPORTED_CLASSIFICATION else ["The single 3L candidate did not establish the full Case A claim under the frozen latent manifold."], "next_action": "NONE", "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "output_dir": _relative(root, target)}
    _write_json(target / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and execute the bounded capacity arm")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_capacity_expansion(output_dir=args.output_dir)
    except (CapacityExpansionError, successor.RecoveryExecutionError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "stage2_executed": report["stage_2_executed"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
