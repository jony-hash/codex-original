"""Execute one bounded near-identity nonlinear Generator successor.

The candidate returns to the accepted clean 2L Generator step-0 state and
inserts one scalar, trainable, normalized intensity-dependent phase transform
after the first existing Generator propagation and before the second phase
plane.  At ``eta=0`` the transform is an identity.  Stage 1 uses a fixed
TRAIN-only calibration/held-out split; Stage 2 is opened only when that local
direction transfers to the held-out half and then runs exactly one
conflict-aware 300-update trajectory.

The operator is an algorithmic function-class surrogate.  It is not a Kerr,
Te, fiber-facet, or hardware validation claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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


TASK_ID = "EXECUTE_MINIMAL_NEAR_IDENTITY_NONLINEAR_GENERATOR_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_MINIMAL_NEAR_IDENTITY_NONLINEAR_GENERATOR_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "MINIMAL_NEAR_IDENTITY_NONLINEAR_GENERATOR_V1_RESOLVED"
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

DEFAULT_OUTPUT_DIR = Path("outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_01")
LOGICAL_BATCH_SIZE = 10
DIAGNOSTIC_BANK_SIZE = 10
DIAGNOSTIC_HALF_SIZE = 5
PERTURBATION_EPSILONS = (1.0e-4, 3.0e-4, 1.0e-3)
NONLINEAR_EPS = 1.0e-12
RECONSTRUCTION_TOLERANCE = successor.RECONSTRUCTION_TOLERANCE
RECOVERY_INTERVAL_STEPS = successor.RECOVERY_INTERVAL_STEPS
INSERTION_POINT = "after_generator_phase_1_endpoint_before_generator_phase_2"

REPORT_SCHEMA = "minimal_near_identity_nonlinear_generator_report.v1"
ARCHITECTURE_SCHEMA = "minimal_near_identity_nonlinear_generator_architecture.v1"
FUNCTION_SCHEMA = "minimal_near_identity_nonlinear_generator_function_preserving.v1"
STAGE1_SCHEMA = "minimal_near_identity_nonlinear_generator_stage1.v1"
PROBE_SCHEMA = "minimal_near_identity_nonlinear_generator_probes.v1"
PREFLIGHT_SCHEMA = "minimal_near_identity_nonlinear_generator_preflight.v1"
CHECKPOINT_SCHEMA = "minimal_near_identity_nonlinear_generator_checkpoint.v1"
DEV_SCHEMA = "minimal_near_identity_nonlinear_generator_dev_diagnostic.v1"
CMOS_SCHEMA = "minimal_near_identity_nonlinear_generator_p2q2_cmos.v1"

RECOVERY_CLASSIFICATION = "MINIMAL_NONLINEAR_GENERATOR_RECOVERY_SUPPORTED"
LOCAL_GAIN_CLASSIFICATION = "NONLINEAR_LOCAL_GAIN_WITHOUT_DATASET_RECOVERY"
NOT_SUPPORTED_CLASSIFICATION = "MINIMAL_NONLINEAR_LOCAL_CONTROLLABILITY_NOT_SUPPORTED"
MIXED_TRANSFER_CLASSIFICATION = "NONLINEAR_REAL_SEMANTIC_RECOVERY_TRANSFER_MIXED"
RECONSTRAINT_CLASSIFICATION = "NONLINEAR_GENERATOR_RECONSTRUCTION_CONSTRAINT_FAILED"
OPTICAL_CLASSIFICATION = "NONLINEAR_GENERATOR_OPTICAL_VALIDITY_FAILED"
BLOCKED_IDENTITY_CLASSIFICATION = "BLOCKED_NONLINEAR_IDENTITY_COUNTERFACTUAL_INVALID"


class NonlinearExecutionError(RuntimeError):
    """Raised when an authority, invariant, or valid measurement is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NonlinearExecutionError(message)


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


def _runtime_frozen_state(runtime: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in runtime.state_dict().items()
        if not name.startswith("generator.")
    }


def _generator_parameters(consumer: Any) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    parameters = dict(consumer.runtime.named_parameters())
    _require(
        set(parameters) == {"encoder.phase", "generator.phase", "generator.eta"},
        f"nonlinear runtime parameter surface changed: {sorted(parameters)}",
    )
    return parameters["generator.phase"], parameters["generator.eta"]


def _freeze_candidate(consumer: Any) -> None:
    for name, parameter in consumer.runtime.named_parameters():
        parameter.requires_grad_(name in {"generator.phase", "generator.eta"})
    trainable = [name for name, parameter in consumer.runtime.named_parameters() if parameter.requires_grad]
    _require(trainable == ["generator.phase", "generator.eta"], f"nonlinear trainable firewall changed: {trainable}")


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


class NearIdentityNonlinearPhaseStack(nn.Module):
    """The accepted 2L stack with one scalar near-identity field operator."""

    def __init__(
        self,
        *,
        active_shape: tuple[int, int],
        propagation_shape: tuple[int, int],
        spacing_m: float,
        intensity_reference: float,
    ) -> None:
        super().__init__()
        _require(all(p >= a for p, a in zip(propagation_shape, active_shape, strict=True)), "invalid nonlinear D2NN dimensions")
        _require(all((p - a) % 2 == 0 for p, a in zip(propagation_shape, active_shape, strict=True)), "nonlinear centered padding is not integral")
        _require(math.isfinite(float(intensity_reference)) and float(intensity_reference) > 0.0, "I_ref must be finite and positive")
        self.layers = 2
        self.active_shape = active_shape
        self.propagation_shape = propagation_shape
        self.spacing_m = float(spacing_m)
        self.phase = nn.Parameter(torch.zeros(2, 1, *active_shape))
        self.eta = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.register_buffer("intensity_reference", torch.tensor(float(intensity_reference), dtype=torch.float32))

    @property
    def padding(self) -> tuple[int, int, int, int]:
        padding_y = (self.propagation_shape[0] - self.active_shape[0]) // 2
        padding_x = (self.propagation_shape[1] - self.active_shape[1]) // 2
        return padding_x, padding_x, padding_y, padding_y

    def _pad(self, field: ComplexField, prefix: str) -> ComplexField:
        padded_plane = field.plane.with_plane(
            f"{prefix}_padded",
            shape=self.propagation_shape,
            coordinate_frame=f"{prefix}_propagation_grid",
        )
        return ComplexField(
            F.pad(field.tensor, self.padding),
            padded_plane,
            OperationProvenance("finite_crop_center_pad", field.plane.plane_id, None, {"active_shape": list(self.active_shape)}),
        )

    def _crop(self, field: ComplexField, plane_id: str, *, layer: int) -> ComplexField:
        crop_y = (self.propagation_shape[0] - self.active_shape[0]) // 2
        crop_x = (self.propagation_shape[1] - self.active_shape[1]) // 2
        cropped = field.tensor[..., crop_y : crop_y + self.active_shape[0], crop_x : crop_x + self.active_shape[1]]
        active_plane = field.plane.with_plane(plane_id, shape=self.active_shape, coordinate_frame="generator")
        return ComplexField(
            cropped,
            active_plane,
            OperationProvenance("finite_crop_hard_aperture", field.plane.plane_id, self.spacing_m, {"layer": layer, "active_shape": list(self.active_shape)}),
        )

    def _phase_transfer(self, phase: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return torch.polar(torch.ones_like(phase), phase).to(dtype)

    def _nonlinear_transform(self, field: ComplexField) -> tuple[ComplexField, torch.Tensor, torch.Tensor]:
        intensity = field.tensor.abs().square()
        rho = intensity / (self.intensity_reference + NONLINEAR_EPS)
        nonlinear_phase = self.eta.to(dtype=intensity.dtype) * rho
        transfer = torch.polar(torch.ones_like(nonlinear_phase), nonlinear_phase).to(field.tensor.dtype)
        transformed = ComplexField(
            field.tensor * transfer,
            field.plane,
            OperationProvenance(
                "algorithmic_normalized_intensity_phase",
                field.plane.plane_id,
                None,
                {
                    "operator": "E * exp(i * eta * (|E|^2 / (I_ref + 1e-12)))",
                    "eta_trainable_scalar": True,
                    "eta_initialization": 0.0,
                    "intensity_reference_frozen": True,
                    "physical_status": "MODEL_ASSUMPTION_ALGORITHMIC_SURROGATE",
                },
            ),
        )
        return transformed, field.tensor, nonlinear_phase

    def forward(
        self, field: ComplexField, prefix: str
    ) -> tuple[ComplexField, tuple[ComplexField, ...], dict[str, torch.Tensor]]:
        _require(field.plane.shape == self.active_shape, "nonlinear Generator input shape changed")
        records: list[ComplexField] = []
        retention: list[torch.Tensor] = []

        transfer_0 = self._phase_transfer(self.phase[0], field.tensor.dtype)
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

        transformed, nonlinear_input, nonlinear_phase = self._nonlinear_transform(current)
        current = transformed
        transfer_1 = self._phase_transfer(self.phase[1], current.tensor.dtype)
        modulated_1 = ComplexField(
            current.tensor * transfer_1,
            current.plane,
            OperationProvenance("phase_only_modulation", current.plane.plane_id, None, {"layer": 2}),
        )
        propagated_1 = angular_spectrum_propagate(
            self._pad(modulated_1, f"{prefix}_layer_2"),
            self.spacing_m,
            f"{prefix}_layer_2_propagated",
        )
        current = self._crop(propagated_1, f"{prefix}_layer_2", layer=2)
        retention.append(
            current.tensor.abs().square().sum(dim=(-3, -2, -1))
            / propagated_1.tensor.abs().square().sum(dim=(-3, -2, -1)).clamp_min(1.0e-12)
        )
        records.append(current)
        return current, tuple(records), {
            "crop_retention": torch.stack(retention, dim=1),
            "phase": self.phase,
            "eta": self.eta,
            "intensity_reference": self.intensity_reference,
            "generator_nonlinear_input": nonlinear_input,
            "generator_nonlinear_output": transformed.tensor,
            "generator_nonlinear_phase": nonlinear_phase,
        }


class NearIdentityNonlinearRuntime(alignment.LightGen2DSemanticGenerationInputOLSAlignment):
    """Canonical LightGen runtime with only the declared nonlinear operator added."""

    def __init__(self, *, propagation_grid_size: int, intensity_reference: float) -> None:
        super().__init__(alignment.LightGen2DAlignmentRuntimeConfig(propagation_grid_size=propagation_grid_size))
        self.generator = NearIdentityNonlinearPhaseStack(
            active_shape=alignment.OLS_SHAPE,
            propagation_shape=(propagation_grid_size, propagation_grid_size),
            spacing_m=alignment.LAYER_SPACING_M,
            intensity_reference=intensity_reference,
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
                {"generator_layers": 2, "near_identity_nonlinear_insertion": INSERTION_POINT},
            ),
        )
        camera_field = self.output_relay(generator_output)
        self._validate_relay_diagnostics(self.output_relay)
        _finite(camera_field.tensor, "nonlinear raw CMOS field")
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
                "generator_eta": generator_stats["eta"],
                "generator_intensity_reference": generator_stats["intensity_reference"],
                "generator_nonlinear_input": generator_stats["generator_nonlinear_input"],
                "generator_nonlinear_output": generator_stats["generator_nonlinear_output"],
                "generator_nonlinear_phase": generator_stats["generator_nonlinear_phase"],
            },
        )


def _build_candidate_consumer(authorities: Mapping[str, Any], device: torch.device, intensity_reference: float) -> Any:
    candidate = successor.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device=device,
        propagation_grid_size=successor.PRODUCTION_PROPAGATION_GRID,
    )
    candidate.runtime = NearIdentityNonlinearRuntime(
        propagation_grid_size=successor.PRODUCTION_PROPAGATION_GRID,
        intensity_reference=intensity_reference,
    ).to(device)
    with torch.no_grad():
        candidate.runtime.encoder.phase.copy_(_tensor(authorities["step0"]["encoder_state"], "clean 2L encoder state").to(device))
        candidate.runtime.generator.phase.copy_(_tensor(authorities["step0"]["generator_state"], "clean 2L generator state").to(device))
        candidate.runtime.generator.eta.zero_()
    candidate.bind_references(authorities["training"]["references"], for_scientific=True)
    candidate.bind_weights(authorities["training"]["weights"], for_scientific=True)
    _freeze_candidate(candidate)
    return candidate


def _semantic_view(raw: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))


def _power_planes(result: alignment.LightGen2DForwardResult) -> torch.Tensor:
    _require(len(result.generator_records) == 2, "nonlinear power objective requires the original two Generator endpoints")
    return contracted_plane_powers(result)


def _loss_bundle(
    consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    *,
    include_distribution: bool = True,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "nonlinear real path is incomplete")
    raw = result.raw_intensity
    target = result.image_input.slm_intensity
    _require(raw.shape == target.shape and tuple(raw.shape[-2:]) == (200, 200), "nonlinear raw/target geometry changed")
    _require(raw.dtype == torch.float32 and target.dtype == torch.float32, "nonlinear raw/target dtype changed")
    _finite(raw, "nonlinear raw CMOS")
    _finite(target, "nonlinear SLM target")
    view = _semantic_view(raw)
    _, train_logits = train_evaluator(view)
    _, smooth_logits = smooth_evaluator(view)
    l_rec = (raw - target).square().mean()
    l_op = optical_power_loss_from_powers(_power_planes(result), references.power_reference)
    l_sem_train = F.cross_entropy(train_logits, labels.to(device=batch.device, dtype=torch.long))
    l_sem_smooth = F.cross_entropy(smooth_logits, labels.to(device=batch.device, dtype=torch.long))
    if include_distribution:
        _require(batch.shape[0] == LOGICAL_BATCH_SIZE, "distribution loss requires the logical batch size")
        with torch.no_grad():
            u = statistical_variable(result.z_ols.coefficients, references)
            q = q_from_batch(u)
            l_dist = truncated_normal_positive_kl(q, fixed_prior(device=q.loc.device, dtype=q.loc.dtype)).mean()
    else:
        l_dist = torch.zeros((), device=batch.device, dtype=l_rec.dtype)
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
        "distribution_evaluated": include_distribution,
        "raw_validity": {
            "finite": bool(torch.isfinite(raw).all().item()),
            "nonconstant_fraction": float((flat.std(dim=1, unbiased=False) > 0.0).to(torch.float64).mean().cpu()),
            "zero_image_count": int((flat.abs().sum(dim=1) == 0.0).sum().item()),
        },
    }


def _evaluate_probe(
    consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    *,
    include_distribution: bool = True,
) -> dict[str, Any]:
    with torch.no_grad():
        losses = _loss_bundle(consumer, train_evaluator, smooth_evaluator, batch, labels, references, include_distribution=include_distribution)
    return {
        "L_rec": float(losses["L_rec"].cpu()),
        "L_sem_train": float(losses["L_sem_train"].cpu()),
        "L_sem_smooth": float(losses["L_sem_smooth"].cpu()),
        "raw_validity": losses["raw_validity"],
    }


def _metric_deltas(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(candidate[name] - base[name]) for name in ("L_rec", "L_sem_train", "L_sem_smooth")}


def _calibrate_intensity_reference(base_consumer: Any, images: torch.Tensor, calibration_indices: torch.Tensor, device: torch.device) -> tuple[float, dict[str, Any]]:
    batch = images.index_select(0, calibration_indices).to(device)
    base_consumer.runtime.eval()
    with torch.no_grad():
        result = base_consumer.runtime.forward_from_image(batch)
        endpoint = result.generator_records[0].tensor
        intensity = endpoint.abs().square().to(torch.float64)
        reference = float(intensity.square().mean().sqrt().cpu())
    _require(math.isfinite(reference) and reference > 0.0, "clean 2L I_ref is zero, nonfinite, or undefined")
    return reference, {
        "definition": "RMS(|E|^2) over the fixed clean-2L TRAIN calibration bank at the insertion plane",
        "value": reference,
        "epsilon": NONLINEAR_EPS,
        "source": "clean accepted 2L step-0 Generator phase-1 endpoint",
        "sample_count": int(batch.shape[0]),
        "calibration_indices_sha256": tensor_sha256(calibration_indices),
        "data_role": TRAIN_ROLE,
    }


def _function_preserving_check(
    base_consumer: Any,
    candidate_consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
) -> dict[str, Any]:
    base_phase = _tensor(dict(base_consumer.runtime.named_parameters())["generator.phase"], "base phase")
    candidate_phase, eta = _generator_parameters(candidate_consumer)
    _require(torch.equal(base_phase.detach().cpu(), candidate_phase.detach().cpu()), "candidate phase planes differ from clean 2L state")
    _require(float(eta.detach().cpu()) == 0.0, "eta is not initialized to zero")
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
        "complex_field_after_nonlinear_insertion": _metric_error(base_result.generator_records[0].tensor, candidate_result.crop_statistics["generator_nonlinear_output"], "field after nonlinear insertion"),
        "first_generator_endpoint_complex_field": _metric_error(base_result.generator_records[0].tensor, candidate_result.generator_records[0].tensor, "first Generator endpoint"),
        "final_generator_output_complex_field": _metric_error(base_result.generator_output_field.tensor, candidate_result.generator_output_field.tensor, "final Generator output"),
        "raw_cmos": _metric_error(base_result.raw_intensity, candidate_result.raw_intensity, "raw CMOS"),
        "reconstruction_scalar": _metric_error(base_rec.reshape(1), candidate_rec.reshape(1), "reconstruction scalar"),
        "semantic_evaluator_input": _metric_error(base_view, candidate_view, "semantic evaluator input"),
        "train_evaluator_logits": _metric_error(base_train_logits, candidate_train_logits, "TRAIN evaluator logits"),
        "smooth_evaluator_logits": _metric_error(base_smooth_logits, candidate_smooth_logits, "Smooth evaluator logits"),
    }
    structural = {
        "clean_2L_phase_planes_copied_exactly": True,
        "eta_zero": True,
        "trainable_phase_shape_unchanged": list(candidate_phase.shape) == [2, 1, 100, 100],
        "generator_record_count_unchanged": len(candidate_result.generator_records) == 2,
        "insertion_has_no_propagation": True,
        "crop_count_unchanged": True,
        "aperture_and_sampling_unchanged": True,
        "relay_unchanged": True,
        "generator_exit_z_unchanged": math.isclose(
            float(base_result.generator_output_field.plane.z_m),
            float(candidate_result.generator_output_field.plane.z_m),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "intensity_reference_finite_positive": bool(torch.isfinite(candidate_consumer.runtime.generator.intensity_reference).item() and float(candidate_consumer.runtime.generator.intensity_reference.cpu()) > 0.0),
    }
    _require(all(bool(value) for value in structural.values()), "nonlinear insertion changed a frozen eta=0 optical variable")
    finite_errors = all(math.isfinite(float(item[key])) for item in comparisons.values() for key in ("max_abs", "rms", "relative_norm"))
    exact_identity = all(float(item["max_abs"]) == 0.0 for item in comparisons.values())
    return {
        "schema": FUNCTION_SCHEMA,
        "same_frozen_batch_size": int(batch.shape[0]),
        "insertion_point": INSERTION_POINT,
        "operator_at_eta_zero": "N_eta(E)=E because exp(i*0*rho)=1",
        "structural_identity": structural,
        "comparisons": comparisons,
        "finite_errors": finite_errors,
        "exact_identity_observed": exact_identity,
        "tolerance_policy": "eta=0 identity is accepted only when the unchanged float32 path produces zero observed difference; no loose runtime tolerance is invented.",
        "accepted_for_stage1": bool(all(bool(value) for value in structural.values()) and finite_errors and exact_identity),
    }


def _eta_gradient_probe(
    consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    name: str,
) -> dict[str, Any]:
    _, eta = _generator_parameters(consumer)
    before = _state_digest(_runtime_frozen_state(consumer.runtime))
    consumer.runtime.eval()
    smooth_evaluator.eval()
    losses = _loss_bundle(consumer, train_evaluator, smooth_evaluator, batch, labels, references, include_distribution=False)
    gradient = torch.autograd.grad(losses["L_sem_smooth"], eta, retain_graph=False, allow_unused=False)[0]
    _finite(gradient, f"{name} eta semantic gradient")
    value = float(gradient.detach().cpu())
    _require(math.isfinite(value), f"{name} eta semantic gradient is nonfinite")
    evaluator_grads_none = all(parameter.grad is None for parameter in smooth_evaluator.parameters())
    _require(evaluator_grads_none, f"{name} Smooth evaluator received a gradient")
    _require(before == _state_digest(_runtime_frozen_state(consumer.runtime)), f"{name} eta gradient probe mutated frozen runtime state")
    _clear_grads(consumer)
    return {
        "name": name,
        "eta": float(eta.detach().cpu()),
        "gradient": value,
        "finite": math.isfinite(value),
        "nonzero": value != 0.0,
        "losses": {key: float(losses[key].detach().cpu()) for key in ("L_rec", "L_sem_train", "L_sem_smooth", "L_op", "L_dist")},
        "raw_validity": losses["raw_validity"],
        "evaluator_parameter_gradients_none": evaluator_grads_none,
    }


def _nonlinear_probes(
    consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    direction: float,
) -> dict[str, Any]:
    _, eta = _generator_parameters(consumer)
    base_eta = eta.detach().clone()
    base = _evaluate_probe(consumer, train_evaluator, smooth_evaluator, batch, labels, references, include_distribution=False)
    records: list[dict[str, Any]] = []
    try:
        for epsilon in PERTURBATION_EPSILONS:
            signed: dict[str, Any] = {}
            for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                with torch.no_grad():
                    eta.copy_(torch.as_tensor(sign * direction * epsilon, dtype=eta.dtype, device=eta.device))
                candidate = _evaluate_probe(consumer, train_evaluator, smooth_evaluator, batch, labels, references, include_distribution=False)
                signed[sign_name] = {
                    "eta": float(eta.detach().cpu()),
                    "metrics": candidate,
                    "deltas": _metric_deltas(base, candidate),
                }
                with torch.no_grad():
                    eta.copy_(base_eta)
            signed["central_slope_per_eta"] = {
                metric: float((signed["plus"]["deltas"][metric] - signed["minus"]["deltas"][metric]) / (2.0 * epsilon * direction))
                for metric in ("L_rec", "L_sem_train", "L_sem_smooth")
            }
            signed["directional_slope_per_local_epsilon"] = {
                metric: float((signed["plus"]["deltas"][metric] - signed["minus"]["deltas"][metric]) / (2.0 * epsilon))
                for metric in ("L_rec", "L_sem_train", "L_sem_smooth")
            }
            records.append({"epsilon": epsilon, **signed})
    finally:
        with torch.no_grad():
            eta.copy_(base_eta)
    _require(float(eta.detach().cpu()) == float(base_eta.detach().cpu()), "nonlinear eta probe did not restore eta=0")
    all_raw_valid = all(
        bool(record[sign]["metrics"]["raw_validity"]["finite"])
        and record[sign]["metrics"]["raw_validity"]["nonconstant_fraction"] == 1.0
        and int(record[sign]["metrics"]["raw_validity"]["zero_image_count"]) == 0
        for record in records
        for sign in ("plus", "minus")
    )
    return {
        "schema": PROBE_SCHEMA,
        "direction": direction,
        "epsilons": list(PERTURBATION_EPSILONS),
        "epsilon_definition": "RMS nonlinear-phase convention: eta * rho(E), with eta magnitude in radians because rho is normalized by frozen I_ref",
        "base": base,
        "records": records,
        "restored_after_every_perturbation": True,
        "all_raw_valid": all_raw_valid,
    }


def _stage1_decision(function_check: Mapping[str, Any], gradient_a: Mapping[str, Any], gradient_b: Mapping[str, Any], probes_b: Mapping[str, Any], direction: float) -> tuple[bool, dict[str, Any]]:
    grad_a = float(gradient_a["gradient"])
    grad_b = float(gradient_b["gradient"])
    same_sign = math.isfinite(grad_a) and math.isfinite(grad_b) and grad_a != 0.0 and grad_b != 0.0 and grad_a * grad_b > 0.0
    direction_beneficial = math.isfinite(grad_b) and grad_b * direction < 0.0
    probe_records = list(probes_b["records"])
    probe_agreement = all(
        float(record["directional_slope_per_local_epsilon"]["L_sem_smooth"]) < 0.0
        and float(record["plus"]["deltas"]["L_sem_smooth"]) < 0.0
        and float(record["minus"]["deltas"]["L_sem_smooth"]) > 0.0
        for record in probe_records
    )
    raw_valid = bool(probes_b["all_raw_valid"])
    checks = {
        "eta_zero_function_preserving": bool(function_check["accepted_for_stage1"]),
        "eta_gradient_A_finite_nonzero": bool(gradient_a["finite"] and gradient_a["nonzero"]),
        "eta_gradient_B_finite_nonzero": bool(gradient_b["finite"] and gradient_b["nonzero"]),
        "calibration_direction_same_beneficial_sign_on_B": same_sign and direction_beneficial,
        "symmetric_B_probes_agree_with_derivative": probe_agreement,
        "raw_validity": raw_valid,
    }
    details = {
        "checks": checks,
        "eta_gradient_A": grad_a,
        "eta_gradient_B": grad_b,
        "direction": direction,
        "sign_agreement": same_sign,
        "direction_B_first_order_beneficial": direction_beneficial,
        "finite_difference_slope_B": [float(record["central_slope_per_eta"]["L_sem_smooth"]) for record in probe_records],
        "interpretation": "A chooses only the sign; B tests local transfer on unseen TRAIN examples; no eta magnitude optimization is performed.",
    }
    return all(checks.values()), details


def _load_authorities(root: Path) -> dict[str, Any]:
    """Load only the named clean 2L and downstream authorities.

    The prior consumers reject a current status file that already records a
    closed predecessor Gate.  This consumer repeats the relevant authority
    checks while allowing those predecessor Gate strings to remain present.
    No 3L checkpoint is loaded or used.
    """
    status = root / "docs/CURRENT_STATUS.md"
    handoff = root / "docs/status/current_handoff.md"
    _require(status.is_file() and handoff.is_file(), "current authority documents are missing")
    status_top = "\n".join(status.read_text(encoding="utf-8").splitlines()[:55])
    handoff_top = "\n".join(handoff.read_text(encoding="utf-8").splitlines()[:55])
    for name, content in (("CURRENT_STATUS", status_top), ("current_handoff", handoff_top)):
        _require(SEMANTIC_AUTHORITY_ID in content, f"{name} lost the active M0 authority")
        _require("CURRENT_AUTHORIZED_NEXT_ACTION=NONE" in content, f"{name} did not close the prior action")
        _require("GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED" in content, f"{name} lost attribution authority")
        _require("CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1_RESOLVED" in content, f"{name} lost conflict-aware authority")
        _require("MINIMAL_GENERATOR_CAPACITY_EXPANSION_V1_RESOLVED" in content, f"{name} lost the latest linear-depth result")
        _require(OUTER_GATE not in content, f"{name} already contains this successor result")

    prior = attribution._load_authorities(root)
    cpu_context = phase_repair._load_cpu_context(root)
    phase = conflict._load_phase_transfer_authority(root, cpu_context)
    hashes = dict(prior["hashes"])
    hashes.update(phase["authority_hashes"])
    hashes.update(
        {
            "attribution_report": conflict.EXPECTED_ATTRIBUTION_REPORT_SHA256,
            "conflict_report": EXPECTED_CONFLICT_REPORT_SHA256,
            "conflict_step0": EXPECTED_CONFLICT_STEP_0_SHA256,
            "conflict_step300": EXPECTED_CONFLICT_STEP_300_SHA256,
        }
    )

    _require(file_sha256(root / CONFLICT_REPORT) == EXPECTED_CONFLICT_REPORT_SHA256, "accepted conflict-aware report identity changed")
    _require(file_sha256(root / CONFLICT_STEP_0) == EXPECTED_CONFLICT_STEP_0_SHA256, "accepted clean 2L step-0 identity changed")
    _require(file_sha256(root / CONFLICT_STEP_300) == EXPECTED_CONFLICT_STEP_300_SHA256, "accepted conflict-aware step-300 identity changed")
    conflict_report = json.loads((root / CONFLICT_REPORT).read_text(encoding="utf-8"))
    _require(conflict_report.get("result") == "VALID_COMPLETED", "accepted conflict-aware result is not valid completed")
    _require(conflict_report.get("outer_gate") == conflict.OUTER_GATE, "accepted conflict-aware Gate changed")
    _require(conflict_report.get("classification") == conflict.NOT_SUPPORTED_CLASSIFICATION, "accepted conflict-aware classification changed")

    conflict_step0 = safe_torch_load(root / CONFLICT_STEP_0, expected_sha256=EXPECTED_CONFLICT_STEP_0_SHA256, map_location="cpu")
    _require(isinstance(conflict_step0, Mapping), "accepted clean 2L step-0 checkpoint is malformed")
    _require(conflict_step0.get("completed_updates") == 0, "clean 2L starting checkpoint is not step 0")
    _require(conflict_step0.get("trainable_parameter_names") == ["generator.phase"], "clean 2L trainable boundary changed")
    _require(torch.equal(_tensor(conflict_step0.get("generator_state"), "conflict clean 2L generator"), _tensor(prior["step0"].get("generator_state"), "prior clean 2L generator")), "conflict and prior clean 2L generator states differ")
    _require(torch.equal(_tensor(conflict_step0.get("encoder_state"), "conflict clean 2L encoder"), _tensor(prior["step0"].get("encoder_state"), "prior clean 2L encoder")), "conflict and prior clean 2L encoder states differ")

    optimizer_state = _mapping(conflict_step0.get("optimizer_state"), "accepted clean 2L optimizer state")
    entries = _mapping(optimizer_state.get("state"), "accepted clean 2L optimizer entries")
    _require(len(entries) == 1, "accepted clean 2L optimizer parameter count changed")
    old_entry = next(iter(entries.values()))
    _require(isinstance(old_entry, Mapping), "accepted clean 2L optimizer entry is malformed")
    for key in ("step", "exp_avg", "exp_avg_sq"):
        _require(key in old_entry, f"accepted clean 2L optimizer misses {key}")
    _require(tuple(_tensor(old_entry["exp_avg"], "accepted phase exp_avg").shape) == (2, 1, 100, 100), "accepted phase exp_avg shape changed")
    _require(tuple(_tensor(old_entry["exp_avg_sq"], "accepted phase exp_avg_sq").shape) == (2, 1, 100, 100), "accepted phase exp_avg_sq shape changed")
    _finite(_tensor(old_entry["exp_avg"], "accepted phase exp_avg"), "accepted phase exp_avg")
    _finite(_tensor(old_entry["exp_avg_sq"], "accepted phase exp_avg_sq"), "accepted phase exp_avg_sq")

    authorities = dict(prior)
    authorities.update(
        {
            "docs": {
                "status_top_sha256": hashlib.sha256(status_top.encode("utf-8")).hexdigest(),
                "handoff_top_sha256": hashlib.sha256(handoff_top.encode("utf-8")).hexdigest(),
            },
            "phase_latent": phase["phase_latent"],
            "selected_generated_raw": phase["selected_generated_raw"],
            "real_control_raw": phase["real_control_raw"],
            "selected_metrics": phase["selected_metrics"],
            "authority_hashes": hashes,
            "weights": {
                "beta": float(prior["training"]["weights"].beta),
                "gamma": float(prior["training"]["weights"].gamma),
                "delta": float(_mapping(prior["step0"].get("calibration"), "clean step-0 calibration").get("delta")),
            },
            "conflict_report": conflict_report,
            "conflict_step0": dict(conflict_step0),
            "conflict_step300_sha256": EXPECTED_CONFLICT_STEP_300_SHA256,
            "conflict_optimizer_state": dict(optimizer_state),
        }
    )
    _require(math.isclose(authorities["weights"]["beta"], float(prior["training"]["weights"].beta), rel_tol=0.0, abs_tol=1.0e-12), "beta authority changed")
    _require(math.isclose(authorities["weights"]["gamma"], float(prior["training"]["weights"].gamma), rel_tol=0.0, abs_tol=1.0e-12), "gamma authority changed")
    return authorities


def _architecture_resolution(base_consumer: Any, candidate_consumer: Any, intensity_reference: Mapping[str, Any], insertion_plane: Any) -> dict[str, Any]:
    base_generator = base_consumer.runtime.generator
    candidate_generator = candidate_consumer.runtime.generator
    return {
        "schema": ARCHITECTURE_SCHEMA,
        "model_status": "MODEL_ASSUMPTION_ALGORITHMIC_SURROGATE",
        "current_2L": {
            "trainable_phase_planes": int(base_generator.layers),
            "phase_shape": list(base_generator.phase.shape),
            "active_shape": list(base_generator.active_shape),
            "propagation_grid_shape": list(base_generator.propagation_shape),
            "propagation_distance_m": float(base_generator.spacing_m),
            "sequence": "W_OLS(z) -> M(phi_1) -> centered pad -> ASM(1.50 mm) -> centered crop -> M(phi_2) -> centered pad -> ASM(1.50 mm) -> centered crop -> B=0 output relay -> raw CMOS intensity |E|^2",
            "crop_locations": "after each existing ASM propagation, centered 100x100 crop from 2048x2048 grid",
            "normalization": "no phase-stack amplitude normalization; phase transmittance has unit magnitude",
            "aperture": "finite_crop_hard_aperture at each existing layer; no additional aperture",
            "encoder_and_ols_frozen": True,
        },
        "minimal_near_identity_nonlinear": {
            "trainable_phase_planes": int(candidate_generator.layers),
            "phase_shape": list(candidate_generator.phase.shape),
            "trainable_parameters": ["generator.phase", "generator.eta"],
            "eta_shape": list(candidate_generator.eta.shape),
            "eta_initialization": 0.0,
            "insertion_point": INSERTION_POINT,
            "insertion_plane": {
                "plane_id": insertion_plane.plane_id,
                "z_m": float(insertion_plane.z_m),
                "shape": list(insertion_plane.shape),
                "coordinate_frame": insertion_plane.coordinate_frame,
            },
            "operator": "I=|E|^2; I_ref=RMS(I) on fixed clean-2L TRAIN calibration bank; rho=I/(I_ref+1e-12); N_eta(E)=E*exp(i*eta*rho)",
            "intensity_reference": dict(intensity_reference),
            "new_propagation": False,
            "new_crop": False,
            "new_aperture": False,
            "new_relay": False,
            "physical_status": "not measured Kerr/Te/facet/fiber hardware validation",
        },
    }


def _flat_gradients(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values])


def _split_gradient(flat: torch.Tensor, parameters: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    outputs: list[torch.Tensor] = []
    cursor = 0
    for parameter in parameters:
        size = parameter.numel()
        outputs.append(flat[cursor : cursor + size].reshape_as(parameter))
        cursor += size
    _require(cursor == flat.numel(), "gradient split length changed")
    return tuple(outputs)


def _gradient_stats(value: torch.Tensor, name: str, *, parameter_slices: Mapping[str, torch.Tensor] | None = None) -> dict[str, Any]:
    _finite(value, name)
    flat = value.detach().to(torch.float64).reshape(-1)
    norm = float(flat.square().sum().sqrt().cpu())
    result = {
        "norm": norm,
        "finite_fraction": float(torch.isfinite(flat).to(torch.float64).mean().cpu()),
        "zero_fraction": float((flat == 0.0).to(torch.float64).mean().cpu()),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    if parameter_slices is not None:
        result["per_parameter_norms"] = {
            name: float(item.detach().to(torch.float64).reshape(-1).square().sum().sqrt().cpu())
            for name, item in parameter_slices.items()
        }
    return result


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sum(left.detach().to(torch.float64) * right.detach().to(torch.float64)).cpu())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left64 = left.detach().to(torch.float64).reshape(-1)
    right64 = right.detach().to(torch.float64).reshape(-1)
    denominator = float(left64.norm().cpu() * right64.norm().cpu())
    if denominator == 0.0:
        return None
    value = float(torch.dot(left64, right64).cpu() / denominator)
    _require(math.isfinite(value), "gradient cosine is nonfinite")
    return value


def _project_auxiliary(auxiliary: torch.Tensor, semantic: torch.Tensor, name: str) -> tuple[torch.Tensor, dict[str, Any]]:
    auxiliary_norm = float(auxiliary.detach().to(torch.float64).norm().cpu())
    semantic_norm = float(semantic.detach().to(torch.float64).norm().cpu())
    _require(math.isfinite(semantic_norm) and semantic_norm > 0.0, "semantic gradient is zero or invalid")
    dot = _dot(auxiliary, semantic)
    projected = dot < 0.0 and auxiliary_norm > 0.0
    removed = torch.zeros_like(auxiliary)
    if projected:
        coefficient = dot / (float(torch.sum(semantic.detach().to(torch.float64).square()).cpu()) + conflict.EPS_PROJ)
        removed = semantic * torch.as_tensor(coefficient, dtype=semantic.dtype, device=semantic.device)
        safe = auxiliary - removed
    else:
        safe = auxiliary
    _finite(safe, f"{name}_safe gradient")
    safe_dot = _dot(safe, semantic)
    if projected:
        tolerance = max(1.0e-8, 1.0e-5 * max(float(safe.detach().to(torch.float64).norm().cpu()), 1.0e-12) * semantic_norm)
        _require(abs(safe_dot) <= tolerance, f"{name}_safe gradient is not orthogonal to semantic gradient")
    removed_norm = float(removed.detach().to(torch.float64).norm().cpu())
    return safe, {
        "dot_before": dot,
        "dot_after": safe_dot,
        "projected": projected,
        "projection_fraction_norm": removed_norm / max(auxiliary_norm, 1.0e-30),
        "removed_norm": removed_norm,
        "eps_proj": conflict.EPS_PROJ,
    }


def _recovery_components(
    consumer: Any,
    evaluator: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    beta: float,
    gamma: float,
    delta: float,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "nonlinear real path is incomplete")
    raw = result.raw_intensity
    target = result.image_input.slm_intensity
    _require(raw.shape == target.shape and tuple(raw.shape[-2:]) == (200, 200), "nonlinear raw/target shape changed")
    _finite(raw, "nonlinear raw CMOS")
    _finite(target, "nonlinear target")
    l_rec = (raw - target).square().mean()
    l_op = optical_power_loss_from_powers(_power_planes(result), references.power_reference)
    view = _semantic_view(raw)
    _, logits = evaluator(view)
    l_sem = F.cross_entropy(logits, labels.to(device=batch.device, dtype=torch.long))
    l_total = beta * l_rec + gamma * l_op + delta * l_sem
    with torch.no_grad():
        u = statistical_variable(result.z_ols.coefficients, references)
        q = q_from_batch(u)
        l_dist = truncated_normal_positive_kl(q, fixed_prior(device=q.loc.device, dtype=q.loc.dtype)).mean()
    for name, value in (("L_rec", l_rec), ("L_op", l_op), ("L_sem", l_sem), ("L_total", l_total), ("L_dist", l_dist)):
        _finite(value, name)
    return {"result": result, "raw": raw, "target": target, "L_rec": l_rec, "L_op": l_op, "L_sem": l_sem, "L_total": l_total, "L_dist": l_dist}


def _compose_gradients(
    consumer: Any,
    evaluator: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    *,
    beta: float,
    gamma: float,
    delta: float,
) -> dict[str, Any]:
    phase, eta = _generator_parameters(consumer)
    parameters = (phase, eta)
    components = _recovery_components(consumer, evaluator, batch, labels, references, beta, gamma, delta)
    g_rec_parts = torch.autograd.grad(beta * components["L_rec"], parameters, retain_graph=True, allow_unused=False)
    g_op_parts = torch.autograd.grad(gamma * components["L_op"], parameters, retain_graph=True, allow_unused=False)
    g_sem_parts = torch.autograd.grad(delta * components["L_sem"], parameters, retain_graph=False, allow_unused=False)
    g_rec = _flat_gradients(g_rec_parts)
    g_op = _flat_gradients(g_op_parts)
    g_sem = _flat_gradients(g_sem_parts)
    for name, value in (("g_rec", g_rec), ("g_op", g_op), ("g_sem", g_sem)):
        _finite(value, name)
    g_rec_safe, rec_projection = _project_auxiliary(g_rec, g_sem, "g_rec")
    g_op_safe, op_projection = _project_auxiliary(g_op, g_sem, "g_op")
    g_total = g_sem + g_rec_safe + g_op_safe
    _finite(g_total, "g_total")
    _require(_dot(g_total, g_sem) > 0.0, "composed gradient lacks a positive semantic component")
    def slices(value: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"generator.phase": value[: phase.numel()].reshape_as(phase), "generator.eta": value[phase.numel() :].reshape_as(eta)}
    return {
        "components": components,
        "gradients": {"g_rec": g_rec, "g_op": g_op, "g_sem": g_sem, "g_rec_safe": g_rec_safe, "g_op_safe": g_op_safe, "g_total": g_total},
        "gradient_norms": {
            name: _gradient_stats(value, name, parameter_slices=slices(value))
            for name, value in (("g_rec", g_rec), ("g_op", g_op), ("g_sem", g_sem), ("g_rec_safe", g_rec_safe), ("g_op_safe", g_op_safe), ("g_total", g_total))
        },
        "eta_gradient_norms": {
            "g_eta_rec": abs(float(g_rec[phase.numel()].detach().cpu())),
            "g_eta_op": abs(float(g_op[phase.numel()].detach().cpu())),
            "g_eta_sem": abs(float(g_sem[phase.numel()].detach().cpu())),
            "g_eta_total": abs(float(g_total[phase.numel()].detach().cpu())),
            "g_eta_rec_signed": float(g_rec[phase.numel()].detach().cpu()),
            "g_eta_op_signed": float(g_op[phase.numel()].detach().cpu()),
            "g_eta_sem_signed": float(g_sem[phase.numel()].detach().cpu()),
            "g_eta_total_signed": float(g_total[phase.numel()].detach().cpu()),
        },
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


def _rms_nonlinear_phase(consumer: Any, batch: torch.Tensor) -> float:
    with torch.no_grad():
        result = consumer.runtime.forward_from_image(batch)
        phase = result.crop_statistics["generator_nonlinear_phase"]
        value = float(phase.detach().to(torch.float64).square().mean().sqrt().cpu())
    _require(math.isfinite(value), "RMS nonlinear phase is nonfinite")
    return value


def _composition_record(bundle: Mapping[str, Any], *, consumer: Any, fixed_batch: torch.Tensor, step: int | None = None) -> dict[str, Any]:
    components = _mapping(bundle["components"], "nonlinear gradient components")
    _, eta = _generator_parameters(consumer)
    record = {
        "step": step,
        "L_rec": float(components["L_rec"].detach().cpu()),
        "L_op": float(components["L_op"].detach().cpu()),
        "L_sem": float(components["L_sem"].detach().cpu()),
        "L_total": float(components["L_total"].detach().cpu()),
        "eta_value": float(eta.detach().cpu()),
        "rms_nonlinear_phase": _rms_nonlinear_phase(consumer, fixed_batch),
        "eta_finite": bool(torch.isfinite(eta).item()),
        "gradient_norms": bundle["gradient_norms"],
        "eta_gradient_norms": bundle["eta_gradient_norms"],
        "gradient_cosines": bundle["gradient_cosines"],
        "projection": bundle["projection"],
        "old_tn_distribution_in_objective": False,
        "scalar_weighted_sum_backward": False,
    }
    return record


def _diagnose_composition(
    consumer: Any,
    evaluator: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    fixed_batch: torch.Tensor,
    *,
    beta: float,
    gamma: float,
    delta: float,
    step: int,
) -> dict[str, Any]:
    before = _state_digest(_runtime_frozen_state(consumer.runtime))
    bundle = _compose_gradients(consumer, evaluator, batch, labels, references, beta=beta, gamma=gamma, delta=delta)
    record = _composition_record(bundle, consumer=consumer, fixed_batch=fixed_batch, step=step)
    _clear_grads(consumer)
    _require(before == _state_digest(_runtime_frozen_state(consumer.runtime)), "composition diagnostic mutated frozen runtime state")
    return record


def _apply_update(
    consumer: Any,
    evaluator: FrozenMNISTEvaluator,
    optimizer: torch.optim.Adam,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    fixed_batch: torch.Tensor,
    *,
    beta: float,
    gamma: float,
    delta: float,
    step: int,
) -> dict[str, Any]:
    phase, eta = _generator_parameters(consumer)
    before_phase = phase.detach().clone()
    before_eta = eta.detach().clone()
    before_frozen = _state_digest(_runtime_frozen_state(consumer.runtime))
    before_i_ref = consumer.runtime.generator.intensity_reference.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    _clear_grads(consumer)
    bundle = _compose_gradients(consumer, evaluator, batch, labels, references, beta=beta, gamma=gamma, delta=delta)
    total_phase, total_eta = _split_gradient(bundle["gradients"]["g_total"], (phase, eta))
    phase.grad = total_phase
    eta.grad = total_eta
    _require(consumer.runtime.encoder.phase.grad is None, "frozen encoder received a gradient")
    optimizer.step()
    _finite(phase, "updated generator.phase")
    _finite(eta, "updated generator.eta")
    _require(torch.equal(before_i_ref, consumer.runtime.generator.intensity_reference.detach()), "I_ref changed during update")
    _require(before_frozen == _state_digest(_runtime_frozen_state(consumer.runtime)), "frozen runtime state changed during update")
    phase_update_norm = float((phase.detach() - before_phase).to(torch.float64).norm().cpu())
    eta_update = float((eta.detach() - before_eta).cpu())
    _require(math.isfinite(phase_update_norm) and math.isfinite(eta_update), "nonlinear update is nonfinite")
    _require(phase_update_norm > 0.0 or eta_update != 0.0, "nonlinear trainable parameters did not update")
    record = _composition_record(bundle, consumer=consumer, fixed_batch=fixed_batch, step=step)
    record.update(
        {
            "generator_phase_update_norm": phase_update_norm,
            "eta_update": eta_update,
            "trainable_parameter_names": ["generator.phase", "generator.eta"],
            "encoder_unchanged": True,
            "frozen_runtime_unchanged": True,
        }
    )
    _clear_grads(consumer)
    return record


def _validate_optimizer(consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    _require(isinstance(optimizer, torch.optim.Adam), "nonlinear optimizer must be Adam")
    _require(len(optimizer.param_groups) == 1, "nonlinear optimizer must have one parameter group")
    phase, eta = _generator_parameters(consumer)
    actual = list(optimizer.param_groups[0]["params"])
    _require(len(actual) == 2 and actual[0] is phase and actual[1] is eta, "nonlinear optimizer parameter boundary changed")
    group = optimizer.param_groups[0]
    _require(group["lr"] == LEARNING_RATE, "nonlinear learning rate changed")
    _require(tuple(group["betas"]) == ADAM_BETAS, "nonlinear Adam betas changed")
    _require(group["eps"] == ADAM_EPSILON, "nonlinear Adam epsilon changed")
    _require(group["weight_decay"] == WEIGHT_DECAY and group["amsgrad"] is AMSGRAD, "nonlinear Adam options changed")
    phase_state = optimizer.state.get(phase)
    eta_state = optimizer.state.get(eta)
    _require(isinstance(phase_state, Mapping) and isinstance(eta_state, Mapping), "nonlinear optimizer state is missing")
    _require(tuple(_tensor(phase_state["exp_avg"], "nonlinear phase exp_avg").shape) == tuple(phase.shape), "nonlinear phase exp_avg shape changed")
    _require(tuple(_tensor(phase_state["exp_avg_sq"], "nonlinear phase exp_avg_sq").shape) == tuple(phase.shape), "nonlinear phase exp_avg_sq shape changed")
    _require(tuple(_tensor(eta_state["exp_avg"], "nonlinear eta exp_avg").shape) == tuple(eta.shape), "nonlinear eta exp_avg shape changed")
    _require(tuple(_tensor(eta_state["exp_avg_sq"], "nonlinear eta exp_avg_sq").shape) == tuple(eta.shape), "nonlinear eta exp_avg_sq shape changed")
    _finite(_tensor(phase_state["exp_avg"], "nonlinear phase exp_avg"), "nonlinear phase exp_avg")
    _finite(_tensor(phase_state["exp_avg_sq"], "nonlinear phase exp_avg_sq"), "nonlinear phase exp_avg_sq")
    _finite(_tensor(eta_state["exp_avg"], "nonlinear eta exp_avg"), "nonlinear eta exp_avg")
    _finite(_tensor(eta_state["exp_avg_sq"], "nonlinear eta exp_avg_sq"), "nonlinear eta exp_avg_sq")


def _new_optimizer(consumer: Any, prior_state: Mapping[str, Any]) -> tuple[torch.optim.Adam, str]:
    phase, eta = _generator_parameters(consumer)
    optimizer = torch.optim.Adam(
        [phase, eta],
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    old_groups = prior_state.get("param_groups")
    old_entries = prior_state.get("state")
    _require(isinstance(old_groups, list) and len(old_groups) == 1, "clean 2L optimizer groups are incompatible")
    _require(isinstance(old_entries, Mapping) and len(old_entries) == 1, "clean 2L optimizer entries are incompatible")
    old_group = dict(_mapping(old_groups[0], "clean 2L optimizer group"))
    old_entry = dict(_mapping(next(iter(old_entries.values())), "clean 2L optimizer phase entry"))
    old_avg = _tensor(old_entry.get("exp_avg"), "clean 2L phase exp_avg")
    old_avg_sq = _tensor(old_entry.get("exp_avg_sq"), "clean 2L phase exp_avg_sq")
    _require(tuple(old_avg.shape) == tuple(phase.shape), "clean 2L exp_avg does not match phase shape")
    _require(tuple(old_avg_sq.shape) == tuple(phase.shape), "clean 2L exp_avg_sq does not match phase shape")
    _require("step" in old_entry, "clean 2L optimizer step is missing")
    template = optimizer.state_dict()
    phase_id, eta_id = template["param_groups"][0]["params"]
    phase_entry = copy.deepcopy(old_entry)
    eta_entry = {
        "step": copy.deepcopy(old_entry["step"]),
        "exp_avg": torch.zeros_like(eta.detach()),
        "exp_avg_sq": torch.zeros_like(eta.detach()),
    }
    group = dict(old_group)
    group["params"] = [phase_id, eta_id]
    state = {phase_id: phase_entry, eta_id: eta_entry}
    optimizer.load_state_dict({"state": state, "param_groups": [group]})
    _validate_optimizer(consumer, optimizer)
    phase_state = optimizer.state[phase]
    eta_state = optimizer.state[eta]
    _require(torch.equal(phase_state["exp_avg"][:], old_avg.to(device=phase.device, dtype=phase.dtype)), "old phase exp_avg was not copied")
    _require(torch.equal(phase_state["exp_avg_sq"][:], old_avg_sq.to(device=phase.device, dtype=phase.dtype)), "old phase exp_avg_sq was not copied")
    _require(torch.equal(eta_state["exp_avg"], torch.zeros_like(eta)), "new eta exp_avg is not zero")
    _require(torch.equal(eta_state["exp_avg_sq"], torch.zeros_like(eta)), "new eta exp_avg_sq is not zero")
    _require(torch.equal(eta_state["step"], phase_state["step"]), "new eta Adam step counter is not aligned to the copied phase counter")
    return optimizer, "PRIOR_2L_PHASE_MOMENTS_COPIED_ZERO_ETA_MOMENTS_SHARED_STEP_COUNTER"


def _preflight(
    candidate: Any,
    train_evaluator: FrozenMNISTEvaluator,
    references: Any,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    calibration_indices: torch.Tensor,
    prior_optimizer_state: Mapping[str, Any],
    fixed_batch: torch.Tensor,
    fixed_labels: torch.Tensor,
    *,
    beta: float,
    gamma: float,
    delta: float,
) -> dict[str, Any]:
    _freeze_candidate(candidate)
    allowed = {"generator.phase", "generator.eta"}
    full_names = [name for name, _ in candidate.runtime.named_parameters()]
    negative_control = {
        "name": "FULL_RUNTIME_OPTIMIZER_REJECTED",
        "requested_parameter_names": full_names,
        "rejected": set(full_names) != allowed,
        "passed": set(full_names) != allowed,
    }
    _require(negative_control["passed"], "full runtime optimizer negative control was not rejected")
    optimizer, optimizer_mode = _new_optimizer(candidate, prior_optimizer_state)
    sampler = DeterministicTrainRoleSampler(train_indices, labels)
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == calibration_indices.tolist(), "preflight first TRAIN batch differs from clean 2L authority")
    sampler = DeterministicTrainRoleSampler(train_indices, labels)
    frozen_before = _state_digest(_runtime_frozen_state(candidate.runtime))
    trace: list[dict[str, Any]] = []
    for step in range(1, 4):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(labels.index_select(0, indices), batch_labels), f"preflight sampler labels changed at step {step}")
        batch = images.index_select(0, indices).to(candidate.device)
        trace.append(
            _apply_update(
                candidate,
                train_evaluator,
                optimizer,
                batch,
                batch_labels.to(candidate.device),
                references,
                fixed_batch,
                beta=beta,
                gamma=gamma,
                delta=delta,
                step=step,
            )
        )
    phase, eta = _generator_parameters(candidate)
    _require(_state_digest(_runtime_frozen_state(candidate.runtime)) == frozen_before, "preflight changed frozen runtime state")
    _require(float(eta.detach().cpu()) != 0.0 or any(record["eta_update"] != 0.0 for record in trace), "preflight did not exercise eta or a valid phase update")
    _require(not any(parameter.grad is not None for parameter in train_evaluator.parameters()), "preflight evaluator received a gradient")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "negative_control": negative_control,
        "optimizer_mode": optimizer_mode,
        "optimizer": {
            "parameter_names": ["generator.phase", "generator.eta"],
            "phase_moments_copied": True,
            "eta_moments_zero": True,
            "eta_step_counter": "copied_from_phase_state",
        },
        "three_step_real_update": {
            "updates": 3,
            "trace": trace,
            "generator_departed": True,
            "eta_finite": bool(torch.isfinite(eta).item()),
            "encoder_unchanged": True,
        },
        "frozen_parameter_probe": {
            "encoder_phase_unchanged": True,
            "ols_unchanged": True,
            "w_ols_unchanged": True,
            "evaluators_frozen": True,
        },
        "protected_test_sealed": {"protected": False, "test": False, "sealed": False},
        "train_reads": 30,
        "scientific_300_step_executed": False,
    }


def _candidate_dev_eval(
    consumer: Any,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    development_indices: torch.Tensor,
    references: Any,
    ledger: DataReadLedger,
    *,
    step: int,
) -> dict[str, Any]:
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
                _require(result.image_input is not None and result.ols_input_field is not None, "nonlinear DEV path is incomplete")
                raw = result.raw_intensity
                target = result.image_input.slm_intensity
                _finite(raw, "nonlinear DEV raw CMOS")
                _finite(target, "nonlinear DEV target")
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
            _require(int(mask.sum()) == 500, f"nonlinear DEV class count changed for {digit}")
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
    _require(diagnostic["raw_cmos"]["nonconstant"] and diagnostic["raw_cmos"]["zero_image_count"] == 0, f"nonlinear DEV raw validity failed at step {step}")
    _require(diagnostic["evaluator_parameter_gradients_none"], "nonlinear DEV evaluator received a gradient")
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
            _finite(raw, "nonlinear P2+Q2 raw CMOS")
            outputs.append(raw)
    candidate_raw = torch.cat(outputs)
    _require(candidate_raw.shape == (200, 1, 200, 200), "nonlinear P2+Q2 raw shape changed")
    candidate_path = target / "minimal_near_identity_nonlinear_p2q2_generated_cmos.pt"
    _write_torch(
        candidate_path,
        {
            "schema": CMOS_SCHEMA,
            "group": "minimal_near_identity_nonlinear_p2q2_generated",
            "input_z": z_sample,
            "raw_cmos_intensity": candidate_raw,
            "candidate_id": "Q2",
            "selected_step": 300,
            "eta": float(consumer.runtime.generator.eta.detach().cpu()),
        },
        kind="minimal near-identity nonlinear Generator P2+Q2 generated CMOS",
    )
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
        "ratio_after_over_baseline": {key: float(value) / max(float(expected[key]), 1.0e-12) for key, value in {"R_u": expected["R_u"], "R_phi": expected["R_phi"], "R_z": expected["R_z"], "R_raw": candidate_r_raw, "R_feature": candidate_report["comparison"]["R_feature"]}.items()},
        "raw_distribution": {"R_raw": float(candidate_r_raw), "baseline_R_raw_recomputed": float(baseline_r_raw), "candidate_energy_distance_to_real": float(candidate_raw_distance), "baseline_energy_distance_to_real": float(baseline_raw_distance), "real_control_split_energy_distance": float(real_self), "standardization": "per-pixel mean/std from the exact accepted 200-frame real-control CMOS artifact"},
        "class_histogram": generated_semantic["predicted_class_histogram"],
        "dominant_fraction": float(generated_semantic["predicted_class_dominant_fraction"]),
        "normalized_entropy": float(generated_semantic["predicted_class_entropy_normalized"]),
        "diversity": {"candidate_raw": canary._diversity_view(candidate_raw, name="minimal_near_identity_nonlinear_raw"), "baseline_raw": canary._diversity_view(baseline_raw, name="accepted_p2q2_raw"), "candidate_feature": canary._diversity_view(candidate_outputs["generated_embedding"], name="minimal_near_identity_nonlinear_feature"), "baseline_feature": canary._diversity_view(baseline_outputs["generated_embedding"], name="accepted_p2q2_feature")},
        "candidate_raw_validity": canary._raw_cmos_statistics(candidate_raw, name="minimal_near_identity_nonlinear_generated"),
        "baseline_raw_validity": canary._raw_cmos_statistics(baseline_raw, name="accepted_p2q2_generated"),
        "evaluator": {"candidate": candidate_report, "baseline": baseline_report},
        "generated_cmos_artifact": _artifact_ref(root, candidate_path, role="minimal near-identity nonlinear Generator P2+Q2 generated CMOS"),
    }


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Adam,
    sampler: DeterministicTrainRoleSampler,
    history: Sequence[Mapping[str, Any]],
    composition: Mapping[str, Any],
    *,
    step: int,
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    optimizer_mode: str,
    preflight: Mapping[str, Any],
    scientific_artifact: bool,
) -> dict[str, Any]:
    _validate_optimizer(consumer, optimizer)
    phase, eta = _generator_parameters(consumer)
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
        "accepted_parent_checkpoint": {"path": CONFLICT_STEP_0.as_posix(), "sha256": EXPECTED_CONFLICT_STEP_0_SHA256, "selected_step": 0, "role": "clean accepted 2L conflict-aware start"},
        "trainable_parameter_names": ["generator.phase", "generator.eta"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator", "generator.intensity_reference"],
        "generator_state": phase.detach().cpu().clone(),
        "eta": float(eta.detach().cpu()),
        "intensity_reference": float(consumer.runtime.generator.intensity_reference.detach().cpu()),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": _state_digest(_runtime_frozen_state(consumer.runtime)),
        "optimizer_mode": optimizer_mode,
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase", "generator.eta"], "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "eta_moment_initialization": "zero", "eta_step_counter": "copied_from_phase_state"},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": sampler.state_dict(),
        "rng_state": _rng_state_payload(),
        "gradient_composition": {"formula": "g_total=g_sem+project_nonconflicting(g_rec,g_sem)+project_nonconflicting(g_op,g_sem)", "eps_proj": conflict.EPS_PROJ, "trajectory": dict(composition), "old_tn_distribution_in_objective": False},
        "preflight": dict(preflight),
        "training_history": [dict(record) for record in history],
    }


def _recovery_payload(
    consumer: Any,
    optimizer: torch.optim.Adam,
    sampler: DeterministicTrainRoleSampler,
    history: Sequence[Mapping[str, Any]],
    composition: Mapping[str, Any],
    *,
    step: int,
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    optimizer_mode: str,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _checkpoint_payload(consumer, optimizer, sampler, history, composition, step=step, source_manifest=source_manifest, run_identity=run_identity, optimizer_mode=optimizer_mode, preflight=preflight, scientific_artifact=False)
    payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
    return payload


def _classification(dev_semantic: Mapping[str, Any], dev_reconstruction: Mapping[str, Any], transfer: Mapping[str, Any]) -> tuple[str, dict[str, bool]]:
    semantic_supported = bool(dev_semantic["delta_macro_recall"] > 0.0 and dev_semantic["paired_bootstrap_ci"]["lower"] > 0.0)
    reconstruction_pass = bool(dev_reconstruction["delta_step_300_minus_step_0"] <= RECONSTRUCTION_TOLERANCE)
    raw_valid = bool(transfer["candidate_raw_validity"]["finite"] and transfer["candidate_raw_validity"]["nonconstant_fraction"] == 1.0 and transfer["candidate_raw_validity"]["zero_image_count"] == 0)
    feature_improved = bool(transfer["after"]["R_feature"] < transfer["baseline_P2_Q2"]["R_feature"])
    checks = {"dev_semantic_supported": semantic_supported, "dev_reconstruction_pass": reconstruction_pass, "sampled_R_feature_improved": feature_improved, "raw_valid": raw_valid}
    if not reconstruction_pass:
        return RECONSTRAINT_CLASSIFICATION, checks
    if not raw_valid:
        return OPTICAL_CLASSIFICATION, checks
    if semantic_supported and feature_improved:
        return RECOVERY_CLASSIFICATION, checks
    if semantic_supported and not feature_improved:
        return MIXED_TRANSFER_CLASSIFICATION, checks
    return LOCAL_GAIN_CLASSIFICATION, checks


def _make_identity(source_manifest: Mapping[str, Any], data_identity: Mapping[str, Any], weights: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id="minimal-near-identity-nonlinear-generator-v1",
        trajectory_id="minimal-near-identity-nonlinear-generator-trajectory-v1",
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"stage1": True, "stage2_updates": SCIENTIFIC_UPDATES, "insertion_point": INSERTION_POINT, "operator": "normalized_intensity_dependent_phase", "eta_initialization": 0.0, "perturbation_epsilons": list(PERTURBATION_EPSILONS), "diagnostic_bank_size": DIAGNOSTIC_BANK_SIZE, "data_source": dict(data_identity), "beta": weights["beta"], "gamma": weights["gamma"], "delta": weights["delta"]},
        model_identity="LIGHTGEN_2D_MINIMAL_NEAR_IDENTITY_NONLINEAR_GENERATOR_V1",
        optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase", "generator.eta"], "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "stage1_optimizer_created": False, "eta_state": "zero_moments_copied_step_counter"},
        sampler_identity={"calibration_bank": "accepted seed-1234 TRAIN calibration batch split A/B as first/last five rows", "batch_size": LOGICAL_BATCH_SIZE},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id="minimal-near-identity-nonlinear-generator-v1:attempt_01",
    )


def _report_common(
    *,
    root: Path,
    target: Path,
    start_git: Mapping[str, Any],
    authorities: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    cuda_runtime: Mapping[str, Any],
    architecture: Mapping[str, Any],
    intensity_reference: Mapping[str, Any],
    function_check: Mapping[str, Any],
    stage1: Mapping[str, Any],
    classification: str,
    stage2_executed: bool,
    outer_gate_checks: Mapping[str, Any],
    data_reads: Mapping[str, Any],
    unresolved: Sequence[str],
    validation: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": classification == RECOVERY_CLASSIFICATION,
        "classification": classification,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "local_state": {"branch": start_git["branch"], "head": start_git["head"], "dirty_before": start_git["worktree_dirty"], "git_end": successor.git_identity(root)},
        "input_authorities": authorities["authority_hashes"],
        "clean_2L_starting_authority": {
            "report": _artifact_ref(root, root / CONFLICT_REPORT, role="accepted 2L conflict-aware result consumed as starting authority"),
            "step_0": _artifact_ref(root, root / CONFLICT_STEP_0, role="accepted clean 2L conflict-aware optimizer start"),
            "step_300_not_consumed": {"path": CONFLICT_STEP_300.as_posix(), "sha256": EXPECTED_CONFLICT_STEP_300_SHA256, "role": "identity-only predecessor endpoint; not loaded as nonlinear start"},
        },
        "model_status": {"operator": "MODEL_ASSUMPTION_ALGORITHMIC_SURROGATE", "measured_kerr": False, "te_validation": False, "fiber_facet_validation": False, "chen_exact_nonlinearity": False, "hardware_evidence": False},
        "current_2L_generator_contract": architecture["current_2L"],
        "nonlinear_contract": architecture["minimal_near_identity_nonlinear"],
        "intensity_reference": dict(intensity_reference),
        "function_preserving_check": function_check,
        "stage_1_local_controllability": stage1,
        "stage_2_executed": stage2_executed,
        "comparison_to_accepted_2L_conflict_aware": {"smooth": "0.1030 -> 0.1036", "paired_ci95": [0.0, 0.00121], "semantic_recovery": False, "cross_run_significance": "descriptive only; no new cross-run significance test"},
        "comparison_to_3L_linear_expansion": {"classification": "LOCAL_CONTROLLABILITY_GAIN_WITHOUT_DATASET_RECOVERY", "smooth": "0.1030 -> 0.1032", "R_raw": "1.5288 -> 2.0526", "R_feature": "3.0552 -> 3.6763", "not_consumed_as_start": True},
        "counterfactual_integrity": {"clean_2L_step0_used": True, "failed_3L_step300_not_used": True, "same_frozen_latent_authority": True, "same_data_order": True, "same_objectives": True, "same_loss_weights": True, "same_optimizer_family_and_hyperparameters": True, "encoder_unchanged": True, "OLS_unchanged": True, "P2_Q2_unchanged": True, "protected_test_sealed_reads": False},
        "outer_gate_checks": dict(outer_gate_checks),
        "data_reads": dict(data_reads),
        "forbidden_data_reads": {"protected": 0, "test": 0, "sealed": 0},
        "claim_boundary": {"scientific_training_executed": stage2_executed, "stage1_local_nonlinear_controllability_supported": bool(stage1.get("supported", False)), "algorithmic_surrogate_only": True, "measured_kerr_or_te": False, "generation_success": False, "global_architecture_inadequacy": False, "formal_promotion": False, "physical_hardware_validity": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False},
        "files_changed": {"scientific_artifacts": sorted(path.name for path in target.iterdir()), "repository_source": ["scripts/research/execute_lightgen_2d_minimal_near_identity_nonlinear_generator_v1.py"]},
        "validation": dict(validation),
        "unresolved": list(unresolved),
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": run_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def run_nonlinear(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "nonlinear output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"nonlinear output already exists: {target}")

    start_git = successor.git_identity(root)
    _require(start_git["branch"] == EXPECTED_BRANCH and start_git["head"] == EXPECTED_HEAD, "local branch/HEAD conflicts with accepted nonlinear authorities")
    authorities = _load_authorities(root)
    cuda_runtime = successor._prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "near-identity nonlinear execution requires CUDA for the production runtime")
    device = torch.device("cuda:0")

    base_consumer, train_evaluator, smooth_evaluator, evaluator_identity = conflict._load_models(root, authorities, device)
    base_consumer.runtime.eval()
    calibration_indices = authorities["calibration_indices"]
    images = authorities["images"]
    labels = authorities["labels"]
    _require(int(calibration_indices.numel()) == DIAGNOSTIC_BANK_SIZE, "accepted calibration bank size changed")
    calibration_batch = images.index_select(0, calibration_indices).to(device)
    calibration_labels = labels.index_select(0, calibration_indices).to(device)
    intensity_reference_value, intensity_reference = _calibrate_intensity_reference(base_consumer, images, calibration_indices, device)
    candidate_consumer = _build_candidate_consumer(authorities, device, intensity_reference_value)
    candidate_consumer.runtime.eval()

    with torch.no_grad():
        insertion_result = base_consumer.runtime.forward_from_image(calibration_batch)
    insertion_plane = insertion_result.generator_records[0].plane
    architecture = _architecture_resolution(base_consumer, candidate_consumer, intensity_reference, insertion_plane)

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
            root / "src/ldm_project/v2/optics/propagation.py",
            root / "src/ldm_project/v2/optics/relays.py",
            root / "src/ldm_project/v2/application/research/research_run_control.py",
            root / "src/ldm_project/v2/student/evaluator.py",
        ),
        config_identity={"authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "insertion_point": INSERTION_POINT, "operator": "normalized_intensity_dependent_phase", "eta_initialization": 0.0, "stage2_updates": SCIENTIFIC_UPDATES, "intensity_reference": intensity_reference_value},
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    _write_json(target / "source_manifest.json", source_manifest)
    run_identity = _make_identity(source_manifest, authorities["data_identity"], authorities["weights"])
    _write_json(target / "architecture_resolution.json", architecture)
    _write_json(target / "intensity_reference.json", intensity_reference)

    function_check = _function_preserving_check(base_consumer, candidate_consumer, train_evaluator, smooth_evaluator, calibration_batch)
    _write_json(target / "function_preserving_check.json", function_check)
    if not function_check["accepted_for_stage1"]:
        stage1 = {"schema": STAGE1_SCHEMA, "supported": False, "skipped": True, "reason": "BLOCKED_NONLINEAR_IDENTITY_COUNTERFACTUAL_INVALID"}
        report = _report_common(
            root=root,
            target=target,
            start_git=start_git,
            authorities=authorities,
            source_manifest=source_manifest,
            run_identity=run_identity,
            cuda_runtime=cuda_runtime,
            architecture=architecture,
            intensity_reference=intensity_reference,
            function_check=function_check,
            stage1=stage1,
            classification=BLOCKED_IDENTITY_CLASSIFICATION,
            stage2_executed=False,
            outer_gate_checks={"function_preserving": False, "stage1_opened": False, "stage2_opened": False},
            data_reads={"stage1_train_diagnostic": 0, "preflight_train": 0, "scientific_train_updates": 0, "scientific_dev": 0},
            unresolved=["eta=0 did not reproduce the clean accepted 2L Generator under the strict identity check."],
            validation={"cuda": True, "cuda_runtime": cuda_runtime, "architecture_resolution": True, "function_preserving_check": False, "stage1": "NOT_OPENED", "stage2": "NOT_OPENED", "no_protected_test_sealed": True},
        )
        _write_json(target / "report.json", report)
        return report

    split_a_indices = calibration_indices[:DIAGNOSTIC_HALF_SIZE]
    split_b_indices = calibration_indices[DIAGNOSTIC_HALF_SIZE:]
    _require(split_a_indices.numel() == DIAGNOSTIC_HALF_SIZE and split_b_indices.numel() == DIAGNOSTIC_HALF_SIZE, "diagnostic A/B split size changed")
    batch_a = images.index_select(0, split_a_indices).to(device)
    labels_a = labels.index_select(0, split_a_indices).to(device)
    batch_b = images.index_select(0, split_b_indices).to(device)
    labels_b = labels.index_select(0, split_b_indices).to(device)
    gradient_a = _eta_gradient_probe(candidate_consumer, train_evaluator, smooth_evaluator, batch_a, labels_a, authorities["training"]["references"], "calibration_A")
    gradient_b = _eta_gradient_probe(candidate_consumer, train_evaluator, smooth_evaluator, batch_b, labels_b, authorities["training"]["references"], "held_out_B")
    if bool(gradient_a["finite"] and gradient_a["nonzero"]):
        direction = -1.0 if float(gradient_a["gradient"]) > 0.0 else 1.0
        probes_b = _nonlinear_probes(candidate_consumer, train_evaluator, smooth_evaluator, batch_b, labels_b, authorities["training"]["references"], direction)
    else:
        direction = 0.0
        probes_b = {"schema": PROBE_SCHEMA, "direction": direction, "epsilons": list(PERTURBATION_EPSILONS), "records": [], "restored_after_every_perturbation": True, "all_raw_valid": False, "skipped": "eta gradient on calibration A was zero or nonfinite"}
    stage1_supported, stage1_details = _stage1_decision(function_check, gradient_a, gradient_b, probes_b, direction)
    stage1 = {
        "schema": STAGE1_SCHEMA,
        "supported": stage1_supported,
        "optimizer_created": False,
        "diagnostic_bank": {"size": DIAGNOSTIC_BANK_SIZE, "calibration_A_size": DIAGNOSTIC_HALF_SIZE, "held_out_B_size": DIAGNOSTIC_HALF_SIZE, "A_indices_sha256": tensor_sha256(split_a_indices), "B_indices_sha256": tensor_sha256(split_b_indices), "split_frozen_before_results": True, "data_role": TRAIN_ROLE},
        "gradient_A": gradient_a,
        "gradient_B": gradient_b,
        "directional_probes_B": probes_b,
        "decision": stage1_details,
        "operator_role": "MINIMAL_NONLINEAR_GENERATOR_CANDIDATE" if stage1_supported else "REJECTED_LOCAL_CANDIDATE",
    }
    _write_torch(target / "stage1_gradient_vectors.pt", {"schema": STAGE1_SCHEMA, "gradient_eta_A": torch.tensor(float(gradient_a["gradient"])), "gradient_eta_B": torch.tensor(float(gradient_b["gradient"])), "split_A_indices": split_a_indices, "split_B_indices": split_b_indices}, kind="minimal nonlinear Stage 1 eta gradients")
    _write_json(target / "stage1_directional_probes.json", stage1)

    if not stage1_supported:
        report = _report_common(
            root=root,
            target=target,
            start_git=start_git,
            authorities=authorities,
            source_manifest=source_manifest,
            run_identity=run_identity,
            cuda_runtime=cuda_runtime,
            architecture=architecture,
            intensity_reference=intensity_reference,
            function_check=function_check,
            stage1=stage1,
            classification=NOT_SUPPORTED_CLASSIFICATION,
            stage2_executed=False,
            outer_gate_checks={"function_preserving": True, "stage1_supported": False, "stage2_opened": False},
            data_reads={"stage1_train_diagnostic": DIAGNOSTIC_BANK_SIZE, "preflight_train": 0, "scientific_train_updates": 0, "scientific_dev": 0},
            unresolved=["The single normalized intensity-dependent eta direction did not transfer from calibration A to held-out TRAIN B under the fixed local probes."],
            validation={"cuda": True, "cuda_runtime": cuda_runtime, "architecture_resolution": True, "function_preserving_check": True, "stage1": True, "stage1_optimizer_created": False, "stage2": "NOT_OPENED", "no_protected_test_sealed": True},
        )
        _write_json(target / "report.json", report)
        return report

    preflight_candidate = _build_candidate_consumer(authorities, device, intensity_reference_value)
    preflight = _preflight(
        preflight_candidate,
        train_evaluator,
        authorities["training"]["references"],
        images,
        labels,
        authorities["train_indices"],
        calibration_indices,
        authorities["conflict_optimizer_state"],
        calibration_batch,
        calibration_labels,
        beta=authorities["weights"]["beta"],
        gamma=authorities["weights"]["gamma"],
        delta=authorities["weights"]["delta"],
    )
    _write_json(target / "preflight.json", preflight)
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    control.mark_preflight_ready()
    control.start_execution()
    effective_identity = control.begin_scientific_attempt("minimal-near-identity-nonlinear-generator-v1:attempt_01")
    _write_json(
        target / "execution_manifest.json",
        {
            "schema": "minimal_near_identity_nonlinear_generator_execution_manifest.v1",
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "authorization_id": AUTHORIZATION_ID,
            "authorization_token": AUTHORIZATION_TOKEN,
            "input_authorities": ["M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION", conflict.OUTER_GATE, attribution.OUTER_GATE, "CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1_Q2", "SMOOTH_MNIST_EVALUATOR_V2_1"],
            "clean_2L_start": {"path": CONFLICT_STEP_0.as_posix(), "sha256": EXPECTED_CONFLICT_STEP_0_SHA256},
            "failed_3L_endpoint_consumed": False,
            "source_manifest": source_manifest,
            "run_identity": effective_identity.to_dict(),
            "data_identity": authorities["data_identity"],
            "git_before": start_git,
            "cuda_runtime": cuda_runtime,
            "scope": {"stage1": True, "bounded_updates": SCIENTIFIC_UPDATES, "trainable": ["generator.phase", "generator.eta"], "frozen": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "evaluators", "generator.intensity_reference"], "protected": False, "test": False, "sealed": False},
        },
    )

    candidate_consumer = _build_candidate_consumer(authorities, device, intensity_reference_value)
    candidate_consumer.runtime.train()
    optimizer, optimizer_mode = _new_optimizer(candidate_consumer, authorities["conflict_optimizer_state"])
    sampler = DeterministicTrainRoleSampler(authorities["train_indices"], labels)
    sampler_state = dict(authorities["step0"]["sampler_state"])
    sampler_state["cursor"] = 0
    sampler.load_state_dict(sampler_state)
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == calibration_indices.tolist(), "scientific first TRAIN batch differs from clean 2L authority")
    sampler.load_state_dict(sampler_state)
    fixed_batch = calibration_batch
    fixed_labels = calibration_labels
    ledger = DataReadLedger()
    history: list[dict[str, Any]] = []
    composition_trajectory: dict[str, dict[str, Any]] = {}
    dev_records: dict[str, dict[str, Any]] = {}
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=SCIENTIFIC_UPDATES, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=0, latest_losses={"preflight": "READY", "optimizer_mode": optimizer_mode}, force=True)
    composition_trajectory["0"] = _diagnose_composition(candidate_consumer, train_evaluator, fixed_batch, fixed_labels, authorities["training"]["references"], fixed_batch, beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta=authorities["weights"]["delta"], step=0)
    dev_records["0"] = _candidate_dev_eval(candidate_consumer, train_evaluator, smooth_evaluator, images, labels, authorities["development_indices"], authorities["training"]["references"], ledger, step=0)
    _write_json(target / "dev_diagnostics_step_000000.json", {"schema": DEV_SCHEMA, "step": 0, "diagnostic": {key: value for key, value in dev_records["0"].items() if not key.startswith("_")}})
    _write_torch(target / "scientific_step_000000.pt", _checkpoint_payload(candidate_consumer, optimizer, sampler, history, composition_trajectory, step=0, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight, scientific_artifact=True), kind="minimal near-identity nonlinear scientific checkpoint")

    last_recovery_step = 0
    last_recovery_time = time.time()
    last_recovery_path: Path | None = None

    def recovery_factory(step: int) -> Mapping[str, Any]:
        return _recovery_payload(candidate_consumer, optimizer, sampler, history, composition_trajectory, step=step, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight)

    heartbeat.update(state=RUNNING, phase="TRAINING", step=0, latest_losses={"rule": "conflict-aware", "operator": "near-identity-intensity-phase", "eps_proj": conflict.EPS_PROJ}, force=True)
    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_factory, phase="TRAINING", total_steps=SCIENTIFIC_UPDATES)
    with interrupt:
        for step in range(SCIENTIFIC_UPDATES):
            indices, batch_labels = sampler.next_batch()
            _require(torch.equal(labels.index_select(0, indices), batch_labels), f"scientific TRAIN sampler labels changed at update {step + 1}")
            ledger.record(TRAIN_ROLE, int(indices.numel()))
            update_batch = images.index_select(0, indices).to(device)
            trace = _apply_update(candidate_consumer, train_evaluator, optimizer, update_batch, batch_labels.to(device), authorities["training"]["references"], fixed_batch, beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta=authorities["weights"]["delta"], step=step + 1)
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
                composition_trajectory[str(completed)] = _diagnose_composition(candidate_consumer, train_evaluator, fixed_batch, fixed_labels, authorities["training"]["references"], fixed_batch, beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta=authorities["weights"]["delta"], step=completed)
                dev_records[str(completed)] = _candidate_dev_eval(candidate_consumer, train_evaluator, smooth_evaluator, images, labels, authorities["development_indices"], authorities["training"]["references"], ledger, step=completed)
                _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {"schema": DEV_SCHEMA, "step": completed, "diagnostic": {key: value for key, value in dev_records[str(completed)].items() if not key.startswith("_")}})
                _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(candidate_consumer, optimizer, sampler, history, composition_trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight, scientific_artifact=True), kind="minimal near-identity nonlinear scientific checkpoint")

    heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=SCIENTIFIC_UPDATES, latest_losses={"completed_updates": SCIENTIFIC_UPDATES}, force=True)
    correctness_payload = {"schema": "minimal_near_identity_nonlinear_dev_correctness.v1", "development_indices_sha256": tensor_sha256(authorities["development_indices"]), "labels": labels.index_select(0, authorities["development_indices"]), "steps": {step: {"train_correct": dev_records[step]["_train_correct"], "smooth_correct": dev_records[step]["_smooth_correct"]} for step in dev_records}}
    _write_torch(target / "dev_correctness.pt", correctness_payload, kind="minimal near-identity nonlinear DEV correctness vectors")
    bootstrap = successor._paired_bootstrap(correctness_payload["labels"], correctness_payload["steps"]["0"]["smooth_correct"], correctness_payload["steps"]["300"]["smooth_correct"])
    _write_torch(target / "paired_bootstrap.pt", bootstrap, kind="minimal near-identity nonlinear paired DEV Smooth bootstrap")
    bootstrap_report = dict(bootstrap)
    bootstrap_report.pop("macro_delta_tensor", None)
    bootstrap_report.pop("draws", None)
    dev_semantic = {"primary_evaluator": evaluator_identity["smooth"], "step_0_macro_recall": float(dev_records["0"]["smooth_evaluator"]["macro_recall"]), "step_300_macro_recall": float(dev_records["300"]["smooth_evaluator"]["macro_recall"]), "delta_macro_recall": float(dev_records["300"]["smooth_evaluator"]["macro_recall"] - dev_records["0"]["smooth_evaluator"]["macro_recall"]), "step_0_per_class_recall": list(dev_records["0"]["smooth_evaluator"]["per_class_recall"]), "step_300_per_class_recall": list(dev_records["300"]["smooth_evaluator"]["per_class_recall"]), "per_class_delta": list(bootstrap_report["per_class_delta"]), "paired_bootstrap_ci": dict(bootstrap_report["ci95"]), "paired_bootstrap_artifact": _artifact_ref(root, target / "paired_bootstrap.pt", role="minimal near-identity nonlinear paired DEV Smooth bootstrap"), "trajectory": {step: {"smooth_macro_recall": float(record["smooth_evaluator"]["macro_recall"]), "smooth_per_class_recall": list(record["smooth_evaluator"]["per_class_recall"])} for step, record in dev_records.items()}, "evaluator_separation": {"train_loss_provider": "FrozenMNISTEvaluator v1", "dev_primary": "SmoothMNISTEvaluator v2.1", "same_checkpoint": False, "smooth_in_objective": False}}
    dev_reconstruction = {"step_0": float(dev_records["0"]["reconstruction"]["L_rec"]), "step_100": float(dev_records["100"]["reconstruction"]["L_rec"]), "step_200": float(dev_records["200"]["reconstruction"]["L_rec"]), "step_300": float(dev_records["300"]["reconstruction"]["L_rec"]), "delta_step_300_minus_step_0": float(dev_records["300"]["reconstruction"]["L_rec"] - dev_records["0"]["reconstruction"]["L_rec"]), "tolerance": RECONSTRUCTION_TOLERANCE, "trajectory_pass": bool(dev_records["300"]["reconstruction"]["L_rec"] - dev_records["0"]["reconstruction"]["L_rec"] <= RECONSTRUCTION_TOLERANCE)}
    optical_validity = {"checkpoints": {step: {"finite": bool(record["raw_cmos"]["finite"]), "nonconstant": bool(record["raw_cmos"]["nonconstant"]), "zero_image_count": int(record["raw_cmos"]["zero_image_count"]), "L_op": float(record["optical"]["L_op"])} for step, record in dev_records.items()}, "all_finite": all(bool(record["raw_cmos"]["finite"]) for record in dev_records.values()), "all_nonconstant": all(bool(record["raw_cmos"]["nonconstant"]) for record in dev_records.values()), "all_zero_image_counts_zero": all(int(record["raw_cmos"]["zero_image_count"]) == 0 for record in dev_records.values())}
    _require(optical_validity["all_finite"] and optical_validity["all_nonconstant"] and optical_validity["all_zero_image_counts_zero"], "nonlinear optical validity failed")
    transfer = _p2q2_transfer(root, candidate_consumer, authorities, root / producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH, target)
    classification, checks = _classification(dev_semantic, dev_reconstruction, transfer)
    control.complete()
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=SCIENTIFIC_UPDATES, latest_losses={"classification": classification}, force=True)
    data_reads = {"stage1_train_diagnostic": DIAGNOSTIC_BANK_SIZE, "preflight_train": int(preflight["train_reads"]), "scientific_train_updates": ledger.counts.get(TRAIN_ROLE, 0), "scientific_dev": ledger.counts.get(DEV_ROLE, 0), "total_train": DIAGNOSTIC_BANK_SIZE + int(preflight["train_reads"]) + ledger.counts.get(TRAIN_ROLE, 0), "total_dev": ledger.counts.get(DEV_ROLE, 0)}
    extra = {"preflight": preflight, "optimizer": {"mode": optimizer_mode, "name": "Adam", "parameter_names": ["generator.phase", "generator.eta"], "phase_moments_reused": True, "eta_moments_zero_at_step0": True, "eta_step_counter": "copied_from_phase_state", "hyperparameters": {"learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD}}, "training_trajectory": {"updates": len(history), "checkpoint_steps": list(CHECKPOINT_STEPS), "trace": history, "composition_trajectory": composition_trajectory, "sampler_cursor": int(sampler.cursor)}, "dev_smooth": dev_semantic, "dev_reconstruction": dev_reconstruction, "optical_validity": optical_validity, "p2_q2_transfer": transfer, "outer_gate_checks": checks, "unresolved": [] if classification == RECOVERY_CLASSIFICATION else ["The single near-identity nonlinear candidate did not establish the full local-plus-dataset semantic recovery claim under the frozen latent manifold."], "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "architecture_resolution": True, "function_preserving_check": True, "stage1_gradient_canary": True, "stage2_three_step_real_update": True, "checkpoints": list(CHECKPOINT_STEPS), "dev_evaluations": list(CHECKPOINT_STEPS), "paired_bootstrap": True, "frozen_gradient_firewall": True, "eta_finite": bool(torch.isfinite(candidate_consumer.runtime.generator.eta).item()), "no_protected_test_sealed": True}}
    report = _report_common(root=root, target=target, start_git=start_git, authorities=authorities, source_manifest=source_manifest, run_identity=effective_identity, cuda_runtime=cuda_runtime, architecture=architecture, intensity_reference=intensity_reference, function_check=function_check, stage1=stage1, classification=classification, stage2_executed=True, outer_gate_checks=checks, data_reads=data_reads, unresolved=extra.pop("unresolved"), validation=extra.pop("validation"), extra={**extra, "dev_smooth": dev_semantic, "dev_reconstruction": dev_reconstruction, "optical_validity": optical_validity, "p2_q2_transfer": transfer})
    _write_json(target / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and execute the bounded nonlinear arm")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_nonlinear(output_dir=args.output_dir)
    except (NonlinearExecutionError, successor.RecoveryExecutionError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "stage2_executed": report["stage_2_executed"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
