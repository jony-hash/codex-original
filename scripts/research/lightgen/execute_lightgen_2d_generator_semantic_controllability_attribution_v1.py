"""Run a no-update attribution probe for the frozen LightGen Generator.

The consumer compares reconstruction, TRAIN semantic, and independent Smooth
semantic gradients at the accepted step-0 and step-300 successor states.  It
also evaluates symmetric, parameter-norm-relative perturbations at step 0.
No optimizer is created and every perturbation is restored before the next
measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as producer
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as phase_repair
from scripts.research.lightgen import execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 as successor
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import DataReadLedger, LightGen2DChenBOGTComplexOLSTrainingConsumer, PRODUCTION_PROPAGATION_GRID, TRAIN_ROLE, git_identity
from ldm_project.v2.application.research.shared.research_run_control import ResearchRunControl, ResearchRunIdentity, capture_execution_source_manifest
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
)


TASK_ID = "EXECUTE_GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_V1"
OUTER_GATE = "GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
AUTHORIZATION_ID = "USER_MODE_IMPLEMENT_LOCAL_DELTA_20260918"
RUN_ID = "generator-semantic-controllability-attribution-v1"
TRAJECTORY_ID = "generator-semantic-controllability-attribution-trajectory-v1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

ACCEPTED_SUCCESSOR_RUN = Path(
    "outputs/20260918_frozen_latent_generator_semantic_recovery_v1/runs/attempt_01"
)
ACCEPTED_SUCCESSOR_REPORT = ACCEPTED_SUCCESSOR_RUN / "report.json"
ACCEPTED_STEP_0 = ACCEPTED_SUCCESSOR_RUN / "scientific_step_000000.pt"
ACCEPTED_STEP_300 = ACCEPTED_SUCCESSOR_RUN / "scientific_step_000300.pt"
EXPECTED_SUCCESSOR_REPORT_SHA256 = "fcb85e17e4b3d258fbe76d975103869e8d18a11252965dd5581f02a0acdb19b3"
EXPECTED_STEP_0_SHA256 = "bf103a0a876d78e5e07c8095c82c40a08f2d829b85e01a28f424879936d61be7"
EXPECTED_STEP_300_SHA256 = "58bb27b4fb7b37667118bcbfa44df3ae6b3485db8b6326280aba9190516b23a0"

SMOOTH_CHECKPOINT = successor.SMOOTH_CHECKPOINT
EXPECTED_SMOOTH_CHECKPOINT_SHA256 = successor.EXPECTED_SMOOTH_CHECKPOINT_SHA256
EXPECTED_TRAIN_EVALUATOR_SHA256 = producer.EVALUATOR_CHECKPOINT_SHA256
TRAINING_SOURCE_SHA256 = successor.TRAINING_SOURCE_SHA256

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260918_generator_semantic_controllability_attribution_v1/runs/attempt_01"
)
PERTURBATION_EPSILONS = (1.0e-4, 3.0e-4, 1.0e-3)
LOGICAL_BATCH_SIZE = 10
REPORT_SCHEMA = "generator_semantic_controllability_attribution_report.v1"
GRADIENT_SCHEMA = "generator_semantic_controllability_attribution_gradients.v1"
PROBE_SCHEMA = "generator_semantic_controllability_attribution_directional_probes.v1"

ATTRIBUTION_MISMATCH = "SEMANTIC_OBJECTIVE_TRANSFER_MISMATCH_SUPPORTED"
ATTRIBUTION_CONTROLLABILITY = "GENERATOR_SEMANTIC_CONTROLLABILITY_LIMITATION_SUPPORTED"
ATTRIBUTION_CONFLICT = "RECONSTRUCTION_SEMANTIC_GRADIENT_CONFLICT_SUPPORTED"
ATTRIBUTION_MIXED = "GENERATOR_SEMANTIC_CONTROLLABILITY_MIXED"


class AttributionExecutionError(RuntimeError):
    """Raised when an attribution authority or fail-closed boundary is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AttributionExecutionError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _runtime_state(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in consumer.runtime.state_dict().items()
    }


def _generator(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> torch.nn.Parameter:
    parameters = dict(consumer.runtime.named_parameters())
    _require(set(parameters) == {"encoder.phase", "generator.phase"}, "runtime parameter surface changed")
    return parameters["generator.phase"]


def _freeze_generator_only(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> None:
    for name, parameter in consumer.runtime.named_parameters():
        parameter.requires_grad_(name == "generator.phase")
    trainable = [name for name, parameter in consumer.runtime.named_parameters() if parameter.requires_grad]
    _require(trainable == ["generator.phase"], f"attribution trainable firewall changed: {trainable}")


def _load_authorities(root: Path) -> dict[str, Any]:
    report_path = root / ACCEPTED_SUCCESSOR_REPORT
    _require(file_sha256(report_path) == EXPECTED_SUCCESSOR_REPORT_SHA256, "accepted successor report identity changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("result") == "VALID_COMPLETED", "accepted successor result is not VALID_COMPLETED")
    _require(report.get("outer_gate") == successor.OUTER_GATE, "accepted successor outer gate changed")
    _require(report.get("classification") == successor.NOT_SUPPORTED_CLASSIFICATION, "accepted successor classification changed")
    _require(report.get("training_trajectory", {}).get("updates") == 300, "accepted successor update count changed")

    step0 = safe_torch_load(root / ACCEPTED_STEP_0, expected_sha256=EXPECTED_STEP_0_SHA256, map_location="cpu")
    step300 = safe_torch_load(root / ACCEPTED_STEP_300, expected_sha256=EXPECTED_STEP_300_SHA256, map_location="cpu")
    for name, checkpoint, expected_step in (("step-0", step0, 0), ("step-300", step300, 300)):
        _require(isinstance(checkpoint, Mapping), f"accepted {name} checkpoint is malformed")
        _require(checkpoint.get("completed_updates") == expected_step, f"accepted {name} checkpoint step changed")
        _require(checkpoint.get("scientific_artifact") is True and checkpoint.get("operational_artifact") is False, f"accepted {name} checkpoint role changed")
        _require(checkpoint.get("trainable_parameter_names") == ["generator.phase"], f"accepted {name} trainable boundary changed")
        _require("encoder.phase" in checkpoint.get("frozen_parameter_names", []), f"accepted {name} encoder freeze is missing")
        _require(checkpoint.get("outer_gate") == successor.OUTER_GATE, f"accepted {name} checkpoint gate changed")
        _tensor(checkpoint.get("generator_state"), f"accepted {name} generator state")
        _tensor(checkpoint.get("encoder_state"), f"accepted {name} encoder state")

    step0_encoder = _tensor(step0.get("encoder_state"), "step-0 encoder").cpu()
    step300_encoder = _tensor(step300.get("encoder_state"), "step-300 encoder").cpu()
    step0_generator = _tensor(step0.get("generator_state"), "step-0 generator").cpu()
    step300_generator = _tensor(step300.get("generator_state"), "step-300 generator").cpu()
    _require(torch.equal(step0_encoder, step300_encoder), "accepted encoder state changed between step 0 and step 300")
    _require(not torch.equal(step0_generator, step300_generator), "accepted generator state did not change between step 0 and step 300")

    cpu_context = phase_repair._load_cpu_context(root)
    training = successor._load_successor_training_authority(root, {"training": cpu_context})
    images, labels, train_indices, development_indices, data_identity = producer._load_scientific_dataset(root)
    _require(data_identity.get("source_sha256") == TRAINING_SOURCE_SHA256, "TRAIN source identity changed")
    _require(data_identity == training["data_identity"], "accepted TRAIN/DEV data identity changed")
    calibration_identity = _mapping(training["weights"].calibration_batch_identity, "accepted calibration identity")
    source_indices = calibration_identity.get("source_indices")
    _require(isinstance(source_indices, list) and len(source_indices) == LOGICAL_BATCH_SIZE, "accepted calibration indices are unavailable")
    calibration_indices = torch.tensor(source_indices, dtype=torch.long)
    calibration_labels = labels.index_select(0, calibration_indices)
    _require(calibration_labels.tolist() == list(range(10)), "accepted calibration batch labels changed")
    _require(torch.all(torch.isin(calibration_indices, train_indices)), "accepted calibration batch left TRAIN role")

    return {
        "report": report,
        "step0": dict(step0),
        "step300": dict(step300),
        "training": training,
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "development_indices": development_indices,
        "data_identity": data_identity,
        "calibration_indices": calibration_indices,
        "calibration_labels": calibration_labels,
        "hashes": {
            "successor_report": EXPECTED_SUCCESSOR_REPORT_SHA256,
            "step0": EXPECTED_STEP_0_SHA256,
            "step300": EXPECTED_STEP_300_SHA256,
            "parent_report": successor.EXPECTED_SOURCE_REPORT_SHA256,
            "parent_checkpoint_step300": successor.EXPECTED_SOURCE_CHECKPOINT_SHA256,
            "generation_bundle": successor.EXPECTED_SOURCE_BUNDLE_SHA256,
            "frozen_references": successor.EXPECTED_SOURCE_REFERENCES_SHA256,
            "frozen_weights": successor.EXPECTED_SOURCE_WEIGHTS_SHA256,
            "smooth_evaluator": EXPECTED_SMOOTH_CHECKPOINT_SHA256,
        },
    }


def _load_models(
    root: Path,
    authorities: Mapping[str, Any],
    device: torch.device,
) -> tuple[LightGen2DChenBOGTComplexOLSTrainingConsumer, FrozenMNISTEvaluator, SmoothMNISTEvaluator, dict[str, Any]]:
    training = _mapping(authorities.get("training"), "training authority")
    consumer = LightGen2DChenBOGTComplexOLSTrainingConsumer(device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    bundle = _mapping(training.get("bundle"), "accepted generation bundle")
    _require(consumer.runtime_identity() == _mapping(bundle.get("ols"), "bundle OLS").get("interface"), "runtime identity differs from accepted bundle")
    consumer.load_trainable_state_dict({
        "encoder.phase": authorities["step0"]["encoder_state"],
        "generator.phase": authorities["step0"]["generator_state"],
    })
    consumer.bind_references(training["references"], for_scientific=True)
    consumer.bind_weights(training["weights"], for_scientific=True)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()

    train_checkpoint = root / producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH
    _require(file_sha256(train_checkpoint) == EXPECTED_TRAIN_EVALUATOR_SHA256, "TRAIN semantic evaluator identity changed")
    train_evaluator, train_payload = load_frozen_evaluator(train_checkpoint, device=device)
    _require(all(not parameter.requires_grad for parameter in train_evaluator.parameters()), "TRAIN evaluator is not frozen")
    smooth_evaluator, smooth_identity = successor._load_smooth_evaluator(root, device)
    return consumer, train_evaluator, smooth_evaluator, {
        "train_checkpoint": _relative(root, train_checkpoint),
        "train_checkpoint_sha256": EXPECTED_TRAIN_EVALUATOR_SHA256,
        "train_architecture": train_payload.get("architecture_kind"),
        "smooth": smooth_identity,
    }


def _semantic_view(raw: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))


def _loss_bundle(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "attribution real path is incomplete")
    raw = result.raw_intensity
    target = result.image_input.slm_intensity
    _require(raw.shape == target.shape and tuple(raw.shape[-2:]) == (200, 200), "attribution raw/target geometry changed")
    _require(raw.dtype == torch.float32 and target.dtype == torch.float32, "attribution raw/target dtype changed")
    _finite_tensor(raw, "attribution raw CMOS")
    _finite_tensor(target, "attribution SLM target")
    view = _semantic_view(raw)
    _, train_logits = train_evaluator(view)
    _, smooth_logits = smooth_evaluator(view)
    l_rec = (raw - target).square().mean()
    l_sem_train = F.cross_entropy(train_logits, labels.to(device=batch.device, dtype=torch.long))
    l_sem_smooth = F.cross_entropy(smooth_logits, labels.to(device=batch.device, dtype=torch.long))
    for name, value in (("L_rec", l_rec), ("L_sem_train", l_sem_train), ("L_sem_smooth", l_sem_smooth)):
        _finite_tensor(value, name)
    flat = raw.detach().to(torch.float64).flatten(1)
    sample_std = flat.std(dim=1, unbiased=False)
    return {
        "result": result,
        "raw": raw,
        "target": target,
        "L_rec": l_rec,
        "L_sem_train": l_sem_train,
        "L_sem_smooth": l_sem_smooth,
        "raw_validity": {
            "finite": bool(torch.isfinite(raw).all().item()),
            "nonconstant_fraction": float((sample_std > 0.0).to(torch.float64).mean().item()),
            "zero_image_count": int((flat.abs().sum(dim=1) == 0.0).sum().item()),
        },
    }


def _gradient_norm(value: torch.Tensor, name: str) -> float:
    _finite_tensor(value, name)
    norm = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(norm) and norm > 0.0, f"{name} is zero or invalid")
    return norm


def _gradient_stats(value: torch.Tensor, name: str) -> dict[str, Any]:
    norm = _gradient_norm(value, name)
    flat = value.detach().flatten()
    finite_fraction = float(torch.isfinite(flat).to(torch.float64).mean().cpu())
    zero_fraction = float((flat == 0.0).to(torch.float64).mean().cpu())
    _require(finite_fraction == 1.0, f"{name} has nonfinite elements")
    layer_norms = [
        float(layer.detach().to(torch.float64).square().sum().sqrt().cpu())
        for layer in value.detach().reshape(value.shape[0], -1)
    ]
    return {
        "norm": norm,
        "finite_fraction": finite_fraction,
        "zero_fraction": zero_fraction,
        "per_generator_layer_norms": {f"generator.phase[{index}]": layer_norm for index, layer_norm in enumerate(layer_norms)},
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.detach().to(torch.float64).flatten()
    right64 = right.detach().to(torch.float64).flatten()
    denominator = float(left64.norm().cpu() * right64.norm().cpu())
    _require(denominator > 0.0 and math.isfinite(denominator), "gradient cosine denominator is invalid")
    value = float(torch.dot(left64, right64).cpu() / denominator)
    _require(math.isfinite(value), "gradient cosine is nonfinite")
    return value


def _analyze_gradients(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    checkpoint: Mapping[str, Any],
    checkpoint_name: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    consumer.load_trainable_state_dict({
        "encoder.phase": checkpoint["encoder_state"],
        "generator.phase": checkpoint["generator_state"],
    })
    _freeze_generator_only(consumer)
    consumer.runtime.eval()
    generator = _generator(consumer)
    before = _runtime_state(consumer)
    _require(generator.shape == (2, 1, 100, 100), "generator phase layer shape changed")
    losses = _loss_bundle(consumer, train_evaluator, smooth_evaluator, batch, labels)
    gradients = {
        "g_rec": torch.autograd.grad(losses["L_rec"], generator, retain_graph=True, allow_unused=False)[0],
        "g_sem_train": torch.autograd.grad(losses["L_sem_train"], generator, retain_graph=True, allow_unused=False)[0],
        "g_sem_smooth": torch.autograd.grad(losses["L_sem_smooth"], generator, retain_graph=False, allow_unused=False)[0],
    }
    stats = {name: _gradient_stats(value, f"{checkpoint_name} {name}") for name, value in gradients.items()}
    result = {
        "checkpoint": checkpoint_name,
        "completed_updates": int(checkpoint["completed_updates"]),
        "losses": {name: float(losses[name].detach().cpu()) for name in ("L_rec", "L_sem_train", "L_sem_smooth")},
        "raw_validity": losses["raw_validity"],
        "gradient_norms": stats,
        "cosines": {
            "C_train_smooth": _cosine(gradients["g_sem_train"], gradients["g_sem_smooth"]),
            "C_rec_smooth": _cosine(gradients["g_rec"], gradients["g_sem_smooth"]),
            "C_rec_train": _cosine(gradients["g_rec"], gradients["g_sem_train"]),
        },
        "smooth_objective": {
            "definition": "CE(SmoothMNISTEvaluator(adaptive_avg_pool2d(clamp(raw_cmos,[0,1]),28x28)), label)",
            "role": "TRAIN calibration batch local differentiable surrogate; not the formal DEV macro-recall metric",
        },
        "evaluator_parameter_gradients_none": bool(
            all(parameter.grad is None for parameter in train_evaluator.parameters())
            and all(parameter.grad is None for parameter in smooth_evaluator.parameters())
        ),
    }
    _require(result["evaluator_parameter_gradients_none"], f"{checkpoint_name} evaluator received a gradient")
    _require(_state_digest(before) == _state_digest(_runtime_state(consumer)), f"{checkpoint_name} gradient probe mutated runtime state")
    for name, parameter in consumer.runtime.named_parameters():
        if name != "generator.phase":
            _require(parameter.grad is None, f"{checkpoint_name} frozen parameter received a gradient: {name}")
    for parameter in consumer.runtime.parameters():
        parameter.grad = None
    return result, gradients


def _metric_deltas(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(candidate[name] - base[name])
        for name in ("L_rec", "L_sem_train", "L_sem_smooth")
    }


def _evaluate_no_grad(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        losses = _loss_bundle(consumer, train_evaluator, smooth_evaluator, batch, labels)
    return {
        "L_rec": float(losses["L_rec"].cpu()),
        "L_sem_train": float(losses["L_sem_train"].cpu()),
        "L_sem_smooth": float(losses["L_sem_smooth"].cpu()),
        "raw_validity": losses["raw_validity"],
    }


def _directional_probes(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    train_evaluator: FrozenMNISTEvaluator,
    smooth_evaluator: SmoothMNISTEvaluator,
    batch: torch.Tensor,
    labels: torch.Tensor,
    gradients: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    generator = _generator(consumer)
    base_state = generator.detach().clone()
    parameter_norm = float(base_state.to(torch.float64).norm().cpu())
    _require(math.isfinite(parameter_norm) and parameter_norm > 0.0, "step-0 generator parameter norm is invalid")
    base = _evaluate_no_grad(consumer, train_evaluator, smooth_evaluator, batch, labels)
    directions = {
        "semantic_train": -gradients["g_sem_train"].detach() / gradients["g_sem_train"].detach().norm(),
        "semantic_smooth": -gradients["g_sem_smooth"].detach() / gradients["g_sem_smooth"].detach().norm(),
        "reconstruction": -gradients["g_rec"].detach() / gradients["g_rec"].detach().norm(),
    }
    probes: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "checkpoint": "step-0",
        "epsilon_definition": "epsilon * ||generator.phase||_2 along a unit generator-phase direction",
        "epsilons": list(PERTURBATION_EPSILONS),
        "parameter_norm": parameter_norm,
        "base": base,
        "directions": {},
    }
    try:
        for direction_name, direction in directions.items():
            _finite_tensor(direction, f"{direction_name} direction")
            direction_records: list[dict[str, Any]] = []
            for epsilon in PERTURBATION_EPSILONS:
                signed: dict[str, Any] = {}
                for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                    with torch.no_grad():
                        generator.copy_(base_state + sign * epsilon * parameter_norm * direction)
                    candidate = _evaluate_no_grad(consumer, train_evaluator, smooth_evaluator, batch, labels)
                    signed[sign_name] = {
                        "metrics": candidate,
                        "deltas": _metric_deltas(base, candidate),
                    }
                    with torch.no_grad():
                        generator.copy_(base_state)
                plus = signed["plus"]["deltas"]
                minus = signed["minus"]["deltas"]
                signed["central_slope_per_relative_parameter_norm"] = {
                    metric: float((plus[metric] - minus[metric]) / (2.0 * epsilon))
                    for metric in ("L_rec", "L_sem_train", "L_sem_smooth")
                }
                direction_records.append({"epsilon": epsilon, **signed})
            probes["directions"][direction_name] = direction_records
    finally:
        with torch.no_grad():
            generator.copy_(base_state)
    _require(torch.equal(generator.detach().cpu(), base_state.detach().cpu()), "directional probe did not restore step-0 generator")
    probes["restored_after_every_perturbation"] = True
    return probes


def _stable_directional_sign(probes: Mapping[str, Any], direction: str, metric: str, *, plus_decreases: bool) -> bool:
    records = _mapping(probes["directions"], "directional probe directions")[direction]
    for record in records:
        plus = float(_mapping(record["plus"], "plus probe")["deltas"][metric])
        minus = float(_mapping(record["minus"], "minus probe")["deltas"][metric])
        if plus_decreases:
            if not (plus < 0.0 and minus > 0.0):
                return False
        elif not (plus > 0.0 and minus < 0.0):
            return False
    return True


def _median_abs_slope(probes: Mapping[str, Any], direction: str, metric: str) -> float:
    values = [
        abs(float(record["central_slope_per_relative_parameter_norm"][metric]))
        for record in _mapping(probes["directions"], "directional probe directions")[direction]
    ]
    values.sort()
    return values[len(values) // 2]


def _classify(
    step0: Mapping[str, Any],
    step300: Mapping[str, Any],
    probes: Mapping[str, Any],
    accepted_result: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str, str, str]:
    c_train_smooth = float(step0["cosines"]["C_train_smooth"])
    c_rec_smooth = float(step0["cosines"]["C_rec_smooth"])
    train_direction_improves_smooth = _stable_directional_sign(probes, "semantic_train", "L_sem_smooth", plus_decreases=True)
    smooth_direction_improves_smooth = _stable_directional_sign(probes, "semantic_smooth", "L_sem_smooth", plus_decreases=True)
    rec_direction_improves_rec = _stable_directional_sign(probes, "reconstruction", "L_rec", plus_decreases=True)
    rec_direction_worsens_smooth = _stable_directional_sign(probes, "reconstruction", "L_sem_smooth", plus_decreases=False)
    train_direction_worsens_smooth = _stable_directional_sign(probes, "semantic_train", "L_sem_smooth", plus_decreases=False)
    rec_response = _median_abs_slope(probes, "reconstruction", "L_rec")
    smooth_response = _median_abs_slope(probes, "semantic_smooth", "L_sem_smooth")
    primary_semantic_not_supported = accepted_result.get("classification") != successor.SUPPORTED_CLASSIFICATION
    objective_transfer = c_train_smooth <= 0.0 and train_direction_worsens_smooth
    conflict = c_rec_smooth < 0.0 and rec_direction_improves_rec and rec_direction_worsens_smooth
    controllability = (
        c_train_smooth > 0.0
        and train_direction_improves_smooth
        and smooth_direction_improves_smooth
        and primary_semantic_not_supported
        and smooth_response < rec_response
    )
    if objective_transfer:
        primary = ATTRIBUTION_MISMATCH
        recommendation = "Redesign the semantic objective/evaluator transfer while retaining the current Generator architecture; do not train automatically."
        mechanism = "The TRAIN semantic descent direction is directionally incompatible with the independent Smooth diagnostic."
    elif controllability:
        primary = ATTRIBUTION_CONTROLLABILITY
        recommendation = "Increase the minimal Generator controllable subspace/optical synthesis capacity; do not modify architecture automatically."
        mechanism = "TRAIN and Smooth semantic directions agree, but the local Smooth response is weaker than the reconstruction response and the accepted run did not improve independent semantics."
    else:
        primary = ATTRIBUTION_MIXED
        recommendation = "No single repair direction is justified by this probe; keep objective and architecture frozen until a new bounded design decision selects one."
        mechanism = "The sign-stable local evidence does not isolate objective transfer mismatch from Generator controllability or reconstruction interaction."
    secondary = ATTRIBUTION_CONFLICT if conflict else "NONE"
    details = {
        "raw_sign_rules_only": True,
        "objective_transfer_condition": objective_transfer,
        "controllability_condition": controllability,
        "conflict_condition": conflict,
        "primary_semantic_not_supported": primary_semantic_not_supported,
        "train_direction_improves_smooth": train_direction_improves_smooth,
        "smooth_direction_improves_smooth": smooth_direction_improves_smooth,
        "reconstruction_direction_improves_reconstruction": rec_direction_improves_rec,
        "reconstruction_direction_worsens_smooth": rec_direction_worsens_smooth,
        "median_abs_smooth_response_per_relative_parameter_norm": smooth_response,
        "median_abs_reconstruction_response_per_relative_parameter_norm": rec_response,
        "step0_C_train_smooth": c_train_smooth,
        "step0_C_rec_smooth": c_rec_smooth,
        "step300_C_train_smooth": float(step300["cosines"]["C_train_smooth"]),
        "step300_C_rec_smooth": float(step300["cosines"]["C_rec_smooth"]),
    }
    return primary, details, secondary, recommendation, mechanism


def _make_identity(source_manifest: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={
            "probe_only": True,
            "epsilons": list(PERTURBATION_EPSILONS),
            "smooth_objective": "CE on independent SmoothMNISTEvaluator over accepted TRAIN calibration batch",
            "step_comparison": [0, 300],
        },
        model_identity="LIGHTGEN_2D_GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_V1",
        optimizer_identity={"created": False, "parameter_updates": 0},
        sampler_identity={"calibration_batch": "accepted seed-1234 TRAIN calibration batch", "batch_size": LOGICAL_BATCH_SIZE},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=SCIENTIFIC_ATTEMPT_ID,
    )


def run_analysis(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "attribution output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"attribution output already exists: {target}")

    local_state = git_identity(root)
    _require(local_state["branch"] == EXPECTED_BRANCH and local_state["head"] == EXPECTED_HEAD, "local branch/HEAD conflicts with accepted attribution authority")
    authorities = _load_authorities(root)
    cuda_runtime = successor._prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "attribution requires CUDA for the accepted production runtime")
    device = torch.device("cuda:0")
    consumer, train_evaluator, smooth_evaluator, evaluator_identity = _load_models(root, authorities, device)
    step0_runtime_digest = _state_digest(_runtime_state(consumer))

    calibration_indices = authorities["calibration_indices"]
    images = authorities["images"]
    labels = authorities["labels"]
    batch = images.index_select(0, calibration_indices).to(device)
    batch_labels = labels.index_select(0, calibration_indices).to(device)
    ledger = DataReadLedger()
    ledger.record(TRAIN_ROLE, int(batch.shape[0]))
    ledger.assert_forbidden_zero()

    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            Path(successor.__file__).resolve(),
            Path(producer.__file__).resolve(),
            Path(phase_repair.__file__).resolve(),
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
            root / "src/ldm_project/v2/application/research/shared/research_run_control.py",
            root / "src/ldm_project/v2/student/evaluator.py",
        ),
        config_identity={
            "authorization_mode": "IMPLEMENT LOCAL DELTA",
            "outer_gate": OUTER_GATE,
            "no_update": True,
            "epsilons": list(PERTURBATION_EPSILONS),
            "input_hashes": authorities["hashes"],
            "calibration_indices_sha256": tensor_sha256(calibration_indices),
        },
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    run_identity = _make_identity(source_manifest)
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    control.mark_preflight_ready()
    control.start_execution()
    effective_identity = control.begin_scientific_attempt(SCIENTIFIC_ATTEMPT_ID)
    _write_json(target / "source_manifest.json", source_manifest)
    _write_json(
        target / "execution_manifest.json",
        {
            "schema": "generator_semantic_controllability_attribution_execution_manifest.v1",
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "authorization": {"mode": "IMPLEMENT", "authorization_id": AUTHORIZATION_ID, "scope": "no-update attribution probes"},
            "input_authorities": [
                "FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1_RESOLVED",
                "CHEN_BOGT_ON_COMPLEX_OLS_TRAINING_V1_STEP_300",
                "SMOOTH_MNIST_EVALUATOR_V2_1",
                "P2/Q2 frozen contextual authority only",
            ],
            "source_manifest": source_manifest,
            "run_identity": effective_identity.to_dict(),
            "local_state": local_state,
            "cuda_runtime": cuda_runtime,
            "scope": {
                "parameter_updates": 0,
                "optimizer_created": False,
                "train_role_rows": int(batch.shape[0]),
                "dev_rows": 0,
                "protected": False,
                "test": False,
                "sealed": False,
            },
        },
    )

    step0_result, step0_gradients = _analyze_gradients(
        consumer, train_evaluator, smooth_evaluator, batch, batch_labels, authorities["step0"], "step-0"
    )
    runtime_after_step0 = _runtime_state(consumer)
    probes = _directional_probes(consumer, train_evaluator, smooth_evaluator, batch, batch_labels, step0_gradients)
    _require(_state_digest(runtime_after_step0) == _state_digest(_runtime_state(consumer)), "directional probes changed runtime state")
    step300_result, step300_gradients = _analyze_gradients(
        consumer, train_evaluator, smooth_evaluator, batch, batch_labels, authorities["step300"], "step-300"
    )
    step0_state = {"encoder.phase": authorities["step0"]["encoder_state"], "generator.phase": authorities["step0"]["generator_state"]}
    consumer.load_trainable_state_dict(step0_state)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()
    _require(_state_digest(_runtime_state(consumer)) == step0_runtime_digest, "final restoration self-check failed")
    control.complete()

    gradient_payload = {
        "schema": GRADIENT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "checkpoint_hashes": {"step0": EXPECTED_STEP_0_SHA256, "step300": EXPECTED_STEP_300_SHA256},
        "calibration_indices": calibration_indices,
        "step0": {name: value.detach().cpu() for name, value in step0_gradients.items()},
        "step300": {name: value.detach().cpu() for name, value in step300_gradients.items()},
    }
    _write_torch(target / "gradient_vectors.pt", gradient_payload, kind="generator attribution gradients")
    _write_json(target / "directional_probes.json", probes)
    primary, classification_details, secondary, recommendation, mechanism = _classify(
        step0_result, step300_result, probes, authorities["report"]
    )
    generator0 = authorities["step0"]["generator_state"].to(torch.float64).flatten()
    generator300 = authorities["step300"]["generator_state"].to(torch.float64).flatten()
    delta_generator = generator300 - generator0
    generator_delta_norm = float(delta_generator.norm().cpu())
    generator0_norm = float(generator0.norm().cpu())
    geometry_change = {
        "generator_step0_norm": generator0_norm,
        "generator_step300_norm": float(generator300.norm().cpu()),
        "generator_delta_norm": generator_delta_norm,
        "generator_relative_delta_norm": generator_delta_norm / max(generator0_norm, 1.0e-12),
        "generator_state_cosine": float(torch.dot(generator0, generator300).cpu() / (generator0.norm() * generator300.norm()).cpu()),
        "C_train_smooth_delta": float(step300_result["cosines"]["C_train_smooth"] - step0_result["cosines"]["C_train_smooth"]),
        "C_rec_smooth_delta": float(step300_result["cosines"]["C_rec_smooth"] - step0_result["cosines"]["C_rec_smooth"]),
        "C_rec_train_delta": float(step300_result["cosines"]["C_rec_train"] - step0_result["cosines"]["C_rec_train"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "authorization_consumed": {"mode": "IMPLEMENT", "authorization_id": AUTHORIZATION_ID, "one_authorization": True},
        "local_state": {"branch": local_state["branch"], "head": local_state["head"], "dirty": local_state["worktree_dirty"]},
        "input_authorities": {
            "accepted_successor_report": _artifact_ref(root, root / ACCEPTED_SUCCESSOR_REPORT, role="accepted semantic-recovery result"),
            "accepted_step0_checkpoint": _artifact_ref(root, root / ACCEPTED_STEP_0, role="accepted step-0 Generator state"),
            "accepted_step300_checkpoint": _artifact_ref(root, root / ACCEPTED_STEP_300, role="accepted step-300 Generator state"),
            "parent_training": authorities["hashes"],
            "data_identity": authorities["data_identity"],
            "calibration_batch": {"source_indices": calibration_indices.tolist(), "labels": authorities["calibration_labels"].tolist(), "role": TRAIN_ROLE, "tensor_sha256": tensor_sha256(calibration_indices)},
        },
        "frozen_context": {
            "encoder": True,
            "OLS": True,
            "W_OLS": True,
            "latent_definitions": True,
            "generator_architecture": True,
            "P2": "frozen contextual authority only",
            "Q2": "frozen contextual authority only",
            "evaluator_parameters": True,
        },
        "objective_context": {
            "reconstruction": "exact existing raw CMOS versus SLM intensity MSE",
            "train_semantic": "existing FrozenMNISTEvaluator CE after clamp(raw,[0,1]) and adaptive_avg_pool2d to 28x28",
            "smooth_semantic": step0_result["smooth_objective"],
            "formal_primary_metric_unchanged": True,
        },
        "gradient_norms": {"step0": step0_result["gradient_norms"], "step300": step300_result["gradient_norms"]},
        "gradient_cosines": {"step0": step0_result["cosines"], "step300": step300_result["cosines"]},
        "directional_probes": {
            "artifact": _artifact_ref(root, target / "directional_probes.json", role="symmetric local directional probes"),
            "checkpoint": "step-0",
            "directions": ["semantic-train", "Smooth-semantic", "reconstruction"],
            "epsilons": list(PERTURBATION_EPSILONS),
            "all_perturbations_restored": bool(probes["restored_after_every_perturbation"]),
            "raw_validity_checked": True,
        },
        "semantic_controllability": {
            "step0": step0_result["smooth_objective"],
            "local_response": classification_details,
            "gradient_artifact": _artifact_ref(root, target / "gradient_vectors.pt", role="step-0 and step-300 gradient vectors"),
        },
        "step0_to_step300_geometry_change": geometry_change,
        "primary_attribution": primary,
        "secondary_conflict": secondary,
        "mechanism": mechanism,
        "recommended_single_repair_direction": recommendation,
        "outer_gate_decision": {"classification": primary, "secondary": secondary, "raw_values_only": True},
        "evaluator_identity": evaluator_identity,
        "no_update_preflight": {
            "status": "READY",
            "optimizer_created": False,
            "parameter_updates": 0,
            "runtime_state_restored": True,
            "frozen_parameter_gradient_firewall": True,
            "evaluator_parameter_gradients_none": True,
            "independent_smooth_reference": True,
        },
        "data_reads": ledger.to_dict(),
        "forbidden_data_reads": {"protected": ledger.counts.get("protected", 0), "test": ledger.counts.get("test", 0), "sealed": ledger.counts.get("sealed", 0)},
        "claim_boundary": {
            "local_attribution_only": True,
            "global_architecture_inadequacy": False,
            "global_optimum": False,
            "generation_acceptance": False,
            "physical_validity": False,
            "training_executed": False,
            "parameter_persisted": False,
        },
        "files_changed": {
            "repository_source": ["scripts/research/lightgen/execute_lightgen_2d_generator_semantic_controllability_attribution_v1.py"],
            "scientific_artifacts": sorted(path.name for path in target.iterdir()),
        },
        "validation": {
            "cuda": True,
            "cuda_runtime": cuda_runtime,
            "accepted_step0_step300_resolved": True,
            "smooth_evaluator_independent": True,
            "no_update": True,
            "symmetric_directional_probes": True,
            "protected_test_sealed_zero": True,
        },
        "unresolved": ["Attribution is local to the accepted calibration batch and frozen runtime; it is not a global architecture conclusion."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the bounded no-update attribution consumer")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "outer_gate": OUTER_GATE, "no_update": True, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_analysis(output_dir=args.output_dir)
    except AttributionExecutionError as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "primary_attribution": report["primary_attribution"], "secondary_conflict": report["secondary_conflict"], "outer_gate": report["outer_gate"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
