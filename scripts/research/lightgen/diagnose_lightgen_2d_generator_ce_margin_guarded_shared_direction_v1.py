"""Freeze-test a Generator direction guarded by evaluator CE and class margin."""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    check_resume_compatibility,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as base
from scripts.research.lightgen import diagnose_lightgen_2d_generator_class_guarded_shared_direction_v1 as class_guard
from scripts.research.lightgen import diagnose_lightgen_2d_generator_finite_displacement_response_v1 as finite
from scripts.research.lightgen import diagnose_lightgen_2d_generator_semantic_direction_adam_v1 as direction
from scripts.research.lightgen import diagnose_lightgen_2d_generator_shared_direction_transfer_v1 as shared


TASK_ID = "M0_GENERATOR_CE_MARGIN_GUARDED_SHARED_DIRECTION_V1"
REPORT_SCHEMA = "m0_generator_ce_margin_guarded_shared_direction_report.v2"
MANIFEST_SCHEMA = "m0_generator_ce_margin_guarded_shared_direction_manifest.v2"
BLOCK_SCHEMA = "m0_generator_ce_margin_guarded_readout_block.v2"
GRADIENT_SCHEMA = "m0_generator_ce_margin_guard_gradients.v2"
PROJECTION_SCHEMA = "m0_generator_ce_margin_guard_projection.v2"
FIT_ROWS = 200
HISTORIC_ROWS = 100
CONFIRMATION_ROWS = 300
TOTAL_ROWS = FIT_ROWS + HISTORIC_ROWS + CONFIRMATION_ROWS
CHUNK_SIZE = 100
CHUNK_COUNT = TOTAL_ROWS // CHUNK_SIZE
BATCH_SIZE = 10
EVALUATORS = ("E_S", "E_I", "E_C", "E_D")
GUARD_EVALUATORS = ("E_S", "E_I", "E_C")
FOCUS_CLASSES = (1, 7)
DISPLAY_CLASSES = (1, 7, 0, 4, 8, 9)
GUARD_PAIRS = tuple((name, class_id) for name in GUARD_EVALUATORS for class_id in range(10))
FLOAT_RTOL = 2e-5
FLOAT_ATOL = 2e-7
ETA_FLOOR = 1e-4
STABLE_ROWS = 24
KKT_TOLERANCE = 1e-8
ZERO_DIRECTION_RELATIVE = 1e-8
DERIVATIVE_RELATIVE_TOLERANCE = 2e-5
ESTIMATED_OUTPUT_BYTES = 1_000_000_000
STORAGE_RESERVE_BYTES = int(1.5 * 1024**3)
DEFAULT_CLASS_GUARD = Path("outputs/20261006_m0_generator_class_guarded_shared_direction_v1/runs/attempt_03")
DEFAULT_OUTPUT = Path("outputs/20261006_m0_generator_ce_margin_guarded_shared_direction_v1/runs/attempt_07_strict_fp32_compatibility")


class CEMarginDirectionError(RuntimeError):
    """An input, preflight, recovery, or frozen-measurement invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CEMarginDirectionError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().to(device="cpu", dtype=torch.float64).square().mean().sqrt())


def _resource_observation(device: torch.device) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "global_used_fraction_observation": (total_bytes - free_bytes) / total_bytes if total_bytes else None,
        "pytorch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "pytorch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _load_inputs(root: Path, class_guard_dir: Path) -> dict[str, Any]:
    _require(class_guard_dir.is_dir(), f"explicit class-guard directory is missing: {class_guard_dir}")
    population = json.loads((class_guard_dir / "input_population.json").read_text(encoding="utf-8"))
    report = json.loads((class_guard_dir / "report.json").read_text(encoding="utf-8"))
    _require(
        population.get("task_id") == class_guard.TASK_ID
        and report.get("task_id") == class_guard.TASK_ID
        and report.get("status") == "VALID_COMPLETED"
        and report.get("classification") == "CLASS_GUARDED_RESPONSE_WITH_TRADEOFF",
        "explicit class-guard parent is not the declared valid attempt_03 result",
    )
    shared_dir = _resolve(root, population["shared_direction_directory"])
    inputs = class_guard._load_inputs(root, shared_dir)
    _require(
        inputs["source_indices_600"].tolist() == population["source_row_indices"]
        and inputs["labels_600"].tolist() == population["labels"]
        and inputs["roles_600"] == population["roles"],
        "class-guard input population differs from its declared upstream TRAIN population",
    )
    for key, field in (
        ("parent_ce_directory", "artifact_dir"),
        ("corrected_direction_directory", "direction_dir"),
        ("finite_response_directory", "finite_dir"),
    ):
        _require(_resolve(root, population[key]) == inputs[field], f"class-guard {key} differs from its resolved source")
    _require(inputs["source_indices_600"].shape == inputs["labels_600"].shape == (TOTAL_ROWS,), "fixed population must contain exactly six hundred rows")
    _require(inputs["roles_600"] == ["fit"] * FIT_ROWS + ["historic_holdout"] * HISTORIC_ROWS + ["confirmation"] * CONFIRMATION_ROWS, "fixed population roles or order changed")
    _require(torch.bincount(inputs["labels_600"], minlength=10).tolist() == [60] * 10, "fixed population must have sixty rows per class")
    _require(inputs["directions"]["equal_rms_directions"]["S_GD"].shape == inputs["parameters"]["B_class_ce"]["generator.phase"].shape, "source direction geometry differs from frozen Generator phase")
    return {**inputs, "class_guard_dir": class_guard_dir, "class_guard_population": population, "class_guard_report": report}


def _population_record(inputs: Mapping[str, Any], output: Path) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "class_guard_directory": inputs["class_guard_dir"].as_posix(),
        "shared_direction_directory": inputs["shared_dir"].as_posix(),
        "parent_ce_directory": inputs["artifact_dir"].as_posix(),
        "corrected_direction_directory": inputs["direction_dir"].as_posix(),
        "finite_response_directory": inputs["finite_dir"].as_posix(),
        "checkpoint": inputs["paths"]["checkpoint_B_class_ce"].as_posix(),
        "source_row_indices": inputs["source_indices_600"].tolist(),
        "labels": inputs["labels_600"].tolist(),
        "roles": list(inputs["roles_600"]),
        "role_counts": {"fit": FIT_ROWS, "historic_holdout": HISTORIC_ROWS, "confirmation": CONFIRMATION_ROWS},
        "row_order": "parent attempt_01 original order",
        "direction_rows": "first 200 fit rows only",
        "confirmation_used_for_gradient_or_projection": False,
        "e_d_used_for_direction_construction": False,
        "guard_constraints": [f"{name}:class{class_id}:CE" for name, class_id in GUARD_PAIRS]
        + [f"{name}:class{class_id}:margin" for name, class_id in GUARD_PAIRS],
        "candidate_scale": "projected vector rescaled to source S-GD reference RMS",
        "conditions": ["baseline", "baseline_repeat", "ce_margin_plus1", "ce_margin_minus1"],
        "margin_definition": direction.MARGIN_DEFINITION,
        "output_directory": output.as_posix(),
    }


def _identity(inputs: Mapping[str, Any], population: Mapping[str, Any], output: Path) -> ResearchRunIdentity:
    payload = inputs["payloads"]["B_class_ce"]
    attempt = output.name.replace("_", "-")
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-{attempt}",
        trajectory_id="frozen-B-step2500-classwise-CE-and-margin-guarded-Generator-displacement",
        semantic_authority_id=TASK_ID,
        data_role="fixed 600 canonical TRAIN rows; first 200 fit only, last 300 confirmation only",
        config_identity={
            "propagation_grid": 2048,
            "encoder_adam_step": 2200,
            "generator_adam_step": 2500,
            "fit_rows": FIT_ROWS,
            "historic_rows": HISTORIC_ROWS,
            "confirmation_rows": CONFIRMATION_ROWS,
            "guard_constraints": 60,
            "projection": "float64 nearest feasible point; unit-L2 nonzero rows; KKT tolerance 1e-8",
            "candidate_rms": float(inputs["directions"]["reference_rms"]),
            "margin_definition": direction.MARGIN_DEFINITION,
        },
        model_identity={
            "parameter_shapes": {name: list(value.shape) for name, value in payload["trainable_parameter_state"].items()},
            "source_arm": "B_class_ce",
            "completed_updates": 300,
        },
        optimizer_identity={"class": "saved Adam state read only; no optimizer constructed or stepped", "source_steps": payload["optimizer_step_by_parameter"]},
        sampler_identity={"source_indices": population["source_row_indices"], "labels": population["labels"], "roles": population["roles"]},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:{attempt}",
    )


def _saved_ce_gradients(inputs: Mapping[str, Any]) -> dict[str, Any]:
    path = inputs["class_guard_dir"] / "class_guard_gradients.pt"
    bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    _require(bundle.get("schema") == class_guard.GRADIENT_SCHEMA, "saved class CE gradients have unknown semantics")
    _require(set(bundle.get("gradients", {})) == set(GUARD_EVALUATORS), "saved class CE gradient evaluators differ")
    _require(bundle.get("gradient_definition") == "mean CE over twenty fit rows; equal mean of two ordered ten-row batch gradients", "saved class CE gradient definition changed")
    class_guard._validate_gradient_population(bundle, inputs["labels_600"][:FIT_ROWS], inputs["source_indices_600"][:FIT_ROWS])
    agreement = class_guard._validate_class_gradient_means(bundle, inputs["direction_blocks"])
    _require(all(item["absolute_l2_error"] <= item["allowed_l2_error"] for item in agreement.values()), "saved CE gradients fail the independent mixed-batch mean check")
    return {"bundle": bundle, "mixed_batch_reference_agreement": agreement}


def _batch_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
    fixed_competitor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return full-logit true-minus-wrong margins and explicit tie metadata."""
    _require(logits.ndim == 2 and labels.shape == (logits.shape[0],), "margin inputs have incompatible shapes")
    rows = torch.arange(logits.shape[0], device=logits.device)
    if fixed_competitor is None:
        detached_wrong = logits.detach().clone()
        detached_wrong[rows, labels] = -torch.inf
        best_value = detached_wrong.max(dim=1).values
        tied = detached_wrong.eq(best_value[:, None]).sum(dim=1) > 1
        competitor = detached_wrong.argmax(dim=1)
    else:
        competitor = fixed_competitor.to(device=logits.device, dtype=torch.long)
        tied = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
    margins = logits[rows, labels] - logits[rows, competitor]
    return margins, competitor, tied, logits[rows, labels]


def _score_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    fixed_competitor: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Score production float32 logits in float64 while preserving autograd."""
    _require(logits.ndim == 2 and labels.shape == (logits.shape[0],), "score logits and labels have incompatible shapes")
    labels = labels.to(device=logits.device, dtype=torch.long)
    score_logits = logits.to(dtype=torch.float64)
    ce = F.cross_entropy(score_logits, labels, reduction="none")
    margin, competitor, tied, true_logits = _batch_margin(score_logits, labels)
    if fixed_competitor is None:
        fixed_margin = margin
    else:
        fixed_margin = _batch_margin(score_logits, labels, fixed_competitor)[0]
    return {
        "ce": ce,
        "margin": margin,
        "fixed_margin": fixed_margin,
        "actual_competitor": competitor,
        "actual_tied": tied,
        "true_logit": true_logits,
    }


def _paired_finite_difference(plus: torch.Tensor, minus: torch.Tensor, epsilon: float) -> tuple[float, torch.Tensor]:
    _require(plus.shape == minus.shape and plus.ndim == 1 and epsilon > 0.0, "paired finite-difference scores are malformed")
    paired_slopes = (plus.detach().to(device="cpu", dtype=torch.float64) - minus.detach().to(device="cpu", dtype=torch.float64)) / (2.0 * epsilon)
    return float(paired_slopes.mean()), paired_slopes


def _forward_evaluator(
    evaluator: torch.nn.Module,
    view: torch.Tensor,
    *,
    capture_branches: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not capture_branches:
        output = evaluator(view)
        return (output if torch.is_tensor(output) else output[1]), {}

    trace: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    counters = {"relu": 0, "maxpool2d": 0}
    for module in evaluator.modules():
        if isinstance(module, torch.nn.ReLU):
            key = f"relu_{counters['relu']:03d}"
            counters["relu"] += 1

            def record_relu(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor, *, name: str = key) -> None:
                trace[name] = output.detach().gt(0).to(device="cpu")

            handles.append(module.register_forward_hook(record_relu))
        elif isinstance(module, torch.nn.MaxPool2d):
            key = f"maxpool2d_{counters['maxpool2d']:03d}"
            counters["maxpool2d"] += 1

            def record_pool(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor, *, name: str = key) -> None:
                values = F.max_pool2d(
                    inputs[0].detach(),
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    ceil_mode=module.ceil_mode,
                    return_indices=True,
                )[1]
                trace[name] = values.to(device="cpu")

            handles.append(module.register_forward_hook(record_pool))
    try:
        output = evaluator(view)
    finally:
        for handle in handles:
            handle.remove()
    return (output if torch.is_tensor(output) else output[1]), trace


def _numeric_settings() -> dict[str, Any]:
    try:
        autocast_cuda = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        autocast_cuda = bool(torch.is_autocast_enabled())
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_autocast_enabled": autocast_cuda,
        "torch_version": str(torch.__version__),
    }


@contextlib.contextmanager
def _strict_float32_settings():
    previous = _numeric_settings()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        with torch.autocast(device_type="cuda", enabled=False):
            yield previous
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]
        torch.set_float32_matmul_precision(previous["float32_matmul_precision"])


def _mean_margin_gradients(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    generator_phase: torch.nn.Parameter,
    images: torch.Tensor,
    labels: torch.Tensor,
    source_indices: torch.Tensor,
) -> dict[str, Any]:
    _require(images.shape[0] == labels.shape[0] == source_indices.shape[0] == FIT_ROWS, "margin gradients must consume exactly the two hundred fit rows")
    values: dict[str, dict[int, torch.Tensor]] = {name: {} for name in GUARD_EVALUATORS}
    row_maps: dict[str, dict[str, Any]] = {name: {} for name in GUARD_EVALUATORS}
    stats: dict[str, dict[str, Any]] = {name: {} for name in GUARD_EVALUATORS}
    independent_errors: dict[str, dict[str, float]] = {}
    exact_ties: list[dict[str, Any]] = []
    for evaluator_name in GUARD_EVALUATORS:
        independent_errors[evaluator_name] = {}
        for class_id in range(10):
            positions = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)
            _require(positions.numel() == 20, f"{evaluator_name} class {class_id} must have twenty fit rows")
            pieces: list[torch.Tensor] = []
            batch_values: list[torch.Tensor] = []
            competitor_by_position: dict[int, int] = {}
            tie_positions: list[int] = []
            component_diffs: list[tuple[float, float, float]] = []
            for batch_positions in (positions[:10], positions[10:]):
                selected_images = images.index_select(0, batch_positions).to(consumer.device)
                selected_labels = labels.index_select(0, batch_positions).to(consumer.device)
                with torch.inference_mode(False):
                    breakdown = consumer.forward_loss(selected_images, data_role=direction.bogt.TRAIN_ROLE)
                    view = canonical.cmos_view(breakdown.result.raw_intensity)
                    logits = evaluators[evaluator_name](view)[1]
                    margins, competitors, tied, true_logits = _batch_margin(logits, selected_labels)
                    batch_values.append(margins.detach().cpu())
                    for local_index, position in enumerate(batch_positions.tolist()):
                        competitor_by_position[int(position)] = int(competitors[local_index].item())
                        if bool(tied[local_index]):
                            tie_positions.append(int(position))
                            exact_ties.append({
                                "evaluator": evaluator_name,
                                "class_id": class_id,
                                "fit_position": int(position),
                                "source_row_index": int(source_indices[position]),
                                "margin": float(margins[local_index].detach().cpu()),
                                "max_wrong_classes": torch.nonzero(
                                    logits[local_index].detach().masked_fill(
                                        F.one_hot(selected_labels[local_index], num_classes=logits.shape[1]).bool(), -torch.inf
                                    ) == logits[local_index].detach().masked_fill(
                                        F.one_hot(selected_labels[local_index], num_classes=logits.shape[1]).bool(), -torch.inf
                                    ).max(),
                                    as_tuple=False,
                                ).reshape(-1).tolist(),
                            })
                    if not bool(tied.any()):
                        margin_gradient = torch.autograd.grad(margins.mean(), generator_phase, retain_graph=class_id in (1, 7, 8, 9), create_graph=False)[0]
                        pieces.append(margin_gradient.detach().to(device="cpu", dtype=torch.float32))
                        if class_id in (1, 7, 8, 9):
                            wrong_rows = torch.arange(logits.shape[0], device=logits.device)
                            wrong_logits = logits[wrong_rows, competitors]
                            true_mean = true_logits.mean()
                            wrong_mean = wrong_logits.mean()
                            component_reference = torch.autograd.grad(
                                (true_mean, wrong_mean),
                                generator_phase,
                                grad_outputs=(torch.ones_like(true_mean), -torch.ones_like(wrong_mean)),
                                retain_graph=False,
                                create_graph=False,
                            )[0]
                            error = (margin_gradient.detach().to(torch.float64) - component_reference.detach().to(torch.float64)).norm()
                            reference = component_reference.detach().to(torch.float64).norm()
                            allowed = DERIVATIVE_RELATIVE_TOLERANCE * reference + 1e-12
                            component_diffs.append((float(error), float(reference), float(allowed)))
                del selected_images, selected_labels, breakdown, view, logits, margins, competitors, tied, true_logits
            row_maps[evaluator_name][str(class_id)] = {
                "fit_positions": positions.tolist(),
                "source_row_indices": source_indices[positions].tolist(),
                "max_wrong_classes_in_fit_order": [competitor_by_position[int(position)] for position in positions.tolist()],
                "exact_tie_fit_positions": tie_positions,
            }
            stats[evaluator_name][str(class_id)] = {
                "mean_margin": float(torch.cat(batch_values).to(torch.float64).mean()),
                "rows": int(positions.numel()),
                "exact_tie_count": len(tie_positions),
            }
            if tie_positions:
                continue
            _require(len(pieces) == 2, f"{evaluator_name} class {class_id} did not produce both batch gradients")
            values[evaluator_name][class_id] = torch.stack(pieces).mean(0)
            if component_diffs:
                key = str(class_id)
                independent_errors[evaluator_name][key] = {
                    "maximum_absolute_l2_error": max(item[0] for item in component_diffs),
                    "maximum_allowed_l2_error": max(item[2] for item in component_diffs),
                    "passed": all(item[0] <= item[2] for item in component_diffs),
                }
    return {
        "schema": GRADIENT_SCHEMA,
        "gradient_definition": "mean full-autograd true-logit minus unique maximum wrong-logit over twenty fit rows; equal mean of two ordered ten-row batch gradients",
        "gradients": values,
        "fit_row_map": row_maps,
        "fit_mean_margins": stats,
        "exact_ties": exact_ties,
        "unique_max_wrong_component_checks": independent_errors,
        "component_reference_method": "one two-output VJP with +1 weight on the true-logit mean and -1 weight on the baseline unique-max-wrong-logit mean; independently consumes both live logits",
        "candidate_unavailable_for_ties": bool(exact_ties),
    }


def _validate_margin_population(bundle: Mapping[str, Any], labels: torch.Tensor, source_indices: torch.Tensor) -> None:
    _require(labels.shape == source_indices.shape == (FIT_ROWS,), "margin-gradient population must cover exactly two hundred fit rows")
    _require(bundle.get("schema") == GRADIENT_SCHEMA, "margin gradients have unknown semantics")
    row_map = bundle.get("fit_row_map", {})
    _require(set(row_map) == set(GUARD_EVALUATORS), "margin-gradient evaluator map is incomplete")
    for evaluator in GUARD_EVALUATORS:
        _require(set(row_map[evaluator]) == {str(class_id) for class_id in range(10)}, f"{evaluator} margin-gradient class map is incomplete")
        for class_id in range(10):
            expected = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)
            saved = row_map[evaluator][str(class_id)]
            _require(saved.get("fit_positions") == expected.tolist(), f"{evaluator} class {class_id} margin gradient uses the wrong fit-row mask")
            _require(saved.get("source_row_indices") == source_indices[expected].tolist(), f"{evaluator} class {class_id} margin-gradient source rows changed")
            tie_positions = saved.get("exact_tie_fit_positions", [])
            _require(set(tie_positions).issubset(set(expected.tolist())), f"{evaluator} class {class_id} exact-tie rows are outside the fit population")
            has_gradient = class_id in bundle.get("gradients", {}).get(evaluator, {})
            _require(has_gradient == (not tie_positions), f"{evaluator} class {class_id} tie marker and stored gradient disagree")


def _constraint_rows(
    ce_gradients: Mapping[str, Mapping[int, torch.Tensor]],
    margin_gradients: Mapping[str, Mapping[int, torch.Tensor]],
) -> tuple[torch.Tensor, list[str], dict[str, float], list[str]]:
    names: list[str] = []
    rows: list[torch.Tensor] = []
    norms: dict[str, float] = {}
    zero: list[str] = []
    first = next(iter(ce_gradients["E_S"].values()), None)
    if first is None:
        first = next(iter(margin_gradients["E_S"].values()), None)
    _require(torch.is_tensor(first), "no differentiable fit constraint gradient is available")
    shape = first.shape
    for evaluator in GUARD_EVALUATORS:
        for class_id in range(10):
            for kind, bundle, sign in (("CE", ce_gradients, 1.0), ("margin", margin_gradients, -1.0)):
                name = f"{evaluator}:class{class_id}:{kind}"
                _require(class_id in bundle.get(evaluator, {}), f"constraint gradient is missing for {name}")
                gradient = bundle[evaluator][class_id].detach().to(device="cpu", dtype=torch.float64)
                _require(gradient.shape == shape and bool(torch.isfinite(gradient).all()), f"constraint gradient {name} has invalid shape or values")
                signed = gradient.reshape(-1) * sign
                norm = float(signed.norm())
                norms[name] = norm
                if norm == 0.0:
                    zero.append(name)
                    continue
                names.append(name)
                rows.append(signed / norm)
    return (torch.stack(rows) if rows else torch.empty((0, first.numel()), dtype=torch.float64)), names, norms, zero


def _verify_candidate_guards(
    candidate: torch.Tensor,
    ce_gradients: Mapping[str, Mapping[int, torch.Tensor]],
    margin_gradients: Mapping[str, Mapping[int, torch.Tensor]],
    tolerance: float,
) -> dict[str, dict[str, float]]:
    displacement = candidate.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    products: dict[str, dict[str, float]] = {"ce": {}, "margin": {}}
    for evaluator in GUARD_EVALUATORS:
        for class_id in range(10):
            for kind, bundle, should_be_nonpositive in (
                ("ce", ce_gradients, True),
                ("margin", margin_gradients, False),
            ):
                gradient = bundle[evaluator][class_id].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
                norm = float(gradient.norm())
                if norm == 0.0:
                    continue
                value = float(torch.dot(gradient / norm, displacement))
                valid = value <= tolerance if should_be_nonpositive else value >= -tolerance
                _require(valid, f"equal-RMS candidate violates the independently checked {kind} guard for {evaluator} class {class_id}")
                products[kind][f"{evaluator}:class{class_id}"] = value
    return products


def _dual_project(source: torch.Tensor, rows: torch.Tensor) -> dict[str, Any]:
    """Project onto homogeneous halfspaces with the production dual solver."""
    source64 = source.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    matrix = rows.detach().to(device="cpu", dtype=torch.float64).reshape(-1, source64.numel())
    _require(bool(torch.isfinite(source64).all()) and source64.numel() > 0, "projection source is malformed")
    _require(bool(torch.isfinite(matrix).all()), "projection constraints contain nonfinite values")
    if matrix.shape[0]:
        gram = matrix @ matrix.T
        linear = matrix @ source64
        diagonal = torch.diagonal(gram)
        _require(bool((diagonal > 0.0).all()), "zero constraints must be removed before projection")
        multipliers = torch.zeros(matrix.shape[0], dtype=torch.float64)
        for sweep in range(1, 1_000_001):
            for index in range(matrix.shape[0]):
                residual = linear[index] - torch.dot(gram[index], multipliers)
                multipliers[index] = max(0.0, float(multipliers[index] + residual / diagonal[index]))
            projected = source64 - matrix.T @ multipliers
            products = matrix @ projected
            primal = float(torch.clamp(products, min=0.0).max())
            complementarity = float((multipliers * products).abs().max())
            stationarity = float((projected - source64 + matrix.T @ multipliers).abs().max())
            if max(primal, complementarity, stationarity) <= KKT_TOLERANCE:
                break
        else:
            raise CEMarginDirectionError("sixty-constraint nearest-point projection did not converge to its KKT tolerance")
    else:
        projected = source64.clone()
        multipliers = torch.empty(0, dtype=torch.float64)
        products = torch.empty(0, dtype=torch.float64)
        sweep = 0
        primal = complementarity = stationarity = 0.0
    return {
        "projected": projected,
        "multipliers": multipliers,
        "products": products,
        "sweeps": sweep,
        "residuals": {"primal_feasibility": primal, "complementarity": complementarity, "stationarity": stationarity, "maximum": max(primal, complementarity, stationarity), "tolerance": KKT_TOLERANCE},
    }


def _project(source: torch.Tensor, ce_gradients: Mapping[str, Mapping[int, torch.Tensor]], margin_gradients: Mapping[str, Mapping[int, torch.Tensor]], reference_rms: float) -> dict[str, Any]:
    source64 = source.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    _require(bool(torch.isfinite(source64).all()) and source64.numel() > 0, "pure-CE source displacement is malformed")
    tie_candidates = any(not set(range(10)).issubset(set(margin_gradients.get(name, {}))) for name in GUARD_EVALUATORS)
    if tie_candidates:
        missing = {
            evaluator: sorted(set(range(10)) - set(margin_gradients.get(evaluator, {})))
            for evaluator in GUARD_EVALUATORS
            if not set(range(10)).issubset(set(margin_gradients.get(evaluator, {})))
        }
        return {
            "schema": PROJECTION_SCHEMA,
            "candidate_available": False,
            "candidate_equal_rms_direction": None,
            "candidate_unavailable_reason": "an exact maximum-wrong-logit tie makes at least one fit margin derivative undefined",
            "exact_tie_exit": True,
            "missing_margin_gradient_classes": missing,
            "source_rms": _rms(source),
            "reference_rms": reference_rms,
        }
    rows, names, raw_norms, zero = _constraint_rows(ce_gradients, margin_gradients)
    solved = _dual_project(source, rows)
    projected = solved["projected"]
    multipliers = solved["multipliers"]
    all_products = solved["products"]
    source_norm = float(source64.norm())
    projected_norm = float(projected.norm())
    retained = projected_norm / source_norm if source_norm else 0.0
    candidate_unavailable = source_norm == 0.0 or retained <= ZERO_DIRECTION_RELATIVE or projected_norm == 0.0
    if candidate_unavailable:
        candidate = None
        scaled_products: list[float] = []
    else:
        projected_rms = _rms(projected.reshape_as(source))
        scale_factor = reference_rms / projected_rms
        candidate = projected * scale_factor
        candidate = candidate.reshape_as(source).to(dtype=source.dtype).contiguous()
        candidate64 = candidate.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        scaled_products_tensor = rows @ candidate64
        scaled_products = scaled_products_tensor.tolist()
        rounding_bound = 0.0
        if rows.shape[0] and candidate.dtype.is_floating_point:
            rounding_bound = float(torch.finfo(candidate.dtype).eps * (rows.abs() @ candidate64.abs()).max())
        scaled_allowed = max(1e-12, KKT_TOLERANCE * scale_factor * 2.0 + rounding_bound)
        max_positive = max(scaled_products, default=0.0)
        _require(max_positive <= scaled_allowed, f"equal-RMS candidate violates a normalized fit constraint after scaling: max_positive={max_positive:.9g}, allowed={scaled_allowed:.9g}, scale_factor={scale_factor:.9g}, rounding_bound={rounding_bound:.9g}")
        guard_products = _verify_candidate_guards(candidate, ce_gradients, margin_gradients, scaled_allowed)
    all_products = rows @ projected if rows.shape[0] else torch.empty(0, dtype=torch.float64)
    multipliers_map = {name: float(value) for name, value in zip(names, multipliers.tolist(), strict=True)}
    product_map = {name: float(value) for name, value in zip(names, all_products.tolist(), strict=True)}
    return {
        "schema": PROJECTION_SCHEMA,
        "candidate_available": candidate is not None,
        "candidate_equal_rms_direction": candidate,
        "projected_minimum_distance_vector": projected.reshape_as(source),
        "candidate_unavailable_reason": "projection is within the declared numerical zero range" if candidate is None else None,
        "exact_tie_exit": False,
        "constraint_names": names,
        "zero_gradient_constraints": zero,
        "raw_constraint_gradient_norms": raw_norms,
        "unit_constraint_products_at_projection": product_map,
        "multipliers": multipliers_map,
        "unit_constraint_products_after_equal_rms": dict(zip(names, scaled_products, strict=True)),
        "independent_guard_products_after_equal_rms": guard_products if candidate is not None else {"ce": {}, "margin": {}},
        "kkt_residuals": solved["residuals"],
        "solver": "deterministic cyclic nonnegative dual coordinate descent in float64",
        "solver_sweeps": solved["sweeps"],
        "source_rms": _rms(source),
        "projected_vector_rms": _rms(projected.reshape_as(source)),
        "reference_rms": reference_rms,
        "candidate_rms": None if candidate is None else _rms(candidate),
        "projected_to_source_l2_ratio": retained,
        "direction_cosine": float(torch.dot(projected, source64) / (projected.norm() * source64.norm())) if projected_norm and source_norm else None,
        "scaled_feasibility_tolerance": scaled_allowed if candidate is not None else None,
        "projection_rescale_factor": reference_rms / _rms(projected.reshape_as(source)) if candidate is not None else None,
    }


def _cpu_references() -> dict[str, Any]:
    epsilon = 1e-5
    phi = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([0.0, 1.0, 0.9], dtype=torch.float64)
    slopes = torch.tensor([0.0, 0.01, -0.1], dtype=torch.float64)
    logits = offsets + phi * slopes
    label = torch.tensor([0])
    ce = F.cross_entropy(logits.reshape(1, 3), label)
    margin, wrong, tied, _ = _batch_margin(logits.reshape(1, 3), label)
    ce_gradient = torch.autograd.grad(ce, phi, retain_graph=True)[0]
    margin_gradient = torch.autograd.grad(margin.mean(), phi)[0]
    with torch.no_grad():
        ce_plus = F.cross_entropy((offsets + epsilon * slopes).reshape(1, 3), label)
        ce_minus = F.cross_entropy((offsets - epsilon * slopes).reshape(1, 3), label)
        margin_plus = (offsets[0] + epsilon * slopes[0]) - (offsets[1] + epsilon * slopes[1])
        margin_minus = (offsets[0] - epsilon * slopes[0]) - (offsets[1] - epsilon * slopes[1])
    ce_fd = float((ce_plus - ce_minus) / (2 * epsilon))
    margin_fd = float((margin_plus - margin_minus) / (2 * epsilon))
    _require(not bool(tied[0]) and int(wrong[0]) == 1, "independent CE/margin reference did not select a unique wrong class")
    _require(float(ce_gradient) < 0.0 and float(margin_gradient) < 0.0, "CE-decrease/margin-decrease counterexample did not retain its opposing sign")
    _require(abs(float(ce_gradient) - ce_fd) < 1e-8 and abs(float(margin_gradient) - margin_fd) < 1e-8, "analytic CE or margin derivative differs from central difference")

    common_shift = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    shifted = torch.tensor([0.0, 1.0, -5.0], dtype=torch.float64) + common_shift
    common_margin, _, _, _ = _batch_margin(shifted.reshape(1, 3), label)
    common_derivative = torch.autograd.grad(common_margin.sum(), common_shift)[0]
    _require(abs(float(common_derivative)) < 1e-12, "a common logit shift changed the true-minus-wrong margin")

    tie_logits = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float64)
    _, _, tie_flag, _ = _batch_margin(tie_logits, label)
    _require(bool(tie_flag[0]), "exact wrong-logit tie was not preserved")

    source = torch.tensor([0.8, -0.3], dtype=torch.float64)
    constraint = torch.tensor([[1.0, 0.0], [0.6, 0.8], [-0.6, 0.8]], dtype=torch.float64)
    projected = _dual_project(source, constraint)["projected"]
    active_reference = class_guard._projection_active_subset_reference(source, constraint)
    _require(torch.allclose(projected, active_reference, atol=1e-10, rtol=1e-10), "independent active-set projection reference is inconsistent")
    return {
        "passed": True,
        "ce_decrease_margin_decrease": {"ce_derivative": float(ce_gradient), "ce_central_difference": ce_fd, "margin_derivative": float(margin_gradient), "margin_central_difference": margin_fd, "unique_wrong_class": int(wrong[0])},
        "common_logit_shift_margin_derivative": float(common_derivative),
        "exact_tie_preserved": True,
        "small_projection_reference": {"candidate": projected.tolist(), "active_set_reference": active_reference.tolist()},
    }


def _snapshot_checks(consumer: Any, evaluators: Mapping[str, torch.nn.Module], parameters: Mapping[str, torch.nn.Parameter], runtime_before: Mapping[str, Any], evaluator_before: Mapping[str, Mapping[str, Any]], frozen_inputs_before: Mapping[str, Any], optimizer_before: Any, inputs: Mapping[str, Any]) -> dict[str, bool]:
    unchanged = finite._tree_equal(runtime_before, finite._snapshot(consumer.runtime))
    inputs_unchanged = finite._tree_equal(frozen_inputs_before, direction._consumer_frozen_inputs_snapshot(consumer))
    adam_unchanged = finite._tree_equal(optimizer_before, direction._cpu_tree(inputs["optimizers"]["B_class_ce"]))
    evaluators_unchanged = all(finite._tree_equal(evaluator_before[name], finite._snapshot(model)) for name, model in evaluators.items())
    parameters_frozen = all(parameter.grad is None and not parameter.requires_grad for parameter in parameters.values())
    evaluator_frozen = all(parameter.grad is None and not parameter.requires_grad for model in evaluators.values() for parameter in model.parameters())
    result = {"model_and_scientific_buffers_unchanged": unchanged, "frozen_inputs_unchanged": inputs_unchanged, "source_adam_unchanged": adam_unchanged, "evaluators_unchanged": evaluators_unchanged, "parameters_frozen_and_grad_free": parameters_frozen, "evaluators_frozen_and_grad_free": evaluator_frozen}
    _require(all(result.values()), "a frozen model, reference, optimizer, or evaluator invariant changed")
    return result


def _target_controls(evaluators: Mapping[str, torch.nn.Module], inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return class_guard._target_controls(evaluators, inputs, device)


def _fixed_path_checks(consumer: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    with torch.inference_mode(False), torch.no_grad():
        for role, position in (("fit", 0), ("confirmation", FIT_ROWS + HISTORIC_ROWS)):
            images = inputs["images_600"][position:position + BATCH_SIZE].to(consumer.device)
            encoded = consumer.runtime.forward_from_image(images)
            expected = inputs["readouts"]["B_class_ce"]["z"][position:position + BATCH_SIZE].to(consumer.device)
            decoded = consumer.runtime.forward_from_ols(expected)
            _require(torch.allclose(encoded.z_ols.coefficients, expected, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{role} Encoder OLS differs from the saved B readout")
            _require(torch.allclose(encoded.raw_intensity, decoded.raw_intensity, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{role} Encoder and cached OLS decode paths differ")
            checks[role] = {"source_rows": inputs["source_indices_600"][position:position + BATCH_SIZE].tolist(), "encoder_ols_matches_saved": True, "encoder_and_fixed_ols_cmos_match": True}
    return checks


def _strict_baseline_replay(consumer: Any, evaluators: Mapping[str, torch.nn.Module], inputs: Mapping[str, Any]) -> dict[str, Any]:
    chunk_results: list[dict[str, Any]] = []
    for start in range(0, TOTAL_ROWS, CHUNK_SIZE):
        try:
            records = [shared._read_fixed_batch(consumer, evaluators, inputs, batch_start) for batch_start in range(start, start + CHUNK_SIZE, BATCH_SIZE)]
            block = shared._concat_batch_records(records, start, "baseline")
            errors = shared._verify_baseline_block(block, inputs)
            chunk_results.append({"row_start": start, "row_stop": start + CHUNK_SIZE, "passed": True, "maximum_absolute_errors": errors})
        except (CEMarginDirectionError, shared.SharedDirectionError, finite.FiniteResponseError, ValueError) as error:
            chunk_results.append({"row_start": start, "row_stop": start + CHUNK_SIZE, "passed": False, "error": f"{type(error).__name__}: {error}"})
    return {
        "passed": len(chunk_results) == TOTAL_ROWS // CHUNK_SIZE and all(item["passed"] for item in chunk_results),
        "rows_checked": TOTAL_ROWS,
        "row_roles": {"fit": FIT_ROWS, "historic_holdout": HISTORIC_ROWS, "confirmation": CONFIRMATION_ROWS},
        "chunks": chunk_results,
    }


def _strict_mode_compatibility_impl(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    inputs: Mapping[str, Any],
    phase: torch.nn.Parameter,
    baseline_phase: torch.Tensor,
    ce_source: Mapping[str, Any],
) -> dict[str, Any]:
    original_settings = _numeric_settings()
    result: dict[str, Any] = {"executed": True, "mode": "strict_fp32_no_tf32", "strict_mode_comparison_is_not_itself_ready": True}
    margin_gradients: dict[str, Any] | None = None
    projection: dict[str, Any] | None = None
    fd_checks: dict[str, Any] | None = None
    diagnostic_snapshots: dict[str, Any] = {}
    with _strict_float32_settings():
        result["strict_settings"] = _numeric_settings()
        replay = _strict_baseline_replay(consumer, evaluators, inputs)
        result["full_population_baseline_replay"] = replay
        if not replay["passed"]:
            result["passed"] = False
            result["stopped_after"] = "baseline_replay"
            return result

        try:
            path_checks = _fixed_path_checks(consumer, inputs)
            result["real_encoder_vs_fixed_ols"] = {"passed": True, "checks": path_checks}
        except (CEMarginDirectionError, shared.SharedDirectionError, ValueError) as error:
            result["real_encoder_vs_fixed_ols"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            result["passed"] = False
            result["stopped_after"] = "real_encoder_vs_fixed_ols"
            return result

        target_controls = _target_controls(evaluators, inputs, consumer.device)
        target_qualified = finite._qualify_target_controls(target_controls)
        result["target_positive_controls"] = target_controls
        result["target_positive_controls_qualified"] = target_qualified
        if not target_qualified:
            result["passed"] = False
            result["stopped_after"] = "target_positive_controls"
            return result

        strict_ce = class_guard._class_ce_gradients(
            consumer,
            evaluators,
            phase,
            inputs["images_600"][:FIT_ROWS],
            inputs["labels_600"][:FIT_ROWS],
        )
        ce_comparison: dict[str, dict[str, Any]] = {}
        for evaluator_name in GUARD_EVALUATORS:
            ce_comparison[evaluator_name] = {}
            for class_id in range(10):
                expected = ce_source["gradients"][evaluator_name][class_id].detach().to(device="cpu", dtype=torch.float64)
                actual = strict_ce["gradients"][evaluator_name][class_id].detach().to(device="cpu", dtype=torch.float64)
                error = float((actual - expected).norm())
                reference_norm = float(expected.norm())
                allowed = class_guard.MIXED_CE_GRADIENT_RELATIVE_TOLERANCE * reference_norm + 1e-12
                ce_comparison[evaluator_name][str(class_id)] = {
                    "absolute_l2_error": error,
                    "reference_l2_norm": reference_norm,
                    "allowed_l2_error": allowed,
                    "passed": error <= allowed,
                }
        ce_source_passed = all(item["passed"] for evaluator in ce_comparison.values() for item in evaluator.values())
        result["saved_ce_gradient_source_consistency"] = {
            "passed": ce_source_passed,
            "relative_tolerance": class_guard.MIXED_CE_GRADIENT_RELATIVE_TOLERANCE,
            "checks": ce_comparison,
        }
        if not ce_source_passed:
            result["passed"] = False
            result["stopped_after"] = "saved_ce_gradient_source_consistency"
            return result

        margin_gradients = _mean_margin_gradients(
            consumer,
            evaluators,
            phase,
            inputs["images_600"][:FIT_ROWS],
            inputs["labels_600"][:FIT_ROWS],
            inputs["source_indices_600"][:FIT_ROWS],
        )
        _validate_margin_population(margin_gradients, inputs["labels_600"][:FIT_ROWS], inputs["source_indices_600"][:FIT_ROWS])
        margin_gradient_passed = all(item["passed"] for evaluator in margin_gradients["unique_max_wrong_component_checks"].values() for item in evaluator.values())
        result["strict_margin_gradient_production"] = {
            "passed": margin_gradient_passed,
            "all_fit_rows_accounted_for": all(len(margin_gradients["fit_row_map"][name][str(class_id)]["fit_positions"]) == 20 for name in GUARD_EVALUATORS for class_id in range(10)),
            "definition": margin_gradients["gradient_definition"],
            "unique_max_wrong_component_checks": margin_gradients["unique_max_wrong_component_checks"],
            "exact_ties": margin_gradients["exact_ties"],
        }
        if not margin_gradient_passed:
            result["passed"] = False
            result["stopped_after"] = "strict_margin_gradient_production"
            return result

        projection = _project(
            inputs["directions"]["equal_rms_directions"]["S_GD"],
            ce_source["gradients"],
            margin_gradients["gradients"],
            float(inputs["directions"]["reference_rms"]),
        )
        result["strict_projection"] = {
            "passed": projection.get("exact_tie_exit") is True or projection["kkt_residuals"]["maximum"] <= KKT_TOLERANCE,
            "candidate_available": projection["candidate_available"],
            "kkt_residuals": projection.get("kkt_residuals"),
            "projected_to_source_l2_ratio": projection.get("projected_to_source_l2_ratio"),
            "reference_rms": projection["reference_rms"],
            "candidate_rms": projection.get("candidate_rms"),
        }
        if not result["strict_projection"]["passed"]:
            result["passed"] = False
            result["stopped_after"] = "strict_projection"
            return result
        fd_checks = _finite_differences(
            consumer,
            evaluators,
            inputs,
            phase,
            baseline_phase,
            projection["candidate_equal_rms_direction"],
            ce_source["gradients"],
            margin_gradients,
        )
        diagnostic_snapshots = fd_checks.pop("diagnostic_snapshots", {})
        if not fd_checks.get("passed"):
            result["_diagnostic_snapshots"] = diagnostic_snapshots
        result["strict_selected_mode_finite_difference_checks"] = fd_checks
        result["passed"] = fd_checks.get("passed") is True and (
            fd_checks.get("not_applicable") is True
            or (fd_checks.get("executed_check_count") == 48 and len(fd_checks.get("checks", {})) == 48)
        )
        if not result["passed"]:
            result["stopped_after"] = "strict_selected_mode_finite_difference_checks"
    result["settings_before"] = original_settings
    result["settings_after_restore"] = _numeric_settings()
    result["settings_restored"] = result["settings_before"] == result["settings_after_restore"]
    result["passed"] = bool(result.get("passed")) and result["settings_restored"]
    result["strict_margin_gradients"] = margin_gradients
    result["strict_projection_payload"] = projection
    return result


def _strict_mode_compatibility(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    inputs: Mapping[str, Any],
    phase: torch.nn.Parameter,
    baseline_phase: torch.Tensor,
    ce_source: Mapping[str, Any],
) -> dict[str, Any]:
    settings_before = _numeric_settings()
    try:
        result = _strict_mode_compatibility_impl(consumer, evaluators, inputs, phase, baseline_phase, ce_source)
    except (CEMarginDirectionError, class_guard.ClassGuardedDirectionError, shared.SharedDirectionError, finite.FiniteResponseError, ValueError) as error:
        result = {
            "executed": True,
            "mode": "strict_fp32_no_tf32",
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "strict_mode_comparison_is_not_itself_ready": True,
        }
    settings_after = _numeric_settings()
    result["settings_before"] = settings_before
    result["settings_after_restore"] = settings_after
    result["settings_restored"] = settings_before == settings_after
    result["passed"] = bool(result.get("passed")) and result["settings_restored"]
    return result


def _objective_batch(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
    fixed_competitors: Mapping[str, torch.Tensor] | None = None,
    *,
    capture_diagnostics: bool = False,
) -> dict[str, Any]:
    output: dict[str, dict[str, list[torch.Tensor]]] = {name: {"ce": [], "margin": [], "actual_competitor": [], "actual_tied": [], "fixed_margin": [], "logits": []} for name in GUARD_EVALUATORS}
    raw_pieces: list[torch.Tensor] = []
    view_pieces: list[torch.Tensor] = []
    branch_pieces: dict[str, dict[str, list[torch.Tensor]]] = {name: {} for name in GUARD_EVALUATORS}
    for start in range(0, images.shape[0], BATCH_SIZE):
        selected_images = images[start:start + BATCH_SIZE].to(consumer.device)
        selected_labels = labels[start:start + BATCH_SIZE].to(consumer.device)
        with torch.inference_mode(False), torch.no_grad():
            breakdown = consumer.forward_loss(selected_images, data_role=direction.bogt.TRAIN_ROLE)
            raw = breakdown.result.raw_intensity
            view = canonical.cmos_view(raw)
            if capture_diagnostics:
                raw_pieces.append(raw.detach().cpu())
                view_pieces.append(view.detach().cpu())
            for name in GUARD_EVALUATORS:
                logits, branch_trace = _forward_evaluator(evaluators[name], view, capture_branches=capture_diagnostics)
                scores = _score_logits(logits, selected_labels)
                output[name]["ce"].append(scores["ce"].detach().cpu())
                output[name]["margin"].append(scores["margin"].detach().cpu())
                output[name]["actual_competitor"].append(scores["actual_competitor"].detach().cpu())
                output[name]["actual_tied"].append(scores["actual_tied"].detach().cpu())
                output[name]["logits"].append(logits.detach().cpu())
                if fixed_competitors is not None:
                    fixed = fixed_competitors[name][start:start + len(selected_labels)].to(consumer.device)
                    fixed_margin = _score_logits(logits, selected_labels, fixed)["fixed_margin"]
                    output[name]["fixed_margin"].append(fixed_margin.detach().cpu())
                else:
                    output[name]["fixed_margin"].append(scores["margin"].detach().cpu())
                if capture_diagnostics:
                    for key, value in branch_trace.items():
                        branch_pieces[name].setdefault(key, []).append(value)
        del selected_images, selected_labels, breakdown, raw, view, logits
    result: dict[str, Any] = {name: {key: torch.cat(pieces) for key, pieces in values.items()} for name, values in output.items()}
    if capture_diagnostics:
        result["diagnostics"] = {
            "raw": torch.cat(raw_pieces),
            "view": torch.cat(view_pieces),
            "evaluators": {
                name: {
                    "logits": result[name]["logits"],
                    "actual_competitor": result[name]["actual_competitor"],
                    "actual_tied": result[name]["actual_tied"],
                    "branch_signatures": {key: torch.cat(pieces) for key, pieces in branch_pieces[name].items()},
                }
                for name in GUARD_EVALUATORS
            },
        }
        raw = result["diagnostics"]["raw"]
        result["diagnostics"]["clamp_signatures"] = {
            "raw_below_zero": raw.le(0),
            "raw_at_zero": raw.eq(0),
            "raw_above_one": raw.gt(1),
            "raw_at_one": raw.eq(1),
        }
    return result


def _branch_change_summary(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in GUARD_EVALUATORS:
        first = before["evaluators"][name]["branch_signatures"]
        second = after["evaluators"][name]["branch_signatures"]
        _require(set(first) == set(second), f"{name} nonlinear branch inventory changed during finite-difference tracing")
        summary[name] = {}
        for key in sorted(first):
            left, right = first[key], second[key]
            _require(left.shape == right.shape and left.shape[0] == FIT_ROWS // 10, f"{name} branch trace {key} changed shape")
            summary[name][key] = {
                "changed_elements_per_row": (left.ne(right).reshape(left.shape[0], -1).sum(dim=1)).tolist(),
                "rows_with_any_change": int(left.ne(right).reshape(left.shape[0], -1).any(dim=1).sum()),
            }
    clamp: dict[str, Any] = {}
    for key in before["clamp_signatures"]:
        left, right = before["clamp_signatures"][key], after["clamp_signatures"][key]
        clamp[key] = {
            "changed_elements_per_row": left.ne(right).reshape(left.shape[0], -1).sum(dim=1).tolist(),
            "rows_with_any_change": int(left.ne(right).reshape(left.shape[0], -1).any(dim=1).sum()),
        }
    return {"evaluators": summary, "clamp": clamp}


def _finite_differences(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    inputs: Mapping[str, Any],
    phase: torch.nn.Parameter,
    baseline_phase: torch.Tensor,
    candidate: torch.Tensor | None,
    ce_gradients: Mapping[str, Mapping[int, torch.Tensor]],
    margin_bundle: Mapping[str, Any],
    *,
    epsilons: Sequence[float] = (1.0 / 8.0, 1.0 / 16.0),
) -> dict[str, Any]:
    if candidate is None:
        return {"passed": True, "not_applicable": True, "reason": margin_bundle.get("unavailable_reason", "projection has numerical zero norm"), "checks": {}}
    fit_images = inputs["images_600"][:FIT_ROWS]
    fit_labels = inputs["labels_600"][:FIT_ROWS]
    checks: dict[str, Any] = {}
    diagnostic_snapshots: dict[str, Any] = {"baseline": {}, "interventions": {}, "check_vectors": {}}
    competitors = margin_bundle["fit_row_map"]
    original_phase = phase.detach().cpu().clone()
    _require(torch.equal(original_phase, baseline_phase.detach().cpu()), "finite-difference phase does not start at the declared baseline")
    fixed_maps = {
        class_id: {
            name: torch.tensor(competitors[name][str(class_id)]["max_wrong_classes_in_fit_order"], dtype=torch.long)
            for name in GUARD_EVALUATORS
        }
        for class_id in (1, 7, 8, 9)
    }
    try:
        for class_id in (1, 7, 8, 9):
            positions = torch.nonzero(fit_labels == class_id, as_tuple=False).reshape(-1)
            measured = _objective_batch(
                consumer,
                evaluators,
                fit_images.index_select(0, positions),
                fit_labels.index_select(0, positions),
                fixed_maps[class_id],
                capture_diagnostics=True,
            )
            diagnostic_snapshots["baseline"][str(class_id)] = {
                "fit_positions": positions.clone(),
                "source_row_indices": inputs["source_indices_600"][positions].clone(),
                "phase": baseline_phase.detach().cpu().clone(),
                "phase_displacement": torch.zeros_like(baseline_phase.detach().cpu()),
                "fixed_competitors": fixed_maps[class_id],
                **measured["diagnostics"],
            }
        for epsilon in epsilons:
            sign_data: dict[float, dict[int, dict[str, Any]]] = {1.0: {}, -1.0: {}}
            sign_phases: dict[float, torch.Tensor] = {}
            for sign in (1.0, -1.0):
                state_key = f"epsilon_{epsilon:g}_{'plus' if sign > 0 else 'minus'}"
                state_values: dict[int, dict[str, Any]] = {}
                try:
                    shared._phase_set(phase, baseline_phase, candidate * (epsilon * sign))
                    sign_phases[sign] = phase.detach().cpu().to(torch.float64).clone()
                    for class_id in (1, 7, 8, 9):
                        positions = torch.nonzero(fit_labels == class_id, as_tuple=False).reshape(-1)
                        measured = _objective_batch(
                            consumer,
                            evaluators,
                            fit_images.index_select(0, positions),
                            fit_labels.index_select(0, positions),
                            fixed_maps[class_id],
                            capture_diagnostics=True,
                        )
                        state_values[class_id] = measured
                        diagnostic_snapshots["interventions"].setdefault(state_key, {})[str(class_id)] = {
                            "fit_positions": positions.clone(),
                            "source_row_indices": inputs["source_indices_600"][positions].clone(),
                            "phase": sign_phases[sign].clone(),
                            "phase_displacement": sign_phases[sign] - baseline_phase.detach().cpu().to(torch.float64),
                            "fixed_competitors": fixed_maps[class_id],
                            **measured["diagnostics"],
                        }
                finally:
                    finite._restore_phase(phase, baseline_phase)
                sign_data[sign] = state_values
            _require(1.0 in sign_phases and -1.0 in sign_phases, "finite-difference intervention phases were not captured")
            phase_rate = (sign_phases[1.0] - sign_phases[-1.0]).reshape(-1) / (2.0 * epsilon)
            for class_id in (1, 7, 8, 9):
                plus_data = sign_data[1.0][class_id]
                minus_data = sign_data[-1.0][class_id]
                base_data = diagnostic_snapshots["baseline"][str(class_id)]
                branch_plus = _branch_change_summary(base_data, plus_data["diagnostics"])
                branch_minus = _branch_change_summary(base_data, minus_data["diagnostics"])
                for evaluator in GUARD_EVALUATORS:
                    switches_plus = int((plus_data[evaluator]["actual_competitor"] != fixed_maps[class_id][evaluator]).sum())
                    switches_minus = int((minus_data[evaluator]["actual_competitor"] != fixed_maps[class_id][evaluator]).sum())
                    for kind, gradient in (("CE", ce_gradients[evaluator][class_id]), ("margin", margin_bundle["gradients"][evaluator][class_id])):
                        score_key = "ce" if kind == "CE" else "fixed_margin"
                        plus_rows = plus_data[evaluator][score_key].to(torch.float64)
                        minus_rows = minus_data[evaluator][score_key].to(torch.float64)
                        observed, rowwise_slopes = _paired_finite_difference(plus_rows, minus_rows, epsilon)
                        logits_plus = plus_data["diagnostics"]["evaluators"][evaluator]["logits"]
                        logits_minus = minus_data["diagnostics"]["evaluators"][evaluator]["logits"]
                        if kind == "CE":
                            legacy_plus = F.cross_entropy(logits_plus, fit_labels.index_select(0, torch.nonzero(fit_labels == class_id, as_tuple=False).reshape(-1)), reduction="none").to(torch.float64)
                            legacy_minus = F.cross_entropy(logits_minus, fit_labels.index_select(0, torch.nonzero(fit_labels == class_id, as_tuple=False).reshape(-1)), reduction="none").to(torch.float64)
                        else:
                            local_labels = fit_labels.index_select(0, torch.nonzero(fit_labels == class_id, as_tuple=False).reshape(-1))
                            legacy_plus = _batch_margin(logits_plus, local_labels)[0].to(torch.float64)
                            legacy_minus = _batch_margin(logits_minus, local_labels)[0].to(torch.float64)
                        legacy_observed = float(((legacy_plus - legacy_minus) / (2.0 * epsilon)).mean())
                        predicted = float(torch.dot(gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1), phase_rate))
                        tolerance = 0.10 * abs(predicted) + 1e-4
                        key = f"{evaluator}:class{class_id}:{kind}:epsilon{epsilon:g}"
                        checks[key] = {
                            "evaluator": evaluator,
                            "class_id": class_id,
                            "quantity": kind,
                            "epsilon": float(epsilon),
                            "observed_slope": observed,
                            "legacy_float32_reduce_then_subtract_slope": legacy_observed,
                            "score_reduction_delta": observed - legacy_observed,
                            "predicted_slope_from_actual_phase_difference": predicted,
                            "absolute_error": abs(observed - predicted),
                            "tolerance": tolerance,
                            "max_wrong_branch_switches_plus": switches_plus,
                            "max_wrong_branch_switches_minus": switches_minus,
                            "nonlinear_branch_changes_plus": branch_plus,
                            "nonlinear_branch_changes_minus": branch_minus,
                            "passed": abs(observed - predicted) <= tolerance,
                        }
                        diagnostic_snapshots["check_vectors"][key] = {
                            "phase_minus": sign_phases[-1.0].clone(),
                            "phase_plus": sign_phases[1.0].clone(),
                            "actual_phase_rate": phase_rate.clone(),
                            "rowwise_observed_slopes": rowwise_slopes.clone(),
                        }
    finally:
        finite._restore_phase(phase, original_phase)
    return {
        "method": "float64 logits scoring; rowwise plus-minus pairing then float64 mean",
        "numeric_settings": _numeric_settings(),
        "passed": len(checks) == len(GUARD_EVALUATORS) * 4 * 2 * len(epsilons) and all(item["passed"] for item in checks.values()),
        "not_applicable": False,
        "executed_check_count": len(checks),
        "checks": checks,
        "diagnostic_snapshots": diagnostic_snapshots,
    }


def _layerwise_failure_diagnostics(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    inputs: Mapping[str, Any],
    phase: torch.nn.Parameter,
    ce_gradients: Mapping[str, Mapping[int, torch.Tensor]],
    margin_bundle: Mapping[str, Any],
    finite_difference: Mapping[str, Any],
) -> dict[str, Any]:
    failed = [item for item in finite_difference["checks"].values() if not item["passed"]]
    if not failed:
        return {"applicable": False, "reason": "all declared finite-difference checks passed"}
    images = inputs["images_600"][:FIT_ROWS]
    labels = inputs["labels_600"][:FIT_ROWS]
    failed_pairs = {(item["evaluator"], int(item["class_id"])) for item in failed}
    output_differences: dict[str, Any] = {}
    path_gradients: dict[tuple[str, int, str], dict[str, Any]] = {}
    for evaluator_name, class_id in sorted(failed_pairs):
        positions = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)
        competitors = torch.tensor(
            margin_bundle["fit_row_map"][evaluator_name][str(class_id)]["max_wrong_classes_in_fit_order"],
            dtype=torch.long,
        )
        baseline = finite_difference["diagnostic_snapshots"]["baseline"][str(class_id)]
        gradient_parts: dict[str, dict[str, list[torch.Tensor]]] = {
            kind: {name: [] for name in ("phase", "raw", "view")} for kind in ("CE", "margin")
        }
        maximum_differences = {"raw": 0.0, "view": 0.0, "logits": 0.0}
        for batch_index, start in enumerate((0, BATCH_SIZE)):
            batch_positions = positions[start:start + BATCH_SIZE]
            selected_images = images.index_select(0, batch_positions).to(consumer.device)
            selected_labels = labels.index_select(0, batch_positions).to(consumer.device)
            fixed = competitors[start:start + BATCH_SIZE].to(consumer.device)
            with torch.inference_mode(False):
                breakdown = consumer.forward_loss(selected_images, data_role=direction.bogt.TRAIN_ROLE)
                raw = breakdown.result.raw_intensity
                view = canonical.cmos_view(raw)
                logits, _ = _forward_evaluator(evaluators[evaluator_name], view, capture_branches=True)
                baseline_rows = slice(start, start + BATCH_SIZE)
                for name, value in (
                    ("raw", raw),
                    ("view", view),
                    ("logits", logits),
                ):
                    reference = baseline[name] if name != "logits" else baseline["evaluators"][evaluator_name]["logits"]
                    difference = float((value.detach().cpu().to(torch.float64) - reference[baseline_rows].to(torch.float64)).abs().max())
                    maximum_differences[name] = max(maximum_differences[name], difference)
                scores = _score_logits(logits, selected_labels, fixed)
                objectives = {"CE": scores["ce"].mean(), "margin": scores["fixed_margin"].mean()}
                for objective_index, (kind, scalar) in enumerate(objectives.items()):
                    grads = torch.autograd.grad(
                        scalar,
                        (phase, raw, view),
                        retain_graph=objective_index < len(objectives) - 1,
                        create_graph=False,
                        allow_unused=False,
                    )
                    for name, gradient in zip(("phase", "raw", "view"), grads, strict=True):
                        gradient_parts[kind][name].append(gradient.detach().to(device="cpu", dtype=torch.float64))
            del selected_images, selected_labels, breakdown, raw, view, logits, scores, objectives
        output_differences[f"{evaluator_name}:class{class_id}"] = {
            "max_abs_no_grad_to_autograd_forward_difference": maximum_differences,
            "phase_path_gradient_reference_errors": {},
        }
        for kind in ("CE", "margin"):
            grads = {name: torch.stack(parts).mean(0) if name == "phase" else torch.cat(parts) / 2.0 for name, parts in gradient_parts[kind].items()}
            stored = ce_gradients[evaluator_name][class_id] if kind == "CE" else margin_bundle["gradients"][evaluator_name][class_id]
            phase_gradient = grads["phase"].reshape(-1)
            stored64 = stored.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            output_differences[f"{evaluator_name}:class{class_id}"]["phase_path_gradient_reference_errors"][kind] = {
                "l2_error": float((phase_gradient - stored64).norm()),
                "reference_l2_norm": float(stored64.norm()),
            }
            path_gradients[(evaluator_name, class_id, kind)] = grads

    response_checks: dict[str, Any] = {}
    for item in failed:
        evaluator_name = item["evaluator"]
        class_id = int(item["class_id"])
        kind = item["quantity"]
        epsilon = float(item["epsilon"])
        plus_key = f"epsilon_{epsilon:g}_plus"
        minus_key = f"epsilon_{epsilon:g}_minus"
        plus = finite_difference["diagnostic_snapshots"]["interventions"][plus_key][str(class_id)]
        minus = finite_difference["diagnostic_snapshots"]["interventions"][minus_key][str(class_id)]
        gradients = path_gradients[(evaluator_name, class_id, kind)]
        denominator = 2.0 * epsilon
        phase_vectors = finite_difference["diagnostic_snapshots"]["check_vectors"][
            f"{evaluator_name}:class{class_id}:{kind}:epsilon{epsilon:g}"
        ]
        phase_rate = (phase_vectors["phase_plus"] - phase_vectors["phase_minus"]).reshape(-1).to(torch.float64) / denominator
        raw_rate = (plus["raw"].to(torch.float64) - minus["raw"].to(torch.float64)) / denominator
        view_rate = (plus["view"].to(torch.float64) - minus["view"].to(torch.float64)) / denominator
        response_checks[f"{evaluator_name}:class{class_id}:{kind}:epsilon{epsilon:g}"] = {
            "observed_score_slope": item["observed_slope"],
            "phase_gradient_prediction": float(torch.dot(gradients["phase"].reshape(-1), phase_rate)),
            "raw_cmos_gradient_prediction": float((gradients["raw"] * raw_rate).sum()),
            "evaluator_view_gradient_prediction": float((gradients["view"] * view_rate).sum()),
            "raw_cmos_delta_max_abs": float((plus["raw"] - minus["raw"]).abs().max()),
            "evaluator_view_delta_max_abs": float((plus["view"] - minus["view"]).abs().max()),
        }
    return {
        "applicable": True,
        "baseline_forward_comparison": output_differences,
        "failed_response_path_decomposition": response_checks,
        "nonlinear_branch_changes": {key: {
            "plus": item["nonlinear_branch_changes_plus"],
            "minus": item["nonlinear_branch_changes_minus"],
        } for key, item in finite_difference["checks"].items() if not item["passed"]},
    }


def _cpu_lifecycle_check(parent: Path) -> dict[str, bool]:
    with tempfile.TemporaryDirectory(prefix="lightgen_ce_margin_lifecycle_", dir=parent) as scratch_name:
        scratch = Path(scratch_name)
        tensor_path = scratch / "probe.pt"
        finite._write_torch_new(tensor_path, {"task_id": TASK_ID, "value": torch.tensor([1.25])}, "preflight lifecycle probe")
        loaded = torch.load(tensor_path, map_location="cpu", weights_only=True)
        _require(torch.equal(loaded["value"], torch.tensor([1.25])), "lifecycle write/read probe changed its tensor")
        manifest_path = scratch / "resume_manifest.json"
        manifest = {"task_id": TASK_ID, "status": "INTERRUPTED", "completed_chunks": ["one.pt"]}
        finite._write_json_new(manifest_path, manifest, "preflight resume probe")
        resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(resumed["status"] == "INTERRUPTED" and resumed["completed_chunks"] == ["one.pt"], "interrupted manifest cursor changed")
        resumed["status"] = "COMPLETED"
        finite._write_json_replace(manifest_path, resumed)
        terminal = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(terminal == resumed and tensor_path.is_file(), "terminal manifest self-test added or lost work")
    return {"immutable_write_read_passed": True, "interrupted_manifest_recovery_passed": True, "completed_resume_without_forward_passed": True}


def _preflight(inputs: Mapping[str, Any], output: Path, identity: ResearchRunIdentity, started: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for CE-margin frozen preflight")
    runtime_dll = direction._runtime_dll_probe()
    _require(runtime_dll.get("status") in {"already_present", "prepended", "not_applicable"}, "CUDA runtime DLL readiness failed")
    cpu_references = _cpu_references()
    output_parent = output
    while not output_parent.exists():
        output_parent = output_parent.parent
    disk = shutil.disk_usage(output_parent)
    disk_observation = {
        "checked_path": output_parent.as_posix(),
        "free_bytes_observed": int(disk.free),
        "estimated_output_bytes_observation": ESTIMATED_OUTPUT_BYTES,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_bytes": ESTIMATED_OUTPUT_BYTES + STORAGE_RESERVE_BYTES,
        "passed": disk.free >= ESTIMATED_OUTPUT_BYTES + STORAGE_RESERVE_BYTES,
    }
    _require(disk_observation["passed"], "output volume lacks estimated artifacts plus the 1.5 GiB reserve")
    lifecycle = _cpu_lifecycle_check(output_parent)

    consumer, parameters = direction._new_consumer(inputs, "B_class_ce", device)
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    evaluators = base.load_evaluators(inputs["paths"], device)
    for model in evaluators.values():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    runtime_before = finite._snapshot(consumer.runtime)
    evaluator_before = {name: finite._snapshot(model) for name, model in evaluators.items()}
    frozen_inputs_before = direction._consumer_frozen_inputs_snapshot(consumer)
    optimizer_before = direction._cpu_tree(inputs["optimizers"]["B_class_ce"])
    baseline_phase = parameters["generator.phase"].detach().cpu().clone()
    target_controls = _target_controls(evaluators, inputs, device)
    target_qualified = finite._qualify_target_controls(target_controls)
    _require(target_qualified, "one or more evaluator target-image positive controls failed")
    path_checks = _fixed_path_checks(consumer, inputs)

    phase = parameters["generator.phase"]
    settings_before = _numeric_settings()
    fd_diagnostic_snapshots: dict[str, Any] = {}
    localization: dict[str, Any] = {"applicable": False}
    extra_scale_comparison: dict[str, Any] = {"executed": False}
    strict_precision_comparison: dict[str, Any] = {"executed": False}
    strict_mode_adoption: dict[str, Any] = {"executed": False, "passed": False, "reason": "current-mode checks did not require a numerical-mode change"}
    selected_forward_mode = "current_default"
    fd_exception: str | None = None
    full_finite_difference: dict[str, Any] | None = None
    phase.requires_grad_(True)
    try:
        ce_saved = _saved_ce_gradients(inputs)
        margin_gradients = _mean_margin_gradients(
            consumer,
            evaluators,
            phase,
            inputs["images_600"][:FIT_ROWS],
            inputs["labels_600"][:FIT_ROWS],
            inputs["source_indices_600"][:FIT_ROWS],
        )
        _validate_margin_population(margin_gradients, inputs["labels_600"][:FIT_ROWS], inputs["source_indices_600"][:FIT_ROWS])
        ties = margin_gradients["candidate_unavailable_for_ties"]
        if ties:
            projection = _project(
                inputs["directions"]["equal_rms_directions"]["S_GD"],
                ce_saved["bundle"]["gradients"],
                margin_gradients["gradients"],
                float(inputs["directions"]["reference_rms"]),
            )
            projection["exact_ties"] = margin_gradients["exact_ties"]
            fd_checks = {"method": "not applicable: exact fit tie", "numeric_settings": settings_before, "passed": True, "not_applicable": True, "reason": "fit margin has an exact maximizing-class tie; no substitute derivative was selected", "executed_check_count": 0, "checks": {}}
        else:
            component_checks = margin_gradients["unique_max_wrong_component_checks"]
            _require(all(item["passed"] for evaluator in component_checks.values() for item in evaluator.values()), "one or more per-evaluator margin gradients differ from true-logit minus baseline-rival gradients")
            projection = _project(
                inputs["directions"]["equal_rms_directions"]["S_GD"],
                ce_saved["bundle"]["gradients"],
                margin_gradients["gradients"],
                float(inputs["directions"]["reference_rms"]),
            )
            try:
                full_finite_difference = _finite_differences(
                    consumer,
                    evaluators,
                    inputs,
                    phase,
                    baseline_phase,
                    projection["candidate_equal_rms_direction"],
                    ce_saved["bundle"]["gradients"],
                    margin_gradients,
                )
            except Exception as error:
                fd_exception = f"{type(error).__name__}: {error}"
                fd_checks = {
                    "method": "float64 logits scoring; rowwise plus-minus pairing then float64 mean",
                    "numeric_settings": settings_before,
                    "passed": False,
                    "not_applicable": False,
                    "executed_check_count": 0,
                    "checks": {},
                    "diagnostic_error": fd_exception,
                }
            else:
                fd_diagnostic_snapshots = full_finite_difference.pop("diagnostic_snapshots", {})
                fd_checks = full_finite_difference
                if not fd_checks["passed"]:
                    diagnostic_input = {**fd_checks, "diagnostic_snapshots": fd_diagnostic_snapshots}
                    try:
                        localization = _layerwise_failure_diagnostics(
                            consumer,
                            evaluators,
                            inputs,
                            phase,
                            ce_saved["bundle"]["gradients"],
                            margin_gradients,
                            diagnostic_input,
                        )
                    except Exception as error:
                        localization = {"applicable": True, "diagnostic_error": f"{type(error).__name__}: {error}"}
                    try:
                        extra = _finite_differences(
                            consumer,
                            evaluators,
                            inputs,
                            phase,
                            baseline_phase,
                            projection["candidate_equal_rms_direction"],
                            ce_saved["bundle"]["gradients"],
                            margin_gradients,
                            epsilons=(1.0 / 4.0, 1.0 / 32.0),
                        )
                        extra.pop("diagnostic_snapshots", None)
                        extra_scale_comparison = {
                            "executed": True,
                            "method": extra["method"],
                            "passed_count": sum(bool(item["passed"]) for item in extra["checks"].values()),
                            "executed_check_count": extra["executed_check_count"],
                            "checks": extra["checks"],
                        }
                    except Exception as error:
                        extra_scale_comparison = {"executed": False, "diagnostic_error": f"{type(error).__name__}: {error}"}
                    try:
                        with _strict_float32_settings() as original_settings:
                            strict = _finite_differences(
                                consumer,
                                evaluators,
                                inputs,
                                phase,
                                baseline_phase,
                                projection["candidate_equal_rms_direction"],
                                ce_saved["bundle"]["gradients"],
                                margin_gradients,
                            )
                        strict.pop("diagnostic_snapshots", None)
                        strict_precision_comparison = {
                            "executed": True,
                            "passed": strict.get("passed") is True and (strict.get("not_applicable") is True or strict.get("executed_check_count") == 48),
                            "original_settings": original_settings,
                            "strict_settings_after_context": _numeric_settings(),
                            "passed_count": sum(bool(item["passed"]) for item in strict["checks"].values()),
                            "executed_check_count": strict["executed_check_count"],
                            "checks": strict["checks"],
                            "strict_mode_is_diagnostic_only": True,
                        }
                    except Exception as error:
                        strict_precision_comparison = {"executed": False, "diagnostic_error": f"{type(error).__name__}: {error}", "strict_mode_is_diagnostic_only": True}
                    fd_checks["localization"] = localization
                    fd_checks["extra_scale_comparison"] = extra_scale_comparison
                    fd_checks["strict_precision_comparison"] = strict_precision_comparison
                    if strict_precision_comparison.get("passed") is True:
                        strict_mode_adoption = _strict_mode_compatibility(
                            consumer,
                            evaluators,
                            inputs,
                            phase,
                            baseline_phase,
                            ce_saved["bundle"],
                        )
                        adoption_public = {
                            key: value
                            for key, value in strict_mode_adoption.items()
                            if key not in {"strict_margin_gradients", "strict_projection_payload", "_diagnostic_snapshots"}
                        }
                        if strict_mode_adoption.get("passed") is True:
                            default_diagnostic = fd_checks
                            fd_checks = dict(strict_mode_adoption["strict_selected_mode_finite_difference_checks"])
                            fd_checks["mode_diagnostics"] = {
                                "default_mode_original_check_set": default_diagnostic,
                                "strict_mode_diagnostic_only": strict_precision_comparison,
                                "strict_mode_adoption_check": adoption_public,
                                "strict_diagnostic_alone_did_not_authorize_ready": True,
                            }
                            selected_forward_mode = "strict_fp32_no_tf32"
                            margin_gradients = strict_mode_adoption["strict_margin_gradients"]
                            projection = strict_mode_adoption["strict_projection_payload"]
                            path_checks = strict_mode_adoption["real_encoder_vs_fixed_ols"]["checks"]
                            target_controls = strict_mode_adoption["target_positive_controls"]
                            target_qualified = strict_mode_adoption["target_positive_controls_qualified"]
                            if strict_mode_adoption.get("_diagnostic_snapshots"):
                                fd_diagnostic_snapshots = {"strict_selected_mode": strict_mode_adoption["_diagnostic_snapshots"]}
                        else:
                            fd_checks["strict_mode_adoption_check"] = adoption_public
                            if strict_mode_adoption.get("_diagnostic_snapshots"):
                                fd_diagnostic_snapshots["strict_selected_mode"] = strict_mode_adoption["_diagnostic_snapshots"]

        actual_displacements: dict[str, torch.Tensor] = {}
        if projection["candidate_available"]:
            for name, planned in (("ce_margin_plus1", projection["candidate_equal_rms_direction"]), ("ce_margin_minus1", -projection["candidate_equal_rms_direction"])):
                actual_displacements[name] = shared._phase_set(phase, baseline_phase, planned)
                finite._restore_phase(phase, baseline_phase)
        projection = {**projection, "actual_float32_displacements": actual_displacements}
    finally:
        finite._restore_phase(phase, baseline_phase)
        phase.grad = None
        phase.requires_grad_(False)

    _require(phase.grad is None, "read-only gradients populated Generator .grad")
    for name, model in evaluators.items():
        _require(finite._tree_equal(evaluator_before[name], finite._snapshot(model)), f"preflight changed evaluator {name}")
    frozen = finite._tree_equal(runtime_before, finite._snapshot(consumer.runtime))
    frozen_inputs = finite._tree_equal(frozen_inputs_before, direction._consumer_frozen_inputs_snapshot(consumer))
    adam_unchanged = finite._tree_equal(optimizer_before, direction._cpu_tree(inputs["optimizers"]["B_class_ce"]))
    evaluator_frozen = all(parameter.grad is None and not parameter.requires_grad for model in evaluators.values() for parameter in model.parameters())
    parameters_empty = all(parameter.grad is None and not parameter.requires_grad for parameter in parameters.values())
    _require(frozen and frozen_inputs and adam_unchanged and evaluator_frozen and parameters_empty, "preflight changed a frozen state, source Adam, or gradient buffer")
    cuda_after = _resource_observation(device)
    fd_count_ok = fd_checks.get("not_applicable") is True or (
        fd_checks.get("executed_check_count") == 48
        and len(fd_checks.get("checks", {})) == 48
        and all(item.get("passed") is not None for item in fd_checks.get("checks", {}).values())
    )
    result = {
        "schema": "m0_generator_ce_margin_guarded_preflight.v2",
        "task_id": TASK_ID,
        "status": "READY" if fd_checks.get("passed") is True and fd_count_ok else "INVALID_HARNESS",
        "cpu_independent_references": cpu_references,
        "target_positive_controls": target_controls,
        "target_positive_controls_qualified": target_qualified,
        "real_encoder_vs_fixed_ols": path_checks,
        "saved_ce_gradient_reference": {"passed": True, "mixed_batch_reference_agreement": ce_saved["mixed_batch_reference_agreement"]},
        "margin_gradient_production": {
            "passed": all(item["passed"] for evaluator in margin_gradients["unique_max_wrong_component_checks"].values() for item in evaluator.values()),
            "differentiable_all_constraints": not ties,
            "all_fit_rows_accounted_for": all(len(margin_gradients["fit_row_map"][evaluator][str(class_id)]["fit_positions"]) == 20 for evaluator in GUARD_EVALUATORS for class_id in range(10)),
            "definition": margin_gradients["gradient_definition"],
            "unique_max_wrong_component_checks": margin_gradients["unique_max_wrong_component_checks"],
            "exact_ties": margin_gradients["exact_ties"],
            "candidate_unavailable_for_ties": ties,
        },
        "projection": {
            "passed": projection.get("exact_tie_exit") is True or projection["kkt_residuals"]["maximum"] <= KKT_TOLERANCE,
            "candidate_available": projection["candidate_available"],
            "kkt_residuals": projection.get("kkt_residuals"),
            "zero_gradient_constraints": projection.get("zero_gradient_constraints", []),
            "projected_to_source_l2_ratio": projection.get("projected_to_source_l2_ratio"),
            "reference_rms": projection["reference_rms"],
            "candidate_rms": projection.get("candidate_rms"),
        },
        "finite_difference_checks": fd_checks,
        "numerical_scoring": {
            "production_forward_dtype_unchanged": "float32",
            "score_dtype": "float64",
            "finite_difference_reduction": "rowwise paired plus-minus differences followed by float64 mean",
            "settings_before": settings_before,
            "settings_after": _numeric_settings(),
            "required_original_check_count": 48,
            "executed_original_check_count": fd_checks.get("executed_check_count", 0),
            "required_original_check_set_satisfied": fd_count_ok,
            "selected_forward_mode": selected_forward_mode,
            "strict_mode_diagnostic_only": True,
            "strict_mode_adoption_check": {key: value for key, value in strict_mode_adoption.items() if key not in {"strict_margin_gradients", "strict_projection_payload", "_diagnostic_snapshots"}},
            "fd_exception": fd_exception,
        },
        "storage_lifecycle": lifecycle,
        "disk_observation": disk_observation,
        "runtime_dll": runtime_dll,
        "cuda_before": _resource_observation(device),
        "cuda_after": cuda_after,
        "model_and_scientific_buffers_unchanged": frozen,
        "frozen_inputs_unchanged": frozen_inputs,
        "source_adam_unchanged": adam_unchanged,
        "evaluator_state_unchanged": evaluator_frozen,
        "parameter_grad_buffers_empty": parameters_empty,
        "optimizer_updates": 0,
        "elapsed_seconds_observation_only": time.perf_counter() - started,
    }
    if result["status"] != "READY":
        result["_diagnostic_snapshots"] = fd_diagnostic_snapshots
    return result, {"ce": ce_saved["bundle"], "margin": margin_gradients}, projection


def _ready_receipt(control: ResearchRunControl, preflight: Mapping[str, Any]) -> PreflightReceipt:
    finite_checks = preflight.get("finite_difference_checks", {})
    numeric = preflight.get("numerical_scoring", {})
    finite_check_set_valid = finite_checks.get("not_applicable") is True or (
        finite_checks.get("method") == "float64 logits scoring; rowwise plus-minus pairing then float64 mean"
        and finite_checks.get("executed_check_count") == 48
        and len(finite_checks.get("checks", {})) == 48
        and all(item.get("passed") is not None for item in finite_checks.get("checks", {}).values())
    )
    selected_mode = numeric.get("selected_forward_mode")
    selected_mode_valid = selected_mode == "current_default" or (
        selected_mode == "strict_fp32_no_tf32"
        and numeric.get("strict_mode_adoption_check", {}).get("passed") is True
        and numeric.get("strict_mode_adoption_check", {}).get("settings_restored") is True
    )
    correctness = (
        preflight["status"] == "READY"
        and preflight["cpu_independent_references"].get("passed") is True
        and preflight["target_positive_controls_qualified"] is True
        and set(preflight["real_encoder_vs_fixed_ols"]) == {"fit", "confirmation"}
        and preflight["saved_ce_gradient_reference"].get("passed") is True
        and preflight["margin_gradient_production"].get("passed") is True
        and preflight["margin_gradient_production"].get("all_fit_rows_accounted_for") is True
        and preflight["projection"].get("passed") is True
        and finite_checks.get("passed") is True
        and finite_check_set_valid
        and numeric.get("production_forward_dtype_unchanged") == "float32"
        and numeric.get("score_dtype") == "float64"
        and numeric.get("finite_difference_reduction") == "rowwise paired plus-minus differences followed by float64 mean"
        and numeric.get("required_original_check_set_satisfied") is True
        and selected_mode_valid
        and preflight["model_and_scientific_buffers_unchanged"] is True
        and preflight["source_adam_unchanged"] is True
        and preflight["evaluator_state_unchanged"] is True
        and preflight["parameter_grad_buffers_empty"] is True
        and preflight["optimizer_updates"] == 0
        and all(preflight["storage_lifecycle"].get(key) is True for key in ("immutable_write_read_passed", "interrupted_manifest_recovery_passed", "completed_resume_without_forward_passed"))
    )
    cuda = [preflight.get("cuda_before", {}), preflight.get("cuda_after", {})]
    resources = all(item.get("total_bytes", 0) > 0 and 0 <= item.get("free_bytes", -1) <= item.get("total_bytes", 0) for item in cuda)
    resources = resources and preflight["disk_observation"].get("passed") is True and preflight["runtime_dll"].get("status") in {"already_present", "prepended", "not_applicable"}
    implementation = correctness and preflight["optimizer_updates"] == 0
    return control.make_ready_receipt(
        correctness={"passed": correctness, "status": "PASS" if correctness else "BLOCKED", "independent_reference": preflight["cpu_independent_references"].get("passed") is True, "per_evaluator_gradients": preflight["margin_gradient_production"].get("passed") is True and preflight["margin_gradient_production"].get("all_fit_rows_accounted_for") is True, "projection_and_finite_difference": preflight["projection"].get("passed") is True and finite_checks.get("passed") is True and finite_check_set_valid, "numerical_scoring_and_full_check_set": numeric.get("required_original_check_set_satisfied") is True and selected_mode_valid, "frozen_state": preflight["model_and_scientific_buffers_unchanged"] and preflight["source_adam_unchanged"] and preflight["evaluator_state_unchanged"]},
        resource={"passed": resources, "status": "PASS" if resources else "BLOCKED", "device_telemetry_valid": resources, "whole_device_memory_observation_only": True, "storage": preflight["disk_observation"], "runtime_dll": preflight["runtime_dll"]},
        implementation={"passed": implementation, "status": "PASS" if implementation else "BLOCKED", "selected_engine": "cuda_2048_frozen_ce_margin_guarded_displacement", "optimizer_updates_zero": True, "constraint_count": 60},
        selected_engine="cuda_2048_frozen_ce_margin_guarded_displacement",
        resource_envelope={"cuda_before": preflight.get("cuda_before"), "cuda_after": preflight.get("cuda_after"), "storage": preflight.get("disk_observation")},
    )


def _manifest_seed(identity: ResearchRunIdentity, population: Mapping[str, Any], receipt: PreflightReceipt, preflight: Mapping[str, Any], projection: Mapping[str, Any]) -> dict[str, Any]:
    conditions = ["baseline", "baseline_repeat"]
    if projection["candidate_available"]:
        conditions.extend(("ce_margin_plus1", "ce_margin_minus1"))
    return {
        "schema": MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "measurement_semantics": "v2 separate E_S/E_I/E_C class CE nonincrease and true-minus-unique-max-wrong margin nondecrease; float64 logits scoring and rowwise paired finite differences; one projected common phase displacement",
        "finite_difference_reduction": preflight["numerical_scoring"]["finite_difference_reduction"],
        "required_finite_difference_count": preflight["numerical_scoring"]["required_original_check_count"],
        "selected_forward_mode": preflight["numerical_scoring"]["selected_forward_mode"],
        "identity": identity.to_dict(),
        "population": dict(population),
        "receipt": receipt.to_dict(),
        "preflight": dict(preflight),
        "measured_conditions": conditions,
        "completed_chunks": [],
        "status": "READY",
    }


def _validate_manifest(path: Path, target: Path, identity: ResearchRunIdentity, population: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.resolve() == (target / "resume_manifest.json").resolve() and path.is_file(), "resume requires this output's explicit resume_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("task_id") == TASK_ID
        and manifest.get("measurement_semantics", "").startswith("v2 separate E_S/E_I/E_C class CE")
        and manifest.get("finite_difference_reduction") == "rowwise paired plus-minus differences followed by float64 mean"
        and manifest.get("required_finite_difference_count") == 48,
        "resume manifest has old or incomplete numerical measurement semantics",
    )
    _require(manifest.get("selected_forward_mode") in {"current_default", "strict_fp32_no_tf32"}, "resume manifest has no validated forward precision mode")
    _require(manifest.get("selected_forward_mode") == manifest.get("preflight", {}).get("numerical_scoring", {}).get("selected_forward_mode"), "resume manifest precision mode differs from its READY preflight")
    _require(manifest.get("population") == dict(population), "resume population, input paths, rows, labels, or roles changed")
    check_resume_compatibility(identity, manifest.get("identity", {}))
    _require(manifest.get("status") in {"READY", "RUNNING", "INTERRUPTED", "COMPLETED"}, "resume manifest is not resumable")
    return manifest


def _condition_name(condition: str, index: int) -> str:
    start = index * CHUNK_SIZE
    return f"readout_{condition}_rows{start:03d}_{start + CHUNK_SIZE:03d}.pt"


def _validate_readout(path: Path, condition: str, index: int, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(payload.get("schema") == BLOCK_SCHEMA and payload.get("task_id") == TASK_ID and payload.get("condition") == condition and payload.get("chunk_index") == index, "readout chunk identity or semantics mismatch")
    block = payload["block"]
    start, stop = index * CHUNK_SIZE, (index + 1) * CHUNK_SIZE
    _require(block.get("row_start") == start and block.get("row_stop") == stop and block.get("kind") == condition, "readout chunk row boundaries changed")
    _require(torch.equal(block["source_row_indices"], inputs["source_indices_600"][start:stop]), "readout chunk row indices changed")
    _require(torch.equal(block["labels"], inputs["labels_600"][start:stop]) and block["roles"] == list(inputs["roles_600"][start:stop]), "readout chunk labels or roles changed")
    _require(tuple(block["raw_cmos"].shape) == (CHUNK_SIZE, 1, 200, 200) and bool(torch.isfinite(block["raw_cmos"]).all()), "readout raw CMOS is malformed")
    for name in EVALUATORS:
        logits = block.get("evaluator_logits", {}).get(name)
        _require(torch.is_tensor(logits) and logits.shape == (CHUNK_SIZE, 10) and bool(torch.isfinite(logits).all()), f"{name} readout logits are missing or nonfinite")
    return block


def _save_readout(path: Path, condition: str, index: int, block: Mapping[str, Any]) -> None:
    finite._write_torch_new(path, {"schema": BLOCK_SCHEMA, "task_id": TASK_ID, "condition": condition, "chunk_index": index, "block": dict(block)}, "immutable CE-margin guarded readout chunk")


def _measure_chunk(consumer: Any, evaluators: Mapping[str, torch.nn.Module], inputs: Mapping[str, Any], index: int, condition: str, path: Path, baseline_block: Mapping[str, Any] | None = None, baseline: bool = False) -> dict[str, Any]:
    start = index * CHUNK_SIZE
    records = [shared._read_fixed_batch(consumer, evaluators, inputs, batch_start) for batch_start in range(start, start + CHUNK_SIZE, BATCH_SIZE)]
    block = shared._concat_batch_records(records, start, condition)
    if baseline:
        block["parent_readout_max_abs_errors"] = shared._verify_baseline_block(block, inputs)
    if baseline_block is not None:
        switched = {name: 0 for name in EVALUATORS}
        for evaluator in EVALUATORS:
            for row_index in range(CHUNK_SIZE):
                old = baseline_block["evaluator_logits"][evaluator][row_index].argmax().item()
                new = block["evaluator_logits"][evaluator][row_index].argmax().item()
                switched[evaluator] += int(old != new)
        block["prediction_switches_from_baseline"] = switched
    _save_readout(path, condition, index, block)
    return block


def _measure_condition(target: Path, manifest: dict[str, Any], condition: str, inputs: Mapping[str, Any], consumer: Any, evaluators: Mapping[str, torch.nn.Module], baseline: bool = False) -> None:
    for index in range(CHUNK_COUNT):
        name = _condition_name(condition, index)
        path = target / name
        if name in manifest["completed_chunks"]:
            _validate_readout(path, condition, index, inputs)
            continue
        if path.exists():
            _validate_readout(path, condition, index, inputs)
            manifest["completed_chunks"] = sorted(set(manifest["completed_chunks"]) | {name})
            finite._write_json_replace(target / "resume_manifest.json", manifest)
            continue
        baseline_block = None if condition in {"baseline", "baseline_repeat"} else _validate_readout(target / _condition_name("baseline", index), "baseline", index, inputs)
        _measure_chunk(consumer, evaluators, inputs, index, condition, path, baseline_block, baseline=baseline)
        manifest["completed_chunks"] = sorted(set(manifest["completed_chunks"]) | {name})
        finite._write_json_replace(target / "resume_manifest.json", manifest)
    expected = {_condition_name(condition, index) for index in range(CHUNK_COUNT)}
    _require(expected.issubset(set(manifest["completed_chunks"])), f"condition {condition} did not complete all six row chunks")


@contextlib.contextmanager
def _production_numeric_mode(preflight: Mapping[str, Any]):
    numerical = preflight.get("numerical_scoring", {})
    mode = numerical.get("selected_forward_mode")
    if mode == "strict_fp32_no_tf32":
        with _strict_float32_settings():
            settings = _numeric_settings()
            _require(not settings["cuda_matmul_allow_tf32"] and not settings["cudnn_allow_tf32"], "strict production mode did not disable TF32")
            _require(settings["float32_matmul_precision"] == "highest" and not settings["cuda_autocast_enabled"], "strict production mode did not use highest FP32 without autocast")
            yield
        return
    _require(mode == "current_default", "saved preflight has an unknown production numeric mode")
    expected = numerical.get("settings_before", {})
    actual = _numeric_settings()
    for key in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision", "cuda_autocast_enabled"):
        _require(actual.get(key) == expected.get(key), f"production numeric setting {key} changed after preflight")
    yield


def _measure_all_conditions(
    target: Path,
    manifest: dict[str, Any],
    conditions: Sequence[str],
    inputs: Mapping[str, Any],
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    parameters: Mapping[str, torch.nn.Parameter],
    baseline_phase: torch.Tensor,
    projection: Mapping[str, Any],
    runtime_before: Mapping[str, Any],
    evaluator_before: Mapping[str, Any],
    frozen_inputs_before: Mapping[str, Any],
    optimizer_before: Mapping[str, Any],
) -> None:
    phase = parameters["generator.phase"]
    _require(torch.equal(phase.detach().cpu(), baseline_phase), "loaded Generator phase differs from saved frozen B endpoint")
    for condition in conditions:
        if condition == "baseline":
            _measure_condition(target, manifest, condition, inputs, consumer, evaluators, baseline=True)
            finite._restore_phase(phase, baseline_phase)
        elif condition == "baseline_repeat":
            _measure_condition(target, manifest, condition, inputs, consumer, evaluators)
            finite._restore_phase(phase, baseline_phase)
        else:
            delta = projection["candidate_equal_rms_direction"]
            if condition == "ce_margin_minus1":
                delta = -delta
            applied = shared._phase_set(phase, baseline_phase, delta)
            saved = torch.load(target / "actual_displacements.pt", map_location="cpu", weights_only=True)
            _require(torch.equal(applied, saved["actual_float32"][condition]), f"actual float32 displacement changed for {condition}")
            _measure_condition(target, manifest, condition, inputs, consumer, evaluators)
            finite._restore_phase(phase, baseline_phase)
        checks = _snapshot_checks(consumer, evaluators, parameters, runtime_before, evaluator_before, frozen_inputs_before, optimizer_before, inputs)
        manifest["last_frozen_state_check"] = checks
        finite._write_json_replace(target / "resume_manifest.json", manifest)
    expected = {_condition_name(condition, index) for condition in conditions for index in range(CHUNK_COUNT)}
    _require(expected.issubset(set(manifest["completed_chunks"])), "one or more required readout chunks are missing")


def _collect_rows(target: Path, condition: str, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(CHUNK_COUNT):
        block = _validate_readout(target / _condition_name(condition, index), condition, index, inputs)
        for local in range(CHUNK_SIZE):
            ed_logits = block["evaluator_logits"]["E_D"][local]
            label_tensor = torch.tensor([int(block["labels"][local])], dtype=torch.long)
            ed_score = _score_logits(ed_logits.reshape(1, -1), label_tensor)
            margin = block["margins"][local]
            image = block["image_metrics"][local]
            row = {
                "position": index * CHUNK_SIZE + local,
                "source_row_index": int(block["source_row_indices"][local]),
                "label": int(block["labels"][local]),
                "role": block["roles"][local],
                "ed_margin": float(ed_score["margin"][0]),
                "ed_best_wrong_class": int(ed_score["actual_competitor"][0]),
                "ed_best_wrong_tied_classes": list(margin["best_wrong_tied_classes"]),
                "raw_mse": float(image["raw_mse"]),
                "raw_foreground_mse": float(image["raw_foreground_mse"]),
                "raw_background_mse": float(image["raw_background_mse"]),
                "view_mse": float(image["view_mse"]),
                "view_foreground_mse": float(image["view_foreground_mse"]),
                "view_background_mse": float(image["view_background_mse"]),
                "raw_fraction_above_one": float(image["raw_fraction_above_one"]),
                "raw_zero_image": bool(image["raw_zero_image"]),
                "raw_constant_image": bool(image["raw_constant_image"]),
                "condition": condition,
            }
            for evaluator in EVALUATORS:
                logits = block["evaluator_logits"][evaluator][local]
                scores = _score_logits(logits.reshape(1, -1), label_tensor)
                row[f"{evaluator}_prediction"] = int(logits.argmax())
                row[f"{evaluator}_correct"] = int(row[f"{evaluator}_prediction"] == row["label"])
                row[f"{evaluator}_ce"] = float(scores["ce"][0])
                row[f"{evaluator}_margin"] = float(scores["margin"][0])
                row[f"{evaluator}_max_wrong_class"] = int(scores["actual_competitor"][0])
            result.append(row)
    return result


def _metric_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    output: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for role in ("fit", "historic_holdout", "confirmation"):
        output[role] = {}
        selected = [row for row in rows if row["role"] == role]
        for evaluator in EVALUATORS:
            output[role][evaluator] = {}
            for class_id in range(10):
                entries = [row for row in selected if row["label"] == class_id]
                output[role][evaluator][str(class_id)] = {
                    "rows": float(len(entries)),
                    "mean_ce": sum(float(row[f"{evaluator}_ce"]) for row in entries) / len(entries) if entries else math.nan,
                    "mean_margin": sum(float(row[f"{evaluator}_margin"]) for row in entries) / len(entries) if entries else math.nan,
                    "correct": float(sum(int(row[f"{evaluator}_correct"]) for row in entries)),
                    "prediction_counts": [float(sum(int(row[f"{evaluator}_prediction"]) == value for row in entries)) for value in range(10)],
                }
    return output


def _delta_matrix(before: Mapping[str, Any], after: Mapping[str, Any], *, field: str) -> dict[str, dict[str, dict[str, float]]]:
    return {
        evaluator: {
            str(class_id): {
                role: float(after[role][evaluator][str(class_id)][field] - before[role][evaluator][str(class_id)][field])
                for role in ("fit", "historic_holdout", "confirmation")
            }
            for class_id in range(10)
        }
        for evaluator in EVALUATORS
    }


def _decision(baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], base_matrix: Mapping[str, Any], candidate_matrix: Mapping[str, Any], eta_margin: float, eta_ce: float, eta_train_margin: float, *, candidate_available: bool = True) -> dict[str, Any]:
    if not candidate_available:
        return {"classification": "MIXED_OR_INSUFFICIENT", "candidate_available": False, "candidate_unavailable_reason": "projection is tied or within the declared numerical zero range", "training_recommendation": "NONE", "next_action": "NONE", "claim_boundary": "fixed-fitness local diagnostic only; no claim about absent alternatives or architecture capacity"}
    base_by_source = {int(row["source_row_index"]): row for row in baseline}
    confirm_base = [row for row in baseline if row["role"] == "confirmation"]
    confirm_candidate = [row for row in candidate if row["role"] == "confirmation"]
    _require(len(confirm_base) == len(confirm_candidate) == CONFIRMATION_ROWS, "final decision requires all three hundred confirmation rows")
    margin_delta = {int(row["source_row_index"]): float(row["ed_margin"]) - float(base_by_source[int(row["source_row_index"])] ["ed_margin"]) for row in candidate}
    focus_response = {
        str(class_id): {
            "rows": sum(row["label"] == class_id for row in confirm_candidate),
            "beneficial_rows": sum(row["label"] == class_id and margin_delta[int(row["source_row_index"])] > eta_margin for row in confirm_candidate),
            "harmful_rows": sum(row["label"] == class_id and margin_delta[int(row["source_row_index"])] < -eta_margin for row in confirm_candidate),
            "mean_margin_delta": sum(margin_delta[int(row["source_row_index"])] for row in confirm_candidate if row["label"] == class_id) / 30.0,
        }
        for class_id in FOCUS_CLASSES
    }
    other_changes: dict[str, Any] = {}
    stable_harm: list[int] = []
    mean_failures: list[int] = []
    for class_id in range(10):
        entries = [row for row in confirm_candidate if row["label"] == class_id]
        changes = [margin_delta[int(row["source_row_index"])] for row in entries]
        harm = sum(value < -eta_margin for value in changes)
        mean_change = sum(changes) / len(changes)
        if class_id not in FOCUS_CLASSES:
            if harm >= STABLE_ROWS:
                stable_harm.append(class_id)
            if mean_change < -eta_margin:
                mean_failures.append(class_id)
        other_changes[str(class_id)] = {"rows": len(entries), "beneficial_rows": sum(value > eta_margin for value in changes), "harmful_rows": harm, "boundary_rows": sum(abs(value) <= eta_margin for value in changes), "mean_margin_delta": mean_change}
    ce_deltas = {f"{e}:{c}": float(candidate_matrix["fit"][e][str(c)]["mean_ce"] - base_matrix["fit"][e][str(c)]["mean_ce"]) for e, c in GUARD_PAIRS}
    train_margin_deltas = {f"{e}:{c}": float(candidate_matrix["fit"][e][str(c)]["mean_margin"] - base_matrix["fit"][e][str(c)]["mean_margin"]) for e, c in GUARD_PAIRS}
    ce_failures = {key: value for key, value in ce_deltas.items() if value > eta_ce}
    train_margin_failures = {key: value for key, value in train_margin_deltas.items() if value < -eta_train_margin}
    focus_ok = all(item["beneficial_rows"] >= STABLE_ROWS for item in focus_response.values())
    other_margin_ok = not stable_harm and not mean_failures
    base_accuracy = shared._class_metrics(confirm_base)
    cand_accuracy = shared._class_metrics(confirm_candidate)
    correct_declines = {str(c): base_accuracy["correct"][c] - cand_accuracy["correct"][c] for c in range(10)}
    accuracy_ok = all(value <= 1 for value in correct_declines.values()) and cand_accuracy["macro_recall"] >= base_accuracy["macro_recall"]
    base_raw_mse = sum(float(row["raw_mse"]) for row in confirm_base) / CONFIRMATION_ROWS
    cand_raw_mse = sum(float(row["raw_mse"]) for row in confirm_candidate) / CONFIRMATION_ROWS
    raw_clip = sum(float(row["raw_fraction_above_one"]) for row in confirm_candidate) / CONFIRMATION_ROWS
    class_clip = {str(c): sum(float(row["raw_fraction_above_one"]) for row in confirm_candidate if row["label"] == c) / 30.0 for c in range(10)}
    image_valid = all(math.isfinite(float(row["raw_mse"])) and not row["raw_zero_image"] and not row["raw_constant_image"] for row in confirm_candidate)
    guards_passed = not ce_failures and not train_margin_failures and focus_ok and other_margin_ok and accuracy_ok and cand_raw_mse <= base_raw_mse * 1.05 and raw_clip <= 0.01 and all(class_clip[str(c)] <= 0.01 for c in FOCUS_CLASSES) and image_valid
    classification = "CE_MARGIN_GUARDED_SHARED_LOCAL_RESPONSE_SUPPORTED" if guards_passed else "CE_MARGIN_GUARDED_RESPONSE_WITH_TRADEOFF" if focus_ok else "MIXED_OR_INSUFFICIENT"
    return {
        "classification": classification,
        "candidate_available": True,
        "confirmation_focus_response": focus_response,
        "confirmation_other_class_margin_changes": other_changes,
        "stable_harmful_other_classes": stable_harm,
        "other_class_mean_margin_failures": mean_failures,
        "fit_ce_guard_deltas": ce_deltas,
        "fit_ce_guard_failures": ce_failures,
        "fit_margin_guard_deltas": train_margin_deltas,
        "fit_margin_guard_failures": train_margin_failures,
        "accuracy_correct_declines_by_class": correct_declines,
        "accuracy_guard_passed": accuracy_ok,
        "baseline_macro_recall": base_accuracy["macro_recall"],
        "candidate_macro_recall": cand_accuracy["macro_recall"],
        "baseline_confirmation_raw_mse_mean": base_raw_mse,
        "candidate_confirmation_raw_mse_mean": cand_raw_mse,
        "raw_mse_guard_passed": cand_raw_mse <= base_raw_mse * 1.05,
        "candidate_confirmation_raw_fraction_above_one_mean": raw_clip,
        "candidate_confirmation_raw_fraction_above_one_by_class": class_clip,
        "clipping_guard_passed": raw_clip <= 0.01 and all(class_clip[str(c)] <= 0.01 for c in FOCUS_CLASSES),
        "no_zero_or_constant_confirmation_images": image_valid,
        "eta_e_d_margin": eta_margin,
        "eta_fit_ce": eta_ce,
        "eta_fit_train_margin": eta_train_margin,
        "minimum_focus_beneficial_rows": STABLE_ROWS,
        "fit_guards_passed": not ce_failures and not train_margin_failures,
        "class_margin_guard_passed": other_margin_ok,
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "claim_boundary": "one equal-RMS frozen common phase displacement; local response only, not accumulated training, human readability, generalization, or architecture-capacity evidence",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), "cannot write an empty row table")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
    writer.writeheader()
    writer.writerows(rows)
    finite._write_bytes_new(path, buffer.getvalue().encode("utf-8"), "CE-margin guarded row metrics")


def _write_review_pages(target: Path, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    shared_dir = inputs["shared_dir"]
    class_guard_dir = inputs["class_guard_dir"]
    pages: list[dict[str, Any]] = []
    panel_names = ("target", "baseline", "pure_ce_plus", "ce_only_guard_plus", "ce_margin_guard_plus")
    labels = {"target": "target", "baseline": "baseline", "pure_ce_plus": "pure CE", "ce_only_guard_plus": "CE guard", "ce_margin_guard_plus": "CE+margin"}
    size, row_h, gap, header = 90, 19, 10, 28
    panel_x = [190 + i * (size + gap) for i in range(len(panel_names))]
    for class_id in DISPLAY_CLASSES:
        positions = [i for i, label in enumerate(inputs["labels_600"].tolist()) if label == class_id]
        width = panel_x[-1] + size + 8
        image = Image.new("RGB", (width, header + len(positions) * row_h), "white")
        draw = ImageDraw.Draw(image)
        for index, name in enumerate(panel_names):
            draw.text((panel_x[index], 6), labels[name], fill=(0, 0, 0))
        for row_index, position in enumerate(positions):
            chunk_index, local = divmod(position, CHUNK_SIZE)
            baseline = _validate_readout(target / _condition_name("baseline", chunk_index), "baseline", chunk_index, inputs)
            candidate = _validate_readout(target / _condition_name("ce_margin_plus1", chunk_index), "ce_margin_plus1", chunk_index, inputs) if (target / _condition_name("ce_margin_plus1", chunk_index)).is_file() else None
            pure = shared._load_chunk(shared_dir / shared._chunk_name("S_GD_equal_rms_plus1", chunk_index), "S_GD_equal_rms_plus1", chunk_index, inputs)
            guarded = class_guard._load_chunk(class_guard_dir, "guarded_plus1", chunk_index, inputs)
            row_y = header + row_index * row_h
            ed_prediction = int(baseline["evaluator_logits"]["E_D"][local].argmax())
            role = inputs["roles_600"][position]
            draw.text((4, row_y + 5), f"{role} {int(inputs['source_indices_600'][position])} ED:{ed_prediction}", fill=(0, 0, 0))
            views = [canonical.cmos_view(inputs["images_600"][position:position + 1])[0, 0], canonical.cmos_view(baseline["raw_cmos"][local:local + 1])[0, 0], canonical.cmos_view(pure["raw_cmos"][local:local + 1])[0, 0], canonical.cmos_view(guarded["raw_cmos"][local:local + 1])[0, 0]]
            if candidate is not None:
                views.append(canonical.cmos_view(candidate["raw_cmos"][local:local + 1])[0, 0])
            else:
                views.append(torch.zeros_like(views[0]))
            for panel_index, view in enumerate(views):
                pixels = view.detach().cpu().clamp(0.0, 1.0).mul(255).round().to(torch.uint8).numpy()
                thumb = Image.fromarray(pixels, mode="L").resize((size, size), Image.Resampling.NEAREST).convert("RGB")
                image.paste(thumb, (panel_x[panel_index], row_y))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        filename = f"review_class_{class_id}.png"
        finite._write_bytes_new(target / filename, buffer.getvalue(), "fixed-scale class image review")
        pages.append({"class_id": class_id, "rows": len(positions), "panels": list(panel_names), "path": filename, "view": "28x28 clamped [0,1] adaptive average pooling; no per-image rescaling"})
    return pages


def _summary(target: Path, condition: str, inputs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _collect_rows(target, condition, inputs)
    base_summary = class_guard._role_summary(rows)
    matrix = _metric_matrix(rows)
    return rows, {"classification_metrics": base_summary, "ce_margin_means": matrix}


def _finalize(target: Path, inputs: Mapping[str, Any], identity: ResearchRunIdentity, preflight: Mapping[str, Any], gradients: Mapping[str, Any], projection: Mapping[str, Any], started: float) -> dict[str, Any]:
    conditions = ["baseline", "baseline_repeat"] + (["ce_margin_plus1", "ce_margin_minus1"] if projection["candidate_available"] else [])
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    matrices: dict[str, Any] = {}
    for condition in conditions:
        rows, summary = _summary(target, condition, inputs)
        rows_by_condition[condition] = rows
        summaries[condition] = summary["classification_metrics"]
        matrices[condition] = summary["ce_margin_means"]
    baseline, repeated = rows_by_condition["baseline"], rows_by_condition["baseline_repeat"]
    by_source = {int(row["source_row_index"]): row for row in repeated}
    eta_e_d = max(ETA_FLOOR, 5.0 * max(abs(float(row["ed_margin"]) - float(by_source[int(row["source_row_index"])]["ed_margin"])) for row in baseline))
    ce_repeat_error = max(abs(matrices["baseline"]["fit"][e][str(c)]["mean_ce"] - matrices["baseline_repeat"]["fit"][e][str(c)]["mean_ce"]) for e, c in GUARD_PAIRS)
    margin_repeat_error = max(abs(matrices["baseline"]["fit"][e][str(c)]["mean_margin"] - matrices["baseline_repeat"]["fit"][e][str(c)]["mean_margin"]) for e, c in GUARD_PAIRS)
    eta_ce = max(ETA_FLOOR, 5.0 * ce_repeat_error)
    eta_train_margin = max(ETA_FLOOR, 5.0 * margin_repeat_error)
    decision = _decision(
        baseline,
        rows_by_condition.get("ce_margin_plus1", []),
        matrices["baseline"],
        matrices.get("ce_margin_plus1", matrices["baseline"]),
        eta_e_d,
        eta_ce,
        eta_train_margin,
        candidate_available=bool(projection["candidate_available"]),
    )
    all_rows = [{"state": state, **row} for state in conditions for row in rows_by_condition[state]]
    _write_csv(target / "row_metrics.csv", all_rows)
    deltas: dict[str, Any] = {}
    if projection["candidate_available"]:
        for condition in ("ce_margin_plus1", "ce_margin_minus1"):
            deltas[condition] = {"ce": _delta_matrix(matrices["baseline"], matrices[condition], field="mean_ce"), "margin": _delta_matrix(matrices["baseline"], matrices[condition], field="mean_margin")}
    review = _write_review_pages(target, inputs) if projection["candidate_available"] else []
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": decision["classification"],
        "research_run_identity": identity.to_dict(),
        "scientific_updates": 0,
        "optimizer_updates": 0,
        "phase_state_interventions": 2 if projection["candidate_available"] else 0,
        "formal_measurement_rows": TOTAL_ROWS * len(conditions),
        "population_roles": {"fit": FIT_ROWS, "historic_holdout": HISTORIC_ROWS, "confirmation": CONFIRMATION_ROWS},
        "parent_result_preserved": {"class_guard_status": inputs["class_guard_report"]["status"], "class_guard_classification": inputs["class_guard_report"]["classification"], "parent_ce_status": inputs["parent_report"]["status"], "parent_ce_classification": inputs["parent_report"]["classification"]},
        "preflight": dict(preflight),
        "fit_gradient_construction": {
            "evaluator_guards": list(GUARD_EVALUATORS),
            "class_ce_gradient_source": "saved class_guard attempt_03 gradients, row-mapped and rechecked against original twenty mixed-batch gradients",
            "margin_gradient_definition": gradients["margin"]["gradient_definition"],
            "confirmation_used_for_gradient_or_projection": False,
            "e_d_used_for_direction_construction": False,
            "margin_fit_row_map": gradients["margin"]["fit_row_map"],
            "margin_gradient_component_checks": gradients["margin"]["unique_max_wrong_component_checks"],
            "exact_ties": gradients["margin"]["exact_ties"],
        },
        "projection": {key: value for key, value in projection.items() if key not in {"candidate_equal_rms_direction", "projected_minimum_distance_vector", "actual_float32_displacements"}},
        "baseline_repeat": {"eta_e_d_margin": eta_e_d, "max_fit_class_ce_repeat_error": ce_repeat_error, "eta_fit_ce": eta_ce, "max_fit_class_margin_repeat_error": margin_repeat_error, "eta_fit_train_margin": eta_train_margin},
        "summaries_by_condition_and_role": summaries,
        "fit_confirmation_ce_margin_deltas_by_condition": deltas,
        "confirmation_decision": decision,
        "image_review": review,
        "evidence_artifacts": {"input_population": "input_population.json", "class_gradients": "ce_margin_guard_gradients.pt", "projection": "ce_margin_guard_projection.pt", "actual_displacements": "actual_displacements.pt", "preflight": "preflight.json", "manifest": "resume_manifest.json", "readout_chunks": [_condition_name(condition, index) for condition in conditions for index in range(CHUNK_COUNT)], "row_metrics": "row_metrics.csv", "review_images": [item["path"] for item in review]},
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "elapsed_seconds_observation_only": time.perf_counter() - started,
    }
    return report


def _save_preflight(target: Path, population: Mapping[str, Any], preflight: Mapping[str, Any], receipt: PreflightReceipt, gradients: Mapping[str, Any], projection: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    _require(preflight.get("status") == "READY" and receipt.status == "READY", "cannot save an unverified preflight as READY")
    _require(preflight.get("numerical_scoring", {}).get("required_original_check_set_satisfied") is True, "READY preflight lacks executed numerical scoring evidence")
    finite._write_json_new(target / "input_population.json", dict(population), "fixed CE-margin guarded population")
    finite._write_torch_new(target / "ce_margin_guard_gradients.pt", gradients, "saved CE and new class margin gradients")
    finite._write_torch_new(target / "ce_margin_guard_projection.pt", projection, "nearest CE-margin feasible displacement")
    finite._write_torch_new(target / "actual_displacements.pt", {"planned": {"ce_margin_plus1": projection.get("candidate_equal_rms_direction"), "ce_margin_minus1": None if projection.get("candidate_equal_rms_direction") is None else -projection["candidate_equal_rms_direction"]}, "actual_float32": projection.get("actual_float32_displacements", {})}, "planned and actual float32 phase changes")
    finite._write_json_new(target / "preflight.json", {"schema": "m0_generator_ce_margin_guarded_preflight.v2", "task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": dict(preflight)}, "executed CE-margin preflight")
    finite._write_json_new(target / "resume_manifest.json", dict(manifest), "CE-margin measurement manifest")


def _save_preflight_failure(
    target: Path,
    population: Mapping[str, Any],
    preflight: Mapping[str, Any],
    gradients: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> None:
    _require(preflight.get("status") == "INVALID_HARNESS", "failure evidence requires an invalid preflight status")
    snapshots = dict(preflight.get("_diagnostic_snapshots", {}))
    public_preflight = {key: value for key, value in preflight.items() if not key.startswith("_")}
    finite._write_json_new(target / "input_population.json", dict(population), "fixed CE-margin guarded population")
    finite._write_torch_new(target / "ce_margin_guard_gradients.pt", gradients, "saved CE and new class margin gradients")
    finite._write_torch_new(target / "ce_margin_guard_projection.pt", projection, "nearest CE-margin feasible displacement")
    finite._write_json_new(target / "preflight_diagnostics.json", public_preflight, "failed executed CE-margin preflight diagnostics")
    finite._write_torch_new(
        target / "preflight_numeric_evidence.pt",
        {
            "schema": "m0_generator_ce_margin_guarded_numeric_evidence.v2",
            "task_id": TASK_ID,
            "finite_difference_checks": preflight.get("finite_difference_checks", {}),
            "diagnostic_snapshots": snapshots,
            "projection_candidate": projection.get("candidate_equal_rms_direction"),
            "actual_float32_displacements": projection.get("actual_float32_displacements", {}),
        },
        "all executed finite-difference evidence and row-level diagnostics",
    )


def _load_saved_preflight(target: Path, inputs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], PreflightReceipt]:
    preflight_file = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
    _require(preflight_file.get("schema") == "m0_generator_ce_margin_guarded_preflight.v2" and preflight_file.get("task_id") == TASK_ID and preflight_file.get("status") == "READY", "resume lacks this task's executed v2 READY preflight")
    preflight = preflight_file["preflight"]
    _require(preflight.get("numerical_scoring", {}).get("required_original_check_set_satisfied") is True, "resume preflight lacks verified float64 paired finite-difference evidence")
    selected_mode = preflight.get("numerical_scoring", {}).get("selected_forward_mode")
    _require(selected_mode in {"current_default", "strict_fp32_no_tf32"}, "resume preflight lacks a selected production numerical mode")
    if selected_mode == "strict_fp32_no_tf32":
        adoption = preflight["numerical_scoring"].get("strict_mode_adoption_check", {})
        _require(adoption.get("passed") is True and adoption.get("settings_restored") is True, "strict numerical mode was selected without passing compatibility checks")
    finite_checks = preflight.get("finite_difference_checks", {})
    _require(finite_checks.get("not_applicable") is True or (finite_checks.get("executed_check_count") == 48 and len(finite_checks.get("checks", {})) == 48 and all(item.get("passed") is True for item in finite_checks.get("checks", {}).values())), "resume preflight does not contain all passing original finite-difference checks")
    receipt = PreflightReceipt.from_mapping(preflight_file["receipt"])
    gradients = torch.load(target / "ce_margin_guard_gradients.pt", map_location="cpu", weights_only=True)
    projection = torch.load(target / "ce_margin_guard_projection.pt", map_location="cpu", weights_only=True)
    _require(gradients.get("schema") == GRADIENT_SCHEMA and projection.get("schema") == PROJECTION_SCHEMA, "saved gradients or projection have different measurement semantics")
    ce_now = _saved_ce_gradients(inputs)
    _require(ce_now["mixed_batch_reference_agreement"] == gradients["ce"]["mixed_batch_reference_agreement"], "saved CE gradient reference evidence changed")
    _require(all(torch.equal(ce_now["bundle"]["gradients"][e][c], gradients["ce"]["bundle"]["gradients"][e][c]) for e in GUARD_EVALUATORS for c in range(10)), "saved CE gradient tensor differs from its explicit source")
    margin = gradients["margin"]
    _require(margin["schema"] == GRADIENT_SCHEMA and margin["gradient_definition"].startswith("mean full-autograd true-logit"), "saved margin gradient definition differs")
    _validate_margin_population(margin, inputs["labels_600"][:FIT_ROWS], inputs["source_indices_600"][:FIT_ROWS])
    source = inputs["directions"]["equal_rms_directions"]["S_GD"]
    if projection["candidate_available"]:
        recomputed = _project(source, gradients["ce"]["bundle"]["gradients"], margin["gradients"], float(inputs["directions"]["reference_rms"]))
        _require(torch.allclose(recomputed["projected_minimum_distance_vector"], projection["projected_minimum_distance_vector"], rtol=1e-10, atol=1e-12), "saved projection differs from its fit constraints")
        _require(torch.allclose(recomputed["candidate_equal_rms_direction"], projection["candidate_equal_rms_direction"], rtol=1e-7, atol=1e-12), "saved equal-RMS displacement differs from the recomputed candidate")
    else:
        _require(projection["candidate_equal_rms_direction"] is None, "unavailable projection contains a candidate")
    return preflight, gradients, projection, receipt


def _terminal_report(target: Path, manifest: Mapping[str, Any], identity: ResearchRunIdentity) -> dict[str, Any] | None:
    report_path = target / "report.json"
    if manifest.get("status") == "COMPLETED":
        _require(report_path.is_file(), "completed manifest lacks final report")
    elif not report_path.is_file():
        return None
    expected = {_condition_name(condition, index) for condition in manifest.get("measured_conditions", []) for index in range(CHUNK_COUNT)}
    _require(expected.issubset(set(manifest.get("completed_chunks", []))) and all((target / name).is_file() for name in expected), "terminal report lacks complete immutable chunks")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema") == REPORT_SCHEMA and report.get("task_id") == TASK_ID and report.get("status") == "VALID_COMPLETED" and report.get("research_run_identity") == identity.to_dict(), "saved terminal report identity is invalid")
    if manifest.get("status") != "COMPLETED":
        finite._write_json_replace(target / "resume_manifest.json", {**manifest, "status": "COMPLETED"})
    return report


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    class_guard_dir = _resolve(root, arguments.class_guard_dir)
    target = _resolve(root, arguments.output_dir)
    outputs = (root / "outputs").resolve()
    _require(target != outputs and outputs in target.parents, "output must be a named child of outputs/")
    _require(not (arguments.preflight_only and arguments.resume_manifest), "preflight-only cannot resume")
    _require(arguments.resume_manifest is not None or not target.exists(), "output exists; resume only with its explicit manifest or choose a new attempt")
    inputs = _load_inputs(root, class_guard_dir)
    population = _population_record(inputs, target)
    identity = _identity(inputs, population, target)
    existing = None
    if arguments.resume_manifest is not None:
        _require(target.is_dir(), "resume output directory is missing")
        existing = _validate_manifest(_resolve(root, arguments.resume_manifest), target, identity, population)
        terminal = _terminal_report(target, existing, identity)
        if terminal is not None:
            return terminal

    started = time.perf_counter()
    if existing is None:
        preflight, gradients, projection = _preflight(inputs, target, identity, started)
        run_control = ResearchRunControl(identity=identity, authorization_id=f"{TASK_ID}:authorization:01")
        if preflight.get("status") != "READY":
            target.mkdir(parents=True, exist_ok=False)
            _save_preflight_failure(target, population, preflight, gradients, projection)
            run_control.invalid_harness()
            raise CEMarginDirectionError(
                "executed numerical preflight failed; all completed checks and row-level finite-difference evidence were preserved"
            )
        receipt = _ready_receipt(run_control, preflight)
        if receipt.status != "READY":
            target.mkdir(parents=True, exist_ok=False)
            _save_preflight_failure(target, population, {**preflight, "status": "INVALID_HARNESS"}, gradients, projection)
            run_control.invalid_harness()
            raise CEMarginDirectionError("executed no-update preflight lacked a required READY check; diagnostics were preserved")
        run_control.mark_preflight_ready(receipt)
        manifest = _manifest_seed(identity, population, receipt, preflight, projection)
        if arguments.preflight_only:
            target.mkdir(parents=True, exist_ok=False)
            _save_preflight(target, population, preflight, receipt, gradients, projection, manifest)
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "optimizer_updates": 0, "next_action": "NONE"}
        run_control.start_execution()
        run_control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt)
        _save_preflight(target, population, preflight, receipt, gradients, projection, manifest)
    else:
        preflight, gradients, projection, receipt = _load_saved_preflight(target, inputs)
        _require(existing.get("preflight") == preflight and existing.get("receipt") == receipt.to_dict(), "resume manifest differs from its saved READY preflight")
        run_control = ResearchRunControl(identity=identity, authorization_id=receipt.authorization_id)
        run_control.mark_preflight_ready(receipt)
        run_control.start_execution()
        run_control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt, resume=True)
        manifest = dict(existing)
        manifest["status"] = "RUNNING"
        finite._write_json_replace(target / "resume_manifest.json", manifest)

    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA became unavailable before frozen measurement")
    runtime_dll = direction._runtime_dll_probe()
    _require(runtime_dll.get("status") in {"already_present", "prepended", "not_applicable"}, "CUDA runtime DLL readiness changed before measurement")
    consumer, parameters = direction._new_consumer(inputs, "B_class_ce", device)
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    evaluators = base.load_evaluators(inputs["paths"], device)
    for model in evaluators.values():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    runtime_before = finite._snapshot(consumer.runtime)
    evaluator_before = {name: finite._snapshot(model) for name, model in evaluators.items()}
    frozen_inputs_before = direction._consumer_frozen_inputs_snapshot(consumer)
    optimizer_before = direction._cpu_tree(inputs["optimizers"]["B_class_ce"])
    phase = parameters["generator.phase"]
    baseline_phase = phase.detach().cpu().clone()
    conditions = list(manifest["measured_conditions"])
    try:
        with _production_numeric_mode(preflight):
            _measure_all_conditions(
                target,
                manifest,
                conditions,
                inputs,
                consumer,
                evaluators,
                parameters,
                baseline_phase,
                projection,
                runtime_before,
                evaluator_before,
                frozen_inputs_before,
                optimizer_before,
            )
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"
        finite._write_json_replace(target / "resume_manifest.json", manifest)
        run_control.interrupted()
        raise
    except Exception:
        finite._restore_phase(phase, baseline_phase)
        manifest["status"] = "INVALID_HARNESS"
        finite._write_json_replace(target / "resume_manifest.json", manifest)
        run_control.invalid_harness()
        raise

    try:
        report = _finalize(target, inputs, identity, preflight, gradients, projection, started)
        report["final_frozen_state"] = _snapshot_checks(consumer, evaluators, parameters, runtime_before, evaluator_before, frozen_inputs_before, optimizer_before, inputs)
    except Exception:
        manifest["status"] = "INVALID_HARNESS"
        finite._write_json_replace(target / "resume_manifest.json", manifest)
        run_control.invalid_harness()
        raise
    finite._write_json_new(target / "report.json", report, "one CE-margin guarded frozen direction decision")
    manifest["status"] = "COMPLETED"
    finite._write_json_replace(target / "resume_manifest.json", manifest)
    run_control.complete()
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-guard-dir", type=Path, default=DEFAULT_CLASS_GUARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        result = _run(args)
    except (CEMarginDirectionError, class_guard.ClassGuardedDirectionError, shared.SharedDirectionError, finite.FiniteResponseError, direction.DirectionDiagnosticError, base.DiagnosticError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        target = _resolve(_root(), args.output_dir)
        failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "reason": f"{type(error).__name__}: {error}", "scientific_updates": 0, "optimizer_updates": 0, "next_action": "NONE"}
        outputs = (_root() / "outputs").resolve()
        try:
            if target != outputs and outputs in target.parents:
                target.mkdir(parents=True, exist_ok=True)
                preserved = [
                    name
                    for name in (
                        "preflight_diagnostics.json",
                        "preflight_numeric_evidence.pt",
                        "ce_margin_guard_gradients.pt",
                        "ce_margin_guard_projection.pt",
                    )
                    if (target / name).is_file()
                ]
                if preserved:
                    failure["evidence_artifacts"] = preserved
                if not (target / "report.json").exists() and not (target / "failure.json").exists():
                    finite._write_json_new(target / "failure.json", failure, "CE-margin guarded invalid harness")
                manifest_path = target / "resume_manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("status") not in {"COMPLETED", "INVALID_HARNESS"}:
                        manifest["status"] = "INVALID_HARNESS"
                        finite._write_json_replace(manifest_path, manifest)
        except (OSError, RuntimeError, ValueError):
            pass
        print(json.dumps(failure, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
