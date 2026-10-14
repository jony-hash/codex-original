"""Run the fixed 300-update small-versus-large LightGen Generator screen."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import re
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.checkpoint import checkpoint

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as bogt
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    LightGen2DForwardResult,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from ldm_project.v2.optics.propagation import angular_spectrum_propagate
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_balanced_reconstruction_v1 as balanced
from scripts.research.lightgen import execute_lightgen_2d_generator_semantic_target_comparison_v1 as semantic


TASK_ID = "M0_GENERATOR_LARGE_CAPACITY_QUICK_COMPARISON_V1"
REPORT_SCHEMA = "m0_generator_large_capacity_quick_comparison_report.v1"
PREFLIGHT_SCHEMA = "m0_generator_large_capacity_quick_comparison_preflight.v1"
CHECKPOINT_SCHEMA = "m0_generator_large_capacity_quick_comparison_checkpoint.v1"
DEFAULT_PARENT = Path("outputs/20261004_m0_generator_semantic_target_comparison_v1/runs/attempt_01")
DEFAULT_OUTPUT = Path("outputs/20261007_m0_generator_large_capacity_quick_comparison_v1/runs/attempt_03")
FIT_ROWS, HISTORIC_ROWS, CONFIRMATION_ROWS, TOTAL_ROWS = 200, 100, 300, 600
BATCH_SIZE, BATCH_COUNT, UPDATES_PER_ARM, CHECKPOINT_INTERVAL = 10, 20, 300, 100
LARGE_GRADIENT_MICROBATCH_SIZE = 2
SEED = 2026100702
ARM_NAMES = ("A_small", "B_large")
EVALUATOR_NAMES = ("E_S", "E_I", "E_C", "E_D")
TRAINING_EVALUATORS = ("E_S", "E_I", "E_C")
FOCUS_CLASSES = (1, 7)
DISPLAY_CLASSES = (1, 7, 0, 4, 8)
COMPARE_CLASSES = (0, 4, 8)
SOURCE_STEP = 2500
GENERATOR_LEARNING_RATE = 1.0e-3
GENERATOR_BETAS = (0.9, 0.999)
GENERATOR_EPS = 1.0e-8
GENERATOR_WEIGHT_DECAY = 0.0
PHASE_OLD_COUNT = 2 * 100 * 100
PHASE_NEW_COUNT = 2 * 400 * 400
PHASE_LARGE_COUNT = PHASE_OLD_COUNT + PHASE_NEW_COUNT
DISPLAY_STATES = ("target", "A_start", "B_start", "A_step0300", "B_step0300")
CHECKPOINT_PATTERN = re.compile(r"arm_(a|b)_step(\d{4})\.pt$")
RESOURCE_TELEMETRY_NAME = "resource_telemetry_large_capacity.jsonl"
RTOL, ATOL = 2.0e-5, 2.0e-7
DISK_SAFETY_BYTES = 1_610_612_736


class LargeCapacityRunError(RuntimeError):
    """Invalid task input, violated execution invariant, or incomplete result."""


class TrainingInterrupted(RuntimeError):
    """A safe boundary was checkpointed after an interrupt."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LargeCapacityRunError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _cpu_tree(value: Any) -> Any:
    return balanced._cpu_tree(value)


def _tree_equal(left: Any, right: Any) -> bool:
    return balanced._tree_equal(left, right)


def _state_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return balanced._state_snapshot(module)


def _assert_state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], name: str) -> None:
    balanced._assert_state_equal(left, right, name)


def _rng_state() -> dict[str, Any]:
    return balanced._rng_state()


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    balanced._restore_rng_state(state)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def strict_fp32_mode():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        with torch.autocast(device_type="cuda", enabled=False):
            yield {
                "dtype": "float32",
                "autocast": False,
                "cuda_matmul_tf32": False,
                "cudnn_tf32": False,
                "float32_matmul_precision": "highest",
            }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.set_float32_matmul_precision(old_precision)


class ExpandedGenerator(nn.Module):
    """Four phase planes with two transparent 400x400 windows and no middle crop."""

    def __init__(
        self,
        old_phase: torch.Tensor,
        *,
        propagation_grid_size: int = 2048,
        active_size: int = 100,
        expanded_size: int = 400,
        half_spacing_m: float = 0.75e-3,
    ) -> None:
        super().__init__()
        _require(tuple(old_phase.shape) == (2, 1, active_size, active_size), "old Generator phase layout differs")
        _require(propagation_grid_size >= expanded_size >= active_size, "expanded phase window does not fit propagation grid")
        self.propagation_grid_size = propagation_grid_size
        self.active_size = active_size
        self.expanded_size = expanded_size
        self.half_spacing_m = half_spacing_m
        old_count = 2 * active_size * active_size
        new_count = 2 * expanded_size * expanded_size
        self.phase = nn.Parameter(torch.zeros(old_count + new_count, dtype=old_phase.dtype, device=old_phase.device))
        with torch.no_grad():
            self.phase[:old_count].copy_(old_phase.reshape(-1))
        self.last_extra_plane_powers: torch.Tensor | None = None

    @property
    def old_phase_count(self) -> int:
        return 2 * self.active_size * self.active_size

    @property
    def phase_layout(self) -> dict[str, Any]:
        return {
            "parameter_shape": [self.phase.numel()],
            "parameter_count": int(self.phase.numel()),
            "old_planes": [
                {"slice": [0, self.active_size**2], "shape": [1, self.active_size, self.active_size], "z_relative_m": 0.0},
                {"slice": [self.active_size**2, 2 * self.active_size**2], "shape": [1, self.active_size, self.active_size], "z_relative_m": 1.5e-3},
            ],
            "new_planes": [
                {"slice": [self.old_phase_count, self.old_phase_count + self.expanded_size**2], "shape": [1, self.expanded_size, self.expanded_size], "z_relative_m": 0.75e-3},
                {"slice": [self.old_phase_count + self.expanded_size**2, self.phase.numel()], "shape": [1, self.expanded_size, self.expanded_size], "z_relative_m": 2.25e-3},
            ],
            "propagation_grid_size": self.propagation_grid_size,
            "new_phase_support_fov_m": [self.expanded_size * 2e-6, self.expanded_size * 2e-6],
            "outside_new_phase_window": "transmission=1, no additional crop",
        }

    def _old_layer_phase(self, index: int) -> torch.Tensor:
        start = index * self.active_size**2
        return self.phase[start : start + self.active_size**2].view(1, self.active_size, self.active_size)

    def _new_layer_phase(self, index: int) -> torch.Tensor:
        start = self.old_phase_count + index * self.expanded_size**2
        return self.phase[start : start + self.expanded_size**2].view(1, self.expanded_size, self.expanded_size)

    @staticmethod
    def _modulate(field: ComplexField, phase: torch.Tensor, operation: str, upstream: str) -> ComplexField:
        transfer = torch.polar(torch.ones_like(phase), phase).to(field.tensor.dtype)
        return ComplexField(
            field.tensor * transfer,
            field.plane,
            OperationProvenance(operation, upstream, None, {"phase_shape": list(phase.shape)}),
        )

    def _pad_active(self, field: ComplexField, plane_id: str) -> ComplexField:
        grid = self.propagation_grid_size
        dy = (grid - self.active_size) // 2
        dx = (grid - self.active_size) // 2
        padded_plane = field.plane.with_plane(
            f"{plane_id}_padded",
            shape=(grid, grid),
            coordinate_frame=f"{plane_id}_propagation_grid",
        )
        return ComplexField(
            F.pad(field.tensor, (dx, dx, dy, dy)),
            padded_plane,
            OperationProvenance(
                "finite_crop_center_pad",
                field.plane.plane_id,
                None,
                {"active_shape": [self.active_size, self.active_size], "propagation_grid": [grid, grid]},
            ),
        )

    def _expanded_phase_deviation(self, field: ComplexField, phase_window: torch.Tensor) -> tuple[ComplexField, torch.Tensor]:
        grid = self.propagation_grid_size
        size = self.expanded_size
        pad = (grid - size) // 2
        phase_full = F.pad(phase_window, (pad, pad, pad, pad))
        transfer = torch.polar(torch.ones_like(phase_full), phase_full).to(field.tensor.dtype)
        deviation = ComplexField(
            field.tensor * (transfer - 1.0),
            field.plane,
            OperationProvenance(
                "centered_phase_window_deviation_from_transmission_one",
                field.plane.plane_id,
                None,
                {
                    "active_phase_shape": [size, size],
                    "propagation_grid_shape": [grid, grid],
                    "outside_window_transmission": 1.0,
                    "intermediate_crop": False,
                },
            ),
        )
        power = bogt.physical_plane_power(field)
        return deviation, power.detach()

    def _propagate_interval_with_inserted_phase(
        self,
        field: ComplexField,
        index: int,
        prefix: str,
    ) -> tuple[ComplexField, torch.Tensor]:
        """Apply an inserted phase through a baseline-plus-correction identity.

        ``P(d/2) M P(d/2) x = P(d) x + P(d/2) (M - I) P(d/2) x``.
        This preserves the physical phase-plane derivative while making the
        zero-phase path use the same single-step production propagation as the
        original Generator, avoiding a false mismatch from two FP32 FFT pairs.
        """
        phase_window = self._new_layer_phase(index)

        def calculate(input_tensor: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            input_field = ComplexField(input_tensor, field.plane, field.provenance)
            padded = self._pad_active(input_field, f"{prefix}_padded_for_split_phase")
            midpoint = angular_spectrum_propagate(
                padded,
                self.half_spacing_m,
                f"{prefix}_inserted_plane_{index + 1}",
            )
            deviation, power_value = self._expanded_phase_deviation(midpoint, phase)
            baseline = angular_spectrum_propagate(
                padded,
                2.0 * self.half_spacing_m,
                f"{prefix}_original_spacing_baseline_{index + 1}",
            )
            correction = angular_spectrum_propagate(
                deviation,
                self.half_spacing_m,
                f"{prefix}_inserted_phase_correction_{index + 1}",
            )
            return baseline.tensor + correction.tensor, power_value

        if torch.is_grad_enabled() and (field.tensor.requires_grad or phase_window.requires_grad):
            combined_tensor, power = checkpoint(calculate, field.tensor, phase_window, use_reentrant=False)
        else:
            combined_tensor, power = calculate(field.tensor, phase_window)
        padded_plane = field.plane.with_plane(
            f"{prefix}_padded_for_split_phase_padded",
            shape=(self.propagation_grid_size, self.propagation_grid_size),
            coordinate_frame=f"{prefix}_padded_for_split_phase_propagation_grid",
        )
        baseline_plane = padded_plane.with_plane(
            f"{prefix}_original_spacing_baseline_{index + 1}",
            z_m=field.plane.z_m + 2.0 * self.half_spacing_m,
        )
        combined = ComplexField(
            combined_tensor,
            baseline_plane,
            OperationProvenance(
                "baseline_plus_inserted_phase_propagation_correction",
                f"{prefix}_inserted_plane_{index + 1}",
                self.half_spacing_m,
                {
                    "baseline_distance_m": 2.0 * self.half_spacing_m,
                    "correction_distance_m": self.half_spacing_m,
                    "outside_window_transmission": 1.0,
                    "intermediate_crop": False,
                },
            ),
        )
        return combined, power

    def _crop_active(self, field: ComplexField, relative_id: str) -> tuple[ComplexField, torch.Tensor]:
        grid = self.propagation_grid_size
        pad = (grid - self.active_size) // 2
        cropped_tensor = field.tensor[..., pad : pad + self.active_size, pad : pad + self.active_size]
        active_plane = field.plane.with_plane(
            relative_id,
            shape=(self.active_size, self.active_size),
            coordinate_frame="generator",
        )
        active = ComplexField(
            cropped_tensor,
            active_plane,
            OperationProvenance(
                "finite_crop_hard_aperture",
                field.plane.plane_id,
                self.half_spacing_m,
                {"active_shape": [self.active_size, self.active_size]},
            ),
        )
        retained = active.tensor.abs().square().sum(dim=(-3, -2, -1)) / field.tensor.abs().square().sum(dim=(-3, -2, -1)).clamp_min(1e-12)
        return active, retained

    def forward(self, field: ComplexField, prefix: str):
        _require(tuple(field.tensor.shape[-2:]) == (self.active_size, self.active_size), "expanded Generator entrance shape differs")
        first = self._modulate(field, self._old_layer_phase(0), "phase_only_modulation_original_100", field.plane.plane_id)
        propagated_150, power_first = self._propagate_interval_with_inserted_phase(first, 0, f"{prefix}_first_interval")
        active_150, retention_first = self._crop_active(propagated_150, f"{prefix}_original_plane_150mm")

        second = self._modulate(active_150, self._old_layer_phase(1), "phase_only_modulation_original_100", active_150.plane.plane_id)
        propagated_300, power_second = self._propagate_interval_with_inserted_phase(second, 1, f"{prefix}_second_interval")
        final, retention_second = self._crop_active(propagated_300, f"{prefix}_original_exit_300mm")
        self.last_extra_plane_powers = torch.stack((power_first, power_second), dim=1)
        crop_retention = torch.stack((retention_first, retention_second), dim=1)
        return final, (active_150, final), {
            "crop_retention": crop_retention,
            "phase": self.phase,
            "extra_plane_powers": self.last_extra_plane_powers,
            "phase_layout": self.phase_layout,
        }


def _copy_phase_(consumer: Any, source_state: Mapping[str, torch.Tensor], *, arm: str) -> None:
    state = source_state.get("trainable_parameter_state")
    _require(isinstance(state, Mapping) and set(state) == {"encoder.phase", "generator.phase"}, "source B endpoint phase state is incomplete")
    _require(tuple(state["encoder.phase"].shape) == (2, 1, 100, 100), "source Encoder phase shape differs")
    _require(tuple(state["generator.phase"].shape) == (2, 1, 100, 100), "source Generator phase shape differs")
    named = dict(consumer.runtime.named_parameters())
    with torch.no_grad():
        named["encoder.phase"].copy_(state["encoder.phase"].to(consumer.device))
        if arm == "A_small":
            named["generator.phase"].copy_(state["generator.phase"].to(consumer.device))
        else:
            old_generator = consumer.runtime.generator
            expanded = ExpandedGenerator(
                state["generator.phase"].to(consumer.device),
                propagation_grid_size=consumer.runtime.config.propagation_grid_size,
            ).to(consumer.device)
            _require(torch.equal(expanded.phase[:PHASE_OLD_COUNT].cpu(), state["generator.phase"].reshape(-1)), "expanded Generator did not retain source phases exactly")
            _require(torch.count_nonzero(expanded.phase[PHASE_OLD_COUNT:]).item() == 0, "new expanded phases did not start at zero")
            consumer.runtime.generator = expanded
            del old_generator
    consumer.runtime.eval()


def _parameters(consumer: Any) -> dict[str, nn.Parameter]:
    values = dict(consumer.runtime.named_parameters())
    _require(tuple(values) == ("encoder.phase", "generator.phase"), f"runtime trainable parameter groups changed: {tuple(values)}")
    _require(tuple(values["encoder.phase"].shape) == (2, 1, 100, 100), "Encoder phase shape changed")
    if isinstance(consumer.runtime.generator, ExpandedGenerator):
        _require(tuple(values["generator.phase"].shape) == (PHASE_LARGE_COUNT,), "large Generator phase shape changed")
    else:
        _require(tuple(values["generator.phase"].shape) == (2, 1, 100, 100), "small Generator phase shape changed")
    return values


def _source_checkpoint(parent: Path) -> tuple[dict[str, Any], Path]:
    path = parent / "arm_b_step0300.pt"
    _require(path.is_file(), f"explicit parent B endpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(payload.get("schema") == "m0_generator_semantic_target_comparison_checkpoint.v1", "parent B checkpoint schema mismatch")
    _require(payload.get("task_id") == "M0_GENERATOR_SEMANTIC_TARGET_COMPARISON_V1", "parent B checkpoint task mismatch")
    _require(payload.get("arm") == "B_class_ce" and payload.get("completed_updates") == 300, "parent checkpoint is not the completed B endpoint")
    _require(payload.get("source_checkpoint_step") == 2200, "parent B endpoint source step changed")
    _require(payload.get("optimizer_step_by_parameter") == {"encoder.phase": 2200, "generator.phase": 2500}, "parent B source Adam steps differ")
    optimizer_state = payload.get("optimizer_state")
    _require(isinstance(optimizer_state, Mapping) and len(optimizer_state.get("state", {})) == 2, "parent B source Adam state is incomplete")
    groups = optimizer_state.get("param_groups")
    _require(isinstance(groups, list) and len(groups) == 1 and len(groups[0].get("params", [])) == 2, "parent B source Adam parameter order changed")
    actual_steps = [
        int(float(optimizer_state["state"][parameter_id]["step"].item()))
        for parameter_id in groups[0]["params"]
    ]
    _require(actual_steps == [2200, 2500], f"loaded parent B Adam steps differ from recorded values: {actual_steps}")
    return dict(payload), path


def _load_inputs(root: Path, parent: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], dict[str, Any]]:
    population_path = parent / "input_population.json"
    report_path = parent / "report.json"
    _require(population_path.is_file() and report_path.is_file(), "parent experiment population and report are required")
    parent_population = json.loads(population_path.read_text(encoding="utf-8"))
    parent_report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(parent_population.get("task_id") == "M0_GENERATOR_SEMANTIC_TARGET_COMPARISON_V1", "parent population task identity differs")
    _require(parent_report.get("task_id") == "M0_GENERATOR_SEMANTIC_TARGET_COMPARISON_V1", "parent report task identity differs")
    _require(parent_report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"}, "parent experiment lacks a terminal scientific status")
    _require(len(parent_population.get("source_row_indices", [])) == TOTAL_ROWS, "parent population must contain its fixed 600 rows")
    _require(len(parent_population.get("labels", [])) == TOTAL_ROWS and len(parent_population.get("roles", [])) == TOTAL_ROWS, "parent population labels/roles differ")
    _require(parent_population["roles"] == ["fit"] * FIT_ROWS + ["historic_holdout"] * HISTORIC_ROWS + ["confirmation"] * CONFIRMATION_ROWS, "parent row roles or order differ")
    counts = {role: parent_population["roles"].count(role) for role in set(parent_population["roles"])}
    _require(counts == {"fit": FIT_ROWS, "historic_holdout": HISTORIC_ROWS, "confirmation": CONFIRMATION_ROWS}, "parent population role counts differ")
    raw_paths = parent_population.get("input_paths")
    _require(isinstance(raw_paths, Mapping), "parent population input paths are missing")
    names = ("step1200", "step2200", "readout1200", "readout2200", "training_data", "e_s", "e_i", "e_c", "e_d")
    _require(set(raw_paths) == set(names), "parent input path set differs from the completed semantic-target task")
    paths = {name: _resolve(root, raw_paths[name]) for name in names}
    for name, path in paths.items():
        _require(path.is_file(), f"required parent input is missing: {name}: {path}")
    inputs = semantic._load_inputs(root, paths)
    inputs["input_paths"] = {name: paths[name].as_posix() for name in names}
    _require(inputs["source_indices"].tolist() == parent_population["source_row_indices"], "regenerated fixed TRAIN indices differ from parent population")
    _require(inputs["labels"].tolist() == parent_population["labels"], "regenerated TRAIN labels differ from parent population")
    _require(inputs["images"].shape == (TOTAL_ROWS, 1, 28, 28), "fixed population image shape changed")
    _require(torch.unique(inputs["source_indices"]).numel() == TOTAL_ROWS, "fixed population contains duplicate source rows")
    _require(torch.bincount(inputs["labels"][:FIT_ROWS], minlength=10).tolist() == [20] * 10, "fit population is not 20 rows per class")
    _require(torch.bincount(inputs["labels"][FIT_ROWS : FIT_ROWS + HISTORIC_ROWS], minlength=10).tolist() == [10] * 10, "historic population is not 10 rows per class")
    _require(torch.bincount(inputs["labels"][FIT_ROWS + HISTORIC_ROWS :], minlength=10).tolist() == [30] * 10, "confirmation population is not 30 rows per class")
    source_state, source_path = _source_checkpoint(parent)
    optimizer_steps = source_state.get("optimizer_step_by_parameter")
    _require(optimizer_steps == {"encoder.phase": 2200, "generator.phase": 2500}, "source optimizer lineage differs from the declared endpoint")
    state = source_state["trainable_parameter_state"]
    _require(torch.isfinite(state["encoder.phase"]).all() and torch.isfinite(state["generator.phase"]).all(), "source B phase state is nonfinite")
    _require(torch.equal(source_state["references"]["power_reference"], inputs["references"]["power_reference"]), "parent source and shared reference power differ")
    for key, expected in (("alpha", semantic.ALPHA), ("beta", semantic.BETA), ("gamma", semantic.GAMMA)):
        _require(math.isclose(float(source_state["weights"][key]), expected, rel_tol=0.0, abs_tol=1e-15), f"source BOGT {key} differs")
    _require(math.isclose(float(source_state["weights"].get("delta_sic", semantic.DELTA_SIC)), semantic.DELTA_SIC, rel_tol=0.0, abs_tol=1e-15), "source delta_SIC differs")
    _require(math.isclose(semantic.KAPPA, 0.6936758419764932, rel_tol=0.0, abs_tol=1e-15), "semantic kappa differs")
    expected_population = {
        "task_id": TASK_ID,
        "parent_dir": parent.as_posix(),
        "source_checkpoint": source_path.as_posix(),
        "source_checkpoint_step": SOURCE_STEP,
        "source_optimizer_steps": dict(optimizer_steps),
        "source_row_indices": list(parent_population["source_row_indices"]),
        "labels": list(parent_population["labels"]),
        "roles": list(parent_population["roles"]),
        "fit_rows": FIT_ROWS,
        "historic_comparison_rows": HISTORIC_ROWS,
        "confirmation_rows": CONFIRMATION_ROWS,
        "fit_batch_size": BATCH_SIZE,
        "fit_batch_count": BATCH_COUNT,
        "updates_per_arm": UPDATES_PER_ARM,
        "seed": SEED,
        "input_paths": {name: paths[name].as_posix() for name in names},
        "source_task_terminal_status": parent_report["status"],
        "source_task_classification": parent_report.get("classification"),
        "confirmation_is_canonical_train_not_claimed_unseen": True,
    }
    return inputs, expected_population, paths, source_state


def _new_endpoint(inputs: Mapping[str, Any], source_state: Mapping[str, Any], arm: str, device: torch.device):
    source_model = inputs["payloads"]["step2200"]
    consumer, source_optimizer, _initial_parameters, references, weights = diagnostic._new_endpoint(
        "step2200",
        source_model,
        inputs["references"],
        inputs["weights"],
        device,
    )
    _require(consumer.runtime.config.propagation_grid_size == 2048, "large-capacity comparison requires the production 2048 grid")
    _require(math.isclose(consumer.weights.alpha, 0.01596223111734585, rel_tol=0.0, abs_tol=1e-15), "BOGT alpha differs")
    _require(math.isclose(consumer.weights.beta, 2.4509953404250195, rel_tol=0.0, abs_tol=1e-15), "BOGT beta differs")
    _require(math.isclose(consumer.weights.gamma, 0.5330424284576345, rel_tol=0.0, abs_tol=1e-15), "BOGT gamma differs")
    _require(math.isclose(semantic.KAPPA, 0.6936758419764932, rel_tol=0.0, abs_tol=1e-15), "semantic kappa differs")
    _require(_tree_equal(references.to_payload(), bogt.FrozenBogtReferences.from_payload(source_state["references"]).to_payload()), "parent B reference payload differs from source authority")
    _require(_tree_equal(weights.to_payload(), bogt.BogtLossWeights.from_payload(source_state["weights"]).to_payload()), "parent B BOGT weights differ from source authority")
    _copy_phase_(consumer, source_state, arm=arm)
    parameters = _parameters(consumer)
    optimizer = _new_generator_optimizer(parameters["generator.phase"])
    del source_optimizer
    return consumer, optimizer, parameters


def _new_generator_optimizer(phase: nn.Parameter) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(
        [phase],
        lr=GENERATOR_LEARNING_RATE,
        betas=GENERATOR_BETAS,
        eps=GENERATOR_EPS,
        weight_decay=GENERATOR_WEIGHT_DECAY,
    )
    _require(len(optimizer.state) == 0 and len(optimizer.param_groups) == 1, "Generator Adam was not freshly reset")
    _require(len(optimizer.param_groups[0]["params"]) == 1 and optimizer.param_groups[0]["params"][0] is phase, "Generator Adam parameter scope differs")
    return optimizer


def training_batch(images: torch.Tensor, labels: torch.Tensor, batch_index: int, *, data_role: str = "train"):
    _require(data_role == "train", "only canonical TRAIN fit rows may enter training")
    _require(images.shape == (TOTAL_ROWS, 1, 28, 28) and labels.shape == (TOTAL_ROWS,), "fixed 600-row data/role contract changed")
    _require(0 <= batch_index < BATCH_COUNT, "training batch index is outside the fixed 20 batches")
    start, stop = batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE
    _require(stop <= FIT_ROWS, "holdout or confirmation row would enter a training batch")
    selected_labels = labels[start:stop]
    _require(torch.bincount(selected_labels, minlength=10).tolist() == [1] * 10, "fixed batch is not one row from each class")
    return images[start:stop], selected_labels, torch.arange(start, stop, dtype=torch.long)


def _evaluator_logits(model: nn.Module, view: torch.Tensor, name: str) -> torch.Tensor:
    output = model(view)
    if name == "E_D":
        return output
    _require(isinstance(output, (tuple, list)) and len(output) == 2, f"{name} output surface changed")
    return output[1]


def _gradient(loss: torch.Tensor, parameters: Mapping[str, nn.Parameter], *, retain_graph: bool):
    names = ("encoder.phase", "generator.phase")
    grads = torch.autograd.grad(
        loss,
        tuple(parameters[name] for name in names),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return {
        name: torch.zeros_like(parameters[name]) if value is None else value.detach()
        for name, value in zip(names, grads, strict=True)
    }


def _loss_gradients(consumer: Any, parameters: Mapping[str, nn.Parameter], evaluators: Mapping[str, nn.Module], images: torch.Tensor, labels: torch.Tensor):
    breakdown = consumer.forward_loss(images, data_role=bogt.TRAIN_ROLE)
    result = breakdown.result
    raw = result.raw_intensity
    target_raw = result.image_input.slm_intensity
    view = canonical.cmos_view(raw)
    losses: dict[str, torch.Tensor] = {
        "L_dist": breakdown.L_dist,
        "L_rec": (raw - target_raw).square().mean(),
        "L_op": breakdown.L_op,
    }
    for name in TRAINING_EVALUATORS:
        logits = _evaluator_logits(evaluators[name], view, name)
        losses[f"{name}_ce"] = F.cross_entropy(logits, labels.to(logits.device), reduction="mean")
    grad_names = ("L_dist", "L_rec", "L_op", "E_S_ce", "E_I_ce", "E_C_ce")
    gradients = {
        name: _gradient(losses[name], parameters, retain_graph=index < len(grad_names) - 1)
        for index, name in enumerate(grad_names)
    }
    semantic_gradients = {name: gradients[f"{name}_ce"] for name in TRAINING_EVALUATORS}
    canonical_grad, unavailable = diagnostic._safe_canonical_gradient(
        {key: semantic_gradients[name] for name, key in zip(TRAINING_EVALUATORS, ("L_S", "L_I", "L_C"), strict=True)}
    )
    _require(canonical_grad is not None, f"canonical three-evaluator semantic direction is unavailable: {unavailable}")
    combined = {}
    for name in ("encoder.phase", "generator.phase"):
        combined[name] = (
            semantic.ALPHA * gradients["L_dist"][name]
            + semantic.BETA * gradients["L_rec"][name]
            + semantic.GAMMA * gradients["L_op"][name]
            + semantic.KAPPA * canonical_grad[name]
        )
        _require(bool(torch.isfinite(combined[name]).all().item()), f"{name} combined direction is nonfinite")
    _require(all(parameter.grad is None for parameter in parameters.values()), "read-only autograd populated optical .grad")
    return breakdown.plane_powers.detach(), losses, gradients, combined


def _dist_loss_and_ols_gradient(coefficients: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate the exact logical-batch BOGT distribution loss at OLS output."""
    ols = coefficients.detach().requires_grad_(True)
    u = bogt.statistical_variable(ols, scale)
    q = bogt.q_from_batch(u)
    prior = bogt.fixed_prior(device=ols.device, dtype=torch.float64)
    loss = bogt.truncated_normal_positive_kl(q, prior).mean()
    gradient = torch.autograd.grad(loss, ols)[0]
    _require(bool(torch.isfinite(gradient.real).all().item()) and bool(torch.isfinite(gradient.imag).all().item()), "logical-batch L_dist OLS gradient is nonfinite")
    return loss.detach(), gradient.detach()


def _verify_large_microbatch_gradient_identity() -> dict[str, Any]:
    """Check the batch-coupled OLS proxy against full autograd and finite differences."""
    generator = torch.Generator(device="cpu").manual_seed(2026100703)
    theta = torch.tensor([0.18, -0.11], dtype=torch.float64, requires_grad=True)
    features = torch.randn((BATCH_SIZE, bogt.OLS_PORT_COUNT), generator=generator, dtype=torch.float64)
    offset = torch.linspace(-0.2, 0.3, bogt.OLS_PORT_COUNT, dtype=torch.float64).view(1, -1)
    scale = torch.linspace(0.4, 1.6, bogt.OLS_PORT_COUNT, dtype=torch.float64)

    def coefficients(value: torch.Tensor) -> torch.Tensor:
        return torch.complex(value[0] * features + offset, value[1] * features.flip(0) - 0.3 * offset)

    def loss_for(value: torch.Tensor) -> torch.Tensor:
        q = bogt.q_from_batch(bogt.statistical_variable(coefficients(value), scale))
        prior = bogt.fixed_prior(device=value.device, dtype=torch.float64)
        return bogt.truncated_normal_positive_kl(q, prior).mean()

    reference_loss = loss_for(theta)
    reference_gradient = torch.autograd.grad(reference_loss, theta)[0]
    saved_loss, d_loss_d_ols = _dist_loss_and_ols_gradient(coefficients(theta.detach()), scale)
    micro_gradient = torch.zeros_like(theta)
    for start in range(0, BATCH_SIZE, LARGE_GRADIENT_MICROBATCH_SIZE):
        stop = min(BATCH_SIZE, start + LARGE_GRADIENT_MICROBATCH_SIZE)
        z_chunk = coefficients(theta)[start:stop]
        proxy = torch.real(torch.conj(d_loss_d_ols[start:stop]) * z_chunk).sum()
        micro_gradient += torch.autograd.grad(proxy, theta, retain_graph=stop < BATCH_SIZE)[0]
    finite_difference_epsilon = 1.0e-5
    plus = theta.detach().clone()
    minus = theta.detach().clone()
    plus[0] += finite_difference_epsilon
    minus[0] -= finite_difference_epsilon
    finite_difference = float((loss_for(plus) - loss_for(minus)).item() / (2.0 * finite_difference_epsilon))
    wrong = torch.zeros_like(theta)
    for start in range(0, BATCH_SIZE, LARGE_GRADIENT_MICROBATCH_SIZE):
        stop = min(BATCH_SIZE, start + LARGE_GRADIENT_MICROBATCH_SIZE)
        z_chunk = coefficients(theta)[start:stop]
        incorrect_row_gradient = d_loss_d_ols.mean(dim=0, keepdim=True).expand_as(z_chunk)
        wrong_proxy = torch.real(torch.conj(incorrect_row_gradient) * z_chunk).sum()
        wrong += torch.autograd.grad(wrong_proxy, theta, retain_graph=stop < BATCH_SIZE)[0]
    max_error = float((micro_gradient - reference_gradient).abs().max().item())
    finite_difference_error = abs(float(reference_gradient[0].item()) - finite_difference)
    _require(torch.allclose(saved_loss, reference_loss.detach(), rtol=1e-13, atol=1e-13), "microbatch OLS proxy changed logical-batch L_dist")
    _require(max_error <= 2e-11, f"microbatch OLS chain rule differs from full-batch autograd: {max_error}")
    _require(finite_difference_error <= 2e-7 + 2e-5 * abs(finite_difference), f"logical-batch L_dist gradient differs from finite difference: {finite_difference_error}")
    _require(not torch.allclose(wrong, reference_gradient, rtol=1e-4, atol=1e-7), "negative control failed to detect incorrect row-gradient broadcasting")
    return {
        "passed": True,
        "logical_batch_rows": BATCH_SIZE,
        "microbatch_rows": LARGE_GRADIENT_MICROBATCH_SIZE,
        "loss_abs_error": abs(float(saved_loss.item() - reference_loss.detach().item())),
        "max_gradient_abs_error_vs_full_batch_autograd": max_error,
        "gradient_abs_error_vs_central_difference": finite_difference_error,
        "wrong_broadcast_negative_control_rejected": True,
    }


def _large_microbatch_loss_gradients(
    consumer: Any,
    parameters: Mapping[str, nn.Parameter],
    evaluators: Mapping[str, nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
):
    """Accumulate exact logical-batch gradients with bounded optical microbatches.

    L_dist couples all ten OLS rows through its batch moments. Its derivative is
    first computed at the complete no-grad OLS batch, then chained through each
    independently recomputed optical microbatch. The other objectives are row
    means and are accumulated with their exact row-count weights.
    """
    _require(images.shape[0] == BATCH_SIZE and labels.shape[0] == BATCH_SIZE, "large-arm logical update must retain all ten rows")
    references = consumer.references
    _require(references is not None, "large-arm microbatching requires the loaded frozen BOGT references")

    coefficients_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, BATCH_SIZE, LARGE_GRADIENT_MICROBATCH_SIZE):
            stop = min(BATCH_SIZE, start + LARGE_GRADIENT_MICROBATCH_SIZE)
            result = consumer.runtime.forward_from_image(images[start:stop])
            coefficients_parts.append(result.z_ols.coefficients.detach())
            del result
    full_coefficients = torch.cat(coefficients_parts, dim=0)
    l_dist, d_dist_d_ols = _dist_loss_and_ols_gradient(full_coefficients, references.scale)

    gradient_names = ("encoder.phase", "generator.phase")
    dist_gradients = {name: torch.zeros_like(parameters[name]) for name in gradient_names}
    other_bogt_gradients = {name: torch.zeros_like(parameters[name]) for name in gradient_names}
    evaluator_gradients = {
        name: {parameter_name: torch.zeros_like(parameters[parameter_name]) for parameter_name in gradient_names}
        for name in TRAINING_EVALUATORS
    }
    loss_totals = {"L_rec": 0.0, "L_op": 0.0, **{f"{name}_ce": 0.0 for name in TRAINING_EVALUATORS}}
    power_parts: list[torch.Tensor] = []
    extra_power_parts: list[torch.Tensor] = []

    for start in range(0, BATCH_SIZE, LARGE_GRADIENT_MICROBATCH_SIZE):
        stop = min(BATCH_SIZE, start + LARGE_GRADIENT_MICROBATCH_SIZE)
        weight = (stop - start) / BATCH_SIZE
        result = consumer.runtime.forward_from_image(images[start:stop])
        raw = result.raw_intensity
        target_raw = result.image_input.slm_intensity
        z = result.z_ols.coefficients
        d_dist_chunk = d_dist_d_ols[start:stop].to(dtype=z.dtype, device=z.device)
        l_dist_proxy = torch.real(torch.conj(d_dist_chunk) * z).sum()
        l_rec = (raw - target_raw).square().mean()
        planes = bogt.contracted_plane_powers(result)
        l_op = bogt.optical_power_loss_from_powers(planes, references.power_reference)
        view = canonical.cmos_view(raw)
        ce_losses = {
            f"{name}_ce": F.cross_entropy(
                _evaluator_logits(evaluators[name], view, name), labels[start:stop], reduction="mean"
            )
            for name in TRAINING_EVALUATORS
        }

        dist_chunk_gradients = _gradient(l_dist_proxy, parameters, retain_graph=True)
        other_bogt = semantic.BETA * l_rec + semantic.GAMMA * l_op
        other_bogt_chunk_gradients = _gradient(other_bogt, parameters, retain_graph=True)
        ce_chunk_gradients: dict[str, dict[str, torch.Tensor]] = {}
        for index, evaluator_name in enumerate(TRAINING_EVALUATORS):
            ce_chunk_gradients[evaluator_name] = _gradient(
                ce_losses[f"{evaluator_name}_ce"],
                parameters,
                retain_graph=index < len(TRAINING_EVALUATORS) - 1,
            )

        for name in gradient_names:
            dist_gradients[name] += dist_chunk_gradients[name]
            other_bogt_gradients[name] += weight * other_bogt_chunk_gradients[name]
            for evaluator_name in TRAINING_EVALUATORS:
                evaluator_gradients[evaluator_name][name] += weight * ce_chunk_gradients[evaluator_name][name]
        loss_totals["L_rec"] += weight * float(l_rec.detach().to(torch.float64).cpu())
        loss_totals["L_op"] += weight * float(l_op.detach().to(torch.float64).cpu())
        for name, loss in ce_losses.items():
            loss_totals[name] += weight * float(loss.detach().to(torch.float64).cpu())
        power_parts.append(planes.detach())
        generator = consumer.runtime.generator
        if isinstance(generator, ExpandedGenerator):
            _require(generator.last_extra_plane_powers is not None, "large Generator omitted new phase-plane powers")
            extra_power_parts.append(generator.last_extra_plane_powers.detach())
        del result, raw, target_raw, z, d_dist_chunk, l_dist_proxy, l_rec, planes, l_op, view, ce_losses
        del dist_chunk_gradients, other_bogt, other_bogt_chunk_gradients, ce_chunk_gradients

    semantic_grad, unavailable = diagnostic._safe_canonical_gradient(
        {key: evaluator_gradients[name] for name, key in zip(TRAINING_EVALUATORS, ("L_S", "L_I", "L_C"), strict=True)}
    )
    _require(semantic_grad is not None, f"canonical three-evaluator semantic direction is unavailable: {unavailable}")
    zero_dist_generator = _vector_norm(dist_gradients["generator.phase"])
    _require(zero_dist_generator == 0.0, "logical-batch L_dist produced a Generator gradient despite the structural zero")
    combined = {
        name: (
            semantic.ALPHA * dist_gradients[name]
            + other_bogt_gradients[name]
            + semantic.KAPPA * semantic_grad[name]
        )
        for name in gradient_names
    }
    _require(all(bool(torch.isfinite(value).all().item()) for value in combined.values()), "microbatched combined gradient is nonfinite")
    _require(all(parameter.grad is None for parameter in parameters.values()), "microbatched read-only autograd populated optical .grad")
    losses = {"L_dist": l_dist, **{name: torch.tensor(value, device=images.device) for name, value in loss_totals.items()}}
    gradients = {"L_dist": dist_gradients}
    plane_powers = torch.cat(power_parts, dim=0)
    if extra_power_parts:
        consumer.runtime.generator.last_extra_plane_powers = torch.cat(extra_power_parts, dim=0)
    return plane_powers, losses, gradients, combined


def _phase_parts(gradient: torch.Tensor, arm: str) -> tuple[torch.Tensor, torch.Tensor]:
    flat = gradient.reshape(-1)
    if arm == "B_large":
        return flat[:PHASE_OLD_COUNT], flat[PHASE_OLD_COUNT:]
    return flat, flat.new_empty((0,))


def _vector_norm(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float64).norm().cpu())


def _vector_rms(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float64).square().mean().sqrt().cpu())


def _train_step(consumer: Any, optimizer: torch.optim.Optimizer, parameters: Mapping[str, nn.Parameter], evaluators: Mapping[str, nn.Module], inputs: Mapping[str, Any], update_index: int, *, arm: str):
    images_cpu, labels_cpu, positions = training_batch(inputs["images"], inputs["labels"], update_index % BATCH_COUNT)
    batch = images_cpu.to(consumer.device)
    labels = labels_cpu.to(consumer.device)
    if arm == "B_large":
        plane_powers, losses, gradients, combined = _large_microbatch_loss_gradients(consumer, parameters, evaluators, batch, labels)
    else:
        plane_powers, losses, gradients, combined = _loss_gradients(consumer, parameters, evaluators, batch, labels)
    generator_gradient = combined["generator.phase"]
    _require(tuple(generator_gradient.shape) == tuple(parameters["generator.phase"].shape), "Generator gradient shape differs from the declared phase layout")
    dist_generator_norm = _vector_norm(gradients["L_dist"]["generator.phase"])
    _require(dist_generator_norm == 0.0, "L_dist produced a Generator gradient despite the structural zero")
    old_grad, new_grad = _phase_parts(generator_gradient, arm)
    _require(bool(torch.isfinite(generator_gradient).all().item()), "Generator phase gradient is nonfinite")
    if arm == "B_large":
        _require(_vector_norm(new_grad) > 0.0, "new 400x400 phase planes received no gradient")
    phase = parameters["generator.phase"]
    before = phase.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    phase.grad = generator_gradient.detach().clone()
    _require(parameters["encoder.phase"].grad is None, "Encoder gradient entered the optimizer")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    delta = phase.detach() - before
    old_delta, new_delta = _phase_parts(delta, arm)
    state = optimizer.state.get(phase, {})
    step = int(float(state["step"].item())) if "step" in state else 0
    _require(step == update_index + 1, "fresh Generator Adam step differs from task update count")
    extra_power = None
    generator = consumer.runtime.generator
    if isinstance(generator, ExpandedGenerator):
        _require(generator.last_extra_plane_powers is not None, "large Generator omitted new phase-plane powers")
        extra_power = generator.last_extra_plane_powers.detach().cpu().to(torch.float64).mean(dim=0).tolist()
    record = {
        "update": update_index + 1,
        "arm": arm,
        "batch_index": int(positions[0].item() // BATCH_SIZE),
        "fit_positions": positions.tolist(),
        "source_row_indices": inputs["source_indices"].index_select(0, positions).tolist(),
        "labels": labels_cpu.tolist(),
        "semantic_objective": "mean_E_S_E_I_E_C_true_class_cross_entropy",
        "ce_by_evaluator": {name: float(losses[f"{name}_ce"].detach().cpu()) for name in TRAINING_EVALUATORS},
        "raw_mse": float(losses["L_rec"].detach().cpu()),
        "l_dist": float(losses["L_dist"].detach().cpu()),
        "l_op": float(losses["L_op"].detach().cpu()),
        "bogt_weighted_components": {
            "alpha_l_dist": semantic.ALPHA * float(losses["L_dist"].detach().cpu()),
            "beta_l_rec": semantic.BETA * float(losses["L_rec"].detach().cpu()),
            "gamma_l_op": semantic.GAMMA * float(losses["L_op"].detach().cpu()),
            "kappa_joint_semantic_direction": semantic.KAPPA,
        },
        "five_plane_power_mean_by_batch": plane_powers.detach().to(torch.float64).mean(dim=0).cpu().tolist(),
        "new_phase_plane_power_mean_by_batch": extra_power,
        "encoder_read_only_gradient_norm": _vector_norm(combined["encoder.phase"]),
        "generator_gradient_norm": _vector_norm(generator_gradient),
        "generator_gradient_rms": _vector_rms(generator_gradient),
        "old_100_plane_gradient_norm": _vector_norm(old_grad),
        "new_400_plane_gradient_norm": _vector_norm(new_grad) if arm == "B_large" else None,
        "old_100_plane_displacement_rms": _vector_rms(old_delta),
        "new_400_plane_displacement_rms": _vector_rms(new_delta) if arm == "B_large" else None,
        "generator_displacement_rms": _vector_rms(delta),
        "generator_adam_step": step,
        "l_dist_generator_gradient_norm": dist_generator_norm,
        "all_parameters_finite": bool(torch.isfinite(phase).all().item()),
    }
    _require(record["all_parameters_finite"], "Generator phase became nonfinite after Adam")
    _require(all(parameter.grad is None for parameter in parameters.values()), "training step left optical parameter gradients populated")
    del plane_powers, losses, gradients, combined
    return record


def _target_metrics(evaluators: Mapping[str, nn.Module], inputs: Mapping[str, Any]) -> dict[str, Any]:
    view = canonical.cmos_view(inputs["images"])
    output: dict[str, Any] = {}
    with torch.no_grad():
        for name in EVALUATOR_NAMES:
            model = evaluators[name]
            pieces = []
            for start in range(0, TOTAL_ROWS, 100):
                logits = _evaluator_logits(model, view[start : start + 100].to(next(model.parameters()).device), name)
                pieces.append(logits.detach().cpu())
            all_logits = torch.cat(pieces, dim=0)
            metrics = semantic._class_metrics(inputs["labels"], all_logits)
            output[name] = {"logits": all_logits, "metrics": metrics}
    return output


def _target_control_passed(target: Mapping[str, Any]) -> bool:
    for name in EVALUATOR_NAMES:
        confirmation = target[name]["metrics"]["by_role"]["confirmation"]
        if float(confirmation["macro_recall"]) < 0.90:
            return False
        if any(int(confirmation["per_class_correct"][class_id]) < 24 for class_id in FOCUS_CLASSES):
            return False
    return True


def _snapshot_parameters(parameters: Mapping[str, nn.Parameter]) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}


def _frozen_runtime_state(consumer: Any) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in _state_snapshot(consumer.runtime).items()
        if name not in {"encoder.phase", "generator.phase"}
    }


def _assert_disposable_frozen(consumer: Any, parameters: Mapping[str, nn.Parameter], encoder_before: torch.Tensor, runtime_before: Mapping[str, torch.Tensor], evaluators: Mapping[str, nn.Module], evaluator_before: Mapping[str, Mapping[str, torch.Tensor]], label: str) -> None:
    _require(torch.equal(parameters["encoder.phase"].detach().cpu(), encoder_before), f"{label} changed Encoder phase")
    _assert_state_equal(runtime_before, _frozen_runtime_state(consumer), f"{label} OLS/geometry/scientific buffers")
    for name, model in evaluators.items():
        _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"{label} {name}")
        _require(all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()), f"{label} changed {name} evaluator isolation")
    _require(all(parameter.grad is None for parameter in parameters.values()), f"{label} left optical parameter gradients populated")


def _write_torch_new(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    diagnostic._write_torch_new(path, payload, kind)


def _write_json_new(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    diagnostic._write_json_new(path, payload, kind)


def _write_invalid(target: Path, phase: str, error: Exception, started: float, progress: Mapping[str, Any] | None = None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    path = target / "invalid_harness.json"
    if path.exists():
        return
    payload = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "INVALID_HARNESS",
        "classification": None,
        "training_recommendation": None,
        "phase": phase,
        "failure_type": type(error).__name__,
        "failure": str(error),
        "checks_executed": dict(progress or {}),
        "elapsed_seconds_observation_only": time.perf_counter() - started,
        "next_action": "NONE",
    }
    _write_json_new(path, payload, "invalid LightGen large-capacity comparison harness result")


def _write_probe_and_storage_checks(target: Path, source_checkpoint: Path) -> dict[str, Any]:
    root = _root()
    temp_root = Path(tempfile.gettempdir()).resolve()
    _require(root not in temp_root.parents and temp_root != root, "pytest/runtime temp root must remain outside the repository")
    with tempfile.TemporaryDirectory(prefix="lightgen_large_external_") as temp_name:
        temp_file = Path(temp_name) / "probe.bin"
        temp_file.write_bytes(b"lightgen-large-capacity-io-probe")
        _require(temp_file.read_bytes() == b"lightgen-large-capacity-io-probe", "external temporary directory write/read failed")
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target
    while not existing.exists():
        existing = existing.parent
    free = int(shutil.disk_usage(existing).free)
    # Four 600-row raw endpoints, role metrics, all eight periodic checkpoints, figures and run logs.
    endpoint_bytes = 4 * TOTAL_ROWS * 200 * 200 * 4
    checkpoint_bytes = 8 * (source_checkpoint.stat().st_size + PHASE_LARGE_COUNT * 12)
    estimate = endpoint_bytes + checkpoint_bytes + 600_000_000
    needed = estimate + DISK_SAFETY_BYTES
    _require(free >= needed, f"output volume has {free} bytes but needs {needed} including the 1.5 GiB reserve")
    with tempfile.TemporaryDirectory(prefix="lightgen_large_output_", dir=target.parent) as temp_name:
        probe = Path(temp_name) / "write_read.pt"
        _write_torch_new(probe, {"probe": torch.tensor([1.0, 2.0])}, "temporary output path write/read probe")
    return {
        "external_temporary_directory": temp_root.as_posix(),
        "external_temporary_write_read": True,
        "output_parent": existing.as_posix(),
        "free_bytes": free,
        "estimated_output_bytes": estimate,
        "safety_margin_bytes": DISK_SAFETY_BYTES,
        "required_bytes": needed,
        "temporary_output_write_read": True,
    }


def _gpu_snapshot(device: torch.device) -> dict[str, Any]:
    return diagnostic._resource_snapshot(device)


def _check_gpu(device: torch.device) -> dict[str, Any]:
    snapshot = diagnostic._resource_snapshot(device)
    if snapshot["global_used_fraction"] > 0.90:
        raise diagnostic.DiagnosticError(
            "PREFLIGHT_BLOCKED(resource): whole-device use exceeds the 90% safety line; "
            f"observed resource snapshot={snapshot}"
        )
    return snapshot


def _repeatability(left: Any, right: Any, path: str = "root") -> tuple[bool, dict[str, Any]]:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape and left.dtype == right.dtype):
            return False, {"path": path, "reason": "tensor shape or dtype differs"}
        if left.is_floating_point() or left.is_complex():
            if left.numel() == 0:
                maximum = 0.0
            else:
                maximum = float((left - right).abs().max().item())
            return torch.allclose(left, right, rtol=RTOL, atol=ATOL), {"path": path, "max_abs_difference": maximum, "rtol": RTOL, "atol": ATOL}
        exact = torch.equal(left, right)
        return exact, {"path": path, "exact": exact}
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right)):
            return False, {"path": path, "reason": "mapping keys differ"}
        details = []
        for key in left:
            passed, detail = _repeatability(left[key], right[key], f"{path}.{key}")
            details.append(detail)
            if not passed:
                return False, detail
        return True, {"path": path, "children": details}
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not (type(left) is type(right) and len(left) == len(right)):
            return False, {"path": path, "reason": "sequence shape differs"}
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            passed, detail = _repeatability(first, second, f"{path}[{index}]")
            if not passed:
                return False, detail
        return True, {"path": path, "count": len(left)}
    if isinstance(left, (float, int)) and isinstance(right, (float, int)) and not isinstance(left, bool) and not isinstance(right, bool):
        passed = math.isclose(float(left), float(right), rel_tol=RTOL, abs_tol=ATOL)
        return passed, {"path": path, "left": left, "right": right, "rtol": RTOL, "atol": ATOL}
    return left == right, {"path": path, "left": left, "right": right}


def _check_ce_derivative(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    logits64 = logits.detach().to(torch.float64).requires_grad_(True)
    loss = F.cross_entropy(logits64, labels.to(logits64.device), reduction="mean")
    actual = torch.autograd.grad(loss, logits64)[0]
    expected = logits64.detach().softmax(dim=1)
    expected.scatter_add_(1, labels.to(expected.device).view(-1, 1), torch.full((labels.numel(), 1), -1.0, dtype=expected.dtype, device=expected.device))
    expected /= labels.numel()
    error = float((actual - expected).abs().max().item())
    _require(error <= 2e-14, f"CE derivative differs from independent softmax reference: {error}")
    return {"passed": True, "max_abs_gradient_error": error, "reference": "(softmax(logits)-one_hot(labels))/batch"}


def _negative_control(inputs: Mapping[str, Any], source_state: Mapping[str, Any], evaluators: Mapping[str, nn.Module], device: torch.device) -> dict[str, Any]:
    consumer, optimizer, parameters = _new_endpoint(inputs, source_state, "B_large", device)
    batch_cpu, labels_cpu, _ = training_batch(inputs["images"], inputs["labels"], 0)
    batch = batch_cpu.to(device)
    labels = labels_cpu.to(device)
    wrong = labels.roll(1)
    differences: dict[str, float] = {}
    derivative: dict[str, Any] | None = None
    true_gradients = {name: torch.zeros_like(parameters["generator.phase"]) for name in TRAINING_EVALUATORS}
    wrong_gradients = {name: torch.zeros_like(parameters["generator.phase"]) for name in TRAINING_EVALUATORS}
    for start in range(0, BATCH_SIZE, LARGE_GRADIENT_MICROBATCH_SIZE):
        stop = min(BATCH_SIZE, start + LARGE_GRADIENT_MICROBATCH_SIZE)
        result = consumer.runtime.forward_from_image(batch[start:stop])
        view = canonical.cmos_view(result.raw_intensity)
        for index, name in enumerate(TRAINING_EVALUATORS):
            logits = _evaluator_logits(evaluators[name], view, name)
            if name == "E_S" and derivative is None:
                derivative = _check_ce_derivative(logits, labels[start:stop])
            true_loss = F.cross_entropy(logits, labels[start:stop])
            wrong_loss = F.cross_entropy(logits, wrong[start:stop])
            true_grad = torch.autograd.grad(true_loss, parameters["generator.phase"], retain_graph=True)[0]
            wrong_grad = torch.autograd.grad(
                wrong_loss,
                parameters["generator.phase"],
                retain_graph=index < len(TRAINING_EVALUATORS) - 1,
            )[0]
            row_weight = (stop - start) / BATCH_SIZE
            true_gradients[name] += row_weight * true_grad
            wrong_gradients[name] += row_weight * wrong_grad
        del result, view, logits, true_loss, wrong_loss, true_grad, wrong_grad
    for name in TRAINING_EVALUATORS:
        differences[name] = _vector_norm(true_gradients[name] - wrong_gradients[name])
        _require(differences[name] > 0.0, f"wrong-label negative control did not change {name} Generator CE gradient")
    _require(derivative is not None, "independent CE derivative reference was not executed")
    _require(all(parameter.grad is None for parameter in parameters.values()), "negative control wrote into parameter .grad")
    del consumer, optimizer, parameters, true_gradients, wrong_gradients
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "passed": True,
        "microbatch_rows": LARGE_GRADIENT_MICROBATCH_SIZE,
        "wrong_label_generator_gradient_difference_norm": differences,
        "independent_ce_derivative": derivative,
    }


def _three_step_trial(arm: str, inputs: Mapping[str, Any], source_state: Mapping[str, Any], evaluators: Mapping[str, nn.Module], initial_rng: Mapping[str, Any], *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _restore_rng_state(initial_rng)
    device = torch.device("cuda:0")
    consumer, optimizer, parameters = _new_endpoint(inputs, source_state, arm, device)
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    runtime_before = _frozen_runtime_state(consumer)
    evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
    records = []
    resource_samples = []
    for update in range(3):
        gc.collect()
        torch.cuda.empty_cache()
        resource_before = _check_gpu(device)
        resource_samples.append(resource_before)
        record = _train_step(consumer, optimizer, parameters, evaluators, inputs, update, arm=arm)
        records.append(record)
        _assert_disposable_frozen(consumer, parameters, encoder_before, runtime_before, evaluators, evaluator_before, label)
        # A 2048-grid Generator step can leave several GiB in PyTorch's idle
        # caching allocator. Release those blocks before the next safety check.
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "arm": arm,
        "label": label,
        "updates": len(records),
        "records": records,
        "model": _snapshot_parameters(parameters),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "resource_samples": resource_samples,
        "new_phase_gradient_nonzero": arm != "B_large" or all(row["new_400_plane_gradient_norm"] > 0.0 for row in records),
    }
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result, evaluator_before


def _recover_probe(inputs: Mapping[str, Any], source_state: Mapping[str, Any], evaluators: Mapping[str, nn.Module], initial_rng: Mapping[str, Any], population: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    device = torch.device("cuda:0")
    _restore_rng_state(initial_rng)
    consumer, optimizer, parameters = _new_endpoint(inputs, source_state, "A_small", device)
    first = _train_step(consumer, optimizer, parameters, evaluators, inputs, 0, arm="A_small")
    payload = _checkpoint_payload(consumer, optimizer, "A_small", 1, [first], population)
    path = workspace / "resume_probe.pt"
    _write_torch_new(path, payload, "disposable A-arm checkpoint recovery preflight")
    restored = torch.load(path, map_location="cpu", weights_only=True)
    continuous_records = [
        _train_step(consumer, optimizer, parameters, evaluators, inputs, index, arm="A_small")
        for index in (1, 2)
    ]
    continuous = {"parameters": _snapshot_parameters(parameters), "optimizer": _cpu_tree(optimizer.state_dict()), "records": continuous_records}
    del consumer, optimizer, parameters
    consumer, optimizer, parameters = _new_endpoint(inputs, source_state, "A_small", device)
    _restore_checkpoint(restored, consumer, optimizer, expected_arm="A_small")
    resumed_records = [
        _train_step(consumer, optimizer, parameters, evaluators, inputs, index, arm="A_small")
        for index in (1, 2)
    ]
    resumed = {"parameters": _snapshot_parameters(parameters), "optimizer": _cpu_tree(optimizer.state_dict()), "records": resumed_records}
    passed, detail = _repeatability(continuous, resumed)
    _require(passed, f"continuous versus resumed disposable A path differs: {detail}")
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return {"passed": True, "continuous_vs_resumed": detail, "saved_batch_cursor": restored["next_batch_index"], "resume_batches": [1, 2], "fresh_adam_step_after_resume": 3}


def _preflight(inputs: Mapping[str, Any], source_state: Mapping[str, Any], source_path: Path, population: Mapping[str, Any], target: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available() and torch.cuda.device_count() > 0, "CUDA device 0 is unavailable")
    device = torch.device("cuda:0")
    disk = _write_probe_and_storage_checks(target, source_path)
    cuda_before = _check_gpu(device)
    started = time.perf_counter()
    progress: dict[str, Any] = {}
    initial_rng = _rng_state()
    evaluators = None
    evaluator_before: dict[str, Any] = {}
    try:
        progress["logical_batch_microgradient_equivalence"] = {
            "executed": True,
            **_verify_large_microbatch_gradient_identity(),
        }
        evaluators = diagnostic.load_evaluators({name: Path(path) for name, path in inputs["input_paths"].items()}, device)
        evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
        target_control = _target_metrics(evaluators, inputs)
        target_passed = _target_control_passed(target_control)
        progress["four_evaluator_target_positive_control"] = {
            "executed": True,
            "passed": target_passed,
            "metrics": {name: target_control[name]["metrics"]["by_role"]["confirmation"] for name in EVALUATOR_NAMES},
        }
        _require(target_passed, "one or more frozen evaluator target-image controls fell below the declared threshold")

        _restore_rng_state(initial_rng)
        small, _small_optimizer, small_params = _new_endpoint(inputs, source_state, "A_small", device)
        _restore_rng_state(initial_rng)
        large, _large_optimizer, large_params = _new_endpoint(inputs, source_state, "B_large", device)
        for model in (small, large):
            model.runtime.eval()
        batch = inputs["images"][:BATCH_SIZE].to(device)
        with torch.no_grad():
            small_result = small.runtime.forward_from_image(batch)
            large_result = large.runtime.forward_from_image(batch)
            _require(torch.allclose(small_result.generator_output_field.tensor, large_result.generator_output_field.tensor, rtol=RTOL, atol=ATOL), "zero-new-phase large Generator exit field differs from original small Generator")
            _require(torch.allclose(small_result.raw_intensity, large_result.raw_intensity, rtol=RTOL, atol=ATOL), "zero-new-phase large Generator raw CMOS differs from original")
            _require(torch.allclose(small_result.z_ols.coefficients, large_result.z_ols.coefficients, rtol=RTOL, atol=ATOL), "expanded Generator changed Encoder/OLS coefficients")
        zero_insert = {
            "passed": True,
            "rows": BATCH_SIZE,
            "zero_added_phase_count": PHASE_NEW_COUNT,
            "generator_exit_max_abs_difference": float((small_result.generator_output_field.tensor - large_result.generator_output_field.tensor).abs().max().item()),
            "raw_cmos_max_abs_difference": float((small_result.raw_intensity - large_result.raw_intensity).abs().max().item()),
            "rtol": RTOL,
            "atol": ATOL,
        }
        progress["zero_phase_expansion_equivalence"] = {"executed": True, **zero_insert}
        del small_result, large_result, small, large, _small_optimizer, _large_optimizer, small_params, large_params
        gc.collect()
        torch.cuda.empty_cache()

        # The model derivative path is exercised by A/A and both A/B disposable updates below.
        primary_a, evaluator_before = _three_step_trial("A_small", inputs, source_state, evaluators, initial_rng, label="A_primary")
        replay_a, _ = _three_step_trial("A_small", inputs, source_state, evaluators, initial_rng, label="A_replay")
        passed, repeat_detail = _repeatability(
            {"model": primary_a["model"], "optimizer": primary_a["optimizer"], "records": primary_a["records"]},
            {"model": replay_a["model"], "optimizer": replay_a["optimizer"], "records": replay_a["records"]},
        )
        _require(passed, f"A/A disposable update repeatability failed: {repeat_detail}")
        progress["A_A_disposable_repeatability"] = {"executed": True, "passed": True, "detail": repeat_detail}
        large_b, _ = _three_step_trial("B_large", inputs, source_state, evaluators, initial_rng, label="B_large")
        _require(large_b["new_phase_gradient_nonzero"], "large model new phase planes did not receive gradients")
        progress["three_real_updates_and_frozen_scope"] = {
            "executed": True,
            "passed": True,
            "A_updates": primary_a["updates"],
            "A_replay_updates": replay_a["updates"],
            "B_updates": large_b["updates"],
            "B_new_phase_gradient_nonzero": large_b["new_phase_gradient_nonzero"],
            "A_trainable_count": 20_000,
            "B_trainable_count": PHASE_LARGE_COUNT,
            "A_A_repeatability": repeat_detail,
            "resource_samples_by_trial": {
                "A_primary": primary_a["resource_samples"],
                "A_replay": replay_a["resource_samples"],
                "B_large": large_b["resource_samples"],
            },
            "disposable_trials": {
                "A_primary": primary_a["records"],
                "A_replay": replay_a["records"],
                "B_large": large_b["records"],
            },
        }
        del primary_a, replay_a, large_b
        gc.collect()
        torch.cuda.empty_cache()

        negative = _negative_control(inputs, source_state, evaluators, device)
        progress["wrong_label_ce_negative_control"] = {"executed": True, **negative}
        with tempfile.TemporaryDirectory(prefix="lightgen_large_resume_", dir=target.parent) as temp_name:
            recovery = _recover_probe(inputs, source_state, evaluators, initial_rng, population, Path(temp_name))
        progress["interrupt_resume"] = {"executed": True, **recovery}
        for name, model in evaluators.items():
            _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"preflight final {name}")
            _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"preflight {name} evaluator lost frozen status")
        progress["evaluator_freeze"] = {"executed": True, "passed": True, "evaluators": list(EVALUATOR_NAMES)}
        cuda_after = _check_gpu(device)
        _restore_rng_state(initial_rng)
        return {
            "schema": PREFLIGHT_SCHEMA,
            "task_id": TASK_ID,
            "status": "READY",
            "device": str(device),
            "numeric_mode": {
                "dtype": "float32",
                "autocast": False,
                "cuda_matmul_tf32": False,
                "cudnn_tf32": False,
                "float32_matmul_precision": "highest",
            },
            "propagation_grid": 2048,
            "updates_per_arm": UPDATES_PER_ARM,
            "source_checkpoint_step": SOURCE_STEP,
            "source_optimizer_steps_discarded_after_validation": {"encoder.phase": 2200, "generator.phase": 2500},
            "fresh_generator_adam": {"lr": GENERATOR_LEARNING_RATE, "betas": list(GENERATOR_BETAS), "eps": GENERATOR_EPS, "weight_decay": GENERATOR_WEIGHT_DECAY, "initial_step": 0},
            "parameter_counts": {"A_small": 20_000, "B_large": PHASE_LARGE_COUNT, "ratio": PHASE_LARGE_COUNT / 20_000},
            "gradient_execution": {
                "logical_batch_size": BATCH_SIZE,
                "large_arm_microbatch_size": LARGE_GRADIENT_MICROBATCH_SIZE,
                "L_dist_batch_statistics_preserved": True,
                "microbatch_gradient_check": progress["logical_batch_microgradient_equivalence"],
            },
            "disk_observation": disk,
            "cuda_before": cuda_before,
            "cuda_after": cuda_after,
            "target_image_positive_control": progress["four_evaluator_target_positive_control"],
            "checks": progress,
            "duration_seconds_observation_only": time.perf_counter() - started,
        }
    except Exception as error:
        if not (target / "preflight_diagnostics.json").exists():
            _write_json_new(target / "preflight_diagnostics.json", {
                "schema": PREFLIGHT_SCHEMA,
                "task_id": TASK_ID,
                "status": "INVALID_HARNESS",
                "checks": progress,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "duration_seconds_observation_only": time.perf_counter() - started,
            }, "executed preflight checks and first invalid-harness exception")
        raise
    finally:
        _restore_rng_state(initial_rng)
        evaluators = None
        gc.collect()
        torch.cuda.empty_cache()


def _plane_metadata(field: ComplexField) -> dict[str, Any]:
    return {"plane": field.plane.to_dict(), "provenance": field.provenance.to_dict()}


def _all_fields_metadata(result: LightGen2DForwardResult, arm: str, consumer: Any) -> dict[str, Any]:
    metadata = {
        "input_slm_field": _plane_metadata(result.image_input.slm_field),
        "encoder_records": [_plane_metadata(field) for field in result.encoder_records],
        "generator_input_field": _plane_metadata(result.generator_input_field),
        "generator_records": [_plane_metadata(field) for field in result.generator_records],
        "camera_plane": result.raw_camera.plane.to_dict(),
        "camera_provenance": result.raw_camera.provenance.to_dict(),
    }
    if arm == "B_large":
        generator = consumer.runtime.generator
        _require(isinstance(generator, ExpandedGenerator), "large endpoint lost its expanded Generator")
        entrance_z = float(result.generator_input_field.plane.z_m)
        metadata["new_phase_planes"] = [
            {
                "plane_id": "lightgen_generator_new_phase_075mm",
                "shape": [2048, 2048],
                "pixel_pitch_m": [2.0e-6, 2.0e-6],
                "z_m": entrance_z + 0.75e-3,
                "phase_support_shape": [400, 400],
                "phase_support_fov_m": [800.0e-6, 800.0e-6],
                "operation": "phase_only_modulation_centered_400_window",
                "outside_window_transmission": 1.0,
                "intermediate_crop": False,
            },
            {
                "plane_id": "lightgen_generator_new_phase_225mm",
                "shape": [2048, 2048],
                "pixel_pitch_m": [2.0e-6, 2.0e-6],
                "z_m": entrance_z + 2.25e-3,
                "phase_support_shape": [400, 400],
                "phase_support_fov_m": [800.0e-6, 800.0e-6],
                "operation": "phase_only_modulation_centered_400_window",
                "outside_window_transmission": 1.0,
                "intermediate_crop": False,
            },
        ]
    else:
        metadata["new_phase_planes"] = []
    return metadata


def _forward_all(consumer: Any, evaluators: Mapping[str, nn.Module], inputs: Mapping[str, Any], arm: str) -> dict[str, torch.Tensor]:
    rows = inputs["images"].shape[0]
    device = consumer.device
    z = torch.empty((rows, 100), dtype=torch.complex64)
    raw = torch.empty((rows, 1, 200, 200), dtype=torch.float32)
    view = torch.empty((rows, 1, 28, 28), dtype=torch.float32)
    target_raw = torch.empty_like(raw)
    power = torch.empty((rows, 5), dtype=torch.float64)
    extra_power = torch.empty((rows, 2), dtype=torch.float64) if arm == "B_large" else None
    logits = {name: torch.empty((rows, 10), dtype=torch.float32) for name in EVALUATOR_NAMES}
    fields_metadata: dict[str, Any] | None = None
    for start in range(0, rows, BATCH_SIZE):
        stop = min(rows, start + BATCH_SIZE)
        batch = inputs["images"][start:stop].to(device)
        with torch.no_grad():
            result = consumer.runtime.forward_from_image(batch)
            current_view = canonical.cmos_view(result.raw_intensity)
            z[start:stop].copy_(result.z_ols.coefficients.detach().cpu())
            raw[start:stop].copy_(result.raw_intensity.detach().cpu())
            view[start:stop].copy_(current_view.detach().cpu())
            power[start:stop].copy_(bogt.contracted_plane_powers(result).detach().cpu())
            _require(result.image_input is not None, "full image path omitted its SLM target")
            target_raw[start:stop].copy_(result.image_input.slm_intensity.detach().cpu())
            if extra_power is not None:
                generator = consumer.runtime.generator
                _require(isinstance(generator, ExpandedGenerator) and generator.last_extra_plane_powers is not None, "large endpoint omitted new-plane power")
                extra_power[start:stop].copy_(generator.last_extra_plane_powers.detach().to(torch.float64).cpu())
            for name in EVALUATOR_NAMES:
                logits[name][start:stop].copy_(_evaluator_logits(evaluators[name], current_view, name).detach().cpu())
            if fields_metadata is None:
                fields_metadata = _all_fields_metadata(result, arm, consumer)
        del result, current_view
        if (start // BATCH_SIZE) % 10 == 0:
            _check_gpu(device)
    payload: dict[str, torch.Tensor] = {
        "z": z,
        "raw_cmos": raw,
        "evaluator_view": view,
        "target_slm_intensity": target_raw,
        "power_by_plane": power,
        "source_indices": inputs["source_indices"].clone(),
        "labels": inputs["labels"].clone(),
        **{f"{name.lower()}_logits": value for name, value in logits.items()},
    }
    if extra_power is not None:
        payload["new_phase_plane_power"] = extra_power
    for name, value in payload.items():
        if torch.is_tensor(value):
            diagnostic._finite_tensor(value.real if torch.is_complex(value) else value, f"{arm} {name}")
    _require(fields_metadata is not None, "endpoint did not record physical plane metadata")
    payload["plane_metadata"] = fields_metadata
    for name, model in evaluators.items():
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"{name} acquired gradients or lost frozen status during measurement")
    return payload


def _row_scores(readouts: Mapping[str, Any], labels: torch.Tensor) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    labels64 = labels.to(torch.long).cpu()
    row_values: dict[str, torch.Tensor] = {}
    summaries: dict[str, Any] = {}
    for name in EVALUATOR_NAMES:
        logits = readouts[f"{name.lower()}_logits"].to(torch.float64).cpu()
        true = logits.gather(1, labels64.view(-1, 1)).squeeze(1)
        wrong = logits.scatter(1, labels64.view(-1, 1), -torch.inf).max(dim=1).values
        margin = true - wrong
        ce = F.cross_entropy(logits, labels64, reduction="none")
        predictions = logits.argmax(dim=1)
        metric = semantic._class_metrics(labels64, logits.to(torch.float32))
        row_values[f"{name.lower()}_ce_float64"] = ce
        row_values[f"{name.lower()}_margin_float64"] = margin
        row_values[f"{name.lower()}_predictions"] = predictions
        summaries[name] = {
            "by_role": metric["by_role"],
            "prediction_counts": torch.bincount(predictions, minlength=10).tolist(),
            "mean_true_class_ce_float64": float(ce.mean().item()),
            "mean_true_class_margin_float64": float(margin.mean().item()),
        }
    return summaries, row_values


def _measure_statistics(readouts: Mapping[str, Any], inputs: Mapping[str, Any], endpoint: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    labels = inputs["labels"].cpu()
    metrics, score_rows = _row_scores(readouts, labels)
    raw = readouts["raw_cmos"].to(torch.float64)
    target = readouts["target_slm_intensity"].to(torch.float64)
    image_rows = diagnostic._row_reconstruction_metrics(raw.to(torch.float32), target.to(torch.float32), inputs["images"])
    raw_mse = (raw - target).square().flatten(1).mean(dim=1)
    raw_above_one = (raw > 1.0).to(torch.float64).flatten(1).mean(dim=1)
    raw_zero = raw.flatten(1).abs().amax(dim=1) == 0.0
    raw_constant = raw.flatten(1).amax(dim=1) == raw.flatten(1).amin(dim=1)
    view = readouts["evaluator_view"].to(torch.float64)
    target_view = canonical.cmos_view(inputs["images"]).to(torch.float64)
    view_mse = (view - target_view).square().flatten(1).mean(dim=1)
    row_values = {
        "source_indices": inputs["source_indices"].cpu(),
        "labels": labels,
        "role": torch.tensor([0] * FIT_ROWS + [1] * HISTORIC_ROWS + [2] * CONFIRMATION_ROWS, dtype=torch.long),
        "raw_mse_float64": raw_mse,
        "raw_foreground_mse": image_rows["raw_foreground_mse"].to(torch.float64),
        "raw_background_mse": image_rows["raw_background_mse"].to(torch.float64),
        "view_mse_float64": view_mse,
        "view_foreground_mse": image_rows["evaluator_view_foreground_mse"].to(torch.float64),
        "view_background_mse": image_rows["evaluator_view_background_mse"].to(torch.float64),
        "raw_fraction_above_one": raw_above_one,
        "raw_zero_image": raw_zero,
        "raw_constant_image": raw_constant,
        **score_rows,
    }
    role_slices = {
        "fit": slice(0, FIT_ROWS),
        "historic_holdout": slice(FIT_ROWS, FIT_ROWS + HISTORIC_ROWS),
        "confirmation": slice(FIT_ROWS + HISTORIC_ROWS, TOTAL_ROWS),
        "all": slice(0, TOTAL_ROWS),
    }
    image_summaries: dict[str, Any] = {}
    for role, selected in role_slices.items():
        positions = torch.arange(TOTAL_ROWS)[selected]
        role_data: dict[str, Any] = {}
        for class_id in range(10):
            class_positions = positions[labels.index_select(0, positions) == class_id]
            if role != "all" and class_positions.numel() == 0:
                continue
            def mean(key: str) -> float:
                return float(row_values[key].index_select(0, class_positions).to(torch.float64).mean().item()) if class_positions.numel() else 0.0
            role_data[str(class_id)] = {
                "rows": int(class_positions.numel()),
                "source_row_indices": inputs["source_indices"].index_select(0, class_positions).tolist(),
                "raw_mse": mean("raw_mse_float64"),
                "raw_foreground_mse": mean("raw_foreground_mse"),
                "raw_background_mse": mean("raw_background_mse"),
                "view_mse": mean("view_mse_float64"),
                "view_foreground_mse": mean("view_foreground_mse"),
                "view_background_mse": mean("view_background_mse"),
                "raw_fraction_above_one": mean("raw_fraction_above_one"),
                "raw_zero_image_count": int(row_values["raw_zero_image"].index_select(0, class_positions).sum().item()),
                "raw_constant_image_count": int(row_values["raw_constant_image"].index_select(0, class_positions).sum().item()),
            }
        role_data["overall"] = {
            "rows": int(positions.numel()),
            "raw_mse": float(raw_mse.index_select(0, positions).mean().item()),
            "raw_fraction_above_one": float(raw_above_one.index_select(0, positions).mean().item()),
            "raw_zero_image_count": int(raw_zero.index_select(0, positions).sum().item()),
            "raw_constant_image_count": int(raw_constant.index_select(0, positions).sum().item()),
        }
        image_summaries[role] = role_data
    summary = {
        "endpoint": endpoint,
        "evaluators": metrics,
        "image_metrics_by_role": image_summaries,
        "raw_software_clipping_definition": "raw CMOS fraction of pixels > 1.0",
        "foreground_definition": "target intensity > 0.1; raw 200x200 and evaluator 28x28 views reported separately",
        "all_finite": all(bool(torch.isfinite(value).all().item()) for value in row_values.values() if torch.is_tensor(value) and value.is_floating_point()),
        "has_zero_image": bool(raw_zero.any().item()),
        "has_constant_image": bool(raw_constant.any().item()),
        "maximum_raw_intensity": float(raw.max().item()),
    }
    _require(summary["all_finite"], f"{endpoint} row metrics contain nonfinite values")
    return summary, row_values


def _validate_readout(readouts: Mapping[str, Any], inputs: Mapping[str, Any], arm: str) -> None:
    _require(torch.equal(readouts["source_indices"].cpu(), inputs["source_indices"].cpu()), f"{arm} readout row indices changed")
    _require(torch.equal(readouts["labels"].cpu(), inputs["labels"].cpu()), f"{arm} readout labels changed")
    _require(tuple(readouts["z"].shape) == (TOTAL_ROWS, 100) and readouts["z"].dtype == torch.complex64, f"{arm} OLS output shape/dtype changed")
    _require(tuple(readouts["raw_cmos"].shape) == (TOTAL_ROWS, 1, 200, 200), f"{arm} raw CMOS shape changed")
    _require(tuple(readouts["evaluator_view"].shape) == (TOTAL_ROWS, 1, 28, 28), f"{arm} evaluator view shape changed")
    for name in EVALUATOR_NAMES:
        _require(tuple(readouts[f"{name.lower()}_logits"].shape) == (TOTAL_ROWS, 10), f"{arm} {name} logits shape changed")
    if arm in {"B_large", "B_start", "B_step0300"}:
        _require(tuple(readouts["new_phase_plane_power"].shape) == (TOTAL_ROWS, 2), f"{arm} new phase-plane power shape changed")


def _save_measurement(target: Path, endpoint: str, readouts: Mapping[str, Any], inputs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    readout_path = target / f"readouts_{endpoint}.pt"
    if readout_path.exists():
        saved = torch.load(readout_path, map_location="cpu", weights_only=True)
        _require(_tree_equal(saved, readouts), f"saved {endpoint} readouts differ from regenerated output")
        readouts = saved
    else:
        _write_torch_new(readout_path, readouts, f"600-row {endpoint} full optical readouts")
    summary, rows = _measure_statistics(readouts, inputs, endpoint)
    row_path = target / f"row_metrics_{endpoint}.pt"
    if row_path.exists():
        saved_rows = torch.load(row_path, map_location="cpu", weights_only=True)
        _require(_tree_equal(saved_rows, rows), f"saved {endpoint} per-row statistics differ")
        rows = saved_rows
    else:
        _write_torch_new(row_path, rows, f"600-row {endpoint} float64 scores and image error statistics")
    summary_path = target / f"metrics_{endpoint}.json"
    if summary_path.exists():
        _require(json.loads(summary_path.read_text(encoding="utf-8")) == summary, f"saved {endpoint} summary differs")
    else:
        _write_json_new(summary_path, summary, f"role-separated metrics for {endpoint}")
    return summary, rows


def _endpoint_accuracy(summary: Mapping[str, Any], role: str = "confirmation") -> dict[str, Any]:
    return summary["evaluators"]["E_D"]["by_role"][role]


def _correct_by_class(summary: Mapping[str, Any], role: str = "confirmation") -> list[int]:
    return [int(value) for value in _endpoint_accuracy(summary, role)["per_class_correct"]]


def _paired_correctness(base: Mapping[str, Any], changed: Mapping[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    base_predictions = base["e_d_logits"].to(torch.float64).argmax(dim=1)
    changed_predictions = changed["e_d_logits"].to(torch.float64).argmax(dim=1)
    labels_cpu = labels.cpu()
    output: dict[str, Any] = {}
    start, stop = FIT_ROWS + HISTORIC_ROWS, TOTAL_ROWS
    for class_id in range(10):
        pos = torch.arange(start, stop)[labels_cpu[start:stop] == class_id]
        first_correct = base_predictions.index_select(0, pos) == labels_cpu.index_select(0, pos)
        second_correct = changed_predictions.index_select(0, pos) == labels_cpu.index_select(0, pos)
        output[str(class_id)] = {
            "rows": int(pos.numel()),
            "correct_to_correct": int((first_correct & second_correct).sum().item()),
            "wrong_to_correct": int((~first_correct & second_correct).sum().item()),
            "correct_to_wrong": int((first_correct & ~second_correct).sum().item()),
            "wrong_to_wrong": int((~first_correct & ~second_correct).sum().item()),
            "predicted_class_transitions": {
                f"{int(a)}->{int(b)}": int(((base_predictions.index_select(0, pos) == a) & (changed_predictions.index_select(0, pos) == b)).sum().item())
                for a in torch.unique(base_predictions.index_select(0, pos))
                for b in torch.unique(changed_predictions.index_select(0, pos))
                if bool(((base_predictions.index_select(0, pos) == a) & (changed_predictions.index_select(0, pos) == b)).any().item())
            },
        }
    return output


def _final_decision(labels: torch.Tensor, summaries: Mapping[str, Mapping[str, Any]], readouts: Mapping[str, Mapping[str, Any]], frozen_valid: bool) -> dict[str, Any]:
    a_metrics = _endpoint_accuracy(summaries["A_step0300"])
    start_metrics = _endpoint_accuracy(summaries["B_start"])
    b_metrics = _endpoint_accuracy(summaries["B_step0300"])
    a_correct = _correct_by_class(summaries["A_step0300"])
    start_correct = _correct_by_class(summaries["B_start"])
    b_correct = _correct_by_class(summaries["B_step0300"])
    focus_deltas = {
        str(class_id): {
            "vs_A_endpoint": b_correct[class_id] - a_correct[class_id],
            "vs_B_start": b_correct[class_id] - start_correct[class_id],
        }
        for class_id in FOCUS_CLASSES
    }
    focus_supported = all(
        b_correct[class_id] >= a_correct[class_id] + 3
        and b_correct[class_id] >= start_correct[class_id] + 3
        for class_id in FOCUS_CLASSES
    )
    a_macro = float(a_metrics["macro_recall"])
    start_macro = float(start_metrics["macro_recall"])
    b_macro = float(b_metrics["macro_recall"])
    all_class_guard = all(
        b_correct[class_id] >= a_correct[class_id] - 3
        and b_correct[class_id] >= start_correct[class_id] - 3
        for class_id in range(10)
    )
    a_raw = float(summaries["A_step0300"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"])
    start_raw = float(summaries["B_start"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"])
    b_raw = float(summaries["B_step0300"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"])
    clipping = summaries["B_step0300"]["image_metrics_by_role"]["confirmation"]
    clip_guard = clipping["overall"]["raw_fraction_above_one"] <= 0.01 and all(
        clipping[str(class_id)]["raw_fraction_above_one"] <= 0.01 for class_id in FOCUS_CLASSES
    )
    image_guard = (
        b_raw <= 1.05 * a_raw
        and b_raw <= 1.05 * start_raw
        and clip_guard
        and not summaries["B_step0300"]["has_zero_image"]
        and not summaries["B_step0300"]["has_constant_image"]
        and summaries["B_step0300"]["all_finite"]
    )
    conditions = {
        "focus_classes_each_gain_at_least_3_correct_over_A_and_B_start": focus_supported,
        "macro_recall_at_least_A_plus_0.05_and_not_below_B_start": b_macro >= a_macro + 0.05 and b_macro >= start_macro,
        "no_class_loses_more_than_3_correct_vs_either_comparator": all_class_guard,
        "raw_mse_and_clipping_image_guard": image_guard,
        "all_four_endpoints_finite_and_frozen_scope_valid": frozen_valid,
    }
    if all(conditions.values()):
        classification = "QUICK_CAPACITY_GAIN_SUPPORTED"
        status = "VALID_COMPLETED"
        recommendation = "PRIORITIZE_LARGE_GENERATOR_FOR_NEXT_CUMULATIVE_TRAINING_DESIGN"
    else:
        any_focus_gain = any(
            b_correct[class_id] > a_correct[class_id] and b_correct[class_id] > start_correct[class_id]
            for class_id in FOCUS_CLASSES
        )
        if any_focus_gain:
            classification = "MIXED_OR_INSUFFICIENT"
            status = "VALID_FAILED"
        else:
            classification = "NOT_SUPPORTED_WITHIN_300_UPDATES"
            status = "VALID_FAILED"
        recommendation = "NONE"
    return {
        "status": status,
        "classification": classification,
        "conditions": conditions,
        "focus_class_correct_deltas": focus_deltas,
        "macro_recall": {"A_step0300": a_macro, "B_start": start_macro, "B_step0300": b_macro},
        "confirmation_raw_mse": {"A_step0300": a_raw, "B_start": start_raw, "B_step0300": b_raw},
        "confirmation_B_step0300_clipping": {
            "overall": clipping["overall"]["raw_fraction_above_one"],
            "class_1": clipping["1"]["raw_fraction_above_one"],
            "class_7": clipping["7"]["raw_fraction_above_one"],
        },
        "class_correct": {"A_step0300": a_correct, "B_start": start_correct, "B_step0300": b_correct},
        "training_recommendation": recommendation,
        "next_action": "NONE",
        "stop": True,
    }


def _checkpoint_name(arm: str, completed: int) -> str:
    letter = "a" if arm == "A_small" else "b"
    return f"arm_{letter}_step{completed:04d}.pt"


def _checkpoint_payload(consumer: Any, optimizer: torch.optim.Optimizer, arm: str, completed: int, history: Sequence[Mapping[str, Any]], population: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _parameters(consumer)
    generator_step = int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) if completed else 0
    _require(generator_step == completed, "saved fresh Generator Adam step differs from completed updates")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "arm": arm,
        "architecture_layout": "two_original_100x100_phase_planes" if arm == "A_small" else "two_original_100x100_then_two_400x400_phase_planes_flattened",
        "source_checkpoint_step": SOURCE_STEP,
        "source_optimizer_steps": {"encoder.phase": 2200, "generator.phase": 2500},
        "completed_updates": completed,
        "next_batch_index": completed % BATCH_COUNT,
        "population": dict(population),
        "trainable_parameter_state": _snapshot_parameters(parameters),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "optimizer_parameter_names": ["generator.phase"],
        "optimizer_step": generator_step,
        "frozen_runtime_state": _frozen_runtime_state(consumer),
        "references": consumer.references.to_payload(),
        "weights": consumer.weights.to_payload(),
        "training_history": [dict(row) for row in history],
        "rng_state": _rng_state(),
        "numeric_mode": {
            "dtype": "float32",
            "autocast": False,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "float32_matmul_precision": "highest",
        },
        "next_batch_cursor": completed % BATCH_COUNT,
    }


def _save_checkpoint(target: Path, consumer: Any, optimizer: torch.optim.Optimizer, arm: str, completed: int, history: Sequence[Mapping[str, Any]], population: Mapping[str, Any]) -> Path:
    path = target / _checkpoint_name(arm, completed)
    diagnostic._check_output_budget(target)
    _write_torch_new(path, _checkpoint_payload(consumer, optimizer, arm, completed, history, population), "explicit 100-step Generator-only recovery checkpoint")
    return path


def _arm_order(arm: str, completed: int) -> int:
    return completed if arm == "A_small" else UPDATES_PER_ARM + 1 + completed


def _read_checkpoint(path: Path, target: Path, population: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.parent.resolve() == target.resolve() and path.is_file(), "resume requires an explicit checkpoint inside this task output")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping) and payload.get("schema") == CHECKPOINT_SCHEMA, "resume checkpoint schema mismatch")
    _require(payload.get("task_id") == TASK_ID and payload.get("source_checkpoint_step") == SOURCE_STEP, "resume task/source endpoint differs")
    _require(payload.get("population") == dict(population), "resume population rows, labels, roles, or inputs changed")
    arm, completed = payload.get("arm"), payload.get("completed_updates")
    _require(arm in ARM_NAMES and isinstance(completed, int) and 0 <= completed <= UPDATES_PER_ARM, "resume arm/update count is invalid")
    _require(payload.get("next_batch_index") == completed % BATCH_COUNT and payload.get("next_batch_cursor") == completed % BATCH_COUNT, "resume batch cursor differs")
    _require(path.name == _checkpoint_name(arm, completed), "resume filename does not match its actual state")
    _require(payload.get("optimizer_step") == completed and payload.get("optimizer_parameter_names") == ["generator.phase"], "resume does not contain the reset Generator-only Adam")
    _require(payload.get("numeric_mode", {}).get("float32_matmul_precision") == "highest", "resume numerical mode differs")
    later = []
    for candidate in target.glob("arm_*_step*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if match is None:
            continue
        candidate_arm = "A_small" if match.group(1) == "a" else "B_large"
        candidate_step = int(match.group(2))
        if _arm_order(candidate_arm, candidate_step) > _arm_order(arm, completed):
            later.append(candidate.name)
    _require(not later, f"explicit resume checkpoint is older than saved task state: {sorted(later)}")
    return dict(payload)


def _restore_checkpoint(payload: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer, *, expected_arm: str) -> None:
    _require(payload.get("arm") == expected_arm, "resume checkpoint selects the wrong arm")
    parameters = _parameters(consumer)
    state = payload.get("trainable_parameter_state")
    _require(isinstance(state, Mapping) and set(state) == set(parameters), "resume trainable parameter names differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            _require(tuple(value.shape) == tuple(parameter.shape) and value.dtype == parameter.dtype, f"resume shape/dtype changed for {name}")
            _require(bool(torch.isfinite(value).all().item()), f"resume parameter {name} is nonfinite")
            parameter.copy_(value.to(parameter.device))
    optimizer.load_state_dict(payload["optimizer_state"])
    completed = int(payload["completed_updates"])
    current_step = int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) if completed else 0
    _require(current_step == completed, "restored fresh Generator Adam step differs")
    _restore_rng_state(payload["rng_state"])
    _assert_state_equal(payload["frozen_runtime_state"], _frozen_runtime_state(consumer), "resumed frozen runtime")


def _completed_resume_result(target: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any] | None:
    report_path = target / "report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("task_id") == TASK_ID and report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"}, "saved terminal report is invalid")
    _require(checkpoint.get("arm") == "B_large" and int(checkpoint.get("completed_updates", -1)) == UPDATES_PER_ARM, "completed output requires explicit final B checkpoint")
    return report


def _resource_telemetry(target: Path, device: torch.device, phase: str, progress: Mapping[str, Any], started: float) -> dict[str, Any]:
    record = {
        "task_id": TASK_ID,
        "phase": phase,
        "progress": dict(progress),
        "elapsed_seconds_observation_only": time.perf_counter() - started,
        "resource": _gpu_snapshot(device),
    }
    with (target / RESOURCE_TELEMETRY_NAME).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
    _require(record["resource"]["global_used_fraction"] <= 0.90, "whole-device use exceeded the 90% safety line")
    return record


def _run_arm(arm: str, inputs: Mapping[str, Any], source_state: Mapping[str, Any], evaluators: Mapping[str, nn.Module], target: Path, population: Mapping[str, Any], initial_rng: Mapping[str, Any], *, start_from: Mapping[str, Any] | None, started: float) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda:0")
    if start_from is None:
        _restore_rng_state(initial_rng)
    consumer, optimizer, parameters = _new_endpoint(inputs, source_state, arm, device)
    if start_from is None:
        completed = 0
        history: list[dict[str, Any]] = []
        _save_checkpoint(target, consumer, optimizer, arm, completed, history, population)
    else:
        _restore_checkpoint(start_from, consumer, optimizer, expected_arm=arm)
        completed = int(start_from["completed_updates"])
        history = [dict(record) for record in start_from.get("training_history", [])]
        _require(len(history) == completed, "checkpoint training history does not match completed updates")
    evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    frozen_before = _frozen_runtime_state(consumer)
    signal_requested = False
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal signal_requested
        signal_requested = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while completed < UPDATES_PER_ARM:
            torch.cuda.empty_cache()
            _check_gpu(device)
            record = _train_step(consumer, optimizer, parameters, evaluators, inputs, completed, arm=arm)
            completed += 1
            history.append(record)
            gc.collect()
            torch.cuda.empty_cache()
            interrupted = signal_requested
            if completed % CHECKPOINT_INTERVAL == 0 or interrupted:
                checkpoint_path = _save_checkpoint(target, consumer, optimizer, arm, completed, history, population)
                _resource_telemetry(target, device, f"{arm}_checkpoint", {"updates": completed, "checkpoint": checkpoint_path.name}, started)
                _require(torch.equal(parameters["encoder.phase"].detach().cpu(), encoder_before), f"{arm} changed Encoder phase")
                _assert_state_equal(frozen_before, _frozen_runtime_state(consumer), f"{arm} OLS/geometry/scientific buffers")
                for name, model in evaluators.items():
                    _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"{arm} evaluator {name}")
                    _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"{name} evaluator isolation changed")
                gc.collect()
                torch.cuda.empty_cache()
            if interrupted:
                raise TrainingInterrupted(f"saved explicit checkpoint {checkpoint_path.name}")
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    _require(completed == UPDATES_PER_ARM and len(history) == UPDATES_PER_ARM, f"{arm} did not complete the fixed 300 updates")
    _require(torch.equal(parameters["encoder.phase"].detach().cpu(), encoder_before), f"{arm} changed Encoder parameters")
    _assert_state_equal(frozen_before, _frozen_runtime_state(consumer), f"{arm} frozen runtime at end")
    for name, model in evaluators.items():
        _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"{arm} evaluator {name} at end")
    endpoint = "A_step0300" if arm == "A_small" else "B_step0300"
    readout_path = target / f"readouts_{endpoint}.pt"
    if readout_path.is_file():
        readouts = torch.load(readout_path, map_location="cpu", weights_only=True)
        _validate_readout(readouts, inputs, endpoint)
        summary, row_metrics = _save_measurement(target, endpoint, readouts, inputs)
    else:
        readouts = _forward_all(consumer, evaluators, inputs, arm)
        _validate_readout(readouts, inputs, endpoint)
        summary, row_metrics = _save_measurement(target, endpoint, readouts, inputs)
    history_path = target / f"training_history_{arm}.json"
    history_payload = {"task_id": TASK_ID, "arm": arm, "updates": len(history), "records": history}
    if history_path.exists():
        _require(json.loads(history_path.read_text(encoding="utf-8")) == history_payload, f"saved {arm} training history changed")
    else:
        _write_json_new(history_path, history_payload, f"{arm} complete 300-step training record")
    checkpoint_path = target / _checkpoint_name(arm, UPDATES_PER_ARM)
    if not checkpoint_path.exists():
        _save_checkpoint(target, consumer, optimizer, arm, UPDATES_PER_ARM, history, population)
    result = {
        "arm": arm,
        "completed_updates": completed,
        "readouts": readouts,
        "summary": summary,
        "row_metrics": row_metrics,
        "frozen_scope_valid": True,
    }
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result, {"history": history, "frozen_scope_valid": True}


def _class_render(view: torch.Tensor, scale: int = 4) -> Image.Image:
    image = view.detach().to(torch.float32).cpu().squeeze().clamp(0.0, 1.0)
    array = torch.round((1.0 - image) * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(array, mode="L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)


def _make_comparison_pages(target: Path, inputs: Mapping[str, Any], readouts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    font_scale = 4
    cell = 28 * font_scale
    row_height = cell + 37
    headers = ("Target", "A start", "B start", "A 300", "B 300")
    page_rows = 30
    labels = inputs["labels"].cpu()
    selected_sets = {
        "class_1_7_and_controls_0_4_8": torch.nonzero(torch.isin(labels, torch.tensor(DISPLAY_CLASSES)), as_tuple=False).flatten().tolist(),
        "class_9": torch.nonzero(labels == 9, as_tuple=False).flatten().tolist(),
    }
    page_paths: dict[str, list[str]] = {}
    for set_name, positions in selected_sets.items():
        page_paths[set_name] = []
        for page_start in range(0, len(positions), page_rows):
            page_positions = positions[page_start : page_start + page_rows]
            width = 1010
            height = 58 + len(page_positions) * row_height
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"{set_name} rows {page_start + 1}-{page_start + len(page_positions)} of {len(positions)}; fixed [0,1] grayscale", fill="black")
            column_x = (430, 535, 640, 745, 850)
            for x, header in zip(column_x, headers, strict=True):
                draw.text((x, 35), header, fill="black")
            for local_row, position in enumerate(page_positions):
                y = 55 + local_row * row_height
                source = int(inputs["source_indices"][position])
                label = int(labels[position])
                role = "fit" if position < FIT_ROWS else ("history" if position < FIT_ROWS + HISTORIC_ROWS else "confirm")
                draw.text((8, y + 8), f"row={source} y={label} {role}", fill="black")
                target_view = inputs["images"][position]
                panels = [target_view] + [
                    readouts[name]["evaluator_view"][position]
                    for name in DISPLAY_STATES[1:]
                ]
                for col, panel in enumerate(panels):
                    canvas.paste(_class_render(panel, font_scale), (column_x[col], y))
                draw.text((column_x[1], y + cell + 2), f"{int(readouts['A_start']['e_d_logits'][position].argmax())}", fill="black")
                draw.text((column_x[2], y + cell + 2), f"{int(readouts['B_start']['e_d_logits'][position].argmax())}", fill="black")
                draw.text((column_x[3], y + cell + 2), f"{int(readouts['A_step0300']['e_d_logits'][position].argmax())}", fill="black")
                draw.text((column_x[4], y + cell + 2), f"{int(readouts['B_step0300']['e_d_logits'][position].argmax())}", fill="black")
            filename = f"comparison_{set_name}_page_{page_start // page_rows + 1:02d}.png"
            canvas.save(target / filename, optimize=True)
            page_paths[set_name].append(filename)
    return {
        "pages": page_paths,
        "rows": {name: len(positions) for name, positions in selected_sets.items()},
        "page_rows": page_rows,
        "panel_order": list(DISPLAY_STATES),
        "fixed_display_range": [0.0, 1.0],
        "no_per_image_rescaling": True,
        "e_d_prediction_text_below_generated_views": True,
    }


def _save_training_history(target: Path, histories: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    for arm in ARM_NAMES:
        path = target / f"training_history_{arm}.json"
        payload = {"task_id": TASK_ID, "arm": arm, "updates": len(histories[arm]), "records": list(histories[arm])}
        if path.exists():
            _require(json.loads(path.read_text(encoding="utf-8")) == payload, f"{arm} training history differs")
        else:
            _write_json_new(path, payload, f"{arm} complete fixed-budget training history")


def _make_identity(population: Mapping[str, Any], source_state: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> ResearchRunIdentity:
    parameters = _parameters(consumer)
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="fixed-encoder-ols-small-vs-17x-large-generator-300-updates",
        semantic_authority_id="M0_GENERATOR_SEMANTIC_TARGET_COMPARISON_V1",
        data_role="canonical TRAIN fit=200 update rows; historic comparison=100; confirmation=300; confirmation excluded from gradients",
        config_identity={
            "propagation_grid": 2048,
            "phase_planes_A": [100, 100],
            "phase_planes_B": [100, 400, 100, 400],
            "wavelength_m": 532e-9,
            "pixel_pitch_m": 2e-6,
            "original_layer_spacing_m": 1.5e-3,
            "updates_per_arm": UPDATES_PER_ARM,
            "logical_batch_size": BATCH_SIZE,
            "large_arm_gradient_microbatch_size": LARGE_GRADIENT_MICROBATCH_SIZE,
            "large_arm_l_dist_statistics": "full_logical_batch_ols_gradient_proxy_then_microbatch_chain_rule",
            "seed": SEED,
            "numeric_mode": "strict_fp32_no_tf32_highest_matmul_precision",
            "alpha": semantic.ALPHA,
            "beta": semantic.BETA,
            "gamma": semantic.GAMMA,
            "delta_sic": semantic.DELTA_SIC,
            "kappa": semantic.KAPPA,
        },
        model_identity={
            "source_checkpoint_step": SOURCE_STEP,
            "source_optimizer_steps": dict(source_state["optimizer_step_by_parameter"]),
            "parameter_shapes": {name: list(parameter.shape) for name, parameter in parameters.items()},
            "parameter_counts": {name: parameter.numel() for name, parameter in parameters.items()},
            "all_parameters_finite": all(bool(torch.isfinite(parameter).all().item()) for parameter in parameters.values()),
        },
        optimizer_identity={
            "class": type(optimizer).__name__,
            "parameter_names": ["generator.phase"],
            "lr": GENERATOR_LEARNING_RATE,
            "betas": list(GENERATOR_BETAS),
            "eps": GENERATOR_EPS,
            "weight_decay": GENERATOR_WEIGHT_DECAY,
            "reset": True,
            "source_history_steps_not_used": dict(source_state["optimizer_step_by_parameter"]),
        },
        sampler_identity={
            "fit_source_rows": list(population["source_row_indices"][:FIT_ROWS]),
            "fit_batch_size": BATCH_SIZE,
            "fit_batch_count": BATCH_COUNT,
            "cycles": UPDATES_PER_ARM // BATCH_COUNT,
            "confirmation_source_rows": list(population["source_row_indices"][FIT_ROWS + HISTORIC_ROWS :]),
        },
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )


def _ready_receipt(control: ResearchRunControl, preflight: Mapping[str, Any]) -> PreflightReceipt:
    checks = preflight["checks"]
    return control.make_ready_receipt(
        correctness={
            "passed": True,
            "status": "PASS",
            "zero_phase_expansion_equivalence": checks["zero_phase_expansion_equivalence"]["passed"],
            "four_target_positive_controls": checks["four_evaluator_target_positive_control"]["passed"],
            "wrong_label_negative_control_and_ce_reference": checks["wrong_label_ce_negative_control"]["passed"],
            "encoder_ols_evaluators_and_scientific_buffers_frozen": checks["three_real_updates_and_frozen_scope"]["passed"],
        },
        resource={
            "passed": True,
            "status": "PASS",
            "cuda_runtime": True,
            "whole_device_safety_line": 0.90,
            "resource_observations": {"before": preflight["cuda_before"], "after": preflight["cuda_after"]},
            "disk_observation": preflight["disk_observation"],
        },
        implementation={
            "passed": True,
            "status": "PASS",
            "selected_engine": "cuda_2048_expanded_generator_only",
            "three_real_updates_per_arm": True,
            "a_a_repeatability": checks["three_real_updates_and_frozen_scope"]["A_A_repeatability"],
            "checkpoint_recovery": checks["interrupt_resume"]["passed"],
            "strict_fp32": True,
        },
        selected_engine="cuda_2048_expanded_generator_only",
        resource_envelope={"cuda": preflight["cuda_after"], "disk": preflight["disk_observation"]},
    )


def _prepare_output(root: Path, target: Path, *, resume: bool) -> None:
    output_root = (root / "outputs").resolve()
    target = target.resolve()
    _require(target != output_root and output_root in target.parents, "output directory must be a named child of outputs/")
    parts = target.relative_to(output_root).parts
    _require(len(parts) == 3 and re.fullmatch(r"\d{8}_[A-Za-z0-9][A-Za-z0-9_-]*", parts[0]) is not None and parts[1] == "runs", "output must be outputs/YYYYMMDD_<version>/runs/<attempt>")
    _require(not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in parts), "protected or temporary output namespace is forbidden")
    if target.exists():
        _require(target.is_dir() and resume, "output exists; use --resume with an explicit checkpoint")
    else:
        _require(not resume, "resume output directory does not exist")
        target.mkdir(parents=True, exist_ok=False)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="explicit checkpoint required with --resume")
    return parser.parse_args()


def execute(arguments: argparse.Namespace | None = None) -> dict[str, Any]:
    arguments = arguments or _arguments()
    root = _root()
    parent = _resolve(root, arguments.parent_dir)
    target = _resolve(root, arguments.output_dir)
    started = time.perf_counter()
    _require(bool(arguments.resume) == (arguments.checkpoint is not None), "--resume requires an explicit --checkpoint path")
    _require(not (arguments.preflight_only and arguments.resume), "--preflight-only cannot be combined with --resume")
    parent_is_explicit_default = parent.is_dir()
    _require(parent_is_explicit_default, f"parent experiment directory is missing: {parent}")
    existing_run_control: ResearchRunControl | None = None
    phase = "input_validation"
    progress: dict[str, Any] = {}
    try:
        inputs, population, paths, source_state = _load_inputs(root, parent)
        phase = "preflight"
        if arguments.resume:
            _require(target.is_dir(), "resume output directory is missing")
            saved_population = json.loads((target / "input_population.json").read_text(encoding="utf-8"))
            _require(saved_population == population, "resume population or explicit input paths changed")
            saved_preflight = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
            _require(saved_preflight.get("task_id") == TASK_ID and saved_preflight.get("status") == "READY", "resume preflight is not READY")
            receipt = PreflightReceipt.from_mapping(saved_preflight["receipt"])
            selected = _read_checkpoint(_resolve(root, arguments.checkpoint), target, population)
            completed = _completed_resume_result(target, selected)
            if completed is not None:
                return completed
            preflight = saved_preflight["preflight"]
        else:
            _require(not target.exists(), f"output exists; use --resume with an explicit checkpoint: {target}")
            if arguments.preflight_only:
                _prepare_output(root, target, resume=False)
            with strict_fp32_mode():
                preflight = _preflight(inputs, source_state, parent / "arm_b_step0300.pt", population, target)
            progress["preflight_ready"] = True
            _require(preflight.get("status") == "READY", "normal execution did not receive an executed READY preflight")
            if arguments.preflight_only:
                control_consumer, control_optimizer, _ = _new_endpoint(inputs, source_state, "A_small", torch.device("cuda:0"))
                identity = _make_identity(population, source_state, control_consumer, control_optimizer)
                control = ResearchRunControl(identity=identity, authorization_id=f"{TASK_ID}:authorization:01")
                receipt = _ready_receipt(control, preflight)
                control.mark_preflight_ready(receipt)
                _write_json_new(target / "input_population.json", population, "fixed parent LightGen A/B population")
                _write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight}, "standalone READY preflight")
                del control_consumer, control_optimizer
                return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "training_updates": 0, "next_action": "NONE"}
            control_consumer, control_optimizer, _ = _new_endpoint(inputs, source_state, "A_small", torch.device("cuda:0"))
            identity = _make_identity(population, source_state, control_consumer, control_optimizer)
            del control_consumer, control_optimizer
            existing_run_control = ResearchRunControl(identity=identity, authorization_id=f"{TASK_ID}:authorization:01")
            receipt = _ready_receipt(existing_run_control, preflight)
            existing_run_control.mark_preflight_ready(receipt)
            existing_run_control.start_execution()
            existing_run_control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt)
            _write_json_new(target / "input_population.json", population, "fixed 600-row LightGen small/large comparison population")
            _write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight}, "executed zero-update and three-step READY preflight")
            _seed_all(SEED)
            initial_rng = _rng_state()
            _write_torch_new(target / "initial_rng_state.pt", initial_rng, "common small/large arm RNG start")
            selected = None
        if arguments.resume:
            # Resume binds the existing READY receipt to the explicit checkpoint and actual input state.
            control_consumer, control_optimizer, _ = _new_endpoint(inputs, source_state, "A_small", torch.device("cuda:0"))
            identity = _make_identity(population, source_state, control_consumer, control_optimizer)
            del control_consumer, control_optimizer
            existing_run_control = ResearchRunControl(identity=identity, authorization_id=receipt.authorization_id)
            existing_run_control.mark_preflight_ready(receipt)
            existing_run_control.start_execution()
            existing_run_control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)
            initial_rng = torch.load(target / "initial_rng_state.pt", map_location="cpu", weights_only=True)

        phase = "measurement_and_training"
        device = torch.device("cuda:0")
        with strict_fp32_mode() as numeric_mode:
            evaluators = diagnostic.load_evaluators(paths, device)
            evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
            target_control = _target_metrics(evaluators, inputs)
            _require(_target_control_passed(target_control), "target evaluator positive control changed after preflight")
            target_payload = {name: {"logits": value["logits"], "metrics": value["metrics"]} for name, value in target_control.items()}
            target_control_path = target / "target_image_evaluator_positive_controls.pt"
            if target_control_path.exists():
                saved_target_payload = torch.load(target_control_path, map_location="cpu", weights_only=True)
                _require(_tree_equal(saved_target_payload, target_payload), "saved target-image evaluator positive controls changed during recovery")
            else:
                _write_torch_new(target_control_path, target_payload, "four frozen target-image evaluator positive controls")
            readouts: dict[str, Mapping[str, Any]] = {}
            summaries: dict[str, Mapping[str, Any]] = {}
            for endpoint, arm in (("A_start", "A_small"), ("B_start", "B_large")):
                endpoint_path = target / f"readouts_{endpoint}.pt"
                if endpoint_path.exists():
                    current = torch.load(endpoint_path, map_location="cpu", weights_only=True)
                    _validate_readout(current, inputs, endpoint)
                    summary, _rows = _save_measurement(target, endpoint, current, inputs)
                else:
                    consumer, optimizer, _parameters_value = _new_endpoint(inputs, source_state, arm, device)
                    current = _forward_all(consumer, evaluators, inputs, arm)
                    _validate_readout(current, inputs, endpoint)
                    summary, _rows = _save_measurement(target, endpoint, current, inputs)
                    del consumer, optimizer, _parameters_value
                    gc.collect()
                    torch.cuda.empty_cache()
                readouts[endpoint] = current
                summaries[endpoint] = summary
            base_equal = torch.allclose(readouts["A_start"]["raw_cmos"], readouts["B_start"]["raw_cmos"], rtol=RTOL, atol=ATOL)
            _require(base_equal, "A and B starting outputs differ after zero-phase expansion")
            baseline_equivalence = {
                "passed": True,
                "all_600_rows_checked": True,
                "raw_cmos_max_abs_difference": float((readouts["A_start"]["raw_cmos"] - readouts["B_start"]["raw_cmos"]).abs().max().item()),
                "z_max_abs_difference": float((readouts["A_start"]["z"] - readouts["B_start"]["z"]).abs().max().item()),
                "rtol": RTOL,
                "atol": ATOL,
            }

            histories: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_NAMES}
            resume_arm = selected["arm"] if arguments.resume else None
            if arguments.resume:
                histories[resume_arm] = [dict(row) for row in selected.get("training_history", [])]
            if not arguments.resume or resume_arm == "A_small":
                a_result, a_extra = _run_arm(
                    "A_small", inputs, source_state, evaluators, target, population, initial_rng,
                    start_from=selected if arguments.resume else None, started=started,
                )
                histories["A_small"] = a_extra["history"]
                readouts["A_step0300"] = a_result["readouts"]
                summaries["A_step0300"] = a_result["summary"]
            else:
                a_readout = torch.load(target / "readouts_A_step0300.pt", map_location="cpu", weights_only=True)
                _validate_readout(a_readout, inputs, "A_step0300")
                readouts["A_step0300"] = a_readout
                summaries["A_step0300"], _ = _save_measurement(target, "A_step0300", a_readout, inputs)
                a_history = json.loads((target / "training_history_A_small.json").read_text(encoding="utf-8"))["records"]
                histories["A_small"] = a_history
            if not arguments.resume or resume_arm in (None, "A_small"):
                b_start_from = None
            else:
                b_start_from = selected
            b_result, b_extra = _run_arm(
                "B_large", inputs, source_state, evaluators, target, population, initial_rng,
                start_from=b_start_from, started=started,
            )
            histories["B_large"] = b_extra["history"]
            readouts["B_step0300"] = b_result["readouts"]
            summaries["B_step0300"] = b_result["summary"]
            _require(set(readouts) == set(DISPLAY_STATES[1:]), "one or more fixed endpoints are missing")
            _save_training_history(target, histories)
            for name, model in evaluators.items():
                _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"final {name} evaluator")
                _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"final {name} evaluator state is not frozen")
            _resource_telemetry(target, device, "measurement_complete", {"A_updates": UPDATES_PER_ARM, "B_updates": UPDATES_PER_ARM}, started)

            transitions = {
                "B_step0300_vs_A_step0300": _paired_correctness(readouts["A_step0300"], readouts["B_step0300"], inputs["labels"]),
                "B_step0300_vs_B_start": _paired_correctness(readouts["B_start"], readouts["B_step0300"], inputs["labels"]),
            }
            final = _final_decision(inputs["labels"], summaries, readouts, frozen_valid=True)
            montage = _make_comparison_pages(target, inputs, readouts)
            classification_report = {
                "schema": REPORT_SCHEMA,
                "task_id": TASK_ID,
                **final,
                "scientific_question": "whether a 17x Generator phase parameter expansion improves class 1/7 E_D correct counts against both the small endpoint and large start within the same 300-update rule",
                "population": population,
                "source_checkpoint": {
                    "path": population["source_checkpoint"],
                    "source_step": SOURCE_STEP,
                    "source_optimizer_steps_validated_and_reset_for_generator": population["source_optimizer_steps"],
                    "source_terminal_status": population["source_task_terminal_status"],
                    "source_terminal_classification": population["source_task_classification"],
                },
                "architecture": {
                    "A_small": {"generator_parameter_shape": [2, 1, 100, 100], "generator_parameter_count": 20_000, "added_planes": 0},
                    "B_large": {
                        "generator_parameter_shape": [PHASE_LARGE_COUNT],
                        "generator_parameter_count": PHASE_LARGE_COUNT,
                        "old_phase_count": PHASE_OLD_COUNT,
                        "new_phase_count": PHASE_NEW_COUNT,
                        "parameter_ratio": PHASE_LARGE_COUNT / 20_000,
                        "layout": [
                            {"name": "original_phase_0", "shape": [1, 100, 100], "z_relative_m": 0.0},
                            {"name": "new_phase_0", "shape": [1, 400, 400], "z_relative_m": 0.75e-3},
                            {"name": "original_phase_1", "shape": [1, 100, 100], "z_relative_m": 1.5e-3},
                            {"name": "new_phase_1", "shape": [1, 400, 400], "z_relative_m": 2.25e-3},
                        ],
                        "new_phase_window_fov_m": [800e-6, 800e-6],
                        "outside_window_transmission": 1.0,
                        "intermediate_crop": False,
                    },
                    "propagation_grid": 2048,
                    "wavelength_m": 532e-9,
                    "pixel_pitch_m": 2e-6,
                    "old_crop_locations_mm_relative_to_generator_input": [1.5, 3.0],
                    "baseline_equivalence": baseline_equivalence,
                },
                "training": {
                    "updates_per_arm": UPDATES_PER_ARM,
                    "fit_batches": BATCH_COUNT,
                    "batch_size": BATCH_SIZE,
                    "large_arm_gradient_microbatch_size": LARGE_GRADIENT_MICROBATCH_SIZE,
                    "large_arm_l_dist_batch_statistics_preserved": True,
                    "cycles": UPDATES_PER_ARM // BATCH_COUNT,
                    "parameter_scope": "Generator phase only",
                    "source_Adam_steps_retained_as_provenance_not_used": population["source_optimizer_steps"],
                    "fresh_Adam": {"lr": GENERATOR_LEARNING_RATE, "betas": list(GENERATOR_BETAS), "eps": GENERATOR_EPS, "weight_decay": GENERATOR_WEIGHT_DECAY},
                    "semantic_objective": "three evaluator true-class cross entropy with existing joint Encoder/Generator L2 normalization, equal weighting and joint RMS normalization",
                    "bogt_weights": {"alpha": semantic.ALPHA, "beta": semantic.BETA, "gamma": semantic.GAMMA, "delta_sic": semantic.DELTA_SIC, "kappa": semantic.KAPPA},
                    "numeric_mode": numeric_mode,
                    "history_artifacts": {arm: f"training_history_{arm}.json" for arm in ARM_NAMES},
                },
                "endpoint_metrics": summaries,
                "paired_correctness_transitions_confirmation": transitions,
                "readout_artifacts": {name: f"readouts_{name}.pt" for name in readouts},
                "row_metric_artifacts": {name: f"row_metrics_{name}.pt" for name in readouts},
                "metric_artifacts": {name: f"metrics_{name}.json" for name in readouts},
                "checkpoint_artifacts": sorted(path.name for path in target.glob("arm_*_step*.pt")),
                "resource_telemetry_artifact": RESOURCE_TELEMETRY_NAME,
                "comparison_images": montage,
                "preflight_artifact": "preflight.json",
                "elapsed_seconds_observation_only": time.perf_counter() - started,
                "next_action": "NONE",
                "stop": True,
            }
            if existing_run_control is not None:
                if classification_report["status"] == "VALID_COMPLETED":
                    existing_run_control.complete()
                else:
                    existing_run_control.valid_failed()
                classification_report["research_run_state"] = existing_run_control.state
            _write_json_new(target / "report.json", classification_report, "terminal small-versus-large Generator scientific decision")
            return classification_report
    except TrainingInterrupted as error:
        if existing_run_control is not None:
            existing_run_control.interrupted()
        interrupted = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "status": "INTERRUPTED",
            "classification": None,
            "training_recommendation": None,
            "failure": str(error),
            "next_action": "resume only from an explicit saved checkpoint",
        }
        if target.is_dir() and not (target / "interrupted.json").exists():
            _write_json_new(target / "interrupted.json", interrupted, "interrupted fixed-budget A/B experiment")
        return interrupted
    except Exception as error:
        if existing_run_control is not None and existing_run_control.state in {"RUN_PREPARING", "RUNNING", "SCIENTIFIC_ACTIVE"}:
            existing_run_control.invalid_harness()
        if not arguments.preflight_only or target.is_dir():
            _write_invalid(target, phase, error, started, progress)
        raise


def main() -> None:
    result = execute()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
