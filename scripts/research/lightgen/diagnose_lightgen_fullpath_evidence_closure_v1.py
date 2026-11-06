"""Close the direct-OLS versus full SLM-path evidence chain without updates."""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from ldm_project.v2.evidence.io import cpu_copy

from ldm_project.v2.application.research.shared.research_run_control import (
    CORRECTNESS_INVARIANTS,
    IMPLEMENTATION_SELECTION,
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    RESOURCE_ENVELOPE,
)
from scripts.research.lightgen import (
    diagnose_lightgen_fullpath_zero_update_attribution_v1 as attribution,
)
from scripts.research.lightgen import (
    diagnose_lightgen_terminal_projection_alignment_v1 as alignment,
)
from scripts.research.lightgen import (
    execute_lightgen_g1_slm_projected_mapping_v2 as projected,
)
from scripts.research.lightgen import (
    execute_lightgen_small_joint_update_readability_pilot_v1 as joint,
)
from scripts.research.lightgen import (
    execute_lightgen_recent_optical_fullpath_semantic_v1 as recent,
)

ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_FULLPATH_EVIDENCE_CLOSURE_V1"
AUTHORIZATION_ID = "USER_APPROVED_LIGHTGEN_FULLPATH_EVIDENCE_CLOSURE_20261102"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/fullpath_evidence_closure_v1.yaml"
OUTPUT_DIR = (
    ROOT / "outputs/20261102_lightgen_fullpath_evidence_closure_v1/runs/attempt_01"
)
BACKUP_DIR = (
    ROOT
    / "outputs/20261102_lightgen_fullpath_evidence_closure_v1/runs/autodl_backup_attempt_01"
)
REVIEW_DIR = (
    ROOT
    / "outputs/20261102_lightgen_fullpath_evidence_closure_v1/runs/windows_review_01"
)
CLASS_COUNT = 10
ROWS_PER_CLASS = 200
POPULATION_ROWS = CLASS_COUNT * ROWS_PER_CLASS
BATCH_SIZE = 10
CHUNK_ROWS = 100
BOOTSTRAP_COUNT = 2_000
BOOTSTRAP_SEED = 202611021170
RECALL_MARGIN = 0.02
DIVERSITY_MIN = 0.8
REINJECTION_TOLERANCE = 1.0e-12
CROSS_GRID_TOLERANCE = 1.0e-4
GRADIENT_RTOL = 1.0e-5
GRADIENT_ATOL = 1.0e-8
CONTROL_ROWS = torch.tensor(
    [
        label * ROWS_PER_CLASS + offset
        for label in range(CLASS_COUNT)
        for offset in range(2)
    ],
    dtype=torch.long,
)
RUN_ID = "lightgen-fullpath-evidence-closure-attempt-01"
REPAIR_TASK_ID = "LIGHTGEN_FULLPATH_EVIDENCE_CLOSURE_REPAIR_V1"
REPAIR_AUTHORIZATION_ID = "USER_APPROVED_LIGHTGEN_FULLPATH_EVIDENCE_CLOSURE_REPAIR_20261102"
REPAIR_CONFIG = ROOT / "configs/research/lightgen/fullpath_evidence_closure_repair_v1.yaml"
REPAIR_OUTPUT_DIR = (
    ROOT / "outputs/20261102_lightgen_fullpath_evidence_closure_repair_v1/runs/attempt_02"
)
REPAIR_BACKUP_DIR = (
    ROOT / "outputs/20261102_lightgen_fullpath_evidence_closure_repair_v1/runs/autodl_backup_attempt_02"
)
REPAIR_REVIEW_DIR = (
    ROOT / "outputs/20261102_lightgen_fullpath_evidence_closure_repair_v1/runs/windows_review_02"
)
ENGINEERING_SEED = 202611021171
ENGINEERING_UPDATES = 4
ENGINEERING_RECOVERY_RTOL = 2.0e-6
ENGINEERING_RECOVERY_ATOL = 5.0e-8


def _require(
    condition: bool, message: str, error: type[Exception] = RuntimeError
) -> None:
    if not condition:
        raise error(message)


class _OptimizerEventAudit:
    """Count optimizer construction and successful step calls in one process."""

    def __init__(self) -> None:
        self.optimizer_instances = 0
        self.optimizer_updates = 0
        self._original_init: Any = None

    def __enter__(self) -> "_OptimizerEventAudit":
        self._original_init = torch.optim.Optimizer.__init__
        original_init = self._original_init

        def counted_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            original_init(instance, *args, **kwargs)
            self.optimizer_instances += 1
            original_step = instance.step

            def counted_step(*step_args: Any, **step_kwargs: Any) -> Any:
                result = original_step(*step_args, **step_kwargs)
                self.optimizer_updates += 1
                return result

            instance.step = counted_step

        torch.optim.Optimizer.__init__ = counted_init
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        torch.optim.Optimizer.__init__ = self._original_init

    def snapshot(self) -> dict[str, int]:
        return {
            "optimizer_instances": self.optimizer_instances,
            "optimizer_updates": self.optimizer_updates,
        }

    def require_zero(self, context: str) -> None:
        _require(
            self.optimizer_instances == 0 and self.optimizer_updates == 0,
            f"{context} must not construct or update an optimizer; observed {self.snapshot()}",
        )


def _path(value: str | Path) -> Path:
    target = Path(value)
    return target if target.is_absolute() else ROOT / target


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}", ValueError)
    return value


def _write_json(
    path: Path, value: Mapping[str, Any], *, exclusive: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _save_torch(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not exclusive or not path.exists(),
        f"refusing to replace task evidence: {path}",
        FileExistsError,
    )
    temporary = path.with_name(path.name + f".{time.time_ns()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_torch(path: Path) -> Any:
    _require(path.is_file(), f"required evidence is missing: {path}", FileNotFoundError)
    return torch.load(path, map_location="cpu", weights_only=False)


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(
        isinstance(value, dict)
        and value.get("schema") == "lightgen_fullpath_evidence_closure.v1"
        and value.get("task_id") == TASK_ID
        and value.get("authorization_id") == AUTHORIZATION_ID
        and value.get("evidence_tier") == "diagnostic_only",
        "full-path evidence-closure config identity changed",
        ValueError,
    )
    _require(
        value.get("sources", {}).get("expected_training_run_id")
        == "d88d5bfc2b29488884a88512152815d6"
        and value["sources"].get("expected_endpoint_global_step") == 52_000
        and value["sources"].get("expected_encoder_adam_step") == 47_000
        and value["sources"].get("expected_generator_adam_step") == 52_000
        and value["sources"].get("expected_population_seed_base") == 202610300930
        and value["sources"].get("expected_mapper_run_id")
        == "3521afc652f2427eac984485971340c4",
        "registered endpoint, mapper, or consumed-population identity changed",
        ValueError,
    )
    _require(
        value.get("population")
        == {
            "role": "consumed_train_diagnostic",
            "rows": 2_000,
            "rows_per_class": 200,
            "class_count": 10,
            "same_saved_targets_and_proposals": True,
            "official_test_reads": 0,
        },
        "registered population identity or role changed",
        ValueError,
    )
    _require(
        value.get("measurement")
        == {
            "endpoint": "E1G1",
            "fullpath_conditions": ["P0_E1G1", "P1_E1G1"],
            "direct_condition": "D_DIRECT_OLS",
            "production_grid": 2048,
            "reference_grid": 4096,
            "batch_size": 10,
            "chunk_rows": 100,
            "control_rows_per_class": 2,
            "direct_ols_reinjection_max_abs": REINJECTION_TOLERANCE,
            "direct_ols_cross_grid_relative_max": CROSS_GRID_TOLERANCE,
            "f1_reinjection_max_abs": REINJECTION_TOLERANCE,
            "bootstrap_resamples": BOOTSTRAP_COUNT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "macro_recall_noninferiority_margin": RECALL_MARGIN,
            "diversity_retention_min": DIVERSITY_MIN,
            "training_gradient_rtol": GRADIENT_RTOL,
            "training_gradient_atol": GRADIENT_ATOL,
        },
        "registered measurement or path-retention thresholds changed",
        ValueError,
    )
    _require(
        value.get("training_contract")
        == {
            "forward": "TRAIN_image_to_SLM_to_Encoder_to_Gaussian_OLS_to_Generator_to_raw_CMOS",
            "objective": "existing_BOGT_plus_fixed_SSIM",
            "trainable_groups": ["encoder_phase", "generator_phase"],
            "allowed_single_variable_freezes": ["encoder_phase", "generator_phase"],
            "frozen_groups": [
                "mapper",
                "input_projection",
                "prior",
                "modal_scale",
                "camera_gain",
                "evaluators",
            ],
            "optimizer_instances": 0,
            "optimizer_updates": 0,
            "prior_refits": 0,
            "prior_samples": 0,
        },
        "full-path training contract changed",
        ValueError,
    )
    _require(
        value.get("output")
        == {
            "run_dir": OUTPUT_DIR.relative_to(ROOT).as_posix(),
            "backup_dir": BACKUP_DIR.relative_to(ROOT).as_posix(),
            "review_dir": REVIEW_DIR.relative_to(ROOT).as_posix(),
            "direct_ols_2048_rows": 2_000,
            "direct_ols_reinjection_2048_rows": 20,
            "direct_ols_reference_4096_rows": 20,
            "fullpath_training_no_update_rows": 10,
            "p0_fullpath_reruns": 0,
            "p1_fullpath_reruns": 0,
            "optimizer_instances": 0,
            "optimizer_updates": 0,
            "prior_refits": 0,
            "prior_samples": 0,
            "official_test_reads": 0,
            "preserve_source_artifacts": True,
            "stop_after_one_path_retention_decision": True,
        },
        "registered output paths or zero-update boundary changed",
        ValueError,
    )
    return value


def _read_repair_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(
        isinstance(value, dict)
        and value.get("schema") == "lightgen_fullpath_evidence_closure_repair.v1"
        and value.get("task_id") == REPAIR_TASK_ID
        and value.get("authorization_id") == REPAIR_AUTHORIZATION_ID
        and value.get("evidence_tier") == "engineering_repair_only",
        "full-path evidence-closure repair config identity changed",
        ValueError,
    )
    _require(
        value.get("sources", {}).get("expected_training_run_id")
        == "d88d5bfc2b29488884a88512152815d6"
        and value["sources"].get("expected_endpoint_global_step") == 52_000
        and value["sources"].get("expected_encoder_adam_step") == 47_000
        and value["sources"].get("expected_generator_adam_step") == 52_000
        and value["sources"].get("generation_population_role")
        == "consumed_generation_diagnostic"
        and value["sources"].get("source_population_role_recorded_as")
        == "consumed_train_diagnostic",
        "repair config must preserve the frozen endpoint and explicitly correct population role",
        ValueError,
    )
    _require(
        value.get("engineering")
        == {
            "optimizer_groups": ["encoder_phase", "generator_phase"],
            "single_group_freeze_supported": False,
            "schedule_rows": [0, 1, 2, 3],
            "required_update_calls": 5,
            "sequence": [
                "three_continuous",
                "fourth_direct",
                "fourth_independent_restore",
            ],
            "checkpoint_after_continuous_updates": 3,
            "recovery_rtol": ENGINEERING_RECOVERY_RTOL,
            "recovery_atol": ENGINEERING_RECOVERY_ATOL,
            "deterministic_seed": ENGINEERING_SEED,
            "negative_controls": [
                "detach_rec",
                "detach_ssim",
                "detach_rec_and_ssim",
            ],
            "rollback_isolated_candidate": True,
            "scientific_updates": 0,
            "prior_refits": 0,
            "prior_samples": 0,
            "official_test_reads": 0,
        },
        "repair engineering sequence or zero-science boundary changed",
        ValueError,
    )
    _require(
        value.get("evidence_consumer")
        == {
            "direct_cross_grid_tolerance": CROSS_GRID_TOLERANCE,
            "p1_reinjection_max_abs": REINJECTION_TOLERANCE,
            "fixed_population_rows": POPULATION_ROWS,
            "bootstrap_resamples": BOOTSTRAP_COUNT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "macro_recall_noninferiority_margin": RECALL_MARGIN,
            "diversity_retention_min": DIVERSITY_MIN,
            "fixed_examples_per_class": 2,
            "response_finite_difference_epsilon": 2.0e-4,
            "response_finite_difference_seed": 202611021050,
            "response_finite_difference_tolerance": 1.0e-5,
            "response_finite_difference_endpoint_global_step": 52_000,
        },
        "repair evidence consumer thresholds or saved population changed",
        ValueError,
    )
    _require(
        value.get("output")
        == {
            "run_dir": REPAIR_OUTPUT_DIR.relative_to(ROOT).as_posix(),
            "backup_dir": REPAIR_BACKUP_DIR.relative_to(ROOT).as_posix(),
            "review_dir": REPAIR_REVIEW_DIR.relative_to(ROOT).as_posix(),
            "preserve_source_artifacts": True,
        },
        "repair output paths changed",
        ValueError,
    )
    return value


def _load_p1_output(
    artifact_dir: Path, assets: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = _load_torch(artifact_dir / "population_manifest.pt")
    generation = assets["generation"]
    _require(
        manifest.get("task_id") == alignment.TASK_ID
        and manifest.get("population_role") == "consumed_train_diagnostic"
        and manifest.get("population_rows") == POPULATION_ROWS
        and manifest.get("candidate_projection") == "R1_terminal_E1G1_plus_P1"
        and torch.equal(manifest.get("labels"), generation["labels"])
        and torch.equal(manifest.get("proposal_ids"), generation["proposal_ids"])
        and torch.equal(
            manifest.get("prior_targets"), generation["saved_prior_targets"]
        ),
        "saved P1 population manifest does not bind the fixed prior targets",
    )
    commands = torch.empty(
        (POPULATION_ROWS, 1, *alignment.IMAGE_SHAPE), dtype=torch.float32
    )
    controls = _load_torch(artifact_dir / "controls_2048.pt")
    remaining = torch.tensor(
        [
            row
            for row in range(POPULATION_ROWS)
            if row not in set(alignment.CONTROL_ROWS.tolist())
        ],
        dtype=torch.long,
    )
    commands[alignment.CONTROL_ROWS] = controls["actual_slm_commands_forwarded"].float()
    for start in range(0, len(remaining), alignment.CHUNK_SIZE):
        indices = remaining[start : start + alignment.CHUNK_SIZE]
        shard = _load_torch(artifact_dir / alignment._safe_shard_name(indices.tolist()))
        alignment._validate_shard(
            shard, indices, commands, assets, compare_commands=False
        )
        commands[indices] = shard["actual_slm_commands_forwarded"].float()
    output = alignment._load_candidate_output(artifact_dir, assets, commands)
    _require(
        torch.equal(output["labels"], generation["labels"])
        and torch.equal(output["proposal_ids"], generation["proposal_ids"])
        and torch.equal(output["targets"], generation["saved_prior_targets"])
        and torch.equal(output["commands"], commands)
        and tuple(output["coefficients"].shape) == (POPULATION_ROWS, 100)
        and tuple(output["raw_cmos"].shape)
        == (POPULATION_ROWS, 1, *alignment.IMAGE_SHAPE),
        "saved P1 optical output is incomplete or misbound",
    )
    cross_grid = _load_torch(artifact_dir / "cross_grid_controls.pt")
    control_indices = CONTROL_ROWS
    _require(
        cross_grid.get("task_id") == alignment.TASK_ID
        and torch.equal(cross_grid.get("population_indices"), control_indices)
        and torch.equal(
            cross_grid.get("labels"), generation["labels"].index_select(0, control_indices)
        )
        and torch.equal(
            cross_grid.get("proposal_ids"),
            generation["proposal_ids"].index_select(0, control_indices),
        )
        and torch.equal(
            cross_grid.get("actual_slm_commands_forwarded"),
            output["commands"].index_select(0, control_indices),
        )
        and torch.equal(
            cross_grid.get("coefficients_2048"),
            output["coefficients"].index_select(0, control_indices),
        )
        and torch.equal(
            cross_grid.get("raw_cmos_2048"),
            output["raw_cmos"].index_select(0, control_indices),
        )
        and tuple(cross_grid.get("coefficients_4096", torch.empty(0)).shape)
        == (len(control_indices), 100)
        and tuple(cross_grid.get("raw_cmos_4096", torch.empty(0)).shape)
        == (len(control_indices), 1, *alignment.IMAGE_SHAPE),
        "saved P1 cross-grid controls do not bind the same rows, commands, targets, and labels",
    )
    output["cross_grid_controls"] = cross_grid
    return output


def _load_assets(
    config: Mapping[str, Any],
    device: torch.device,
    projection_source_dir: Path | None = None,
    projection_review_path: Path | None = None,
    training_preflight_path: Path | None = None,
    gradient_diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    sources = config["sources"]
    projection_config = alignment._read_config(
        _path(sources["terminal_projection_config"])
    )
    assets = alignment._load_assets(projection_config, device)
    projection_dir = _path(
        projection_source_dir or sources["terminal_projection_backup"]
    ).resolve()
    selected_review_path = _path(
        projection_review_path
        or (_path(sources["terminal_projection_review"]) / "independent_review.json")
    ).resolve()
    projection_review = _read_json(selected_review_path)
    _require(
        projection_review.get("passed") is True
        and projection_review.get("per_sample_rows") == POPULATION_ROWS
        and projection_review.get("semantic_recomputed_from_raw_cmos") is True,
        "P1 terminal-projection evidence has not passed its independent saved-result review",
    )
    p1 = _load_p1_output(projection_dir, assets)
    p0_bundle = assets["attribution_outputs"]
    _require(
        p0_bundle.get("task_id") == attribution.TASK_ID
        and p0_bundle.get("population_role") == "consumed_diagnostic_population"
        and torch.equal(p0_bundle.get("labels"), assets["generation"]["labels"])
        and torch.equal(
            p0_bundle.get("proposal_ids"), assets["generation"]["proposal_ids"]
        )
        and torch.equal(p0_bundle.get("commands"), assets["baseline_commands"])
        and p0_bundle.get("phase_sources", {})
        .get("E1G1", {})
        .get("source_training_run_id")
        == sources["expected_training_run_id"],
        "P0 full-path output does not bind the same E1G1 endpoint, labels, proposals, and commands",
    )
    p0 = p0_bundle["conditions"]["E1G1"]
    _require(
        torch.equal(p0["coefficients"], assets["baseline_outputs"]["coefficients"])
        and torch.equal(p0["raw_cmos"], assets["baseline_outputs"]["raw_cmos"]),
        "P0 output differs from the independently loaded same-endpoint source",
    )
    recent_backup = _path(sources["fullpath_backup"])
    selected_training_preflight = _path(
        training_preflight_path or (recent_backup / "training_preflight.json")
    ).resolve()
    training_preflight = _read_json(selected_training_preflight)
    historical_preflight_scope = _validate_historical_d0_training_preflight(
        training_preflight
    )
    selected_gradient_diagnostics = _path(
        gradient_diagnostics_path
        or (_path(sources["attribution_backup"]) / "loss_gradient_diagnostics.json")
    ).resolve()
    gradient_path = selected_gradient_diagnostics
    gradient = _read_json(gradient_path)
    attribution_source_identity = _read_json(
        selected_gradient_diagnostics.parent / "source_identity.json"
    )
    terminal_gradient_scope = _validate_terminal_gradient_diagnostics(
        gradient,
        attribution_source_identity,
        expected_training_run_id=sources["expected_training_run_id"],
    )
    training_evidence_scope = {
        "schema": "lightgen_fullpath_training_evidence_scope.v1",
        "source_training_run_id": sources["expected_training_run_id"],
        "terminal_endpoint": {
            "global_step": 52_000,
            "encoder_adam_step": 47_000,
            "generator_adam_step": 52_000,
        },
        "historical_training_preflight": {
            **historical_preflight_scope,
            "source": (
                Path(sources["fullpath_backup"]) / "training_preflight.json"
            ).as_posix(),
            "role": "historical_D0_entrypoint_validation_only_not_E1G1_terminal_evidence",
        },
        "terminal_gradient_diagnostics": {
            **terminal_gradient_scope,
            "source": (
                Path(sources["attribution_backup"])
                / "loss_gradient_diagnostics.json"
            ).as_posix(),
            "binding_source": (
                Path(sources["attribution_backup"]) / "source_identity.json"
            ).as_posix(),
            "role": "frozen_E1G1_step_52000_gradient_diagnostic_no_update",
        },
    }
    return {
        "projection_config": projection_config,
        "alignment": assets,
        "runtime": assets["runtime"],
        "generation": assets["generation"],
        "cohort": assets["population"]["cohort"],
        "p0": p0,
        "p1": p1,
        "training_preflight": training_preflight,
        "training_preflight_path": selected_training_preflight,
        "historical_preflight_scope": historical_preflight_scope,
        "gradient_diagnostics": gradient,
        "gradient_diagnostics_path": selected_gradient_diagnostics,
        "training_evidence_scope": training_evidence_scope,
        "projection_dir": projection_dir,
        "projection_review_path": selected_review_path,
    }


def _validate_historical_d0_training_preflight(
    training_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    selected_checkpoint = str(training_preflight.get("selected_checkpoint", ""))
    normalized_checkpoint = selected_checkpoint.replace("\\", "/")
    gradients = training_preflight.get("cmos_only_phase_gradient_norms", {})
    _require(
        training_preflight.get("status") == "READY"
        and training_preflight.get("selected_endpoint") == "D0"
        and training_preflight.get("selected_global_step") == 50_000
        and selected_checkpoint
        and normalized_checkpoint.endswith("/D0/checkpoint_step_005000.pt")
        and training_preflight.get("initial_optimizer_steps") == [45_000, 50_000]
        and training_preflight.get("preflight_rolled_back") is True
        and training_preflight.get("mapper_only_optimizer_rejected") is True
        and training_preflight.get("scientific_updates") == 0
        and training_preflight.get("official_test_reads") == 0
        and training_preflight.get("recovery", {}).get("passed") is True
        and training_preflight.get(
            "frozen_adapter_projection_prior_camera_and_evaluators_unchanged"
        )
        is True
        and math.isfinite(float(gradients.get("encoder_cmos_gradient_norm", 0.0)))
        and float(gradients.get("encoder_cmos_gradient_norm", 0.0)) > 0.0
        and math.isfinite(float(gradients.get("generator_cmos_gradient_norm", 0.0)))
        and float(gradients.get("generator_cmos_gradient_norm", 0.0)) > 0.0,
        "historical training preflight must be explicitly bound to D0 step 50000",
    )
    return {
        "endpoint": "D0",
        "global_step": 50_000,
        "optimizer_steps": [45_000, 50_000],
        "selected_checkpoint": selected_checkpoint,
        "preflight_rolled_back": True,
        "recovery_passed": True,
        "encoder_cmos_gradient_norm": float(
            gradients["encoder_cmos_gradient_norm"]
        ),
        "generator_cmos_gradient_norm": float(
            gradients["generator_cmos_gradient_norm"]
        ),
    }


def _validate_terminal_gradient_diagnostics(
    gradient: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    *,
    expected_training_run_id: str,
) -> dict[str, Any]:
    endpoint = gradient.get("endpoints", {}).get("step_02000", {})
    batches = endpoint.get("batches", [])
    finite_difference = endpoint.get("finite_difference", [])
    _require(
        source_identity.get("task_id") == attribution.TASK_ID
        and source_identity.get("terminal_run_id") == expected_training_run_id
        and source_identity.get("terminal_global_step") == 52_000
        and source_identity.get("optimizer_steps")
        == {"encoder_adam_step": 47_000, "generator_adam_step": 52_000}
        and gradient.get("task_id") == attribution.TASK_ID
        and gradient.get("optimizer_updates") == 0
        and endpoint.get("model_unchanged") is True
        and len(batches) == 10
        and len(finite_difference) == 4,
        "terminal gradient diagnostics are not bound to the registered E1G1 endpoint",
    )
    for batch in batches:
        phase_groups = batch.get("phase_groups", {})
        for group in ("encoder", "generator"):
            total_norm = float(phase_groups.get(group, {}).get("total_gradient_norm", 0.0))
            _require(
                math.isfinite(total_norm) and total_norm > 0.0,
                f"saved {group} terminal gradient norm is missing or zero",
            )
            for component in ("rec", "ssim"):
                component_norm = float(
                    phase_groups[group].get("components", {})
                    .get(component, {}).get("norm", 0.0)
                )
                _require(
                    math.isfinite(component_norm) and component_norm > 0.0,
                    f"saved {group} {component} gradient norm is missing or zero",
                )
    seen: set[tuple[str, float]] = set()
    for record in finite_difference:
        group = str(record.get("group", ""))
        requested = float(record.get("requested_rad", math.nan))
        analytic = float(record.get("analytic_directional_derivative", math.nan))
        central = float(record.get("central_difference", math.nan))
        reported_error = float(record.get("relative_error", math.nan))
        _require(
            group in {"encoder", "generator"}
            and requested in {0.005, 0.01}
            and math.isfinite(analytic)
            and math.isfinite(central)
            and math.isfinite(reported_error)
            and analytic * central > 0.0
            and math.isclose(
                reported_error,
                abs(analytic - central) / max(abs(analytic), abs(central)),
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ),
            "saved terminal finite-difference record is incomplete or internally inconsistent",
        )
        seen.add((group, requested))
    _require(
        seen == {
            ("encoder", 0.005), ("encoder", 0.01),
            ("generator", 0.005), ("generator", 0.01),
        },
        "saved terminal finite differences omit an optical phase group or perturbation",
    )
    return {
        "diagnostic_endpoint_key": "step_02000",
        "bound_global_step": 52_000,
        "model_unchanged": True,
        "optimizer_updates": 0,
        "batches": len(batches),
        "finite_difference_records": len(finite_difference),
        "finite_difference_groups": ["encoder", "generator"],
    }


def _terminal_checkpoint(assets: Mapping[str, Any]) -> Mapping[str, Any]:
    checkpoint = assets["alignment"]["assets"]["checkpoints"][2_000]
    _require(
        checkpoint.get("run_id") == "d88d5bfc2b29488884a88512152815d6"
        and checkpoint.get("completed_global_step") == 52_000
        and checkpoint.get("optimizer_steps")
        == {
            "encoder_adam_step": 47_000,
            "generator_adam_step": 52_000,
        },
        "terminal optical checkpoint or its actual Adam counters changed",
    )
    return checkpoint


def _new_direct_model(
    assets: Mapping[str, Any], grid: int, device: torch.device
) -> torch.nn.Module:
    checkpoint = _terminal_checkpoint(assets)
    model = recent._new_endpoint_model(checkpoint, grid, assets["runtime"])
    model.eval()
    physical = model.runtime.physical_optics
    mode = model.runtime.ols.mode_identity()
    model_device = next(model.parameters()).device
    expected_device = torch.device(device)
    device_matches = model_device.type == expected_device.type and (
        expected_device.index is None or model_device.index == expected_device.index
    )
    _require(
        model.runtime.ols.basis_id == "gaussian_s12_v1"
        and mode.get("transport") == "T=I_100 coefficient space"
        and physical is not None
        and physical.global_fiber_geometry is False
        and physical.fiber_attenuation_db_per_km == 0.0
        and physical.material is None
        and device_matches,
        "direct OLS reference is not the registered zero-attenuation, no-material Gaussian backend",
    )
    _require(
        model.runtime.config.propagation_grid_size == grid
        and model.runtime.camera_plane.shape == alignment.IMAGE_SHAPE,
        "direct OLS reference grid or CMOS geometry changed",
    )
    return model


def _direct_batch(
    model: torch.nn.Module,
    targets: torch.Tensor,
    camera_area: float,
    camera_gain: float,
) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        measured = projected._forward_target(
            model,
            targets.to(
                device=next(model.parameters()).device,
                dtype=torch.complex128,
            ),
        )
        readouts = (
            recent.pilot._readout_for_evaluator(
                measured["raw_cmos"].detach().cpu(),
                camera_area,
                camera_gain,
            )
            .detach()
            .cpu()
            .float()
        )
    _require(
        tuple(measured["raw_cmos"].shape) == (len(targets), 1, *alignment.IMAGE_SHAPE)
        and tuple(readouts.shape) == (len(targets), 1, 28, 28)
        and torch.isfinite(measured["raw_cmos"]).all().item(),
        "direct OLS -> Generator -> CMOS forward returned invalid output",
    )
    return {
        "raw_cmos": measured["raw_cmos"].detach().cpu().double(),
        "w_per_pixel": measured["w_per_pixel"].detach().cpu().double(),
        "readouts": readouts,
    }


def _validate_direct_shard(
    shard: Mapping[str, Any], indices: torch.Tensor, assets: Mapping[str, Any]
) -> None:
    generation = assets["generation"]
    runtime = assets["runtime"]
    endpoint = _terminal_checkpoint(assets)
    area = float(runtime["camera_area"])
    _require(
        shard.get("task_id") == TASK_ID
        and shard.get("schema") == "lightgen_direct_ols_population_shard.v1"
        and shard.get("condition") == "D_DIRECT_OLS"
        and shard.get("population_role")
        in {"consumed_train_diagnostic", "consumed_generation_diagnostic"}
        and shard.get("endpoint_source_run_id") == endpoint.get("run_id")
        and shard.get("endpoint_global_step") == 52_000
        and shard.get("encoder_adam_step") == 47_000
        and shard.get("generator_adam_step") == 52_000
        and shard.get("grid_size") == 2048
        and shard.get("ols_basis_id") == "gaussian_s12_v1"
        and shard.get("ols_transport") == "T=I_100 coefficient space"
        and shard.get("modal_coefficient_scale")
        == "registered_native_complex_OLS_scale"
        and shard.get("raw_cmos_unit") == "W/m^2"
        and shard.get("w_per_pixel_unit") == "W/pixel"
        and shard.get("readout_unit") == "dimensionless_[0,1]"
        and math.isclose(
            float(shard.get("camera_pixel_area_m2", math.nan)),
            area,
            rel_tol=0.0,
            abs_tol=1.0e-24,
        )
        and math.isclose(
            float(shard.get("camera_gain_w_inverse", math.nan)),
            float(runtime["camera_gain"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and torch.equal(shard.get("population_indices"), indices)
        and torch.equal(
            shard.get("labels"), generation["labels"].index_select(0, indices)
        )
        and torch.equal(
            shard.get("proposal_ids"),
            generation["proposal_ids"].index_select(0, indices),
        )
        and torch.equal(
            shard.get("targets"),
            generation["saved_prior_targets"].index_select(0, indices),
        )
        and torch.equal(shard.get("coefficients"), shard.get("targets"))
        and tuple(shard.get("raw_cmos", torch.empty(0)).shape)
        == (len(indices), 1, *alignment.IMAGE_SHAPE)
        and tuple(shard.get("w_per_pixel", torch.empty(0)).shape)
        == (len(indices), 1, *alignment.IMAGE_SHAPE)
        and tuple(shard.get("readouts", torch.empty(0)).shape)
        == (len(indices), 1, 28, 28)
        and tuple(shard.get("modal_power_native", torch.empty(0)).shape)
        == (len(indices),)
        and torch.isfinite(shard["modal_power_native"]).all().item()
        and torch.isfinite(shard["raw_cmos"]).all().item()
        and torch.isfinite(shard["w_per_pixel"]).all().item()
        and torch.isfinite(shard["readouts"]).all().item()
        and bool((shard["raw_cmos"] >= 0.0).all())
        and bool(((shard["readouts"] >= 0.0) & (shard["readouts"] <= 1.0)).all()),
        "saved D direct-OLS shard is incomplete, duplicated, or bound to another proposal",
    )
    _require(
        torch.allclose(
            shard["coefficients"].abs().square().sum(dim=1),
            shard["modal_power_native"],
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        and torch.allclose(
            shard["w_per_pixel"], shard["raw_cmos"] * area, rtol=1.0e-12, atol=1.0e-14
        ),
        "saved D CMOS pixel power does not use the registered W/m^2 to W/pixel conversion",
    )


def _preflight_training_forward(
    assets: Mapping[str, Any], model: torch.nn.Module, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = assets["runtime"]
    saved = assets["gradient_diagnostics"]["endpoints"]["step_02000"]["batches"][0]
    rows = torch.as_tensor(saved["sample_rows"], dtype=torch.long)
    labels = runtime["labels"].long().cpu()
    train_indices = runtime["train_indices"].long().cpu()
    _require(
        tuple(rows.shape) == (BATCH_SIZE,)
        and bool(torch.isin(rows, train_indices).all())
        and torch.equal(labels.index_select(0, rows), torch.arange(CLASS_COUNT))
        and saved["labels"] == list(range(CLASS_COUNT)),
        "saved first training batch is not the actual class-balanced TRAIN schedule",
    )
    encoder, generator = recent.base._phase_groups(model)
    for parameter in model.parameters():
        parameter.requires_grad_(parameter is encoder or parameter is generator)
        parameter.grad = None
    source = {
        "checkpoint": dict(checkpoint),
        "training_labels": labels,
        "training_indices": train_indices,
        "training_pixels": runtime["pixels"],
    }
    state_before = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    optimizer_state_before = copy.deepcopy(checkpoint["optimizer"])
    evaluator_modules = {
        name: module
        for name, module in (
            ("ED", runtime["input_bundle"]["evaluator_ed"]),
            ("EC", runtime["input_bundle"]["evaluator_ec"]),
        )
    }
    frozen_evaluators_before = {
        name: {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }
        for name, module in evaluator_modules.items()
    }
    _require(
        _terminal_checkpoint(assets).get("optimizer_steps")
        == checkpoint.get("optimizer_steps"),
        "training preflight checkpoint identity changed",
    )
    rows_device = rows.to(runtime["device"])
    parts, ssim, _native_gradient, result = recent.decoder._loss_parts(
        model,
        runtime["pixels"],
        rows_device,
        runtime["device"],
        checkpoint["scale"].to(runtime["device"]),
        checkpoint["power_reference"].to(runtime["device"]),
        checkpoint["camera_gain"].to(runtime["device"]),
    )
    witness = recent.joint._validate_fullpath_training_forward(
        model,
        runtime["pixels"],
        rows_device,
        source,
        result,
        parts,
        dict(checkpoint),
        ssim,
    )
    weights = checkpoint["weights"]
    weighted = {
        "dist": float(weights["dist"]) * parts["dist"],
        "rec": float(weights["rec"]) * parts["rec"],
        "op": float(weights["op"]) * parts["op"],
        "ssim": float(checkpoint["structural_weight"]) * ssim,
    }
    total = sum(weighted.values())
    params = (encoder, generator)
    names = ("dist", "rec", "op", "ssim")
    gradients: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    for name in names:
        gradients[name] = torch.autograd.grad(
            weighted[name], params, retain_graph=True, allow_unused=True
        )
    total_gradients = torch.autograd.grad(
        total, params, retain_graph=True, allow_unused=True
    )
    cmos = weighted["rec"] + weighted["ssim"]
    cmos_gradients = torch.autograd.grad(
        cmos, params, retain_graph=True, allow_unused=True
    )
    expected = assets["gradient_diagnostics"]["endpoints"]["step_02000"]["batches"][0]
    _require(
        math.isclose(
            float(total.detach()),
            float(expected["total"]),
            rel_tol=GRADIENT_RTOL,
            abs_tol=GRADIENT_ATOL,
        ),
        "same-batch production BOGT+SSIM loss differs from the saved terminal diagnostic",
    )
    observed_losses: dict[str, float] = {}
    gradient_rows: dict[str, Any] = {}
    for name in names:
        observed_losses[name] = float(weighted[name].detach())
        _require(
            math.isclose(
                observed_losses[name],
                float(expected["weighted_losses"][name]),
                rel_tol=GRADIENT_RTOL,
                abs_tol=GRADIENT_ATOL,
            ),
            f"same-batch weighted {name} loss differs from its saved diagnostic",
        )
    for group_index, group_name in enumerate(("encoder", "generator")):
        components = {}
        for name in names:
            gradient = gradients[name][group_index]
            norm = (
                0.0
                if gradient is None
                else float(torch.linalg.vector_norm(gradient.detach().double()))
            )
            reference = expected["phase_groups"][group_name]["components"][name]
            _require(
                math.isclose(
                    norm,
                    float(reference["norm"]),
                    rel_tol=GRADIENT_RTOL,
                    abs_tol=GRADIENT_ATOL,
                ),
                f"same-batch weighted {name} gradient for {group_name} differs from its saved diagnostic",
            )
            components[name] = {"norm": norm, "structural_zero": gradient is None}
        total_gradient = total_gradients[group_index]
        total_norm = (
            0.0
            if total_gradient is None
            else float(torch.linalg.vector_norm(total_gradient.detach().double()))
        )
        _require(
            math.isclose(
                total_norm,
                float(expected["phase_groups"][group_name]["total_gradient_norm"]),
                rel_tol=GRADIENT_RTOL,
                abs_tol=GRADIENT_ATOL,
            ),
            f"same-batch total gradient for {group_name} differs from its saved diagnostic",
        )
        cmos_gradient = cmos_gradients[group_index]
        cmos_norm = (
            0.0
            if cmos_gradient is None
            else float(torch.linalg.vector_norm(cmos_gradient.detach().double()))
        )
        _require(
            math.isfinite(cmos_norm) and cmos_norm > 0.0,
            f"same-forward CMOS loss is disconnected from {group_name} phase",
        )
        gradient_rows[group_name] = {
            "weighted_component_gradients": components,
            "total_gradient_norm": total_norm,
            "cmos_loss_gradient_norm": cmos_norm,
        }
    _require(
        all(parameter.grad is None for parameter in model.parameters())
        and all(
            torch.equal(state_before[key], value.detach().cpu())
            for key, value in model.state_dict().items()
        ),
        "zero-update training path check changed phase or non-phase model state",
    )
    _require(
        recent.base._assert_close(
            checkpoint["optimizer"], optimizer_state_before, "closure_Adam_payload"
        )
        == 0.0,
        "training diagnostic mutated the saved Adam payload",
    )
    frozen_evaluators_unchanged = all(
        set(frozen_evaluators_before[name]) == set(module.state_dict())
        and all(
            torch.equal(frozen_evaluators_before[name][key], value.detach().cpu())
            for key, value in module.state_dict().items()
        )
        for name, module in evaluator_modules.items()
    )
    _require(
        frozen_evaluators_unchanged, "training diagnostic changed a frozen evaluator"
    )
    return {
        "task_id": TASK_ID,
        "status": "PASSED",
        "sample_rows": rows.tolist(),
        "labels": labels.index_select(0, rows).tolist(),
        "fullpath_witness": witness,
        "weighted_losses": observed_losses,
        "total_loss": float(total.detach()),
        "phase_gradients": gradient_rows,
        "saved_gradient_reference": "attempt_05/loss_gradient_diagnostics.json step_02000 batch_000",
        "saved_finite_difference_reused": True,
        "optimizer_instances": 0,
        "optimizer_updates": 0,
        "model_unchanged": True,
        "adam_payload_unchanged": True,
        "frozen_evaluators_unchanged": frozen_evaluators_unchanged,
    }


def _phase_snapshots(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _row_relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    _require(
        reference.ndim >= 2
        and candidate.shape == reference.shape
        and torch.isfinite(reference).all().item()
        and torch.isfinite(candidate).all().item(),
        "relative error inputs must be finite matching row arrays",
        ValueError,
    )
    dtype = torch.complex128 if reference.is_complex() else torch.float64
    reference = reference.detach().cpu().to(dtype).reshape(reference.shape[0], -1)
    candidate = candidate.detach().cpu().to(dtype).reshape(candidate.shape[0], -1)
    return torch.linalg.vector_norm(candidate - reference, dim=1) / torch.linalg.vector_norm(
        reference, dim=1
    ).clamp_min(1.0e-30)


def _aggregate_relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    dtype = torch.complex128 if reference.is_complex() else torch.float64
    reference = reference.detach().cpu().to(dtype).reshape(-1)
    candidate = candidate.detach().cpu().to(dtype).reshape(-1)
    return float(
        torch.linalg.vector_norm(candidate - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1.0e-30)
    )


def _assert_state_equal(
    before: Mapping[str, torch.Tensor], model: torch.nn.Module, label: str
) -> None:
    current = model.state_dict()
    _require(
        set(before) == set(current)
        and all(
            torch.equal(before[key], value.detach().cpu())
            for key, value in current.items()
        ),
        f"frozen {label} state changed",
    )


def _make_identity(config: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=RUN_ID,
        semantic_authority_id=AUTHORIZATION_ID,
        data_role="consumed_train_diagnostic_same_saved_prior_population",
        config_identity=config,
        model_identity={
            "training_run_id": config["sources"]["expected_training_run_id"],
            "global_step": 52_000,
            "encoder_adam_step": 47_000,
            "generator_adam_step": 52_000,
            "architecture": "2+2_100x100_gaussian_s12",
        },
        optimizer_identity={"instances": 0, "updates": 0},
        sampler_identity={
            "population_seed_base": 202610300930,
            "rows": 2000,
            "new_prior_samples": 0,
        },
        execution_source_semantics_id=f"{TASK_ID}:fixed-consumed-population:zero-update",
        scientific_attempt_id=f"{TASK_ID}:attempt_01",
    )


def _make_preflight_receipt(
    control: ResearchRunControl,
    correctness: Mapping[str, Any],
    resource: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> PreflightReceipt:
    selected_engine = implementation.get(
        "selected_engine", implementation.get("engine")
    )
    _require(
        isinstance(selected_engine, str) and bool(selected_engine.strip()),
        "preflight implementation must declare its selected engine",
    )
    return control.make_ready_receipt(
        correctness=correctness,
        resource=resource,
        implementation=implementation,
        selected_engine=selected_engine,
        resource_envelope={
            "cuda": True,
            "population_rows": POPULATION_ROWS,
            "no_elapsed_time_cap": True,
        },
        created_at=time.time(),
    )


def _preflight(
    config: Mapping[str, Any], assets: Mapping[str, Any], device: torch.device
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    runtime = assets["runtime"]
    frozen_snapshot = recent._snapshot_frozen(runtime, assets["alignment"]["mapper"])
    checkpoint = _terminal_checkpoint(assets)
    model2048 = _new_direct_model(assets, 2048, device)
    model2048_state = _phase_snapshots(model2048)
    controls_targets = assets["generation"]["saved_prior_targets"].index_select(
        0, CONTROL_ROWS
    )
    controls_output = _direct_batch(
        model2048,
        controls_targets,
        runtime["camera_area"],
        float(runtime["camera_gain"]),
    )
    p1_coefficients = assets["p1"]["coefficients"].index_select(0, CONTROL_ROWS)
    with torch.inference_mode():
        injected = projected._forward_target(
            model2048,
            p1_coefficients.to(device=device, dtype=torch.complex128),
        )
    p1_raw = assets["p1"]["raw_cmos"].index_select(0, CONTROL_ROWS).to(torch.float64)
    p1_w_pixel = p1_raw * float(runtime["camera_area"])
    raw_injection_error = float(
        (injected["raw_cmos"].detach().cpu().double() - p1_raw).abs().max()
    )
    wpixel_injection_error = float(
        (injected["w_per_pixel"].detach().cpu().double() - p1_w_pixel).abs().max()
    )
    _require(
        raw_injection_error <= REINJECTION_TOLERANCE
        and wpixel_injection_error <= REINJECTION_TOLERANCE,
        "actual P1 OLS coefficient reinjection does not reproduce its saved Generator/CMOS output",
    )

    model4096 = _new_direct_model(assets, 4096, device)
    reference_output = _direct_batch(
        model4096,
        controls_targets,
        runtime["camera_area"],
        float(runtime["camera_gain"]),
    )
    raw_row_errors = _row_relative_l2(
        controls_output["raw_cmos"], reference_output["raw_cmos"]
    )
    wpixel_row_errors = _row_relative_l2(
        controls_output["w_per_pixel"], reference_output["w_per_pixel"]
    )
    raw_relative = float(raw_row_errors.max())
    wpixel_relative = float(wpixel_row_errors.max())
    _require(
        raw_relative <= CROSS_GRID_TOLERANCE
        and wpixel_relative <= CROSS_GRID_TOLERANCE,
        "direct OLS 2048/4096 per-row CMOS or W/pixel agreement failed",
    )

    train_model = recent._new_endpoint_model(checkpoint, 2048, runtime)
    train_model.train()
    training_check = _preflight_training_forward(assets, train_model, checkpoint)
    _assert_state_equal(model2048_state, model2048, "2048 direct reference model")
    _require(
        training_check["model_unchanged"] is True
        and training_check["adam_payload_unchanged"] is True
        and training_check["optimizer_instances"]
        == training_check["optimizer_updates"]
        == 0,
        "full-path zero-update preflight did not preserve the endpoint and Adam state",
    )
    recent._assert_frozen_unchanged(
        frozen_snapshot, runtime, assets["alignment"]["mapper"]
    )
    state = {
        "schema": "lightgen_fullpath_evidence_closure_preflight.v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "source_training_run_id": checkpoint.get("run_id"),
        "endpoint_global_step": checkpoint.get("completed_global_step"),
        "optimizer_steps": checkpoint.get("optimizer_steps"),
        "population_rows": POPULATION_ROWS,
        "population_role": "consumed_generation_diagnostic",
        "direct_ols_2048_controls": 20,
        "direct_ols_4096_controls": 20,
        "f1_actual_ols_reinjection_controls": 20,
        "training_no_update_rows": 10,
        "frozen_inputs_unchanged": True,
        "optimizer_instances": 0,
        "optimizer_updates": 0,
        "prior_refits": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "f1_reinjection": {
            "raw_cmos_max_abs_error": raw_injection_error,
            "w_per_pixel_max_abs_error": wpixel_injection_error,
            "tolerance": REINJECTION_TOLERANCE,
            "passed": True,
        },
        "direct_ols_cross_grid": {
            "input_coefficients_identical_across_grid": True,
            "raw_cmos_max_relative_error": raw_relative,
            "raw_cmos_aggregate_relative_error": _aggregate_relative_l2(
                controls_output["raw_cmos"], reference_output["raw_cmos"]
            ),
            "raw_cmos_per_row_relative_errors": raw_row_errors.tolist(),
            "w_per_pixel_max_relative_error": wpixel_relative,
            "w_per_pixel_aggregate_relative_error": _aggregate_relative_l2(
                controls_output["w_per_pixel"], reference_output["w_per_pixel"]
            ),
            "w_per_pixel_per_row_relative_errors": wpixel_row_errors.tolist(),
            "tolerance": CROSS_GRID_TOLERANCE,
            "passed": True,
        },
        "training_path": training_check,
        "registered_training_preflight_reused": assets["training_preflight"],
    }
    controls = {
        "direct_2048": {
            "schema": "lightgen_direct_ols_population_shard.v1",
            "task_id": TASK_ID,
            "condition": "D_DIRECT_OLS",
            "population_role": "consumed_generation_diagnostic",
            "endpoint_source_run_id": checkpoint["run_id"],
            "endpoint_global_step": 52_000,
            "encoder_adam_step": 47_000,
            "generator_adam_step": 52_000,
            "grid_size": 2048,
            "ols_basis_id": "gaussian_s12_v1",
            "ols_transport": "T=I_100 coefficient space",
            "modal_coefficient_scale": "registered_native_complex_OLS_scale",
            "raw_cmos_unit": "W/m^2",
            "w_per_pixel_unit": "W/pixel",
            "readout_unit": "dimensionless_[0,1]",
            "camera_pixel_area_m2": float(runtime["camera_area"]),
            "camera_gain_w_inverse": float(runtime["camera_gain"]),
            "population_indices": CONTROL_ROWS,
            "labels": assets["generation"]["labels"].index_select(0, CONTROL_ROWS),
            "proposal_ids": assets["generation"]["proposal_ids"].index_select(
                0, CONTROL_ROWS
            ),
            "targets": controls_targets,
            "coefficients": controls_targets.clone(),
            "raw_cmos": controls_output["raw_cmos"],
            "w_per_pixel": controls_output["w_per_pixel"],
            "readouts": controls_output["readouts"],
            "camera_pixel_area_m2": float(runtime["camera_area"]),
            "modal_power_native": controls_targets.abs().square().sum(dim=1),
            "input_boundary": "saved_complex_target_z_at_Gaussian_OLS_interface",
            "output_boundary": "raw_CMOS_intensity_W_per_m2",
        },
        "direct_4096": {
            "population_indices": CONTROL_ROWS,
            "labels": assets["generation"]["labels"].index_select(0, CONTROL_ROWS),
            "proposal_ids": assets["generation"]["proposal_ids"].index_select(
                0, CONTROL_ROWS
            ),
            "raw_cmos": reference_output["raw_cmos"],
            "w_per_pixel": reference_output["w_per_pixel"],
            "readouts": reference_output["readouts"],
            "coefficients": controls_targets.clone(),
            "camera_pixel_area_m2": float(runtime["camera_area"]),
        },
        "training": training_check,
    }
    del model2048, model4096, train_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, controls


def _save_direct_shard(
    path: Path,
    indices: torch.Tensor,
    measured: Mapping[str, torch.Tensor],
    assets: Mapping[str, Any],
) -> None:
    generation = assets["generation"]
    targets = generation["saved_prior_targets"].index_select(0, indices)
    runtime = assets["runtime"]
    endpoint = _terminal_checkpoint(assets)
    _save_torch(
        path,
        {
            "schema": "lightgen_direct_ols_population_shard.v1",
            "task_id": TASK_ID,
            "condition": "D_DIRECT_OLS",
            "population_role": "consumed_generation_diagnostic",
            "endpoint_source_run_id": endpoint["run_id"],
            "endpoint_global_step": 52_000,
            "encoder_adam_step": 47_000,
            "generator_adam_step": 52_000,
            "grid_size": 2048,
            "ols_basis_id": "gaussian_s12_v1",
            "ols_transport": "T=I_100 coefficient space",
            "modal_coefficient_scale": "registered_native_complex_OLS_scale",
            "raw_cmos_unit": "W/m^2",
            "w_per_pixel_unit": "W/pixel",
            "readout_unit": "dimensionless_[0,1]",
            "camera_pixel_area_m2": float(runtime["camera_area"]),
            "camera_gain_w_inverse": float(runtime["camera_gain"]),
            "population_indices": indices.long().cpu(),
            "labels": generation["labels"].index_select(0, indices).long().cpu(),
            "proposal_ids": generation["proposal_ids"]
            .index_select(0, indices)
            .long()
            .cpu(),
            "targets": targets.cpu(),
            "coefficients": targets.cpu(),
            "raw_cmos": measured["raw_cmos"].cpu().double(),
            "w_per_pixel": measured["w_per_pixel"].cpu().double(),
            "readouts": measured["readouts"].cpu().float(),
            "modal_power_native": targets.abs().square().sum(dim=1).cpu().double(),
            "input_boundary": "saved_complex_target_z_at_Gaussian_OLS_interface",
            "output_boundary": "raw_CMOS_intensity_W_per_m2",
        },
    )


def _measure_direct_population(
    artifact_dir: Path,
    assets: Mapping[str, Any],
    controls: Mapping[str, Any],
    device: torch.device,
) -> None:
    runtime = assets["runtime"]
    model = _new_direct_model(assets, 2048, device)
    control = controls["direct_2048"]
    _validate_direct_shard(control, CONTROL_ROWS, assets)
    _require(
        torch.equal(control["population_indices"], CONTROL_ROWS)
        and torch.equal(
            control["labels"],
            assets["generation"]["labels"].index_select(0, CONTROL_ROWS),
        )
        and torch.equal(
            control["proposal_ids"],
            assets["generation"]["proposal_ids"].index_select(0, CONTROL_ROWS),
        ),
        "saved direct-OLS controls lost their original target rows or labels",
    )
    controls_path = artifact_dir / "direct_ols_controls_2048.pt"
    if controls_path.is_file():
        prior = _load_torch(controls_path)
        _validate_direct_shard(prior, CONTROL_ROWS, assets)
        _require(
            torch.equal(prior.get("targets"), control["targets"])
            and torch.equal(prior.get("raw_cmos"), control["raw_cmos"])
            and torch.equal(prior.get("w_per_pixel"), control["w_per_pixel"]),
            "saved direct-OLS control shard differs from its READY preflight",
        )
    else:
        _save_torch(controls_path, control)
    chunks = [
        (start, min(start + CHUNK_ROWS, POPULATION_ROWS))
        for start in range(0, POPULATION_ROWS, CHUNK_ROWS)
    ]
    control_lookup = {
        int(index): offset for offset, index in enumerate(CONTROL_ROWS.tolist())
    }
    for start, end in chunks:
        indices = torch.arange(start, end, dtype=torch.long)
        shard_path = artifact_dir / f"direct_ols_2048_{start:04d}_{end:04d}.pt"
        if shard_path.is_file():
            _validate_direct_shard(_load_torch(shard_path), indices, assets)
            continue
        existing_indices = [
            int(index) for index in indices.tolist() if int(index) in control_lookup
        ]
        missing_indices = [
            int(index) for index in indices.tolist() if int(index) not in control_lookup
        ]
        if existing_indices:
            offsets = torch.tensor(
                [control_lookup[index] for index in existing_indices], dtype=torch.long
            )
            rows = torch.tensor(existing_indices, dtype=torch.long)
            raw = control["raw_cmos"].index_select(0, offsets)
            w_per_pixel = control["w_per_pixel"].index_select(0, offsets)
            readouts = control["readouts"].index_select(0, offsets)
        else:
            rows = torch.empty(0, dtype=torch.long)
            raw = torch.empty((0, 1, *alignment.IMAGE_SHAPE), dtype=torch.float64)
            w_per_pixel = torch.empty_like(raw)
            readouts = torch.empty((0, 1, 28, 28), dtype=torch.float32)
        if missing_indices:
            pieces = []
            for batch_start in range(0, len(missing_indices), BATCH_SIZE):
                batch_rows = torch.tensor(
                    missing_indices[batch_start : batch_start + BATCH_SIZE],
                    dtype=torch.long,
                )
                targets = assets["generation"]["saved_prior_targets"].index_select(
                    0, batch_rows
                )
                pieces.append(
                    _direct_batch(
                        model,
                        targets,
                        runtime["camera_area"],
                        float(runtime["camera_gain"]),
                    )
                )
            missing_raw = torch.cat([part["raw_cmos"] for part in pieces])
            missing_w_per_pixel = torch.cat([part["w_per_pixel"] for part in pieces])
            missing_readouts = torch.cat([part["readouts"] for part in pieces])
            rows = torch.cat((rows, torch.tensor(missing_indices, dtype=torch.long)))
            raw = torch.cat((raw, missing_raw))
            w_per_pixel = torch.cat((w_per_pixel, missing_w_per_pixel))
            readouts = torch.cat((readouts, missing_readouts))
        order = torch.argsort(rows)
        ordered_indices = rows.index_select(0, order)
        _require(
            torch.equal(ordered_indices, indices),
            "direct OLS production chunk has missing or duplicate rows",
        )
        measured = {
            "raw_cmos": raw.index_select(0, order),
            "w_per_pixel": w_per_pixel.index_select(0, order),
            "readouts": readouts.index_select(0, order),
        }
        _save_direct_shard(shard_path, indices, measured, assets)
        _write_json(
            artifact_dir / "measurement_progress.json",
            {
                "task_id": TASK_ID,
                "status": "DIRECT_OLS_MEASURING",
                "completed_rows": end,
                "total_rows": POPULATION_ROWS,
                "new_direct_ols_2048_rows": min(end, POPULATION_ROWS),
                "optimizer_instances": 0,
                "optimizer_updates": 0,
            },
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_direct_population(
    artifact_dir: Path, assets: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    coefficients = torch.empty((POPULATION_ROWS, 100), dtype=torch.complex128)
    raw = torch.empty((POPULATION_ROWS, 1, *alignment.IMAGE_SHAPE), dtype=torch.float64)
    w_per_pixel = torch.empty(
        (POPULATION_ROWS, 1, *alignment.IMAGE_SHAPE), dtype=torch.float64
    )
    readouts = torch.empty((POPULATION_ROWS, 1, 28, 28), dtype=torch.float32)
    seen = torch.zeros(POPULATION_ROWS, dtype=torch.bool)
    for start in range(0, POPULATION_ROWS, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, POPULATION_ROWS)
        indices = torch.arange(start, end, dtype=torch.long)
        shard = _load_torch(artifact_dir / f"direct_ols_2048_{start:04d}_{end:04d}.pt")
        _validate_direct_shard(shard, indices, assets)
        _require(
            not seen[indices].any().item(),
            "direct OLS consumer found duplicate population rows",
        )
        coefficients[indices] = shard["coefficients"]
        raw[indices] = shard["raw_cmos"]
        w_per_pixel[indices] = shard["w_per_pixel"]
        readouts[indices] = shard["readouts"]
        seen[indices] = True
    _require(
        seen.all().item(),
        "direct OLS consumer rejects an incomplete 2,000-row population",
    )
    return {
        "coefficients": coefficients,
        "raw_cmos": raw,
        "w_per_pixel": w_per_pixel,
        "readouts": readouts,
        "labels": assets["generation"]["labels"],
        "proposal_ids": assets["generation"]["proposal_ids"],
        "targets": assets["generation"]["saved_prior_targets"],
    }


def _condition_readouts(
    assets: Mapping[str, Any], raw_cmos: torch.Tensor
) -> torch.Tensor:
    runtime = assets["runtime"]
    return (
        recent.pilot._readout_for_evaluator(
            raw_cmos.detach().cpu().double(),
            runtime["camera_area"],
            float(runtime["camera_gain"]),
        )
        .detach()
        .cpu()
        .float()
    )


def _bootstrap_pair(
    labels: np.ndarray,
    direct: np.ndarray,
    candidate: np.ndarray,
    draws: Sequence[np.ndarray],
) -> dict[str, Any]:
    correct_direct = direct == labels
    correct_candidate = candidate == labels
    class_deltas, class_recalls = {}, {}
    boot = np.zeros(BOOTSTRAP_COUNT, dtype=np.float64)
    for label, sample in enumerate(draws):
        indices = np.flatnonzero(labels == label)
        a = correct_direct[indices].astype(np.float64)
        b = correct_candidate[indices].astype(np.float64)
        delta = float(b.mean() - a.mean())
        class_deltas[str(label)] = delta
        class_recalls[str(label)] = {
            "direct": float(a.mean()),
            "candidate": float(b.mean()),
            "delta": delta,
        }
        boot += (b[sample].mean(axis=1) - a[sample].mean(axis=1)) / CLASS_COUNT
    lower, upper = np.quantile(boot, [0.025, 0.975], method="linear")
    return {
        "point_macro_recall_delta": float(np.mean(list(class_deltas.values()))),
        "paired_class_stratified_ci95": [float(lower), float(upper)],
        "bootstrap_resamples": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "per_class": class_recalls,
    }


def _path_retention_decision(
    intervals: Mapping[str, Sequence[float]],
    diversity_ratios: Mapping[str, float],
) -> str:
    _require(
        set(intervals) == {"ED", "EC"}
        and all(
            len(value) == 2 and all(math.isfinite(float(item)) for item in value)
            for value in intervals.values()
        )
        and set(diversity_ratios) == {str(label) for label in range(CLASS_COUNT)}
        and all(math.isfinite(float(value)) for value in diversity_ratios.values()),
        "path decision requires both evaluator intervals and every class diversity ratio",
        ValueError,
    )
    if all(float(intervals[name][0]) >= -RECALL_MARGIN for name in ("ED", "EC")):
        if all(
            float(diversity_ratios[str(label)]) >= DIVERSITY_MIN
            for label in range(CLASS_COUNT)
        ):
            return "WITHIN_PREDECLARED_MARGIN_ON_CONSUMED_POPULATION"
        return "DIVERSITY_PROTECTION_FAILED"
    if any(float(intervals[name][1]) < -RECALL_MARGIN for name in ("ED", "EC")):
        return "DEGRADATION_BEYOND_MARGIN_SUPPORTED"
    return "EVIDENCE_INSUFFICIENT_AT_NONINFERIORITY_MARGIN"


def _numpy_path_retention_decision(
    intervals: Mapping[str, Sequence[float]],
    diversity_ratios: Mapping[str, float],
) -> str:
    """Independent scalar implementation used by the Windows saved-result consumer."""
    if set(intervals) != {"ED", "EC"}:
        raise ValueError("independent decision requires ED and EC intervals")
    expected_classes = {str(label) for label in range(CLASS_COUNT)}
    if set(diversity_ratios) != expected_classes:
        raise ValueError("independent decision requires all ten class diversity ratios")
    for bounds in intervals.values():
        if len(bounds) != 2 or not all(math.isfinite(float(value)) for value in bounds):
            raise ValueError("independent decision received a malformed interval")
    if not all(math.isfinite(float(value)) for value in diversity_ratios.values()):
        raise ValueError("independent decision received a nonfinite diversity ratio")
    lower_bounds_pass = all(
        float(intervals[evaluator][0]) >= -RECALL_MARGIN
        for evaluator in ("ED", "EC")
    )
    diversity_pass = all(
        float(diversity_ratios[str(label)]) >= DIVERSITY_MIN
        for label in range(CLASS_COUNT)
    )
    if lower_bounds_pass and diversity_pass:
        return "WITHIN_PREDECLARED_MARGIN_ON_CONSUMED_POPULATION"
    if lower_bounds_pass:
        return "DIVERSITY_PROTECTION_FAILED"
    if any(
        float(intervals[evaluator][1]) < -RECALL_MARGIN
        for evaluator in ("ED", "EC")
    ):
        return "DEGRADATION_BEYOND_MARGIN_SUPPORTED"
    return "EVIDENCE_INSUFFICIENT_AT_NONINFERIORITY_MARGIN"


def _overall_path_decision(path_decisions: Mapping[str, str]) -> str:
    _require(
        set(path_decisions) == {"P0_E1G1", "P1_E1G1"},
        "overall path decision requires both fixed SLM paths",
        ValueError,
    )
    passed = [
        path_decisions[name] == "WITHIN_PREDECLARED_MARGIN_ON_CONSUMED_POPULATION"
        for name in ("P0_E1G1", "P1_E1G1")
    ]
    if all(passed):
        return "BOTH_FULL_PATHS_WITHIN_PREDECLARED_MARGIN"
    if any(passed):
        return "ONLY_ONE_FULL_PATH_WITHIN_PREDECLARED_MARGIN"
    return "NEITHER_FULL_PATH_WITHIN_PREDECLARED_MARGIN"


def _condition_summary(
    assets: Mapping[str, Any], raw_cmos: torch.Tensor
) -> tuple[dict[str, Any], torch.Tensor]:
    readouts = _condition_readouts(assets, raw_cmos)
    summary = recent.frame._summary_from_readouts(
        assets["runtime"]["input_bundle"],
        assets["cohort"],
        readouts,
    )
    return summary, readouts


def _predictions(summary: Mapping[str, Any], evaluator: str) -> np.ndarray:
    values = summary["conditions"]["direct_decoder"]["evaluators"][evaluator][
        "predicted_labels"
    ]
    result = np.asarray(values, dtype=np.int64)
    _require(
        result.shape == (POPULATION_ROWS,),
        f"{evaluator} does not cover the complete population",
    )
    return result


def _phase_aligned_relative_error(
    actual: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    actual = actual.detach().cpu().to(torch.complex128)
    target = target.detach().cpu().to(torch.complex128)
    inner = (target.conj() * actual).sum(dim=1)
    phase = inner / inner.abs().clamp_min(1.0e-30)
    aligned = actual * phase.conj().unsqueeze(1)
    return torch.linalg.vector_norm(aligned - target, dim=1) / torch.linalg.vector_norm(
        target, dim=1
    ).clamp_min(1.0e-30)


def _analyze(
    assets: Mapping[str, Any], direct: Mapping[str, torch.Tensor]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor]]:
    generation = assets["generation"]
    labels = generation["labels"].long().cpu().numpy()
    _require(
        np.array_equal(labels, np.repeat(np.arange(CLASS_COUNT), ROWS_PER_CLASS)),
        "final analysis requires the exact ordered 200-per-class population",
    )
    p0 = assets["p0"]
    p1 = assets["p1"]
    conditions = {
        "D_DIRECT_OLS": {
            "coefficients": direct["coefficients"],
            "raw_cmos": direct["raw_cmos"],
        },
        "P0_E1G1": {"coefficients": p0["coefficients"], "raw_cmos": p0["raw_cmos"]},
        "P1_E1G1": {"coefficients": p1["coefficients"], "raw_cmos": p1["raw_cmos"]},
    }
    summaries, readouts = {}, {}
    for name, output in conditions.items():
        _require(
            torch.isfinite(output["raw_cmos"]).all().item()
            and torch.isfinite(output["coefficients"].real).all().item()
            and torch.isfinite(output["coefficients"].imag).all().item(),
            f"{name} contains nonfinite optical results",
        )
        summaries[name], readouts[name] = _condition_summary(assets, output["raw_cmos"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [
        rng.integers(0, ROWS_PER_CLASS, size=(BOOTSTRAP_COUNT, ROWS_PER_CLASS))
        for _ in range(CLASS_COUNT)
    ]
    result: dict[str, Any] = {
        "task_id": TASK_ID,
        "population_role": "consumed_generation_diagnostic",
        "population_rows": POPULATION_ROWS,
        "conditions": {},
        "comparisons": {},
        "bootstrap": {
            "resamples": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "sampling": "same within-class row draws reused for both paths and evaluators",
        },
        "optimizer_instances": 0,
        "optimizer_updates": 0,
        "prior_refits": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "human_readability": "not_measured",
        "hardware_acceptance": "not_measured",
    }
    macro = {}
    for name, output in conditions.items():
        metrics = {}
        for evaluator in ("ED", "EC"):
            item = summaries[name]["conditions"]["direct_decoder"]["evaluators"][
                evaluator
            ]
            metrics[evaluator] = {
                "macro_recall": float(item["macro_recall"]),
                "per_class_recall": item["per_class_recall"],
                "predicted_labels": item["predicted_labels"],
            }
        raw = output["raw_cmos"].detach().cpu().double()
        coeff = output["coefficients"].detach().cpu().to(torch.complex128)
        target = generation["saved_prior_targets"].detach().cpu().to(torch.complex128)
        gained = (
            raw
            * float(assets["runtime"]["camera_area"])
            * float(assets["runtime"]["camera_gain"])
        )
        result["conditions"][name] = {
            "evaluators": {
                key: {
                    key2: value
                    for key2, value in val.items()
                    if key2 != "predicted_labels"
                }
                for key, val in metrics.items()
            },
            "predicted_labels": {
                key: val["predicted_labels"] for key, val in metrics.items()
            },
            "mean_total_modal_power": float(coeff.abs().square().sum(1).mean()),
            "mean_raw_cmos_w_per_pixel": float(
                (raw * float(assets["runtime"]["camera_area"])).mean()
            ),
            "display_clipped_pixel_fraction": float(
                ((gained < 0.0) | (gained > 1.0)).double().mean()
            ),
            "ols_target_phase_aligned_relative_error_mean": float(
                _phase_aligned_relative_error(coeff, target).mean()
            ),
            "same_class_diversity": summaries[name]["same_class_diversity"][
                "direct_decoder"
            ]["by_class"],
            "exact_duplicate_count": summaries[name]["same_class_diversity"][
                "direct_decoder"
            ]["exact_duplicate_count"],
        }
        macro[name] = {
            evaluator: _predictions(summaries[name], evaluator)
            for evaluator in ("ED", "EC")
        }

    for name in ("P0_E1G1", "P1_E1G1"):
        comparison = {}
        for evaluator in ("ED", "EC"):
            paired = _bootstrap_pair(
                labels, macro["D_DIRECT_OLS"][evaluator], macro[name][evaluator], draws
            )
            lowers, uppers = paired["paired_class_stratified_ci95"]
            recall_pass = lowers >= -RECALL_MARGIN
            comparison[evaluator] = {
                **paired,
                "noninferiority_margin": -RECALL_MARGIN,
                "lower_bound_at_least_minus_2pp": bool(recall_pass),
            }
        diversity = attribution._diversity_ratios(
            summaries["D_DIRECT_OLS"], summaries[name]
        )
        diversity_pass = all(
            math.isfinite(value) and value >= DIVERSITY_MIN
            for value in diversity.values()
        )
        comparison["class_diversity_retention"] = {
            "per_class_ratio": diversity,
            "minimum_ratio": min(diversity.values()),
            "minimum_required": DIVERSITY_MIN,
            "all_classes_pass": bool(diversity_pass),
        }
        comparison["path_decision"] = _path_retention_decision(
            {
                key: comparison[key]["paired_class_stratified_ci95"]
                for key in ("ED", "EC")
            },
            diversity,
        )
        result["comparisons"][name] = comparison
    path_decisions = {
        name: result["comparisons"][name]["path_decision"]
        for name in ("P0_E1G1", "P1_E1G1")
    }
    result["decision"] = _overall_path_decision(path_decisions)
    both_pass = result["decision"] == "BOTH_FULL_PATHS_WITHIN_PREDECLARED_MARGIN"
    result["both_paths_pass"] = bool(both_pass)
    per_sample: list[dict[str, Any]] = []
    condition_coeff = {
        name: value["coefficients"].detach().cpu().to(torch.complex128)
        for name, value in conditions.items()
    }
    condition_raw = {
        name: value["raw_cmos"].detach().cpu().double()
        for name, value in conditions.items()
    }
    cmos_rms = {
        name: float(
            (condition_raw[name] - condition_raw["D_DIRECT_OLS"]).square().mean().sqrt()
        )
        for name in conditions
    }
    for index, label in enumerate(labels.tolist()):
        row: dict[str, Any] = {
            "population_index": index,
            "label": int(label),
            "proposal_id": int(generation["proposal_ids"][index]),
            "modal_power_direct": float(
                condition_coeff["D_DIRECT_OLS"][index].abs().square().sum()
            ),
            "cmos_rms_direct": cmos_rms["D_DIRECT_OLS"],
        }
        for name in ("P0_E1G1", "P1_E1G1"):
            row[f"{name}_modal_power"] = float(
                condition_coeff[name][index].abs().square().sum()
            )
            row[f"{name}_raw_cmos_rms_delta"] = float(
                (condition_raw[name][index] - condition_raw["D_DIRECT_OLS"][index])
                .square()
                .mean()
                .sqrt()
            )
            row[f"{name}_ols_phase_aligned_relative_error"] = float(
                _phase_aligned_relative_error(
                    condition_coeff[name][index : index + 1],
                    generation["saved_prior_targets"][index : index + 1],
                )[0]
            )
            for evaluator in ("ED", "EC"):
                direct_prediction = int(macro["D_DIRECT_OLS"][evaluator][index])
                candidate_prediction = int(macro[name][evaluator][index])
                row[f"{evaluator}_direct_prediction"] = direct_prediction
                row[f"{evaluator}_{name}_prediction"] = candidate_prediction
                row[f"{evaluator}_{name}_direct_correct"] = direct_prediction == label
                row[f"{evaluator}_{name}_candidate_correct"] = (
                    candidate_prediction == label
                )
                row[f"{evaluator}_{name}_correctness_transition"] = (
                    "correct_to_correct"
                    if direct_prediction == label and candidate_prediction == label
                    else "correct_to_wrong"
                    if direct_prediction == label and candidate_prediction != label
                    else "wrong_to_correct"
                    if direct_prediction != label and candidate_prediction == label
                    else "wrong_to_wrong"
                )
        per_sample.append(row)
    result["correctness_transition_counts"] = {
        evaluator: {
            name: {
                transition: sum(
                    row[f"{evaluator}_{name}_correctness_transition"] == transition
                    for row in per_sample
                )
                for transition in (
                    "correct_to_correct",
                    "correct_to_wrong",
                    "wrong_to_correct",
                    "wrong_to_wrong",
                )
            }
            for name in ("P0_E1G1", "P1_E1G1")
        }
        for evaluator in ("ED", "EC")
    }
    result["raw_cmos_rms_delta_from_direct"] = {
        name: cmos_rms[name] for name in conditions
    }
    result["asset_reuse"] = {
        "P0_fullpath_rows_reused": POPULATION_ROWS,
        "P1_fullpath_rows_reused": POPULATION_ROWS,
        "saved_targets_and_proposals_shared": True,
        "direct_OLS_rows_measured": POPULATION_ROWS,
        "training_and_Adam_states_reused": True,
        "official_test_reads": 0,
    }
    result["interpretation"] = (
        "A pass is bounded noninferiority on the consumed numerical population, not exact equivalence, "
        "human readability, full-distribution validation, or hardware acceptance."
    )
    return result, per_sample, readouts


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), "refusing to write an empty per-sample table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, result: Mapping[str, Any]) -> None:
    width, height = 1400, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((32, 20), "Direct OLS reference vs complete SLM paths", fill="black")
    colors = {
        "D_DIRECT_OLS": (55, 80, 145),
        "P0_E1G1": (58, 143, 110),
        "P1_E1G1": (211, 125, 55),
    }
    panels = [("ED", 80, 60), ("EC", 80, 330)]
    for evaluator, left, top in panels:
        draw.text((left, top - 22), f"{evaluator} per-class recall", fill="black")
        draw.line((left, top + 190, left + 1240, top + 190), fill=(80, 80, 80), width=2)
        for label in range(CLASS_COUNT):
            center = left + 42 + label * 120
            for offset, name in enumerate(("D_DIRECT_OLS", "P0_E1G1", "P1_E1G1")):
                value = result["conditions"][name]["evaluators"][evaluator][
                    "per_class_recall"
                ][label]
                bar = int(max(0.0, min(1.0, float(value))) * 170)
                x = center + (offset - 1) * 22
                draw.rectangle(
                    (x - 7, top + 185 - bar, x + 7, top + 185), fill=colors[name]
                )
            draw.text((center - 5, top + 194), str(label), fill="black")
    draw.text(
        (80, 645),
        "Blue: direct OLS | Green: P0 full SLM path | Orange: P1 full SLM path",
        fill="black",
    )
    image.save(path)


def _write_example_image_comparison(
    path: Path,
    labels: np.ndarray,
    views: Mapping[str, np.ndarray],
    raw_cmos: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    names = ("D_DIRECT_OLS", "P0_E1G1", "P1_E1G1")
    _require(
        set(views) == set(names)
        and set(raw_cmos) == set(names)
        and np.asarray(labels).shape == (POPULATION_ROWS,),
        "example atlas requires aligned D/P0/P1 fixed views and raw CMOS arrays",
        ValueError,
    )
    indices = CONTROL_ROWS.tolist()
    raw_arrays = {
        name: raw_cmos[name].detach().cpu().double().numpy()[:, 0]
        for name in names
    }
    raw_scale = max(float(np.max(raw_arrays[name][indices])) for name in names)
    _require(math.isfinite(raw_scale) and raw_scale > 0.0,
             "raw CMOS atlas has no positive shared display scale", ValueError)
    cell = 112
    row_height = 132
    left = 58
    group_width = 2 * cell + 72
    image = Image.new("RGB", (left + 3 * group_width + 12, 80 + len(indices) * row_height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 15), "Same 20 saved targets: D / P0 / P1 | fixed view and raw CMOS", fill="black")
    for group_index, name in enumerate(names):
        x = left + group_index * group_width
        draw.text((x, 42), name, fill="black")
        draw.text((x, 60), "fixed 28x28", fill="black")
        draw.text((x + cell + 5, 60), "raw 200x200", fill="black")
    for row, index in enumerate(indices):
        y = 80 + row * row_height
        draw.text((5, y + 45), f"c{int(labels[index])} r{index}", fill="black")
        for group_index, name in enumerate(names):
            x = left + group_index * group_width
            view = np.asarray(views[name][index, 0], dtype=np.float64)
            view_u8 = np.clip(view, 0.0, 1.0)
            view_image = Image.fromarray(np.rint(view_u8 * 255.0).astype(np.uint8), mode="L")
            view_image = view_image.resize((cell, cell), Image.Resampling.NEAREST).convert("RGB")
            raw = raw_arrays[name][index]
            raw_u8 = np.rint(np.clip(raw / raw_scale, 0.0, 1.0) * 255.0).astype(np.uint8)
            raw_image = Image.fromarray(raw_u8, mode="L")
            raw_image = raw_image.resize((cell, cell), Image.Resampling.NEAREST).convert("RGB")
            image.paste(view_image, (x, y))
            image.paste(raw_image, (x + cell + 5, y))
            draw.text(
                (x + cell + 5, y + cell + 2),
                f"mean={float(raw.mean()):.3g}",
                fill="black",
            )
            draw.text((x, y + cell + 2), f"mean={float(view.mean()):.3g}", fill="black")
    image.save(path)
    return {
        "population_rows": len(indices),
        "population_indices": indices,
        "rows_per_class": 2,
        "raw_cmos_visualization_common_max_w_per_m2": raw_scale,
        "raw_cmos_values_remain_in_saved_arrays": True,
        "fixed_display_is_unchanged_clamp_and_pool_view": True,
    }


def _paper_alignment_record() -> dict[str, Any]:
    return {
        "paper": {
            "authors": "Yitong Chen et al.",
            "title": "All-optical synthesis chip for large-scale intelligent semantic vision generation",
            "journal": "Science",
            "volume": 390,
            "pages": "1259-1265",
            "year": 2025,
            "doi": "10.1126/science.adv7434",
            "main_text_location": "page 2",
            "supplement_location": "Supplementary Materials page 2",
            "supplement_generation_task_location": "Supplementary Materials pages 2-3",
            "primary_link": "https://doi.org/10.1126/science.adv7434",
        },
        "comparisons": [
            {
                "stage": "training_forward",
                "paper": "Encoder learns the optical latent space and semantic operations are represented in OLS.",
                "project": "TRAIN image -> amplitude SLM -> 2-layer Encoder -> Gaussian OLS -> 2-layer Generator -> raw CMOS; existing BOGT plus fixed SSIM consumes the same forward.",
                "alignment": "full optical training path is enforced; the 2+2, 100x100 model and physical parameters are project-scale choices",
            },
            {
                "stage": "objective",
                "paper": "BOGT includes latent-distribution, output-reconstruction and optical-field/power terms.",
                "project": "existing dist/rec/op terms plus fixed SSIM",
                "alignment": "SSIM is an explicit project extension; this task does not alter losses",
            },
            {
                "stage": "semantic_generation",
                "paper": "Supplementary Materials pages 2-3 distinguish OLS manipulation generation using the Generator from image encoding through the Encoder.",
                "project": "reuse saved OLS targets; frozen mapper plus projection prepares legal amplitude-SLM inputs, then the project sends the command through Encoder and Generator",
                "alignment": "the mapper/projection and re-entry through Encoder are project additions; direct OLS-to-Generator is the closer paper-style generation boundary",
            },
            {
                "stage": "inference",
                "paper": "the paper's learned OLS representation is operated on by the Generator for synthesis; Encoder serves the input/image encoding path",
                "project": "mapper+projection -> SLM -> Encoder -> OLS -> Generator -> raw CMOS",
                "alignment": "P0/P1 are the project's full SLM paths; direct OLS-to-Generator is a numerical boundary reference and must not be presented as the same inference path",
            },
        ],
        "scope": "project-to-paper method mapping; not a claim of exact reproduction",
    }


def _write_closeout(
    path: Path,
    result: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    reviewed: bool,
    optimizer_events: Mapping[str, int] | None = None,
) -> None:
    reuse = result["asset_reuse"]
    training = preflight["training_path"]
    endpoint = preflight["endpoint_global_step"]
    optimizer_steps = preflight["optimizer_steps"]
    table_rows = [
        "| 条件 | E_D 宏召回 | E_C 宏召回 | E_D 相对 D Δ（95% CI） | E_C 相对 D Δ（95% CI） | 最低逐类多样性保留 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    direct = result["conditions"]["D_DIRECT_OLS"]
    table_rows.append(
        f"| D 直接 OLS | {direct['evaluators']['ED']['macro_recall']:.2%} | "
        f"{direct['evaluators']['EC']['macro_recall']:.2%} | — | — | — | 参照 |"
    )
    classwise_declines = []
    for name in ("P0_E1G1", "P1_E1G1"):
        condition = result["conditions"][name]
        comparison = result["comparisons"][name]
        cells = []
        for evaluator in ("ED", "EC"):
            delta = comparison[evaluator]["point_macro_recall_delta"]
            lower, upper = comparison[evaluator]["paired_class_stratified_ci95"]
            cells.append(
                f"{delta * 100:+.2f} pp [{lower * 100:+.2f}, {upper * 100:+.2f}]"
            )
            declines = [
                f"{evaluator} 类 {label}: {item['delta'] * 100:+.1f} pp"
                for label, item in comparison[evaluator]["per_class"].items()
                if item["delta"] < 0.0
            ]
            if declines:
                classwise_declines.append(f"{name}: " + "；".join(declines))
        table_rows.append(
            f"| {name} | {condition['evaluators']['ED']['macro_recall']:.2%} | "
            f"{condition['evaluators']['EC']['macro_recall']:.2%} | {cells[0]} | "
            f"{cells[1]} | "
            f"{comparison['class_diversity_retention']['minimum_ratio']:.2%} | "
            f"{comparison['path_decision']} |"
        )
    lines = [
        "# LightGen 完整光路证据闭环",
        "",
        "## 结果与状态",
        "",
        f"状态：`{'独立复审完成' if reviewed else 'AutoDL 数值测量完成'}`；唯一决策为 `{result['decision']}`。",
        f"本轮科研更新、先验重拟合、新先验采样和官方 TEST 读取均为 0；当前执行进程实际观察到优化器事件 {dict(optimizer_events or {'optimizer_instances': 0, 'optimizer_updates': 0})}。",
        "历史 AutoDL 闭环阶段未 instrument optimizer 事件；旧报告中的 instances=0 已撤回，历史实例与更新的完整总数未知。",
        "",
        "## 科学认识",
        "",
        "D 为保存目标复系数直接进入同一 Gaussian OLS 后端；P0、P1 使用同一批保存先验目标和 proposal，实际经过完整 SLM→Encoder→OLS→Generator→raw CMOS。该2000条人口是已消费先验生成诊断人口，历史元数据曾误标为 TRAIN 诊断。",
        "",
        *table_rows,
        "",
        f"P0 判定：`{result['comparisons']['P0_E1G1']['path_decision']}`；P1 判定：`{result['comparisons']['P1_E1G1']['path_decision']}`。",
        "两条路径同时满足预设 2 个百分点非劣界值和逐类 80% 多样性保护，才支持本人口内的双路径保留结论。",
        "逐类变化并非全部为零；下降类别如下：" + ("；".join(classwise_declines) if classwise_declines else "无。"),
        f"相对 D 的 raw CMOS RMS 差：P0 {result['raw_cmos_rms_delta_from_direct']['P0_E1G1']:.6g} W/m²，P1 {result['raw_cmos_rms_delta_from_direct']['P1_E1G1']:.6g} W/m²。",
        f"固定显示转换截断比例：D {direct['display_clipped_pixel_fraction']:.3%}，P0 {result['conditions']['P0_E1G1']['display_clipped_pixel_fraction']:.3%}，P1 {result['conditions']['P1_E1G1']['display_clipped_pixel_fraction']:.3%}。OLS 目标相位对齐相对误差均值：P0 {result['conditions']['P0_E1G1']['ols_target_phase_aligned_relative_error_mean']:.3%}，P1 {result['conditions']['P1_E1G1']['ols_target_phase_aligned_relative_error_mean']:.3%}。",
        "",
        "## Chen 等 Science 论文训练与推理对应",
        "",
        "依据：Chen et al., *Science* 390, 1259–1265 (2025), DOI [10.1126/science.adv7434](https://doi.org/10.1126/science.adv7434)；主文第 2 页及 Supplementary Materials 第 2 页。",
        "",
        "| 环节 | 论文方法要点 | 本项目实现 | 对齐边界 |",
        "|---|---|---|---|",
        "| 训练前向 | Encoder 学习 OLS；语义表征在 OLS 中采样或操作 | TRAIN 图像→振幅 SLM→2 层 Encoder→Gaussian OLS→2 层 Generator→CMOS；BOGT+固定 SSIM 消费同一次前向 | 完整路径原则一致；当前 2+2、100×100 与物理参数是项目配置 |",
        "| 损失 | BOGT 包含分布、重构与光场/功率约束 | 复用 dist/rec/op，并附加固定 SSIM | SSIM 是项目扩展；本轮不改损失 |",
        "| 语义生成 | 从学习到的 OLS 表示采样或操作语义 | 复用保存 OLS 目标，经冻结 mapper+投影生成合法 SLM 输入 | mapper+投影是项目增加的输入适配器，不等于论文原生 OLS 采样 |",
        "| 推理 | Encoder/OLS/Generator 光学路径形成生成 | mapper+投影→SLM→Encoder→OLS→Generator→raw CMOS | D 直接 OLS 仅作边界参照；P0/P1 才是完整 SLM 路径 |",
        "",
        "## 目标缺口",
        "",
        "本轮只闭合当前固定数值模型与已消费人口上的直接 OLS 对照；不证明人眼可读性、总体 G1 分布、论文规模/物理参数等同或实物部署。训练/生成先验匹配问题仍待后续独立单变量研究。",
        "",
        "## 工程完成与验证",
        "",
        "资产复用与条件身份：",
        "| 资产/条件 | 本轮处理 | 身份或数量 |",
        "|---|---|---:|",
        f"| E1G1 checkpoint | 冻结复用 | run `{preflight['source_training_run_id']}`；global step {endpoint}；Encoder/Generator Adam {optimizer_steps['encoder_adam_step']}/{optimizer_steps['generator_adam_step']} |",
        f"| P0_E1G1 | 复用保存完整路径输出 | {reuse['P0_fullpath_rows_reused']} 条 |",
        f"| P1_E1G1 | 复用保存完整路径输出 | {reuse['P1_fullpath_rows_reused']} 条 |",
        f"| D_DIRECT_OLS | 新测同目标直接 OLS 后端输出 | {reuse['direct_OLS_rows_measured']} 条 |",
        f"| 先验目标/proposal 与输入适配器 | 固定并跨条件共用；mapper、投影、prior、模态尺度、相机增益冻结 | {POPULATION_ROWS} 条；新增采样 0 |",
        "",
        f"直接 OLS 2048/4096 raw CMOS 相对误差：{preflight['direct_ols_cross_grid']['raw_cmos_max_relative_error']:.8g}（阈值 {CROSS_GRID_TOLERANCE:g}）。",
        f"P1 实际 OLS 系数重注入 raw CMOS 最大绝对误差：{preflight['f1_reinjection']['raw_cmos_max_abs_error']:.8g}。",
        f"终点训练入口无更新批次：{len(training['sample_rows'])} 张；Encoder CMOS 梯度范数 {training['phase_gradients']['encoder']['cmos_loss_gradient_norm']:.8g}，Generator CMOS 梯度范数 {training['phase_gradients']['generator']['cmos_loss_gradient_norm']:.8g}。",
        "联合训练入口在 BOGT+SSIM 总损失更新前核对真实 TRAIN 行、合法振幅 SLM 输入、Encoder 层记录、当前 OLS 系数、Generator 层记录及 raw CMOS 单位；直接 OLS 注入或与来源 TRAIN 图像不符的输入会被拒绝。",
        "评估器、光学模型、输入资产及 Adam 保存状态均保持冻结；既有中心差分记录复用。",
        "恢复说明：首轮分片写入发现 W/pixel 丢失后，保留已通过预检、控制读出和同一 run identity；修复后通过显式 `--resume` 完成。由于失败分片未落盘，首个100行分片中80条新前向需重算；最终人口仍严格为原定2,000条，历史失败遥测与源文件覆盖备份均保留。",
        "",
        "## 决策影响与下一步",
        "",
        "该结果只决定 P0/P1 是否在本次数值人口的预定容差内保留直接 OLS 参照性能。通过后再设计独立人口验证；未通过时根据具体失败路径确定下一项单变量研究。本轮不自动追加训练。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _measure(
    config: Mapping[str, Any],
    artifact_dir: Path,
    *,
    resume: bool,
    projection_source_dir: Path | None = None,
    projection_review_path: Path | None = None,
    training_preflight_path: Path | None = None,
    gradient_diagnostics_path: Path | None = None,
    optimizer_audit: _OptimizerEventAudit,
) -> None:
    _require(
        torch.cuda.is_available(),
        "AutoDL CUDA is required for the registered optical measurements",
    )
    device = torch.device("cuda")
    selected_projection_source = _path(
        projection_source_dir
        or config["sources"]["terminal_projection_backup"]
    ).resolve()
    selected_projection_review = _path(
        projection_review_path
        or (_path(config["sources"]["terminal_projection_review"])
            / "independent_review.json")
    ).resolve()
    selected_training_preflight = _path(
        training_preflight_path
        or (_path(config["sources"]["fullpath_backup"]) / "training_preflight.json")
    ).resolve()
    selected_gradient_diagnostics = _path(
        gradient_diagnostics_path
        or (_path(config["sources"]["attribution_backup"])
            / "loss_gradient_diagnostics.json")
    ).resolve()
    assets = _load_assets(
        config,
        device,
        selected_projection_source,
        selected_projection_review,
        selected_training_preflight,
        selected_gradient_diagnostics,
    )
    if resume:
        _require(
            artifact_dir.is_dir(),
            "--resume requires the explicit existing attempt directory",
            FileNotFoundError,
        )
        _require(
            _read_json(artifact_dir / "task_state.json").get("status")
            != "COMPLETE_DIAGNOSTIC",
            "completed evidence cannot be reopened for measurement",
        )
        _require(
            yaml.safe_load(
                (artifact_dir / "run_config.yaml").read_text(encoding="utf-8")
            )
            == config,
            "saved attempt config differs from the explicit registered task",
        )
        preflight = _load_torch(artifact_dir / "preflight.pt")
        control_tensors = _load_torch(artifact_dir / "preflight_controls.pt")
        source_binding = _read_json(artifact_dir / "projection_source_binding.json")
        _require(
            source_binding.get("task_id") == TASK_ID
            and source_binding.get("source_artifact_dir")
            == str(selected_projection_source)
            and source_binding.get("source_review_path")
            == str(selected_projection_review)
            and source_binding.get("training_preflight_path")
            == str(selected_training_preflight)
            and source_binding.get("gradient_diagnostics_path")
            == str(selected_gradient_diagnostics)
            and source_binding.get("population_role")
            in {"consumed_train_diagnostic", "consumed_generation_diagnostic"}
            and source_binding.get("population_rows") == POPULATION_ROWS,
            "explicit recovery P1 source directory or population binding changed",
        )
        receipt = PreflightReceipt.from_mapping(
            _read_json(artifact_dir / "preflight_ready.json")
        )
        identity = ResearchRunIdentity.from_mapping(
            _read_json(artifact_dir / "run_identity.json")
        )
        control = ResearchRunControl(identity, authorization_id=AUTHORIZATION_ID)
        control.mark_preflight_ready(receipt)
        control.resume_scientific_attempt(
            artifact_dir, identity.scientific_attempt_id or RUN_ID, receipt
        )
        _require(
            preflight.get("status") == "PASSED"
            and control_tensors.get("task_id") == TASK_ID,
            "saved preflight receipt or controls cannot authorize recovery",
        )
    else:
        _require(
            not artifact_dir.exists(),
            f"refusing to replace an existing attempt: {artifact_dir}",
            FileExistsError,
        )
        preflight, control_tensors = _preflight(config, assets, device)
        optimizer_audit.require_zero("frozen measurement preflight")
        preflight["optimizer_events_observed"] = optimizer_audit.snapshot()
        identity = _make_identity(config)
        control = ResearchRunControl(identity, authorization_id=AUTHORIZATION_ID)
        receipt = _make_preflight_receipt(
            control,
            correctness={
                "passed": preflight["status"] == "PASSED",
                "status": "PASSED",
                "population_and_endpoint_bound": True,
                "direct_ols_reinjection_passed": preflight["f1_reinjection"]["passed"],
                "direct_ols_cross_grid_passed": preflight["direct_ols_cross_grid"][
                    "passed"
                ],
                "fullpath_training_no_update_passed": preflight["training_path"][
                    "status"
                ]
                == "PASSED",
            },
            resource={
                "passed": True,
                "status": "PASSED",
                "cuda_available": True,
                "production_batch_completed": True,
                "peak_memory_bytes_observation": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            },
            implementation={
                "passed": True,
                "status": "PASSED",
                "selected_engine": "frozen_Gaussian_OLS_and_fullpath_CUDA_2048",
                **optimizer_audit.snapshot(),
            },
        )
        control.mark_preflight_ready(receipt)
        control.open_scientific_attempt(artifact_dir, f"{TASK_ID}:attempt_01", receipt)
        (artifact_dir / "run_config.yaml").write_text(
            yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        _write_json(
            artifact_dir / "run_identity.json",
            control.identity.to_dict(),
            exclusive=True,
        )
        _write_json(
            artifact_dir / "preflight_ready.json", receipt.to_dict(), exclusive=True
        )
        _write_json(
            artifact_dir / "projection_source_binding.json",
            {
                "task_id": TASK_ID,
                "source_artifact_dir": str(selected_projection_source),
                "source_review_path": str(selected_projection_review),
                "training_preflight_path": str(selected_training_preflight),
                "gradient_diagnostics_path": str(selected_gradient_diagnostics),
                "source_task_id": alignment.TASK_ID,
                "population_role": "consumed_generation_diagnostic",
                "population_rows": POPULATION_ROWS,
                "endpoint": "P1_E1G1",
                "endpoint_global_step": 52_000,
                "source_manifest_validated": True,
            },
            exclusive=True,
        )
        _save_torch(artifact_dir / "preflight.pt", preflight)
        control_tensors = {
            "task_id": TASK_ID,
            "direct_2048": control_tensors["direct_2048"],
            "direct_4096": control_tensors["direct_4096"],
            "training": control_tensors["training"],
        }
        _save_torch(artifact_dir / "preflight_controls.pt", control_tensors)
    _write_json(
        artifact_dir / "task_state.json",
        {
            "task_id": TASK_ID,
            "status": "MEASURING",
            "direct_ols_rows": POPULATION_ROWS,
            "optimizer_instances": 0,
            "optimizer_updates": 0,
            "official_test_reads": 0,
        },
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _measure_direct_population(artifact_dir, assets, control_tensors, device)
    direct = _load_direct_population(artifact_dir, assets)
    result, rows, readouts = _analyze(assets, direct)
    optimizer_audit.require_zero("frozen population measurement")
    _write_json(artifact_dir / "final_decision.json", result)
    _write_json(
        artifact_dir / "paper_training_inference_alignment.json",
        _paper_alignment_record(),
    )
    _write_csv(artifact_dir / "per_sample_results.csv", rows)
    _save_torch(artifact_dir / "condition_readouts.pt", readouts)
    _write_plot(artifact_dir / "condition_comparison.png", result)
    _write_json(
        artifact_dir / "measurement_status.json",
        {
            "task_id": TASK_ID,
            "status": "COMPLETE_DIAGNOSTIC",
            "decision": result["decision"],
            "production_rows": POPULATION_ROWS,
            "direct_ols_2048_rows": POPULATION_ROWS,
            "direct_ols_reinjection_rows": 20,
            "direct_ols_4096_reference_rows": 20,
            "training_no_update_rows": BATCH_SIZE,
            **optimizer_audit.snapshot(),
            "prior_refits": 0,
            "prior_samples": 0,
            "official_test_reads": 0,
        },
    )
    _write_closeout(
        artifact_dir / "closeout_report.md", result, preflight, reviewed=False,
        optimizer_events=optimizer_audit.snapshot(),
    )
    control.complete()
    _write_json(
        artifact_dir / "task_state.json",
        {
            "task_id": TASK_ID,
            "status": "COMPLETE_DIAGNOSTIC",
            "lifecycle_state": control.state,
            "decision": result["decision"],
            "completed_direct_ols_rows": POPULATION_ROWS,
            **optimizer_audit.snapshot(),
            "official_test_reads": 0,
        },
    )


def _exact_tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
            left.detach().cpu(), right.detach().cpu()
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _exact_tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _exact_tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _independent_tree_close(
    left: Any,
    right: Any,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, float]:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, math.inf
        a = left.detach().cpu().numpy()
        b = right.detach().cpu().numpy()
        if np.issubdtype(a.dtype, np.inexact):
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                return False, math.inf
            delta = float(np.max(np.abs(a - b))) if a.size else 0.0
            return bool(np.allclose(a, b, rtol=rtol, atol=atol)), delta
        equal = bool(np.array_equal(a, b))
        return equal, 0.0 if equal else math.inf
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False, math.inf
        values = [
            _independent_tree_close(left[key], right[key], rtol=rtol, atol=atol)
            for key in left
        ]
        return all(item[0] for item in values), max(
            (item[1] for item in values), default=0.0
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            return False, math.inf
        values = [
            _independent_tree_close(a, b, rtol=rtol, atol=atol)
            for a, b in zip(left, right, strict=True)
        ]
        return all(item[0] for item in values), max(
            (item[1] for item in values), default=0.0
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) or isinstance(right, float):
            close = math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)
            return close, abs(float(left) - float(right))
    return left == right, 0.0 if left == right else math.inf


def _repair_inputs(
    repair_config: Mapping[str, Any], base_config_path: Path, device: torch.device
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    base_config = _read_config(base_config_path)
    assets = _load_assets(base_config, device)
    sources = repair_config["sources"]
    checkpoint_path = _path(sources["terminal_checkpoint"]).resolve()
    schedule_path = _path(sources["training_schedule"]).resolve()
    _require(checkpoint_path.is_file() and schedule_path.is_file(),
             "explicit terminal checkpoint or original training schedule is missing",
             FileNotFoundError)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    registered = _terminal_checkpoint(assets)
    _require(
        checkpoint.get("run_id") == sources["expected_training_run_id"]
        and checkpoint.get("completed_global_step")
        == sources["expected_endpoint_global_step"]
        and checkpoint.get("optimizer_steps")
        == {
            "encoder_adam_step": sources["expected_encoder_adam_step"],
            "generator_adam_step": sources["expected_generator_adam_step"],
        }
        and recent._tree_max_delta(checkpoint, registered) == 0.0,
        "explicit E1G1 terminal checkpoint differs from the registered P1 endpoint",
    )
    schedule = torch.load(schedule_path, map_location="cpu", weights_only=False)
    runtime = assets["runtime"]
    _require(
        torch.is_tensor(schedule)
        and schedule.dtype == torch.int64
        and tuple(schedule.shape) == (2_000, BATCH_SIZE)
        and torch.equal(schedule, checkpoint.get("schedule")),
        "original 2,000-update TRAIN schedule differs from the terminal checkpoint",
    )
    rows = schedule[:4].long().clone()
    labels = runtime["labels"].long().cpu()
    train_indices = runtime["train_indices"].long().cpu()
    _require(
        torch.isin(rows, train_indices).all().item()
        and all(
            torch.equal(labels.index_select(0, row), torch.arange(BATCH_SIZE))
            and row.unique().numel() == BATCH_SIZE
            for row in rows
        ),
        "the fixed original schedule's first four batches are not class-balanced unique TRAIN rows",
    )
    assets["repair_checkpoint"] = checkpoint
    assets["repair_checkpoint_path"] = checkpoint_path
    assets["repair_schedule_path"] = schedule_path
    assets["repair_schedule_rows"] = rows
    assets["repair_base_config"] = base_config
    return assets, checkpoint, rows


def _write_engineering_update_log(
    artifact_dir: Path, calls: Sequence[Mapping[str, Any]], *, status: str
) -> None:
    _write_json(
        artifact_dir / "engineering_update_log.json",
        {
            "schema": "lightgen_fullpath_engineering_update_calls.v1",
            "task_id": REPAIR_TASK_ID,
            "status": status,
            "expected_update_calls": 5,
            "calls": [dict(item) for item in calls],
        },
    )


def _repair_call_start(
    checkpoint: Mapping[str, Any], rows: torch.Tensor, *, call_number: int,
    branch: str, schedule_row: int,
) -> dict[str, Any]:
    start_step = int(checkpoint["completed_global_step"])
    optimizer_steps = checkpoint["optimizer_steps"]
    return {
        "call_number": call_number,
        "branch": branch,
        "status": "STARTED",
        "schedule_row_index": schedule_row,
        "forward_global_step": start_step + schedule_row,
        "completed_global_step": start_step + schedule_row + 1,
        "sample_rows": [int(value) for value in rows.tolist()],
        "source_checkpoint_run_id": checkpoint["run_id"],
        "optimizer_steps_before": {
            "encoder": int(optimizer_steps["encoder_adam_step"]) + schedule_row,
            "generator": int(optimizer_steps["generator_adam_step"]) + schedule_row,
        },
    }


def _run_detached_cmos_negative_control(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pixels: torch.Tensor,
    rows: torch.Tensor,
    device: torch.device,
    source: dict[str, Any],
    audit: _OptimizerEventAudit,
) -> dict[str, Any]:
    encoder, generator = recent.base._phase_groups(model)
    calls_before = audit.optimizer_updates
    original_loss_parts = joint.decoder_ref._loss_parts
    cases: dict[str, Any] = {}
    expected_rejections = {
        "detach_rec": "reconstruction CMOS loss is detached",
        "detach_ssim": "SSIM CMOS loss is detached",
        "detach_rec_and_ssim": "reconstruction CMOS loss is detached",
    }
    for fault, expected_message in expected_rejections.items():
        model_before = recent.base._state_snapshot(model)
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        observed: dict[str, float] = {}

        def detached_producer(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
            parts, ssim, native_gradient, result = original_loss_parts(*args, **kwargs)
            detached = dict(parts)
            if fault in {"detach_rec", "detach_rec_and_ssim"}:
                detached["rec"] = detached["rec"].detach()
            if fault in {"detach_ssim", "detach_rec_and_ssim"}:
                ssim = ssim.detach()
            objective = joint.decoder_ref._objective(
                detached,
                ssim,
                None,
                source["checkpoint"]["weights"],
                float(source["checkpoint"]["structural_weight"]),
                None,
            )
            gradients = torch.autograd.grad(
                objective, (encoder, generator), retain_graph=True, allow_unused=True
            )
            observed["remaining_objective_gradient_norm"] = sum(
                0.0
                if gradient is None
                else float(torch.linalg.vector_norm(gradient.detach().double()))
                for gradient in gradients
            )
            return detached, ssim, native_gradient, result

        joint.decoder_ref._loss_parts = detached_producer
        message = ""
        try:
            joint._joint_update(model, optimizer, pixels, rows, device, source)
        except RuntimeError as error:
            message = str(error)
        finally:
            joint.decoder_ref._loss_parts = original_loss_parts
        unchanged = (
            _exact_tree_equal(model_before, recent.base._state_snapshot(model))
            and _exact_tree_equal(optimizer_before, optimizer.state_dict())
        )
        _require(
            expected_message in message
            and audit.optimizer_updates == calls_before
            and unchanged,
            f"{fault} CMOS negative control was not rejected before Adam.step with model/Adam unchanged",
        )
        case = {
            "status": "REJECTED_BEFORE_OPTIMIZER_STEP",
            "injected_fault": fault,
            "remaining_objective_gradient_norm": observed.get(
                "remaining_objective_gradient_norm", 0.0
            ),
            "optimizer_update_calls": 0,
            "model_unchanged": unchanged,
            "adam_unchanged": unchanged,
            "rejection": message,
        }
        if fault == "detach_rec_and_ssim":
            case["non_cmos_total_gradient_norm"] = case[
                "remaining_objective_gradient_norm"
            ]
            _require(
                case["non_cmos_total_gradient_norm"] > 0.0,
                "joint CMOS negative control must retain nonzero non-CMOS gradients",
            )
        cases[fault] = case
    return {
        "status": "ALL_REJECTED_BEFORE_OPTIMIZER_STEP",
        "cases": cases,
    }


def _reused_response_finite_difference_record(
    repair_config: Mapping[str, Any],
) -> dict[str, Any]:
    projection_dir = _path(
        repair_config["sources"]["terminal_projection_backup"]
    ).resolve()
    preflight = _load_torch(projection_dir / "production_preflight_revalidation.pt")
    preflight_config = alignment._read_config(
        _path(
            _read_config(_path(repair_config["sources"]["base_config"]))[
                "sources"]["terminal_projection_config"]
            )
        )
    measurement = preflight_config["measurement"]
    expected_fd = repair_config["evidence_consumer"]
    response = preflight.get("response_build", {})
    source = inspect.getsource(alignment._build_response)
    _require(
        preflight.get("task_id") == alignment.TASK_ID
        and preflight.get("status") == "PASSED"
        and preflight.get("endpoint") == "E1G1"
        and preflight.get("endpoint_global_step")
        == expected_fd["response_finite_difference_endpoint_global_step"]
        and measurement["response_fd_epsilon"]
        == expected_fd["response_finite_difference_epsilon"]
        and response.get("response_derivative_tolerance")
        == expected_fd["response_finite_difference_tolerance"]
        and math.isfinite(float(response.get("response_derivative_relative_error", math.inf)))
        and float(response["response_derivative_relative_error"])
        <= expected_fd["response_finite_difference_tolerance"]
        and float(response.get("terminal_model_state_max_delta", math.inf)) == 0.0
        and "manual_seed(202611021050)" in source
        and "epsilon = 2.0e-4" in source
        and "analytical = response @ direction.reshape(-1).to(torch.complex128)" in source
        and "(plus - minus) / (2.0 * epsilon)" in source,
        "saved terminal response finite-difference metadata is missing or inconsistent",
    )
    return {
        "status": "PASSED_REUSED_SAVED_PRODUCER_RECORD",
        "endpoint": preflight["endpoint"],
        "endpoint_global_step": preflight["endpoint_global_step"],
        "grid_size": 2048,
        "probe_batch_size": 1,
        "probe": "uniform amplitude 0.5 plus/minus a seeded Rademacher 200x200 direction",
        "direction_seed": expected_fd["response_finite_difference_seed"],
        "epsilon_amplitude": expected_fd["response_finite_difference_epsilon"],
        "analytic_direction": "saved terminal response matrix @ same seeded direction",
        "finite_difference_direction": "central difference of forward_from_image((center +/- epsilon * direction)^2)",
        "relative_error": float(response["response_derivative_relative_error"]),
        "relative_error_tolerance": expected_fd["response_finite_difference_tolerance"],
        "model_restoration_max_abs_delta": float(response["terminal_model_state_max_delta"]),
        "model_restored": True,
        "optimizer_events": {
            "historical_reported_instances": response.get("optimizer_instances"),
            "historical_reported_updates": response.get("optimizer_updates"),
            "historical_count_instrumentation": "not instrumented; neither historical optimizer field is accepted as an observed event count",
        },
        "perturbed_vectors_persisted": False,
        "review_limit": "saved derivative scalar, endpoint, fixed probe definition and model restoration were rechecked; the plus/minus vector outputs were not archived",
    }


def _measure_p1_reinjection(
    assets: Mapping[str, Any], device: torch.device,
) -> dict[str, Any]:
    runtime = assets["runtime"]
    p1 = assets["p1"]
    model = _new_direct_model(assets, 2048, device)
    indices = CONTROL_ROWS.clone()
    coefficients = p1["coefficients"].index_select(0, indices).detach().cpu().to(
        torch.complex128
    )
    with torch.inference_mode():
        measured = projected._forward_target(
            model, coefficients.to(device=device, dtype=torch.complex128)
        )
    raw = measured["raw_cmos"].detach().cpu().double()
    w_pixel = measured["w_per_pixel"].detach().cpu().double()
    target_raw = p1["raw_cmos"].index_select(0, indices).detach().cpu().double()
    area = float(runtime["camera_area"])
    target_w_pixel = target_raw * area
    raw_error = float((raw - target_raw).abs().max())
    w_pixel_error = float((w_pixel - target_w_pixel).abs().max())
    _require(
        raw_error <= REINJECTION_TOLERANCE
        and w_pixel_error <= REINJECTION_TOLERANCE,
        "P1 actual-OLS reinjection did not reproduce saved raw CMOS and W/pixel outputs",
    )
    return {
        "schema": "lightgen_p1_actual_ols_reinjection_controls.v1",
        "task_id": REPAIR_TASK_ID,
        "population_role": "consumed_generation_diagnostic",
        "condition": "P1_E1G1",
        "endpoint_source_run_id": _terminal_checkpoint(assets)["run_id"],
        "endpoint_global_step": 52_000,
        "encoder_adam_step": 47_000,
        "generator_adam_step": 52_000,
        "grid_size": 2048,
        "population_indices": indices,
        "labels": p1["labels"].index_select(0, indices).cpu(),
        "proposal_ids": p1["proposal_ids"].index_select(0, indices).cpu(),
        "actual_slm_commands_forwarded": p1["commands"].index_select(0, indices).cpu(),
        "input_ols_coefficients": coefficients,
        "raw_cmos": raw,
        "w_per_pixel": w_pixel,
        "target_raw_cmos": target_raw,
        "target_w_per_pixel": target_w_pixel,
        "raw_cmos_unit": "W/m^2",
        "w_per_pixel_unit": "W/pixel",
        "camera_pixel_area_m2": area,
        "camera_gain_w_inverse": float(runtime["camera_gain"]),
        "ols_basis_id": model.runtime.ols.basis_id,
        "ols_transport": model.runtime.ols.mode_identity().get("transport"),
        "modal_coefficient_scale": "registered_native_complex_OLS_scale",
        "output_relay": "same registered Generator output relay",
        "raw_cmos_max_abs_error": raw_error,
        "w_per_pixel_max_abs_error": w_pixel_error,
        "tolerance": REINJECTION_TOLERANCE,
    }


def _repair_source_dict(
    checkpoint: Mapping[str, Any],
    assets: Mapping[str, Any],
    artifact_dir: Path,
    update_number: int,
) -> dict[str, Any]:
    runtime = assets["runtime"]
    return {
        "checkpoint": dict(checkpoint),
        "training_labels": runtime["labels"].long().cpu(),
        "training_indices": runtime["train_indices"].long().cpu(),
        "training_pixels": runtime["pixels"],
        "training_witness_dir": artifact_dir / "training_witnesses",
        "training_capture_updates": (1, 3, 4),
        "active_training_update": update_number,
    }


def _complete_repair_call(
    call: dict[str, Any],
    loss: Mapping[str, Any],
    before: tuple[int, int],
    after: tuple[int, int],
) -> None:
    call.update(
        {
            "status": "COMPLETED",
            "loss": dict(loss),
            "optimizer_steps_after": {
                "encoder": int(after[0]),
                "generator": int(after[1]),
            },
            "optimizer_step_delta": {
                "encoder": int(after[0] - before[0]),
                "generator": int(after[1] - before[1]),
            },
            "witness_is_pre_update_forward": True,
        }
    )


def _reconcile_saved_direct_fourth_call(
    artifact_dir: Path,
    calls: list[dict[str, Any]],
    checkpoint: Mapping[str, Any],
    schedule_rows: torch.Tensor,
) -> dict[str, Any]:
    _require(
        len(calls) == 4
        and calls[3].get("call_number") == 4
        and calls[3].get("branch") == "fourth_direct"
        and calls[3].get("schedule_row_index") == 3
        and calls[3].get("sample_rows") == schedule_rows[3].tolist(),
        "saved direct fourth update log is not bound to the fourth original TRAIN batch",
    )
    direct = _load_torch(artifact_dir / "engineering_direct_branch.pt")
    witness = _load_torch(
        artifact_dir / "training_witnesses" / "training_path_witness_update_00004.pt"
    )
    _require(
        direct.get("task_id") == REPAIR_TASK_ID
        and direct.get("branch") == "fourth_direct"
        and direct.get("diagnostic_only_no_candidate") is True
        and direct.get("source_checkpoint_run_id") == checkpoint["run_id"]
        and direct.get("source_global_step") == 52_000
        and direct.get("completed_global_step") == 52_004
        and direct.get("schedule_row_index") == 3
        and torch.equal(direct.get("rows"), schedule_rows[3])
        and direct.get("optimizer_steps") == (47_004, 52_004)
        and witness.get("source_training_run_id") == checkpoint["run_id"]
        and witness.get("completed_update") == 4
        and witness.get("forward_global_step") == 52_003
        and witness.get("completed_global_step") == 52_004
        and witness.get("optimizer_steps_before")
        == {"encoder": 47_003, "generator": 52_003}
        and witness.get("optimizer_steps_after")
        == {"encoder": 47_004, "generator": 52_004}
        and torch.equal(witness.get("source_rows"), schedule_rows[3]),
        "saved direct fourth branch or pre-update witness does not prove a completed fourth update",
    )
    loss = direct.get("loss")
    _require(
        isinstance(loss, Mapping)
        and {"dist", "rec", "op", "ssim_loss", "total"}.issubset(loss)
        and all(math.isfinite(float(value)) for value in loss.values()),
        "saved direct fourth branch has incomplete or nonfinite loss values",
    )
    completion_source = "already_completed_call_log"
    if calls[3].get("status") == "STARTED":
        _complete_repair_call(
            calls[3], loss, (47_003, 52_003), (47_004, 52_004)
        )
        calls[3]["completion_reconciled_from_saved_branch"] = True
        completion_source = "saved_direct_branch_and_pre_update_witness"
    else:
        _require(
            calls[3].get("status") == "COMPLETED"
            and calls[3].get("loss") == dict(loss)
            and calls[3].get("optimizer_steps_after")
            == {"encoder": 47_004, "generator": 52_004},
            "completed direct fourth call disagrees with its saved branch result",
        )
    return {
        "task_id": REPAIR_TASK_ID,
        "status": "PASSED",
        "call_number": 4,
        "schedule_row_index": 3,
        "source_rows": schedule_rows[3].tolist(),
        "optimizer_steps": [47_004, 52_004],
        "completion_source": completion_source,
        "saved_branch": "engineering_direct_branch.pt",
        "saved_witness": "training_witnesses/training_path_witness_update_00004.pt",
    }


def _run_repair_worker(
    repair_config: Mapping[str, Any],
    repair_config_path: Path,
    artifact_dir: Path,
) -> None:
    _require(torch.cuda.is_available(), "independent recovery worker requires AutoDL CUDA")
    device = torch.device("cuda")
    base_config_path = _path(repair_config["sources"]["base_config"]).resolve()
    worker_events = _OptimizerEventAudit()
    with worker_events:
        assets, checkpoint, schedule_rows = _repair_inputs(
            repair_config, base_config_path, device
        )
        worker_events.require_zero("independent recovery source loading")
        recovery = _load_torch(artifact_dir / "engineering_recovery_input.pt")
        _require(
            recovery.get("task_id") == REPAIR_TASK_ID
            and recovery.get("completed_updates") == 3
            and recovery.get("origin_checkpoint_run_id") == checkpoint["run_id"]
            and torch.equal(recovery.get("schedule_rows"), schedule_rows)
            and torch.equal(recovery.get("next_batch_rows"), schedule_rows[3]),
            "independent recovery payload does not match the original three-step state and fourth batch",
        )
        model, optimizer = recent._optimizer_and_model(
            checkpoint,
            assets["runtime"],
            expected_steps=(47_000, 52_000),
        )
        model.load_state_dict(recovery["model"], strict=True)
        optimizer.load_state_dict(copy.deepcopy(recovery["optimizer"]))
        encoder, generator = recent.base._phase_groups(model)
        _require(
            recent._optimizer_phase_steps(optimizer, encoder, generator)
            == (47_003, 52_003),
            "restored Adam moments are not mapped to the Encoder/Generator third-step state",
        )
        torch.random.set_rng_state(recovery["cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in recovery["cuda_rng"]])
        source = _repair_source_dict(checkpoint, assets, artifact_dir, 4)
        calls_record = _read_json(artifact_dir / "engineering_update_log.json")
        calls = calls_record.get("calls", [])
        direct_reconciliation = _reconcile_saved_direct_fourth_call(
            artifact_dir, calls, checkpoint, schedule_rows
        )
        if direct_reconciliation["completion_source"] != "already_completed_call_log":
            _write_json(
                artifact_dir / "direct_fourth_reconciliation.json",
                direct_reconciliation,
                exclusive=True,
            )
            _write_engineering_update_log(
                artifact_dir, calls, status="DIRECT_FOURTH_RECONCILED"
            )
        _require(
            calls_record.get("task_id") == REPAIR_TASK_ID
            and len(calls) == 4
            and all(call.get("status") == "COMPLETED" for call in calls)
            and calls[2].get("branch") == "three_continuous"
            and calls[3].get("branch") == "fourth_direct",
            "worker will only continue the explicit three-step snapshot after a completed direct fourth branch",
        )
        call = _repair_call_start(
            checkpoint,
            schedule_rows[3],
            call_number=5,
            branch="fourth_independent_restore",
            schedule_row=3,
        )
        call["optimizer_steps_before"] = {
            "encoder": 47_003,
            "generator": 52_003,
        }
        calls.append(call)
        _write_engineering_update_log(artifact_dir, calls, status="RECOVERING")
        steps_before = recent._optimizer_phase_steps(optimizer, encoder, generator)
        source["active_training_update"] = 4
        loss = joint._joint_update(
            model,
            optimizer,
            assets["runtime"]["pixels"],
            schedule_rows[3].to(device),
            device,
            source,
        )
        steps_after = recent._optimizer_phase_steps(optimizer, encoder, generator)
        _require(steps_after == (47_004, 52_004),
                 "independent fourth update did not advance both Adam step counters")
        _complete_repair_call(call, loss, steps_before, steps_after)
        recovery_result = {
            "task_id": REPAIR_TASK_ID,
            "branch": "fourth_independent_restore",
            "diagnostic_only_no_candidate": True,
            "completed_updates": 4,
            "source_checkpoint_run_id": checkpoint["run_id"],
            "source_global_step": 52_000,
            "completed_global_step": 52_004,
            "schedule_row_index": 3,
            "rows": schedule_rows[3].clone(),
            "loss": dict(loss),
            "model": recent.base._state_snapshot(model),
            "optimizer": cpu_copy(optimizer.state_dict()),
            "optimizer_steps": recent._optimizer_phase_steps(
                optimizer, encoder, generator
            ),
            "cpu_rng": torch.random.get_rng_state().cpu().clone(),
            "cuda_rng": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
        }
        _save_torch(
            artifact_dir / "engineering_recovery_branch.pt", recovery_result
        )
        _write_engineering_update_log(artifact_dir, calls, status="RECOVERED_BRANCH_COMPLETE")
        _write_json(
            artifact_dir / "recovery_worker_optimizer_events.json",
            worker_events.snapshot(),
            exclusive=True,
        )
        _write_json(
            artifact_dir / "recovery_worker_status.json",
            {
                "task_id": REPAIR_TASK_ID,
                "status": "PASSED",
                "completed_updates": 4,
                "update_calls": 1,
                "optimizer_events": worker_events.snapshot(),
            },
            exclusive=True,
        )


def _run_repair_evidence(
    repair_config: Mapping[str, Any],
    repair_config_path: Path,
    artifact_dir: Path,
) -> None:
    _require(
        artifact_dir.resolve() == REPAIR_OUTPUT_DIR.resolve(),
        f"repair execution must use the explicit attempt_02 output path: {REPAIR_OUTPUT_DIR}",
        ValueError,
    )
    _require(not artifact_dir.exists(),
             f"refusing to replace an existing repair attempt: {artifact_dir}",
             FileExistsError)
    _require(torch.cuda.is_available(), "repair execution requires AutoDL CUDA")
    base_config_path = _path(repair_config["sources"]["base_config"]).resolve()
    device = torch.device("cuda")
    optimizer_audit = _OptimizerEventAudit()
    with optimizer_audit:
        assets, checkpoint, schedule_rows = _repair_inputs(
            repair_config, base_config_path, device
        )
        optimizer_audit.require_zero("repair source and asset loading")
        training_evidence_scope = assets["training_evidence_scope"]
        # Do not create a partial attempt until all frozen source inputs and
        # endpoint identities have passed the read-only preflight.
        artifact_dir.mkdir(parents=True, exist_ok=False)
        (artifact_dir / "run_config.yaml").write_text(
            yaml.safe_dump(dict(repair_config), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        _write_json(
            artifact_dir / "training_evidence_scope.json",
            training_evidence_scope,
            exclusive=True,
        )
        _write_json(
            artifact_dir / "repair_execution_status.json",
            {
                "task_id": REPAIR_TASK_ID,
                "status": "RUNNING_ENGINEERING_PREFLIGHT",
                "scientific_updates": 0,
                "prior_refits": 0,
                "prior_samples": 0,
                "official_test_reads": 0,
            },
            exclusive=True,
        )
        runtime = assets["runtime"]
        frozen_before = recent._snapshot_frozen(
            runtime, assets["alignment"]["mapper"]
        )
        rollback_cpu_rng = torch.random.get_rng_state().cpu().clone()
        rollback_cuda_rng = [
            state.cpu().clone() for state in torch.cuda.get_rng_state_all()
        ]
        torch.manual_seed(ENGINEERING_SEED)
        torch.cuda.manual_seed_all(ENGINEERING_SEED)
        model, optimizer = recent._optimizer_and_model(
            checkpoint,
            runtime,
            expected_steps=(47_000, 52_000),
        )
        rollback_model_state = recent.base._state_snapshot(model)
        rollback_optimizer_state = cpu_copy(optimizer.state_dict())
        _require(
            _exact_tree_equal(rollback_model_state, checkpoint["model"])
            and _exact_tree_equal(rollback_optimizer_state, checkpoint["optimizer"]),
            "isolated engineering model/Adam did not start at the frozen terminal checkpoint",
        )
        encoder, generator = recent.base._phase_groups(model)
        source = _repair_source_dict(checkpoint, assets, artifact_dir, 1)
        negative_control = _run_detached_cmos_negative_control(
            model,
            optimizer,
            runtime["pixels"],
            schedule_rows[0].to(device),
            device,
            source,
            optimizer_audit,
        )
        _write_json(
            artifact_dir / "detached_cmos_negative_control.json",
            negative_control,
            exclusive=True,
        )
        calls: list[dict[str, Any]] = []
        _write_engineering_update_log(artifact_dir, calls, status="UPDATING")
        update_losses: list[dict[str, Any]] = []
        for index in range(3):
            source["active_training_update"] = index + 1
            before = recent._optimizer_phase_steps(optimizer, encoder, generator)
            call = _repair_call_start(
                checkpoint,
                schedule_rows[index],
                call_number=index + 1,
                branch="three_continuous",
                schedule_row=index,
            )
            call["optimizer_steps_before"] = {
                "encoder": before[0], "generator": before[1],
            }
            calls.append(call)
            _write_engineering_update_log(artifact_dir, calls, status="UPDATING")
            loss = joint._joint_update(
                model,
                optimizer,
                runtime["pixels"],
                schedule_rows[index].to(device),
                device,
                source,
            )
            after = recent._optimizer_phase_steps(optimizer, encoder, generator)
            _require(after == (47_000 + index + 1, 52_000 + index + 1),
                     f"continuous engineering update {index + 1} advanced the wrong Adam steps")
            _complete_repair_call(call, loss, before, after)
            update_losses.append(dict(loss))
            _write_engineering_update_log(artifact_dir, calls, status="UPDATING")

        _require(len(calls) == 3 and optimizer_audit.optimizer_updates == 3,
                 "three-step engineering segment did not make exactly three successful optimizer calls")
        recovery_input = {
            "task_id": REPAIR_TASK_ID,
            "diagnostic_only_no_candidate": True,
            "completed_updates": 3,
            "origin_checkpoint_run_id": checkpoint["run_id"],
            "origin_global_step": checkpoint["completed_global_step"],
            "completed_global_step": 52_003,
            "schedule_rows": schedule_rows.clone(),
            "next_batch_rows": schedule_rows[3].clone(),
            "model": recent.base._state_snapshot(model),
            "optimizer": cpu_copy(optimizer.state_dict()),
            "optimizer_steps": recent._optimizer_phase_steps(
                optimizer, encoder, generator
            ),
            "cpu_rng": torch.random.get_rng_state().cpu().clone(),
            "cuda_rng": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
        }
        _save_torch(
            artifact_dir / "engineering_recovery_input.pt", recovery_input
        )
        source["active_training_update"] = 4
        before = recent._optimizer_phase_steps(optimizer, encoder, generator)
        direct_call = _repair_call_start(
            checkpoint,
            schedule_rows[3],
            call_number=4,
            branch="fourth_direct",
            schedule_row=3,
        )
        direct_call["optimizer_steps_before"] = {
            "encoder": before[0], "generator": before[1],
        }
        calls.append(direct_call)
        _write_engineering_update_log(artifact_dir, calls, status="DIRECT_FOURTH")
        direct_loss = joint._joint_update(
            model,
            optimizer,
            runtime["pixels"],
            schedule_rows[3].to(device),
            device,
            source,
        )
        after = recent._optimizer_phase_steps(optimizer, encoder, generator)
        _require(after == (47_004, 52_004),
                 "direct fourth update advanced the wrong Encoder/Generator Adam counters")
        _complete_repair_call(direct_call, direct_loss, before, after)
        _write_engineering_update_log(
            artifact_dir, calls, status="DIRECT_FOURTH_COMPLETE"
        )
        direct_branch = {
            "task_id": REPAIR_TASK_ID,
            "branch": "fourth_direct",
            "diagnostic_only_no_candidate": True,
            "completed_updates": 4,
            "source_checkpoint_run_id": checkpoint["run_id"],
            "source_global_step": 52_000,
            "completed_global_step": 52_004,
            "schedule_row_index": 3,
            "rows": schedule_rows[3].clone(),
            "loss": dict(direct_loss),
            "model": recent.base._state_snapshot(model),
            "optimizer": cpu_copy(optimizer.state_dict()),
            "optimizer_steps": recent._optimizer_phase_steps(
                optimizer, encoder, generator
            ),
            "cpu_rng": torch.random.get_rng_state().cpu().clone(),
            "cuda_rng": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
        }
        _save_torch(artifact_dir / "engineering_direct_branch.pt", direct_branch)

        reinjection = _measure_p1_reinjection(assets, device)
        _save_torch(artifact_dir / "p1_actual_ols_reinjection.pt", reinjection)
        recent._assert_frozen_unchanged(
            frozen_before, runtime, assets["alignment"]["mapper"]
        )
        reloaded_checkpoint = torch.load(
            assets["repair_checkpoint_path"], map_location="cpu", weights_only=False
        )
        _require(
            recent._tree_max_delta(checkpoint, reloaded_checkpoint) == 0.0,
            "source terminal checkpoint or Adam state changed during isolated engineering updates",
        )
        _require(
            optimizer_audit.snapshot()
            == {"optimizer_instances": 1, "optimizer_updates": 4},
            "parent engineering process optimizer events differ from one instance and four update calls",
        )
        model.load_state_dict(rollback_model_state, strict=True)
        optimizer.load_state_dict(copy.deepcopy(rollback_optimizer_state))
        optimizer.zero_grad(set_to_none=True)
        model_rollback_matches = _exact_tree_equal(
            rollback_model_state, recent.base._state_snapshot(model)
        )
        adam_rollback_matches = _exact_tree_equal(
            rollback_optimizer_state, cpu_copy(optimizer.state_dict())
        ) and recent._optimizer_phase_steps(optimizer, encoder, generator) == (
            47_000, 52_000,
        )
        _require(
            model_rollback_matches and adam_rollback_matches
            and all(parameter.grad is None for parameter in model.parameters()),
            "isolated engineering updates did not roll back to the source model and Adam state",
        )
        torch.random.set_rng_state(rollback_cpu_rng)
        torch.cuda.set_rng_state_all(rollback_cuda_rng)
        cpu_rng_rollback_matches = torch.equal(
            torch.random.get_rng_state().cpu(), rollback_cpu_rng
        )
        cuda_rng_rollback_matches = _exact_tree_equal(
            [state.cpu() for state in torch.cuda.get_rng_state_all()],
            rollback_cuda_rng,
        )
        _require(
            cpu_rng_rollback_matches and cuda_rng_rollback_matches,
            "isolated engineering process did not restore the caller RNG state",
        )
        _write_json(
            artifact_dir / "engineering_rollback_status.json",
            {
                "task_id": REPAIR_TASK_ID,
                "status": "ROLLED_BACK_TO_SOURCE_ENDPOINT",
                "source_training_run_id": checkpoint["run_id"],
                "source_global_step": 52_000,
                "source_optimizer_steps": [47_000, 52_000],
                "model_restored": model_rollback_matches,
                "adam_restored": adam_rollback_matches,
                "phase_gradients_cleared": all(
                    parameter.grad is None for parameter in model.parameters()
                ),
                "cpu_rng_restored": cpu_rng_rollback_matches,
                "cuda_rng_restored": cuda_rng_rollback_matches,
                "source_checkpoint_unchanged": True,
                "frozen_assets_unchanged": True,
                "isolated_states_are_not_candidates": True,
            },
            exclusive=True,
        )
        del optimizer, model, assets, runtime, encoder, generator, source
        del recovery_input, direct_branch, frozen_before, reloaded_checkpoint
        del rollback_model_state, rollback_optimizer_state
        del rollback_cpu_rng, rollback_cuda_rng
        torch.cuda.empty_cache()

        command = [
            sys.executable,
            "-m",
            "scripts.research.lightgen.diagnose_lightgen_fullpath_evidence_closure_v1",
            "--config",
            str(repair_config_path.resolve()),
            "--artifact-dir",
            str(artifact_dir.resolve()),
            "--repair-worker",
        ]
        worker = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        (artifact_dir / "recovery_worker_stdout.txt").write_text(
            worker.stdout, encoding="utf-8"
        )
        (artifact_dir / "recovery_worker_stderr.txt").write_text(
            worker.stderr, encoding="utf-8"
        )
        _require(worker.returncode == 0,
                 f"independent-process recovery worker failed with exit {worker.returncode}")
        recovery_branch = _load_torch(artifact_dir / "engineering_recovery_branch.pt")
        worker_events = _read_json(
            artifact_dir / "recovery_worker_optimizer_events.json"
        )
        _require(worker_events == {"optimizer_instances": 1, "optimizer_updates": 1},
                 "independent recovery process did not report one optimizer and one successful step")
        direct = _load_torch(artifact_dir / "engineering_direct_branch.pt")
        checks = {
            "next_batch_identity": torch.equal(direct["rows"], recovery_branch["rows"])
            and torch.equal(direct["rows"], schedule_rows[3]),
            "model_state_within_recovery_tolerance": recent._states_close(
                direct["model"], recovery_branch["model"]
            ),
            "Adam_state_within_recovery_tolerance": recent._states_close(
                direct["optimizer"], recovery_branch["optimizer"]
            ),
            "loss_within_recovery_tolerance": all(
                math.isclose(
                    float(direct["loss"][name]),
                    float(recovery_branch["loss"][name]),
                    rel_tol=ENGINEERING_RECOVERY_RTOL,
                    abs_tol=ENGINEERING_RECOVERY_ATOL,
                )
                for name in direct["loss"]
            ),
            "Adam_steps_identical": direct["optimizer_steps"]
            == recovery_branch["optimizer_steps"]
            == (47_004, 52_004),
            "RNG_state_identical": _exact_tree_equal(
                {
                    "cpu": direct["cpu_rng"],
                    "cuda": direct["cuda_rng"],
                },
                {
                    "cpu": recovery_branch["cpu_rng"],
                    "cuda": recovery_branch["cuda_rng"],
                },
            ),
        }
        _require(all(checks.values()),
                 "direct and independently restored fourth updates differ beyond the existing recovery tolerance")
        _write_json(
            artifact_dir / "engineering_recovery_comparison.json",
            {
                "task_id": REPAIR_TASK_ID,
                "passed": True,
                "checks": checks,
                "recovery_rtol": ENGINEERING_RECOVERY_RTOL,
                "recovery_atol": ENGINEERING_RECOVERY_ATOL,
                "direct_optimizer_events": optimizer_audit.snapshot(),
                "independent_process_optimizer_events": worker_events,
            },
            exclusive=True,
        )
        finite_difference = _reused_response_finite_difference_record(repair_config)
        _write_json(
            artifact_dir / "reused_response_finite_difference.json",
            finite_difference,
            exclusive=True,
        )
        calls_record = _read_json(artifact_dir / "engineering_update_log.json")
        _require(
            len(calls_record.get("calls", [])) == 5
            and all(call.get("status") == "COMPLETED" for call in calls_record["calls"])
            and [call["branch"] for call in calls_record["calls"]]
            == [
                "three_continuous",
                "three_continuous",
                "three_continuous",
                "fourth_direct",
                "fourth_independent_restore",
            ],
            "engineering update call log is incomplete, reordered, or duplicated",
        )
        total_events = {
            "optimizer_instances": optimizer_audit.optimizer_instances
            + worker_events["optimizer_instances"],
            "optimizer_updates": optimizer_audit.optimizer_updates
            + worker_events["optimizer_updates"],
        }
        _write_json(
            artifact_dir / "optimizer_events.json",
            {
                "parent_process": optimizer_audit.snapshot(),
                "independent_recovery_process": worker_events,
                "total": total_events,
                "read_only_measurement_and_review_expected": {
                    "optimizer_instances": 0,
                    "optimizer_updates": 0,
                },
                "scientific_updates": 0,
            },
            exclusive=True,
        )
        _write_json(
            artifact_dir / "repair_execution_status.json",
            {
                "task_id": REPAIR_TASK_ID,
                "status": "ENGINEERING_REPAIR_COMPLETE",
                "source_training_run_id": checkpoint["run_id"],
                "source_global_step": 52_000,
                "source_optimizer_steps": {
                    "encoder": 47_000,
                    "generator": 52_000,
                },
                "engineering_update_calls": 5,
                "optimizer_events": total_events,
                "negative_control_passed": True,
                "recovery_comparison_passed": True,
                "frozen_assets_unchanged": True,
                "source_checkpoint_unchanged": True,
                "isolated_engineering_states_rolled_back": True,
                "isolated_engineering_states_are_not_candidates": True,
                "training_evidence_scope": training_evidence_scope,
                "scientific_updates": 0,
                "prior_refits": 0,
                "prior_samples": 0,
                "official_test_reads": 0,
            },
            exclusive=True,
        )


def _finalize_repair_after_worker_recovery(
    repair_config: Mapping[str, Any], artifact_dir: Path
) -> None:
    _require(
        artifact_dir.resolve() == REPAIR_OUTPUT_DIR.resolve()
        and artifact_dir.is_dir(),
        "repair finalization requires the explicit existing attempt_02 directory",
        ValueError,
    )
    saved_config = yaml.safe_load(
        (artifact_dir / "run_config.yaml").read_text(encoding="utf-8")
    )
    execution = _read_json(artifact_dir / "repair_execution_status.json")
    parent_log = (artifact_dir / "parent_run_failure.log").read_text(encoding="utf-8")
    failed_worker_log = (artifact_dir / "recovery_worker_stderr.txt").read_text(
        encoding="utf-8"
    )
    _require(
        saved_config == repair_config
        and execution.get("task_id") == REPAIR_TASK_ID
        and execution.get("status") == "RUNNING_ENGINEERING_PREFLIGHT"
        and "independent-process recovery worker failed with exit 1" in parent_log
        and "selected endpoint Adam state differs: expected (47003, 52003), got (47000, 52000)"
        in failed_worker_log,
        "repair finalizer requires the preserved attempt_02 worker-step failure and unchanged run config",
    )
    update_log = _read_json(artifact_dir / "engineering_update_log.json")
    calls = update_log.get("calls", [])
    _require(
        update_log.get("task_id") == REPAIR_TASK_ID
        and update_log.get("status") == "RECOVERED_BRANCH_COMPLETE"
        and len(calls) == 5
        and all(call.get("status") == "COMPLETED" for call in calls)
        and [call.get("call_number") for call in calls] == [1, 2, 3, 4, 5]
        and [call.get("branch") for call in calls]
        == [
            "three_continuous", "three_continuous", "three_continuous",
            "fourth_direct", "fourth_independent_restore",
        ],
        "repair finalizer requires the five completed registered calls",
    )
    direct = _load_torch(artifact_dir / "engineering_direct_branch.pt")
    restored = _load_torch(artifact_dir / "engineering_recovery_branch.pt")
    recovery_input = _load_torch(artifact_dir / "engineering_recovery_input.pt")
    _require(
        direct.get("task_id") == REPAIR_TASK_ID
        and restored.get("task_id") == REPAIR_TASK_ID
        and direct.get("diagnostic_only_no_candidate") is True
        and restored.get("diagnostic_only_no_candidate") is True
        and recovery_input.get("diagnostic_only_no_candidate") is True
        and direct.get("source_checkpoint_run_id")
        == repair_config["sources"]["expected_training_run_id"]
        and restored.get("source_checkpoint_run_id")
        == repair_config["sources"]["expected_training_run_id"]
        and direct.get("completed_global_step") == restored.get("completed_global_step")
        == 52_004
        and recovery_input.get("completed_global_step") == 52_003
        and torch.equal(direct.get("rows"), restored.get("rows"))
        and direct.get("rows").tolist() == calls[3].get("sample_rows")
        and direct.get("optimizer_steps") == restored.get("optimizer_steps")
        == (47_004, 52_004),
        "repair finalizer rejects a recovery branch with a different endpoint, batch, or Adam step",
    )
    checks = {
        "next_batch_identity": torch.equal(direct["rows"], restored["rows"]),
        "model_state_within_recovery_tolerance": recent._states_close(
            direct["model"], restored["model"]
        ),
        "Adam_state_within_recovery_tolerance": recent._states_close(
            direct["optimizer"], restored["optimizer"]
        ),
        "loss_within_recovery_tolerance": all(
            math.isclose(
                float(direct["loss"][name]),
                float(restored["loss"][name]),
                rel_tol=ENGINEERING_RECOVERY_RTOL,
                abs_tol=ENGINEERING_RECOVERY_ATOL,
            )
            for name in direct["loss"]
        ),
        "Adam_steps_identical": direct["optimizer_steps"]
        == restored["optimizer_steps"]
        == (47_004, 52_004),
        "RNG_state_identical": _exact_tree_equal(
            {"cpu": direct["cpu_rng"], "cuda": direct["cuda_rng"]},
            {"cpu": restored["cpu_rng"], "cuda": restored["cuda_rng"]},
        ),
    }
    _require(all(checks.values()),
             "repaired independent recovery branch differs from the direct fourth update")
    rollback = _read_json(artifact_dir / "engineering_rollback_status.json")
    _require(
        rollback.get("status") == "ROLLED_BACK_TO_SOURCE_ENDPOINT"
        and rollback.get("source_global_step") == 52_000
        and rollback.get("source_optimizer_steps") == [47_000, 52_000]
        and rollback.get("model_restored") is True
        and rollback.get("adam_restored") is True
        and rollback.get("cpu_rng_restored") is True
        and rollback.get("cuda_rng_restored") is True
        and rollback.get("source_checkpoint_unchanged") is True
        and rollback.get("frozen_assets_unchanged") is True,
        "repair finalizer rejects an incomplete source rollback record",
    )
    worker_events = _read_json(
        artifact_dir / "recovery_worker_optimizer_events.json"
    )
    _require(worker_events == {"optimizer_instances": 1, "optimizer_updates": 1},
             "successful recovery process must record one instance and one update")
    recovery_comparison = {
        "task_id": REPAIR_TASK_ID,
        "passed": True,
        "checks": checks,
        "recovery_rtol": ENGINEERING_RECOVERY_RTOL,
        "recovery_atol": ENGINEERING_RECOVERY_ATOL,
        "direct_optimizer_events": {"optimizer_instances": 1, "optimizer_updates": 4},
        "independent_process_optimizer_events": worker_events,
    }
    _write_json(
        artifact_dir / "engineering_recovery_comparison.json",
        recovery_comparison,
        exclusive=True,
    )
    _write_json(
        artifact_dir / "reused_response_finite_difference.json",
        _reused_response_finite_difference_record(repair_config),
        exclusive=True,
    )
    failed_worker_events = {
        "optimizer_instances": 1,
        "optimizer_updates": 0,
        "status": "FAILED_BEFORE_UPDATE",
        "basis": "preserved traceback reaches the post-constructor Adam step-count assertion; the process failed before _joint_update and did not persist its event snapshot",
        "trace_file": "recovery_worker_stderr.txt",
    }
    parent_events = {"optimizer_instances": 1, "optimizer_updates": 4}
    total_events = {
        "optimizer_instances": 3,
        "optimizer_updates": 5,
    }
    reconciliation = {
        "task_id": REPAIR_TASK_ID,
        "status": "RECONCILED_AFTER_WORKER_HARNESS_FAILURE",
        "parent_process": {
            **parent_events,
            "basis": "parent reached its worker-return error check after the in-process optimizer counter assertion and source rollback",
        },
        "failed_recovery_process": failed_worker_events,
        "successful_recovery_process": worker_events,
        "total": total_events,
    }
    _write_json(
        artifact_dir / "optimizer_event_reconciliation.json",
        reconciliation,
        exclusive=True,
    )
    _write_json(
        artifact_dir / "optimizer_events.json",
        {
            "parent_process": parent_events,
            "failed_recovery_process": failed_worker_events,
            "independent_recovery_process": worker_events,
            "total": total_events,
            "recovery_worker_failure_reconciled": True,
            "read_only_measurement_and_review_expected": {
                "optimizer_instances": 0,
                "optimizer_updates": 0,
            },
            "scientific_updates": 0,
        },
        exclusive=True,
    )
    _write_json(
        artifact_dir / "repair_execution_status.json",
        {
            "task_id": REPAIR_TASK_ID,
            "status": "ENGINEERING_REPAIR_COMPLETE",
            "source_training_run_id": repair_config["sources"]["expected_training_run_id"],
            "source_global_step": 52_000,
            "source_optimizer_steps": {"encoder": 47_000, "generator": 52_000},
            "engineering_update_calls": 5,
            "optimizer_events": total_events,
            "optimizer_event_reconciled_after_worker_failure": True,
            "negative_control_passed": True,
            "recovery_comparison_passed": True,
            "frozen_assets_unchanged": True,
            "source_checkpoint_unchanged": True,
            "isolated_engineering_states_rolled_back": True,
            "isolated_engineering_states_are_not_candidates": True,
            "training_evidence_scope": _read_json(
                artifact_dir / "training_evidence_scope.json"
            ),
            "scientific_updates": 0,
            "prior_refits": 0,
            "prior_samples": 0,
            "official_test_reads": 0,
        },
    )


def _numpy_readout(
    raw_cmos: np.ndarray, camera_area: float, camera_gain: float
) -> np.ndarray:
    raw = np.asarray(raw_cmos, dtype=np.float64)
    if raw.ndim != 4 or raw.shape[1:] != (1, 200, 200) or not np.isfinite(raw).all():
        raise ValueError("raw CMOS population has an invalid shape or nonfinite pixels")
    gained = np.clip(raw * float(camera_area) * float(camera_gain), 0.0, 1.0)[:, 0]
    result = np.empty((len(raw), 1, 28, 28), dtype=np.float32)
    for out_y in range(28):
        y0, y1 = (out_y * 200) // 28, ((out_y + 1) * 200 + 27) // 28
        for out_x in range(28):
            x0, x1 = (out_x * 200) // 28, ((out_x + 1) * 200 + 27) // 28
            result[:, 0, out_y, out_x] = (
                gained[:, y0:y1, x0:x1].mean(axis=(1, 2)).astype(np.float32)
            )
    return result


def _numpy_row_relative_l2(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim < 2:
        raise ValueError("independent cross-grid arrays must have matching row shapes")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("independent cross-grid arrays contain nonfinite values")
    reference_rows = reference.reshape(reference.shape[0], -1)
    candidate_rows = candidate.reshape(candidate.shape[0], -1)
    numerator = np.linalg.norm(candidate_rows - reference_rows, axis=1)
    denominator = np.maximum(np.linalg.norm(reference_rows, axis=1), 1.0e-30)
    return numerator / denominator


def _numpy_aggregate_relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim < 2:
        raise ValueError("independent aggregate cross-grid arrays must have matching row shapes")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("independent aggregate cross-grid arrays contain nonfinite values")
    numerator = np.linalg.norm((candidate - reference).reshape(-1))
    denominator = max(float(np.linalg.norm(reference.reshape(-1))), 1.0e-30)
    return float(numerator / denominator)


def _independent_diversity(
    readouts: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    output = {}
    flat = np.asarray(readouts, dtype=np.float64).reshape(len(labels), -1)
    for label in range(CLASS_COUNT):
        values = flat[labels == label]
        pair_sum = 0.0
        pair_count = 0
        for index in range(len(values) - 1):
            difference = values[index + 1 :] - values[index]
            pair_sum += float(
                np.sqrt(np.einsum("ij,ij->i", difference, difference)).sum()
            )
            pair_count += len(difference)
        output[str(label)] = pair_sum / pair_count / math.sqrt(flat.shape[1])
    return output


def _independent_bootstrap(
    labels: np.ndarray,
    direct: Mapping[str, np.ndarray],
    paths: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = [
        rng.integers(0, ROWS_PER_CLASS, (BOOTSTRAP_COUNT, ROWS_PER_CLASS))
        for _ in range(CLASS_COUNT)
    ]
    output = {}
    for path_name, values in paths.items():
        output[path_name] = {}
        for evaluator in ("ED", "EC"):
            direct_correct = direct[evaluator] == labels
            path_correct = values[evaluator] == labels
            draws = np.zeros(BOOTSTRAP_COUNT, dtype=np.float64)
            per_class = {}
            point = []
            for label in range(CLASS_COUNT):
                positions = np.flatnonzero(labels == label)
                a = direct_correct[positions].astype(np.float64)
                b = path_correct[positions].astype(np.float64)
                delta = float(b.mean() - a.mean())
                point.append(delta)
                per_class[str(label)] = delta
                draws += (
                    b[samples[label]].mean(axis=1) - a[samples[label]].mean(axis=1)
                ) / CLASS_COUNT
            lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
            output[path_name][evaluator] = {
                "point_macro_recall_delta": float(np.mean(point)),
                "paired_class_stratified_ci95": [float(lower), float(upper)],
                "per_class_delta": per_class,
            }
    return output


def _load_saved_conditions(
    config: Mapping[str, Any],
    artifact_dir: Path,
    device: torch.device,
    projection_source_dir: Path | None = None,
    projection_review_path: Path | None = None,
    training_preflight_path: Path | None = None,
    gradient_diagnostics_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assets = _load_assets(
        config,
        device,
        projection_source_dir,
        projection_review_path,
        training_preflight_path,
        gradient_diagnostics_path,
    )
    saved_config = yaml.safe_load(
        (artifact_dir / "run_config.yaml").read_text(encoding="utf-8")
    )
    _require(
        saved_config == config,
        "saved numerical run config differs from the registered Windows config",
    )
    status = _read_json(artifact_dir / "measurement_status.json")
    task_state = _read_json(artifact_dir / "task_state.json")
    identity = ResearchRunIdentity.from_mapping(
        _read_json(artifact_dir / "run_identity.json")
    )
    receipt = PreflightReceipt.from_mapping(
        _read_json(artifact_dir / "preflight_ready.json")
    )
    preflight = _load_torch(artifact_dir / "preflight.pt")
    controls = _load_torch(artifact_dir / "preflight_controls.pt")
    source_binding = _read_json(artifact_dir / "projection_source_binding.json")
    _require(
        status.get("task_id") == TASK_ID
        and status.get("status") == "COMPLETE_DIAGNOSTIC"
        and status.get("direct_ols_2048_rows") == POPULATION_ROWS
        and status.get("optimizer_updates") == 0
        and task_state.get("task_id") == TASK_ID
        and task_state.get("status") == "COMPLETE_DIAGNOSTIC"
        and task_state.get("completed_direct_ols_rows") == POPULATION_ROWS
        and identity.task_id == TASK_ID
        and identity.semantic_authority_id == AUTHORIZATION_ID
        and identity.model_identity.get("global_step") == 52_000
        and receipt.status == "READY"
        and receipt.authorization_id == AUTHORIZATION_ID
        and all(
            receipt.checks[name].get("passed") is True
            for name in (
                CORRECTNESS_INVARIANTS,
                RESOURCE_ENVELOPE,
                IMPLEMENTATION_SELECTION,
            )
        )
        and preflight.get("task_id") == TASK_ID
        and preflight.get("status") == "PASSED"
        and preflight.get("optimizer_updates") == 0,
        "Windows consumer rejects an incomplete or updated direct-OLS result",
    )
    _require(
        source_binding.get("task_id") == TASK_ID
        and source_binding.get("source_task_id") == alignment.TASK_ID
        and isinstance(source_binding.get("source_artifact_dir"), str)
        and bool(source_binding["source_artifact_dir"])
        and isinstance(source_binding.get("source_review_path"), str)
        and bool(source_binding["source_review_path"])
        and isinstance(source_binding.get("training_preflight_path"), str)
        and bool(source_binding["training_preflight_path"])
        and isinstance(source_binding.get("gradient_diagnostics_path"), str)
        and bool(source_binding["gradient_diagnostics_path"])
        and source_binding.get("population_role")
        in {"consumed_train_diagnostic", "consumed_generation_diagnostic"}
        and source_binding.get("population_rows") == POPULATION_ROWS
        and source_binding.get("endpoint") == "P1_E1G1"
        and source_binding.get("endpoint_global_step") == 52_000
        and source_binding.get("source_manifest_validated") is True,
        "Windows consumer rejects a missing P1 source-path or population binding",
    )
    _validate_direct_shard(controls.get("direct_2048", {}), CONTROL_ROWS, assets)
    _require(
        preflight.get("f1_reinjection", {}).get("passed") is True
        and preflight.get("direct_ols_cross_grid", {}).get("passed") is True
        and preflight.get("training_path", {}).get("status") == "PASSED"
        and controls.get("task_id") == TASK_ID,
        "Windows consumer rejects missing direct-OLS, training-path, or READY control evidence",
    )
    direct = _load_direct_population(artifact_dir, assets)
    conditions = {
        "D_DIRECT_OLS": {
            "coefficients": direct["coefficients"],
            "raw_cmos": direct["raw_cmos"],
        },
        "P0_E1G1": {
            "coefficients": assets["p0"]["coefficients"],
            "raw_cmos": assets["p0"]["raw_cmos"],
        },
        "P1_E1G1": {
            "coefficients": assets["p1"]["coefficients"],
            "raw_cmos": assets["p1"]["raw_cmos"],
        },
    }
    for name, output in conditions.items():
        _require(
            tuple(output["coefficients"].shape) == (POPULATION_ROWS, 100)
            and tuple(output["raw_cmos"].shape) == (POPULATION_ROWS, 1, 200, 200)
            and torch.equal(
                assets["generation"]["labels"], torch.arange(10).repeat_interleave(200)
            ),
            f"Windows consumer rejects an incomplete {name} population",
        )
    _require(
        torch.equal(direct["targets"], assets["generation"]["saved_prior_targets"])
        and torch.equal(direct["coefficients"], direct["targets"])
        and torch.equal(assets["p1"]["targets"], direct["targets"])
        and torch.equal(assets["p1"]["labels"], direct["labels"])
        and torch.equal(assets["p1"]["proposal_ids"], direct["proposal_ids"])
        and torch.equal(
            assets["p1"]["commands"].clamp(0.0, 1.0), assets["p1"]["commands"]
        ),
        "Windows consumer rejects mismatched targets, labels, proposals, or P1 SLM commands",
    )
    return assets, {"direct": direct, "conditions": conditions}


def _validate_p1_reinjection_controls(
    controls: Mapping[str, Any], assets: Mapping[str, Any]
) -> dict[str, Any]:
    indices = CONTROL_ROWS
    generation = assets["generation"]
    p1 = assets["p1"]
    expected_coefficients = p1["coefficients"].index_select(0, indices).cpu().to(
        torch.complex128
    )
    expected_raw = p1["raw_cmos"].index_select(0, indices).cpu().double()
    expected_labels = generation["labels"].index_select(0, indices).cpu()
    expected_proposals = generation["proposal_ids"].index_select(0, indices).cpu()
    expected_commands = p1["commands"].index_select(0, indices).cpu()
    raw = controls.get("raw_cmos")
    w_pixel = controls.get("w_per_pixel")
    target_raw = controls.get("target_raw_cmos")
    target_w_pixel = controls.get("target_w_per_pixel")
    coefficients = controls.get("input_ols_coefficients")
    _require(
        controls.get("task_id") == REPAIR_TASK_ID
        and controls.get("population_role") == "consumed_generation_diagnostic"
        and controls.get("condition") == "P1_E1G1"
        and controls.get("endpoint_source_run_id")
        == _terminal_checkpoint(assets)["run_id"]
        and controls.get("endpoint_global_step") == 52_000
        and controls.get("encoder_adam_step") == 47_000
        and controls.get("generator_adam_step") == 52_000
        and controls.get("grid_size") == 2048
        and torch.equal(controls.get("population_indices"), indices)
        and torch.equal(controls.get("labels"), expected_labels)
        and torch.equal(controls.get("proposal_ids"), expected_proposals)
        and torch.equal(controls.get("actual_slm_commands_forwarded"), expected_commands)
        and torch.equal(coefficients, expected_coefficients)
        and controls.get("raw_cmos_unit") == "W/m^2"
        and controls.get("w_per_pixel_unit") == "W/pixel"
        and controls.get("ols_basis_id") == "gaussian_s12_v1"
        and controls.get("ols_transport") == "T=I_100 coefficient space"
        and controls.get("modal_coefficient_scale")
        == "registered_native_complex_OLS_scale"
        and math.isclose(
            float(controls.get("camera_pixel_area_m2", math.nan)),
            float(assets["runtime"]["camera_area"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and torch.is_tensor(raw)
        and torch.is_tensor(w_pixel)
        and torch.is_tensor(target_raw)
        and torch.is_tensor(target_w_pixel)
        and tuple(raw.shape) == tuple(target_raw.shape) == (20, 1, 200, 200)
        and tuple(w_pixel.shape) == tuple(target_w_pixel.shape) == (20, 1, 200, 200),
        "final consumer rejects a wrong P1 injection row, target, command, endpoint, mode, or unit",
    )
    raw_error = float((raw.cpu().double() - expected_raw).abs().max())
    wpixel_error = float(
        (w_pixel.cpu().double() - expected_raw * float(assets["runtime"]["camera_area"]))
        .abs()
        .max()
    )
    target_error = float((target_raw.cpu().double() - expected_raw).abs().max())
    target_wpixel_error = float(
        (
            target_w_pixel.cpu().double()
            - expected_raw * float(assets["runtime"]["camera_area"])
        )
        .abs()
        .max()
    )
    _require(
        max(raw_error, wpixel_error, target_error, target_wpixel_error)
        <= REINJECTION_TOLERANCE,
        "final consumer recomputed a P1 reinjection error above 1e-12",
    )
    return {
        "rows": len(indices),
        "indices": indices.tolist(),
        "raw_cmos_max_abs_error": raw_error,
        "w_per_pixel_max_abs_error": wpixel_error,
        "target_raw_cmos_max_abs_error": target_error,
        "target_w_per_pixel_max_abs_error": target_wpixel_error,
        "tolerance": REINJECTION_TOLERANCE,
        "passed": True,
    }


def _recompute_saved_cross_grid_controls(
    assets: Mapping[str, Any], old_closure_backup: Path
) -> dict[str, Any]:
    saved_controls = _load_torch(old_closure_backup / "preflight_controls.pt")
    direct2048 = saved_controls.get("direct_2048", {})
    direct4096 = saved_controls.get("direct_4096", {})
    _validate_direct_shard(direct2048, CONTROL_ROWS, assets)
    expected_target = assets["generation"]["saved_prior_targets"].index_select(
        0, CONTROL_ROWS
    )
    _require(
        torch.equal(direct2048.get("coefficients"), expected_target)
        and torch.equal(direct4096.get("coefficients"), expected_target)
        and tuple(direct4096.get("raw_cmos", torch.empty(0)).shape)
        == (20, 1, 200, 200)
        and tuple(direct4096.get("w_per_pixel", torch.empty(0)).shape)
        == (20, 1, 200, 200)
        and torch.isfinite(direct4096["raw_cmos"]).all().item()
        and torch.isfinite(direct4096["w_per_pixel"]).all().item(),
        "saved D 4096 controls are missing, misordered, or bound to different OLS targets",
    )
    d_coeff = _numpy_row_relative_l2(
        direct2048["coefficients"].numpy(), direct4096["coefficients"].numpy()
    )
    d_cmos = _numpy_row_relative_l2(
        direct2048["raw_cmos"].numpy(), direct4096["raw_cmos"].numpy()
    )
    d_wpixel = _numpy_row_relative_l2(
        direct2048["w_per_pixel"].numpy(), direct4096["w_per_pixel"].numpy()
    )
    p1_cross = assets["p1"]["cross_grid_controls"]
    p1_coeff = _numpy_row_relative_l2(
        p1_cross["coefficients_2048"].cpu().numpy(),
        p1_cross["coefficients_4096"].cpu().numpy(),
    )
    p1_cmos = _numpy_row_relative_l2(
        p1_cross["raw_cmos_2048"].cpu().numpy(),
        p1_cross["raw_cmos_4096"].cpu().numpy(),
    )
    errors = {
        "D_DIRECT_OLS": {
            "coefficient_per_row": d_coeff.tolist(),
            "raw_cmos_per_row": d_cmos.tolist(),
            "w_per_pixel_per_row": d_wpixel.tolist(),
            "coefficient_max_relative_error": float(np.max(d_coeff)),
            "raw_cmos_max_relative_error": float(np.max(d_cmos)),
            "w_per_pixel_max_relative_error": float(np.max(d_wpixel)),
            "coefficient_aggregate_relative_error": _numpy_aggregate_relative_l2(
                direct2048["coefficients"].numpy(),
                direct4096["coefficients"].numpy(),
            ),
            "raw_cmos_aggregate_relative_error": _numpy_aggregate_relative_l2(
                direct2048["raw_cmos"].numpy(), direct4096["raw_cmos"].numpy()
            ),
            "w_per_pixel_aggregate_relative_error": _numpy_aggregate_relative_l2(
                direct2048["w_per_pixel"].numpy(), direct4096["w_per_pixel"].numpy()
            ),
        },
        "P1_E1G1": {
            "coefficient_per_row": p1_coeff.tolist(),
            "raw_cmos_per_row": p1_cmos.tolist(),
            "coefficient_max_relative_error": float(np.max(p1_coeff)),
            "raw_cmos_max_relative_error": float(np.max(p1_cmos)),
            "coefficient_aggregate_relative_error": _numpy_aggregate_relative_l2(
                p1_cross["coefficients_2048"].cpu().numpy(),
                p1_cross["coefficients_4096"].cpu().numpy(),
            ),
            "raw_cmos_aggregate_relative_error": _numpy_aggregate_relative_l2(
                p1_cross["raw_cmos_2048"].cpu().numpy(),
                p1_cross["raw_cmos_4096"].cpu().numpy(),
            ),
            "w_per_pixel_cross_grid_available": False,
        },
        "P0_E1G1": {
            "cross_grid_controls_available": False,
            "reason": "the retained P0 measurement has 2048 full-population output but no paired 4096 control array",
        },
        "tolerance": CROSS_GRID_TOLERANCE,
    }
    passed = max(
        errors["D_DIRECT_OLS"]["coefficient_max_relative_error"],
        errors["D_DIRECT_OLS"]["raw_cmos_max_relative_error"],
        errors["D_DIRECT_OLS"]["w_per_pixel_max_relative_error"],
        errors["P1_E1G1"]["coefficient_max_relative_error"],
        errors["P1_E1G1"]["raw_cmos_max_relative_error"],
    ) <= CROSS_GRID_TOLERANCE
    _require(
        passed,
        "independent NumPy cross-grid consumer found a saved 2048/4096 error above 1e-4",
    )
    errors["passed"] = True
    return errors


def _verify_saved(
    config: Mapping[str, Any],
    artifact_dir: Path,
    review_dir: Path,
    *,
    projection_source_dir: Path | None = None,
    projection_review_path: Path | None = None,
    training_preflight_path: Path | None = None,
    gradient_diagnostics_path: Path | None = None,
    optimizer_audit: _OptimizerEventAudit,
    cross_grid_recalculation: Mapping[str, Any] | None = None,
) -> None:
    _require(
        artifact_dir.resolve() == BACKUP_DIR.resolve()
        and review_dir.resolve() in {REVIEW_DIR.resolve(), REPAIR_REVIEW_DIR.resolve()},
        "saved verification requires the explicit Windows backup and registered review path",
        ValueError,
    )
    assets, saved = _load_saved_conditions(
        config,
        artifact_dir,
        torch.device("cpu"),
        projection_source_dir,
        projection_review_path,
        training_preflight_path,
        gradient_diagnostics_path,
    )
    optimizer_audit.require_zero("Windows saved-result review")
    if review_dir.resolve() == REPAIR_REVIEW_DIR.resolve():
        _require(
            isinstance(cross_grid_recalculation, Mapping)
            and cross_grid_recalculation.get("passed") is True,
            "repair result consumer requires the independently recomputed saved-array cross-grid gate",
        )
    runtime = assets["runtime"]
    labels = assets["generation"]["labels"].cpu().numpy().astype(np.int64)
    summaries, numpy_views, torch_views = {}, {}, {}
    per_condition = {}
    for name, output in saved["conditions"].items():
        raw = output["raw_cmos"].detach().cpu().double()
        numpy_view = _numpy_readout(
            raw.numpy(), runtime["camera_area"], float(runtime["camera_gain"])
        )
        torch_view = _condition_readouts(assets, raw)
        _require(
            float(np.max(np.abs(numpy_view - torch_view.numpy()))) <= 2.0e-6,
            f"NumPy fixed-display conversion differs from the project consumer for {name}",
        )
        summary = recent.frame._summary_from_readouts(
            runtime["input_bundle"],
            assets["cohort"],
            torch.from_numpy(numpy_view.copy()),
        )
        summaries[name], numpy_views[name], torch_views[name] = (
            summary,
            numpy_view,
            torch_view,
        )
        per_condition[name] = {
            evaluator: np.asarray(
                summary["conditions"]["direct_decoder"]["evaluators"][evaluator][
                    "predicted_labels"
                ],
                dtype=np.int64,
            )
            for evaluator in ("ED", "EC")
        }
    direct_predictions = per_condition["D_DIRECT_OLS"]
    path_predictions = {name: per_condition[name] for name in ("P0_E1G1", "P1_E1G1")}
    independent_bootstrap = _independent_bootstrap(
        labels, direct_predictions, path_predictions
    )
    independent_diversity = {
        name: _independent_diversity(numpy_views[name], labels)
        for name in ("D_DIRECT_OLS", "P0_E1G1", "P1_E1G1")
    }
    source_binding = _read_json(artifact_dir / "projection_source_binding.json")
    torch_result, rows, _readouts = _analyze(assets, saved["direct"])
    torch_result["optimizer_instances"] = optimizer_audit.optimizer_instances
    torch_result["optimizer_updates"] = optimizer_audit.optimizer_updates
    torch_result["historical_optimizer_instance_count"] = "UNKNOWN_UNINSTRUMENTED"
    torch_result["historical_optimizer_update_count"] = "UNKNOWN_UNINSTRUMENTED"
    torch_result["corrected_population_role"] = "consumed_generation_diagnostic"
    comparison_checks = {}
    numpy_path_decisions = {}
    for name in ("P0_E1G1", "P1_E1G1"):
        expected = torch_result["comparisons"][name]
        comparisons = {}
        for evaluator in ("ED", "EC"):
            actual_ci = independent_bootstrap[name][evaluator][
                "paired_class_stratified_ci95"
            ]
            expected_ci = expected[evaluator]["paired_class_stratified_ci95"]
            _require(
                np.allclose(actual_ci, expected_ci, rtol=0.0, atol=1.0e-12),
                f"independent NumPy bootstrap interval differs for {name}/{evaluator}",
            )
            comparisons[evaluator] = {
                "CI95_max_abs_error": float(
                    np.max(np.abs(np.asarray(actual_ci) - np.asarray(expected_ci)))
                ),
                "matches": True,
            }
        numpy_diversity = independent_diversity[name]
        torch_diversity = expected["class_diversity_retention"]["per_class_ratio"]
        ratios = {
            str(label): numpy_diversity[str(label)]
            / max(independent_diversity["D_DIRECT_OLS"][str(label)], 1.0e-30)
            for label in range(CLASS_COUNT)
        }
        _require(
            all(
                math.isclose(
                    ratios[str(label)],
                    float(torch_diversity[str(label)]),
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-10,
                )
                for label in range(CLASS_COUNT)
            ),
            f"independent NumPy class diversity differs for {name}",
        )
        numpy_path_decisions[name] = _numpy_path_retention_decision(
            {
                evaluator: independent_bootstrap[name][evaluator][
                    "paired_class_stratified_ci95"
                ]
                for evaluator in ("ED", "EC")
            },
            ratios,
        )
        _require(
            numpy_path_decisions[name] == expected["path_decision"],
            f"independent NumPy path decision differs for {name}",
        )
        comparisons["diversity"] = {
            "per_class_ratio_max_abs_error": max(
                abs(ratios[str(label)] - float(torch_diversity[str(label)]))
                for label in range(CLASS_COUNT)
            ),
            "matches": True,
        }
        comparisons["path_decision"] = {
            "numpy": numpy_path_decisions[name],
            "production": expected["path_decision"],
            "matches": True,
        }
        comparison_checks[name] = comparisons
    numpy_decision = _overall_path_decision(numpy_path_decisions)
    _require(
        numpy_decision == torch_result["decision"]
        and numpy_decision
        == _read_json(artifact_dir / "final_decision.json").get("decision"),
        "Windows raw CMOS consumer does not reproduce the saved path decision",
    )
    review_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        review_dir / "independent_review.json",
        {
            "task_id": TASK_ID,
            "p1_source_artifact_dir": source_binding["source_artifact_dir"],
            "p1_source_review_path": source_binding["source_review_path"],
            "training_preflight_path": source_binding["training_preflight_path"],
            "gradient_diagnostics_path": source_binding[
                "gradient_diagnostics_path"
            ],
            "passed": True,
            "decision": torch_result["decision"],
            "saved_array_cross_grid_gate": (
                dict(cross_grid_recalculation)
                if cross_grid_recalculation is not None
                else None
            ),
            "both_paths_pass": torch_result["both_paths_pass"],
            "independent_numpy_path_decisions": numpy_path_decisions,
            "independent_numpy_overall_decision": numpy_decision,
            "independent_display_conversion": "NumPy clamp and adaptive average pooling from saved raw CMOS",
            "display_max_abs_errors": {
                name: float(
                    np.max(np.abs(numpy_views[name] - torch_views[name].numpy()))
                )
                for name in numpy_views
            },
            "raw_cmos_recomputed_evaluator_predictions": True,
            "independent_numpy_bootstrap": independent_bootstrap,
            "independent_numpy_diversity": independent_diversity,
            "comparison_checks": comparison_checks,
            "population_rows": POPULATION_ROWS,
            **optimizer_audit.snapshot(),
            "historical_optimizer_instance_count": "UNKNOWN_UNINSTRUMENTED",
            "historical_optimizer_updates": "UNKNOWN_UNINSTRUMENTED",
            "historical_zero_instance_claim_invalidated": True,
            "corrected_population_role": "consumed_generation_diagnostic",
            "source_population_role_label": source_binding["population_role"],
            "official_test_reads": 0,
        },
        exclusive=True,
    )
    _write_json(
        review_dir / "recomputed_final_decision.json", torch_result, exclusive=True
    )
    _write_json(
        review_dir / "paper_training_inference_alignment.json",
        _paper_alignment_record(),
        exclusive=True,
    )
    _write_csv(review_dir / "per_sample_results.csv", rows)
    _write_plot(review_dir / "condition_comparison.png", torch_result)
    atlas = _write_example_image_comparison(
        review_dir / "fixed_display_and_raw_cmos_examples.png",
        labels,
        numpy_views,
        {name: saved["conditions"][name]["raw_cmos"] for name in numpy_views},
    )
    _write_json(review_dir / "fixed_display_and_raw_cmos_examples.json", atlas)
    _write_closeout(
        review_dir / "closeout_report.md",
        torch_result,
        _load_torch(artifact_dir / "preflight.pt"),
        reviewed=True,
        optimizer_events=optimizer_audit.snapshot(),
    )


def _validate_repair_witnesses(
    artifact_dir: Path,
    schedule: torch.Tensor,
    labels: torch.Tensor,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    witness_dir = artifact_dir / "training_witnesses"
    expected_updates = (1, 3, 4)
    files = sorted(witness_dir.glob("training_path_witness_update_*.pt"))
    _require(
        {path.name for path in files}
        == {f"training_path_witness_update_{value:05d}.pt" for value in expected_updates},
        "training path witness set is missing, duplicated, or contains an unplanned update",
    )
    records = []
    start_steps = (
        int(checkpoint["optimizer_steps"]["encoder_adam_step"]),
        int(checkpoint["optimizer_steps"]["generator_adam_step"]),
    )
    for update in expected_updates:
        witness = _load_torch(
            witness_dir / f"training_path_witness_update_{update:05d}.pt"
        )
        row = schedule[update - 1].long()
        expected_labels = labels.index_select(0, row)
        before = {
            "encoder": start_steps[0] + update - 1,
            "generator": start_steps[1] + update - 1,
        }
        after = {name: value + 1 for name, value in before.items()}
        _require(
            witness.get("schema") == "lightgen_fullpath_training_witness.v1"
            and witness.get("source_training_run_id") == checkpoint["run_id"]
            and witness.get("forward_global_step")
            == int(checkpoint["completed_global_step"]) + update - 1
            and witness.get("completed_global_step")
            == int(checkpoint["completed_global_step"]) + update
            and witness.get("completed_update") == update
            and witness.get("optimizer_steps_before") == before
            and witness.get("optimizer_steps_after") == after
            and torch.equal(witness.get("source_rows"), row)
            and torch.equal(witness.get("source_labels"), expected_labels)
            and tuple(witness.get("ols_coefficients", torch.empty(0)).shape)
            == (BATCH_SIZE, 100)
            and torch.is_complex(witness["ols_coefficients"])
            and tuple(witness.get("plane_powers_w", torch.empty(0)).shape)
            == (BATCH_SIZE, 5)
            and tuple(witness.get("raw_cmos_w_per_m2", torch.empty(0)).shape)
            == (BATCH_SIZE, 1, 200, 200)
            and torch.isfinite(witness["ols_coefficients"].real).all().item()
            and torch.isfinite(witness["ols_coefficients"].imag).all().item()
            and torch.isfinite(witness["plane_powers_w"]).all().item()
            and torch.isfinite(witness["raw_cmos_w_per_m2"]).all().item()
            and bool(((witness["slm_amplitude"] >= 0.0)
                      & (witness["slm_amplitude"] <= 1.0)).all())
            and torch.allclose(
                witness["slm_amplitude"].square(),
                witness["slm_intensity"],
                rtol=1.0e-6,
                atol=1.0e-7,
            ),
            f"training witness for update {update} is not bound to its actual pre-update TRAIN forward",
        )
        records.append(
            {
                "update": update,
                "forward_global_step": witness["forward_global_step"],
                "completed_global_step": witness["completed_global_step"],
                "source_rows": row.tolist(),
                "optimizer_steps_before": before,
                "optimizer_steps_after": after,
                "pre_update_fullpath_fields_present": True,
            }
        )
    return {"status": "PASSED", "witnesses": records}


def _write_repair_closeout(
    path: Path,
    decision: Mapping[str, Any],
    status: Mapping[str, Any],
    recovery: Mapping[str, Any],
    controls: Mapping[str, Any],
    finite_difference: Mapping[str, Any],
) -> None:
    rows = [
        "# 完整光路工程闭环修复与阶段收尾",
        "",
        "## 结果与状态",
        "",
        f"修复验收：`{status['status']}`；D/P0/P1 路径复核：`{decision['decision']}`。",
        "本轮科学训练更新、先验重拟合、新先验采样和官方 TEST 读取均为 0。隔离工程演练含 3 步连续更新、直接第 4 步和独立进程恢复第 4 步，共 5 次更新调用；这些状态仅用于工程比对，不是候选模型。",
        f"优化器事件实测为 {status['optimizer_events']['optimizer_instances']} 个实例、{status['optimizer_events']['optimizer_updates']} 次成功更新；源终点检查点、Adam 和冻结资产保持原值。",
        "工程演练使用的模型、Adam、梯度及 CPU/CUDA RNG 已恢复到源终点调用前状态；隔离更新状态不作为候选。",
        "历史闭环报告中的 optimizer instances=0 予以勘误：旧只读加载器构造过 Adam 对象，但没有完整事件计数；历史实例与更新的总数均标为未知。修复后的只读加载和 Windows 复核各自实测为 0 个实例、0 次更新。",
        "训练证据来源分开绑定：历史 `training_preflight.json` 是 D0/global step 50000、Adam 45000/50000 的入口验证，只作历史佐证；E1G1/global step 52000 的固定批次损失与梯度来自 attribution `step_02000` 诊断，并由同一训练 run 的 source identity 绑定。D0 预检不作为 E1G1 终点证据。",
        "",
        "## 科学认识",
        "",
        f"D/P0/P1 唯一复核结论沿用独立 NumPy 消费者计算的 `{decision['decision']}`。该结果只适用于已消费的 2,000 条固定先验生成诊断人口。旧 source manifest 标注 `consumed_train_diagnostic` 与实际保存先验目标/proposal 身份不符；本报告更正为 `consumed_generation_diagnostic`，真实 TRAIN 只出现在训练日程和更新见证中。",
        "完整路径仍为训练图像→原生振幅 SLM→Encoder→Gaussian OLS→Generator→raw CMOS；推理为保存目标经 mapper/投影制备 SLM 命令，再经过同一完整数值路径。D 直接 OLS 仅作边界参照。",
        "",
        "## 目标缺口",
        "",
        "通过已消费人口上的路径保留判断不等于独立人口泛化、人工可读性或实物光路通过。下一阶段草案固定同一终点/适配器，新人口每类 200 条，D/P0/P1 同行配对，沿用两评价器 2 pp 非劣界和逐类 80% 多样性保护；本轮没有生成该人口。",
        "实物缺口仍包括：SLM 振幅/灰阶/残余相位标定；相位器件与 SLM/CMOS 的坐标及继像注册；器件间复传播；OLS/光纤复传输与功率损耗/串扰；相机响应、线性区和曝光标定；最终端到端样本闭环。当前身份传输、理想 SLM 和数值 CMOS 不构成硬件测量。",
        "",
        "## 工程完成与验证",
        "",
        f"- 更新消费者：BOGT+SSIM 的重构和 SSIM 分别与同一次 CMOS 前向的规范梯度核对；两相位组 CMOS 梯度必须有限且非零，单组冻结被拒绝。",
        f"- 断梯度负控：总非 CMOS 梯度仍非零时，CMOS 项断开在 optimizer.step 前被拒绝；模型和 Adam 未改变。",
        f"- 独立恢复：模型、Adam、下一批、loss、RNG 与见证逐项按 rtol={recovery['recovery_rtol']:g}, atol={recovery['recovery_atol']:g} 复核；状态通过。",
        f"- D 2048/4096 与 P1 2048/4096 控制误差由保存数组逐条 NumPy 重算，阈值 {controls['tolerance']:g}；P1 实际 OLS 系数后端重注入 20 条，raw CMOS 与 W/pixel 最大误差分别为 {controls['p1_reinjection']['raw_cmos_max_abs_error']:.3g}、{controls['p1_reinjection']['w_per_pixel_max_abs_error']:.3g}。",
        f"- 既有响应矩阵有限差分：端点 step {finite_difference['endpoint_global_step']}、epsilon {finite_difference['epsilon_amplitude']:g}、相对误差 {finite_difference['relative_error']:.3g} / 阈值 {finite_difference['relative_error_tolerance']:g}，模型恢复差 {finite_difference['model_restoration_max_abs_delta']:g}。正负输出向量未保存在旧 run，报告保留这一历史归档限制。",
        "- Windows 从 raw CMOS 重新生成固定显示和 E_D/E_C 预测，独立实现配对 bootstrap、逐类多样性及路径判据；新增 D/P0/P1 同行固定视图与 raw CMOS 图册。",
        "",
        "## 决策影响与下一步",
        "",
        "保留近期科研权重和全部历史产物；本次工程闭环完成后停止。后继动作建议是新人口的零更新验证，再据此讨论单变量训练方向。此建议不是新人口采样、训练、模型采用或部署授权。Chen 等 Science 论文的 OLS 生成操作与项目完整 SLM→Encoder→OLS→Generator 路径继续分开描述；BOGT 的项目固定 SSIM 附加项、2+2 架构和输入 mapper/投影均标为项目差异。",
        "",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def _validate_repair_optimizer_event_record(
    events: Mapping[str, Any], execution: Mapping[str, Any]
) -> dict[str, int]:
    parent = events.get("parent_process")
    worker = events.get("independent_recovery_process")
    failed = events.get("failed_recovery_process")
    expected_total = {"optimizer_instances": 2, "optimizer_updates": 5}
    _require(
        parent == {"optimizer_instances": 1, "optimizer_updates": 4}
        and worker == {"optimizer_instances": 1, "optimizer_updates": 1},
        "Windows consumer rejects missing or inconsistent parent/recovery optimizer events",
    )
    if failed is not None:
        _require(
            isinstance(failed, Mapping)
            and failed.get("optimizer_instances") == 1
            and failed.get("optimizer_updates") == 0
            and failed.get("status") == "FAILED_BEFORE_UPDATE"
            and failed.get("trace_file") == "recovery_worker_stderr.txt"
            and events.get("recovery_worker_failure_reconciled") is True,
            "Windows consumer rejects an unbound failed recovery-worker optimizer event",
        )
        expected_total = {"optimizer_instances": 3, "optimizer_updates": 5}
    _require(
        events.get("total") == expected_total
        and execution.get("optimizer_events") == expected_total,
        "Windows consumer rejects the total optimizer event count",
    )
    return expected_total


def _verify_repair_saved(
    repair_config: Mapping[str, Any],
    repair_config_path: Path,
    artifact_dir: Path,
    review_dir: Path,
) -> None:
    _require(
        artifact_dir.resolve() == REPAIR_BACKUP_DIR.resolve()
        and review_dir.resolve() == REPAIR_REVIEW_DIR.resolve(),
        "repair review requires the explicit Windows backup and review paths",
        ValueError,
    )
    _require(artifact_dir.is_dir(), "repair backup directory is missing", FileNotFoundError)
    saved_config = yaml.safe_load(
        (artifact_dir / "run_config.yaml").read_text(encoding="utf-8")
    )
    _require(saved_config == repair_config,
             "repair backup configuration differs from the registered repair plan")
    execution = _read_json(artifact_dir / "repair_execution_status.json")
    events = _read_json(artifact_dir / "optimizer_events.json")
    update_log = _read_json(artifact_dir / "engineering_update_log.json")
    negative = _read_json(artifact_dir / "detached_cmos_negative_control.json")
    saved_recovery = _read_json(artifact_dir / "engineering_recovery_comparison.json")
    rollback = _read_json(artifact_dir / "engineering_rollback_status.json")
    training_evidence_scope = _read_json(
        artifact_dir / "training_evidence_scope.json"
    )
    _require(
        execution.get("task_id") == REPAIR_TASK_ID
        and execution.get("status") == "ENGINEERING_REPAIR_COMPLETE"
        and execution.get("source_training_run_id")
        == repair_config["sources"]["expected_training_run_id"]
        and execution.get("source_global_step") == 52_000
        and execution.get("engineering_update_calls") == 5
        and execution.get("negative_control_passed") is True
        and execution.get("recovery_comparison_passed") is True
        and execution.get("frozen_assets_unchanged") is True
        and execution.get("source_checkpoint_unchanged") is True
        and execution.get("isolated_engineering_states_rolled_back") is True
        and execution.get("training_evidence_scope") == training_evidence_scope
        and execution.get("scientific_updates") == 0
        and execution.get("prior_refits") == execution.get("prior_samples") == 0
        and execution.get("official_test_reads") == 0,
        "Windows consumer rejects incomplete engineering repair status",
    )
    _require(
        rollback.get("task_id") == REPAIR_TASK_ID
        and rollback.get("status") == "ROLLED_BACK_TO_SOURCE_ENDPOINT"
        and rollback.get("source_training_run_id")
        == repair_config["sources"]["expected_training_run_id"]
        and rollback.get("source_global_step") == 52_000
        and rollback.get("source_optimizer_steps") == [47_000, 52_000]
        and rollback.get("model_restored") is True
        and rollback.get("adam_restored") is True
        and rollback.get("phase_gradients_cleared") is True
        and rollback.get("cpu_rng_restored") is True
        and rollback.get("cuda_rng_restored") is True
        and rollback.get("source_checkpoint_unchanged") is True
        and rollback.get("frozen_assets_unchanged") is True
        and rollback.get("isolated_states_are_not_candidates") is True,
        "Windows consumer rejects a missing or incomplete source-endpoint rollback record",
    )
    _require(
        update_log.get("task_id") == REPAIR_TASK_ID
        and update_log.get("status") == "RECOVERED_BRANCH_COMPLETE"
        and update_log.get("expected_update_calls") == 5
        and len(update_log.get("calls", [])) == 5
        and [call.get("call_number") for call in update_log["calls"]]
        == [1, 2, 3, 4, 5]
        and [call.get("branch") for call in update_log["calls"]]
        == [
            "three_continuous",
            "three_continuous",
            "three_continuous",
            "fourth_direct",
            "fourth_independent_restore",
        ]
        and all(call.get("status") == "COMPLETED" for call in update_log["calls"]),
        "Windows consumer rejects missing, duplicate, or reordered engineering optimizer calls",
    )
    if update_log["calls"][3].get("completion_reconciled_from_saved_branch") is True:
        direct_reconciliation = _read_json(
            artifact_dir / "direct_fourth_reconciliation.json"
        )
        _require(
            direct_reconciliation.get("status") == "PASSED"
            and direct_reconciliation.get("call_number") == 4
            and direct_reconciliation.get("completion_source")
            == "saved_direct_branch_and_pre_update_witness"
            and direct_reconciliation.get("source_rows")
            == update_log["calls"][3].get("sample_rows")
            and direct_reconciliation.get("optimizer_steps") == [47_004, 52_004],
            "Windows consumer rejects an unsupported direct fourth-call log reconciliation",
        )
    reconciled_events = _validate_repair_optimizer_event_record(events, execution)
    if "failed_recovery_process" in events:
        event_reconciliation = _read_json(
            artifact_dir / "optimizer_event_reconciliation.json"
        )
        failed_worker_log = (artifact_dir / "recovery_worker_stderr.txt").read_text(
            encoding="utf-8"
        )
        parent_run_log = (artifact_dir / "parent_run_failure.log").read_text(
            encoding="utf-8"
        )
        _require(
            event_reconciliation.get("status")
            == "RECONCILED_AFTER_WORKER_HARNESS_FAILURE"
            and event_reconciliation.get("parent_process", {}).get(
                "optimizer_instances"
            ) == 1
            and event_reconciliation.get("parent_process", {}).get(
                "optimizer_updates"
            ) == 4
            and event_reconciliation.get("failed_recovery_process")
            == events["failed_recovery_process"]
            and event_reconciliation.get("successful_recovery_process")
            == events["independent_recovery_process"]
            and event_reconciliation.get("total") == reconciled_events
            and execution.get("optimizer_event_reconciled_after_worker_failure") is True,
            "Windows consumer rejects the optimizer event reconciliation after the failed worker",
        )
        _require(
            "selected endpoint Adam state differs: expected (47003, 52003), got (47000, 52000)"
            in failed_worker_log
            and "independent-process recovery worker failed with exit 1" in parent_run_log,
            "Windows consumer rejects event reconciliation without the preserved worker and parent failure traces",
        )
    _require(
        negative.get("status") == "ALL_REJECTED_BEFORE_OPTIMIZER_STEP"
        and set(negative.get("cases", {}))
        == {"detach_rec", "detach_ssim", "detach_rec_and_ssim"}
        and all(
            case.get("status") == "REJECTED_BEFORE_OPTIMIZER_STEP"
            and case.get("optimizer_update_calls") == 0
            and case.get("model_unchanged") is True
            and case.get("adam_unchanged") is True
            for case in negative["cases"].values()
        )
        and negative["cases"]["detach_rec"].get("rejection", "").find(
            "reconstruction CMOS loss is detached"
        ) >= 0
        and "SSIM CMOS loss is detached" in negative["cases"]["detach_ssim"].get(
            "rejection", ""
        )
        and negative["cases"]["detach_rec_and_ssim"].get(
            "non_cmos_total_gradient_norm", 0.0
        ) > 0.0,
        "Windows consumer rejects missing, weak, or post-update CMOS gradient negative controls",
    )

    direct = _load_torch(artifact_dir / "engineering_direct_branch.pt")
    restored = _load_torch(artifact_dir / "engineering_recovery_branch.pt")
    recovery_input = _load_torch(artifact_dir / "engineering_recovery_input.pt")
    source_checkpoint = torch.load(
        _path(repair_config["sources"]["terminal_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    _require(
        source_checkpoint.get("run_id")
        == repair_config["sources"]["expected_training_run_id"]
        and source_checkpoint.get("completed_global_step") == 52_000
        and source_checkpoint.get("optimizer_steps")
        == {"encoder_adam_step": 47_000, "generator_adam_step": 52_000}
        and recovery_input.get("task_id") == REPAIR_TASK_ID
        and recovery_input.get("diagnostic_only_no_candidate") is True
        and recovery_input.get("completed_updates") == 3
        and recovery_input.get("completed_global_step") == 52_003
        and recovery_input.get("optimizer_steps") == (47_003, 52_003)
        and torch.equal(direct.get("rows"), restored.get("rows"))
        and torch.equal(direct.get("rows"), recovery_input.get("next_batch_rows"))
        and direct.get("source_checkpoint_run_id") == source_checkpoint["run_id"]
        and restored.get("source_checkpoint_run_id") == source_checkpoint["run_id"]
        and direct.get("optimizer_steps") == restored.get("optimizer_steps")
        == (47_004, 52_004),
        "Windows consumer rejects a recovery branch bound to another checkpoint, step, or batch",
    )
    model_close, model_delta = _independent_tree_close(
        direct["model"], restored["model"],
        rtol=ENGINEERING_RECOVERY_RTOL, atol=ENGINEERING_RECOVERY_ATOL,
    )
    adam_close, adam_delta = _independent_tree_close(
        direct["optimizer"], restored["optimizer"],
        rtol=ENGINEERING_RECOVERY_RTOL, atol=ENGINEERING_RECOVERY_ATOL,
    )
    loss_close = all(
        math.isclose(
            float(direct["loss"][key]),
            float(restored["loss"][key]),
            rel_tol=ENGINEERING_RECOVERY_RTOL,
            abs_tol=ENGINEERING_RECOVERY_ATOL,
        )
        for key in direct["loss"]
    )
    rng_identical = _exact_tree_equal(
        {"cpu": direct["cpu_rng"], "cuda": direct["cuda_rng"]},
        {"cpu": restored["cpu_rng"], "cuda": restored["cuda_rng"]},
    )
    _require(
        model_close and adam_close and loss_close and rng_identical
        and saved_recovery.get("passed") is True
        and all(saved_recovery.get("checks", {}).values()),
        "independent NumPy/CPU recovery consumer found a direct-versus-restored mismatch",
    )

    base_config_path = _path(repair_config["sources"]["base_config"]).resolve()
    base_config = _read_config(base_config_path)
    _require(
        _path(repair_config["sources"]["old_closure_backup"]).resolve()
        == BACKUP_DIR.resolve()
        and _path(repair_config["sources"]["terminal_projection_backup"]).resolve()
        == _path(base_config["sources"]["terminal_projection_backup"]).resolve(),
        "repair sources do not name the fixed consumed D/P0/P1 and P1 source artifacts",
    )
    optimizer_audit = _OptimizerEventAudit()
    with optimizer_audit:
        assets = _load_assets(base_config, torch.device("cpu"))
        _require(
            training_evidence_scope == assets["training_evidence_scope"],
            "Windows consumer rejects historical D0 evidence misbound as E1G1 terminal evidence",
        )
        generation = assets["generation"]
        labels = generation["labels"].long().cpu()
        source_train_labels = assets["runtime"]["labels"].long().cpu()
        schedule = torch.load(
            _path(repair_config["sources"]["training_schedule"]),
            map_location="cpu",
            weights_only=False,
        )
        _require(
            torch.equal(schedule, source_checkpoint["schedule"])
            and torch.isin(schedule[:4], assets["runtime"]["train_indices"].long().cpu()).all().item()
            and all(
                torch.equal(source_train_labels.index_select(0, row), torch.arange(BATCH_SIZE))
                for row in schedule[:4]
            ),
            "Windows consumer rejects training schedule rows or their source TRAIN role",
        )
        witness_review = _validate_repair_witnesses(
            artifact_dir,
            schedule,
            assets["runtime"]["labels"].long().cpu(),
            source_checkpoint,
        )
        reinjection = _validate_p1_reinjection_controls(
            _load_torch(artifact_dir / "p1_actual_ols_reinjection.pt"), assets
        )
        cross_grid = _recompute_saved_cross_grid_controls(
            assets,
            _path(repair_config["sources"]["old_closure_backup"]).resolve(),
        )
        finite_difference = _reused_response_finite_difference_record(repair_config)
        _require(
            _read_json(artifact_dir / "reused_response_finite_difference.json")
            == finite_difference,
            "Windows consumer rejects a missing or altered saved finite-difference record",
        )
        _verify_saved(
            base_config,
            BACKUP_DIR,
            review_dir,
            optimizer_audit=optimizer_audit,
            cross_grid_recalculation=cross_grid,
        )
        optimizer_audit.require_zero("independent Windows result review")

    final_decision = _read_json(review_dir / "recomputed_final_decision.json")
    independent_review_path = review_dir / "independent_review.json"
    independent_review = _read_json(independent_review_path)
    _require(
        final_decision.get("decision") == independent_review.get("decision")
        and final_decision.get("decision")
        in {
            "BOTH_FULL_PATHS_WITHIN_PREDECLARED_MARGIN",
            "ONLY_ONE_FULL_PATH_WITHIN_PREDECLARED_MARGIN",
            "NEITHER_FULL_PATH_WITHIN_PREDECLARED_MARGIN",
        },
        "repair reviewer cannot consume a unique finite D/P0/P1 path decision",
    )
    repair_review = {
        "task_id": REPAIR_TASK_ID,
        "passed": True,
        "path_decision": final_decision["decision"],
        "engineering_update_calls": 5,
        "optimizer_events": events["total"],
        "rollback": rollback,
        "read_only_windows_optimizer_events": optimizer_audit.snapshot(),
        "negative_controls": negative,
        "recovery_comparison": {
            "model_max_abs_delta": model_delta,
            "adam_max_abs_delta": adam_delta,
            "loss_matches": loss_close,
            "rng_identical": rng_identical,
            "checks": saved_recovery["checks"],
        },
        "witness_review": witness_review,
        "training_evidence_scope": training_evidence_scope,
        "p1_reinjection": reinjection,
        "saved_cross_grid_recomputation": cross_grid,
        "finite_difference": finite_difference,
        "historical_optimizer_instance_count": "UNKNOWN_UNINSTRUMENTED",
        "historical_optimizer_update_count": "UNKNOWN_UNINSTRUMENTED",
        "historical_optimizer_zero_claim_invalidated": True,
        "historical_population_role": repair_config["sources"]["source_population_role_recorded_as"],
        "corrected_population_role": repair_config["sources"]["generation_population_role"],
        "official_test_reads": 0,
        "scientific_updates": 0,
        "source_checkpoint_unchanged": execution["source_checkpoint_unchanged"],
    }
    _write_json(review_dir / "repair_independent_review.json", repair_review, exclusive=True)
    independent_review.update(
        {
            "repair_review": "repair_independent_review.json",
            "historical_optimizer_instance_count": "UNKNOWN_UNINSTRUMENTED",
            "historical_optimizer_update_count": "UNKNOWN_UNINSTRUMENTED",
            "historical_zero_instance_claim_invalidated": True,
            "corrected_population_role": "consumed_generation_diagnostic",
            "repair_read_only_optimizer_events": optimizer_audit.snapshot(),
            "repair_optimizer_events": events["total"],
        }
    )
    _write_json(independent_review_path, independent_review)
    _write_repair_closeout(
        review_dir / "closeout_report.md",
        final_decision,
        execution,
        {
            **saved_recovery,
            "recovery_rtol": ENGINEERING_RECOVERY_RTOL,
            "recovery_atol": ENGINEERING_RECOVERY_ATOL,
        },
        {
            "tolerance": CROSS_GRID_TOLERANCE,
            "p1_reinjection": reinjection,
        },
        finite_difference,
    )
    _write_json(
        review_dir / "independent_numpy_cross_grid_recalculation.json",
        cross_grid,
        exclusive=True,
    )
    _write_json(
        review_dir / "next_stage_independent_population_plan.json",
        {
            "status": "DRAFT_ONLY_NOT_AUTHORIZED",
            "question": "Do the fixed D, P0, and P1 paths retain direct-OLS semantic performance on a new independent generation population?",
            "frozen_endpoint": "d88d5bfc2b29488884a88512152815d6:step_52000",
            "rows": 2_000,
            "rows_per_class": 200,
            "conditions": ["D_DIRECT_OLS", "P0_E1G1", "P1_E1G1"],
            "paired_targets_across_conditions": True,
            "new_prior_samples": 2_000,
            "new_prior_samples_performed_in_current_task": 0,
            "training_updates": 0,
            "decision": "retain both paths only if ED and EC paired bootstrap lower bounds are at least -0.02 and every class diversity ratio is at least 0.80",
            "bootstrap_resamples": BOOTSTRAP_COUNT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "stop_point": "one independent-population decision; no automatic training",
        },
        exclusive=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument(
        "--projection-source-dir",
        type=Path,
        help="explicit P1 source directory; defaults to the configured backup path",
    )
    parser.add_argument(
        "--projection-review-file",
        type=Path,
        help="explicit P1 independent-review JSON; defaults to the configured review path",
    )
    parser.add_argument(
        "--training-preflight-file",
        type=Path,
        help="explicit saved training preflight JSON; defaults to the configured backup path",
    )
    parser.add_argument(
        "--gradient-diagnostics-file",
        type=Path,
        help="explicit saved terminal gradient JSON; defaults to the configured backup path",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--measure", action="store_true")
    mode.add_argument("--verify-saved", action="store_true")
    mode.add_argument("--repair-evidence", action="store_true")
    mode.add_argument("--repair-finalize", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--verify-repair-saved", action="store_true")
    mode.add_argument("--repair-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    if (
        args.repair_evidence
        or args.repair_finalize
        or args.verify_repair_saved
        or args.repair_worker
    ):
        repair_config = _read_repair_config(config_path)
        _require(
            not args.resume and args.projection_source_dir is None
            and args.projection_review_file is None
            and args.training_preflight_file is None
            and args.gradient_diagnostics_file is None,
            "repair modes use the explicit repair config and do not accept measurement overrides",
            ValueError,
        )
        _require(args.artifact_dir is not None,
                 "repair mode requires an explicit --artifact-dir", ValueError)
        artifact_dir = args.artifact_dir.resolve()
        if args.repair_evidence:
            _require(args.review_dir is None,
                     "--review-dir only applies to a saved-result review", ValueError)
            _run_repair_evidence(repair_config, config_path, artifact_dir)
        elif args.repair_finalize:
            _require(args.review_dir is None,
                     "repair finalization does not accept --review-dir", ValueError)
            _finalize_repair_after_worker_recovery(repair_config, artifact_dir)
        elif args.repair_worker:
            _require(args.review_dir is None,
                     "repair worker does not accept --review-dir", ValueError)
            _run_repair_worker(repair_config, config_path, artifact_dir)
        else:
            review_dir = (args.review_dir or REPAIR_REVIEW_DIR).resolve()
            _verify_repair_saved(
                repair_config,
                config_path,
                artifact_dir,
                review_dir,
            )
    else:
        _require(not args.repair_worker,
                 "repair worker requires the repair config", ValueError)
        config = _read_config(config_path)
        artifact_dir = (args.artifact_dir or OUTPUT_DIR).resolve()
        optimizer_audit = _OptimizerEventAudit()
        with optimizer_audit:
            if args.measure:
                _require(
                    args.review_dir is None,
                    "--review-dir only applies to --verify-saved",
                    ValueError,
                )
                _require(
                    artifact_dir == OUTPUT_DIR.resolve(),
                    f"measurement output must be the explicit attempt_01 path: {OUTPUT_DIR}",
                    ValueError,
                )
                _measure(
                    config,
                    artifact_dir,
                    resume=args.resume,
                    projection_source_dir=args.projection_source_dir,
                    projection_review_path=args.projection_review_file,
                    training_preflight_path=args.training_preflight_file,
                    gradient_diagnostics_path=args.gradient_diagnostics_file,
                    optimizer_audit=optimizer_audit,
                )
            else:
                _require(not args.resume, "--resume only applies to --measure", ValueError)
                review_dir = (args.review_dir or REVIEW_DIR).resolve()
                _verify_saved(
                    config,
                    artifact_dir,
                    review_dir,
                    projection_source_dir=args.projection_source_dir,
                    projection_review_path=args.projection_review_file,
                    training_preflight_path=args.training_preflight_file,
                    gradient_diagnostics_path=args.gradient_diagnostics_file,
                    optimizer_audit=optimizer_audit,
                )
        optimizer_audit.require_zero("read-only review process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
