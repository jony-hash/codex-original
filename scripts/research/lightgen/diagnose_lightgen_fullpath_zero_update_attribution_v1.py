"""Attribute a frozen full-path semantic screen with no parameter updates."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.image_readout import (
    resize_view as lightgen_resize_view,
    ssim_batch as lightgen_ssim_batch,
)
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from scripts.research.lightgen import execute_lightgen_g1_fixed_frame_semantic_v1 as frame
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as base
from scripts.research.lightgen import execute_lightgen_recent_optical_fullpath_semantic_v1 as recent
from scripts.research.lightgen import execute_lightgen_small_decoder_native_gradient_ab_v1 as decoder


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_FULLPATH_ZERO_UPDATE_ATTRIBUTION_V1"
CONSUMED_POPULATION_ROLE = "consumed_diagnostic_population"
CONFIG_PATH = ROOT / "configs/research/lightgen/fullpath_zero_update_attribution_v1.yaml"
OUTPUT_DIR = ROOT / "outputs/20261031_lightgen_fullpath_zero_update_attribution_v1/runs/attempt_05"
CONDITION_ORDER = ("E0G0", "E1G1", "E0G1", "E1G0")
CONDITION_PHASE_STEPS = {
    "E0G0": {"encoder": 0, "generator": 0},
    "E1G1": {"encoder": 2_000, "generator": 2_000},
    "E0G1": {"encoder": 0, "generator": 2_000},
    "E1G0": {"encoder": 2_000, "generator": 0},
}
STEP_COUNTS = (0, 500, 1_000, 1_500, 2_000)
BOOTSTRAP_COUNT = 2_000
BOOTSTRAP_SEED = 202610310970
CROSS_GRID_TOLERANCE = 1.0e-4
FD_STEPS = (1.0e-2, 5.0e-3)


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_new(path: Path, value: Any) -> None:
    _require(not path.exists(), f"refusing to replace existing evidence: {path}", FileExistsError)
    path.parent.mkdir(parents=True, exist_ok=True)
    recent._save_torch(path, value, new=True)


def _same_state(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
            left.detach().cpu(), right.detach().cpu()
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _read_config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == CONFIG_PATH.resolve(), "only the registered attribution config is allowed",
             ValueError)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict)
             and config.get("schema") == "lightgen_fullpath_zero_update_attribution.v1"
             and config.get("task_id") == TASK_ID
             and config.get("evidence_tier") == "diagnostic_only"
             and config.get("population") == {
                 "use_saved_final_population_only": True,
                 "rows": 2_000,
                 "rows_per_class": 200,
                 "official_test_reads": 0,
             }
             and config.get("diagnostics", {}).get("crossed_conditions") == ["E0G1", "E1G0"]
             and config["diagnostics"].get("production_grid") == 2048
             and config["diagnostics"].get("reference_grid") == 4096
             and config["diagnostics"].get("positive_controls_per_class") == 2
             and config["diagnostics"].get("existing_cross_grid_tolerance") == CROSS_GRID_TOLERANCE
             and config["diagnostics"].get("fixed_train_batches") == 10
             and config["diagnostics"].get("batch_size") == 10
             and config["diagnostics"].get("finite_difference_steps_rad") == list(FD_STEPS)
             and config["diagnostics"].get("bootstrap_resamples") == BOOTSTRAP_COUNT
             and config["diagnostics"].get("bootstrap_seed") == BOOTSTRAP_SEED
             and config["diagnostics"].get("minimum_diversity_retention") == 0.8,
             "registered zero-update attribution contract changed", ValueError)
    _require(config["output"].get("authorized_optimizer_updates") == 0
             and config["output"].get("run_dir") == OUTPUT_DIR.relative_to(ROOT).as_posix()
             and config["sources"].get("source_run")
             == "outputs/20261030_lightgen_recent_optical_fullpath_semantic_v1/runs/attempt_01"
             and config["sources"].get("verified_backup")
             == "outputs/20261030_lightgen_recent_optical_fullpath_semantic_v1/runs/autodl_backup_attempt_01",
             "source, output, or zero-update scope changed", ValueError)
    _require(config["sources"].get("expected_training_run_id")
             == "d88d5bfc2b29488884a88512152815d6"
             and config["sources"].get("expected_population_seed_base") == 202610300930,
             "source run or saved population identity changed", ValueError)
    return config


def _source_paths(config: Mapping[str, Any], *, use_verified_backup: bool = False
                 ) -> tuple[dict[str, Any], Path]:
    previous_config_path = ROOT / str(config["sources"]["source_config"])
    previous_config = recent._read_config(previous_config_path)
    source_key = "verified_backup" if use_verified_backup else "source_run"
    source_run = (ROOT / str(config["sources"][source_key])).resolve()
    required = [
        source_run / "final_population_and_commands.pt",
        source_run / "final_reconsumed_outputs.pt",
        source_run / "final_condition_registry.json",
        source_run / "final_decision.json",
        source_run / "scientific_attempt" / "task_state.json",
        source_run / "scientific_attempt" / "training_summary.json",
        source_run / "scientific_attempt" / "training_schedule.pt",
        *(source_run / "scientific_attempt" / f"checkpoint_step_{step:05d}.pt"
          for step in STEP_COUNTS[:-1]),
        source_run / "scientific_attempt" / "checkpoint_step_02000.pt",
        source_run / "scientific_attempt" / "terminal_checkpoint.pt",
        *(source_run / f"nearest_train_{name}.json"
          for name in ("C45k", "selected_initial", "final_candidate")),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    _require(not missing, f"explicit source run is incomplete: {missing}", FileNotFoundError)
    return previous_config, source_run


def _load_saved_assets(config: Mapping[str, Any], device: torch.device, *,
                       use_verified_backup: bool = False
                       ) -> dict[str, Any]:
    previous_config, source_run = _source_paths(
        config, use_verified_backup=use_verified_backup
    )
    runtime = recent._load_inputs(previous_config, device)
    endpoints = recent._load_endpoints(previous_config, runtime)
    selected = {**endpoints["D0"], "name": "D0"}
    attempt = source_run / "scientific_attempt"
    state = json.loads((attempt / "task_state.json").read_text(encoding="utf-8"))
    training = json.loads((attempt / "training_summary.json").read_text(encoding="utf-8"))
    schedule_payload = torch.load(attempt / "training_schedule.pt", map_location="cpu",
                                  weights_only=False)
    schedule = schedule_payload.get("schedule") if isinstance(schedule_payload, Mapping) else schedule_payload
    _require(training.get("task_id") == recent.TASK_ID
             and training.get("selected_endpoint") == "D0"
             and training.get("completed_updates") == 2_000
             and training.get("completed_global_step") == 52_000
             and state.get("completed_updates") == 2_000
             and state.get("authorized_updates") == 2_000
             and state.get("state") == "VALID_FAILED"
             and torch.is_tensor(schedule) and tuple(schedule.shape) == (2_000, 10),
             "prior 2,000-step D0 training identity or schedule is incomplete")
    run_id = state.get("run_id")
    _require(isinstance(run_id, str) and len(run_id) == 32
             and run_id == config["sources"]["expected_training_run_id"],
             "prior training run identity is malformed")
    checkpoints = {}
    for updates in STEP_COUNTS:
        name = "terminal_checkpoint.pt" if updates == 2_000 else f"checkpoint_step_{updates:05d}.pt"
        checkpoint = torch.load(attempt / name, map_location="cpu", weights_only=False)
        validation_checkpoint = checkpoint
        stored_source_path = checkpoint.get("source_checkpoint")
        selected_path = Path(selected["path"]).resolve()
        if stored_source_path != str(selected_path):
            remote_prefix = "/root/autodl-tmp/codex-original/"
            _require(use_verified_backup and isinstance(stored_source_path, str)
                     and stored_source_path.startswith(remote_prefix),
                     "saved checkpoint source path differs from the explicit selected endpoint")
            relative_source = stored_source_path[len(remote_prefix):]
            mapped_source = (ROOT / Path(relative_source)).resolve()
            _require(mapped_source == selected_path,
                     "saved AutoDL checkpoint source path does not map to the explicit local endpoint")
            validation_checkpoint = dict(checkpoint)
            validation_checkpoint["source_checkpoint"] = str(selected_path)
        _require(recent._validate_training_checkpoint(
            validation_checkpoint, selected, schedule, previous_config, run_id
        ) == updates, f"saved checkpoint is not step {updates}")
        checkpoints[updates] = checkpoint
    terminal = torch.load(attempt / "terminal_checkpoint.pt", map_location="cpu", weights_only=False)
    step_2000 = checkpoints[2_000]
    _require(recent._states_close(terminal["model"], step_2000["model"])
             and recent._states_close(terminal["optimizer"], step_2000["optimizer"])
             and terminal.get("optimizer_steps") == {
                 "encoder_adam_step": 47_000, "generator_adam_step": 52_000,
             },
             "terminal checkpoint differs from the exact D0 continuation")

    population = torch.load(source_run / "final_population_and_commands.pt",
                            map_location="cpu", weights_only=False)
    source_decision = json.loads((source_run / "final_decision.json").read_text(encoding="utf-8"))
    source_registry = json.loads((source_run / "final_condition_registry.json").read_text(encoding="utf-8"))
    source_reconsumption = json.loads((source_run / "final_reconsumption.json").read_text(encoding="utf-8"))
    cohort, commands = population.get("cohort"), population.get("commands")
    recent._validate_cohort(cohort, 200)
    recent._assert_legal_commands(commands)
    _require(population.get("schema") == f"{recent.TASK_ID}.final_population.v1"
             and population.get("seed_base") == config["sources"]["expected_population_seed_base"]
             and cohort.get("role") == "new_final_population_not_consumed_during_route_or_training"
             and tuple(commands.shape) == (2_000, 1, 200, 200)
             and torch.equal(cohort["labels"], torch.arange(10).repeat_interleave(200)),
             "saved consumed diagnostic population identity or row order changed")
    _require(source_decision.get("status") == "VALID_FAILED"
             and source_decision.get("scientific_updates") == 2_000
             and source_decision.get("final_population_rows") == 2_000
             and source_reconsumption.get("population_rows") == 2_000
             and source_reconsumption.get("same_saved_slm_commands_for_all_conditions") is True
             and set(source_registry.get("conditions", {}))
             == {"C45k", "selected_initial", "final_candidate"},
             "source final decision, condition registry, or reconsumption receipt changed")
    saved = torch.load(source_run / "final_reconsumed_outputs.pt", map_location="cpu",
                       weights_only=False)
    _require(saved.get("task_id") == recent.TASK_ID
             and set(saved.get("conditions", {})) == {"C45k", "selected_initial", "final_candidate"},
             "prior raw optical output bundle is incomplete")
    initial, final = saved["conditions"]["selected_initial"], saved["conditions"]["final_candidate"]
    for label, item in (("C45k", saved["conditions"]["C45k"]),
                        ("selected_initial", initial), ("final_candidate", final)):
        _require(tuple(item["raw_cmos"].shape) == (2_000, 1, 200, 200)
                 and tuple(item["coefficients"].shape) == (2_000, 100)
                 and tuple(item["readouts"].shape) == (2_000, 1, 28, 28)
                 and torch.isfinite(item["raw_cmos"]).all().item(),
                 f"saved {label} outputs are incomplete or nonfinite")
    return {
        "source_run": source_run,
        "previous_config": previous_config,
        "runtime": runtime,
        "endpoints": endpoints,
        "selected": selected,
        "schedule": schedule,
        "checkpoints": checkpoints,
        "population": population,
        "commands": commands.float(),
        "cohort": cohort,
        "saved": saved,
        "initial": initial,
        "final": final,
        "run_id": run_id,
    }


def _condition_checkpoint(assets: Mapping[str, Any], name: str) -> dict[str, Any]:
    start = assets["checkpoints"][0]
    end = assets["checkpoints"][2_000]
    checkpoint = copy.copy(start)
    model = dict(start["model"])
    if name == "E0G0":
        pass
    elif name == "E1G1":
        model = end["model"]
    elif name == "E0G1":
        model["runtime.generator.phase"] = end["model"]["runtime.generator.phase"]
    elif name == "E1G0":
        model["runtime.encoder.phase"] = end["model"]["runtime.encoder.phase"]
    else:
        raise ValueError(f"unsupported crossed condition: {name}")
    checkpoint["model"] = model
    _require(set(model) == set(start["model"])
             and all(torch.equal(value, start["model"][key])
                     for key, value in model.items()
                     if key not in {"runtime.encoder.phase", "runtime.generator.phase"}),
             f"{name} changed a non-phase model component")
    return checkpoint


def _validate_condition_phase_provenance(bundle: Mapping[str, Any],
                                         assets: Mapping[str, Any]) -> None:
    expected_sources = {
        name: {"encoder_step": steps["encoder"],
               "generator_step": steps["generator"],
               "source_training_run_id": assets["run_id"]}
        for name, steps in CONDITION_PHASE_STEPS.items()
    }
    _require(bundle.get("phase_sources") == expected_sources,
             "saved crossed-condition phase source labels are incorrect")
    saved_phases = bundle.get("phase_states")
    _require(isinstance(saved_phases, Mapping) and set(saved_phases) == set(CONDITION_ORDER),
             "saved crossed-condition phase provenance is incomplete")
    for name in CONDITION_ORDER:
        expected = _condition_checkpoint(assets, name)["model"]
        phases = saved_phases[name]
        _require(isinstance(phases, Mapping)
                 and set(phases) == {"encoder", "generator"}
                 and all(torch.is_tensor(phases[key]) for key in ("encoder", "generator"))
                 and torch.equal(phases["encoder"], expected["runtime.encoder.phase"])
                 and torch.equal(phases["generator"], expected["runtime.generator.phase"]),
                 f"saved {name} output is bound to the wrong phase source")


def _validate_saved_population_identity(
    labels: torch.Tensor,
    proposal_ids: torch.Tensor,
    commands: torch.Tensor,
    saved_labels: torch.Tensor,
    saved_proposal_ids: torch.Tensor,
    saved_commands: torch.Tensor,
) -> None:
    _require(all(torch.is_tensor(value) for value in (
        labels, proposal_ids, commands, saved_labels, saved_proposal_ids, saved_commands
    )), "consumer requires tensor labels, proposal IDs, and SLM commands")
    expected_labels = torch.arange(10).repeat_interleave(200)
    _require(tuple(labels.shape) == (2_000,)
             and tuple(proposal_ids.shape) == (2_000,)
             and tuple(commands.shape) == (2_000, 1, 200, 200)
             and tuple(saved_labels.shape) == (2_000,)
             and tuple(saved_proposal_ids.shape) == (2_000,)
             and tuple(saved_commands.shape) == (2_000, 1, 200, 200)
             and saved_commands.dtype == commands.dtype
             and torch.isfinite(saved_commands).all().item()
             and torch.equal(labels.cpu(), expected_labels)
             and torch.equal(saved_labels.cpu(), labels.cpu())
             and torch.equal(saved_proposal_ids.cpu(), proposal_ids.cpu())
             and torch.equal(saved_commands.cpu(), commands.cpu()),
             "consumer rejects an incomplete or misaligned saved population/command bundle")


def _measure_fullpath(checkpoint: Mapping[str, Any], commands: torch.Tensor,
                      runtime: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return recent._measure_commands(checkpoint, commands, runtime)


def _first_two_rows_per_class(labels: torch.Tensor) -> torch.Tensor:
    expected = torch.arange(10).repeat_interleave(200)
    _require(tuple(labels.shape) == (2_000,) and torch.equal(labels.cpu(), expected),
             "fixed positive-control rows require the ordered 200-per-class population")
    return torch.tensor([label * 200 + offset for label in range(10)
                         for offset in range(2)], dtype=torch.long)


def _replay_endpoints(assets: Mapping[str, Any]) -> dict[str, Any]:
    row_indices = _first_two_rows_per_class(assets["cohort"]["labels"])
    runtime = assets["runtime"]
    commands = assets["commands"].index_select(0, row_indices)
    observations = []
    for condition, checkpoint, saved_key in (
        ("E0G0", assets["checkpoints"][0], "selected_initial"),
        ("E1G1", assets["checkpoints"][2_000], "final_candidate"),
    ):
        measured = _measure_fullpath(checkpoint, commands, runtime)
        prior = assets["saved"]["conditions"][saved_key]
        for key in ("raw_cmos", "coefficients", "readouts"):
            expected = prior[key].index_select(0, row_indices)
            error = float((measured[key] - expected).abs().max())
            _require(error <= 1.0e-12, f"saved {condition} {key} does not replay")
            observations.append({"condition": condition, "quantity": key, "max_abs_error": error})
    return {"rows": 20, "rows_per_class": 2,
            "population_indices": row_indices.tolist(),
            "passed": True, "checks": observations}


def _measure_cross_conditions(assets: Mapping[str, Any], output_dir: Path
                              ) -> dict[str, Any]:
    runtime, commands = assets["runtime"], assets["commands"]
    labels = assets["cohort"]["labels"].long().cpu()
    results: dict[str, Any] = {
        "E0G0": {"condition": "E0G0", "raw_cmos": assets["initial"]["raw_cmos"],
                 "coefficients": assets["initial"]["coefficients"],
                 "readouts": assets["initial"]["readouts"]},
        "E1G1": {"condition": "E1G1", "raw_cmos": assets["final"]["raw_cmos"],
                 "coefficients": assets["final"]["coefficients"],
                 "readouts": assets["final"]["readouts"]},
    }
    for name in ("E0G1", "E1G0"):
        checkpoint = _condition_checkpoint(assets, name)
        measured = _measure_fullpath(checkpoint, commands, runtime)
        results[name] = {"condition": name, **measured}
        del checkpoint
        torch.cuda.empty_cache()
    summaries = {}
    for name in CONDITION_ORDER:
        summaries[name] = frame._summary_from_readouts(
            runtime["input_bundle"], assets["cohort"], results[name]["readouts"]
        )
    phase_sources = {
        name: {"encoder_step": CONDITION_PHASE_STEPS[name]["encoder"],
               "generator_step": CONDITION_PHASE_STEPS[name]["generator"],
               "source_training_run_id": assets["run_id"]}
        for name in CONDITION_ORDER
    }
    phase_states = {}
    for name in CONDITION_ORDER:
        state = _condition_checkpoint(assets, name)["model"]
        phase_states[name] = {
            "encoder": state["runtime.encoder.phase"].detach().cpu().clone(),
            "generator": state["runtime.generator.phase"].detach().cpu().clone(),
        }
    _save_new(output_dir / "condition_outputs.pt", {
        "task_id": TASK_ID,
        "population_role": CONSUMED_POPULATION_ROLE,
        "labels": labels,
        "proposal_ids": assets["cohort"]["proposal_ids"].clone(),
        "commands": commands.clone(),
        "phase_sources": phase_sources,
        "phase_states": phase_states,
        "conditions": {name: {key: results[name][key]
                              for key in ("raw_cmos", "coefficients", "readouts")}
                       for name in CONDITION_ORDER},
    })
    return {"outputs": results, "summaries": summaries}


def _saved_decision_values_match(actual: Any, saved: Any) -> bool:
    if isinstance(actual, Mapping) and isinstance(saved, Mapping):
        return (set(actual) == set(saved)
                and all(_saved_decision_values_match(actual[key], saved[key]) for key in actual))
    if isinstance(actual, list) and isinstance(saved, list):
        return (len(actual) == len(saved)
                and all(_saved_decision_values_match(left, right)
                        for left, right in zip(actual, saved, strict=True)))
    if isinstance(actual, bool) or isinstance(saved, bool):
        return isinstance(actual, bool) and isinstance(saved, bool) and actual == saved
    if isinstance(actual, float) or isinstance(saved, float):
        return (isinstance(actual, (int, float)) and isinstance(saved, (int, float))
                and math.isclose(float(actual), float(saved), rel_tol=1.0e-12, abs_tol=1.0e-12))
    return type(actual) is type(saved) and actual == saved


def _reconsume_saved_conditions(output_dir: Path, config: Mapping[str, Any],
                                *, use_verified_backup: bool = False
                                ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the decision from saved commands and actual full-path outputs."""
    saved_config_path = output_dir / "run_config.yaml"
    _require(saved_config_path.is_file()
             and yaml.safe_load(saved_config_path.read_text(encoding="utf-8")) == config,
             "saved run configuration is missing or differs from the registered task")
    assets = _load_saved_assets(
        config, torch.device("cpu"), use_verified_backup=use_verified_backup
    )
    runtime, population, cohort = assets["runtime"], assets["population"], assets["cohort"]
    output_bundle = torch.load(output_dir / "condition_outputs.pt", map_location="cpu",
                               weights_only=False)
    _validate_saved_population_identity(
        cohort.get("labels", torch.empty(0)), cohort.get("proposal_ids", torch.empty(0)),
        population.get("commands", torch.empty(0)),
        output_bundle.get("labels", torch.empty(0)),
        output_bundle.get("proposal_ids", torch.empty(0)),
        output_bundle.get("commands", torch.empty(0)),
    )
    _require(output_bundle.get("task_id") == TASK_ID
             and output_bundle.get("population_role") == CONSUMED_POPULATION_ROLE
             and set(output_bundle.get("conditions", {})) == set(CONDITION_ORDER),
             "saved output bundle lacks a registered condition or population role")
    _validate_condition_phase_provenance(output_bundle, assets)
    summaries = {}
    reconsumed_readouts = {}
    for name in CONDITION_ORDER:
        item = output_bundle["conditions"][name]
        _require(tuple(item.get("raw_cmos", torch.empty(0)).shape) == (2_000, 1, 200, 200)
                 and tuple(item.get("coefficients", torch.empty(0)).shape) == (2_000, 100)
                 and tuple(item.get("readouts", torch.empty(0)).shape) == (2_000, 1, 28, 28)
                 and all(torch.isfinite(item[key]).all().item()
                         for key in ("raw_cmos", "coefficients", "readouts")),
                 f"saved {name} output has missing rows or invalid numeric values")
        readouts = recent.pilot._readout_for_evaluator(
            item["raw_cmos"], runtime["camera_area"], runtime["camera_gain"]
        ).cpu()
        readout_error = float((readouts - item["readouts"]).abs().max())
        _require(readout_error <= 1.0e-12,
                 f"saved {name} evaluator readouts do not reproduce from raw CMOS")
        reconsumed_readouts[name] = readouts
        summaries[name] = frame._summary_from_readouts(
            runtime["input_bundle"], cohort, readouts
        )
    decision = _attribution_decision(cohort["labels"], summaries)
    saved_decision = json.loads((output_dir / "attribution_decision.json").read_text(encoding="utf-8"))
    _require(_saved_decision_values_match(decision, saved_decision),
             "reconsumed metrics differ from the saved attribution decision")
    return decision, {"population_rows": 2_000, "conditions": list(CONDITION_ORDER),
                      "recomputed_from_saved_readouts": True,
                      "readouts_recomputed_from_raw_cmos": True,
                      "max_readout_reproduction_error": max(
                          float((reconsumed_readouts[name]
                                 - output_bundle["conditions"][name]["readouts"]).abs().max())
                          for name in CONDITION_ORDER
                      ),
                      "commands_match_population": True,
                      "route": decision["route"], "official_test_reads": 0}


def _paired_bootstrap(labels: torch.Tensor, baseline: Sequence[int], candidate: Sequence[int],
                      *, seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_COUNT
                      ) -> dict[str, Any]:
    y = labels.detach().cpu().numpy().astype(np.int64)
    a = np.asarray(baseline, dtype=np.int64)
    b = np.asarray(candidate, dtype=np.int64)
    _require(y.shape == a.shape == b.shape and y.shape == (2_000,)
             and np.array_equal(np.bincount(y, minlength=10), np.full(10, 200)),
             "paired bootstrap requires the complete 2,000-row balanced population")
    correct_a, correct_b = (a == y), (b == y)
    rng = np.random.default_rng(seed)
    draws = []
    per_class = {}
    point = []
    for label in range(10):
        ix = np.flatnonzero(y == label)
        da, db = correct_a[ix].astype(np.float64), correct_b[ix].astype(np.float64)
        point.append(float(db.mean() - da.mean()))
        per_class[str(label)] = {"baseline_recall": float(da.mean()),
                                 "candidate_recall": float(db.mean()),
                                 "delta": float(db.mean() - da.mean())}
        draws.append(rng.integers(0, len(ix), size=(resamples, len(ix))))
    delta = np.zeros(resamples, dtype=np.float64)
    for ix, sample in zip((np.flatnonzero(y == label) for label in range(10)), draws, strict=True):
        delta += (correct_b[ix][sample].mean(axis=1)
                  - correct_a[ix][sample].mean(axis=1)) / 10.0
    lower, upper = np.quantile(delta, (0.025, 0.975), method="linear")
    return {"point_macro_recall_delta": float(np.mean(point)),
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "confidence_level": 0.95,
            "paired_class_stratified_ci95": [float(lower), float(upper)],
            "lower_bound_positive": bool(lower > 0.0), "per_class": per_class}


def _factorial_interaction_bootstrap(
    labels: torch.Tensor,
    predictions: Mapping[str, Sequence[int]],
    evaluator: str,
    *, seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_COUNT,
) -> dict[str, Any]:
    y = labels.detach().cpu().numpy().astype(np.int64)
    correct = {name: (np.asarray(values, dtype=np.int64) == y).astype(np.float64)
               for name, values in predictions.items()}
    _require(y.shape == (2_000,) and set(correct) == set(CONDITION_ORDER)
             and np.array_equal(np.bincount(y, minlength=10), np.full(10, 200)),
             "factorial interaction requires all four predictions on the complete population")
    rng = np.random.default_rng(seed)
    boot = np.zeros(resamples, dtype=np.float64)
    class_effects = {}
    point_effects = []
    for label in range(10):
        ix = np.flatnonzero(y == label)
        effect = ((correct["E1G1"][ix].mean() - correct["E0G1"][ix].mean())
                  - (correct["E1G0"][ix].mean() - correct["E0G0"][ix].mean()))
        class_effects[str(label)] = float(effect)
        point_effects.append(float(effect))
        sample = rng.integers(0, len(ix), size=(resamples, len(ix)))
        boot += (
            (correct["E1G1"][ix][sample].mean(axis=1)
             - correct["E0G1"][ix][sample].mean(axis=1))
            - (correct["E1G0"][ix][sample].mean(axis=1)
               - correct["E0G0"][ix][sample].mean(axis=1))
        ) / 10.0
    lower, upper = np.quantile(boot, (0.025, 0.975), method="linear")
    return {"evaluator": evaluator,
            "definition": "(E1G1-E0G1)-(E1G0-E0G0) in class macro recall",
            "point_macro_recall_interaction": float(np.mean(point_effects)),
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "confidence_level": 0.95,
            "class_stratified_ci95": [float(lower), float(upper)],
            "per_class_interaction": class_effects}


def _evaluator_summary(summary: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return summary["conditions"]["direct_decoder"]["evaluators"][key]


def _diversity_ratios(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    base_rows = reference["same_class_diversity"]["direct_decoder"]["by_class"]
    candidate_rows = candidate["same_class_diversity"]["direct_decoder"]["by_class"]
    return {str(label): float(candidate_rows[str(label)]["mean_pairwise_normalized_l2"])
            / max(float(base_rows[str(label)]["mean_pairwise_normalized_l2"]), 1.0e-30)
            for label in range(10)}


def _condition_metrics(labels: torch.Tensor, summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {name: {key: {
        "macro_recall": _evaluator_summary(summaries[name], key)["macro_recall"],
        "per_class_recall": _evaluator_summary(summaries[name], key)["per_class_recall"],
        "predicted_labels": _evaluator_summary(summaries[name], key)["predicted_labels"],
    } for key in ("ED", "EC")} for name in CONDITION_ORDER}


def _attribution_decision(labels: torch.Tensor, summaries: Mapping[str, Mapping[str, Any]],
                          *, resamples: int = BOOTSTRAP_COUNT,
                          seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    metrics = _condition_metrics(labels, summaries)
    pairs = {
        "restore_old_encoder": ("E1G1", "E0G1"),
        "restore_old_generator": ("E1G1", "E1G0"),
        "encoder_effect_at_old_generator": ("E0G0", "E1G0"),
        "encoder_effect_at_final_generator": ("E0G1", "E1G1"),
        "generator_effect_at_old_encoder": ("E0G0", "E0G1"),
        "generator_effect_at_final_encoder": ("E1G0", "E1G1"),
    }
    effects: dict[str, Any] = {}
    for name, (baseline, candidate) in pairs.items():
        effects[name] = {}
        for evaluator in ("ED", "EC"):
            effects[name][evaluator] = _paired_bootstrap(
                labels,
                metrics[baseline][evaluator]["predicted_labels"],
                metrics[candidate][evaluator]["predicted_labels"],
                seed=seed, resamples=resamples,
            )
        effects[name]["diversity_retention"] = _diversity_ratios(
            summaries[baseline], summaries[candidate]
        )
    interactions = {
        evaluator: _factorial_interaction_bootstrap(
            labels,
            {name: metrics[name][evaluator]["predicted_labels"] for name in CONDITION_ORDER},
            evaluator,
            seed=seed,
            resamples=resamples,
        )
        for evaluator in ("ED", "EC")
    }
    enc_old = effects["encoder_effect_at_old_generator"]["ED"]["paired_class_stratified_ci95"]
    enc_final = effects["encoder_effect_at_final_generator"]["ED"]["paired_class_stratified_ci95"]
    gen_old = effects["generator_effect_at_old_encoder"]["ED"]["paired_class_stratified_ci95"]
    gen_final = effects["generator_effect_at_final_encoder"]["ED"]["paired_class_stratified_ci95"]
    enc_old_point = effects["encoder_effect_at_old_generator"]["ED"]["point_macro_recall_delta"]
    enc_final_point = effects["encoder_effect_at_final_generator"]["ED"]["point_macro_recall_delta"]
    gen_old_point = effects["generator_effect_at_old_encoder"]["ED"]["point_macro_recall_delta"]
    gen_final_point = effects["generator_effect_at_final_encoder"]["ED"]["point_macro_recall_delta"]
    reversals = {
        "encoder": enc_old_point * enc_final_point < 0.0,
        "generator": gen_old_point * gen_final_point < 0.0,
    }
    interval_reversals = {
        "encoder": (enc_old[0] > 0.0 and enc_final[1] < 0.0)
                    or (enc_old[1] < 0.0 and enc_final[0] > 0.0),
        "generator": (gen_old[0] > 0.0 and gen_final[1] < 0.0)
                     or (gen_old[1] < 0.0 and gen_final[0] > 0.0),
    }
    restoration_support = {}
    for name in ("restore_old_encoder", "restore_old_generator"):
        ed = effects[name]["ED"]
        ec_delta = effects[name]["EC"]["point_macro_recall_delta"]
        diversity = effects[name]["diversity_retention"]
        restoration_support[name] = bool(
            ed["paired_class_stratified_ci95"][0] > 0.0
            and ec_delta >= 0.0
            and all(value >= 0.8 for value in diversity.values())
        )
    support_values = list(restoration_support.values())
    if all(support_values) or any(reversals.values()):
        route = "MIXED_EFFECT"
    elif support_values[0] and not support_values[1]:
        route = "ENCODER_PRIORITY"
    elif support_values[1] and not support_values[0]:
        route = "GENERATOR_PRIORITY"
    else:
        route = "UNRESOLVED"
    return {"task_id": TASK_ID, "population_role": "consumed_diagnostic_population",
            "question": "Does the saved population support prioritizing Encoder, Generator, or neither?",
            "population_rows": 2_000, "rows_per_class": {str(label): 200 for label in range(10)},
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "effect_orientation": "delta is second listed condition minus first listed condition",
            "effects": effects, "factorial_interaction": interactions,
            "restoration_support": restoration_support,
            "module_effect_reversals_by_point_estimate": reversals,
            "module_effect_reversals_with_separated_intervals": interval_reversals,
            "route": route,
            "optimizer_updates": 0, "human_readability": "not_measured",
            "hardware_deployment": "not_measured", "full_G1_distribution": "not_established"}


def _fixed_train_batches(assets: Mapping[str, Any]) -> torch.Tensor:
    schedule = assets["schedule"]
    rows = schedule[:10].long().cpu()
    pixels = assets["runtime"]["pixels"]
    labels = assets["runtime"]["labels"]
    train_indices = assets["runtime"]["train_indices"]
    train_set = set(int(value) for value in train_indices.tolist())
    _require(tuple(rows.shape) == (10, 10)
             and all(int(row) in train_set for row in rows.flatten())
             and torch.equal(labels.index_select(0, rows.flatten()).reshape(10, 10),
                             torch.arange(10).expand(10, 10)),
             "first ten saved batches differ from the balanced TRAIN schedule")
    _require(pixels.shape[0] >= 60_000, "MNIST training source is incomplete")
    return rows


def _fixed_development_rows(assets: Mapping[str, Any]) -> torch.Tensor:
    labels, dev_indices = assets["runtime"]["labels"], assets["runtime"]["dev_indices"]
    selected = []
    for label in range(10):
        candidates = dev_indices[labels[dev_indices] == label]
        _require(candidates.numel() >= 20, f"DEV class {label} has fewer than 20 rows")
        selected.extend(int(value) for value in candidates[:20].tolist())
    rows = torch.tensor(selected, dtype=torch.long)
    selected_labels = labels.index_select(0, rows)
    _require(tuple(rows.shape) == (200,)
             and torch.equal(torch.bincount(selected_labels, minlength=10),
                             torch.full((10,), 20))
             and bool(torch.isin(rows, dev_indices).all()),
             "fixed DEV reconstruction subset is incomplete or includes non-DEV rows")
    return rows


def _development_reconstruction_diagnostic(assets: Mapping[str, Any]) -> dict[str, Any]:
    runtime = assets["runtime"]
    rows = _fixed_development_rows(assets)
    labels = runtime["labels"].index_select(0, rows).cpu()
    per_endpoint = {}
    for name, checkpoint in (("step_00000", assets["checkpoints"][0]),
                             ("step_02000", assets["checkpoints"][2_000])):
        model = recent._new_endpoint_model(checkpoint, recent.GRID_PRODUCTION, runtime)
        rec_values, ssim_values, readouts = [], [], []
        with torch.inference_mode():
            for start in range(0, len(rows), 10):
                batch_rows = rows[start:start + 10]
                target_native = image_batch(
                    runtime["pixels"], batch_rows, "small", runtime["device"]
                )
                target28 = runtime["pixels"].index_select(
                    0, batch_rows
                ).unsqueeze(1).to(device=runtime["device"], dtype=torch.float32) / 255.0
                result = base._forward(model, target_native)
                gained = result["physical_w_per_pixel"] * float(runtime["camera_gain"])
                rec_each = (gained - target_native).square().mean((1, 2, 3))
                view = lightgen_resize_view(gained)
                ssim_each = lightgen_ssim_batch(view, target28)
                _require(tuple(rec_each.shape) == tuple(ssim_each.shape) == (len(batch_rows),)
                         and torch.isfinite(rec_each).all().item()
                         and torch.isfinite(ssim_each).all().item(),
                         f"{name} DEV reconstruction batch is malformed")
                rec_values.append(rec_each.detach().cpu())
                ssim_values.append(ssim_each.detach().cpu())
                readouts.append(view.detach().cpu())
        rec_tensor = torch.cat(rec_values)
        ssim_tensor = torch.cat(ssim_values)
        readout_tensor = torch.cat(readouts)
        _require(tuple(rec_tensor.shape) == tuple(ssim_tensor.shape) == (200,)
                 and tuple(readout_tensor.shape) == (200, 1, 28, 28),
                 f"{name} DEV reconstruction omitted fixed rows")
        per_endpoint[name] = {
            "mean_rec": float(rec_tensor.mean()),
            "mean_weighted_rec": float(float(checkpoint["weights"]["rec"]) * rec_tensor.mean()),
            "mean_ssim_similarity": float(ssim_tensor.mean()),
            "per_class": {str(label): {
                "mean_rec": float(rec_tensor[labels == label].mean()),
                "mean_ssim_similarity": float(ssim_tensor[labels == label].mean()),
            } for label in range(10)},
            "per_sample_rec": rec_tensor.tolist(),
            "per_sample_ssim_similarity": ssim_tensor.tolist(),
            "readouts": readout_tensor,
        }
        del model
        torch.cuda.empty_cache()
    initial, terminal = per_endpoint["step_00000"], per_endpoint["step_02000"]
    per_class_delta = {str(label): {
        "rec_delta_terminal_minus_start":
            terminal["per_class"][str(label)]["mean_rec"]
            - initial["per_class"][str(label)]["mean_rec"],
        "ssim_delta_terminal_minus_start":
            terminal["per_class"][str(label)]["mean_ssim_similarity"]
            - initial["per_class"][str(label)]["mean_ssim_similarity"],
    } for label in range(10)}
    return {
        "task_id": TASK_ID, "population_role": "fixed_DEV_reconstruction_diagnostic",
        "rows": 200, "rows_per_class": 20, "official_test_reads": 0,
        "source_rows": rows.tolist(), "labels": labels.tolist(),
        "target_images": runtime["pixels"].index_select(0, rows).float().div(255.0),
        "comparison": {
            "mean_rec_delta_terminal_minus_start":
                terminal["mean_rec"] - initial["mean_rec"],
            "mean_ssim_delta_terminal_minus_start":
                terminal["mean_ssim_similarity"] - initial["mean_ssim_similarity"],
            "per_class": per_class_delta,
        },
        "endpoints": {name: {key: value for key, value in endpoint.items()
                             if key != "readouts"}
                      for name, endpoint in per_endpoint.items()},
        "readouts": {name: endpoint["readouts"] for name, endpoint in per_endpoint.items()},
    }


def _write_development_reconstruction_rows(path: Path,
                                          diagnostic: Mapping[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["population_index", "source_dev_row", "label",
                         "step_00000_rec", "step_02000_rec", "terminal_minus_start_rec",
                         "step_00000_ssim", "step_02000_ssim", "terminal_minus_start_ssim"])
        start = diagnostic["endpoints"]["step_00000"]
        terminal = diagnostic["endpoints"]["step_02000"]
        for index, (source_row, label) in enumerate(zip(
            diagnostic["source_rows"], diagnostic["labels"], strict=True
        )):
            rec0, rec1 = start["per_sample_rec"][index], terminal["per_sample_rec"][index]
            ssim0 = start["per_sample_ssim_similarity"][index]
            ssim1 = terminal["per_sample_ssim_similarity"][index]
            writer.writerow([index, source_row, label, rec0, rec1, rec1 - rec0,
                             ssim0, ssim1, ssim1 - ssim0])


def _write_development_reconstruction_figure(path: Path,
                                             diagnostic: Mapping[str, Any]) -> None:
    rows, labels = diagnostic["source_rows"], diagnostic["labels"]
    readouts = diagnostic["readouts"]
    cell, gap = 40, 6
    header = 22
    image = Image.new("RGB", (3 * cell + 4 * gap + 32, header + len(rows) * (cell + 1)), "white")
    draw = ImageDraw.Draw(image)
    for column, title in enumerate(("target", "step 0", "step 2000")):
        draw.text((32 + gap + column * (cell + gap), 4), title, fill="black")
    for index, label in enumerate(labels):
        target = diagnostic["target_images"][index].clamp(0, 1)
        views = (target, readouts["step_00000"][index, 0],
                 readouts["step_02000"][index, 0])
        y = header + index * (cell + 1)
        if index % 20 == 0:
            draw.text((0, y + 8), f"{label}", fill="black")
        for column, tensor in enumerate(views):
            pixels = (tensor.detach().cpu().numpy() * 255).round().astype(np.uint8)
            tile = Image.fromarray(pixels, mode="L").resize(
                (cell, cell), Image.Resampling.NEAREST
            ).convert("RGB")
            image.paste(tile, (32 + gap + column * (cell + gap), y))
    image.save(path)


def _loss_components_and_gradients(model: torch.nn.Module, pixels: torch.Tensor,
                                   rows: torch.Tensor, runtime: Mapping[str, Any],
                                   checkpoint: Mapping[str, Any]
                                   ) -> tuple[dict[str, float], dict[str, Any], torch.Tensor]:
    device = runtime["device"]
    encoder, generator = base._phase_groups(model)
    parts, ssim, _native_gradient, result = decoder._loss_parts(
        model, pixels, rows, device, checkpoint["scale"].to(device),
        checkpoint["power_reference"].to(device), checkpoint["camera_gain"].to(device),
    )
    names = ("dist", "rec", "op", "ssim")
    weighted = {
        "dist": float(checkpoint["weights"]["dist"]) * parts["dist"],
        "rec": float(checkpoint["weights"]["rec"]) * parts["rec"],
        "op": float(checkpoint["weights"]["op"]) * parts["op"],
        "ssim": float(checkpoint["structural_weight"]) * ssim,
    }
    objective = sum(weighted.values())
    parameters = (encoder, generator)
    gradients: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    for name in names:
        gradients[name] = torch.autograd.grad(weighted[name], parameters, retain_graph=True,
                                              allow_unused=True)
    totals = tuple(sum((gradients[name][index] if gradients[name][index] is not None
                        else torch.zeros_like(parameter)) for name in names)
                   for index, parameter in enumerate(parameters))
    summary: dict[str, Any] = {"weighted_losses": {name: float(value.detach())
                                                    for name, value in weighted.items()},
                               "total": float(objective.detach()),
                               "mean_plane_power_w": float(result["plane_powers_w"].detach().double().mean()),
                               "phase_groups": {}}
    for index, (group_name, parameter) in enumerate((
            ("encoder", encoder), ("generator", generator))):
        group_gradients = {}
        for name in names:
            gradient = gradients[name][index]
            if gradient is None:
                group_gradients[name] = None
                continue
            group_gradients[name] = gradient.detach()
        total_gradient = totals[index].detach()
        total_norm = float(torch.linalg.vector_norm(total_gradient.double()).cpu())
        unit_descent = -total_gradient / max(total_norm, 1.0e-30)
        components = {}
        for name, gradient in group_gradients.items():
            if gradient is None:
                components[name] = {"structural_zero": True, "norm": 0.0,
                                    "directional_derivative_along_total_descent": 0.0}
            else:
                components[name] = {
                    "structural_zero": False,
                    "norm": float(torch.linalg.vector_norm(gradient.double()).cpu()),
                    "directional_derivative_along_total_descent": float(
                        (gradient.double() * unit_descent.double()).sum().cpu()
                    ),
                }
        cosine = {}
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                a, b = group_gradients[left], group_gradients[right]
                if a is None or b is None:
                    cosine[f"{left}:{right}"] = None
                else:
                    denom = torch.linalg.vector_norm(a.double()) * torch.linalg.vector_norm(b.double())
                    cosine[f"{left}:{right}"] = float((a.double() * b.double()).sum().cpu()
                                                        / max(float(denom.cpu()), 1.0e-30))
        summary["phase_groups"][group_name] = {"total_gradient_norm": total_norm,
                                                "components": components,
                                                "pairwise_gradient_cosines": cosine}
    return summary["weighted_losses"] | {"total": summary["total"]}, summary, objective


def _finite_difference_checks(model: torch.nn.Module, pixels: torch.Tensor, rows: torch.Tensor,
                              runtime: Mapping[str, Any], checkpoint: Mapping[str, Any]
                              ) -> list[dict[str, Any]]:
    device = runtime["device"]
    encoder, generator = base._phase_groups(model)
    params = {"encoder": encoder, "generator": generator}

    def objective() -> torch.Tensor:
        parts, ssim, _native_gradient, _result = decoder._loss_parts(
            model, pixels, rows, device, checkpoint["scale"].to(device),
            checkpoint["power_reference"].to(device), checkpoint["camera_gain"].to(device),
        )
        return decoder._objective(parts, ssim, None, checkpoint["weights"],
                                  float(checkpoint["structural_weight"]), None)

    results = []
    for group_name, parameter in params.items():
        value = objective()
        gradient = torch.autograd.grad(value, parameter)[0]
        _require(torch.isfinite(gradient).all().item()
                 and float(torch.linalg.vector_norm(gradient).cpu()) > 0.0,
                 f"{group_name} analytic objective gradient is invalid")
        direction = (gradient.detach().double()
                     / torch.linalg.vector_norm(gradient.detach().double())).to(parameter.dtype)
        original = parameter.detach().clone()
        for epsilon in FD_STEPS:
            plus_state, minus_state = original + epsilon * direction, original - epsilon * direction
            represented = float(((((plus_state.double() - original.double())
                                   + (original.double() - minus_state.double()))
                                  * direction.double()).sum()) * 0.5)
            try:
                with torch.no_grad():
                    parameter.copy_(plus_state)
                plus = float(objective().detach())
                with torch.no_grad():
                    parameter.copy_(minus_state)
                minus = float(objective().detach())
            finally:
                with torch.no_grad():
                    parameter.copy_(original)
            analytic = float((gradient.detach().double() * direction.double()).sum().cpu())
            finite = (plus - minus) / (2.0 * represented)
            relative = abs(analytic - finite) / max(abs(analytic), abs(finite), 1.0e-7)
            _require(represented > 0.0 and math.isfinite(finite)
                     and analytic * finite > 0.0 and relative <= 0.05,
                     f"{group_name} center difference failed at {epsilon}: relative={relative}")
            results.append({"group": group_name, "requested_rad": epsilon,
                            "representable_half_step_rad": represented,
                            "analytic_directional_derivative": analytic,
                            "central_difference": finite, "relative_error": relative})
        _require(torch.equal(parameter.detach(), original),
                 f"{group_name} finite difference did not restore its phase tensor")
    return results


def _gradient_diagnostics(assets: Mapping[str, Any]) -> dict[str, Any]:
    runtime = assets["runtime"]
    rows = _fixed_train_batches(assets)
    results = {}
    optical_config = decoder._read_config(recent._path(
        assets["previous_config"],
        assets["previous_config"]["sources"]["optical_source_config"],
    ))
    for name, checkpoint in (("step_00000", assets["checkpoints"][0]),
                             ("step_02000", assets["checkpoints"][2_000])):
        model = base._model("small", recent.GRID_PRODUCTION, runtime["device"], optical_config)
        model.load_state_dict(checkpoint["model"], strict=True)
        encoder, generator = base._phase_groups(model)
        for parameter in model.parameters():
            parameter.requires_grad_(parameter is encoder or parameter is generator)
        original_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
        per_batch = []
        finite_difference = None
        for batch_index, batch_rows in enumerate(rows):
            model.zero_grad(set_to_none=True)
            losses, components, _objective_value = _loss_components_and_gradients(
                model, runtime["pixels"], batch_rows, runtime, checkpoint
            )
            per_batch.append({"batch": batch_index, "sample_rows": batch_rows.tolist(),
                              "labels": runtime["labels"].index_select(0, batch_rows).tolist(),
                              **components})
            if batch_index == 0:
                finite_difference = _finite_difference_checks(
                    model, runtime["pixels"], batch_rows, runtime, checkpoint
                )
        _require(all(torch.equal(value, model.state_dict()[key])
                     for key, value in original_model.items()),
                 f"{name} gradient measurements changed model state")
        results[name] = {"batches": per_batch, "finite_difference": finite_difference,
                         "model_unchanged": True}
        del model
        torch.cuda.empty_cache()
    return {"task_id": TASK_ID, "optimizer_updates": 0,
            "loss_names": ["dist", "rec", "op", "ssim"],
            "batches_per_checkpoint": 10, "batch_size": 10,
            "sample_schedule_source": "first_ten_batches_of_saved_D0_schedule",
            "endpoints": results}


def _phase_drift(assets: Mapping[str, Any]) -> dict[str, Any]:
    origin = assets["checkpoints"][0]["model"]
    results = {}
    for step, checkpoint in assets["checkpoints"].items():
        results[str(step)] = {}
        for group, key in (("encoder", "runtime.encoder.phase"),
                           ("generator", "runtime.generator.phase")):
            delta = checkpoint["model"][key].double() - origin[key].double()
            circular = torch.atan2(torch.sin(delta), torch.cos(delta))
            results[str(step)][group] = {
                "rms_wrapped_phase_delta_rad": float(circular.square().mean().sqrt()),
                "max_abs_wrapped_phase_delta_rad": float(circular.abs().max()),
                "per_layer_rms_wrapped_phase_delta_rad": [
                    float(layer.square().mean().sqrt()) for layer in circular
                ],
            }
    return results


def _camera_statistics(raw: torch.Tensor, runtime: Mapping[str, Any]) -> dict[str, Any]:
    calibrated = raw.double() * float(runtime["camera_area"]) * float(runtime["camera_gain"])
    clipped = calibrated.clamp(0.0, 1.0)
    total = calibrated.numel()
    return {"raw_cmos_pixel_count": total,
            "raw_cmos_min_w_per_pixel": float(raw.min()),
            "raw_cmos_max_w_per_pixel": float(raw.max()),
            "calibrated_max": float(calibrated.max()),
            "below_zero_fraction": float((calibrated < 0.0).sum()) / total,
            "above_one_fraction": float((calibrated > 1.0).sum()) / total,
            "clipped_to_zero_fraction": float((clipped == 0.0).sum()) / total,
            "clipped_to_one_fraction": float((clipped == 1.0).sum()) / total}


def _modal_power_summary(coefficients: torch.Tensor) -> dict[str, float]:
    per_row = coefficients.detach().cpu().abs().double().square().sum(dim=1)
    return {"definition": "sum(abs(fullpath complex OLS coefficients)^2) per sample",
            "mean": float(per_row.mean()), "median": float(per_row.median()),
            "minimum": float(per_row.min()), "maximum": float(per_row.max())}


def _pairwise_optical_drift(left: Mapping[str, torch.Tensor],
                            right: Mapping[str, torch.Tensor]) -> dict[str, float]:
    left_coeff, right_coeff = left["coefficients"], right["coefficients"]
    _require(left_coeff.shape == right_coeff.shape,
             "paired optical coefficient arrays have different population shapes")
    delta = (left_coeff - right_coeff).abs().double()
    row_l2 = torch.linalg.vector_norm(delta.flatten(1), dim=1)
    left_power = left_coeff.detach().cpu().abs().double().square().sum(dim=1)
    right_power = right_coeff.detach().cpu().abs().double().square().sum(dim=1)
    return {"complex_coefficient_absolute_rms": float(delta.square().mean().sqrt()),
            "mean_per_sample_complex_coefficient_l2": float(row_l2.mean()),
            "mean_total_modal_power_delta": float((right_power - left_power).mean()),
            "total_modal_power_delta_rms": float((right_power - left_power).square().mean().sqrt()),
            "raw_cmos_pixel_rms": float((left["raw_cmos"].double()
                                          - right["raw_cmos"].double()).square().mean().sqrt())}


def _prediction_change_summary(labels: torch.Tensor,
                              baseline: Mapping[str, Any],
                              candidate: Mapping[str, Any]) -> dict[str, Any]:
    y = labels.detach().cpu().long().numpy()
    result = {}
    for evaluator in ("ED", "EC"):
        a = np.asarray(_evaluator_summary(baseline, evaluator)["predicted_labels"], dtype=np.int64)
        b = np.asarray(_evaluator_summary(candidate, evaluator)["predicted_labels"], dtype=np.int64)
        _require(a.shape == b.shape == y.shape, "saved prediction rows are incomplete")
        improved = np.flatnonzero((a != y) & (b == y))
        degraded = np.flatnonzero((a == y) & (b != y))
        result[evaluator] = {
            "improved_count": int(improved.size),
            "degraded_count": int(degraded.size),
            "net_correct_count_change": int(improved.size - degraded.size),
            "improved_population_indices": improved.tolist(),
            "degraded_population_indices": degraded.tolist(),
            "per_class": {str(label): {
                "improved": int(np.sum((y[improved] == label))),
                "degraded": int(np.sum((y[degraded] == label))),
            } for label in range(10)},
        }
    ed = np.asarray(_evaluator_summary(candidate, "ED")["predicted_labels"], dtype=np.int64)
    ec = np.asarray(_evaluator_summary(candidate, "EC")["predicted_labels"], dtype=np.int64)
    disagreement = np.flatnonzero(ed != ec)
    result["ED_EC_disagreement"] = {
        "count": int(disagreement.size),
        "population_indices": disagreement.tolist(),
        "per_class": {str(label): int(np.sum(y[disagreement] == label))
                      for label in range(10)},
    }
    return result


def _nearest_training_summary(source_run: Path, labels: torch.Tensor) -> dict[str, Any]:
    result = {}
    expected = labels.detach().cpu().long().tolist()
    for condition in ("C45k", "selected_initial", "final_candidate"):
        path = source_run / f"nearest_train_{condition}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows")
        _require(payload.get("rows") is not None and len(rows) == 2_000
                 and all(row.get("population_index") == index
                         and row.get("label") == expected[index]
                         and math.isfinite(float(row["nearest_same_class_l2_over_sqrt784"]))
                         for index, row in enumerate(rows)),
                 f"saved nearest-TRAIN evidence for {condition} is incomplete or reordered")
        distances = np.asarray([row["nearest_same_class_l2_over_sqrt784"] for row in rows],
                               dtype=np.float64)
        per_class = {}
        for label in range(10):
            selected = distances[np.asarray(expected) == label]
            per_class[str(label)] = {"mean": float(selected.mean()),
                                     "median": float(np.median(selected)),
                                     "p10": float(np.quantile(selected, 0.10)),
                                     "p90": float(np.quantile(selected, 0.90))}
        result[condition] = {
            "source": path.relative_to(ROOT).as_posix(),
            "definition": payload.get("definition"),
            "rows": len(rows), "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "p10": float(np.quantile(distances, 0.10)),
            "p90": float(np.quantile(distances, 0.90)),
            "minimum": float(distances.min()), "maximum": float(distances.max()),
            "per_class": per_class,
            "interpretation_limit": "nearest-TRAIN proximity is a memorization diagnostic, not novelty proof",
        }
    return result


def _offline_saved_audit(assets: Mapping[str, Any], measured: Mapping[str, Any]) -> dict[str, Any]:
    rows = assets["cohort"]["labels"].long().cpu()
    outputs: dict[str, Mapping[str, torch.Tensor]] = dict(measured["outputs"])
    summaries: dict[str, Mapping[str, Any]] = dict(measured["summaries"])
    for source_name, source_key in (("C45k", "C45k"),
                                    ("selected_initial", "selected_initial"),
                                    ("final_candidate", "final_candidate")):
        outputs[source_name] = assets["saved"]["conditions"][source_key]
        summaries[source_name] = frame._summary_from_readouts(
            assets["runtime"]["input_bundle"], assets["cohort"],
            outputs[source_name]["readouts"],
        )
    audit = {"task_id": TASK_ID, "population_role": "consumed_diagnostic_population",
             "rows": int(rows.numel()), "rows_per_class": {str(i): 200 for i in range(10)},
             "official_test_reads": 0, "conditions": {}, "pairwise_drift": {},
             "paired_prediction_changes": {}, "nearest_training_diagnostics":
                 _nearest_training_summary(assets["source_run"], rows)}
    for name in ("C45k", "selected_initial", "final_candidate", *CONDITION_ORDER):
        raw = outputs[name]["raw_cmos"]
        coeffs = outputs[name]["coefficients"]
        readouts = outputs[name]["readouts"]
        audit["conditions"][name] = {
            "camera": _camera_statistics(raw, assets["runtime"]),
            "total_modal_power": _modal_power_summary(coeffs),
            "coefficient_delta_from_E0G0_rms": float(
                (coeffs - outputs["E0G0"]["coefficients"]).abs().square().mean().sqrt()
            ),
            "raw_cmos_delta_from_E0G0_rms": float(
                (raw - outputs["E0G0"]["raw_cmos"]).double().square().mean().sqrt()
            ),
            "readout_exact_duplicate_count": summaries[name]["same_class_diversity"]
                ["direct_decoder"]["exact_duplicate_count"],
            "same_class_diversity": summaries[name]["same_class_diversity"]
                ["direct_decoder"]["by_class"],
            "evaluator_results": summaries[name]["conditions"]["direct_decoder"]["evaluators"],
            "ed_ec_prediction_agreement": summaries[name]["conditions"]["direct_decoder"]
                ["ED_EC_prediction_agreement"],
            "prediction_changes_vs_E0G0": _prediction_change_summary(
                rows, summaries["E0G0"], summaries[name]
            ),
            "saved_readouts_shape": list(readouts.shape),
        }
    drift_pairs = (
        ("C45k", "selected_initial"),
        ("selected_initial", "final_candidate"),
        ("C45k", "final_candidate"),
        ("E0G0", "E0G1"), ("E0G0", "E1G0"), ("E0G0", "E1G1"),
    )
    for left, right in drift_pairs:
        audit["pairwise_drift"][f"{left}->{right}"] = _pairwise_optical_drift(
            outputs[left], outputs[right]
        )
    for name in ("C45k", "selected_initial", "final_candidate", "E0G1", "E1G0"):
        audit["paired_prediction_changes"][f"E0G0->{name}"] = _prediction_change_summary(
            rows, summaries["E0G0"], summaries[name]
        )
    return audit


def _positive_controls(assets: Mapping[str, Any]) -> dict[str, Any]:
    controls = recent._load_solver_positive_controls(assets["previous_config"])
    _require(len(controls) == 20, "the saved positive controls must contain two rows per class")
    rows = []
    runtime = assets["runtime"]
    for condition in ("E0G1", "E1G0"):
        checkpoint = _condition_checkpoint(assets, condition)
        models = {grid: recent._new_endpoint_model(checkpoint, grid, runtime)
                  for grid in (recent.GRID_PRODUCTION, recent.GRID_REFERENCE)}
        with torch.inference_mode():
            for entry in controls:
                target = entry["target"].to(runtime["device"], dtype=torch.complex128)
                command = entry["command"].to(runtime["device"])
                outputs, direct = {}, {}
                for grid in (recent.GRID_PRODUCTION, recent.GRID_REFERENCE):
                    outputs[grid] = projected._forward_amplitude(models[grid], command)
                    direct[grid] = projected._forward_target(models[grid], target)["raw_cmos"]
                errors = projected._error_row(outputs, target, direct, 0)
                # This control checks the existing 2048-versus-4096 agreement
                # tolerance. Errors against the solver target are separate
                # reachability metrics and are not cross-grid errors.
                max_error = max(
                    float(errors["cross_grid_coefficient"]),
                    float(errors["cross_grid_native_cmos"]),
                )
                _require(max_error <= CROSS_GRID_TOLERANCE,
                         f"{condition} positive control exceeded existing tolerance: {errors}")
                rows.append({"condition": condition, "label": entry["label"],
                             "proposal_id": entry["proposal_id"], "metrics": errors,
                             "passed_existing_tolerance": True})
        del models
        torch.cuda.empty_cache()
    return {"task_id": TASK_ID, "rows": rows, "rows_per_class_per_condition": 2,
            "reference_grid": recent.GRID_REFERENCE,
            "existing_numerical_tolerance": CROSS_GRID_TOLERANCE,
            "semantic_evidence": False, "hardware_evidence": False}


def _write_rows(path: Path, assets: Mapping[str, Any], measured: Mapping[str, Any]) -> None:
    labels = assets["cohort"]["labels"].long().tolist()
    proposal_ids = assets["cohort"]["proposal_ids"].long().tolist()
    outputs, summaries = measured["outputs"], measured["summaries"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["population_index", "label", "proposal_id"] + [
            f"{name}_{evaluator}_prediction" for name in CONDITION_ORDER
            for evaluator in ("ED", "EC")
        ] + [f"{name}_coefficient_delta_l2" for name in CONDITION_ORDER[1:]]
          + [f"{name}_raw_cmos_delta_rms" for name in CONDITION_ORDER[1:]])
        for index, (label, proposal_id) in enumerate(zip(labels, proposal_ids, strict=True)):
            values: list[Any] = [index, label, proposal_id]
            for name in CONDITION_ORDER:
                evaluator_rows = summaries[name]["conditions"]["direct_decoder"]["evaluators"]
                values.extend(int(evaluator_rows[key]["predicted_labels"][index])
                              for key in ("ED", "EC"))
            for name in CONDITION_ORDER[1:]:
                values.append(float(torch.linalg.vector_norm(
                    outputs[name]["coefficients"][index] - outputs["E0G0"]["coefficients"][index]
                )))
            for name in CONDITION_ORDER[1:]:
                values.append(float((outputs[name]["raw_cmos"][index]
                                     - outputs["E0G0"]["raw_cmos"][index]).double().square().mean().sqrt()))
            writer.writerow(values)


def _write_figure(path: Path, measured: Mapping[str, Any], labels: torch.Tensor) -> None:
    chosen = [int(torch.nonzero(labels == label, as_tuple=False)[0]) for label in range(10)]
    image = Image.new("RGB", (4 * 132, 10 * 132), "white")
    draw = ImageDraw.Draw(image)
    for column, name in enumerate(CONDITION_ORDER):
        draw.text((column * 132 + 5, 4), name, fill="black")
        view = measured["outputs"][name]["readouts"].clamp(0.0, 1.0)
        for row, index in enumerate(chosen):
            digit = (view[index, 0] * 255).byte().numpy()
            tile = Image.fromarray(digit, mode="L").resize((112, 112), Image.Resampling.NEAREST).convert("RGB")
            image.paste(tile, (column * 132 + 5, row * 132 + 20))
            draw.text((3 * 132 + 120, row * 132 + 53), str(int(labels[index])), fill="black")
    image.save(path)


def _closeout(decision: Mapping[str, Any], gradient: Mapping[str, Any],
              positive: Mapping[str, Any], replay: Mapping[str, Any],
              drift: Mapping[str, Any], output_dir: Path) -> None:
    recommendation = {
        "ENCODER_PRIORITY": "建议下一项单变量研究固定 Generator，只检查固定输入适配器与 Encoder 更新的配合；本任务未授权执行。",
        "GENERATOR_PRIORITY": "建议下一项单变量研究固定 Encoder，只检查 Generator 更新的输出取舍；本任务未授权执行。",
        "MIXED_EFFECT": "模块贡献或交互无法归为单一模块；建议先设计能拆分交互的单变量机制检查，不启动训练。",
        "UNRESOLVED": "当前人口不能定位责任模块；建议先复核已保存端点及诊断量，不启动调参或追加训练。",
    }[str(decision["route"])]
    report = [
        "# 零更新归因与路线选择收尾报告",
        "",
        "## 结果与状态",
        "",
        f"任务状态：完成数值诊断；唯一归因结论为 `{decision['route']}`。",
        f"消费人口为既有 2,000 条（每类 200），新增参数更新 `{decision['optimizer_updates']}` 次。",
        "",
        "## 科学认识",
        "",
        "四条件交叉比较拆分了 Encoder、Generator 和两者交互的影响；保存权重的批次损失梯度只描述局部训练目标方向。",
        f"起点与终点保存输出复放：{replay['rows']} 行通过；跨网格正控：{len(positive['rows'])} 行通过原有容差。",
        "训练损失项的测量是局部机制证据，不证明语义失败的因果来源。",
        "",
        "## 目标缺口",
        "",
        "结论限于已消费数值人口。全体 G1 泛化、人眼可读性、器件标定及实物部署仍未测量。",
        "",
        "## 工程完成与验证",
        "",
        f"复核了 {len(drift)} 个保存步数点的相位变化；10 个固定 TRAIN 批次分别在 step 0 与 step 2000 测量。",
        f"DEV 重构对照固定每类 20 张、共 200 张；交叉正控共 {len(positive['rows'])} 条；优化器更新为 0。",
        "交付物包括逐样本 CSV、四条件图、原始 CMOS/系数/读出张量、梯度分解和决策 JSON。",
        "",
        "## 决策影响与下一步",
        "",
        recommendation,
        "以上仅为建议；本轮不执行后继训练、重采样、损失改权、结构扩容或硬件操作。",
        f"模块恢复判据：{json.dumps(decision['restoration_support'], ensure_ascii=False)}；交互效应：{json.dumps(decision['factorial_interaction'], ensure_ascii=False)}。",
    ]
    (output_dir / "closeout_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _negative_control_rejections(assets: Mapping[str, Any]) -> dict[str, Any]:
    cohort, commands = assets["cohort"], assets["commands"]
    rejected = {}
    try:
        _validate_saved_population_identity(
            cohort["labels"], cohort["proposal_ids"], commands,
            cohort["labels"].roll(1), cohort["proposal_ids"], commands,
        )
    except RuntimeError:
        rejected["reordered_label_rows"] = True
    else:
        raise RuntimeError("consumer negative control failed to reject reordered rows")
    try:
        _validate_saved_population_identity(
            cohort["labels"], cohort["proposal_ids"], commands,
            cohort["labels"], cohort["proposal_ids"], commands.roll(1, dims=0),
        )
    except RuntimeError:
        rejected["wrong_saved_commands"] = True
    else:
        raise RuntimeError("consumer negative control failed to reject misbound commands")
    bundle = {
        "phase_sources": {
            name: {"encoder_step": CONDITION_PHASE_STEPS[name]["encoder"],
                   "generator_step": CONDITION_PHASE_STEPS[name]["generator"],
                   "source_training_run_id": assets["run_id"]}
            for name in CONDITION_ORDER
        },
        "phase_states": {
            name: {
                "encoder": _condition_checkpoint(assets, name)["model"]
                    ["runtime.encoder.phase"].clone(),
                "generator": _condition_checkpoint(assets, name)["model"]
                    ["runtime.generator.phase"].clone(),
            } for name in CONDITION_ORDER
        },
    }
    bundle["phase_states"]["E0G1"]["encoder"].view(-1)[0] += 0.01
    try:
        _validate_condition_phase_provenance(bundle, assets)
    except RuntimeError:
        rejected["wrong_crossed_phase_source"] = True
    else:
        raise RuntimeError("consumer negative control failed to reject a wrong phase source")
    return {"task_id": TASK_ID, "all_rejected": True,
            "rejected_cases": sorted(rejected), "negative_control_only": True}


def _production_preflight(assets: Mapping[str, Any]) -> dict[str, Any]:
    command = assets["commands"][:1]
    checks = []
    for name in CONDITION_ORDER:
        checkpoint = _condition_checkpoint(assets, name)
        output = _measure_fullpath(checkpoint, command, assets["runtime"])
        _require(tuple(output["coefficients"].shape) == (1, 100)
                 and tuple(output["raw_cmos"].shape) == (1, 1, 200, 200)
                 and tuple(output["readouts"].shape) == (1, 1, 28, 28)
                 and all(torch.isfinite(output[key]).all().item()
                         for key in ("coefficients", "raw_cmos", "readouts")),
                 f"representative {name} full-path preflight returned invalid outputs")
        checks.append({"condition": name, "rows": 1,
                       "coefficient_norm": float(torch.linalg.vector_norm(
                           output["coefficients"], dim=1)[0]),
                       "raw_cmos_max": float(output["raw_cmos"].max()),
                       "readout_range": [float(output["readouts"].min()),
                                         float(output["readouts"].max())]})
        del checkpoint, output
        torch.cuda.empty_cache()
    return {"task_id": TASK_ID, "status": "PASSED",
            "same_saved_slm_command": True, "optimizer_updates": 0,
            "production_path": "saved command -> current Encoder -> Gaussian OLS -> current Generator -> raw CMOS",
            "conditions": checks,
            "consumer_negative_controls": _negative_control_rejections(assets)}


def _asset_reuse_table(config: Mapping[str, Any], assets: Mapping[str, Any]) -> dict[str, Any]:
    source_run = (ROOT / config["sources"]["source_run"]).resolve()
    return {"task_id": TASK_ID, "source_run_id": assets["run_id"], "entries": [
        {"asset": "frozen mapper, physical-phase prior, fixed projection, evaluator, camera setup",
         "source": config["sources"]["source_config"],
         "use": "fixed input adapter and fixed evaluation; no updates"},
        {"asset": "saved 2,000-row population and legal SLM commands",
         "source": (source_run / "final_population_and_commands.pt").relative_to(ROOT).as_posix(),
         "use": "same command rows for all crossed conditions; population already consumed"},
        {"asset": "C45k, selected-initial, terminal optical output bundles",
         "source": (source_run / "final_reconsumed_outputs.pt").relative_to(ROOT).as_posix(),
         "use": "offline endpoint comparison and E0G0/E1G1 output reuse"},
        {"asset": "D0 step 0/500/1000/1500/2000 checkpoints, terminal state, schedule, Adam history",
         "source": (source_run / "scientific_attempt").relative_to(ROOT).as_posix(),
         "use": "phase provenance, drift, fixed batches, source-state preservation"},
        {"asset": "C45k and endpoint nearest-TRAIN diagnostics",
         "source": "nearest_train_C45k.json, nearest_train_selected_initial.json, nearest_train_final_candidate.json",
         "use": "reuse training-proximity diagnostics; do not interpret as novelty proof"},
    ]}


def _assert_protected_unchanged(assets: Mapping[str, Any],
                                snapshot: Mapping[str, Any]) -> None:
    _require(all(_same_state(model, assets["checkpoints"][step]["model"])
                 for step, model in snapshot["models"].items())
             and all(_same_state(optimizer, assets["checkpoints"][step]["optimizer"])
                     for step, optimizer in snapshot["optimizers"].items()),
             "source optical model or Adam state changed during zero-update diagnostics")
    _require(torch.equal(assets["schedule"], snapshot["schedule"])
             and torch.equal(assets["commands"], snapshot["commands"])
             and all(torch.equal(assets["cohort"][key], value)
                     for key, value in snapshot["population_rows"].items())
             and _same_state(snapshot["static_inputs"], {
                 key: assets["runtime"][key]
                 for key in ("prior", "projector", "mapper_checkpoint", "camera_gain")
             } | {key: assets["runtime"][key].detach().cpu()
                  for key in ("matrix", "inverse", "scale", "pixels", "labels",
                              "train_indices", "dev_indices")}),
             "saved schedule, input population, mapper/prior/projection, or training data changed")
    for name, model in (("ED", assets["runtime"]["input_bundle"]["evaluator_ed"]),
                        ("EC", assets["runtime"]["input_bundle"]["evaluator_ec"])):
        _require(all(torch.equal(value, model.state_dict()[key].detach().cpu())
                     for key, value in snapshot["evaluators"][name].items()),
                 f"frozen {name} evaluator state changed")


def _run(config_path: Path) -> dict[str, Any]:
    config = _read_config(config_path)
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for the registered optical forwards")
    _require(not OUTPUT_DIR.exists(), f"refusing to reuse an existing output directory: {OUTPUT_DIR}",
             FileExistsError)
    assets = _load_saved_assets(config, torch.device("cuda"))
    protected_snapshot = {
        "models": {step: copy.deepcopy(checkpoint["model"])
                   for step, checkpoint in assets["checkpoints"].items()},
        "optimizers": {step: copy.deepcopy(checkpoint["optimizer"])
                       for step, checkpoint in assets["checkpoints"].items()},
        "schedule": assets["schedule"].clone(),
        "commands": assets["commands"].clone(),
        "population_rows": {
            key: assets["cohort"][key].clone()
            for key in ("labels", "proposal_ids", "coefficients")
        },
        "static_inputs": copy.deepcopy({
            key: assets["runtime"][key]
            for key in ("prior", "projector", "mapper_checkpoint", "camera_gain")
        } | {key: assets["runtime"][key].detach().cpu().clone()
             for key in ("matrix", "inverse", "scale", "pixels", "labels",
                         "train_indices", "dev_indices")}),
        "evaluators": {
            name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            for name, model in (("ED", assets["runtime"]["input_bundle"]["evaluator_ed"]),
                                ("EC", assets["runtime"]["input_bundle"]["evaluator_ec"]))
        },
    }
    preflight = _production_preflight(assets)
    replay = _replay_endpoints(assets)
    drift = _phase_drift(assets)
    positive = _positive_controls(assets)
    gradients = _gradient_diagnostics(assets)
    development = _development_reconstruction_diagnostic(assets)
    _assert_protected_unchanged(assets, protected_snapshot)
    source_cp = assets["selected"]["checkpoint"]
    for step, checkpoint in assets["checkpoints"].items():
        for key in ("weights", "structural_weight", "scale", "power_reference", "camera_gain"):
            _require(recent._states_close(checkpoint.get(key), source_cp.get(key)),
                     f"saved step {step} changed fixed training reference {key}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    _write_json(OUTPUT_DIR / "task_contract.json", {
        "task_id": TASK_ID, "evidence_tier": "diagnostic_only",
        "source_backup": config["sources"]["verified_backup"],
        "source_execution_run": config["sources"]["source_run"],
        "optimizer_updates": 0, "population_resampled": False,
        "conditions": list(CONDITION_ORDER), "official_test_reads": 0,
        "stop": "one attribution decision; no automatic successor experiment",
    })
    (OUTPUT_DIR / "run_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _write_json(OUTPUT_DIR / "asset_reuse_table.json", _asset_reuse_table(config, assets))
    _write_json(OUTPUT_DIR / "production_preflight.json", preflight)
    _write_json(OUTPUT_DIR / "source_identity.json", {
        "task_id": TASK_ID,
        "source_training_task_id": recent.TASK_ID,
        "source_training_run_id": assets["run_id"],
        "selected_source_endpoint": "D0",
        "selected_endpoint_run_id": assets["selected"]["checkpoint"]["run_id"],
        "terminal_run_id": assets["checkpoints"][2_000]["run_id"],
        "terminal_global_step": assets["checkpoints"][2_000]["completed_global_step"],
        "optimizer_steps": assets["checkpoints"][2_000]["optimizer_steps"],
        "source_saved_population_role": assets["cohort"]["role"],
        "analysis_population_role": CONSUMED_POPULATION_ROLE,
        "population_rows": 2_000,
        "verified_windows_backup": config["sources"]["verified_backup"],
    })
    _write_json(OUTPUT_DIR / "endpoint_replay.json", replay)
    _write_json(OUTPUT_DIR / "phase_drift.json", drift)
    _write_json(OUTPUT_DIR / "cross_grid_positive_controls.json", positive)
    _write_json(OUTPUT_DIR / "loss_gradient_diagnostics.json", gradients)
    development_report = {key: value for key, value in development.items()
                          if key not in {"readouts", "target_images"}}
    _write_json(OUTPUT_DIR / "development_reconstruction.json", development_report)
    _write_development_reconstruction_rows(
        OUTPUT_DIR / "development_reconstruction_per_sample.csv", development
    )
    _write_development_reconstruction_figure(
        OUTPUT_DIR / "development_reconstruction.png", development
    )
    measured = _measure_cross_conditions(assets, OUTPUT_DIR)
    labels = assets["cohort"]["labels"].long().cpu()
    decision = _attribution_decision(labels, measured["summaries"])
    _write_json(OUTPUT_DIR / "attribution_decision.json", decision)
    _write_rows(OUTPUT_DIR / "per_sample_results.csv", assets, measured)
    _write_figure(OUTPUT_DIR / "crossed_conditions.png", measured, labels)
    offline = _offline_saved_audit(assets, measured)
    _write_json(OUTPUT_DIR / "offline_saved_evidence_audit.json", offline)
    _assert_protected_unchanged(assets, protected_snapshot)
    _write_json(OUTPUT_DIR / "measurement_status.json", {
        "status": "COMPLETE_DIAGNOSTIC", "task_id": TASK_ID,
        "optimizer_updates": 0, "population_rows": 2_000,
        "cross_condition_rows": 4_000, "positive_control_rows": 40,
        "dev_reconstruction_rows": 200, "gradient_batches": 20,
        "decision_route": decision["route"],
        "all_source_model_adam_and_evaluator_states_unchanged": True,
    })
    _closeout(decision, gradients, positive, replay, drift, OUTPUT_DIR)
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--verify-saved", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    config = _read_config(args.config.resolve())
    if args.verify_saved:
        _require(args.artifact_dir is not None,
                 "--verify-saved requires the exact local autodl_backup_attempt_05 path", ValueError)
        artifact_dir = args.artifact_dir.resolve()
        expected = (ROOT / "outputs/20261031_lightgen_fullpath_zero_update_attribution_v1/runs"
                    / "autodl_backup_attempt_05").resolve()
        _require(artifact_dir == expected and artifact_dir.is_dir(),
                 "independent review input must be the registered local backup directory", ValueError)
        review_dir = artifact_dir.parent / "windows_review_01"
        _require(not review_dir.exists(), f"refusing to replace prior Windows review: {review_dir}",
                 FileExistsError)
        decision, review = _reconsume_saved_conditions(
            artifact_dir, config, use_verified_backup=True
        )
        review_dir.mkdir(parents=True, exist_ok=False)
        _write_json(review_dir / "independent_review.json", review)
        _write_json(review_dir / "recomputed_attribution_decision.json", decision)
        print(json.dumps({"task_id": TASK_ID, "windows_review": str(review_dir),
                          "route": decision["route"]}, ensure_ascii=False))
        return 0
    _require(args.artifact_dir is None,
             "--artifact-dir is only valid with --verify-saved", ValueError)
    result = _run(args.config.resolve())
    print(json.dumps({"task_id": TASK_ID, "route": result["route"],
                      "optimizer_updates": result["optimizer_updates"],
                      "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
