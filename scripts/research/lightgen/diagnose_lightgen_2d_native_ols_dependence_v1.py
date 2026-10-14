"""Frozen counterfactual diagnosis of native complex OLS dependence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as bogt
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from ldm_project.v2.config import load_config
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_json, atomic_write_new_json
from ldm_project.v2.optics.lightgen import LightGenAmplitudeFiberOLSSystemV2
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as class17
from scripts.research.lightgen import execute_lightgen_2d_existing_400_frozen_replay_v1 as replay400


TASK_ID = "M0_NATIVE_OLS_DEPENDENCE_AND_PRESENTATION_DIAGNOSTIC_V1"
REPORT_SCHEMA = "m0_native_ols_dependence_and_presentation_report.v1"
MANIFEST_SCHEMA = "m0_native_ols_dependence_and_presentation_resume.v1"
FIT_ROWS = 200
OBSERVATION_ROWS = 100
TOTAL_ROWS = 300
FOCUS_CLASSES = (1, 7)
CONTROL_CLASSES = (0, 4, 8)
PORT_SHUFFLE_SEEDS = (2026100401, 2026100402)
WHOLE_ROW_SHUFFLE_SEED = 2026100403
CHUNK_ROWS = 10
FORWARD_BATCH_ROWS = 2
MAINLINE_STEP = 2200
PROXY_STEP = 600
DEFAULT_POPULATION = Path("outputs/20261002_m0_class17_mapping_diagnostic_v2/runs/attempt_01/input_population.json")
DEFAULT_CLASS17_REPORT = Path("outputs/20261002_m0_class17_mapping_diagnostic_v2/runs/attempt_01/report.json")
DEFAULT_PROXY_ARTIFACT = Path("outputs/20260928_lightgen_400_mnist_semantic_generation_baseline_v1/runs/autodl_attempt_01")
DEFAULT_PROXY_CONFIG = Path("configs/v2/diagnostic/lightgen_amplitude_fiber_ols_400.yaml")
DEFAULT_PROXY_REPLAY = Path("outputs/20261002_m0_existing_400_frozen_replay_validity_v1/runs/attempt_01")
DEFAULT_TRAINING_EVALUATOR = Path("outputs/20260724_v2/runs/20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/frozen_mnist_evaluator.pt")
DEFAULT_DATA = Path("data/MNIST/processed/training.pt")
DEFAULT_OUTPUT = Path("outputs/20261004_m0_native_ols_dependence_and_presentation_diagnostic_v1/runs/attempt_01")
SCENARIOS = ("port_shuffle_seed_2026100401_unmatched", "port_shuffle_seed_2026100402_unmatched", "whole_row_control")


class DiagnosticError(RuntimeError):
    """An input, freeze, execution, or evidence invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_torch(path: Path, *, mmap: bool = True) -> Any:
    _require(path.is_file(), f"required input is missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON input is not an object: {path}")
    return value


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous().clone() for key, value in module.state_dict().items()}


def _state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def _assert_frozen(module: torch.nn.Module, expected: Mapping[str, torch.Tensor], name: str) -> None:
    _require(_state_equal(_snapshot(module), expected), f"{name} parameters or buffers changed during read-only measurement")
    _require(all(not p.requires_grad and p.grad is None for p in module.parameters()), f"{name} is not frozen with empty gradients")


def _paths(root: Path, args: argparse.Namespace) -> dict[str, Path]:
    population_path = _resolve(root, args.reference_population)
    population = _read_json(population_path)
    _require(population.get("task_id") == "M0_CLASS17_MAPPING_DIAGNOSTIC_V2", "fixed population is not from the named class17 diagnostic")
    named = population.get("input_paths")
    _require(isinstance(named, Mapping), "fixed population does not name its inputs")
    result: dict[str, Path] = {"reference_population": population_path}
    for key in ("step1200", "step2200", "readout1200", "readout2200", "training_data", "e_s", "e_i", "e_c", "e_d"):
        value = named.get(key)
        _require(isinstance(value, str) and value, f"fixed population is missing input path {key}")
        result[key] = _resolve(root, value)
    named_data = _resolve(root, named["training_data"])
    requested_data = _resolve(root, args.training_data) if args.training_data else named_data
    _require(requested_data == named_data, "training data path must be the canonical mother file named by the fixed population")
    result.update(
        class17_report=_resolve(root, args.class17_report),
        proxy_artifact=_resolve(root, args.proxy_artifact_dir),
        proxy_config=_resolve(root, args.proxy_config),
        proxy_replay=_resolve(root, args.proxy_replay_dir),
        training_evaluator=_resolve(root, args.training_evaluator),
        data=requested_data,
        training_data=requested_data,
        output_dir=_resolve(root, args.output_dir),
    )
    return result


def _validate_prior_authorities(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    class_report = _read_json(paths["class17_report"])
    _require(class_report.get("task_id") == "M0_CLASS17_MAPPING_DIAGNOSTIC_V2", "class17 report authority mismatch")
    _require(class_report.get("status") == "VALID_COMPLETED" and class_report.get("next_action") == "NONE", "class17 diagnostic is not a completed, stopped measurement")
    proxy_report = _read_json(paths["proxy_replay"] / "report.json")
    _require(proxy_report.get("task_id") == "M0_EXISTING_400_FROZEN_REPLAY_VALIDITY_V1", "400 frozen replay report authority mismatch")
    _require(proxy_report.get("status") == "VALID_COMPLETED" and proxy_report.get("classification") == "FROZEN_REPLAY_CONFIRMED", "400 model replay was not confirmed")
    _require(proxy_report.get("training_updates") == 0 and proxy_report.get("next_action") == "NONE", "400 replay is not a frozen stopped result")
    return class_report, proxy_report


def _load_exact_inputs(paths: Mapping[str, Path], device: torch.device) -> dict[str, Any]:
    loaded = class17.load_inputs(
        _root(),
        {key: paths[key] for key in ("step1200", "step2200", "readout1200", "readout2200", "training_data", "e_s", "e_i", "e_c", "e_d", "output_dir")},
    )
    _require(loaded["checkpoint_data"]["step1200"].get("seed") == 1234, "step1200 reference source split is not seed1234 TRAIN")
    reference_model = bogt.FrozenBogtReferences.from_payload(loaded["references"])
    weight_model = bogt.BogtLossWeights.from_payload(loaded["weights"])
    _require(reference_model.data_role == bogt.TRAIN_ROLE and weight_model.data_role == bogt.TRAIN_ROLE, "inherited BOGT references are not TRAIN-scoped")
    _require(loaded["payloads"]["step2200"].get("references") is None and loaded["payloads"]["step2200"].get("weights") is None, "step2200 supplied a competing calibration")
    consumer = bogt.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device=device,
        propagation_grid_size=bogt.PRODUCTION_PROPAGATION_GRID,
    )
    consumer.load_trainable_state_dict(loaded["payloads"]["step2200"]["trainable_parameter_state"])
    consumer.runtime.eval()
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
    _require(consumer.runtime.config.propagation_grid_size == 2048, "mainline runtime is not the production 2048 propagation grid")
    mainline_state = _snapshot(consumer.runtime)

    model_path = paths["proxy_artifact"] / "model_step_000600.pt"
    proxy_checkpoint = _load_torch(model_path)
    _require(proxy_checkpoint.get("schema") == "lightgen_400_mnist_semantic_baseline_checkpoint.v1" and proxy_checkpoint.get("completed_updates") == PROXY_STEP, "proxy checkpoint is not the saved step600 model")
    _require("optimizer" not in proxy_checkpoint and "optimizer_state" not in proxy_checkpoint, "proxy step600 unexpectedly contains optimizer state")
    config = load_config(paths["proxy_config"])
    config_record = replay400.validate_system_config(config)
    proxy_model = LightGenAmplitudeFiberOLSSystemV2(config.system).to(device)
    proxy_model.load_state_dict(proxy_checkpoint["model"], strict=True)
    proxy_model.eval()
    for parameter in proxy_model.parameters():
        parameter.requires_grad_(False)
    proxy_state = _snapshot(proxy_model)

    replay_manifest = _read_json(paths["proxy_replay"] / "resume_manifest.json")
    _require(replay_manifest.get("status") == "VALID_COMPLETED" and replay_manifest.get("updates") == 0, "400 frozen replay manifest is not terminal and update-free")
    replay_frozen_state = _load_torch(paths["proxy_replay"] / "frozen_model_state.pt")
    _require(isinstance(replay_frozen_state, Mapping) and _state_equal(proxy_state, replay_frozen_state), "400 replay frozen model differs from the explicit step600 checkpoint")
    old_population = _read_json(paths["proxy_replay"] / "input_population.json")
    new_population = _read_json(paths["reference_population"])
    _require(old_population.get("source_row_indices") == new_population.get("source_row_indices"), "400 replay used a different fixed row population")
    _require(old_population.get("labels") == new_population.get("labels"), "400 replay used different fixed row labels")
    replay_inputs = old_population.get("input_paths", {})
    current_inputs = new_population.get("input_paths", {})
    for name in ("e_s", "e_i", "e_c", "e_d"):
        _require(replay_inputs.get(name) == current_inputs.get(name), f"400 replay evaluator path differs for {name}")

    evaluators = replay400._load_frozen_evaluators(
        {name: paths[name] for name in ("e_s", "e_i", "e_c", "e_d")},
        paths["training_evaluator"],
        device,
    )
    replay_evaluator_states = _load_torch(paths["proxy_replay"] / "frozen_evaluator_states.pt")
    _require(isinstance(replay_evaluator_states, Mapping), "400 replay evaluator states are missing")
    for name, evaluator in evaluators.items():
        prior_state = replay_evaluator_states.get(name)
        _require(isinstance(prior_state, Mapping) and _state_equal(_snapshot(evaluator), prior_state), f"400 replay evaluator state differs for {name}")

    final_readouts = _load_torch(paths["proxy_replay"] / "final_readouts.pt")
    _require(isinstance(final_readouts, Mapping), "400 replay final readouts are invalid")
    indices = loaded["source_indices"].clone().long()
    labels = loaded["labels"].clone().long()
    images = loaded["images"].clone().float()
    _require(torch.equal(final_readouts["reference_source_row_indices"], indices), "400 replay source-row indices differ from the fixed TRAIN population")
    _require(torch.equal(final_readouts["reference_labels"], labels), "400 replay reference labels differ from the fixed TRAIN population")
    _require(final_readouts["reference_ols"].shape == (TOTAL_ROWS, 100) and torch.is_complex(final_readouts["reference_ols"]), "400 replay reference OLS shape/dtype changed")
    _require(final_readouts["reference_raw_cmos_400x400"].shape == (TOTAL_ROWS, 1, 400, 400), "400 replay reference raw CMOS shape changed")
    _require(final_readouts["reference_views_28x28"].shape == (TOTAL_ROWS, 1, 28, 28), "400 replay reference view shape changed")
    _require(set(final_readouts["reference_logits"]) == set(evaluators), "400 replay evaluator readout set changed")

    power_record = _read_json(paths["proxy_replay"] / "optical_plane_and_power.json")
    reference_planes = power_record["reference_encoder_OLS_generator"]["planes"]
    emitted_plane = next(item for item in reference_planes if item.get("plane_id") == "lightgen_fiber_reemission")
    proxy_base_power = torch.as_tensor(emitted_plane.get("power_proxy_by_row"), dtype=torch.float64)
    _require(proxy_base_power.shape == (TOTAL_ROWS,) and bool(torch.isfinite(proxy_base_power).all()) and bool((proxy_base_power > 0).all()), "400 replay lacks per-row physical re-emission power")
    _require(power_record.get("ols_power_quantity") == "sum(|z_k|^2) in 100-mode coefficient space", "400 replay power field labels changed")
    _require("sum(|U(x,y)|^2)*pixel_pitch_x*pixel_pitch_y" in power_record.get("spatial_power_quantity", ""), "400 replay does not define the physical plane-power quantity")

    return {
        "consumer": consumer,
        "mainline_state": mainline_state,
        "proxy_model": proxy_model,
        "proxy_state": proxy_state,
        "proxy_config_record": config_record,
        "evaluators": evaluators,
        "evaluator_states": {name: _snapshot(model) for name, model in evaluators.items()},
        "indices": indices,
        "labels": labels,
        "images": images,
        "final_readouts": final_readouts,
        "proxy_base_power": proxy_base_power,
        "mainline_checkpoint_data": loaded["checkpoint_data"],
        "mainline_parameters": loaded["payloads"]["step2200"]["trainable_parameter_state"],
        "proxy_model_path": model_path,
    }


def _proxy_target(images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    intensity = F.interpolate(images.to(device), size=(400, 400), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    return intensity.sqrt(), intensity


def _resource_snapshot(device: torch.device) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_total_bytes": int(total_bytes),
        "device_free_bytes": int(free_bytes),
        "device_used_bytes": int(total_bytes - free_bytes),
        "pytorch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "pytorch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _preflight(inputs: Mapping[str, Any], paths: Mapping[str, Path], device: torch.device) -> dict[str, Any]:
    consumer = inputs["consumer"]
    proxy = inputs["proxy_model"]
    evaluators = inputs["evaluators"]
    main_before = inputs["mainline_state"]
    proxy_before = inputs["proxy_state"]
    eval_before = inputs["evaluator_states"]
    started = time.perf_counter()
    resource_before = _resource_snapshot(device)
    positive_logits: dict[str, list[torch.Tensor]] = {name: [] for name in evaluators}
    with torch.no_grad():
        target = inputs["images"].to(device)
        for batch in target.split(30):
            view = F.adaptive_avg_pool2d(batch.clamp(0.0, 1.0), (28, 28))
            for name, evaluator in evaluators.items():
                positive_logits[name].append(replay400.evaluator_logits(evaluator, view).detach().cpu())
        image_batch = inputs["images"][:2].to(device)
        main = consumer.runtime.forward_from_image(image_batch)
        main_replay = consumer.runtime.forward_from_ols(main.z_ols.coefficients)
        _require(torch.allclose(main.raw_intensity, main_replay.raw_intensity, rtol=1.0e-5, atol=1.0e-6), "mainline real-image and external-OLS paths disagree on the same coefficients")
        amplitude, _ = _proxy_target(inputs["images"][:2], device)
        proxy_result = proxy(proxy.command(amplitude))
        proxy_replay = proxy.forward_from_ols(proxy_result.ols.coefficients)
        _require(torch.allclose(proxy_result.raw_camera.tensor, proxy_replay.raw_camera.tensor, rtol=1.0e-5, atol=1.0e-6), "400 proxy input and external-OLS paths disagree on the same coefficients")
        stored_proxy_raw = inputs["final_readouts"]["reference_raw_cmos_400x400"][:2].to(device)
        _require(torch.allclose(proxy_replay.raw_camera.tensor, stored_proxy_raw, rtol=1.0e-5, atol=1.0e-6), "step600 direct OLS replay does not reproduce the prior frozen 300-row outputs")
    positive_metrics = {}
    for name in evaluators:
        logits = torch.cat(positive_logits[name])
        positive_metrics[name] = replay400.confusion_metrics(logits.argmax(1), inputs["labels"])
    _assert_frozen(consumer.runtime, main_before, "step2200 mainline runtime")
    _assert_frozen(proxy, proxy_before, "step600 400 proxy")
    for name, model in evaluators.items():
        _assert_frozen(model, eval_before[name], name)
    resource_after = _resource_snapshot(device)
    return {
        "status": "READY",
        "device": resource_after,
        "resource_before": resource_before,
        "resource_after": resource_after,
        "durations_seconds": {"preflight": time.perf_counter() - started},
        "frozen_forward_checks": {
            "mainline_image_to_OLS_to_generator": True,
            "mainline_external_OLS_same_generator": True,
            "proxy_input_to_OLS_to_generator": True,
            "proxy_external_OLS_same_generator": True,
            "proxy_replay_matches_existing_300_rows": True,
            "all_parameters_buffers_and_evaluators_unchanged": True,
            "optimizer_created": False,
            "autograd_updates": 0,
        },
        "target_positive_control": positive_metrics,
        "output_drive_free_bytes": int(shutil.disk_usage(paths["output_dir"].parent if paths["output_dir"].parent.exists() else _root()).free),
    }


def _metric_split(labels: torch.Tensor, logits: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    partitions = {"all_300_train_rows_descriptive": torch.arange(TOTAL_ROWS), "fit_200_rows": torch.arange(FIT_ROWS), "observation_100_rows": torch.arange(FIT_ROWS, TOTAL_ROWS)}
    for partition, indices in partitions.items():
        ys = labels.index_select(0, indices)
        result[partition] = {}
        for name, values in logits.items():
            local = values.index_select(0, indices)
            result[partition][name] = replay400.confusion_metrics(local.argmax(1), ys)
    return result


def _port_permutation_indices(labels: torch.Tensor, *, seed: int, shared_row_permutation: bool = False) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mapping = torch.arange(TOTAL_ROWS, dtype=torch.long).unsqueeze(1).expand(-1, 100).clone()
    for left, right in ((0, FIT_ROWS), (FIT_ROWS, TOTAL_ROWS)):
        for digit in range(10):
            rows = torch.arange(left, right)[labels[left:right] == digit]
            _require(rows.numel() == (20 if left == 0 else 10), "fixed class rows changed inside permutation partition")
            if shared_row_permutation:
                order = rows[torch.randperm(rows.numel(), generator=generator)]
                mapping[rows] = order[:, None]
            else:
                for port in range(100):
                    mapping[rows, port] = rows[torch.randperm(rows.numel(), generator=generator)]
    return mapping


def _shuffled_coefficients(coefficients: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    _require(coefficients.shape == (TOTAL_ROWS, 100) and torch.is_complex(coefficients), "source OLS coefficients have the wrong geometry")
    _require(mapping.shape == (TOTAL_ROWS, 100), "OLS permutation map has the wrong geometry")
    output = coefficients.gather(0, mapping)
    _require(bool(torch.isfinite(output).all()), "shuffled OLS contains nonfinite values")
    return output.contiguous()


def _verify_port_marginals(source: torch.Tensor, shuffled: torch.Tensor, mapping: torch.Tensor, labels: torch.Tensor) -> None:
    for left, right in ((0, FIT_ROWS), (FIT_ROWS, TOTAL_ROWS)):
        for digit in range(10):
            rows = torch.arange(left, right)[labels[left:right] == digit]
            for port in range(100):
                source_rows = mapping[rows, port]
                _require(torch.unique(source_rows).numel() == rows.numel(), "independent port shuffle reused a source row within a class/role marginal")
                _require(bool((labels.index_select(0, source_rows) == digit).all()), "independent port shuffle crossed class labels")
                _require(torch.equal(shuffled[rows, port], source[source_rows, port]), "independent port shuffle changed a complex coefficient pair")


def _scenario_key(system: str, scenario: str, chunk_index: int) -> str:
    return f"{system}/{scenario}/{chunk_index:04d}"


def _chunk_path(run_dir: Path, system: str, scenario: str, chunk_index: int) -> Path:
    return run_dir / "chunks" / system / scenario / f"chunk_{chunk_index:04d}.pt"


def _load_saved_chunk(
    path: Path,
    expected: Mapping[str, Any],
    labels: torch.Tensor,
    source_rows: torch.Tensor,
    evaluator_names: tuple[str, ...],
) -> dict[str, Any]:
    payload = _load_torch(path, mmap=False)
    _require(isinstance(payload, Mapping), f"saved measurement chunk is invalid: {path}")
    for key in ("system", "scenario", "start", "stop"):
        _require(payload.get(key) == expected[key], f"saved measurement chunk identity changed: {path}:{key}")
    row_count = expected["stop"] - expected["start"]
    saved_labels = payload.get("labels")
    saved_rows = payload.get("source_row_indices")
    _require(torch.is_tensor(saved_labels) and torch.equal(saved_labels, labels), f"saved measurement chunk labels changed: {path}")
    _require(torch.is_tensor(saved_rows) and torch.equal(saved_rows, source_rows), f"saved measurement chunk TRAIN rows changed: {path}")
    coefficients = payload.get("ols_coefficients")
    _require(torch.is_tensor(coefficients) and coefficients.shape == (row_count, 100) and torch.is_complex(coefficients) and bool(torch.isfinite(coefficients).all()), f"saved complex OLS chunk is invalid: {path}")
    raw = payload.get("raw_cmos")
    geometry = (1, 200, 200) if expected["system"] == "mainline" else (1, 400, 400)
    _require(torch.is_tensor(raw) and tuple(raw.shape) == (row_count, *geometry) and bool(torch.isfinite(raw).all()), f"saved raw CMOS chunk is invalid: {path}")
    view = payload.get("evaluator_view_28x28")
    _require(torch.is_tensor(view) and tuple(view.shape) == (row_count, 1, 28, 28) and bool(torch.isfinite(view).all()), f"saved evaluator-view chunk is invalid: {path}")
    power = payload.get("input_plane_integrated_power")
    _require(torch.is_tensor(power) and power.shape == (row_count,) and bool(torch.isfinite(power).all()) and bool((power > 0).all()), f"saved generator-input power chunk is invalid: {path}")
    logits = payload.get("evaluator_logits")
    _require(isinstance(logits, Mapping) and set(logits) == set(evaluator_names), f"saved evaluator logits are incomplete: {path}")
    for name in evaluator_names:
        _require(torch.is_tensor(logits[name]) and tuple(logits[name].shape) == (row_count, 10) and bool(torch.isfinite(logits[name]).all()), f"saved {name} logits are invalid: {path}")
    _require(payload.get("updates") == 0, f"saved measurement chunk records updates: {path}")
    return dict(payload)


def _scenario_forward(model: Any, system: str, values: torch.Tensor | None, images: torch.Tensor | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if system == "mainline" and images is not None:
        result = model.runtime.forward_from_image(images.to(device))
        return result.raw_intensity.detach(), bogt.physical_plane_power(result.generator_input_field).detach().to(torch.float64), result.z_ols.coefficients.detach()
    if system == "mainline":
        result = model.runtime.forward_from_ols(values.to(device))
        return result.raw_intensity.detach(), bogt.physical_plane_power(result.generator_input_field).detach().to(torch.float64), values.detach()
    if images is not None:
        amplitude, _ = _proxy_target(images, device)
        result = model(model.command(amplitude))
    else:
        result = model.forward_from_ols(values.to(device))
    emitted = result.optical_trace[0]
    tensor = emitted.tensor
    pixel_area = float(emitted.plane.pixel_pitch_m[0] * emitted.plane.pixel_pitch_m[1])
    power = tensor.abs().to(torch.float64).square().flatten(1).sum(1) * pixel_area
    coefficients = result.ols.coefficients if images is not None else values
    return result.raw_camera.tensor.detach(), power.detach(), coefficients.detach()


def _evaluate_raw(raw: torch.Tensor, system: str, evaluators: Mapping[str, torch.nn.Module], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if system == "mainline":
        view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))
    else:
        view = replay400.raw_400_to_view(raw)
    logits: dict[str, torch.Tensor] = {}
    for name, model in evaluators.items():
        logits[name] = replay400.evaluator_logits(model, view.to(device)).detach().cpu()
    return view.detach().cpu(), logits


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_unix"] = time.time()
    atomic_write_json(path, manifest)


def _measure_scenario(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    system: str,
    scenario: str,
    model: Any,
    labels: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    evaluators: Mapping[str, torch.nn.Module],
    coefficients: torch.Tensor | None = None,
    images: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    _require((coefficients is None) != (images is None), "scenario requires exactly one of OLS coefficients or original images")
    collected: dict[str, list[torch.Tensor]] = {"coefficients": [], "power": []}
    collected["logits"] = []
    total = labels.numel()
    for chunk_index, start in enumerate(range(0, total, CHUNK_ROWS)):
        stop = min(total, start + CHUNK_ROWS)
        key = _scenario_key(system, scenario, chunk_index)
        chunk_file = _chunk_path(run_dir, system, scenario, chunk_index)
        expected = {"system": system, "scenario": scenario, "start": start, "stop": stop}
        row_labels = labels[start:stop]
        source_rows = indices[start:stop]
        saved = manifest["completed_chunks"].get(key)
        if saved is not None:
            _require((run_dir / saved).resolve() == chunk_file.resolve(), f"resume chunk path differs from task identity: {key}")
            payload = _load_saved_chunk(chunk_file, expected, row_labels, source_rows, tuple(evaluators))
        elif chunk_file.exists():
            payload = _load_saved_chunk(chunk_file, expected, row_labels, source_rows, tuple(evaluators))
            manifest["completed_chunks"][key] = chunk_file.relative_to(run_dir).as_posix()
            _save_manifest(manifest_path, manifest)
        else:
            raw_parts: list[torch.Tensor] = []
            power_parts: list[torch.Tensor] = []
            coefficient_parts: list[torch.Tensor] = []
            view_parts: list[torch.Tensor] = []
            logit_parts: dict[str, list[torch.Tensor]] = {name: [] for name in evaluators}
            for batch_start in range(start, stop, FORWARD_BATCH_ROWS):
                batch_stop = min(stop, batch_start + FORWARD_BATCH_ROWS)
                row_slice = slice(batch_start, batch_stop)
                z_batch = coefficients[row_slice] if coefficients is not None else None
                image_batch = images[row_slice] if images is not None else None
                with torch.no_grad():
                    raw_device, power_device, measured_coefficients = _scenario_forward(model, system, z_batch, image_batch, device)
                    _require(tuple(raw_device.shape[1:]) == ((1, 200, 200) if system == "mainline" else (1, 400, 400)), f"{system} raw CMOS geometry changed")
                    _require(bool(torch.isfinite(raw_device).all()) and bool(torch.isfinite(power_device).all()) and bool((power_device > 0).all()), f"{system} produced invalid raw output or optical plane power")
                    _require(measured_coefficients.shape == (batch_stop - batch_start, 100) and bool(torch.isfinite(measured_coefficients).all()), f"{system} produced invalid complex OLS coefficients")
                    view, logits = _evaluate_raw(raw_device, system, evaluators, device)
                raw_parts.append(raw_device.detach().cpu().contiguous())
                power_parts.append(power_device.detach().cpu().to(torch.float64))
                coefficient_parts.append(measured_coefficients.detach().cpu().contiguous())
                view_parts.append(view)
                for name in evaluators:
                    logit_parts[name].append(logits[name])
                del raw_device, power_device, measured_coefficients, view, logits
            z_cpu = torch.cat(coefficient_parts)
            raw_cpu = torch.cat(raw_parts)
            power_cpu = torch.cat(power_parts)
            view_cpu = torch.cat(view_parts)
            logits_cpu = {name: torch.cat(parts) for name, parts in logit_parts.items()}
            payload = {
                **expected,
                "source_row_indices": indices[row_slice].clone(),
                "labels": labels[row_slice].clone(),
                "ols_coefficients": z_cpu,
                "raw_cmos": raw_cpu,
                "evaluator_view_28x28": view_cpu,
                "input_plane_integrated_power": power_cpu,
                "evaluator_logits": logits_cpu,
                "updates": 0,
            }
            chunk_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save_new(chunk_file, payload)
            manifest["completed_chunks"][key] = chunk_file.relative_to(run_dir).as_posix()
            _save_manifest(manifest_path, manifest)
        collected["power"].append(payload["input_plane_integrated_power"].to(torch.float64))
        collected["coefficients"].append(payload["ols_coefficients"])
        collected["logits"].append(payload["evaluator_logits"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return {
        "coefficients": torch.cat(collected["coefficients"]) if collected["coefficients"] else torch.empty((0, 100), dtype=torch.complex64),
        "power": torch.cat(collected["power"]),
        "logits": {name: torch.cat([chunk[name] for chunk in collected["logits"]]) for name in evaluators},
    }


def _power_match_coefficients(coefficients: torch.Tensor, target_power: torch.Tensor, source_power: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _require(coefficients.ndim == 2 and coefficients.shape[1] == 100 and target_power.shape == source_power.shape == (coefficients.shape[0],), "power match vectors are misaligned")
    _require(bool(torch.isfinite(target_power).all()) and bool(torch.isfinite(source_power).all()) and bool((target_power > 0).all()) and bool((source_power > 0).all()), "power matching requires finite positive measured plane powers")
    scales = torch.sqrt(target_power.to(torch.float64) / source_power.to(torch.float64)).to(coefficients.real.dtype)
    matched = coefficients * scales.unsqueeze(1)
    _require(bool(torch.isfinite(matched).all()), "power-matched coefficients are not finite")
    return matched.contiguous(), scales


def _exact_one_sided_mcnemar(original: torch.Tensor, counterfactual: torch.Tensor, labels: torch.Tensor, class_id: int) -> dict[str, Any]:
    mask = labels == class_id
    _require(int(mask.sum()) == 10, "paired class observation must contain ten rows")
    original_correct = original[mask] == labels[mask]
    counter_correct = counterfactual[mask] == labels[mask]
    wins = int((original_correct & ~counter_correct).sum())
    losses = int((~original_correct & counter_correct).sum())
    discordant = wins + losses
    pvalue = (sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / (2**discordant)) if discordant else 1.0
    return {"class_id": class_id, "original_only_correct": wins, "counterfactual_only_correct": losses, "discordant_pairs": discordant, "one_sided_exact_p_original_better": pvalue}


def _holm(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalue))
        adjusted[key] = running
    return adjusted


def _gaussian_blur(view: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    coords = torch.arange(-2, 3, dtype=view.dtype, device=view.device)
    kernel1 = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel1 = kernel1 / kernel1.sum()
    kernel = (kernel1[:, None] * kernel1[None, :]).view(1, 1, 5, 5)
    padded = F.pad(view, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(padded, kernel).clamp(0.0, 1.0)


def _shift_right_one(view: torch.Tensor) -> torch.Tensor:
    padded = F.pad(view, (1, 0, 0, 0), mode="reflect")
    return padded[..., :, :-1]


def _presentation_readout(views: torch.Tensor, labels: torch.Tensor, evaluators: Mapping[str, torch.nn.Module], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    transforms = {"fixed_view_original": lambda x: x, "fixed_view_shift_right_1px": _shift_right_one, "fixed_view_gaussian_5x5_sigma_0_5_reflect": _gaussian_blur}
    for transform_name, transform in transforms.items():
        logits: dict[str, list[torch.Tensor]] = {name: [] for name in evaluators}
        for batch in views.split(30):
            changed = transform(batch.clamp(0.0, 1.0))
            with torch.no_grad():
                for name, model in evaluators.items():
                    logits[name].append(replay400.evaluator_logits(model, changed.to(device)).detach().cpu())
        result[transform_name] = {name: replay400.confusion_metrics(torch.cat(chunks).argmax(1), labels) for name, chunks in logits.items()}
    return result


def _collect_views(run_dir: Path, system: str, scenario: str, count: int = TOTAL_ROWS) -> torch.Tensor:
    values = []
    for chunk_index, start in enumerate(range(0, count, CHUNK_ROWS)):
        path = _chunk_path(run_dir, system, scenario, chunk_index)
        payload = _load_torch(path, mmap=False)
        values.append(payload["evaluator_view_28x28"])
    return torch.cat(values)


def _collect_raw(run_dir: Path, system: str, scenario: str) -> torch.Tensor:
    values = []
    for chunk_index, _ in enumerate(range(0, TOTAL_ROWS, CHUNK_ROWS)):
        payload = _load_torch(_chunk_path(run_dir, system, scenario, chunk_index), mmap=False)
        values.append(payload["raw_cmos"])
    return torch.cat(values)


def _collect_logits_from_chunks(run_dir: Path, system: str, scenario: str, evaluator_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in evaluator_names}
    for chunk_index, start in enumerate(range(0, TOTAL_ROWS, CHUNK_ROWS)):
        payload = _load_torch(_chunk_path(run_dir, system, scenario, chunk_index), mmap=False)
        for name in evaluator_names:
            collected[name].append(payload["evaluator_logits"][name])
    return {name: torch.cat(parts) for name, parts in collected.items()}


def _row_metrics(labels: torch.Tensor, logits: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, values in logits.items():
        predictions = values.argmax(1)
        true_logit = values.gather(1, labels[:, None]).squeeze(1)
        masked = values.clone()
        masked.scatter_(1, labels[:, None], -torch.inf)
        margin = true_logit - masked.max(1).values
        output[name] = {
            "prediction": predictions.tolist(),
            "correct": (predictions == labels).tolist(),
            "true_class_logit_margin": margin.tolist(),
        }
    return output


def _comparison_for_system(
    system: str,
    labels: torch.Tensor,
    base_logits: Mapping[str, torch.Tensor],
    shuffled_logits: Mapping[int, Mapping[str, torch.Tensor]],
    matched_logits: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    observation = torch.arange(FIT_ROWS, TOTAL_ROWS)
    obs_labels = labels[observation]
    base_predictions = base_logits["E_D"].argmax(1).index_select(0, observation)
    result: dict[str, Any] = {"baseline_support_floor": {}, "seed_comparisons": {}}
    baseline_metrics = _metric_split(labels, base_logits)["observation_100_rows"]["E_D"]
    baseline_ec = _metric_split(labels, base_logits)["observation_100_rows"]["E_C"]
    for digit in FOCUS_CLASSES:
        result["baseline_support_floor"][str(digit)] = {
            "E_D_recall_at_least_0_50": baseline_metrics["per_class"][digit]["recall"] >= 0.50,
            "E_C_recall_at_least_0_50": baseline_ec["per_class"][digit]["recall"] >= 0.50,
            "E_D_recall": baseline_metrics["per_class"][digit]["recall"],
            "E_C_recall": baseline_ec["per_class"][digit]["recall"],
        }
    for seed in PORT_SHUFFLE_SEEDS:
        name = f"port_shuffle_seed_{seed}_power_matched"
        ed = matched_logits[seed]["E_D"].argmax(1).index_select(0, observation)
        ec_base = baseline_ec
        ec_shuf = _metric_split(labels, matched_logits[seed])["observation_100_rows"]["E_C"]
        ed_base = baseline_metrics
        ed_shuf = _metric_split(labels, matched_logits[seed])["observation_100_rows"]["E_D"]
        paired = {str(digit): _exact_one_sided_mcnemar(base_predictions, ed, obs_labels, digit) for digit in FOCUS_CLASSES}
        result["seed_comparisons"][str(seed)] = {
            "scenario": name,
            "E_D": ed_shuf,
            "E_C": ec_shuf,
            "E_D_macro_recall_drop": ed_base["macro_recall"] - ed_shuf["macro_recall"],
            "per_class_E_D_recall_drop": {str(digit): ed_base["per_class"][digit]["recall"] - ed_shuf["per_class"][digit]["recall"] for digit in FOCUS_CLASSES},
            "per_class_E_C_recall_drop": {str(digit): ec_base["per_class"][digit]["recall"] - ec_shuf["per_class"][digit]["recall"] for digit in FOCUS_CLASSES},
            "paired_E_D_mcnemar": paired,
        }
    return result


def _endpoint_support(comparison: Mapping[str, Any], adjusted_p: Mapping[str, float], endpoint: str) -> bool:
    floors = comparison["baseline_support_floor"]
    if not all(floors[str(digit)]["E_D_recall_at_least_0_50"] and floors[str(digit)]["E_C_recall_at_least_0_50"] for digit in FOCUS_CLASSES):
        return False
    for seed in PORT_SHUFFLE_SEEDS:
        result = comparison["seed_comparisons"][str(seed)]
        if result["E_D_macro_recall_drop"] < 0.10:
            return False
        for digit in FOCUS_CLASSES:
            if result["per_class_E_D_recall_drop"][str(digit)] < 0.20:
                return False
            if result["per_class_E_C_recall_drop"][str(digit)] < 0.10:
                return False
    for digit in FOCUS_CLASSES:
        key = f"{endpoint}:class{digit}"
        if adjusted_p.get(key, 1.0) > 0.05:
            return False
    return True


def _draw_observation_montage(path: Path, main_original: torch.Tensor, main_shuffled: torch.Tensor, proxy_original: torch.Tensor, proxy_shuffled: torch.Tensor, labels: torch.Tensor) -> None:
    cell = 76
    gutter = 5
    rows = (("2048 mainline · digit 1", 1, main_original, main_shuffled), ("2048 mainline · digit 7", 7, main_original, main_shuffled), ("400 proxy · digit 1", 1, proxy_original, proxy_shuffled), ("400 proxy · digit 7", 7, proxy_original, proxy_shuffled))
    width = 20 * (cell + gutter) + 240
    height = len(rows) * (cell + 36) + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (title, digit, original, shuffled) in enumerate(rows):
        top = 10 + row_index * (cell + 36)
        draw.text((8, top + 22), title, fill="black")
        locations = torch.nonzero(labels == digit).flatten().tolist()
        locations = [index for index in locations if index >= FIT_ROWS]
        for column, sample in enumerate(locations):
            for side, bank in enumerate((original, shuffled)):
                view = bank[sample, 0].clamp(0, 1)
                image = Image.fromarray((view.numpy() * 255.0).round().astype("uint8"), mode="L").resize((cell, cell), Image.Resampling.NEAREST).convert("RGB")
                x = 240 + (side * 10 + column) * (cell + gutter)
                canvas.paste(image, (x, top))
                if column == 0:
                    draw.text((x, top + cell + 2), "orig" if side == 0 else "matched", fill="black")
    canvas.save(path)


def _save_final_reports(
    run_dir: Path,
    manifest: dict[str, Any],
    inputs: Mapping[str, Any],
    paths: Mapping[str, Path],
    preflight: Mapping[str, Any],
    main_base: Mapping[str, Any],
    main_rows: Mapping[str, Any],
    proxy_base_logits: Mapping[str, torch.Tensor],
    proxy_base_power: torch.Tensor,
    proxy_shuffled: Mapping[int, Mapping[str, Any]],
    proxy_matched: Mapping[int, Mapping[str, Any]],
    main_shuffled: Mapping[int, Mapping[str, Any]],
    main_matched: Mapping[int, Mapping[str, Any]],
    whole_logits: Mapping[str, Mapping[str, torch.Tensor]],
    sensitivity: Mapping[str, Any],
    generated_history_metrics: Mapping[str, Any],
    output_paths: Mapping[str, str],
    started: float,
) -> dict[str, Any]:
    labels = inputs["labels"]
    all_names = tuple(inputs["evaluators"])
    main_base_metrics = _metric_split(labels, main_base["logits"])
    proxy_base_metrics = _metric_split(labels, proxy_base_logits)
    main_comparison = _comparison_for_system("mainline", labels, main_base["logits"], {seed: main_shuffled[seed]["logits"] for seed in PORT_SHUFFLE_SEEDS}, {seed: main_matched[seed]["logits"] for seed in PORT_SHUFFLE_SEEDS})
    proxy_comparison = _comparison_for_system("proxy400", labels, proxy_base_logits, {seed: proxy_shuffled[seed]["logits"] for seed in PORT_SHUFFLE_SEEDS}, {seed: proxy_matched[seed]["logits"] for seed in PORT_SHUFFLE_SEEDS})
    pvalues: dict[str, float] = {}
    for endpoint, comparison in (("mainline", main_comparison), ("proxy400", proxy_comparison)):
        first_seed = comparison["seed_comparisons"][str(PORT_SHUFFLE_SEEDS[0])]["paired_E_D_mcnemar"]
        for digit in FOCUS_CLASSES:
            pvalues[f"{endpoint}:class{digit}"] = first_seed[str(digit)]["one_sided_exact_p_original_better"]
    adjusted = _holm(pvalues)
    main_support = _endpoint_support(main_comparison, adjusted, "mainline")
    proxy_support = _endpoint_support(proxy_comparison, adjusted, "proxy400")
    if main_support:
        classification = "NATIVE_OLS_JOINT_DEPENDENCE_CANDIDATE"
        recommendation = "只建议将跨端口复 OLS 依赖作为一项后续受控诊断候选；本报告不授权或启动训练。"
    elif proxy_support:
        classification = "PROXY_ONLY_DEPENDENCE_SUPPORT"
        recommendation = "NONE：依赖效应只在400代理光路得到支持，不足以改变LightGen主线训练方向。"
    elif not all(main_comparison["baseline_support_floor"][str(d)] ["E_D_recall_at_least_0_50"] and main_comparison["baseline_support_floor"][str(d)]["E_C_recall_at_least_0_50"] for d in FOCUS_CLASSES):
        classification = "MIXED_OR_INSUFFICIENT_BASELINE_FLOOR"
        recommendation = "NONE：主线类别1/7的未置换读数未跨过预设基线支持线，置换效应无法归因。"
    elif any(_endpoint_support(comparison, adjusted, endpoint) for endpoint, comparison in (("mainline", main_comparison), ("proxy400", proxy_comparison))):
        classification = "MIXED_OR_INSUFFICIENT"
        recommendation = "NONE：模型路径或类别之间的依赖证据不一致。"
    else:
        classification = "JOINT_DEPENDENCE_NOT_SUPPORTED_WITHIN_SCOPE"
        recommendation = "NONE：在本次固定人口、模型状态和两种置换下，未获得预设阈值要求的跨端口依赖证据。"
    positive = preflight["target_positive_control"]["E_D"]
    measurement_qualification = positive["macro_recall"] >= 0.90 and positive["recall_by_class"][1] >= 0.80 and positive["recall_by_class"][7] >= 0.80
    if not measurement_qualification:
        main_support = False
        proxy_support = False
        classification = "MEASUREMENT_QUALIFICATION_INSUFFICIENT"
        recommendation = "NONE：真实目标 E_D 正控未达预声明资格线，置换读数只保留为描述性测量，不作模型归因。"
    whole_control = {}
    for endpoint in ("mainline", "proxy400"):
        original = main_base["logits"] if endpoint == "mainline" else proxy_base_logits
        shuffled = whole_logits[endpoint]
        original_metrics = _metric_split(labels, original)["observation_100_rows"]
        shuffled_metrics = _metric_split(labels, shuffled)["observation_100_rows"]
        equality = {name: original_metrics[name]["confusion"] == shuffled_metrics[name]["confusion"] for name in all_names}
        _require(all(equality.values()), f"{endpoint} shared whole-row negative control changed class confusion")
        whole_control[endpoint] = {"confusion_equal_per_evaluator": equality}

    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": classification,
        "decision_basis": "Unmatched within-class, within-fit/observation independent port permutations preserve each per-port complex marginal exactly. Power-matched versions apply a row-global scale and therefore no longer preserve those exact marginals; they isolate the effect of joint OLS arrangement at equal measured generator-input plane power. All optical parameters and evaluators remain frozen.",
        "scientific_claim": "This is a fixed-population counterfactual diagnosis of whether joint complex OLS row structure contributes to evaluator readout. It does not establish independent generalization, successful full semantic generation, or an architecture-capacity limit.",
        "training_updates": 0,
        "optimizer_created": False,
        "autograd_enabled": False,
        "restricted_reads": {"DEV": False, "protected_test": False, "sealed": False, "other_MNIST_rows_materialized": False},
        "data_roles": {
            "rows": "300 rows from canonical seed1234 TRAIN; first 200 are fit-role diagnostic inputs, last 100 are observation-role diagnostic inputs",
            "not_independent_generalization": True,
            "row_indices": inputs["indices"].tolist(),
            "labels": labels.tolist(),
            "fit_count": FIT_ROWS,
            "observation_count": OBSERVATION_ROWS,
            "class_counts": torch.bincount(labels, minlength=10).tolist(),
        },
        "models": {
            "mainline": {"checkpoint_step": MAINLINE_STEP, "optics": "complete LightGen 2048 propagation grid; real image-to-Encoder-to-OLS-to-W_OLS-to-Generator path", "configuration": {"production_propagation_grid": 2048, "active_cmos": [200, 200], "ols_ports": 100}, "shared_reference_origin_step": 1200},
            "proxy400": {"checkpoint_step": PROXY_STEP, "optics": "saved 400x400 2-layer Encoder, 100 fiber modes, 4-layer Generator; original 512 propagation grid; diagnostic proxy", "configuration": inputs["proxy_config_record"], "base_300_row_readout_reused_from": str(paths["proxy_replay"] / "final_readouts.pt")},
            "cross_model_comparison": "descriptive only; geometry, optical path, and training state differ",
        },
        "counterfactual_design": {
            "ols_port_order": "row-major j=10*r+c",
            "independent_per_port_seeds": list(PORT_SHUFFLE_SEEDS),
            "permutation_partitions": "within class and separately within the first 200 fit-role and last 100 observation-role rows",
            "complex_coefficients": "real and imaginary parts move together as one coefficient; different ports use independent row permutations",
            "unmatched_marginals": "each class/role/port complex coefficient multiset is preserved exactly",
            "power_match": "whole shuffled complex row is scaled by sqrt(original measured generator-input plane power / shuffled measured generator-input plane power); reports that this scaling changes exact per-port marginals",
            "mainline_power_measurement": "integrated |W_OLS(z)|^2 on the physical generator entrance plane using its PlaneSpec pixel area",
            "proxy_power_measurement": "integrated |sum_j z_j v_j|^2 on the fiber re-emission plane using the original 400 model PlaneSpec pixel area",
            "negative_control": "one shared within-class row permutation moves all 100 complex ports together",
        },
        "final_frozen_state_assertion": inputs.get("final_freeze_assertion"),
        "preflight": preflight,
        "target_positive_control": preflight["target_positive_control"],
        "observed": {
            "mainline_original_300": main_base_metrics,
            "proxy400_original_300_reused": proxy_base_metrics,
            "mainline_port_shuffle_unmatched": {str(seed): _metric_split(labels, main_shuffled[seed]["logits"]) for seed in PORT_SHUFFLE_SEEDS},
            "proxy400_port_shuffle_unmatched": {str(seed): _metric_split(labels, proxy_shuffled[seed]["logits"]) for seed in PORT_SHUFFLE_SEEDS},
            "mainline_port_shuffle_power_matched": {str(seed): _metric_split(labels, main_matched[seed]["logits"]) for seed in PORT_SHUFFLE_SEEDS},
            "proxy400_port_shuffle_power_matched": {str(seed): _metric_split(labels, proxy_matched[seed]["logits"]) for seed in PORT_SHUFFLE_SEEDS},
            "shared_whole_row_negative_control": whole_control,
            "power_match": {
                endpoint: {str(seed): {"baseline_integrated_power_mean": float((main_base["power"] if endpoint == "mainline" else proxy_base_power).mean()), "counterfactual_power_mean": float(matched[seed]["power"].mean()), "relative_error_max": float((((matched[seed]["power"] - (main_base["power"] if endpoint == "mainline" else proxy_base_power)).abs()) / (main_base["power"] if endpoint == "mainline" else proxy_base_power)).max())} for seed in PORT_SHUFFLE_SEEDS}
                for endpoint, matched in (("mainline", main_matched), ("proxy400", proxy_matched))
            },
            "presentation_sensitivity_fixed_views": sensitivity,
            "saved_400_generated_2000_readout_reused": generated_history_metrics,
        },
        "primary_comparison": {
            "population": "100 held-back rows within the same fixed TRAIN population; descriptive, not a separate split",
            "focus_classes": list(FOCUS_CLASSES),
            "test": "one-sided exact paired McNemar, original correct vs power-matched port-shuffled correct; first shuffle seed is primary",
            "family_size": 4,
            "raw_p": pvalues,
            "holm_adjusted_p": adjusted,
            "mainline_support": main_support,
            "proxy400_support": proxy_support,
            "target_E_D_positive_control_qualified": measurement_qualification,
            "predeclared_effect_thresholds": {"baseline_E_D_and_E_C_class_recall": 0.50, "E_D_class_drop_each_seed": 0.20, "E_C_class_drop_each_seed": 0.10, "E_D_macro_drop_each_seed": 0.10, "primary_adjusted_p": 0.05},
            "mainline_diagnosis": main_comparison,
            "proxy400_diagnosis": proxy_comparison,
        },
        "per_row_readouts": {
            "mainline_original": _row_metrics(labels, main_base["logits"]),
            "mainline_power_matched_seed_2026100401": _row_metrics(labels, main_matched[PORT_SHUFFLE_SEEDS[0]]["logits"]),
            "proxy400_original": _row_metrics(labels, proxy_base_logits),
            "proxy400_power_matched_seed_2026100401": _row_metrics(labels, proxy_matched[PORT_SHUFFLE_SEEDS[0]]["logits"]),
        },
        "artifacts": dict(output_paths),
        "training_recommendation": recommendation,
        "next_action": "NONE",
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report


def _make_identity(inputs: Mapping[str, Any], paths: Mapping[str, Path]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="step2200-mainline-versus-step600-400-proxy-fixed-train-ols-counterfactuals",
        semantic_authority_id="M0_NATIVE_OLS_DEPENDENCE_AND_PRESENTATION_DIAGNOSTIC_V1",
        data_role="fixed canonical TRAIN 200 fit-role and 100 observation-role rows; not independent generalization",
        config_identity={"mainline_propagation_grid": 2048, "proxy_config": str(paths["proxy_config"]), "proxy_config_values": inputs["proxy_config_record"]},
        model_identity={"mainline_step": MAINLINE_STEP, "mainline_parameter_shapes": {key: list(value.shape) for key, value in inputs["mainline_parameters"].items()}, "proxy_step": PROXY_STEP, "proxy_model": "diagnostic_lightgen_amplitude_fiber_ols_400"},
        optimizer_identity={"created": False, "updates": 0},
        sampler_identity={"kind": "fixed within-class row and independent per-port counterfactual permutations", "port_seeds": list(PORT_SHUFFLE_SEEDS), "whole_row_seed": WHOLE_ROW_SHUFFLE_SEED},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )


def _receipt(control: ResearchRunControl, preflight: Mapping[str, Any]) -> PreflightReceipt:
    return control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "frozen_models_and_evaluators": True, "fixed_train_population": True, "production_and_proxy_paths": True, "no_autograd_updates": True},
        resource={"passed": True, "status": "PASS", "cuda_runtime": True, "telemetry": preflight["resource_after"]},
        implementation={"passed": True, "status": "PASS", "selected_engine": "cuda_frozen_no_grad", "output_checks": "explicit paths, chunked writes and resumable readouts"},
        selected_engine="cuda_frozen_no_grad",
        resource_envelope={"cuda": preflight["resource_after"], "disk_free_bytes_observed": preflight["output_drive_free_bytes"]},
    )


def _write_preflight_only(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_new_json(path, dict(record))


def _validate_resume_identity(manifest: Mapping[str, Any], identity: ResearchRunIdentity, paths: Mapping[str, Path], inputs: Mapping[str, Any]) -> None:
    _require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("task_id") == TASK_ID, "resume manifest belongs to another task")
    actual_identity = ResearchRunIdentity.from_mapping(manifest.get("identity", {}))
    _require(actual_identity.execution_semantics() == identity.execution_semantics(), "resume manifest model/config/data semantics changed")
    _require(Path(manifest.get("output_dir", "")).resolve() == paths["output_dir"], "resume manifest output path changed")
    _require(manifest.get("population_indices") == inputs["indices"].tolist() and manifest.get("population_labels") == inputs["labels"].tolist(), "resume manifest fixed population changed")
    receipt = manifest.get("preflight_receipt")
    _require(isinstance(receipt, Mapping) and receipt.get("status") == "READY", "resume manifest has no consumable READY receipt")


def _run(args: argparse.Namespace) -> int:
    root = _root()
    paths = _paths(root, args)
    class_report, proxy_report = _validate_prior_authorities(paths)
    _require(paths["output_dir"] != paths["proxy_replay"] and paths["output_dir"] != paths["proxy_artifact"], "output directory overlaps a protected input")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the frozen production-path diagnostic")
    device = torch.device("cuda:0")
    inputs = _load_exact_inputs(paths, device)
    identity = _make_identity(inputs, paths)
    run_dir = paths["output_dir"]
    manifest_path = run_dir / "resume_manifest.json"
    resume_manifest_path = _resolve(root, args.resume_manifest) if args.resume_manifest else None
    if run_dir.exists() and resume_manifest_path is None:
        raise DiagnosticError(f"output directory already exists; explicit --resume-manifest is required: {run_dir}")
    if resume_manifest_path is not None:
        _require(resume_manifest_path.resolve() == manifest_path.resolve(), "--resume-manifest must name this output directory's resume_manifest.json")
        manifest = _read_json(resume_manifest_path)
        _validate_resume_identity(manifest, identity, paths, inputs)
        frozen_parameter_file = run_dir / "mainline_frozen_parameters.pt"
        _require(frozen_parameter_file.is_file(), "resume output is missing the saved mainline parameter state")
        saved_parameters = _load_torch(frozen_parameter_file, mmap=False)
        _require(_state_equal(saved_parameters, inputs["mainline_parameters"]), "resume mainline parameter values differ from the explicit step2200 checkpoint")
        if manifest.get("status") == "VALID_COMPLETED":
            report_path = run_dir / "report.json"
            _require(report_path.is_file(), "completed resume manifest has no terminal report")
            report = _read_json(report_path)
            _require(report.get("status") == "VALID_COMPLETED" and report.get("next_action") == "NONE", "completed task report is not terminal")
            print(json.dumps({"status": report["status"], "classification": report["classification"], "report": str(report_path), "additional_forward": False}, ensure_ascii=False))
            return 0

    started = time.perf_counter()
    preflight = _preflight(inputs, paths, device)
    preflight["input_authorities"] = {"class17_status": class_report["status"], "class17_classification": class_report["classification"], "proxy_replay_status": proxy_report["status"], "proxy_replay_classification": proxy_report["classification"]}
    control = ResearchRunControl(identity=identity, authorization_id=f"{TASK_ID}:authorization:01")
    receipt = _receipt(control, preflight)
    control.mark_preflight_ready(receipt)
    if args.preflight_only:
        control.start_execution()
        record = {"task_id": TASK_ID, "status": "PREFLIGHT_READY", "receipt": receipt.to_dict(), "preflight": preflight, "training_updates": 0, "next_action": "NONE"}
        preflight_path = run_dir.parent / f"{run_dir.name}_preflight_only.json"
        _write_preflight_only(preflight_path, record)
        print(json.dumps({"status": "PREFLIGHT_READY", "preflight": str(preflight_path), "training_updates": 0}, ensure_ascii=False))
        return 0

    if resume_manifest_path is None:
        control.start_execution()
        control.open_scientific_attempt(run_dir, f"{TASK_ID}:attempt-01", receipt)
        main_parameters_path = run_dir / "mainline_frozen_parameters.pt"
        atomic_torch_save_new(main_parameters_path, inputs["mainline_parameters"])
        atomic_write_new_json(run_dir / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight})
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "task_id": TASK_ID,
            "status": "RUNNING",
            "output_dir": str(run_dir),
            "identity": control.identity.to_dict(),
            "preflight_receipt": receipt.to_dict(),
            "population_indices": inputs["indices"].tolist(),
            "population_labels": inputs["labels"].tolist(),
            "completed_chunks": {},
            "updates": 0,
            "created_at_unix": time.time(),
        }
        atomic_write_new_json(manifest_path, manifest)
    else:
        receipt = PreflightReceipt.from_mapping(manifest["preflight_receipt"])
        control.mark_preflight_ready(receipt)
        control.start_execution()
        control.resume_scientific_attempt(run_dir, f"{TASK_ID}:attempt-01", receipt)
        manifest["status"] = "RUNNING"
        _save_manifest(manifest_path, manifest)

    labels = inputs["labels"]
    indices = inputs["indices"]
    images = inputs["images"]
    evaluators = inputs["evaluators"]
    try:
        with torch.no_grad():
            main_base = _measure_scenario(
                run_dir=run_dir, manifest=manifest, manifest_path=manifest_path, system="mainline", scenario="original_encoded",
                model=inputs["consumer"], labels=labels, indices=indices, device=device, evaluators=evaluators, images=images,
            )
            proxy_base_logits = {name: values.detach().cpu() for name, values in inputs["final_readouts"]["reference_logits"].items()}
            proxy_base = {
                "coefficients": inputs["final_readouts"]["reference_ols"].detach().cpu(),
                "power": inputs["proxy_base_power"].clone(),
                "logits": proxy_base_logits,
            }
            maps: dict[str, torch.Tensor] = {}
            coefficients_by_endpoint = {"mainline": main_base["coefficients"], "proxy400": proxy_base["coefficients"]}
            for system, coefficients in coefficients_by_endpoint.items():
                for seed in PORT_SHUFFLE_SEEDS:
                    mapping = _port_permutation_indices(labels, seed=seed)
                    shuffled = _shuffled_coefficients(coefficients, mapping)
                    _verify_port_marginals(coefficients, shuffled, mapping, labels)
                    maps[f"{system}:port:{seed}"] = mapping
                mapping = _port_permutation_indices(labels, seed=WHOLE_ROW_SHUFFLE_SEED, shared_row_permutation=True)
                maps[f"{system}:whole_row"] = mapping
            permutation_path = run_dir / "counterfactual_permutation_maps.pt"
            permutation_payload = {"schema": "native_ols_permutation_maps.v1", "maps": maps, "fit_rows": FIT_ROWS, "observation_rows": OBSERVATION_ROWS, "port_order": "row-major j=10*r+c", "complex_pair_preserved": True}
            if permutation_path.exists():
                prior_maps = _load_torch(permutation_path, mmap=False)
                _require(prior_maps.get("schema") == permutation_payload["schema"] and set(prior_maps.get("maps", {})) == set(maps), "saved counterfactual permutation maps changed")
                _require(all(torch.equal(prior_maps["maps"][key], value) for key, value in maps.items()), "saved counterfactual permutation map values changed")
            else:
                atomic_torch_save_new(permutation_path, permutation_payload)

            main_shuffled: dict[int, dict[str, Any]] = {}
            proxy_shuffled: dict[int, dict[str, Any]] = {}
            whole_logits: dict[str, Mapping[str, torch.Tensor]] = {}
            for system, model, base, target in (
                ("mainline", inputs["consumer"], main_base, main_shuffled),
                ("proxy400", inputs["proxy_model"], proxy_base, proxy_shuffled),
            ):
                for seed in PORT_SHUFFLE_SEEDS:
                    scenario = f"port_shuffle_seed_{seed}_unmatched"
                    z_shuffled = _shuffled_coefficients(base["coefficients"], maps[f"{system}:port:{seed}"])
                    target[seed] = _measure_scenario(
                        run_dir=run_dir, manifest=manifest, manifest_path=manifest_path, system=system, scenario=scenario,
                        model=model, labels=labels, indices=indices, device=device, evaluators=evaluators, coefficients=z_shuffled,
                    )
                z_whole = _shuffled_coefficients(base["coefficients"], maps[f"{system}:whole_row"])
                whole = _measure_scenario(
                    run_dir=run_dir, manifest=manifest, manifest_path=manifest_path, system=system, scenario="whole_row_control",
                    model=model, labels=labels, indices=indices, device=device, evaluators=evaluators, coefficients=z_whole,
                )
                baseline_raw = _collect_raw(run_dir, "mainline", "original_encoded") if system == "mainline" else inputs["final_readouts"]["reference_raw_cmos_400x400"].detach().cpu()
                whole_raw = _collect_raw(run_dir, system, "whole_row_control")
                source_rows = maps[f"{system}:whole_row"][:, 0]
                permuted_expected = baseline_raw.index_select(0, source_rows)
                _require(torch.allclose(whole_raw, permuted_expected, rtol=1.0e-5, atol=1.0e-6), f"{system} shared whole-row negative control did not replay complete OLS rows")
                expected_power = base["power"].index_select(0, source_rows)
                _require(torch.allclose(whole["power"], expected_power, rtol=1.0e-4, atol=1.0e-15), f"{system} shared whole-row control changed integrated generator-input power")
                whole_logits[system] = whole["logits"]

            main_matched: dict[int, dict[str, Any]] = {}
            proxy_matched: dict[int, dict[str, Any]] = {}
            for system, model, base, shuffled_map, target in (
                ("mainline", inputs["consumer"], main_base, main_shuffled, main_matched),
                ("proxy400", inputs["proxy_model"], proxy_base, proxy_shuffled, proxy_matched),
            ):
                for seed in PORT_SHUFFLE_SEEDS:
                    unscaled = shuffled_map[seed]
                    z_unmatched = _shuffled_coefficients(base["coefficients"], maps[f"{system}:port:{seed}"])
                    z_matched, scales = _power_match_coefficients(z_unmatched, base["power"], unscaled["power"])
                    power_match_path = run_dir / f"{system}_power_match_seed_{seed}.pt"
                    power_match_payload = {"coefficients": z_matched, "row_scale": scales, "target_integrated_power": base["power"], "unmatched_integrated_power": unscaled["power"], "fit_count": FIT_ROWS, "observation_count": OBSERVATION_ROWS}
                    if power_match_path.exists():
                        saved_power_match = _load_torch(power_match_path, mmap=False)
                        _require(torch.equal(saved_power_match["coefficients"], z_matched) and torch.equal(saved_power_match["row_scale"], scales), f"saved power-matched OLS coefficients changed: {system}/{seed}")
                    else:
                        atomic_torch_save_new(power_match_path, power_match_payload)
                    scenario = f"port_shuffle_seed_{seed}_power_matched"
                    target[seed] = _measure_scenario(
                        run_dir=run_dir, manifest=manifest, manifest_path=manifest_path, system=system, scenario=scenario,
                        model=model, labels=labels, indices=indices, device=device, evaluators=evaluators, coefficients=z_matched,
                    )
                    relative_error = ((target[seed]["power"] - base["power"]).abs() / base["power"]).max()
                    _require(float(relative_error) <= 1.0e-4, f"{system} power-matched counterfactual did not match physical generator-input power")

            for module, state, name in ((inputs["consumer"].runtime, inputs["mainline_state"], "mainline"), (inputs["proxy_model"], inputs["proxy_state"], "proxy400")):
                _assert_frozen(module, state, name)
            for name, model in evaluators.items():
                _assert_frozen(model, inputs["evaluator_states"][name], name)
            inputs["final_freeze_assertion"] = {
                "mainline_model_and_scientific_buffers_unchanged": True,
                "proxy_model_and_fiber_buffers_unchanged": True,
                "all_five_evaluators_unchanged_and_without_gradients": True,
                "optimizer_created": False,
                "training_updates": 0,
            }

            main_base_views = _collect_views(run_dir, "mainline", "original_encoded")
            proxy_base_views = inputs["final_readouts"]["reference_views_28x28"].detach().cpu()
            sensitivity = {
                "mainline": _presentation_readout(main_base_views, labels, evaluators, device),
                "proxy400": _presentation_readout(proxy_base_views, labels, evaluators, device),
            }
            saved_generated_logits = inputs["final_readouts"]["generated_logits"]
            saved_generated_labels = inputs["final_readouts"]["generated_labels"]
            _require(saved_generated_labels.shape == (2000,) and torch.equal(torch.bincount(saved_generated_labels, minlength=10), torch.full((10,), 200)), "saved 400 generated bank is not the original 200-per-class population")
            generated_history_metrics = {name: replay400.confusion_metrics(values.argmax(1).cpu(), saved_generated_labels.cpu()) for name, values in saved_generated_logits.items()}

            montage_path = run_dir / "class17_observation_power_matched_comparison.png"
            main_matched_views = _collect_views(run_dir, "mainline", f"port_shuffle_seed_{PORT_SHUFFLE_SEEDS[0]}_power_matched")
            proxy_matched_views = _collect_views(run_dir, "proxy400", f"port_shuffle_seed_{PORT_SHUFFLE_SEEDS[0]}_power_matched")
            _draw_observation_montage(montage_path, main_base_views, main_matched_views, proxy_base_views, proxy_matched_views, labels)

            output_paths = {"report": "report.json", "manifest": "resume_manifest.json", "montage": montage_path.name, "permutation_maps": "counterfactual_permutation_maps.pt", "chunks": "chunks/<system>/<scenario>/chunk_NNNN.pt", "power_matched_coefficients": "<system>_power_match_seed_<seed>.pt", "mainline_frozen_parameters": "mainline_frozen_parameters.pt", "proxy_frozen_replay_inputs": str(paths["proxy_replay"])}
            report = _save_final_reports(run_dir, manifest, inputs, paths, preflight, main_base, {"unused": None}, proxy_base_logits, inputs["proxy_base_power"], proxy_shuffled, proxy_matched, main_shuffled, main_matched, whole_logits, sensitivity, generated_history_metrics, output_paths, started)
            atomic_write_new_json(run_dir / "report.json", report)
            manifest["status"] = "VALID_COMPLETED"
            manifest["updates"] = 0
            manifest["report"] = "report.json"
            _save_manifest(manifest_path, manifest)
            control.complete()
            print(json.dumps({"status": report["status"], "classification": report["classification"], "training_recommendation": report["training_recommendation"], "report": str(run_dir / "report.json"), "next_action": "NONE"}, ensure_ascii=False))
            return 0
    except KeyboardInterrupt:
        control.interrupted()
        manifest["status"] = "RUN_INTERRUPTED"
        _save_manifest(manifest_path, manifest)
        raise
    except Exception as error:
        control.invalid_harness()
        manifest["status"] = "INVALID_HARNESS"
        manifest["scientific_classification"] = None
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        _save_manifest(manifest_path, manifest)
        atomic_write_new_json(run_dir / "invalid_harness.json", {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "error_type": type(error).__name__, "error": str(error), "updates": 0})
        raise


def _parser() -> argparse.ArgumentParser:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-population", default=str(DEFAULT_POPULATION))
    parser.add_argument("--class17-report", default=str(DEFAULT_CLASS17_REPORT))
    parser.add_argument("--proxy-artifact-dir", default=str(DEFAULT_PROXY_ARTIFACT))
    parser.add_argument("--proxy-config", default=str(DEFAULT_PROXY_CONFIG))
    parser.add_argument("--proxy-replay-dir", default=str(DEFAULT_PROXY_REPLAY))
    parser.add_argument("--training-evaluator", default=str(DEFAULT_TRAINING_EVALUATOR))
    parser.add_argument("--training-data", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-manifest", default=None)
    parser.set_defaults(root=root)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return _run(args)
    except Exception as error:
        print(json.dumps({"status": "INVALID_HARNESS", "classification": None, "error_type": type(error).__name__, "error": str(error), "training_updates": 0}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
