"""Capture A1000 and B5400 fields on one shared, fixed 20-row display sample."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ldm_project.v2.optics import d2nn
from scripts.research.lightgen import visualize_lightgen_architecture_comparison_v1 as capture


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/20261010_lightgen_best_endpoints_10node_visualization_v1/runs/attempt_01"
OLD_OUTPUT = ROOT / "outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01"
SMALL_NAME = "small_A1000"
LARGE_NAME = "large_B5400_fg"
DISPLAY_NODE_MAP = {
    SMALL_NAME: (
        ("slm_output_field", "slm_output_field"),
        ("encoder_layer_1_after_crop", "encoder_layer_1_after_crop"),
        ("encoder_layer_2_after_crop", "encoder_layer_2_after_crop"),
        ("ols_coefficients_10x10", "ols_coefficients_10x10"),
        ("ols_reemission_field", "ols_reemission_field"),
        ("generator_z0p75_free_reference", "generator_z0p75_free_reference"),
        ("generator_layer_1_after_crop", "generator_layer_1_after_crop"),
        ("generator_z2p25_free_reference", "generator_z2p25_free_reference"),
        ("generator_layer_2_after_crop", "generator_layer_2_after_crop"),
        ("cmos_incident_field", "cmos_incident_field"),
    ),
    LARGE_NAME: (
        ("slm_output_field", "slm_output_field"),
        ("encoder_layer_1_after_crop", "encoder_layer_1_after_crop"),
        ("encoder_layer_2_after_crop", "encoder_layer_2_after_crop"),
        ("ols_coefficients_10x10", "ols_coefficients_10x10"),
        ("ols_reemission_field", "ols_reemission_field"),
        ("generator_z0p75_after_phase", "generator_z0p75_after_phase"),
        ("generator_layer_1_after_crop", "generator_layer_1_after_crop"),
        ("generator_z2p25_after_phase", "generator_z2p25_after_phase"),
        ("generator_layer_2_after_crop", "generator_layer_2_after_crop"),
        ("cmos_incident_field", "cmos_incident_field"),
    ),
}


def _endpoint(name: str) -> capture.Endpoint:
    return next(item for item in capture.ENDPOINTS if item.name == name)


def _selection(group: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    positions = list(map(int, group["selected"]))
    rows = [int(group["rows"][position]) for position in positions]
    labels = [int(group["labels"][position]) for position in positions]
    capture.require(len(positions) == 20 and len(set(rows)) == 20, "A1000 display sample must contain 20 unique selected rows")
    capture.require([labels.count(label) for label in range(10)] == [2] * 10, "A1000 display sample must contain two rows per class")
    return positions, rows, labels


def _selected_schema_bytes(schema: dict[str, Any], endpoint_name: str) -> int:
    names = [source for source, _display in DISPLAY_NODE_MAP[endpoint_name]]
    missing = [name for name in names if name not in schema]
    capture.require(not missing, f"{endpoint_name}: preflight omitted display nodes {missing}")
    return sum(int(schema[name]["bytes_per_sample"]) * 20 for name in names)


def _disk_estimate(preflight: dict[str, Any]) -> dict[str, Any]:
    per_endpoint = {
        SMALL_NAME: _selected_schema_bytes(preflight["field_storage"][SMALL_NAME], SMALL_NAME),
        LARGE_NAME: _selected_schema_bytes(preflight["field_storage"][LARGE_NAME], LARGE_NAME),
    }
    field_bytes = sum(per_endpoint.values())
    with_margin = math.ceil(field_bytes * 1.2)
    other_allowance = 2_000_000_000
    reserve = 20 * 1024**3
    estimated = with_margin + other_allowance
    free = shutil.disk_usage(ROOT).free
    capture.require(free > estimated + reserve, f"D: free space is below estimate plus 20 GiB reserve ({free} bytes free)")
    return {
        "field_bytes_by_endpoint": per_endpoint,
        "field_bytes_with_20pct_margin": with_margin,
        "other_outputs_allowance_bytes": other_allowance,
        "estimated_total_bytes": estimated,
        "d_free_bytes_before_run": free,
        "reserve_required_bytes": reserve,
    }


def _check_exception_restoration(consumer: Any, endpoint: capture.Endpoint, rows: list[int], labels: list[int]) -> dict[str, Any]:
    runtime = consumer.runtime
    collector = capture.FieldCollector.create(endpoint, rows[:1], labels[:1])
    original_d2nn = d2nn.angular_spectrum_propagate
    original_capacity = capture.capacity.angular_spectrum_propagate
    original_project = runtime.ols.project_state.__func__
    original_writer = runtime.ols.write_to_generator_input.__func__
    original_adapter = type(runtime.input_adapter).__call__
    original_input_relay = type(runtime.input_relay).__call__
    original_output_relay = type(runtime.output_relay).__call__

    class ExpectedExit(Exception):
        pass

    try:
        with capture.observe_runtime(consumer, collector):
            raise ExpectedExit
    except ExpectedExit:
        pass
    restored = (
        d2nn.angular_spectrum_propagate is original_d2nn
        and capture.capacity.angular_spectrum_propagate is original_capacity
        and runtime.ols.project_state.__func__ is original_project
        and runtime.ols.write_to_generator_input.__func__ is original_writer
        and type(runtime.input_adapter).__call__ is original_adapter
        and type(runtime.input_relay).__call__ is original_input_relay
        and type(runtime.output_relay).__call__ is original_output_relay
        and "_expanded_phase_deviation" not in runtime.generator.__dict__
    )
    capture.require(restored, f"{endpoint.name}: observer hooks did not restore after an exceptional exit")
    return {"executed": True, "passed": True}


def _verify_source_small_fields(
    source_group: dict[str, Any], positions: list[int], rows: list[int], labels: list[int]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    manifest = capture.load_json(OLD_OUTPUT / "field_capture_manifest.json")
    endpoint_record = next(item for item in manifest["endpoints"] if item["endpoint"] == SMALL_NAME)
    expected_positions = set(positions)
    found: dict[int, dict[str, Any]] = {}
    node_names = [source for source, _display in DISPLAY_NODE_MAP[SMALL_NAME]]
    for chunk_record in endpoint_record["chunks"]:
        chunk_path = OLD_OUTPUT / chunk_record["path"]
        chunk = capture._load_field_chunk(chunk_path, expected_endpoint=SMALL_NAME)
        chunk_positions = chunk["positions"].tolist()
        for local, position in enumerate(chunk_positions):
            if position not in expected_positions:
                continue
            capture.require(position not in found, f"A1000 saved fields duplicate selected position {position}")
            source_index = positions.index(position)
            capture.require(int(chunk["source_row_indices"][local]) == rows[source_index], "A1000 saved field source row differs from selected row")
            capture.require(int(chunk["labels"][local]) == labels[source_index], "A1000 saved field label differs from selected label")
            missing = [node for node in node_names if node not in chunk["fields"]]
            capture.require(not missing, f"A1000 saved chunk is missing display nodes {missing}")
            fields = {node: chunk["fields"][node][local : local + 1].clone() for node in node_names}
            metadata = {node: dict(chunk["field_metadata"][node]) for node in node_names}
            found[position] = {"fields": fields, "metadata": metadata}
        del chunk
    capture.require(set(found) == expected_positions, "A1000 saved chunks do not cover the selected 20 rows")
    return [dict(item) for item in endpoint_record["chunks"]], found


def _write_small_field_chunks(
    endpoint: capture.Endpoint,
    positions: list[int],
    rows: list[int],
    labels: list[int],
    source_fields: dict[int, dict[str, Any]],
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    chunk_records: list[dict[str, Any]] = []
    node_names = [source for source, _display in DISPLAY_NODE_MAP[SMALL_NAME]]
    for chunk_index, start in enumerate(range(0, 20, 10)):
        stop = start + 10
        local_positions, local_rows, local_labels = positions[start:stop], rows[start:stop], labels[start:stop]
        collector = capture.FieldCollector.create(endpoint, local_rows, local_labels)
        for node in node_names:
            collector.records[node] = torch.cat([source_fields[position]["fields"][node] for position in local_positions], dim=0).contiguous()
            metadata = dict(source_fields[local_positions[0]]["metadata"][node])
            metadata.update({
                "endpoint": SMALL_NAME,
                "source_row_indices": local_rows,
                "labels": local_labels,
                "data_roles": ["confirmation"] * len(local_rows),
                "node_name": node,
                "shape": list(collector.records[node].shape),
                "complex_dtype": str(collector.records[node].dtype),
            })
            collector.metadata[node] = metadata
        path = OUTPUT / "fields" / SMALL_NAME / f"field_chunk_{chunk_index:03d}.pt"
        if path.exists():
            capture.require(resume, f"refusing to overwrite field chunk: {path}")
            saved = capture._load_field_chunk(path, expected_endpoint=SMALL_NAME)
            capture.require(saved["positions"].tolist() == list(range(start, stop)), f"{SMALL_NAME}: partial saved positions differ")
            capture.require(saved["source_row_indices"].tolist() == local_rows and saved["labels"].tolist() == local_labels, f"{SMALL_NAME}: partial saved row/label mapping differs")
            capture.require(set(saved["fields"]) == set(node_names), f"{SMALL_NAME}: partial saved node set differs")
            capture.require(all(torch.equal(saved["fields"][node], collector.records[node]) for node in node_names), f"{SMALL_NAME}: partial saved field values differ from the explicit A1000 source")
            record = {"path": path.relative_to(OUTPUT).as_posix(), "rows": local_rows, "nodes": sorted(node_names), "reused_verified": True}
        else:
            record = capture._save_field_chunk(path, collector, list(range(start, stop)))
        chunk_records.append(record)
    return chunk_records


def _parent_role_map(population_path: Path) -> dict[int, str]:
    population = capture.load_json(population_path)
    return dict(zip(map(int, population["source_row_indices"]), map(str, population["roles"]), strict=True))


def _capture_large_shared_sample(
    endpoint: capture.Endpoint,
    images: torch.Tensor,
    rows: list[int],
    labels: list[int],
    positions: list[int],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shared_input_endpoint = replace(endpoint, group="main", parent_batch=1)
    payload = capture.load_cpu(endpoint.checkpoint)
    state = capture._validate_checkpoint(endpoint, payload)
    consumer = capture._model_for(endpoint, payload, state, torch.device("cuda:0"))
    parameter_before = {name: item.detach().cpu().clone() for name, item in consumer.runtime.named_parameters()}
    buffer_before = {name: item.detach().cpu().clone() for name, item in consumer.runtime.named_buffers()}
    exception_restore = _check_exception_restoration(consumer, shared_input_endpoint, rows, labels)
    chunk_records = []
    start_time = time.perf_counter()
    nodes = [source for source, _display in DISPLAY_NODE_MAP[LARGE_NAME]]
    for index in range(20):
        batch = images[index : index + 1]
        baseline = capture._forward(consumer, batch)
        baseline_raw = baseline.raw_intensity.detach().cpu().clone()
        baseline_ols = baseline.z_ols.coefficients.detach().cpu().clone()
        del baseline
        collector = capture.FieldCollector.create(shared_input_endpoint, [rows[index]], [labels[index]])
        with capture.observe_runtime(consumer, collector):
            observed = capture._forward(consumer, batch)
        equal = torch.equal(observed.raw_intensity.detach().cpu(), baseline_raw) and torch.equal(observed.z_ols.coefficients.detach().cpu(), baseline_ols)
        capture.require(equal, f"{LARGE_NAME}: observation hooks changed output on shared row {rows[index]}")
        camera = capture._camera_identity(observed, collector)
        capture.require(all(node in collector.records for node in nodes), f"{LARGE_NAME}: a required 10-node field was not captured")
        capture.require(len(collector.checks["physical_phase_multiplication"]) == 2, f"{LARGE_NAME}: expected both full-field phase multiplication checks")
        collector.records = {node: collector.records[node] for node in nodes}
        collector.metadata = {node: collector.metadata[node] for node in nodes}
        actual_parameters = dict(consumer.runtime.named_parameters())
        actual_buffers = dict(consumer.runtime.named_buffers())
        capture.require(all(torch.equal(parameter_before[name], actual_parameters[name].detach().cpu()) for name in parameter_before), f"{LARGE_NAME}: model parameter changed during replay")
        capture.require(set(buffer_before) == set(actual_buffers) and all(torch.equal(buffer_before[name], actual_buffers[name].detach().cpu()) for name in buffer_before), f"{LARGE_NAME}: model buffer changed during replay")
        capture.require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), f"{LARGE_NAME}: gradient populated during replay")
        chunk_path = OUTPUT / "fields" / LARGE_NAME / f"field_chunk_{index:03d}.pt"
        if chunk_path.exists():
            capture.require(resume, f"refusing to overwrite field chunk: {chunk_path}")
            saved = capture._load_field_chunk(chunk_path, expected_endpoint=LARGE_NAME)
            capture.require(saved["positions"].tolist() == [index], f"{LARGE_NAME}: existing partial position differs")
            capture.require(saved["source_row_indices"].tolist() == [rows[index]] and saved["labels"].tolist() == [labels[index]], f"{LARGE_NAME}: existing partial row/label mapping differs")
            capture.require(set(saved["fields"]) == set(nodes), f"{LARGE_NAME}: existing partial node set differs")
            capture.require(all(torch.equal(saved["fields"][node], collector.records[node]) for node in nodes), f"{LARGE_NAME}: existing partial field differs from a fresh frozen replay")
            record = {"path": chunk_path.relative_to(OUTPUT).as_posix(), "rows": [rows[index]], "nodes": sorted(nodes), "reused_verified": True}
        else:
            record = capture._save_field_chunk(chunk_path, collector, [index])
        record["checks"] = {
            "observer_output_equal_to_unobserved_forward": {"executed": True, "passed": True, "raw_cmos_bitwise_equal": True, "ols_bitwise_equal": True},
            "camera_intensity_from_incident_field": camera,
            "physical_phase_multiplication": collector.checks["physical_phase_multiplication"],
        }
        chunk_records.append(record)
        progress_path = OUTPUT / "progress" / f"large_row_{index + 1:02d}.json"
        progress_value = {"status": "CAPTURED_ROW", "endpoint": LARGE_NAME, "display_position": index, "source_row_index": rows[index], "label": labels[index]}
        if progress_path.exists():
            capture.require(resume and capture.load_json(progress_path) == progress_value, f"refusing to replace conflicting progress evidence: {progress_path}")
        else:
            capture._write_json(progress_path, progress_value)
        del observed, collector, baseline_raw, baseline_ols, camera
        gc.collect()
        torch.cuda.empty_cache()
    del consumer
    gc.collect()
    torch.cuda.empty_cache()
    return chunk_records, {"exceptional_hook_restoration": exception_restore, "elapsed_seconds_observation_only": time.perf_counter() - start_time}


def _population_overlap(rows: list[int], endpoint: capture.Endpoint) -> dict[str, Any]:
    roles = _parent_role_map(endpoint.population)
    fit = sorted(row for row in rows if roles.get(row) == "fit")
    confirmation = sorted(row for row in rows if roles.get(row) == "confirmation")
    outside = sorted(row for row in rows if row not in roles)
    return {
        "selected_rows_in_parent_fit_population": fit,
        "selected_rows_in_parent_confirmation_population": confirmation,
        "selected_rows_outside_parent_population": outside,
    }


def preflight(*, allow_existing: bool = False) -> dict[str, Any]:
    capture.require(allow_existing or not OUTPUT.exists(), f"refusing to overwrite existing output directory: {OUTPUT}")
    capture.require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "CUDA device 0 is unavailable")
    groups, _source_groups, image_rows, all_labels = capture._prepare_sources()
    positions, rows, labels = _selection(groups["main"])
    capture.require(bool(torch.equal(all_labels.index_select(0, torch.tensor(rows)), torch.tensor(labels))), "selected canonical labels differ")
    prior_endpoints = capture.ENDPOINTS
    targets = (_endpoint(SMALL_NAME), _endpoint(LARGE_NAME))
    try:
        capture.ENDPOINTS = targets
        result = capture._preflight(groups, image_rows, all_labels)
    finally:
        capture.ENDPOINTS = prior_endpoints
    capture.require(result["status"] == "READY", "no-update A1000/B5400 parent-readout preflight failed")
    disk = _disk_estimate(result)
    return {
        "status": "READY",
        "endpoints": result["endpoints"],
        "optimizer_updates": result["optimizer_updates"],
        "optimizers_created": result["optimizers_created"],
        "backward_calls": result["backward_calls"],
        "new_ols_samples": result["new_ols_samples"],
        "disk_estimate": disk,
        "selected_positions": positions,
        "selected_source_rows": rows,
        "selected_labels": labels,
        "role": "canonical TRAIN rows selected from the A1000 confirmation population; visualization only",
    }


def _verify_partial_attempt(checked: dict[str, Any]) -> None:
    capture.require(OUTPUT.is_dir(), "partial output path is not a directory")
    for relative in ("fields/small_A1000", "fields/large_B5400_fg"):
        capture.require((OUTPUT / relative).is_dir(), f"partial output lacks expected directory {relative}")
    for relative in ("field_capture_manifest.json", "sample_selection.json", "capture_report.json", "visualization_build_report.json"):
        capture.require(not (OUTPUT / relative).exists(), f"partial attempt already contains completion output {relative}")
    previous = capture.load_json(OUTPUT / "preflight.json")
    capture.require(previous.get("status") == "READY", "partial attempt does not contain a READY preflight")
    for key in ("selected_positions", "selected_source_rows", "selected_labels", "role"):
        capture.require(previous.get(key) == checked[key], f"partial attempt input identity differs at {key}")
    capture.require(previous.get("endpoints") == checked["endpoints"], "partial attempt parent replay checks differ from the current frozen preflight")
    for relative in ("preflight.json", "disk_estimate.json"):
        capture.require((OUTPUT / relative).is_file(), f"partial attempt is missing {relative}")


def run(*, resume_partial: bool = False) -> dict[str, Any]:
    if resume_partial:
        checked = preflight(allow_existing=True)
        _verify_partial_attempt(checked)
    else:
        capture.require(not OUTPUT.exists(), f"refusing to overwrite existing output directory: {OUTPUT}")
        checked = preflight()
        OUTPUT.mkdir(parents=True, exist_ok=False)
        (OUTPUT / "fields").mkdir()
        (OUTPUT / "fields" / SMALL_NAME).mkdir()
        (OUTPUT / "fields" / LARGE_NAME).mkdir()
        capture._write_json(OUTPUT / "preflight.json", checked)
        capture._write_json(OUTPUT / "disk_estimate.json", checked["disk_estimate"])
    groups, _source_groups, image_rows, all_labels = capture._prepare_sources()
    positions, rows, labels = checked["selected_positions"], checked["selected_source_rows"], checked["selected_labels"]
    images, selected_rows, selected_labels, selected_positions = capture._selected_inputs(groups["main"], image_rows, all_labels)
    capture.require(selected_rows.tolist() == rows and selected_labels.tolist() == labels and selected_positions == positions, "reloaded shared sample selection changed")

    small = _endpoint(SMALL_NAME)
    large = _endpoint(LARGE_NAME)
    original_capture_output = capture.OUTPUT
    capture.OUTPUT = OUTPUT
    try:
        source_chunks, small_fields = _verify_source_small_fields(groups["main"], positions, rows, labels)
        small_chunks = _write_small_field_chunks(small, positions, rows, labels, small_fields, resume=resume_partial)
        progress_dir = OUTPUT / "progress"
        progress_dir.mkdir(exist_ok=True)
        start_record = {"status": "CAPTURING", "small_rows_verified": 20, "large_total_rows": 20}
        start_path = progress_dir / "capture_started.json"
        if start_path.exists():
            capture.require(resume_partial and capture.load_json(start_path) == start_record, "refusing to replace capture start evidence")
        else:
            capture._write_json(start_path, start_record)
        large_chunks, large_checks = _capture_large_shared_sample(large, images, rows, labels, positions, resume=resume_partial)
    finally:
        capture.OUTPUT = original_capture_output

    overlap = _population_overlap(rows, large)
    capture_manifest = {
        "schema": "lightgen_best_endpoints_10node_field_capture.v1",
        "endpoints": [
            {
                "endpoint": SMALL_NAME,
                "checkpoint": small.checkpoint.as_posix(),
                "checkpoint_step": 1000,
                "field_source": "existing verified native A1000 replay chunks",
                "source_chunk_paths": [item["path"] for item in source_chunks],
                "display_source_population": small.population.as_posix(),
                "rows": rows,
                "labels": labels,
                "data_roles": ["A1000 confirmation"] * 20,
                "chunks": small_chunks,
            },
            {
                "endpoint": LARGE_NAME,
                "checkpoint": large.checkpoint.as_posix(),
                "checkpoint_step": 5400,
                "parent_task": "M0_GENERATOR_FOREGROUND_OBJECTIVE_TRAINING_COMPARISON_V1",
                "parent_readout_preflight_endpoint": "B5400",
                "field_source": "frozen CUDA replay of the identical A1000 selected input tensors",
                "display_source_population": small.population.as_posix(),
                "parent_population": large.population.as_posix(),
                "rows": rows,
                "labels": labels,
                "source_selection_role": ["A1000 confirmation"] * 20,
                "parent_population_overlap": overlap,
                "parent_population_role_note": "all selected rows are outside the B5400 parent fit and confirmation rows; all remain canonical TRAIN, so this is not independent generalization",
                "checks": large_checks,
                "chunks": large_chunks,
            },
        ],
        "nodes": [
            {"index": index + 1, "display_name": display, "small_source_node": small_node, "large_source_node": large_node}
            for index, (display, small_node, large_node) in enumerate([
                ("SLM 输出", "slm_output_field", "slm_output_field"),
                ("Encoder 第一层传播后裁窗", "encoder_layer_1_after_crop", "encoder_layer_1_after_crop"),
                ("Encoder 第二层传播后裁窗", "encoder_layer_2_after_crop", "encoder_layer_2_after_crop"),
                ("100维复 OLS 系数（行优先端口顺序）", "ols_coefficients_10x10", "ols_coefficients_10x10"),
                ("OLS 再发射场 / Generator 输入", "ols_reemission_field", "ols_reemission_field"),
                ("Generator z=0.75 mm 面", "generator_z0p75_free_reference", "generator_z0p75_after_phase"),
                ("Generator 原始第一层传播后裁窗", "generator_layer_1_after_crop", "generator_layer_1_after_crop"),
                ("Generator z=2.25 mm 面", "generator_z2p25_free_reference", "generator_z2p25_after_phase"),
                ("Generator 原始第二层传播后裁窗", "generator_layer_2_after_crop", "generator_layer_2_after_crop"),
                ("CMOS 入射复场", "cmos_incident_field", "cmos_incident_field"),
            ])
        ],
        "physical_constants": {"wavelength_m": capture.WAVELENGTH_M, "pixel_pitch_m_reference": capture.PIXEL_PITCH_M, "propagation_grid": [capture.GRID, capture.GRID]},
        "source_role": "canonical TRAIN display examples; no claim of independent test performance",
        "training_updates": 0,
        "optimizer_instances": 0,
        "backward_calls": 0,
        "new_ols_samples": 0,
    }
    capture._write_json(OUTPUT / "field_capture_manifest.json", capture_manifest)
    selection = {
        "source_population": groups["main"]["population"],
        "source_population_role": "A1000 confirmation",
        "selected_positions": positions,
        "selected_source_rows": rows,
        "selected_labels": labels,
        "rows_per_class": [2] * 10,
        "shared_input_pair": [SMALL_NAME, LARGE_NAME],
        "large_parent_population_overlap": overlap,
        "selection_rule": "first two confirmation rows per class in saved A1000 population order",
    }
    capture._write_json(OUTPUT / "sample_selection.json", selection)
    capture._write_json(OUTPUT / "progress" / "capture_complete.json", {"status": "CAPTURED", "small_rows": 20, "large_rows": 20})
    result = {"status": "CAPTURED", "output_dir": OUTPUT.as_posix(), "rows": 20, "nodes_per_endpoint": 10, "optimizer_updates": 0, "new_ols_samples": 0}
    capture._write_json(OUTPUT / "capture_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-partial", action="store_true", help="continue this exact attempt after rechecking its parent replay and saved rows")
    args = parser.parse_args()
    capture.require(not (args.preflight_only and args.resume_partial), "--preflight-only and --resume-partial cannot be combined")
    result = preflight() if args.preflight_only else run(resume_partial=args.resume_partial)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
