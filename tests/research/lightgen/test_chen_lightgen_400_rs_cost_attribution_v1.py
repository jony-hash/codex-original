from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_finite_pixel import (
    FinitePixelConfig,
    FinitePixelRSLinear,
)
from scripts.research.lightgen import execute_chen_lightgen_400_rs_cost_attribution_v1 as cost


def _tiny_rs_linear() -> FinitePixelRSLinear:
    config = FinitePixelConfig(
        quadrature_order=2,
        window_cells=8,
        distance_m=3.0e-3,
        checkpoint_propagation=False,
        propagation_backend="rs_linear_cached",
        rs_compute_precision="complex64",
        rs_pair_block=1,
    )
    return FinitePixelRSLinear(order=2, active_shape=(8, 8), config=config)


def test_trace_counts_actual_cached_rs_forward_adjoint_and_reuse() -> None:
    propagator = _tiny_rs_linear()
    model = SimpleNamespace(propagator=propagator)
    source = torch.ones((1, 4, 8, 8), dtype=torch.complex64, requires_grad=True)

    with cost._OperationTrace(model) as trace:
        propagated, power, center = propagator(source, output_shape=(8, 8))
        assert center is None
        (propagated.abs().square().mean() + power.mean()).backward()

        source.grad = None
        propagated, power, center = propagator(source, output_shape=(8, 8))
        assert center is None
        (propagated.abs().square().mean() + power.mean()).backward()

    assert trace.counts["rs_forward_calls"] == 2
    assert trace.counts["rs_adjoint_calls"] == 2
    assert trace.counts["kernel_cache_builds"] == 1
    assert trace.counts["kernel_cache_misses"] == 1
    assert trace.counts["kernel_cache_hits"] == 3
    assert trace.counts["pair_spectrum_materializations"] == 64
    assert trace.counts.get("center_observation_requests", 0) == 0
    assert trace.counts.get("full_window_observation_requests", 0) == 0


def test_call_count_consumer_rejects_missing_checkpoint_recomputation() -> None:
    checkpoint_on = {
        "rs_forward_calls": 70,
        "rs_adjoint_calls": 35,
        "linear_propagation_forward_calls": 70,
        "center_observation_requests": 0,
        "full_window_observation_requests": 0,
        "kernel_cache_hits": 105,
        "kernel_cache_misses": 0,
        "pair_spectrum_materializations": 105 * cost.PAIR_MATERIALIZATIONS_PER_RS_CALL,
    }
    cost._validate_call_counts(checkpoint_on, checkpoint=True)
    checkpoint_on["rs_forward_calls"] = 35
    with pytest.raises(cost.HarnessError, match="forward/adjoint/recomputation"):
        cost._validate_call_counts(checkpoint_on, checkpoint=True)

    no_recompute = checkpoint_on | {
        "rs_forward_calls": cost.BASE_RS_FORWARD_CALLS_PER_BATCH10,
        "linear_propagation_forward_calls": cost.BASE_RS_FORWARD_CALLS_PER_BATCH10,
        "kernel_cache_hits": 70,
        "pair_spectrum_materializations": 70 * cost.PAIR_MATERIALIZATIONS_PER_RS_CALL,
    }
    cost._validate_call_counts(no_recompute, checkpoint=True)

    missing_cache_replay = checkpoint_on | {"rs_forward_calls": 70,
                                            "linear_propagation_forward_calls": 70,
                                            "kernel_cache_hits": 70}
    with pytest.raises(cost.HarnessError, match="warm cache for every RS forward"):
        cost._validate_call_counts(missing_cache_replay, checkpoint=True)

    checkpoint_on["rs_forward_calls"] = 70
    checkpoint_on["center_observation_requests"] = 1
    with pytest.raises(cost.HarnessError, match="diagnostic observations"):
        cost._validate_call_counts(checkpoint_on, checkpoint=True)


def test_profile_summary_requires_one_forward_and_one_adjoint_without_inclusive_total() -> None:
    records = [
        {"direction": "forward", "profile_scope": "one production cached RS operator call; bounded",
         "top_operators": [{"operator": "aten::fft_fft2", "calls": 1,
                            "self_cuda_time_us": 3.0, "self_cpu_time_us": 1.0}]},
        {"direction": "adjoint", "profile_scope": "one production cached RS operator call; bounded",
         "top_operators": [{"operator": "aten::fft_fft2", "calls": 1,
                            "self_cuda_time_us": 3.0, "self_cpu_time_us": 1.0}]},
    ]
    cost._validate_profile_summaries(records)

    duplicated = [records[0], records[0] | {"total_cuda_time_us": 6.0}]
    with pytest.raises(cost.HarnessError, match="one forward and one adjoint"):
        cost._validate_profile_summaries(duplicated)

    inclusive = [records[0] | {"total_cuda_time_us": 9.0}, records[1]]
    with pytest.raises(cost.HarnessError, match="inclusive aggregate"):
        cost._validate_profile_summaries(inclusive)


def test_checkpoint_comparison_consumes_outputs_and_all_six_layer_gradients() -> None:
    reference = {
        "loss": 0.25,
        "gradients": {"encoder_phase": torch.ones((2, 3)),
                      "decoder_phase": torch.ones((4, 3))},
        "w_per_pixel": torch.ones((2, 1, 3, 3)),
    }
    candidate = {key: value.clone() if isinstance(value, torch.Tensor) else value
                 for key, value in reference.items()}
    candidate["gradients"] = {key: value.clone() for key, value in reference["gradients"].items()}
    assert cost._compare_case(candidate, reference)["status"] == "NUMERICALLY_EQUIVALENT"

    candidate["gradients"]["decoder_phase"][3, 0] += 1.0
    result = cost._compare_case(candidate, reference)
    assert result["status"] == "NUMERICAL_NOT_EQUIVALENT"
    assert result["max_gradient_relative_error"] > cost.GRAD_RTOL


def test_saved_qualification_without_full_w_per_pixel_only_qualifies_loss_and_gradients() -> None:
    reference = {"loss": 0.25,
                "gradients": {"encoder_phase": torch.ones((2, 3)),
                              "decoder_phase": torch.ones((4, 3))}}
    candidate = {"loss": 0.25, "gradients": reference["gradients"],
                 "w_per_pixel": torch.ones((10, 1, 400, 400))}
    comparison = cost._compare_case(candidate, reference)
    assert comparison["status"] == cost.LOSS_GRADIENT_ONLY_STATUS
    assert comparison["w_per_pixel_reference_available"] is False
    assert comparison["w_per_pixel_relative_error"] is None


def test_fixed_production_readout_gain_is_applied_to_physical_power() -> None:
    physical = torch.tensor([1.0e-9, 2.0e-9], dtype=torch.float32)
    prediction = cost._fixed_readout_prediction(physical)
    expected = cost.q12.FIXED_GAIN * physical.to(torch.float64)
    assert prediction.dtype == torch.float64
    assert torch.equal(prediction, expected)


def test_gradient_shape_check_matches_the_production_single_channel_parameters() -> None:
    gradients = {"encoder_phase": torch.zeros((2, 1, 400, 400)),
                 "decoder_phase": torch.zeros((4, 1, 400, 400))}
    cost._validate_gradient_shapes(gradients)
    gradients["encoder_phase"] = torch.zeros((2, 400, 400))
    with pytest.raises(cost.HarnessError, match="single-channel phase-layer gradients"):
        cost._validate_gradient_shapes(gradients)


def test_saved_rows_and_measurement_order_reject_misaligned_or_reordered_evidence() -> None:
    fixed_rows = torch.tensor([10, 20, 30], dtype=torch.long)
    cost._validate_saved_rows([10, 20, 30], fixed_rows)
    with pytest.raises(cost.HarnessError, match="differ from the fixed fit population"):
        cost._validate_saved_rows([10, 30, 20], fixed_rows)

    cost._validate_measurement_order(list(cost.FULL_BATCH_ORDER), checkpoint_off_eligible=True)
    with pytest.raises(cost.HarnessError, match="fixed predeclared order"):
        cost._validate_measurement_order(list(reversed(cost.FULL_BATCH_ORDER)),
                                         checkpoint_off_eligible=True)
    cost._validate_measurement_order(["checkpoint_on_1", "checkpoint_on_2"],
                                     checkpoint_off_eligible=False)


def test_fixed_population_validation_uses_the_saved_fit_and_confirmation_contract() -> None:
    labels = torch.arange(10).repeat_interleave(30)
    fit_rows_by_class = torch.stack([torch.arange(label * 30, label * 30 + 10)
                                     for label in range(10)])
    confirm_rows = torch.cat([torch.arange(label * 30 + 10, label * 30 + 30)
                              for label in range(10)])
    population = {"fit_rows": fit_rows_by_class.flatten(),
                  "confirm_rows": confirm_rows,
                  "fit_rows_by_class": fit_rows_by_class}

    rows = cost._validate_fixed_population(population, labels)
    assert rows.tolist() == list(range(0, 300, 30))

    population["confirm_rows"] = confirm_rows.roll(1)
    with pytest.raises(cost.q12.HarnessCheckError, match="fixed population labels"):
        cost._validate_fixed_population(population, labels)


def test_execution_does_not_consume_an_incomplete_preflight(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt_01"
    output_dir.mkdir()
    args = argparse.Namespace()
    with pytest.raises(cost.HarnessError, match="complete saved --preflight-only receipt"):
        cost._load_preflight(output_dir, args)


def test_speed_bound_rejects_a_double_counted_component() -> None:
    assert cost._safe_speed_bound(10.0, 4.0)["optimistic_max_speedup_if_component_were_free"] == pytest.approx(
        1.0 / 0.6)
    with pytest.raises(cost.HarnessError, match="double-counted"):
        cost._safe_speed_bound(10.0, 11.0)


def test_saved_cost_case_consumer_revalidates_production_values_and_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt_01"
    measurement_dir = output_dir / "measurements"
    measurement_dir.mkdir(parents=True)
    rows = torch.arange(10, dtype=torch.long)
    gradients = {"encoder_phase": torch.zeros((2, 1, 400, 400)),
                 "decoder_phase": torch.zeros((4, 1, 400, 400))}
    camera = torch.zeros((10, 1, 400, 400))
    counts = {"rs_forward_calls": 35, "rs_adjoint_calls": 35,
              "linear_propagation_forward_calls": 35,
              "kernel_cache_hits": 70, "kernel_cache_misses": 0,
              "pair_spectrum_materializations": 70 * cost.PAIR_MATERIALIZATIONS_PER_RS_CALL,
              "center_observation_requests": 0, "full_window_observation_requests": 0}
    timing = {"host_wall_seconds_by_stage": {"optical_forward_and_power": 1.0,
                                              "backward": 1.0},
              "cuda_event_elapsed_ms_by_stage": {"optical_forward_and_power": 1000.0,
                                                   "backward": 1000.0}}
    tensor_path = measurement_dir / "checkpoint_on_1.pt"
    torch.save({"schema": "chen_lightgen_400_rs_cost_case.v1",
                "task_id": cost.TASK_ID, "case": "checkpoint_on_1",
                "rows": rows, "w_per_pixel": camera, "gradients": gradients,
                "loss": 0.25, "checkpoint_propagation": True,
                "elapsed_seconds": 5.0, "peak_reserved_vram_bytes": 1024,
                "operator_call_counts": counts, "stage_timing": timing}, tensor_path)
    summary = {"case": "checkpoint_on_1", "checkpoint_propagation": True,
               "rows": rows.tolist(), "loss": 0.25, "elapsed_seconds": 5.0,
               "peak_reserved_vram_bytes": 1024, "operator_call_counts": counts,
               "stage_timing": timing, "tensor_payload": "/attempt_01/measurements/checkpoint_on_1.pt",
               "cache_hit_ratio": 1.0, "cache_cold_build": None,
               "cache_info_before_repetition": {"cache_bytes": 1024},
               "loss_relative_error": 0.0, "max_gradient_relative_error": 0.0,
               "w_per_pixel_relative_error": 0.0, "status": "NUMERICALLY_EQUIVALENT"}
    (output_dir / "checkpoint_on_1_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    reference = {"loss": 0.25, "gradients": gradients, "w_per_pixel": camera}

    loaded, run = cost._load_saved_case(output_dir, "checkpoint_on_1", rows, reference,
                                        checkpoint=True)
    assert loaded["status"] == "NUMERICALLY_EQUIVALENT"
    assert torch.equal(run["w_per_pixel"], camera)

    summary["rows"] = rows.roll(1).tolist()
    (output_dir / "checkpoint_on_1_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(cost.HarnessError, match="summary does not match its production tensor"):
        cost._load_saved_case(output_dir, "checkpoint_on_1", rows, reference, checkpoint=True)


def test_cost_report_uses_the_actual_preflight_cost_field_and_preserves_wall_time() -> None:
    args = SimpleNamespace(output_dir=Path("outputs/attempt_01"))
    preflight = {"receipt": {"status": "PREFLIGHT_PASSED", "elapsed_seconds": 20.0,
        "explicit_inputs": {}, "rows": list(range(10)), "beta": 2.0, "eta": 0.5,
        "cache_cold_build": {"host_seconds": 0.4},
        "checkpoint_on_uninstrumented_production_cost_seconds": 535.0,
        "saved_qualification_cost_seconds": 534.5}}
    intervals = {"by_direction_ms": {"forward": 267_000.0, "adjoint": 267_000.0}}
    cases = {name: {"elapsed_seconds": seconds,
                    "checkpoint_propagation": name.startswith("checkpoint_on"),
                    "operator_cuda_intervals": intervals,
                    "cache_info_before_repetition": {"cache_bytes": 1024}}
             for name, seconds in zip(cost.FULL_BATCH_ORDER,
                                      (535.0, 534.8, 534.5, 535.3), strict=True)}
    report = cost._build_cost_report(
        args=args, preflight=preflight, rows=torch.arange(10),
        resource_probe={"status": "NUMERICALLY_EQUIVALENT"}, cases=cases,
        execution_cache_build={"host_seconds": 0.4}, profile_summaries=[],
        execution_elapsed_seconds=2900.0, parameter_and_buffer_state_unchanged=True,
        finalization_mode="TEST")

    assert report["checkpoint_on_uninstrumented_preflight_seconds"] == 535.0
    assert report["execution_elapsed_seconds"] == 2900.0
    assert report["total_diagnostic_elapsed_seconds"] == 2920.0
    assert report["measurement_order"] == list(cost.FULL_BATCH_ORDER)


def _write_previous_qualification(root: Path) -> argparse.Namespace:
    qualification_dir = root / "outputs" / "qualification" / "runs" / "attempt_01"
    qualification_dir.mkdir(parents=True)
    inputs = {
        "source_checkpoint": root / "outputs" / "source.pt",
        "source_report": root / "outputs" / "source.json",
        "source_start_state": root / "outputs" / "start.pt",
        "source_endpoint_checkpoint": root / "outputs" / "endpoint.pt",
        "fixed_population": root / "outputs" / "population.pt",
        "config": root / "configs" / "config.yaml",
        "data": root / "data" / "training.pt",
        "evaluator": root / "outputs" / "evaluator.pt",
        "qualification_report": root / "outputs" / "qualification.md",
        "structure_coefficient": root / "outputs" / "eta.json",
        "qualification_dir": qualification_dir,
        "output_dir": root / "outputs" / "qualification" / "runs" / "attempt_01",
    }
    explicit = {key: f"/root/autodl-tmp/codex-original/{path.relative_to(root).as_posix()}"
                for key, path in inputs.items() if key != "qualification_dir"}
    explicit["output_dir"] = "/root/autodl-tmp/codex-original/outputs/qualification/runs/attempt_01"
    (qualification_dir / "run_manifest.json").write_text(json.dumps({
        "schema": cost.QUALIFICATION_SCHEMA,
        "task_id": cost.QUALIFICATION_TASK_ID,
        "explicit_inputs": explicit,
    }), encoding="utf-8")
    (qualification_dir / "final_report.json").write_text(json.dumps({
        "status": "VALID_COMPLETED_BOTH_PRECISIONS_QUALIFIED",
        "complex128_qualified": True, "complex64_qualified": True,
        "optimizer_updates": 0,
    }), encoding="utf-8")
    cost_dir = qualification_dir / "cost"
    gradient_dir = qualification_dir / "gradients"
    cost_dir.mkdir()
    gradient_dir.mkdir()
    (cost_dir / "cost_complex64_B_microbatch_2.json").write_text(json.dumps({
        "precision": "complex64", "branch": "B", "microbatch": 2,
        "logical_batch": 10, "checkpoint_propagation": True,
        "adam_calls": 0, "optimizer_calls": 0, "optimizer_updates": 0,
        "rows": list(range(10)),
    }), encoding="utf-8")
    (gradient_dir / "cost_complex64_B_microbatch_2.pt").write_bytes(b"required evidence marker")
    args = SimpleNamespace(**{key: value for key, value in inputs.items()
                              if key not in {"qualification_dir"}})
    args.qualification_dir = qualification_dir
    args.output_dir = root / "outputs" / "newtask" / "runs" / "attempt_01"
    return args


def test_previous_evidence_path_identity_is_checked_and_missing_file_rejected(tmp_path: Path) -> None:
    root = tmp_path / "codex-original"
    args = _write_previous_qualification(root)
    validated = cost._validate_previous_evidence(args)
    assert validated["metric"]["rows"] == list(range(10))

    mismatched_args = argparse.Namespace(**vars(args))
    mismatched_args.source_report = root / "outputs" / "other_source.json"
    with pytest.raises(cost.HarnessError, match="differs from the qualified input path"):
        cost._validate_previous_evidence(mismatched_args)

    (args.qualification_dir / "gradients" / "cost_complex64_B_microbatch_2.pt").unlink()
    with pytest.raises(cost.HarnessError, match="saved qualification evidence is missing"):
        cost._validate_previous_evidence(args)
