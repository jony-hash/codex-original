import math
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.research.lightgen import diagnose_chen_lightgen_400_sampling_closure_v1 as sampling_closure

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    ChenLightGen400Baseline,
    LightGen400Config,
    integrate_camera_pixel_intensity,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.propagation import (
    angular_spectrum_propagate,
    clear_production_transfer_cache,
    reference_angular_spectrum_propagate,
)


def test_camera_integrates_subpixel_intensity_before_area_reduction() -> None:
    phase = torch.tensor([-0.6, -0.2, 0.2, 0.6], dtype=torch.float64).reshape(1, 1, 2, 2)
    field = torch.polar(torch.ones_like(phase), phase)
    integrated = integrate_camera_pixel_intensity(field.abs().square(), 2)
    coherent_average_then_square = field.mean(dim=(-1, -2), keepdim=True).abs().square()

    assert integrated.shape == (1, 1, 1, 1)
    assert integrated.item() == pytest.approx(1.0)
    assert integrated.item() > coherent_average_then_square.item()
    assert integrate_camera_pixel_intensity(torch.full((1, 1, 4, 4), 0.5), 2).mean().item() == pytest.approx(0.5)


def test_sampling_forward_keeps_physical_task_and_display_units_separate(monkeypatch) -> None:
    physical = torch.full((1, 1, 400, 400), 2.0e-12, dtype=torch.float32)
    target = torch.full_like(physical, 0.25)
    coefficients = torch.zeros((1, 100), dtype=torch.complex64)
    coefficients[0, 0] = 1.0e-6 + 2.0e-7j
    result = {
        "physical_w_per_pixel": physical,
        "ols_coefficients": coefficients,
        "input_power_w": torch.tensor([3.0e-5]),
        "plane_powers_w": torch.zeros((1, 7)),
        "ols_input_power_w": torch.tensor([2.4e-5]),
        "ols_output_power_w": torch.tensor([1.0e-6]),
        "camera_roi_power_w": physical.sum(dim=(1, 2, 3)),
        "camera_full_field_power_w": physical.sum(dim=(1, 2, 3)),
        "camera_outside_roi_power_w": torch.zeros((1,)),
    }

    class FixedModel:
        config = SimpleNamespace(propagation_grid_size=4096)

        def eval(self):
            return self

        def forward_from_image(self, _target, *, field_callback):
            assert field_callback is not None
            return result

    monkeypatch.setattr(sampling_closure.baseline, "_image_batch", lambda *_args: (target, None))
    monkeypatch.setattr(sampling_closure.baseline, "_reconstruction_batch_metrics",
                        lambda *_args: (None, torch.zeros((1,)), torch.full((1,), 0.5)))
    ctx = {"device": torch.device("cpu"), "pixels": torch.empty((0,)), "labels": torch.tensor([0])}

    record, _, _ = sampling_closure._one_forward(ctx, "start", 1, 0, model_override=FixedModel(),
                                                 capture_fields=False)

    assert torch.equal(record["physical_w_per_pixel"], physical)
    torch.testing.assert_close(record["task_intensity_untruncated"], physical * sampling_closure.GAIN)
    assert torch.equal(record["display_intensity_clipped"],
                       record["task_intensity_untruncated"].clamp(0.0, 1.0))
    assert record["task_intensity_untruncated"].mean().item() < 0.1


def test_fourfold_sampling_preserves_400_physical_phase_pixels() -> None:
    model = ChenLightGen400Baseline(LightGen400Config(2048, False, 4))
    assert model.config.active_shape == (1600, 1600)
    assert model.config.pixel_pitch_m == pytest.approx(0.75e-6)
    assert model.encoder_phase.numel() + model.decoder_phase.numel() == 960_000

    field = torch.ones((1, 1, 2048, 2048), dtype=torch.complex64)
    phase = torch.zeros((1, 1, 400, 400))
    phase[0, 0, 7, 11] = 0.75
    encoded = model._phase(field, phase)[0, 0, 224 + 7 * 4:224 + 8 * 4, 224 + 11 * 4:224 + 12 * 4]
    expected = torch.polar(torch.full((4, 4), math.sqrt(0.889)), torch.full((4, 4), 0.75))
    torch.testing.assert_close(encoded, expected, rtol=0.0, atol=1.0e-7)


def test_ols_decoder_path_returns_finite_cmos_and_power_diagnostics() -> None:
    model = ChenLightGen400Baseline(LightGen400Config(512, False, 1)).eval()
    coefficients = torch.zeros((1, 100), dtype=torch.complex64)
    coefficients[0, 44] = torch.tensor(1.0e-5 + 2.0e-6j)
    with torch.no_grad():
        result = model.forward_from_ols(coefficients, return_diagnostics=True)

    assert result["physical_w_per_pixel"].shape == (1, 1, 400, 400)
    assert result["plane_powers_w"].shape == (1, 5)
    assert torch.isfinite(result["physical_w_per_pixel"]).all()
    assert result["ols_reemitted_power_w"].item() > 0.0
    assert result["camera_roi_power_w"].item() >= 0.0


def test_diagnostic_callback_exposes_full_generator_entry_plane_for_common_roi() -> None:
    model = ChenLightGen400Baseline(LightGen400Config(512, False, 1)).eval()
    intensity = torch.full((1, 1, 400, 400), 0.5, dtype=torch.float32)
    fields = {}
    with torch.no_grad():
        result = model.forward_from_image(intensity, field_callback=lambda name, field: fields.__setitem__(name, field))

    assert fields["generator_entry"].tensor.shape[-2:] == (512, 512)
    assert fields["ols_input"].tensor.shape[-2:] == (512, 512)
    assert fields["camera_full_field"].tensor.shape[-2:] == (512, 512)
    assert result["physical_w_per_pixel"].shape == (1, 1, 400, 400)
    assert torch.isfinite(result["physical_w_per_pixel"]).all()


def test_chunked_angular_spectrum_matches_independent_value_and_phase_derivative() -> None:
    torch.manual_seed(20261017)
    side = 512
    plane = PlaneSpec("sampling_chunk_test", (side, side), (3.0e-6, 3.0e-6), 532.0e-9, 0.0, "sampling_test")
    base = torch.randn((1, 1, side, side), dtype=torch.float64)
    base = torch.complex(base, torch.randn_like(base))
    theta_candidate = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    theta_reference = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    distance = 3.0e-3

    clear_production_transfer_cache()
    candidate = angular_spectrum_propagate(
        ComplexField(base * torch.exp(1j * theta_candidate), plane,
                     OperationProvenance("candidate", None, None)), distance, "candidate_out"
    ).tensor
    reference = reference_angular_spectrum_propagate(
        ComplexField(base * torch.exp(1j * theta_reference), plane,
                     OperationProvenance("reference", None, None)), distance, "reference_out"
    ).tensor
    candidate_derivative = torch.autograd.grad(candidate.real[0, 0, 13, 17], theta_candidate)[0]
    reference_derivative = torch.autograd.grad(reference.real[0, 0, 13, 17], theta_reference)[0]
    clear_production_transfer_cache()

    torch.testing.assert_close(candidate, reference, rtol=1.0e-9, atol=2.0e-11)
    torch.testing.assert_close(candidate_derivative, reference_derivative, rtol=1.0e-9, atol=2.0e-11)


def test_signal_recovery_launches_worker_and_resume_as_importable_modules(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "recovery"
    args = SimpleNamespace(**{
        name: tmp_path / name
        for name in ("baseline_checkpoint", "baseline_report", "joint_report", "source_manifest",
                     "fixed_population", "data", "config", "start_state", "endpoint_checkpoint")
    })
    launched = []

    class CompletedWorker:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, received_signal):
            torch.save({"signal_received": received_signal, "completed_production_forward": True},
                       output / "recovery_sample.pt")
            self.returncode = 0

        def communicate(self):
            return "", ""

    def fake_popen(command, *, cwd, **kwargs):
        launched.append((command, cwd))
        assert command[1:3] == ["-m", sampling_closure.MODULE_NAME]
        recovery_dir = Path(command[command.index("--recovery-dir") + 1])
        (recovery_dir / "signal_started.json").write_text(json.dumps({
            "status": "PRODUCTION_FORWARD_ACTIVE", "first_completed_plane": "encoder_1"
        }), encoding="utf-8")
        return CompletedWorker()

    def fake_run(command, *, cwd, **kwargs):
        launched.append((command, cwd))
        assert command[1:3] == ["-m", sampling_closure.MODULE_NAME]
        recovery_dir = Path(command[command.index("--recovery-dir") + 1])
        (recovery_dir / "resume_validation.json").write_text(json.dumps({
            "status": "RESUMED_WITHOUT_RECOMPUTE", "new_optical_forwards": 0, "row_consumed_once": True
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sampling_closure.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sampling_closure.subprocess, "run", fake_run)
    result = sampling_closure._measurement_recovery_probe(args, output)

    assert len(launched) == 2
    assert result["saved_block_readback"] is True
    assert result["new_optical_forwards_during_resume"] == 0


def test_decoder_probe_cases_publish_to_their_own_recovery_paths(tmp_path: Path, monkeypatch) -> None:
    args = SimpleNamespace(start_state=tmp_path / "start.pt", endpoint_checkpoint=tmp_path / "end.pt")
    ctx = {
        "args": args,
        "rows": torch.tensor([0, 1]),
        "selected_labels": torch.tensor([0, 1]),
        "labels": torch.tensor([0, 1]),
    }
    identity = sampling_closure._measurement_identity(ctx)
    for row_index, row in enumerate(ctx["rows"].tolist()):
        by_state = {}
        for state in sampling_closure.STATES:
            by_factor = {}
            for factor in sampling_closure.FACTORS:
                coefficients = torch.zeros((100,), dtype=torch.complex64)
                coefficients[row_index] = complex(1.0, 0.25)
                by_factor[str(factor)] = {
                    "coefficients": coefficients,
                    "physical_w_per_pixel": torch.zeros((1, 1, 400, 400)),
                    "task_intensity_untruncated": torch.zeros((1, 1, 400, 400)),
                    "display_intensity_clipped": torch.zeros((1, 1, 400, 400)),
                    "target": torch.zeros((1, 1, 400, 400)),
                    "metrics": {"state": state, "row": row, "label": row_index, "factor": factor,
                                "gain_w_inverse": sampling_closure.GAIN, "mse400": 0.0,
                                "camera_roi_power_w": 0.0},
                }
            by_state[state] = by_factor
        torch.save({
            "schema": sampling_closure.MEASUREMENT_BLOCK_SCHEMA,
            "identity": identity,
            "row_index": row_index,
            "row": row,
            "label": row_index,
            "by_factor": by_state,
        }, tmp_path / f"sample_{row_index:02d}.pt")

    def decode(_ctx, state, row, coefficients, factor):
        record = {
            "metrics": {"state": state, "row": row, "factor": factor},
            "physical_w_per_pixel": torch.zeros((1, 1, 400, 400)),
            "task_intensity_untruncated": torch.zeros((1, 1, 400, 400)),
            "display_intensity_clipped": torch.zeros((1, 1, 400, 400)),
            "port_power_w": coefficients.abs().square(),
            "decoder_plane_powers_w": torch.zeros((5,)),
        }
        fields = {"camera": {
            "physical_pixel_center": torch.ones((1,)),
            "physical_pixel_average": torch.ones((1,)),
        }}
        return record, fields

    monkeypatch.setattr(sampling_closure, "_decode_saved_coefficients", decode)
    monkeypatch.setattr(sampling_closure.baseline, "_complex_relative_errors",
                        lambda _candidate, _reference: ({"camera": 0.0}, 0.0))

    result = sampling_closure._run_fixed_coefficient_decoder_probes(ctx, tmp_path)

    assert result == {"status": "DECODER_PROBES_COMPLETE", "completed_cases": 4}
    for state in sampling_closure.STATES:
        for row_index in (0, 1):
            path = tmp_path / f"decoder_probe_{state}_{row_index:02d}.pt"
            block = torch.load(path, map_location="cpu", weights_only=False)
            assert block["state"] == state
            assert block["row_index"] == row_index
            assert block["row"] == row_index


def test_cached_unit_migration_recovers_physical_camera_power_and_rejects_mismatch(tmp_path: Path) -> None:
    args = SimpleNamespace(start_state=tmp_path / "start.pt", endpoint_checkpoint=tmp_path / "end.pt")
    rows = torch.arange(10)
    labels = torch.arange(10)
    ctx = {"args": args, "rows": rows, "selected_labels": labels}
    source_identity = {
        "task_id": sampling_closure.TASK_ID,
        "authorization_id": sampling_closure.AUTHORITY,
        "start_checkpoint": "/remote/start_model_state.pt",
        "endpoint_checkpoint": "/remote/checkpoint_step_000500.pt",
        "joint_updates": 500,
        "fixed_rows": rows.tolist(),
        "fixed_labels": labels.tolist(),
        "factors": list(sampling_closure.FACTORS),
        "grids": {str(k): v for k, v in sampling_closure.GRIDS.items()},
        "gain_w_inverse": sampling_closure.GAIN,
    }
    target = torch.full((1, 1, 400, 400), 0.25)
    by_state = {}
    for state in sampling_closure.STATES:
        by_factor = {}
        for factor in sampling_closure.FACTORS:
            task = torch.full((1, 1, 400, 400), 0.1)
            physical = task / sampling_closure.GAIN
            coefficients = torch.zeros((100,), dtype=torch.complex64)
            camera_power = float(physical.sum())
            by_factor[str(factor)] = {
                "coefficients": coefficients,
                "port_power_w": coefficients.abs().square(),
                "physical_w_per_pixel": task,
                "target": target,
                "metrics": {"state": state, "row": 0, "label": 0, "factor": factor,
                            "gain_w_inverse": sampling_closure.GAIN,
                            "mse400": float((task - target).square().mean()),
                            "camera_roi_power_w": camera_power},
            }
        by_state[state] = by_factor
    source = {"schema": "chen_lightgen_400_sampling_measurement_block.v1",
              "identity": source_identity, "row_index": 0, "row": 0, "label": 0,
              "by_factor": by_state, "sampling_comparisons": {state: {} for state in sampling_closure.STATES}}

    corrected = sampling_closure._migrate_cached_sample_block(source, ctx, 0, source_identity)
    converted = corrected["by_factor"]["start"]["1"]
    assert corrected["schema"] == sampling_closure.MEASUREMENT_BLOCK_SCHEMA
    torch.testing.assert_close(converted["physical_w_per_pixel"], torch.full_like(target, 0.1 / sampling_closure.GAIN))
    assert torch.equal(converted["task_intensity_untruncated"], torch.full_like(target, 0.1))
    assert torch.equal(converted["display_intensity_clipped"], torch.full_like(target, 0.1))

    bad = dict(source)
    bad_states = {state: dict(by_state[state]) for state in sampling_closure.STATES}
    bad_factors = dict(bad_states["start"])
    bad_item = dict(bad_factors["1"])
    bad_item["physical_w_per_pixel"] = bad_item["physical_w_per_pixel"] * 2.0
    bad_factors["1"] = bad_item
    bad_states["start"] = bad_factors
    bad["by_factor"] = bad_states
    with pytest.raises(RuntimeError, match="does not store the measured task intensity"):
        sampling_closure._migrate_cached_sample_block(bad, ctx, 0, source_identity)


def test_production_propagation_preflight_compares_reference_tensor_values() -> None:
    result = sampling_closure._propagation_reference_check()

    assert result["passed"] is True
    assert result["analytic_relative_l2"] <= 1.0e-10
    assert result["independent_reference_relative_l2"] <= 1.0e-10


def test_preflight_correctness_group_is_consumable_and_fails_closed() -> None:
    identity = sampling_closure.ResearchRunIdentity(
        task_id="sampling-closure-test", run_id="run-1", trajectory_id="fixed-probe",
        semantic_authority_id="authorized-test", data_role="TRAIN", config_identity={},
        model_identity={}, optimizer_identity={}, sampler_identity={},
    )
    control = sampling_closure.ResearchRunControl(identity, authorization_id="sampling-test")
    correctness = sampling_closure._ready_correctness_group({"production_reference": {"passed": True}})
    receipt = control.make_ready_receipt(
        correctness=correctness,
        resource={"passed": True},
        implementation={"passed": True, "engine": "frozen-test", "selected_engine": "frozen-test"},
        selected_engine="frozen-test",
    )

    control.mark_preflight_ready(receipt)
    assert receipt.checks["CORRECTNESS_INVARIANTS"]["passed"] is True
    assert receipt.checks["CORRECTNESS_INVARIANTS"]["checks"]["production_reference"]["passed"] is True
    with pytest.raises(RuntimeError, match="preflight checks did not pass"):
        sampling_closure._ready_correctness_group({"production_reference": {"passed": False}})


def test_physical_pixel_field_observer_uses_common_coordinates_and_box_integral() -> None:
    period_m = 8.0e-6
    roi_pixels = 32
    roi_m = roi_pixels * sampling_closure.PIXEL_PITCH_M
    fields_by_factor = {}
    variants = {}
    legacy_views = {}

    for factor in sampling_closure.FACTORS:
        count = 128 * factor
        pitch = sampling_closure.PIXEL_PITCH_M / factor
        axis = (torch.arange(count, dtype=torch.float64) - count / 2 + 0.5) * pitch
        field = torch.exp(2j * math.pi * axis[None, :] / period_m).expand(count, -1).to(torch.complex64)
        physical = sampling_closure._spectral_physical_pixel_observables(field, factor, roi_m=roi_m)
        fields_by_factor[factor] = {"plane": physical}
        variants[factor] = {
            "coefficients": torch.ones((100,), dtype=torch.complex64),
            "port_power_w": torch.ones((100,), dtype=torch.float64),
            "physical_w_per_pixel": torch.ones((1, 1, 400, 400)),
            "metrics": {"mse400": 0.0, "psnr28_db": 0.0, "ssim28": 0.0,
                        "saturation_fraction": 0.0},
        }
        centers = (torch.arange(roi_pixels, dtype=torch.float64) - roi_pixels / 2 + 0.5) * sampling_closure.PIXEL_PITCH_M
        expected_line = torch.exp(2j * math.pi * centers / period_m).to(torch.complex64)
        expected_center = expected_line[None, :].expand(roi_pixels, -1)
        box_factor = (torch.sinc(torch.tensor(
            sampling_closure.PIXEL_PITCH_M / period_m, dtype=torch.float64
        ))).to(torch.complex64)
        torch.testing.assert_close(physical["physical_pixel_center"], expected_center,
                                   rtol=1.0e-4, atol=1.0e-4)
        torch.testing.assert_close(physical["physical_pixel_average"], expected_center * box_factor,
                                   rtol=1.0e-4, atol=1.0e-4)

        roi_fine = roi_pixels * factor
        top = (count - roi_fine) // 2
        cropped = field[top:top + roi_fine, top:top + roi_fine]
        shaped = cropped.reshape(roi_pixels, factor, roi_pixels, factor)
        if factor > 1:
            lo = factor // 2 - 1
            legacy_views[factor] = shaped[:, lo:lo + 2, :, lo:lo + 2].mean(dim=(1, 3))
        else:
            legacy_views[factor] = shaped

    comparison = sampling_closure._compare_sample(variants, fields_by_factor)
    assert max(value["max_relative_l2"] for value in comparison["field_checks"].values()) < 1.0e-4
    legacy_error = torch.linalg.vector_norm(legacy_views[2] - legacy_views[4]) / torch.linalg.vector_norm(legacy_views[4])
    assert legacy_error.item() > 1.0e-2


def test_analytic_observer_regression_records_old_error_and_new_consistency() -> None:
    result = sampling_closure._analytic_physical_pixel_observer_check()

    assert result["passed"] is True
    assert result["max_analytic_error"] <= sampling_closure.WINDOW_TOLERANCE
    assert max(result["same_observable_cross_factor_errors"].values()) <= sampling_closure.WINDOW_TOLERANCE
    legacy = result["legacy_comparator_counterexample"]
    assert legacy["factor_2_vs_4_center_relative_l2"] == pytest.approx(0.1311, abs=5.0e-4)
    assert legacy["factor_1_vs_4_area_relative_l2"] == pytest.approx(0.2568, abs=5.0e-4)


def test_finite_pixel_rs_reference_converges_and_reports_path_residual() -> None:
    result = sampling_closure._rayleigh_sommerfeld_pixel_reference_check(torch.device("cpu"))

    assert set(result["patterns"]) == {"gaussian", "tilted", "checkerboard"}
    assert result["production_comparison_executed"] is False
    assert result["physical_window_matches_production"] is True
    assert result["physical_window_m"] == pytest.approx(4096 * sampling_closure.PIXEL_PITCH_M)
    assert all(item["gauss_order_8_vs_12_relative_l2"] <= sampling_closure.WINDOW_TOLERANCE
               for item in result["patterns"].values())
    assert all(item["rs_center_difference_relative_error"] <= sampling_closure.FD_TOLERANCE
               for item in result["patterns"].values())
    assert result["passed"] == all(item["passed"] for item in result["patterns"].values())


def test_sampling_decision_blocks_candidate_when_independent_rs_fails() -> None:
    comparison = {
        state: {
            "coefficient_checks": {
                "factor_1_vs_4": {"complex_relative_l2_after_global_phase": 0.0},
                "factor_2_vs_4": {"complex_relative_l2_after_global_phase": 0.0},
            }
        }
        for state in sampling_closure.STATES
    }
    gradients = {
        "fixed_probe": {
            "comparisons": {
                group: {
                    "factor_2": {"relative_l2_vs_factor_1": 0.0},
                    "factor_4": {"relative_l2_vs_factor_1": 0.0},
                    "factor_2_vs_factor_4": {"relative_l2": 0.0},
                }
                for group in ("encoder_phase", "decoder_phase")
            }
        }
    }
    checks = {
        "gaussian_by_factor": {},
        "gradient_checks": gradients,
        "finite_difference_checks": {},
        "factor_2_window_expansion": {"required": True, "passed": True},
        "finite_pixel_rayleigh_sommerfeld": {"passed": False, "patterns": {}},
    }
    blocks = [{"sampling_comparisons": comparison}]
    field_rows = [
        {"state": sampling_closure.STATE_START, "row": 0, "plane": "encoder_1",
         "relative_l2": 0.0, "comparison": "factor_1_vs_4_physical_pixel_center"},
        {"state": sampling_closure.STATE_START, "row": 0, "plane": "encoder_1",
         "relative_l2": 0.0, "comparison": "factor_2_vs_4_physical_pixel_center"},
    ]

    result = sampling_closure._sampling_decision(blocks, [], field_rows, checks)

    assert result["factor1_pass"] is False
    assert result["factor2_pass"] is False
    assert result["status"] == "NUMERICAL_NOT_QUALIFIED"
