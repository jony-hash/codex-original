from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.research.lightgen import diagnose_lightgen_frozen_slm_ols_reachability_v1 as reachability


class _TinyOpticalProducer(torch.nn.Module):
    """Small differentiable producer used to exercise the saved-output decision consumer."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64), requires_grad=False)
        self.runtime = SimpleNamespace(
            camera_plane=SimpleNamespace(pixel_pitch_m=(2.5e-6, 2.5e-6))
        )

    def _camera(self, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_value = coefficients.abs().square().mean(dim=1)
        raw = raw_value[:, None, None, None].expand(-1, 1, 200, 200)
        physical = raw * (2.5e-6**2)
        return raw, physical

    def forward_from_image(self, intensity: torch.Tensor) -> dict[str, torch.Tensor]:
        amplitude = intensity.clamp_min(0.0).sqrt()
        values = amplitude.flatten(1)[:, :100]
        coefficients = torch.complex(values, torch.zeros_like(values))
        raw, physical = self._camera(coefficients)
        return {
            "ols_coefficients": coefficients,
            "raw_intensity_w_per_m2": raw,
            "physical_w_per_pixel": physical,
            "plane_powers_w": torch.ones((values.shape[0], 5), dtype=values.dtype),
        }

    def forward_from_ols(self, coefficients: torch.Tensor) -> dict[str, torch.Tensor]:
        raw, physical = self._camera(coefficients)
        return {"raw_intensity_w_per_m2": raw, "physical_w_per_pixel": physical}


class _SpatialOpticalProducer(_TinyOpticalProducer):
    def forward_from_ols(self, coefficients: torch.Tensor) -> dict[str, torch.Tensor]:
        pattern = torch.linspace(0.0, 1.0, 200 * 200, dtype=torch.float32).reshape(1, 1, 200, 200)
        physical = pattern.expand(coefficients.shape[0], -1, -1, -1)
        return {"raw_intensity_w_per_m2": physical, "physical_w_per_pixel": physical}


def test_fixed_solver_controls_public_phase_and_rejects_impossible_and_overrange_targets() -> None:
    controls = reachability._analytic_solver_controls()
    assert controls["known_command_global_phase_relative_error"] <= 1e-6
    assert controls["impossible_two_port_relative_error"] > 1e-4
    assert controls["amplitude_overflow_target_relative_error"] > 1e-4
    assert controls["commands_bounded_0_1"] is True


def test_bounded_solver_resumes_saved_state_without_exceeding_its_original_budget() -> None:
    matrix = torch.eye(2, dtype=torch.complex128)
    target = torch.tensor([0.23, 0.78], dtype=torch.complex128) * torch.exp(
        torch.tensor(0.73j, dtype=torch.complex128)
    )
    saved_states: list[dict[str, torch.Tensor | int]] = []

    complete = reachability._bounded_phase_solve(matrix, target)
    interrupted = reachability._bounded_phase_solve(
        matrix, target, checkpoint=saved_states.append, checkpoint_interval=1,
    )

    assert saved_states
    resumed = reachability._bounded_phase_solve(matrix, target, initial_state=saved_states[0])
    assert resumed["iterations"] <= 5000
    assert torch.equal(resumed["amplitude"], complete["amplitude"])
    assert resumed["relative_error"] == complete["relative_error"]
    assert interrupted["iterations"] == complete["iterations"]

    invalid_state = dict(saved_states[0])
    invalid_state["amplitudes"] = saved_states[0]["amplitudes"].clone()
    invalid_state["amplitudes"][0, 0] = 1.2
    with pytest.raises(RuntimeError, match="out-of-range"):
        reachability._bounded_phase_solve(matrix, target, initial_state=invalid_state)


def test_config_keeps_the_authorized_population_solver_and_output_fixed() -> None:
    config = reachability._load_yaml(reachability.DEFAULT_CONFIG)
    reachability._validate_config(config, reachability.DEFAULT_CONFIG)
    assert config["source"]["samples_per_method"] == 100
    assert config["source"]["samples_per_class_per_method"] == 10
    assert config["solver"]["phase_starts"] == 8
    assert config["solver"]["max_steps_per_start"] == 5000
    assert config["output"]["experiment_root"].endswith("runs/attempt_01")


def test_new_model_preserves_complex_gaussian_transport_at_float64_precision() -> None:
    source = reachability.base.SmallMatchedAdapter(2048, reachability.GAUSSIAN_OLS_MODEL)
    expected_transport = source.runtime.ols.operator.transport.detach().clone()
    checkpoint = {"model": source.state_dict()}

    model = reachability._new_model(checkpoint, 2048, torch.device("cpu"))

    actual_transport = model.runtime.ols.operator.transport
    assert actual_transport.dtype == torch.complex128
    assert torch.equal(actual_transport, expected_transport)
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())


def test_reference_replay_pools_each_full_native_image(tmp_path: Path) -> None:
    target = torch.full((100,), 0.5 + 0.0j, dtype=torch.complex128)
    gain = 1.0
    native_pattern = torch.linspace(0.0, 1.0, 200 * 200).reshape(1, 1, 200, 200)
    view_pattern = torch.nn.functional.adaptive_avg_pool2d(native_pattern, (28, 28))
    entries: list[dict[str, int | str]] = []
    cohorts: dict[str, dict[str, torch.Tensor]] = {}
    replay_indices: dict[str, dict[int, int]] = {}
    replay: dict[str, dict[str, torch.Tensor]] = {}
    for method in reachability.METHODS:
        cohorts[method] = {"ols_coefficients": target.repeat(100, 1)}
        replay_indices[method] = {row: row for row in range(100)}
        for row in range(100):
            entries.append({"method": method, "requested_label": row % 10, "cohort_row": row})
        replay[method] = {
            "native200": native_pattern.repeat(100, 1, 1, 1),
            "view28": view_pattern.repeat(100, 1, 1, 1),
        }
    inputs = {
        "entries": entries, "cohorts": cohorts,
        "replay_indices": replay_indices, "replay": replay,
    }

    result = reachability._reference_replay_all(
        _SpatialOpticalProducer(), inputs, gain, tmp_path,
    )

    assert result["rows"] == 200
    assert result["all_passed"] is True
    saved = torch.load(tmp_path / "reference_replay.pt", map_location="cpu", weights_only=False)
    assert saved["methods"]["G1"]["native200"].shape == (100, 1, 200, 200)
    assert saved["methods"]["G1"]["view28"].shape == (100, 1, 28, 28)


def test_run_item_saves_amplitude_and_intensity_commands_as_slm_images(tmp_path: Path) -> None:
    target = torch.full((100,), 0.5 + 0.0j, dtype=torch.complex128)
    item = {
        "method": "G1", "requested_label": 0, "cohort_row": 0,
        "proposal_id": 17, "component_id": 3,
    }
    matrix = torch.zeros((100, 40_000), dtype=torch.complex128)
    matrix[:, :100] = torch.eye(100, dtype=torch.complex128)
    inputs = {
        "entries": [item],
        "cohorts": {"G1": {"ols_coefficients": target.reshape(1, 100)}},
        "replay_indices": {"G1": {0: 0}},
        "replay": {"G1": {
            "native200": torch.zeros((1, 1, 200, 200)),
            "view28": torch.zeros((1, 1, 28, 28)),
        }},
        "checkpoint": {"camera_gain": 1.0},
    }
    config = {
        "solver": {
            "phase_starts": 8, "max_steps_per_start": 5000,
            "coefficient_stop_relative_error": 1e-6,
            "witness_coefficient_relative_error": 1e-4,
            "witness_cmos_intensity_relative_error": 1e-4,
            "checkpoint_interval_steps": 500,
        },
        "preflight": {"cross_grid_relative_error_max": 1e-4},
    }
    models = {2048: _TinyOpticalProducer(), 4096: _TinyOpticalProducer()}

    reachability._run_item(item, inputs, matrix, models, tmp_path, config, torch.device("cpu"))
    saved = torch.load(tmp_path / "samples" / "G1_00000.pt", map_location="cpu", weights_only=False)

    assert saved["amplitude_command"].shape == (200, 200)
    assert saved["intensity_command"].shape == (200, 200)
    assert torch.equal(saved["intensity_command"], saved["amplitude_command"].square())


def test_final_consumer_and_windows_reconsumer_recompute_the_fixed_200_rows(tmp_path: Path) -> None:
    device = torch.device("cpu")
    matrix = torch.zeros((100, 40_000), dtype=torch.complex128)
    matrix[:, :100] = torch.eye(100, dtype=torch.complex128)
    amplitude = torch.full((200, 200), 0.5, dtype=torch.float32)
    target = torch.full((100,), 0.5 + 0.0j, dtype=torch.complex128)
    camera_area = 2.5e-6**2
    gain = 1.0e9
    expected_intensity = (target.abs().square().mean() * camera_area * gain).clamp(0.0, 1.0)
    entries: list[dict[str, int | str]] = []
    cohorts: dict[str, dict[str, torch.Tensor]] = {}
    replay_indices: dict[str, dict[int, int]] = {}
    replay: dict[str, dict[str, torch.Tensor]] = {}
    for method in reachability.METHODS:
        cohorts[method] = {
            "ols_coefficients": target.repeat(100, 1),
        }
        replay_indices[method] = {}
        native = torch.full((100, 1, 200, 200), float(expected_intensity), dtype=torch.float32)
        view = torch.full((100, 1, 28, 28), float(expected_intensity), dtype=torch.float32)
        rows = []
        for index in range(100):
            label = index % 10
            item = {
                "method": method, "requested_label": label, "cohort_row": index,
                "proposal_id": index + (0 if method == "G1" else 1000),
                "component_id": label,
            }
            entries.append(item)
            rows.append(item)
            replay_indices[method][index] = index
            sample = {
                "task_id": reachability.TASK_ID,
                "sample_key": reachability._item_key(item),
                    "method": method,
                    "label": label,
                    "stream": 0,
                "cohort_row": index,
                "proposal_id": item["proposal_id"],
                "component_id": label,
                    "target_coefficients": target.to(torch.complex64),
                "amplitude_command": amplitude.reshape(-1),
                "intensity_command": amplitude.square().reshape(-1),
                "numeric_reproduction_witness": True,
                "grids": {
                    str(grid): {
                        "actual_raw_intensity_w_per_m2": torch.full((200, 200), 0.25),
                        "direct_raw_intensity_w_per_m2": torch.full((200, 200), 0.25),
                        "actual_coefficients": target.to(torch.complex64),
                        "direct_coefficients": target.to(torch.complex64),
                    }
                    for grid in (2048, 4096)
                },
            }
            sample_path = tmp_path / "samples" / f"{reachability._item_key(item)}.pt"
            sample_path.parent.mkdir(exist_ok=True)
            torch.save(sample, sample_path)
        replay[method] = {"native200": native, "view28": view}

    inputs = {
        "entries": entries,
        "cohorts": cohorts,
        "replay_indices": replay_indices,
        "replay": replay,
        "checkpoint": {"camera_gain": gain},
    }
    config = {"preflight": {"reference_replay_rtol": 2e-6, "reference_replay_atol": 5e-8}}
    models = {2048: _TinyOpticalProducer(), 4096: _TinyOpticalProducer()}
    identity = {"run_id": "fixed-consumer-test"}

    result = reachability._final_decision(tmp_path, inputs, config, matrix, models, device, identity)
    commands = torch.load(
        tmp_path / "amplitude_and_intensity_commands.pt", map_location="cpu", weights_only=False
    )

    assert result["passed"] == 200
    assert result["population"] == {"G1": 100, "G4": 100, "total": 200, "class_count_per_method": 10}
    assert result["by_method"]["G1"]["passed"] == 100
    assert result["by_method"]["G4"]["passed"] == 100
    assert result["matrix_and_grid_replay_recomputed_from_saved_amplitude_commands"] is True
    assert commands["amplitude_command"].shape == (200, 1, 200, 200)
    assert commands["intensity_command"].shape == (200, 1, 200, 200)
    assert len(commands["sample_identities"]) == 200

    reference = reachability._reference_replay_all(
        _TinyOpticalProducer(), inputs, gain, tmp_path,
        result_name="reference_replay_final.json", run_id=identity["run_id"],
    )
    final = result
    final["original_reference_replay_final"] = {
        "rows": reference["rows"], "all_passed": reference["all_passed"],
        "max_relative_error": reference["max_relative_error"],
        "max_view28_relative_error": reference["max_view28_relative_error"],
    }
    reachability._atomic_json(tmp_path / "final_decision.json", final)
    preflight = {
        "task_id": reachability.TASK_ID,
        "run_id": identity["run_id"],
        "status": "PREFLIGHT_PASSED",
        "matrix_prediction_relative_error": 0.0,
        "direction_derivative_relative_error": 0.0,
        "analytic_controls": {
            "known_command_global_phase_relative_error": 0.0,
            "impossible_two_port_relative_error": 1.0,
            "amplitude_overflow_target_relative_error": 1.0,
        },
        "reference_replay": {"rows": 200, "all_passed": True},
    }
    reachability._atomic_json(tmp_path / "preflight.json", preflight)
    reachability._atomic_torch(tmp_path / "response_matrix.pt", {
        "task_id": reachability.TASK_ID, "matrix": matrix,
    })
    reachability._write_report(tmp_path, final, preflight)
    (tmp_path / "independent_review.md").write_text("Test review evidence.\n", encoding="utf-8")
    closed_identity = {"task_id": reachability.TASK_ID, "run_id": identity["run_id"], "status": "CLOSED"}

    windows = reachability._verify_evidence_bundle(tmp_path, closed_identity, inputs)

    assert windows["status"] == "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED"
    assert windows["population"]["G1"] == {"denominator": 100, "passed": 100}
    assert windows["population"]["G4"] == {"denominator": 100, "passed": 100}
    assert windows["recomputed_witness_metrics"]["max_phase_aligned_coefficient_relative_error"] <= 1e-4
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.count("Windows 端独立重消费再次重算 200 条指令") == 1
    assert report.count("独立复审证据见 independent_review.md。") == 1

    commands["amplitude_command"][0, 0, 0, 0] = 1.2
    reachability._atomic_torch(tmp_path / "amplitude_and_intensity_commands.pt", commands)
    with pytest.raises(RuntimeError, match="out of range"):
        reachability._verify_evidence_bundle(tmp_path, closed_identity, inputs)
