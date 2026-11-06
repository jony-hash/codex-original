from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.lightgen import execute_chen_lightgen_400_joint_reconstruction_v1 as joint


class _LinearReadout:
    def eval(self) -> "_LinearReadout":
        return self

    def forward_from_image(self, intensity: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"physical_w_per_pixel": intensity * 0.5}


def test_fit_gain_uses_global_float64_least_squares(monkeypatch) -> None:
    targets = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float32).reshape(4, 1, 1, 1)
    rows = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    monkeypatch.setattr(joint.baseline, "_image_batch",
                        lambda data, selected, device: (data.index_select(0, selected), data.index_select(0, selected)))
    result = joint._fixed_gain(_LinearReadout(), {"fit_rows": rows}, targets, torch.device("cpu"))
    assert result["fit_rows"] == 4
    assert result["estimator"] == "float64_global_least_squares_at_4096_start"
    assert result["gain_w_inverse"] == 2.0
    assert isinstance(result["numerator"], float)
    assert isinstance(result["denominator"], float)


def test_fit_batch_cursor_preserves_saved_class_order() -> None:
    saved = torch.arange(100, dtype=torch.long).reshape(10, 10) + 1_000
    population = {"fit_rows_by_class": saved}
    for cursor in range(10):
        assert torch.equal(joint._fit_indices(population, cursor), saved[:, cursor])
    try:
        joint._fit_indices(population, 10)
    except RuntimeError as error:
        assert "cursor" in str(error)
    else:
        raise AssertionError("out-of-range cursor was accepted")


def test_task_units_keep_physical_power_and_clip_only_display() -> None:
    physical = torch.tensor([[[[0.0, 0.25, 0.75]]]], dtype=torch.float32)
    task, display = joint._task_outputs({"physical_w_per_pixel": physical}, 2.0, torch.device("cpu"))
    assert torch.equal(task, torch.tensor([[[[0.0, 0.5, 1.5]]]]))
    assert torch.equal(display, torch.tensor([[[[0.0, 0.5, 1.0]]]]))


def test_window_gate_consumes_first_two_confirmation_rows_per_class(monkeypatch) -> None:
    population = {"confirm_rows_by_class": torch.arange(200, dtype=torch.long).reshape(10, 20)}
    pixels = torch.zeros(200, 1, 1, 1)
    monkeypatch.setattr(joint.baseline, "_image_batch",
                        lambda data, rows, device: (data.index_select(0, rows), data.index_select(0, rows)))
    monkeypatch.setattr(joint.baseline, "_window_reference",
                        lambda model, sample, grids, roi, **kwargs: (4096, {
                            4096: {"complex_relative_error_vs_largest_window": {"encoder": 0.0, "decoder": 0.0}}
                        }))
    result = joint._window_checks(object(), population, pixels, torch.zeros(200, dtype=torch.long),
                                  {"optics": {"common_complex_comparison_roi_m": 1.5e-3}},
                                  torch.device("cpu"))
    assert result["passed"]
    assert len(result["samples"]) == 20
    assert [item["row"] for item in result["samples"]] == [row for label in range(10)
                                                             for row in range(20 * label, 20 * label + 2)]


def test_run_lets_lifecycle_create_fresh_attempt_directory(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "attempt_02"
    preflight_dir = tmp_path / "preflight"
    population = {"rows": torch.arange(3)}
    class _Reference:
        def cpu(self) -> torch.Tensor:
            return torch.tensor(1.0)

        def to(self, _device: torch.device) -> "_Reference":
            return self

    references = {"camera_gain": _Reference(), "frozen": _Reference()}
    preflight = {
        "fixed_gain": {"gain_w_inverse": 1.0},
        "training": {"beta_rec": 1.0},
        "run_identity": {
            "task_id": joint.TASK_ID,
            "run_id": "run-id",
            "trajectory_id": "trajectory",
            "semantic_authority_id": "authority",
            "data_role": "canonical_train",
            "config_identity": {},
            "model_identity": {},
            "optimizer_identity": {},
            "sampler_identity": {},
        },
    }
    receipt = {"status": "READY"}
    seen: dict[str, object] = {}

    class _Model:
        def load_state_dict(self, state: object, strict: bool) -> None:
            seen["loaded_state"] = state

    class _Lifecycle:
        def __init__(self, identity: object, authorization_id: str) -> None:
            seen["authorization_id"] = authorization_id

        def mark_preflight_ready(self, selected_receipt: object) -> None:
            assert selected_receipt is receipt

        def open_scientific_attempt(self, path: Path, attempt_id: str, receipt: object) -> None:
            seen["path_absent_at_admission"] = not path.exists()
            seen["attempt_id"] = attempt_id
            assert not path.exists(), "fresh attempt directory was created before lifecycle admission"
            path.mkdir(parents=True, exist_ok=False)
            raise RuntimeError("stop after lifecycle admission")

    model = _Model()
    monkeypatch.setattr(joint.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(joint, "_read_ready", lambda *_args: (preflight, receipt))
    monkeypatch.setattr(joint, "_source_inputs", lambda *_args: (
        {}, {}, {}, references, object(), model, population, {}, torch.zeros(1), torch.zeros(1),
        torch.zeros(1, dtype=torch.long), object(), "evaluator.pt",
    ))
    monkeypatch.setattr(joint.torch, "load", lambda path, **_kwargs: population if Path(path).name == "fixed_population.pt" else {})
    monkeypatch.setattr(joint.baseline, "_assert_state_close", lambda *_args: 0.0)
    monkeypatch.setattr(joint, "_optimizer", lambda _model: object())
    monkeypatch.setattr(joint, "ResearchRunControl", _Lifecycle)
    args = type("Args", (), {
        "mode": "run",
        "output_dir": run_dir,
        "preflight_dir": preflight_dir,
        "resume_checkpoint": None,
        "checkpoint": "checkpoint.pt",
        "source_report": "source_report.json",
        "fixed_population": "fixed_population.pt",
        "config": "config.yaml",
        "data": "training.pt",
        "evaluator": None,
    })()

    with pytest.raises(RuntimeError, match="stop after lifecycle admission"):
        joint._run(args)

    assert seen["path_absent_at_admission"] is True
    assert seen["attempt_id"] == "attempt_02"
    assert run_dir.is_dir()

