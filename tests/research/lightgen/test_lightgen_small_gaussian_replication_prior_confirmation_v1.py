from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

import scripts.research.lightgen.execute_lightgen_small_gaussian_replication_prior_confirmation_v1 as task
import scripts.research.lightgen.execute_lightgen_small_conditional_mixture_prior_ab_v1 as prior


def _synthetic_labels() -> tuple[torch.Tensor, torch.Tensor]:
    counts = task.TRAIN_LABEL_COUNTS
    train_labels = torch.cat([torch.full((count,), label, dtype=torch.long)
                              for label, count in enumerate(counts)])
    labels = torch.cat((train_labels, torch.arange(10).repeat_interleave(500)))
    return labels, torch.arange(task.TRAIN_COUNT)


def test_registered_config_and_reference_schedule_consumer() -> None:
    config, runtime = task._read_config(
        Path("configs/research/lightgen/small_gaussian_replication_prior_confirmation_v1.yaml")
    )
    assert runtime["optics"]["small"]["production_propagation_grid"] == 2048
    assert config["training"]["seeds"] == list(task.SEEDS)

    labels, train_rows = _synthetic_labels()
    schedule = task._full_schedule(labels, train_rows, task.REFERENCE_SEED)
    reference = {"schedule": schedule[35_000:45_000]}
    receipt = task._validate_reference_schedule(
        schedule, reference, schedule[30_000:35_000],
    )
    assert receipt["both_saved_segments_replayed_rowwise"] is True
    wrong_segment = schedule[30_000:35_000].clone()
    wrong_segment[-1, -1] = train_rows[0]
    with pytest.raises(RuntimeError, match="30,000-35,000"):
        task._validate_reference_schedule(schedule, reference, wrong_segment)


def test_g1_sampling_is_seeded_power_bounded_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    fit = {"mean": torch.zeros(200, dtype=torch.float64),
           "cholesky": torch.eye(200, dtype=torch.float64) * 1e-6}
    scale = torch.full((100,), 1e-10, dtype=torch.float64)
    first = task._sample_g1_class(fit, scale, label=3, stream=2)
    second = task._sample_g1_class(fit, scale, label=3, stream=2)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[2], second[2])
    assert first[0].shape == (task.SAMPLE_COUNT, 100)
    assert torch.unique(first[2]).numel() == task.SAMPLE_COUNT
    assert bool((first[0].abs().square().sum(1) <= task.base.MAX_MODAL_POWER_W).all())
    assert first[3]["normal_seed"] == task.G1_NORMAL_SEED_BASE + 200 + 3

    monkeypatch.setattr(task, "MAX_PROPOSALS", 0)
    with pytest.raises(RuntimeError, match="proposal cap"):
        task._sample_g1_class(fit, scale, label=3, stream=2)


def test_k1_mixture_matches_single_gaussian_and_cholesky_orientation() -> None:
    features = torch.tensor([
        [-1.0, 0.0], [-0.5, 0.25], [0.0, 1.0], [0.5, -0.25], [1.0, 0.5], [1.5, -0.5],
    ], dtype=torch.float64)
    single = prior._fit_single_class(features)
    mixture = prior._fit_mixture_class(features, 1, seed=37, ridge=single["ridge"])
    assert torch.allclose(mixture["weights"], torch.ones(1, dtype=torch.float64))
    assert torch.allclose(mixture["means"][0], single["mean"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(mixture["covariances"][0], single["covariance"], atol=1e-12, rtol=1e-12)
    regularized = mixture["regularized_covariances"][0]
    lower = mixture["choleskies"][0]
    assert torch.allclose(lower @ lower.T, regularized, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(lower.T @ lower, regularized, atol=1e-6, rtol=1e-6)


def test_generation_summary_rejects_incomplete_class_population(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task, "SAMPLE_COUNT", 2)
    with pytest.raises(RuntimeError, match="incomplete population"):
        task._cohort_summary([], samples_per_class=2)


def test_multistream_bootstrap_consumes_all_strata_and_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task, "SAMPLE_COUNT", 2)
    monkeypatch.setattr(task, "BOOTSTRAP_RESAMPLES", 80)
    metrics = (
        "E_D_correct", "E_C_correct", "proxy_clipping_area_fraction",
        "proxy_clipped_excess_fraction", "capture_power_w", "view_mean_intensity",
        "raw_peak_w_per_pixel", "E_D_requested_margin", "E_C_requested_margin",
    )
    rows_by_method: dict[str, dict[int, list[dict[str, float | int]]]] = {"G1": {}, "G4": {}}
    for method in ("G1", "G4"):
        for stream in range(3):
            rows: list[dict[str, float | int]] = []
            for label in range(10):
                for sample in range(2):
                    row: dict[str, float | int] = {"requested_label": label}
                    for metric in metrics:
                        if metric in {"E_D_correct", "E_C_correct"}:
                            row[metric] = int(method == "G4" and sample == 0)
                        else:
                            row[metric] = float(label + stream + sample + (method == "G4"))
                    rows.append(row)
            rows_by_method[method][stream] = rows

    first = task._bootstrap_multistream(rows_by_method, seed=17)
    second = task._bootstrap_multistream(rows_by_method, seed=17)
    assert first == second
    assert first["resamples"] == 80
    assert first["differences"]["E_D_correct"]["observed_difference"] == 0.5
    assert first["differences"]["E_C_correct"]["observed_difference"] == 0.5
    assert len(first["per_class_differences"]["E_D_correct"]) == 10


def test_resume_state_requires_matching_identity_and_incomplete_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {"schema": "test"}
    state_path = tmp_path / "progress.json"
    state_path.write_text(json.dumps({
        "task_id": task.TASK_ID, "config_identity": config, "output_dir": str(run_dir),
    }), encoding="utf-8")
    assert task._read_resume_state(state_path, run_dir, config)["task_id"] == task.TASK_ID

    (run_dir / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="completed task"):
        task._read_resume_state(state_path, run_dir, config)


def test_fresh_adam_checkpoint_records_zero_updates_before_first_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.zeros(1))
            self.decoder_phase = torch.nn.Parameter(torch.zeros(1))

    model = TinyModel()
    optimizer = torch.optim.Adam((model.encoder_phase, model.decoder_phase), lr=1e-3)
    monkeypatch.setattr(
        task.base, "_phase_groups",
        lambda loaded: (loaded.encoder_phase, loaded.decoder_phase),
    )
    monkeypatch.setattr(task, "_model_identity", lambda: {"basis": "gaussian_s12_v1"})
    monkeypatch.setattr(task, "_rng_state", lambda: {"rng": torch.tensor([0])})

    payload = task._checkpoint_payload(
        model=model, optimizer=optimizer, seed=2026102151, run_id="seed_2026102151",
        schedule=torch.zeros(45_000, 10, dtype=torch.long), scale=torch.ones(100),
        power_reference=torch.ones(5), gain=torch.tensor(1.0),
        weights={"reconstruction": 1.0}, lambda_ssim=None, step=0,
        history=[], elapsed=0.0,
    )

    assert payload["optimizer_step_count"] == 0
    assert payload["completed_global_step"] == 0

    (model.encoder_phase.sum() + model.decoder_phase.sum()).backward()
    optimizer.step()
    assert task._phase_optimizer_update_count(optimizer, model) == 1
    optimizer.state.pop(model.decoder_phase)
    with pytest.raises(RuntimeError, match="partially initialized"):
        task._phase_optimizer_update_count(optimizer, model)


def test_feature_cache_saves_train_labels_aligned_to_global_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task, "TRAIN_COUNT", 2)
    monkeypatch.setattr(task, "BATCH_SIZE", 1)
    monkeypatch.setattr(task, "image_batch", lambda pixels, rows, *args: rows.float().unsqueeze(1))
    monkeypatch.setattr(task.base, "_forward", lambda model, batch: {
        "ols_coefficients": torch.zeros(batch.shape[0], 100, dtype=torch.complex64),
    })
    monkeypatch.setattr(task.base, "_gauge_and_scale", lambda coefficients, scale: coefficients)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    output_path = tmp_path / "features.pt"
    train_rows = torch.tensor([2, 7], dtype=torch.long)
    train_labels = torch.tensor([4, 9], dtype=torch.long)

    task._encode_features(
        model=torch.nn.Identity(), pixels=torch.zeros(10, 1), train_rows=train_rows,
        labels=train_labels, scale=torch.ones(100), device=torch.device("cpu"),
        output_path=output_path, seed=7,
    )

    saved = torch.load(output_path, map_location="cpu", weights_only=False)
    assert torch.equal(saved["rows"], train_rows)
    assert torch.equal(saved["labels"], train_labels)


def test_checkpoint_restore_consumes_attempt_level_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.zeros(1))
            self.decoder_phase = torch.nn.Parameter(torch.zeros(1))

    run_root = tmp_path / "attempt"
    seed_dir = run_root / "seed_7"
    seed_dir.mkdir(parents=True)
    config = {"schema": "fixed"}
    (run_root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = TinyModel()
    optimizer = torch.optim.Adam((model.encoder_phase, model.decoder_phase), lr=1e-3)
    (model.encoder_phase.sum() + model.decoder_phase.sum()).backward()
    optimizer.step()
    schedule = torch.arange(30, dtype=torch.long).reshape(3, 10)
    checkpoint = {
        "schema": "lightgen_small_gaussian_replication_checkpoint.v1",
        "task_id": task.TASK_ID, "seed": 7, "model_identity": {"basis": "gaussian_s12_v1"},
        "completed_global_step": 1, "completed_updates": 1, "sample_cursor": 1,
        "next_sample_rows": schedule[1].clone(), "optimizer_step_count": 1,
        "schedule": schedule.clone(), "official_test_reads": 0,
        "model": task.base._cpu_copy(model.state_dict()),
        "optimizer": task.base._cpu_copy(optimizer.state_dict()),
        "scale": torch.ones(100), "power_reference": torch.ones(5),
        "camera_gain": torch.tensor(1.0), "weights": {"reconstruction": 1.0},
    }
    checkpoint_path = seed_dir / "checkpoint_step_000001.pt"
    torch.save(checkpoint, checkpoint_path)
    torch.save({
        "schedule": schedule, "scale": checkpoint["scale"],
        "power_reference": checkpoint["power_reference"],
        "camera_gain": checkpoint["camera_gain"], "weights": checkpoint["weights"],
    }, seed_dir / "initial_references.pt")
    monkeypatch.setattr(task.base, "_phase_groups",
                        lambda loaded: (loaded.encoder_phase, loaded.decoder_phase))
    monkeypatch.setattr(task.base, "_model", lambda *args: TinyModel())
    monkeypatch.setattr(task, "_model_identity", lambda: {"basis": "gaussian_s12_v1"})
    monkeypatch.setattr(task, "_set_rng_state", lambda *_args: None)

    restored, restored_model, restored_optimizer = task._restore_checkpoint(
        checkpoint_path=checkpoint_path, run_dir=seed_dir, seed=7, config=config,
        runtime={}, schedule=schedule, device=torch.device("cpu"),
    )

    assert restored["completed_global_step"] == 1
    assert task._phase_optimizer_update_count(restored_optimizer, restored_model) == 1


def test_report_links_c_dev_examples_to_reference_gallery(tmp_path: Path) -> None:
    summary = {
        "E_D_macro_recall": 0.1,
        "E_C_macro_recall": 0.2,
        "macro_proxy_clipping_area_fraction": 0.03,
    }
    report = {
        "status": "COMPLETE_DIAGNOSTIC",
        "decision": {"result": "diagnostic"},
        "historical_system_quality_standard": {},
        "endpoints": {
            "C_seed_2026101501": {
                "generation": {
                    "G1": {"three_stream_equal_weight": summary},
                    "G4": {"three_stream_equal_weight": summary},
                    "shape_diversity": {"G1": {"minimum_ratio": 0.1},
                                        "G4": {"minimum_ratio": 0.2}},
                }
            }
        },
    }
    report_path = tmp_path / "report.html"
    task._write_report_html(report_path, report)
    rendered = report_path.read_text(encoding="utf-8")
    assert "href='reference_C/gallery/dev_examples.html'" in rendered
    assert "href='generation/C_seed_2026101501/gallery/dev_examples.html'" not in rendered


def test_source_cohort_replay_passes_both_identity_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    labels = torch.arange(10)
    images = torch.zeros(10, 1, 28, 28)
    logits = torch.zeros(10, 10)
    saved = {
        "run_id": "prior-run", "source_run_id": "source-run",
        "ols_coefficients": torch.zeros(10, 100, dtype=torch.complex64),
        "requested_labels": labels, "component_ids": torch.zeros(10, dtype=torch.long),
        "proposal_ids": torch.arange(10), "images": images,
        "ed_logits": logits, "ec_logits": logits,
        "ed_predictions": labels, "ec_predictions": labels,
    }
    for method in ("G1", "G4"):
        torch.save(saved, prior_dir / f"generated_{method}.pt")

    calls: list[tuple[str, str, str]] = []

    def validate(payload: dict[str, object], method: str, run_id: str, source_run_id: str) -> None:
        assert payload["run_id"] == run_id
        assert payload["source_run_id"] == source_run_id
        calls.append((method, run_id, source_run_id))

    def decode(method: str, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        return {"images": images, "ed_logits": logits, "ec_logits": logits,
                "ed_predictions": labels, "ec_predictions": labels}

    monkeypatch.setattr(task.prior, "_validate_saved_cohort", validate)
    monkeypatch.setattr(task.prior, "_decode_cohort", decode)
    monkeypatch.setattr(task.prior, "_validate_readout_numpy", lambda _: {"passed": True})

    receipt = task._replay_source_cohorts(
        model=torch.nn.Identity(), source_checkpoint={"camera_gain": torch.tensor(1.0)},
        prior_dir=prior_dir, evaluator_d=torch.nn.Identity(), evaluator_c=torch.nn.Identity(),
        device=torch.device("cpu"), output_dir=output_dir,
        expected_prior_run_id="prior-run", expected_source_run_id="source-run",
    )
    assert receipt["passed"] is True
    assert calls == [("G1", "prior-run", "source-run"), ("G4", "prior-run", "source-run")]


def test_generation_endpoint_selects_one_first_sample_from_each_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task, "SAMPLE_COUNT", 2)
    monkeypatch.setattr(task, "BOOTSTRAP_RESAMPLES", 5)
    labels = torch.arange(10).repeat_interleave(2)
    coefficient_values: list[torch.Tensor] = []
    window_inputs: list[torch.Tensor] = []
    phase_inputs: list[torch.Tensor] = []

    def sample_population(method: str, priors: dict[str, object], scale: torch.Tensor,
                          stream: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                torch.Tensor, dict[str, object]]:
        coefficients = torch.zeros(20, 100, dtype=torch.complex64)
        coefficients[:, 0] = labels.to(torch.float32)
        coefficient_values.append(coefficients)
        return coefficients, labels, torch.zeros(20, dtype=torch.long), torch.arange(2).repeat(10), {}

    def decode(method: str, model: torch.nn.Module, evaluator_d: torch.nn.Module,
               evaluator_c: torch.nn.Module, coefficients: torch.Tensor, requested: torch.Tensor,
               components: torch.Tensor, proposals: torch.Tensor, gain: torch.Tensor,
               device: torch.device, capture_count: int = 10) -> dict[str, torch.Tensor]:
        count = requested.numel()
        logits = torch.zeros(count, 10)
        return {
            "images": torch.zeros(count, 1, 28, 28), "requested_labels": requested,
            "component_ids": components, "proposal_ids": proposals,
            "ols_coefficients": coefficients,
            "ed_predictions": requested, "ec_predictions": requested,
            "ed_requested_margin": torch.ones(count), "ec_requested_margin": torch.ones(count),
            "ed_top_margin": torch.ones(count), "ec_top_margin": torch.ones(count),
            "capture_power_w": torch.ones(count), "raw_mean_w_per_pixel": torch.ones(count),
            "raw_peak_w_per_pixel": torch.ones(count), "view_mean_intensity": torch.full((count,), 0.2),
            "proxy_clipping_area_fraction": torch.zeros(count),
            "proxy_clipped_excess_fraction": torch.zeros(count),
            "ed_logits": logits, "ec_logits": logits,
        }

    def window_check(model: torch.nn.Module, coefficients: torch.Tensor,
                     runtime: dict[str, object], device: torch.device) -> dict[str, object]:
        window_inputs.append(coefficients[:, 0].real.clone())
        return {"sample_count": 10, "requested_labels": list(range(10))}

    def phase_check(model: torch.nn.Module, coefficients: torch.Tensor,
                    device: torch.device) -> dict[str, object]:
        phase_inputs.append(coefficients[:, 0].real.clone())
        return {"passed": True}

    shape = {method: {"minimum_ratio": 1.0,
                      "per_class": [{"label": label, "ratio": 1.0} for label in range(10)]}
             for method in ("G1", "G4")}
    monkeypatch.setattr(task, "_sample_population", sample_population)
    monkeypatch.setattr(task.prior, "_decode_cohort", decode)
    monkeypatch.setattr(task, "_generated_window_check", window_check)
    monkeypatch.setattr(task, "_global_phase_check", phase_check)
    monkeypatch.setattr(task, "_shape_ratios", lambda *args: shape)
    monkeypatch.setattr(task, "_phase_gauge_association", lambda *args: {"passed": True})
    monkeypatch.setattr(task.base, "_pair_distances", lambda *args: torch.ones(1))
    monkeypatch.setattr(task.prior, "_validate_readout_numpy", lambda *_: {"passed": True})
    monkeypatch.setattr(task.prior, "_write_gallery", lambda *_: None)
    cohort_summary = task._cohort_summary
    monkeypatch.setattr(
        task, "_cohort_summary",
        lambda rows, samples_per_class=2: cohort_summary(rows, samples_per_class),
    )

    result = task._run_generation_endpoint(
        model_name="test_seed", seed=7, model=torch.nn.Identity(), priors={},
        scale=torch.ones(100), gain=torch.tensor(1.0), evaluator_d=torch.nn.Identity(),
        evaluator_c=torch.nn.Identity(), device=torch.device("cpu"),
        pixels=torch.zeros(10, 28, 28, dtype=torch.uint8), dev_rows=torch.arange(10),
        all_labels=torch.arange(10), output_dir=tmp_path, bootstrap_seed=17,
        assignments=torch.zeros(1, dtype=torch.long), train_features=torch.zeros(1, 200),
        train_labels=torch.zeros(1, dtype=torch.long), config={}, runtime={},
    )
    expected_first_by_class = torch.arange(10, dtype=torch.float32)
    assert len(result["rows"]) == 120
    assert len(coefficient_values) == 6
    assert len(window_inputs) == len(phase_inputs) == 2
    assert all(torch.equal(values, expected_first_by_class) for values in window_inputs + phase_inputs)


def test_recovery_worker_uses_importable_module_from_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.zeros(1))
            self.decoder_phase = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(task.base, "_model", lambda *args: TinyModel())
    monkeypatch.setattr(
        task.base, "_phase_groups",
        lambda model: (model.encoder_phase, model.decoder_phase),
    )
    monkeypatch.setattr(task, "_update_targets", lambda *args: (torch.zeros(10, 1), torch.zeros(10, 1)))
    monkeypatch.setattr(task, "_rng_state", lambda: {"test": torch.tensor(1)})
    monkeypatch.setattr(task, "_set_rng_state", lambda *_: None)
    monkeypatch.setattr(task, "_tree_max_error", lambda *args: 0.0)

    def train_one(model: torch.nn.Module, optimizer: torch.optim.Adam, *args: object) -> dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        loss = model.encoder_phase.sum() + model.decoder_phase.sum()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    monkeypatch.setattr(task, "_train_one", train_one)
    original_run = task.subprocess.run
    module_name = "scripts.research.lightgen.execute_lightgen_small_gaussian_replication_prior_confirmation_v1"
    recovery_output = {"model": {}, "optimizer": {}, "metrics": {}, "rng": {}, "seed": 5, "stage": "bogt"}

    def run_worker(command: list[str], *, cwd: Path, capture_output: bool,
                   text: bool, check: bool) -> object:
        assert command[1:3] == ["-m", module_name]
        assert Path(cwd).resolve() == task.ROOT.resolve()
        probe = original_run(
            [task.sys.executable, "-c", f"import {module_name}"],
            cwd=cwd, capture_output=True, text=True, check=False,
        )
        assert probe.returncode == 0, probe.stderr
        worker_dir = Path(command[command.index("--recovery-worker-dir") + 1])
        atomic_save = task.atomic_torch_save
        atomic_save(worker_dir / "recovery_output.pt", recovery_output)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(task.subprocess, "run", run_worker)
    config_path = Path("configs/research/lightgen/small_gaussian_replication_prior_confirmation_v1.yaml")
    model = TinyModel()
    optimizer = torch.optim.Adam((model.encoder_phase, model.decoder_phase), lr=1e-3)
    result = task._preflight_recovery(
        model=model, optimizer=optimizer, pixels=torch.zeros(20, 28, 28, dtype=torch.uint8),
        schedule=torch.zeros(10, 10, dtype=torch.long), start_step=0, seed=5, stage="bogt",
        scale=torch.ones(100), power_reference=torch.ones(5), gain=torch.tensor(1.0),
        weights={"reconstruction": 1.0}, lambda_ssim=None, config_path=config_path,
        directory=tmp_path / "recovery",
    )
    assert result["passed"] is True
    assert result["independent_process_fourth_update"] is True


def test_recovery_worker_cli_does_not_require_normal_run_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/research/lightgen/small_gaussian_replication_prior_confirmation_v1.yaml")
    expected_config = (Path.cwd() / config_path).resolve()
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(task.sys, "argv", [
        "execute_lightgen_small_gaussian_replication_prior_confirmation_v1.py",
        "--recovery-worker-dir", str(tmp_path), "--config", str(config_path),
    ])
    monkeypatch.setattr(
        task, "_recovery_worker",
        lambda directory, loaded_config: calls.append((directory, loaded_config)),
    )
    task.main()
    assert calls == [(tmp_path.resolve(), expected_config)]
