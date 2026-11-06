from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.research.lightgen import execute_lightgen_small_joint_update_readability_pilot_v1 as pilot


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for package in ("R", "D"):
        for index in range(200):
            label = index % 10 if index % 5 else "unreadable"
            features = np.zeros(23, dtype=np.float64)
            features[0] = index % 10
            features[20:] = [index / 200, index / 100, (index % 3) / 3]
            records.append({
                "package": package,
                "card_id": f"R{index + 1:04d}",
                "branch": "G1" if package == "R" else ("D0" if index % 2 == 0 else "Dgrad"),
                "method": "G1" if package == "R" else "G4",
                "requested_label": index % 10,
                "human_label": label,
                "source_identity": [package, 0, index % 10, index],
                "features": features,
                "ed_prediction": index % 10,
                "ec_prediction": (index + 1) % 10,
            })
    return records


def test_auxiliary_fit_masks_unreadable_from_digit_head(monkeypatch) -> None:
    ce_targets: list[torch.Tensor] = []
    bce_targets: list[torch.Tensor] = []
    real_ce = pilot.F.cross_entropy
    real_bce = pilot.F.binary_cross_entropy_with_logits

    def capture_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not ce_targets:
            ce_targets.append(targets.detach().clone())
        return real_ce(logits, targets)

    def capture_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not bce_targets:
            bce_targets.append(targets.detach().clone())
        return real_bce(logits, targets)

    monkeypatch.setattr(pilot.F, "cross_entropy", capture_ce)
    monkeypatch.setattr(pilot.F, "binary_cross_entropy_with_logits", capture_bce)
    features = np.zeros((3, 23), dtype=np.float64)
    features[:, 0] = [0.0, 1.0, 2.0]
    fitted = pilot._fit_auxiliary_model(features, [2, "unreadable", 7])

    assert fitted["optimizer_steps"] == 500
    assert ce_targets[0].tolist() == [2, 7]
    assert bce_targets[0].tolist() == [1.0, 0.0, 1.0]
    assert fitted["scale"][1:].eq(1).all()


def test_review_prototype_consumes_whole_package_folds_and_writes_oof(tmp_path: Path) -> None:
    records = _records()
    captured: list[tuple[np.ndarray, list[int | str]]] = []
    original = pilot._fit_auxiliary_model

    def inspect_training_fold(features: np.ndarray, labels: list[int | str], steps: int = 500):
        captured.append((features.copy(), list(labels)))
        return original(features, labels, steps)

    with patch.object(pilot, "_fit_auxiliary_model", side_effect=inspect_training_fold):
        report = pilot._run_auxiliary_prototype(records, {"records": 400}, tmp_path)

    assert [fold["training_package"] for fold in report["folds"]] == ["R", "D"]
    assert [fold["validation_package"] for fold in report["folds"]] == ["D", "R"]
    assert [len(features) for features, _labels in captured] == [200, 200, 400]
    assert report["optimizer_calls"] == 1500
    assert len(report["out_of_fold"]["confusion_rows_human_0_to_9_then_unreadable"]) == 11
    with (tmp_path / "out_of_fold_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 400
    assert {row["package"] for row in rows} == {"R", "D"}
    assert (tmp_path / "development_model.pt").is_file()


def test_generation_effects_share_class_paired_bootstrap_indices(monkeypatch) -> None:
    control_rows = []
    joint_rows = []
    for label in range(10):
        for within_class in range(1000):
            row = label * 1000 + within_class
            shared = {
                "cohort_row": row, "requested_label": label,
                "proposal_id": row + 100, "component_id": label % 4,
                "ed_prediction": label, "ec_prediction": label,
                "view_mean_intensity": 0.1, "proxy_clipping_area_fraction": 0.02,
                "capture_power_w": 1e-6,
            }
            control_rows.append(shared)
            joint_rows.append({**shared, "view_mean_intensity": 0.11,
                               "proxy_clipping_area_fraction": 0.01,
                               "capture_power_w": 1.1e-6})

    class CountingRng:
        def __init__(self) -> None:
            self.calls = 0

        def integers(self, low: int, high: int, size: int) -> np.ndarray:
            self.calls += 1
            return np.arange(size, dtype=np.int64) % (high - low) + low

    rng = CountingRng()
    monkeypatch.setattr(pilot, "BOOTSTRAP_RESAMPLES", 3)
    monkeypatch.setattr(pilot.np.random, "default_rng", lambda _seed: rng)
    effects = pilot._paired_generation_effects(control_rows, joint_rows)

    assert rng.calls == 30
    assert all(item["shared_resampling_indices_across_metrics"] for item in effects.values())
    assert effects["ed_correct"]["point_estimate"] == 0.0
    assert np.isclose(effects["view_mean_intensity"]["point_estimate"], 0.01)


def test_joint_update_config_keeps_historical_system_gate_unchanged() -> None:
    config = pilot._read_config(pilot.ROOT / "configs/research/lightgen/small_joint_update_readability_pilot_v1.yaml")
    assert config["quality_gate"]["reconstruction_psnr_db_min"] == 20.0
    assert config["quality_gate"]["reconstruction_ssim_min"] == 0.80
    assert config["training"]["trainable_groups"] == ["encoder_phase", "decoder_phase"]


def test_resume_identity_preserves_distinct_registry_and_checkpoint_run_ids() -> None:
    config = {"execution_source_semantics_id": "small_joint_update_from_c45k_v1"}
    model_identity = {"ols": "gaussian_s12_v1"}
    identity_record = {
        "authorization_id": f"{pilot.TASK_ID}:seed:{pilot.SEED}:preflight_and_execution",
        "run_id": "checkpoint_progress_run_id",
        "source_run_id": "fixed_source_run_id",
        "identity": {
            "schema": "research_run_identity.v1",
            "task_id": pilot.TASK_ID,
            "run_id": "research_registry_run_id",
            "trajectory_id": f"{pilot.TASK_ID}:seed:{pilot.SEED}:source_step:{pilot.SOURCE_STEP}",
            "semantic_authority_id": pilot.AUTHORITY,
            "data_role": "MNIST_TRAIN_55000_DEV_5000_OFFICIAL_TEST_UNREAD",
            "config_identity": config,
            "model_identity": model_identity,
            "optimizer_identity": {},
            "sampler_identity": {},
            "execution_source_semantics_id": config["execution_source_semantics_id"],
            "scientific_attempt_id": f"joint_seed_{pilot.SEED}_from_C{pilot.SOURCE_STEP}",
        },
    }

    identity, checkpoint_run_id = pilot._validate_resume_identity(
        identity_record, {"run_id": "checkpoint_progress_run_id"},
        config, "fixed_source_run_id", model_identity,
    )

    assert identity.run_id == "research_registry_run_id"
    assert checkpoint_run_id == "checkpoint_progress_run_id"
    with pytest.raises(RuntimeError, match="resume run identity changed"):
        pilot._validate_resume_identity(
            identity_record, {"run_id": "different_progress_run_id"},
            config, "fixed_source_run_id", model_identity,
        )


def test_joint_restore_consumes_named_optimizer_steps_and_rejects_wrong_phase_count(monkeypatch) -> None:
    class TinyJointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Parameter(torch.ones(2))
            self.decoder = torch.nn.Parameter(torch.zeros(2))

    reference = TinyJointModel()
    reference_optimizer = torch.optim.Adam((reference.encoder, reference.decoder), lr=1e-3)
    for parameter in (reference.encoder, reference.decoder):
        reference_optimizer.state[parameter] = {
            "step": torch.tensor(50000.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }
    payload = {
        "model": reference.state_dict(),
        "optimizer": reference_optimizer.state_dict(),
        "optimizer_steps": {"encoder_adam_step": 50000, "decoder_adam_step": 50000},
    }
    monkeypatch.setattr(pilot.base, "_model", lambda *_args: TinyJointModel())
    monkeypatch.setattr(pilot.base, "_phase_groups", lambda model: (model.encoder, model.decoder))

    model, optimizer = pilot._restore_joint_state(payload, {}, torch.device("cpu"))
    assert pilot._optimizer_steps(optimizer, model.encoder, model.decoder) == (50000, 50000)

    mismatched = {**payload, "optimizer_steps": {**payload["optimizer_steps"], "encoder_adam_step": 45000}}
    with pytest.raises(RuntimeError, match="Adam step-to-phase mapping changed"):
        pilot._restore_joint_state(mismatched, {}, torch.device("cpu"))


def test_machine_report_html_consumes_the_registered_primary_gate(tmp_path: Path) -> None:
    report = {
        "status": "COMPLETE_DIAGNOSTIC",
        "scientific_decision": "STRUCTURAL_INCREMENT_SUPPORTED",
        "primary_gate": {"metric": "mean_DEV_SSIM_J_minus_D0", "point_estimate": 0.0065,
                         "ci_95": [0.0063, 0.0067]},
    }

    pilot._write_optical_html(tmp_path, report)

    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "J−D0平均DEV SSIM：0.006500" in page
    assert "[0.006300, 0.006700]" in page
    assert "STRUCTURAL_INCREMENT_SUPPORTED" in page


def test_saved_report_closeout_rejects_incomplete_population_and_accepts_complete(tmp_path: Path) -> None:
    optical = tmp_path / "optical"
    optical.mkdir()
    required = (
        "checkpoint_step_005000.pt", "terminal_model.pt", "J/endpoint_summary.json",
        "dev_comparison.png", "d0_native_g4.png", "j_native_g4.png",
        "J/generated_cohort.pt", "J/dev_display_examples.pt",
    )
    for relative in required:
        path = optical / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"evidence")
    for relative, count in (("J/reconstruction_metrics_dev.csv", 5_000),
                            ("J/generation_metrics_per_image.csv", 10_000)):
        path = optical / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row"])
            writer.writerows([[index] for index in range(count)])
    effect = {"point_estimate": 0.0065, "ci_95": [0.0063, 0.0067]}
    report = {
        "schema": "lightgen_small_joint_update_readability_report.v1",
        "task_id": pilot.TASK_ID,
        "run_id": "saved-run",
        "status": "COMPLETE_DIAGNOSTIC",
        "measurement_complete": True,
        "training": {"new_scientific_updates": pilot.UPDATES,
                     "global_step": pilot.ENDPOINT_STEP},
        "primary_gate": {"metric": "mean_DEV_SSIM_J_minus_D0", **effect},
    }
    (optical / "machine_report.json").write_text(json.dumps(report), encoding="utf-8")
    (optical / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (optical / "J/endpoint_summary.json").write_text(json.dumps({
        "measurement_complete": True, "global_step": pilot.ENDPOINT_STEP,
        "primary_effects": {"ssim": effect},
    }), encoding="utf-8")
    progress = {
        "stage": "EVALUATING", "completed_updates": pilot.UPDATES,
        "global_step": pilot.ENDPOINT_STEP, "status": "RUNNING",
        "last_complete_save_unix": 1.0, "run_id": "saved-run",
    }

    assert pilot._validate_saved_machine_report(optical, progress) == report
    (optical / "J/generation_metrics_per_image.csv").unlink()
    with pytest.raises(RuntimeError, match="missing required artifacts"):
        pilot._validate_saved_machine_report(optical, progress)

