from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from ldm_project.config import load_training_config, validate_physical_forward_contract


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = ROOT / "configs" / "train_v53_physical_forward_from_scratch.yaml"
EVAL_CONFIG = ROOT / "configs" / "eval_v53_physical_forward_from_scratch.yaml"
READABLE_SPLIT_ROOT = ROOT / "outputs" / "20260514_unversioned" / "runs" / (
    "20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_diffusion_"
    "e40_gs40_readable_split_from_full15k_cuda"
) / "artifacts"


def test_v53_train_and_eval_configs_share_physical_contract_and_core_objective():
    train = load_training_config(TRAIN_CONFIG)
    evaluation = load_training_config(EVAL_CONFIG)
    assert train.one_shot.physical_forward_contract is False
    assert evaluation.one_shot.physical_forward_contract is False

    assert train.mode == "one_shot_distill"
    assert train.batch_size == 16
    assert train.learning_rate == pytest.approx(5.0e-6)
    assert train.one_shot_epochs == 5
    assert evaluation.mode == "one_shot_distill_eval"
    assert evaluation.one_shot_epochs == 0
    assert train.one_shot.d2nn == evaluation.one_shot.d2nn
    assert train.one_shot.camera_recon_weight == pytest.approx(1.0)
    for name in (
        "shape_loss_weight",
        "edge_loss_weight",
        "camera_topology_weight",
        "camera_readability_weight",
        "student_label_margin_weight",
        "aux_semantic_weight",
        "aux_topology_weight",
        "code_diversity_weight",
    ):
        assert float(getattr(train.one_shot, name)) == 0.0


def test_v53_rejects_interventions_and_non_identity_ols():
    config = load_training_config(TRAIN_CONFIG)
    config.one_shot.physical_forward_contract = True
    config.physics.physical_forward_contract = True
    config.physics.front_path_model = "coherent_imaging_4f"
    for field, value, message in (
        ("ols_kind", "tanh", "identity OLS"),
        ("target_code_diagnostic_mode", "small_sharp_stroke_digit_code", "target-code"),
        ("target_code_bridge_source", "encoded_amplitude", "target-code"),
        ("ols_transfer_calibration", "phase_screen", "calibration"),
    ):
        candidate = copy.deepcopy(config)
        setattr(candidate.one_shot, field, value)
        with pytest.raises(ValueError, match=message):
            validate_physical_forward_contract(candidate, candidate.physics)


def test_readable_split_teacher_targets_are_present_and_explicitly_separate():
    assert (READABLE_SPLIT_ROOT / "teacher_distill_dataset_readable_train.pt").is_file()
    assert (READABLE_SPLIT_ROOT / "teacher_distill_dataset_readable_val.pt").is_file()
    assert (READABLE_SPLIT_ROOT / "teacher_distill_dataset_readable_test.pt").is_file()
    assert "teacher_distill" in READABLE_SPLIT_ROOT.as_posix()
    assert "one_shot" not in READABLE_SPLIT_ROOT.name


def test_cuda_preflight_is_cuda_only_and_does_not_load_checkpoint_or_train():
    source = (ROOT / "tools" / "validate_v53_cuda_physical_preflight.py").read_text(encoding="utf-8")
    assert "load_one_shot_checkpoint" not in source
    assert "run_one_shot_distill_training" not in source
    assert "one_shot_distill_eval" not in source

    spec = importlib.util.spec_from_file_location(
        "v53_preflight",
        ROOT / "tools" / "validate_v53_cuda_physical_preflight.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.run_preflight(TRAIN_CONFIG, device_name="cpu")
    assert report["status"] == "physical-forward-model-invalid"
    assert "CUDA" in report["reason"] or "cuda" in report["reason"]
