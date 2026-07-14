from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from ldm_project.config import load_training_config, validate_physical_forward_contract


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "configs" / "train_v56_coherent_relay_from_scratch.yaml"
EVAL = ROOT / "configs" / "eval_v56_coherent_relay_from_scratch.yaml"
PREFLIGHT = ROOT / "tools" / "validate_v56_cuda_coherent_relay_preflight.py"


def test_v56_configs_lock_new_relay_and_core_objective():
    config_text = TRAIN.read_text(encoding="utf-8") + EVAL.read_text(encoding="utf-8")
    for legacy in ("v47", "v51", "v53", "one_shot_checkpoint", "checkpoint_path"):
        assert legacy not in config_text
    train = load_training_config(TRAIN)
    evaluation = load_training_config(EVAL)
    validate_physical_forward_contract(train, train.physics)
    validate_physical_forward_contract(evaluation, evaluation.physics)
    assert train.mode == "one_shot_distill"
    assert train.one_shot_epochs == 5
    assert train.learning_rate == pytest.approx(5e-6)
    assert train.seed == evaluation.seed == 1234
    assert train.physics.front_path_model == "coherent_imaging_4f"
    assert train.physics.back_path_model == "free_space"
    assert train.one_shot.ols_kind == "identity"
    assert train.one_shot.d2nn.grid_size == 100
    assert train.one_shot.d2nn.pixel_size == pytest.approx(2e-6)
    assert train.one_shot.d2nn.layer_spacing == pytest.approx(1.5e-3)
    assert train.one_shot.camera_recon_weight == pytest.approx(1.0)
    assert train.one_shot.d2nn == evaluation.one_shot.d2nn
    for name in (
        "shape_loss_weight", "edge_loss_weight", "camera_topology_weight",
        "camera_readability_weight", "student_label_margin_weight",
        "aux_semantic_weight", "aux_topology_weight", "code_diversity_weight",
        "cross_class_pairwise_weight",
    ):
        assert float(getattr(train.one_shot, name)) == 0.0


def test_v56_rejects_interventions():
    config = load_training_config(TRAIN)
    for field, value in (
        ("ols_kind", "tanh"),
        ("target_code_diagnostic_mode", "small_sharp_stroke_digit_code"),
        ("target_code_bridge_source", "encoded_amplitude"),
        ("ols_transfer_calibration", "phase_screen"),
    ):
        candidate = copy.deepcopy(config)
        setattr(candidate.one_shot, field, value)
        with pytest.raises(ValueError):
            validate_physical_forward_contract(candidate, candidate.physics)


def test_v56_preflight_is_cuda_only_and_has_no_checkpoint_or_training_path():
    source = PREFLIGHT.read_text(encoding="utf-8")
    assert "load_one_shot_checkpoint" not in source
    assert "run_one_shot_distill_training" not in source
    assert "one_shot_distill_eval" not in source
    spec = importlib.util.spec_from_file_location("v56_preflight", PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_preflight(TRAIN, ROOT / "missing-v55-reference.json", device_name="cpu")
    assert report["status"] == "physical-forward-model-invalid"
    assert report["diagnostic_only"] is True


def test_v56_preflight_requires_valid_v55_reference(tmp_path: Path):
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"status": "physical-forward-model-invalid"}), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("v56_preflight_ref", PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_preflight(TRAIN, reference, device_name="cpu")
    assert report["status"] == "physical-forward-model-invalid"
    assert "v58.2" in report["reasons"][0]
