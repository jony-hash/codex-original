from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ldm_project.v2.application.cli import main
from ldm_project.v2.config import load_config


def _config(path: Path) -> None:
    path.write_text(
        """schema_version: 2
system:
  kind: formal_amplitude_finite_crop_identity
run:
  run_name: unit
  output_root: outputs
  evidence_tier: diagnostic
  device: cpu
""",
        encoding="utf-8",
    )


def test_v2_config_is_strict_but_v1_loader_can_parse_the_file(tmp_path: Path) -> None:
    path = tmp_path / "v2.yaml"
    _config(path)
    assert load_config(path).schema_version == 2
    from ldm_project.config import load_training_config

    assert load_training_config(path).mode == "first_stage"


def test_evaluate_requires_checkpoint_driven_acceptance_inputs(tmp_path: Path) -> None:
    config = tmp_path / "v2.yaml"
    _config(config)
    output = tmp_path / "metrics.json"
    with pytest.raises(SystemExit):
        main(["student", "evaluate", "--config", str(config), "--output", str(output)])


def test_non_identity_acceptance_is_rejected_before_model_construction(tmp_path: Path) -> None:
    config = tmp_path / "formal.yaml"
    config.write_text(
        """schema_version: 2
system:
  kind: formal_amplitude_finite_crop_identity
prototype_calibration:
  epochs: 2
  learning_rate: 0.0005
  fixed_exposure_scale: 1.0
run:
  run_name: unit
  output_root: outputs
  evidence_tier: acceptance
  device: cpu
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="only conditional_identity_logit_amplitude_v2"):
        load_config(config)
