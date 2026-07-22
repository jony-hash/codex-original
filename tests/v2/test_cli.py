from __future__ import annotations

from pathlib import Path

import torch

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


def test_evaluate_subcommand_writes_diagnostic_metrics(tmp_path: Path) -> None:
    config = tmp_path / "v2.yaml"
    _config(config)
    prediction = tmp_path / "prediction.pt"
    target = tmp_path / "target.pt"
    output = tmp_path / "metrics.json"
    torch.save(torch.ones(2, 1, 4, 4), prediction)
    torch.save(torch.ones(2, 1, 4, 4), target)
    assert main(["student", "evaluate", "--config", str(config), "--predictions", str(prediction), "--targets", str(target), "--output", str(output)]) == 0
    assert '"diagnostic_only": true' in output.read_text(encoding="utf-8")
