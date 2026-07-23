from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.cli import main
from ldm_project.v2.config import load_config
from ldm_project.v2.evidence import canonical_sha256, tensor_artifact_ref, tensor_sha256
from ldm_project.v2.student.clean_targets import CleanTargetManifestV2


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


def test_formal_calibration_rejects_missing_reachability_before_model_construction(tmp_path: Path) -> None:
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
    clean = tmp_path / "clean.pt"
    torch.save(torch.full((10, 1, 200, 200), 0.5), clean)
    loaded = load_config(config)
    manifest = CleanTargetManifestV2(
        2,
        str(tmp_path),
        "train",
        128,
        tuple(range(10)),
        tensor_artifact_ref(clean, role="clean_mnist_targets", git_commit="unit"),
        tensor_sha256(torch.load(clean, weights_only=False)),
        (10, 1, 200, 200),
        "torch.float32",
        canonical_sha256(loaded.model_dump(mode="json")),
        "unit",
        system_contract_hash=canonical_sha256(loaded.system.model_dump(mode="json")),
        producer_config_hash=canonical_sha256(loaded.model_dump(mode="json")),
    )
    path = tmp_path / "clean.json"
    path.write_text(json.dumps(manifest.to_dict() | {"manifest_hash": manifest.manifest_hash}), encoding="utf-8")
    with pytest.raises(ValueError, match="reachability-evidence"):
        main(["student", "calibrate-prototypes", "--config", str(config), "--clean-target-manifest", str(path)])
