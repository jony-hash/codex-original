from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepoLayoutTests(unittest.TestCase):
    def test_expected_scaffold_files_exist(self) -> None:
        expected = [
            ROOT / "pyproject.toml",
            ROOT / "pyrightconfig.json",
            ROOT / "train.py",
            ROOT / "configs" / "physics_cfg.yaml",
            ROOT / "configs" / "train_ldm.yaml",
            ROOT / "src" / "ldm_project" / "data.py",
            ROOT / "src" / "ldm_project" / "first_stage.py",
            ROOT / "src" / "ldm_project" / "diffusion.py",
            ROOT / "src" / "ldm_project" / "run_manager.py",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"missing expected file: {path}")

        self.assertFalse((ROOT / "train_ldm.py").exists(), "legacy entry script should be removed: train_ldm.py")
        self.assertFalse((ROOT / "sample_ldm.py").exists(), "legacy entry script should be removed: sample_ldm.py")

    def test_architecture_doc_mentions_optical_path_vae(self) -> None:
        architecture_doc = ROOT / "docs" / "ldm_migration" / "ldm_architecture_spec.md"
        content = architecture_doc.read_text(encoding="utf-8")
        self.assertIn("OpticalPathVAE", content)
        self.assertIn("classifier-free guidance", content)

    def test_train_config_defaults_to_mnist_subset(self) -> None:
        config_path = ROOT / "configs" / "train_ldm.yaml"
        content = config_path.read_text(encoding="utf-8")
        self.assertIn("dataset: mnist", content)
        self.assertIn("validation_fraction: 0.1", content)
        self.assertIn("subset_fraction: 0.1667", content)
        self.assertIn("encoder_fusion_mode: optical_only", content)
        self.assertIn("optical_field_mode: complex_cartesian", content)
        self.assertIn("primary_output: camera", content)
        self.assertIn("mode: mnist_complex_slm", content)
        self.assertIn("latent_source: mu", content)
        self.assertIn("base_channels: 64", content)


if __name__ == "__main__":
    unittest.main()
