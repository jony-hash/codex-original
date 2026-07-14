from __future__ import annotations

from pathlib import Path
import unittest


class V51OpticalFieldHarnessTests(unittest.TestCase):
    def test_v51_config_is_continuous_complex_field_sample_only(self) -> None:
        from ldm_project.config import load_training_config

        config = load_training_config(
            Path("experiments/archive/pre_v56/configs/sample_twophoton_100x100_v51_real_complex_field_probe_a_s025.yaml")
        )

        self.assertEqual(config.mode, "one_shot_sample")
        self.assertEqual(config.seed, 1234)
        self.assertEqual(config.physics.front_path_model, "relay_coherent_psf")
        self.assertEqual(config.physics.front_relay_psf_sigma_px, 0.0)
        self.assertEqual(config.physics.back_path_model, "relay_psf")
        self.assertEqual(config.one_shot.target_code_diagnostic_mode, "none")
        self.assertEqual(config.one_shot.target_code_bridge_source, "none")
        self.assertEqual(config.one_shot.d2nn.grid_size, 100)
        self.assertEqual(config.one_shot.d2nn.propagation_grid_size, 1024)
        self.assertAlmostEqual(config.one_shot.d2nn.pixel_size, 2.0e-6)
        self.assertAlmostEqual(config.one_shot.d2nn.layer_spacing, 0.25e-3)

    def test_v51_launcher_contains_one_cuda_sample_and_no_training_or_eval(self) -> None:
        text = Path("experiments/archive/pre_v57/tools/run_v51_real_complex_field_sample.sh").read_text(encoding="utf-8")

        self.assertEqual(text.count("--stage one_shot_sample"), 1)
        self.assertEqual(text.count("--device cuda"), 1)
        self.assertNotIn("--stage one_shot_distill", text)
        self.assertNotIn("--stage one_shot_distill_eval", text)
        self.assertNotIn("sweep", text)
