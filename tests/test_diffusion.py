from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for diffusion tests")
class ConditionedDiffusionTests(unittest.TestCase):
    def test_conditioned_unet_preserves_odd_spatial_shape(self) -> None:
        import torch

        from ldm_project.config import DiffusionConfig
        from ldm_project.diffusion import LatentDiffusionUNet

        model = LatentDiffusionUNet(4, DiffusionConfig())
        # 以 tuple 传 shape，避免静态类型检查将 randn 误匹配到 names 重载。
        latents = torch.randn((2, 4, 25, 25))
        timesteps = torch.tensor([0, 17], dtype=torch.long)
        labels = torch.tensor([1, 7], dtype=torch.long)

        context = model.encode_condition(labels, dropout_prob=0.0)
        outputs = model(latents, timesteps, context)

        self.assertEqual(outputs.shape, latents.shape)

    def test_force_unconditional_matches_full_condition_dropout(self) -> None:
        import torch

        from ldm_project.config import DiffusionConfig
        from ldm_project.diffusion import LatentDiffusionUNet

        model = LatentDiffusionUNet(4, DiffusionConfig())
        labels = torch.tensor([3, 5], dtype=torch.long)

        unconditional = model.encode_condition(labels, force_unconditional=True, dropout_prob=0.0)
        dropped = model.encode_condition(labels, dropout_prob=1.0)

        self.assertTrue(torch.allclose(unconditional, dropped))

    def test_cfg_formula_matches_expected_values(self) -> None:
        import torch

        from ldm_project.training import apply_classifier_free_guidance

        conditioned = torch.tensor([2.0])
        unconditioned = torch.tensor([1.0])

        self.assertTrue(
            torch.allclose(
                apply_classifier_free_guidance(conditioned, unconditioned, guidance_scale=1.0),
                conditioned,
            )
        )
        self.assertTrue(
            torch.allclose(
                apply_classifier_free_guidance(conditioned, unconditioned, guidance_scale=3.0),
                torch.tensor([4.0]),
            )
        )


if __name__ == "__main__":
    unittest.main()
