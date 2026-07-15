# ruff: noqa: F401,F403,F405,F811
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from tests.tests_support import (
    _assert_nonblank_phase_columns_are_labeled,
    _cuda_device,
    _dummy_one_shot_outputs,
)
@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for one-shot tests")
class OneShotCleanTargetTests(unittest.TestCase):
    """Tests for clean readable target generation and associated losses."""

    def _check_soft_mask_range(self, mask) -> None:
        self.assertTrue(mask.isfinite().all())
        self.assertGreaterEqual(mask.min().item(), 0.0)
        self.assertLessEqual(mask.max().item(), 1.0)

    def test_build_clean_readable_target_shape_and_range(self) -> None:
        import torch

        from ldm_project.one_shot_losses import _build_clean_readable_target

        teacher = torch.rand(4, 1, 32, 32) * 0.8 + 0.1
        result = _build_clean_readable_target(teacher)

        self.assertIn("clean_target", result)
        self.assertIn("clean_target_fg_ratio", result)
        clean = result["clean_target"]
        self.assertEqual(tuple(clean.shape), (4, 1, 32, 32))
        self._check_soft_mask_range(clean)
        # Foreground ratio should be in reasonable range for random data.
        fg_ratio = float(result["clean_target_fg_ratio"].item())
        self.assertGreater(fg_ratio, 0.0)
        self.assertLessEqual(fg_ratio, 1.0)

    def test_build_clean_readable_target_sharp_edges(self) -> None:
        """Clean target should preserve sharp edges from a step-function input."""
        import torch

        from ldm_project.one_shot_losses import _build_clean_readable_target

        step = torch.zeros(1, 1, 32, 32)
        step[:, :, 8:24, 8:24] = 1.0
        result = _build_clean_readable_target(step)
        clean = result["clean_target"]
        self._check_soft_mask_range(clean)
        # Most of the foreground region should be high (>0.5).
        interior = clean[:, :, 10:22, 10:22]
        self.assertGreater(interior.mean().item(), 0.7)
        # Most of the background should be low (<0.5).
        exterior = clean[:, :, :6, :6]
        self.assertLess(exterior.mean().item(), 0.3)
        # Opening shouldn't fully erode the foreground.
        fg_area = (clean > 0.5).float().sum().item()
        expected_area = 16 * 16
        self.assertGreater(fg_area, expected_area * 0.5)
        self.assertLess(fg_area, expected_area * 1.5)

    def test_build_clean_readable_target_closes_small_holes(self) -> None:
        """Morphology close should fill small holes in the foreground."""
        import torch

        from ldm_project.one_shot_losses import _build_clean_readable_target

        ring = torch.ones(1, 1, 32, 32) * 0.1
        ring[:, :, 6:26, 6:26] = 0.8
        ring[:, :, 12:20, 12:20] = 0.1  # hole in foreground
        result = _build_clean_readable_target(ring)
        clean = result["clean_target"]
        self._check_soft_mask_range(clean)
        # The hole region should be at least partially filled by closing.
        hole_region = clean[:, :, 14:18, 14:18]
        self.assertGreater(hole_region.mean().item(), 0.2)

    def test_build_clean_readable_target_opens_small_speckles(self) -> None:
        """Morphology open should suppress small background speckles."""
        import torch

        from ldm_project.one_shot_losses import _build_clean_readable_target

        speckled = torch.zeros(1, 1, 32, 32)
        speckled[:, :, 8:24, 8:24] = 0.8  # foreground block
        speckled[:, :, 2:4, 2:4] = 0.9  # small speckle in background
        speckled[:, :, 28:30, 28:30] = 0.9  # small speckle in background
        result = _build_clean_readable_target(speckled)
        clean = result["clean_target"]
        self._check_soft_mask_range(clean)
        # Speckle regions should be suppressed below 0.5.
        speckle1 = clean[:, :, 2:4, 2:4]
        speckle2 = clean[:, :, 28:30, 28:30]
        self.assertLess(speckle1.mean().item(), 0.5)
        self.assertLess(speckle2.mean().item(), 0.5)
        # Main foreground block should survive.
        main_block = clean[:, :, 10:22, 10:22]
        self.assertGreater(main_block.mean().item(), 0.5)

    def test_clean_losses_backpropagate_with_clean_target_enabled(self) -> None:
        """All clean-target losses should backpropagate when clean_target_enabled=True."""
        import torch

        from ldm_project.training import one_shot_loss

        target = (torch.rand(2, 1, 16, 16) > 0.3).float()
        outputs = {
            "camera_plane": torch.rand(2, 1, 16, 16, requires_grad=True),
            "camera_plane_loss": torch.rand(2, 1, 16, 16, requires_grad=True),
            "slm_amplitude": torch.rand(2, 1, 16, 16),
        }
        losses = one_shot_loss(
            outputs,
            target,
            clean_target_enabled=True,
            clean_foreground_dice_weight=0.5,
            clean_foreground_hole_weight=0.3,
            clean_background_speckle_weight=0.1,
            clean_background_hotspot_weight=0.5,
            clean_background_highfreq_weight=0.3,
        )
        self.assertIn("clean_target_fg_ratio", losses)
        self.assertGreater(float(losses["clean_target_fg_ratio"]), 0.0)
        self.assertGreater(float(losses["clean_dice_loss"]), 0.0)
        self.assertGreater(float(losses["clean_fg_hole_loss"]), 0.0)
        self.assertGreater(float(losses["clean_bg_speckle_loss"]), 0.0)
        losses["loss"].backward()
        # Camera plane should have received gradients from clean losses.
        self.assertIsNotNone(outputs["camera_plane"].grad)
        self.assertGreater(outputs["camera_plane"].grad.abs().sum().item(), 0.0)

    def test_clean_losses_noop_when_clean_target_disabled(self) -> None:
        """clean_target_enabled=False should produce zero clean losses."""
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.rand(2, 1, 16, 16)
        outputs = {
            "camera_plane": torch.rand(2, 1, 16, 16, requires_grad=True),
            "camera_plane_loss": torch.rand(2, 1, 16, 16, requires_grad=True),
            "slm_amplitude": torch.rand(2, 1, 16, 16),
        }
        losses = one_shot_loss(
            outputs,
            target,
            clean_target_enabled=False,
        )
        self.assertEqual(float(losses["clean_dice_loss"]), 0.0)
        self.assertEqual(float(losses["clean_fg_hole_loss"]), 0.0)
        self.assertEqual(float(losses["clean_bg_speckle_loss"]), 0.0)
        self.assertEqual(float(losses["clean_bg_hotspot_loss"]), 0.0)
        self.assertEqual(float(losses["clean_bg_highfreq_loss"]), 0.0)
        self.assertIn("clean_target_fg_ratio", losses)
        self.assertEqual(float(losses["clean_target_fg_ratio"]), 0.0)

    def test_encoder_input_topology_loss_defaults_to_noop(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.rand(2, 1, 16, 16)
        outputs = {
            "camera_plane": torch.rand(2, 1, 16, 16, requires_grad=True),
            "slm_amplitude": torch.rand(2, 1, 16, 16),
        }

        losses = one_shot_loss(outputs, target)

        self.assertEqual(float(losses["encoder_input_topology_loss"]), 0.0)
        self.assertEqual(float(losses["encoder_input_topology_dice"]), 0.0)
        self.assertEqual(float(losses["encoder_input_topology_mse"]), 0.0)

    def test_encoder_input_topology_loss_backpropagates_when_enabled(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 4:12, 6:10] = 1.0
        encoder_input = torch.zeros(2, 1, 16, 16, requires_grad=True)
        encoder_grid = torch.zeros(2, 1, 8, 8, requires_grad=True)
        with torch.no_grad():
            encoder_input[:, :, 2:10, 2:6] = 0.8
            encoder_grid[:, :, 1:5, 1:4] = 0.8
        outputs = {
            "camera_plane": torch.rand(2, 1, 16, 16, requires_grad=True),
            "slm_amplitude": torch.rand(2, 1, 16, 16),
            "encoder_input_amplitude": encoder_input,
            "encoder_input_grid_amplitude": encoder_grid,
        }

        losses = one_shot_loss(
            outputs,
            target,
            encoder_input_topology_weight=0.5,
            encoder_input_topology_dice_weight=1.0,
            encoder_input_topology_mse_weight=0.5,
        )

        self.assertGreater(float(losses["encoder_input_topology_loss"]), 0.0)
        self.assertGreaterEqual(float(losses["encoder_input_topology_dice"]), 0.0)
        self.assertLessEqual(float(losses["encoder_input_topology_dice"]), 1.0)
        self.assertGreater(float(losses["encoder_input_topology_mse"]), 0.0)
        losses["loss"].backward()
        self.assertIsNotNone(encoder_input.grad)
        self.assertIsNotNone(encoder_grid.grad)
        self.assertTrue(torch.isfinite(encoder_input.grad).all().item())
        self.assertTrue(torch.isfinite(encoder_grid.grad).all().item())
        self.assertGreater(encoder_input.grad.abs().sum().item(), 0.0)
        self.assertGreater(encoder_grid.grad.abs().sum().item(), 0.0)

    def test_encoder_input_topology_loss_fails_without_encoder_planes(self) -> None:
        import torch

        from ldm_project.training import one_shot_loss

        target = torch.rand(2, 1, 16, 16)
        outputs = {
            "camera_plane": torch.rand(2, 1, 16, 16),
            "slm_amplitude": torch.rand(2, 1, 16, 16),
        }

        with self.assertRaisesRegex(ValueError, "encoder_input_topology_weight requires"):
            one_shot_loss(outputs, target, encoder_input_topology_weight=0.1)

    def test_display_speckle_loss_targets_background(self) -> None:
        import torch

        from ldm_project.one_shot_losses import _student_display_speckle_metrics

        camera = torch.zeros(1, 1, 24, 24, requires_grad=True)
        with torch.no_grad():
            camera[:, :, 8:16, 8:16] = 0.8
            camera[:, :, :6, :6] = 0.4
        foreground = torch.zeros_like(camera)
        foreground[:, :, 8:16, 8:16] = 1.0
        background = 1.0 - foreground

        metrics = _student_display_speckle_metrics(camera, foreground, background)
        self.assertGreater(float(metrics["display_speckle_base"]), 0.0)
        metrics["display_speckle_base"].backward()

        self.assertIsNotNone(camera.grad)
        background_grad = (camera.grad.abs() * background).sum().item()
        foreground_core_grad = camera.grad[:, :, 10:14, 10:14].abs().sum().item()
        self.assertGreater(background_grad, 0.0)
        self.assertLess(foreground_core_grad, background_grad)

    def test_strict_readability_treats_clean_target_metrics_as_diagnostics(self) -> None:
        from ldm_project.one_shot_quality import _one_shot_distill_strict_readability_score

        good_metrics = {
            "paired_corr": 0.80,
            "paired_dice": 0.75,
            "camera_high_saturation_ratio": 0.01,
            "loss": 0.35,
            "camera_recon_loss": 0.12,
            "camera_foreground_dice": 0.50,
            "camera_saturation_ratio": 0.06,
            "foreground_hole_ratio": 0.05,
            "background_speckle_ratio": 0.06,
            "background_hotspot_ratio": 0.10,
            "background_highfreq_energy": 0.001,
            "readable_contrast": 4.6,
            "slm_tv": 0.35,
            "student_label_correct_ratio": 0.85,
            "student_label_margin_mean": 0.04,
            "clean_foreground_dice": 0.60,
            "clean_fg_hole_ratio": 0.30,
            "clean_target_fg_ratio": 0.33,
        }
        good_score = _one_shot_distill_strict_readability_score(good_metrics)

        poor_metrics = dict(good_metrics)
        poor_metrics["clean_foreground_dice"] = 0.30
        poor_metrics["clean_fg_hole_ratio"] = 0.60
        poor_metrics["clean_target_fg_ratio"] = 0.55
        poor_score = _one_shot_distill_strict_readability_score(poor_metrics)

        self.assertEqual(poor_score, good_score)

    def test_clean_target_fg_ratio_recorded_in_history_keys(self) -> None:
        from ldm_project.one_shot_training_utils import ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS

        self.assertIn("clean_target_fg_ratio", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
        self.assertIn("aux_detail_gate_fg_ratio", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
        self.assertIn("aux_detail_active_fg_ratio", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
        self.assertIn("residual_background_mask_ratio", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
        self.assertIn("encoder_input_topology_loss", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
        self.assertIn("encoder_input_topology_dice", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
        self.assertIn("encoder_input_topology_mse", ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS)
