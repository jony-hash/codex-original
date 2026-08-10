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
@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for one-shot tests")
class TestStudentLosses(unittest.TestCase):
        def _minmax_norm(tensor):
            mn = tensor.amin(dim=(-2, -1), keepdim=True)
            mx = tensor.amax(dim=(-2, -1), keepdim=True)
            return (tensor - mn) / (mx - mn).clamp_min(1e-6)

        def test_one_shot_loss_penalizes_same_class_code_collapse(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.tensor(
                [
                    [[[0.0, 1.0], [0.0, 1.0]]],
                    [[[1.0, 0.0], [1.0, 0.0]]],
                ],
                dtype=torch.float32,
            )
            collapsed = torch.full_like(target, 0.5, requires_grad=True)
            outputs = {
                "camera_plane": collapsed,
                "slm_amplitude": torch.ones_like(target),
            }
            labels = torch.tensor([0, 0], dtype=torch.long)

            losses = one_shot_loss(
                outputs,
                target,
                class_labels=labels,
                code_diversity_weight=0.1,
                sample_code_diversity_weight=0.2,
                teacher_pairwise_diversity_weight=0.02,
                diversity_min_teacher_mse=1.0e-4,
            )

            self.assertGreater(losses["code_diversity_penalty"].item(), 0.0)
            self.assertGreater(losses["sample_code_diversity_penalty"].item(), 0.0)
            self.assertGreater(losses["teacher_pairwise_mse"].item(), losses["student_pairwise_mse"].item())
            self.assertEqual(losses["diversity_ratio"].item(), 0.0)
            self.assertEqual(losses["cross_class_pair_count"].item(), 0.0)
            losses["loss"].backward()
            self.assertIsNotNone(collapsed.grad)

        def test_one_shot_loss_supports_weak_class_shape_and_edge_terms(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.ones(2, 1, 4, 4, dtype=torch.float32)
            prediction = target.clone()
            prediction[0] = 0.0
            prediction.requires_grad_(True)
            outputs = {
                "camera_plane": prediction,
                "slm_amplitude": torch.linspace(0.0, 1.0, steps=32, dtype=torch.float32).reshape(2, 1, 4, 4),
            }
            labels = torch.tensor([3, 0], dtype=torch.long)

            unweighted = one_shot_loss(outputs, target, class_labels=labels)
            weighted = one_shot_loss(
                outputs,
                target,
                class_labels=labels,
                class_recon_weight_labels=(3,),
                class_recon_weight=3.0,
                shape_loss_weight=0.1,
                edge_loss_weight=0.01,
            )

            self.assertGreater(weighted["camera_recon_loss"].item(), unweighted["camera_recon_loss"].item())
            self.assertGreater(weighted["shape_loss"].item(), 0.0)
            self.assertGreaterEqual(weighted["edge_loss"].item(), 0.0)
            self.assertGreater(weighted["class_weight_mean"].item(), 1.0)
            weighted["loss"].backward()
            self.assertIsNotNone(prediction.grad)

        def test_one_shot_loss_supports_aux_detail_and_residual_penalties(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.zeros(2, 1, 4, 4, dtype=torch.float32)
            target[:, :, 1:3, 1:3] = 1.0
            aux_detail = torch.zeros_like(target)
            aux_detail[:, :, :, 1:3] = 1.0
            prediction = target.clone().requires_grad_(True)
            residual = torch.full_like(target, 0.25, requires_grad=True)
            outputs = {
                "camera_plane": prediction,
                "slm_amplitude": torch.ones_like(target),
                "generator_residual_map": residual,
                "generator_base_command": torch.zeros_like(target),
                "generator_residual_abs_mean": residual.detach().abs().flatten(1).mean(dim=1),
                "generator_residual_bg_abs_mean": residual.detach().abs().flatten(1).mean(dim=1),
            }

            losses = one_shot_loss(
                outputs,
                target,
                aux_detail_target=aux_detail,
                aux_detail_shape_weight=0.1,
                aux_detail_edge_weight=0.1,
                residual_l1_weight=0.2,
                residual_background_weight=0.3,
            )

            self.assertGreater(losses["aux_detail_shape_loss"].item(), 0.0)
            self.assertGreater(losses["aux_detail_edge_loss"].item(), 0.0)
            self.assertGreater(losses["residual_l1_loss"].item(), 0.0)
            self.assertGreater(losses["residual_background_loss"].item(), 0.0)
            self.assertGreater(losses["student_base_mse"].item(), 0.0)
            self.assertGreater(losses["student_base_foreground_mse"].item(), 0.0)
            self.assertEqual(losses["student_base_background_mse"].item(), 0.0)
            losses["loss"].backward()
            self.assertIsNotNone(prediction.grad)
            self.assertIsNotNone(residual.grad)

        def test_one_shot_loss_aux_detail_foreground_gate_blocks_off_template_detail(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.zeros(1, 1, 5, 5, dtype=torch.float32)
            target[:, :, 1:3, 1:3] = 1.0
            aux_detail = torch.zeros_like(target)
            aux_detail[:, :, 4:, 4:] = 1.0
            prediction = target.clone().requires_grad_(True)
            outputs = {
                "camera_plane": prediction,
                "slm_amplitude": torch.ones_like(target),
            }

            losses = one_shot_loss(
                outputs,
                target,
                aux_detail_target=aux_detail,
                aux_detail_shape_weight=0.1,
                aux_detail_edge_weight=0.1,
                aux_detail_foreground_gate_dilation=1,
            )

            self.assertGreater(losses["aux_detail_gate_fg_ratio"].item(), 0.0)
            self.assertEqual(losses["aux_detail_active_fg_ratio"].item(), 0.0)
            self.assertEqual(losses["aux_detail_shape_loss"].item(), 0.0)
            self.assertEqual(losses["aux_detail_edge_loss"].item(), 0.0)

        def test_one_shot_loss_residual_background_uses_foreground_gate_complement(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.zeros(1, 1, 5, 5, dtype=torch.float32)
            target[:, :, 2, 2] = 1.0
            residual = torch.full_like(target, 2.0, requires_grad=True)
            with torch.no_grad():
                residual[:, :, 1:4, 1:4] = 1.0
            outputs = {
                "camera_plane": target.clone(),
                "slm_amplitude": torch.ones_like(target),
                "generator_residual_map": residual,
            }

            losses = one_shot_loss(
                outputs,
                target,
                residual_background_weight=0.5,
                aux_detail_foreground_gate_dilation=3,
            )

            self.assertAlmostEqual(losses["aux_detail_gate_fg_ratio"].item(), 9.0 / 25.0, places=6)
            self.assertAlmostEqual(losses["residual_background_mask_ratio"].item(), 16.0 / 25.0, places=6)
            self.assertAlmostEqual(losses["residual_background_penalty"].item(), 2.0, places=6)
            self.assertAlmostEqual(losses["residual_background_loss"].item(), 1.0, places=6)
            losses["loss"].backward()
            self.assertIsNotNone(residual.grad)

        def test_one_shot_loss_student_base_metrics_default_zero_without_base_command(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.ones(1, 1, 4, 4, dtype=torch.float32)
            losses = one_shot_loss(
                {
                    "camera_plane": torch.zeros_like(target),
                    "slm_amplitude": torch.ones_like(target),
                },
                target,
            )

            self.assertEqual(losses["student_base_mse"].item(), 0.0)
            self.assertEqual(losses["student_base_foreground_mse"].item(), 0.0)
            self.assertEqual(losses["student_base_background_mse"].item(), 0.0)

        def test_one_shot_readability_loss_defaults_off_and_activates(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.rand(2, 1, 16, 16)
            outputs = {
                "camera_plane": torch.rand(2, 1, 16, 16),
                "slm_amplitude": torch.ones(2, 1, 16, 16),
            }

            off = one_shot_loss(outputs, target)
            on = one_shot_loss(outputs, target, camera_readability_weight=1.0)

            self.assertEqual(off["camera_readability_loss"].item(), 0.0)
            self.assertEqual(off["camera_readability_penalty"].item(), 0.0)
            self.assertGreater(on["camera_readability_loss"].item(), 0.0)
            self.assertGreater(on["camera_readability_penalty"].item(), 0.0)
            self.assertLessEqual(on["camera_readability_loss"].item(), on["loss"].item())

        def test_one_shot_loss_activates_background_speckle_and_contrast_terms(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.zeros(2, 1, 8, 8, dtype=torch.float32)
            target[:, :, 2:6, 2:6] = 1.0
            prediction = target * 0.02
            prediction[:, :, 0, 0] = 0.5
            prediction.requires_grad_(True)
            outputs = {
                "camera_plane": prediction,
                "slm_amplitude": torch.ones(2, 1, 8, 8),
            }

            losses = one_shot_loss(
                outputs,
                target,
                camera_background_speckle_weight=0.1,
                camera_background_hotspot_weight=0.1,
                camera_background_hotspot_threshold=0.08,
                camera_background_highfreq_weight=0.1,
                camera_readable_contrast_weight=0.1,
                camera_readable_contrast_target=4.5,
            )

            self.assertGreater(losses["camera_background_speckle_loss"].item(), 0.0)
            self.assertGreater(losses["background_speckle_ratio"].item(), 0.0)
            self.assertGreater(losses["camera_background_hotspot_loss"].item(), 0.0)
            self.assertGreater(losses["background_hotspot_ratio"].item(), 0.0)
            self.assertGreater(losses["camera_background_highfreq_loss"].item(), 0.0)
            self.assertGreater(losses["background_highfreq_energy"].item(), 0.0)
            self.assertGreater(losses["camera_readable_contrast_loss"].item(), 0.0)
            self.assertLess(losses["readable_contrast"].item(), 4.5)
            losses["loss"].backward()
            self.assertIsNotNone(prediction.grad)

        def test_one_shot_loss_reports_readability_diagnostics_when_weights_are_zero(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.zeros(2, 1, 8, 8, dtype=torch.float32)
            target[:, :, 2:6, 2:6] = 1.0
            prediction = target * 0.02
            prediction[:, :, 0, 0] = 0.5
            outputs = {
                "camera_plane": prediction,
                "slm_amplitude": torch.ones(2, 1, 8, 8),
            }

            losses = one_shot_loss(
                outputs,
                target,
                camera_background_hotspot_threshold=0.08,
                camera_readable_contrast_target=4.5,
            )

            self.assertEqual(losses["camera_artifact_loss"].item(), 0.0)
            self.assertEqual(losses["camera_background_speckle_loss"].item(), 0.0)
            self.assertEqual(losses["camera_background_hotspot_loss"].item(), 0.0)
            self.assertEqual(losses["camera_background_highfreq_loss"].item(), 0.0)
            self.assertEqual(losses["camera_readable_contrast_loss"].item(), 0.0)
            self.assertGreater(losses["foreground_hole_ratio"].item(), 0.0)
            self.assertGreater(losses["background_speckle_ratio"].item(), 0.0)
            self.assertGreater(losses["background_hotspot_ratio"].item(), 0.0)
            self.assertGreater(losses["background_highfreq_energy"].item(), 0.0)
            self.assertLess(losses["readable_contrast"].item(), 4.5)

        def test_save_epoch_distill_checkpoint_persists_epoch_records_to_meta(self) -> None:
            import torch

            from ldm_project.config import TrainingConfig
            from ldm_project.one_shot_distill_training import _save_epoch_distill_checkpoint

            class DummyModel:
                def state_dict(self):
                    return {"weight": torch.tensor(1.0)}

            class RecordingRunManager:
                def __init__(self, run_dir: Path) -> None:
                    self.run_dir = run_dir
                    self.checkpoints_dir = run_dir / "checkpoints"
                    self.meta_payload = None

                def save_checkpoint(self, name, payload):
                    self.saved_checkpoint = (name, dict(payload))
                    return self.checkpoints_dir / name

                def write_meta(self, meta):
                    self.meta_payload = dict(meta)

            with tempfile.TemporaryDirectory() as tmpdir:
                run_manager = RecordingRunManager(Path(tmpdir))
                train_cfg = TrainingConfig(one_shot_distill_epoch_checkpoint_interval=2)
                run_meta: dict[str, object] = {}

                _save_epoch_distill_checkpoint(
                    run_manager,
                    run_meta,
                    one_shot_model=DummyModel(),
                    train_cfg=train_cfg,
                    physics={},
                    epoch_number=2,
                    one_shot_model_version="test",
                    distill_dataset="distill.pt",
                    validation_metrics={"diversity_ratio": 0.7, "eval_corr": 0.8, "eval_dice": 0.9, "best_loss": 0.1},
                )

                self.assertIsNotNone(run_manager.meta_payload)
                self.assertEqual(run_manager.meta_payload["epoch_checkpoints"][0]["epoch"], 2)
                self.assertTrue(
                    str(run_manager.meta_payload["epoch_checkpoints"][0]["path"]).endswith(
                        "one_shot_distill_epoch_0002.pt"
                    )
                )

        def test_one_shot_loss_student_label_margin_prefers_matching_prototype(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            proto_zero = torch.zeros(1, 1, 4, 4)
            proto_zero[:, :, :2, :2] = 1.0
            proto_one = torch.zeros(1, 1, 4, 4)
            proto_one[:, :, 2:, 2:] = 1.0
            prototypes = torch.cat([proto_zero, proto_one], dim=0)
            labels = torch.tensor([0, 1], dtype=torch.long)
            target = prototypes.index_select(0, labels)
            slm = torch.ones(2, 1, 4, 4)

            matched = one_shot_loss(
                {"camera_plane": target.clone(), "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_weight=1.0,
                student_label_margin_target=0.05,
                student_label_margin_prototypes=prototypes,
            )
            swapped = one_shot_loss(
                {"camera_plane": target.flip(0).clone(), "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_weight=1.0,
                student_label_margin_target=0.05,
                student_label_margin_prototypes=prototypes,
            )

            self.assertGreater(matched["student_label_correct_ratio"].item(), swapped["student_label_correct_ratio"].item())
            self.assertLess(matched["student_label_margin_loss"].item(), swapped["student_label_margin_loss"].item())
            self.assertGreater(matched["student_label_margin_mean"].item(), swapped["student_label_margin_mean"].item())

        def test_one_shot_loss_student_prototype_ce_prefers_matching_prototype(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            proto_zero = torch.zeros(1, 1, 4, 4)
            proto_zero[:, :, :2, :2] = 1.0
            proto_one = torch.zeros(1, 1, 4, 4)
            proto_one[:, :, 2:, 2:] = 1.0
            prototypes = torch.cat([proto_zero, proto_one], dim=0)
            labels = torch.tensor([0, 1], dtype=torch.long)
            target = prototypes.index_select(0, labels)
            slm = torch.ones(2, 1, 4, 4)

            matched = one_shot_loss(
                {"camera_plane": target.clone(), "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_prototype_ce_weight=1.0,
                student_prototype_ce_labels=(0, 1),
                student_prototype_ce_temperature=0.1,
            )
            swapped = one_shot_loss(
                {"camera_plane": target.flip(0).clone(), "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_prototype_ce_weight=1.0,
                student_prototype_ce_labels=(0, 1),
                student_prototype_ce_temperature=0.1,
            )

            self.assertLess(matched["student_prototype_ce_loss"].item(), swapped["student_prototype_ce_loss"].item())
            self.assertGreater(
                matched["student_prototype_ce_correct_ratio"].item(),
                swapped["student_prototype_ce_correct_ratio"].item(),
            )

        def test_one_shot_loss_student_negative_label_margin_targets_specific_pair(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            prototypes = torch.zeros(10, 1, 4, 4)
            prototypes[3, :, :2, :2] = 1.0
            prototypes[5, :, 2:, 2:] = 1.0
            labels = torch.tensor([3], dtype=torch.long)
            target = prototypes.index_select(0, labels)
            slm = torch.ones(1, 1, 4, 4)

            matched = one_shot_loss(
                {"camera_plane": target.clone(), "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_negative_label_margin_weight=1.0,
                student_negative_label_margin_config=((3, 5, 0.05),),
            )
            confused = one_shot_loss(
                {"camera_plane": prototypes[5:6].clone(), "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_negative_label_margin_weight=1.0,
                student_negative_label_margin_config=((3, 5, 0.05),),
            )
            inactive = one_shot_loss(
                {"camera_plane": prototypes[5:6].clone(), "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_negative_label_margin_weight=1.0,
                student_negative_label_margin_config=((7, 9, 0.05),),
            )

            self.assertLess(
                matched["student_negative_label_margin_loss"].item(),
                confused["student_negative_label_margin_loss"].item(),
            )
            self.assertGreater(confused["student_negative_label_margin_count"].item(), 0.0)
            self.assertEqual(inactive["student_negative_label_margin_count"].item(), 0.0)
            self.assertEqual(inactive["student_negative_label_margin_loss"].item(), 0.0)

        def test_one_shot_loss_student_roi_prototype_ce_prefers_foreground_match(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            prototypes = torch.zeros(2, 1, 4, 4)
            prototypes[0, :, :2, :2] = torch.tensor([[1.0, 0.8], [0.6, 0.4]])
            prototypes[1, :, :2, :2] = torch.tensor([[0.4, 0.6], [0.8, 1.0]])
            labels = torch.tensor([0, 1], dtype=torch.long)
            target = prototypes.index_select(0, labels)
            slm = torch.ones(2, 1, 4, 4)
            background_noise = torch.zeros_like(target)
            background_noise[:, :, 2:, 2:] = 20.0

            matched = one_shot_loss(
                {"camera_plane": target + background_noise, "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_roi_prototype_ce_weight=1.0,
                student_roi_prototype_ce_labels=(0, 1),
                student_roi_prototype_ce_temperature=0.1,
                student_roi_foreground_threshold=0.5,
            )
            swapped = one_shot_loss(
                {"camera_plane": target.flip(0) + background_noise, "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_roi_prototype_ce_weight=1.0,
                student_roi_prototype_ce_labels=(0, 1),
                student_roi_prototype_ce_temperature=0.1,
                student_roi_foreground_threshold=0.5,
            )

            self.assertLess(
                matched["student_roi_prototype_ce_loss"].item(),
                swapped["student_roi_prototype_ce_loss"].item(),
            )
            self.assertGreater(
                matched["student_roi_prototype_ce_correct_ratio"].item(),
                swapped["student_roi_prototype_ce_correct_ratio"].item(),
            )

        def test_one_shot_loss_student_roi_negative_label_margin_targets_specific_pair(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            prototypes = torch.zeros(10, 1, 4, 4)
            prototypes[3, :, :2, :2] = torch.tensor([[1.0, 0.8], [0.6, 0.4]])
            prototypes[5, :, :2, :2] = torch.tensor([[0.4, 0.6], [0.8, 1.0]])
            labels = torch.tensor([3], dtype=torch.long)
            target = prototypes.index_select(0, labels)
            slm = torch.ones(1, 1, 4, 4)

            matched = one_shot_loss(
                {"camera_plane": target.clone(), "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_roi_negative_label_margin_weight=1.0,
                student_roi_negative_label_margin_config=((3, 5, 0.05),),
                student_roi_foreground_threshold=0.5,
            )
            confused = one_shot_loss(
                {"camera_plane": prototypes[5:6].clone(), "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_roi_negative_label_margin_weight=1.0,
                student_roi_negative_label_margin_config=((3, 5, 0.05),),
                student_roi_foreground_threshold=0.5,
            )
            inactive = one_shot_loss(
                {"camera_plane": prototypes[5:6].clone(), "slm_amplitude": slm},
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_roi_negative_label_margin_weight=1.0,
                student_roi_negative_label_margin_config=((7, 9, 0.05),),
                student_roi_foreground_threshold=0.5,
            )

            self.assertLess(
                matched["student_roi_negative_label_margin_loss"].item(),
                confused["student_roi_negative_label_margin_loss"].item(),
            )
            self.assertGreater(confused["student_roi_negative_label_margin_count"].item(), 0.0)
            self.assertEqual(inactive["student_roi_negative_label_margin_count"].item(), 0.0)
            self.assertEqual(inactive["student_roi_negative_label_margin_loss"].item(), 0.0)

        def test_one_shot_loss_student_prototype_objectives_default_off(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            prototypes = torch.rand(2, 1, 4, 4)
            labels = torch.tensor([0, 1], dtype=torch.long)
            target = torch.rand(2, 1, 4, 4)
            outputs = {"camera_plane": torch.rand(2, 1, 4, 4), "slm_amplitude": torch.ones(2, 1, 4, 4)}

            base = one_shot_loss(outputs, target, class_labels=labels)
            disabled = one_shot_loss(
                outputs,
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_prototype_ce_labels=(0, 1),
                student_negative_label_margin_config=((0, 1, 0.05),),
                student_roi_prototype_ce_labels=(0, 1),
                student_roi_negative_label_margin_config=((0, 1, 0.05),),
            )

            self.assertAlmostEqual(base["loss"].item(), disabled["loss"].item(), places=8)
            self.assertEqual(disabled["student_prototype_ce_loss"].item(), 0.0)
            self.assertEqual(disabled["student_negative_label_margin_loss"].item(), 0.0)
            self.assertEqual(disabled["student_roi_prototype_ce_loss"].item(), 0.0)
            self.assertEqual(disabled["student_roi_negative_label_margin_loss"].item(), 0.0)
            self.assertEqual(disabled["aux_semantic_loss"].item(), 0.0)
            self.assertEqual(disabled["aux_semantic_prototype_ce_loss"].item(), 0.0)
            self.assertEqual(disabled["aux_semantic_negative_label_margin_loss"].item(), 0.0)
            self.assertEqual(disabled["aux_topology_loss"].item(), 0.0)
            self.assertEqual(disabled["aux_topology_dice"].item(), 0.0)
            self.assertEqual(disabled["aux_topology_mse"].item(), 0.0)

        def test_one_shot_loss_aux_semantic_plane_resizes_and_backpropagates(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            prototypes = torch.zeros(2, 1, 8, 8)
            prototypes[0, :, :4, :4] = 1.0
            prototypes[1, :, 4:, 4:] = 1.0
            labels = torch.tensor([0, 1], dtype=torch.long)
            target = prototypes.index_select(0, labels)
            aux_plane = torch.nn.Parameter(target[:, :, ::2, ::2].clone() * 0.8 + 0.1)
            outputs = {
                "camera_plane": torch.rand(2, 1, 8, 8),
                "slm_amplitude": torch.ones(2, 1, 8, 8),
                "chip_intensity_primary": aux_plane,
            }

            losses = one_shot_loss(
                outputs,
                target,
                class_labels=labels,
                student_label_margin_prototypes=prototypes,
                student_prototype_ce_labels=(0, 1),
                student_negative_label_margin_config=((0, 1, 0.05),),
                student_prototype_ce_temperature=0.1,
                aux_semantic_plane_key="chip_intensity_primary",
                aux_semantic_weight=1.0,
                aux_semantic_prototype_ce_weight=1.0,
                aux_semantic_negative_label_margin_weight=0.5,
            )

            self.assertGreater(float(losses["aux_semantic_loss"]), 0.0)
            self.assertGreater(float(losses["aux_semantic_prototype_ce_count"]), 0.0)
            self.assertGreater(float(losses["aux_semantic_negative_label_margin_count"]), 0.0)
            losses["loss"].backward()
            self.assertIsNotNone(aux_plane.grad)
            self.assertTrue(torch.isfinite(aux_plane.grad).all().item())
            self.assertGreater(float(aux_plane.grad.abs().sum()), 0.0)

        def test_one_shot_loss_aux_semantic_plane_fails_loudly_when_missing(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.rand(2, 1, 8, 8)
            labels = torch.tensor([0, 1], dtype=torch.long)
            prototypes = torch.rand(2, 1, 8, 8)
            outputs = {
                "camera_plane": torch.rand(2, 1, 8, 8),
                "slm_amplitude": torch.ones(2, 1, 8, 8),
            }

            with self.assertRaisesRegex(ValueError, "aux_semantic_plane_key"):
                one_shot_loss(
                    outputs,
                    target,
                    class_labels=labels,
                    student_label_margin_prototypes=prototypes,
                    aux_semantic_plane_key="chip_intensity_primary",
                    aux_semantic_weight=1.0,
                    aux_semantic_prototype_ce_weight=1.0,
                )

        def test_one_shot_loss_aux_topology_plane_resizes_and_backpropagates(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.zeros(2, 1, 8, 8)
            target[:, :, 2:6, 2:6] = 1.0
            for plane_key in ("decoder_output_amplitude", "target_code_amplitude", "decoder_input_amplitude"):
                with self.subTest(aux_topology_plane_key=plane_key):
                    aux_plane = torch.nn.Parameter(target[:, :, ::2, ::2].clone() * 0.5 + 0.2)
                    outputs = {
                        "camera_plane": torch.rand(2, 1, 8, 8),
                        "slm_amplitude": torch.ones(2, 1, 8, 8),
                        plane_key: aux_plane,
                    }

                    losses = one_shot_loss(
                        outputs,
                        target,
                        aux_topology_plane_key=plane_key,
                        aux_topology_weight=1.0,
                        aux_topology_dice_weight=1.0,
                        aux_topology_mse_weight=0.5,
                    )

                    self.assertGreater(float(losses["aux_topology_loss"]), 0.0)
                    self.assertGreaterEqual(float(losses["aux_topology_dice"]), 0.0)
                    self.assertLessEqual(float(losses["aux_topology_dice"]), 1.0)
                    self.assertGreater(float(losses["aux_topology_mse"]), 0.0)
                    losses["loss"].backward()
                    self.assertIsNotNone(aux_plane.grad)
                    self.assertTrue(torch.isfinite(aux_plane.grad).all().item())
                    self.assertGreater(float(aux_plane.grad.abs().sum()), 0.0)

        def test_one_shot_loss_aux_topology_plane_fails_loudly_when_missing(self) -> None:
            import torch

            from ldm_project.training import one_shot_loss

            target = torch.rand(2, 1, 8, 8)
            outputs = {
                "camera_plane": torch.rand(2, 1, 8, 8),
                "slm_amplitude": torch.ones(2, 1, 8, 8),
            }

            with self.assertRaisesRegex(ValueError, "aux_topology_plane_key"):
                one_shot_loss(
                    outputs,
                    target,
                    aux_topology_plane_key="decoder_output_amplitude",
                    aux_topology_weight=1.0,
                )

        def test_one_shot_distill_eval_readability_metrics_present_and_bounded(self) -> None:
            import torch

            from ldm_project.one_shot_eval_metrics import compute_one_shot_distill_eval_metrics

            teacher = torch.tensor(
                [
                    [[[0.0, 1.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 0.0]]],
                    [[[1.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 1.0]]],
                ],
                dtype=torch.float32,
            )
            student = teacher.clone()
            student[0, 0, 1, 0] = 0.05  # fake speckle in background
            student[2, 0, 0, 1] = 0.05
            slm = torch.ones(4, 1, 2, 2, dtype=torch.float32)
            labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

            metrics = compute_one_shot_distill_eval_metrics(
                torch,
                teacher_camera=teacher,
                student_camera=student,
                slm_amplitude=slm,
                class_labels=labels,
            )

            for key in ("foreground_ratio", "background_speckle_ratio", "readable_contrast"):
                self.assertIn(key, metrics["overall"], f"missing overall.{key}")
                self.assertTrue(math.isfinite(metrics["overall"][key]))
                self.assertGreaterEqual(metrics["overall"][key], 0.0)
                self.assertIn(key, metrics["per_class"]["0"], f"missing per_class.0.{key}")
                self.assertTrue(math.isfinite(metrics["per_class"]["0"][key]))
                self.assertGreaterEqual(metrics["per_class"]["0"][key], 0.0)

            # background_speckle_ratio should be 0 when there is no background region
            no_bg_teacher = torch.ones(1, 1, 2, 2, dtype=torch.float32)
            no_bg_student = torch.ones(1, 1, 2, 2, dtype=torch.float32)
            no_bg_slm = torch.ones(1, 1, 2, 2, dtype=torch.float32)
            no_bg_labels = torch.tensor([0], dtype=torch.long)
            no_bg_metrics = compute_one_shot_distill_eval_metrics(
                torch,
                teacher_camera=no_bg_teacher,
                student_camera=no_bg_student,
                slm_amplitude=no_bg_slm,
                class_labels=no_bg_labels,
            )
            self.assertEqual(no_bg_metrics["overall"]["background_speckle_ratio"], 0.0)

        def test_direct_oracle_eval_visuals_keep_only_readable_columns_and_no_phase(self) -> None:
            from ldm_project.one_shot_eval import _direct_readable_eval_visuals
            from ldm_project.visualization_core import VISUALIZATION_INTENSITY_KEY, VISUALIZATION_PHASE_KEY

            visuals = {
                VISUALIZATION_INTENSITY_KEY: {
                    "digit": object(),
                    "teacher intensity target": object(),
                    "SLM command (amplitude control)": object(),
                    "student camera intensity": object(),
                    "intensity abs error": object(),
                    "encoded out I": object(),
                    "ols out I": object(),
                    "decoder out I": object(),
                },
                VISUALIZATION_PHASE_KEY: {
                    "encoded out phase": object(),
                    "ols out phase": object(),
                },
            }

            direct = _direct_readable_eval_visuals(visuals)

            self.assertEqual(
                tuple(direct[VISUALIZATION_INTENSITY_KEY].keys()),
                (
                    "digit",
                    "teacher intensity target",
                    "SLM command (amplitude control)",
                    "student camera intensity",
                    "intensity abs error",
                ),
            )
            self.assertEqual(direct[VISUALIZATION_PHASE_KEY], {})

        def test_student_label_audit_metrics_and_strict_acceptance(self) -> None:
            from ldm_project.one_shot_eval_metrics import (
                add_student_label_audit_metrics,
                compute_strict_student_readability_acceptance,
            )

            metrics = {
                "overall": {
                    "count": 4,
                    "paired_corr": 0.80,
                    "paired_dice": 0.73,
                    "background_speckle_ratio": 0.07,
                    "foreground_hole_ratio": 0.05,
                    "student_saturation_ratio": 0.07,
                    "student_high_saturation_ratio": 0.01,
                    "readable_contrast": 4.8,
                },
                "per_class": {
                    "0": {"paired_dice": 0.70},
                    "1": {"paired_dice": 0.69},
                },
            }
            audit_payload = {
                "label_correct_count": 4,
                "label_correct_ratio": 1.0,
                "overall": {"label_margin": {"mean": 0.06}},
                "per_class": {
                    "0": {
                        "count": 2,
                        "label_correct_count": 2,
                        "label_correct_ratio": 1.0,
                        "label_margin": {"mean": 0.05},
                    },
                    "1": {
                        "count": 2,
                        "label_correct_count": 2,
                        "label_correct_ratio": 1.0,
                        "label_margin": {"mean": 0.04},
                    },
                },
                "per_sample": [
                    {"requested_label": 0, "nearest_label": 0},
                    {"requested_label": 0, "nearest_label": 0},
                    {"requested_label": 1, "nearest_label": 1},
                    {"requested_label": 1, "nearest_label": 1},
                ],
            }

            add_student_label_audit_metrics(metrics, audit_payload)
            acceptance = compute_strict_student_readability_acceptance(metrics)

            self.assertTrue(acceptance["passed"])
            self.assertEqual(metrics["overall"]["student_label_correct_count"], 4)
            self.assertEqual(metrics["overall"]["min_class_student_label_correct_ratio"], 1.0)
            self.assertAlmostEqual(metrics["overall"]["min_class_student_label_margin_mean"], 0.04, places=6)
            self.assertEqual(metrics["per_class"]["0"]["student_nearest_label_confusion"], {"0": 2})
            self.assertEqual(
                set(acceptance["checks"]),
                {"paired_corr", "paired_dice", "student_high_saturation_ratio"},
            )

            metrics["per_class"]["1"]["student_label_correct_ratio"] = 0.5
            diagnostics_only = compute_strict_student_readability_acceptance(metrics)
            self.assertTrue(diagnostics_only["passed"])
            metrics["overall"]["paired_corr"] = 0.77
            self.assertFalse(compute_strict_student_readability_acceptance(metrics)["passed"])
