from __future__ import annotations

import unittest


class TeacherTrainingUtilsTests(unittest.TestCase):
    def test_history_schema_locked_extra_keys_not_leaked(self) -> None:
        import torch

        from ldm_project.teacher_training_utils import (
            FIRST_STAGE_HISTORY_LOSS_KEYS,
            _record_first_stage_train_step,
        )

        history_records: dict[str, list[dict[str, object]]] = {}

        class _MockRunManager:
            def append_history(self, name: str, entry: dict[str, object]) -> None:
                history_records.setdefault(name, []).append(entry)

        mock = _MockRunManager()
        losses: dict[str, object] = {
            key: torch.tensor(0.0) for key in FIRST_STAGE_HISTORY_LOSS_KEYS
        }
        losses["future_extra_metric"] = torch.tensor(1.0)

        _record_first_stage_train_step(
            mock,
            epoch_number=1,
            step_record={"global_step": 0, "stage_step": 0},
            losses=losses,
            image_fusion_weight=0.5,
        )

        self.assertIn("first_stage", history_records)
        entry = history_records["first_stage"][0]

        for key in FIRST_STAGE_HISTORY_LOSS_KEYS:
            self.assertIn(key, entry, f"first_stage history missing key: {key}")
        self.assertIn("epoch", entry)
        self.assertIn("global_step", entry)
        self.assertIn("stage_step", entry)
        self.assertIn("image_fusion_weight", entry)
        self.assertNotIn("future_extra_metric", entry)

    def test_history_keys_are_subset_of_first_stage_loss_keys(self) -> None:
        import torch

        from ldm_project.config import FirstStageConfig, PhysicsConfig
        from ldm_project.first_stage import OpticalPathVAE
        from ldm_project.teacher_training_utils import FIRST_STAGE_HISTORY_LOSS_KEYS
        from ldm_project.training import first_stage_loss

        physics = PhysicsConfig()
        first_stage_cfg = FirstStageConfig()
        model = OpticalPathVAE(physics, first_stage_cfg)
        input_batch = torch.rand(2, 1, 200, 200)

        with torch.no_grad():
            outputs = model(input_batch)

        losses = first_stage_loss(
            outputs,
            input_batch,
            global_step=0,
            total_steps=100,
            target_beta=0.001,
            warmup_ratio=0.1,
        )

        missing = [key for key in FIRST_STAGE_HISTORY_LOSS_KEYS if key not in losses]
        self.assertEqual(
            missing,
            [],
            f"FIRST_STAGE_HISTORY_LOSS_KEYS contains keys not in first_stage_loss() output: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
