from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for strict optical-field tests")
class StrictOpticalFieldTests(unittest.TestCase):
    def test_build_views_derives_raw_amplitude_and_phase_from_real_imag(self) -> None:
        import torch

        from ldm_project.strict_optical_fields import (
            build_optical_field_record,
            build_strict_optical_field_views,
            validate_optical_field_chain,
        )

        field = torch.full((1, 1, 2, 2), complex(3.0, 4.0), dtype=torch.complex64)
        fields = {
            "slm_output": build_optical_field_record(field, upstream=None, operation="slm", phase_available=True),
            "encoder_output": build_optical_field_record(field, upstream="slm_output", operation="encoder", phase_available=True),
            "ols_input": build_optical_field_record(field, upstream="encoder_output", operation="ols_input", phase_available=True),
            "ols_output": build_optical_field_record(field, upstream="ols_input", operation="ols", phase_available=True),
            "decoder_input": build_optical_field_record(field, upstream="ols_output", operation="decoder_input", phase_available=True),
            "chip": build_optical_field_record(field, upstream="decoder_input", operation="decoder", phase_available=True),
        }

        chain = validate_optical_field_chain(
            fields,
            front_path_model="relay_coherent_psf",
            target_code_active=False,
            bridge_active=False,
        )
        views = build_strict_optical_field_views(fields, chain)

        self.assertEqual(chain["status"], "valid")
        self.assertTrue(torch.equal(views["amplitude"]["chip |E|"], torch.full((1, 1, 2, 2), 5.0)))
        self.assertTrue(torch.allclose(views["phase"]["chip phase [rad]"], torch.full((1, 1, 2, 2), math.atan2(4.0, 3.0))))
        self.assertEqual(views["amplitude_vmax"], 5.0)

    def test_target_code_intervention_rejects_full_chain_and_limits_views_to_ols_output(self) -> None:
        import torch

        from ldm_project.strict_optical_fields import (
            build_optical_field_record,
            build_strict_optical_field_views,
            validate_optical_field_chain,
        )

        field = torch.ones((1, 1, 2, 2), dtype=torch.complex64)
        fields = {
            "slm_output": build_optical_field_record(field, upstream=None, operation="slm", phase_available=True),
            "encoder_output": build_optical_field_record(field, upstream="slm_output", operation="encoder", phase_available=True),
            "ols_input": build_optical_field_record(field, upstream="encoder_output", operation="ols_input", phase_available=True),
            "ols_output": build_optical_field_record(field, upstream="ols_input", operation="ols", phase_available=True),
            "decoder_input": build_optical_field_record(field, upstream="ols_output", operation="target_code", phase_available=False, source_kind="target_code_intervention"),
            "chip": build_optical_field_record(field, upstream="decoder_input", operation="decoder", phase_available=False, source_kind="downstream_of_intervention"),
        }

        chain = validate_optical_field_chain(
            fields,
            front_path_model="relay_coherent_psf",
            target_code_active=True,
            bridge_active=False,
        )
        views = build_strict_optical_field_views(fields, chain)

        self.assertEqual(chain["status"], "field-chain-invalid")
        self.assertIn("target-code-intervention", chain["reasons"])
        self.assertFalse(chain["full_chain_available"])
        self.assertEqual(list(views["amplitude"]), ["SLM output |E|", "encoder output |E|", "OLS input |E|", "OLS output |E|"])
        self.assertNotIn("chip |E|", views["amplitude"])

    def test_intensity_relay_marks_phase_unavailable_and_rejects_full_chain(self) -> None:
        import torch

        from ldm_project.strict_optical_fields import build_optical_field_record, validate_optical_field_chain

        field = torch.ones((1, 1, 2, 2), dtype=torch.complex64)
        chain = validate_optical_field_chain(
            {"slm_output": build_optical_field_record(field, upstream=None, operation="slm", phase_available=True)},
            front_path_model="relay_intensity_psf",
            target_code_active=False,
            bridge_active=False,
        )

        self.assertEqual(chain["status"], "field-chain-invalid")
        self.assertIn("intensity-only-front-relay", chain["reasons"])

    def test_sample_artifact_marks_target_intervention_invalid_without_full_chain_claim(self) -> None:
        import torch

        from ldm_project.one_shot_eval import build_strict_optical_field_artifact
        from ldm_project.strict_optical_fields import build_optical_field_record

        field = torch.ones((1, 1, 2, 2), dtype=torch.complex64)
        fields = {
            "slm_output": build_optical_field_record(field, upstream=None, operation="slm", phase_available=True),
            "ols_output": build_optical_field_record(field, upstream="slm_output", operation="ols", phase_available=True),
            "decoder_input": build_optical_field_record(field, upstream="ols_output", operation="target", phase_available=False, source_kind="target_code_intervention"),
        }

        artifact = build_strict_optical_field_artifact(
            fields,
            front_path_model="relay_coherent_psf",
            target_code_active=True,
            bridge_active=False,
        )

        self.assertEqual(artifact["status"], "field-chain-invalid")
        self.assertFalse(artifact["full_chain_available"])
        self.assertEqual(artifact["render_plane_names"], ["slm_output", "ols_output"])

    def test_renderer_uses_radian_phase_and_writes_shared_raw_amplitude_scale(self) -> None:
        import tempfile

        import torch

        from ldm_project.run_manager import RunManager
        from ldm_project.strict_optical_fields import (
            build_optical_field_record,
            build_strict_optical_field_views,
            save_strict_optical_field_grids,
            validate_optical_field_chain,
        )

        field = torch.full((1, 1, 2, 2), complex(3.0, 4.0), dtype=torch.complex64)
        fields = {
            "slm_output": build_optical_field_record(field, upstream=None, operation="slm", phase_available=True),
            "encoder_output": build_optical_field_record(field, upstream="slm_output", operation="encoder", phase_available=True),
            "ols_input": build_optical_field_record(field, upstream="encoder_output", operation="ols_input", phase_available=True),
            "ols_output": build_optical_field_record(field, upstream="ols_input", operation="ols", phase_available=True),
            "decoder_input": build_optical_field_record(field, upstream="ols_output", operation="decoder_input", phase_available=True),
            "chip": build_optical_field_record(field, upstream="decoder_input", operation="decoder", phase_available=True),
        }
        chain = validate_optical_field_chain(
            fields,
            front_path_model="relay_coherent_psf",
            target_code_active=False,
            bridge_active=False,
        )
        views = build_strict_optical_field_views(fields, chain)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RunManager(Path(tmpdir))
            paths = save_strict_optical_field_grids(
                manager,
                views,
                manager.images_dir / "known_field.png",
                title_prefix="known field",
                max_items=1,
                row_labels=["0"],
            )

            self.assertTrue(paths["amplitude"].is_file())
            self.assertTrue(paths["phase"].is_file())
            self.assertEqual(views["phase_limits_rad"], [-math.pi, math.pi])
            self.assertEqual(views["amplitude_vmax"], 5.0)

    def test_sampling_metadata_rejects_unannotated_resize_transition(self) -> None:
        import torch

        from ldm_project.strict_optical_fields import build_optical_field_record, validate_optical_field_chain

        field_200 = torch.ones((1, 1, 200, 200), dtype=torch.complex64)
        field_100 = torch.ones((1, 1, 100, 100), dtype=torch.complex64)
        fields = {
            "slm_output": build_optical_field_record(
                field_200,
                upstream=None,
                operation="slm",
                phase_available=True,
                sampling={"shape": [200, 200], "pixel_size_x": 2.5e-6, "pixel_size_y": 2.5e-6, "fov_x": 500e-6, "fov_y": 500e-6},
            ),
            "encoder_input": build_optical_field_record(
                field_100,
                upstream="slm_output",
                operation="generic_resize",
                phase_available=True,
                sampling={"shape": [100, 100], "pixel_size_x": 2e-6, "pixel_size_y": 2e-6, "fov_x": 200e-6, "fov_y": 200e-6},
            ),
        }

        chain = validate_optical_field_chain(
            fields,
            front_path_model="free_space",
            target_code_active=False,
            bridge_active=False,
        )

        self.assertEqual(chain["status"], "harness-invalid")
        self.assertTrue(any("generic_resize" in error for error in chain["errors"]))
