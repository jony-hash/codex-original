from __future__ import annotations

import math

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
    LightGenPhysicalOpticsConfig,
)


def test_physical_forward_exposes_watt_per_pixel_and_fixed_gain_sensor_outputs():
    gain = 1.0e6
    physical = LightGenPhysicalOpticsConfig(
        input_power_w=249.3e-6,
        layer_power_transmission=0.889,
        global_fiber_geometry=True,
        fiber_pitch_m=125e-6,
        fiber_cladding_diameter_m=125e-6,
        fiber_mfd_m=14.222e-6,
        fiber_length_m=8.5e-3,
        camera_gain_w_inverse=gain,
    )
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=1024, physical_optics=physical)
    ).eval()
    images = torch.stack((torch.ones(1, 28, 28), torch.zeros(1, 28, 28)))

    with torch.no_grad():
        result = runtime.forward_from_image(images)

    pixel_area = math.prod(runtime.camera_plane.pixel_pitch_m)
    expected_power = result.raw_camera.tensor * pixel_area
    assert result.physical_sensor_output_w_per_pixel is not None
    assert result.task_sensor_output is not None
    assert result.task_sensor_camera_gain_w_inverse == gain
    assert torch.equal(result.physical_sensor_output_w_per_pixel, expected_power)
    assert torch.equal(result.task_sensor_output, (gain * expected_power).clamp(0.0, 1.0))

    trace = result.physical_power_trace
    assert trace is not None
    sensor_total = result.physical_sensor_output_w_per_pixel.flatten(1).sum(1)
    assert torch.allclose(sensor_total, trace["P_camera_W"], rtol=1e-6, atol=1e-12)
    assert sensor_total[0] > sensor_total[1]
    assert float(result.task_sensor_output.max()) < 1.0


def test_physical_sensor_output_does_not_silently_invent_camera_gain():
    physical = LightGenPhysicalOpticsConfig(
        input_power_w=249.3e-6,
        global_fiber_geometry=True,
        fiber_pitch_m=125e-6,
        fiber_cladding_diameter_m=125e-6,
        fiber_mfd_m=14.222e-6,
        fiber_length_m=8.5e-3,
    )
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=1024, physical_optics=physical)
    ).eval()
    with torch.no_grad():
        result = runtime.forward_from_image(torch.ones(1, 1, 28, 28))

    assert result.physical_sensor_output_w_per_pixel is not None
    assert result.task_sensor_output is None
    assert result.task_sensor_camera_gain_w_inverse is None
