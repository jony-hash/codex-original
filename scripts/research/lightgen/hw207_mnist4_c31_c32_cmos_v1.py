"""HW207 ideal single-screen diagnostic forward path; not a hardware model."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from scripts.research.lightgen.hw207_mnist4_c2_v1 import coherent_pupil
from ldm_project.v2.application.research.lightgen import hw207_spatial_geometry as geo
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from ldm_project.v2.optics.propagation import angular_spectrum_propagate
from ldm_project.v2.optics.relays import CoherentImagingRelay


@dataclass
class ForwardConfiguration:
    source_matrix: torch.Tensor
    target_matrix: torch.Tensor
    affine: tuple
    input_plane: PlaneSpec
    screen: FiniteCropPhaseStack
    relay: CoherentImagingRelay
    camera_matrix: torch.Tensor
    rois: tuple
    lattice_shape: tuple
    magnification: float
    defocus_um: float
    device: torch.device


def prepare_configuration(candidate, device, defocus_um=0.0):
    """Prepare the fixed 0-degree/nearest C2.1 map and ideal 4f output."""
    device = torch.device(device)
    fi1, fi2 = candidate['input_focal_lengths_mm']
    fo1, fo2 = candidate['output_focal_lengths_mm']
    affine = geo.scaled_rotation(-fi2 / fi1, 0.0)
    source, lattice = geo.build_source_sampling_matrix(affine, method='nearest')
    target = geo.build_target_area_matrix(lattice, affine, fill_factor=0.92)
    plane = PlaneSpec('c31_input', (100, 100), (2e-6, 2e-6), 532e-9, 0.0, 'ideal_target')
    screen = FiniteCropPhaseStack(layers=1, active_shape=(100, 100),
                                  propagation_shape=(512, 512), spacing_m=0.0).to(device)
    screen.requires_grad_(False)
    screen_output = plane.with_plane('c31_screen_layer_1')
    m = -fo2 / fo1
    f1, f2 = fo1 * 1e-3, fo2 * 1e-3
    segments = (f1, f1 + f2, f2)
    target_plane = screen_output.with_plane('c31_image', z_m=sum(segments),
                                            pixel_pitch_m=(abs(m) * 2e-6,) * 2)
    relay = CoherentImagingRelay(screen_output, target_plane, m, name='c31_output',
                                 segments_m=segments, focal_lengths_m=(f1, f2),
                                 clear_aperture_m=6.384e-3)
    camera_matrix, _ = geo.build_cmos_area_matrix(((1., 0.), (0., 1.)),
        object_pixels=256, object_pixel_um=2 * abs(m), sensor_shape=(768, 768),
        sensor_pixel_um=6.5)
    width = 200 * abs(m)
    start = math.floor((2048 * 6.5 / 2 - width / 2) / 6.5) - 640
    end = math.ceil((2048 * 6.5 / 2 + width / 2) / 6.5) - 640
    if (end - start) % 2 or not (0 <= start < end <= 768):
        raise ValueError('Projected image cannot form four equal fixed pixel ROIs')
    mid = (start + end) // 2
    rois = ((start, mid, start, mid), (start, mid, mid, end),
            (mid, end, start, mid), (mid, end, mid, end))
    return ForwardConfiguration(source.to(device), target.to(device), affine, plane,
        screen, relay, camera_matrix.to(device), rois, tuple(lattice['shape']),
        m, float(defocus_um), device)


@torch.no_grad()
def forward_hw207_mnist4(image, configuration):
    """Return fields, pixel-integrated energy, fixed ROI scores and energy ledger."""
    c = configuration
    if image.ndim != 3 or image.shape[-2:] != (28, 28) or image.dtype != torch.uint8:
        raise ValueError('image must be uint8 [batch,28,28]')
    binary = (image.to(c.device) >= 128).float()
    mapped = geo.map_binary_images(binary, c.source_matrix, c.target_matrix, c.affine)
    amplitude = mapped['irradiance'].clamp_min(0).sqrt()
    field = torch.complex(amplitude, torch.zeros_like(amplitude))
    padded = F.pad(field, (206, 206, 206, 206))
    filtered = coherent_pupil(padded, 2.0, 0.1, 0.0, 0.532)
    chip_input = filtered[:, 206:306, 206:306]
    source = ComplexField(chip_input[:, None], c.input_plane,
        OperationProvenance('ideal_input_relay_pupil_crop', None, None))
    chip_output, layers, layer_stats = c.screen(source, 'c31_screen')
    image_field = c.relay(chip_output)
    padded_camera = F.pad(image_field.tensor, (206, 206, 206, 206))
    camera_plane = image_field.plane.with_plane('c31_camera_pad', shape=(512, 512))
    camera_field = angular_spectrum_propagate(ComplexField(padded_camera, camera_plane,
        OperationProvenance('camera_zero_padding', image_field.plane.plane_id, None)),
        c.defocus_um * 1e-6, 'c31_camera_defocused')
    crop = camera_field.tensor[:, 0, 128:384, 128:384]
    intensity = crop.abs().square()
    pixels = torch.sparse.mm(c.camera_matrix, intensity.flatten(1).double().T).T
    pixels = pixels.reshape(-1, 768, 768)
    scores = torch.stack([pixels[:, r0:r1, c0:c1].sum((1, 2))
                          for r0, r1, c0, c1 in c.rois], dim=1)
    maximum = scores.max(1, keepdim=True).values
    tied = torch.isclose(scores, maximum, rtol=1e-12, atol=0).sum(1) != 1
    total = pixels.sum((1, 2))
    prediction = torch.where((total > 0) & ~tied, scores.argmax(1), -1)
    pitch_um = abs(c.magnification) * 2
    ledger = {
        'dmd_active_area_um2': mapped['commands'].double().sum(1) * 7.56**2 * .92,
        'mapped_input_energy_um2': mapped['irradiance'].double().sum((1, 2)) * 4,
        'lightgen_input_energy_um2': chip_input.abs().square().double().sum((1, 2)) * 4,
        'lightgen_output_energy_um2': chip_output.tensor.abs().square().double().sum((1, 2, 3)) * 4,
        'relay_output_energy_um2': image_field.tensor.abs().square().double().sum((1, 2, 3)) * pitch_um**2,
        'defocus_window_energy_um2': camera_field.tensor.abs().square().double().sum((1, 2, 3)) * pitch_um**2,
        'camera_crop_energy_um2': intensity.double().sum((1, 2)) * pitch_um**2,
        'detected_energy_um2': total,
    }
    return {'dmd_binary_pattern': mapped['commands'], 'lightgen_input_field': chip_input,
            'lightgen_output_field': chip_output.tensor[:, 0], 'layer_fields': layers,
            'layer_power': layer_stats['layer_power'], 'camera_ideal_intensity': intensity,
            'cmos_pixel_integral': pixels, 'roi_scores': scores,
            'outside_roi_energy': (total - scores.sum(1)).clamp_min(0),
            'detected_energy': total, 'prediction': prediction, 'energy': ledger}
