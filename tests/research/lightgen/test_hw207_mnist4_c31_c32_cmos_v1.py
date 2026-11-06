import json
from pathlib import Path

import pytest
import torch

from scripts.research.lightgen.hw207_mnist4_c31_c32_cmos_v1 import prepare_configuration, forward_hw207_mnist4
from ldm_project.v2.application.research.lightgen.hw207_spatial_geometry import roi_scores
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from ldm_project.v2.optics.relays import path_abcd


@pytest.fixture(scope='module')
def context():
    path = Path('configs/research/lightgen/hw207_mnist4_c21_c3_spatial_v1.json')
    candidate = json.loads(path.read_text(encoding='utf-8'))['relay_candidates'][0]
    return prepare_configuration(candidate, 'cpu')


def test_zero_and_tie_abstain(context):
    result = forward_hw207_mnist4(torch.zeros((1, 28, 28), dtype=torch.uint8), context)
    assert result['prediction'].tolist() == [-1]
    assert result['detected_energy'].tolist() == [0]
    assert roi_scores(torch.ones(768, 768), context.rois)['prediction'] == -1


def test_production_forward_energy_and_batch(context):
    image = torch.zeros((2, 28, 28), dtype=torch.uint8)
    image[:, 8:20, 10:18] = 255
    result = forward_hw207_mnist4(image, context)
    single = forward_hw207_mnist4(image[:1], context)
    assert result['lightgen_output_field'].shape == (2, 100, 100)
    assert result['cmos_pixel_integral'].shape == (2, 768, 768)
    assert torch.allclose(result['roi_scores'][0], single['roi_scores'][0], rtol=1e-5, atol=1e-8)
    ref = roi_scores(result['cmos_pixel_integral'][0], context.rois)
    assert torch.allclose(result['roi_scores'][0], ref['scores'])
    assert result['prediction'][0].item() == ref['prediction']
    assert torch.allclose(result['energy']['camera_crop_energy_um2'], result['detected_energy'], rtol=1e-8)
    assert context.magnification == -1


def test_relay_analytic_and_precision(context):
    provenance = OperationProvenance('uniform_probe', None, None)
    single = context.relay(ComplexField(torch.ones(1, 1, 100, 100, dtype=torch.complex64), context.relay.source, provenance)).tensor
    double = context.relay(ComplexField(torch.ones(1, 1, 100, 100, dtype=torch.complex128), context.relay.source, provenance)).tensor
    assert torch.allclose(double, torch.ones_like(double), atol=1e-10, rtol=1e-10)
    assert torch.linalg.vector_norm(single.to(torch.complex128) - double) / torch.linalg.vector_norm(double) < 1e-5
    for f2 in (30, 150, 200):
        f1 = .03
        f2 = f2 / 1000
        matrix = path_abcd((f1, f1 + f2, f2), (f1, f2))
        assert matrix.a == pytest.approx(-f2 / f1)
        assert matrix.b_m == pytest.approx(0, abs=1e-12)
    assert context.camera_matrix.values().sum().item() == pytest.approx(256 * 256 * 4, rel=1e-10)
