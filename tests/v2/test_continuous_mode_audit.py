from __future__ import annotations

import torch
import pytest

from ldm_project.v2.application.conditional_continuous_audit import _lowpass_modes, _pca, _stable_kmeans
from ldm_project.v2.application.conditional_continuous_reachability import SAFE_AXIS_LOGIT_RMS, SAFE_PROBE_ALPHA, _ContinuousProbeGenerator, _full_contact_sheet, _temporary_delta


def test_continuous_pca_is_deterministic_and_rank_eight() -> None:
    values = torch.randn(64, 16, generator=torch.Generator().manual_seed(1234))
    first = _pca(values, components=8)
    second = _pca(values, components=8)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert first[2].shape == (16, 8)


def test_continuous_kmeans_is_deterministic_and_reports_small_clusters() -> None:
    values = torch.cat((torch.zeros(12, 3), torch.ones(4, 3), torch.full((2, 3), 5.0)))
    first = _stable_kmeans(values, k=4, iterations=20)
    second = _stable_kmeans(values, k=4, iterations=20)
    assert torch.equal(first[0], second[0])
    counts = [int((first[0] == index).sum()) for index in range(4)]
    assert min(counts) <= 2


def test_continuous_modes_are_band_limited_centered_and_finite() -> None:
    modes = torch.randn(8, 28, 28, generator=torch.Generator().manual_seed(9))
    filtered = _lowpass_modes(modes, shape=(200, 200), pixel_pitch_m=2.5e-6, max_cycles_per_mm=20.0)
    assert filtered.shape == (8, 200, 200)
    assert torch.isfinite(filtered).all()
    assert torch.allclose(filtered.mean(dim=(-2, -1)), torch.zeros(8), atol=1.0e-5)
    assert torch.linalg.matrix_rank(filtered.flatten(1)) == 8


def test_reachability_probe_uses_fixed_safe_logit_rms_and_zero_code() -> None:
    carrier = torch.zeros(10, 200, 200)
    modes = torch.randn(10, 8, 200, 200, generator=torch.Generator().manual_seed(13))
    generator = _ContinuousProbeGenerator(carrier, modes)
    codes = torch.zeros(1, 8)
    assert SAFE_PROBE_ALPHA * torch.tanh(torch.tensor(1.0)).item() == pytest.approx(SAFE_AXIS_LOGIT_RMS)
    assert generator.gain != pytest.approx(0.5)
    assert torch.equal(generator(codes, torch.zeros(1, dtype=torch.long)), torch.full((1, 1, 200, 200), 0.5))


def test_stateless_delta_restores_parameter_exactly() -> None:
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    original = parameter.detach().clone()
    with _temporary_delta(parameter, torch.ones_like(parameter)):
        assert torch.equal(parameter, original + 1.0)
    assert torch.equal(parameter, original)


@pytest.mark.scheduled_diagnostic
def test_reachability_contact_sheet_contains_all_170_samples(tmp_path) -> None:
    raw = torch.arange(170 * 1 * 4 * 4, dtype=torch.float32).reshape(170, 1, 4, 4)
    labels = torch.arange(170, dtype=torch.long).remainder(10)
    output = tmp_path / "panel.png"
    digest = _full_contact_sheet(output, raw, labels, 1.0)
    assert output.exists()
    assert len(digest) == 64
