import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[3] / "scripts/research/lightgen/hw207_mnist4_roi_v1.py"
spec = importlib.util.spec_from_file_location("hw207_roi", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

RECTS = [(1, 3, 1, 3), (1, 3, 5, 7), (5, 7, 1, 3), (5, 7, 5, 7)]


def test_four_regions_and_outside():
    for label, (y0, _, x0, _) in enumerate(RECTS):
        image = np.zeros((8, 8))
        image[y0, x0] = 4
        image[0, 0] = 1
        result = module.readout(image, RECTS, label)
        assert result["prediction"] == label
        assert result["scores"][label] == 4
        assert result["outside_energy"] == 1
        assert result["correct_fraction"] == pytest.approx(0.8)
        assert result["strongest_wrong"] == 0
        assert result["top_margin"] == 4


def test_ties_and_invalid_geometry():
    image = np.zeros((8, 8))
    assert module.readout(image, RECTS)["prediction"] is None
    image[1, 1] = image[1, 5] = 1
    assert module.readout(image, RECTS)["prediction"] is None
    with pytest.raises(ValueError):
        module.readout(image, [RECTS[0]] * 4)
    with pytest.raises(ValueError):
        module.readout(image, [(0, 9, 0, 1)] * 4)
    image[0, 0] = -1
    with pytest.raises(ValueError):
        module.readout(image, RECTS)

@pytest.mark.parametrize("invalid", [True, 1.0, -1, 4])
def test_invalid_target(invalid):
    with pytest.raises(ValueError, match="invalid target"):
        module.readout(np.zeros((8, 8)), RECTS, invalid)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, 1j, "1"])
def test_nonphysical_input(invalid):
    with pytest.raises(ValueError, match="intensity"):
        module.readout(np.full((8, 8), invalid), RECTS)


def test_float32_accumulation_uses_float64():
    image = np.full((8, 8), 2e38, dtype=np.float32)
    result = module.readout(image, RECTS, np.int64(0))
    assert result["prediction"] is None
    assert result["roi_total"] == pytest.approx(32e38)
    assert result["outside_energy"] == pytest.approx(96e38)
    assert result["correct_fraction"] == pytest.approx(1 / 16)
