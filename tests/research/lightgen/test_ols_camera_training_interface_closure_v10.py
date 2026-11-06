from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.research.lightgen import diagnose_ols_camera_training_interface_closure_v10 as v10


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/20261016_ols_decoder_camera_interface_v9/runs/attempt_02"
CONFIG = ROOT / "configs/research/lightgen/ols_camera_training_interface_closure_v10.yaml"


def test_clipped_camera_factor_spectrum_matches_explicit_matrix() -> None:
    rng = np.random.default_rng(20261017)
    left = rng.normal(size=(23, 5))
    right = rng.normal(size=(5, 11))
    actual = v10._small_product_spectrum(left, right)
    expected = np.linalg.svd(left @ right, compute_uv=False)
    np.testing.assert_allclose(actual, expected[: len(actual)], rtol=1.0e-12, atol=1.0e-12)


def test_v9_saved_input_preflight_rejects_bad_clip_or_missing_input(tmp_path: Path) -> None:
    import json

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = v10._preflight(SOURCE, config)
    assert result["passed"] is True
    assert result["context_archive_count"] == 16
    with pytest.raises(ValueError, match="expected 16"):
        v10._preflight(tmp_path, config)
