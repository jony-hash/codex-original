from __future__ import annotations

import numpy as np

from scripts.research.lightgen import attribute_chen_lightgen_400_ols_v1 as attribution


def test_atlas_readout_selects_sample_arrays_but_preserves_scalar_metadata(tmp_path) -> None:
    unit_dir = tmp_path / "unit_readouts" / "R0"
    unit_dir.mkdir(parents=True)
    sample_ids = np.arange(10, dtype=np.int64)
    task = np.arange(40, dtype=np.float32).reshape(10, 1, 2, 2)
    np.savez_compressed(
        unit_dir / "reconstruction_0001.npz",
        sample_ids=sample_ids,
        physical_w_per_pixel=task,
        original_gain_w_inverse=np.asarray(2.5, dtype=np.float64),
    )

    loaded = attribution._load_unit_readout(unit_dir / "reconstruction", position=13)

    np.testing.assert_array_equal(loaded["sample_ids"], sample_ids[3])
    np.testing.assert_array_equal(loaded["physical_w_per_pixel"], task[3])
    np.testing.assert_array_equal(loaded["original_gain_w_inverse"], np.asarray(2.5))
