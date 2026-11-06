"""Focused tests for the ADMA2026 Te proxy producer and its consumers."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.lightgen import diagnose_lightgen_te_adma2026_proxy_v1 as diagnostic


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/research/lightgen/te_adma2026_proxy_v1.json"


@pytest.fixture
def config():
    return diagnostic.load_config(CONFIG)


def test_config_preserves_unknown_response_wavelength_and_532_spectrum_proxy(config):
    assert config["response"]["wavelength_nm"] is None
    assert config["response"]["quantity"] == "normalized_transmission"
    assert config["spectrum"]["target_wavelength_nm"] == 532.0
    assert config["local_linear_model"]["common_output_reference_plane_nm"] == 50.0


@pytest.mark.parametrize(
    ("group", "key", "value", "message"),
    [
        ("response", "ordinate_direction", "increases_upward", "downward in image coordinates"),
        ("response", "quantity", "absolute_output_energy", "absolute output energy"),
        ("spectrum", "target_wavelength_nm", 1530.0, "fixed at 532 nm"),
    ],
)
def test_production_config_validator_rejects_wrong_axis_quantity_and_wavelength(config, group, key, value, message):
    config[group][key] = value
    with pytest.raises(ValueError, match=message):
        diagnostic.validate_config(config)


def test_piecewise_response_keeps_values_above_one_and_rejects_extrapolation():
    xp = np.asarray([0.5, 1.0, 1.5])
    fp = np.asarray([1.01, 1.0, 0.91])
    result = diagnostic._piecewise_linear(np.asarray([0.5, 1.25, 1.5]), xp, fp)
    assert result[0] == pytest.approx(1.01)
    assert result[-1] == pytest.approx(0.91)
    with pytest.raises(ValueError, match="extrapolation"):
        diagnostic._piecewise_linear(np.asarray([0.49]), xp, fp)


def test_preflight_then_measurement_consumes_real_sources_and_saves_bounded_results(tmp_path):
    output = tmp_path / "diagnostic-only_attempt_01"
    assert diagnostic.main(["preflight", "--config", str(CONFIG), "--output-dir", str(output)]) == 0
    assert diagnostic.main(["measure", "--config", str(CONFIG), "--output-dir", str(output)]) == 0
    payload = json.loads((output / "measurement.json").read_text(encoding="utf-8"))
    report = (output / "report.md").read_text(encoding="utf-8")
    assert payload["status"] == "valid"
    assert payload["response_curve_identity"]["wavelength_nm"] is None
    assert payload["response_curve_identity"]["point_count_in_interpolation"] == 22
    assert payload["local_linear_optics"]["reference_plane_nm"] == pytest.approx(50.0, abs=1e-12)
    assert payload["local_linear_optics"]["independent_boundary_solution"]["delta_t"] <= 1e-10
    assert "绝对输出能量标定" in report
    assert (output / "fig2c_points.csv").is_file()
    detection = payload["response_extraction_audit"]
    assert detection["unresolved_component_count"] == 8
    assert len(detection["unmatched_center_indices"]) == 3
    with (output / "fig2c_unresolved_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        unresolved = list(csv.DictReader(stream))
    assert len(unresolved) == 11
    assert "8 个合并/截断连通块和 3 个无圆拟合匹配的中心候选排除，共 11 个未解析候选；另记录 35 个不匹配边缘圆候选" in report
    assert (output / "figs2_n_3200px.csv").is_file()
    assert (output / "te_532_local_linear.csv").is_file()
    for name in ("n", "k"):
        with (output / f"figs2_{name}_3200px.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        wavelengths = [float(row["wavelength_nm"]) for row in rows]
        assert min(wavelengths) >= 500.0
        assert max(wavelengths) <= 2000.0
    assert (output / "adma2026_fig2c_normalized_response.pdf").is_file()
    assert (output / "adma2026_figs2_te_nk_spectrum.pdf").is_file()
