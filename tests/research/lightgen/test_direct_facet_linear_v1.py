"""Focused tests for the isolated direct-facet thin-film diagnostic."""

import csv
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from scripts.research.lightgen import diagnose_lightgen_direct_facet_linear_v1 as diagnostic


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/research/lightgen/direct_facet_linear_v1.yaml"
REGISTER = REPO_ROOT / "research_assets/lightgen/hardware_g1_material_evidence_20261028_v1/material_parameter_register.csv"


@pytest.fixture
def model():
    return diagnostic.load_model_inputs(diagnostic.load_config(CONFIG))


def test_formula_matches_independent_four_boundary_solve(model):
    n_te = complex(model["n_te"], model["k_te"])
    result = diagnostic.single_layer_coefficients(
        model["n_glass"], n_te, model["n_air"], model["te_thickness_m"], model["wavelength_m"]
    )
    ref_r, ref_t = diagnostic.solve_boundary_continuity_independently(
        model["n_glass"], n_te, model["n_air"], model["te_thickness_m"], model["wavelength_m"]
    )
    assert abs(result["r"] - ref_r) <= 1e-10
    assert abs(result["t"] - ref_t) <= 1e-10


def test_zero_thickness_reduces_to_bare_interface_on_common_plane(model):
    result = diagnostic.single_layer_coefficients(
        model["n_glass"], complex(model["n_te"], model["k_te"]), model["n_air"], 0.0, model["wavelength_m"]
    )
    bare_r = (model["n_glass"] - model["n_air"]) / (model["n_glass"] + model["n_air"])
    bare_t = 2.0 * model["n_glass"] / (model["n_glass"] + model["n_air"])
    assert abs(result["r"] - bare_r) <= 1e-12
    assert abs(result["t"] - bare_t) <= 1e-12
    assert abs(np.angle(result["t"] * np.conj(bare_t))) <= 1e-12


def test_lossless_film_has_zero_absorption_and_conserves_power(model):
    n_te = complex(model["n_te"], 0.0)
    result = diagnostic.single_layer_coefficients(
        model["n_glass"], n_te, model["n_air"], model["te_thickness_m"], model["wavelength_m"]
    )
    power = diagnostic.power_metrics(result["r"], result["t"], model["n_glass"], model["n_air"])
    a_integral = diagnostic.integrate_absorption(
        model["te_thickness_m"], result["forward_0"], result["backward_0"], n_te,
        model["wavelength_m"], model["n_glass"], 128,
    )
    assert a_integral == 0.0
    assert abs(power["A_flux"]) <= 1e-12
    assert abs(power["R"] + power["T"] - 1.0) <= 1e-12


def test_passive_film_absorption_matches_independent_dissipation_integral(model):
    n_te = complex(model["n_te"], model["k_te"])
    result = diagnostic.single_layer_coefficients(
        model["n_glass"], n_te, model["n_air"], model["te_thickness_m"], model["wavelength_m"]
    )
    a64 = diagnostic.integrate_absorption(
        model["te_thickness_m"], result["forward_0"], result["backward_0"], n_te,
        model["wavelength_m"], model["n_glass"], 64,
    )
    a128 = diagnostic.integrate_absorption(
        model["te_thickness_m"], result["forward_0"], result["backward_0"], n_te,
        model["wavelength_m"], model["n_glass"], 128,
    )
    metrics = diagnostic.power_metrics(result["r"], result["t"], model["n_glass"], model["n_air"])
    assert a128 > 0.0
    assert abs(a64 - a128) <= 1e-9
    assert abs(metrics["A_flux"] - a128) <= 1e-9


def test_omitting_refractive_index_power_factor_is_rejected(model):
    r = (model["n_glass"] - model["n_air"]) / (model["n_glass"] + model["n_air"])
    t = 2.0 * model["n_glass"] / (model["n_glass"] + model["n_air"])
    wrong_t_power = abs(t) ** 2
    correct_t_power = (model["n_air"] / model["n_glass"]) * wrong_t_power
    wrong_metrics = {"R": abs(r) ** 2, "T": wrong_t_power, "A_flux": 1.0 - abs(r) ** 2 - wrong_t_power}
    assert not np.isclose(wrong_t_power, correct_t_power, rtol=1e-6)
    with pytest.raises(ValueError, match="Unphysical negative"):
        diagnostic.validate_power_balance(wrong_metrics, integrated_absorption=0.0)


def _original_malformed_register_text():
    with REGISTER.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    malformed_ids = {
        "TE-NW-S-A2-023", "TE-NW-S-A2-057", "TE-NW-S-A2-380", "TE-NW-S-A2-574",
        "TE-NW-L-A2-032", "TE-NW-L-A2-609", "TE-NW-L-A2-790",
    }
    broken = [rows[0]]
    for row in rows[1:]:
        if row[0] in malformed_ids:
            broken.append(row[:11] + [row[11] + ";" + row[12]])
        else:
            broken.append(row)
    assert sum(len(row) == 12 for row in broken[1:]) == 7
    buffer = StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(broken)
    return buffer.getvalue()


def test_parameter_register_rejects_original_seven_short_rows_and_accepts_repair(tmp_path):
    malformed_path = tmp_path / "original_malformed.csv"
    malformed_path.write_text(_original_malformed_register_text(), encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="12 fields, expected 13"):
        diagnostic.parse_parameter_register(malformed_path)

    rows = diagnostic.parse_parameter_register(REGISTER)
    assert rows and all(row["model_use"].strip() for row in rows)
    assert sum(row["record_id"].startswith("TE-NW-") and "A2" in row["record_id"] for row in rows) == 7


def test_preflight_then_measurement_saves_report_from_actual_json(tmp_path):
    out = tmp_path / "diagnostic-only"
    assert diagnostic.main(["preflight", "--config", str(CONFIG), "--output-dir", str(out)]) == 0
    assert diagnostic.main(["measure", "--config", str(CONFIG), "--output-dir", str(out)]) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    payload = json.loads((out / "linear_diagnostic.json").read_text(encoding="utf-8"))
    assert payload["status"] == "valid"
    assert f"{payload['coated']['T']:.12g}" in report
    assert (out / "bare_vs_coated.csv").is_file()
    assert (out / "te_field_absorption_profile.csv").is_file()
