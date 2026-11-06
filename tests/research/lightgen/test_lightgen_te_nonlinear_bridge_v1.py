import csv
import copy
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from PIL import Image

from scripts.research.lightgen import diagnose_lightgen_te_nonlinear_bridge_v1 as bridge


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/research/lightgen/te_nonlinear_bridge_v1.json"
OUTPUT = ROOT / "outputs/20261029_lightgen_te_nonlinear_bridge_v1/runs/diagnostic-only_attempt_02"
OLD_OUTPUT = ROOT / "outputs/20261029_lightgen_te_nonlinear_bridge_v1/runs/diagnostic-only_attempt_01"


def test_reversed_transmission_axis_is_rejected() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    axis = config["s9"]["panel_c"]["y_axis"]
    axis["values"] = list(reversed(axis["values"]))
    with pytest.raises(ValueError, match="ordinate values must increase upward"):
        bridge.validate_config(config)


def test_532nm_curve_cannot_be_relabelled_from_s9() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["s9"]["probe_wavelength_nm"] = 532
    with pytest.raises(ValueError, match="1530 nm pump and 1520 nm CW probe"):
        bridge.validate_config(config)


def test_normalized_response_cannot_be_relabelled_as_absolute_output() -> None:
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config["s9"]["ordinate"] = "absolute output energy"
    with pytest.raises(ValueError, match="normalized intensity response only"):
        bridge.validate_config(config)


def test_corrected_s11_separates_the_191_and_223_fluence_traces() -> None:
    corrected = list(csv.DictReader((OUTPUT / "s11a_delay_curves.csv").open(encoding="utf-8-sig", newline="")))
    old = list(csv.DictReader((OLD_OUTPUT / "s11a_delay_curves.csv").open(encoding="utf-8-sig", newline="")))
    corrected_anchor = {}
    old_anchor = {}
    for row in corrected:
        if float(row["x_pixel"]) == 258.0 and row["pump_fluence_pJ_um2"] in {"2.23", "1.91"}:
            corrected_anchor[row["pump_fluence_pJ_um2"]] = float(row["y_pixel_method_b"])
    for row in old:
        if float(row["x_pixel"]) == 258.0 and row["pump_fluence_pJ_um2"] in {"2.23", "1.91"}:
            old_anchor[row["pump_fluence_pJ_um2"]] = float(row["y_pixel"])
    assert corrected_anchor == {"2.23": 488.0, "1.91": 397.5}
    assert old_anchor["1.91"] == 483.5
    assert abs(old_anchor["1.91"] - corrected_anchor["1.91"]) > 80


def test_production_s11_extractor_confirms_source_anchors_independently() -> None:
    config = bridge.load_config(CONFIG)
    sources = bridge._resolve_sources(config)
    with zipfile.ZipFile(sources["si_docx"]) as archive:
        raster = np.asarray(Image.open(BytesIO(archive.read(config["paper"]["si_figure_media_members"]["s11"]))).convert("RGB"))
    curves, missing, summary = bridge._trace_s11_dual_method(
        raster, config["s11"]["panel_a"], config["s11"]["pump_fluences"], config["digitization"]
    )
    anchors = {row["pump_fluence_pJ_um2"]: row for row in summary["anchor_checks"]}
    assert anchors[1.91]["selected_y_pixel"] == 397.5
    assert anchors[2.23]["selected_y_pixel"] == 488.0
    assert max(row["absolute_difference_px"] for row in summary["anchor_checks"]) <= 3.0
    assert summary["interpolation_across_segments"] is False
    assert len(missing) > 0
    assert len(curves["1.91"]) < config["s11"]["panel_a"]["plot_bounds_pixels"][2]


def test_s11_segment_interpolation_rejects_domain_edges_and_visible_gaps() -> None:
    rows = list(csv.DictReader((OUTPUT / "s11a_delay_curves.csv").open(encoding="utf-8-sig", newline="")))
    curve = [row for row in rows if row["pump_fluence_pJ_um2"] == "2.55"]
    segment_ids = sorted({int(row["segment_id"]) for row in curve})
    assert len(segment_ids) > 1
    first_segment = [row for row in curve if int(row["segment_id"]) == segment_ids[0]]
    next_segment = [row for row in curve if int(row["segment_id"]) == segment_ids[1]]
    query_in_gap = (float(first_segment[-1]["delay_ps"]) + float(next_segment[0]["delay_ps"])) / 2
    with pytest.raises(ValueError, match="outside the selected visible curve segment"):
        bridge._interpolate_within_segment(curve, query_in_gap, segment_ids[0], "delay_ps", "normalized_transmission")
    with pytest.raises(ValueError, match="outside the selected visible curve segment"):
        bridge._interpolate_within_segment(curve, float(first_segment[0]["delay_ps"]) - 1, segment_ids[0], "delay_ps", "normalized_transmission")


def test_s9_shared_reference_is_consumed_by_both_digitized_curves() -> None:
    saved = json.loads((OUTPUT / "measurement.json").read_text(encoding="utf-8"))
    with (OUTPUT / "s9_normalized_response_bridge.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ref = saved["s9"]["shared_comparison_reference"]
    assert rows
    assert float(rows[0]["pump_energy_pJ"]) == pytest.approx(ref["energy_pJ"])
    assert float(rows[0]["tr_norm_relative_to_Eref"]) == pytest.approx(1.0)
    assert float(rows[0]["Tk_relative_to_kref"]) == pytest.approx(1.0)
    for row in rows:
        assert float(row["tr_norm_relative_to_Eref"]) == pytest.approx(
            float(row["tr_norm_S9c_original_initial_reference"]) / ref["s9c_at_reference"], abs=1e-12
        )
        assert float(row["Tk_relative_to_kref"]) == pytest.approx(
            float(row["T_k_absolute_FDTD_marker_interpolation"]) / ref["FDTD_T_at_reference"], abs=1e-12
        )
        assert row["independent_validation"] == "False"


def test_changed_semantic_config_cannot_reuse_old_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(bridge.load_config(CONFIG))
    config["s11"]["panel_a"]["method_b_anchor_y_pixels_by_fluence"][1] += 0.1
    monkeypatch.setattr(bridge, "load_config", lambda path: config)
    with pytest.raises(ValueError, match="preflight receipt is stale"):
        bridge.preflight(CONFIG, OUTPUT)


def test_saved_coherent_field_power_keeps_interference_cross_term() -> None:
    # Two equal coherent contributions in one sample produce four units of
    # irradiance, while the incorrect incoherent sum would produce two.
    contributions = np.asarray([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    coherent_field = np.sum(contributions)
    measured = bridge._intensity_metrics(np.asarray([[coherent_field]], dtype=np.complex128), 1.0)
    incorrect_incoherent_power = float(np.sum(np.abs(contributions) ** 2))
    assert measured["total_power_W"] == 4.0
    assert measured["total_power_W"] != incorrect_incoherent_power


def test_corrected_production_saved_measurement_and_report_close_the_gate() -> None:
    preflight = json.loads((OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    saved = json.loads((OUTPUT / "measurement.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "READY_DIAGNOSTIC_ONLY"
    assert saved["status"] == "COMPLETE_DIAGNOSTIC_ONLY"
    assert saved["s9"]["pump_wavelength_nm"] == 1530
    assert saved["s9"]["probe_wavelength_nm"] == 1520
    assert saved["s9"]["s9d_detected_marker_count"] == 7
    assert saved["s9"]["bridge_exclusion_count"] >= 1
    assert saved["s9"]["anchor"]["normalization_identity_r_S9c"] == 1.0
    assert saved["s9"]["anchor"]["normalization_identity_Tk"] == 1.0
    assert saved["s11"]["digitization_pixels_are_independent_experiments"] is False
    assert saved["s11"]["missing_interval_count"] > 0
    assert saved["g1_membrane_plane"]["recomputed_in_attempt_02"] is False
    assert saved["output_consumer_verification"]["s11_segments_do_not_bridge_missing_pixels"] is True
    assert all((OUTPUT / name).is_file() and (OUTPUT / name).stat().st_size > 0 for name in saved["output_consumer_verification"]["required_files_present_and_nonempty"])
    report = (OUTPUT / "report.md").read_text(encoding="utf-8")
    assert "attempt 01 原始内容保留" in report
    assert "没有运行 G1 非线性传播、训练或实物工作" in report
    assert "no output phase" in " ".join(saved["g1_nonlinear_admission"]["reason"])
