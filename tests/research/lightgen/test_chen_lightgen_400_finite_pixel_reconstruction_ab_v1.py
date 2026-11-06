from __future__ import annotations

import torch

from scripts.research.lightgen import execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1 as runner


def _comparison(row: int, label: int) -> dict:
    plane = {"pixel_center": torch.ones((1, 1, 1, 1), dtype=torch.complex64),
             "pixel_area_average": torch.ones((1, 1, 1, 1), dtype=torch.complex64)}
    result = {
        "finite_pixel_plane_values": tuple(dict(plane) for _ in range(7)),
        "ols_coefficients": torch.ones((1, 100), dtype=torch.complex64),
        "camera_roi_w_per_pixel": torch.ones((1, 1, 500, 500)),
        "physical_w_per_pixel": torch.ones((1, 1, 400, 400)),
        "plane_powers_w": torch.ones((1, 7)),
        "input_power_w": torch.ones((1,)),
    }
    return {"result": result, "row": row, "label": label}


def test_full_chain_qualification_computes_missing_q12_and_reuses_valid_partial_cache(
    tmp_path, monkeypatch,
) -> None:
    outdir = tmp_path / "run"
    outdir.mkdir()
    state = "source_step_30000"
    population = {"confirm_rows_by_class": torch.arange(100, 110).view(10, 1)}
    phases = {state: {}}
    for label in (0, 2):
        row = 100 + label
        raw = _comparison(row, label)
        runner._save_full_chain_raw(outdir, "reference", state, raw, 12)
    runner._save_full_chain_raw(outdir, "candidate", state, _comparison(100, 0), 8)

    calls: list[tuple[int, int, int]] = []

    def fake_forward(_phases, row, label, _pixels, order, _device, **_kwargs):
        calls.append((row, label, order))
        return _comparison(row, label), {}

    monkeypatch.setattr(runner, "_full_chain_comparison", fake_forward)
    result = runner._full_chain_qualification(
        phases, population, torch.arange(10), torch.zeros((1, 1)), [8], torch.device("cpu"), outdir
    )

    # Eight missing q12 references and nine missing q8 candidates are produced;
    # the cached two references and one candidate are consumed without propagation.
    assert len(calls) == 17
    assert result["forwards_completed"] == result["forwards_expected"] == 17
    assert result["orders"]["8"]["status"] == "NUMERICALLY_QUALIFIED"
    assert len(result["orders"]["8"]["records"]) == 10
    for label, row in enumerate(range(100, 110)):
        assert runner._raw_observation_path(outdir, "reference", state, row, 12).is_file()
        assert runner._raw_observation_path(outdir, "candidate", state, row, 8).is_file()


def test_import_qualification_cache_validates_and_copies_only_expected_observations(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    rows = torch.arange(100, 110).view(10, 1)
    population = {"confirm_rows_by_class": rows}
    phases = {"source_step_30000": {}, "historical_500_step_endpoint": {}}
    for state in phases:
        for label in (0, 2):
            row = int(rows[label, 0])
            for order in (12, 16):
                runner._save_full_chain_raw(
                    source, "reference", state, _comparison(row, label), order
                )
    runner._save_full_chain_raw(source, "candidate", "source_step_30000",
                                _comparison(100, 0), 8)

    imported = runner._import_qualification_cache(source, destination, phases, population)

    assert imported["imported_count"] == 9
    assert len(imported["files"]) == 9
    for entry in imported["files"]:
        path = runner._raw_observation_path(
            destination, entry["stage"], entry["state"], entry["row"], entry["order"]
        )
        assert path.is_file()
        runner._load_full_chain_raw(destination, entry["stage"], entry["state"],
                                    entry["row"], entry["label"], entry["order"])
