from __future__ import annotations

import json

import pytest
import torch

from ldm_project.v2.application.research import fiber_attribution as attribution


def test_attribution_variants_are_fixed_and_ordered() -> None:
    assert attribution.VARIANTS == ("I0", "S0", "P1", "L1", "O1", "M1", "S0-D0", "M1-D0")
    assert attribution.PANELS == (4242, 104729, 161803)


def test_bootstrap_is_reproducible_and_uses_registered_seed() -> None:
    rows = {}
    for variant in attribution.VARIANTS:
        rows[variant] = {
            panel: {"secondary_correctness": torch.tensor([(index + panel) % 3 == 0 for index in range(30)])}
            for panel in attribution.PANELS
        }
    first = attribution._bootstrap(rows)
    second = attribution._bootstrap(rows)
    assert first == second
    assert first["coupling"]["seed"] == 90210
    assert first["coupling"]["replicates"] == 10000


def test_bootstrap_direction_preserves_coupling_drop() -> None:
    rows = {}
    for variant in attribution.VARIANTS:
        rows[variant] = {
            panel: {"secondary_correctness": torch.ones(30, dtype=torch.bool)}
            for panel in attribution.PANELS
        }
    rows["S0"] = {panel: {"secondary_correctness": torch.zeros(30, dtype=torch.bool)} for panel in attribution.PANELS}
    result = attribution._bootstrap(rows)
    assert result["classification"]["ranked_effects"][0] == ("coupling", 1.0)


def test_load_tensor_rejects_path_escape(tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.pt"
    torch.save(torch.ones(1), outside)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        attribution._load_tensor(root, {"relative_path": "../outside.pt", "sha256": "bad"})


def test_v3_requires_structurally_valid_negative_closeout(tmp_path) -> None:
    path = tmp_path / "closeout.json"
    path.write_text(json.dumps({"schema": "fiber_image_closeout.v2", "gate": "valid", "conclusion": "no_reproducible_fiber_gain"}), encoding="utf-8")
    with pytest.raises(ValueError, match="structurally valid"):
        attribution._v3(path)


def test_attribution_closeout_is_never_formal(tmp_path, monkeypatch) -> None:
    measurement = tmp_path / "measurement.json"
    measurement.write_text(json.dumps({"schema": "fiber_attribution_measurement.v1", "gate": "valid", "git_commit": "x", "bootstrap": {"classification": {"conclusion": "distributed_or_inconclusive"}}}), encoding="utf-8")
    presentation = tmp_path / "presentation.json"
    presentation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(attribution.image, "_git", lambda: "x")
    monkeypatch.setattr(attribution.image, "_dirty", lambda: False)
    output = tmp_path / "closeout.json"
    attribution.closeout(type("Args", (), {"measurement_evidence": str(measurement), "presentation_evidence": str(presentation), "output": str(output)})())
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate"] == "valid"
    assert payload["formal_promotion"] is False
