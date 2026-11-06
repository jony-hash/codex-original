from pathlib import Path
import json

import pytest
import torch

from scripts.research.lightgen import hw207_mnist4_c0_v1 as c0


def test_split_roles_and_reproducibility():
    labels = torch.arange(10).repeat(20)
    train, val = c0.split(labels, 207)
    assert len(train) == 72 and len(val) == 8
    assert set(train.tolist()).isdisjoint(val.tolist())
    assert set(torch.cat((train, val)).tolist()) == set(torch.where(labels < 4)[0].tolist())
    for actual, repeated in zip((train, val), c0.split(labels, 207)):
        torch.testing.assert_close(actual, repeated)


@pytest.mark.parametrize("binary", [False, True])
def test_input_and_mapping(binary):
    raw = torch.zeros((2, 28, 28), dtype=torch.uint8)
    raw[0, 3:12, 2:20] = 255
    raw[1, 4, 5:7] = torch.tensor([127, 128])
    x = c0.features(raw, binary)
    assert x[1, 4, 5:7].tolist() == pytest.approx([0, 1] if binary else [127/255, 128/255])
    mapped = c0.map_image(x, binary)
    assert mapped.shape == (2, 100, 100)
    assert mapped.min() >= 0 and mapped.max() <= 1
    if binary:
        assert set(mapped.unique().tolist()) == {0, 1}
    assert c0.diagnose(raw, binary, 1)["count"] == 2


def test_uniform_mapping_identity_and_status():
    ones = torch.ones(1, 28, 28)
    for binary in (False, True):
        torch.testing.assert_close(c0.morphology(ones), c0.morphology(c0.map_image(ones, binary)))
    assert c0.TARGET["source_features"] == 784
    assert c0.TARGET["identity"] == "HW207_TARGET_DIAGNOSTIC"
    assert c0.CURRENT["identity"] == "HW207_CURRENT_MAPPING"
    assert c0.CURRENT["status"] == "BLOCKED"


def test_selection_uses_validation_only_and_is_reproducible():
    rng = torch.Generator().manual_seed(3)
    x = torch.rand(40, 784, generator=rng)
    y = torch.arange(4).repeat(10)
    cfg = {"seeds": [11], "decays": [0, 0.001], "epochs": 2, "batch": 8, "lr": 0.15}
    best, grid = c0.select(x[:32], y[:32], x[32:], y[32:], cfg)
    repeat, _ = c0.select(x[:32], y[:32], x[32:], y[32:], cfg)
    assert best[1] == repeat[1]
    assert all("test" not in run for candidate in grid for run in candidate["runs"])
    torch.testing.assert_close(best[2][0][0].weight, repeat[2][0][0].weight, atol=0, rtol=0)
    assert best[2][0][1]["validation"]["support"] == [2, 2, 2, 2]


def test_saved_population_if_requested():
    import os

    directory = os.environ.get("HW207_C0_RESULT")
    if directory is None:
        pytest.skip("set HW207_C0_RESULT to verify the full run")
    path = Path(directory)
    indices = json.loads((path / "indices.json").read_text(encoding="utf-8"))
    sets = [{(entry["source"], row) for row in entry["original_indices"]} for entry in indices.values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == 28911
    for mode in ("gray", "binary"):
        mapped = json.loads((path / f"mapping_{mode}.json").read_text(encoding="utf-8"))
        assert sum(mapped[k]["count"] for k in indices) == 28911
        report = json.loads((path / f"classification_{mode}.json").read_text(encoding="utf-8"))
        assert len(report["runs"]) == 3
        assert all(sum(run["test"]["support"]) == 4157 for run in report["runs"])
