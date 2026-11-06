from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.research.lightgen import audit_lightgen_small_class9_bounded_review_extension_v1 as extension
from scripts.research.lightgen import audit_lightgen_small_class9_saved_defect_diagnosis_v1 as base


def _inputs() -> SimpleNamespace:
    blocks = {}
    for cell in ("AA", "BB"):
        for label in (9, 4, 8):
            predicted = torch.full((1000,), label, dtype=torch.long)
            if label == 9:
                predicted[:300] = 4
                predicted[300:600] = 8
            logits = torch.zeros((1000, 10))
            logits[torch.arange(1000), predicted] = 2
            blocks[(cell, label)] = {"conditions": {f"{cell}_g1p00": {
                "logits": logits, "predicted_labels": predicted,
                "images": torch.zeros((1, 1, 1, 1)).expand(1000, 1, 28, 28),
            }}}
    return SimpleNamespace(blocks=blocks, proposal_ids=np.arange(1000))


def _parent() -> tuple[dict, dict]:
    rows = []
    for index in range(60):
        primary = index < 20
        rows.append({"blind_id": f"old_{index}", "cell": "AA" if index < 10 else "BB" if index < 20 else "BA",
                     "requested_label": 9, "proposal_id": index, "ed_prediction": 4 if primary else 9})
    return {"entries": rows, "review_id": "parent-review"}, {row["blind_id"]: "9" for row in rows}


def _manifest() -> tuple[dict, dict, dict, dict]:
    parent, parent_labels = _parent()
    entries = extension._select(_inputs(), parent)
    manifest = {"task_id": extension.TASK_ID, "review_id": "extension-review", "entries": entries}
    labels = {row["blind_id"]: "9" for row in entries}
    return parent, parent_labels, manifest, labels


def test_selection_is_fixed_unused_and_exactly_40_primary_plus_16_controls() -> None:
    inputs = _inputs()
    parent, _ = _parent()
    entries = extension._select(inputs, parent)
    assert entries == extension._select(inputs, parent)
    old = {(r["requested_label"], r["proposal_id"]) for r in parent["entries"]}
    new = {(r["requested_label"], r["proposal_id"]) for r in entries}
    assert len(new) == 56 and not old.intersection(new)
    for cell in ("AA", "BB"):
        selected = [r for r in entries if r["cell"] == cell]
        assert len(selected) == 28
        for stratum, label, prediction, count in extension.STRATA:
            rows = [r for r in selected if r["stratum"] == stratum]
            assert len(rows) == count
            assert all(r["requested_label"] == label for r in rows)
            if prediction is not None:
                assert all(r["ed_prediction"] == prediction for r in rows)


def test_selection_rejects_insufficient_fixed_stratum() -> None:
    inputs = _inputs()
    parent, _ = _parent()
    values = inputs.blocks[("AA", 9)]["conditions"]["AA_g1p00"]
    values["logits"].zero_()
    values["logits"][:, 9] = 2
    values["predicted_labels"].fill_(9)
    with pytest.raises(base.EvidenceError, match="insufficient"):
        extension._select(inputs, parent)


@pytest.mark.parametrize("corruption", ["review", "task", "duplicate", "missing", "invalid"])
def test_rating_boundary_rejects_wrong_review_or_incomplete_labels(corruption: str) -> None:
    _, _, manifest, labels = _manifest()
    payload = {"schema": "lightgen_blind_ratings.v1", "task_id": extension.TASK_ID,
               "review_id": manifest["review_id"], "ratings": [{"blind_id": k, "rating": v} for k, v in labels.items()]}
    assert extension._validate_ratings(payload, manifest) == labels
    if corruption == "review":
        payload["review_id"] = "parent-review"
    elif corruption == "task":
        payload["task_id"] = base.TASK_ID
    elif corruption == "duplicate":
        payload["ratings"][1]["blind_id"] = payload["ratings"][0]["blind_id"]
    elif corruption == "missing":
        payload["ratings"].pop()
    else:
        payload["ratings"][0]["rating"] = "10"
    with pytest.raises(base.EvidenceError):
        extension._validate_ratings(payload, manifest)


@pytest.mark.parametrize("rating,category", [("9", "readable9"), ("4", "readable_other"), ("unreadable", "unreadable")])
def test_fixed_primary60_decision_consumes_both_rating_waves(rating: str, category: str) -> None:
    parent, old, manifest, new = _manifest()
    old = {k: rating for k in old}
    new = {k: rating for k in new}
    result = extension._diagnose(parent, old, manifest, new)
    assert result["counts"][category] == 60
    assert result["per_wave"]["parent20"][category] == 20
    assert result["per_wave"]["extension40"][category] == 40
    assert result["decision_category"] == category


def test_pooled_majority_cannot_override_opposite_cell_result() -> None:
    parent, old, manifest, new = _manifest()
    aa = []
    for mapping, labels in ((parent, old), (manifest, new)):
        for row in mapping["entries"]:
            if row["cell"] == "AA" and row["requested_label"] == 9 and row["ed_prediction"] in (4, 8):
                aa.append((row["blind_id"], labels))
    for card, labels in aa[:20]:
        labels[card] = "unreadable"
    result = extension._diagnose(parent, old, manifest, new)
    assert result["counts"]["readable9"] == 40
    assert result["wilson_95"]["readable9"][0] > 0.5
    assert result["decision_category"] is None


def test_blind_html_uses_extension_identity_and_variable_card_count() -> None:
    _, _, manifest, _ = _manifest()
    cards = base._public_cards(manifest["entries"], _inputs().blocks)
    html = base._review_html("new-review", cards, task_id=extension.TASK_ID,
                             download_name="lightgen_class9_extension_ratings.json")
    assert f'taskId="{extension.TASK_ID}"' in html
    assert 'lightgen_class9_extension_ratings.json' in html
    assert '${cards.length} 条标签已导出' in html
    assert 'ed_prediction' not in html and 'proposal_id' not in html
