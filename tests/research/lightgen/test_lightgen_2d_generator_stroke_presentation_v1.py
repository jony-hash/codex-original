from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from scripts.research.lightgen import audit_lightgen_2d_generator_stroke_presentation_v1 as audit
from scripts.research.lightgen import execute_lightgen_2d_generator_stroke_fidelity_comparison_v1 as stroke


def test_adaptive_pool_matches_independent_noninteger_window_reference() -> None:
    image = torch.arange(200 * 200, dtype=torch.float64).reshape(1, 1, 200, 200) / (200 * 200)
    actual = F.adaptive_avg_pool2d(image, (28, 28))[0, 0]
    expected = torch.tensor(audit._independent_adaptive_average(image[0, 0].tolist()), dtype=torch.float64)
    assert torch.allclose(actual, expected, rtol=0, atol=2e-15)


def test_clamp_order_and_quantization_are_separately_measured() -> None:
    raw = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    raw[:, :, :7, :7] = 2.0
    stages = audit._image_stages(raw)
    assert torch.equal(stages["production_clamp_then_pool"], F.adaptive_avg_pool2d(raw.clamp(0, 1), (28, 28)))
    assert torch.equal(stages["counterfactual_pool_then_clamp"], F.adaptive_avg_pool2d(raw, (28, 28)).clamp(0, 1))
    assert float((stages["production_clamp_then_pool"] - stages["counterfactual_pool_then_clamp"]).abs().max()) > 0.2
    loss_before_pool = F.adaptive_avg_pool2d(raw - raw.clamp(0, 1), (28, 28))
    assert torch.allclose(stages["raw_pool_unclipped"] - stages["production_clamp_then_pool"], loss_before_pool)
    assert float((stages["page_uint8_roundtrip"] - stages["production_clamp_then_pool"]).abs().max()) <= 0.5 / 255 + 1e-7


def test_review_html_pngs_match_production_quantization_and_hide_no_items() -> None:
    view = torch.zeros((1, 28, 28), dtype=torch.float32)
    view[:, 2:5, 8:12] = torch.tensor([[0.1, 0.25, 0.5, 1.0], [0.2, 0.3, 0.6, 0.9], [0.0, 0.4, 0.7, 0.8]])
    uri = stroke._image_data_uri(view[0])
    items = [{"item_id": f"item-{index}", "image": uri} for index in range(stroke.REVIEW_TOTAL)]
    html = stroke._review_html(items, "session")
    pngs, uris = audit._review_pngs(html)
    assert len(pngs) == stroke.REVIEW_TOTAL
    assert uris["item-0"] == uri
    with pytest.raises(audit.PresentationAuditError, match="item count"):
        audit._review_pngs(stroke._review_html(items[:-1], "session"))


def _synthetic_review_inputs() -> tuple[dict[str, object], dict[str, dict[str, torch.Tensor]], dict[str, object]]:
    labels = torch.tensor(list(range(10)) * 20 + list(range(10)) * 30, dtype=torch.long)
    rows = torch.arange(500, dtype=torch.long) + 10_000
    images = torch.zeros((500, 1, 28, 28), dtype=torch.float32)
    for index, digit in enumerate(labels.tolist()):
        images[index, 0, digit, digit] = 1.0
    raw = torch.zeros((500, 1, 200, 200), dtype=torch.float32)
    target = torch.nn.functional.interpolate(images, size=(200, 200), mode="bilinear", align_corners=False)
    logits = torch.zeros((500, 10), dtype=torch.float32)
    logits[torch.arange(500), labels] = 10.0
    z = torch.zeros((500, 100), dtype=torch.complex64)
    readout = {
        "source_indices": rows.clone(),
        "labels": labels.clone(),
        "raw_cmos": raw,
        "evaluator_view": torch.zeros((500, 1, 28, 28), dtype=torch.float32),
        "target_slm_intensity": target,
        "z": z,
        "e_s_logits": logits.clone(),
        "e_i_logits": logits.clone(),
        "e_c_logits": logits.clone(),
        "e_d_logits": logits,
    }
    readouts = {name: dict(readout) for name in audit.ENDPOINT_FILES}
    inputs = {"rows": rows, "labels": labels, "images": images}
    population = {"source_row_indices": rows.tolist(), "labels": labels.tolist(), "roles": ["fit"] * 200 + ["confirmation"] * 300}
    return inputs, readouts, population


def test_production_review_mapping_matches_saved_views_and_hidden_repeats(tmp_path) -> None:
    inputs, readouts, population = _synthetic_review_inputs()
    stroke._prepare_review(tmp_path, inputs, readouts)
    mapping = json.loads((tmp_path / "review_mapping.json").read_text(encoding="utf-8"))["items"]
    html = (tmp_path / "blind_review.html").read_text(encoding="utf-8")
    pngs, uris = audit._review_pngs(html)
    checks = audit._validate_population_and_readouts(population, readouts, mapping, pngs, uris)
    assert checks["primary_generated_pngs_match_saved_views"] == 600
    assert checks["hidden_repeat_pngs_pixel_identical"] == 30

    altered = dict(readouts["B5400"])
    altered["labels"] = altered["labels"].clone()
    altered["labels"][0] = (altered["labels"][0] + 1) % 10
    bad = {**readouts, "B5400": altered}
    with pytest.raises(audit.PresentationAuditError, match="labels differ"):
        audit._validate_population_and_readouts(population, bad, mapping, pngs, uris)

    wrong_rows = dict(readouts["B5400"])
    wrong_rows["source_indices"] = wrong_rows["source_indices"].clone()
    wrong_rows["source_indices"][0] += 1
    with pytest.raises(audit.PresentationAuditError, match="row mapping differs"):
        audit._validate_population_and_readouts(population, {**readouts, "B5400": wrong_rows}, mapping, pngs, uris)

    wrong_view = dict(readouts["B5400"])
    wrong_view["evaluator_view"] = wrong_view["evaluator_view"].clone()
    wrong_view["evaluator_view"][0, 0, 0, 0] = 0.5
    with pytest.raises(audit.PresentationAuditError, match="saved view differs"):
        audit._validate_population_and_readouts(population, {**readouts, "B5400": wrong_view}, mapping, pngs, uris)

    first_generated = next(row for row in mapping if row["kind"] == "primary_generated" and row["endpoint"] == "B")
    stretched = dict(uris)
    stretched[str(first_generated["item_id"])] = stroke._image_data_uri(torch.ones((1, 28, 28), dtype=torch.float32))
    with pytest.raises(audit.PresentationAuditError, match="review PNG differs"):
        audit._validate_population_and_readouts(population, readouts, mapping, pngs, stretched)
