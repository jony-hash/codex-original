from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from scripts.research.lightgen import diagnose_lightgen_2d_generator_foreground_objective_gradient_v1 as audit


def _image_pair() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    target[:, :, 60:140, 80:120] = 0.8
    generated = target + 0.03
    generated[:, :, :20, :20] = 0.02
    return generated, target


def test_qfg_matches_independent_derivative_and_finite_difference() -> None:
    generated, target = _image_pair()
    generated.requires_grad_(True)
    q_fg = audit._qfg_components(generated, target)["q_fg"].sum()
    gradient, = torch.autograd.grad(q_fg, generated)
    reference = audit._qfg_gradient_reference(generated.detach(), target)
    assert torch.allclose(gradient, reference, rtol=1e-12, atol=1e-14)

    direction = torch.sin(torch.arange(generated.numel(), dtype=torch.float64)).reshape_as(generated)
    epsilon = 1e-5
    plus = audit._qfg_components(generated.detach() + epsilon * direction, target)["q_fg"].item()
    minus = audit._qfg_components(generated.detach() - epsilon * direction, target)["q_fg"].item()
    observed = (plus - minus) / (2 * epsilon)
    predicted = float((gradient * direction).sum())
    assert abs(observed - predicted) <= 1e-8


def test_qfg_is_scale_invariant_and_mass_derivative_cannot_be_detached() -> None:
    generated, target = _image_pair()
    values = audit._qfg_components(generated, target)
    scaled = audit._qfg_components(generated * 0.5, target)
    assert torch.equal(values["q_fg"], scaled["q_fg"])

    generated.requires_grad_(True)
    q_fg = audit._qfg_components(generated, target)["q_fg"].sum()
    gradient, = torch.autograd.grad(q_fg, generated)
    assert abs(float((gradient * generated).sum())) < 1e-12

    components = audit._qfg_components(generated.detach(), target)
    wrong = 2 * components["mask"] * (components["p_image"] - components["p_target"])
    wrong = wrong / components["denominator"][:, None] / generated.detach().sum()
    wrong = wrong.reshape_as(generated)
    assert abs(float((wrong * generated.detach()).sum())) > 1e-4


def _rows_from_images(candidate: torch.Tensor) -> list[dict[str, object]]:
    base, target = _image_pair()
    rows = []
    for scenario, image in (("baseline", base), ("base_plus", base), ("base_minus", base), ("candidate_plus", candidate), ("candidate_minus", base)):
        q_fg = float(audit._qfg_components(image, target)["q_fg"][0])
        for label in audit.FOCUS_CLASSES:
            for repeat in range(30):
                rows.append({"scenario": scenario, "role": "confirmation", "label": label, "q_fg": q_fg})
    return rows


def test_production_qfg_values_flow_to_local_improvement_decision() -> None:
    base, target = _image_pair()
    candidate = target.clone()
    rows = _rows_from_images(candidate)
    result = audit._decision(rows)
    assert result["classification"] == "CANDIDATE_LOCAL_RELATIVE_IMPROVEMENT"
    assert result["difference_candidate_minus_base_plus"] < -result["tolerance"]


def test_decision_distinguishes_regression_and_numerical_tie() -> None:
    base, _target = _image_pair()
    same_rows = _rows_from_images(base)
    same = audit._decision(same_rows)
    assert same["classification"] == "NO_DIFFERENCE_BEYOND_NUMERICAL_TOLERANCE"

    worse = base.clone()
    worse[:, :, 20:40, 20:40] = 0.3
    regression = audit._decision(_rows_from_images(worse))
    assert regression["classification"] == "CANDIDATE_LOCAL_RELATIVE_REGRESSION"


def test_qfg_rejects_zero_mass_and_nonfinite_inputs() -> None:
    generated, target = _image_pair()
    with torch.no_grad():
        zero = torch.zeros_like(generated)
    try:
        audit._qfg_components(zero, target)
    except audit.GradientAuditError as error:
        assert "zero intensity mass" in str(error)
    else:
        raise AssertionError("zero-mass images must be rejected")

    invalid = generated.clone()
    invalid[0, 0, 0, 0] = float("nan")
    try:
        audit._qfg_components(invalid, target)
    except audit.GradientAuditError as error:
        assert "nonfinite" in str(error)
    else:
        raise AssertionError("nonfinite images must be rejected")

    negative = generated.clone()
    negative[0, 0, 0, 0] = -1.0
    with pytest.raises(audit.GradientAuditError, match="negative intensity"):
        audit._qfg_components(negative, target)


def test_candidate_gradient_adds_only_the_declared_qfg_term() -> None:
    production = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    q_fg = torch.tensor([-0.25, 3.0, 2.0], dtype=torch.float64)

    assert torch.equal(audit._candidate_generator_gradient(production, q_fg, 0.0), production)
    weight = 0.4
    candidate = audit._candidate_generator_gradient(production, q_fg, weight)
    expected = production + audit.semantic.BETA * weight * q_fg
    assert torch.equal(candidate, expected)
    assert torch.equal(candidate - production, audit.semantic.BETA * weight * q_fg)


def test_primary_decision_excludes_fit_and_nonfocus_rows() -> None:
    base, target = _image_pair()
    rows = _rows_from_images(target)
    result = audit._decision(rows)
    extra_rows = rows + [
        {"scenario": "candidate_plus", "role": "fit", "label": 2, "q_fg": 1.0e6},
        {"scenario": "candidate_plus", "role": "confirmation", "label": 0, "q_fg": 1.0e6},
    ]
    assert audit._decision(extra_rows) == result


def test_foreground_gain_can_coexist_with_background_cost() -> None:
    target = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    target[0, 0, 80, 90] = 0.5
    target[0, 0, 80, 110] = 0.5
    baseline = torch.zeros_like(target)
    baseline[0, 0, 80, 90] = 0.4
    baseline[0, 0, 80, 110] = 0.3
    baseline[0, 0, 20, 20:30] = 0.03
    candidate = torch.zeros_like(target)
    candidate[0, 0, 80, 90] = 0.45
    candidate[0, 0, 80, 110] = 0.45
    candidate[0, 0, 20, 20] = 0.1

    base_metrics = audit._qfg_components(baseline, target)
    candidate_metrics = audit._qfg_components(candidate, target)
    assert candidate_metrics["q_fg"].item() < base_metrics["q_fg"].item()
    assert candidate_metrics["q_bg"].item() > base_metrics["q_bg"].item()

    rows = []
    for scenario in audit.SCENARIOS:
        image = candidate if scenario == "candidate_plus" else baseline
        q_fg = float(audit._qfg_components(image, target)["q_fg"][0])
        for label in audit.FOCUS_CLASSES:
            rows.extend({"scenario": scenario, "role": "confirmation", "label": label, "q_fg": q_fg} for _ in range(30))
    assert audit._decision(rows)["classification"] == "CANDIDATE_LOCAL_RELATIVE_IMPROVEMENT"


def _write_single_readout_chunk(tmp_path, monkeypatch, *, row_index: int, view_delta: float = 0.0) -> dict[str, object]:
    monkeypatch.setattr(audit, "SCENARIOS", ("baseline",))
    monkeypatch.setattr(audit, "TOTAL_ROWS", 1)
    monkeypatch.setattr(audit, "CHUNK_SIZE", 1)
    raw = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    raw[:, :, 50:150, 90:110] = 0.5
    view = audit.diagnostic.canonical.cmos_view(raw)
    view[:, :, 0, 0] += view_delta
    payload = {
        "scenario": "baseline",
        "positions": torch.tensor([0]),
        "source_row_indices": torch.tensor([row_index]),
        "labels": torch.tensor([2]),
        "roles": ["confirmation"],
        "raw_cmos": raw,
        "production_view": view,
        "e_d_logits": torch.zeros((1, 10), dtype=torch.float32),
    }
    chunk = tmp_path / "readouts" / "baseline" / "chunk_0000.pt"
    chunk.parent.mkdir(parents=True)
    torch.save(payload, chunk)
    return {
        "population": {"source_row_indices": [row_index]},
        "labels": torch.tensor([2]),
        "roles": ["confirmation"],
        "readout": {"target_slm_intensity": raw.clone()},
        "pixel_area_m2": 1.0,
    }


def test_saved_chunk_rejects_source_row_mismatch(tmp_path, monkeypatch) -> None:
    inputs = _write_single_readout_chunk(tmp_path, monkeypatch, row_index=99)
    inputs["population"]["source_row_indices"] = [100]
    with pytest.raises(audit.GradientAuditError, match="source row/label mapping differs"):
        audit._recheck_readout_chunks(tmp_path, inputs)


def test_saved_chunk_rejects_inconsistent_production_view(tmp_path, monkeypatch) -> None:
    inputs = _write_single_readout_chunk(tmp_path, monkeypatch, row_index=99, view_delta=0.1)
    with pytest.raises(audit.GradientAuditError, match="production view fails replay"):
        audit._recheck_readout_chunks(tmp_path, inputs)


def test_failed_phase_probe_restores_source_before_propagating_error() -> None:
    _generated, target = _image_pair()
    source = torch.tensor([0.0], dtype=torch.float64)
    phase = torch.nn.Parameter(source.clone())
    sensitivity = torch.zeros_like(target)
    sensitivity[0, 0, 20, 20] = 0.01

    class FailingRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def forward_from_image(self, _image: torch.Tensor) -> SimpleNamespace:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected finite-difference forward failure")
            raw = target + 0.03 + phase.reshape(1, 1, 1, 1) * sensitivity
            return SimpleNamespace(raw_intensity=raw, image_input=SimpleNamespace(slm_intensity=target))

    runtime = FailingRuntime()
    consumer = SimpleNamespace(runtime=runtime)
    with pytest.raises(RuntimeError, match="injected finite-difference forward failure"):
        audit._check_fd(
            consumer,
            {"generator.phase": phase},
            torch.zeros((1, 1, 28, 28), dtype=torch.float64),
            torch.ones_like(phase),
            epsilon=1.0e-3,
        )
    assert torch.equal(phase.detach(), source)
