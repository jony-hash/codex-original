from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.research.lightgen import audit_lightgen_g4_phase_reference_representation_v1 as audit


def _interleave_complex(values: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(values.to(torch.complex128)).reshape(values.shape[0], -1).contiguous()


def _example_features() -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.zeros((2, 100), dtype=torch.complex128)
    raw[0, 0] = 3.0 + 0.0j
    raw[0, 1] = 2.0 + 0.0j
    raw[1, 0] = 2.0 + 0.0j
    raw[1, 1] = 2.0 + 0.0j
    scale = torch.ones(100, dtype=torch.float64)
    scale[0] = 4.0
    features = raw / scale.sqrt().unsqueeze(0)
    return _interleave_complex(features), scale


def _tiny_endpoint(features: torch.Tensor, scale: torch.Tensor) -> audit.EndpointData:
    spec = audit.EndpointSpec("C_seed_2026101501", 2026101501,
                              Path("checkpoint.pt"), Path("cache.pt"), Path("priors.pt"))
    checkpoint = {"run_id": "test-run", "completed_global_step": 45_000}
    cache: dict[str, object] = {}
    priors: dict[str, object] = {}
    rows = torch.tensor([11, 22], dtype=torch.int64)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    assignments = torch.tensor([0, 2], dtype=torch.int64)
    return audit.EndpointData(
        spec=spec,
        checkpoint=checkpoint,
        cache=cache,
        priors=priors,
        rows=rows,
        labels=labels,
        features=features,
        assignments=assignments,
        scale=scale,
        class_counts=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        source_identity={},
    )


def _write_phase_rows(path: Path, endpoint: audit.EndpointData, metrics: dict[str, torch.Tensor]) -> None:
    arrays = audit._to_numpy(metrics)
    rows = list(audit._phase_output_rows(endpoint, arrays))
    audit._write_csv(path, audit.PHASE_ROW_FIELDS, rows)


def test_phase_reference_restores_scale_and_rejects_old_standardized_pivot() -> None:
    features, scale = _example_features()
    metrics = audit._phase_metrics_torch(features, scale)

    assert metrics["actual_raw_phase_reference_channel"].tolist() == [0, 0]
    assert metrics["standardized_feature_max_channel"].tolist() == [1, 1]
    assert metrics["raw_amplitude_gap"].tolist() == [1.0, 0.0]
    assert metrics["exact_top_tie"].tolist() == [False, True]

    # The previously used standardized maximum would be rejected by the output consumer.
    assert not np.array_equal(
        metrics["actual_raw_phase_reference_channel"].numpy(),
        metrics["standardized_feature_max_channel"].numpy(),
    )


def test_common_phase_preserves_recovered_amplitude_and_pivot() -> None:
    features, scale = _example_features()
    original = audit._phase_metrics_torch(features, scale)
    feature_complex = audit._feature_complex(features)
    phase = torch.exp(torch.tensor(0.731, dtype=torch.float64) * 1j)
    rotated = _interleave_complex(feature_complex * phase)
    changed = audit._phase_metrics_torch(rotated, scale)

    torch.testing.assert_close(original["raw_amplitude_max"], changed["raw_amplitude_max"])
    torch.testing.assert_close(original["raw_amplitude_second"], changed["raw_amplitude_second"])
    assert torch.equal(original["actual_raw_phase_reference_channel"],
                       changed["actual_raw_phase_reference_channel"])


def test_exact_tie_uses_first_channel() -> None:
    features, scale = _example_features()
    metrics = audit._phase_metrics_torch(features, scale)

    assert int(metrics["actual_raw_phase_reference_channel"][1]) == 0
    assert bool(metrics["exact_top_tie"][1])


def test_preflight_counterexamples_cover_scale_phase_tie_and_nmi() -> None:
    result = audit._validate_counterexamples(torch.device("cpu"))

    assert result["scale_counterexample"] == {
        "actual_raw_pivot": 0,
        "standardized_max_pivot": 1,
        "faulty_standardized_pivot_rejected": True,
    }
    assert result["common_phase_invariance"]
    assert result["exact_tie_first_channel"] == 0
    assert result["perfect_association_nmi"] == pytest.approx(1.0)
    assert result["independence_nmi"] == pytest.approx(0.0)


def test_torch_nmi_matches_independent_numpy_formula() -> None:
    counts = torch.tensor([[4, 0, 1], [0, 3, 2]], dtype=torch.int64)

    actual = float(audit._nmi_from_counts_torch(counts))
    expected = audit._numpy_nmi(counts.numpy())

    assert abs(actual - expected) <= audit.NMI_TOLERANCE
    perfect = torch.tensor([[5, 0], [0, 5]], dtype=torch.int64)
    independent = torch.tensor([[5, 5], [5, 5]], dtype=torch.int64)
    assert float(audit._nmi_from_counts_torch(perfect)) == pytest.approx(1.0, abs=audit.NMI_TOLERANCE)
    assert float(audit._nmi_from_counts_torch(independent)) == pytest.approx(0.0, abs=audit.NMI_TOLERANCE)


def test_generation_population_consumer_rejects_missing_group_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "SAMPLES_PER_CLASS_STREAM", 1)
    groups = list(itertools.product(range(3), range(2), range(3), range(10)))
    arrays = {
        "model": np.asarray([item[0] for item in groups], dtype=np.int8),
        "method": np.asarray([item[1] for item in groups], dtype=np.int8),
        "stream": np.asarray([item[2] for item in groups], dtype=np.int8),
        "label": np.asarray([item[3] for item in groups], dtype=np.int8),
        "proposal": np.zeros(len(groups), dtype=np.int16),
        "component": np.zeros(len(groups), dtype=np.int16),
        "ed_correct": np.zeros(len(groups), dtype=np.int8),
        "ec_correct": np.zeros(len(groups), dtype=np.int8),
    }

    audit._validate_generation_population(arrays)
    arrays["label"][0] = 1
    with pytest.raises(audit.AuditError, match="missing or extra model/method/stream/class group"):
        audit._validate_generation_population(arrays)


def test_generation_preflight_selects_first_two_rows_per_saved_group() -> None:
    rows = list(itertools.product(range(3), range(2), range(3), range(10), range(3)))
    model = np.asarray([row[0] for row in rows], dtype=np.int8)
    method = np.asarray([row[1] for row in rows], dtype=np.int8)
    stream = np.asarray([row[2] for row in rows], dtype=np.int8)
    label = np.asarray([row[3] for row in rows], dtype=np.int8)
    proposal = np.asarray([row[4] for row in rows], dtype=np.int16)
    component = np.asarray([0 if row[1] == 0 else row[4] % 4 for row in rows], dtype=np.int16)
    zeros = np.zeros(len(rows), dtype=np.int8)
    metrics = {name: np.arange(len(rows), dtype=np.float64) for name in (
        "E_D_requested_margin", "E_C_requested_margin", "E_D_top_margin", "E_C_top_margin",
        "proxy_clipping_area_fraction", "proxy_clipped_excess_fraction",
    )}
    data = audit.GenerationData(
        model=model,
        stream=stream,
        method=method,
        label=label,
        component=component,
        proposal=proposal,
        ed_prediction=label.copy(),
        ec_prediction=label.copy(),
        ed_correct=np.ones(len(rows), dtype=np.int8),
        ec_correct=np.ones(len(rows), dtype=np.int8),
        metrics=metrics,
    )

    selected = audit._select_generation_preflight(data)

    assert selected.model.size == 3 * 2 * 3 * 10 * 2
    assert set(selected.proposal.tolist()) == {0, 1}
    selected_groups = np.stack((selected.model, selected.method, selected.stream, selected.label), axis=1)
    assert np.unique(selected_groups, axis=0).shape[0] == 3 * 2 * 3 * 10
    for group in np.unique(selected_groups, axis=0):
        mask = np.all(selected_groups == group, axis=1)
        assert int(mask.sum()) == 2


def test_phase_output_consumer_rejects_wrong_endpoint_and_row_and_missing_statistic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "EXPECTED_ROWS", 2)
    features, scale = _example_features()
    endpoint = _tiny_endpoint(features, scale)
    metrics = audit._phase_metrics_torch(features, scale)
    expected = audit._to_numpy(metrics)
    path = tmp_path / "phase.csv"
    _write_phase_rows(path, endpoint, metrics)

    assert audit._verify_phase_output(path, endpoint, expected) == 2

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    faulty_rows = [dict(row) for row in rows]
    faulty_rows[0]["actual_raw_phase_reference_channel"] = faulty_rows[0]["standardized_feature_max_channel"]
    audit._write_csv(path, audit.PHASE_ROW_FIELDS, faulty_rows)
    with pytest.raises(audit.AuditError, match="wrong actual pivot"):
        audit._verify_phase_output(path, endpoint, expected)

    audit._write_csv(path, audit.PHASE_ROW_FIELDS, rows)
    rows[0]["endpoint"] = "seed_2026102151"
    audit._write_csv(path, audit.PHASE_ROW_FIELDS, rows)
    with pytest.raises(audit.AuditError, match="wrong endpoint"):
        audit._verify_phase_output(path, endpoint, expected)

    rows[0]["endpoint"] = endpoint.spec.name
    rows[0]["source_row"] = "999"
    audit._write_csv(path, audit.PHASE_ROW_FIELDS, rows)
    with pytest.raises(audit.AuditError, match="wrong source row"):
        audit._verify_phase_output(path, endpoint, expected)

    fields = tuple(name for name in audit.PHASE_ROW_FIELDS if name != "actual_raw_phase_reference_channel")
    missing_column_rows = [
        {name: row[name] for name in fields}
        for row in rows
    ]
    audit._write_csv(path, fields, missing_column_rows)
    with pytest.raises(audit.AuditError, match="renamed statistic columns"):
        audit._verify_phase_output(path, endpoint, expected)


def test_generation_total_consumer_rejects_evaluator_metric_mixup(tmp_path: Path) -> None:
    model = audit.ENDPOINT_NAMES[0]
    expected = [{
        "model": model,
        "method": "G1",
        "population": 2,
        "samples_per_class": 1,
        "E_D_correct_count": 1,
        "E_D_macro_recall": 0.5,
        "E_C_correct_count": 2,
        "E_C_macro_recall": 1.0,
        "macro_E_D_requested_margin": 0.2,
        "macro_E_C_requested_margin": 0.3,
        "macro_E_D_top_margin": 0.4,
        "macro_E_C_top_margin": 0.5,
        "macro_proxy_clipping_area_fraction": 0.01,
        "macro_proxy_clipped_excess_fraction": 0.02,
    }]
    history = {"endpoints": {model: {"generation": {"G1": {"three_stream_equal_weight": {
        "E_D_macro_recall": 0.5,
        "E_C_macro_recall": 1.0,
        "macro_proxy_clipping_area_fraction": 0.01,
        "macro_proxy_clipped_excess_fraction": 0.02,
    }}}}}}
    path = tmp_path / "totals.csv"
    row = dict(expected[0])
    row["E_D_correct_count"], row["E_C_correct_count"] = 2, 1
    audit._write_csv(path, tuple(expected[0]), [row])

    with pytest.raises(audit.AuditError, match="wrong E_D count"):
        audit._verify_generation_output_consumer(path, expected, history)
