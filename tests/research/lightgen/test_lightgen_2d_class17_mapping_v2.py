from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research.lightgen import canonical_semantics
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic


def _population() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    train_rows, _ = diagnostic.stratified_mnist_indices(labels, seed=diagnostic.SPLIT_SEED, validation_per_label=500)
    by_class = [train_rows[labels.index_select(0, train_rows) == class_id] for class_id in range(10)]
    indices = torch.stack(
        [
            torch.stack([by_class[class_id][offset] for class_id in range(10)])
            for offset in range(30)
        ]
    ).flatten()
    selected_labels = labels.index_select(0, indices)
    return indices, selected_labels, labels


def test_fixed_population_requires_train_rows_and_disjoint_class_balanced_splits() -> None:
    indices, labels, source_labels = _population()
    result = diagnostic._validate_population(indices, labels, indices.clone(), labels.clone(), source_labels)
    assert result["fit_class_counts"] == [20] * 10
    assert result["observation_class_counts"] == [10] * 10
    assert result["fit_observation_disjoint"] is True

    leaked = indices.clone()
    _train_rows, validation_rows = diagnostic.stratified_mnist_indices(
        source_labels,
        seed=diagnostic.SPLIT_SEED,
        validation_per_label=500,
    )
    class_zero_validation = validation_rows[source_labels.index_select(0, validation_rows) == 0][0]
    class_zero_position = torch.nonzero(labels == 0, as_tuple=False).flatten()[0]
    leaked[class_zero_position] = class_zero_validation
    with pytest.raises(diagnostic.DiagnosticError, match="non-TRAIN"):
        diagnostic._validate_population(leaked, labels, leaked.clone(), labels, source_labels)


def test_population_rejects_endpoint_and_fit_observation_mismatch() -> None:
    indices, labels, source_labels = _population()
    other_indices = indices.clone()
    other_indices[-1] += 10
    with pytest.raises(diagnostic.DiagnosticError, match="source-row order"):
        diagnostic._validate_population(indices, labels, other_indices, labels, source_labels)

    train_rows, _validation_rows = diagnostic.stratified_mnist_indices(
        source_labels,
        seed=diagnostic.SPLIT_SEED,
        validation_per_label=500,
    )
    unbalanced_indices = indices.clone()
    class_one_unused = train_rows[source_labels.index_select(0, train_rows) == 1][-1]
    unbalanced_indices[0] = class_one_unused
    unbalanced_labels = source_labels.index_select(0, unbalanced_indices)
    with pytest.raises(diagnostic.DiagnosticError, match="20-per-class"):
        diagnostic._validate_population(unbalanced_indices, unbalanced_labels, unbalanced_indices, unbalanced_labels, source_labels)


def test_named_step_checkpoints_and_fixed_rows_load_from_explicit_paths() -> None:
    root = diagnostic._root()
    paths = {name: diagnostic._resolve(root, relative) for name, relative in diagnostic.DEFAULTS.items()}
    loaded = diagnostic.load_inputs(root, paths)
    assert loaded["row_contract"]["count"] == 300
    assert loaded["row_contract"]["canonical_train_only"] is True
    assert loaded["payloads"]["step1200"]["optimizer_step_vector"] == [1200, 1200]
    assert loaded["payloads"]["step2200"]["optimizer_step_vector"] == [2200, 2200]
    assert loaded["payloads"]["step2200"].get("references") is None
    assert loaded["payloads"]["step2200"].get("weights") is None
    assert loaded["weights"]["alpha"] == loaded["payloads"]["step1200"]["weights"]["alpha"]
    assert loaded["images"].shape == (300, 1, 28, 28)
    assert torch.isfinite(loaded["images"]).all()


def test_manual_adam_candidates_match_torch_adam_on_toy_parameters() -> None:
    parameters = {
        "encoder.phase": torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float32)),
        "generator.phase": torch.nn.Parameter(torch.tensor([-0.3, 0.5], dtype=torch.float32)),
    }
    optimizer = torch.optim.Adam(
        [parameters["encoder.phase"], parameters["generator.phase"]],
        lr=diagnostic.EXPECTED_ADAM["learning_rate"],
        betas=tuple(diagnostic.EXPECTED_ADAM["betas"]),
        eps=diagnostic.EXPECTED_ADAM["eps"],
        weight_decay=diagnostic.EXPECTED_ADAM["weight_decay"],
        amsgrad=diagnostic.EXPECTED_ADAM["amsgrad"],
    )
    gradients = {
        "encoder.phase": torch.tensor([0.1, -0.2], dtype=torch.float32),
        "generator.phase": torch.tensor([-0.05, 0.3], dtype=torch.float32),
    }
    for index, name in enumerate(("encoder.phase", "generator.phase")):
        optimizer.state[parameters[name]] = {
            "step": torch.tensor(4.0),
            "exp_avg": torch.tensor([0.01, -0.03], dtype=torch.float32) + index * 0.002,
            "exp_avg_sq": torch.tensor([0.02, 0.04], dtype=torch.float32) + index * 0.003,
        }
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    actual = diagnostic._adam_candidate(gradients, optimizer, parameters)

    reference_parameters = {
        "encoder.phase": torch.nn.Parameter(before["encoder.phase"].clone()),
        "generator.phase": torch.nn.Parameter(before["generator.phase"].clone()),
    }
    reference_optimizer = torch.optim.Adam(
        [reference_parameters["encoder.phase"], reference_parameters["generator.phase"]],
        lr=diagnostic.EXPECTED_ADAM["learning_rate"],
        betas=tuple(diagnostic.EXPECTED_ADAM["betas"]),
        eps=diagnostic.EXPECTED_ADAM["eps"],
        weight_decay=diagnostic.EXPECTED_ADAM["weight_decay"],
        amsgrad=diagnostic.EXPECTED_ADAM["amsgrad"],
    )
    reference_optimizer.load_state_dict(optimizer.state_dict())
    for name, parameter in reference_parameters.items():
        parameter.grad = gradients[name].clone()
    reference_optimizer.step()
    for name, parameter in reference_parameters.items():
        torch.testing.assert_close(actual[name], parameter.detach() - before[name], rtol=1e-6, atol=3e-8)

    fresh = diagnostic._adam_candidate(gradients, optimizer, parameters, zero_history=True)
    fresh_optimizer = torch.optim.Adam(
        [torch.nn.Parameter(before["encoder.phase"].clone()), torch.nn.Parameter(before["generator.phase"].clone())],
        lr=diagnostic.EXPECTED_ADAM["learning_rate"],
        betas=tuple(diagnostic.EXPECTED_ADAM["betas"]),
        eps=diagnostic.EXPECTED_ADAM["eps"],
        weight_decay=diagnostic.EXPECTED_ADAM["weight_decay"],
        amsgrad=diagnostic.EXPECTED_ADAM["amsgrad"],
    )
    for parameter, name in zip(fresh_optimizer.param_groups[0]["params"], ("encoder.phase", "generator.phase"), strict=True):
        parameter.grad = gradients[name].clone()
    fresh_before = [parameter.detach().clone() for parameter in fresh_optimizer.param_groups[0]["params"]]
    fresh_optimizer.step()
    for index, name in enumerate(("encoder.phase", "generator.phase")):
        torch.testing.assert_close(fresh[name], fresh_optimizer.param_groups[0]["params"][index].detach() - fresh_before[index], rtol=1e-6, atol=3e-8)


def test_max_q_gauge_removes_only_global_phase_and_cmos_views_expose_clipping() -> None:
    generator = torch.Generator().manual_seed(17)
    z = torch.complex(
        torch.randn((12, 100), generator=generator),
        torch.randn((12, 100), generator=generator),
    )
    shifted = z * torch.exp(torch.tensor(0.71j))
    features = diagnostic._probe_features(z, torch.zeros((12, 1, 200, 200)), torch.ones(100))
    shifted_features = diagnostic._probe_features(shifted, torch.zeros((12, 1, 200, 200)), torch.ones(100))
    torch.testing.assert_close(features["ols_gauge"], shifted_features["ols_gauge"], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(features["ols_raw"], shifted_features["ols_raw"])

    raw = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    raw[..., 0, 0] = 1.4
    pooled_raw, evaluator_view = diagnostic._output_views(raw)
    assert pooled_raw.max() > evaluator_view.max()
    assert float((raw > 1.0).float().mean()) > 0.0
    torch.testing.assert_close(evaluator_view.max(), torch.tensor(1.0 / 64))


def test_fixed_prediction_permutation_holm_and_conservative_support() -> None:
    labels = torch.arange(10).repeat(10)
    predictions = torch.zeros(100, dtype=torch.long)
    class_one_rows = torch.nonzero(labels == 1, as_tuple=False).flatten()
    predictions[class_one_rows[:5]] = 1
    predictions[torch.nonzero(labels != 1, as_tuple=False).flatten()[:5]] = 1
    p_value = diagnostic._exact_fixed_prediction_pvalue(labels, predictions, 1)
    assert 0.0 < p_value < 0.05
    adjusted = diagnostic._holm_adjust({"a": p_value, "b": 0.04, "c": 0.5})
    assert adjusted["a"] >= p_value
    assert adjusted["c"] == 0.5

    metrics = {
        "per_class_correct": [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        "macro_recall": 0.50,
    }
    negative = {"macro_recall": 0.20}
    assert diagnostic._probe_support(metrics, negative, 0.05, 1) is True
    assert diagnostic._probe_support(metrics, negative, 0.051, 1) is False
    assert diagnostic._ed_support(metrics, 0.05, 1) is True


def test_zero_gradient_is_undecidable_and_snapshot_detects_state_change() -> None:
    zeros = {
        key: torch.zeros((2,), dtype=torch.float32)
        for key in ("encoder.phase", "generator.phase")
    }
    semantic, reason = diagnostic._safe_canonical_gradient(
        {"L_S": zeros, "L_I": zeros, "L_C": zeros}
    )
    assert semantic is None
    assert reason is not None
    assert diagnostic._direction_record(torch.zeros(4), torch.ones(4))["status"] == "UNDECIDABLE_ZERO_GRADIENT"

    module = torch.nn.Linear(3, 2)
    frozen = diagnostic._state_snapshot(module)
    assert diagnostic._state_equal(frozen, diagnostic._state_snapshot(module))
    with torch.no_grad():
        module.weight[0, 0] += 1.0
    assert not diagnostic._state_equal(frozen, diagnostic._state_snapshot(module))


def test_resume_accepts_only_matching_population_and_ready_preflight(tmp_path: Path) -> None:
    population = {"task_id": diagnostic.TASK_ID, "source_row_indices": [2, 3], "labels": [1, 7]}
    (tmp_path / "input_population.json").write_text(json.dumps(population), encoding="utf-8")
    incomplete = diagnostic._validate_resume(tmp_path, population)
    assert incomplete["complete"] is False
    assert incomplete["preflight"] is None
    assert incomplete["preflight_artifact"] == "preflight.json"
    (tmp_path / "preflight.json").write_text(json.dumps({"task_id": diagnostic.TASK_ID, "status": "READY"}), encoding="utf-8")
    ready = diagnostic._validate_resume(tmp_path, population)
    assert ready["complete"] is False
    assert ready["preflight"]["status"] == "READY"
    assert ready["preflight_artifact"] == "preflight.json"
    with pytest.raises(diagnostic.DiagnosticError, match="rows, labels"):
        diagnostic._validate_resume(tmp_path, {**population, "labels": [1, 1]})
    (tmp_path / "report.json").write_text(json.dumps({"task_id": diagnostic.TASK_ID, "status": "VALID_COMPLETED"}), encoding="utf-8")
    assert diagnostic._validate_resume(tmp_path, population)["complete"] is True


def test_probe_view_helper_matches_canonical_evaluator_view() -> None:
    raw = torch.rand((3, 1, 200, 200), generator=torch.Generator().manual_seed(4))
    views = diagnostic._probe_features(
        torch.complex(torch.ones((3, 100)), torch.zeros((3, 100))),
        raw,
        torch.ones(100),
    )
    expected = canonical_semantics.cmos_view(raw).flatten(1).to(torch.float64)
    torch.testing.assert_close(views["cmos_evaluator_view"], expected)


def test_probe_report_view_converts_saved_predictions_to_json_values() -> None:
    payload = {
        "fit_rows": diagnostic.FIT_ROWS,
        "observation_rows": diagnostic.OBSERVATION_ROWS,
        "records": {},
        "predictions": {"ols_gauge": torch.tensor([1, 7, 1], dtype=torch.long)},
    }

    view = diagnostic._stage_probe_view(payload)
    assert view["predictions"] == {"ols_gauge": [1, 7, 1]}
    json.dumps(view)
