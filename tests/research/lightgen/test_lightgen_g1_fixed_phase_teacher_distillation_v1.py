from __future__ import annotations

import copy
import json

import pytest
import torch

from scripts.research.lightgen import (
    execute_lightgen_g1_fixed_phase_teacher_distillation_v1 as teacher,
    execute_lightgen_g1_slm_projected_mapping_v2 as projected,
)


def test_registered_teacher_distillation_contract_is_fixed() -> None:
    config = teacher._config(teacher.DEFAULT_CONFIG)

    assert config["population"]["train_per_class"] == 200
    assert config["population"]["development_per_class"] == 20
    assert config["population"]["holdout_per_class"] == 200
    assert config["teacher"]["phase_rad"] == 0.0
    assert config["training"]["new_updates"] == 15_000
    assert config["training"]["teacher_lambda"] == 1.0
    assert config["training"]["box_lambda"] == 5.0


def test_registered_config_rejects_a_changed_population_seed(monkeypatch) -> None:
    config = teacher._config(teacher.DEFAULT_CONFIG)
    config["population"]["holdout_seed_base"] += 1
    monkeypatch.setattr(teacher.yaml, "safe_load", lambda _: config)

    with pytest.raises(ValueError, match="registered fixed task contract changed: population"):
        teacher._config(teacher.DEFAULT_CONFIG)


def test_preflight_repair_selection_is_followed_without_overwriting_failed_evidence(tmp_path) -> None:
    config = teacher._config(teacher.DEFAULT_CONFIG)
    runs = tmp_path / "runs"
    base = runs / "preflight_01"
    repair = runs / "preflight_repair_02"
    base.mkdir(parents=True)
    repair.mkdir()
    (base / "preflight_summary.json").write_text(
        json.dumps({"task_id": teacher.TASK_ID, "status": "PREFLIGHT_BLOCKED"}),
        encoding="utf-8")
    config["output"] = {
        "runs_root": str(runs),
        "source_dir": str(runs / "source_inputs_01"),
        "population_dir": str(runs / "population_01"),
        "teacher_dir": str(runs / "teacher_01"),
        "preflight_dir": str(base),
        "attempt_dir": str(runs / "attempt_01"),
        "verification_dir": str(runs / "windows_verify_01"),
        "windows_backup_dir": str(runs / "autodl_backup_attempt_01"),
    }
    selected = runs / "preflight_selection.json"
    selected.write_text(json.dumps({
        "task_id": teacher.TASK_ID,
        "selected_path": "preflight_repair_02",
        "status": "READY",
    }), encoding="utf-8")
    for key in ("source_dir", "population_dir", "teacher_dir", "attempt_dir"):
        teacher._path(config["output"][key]).mkdir(parents=True, exist_ok=True)
    (repair / "ready_receipt.json").write_text("{}", encoding="utf-8")

    roots = teacher._roots(config)

    assert roots["preflight_dir"] == repair
    assert teacher._next_preflight_dir(config) == runs / "preflight_repair_03"
    assert json.loads(selected.read_text(encoding="utf-8"))["selected_path"] == "preflight_repair_02"
    manifest = teacher._write_transfer_manifest(config)
    assert "preflight_selection.json" in manifest["files"]
    assert "preflight_repair_02/ready_receipt.json" in manifest["files"]


def test_population_proposal_identity_is_scoped_by_class(monkeypatch) -> None:
    labels = torch.arange(10).repeat_interleave(2)
    coefficients = torch.zeros((20, 100), dtype=torch.complex128)
    coefficients[:, 0] = 1.0e-3
    cohort = {
        "labels": labels,
        "proposal_ids": torch.tensor([0, 1] * 10),
        "coefficients": coefficients,
    }
    monkeypatch.setattr(teacher.fixed, "_cohort", lambda *args, **kwargs: (cohort, {}))

    sampled, _ = teacher._sample(
        {"population": {}}, {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}},
        {"scale": torch.ones(100)}, role="fixed_train2000", per_class=2, seed_base=1)

    assert sampled["proposal_ids"].tolist() == [0, 1] * 10


def test_independent_numpy_metrics_match_production_metric_definition() -> None:
    target = torch.tensor([[0.5 + 0.25j, -0.2 + 0.4j]], dtype=torch.complex128)
    predicted = target * torch.exp(torch.tensor(0.31j, dtype=torch.complex128)) * 0.97
    raw_reference = torch.tensor([[[[0.3, 0.7], [0.4, 0.6]]]], dtype=torch.float64)
    raw_actual = raw_reference * 0.95
    outputs = {
        2048: {"coefficients": predicted, "raw_cmos": raw_actual},
        4096: {"coefficients": predicted * 1.001, "raw_cmos": raw_actual * 1.002},
    }
    references = {2048: raw_reference, 4096: raw_reference * 1.001}

    expected = projected._error_row(outputs, target, references, 0)
    actual = teacher._numpy_metric_row(outputs, target, references)

    assert set(actual) == set(teacher.METRIC_KEYS)
    assert actual == pytest.approx(expected, abs=1.0e-14, rel=0.0)


def _small_linear_inputs() -> dict[str, torch.Tensor]:
    matrix = torch.zeros((200, 40_000), dtype=torch.float64)
    matrix[torch.arange(200), torch.arange(200)] = 1.0
    paired = matrix.reshape(100, 2, 40_000).permute(0, 2, 1).contiguous()
    response = torch.view_as_complex(paired)
    return {"matrix": matrix, "inverse": matrix.T.contiguous(),
            "scale": torch.ones(100, dtype=torch.float64),
            "response_matrix": response}


def test_teacher_term_changes_the_actual_mapper_update() -> None:
    torch.manual_seed(202610300701)
    initial = projected.AmplitudeMapper().state_dict()
    target = torch.zeros((1, 100), dtype=torch.complex128)
    target[0, 0] = 0.3 + 0.2j
    inputs = _small_linear_inputs()
    config = {"training": {"box_lambda": 5.0, "teacher_lambda": 1.0}}
    zero_teacher = torch.zeros((1, 40_000), dtype=torch.float32)
    one_teacher = torch.ones((1, 40_000), dtype=torch.float32)
    final_states = []
    losses = []

    for teacher_command in (zero_teacher, one_teacher):
        mapper = projected.AmplitudeMapper()
        mapper.load_state_dict(copy.deepcopy(initial))
        optimizer = torch.optim.Adam(mapper.parameters(), lr=1.0e-4,
                                     betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0)
        record = teacher._update(mapper, optimizer, target, teacher_command, inputs,
                                 model=None, device=torch.device("cpu"),
                                 engine="response_matrix", config=config, audit=False)
        losses.append(record["teacher_loss"])
        final_states.append({key: value.detach().clone() for key, value in mapper.state_dict().items()})

    assert losses[0] != losses[1]
    assert teacher.fixed._compare_states(final_states[0], final_states[1]) > 0.0
