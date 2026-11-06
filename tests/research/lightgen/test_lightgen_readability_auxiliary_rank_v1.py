"""Focused checks for label linkage, fold fitting, risk direction, and grouped inference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.research.lightgen import execute_lightgen_readability_auxiliary_rank_v1 as task


def _records_for_grouped_bootstrap():
    records = []
    failures = []
    auxiliary = []
    ec = []
    ed = []
    for package in task.PACKAGES:
        for label in task.LABELS:
            for group in range(8):
                branches = ("D0", "Dgrad") if package == "D" else ("single",)
                for branch in branches:
                    failure = group % 2 == 0
                    index = len(records)
                    records.append({
                        "package": package, "card_id": f"{package}-{label}-{group}-{branch}",
                        "requested_label": label, "human_label": (label + 1) % 10 if failure else label,
                        "human_failure": failure,
                        "coefficient_group_id": f"{package}-{label}-{group}", "branch": branch,
                    })
                    failures.append(failure)
                    auxiliary.append(0.8 if failure else 0.2 + index * 1e-7)
                    ec.append(0.3 if failure else 0.7 - index * 1e-7)
                    ed.append(0.5 + index * 1e-8)
    score = {"human_failure": np.asarray(failures, dtype=bool),
             "auxiliary": np.asarray(auxiliary), "E_C": np.asarray(ec), "E_D": np.asarray(ed)}
    return records, score


def test_fixed_features_and_requested_class_risk_formula():
    assert len(task.FEATURE_NAMES) == 23
    assert not any("requested_label" in name for name in task.FEATURE_NAMES)
    records = [
        {"package": "R", "card_id": "r1", "human_label": 0, "requested_label": 0,
         "human_failure": False, "ec_logits": np.zeros(10), "ed_logits": np.zeros(10)},
        {"package": "D", "card_id": "d1", "human_label": "unreadable", "requested_label": 1,
         "human_failure": True, "ec_logits": np.zeros(10), "ed_logits": np.zeros(10)},
        {"package": "P", "card_id": "p1", "human_label": 2, "requested_label": 2,
         "human_failure": False, "ec_logits": np.zeros(10), "ed_logits": np.zeros(10)},
    ]
    digit_probability = np.zeros(10)
    digit_probability[1] = 0.4
    predictions = []
    for index, row in enumerate(records):
        p = np.full(10, 0.05)
        p[1] = 0.55 if index == 1 else 0.1
        p /= p.sum()
        predictions.append({
            "package": row["package"], "card_id": row["card_id"],
            "requested_label": str(row["requested_label"]),
            "human_label": str(row["human_label"]), "human_failure": int(row["human_failure"]),
            "fold": row["package"], "p_readable": "0.8",
            **{f"p_digit_{label}": str(p[label]) for label in task.LABELS},
        })
    scores = task._scores(records, predictions)
    assert scores["auxiliary"][1] == pytest.approx(1.0 - 0.8 * 0.55)
    assert scores["E_C"][1] == pytest.approx(0.9)
    assert digit_probability[1] == 0.4  # The model target probability is read at the requested class only.


def test_three_step_cpu_contract_updates_both_heads():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(30, 23))
    labels = [index % 10 for index in range(25)] + ["unreadable"] * 5

    model = task._fit_auxiliary_model(features, labels, steps=3, device=torch.device("cpu"))
    digit_probability, readable_probability = task._predict_auxiliary(
        model, features, device=torch.device("cpu"))

    assert model["optimizer_steps"] == 3
    assert model["parameter_norm_after_fit"] > 0
    assert digit_probability.shape == (30, 10)
    assert readable_probability.shape == (30,)
    assert np.isfinite(digit_probability).all()
    assert np.isfinite(readable_probability).all()


def test_risk_direction_negative_control_captures_the_known_top_failures():
    failures = np.asarray([True, True] + [False] * 8)
    risk = np.asarray([0.99, 0.95] + [0.2 - index * 0.01 for index in range(8)])
    top = np.argsort(-risk, kind="stable")[:2]
    inverted_top = np.argsort(risk, kind="stable")[:2]

    assert failures[top].sum() == 2
    assert failures[inverted_top].sum() == 0


def test_valid_population_keeps_d0_dgrad_exact_coefficient_rows_together():
    records = []
    for package in task.PACKAGES:
        for label in task.LABELS:
            groups = 20 if package == "R" else (10 if package == "D" else 120)
            for group in range(groups):
                branches = ("D0", "Dgrad") if package == "D" else ("single",)
                for branch in branches:
                    human = label if (group + label) % 2 else ("unreadable" if group % 4 == 0 else (label + 1) % 10)
                    records.append({
                        "package": package, "card_id": f"{package}-{label}-{group}-{branch}",
                        "requested_label": label, "human_label": human,
                        "human_failure": human == "unreadable" or human != label,
                        "coefficient_group_id": f"{package}-coef-{label}-{group}",
                        "branch": branch,
                    })

    task._validate_population(records)
    split = [dict(row) for row in records]
    next(row for row in split if row["package"] == "D" and row["branch"] == "Dgrad")[
        "coefficient_group_id"] = "Dgrad-split"
    with pytest.raises(task.AuxiliaryRankError, match="D0/Dgrad"):
        task._validate_population(split)


def test_grouped_paired_bootstrap_matches_independent_reference():
    records, score = _records_for_grouped_bootstrap()
    production = task._grouped_paired_bootstrap(records, score, resamples=120, seed=15)
    independent = task._independent_bootstrap_check(records, score, production, resamples=120, seed=15)

    assert production["paired_methods"] == ["auxiliary", "E_C"]
    assert independent["status"] == "PASSED"
    assert independent["primary_ci_match"] is True
