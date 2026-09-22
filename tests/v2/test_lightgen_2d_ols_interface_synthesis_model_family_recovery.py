"""Focused checks for zero-forward N4 report recovery and terminal sampler handling."""

from __future__ import annotations

import pytest

from scripts.research import (
    execute_lightgen_2d_ols_interface_synthesis_model_family_comparison_v1 as execution,
)


def test_terminal_sampler_resume_validation_does_not_construct_or_consume_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_constructed(*args: object, **kwargs: object) -> None:
        raise AssertionError("terminal sampler validation must not construct or consume a sampler")

    monkeypatch.setattr(execution.V10, "DeterministicTrainRoleSampler", fail_if_constructed)

    execution._validate_sampler_resume(
        {},
        {"cursor": execution.SCIENTIFIC_STEPS, "steps": execution.SCIENTIFIC_STEPS},
        step=execution.SCIENTIFIC_STEPS,
    )


def test_terminal_sampler_resume_validation_rejects_nonexhausted_state() -> None:
    with pytest.raises(execution.InvalidN4, match="final sampler cursor is not exhausted"):
        execution._validate_sampler_resume(
            {},
            {"cursor": execution.SCIENTIFIC_STEPS - 1, "steps": execution.SCIENTIFIC_STEPS},
            step=execution.SCIENTIFIC_STEPS,
        )
