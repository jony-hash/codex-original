"""Tests for the post-step-492 candidate S3 consumer boundary."""

from __future__ import annotations

import pytest
import torch

from ldm_project.v2.student.identity_generator import IdentityAnchoredConditionalAmplitudeGenerator
from scripts.research.capacity import run_frozen_s3_on_controller_semantic_geometry_candidate as candidate_runner
from scripts.research.scaled_chen import run_scaled_chen_controller_to_ols_writability_audit as s3


def _candidate_payload() -> dict[str, object]:
    generator = IdentityAnchoredConditionalAmplitudeGenerator()
    return {
        "schema": "controller_semantic_geometry_final_controller_state.v1",
        "objective_family": candidate_runner.training.OBJECTIVE_FAMILY,
        "checkpoint_selection": "FINAL_AUTHORIZED_STEP_ONLY",
        "step": candidate_runner.training.HORIZON,
        "training_authorization": candidate_runner.training.AUTHORIZATION_TOKEN,
        "training_authorized": True,
        "formal_promotion": False,
        "s3_classification": "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED",
        "controller_state": {
            name: value.detach().clone() for name, value in generator.state_dict().items()
        },
    }


def test_candidate_state_loader_accepts_exact_generator_state() -> None:
    state = candidate_runner._load_candidate_state_payload(_candidate_payload())

    assert set(state) == set(IdentityAnchoredConditionalAmplitudeGenerator().state_dict())
    assert all(torch.isfinite(value).all() for value in state.values())


def test_candidate_state_loader_rejects_non_generator_state_key() -> None:
    payload = _candidate_payload()
    state = dict(payload["controller_state"])
    state["system.encoder.phase"] = torch.zeros(1)
    payload["controller_state"] = state

    with pytest.raises(ValueError, match="generator-only"):
        candidate_runner._load_candidate_state_payload(payload)


def test_s3_candidate_override_requires_provenance_binding(tmp_path) -> None:
    generator = IdentityAnchoredConditionalAmplitudeGenerator()

    with pytest.raises(ValueError, match="candidate state and candidate binding"):
        s3.run(
            output_path=tmp_path / "unused",
            controller_state_override=generator.state_dict(),
        )
