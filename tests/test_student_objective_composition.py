from __future__ import annotations

import pytest
import torch

from ldm_project.student.objective_composition import (
    EXPERIMENTAL_TERM_KEYS,
    FORMAL_TERM_KEYS,
    compose_one_shot_total,
)


def test_formal_camera_term_is_required() -> None:
    with pytest.raises(KeyError, match="camera_recon_loss"):
        compose_one_shot_total({})


def test_composer_keeps_formal_then_experimental_order() -> None:
    terms = {"camera_recon_loss": torch.tensor(1.0)}
    terms.update({key: torch.tensor(float(index + 1)) for index, key in enumerate(EXPERIMENTAL_TERM_KEYS)})
    expected = torch.tensor(1.0 + sum(range(1, len(EXPERIMENTAL_TERM_KEYS) + 1)))
    actual = compose_one_shot_total(terms)
    assert FORMAL_TERM_KEYS == ("camera_recon_loss",)
    assert torch.equal(actual, expected)


def test_missing_experimental_terms_are_zero_by_contract() -> None:
    formal = torch.tensor(0.25, requires_grad=True)
    actual = compose_one_shot_total({"camera_recon_loss": formal})
    assert actual is formal
