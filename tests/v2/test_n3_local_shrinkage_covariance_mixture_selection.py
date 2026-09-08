from __future__ import annotations

import torch

from scripts.research.execute_n3_local_shrinkage_covariance_mixture_selection_v1 import (
    COMPONENT_INSUFFICIENT,
    hard_assignment_from_responsibilities,
    component_insufficiency_status,
)


def test_hard_assignment_uses_lowest_canonical_index_on_tie() -> None:
    responsibilities = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [0.1, 0.7, 0.2],
            [0.0, 0.4, 0.6],
        ],
        dtype=torch.float64,
    )
    assert hard_assignment_from_responsibilities(responsibilities).tolist() == [0, 1, 2]


def test_component_insufficiency_is_fail_closed() -> None:
    assert component_insufficiency_status(1) == COMPONENT_INSUFFICIENT
    assert component_insufficiency_status(2) is None
