from __future__ import annotations

import torch

from ldm_project.distillation.targets import build_clean_readable_target
from ldm_project.student.objectives import _build_clean_readable_target


def test_clean_target_legacy_alias_is_neutral_implementation() -> None:
    source = torch.linspace(0.0, 1.0, steps=2 * 1 * 20 * 20).reshape(2, 1, 20, 20)
    canonical = build_clean_readable_target(source)
    legacy = _build_clean_readable_target(source)
    assert torch.equal(canonical["clean_target"], legacy["clean_target"])
    assert torch.equal(canonical["clean_target_fg_ratio"], legacy["clean_target_fg_ratio"])
