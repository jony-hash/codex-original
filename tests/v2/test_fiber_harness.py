from __future__ import annotations

import json
from pathlib import Path

from ldm_project.v2.application.research.fiber_robustness import dispatch
from argparse import Namespace


def test_validation_is_consumed_and_dirty_preflight_fails_closed(tmp_path: Path) -> None:
    validation = tmp_path / "validation.json"
    preflight = tmp_path / "preflight.json"
    assert dispatch(Namespace(research_action="validate", output=str(validation))) == 0
    assert dispatch(Namespace(research_action="preflight", validate_evidence=str(validation), output=str(preflight), dirty_patch_hash="dirty")) == 2
    assert json.loads(preflight.read_text())["gate"] == "invalid_harness"
