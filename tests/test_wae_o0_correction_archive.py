from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
ARCHIVE = ROOT / "experiments/archive/v2_wae_o0_v1_invalid_harness"


def test_invalid_harness_capsule_index_matches_byte_copies() -> None:
    index = json.loads((ARCHIVE / "archive_index.json").read_text(encoding="utf-8"))
    assert index["status"] == "invalid_harness"
    for entry in index["files"]:
        path = ARCHIVE / entry["archive_path"]
        assert path.is_file()
        value = path.read_bytes()
        assert len(value) == entry["bytes"]
        assert hashlib.sha256(value).hexdigest() == entry["sha256"]


def test_invalid_harness_cannot_authorize_downstream() -> None:
    summary = json.loads(
        (ARCHIVE / "correction_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "invalid_harness"
    assert summary["training_authorized"] is False
    assert summary["o0_g_authorized"] is False
    assert summary["smf_authorized"] is False
    assert summary["replacement_contract"]["old_checkpoint_may_warm_start"] is False
