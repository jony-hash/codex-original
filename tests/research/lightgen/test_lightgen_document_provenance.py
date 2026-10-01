from __future__ import annotations

from pathlib import Path

from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1 as sampled
from scripts.research.lightgen import execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1 as conflict
from scripts.research.lightgen import execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 as frozen_latent


def test_status_and_handoff_text_are_provenance_not_loader_gates(tmp_path: Path) -> None:
    (tmp_path / "docs/status").mkdir(parents=True)
    status = tmp_path / "docs/CURRENT_STATUS.md"
    handoff = tmp_path / "docs/status/current_handoff.md"
    status.write_text("old project note\n", encoding="utf-8")
    handoff.write_text("old handoff note\n", encoding="utf-8")

    before = [
        sampled._doc_top_guard(tmp_path),
        conflict._validate_current_docs(tmp_path),
        frozen_latent._validate_current_docs(tmp_path),
    ]

    status.write_text("updated project note\n", encoding="utf-8")
    handoff.write_text("updated handoff note\n", encoding="utf-8")
    after = [
        sampled._doc_top_guard(tmp_path),
        conflict._validate_current_docs(tmp_path),
        frozen_latent._validate_current_docs(tmp_path),
    ]

    assert all(item for item in before + after)
    assert before != after
