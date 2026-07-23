from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, require_evidence_action, require_passed_evidence, tensor_artifact_ref
from ldm_project.v2.student.direct_template import DirectTemplatePrototypeArtifactV2


def _artifact(tmp_path: Path) -> DirectTemplatePrototypeArtifactV2:
    commands = torch.full((10, 1, 8, 8), 0.5, dtype=torch.float32)
    raw = torch.ones_like(commands)
    supports = (commands[:, 0] >= 0.2).to(torch.float32)
    paths = {"commands": tmp_path / "commands.pt", "raw": tmp_path / "raw.pt", "supports": tmp_path / "supports.pt"}
    torch.save(commands, paths["commands"])
    torch.save(raw, paths["raw"])
    torch.save(supports, paths["supports"])
    return DirectTemplatePrototypeArtifactV2(2, "system", "config", "clean", "checkpoint", "admission", "replay", 1.0, tensor_artifact_ref(paths["commands"], role="commands"), tensor_artifact_ref(paths["raw"], role="raw"), tensor_artifact_ref(paths["supports"], role="supports"))


def test_direct_template_prototype_artifact_is_content_addressed_and_support_locked(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    artifact.validate()
    payload = artifact.to_dict() | {"artifact_hash": artifact.artifact_hash}
    restored = DirectTemplatePrototypeArtifactV2.from_dict(json.loads(json.dumps(payload)))
    assert restored.artifact_hash == artifact.artifact_hash
    torch.save(torch.zeros((10, 8, 8)), tmp_path / "supports.pt")
    with pytest.raises(ValueError, match="artifact content no longer matches"):
        artifact.validate()


def test_admitted_direct_template_only_authorizes_frozen_diagnostic_handoff() -> None:
    context = EvidenceContextV2("system", "config", dataset_hash="clean", checkpoint_hash="checkpoint")
    admission = EvidenceRecordV2(2, "direct_template_control_demonstrated", "diagnostic", "diagnostic", "commit", None, "config", context, {}, (GateResult("automatic", True, {}), GateResult("manual", True, {})), ("freeze_direct_template_prototype",))
    require_evidence_action(admission, kind="direct_template_control_demonstrated", action="freeze_direct_template_prototype", expected_context=context, allow_diagnostic=True)
    with pytest.raises(ValueError, match="must be passed"):
        require_passed_evidence(admission, kind="direct_template_control_demonstrated")
    local_capacity = EvidenceRecordV2(2, "direct_template_local_capacity", "diagnostic", "diagnostic", "commit", None, "config", context, {}, (GateResult("capacity", True, {}),), ())
    with pytest.raises(ValueError, match="must be passed"):
        require_passed_evidence(local_capacity, kind="direct_template_local_capacity")
