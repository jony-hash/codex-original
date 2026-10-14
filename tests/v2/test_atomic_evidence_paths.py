"""Atomic evidence publication must work with long Windows paths."""
import os
from pathlib import Path

import pytest
import torch

from ldm_project.v2.evidence.io import atomic_torch_save, atomic_torch_save_new, atomic_write_json, atomic_write_new_json, file_sha256, safe_torch_load


@pytest.mark.skipif(os.name != "nt", reason="Windows path boundary")
@pytest.mark.parametrize("writer", [atomic_write_json, atomic_write_new_json, atomic_torch_save, atomic_torch_save_new])
def test_atomic_writer_accepts_long_directory_and_preserves_publication(writer, tmp_path: Path) -> None:
    directory = tmp_path
    while len(str(directory)) < 270:
        directory /= "long_evidence_directory"
    target = directory / "report.pt"
    payload = {"value": 7}
    writer(target, payload)
    assert len(file_sha256(target)) == 64
    extended = Path("\\\\?\\" + str(target.resolve()))
    if writer in (atomic_write_json, atomic_write_new_json):
        import json
        assert json.loads(extended.read_text()) == payload
    else:
        with extended.open("rb") as handle:
            assert torch.load(handle, weights_only=True) == payload
        assert safe_torch_load(target) == payload
    if writer in (atomic_write_new_json, atomic_torch_save_new):
        with pytest.raises(FileExistsError):
            writer(target, {"value": 8})
    else:
        writer(target, {"value": 8})
    assert not list(extended.parent.glob("*.tmp"))
