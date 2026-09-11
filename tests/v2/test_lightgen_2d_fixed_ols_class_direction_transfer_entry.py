"""Pre-scientific wiring smoke for the class-direction transfer diagnosis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/research/diagnose_lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("class_direction_transfer_entry_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pre_scientific_entry_smoke_uses_canonical_v4_evaluator_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entrypoint = _load_entrypoint()
    v4_diag = entrypoint._load_v4_diagnostic_helper(ROOT)
    assert not hasattr(v4_diag, "_load_evaluators")
    v4 = v4_diag._load_module(ROOT)
    assert callable(v4._load_evaluators)

    calls: list[str] = []
    captured: dict[str, object] = {}

    class StopBeforeScientificData(RuntimeError):
        pass

    runtime = SimpleNamespace(eval=lambda: calls.append("runtime_eval"))
    consumer = SimpleNamespace(runtime=runtime)
    frozen = SimpleNamespace(
        eval=lambda: calls.append("frozen_eval"),
        parameters=lambda: (),
    )
    smooth = SimpleNamespace(
        eval=lambda: calls.append("smooth_eval"),
        parameters=lambda: (),
    )
    evaluator_states = {
        "frozen_state_sha256": "b" * 64,
        "smooth_state_sha256": "a" * 64,
    }
    contract_authorities = {
        "evaluators": {
            "smooth_checkpoint": {
                "path": "checkpoint/smooth-v2.1.pt",
                "role": "Smooth v2.1 evaluator checkpoint",
                "sha256": v4_diag.SMOOTH_EVALUATOR_SHA256,
            }
        }
    }

    def load_evaluators(root, authorities, device):
        assert root == ROOT
        assert authorities is contract_authorities
        assert device == torch.device("cpu")
        calls.append("canonical_preflight_evaluator_loader")
        return frozen, smooth, evaluator_states

    def load_roles(root, preflight):
        assert root == ROOT
        calls.append("roles_entry")
        raise StopBeforeScientificData("data roles must not be entered by this smoke")

    preflight = SimpleNamespace(
        _cuda_device_facts=lambda: calls.append("device_preflight")
        or (
            torch.device("cpu"),
            {
                "compute_capability": [8, 6],
                "vram_bytes": entrypoint.EXPECTED_VRAM_BYTES,
            },
        ),
        _load_evaluators=load_evaluators,
    )

    def resolve_authorities(root, v4, consumer_git):
        assert root == ROOT
        assert v4 is v4_module
        assert consumer_git["head"] == entrypoint.EXPECTED_CONSUMER_HEAD
        calls.append("authority_resolution")
        return (
            preflight,
            {"schema": "fixture-contract"},
            contract_authorities,
            {"checkpoint": {"schema": "fixture"}},
            {"schema": "fixture-preflight"},
            {
                "state": {"fixture": torch.tensor(1)},
                "git_provenance": {"producer": {"head": "producer"}, "consumer": {"head": "consumer"}},
            },
        )

    v4_module = v4
    monkeypatch.setattr(v4, "_load_v2_contract_before_seed", lambda root: calls.append("v2_contract") or {"schema": "fixture"})
    monkeypatch.setattr(v4, "_load_roles", load_roles)
    v4_diag._load_module = lambda root: calls.append("v4_module") or v4_module
    v4_diag._resolve_authorities = resolve_authorities
    v4_diag._load_runtime = lambda root, v4, preflight, v2_contract, state, device: calls.append("runtime_assembly") or consumer
    original_validate_identity = v4_diag._validate_smooth_evaluator_identity

    def validate_identity(authorities, states):
        identity = original_validate_identity(authorities, states)
        captured["smooth_identity"] = identity
        calls.append("smooth_identity")
        return identity

    v4_diag._validate_smooth_evaluator_identity = validate_identity

    monkeypatch.setattr(entrypoint, "_load_v4_diagnostic_helper", lambda root: v4_diag)
    monkeypatch.setattr(
        entrypoint,
        "git_facts",
        lambda root: {
            "branch": entrypoint.EXPECTED_BRANCH,
            "head": entrypoint.EXPECTED_CONSUMER_HEAD,
            "status_lines": [],
        },
    )
    monkeypatch.setattr(entrypoint.shutil, "disk_usage", lambda path: SimpleNamespace(free=entrypoint.MIN_FREE_BYTES_FOR_OUTPUT))
    original_relative = entrypoint._relative

    def relative_or_temp_path(root, path):
        try:
            return original_relative(root, path)
        except ValueError:
            return path.resolve().as_posix()

    monkeypatch.setattr(entrypoint, "_relative", relative_or_temp_path)
    monkeypatch.setattr(
        entrypoint,
        "_authority_payload",
        lambda root, helper, v4, preflight_module, authorities, v3, v4_auth, smooth_identity: captured.update(
            authority_payload_smooth_identity=smooth_identity
        )
        or {"semantic_authority_id": "fixture", "git_provenance": {"producer": {}, "consumer": {}}},
    )

    result = entrypoint.run_diagnosis(repo_root=ROOT, output_dir=tmp_path / "entry-smoke")

    assert result["result"] == "INVALID"
    assert result["status"] == "invalid_harness"
    assert result["diagnosis_executed"] is False
    assert result["scientific_forward_count"] == 0
    assert result["optimizer_created"] is False
    assert result["backward_executed"] is False
    assert calls == [
        "v4_module",
        "authority_resolution",
        "device_preflight",
        "v2_contract",
        "runtime_assembly",
        "runtime_eval",
        "canonical_preflight_evaluator_loader",
        "frozen_eval",
        "smooth_eval",
        "smooth_identity",
        "roles_entry",
    ]
    assert captured["smooth_identity"]["cross_domain_equality_required"] is False
    assert captured["authority_payload_smooth_identity"] == captured["smooth_identity"]
