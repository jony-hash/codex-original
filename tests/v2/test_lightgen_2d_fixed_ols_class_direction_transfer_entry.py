"""Pre-scientific wiring smoke for the class-direction transfer diagnosis."""

from __future__ import annotations

import importlib.util
import inspect
import json
import tempfile
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
        "_prepare_cuda_runtime_dll_search_path",
        lambda: calls.append("cuda_environment") or {"status": "fixture", "scope": "current_python_process_only"},
    )
    monkeypatch.setattr(
        entrypoint,
        "_synthetic_runtime_smoke",
        lambda v4, consumer, device: calls.append("synthetic_runtime_smoke")
        or {"status": "passed", "data_role": "synthetic_no_data", "data_roles_opened": False},
    )
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
        "cuda_environment",
        "device_preflight",
        "v2_contract",
        "runtime_assembly",
        "runtime_eval",
        "canonical_preflight_evaluator_loader",
        "frozen_eval",
        "smooth_eval",
        "smooth_identity",
        "synthetic_runtime_smoke",
        "roles_entry",
    ]
    assert captured["smooth_identity"]["cross_domain_equality_required"] is False
    assert captured["authority_payload_smooth_identity"] == captured["smooth_identity"]


def test_cuda_runtime_dll_search_path_is_environment_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entrypoint = _load_entrypoint()
    environment_bin = tmp_path / "bin"
    environment_bin.mkdir()
    (environment_bin / "nvrtc-builtins64_121.dll").write_bytes(b"fixture")
    monkeypatch.setattr(entrypoint.os, "name", "nt")
    monkeypatch.setattr(entrypoint.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(entrypoint.torch.version, "cuda", "12.1")
    monkeypatch.setenv("PATH", "fixture-existing-path")

    result = entrypoint._prepare_cuda_runtime_dll_search_path()

    assert result["status"] == "prepended"
    assert result["scope"] == "current_python_process_only"
    assert result["expected_dll"] == "nvrtc-builtins64_121.dll"
    assert result["expected_dll_exists"] is True
    assert result["path_changed"] is True
    assert result["environment_bin"] == str(environment_bin.resolve())
    assert entrypoint.os.environ["PATH"].split(entrypoint.os.pathsep)[0] == str(environment_bin.resolve())


def test_cuda_runtime_dll_search_path_missing_is_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entrypoint = _load_entrypoint()
    monkeypatch.setattr(entrypoint.os, "name", "nt")
    monkeypatch.setattr(entrypoint.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(entrypoint.torch.version, "cuda", "12.1")

    with pytest.raises(entrypoint.DiagnosisBlocked) as caught:
        entrypoint._prepare_cuda_runtime_dll_search_path()

    assert caught.value.classification == "CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED"
    assert caught.value.stage == "cuda_dll_search_path"
    assert caught.value.details["expected_dll"] == "nvrtc-builtins64_121.dll"
    assert caught.value.details["expected_dll_exists"] is False


def test_synthetic_runtime_smoke_maps_nvrtc_failure_to_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires the CUDA device used by the real runtime smoke")
    entrypoint = _load_entrypoint()

    class FailingRuntime:
        config = SimpleNamespace(propagation_grid_size=entrypoint.PRODUCTION_GRID)

        def forward_from_ols(self, z):
            raise RuntimeError("nvrtc: error: failed to open nvrtc-builtins64_121.dll")

        def parameters(self):
            return ()

    consumer = SimpleNamespace(runtime=FailingRuntime())
    v4 = SimpleNamespace()
    monkeypatch.setattr(entrypoint, "_runtime_digest", lambda consumer: "state")

    with pytest.raises(entrypoint.DiagnosisBlocked) as caught:
        entrypoint._synthetic_runtime_smoke(v4, consumer, torch.device("cuda:0"))

    assert caught.value.classification == "CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED"
    assert caught.value.stage == "synthetic_runtime_smoke"
    assert caught.value.details["data_roles_opened"] is False


def test_missing_raw_summary_binding_fails_before_scientific_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    entrypoint = _load_entrypoint()
    monkeypatch.setattr(entrypoint, "_raw_summary", None)

    with pytest.raises(entrypoint.InvalidDiagnosis, match="_raw_summary"):
        entrypoint._validate_execution_core_bindings()


def test_class_direction_critical_path_uses_resolved_local_helpers() -> None:
    entrypoint = _load_entrypoint()
    entrypoint._validate_execution_core_bindings()
    source = "\n".join(
        (
            inspect.getsource(entrypoint.run_class_direction_diagnosis_core),
            inspect.getsource(entrypoint._raw_change_summary),
            inspect.getsource(entrypoint._bootstrap),
        )
    )
    assert "v4._" not in source
    assert "v4_diag._" not in source
    for name in (
        "_validate_runtime_result",
        "_raw_summary",
        "_bootstrap_deltas",
        "run_class_direction_diagnosis_core",
        "_write_report_and_lineage",
    ):
        assert callable(getattr(entrypoint, name))


def test_full_path_synthetic_core_reaches_report_lineage_and_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    entrypoint = _load_entrypoint()
    forbidden_calls: list[str] = []
    monkeypatch.setattr(entrypoint, "_load_v4_diagnostic_helper", lambda root: forbidden_calls.append("helper"))

    samples_per_class = 4
    train_labels = torch.arange(10, dtype=torch.long).repeat_interleave(samples_per_class)
    selection_labels = train_labels.clone()
    train_count = int(train_labels.numel())
    selection_count = int(selection_labels.numel())
    pixel_grid = torch.linspace(0.0, 1.0, 28, dtype=torch.float32)
    image_base = pixel_grid.reshape(1, 1, 28) + pixel_grid.reshape(1, 28, 1)
    train_images = (0.02 * train_labels[:, None, None, None].to(torch.float32) + 0.01 * image_base).clamp(0.0, 1.0)
    selection_images = (0.03 * selection_labels[:, None, None, None].to(torch.float32) + 0.015 * image_base).clamp(0.0, 1.0)
    coordinates = torch.linspace(-1.0, 1.0, 100, dtype=torch.float32)
    train_real = 0.05 * train_labels[:, None].to(torch.float32) + 0.001 * torch.arange(train_count, dtype=torch.float32)[:, None] + coordinates[None]
    train_imag = 0.03 * train_labels[:, None].to(torch.float32) - 0.0005 * torch.arange(train_count, dtype=torch.float32)[:, None] - coordinates[None]
    train_z = torch.complex(train_real, train_imag).to(torch.complex64)
    selection_real = 0.05 * selection_labels[:, None].to(torch.float32) + 0.002 * torch.arange(selection_count, dtype=torch.float32)[:, None] + coordinates[None]
    selection_imag = 0.03 * selection_labels[:, None].to(torch.float32) - 0.001 * torch.arange(selection_count, dtype=torch.float32)[:, None] - coordinates[None]
    selection_z = torch.complex(selection_real, selection_imag).to(torch.complex64)
    raw_x = torch.linspace(0.0, 1.0, 200, dtype=torch.float32).reshape(1, 1, 200, 1)
    raw_y = torch.linspace(1.0, 0.0, 200, dtype=torch.float32).reshape(1, 1, 1, 200)

    def fake_generator(z: torch.Tensor) -> torch.Tensor:
        sample = z.real[:, 0].reshape(-1, 1, 1, 1)
        contrast = z.imag[:, 1].reshape(-1, 1, 1, 1)
        return (0.1 + 0.02 * raw_x + 0.03 * raw_y + 0.002 * sample + 0.001 * contrast).expand(-1, 1, 200, 200).contiguous()

    calls = {"canonical": 0, "shifted": 0, "smooth": 0}

    def canonical_forward(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        calls["canonical"] += 1
        assert images is selection_images
        z = selection_z.clone()
        return z, fake_generator(z), {"canonical_replay_max_abs_difference": 0.0}

    def shifted_forward(z: torch.Tensor) -> torch.Tensor:
        calls["shifted"] += 1
        return fake_generator(z)

    def smooth_forward(raw: torch.Tensor) -> torch.Tensor:
        calls["smooth"] += 1
        mean = raw.flatten(1).mean(dim=1)
        return torch.stack([mean * float(index + 1) for index in range(10)], dim=1)

    temporary_root = tempfile.TemporaryDirectory(prefix="cg_")
    artifact_root = Path(temporary_root.name)
    run_dir = artifact_root / "synthetic-run"
    run_dir.mkdir()

    def persist(observations, metadata):
        return entrypoint._save_sufficient_observations(artifact_root, run_dir, observations, metadata)

    core = entrypoint.run_class_direction_diagnosis_core(
        train_z=train_z,
        train_images=train_images,
        train_labels=train_labels,
        selection_images=selection_images,
        selection_labels=selection_labels,
        canonical_forward=canonical_forward,
        shifted_forward=shifted_forward,
        smooth_forward=smooth_forward,
        selection_indices=torch.arange(selection_count, dtype=torch.long),
        samples_per_class=samples_per_class,
        bootstrap_replicates=17,
        bootstrap_seed=271828,
        persist_sufficient_observations=persist,
    )

    assert forbidden_calls == []
    assert calls == {"canonical": 1, "shifted": 1, "smooth": 2}
    assert core["classification"] in entrypoint.CLASSIFICATIONS
    assert len(core["report_fields"]["directional_consistency"]["pairs"]) == 10
    assert all(item["sample_count"] == samples_per_class for item in core["report_fields"]["directional_consistency"]["pairs"])
    observation_ref = core["observation_artifact"]
    assert observation_ref["schema"] == entrypoint.SUFFICIENT_OBSERVATIONS_SCHEMA
    observation_path = run_dir / entrypoint.SUFFICIENT_OBSERVATIONS_NAME
    loaded_observations = entrypoint.safe_torch_load(observation_path, expected_sha256=observation_ref["sha256"], map_location="cpu")
    assert loaded_observations["schema"] == entrypoint.SUFFICIENT_OBSERVATIONS_SCHEMA
    assert sorted(loaded_observations["protocol"]["observation_tensors"]) == observation_ref["observation_tensors"]

    payload = {
        "schema": entrypoint.REPORT_SCHEMA,
        "version": 1,
        "task_id": entrypoint.TASK_ID,
        "outer_gate": entrypoint.OUTER_GATE,
        "result": "VALID_COMPLETED",
        "status": "valid_completed",
        "classification": core["classification"],
        "entrypoint": {"path": "synthetic", "sha256": "0" * 64},
        "authorities": {"semantic_authority_id": "synthetic_no_data"},
        "git_provenance": {"consumer": {"branch": "synthetic", "head": "0" * 40}},
        "provenance": {"git_start": {"branch": "synthetic", "head": "0" * 40}},
        "claim_boundary": {"optimizer_created": False, "backward_executed": False, "optical_update_executed": False, "training_executed": False},
        "runtime": {"data_role": "synthetic_no_data", "production_cuda_smoke": "separate_required_check"},
        "data": {"synthetic_train_rows": train_count, "synthetic_dev_rows": selection_count, "real_project_data_accessed": False},
        "governance": {"training_authorized": False, "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False},
    }
    payload.update(entrypoint._core_report_fields(core))
    report_ref, lineage_ref = entrypoint._write_report_and_lineage(artifact_root, run_dir, payload)
    report = json.loads((artifact_root / report_ref["path"]).read_text(encoding="utf-8"))
    lineage = json.loads((artifact_root / lineage_ref["path"]).read_text(encoding="utf-8"))
    assert report["schema"] == entrypoint.REPORT_SCHEMA
    assert report["classification"] == core["classification"]
    assert lineage["schema"] == entrypoint.LINEAGE_SCHEMA
    assert lineage["artifacts"]["report"]["sha256"] == report_ref["sha256"]


@pytest.mark.scheduled_diagnostic
def test_production_like_cuda_runtime_smoke_passes_without_data_roles() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires the configured CUDA device")
    entrypoint = _load_entrypoint()
    v4_diag = entrypoint._load_v4_diagnostic_helper(ROOT)
    v4 = v4_diag._load_module(ROOT)
    started = entrypoint.git_facts(ROOT)
    preflight, _contract, authorities, _v3, _preflight_authority, v4_authority = v4_diag._resolve_authorities(ROOT, v4, started)
    entrypoint._prepare_cuda_runtime_dll_search_path()
    device, _device_facts = preflight._cuda_device_facts()
    v2_contract = v4._load_v2_contract_before_seed(ROOT)
    consumer = v4_diag._load_runtime(ROOT, v4, preflight, v2_contract, v4_authority["state"], device)
    try:
        result = entrypoint._synthetic_runtime_smoke(v4, consumer, device)
    finally:
        del consumer
        torch.cuda.empty_cache()
    assert result["status"] == "passed"
    assert result["data_roles_opened"] is False
    assert result["raw_cmos"]["shape"] == [2, 1, 200, 200]
    assert authorities is not None
