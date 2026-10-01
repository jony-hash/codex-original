"""Evaluate frozen N4 endpoints for generator/extra-screen attribution.

This entrypoint creates new evaluation observations for C0, the original C1
step-300 checkpoint, and the original C2 step-300 checkpoint.  It never
constructs an optimizer and never changes a loaded model state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def _load_n4_execution() -> Any:
    path = Path(__file__).parents[1] / (
        "execute_lightgen_2d_ols_interface_synthesis_model_family_comparison_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "lightgen_n4_execution_authority_endpoint_attribution", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("N4 execution authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


N4 = _load_n4_execution()
V10 = N4.V10
V11 = N4.V11

TASK_ID = "EXECUTE_N4_ENDPOINT_ATTRIBUTION_EVALUATION_V1"
DESIGN_AUTHORITY = "DEFINE_N4_OLS_INTERFACE_SYNTHESIS_MODEL_FAMILY_CONTRACT_V1"
OUTER_GATE = "N4_ENDPOINT_ATTRIBUTION_RESOLVED"
EXPECTED_BRANCH = N4.EXPECTED_BRANCH
EXPECTED_HEAD = N4.EXPECTED_HEAD

BASE_OUTPUT = Path("outputs/20260914_n4_endpoint_attribution/runs")
PREFLIGHT_DIR = BASE_OUTPUT / "lightgen_2d_n4_endpoint_attribution_preflight_v1_attempt_01"
EVALUATION_DIR = BASE_OUTPUT / "lightgen_2d_n4_endpoint_attribution_evaluation_v1_attempt_01"

PREFLIGHT_REPORT_NAME = "lightgen_2d_n4_endpoint_attribution_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_n4_endpoint_attribution_preflight_lineage.v1.json"
INPUT_NAME = "lightgen_2d_n4_endpoint_attribution_input.v1.json"
REPORT_NAME = "lightgen_2d_n4_endpoint_attribution_report.v1.json"
LINEAGE_NAME = "lightgen_2d_n4_endpoint_attribution_lineage.v1.json"
SUFFICIENT_NAME = "lightgen_2d_n4_endpoint_attribution_sufficient_observations.v1.pt"
RAW_NAME = "lightgen_2d_n4_endpoint_attribution_cyclic_raw_outputs.v1.pt"

N4_CONTRACT_PATHS = {
    "outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_contract_v1_attempt_04/lightgen_2d_ols_interface_synthesis_model_family_contract.v1.json": "7b731b69d109545f9e75530cf1fd2efd3f2ccee95225028463c3bc24558b5af2",
    "outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_contract_v1_attempt_05/lightgen_2d_ols_interface_synthesis_model_family_contract.v1.json": "ae3b90bb84e7748094217ff04ef8e9141c9dd1ab5e89a21376efc73978ee2cd3",
}
EVALUATION_CONTRACT_PATH = Path(next(path for path, sha in N4_CONTRACT_PATHS.items() if sha.endswith("ee2cd3")))
EVALUATION_CONTRACT_SHA256 = N4_CONTRACT_PATHS[EVALUATION_CONTRACT_PATH.as_posix()]

C1_CHECKPOINT_PATH = Path(
    "outputs/20260913_n4/runs/"
    "lightgen_2d_ols_interface_synthesis_model_family_c1_attempt_01/"
    "lightgen_2d_n4_c1_step_300_checkpoint.v1.pt"
)
C1_CHECKPOINT_SHA256 = "d7a5581df3fb460337e8655865a200886739cdeab0fb6074213cc4b2cc66c6e9"
C1_REPORT_PATH = Path(
    "outputs/20260913_n4/runs/"
    "lightgen_2d_ols_interface_synthesis_model_family_c1_attempt_01/"
    "lightgen_2d_n4_c1_scientific_report.v1.json"
)
C1_REPORT_SHA256 = "2b6b8c9d4b5bbe2c48eab18ee0d855c95f699cc0abeb93202bfe6340e51c6e61"
C1_LINEAGE_PATH = Path(
    "outputs/20260913_n4/runs/"
    "lightgen_2d_ols_interface_synthesis_model_family_c1_attempt_01/"
    "lightgen_2d_n4_c1_scientific_lineage.v1.json"
)
C1_LINEAGE_SHA256 = "ee1d07864a021e218f682cc54003628e245d87f79844760ce8a07abf198457d9"

C2_CHECKPOINT_PATH = Path(
    "outputs/20260913_n4/runs/"
    "lightgen_2d_ols_interface_synthesis_model_family_c2_attempt_01/"
    "lightgen_2d_n4_c2_step_300_checkpoint.v1.pt"
)
C2_CHECKPOINT_SHA256 = "9b4959da192c7a4810db29a3571b7e32050c97b73da80f034335e97f2ea6fd20"
C2_REPORT_PATH = Path(
    "outputs/20260913_n4/runs/"
    "lightgen_2d_ols_interface_synthesis_model_family_c2_attempt_01/"
    "lightgen_2d_n4_c2_scientific_report.v1.json"
)
C2_REPORT_SHA256 = "660454787c3df9d5b6aa69facfd9ec94acc67c19ac08cff1e23cdcf754b81003"
C2_LINEAGE_PATH = Path(
    "outputs/20260913_n4/runs/"
    "lightgen_2d_ols_interface_synthesis_model_family_c2_attempt_01/"
    "lightgen_2d_n4_c2_scientific_lineage.v1.json"
)
C2_LINEAGE_SHA256 = "21c6840d62012a9fa7a9cca6ffd2cc377aa3be3e2832223f5aead1499763edbd"

CLASS_COUNT = N4.CLASS_COUNT
OLS_PORT_COUNT = N4.OLS_PORT_COUNT
PRODUCTION_GRID = N4.PRODUCTION_GRID
SMOKE_GRID = N4.SMOKE_GRID
LOGICAL_BATCH_SIZE = N4.LOGICAL_BATCH_SIZE
SELECTION_ROWS = N4.SELECTION_ROWS
SELECTION_PER_CLASS = N4.SELECTION_PER_CLASS
SELECTION_SEED = N4.SELECTION_SEED
SELECTION_INDICES_SHA256 = N4.SELECTION_INDICES_SHA256
SELECTION_LABELS_SHA256 = N4.SELECTION_LABELS_SHA256
BOOTSTRAP_REPLICATES = N4.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = N4.BOOTSTRAP_SEED
MATERIAL_DELTA = N4.ARCHITECTURE_REQUIRED_DELTA
STAGE_A_FIREWALL = N4.STAGE_A_FIREWALL
LAMBDA_TRANSFER = N4.LAMBDA_TRANSFER
LAMBDA_IMAGE = N4.LAMBDA_IMAGE
S_TAU = N4.S_TAU


class InvalidAttribution(RuntimeError):
    """The evaluation-only endpoint attribution harness is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidAttribution(message)


def _path(root: Path, value: str | Path) -> Path:
    relative = Path(value)
    _require(not relative.is_absolute(), f"absolute path is not allowed: {relative}")
    return (root / relative).resolve()


def _relative(root: Path, value: Path) -> str:
    return value.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidAttribution(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    return N4._file_sha256(path)


def _validate_sha(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = _sha256(path)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")


def _state_digest(state: Mapping[str, Any]) -> str:
    return N4._digest_tree(N4._cpu_tree(dict(state)))


def _tensor_finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value.real).all()), f"{name} real values are not finite")
    if torch.is_complex(value):
        _require(bool(torch.isfinite(value.imag).all()), f"{name} imaginary values are not finite")


def _artifact_ref(root: Path, path: Path, expected: str, name: str) -> dict[str, str]:
    _validate_sha(path, expected, name)
    return {"path": _relative(root, path), "sha256": expected}


def _validate_contract(root: Path, path: Path, expected: str) -> tuple[dict[str, Any], dict[str, str]]:
    ref = _artifact_ref(root, path, expected, "N4 contract")
    contract = _read_json(path, "N4 contract")
    _require(
        contract.get("schema") == "lightgen_2d_ols_interface_synthesis_model_family_contract.v1"
        and contract.get("result") == "READY"
        and contract.get("status") == "valid_completed"
        and contract.get("semantic_authority") == DESIGN_AUTHORITY,
        "N4 contract authority changed",
    )
    _require(contract.get("git", {}).get("branch") == EXPECTED_BRANCH, "N4 contract branch changed")
    _require(contract.get("git", {}).get("head") == EXPECTED_HEAD, "N4 contract HEAD changed")
    _require(contract.get("family", {}).get("fixed_rank") == N4.ADAPTER_RANK, "N4 fixed rank changed")
    objective = contract.get("objective", {})
    _require(float(objective.get("lambda_transfer", 0.0)) == LAMBDA_TRANSFER, "N4 transfer objective changed")
    _require(float(objective.get("lambda_image", 0.0)) == LAMBDA_IMAGE, "N4 image objective changed")
    selection = contract.get("selection", {})
    _require(selection.get("rows") == SELECTION_ROWS and selection.get("per_class") == SELECTION_PER_CLASS and selection.get("seed") == SELECTION_SEED, "N4 selection protocol changed")
    return contract, ref


def _validate_n4_contracts(root: Path) -> dict[str, Any]:
    contracts: dict[str, Any] = {}
    for relative, expected in N4_CONTRACT_PATHS.items():
        contract, ref = _validate_contract(root, _path(root, relative), expected)
        contracts[relative] = {"authority": contract["semantic_authority"], "ref": ref}
    _require(len(contracts) == 2, "N4 contract identity set is incomplete")
    return contracts


def _validate_checkpoint(root: Path, candidate: str, path: Path, expected_sha: str) -> dict[str, Any]:
    checkpoint_ref = _artifact_ref(root, path, expected_sha, f"N4 {candidate} step-300 checkpoint")
    checkpoint = N4._mapping(
        N4.safe_torch_load(path, expected_sha256=expected_sha, map_location="cpu"),
        f"N4 {candidate} checkpoint",
    )
    _require(checkpoint.get("candidate") == candidate and checkpoint.get("checkpoint_step") == 300, f"N4 {candidate} checkpoint identity changed")
    _require(checkpoint.get("schema") == f"lightgen_2d_n4_model_family_{candidate.lower()}_checkpoint.v1", f"N4 {candidate} checkpoint schema changed")
    contract_ref = N4._mapping(checkpoint.get("contract"), f"N4 {candidate} checkpoint contract")
    contract_path = str(contract_ref.get("path"))
    _require(contract_path in N4_CONTRACT_PATHS, f"N4 {candidate} checkpoint contract is not an accepted N4 authority")
    _require(str(contract_ref.get("sha256")) == N4_CONTRACT_PATHS[contract_path], f"N4 {candidate} checkpoint contract hash changed")
    contract, _ = _validate_contract(root, _path(root, contract_path), N4_CONTRACT_PATHS[contract_path])
    _require(contract.get("outer_gate") == "N4_OLS_INTERFACE_SYNTHESIS_COMPARISON_RESOLVED", "N4 checkpoint outer-gate contract changed")
    _require(checkpoint.get("lambda_transfer") == LAMBDA_TRANSFER and checkpoint.get("lambda_image") == LAMBDA_IMAGE, f"N4 {candidate} checkpoint objective changed")
    for key in ("stage_b_executed", "generation_executed", "formal_promotion"):
        _require(checkpoint.get(key) is False, f"N4 {candidate} checkpoint boundary changed: {key}")
    data_roles = N4._mapping(checkpoint.get("data_roles"), f"N4 {candidate} checkpoint data roles")
    _require(all(data_roles.get(key) is False for key in ("protected", "official_test", "sealed")), f"N4 {candidate} checkpoint data boundary changed")

    runtime_state = dict(N4._mapping(checkpoint.get("runtime_state"), f"N4 {candidate} runtime state"))
    adapter_state = dict(N4._mapping(checkpoint.get("adapter_state"), f"N4 {candidate} adapter state"))
    expected_adapter = {"u_real", "u_imag", "v_real", "v_imag"}
    _require(set(adapter_state) == expected_adapter, f"N4 {candidate} adapter state surface changed")
    for name, value in adapter_state.items():
        _require(tuple(value.shape) == (OLS_PORT_COUNT, N4.ADAPTER_RANK), f"N4 {candidate} adapter shape changed: {name}")
        _tensor_finite(value, f"N4 {candidate} adapter {name}")
    common_runtime = {"encoder.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}
    if candidate == "C1":
        _require(set(runtime_state) == common_runtime | {"generator.phase"}, "N4 C1 runtime state surface changed")
        _require(tuple(runtime_state["generator.phase"].shape) == (2, 1, 100, 100), "N4 C1 generator shape changed")
        _require(checkpoint.get("trainable_parameter_names") == N4.ADAPTER_PARAMETER_NAMES + ["generator.phase"], "N4 C1 trainable surface changed")
    else:
        _require(set(runtime_state) == common_runtime | {"generator.extra_phase", "generator.stack.phase"}, "N4 C2 runtime state surface changed")
        _require(tuple(runtime_state["generator.extra_phase"].shape) == (1, 1, 100, 100), "N4 C2 extra-screen shape changed")
        _require(tuple(runtime_state["generator.stack.phase"].shape) == (2, 1, 100, 100), "N4 C2 generator stack shape changed")
        _require(checkpoint.get("trainable_parameter_names") == N4.ADAPTER_PARAMETER_NAMES + ["generator.extra_phase", "generator.stack.phase"], "N4 C2 trainable surface changed")
    for name, value in runtime_state.items():
        _tensor_finite(value, f"N4 {candidate} runtime {name}")
    _require(V10.V6.tensor_sha256(runtime_state["encoder.phase"]) == N4.V3_ENCODER_SHA256, f"N4 {candidate} encoder identity changed")
    _require(torch.equal(runtime_state["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), f"N4 {candidate} OLS transport changed")
    return {"candidate": candidate, "checkpoint": checkpoint, "checkpoint_ref": checkpoint_ref, "runtime_state": runtime_state, "adapter_state": adapter_state, "runtime_digest": _state_digest(runtime_state), "adapter_digest": _state_digest(adapter_state)}


def _validate_original_reports(root: Path) -> dict[str, Any]:
    c1_report_ref = _artifact_ref(root, _path(root, C1_REPORT_PATH), C1_REPORT_SHA256, "original N4 C1 report")
    c1_lineage_ref = _artifact_ref(root, _path(root, C1_LINEAGE_PATH), C1_LINEAGE_SHA256, "original N4 C1 lineage")
    c2_report_ref = _artifact_ref(root, _path(root, C2_REPORT_PATH), C2_REPORT_SHA256, "original N4 C2 report")
    c2_lineage_ref = _artifact_ref(root, _path(root, C2_LINEAGE_PATH), C2_LINEAGE_SHA256, "original N4 C2 lineage")
    c1_report = _read_json(_path(root, C1_REPORT_PATH), "original N4 C1 report")
    c2_report = _read_json(_path(root, C2_REPORT_PATH), "original N4 C2 report")
    _require(c1_report.get("result") == "INVALID" and c1_report.get("scientific_updates_completed") == 300, "original N4 C1 report identity changed")
    _require(c2_report.get("result") == "VALID_COMPLETED" and c2_report.get("classification") == "N4_CANDIDATE_ADEQUATE", "original N4 C2 report identity changed")
    return {"C1": {"report": c1_report_ref, "lineage": c1_lineage_ref, "result": c1_report.get("result"), "classification": c1_report.get("classification")}, "C2": {"report": c2_report_ref, "lineage": c2_lineage_ref, "result": c2_report.get("result"), "classification": c2_report.get("classification")}}


def _new_c0(bundle: Mapping[str, Any], device: torch.device) -> tuple[Any, Any]:
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V10.V8._load_runtime_state(consumer, bundle["c0_runtime_state"])
    adapter = V11.LowRankAdapter(device=device)
    adapter.load_state_dict(dict(bundle["c0_adapter_state"]), strict=True)
    V10.V8._configure_all_frozen(consumer)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    consumer.runtime.eval()
    adapter.eval()
    return consumer, adapter


def _new_checkpoint_endpoint(bundle: Mapping[str, Any], checkpoint: Mapping[str, Any], candidate: str, device: torch.device) -> tuple[Any, Any]:
    consumer, adapter = N4._new_candidate(
        bundle,
        candidate,
        device,
        PRODUCTION_GRID,
        runtime_state=checkpoint["runtime_state"],
        adapter_state=checkpoint["adapter_state"],
    )
    N4._configure_candidate(consumer, adapter, candidate, trainable=False)
    consumer.runtime.eval()
    adapter.eval()
    return consumer, adapter


def _generator_diagnostics(bundle: Mapping[str, Any], consumer: Any, candidate: str) -> dict[str, Any]:
    if candidate == "C2":
        initial_state = {
            "generator.stack.phase": bundle["c0_runtime_state"]["generator.phase"],
            "generator.extra_phase": torch.zeros((1, 1, 100, 100), dtype=torch.float32),
        }
        return N4._generator_observation(consumer, "C2", initial_state)
    return N4._generator_observation(consumer, "C1", {"generator.phase": bundle["c0_runtime_state"]["generator.phase"]})


def _public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _public(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_public(item) for item in value]
    if torch.is_tensor(value):
        raise InvalidAttribution("tensor leaked into JSON report")
    return value


def _preflight_endpoint(bundle: Mapping[str, Any], endpoint: str, checkpoint: Mapping[str, Any] | None, device: torch.device) -> dict[str, Any]:
    if endpoint == "C0":
        consumer, adapter = _new_c0(bundle, device)
    else:
        _require(checkpoint is not None, f"N4 {endpoint} checkpoint is unavailable")
        consumer, adapter = _new_checkpoint_endpoint(bundle, checkpoint, endpoint, device)
    runtime_before = _state_digest(consumer.runtime.state_dict())
    adapter_before = _state_digest(adapter.state_dict())
    synthetic = torch.linspace(0.0, 1.0, CLASS_COUNT * 28 * 28, dtype=torch.float32).reshape(CLASS_COUNT, 1, 28, 28).to(device)
    labels = torch.arange(CLASS_COUNT, dtype=torch.long)
    with torch.no_grad():
        evaluation = N4._evaluate(consumer, adapter, bundle["authorities"]["smooth"], synthetic, labels, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"], step=0, train_trace=None)
    _require(runtime_before == _state_digest(consumer.runtime.state_dict()), f"N4 {endpoint} synthetic evaluation changed runtime state")
    _require(adapter_before == _state_digest(adapter.state_dict()), f"N4 {endpoint} synthetic evaluation changed adapter state")
    _require(evaluation["raw_validity"]["finite"] is True, f"N4 {endpoint} synthetic raw is not finite")
    return {"endpoint": endpoint, "synthetic_rows": CLASS_COUNT, "raw_shape": list(evaluation["raw_validity"]["shape"]), "runtime_digest_unchanged": True, "adapter_digest_unchanged": True, "training_updates": 0, "optimizer_updates": 0}


def run_preflight(root: Path) -> dict[str, Any]:
    _require(not PREFLIGHT_DIR.exists(), f"refusing to reuse endpoint-attribution preflight directory: {_path(root, PREFLIGHT_DIR)}")
    started = V10.V6.git_facts(root)
    _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from N4 endpoint attribution authority")
    contracts = _validate_n4_contracts(root)
    bundle = N4._validate_v11_authorities(root)
    original_reports = _validate_original_reports(root)
    _validate_checkpoint(root, "C1", _path(root, C1_CHECKPOINT_PATH), C1_CHECKPOINT_SHA256)
    _validate_checkpoint(root, "C2", _path(root, C2_CHECKPOINT_PATH), C2_CHECKPOINT_SHA256)
    V10.V6._prepare_cuda_runtime_dll_search_path()
    device, device_facts = V10.V6._cuda_facts()
    _require(device.type == "cuda", "CUDA is required for the endpoint attribution evaluator")
    bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
    checks = {}
    checks["C0"] = _preflight_endpoint(bundle, "C0", None, device)
    c1 = _validate_checkpoint(root, "C1", _path(root, C1_CHECKPOINT_PATH), C1_CHECKPOINT_SHA256)
    checks["C1"] = _preflight_endpoint(bundle, "C1", c1, device)
    c2 = _validate_checkpoint(root, "C2", _path(root, C2_CHECKPOINT_PATH), C2_CHECKPOINT_SHA256)
    checks["C2"] = _preflight_endpoint(bundle, "C2", c2, device)
    payload = {"schema": "lightgen_2d_n4_endpoint_attribution_preflight_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "AUTHORIZABLE", "status": "valid_completed", "classification": "LIGHTGEN_2D_N4_ENDPOINT_ATTRIBUTION_AUTHORIZABLE", "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "provenance": {"git_start": started, "git_end": V10.V6.git_facts(root)}, "contracts": contracts, "original_reports": original_reports, "runtime": {"device": device_facts, "grid": PRODUCTION_GRID, "batch": LOGICAL_BATCH_SIZE}, "checks": checks, "claim_boundary": {"training_updates": 0, "optimizer_updates": 0, "scientific_dev_evaluations": 0, "scientific_cyclic_evaluations": 0, "model_state_modified": False, "checkpoint_modified": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False}, "next": "proceed to one symmetric C0/C1/C2 evaluation-only comparison"}
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=False)
    report_path = PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME
    N4.atomic_write_new_json(report_path, payload, artifact_kind="N4 endpoint attribution preflight report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_n4_endpoint_attribution_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "claim_boundary": payload["claim_boundary"]}, artifact_kind="N4 endpoint attribution preflight lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "training_updates": 0, "optimizer_updates": 0}


def _load_preflight(root: Path) -> dict[str, Any]:
    report_path = _path(root, PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = _path(root, PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    _require(report_path.is_file() and lineage_path.is_file(), "endpoint attribution preflight is unavailable")
    report = _read_json(report_path, "endpoint attribution preflight report")
    lineage = _read_json(lineage_path, "endpoint attribution preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed", "endpoint attribution preflight is not authorizable")
    _require(lineage.get("report", {}).get("sha256") == _sha256(report_path), "endpoint attribution preflight lineage binding changed")
    _require(report.get("claim_boundary", {}).get("training_updates") == 0 and report.get("claim_boundary", {}).get("optimizer_updates") == 0, "endpoint attribution preflight update boundary changed")
    return {"report": report, "report_ref": {"path": _relative(root, report_path), "sha256": _sha256(report_path)}, "lineage_ref": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}}


def _evaluate_endpoint(root: Path, bundle: Mapping[str, Any], endpoint: str, checkpoint: Mapping[str, Any] | None, dev_images: torch.Tensor, dev_labels: torch.Tensor, selection_images: torch.Tensor, selection_labels: torch.Tensor, selection_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    if endpoint == "C0":
        consumer, adapter = _new_c0(bundle, device)
    else:
        _require(checkpoint is not None, f"N4 {endpoint} endpoint checkpoint is unavailable")
        consumer, adapter = _new_checkpoint_endpoint(bundle, checkpoint, endpoint, device)
    runtime_before = _state_digest(consumer.runtime.state_dict())
    adapter_before = _state_digest(adapter.state_dict())
    with torch.no_grad():
        evaluation = N4._evaluate(consumer, adapter, bundle["authorities"]["smooth"], dev_images, dev_labels, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"], step=300, train_trace=None)
        _require(runtime_before == _state_digest(consumer.runtime.state_dict()), f"N4 {endpoint} DEV evaluation changed runtime state")
        _require(adapter_before == _state_digest(adapter.state_dict()), f"N4 {endpoint} DEV evaluation changed adapter state")
        cyclic = V10._cyclic_transfer_v10(consumer, adapter, bundle["authorities"]["smooth"], selection_images, selection_labels, selection_indices, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"])
    _require(runtime_before == _state_digest(consumer.runtime.state_dict()), f"N4 {endpoint} cyclic evaluation changed runtime state")
    _require(adapter_before == _state_digest(adapter.state_dict()), f"N4 {endpoint} cyclic evaluation changed adapter state")
    dev_logits = evaluation["_smooth_logits"].detach().cpu()
    dev_margins = evaluation["_margins"].detach().cpu()
    dev_anchor_margins = evaluation["_anchor_correct_margin"].detach().cpu()
    dev_correct = dev_logits.argmax(dim=1).eq(dev_labels.detach().cpu())
    cyclic_observations = cyclic["_observations"]
    adapter_diagnostics = V10._adapter_observation(adapter)
    generator_diagnostics = _generator_diagnostics(bundle, consumer, endpoint if endpoint != "C0" else "C1")
    endpoint_record = {"endpoint": endpoint, "evaluation": _public(evaluation), "cyclic_transfer": _public(cyclic), "adapter_diagnostics": _public(adapter_diagnostics), "generator_phase_diagnostics": _public(generator_diagnostics), "runtime_state_digest": runtime_before, "adapter_state_digest": adapter_before, "training_updates": 0, "optimizer_updates": 0, "forward_mode": "torch.no_grad", "model_state_modified": False}
    return {"record": endpoint_record, "dev_logits": dev_logits, "dev_margins": dev_margins, "dev_anchor_margins": dev_anchor_margins, "dev_correct": dev_correct, "cyclic_observations": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in cyclic_observations.items()}, "raw_base": cyclic_observations["raw_base"].detach().cpu(), "raw_shifted": cyclic_observations["raw_shifted"].detach().cpu()}


def _paired_statistics(left: torch.Tensor, right: torch.Tensor, labels: torch.Tensor, *, name: str) -> dict[str, Any]:
    delta = right.to(torch.float64) - left.to(torch.float64)
    _require(delta.ndim == 1 and delta.shape == labels.shape, f"{name} paired shape is invalid")
    bootstrap = V10.V6._paired_bootstrap(delta, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    return {"comparison": name, "delta_definition": "right-left per identical DEV row", "observed_delta": float(delta.mean()), "bootstrap": bootstrap, "material": bool(float(delta.mean()) >= MATERIAL_DELTA and float(bootstrap["ci_lower"]) > 0.0), "delta_records": delta}


def _classify_attribution(pairwise: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    c1 = pairwise["C1_minus_C0"]
    c2 = pairwise["C2_minus_C0"]
    extra = pairwise["C2_minus_C1"]
    c1_material = bool(c1["material"])
    c2_material = bool(c2["material"])
    extra_material = bool(extra["material"])
    extra_small = float(extra["bootstrap"]["ci_upper"]) < MATERIAL_DELTA
    if c1_material and extra_small:
        classification = "GENERATOR_COADAPTATION_DOMINANT"
    elif c1_material and extra_material:
        classification = "GENERATOR_COADAPTATION_PLUS_EXTRA_PHASE_CAPACITY_SUPPORTED"
    elif not c1_material and extra_material:
        classification = "EXTRA_PHASE_CONTROL_DOMINANT"
    else:
        classification = "ATTRIBUTION_MIXED_OR_UNRESOLVED"
    return {"classification": classification, "material_delta_threshold": MATERIAL_DELTA, "material_ci_lower_threshold": 0.0, "small_increment_ci_upper_threshold": MATERIAL_DELTA, "rules": {"c1_material": c1_material, "c2_material": c2_material, "extra_material": extra_material, "extra_small": extra_small}}


def _write_tensor_artifact(root: Path, path: Path, payload: Mapping[str, Any], expected_schema: str, artifact_kind: str) -> dict[str, str]:
    N4.atomic_torch_save_new(path, dict(payload), artifact_kind=artifact_kind)
    digest = _sha256(path)
    loaded = N4._mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), artifact_kind)
    _require(loaded.get("schema") == expected_schema, f"{artifact_kind} schema reload changed")
    return {"path": _relative(root, path), "sha256": digest}


def run_evaluation(root: Path) -> dict[str, Any]:
    _require(not EVALUATION_DIR.exists(), f"refusing to reuse endpoint-attribution evaluation directory: {_path(root, EVALUATION_DIR)}")
    preflight = _load_preflight(root)
    started = V10.V6.git_facts(root)
    _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from endpoint attribution preflight")
    contracts = _validate_n4_contracts(root)
    bundle = N4._validate_v11_authorities(root)
    original_reports = _validate_original_reports(root)
    c1 = _validate_checkpoint(root, "C1", _path(root, C1_CHECKPOINT_PATH), C1_CHECKPOINT_SHA256)
    c2 = _validate_checkpoint(root, "C2", _path(root, C2_CHECKPOINT_PATH), C2_CHECKPOINT_SHA256)
    V10.V6._prepare_cuda_runtime_dll_search_path()
    device, device_facts = V10.V6._cuda_facts()
    _require(device.type == "cuda", "CUDA is required for the endpoint attribution evaluator")
    bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
    images, labels, _train_indices, development_indices, dataset = V10.V6._load_roles(root)
    dev_images = images.index_select(0, development_indices).contiguous()
    dev_labels = labels.index_select(0, development_indices).contiguous()
    selection_images, selection_labels, selection_indices = V10.V6._load_selection(root, images, labels, development_indices)
    _require(V10.V6.tensor_sha256(selection_indices) == SELECTION_INDICES_SHA256, "selection indices changed")
    _require(V10.V6.tensor_sha256(selection_labels) == SELECTION_LABELS_SHA256, "selection labels changed")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=False)
    input_payload = {"schema": "lightgen_2d_n4_endpoint_attribution_input.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "contract": {"path": EVALUATION_CONTRACT_PATH.as_posix(), "sha256": EVALUATION_CONTRACT_SHA256}, "endpoints": {"C0": {"authority": "V11 rank9 step300 frozen endpoint", "report": {"path": N4.V11_REPORT_PATH.as_posix(), "sha256": N4.V11_REPORT_SHA256}, "checkpoint": {"path": N4.V11_CHECKPOINT_PATH.as_posix(), "sha256": N4.V11_CHECKPOINT_SHA256}}, "C1": {"checkpoint": {"path": C1_CHECKPOINT_PATH.as_posix(), "sha256": C1_CHECKPOINT_SHA256}, "original_report": original_reports["C1"]["report"]}, "C2": {"checkpoint": {"path": C2_CHECKPOINT_PATH.as_posix(), "sha256": C2_CHECKPOINT_SHA256}, "original_report": original_reports["C2"]["report"]}}, "protocol": {"evaluator": "SmoothMNISTEvaluator v2.1 frozen", "forward": "torch.no_grad", "grid": PRODUCTION_GRID, "batch": LOGICAL_BATCH_SIZE, "dev_rows": int(dev_labels.numel()), "selection_rows": SELECTION_ROWS, "selection_per_class": SELECTION_PER_CLASS, "selection_seed": SELECTION_SEED, "cyclic_target_rule": "t=(y+1) mod 10", "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "material_delta": MATERIAL_DELTA, "objective_context": {"lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "s_tau": S_TAU}}, "data_roles": {"DEV": int(dev_labels.numel()), "TRAIN_for_optimization": 0, "protected": 0, "official_test": 0, "sealed": 0}, "claim_boundary": {"training_updates": 0, "optimizer_updates": 0, "backward": False, "checkpoint_modified": False, "new_model_candidate": False, "stage_b": False, "generation": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False, "original_n4_report_rewritten": False}, "one_shot": {"evaluation_comparison_attempts": 1, "c0_evaluations": 1, "c1_evaluations": 1, "c2_evaluations": 1, "retry": False}}
    input_path = EVALUATION_DIR / INPUT_NAME
    N4.atomic_write_new_json(input_path, input_payload, artifact_kind="N4 endpoint attribution input")
    endpoint_inputs = {"C0": None, "C1": c1, "C2": c2}
    endpoint_results: dict[str, dict[str, Any]] = {}
    for endpoint in ("C0", "C1", "C2"):
        print(json.dumps({"event": "N4_endpoint_attribution_start", "endpoint": endpoint}, ensure_ascii=False), flush=True)
        endpoint_results[endpoint] = _evaluate_endpoint(root, bundle, endpoint, endpoint_inputs[endpoint], dev_images, dev_labels, selection_images, selection_labels, selection_indices, device)
        print(json.dumps({"event": "N4_endpoint_attribution_complete", "endpoint": endpoint}, ensure_ascii=False), flush=True)

    dev_labels_cpu = dev_labels.cpu()
    pairwise: dict[str, Any] = {
        "C1_minus_C0": _paired_statistics(endpoint_results["C0"]["dev_correct"], endpoint_results["C1"]["dev_correct"], dev_labels_cpu, name="R_C1-R_C0"),
        "C2_minus_C0": _paired_statistics(endpoint_results["C0"]["dev_correct"], endpoint_results["C2"]["dev_correct"], dev_labels_cpu, name="R_C2-R_C0"),
        "C2_minus_C1": _paired_statistics(endpoint_results["C1"]["dev_correct"], endpoint_results["C2"]["dev_correct"], dev_labels_cpu, name="R_C2-R_C1"),
    }
    attribution = _classify_attribution(pairwise)

    sufficient_payload: dict[str, Any] = {"schema": "lightgen_2d_n4_endpoint_attribution_sufficient_observations.v1", "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "dev_indices": development_indices.cpu(), "dev_labels": dev_labels_cpu, "selection_indices": selection_indices.cpu(), "selection_labels": selection_labels.cpu(), "endpoints": {}, "paired": {}}
    raw_payload: dict[str, Any] = {"schema": "lightgen_2d_n4_endpoint_attribution_cyclic_raw_outputs.v1", "version": 1, "task_id": TASK_ID, "selection_indices": selection_indices.cpu(), "selection_labels": selection_labels.cpu(), "endpoints": {}}
    for endpoint, result in endpoint_results.items():
        sufficient_payload["endpoints"][endpoint] = {"dev_smooth_logits": result["dev_logits"], "dev_smooth_correct": result["dev_correct"], "dev_ols_margins": result["dev_margins"], "dev_anchor_margins": result["dev_anchor_margins"], "cyclic": {key: value for key, value in result["cyclic_observations"].items() if key not in {"raw_base", "raw_shifted"}}}
        raw_payload["endpoints"][endpoint] = {"raw_base": result["raw_base"], "raw_shifted": result["raw_shifted"]}
    for name, value in pairwise.items():
        sufficient_payload["paired"][name] = value["delta_records"]
    sufficient_ref = _write_tensor_artifact(root, EVALUATION_DIR / SUFFICIENT_NAME, sufficient_payload, "lightgen_2d_n4_endpoint_attribution_sufficient_observations.v1", "N4 endpoint attribution sufficient observations")
    raw_ref = _write_tensor_artifact(root, EVALUATION_DIR / RAW_NAME, raw_payload, "lightgen_2d_n4_endpoint_attribution_cyclic_raw_outputs.v1", "N4 endpoint attribution cyclic raw outputs")

    public_pairwise = {name: {key: value for key, value in record.items() if key != "delta_records"} for name, record in pairwise.items()}
    endpoint_public = {name: result["record"] for name, result in endpoint_results.items()}
    smooth_values = {name: float(result["record"]["evaluation"]["smooth"]["macro_recall"]) for name, result in endpoint_results.items()}
    stage_a = {name: {"R300": value, "threshold": STAGE_A_FIREWALL, "status": "STAGE_A_FIREWALL_PASSED" if value >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False} for name, value in smooth_values.items()}
    report_payload = {"schema": "lightgen_2d_n4_endpoint_attribution_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "VALID_COMPLETED", "status": "valid_completed", "classification": attribution["classification"], "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "provenance": {"git_start": started, "git_end": V10.V6.git_facts(root)}, "preflight": preflight, "contracts": contracts, "original_n4_status": {"aggregate": "UNRESOLVED", "original_c1_report": original_reports["C1"], "original_c2_report": original_reports["C2"], "rewritten": False}, "runtime": {"device": device_facts, "grid": PRODUCTION_GRID, "batch": LOGICAL_BATCH_SIZE, "dataset_roles": _public(dataset)}, "protocol": input_payload["protocol"], "endpoints": endpoint_public, "smooth": smooth_values, "pairwise": public_pairwise, "attribution": attribution, "stage_a": stage_a, "artifacts": {"input": {"path": _relative(root, input_path), "sha256": _sha256(input_path)}, "sufficient_observations": sufficient_ref, "cyclic_raw_outputs": raw_ref}, "claim_boundary": input_payload["claim_boundary"], "training_updates": 0, "optimizer_updates": 0, "scientific_forward_mode": "evaluation-only torch.no_grad; no optimizer/backward", "next": "attribution evidence is bounded; original N4 aggregate remains unresolved and no Stage B or generation is authorized"}
    report_path = EVALUATION_DIR / REPORT_NAME
    N4.atomic_write_new_json(report_path, report_payload, artifact_kind="N4 endpoint attribution report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_payload = {"schema": "lightgen_2d_n4_endpoint_attribution_lineage.v1", "version": 1, "task_id": TASK_ID, "result": report_payload["result"], "status": report_payload["status"], "classification": report_payload["classification"], "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "report": report_ref, "artifacts": report_payload["artifacts"], "endpoint_authorities": report_payload["endpoints"], "claim_boundary": report_payload["claim_boundary"], "scientific_observations_new": True, "training_updates": 0, "optimizer_updates": 0, "original_n4_aggregate_rewritten": False}
    lineage_path = EVALUATION_DIR / LINEAGE_NAME
    N4.atomic_write_new_json(lineage_path, lineage_payload, artifact_kind="N4 endpoint attribution lineage")
    _require(_read_json(report_path, "endpoint attribution report reload").get("classification") == report_payload["classification"], "endpoint attribution report reload changed")
    _require(_read_json(lineage_path, "endpoint attribution lineage reload").get("report", {}).get("sha256") == report_ref["sha256"], "endpoint attribution lineage reload changed")
    return {"result": "VALID_COMPLETED", "status": "valid_completed", "classification": attribution["classification"], "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "training_updates": 0, "optimizer_updates": 0, "original_n4_aggregate": "UNRESOLVED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "evaluate"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[4]).resolve()
    result = run_preflight(root) if args.action == "preflight" else run_evaluation(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
