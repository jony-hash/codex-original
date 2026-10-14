"""Attribute the accepted step-9000 structural/classifier tradeoff.

This is a TRAIN/DEV diagnostic with no optimizer and no persistent model
update.  It consumes the accepted evaluator-agnostic structural-alignment
run, constructs frozen structural and SIC directions at step 9000, probes
them symmetrically on a disjoint TRAIN bank, and evaluates the same temporary
states with the frozen 337D structural authority on a fixed DEV subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import DataReadLedger, TRAIN_ROLE, contracted_plane_powers, optical_power_loss_from_powers, git_identity
from ldm_project.v2.application.research.shared.research_run_control import ResearchRunControl, ResearchRunIdentity, capture_execution_source_manifest
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)

from scripts.research.lightgen import execute_lightgen_2d_attribute_post3000_generator_semantic_realization_ceiling_v1 as prior_attribute
from scripts.research.lightgen import execute_lightgen_2d_evaluator_agnostic_structural_image_alignment_v1 as structural_alignment
from scripts.research.lightgen import execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1 as prior


TASK_ID = "ATTRIBUTE_STEP9000_STRUCTURAL_CLASSIFIER_SEMANTIC_TRADEOFF_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_ATTRIBUTE_STEP9000_STRUCTURAL_CLASSIFIER_SEMANTIC_TRADEOFF_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "STEP9000_STRUCTURAL_CLASSIFIER_SEMANTIC_TRADEOFF_ATTRIBUTED"
SEMANTIC_AUTHORITY_ID = prior.SEMANTIC_AUTHORITY_ID
RUN_ID = "attribute-step9000-structural-classifier-semantic-tradeoff-v1"
TRAJECTORY_ID = f"{RUN_ID}:trajectory"

EXPECTED_BRANCH = prior.EXPECTED_BRANCH
EXPECTED_HEAD = prior.EXPECTED_HEAD

STRUCTURAL_RUN = Path("outputs/20260921_evaluator_agnostic_structural_image_alignment_v1/runs/attempt_03")
STRUCTURAL_REPORT = STRUCTURAL_RUN / "report.json"
STRUCTURAL_REPORT_SHA256 = "83ae194a80699557c40864dcf994e73027f6c1ca7ecbaac5ccf1cd03e46fdc30"
SURROGATE_NORMALIZATION = STRUCTURAL_RUN / "surrogate_normalization.pt"
SURROGATE_NORMALIZATION_SHA256 = "63d0e11aefadd55acb7248b91b631a44a6a9aa4a6da4ca624e35b3276cd2cf98"

CHECKPOINT_STEPS = (6000, 6100, 6200, 6300, 6600, 7000, 7500, 8000, 8500, 9000)
KEY_STEPS = (6000, 6300, 9000)
CHECKPOINT_SHA256 = {
    6000: "85f7a831a821f05faa7fcf4f86be3adb21c9d6327d84d1363ffc13d823ffc8b7",
    6100: "b4670bbc81d66027c9c46b617f1e7aeb070a0753d26d0020778c328a0b3aa52c",
    6200: "98b898acb7e54e164132db3c6e1d339af5b0e34e078456a265447be845af2a89",
    6300: "708b46602e210a3ab2555dd66977f181c1147bb8338f6e050776c90d18157564",
    6600: "51084e65faecffec0d17da39d289278cfd05d251b29f785679144b37c7abb308",
    7000: "09e953c9cabefee7ba80d65ecd338bf17697f771be17b5ea69c12e190839a167",
    7500: "637715a1c563e79c4c2abf2f4545f13d6a318108e39879fdc75a5142c4ab79c8",
    8000: "dc4e476b625b6b3224a98be9ff64a74ac896ef56b5146d107d72266bf4d77da2",
    8500: "d739691a1a27c86a0e2c0ee9d87b4c67b129eca626fd1f3c630a39a5bfc9ffe9",
    9000: "14e76f3273e6f794cbadaee63a881bd686645016c5c5aba62c727ce9860afcee",
}
ENDPOINT_SHA256 = {
    6000: {
        "dev": "6d4835fdcc7b9cdacbb4fc35697179bf6abd5b87b7b24874936bdd2700e6aece",
        "s1": "b2dd4f722ff73783782860bd98c3e2163e071f5603f8626a37c37c2aa5b25eb0",
    },
    6300: {
        "dev": "1822a35becc3d2e2fc7aa409f2c48b733c17a54ebdc2ff8d25b8bee49ed6a59b",
        "s1": "c18d64e76e730f71df4e9142f976ffae0d40d92fc4c722fd392630715d7e96ff",
    },
    9000: {
        "dev": "4f62e0d14ad3e76ae4d9352a8fc1e98719bf73556c004d7d65bda87523dfad01",
        "s1": "e5fc39527b76e7701b3ad8837adcf58899e8b410f1018a8c91e9f16feca2c330",
    },
}

PERTURBATION_EPSILONS = (1.0e-4, 3.0e-4, 1.0e-3)
MICRO_BATCH_SIZE = 10
DEV_ROWS_PER_CLASS = 10
FEATURE_EPS = 1.0e-12
JOINT_DEGENERACY_TOLERANCE = 1.0e-12

REPORT_SCHEMA = "step9000_structural_classifier_semantic_tradeoff_attribution_report.v1"
GRADIENT_SCHEMA = "step9000_structural_classifier_semantic_tradeoff_gradients.v1"
PROBE_SCHEMA = "step9000_structural_classifier_semantic_tradeoff_train_probes.v1"
HARD_PROBE_SCHEMA = "step9000_structural_classifier_semantic_tradeoff_hard_structural_probes.v1"

CLASSIFICATION_A = "SHARED_STRUCTURAL_AND_CLASSIFIER_SEMANTIC_DIRECTION_AVAILABLE"
CLASSIFICATION_B = "STRUCTURAL_VS_CLASSIFIER_OBJECTIVE_CONFLICT_SUPPORTED"
CLASSIFICATION_C = "TWO_LAYER_OPTICAL_JACOBIAN_STRUCTURAL_SEMANTIC_CONFLICT_SUPPORTED"
CLASSIFICATION_D = "OPTIMIZER_STATE_MEDIATED_STRUCTURAL_SEMANTIC_TRADEOFF_SUPPORTED"
CLASSIFICATION_E = "STRUCTURAL_SEMANTIC_PARETO_CONFLICT_UNRESOLVED"
CLASSIFICATION_F = "STEP9000_STRUCTURAL_CLASSIFIER_SEMANTIC_TRADEOFF_MIXED"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260921_attribute_step9000_structural_classifier_semantic_tradeoff_v1/runs/attempt_01"
)


class TradeoffAttributionError(RuntimeError):
    """Raised when a named authority or diagnostic invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TradeoffAttributionError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path if path.is_absolute() else root / path
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_write_new_json(path, dict(payload), artifact_kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _generator(consumer: Any) -> torch.nn.Parameter:
    return prior._generator(consumer)


def _runtime_state(consumer: Any) -> dict[str, torch.Tensor]:
    return prior._runtime_state(consumer)


def _freeze_generator_only(consumer: Any) -> None:
    prior._freeze_generator_only(consumer)


def _clear_grads(consumer: Any, *evaluators: torch.nn.Module) -> None:
    prior._clear_runtime_grads(consumer)
    prior._clear_evaluator_grads(*evaluators)


def _norm(value: torch.Tensor, name: str, *, allow_zero: bool = False) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result) and (allow_zero or result > 0.0), f"{name} is zero or invalid")
    return result


def _rms(value: torch.Tensor, name: str, *, allow_zero: bool = False) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().mean().sqrt().cpu())
    _require(math.isfinite(result) and (allow_zero or result > 0.0), f"{name} RMS is zero or invalid")
    return result


def _cosine(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    left_cpu = left.detach().to(device="cpu", dtype=torch.float64)
    right_cpu = right.detach().to(device="cpu", dtype=torch.float64)
    value = float(torch.sum(left_cpu * right_cpu))
    result = value / (_norm(left, f"{name} left") * _norm(right, f"{name} right"))
    _require(math.isfinite(result), f"{name} is nonfinite")
    return result


def _summary(value: torch.Tensor, name: str) -> dict[str, Any]:
    flat = value.detach().cpu().to(torch.float64).reshape(-1)
    _finite(flat, name)
    _require(flat.numel() > 0, f"{name} is empty")
    quantiles = torch.quantile(flat, torch.tensor([0.01, 0.50, 0.99], dtype=torch.float64))
    return {
        "count": int(flat.numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std_population": float(flat.std(unbiased=False)),
        "quantiles": {"0.01": float(quantiles[0]), "0.5": float(quantiles[1]), "0.99": float(quantiles[2])},
    }


def _gradient_stats(value: torch.Tensor, name: str, *, phase: bool = False) -> dict[str, Any]:
    _finite(value, name)
    result: dict[str, Any] = {
        "norm": _norm(value, name),
        "rms": _rms(value, name),
        "shape": list(value.shape),
        "finite_fraction": float(torch.isfinite(value).to(torch.float64).mean().cpu()),
        "zero_fraction": float((value == 0.0).to(torch.float64).mean().cpu()),
    }
    if phase:
        result["per_phase_plane_norms"] = {
            f"generator.phase[{index}]": _norm(value[index], f"{name}[{index}]")
            for index in range(int(value.shape[0]))
        }
    return result


def _normalize_descent(value: torch.Tensor, name: str) -> torch.Tensor:
    direction = -value.detach() / _rms(value, name)
    _require(math.isclose(_rms(direction, f"{name} normalized"), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-7), f"{name} normalization failed")
    return direction


def _class_geometry(values: Mapping[str, torch.Tensor], name: str) -> dict[str, Any]:
    required = {"structural", "E_S", "E_I", "E_C"}
    _require(set(values) == required, f"{name} vector set changed")
    norms = {key: _norm(value, f"{name} {key}") for key, value in values.items()}
    units = {key: value.detach() / norms[key] for key, value in values.items()}
    pairwise = {
        "structural_E_S": _cosine(units["structural"], units["E_S"], f"{name} structural/E_S"),
        "structural_E_I": _cosine(units["structural"], units["E_I"], f"{name} structural/E_I"),
        "structural_E_C": _cosine(units["structural"], units["E_C"], f"{name} structural/E_C"),
        "E_S_E_I": _cosine(units["E_S"], units["E_I"], f"{name} E_S/E_I"),
        "E_S_E_C": _cosine(units["E_S"], units["E_C"], f"{name} E_S/E_C"),
        "E_I_E_C": _cosine(units["E_I"], units["E_C"], f"{name} E_I/E_C"),
    }
    v_sic = units["E_S"] + units["E_I"] + units["E_C"]
    v_sic_norm = _norm(v_sic, f"{name} v_SIC")
    u_sic = v_sic / v_sic_norm
    v_joint = units["structural"] + u_sic
    v_joint_norm = _norm(v_joint, f"{name} v_joint")
    return {
        "gradient_norms": norms,
        "pairwise_cosines": pairwise,
        "v_SIC": {"l2_norm": v_sic_norm, "rms": _rms(v_sic, f"{name} v_SIC")},
        "u_SIC": {"l2_norm": _norm(u_sic, f"{name} u_SIC"), "rms": _rms(u_sic, f"{name} u_SIC")},
        "v_joint": {"l2_norm": v_joint_norm, "rms": _rms(v_joint, f"{name} v_joint")},
        "cos_structural_v_SIC": _cosine(units["structural"], v_sic, f"{name} structural/v_SIC"),
        "cos_structural_u_SIC": _cosine(units["structural"], u_sic, f"{name} structural/u_SIC"),
        "cosine_to_common_resultant": {key: _cosine(value, v_joint, f"{name} {key}/v_joint") for key, value in units.items()},
        "common_direction_defined": True,
    }


def _class_summary(labels: torch.Tensor, indices: torch.Tensor, tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    labels = labels.index_select(0, indices).cpu().to(torch.long)
    result: dict[str, Any] = {}
    for digit in range(10):
        positions = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(positions.numel() > 0, f"TRAIN A class {digit} is empty")
        item: dict[str, Any] = {"rows": int(positions.numel())}
        for name, value in tensors.items():
            selected = value.index_select(0, positions)
            row_norms = selected.to(torch.float64).flatten(1).square().sum(dim=1).sqrt()
            item[f"{name}_row_norm"] = _summary(row_norms, f"class {digit} {name} row norm")
        pairs = {}
        names = list(tensors)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                left = tensors[left_name].index_select(0, positions).flatten(1)
                right = tensors[right_name].index_select(0, positions).flatten(1)
                pairs[f"{left_name}_{right_name}"] = float(
                    F.cosine_similarity(left.to(torch.float64), right.to(torch.float64), dim=1).mean()
                )
        item["rowwise_pairwise_cosines"] = pairs
        result[str(digit)] = item
    return result


def _load_checkpoint(root: Path, run_dir: Path, step: int) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    path = root / run_dir / f"scientific_step_{step:06d}.pt"
    expected = CHECKPOINT_SHA256[step]
    _require(file_sha256(path) == expected, f"accepted step-{step} checkpoint identity changed")
    payload = _mapping(safe_torch_load(path, expected_sha256=expected, map_location="cpu"), f"step-{step} checkpoint")
    _require(payload.get("schema") == structural_alignment.CHECKPOINT_SCHEMA, f"step-{step} checkpoint schema changed")
    _require(payload.get("completed_updates") == step, f"step-{step} completed update count changed")
    _require(payload.get("scientific_artifact") is True and payload.get("operational_artifact") is False, f"step-{step} artifact role changed")
    _require(payload.get("trainable_parameter_names") == ["generator.phase"], f"step-{step} trainable boundary changed")
    encoder = _tensor(payload.get("encoder_state"), f"step-{step} encoder state")
    generator = _tensor(payload.get("generator_state"), f"step-{step} generator state")
    _require(encoder.shape == generator.shape == (2, 1, 100, 100), f"step-{step} phase geometry changed")
    _require(encoder.dtype == generator.dtype == torch.float32, f"step-{step} phase dtype changed")
    _finite(encoder, f"step-{step} encoder state")
    _finite(generator, f"step-{step} generator state")
    optimizer_configuration = _mapping(payload.get("optimizer_configuration"), f"step-{step} optimizer configuration")
    _require(float(optimizer_configuration.get("internal_step", -1.0)) == float(step + 300), f"step-{step} Adam internal step changed")
    _require(optimizer_configuration.get("parameter_names") == ["generator.phase"], f"step-{step} Adam parameter boundary changed")
    _require(isinstance(payload.get("optimizer_state"), Mapping), f"step-{step} full Adam state missing")
    return {"encoder.phase": encoder.cpu(), "generator.phase": generator.cpu()}, payload


def _load_authorities(root: Path) -> dict[str, Any]:
    local_state = git_identity(root)
    _require(local_state["branch"] == EXPECTED_BRANCH and local_state["head"] == EXPECTED_HEAD, "branch/HEAD conflicts with accepted authorities")
    authorities = structural_alignment._load_authorities(root)
    report_path = root / STRUCTURAL_REPORT
    _require(file_sha256(report_path) == STRUCTURAL_REPORT_SHA256, "accepted structural alignment report identity changed")
    report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "accepted structural alignment report")
    _require(report.get("result") == "VALID_COMPLETED" and report.get("status") == "VALID_COMPLETED", "accepted structural alignment is not valid completed")
    _require(report.get("classification") == structural_alignment.CLASSIFICATION_D, "accepted structural tradeoff classification changed")
    _require(report.get("stage2", {}).get("executed") is True and report.get("stage2", {}).get("final_step") == 9000, "accepted step9000 endpoint changed")
    _require(report.get("training_trajectory", {}).get("checkpoint_steps_persisted") == list(CHECKPOINT_STEPS), "accepted checkpoint schedule changed")
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    payloads: dict[int, Mapping[str, Any]] = {}
    for step in CHECKPOINT_STEPS:
        checkpoints[step], payloads[step] = _load_checkpoint(root, STRUCTURAL_RUN, step)
    encoder0 = checkpoints[6000]["encoder.phase"]
    for step in CHECKPOINT_STEPS:
        _require(torch.equal(encoder0, checkpoints[step]["encoder.phase"]), f"encoder changed at step {step}")
    for step in CHECKPOINT_STEPS[1:]:
        _require(not torch.equal(checkpoints[6000]["generator.phase"], checkpoints[step]["generator.phase"]), f"generator did not change at step {step}")

    endpoint_records: dict[int, dict[str, Mapping[str, Any]]] = {}
    for step in KEY_STEPS:
        endpoint_records[step] = {}
        for kind, filename in (("dev", f"dev_diagnostics_step_{step:06d}.json"), ("s1", f"s1_transfer_step_{step:06d}.json")):
            path = root / STRUCTURAL_RUN / filename
            _require(file_sha256(path) == ENDPOINT_SHA256[step][kind], f"accepted step-{step} {kind} diagnostic identity changed")
            endpoint_records[step][kind] = _mapping(json.loads(path.read_text(encoding="utf-8")), f"step-{step} {kind} diagnostic")
    bank = prior_attribute._load_diagnostic_bank(root, authorities)
    _require(bank["a_indices"].numel() == bank["b_indices"].numel() == 20, "accepted A/B bank size changed")
    _require(torch.unique(torch.cat((bank["a_indices"], bank["b_indices"]))).numel() == 40, "accepted A/B bank overlaps")
    _require(torch.bincount(bank["a_labels"], minlength=10).tolist() == [2] * 10, "accepted A bank balance changed")
    _require(torch.bincount(bank["b_labels"], minlength=10).tolist() == [2] * 10, "accepted B bank balance changed")
    authorities.update({
        "local_state": local_state,
        "structural_run_report": report,
        "structural_run_dir": root / STRUCTURAL_RUN,
        "checkpoints": checkpoints,
        "checkpoint_payloads": payloads,
        "endpoint_records": endpoint_records,
        "attribution_bank": bank,
        "authority_hashes": {
            **authorities["authority_hashes"],
            "accepted_structural_alignment_report": STRUCTURAL_REPORT_SHA256,
            "accepted_structural_alignment_surrogate_normalization": SURROGATE_NORMALIZATION_SHA256,
            **{f"accepted_structural_step_{step}": CHECKPOINT_SHA256[step] for step in CHECKPOINT_STEPS},
            **{f"accepted_structural_{kind}_step_{step}": ENDPOINT_SHA256[step][kind] for step in KEY_STEPS for kind in ("dev", "s1")},
        },
    })
    return authorities


def _load_models(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[Any, Any, Any, Any, Any]:
    prior.robust._prepare_cuda_runtime()
    consumer, smooth, independent, ec, ed = prior._load_models(root, authorities, device)
    consumer.runtime.eval()
    smooth.eval()
    independent.eval()
    ec.eval()
    ed.eval()
    for name, model in (("E_S", smooth), ("E_I", independent), ("E_C", ec), ("E_D", ed)):
        _require(all(not parameter.requires_grad for parameter in model.parameters()), f"{name} is not frozen")
    return consumer, smooth, independent, ec, ed


def _load_surrogate_normalization(root: Path, authorities: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    run_path = root / SURROGATE_NORMALIZATION
    _require(file_sha256(run_path) == SURROGATE_NORMALIZATION_SHA256, "accepted surrogate normalization identity changed")
    payload = _mapping(safe_torch_load(run_path, expected_sha256=SURROGATE_NORMALIZATION_SHA256, map_location="cpu"), "accepted surrogate normalization")
    mean = _tensor(payload.get("mean"), "surrogate mean")
    scale = _tensor(payload.get("scale"), "surrogate scale")
    _require(payload.get("train_only") is True and mean.shape == scale.shape == (structural_alignment.STRUCTURAL_DIMENSION,), "surrogate normalization contract changed")
    fitted_mean, fitted_scale = structural_alignment._fit_surrogate_normalization(authorities["images"].index_select(0, authorities["train_indices"]))
    _require(torch.equal(mean, fitted_mean) and torch.equal(scale, fitted_scale), "accepted TRAIN surrogate normalization changed")
    return mean, scale


def _select_dev_subset(labels: torch.Tensor, development_indices: torch.Tensor) -> torch.Tensor:
    labels_dev = labels.index_select(0, development_indices)
    selected: list[torch.Tensor] = []
    for digit in range(10):
        positions = torch.nonzero(labels_dev == digit, as_tuple=False).flatten()
        _require(positions.numel() >= DEV_ROWS_PER_CLASS, f"DEV class {digit} has too few rows")
        selected.append(development_indices.index_select(0, positions[:DEV_ROWS_PER_CLASS]))
    result = torch.cat(selected).to(torch.long).cpu()
    _require(result.numel() == 100 and torch.bincount(labels.index_select(0, result), minlength=10).tolist() == [DEV_ROWS_PER_CLASS] * 10, "fixed DEV subset is not class balanced")
    return result


def _gradient_batch(consumer: Any, smooth: Any, independent: Any, ec: Any, batch: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor) -> dict[str, Any]:
    generator = _generator(consumer)
    structural_bundle = structural_alignment._structural_bundle(consumer, batch, references, mean, scale)
    semantic_bundle = prior._loss_bundle(consumer, smooth, independent, ec, batch, references)
    _require(torch.allclose(structural_bundle["raw"].detach(), semantic_bundle["raw"].detach(), rtol=0.0, atol=1.0e-6), "structural and semantic forward paths disagree")
    losses = {
        "structural": structural_bundle["losses"]["L_struct"],
        "E_S": semantic_bundle["losses"]["L_S"],
        "E_I": semantic_bundle["losses"]["L_I"],
        "E_C": semantic_bundle["losses"]["L_C"],
    }
    outputs = {"structural": structural_bundle["raw"], "E_S": semantic_bundle["raw"], "E_I": semantic_bundle["raw"], "E_C": semantic_bundle["raw"]}
    h: dict[str, torch.Tensor] = {}
    g: dict[str, torch.Tensor] = {}
    h["structural"], g["structural"] = torch.autograd.grad(losses["structural"], (outputs["structural"], generator), retain_graph=False, allow_unused=False)
    semantic_names = ("E_S", "E_I", "E_C")
    for index, name in enumerate(semantic_names):
        h[name], g[name] = torch.autograd.grad(losses[name], (outputs[name], generator), retain_graph=index < len(semantic_names) - 1, allow_unused=False)
    for name, value in {**h, **g}.items():
        _finite(value, f"gradient batch {name}")
    target_image = structural_bundle["result"].image_input.slm_intensity if structural_bundle["result"].image_input is not None else None
    _require(target_image is not None, "gradient batch target image is missing")
    return {"losses": losses, "h": {key: value.detach() for key, value in h.items()}, "g": {key: value.detach() for key, value in g.items()}, "raw": structural_bundle["raw"].detach(), "target_image": target_image.detach()}


def _analyze_step(consumer: Any, smooth: Any, independent: Any, ec: Any, images: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, references: Any, checkpoint: Mapping[str, torch.Tensor], mean: torch.Tensor, scale: torch.Tensor, step: int, ledger: DataReadLedger, *, classwise: bool = False) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()
    smooth.eval()
    independent.eval()
    ec.eval()
    _require(indices.numel() % MICRO_BATCH_SIZE == 0, "TRAIN A bank is not divisible by the fixed micro-batch")
    row_count = int(indices.numel())
    h_chunks = {name: [] for name in ("structural", "E_S", "E_I", "E_C")}
    g_accum = {name: torch.zeros_like(_generator(consumer)) for name in ("structural", "E_S", "E_I", "E_C")}
    scalar_totals = {name: 0.0 for name in ("structural", "E_S", "E_I", "E_C")}
    labels_a = labels.index_select(0, indices).cpu()
    for start in range(0, row_count, MICRO_BATCH_SIZE):
        batch_indices = indices[start : start + MICRO_BATCH_SIZE]
        batch = images.index_select(0, batch_indices).to(consumer.device)
        values = _gradient_batch(consumer, smooth, independent, ec, batch, references, mean, scale)
        weight = batch_indices.numel() / row_count
        for name in h_chunks:
            h_chunks[name].append((weight * values["h"][name]).cpu())
            g_accum[name] += weight * values["g"][name]
            scalar_totals[name] += weight * float(values["losses"][name].cpu())
        ledger.record(TRAIN_ROLE, int(batch_indices.numel()))
        _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), f"step {step} populated runtime .grad")
        _clear_grads(consumer, smooth, independent, ec)
    h_all = {name: torch.cat(values, dim=0) for name, values in h_chunks.items()}
    for name, value in {**h_all, **g_accum}.items():
        _finite(value, f"step {step} {name}")
    output_geometry = _class_geometry(h_all, f"step {step} output")
    phase_geometry = _class_geometry(g_accum, f"step {step} phase")
    class_summary = _class_summary(labels, indices, h_all)
    phase_class_summary: dict[str, Any] = {}
    if classwise:
        for digit in range(10):
            positions = torch.nonzero(labels_a == digit, as_tuple=False).flatten()
            class_indices = indices.index_select(0, positions)
            values = _gradient_batch(consumer, smooth, independent, ec, images.index_select(0, class_indices).to(consumer.device), references, mean, scale)
            phase_values = values["g"]
            phase_class_summary[str(digit)] = {
                "rows": int(class_indices.numel()),
                "gradient_norms": {name: _norm(value, f"class {digit} {name}") for name, value in phase_values.items()},
                "pairwise_cosines": {f"{left}_{right}": _cosine(phase_values[left], phase_values[right], f"class {digit} {left}/{right}") for left, right in (("structural", "E_S"), ("structural", "E_I"), ("structural", "E_C"), ("E_S", "E_I"), ("E_S", "E_C"), ("E_I", "E_C"))},
            }
            ledger.record(TRAIN_ROLE, int(class_indices.numel()))
            _clear_grads(consumer, smooth, independent, ec)
    result = {
        "step": step,
        "rows": row_count,
        "losses": scalar_totals,
        "output_space": {**output_geometry, "gradient_stats": {name: _gradient_stats(value, f"step {step} h_{name}") for name, value in h_all.items()}, "class_summary": class_summary},
        "phase_space": {**phase_geometry, "gradient_stats": {name: _gradient_stats(value, f"step {step} g_{name}", phase=True) for name, value in g_accum.items()}, "class_summary": phase_class_summary},
        "train_labels": torch.bincount(labels_a, minlength=10).tolist(),
        "trainable_parameter_names": ["generator.phase"],
        "frozen_evaluator_parameters": True,
        "E_D_in_backward": False,
    }
    tensors = {**{f"h_{name}": value for name, value in h_all.items()}, **{f"g_{name}": value for name, value in g_accum.items()}}
    return result, tensors


def _loss_values_no_grad(consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, images: torch.Tensor, labels: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor, ledger: DataReadLedger, *, role: str) -> dict[str, Any]:
    total_rows = int(images.shape[0])
    totals = {name: 0.0 for name in ("L_struct", "L_S", "L_I", "L_C", "L_rec", "L_op")}
    logits = {name: [] for name in ("E_S", "E_I", "E_C", "E_D")}
    raw_chunks: list[torch.Tensor] = []
    consumer.runtime.eval()
    smooth.eval()
    independent.eval()
    ec.eval()
    ed.eval()
    with torch.inference_mode():
        for start in range(0, total_rows, MICRO_BATCH_SIZE):
            batch = images[start : start + MICRO_BATCH_SIZE].to(consumer.device)
            result = consumer.runtime.forward_from_image(batch)
            raw = result.raw_intensity
            target_image = result.image_input.slm_intensity if result.image_input is not None else None
            _require(target_image is not None, "no-gradient target image is missing")
            raw_view = prior._cmos_view(raw)
            target_view = prior._cmos_view(batch)
            s_features, s_logits = smooth(raw_view)
            i_features, i_logits = independent(raw_view)
            c_features, c_logits = ec(raw_view)
            d_logits = ed(raw_view)
            s_target, _ = smooth(target_view)
            i_target, _ = independent(target_view)
            c_target, _ = ec(target_view)
            generated = structural_alignment._normalized_surrogate(structural_alignment._structural_surrogate(raw_view), mean.to(consumer.device), scale.to(consumer.device))
            target = structural_alignment._normalized_surrogate(structural_alignment._structural_surrogate(target_view), mean.to(consumer.device), scale.to(consumer.device))
            values = {
                "L_struct": (generated - target).square().mean(),
                "L_S": prior._normalized_feature_loss(s_features, s_target, "L_S"),
                "L_I": prior._normalized_feature_loss(i_features, i_target, "L_I"),
                "L_C": prior._normalized_feature_loss(c_features, c_target, "L_C"),
                "L_rec": (raw - target_image).square().mean(),
                "L_op": optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference),
            }
            for name, value in values.items():
                _finite(value, f"no-gradient {name}")
                totals[name] += float(value.cpu()) * batch.shape[0] / total_rows
            for name, value in (("E_S", s_logits), ("E_I", i_logits), ("E_C", c_logits), ("E_D", d_logits)):
                logits[name].append(value.cpu())
            raw_chunks.append(raw.cpu())
            ledger.record(role, int(batch.shape[0]))
    raw_all = torch.cat(raw_chunks, dim=0)
    metrics = {name: prior._class_metrics(labels, torch.cat(values).argmax(dim=1), label_source="canonical TRAIN label") for name, values in logits.items()}
    return {"losses": totals, "semantic": metrics, "raw_validity": prior._raw_validity(raw_all), "rows": total_rows}


def _metric_deltas(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(candidate["losses"][name] - base["losses"][name]) for name in ("L_struct", "L_S", "L_I", "L_C", "L_rec", "L_op")}


def _stable_descent(records: Mapping[str, Any], direction: str, metric: str) -> bool:
    values = records.get(direction, [])
    return bool(values) and all(float(item["plus"]["deltas"][metric]) < 0.0 and float(item["minus"]["deltas"][metric]) > 0.0 for item in values)


def _directions(analysis: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    g_struct = tensors["g_structural"]
    g_sic = tensors["g_E_S"] / _norm(tensors["g_E_S"], "g_E_S") + tensors["g_E_I"] / _norm(tensors["g_E_I"], "g_E_I") + tensors["g_E_C"] / _norm(tensors["g_E_C"], "g_E_C")
    u_struct = g_struct / _norm(g_struct, "g_structural")
    u_sic = g_sic / _norm(g_sic, "v_SIC")
    v_joint = u_struct + u_sic
    _require(_norm(v_joint, "v_joint") > JOINT_DEGENERACY_TOLERANCE, "step9000 joint direction is degenerate")
    directions = {"d_struct": _normalize_descent(u_struct, "d_struct"), "d_SIC": _normalize_descent(u_sic, "d_SIC"), "d_joint": _normalize_descent(v_joint, "d_joint")}
    details = {
        "u_struct_l2_norm": _norm(u_struct, "u_struct"),
        "v_SIC_l2_norm": _norm(g_sic, "v_SIC"),
        "u_SIC_l2_norm": _norm(u_sic, "u_SIC"),
        "v_joint_l2_norm": _norm(v_joint, "v_joint"),
        "cos_structural_v_SIC": _cosine(g_struct, g_sic, "g_structural/v_SIC"),
        "definitions": {"u_struct": "g_struct/||g_struct||_2", "v_SIC": "sum(g_k/||g_k||_2), k in {S,I,C}", "u_SIC": "v_SIC/||v_SIC||_2", "v_joint": "u_struct+u_SIC", "d_struct": "-u_struct/RMS(u_struct)", "d_SIC": "-u_SIC/RMS(u_SIC)", "d_joint": "-v_joint/RMS(v_joint)"},
    }
    return directions, details


def _run_train_probes(consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, images: torch.Tensor, labels: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor, checkpoint: Mapping[str, torch.Tensor], directions: Mapping[str, torch.Tensor], ledger: DataReadLedger) -> dict[str, Any]:
    consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    base_state = _runtime_state(consumer)
    base = _loss_values_no_grad(consumer, smooth, independent, ec, ed, images, labels, references, mean, scale, ledger, role=TRAIN_ROLE)
    records: dict[str, list[dict[str, Any]]] = {}
    try:
        for direction_name, direction in directions.items():
            epsilon_records: list[dict[str, Any]] = []
            for epsilon in PERTURBATION_EPSILONS:
                signed: dict[str, Any] = {}
                for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                    consumer.load_trainable_state_dict(checkpoint)
                    _freeze_generator_only(consumer)
                    with torch.no_grad():
                        _generator(consumer).add_(sign * epsilon * direction.to(consumer.device))
                    candidate = _loss_values_no_grad(consumer, smooth, independent, ec, ed, images, labels, references, mean, scale, ledger, role=TRAIN_ROLE)
                    signed[sign_name] = {"deltas": _metric_deltas(base, candidate), "losses": candidate["losses"], "semantic": {name: candidate["semantic"][name]["macro_recall"] for name in ("E_S", "E_I", "E_C", "E_D")}, "raw_validity": candidate["raw_validity"]}
                    consumer.load_trainable_state_dict(checkpoint)
                    _freeze_generator_only(consumer)
                    _require(_state_digest(_runtime_state(consumer)) == _state_digest(base_state), f"{direction_name} train probe was not restored")
                signed["central_slope_per_epsilon"] = {metric: float((signed["plus"]["deltas"][metric] - signed["minus"]["deltas"][metric]) / (2.0 * epsilon)) for metric in ("L_struct", "L_S", "L_I", "L_C", "L_rec", "L_op")}
                epsilon_records.append({"epsilon": epsilon, **signed})
            records[direction_name] = epsilon_records
    finally:
        consumer.load_trainable_state_dict(checkpoint)
        _freeze_generator_only(consumer)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(base_state), "train probes did not restore step9000")
    return {"schema": PROBE_SCHEMA, "rows": int(images.shape[0]), "epsilon_definition": "epsilon RMS radians added to generator.phase", "epsilons": list(PERTURBATION_EPSILONS), "base": base, "directions": records, "stable_descent": {direction: {metric: _stable_descent(records, direction, metric) for metric in ("L_struct", "L_S", "L_I", "L_C", "L_rec", "L_op")} for direction in records}, "checkpoint_restored_after_every_probe": True}


def _evaluate_hard_dev(consumer: Any, ed: Any, authorities: Mapping[str, Any], indices: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, reference: Mapping[str, Any], device: torch.device, ledger: DataReadLedger, *, name: str, checkpoint: Mapping[str, torch.Tensor] | None = None) -> dict[str, Any]:
    if checkpoint is not None:
        consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    labels = authorities["labels"].index_select(0, indices).cpu()
    views: list[torch.Tensor] = []
    ed_logits: list[torch.Tensor] = []
    raw_nonconstant = 0
    raw_zero = 0
    with torch.inference_mode():
        for start in range(0, int(indices.numel()), MICRO_BATCH_SIZE):
            batch_indices = indices[start : start + MICRO_BATCH_SIZE]
            batch = authorities["images"].index_select(0, batch_indices).to(device)
            result = consumer.runtime.forward_from_image(batch)
            raw = result.raw_intensity
            _finite(raw, f"{name} raw")
            view = prior._cmos_view(raw)
            views.append(view.cpu())
            ed_logits.append(ed(view).cpu())
            flat = raw.to(torch.float64).flatten(1)
            raw_nonconstant += int((flat.std(dim=1, unbiased=False) > 0.0).sum())
            raw_zero += int((flat.abs().sum(dim=1) == 0.0).sum())
            ledger.record("dev", int(batch_indices.numel()))
    all_views = torch.cat(views, dim=0)
    source, public = structural_alignment._hard_structural_source(name, all_views, labels, reference, device)
    log_rho = source["nearest_same_class_distance"].to(torch.float64) / source["q95_rows"].to(torch.float64)
    _finite(log_rho, f"{name} log rho")
    return {"name": name, "rows": int(indices.numel()), "structural": {key: value for key, value in public.items() if not key.startswith("_")}, "macro_log_rho": float(log_rho.log().mean()), "ssr": float(public["structural_support_rate"]), "E_D": prior._class_metrics(labels, torch.cat(ed_logits).argmax(dim=1), label_source="canonical DEV label"), "raw_validity": {"finite": True, "nonconstant_fraction": float(raw_nonconstant / int(indices.numel())), "zero_image_count": raw_zero}, "indices_sha256": tensor_sha256(indices)}


def _run_hard_probes(consumer: Any, ed: Any, authorities: Mapping[str, Any], indices: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, checkpoint: Mapping[str, torch.Tensor], directions: Mapping[str, torch.Tensor], reference: Mapping[str, Any], device: torch.device, ledger: DataReadLedger) -> dict[str, Any]:
    base = _evaluate_hard_dev(consumer, ed, authorities, indices, mean, scale, reference, device, ledger, name="step9000_base", checkpoint=checkpoint)
    records: dict[str, list[dict[str, Any]]] = {}
    base_state = _runtime_state(consumer)
    try:
        for direction_name, direction in directions.items():
            epsilon_records: list[dict[str, Any]] = []
            for epsilon in PERTURBATION_EPSILONS:
                signed: dict[str, Any] = {}
                for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                    consumer.load_trainable_state_dict(checkpoint)
                    _freeze_generator_only(consumer)
                    with torch.no_grad():
                        _generator(consumer).add_(sign * epsilon * direction.to(device))
                    candidate = _evaluate_hard_dev(consumer, ed, authorities, indices, mean, scale, reference, device, ledger, name=f"step9000_{direction_name}_{sign_name}_{epsilon:g}")
                    signed[sign_name] = {"macro_log_rho": candidate["macro_log_rho"], "ssr": candidate["ssr"], "delta_macro_log_rho": candidate["macro_log_rho"] - base["macro_log_rho"], "delta_ssr": candidate["ssr"] - base["ssr"], "support_by_class": candidate["structural"]["support_by_class"], "raw_validity": candidate["raw_validity"]}
                    consumer.load_trainable_state_dict(checkpoint)
                    _freeze_generator_only(consumer)
                    _require(_state_digest(_runtime_state(consumer)) == _state_digest(base_state), f"{direction_name} hard probe was not restored")
                signed["central_slope_per_epsilon"] = {"macro_log_rho": float((signed["plus"]["delta_macro_log_rho"] - signed["minus"]["delta_macro_log_rho"]) / (2.0 * epsilon)), "ssr": float((signed["plus"]["delta_ssr"] - signed["minus"]["delta_ssr"]) / (2.0 * epsilon))}
                epsilon_records.append({"epsilon": epsilon, **signed})
            records[direction_name] = epsilon_records
    finally:
        consumer.load_trainable_state_dict(checkpoint)
        _freeze_generator_only(consumer)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(base_state), "hard probes did not restore step9000")
    return {"schema": HARD_PROBE_SCHEMA, "dev_subset_rule": "first 10 canonical DEV rows per class", "rows": int(indices.numel()), "base": base, "directions": records, "checkpoint_restored_after_every_probe": True, "DEV_gradients": False}


def _endpoint_summary(record: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    structural = _mapping(record.get("structural"), f"{kind} structural endpoint")
    summary: dict[str, Any] = {
        "step": int(record["step"]),
        "rows": int(record["rows"]),
        "macro_log_rho": float(_mapping(structural.get("log_rho_struct"), f"{kind} log rho").get("mean")),
        "ssr": float(structural.get("structural_support_rate")),
        "reconstruction": record.get("L_rec"),
        "optical": record.get("L_op"),
        "raw_validity": record.get("raw_cmos") if kind == "DEV" else record.get("raw_validity"),
    }
    metrics = record.get("semantic") if kind == "S1" else record
    summary["macro_recall"] = {name: float(_mapping(metrics.get(name), f"{kind} {name}").get("macro_recall")) for name in ("E_S", "E_I", "E_C", "E_D")} if kind == "DEV" else {name: float(_mapping(_mapping(metrics.get(name), f"{kind} {name}"), "semantic metric").get("macro_recall")) for name in ("E_S", "E_I", "E_C", "E_D")}
    if kind == "S1":
        summary["R_raw"] = record.get("R_raw")
        summary["R_feature"] = record.get("R_feature")
    return summary


def _optimizer_state_summary(payload: Mapping[str, Any], directions: Mapping[str, torch.Tensor] | None = None) -> dict[str, Any]:
    optimizer = _mapping(payload.get("optimizer_state"), "optimizer state")
    state = _mapping(optimizer.get("state"), "optimizer slots")
    slot = state.get(0, state.get("0"))
    slot = _mapping(slot, "generator Adam slot")
    exp_avg = _tensor(slot.get("exp_avg"), "Adam exp_avg").to(torch.float64)
    exp_avg_sq = _tensor(slot.get("exp_avg_sq"), "Adam exp_avg_sq").to(torch.float64)
    step = float(_tensor(slot.get("step"), "Adam step").item())
    _finite(exp_avg, "Adam exp_avg")
    _finite(exp_avg_sq, "Adam exp_avg_sq")
    groups = optimizer.get("param_groups")
    _require(isinstance(groups, list) and groups and isinstance(groups[0], Mapping), "Adam parameter groups are malformed")
    group = groups[0]
    beta1, beta2 = (float(value) for value in group.get("betas", [0.9, 0.999]))
    epsilon = float(group.get("eps", 1.0e-8))
    learning_rate = float(group.get("lr", 1.0e-3))
    effective_gradient = exp_avg / (1.0 - beta1**step) / ((exp_avg_sq / (1.0 - beta2**step)).sqrt() + epsilon)
    effective_descent = -effective_gradient
    result = {
        "internal_step": step,
        "first_moment": _summary(exp_avg, "Adam first moment"),
        "second_moment": _summary(exp_avg_sq, "Adam second moment"),
        "effective_gradient_rms": _rms(effective_gradient, "Adam effective gradient"),
        "effective_descent_rms": _rms(effective_descent, "Adam effective descent"),
        "learning_rate": learning_rate,
        "betas": [beta1, beta2],
        "eps": epsilon,
        "direction_cosines": {},
    }
    if directions:
        for name, value in directions.items():
            result["direction_cosines"][f"effective_gradient_vs_{name}"] = _cosine(effective_gradient, value, f"Adam gradient/{name}")
            result["direction_cosines"][f"effective_descent_vs_{name}"] = _cosine(effective_descent, value, f"Adam descent/{name}")
    return result


def _build_trajectory(authorities: Mapping[str, Any], analyses: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    endpoints = authorities["endpoint_records"]
    records: dict[str, Any] = {}
    for step in CHECKPOINT_STEPS:
        payload = authorities["checkpoint_payloads"][step]
        tail = payload.get("history_tail", [])
        last = tail[-1] if tail and int(tail[-1].get("step", -1)) == step else None
        item: dict[str, Any] = {
            "step": step,
            "checkpoint_sha256": CHECKPOINT_SHA256[step],
            "train_update_tail": None if last is None else {key: last.get(key) for key in ("step", "L_struct", "L_rec", "L_op", "phase_update_rms", "phase_displacement_from_step6000", "structural_gradient_degenerate")},
            "optimizer": _optimizer_state_summary(payload, None if step not in analyses else {"g_structural": analyses[step]["_g_structural"], "v_SIC": analyses[step]["_v_SIC"]}),
            "real_DEV_endpoint": None,
            "S1_endpoint": None,
        }
        if step in endpoints:
            item["real_DEV_endpoint"] = _endpoint_summary(endpoints[step]["dev"], kind="DEV")
            item["S1_endpoint"] = _endpoint_summary(endpoints[step]["s1"], kind="S1")
        records[str(step)] = item
    early = {name: records["6300"]["real_DEV_endpoint"]["macro_recall"][name] - records["6000"]["real_DEV_endpoint"]["macro_recall"][name] for name in ("E_S", "E_I", "E_C", "E_D")}
    final = {name: records["9000"]["real_DEV_endpoint"]["macro_recall"][name] - records["6000"]["real_DEV_endpoint"]["macro_recall"][name] for name in ("E_S", "E_I", "E_C", "E_D")}
    if all(value < 0.0 for value in early.values()):
        timing = "immediate"
    elif any(value < 0.0 for value in early.values()) and any(value < 0.0 for value in final.values()):
        timing = "evaluator-specific-early"
    elif any(value < 0.0 for value in final.values()):
        timing = "late"
    else:
        timing = "none"
    return {"steps": list(CHECKPOINT_STEPS), "records": records, "semantic_tradeoff_timing": timing, "DEV_macro_recall_delta_6300_minus_6000": early, "DEV_macro_recall_delta_9000_minus_6000": final, "definitions": {"train_update_tail": "last persisted structural-training update in each scientific checkpoint history_tail", "macro_log_rho": "mean frozen 337D log(distance/q95) from accepted endpoint diagnostic", "ssr": "frozen 337D structural support rate", "missing_intermediate_endpoint": "null means no accepted DEV/S1 endpoint artifact existed at that step"}}


def _classify(analysis: Mapping[str, Any], probes: Mapping[str, Any], hard: Mapping[str, Any], trajectory: Mapping[str, Any], optimizer9000: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    output = analysis["output_space"]
    phase = analysis["phase_space"]
    output_conflict = float(output["cos_structural_v_SIC"]) < 0.0
    phase_conflict = float(phase["cos_structural_v_SIC"]) < 0.0
    probe_records = probes["directions"]
    joint_all = all(_stable_descent(probe_records, "d_joint", metric) for metric in ("L_struct", "L_S", "L_I", "L_C"))
    struct_semantic = _stable_descent(probe_records, "d_struct", "L_struct") and all(not _stable_descent(probe_records, "d_struct", metric) for metric in ("L_S", "L_I", "L_C"))
    sic_structural_tradeoff = _stable_descent(probe_records, "d_SIC", "L_struct") is False
    pareto_conflict = not joint_all and not _stable_descent(probe_records, "d_struct", "L_struct") and not _stable_descent(probe_records, "d_SIC", "L_S")
    hard_joint = all(float(item["plus"]["delta_macro_log_rho"]) < 0.0 for item in hard["directions"].get("d_joint", []))
    effective = optimizer9000.get("direction_cosines", {})
    optimizer_misaligned = float(effective.get("effective_descent_vs_g_structural", 0.0)) < 0.0 or float(effective.get("effective_descent_vs_v_SIC", 0.0)) < 0.0
    flags = {
        "output_structural_classifier_conflict": output_conflict,
        "phase_structural_classifier_conflict": phase_conflict,
        "d_joint_improves_structural_and_all_three_classifier_losses": joint_all,
        "d_structural_improves_structure_but_not_all_classifier_losses": struct_semantic,
        "d_SIC_not_stably_improving_structure": sic_structural_tradeoff,
        "hard_DEV_d_joint_improves_macro_log_rho": hard_joint,
        "optimizer_effective_descent_misaligned": optimizer_misaligned,
        "trajectory_semantic_tradeoff_timing": trajectory.get("semantic_tradeoff_timing"),
    }
    if joint_all and not output_conflict and not phase_conflict:
        classification = CLASSIFICATION_A
    elif pareto_conflict:
        classification = CLASSIFICATION_E
    elif output_conflict:
        classification = CLASSIFICATION_B
    elif phase_conflict:
        classification = CLASSIFICATION_C
    elif optimizer_misaligned and (not joint_all or trajectory.get("semantic_tradeoff_timing") != "none"):
        classification = CLASSIFICATION_D
    else:
        classification = CLASSIFICATION_F
    return classification, {"flags": flags, "classification_order": ["A_shared", "E_pareto", "B_output_objective", "C_phase_jacobian", "D_optimizer_state", "F_mixed"]}


def _preflight(consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, authorities: Mapping[str, Any], mean: torch.Tensor, scale: torch.Tensor, analysis9000: Mapping[str, Any], tensors9000: Mapping[str, torch.Tensor], directions: Mapping[str, torch.Tensor], ledger: DataReadLedger) -> dict[str, Any]:
    initial_state = _state_digest(_runtime_state(consumer))
    initial_rng = prior._rng_snapshot()
    batch_indices = authorities["attribution_bank"]["b_indices"]
    batch_images = authorities["images"].index_select(0, batch_indices).to(consumer.device)
    before = _loss_values_no_grad(consumer, smooth, independent, ec, ed, batch_images, authorities["labels"].index_select(0, batch_indices), authorities["training"]["references"], mean, scale, ledger, role=TRAIN_ROLE)
    with torch.no_grad():
        _generator(consumer).add_(PERTURBATION_EPSILONS[0] * directions["d_joint"].to(consumer.device))
    after = _loss_values_no_grad(consumer, smooth, independent, ec, ed, batch_images, authorities["labels"].index_select(0, batch_indices), authorities["training"]["references"], mean, scale, ledger, role=TRAIN_ROLE)
    consumer.load_trainable_state_dict(authorities["checkpoints"][9000])
    _freeze_generator_only(consumer)
    _clear_grads(consumer, smooth, independent, ec, ed)
    _require(_state_digest(_runtime_state(consumer)) == initial_state, "preflight changed runtime state")
    _require(prior._rng_equal(prior._rng_snapshot(), initial_rng), "preflight changed RNG")
    _require(all(parameter.grad is None for model in (smooth, independent, ec, ed) for parameter in model.parameters()), "preflight left evaluator gradients")
    return {"schema": "step9000_structural_classifier_semantic_tradeoff_preflight.v1", "status": "READY", "optimizer_created": False, "parameter_updates": 0, "representative_train_rows": int(batch_indices.numel()), "gradient_analysis": {"step": 9000, "A_rows": int(analysis9000["rows"]), "structural_gradient_finite": True, "SIC_gradient_finite": True, "joint_direction_defined": True}, "temporary_probe": {"epsilon": PERTURBATION_EPSILONS[0], "direction": "d_joint", "before": before["losses"], "after": after["losses"], "runtime_restored": True}, "negative_control": "symmetric opposite-sign probe retained in the execution artifact", "trainable_parameter_names": ["generator.phase"], "frozen_evaluator_parameters": True, "E_D_in_backward": False, "DEV_gradients": False, "protected_test_sealed": {"protected": 0, "test": 0, "sealed": 0}}


def _make_identity(source_manifest: Mapping[str, Any], bank: Mapping[str, torch.Tensor], dev_indices: torch.Tensor, attempt_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"no_update": True, "checkpoint_steps": list(CHECKPOINT_STEPS), "key_steps": list(KEY_STEPS), "A_B_bank_sha256": {name: tensor_sha256(value) for name, value in bank.items()}, "DEV_subset_sha256": tensor_sha256(dev_indices), "epsilons": list(PERTURBATION_EPSILONS)},
        model_identity="LIGHTGEN_2D_STEP9000_STRUCTURAL_CLASSIFIER_SEMANTIC_TRADEOFF_ATTRIBUTION_V1",
        optimizer_identity={"created": False, "parameter_updates": 0, "stored_Adam_state": "read-only checkpoint diagnostic"},
        sampler_identity={"kind": "accepted deterministic TRAIN-only A/B bank", "seed": prior_attribute.BANK_SEED},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=attempt_id,
    )


def run_analysis(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"output already exists: {target}")

    authorities = _load_authorities(root)
    cuda_runtime = prior.robust._prepare_cuda_runtime()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "step9000 attribution requires CUDA")
    device = torch.device("cuda:0")
    mean, scale = _load_surrogate_normalization(root, authorities)
    consumer, smooth, independent, ec, ed = _load_models(root, authorities, device)
    checkpoint9000 = authorities["checkpoints"][9000]
    consumer.load_trainable_state_dict(checkpoint9000)
    _freeze_generator_only(consumer)
    initial_step9000_runtime = _runtime_state(consumer)
    bank = authorities["attribution_bank"]
    images = authorities["images"]
    labels = authorities["labels"]
    references = authorities["training"]["references"]
    ledger = DataReadLedger()
    analysis_records: dict[int, dict[str, Any]] = {}
    analysis_tensors: dict[int, dict[str, torch.Tensor]] = {}
    for step in KEY_STEPS:
        record, tensors = _analyze_step(consumer, smooth, independent, ec, images, labels, bank["a_indices"], references, authorities["checkpoints"][step], mean, scale, step, ledger, classwise=step == 9000)
        record["_g_structural"] = tensors["g_structural"]
        g_sic = tensors["g_E_S"] / _norm(tensors["g_E_S"], f"step {step} g_E_S") + tensors["g_E_I"] / _norm(tensors["g_E_I"], f"step {step} g_E_I") + tensors["g_E_C"] / _norm(tensors["g_E_C"], f"step {step} g_E_C")
        record["_v_SIC"] = g_sic
        analysis_records[step] = record
        analysis_tensors[step] = tensors
    directions, direction_details = _directions(analysis_records[9000], analysis_tensors[9000])
    preflight = _preflight(consumer, smooth, independent, ec, ed, authorities, mean, scale, analysis_records[9000], analysis_tensors[9000], directions, ledger)
    probes = _run_train_probes(consumer, smooth, independent, ec, ed, images.index_select(0, bank["b_indices"]), labels.index_select(0, bank["b_indices"]), references, mean, scale, checkpoint9000, directions, ledger)
    dev_subset = _select_dev_subset(labels, authorities["development_indices"])
    hard_probes = _run_hard_probes(consumer, ed, authorities, dev_subset, mean, scale, checkpoint9000, directions, authorities["structural_reference"], device, ledger)
    trajectory = _build_trajectory(authorities, analysis_records)

    train_a_eval = _loss_values_no_grad(consumer, smooth, independent, ec, ed, images.index_select(0, bank["a_indices"]), labels.index_select(0, bank["a_indices"]), references, mean, scale, ledger, role=TRAIN_ROLE)
    train_b_eval = probes["base"]
    s1_summary = {str(step): _endpoint_summary(authorities["endpoint_records"][step]["s1"], kind="S1") for step in KEY_STEPS}
    dev_summary = {str(step): _endpoint_summary(authorities["endpoint_records"][step]["dev"], kind="DEV") for step in KEY_STEPS}
    optimizer9000 = trajectory["records"]["9000"]["optimizer"]
    classification, classification_evidence = _classify(analysis_records[9000], probes, hard_probes, trajectory, optimizer9000)

    initial_evaluator_states = {name: prior_attribute._state_digest(model.state_dict()) for name, model in (("E_S", smooth), ("E_I", independent), ("E_C", ec), ("E_D", ed))}
    consumer.load_trainable_state_dict(checkpoint9000)
    _freeze_generator_only(consumer)
    _clear_grads(consumer, smooth, independent, ec, ed)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_step9000_runtime), "final step9000 runtime restoration failed")
    _require(initial_evaluator_states == {name: prior_attribute._state_digest(model.state_dict()) for name, model in (("E_S", smooth), ("E_I", independent), ("E_C", ec), ("E_D", ed))}, "frozen evaluator state changed")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "runtime gradients remained after attribution")
    _require(all(ledger.counts.get(role, 0) == 0 for role in ("protected", "test", "sealed")), "protected/test/sealed data was read")

    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(Path(__file__).resolve(), Path(prior.__file__).resolve(), Path(structural_alignment.__file__).resolve(), Path(prior_attribute.__file__).resolve(), root / STRUCTURAL_REPORT, root / SURROGATE_NORMALIZATION, root / prior_attribute.REPLAY_BANK_PAYLOAD, root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py", root / "src/ldm_project/v2/student/evaluator.py"),
        config_identity={"authorization_mode": "IMPLEMENT_PREPARED", "authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "no_update": True, "checkpoint_steps": list(CHECKPOINT_STEPS), "key_steps": list(KEY_STEPS), "epsilons": list(PERTURBATION_EPSILONS), "bank_indices_sha256": {name: tensor_sha256(value) for name, value in bank.items()}, "dev_subset_sha256": tensor_sha256(dev_subset), "input_authority_hashes": authorities["authority_hashes"]},
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    attempt_id = f"{RUN_ID}:{target.name}"
    identity = _make_identity(source_manifest, bank, dev_subset, attempt_id)
    control = ResearchRunControl(identity, AUTHORIZATION_ID)
    cuda_free, cuda_total = torch.cuda.mem_get_info(device)
    receipt = control.make_ready_receipt(
        correctness={
            "passed": bool(
                preflight.get("status") == "READY"
                and preflight.get("optimizer_created") is False
                and preflight.get("parameter_updates") == 0
                and preflight.get("temporary_probe", {}).get("runtime_restored") is True
                and _state_digest(_runtime_state(consumer)) == _state_digest(initial_step9000_runtime)
                and preflight.get("gradient_analysis", {}).get("structural_gradient_finite") is True
                and preflight.get("gradient_analysis", {}).get("SIC_gradient_finite") is True
                and initial_evaluator_states == {name: prior_attribute._state_digest(model.state_dict()) for name, model in (("E_S", smooth), ("E_I", independent), ("E_C", ec), ("E_D", ed))}
                and all(ledger.counts.get(role, 0) == 0 for role in ("protected", "test", "sealed"))
            ),
            "finite_gradient_analysis": bool(preflight.get("gradient_analysis", {}).get("structural_gradient_finite") and preflight.get("gradient_analysis", {}).get("SIC_gradient_finite")),
            "frozen_evaluators": bool(initial_evaluator_states == {name: prior_attribute._state_digest(model.state_dict()) for name, model in (("E_S", smooth), ("E_I", independent), ("E_C", ec), ("E_D", ed))}),
            "protected_test_sealed_zero": bool(all(ledger.counts.get(role, 0) == 0 for role in ("protected", "test", "sealed"))),
        },
        resource={"passed": int(cuda_total) > 0 and 0 <= int(cuda_free) <= int(cuda_total), "free_bytes": int(cuda_free), "total_bytes": int(cuda_total), "updates": 0},
        implementation={"passed": preflight.get("optimizer_created") is False and preflight.get("parameter_updates") == 0, "selected_engine": "read_only_cuda_attribution"},
        selected_engine="read_only_cuda_attribution",
        created_at=time.time(),
    )
    control.mark_preflight_ready(receipt)
    control.start_execution()
    effective_identity = control.open_scientific_attempt(target, attempt_id, receipt, resume=True)
    _write_json(target / "preflight_receipt.json", receipt.to_dict(), kind="step9000 attribution preflight receipt")

    _write_torch(target / "diagnostic_bank.pt", {"schema": "step9000_structural_classifier_semantic_tradeoff_bank.v1", "task_id": TASK_ID, "data_role": TRAIN_ROLE, "A_indices": bank["a_indices"], "A_labels": bank["a_labels"], "B_indices": bank["b_indices"], "B_labels": bank["b_labels"], "DEV_indices": dev_subset, "DEV_labels": labels.index_select(0, dev_subset), "A_B_source": "accepted cross-evaluator semantic-divergence bank", "DEV_rule": "first 10 canonical DEV rows per class"}, kind="frozen TRAIN A/B and DEV diagnostic bank")
    _write_json(target / "source_manifest.json", source_manifest, kind="step9000 attribution source manifest")
    _write_json(target / "preflight.json", preflight, kind="step9000 attribution preflight")
    _write_json(target / "execution_manifest.json", {"schema": "step9000_structural_classifier_semantic_tradeoff_execution_manifest.v1", "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorization": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN, "one_authorization": True}, "input_authorities": authorities["authority_hashes"], "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "local_state": authorities["local_state"], "cuda_runtime": cuda_runtime, "scope": {"optimizer_created": False, "parameter_updates": 0, "train_rows_unique": 40, "dev_rows_unique": 100, "protected": False, "test": False, "sealed": False, "sampler_changed": False, "architecture_changed": False, "DEV_gradients": False}}, kind="step9000 attribution execution manifest")

    gradient_payload = {"schema": GRADIENT_SCHEMA, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "checkpoint_hashes": {str(step): CHECKPOINT_SHA256[step] for step in KEY_STEPS}, "A_B_indices": {name: bank[name] for name in ("a_indices", "a_labels", "b_indices", "b_labels")}, "key_steps": {str(step): {name: value.detach().cpu() for name, value in analysis_tensors[step].items()} for step in KEY_STEPS}, "direction_definitions": direction_details}
    _write_torch(target / "gradient_attribution.pt", gradient_payload, kind="step9000 structural/classifier gradient attribution tensors")
    _write_json(target / "directional_probes.json", probes, kind="held-out TRAIN B symmetric structural/classifier probes")
    _write_json(target / "hard_structural_probes.json", hard_probes, kind="fixed DEV no-gradient hard structural probes")
    _write_json(target / "trajectory.json", trajectory, kind="accepted step6000-to-step9000 trajectory attribution")

    public_analysis = {str(step): {key: value for key, value in record.items() if not key.startswith("_")} for step, record in analysis_records.items()}
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "classification": classification,
        "authorization_consumed": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN, "scope": "one-shot step9000 no-update structural/classifier semantic tradeoff attribution"},
        "local_state": authorities["local_state"],
        "input_authorities": {"accepted_structural_alignment_report": _artifact_ref(root, root / STRUCTURAL_REPORT, role="accepted evaluator-agnostic structural alignment result"), "accepted_structural_alignment_run": _artifact_ref(root, root / STRUCTURAL_RUN / "execution_manifest.json", role="accepted step6000-to-step9000 structural execution manifest"), "accepted_checkpoints": {str(step): _artifact_ref(root, root / STRUCTURAL_RUN / f"scientific_step_{step:06d}.pt", role=f"accepted Generator step {step}") for step in CHECKPOINT_STEPS}, "accepted_DEV_endpoints": {str(step): _artifact_ref(root, root / STRUCTURAL_RUN / f"dev_diagnostics_step_{step:06d}.json", role=f"accepted DEV step {step} endpoint") for step in KEY_STEPS}, "accepted_S1_endpoints": {str(step): _artifact_ref(root, root / STRUCTURAL_RUN / f"s1_transfer_step_{step:06d}.json", role=f"accepted S1 step {step} endpoint") for step in KEY_STEPS}, "accepted_structural_reference": _artifact_ref(root, root / structural_alignment.STRUCTURAL_REFERENCE, role="accepted frozen 337D structural reference"), "accepted_surrogate_normalization": _artifact_ref(root, root / SURROGATE_NORMALIZATION, role="accepted TRAIN-only 336D surrogate normalization"), "accepted_A_B_bank": _artifact_ref(root, root / prior_attribute.REPLAY_BANK_PAYLOAD, role="accepted TRAIN-only A/B bank"), "authority_hashes": authorities["authority_hashes"]},
        "supported_bottleneck": "STRUCTURAL_RECOVERY_WITH_CLASSIFIER_SEMANTIC_TRADEOFF",
        "objective": "Attribute whether step9000 structural morphology and E_S/E_I/E_C semantic objectives are jointly locally realizable in the frozen two-layer Generator.",
        "diagnostic_roles": {"TRAIN_A": {"role": "direction construction and held-out class summaries", "rows": int(bank["a_indices"].numel()), "class_balanced": True}, "TRAIN_B": {"role": "disjoint symmetric temporary probes", "rows": int(bank["b_indices"].numel()), "class_balanced": True}, "DEV_subset": {"role": "no-gradient frozen 337D structural confirmation", "rows": int(dev_subset.numel()), "class_balanced": True, "selection_or_gradient_use": False}, "S1": {"role": "accepted endpoint descriptive context", "rows": 2000, "selection_or_gradient_use": False}, "protected_test_sealed": {"protected": 0, "test": 0, "sealed": 0}},
        "primary_representation": {"object": "real Encoder-produced optical path with generator.phase attribution", "generator_phase_shape": [2, 1, 100, 100], "complex_OLS_boundary": "accepted complex OLS/Encoder input path unchanged", "structural_output_space": "raw 200x200 CMOS tensor before evaluator preprocessing"},
        "frozen_objectives": {"L_struct": "accepted normalized 336D differentiable structural surrogate loss", "L_S": "accepted normalized E_S feature loss", "L_I": "accepted normalized E_I feature loss", "L_C": "accepted normalized E_C feature loss", "target_features_stop_gradient": True, "E_D_role": "descriptive confirmation only; absent from backward"},
        "preflight": preflight,
        "stage1_trajectory_tradeoff_decomposition": trajectory,
        "stage2_output_space_attribution": {"scope": "step9000 TRAIN A", "records": public_analysis["9000"]["output_space"], "definition": "h_k = dL_k/d(raw CMOS), k in {structural,S,I,C}; v_hSIC is equal-weight sum of L2-normalized classifier output gradients"},
        "stage3_generator_phase_attribution": {"scope": "step9000 TRAIN A", "records": public_analysis["9000"]["phase_space"], "definition": "g_k = dL_k/d(generator.phase); v_SIC is equal-weight sum of L2-normalized g_S,g_I,g_C"},
        "stage4_joint_direction": {"scope": "step9000 TRAIN A", "direction_details": direction_details, "directions": {name: {"rms": _rms(value, name), "shape": list(value.shape)} for name, value in directions.items()}, "optimizer_update": False},
        "stage5_held_out_TRAIN_B_symmetric_probes": {"artifact": _artifact_ref(root, target / "directional_probes.json", role="held-out TRAIN B directional probes"), "directions": {"d_struct": "-u_struct/RMS(u_struct)", "d_SIC": "-u_SIC/RMS(u_SIC)", "d_joint": "-v_joint/RMS(v_joint)"}, "epsilons": list(PERTURBATION_EPSILONS), "temporary_parameter_restore": True, "optimizer_created": False},
        "stage6_hard_337D_DEV_confirmation": {"artifact": _artifact_ref(root, target / "hard_structural_probes.json", role="fixed DEV hard structural probes"), "DEV_subset_rule": "first 10 canonical DEV rows per class", "no_DEV_gradients": True, "no_direction_redefinition": True},
        "stage7_E_D_descriptive": {"role": "descriptive only", "TRAIN_A": train_a_eval["semantic"]["E_D"], "TRAIN_B": train_b_eval["semantic"]["E_D"], "DEV_subset": hard_probes["base"]["E_D"], "accepted_endpoint_DEV": {str(step): dev_summary[str(step)]["macro_recall"]["E_D"] for step in KEY_STEPS}, "accepted_endpoint_S1": {str(step): s1_summary[str(step)]["macro_recall"]["E_D"] for step in KEY_STEPS}, "in_backward": False},
        "stage8_optimizer_state_analysis": {"step6000": trajectory["records"]["6000"]["optimizer"], "step6300": trajectory["records"]["6300"]["optimizer"], "step9000": trajectory["records"]["9000"]["optimizer"], "optimizer_created": False, "state_read_only": True},
        "stage9_S1_descriptive_context": {"accepted_endpoints": s1_summary},
        "accepted_endpoint_context": {"real_DEV": dev_summary, "S1": s1_summary},
        "classification_evidence": classification_evidence,
        "counterfactual_integrity": {"optimizer_created": False, "parameter_updates": 0, "persistent_probe_changes": False, "checkpoint_restored_after_every_probe": True, "runtime_restored_to_step9000": True, "encoder_frozen": True, "architecture_changed": False, "sampler_changed": False, "evaluator_search": False, "E_S_frozen": True, "E_I_frozen": True, "E_C_frozen": True, "E_D_frozen": True, "E_D_in_backward": False, "DEV_gradients": False, "protected_test_sealed": False},
        "claim_boundary": {"local_frozen_step9000_attribution_only": True, "accepted_trajectory_context_only": True, "TRAIN_A_B_only_for_gradient_and_loss_probes": True, "fixed_DEV_subset_descriptive_only": True, "global_architecture_inadequacy": False, "sampler_selection": False, "generator_training": False, "downstream_generation_acceptance": False, "physical_hardware_validity": False, "promotion": False},
        "files_changed": {"repository_source": [_relative(root, Path(__file__).resolve())], "scientific_artifacts": sorted(path.name for path in target.iterdir())},
        "data_reads": ledger.to_dict(),
        "forbidden_data_reads": {"protected": ledger.counts.get("protected", 0), "test": ledger.counts.get("test", 0), "sealed": ledger.counts.get("sealed", 0)},
        "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "checkpoint_schedule_complete": list(CHECKPOINT_STEPS), "key_steps_complete": list(KEY_STEPS), "symmetric_epsilons": list(PERTURBATION_EPSILONS), "train_A_B_class_balanced": True, "no_optimizer": True, "no_persistent_update": True, "DEV_gradients": False, "protected_test_sealed_zero": True},
        "unresolved": ["This is local attribution for the accepted two-layer Generator, frozen evaluators, accepted structural surrogate/reference, and fixed A/B/DEV diagnostic sets; it does not establish a global Pareto frontier or hardware readiness."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    control.complete()
    _write_json(target / "report.json", report, kind="step9000 structural/classifier tradeoff attribution report")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and run the no-update attribution")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "no_update": True, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_analysis(output_dir=args.output_dir)
    except (TradeoffAttributionError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "output_dir": report["output_dir"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
