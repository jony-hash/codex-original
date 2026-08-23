"""Small research-lifecycle artifact and manifest boundaries.

This module is deliberately scoped to the electronic O0 research pilots.  It
does not alter formal runtime evidence or resolve implicit/latest outputs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import traceback
from uuid import uuid4
from pathlib import Path
from typing import Any, Mapping

from ...evidence import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)


MANIFEST_SCHEMA = "research_run_manifest.v1"
POSTERIOR_SCHEMA = "encoded_posterior.v1"
POSTERIOR_CONTRACT = "encoded_posterior.v1/frozen-wae-mnist-split"

_PROGRAM_ACTIONS = {
    "electronic-o0-latent-transport": {
        "validate",
        "encode-posterior",
        "attribute",
        "preflight",
        "fit-flow",
        "measure-flow",
        "closeout",
        "canary",
        "screen",
    },
    "electronic-o0-mixture-transport": {
        "validate",
        "encode-posterior",
        "preflight",
        "fit-mixture",
        "calibrate-temperature",
        "screen-measure",
        "closeout",
        "canary",
        "screen",
    },
    "electronic-o0-prior-native-swae": {
        "validate",
        "audit",
        "preflight",
        "screen",
        "confirm",
        "locked-final",
        "report",
        "closeout",
    },
}

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("research run manifest must be a mapping")
    return payload


def _one(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def load_manifest(path: str | Path, *, program: str, action: str) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = _yaml(manifest_path)
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported research manifest schema: {payload.get('schema')!r}"
        )
    if payload.get("program") != program:
        raise ValueError("research manifest program does not match the CLI program")
    if program not in _PROGRAM_ACTIONS:
        raise ValueError(f"unsupported research manifest program: {program}")
    declared_action = payload.get("action")
    if declared_action != action:
        raise ValueError("research manifest action does not match the CLI action")
    if action not in _PROGRAM_ACTIONS[program]:
        raise ValueError(f"action {action!r} is not part of the manifest lifecycle")
    required = {
        "schema",
        "program",
        "action",
        "config",
        "dataset_root",
        "device",
        "deterministic",
        "inputs",
        "artifacts",
        "seeds",
        "output_root",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"research manifest missing required fields: {', '.join(missing)}"
        )
    if not isinstance(payload["config"], str) or not isinstance(
        payload["dataset_root"], str
    ):
        raise ValueError("research manifest config and dataset_root must be paths")
    if payload.get("device") != "cuda":
        raise ValueError("research manifest device must be cuda")
    deterministic = payload["deterministic"]
    if not isinstance(deterministic, dict):
        raise ValueError("research manifest deterministic section must be a mapping")
    if deterministic.get("cublas_workspace_config") != ":4096:8":
        raise ValueError(
            "CUDA research manifest requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    for section in ("inputs", "artifacts", "seeds"):
        if not isinstance(payload[section], Mapping):
            raise ValueError(f"research manifest {section} section must be a mapping")
    inputs = payload["inputs"]
    for section in ("checkpoints", "evaluator_evidence", "upstream_evidence"):
        if not isinstance(inputs.get(section), Mapping):
            raise ValueError(f"research manifest inputs.{section} must be a mapping")
    artifacts = payload["artifacts"]
    for section in ("posterior", "fit", "calibration", "measurement"):
        if section not in artifacts:
            raise ValueError(f"research manifest artifacts.{section} is required")
        if not isinstance(artifacts[section], Mapping):
            raise ValueError(f"research manifest artifacts.{section} must be a mapping")
    if not isinstance(payload.get("output_root"), str) or not payload["output_root"]:
        raise ValueError("research manifest requires output_root")
    output_root = Path(payload["output_root"])
    if (
        output_root.name.lower() in {"latest", "final_pt", "runs"}
        or "latest" in output_root.parts
    ):
        raise ValueError(
            "research manifest output_root must be an explicit versioned path"
        )
    config = Path(payload["config"])
    dataset_root = Path(payload["dataset_root"])
    if not config.is_file():
        raise FileNotFoundError(f"research manifest config is missing: {config}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"research manifest dataset_root is missing: {dataset_root}"
        )
    _validate_input_section(config, "config", None)
    for section_name in ("checkpoints", "evaluator_evidence", "upstream_evidence"):
        for role, value in inputs[section_name].items():
            _validate_input_section(value, role, section_name)
    artifact_section = payload["artifacts"]
    peer_values = artifact_section.get("peer_confirm_evidence", payload.get("peer_confirm_evidence"))
    if peer_values is not None:
        if not isinstance(peer_values, list) or not peer_values:
            raise ValueError("research manifest peer_confirm_evidence must be a non-empty list")
        for index, value in enumerate(peer_values):
            _validate_input_section(value, f"peer_confirm_evidence[{index}]", "artifacts")
    payload["_manifest_path"] = str(manifest_path.resolve())
    payload["_invocation_mode"] = "manifest"
    return payload


def _validate_input_section(value: Any, role: str, section: str | None) -> None:
    candidate = _path_from(value)
    if candidate is None:
        raise ValueError(f"research manifest input {role} must declare a path")
    target = Path(candidate)
    if not target.is_file():
        raise FileNotFoundError(f"research manifest input {role} is missing: {target}")
    if isinstance(value, Mapping):
        declared_role = value.get("role")
        if declared_role != role:
            raise ValueError(
                f"research manifest input {role} role does not match its declared role"
            )
        declared_hash = value.get("sha256")
        if (
            declared_hash is None
            or not isinstance(declared_hash, str)
            or not _HASH_RE.fullmatch(declared_hash)
        ):
            raise ValueError(f"research manifest input {role} requires a sha256")
        actual = file_sha256(target)
        if actual != declared_hash:
            raise ValueError(f"research manifest input {role} hash mismatch")


def _sections(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    inputs = manifest.get("inputs", {})
    if not isinstance(inputs, Mapping):
        inputs = {}
    return tuple(
        section
        for section in (
            inputs.get("checkpoints", {}),
            inputs.get("evaluator_evidence", {}),
            inputs.get("upstream_evidence", {}),
            manifest.get("artifacts", {}),
        )
        if isinstance(section, Mapping)
    )


def _path_from(value: Any) -> str | None:
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, Mapping):
        candidate = _one(value, "path", "file", "evidence", "artifact")
        return str(candidate) if candidate is not None else None
    return None


def manifest_value(manifest: Mapping[str, Any], *names: str) -> str | None:
    for section in _sections(manifest):
        for name in names:
            value = _path_from(section.get(name))
            if value is not None:
                return value
    for name in names:
        value = manifest.get(name)
        if isinstance(value, (str, int, float)):
            return str(value)
    return None


def manifest_args(
    namespace: Any, *, program: str, action: str, field_names: tuple[str, ...]
) -> Any:
    """Expand a manifest into an argparse namespace and reject mixed inputs."""
    path = getattr(namespace, "run_manifest", None)
    if not path:
        return namespace
    manifest = load_manifest(path, program=program, action=action)
    for field in field_names:
        if getattr(namespace, field, None) not in (None, False, []):
            raise ValueError(
                f"--run-manifest cannot be mixed with --{field.replace('_', '-')}"
            )
    namespace._run_manifest = manifest
    namespace._manifest_hashes = _manifest_hashes(manifest)
    namespace.config = str(manifest["config"])
    namespace.dataset_root = str(manifest["dataset_root"])
    namespace.device = str(manifest.get("device", "cuda"))
    output = manifest_value(manifest, "output", "evidence", "output_evidence")
    if output is None:
        output = str(Path(manifest["output_root"]) / f"{action}.json")
    namespace.output = output
    aliases = {
        "wae_training_checkpoint": (
            "wae_training",
            "wae_training_checkpoint",
            "training_checkpoint",
        ),
        "wae_decoder_checkpoint": (
            "wae_decoder",
            "wae_decoder_checkpoint",
            "decoder_checkpoint",
        ),
        "wae_confirm_evidence": ("wae_confirm", "wae_confirm_evidence"),
        "latent_transport_attribution": (
            "latent_attribution",
            "latent_transport_attribution",
        ),
        "latent_transport_confirm_evidence": (
            "latent_confirm",
            "latent_transport_confirm_evidence",
        ),
        "mixture_closeout": ("mixture_closeout", "mixture_transport_closeout"),
        "primary_evaluator_checkpoint": (
            "primary_evaluator",
            "primary_evaluator_checkpoint",
        ),
        "secondary_evaluator_checkpoint": (
            "secondary_evaluator",
            "secondary_evaluator_checkpoint",
        ),
        "primary_qualification_evidence": (
            "primary_qualification",
            "primary_qualification_evidence",
        ),
        "secondary_qualification_evidence": (
            "secondary_qualification",
            "secondary_qualification_evidence",
        ),
        "validation_evidence": ("validation", "validation_evidence"),
        "attribution_evidence": ("attribution", "attribution_evidence"),
        "preflight_evidence": ("preflight", "preflight_evidence"),
        "fit_evidence": ("fit", "fit_evidence"),
        "calibration_evidence": ("calibration", "calibration_evidence"),
        "screen_evidence": ("screen", "screen_evidence"),
        "measurement_evidence": ("measurement", "measurement_evidence"),
        "locked_evidence": ("locked", "locked_evidence"),
        "posterior_artifact": ("posterior", "posterior_artifact"),
        "audit_evidence": ("audit", "audit_evidence"),
        "confirm_evidence": ("confirm", "confirm_evidence"),
        "checkpoint": ("checkpoint", "decoder_checkpoint_artifact"),
        "resume_checkpoint": ("resume_checkpoint", "resume"),
        "anchor_confirm_evidence": ("anchor_confirm", "anchor_confirm_evidence"),
    }
    for field, names in aliases.items():
        value = manifest_value(manifest, *names)
        if value is not None:
            setattr(namespace, field, value)
    peer_values = manifest_list_value(manifest, "peer_confirm_evidence", "peer_confirms")
    if peer_values is not None:
        namespace.peer_confirm_evidence = peer_values
    posterior_ref = (
        manifest.get("artifacts", {}).get("posterior")
        if isinstance(manifest.get("artifacts", {}), Mapping)
        else None
    )
    if isinstance(posterior_ref, Mapping) and posterior_ref.get("sha256"):
        namespace.posterior_artifact_sha256 = str(posterior_ref["sha256"])
    seeds = manifest.get("seeds", {})
    if isinstance(seeds, Mapping):
        for key in ("flow", "mixture", "swae", "run", "seed"):
            if key in seeds:
                namespace.seed = int(
                    seeds[key][0] if isinstance(seeds[key], list) else seeds[key]
                )
                break
    namespace.run_manifest = str(path)
    namespace.invocation_mode = "manifest"
    return namespace


def _manifest_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for section in _sections(manifest)[:3]:
        for value in section.values():
            if (
                isinstance(value, Mapping)
                and isinstance(value.get("path"), str)
                and isinstance(value.get("sha256"), str)
            ):
                hashes[str(Path(value["path"]).resolve())] = value["sha256"]
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        for value in artifacts.values():
            if (
                isinstance(value, Mapping)
                and isinstance(value.get("path"), str)
                and isinstance(value.get("sha256"), str)
            ):
                hashes[str(Path(value["path"]).resolve())] = value["sha256"]
            elif isinstance(value, list):
                for item in value:
                    if (
                        isinstance(item, Mapping)
                        and isinstance(item.get("path"), str)
                        and isinstance(item.get("sha256"), str)
                    ):
                        hashes[str(Path(item["path"]).resolve())] = item["sha256"]
    return hashes


def manifest_list_value(manifest: Mapping[str, Any], *names: str) -> list[str] | None:
    """Resolve a validated list of artifact paths from a manifest boundary."""
    sections = _sections(manifest) + (manifest,)
    for section in sections:
        for name in names:
            values = section.get(name) if isinstance(section, Mapping) else None
            if values is None:
                continue
            if not isinstance(values, list):
                raise ValueError(f"manifest field {name} must be a list")
            paths = []
            for value in values:
                candidate = _path_from(value)
                if candidate is None:
                    raise ValueError(f"manifest field {name} contains a value without a path")
                paths.append(candidate)
            return paths
    return None


def validate_manifest_namespace(namespace: Any, *, action: str) -> None:
    """Complete read-only validation before config/model/CUDA/output work."""
    if not getattr(namespace, "_run_manifest", None):
        return
    manifest = namespace._run_manifest
    for name in (
        "wae_training_checkpoint",
        "wae_decoder_checkpoint",
        "wae_confirm_evidence",
        "latent_transport_confirm_evidence",
        "mixture_closeout",
        "primary_evaluator_checkpoint",
        "secondary_evaluator_checkpoint",
        "primary_qualification_evidence",
        "secondary_qualification_evidence",
        "validation_evidence",
        "attribution_evidence",
        "preflight_evidence",
        "fit_evidence",
        "calibration_evidence",
        "screen_evidence",
        "measurement_evidence",
        "locked_evidence",
        "flow_checkpoint",
        "mixture_checkpoint",
        "posterior_artifact",
        "audit_evidence",
        "preflight_evidence",
        "screen_evidence",
        "confirm_evidence",
        "measurement_evidence",
        "checkpoint",
        "resume_checkpoint", "anchor_confirm_evidence", "peer_confirm_evidence",
    ):
        value = getattr(namespace, name, None)
        if not value:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            target = Path(item)
            declared = namespace._manifest_hashes.get(str(target.resolve()))
            if target.exists():
                if not target.is_file():
                    raise ValueError(f"manifest input {name} is not a file: {target}")
                if declared is not None and file_sha256(target) != declared:
                    raise ValueError(f"manifest input {name} hash mismatch")
            elif name != "posterior_artifact" or action != "encode-posterior":
                raise FileNotFoundError(f"manifest input {name} is missing: {target}")
    output = Path(namespace.output)
    root = Path(manifest["output_root"]).resolve()
    if output.exists() and not output.is_file():
        raise ValueError(f"manifest output is not a file: {output}")
    if root not in output.resolve().parents and output.resolve() != root:
        raise ValueError("manifest output must remain under output_root")
    if action != "canary":
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if status:
            raise RuntimeError("manifest evidence requires a clean worktree")


def validate_evidence_inputs(namespace: Any, *, program: str, action: str) -> None:
    """Validate consumer evidence before deterministic/CUDA setup."""
    if not getattr(namespace, "_run_manifest", None):
        return
    prefix = (
        "electronic_o0_latent_transport"
        if program.endswith("latent-transport")
        else "electronic_o0_mixture_transport"
    )
    expected: dict[str, tuple[str, ...]] = {
        "attribute": (f"{prefix}_validation.v1",),
        "preflight": (f"{prefix}_attribution.v1", f"{prefix}_validation.v1"),
        "fit-flow": (f"{prefix}_preflight.v1",),
        "fit-mixture": (f"{prefix}_preflight.v1",),
        "measure-flow": (f"{prefix}_fit-flow.v1",),
        "screen": (
            f"{prefix}_fit-flow.v1"
            if program.endswith("latent-transport")
            else f"{prefix}_calibrate-temperature.v1",
        ),
        "calibrate-temperature": (f"{prefix}_fit-mixture.v1",),
        "screen-measure": (f"{prefix}_calibrate-temperature.v1",),
        "closeout": (
            f"{prefix}_measure-flow.v1",
            f"{prefix}_screen-measure.v1",
            f"{prefix}_screen.v1",
        ),
    }
    fields = {
        "attribute": ("validation_evidence",),
        "preflight": ("attribution_evidence", "validation_evidence"),
        "fit-flow": ("preflight_evidence",),
        "fit-mixture": ("preflight_evidence",),
        "measure-flow": ("fit_evidence",),
        "screen": ("fit_evidence",)
        if program.endswith("latent-transport")
        else ("calibration_evidence",),
        "calibrate-temperature": ("fit_evidence",),
        "screen-measure": ("calibration_evidence",),
        "closeout": ("measurement_evidence",),
    }.get(action, ())
    producer_contracts: dict[str, str] = {
        "attribute": f"{prefix}-validation.v1",
        "preflight": "latent-attribute.v1"
        if program.endswith("latent-transport")
        else "mixture-validate.v1",
        "fit-flow": "latent-preflight.v1",
        "fit-mixture": "mixture-preflight.v1",
        "measure-flow": "latent-fit-flow.v1",
        "screen": "latent-fit-flow.v1"
        if program.endswith("latent-transport")
        else "mixture-calibration.v1",
        "calibrate-temperature": "mixture-fit.v1",
        "screen-measure": "mixture-calibration.v1",
        "closeout": "latent-measurement.v1"
        if program.endswith("latent-transport")
        else "mixture-measurement.v1",
    }
    for field in fields:
        path = getattr(namespace, field, None)
        if not path:
            continue
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{field} is not readable evidence") from exc
        if not isinstance(payload, dict) or payload.get("status") != "valid":
            raise ValueError(f"{field} must be valid evidence")
        allowed = expected.get(action, ())
        if allowed and payload.get("schema") not in allowed:
            raise ValueError(f"{field} schema is not permitted for {action}")
        if (
            payload.get("invocation_mode") != "manifest"
            or payload.get("legacy_diagnostic_only") is not False
        ):
            raise ValueError(
                f"{field} is legacy or has no manifest invocation contract"
            )
        if action not in payload.get("permitted_consumers", ()):
            raise ValueError(f"{field} does not permit consumer action {action}")
        if (
            not isinstance(payload.get("consumer_contract"), str)
            or not payload["consumer_contract"]
        ):
            raise ValueError(f"{field} is missing its consumer contract")
        expected_contract = producer_contracts.get(action)
        if (
            expected_contract
            and payload.get("producer_contract_revision") != expected_contract
        ):
            raise ValueError(
                f"{field} producer contract is not compatible with {action}"
            )
        for artifact_key, hash_key in (
            ("flow_checkpoint", "flow_checkpoint_sha256"),
            ("training_checkpoint", "training_checkpoint_sha256"),
            ("mixture_checkpoint", "mixture_checkpoint_sha256"),
            ("calibration_artifact", "calibration_artifact_sha256"),
        ):
            artifact_path = payload.get(artifact_key)
            if artifact_path is not None:
                if hash_key not in payload:
                    raise ValueError(f"{field} is missing {hash_key}")
                require_artifact_hash(
                    artifact_path, payload[hash_key], role=f"{field}.{artifact_key}"
                )
        expected_posterior = getattr(namespace, "posterior_artifact", None)
        declared_inputs = payload.get("input_artifact_hashes", {})
        if (
            expected_posterior
            and isinstance(declared_inputs, Mapping)
            and declared_inputs.get("posterior") is not None
        ):
            if declared_inputs["posterior"] != file_sha256(expected_posterior):
                raise ValueError(f"{field} posterior input hash is stale")


def prepare_args(
    namespace: Any,
    *,
    program: str,
    action: str,
    required: tuple[str, ...],
    defaults: Mapping[str, Any],
    mixed_fields: tuple[str, ...],
) -> Any:
    """Resolve a manifest or validate the legacy compatibility invocation."""
    if getattr(namespace, "run_manifest", None):
        namespace = manifest_args(
            namespace, program=program, action=action, field_names=mixed_fields
        )
    else:
        for name, value in defaults.items():
            if getattr(namespace, name, None) is None:
                setattr(namespace, name, value)
        namespace.invocation_mode = "legacy"
        namespace.legacy_diagnostic_only = True
    missing = [name for name in required if not getattr(namespace, name, None)]
    if missing:
        raise ValueError(
            f"{program} {action} requires: {', '.join('--' + name.replace('_', '-') for name in missing)}"
        )
    validate_manifest_namespace(namespace, action=action)
    return namespace


def write_posterior(
    path: str | Path, posterior: Mapping[str, Any], metadata: Mapping[str, Any]
) -> str:
    tensors = {
        name: value.detach().cpu()
        for name, value in posterior.items()
        if hasattr(value, "detach")
    }
    if not tensors:
        raise ValueError("posterior artifact requires tensor values")
    payload = {
        "schema": POSTERIOR_SCHEMA,
        "contract": POSTERIOR_CONTRACT,
        "metadata": dict(metadata),
        "tensors": tensors,
        "tensor_hashes": {
            name: tensor_sha256(value) for name, value in tensors.items()
        },
        "immutable": True,
    }
    target = Path(path)
    if target.suffix != ".pt":
        raise ValueError("posterior artifact path must end in .pt")
    sidecar = target.with_suffix(".json")
    payload_hash = canonical_sha256(payload)
    if target.exists() or sidecar.exists():
        if sidecar.exists() and not target.exists():
            raise FileExistsError("posterior sidecar exists without its artifact")
        if target.exists() and not target.is_file():
            raise FileExistsError("posterior artifact path is not a file")
        if target.is_file() and not sidecar.exists():
            existing = safe_torch_load(target)
            if (
                not isinstance(existing, dict)
                or canonical_sha256(existing) != payload_hash
            ):
                raise FileExistsError(
                    "orphaned posterior artifact has different content"
                )
            artifact_hash = file_sha256(target)
            atomic_write_json(
                sidecar,
                _posterior_sidecar(
                    artifact_hash, payload_hash, metadata, payload["tensor_hashes"]
                ),
            )
            return artifact_hash
        existing = _read_posterior_payload(target, sidecar)
        if canonical_sha256(existing) != payload_hash:
            raise FileExistsError(
                "posterior artifact already exists with different content"
            )
        return file_sha256(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary_pt = Path(handle.name)
    temporary_json = temporary_pt.with_name(temporary_pt.name + ".json")
    try:
        import torch

        torch.save(payload, temporary_pt)
        with temporary_pt.open("rb") as handle:
            os.fsync(handle.fileno())
        artifact_hash = file_sha256(temporary_pt)
        temporary_json.write_text(
            json.dumps(
                _posterior_sidecar(
                    artifact_hash, payload_hash, metadata, payload["tensor_hashes"]
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with temporary_json.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_pt, target)
        # The sidecar is the commit marker: consumers reject the intermediate state.
        os.replace(temporary_json, sidecar)
    finally:
        temporary_pt.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
    return artifact_hash


def _posterior_sidecar(
    artifact_hash: str,
    payload_hash: str,
    metadata: Mapping[str, Any],
    tensor_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": POSTERIOR_SCHEMA,
        "contract": POSTERIOR_CONTRACT,
        "artifact_sha256": artifact_hash,
        "payload_sha256": payload_hash,
        "metadata": dict(metadata),
        "tensor_hashes": dict(tensor_hashes),
        "immutable": True,
    }


def _read_posterior_payload(target: Path, sidecar: Path) -> dict[str, Any]:
    if not target.is_file() or not sidecar.is_file():
        raise ValueError("posterior artifact or immutable sidecar is missing")
    try:
        sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("posterior artifact sidecar is not valid JSON") from exc
    if (
        not isinstance(sidecar_payload, dict)
        or sidecar_payload.get("schema") != POSTERIOR_SCHEMA
        or sidecar_payload.get("contract") != POSTERIOR_CONTRACT
        or sidecar_payload.get("immutable") is not True
        or sidecar_payload.get("artifact_sha256") != file_sha256(target)
    ):
        raise ValueError("posterior artifact sidecar or file hash is invalid")
    payload = safe_torch_load(target)
    if not isinstance(payload, dict) or canonical_sha256(
        payload
    ) != sidecar_payload.get("payload_sha256"):
        raise ValueError("posterior artifact payload hash is invalid")
    if payload.get("tensor_hashes") != sidecar_payload.get(
        "tensor_hashes"
    ) or payload.get("metadata") != sidecar_payload.get("metadata"):
        raise ValueError("posterior artifact sidecar content does not match payload")
    return payload


def load_posterior(
    path: str | Path, *, expected: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    target = Path(path)
    payload = _read_posterior_payload(target, target.with_suffix(".json"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != POSTERIOR_SCHEMA
        or payload.get("contract") != POSTERIOR_CONTRACT
        or payload.get("immutable") is not True
    ):
        raise ValueError(
            "posterior artifact schema or immutability contract is invalid"
        )
    tensors = payload.get("tensors")
    hashes = payload.get("tensor_hashes")
    if not isinstance(tensors, dict) or not isinstance(hashes, dict):
        raise ValueError("posterior artifact tensor payload is invalid")
    for name, value in tensors.items():
        if name not in hashes or tensor_sha256(value) != hashes[name]:
            raise ValueError(f"posterior artifact tensor hash mismatch: {name}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("posterior artifact metadata is invalid")
    for key, value in (expected or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f"posterior artifact metadata mismatch: {key}")
    return tensors


def reuse_decision(
    stage: str,
    *,
    producer_contract: str,
    expected_inputs: Mapping[str, Any],
    artifact: str | Path,
) -> dict[str, Any]:
    target = Path(artifact)
    sidecar = target.with_suffix(".json")
    if not target.is_file() or not sidecar.is_file():
        return {"stage": stage, "decision": "run", "reason": "artifact_missing"}
    try:
        if stage == "posterior" or target.suffix == ".pt":
            payload = _read_posterior_payload(target, sidecar)
        else:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("stage evidence must be a JSON object")
            declared_hash = payload.get("output_artifact_sha256") or payload.get(
                "artifact_sha256"
            )
            if declared_hash is not None and declared_hash != file_sha256(target):
                raise ValueError("stage evidence self hash is invalid")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "stage": stage,
            "decision": "run",
            "reason": "artifact_invalid",
            "detail": str(exc),
        }
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not metadata and isinstance(payload, dict):
        metadata = payload
    mismatches: list[str] = []
    actual_contract = metadata.get("producer_contract") or metadata.get(
        "producer_contract_revision"
    )
    if actual_contract != producer_contract:
        mismatches.append("producer_contract_revision")
    for key, value in expected_inputs.items():
        observed = metadata.get(key)
        if observed is None and isinstance(
            metadata.get("input_artifact_hashes"), Mapping
        ):
            observed = metadata["input_artifact_hashes"].get(key)
        if observed != value:
            mismatches.append(key)
    if mismatches:
        return {
            "stage": stage,
            "decision": "run",
            "reason": "contract_mismatch",
            "mismatches": mismatches,
        }
    return {
        "stage": stage,
        "decision": "reuse",
        "reason": "exact_contract_match",
        "artifact_sha256": file_sha256(target),
    }


def stage_action_decision(
    stage: str,
    *,
    producer_contract: str,
    expected_inputs: Mapping[str, Any],
    evidence: str | Path,
) -> dict[str, Any]:
    """Return REUSE/RUN or reject an immutable incompatible output path."""
    decision = reuse_decision(
        stage,
        producer_contract=producer_contract,
        expected_inputs=expected_inputs,
        artifact=evidence,
    )
    if (
        decision["decision"] == "run"
        and Path(evidence).exists()
        and decision["reason"] != "artifact_missing"
    ):
        raise ValueError(
            f"{stage} REJECT: earliest invalid stage={stage}; fields={decision.get('mismatches', decision.get('detail', decision['reason']))}"
        )
    return {**decision, "action": "REUSE" if decision["decision"] == "reuse" else "RUN"}


def stage_record(
    *,
    stage: str,
    lineage: Mapping[str, Any],
    producer_contract: str,
    input_artifact_hashes: Mapping[str, str],
    output_artifact: str | Path,
    consumer_contract: str,
    permitted_consumers: list[str],
    status: str,
    invocation_mode: str = "manifest",
    clean_worktree: bool | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "producer_commit": lineage.get("git", {}).get("commit"),
        "producer_contract_revision": producer_contract,
        "input_artifact_hashes": dict(input_artifact_hashes),
        "output_artifact": str(output_artifact),
        "consumer_contract": consumer_contract,
        "permitted_consumers": list(permitted_consumers),
        "status": status,
        "invocation_mode": invocation_mode,
        "legacy_diagnostic_only": invocation_mode == "legacy",
        "clean_worktree": clean_worktree,
    }


def require_artifact_hash(
    path: str | Path, expected_sha256: str | None, *, role: str
) -> str:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{role} artifact is missing: {target}")
    actual = file_sha256(target)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"{role} artifact content no longer matches its evidence")
    return actual


def write_failure_record(
    manifest: Mapping[str, Any],
    *,
    stage: str,
    exc: BaseException,
    completed: Mapping[str, Any] | None = None,
) -> Path:
    attempt_id = f"{stage}-{uuid4().hex}"
    root = Path(str(manifest["output_root"])) / "failed-runs" / attempt_id
    root.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "research_failed_run.v1",
        "stage": stage,
        "attempt_id": attempt_id,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback_sha256": canonical_sha256(traceback.format_exc()),
        "earliest_failed_stage": stage,
        "failed_fields": list(getattr(exc, "failed_fields", ())),
        "reuse_decisions": dict(manifest.get("_reuse_decisions", {})),
        "completed_artifacts": dict(completed or {}),
        "scientific_classification": None,
        "permitted_next_actions": [],
        "manifest_sha256": file_sha256(manifest["_manifest_path"]),
    }
    _atomic_write_text(root / "safe-error.txt", f"{type(exc).__name__}: {exc}\n")
    atomic_write_json(root / "completed-artifacts.json", dict(completed or {}))
    atomic_write_json(root / "failure.json", payload)
    return root


def _atomic_write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
