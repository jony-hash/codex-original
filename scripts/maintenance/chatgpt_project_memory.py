"""Build and verify a small, fail-closed Sources package for ChatGPT Projects.

The package is deliberately not an automatic synchronizer.  It is a local,
auditable boundary: the tool copies a fixed set of project/history documents,
records the Git state and file hashes, and rejects stale, modified, secret-
bearing, or unexpected content before a user uploads the Markdown files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
MANIFEST_FILENAME = ".chatgpt-sources-manifest.json"
PACKAGE_KIND = "chatgpt_project_sources"


class ProjectMemoryError(RuntimeError):
    """A user-input or package-integrity failure at this maintenance boundary."""


@dataclass(frozen=True)
class SourceSpec:
    output: str
    source_root: str
    relative_source: str


PROJECT_SOURCE_SPECS = (
    SourceSpec("PROJECT_MEMORY.md", "project", "docs/PROJECT_MEMORY.md"),
    SourceSpec("CURRENT_STATUS.md", "project", "docs/CURRENT_STATUS.md"),
    SourceSpec(
        "CURRENT_HANDOFF.md",
        "project",
        "docs/status/current_handoff.md",
    ),
    SourceSpec(
        "V2_RUNTIME.md",
        "project",
        "docs/architecture/v2_runtime.md",
    ),
)
HISTORY_SOURCE_SPECS = (
    SourceSpec("HISTORY_CONTEXT.md", "history", "HISTORY_CONTEXT.md"),
    SourceSpec("CHAT_INDEX.md", "history", "CHAT_INDEX.md"),
)

SECRET_PATTERNS = (
    (
        "bearer token",
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    ),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    (
        "GitHub token",
        re.compile(r"\b(?:ghp|gho|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}"),
    ),
    (
        "private key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
        ),
    ),
    (
        "sensitive assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|api[ _-]?key)\s*[:=]\s*"
            r"[^\s`\"']{3,}"
        ),
    ),
)
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _strip_extended_prefix(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value)
    if text.startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[len("\\\\?\\UNC\\") :]
    if text.startswith("\\\\?\\"):
        return text[4:]
    return text


def canonical_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a path while treating Windows extended paths as equivalent."""

    return Path(_strip_extended_prefix(value)).expanduser().resolve(strict=False)


def canonical_path_key(value: str | os.PathLike[str]) -> str:
    text = str(canonical_path(value)).rstrip("\\/")
    return os.path.normcase(text)


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return canonical_path_key(child) == canonical_path_key(parent)
    return True


def _resolve_directory(value: str | os.PathLike[str], label: str) -> Path:
    raw = Path(_strip_extended_prefix(value))
    if raw.is_symlink():
        raise ProjectMemoryError(f"{label} must not be a symlink: {value}")
    try:
        resolved = raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ProjectMemoryError(f"{label} is not accessible: {value}") from exc
    if not resolved.is_dir():
        raise ProjectMemoryError(f"{label} is not a directory: {value}")
    return resolved


def _resolve_file(value: Path, label: str) -> Path:
    if value.is_symlink():
        raise ProjectMemoryError(f"{label} must not be a symlink: {value}")
    try:
        resolved = value.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ProjectMemoryError(f"{label} is missing or inaccessible: {value}") from exc
    if not resolved.is_file():
        raise ProjectMemoryError(f"{label} is not a regular file: {value}")
    return resolved


def _validate_roots(
    project_root_arg: str | os.PathLike[str],
    history_root_arg: str | os.PathLike[str],
    output_root_arg: str | os.PathLike[str],
) -> tuple[Path, Path, Path]:
    project_root = _resolve_directory(project_root_arg, "project root")
    history_root = _resolve_directory(history_root_arg, "history root")

    output_raw = Path(_strip_extended_prefix(output_root_arg))
    if output_raw.is_symlink():
        raise ProjectMemoryError(f"output root must not be a symlink: {output_root_arg}")
    output_root = canonical_path(output_raw)
    if canonical_path_key(output_root) == canonical_path_key(project_root) or _is_within(
        output_root, project_root
    ):
        raise ProjectMemoryError("output root must not be the project root or inside it")
    if canonical_path_key(output_root) == canonical_path_key(history_root):
        raise ProjectMemoryError("output root must not be the history root")
    if _is_within(output_root, history_root / "summaries"):
        raise ProjectMemoryError("output root must not be inside the history summaries directory")
    if not output_root.parent.is_dir():
        raise ProjectMemoryError(f"output root parent is missing: {output_root.parent}")
    return project_root, history_root, output_root


def _safe_relative(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProjectMemoryError(f"{label} must be a non-empty relative path")
    if "\\" in value:
        raise ProjectMemoryError(f"{label} must use POSIX separators: {value}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ProjectMemoryError(f"{label} escapes its root: {value}")
    return relative.as_posix()


def _validate_session_id(value: str) -> str:
    if not isinstance(value, str) or not SESSION_ID_PATTERN.fullmatch(value):
        raise ProjectMemoryError(f"invalid session ID: {value!r}")
    return value


def _selected_session_ids(values: Iterable[str]) -> list[str]:
    selected = [_validate_session_id(value) for value in values]
    if len(selected) != len(set(selected)):
        raise ProjectMemoryError("duplicate --session-id values are not allowed")
    return sorted(selected)


def _source_specs(selected_ids: Sequence[str]) -> list[SourceSpec]:
    specs = list(PROJECT_SOURCE_SPECS + HISTORY_SOURCE_SPECS)
    specs.extend(
        SourceSpec(
            f"summaries/{session_id}.md",
            "history",
            f"summaries/{session_id}.md",
        )
        for session_id in selected_ids
    )
    return specs


def _source_path(
    spec: SourceSpec,
    project_root: Path,
    history_root: Path,
) -> Path:
    relative = _safe_relative(spec.relative_source, "source path")
    source_base = project_root if spec.source_root == "project" else history_root
    candidate = source_base / Path(*PurePosixPath(relative).parts)
    resolved = _resolve_file(candidate, f"source file {relative}")
    if not _is_within(resolved, source_base):
        raise ProjectMemoryError(f"source file escapes its root: {candidate}")
    return resolved


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ProjectMemoryError(f"cannot read {label}: {path}") from exc


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _secret_findings(relative_name: str, data: bytes) -> list[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectMemoryError(
            f"Sources content must be UTF-8; cannot scan {relative_name}"
        ) from exc
    findings = [
        f"{relative_name}: {label}"
        for label, pattern in SECRET_PATTERNS
        if pattern.search(text)
    ]
    return findings


def _reject_secrets(contents: Iterable[tuple[str, bytes]]) -> None:
    findings: list[str] = []
    for name, data in contents:
        findings.extend(_secret_findings(name, data))
    if findings:
        raise ProjectMemoryError(
            "secret scan failed: " + "; ".join(sorted(findings))
        )


def _run_git(project_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise ProjectMemoryError("Git is required to build or verify Sources") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown Git error"
        raise ProjectMemoryError(f"Git command failed: {detail}")
    return result.stdout


def _git_state(project_root: Path) -> dict[str, Any]:
    top_level = _resolve_directory(
        _run_git(project_root, "rev-parse", "--show-toplevel").strip(),
        "Git top-level",
    )
    if canonical_path_key(top_level) != canonical_path_key(project_root):
        raise ProjectMemoryError(
            f"project root is not the Git worktree root: {project_root}"
        )
    head = _run_git(project_root, "rev-parse", "HEAD").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", head):
        raise ProjectMemoryError("Git HEAD is not a full commit ID")
    raw_status = _run_git(
        project_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    )
    dirty_paths = sorted(entry for entry in raw_status.split("\0") if entry)
    return {"git_head": head.lower(), "dirty_paths": dirty_paths}


def _json_load(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectMemoryError(f"invalid {label}: {path}") from exc
    if not isinstance(data, dict):
        raise ProjectMemoryError(f"{label} must contain a JSON object: {path}")
    return data


def _hash_entry(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ProjectMemoryError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_manifest_structure(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProjectMemoryError("unsupported or missing Sources manifest schema")
    if manifest.get("kind") != PACKAGE_KIND:
        raise ProjectMemoryError("unexpected Sources manifest kind")
    for field in ("project_root", "history_root", "output_root"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ProjectMemoryError(f"manifest field {field!r} is invalid")
    if not isinstance(manifest.get("git_head"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["git_head"]
    ):
        raise ProjectMemoryError("manifest git_head is invalid")
    dirty_paths = manifest.get("dirty_paths")
    if not isinstance(dirty_paths, list) or any(
        not isinstance(path, str) for path in dirty_paths
    ):
        raise ProjectMemoryError("manifest dirty_paths is invalid")
    selected_ids = manifest.get("selected_session_ids")
    if not isinstance(selected_ids, list) or any(
        not isinstance(value, str) for value in selected_ids
    ):
        raise ProjectMemoryError("manifest selected_session_ids is invalid")
    if selected_ids != sorted(set(selected_ids)):
        raise ProjectMemoryError("manifest selected_session_ids is not canonical")
    for value in selected_ids:
        _validate_session_id(value)

    sources = manifest.get("sources")
    managed_files = manifest.get("managed_files")
    if not isinstance(sources, list) or not isinstance(managed_files, list):
        raise ProjectMemoryError("manifest source inventory is missing")
    if any(not isinstance(value, str) for value in managed_files):
        raise ProjectMemoryError("manifest managed_files is invalid")
    normalized_managed = [_safe_relative(value, "manifest managed file") for value in managed_files]
    if normalized_managed != sorted(set(normalized_managed)):
        raise ProjectMemoryError("manifest managed_files is not canonical")
    if MANIFEST_FILENAME in normalized_managed:
        raise ProjectMemoryError("manifest cannot manage itself as a source file")

    seen_outputs: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ProjectMemoryError("manifest source entry is invalid")
        source_root = source.get("source_root")
        if source_root not in {"project", "history"}:
            raise ProjectMemoryError("manifest source_root is invalid")
        _safe_relative(source.get("source", ""), "manifest source")
        output = _safe_relative(source.get("output", ""), "manifest output")
        if output in seen_outputs:
            raise ProjectMemoryError(f"duplicate manifest output: {output}")
        seen_outputs.add(output)
        _hash_entry(source.get("source_sha256"), f"source hash for {output}")
        _hash_entry(source.get("output_sha256"), f"output hash for {output}")
        if source["source_sha256"] != source["output_sha256"]:
            raise ProjectMemoryError(f"source/output hash mismatch in manifest: {output}")
        for field in ("source_bytes", "output_bytes"):
            value = source.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProjectMemoryError(f"manifest {field} is invalid for {output}")
        if source["source_bytes"] != source["output_bytes"]:
            raise ProjectMemoryError(f"source/output size mismatch in manifest: {output}")
    if sorted(seen_outputs) != normalized_managed:
        raise ProjectMemoryError("manifest managed_files does not match its sources")
    source_count = manifest.get("source_count")
    if source_count != len(sources):
        raise ProjectMemoryError("manifest source_count does not match its sources")


def _expected_manifest_specs(selected_ids: Sequence[str]) -> list[SourceSpec]:
    return _source_specs(selected_ids)


def _validate_manifest_configuration(
    manifest: dict[str, Any],
    project_root: Path,
    history_root: Path,
    output_root: Path,
) -> None:
    _validate_manifest_structure(manifest)
    expected_paths = {
        "project_root": project_root,
        "history_root": history_root,
        "output_root": output_root,
    }
    for field, expected in expected_paths.items():
        if canonical_path_key(manifest[field]) != canonical_path_key(expected):
            raise ProjectMemoryError(f"manifest {field} does not match the requested path")

    selected_ids = manifest["selected_session_ids"]
    expected_specs = _expected_manifest_specs(selected_ids)
    actual_specs = [
        SourceSpec(source["output"], source["source_root"], source["source"])
        for source in manifest["sources"]
    ]
    if sorted(actual_specs, key=lambda spec: spec.output) != sorted(
        expected_specs, key=lambda spec: spec.output
    ):
        raise ProjectMemoryError("manifest source set is not the fixed Sources contract")


def _output_entries(output_root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    try:
        entries = list(output_root.rglob("*"))
    except OSError as exc:
        raise ProjectMemoryError(f"cannot inspect output root: {output_root}") from exc
    for entry in entries:
        relative = entry.relative_to(output_root).as_posix()
        if entry.is_symlink():
            raise ProjectMemoryError(f"symlink is not allowed in Sources output: {relative}")
        if entry.is_dir():
            directories.add(relative)
        elif entry.is_file():
            files.add(relative)
        else:
            raise ProjectMemoryError(f"unsupported output entry: {relative}")
    return files, directories


def _validate_output_layout(
    output_root: Path,
    manifest: dict[str, Any],
) -> None:
    files, directories = _output_entries(output_root)
    managed_files = set(manifest["managed_files"])
    expected_files = managed_files | {MANIFEST_FILENAME}
    if files != expected_files:
        missing = sorted(expected_files - files)
        unknown = sorted(files - expected_files)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ProjectMemoryError("Sources output file set mismatch: " + "; ".join(details))

    allowed_directories = {"summaries"} if any(
        value.startswith("summaries/") for value in managed_files
    ) else set()
    unknown_directories = sorted(directories - allowed_directories)
    if unknown_directories:
        raise ProjectMemoryError(
            "unknown directories in Sources output: " + ",".join(unknown_directories)
        )
    if "summaries" in directories and "summaries" not in allowed_directories:
        raise ProjectMemoryError("empty unmanaged summaries directory in Sources output")


def _validate_output_hashes(output_root: Path, manifest: dict[str, Any]) -> None:
    by_output = {source["output"]: source for source in manifest["sources"]}
    for output in manifest["managed_files"]:
        data = _read_bytes(output_root / Path(*PurePosixPath(output).parts), output)
        entry = by_output[output]
        actual_hash = sha256_bytes(data)
        if actual_hash != entry["output_sha256"]:
            raise ProjectMemoryError(f"output hash mismatch: {output}")
        if len(data) != entry["output_bytes"]:
            raise ProjectMemoryError(f"output size mismatch: {output}")
    manifest_data = _read_bytes(output_root / MANIFEST_FILENAME, "Sources manifest")
    _reject_secrets(
        [(output, _read_bytes(output_root / Path(*PurePosixPath(output).parts), output)) for output in manifest["managed_files"]]
        + [(MANIFEST_FILENAME, manifest_data)]
    )


def _validate_existing_bundle(
    output_root: Path,
    manifest: dict[str, Any],
    project_root: Path,
    history_root: Path,
) -> None:
    _validate_manifest_configuration(manifest, project_root, history_root, output_root)
    _validate_output_layout(output_root, manifest)
    _validate_output_hashes(output_root, manifest)


def _read_sources(
    project_root: Path,
    history_root: Path,
    selected_ids: Sequence[str],
) -> list[tuple[SourceSpec, Path, bytes]]:
    records: list[tuple[SourceSpec, Path, bytes]] = []
    for spec in _source_specs(selected_ids):
        source_path = _source_path(spec, project_root, history_root)
        data = _read_bytes(source_path, spec.relative_source)
        records.append((spec, source_path, data))
    _reject_secrets((spec.relative_source, data) for spec, _, data in records)
    return records


def _manifest_for_sources(
    project_root: Path,
    history_root: Path,
    output_root: Path,
    git_state: dict[str, Any],
    selected_ids: Sequence[str],
    records: Sequence[tuple[SourceSpec, Path, bytes]],
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for spec, _, data in records:
        digest = sha256_bytes(data)
        sources.append(
            {
                "source_root": spec.source_root,
                "source": spec.relative_source,
                "output": spec.output,
                "source_sha256": digest,
                "source_bytes": len(data),
                "output_sha256": digest,
                "output_bytes": len(data),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PACKAGE_KIND,
        "project_root": str(project_root),
        "history_root": str(history_root),
        "output_root": str(output_root),
        "git_head": git_state["git_head"],
        "dirty_paths": git_state["dirty_paths"],
        "selected_session_ids": list(selected_ids),
        "source_count": len(sources),
        "managed_files": sorted(source["output"] for source in sources),
        "sources": sources,
    }


def _json_bytes(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_staged_bundle(
    stage_root: Path,
    manifest: dict[str, Any],
    records: Sequence[tuple[SourceSpec, Path, bytes]],
) -> None:
    for spec, _, data in records:
        target = stage_root / Path(*PurePosixPath(spec.output).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (stage_root / MANIFEST_FILENAME).write_bytes(_json_bytes(manifest))
    _validate_manifest_structure(manifest)
    _validate_output_layout(stage_root, manifest)
    _validate_output_hashes(stage_root, manifest)


def _commit_bundle(
    output_root: Path,
    stage_root: Path,
    old_managed_files: Sequence[str],
    new_managed_files: Sequence[str],
) -> None:
    output_root.mkdir(parents=False, exist_ok=True)
    old_files = (
        [MANIFEST_FILENAME, *old_managed_files] if old_managed_files else []
    )
    new_files = [MANIFEST_FILENAME, *new_managed_files]
    backup_root: Path | None = None
    try:
        if old_files:
            backup_root = Path(
                tempfile.mkdtemp(
                    prefix=".chatgpt-sources-backup-",
                    dir=str(output_root.parent),
                )
            )
            for relative in old_files:
                source = output_root / Path(*PurePosixPath(relative).parts)
                if not source.is_file():
                    raise ProjectMemoryError(
                        f"managed output disappeared during build: {relative}"
                    )
                destination = backup_root / Path(*PurePosixPath(relative).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)

        for relative in new_files:
            source = stage_root / Path(*PurePosixPath(relative).parts)
            destination = output_root / Path(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)

        summaries = output_root / "summaries"
        if summaries.is_dir() and not any(summaries.iterdir()):
            summaries.rmdir()
        if backup_root is not None:
            shutil.rmtree(backup_root)
            backup_root = None
    except (OSError, ProjectMemoryError):
        for relative in new_files:
            target = output_root / Path(*PurePosixPath(relative).parts)
            if target.is_file():
                target.unlink()
        if backup_root is not None:
            for relative in old_files:
                source = backup_root / Path(*PurePosixPath(relative).parts)
                if source.is_file():
                    destination = output_root / Path(*PurePosixPath(relative).parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, destination)
            shutil.rmtree(backup_root)
        raise


def _existing_manifest(
    output_root: Path,
    project_root: Path,
    history_root: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not output_root.exists():
        return None, []
    if not output_root.is_dir():
        raise ProjectMemoryError(f"output root is not a directory: {output_root}")
    files, directories = _output_entries(output_root)
    if not files and not directories:
        return None, []
    manifest_path = output_root / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ProjectMemoryError(
            "existing output contains unmanaged files or no valid Sources manifest"
        )
    manifest = _json_load(manifest_path, "Sources manifest")
    _validate_existing_bundle(output_root, manifest, project_root, history_root)
    return manifest, list(manifest["managed_files"])


def _build(
    project_root_arg: str,
    history_root_arg: str,
    output_root_arg: str,
    requested_session_ids: Sequence[str],
) -> dict[str, Any]:
    project_root, history_root, output_root = _validate_roots(
        project_root_arg, history_root_arg, output_root_arg
    )
    if not (project_root / ".git").exists():
        raise ProjectMemoryError(f"project root is not a Git worktree: {project_root}")
    selected_ids = _selected_session_ids(requested_session_ids)
    records = _read_sources(project_root, history_root, selected_ids)
    git_state = _git_state(project_root)
    _, old_managed_files = _existing_manifest(output_root, project_root, history_root)
    manifest = _manifest_for_sources(
        project_root,
        history_root,
        output_root,
        git_state,
        selected_ids,
        records,
    )

    stage_root = Path(
        tempfile.mkdtemp(prefix=".chatgpt-sources-build-", dir=str(output_root.parent))
    )
    try:
        _write_staged_bundle(stage_root, manifest, records)
        _commit_bundle(
            output_root,
            stage_root,
            old_managed_files,
            manifest["managed_files"],
        )
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)
    return {
        "status": "build_ok",
        "schema_version": SCHEMA_VERSION,
        "project_root": str(project_root),
        "history_root": str(history_root),
        "output_root": str(output_root),
        "git_head": git_state["git_head"],
        "dirty_paths": git_state["dirty_paths"],
        "source_count": manifest["source_count"],
        "selected_session_ids": selected_ids,
        "output_files": manifest["managed_files"],
    }


def _verify(
    project_root_arg: str,
    history_root_arg: str,
    output_root_arg: str,
) -> dict[str, Any]:
    project_root, history_root, output_root = _validate_roots(
        project_root_arg, history_root_arg, output_root_arg
    )
    if not output_root.is_dir():
        raise ProjectMemoryError(f"Sources output does not exist: {output_root}")
    manifest_path = output_root / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ProjectMemoryError("Sources manifest is missing")
    manifest = _json_load(manifest_path, "Sources manifest")
    _validate_manifest_configuration(manifest, project_root, history_root, output_root)
    _validate_output_layout(output_root, manifest)

    selected_ids = manifest["selected_session_ids"]
    records = _read_sources(project_root, history_root, selected_ids)
    by_spec = {
        (spec.output, spec.source_root, spec.relative_source): (source_path, data)
        for spec, source_path, data in records
    }
    for source in manifest["sources"]:
        key = (source["output"], source["source_root"], source["source"])
        source_path, data = by_spec[key]
        source_hash = sha256_bytes(data)
        if source_hash != source["source_sha256"]:
            raise ProjectMemoryError(f"source hash mismatch: {source_path}")
        if len(data) != source["source_bytes"]:
            raise ProjectMemoryError(f"source size mismatch: {source_path}")
    _validate_output_hashes(output_root, manifest)
    git_state = _git_state(project_root)
    if git_state["git_head"] != manifest["git_head"]:
        raise ProjectMemoryError("Git HEAD changed since Sources were built")
    if git_state["dirty_paths"] != manifest["dirty_paths"]:
        raise ProjectMemoryError("Git dirty paths changed since Sources were built")
    return {
        "status": "verify_ok",
        "schema_version": SCHEMA_VERSION,
        "project_root": str(project_root),
        "history_root": str(history_root),
        "output_root": str(output_root),
        "git_head": git_state["git_head"],
        "dirty_paths": git_state["dirty_paths"],
        "source_count": manifest["source_count"],
        "selected_session_ids": selected_ids,
        "output_files": manifest["managed_files"],
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--history-root", required=True)
    parser.add_argument("--output-root", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build a controlled Sources package")
    _add_common_arguments(build_parser)
    build_parser.add_argument("--session-id", action="append", default=[])

    verify_parser = subparsers.add_parser("verify", help="verify a controlled Sources package")
    _add_common_arguments(verify_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            result = _build(
                args.project_root,
                args.history_root,
                args.output_root,
                args.session_id,
            )
        else:
            result = _verify(args.project_root, args.history_root, args.output_root)
    except ProjectMemoryError as exc:
        print(f"PROJECT MEMORY STOPPED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
