"""Restore the immutable WSL outputs archives into a Windows outputs tree.

The command is deliberately fail-closed.  It validates the archive index and
each tar before installing anything, never overwrites an existing file, and
records only the Linux-to-Windows path mappings that are required for NTFS
compatibility.  The source archives remain immutable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from typing import Any, Iterable


PLAN_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
INDEX_FILENAME = "outputs-index.json"
MANIFEST_FILENAME = "outputs-manifest.jsonl"
METADATA_DIRECTORY = "_wsl_restore_metadata"
MAX_WINDOWS_PATH = 240
MIN_FREE_BYTES = 10 * 1024**3
START_MARGIN_BYTES = 20 * 1024**3
CHUNK_SIZE = 8 * 1024 * 1024
CANARY_INDICES = (1, 11, 158, 2)
ARCHIVE_NAME_RE = re.compile(r"^archive_[0-9]{4}\.tar$")
INVALID_COMPONENT_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
RESERVED_COMPONENT_RE = re.compile(
    r"(?i)^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)"
)
TRAILING_COMPONENT_SUFFIX = ".__linux_trailing_dot__"
LONG_COMPONENT_SUFFIX_LENGTH = 13


class RestoreError(RuntimeError):
    """A boundary or archive-integrity failure that must stop restoration."""


@dataclass(frozen=True)
class ArchiveRecord:
    index: int
    top_level: str
    archive_name: str
    file_count: int
    source_bytes: int
    archive_bytes: int
    archive_sha256: str


@dataclass(frozen=True)
class MemberRecord:
    member_name: str
    relative_path: str
    target_relative_path: str
    size: int
    kind: str
    reasons: tuple[str, ...]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise RestoreError(f"temporary metadata file already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_ready(payload), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RestoreError(f"metadata file already exists: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest().upper()


def _same_file_content(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    return _sha256_file(left) == _sha256_file(right)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_existing_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise RestoreError(f"{label} must be an existing directory: {resolved}")
    return resolved


def _resolve_output_target(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise RestoreError(f"target outputs directory must exist: {resolved}")
    return resolved


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise RestoreError(f"expected a regular file: {path}")
    return _sha256_file(path)


def _load_index(archive_root: Path) -> tuple[dict[str, Any], str]:
    path = archive_root / INDEX_FILENAME
    if not path.is_file():
        raise RestoreError(f"missing outputs index: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RestoreError(f"cannot read outputs index: {path}") from exc
    if not isinstance(payload, dict):
        raise RestoreError("outputs index must be an object")
    if payload.get("schema_version") != 1:
        raise RestoreError("unsupported outputs index schema")
    return payload, _file_sha256(path)


def _load_manifest(archive_root: Path) -> tuple[list[ArchiveRecord], str]:
    path = archive_root / MANIFEST_FILENAME
    if not path.is_file():
        raise RestoreError(f"missing outputs manifest: {path}")
    records: list[ArchiveRecord] = []
    seen_indices: set[int] = set()
    seen_top_levels: set[str] = set()
    seen_archives: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RestoreError(f"cannot read outputs manifest: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RestoreError(f"invalid outputs manifest JSON at line {line_number}") from exc
        if not isinstance(payload, dict):
            raise RestoreError(f"outputs manifest entry is not an object at line {line_number}")
        try:
            index = int(payload["index"])
            relative_path = str(payload["relative_path"])
            archive_name = str(payload["archive_name"])
            file_count = int(payload["file_count"])
            source_bytes = int(payload["apparent_bytes"])
            archive_bytes = int(payload["archive_bytes"])
            archive_sha256 = str(payload["sha256"]).upper()
        except (KeyError, TypeError, ValueError) as exc:
            raise RestoreError(f"invalid outputs manifest fields at line {line_number}") from exc
        if not relative_path.startswith("outputs/"):
            raise RestoreError(f"manifest path is outside outputs: {relative_path}")
        top_level = relative_path.removeprefix("outputs/")
        if "/" in top_level or not top_level:
            raise RestoreError(f"manifest entry is not a top-level outputs path: {relative_path}")
        if not ARCHIVE_NAME_RE.fullmatch(archive_name):
            raise RestoreError(f"invalid archive name: {archive_name}")
        if index <= 0 or file_count < 0 or source_bytes < 0 or archive_bytes < 0:
            raise RestoreError(f"negative or zero-invalid manifest values at line {line_number}")
        if not re.fullmatch(r"[0-9A-F]{64}", archive_sha256):
            raise RestoreError(f"invalid archive SHA-256 at line {line_number}")
        if index in seen_indices or top_level in seen_top_levels or archive_name in seen_archives:
            raise RestoreError(f"duplicate outputs manifest entry at line {line_number}")
        seen_indices.add(index)
        seen_top_levels.add(top_level)
        seen_archives.add(archive_name)
        records.append(
            ArchiveRecord(
                index=index,
                top_level=top_level,
                archive_name=archive_name,
                file_count=file_count,
                source_bytes=source_bytes,
                archive_bytes=archive_bytes,
                archive_sha256=archive_sha256,
            )
        )
    records.sort(key=lambda record: record.index)
    if [record.index for record in records] != list(range(1, len(records) + 1)):
        raise RestoreError("outputs manifest indices are not contiguous")
    return records, _file_sha256(path)


def _validate_index(index: dict[str, Any], records: list[ArchiveRecord]) -> None:
    expected_count = index.get("top_level_count")
    if expected_count != len(records):
        raise RestoreError(
            f"top-level count mismatch: index={expected_count}, manifest={len(records)}"
        )
    if index.get("archived_count") != len(records):
        raise RestoreError("outputs index archived_count does not match manifest")
    manifest_bytes = sum(record.source_bytes for record in records)
    if index.get("source_apparent_bytes") != manifest_bytes:
        raise RestoreError("outputs source byte count does not match manifest")


def _normalise_member_name(name: str) -> tuple[str, ...]:
    if "\\" in name:
        raise RestoreError(f"tar member uses a backslash: {name}")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise RestoreError(f"absolute tar member is forbidden: {name}")
    parts = PurePosixPath(name.rstrip("/")).parts
    if not parts or parts[0] != "outputs":
        raise RestoreError(f"tar member is outside outputs: {name}")
    if any(part in {"", ".", ".."} for part in parts):
        raise RestoreError(f"unsafe tar member path: {name}")
    return parts


def _validate_component(component: str) -> None:
    if not component or INVALID_COMPONENT_RE.search(component):
        raise RestoreError(f"tar member contains a Windows-invalid component: {component!r}")
    if RESERVED_COMPONENT_RE.fullmatch(component):
        raise RestoreError(f"tar member contains a Windows-reserved component: {component!r}")


def _map_components(
    relative_parts: tuple[str, ...],
    target_root: Path,
    source_relative: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapped = list(relative_parts)
    reasons: list[str] = []
    for index, component in enumerate(mapped):
        _validate_component(component)
        if component.rstrip(". ") != component:
            stripped = component.rstrip(". ") or "_linux_name"
            mapped[index] = stripped + TRAILING_COMPONENT_SUFFIX
            reasons.append("trailing_dot_or_space")

    def target_length() -> int:
        return len(str(target_root.joinpath(*mapped)))

    while target_length() > MAX_WINDOWS_PATH:
        directory_indices = list(range(max(0, len(mapped) - 1)))
        candidates = directory_indices or [len(mapped) - 1]
        index = max(candidates, key=lambda candidate: (len(mapped[candidate]), -candidate))
        original = mapped[index]
        digest = hashlib.sha256(f"outputs/{source_relative}|{index}".encode("utf-8")).hexdigest()[:12]
        excess = target_length() - MAX_WINDOWS_PATH
        keep = max(1, len(original) - excess - LONG_COMPONENT_SUFFIX_LENGTH)
        shortened = original[:keep] + "~" + digest
        if shortened == original:
            shortened = original[: max(1, len(original) - 1)] + "~" + digest
        mapped[index] = shortened
        reasons.append("path_length")
    if target_length() > MAX_WINDOWS_PATH:
        raise RestoreError(f"cannot map tar member below Windows path limit: {source_relative}")
    return tuple(mapped), tuple(dict.fromkeys(reasons))


def _member_records(
    record: ArchiveRecord,
    archive_path: Path,
    target_root: Path,
) -> list[MemberRecord]:
    entries: list[MemberRecord] = []
    seen_members: set[str] = set()
    seen_targets: dict[str, str] = {}
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive:
            parts = _normalise_member_name(member.name)
            if len(parts) < 2 or parts[1] != record.top_level:
                raise RestoreError(
                    f"archive member does not belong to {record.top_level}: {member.name}"
                )
            normalized = "/".join(parts)
            if normalized in seen_members:
                raise RestoreError(f"duplicate tar member: {member.name}")
            seen_members.add(normalized)
            relative_parts = parts[1:]
            mapped_parts, reasons = _map_components(
                relative_parts,
                target_root,
                "/".join(relative_parts),
            )
            target_relative = "/".join(mapped_parts)
            target_key = target_relative.casefold()
            previous = seen_targets.get(target_key)
            if previous is not None and previous != target_relative:
                raise RestoreError(
                    f"Windows case-insensitive path collision: {previous} and {target_relative}"
                )
            seen_targets[target_key] = target_relative
            if member.isdir():
                kind = "directory"
                size = 0
            elif member.isfile():
                kind = "file"
                size = member.size
            else:
                raise RestoreError(f"unsupported tar member type: {member.name}")
            entries.append(
                MemberRecord(
                    member_name=member.name,
                    relative_path="/".join(relative_parts),
                    target_relative_path=target_relative,
                    size=size,
                    kind=kind,
                    reasons=reasons,
                )
            )
    return entries


def _archive_path(archive_root: Path, record: ArchiveRecord) -> Path:
    candidate = (archive_root / record.archive_name).resolve()
    try:
        candidate.relative_to(archive_root)
    except ValueError as exc:
        raise RestoreError(f"archive escapes archive root: {candidate}") from exc
    return candidate


def _verify_archive_source(archive_root: Path, record: ArchiveRecord) -> Path:
    path = _archive_path(archive_root, record)
    if not path.is_file():
        raise RestoreError(f"archive is missing: {path}")
    if path.stat().st_size != record.archive_bytes:
        raise RestoreError(f"archive size changed: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != record.archive_sha256:
        raise RestoreError(f"archive SHA-256 changed: {path}")
    return path


def _target_path(target_root: Path, relative: str) -> Path:
    path = target_root.joinpath(*PurePosixPath(relative).parts)
    try:
        path.resolve(strict=False).relative_to(target_root)
    except ValueError as exc:
        raise RestoreError(f"mapped target escapes outputs: {relative}") from exc
    return path


def _extended_windows_path(path: Path) -> Path:
    text = str(path)
    if text.startswith("\\\\?\\") or not re.match(r"^[A-Za-z]:[\\/]", text):
        return path
    return Path("\\\\?\\" + text)


def _plan_archive(
    archive_root: Path,
    target_root: Path,
    record: ArchiveRecord,
) -> dict[str, Any]:
    archive_path = _verify_archive_source(archive_root, record)
    members = _member_records(record, archive_path, target_root)
    file_members = [member for member in members if member.kind == "file"]
    source_bytes = sum(member.size for member in file_members)
    if len(file_members) != record.file_count or source_bytes != record.source_bytes:
        raise RestoreError(
            f"archive content mismatch for {record.archive_name}: "
            f"files={len(file_members)}/{record.file_count}, "
            f"bytes={source_bytes}/{record.source_bytes}"
        )
    existing_files: list[str] = []
    for member in file_members:
        target = _target_path(target_root, member.target_relative_path)
        if target.exists():
            if not target.is_file():
                raise RestoreError(f"target file collides with a directory: {target}")
            existing_files.append(member.target_relative_path)
    return {
        "index": record.index,
        "top_level": record.top_level,
        "archive_name": record.archive_name,
        "file_count": record.file_count,
        "source_bytes": record.source_bytes,
        "archive_bytes": record.archive_bytes,
        "archive_sha256": record.archive_sha256,
        "existing_file_count": len(existing_files),
        "existing_files": existing_files,
        "members": [
            {
                "member_name": member.member_name,
                "relative_path": member.relative_path,
                "target_relative_path": member.target_relative_path,
                "size": member.size,
                "kind": member.kind,
                "reasons": member.reasons,
            }
            for member in members
        ],
    }


def _check_plan_target_collisions(plan_archives: list[dict[str, Any]], target_root: Path) -> None:
    seen_targets: dict[str, tuple[str, str]] = {}
    for archive in plan_archives:
        for member in archive["members"]:
            target_relative = str(member["target_relative_path"])
            target_key = target_relative.casefold()
            source = f"{archive['archive_name']}:{member['member_name']}"
            previous = seen_targets.get(target_key)
            if previous is not None and previous[0] != target_relative:
                raise RestoreError(
                    f"cross-archive Windows path collision: {previous[0]} ({previous[1]}) "
                    f"and {target_relative} ({source})"
                )
            seen_targets[target_key] = (target_relative, source)
            target = _target_path(target_root, target_relative)
            if member["kind"] == "file" and target.exists() and not target.is_file():
                raise RestoreError(f"target file collides with a directory: {target}")


def _check_free_space(target_root: Path, required_bytes: int, *, initial: bool) -> int:
    free_bytes = shutil.disk_usage(target_root).free
    margin = START_MARGIN_BYTES if initial else MIN_FREE_BYTES
    if free_bytes < required_bytes + margin:
        raise RestoreError(
            f"insufficient free space for outputs restore: free={free_bytes}, "
            f"required={required_bytes}+{margin}"
        )
    return free_bytes


def build_plan(archive_root: Path, target_root: Path) -> dict[str, Any]:
    archive_root = _resolve_existing_directory(archive_root, "archive root")
    target_root = _resolve_output_target(target_root)
    index, index_sha256 = _load_index(archive_root)
    records, manifest_sha256 = _load_manifest(archive_root)
    _validate_index(index, records)
    plan_archives = [_plan_archive(archive_root, target_root, record) for record in records]
    _check_plan_target_collisions(plan_archives, target_root)
    source_bytes = sum(record.source_bytes for record in records)
    free_bytes = _check_free_space(target_root, source_bytes, initial=True)
    mapped_entries = [
        {
            "archive_name": archive["archive_name"],
            "member_name": member["member_name"],
            "linux_relative_path": "outputs/" + member["relative_path"],
            "windows_relative_path": "outputs/" + member["target_relative_path"],
            "reasons": member["reasons"],
        }
        for archive in plan_archives
        for member in archive["members"]
        if member["reasons"]
    ]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at_utc": _now_utc(),
        "archive_root": archive_root,
        "target_outputs": target_root,
        "index_file": archive_root / INDEX_FILENAME,
        "index_sha256": index_sha256,
        "manifest_file": archive_root / MANIFEST_FILENAME,
        "manifest_sha256": manifest_sha256,
        "top_level_count": len(records),
        "source_file_count": sum(record.file_count for record in records),
        "source_bytes": source_bytes,
        "initial_free_bytes": free_bytes,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "start_margin_bytes": START_MARGIN_BYTES,
        "mapped_entry_count": len(mapped_entries),
        "expected_existing_file_count": sum(
            archive["existing_file_count"] for archive in plan_archives
        ),
        "archives": plan_archives,
        "path_mappings": mapped_entries,
    }


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RestoreError(f"cannot read restore plan: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise RestoreError("unsupported restore plan")
    if not isinstance(payload.get("archives"), list) or not payload["archives"]:
        raise RestoreError("restore plan has no archives")
    return payload


def _current_source_hashes(plan: dict[str, Any]) -> tuple[str, str]:
    index_path = Path(plan["index_file"])
    manifest_path = Path(plan["manifest_file"])
    return _file_sha256(index_path), _file_sha256(manifest_path)


def _assert_plan_sources_unchanged(plan: dict[str, Any]) -> None:
    index_sha256, manifest_sha256 = _current_source_hashes(plan)
    if index_sha256 != plan["index_sha256"] or manifest_sha256 != plan["manifest_sha256"]:
        raise RestoreError("outputs index or manifest changed after dry-run")


def _state_path(state_root: Path) -> Path:
    return state_root / "restore-state.json"


def _load_or_create_state(plan: dict[str, Any], state_root: Path, plan_file: Path) -> dict[str, Any]:
    state_root = state_root.resolve()
    state_path = _state_path(state_root)
    plan_sha256 = _file_sha256(plan_file)
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RestoreError(f"cannot read restore state: {state_path}") from exc
        if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise RestoreError("unsupported restore state")
        if state.get("plan_sha256") != plan_sha256:
            raise RestoreError("restore state belongs to a different plan")
        return state
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "created_at_utc": _now_utc(),
        "updated_at_utc": _now_utc(),
        "archive_status": {},
    }
    _write_json_atomic(state_path, state)
    return state


def _load_existing_state(plan: dict[str, Any], state_root: Path, plan_file: Path) -> dict[str, Any]:
    state_path = _state_path(state_root)
    if not state_path.is_file():
        raise RestoreError(f"restore state is missing: {state_path}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RestoreError(f"cannot read restore state: {state_path}") from exc
    if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RestoreError("unsupported restore state")
    if state.get("plan_sha256") != _file_sha256(plan_file):
        raise RestoreError("restore state belongs to a different plan")
    return state


def _save_state(state: dict[str, Any], state_root: Path) -> None:
    state["updated_at_utc"] = _now_utc()
    _write_json_atomic(_state_path(state_root), state)


def _member_from_payload(payload: dict[str, Any]) -> MemberRecord:
    return MemberRecord(
        member_name=str(payload["member_name"]),
        relative_path=str(payload["relative_path"]),
        target_relative_path=str(payload["target_relative_path"]),
        size=int(payload["size"]),
        kind=str(payload["kind"]),
        reasons=tuple(payload.get("reasons", ())),
    )


def _archive_from_payload(payload: dict[str, Any]) -> ArchiveRecord:
    return ArchiveRecord(
        index=int(payload["index"]),
        top_level=str(payload["top_level"]),
        archive_name=str(payload["archive_name"]),
        file_count=int(payload["file_count"]),
        source_bytes=int(payload["source_bytes"]),
        archive_bytes=int(payload["archive_bytes"]),
        archive_sha256=str(payload["archive_sha256"]),
    )


def _stage_path(staging_root: Path, archive: ArchiveRecord, member: MemberRecord) -> Path:
    path = staging_root / f"{archive.index:04d}" / Path(
        *PurePosixPath(member.target_relative_path).parts
    )
    return _extended_windows_path(path)


def _extract_archive_to_stage(
    archive_root: Path,
    staging_root: Path,
    archive: ArchiveRecord,
    members: list[MemberRecord],
) -> Path:
    archive_path = _verify_archive_source(archive_root, archive)
    archive_stage = staging_root / f"{archive.index:04d}"
    archive_stage.mkdir(parents=True, exist_ok=True)
    member_by_name = {member.member_name: member for member in members}
    with tarfile.open(archive_path, mode="r:") as tar:
        for tar_member in tar:
            member = member_by_name.get(tar_member.name)
            if member is None:
                raise RestoreError(f"tar member differs from dry-run plan: {tar_member.name}")
            destination = _stage_path(staging_root, archive, member)
            if member.kind == "directory":
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not destination.is_file() or destination.stat().st_size != member.size:
                    raise RestoreError(f"staging path already exists with different content: {destination}")
                source = tar.extractfile(tar_member)
                if source is None:
                    raise RestoreError(f"cannot read tar member: {tar_member.name}")
                with source:
                    if _sha256_stream(source) != _sha256_file(destination):
                        raise RestoreError(f"staging content changed: {destination}")
                continue
            source = tar.extractfile(tar_member)
            if source is None:
                raise RestoreError(f"cannot read tar member: {tar_member.name}")
            with source, destination.open("xb") as output:
                copied = 0
                while chunk := source.read(CHUNK_SIZE):
                    output.write(chunk)
                    copied += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if copied != member.size:
                raise RestoreError(f"staged size mismatch: {tar_member.name}")
    staged_files: list[Path] = []
    for root, _, filenames in os.walk(str(_extended_windows_path(archive_stage))):
        staged_files.extend(Path(root) / filename for filename in filenames)
    if len(staged_files) != archive.file_count or sum(path.stat().st_size for path in staged_files) != archive.source_bytes:
        raise RestoreError(f"staging inventory mismatch: {archive.archive_name}")
    return archive_stage


def _verify_existing_collisions(
    target_root: Path,
    archive: ArchiveRecord,
    members: list[MemberRecord],
    archive_stage: Path,
) -> int:
    checked = 0
    for member in members:
        if member.kind != "file":
            continue
        target = _target_path(target_root, member.target_relative_path)
        if not target.exists():
            continue
        staged = _stage_path(archive_stage.parent, archive, member)
        if not staged.is_file():
            raise RestoreError(f"staged collision file is missing: {staged}")
        if not _same_file_content(staged, target):
            raise RestoreError(f"existing target differs from archived file: {target}")
        checked += 1
    return checked


def _install_staged_archive(
    target_root: Path,
    archive: ArchiveRecord,
    members: list[MemberRecord],
    archive_stage: Path,
) -> tuple[int, int]:
    installed = 0
    verified_existing = 0
    for member in members:
        target = _target_path(target_root, member.target_relative_path)
        staged = _stage_path(archive_stage.parent, archive, member)
        if member.kind == "directory":
            if target.exists() and not target.is_dir():
                raise RestoreError(f"target directory collides with a file: {target}")
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists():
            if not _same_file_content(staged, target):
                raise RestoreError(f"existing target differs from archived file: {target}")
            verified_existing += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(staged, target)
        except FileExistsError:
            if not target.is_file() or not _same_file_content(staged, target):
                raise RestoreError(f"target appeared during restore: {target}")
            verified_existing += 1
        except OSError as exc:
            raise RestoreError(f"cannot install staged file without overwrite: {target}") from exc
        else:
            installed += 1
    return installed, verified_existing


def _remove_stage(archive_stage: Path) -> None:
    if archive_stage.exists():
        shutil.rmtree(_extended_windows_path(archive_stage))


def _archive_status_entries(plan_archive: dict[str, Any]) -> list[MemberRecord]:
    return [_member_from_payload(payload) for payload in plan_archive["members"]]


def _restore_order(plan_archives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index = {int(archive["index"]): archive for archive in plan_archives}
    canaries = [by_index[index] for index in CANARY_INDICES if index in by_index]
    canary_set = {int(archive["index"]) for archive in canaries}
    remaining = [archive for archive in plan_archives if int(archive["index"]) not in canary_set]
    return canaries + sorted(remaining, key=lambda archive: int(archive["index"]))


def _write_path_mappings(plan: dict[str, Any], state_root: Path) -> Path:
    path = state_root / "path-map.jsonl"
    if path.exists():
        return path
    rows = [
        {
            "archive_name": item["archive_name"],
            "member_name": item["member_name"],
            "linux_relative_path": item["linux_relative_path"],
            "windows_relative_path": item["windows_relative_path"],
            "reasons": item["reasons"],
        }
        for item in plan["path_mappings"]
    ]
    _write_jsonl_exclusive(path, rows)
    return path


def apply_plan(
    plan_file: Path,
    state_root: Path,
    staging_root: Path,
    *,
    canary_only: bool = False,
) -> dict[str, Any]:
    plan_file = plan_file.resolve()
    plan = _load_plan(plan_file)
    archive_root = _resolve_existing_directory(Path(plan["archive_root"]), "archive root")
    target_root = _resolve_output_target(Path(plan["target_outputs"]))
    state_root = state_root.resolve()
    staging_root = staging_root.resolve()
    if state_root == target_root or state_root in target_root.parents:
        raise RestoreError("state root must not be the outputs root or a parent of it")
    staging_root.mkdir(parents=True, exist_ok=True)
    _assert_plan_sources_unchanged(plan)
    state = _load_or_create_state(plan, state_root, plan_file)
    mapping_path = _write_path_mappings(plan, state_root)
    plan_indices = {int(archive["index"]) for archive in plan["archives"]}
    expected_canaries = set(CANARY_INDICES) & plan_indices
    if not canary_only and expected_canaries == set(CANARY_INDICES):
        incomplete_canaries = [
            str(payload["archive_name"])
            for payload in plan["archives"]
            if int(payload["index"]) in expected_canaries
            and state["archive_status"].get(str(payload["archive_name"]), {}).get("status") != "complete"
        ]
        if incomplete_canaries:
            raise RestoreError(
                "canary archives are incomplete; run apply --canary-only first: "
                + ", ".join(incomplete_canaries)
            )
    if not state["archive_status"]:
        _check_free_space(target_root, int(plan["source_bytes"]), initial=True)
    else:
        _check_free_space(target_root, 0, initial=False)
    completed = 0
    installed_files = 0
    verified_existing = 0
    ordered_archives = _restore_order(plan["archives"])
    if canary_only:
        ordered_archives = ordered_archives[: len(expected_canaries)]
    for payload in ordered_archives:
        archive = _archive_from_payload(payload)
        status = state["archive_status"].get(archive.archive_name, {})
        if status.get("status") == "complete":
            completed += 1
            continue
        members = _archive_status_entries(payload)
        archive_stage = staging_root / f"{archive.index:04d}"
        if status.get("status") == "installed":
            if archive_stage.exists():
                _remove_stage(archive_stage)
            state["archive_status"][archive.archive_name] = {
                "index": archive.index,
                "status": "complete",
                "completed_at_utc": _now_utc(),
            }
            _save_state(state, state_root)
            completed += 1
            continue
        if archive_stage.exists() and status.get("status") not in {"in_progress", "installed"}:
            raise RestoreError(
                f"staging exists without a matching restore state: {archive_stage}"
            )
        _check_free_space(target_root, 0, initial=False)
        state["archive_status"][archive.archive_name] = {
            "index": archive.index,
            "status": "in_progress",
            "started_at_utc": _now_utc(),
        }
        _save_state(state, state_root)
        archive_stage = _extract_archive_to_stage(archive_root, staging_root, archive, members)
        checked = _verify_existing_collisions(target_root, archive, members, archive_stage)
        installed, verified = _install_staged_archive(
            target_root,
            archive,
            members,
            archive_stage,
        )
        state["archive_status"][archive.archive_name] = {
            "index": archive.index,
            "status": "installed",
            "installed_at_utc": _now_utc(),
            "installed_files": installed,
            "verified_existing_files": max(checked, verified),
        }
        _save_state(state, state_root)
        _remove_stage(archive_stage)
        state["archive_status"][archive.archive_name]["status"] = "complete"
        state["archive_status"][archive.archive_name]["completed_at_utc"] = _now_utc()
        _save_state(state, state_root)
        completed += 1
        installed_files += installed
        verified_existing += max(checked, verified)
    remaining = len(plan["archives"]) - completed
    free_bytes = _check_free_space(target_root, 0, initial=False)
    return {
        "status": "canary_ok" if canary_only and remaining > 0 else "apply_ok" if remaining == 0 else "apply_incomplete",
        "completed_archives": completed,
        "total_archives": len(plan["archives"]),
        "installed_files_this_run": installed_files,
        "verified_existing_files_this_run": verified_existing,
        "mapping_file": mapping_path,
        "state_file": _state_path(state_root),
        "staging_root": staging_root,
        "free_bytes": free_bytes,
    }


def _verify_archive_targets(
    target_root: Path,
    payload: dict[str, Any],
) -> tuple[int, int, int]:
    file_count = 0
    source_bytes = 0
    mapped_count = 0
    for member_payload in payload["members"]:
        member = _member_from_payload(member_payload)
        target = _target_path(target_root, member.target_relative_path)
        if member.reasons:
            mapped_count += 1
        if member.kind == "directory":
            if not target.is_dir():
                raise RestoreError(f"restored directory is missing: {target}")
            continue
        if not target.is_file():
            raise RestoreError(f"restored file is missing: {target}")
        if target.stat().st_size != member.size:
            raise RestoreError(f"restored file size changed: {target}")
        file_count += 1
        source_bytes += member.size
    if file_count != payload["file_count"] or source_bytes != payload["source_bytes"]:
        raise RestoreError(f"restored archive inventory mismatch: {payload['archive_name']}")
    return file_count, source_bytes, mapped_count


def verify_plan(plan_file: Path, state_root: Path) -> dict[str, Any]:
    plan_file = plan_file.resolve()
    plan = _load_plan(plan_file)
    archive_root = _resolve_existing_directory(Path(plan["archive_root"]), "archive root")
    target_root = _resolve_output_target(Path(plan["target_outputs"]))
    state_root = state_root.resolve()
    state_path = _state_path(state_root)
    if not state_path.is_file():
        raise RestoreError(f"restore state is missing: {state_path}")
    state = _load_existing_state(plan, state_root, plan_file)
    _assert_plan_sources_unchanged(plan)
    completed = 0
    files = 0
    source_bytes = 0
    mapped_count = 0
    for payload in plan["archives"]:
        archive_name = str(payload["archive_name"])
        if state["archive_status"].get(archive_name, {}).get("status") != "complete":
            raise RestoreError(f"archive is not complete: {archive_name}")
        archive = _archive_from_payload(payload)
        _verify_archive_source(archive_root, archive)
        archive_files, archive_bytes, archive_mapped = _verify_archive_targets(target_root, payload)
        completed += 1
        files += archive_files
        source_bytes += archive_bytes
        mapped_count += archive_mapped
    free_bytes = _check_free_space(target_root, 0, initial=False)
    return {
        "status": "verify_ok",
        "completed_archives": completed,
        "total_archives": len(plan["archives"]),
        "restored_file_entries": files,
        "restored_source_bytes": source_bytes,
        "mapped_entries": mapped_count,
        "expected_source_file_entries": plan["source_file_count"],
        "expected_source_bytes": plan["source_bytes"],
        "free_bytes": free_bytes,
        "state_file": state_path,
        "path_mapping_file": state_root / "path-map.jsonl",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run = subparsers.add_parser("dry-run", help="validate archives and write a restore plan")
    dry_run.add_argument("--archive-root", type=Path, required=True)
    dry_run.add_argument("--target-outputs", type=Path, required=True)
    dry_run.add_argument("--plan-output", type=Path, required=True)

    apply = subparsers.add_parser("apply", help="install a previously validated restore plan")
    apply.add_argument("--plan-file", type=Path, required=True)
    apply.add_argument("--state-root", type=Path, required=True)
    apply.add_argument("--staging-root", type=Path, required=True)
    apply.add_argument("--canary-only", action="store_true")

    verify = subparsers.add_parser("verify", help="verify installed outputs against the plan")
    verify.add_argument("--plan-file", type=Path, required=True)
    verify.add_argument("--state-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "dry-run":
            plan = build_plan(args.archive_root, args.target_outputs)
            _write_json_atomic(args.plan_output.resolve(), plan)
            result = {
                "status": "dry_run_ok",
                "plan_file": args.plan_output.resolve(),
                "top_level_count": plan["top_level_count"],
                "source_file_count": plan["source_file_count"],
                "source_bytes": plan["source_bytes"],
                "initial_free_bytes": plan["initial_free_bytes"],
                "expected_existing_file_count": plan["expected_existing_file_count"],
                "mapped_entry_count": plan["mapped_entry_count"],
            }
        elif args.command == "apply":
            result = apply_plan(
                args.plan_file,
                args.state_root,
                args.staging_root,
                canary_only=args.canary_only,
            )
        else:
            result = verify_plan(args.plan_file, args.state_root)
        print(json.dumps(_json_ready(result), indent=2, ensure_ascii=False))
        return 0
    except (OSError, tarfile.TarError, RestoreError) as exc:
        print(f"WSL OUTPUT RESTORE STOPPED: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
