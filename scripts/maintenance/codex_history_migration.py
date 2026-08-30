#!/usr/bin/env python3
"""Safely merge legacy WSL Codex sessions into the native Windows Codex home.

This is an intentionally explicit, Windows-only migration tool.  It never
imports authentication, config, logs, caches, or plugins.  The source tar is
treated as immutable and the current Codex home is snapshotted before any
database write.

The two-step apply flow is deliberate::

    python ... codex_history_migration.py dry-run
    python ... codex_history_migration.py apply --canary
    # inspect the three sessions in the app, then close the app again
    python ... codex_history_migration.py apply --complete
    python ... codex_history_migration.py verify

Run ``rollback`` only after closing the app and only against the snapshot
created by this tool.  The tool is standard-library-only so it can be run
with the project's Python 3.11 environment after the ChatGPT desktop app is
closed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO, NoReturn


FORMAT_VERSION = 1
DEFAULT_ARCHIVE = Path(r"D:\LDM-Archive\codex-original\backup\wsl-codex-history.tar")
DEFAULT_CODEX_HOME = Path.home() / ".codex"
DEFAULT_PROJECT_ROOT = Path(r"D:\codex-original")
DEFAULT_SNAPSHOT_ROOT = Path.home() / "Documents" / "Codex" / "codex-history-migration-snapshot"
DEFAULT_PLAN_FILE = Path.home() / "Documents" / "Codex" / "codex-history-migration-plan.json"
DEFAULT_CONTEXT_ROOT = Path.home() / "Documents" / "Codex" / "codex-original-history-context"
DEFAULT_PROJECT_NAME = "codex-original · Windows"
DEFAULT_PROJECT_BIND_SNAPSHOT_ROOT = (
    Path.home() / "Documents" / "Codex" / "codex-project-binding-snapshot-20260830"
)
DEFAULT_PROJECT_BIND_MANIFEST = DEFAULT_PROJECT_BIND_SNAPSHOT_ROOT / "binding-manifest.json"
OLD_PROJECT_CWD = "/home/dingzhen/projects/codex-original"
SESSION_ID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.jsonl$",
    re.IGNORECASE,
)
BLOCKED_PROCESS_NAMES = {
    "chatgpt.exe",
    "codex.exe",
    "codex-cli.exe",
    "code.exe",
}
CHUNK_SIZE = 1024 * 1024
EXPECTED_PROJECT_THREAD_COUNT = 214
EXPECTED_PROJECT_VISIBLE_COUNT = 144
EXPECTED_PROJECT_ARCHIVED_COUNT = 70
EXPECTED_MISC_VISIBLE_COUNT = 3


class MigrationError(RuntimeError):
    """A fail-closed migration precondition or integrity error."""


def fail(message: str) -> NoReturn:
    raise MigrationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON manifest {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON manifest is not an object: {path}")
    return value


def normalized_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def session_id_from_name(name: str) -> str | None:
    match = SESSION_ID_RE.search(normalized_member_name(name))
    return match.group(1).lower() if match else None


def session_member_names(archive: Path) -> list[str]:
    if not archive.is_file():
        fail(f"source archive does not exist: {archive}")
    names: list[str] = []
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            name = normalized_member_name(member.name)
            if not member.isfile() or not name.endswith(".jsonl"):
                continue
            if name.startswith(".codex/sessions/") or name.startswith(".codex/archived_sessions/"):
                if session_id_from_name(name):
                    names.append(name)
    return names


def extract_archive_file(archive: Path, member_name: str, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as tar:
        try:
            member = tar.getmember(member_name)
        except KeyError:
            fail(f"archive member is missing: {member_name}")
        source = tar.extractfile(member)
        if source is None:
            fail(f"archive member is not readable: {member_name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=CHUNK_SIZE)


def old_state_from_archive(archive: Path, temporary_root: Path) -> Path:
    destination = temporary_root / "old-state_5.sqlite"
    extract_archive_file(archive, ".codex/state_5.sqlite", destination)
    return destination


def readonly_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        fail(f"Codex state database does not exist: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    escaped = table.replace('"', '""')
    return [row[1] for row in connection.execute(f'PRAGMA table_info("{escaped}")')]


def integrity_check(path: Path) -> str:
    connection = readonly_connection(path)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()
    if result != "ok":
        fail(f"SQLite integrity check failed for {path}: {result}")
    return str(result)


def current_state_summary(codex_home: Path) -> dict[str, Any]:
    database = codex_home / "state_5.sqlite"
    connection = readonly_connection(database)
    try:
        columns = table_columns(connection, "threads")
        rows = connection.execute("SELECT * FROM threads").fetchall()
    finally:
        connection.close()
    ids = {str(row["id"]).lower() for row in rows}
    file_ids: set[str] = set()
    for root_name in ("sessions", "archived_sessions"):
        root = codex_home / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            session_id = session_id_from_name(path.name)
            if session_id:
                file_ids.add(session_id)
    return {
        "database": str(database),
        "columns": columns,
        "thread_count": len(rows),
        "thread_ids": sorted(ids),
        "file_ids": sorted(file_ids),
    }


def old_state_summary(archive: Path, temporary_root: Path) -> dict[str, Any]:
    database = old_state_from_archive(archive, temporary_root)
    integrity_check(database)
    connection = readonly_connection(database)
    try:
        columns = table_columns(connection, "threads")
        rows = connection.execute("SELECT * FROM threads").fetchall()
        edge_columns = table_columns(connection, "thread_spawn_edges")
        edges = connection.execute("SELECT * FROM thread_spawn_edges").fetchall()
    finally:
        connection.close()
    serialized = [{column: row[column] for column in columns} for row in rows]
    return {
        "database": str(database),
        "columns": columns,
        "rows": serialized,
        "edge_columns": edge_columns,
        "edges": [{column: row[column] for column in edge_columns} for row in edges],
    }


def member_inventory(archive: Path) -> dict[str, Any]:
    names = session_member_names(archive)
    if not names:
        fail(f"no Codex session JSONL files found in {archive}")
    by_id: dict[str, list[dict[str, Any]]] = {}
    with tarfile.open(archive, "r:*") as tar:
        members = {normalized_member_name(member.name): member for member in tar if member.isfile()}
        for name in names:
            member = members.get(name)
            if member is None:
                fail(f"archive member disappeared while reading: {name}")
            source_id = session_id_from_name(name)
            if source_id is None:
                fail(f"cannot derive session ID from archive member: {name}")
            source = tar.extractfile(member)
            if source is None:
                fail(f"cannot read archive member: {name}")
            source_hash, source_size = sha256_stream(source)
            by_id.setdefault(source_id, []).append(
                {
                    "member": name,
                    "size": source_size,
                    "sha256": source_hash,
                    "archive_size": member.size,
                }
            )
    for source_id, entries in by_id.items():
        hashes = {entry["sha256"] for entry in entries}
        if len(hashes) != 1:
            fail(f"duplicate session ID has different content: {source_id}")
        # Prefer the normal sessions path when a session exists in both paths.
        entries.sort(key=lambda entry: ("/archived_sessions/" in entry["member"], entry["member"]))
    return {"files": sum(len(entries) for entries in by_id.values()), "by_id": by_id}


def classify_old_rows(old: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    rows = {str(row["id"]).lower(): row for row in old["rows"]}
    file_ids = set(inventory["by_id"])
    row_ids = set(rows)
    if row_ids != file_ids:
        missing_files = sorted(row_ids - file_ids)
        missing_rows = sorted(file_ids - row_ids)
        fail(f"archive/index session mismatch; missing_files={missing_files[:5]}, missing_rows={missing_rows[:5]}")
    for session_id, entries in inventory["by_id"].items():
        if len(entries) > 1 and len({entry["sha256"] for entry in entries}) != 1:
            fail(f"duplicate content mismatch for {session_id}")

    visible_ids = [
        session_id for session_id, row in rows.items() if row.get("source") == "vscode"
    ]
    project_ids = [
        session_id for session_id, row in rows.items() if row.get("cwd") == OLD_PROJECT_CWD
    ]
    hidden_ids = sorted(set(rows) - set(visible_ids))
    main_project_ids = sorted(
        session_id
        for session_id in visible_ids
        if rows[session_id].get("cwd") == OLD_PROJECT_CWD
    )
    misc_visible_ids = sorted(set(visible_ids) - set(main_project_ids))
    return {
        "rows": rows,
        "visible_ids": sorted(visible_ids),
        "project_ids": sorted(project_ids),
        "main_project_ids": main_project_ids,
        "misc_visible_ids": misc_visible_ids,
        "hidden_ids": hidden_ids,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    archive = args.archive
    codex_home = args.codex_home
    if not codex_home.is_dir():
        fail(f"Codex Home does not exist: {codex_home}")
    current = current_state_summary(codex_home)
    with tempfile.TemporaryDirectory(prefix="codex-history-plan-") as temporary:
        old = old_state_summary(archive, Path(temporary))
        inventory = member_inventory(archive)
    classified = classify_old_rows(old, inventory)
    old_ids = set(classified["rows"])
    current_ids = set(current["thread_ids"]) | set(current["file_ids"])
    overlap = sorted(old_ids & current_ids)
    if overlap:
        fail(f"legacy/current session ID collision: {overlap[:10]}")
    current_columns = set(current["columns"])
    old_columns = set(old["columns"])
    if not old_columns.issubset(current_columns):
        fail(f"old threads schema has columns missing in current database: {sorted(old_columns-current_columns)}")
    session_records: list[dict[str, Any]] = []
    for session_id, row in sorted(classified["rows"].items()):
        canonical = inventory["by_id"][session_id][0]
        visible = session_id in set(classified["visible_ids"])
        project_cwd = row.get("cwd") == OLD_PROJECT_CWD
        destination = "sessions" if visible else "archived_sessions"
        archive_member = canonical["member"]
        marker = ".codex/sessions/"
        if marker in archive_member:
            relative = archive_member.split(marker, 1)[1]
        else:
            relative = archive_member.split(".codex/archived_sessions/", 1)[1]
        target_cwd = str(args.project_root) if project_cwd else row.get("cwd")
        session_records.append(
            {
                "id": session_id,
                "source": row.get("source"),
                "title": row.get("title"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "old_cwd": row.get("cwd"),
                "target_cwd": target_cwd,
                "visible": visible,
                "archived": not visible,
                "project_cwd": project_cwd,
                "archive_member": archive_member,
                "relative_path": relative,
                "target_relative_path": f"{destination}/{relative}",
                "source_size": canonical["size"],
                "source_sha256": canonical["sha256"],
            }
        )
    largest = max(session_records, key=lambda record: record["source_size"])
    sorted_by_created = sorted(session_records, key=lambda record: (record["created_at"] or "", record["id"]))
    canary_ids = list(dict.fromkeys([sorted_by_created[0]["id"], sorted_by_created[-1]["id"], largest["id"]]))
    return {
        "format_version": FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "archive": str(archive),
        "archive_size": archive.stat().st_size,
        "archive_mtime_ns": archive.stat().st_mtime_ns,
        "codex_home": str(codex_home),
        "project_root": str(args.project_root),
        "old_project_cwd": OLD_PROJECT_CWD,
        "current_before": {
            "thread_count": current["thread_count"],
            "thread_ids": current["thread_ids"],
            "file_ids": current["file_ids"],
        },
        "old_thread_count": len(classified["rows"]),
        "old_file_count": inventory["files"],
        "visible_count": len(classified["visible_ids"]),
        "project_visible_count": len(classified["main_project_ids"]),
        "misc_visible_count": len(classified["misc_visible_ids"]),
        "hidden_count": len(classified["hidden_ids"]),
        "parent_child_edge_count": len(old["edges"]),
        "canary_ids": canary_ids,
        "session_records": session_records,
        "old_rows": classified["rows"],
        "old_edges": old["edges"],
        "old_edge_columns": old["edge_columns"],
        "old_thread_columns": old["columns"],
    }


def tasklist_names() -> set[str]:
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            check=True,
            capture_output=True,
            text=True,
            encoding="mbcs",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot inspect Windows processes: {exc}")
    names: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.match(r'^"([^"]+)"', line.strip())
        if match:
            names.add(match.group(1).lower())
    return names


def require_apps_closed() -> None:
    blocked = sorted(tasklist_names() & BLOCKED_PROCESS_NAMES)
    if blocked:
        fail("close these applications before apply/rollback: " + ", ".join(blocked))


def copy_db_family(source_home: Path, destination: Path, base_name: str) -> list[str]:
    copied: list[str] = []
    for source in sorted(source_home.glob(base_name + "*")):
        if not source.is_file():
            continue
        target = destination / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(source.name)
    return copied


def create_snapshot(args: argparse.Namespace, plan: dict[str, Any]) -> Path:
    snapshot = args.snapshot_root
    if snapshot.exists() and any(snapshot.iterdir()):
        fail(f"snapshot directory is not empty: {snapshot}")
    snapshot.mkdir(parents=True, exist_ok=True)
    codex_home = args.codex_home
    for directory in ("sessions", "archived_sessions"):
        source = codex_home / directory
        if source.exists():
            shutil.copytree(source, snapshot / directory)
    db_names: list[str] = []
    db_names.extend(copy_db_family(codex_home, snapshot, "state_5.sqlite"))
    db_names.extend(copy_db_family(codex_home, snapshot, "thread_history_1.sqlite"))
    baseline = current_state_summary(codex_home)
    snapshot_info = {
        "format_version": FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "codex_home": str(codex_home),
        "database_files": db_names,
        "baseline_thread_ids": baseline["thread_ids"],
        "baseline_file_ids": baseline["file_ids"],
        "plan_archive": plan["archive"],
    }
    json_dump(snapshot / "snapshot.json", snapshot_info)
    return snapshot


def target_path(args: argparse.Namespace, record: dict[str, Any]) -> Path:
    path = args.codex_home / record["target_relative_path"]
    root = args.codex_home.resolve()
    try:
        path.resolve().relative_to(root)
    except ValueError:
        fail(f"computed target escapes Codex Home: {path}")
    return path


def rewrite_and_stage(
    archive: Path,
    record: dict[str, Any],
    args: argparse.Namespace,
    staging_root: Path,
) -> dict[str, Any]:
    session_id = record["id"]
    stage_path = staging_root / f"{session_id}.jsonl"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    changed_metadata = False
    source_hash = hashlib.sha256()
    transformed_hash = hashlib.sha256()
    body_hash = hashlib.sha256()
    source_size = 0
    transformed_size = 0
    with tarfile.open(archive, "r:*") as tar:
        member = tar.getmember(record["archive_member"])
        source = tar.extractfile(member)
        if source is None:
            fail(f"cannot extract session member: {record['archive_member']}")
        first_line = source.readline()
        if not first_line:
            fail(f"empty session JSONL: {record['id']}")
        source_hash.update(first_line)
        source_size += len(first_line)
        try:
            first_record = json.loads(first_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(f"invalid session_meta JSON in {record['id']}: {exc}")
        if first_record.get("type") != "session_meta" or not isinstance(first_record.get("payload"), dict):
            fail(f"first record is not session_meta in {record['id']}")
        payload = first_record["payload"]
        if str(payload.get("id", "")).lower() != session_id:
            fail(f"session_meta ID mismatch in {record['id']}")
        if record["project_cwd"] and payload.get("cwd") != record["old_cwd"]:
            fail(f"session_meta cwd mismatch in {record['id']}")
        output_first_line = first_line
        if record["project_cwd"] and payload.get("cwd") != record["target_cwd"]:
            payload["cwd"] = record["target_cwd"]
            output_first_line = (json.dumps(first_record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            changed_metadata = True
        transformed_hash.update(output_first_line)
        transformed_size += len(output_first_line)
        with stage_path.open("wb") as target:
            target.write(output_first_line)
            for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                source_hash.update(chunk)
                body_hash.update(chunk)
                transformed_hash.update(chunk)
                target.write(chunk)
                source_size += len(chunk)
                transformed_size += len(chunk)
    if source_size != record["source_size"] or source_hash.hexdigest() != record["source_sha256"]:
        fail(f"source hash/size changed while staging {record['id']}")
    return {
        "id": session_id,
        "stage_path": str(stage_path),
        "source_sha256": source_hash.hexdigest(),
        "transformed_sha256": transformed_hash.hexdigest(),
        "body_sha256": body_hash.hexdigest(),
        "source_size": source_size,
        "transformed_size": transformed_size,
        "metadata_cwd_rewritten": changed_metadata,
    }


def insert_into_state(
    args: argparse.Namespace,
    plan: dict[str, Any],
    selected: list[dict[str, Any]],
    copied: list[tuple[dict[str, Any], Path]],
) -> None:
    database = args.codex_home / "state_5.sqlite"
    connection = sqlite3.connect(database, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        current_columns = table_columns(connection, "threads")
        old_columns = plan["old_thread_columns"]
        if not set(old_columns).issubset(current_columns):
            fail(f"current threads schema is missing old columns: {sorted(set(old_columns)-set(current_columns))}")
        existing = {row[0].lower() for row in connection.execute("SELECT id FROM threads")}
        selected_ids = {record["id"] for record in selected}
        if existing & selected_ids:
            fail(f"selected session IDs already exist in current state database: {sorted(existing & selected_ids)[:5]}")
        connection.execute("BEGIN IMMEDIATE")
        placeholders = ",".join("?" for _ in current_columns)
        quoted_columns = ",".join(f'"{column.replace(chr(34), chr(34) * 2)}"' for column in current_columns)
        for record in selected:
            old_row = plan["old_rows"][record["id"]]
            values: list[Any] = []
            for column in current_columns:
                values.append(old_row.get(column))
            replacements = {column: index for index, column in enumerate(current_columns)}
            values[replacements["rollout_path"]] = str(target_path(args, record))
            values[replacements["cwd"]] = record["target_cwd"]
            values[replacements["archived"]] = 0 if record["visible"] else 1
            if record["visible"] and "archived_at" in replacements:
                values[replacements["archived_at"]] = None
            if "project_id" in replacements:
                # Local Codex projects are folder-bound; leaving this null avoids
                # inventing a ChatGPT Project ID in the local Codex database.
                values[replacements["project_id"]] = None
            connection.execute(
                f"INSERT INTO threads ({quoted_columns}) VALUES ({placeholders})",
                values,
            )
        edge_columns = table_columns(connection, "thread_spawn_edges")
        if edge_columns:
            for edge in plan["old_edges"]:
                if edge.get("parent_thread_id") not in selected_ids or edge.get("child_thread_id") not in selected_ids:
                    continue
                quoted_edge_columns = ",".join(f'"{column.replace(chr(34), chr(34) * 2)}"' for column in edge_columns)
                edge_placeholders = ",".join("?" for _ in edge_columns)
                connection.execute(
                    f"INSERT OR IGNORE INTO thread_spawn_edges ({quoted_edge_columns}) VALUES ({edge_placeholders})",
                    [edge.get(column) for column in edge_columns],
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def checkpoint_database(database: Path) -> None:
    connection = sqlite3.connect(database, timeout=10)
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result and result[0] != 0:
            fail(f"SQLite WAL checkpoint was busy for {database}: {result}")
        check = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            fail(f"SQLite integrity check failed before merge: {check}")
    finally:
        connection.close()


def load_or_build_plan(args: argparse.Namespace, plan_path: Path) -> dict[str, Any]:
    if plan_path.exists():
        plan = load_json(plan_path)
        if plan.get("format_version") != FORMAT_VERSION:
            fail(f"unsupported migration plan format: {plan.get('format_version')}")
        if plan.get("archive") != str(args.archive) or plan.get("codex_home") != str(args.codex_home):
            fail("migration plan paths do not match the requested source and Codex Home")
        if plan.get("archive_size") != args.archive.stat().st_size or plan.get("archive_mtime_ns") != args.archive.stat().st_mtime_ns:
            fail("source archive changed after the migration plan was created")
        return plan
    plan = build_plan(args)
    json_dump(plan_path, plan)
    return plan


def selected_records(plan: dict[str, Any], mode: str, imported_ids: set[str]) -> list[dict[str, Any]]:
    records = {record["id"]: record for record in plan["session_records"]}
    if mode == "canary":
        ids = plan["canary_ids"]
    elif mode == "complete":
        ids = [record["id"] for record in plan["session_records"] if record["id"] not in imported_ids]
    else:
        fail(f"unknown import mode: {mode}")
    return [records[session_id] for session_id in ids]


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("format_version") != FORMAT_VERSION:
        fail(f"unsupported migration manifest format: {manifest.get('format_version')}")
    return manifest


def apply_import(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    require_apps_closed()
    if not args.project_root.is_dir():
        fail(f"target project folder does not exist: {args.project_root}")
    args.codex_home.mkdir(parents=True, exist_ok=True)
    plan = load_or_build_plan(args, args.plan_file)
    manifest: dict[str, Any]
    if args.manifest_file.exists():
        manifest = load_manifest(args.manifest_file)
        if manifest.get("archive") != plan["archive"] or manifest.get("codex_home") != plan["codex_home"]:
            fail("existing migration manifest belongs to a different source or Codex Home")
        imported_ids = set(manifest.get("imported_ids", []))
        snapshot = Path(manifest["snapshot_root"])
    else:
        if mode != "canary":
            fail("complete import requires a successful canary manifest")
        snapshot = create_snapshot(args, plan)
        manifest = {
            "format_version": FORMAT_VERSION,
            "archive": plan["archive"],
            "codex_home": plan["codex_home"],
            "project_root": plan["project_root"],
            "plan_file": str(args.plan_file),
            "snapshot_root": str(snapshot),
            "imported_ids": [],
            "file_records": [],
            "phase": "snapshot_created",
        }
        imported_ids = set()
    selected = selected_records(plan, mode, imported_ids)
    if not selected:
        fail("no sessions remain for this import phase")
    checkpoint_database(args.codex_home / "state_5.sqlite")
    staging_root = snapshot / "staging" / mode
    copied: list[tuple[dict[str, Any], Path]] = []
    staged: list[dict[str, Any]] = []
    try:
        for record in selected:
            final = target_path(args, record)
            if final.exists():
                fail(f"target session file already exists: {final}")
            staged_record = rewrite_and_stage(args.archive, record, args, staging_root)
            staged.append(staged_record)
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_record["stage_path"], final)
            if sha256_file(final) != staged_record["transformed_sha256"]:
                fail(f"target session hash mismatch after copy: {final}")
            copied.append((record, final))
        insert_into_state(args, plan, selected, copied)
    except Exception:
        quarantine = snapshot / "failed-import-quarantine" / str(int(time.time()))
        for _, final in copied:
            if final.exists():
                quarantine.mkdir(parents=True, exist_ok=True)
                shutil.move(str(final), str(quarantine / final.name))
        raise
    for record, final in copied:
        staged_record = next(item for item in staged if item["id"] == record["id"])
        manifest["imported_ids"].append(record["id"])
        manifest["file_records"].append(
            {
                **record,
                "target_path": str(final),
                "transformed_sha256": staged_record["transformed_sha256"],
                "body_sha256": staged_record["body_sha256"],
                "transformed_size": staged_record["transformed_size"],
                "metadata_cwd_rewritten": staged_record["metadata_cwd_rewritten"],
            }
        )
    manifest["imported_ids"] = sorted(set(manifest["imported_ids"]))
    manifest["phase"] = "canary_complete" if mode == "canary" else "complete"
    json_dump(args.manifest_file, manifest)
    return {
        "phase": manifest["phase"],
        "imported_this_run": len(selected),
        "imported_total": len(manifest["imported_ids"]),
        "manifest": str(args.manifest_file),
        "snapshot": str(snapshot),
        "canary_ids": plan["canary_ids"],
    }


def verify_import(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest_file)
    plan = load_json(Path(manifest["plan_file"]))
    current = current_state_summary(args.codex_home)
    integrity = integrity_check(args.codex_home / "state_5.sqlite")
    imported_ids = set(manifest.get("imported_ids", []))
    baseline_ids = set(plan["current_before"]["thread_ids"])
    current_ids = set(current["thread_ids"])
    if not baseline_ids.issubset(current_ids):
        fail("verification found missing pre-existing Windows Codex threads")
    if imported_ids - current_ids:
        fail(f"verification found missing imported state rows: {sorted(imported_ids-current_ids)[:5]}")
    file_checks: list[dict[str, Any]] = []
    for record in manifest.get("file_records", []):
        path = Path(record["target_path"])
        if not path.is_file():
            fail(f"verification found missing imported session file: {path}")
        digest = sha256_file(path)
        if digest != record["transformed_sha256"]:
            fail(f"verification found changed imported session file: {path}")
        file_checks.append({"id": record["id"], "sha256": digest, "size": path.stat().st_size})
    return {
        "integrity": integrity,
        "current_thread_count": current["thread_count"],
        "imported_thread_count": len(imported_ids),
        "expected_total": len(baseline_ids | imported_ids),
        "file_count": len(file_checks),
        "visible_imported": sum(1 for record in manifest["file_records"] if record["visible"]),
        "archived_imported": sum(1 for record in manifest["file_records"] if record["archived"]),
    }


def canonical_path(value: str | Path) -> str:
    """Compare Windows paths without changing the paths stored in transcripts."""
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(value))))


def binding_manifest_path(args: argparse.Namespace) -> Path:
    return args.binding_manifest or (args.binding_snapshot_root / "binding-manifest.json")


def load_project_binding_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the completed session migration and derive an exact binding set."""
    migration = load_manifest(args.manifest_file)
    if migration.get("phase") != "complete":
        fail("project binding requires a complete session migration manifest")
    if canonical_path(migration.get("archive", "")) != canonical_path(args.archive):
        fail("session migration manifest archive does not match the requested archive")
    if canonical_path(migration.get("codex_home", "")) != canonical_path(args.codex_home):
        fail("session migration manifest Codex Home does not match the requested Codex Home")
    if canonical_path(migration.get("project_root", "")) != canonical_path(args.project_root):
        fail("session migration manifest project root does not match the requested project root")

    records = migration.get("file_records")
    imported_ids = migration.get("imported_ids")
    if not isinstance(records, list) or not isinstance(imported_ids, list):
        fail("session migration manifest has no usable imported session records")
    normalized_ids = [str(session_id).lower() for session_id in imported_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        fail("session migration manifest contains duplicate imported IDs")
    record_ids = [str(record.get("id", "")).lower() for record in records if isinstance(record, dict)]
    if len(records) != len(imported_ids) or set(record_ids) != set(normalized_ids):
        fail("session migration manifest imported IDs and file records do not match")

    candidates = [
        record
        for record in records
        if record.get("metadata_cwd_rewritten") is True
    ]
    misc_visible = [
        record
        for record in records
        if record.get("visible") is True and record.get("metadata_cwd_rewritten") is not True
    ]
    if len(candidates) != EXPECTED_PROJECT_THREAD_COUNT:
        fail(
            "unexpected project binding candidate count: "
            f"{len(candidates)} (expected {EXPECTED_PROJECT_THREAD_COUNT})"
        )
    if len(misc_visible) != EXPECTED_MISC_VISIBLE_COUNT:
        fail(
            "unexpected miscellaneous visible session count: "
            f"{len(misc_visible)} (expected {EXPECTED_MISC_VISIBLE_COUNT})"
        )
    visible_count = sum(1 for record in candidates if record.get("visible") is True)
    archived_count = sum(1 for record in candidates if record.get("archived") is True)
    if visible_count != EXPECTED_PROJECT_VISIBLE_COUNT or archived_count != EXPECTED_PROJECT_ARCHIVED_COUNT:
        fail(
            "unexpected project binding visibility counts: "
            f"visible={visible_count}, archived={archived_count}"
        )
    project_root = canonical_path(args.project_root)
    for record in candidates:
        if record.get("project_cwd") is not True:
            fail(f"project binding candidate is missing project_cwd marker: {record.get('id')}")
        if canonical_path(record.get("target_cwd", "")) != project_root:
            fail(f"project binding candidate has a different target cwd: {record.get('id')}")
    for record in misc_visible:
        if record.get("project_cwd") is True:
            fail(f"miscellaneous session is incorrectly marked project_cwd: {record.get('id')}")
    return migration, candidates, misc_visible


def resolve_local_project(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> tuple[str, list[sqlite3.Row]]:
    """Resolve exactly one UI-created local project rooted at the project path."""
    required_tables = {"projects", "project_roots", "threads"}
    available_tables = {
        str(row[1]) for row in connection.execute("PRAGMA table_list").fetchall()
    }
    missing_tables = sorted(required_tables - available_tables)
    if missing_tables:
        fail(f"Codex state database is missing project tables: {missing_tables}")
    project_rows = connection.execute(
        "SELECT id, name FROM projects WHERE name = ?",
        (args.project_name,),
    ).fetchall()
    if len(project_rows) != 1:
        fail(
            f"expected exactly one local project named {args.project_name!r}; "
            f"found {len(project_rows)}"
        )
    project_id = str(project_rows[0]["id"])
    roots = connection.execute(
        "SELECT project_id, position, path FROM project_roots "
        "WHERE project_id = ? ORDER BY position",
        (project_id,),
    ).fetchall()
    primary_roots = [root for root in roots if int(root["position"]) == 0]
    if len(primary_roots) != 1:
        fail(f"local project has {len(primary_roots)} primary roots; expected exactly one")
    if canonical_path(primary_roots[0]["path"]) != canonical_path(args.project_root):
        fail(
            "local project's primary root does not match the requested project root: "
            f"{primary_roots[0]['path']}"
        )
    all_roots = connection.execute("SELECT project_id, position, path FROM project_roots").fetchall()
    matching_roots = [
        root for root in all_roots if canonical_path(root["path"]) == canonical_path(args.project_root)
    ]
    if len(matching_roots) != 1 or str(matching_roots[0]["project_id"]) != project_id:
        fail("project root is duplicated or is also claimed by another local project")
    return project_id, roots


def thread_project_states(
    connection: sqlite3.Connection,
    session_ids: list[str],
) -> dict[str, str | None]:
    if not session_ids:
        return {}
    placeholders = ",".join("?" for _ in session_ids)
    rows = connection.execute(
        f"SELECT id, project_id FROM threads WHERE id IN ({placeholders})",
        session_ids,
    ).fetchall()
    return {
        str(row["id"]).lower(): (None if row["project_id"] is None else str(row["project_id"]))
        for row in rows
    }


def project_binding_state(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    misc_visible: list[dict[str, Any]],
) -> tuple[str, dict[str, str | None], dict[str, str | None]]:
    connection = readonly_connection(args.codex_home / "state_5.sqlite")
    try:
        project_id, _ = resolve_local_project(connection, args)
        candidate_ids = [str(record["id"]).lower() for record in candidates]
        misc_ids = [str(record["id"]).lower() for record in misc_visible]
        candidate_states = thread_project_states(connection, candidate_ids)
        misc_states = thread_project_states(connection, misc_ids)
    finally:
        connection.close()
    missing_candidates = sorted(set(candidate_ids) - set(candidate_states))
    missing_misc = sorted(set(misc_ids) - set(misc_states))
    if missing_candidates or missing_misc:
        fail(
            "project binding found missing imported thread rows: "
            f"candidates={missing_candidates[:5]}, misc={missing_misc[:5]}"
        )
    if any(value is not None for value in misc_states.values()):
        fail("miscellaneous imported sessions are already assigned to a project")
    return project_id, candidate_states, misc_states


def project_binding_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.codex_home.is_dir():
        fail(f"Codex Home does not exist: {args.codex_home}")
    _, candidates, misc_visible = load_project_binding_inputs(args)
    project_id, candidate_states, _ = project_binding_state(args, candidates, misc_visible)
    current_values = list(candidate_states.values())
    target_count = sum(value == project_id for value in current_values)
    null_count = sum(value is None for value in current_values)
    conflict_count = len(current_values) - target_count - null_count
    if conflict_count:
        fail(f"project binding found {conflict_count} candidate threads assigned to another project")
    if target_count and null_count:
        fail("project binding found a partial existing assignment; refusing to guess")
    return {
        "status": "project_bind_dry_run_ok",
        "project_name": args.project_name,
        "project_id": project_id,
        "project_root": str(args.project_root),
        "candidate_threads": len(candidates),
        "visible_threads": sum(1 for record in candidates if record["visible"]),
        "archived_threads": sum(1 for record in candidates if record["archived"]),
        "misc_visible_unbound": len(misc_visible),
        "already_bound": target_count,
        "ready_to_bind": null_count == len(candidates),
        "binding_snapshot": str(args.binding_snapshot_root),
    }


def create_project_binding_snapshot(
    args: argparse.Namespace,
    project_id: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot = args.binding_snapshot_root
    if snapshot.exists() and any(snapshot.iterdir()):
        fail(f"project binding snapshot directory is not empty: {snapshot}")
    checkpoint_database(args.codex_home / "state_5.sqlite")
    snapshot.mkdir(parents=True, exist_ok=True)
    database_files: list[str] = []
    database_files.extend(copy_db_family(args.codex_home, snapshot, "state_5.sqlite"))
    database_files.extend(copy_db_family(args.codex_home, snapshot, "thread_history_1.sqlite"))
    baseline = current_state_summary(args.codex_home)
    connection = readonly_connection(args.codex_home / "state_5.sqlite")
    try:
        edge_count = connection.execute("SELECT count(*) FROM thread_spawn_edges").fetchone()[0]
        candidate_states = thread_project_states(
            connection,
            [str(record["id"]).lower() for record in candidates],
        )
    finally:
        connection.close()
    database_hashes = {name: sha256_file(snapshot / name) for name in database_files}
    snapshot_info = {
        "format_version": FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "codex_home": str(args.codex_home),
        "project_name": args.project_name,
        "project_id": project_id,
        "project_root": str(args.project_root),
        "database_files": database_files,
        "database_sha256": database_hashes,
        "baseline_thread_ids": baseline["thread_ids"],
        "baseline_edge_count": edge_count,
        "candidate_ids": sorted(candidate_states),
        "candidate_original_project_ids": candidate_states,
    }
    json_dump(snapshot / "snapshot.json", snapshot_info)
    return snapshot_info


def bind_project_rows(
    args: argparse.Namespace,
    project_id: str,
    candidate_ids: list[str],
    original_project_ids: dict[str, str | None],
) -> int:
    database = args.codex_home / "state_5.sqlite"
    connection = sqlite3.connect(database, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("BEGIN IMMEDIATE")
        current = thread_project_states(connection, candidate_ids)
        if set(current) != set(candidate_ids):
            fail("project binding state changed: one or more candidate rows disappeared")
        if any(current[session_id] != original_project_ids[session_id] for session_id in candidate_ids):
            fail("project binding state changed after dry-run; refusing to update")
        for session_id in candidate_ids:
            connection.execute(
                "UPDATE threads SET project_id = ? WHERE id = ? AND project_id IS NULL",
                (project_id, session_id),
            )
        changed = sum(
            1
            for session_id in candidate_ids
            if thread_project_states(connection, [session_id])[session_id] == project_id
        )
        if changed != len(candidate_ids):
            fail(f"project binding updated {changed} rows; expected {len(candidate_ids)}")
        connection.commit()
        return changed
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def apply_project_binding(args: argparse.Namespace) -> dict[str, Any]:
    require_apps_closed()
    if not args.project_root.is_dir():
        fail(f"target project folder does not exist: {args.project_root}")
    migration, candidates, misc_visible = load_project_binding_inputs(args)
    project_id, candidate_states, _ = project_binding_state(args, candidates, misc_visible)
    candidate_ids = sorted(candidate_states)
    if any(value is not None for value in candidate_states.values()):
        if all(value == project_id for value in candidate_states.values()):
            fail("all project threads are already bound; run --verify instead")
        fail("some project threads are already assigned; refusing partial binding")
    manifest_path = binding_manifest_path(args)
    if manifest_path.exists():
        fail(f"project binding manifest already exists: {manifest_path}")
    snapshot_info = create_project_binding_snapshot(args, project_id, candidates)
    original_project_ids = {
        session_id: candidate_states[session_id] for session_id in candidate_ids
    }
    binding = {
        "format_version": FORMAT_VERSION,
        "phase": "snapshot_created",
        "created_at": snapshot_info["created_at"],
        "updated_at": snapshot_info["created_at"],
        "migration_manifest": str(args.manifest_file),
        "codex_home": str(args.codex_home),
        "project_name": args.project_name,
        "project_id": project_id,
        "project_root": str(args.project_root),
        "snapshot_root": str(args.binding_snapshot_root),
        "candidate_ids": candidate_ids,
        "candidate_original_project_ids": original_project_ids,
        "visible_count": sum(1 for record in candidates if record["visible"]),
        "archived_count": sum(1 for record in candidates if record["archived"]),
        "misc_visible_ids": sorted(str(record["id"]).lower() for record in misc_visible),
        "baseline_thread_ids": snapshot_info["baseline_thread_ids"],
        "baseline_edge_count": snapshot_info["baseline_edge_count"],
    }
    json_dump(manifest_path, binding)
    changed = bind_project_rows(args, project_id, candidate_ids, original_project_ids)
    binding["phase"] = "complete"
    binding["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    binding["bound_count"] = changed
    json_dump(manifest_path, binding)
    return {
        "phase": binding["phase"],
        "project_name": args.project_name,
        "project_id": project_id,
        "bound_threads": changed,
        "visible_threads": binding["visible_count"],
        "archived_threads": binding["archived_count"],
        "misc_visible_unbound": len(misc_visible),
        "manifest": str(manifest_path),
        "snapshot": str(args.binding_snapshot_root),
        "migration_manifest": str(migration.get("plan_file", args.manifest_file)),
    }


def load_project_binding_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = binding_manifest_path(args)
    if not manifest_path.is_file():
        fail(f"project binding manifest does not exist: {manifest_path}")
    binding = load_json(manifest_path)
    if binding.get("format_version") != FORMAT_VERSION:
        fail(f"unsupported project binding manifest format: {binding.get('format_version')}")
    if canonical_path(binding.get("codex_home", "")) != canonical_path(args.codex_home):
        fail("project binding manifest Codex Home does not match the requested Codex Home")
    if canonical_path(binding.get("project_root", "")) != canonical_path(args.project_root):
        fail("project binding manifest project root does not match the requested project root")
    if binding.get("project_name") != args.project_name:
        fail("project binding manifest project name does not match the requested project name")
    return binding


def verify_project_binding(args: argparse.Namespace) -> dict[str, Any]:
    binding = load_project_binding_manifest(args)
    if binding.get("phase") != "complete":
        fail(f"project binding is not complete: {binding.get('phase')}")
    migration, candidates, misc_visible = load_project_binding_inputs(args)
    expected_ids = sorted(str(record["id"]).lower() for record in candidates)
    if binding.get("candidate_ids") != expected_ids:
        fail("project binding manifest candidate IDs do not match the migration manifest")
    project_id, candidate_states, misc_states = project_binding_state(args, candidates, misc_visible)
    if project_id != str(binding.get("project_id")):
        fail("local project ID changed after binding")
    if any(value != project_id for value in candidate_states.values()):
        fail("not all project binding candidates point to the target project")
    if any(value is not None for value in misc_states.values()):
        fail("a miscellaneous session was assigned to the target project")
    integrity = integrity_check(args.codex_home / "state_5.sqlite")
    connection = readonly_connection(args.codex_home / "state_5.sqlite")
    try:
        edge_count = connection.execute("SELECT count(*) FROM thread_spawn_edges").fetchone()[0]
    finally:
        connection.close()
    if edge_count != binding.get("baseline_edge_count"):
        fail(
            "parent-child edge count changed during project binding: "
            f"{edge_count} != {binding.get('baseline_edge_count')}"
        )
    session_check = verify_import(args)
    return {
        "status": "project_bind_verify_ok",
        "project_name": args.project_name,
        "project_id": project_id,
        "project_root": str(args.project_root),
        "integrity": integrity,
        "bound_threads": len(candidate_states),
        "visible_threads": sum(1 for record in candidates if record["visible"]),
        "archived_threads": sum(1 for record in candidates if record["archived"]),
        "misc_visible_unbound": len(misc_visible),
        "parent_child_edges": edge_count,
        "session_check": session_check,
        "migration_manifest": str(migration.get("plan_file", args.manifest_file)),
    }


def rollback_project_binding(args: argparse.Namespace) -> dict[str, Any]:
    require_apps_closed()
    binding = load_project_binding_manifest(args)
    if binding.get("phase") != "complete":
        fail(f"project binding is not complete: {binding.get('phase')}")
    snapshot = Path(str(binding["snapshot_root"]))
    snapshot_info = load_json(snapshot / "snapshot.json")
    candidate_ids = [str(session_id).lower() for session_id in binding["candidate_ids"]]
    original = {
        str(session_id).lower(): value
        for session_id, value in binding["candidate_original_project_ids"].items()
    }
    database = args.codex_home / "state_5.sqlite"
    connection = sqlite3.connect(database, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("BEGIN IMMEDIATE")
        current = thread_project_states(connection, candidate_ids)
        if set(current) != set(candidate_ids):
            fail("project binding rollback found missing candidate rows")
        if any(current[session_id] != str(binding["project_id"]) for session_id in candidate_ids):
            fail("project binding rollback found a changed project assignment; refusing to overwrite it")
        for session_id in candidate_ids:
            connection.execute(
                "UPDATE threads SET project_id = ? WHERE id = ?",
                (original[session_id], session_id),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    binding["phase"] = "rolled_back"
    binding["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    binding["rolled_back_count"] = len(candidate_ids)
    json_dump(binding_manifest_path(args), binding)
    return {
        "phase": binding["phase"],
        "project_id": binding["project_id"],
        "unbound_threads": len(candidate_ids),
        "snapshot": str(snapshot),
        "snapshot_baseline_thread_count": len(snapshot_info["baseline_thread_ids"]),
    }


def rollback_import(args: argparse.Namespace) -> dict[str, Any]:
    require_apps_closed()
    manifest = load_manifest(args.manifest_file)
    snapshot = Path(manifest["snapshot_root"])
    snapshot_info = load_json(snapshot / "snapshot.json")
    current = current_state_summary(args.codex_home)
    baseline_ids = set(snapshot_info["baseline_thread_ids"])
    allowed_ids = baseline_ids | set(manifest.get("imported_ids", []))
    unexpected = set(current["thread_ids"]) - allowed_ids
    if unexpected:
        fail("rollback refused because new post-snapshot threads exist: " + ", ".join(sorted(unexpected)[:5]))
    quarantine = snapshot / "rollback-quarantine" / str(int(time.time()))
    quarantine.mkdir(parents=True, exist_ok=True)
    for directory in ("sessions", "archived_sessions"):
        current_dir = args.codex_home / directory
        if current_dir.exists():
            shutil.move(str(current_dir), str(quarantine / directory))
        saved_dir = snapshot / directory
        if saved_dir.exists():
            shutil.copytree(saved_dir, current_dir)
    for database_name in snapshot_info["database_files"]:
        current_file = args.codex_home / database_name
        if current_file.exists():
            shutil.move(str(current_file), str(quarantine / database_name))
        saved_file = snapshot / database_name
        if saved_file.exists():
            shutil.copy2(saved_file, current_file)
    result = verify_snapshot_restore(args, snapshot_info)
    return {"rolled_back": True, "quarantine": str(quarantine), **result}


def verify_snapshot_restore(args: argparse.Namespace, snapshot_info: dict[str, Any]) -> dict[str, Any]:
    current = current_state_summary(args.codex_home)
    if current["thread_ids"] != sorted(snapshot_info["baseline_thread_ids"]):
        fail("rollback verification found a different Windows thread ID set")
    integrity = integrity_check(args.codex_home / "state_5.sqlite")
    return {"thread_count": current["thread_count"], "integrity": integrity}


TOKEN_PATTERNS = [
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{12,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"-----BEGIN [^-]+ PRIVATE KEY-----.*?-----END [^-]+ PRIVATE KEY-----", re.S), "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"(?i)(password|passwd|secret|api[_ -]?key)\s*[:=]\s*[^\s,;]+"), r"\1=[REDACTED]"),
]


def redact(value: Any, limit: int = 1600) -> str:
    text = "" if value is None else str(value)
    for pattern, replacement in TOKEN_PATTERNS:
        text = pattern.sub(replacement, text)
    text = text.replace("\x00", "")
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def display_time(value: Any) -> str:
    """Render Codex's numeric timestamps without changing the source record."""
    if value is None or value == "":
        return ""
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return str(value)
    if timestamp > 10**12:
        timestamp /= 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def topic_for(text: str) -> str:
    lowered = text.lower()
    groups = [
        ("迁移与环境", ("wsl", "windows", "迁移", "conda", "cuda", "codex")),
        ("v2 光学生成", ("v2", "optical", "光学", "d2nn", "slm", "cmos")),
        ("训练与评估", ("train", "训练", "pytest", "test", "测试", "checkpoint")),
        ("v1 与教师模型", ("teacher", "教师", "distill", "蒸馏", "vae")),
        ("研究与实验", ("experiment", "实验", "diagnostic", "诊断", "audit", "审计")),
    ]
    for name, keywords in groups:
        if any(keyword in lowered for keyword in keywords):
            return name
    return "其他项目讨论"


def context_from_archive(args: argparse.Namespace) -> dict[str, Any]:
    args.context_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="codex-history-context-") as temporary:
        old = old_state_summary(args.archive, Path(temporary))
    rows = {str(row["id"]).lower(): row for row in old["rows"]}
    visible = [row for row in rows.values() if row.get("source") == "vscode"]
    visible.sort(key=lambda row: (row.get("created_at") or "", str(row["id"])))
    summary_dir = args.context_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.context_root / "CHAT_CATALOG.jsonl"
    with catalog_path.open("w", encoding="utf-8", newline="\n") as catalog:
        for row in visible:
            session_id = str(row["id"]).lower()
            title = redact(row.get("title"), 300)
            first_message = redact(row.get("first_user_message"), 1200)
            topic = topic_for(f"{title}\n{first_message}")
            record = {
                "session_id": session_id,
                "visibility": "visible_main_chat",
                "source": row.get("source"),
                "created_at": display_time(row.get("created_at")),
                "updated_at": display_time(row.get("updated_at")),
                "title": title,
                "topic": topic,
                "recorded_cwd": row.get("cwd"),
                "git_branch": row.get("git_branch"),
                "git_sha": row.get("git_sha"),
            }
            catalog.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            summary = [
                f"# {title or 'Legacy Codex session'}",
                "",
                f"- Session ID: `{session_id}`",
                f"- Topic: {topic}",
                f"- Created: {display_time(row.get('created_at')) or 'unknown'}",
                f"- Original working directory: `{redact(row.get('cwd'), 500)}`",
                f"- Branch: `{redact(row.get('git_branch'), 300)}`",
                f"- Commit: `{redact(row.get('git_sha'), 120)}`",
                "",
                "## Initial request",
                "",
                first_message or "(No first-user-message preview was recorded.)",
                "",
                "## Full transcript",
                "",
                "Open this session by its ID in the imported local Codex project. The immutable source is the WSL history tar listed in PROJECT_MEMORY.md.",
                "",
            ]
            (summary_dir / f"{session_id}.md").write_text("\n".join(summary), encoding="utf-8", newline="\n")
    topic_counts: dict[str, int] = {}
    for row in visible:
        topic = topic_for(f"{row.get('title') or ''}\n{row.get('first_user_message') or ''}")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    memory = [
        "# Codex 历史上下文",
        "",
        "这是 WSL Codex 历史的脱敏检索层，不是原始 transcript。原始逐轮记录仍以 GPT App 本地 Codex 项目中的 session 为准；不要把本目录文件当作事实源的替代品。",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 原始来源：`{args.archive}`",
        f"- 主聊天数：{len(visible)}",
        f"- 内部线程总数：{len(rows) - len(visible)}（默认归档）",
        f"- 主题分布：{json.dumps(topic_counts, ensure_ascii=False, sort_keys=True)}",
        "- 每条完整历史通过 session ID 在本地 Codex 项目中打开；新聊天只按需读取本目录索引，不把全部历史一次性塞入上下文。",
        "",
        "## 使用顺序",
        "",
        "1. 先读 `CHAT_INDEX.md` 定位主题、日期或标题。",
        "2. 再读 `summaries/<session-id>.md` 获取脱敏入口信息。",
        "3. 需要工具输出或逐轮细节时，在 GPT App 的本地 Codex 项目打开对应 session。",
        "4. 项目当前事实以 `D:\\codex-original\\docs\\CURRENT_STATUS.md` 和 Git 为准。",
        "",
        "## 隐私边界",
        "",
        "本目录不复制认证文件、密钥、原始工具输出或完整消息流；生成器对常见 token、私钥和 password/secret 赋值做了脱敏。生成后仍应人工抽查再添加到 ChatGPT Project。",
        "",
    ]
    (args.context_root / "PROJECT_MEMORY.md").write_text("\n".join(memory), encoding="utf-8", newline="\n")
    index_lines = [
        "# Legacy Codex 聊天索引",
        "",
        "以下是旧 WSL Codex 的 147 条主聊天入口。完整 transcript 不复制到本目录；请使用 Session ID 在 GPT App 本地 Codex 项目中打开。",
        "",
        "| 时间 | 主题 | 标题 | Session ID | 原始工作目录 |",
        "|---|---|---|---|---|",
    ]
    for row in visible:
        session_id = str(row["id"]).lower()
        title = redact(row.get("title"), 180).replace("|", "\\|").replace("\n", " ")
        cwd = redact(row.get("cwd"), 100).replace("|", "\\|")
        topic = topic_for(f"{row.get('title') or ''}\n{row.get('first_user_message') or ''}")
        index_lines.append(f"| {display_time(row.get('created_at'))} | {topic} | {title} | `{session_id}` | `{cwd}` |")
    (args.context_root / "CHAT_INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8", newline="\n")
    return {
        "context_root": str(args.context_root),
        "main_chat_count": len(visible),
        "internal_thread_count": len(rows) - len(visible),
        "files": ["PROJECT_MEMORY.md", "CHAT_INDEX.md", "CHAT_CATALOG.jsonl", "summaries/"],
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--codex-home", type=Path, default=DEFAULT_CODEX_HOME)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--plan-file", type=Path, default=DEFAULT_PLAN_FILE)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_SNAPSHOT_ROOT / "migration-manifest.json")


def add_project_binding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-name", default=DEFAULT_PROJECT_NAME)
    parser.add_argument(
        "--binding-snapshot-root",
        type=Path,
        default=DEFAULT_PROJECT_BIND_SNAPSHOT_ROOT,
    )
    parser.add_argument("--binding-manifest", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_run = subparsers.add_parser("dry-run", help="audit source/current state without writing Codex data")
    add_common_arguments(dry_run)
    apply = subparsers.add_parser("apply", help="apply canary or complete merge after app shutdown")
    add_common_arguments(apply)
    group = apply.add_mutually_exclusive_group(required=True)
    group.add_argument("--canary", action="store_true")
    group.add_argument("--complete", action="store_true")
    verify = subparsers.add_parser("verify", help="verify an applied manifest and imported file hashes")
    add_common_arguments(verify)
    rollback = subparsers.add_parser("rollback", help="restore the pre-import snapshot")
    add_common_arguments(rollback)
    project_bind = subparsers.add_parser(
        "project-bind",
        help="bind migrated project threads to a UI-created local project",
    )
    add_common_arguments(project_bind)
    add_project_binding_arguments(project_bind)
    project_group = project_bind.add_mutually_exclusive_group(required=True)
    project_group.add_argument("--dry-run", action="store_true")
    project_group.add_argument("--apply", action="store_true")
    project_group.add_argument("--verify", action="store_true")
    project_group.add_argument("--rollback", action="store_true")
    context = subparsers.add_parser("context", help="generate the external sanitized knowledge layer")
    context.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    context.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "dry-run":
            plan = build_plan(args)
            json_dump(args.plan_file, plan)
            print(
                json.dumps(
                    {
                        "status": "dry_run_ok",
                        "plan": str(args.plan_file),
                        "current_threads": plan["current_before"]["thread_ids"] and len(plan["current_before"]["thread_ids"]) or 0,
                        "legacy_files": plan["old_file_count"],
                        "legacy_unique_threads": plan["old_thread_count"],
                        "visible_main": plan["visible_count"],
                        "project_visible": plan["project_visible_count"],
                        "misc_visible": plan["misc_visible_count"],
                        "hidden_internal": plan["hidden_count"],
                        "canary_ids": plan["canary_ids"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "apply":
            mode = "canary" if args.canary else "complete"
            print(json.dumps(apply_import(args, mode), ensure_ascii=False, indent=2))
        elif args.command == "verify":
            print(json.dumps(verify_import(args), ensure_ascii=False, indent=2))
        elif args.command == "rollback":
            print(json.dumps(rollback_import(args), ensure_ascii=False, indent=2))
        elif args.command == "project-bind":
            if args.dry_run:
                print(json.dumps(project_binding_dry_run(args), ensure_ascii=False, indent=2))
            elif args.apply:
                print(json.dumps(apply_project_binding(args), ensure_ascii=False, indent=2))
            elif args.verify:
                print(json.dumps(verify_project_binding(args), ensure_ascii=False, indent=2))
            else:
                print(json.dumps(rollback_project_binding(args), ensure_ascii=False, indent=2))
        elif args.command == "context":
            print(json.dumps(context_from_archive(args), ensure_ascii=False, indent=2))
        else:
            parser.error(f"unknown command: {args.command}")
    except MigrationError as exc:
        print(f"MIGRATION STOPPED: {exc}", file=sys.stderr)
        return 2
    except (OSError, sqlite3.Error, tarfile.TarError) as exc:
        print(f"MIGRATION ERROR: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
