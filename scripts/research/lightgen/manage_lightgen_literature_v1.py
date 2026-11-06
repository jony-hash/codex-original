"""Download, validate, and index the explicit LightGen literature catalogue.

This is an archival tool, not a paper discovery crawler. It never executes
downloaded code and never follows references found inside papers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree


SCHEMA_VERSION = 1
DEFAULT_RESERVE_GIB = 20
UNKNOWN_ASSET_ESTIMATE = 64 * 1024 * 1024
USER_AGENT = "LightGenLiteratureArchive/1.0 (explicit project research archive)"
STOP_WORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
VERIFIED_ASSET_STATUSES = {
    "verified_existing",
    "downloaded_verified",
    "downloaded_verified_from_staged_file",
    "downloaded_verified_from_resumed_part",
    "verified_local_reused",
    "imported_local_verified",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_catalog(path: Path) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported catalog schema_version: {catalog.get('schema_version')}")
    records = catalog.get("records")
    if not isinstance(records, list):
        raise ValueError("catalog records must be a list")
    record_ids: set[str] = set()
    doi_by_type: set[tuple[str, str]] = set()
    asset_ids: set[str] = set()
    for record in records:
        record_id = record.get("id")
        if not record_id or record_id in record_ids:
            raise ValueError(f"missing or duplicate record id: {record_id}")
        record_ids.add(record_id)
        doi = (record.get("doi") or "").casefold().strip()
        if doi:
            key = (record.get("record_type", "paper"), doi)
            if key in doi_by_type:
                raise ValueError(f"duplicate DOI in {key[0]} records: {doi}")
            doi_by_type.add(key)
        for asset in record.get("assets", []):
            asset_id = asset.get("id")
            if not asset_id or asset_id in asset_ids:
                raise ValueError(f"missing or duplicate asset id: {asset_id}")
            asset_ids.add(asset_id)
    return catalog


def save_catalog(path: Path, catalog: dict[str, Any]) -> None:
    catalog["catalog_updated_utc"] = utc_now()
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    rel = Path(value)
    if rel.is_absolute():
        raise ValueError(f"catalog paths must be repository-relative: {value}")
    resolved = (repo_root / rel).resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        raise ValueError(f"catalog path escapes repository: {value}")
    return resolved


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def title_match_score(title: str, body: str) -> float:
    tokens = [token for token in normalize_text(title).split() if token not in STOP_WORDS]
    if not tokens:
        return 1.0
    body_tokens = set(normalize_text(body).split())
    return sum(token in body_tokens for token in tokens) / len(tokens)


def check_identity(asset: dict[str, Any], body: str) -> tuple[bool, str]:
    expected_doi = (asset.get("expected_doi") or "").casefold().strip()
    expected_title = asset.get("expected_title") or ""
    normalized_body = body.casefold().replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi_found = bool(expected_doi and expected_doi in normalized_body)
    score = title_match_score(expected_title, body) if expected_title else 0.0
    threshold = float(asset.get("title_match_threshold", 0.45))
    if expected_doi and doi_found:
        return True, "expected DOI found"
    if expected_title and score >= threshold:
        return True, f"title token match {score:.3f}"
    if expected_doi or expected_title:
        return False, f"identity mismatch (doi_found={doi_found}, title_score={score:.3f})"
    return True, "no identity hints configured"


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


class _VisibleHTMLText(HTMLParser):
    """Collect visible text while excluding scripts, styles, and templates."""

    _SKIPPED_TAGS = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in self._SKIPPED_TAGS:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in self._SKIPPED_TAGS and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth and data.strip():
            self.parts.append(data.strip())


def visible_html_text(source: str) -> str:
    parser = _VisibleHTMLText()
    parser.feed(source)
    parser.close()
    return " ".join(parser.parts)


def pdf_text_and_pages(path: Path) -> tuple[str, int]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader(str(path), strict=True)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if reader.is_encrypted:
            raise ValueError("encrypted PDF is not readable")
        return text, len(reader.pages)
    except ImportError:
        try:
            import pymupdf as fitz  # type: ignore[import-not-found,no-redef]

            document = fitz.open(path)
            return "\n".join(page.get_text() for page in document), document.page_count
        except ImportError:
            try:
                import fitz  # type: ignore[import-not-found,no-redef]

                document = fitz.open(path)
                return "\n".join(page.get_text() for page in document), document.page_count
            except ImportError:
                raw = path.read_bytes()
                pages = len(re.findall(rb"/Type\s*/Page\b", raw))
                return decode_text(raw), pages


def validate_asset(asset: dict[str, Any], path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"valid": False, "reason": "file not found", "bytes": 0}
    raw = path.read_bytes()
    size = len(raw)
    fmt = (asset.get("format") or path.suffix.lstrip(".")).casefold()
    result: dict[str, Any] = {"valid": False, "bytes": size, "sha256": hashlib.sha256(raw).hexdigest()}
    if size == 0:
        result["reason"] = "empty file"
        return result
    expected_md5 = asset.get("expected_md5")
    if expected_md5 and hashlib.md5(raw).hexdigest().casefold() != expected_md5.casefold():
        result["reason"] = "publisher checksum mismatch"
        return result

    body = ""
    if fmt == "pdf":
        if not raw.startswith(b"%PDF-"):
            result["reason"] = "PDF signature missing (HTML/error page is not a PDF)"
            return result
        try:
            body, page_count = pdf_text_and_pages(path)
        except Exception as exc:  # parser errors are file-boundary validation failures
            result["reason"] = f"PDF parser rejected file: {type(exc).__name__}: {exc}"
            return result
        if page_count < 1:
            result["reason"] = "PDF contains no readable pages"
            return result
        result["page_count"] = page_count
    elif fmt in {"docx", "xlsx", "zip"}:
        if not zipfile.is_zipfile(path):
            result["reason"] = f"{fmt.upper()}/ZIP container is invalid"
            return result
        try:
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
                if bad_member:
                    result["reason"] = f"corrupt archive member: {bad_member}"
                    return result
                names = set(archive.namelist())
                if fmt == "docx":
                    required = "word/document.xml"
                    if required not in names:
                        result["reason"] = "DOCX has no word/document.xml"
                        return result
                    xml = ElementTree.fromstring(archive.read(required))
                    body = " ".join(part for part in xml.itertext())
                elif fmt == "xlsx":
                    if "xl/workbook.xml" not in names:
                        result["reason"] = "XLSX has no workbook"
                        return result
                    populated_cells = 0
                    worksheet_names = sorted(
                        name for name in names
                        if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
                    )
                    for worksheet_name in worksheet_names:
                        worksheet = ElementTree.fromstring(archive.read(worksheet_name))
                        for cell in worksheet.iter():
                            if cell.tag.rsplit("}", 1)[-1] != "c":
                                continue
                            value = next(
                                (child for child in cell if child.tag.rsplit("}", 1)[-1] in {"v", "is"}),
                                None,
                            )
                            if value is not None and "".join(value.itertext()).strip():
                                populated_cells += 1
                    if not worksheet_names or populated_cells == 0:
                        result["reason"] = "XLSX contains no populated worksheet cells"
                        return result
                    result["worksheet_count"] = len(worksheet_names)
                    result["populated_cells"] = populated_cells
                else:
                    if not names:
                        result["reason"] = "ZIP contains no files"
                        return result
                    expected_prefix = asset.get("expected_archive_prefix")
                    if expected_prefix and not any(name.startswith(expected_prefix) for name in names):
                        result["reason"] = f"ZIP does not contain expected archive prefix: {expected_prefix}"
                        return result
                    if expected_prefix:
                        result["expected_archive_prefix_verified"] = expected_prefix
                    required_members = asset.get("required_members", [])
                    missing_members = [name for name in required_members if name not in names]
                    if missing_members:
                        result["reason"] = f"ZIP is missing required members: {missing_members}"
                        return result
                result["archive_members"] = len(names)
        except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            result["reason"] = f"invalid {fmt.upper()} container: {exc}"
            return result
    elif fmt == "xml":
        try:
            root = ElementTree.fromstring(raw)
            body = " ".join(part for part in root.itertext())
        except ElementTree.ParseError as exc:
            result["reason"] = f"invalid XML: {exc}"
            return result
        expected_doi = (asset.get("expected_doi") or "").casefold().strip()
        article_dois = [
            (element.text or "").casefold().strip()
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1].casefold() == "article-id"
            and (element.text or "").strip().casefold().startswith("10.")
        ]
        if expected_doi and article_dois and expected_doi not in article_dois:
            result["reason"] = f"article XML DOI mismatch: expected {expected_doi}, found {article_dois[0]}"
            return result
    elif fmt == "html":
        source = decode_text(raw)
        if "<html" not in source.casefold() and "<!doctype html" not in source.casefold():
            result["reason"] = "HTML document marker missing"
            return result
        body = visible_html_text(source)
        if re.search(r"(access denied|captcha|sign in to continue|request blocked)", body, re.I):
            result["reason"] = "access/login/error page is not article full text"
            return result
        if not body.strip():
            result["reason"] = "HTML document has no visible text"
            return result
    elif fmt == "csv":
        body = decode_text(raw)
        rows = list(csv.reader(body.splitlines()))
        if len(rows) < 2 or not any(any(cell.strip() for cell in row) for row in rows[1:]):
            result["reason"] = "CSV has no data rows"
            return result
    elif fmt in {"markdown", "yaml", "txt", "text"}:
        body = decode_text(raw)
        if not body.strip():
            result["reason"] = "text file is blank or unreadable"
            return result
    elif fmt == "mp4":
        if len(raw) < 12 or raw[4:8] != b"ftyp":
            result["reason"] = "MP4 ftyp signature missing"
            return result
    elif fmt == "png":
        from PIL import Image

        try:
            with Image.open(path) as picture:
                if picture.format != "PNG":
                    raise ValueError("image is not PNG")
                result["width"], result["height"] = picture.size
                picture.verify()
        except (OSError, ValueError) as exc:
            result["reason"] = f"invalid PNG: {exc}"
            return result
    elif fmt == "json":
        try:
            body = json.dumps(json.loads(raw), ensure_ascii=False)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            result["reason"] = f"invalid JSON: {exc}"
            return result
    else:
        result["reason"] = f"unsupported format: {fmt}"
        return result

    identity_ok, identity_message = check_identity(asset, body)
    if not identity_ok:
        result["reason"] = identity_message
        return result
    result.update(valid=True, reason=identity_message, format=fmt)
    return result


def iter_assets(catalog: dict[str, Any]):
    for record in catalog["records"]:
        for asset in record.get("assets", []):
            yield record, asset


def estimate_pending_bytes(catalog: dict[str, Any], selected: list[tuple[dict[str, Any], dict[str, Any]]]) -> int:
    total = 0
    for _, asset in selected:
        if asset.get("acquisition") != "download" or not asset.get("source_url"):
            continue
        total += int(asset.get("estimated_bytes") or UNKNOWN_ASSET_ESTIMATE)
    return total


def resume_partial_download(asset: dict[str, Any], dest: Path, source_part: Path, reserve_bytes: int) -> dict[str, Any]:
    resume_part = dest.with_name(dest.name + ".resume.part")
    if resume_part.exists():
        staged = validate_asset(asset, resume_part)
        if staged["valid"]:
            os.replace(resume_part, dest)
            asset["status"] = "downloaded_verified_from_resumed_part"
            asset["downloaded_utc"] = utc_now()
            asset["verification"] = staged
            return staged
    else:
        shutil.copyfile(source_part, resume_part)

    offset = resume_part.stat().st_size
    if offset <= 0:
        asset["status"] = "download_incomplete"
        asset["verification"] = {"valid": False, "reason": "empty staged file; original partial preserved"}
        return asset["verification"]
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={offset}-"}
    if asset.get("download_etag"):
        headers["If-Range"] = asset["download_etag"]
    elif asset.get("download_last_modified"):
        headers["If-Range"] = asset["download_last_modified"]
    request = urllib.request.Request(asset["source_url"], headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=int(asset.get("timeout_seconds", 120))) as response:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range.strip(), re.I)
            if response.status != 206 or not match or int(match.group(1)) != offset:
                asset["status"] = "download_incomplete"
                asset["verification"] = {
                    "valid": False,
                    "reason": f"server did not confirm requested byte range from {offset}; partial files preserved",
                }
                return asset["verification"]
            total_length = int(match.group(3)) if match.group(3) != "*" else None
            remaining_header = response.headers.get("Content-Length")
            remaining_length = int(remaining_header) if remaining_header and remaining_header.isdigit() else None
            required_free = reserve_bytes + (total_length - offset if total_length is not None else int(asset.get("estimated_bytes") or UNKNOWN_ASSET_ESTIMATE))
            if shutil.disk_usage(dest.parent).free < required_free:
                asset["status"] = "blocked_insufficient_space"
                asset["verification"] = {"valid": False, "reason": "resumed response would violate the configured free-space reserve"}
                return asset["verification"]
            bytes_written = 0
            with resume_part.open("ab") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    bytes_written += len(chunk)
            final_size = resume_part.stat().st_size
            if remaining_length is not None and bytes_written != remaining_length:
                asset["status"] = "download_incomplete"
                asset["verification"] = {
                    "valid": False,
                    "reason": f"resumed response ended after {bytes_written} bytes; Content-Length was {remaining_length}; partial files preserved",
                }
                return asset["verification"]
            if total_length is not None and final_size != total_length:
                asset["status"] = "download_incomplete"
                asset["verification"] = {
                    "valid": False,
                    "reason": f"resumed file has {final_size} bytes; Content-Range total was {total_length}; partial files preserved",
                }
                return asset["verification"]
            checked = validate_asset(asset, resume_part)
            if not checked["valid"]:
                asset["status"] = "rejected_download_payload" if total_length is not None and final_size == total_length else "download_incomplete"
                asset["verification"] = checked
                return checked
            os.replace(resume_part, dest)
            asset["status"] = "downloaded_verified_from_resumed_part"
            asset["downloaded_utc"] = utc_now()
            asset["verification"] = checked
            return checked
    except urllib.error.HTTPError as exc:
        asset["status"] = f"blocked_http_{exc.code}" if exc.code in {401, 403, 404, 429} else "download_failed"
        asset["verification"] = {"valid": False, "reason": f"HTTP {exc.code}: {exc.reason}; partial files preserved"}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        asset["status"] = "download_incomplete"
        asset["verification"] = {"valid": False, "reason": f"{type(exc).__name__}: {exc}; partial files preserved"}
    except http.client.IncompleteRead as exc:
        if exc.partial:
            with resume_part.open("ab") as stream:
                stream.write(exc.partial)
        asset["status"] = "download_incomplete"
        asset["verification"] = {
            "valid": False,
            "reason": f"resumed transfer incomplete at {resume_part.stat().st_size} bytes; partial files preserved",
        }
    return asset["verification"]


def download_one(repo_root: Path, asset: dict[str, Any], reserve_bytes: int) -> dict[str, Any]:
    if asset.get("acquisition") != "download" or not asset.get("source_url"):
        return {"valid": False, "reason": "asset is not marked for direct download"}
    if not asset.get("repo_path"):
        return {"valid": False, "reason": "download asset has no repository destination"}
    dest = resolve_repo_path(repo_root, asset["repo_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        reused = validate_asset(asset, dest)
        if reused["valid"]:
            asset["status"] = "verified_local_reused"
            asset["verification"] = reused
            return reused
        asset["status"] = "existing_file_invalid_preserved"
        asset["verification"] = reused
        return reused
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        staged = validate_asset(asset, part)
        if staged["valid"]:
            os.replace(part, dest)
            asset["status"] = "downloaded_verified_from_staged_file"
            asset["downloaded_utc"] = utc_now()
            asset["verification"] = staged
            return staged
        if asset.get("status") in {"download_incomplete", "incomplete_part_preserved"}:
            return resume_partial_download(asset, dest, part, reserve_bytes)
        if asset.get("status") != "rejected_download_payload":
            asset["status"] = "incomplete_part_preserved"
            asset["verification"] = {"valid": False, "reason": f"staged file preserved pending validation: {part.name}; {staged.get('reason', '')}"}
        return asset.get("verification", staged)

    free = shutil.disk_usage(dest.parent).free
    minimum = reserve_bytes + int(asset.get("estimated_bytes") or UNKNOWN_ASSET_ESTIMATE)
    if free < minimum:
        asset["status"] = "blocked_insufficient_space"
        asset["verification"] = {"valid": False, "reason": f"free bytes {free} below reserve plus estimate {minimum}"}
        return asset["verification"]

    request = urllib.request.Request(asset["source_url"], headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=int(asset.get("timeout_seconds", 120))) as response:
            content_length = response.headers.get("Content-Length")
            expected_length = int(content_length) if content_length and content_length.isdigit() else None
            if expected_length is not None and shutil.disk_usage(dest.parent).free < reserve_bytes + expected_length:
                asset["status"] = "blocked_insufficient_space"
                asset["verification"] = {"valid": False, "reason": "response size exceeds available space while preserving the configured reserve"}
                return asset["verification"]
            asset["download_etag"] = response.headers.get("ETag") or asset.get("download_etag")
            asset["download_last_modified"] = response.headers.get("Last-Modified") or asset.get("download_last_modified")
            bytes_written = 0
            with part.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    bytes_written += len(chunk)
            if expected_length is not None and bytes_written != expected_length:
                asset["status"] = "download_incomplete"
                asset["verification"] = {
                    "valid": False,
                    "reason": f"response ended after {bytes_written} bytes; Content-Length was {expected_length}; partial file preserved",
                }
                return asset["verification"]
            checked = validate_asset(asset, part)
            if not checked["valid"]:
                asset["status"] = "rejected_download_payload"
                asset["verification"] = checked
                return checked
            os.replace(part, dest)
            checked["valid"] = True
            asset["status"] = "downloaded_verified"
            asset["downloaded_utc"] = utc_now()
            asset["verification"] = checked
            return checked
    except urllib.error.HTTPError as exc:
        asset["status"] = f"blocked_http_{exc.code}" if exc.code in {401, 403, 404, 429} else "download_failed"
        asset["verification"] = {"valid": False, "reason": f"HTTP {exc.code}: {exc.reason}"}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        asset["status"] = "download_incomplete" if part.exists() else "download_failed"
        asset["verification"] = {"valid": False, "reason": f"{type(exc).__name__}: {exc}"}
    except http.client.IncompleteRead as exc:
        if exc.partial:
            with part.open("ab") as stream:
                stream.write(exc.partial)
        asset["status"] = "download_incomplete" if part.exists() else "download_failed"
        asset["verification"] = {
            "valid": False,
            "reason": f"IncompleteRead after {part.stat().st_size if part.exists() else 0} bytes; partial file preserved",
        }
    return asset["verification"]


def selected_assets(catalog: dict[str, Any], record_id: str | None, asset_id: str | None):
    selected = []
    for record, asset in iter_assets(catalog):
        if record_id and record["id"] != record_id:
            continue
        if asset_id and asset["id"] != asset_id:
            continue
        role = (asset.get("role") or "").casefold()
        if record.get("record_type", "paper") == "paper" and any(token in role for token in ("article pdf", "main text", "full text xml", "full text html")):
            asset.setdefault("expected_doi", record.get("doi"))
            asset.setdefault("expected_title", record.get("title"))
        asset.setdefault("license", record.get("license", "not reported"))
        if asset.get("acquisition") == "download":
            asset.setdefault("version", "official source version at acquisition; exact identifier not supplied")
        elif asset.get("acquisition") == "existing":
            asset.setdefault("version", "pre-existing project file; original version date not recorded")
        selected.append((record, asset))
    if record_id and not any(record["id"] == record_id for record, _ in selected):
        raise ValueError(f"no assets selected for record id {record_id}")
    if asset_id and not any(asset["id"] == asset_id for _, asset in selected):
        raise ValueError(f"asset id not found: {asset_id}")
    return selected


def run_download(repo_root: Path, catalog: dict[str, Any], pairs, reserve_gib: float) -> list[dict[str, Any]]:
    reserve = int(reserve_gib * (1024**3))
    estimate = estimate_pending_bytes(catalog, pairs)
    free = shutil.disk_usage(repo_root).free
    if free < reserve + estimate:
        raise OSError(f"D: free space {free} is below 20 GiB reserve plus pending estimate {estimate}")
    results = []
    for record, asset in pairs:
        if asset.get("acquisition") != "download":
            continue
        result = download_one(repo_root, asset, reserve)
        results.append({"record_id": record["id"], "asset_id": asset["id"], **result})
        print(f"{record['id']} / {asset['id']}: {asset.get('status')} — {result.get('reason', '')}")
    return results


def run_verify(repo_root: Path, pairs) -> list[dict[str, Any]]:
    results = []
    for record, asset in pairs:
        if not asset.get("repo_path"):
            continue
        path = resolve_repo_path(repo_root, asset["repo_path"])
        result = validate_asset(asset, path)
        if result["valid"]:
            if asset.get("status") not in VERIFIED_ASSET_STATUSES:
                asset["status"] = "verified_existing"
            asset["verification"] = result
        elif path.exists():
            asset["status"] = "local_validation_failed"
            asset["verification"] = result
        elif path.with_name(path.name + ".part").exists():
            if asset.get("status") != "rejected_download_payload":
                asset["status"] = "incomplete_part_preserved"
                asset["verification"] = {"valid": False, "reason": f"staged file present but not committed: {path.name}.part"}
        elif asset.get("status") not in {
            "blocked_http_401", "blocked_http_403", "blocked_http_404", "blocked_http_429",
            "download_failed", "download_incomplete", "blocked_insufficient_space",
            "rejected_download_payload", "not_located", "available_on_request", "metadata_only",
            "access_form_required", "not_applicable",
        }:
            asset["status"] = "missing_local_file"
            asset["verification"] = result
        results.append({"record_id": record["id"], "asset_id": asset["id"], **asset.get("verification", result)})
    return results


def run_import_local(
    repo_root: Path, catalog_path: Path, catalog: dict[str, Any],
    manifest_path: Path, reserve_gib: float, dry_run: bool,
) -> dict[str, Any]:
    """Validate an explicit local-file mapping before committing any copies."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("local import manifest must be an object")
    entries = manifest.get("entries")
    if manifest.get("schema_version") != 1 or not isinstance(entries, list) or not entries:
        raise ValueError("local import manifest requires schema_version=1 and nonempty entries")
    assets = {(record["id"], asset["id"]): asset for record, asset in selected_assets(catalog, None, None)}
    library_dir = catalog_path.parent.resolve()
    receipt: dict[str, Any] = {"created_utc": utc_now(), "dry_run": dry_run, "entries": []}
    seen: set[tuple[str, str]] = set()
    destinations: dict[Path, str] = {}
    pending = []
    with tempfile.TemporaryDirectory(prefix=".local-import-", dir=library_dir) as staging:
        for index, entry in enumerate(entries):
            required = ("record_id", "asset_id", "source_repo_path", "destination_repo_path")
            if not isinstance(entry, dict) or any(not isinstance(entry.get(name), str) or not entry[name] for name in required):
                raise ValueError("each local import entry requires nonempty record_id, asset_id, source_repo_path, destination_repo_path")
            key = (entry["record_id"], entry["asset_id"])
            if key in seen or key not in assets:
                raise ValueError(f"duplicate or unknown import asset: {key}")
            seen.add(key)
            asset = assets[key]
            source = resolve_repo_path(repo_root, entry["source_repo_path"])
            destination = resolve_repo_path(repo_root, entry["destination_repo_path"])
            destination_parts = destination.relative_to(library_dir).parts if destination.is_relative_to(library_dir) else ()
            if not destination_parts or destination_parts[0] not in {"papers", "devices"}:
                raise ValueError(f"import destination must be inside library papers/ or devices/: {destination}")
            if not source.is_file():
                raise ValueError(f"import source file not found: {source}")
            staged = Path(staging) / str(index)
            member = entry.get("archive_member")
            if member is not None:
                if not isinstance(member, str) or not member:
                    raise ValueError("archive_member must be a nonempty string")
                member_path = PurePosixPath(member)
                if member_path.is_absolute() or ".." in member_path.parts or "\\" in member:
                    raise ValueError(f"unsafe archive member: {member}")
                with zipfile.ZipFile(source) as archive:
                    if archive.namelist().count(member) != 1:
                        raise ValueError(f"archive member must occur exactly once: {member}")
                    info = archive.getinfo(member)
                    if info.is_dir():
                        raise ValueError(f"archive member is not a file: {member}")
                    needed = info.file_size
                    if shutil.disk_usage(library_dir).free < int(reserve_gib * 1024**3) + needed:
                        raise OSError("insufficient space for local import and configured reserve")
                    with archive.open(info) as incoming, staged.open("wb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
            else:
                if shutil.disk_usage(library_dir).free < int(reserve_gib * 1024**3) + source.stat().st_size:
                    raise OSError("insufficient space for local import and configured reserve")
                shutil.copyfile(source, staged)
            validation = validate_asset(asset, staged)
            if not validation["valid"]:
                raise ValueError(f"local import rejected {key}: {validation['reason']}")
            if destination in destinations and destinations[destination] != validation["sha256"]:
                raise ValueError(f"import manifest maps different content to the same destination: {destination}")
            destinations[destination] = validation["sha256"]
            action = "copy"
            if destination.exists():
                if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != validation["sha256"]:
                    raise ValueError(f"local import would overwrite different existing content: {destination}")
                action = "reuse_identical"
            row = {
                "record_id": key[0], "asset_id": key[1],
                "source_repo_path": entry["source_repo_path"], "archive_member": member,
                "previous_repo_path": asset.get("repo_path"), "previous_status": asset.get("status"),
                "destination_repo_path": entry["destination_repo_path"], "action": action,
                "verification": validation,
            }
            receipt["entries"].append(row)
            pending.append((asset, staged, destination, row))
        if not dry_run:
            for asset, staged, destination, row in pending:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if hashlib.sha256(destination.read_bytes()).hexdigest() != row["verification"]["sha256"]:
                        raise ValueError(f"import destination changed during validation: {destination}")
                    row["action"] = "reuse_identical"
                else:
                    os.replace(staged, destination)
                history = asset.setdefault("local_imports", [])
                identity = (row["source_repo_path"], row["archive_member"], row["destination_repo_path"])
                if not any((item.get("source_repo_path"), item.get("archive_member"), item.get("destination_repo_path")) == identity for item in history):
                    history.append({**row, "imported_utc": receipt["created_utc"]})
                asset["repo_path"] = row["destination_repo_path"]
                asset["status"] = "imported_local_verified"
                asset["verification"] = row["verification"]
                asset["status_note"] = "Locally imported and validated; original source and previous acquisition status are retained in local_imports."
    receipt["entry_count"] = len(receipt["entries"])
    return receipt


def asset_status_summary(record: dict[str, Any]) -> str:
    assets = record.get("assets", [])
    if not assets:
        return "no file identified"
    return "; ".join(f"{asset.get('role', asset['id'])}: {asset.get('status', 'unverified')}" for asset in assets)


def local_href(library_dir: Path, repo_root: Path, asset: dict[str, Any]) -> str | None:
    repo_path = asset.get("repo_path")
    if not repo_path:
        return None
    try:
        target = resolve_repo_path(repo_root, repo_path)
    except ValueError:
        return None
    if not target.exists():
        return None
    return Path(os.path.relpath(target, library_dir)).as_posix()


def build_outputs(catalog_path: Path, repo_root: Path, catalog: dict[str, Any]) -> None:
    library_dir = catalog_path.parent
    csv_path = library_dir / "index.csv"
    fieldnames = [
        "id", "record_type", "title", "doi", "authors", "venue", "year", "evidence_class", "collection_group", "topics",
        "project_use", "wavelength_roles", "sample_identity", "temporal_conditions", "intensity_energy_conditions",
        "measured_quantities", "measurement_method", "source_locations", "data_availability",
        "publication_version", "correction_status", "associated_dataset_dois", "file_status",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in catalog["records"]:
            wavelengths = record.get("wavelengths", [])
            identity = record.get("sample_identity", {})
            writer.writerow({
                "id": record["id"], "record_type": record.get("record_type", "paper"), "title": record.get("title", ""),
                "doi": record.get("doi", ""), "authors": "; ".join(record.get("authors", [])), "venue": record.get("venue", ""),
                "year": record.get("year", ""), "evidence_class": record.get("evidence_class", ""),
                "collection_group": record.get("collection_group", "未分类"),
                "topics": "; ".join(record.get("topics", [])), "project_use": record.get("project_use", ""),
                "wavelength_roles": "; ".join(f"{item.get('role', 'unspecified')}={item.get('nm', 'unknown')} nm" for item in wavelengths),
                "sample_identity": "; ".join(f"{key}={value}" for key, value in identity.items() if value not in (None, "", "unknown")),
                "temporal_conditions": record.get("temporal_conditions", ""),
                "intensity_energy_conditions": record.get("intensity_energy_conditions", ""),
                "measured_quantities": record.get("measured_quantities", ""),
                "measurement_method": record.get("measurement_method", ""),
                "source_locations": "; ".join(record.get("source_locations", [])),
                "data_availability": record.get("data_availability", ""),
                "publication_version": record.get("publication_version", ""),
                "correction_status": record.get("correction_status", ""),
                "associated_dataset_dois": "; ".join(record.get("associated_dataset_dois", [])),
                "file_status": asset_status_summary(record),
            })

    bib_lines = ["% Generated from catalog.json; edit the catalog, then rebuild."]
    for record in catalog["records"]:
        if record.get("record_type") not in {"paper", "dataset"}:
            continue
        key = f"{(record.get('authors') or ['LightGen'])[0].split(',')[0].split()[-1]}{record.get('year', '')}{record['id']}"
        kind = "article" if record.get("record_type") == "paper" else "misc"
        fields = {"title": record.get("title"), "author": " and ".join(record.get("authors", [])), "journal": record.get("venue"), "year": record.get("year"), "doi": record.get("doi"), "url": record.get("canonical_url")}
        bib_lines.append(f"@{kind}{{{re.sub(r'[^A-Za-z0-9:_-]', '', key)},")
        for name, value in fields.items():
            if value:
                escaped = str(value).replace("{", "\\{").replace("}", "\\}")
                bib_lines.append(f"  {name} = {{{escaped}}},")
        bib_lines.append("}\n")
    (library_dir / "references.bib").write_text("\n".join(bib_lines), encoding="utf-8")

    report = {"generated_utc": utc_now(), "record_count": len(catalog["records"]), "records": []}
    verified_names = "、".join(f"`{status}`" for status in sorted(VERIFIED_ASSET_STATUSES))
    gaps = ["# 当前下载缺口", "", f"仅 {verified_names} 计入文件已核验。受限、申请可得或尚未定位均不计入下载完成。", ""]
    cards = ["# 阅读卡", ""]
    for record in catalog["records"]:
        assets = record.get("assets", [])
        statuses = [asset.get("status", "unverified") for asset in assets]
        verified_count = sum(status in VERIFIED_ASSET_STATUSES for status in statuses)
        report["records"].append({
            "id": record["id"], "title": record.get("title"), "record_type": record.get("record_type"),
            "asset_count": len(assets), "verified_asset_count": verified_count,
            "asset_statuses": statuses,
            "assets": [{
                "id": asset["id"], "role": asset.get("role"), "status": asset.get("status", "unverified"),
                "source_url": asset.get("source_url"), "repo_path": asset.get("repo_path"),
                "verification": asset.get("verification", {}),
                "local_imports": asset.get("local_imports", []),
            } for asset in assets],
        })
        card = record.get("reading_card", "")
        card_wavelengths = "; ".join(
            f"{item.get('role')}={item.get('nm', 'unknown')} nm" for item in record.get("wavelengths", [])
        ) or "未报告/未核"
        conditions = [
            f"- 样品身份：{json.dumps(record.get('sample_identity', {}), ensure_ascii=False)}",
            f"- 波长角色：{card_wavelengths}",
            f"- 时域：{record.get('temporal_conditions', '未从已取得来源核录。')}",
            f"- 强度/能量：{record.get('intensity_energy_conditions', '未从已取得来源核录。')}",
            f"- 实测量：{record.get('measured_quantities', '未从已取得来源核录。')}",
            f"- 方法：{record.get('measurement_method', '未从已取得来源核录。')}",
            f"- 原文定位：{'; '.join(record.get('source_locations', [])) or '未从已取得来源核录。'}",
            f"- 数据可得性：{record.get('data_availability', '未从已取得来源核录。')}",
            f"- 版本：{record.get('publication_version', '未核录。')}",
            f"- 更正记录：{record.get('correction_status', '未逐项核录。')}",
            f"- 关联数据 DOI：{'; '.join(record.get('associated_dataset_dois', [])) or '未登记。'}",
        ]
        cards.extend([
            f"## {record['title']}", "", f"- DOI/来源：{record.get('canonical_url') or record.get('doi') or '无 DOI'}",
            f"- 可消费内容：{card or '见项目用途字段。'}", *conditions,
            f"- 适用边界：{record.get('limits', '未进一步登记。')}", "",
        ])
        for asset in assets:
            if asset.get("status") not in VERIFIED_ASSET_STATUSES:
                gaps.append(f"- **{record['id']} / {asset.get('role', asset['id'])}** — `{asset.get('status', 'unverified')}`；{asset.get('status_note', asset.get('verification', {}).get('reason', '待核验'))}；来源：{asset.get('source_url') or '未定位'}")
    (library_dir / "coverage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status_counts: dict[str, int] = {}
    for _, asset in iter_assets(catalog):
        status = asset.get("status", "unverified")
        status_counts[status] = status_counts.get(status, 0) + 1
    coverage = [
        "# 文献与公开附件下载覆盖报告", "",
        f"生成时间（UTC）：{report['generated_utc']}",
        f"记录数：{len(catalog['records'])}；附件条目数：{sum(len(record.get('assets', [])) for record in catalog['records'])}",
        f"已本地验证附件：{sum(sum(s in VERIFIED_ASSET_STATUSES for s in r.get('asset_statuses', [])) for r in report['records'])}",
        "", "## 状态计数", "",
    ]
    coverage.extend(f"- `{status}`：{count}" for status, count in sorted(status_counts.items()))
    coverage.extend(["", "## 按记录", ""])
    for item in report["records"]:
        coverage.append(f"- **{item['id']}** — {item['verified_asset_count']}/{item['asset_count']} 个附件已核验；" + "; ".join(item["asset_statuses"]))
    coverage.extend(["", "状态是文件获取/格式/身份验证结果；不表示论文参数适用于目标器件。完整逐附件 URL、路径和验证信息见 `coverage_report.json`。", ""])
    (library_dir / "download_coverage_report.md").write_text("\n".join(coverage), encoding="utf-8")
    (library_dir / "reading_cards.md").write_text("\n".join(cards), encoding="utf-8")
    (library_dir / "open_gaps.md").write_text("\n".join(gaps) + "\n", encoding="utf-8")

    groups = sorted({record.get("collection_group", "未分类") for record in catalog["records"]})
    topic_lines = ["# 按主题浏览文献库", "", "由 catalog.json 生成；正文、补充材料、数据与代码在每条记录下列出。", ""]
    for group in groups:
        topic_lines.extend([f"## {group}", ""])
        for record in catalog["records"]:
            if record.get("collection_group", "未分类") != group:
                continue
            topic_lines.append(f"- **{record['title']}** — {record.get('venue', '')} {record.get('year', '')}；{record.get('doi') or '器件资料'}")
            for asset in record.get("assets", []):
                href = local_href(library_dir, repo_root, asset)
                if href:
                    topic_lines.append(f"  - [{asset.get('role', asset['id'])}](<{href}>) — `{asset.get('status')}`")
            topic_lines.append("")
    (library_dir / "topics.md").write_text("\n".join(topic_lines), encoding="utf-8")
    cards_html = []
    for record in sorted(catalog["records"], key=lambda item: (-int(item.get("year") or 0), item.get("id", ""))):
        bands = []
        for item in record.get("wavelengths", []):
            try:
                nm = float(item["nm"])
            except (TypeError, ValueError, KeyError):
                continue
            if nm == 532:
                bands.append("532")
            elif 500 <= nm <= 560:
                bands.append("near")
            else:
                bands.append("other")
        band_attr = " ".join(sorted(set(bands))) or "unknown"
        assets_html = []
        for asset in record.get("assets", []):
            href = local_href(library_dir, repo_root, asset)
            if href:
                label = f"本地 {asset.get('role', asset['id'])} [{asset.get('status', 'unverified')}]"
                assets_html.append(f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>')
            elif asset.get("source_url"):
                assets_html.append(f'<a href="{html.escape(asset["source_url"], quote=True)}">来源 {html.escape(asset.get("role", asset["id"]))} [{html.escape(asset.get("status", "unverified"))}]</a>')
            else:
                assets_html.append(html.escape(f"{asset.get('role', asset['id'])} [{asset.get('status', 'unverified')}]: {asset.get('status_note', '未定位')}"))
        terms = " ".join([record.get("title") or "", record.get("doi") or "", record.get("venue") or "", " ".join(record.get("topics", [])), record.get("project_use") or ""])
        wavelength_text = "; ".join(
            f"{item.get('role')}={item.get('nm')} nm" for item in record.get("wavelengths", [])
        ) or "未报告/未核"
        cards_html.append(
            f'<article data-band="{band_attr}" data-group="{html.escape(record.get("collection_group", "未分类"), quote=True)}" data-search="{html.escape(terms.casefold(), quote=True)}">'
            f'<h2>{html.escape(record.get("title", "(untitled)"))}</h2>'
            f'<p>{html.escape(record.get("venue", ""))} {html.escape(str(record.get("year", "")))} · {html.escape(record.get("evidence_class", ""))} · {html.escape(record.get("doi") or "no DOI")}</p>'
            f'<p>{html.escape(record.get("project_use", ""))}</p>'
            f'<p><strong>样品：</strong>{html.escape(json.dumps(record.get("sample_identity", {}), ensure_ascii=False))}</p>'
            f'<p><strong>波长角色：</strong>{html.escape(wavelength_text)}</p>'
            f'<p><strong>时域：</strong>{html.escape(str(record.get("temporal_conditions", "未核录")))} · '
            f'<strong>强度/能量：</strong>{html.escape(str(record.get("intensity_energy_conditions", "未核录")))}</p>'
            f'<p><strong>实测量：</strong>{html.escape(str(record.get("measured_quantities", "未核录")))} · '
            f'<strong>方法：</strong>{html.escape(str(record.get("measurement_method", "未核录")))}</p>'
            f'<p><strong>原文定位：</strong>{html.escape("; ".join(record.get("source_locations", [])) or "未核录")}</p>'
            f'<p><strong>版本：</strong>{html.escape(str(record.get("publication_version", "未核录")))} · '
            f'<strong>更正记录：</strong>{html.escape(str(record.get("correction_status", "未核录")))} · '
            f'<strong>关联数据 DOI：</strong>{html.escape("; ".join(record.get("associated_dataset_dois", [])) or "未登记")}</p>'
            f'<p><strong>限制：</strong>{html.escape(record.get("limits", ""))}</p>'
            f'<p class="assets">{" · ".join(assets_html)}</p></article>'
        )
    html_text = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LightGen 文献库</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#20242a}header{position:sticky;top:0;background:#fff;padding:1rem 0;border-bottom:1px solid #ccd}article{border:1px solid #ccd;border-radius:8px;padding:1rem;margin:1rem 0}h1,h2{line-height:1.2}input,select{font:inherit;padding:.5rem;margin:.25rem}.assets a{margin-right:.5rem}small{color:#555}</style>
<h1>LightGen 文献与公开附件库</h1><p>主数据源为 catalog.json。下载状态与参数适用性分开；出现 532 nm 仅说明该记录包含该波长角色，不能自动准入同频非线性模型。</p>
<label>搜索 <input id="search" type="search" placeholder="标题、DOI、材料或用途"></label><label>主题 <select id="group"><option value="all">全部主题</option>__GROUPS__</select></label><label>波长筛选 <select id="band"><option value="all">全部</option><option value="532">精确 532 nm</option><option value="near">500–560 nm</option><option value="other">其他波段</option><option value="unknown">未报告/未核</option></select></label>
<div id="cards">__CARDS__</div><script>const q=document.querySelector('#search'),b=document.querySelector('#band'),g=document.querySelector('#group');function f(){for(const c of document.querySelectorAll('article')){const text=c.dataset.search.includes(q.value.toLowerCase());const band=b.value==='all'||c.dataset.band.split(' ').includes(b.value);const group=g.value==='all'||c.dataset.group===g.value;c.hidden=!(text&&band&&group)}}q.addEventListener('input',f);b.addEventListener('change',f);g.addEventListener('change',f);</script></html>""".replace("__CARDS__", "\n".join(cards_html)).replace("__GROUPS__", "".join(f'<option value="{html.escape(group, quote=True)}">{html.escape(group)}</option>' for group in groups))
    (library_dir / "index.html").write_text(html_text, encoding="utf-8")
    summary = ["# LightGen 文献库", "", "本目录的唯一管理源为 `catalog.json`；CSV、BibTeX、阅读卡、缺口表和离线检索页均从该文件重建。论文及直接关联公开附件按论文身份归档；已有项目文件保留原位置并由目录链接。", "", "## 文件", "", "- `index.html`：离线检索；支持 532 nm 精确、500–560 nm 附近和其他波段筛选。", "- `index.csv` / `references.bib`：表格与引用导出。", "- `reading_cards.md`：逐篇可消费内容、样品条件、原文位置和迁移限制。", "- `download_coverage_report.md` / `coverage_report.json`：逐文件状态与覆盖统计。", "- `open_gaps.md`：受限、未定位或未核验附件。", "", "## 使用", "", "从仓库根目录执行：", "", "```powershell", "python scripts/research/lightgen/manage_lightgen_literature_v1.py verify --catalog research_assets/lightgen/literature_library/catalog.json", "python scripts/research/lightgen/manage_lightgen_literature_v1.py download --catalog research_assets/lightgen/literature_library/catalog.json", "python scripts/research/lightgen/manage_lightgen_literature_v1.py build-index --catalog research_assets/lightgen/literature_library/catalog.json", "```", "", "下载仅处理目录中显式列出的 URL，不抓取参考文献、不执行代码；受限资源不会标为已下载。D 盘保留至少 20 GiB。参数准入需单独复核材料身份、工艺、相态、波长、时域和测量量。", ""]
    summary.extend([
        "## 统一目录与本地导入", "",
        "正文和补充材料按论文放在 `papers/<论文ID>/original/`，全文 HTML/XML 放在 `fulltext/`，源数据放在 `source_data/`，代码归档放在 `code_data/`，提取文本和既有数字化结果放在 `derived/`。器件资料放在 `devices/`。同一文件可由论文记录和数据 DOI 记录共同引用一个管理副本。", "",
        "[按主题浏览](topics.md)覆盖 Science 与光学生成、Te、TaP、CoSi、光纤端面、SLM/CMOS/光源及基础光学。离线 HTML 同时支持主题筛选。", "",
        "手动下载文件先放入 `manual_inbox/`。明确核对论文和附件身份后，填写本地导入清单；不根据文件名自动猜测归属。", "",
        "已处理来稿原件保存在 `manual_inbox/processed/<批次>/`，导入清单及收据保存在 `management/<批次>/`。历史中断文件或错误网页响应保存在 `_incomplete/`，不计入有效文献文件。官方数据包可保留 `data_v3/` 等明确版本目录。", "",
        "```powershell",
        "python scripts/research/lightgen/manage_lightgen_literature_v1.py import-local --catalog research_assets/lightgen/literature_library/catalog.json --import-manifest <仓库相对清单路径> --import-receipt <仓库相对新收据路径> --dry-run",
        "# 预检通过后使用新的收据路径，移除 --dry-run 执行归档。",
        "```", "",
        "本地导入会验证格式和已登记身份，拒绝覆盖不同内容，复用相同内容，并保留原位置和导入记录。预检不更改目录状态，已收到但未解析的文件不自动获得参数准入。", "",
    ])
    summary[2] = "本目录的唯一管理源为 `catalog.json`；CSV、BibTeX、阅读卡、缺口表和离线检索页均从该文件重建。论文、补充材料、数据和代码的管理副本统一保存在本目录；旧研究包保留历史原件，导入来源在 catalog.json 的 local_imports 中登记。"
    (library_dir / "README.md").write_text("\n".join(summary), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "verify", "build-index", "all", "import-local"])
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--record-id")
    parser.add_argument("--asset-id")
    parser.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--import-manifest", type=Path)
    parser.add_argument("--import-receipt", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    catalog_path = args.catalog if args.catalog.is_absolute() else repo_root / args.catalog
    catalog_path = catalog_path.resolve()
    try:
        if args.dry_run and args.command != "import-local":
            raise ValueError("--dry-run is only supported for import-local")
        catalog = load_catalog(catalog_path)
        pairs = selected_assets(catalog, args.record_id, args.asset_id)
        if args.command == "import-local":
            if not args.import_manifest or not args.import_receipt:
                raise ValueError("import-local requires --import-manifest and --import-receipt")
            manifest_path = args.import_manifest if args.import_manifest.is_absolute() else repo_root / args.import_manifest
            receipt_path = args.import_receipt if args.import_receipt.is_absolute() else repo_root / args.import_receipt
            receipt_path = receipt_path.resolve()
            if not receipt_path.is_relative_to(catalog_path.parent) or receipt_path.exists():
                raise ValueError("import receipt must be a new file inside the literature library")
            report = run_import_local(repo_root, catalog_path, catalog, manifest_path, args.reserve_gib, args.dry_run)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"local import {'preflight' if args.dry_run else 'committed'}: {report['entry_count']} entries; receipt {receipt_path}")
        if args.command in {"download", "all"}:
            run_download(repo_root, catalog, pairs, args.reserve_gib)
        if args.command in {"verify", "all"}:
            results = run_verify(repo_root, pairs)
            valid_count = sum(bool(result.get("valid")) for result in results)
            print(f"verified {valid_count}/{len(results)} local asset files")
        if not args.dry_run:
            save_catalog(catalog_path, catalog)
        if args.command in {"build-index", "all", "import-local"} and not args.dry_run:
            build_outputs(catalog_path, repo_root, catalog)
            print(f"index rebuilt in {catalog_path.parent}")
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
