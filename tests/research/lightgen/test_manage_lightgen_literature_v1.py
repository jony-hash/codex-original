from __future__ import annotations

import importlib.util
import hashlib
import json
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "research" / "lightgen" / "manage_lightgen_literature_v1.py"
SPEC = importlib.util.spec_from_file_location("manage_lightgen_literature_v1", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


def test_rejects_html_login_page_masquerading_as_pdf(tmp_path: Path) -> None:
    payload = tmp_path / "paper.pdf"
    payload.write_text("<!doctype html><html><title>Sign in</title>Login to continue</html>", encoding="utf-8")
    result = manager.validate_asset({"format": "pdf", "expected_title": "A real paper"}, payload)
    assert not result["valid"]
    assert "signature missing" in result["reason"]


def test_html_page_ignores_login_terms_inside_script_and_template(tmp_path: Path) -> None:
    payload = tmp_path / "device.html"
    payload.write_text(
        "<!doctype html><html><head><title>Amplitude SLMs</title>"
        "<script>const prompt = 'Sign in to continue';</script></head>"
        "<body><h1>Amplitude SLMs</h1><p>Spatial light modulator product information.</p>"
        "<template><p>Access denied</p></template></body></html>",
        encoding="utf-8",
    )
    result = manager.validate_asset(
        {"format": "html", "expected_title": "Amplitude SLM", "expected_doi": ""},
        payload,
    )
    assert result["valid"]


def test_rejects_visible_html_access_denied_page(tmp_path: Path) -> None:
    payload = tmp_path / "blocked.html"
    payload.write_text(
        "<!doctype html><html><head><title>Blocked</title></head>"
        "<body><h1>Access denied</h1><p>Request blocked.</p></body></html>",
        encoding="utf-8",
    )
    result = manager.validate_asset({"format": "html"}, payload)
    assert not result["valid"]
    assert "access/login/error" in result["reason"]


def test_rejects_xlsx_without_populated_cells(tmp_path: Path) -> None:
    payload = tmp_path / "empty.xlsx"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr("xl/worksheets/sheet1.xml", "<worksheet><sheetData/></worksheet>")
    result = manager.validate_asset({"format": "xlsx"}, payload)
    assert not result["valid"]
    assert "no populated worksheet cells" in result["reason"]


def test_validates_xlsx_has_real_cell_data(tmp_path: Path) -> None:
    payload = tmp_path / "data.xlsx"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheetData><row r=\"1\"><c r=\"A1\" t=\"n\"><v>532</v></c></row></sheetData></worksheet>",
        )
    result = manager.validate_asset({"format": "xlsx"}, payload)
    assert result["valid"]
    assert result["worksheet_count"] == 1
    assert result["populated_cells"] == 1


def test_zip_requires_pinned_repository_prefix(tmp_path: Path) -> None:
    payload = tmp_path / "repo.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("wrong-revision/README.md", "snapshot")
    result = manager.validate_asset(
        {"format": "zip", "expected_archive_prefix": "T-Scope-8b99850796a31855d76fe05553661df93fd0eef3/"},
        payload,
    )
    assert not result["valid"]
    assert "expected archive prefix" in result["reason"]


def test_rejects_xml_with_wrong_doi_even_if_title_is_similar(tmp_path: Path) -> None:
    payload = tmp_path / "article.xml"
    payload.write_text(
        "<article><front><article-meta><article-title>Tellurium nonlinear optics study</article-title>"
        "<article-id>10.1038/wrong-doi</article-id></article-meta></front></article>",
        encoding="utf-8",
    )
    result = manager.validate_asset(
        {"format": "xml", "expected_doi": "10.1038/right-doi", "expected_title": "Tellurium nonlinear optics study", "title_match_threshold": 0.9},
        payload,
    )
    assert not result["valid"]
    assert "DOI mismatch" in result["reason"]


def test_pdf_identity_uses_readable_page_text(tmp_path: Path) -> None:
    import pymupdf as fitz

    payload = tmp_path / "paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A Readable Test Paper DOI 10.1000/readable-paper")
    document.save(payload)
    document.close()
    result = manager.validate_asset(
        {"format": "pdf", "expected_title": "A Readable Test Paper", "expected_doi": "10.1000/readable-paper"},
        payload,
    )
    assert result["valid"]
    assert result["page_count"] == 1


def test_download_path_reuses_valid_file_without_overwriting(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    destination = repo_root / "assets" / "source.zip"
    destination.parent.mkdir(parents=True)
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("source.csv", "x,y\n1,2\n")
    original = destination.read_bytes()
    asset = {
        "id": "source",
        "role": "source ZIP",
        "format": "zip",
        "source_url": "https://invalid.example/should-not-be-requested.zip",
        "repo_path": "assets/source.zip",
        "acquisition": "download",
        "estimated_bytes": 10,
    }
    result = manager.download_one(repo_root, asset, reserve_bytes=0)
    assert result["valid"]
    assert asset["status"] == "verified_local_reused"
    assert destination.read_bytes() == original


def test_download_path_commits_valid_staged_file_after_manifest_fix(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    destination = repo_root / "assets" / "source.zip"
    destination.parent.mkdir(parents=True)
    staged = destination.with_name(destination.name + ".part")
    with zipfile.ZipFile(staged, "w") as archive:
        archive.writestr("source.csv", "x,y\n1,2\n")
    asset = {
        "id": "source",
        "role": "source ZIP",
        "format": "zip",
        "source_url": "https://invalid.example/should-not-be-requested.zip",
        "repo_path": "assets/source.zip",
        "acquisition": "download",
        "estimated_bytes": 10,
    }
    result = manager.download_one(repo_root, asset, reserve_bytes=0)
    assert result["valid"]
    assert asset["status"] == "downloaded_verified_from_staged_file"
    assert destination.is_file()
    assert not staged.exists()


def test_download_marks_short_content_length_as_incomplete_and_preserves_part(tmp_path: Path, monkeypatch) -> None:
    class ShortResponse:
        headers = {"Content-Length": "10"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size: int) -> bytes:
            if not hasattr(self, "sent"):
                self.sent = True
                return b"short"
            return b""

    monkeypatch.setattr(manager.urllib.request, "urlopen", lambda *_args, **_kwargs: ShortResponse())
    repo_root = tmp_path / "repo"
    destination = repo_root / "assets" / "truncated.txt"
    asset = {
        "id": "truncated",
        "role": "text file",
        "format": "txt",
        "source_url": "https://example.invalid/truncated.txt",
        "repo_path": "assets/truncated.txt",
        "acquisition": "download",
        "estimated_bytes": 10,
    }
    result = manager.download_one(repo_root, asset, reserve_bytes=0)
    staged = destination.with_name(destination.name + ".part")
    assert not result["valid"]
    assert asset["status"] == "download_incomplete"
    assert "Content-Length was 10" in result["reason"]
    assert staged.read_bytes() == b"short"


def test_download_resumes_only_confirmed_range_and_validates_full_archive(tmp_path: Path, monkeypatch) -> None:
    from io import BytesIO

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("T-Scope-pinned/README.md", "snapshot")
    complete = buffer.getvalue()
    offset = len(complete) // 2

    class RangeResponse:
        status = 206

        def __init__(self) -> None:
            self.headers = {
                "Content-Range": f"bytes {offset}-{len(complete) - 1}/{len(complete)}",
                "Content-Length": str(len(complete) - offset),
            }
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size: int) -> bytes:
            if self.sent:
                return b""
            self.sent = True
            return complete[offset:]

    def fake_urlopen(request, **_kwargs):
        assert request.get_header("Range") == f"bytes={offset}-"
        return RangeResponse()

    monkeypatch.setattr(manager.urllib.request, "urlopen", fake_urlopen)
    repo_root = tmp_path / "repo"
    destination = repo_root / "assets" / "source.zip"
    destination.parent.mkdir(parents=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(complete[:offset])
    asset = {
        "id": "source",
        "role": "pinned source archive",
        "format": "zip",
        "source_url": "https://example.invalid/pinned.zip",
        "repo_path": "assets/source.zip",
        "acquisition": "download",
        "expected_md5": hashlib.md5(complete).hexdigest(),
        "expected_archive_prefix": "T-Scope-pinned/",
        "status": "download_incomplete",
    }

    result = manager.download_one(repo_root, asset, reserve_bytes=0)

    assert result["valid"]
    assert asset["status"] == "downloaded_verified_from_resumed_part"
    assert result["expected_archive_prefix_verified"] == "T-Scope-pinned/"
    assert destination.read_bytes() == complete
    assert partial.read_bytes() == complete[:offset]


def test_coverage_counts_verified_resumed_download(tmp_path: Path) -> None:
    library_dir = tmp_path / "research_assets" / "lightgen" / "literature_library"
    library_dir.mkdir(parents=True)
    catalog_path = library_dir / "catalog.json"
    catalog = {
        "records": [
            {
                "id": "resumed_dataset",
                "record_type": "dataset",
                "title": "Resumed dataset",
                "assets": [
                    {
                        "id": "release_zip",
                        "role": "release ZIP",
                        "status": "downloaded_verified_from_resumed_part",
                    }
                ],
            }
        ]
    }

    manager.build_outputs(catalog_path, tmp_path, catalog)

    report = json.loads((library_dir / "coverage_report.json").read_text(encoding="utf-8"))
    gaps = (library_dir / "open_gaps.md").read_text(encoding="utf-8")
    assert report["records"][0]["verified_asset_count"] == 1
    assert "release_zip" not in gaps
    assert "`downloaded_verified_from_resumed_part`" in gaps


def test_catalog_rejects_duplicate_doi_in_same_record_class(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {"id": "one", "record_type": "paper", "doi": "10.1/example", "assets": []},
                    {"id": "two", "record_type": "paper", "doi": "10.1/example", "assets": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        manager.load_catalog(path)
    except ValueError as exc:
        assert "duplicate DOI" in str(exc)
    else:
        raise AssertionError("duplicate DOI should be rejected")


def test_selected_article_assets_inherit_paper_identity() -> None:
    catalog = {
        "records": [
            {
                "id": "paper",
                "record_type": "paper",
                "doi": "10.1000/example",
                "title": "A Test Paper",
                "assets": [{"id": "pdf", "role": "article PDF", "format": "pdf"}],
            }
        ]
    }
    _, asset = manager.selected_assets(catalog, None, None)[0]
    assert asset["expected_doi"] == "10.1000/example"
    assert asset["expected_title"] == "A Test Paper"


@pytest.fixture
def local_import_case(tmp_path: Path):
    library = tmp_path / "library"
    library.mkdir()
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "source.txt").write_text("Known study parameters", encoding="utf-8")
    catalog_path = library / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "records": [{
        "id": "paper", "record_type": "paper", "title": "Known study",
        "assets": [{"id": "source", "format": "txt", "expected_title": "Known study", "status": "blocked_http_403"}],
    }]}), encoding="utf-8")
    manifest_path = library / "imports.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "entries": [{
        "record_id": "paper", "asset_id": "source", "source_repo_path": "inbox/source.txt",
        "destination_repo_path": "library/papers/paper/source.txt",
    }]}), encoding="utf-8")
    return {
        "root": tmp_path, "library": library, "catalog": catalog_path, "manifest": manifest_path,
        "destination": library / "papers/paper/source.txt",
        "args": ["import-local", "--repo-root", str(tmp_path), "--catalog", str(catalog_path),
                 "--import-manifest", str(manifest_path), "--reserve-gib", "0"],
    }


def test_local_import_cli_preflight_commit_and_idempotent_reuse(local_import_case) -> None:
    case = local_import_case
    original = case["catalog"].read_bytes()
    assert manager.main([*case["args"], "--import-receipt", str(case["library"] / "preflight.json"), "--dry-run"]) == 0
    assert case["catalog"].read_bytes() == original
    assert not case["destination"].exists()
    assert manager.main([*case["args"], "--import-receipt", str(case["library"] / "committed.json")]) == 0
    assert case["destination"].read_text(encoding="utf-8") == "Known study parameters"
    assert manager.main([*case["args"], "--import-receipt", str(case["library"] / "reused.json")]) == 0
    catalog = json.loads(case["catalog"].read_text(encoding="utf-8"))
    asset = catalog["records"][0]["assets"][0]
    assert asset["status"] == "imported_local_verified"
    assert len(asset["local_imports"]) == 1
    assert asset["local_imports"][0]["previous_status"] == "blocked_http_403"
    coverage = json.loads((case["library"] / "coverage_report.json").read_text(encoding="utf-8"))
    assert coverage["records"][0]["verified_asset_count"] == 1
    assert "papers/paper/source.txt" in (case["library"] / "index.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("fmt,payload", [("pdf", "<html>Known study Login</html>"), ("txt", "Unrelated article content")])
def test_local_import_rejects_invalid_later_item_before_any_commit(local_import_case, fmt, payload) -> None:
    case = local_import_case
    catalog = json.loads(case["catalog"].read_text(encoding="utf-8"))
    catalog["records"][0]["assets"].append({"id": "bad", "format": fmt, "expected_title": "Known study"})
    case["catalog"].write_text(json.dumps(catalog), encoding="utf-8")
    original = case["catalog"].read_bytes()
    (case["root"] / "inbox/bad").write_text(payload, encoding="utf-8")
    manifest = json.loads(case["manifest"].read_text(encoding="utf-8"))
    manifest["entries"].append({"record_id": "paper", "asset_id": "bad", "source_repo_path": "inbox/bad",
                                "destination_repo_path": "library/papers/paper/bad"})
    case["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    assert manager.main([*case["args"], "--import-receipt", str(case["library"] / "bad.json")]) == 2
    assert not case["destination"].exists()
    assert case["catalog"].read_bytes() == original


def test_local_import_refuses_to_replace_different_destination(local_import_case) -> None:
    case = local_import_case
    case["destination"].parent.mkdir(parents=True)
    case["destination"].write_bytes(b"existing different document")
    original = case["catalog"].read_bytes()
    assert manager.main([*case["args"], "--import-receipt", str(case["library"] / "conflict.json")]) == 2
    assert case["destination"].read_bytes() == b"existing different document"
    assert case["catalog"].read_bytes() == original


@pytest.mark.parametrize("member,expected_code", [("dataset/source.txt", 0), ("../source.txt", 2), (0, 2)])
def test_local_import_extracts_explicit_zip_member_and_rejects_traversal(local_import_case, member, expected_code) -> None:
    case = local_import_case
    with zipfile.ZipFile(case["root"] / "inbox/dataset.zip", "w") as archive:
        archive.writestr(member if isinstance(member, str) else "dataset/source.txt", "Known study parameters")
    manifest = json.loads(case["manifest"].read_text(encoding="utf-8"))
    manifest["entries"][0].update(source_repo_path="inbox/dataset.zip", archive_member=member)
    case["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    assert manager.main([*case["args"], "--import-receipt", str(case["library"] / "zip.json")]) == expected_code
    assert case["destination"].exists() == (expected_code == 0)


@pytest.mark.parametrize("destination", ["outside_library.txt", "library"])
def test_local_import_destination_is_confined_to_library(local_import_case, destination) -> None:
    case = local_import_case
    manifest = json.loads(case["manifest"].read_text(encoding="utf-8"))
    manifest["entries"][0]["destination_repo_path"] = destination
    case["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    assert manager.main([*case["args"], "--import-receipt", str(case["library"] / "outside.json")]) == 2
    assert not (case["root"] / "outside_library.txt").exists()
