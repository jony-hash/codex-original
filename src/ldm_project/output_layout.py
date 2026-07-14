from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


DEFAULT_OUTPUT_BASE = Path("outputs")
DEFAULT_OUTPUT_VERSION = "unversioned"

_VERSION_RE = re.compile(r"(?<![A-Za-z0-9])v(?P<body>\d+(?:(?:[._]\d+)|(?:[a-z]\d*))*)")
_VERSION_COMPONENT_RE = re.compile(r"(?P<number>\d+)(?P<suffix>[a-z]*)(?P<suffix_number>\d*)")
_RUN_TIMESTAMP_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})")
_PREFIXED_VERSION_RE = re.compile(r"^(?P<date>\d{8})_(?P<version>.+)$")


@dataclass(frozen=True)
class OutputLayout:
    version: str
    output_root: Path
    explicit: bool


def extract_output_version(*candidates: str | Path | None) -> str:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Path):
            text = candidate.stem
        else:
            raw_text = str(candidate)
            text = Path(raw_text).stem if "/" in raw_text or "\\" in raw_text else raw_text
        match = _VERSION_RE.search(text)
        if match is not None:
            return f"v{match.group('body').replace('.', '_')}"
    return DEFAULT_OUTPUT_VERSION


def version_sort_key(version: str) -> tuple[int, tuple[tuple[int, str, int], ...]]:
    if version == DEFAULT_OUTPUT_VERSION:
        return (0, ())
    normalized = version.replace(".", "_")
    if not normalized.startswith("v"):
        return (2, ((0, normalized, 0),))
    components: list[tuple[int, str, int]] = []
    for part in normalized[1:].split("_"):
        match = _VERSION_COMPONENT_RE.fullmatch(part)
        if match is None:
            components.append((0, part, 0))
            continue
        suffix_number = match.group("suffix_number")
        components.append(
            (
                int(match.group("number")),
                match.group("suffix"),
                int(suffix_number) if suffix_number else 0,
            )
        )
    return (1, tuple(components))


def extract_run_timestamp(run_name: str) -> str | None:
    match = _RUN_TIMESTAMP_RE.match(run_name)
    if match is None:
        return None
    return f"{match.group('date')}_{match.group('time')}"


def extract_prefixed_version(name: str) -> tuple[str, str] | None:
    match = _PREFIXED_VERSION_RE.fullmatch(name)
    if match is None:
        return None
    return match.group("date"), match.group("version")


def find_existing_version_dir(
    version: str,
    *,
    output_base: str | Path = DEFAULT_OUTPUT_BASE,
) -> Path | None:
    base = Path(output_base)
    direct = base / version
    matches: list[Path] = []
    if direct.is_dir():
        matches.append(direct)
    for path in sorted(base.glob(f"*_{version}")):
        if path.is_dir():
            matches.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in matches:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    if not deduped:
        return None
    if len(deduped) > 1:
        names = ", ".join(path.name for path in deduped)
        raise RuntimeError(f"multiple managed output directories found for {version}: {names}")
    return deduped[0]


def earliest_run_timestamp(version_dir: str | Path) -> str | None:
    runs_dir = Path(version_dir) / "runs"
    if not runs_dir.is_dir():
        return None
    timestamps = sorted(
        timestamp
        for timestamp in (
            extract_run_timestamp(path.name)
            for path in runs_dir.iterdir()
            if path.is_dir()
        )
        if timestamp is not None
    )
    if not timestamps:
        return None
    return timestamps[0]


def version_directory_name(
    version: str,
    *,
    date_prefix: str,
) -> str:
    if not re.fullmatch(r"\d{8}", date_prefix):
        raise ValueError(f"date_prefix must be YYYYMMDD, got {date_prefix!r}")
    return f"{date_prefix}_{version}"


def resolve_version_dir(
    version: str,
    *,
    output_base: str | Path = DEFAULT_OUTPUT_BASE,
    now: datetime | None = None,
) -> Path:
    existing = find_existing_version_dir(version, output_base=output_base)
    if existing is not None:
        return existing
    current_date = (now or datetime.now()).strftime("%Y%m%d")
    return Path(output_base) / version_directory_name(version, date_prefix=current_date)


def versioned_output_root(version: str, *, output_base: str | Path = DEFAULT_OUTPUT_BASE) -> Path:
    return resolve_version_dir(version, output_base=output_base) / "runs"


def resolve_output_layout(
    explicit_output_root: str | Path | None,
    *,
    run_name: str | None,
    train_config: str | Path | None,
    output_base: str | Path = DEFAULT_OUTPUT_BASE,
) -> OutputLayout:
    version = extract_output_version(run_name, train_config)
    if explicit_output_root is not None:
        return OutputLayout(version=version, output_root=Path(explicit_output_root), explicit=True)
    return OutputLayout(version=version, output_root=versioned_output_root(version, output_base=output_base), explicit=False)
