"""Validate checked-in Windows environment lock integrity and project coverage."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from argparse import ArgumentParser
from importlib import metadata as importlib_metadata
from pathlib import Path
from urllib.parse import urlsplit


HASH_PATTERN = re.compile(r"--hash=sha256:([0-9a-f]{64})\Z")
PACKAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
CPU_BASELINE = {
    "torch": "2.5.1+cpu",
    "torchvision": "0.20.1+cpu",
    "pytest": "9.0.3",
    "ruff": "0.15.12",
    "setuptools": "79.0.1",
}


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_hash_requirements(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--index-url ") or line.startswith("--extra-index-url "):
            continue
        requirement = line.split()[0]
        if "==" not in requirement:
            raise ValueError(f"{path}:{line_number}: expected an exact == pin")
        name, version = requirement.split("==", 1)
        if not PACKAGE_PATTERN.fullmatch(name) or not version:
            raise ValueError(f"{path}:{line_number}: malformed package pin")
        hashes = [token for token in line.split()[1:] if token.startswith("--hash=")]
        if not hashes or any(not HASH_PATTERN.fullmatch(token) for token in hashes):
            raise ValueError(f"{path}:{line_number}: expected one or more SHA-256 hashes")
        key = canonical_name(name)
        if key in packages:
            raise ValueError(f"{path}:{line_number}: duplicate package {name}")
        packages[key] = version
    if not packages:
        raise ValueError(f"{path}: no locked packages")
    return packages


def verify_conda_lock(root: Path) -> int:
    path = root / "environment-win-explicit.lock.txt"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0] != "@EXPLICIT":
        raise ValueError(f"{path}: missing @EXPLICIT header")
    package_lines = lines[1:]
    seen: set[str] = set()
    for line_number, line in enumerate(package_lines, 2):
        parsed = urlsplit(line)
        filename = parsed.path.rsplit("/", 1)[-1]
        digest = parsed.fragment.removeprefix("sha256:")
        if parsed.scheme != "https" or not filename.endswith((".conda", ".tar.bz2")):
            raise ValueError(f"{path}:{line_number}: expected an HTTPS Conda package URL")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{path}:{line_number}: missing SHA-256 URL fragment")
        if filename in seen:
            raise ValueError(f"{path}:{line_number}: duplicate Conda archive {filename}")
        seen.add(filename)
    summary = json.loads((root / "environment-win-lock.json").read_text(encoding="utf-8"))
    integrity = summary.get("conda_lock_integrity", {})
    if integrity.get("algorithm") != "sha256" or integrity.get("package_count") != len(seen):
        raise ValueError("environment-win-lock.json does not match the explicit lock")
    if summary.get("conda_package_count") != len(seen):
        raise ValueError("environment-win-lock.json Conda package count does not match")
    return len(seen)


def verify_project_coverage(root: Path, cpu_packages: dict[str, str]) -> None:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    for raw_requirement in project.get("dependencies", []):
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$", raw_requirement)
        if not match:
            raise ValueError(f"Unsupported project dependency declaration: {raw_requirement}")
        name, specifier = match.groups()
        key = canonical_name(name)
        if key not in cpu_packages:
            raise ValueError(f"CPU lock does not cover project dependency: {name}")
        exact = re.search(r"(?:===|==)\s*([^,;\s]+)", specifier)
        if exact and cpu_packages[key] != exact.group(1):
            raise ValueError(f"CPU lock pin for {name} conflicts with pyproject.toml")


def verify_installed_pip_packages(root: Path, lock_name: str) -> int:
    packages = read_hash_requirements(root / lock_name)
    for name, expected_version in packages.items():
        try:
            actual_version = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as error:
            raise ValueError(f"locked pip package is not installed: {name}") from error
        if actual_version.strip() != expected_version:
            raise ValueError(f"installed {name}=={actual_version} does not match lock {expected_version}")
    return len(packages)


def channel_matches_locked_source(channel: str, expected_url: str) -> bool:
    parsed = urlsplit(expected_url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if parsed.scheme != "https" or not parsed.netloc or len(path_parts) < 3:
        return False
    channel_name = path_parts[0]
    subdir = path_parts[1]
    if subdir not in {
        "noarch",
        "win-64",
        "linux-64",
        "linux-aarch64",
        "linux-ppc64le",
        "linux-s390x",
        "linux-riscv64",
        "osx-64",
        "osx-arm64",
    }:
        return False
    channel_url = f"{parsed.scheme}://{parsed.netloc}/{channel_name}"
    channel_subdir_url = f"{channel_url}/{subdir}"
    return channel.rstrip("/") in {channel_name, channel_url, channel_subdir_url}


def verify_installed_conda_owned_distributions(specifications: dict[str, dict[str, str]]) -> int:
    metadata_dir = Path(sys.prefix) / "conda-meta"
    if not metadata_dir.is_dir():
        raise ValueError(f"active Python prefix has no Conda package metadata: {sys.prefix}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_dir.glob("*.json")]
    for distribution, specification in specifications.items():
        expected_version = specification.get("version")
        owner = specification.get("owner")
        if not expected_version or not owner:
            raise ValueError(f"invalid Conda-owned distribution specification: {distribution}")
        try:
            actual_version = importlib_metadata.version(distribution).strip()
        except importlib_metadata.PackageNotFoundError as error:
            raise ValueError(f"Conda-owned distribution is not installed: {distribution}") from error
        if actual_version != expected_version:
            raise ValueError(
                f"installed {distribution}=={actual_version} does not match Conda-owned version {expected_version}"
            )
        metadata_path = f"lib/site-packages/{canonical_name(distribution)}-{expected_version}.dist-info/metadata"
        owners = [
            record
            for record in records
            if canonical_name(str(record.get("name", ""))) == canonical_name(owner)
        ]
        owned_paths = {
            str(path).replace("\\", "/").lower()
            for record in owners
            for path in record.get("files", [])
        }
        if not owners or metadata_path not in owned_paths:
            raise ValueError(f"Conda owner {owner} does not own {distribution} distribution metadata")
    return len(specifications)


def verify_installed_conda_packages(root: Path, expected_count: int) -> int:
    prefix = Path(sys.prefix)
    metadata_dir = prefix / "conda-meta"
    if not metadata_dir.is_dir():
        raise ValueError(f"active Python prefix has no Conda package metadata: {prefix}")
    expected: dict[str, tuple[str, str]] = {}
    for line in (root / "environment-win-explicit.lock.txt").read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parsed = urlsplit(line.strip())
        filename = parsed.path.rsplit("/", 1)[-1]
        expected[filename] = (line.strip().split("#", 1)[0], parsed.fragment.removeprefix("sha256:"))
    installed: dict[str, dict[str, object]] = {}
    for path in metadata_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        filename = record.get("fn")
        if filename:
            installed[filename] = record
    if set(installed) != set(expected) or len(expected) != expected_count:
        missing = sorted(set(expected) - set(installed))
        extra = sorted(set(installed) - set(expected))
        raise ValueError(f"installed Conda package set differs from lock (missing={missing}, extra={extra})")
    for filename, (expected_url, expected_sha256) in expected.items():
        record = installed[filename]
        if record.get("url") != expected_url or record.get("sha256") != expected_sha256:
            raise ValueError(f"installed Conda artifact metadata differs from lock: {filename}")
        name, version, build = str(record.get("name")), str(record.get("version")), str(record.get("build"))
        stem = filename.removesuffix(".conda").removesuffix(".tar.bz2")
        if stem != f"{name}-{version}-{build}":
            raise ValueError(f"installed Conda version/build differs from locked filename: {filename}")
        channel = record.get("channel")
        if not isinstance(channel, str) or not channel_matches_locked_source(channel, expected_url):
            raise ValueError(f"installed Conda channel differs from locked URL: {filename}")
    return len(installed)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--verify-installed-cpu-lock", action="store_true")
    parser.add_argument("--verify-installed-windows-pip-lock", action="store_true")
    parser.add_argument("--verify-installed-conda-lock", action="store_true")
    args = parser.parse_args()
    try:
        conda_count = verify_conda_lock(root)
        pip_packages = read_hash_requirements(root / "requirements-win-pip.lock.txt")
        cpu_packages = read_hash_requirements(root / "requirements-win-cpu.lock.txt")
        summary = json.loads((root / "environment-win-lock.json").read_text(encoding="utf-8"))
        pip_integrity = summary.get("pip_lock_integrity", {})
        if pip_integrity.get("algorithm") != "sha256" or pip_integrity.get("package_count") != len(pip_packages):
            raise ValueError("environment-win-lock.json does not match the pip-only Windows lock")
        for name, version in CPU_BASELINE.items():
            if cpu_packages.get(name) != version:
                raise ValueError(f"CPU lock must pin {name}=={version}")
        verify_project_coverage(root, cpu_packages)
        installed_cpu = (
            verify_installed_pip_packages(root, "requirements-win-cpu.lock.txt")
            if args.verify_installed_cpu_lock
            else None
        )
        installed_windows_pip = (
            verify_installed_pip_packages(root, "requirements-win-pip.lock.txt")
            if args.verify_installed_windows_pip_lock
            else None
        )
        installed_conda = verify_installed_conda_packages(root, conda_count) if args.verify_installed_conda_lock else None
        owned_distributions = summary.get("conda_owned_distributions", {})
        if not isinstance(owned_distributions, dict) or not owned_distributions:
            raise ValueError("environment-win-lock.json must declare Conda-owned Python distributions")
        installed_conda_owned = (
            verify_installed_conda_owned_distributions(owned_distributions)
            if args.verify_installed_windows_pip_lock
            else None
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        print(f"environment lock verification failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "conda_packages": conda_count,
                "windows_pip_packages": len(pip_packages),
                "windows_cpu_packages": len(cpu_packages),
                "project_dependencies_covered": True,
                "installed_cpu_packages_verified": installed_cpu,
                "installed_windows_pip_packages_verified": installed_windows_pip,
                "installed_conda_owned_distributions_verified": installed_conda_owned,
                "installed_conda_packages_verified": installed_conda,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
