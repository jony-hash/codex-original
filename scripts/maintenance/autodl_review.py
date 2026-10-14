"""Build a small, explicit AutoDL review bundle and three compact figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


class ReviewError(RuntimeError):
    """The requested evidence package or plot cannot be completed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def _read_json(path: Path) -> Any:
    _require(path.is_file(), f"required evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"required table is missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _repo_relative_source(path_value: str) -> str:
    windows = PureWindowsPath(path_value)
    if windows.is_absolute():
        _require(str(windows).casefold().startswith("d:\\codex-original\\"), "source checkpoint is outside the named Windows repository")
        relative = Path(*windows.parts[2:]).as_posix()
    elif PurePosixPath(path_value).is_absolute():
        posix = PurePosixPath(path_value)
        _require(len(posix.parts) >= 5 and posix.parts[:4] == ("/", "root", "autodl-tmp", "codex-original"), "source checkpoint is outside the named AutoDL repository")
        relative = PurePosixPath(*posix.parts[4:]).as_posix()
    else:
        path = Path(path_value)
        if path.is_absolute():
            parts = path.parts
            _require(len(parts) >= 5 and parts[:4] == ("/", "root", "autodl-tmp", "codex-original"), "source checkpoint is outside the named AutoDL repository")
            relative = Path(*parts[4:]).as_posix()
        else:
            relative = path.as_posix()
    _validate_relative(relative)
    return relative


def _validate_relative(value: str) -> None:
    path = Path(value)
    _require(value and not path.is_absolute() and "\\" not in value, f"manifest path must be a POSIX relative path: {value}")
    _require(all(part not in {"", ".", ".."} for part in value.split("/")), f"manifest path traversal is not allowed: {value}")
    _require(all(part and all(character.isalnum() or character in "._-" for character in part) for part in value.split("/")), f"manifest path contains unsupported characters: {value}")


def _scenario_rows(rows: Sequence[Mapping[str, str]], scenario: str) -> list[Mapping[str, str]]:
    return [row for row in rows if row.get("scenario") == scenario]


def _focus_qfg(rows: Sequence[Mapping[str, str]], scenario: str) -> float:
    selected = [float(row["q_fg"]) for row in _scenario_rows(rows, scenario) if int(row["label"]) in {2, 4}]
    _require(len(selected) == 200, f"{scenario} must contain 100 DEV rows from each focus class")
    return sum(selected) / len(selected)


def _history_files(evidence: Path, seeds: Sequence[int]) -> list[Path]:
    paths = [evidence / "training_history" / f"seed_{seed}_arm_{arm.lower()}.json" for seed in seeds for arm in ("A", "B")]
    missing = [path for path in paths if not path.is_file()]
    _require(not missing, "training history is incomplete: " + ", ".join(path.relative_to(evidence).as_posix() for path in missing))
    return paths


def _plot_evidence(evidence: Path, destination: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = _read_json(evidence / "report.json")
    paired = _read_csv(evidence / "paired_rows.csv")
    class_rows = _read_csv(evidence / "dev_class_summary.csv")
    seeds = [int(seed) for seed in report["training_seeds"]]
    histories = _history_files(evidence, seeds)
    destination.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7})

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.0), sharex=True, constrained_layout=True)
    for path in histories:
        payload = _read_json(path)
        records = payload.get("records", [])
        _require(len(records) == int(report["updates_per_arm_per_seed"]), f"{path.name} does not contain the declared training history")
        label = f"{payload['arm']} / {payload['seed']}"
        updates = [int(record["task_local_update"]) for record in records]
        axes[0].plot(updates, [float(record["q_fg"]) for record in records], linewidth=0.8, alpha=0.85, label=label)
        axes[1].plot(updates, [float(record["raw_mse"]) for record in records], linewidth=0.8, alpha=0.85, label=label)
    axes[0].set_ylabel("Q_fg")
    axes[1].set_ylabel("Raw MSE")
    axes[1].set_xlabel("Task-local update")
    axes[0].set_title("Training telemetry: Q_fg and reconstruction")
    axes[0].legend(ncol=3, frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    line_path = destination / "training_telemetry.png"
    fig.savefig(line_path, dpi=120, metadata={"Title": "Training telemetry"})
    plt.close(fig)

    decision = report["decision"]["seed_decisions"]
    q0 = float(report["decision"]["shared_start_mean_q_fg"])
    labels = [str(seed) for seed in seeds]
    x = list(range(len(seeds)))
    width = 0.24
    fig, axis = plt.subplots(figsize=(7.4, 3.5), constrained_layout=True)
    axis.bar([value - width for value in x], [float(decision[str(seed)]["A"]) for seed in seeds], width, label="A")
    axis.bar(x, [float(decision[str(seed)]["B"]) for seed in seeds], width, label="B")
    axis.bar([value + width for value in x], [float(decision[str(seed)]["Q0"]) for seed in seeds], width, label="Common start Q0")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Focus-class mean Q_fg")
    axis.set_title("DEV endpoint comparison used by the fixed decision")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    endpoint_path = destination / "dev_endpoint_qfg.png"
    fig.savefig(endpoint_path, dpi=120, metadata={"Title": "DEV endpoint Qfg"})
    plt.close(fig)

    scenario_order = ["A4700", *[f"seed_{seed}_{arm}_step{int(report['source_checkpoint_step']) + int(report['updates_per_arm_per_seed'])}" for seed in seeds for arm in ("A", "B")]]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True, constrained_layout=True)
    classes = list(range(10))
    bar_width = 0.11
    total_width = bar_width * len(scenario_order)
    for scenario_index, scenario in enumerate(scenario_order):
        selected = [row for row in class_rows if row.get("scenario") == scenario]
        _require(len(selected) == 10, f"class summary lacks all ten classes for {scenario}")
        selected.sort(key=lambda row: int(row["label"]))
        offset = scenario_index * bar_width - total_width / 2 + bar_width / 2
        axes[0].bar([value + offset for value in classes], [float(row["mean_q_fg"]) for row in selected], bar_width, label=scenario)
        axes[1].bar([value + offset for value in classes], [float(row["mean_raw_cmos_integral"]) * 1e8 for row in selected], bar_width, label=scenario)
    axes[0].set_ylabel("Mean Q_fg")
    axes[1].set_ylabel("Raw intensity integral × 1e8")
    axes[1].set_xlabel("DEV class")
    axes[0].set_xticks(classes)
    axes[0].set_title("Ten-class DEV foreground error and output intensity")
    axes[0].legend(ncol=2, frameon=False, fontsize=6)
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    class_path = destination / "dev_class_qfg_intensity.png"
    fig.savefig(class_path, dpi=120, metadata={"Title": "Ten-class DEV metrics"})
    plt.close(fig)

    computed = {scenario: _focus_qfg(paired, scenario) for scenario in scenario_order}
    _require(math.isclose(computed["A4700"], q0, rel_tol=1e-10, abs_tol=1e-12), "plot source CSV does not reproduce the saved common-start Q_fg")
    for seed in seeds:
        for arm in ("A", "B"):
            key = f"seed_{seed}_{arm}_step{int(report['source_checkpoint_step']) + int(report['updates_per_arm_per_seed'])}"
            expected = float(decision[str(seed)][arm])
            _require(math.isclose(computed[key], expected, rel_tol=1e-10, abs_tol=1e-12), f"plot source CSV does not reproduce saved decision for {key}")
    outputs = [line_path, endpoint_path, class_path]
    total_bytes = sum(path.stat().st_size for path in outputs)
    _require(total_bytes <= 2 * 1024 * 1024, "the three PNG files exceed the 2 MiB delivery limit")
    validation = {
        "status": "VALIDATED",
        "sources": ["paired_rows.csv", "dev_class_summary.csv", "training_history/*.json", "report.json"],
        "focus_qfg_recomputed_from_saved_rows": computed,
        "shared_start_qfg": q0,
        "png_files": [{"path": path.name, "size_bytes": path.stat().st_size} for path in outputs],
        "total_png_bytes": total_bytes,
        "resource_telemetry_curve_created": False,
    }
    (destination / "plot_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def _review_files(evidence: Path, profile: str) -> tuple[list[Path], list[str]]:
    report = _read_json(evidence / "report.json")
    seeds = [int(seed) for seed in report["training_seeds"]]
    required_names = [
        "report.json", "validation.json", "completed.json", "closeout.md", "fixed_parameters.json",
        "input_population.json", "batch_orders.json", "execution_runtime.json", "preflight.json",
        "preflight_trials.json", "progress.json", "consolidation.json", "paired_rows.csv",
        "dev_class_summary.csv", "conditional_sample_rows.csv", "conditional_sample_class_summary.csv",
        "conditional_sample_summary.json", "conditional_sample_bank.pt", "fitted_joint_prior.pt",
    ]
    paths = [evidence / name for name in required_names]
    paths.extend(_history_files(evidence, seeds))
    source_step = int(report["source_checkpoint_step"])
    terminal_step = source_step + int(report["updates_per_arm_per_seed"])
    paths.extend(evidence / "checkpoints" / f"seed_{seed}_arm_{arm.lower()}_step{terminal_step:04d}.pt" for seed in seeds for arm in ("A", "B"))
    paths.extend(sorted(evidence.glob("engineering_failure_*.json")))
    missing = [path for path in paths if not path.is_file()]
    _require(not missing, "review bundle is incomplete: " + ", ".join(path.relative_to(evidence).as_posix() for path in missing))
    if profile == "audit":
        raw_readouts = sorted(path for path in (evidence / "readouts").rglob("*.pt") if path.is_file()) if (evidence / "readouts").is_dir() else []
        _require(bool(raw_readouts), "audit profile requires saved raw readout chunks")
        paths.extend(raw_readouts)
        paths.extend(path for path in (evidence / "ols_fit_coefficients.pt", evidence / "target_scale_coefficients.json") if path.is_file())
    input_population = _read_json(evidence / "input_population.json")
    local_dependencies = [_repo_relative_source(str(input_population["source_checkpoint"]))]
    unique = {path.relative_to(evidence).as_posix(): path for path in paths}
    return [unique[key] for key in sorted(unique)], local_dependencies


def _package(evidence: Path, output: Path, profile: str) -> dict[str, Any]:
    _require(evidence.is_dir(), f"evidence directory does not exist: {evidence}")
    _require(not output.exists(), f"package output must be a new directory: {output}")
    bundle = output / "review_bundle"
    bundle.mkdir(parents=True, exist_ok=False)
    selected, dependencies = _review_files(evidence, profile)
    for source in selected:
        relative = source.relative_to(evidence)
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    plot_validation = _plot_evidence(evidence, bundle / "figures")
    relative_files = [*(f"review_bundle/{path.relative_to(evidence).as_posix()}" for path in selected)]
    relative_files.extend(f"review_bundle/figures/{name}" for name in ("training_telemetry.png", "dev_endpoint_qfg.png", "dev_class_qfg_intensity.png", "plot_validation.json"))
    relative_files.append("review_manifest.json")
    manifest = {
        "schema": "autodl_review_manifest_v1",
        "profile": profile,
        "files": relative_files,
        "required_local_dependencies": dependencies,
        "limitations": ["review profile omits raw pixel readout chunks and regenerable caches" ] if profile == "review" else [],
    }
    manifest_path = output / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total_bytes = sum((output / relative).stat().st_size for relative in relative_files)
    package = {
        "status": "PACKAGED",
        "profile": profile,
        "source_evidence_dir": evidence.as_posix(),
        "output_dir": output.as_posix(),
        "file_count": len(relative_files),
        "package_bytes_including_manifest": total_bytes,
        "reference_bytes": 153 * 1024 * 1024,
        "within_reference_size": total_bytes <= 153 * 1024 * 1024,
        "raw_pixel_audit_complete_in_bundle": profile == "audit",
        "review_png_validation": plot_validation,
        "required_local_dependencies": dependencies,
        "manifest": manifest_path.name,
    }
    (output / "package_report.json").write_text(json.dumps(package, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return package


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("package", "plot"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--evidence-dir", required=True, type=Path)
        subparser.add_argument("--output-dir", required=True, type=Path)
        if command == "package":
            subparser.add_argument("--profile", required=True, choices=("review", "audit"))
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or _arguments()
    evidence = args.evidence_dir.resolve()
    output = args.output_dir.resolve()
    if args.command == "package":
        return _package(evidence, output, args.profile)
    _require(evidence.is_dir(), f"evidence directory does not exist: {evidence}")
    _require(not output.exists(), f"plot output must be a new directory: {output}")
    return _plot_evidence(evidence, output)


if __name__ == "__main__":
    try:
        print(json.dumps(main(), ensure_ascii=False, indent=2, sort_keys=True))
    except ReviewError as error:
        raise SystemExit(str(error)) from error
