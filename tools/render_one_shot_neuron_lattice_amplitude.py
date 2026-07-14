from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from ldm_project.one_shot_visualization import (
    build_one_shot_neuron_lattice_amplitude_tensors,
    build_one_shot_slim_cell_chain_preview_tensors,
)
from ldm_project.run_manager import RunManager


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _backup_deprecated_integer_block_preview(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}_deprecated_integer_block_cells{path.suffix}")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    return backup


def _default_tensor_bundle(run_dir: Path) -> Path:
    sample_bundle = run_dir / "artifacts" / "one_shot_sample_tensors.pt"
    if sample_bundle.exists():
        return sample_bundle
    eval_bundle = run_dir / "artifacts" / "one_shot_distill_eval_tensors.pt"
    if eval_bundle.exists():
        return eval_bundle
    raise FileNotFoundError(f"Unable to find one-shot tensor bundle under {run_dir}")


def _default_image_name(tensor_bundle: Path) -> str:
    if tensor_bundle.name == "one_shot_sample_tensors.pt":
        return "one_shot_sample_neuron_lattice_amplitude.png"
    return "one_shot_distill_eval_neuron_lattice_amplitude.png"


def _default_artifact_name(tensor_bundle: Path) -> str:
    if tensor_bundle.name == "one_shot_sample_tensors.pt":
        return "one_shot_sample_neuron_lattice_amplitude"
    return "one_shot_distill_eval_neuron_lattice_amplitude"


def _default_chain_image_name(tensor_bundle: Path) -> str:
    if tensor_bundle.name == "one_shot_sample_tensors.pt":
        return "one_shot_sample_100x100_cell_chain_preview.png"
    return "one_shot_distill_eval_100x100_cell_chain_preview.png"


def _default_chain_artifact_name(tensor_bundle: Path) -> str:
    if tensor_bundle.name == "one_shot_sample_tensors.pt":
        return "one_shot_sample_100x100_cell_chain_preview"
    return "one_shot_distill_eval_100x100_cell_chain_preview"


def _label_rows(bundle: dict, max_items: int) -> list[str] | None:
    labels = bundle.get("class_labels")
    if labels is None:
        return None
    return [str(int(label)) for label in labels.detach().cpu().tolist()[:max_items]]


def _pixel_size_from_config(run_dir: Path) -> float | None:
    config_path = run_dir / "configs" / "train_config.json"
    if not config_path.exists():
        return None
    config = _load_json(config_path)
    d2nn = config.get("one_shot", {}).get("d2nn", {})
    pixel_size = d2nn.get("pixel_size")
    return None if pixel_size is None else float(pixel_size)


def _sample_strict_status(run_dir: Path) -> dict:
    metrics_path = run_dir / "artifacts" / "one_shot_sample_metrics.json"
    if not metrics_path.exists():
        return {"sample_strict_passed": None}
    metrics = _load_json(metrics_path)
    overall = metrics.get("overall", {})
    corr = float(overall.get("corr", 0.0))
    dice = float(overall.get("dice", 0.0))
    return {
        "sample_corr": corr,
        "sample_dice": dice,
        "sample_strict_passed": bool(corr >= 0.78 and dice >= 0.72),
        "sample_strict_thresholds": {
            "corr": 0.78,
            "dice": 0.72,
        },
    }


def _update_sample_summary(summary_path: Path, rendered: dict, strict_status: dict) -> None:
    summary = _load_json(summary_path)
    summary["human_readability_failed"] = True
    summary["sample_strict_passed"] = strict_status.get("sample_strict_passed")
    summary["accepted_summary_written"] = False
    summary["phase_only_accepted_readiness"] = False
    summary.setdefault("decision", {})
    summary["decision"].update(
        {
            "human_readability_failed": True,
            "sample_promotion": "diagnostic-only-failed-human-readability",
            "do_not_write_accepted_summary": True,
            "do_not_continue_v15_training": True,
            "next_batch_if_training": "v16_visual_readout_reconstruction",
        }
    )
    summary.setdefault("sample", {})
    summary["sample"]["neuron_lattice_amplitude"] = rendered
    if "slim_cell_chain_preview" in rendered:
        summary["sample"]["slim_cell_chain_preview"] = rendered["slim_cell_chain_preview"]
        summary["slim_cell_chain_preview_rendered"] = True
        summary["v4_style_slim_amplitude_rendered"] = True
        summary["deprecated_integer_block_preview_preserved"] = bool(
            rendered["slim_cell_chain_preview"].get("deprecated_integer_block_preview")
        )
    summary["sample"].update(strict_status)
    _write_json(summary_path, summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/20260625_v15/runs/20260629_153304_844767_train_v15_ac_roi_diagnostic_sample"),
        help="Existing one-shot sample/eval run directory to re-render in place.",
    )
    parser.add_argument("--tensor-bundle", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--sample-summary", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    tensor_bundle = args.tensor_bundle or _default_tensor_bundle(run_dir)
    bundle = torch.load(tensor_bundle, map_location="cpu", weights_only=False)
    lattice_tensors, metadata = build_one_shot_neuron_lattice_amplitude_tensors(
        bundle,
        pixel_size=_pixel_size_from_config(run_dir),
    )

    image_name = _default_image_name(tensor_bundle)
    artifact_name = _default_artifact_name(tensor_bundle)
    os.environ.setdefault("MPLCONFIGDIR", (run_dir / "artifacts" / "matplotlib_cache").as_posix())
    run_manager = RunManager(run_dir)
    image_path = run_manager.save_tensor_grid(
        lattice_tensors,
        run_manager.images_dir / image_name,
        title=f"{image_name} | 100x100 neuron-lattice amplitude",
        max_items=int(args.max_items),
        row_labels=_label_rows(bundle, int(args.max_items)),
        interpolation="nearest",
    )
    rendered = {
        **metadata,
        "source_tensor_bundle": tensor_bundle.as_posix(),
        "image": image_path.relative_to(run_dir).as_posix(),
    }
    artifact_path = run_manager.save_artifact_json(artifact_name, rendered)
    rendered["artifact"] = artifact_path.relative_to(run_dir).as_posix()

    chain_tensors, chain_metadata = build_one_shot_slim_cell_chain_preview_tensors(
        bundle,
        pixel_size=_pixel_size_from_config(run_dir),
    )
    chain_image_name = _default_chain_image_name(tensor_bundle)
    chain_artifact_name = _default_chain_artifact_name(tensor_bundle)
    chain_image_destination = run_manager.images_dir / chain_image_name
    chain_artifact_destination = run_manager.artifacts_dir / f"{chain_artifact_name}.json"
    deprecated_image_backup = _backup_deprecated_integer_block_preview(chain_image_destination)
    deprecated_artifact_backup = _backup_deprecated_integer_block_preview(chain_artifact_destination)
    chain_image_path = run_manager.save_tensor_grid(
        chain_tensors,
        chain_image_destination,
        title=f"{chain_image_name} | v4-style slim 100x100 amplitude preview",
        max_items=int(args.max_items),
        row_labels=_label_rows(bundle, int(args.max_items)),
        interpolation="nearest",
    )
    chain_rendered = {
        **chain_metadata,
        "source_tensor_bundle": tensor_bundle.as_posix(),
        "image": chain_image_path.relative_to(run_dir).as_posix(),
        "deprecated_integer_block_preview": {
            "image": None if deprecated_image_backup is None else deprecated_image_backup.relative_to(run_dir).as_posix(),
            "artifact": (
                None
                if deprecated_artifact_backup is None
                else deprecated_artifact_backup.relative_to(run_dir).as_posix()
            ),
        },
    }
    chain_artifact_path = run_manager.save_artifact_json(chain_artifact_name, chain_rendered)
    chain_rendered["artifact"] = chain_artifact_path.relative_to(run_dir).as_posix()
    rendered["slim_cell_chain_preview"] = chain_rendered

    strict_status = _sample_strict_status(run_dir)
    if args.sample_summary is not None:
        _update_sample_summary(args.sample_summary, rendered, strict_status)

    print(json.dumps({"rendered": rendered, **strict_status}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
