from __future__ import annotations

from pathlib import Path

from ..evaluation.d2nn import build_one_shot_d2nn_diagnostics
from ..evaluation.d2nn_visualization import (
    build_one_shot_d2nn_focus_visualization_tensors,
    build_one_shot_d2nn_visualization_tensors,
    build_one_shot_slm_to_encoder_focus_visualization_tensors,
    save_one_shot_d2nn_focus_figures,
)
from ..visualization_core import save_visualization_grids


def _save_one_shot_d2nn_visualization_grids(
    run_manager,
    sample_outputs,
    destination: str | Path,
    *,
    title_label: str,
    max_items: int,
    row_labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, Path]]:
    d2nn_visuals = build_one_shot_d2nn_visualization_tensors(sample_outputs)
    base_path = Path(destination)
    saved_paths: dict[str, dict[str, Path]] = {}
    for group_name, visualization_splits in d2nn_visuals.items():
        if not any(
            visualization_splits.get(split_name) for split_name in visualization_splits
        ):
            continue
        group_row_labels = row_labels if group_name == "layers" else None
        saved_paths[group_name] = save_visualization_grids(
            run_manager,
            visualization_splits,
            base_path.with_name(
                f"{base_path.stem}_d2nn_{group_name}{base_path.suffix}"
            ),
            title_label=f"{title_label} D2NN {group_name}",
            max_items=max_items,
            row_labels=group_row_labels,
        )
    return saved_paths


def _save_one_shot_d2nn_diagnostics(
    run_manager,
    sample_outputs,
    artifact_name: str,
    context: dict[str, object],
) -> Path | None:
    diagnostics = build_one_shot_d2nn_diagnostics(sample_outputs)
    if not diagnostics["available"]:
        return None
    return run_manager.save_artifact_json(
        artifact_name,
        {
            "context": context,
            **diagnostics,
        },
    )


def _save_one_shot_d2nn_focus_visualization(
    run_manager,
    sample_outputs,
    destination: str | Path,
    *,
    title_label: str,
    row_label: str | None = None,
) -> dict[str, Path]:
    focus_visuals = build_one_shot_d2nn_focus_visualization_tensors(sample_outputs)
    if not any(focus_visuals.get(split_name) for split_name in focus_visuals):
        return {}
    return save_one_shot_d2nn_focus_figures(
        run_manager,
        focus_visuals,
        destination,
        title_label=title_label,
        row_label=row_label,
    )


def _save_one_shot_slm_to_encoder_focus_visualization(
    run_manager,
    sample_outputs,
    destination: str | Path,
    *,
    title_label: str,
    row_label: str | None = None,
) -> dict[str, Path]:
    focus_visuals = build_one_shot_slm_to_encoder_focus_visualization_tensors(
        sample_outputs
    )
    if not any(focus_visuals.get(split_name) for split_name in focus_visuals):
        return {}
    return save_one_shot_d2nn_focus_figures(
        run_manager,
        focus_visuals,
        destination,
        title_label=title_label,
        row_label=row_label,
    )
