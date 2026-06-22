from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import textwrap
from typing import Any, Mapping


def _json_ready(value: Any) -> Any:
    # is_dataclass 会对“dataclass 类本身”也返回 True；asdict 仅接受实例。
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    # 先写临时文件再替换，避免训练中断时留下半截 JSON。
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, ensure_ascii=False)
    temp_path.replace(path)


class RunManager:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.checkpoints_dir = run_dir / "checkpoints"
        self.images_dir = run_dir / "images"
        self.samples_dir = self.images_dir / "samples"
        self.first_stage_images_dir = self.images_dir / "first_stage"
        self.diffusion_images_dir = self.images_dir / "diffusion"
        self.configs_dir = run_dir / "configs"
        self.artifacts_dir = run_dir / "artifacts"
        self.meta_path = run_dir / "run_meta.json"
        self.history_path = run_dir / "training_history.json"
        self.history: dict[str, list[dict[str, Any]]] = {}

        for path in (
            self.run_dir,
            self.checkpoints_dir,
            self.images_dir,
            self.samples_dir,
            self.first_stage_images_dir,
            self.diffusion_images_dir,
            self.configs_dir,
            self.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        output_root: str | Path = "outputs/runs",
        purpose: str = "train",
        run_name: str | None = None,
    ) -> "RunManager":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = f"_{run_name}" if run_name else ""
        run_dir = Path(output_root) / f"{timestamp}_{purpose}{suffix}"
        return cls(run_dir)

    def write_meta(self, meta: Mapping[str, Any]) -> None:
        _write_json_atomic(self.meta_path, dict(meta))

    def append_history(self, section: str, record: Mapping[str, Any]) -> None:
        self.history.setdefault(section, []).append(_json_ready(dict(record)))
        _write_json_atomic(self.history_path, self.history)

    def save_config_snapshot(self, name: str, data: Any) -> Path:
        path = self.configs_dir / f"{name}.json"
        _write_json_atomic(path, data)
        return path

    def save_checkpoint(self, name: str, payload: Mapping[str, Any]) -> Path:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to save checkpoints.") from exc

        path = self.checkpoints_dir / name
        torch.save(dict(payload), path)
        return path

    def save_tensor_bundle(self, name: str, payload: Mapping[str, Any]) -> Path:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to save tensor artifacts.") from exc

        path = self.artifacts_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(payload), path)
        return path

    def save_artifact_json(self, name: str, payload: Mapping[str, Any] | list[Any]) -> Path:
        path = self.artifacts_dir / f"{name}.json"
        _write_json_atomic(path, payload)
        return path

    def save_tensor_grid(
        self,
        tensor_map: Mapping[str, Any],
        destination: str | Path,
        title: str = "",
        max_items: int = 4,
        row_labels: list[str] | tuple[str, ...] | None = None,
        cmap: str = "gray",
        phase_colorbar: bool = False,
    ) -> Path:
        path = self.run_dir / destination if not isinstance(destination, Path) else destination
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
        except ImportError:
            relative = path.with_suffix(".pt").relative_to(self.run_dir)
            return self.save_tensor_bundle(str(relative), tensor_map)

        prepared: list[tuple[str, list[Any]]] = []
        rows = 0
        for name, value in tensor_map.items():
            images = self._tensor_to_images(value, max_items=max_items)
            if not images:
                continue
            rows = max(rows, len(images))
            prepared.append((name, images))

        if not prepared or rows == 0:
            raise ValueError("No visualizable tensors were provided.")

        cell_width_inches = 1.55
        cell_height_inches = 1.05
        figure, axes = plt.subplots(
            rows,
            len(prepared),
            squeeze=False,
            figsize=(cell_width_inches * len(prepared), cell_height_inches * rows + 0.8),
            constrained_layout=True,
        )
        display_cmap = cmap
        if phase_colorbar:
            import matplotlib.pyplot as plt
            cmap_obj = plt.get_cmap(cmap).copy()
            cmap_obj.set_bad("white", 0.0)
            display_cmap = cmap_obj
        label_values = list(row_labels or [])[:rows]
        for col, (name, images) in enumerate(prepared):
            column_title = textwrap.fill(
                name.replace("_", " "),
                width=22,
                break_long_words=False,
                break_on_hyphens=False,
            )
            axes[0][col].set_title(column_title, fontsize=8, pad=4)
            for row in range(rows):
                axis = axes[row][col]
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                if col == 0 and row < len(label_values):
                    axis.text(
                        -0.14,
                        0.5,
                        label_values[row],
                        transform=axis.transAxes,
                        ha="right",
                        va="center",
                        fontsize=8,
                    )
                if row >= len(images):
                    continue
                axis.imshow(images[row], cmap=display_cmap, vmin=0.0, vmax=1.0)

        if title:
            figure.suptitle(title)
        if phase_colorbar:
            scalar = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_obj)
            scalar.set_array([])
            colorbar = figure.colorbar(
                scalar,
                ax=axes.ravel().tolist(),
                orientation="horizontal",
                shrink=0.88,
                pad=0.04,
                ticks=(0.0, 0.5, 1.0),
            )
            colorbar.ax.set_xticklabels(("0", "π", "2π"))
            colorbar.set_label("phase", fontsize=8)
        figure.savefig(path, dpi=100)
        plt.close(figure)
        return path

    def _tensor_to_images(self, value: Any, max_items: int) -> list[Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to visualize tensor outputs.") from exc

        if not isinstance(value, torch.Tensor):
            return []

        tensor = value.detach().float().cpu()
        if tensor.ndim == 4:
            items = [tensor[index] for index in range(min(max_items, tensor.shape[0]))]
        elif tensor.ndim == 3:
            items = [tensor]
        elif tensor.ndim == 2:
            items = [tensor.unsqueeze(0)]
        else:
            return []

        images = []
        for item in items:
            if item.ndim == 3 and item.shape[0] > 1:
                image = item.mean(dim=0)
            else:
                image = item.squeeze(0)
            images.append(image.clamp(0.0, 1.0).numpy())
        return images
