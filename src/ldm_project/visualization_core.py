from __future__ import annotations

import math
from pathlib import Path

import torch

DEFAULT_AMPLITUDE_DISPLAY_QUANTILE = 0.995
VISUALIZATION_AMPLITUDE_KEY = "amplitude"
VISUALIZATION_PHASE_KEY = "phase"


def _normalize_for_visualization(tensor):
    min_value = tensor.amin(dim=(-2, -1), keepdim=True)
    max_value = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - min_value) / (max_value - min_value).clamp_min(1e-6)


def _amplitude_for_visualization(tensor, upper_quantile: float = DEFAULT_AMPLITUDE_DISPLAY_QUANTILE):
    if upper_quantile <= 0.0 or upper_quantile > 1.0:
        raise ValueError(f"amplitude display quantile must be within (0, 1], got {upper_quantile}")
    raw = tensor.detach().float().clamp_min(0.0)
    scale = raw.flatten(start_dim=1).quantile(upper_quantile, dim=1)
    scale = scale.reshape(raw.shape[0], *([1] * (raw.ndim - 1))).clamp_min(1e-6)
    return (raw / scale).clamp(0.0, 1.0)


def _robust_amplitude_display(tensor, lower_q: float = 0.02, upper_q: float = 0.98):
    raw = tensor.detach().float().clamp_min(0.0)
    flat = raw.flatten(start_dim=1)
    lo = flat.quantile(lower_q, dim=1)
    hi = flat.quantile(upper_q, dim=1)
    shape = (raw.shape[0], *([1] * (raw.ndim - 1)))
    lo = lo.reshape(shape)
    hi = hi.reshape(shape)
    return ((raw - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def _intensity_for_visualization(amplitude, upper_quantile: float = DEFAULT_AMPLITUDE_DISPLAY_QUANTILE):
    intensity = amplitude.detach().float().clamp_min(0.0).pow(2)
    return _amplitude_for_visualization(intensity, upper_quantile=upper_quantile)


def _load_original_mnist_digits(labels: torch.Tensor, image_size: int, root: str = "data") -> torch.Tensor:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required to load original MNIST digits.") from exc

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    mnist = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    images = []
    for label in labels.tolist():
        found = None
        for img, lbl in mnist:
            if lbl == int(label):
                found = img
                break
        if found is None:
            found = torch.zeros(1, image_size, image_size)
        images.append(found)
    return torch.stack(images)


def _raw_amplitude_for_visualization(tensor):
    return tensor.detach().float().clamp(0.0, 1.0)


def _phase_for_visualization(tensor):
    return tensor.detach().float().remainder(2.0 * math.pi) / (2.0 * math.pi)



def _phase_tensor_map(source: dict, specs: tuple[tuple[str, str], ...]) -> dict[str, object]:
    return {
        label: _phase_for_visualization(source[key])
        for key, label in specs
        if key in source
    }



def _blank_tensor_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.full_like(reference.detach().float(), float("nan"))



def _code_for_visualization(codes):
    code = codes.detach().float().unsqueeze(1).unsqueeze(2)
    return _normalize_for_visualization(code)



def build_code_visualization_tensor(codes):
    return _code_for_visualization(codes)



def enhance_camera_for_display(tensor, lower_quantile: float = 0.01, upper_quantile: float = 0.995, gamma: float = 0.75):
    if lower_quantile < 0.0 or upper_quantile > 1.0 or lower_quantile >= upper_quantile:
        raise ValueError("display quantiles must satisfy 0 <= lower < upper <= 1")
    if gamma <= 0.0:
        raise ValueError(f"display gamma must be positive, got {gamma}")
    raw = tensor.detach().float()
    flat = raw.flatten(start_dim=-2)
    low = flat.quantile(lower_quantile, dim=-1, keepdim=True).unsqueeze(-1)
    high = flat.quantile(upper_quantile, dim=-1, keepdim=True).unsqueeze(-1)
    scaled = (raw - low) / (high - low).clamp_min(1e-6)
    return scaled.clamp(0.0, 1.0).pow(gamma)



def format_output_title(relative_path: str, label: str) -> str:
    return f"{relative_path} | {label}"



def build_first_stage_visualization_tensors(input_batch, outputs):
    # 物理优先先看相机平面是否承载数字，再用 decoder image 诊断 latent 语义。
    return {
        VISUALIZATION_AMPLITUDE_KEY: {
            "input": input_batch,
            "camera": outputs["camera_plane"],
            "image": outputs["image_recon"],
            "pf2": outputs["pf2_intensity"],
            "chip amp": _amplitude_for_visualization(outputs["chip_amplitude"]),
            "probe": outputs["probe_plane"],
        },
        VISUALIZATION_PHASE_KEY: _phase_tensor_map(
            outputs,
            (
                ("slm_phase", "slm phase"),
                ("chip_phase", "chip phase"),
            ),
        ),
    }



def build_diffusion_visualization_tensors(reference_images, sample_outputs):
    # diffusion 阶段先看生成 latent 经物理后级读出的 camera，再看 decoder image 作为辅助。
    return {
        VISUALIZATION_AMPLITUDE_KEY: {
            "reference": reference_images,
            "camera": sample_outputs["camera_plane"],
            "image": sample_outputs["image_recon"],
            "chip amp": _amplitude_for_visualization(sample_outputs["chip_amplitude"]),
            "probe": sample_outputs["probe_plane"],
        },
        VISUALIZATION_PHASE_KEY: _phase_tensor_map(
            sample_outputs,
            (
                ("slm_phase", "slm phase"),
                ("chip_phase", "chip phase"),
            ),
        ),
    }



def _tensor_stats(tensor):
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
    }



def _tensor_diff_summary(left, right):
    delta = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    return {
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
    }



def flatten_visualization_splits(visualization_splits: dict[str, dict[str, object]]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for prefix, tensor_map in visualization_splits.items():
        for key, value in tensor_map.items():
            flat[f"{prefix}.{key}"] = value
    return flat



def labels_for_visual_rows(labels) -> list[str]:
    return [str(int(label)) for label in labels.detach().cpu().tolist()]



def save_visualization_grids(
    run_manager,
    visualization_splits: dict[str, dict[str, object]],
    destination: str | Path,
    *,
    title_label: str | None = None,
    max_items: int = 4,
    row_labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    base_path = Path(destination)
    saved_paths: dict[str, Path] = {}
    for split_name in (VISUALIZATION_AMPLITUDE_KEY, VISUALIZATION_PHASE_KEY):
        tensor_map = visualization_splits.get(split_name, {})
        if not tensor_map:
            continue
        path = base_path.with_name(f"{base_path.stem}_{split_name}{base_path.suffix}")
        title = ""
        if title_label:
            try:
                relative = str(path.relative_to(run_manager.run_dir))
            except ValueError:
                relative = str(path)
            title = format_output_title(relative, f"{title_label} {split_name}")
        saved_paths[split_name] = run_manager.save_tensor_grid(
            tensor_map,
            path,
            title=title,
            max_items=max_items,
            row_labels=row_labels,
            cmap="twilight" if split_name == VISUALIZATION_PHASE_KEY else "gray",
            phase_colorbar=split_name == VISUALIZATION_PHASE_KEY,
        )
    return saved_paths



def build_denoising_trace_visualization_tensors(reference_images, camera_trace: dict[str, "torch.Tensor"]):
    # 每列展示同一条采样轨迹在指定去噪阶段的相机读出，方便直接看噪声如何变成数字。
    return {
        "reference": reference_images,
        **{label: tensor for label, tensor in camera_trace.items()},
    }
