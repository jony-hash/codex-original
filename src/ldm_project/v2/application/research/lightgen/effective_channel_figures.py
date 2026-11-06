from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def create_figures(output_root: Path, endpoint_data: dict[str, dict[str, Any]], cumulative_rows: list[dict[str, Any]], local_spectra: dict[str, Any], camera: dict[str, Any]) -> list[str]:
    channels = ("raw_zero", "equal_ols_power")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(endpoint_data)
    figure_paths: list[str] = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, names, strict=False):
        power = endpoint_data[name]["modal_power_w"].mean(axis=0).reshape(10, 10)
        reference = max(float(power.max()), np.finfo(np.float64).tiny)
        db = 10.0 * np.log10(np.maximum(power / reference, np.finfo(np.float64).tiny))
        im = ax.imshow(db, vmin=-50, vmax=0, cmap="magma")
        ax.set_title(name)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="mean port power relative to brightest (dB)")
    path = output_root / "figures" / "endpoint_mean_power_maps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path.relative_to(output_root).as_posix())

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, names, strict=False):
        rows = [item for item in cumulative_rows if item["endpoint"] == name]
        ax.plot([item["retained_roots"] for item in rows], [item["mean_mse28"] for item in rows], color="#1f77b4")
        native = float(rows[0]["native_mean_mse28"])
        ax.axhline(native, color="#333333", linestyle="--", linewidth=1, label="native")
        ax.set_title(name)
        ax.set_xlabel("strongest roots retained")
        ax.set_ylabel("mean 28x28 MSE")
        ax.grid(alpha=0.2)
    path = output_root / "figures" / "cumulative_root_retention_mse.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path.relative_to(output_root).as_posix())

    exposures = [float(value) for value in camera["exposure_sweep_s"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for endpoint, data in endpoint_data.items():
        for control_index, control in enumerate(channels):
            means = data["observable_port_counts"][:, control_index, :, 1].mean(axis=0)
            axes[control_index].plot(exposures, means, marker="o", label=endpoint)

    for ax, control in zip(axes, channels, strict=True):
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.set_xlabel("proxy exposure (s)")
        ax.set_ylabel("mean roots with proxy SNR >= 3")
        ax.set_title(control)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    path = output_root / "figures" / "proxy_noise_observable_roots.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path.relative_to(output_root).as_posix())

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, names, strict=False):
        record = local_spectra[name]
        for exposure in [str(float(value)) for value in exposures]:
            values = np.asarray(record["exposures"][exposure]["singular_values"], dtype=np.float64)
            normalized = values / max(float(values[0]), np.finfo(np.float64).tiny)
            ax.plot(np.arange(1, len(values) + 1), normalized, label=f"{float(exposure):g}s")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("real coefficient direction rank")
        ax.set_ylabel("noise-weighted singular value / max")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    path = output_root / "figures" / "local_noise_direction_spectra.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path.relative_to(output_root).as_posix())
    return figure_paths
