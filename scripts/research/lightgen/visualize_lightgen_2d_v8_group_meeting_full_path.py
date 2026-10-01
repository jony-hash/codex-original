"""Create the independent V8 group-meeting visualization package.

This is a descriptive, read-only adapter around the already completed V8
runtime.  It loads the immutable V8 authorities, replays exactly two DEV
examples per class through the canonical image path at steps 0 and 300, and
renders the requested eight-node field/semantic evidence package.  It never
trains, backpropagates, changes an optical parameter, writes a checkpoint, or
opens protected/test/sealed data roles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_RELATIVE = Path("outputs/20260912_v8_group_meeting_visualization")
V8_SCRIPT = ROOT / "scripts/research/lightgen/execute_lightgen_2d_absolute_anchor_plus_transfer_warmup_v8.py"
V8_REPORT = ROOT / (
    "outputs/20260911_v8/runs/"
    "lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/"
    "lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_report.v1.json"
)
V8_LINEAGE = V8_REPORT.with_name(
    "lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_lineage.v1.json"
)
V8_CYCLIC_RAW = V8_REPORT.with_name(
    "lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_step300_cyclic_raw_outputs.v1.pt"
)
V8_SUFFICIENT = V8_REPORT.with_name(
    "lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_sufficient_observations.v1.pt"
)

LEGACY_VISUALIZER = ROOT / "scripts/visualization/render_scaled_chen_full_pipeline.py"
AUDIT_ALIGNMENT = Path(
    r"C:\Users\24567\.codex\skills\nature-figure\scripts\audit_panel_alignment.py"
)

CLASS_COUNT = 10
SELECTION_PER_CLASS = 100
DISPLAY_POSITIONS = tuple(
    class_id * SELECTION_PER_CLASS + offset
    for class_id in range(CLASS_COUNT)
    for offset in (0, 1)
)
CYCLIC_POSITIONS = tuple(class_id * SELECTION_PER_CLASS for class_id in range(CLASS_COUNT))
NODE_KEYS = (
    "slm_output",
    "encoder_input",
    "encoder_layer_1",
    "pre_ols",
    "post_ols",
    "decoder_layer_1",
    "decoder_layer_2",
    "cmos_field",
)
NODE_LABELS = (
    "SLM output\n200×200",
    "Front 4f / encoder input\n100×100",
    "Encoder layer 1\n100×100",
    "Encoder layer 2 / Pre-OLS\n100×100",
    "OLS re-emission / W_OLS\n100×100 + z inset",
    "Generator layer 1\n100×100",
    "Generator layer 2\n100×100",
    "CMOS incident field\n200×200",
)
NODE_DISPLAY_SIZE = (100, 100)
PI2 = 2.0 * math.pi


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_facts() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "worktree_dirty_before": bool(status),
        "status_line_count_before": len(status.splitlines()) if status else 0,
        "status_sha256_before": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_lines_before": status.splitlines(),
    }


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} is not finite")


def _tensor_digest(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode("ascii"))
    digest.update(repr(tuple(data.shape)).encode("ascii"))
    digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def _load_checkpoint(v8: Any, root: Path, report: Mapping[str, Any], step: int) -> dict[str, Any]:
    ref = report["checkpoints"][str(step)]
    path = root / str(ref["path"])
    payload = v8.safe_torch_load(
        path, expected_sha256=str(ref["sha256"]), map_location="cpu"
    )
    if not isinstance(payload, dict) or int(payload.get("checkpoint_step", -1)) != step:
        raise RuntimeError(f"V8 checkpoint payload changed at step {step}")
    return payload


def _capture_forward(
    v8: Any,
    consumer: Any,
    images: torch.Tensor,
    positions: Sequence[int],
    *,
    batch_size: int = 2,
) -> dict[str, Any]:
    runtime = consumer.runtime
    relay_capture: dict[str, list[torch.Tensor]] = {"input": [], "output": []}

    # CoherentImagingRelay is intentionally a plain callable rather than an
    # nn.Module, so a temporary delegating callable is the non-mutating hook
    # boundary.  The original operators are restored before returning.
    original_input_relay = runtime.input_relay
    original_output_relay = runtime.output_relay

    class RelayCapture:
        def __init__(self, name: str, relay: Any) -> None:
            self.name = name
            self.relay = relay

        def __call__(self, field: Any) -> Any:
            output = self.relay(field)
            if not hasattr(output, "tensor"):
                raise RuntimeError(f"{self.name} relay did not return a ComplexField")
            relay_capture[self.name].append(output.tensor.detach().cpu().contiguous())
            return output

        def __getattr__(self, name: str) -> Any:
            return getattr(self.relay, name)

    runtime.input_relay = RelayCapture("input", original_input_relay)
    runtime.output_relay = RelayCapture("output", original_output_relay)
    keys = (*NODE_KEYS, "z_ols", "raw_intensity")
    collected: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    before = v8._digest_tree(v8.V6._cpu_tree(runtime.state_dict()))
    try:
        runtime.eval()
        with torch.inference_mode():
            for start in range(0, len(positions), batch_size):
                batch_positions = tuple(positions[start : start + batch_size])
                batch = images[list(batch_positions)].to(consumer.device)
                result = runtime.forward_from_image(batch)
                if result.image_input is None or result.ols_input_field is None:
                    raise RuntimeError("V8 image path did not expose its optical trace")
                batch_fields = {
                    "slm_output": result.image_input.slm_field.tensor,
                    "encoder_input": relay_capture["input"][-1],
                    "encoder_layer_1": result.encoder_records[0].tensor,
                    "pre_ols": result.encoder_records[1].tensor,
                    "post_ols": result.generator_input_field.tensor,
                    "decoder_layer_1": result.generator_records[0].tensor,
                    "decoder_layer_2": result.generator_records[1].tensor,
                    "cmos_field": relay_capture["output"][-1],
                    "z_ols": result.z_ols.coefficients,
                    "raw_intensity": result.raw_intensity,
                }
                if len(result.encoder_records) != 2 or len(result.generator_records) != 2:
                    raise RuntimeError("V8 runtime layer count changed")
                expected_raw = result.raw_camera.tensor
                if not torch.allclose(
                    batch_fields["cmos_field"].abs().square(), expected_raw.detach().cpu(), atol=1.0e-6, rtol=1.0e-6
                ):
                    raise RuntimeError("captured CMOS field does not reproduce canonical raw intensity")
                for key, value in batch_fields.items():
                    if not torch.is_tensor(value):
                        raise RuntimeError(f"captured node is not a tensor: {key}")
                    _finite_tensor(value, key)
                    collected[key].append(value.detach().cpu().contiguous())
    finally:
        runtime.input_relay = original_input_relay
        runtime.output_relay = original_output_relay
    after = v8._digest_tree(v8.V6._cpu_tree(runtime.state_dict()))
    if before != after:
        raise RuntimeError("read-only V8 visualization forward mutated runtime state")
    if any(len(items) == 0 for items in collected.values()):
        raise RuntimeError("V8 visualization capture is incomplete")
    return {key: torch.cat(value, dim=0) for key, value in collected.items()}


def _capture_for_step(
    v8: Any,
    bundle: Mapping[str, Any],
    report: Mapping[str, Any],
    images: torch.Tensor,
    positions: Sequence[int],
    step: int,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    consumer = v8._new_consumer(bundle["v2_contract"], device, v8.PRODUCTION_GRID)
    checkpoint = _load_checkpoint(v8, ROOT, report, step)
    v8._load_runtime_state(consumer, checkpoint["runtime_state"])
    v8._configure_all_frozen(consumer)
    capture = _capture_forward(v8, consumer, images, positions)
    capture["checkpoint"] = {
        "step": step,
        "path": report["checkpoints"][str(step)]["path"],
        "sha256": report["checkpoints"][str(step)]["sha256"],
    }
    return capture


def _downsample_complex(value: torch.Tensor, size: tuple[int, int] = NODE_DISPLAY_SIZE) -> torch.Tensor:
    if value.ndim != 4 or not torch.is_complex(value):
        raise RuntimeError(f"expected complex field [N,1,H,W], got {tuple(value.shape)}")
    real = F.interpolate(value.real.float(), size=size, mode="bilinear", align_corners=False)
    imag = F.interpolate(value.imag.float(), size=size, mode="bilinear", align_corners=False)
    return torch.complex(real, imag)


def _node_views(capture: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    amplitudes: dict[str, np.ndarray] = {}
    phases: dict[str, np.ndarray] = {}
    for key in NODE_KEYS:
        field = capture[key]
        small = _downsample_complex(field)
        amplitudes[key] = small.abs()[:, 0].numpy()
        phases[key] = torch.remainder(torch.angle(small), PI2)[:, 0].numpy()
    return amplitudes, phases


def _node_scales(capture: Mapping[str, Any]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for key in NODE_KEYS:
        values = capture[key].abs().detach().cpu().numpy().reshape(-1)
        scale = float(np.quantile(values, 0.995))
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(f"invalid fixed amplitude scale for {key}: {scale}")
        scales[key] = scale
    return scales


def _z_views(capture: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    z = capture["z_ols"].detach().cpu().reshape(-1, 10, 10)
    _finite_tensor(z, "OLS coefficient matrix")
    scale = float(np.quantile(z.abs().numpy().reshape(-1), 0.995))
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("invalid fixed OLS amplitude scale")
    amplitude = z.abs().numpy()
    phase = torch.remainder(torch.angle(z), PI2).numpy()
    return amplitude, phase, scale


def _base_figure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _load_alignment_helper() -> Any:
    spec = importlib.util.spec_from_file_location("nature_figure_alignment", AUDIT_ALIGNMENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load panel alignment helper: {AUDIT_ALIGNMENT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.require_matplotlib_panel_alignment


def _save_figure(
    fig: plt.Figure,
    base: Path,
    require_panel_labels: bool,
    alignment_helper: Any,
    *,
    exclude_axes: Sequence[Any] = (),
) -> list[str]:
    fig.canvas.draw()
    require_matplotlib_panel_alignment = alignment_helper
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base.with_suffix(".alignment.json")),
        overlay_svg=str(base.with_suffix(".alignment.svg")),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=require_panel_labels,
        strict=True,
        exclude_axes=list(exclude_axes),
    )
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(p.relative_to(ROOT)) for p in (png, pdf)]


def _add_panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _add_ols_inset(axis: Any, z_amp: np.ndarray, index: int, scale: float, *, phase: np.ndarray | None = None) -> Any:
    inset = inset_axes(axis, width="29%", height="29%", loc="lower right", borderpad=0.5)
    if phase is None:
        inset.imshow(np.clip(z_amp[index] / scale, 0.0, 1.0), cmap="magma", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
        inset.set_title("|z|", fontsize=5.5, pad=0.5, color="white")
    else:
        inset.imshow(phase[index], cmap="twilight", vmin=0.0, vmax=PI2, origin="lower", interpolation="nearest")
        inset.set_title("∠z", fontsize=5.5, pad=0.5, color="white")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.4)
    return inset


def _render_path_plate(
    output_dir: Path,
    amplitudes: Mapping[str, np.ndarray],
    phases: Mapping[str, np.ndarray],
    scales: Mapping[str, float],
    z_amp: np.ndarray,
    z_phase: np.ndarray,
    z_scale: float,
    *,
    mode: str,
    alignment_helper: Any,
) -> list[str]:
    rows = len(DISPLAY_POSITIONS)
    fig, axes = plt.subplots(rows, len(NODE_KEYS), figsize=(17.2, 27.0), squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.965, top=0.935, bottom=0.055, wspace=0.045, hspace=0.12)
    last_image = None
    for row, position in enumerate(DISPLAY_POSITIONS):
        class_id = position // SELECTION_PER_CLASS
        ordinal = position % SELECTION_PER_CLASS + 1
        for column, key in enumerate(NODE_KEYS):
            axis = axes[row, column]
            if mode == "amplitude":
                last_image = axis.imshow(
                    np.clip(amplitudes[key][row] / scales[key], 0.0, 1.0),
                    cmap="magma", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest",
                )
                if column == 4:
                    _add_ols_inset(axis, z_amp, row, z_scale)
            else:
                last_image = axis.imshow(
                    phases[key][row], cmap="twilight", vmin=0.0, vmax=PI2,
                    origin="lower", interpolation="nearest",
                )
                if column == 4:
                    _add_ols_inset(axis, z_amp, row, z_scale, phase=z_phase)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(NODE_LABELS[column], fontsize=7.3, pad=5)
            if column == 0:
                axis.set_ylabel(f"y={class_id}\n#{ordinal}", fontsize=6.3, rotation=0, ha="right", va="center", labelpad=11)
            for spine in axis.spines.values():
                spine.set_visible(False)
    if last_image is None:
        raise RuntimeError("path plate did not render")
    if mode == "amplitude":
        scalar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma")
        scalar.set_array([])
        cbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), fraction=0.008, pad=0.006)
        cbar.set_label("amplitude / fixed node-wise 99.5th percentile", fontsize=7)
        title = "V8 step-300 real-path optical trace — shared amplitude scales per node"
        footer = "The fifth column carries a row-major 10×10 OLS coefficient inset; all rows are the registered DEV first-two-per-class panel."
        stem = output_dir / "01_v8_fullpath_amplitude_intensity"
    else:
        scalar = ScalarMappable(norm=Normalize(0.0, PI2), cmap="twilight")
        scalar.set_array([])
        cbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), fraction=0.008, pad=0.006, ticks=[0, math.pi / 2, math.pi, 3 * math.pi / 2, PI2])
        cbar.ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
        cbar.set_label("wrapped phase (rad), fixed [0, 2π)", fontsize=7)
        title = "V8 step-300 real-path optical trace — wrapped complex-field phase"
        footer = "Phase is shown only for complex fields; the CMOS node is the incident complex field before square-law readout, not the detector intensity."
        stem = output_dir / "02_v8_fullpath_phase"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.966)
    fig.text(0.055, 0.027, footer, fontsize=7.3, ha="left")
    return _save_figure(fig, stem, False, alignment_helper, exclude_axes=(cbar.ax,))


def _smooth_logits(evaluator: Any, raw: torch.Tensor) -> torch.Tensor:
    evaluator = evaluator.eval()
    with torch.no_grad():
        _, logits = evaluator(F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28)))
    _finite_tensor(logits, "Smooth logits")
    return logits.detach().cpu()


def _template_distances(raw: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28)).to(torch.float64)
    support = templates.clamp(0.0, 1.0).to(torch.float64)
    return (view[:, None] - support[None]).square().flatten(2).mean(dim=2)


def _rank(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    order = logits.argsort(dim=1, descending=True)
    return (order == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1


def _render_natural_comparison(
    output_dir: Path,
    inputs: torch.Tensor,
    templates: torch.Tensor,
    step0: Mapping[str, Any],
    step300: Mapping[str, Any],
    labels: torch.Tensor,
    evaluator: Any,
    alignment_helper: Any,
) -> tuple[list[str], dict[str, Any]]:
    raw0 = step0["raw_intensity"]
    raw300 = step300["raw_intensity"]
    logits0 = _smooth_logits(evaluator, raw0)
    logits300 = _smooth_logits(evaluator, raw300)
    distances300 = _template_distances(raw300, templates)
    pred0 = logits0.argmax(dim=1)
    pred300 = logits300.argmax(dim=1)
    anchor_pred300 = distances300.argmin(dim=1)
    rank0 = _rank(logits0, labels)
    rank300 = _rank(logits300, labels)
    raw_max = float(torch.quantile(torch.cat((raw0, raw300)).flatten(), 0.995))
    if not math.isfinite(raw_max) or raw_max <= 0.0:
        raise RuntimeError("natural-comparison raw scale is invalid")
    fig, axes = plt.subplots(len(DISPLAY_POSITIONS), 4, figsize=(10.8, 27.0), squeeze=False)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.94, bottom=0.045, wspace=0.07, hspace=0.16)
    headers = ("DEV input", "step 0 raw CMOS", "step 300 raw CMOS", "TRAIN template τᵧ")
    for row, position in enumerate(DISPLAY_POSITIONS):
        label = int(labels[row])
        axes[row, 0].imshow(inputs[row, 0], cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
        axes[row, 1].imshow(raw0[row, 0], cmap="magma", vmin=0.0, vmax=raw_max, origin="lower", interpolation="nearest")
        axes[row, 2].imshow(raw300[row, 0], cmap="magma", vmin=0.0, vmax=raw_max, origin="lower", interpolation="nearest")
        axes[row, 3].imshow(templates[label, 0], cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
        for column, axis in enumerate(axes[row]):
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(headers[column], fontsize=8, pad=5)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(
            f"y={label}\nS0={int(pred0[row])}, r{int(rank0[row])}\nS300={int(pred300[row])}, r{int(rank300[row])}\nA300={int(anchor_pred300[row])}",
            fontsize=6.2,
            rotation=0,
            ha="right",
            va="center",
            labelpad=31,
        )
    fig.suptitle("V8 natural-output comparison on the registered 20-sample DEV panel", fontsize=13, fontweight="bold", y=0.968)
    fig.text(0.115, 0.018, f"Raw-output panels share vmin=0 and vmax={raw_max:.4g}; input/template panels retain their [0,1] image domain. A=template-anchor prediction; rank is the Smooth true-class rank.", fontsize=7.2, ha="left")
    files = _save_figure(fig, output_dir / "03_v8_step0_vs_step300_natural_output", False, alignment_helper)
    return files, {
        "raw_vmax_shared": raw_max,
        "smooth_step0_predictions": pred0.tolist(),
        "smooth_step300_predictions": pred300.tolist(),
        "smooth_step0_true_class_rank": rank0.tolist(),
        "smooth_step300_true_class_rank": rank300.tolist(),
        "template_anchor_step300_predictions": anchor_pred300.tolist(),
    }


def _render_trajectory(output_dir: Path, report: Mapping[str, Any], alignment_helper: Any) -> list[str]:
    steps = [0, 100, 200, 300]
    smooth = [float(report["evaluations"][str(s)]["smooth"]["macro_recall"]) for s in steps]
    anchor = [float(report["evaluations"][str(s)]["template_anchor"]["macro_recall"]) for s in steps]
    image_loss = [float(report["evaluations"][str(s)]["losses"]["L_image"]) for s in steps]
    ols_margin = [float(report["evaluations"][str(s)]["ols_margin"]["mean"]) for s in steps]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.13, wspace=0.30, hspace=0.38)
    specs = [
        (axes[0, 0], smooth, "Smooth macro recall", "recall", "#2f5597"),
        (axes[0, 1], anchor, "Template-anchor macro recall", "recall", "#7f6000"),
        (axes[1, 0], image_loss, "DEV image loss (evaluation-only)", "MSE", "#4f81bd"),
        (axes[1, 1], ols_margin, "OLS margin", "margin", "#548235"),
    ]
    for idx, (axis, values, title, ylabel, color) in enumerate(specs):
        axis.plot(steps, values, marker="o", lw=2.0, color=color)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("V8 checkpoint step")
        axis.set_ylabel(ylabel)
        axis.set_xticks(steps)
        axis.grid(axis="y", color="#d9d9d9", lw=0.6, alpha=0.7)
        _add_panel_label(axis, "abcd"[idx])
    axes[0, 0].axhline(0.937, color="#a61c00", ls="--", lw=1.0)
    axes[0, 0].text(300, 0.937, " Stage-A firewall 0.937", color="#a61c00", fontsize=7, ha="right", va="bottom")
    fig.suptitle("V8 training trajectory: bounded objective and endpoint evidence", fontsize=13, fontweight="bold", y=0.955)
    fig.text(0.10, 0.035, "Smooth Δ=+0.0050 (required +0.0500); anchor recall rises monotonically; image loss improves slightly; OLS margin decreases. The 0.937 line is the unchanged Stage-A-to-B firewall.", fontsize=7.4)
    return _save_figure(fig, output_dir / "04_v8_training_trajectory", True, alignment_helper)


def _render_cyclic_examples(
    output_dir: Path,
    raw_artifact: Mapping[str, Any],
    inputs: torch.Tensor,
    templates: torch.Tensor,
    alignment_helper: Any,
) -> dict[str, Any]:
    labels = raw_artifact["labels"].to(torch.long)
    targets = raw_artifact["targets"].to(torch.long)
    raw_base = raw_artifact["raw_base"].float()
    raw_shift = raw_artifact["raw_shifted"].float()
    positions = list(CYCLIC_POSITIONS)
    # The full registered cyclic artifact is 40 million pixels.  A fixed
    # stride over the immutable tensor is a deterministic global display-scale
    # estimator, not per-sample autoscaling and not a change to any metric.
    raw_flat = torch.cat((raw_base, raw_shift)).flatten()
    raw_max = float(torch.quantile(raw_flat[::64], 0.995))
    if raw_max <= 0.0 or not math.isfinite(raw_max):
        raise RuntimeError("cyclic raw display scale is invalid")
    fig, axes = plt.subplots(10, 5, figsize=(12.0, 22.0), squeeze=False)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.94, bottom=0.045, wspace=0.07, hspace=0.22)
    headers = ("input x", "base raw", "shifted raw", "source τᵧ", "target τₜ")
    template_delta: list[float] = []
    smooth_delta: list[float] = []
    for row, position in enumerate(positions):
        source = int(labels[position])
        target = int(targets[position])
        axes[row, 0].imshow(inputs[position, 0], cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
        axes[row, 1].imshow(raw_base[position, 0], cmap="magma", vmin=0.0, vmax=raw_max, origin="lower", interpolation="nearest")
        axes[row, 2].imshow(raw_shift[position, 0], cmap="magma", vmin=0.0, vmax=raw_max, origin="lower", interpolation="nearest")
        axes[row, 3].imshow(templates[source, 0], cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
        axes[row, 4].imshow(templates[target, 0], cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
        for column, axis in enumerate(axes[row]):
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(headers[column], fontsize=8, pad=5)
            for spine in axis.spines.values():
                spine.set_visible(False)
        tdelta = float(raw_artifact["template_shifted_margin"][position] - raw_artifact["template_base_margin"][position])
        sdelta = float(raw_artifact["smooth_shifted_margin"][position] - raw_artifact["smooth_base_margin"][position])
        template_delta.append(tdelta)
        smooth_delta.append(sdelta)
        axes[row, 0].set_ylabel(f"{source}→{target}\nΔτ={tdelta:+.4f}\nΔSmooth={sdelta:+.3f}", fontsize=6.4, rotation=0, ha="right", va="center", labelpad=29)
    fig.suptitle("V8 registered cyclic transfer examples", fontsize=13, fontweight="bold", y=0.968)
    fig.text(0.11, 0.018, "Controlled latent intervention — not sampled generation. One deterministic first-per-class DEV example is shown for each 0→1,…,9→0 pair; the registered 100-per-class analysis retained positive means for all ten pairs.", fontsize=7.2, ha="left")
    files = _save_figure(fig, output_dir / "05_v8_cyclic_transfer_examples", False, alignment_helper)
    return {"files": files, "raw_vmax_shared": raw_max, "template_delta_examples": template_delta, "smooth_delta_examples": smooth_delta}


def _confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros((CLASS_COUNT, CLASS_COUNT), dtype=torch.int64)
    for true, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[int(true), int(predicted)] += 1
    return matrix


def _render_semantic_analysis(
    output_dir: Path,
    report: Mapping[str, Any],
    sufficient: Mapping[str, Any],
    dev_labels: torch.Tensor,
    alignment_helper: Any,
) -> dict[str, Any]:
    logits0 = sufficient["step0_smooth_logits"].float()
    logits300 = sufficient["step300_smooth_logits"].float()
    if logits0.shape[0] != dev_labels.shape[0] or logits0.shape[0] != 5000:
        raise RuntimeError("full V8 DEV Smooth observation shape changed")
    pred0 = logits0.argmax(dim=1)
    pred300 = logits300.argmax(dim=1)
    conf0 = _confusion_matrix(dev_labels, pred0)
    conf300 = _confusion_matrix(dev_labels, pred300)
    recall0 = conf0.diag().float() / conf0.sum(dim=1).clamp_min(1)
    recall300 = conf300.diag().float() / conf300.sum(dim=1).clamp_min(1)
    delta = recall300 - recall0
    fig = plt.figure(figsize=(12.2, 5.5))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.0), wspace=0.34)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.22, wspace=0.34)
    vmax = int(max(conf0.max(), conf300.max()))
    for axis, matrix, title, panel in zip(axes[:2], (conf0, conf300), ("step 0", "step 300"), ("a", "b"), strict=True):
        axis.imshow(matrix.numpy(), cmap="Blues", vmin=0, vmax=vmax, origin="upper", aspect="auto")
        axis.set_title(f"Smooth confusion — {title}", loc="left", fontweight="bold")
        axis.set_xlabel("predicted class")
        axis.set_ylabel("true class")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
        _add_panel_label(axis, panel)
        for r in range(10):
            for c in range(10):
                value = int(matrix[r, c])
                if value:
                    axis.text(c, r, str(value), ha="center", va="center", fontsize=5.4, color="white" if value > vmax * 0.45 else "#1d2d44")
    bars = axes[2].bar(range(10), delta.numpy(), color=np.where(delta.numpy() >= 0.0, "#548235", "#a61c00"), width=0.75)
    axes[2].axhline(0.0, color="#333333", lw=0.8)
    axes[2].set_title("Per-class recall change", loc="left", fontweight="bold")
    axes[2].set_xlabel("class")
    axes[2].set_ylabel("step 300 − step 0")
    axes[2].set_xticks(range(10))
    axes[2].grid(axis="y", color="#d9d9d9", lw=0.6, alpha=0.7)
    _add_panel_label(axes[2], "c")
    for bar, value in zip(bars, delta.tolist(), strict=True):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + (0.006 if value >= 0 else -0.006), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=6.4)
    fig.suptitle("V8 Smooth semantic analysis on the full 5,000-row DEV role", fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.08, 0.015, f"Macro recall: {float(recall0.mean()):.4f} → {float(recall300.mean()):.4f} (Δ={float(recall300.mean() - recall0.mean()):+.4f}). Rows are DEV only; no protected, official-test or sealed rows are read.", fontsize=7.3)
    files = _save_figure(fig, output_dir / "06_v8_semantic_analysis", True, alignment_helper)
    return {
        "files": files,
        "dev_rows": 5000,
        "confusion_step0": conf0.tolist(),
        "confusion_step300": conf300.tolist(),
        "per_class_recall_step0": recall0.tolist(),
        "per_class_recall_step300": recall300.tolist(),
        "per_class_recall_delta": delta.tolist(),
    }


def _render_evolution(output_dir: Path, alignment_helper: Any) -> list[str]:
    versions = ["V3", "V4", "V5", "V6", "V7", "V8"]
    independent_smooth = [0.1070, 0.1030, 0.1042, 0.1030, 0.1042, 0.1086]
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(versions))
    axis.plot(x, independent_smooth, marker="o", lw=2.3, color="#2f5597")
    axis.scatter([5], [independent_smooth[-1]], s=80, color="#c55a11", zorder=3)
    axis.axhline(0.937, color="#a61c00", ls="--", lw=1.0)
    axis.set_xticks(x, versions)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Smooth macro recall")
    axis.set_xlabel("objective / intervention version")
    axis.grid(axis="y", color="#d9d9d9", lw=0.6, alpha=0.7)
    axis.set_title("Semantic evolution across V1–V8: comparable and historical views separated", loc="left", fontweight="bold")
    _add_panel_label(axis, "a")
    axis.text(0.02, 0.93, "Early historical context (not comparable to the independent Smooth family):\nV1 protected real-path recall = 0.1618\nV2 final Stage-A recall = 0.1360", transform=axis.transAxes, fontsize=8.2, va="top", bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f2f2f2", "edgecolor": "#b7b7b7"})
    axis.text(5.0, 0.155, "Best V1–V8 result under current\nindependent Smooth comparison family", ha="center", va="center", fontsize=8.2, color="#7f3f00", bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff2cc", "edgecolor": "#c55a11"})
    axis.text(0.02, 0.78, "blue: independent Smooth endpoint family", transform=axis.transAxes, fontsize=7.4, color="#2f5597")
    axis.text(0.02, 0.73, "red dashed: unchanged Stage-A firewall", transform=axis.transAxes, fontsize=7.4, color="#a61c00")
    fig.suptitle("V8 is the best endpoint in this bounded comparison family, not a readiness claim", fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.10, 0.025, "V1/V2 are shown as historical context only. The V3–V8 line is an endpoint comparison, not a monotonic capability curve.", fontsize=7.4)
    return _save_figure(fig, output_dir / "07_v1_v8_semantic_evolution", True, alignment_helper)


def _render_summary(
    output_dir: Path,
    inputs: torch.Tensor,
    raw300: torch.Tensor,
    labels: torch.Tensor,
    report: Mapping[str, Any],
    alignment_helper: Any,
) -> list[str]:
    fig = plt.figure(figsize=(16.0, 9.0))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.0), left=0.035, right=0.98, top=0.91, bottom=0.07, wspace=0.08)
    left = fig.add_subplot(grid[0, 0])
    center = fig.add_subplot(grid[0, 1])
    right = fig.add_subplot(grid[0, 2])
    for axis, label in zip((left, center, right), ("a", "b", "c"), strict=True):
        axis.text(0.01, 0.97, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color="#222222")
        axis.set_xticks([])
        axis.set_yticks([])
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.set_title("Eight-node real path", loc="left", fontweight="bold")
    ys = np.linspace(0.92, 0.10, 8)
    for idx, (y, label) in enumerate(zip(ys, NODE_LABELS, strict=True)):
        left.text(0.50, y, f"{idx + 1}  {label.replace(chr(10), ' / ')}", ha="center", va="center", fontsize=8.0, bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eaf2f8" if idx not in (4, 7) else "#fff2cc", "edgecolor": "#5b9bd5" if idx not in (4, 7) else "#c55a11"})
        if idx < 7:
            left.annotate("", xy=(0.50, ys[idx + 1] + 0.045), xytext=(0.50, y - 0.045), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#7f7f7f"})
    center.set_title("Representative V8 natural outputs", loc="left", fontweight="bold")
    center.set_xlim(0, 2)
    center.set_ylim(0, 2)
    center.axis("off")
    reps = (0, 8, 16, 18)
    raw_limit = float(torch.quantile(raw300.flatten(), 0.995))
    tiles = []
    for position in reps:
        tile = raw300[position, 0].detach().cpu().numpy()
        tile = np.clip(tile / max(raw_limit, 1.0e-12), 0.0, 1.0)
        tile = np.repeat(np.repeat(tile[::2, ::2], 1, axis=0), 1, axis=1)
        tiles.append(tile)
    montage = np.block([[tiles[0], tiles[1]], [tiles[2], tiles[3]]])
    center.imshow(montage, cmap="magma", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest", extent=(0, 2, 0, 2), aspect="auto")
    for idx, position in enumerate(reps):
        x = 0.49 if idx % 2 == 0 else 1.49
        y = 1.50 if idx // 2 == 0 else 0.50
        center.text(x, y, f"class {int(labels[position])} · step 300 raw", ha="center", va="center", fontsize=7, color="white", bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.5, "edgecolor": "none"})
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.set_title("What was achieved / what was not", loc="left", fontweight="bold")
    decision = report["decision"]
    smooth0 = float(report["evaluations"]["0"]["smooth"]["macro_recall"])
    smooth300 = float(report["evaluations"]["300"]["smooth"]["macro_recall"])
    anchor0 = float(report["evaluations"]["0"]["template_anchor"]["macro_recall"])
    anchor300 = float(report["evaluations"]["300"]["template_anchor"]["macro_recall"])
    lines = [
        (0.88, "Bounded observation", f"Smooth DEV macro recall {smooth0:.4f} → {smooth300:.4f}\nΔ={smooth300 - smooth0:+.4f}; 95% CI [+0.0028,+0.0074]"),
        (0.66, "Descriptive anchor", f"Template-anchor macro recall {anchor0:.4f} → {anchor300:.4f}\nCyclic transfer retained; all ten pair means positive"),
        (0.42, "Outer result", str(decision["classification"])),
        (0.20, "Boundary", "Stage-A remains closed: 0.1086 < 0.937\nNo Stage B, generation, or formal promotion"),
    ]
    for y, heading, body in lines:
        right.text(0.04, y, heading, fontsize=9.0, fontweight="bold", color="#1f4e79", va="top")
        right.text(0.04, y - 0.045, body, fontsize=8.6, va="top", wrap=True)
    fig.suptitle("V8 group-meeting summary — objective evidence without readiness overclaim", fontsize=17, fontweight="bold", x=0.035, ha="left", y=0.965)
    fig.text(0.035, 0.025, "V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER_NOT_SUPPORTED is a bounded scientific classification; this visualization package is descriptive only.", fontsize=8.4)
    return _save_figure(fig, output_dir / "08_v8_group_meeting_summary_16x9", True, alignment_helper)


def _write_registry(output_dir: Path) -> dict[str, Any]:
    nodes = []
    source_rows = [
        ("result.image_input.slm_field", "complex field", [20, 1, 200, 200], "complex", "amplitude and wrapped phase", True, "legacy slm_output"),
        ("forward hook: runtime.input_relay output", "complex field", [20, 1, 100, 100], "complex", "amplitude and wrapped phase", True, "legacy encoder_input / Front 4f"),
        ("result.encoder_records[0]", "complex field", [20, 1, 100, 100], "complex", "amplitude and wrapped phase", True, "legacy encoder_layer_1"),
        ("result.encoder_records[1] / result.ols_input_field", "complex field", [20, 1, 100, 100], "complex", "amplitude and wrapped phase", True, "legacy pre_ols"),
        ("result.generator_input_field; overlay result.z_ols.coefficients", "spatial complex field plus 100-port state", [20, 1, 100, 100], "complex + complex[20,100]", "field phase plus exact coefficient phase inset", True, "legacy post_ols adapted to W_OLS re-emission"),
        ("result.generator_records[0]", "complex field", [20, 1, 100, 100], "complex", "amplitude and wrapped phase", True, "legacy decoder_layer_1"),
        ("result.generator_records[1]", "complex field", [20, 1, 100, 100], "complex", "amplitude and wrapped phase", True, "legacy decoder_layer_2"),
        ("forward hook: runtime.output_relay output", "camera-incident complex field", [20, 1, 200, 200], "complex", "amplitude and wrapped phase", True, "legacy cmos_field; detector intensity is derived separately"),
    ]
    for index, (key, meaning, shape, dtype, quantity, phase_defined, legacy) in enumerate(source_rows):
        nodes.append({"node_index": index + 1, "key": NODE_KEYS[index], "display_label": NODE_LABELS[index], "physical_meaning": meaning, "source_or_hook": key, "captured_shape": shape, "dtype_domain": dtype, "quantity": quantity, "phase_defined": phase_defined, "legacy_mapping": legacy, "port_order": "row_major j=10*r+c" if index == 4 else None})
    payload = {
        "schema": "v8_group_meeting_node_registry.v1",
        "visualization_role": "descriptive_only",
        "legacy_visualizer": str(LEGACY_VISUALIZER.relative_to(ROOT)),
        "legacy_full_path_stage_keys": ["slm_output", "encoder_input", "encoder_layer_1", "pre_ols", "post_ols", "decoder_layer_1", "decoder_layer_2", "cmos_field"],
        "runtime": "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
        "nodes": nodes,
        "phase_policy": "wrapped actual angle in [0,2π); no phase invented for real intensity; detector intensity is not called a phase field",
    }
    (output_dir / "v8_group_meeting_node_registry.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _write_sample_manifest(output_dir: Path, indices: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    rows = []
    for position in DISPLAY_POSITIONS:
        rows.append({"panel_position": position, "class": int(labels[position]), "class_ordinal": position % SELECTION_PER_CLASS + 1, "global_training_index": int(indices[position]), "selection_order": position, "role": "DEV"})
    payload = {
        "schema": "v8_group_meeting_sample_manifest.v1",
        "selection_authority": {"rows": 1000, "per_class": 100, "seed": 4321, "rule": "registered V8 balanced cyclic selection; first two within each class by frozen selection order"},
        "rows": rows,
        "class_counts": {str(class_id): 2 for class_id in range(CLASS_COUNT)},
        "protected_rows_read": 0,
        "official_test_rows_read": 0,
        "sealed_rows_read": 0,
    }
    (output_dir / "v8_group_meeting_sample_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _write_analysis(
    output_dir: Path,
    report: Mapping[str, Any],
    verification: Mapping[str, Any],
    figure_files: Mapping[str, Sequence[str]],
) -> None:
    decision = report["decision"]
    text = f"""# V8 Group-Meeting Visual Analysis

## 1. Selection and scope

This is an independent `DESCRIPTIVE_VISUALIZATION_ONLY` package for the best bounded V1–V8 endpoint under the current independent Smooth comparison family: `V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER_WARMUP`. It does not define a new gate, retry V8, authorize Stage B, or claim generation readiness.

The main visual panel contains exactly two registered DEV examples per class, classes 0–9, using the first two positions within the frozen V8 balanced selection order. The cyclic-transfer panel uses the first position per class for the fixed 0→1,…,9→0 examples.

## 2. Runtime and architecture

The adapter replays the canonical LightGen-aligned v2 image path at V8 step 0 and step 300 with the exact V8 checkpoints. The optical architecture, 100-port complex OLS, coefficient-space `T=I_100`, shared `W_OLS`, encoder/generator layer counts, relay contracts, and evaluator role are unchanged. The historical eight-node layout is structurally adapted from `scripts/visualization/render_scaled_chen_full_pipeline.py`; its old scientific tensors were not reused.

## 3. Eight-node registry

The exact runtime-to-node mapping, shapes, complex domains, phase policy, and OLS row-major inset are in `v8_group_meeting_node_registry.json`. Node 5 is the spatial `W_OLS` re-emission field and carries the actual `z_ols` 10×10 row-major coefficient inset. Node 8 is the complex camera-incident field; raw CMOS intensity is its square-law readout and is not mislabeled as a phase field.

## 4. Capture and canonical verification

No optical tensor was synthesized for display. The adapter captured relay outputs with non-mutating forward hooks and used the result dataclass for all other nodes. The final raw intensity was checked against the explicit canonical camera readout, and the step-300 capture was compared with the stored V8 canonical `raw_base` and `z_base` observations. Runtime state equality before/after each replay was required. Verification details are recorded in `visualization_metadata.json`.

## 5. Amplitude and intensity

`01_v8_fullpath_amplitude_intensity` uses fixed node-wise 99.5th-percentile amplitude scales computed across all 20 selected examples. It does not perform per-sample autoscaling. The detector-facing raw intensity is shown in the natural-output and cyclic figures with one shared raw display limit within each figure.

## 6. Phase

`02_v8_fullpath_phase` displays the actual wrapped complex angle in `[0,2π)` with a cyclic map and fixed ticks `0, π/2, π, 3π/2, 2π`. The OLS inset uses the actual 100 coefficients reshaped in the frozen row-major `10×10` order. No phase is inferred from a real intensity tensor.

## 7. OLS interface

The OLS identity is preserved: 100 complex coefficients are projected from the encoder output and re-emitted through the fixed coefficient-space identity transport. The inset is a display of the canonical `result.z_ols.coefficients`, not a new port ordering or a learned transport.

## 8. Natural output comparison

`03_v8_step0_vs_step300_natural_output` uses the same 20 samples at both endpoints and includes input, step-0 raw CMOS, step-300 raw CMOS, and the frozen TRAIN template `τ_y`. Labels, Smooth predictions/ranks, and template-anchor predictions are annotations only; they do not turn the visualization into a new acceptance measurement.

## 9. Training trajectory

`04_v8_training_trajectory` reproduces the authoritative V8 checkpoint values at steps 0, 100, 200, and 300: Smooth macro recall, template-anchor macro recall, evaluation-only image loss, and mean OLS margin. It marks the unchanged Stage-A firewall `0.937` and the required Smooth improvement threshold `+0.05` without changing either gate.

## 10. Cyclic transfer

`05_v8_cyclic_transfer_examples` shows one deterministic DEV example for each registered cyclic intervention. The caption explicitly says this is a controlled latent intervention, not sampled generation. The report retained broad cyclic transfer evidence, while the natural Smooth endpoint did not meet the primary gate.

## 11. Full-DEV class-wise analysis

`06_v8_semantic_analysis` uses the stored V8 sufficient observations for all 5,000 DEV rows to show step-0 and step-300 Smooth confusion matrices and per-class recall changes. It reads no protected, official-test, or sealed role.

## 12. Cross-version evolution

`07_v1_v8_semantic_evolution` separates early historical context (V1 protected real-path recall and V2 final Stage-A recall) from the comparable independent Smooth endpoint family V3–V8. V8 is labelled the best endpoint within that bounded family; the plot does not claim a highest value across all historical evaluators or a monotonic capability curve.

## 13. Conclusion

V8 achieved a bounded objective-correction observation: Smooth macro recall moved from `{float(report['evaluations']['0']['smooth']['macro_recall']):.4f}` to `{float(report['evaluations']['300']['smooth']['macro_recall']):.4f}` (paired Δ `{float(decision['primary_gate']['smooth_delta']):+.4f}`, reported 95% CI `[+0.0028,+0.0074]`), the template-anchor diagnostic rose from `{float(report['evaluations']['0']['template_anchor']['macro_recall']):.4f}` to `{float(report['evaluations']['300']['template_anchor']['macro_recall']):.4f}`, and the registered cyclic transfer was retained. It did not achieve the required natural Smooth improvement of `+0.05`, did not reach the `0.937` Stage-A firewall, and therefore did not authorize Stage B, generation, or formal promotion. The scientific classification remains `{decision['classification']}`.

## 14. Three-minute talk track

1. Start at the eight-node strip: the experiment keeps the LightGen-aligned optical path and fixed OLS identity, so the question is about the V8 objective rather than architecture expansion.
2. Walk left-to-right through the amplitude/phase plates. The fields remain complex until the camera readout; the OLS coefficient inset is the actual 100-port row-major state.
3. Show the step-0/step-300 natural panel and say what changed visually, while keeping the labels and raw display scale fixed.
4. Use the trajectory panel to separate the rising template-anchor diagnostic from the much smaller Smooth improvement and point at the unchanged `0.937` firewall.
5. Show the cyclic panel as a controlled intervention, not generation, then use the full-DEV confusion panel to show class-wise concentration.
6. End on the negative boundary: V8 is the best endpoint in this bounded independent comparison family, but the primary natural semantic gate failed; no downstream authorization follows.

## Figure files

"""
    for name, paths in figure_files.items():
        text += f"- `{name}`: {', '.join(paths)}\n"
    text += "\nAll figures have accompanying alignment JSON/SVG outputs. PDF collision and text-size audits are run after rendering; their reports are listed in the visualization metadata.\n"
    (output_dir / "V8_GROUP_MEETING_VISUAL_ANALYSIS.md").write_text(text, encoding="utf-8")


def _audit_pdfs(output_dir: Path) -> dict[str, Any]:
    pdf_text = Path(
        r"C:\Users\24567\.codex\skills\nature-figure\scripts\audit_pdf_text.py"
    )
    collisions = Path(
        r"C:\Users\24567\.codex\skills\nature-figure\scripts\audit_figure_collisions.py"
    )
    reports: dict[str, Any] = {}
    for pdf in sorted(output_dir.glob("*.pdf")):
        text_result = subprocess.run(
            [sys.executable, str(pdf_text), str(pdf), "--min-pt", "5", "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        collision_json = pdf.with_suffix(".collision-audit.json")
        collision_overlay = pdf.with_suffix(".collision-audit.pdf")
        collision_result = subprocess.run(
            [
                sys.executable,
                str(collisions),
                str(pdf),
                "--json-out",
                str(collision_json),
                "--overlay-pdf",
                str(collision_overlay),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        reports[str(pdf.relative_to(ROOT))] = {
            "pdf_text_returncode": text_result.returncode,
            "pdf_text_stdout": text_result.stdout[-4000:],
            "pdf_text_stderr": text_result.stderr[-4000:],
            "collision_returncode": collision_result.returncode,
            "collision_stdout": collision_result.stdout[-4000:],
            "collision_stderr": collision_result.stderr[-4000:],
            "collision_json": str(collision_json.relative_to(ROOT)) if collision_json.exists() else None,
            "collision_overlay": str(collision_overlay.relative_to(ROOT)) if collision_overlay.exists() else None,
        }
    status = "PASS" if reports and all(
        value["pdf_text_returncode"] == 0 and value["collision_returncode"] == 0
        for value in reports.values()
    ) else "FAIL"
    return {
        "status": status,
        "required_commands": [
            "nature-figure validate_figure.py <wrapper> --strict",
            "nature-figure audit_pdf_text.py <figure.pdf> --min-pt 5",
            "nature-figure audit_figure_collisions.py <figure.pdf> --json-out <figure.collision-audit.json>",
        ],
        "reports": reports,
    }


def run(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing visualization directory: {output_dir}")
    if not LEGACY_VISUALIZER.is_file():
        raise RuntimeError(f"legacy visualizer is unavailable: {LEGACY_VISUALIZER}")
    if not V8_REPORT.is_file() or not V8_LINEAGE.is_file():
        raise RuntimeError("V8 scientific report/lineage authority is unavailable")
    git = _git_facts()
    v8 = _load_module(V8_SCRIPT, "lightgen_v8_visualization_authority")
    cuda_runtime = v8.V6._prepare_cuda_runtime_dll_search_path()
    contract, _v2_contract = v8._load_contract(ROOT)
    report = _read_json(V8_REPORT)
    lineage = _read_json(V8_LINEAGE)
    if report.get("result") != "VALID_COMPLETED" or report.get("classification") != "V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER_NOT_SUPPORTED":
        raise RuntimeError("V8 report is not the completed bounded NOT_SUPPORTED authority")
    if lineage.get("result") != "VALID_COMPLETED":
        raise RuntimeError("V8 lineage is not VALID_COMPLETED")
    bundle = v8._resolve_authorities(ROOT, contract)
    images, labels, _train_indices, development_indices, data_roles = v8.V6._load_roles(ROOT)
    selection_images, selection_labels, selection_indices = v8.V6._load_selection(ROOT, images, labels, development_indices)
    if selection_labels.tolist() != [class_id for class_id in range(CLASS_COUNT) for _ in range(SELECTION_PER_CLASS)]:
        raise RuntimeError("registered V8 selection order is not class-balanced in frozen order")
    display_images = selection_images[list(DISPLAY_POSITIONS)].contiguous()
    display_labels = selection_labels[list(DISPLAY_POSITIONS)].contiguous()
    if display_labels.tolist() != [class_id for class_id in range(CLASS_COUNT) for _ in (0, 1)]:
        raise RuntimeError("display panel is not exactly two samples per class")
    raw_artifact = v8.safe_torch_load(V8_CYCLIC_RAW, expected_sha256=report["artifacts"]["cyclic_raw_outputs"]["sha256"], map_location="cpu")
    sufficient = v8.safe_torch_load(V8_SUFFICIENT, expected_sha256=report["artifacts"]["sufficient_observations"]["sha256"], map_location="cpu")
    step0 = _capture_for_step(v8, bundle, report, display_images, tuple(range(len(DISPLAY_POSITIONS))), 0)
    step300 = _capture_for_step(v8, bundle, report, display_images, tuple(range(len(DISPLAY_POSITIONS))), 300)
    canonical_raw = raw_artifact["raw_base"][list(DISPLAY_POSITIONS)].float()
    canonical_z = raw_artifact["z_base"][list(DISPLAY_POSITIONS)]
    raw_diff = (step300["raw_intensity"] - canonical_raw).abs()
    z_diff = (step300["z_ols"] - canonical_z).abs()
    verification = {
        "step300_raw_canonical_max_abs": float(raw_diff.max()),
        "step300_raw_canonical_allclose": bool(torch.allclose(step300["raw_intensity"], canonical_raw, atol=1.0e-5, rtol=1.0e-5)),
        "step300_z_canonical_max_abs": float(z_diff.max()),
        "step300_z_canonical_allclose": bool(torch.allclose(step300["z_ols"], canonical_z, atol=1.0e-5, rtol=1.0e-5)),
        "captured_cmos_square_law_matches_result": True,
        "runtime_mutation_detected": False,
        "forward_count": 2 * len(DISPLAY_POSITIONS),
    }
    if not verification["step300_raw_canonical_allclose"] or not verification["step300_z_canonical_allclose"]:
        raise RuntimeError("captured V8 endpoint disagrees with stored canonical step-300 observations")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_registry(output_dir)
    _write_sample_manifest(output_dir, selection_indices, selection_labels)
    alignment_helper = _load_alignment_helper()

    amplitudes, phases = _node_views(step300)
    scales = _node_scales(step300)
    z_amp, z_phase, z_scale = _z_views(step300)
    figure_files: dict[str, Sequence[str]] = {}
    figure_files["01_v8_fullpath_amplitude_intensity"] = _render_path_plate(
        output_dir, amplitudes, phases, scales, z_amp, z_phase, z_scale,
        mode="amplitude", alignment_helper=alignment_helper,
    )
    figure_files["02_v8_fullpath_phase"] = _render_path_plate(
        output_dir, amplitudes, phases, scales, z_amp, z_phase, z_scale,
        mode="phase", alignment_helper=alignment_helper,
    )
    natural_files, natural_summary = _render_natural_comparison(
        output_dir,
        display_images,
        bundle["templates"],
        step0,
        step300,
        display_labels,
        bundle["smooth"],
        alignment_helper,
    )
    figure_files["03_v8_step0_vs_step300_natural_output"] = natural_files
    figure_files["04_v8_training_trajectory"] = _render_trajectory(output_dir, report, alignment_helper)
    cyclic_summary = _render_cyclic_examples(
        output_dir,
        raw_artifact,
        selection_images,
        bundle["templates"],
        alignment_helper,
    )
    figure_files["05_v8_cyclic_transfer_examples"] = cyclic_summary["files"]
    semantic_summary = _render_semantic_analysis(
        output_dir,
        report,
        sufficient,
        labels.index_select(0, development_indices).contiguous(),
        alignment_helper,
    )
    figure_files["06_v8_semantic_analysis"] = semantic_summary["files"]
    figure_files["07_v1_v8_semantic_evolution"] = _render_evolution(output_dir, alignment_helper)
    figure_files["08_v8_group_meeting_summary_16x9"] = _render_summary(
        output_dir,
        display_images,
        step300["raw_intensity"],
        display_labels,
        report,
        alignment_helper,
    )
    qa = _audit_pdfs(output_dir)
    metadata = {
        "schema": "v8_group_meeting_visualization_metadata.v1",
        "task_classification": "DESCRIPTIVE_VISUALIZATION_ONLY",
        "scientific_authority": {
            "semantic_authority_id": report["classification"],
            "result": report["result"],
            "v8_report": {"path": str(V8_REPORT.relative_to(ROOT)), "sha256": _sha256(V8_REPORT)},
            "v8_lineage": {"path": str(V8_LINEAGE.relative_to(ROOT)), "sha256": _sha256(V8_LINEAGE)},
            "v8_contract": report["contract"],
        },
        "git": git,
        "legacy_visualizer": str(LEGACY_VISUALIZER.relative_to(ROOT)),
        "runtime": {
            "aligned_runtime": report["authorities"]["source"]["aligned_runtime"],
            "optical_architecture": "LightGen-aligned v2; two encoder layers, 100-port complex OLS, two generator layers, fixed T=I_100, shared W_OLS",
            "cuda_runtime": cuda_runtime,
            "checkpoints": {str(step): report["checkpoints"][str(step)] for step in (0, 300)},
            "display_checkpoint": 300,
            "capture_mode": "canonical forward_from_image with non-mutating input/output relay hooks",
        },
        "data_roles": {
            "source_rows_loaded_for_registered_split": int(images.shape[0]),
            "TRAIN_values_used": 0,
            "DEV_visual_rows": len(DISPLAY_POSITIONS),
            "DEV_full_semantic_rows": 5000,
            "protected_rows_read": 0,
            "official_test_rows_read": 0,
            "sealed_rows_read": 0,
        },
        "node_registry": str((output_dir / "v8_group_meeting_node_registry.json").relative_to(ROOT)),
        "sample_manifest": str((output_dir / "v8_group_meeting_sample_manifest.json").relative_to(ROOT)),
        "selection": {
            "registered_rows": 1000,
            "registered_per_class": 100,
            "seed": 4321,
            "registered_indices_sha256": report["selection"]["indices_sha256"],
            "registered_labels_sha256": report["selection"]["labels_sha256"],
            "display_rule": "first two positions within each class in registered V8 selection order",
        },
        "canonical_verification": verification,
        "amplitude_scales_p995": scales,
        "ols_coefficient_amplitude_scale_p995": z_scale,
        "natural_output_summary": natural_summary,
        "semantic_summary": semantic_summary,
        "cyclic_summary": cyclic_summary,
        "figures": {key: list(value) for key, value in figure_files.items()},
        "qa": qa,
        "governance_boundary": {
            "training_executed": False,
            "optimizer_executed": False,
            "backward_executed": False,
            "parameter_update": False,
            "optical_parameter_update": False,
            "checkpoint_written": False,
            "v8_retry": False,
            "v9_or_v10_executed": False,
            "protected_test_sealed_read": False,
            "generation_executed": False,
            "stage_b_authorized": False,
            "formal_promotion": False,
        },
        "backend": {
            "selected": "python",
            "selection_basis": "requested .py adapter and existing Python legacy visualizer",
            "persisted_default": "not_persisted_permission_denied_on_C_users_24567_.config_nature_skills",
        },
    }
    (output_dir / "visualization_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    _write_analysis(output_dir, report, verification, figure_files)
    return {
        "result": "VALID_COMPLETED" if qa["status"] == "PASS" else "INVALID_QA",
        "output_dir": str(output_dir.relative_to(ROOT)),
        "figure_count": len(figure_files),
        "forward_count": verification["forward_count"],
        "canonical_verification": verification,
        "qa_status": qa["status"],
        "classification": report["classification"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT_RELATIVE)
    args = parser.parse_args()
    _base_figure_style()
    result = run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
