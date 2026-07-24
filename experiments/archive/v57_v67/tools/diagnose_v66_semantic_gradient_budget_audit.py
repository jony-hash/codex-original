#!/usr/bin/env python3
"""Zero-training semantic-gradient budget audit for the formal v56 path.

The audit computes only vector-Jacobian products of the frozen model.  It never
creates a training object, mutates parameters, or turns a diagnostic readout
into a formal inference path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint, load_training_config


ROOT = Path(__file__).resolve().parents[1]
V61_PATH = ROOT / "tools/diagnose_v61_offline_decodability_upper_bound.py"
V62_PATH = ROOT / "tools/diagnose_v62_native_conditional_control_audit.py"
V63_PATH = ROOT / "tools/diagnose_v63_code_conditioning_provenance_audit.py"
V64_PATH = ROOT / "tools/diagnose_v64_semantic_band_carriage_audit.py"
V65_PATH = ROOT / "tools/diagnose_v65_common_mode_readout_locality_audit.py"
CHECKPOINT = ROOT / "outputs/20260715_v60/runs/20260715_010531_532021_train_v60_coherent_relay_full_fov_from_scratch/checkpoints/one_shot_distill_best.pt"
CONFIG = ROOT / "configs/train_v56_coherent_relay_from_scratch.yaml"
REFERENCE = ROOT / "outputs/reports/v58_3_b0_relay_cuda_preflight.json"
V61_REPORT = ROOT / "outputs/reports/v61_offline_decodability_upper_bound.json"
V61_TENSORS = ROOT / "outputs/reports/v61_offline_decodability_upper_bound.pt"
V63_REPORT = ROOT / "outputs/reports/v63_code_conditioning_provenance_audit.json"
V63_TENSORS = ROOT / "outputs/reports/v63_code_conditioning_provenance_audit.pt"
V64_REPORT = ROOT / "outputs/reports/v64_semantic_band_carriage_audit.json"
V64_TENSORS = ROOT / "outputs/reports/v64_semantic_band_carriage_audit.pt"
V65_REPORT = ROOT / "outputs/reports/v65_common_mode_readout_locality_audit.json"
V65_TENSORS = ROOT / "outputs/reports/v65_common_mode_readout_locality_audit.pt"
DATA_ROOT = ROOT / "outputs/20260514_unversioned/runs/20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts"
DATASETS = {name: DATA_ROOT / f"teacher_distill_dataset_readable_{name}.pt" for name in ("train", "val", "test")}
OUTPUT = ROOT / "outputs/reports/v66_semantic_gradient_budget_audit.json"

FOREGROUND_THRESHOLD = 0.05
FOREGROUND_WEIGHT = 3.0
BACKGROUND_WEIGHT = 1.0
SUPPRESSION_RATIO = 0.10
AVAILABLE_RATIO = 0.25
MIN_CANCELLATION = 0.25
MAX_CANCELLATION = 0.10


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V61 = _module("v66_v61", V61_PATH)
V62 = _module("v66_v62", V62_PATH)
V63 = _module("v66_v63", V63_PATH)
V64 = _module("v66_v64", V64_PATH)
V65 = _module("v66_v65", V65_PATH)


def _dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _paths() -> dict[str, Path]:
    return {
        "config": CONFIG,
        "checkpoint": CHECKPOINT,
        "reference": REFERENCE,
        "v61_report": V61_REPORT,
        "v61_tensors": V61_TENSORS,
        "v63_report": V63_REPORT,
        "v63_tensors": V63_TENSORS,
        "v64_report": V64_REPORT,
        "v64_tensors": V64_TENSORS,
        "v65_report": V65_REPORT,
        "v65_tensors": V65_TENSORS,
        **{f"{name}_dataset": value for name, value in DATASETS.items()},
    }


def _invalid(paths: Mapping[str, Path], reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "v66-semantic-gradient-budget-audit",
        "status": "harness-invalid",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {key: str(value) for key, value in paths.items()},
        "reasons": reasons,
    }


def _template_decomposition(
    train_target: torch.Tensor,
    train_labels: torch.Tensor,
    test_target: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if train_target.ndim != 4 or tuple(train_target.shape[-2:]) != (200, 200):
        raise ValueError("train targets must be native [N,1,200,200]")
    if test_target.ndim != 4 or tuple(test_target.shape[-2:]) != (200, 200):
        raise ValueError("test targets must be native [N,1,200,200]")
    if train_labels.shape[0] != train_target.shape[0] or test_labels.shape[0] != test_target.shape[0]:
        raise ValueError("template labels and targets have different counts")
    global_template = train_target.float().mean(0, keepdim=True)
    label_templates = torch.stack(
        [train_target[train_labels == label].float().mean(0) for label in range(10)]
    )
    selected = label_templates[test_labels.long()]
    common = -global_template.expand_as(test_target).float()
    label = -(selected - global_template)
    instance = -(test_target.float() - selected)
    residual_error = common + label + instance + test_target.float()
    # These are target-side directions. Runtime adds the camera prediction to
    # the common term before applying the formal VJP.
    return {
        "global_template": global_template,
        "label_templates": label_templates,
        "selected_label_template": selected,
        "common_target_direction": common,
        "label_target_direction": label,
        "instance_target_direction": instance,
    }, {
        "target_direction_decomposition_max_abs": float(residual_error.abs().max()),
        "train_common_rms": float(global_template.square().mean().sqrt()),
        "train_label_template_count_min": float(min((train_labels == label).sum() for label in range(10))),
    }


def residual_components(
    camera: torch.Tensor,
    target: torch.Tensor,
    global_template: torch.Tensor,
    selected_label_template: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the exact camera-target residual decomposition used by the VJP."""
    common = camera - global_template
    label = -(selected_label_template - global_template)
    instance = -(target - selected_label_template)
    return {
        "common": common,
        "label": label,
        "instance": instance,
        "semantic": label + instance,
    }


def _weighted_dot(prediction: torch.Tensor, residual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    foreground = (target > FOREGROUND_THRESHOLD).to(prediction.dtype)
    background = 1.0 - foreground
    foreground_mean = (prediction * residual * foreground).sum() / foreground.sum().clamp_min(1.0)
    background_mean = (prediction * residual * background).sum() / background.sum().clamp_min(1.0)
    return FOREGROUND_WEIGHT * foreground_mean + BACKGROUND_WEIGHT * background_mean


def _weighted_rms(residual: torch.Tensor, target: torch.Tensor) -> float:
    foreground = (target > FOREGROUND_THRESHOLD).to(residual.dtype)
    background = 1.0 - foreground
    value = FOREGROUND_WEIGHT * (residual.square() * foreground).sum() / foreground.sum().clamp_min(1.0)
    value = value + BACKGROUND_WEIGHT * (residual.square() * background).sum() / background.sum().clamp_min(1.0)
    return float(value.clamp_min(0.0).sqrt().detach().cpu())


def _parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    groups = {"generator_e": [], "encoder": [], "decoder": []}
    for name, parameter in model.named_parameters():
        if name.startswith("generator_e."):
            groups["generator_e"].append(parameter)
        elif name.startswith("encoder."):
            groups["encoder"].append(parameter)
        elif name.startswith("decoder."):
            groups["decoder"].append(parameter)
    if any(not values for values in groups.values()):
        raise ValueError(f"formal parameter groups are incomplete: { {key: len(value) for key, value in groups.items()} }")
    return groups


class _GradientAccumulator:
    def __init__(self, parameter_count: int, class_count: int = 10) -> None:
        self.parameter_count = parameter_count
        self.sum_gradient = torch.zeros(parameter_count, dtype=torch.float32)
        self.class_unit_sum = torch.zeros(class_count, parameter_count, dtype=torch.float32)
        self.class_counts = torch.zeros(class_count, dtype=torch.int64)
        self.sum_norm = 0.0
        self.sum_squared_norm = 0.0
        self.count = 0
        self.camera_rms_sum = 0.0

    def add(self, gradients: list[torch.Tensor | None], label: int, camera_rms: float) -> None:
        flat = torch.cat([
            gradient.detach().float().reshape(-1).cpu()
            for gradient in gradients
            if gradient is not None
        ])
        if flat.numel() != self.parameter_count:
            raise ValueError("gradient parameter count changed during audit")
        norm = float(flat.norm())
        unit = flat / max(norm, 1.0e-12)
        self.sum_gradient.add_(flat)
        self.class_unit_sum[int(label)].add_(unit)
        self.class_counts[int(label)] += 1
        self.sum_norm += norm
        self.sum_squared_norm += float(flat.square().sum())
        self.count += 1
        self.camera_rms_sum += camera_rms

    def finish(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("empty gradient accumulator")
        class_counts = self.class_counts.float()
        within_values = []
        for index in range(10):
            count = float(class_counts[index])
            if count > 1:
                within_values.append((float(self.class_unit_sum[index].square().sum()) - count) / (count * (count - 1.0)))
        cross_values = []
        for left in range(10):
            for right in range(left + 1, 10):
                count = float(class_counts[left] * class_counts[right])
                if count > 0:
                    cross_values.append(float(self.class_unit_sum[left].dot(self.class_unit_sum[right])) / count)
        parameter_rms = float((self.sum_squared_norm / (self.count * self.parameter_count)) ** 0.5)
        return {
            "sample_count": self.count,
            "parameter_count": self.parameter_count,
            "parameter_gradient_rms": parameter_rms,
            "mean_parameter_gradient_rms": parameter_rms,
            "cancellation_ratio": float(self.sum_gradient.norm()) / max(self.sum_norm, 1.0e-12),
            "same_class_mean_cosine": _mean(within_values),
            "cross_class_mean_cosine": _mean(cross_values),
            "camera_space_rms_mean": self.camera_rms_sum / self.count,
        }


def _drive_summary(
    accumulators: Mapping[str, Mapping[str, _GradientAccumulator]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component, groups in accumulators.items():
        result[component] = {}
        for group, accumulator in groups.items():
            item = accumulator.finish()
            item["normalized_drive"] = item["parameter_gradient_rms"] / max(item["camera_space_rms_mean"], 1.0e-12)
            result[component][group] = item
    for group in ("generator_e", "encoder", "decoder"):
        common = result["common"][group]["normalized_drive"]
        for component in ("label", "instance", "semantic"):
            result[component][group]["drive_ratio_to_common"] = result[component][group]["normalized_drive"] / max(common, 1.0e-12)
    return result


def decide_gradient(summary: Mapping[str, Any]) -> dict[str, Any]:
    groups = ("generator_e", "encoder", "decoder")
    suppressed = all(
        float(summary[component][group]["drive_ratio_to_common"]) < SUPPRESSION_RATIO
        for group in groups
        for component in ("label", "instance")
    )
    if suppressed:
        return {"status": "semantic-gradient-suppression-candidate"}
    generator = {component: summary[component]["generator_e"] for component in ("label", "instance", "semantic")}
    semantic_drive = max(
        float(generator[component]["drive_ratio_to_common"])
        for component in ("label", "instance")
    )
    semantic_cancellation = float(generator["semantic"]["cancellation_ratio"])
    semantic_cross_cosine = float(generator["semantic"]["cross_class_mean_cosine"])
    if semantic_drive >= AVAILABLE_RATIO and semantic_cancellation <= MAX_CANCELLATION and semantic_cross_cosine <= 0.0:
        return {"status": "batch-cancellation-candidate"}
    d2nn_has_semantic = any(
        all(float(summary[component][group]["drive_ratio_to_common"]) >= AVAILABLE_RATIO for component in ("label", "instance"))
        for group in ("encoder", "decoder")
    )
    if (
        all(float(generator[component]["drive_ratio_to_common"]) >= AVAILABLE_RATIO for component in ("label", "instance"))
        and d2nn_has_semantic
        and semantic_cancellation >= MIN_CANCELLATION
    ):
        return {"status": "semantic-gradient-available-candidate"}
    return {"status": "mixed-semantic-gradient-evidence"}


def _validate_inputs(paths: Mapping[str, Path], datasets: Mapping[str, Mapping[str, Any]]) -> list[str]:
    reasons, _, _ = V65._validate_inputs(paths, datasets)
    v65_report = json.loads(paths["v65_report"].read_text(encoding="utf-8"))
    if v65_report.get("decision", {}).get("status") != "nonlocal-codebook-objective-representation-mismatch-confirmed":
        reasons.append("v65 decision is not the required nonlocal-codebook result")
    v65_bundle = torch.load(paths["v65_tensors"], map_location="cpu", weights_only=False)
    if v65_bundle.get("schema_version") != "v65-common-mode-readout-locality-audit":
        reasons.append("v65 tensor schema is invalid")
    if not torch.equal(v65_bundle.get("class_labels"), datasets["test"]["class_labels"]):
        reasons.append("v65 labels differ from readable test dataset")
    if not torch.equal(v65_bundle.get("teacher_target"), datasets["test"]["teacher_camera"]):
        reasons.append("v65 target differs from readable test dataset")
    for split, payload in datasets.items():
        if int(payload["class_labels"].shape[0]) != (800 if split == "train" else 100):
            reasons.append(f"{split} count is not the fixed readable split")
    return reasons


def _preview(path: Path, target: torch.Tensor, raw: torch.Tensor, global_template: torch.Tensor, selected: torch.Tensor, labels: torch.Tensor) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render the v66 preview") from exc

    chosen = [int(torch.nonzero(labels == label, as_tuple=False).flatten()[0]) for label in range(10)]
    residual = target - selected
    scale = float(residual.abs().flatten().quantile(0.995).clamp_min(1.0e-6))
    rows = [
        ("teacher_target", target),
        ("raw_cmos", raw),
        ("global_template", global_template.expand_as(target)),
        ("label_template", selected),
        ("instance_residual", (0.5 + residual / (2.0 * scale)).clamp(0.0, 1.0)),
    ]
    fig, axes = plt.subplots(len(rows), 10, figsize=(16, 2.2 * len(rows)), squeeze=False)
    for row, (name, values) in enumerate(rows):
        for column, index in enumerate(chosen):
            axes[row, column].imshow(values[index, 0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_title(f"{name}\nlabel {int(labels[index])}", fontsize=7)
            axes[row, column].axis("off")
    fig.suptitle("v66 semantic-gradient budget audit — diagnostic only")
    fig.tight_layout()
    preview = path.with_name(f"{path.stem}_preview.png")
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    return preview


def run(*, device: str = "cuda", output: str | Path = OUTPUT) -> dict[str, Any]:
    paths = _paths()
    output = Path(output)
    reasons = [f"missing input: {key}" for key, path in paths.items() if not path.is_file()]
    if device != "cuda":
        reasons.append("v66 requires CUDA")
    if not torch.cuda.is_available():
        reasons.append("CUDA unavailable")
    if importlib.util.find_spec("matplotlib") is None:
        reasons.append("matplotlib is required to render the v66 preview")
    if reasons:
        report = _invalid(paths, reasons)
        _dump(output, report)
        return report
    datasets = {name: V62._load_bundle(path) for name, path in DATASETS.items()}
    reasons = _validate_inputs(paths, datasets)
    if reasons:
        report = _invalid(paths, reasons)
        _dump(output, report)
        return report
    config = load_training_config(CONFIG)
    model = OneShotOpticalGenerator(config.physics, config.one_shot, image_size=config.first_stage.image_size).cuda().eval()
    load_one_shot_checkpoint(model, torch.load(CHECKPOINT, map_location="cuda", weights_only=False))
    before = V62._state_snapshot(model)
    groups = _parameter_groups(model)
    group_counts = {name: sum(parameter.numel() for parameter in values) for name, values in groups.items()}
    params = [parameter for values in groups.values() for parameter in values]
    group_slices: dict[str, slice] = {}
    offset = 0
    for name, values in groups.items():
        group_slices[name] = slice(offset, offset + len(values))
        offset += len(values)
    targets = {name: payload["teacher_camera"].float().cpu() for name, payload in datasets.items()}
    labels = {name: payload["class_labels"].long().cpu() for name, payload in datasets.items()}
    templates, decomposition_meta = _template_decomposition(targets["train"], labels["train"], targets["test"], labels["test"])
    components = ("common", "label", "instance", "semantic")
    accumulators = {
        component: {group: _GradientAccumulator(count) for group, count in group_counts.items()}
        for component in components
    }
    raw_camera = torch.empty_like(targets["test"])
    try:
        for index in range(int(targets["test"].shape[0])):
            code = datasets["test"]["codes"][index:index + 1].cuda()
            class_label = labels["test"][index:index + 1].cuda()
            target = targets["test"][index:index + 1].cuda()
            global_template = templates["global_template"].cuda()
            selected = templates["selected_label_template"][index:index + 1].cuda()
            with torch.enable_grad():
                outputs = model(code, class_label, return_optical_fields=True)
                V62._validate_forward(outputs, config=config)
                camera_loss = outputs["camera_plane_loss"]
                raw_camera[index] = outputs["physical_camera_plane"].detach().cpu()
                residuals = residual_components(camera_loss, target, global_template, selected)
                camera_residual = residuals["common"] + residuals["label"] + residuals["instance"]
                if not torch.allclose(camera_residual, camera_loss - target, rtol=1.0e-5, atol=1.0e-6):
                    raise ValueError("runtime residual decomposition is not exact")
                for component in components:
                    surrogate = _weighted_dot(camera_loss, residuals[component].detach(), target)
                    gradients = torch.autograd.grad(surrogate, params, retain_graph=component != components[-1], allow_unused=True)
                    for group, parameter_values in groups.items():
                        start, stop = group_slices[group].start, group_slices[group].stop
                        camera_rms = _weighted_rms(residuals[component].detach(), target)
                        accumulators[component][group].add(list(gradients[start:stop]), int(labels["test"][index]), camera_rms)
                del outputs
            del code, class_label, target
            torch.cuda.empty_cache()
        changed = V62._state_changes(model, before)
        if changed:
            report = _invalid(paths, [f"model parameters changed: {changed[:5]}"])
            _dump(output, report)
            return report
        summary = _drive_summary(accumulators)
        report: dict[str, Any] = {
            "schema_version": "v66-semantic-gradient-budget-audit",
            "status": "manual-review-required",
            "diagnostic_only": True,
            "accepted_summary_written": False,
            "inputs": {key: str(value) for key, value in paths.items()},
            "invariants": {
                "student_training": False,
                "optimizer_created": False,
                "parameter_update_count": 0,
                "formal_contract_changed": False,
                "native_camera_sampling": "200x200 @ 2.5 um",
                "formal_loss": "unclamped camera_plane_loss; foreground threshold 0.05; weights 3.0/1.0",
                "target_code_or_bridge": "disabled",
                "parameter_gradients_saved": False,
                "full_field_cache_saved": False,
            },
            "split_counts": {name: int(payload["class_labels"].shape[0]) for name, payload in datasets.items()},
            "parameter_group_counts": group_counts,
            "target_decomposition": decomposition_meta,
            "gradient_summary": summary,
        }
        report["decision"] = decide_gradient(summary)
        tensor_path = output.with_suffix(".pt")
        torch.save({
            "schema_version": report["schema_version"],
            "diagnostic_only": True,
            "accepted_summary_written": False,
            "class_labels": labels["test"],
            "teacher_target": targets["test"],
            "raw_camera": raw_camera,
            "global_template": templates["global_template"],
            "selected_label_template": templates["selected_label_template"],
            "gradient_scalar_summary": summary,
        }, tensor_path)
        report["artifacts"] = {
            "tensors": str(tensor_path),
            "preview": str(_preview(output, targets["test"], raw_camera, templates["global_template"], templates["selected_label_template"], labels["test"])),
        }
    except (RuntimeError, ValueError, OSError) as exc:
        report = {
            "schema_version": "v66-semantic-gradient-budget-audit",
            "status": "harness-invalid",
            "diagnostic_only": True,
            "accepted_summary_written": False,
            "inputs": {key: str(value) for key, value in paths.items()},
            "reasons": [str(exc)],
        }
    finally:
        _dump(output, report)
        del model
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({"status": result["status"], "decision": result.get("decision"), "reasons": result.get("reasons", [])}, sort_keys=True))
    raise SystemExit(0 if result["status"] != "harness-invalid" else 1)


if __name__ == "__main__":
    main()
